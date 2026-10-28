#include "model/model.h"
#include <base/alloc.h>
#include <glog/logging.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include "base/bf16_utils.h"
#include "model/safetensors_reader.h"
#include "op/matmul.h"
#include "../op/kernels/cuda/mha_kernel.cuh"
#include "../op/kernels/cuda/paged_kernels.cuh"  // paged_decode_geometry_ok (partials sizing)
namespace model {
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : is_quant_model_(is_quant_model),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      model_type_(model_type),
      tokenizer_type_(tokenizer_type) {
  // Escape hatch for A/B testing graph capture without rebuilding.
  const char* disable = std::getenv("LLAMA_DISABLE_CUDA_GRAPH");
  if (disable && std::string(disable) == "1") {
    use_cuda_graphs_ = false;
  }
}

void BatchScratch::ensure(int32_t batch, int32_t hidden_dim, int32_t dim, int32_t kv_dim,
                          int32_t ffn_dim, int32_t head_num, int32_t kv_head_num,
                          int32_t head_size, int32_t max_seq_len, int32_t block_table_stride,
                          int32_t block_size, base::DeviceType device, base::DataType dtype,
                          const std::shared_ptr<base::DeviceAllocator>& alloc) {
  const int32_t num_splits =
      device == base::DeviceType::kDeviceCUDA ? kernel::flash_decoding_num_splits(max_seq_len) : 0;
  // dtype can differ between runs (BF16 CUDA model vs. FP32 CPU model);
  // re-allocate when the element type of the cached buffers changed.
  if (this->batch == batch && !hidden.is_empty() && hidden.get_dim(0) == batch &&
      hidden.data_type() == dtype && this->block_table_stride == block_table_stride &&
      this->block_size == block_size && this->num_splits == num_splits) {
    return;
  }
  this->batch = batch;
  this->block_table_stride = block_table_stride;
  this->block_size = block_size;
  this->num_splits = num_splits;

  hidden = tensor::Tensor(dtype, batch, hidden_dim, true, alloc);
  rms_out = tensor::Tensor(dtype, batch, hidden_dim, true, alloc);
  q_batch = tensor::Tensor(dtype, batch, dim, true, alloc);
  mha_out_batch = tensor::Tensor(dtype, batch, dim, true, alloc);
  attn_out = tensor::Tensor(dtype, batch, hidden_dim, true, alloc);
  ffn_norm_out = tensor::Tensor(dtype, batch, hidden_dim, true, alloc);
  w1_out = tensor::Tensor(dtype, batch, ffn_dim, true, alloc);
  w3_out = tensor::Tensor(dtype, batch, ffn_dim, true, alloc);
  w2_out = tensor::Tensor(dtype, batch, hidden_dim, true, alloc);
  key_batch = tensor::Tensor(dtype, batch, kv_dim, true, alloc);
  val_batch = tensor::Tensor(dtype, batch, kv_dim, true, alloc);
  // Fused QKV output [batch, dim + 2*kv_dim] — allocated unconditionally so
  // the fused path never re-allocates (and CUDA graphs stay capture-safe).
  qkv_out = tensor::Tensor(dtype, batch, dim + 2 * kv_dim, true, alloc);
  // Fused gate/up FFN output [batch, 2*ffn_dim], same reasoning.
  w13_out = tensor::Tensor(dtype, batch, 2 * ffn_dim, true, alloc);

  // Host staging for the captured H2D upload nodes (tokens / positions).
  // Page-locked on CUDA so each graph replay's uploads are real DMAs; with
  // pageable staging the driver stages through a bounce buffer and blocks the
  // replaying thread for the duration of every copy.
  auto alloc_host = (device == base::DeviceType::kDeviceCUDA)
                        ? std::shared_ptr<base::DeviceAllocator>(
                              base::PinnedCPUDeviceAllocatorFactory::get_instance())
                        : std::shared_ptr<base::DeviceAllocator>(
                              base::CPUDeviceAllocatorFactory::get_instance());
  input_ids = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, alloc_host);
  positions = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, alloc_host);
  // S4's K residual window is indexed by sequence slot; the slot of each batch
  // row is staged like positions so the upload node is captured once.
  kv_slots = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, alloc_host);
  input_token_num = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true,
                                   base::CPUDeviceAllocatorFactory::get_instance());

  if (device == base::DeviceType::kDeviceCUDA) {
    // Sized for whichever partials layout the step can reach. The paged decode
    // kernel writes fp32 (acc | m | l) rows — 2x the bf16 (o | m | l)
    // footprint — for batch * head_num * paged_decode_num_splits(...) split
    // slots; a continuous-layout step writes the smaller bf16 rows at
    // flash_decoding_num_splits(...) instead. Neither row mix nor layout is
    // known here, so size for the larger one: it satisfies both, and the
    // decode batches here are small enough that the surplus is KBs.
    // Use the kernels' own predicates (not local copies of them) so the two
    // can never disagree.
    const bool fp32_split =
        kernel::paged_decode_geometry_ok(head_size, block_table_stride, block_size);
    const int32_t partial_splits =
        fp32_split ? kernel::paged_decode_num_splits(max_seq_len, head_num, kv_head_num)
                   : num_splits;
    partial_batch = tensor::Tensor(
        dtype, kernel::flash_decoding_partials_elements(batch, head_num, partial_splits, head_size,
                                                        fp32_split),
        true, alloc);
    tokens_cu = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, alloc);
    positions_cu = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, alloc);
    kv_slots_cu = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, alloc);
    block_table_cu = tensor::Tensor(base::DataType::kDataTypeInt32, batch, block_table_stride,
                                    true, alloc);
  }
}

base::Status Model::decode_step(const tensor::Tensor& input_ids,
                                const tensor::Tensor& positions,
                                const tensor::Tensor& block_table,
                                tensor::Tensor& key_cache,
                                tensor::Tensor& value_cache,
                                tensor::Tensor& logits,
                                int32_t table_cols,
                                const tensor::Tensor* key_scale,
                                const tensor::Tensor* value_scale,
                                const tensor::Tensor* key_zero,
                                const tensor::Tensor* value_zero,
                                const tensor::Tensor* key_residual,
                                const tensor::Tensor* kv_slots) {
  const bool use_graph = device_type_ == base::DeviceType::kDeviceCUDA && use_cuda_graphs_;
  if (!use_graph) {
    // Pure decode: every row is a decode row (num_decode_rows == batch), no prefill
    // row map is needed.
    return forward_batch(input_ids, positions, block_table, key_cache, value_cache, logits,
                         true, input_ids.get_dim(0), nullptr, 0, nullptr, table_cols, key_scale,
                         value_scale, key_zero, value_zero, key_residual, kv_slots);
  }

  const int32_t batch = input_ids.get_dim(0);
  // use_graph implies a CUDA device, so under USE_PAGED_ATTENTION this is the
  // paged layout; the resolver centralizes the interpretation for both paths.
  const KVCacheDims dims = resolve_kv_cache_dims(key_cache, block_table, device_type_);
  // S4 stores the int4 K node packed (kv_dim/2 bytes per row); every scratch
  // size below is in LOGICAL dims, so unpack the physical width here.
  const int32_t kv_row = key_cache.get_dim(dims.paged ? 3 : 2);
  const int32_t kv_dim =
      key_cache.data_type() == base::DataType::kDataTypeInt8 ? kv_row * 2 : kv_row;
  const int32_t table_stride = dims.table_stride;
  const int32_t num_slots = dims.paged ? 0 : dims.num_blocks;
  const int32_t max_seq_len = dims.max_seq_len;

  // Host-side bounds validation — the captured kernels assume in-bounds
  // positions/slots (forward_batch's own check only runs for CPU tensors).
  for (int32_t b = 0; b < batch; ++b) {
    if (positions.ptr<int32_t>()[b] >= max_seq_len) {
      LOG(ERROR) << "decode_step: position exceeds KV cache capacity.";
      return base::error::InvalidArgument(
          "Position exceeds KV cache capacity. Increase max_seq_len.");
    }
#ifndef USE_PAGED_ATTENTION
    // Continuous layout: block_table holds slot ids — validate them against
    // the cache's slot count. (Paged mode: block ids are BlockAllocator-
    // issued and validated at allocation time.)
    if (block_table.ptr<int32_t>()[b] >= num_slots) {
      LOG(ERROR) << "decode_step: slot exceeds KV cache capacity.";
      return base::error::InvalidArgument(
          "Slot exceeds KV cache capacity. Increase max_seq_len.");
    }
#endif
  }

  // Pool entry per batch size (LRU-evicted when it grows too large).
  auto& entry = decode_graph_pool_[batch];
  if (!entry) {
    entry = std::make_unique<CudaGraphDecodeEntry>();
    entry->batch = batch;
    entry->scratch = std::make_unique<BatchScratch>();
  }
  entry->last_used = ++decode_graph_clock_;

  constexpr int kMaxDecodeGraphEntries = 16;
  if (static_cast<int>(decode_graph_pool_.size()) > kMaxDecodeGraphEntries) {
    int32_t evict_batch = -1;
    int64_t oldest = std::numeric_limits<int64_t>::max();
    for (auto& [key, e] : decode_graph_pool_) {
      if (e && e->last_used < oldest) {
        oldest = e->last_used;
        evict_batch = key;
      }
    }
    if (evict_batch >= 0 && evict_batch != batch) {
      // The evicted entry's captured graphs bake in the addresses of its
      // scratch — including the pinned host staging its H2D copy nodes read
      // from — and a replay may still be queued on the stream. Drain before
      // the buffers are released. Rare path (only past 16 live batch sizes).
      const auto evict_start = std::chrono::steady_clock::now();
      sync_stream();
      graph_stats_.evict_sync_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - evict_start).count();
      graph_stats_.evictions++;
      decode_graph_pool_.erase(evict_batch);
    }
  }

  // 1. Persistent scratch with the real model dims (allocated OUTSIDE the
  // capture so no cudaMalloc happens while the stream is capturing).
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  const bool is_qwen3 = model_type_ == base::ModelType::kModelTypeQwen3;
  const int32_t m_hidden_dim = is_qwen3 ? config_->hidden_dim_ : config_->dim_;
  const int32_t m_attn_dim = config_->dim_;
  const int32_t m_ffn_dim = is_qwen3 ? config_->immediate_dim_ : config_->hidden_dim_;
  const base::DataType dtype = compute_dtype();
  entry->scratch->ensure(batch, m_hidden_dim, m_attn_dim, kv_dim, m_ffn_dim, config_->head_num_,
                         config_->kv_head_num_, config_->head_size_, max_seq_len, table_stride,
                         dims.block_size, device_type_, dtype, alloc);

  // 2. Stage the new inputs at stable host addresses. forward_batch uploads
  // tokens/positions to the device staging buffers, so those H2D copies become
  // captured "upload nodes" and each graph replay re-reads the fresh host data.
  tensor::Tensor& stage_ids = entry->scratch->input_ids;
  tensor::Tensor& stage_pos = entry->scratch->positions;
  tensor::Tensor& stage_slots = entry->scratch->kv_slots;
  std::memcpy(stage_ids.ptr<int32_t>(), input_ids.ptr<int32_t>(), batch * sizeof(int32_t));
  std::memcpy(stage_pos.ptr<int32_t>(), positions.ptr<int32_t>(), batch * sizeof(int32_t));
  // S4's residual window is indexed by sequence slot. Staged at a stable host
  // address like positions, so the H2D is a captured upload node and a replay
  // reads the fresh slots. bf16 caches pass no slots and skip the copy.
  const bool s4 = kv_slots != nullptr && !kv_slots->is_empty();
  if (s4) {
    std::memcpy(stage_slots.ptr<int32_t>(), kv_slots->ptr<int32_t>(), batch * sizeof(int32_t));
  }

  // Block table: uploaded here, on the stream, BEFORE any capture or replay —
  // not staged into the graph. Its live width is ceil(max_position /
  // block_size) + 1 columns and therefore grows from step to step, while a
  // captured copy would bake in the width of the capture step. The pitch
  // stays the capacity width, so the device tensor's geometry (and every
  // stride/split decision the kernels derive from it) is unchanged; only the
  // bytes moved shrink to the columns this step can reference.
  upload_block_table_cu(cuda_config_->stream, entry->scratch->block_table_cu.ptr<int32_t>(),
                        block_table.ptr<int32_t>(), batch, table_stride, table_cols);

  // Logits carry the scheduler buffer's element type (BF16 for a CUDA BF16
  // model): build the view with the same dtype, aligned with logits.data_type()
  // so the LM-Head GEMM writes straight into the scheduler buffer.
  entry->logits_view =
      tensor::Tensor(logits.data_type(), batch, vocab_size(), false, nullptr,
                     static_cast<void*>(logits.ptr<uint8_t>()));
  entry->logits_view.set_device_type(base::DeviceType::kDeviceCUDA);

  // Diagnostic: run decode steps through the direct (non-captured) path so
  // forward_batch's LLAMA_SECT section timers stay on live streams.
  {
    static const bool no_graph = []() {
      const char* e = std::getenv("LLAMA_NO_GRAPH");
      return e && e[0] == '1';
    }();
    if (no_graph) {
      return forward_batch(stage_ids, stage_pos, block_table, key_cache, value_cache,
                           entry->logits_view, true, batch, nullptr, 0,
                           entry->scratch.get(), table_cols, key_scale, value_scale,
                           key_zero, value_zero, key_residual, s4 ? &stage_slots : nullptr);
    }
  }

  cudaStream_t stream = cuda_config_->stream;
  // Same normalization for the quantization scales: a null pointer and an
  // empty tensor both mean "bf16 pool" (see forward_batch).
  const void* key_scale_ptr =
      (key_scale != nullptr && !key_scale->is_empty()) ? key_scale->get_buffer()->ptr() : nullptr;
  const void* value_scale_ptr = (value_scale != nullptr && !value_scale->is_empty())
                                    ? value_scale->get_buffer()->ptr()
                                    : nullptr;
  // S4 metadata pointers baked into the graph (all null for a bf16 cache).
  const void* key_zero_ptr =
      (key_zero != nullptr && !key_zero->is_empty()) ? key_zero->get_buffer()->ptr() : nullptr;
  const void* value_zero_ptr = (value_zero != nullptr && !value_zero->is_empty())
                                   ? value_zero->get_buffer()->ptr()
                                   : nullptr;
  const void* key_residual_ptr = (key_residual != nullptr && !key_residual->is_empty())
                                     ? key_residual->get_buffer()->ptr()
                                     : nullptr;
  // The slot table lives in this entry's stable device staging, so its
  // address cannot go stale; it is snapshotted for symmetry and to catch a
  // scratch re-allocation that would invalidate the captured upload node.
  const void* kv_slots_ptr = s4 ? entry->scratch->kv_slots_cu.get_buffer()->ptr() : nullptr;
  // Validate the Scheduler-owned pointers baked into the graph at capture
  // time. KV caches and the logits buffer belong to the Scheduler instance;
  // after the Scheduler is destroyed and a new one created (benchmark
  // warm-up/iteration patterns), replaying the old graph would touch freed
  // memory (illegal memory access). Destroy the stale graph so the capture
  // path below re-captures against the current buffers.
  if (entry->exec != nullptr &&
      (entry->captured_key_ptr != key_cache.get_buffer()->ptr() ||
       entry->captured_value_ptr != value_cache.get_buffer()->ptr() ||
       entry->captured_logits_ptr != logits.get_buffer()->ptr() ||
       entry->captured_key_scale_ptr != key_scale_ptr ||
       entry->captured_value_scale_ptr != value_scale_ptr ||
       entry->captured_key_zero_ptr != key_zero_ptr ||
       entry->captured_value_zero_ptr != value_zero_ptr ||
       entry->captured_key_residual_ptr != key_residual_ptr ||
       entry->captured_kv_slots_ptr != kv_slots_ptr ||
       entry->captured_key_slots != num_slots ||
       entry->captured_key_seq_len != max_seq_len ||
       entry->captured_key_blocks != dims.num_blocks ||
       entry->captured_block_size != dims.block_size ||
       entry->captured_table_stride != table_stride)) {
    cudaGraphExecDestroy(entry->exec);
    entry->exec = nullptr;
    // A rebuilt Scheduler is a fresh context — give capture another chance
    // (this also un-sticks an entry whose attempts were exhausted against the
    // old buffers).
    entry->capture_attempts = 0;
    // 让流上空闲后再重新捕获,提高 BeginCapture 成功率(发生在压测边界,
    // 同步开销可忽略)。
    cudaStreamSynchronize(stream);
  }

  if (entry->exec == nullptr && entry->capture_attempts < CudaGraphDecodeEntry::kMaxCaptureAttempts) {
    // 3. First call for this batch size: capture the decode kernels.
    // Timed into graph_stats_ (capture + instantiate only — the graph launch
    // at the end of this block is the step's real decode work, not overhead).
    //
    // The cuBLAS handle is created lazily on first use. When that first use
    // lands inside the capture, cublasCreate itself runs during capture and
    // fails with CUBLAS_STATUS_NOT_INITIALIZED (measured), which invalidates
    // the whole recording — cudaStreamEndCapture returns 901 and the entry
    // degrades to the direct path. Touch the handle here, outside the
    // capture, so every GEMM the recording sees reuses it.
    (void)cuda_config_->cublas_handle();
    const auto capture_start = std::chrono::steady_clock::now();
    cudaError_t cap_err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (cap_err != cudaSuccess) {
      entry->capture_attempts++;
      LOG(WARNING) << "[GRAPH] cudaStreamBeginCapture failed (" << cudaGetErrorString(cap_err)
                   << "); falling back to direct decode for batch=" << batch;
      return forward_batch(stage_ids, stage_pos, block_table, key_cache, value_cache,
                           entry->logits_view, true, batch, nullptr, 0,
                           entry->scratch.get(), table_cols, key_scale, value_scale,
                           key_zero, value_zero, key_residual, s4 ? &stage_slots : nullptr);
    }

    auto status = forward_batch(stage_ids, stage_pos, block_table, key_cache, value_cache,
                           entry->logits_view, true, batch, nullptr, 0,
                           entry->scratch.get(), table_cols, key_scale, value_scale,
                           key_zero, value_zero, key_residual, s4 ? &stage_slots : nullptr);
    cudaGraph_t graph = nullptr;
    cap_err = cudaStreamEndCapture(stream, &graph);
    if (cap_err != cudaSuccess || !status || graph == nullptr) {
      entry->capture_attempts++;
      if (graph) {
        cudaGraphDestroy(graph);
      }
      // Drain the stream (the capture may have been invalidated mid-flight)
      // and re-run directly so this step still produces logits.
      cudaStreamSynchronize(stream);
      cudaGetLastError();
      LOG(WARNING) << "[GRAPH] Capture failed for batch=" << batch
                   << "; falling back to direct decode.";
      return forward_batch(stage_ids, stage_pos, block_table, key_cache, value_cache,
                           entry->logits_view, true, batch, nullptr, 0,
                           entry->scratch.get(), table_cols, key_scale, value_scale,
                           key_zero, value_zero, key_residual, s4 ? &stage_slots : nullptr);
    }

    cap_err = cudaGraphInstantiate(&entry->exec, graph, 0);
    cudaGraphDestroy(graph);
    if (cap_err != cudaSuccess) {
      entry->capture_attempts++;
      LOG(WARNING) << "[GRAPH] cudaGraphInstantiate failed (" << cudaGetErrorString(cap_err)
                   << "); falling back to direct decode for batch=" << batch;
      return forward_batch(stage_ids, stage_pos, block_table, key_cache, value_cache,
                           entry->logits_view, true, batch, nullptr, 0,
                           entry->scratch.get(), table_cols, key_scale, value_scale,
                           key_zero, value_zero, key_residual, s4 ? &stage_slots : nullptr);
    }
    graph_stats_.captures++;
    graph_stats_.capture_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - capture_start).count();
    // Snapshot the Scheduler-owned buffers baked into the graph; replay is
    // validated against these on every subsequent decode step.
    entry->captured_key_ptr = key_cache.get_buffer()->ptr();
    entry->captured_value_ptr = value_cache.get_buffer()->ptr();
    entry->captured_logits_ptr = logits.get_buffer()->ptr();
    entry->captured_key_scale_ptr = key_scale_ptr;
    entry->captured_value_scale_ptr = value_scale_ptr;
    entry->captured_key_zero_ptr = key_zero_ptr;
    entry->captured_value_zero_ptr = value_zero_ptr;
    entry->captured_key_residual_ptr = key_residual_ptr;
    entry->captured_kv_slots_ptr = kv_slots_ptr;
    entry->captured_key_slots = num_slots;
    entry->captured_key_seq_len = max_seq_len;
    entry->captured_key_blocks = dims.num_blocks;
    entry->captured_block_size = dims.block_size;
    entry->captured_table_stride = table_stride;
    // Capture only *records* the decode kernels — the stream is in capture
    // mode, so nothing above has actually run. Launch once so this step still
    // writes its KV row and produces logits; otherwise the first decode step
    // of every batch size is silently dropped and the sampler reads stale
    // logits while that position never reaches the cache.
    const cudaError_t first_launch = cudaGraphLaunch(entry->exec, stream);
    if (first_launch != cudaSuccess) {
      LOG(ERROR) << "[GRAPH] cudaGraphLaunch failed after capture: "
                 << cudaGetErrorString(first_launch);
      return base::error::InternalError("CUDA graph launch failed for the decode step.");
    }
    return status;
  }

  if (entry->exec != nullptr) {
    // 4. Replay: launch the captured decode graph (inputs staged in step 2).
    cudaError_t launch_err = cudaGraphLaunch(entry->exec, stream);
    if (launch_err != cudaSuccess) {
      LOG(ERROR) << "[GRAPH] cudaGraphLaunch failed: " << cudaGetErrorString(launch_err);
      return base::error::InternalError("CUDA graph launch failed for the decode step.");
    }
    return base::error::Success();
  }

  // Capture attempts exhausted: direct batched forward with the stable scratch.
  return forward_batch(stage_ids, stage_pos, block_table, key_cache, value_cache,
                       entry->logits_view, true, batch, nullptr, 0, entry->scratch.get(),
                       table_cols, key_scale, value_scale, key_zero, value_zero, key_residual,
                       s4 ? &stage_slots : nullptr);
}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

base::Status Model::load_hf_model() {
  using namespace base;
  if (model_path_.empty()) {
    return error::PathNotValid("Failed to load the model, the model dir is empty!");
  }

  SafetensorsReader reader;
  auto load_status = reader.load(model_path_);
  if (!load_status) {
    return load_status;
  }

  // Store floating-point weights in the model's compute dtype right away:
  // BF16 bits on CUDA models, FP32 on CPU models. The checkpoint dtype is
  // converted only when it differs, so an FP32 (CPU) run keeps the
  // checkpoint's full precision instead of round-tripping F32 -> BF16 -> F32.
  // Packed integer tensors (AWQ qweight/qzeros) keep int32 verbatim — they
  // are bit fields, not numbers to convert.
  const base::DataType dtype = compute_dtype();
  weight_map_.clear();
  for (const std::string& name : reader.tensor_names()) {
    const int64_t n = reader.numel(name);
    if (n <= 0) {
      LOG(WARNING) << "[MODEL] skip empty tensor '" << name << "'";
      continue;
    }
    const std::string& src_dtype = reader.meta(name).dtype;
    const base::DataType dst_dtype =
        (src_dtype == "I32") ? base::DataType::kDataTypeInt32 : dtype;
    auto storage = std::make_shared<std::vector<uint8_t>>(
        static_cast<size_t>(n) * base::DataTypeSize(dst_dtype));
    if (reader.read_tensor(name, dst_dtype, storage->data()) != n) {
      return error::ModelParseError("Failed to read the weight tensor '" + name + "' from " +
                                    model_path_);
    }
    weight_map_.emplace(name, HostWeight{std::move(storage), n, dst_dtype});
  }
  if (weight_map_.empty()) {
    return error::ModelParseError("No weight tensor was loaded from the model dir " + model_path_);
  }
  LOG(INFO) << "[MODEL] loaded " << weight_map_.size() << " weight tensors ("
            << (dtype == base::DataType::kDataTypeBF16 ? "BF16" : "FP32")
            << ") from " << model_path_;
  return error::Success();
}

const void* Model::get_weight_data(const std::string& name) const {
  auto it = weight_map_.find(name);
  CHECK(it != weight_map_.end()) << "Unknown model weight tensor '" << name << "'";
  return it->second.bytes->data();
}

size_t Model::get_weight_numel(const std::string& name) const {
  auto it = weight_map_.find(name);
  CHECK(it != weight_map_.end()) << "Unknown model weight tensor '" << name << "'";
  return static_cast<size_t>(it->second.numel);
}

base::DataType Model::get_weight_dtype(const std::string& name) const {
  auto it = weight_map_.find(name);
  CHECK(it != weight_map_.end()) << "Unknown model weight tensor '" << name << "'";
  return it->second.data_type;
}

namespace {
// AWQ packs eight signed 4-bit codes into one int32; output-column offset i
// selects the nibble at bit 4 * kAwqOrder[i]. This interleaved order was
// verified against the full-precision counterpart of an AWQ checkpoint.
constexpr int kAwqOrder[8] = {0, 4, 1, 5, 2, 6, 3, 7};

template <typename T>
struct AwqStore;
template <>
struct AwqStore<float> {
  static float load_scale(const void* p, int64_t i) {
    return static_cast<const float*>(p)[i];
  }
  static float store(float v) { return v; }
};
template <>
struct AwqStore<base::bf16_t> {
  static float load_scale(const void* p, int64_t i) {
    return base::bf16_to_fp32(static_cast<const base::bf16_t*>(p)[i]);
  }
  static base::bf16_t store(float v) { return base::fp32_to_bf16(v); }
};

// One packed column (eight output rows) is dequantized at a time, so each
// qweight int32 is read once and its eight codes are stored contiguously along
// K in the [N, K] destination rows.
template <typename T>
void awq_dequantize_impl(const int32_t* qweight, const int32_t* qzeros, const void* scales,
                         int32_t in_features, int32_t out_features, int32_t group_size,
                         bool zero_point, T* dst) {
  const int32_t packed_cols = out_features / 8;
  const int32_t groups = in_features / group_size;
  using Store = AwqStore<T>;
#pragma omp parallel for schedule(static)
  for (int32_t c = 0; c < packed_cols; ++c) {
    const int32_t n0 = c * 8;
    for (int32_t g = 0; g < groups; ++g) {
      const uint32_t zero_packed =
          zero_point ? static_cast<uint32_t>(qzeros[static_cast<int64_t>(g) * packed_cols + c])
                     : 0u;
      int32_t zero[8];
      float scale[8];
      for (int i = 0; i < 8; ++i) {
        const int shift = 4 * kAwqOrder[i];
        zero[i] = zero_point ? static_cast<int>((zero_packed >> shift) & 0xFu) : 8;
        scale[i] =
            Store::load_scale(scales, static_cast<int64_t>(g) * out_features + n0 + i);
      }
      for (int32_t kk = 0; kk < group_size; ++kk) {
        const int32_t k = g * group_size + kk;
        const uint32_t packed =
            static_cast<uint32_t>(qweight[static_cast<int64_t>(k) * packed_cols + c]);
        for (int i = 0; i < 8; ++i) {
          const int q = static_cast<int>((packed >> (4 * kAwqOrder[i])) & 0xFu);
          dst[static_cast<int64_t>(n0 + i) * in_features + k] =
              Store::store(static_cast<float>(q - zero[i]) * scale[i]);
        }
      }
    }
  }
}
}  // namespace

void Model::dequantize_awq_module(const std::string& prefix, int32_t out_features,
                                  int32_t in_features, base::DataType dst_dtype,
                                  void* dst) const {
  CHECK(config_ && config_->weight_quant_ == WeightQuantType::kAwqInt4)
      << "AWQ dequantization requested for a non-AWQ model";
  const int32_t group_size = config_->weight_quant_group_size_;
  CHECK_GT(group_size, 0);
  CHECK_EQ(in_features % group_size, 0);
  CHECK_EQ(out_features % 8, 0);
  const bool zero_point = config_->weight_quant_zero_point_;
  const int32_t packed_cols = out_features / 8;
  const int32_t groups = in_features / group_size;
  const std::string qweight_name = prefix + ".qweight";
  CHECK(weight_map_.count(qweight_name) > 0) << "AWQ module '" << prefix << "' misses "
                                             << qweight_name;
  CHECK_EQ(get_weight_numel(qweight_name),
           static_cast<size_t>(in_features) * packed_cols)
      << "AWQ qweight shape mismatch for '" << prefix << "'";
  const void* qweight = get_weight_data(qweight_name);
  const void* qzeros = nullptr;
  if (zero_point) {
    const std::string qzeros_name = prefix + ".qzeros";
    CHECK(weight_map_.count(qzeros_name) > 0)
        << "AWQ module '" << prefix << "' misses " << qzeros_name;
    CHECK_EQ(get_weight_numel(qzeros_name), static_cast<size_t>(groups) * packed_cols)
        << "AWQ qzeros shape mismatch for '" << prefix << "'";
    qzeros = get_weight_data(qzeros_name);
  }
  const std::string scales_name = prefix + ".scales";
  CHECK(weight_map_.count(scales_name) > 0)
      << "AWQ module '" << prefix << "' misses " << scales_name;
  CHECK_EQ(get_weight_numel(scales_name), static_cast<size_t>(groups) * out_features)
      << "AWQ scales shape mismatch for '" << prefix << "'";
  CHECK_EQ(static_cast<int>(get_weight_dtype(scales_name)), static_cast<int>(dst_dtype))
      << "AWQ scales for '" << prefix << "' were not loaded in the compute dtype";
  const void* scales = get_weight_data(scales_name);

  if (dst_dtype == base::DataType::kDataTypeBF16) {
    awq_dequantize_impl<base::bf16_t>(static_cast<const int32_t*>(qweight),
                                      static_cast<const int32_t*>(qzeros), scales, in_features,
                                      out_features, group_size, zero_point,
                                      static_cast<base::bf16_t*>(dst));
  } else if (dst_dtype == base::DataType::kDataTypeFp32) {
    awq_dequantize_impl<float>(static_cast<const int32_t*>(qweight),
                               static_cast<const int32_t*>(qzeros), scales, in_features,
                               out_features, group_size, zero_point, static_cast<float*>(dst));
  } else {
    LOG(FATAL) << "Unsupported AWQ dequantization target dtype";
  }
}

void Model::release_quantization_payload() {
  // The packed AWQ tensors are only input to layer construction (each module is
  // dequantized into its layer's storage). Drop the host copies once every
  // layer exists so an AWQ checkpoint does not keep ~9 GB of qweight/qzeros
  // resident for the model's lifetime. Full-precision modules keep their
  // entries: their layers hold non-owning views into weight_map_.
  if (!config_ || config_->weight_quant_ != WeightQuantType::kAwqInt4) {
    return;
  }
  const auto ends_with = [](const std::string& name, const char* suffix) {
    const size_t n = std::strlen(suffix);
    return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
  };
  for (auto it = weight_map_.begin(); it != weight_map_.end();) {
    const bool packed = ends_with(it->first, ".qweight") ||
                        ends_with(it->first, ".qzeros") ||
                        ends_with(it->first, ".scales");
    it = packed ? weight_map_.erase(it) : std::next(it);
  }
}

std::shared_ptr<op::Layer> Model::create_linear_layer(const std::string& prefix,
                                                      int32_t out_features, int32_t in_features,
                                                      bool has_bias) {
  const auto cpu_device_type = base::DeviceType::kDeviceCPU;
  const bool cuda = device_type_ == base::DeviceType::kDeviceCUDA;
  const base::DataType dtype = compute_dtype();
  std::shared_ptr<base::DeviceAllocator> alloc =
      cuda ? std::shared_ptr<base::DeviceAllocator>(
                 base::CUDADeviceAllocatorFactory::get_instance())
           : std::shared_ptr<base::DeviceAllocator>(
                 base::CPUDeviceAllocatorFactory::get_instance());

  const bool awq_module =
      config_ && config_->weight_quant_ == WeightQuantType::kAwqInt4 &&
      weight_map_.count(prefix + ".qweight") > 0 &&
      std::find(config_->weight_quant_exclude.begin(), config_->weight_quant_exclude.end(),
                prefix) == config_->weight_quant_exclude.end();
  auto layer = std::make_shared<op::MatmulLayer>(device_type_, out_features, in_features,
                                                 /*is_quant_layer=*/false, has_bias);
  if (awq_module) {
    // Dequantize now: every downstream consumer (single projection, fused
    // QKV/FFN, cuBLAS GEMM) then sees the same dense weight the full-precision
    // checkpoint would have produced.
    const size_t bytes = static_cast<size_t>(out_features) * in_features *
                         base::DataTypeSize(dtype);
    std::vector<uint8_t> host_weight(bytes);
    dequantize_awq_module(prefix, out_features, in_features, dtype, host_weight.data());
    tensor::Tensor weight(dtype, out_features, in_features, true, alloc);
    if (cuda) {
      cudaStream_t stream = cuda_config_ ? cuda_config_->stream : nullptr;
      cudaMemcpyAsync(weight.ptr<uint8_t>(), host_weight.data(), bytes,
                      cudaMemcpyHostToDevice, stream);
      if (stream) cudaStreamSynchronize(stream);
    } else {
      std::memcpy(weight.ptr<uint8_t>(), host_weight.data(), bytes);
    }
    CHECK(layer->set_weight(0, weight));
  } else {
    const std::string weight_name = prefix + ".weight";
    CHECK(weight_map_.count(weight_name) > 0)
        << "Checkpoint misses the weight tensor '" << weight_name << "'";
    CHECK(layer->set_weight(0, {out_features, in_features}, get_weight_data(weight_name),
                            cpu_device_type, get_weight_dtype(weight_name)));
  }

  if (has_bias) {
    const std::string bias_name = prefix + ".bias";
    CHECK(weight_map_.count(bias_name) > 0)
        << "Linear layer '" << prefix << "' misses " << bias_name;
    CHECK(layer->set_bias(0, out_features, get_weight_data(bias_name), cpu_device_type,
                          get_weight_dtype(bias_name)));
  }
  return layer;
}

std::shared_ptr<op::MatmulLayer> Model::create_fused_linear_layer(
    const std::vector<std::string>& prefixes, const std::vector<int32_t>& out_features,
    int32_t in_features, bool has_bias) {
  CHECK(!prefixes.empty());
  CHECK_EQ(prefixes.size(), out_features.size());
  int32_t total_out = 0;
  for (int32_t n : out_features) total_out += n;

  const bool cuda = device_type_ == base::DeviceType::kDeviceCUDA;
  const base::DataType dtype = compute_dtype();
  const size_t elem = base::DataTypeSize(dtype);
  std::shared_ptr<base::DeviceAllocator> alloc =
      cuda ? std::shared_ptr<base::DeviceAllocator>(
                 base::CUDADeviceAllocatorFactory::get_instance())
           : std::shared_ptr<base::DeviceAllocator>(
                 base::CPUDeviceAllocatorFactory::get_instance());

  // Rows are concatenated in prefix order; the forward pass already expects
  // that order (q|k|v and gate|up).
  const size_t weight_bytes = static_cast<size_t>(total_out) * in_features * elem;
  std::vector<uint8_t> host_weight(weight_bytes);
  int32_t row_off = 0;
  for (size_t i = 0; i < prefixes.size(); ++i) {
    const std::string& prefix = prefixes[i];
    const int32_t out_i = out_features[i];
    uint8_t* dst = host_weight.data() + static_cast<int64_t>(row_off) * in_features * elem;
    const bool awq_module =
        config_ && config_->weight_quant_ == WeightQuantType::kAwqInt4 &&
        weight_map_.count(prefix + ".qweight") > 0 &&
        std::find(config_->weight_quant_exclude.begin(), config_->weight_quant_exclude.end(),
                  prefix) == config_->weight_quant_exclude.end();
    if (awq_module) {
      dequantize_awq_module(prefix, out_i, in_features, dtype, dst);
    } else {
      const std::string weight_name = prefix + ".weight";
      CHECK(weight_map_.count(weight_name) > 0)
          << "Checkpoint misses the weight tensor '" << weight_name << "'";
      std::memcpy(dst, get_weight_data(weight_name),
                  static_cast<size_t>(out_i) * in_features * elem);
    }
    row_off += out_i;
  }

  auto layer = std::make_shared<op::MatmulLayer>(device_type_, total_out, in_features,
                                                 /*is_quant_layer=*/false, has_bias);
  tensor::Tensor weight(dtype, total_out, in_features, true, alloc);
  cudaStream_t stream = cuda_config_ ? cuda_config_->stream : nullptr;
  if (cuda) {
    cudaMemcpyAsync(weight.ptr<uint8_t>(), host_weight.data(), weight_bytes,
                    cudaMemcpyHostToDevice, stream);
  } else {
    std::memcpy(weight.ptr<uint8_t>(), host_weight.data(), weight_bytes);
  }
  CHECK(layer->set_weight(0, weight));

  if (has_bias) {
    std::vector<uint8_t> host_bias(static_cast<size_t>(total_out) * elem);
    int32_t bias_off = 0;
    for (size_t i = 0; i < prefixes.size(); ++i) {
      const std::string bias_name = prefixes[i] + ".bias";
      CHECK(weight_map_.count(bias_name) > 0)
          << "Linear layer '" << prefixes[i] << "' misses " << bias_name;
      std::memcpy(host_bias.data() + static_cast<int64_t>(bias_off) * elem,
                  get_weight_data(bias_name), static_cast<size_t>(out_features[i]) * elem);
      bias_off += out_features[i];
    }
    tensor::Tensor bias(dtype, total_out, true, alloc);
    if (cuda) {
      cudaMemcpyAsync(bias.ptr<uint8_t>(), host_bias.data(),
                      static_cast<size_t>(total_out) * elem, cudaMemcpyHostToDevice, stream);
    } else {
      std::memcpy(bias.ptr<uint8_t>(), host_bias.data(),
                  static_cast<size_t>(total_out) * elem);
    }
    CHECK(layer->set_bias(0, bias));
  }

  // The host staging buffers are freed after this call, so the H2D copies must
  // have completed (init-time only).
  if (cuda && stream) cudaStreamSynchronize(stream);
  return layer;
}

void Model::repoint_layers_to_fused(
    const std::shared_ptr<op::MatmulLayer>& fused,
    const std::vector<std::shared_ptr<op::Layer>>& layers,
    const std::vector<int32_t>& out_features) {
  CHECK(fused != nullptr);
  CHECK_EQ(layers.size(), out_features.size());
  const int32_t in_features = fused->in_features();
  const tensor::Tensor& weight = fused->get_weight(0);
  const size_t elem = base::DataTypeSize(weight.data_type());

  int32_t out_off = 0;
  for (size_t i = 0; i < layers.size(); ++i) {
    const int32_t out_i = out_features[i];
    auto target = std::dynamic_pointer_cast<op::MatmulLayer>(layers[i]);
    CHECK(target != nullptr) << "A fused projection can only re-point MatmulLayers.";

    tensor::Tensor view(
        weight.data_type(), out_i, in_features, false, nullptr,
        const_cast<uint8_t*>(
            weight.ptr<uint8_t>(static_cast<int64_t>(out_off) * in_features * elem)));
    view.set_device_type(device_type_);
    CHECK(target->set_weight(0, view));

    if (fused->has_bias() && target->has_bias()) {
      const tensor::Tensor& bias = fused->get_bias(0);
      const size_t bias_elem = base::DataTypeSize(bias.data_type());
      tensor::Tensor bias_view(
          bias.data_type(), out_i, false, nullptr,
          const_cast<uint8_t*>(bias.ptr<uint8_t>(static_cast<int64_t>(out_off) * bias_elem)));
      bias_view.set_device_type(device_type_);
      CHECK(target->set_bias(0, bias_view));
    }
    out_off += out_i;
  }
}

base::Status Model::generate_model_infos(const HfConfig& config) const {
  using namespace base;
  config_->layer_num_ = static_cast<int32_t>(config.num_hidden_layers);
  config_->head_num_ = static_cast<int32_t>(config.num_attention_heads);
  config_->kv_head_num_ = static_cast<int32_t>(config.num_key_value_heads);
  config_->seq_len_ = static_cast<int32_t>(config.max_position_embeddings);
  config_->vocab_size_ = std::abs(static_cast<int32_t>(config.vocab_size));

  if (config_.get() == nullptr || config.hidden_size <= 0 || config_->layer_num_ <= 0 ||
      config_->head_num_ <= 0 || config_->kv_head_num_ <= 0 || config_->seq_len_ <= 0 ||
      config_->vocab_size_ <= 0 || config.intermediate_size <= 0) {
    return error::ModelParseError(
        "The model config.json is missing required fields (hidden_size / "
        "intermediate_size / num_hidden_layers / num_attention_heads / "
        "num_key_value_heads / vocab_size / max_position_embeddings).");
  }

  // Head geometry. HF configs of the Qwen3 family carry head_dim explicitly
  // and decouple the attention width from hidden_size (Qwen3-4B: hidden 2560,
  // 32 heads x 128 = 4096); Llama/Qwen2 omit head_dim, so it defaults to
  // hidden_size / heads and dim_ collapses back to hidden_size there.
  const int64_t head_dim = config.head_dim > 0 ? config.head_dim
                                               : config.hidden_size / config.num_attention_heads;
  config_->head_size_ = static_cast<int32_t>(head_dim);
  config_->kv_dim_ = static_cast<int32_t>(config.num_key_value_heads * head_dim);
  config_->kv_mul_ = config_->head_num_ / config_->kv_head_num_;
  // Engine attention width (Q/K/V/O projection rows, query buffers, MHA):
  // head_num x head_dim in every family.
  config_->dim_ = static_cast<int32_t>(config.num_attention_heads * head_dim);

  // Residual vs. FFN width. Qwen3 keeps hidden_dim_ == hidden_size (its
  // residual-stream dim, used by the embedding / norms / FFN IO wiring) and
  // stores the true FFN width in immediate_dim_; Llama/Qwen2 use dim_ ==
  // hidden_dim_ (attention width == residual width) and hidden_dim_ as the
  // FFN width. model.cpp decode_step and the per-family model files share
  // this convention via model_type_.
  const int64_t ffn_dim = config.intermediate_size;
  if (model_type_ == base::ModelType::kModelTypeQwen3) {
    config_->hidden_dim_ = static_cast<int32_t>(config.hidden_size);
    config_->immediate_dim_ = static_cast<int32_t>(ffn_dim);
  } else {
    config_->hidden_dim_ = static_cast<int32_t>(ffn_dim);  // FFN width
    config_->immediate_dim_ = static_cast<int32_t>(ffn_dim);
  }
  config_->is_shared_weight_ = config.tie_word_embeddings;
  config_->rope_theta_ = config.rope_theta;
  return error::Success();
}

base::Status Model::create_encode_layer() {
  using namespace base;

  // create token encode decode layer
  // Use runtime model_type_ instead of compile-time #ifdef to avoid
  // incorrect encode layer when multiple model support flags are enabled.
  if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
  } else if (model_type_ == ModelType::kModelTypeLLama) {
    encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
  } else if (model_type_ == ModelType::kModelTypeQwen2 ||
             model_type_ == ModelType::kModelTypeQwen3) {
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
  } else {
    return error::InternalError("Unsupported model type for tokenizer creation.");
  }
  if (!encode_layer_) {
    return error::InternalError("Create the encode layer failed.");
  }

  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  // 1. Tokenizer (vocab comes from the tokenizer files; the config.json
  // vocab_size governs the embedding / LM-head table rows and can exceed the
  // tokenizer's vocab, in which case sampling is clipped — see
  // post_processing).
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }
  const int32_t tokenizer_vocab_size = encode_layer_->vocab_size();

  // 2. HF config.json -> TransformerConfig.
  const std::string config_path = model_path_ + "/config.json";
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    return error::PathNotValid("Failed to open the model config " + config_path +
                               ", pass the HF model dir (--model-dir) containing "
                               "config.json + *.safetensors");
  }
  HfConfig hf;
  try {
    nlohmann::json j;
    config_file >> j;
    hf.hidden_size = j.at("hidden_size").get<int64_t>();
    hf.intermediate_size = j.at("intermediate_size").get<int64_t>();
    hf.num_hidden_layers = j.at("num_hidden_layers").get<int64_t>();
    hf.num_attention_heads = j.at("num_attention_heads").get<int64_t>();
    hf.num_key_value_heads =
        j.contains("num_key_value_heads") ? j["num_key_value_heads"].get<int64_t>()
                                          : hf.num_attention_heads;
    hf.vocab_size = j.at("vocab_size").get<int64_t>();
    hf.max_position_embeddings = j.at("max_position_embeddings").get<int64_t>();
    if (j.contains("head_dim")) {
      hf.head_dim = j["head_dim"].get<int64_t>();
    }
    if (j.contains("rope_theta")) {
      hf.rope_theta = j["rope_theta"].get<float>();
    }
    hf.tie_word_embeddings = j.value("tie_word_embeddings", false);
    // Quantized checkpoints describe their weight storage in config.json.
    // Reading it here (rather than taking a CLI flag) keeps the runtime
    // checkpoint-driven and works for every model family below.
    if (j.contains("quantization_config") && !j["quantization_config"].is_null()) {
      const auto& q = j["quantization_config"];
      hf.quant_method = q.value("quant_method", std::string());
      hf.quant_version = q.value("version", std::string());
      if (q.contains("bits")) {
        hf.quant_bits = q["bits"].get<int32_t>();
      } else if (q.contains("w_bit")) {
        hf.quant_bits = q["w_bit"].get<int32_t>();
      }
      if (q.contains("group_size")) {
        hf.quant_group_size = q["group_size"].get<int32_t>();
      } else if (q.contains("q_group_size")) {
        hf.quant_group_size = q["q_group_size"].get<int32_t>();
      }
      hf.quant_zero_point = q.value("zero_point", true);
      if (q.contains("modules_to_not_convert") && !q["modules_to_not_convert"].is_null()) {
        for (const auto& module : q["modules_to_not_convert"]) {
          hf.modules_to_not_convert.push_back(module.get<std::string>());
        }
      }
    }
  } catch (const std::exception& e) {
    return error::ModelParseError("Failed to parse the model config " + config_path + ": " +
                                  e.what());
  }
  auto gen_status = generate_model_infos(hf);
  if (!gen_status) {
    LOG(ERROR) << "Invalid model config in " << config_path;
    return gen_status;
  }
  // Resolve the checkpoint's weight-quantization scheme. AWQ is the only
  // quantized format the runtime implements; anything else must fail loudly
  // instead of being silently treated as full precision.
  if (!hf.quant_method.empty()) {
    if (hf.quant_method != "awq") {
      return error::ModelParseError("Unsupported quantization method '" + hf.quant_method +
                                    "' in " + config_path + " (supported: awq)");
    }
    if (hf.quant_bits != 4) {
      return error::ModelParseError("AWQ with bits=" + std::to_string(hf.quant_bits) +
                                    " is unsupported (only 4-bit AWQ is implemented)");
    }
    if (hf.quant_group_size <= 0) {
      return error::ModelParseError("AWQ group_size must be positive, got " +
                                    std::to_string(hf.quant_group_size));
    }
    // Only autoawq's "gemm" layout is implemented: qweight [K, N/8] with the
    // 4-bit codes packed along N. The "gemv" layout stores [N/8, K] and would
    // decode as garbage, so reject it explicitly.
    if (!hf.quant_version.empty() && hf.quant_version != "gemm") {
      return error::ModelParseError("Unsupported AWQ layout version '" + hf.quant_version +
                                    "' in " + config_path + " (supported: gemm)");
    }
    config_->weight_quant_ = WeightQuantType::kAwqInt4;
    config_->weight_quant_group_size_ = hf.quant_group_size;
    config_->weight_quant_zero_point_ = hf.quant_zero_point;
    config_->weight_quant_exclude = hf.modules_to_not_convert;
    is_quant_model_ = true;
    LOG(INFO) << "[MODEL] AWQ W4A16 checkpoint: group_size=" << hf.quant_group_size
              << ", zero_point=" << (hf.quant_zero_point ? "true" : "false");
  } else if (is_quant_model_) {
    return error::ModelParseError(
        "The model was constructed with is_quant_model=true but " + config_path +
        " carries no quantization_config; the legacy int8 format is no longer supported.");
  }
  if (config_->vocab_size_ != tokenizer_vocab_size) {
    LOG(WARNING) << "[MODEL] config.json vocab_size=" << config_->vocab_size_
                 << " differs from tokenizer vocab_size=" << tokenizer_vocab_size
                 << "; logits beyond the tokenizer range are clipped at sampling";
  }

  // 3. Safetensors weights -> weight_map_ (host, already in compute_dtype()).
  auto weight_status = load_hf_model();
  if (!weight_status) {
    LOG(ERROR) << "Load weights from the model dir " << model_path_ << " failed!";
    return weight_status;
  }

  // 4. Build the parameter / non-parameter layers over weight_map_.
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model dir " << model_path_ << " failed!";
    return layer_create_status;
  }

  return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                                int32_t token_pos) const {
  // First touch of the legacy internal cache (predict/forward path): allocate
  // it now, at the recorded cap. Drivers of the batched scheduler path never
  // reach this, so they never pay for the cache.
  ensure_internal_kv_cache();
  // Use the tensor's actual sequence dimension (dim 1) for stride, not config_->seq_len_,
  // so the internal KV cache can be resized independently of the model's max context length.
  int32_t cache_seq_len = get_buffer(ModelBufferType::kKeyCache).get_dim(1);
  int32_t layer_offset = layer_idx * cache_seq_len * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  const base::DataType dtype = compute_dtype();
  const tensor::Tensor& key_cache = get_buffer(ModelBufferType::kKeyCache);
  const tensor::Tensor& val_cache = get_buffer(ModelBufferType::kValueCache);
  // The cache tensors are const refs, so the element pointers are const;
  // drop const, then widen to void* for the external-view constructors.
  void* key_cache_ptr = (dtype == base::DataType::kDataTypeBF16)
                            ? static_cast<void*>(
                                  const_cast<uint16_t*>(key_cache.ptr<uint16_t>(cache_offset)))
                            : static_cast<void*>(
                                  const_cast<float*>(key_cache.ptr<float>(cache_offset)));
  void* val_cache_ptr = (dtype == base::DataType::kDataTypeBF16)
                            ? static_cast<void*>(
                                  const_cast<uint16_t*>(val_cache.ptr<uint16_t>(cache_offset)))
                            : static_cast<void*>(
                                  const_cast<float*>(val_cache.ptr<float>(cache_offset)));

  tensor::Tensor key(dtype, config_->kv_dim_, false, nullptr, key_cache_ptr);
  tensor::Tensor val(dtype, config_->kv_dim_, false, nullptr, val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }

  // Qwen3 uses hidden_dim_ as the model dimension (≠ dim_ for attention),
  // while Qwen2 and Llama use dim_ (== hidden_dim_).
  // Use runtime model_type_ instead of compile-time #ifdef.
  int32_t model_dim = (model_type_ == base::ModelType::kModelTypeQwen3)
                          ? config_->hidden_dim_
                          : config_->dim_;
  const base::DataType dtype = compute_dtype();
  const size_t elem_size = base::DataTypeSize(dtype);

  // The embedding table / activations carry the model's compute dtype; the
  // returned 1-D tensor is a view over the (device-resident) embedding row.
  // Constructed directly over the row pointer (not assign()): the tensor owns
  // its view buffer exclusively, so set_device_type is allowed to retag it.
  tensor::Tensor input(dtype, model_dim, false, nullptr,
                       static_cast<void*>(input_embeddings.ptr<uint8_t>() +
                                          static_cast<size_t>(index) * model_dim * elem_size));
  input.set_device_type(device_type_);
  return input;
}

std::vector<int32_t> Model::post_processing_batch(const tensor::Tensor& logits) const {
  int32_t batch = logits.get_dim(0);
  int32_t vocab_size = logits.get_dim(1);
  std::vector<int32_t> tokens(batch);
  // Restrict sampling to the tokenizer's vocab range: the config.json vocab
  // can be larger than the tokenizer's (Qwen3: 151936 vs 151665), and logits
  // beyond the tokenizer vocab correspond to reserved rows that cannot be
  // decoded. One batched kernel + one device-to-host copy (was: one kernel,
  // one copy and one stream sync per row).
  int32_t sample_vocab = vocab_size;
  if (encode_layer_ && encode_layer_->vocab_size() > 0) {
    sample_vocab = std::min(sample_vocab, encode_layer_->vocab_size());
  }
  // Sample on the model's stream — the same stream the LM head wrote the
  // logits on. The sampler orders itself after the forward and waits for its
  // own D2H event, so callers must NOT drain the stream first (that only adds
  // a host round trip between the forward and the argmax).
  void* stream = cuda_config_ ? cuda_config_->stream : nullptr;
  if (logits.data_type() == base::DataType::kDataTypeBF16) {
    sampler_->sample_batch_bf16(logits.ptr<uint16_t>(), vocab_size, sample_vocab, batch,
                                tokens.data(), stream);
  } else {
    sampler_->sample_batch(logits.ptr<float>(), vocab_size, sample_vocab, batch, tokens.data(),
                           stream);
  }
  return tokens;
}

void Model::ensure_internal_kv_cache() const {
  if (buffers_.find(ModelBufferType::kKeyCache) != buffers_.end()) return;
  CHECK(config_ != nullptr) << "ensure_internal_kv_cache: model not initialized";
  const int32_t len = internal_kv_len_ > 0 ? internal_kv_len_ : config_->seq_len_;
  CHECK_GT(len, 0) << "ensure_internal_kv_cache: model window is unknown (seq_len=0)";

  auto alloc = (device_type_ == base::DeviceType::kDeviceCUDA)
                   ? std::shared_ptr<base::DeviceAllocator>(
                         base::CUDADeviceAllocatorFactory::get_instance())
                   : base::CPUDeviceAllocatorFactory::get_instance();
  const base::DataType dtype = compute_dtype();
  tensor::Tensor key_cache(dtype, config_->layer_num_, len, config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(dtype, config_->layer_num_, len, config_->kv_dim_, true, alloc);
  if (key_cache.is_empty() || value_cache.is_empty()) {
    const double mb = static_cast<double>(config_->layer_num_) * len * config_->kv_dim_ *
                      static_cast<double>(base::DataTypeSize(dtype)) * 2.0 / (1024.0 * 1024.0);
    LOG(FATAL) << "internal KV cache allocation failed: "
               << config_->layer_num_ << " layers x " << len << " tokens x " << config_->kv_dim_
               << " kv_dim (K+V, " << mb << " MB); shrink it with resize_internal_kv_cache()";
  }
  buffers_[ModelBufferType::kKeyCache] = std::move(key_cache);
  buffers_[ModelBufferType::kValueCache] = std::move(value_cache);
  internal_kv_len_ = len;
}

void Model::resize_internal_kv_cache(int32_t max_seq_len) {
  if (max_seq_len <= 0) return;
  // Clamp to the model window: the RoPE sin/cos tables and the CPU score
  // scratch are sized to config_->seq_len_, so a longer cache could never be
  // used over its full length anyway (positions past the window would read
  // out of those tables).
  const int32_t window = config_ ? config_->seq_len_ : max_seq_len;
  if (max_seq_len > window) {
    LOG(WARNING) << "[MODEL] resize_internal_kv_cache(" << max_seq_len
                 << ") exceeds the model window " << window << "; clamping";
    max_seq_len = window;
  }
  internal_kv_len_ = max_seq_len;

  // If the cache does not exist yet, that is all this call does: the internal
  // cache is legacy single-sequence state (see ensure_internal_kv_cache) and is
  // allocated on first use at the recorded cap. Reserving the whole pool here
  // for every driver — as this function used to — is what made the 6 GB cache
  // dead weight for the batched scheduler path.
  auto it_key = buffers_.find(ModelBufferType::kKeyCache);
  auto it_val = buffers_.find(ModelBufferType::kValueCache);
  if (it_key == buffers_.end() || it_val == buffers_.end()) return;

  // Any captured decode graph may reference the old internal buffers
  // (single-sequence prefill path); invalidate the pool so the next decode
  // re-captures against the resized cache. Cheap: resize happens at
  // benchmark setup time, not per decode step.
  if (!decode_graph_pool_.empty()) {
    // A queued replay may still read the old cache or this entry's pinned
    // staging; drain before dropping the buffers. Setup-time path only.
    sync_stream();
    decode_graph_pool_.clear();
  }

  auto alloc = (device_type_ == base::DeviceType::kDeviceCUDA)
      ? std::shared_ptr<base::DeviceAllocator>(base::CUDADeviceAllocatorFactory::get_instance())
      : base::CPUDeviceAllocatorFactory::get_instance();
  const int32_t num_layers = it_key->second.get_dim(0);
  const int32_t kv_dim = it_key->second.get_dim(2);
  const base::DataType dtype = it_key->second.data_type();
  it_key->second = tensor::Tensor(dtype, num_layers, max_seq_len, kv_dim, true, alloc);
  it_val->second = tensor::Tensor(dtype, num_layers, max_seq_len, kv_dim, true, alloc);
  if (it_key->second.is_empty() || it_val->second.is_empty()) {
    LOG(FATAL) << "internal KV cache resize failed: " << num_layers << " layers x " << max_seq_len
               << " tokens x " << kv_dim << " kv_dim (K+V); out of memory?";
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    base::CUDADeviceAllocatorFactory::get_instance()->free_idle();
  }
}

}  // namespace model