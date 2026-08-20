#include "model/model.h"
#include <base/alloc.h>
#include <glog/logging.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include "../op/kernels/cuda/mha_kernel.cuh"
namespace model {
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {
  // Escape hatch for A/B testing graph capture without rebuilding.
  const char* disable = std::getenv("LLAMA_DISABLE_CUDA_GRAPH");
  if (disable && std::string(disable) == "1") {
    use_cuda_graphs_ = false;
  }
}

void BatchScratch::ensure(int32_t batch, int32_t hidden_dim, int32_t dim, int32_t kv_dim,
                          int32_t ffn_dim, int32_t head_num, int32_t head_size,
                          int32_t max_seq_len, int32_t block_table_stride,
                          base::DeviceType device,
                          const std::shared_ptr<base::DeviceAllocator>& alloc) {
  if (this->batch == batch && !hidden.is_empty() && hidden.get_dim(0) == batch &&
      this->block_table_stride == block_table_stride) {
    return;
  }
  this->batch = batch;
  this->block_table_stride = block_table_stride;

  hidden = tensor::Tensor(base::DataType::kDataTypeFp32, batch, hidden_dim, true, alloc);
  rms_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, hidden_dim, true, alloc);
  q_batch = tensor::Tensor(base::DataType::kDataTypeFp32, batch, dim, true, alloc);
  mha_out_batch = tensor::Tensor(base::DataType::kDataTypeFp32, batch, dim, true, alloc);
  attn_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, hidden_dim, true, alloc);
  ffn_norm_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, hidden_dim, true, alloc);
  w1_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, ffn_dim, true, alloc);
  w3_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, ffn_dim, true, alloc);
  w2_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, hidden_dim, true, alloc);
  key_batch = tensor::Tensor(base::DataType::kDataTypeFp32, batch, kv_dim, true, alloc);
  val_batch = tensor::Tensor(base::DataType::kDataTypeFp32, batch, kv_dim, true, alloc);
  // Fused QKV output [batch, dim + 2*kv_dim] — allocated unconditionally so
  // the fused path never re-allocates (and CUDA graphs stay capture-safe).
  qkv_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, dim + 2 * kv_dim, true, alloc);

  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  input_ids = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, alloc_cpu);
  positions = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, alloc_cpu);
  block_table = tensor::Tensor(base::DataType::kDataTypeInt32, batch, block_table_stride,
                               true, alloc_cpu);
  input_token_num = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, alloc_cpu);

  if (device == base::DeviceType::kDeviceCUDA) {
    int32_t num_splits = kernel::flash_decoding_num_splits(max_seq_len);
    partial_batch = tensor::Tensor(base::DataType::kDataTypeFp32,
                                   static_cast<int64_t>(batch) * head_num * num_splits *
                                       (head_size + 2),
                                   true, alloc);
    tokens_cu = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, alloc);
    positions_cu = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, alloc);
    block_table_cu = tensor::Tensor(base::DataType::kDataTypeInt32, batch, block_table_stride,
                                    true, alloc);
  }
}

base::Status Model::decode_step(const tensor::Tensor& input_ids,
                                const tensor::Tensor& positions,
                                const tensor::Tensor& block_table,
                                tensor::Tensor& key_cache,
                                tensor::Tensor& value_cache,
                                tensor::Tensor& logits) {
  const bool use_graph = device_type_ == base::DeviceType::kDeviceCUDA && use_cuda_graphs_;
  if (!use_graph) {
    return forward_batch(input_ids, positions, block_table, key_cache, value_cache, logits, true);
  }

  const int32_t batch = input_ids.get_dim(0);
  // use_graph implies a CUDA device, so under USE_PAGED_ATTENTION this is the
  // paged layout; the resolver centralizes the interpretation for both paths.
  const KVCacheDims dims = resolve_kv_cache_dims(key_cache, block_table, device_type_);
  const int32_t kv_dim = key_cache.get_dim(dims.paged ? 3 : 2);
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
  entry->scratch->ensure(batch, m_hidden_dim, m_attn_dim, kv_dim, m_ffn_dim,
                         config_->head_num_, config_->head_size_, max_seq_len, table_stride,
                         device_type_, alloc);

  // 2. Stage the new inputs at stable host addresses. forward_batch uploads
  // them to the device staging buffers, so the H2D copies become captured
  // "upload nodes" and each graph replay re-reads the fresh host data.
  tensor::Tensor& stage_ids = entry->scratch->input_ids;
  tensor::Tensor& stage_pos = entry->scratch->positions;
  tensor::Tensor& stage_bt = entry->scratch->block_table;
  std::memcpy(stage_ids.ptr<int32_t>(), input_ids.ptr<int32_t>(), batch * sizeof(int32_t));
  std::memcpy(stage_pos.ptr<int32_t>(), positions.ptr<int32_t>(), batch * sizeof(int32_t));
  std::memcpy(stage_bt.ptr<int32_t>(), block_table.ptr<int32_t>(),
              static_cast<size_t>(batch) * table_stride * sizeof(int32_t));

  entry->logits_view =
      tensor::Tensor(base::DataType::kDataTypeFp32, batch, vocab_size(), false, nullptr,
                     logits.ptr<float>());
  entry->logits_view.set_device_type(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream = cuda_config_->stream;

  // Validate the Scheduler-owned pointers baked into the graph at capture
  // time. KV caches and the logits buffer belong to the Scheduler instance;
  // after the Scheduler is destroyed and a new one created (benchmark
  // warm-up/iteration patterns), replaying the old graph would touch freed
  // memory (illegal memory access). Destroy the stale graph so the capture
  // path below re-captures against the current buffers.
  if (entry->exec != nullptr &&
      (entry->captured_key_ptr != key_cache.ptr<float>() ||
       entry->captured_value_ptr != value_cache.ptr<float>() ||
       entry->captured_logits_ptr != logits.ptr<float>() ||
       entry->captured_key_slots != num_slots ||
       entry->captured_key_seq_len != max_seq_len ||
       entry->captured_key_blocks != dims.num_blocks ||
       entry->captured_block_size != dims.block_size ||
       entry->captured_table_stride != table_stride)) {
    cudaGraphExecDestroy(entry->exec);
    entry->exec = nullptr;
    // 让流上空闲后再重新捕获,提高 BeginCapture 成功率(发生在压测边界,
    // 同步开销可忽略)。
    cudaStreamSynchronize(stream);
  }

  if (entry->exec == nullptr && !entry->capture_failed) {
    // 3. First call for this batch size: capture the decode kernels.
    cudaError_t cap_err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (cap_err != cudaSuccess) {
      entry->capture_failed = true;
      LOG(WARNING) << "[GRAPH] cudaStreamBeginCapture failed (" << cudaGetErrorString(cap_err)
                   << "); falling back to direct decode for batch=" << batch;
      return forward_batch(stage_ids, stage_pos, stage_bt, key_cache, value_cache,
                           entry->logits_view, true, entry->scratch.get());
    }

    auto status = forward_batch(stage_ids, stage_pos, stage_bt, key_cache, value_cache,
                                entry->logits_view, true, entry->scratch.get());
    cudaGraph_t graph = nullptr;
    cap_err = cudaStreamEndCapture(stream, &graph);
    if (cap_err != cudaSuccess || !status || graph == nullptr) {
      entry->capture_failed = true;
      if (graph) {
        cudaGraphDestroy(graph);
      }
      // Drain the stream (the capture may have been invalidated mid-flight)
      // and re-run directly so this step still produces logits.
      cudaStreamSynchronize(stream);
      cudaGetLastError();
      LOG(WARNING) << "[GRAPH] Capture failed for batch=" << batch
                   << "; falling back to direct decode.";
      return forward_batch(stage_ids, stage_pos, stage_bt, key_cache, value_cache,
                           entry->logits_view, true, entry->scratch.get());
    }

    cap_err = cudaGraphInstantiate(&entry->exec, graph, 0);
    cudaGraphDestroy(graph);
    if (cap_err != cudaSuccess) {
      entry->capture_failed = true;
      LOG(WARNING) << "[GRAPH] cudaGraphInstantiate failed (" << cudaGetErrorString(cap_err)
                   << "); falling back to direct decode for batch=" << batch;
      return forward_batch(stage_ids, stage_pos, stage_bt, key_cache, value_cache,
                           entry->logits_view, true, entry->scratch.get());
    }
    // Snapshot the Scheduler-owned buffers baked into the graph; replay is
    // validated against these on every subsequent decode step.
    entry->captured_key_ptr = key_cache.ptr<float>();
    entry->captured_value_ptr = value_cache.ptr<float>();
    entry->captured_logits_ptr = logits.ptr<float>();
    entry->captured_key_slots = num_slots;
    entry->captured_key_seq_len = max_seq_len;
    entry->captured_key_blocks = dims.num_blocks;
    entry->captured_block_size = dims.block_size;
    entry->captured_table_stride = table_stride;
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

  // capture_failed forever: direct batched forward with the stable scratch.
  return forward_batch(stage_ids, stage_pos, stage_bt, key_cache, value_cache,
                       entry->logits_view, true, entry->scratch.get());
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

base::Status Model::read_model_file() {
  using namespace base;
  if (model_path_.empty()) {
    return error::PathNotValid("Failed to open the weight file, the model path is empty!");
  }
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }

  auto config = ModelConfig{};
  if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    return error::ModelParseError(
        "Failed to retrieve the configuration information from the model "
        "file.");
  }
  if (is_quant_model_) {
    if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
      return error::ModelParseError(
          "Failed to retrieve the group size information from the model "
          "file.");
    }
  }

  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    return gen_status;
  }

  if (!is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  } else {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return error::ModelParseError(
        "Failed to retrieve the file size information from the model "
        "file.");
  }
  raw_model_data_->file_size = sb.st_size;

  raw_model_data_->fd = fd;
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory.");
  }
  if (!is_quant_model_) {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
  } else {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig) + sizeof(group_size_);
  }
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                  " into memory, the pointer to weight start address is null");
  }
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;
  config_->immediate_dim_ = config.immediate_dim_;
  // Set RoPE theta based on model type
  config_->rope_theta_ = (model_type_ == base::ModelType::kModelTypeLLama) ? 500000.0f : 1000000.0f;
  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  // Qwen tokenizer size and embedding size is mismatched
  // refer: https://github.com/QwenLM/Qwen2.5/issues/29
  // if (std::abs(config.vocab_size) != config_->vocab_size_) {
  //   return base::error::ModelParseError(
  //       "Vocabulary size mismatch between the model file and the token list.");
  // }
  config_->vocab_size_ = std::abs(config.vocab_size);
  return base::error::Success();
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

  // init sentence piece processor
  // google sentence piece
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }
  // Capture the tokenizer-derived vocab size before read_model_file()
  // overwrites config_->vocab_size_ with the .bin header value.
  const int32_t tokenizer_vocab_size = encode_layer_->vocab_size();
  // mmap
  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
    return mmap_status;
  }
  // Qwen3-style headers store vocab_size=0 (no layout info); restore the
  // tokenizer value or every vocab-sized tensor is allocated with 0 rows.
  // Headers with vocab_size>0 (Qwen2/Llama3) define the on-disk embedding
  // table size that all weight offsets depend on, so keep the header value
  // there and only warn on mismatch.
  if (config_->vocab_size_ <= 0) {
    config_->vocab_size_ = tokenizer_vocab_size;
  } else if (config_->vocab_size_ != tokenizer_vocab_size) {
    LOG(WARNING) << "[MODEL] header vocab_size=" << config_->vocab_size_
                 << " differs from tokenizer vocab_size=" << tokenizer_vocab_size
                 << "; keeping header value for weight layout";
  }
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
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
  // Use the tensor's actual sequence dimension (dim 1) for stride, not config_->seq_len_,
  // so the internal KV cache can be resized independently of the model's max context length.
  int32_t cache_seq_len = get_buffer(ModelBufferType::kKeyCache).get_dim(1);
  int32_t layer_offset = layer_idx * cache_seq_len * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}

base::Status Model::bind_external_kv_cache(const tensor::Tensor& key_cache,
                                           const tensor::Tensor& value_cache,
                                           int32_t slot_id) {
  using namespace base;
  if (kv_bound_) {
    unbind_external_kv_cache();
  }
  // The external cache now uses the head-dim-contiguous layout
  // [num_layers, num_slots, kv_dim, max_seq_len]; the legacy single-sequence
  // kernels index an internal [num_layers, seq, kv_dim] cache, which cannot
  // be expressed as a flat view of the transposed external layout. Serving
  // goes through forward_batch/decode_step (continuous batching), so this
  // legacy bridge is no longer supported — fail loudly instead of silently
  // mis-indexing the cache.
  const int32_t num_slots = key_cache.get_dim(1);
  const int32_t kv_dim = key_cache.get_dim(2);
  const int32_t max_seq_len = key_cache.get_dim(3);
  if (key_cache.get_dim(0) != config_->layer_num_ || kv_dim != config_->kv_dim_ ||
      value_cache.get_dim(0) != config_->layer_num_ || value_cache.get_dim(1) != num_slots ||
      value_cache.get_dim(2) != kv_dim || value_cache.get_dim(3) != max_seq_len) {
    return error::InvalidArgument("External KV cache shape out of range");
  }
  LOG(WARNING) << "bind_external_kv_cache is not supported with the head-dim-contiguous "
                  "external KV layout; use the continuous-batching path (forward_batch / "
                  "decode_step) instead.";
  return error::InvalidArgument(
      "Legacy external-KV binding is unsupported with the head-dim-contiguous cache layout.");
}

void Model::unbind_external_kv_cache() {
  if (!kv_bound_) {
    return;
  }
  buffers_.at(ModelBufferType::kKeyCache) = std::move(kv_key_backup_);
  buffers_.at(ModelBufferType::kValueCache) = std::move(kv_value_backup_);
  kv_bound_ = false;
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

  std::shared_ptr<base::Buffer> input_emb_buffer =
      std::make_shared<base::Buffer>(model_dim * sizeof(float), nullptr,
                                     input_embeddings.ptr<float>(index * model_dim), true);
  tensor::Tensor input(base::DataType::kDataTypeFp32, model_dim);
  input.assign(input_emb_buffer);
  input.set_device_type(device_type_);
  return input;
}

std::vector<int32_t> Model::post_processing_batch(const tensor::Tensor& logits) const {
  int32_t batch = logits.get_dim(0);
  int32_t vocab_size = logits.get_dim(1);
  std::vector<int32_t> tokens(batch);
  // Restrict sampling to the tokenizer's vocab range: the .bin header vocab
  // can be larger than the tokenizer's (Qwen3: 151936 vs 151665), and logits
  // beyond the tokenizer vocab correspond to reserved rows that cannot be
  // decoded. One batched kernel + one device-to-host copy (was: one kernel,
  // one copy and one stream sync per row).
  int32_t sample_vocab = vocab_size;
  if (encode_layer_ && encode_layer_->vocab_size() > 0) {
    sample_vocab = std::min(sample_vocab, encode_layer_->vocab_size());
  }
  sampler_->sample_batch(logits.ptr<float>(), vocab_size, sample_vocab, batch, tokens.data(),
                         nullptr);
  return tokens;
}

void Model::resize_internal_kv_cache(int32_t max_seq_len) {
  if (max_seq_len <= 0) return;
  // Drop any stale external-slot views before touching the real internal cache.
  unbind_external_kv_cache();

  // Any captured decode graph may reference the old internal buffers
  // (single-sequence prefill path); invalidate the pool so the next decode
  // re-captures against the resized cache. Cheap: resize happens at
  // benchmark setup time, not per decode step.
  if (!decode_graph_pool_.empty()) {
    decode_graph_pool_.clear();
  }

  auto alloc = (device_type_ == base::DeviceType::kDeviceCUDA)
      ? std::shared_ptr<base::DeviceAllocator>(base::CUDADeviceAllocatorFactory::get_instance())
      : base::CPUDeviceAllocatorFactory::get_instance();

  auto it_key = buffers_.find(ModelBufferType::kKeyCache);
  auto it_val = buffers_.find(ModelBufferType::kValueCache);

  if (it_key != buffers_.end() && it_val != buffers_.end()) {
    int32_t num_layers = it_key->second.get_dim(0);
    int32_t kv_dim = it_key->second.get_dim(2);

    it_key->second = tensor::Tensor(base::DataType::kDataTypeFp32,
                                     num_layers, max_seq_len, kv_dim, true, alloc);
    it_val->second = tensor::Tensor(base::DataType::kDataTypeFp32,
                                     num_layers, max_seq_len, kv_dim, true, alloc);
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    base::CUDADeviceAllocatorFactory::get_instance()->free_idle();
  }
}

}  // namespace model