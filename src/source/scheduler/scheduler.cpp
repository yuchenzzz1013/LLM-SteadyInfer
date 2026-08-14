#include "scheduler/scheduler.h"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

namespace scheduler {

Scheduler::Scheduler(std::shared_ptr<model::Model> model,
                     int max_batch_size,
                     int max_seq_len,
                     int max_gen_len)
    : model_(model),
      max_batch_size_(max_batch_size),
      max_seq_len_(max_seq_len),
      max_gen_len_(max_gen_len > 0 ? max_gen_len : max_seq_len) {
  int kv_dim = model_->kv_dim();
  int num_layers = model_->layer_num();
  if (kv_dim <= 0 || num_layers <= 0) {
    LOG(FATAL) << "Scheduler: model not properly initialized. kv_dim=" << kv_dim
               << " num_layers=" << num_layers;
  }
  LOG(INFO) << "[SCHED] Creating scheduler: max_batch=" << max_batch_size
            << " max_seq_len=" << max_seq_len
            << " kv_dim=" << kv_dim
            << " num_layers=" << num_layers;
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (model_->device_type() == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }
  kv_manager_ = std::make_unique<KVManager>(num_layers, max_batch_size,
                                             max_seq_len, kv_dim, alloc);

  // Preallocate the decode logits buffer once instead of per decode step.
  int vocab_size = model_->vocab_size();
  if (vocab_size <= 0) {
    LOG(FATAL) << "[SCHED] model vocab_size=" << vocab_size
               << " invalid — model file/tokenizer mismatch";
  }
  logits_ = tensor::Tensor(base::DataType::kDataTypeFp32, max_batch_size,
                           vocab_size, true, alloc);
  if (logits_.is_empty() || logits_.ptr<float>() == nullptr) {
    LOG(FATAL) << "[SCHED] Failed to preallocate logits buffer ["
               << max_batch_size << "," << vocab_size
               << "] — GPU may be out of memory";
  }
  LOG(INFO) << "[SCHED] Scheduler created, KVManager ready";
}

int Scheduler::add_request(const std::vector<int>& prompt_tokens) {
  Sequence seq;
  seq.id = next_seq_id_++;
  seq.prompt_tokens = prompt_tokens;
  seq.num_prompt_tokens = static_cast<int>(prompt_tokens.size());

  // Reject prompt if it does not leave room for at least 1 generation token.
  if (seq.num_prompt_tokens >= max_seq_len_) {
    LOG(ERROR) << "[SCHED] Rejecting request id=" << seq.id
               << ": prompt length " << seq.num_prompt_tokens
               << " >= max_seq_len " << max_seq_len_
               << " — KV cache has no room for generation.";
    return -1;
  }

  // Cap max_gen_len so prompt + generation fits within the KV cache.
  int avail = max_seq_len_ - seq.num_prompt_tokens;
  seq.max_gen_len = std::max(1, std::min(max_gen_len_, avail));

  waiting_queue_.push(seq);
  VLOG(1) << "[SCHED] add_request id=" << seq.id
          << " prompt_len=" << seq.num_prompt_tokens
          << " max_gen_len=" << seq.max_gen_len;
  return seq.id;
}

void Scheduler::step() {
  static int step_count = 0;
  step_count++;

  // Periodically flush the CUDA memory pool to prevent slow leaks from
  // accumulating over many steps. Freeing every step hurts throughput
  // (idle pooled buffers would otherwise be reused), so flush only
  // occasionally. forward_batch allocates ~10 scratch tensors per step that
  // are reused from this pool; flushing too often turns every re-allocation
  // into a cudaMalloc, which can synchronize the device and cause periodic
  // TPOT spikes.
  constexpr int kFreeIdleInterval = 1000;
  if (step_count % kFreeIdleInterval == 0) {
    VLOG(1) << "[SCHED] Periodic pool flush at step " << step_count;
    base::CUDADeviceAllocatorFactory::get_instance()->free_idle();
  }

  VLOG(1) << "[SCHED] === step " << step_count << " begin === "
          << "running=" << running_sequences_.size()
          << " waiting=" << waiting_queue_.size()
          << " finished=" << finished_sequences_.size();

  // Log GPU memory at each step (only at VLOG level to avoid spam)
  {
    size_t free_mem, total_mem;
    cudaError_t e = cudaMemGetInfo(&free_mem, &total_mem);
    if (e == cudaSuccess) {
      VLOG(1) << "[SCHED] GPU free=" << (free_mem >> 20) << "MB"
              << " total=" << (total_mem >> 20) << "MB"
              << " used=" << ((total_mem - free_mem) >> 20) << "MB";
    }
  }

  try_admit_sequences();
  auto batch = select_sequences();
  if (!batch.empty()) {
    VLOG(1) << "[SCHED] batch size=" << batch.size()
            << " is_prefill=" << !batch[0]->is_prefill_complete;
    execute_batch(batch);
  } else {
    VLOG(1) << "[SCHED] empty batch this step";
  }
  update_sequences();
}

bool Scheduler::all_finished() const {
  return waiting_queue_.empty() && running_sequences_.empty();
}

void Scheduler::try_admit_sequences() {
  auto now = std::chrono::steady_clock::now();
  while (!waiting_queue_.empty() && kv_manager_->has_free_slot() &&
         static_cast<int>(running_sequences_.size()) < max_batch_size_) {
    Sequence seq = waiting_queue_.front();
    waiting_queue_.pop();

    int slot = kv_manager_->allocate();
    if (slot < 0) break;

    seq.kv_slot_id = slot;
    seq.admit_time = now;
    running_sequences_.push_back(seq);
  }
}

std::vector<Sequence*> Scheduler::select_sequences() {
  std::vector<Sequence*> batch;

  // Prefill phase: batch every sequence that still needs prefill; their next
  // prompt chunks are flattened into one forward_batch call (token budget is
  // enforced in execute_batch). Prefill and decode run in separate steps so
  // the decode logits handling stays unchanged.
  for (auto& seq : running_sequences_) {
    if (seq.is_finished) continue;
    if (!seq.is_prefill_complete) {
      batch.push_back(&seq);
    }
  }
  if (!batch.empty()) {
    return batch;
  }

  // Decode phase: all running sequences, capped by max_batch_size.
  for (auto& seq : running_sequences_) {
    if (seq.is_finished) continue;
    if (static_cast<int>(batch.size()) >= max_batch_size_) break;
    batch.push_back(&seq);
  }

  return batch;
}

void Scheduler::execute_batch(const std::vector<Sequence*>& batch) {
  if (batch.empty()) return;

  bool is_prefill = !batch[0]->is_prefill_complete;

  if (is_prefill) {
    // Chunked batched prefill: flatten the next chunk of every pending
    // sequence's prompt into one forward_batch call. Each row is a prompt
    // token with its own position and KV slot; the batched kernels write K/V
    // straight into each sequence's external slot (no internal model cache,
    // no per-token predict calls). Logits are skipped — the first generated
    // token is produced by the following decode step, which already handles
    // sequences with empty generated_tokens.
    //
    // Adaptive token budget: prefill and decode never run in the same step,
    // so every prefill step delays decode tokens by its whole latency.
    //   - decode backlog high (>= 50% of max_batch)  -> shrink to 512 to
    //     interrupt prefill earlier and protect TPOT;
    //   - decode backlog low (< 25% of max_batch)    -> grow to 4096 to
    //     shorten TTFT;
    //   - in between                                  -> 1024.
    // The budget moves at most one level per step (dampening) so bursts do
    // not oscillate the chunk size.
    {
      int decode_backlog = 0;
      for (const auto& seq : running_sequences_) {
        if (!seq.is_finished && seq.is_prefill_complete) {
          ++decode_backlog;
        }
      }
      const float pressure =
          max_batch_size_ > 0 ? static_cast<float>(decode_backlog) / max_batch_size_ : 0.f;
      int target = 1024;
      if (pressure >= 0.5f) {
        target = 512;
      } else if (pressure < 0.25f) {
        target = 4096;
      }
      if (target > prefill_token_budget_) {
        prefill_token_budget_ = std::min(target, prefill_token_budget_ * 2);
      } else if (target < prefill_token_budget_) {
        prefill_token_budget_ = std::max(target, prefill_token_budget_ / 2);
      }
      prefill_token_budget_ = std::max(512, std::min(4096, prefill_token_budget_));
      VLOG(1) << "[SCHED] prefill budget=" << prefill_token_budget_
              << " decode_backlog=" << decode_backlog << " pressure=" << pressure;
    }
    const int kMaxPrefillTokensPerStep = prefill_token_budget_;

    std::vector<int32_t> input_ids;
    std::vector<int32_t> positions;
    std::vector<int32_t> kv_offsets;
    input_ids.reserve(kMaxPrefillTokensPerStep);
    positions.reserve(kMaxPrefillTokensPerStep);
    kv_offsets.reserve(kMaxPrefillTokensPerStep);

    for (Sequence* seq : batch) {
      if (seq->is_prefill_complete) continue;
      int remaining = seq->num_prompt_tokens - seq->next_prefill_chunk_start;
      int take = std::min(remaining,
                          kMaxPrefillTokensPerStep - static_cast<int>(input_ids.size()));
      for (int i = 0; i < take; ++i) {
        int p = seq->next_prefill_chunk_start + i;
        input_ids.push_back(seq->prompt_tokens[p]);
        positions.push_back(p);
        kv_offsets.push_back(seq->kv_slot_id);
      }
      seq->next_prefill_chunk_start += take;
      if (seq->next_prefill_chunk_start >= seq->num_prompt_tokens) {
        seq->is_prefill_complete = true;
      }
      if (static_cast<int>(input_ids.size()) >= kMaxPrefillTokensPerStep) break;
    }
    if (input_ids.empty()) return;

    int total_tokens = static_cast<int>(input_ids.size());
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor ids_tensor(base::DataType::kDataTypeInt32, total_tokens, true, alloc_cpu);
    tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, total_tokens, true, alloc_cpu);
    tensor::Tensor off_tensor(base::DataType::kDataTypeInt32, total_tokens, true, alloc_cpu);
    for (int i = 0; i < total_tokens; ++i) {
      ids_tensor.index<int32_t>(i) = input_ids[i];
      pos_tensor.index<int32_t>(i) = positions[i];
      off_tensor.index<int32_t>(i) = kv_offsets[i];
    }

    VLOG(1) << "[SCHED] prefill step: seqs=" << batch.size()
            << " tokens=" << total_tokens;

    auto status = model_->forward_batch(ids_tensor, pos_tensor, off_tensor,
                                        kv_manager_->key_cache(),
                                        kv_manager_->value_cache(),
                                        logits_, false);
    if (!status) {
      LOG(ERROR) << "Batch prefill failed: " << status.get_err_msg();
      // Mark all running sequences as finished to prevent infinite retry loop.
      auto now = std::chrono::steady_clock::now();
      for (auto& seq : running_sequences_) {
        if (!seq.is_finished) {
          LOG(WARNING) << "Force-finishing seq id=" << seq.id
                       << " due to batch prefill failure";
          seq.is_finished = true;
          seq.finish_time = now;
          kv_manager_->deallocate(seq.kv_slot_id);
          finished_sequences_.push_back(seq);
        }
      }
      running_sequences_.clear();
      return;
    }
  } else {
    // Decode: batched forward for all sequences
    auto recon_start = std::chrono::steady_clock::now();

    int batch_size = static_cast<int>(batch.size());
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    VLOG(1) << "[SCHED] decode step: batch_size=" << batch_size
            << " vocab_size=" << model_->vocab_size();

    tensor::Tensor input_ids(base::DataType::kDataTypeInt32, batch_size, true, alloc_cpu);
    tensor::Tensor positions(base::DataType::kDataTypeInt32, batch_size, true, alloc_cpu);
    tensor::Tensor kv_offsets(base::DataType::kDataTypeInt32, batch_size, true, alloc_cpu);

    for (int i = 0; i < batch_size; ++i) {
      Sequence* seq = batch[i];
      int token_id;
      if (seq->generated_tokens.empty()) {
        token_id = seq->prompt_tokens.back();
      } else {
        token_id = seq->generated_tokens.back();
      }
      int position = seq->num_prompt_tokens + seq->num_generated_tokens - 1;

      input_ids.index<int32_t>(i) = token_id;
      positions.index<int32_t>(i) = position;
      kv_offsets.index<int32_t>(i) = seq->kv_slot_id;
      VLOG(2) << "[SCHED] seq[" << i << "] id=" << seq->id
              << " slot=" << seq->kv_slot_id
              << " pos=" << position
              << " token=" << token_id;
    }

    // View into the preallocated logits buffer (no per-step allocation).
    tensor::Tensor logits_view(base::DataType::kDataTypeFp32, batch_size,
                               model_->vocab_size(), false, nullptr,
                               logits_.ptr<float>());
    logits_view.set_device_type(model_->device_type());

    // Record batch reconstruction time (tensor setup overhead)
    {
      auto recon_end = std::chrono::steady_clock::now();
      double recon_ms = std::chrono::duration<double, std::milli>(recon_end - recon_start).count();
      batch_reconstruct_times_ms_.push_back(recon_ms);
    }

    {
      size_t free_mem, total_mem;
      cudaMemGetInfo(&free_mem, &total_mem);
      VLOG(1) << "[SCHED] before forward_batch: GPU free=" << (free_mem >> 20) << "MB";
    }

    // Decode goes through decode_step: first call per batch size captures a
    // CUDA graph, later calls only stage inputs and replay it (falls back to
    // forward_batch automatically when graphs are unavailable).
    auto status = model_->decode_step(input_ids, positions, kv_offsets,
                                      kv_manager_->key_cache(),
                                      kv_manager_->value_cache(),
                                      logits_view);
    if (!status) {
      LOG(ERROR) << "Batch forward failed: " << status.get_err_msg();
      // Mark all running sequences as finished to prevent infinite retry loop.
      // If the GPU is OOM or in an error state, retrying will only produce
      // the same error and flood the logs.
      auto now = std::chrono::steady_clock::now();
      for (auto& seq : running_sequences_) {
        if (!seq.is_finished) {
          LOG(WARNING) << "Force-finishing seq id=" << seq.id
                       << " due to batch forward failure";
          seq.is_finished = true;
          seq.finish_time = now;
          kv_manager_->deallocate(seq.kv_slot_id);
          finished_sequences_.push_back(seq);
        }
      }
      running_sequences_.clear();
      return;
    }

    // Post-processing on GPU: the sampler (ArgmaxSampler) launches a CUDA
    // kernel to find the argmax index directly on the GPU, then copies only
    // the result (a single index) back to the host.  Do NOT call logits.to_cpu()
    // here — that would replace the GPU pointer with a CPU pointer, and the
    // sampler's CUDA kernel would attempt to read host memory from the device,
    // causing an illegal memory access and corrupting the CUDA context.
    // The sampling kernels run on the default stream, so the model stream
    // must be drained before they read the logits.
    model_->sync_stream();
    auto next_tokens = model_->post_processing_batch(logits_view);

    // Update sequences
    auto now = std::chrono::steady_clock::now();
    for (int i = 0; i < batch_size; ++i) {
      if (batch[i]->generated_tokens.empty()) {
        batch[i]->first_token_time = now;
      }
      batch[i]->generated_tokens.push_back(next_tokens[i]);
      batch[i]->num_generated_tokens++;
    }
  }
}

void Scheduler::update_sequences() {
  auto now = std::chrono::steady_clock::now();
  auto it = running_sequences_.begin();
  while (it != running_sequences_.end()) {
    if (it->is_finished) {
      it = running_sequences_.erase(it);
      continue;
    }

    if (it->num_generated_tokens >= it->max_gen_len) {
      it->is_finished = true;
      it->finish_time = now;
      kv_manager_->deallocate(it->kv_slot_id);
      finished_sequences_.push_back(*it);
      it = running_sequences_.erase(it);
      continue;
    }

    if (it->is_prefill_complete && !it->generated_tokens.empty()) {
      int last_token = it->generated_tokens.back();
      if (model_->is_sentence_ending(last_token)) {
        it->is_finished = true;
        it->finish_time = now;
        kv_manager_->deallocate(it->kv_slot_id);
        finished_sequences_.push_back(*it);
        it = running_sequences_.erase(it);
        continue;
      }
    }

    ++it;
  }
}

int Scheduler::get_busy_kv_slots() const {
  return kv_manager_->busy_slot_count();
}

int Scheduler::get_max_kv_seq_len() const {
  return kv_manager_->max_seq_len();
}

}  // namespace scheduler
