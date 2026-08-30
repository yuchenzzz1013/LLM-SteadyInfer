#include "scheduler/scheduler.h"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <string>
#include <utility>

namespace scheduler {

int Scheduler::resolve_block_size(long long avg_prompt_len) {
  if (avg_prompt_len < 64) return 8;    // short prompts: smaller pages, less waste
  if (avg_prompt_len > 1024) return 32; // long prompts: larger pages, better locality
  return 16;
}

Scheduler::Scheduler(std::shared_ptr<model::Model> model,
                     int max_batch_size,
                     int max_seq_len,
                     int max_gen_len,
                     int block_size)
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
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (model_->device_type() == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }
  // Paged KV pool block size: explicit arg > env LLAMA_BLOCK_SIZE > default
  // 16. Drivers pick the workload-adaptive value via resolve_block_size.
  if (block_size > 0) {
    block_size_ = block_size;
  } else {
    const char* bs_env = std::getenv("LLAMA_BLOCK_SIZE");
    if (bs_env && std::atoi(bs_env) >= 1) {
      block_size_ = std::atoi(bs_env);
    }
  }
  kv_manager_ = std::make_unique<KVManager>(num_layers, max_batch_size,
                                             max_seq_len, kv_dim, alloc,
                                             model_->device_type(), block_size_);

#ifdef USE_PAGED_ATTENTION
  // Prefix cache (content-hash block sharing): enabled by default in paged
  // mode; LLAMA_DISABLE_PREFIX_CACHE=1 disables it.
  {
    const char* pc_env = std::getenv("LLAMA_DISABLE_PREFIX_CACHE");
    if (!(pc_env && std::string(pc_env) == "1")) {
      prefix_cache_ = std::make_unique<PrefixCache>(block_size_);
    }
  }
#endif

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
}

Scheduler::~Scheduler() {
#ifdef USE_PAGED_ATTENTION
  // The block pool is pure bookkeeping: refcount leaks (or double frees)
  // silently strand blocks. Verify the invariant before the pool dies.
  if (kv_manager_ && kv_manager_->block_allocator()) {
    if (!kv_manager_->block_allocator()->invariant_holds()) {
      LOG(FATAL) << "[SCHED] BlockAllocator invariant violated at scheduler "
                    "destruction: refcount leak or double-free";
    }
  }
#endif
}

int Scheduler::add_request(const std::vector<int>& prompt_tokens) {
  Sequence seq;
  seq.id = next_seq_id_++;
  seq.arrival_time = std::chrono::steady_clock::now();
  seq.prompt_tokens = prompt_tokens;
  seq.num_prompt_tokens = static_cast<int>(prompt_tokens.size());
  seq.state = SeqState::WAITING;

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

  const int seq_id = seq.id;
  const int seq_prompt_len = seq.num_prompt_tokens;
  const int seq_max_gen_len = seq.max_gen_len;
  waiting_queue_.push_back(std::move(seq));
#ifndef NDEBUG
  VLOG(1) << "[SCHED] add_request id=" << seq_id
          << " prompt_len=" << seq_prompt_len
          << " max_gen_len=" << seq_max_gen_len;
#endif
  return seq_id;
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
  // TPOT spikes. (Block-level recycling never reaches this pool: freed blocks
  // are pure index bookkeeping until the pool itself is destroyed.)
  constexpr int kFreeIdleInterval = 1000;
  if (step_count % kFreeIdleInterval == 0) {
#ifndef NDEBUG
    VLOG(1) << "[SCHED] Periodic pool flush at step " << step_count;
#endif
    base::CUDADeviceAllocatorFactory::get_instance()->free_idle();
  }

#ifndef NDEBUG
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
#endif

  try_admit_sequences();
  auto rows = build_batch_rows();
  if (!rows.empty()) {
#ifndef NDEBUG
    VLOG(1) << "[SCHED] batch size=" << rows.size()
            << " is_pure_decode=" << (rows[0].is_decode ? 1 : 0);
#endif
    execute_batch(rows);
  } else {
#ifndef NDEBUG
    VLOG(1) << "[SCHED] empty batch this step";
#endif
  }
  update_sequences();
}

bool Scheduler::all_finished() const {
  return waiting_queue_.empty() && running_sequences_.empty();
}

// ========== Admission (block-aware, with recompute-style preemption) ==========

void Scheduler::try_admit_sequences() {
  auto now = std::chrono::steady_clock::now();
#ifdef USE_PAGED_ATTENTION
  while (!waiting_queue_.empty() &&
         static_cast<int>(running_sequences_.size()) < max_batch_size_) {
    Sequence& seq = waiting_queue_.front();

    if (seq.state == SeqState::PREEMPTED) {
      // Recompute-style resume: the kept prefix blocks (and their block-table
      // row) are retained; nothing to allocate — just put it back to work.
      Sequence resumed = std::move(seq);
      waiting_queue_.pop_front();
      resumed.state = SeqState::RUNNING;
      resumed.admit_time = now;
#ifndef NDEBUG
      VLOG(1) << "[SCHED] resumed preempted seq id=" << resumed.id
              << " blocks=" << resumed.num_blocks_allocated;
#endif
      running_sequences_.push_back(std::move(resumed));
      continue;
    }

    // WAITING: reserve the prompt's blocks now; generation grows lazily
    // (ensure_blocks_for) so the pool serves more concurrent sequences.
    int prompt_blocks = (seq.num_prompt_tokens + block_size_ - 1) / block_size_;

    // Prefix cache: whole-block prompt prefixes are shared read-only — only
    // the non-shared remainder needs fresh blocks and prefill.
    std::vector<int32_t> shared_blocks;
    int matched = 0;
    const auto lookup_prefix = [&]() {
      matched = 0;
      shared_blocks.clear();
      if (prefix_cache_) {
        matched = prefix_cache_->lookup(seq.prompt_tokens, *kv_manager_->block_allocator(),
                                        &shared_blocks);
      }
    };
    lookup_prefix();
    int private_blocks = prompt_blocks - matched;

    int row = kv_manager_->block_allocator()->allocate_blocks(private_blocks);
    if (row < 0) {
      // Pool exhausted: preempt tail blocks of other RUNNING sequences.
      // If that does not free enough, the request keeps waiting (FCFS order).
      if (!try_preempt_for(private_blocks, seq.id)) {
        break;
      }
      // Preemption may have evicted the very blocks the prefix match shares
      // (the victim can be the cache's owner): re-validate the match before
      // allocating, so a recycled block is never mounted twice.
      lookup_prefix();
      private_blocks = prompt_blocks - matched;
      row = kv_manager_->block_allocator()->allocate_blocks(private_blocks);
      if (row < 0) break;  // defensive; should not happen after a successful preempt
    }
    if (matched > 0) {
      CHECK(kv_manager_->block_allocator()->reserve_shared_prefix(row, shared_blocks));
    }
    Sequence admitted = std::move(seq);
    waiting_queue_.pop_front();
    admitted.kv_slot_id = row;
    admitted.num_blocks_allocated = prompt_blocks;
    admitted.state = SeqState::RUNNING;
    admitted.admit_time = now;
    if (matched > 0) {
      // Skip prefilling the shared region (the TTFT win): resume right after
      // the matched whole-block prefix. A fully matched prompt is
      // prefill-complete at admission — its first decode step is next.
      admitted.next_prefill_chunk_start =
          std::min(matched * block_size_, admitted.num_prompt_tokens);
      admitted.is_prefill_complete =
          (admitted.next_prefill_chunk_start >= admitted.num_prompt_tokens);
    }
#ifndef NDEBUG
    VLOG(1) << "[SCHED] admitted seq id=" << admitted.id
            << " prompt_blocks=" << prompt_blocks
            << " shared_prefix=" << matched << " row=" << row;
#endif
    running_sequences_.push_back(std::move(admitted));
  }
#else
  // Continuous layout: unchanged slot admission.
  while (!waiting_queue_.empty() && kv_manager_->has_free_slot() &&
         static_cast<int>(running_sequences_.size()) < max_batch_size_) {
    Sequence seq = std::move(waiting_queue_.front());
    waiting_queue_.pop_front();

    int slot = kv_manager_->allocate();
    if (slot < 0) break;

    seq.kv_slot_id = slot;
    seq.state = SeqState::RUNNING;
    seq.admit_time = now;
    running_sequences_.push_back(std::move(seq));
  }
#endif
}

bool Scheduler::try_preempt_for(int need_blocks, int skip_seq_id) {
#ifdef USE_PAGED_ATTENTION
  int needed = need_blocks - kv_manager_->block_allocator()->free_block_count();
  if (needed <= 0) return true;

  // Collect victims from the most recently admitted RUNNING sequence to the
  // oldest (FCFS: latecomers yield first), each keeping at least 1 block so
  // its block-table row stays alive while it waits for re-admission.
  std::vector<std::pair<Sequence*, int>> victims;  // (seq, blocks to free)
  for (auto it = running_sequences_.rbegin(); it != running_sequences_.rend(); ++it) {
    if (needed <= 0) break;
    if (it->id == skip_seq_id || it->state != SeqState::RUNNING) continue;
    int evictable = it->num_blocks_allocated - 1;  // keep >= 1 block
    if (evictable <= 0) continue;
    int take = std::min(evictable, needed);
    needed -= take;
    victims.emplace_back(&(*it), take);
  }
  if (needed > 0) {
    // Not enough evictable blocks — do not churn any victim for a request
    // that still cannot fit.
    return false;
  }

  for (auto& [victim, take] : victims) {
    truncate_sequence(*victim, victim->num_blocks_allocated - take);
  }
  // Move every PREEMPTED sequence to the head of the waiting queue (they are
  // in-flight requests — they jump ahead of fresh arrivals on resume).
  auto it = running_sequences_.begin();
  while (it != running_sequences_.end()) {
    if (it->state == SeqState::PREEMPTED) {
      waiting_queue_.push_front(std::move(*it));
      it = running_sequences_.erase(it);
    } else {
      ++it;
    }
  }
  return true;
#else
  UNUSED(need_blocks);
  UNUSED(skip_seq_id);
  return false;  // continuous layout: no blocks, no preemption
#endif
}

void Scheduler::truncate_sequence(Sequence& seq, int keep_blocks) {
#ifdef USE_PAGED_ATTENTION
  const int trunc_pos = keep_blocks * block_size_;
  kv_manager_->block_allocator()->free_blocks_from(seq.kv_slot_id, keep_blocks);
  seq.num_blocks_allocated = keep_blocks;
  if (trunc_pos < seq.num_prompt_tokens) {
    // Evicted into the prompt region: re-prefill from the truncation point.
    // Every generated token (position >= num_prompt_tokens > trunc_pos) loses
    // its KV and is discarded; greedy sampling makes the recompute
    // deterministic, so the regenerated tail matches bit-for-bit.
    seq.next_prefill_chunk_start = trunc_pos;
    seq.is_prefill_complete = false;
    seq.generated_tokens.clear();
    seq.num_generated_tokens = 0;
  } else {
    // Prefill KV retained; drop generated tokens beyond the truncation point.
    int keep_gen = trunc_pos - seq.num_prompt_tokens;
    if (static_cast<int>(seq.generated_tokens.size()) > keep_gen) {
      seq.generated_tokens.resize(keep_gen);
      seq.num_generated_tokens = keep_gen;
    }
  }
  seq.state = SeqState::PREEMPTED;
#ifndef NDEBUG
  VLOG(1) << "[SCHED] preempted seq id=" << seq.id
          << " keep_blocks=" << keep_blocks
          << " trunc_pos=" << trunc_pos
          << " prompt=" << seq.num_prompt_tokens
          << " gen=" << seq.num_generated_tokens;
#endif
#else
  UNUSED(seq);
  UNUSED(keep_blocks);
#endif
}

bool Scheduler::ensure_blocks_for(Sequence* seq, int position) {
#ifdef USE_PAGED_ATTENTION
  const int needed = position / block_size_ + 1;
  if (needed > kv_manager_->max_blocks_per_seq()) {
    LOG(ERROR) << "[SCHED] seq id=" << seq->id << " position " << position
               << " exceeds the per-sequence block capacity";
    return false;
  }
  while (seq->num_blocks_allocated < needed) {
    int grown = kv_manager_->block_allocator()->append_blocks(seq->kv_slot_id, 1);
    if (grown <= seq->num_blocks_allocated) {
      // Pool exhausted: preempt tail blocks of other sequences, then retry.
      if (!try_preempt_for(needed - seq->num_blocks_allocated, seq->id)) {
        return false;
      }
      continue;
    }
    seq->num_blocks_allocated = grown;
  }
  return true;
#else
  UNUSED(seq);
  UNUSED(position);
  return true;  // continuous layout: the whole slot is reserved at admission
#endif
}

// ========== Batch formation (mixed chunked prefill + decode) ==========

std::vector<Scheduler::BatchRow> Scheduler::build_batch_rows() {
  std::vector<BatchRow> rows;
  rows.reserve(max_batch_size_);

  // 1. Decode rows first: every RUNNING, prefill-complete sequence. Decode
  // rows sit at rows[0..num_decode_rows) so the sampler output aligns.
  for (auto& seq : running_sequences_) {
    if (seq.state != SeqState::RUNNING || seq.is_finished) continue;
    if (!seq.is_prefill_complete) continue;
    if (static_cast<int>(rows.size()) >= max_batch_size_) break;
    int position = seq.num_prompt_tokens + seq.num_generated_tokens - 1;
    if (!ensure_blocks_for(&seq, position)) {
      LOG(ERROR) << "[SCHED] seq id=" << seq.id
                 << " cannot grow KV blocks (pool exhausted, no preemption "
                    "victims); force-finishing";
      seq.is_finished = true;
      seq.state = SeqState::FINISHED;
      continue;  // update_sequences retires it
    }
    int token_id = seq.generated_tokens.empty() ? seq.prompt_tokens.back()
                                                : seq.generated_tokens.back();
    rows.push_back(BatchRow{&seq, true, static_cast<int32_t>(token_id),
                            static_cast<int32_t>(position)});
  }

  // 2. Chunked-prefill rows, mixed into the same step so decode never waits
  // for a full prefill pass. The chunk budget adapts to decode backlog
  // pressure (same thresholds as the old prefill-only steps):
  //   - decode backlog high (>= 50% of max_batch)  -> shrink to 512 (protect TPOT);
  //   - decode backlog low (< 25% of max_batch)    -> grow to 4096 (shorten TTFT);
  //   - in between                                  -> 1024.
  // The budget moves at most one level per step (dampening) so bursts do not
  // oscillate the chunk size. The remaining batch capacity (logits buffer is
  // [max_batch, vocab]) caps the chunk as well.
  bool has_prefill_pending = false;
  for (const auto& seq : running_sequences_) {
    if (seq.state == SeqState::RUNNING && !seq.is_finished && !seq.is_prefill_complete) {
      has_prefill_pending = true;
      break;
    }
  }
  if (has_prefill_pending) {
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
#ifndef NDEBUG
      VLOG(1) << "[SCHED] prefill budget=" << prefill_token_budget_
              << " decode_backlog=" << decode_backlog << " pressure=" << pressure;
#endif
    }

    int prefill_taken = 0;
    for (auto& seq : running_sequences_) {
      if (seq.state != SeqState::RUNNING || seq.is_finished) continue;
      if (seq.is_prefill_complete) continue;
      if (prefill_taken >= prefill_token_budget_) break;
      if (static_cast<int>(rows.size()) >= max_batch_size_) break;

      int remaining = seq.num_prompt_tokens - seq.next_prefill_chunk_start;
      int take = std::min(remaining, prefill_token_budget_ - prefill_taken);
      take = std::min(take, max_batch_size_ - static_cast<int>(rows.size()));
      if (take <= 0) break;

      // Lazy growth for the whole chunk up front (its highest position needs
      // the most blocks). On failure, shrink the chunk to what the current
      // blocks already cover so the sequence still makes progress.
      if (!ensure_blocks_for(&seq, seq.next_prefill_chunk_start + take - 1)) {
        int covered = seq.num_blocks_allocated * block_size_;
        take = covered - seq.next_prefill_chunk_start;
        if (take <= 0) {
          LOG(ERROR) << "[SCHED] seq id=" << seq.id
                     << " cannot grow KV blocks for prefill; force-finishing";
          seq.is_finished = true;
          seq.state = SeqState::FINISHED;
          continue;
        }
      }

      for (int i = 0; i < take; ++i) {
        int p = seq.next_prefill_chunk_start + i;
        rows.push_back(BatchRow{&seq, false, static_cast<int32_t>(seq.prompt_tokens[p]),
                                static_cast<int32_t>(p)});
      }
      prefill_taken += take;
      seq.next_prefill_chunk_start += take;
      if (seq.next_prefill_chunk_start >= seq.num_prompt_tokens) {
        seq.is_prefill_complete = true;
        // Prefix cache: only now is the prompt's KV committed — record the
        // full blocks so later identical-prefix requests can share them.
        if (prefix_cache_) {
          prefix_cache_->insert(seq.prompt_tokens, *kv_manager_->block_allocator(),
                                seq.kv_slot_id);
        }
      }
    }
  }

  return rows;
}

// ========== Execution ==========

void Scheduler::force_finish_all(const char* reason) {
  LOG(ERROR) << reason;
  // Mark all running sequences as finished to prevent an infinite retry loop.
  // If the GPU is OOM or in an error state, retrying will only produce the
  // same error and flood the logs.
  auto now = std::chrono::steady_clock::now();
  for (auto& seq : running_sequences_) {
    if (!seq.is_finished) {
      LOG(WARNING) << "Force-finishing seq id=" << seq.id << ": " << reason;
      seq.is_finished = true;
      seq.state = SeqState::FINISHED;
      seq.finish_time = now;
      kv_manager_->deallocate(seq.kv_slot_id);
      finished_sequences_.push_back(seq);
    }
  }
  running_sequences_.clear();
}

void Scheduler::execute_batch(const std::vector<BatchRow>& rows) {
  if (rows.empty()) return;

  const int batch = static_cast<int>(rows.size());
  int num_decode_rows = 0;
  for (const auto& r : rows) {
    if (r.is_decode) ++num_decode_rows;
  }
  const bool any_prefill = num_decode_rows < batch;

  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor input_ids(base::DataType::kDataTypeInt32, batch, true, alloc_cpu);
  tensor::Tensor positions(base::DataType::kDataTypeInt32, batch, true, alloc_cpu);
  // block_table: per-row physical block ids (paged) or slot ids (continuous).
#ifdef USE_PAGED_ATTENTION
  const int32_t table_stride = kv_manager_->max_blocks_per_seq();
  tensor::Tensor block_table(base::DataType::kDataTypeInt32, batch, table_stride, true, alloc_cpu);
#else
  const int32_t table_stride = 1;
  tensor::Tensor block_table(base::DataType::kDataTypeInt32, batch, true, alloc_cpu);
#endif

  for (int i = 0; i < batch; ++i) {
    const BatchRow& r = rows[i];
    input_ids.index<int32_t>(i) = r.token_id;
    positions.index<int32_t>(i) = r.position;
#ifdef USE_PAGED_ATTENTION
    kv_manager_->block_allocator()->copy_block_table_row(
        r.seq->kv_slot_id, block_table.ptr<int32_t>() + static_cast<int64_t>(i) * table_stride);
#else
    block_table.index<int32_t>(i) = r.seq->kv_slot_id;
#endif
#ifndef NDEBUG
    VLOG(2) << "[SCHED] row[" << i << "] seq id=" << r.seq->id
            << " is_decode=" << (r.is_decode ? 1 : 0)
            << " pos=" << r.position
            << " token=" << r.token_id;
#endif
  }

#ifndef NDEBUG
  VLOG(1) << "[SCHED] batch size=" << batch
          << " decode_rows=" << num_decode_rows
          << " prefill_rows=" << (batch - num_decode_rows);
#endif

  if (!any_prefill) {
    // Pure-decode step: the CUDA-graph path (decode_step) with its
    // per-batch-size graph pool.
    auto recon_start = std::chrono::steady_clock::now();

    // View into the preallocated logits buffer (no per-step allocation).
    tensor::Tensor logits_view(base::DataType::kDataTypeFp32, batch,
                               model_->vocab_size(), false, nullptr,
                               logits_.ptr<float>());
    logits_view.set_device_type(model_->device_type());

    // Record batch reconstruction time (tensor setup overhead)
    {
      auto recon_end = std::chrono::steady_clock::now();
      double recon_ms = std::chrono::duration<double, std::milli>(recon_end - recon_start).count();
      batch_reconstruct_times_ms_.push_back(recon_ms);
    }

    auto status = model_->decode_step(input_ids, positions, block_table,
                                      kv_manager_->key_cache(),
                                      kv_manager_->value_cache(),
                                      logits_view);
    if (!status) {
      force_finish_all("batch decode failed");
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
    for (int i = 0; i < batch; ++i) {
      Sequence* seq = rows[i].seq;
      if (seq->generated_tokens.empty()) {
        seq->first_token_time = now;
      } else {
        // ITL: 距上一个生成 token 的间隔(在线压测指标)
        double itl_ms = std::chrono::duration<double, std::milli>(
                            now - seq->last_token_time).count();
        seq->token_timestamps_ms.push_back(itl_ms);
      }
      seq->last_token_time = now;
      seq->generated_tokens.push_back(next_tokens[i]);
      seq->num_generated_tokens++;
    }
  } else {
    // Prefill step (pure, or mixed with decode rows): plain forward_batch —
    // the chunk composition varies per step, so it stays outside the decode
    // CUDA graphs. With decode rows present, logits are computed for the
    // whole batch (prefill-row logits are wasted compute but keep the
    // sampling rows aligned); a pure-prefill step skips the LM head entirely.
    const bool need_logits = num_decode_rows > 0;
    auto status = model_->forward_batch(input_ids, positions, block_table,
                                        kv_manager_->key_cache(),
                                        kv_manager_->value_cache(),
                                        logits_, need_logits);
    if (!status) {
      force_finish_all("batch prefill failed");
      return;
    }

    if (need_logits) {
      // Sample only the decode rows: prefill rows produce no token this step.
      tensor::Tensor logits_view(base::DataType::kDataTypeFp32, batch,
                                 model_->vocab_size(), false, nullptr,
                                 logits_.ptr<float>());
      logits_view.set_device_type(model_->device_type());
      model_->sync_stream();
      auto next_tokens = model_->post_processing_batch(logits_view);

      auto now = std::chrono::steady_clock::now();
      for (int i = 0; i < batch; ++i) {
        if (!rows[i].is_decode) continue;
        Sequence* seq = rows[i].seq;
        if (seq->generated_tokens.empty()) {
          seq->first_token_time = now;
        } else {
          double itl_ms = std::chrono::duration<double, std::milli>(
                              now - seq->last_token_time).count();
          seq->token_timestamps_ms.push_back(itl_ms);
        }
        seq->last_token_time = now;
        seq->generated_tokens.push_back(next_tokens[i]);
        seq->num_generated_tokens++;
      }
    }
  }
}

void Scheduler::update_sequences() {
  auto now = std::chrono::steady_clock::now();
  auto it = running_sequences_.begin();
  while (it != running_sequences_.end()) {
    if (it->is_finished) {
      // Finished mid-build (block-growth failure): retire here.
      it->state = SeqState::FINISHED;
      it->finish_time = now;
      kv_manager_->deallocate(it->kv_slot_id);
      finished_sequences_.push_back(*it);
      it = running_sequences_.erase(it);
      continue;
    }

    if (it->num_generated_tokens >= it->max_gen_len) {
      it->is_finished = true;
      it->state = SeqState::FINISHED;
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
        it->state = SeqState::FINISHED;
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

long long Scheduler::get_allocated_kv_tokens() const {
#ifdef USE_PAGED_ATTENTION
  // Reserved KV capacity = actually allocated blocks x block_size (lazy
  // growth makes this the truthful reservation, unlike busy x max_seq_len).
  long long blocks = 0;
  for (const auto& s : running_sequences_) {
    blocks += s.num_blocks_allocated;
  }
  return blocks * block_size_;
#else
  return static_cast<long long>(get_busy_kv_slots()) * max_seq_len_;
#endif
}

}  // namespace scheduler
