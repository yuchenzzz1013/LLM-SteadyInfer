#pragma once

#include <deque>
#include <memory>
#include <vector>
#include "base/base.h"
#include "kv_manager.h"
#include "model/model.h"
#include "prefix_cache.h"
#include "sequence.h"

namespace scheduler {

class Scheduler {
 public:
  // max_seq_len: per-sequence KV capacity (prompt + generation, in tokens).
  // max_gen_len: per-request generation token limit (clamped to fit in cache).
  // block_size: paged pool block size; 0 = env LLAMA_BLOCK_SIZE, else the
  // workload heuristic (see resolve_block_size).
  Scheduler(std::shared_ptr<model::Model> model,
            int max_batch_size,
            int max_seq_len,
            int max_gen_len = 0,
            int block_size = 0);

  // Verifies the block pool invariant (no refcount leak stranded blocks).
  ~Scheduler();

  int add_request(const std::vector<int>& prompt_tokens);
  void step();                          // Single scheduling iteration
  bool all_finished() const;
  const std::vector<Sequence>& get_finished() const { return finished_sequences_; }
  const std::vector<Sequence>& get_running() const { return running_sequences_; }

  // Batch reconstruction timing (per-decode-step overhead)
  const std::vector<double>& get_batch_reconstruct_times_ms() const {
    return batch_reconstruct_times_ms_;
  }

  // KV cache stats
  int get_busy_kv_slots() const;
  int get_max_kv_seq_len() const;
  // Reserved KV capacity in tokens (paged: allocated blocks x block_size;
  // continuous: busy slots x max_seq_len). 0 when unavailable.
  long long get_allocated_kv_tokens() const;
  int get_block_size() const { return block_size_; }

  // 等待队列长度(在线压测的排队指标)
  int num_waiting() const { return static_cast<int>(waiting_queue_.size()); }

  // Runtime-selected paged block size from the workload's average prompt
  // length: short prompts waste less capacity with small pages, long prompts
  // get better locality with large pages.
  static int resolve_block_size(long long avg_prompt_len);

 private:
  // One flattened batch row: a decode token or one chunked-prefill token.
  struct BatchRow {
    Sequence* seq = nullptr;
    bool is_decode = false;
    int32_t token_id = 0;
    int32_t position = 0;
  };

  void try_admit_sequences();
  // Mixed batch: decode rows first (they consume the sampled logits rows),
  // then chunked-prefill rows up to the adaptive token budget. Pure-decode
  // batches keep the CUDA-graph decode path.
  std::vector<BatchRow> build_batch_rows();
  void execute_batch(const std::vector<BatchRow>& rows);
  void update_sequences();

  // Preemption (recompute-style, no CPU swap): evict tail blocks from the
  // most recently admitted RUNNING sequence until `need_blocks` are free.
  // Victims keep >= 1 block, get truncated to the block boundary, and are
  // moved to the front of the waiting queue. skip_seq_id avoids victimizing
  // the sequence the blocks are being freed FOR.
  bool try_preempt_for(int need_blocks, int skip_seq_id);
  void truncate_sequence(Sequence& seq, int keep_blocks);
  // Lazy block growth: make sure seq owns blocks up to `position`.
  bool ensure_blocks_for(Sequence* seq, int position);
  // Unrecoverable forward failure: retire every running sequence.
  void force_finish_all(const char* reason);

  std::shared_ptr<model::Model> model_;
  std::unique_ptr<KVManager> kv_manager_;
  // Content-hash prefix cache (paged mode only; nullptr when disabled via
  // LLAMA_DISABLE_PREFIX_CACHE=1). Shares whole-block prompt prefixes
  // read-only and skips their prefill.
  std::unique_ptr<PrefixCache> prefix_cache_;
  tensor::Tensor logits_;  // preallocated [max_batch_size, vocab_size] for decode
  std::deque<Sequence> waiting_queue_;  // deque: preempted seqs jump the head
  std::vector<Sequence> running_sequences_;
  std::vector<Sequence> finished_sequences_;
  std::vector<double> batch_reconstruct_times_ms_;  // per decode-step overhead
  int max_batch_size_;
  int max_seq_len_;
  int max_gen_len_;
  int next_seq_id_ = 0;
  int block_size_ = 16;  // paged KV pool block size

  // Adaptive chunked-prefill token budget (see build_batch_rows). Dampened
  // one level per step to avoid oscillation when decode pressure hovers
  // around a threshold.
  int prefill_token_budget_ = 1024;
};

}  // namespace scheduler
