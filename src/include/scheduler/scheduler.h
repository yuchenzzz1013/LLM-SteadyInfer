#pragma once

#include <memory>
#include <queue>
#include <vector>
#include "base/base.h"
#include "kv_manager.h"
#include "model/model.h"
#include "sequence.h"

namespace scheduler {

class Scheduler {
 public:
  // max_seq_len: KV cache capacity (total prompt + generation per slot).
  // max_gen_len: per-request generation token limit (clamped to fit in cache).
  Scheduler(std::shared_ptr<model::Model> model,
            int max_batch_size,
            int max_seq_len,
            int max_gen_len = 0);

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

  // 等待队列长度(在线压测的排队指标)
  int num_waiting() const { return static_cast<int>(waiting_queue_.size()); }

 private:
  void try_admit_sequences();
  std::vector<Sequence*> select_sequences();
  void execute_batch(const std::vector<Sequence*>& batch);
  void update_sequences();

  std::shared_ptr<model::Model> model_;
  std::unique_ptr<KVManager> kv_manager_;
  tensor::Tensor logits_;  // preallocated [max_batch_size, vocab_size] for decode
  std::queue<Sequence> waiting_queue_;
  std::vector<Sequence> running_sequences_;
  std::vector<Sequence> finished_sequences_;
  std::vector<double> batch_reconstruct_times_ms_;  // per decode-step overhead
  int max_batch_size_;
  int max_seq_len_;
  int max_gen_len_;
  int next_seq_id_ = 0;

  // Adaptive chunked-prefill token budget (see execute_batch). Dampened one
  // level per step to avoid oscillation when decode pressure hovers around a
  // threshold.
  int prefill_token_budget_ = 1024;
};

}  // namespace scheduler
