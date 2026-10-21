#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

namespace scheduler {

using TimePoint = std::chrono::steady_clock::time_point;

// Explicit sequence state machine (vLLM-style). is_finished stays as a
// convenience mirror of FINISHED for the metrics/benchmark code.
enum class SeqState {
  WAITING,     // in the waiting queue, no KV blocks yet
  RUNNING,     // admitted, owns KV blocks (may still be prefilling)
  PREEMPTED,   // evicted tail blocks freed; retains its block-table row with
               // the kept prefix blocks, waiting for re-admission (recompute)
  FINISHED,    // done: blocks released, moved to finished_sequences_
};

struct Sequence {
  int id = 0;
  std::vector<int> prompt_tokens;
  std::vector<int> generated_tokens;
  int num_prompt_tokens = 0;
  int num_generated_tokens = 0;
  int kv_slot_id = -1;               // KV Manager slot index / block-table row
  int num_blocks_allocated = 0;      // blocks currently owned by this sequence
  SeqState state = SeqState::WAITING;
  bool is_finished = false;
  bool is_prefill_complete = false;
  int next_prefill_chunk_start = 0;  // chunked prefill progress (tokens prefilled)
  int max_gen_len = 2048;
  // Times this sequence was preempted (recompute-style truncation). Every cut
  // costs the victim progress, so try_preempt_for defers repeat victims and
  // spreads the cost over distinct sequences instead of concentrating it.
  int preempt_count = 0;

  // S4 K-quantization progress. Tokens [0, k_quantized_tokens) have their K
  // in the packed int4 pool; the remaining k_residual_count tokens (always
  // fewer than group_size, and all inside the last, partially filled block)
  // live in the BF16 residual window. The split is derived from the committed
  // KV length and the group size, never incremented blindly, so a truncation
  // that keeps whole blocks rolls it back consistently: every retained block
  // consists of complete groups, hence k_residual_count = 0 after a cut.
  int k_quantized_tokens = 0;
  int k_residual_count = 0;

  // Recompute the S4 counters from the currently committed KV length. Call
  // after a prefill chunk, a decode step, and a truncation. group_size == 0
  // (bf16 cache) leaves the counters at their initial values.
  void sync_kv_quant_state(int group_size) {
    if (group_size <= 0) {
      k_quantized_tokens = 0;
      k_residual_count = 0;
      return;
    }
    const int kv_len = is_prefill_complete ? num_prompt_tokens + num_generated_tokens
                                           : next_prefill_chunk_start;
    k_quantized_tokens = (kv_len / group_size) * group_size;
    k_residual_count = kv_len - k_quantized_tokens;
  }

  // Prefix-cache progress: prefix_hash_chain[b] is the chained hash of prompt
  // block b, and the chain's length is how many leading prompt blocks have
  // already been recorded in the cache. The chain is built incrementally as
  // chunks complete (a block is only hashed/inserted once), and blocks matched
  // from the cache at admission are skipped by the same cursor — their hashes
  // are recomputed from the prompt tokens, which is all the chain needs to keep
  // going (the entries themselves already exist and are deduplicated on insert).
  std::vector<uint64_t> prefix_hash_chain;

  // Per-request latency tracking
  TimePoint arrival_time;            // When the request was submitted (add_request)
  TimePoint admit_time;              // When the request entered running set
  TimePoint first_token_time;        // When first token was generated
  TimePoint last_token_time;         // When last token was generated (for ITL)
  TimePoint finish_time;             // When the request finished
  std::vector<double> token_timestamps_ms;  // ITL between consecutive generated tokens
};

}  // namespace scheduler
