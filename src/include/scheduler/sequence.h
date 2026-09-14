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
  bool first_token_recorded = false;
  std::vector<double> token_timestamps_ms;  // ITL between consecutive generated tokens

  bool is_active() const { return !is_finished; }
  int total_tokens() const { return num_prompt_tokens + num_generated_tokens; }
};

}  // namespace scheduler
