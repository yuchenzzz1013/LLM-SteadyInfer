#pragma once

#include <chrono>
#include <vector>

namespace scheduler {

using TimePoint = std::chrono::steady_clock::time_point;

struct Sequence {
  int id = 0;
  std::vector<int> prompt_tokens;
  std::vector<int> generated_tokens;
  int num_prompt_tokens = 0;
  int num_generated_tokens = 0;
  int kv_slot_id = -1;               // KV Manager slot index
  bool is_finished = false;
  bool is_prefill_complete = false;
  int next_prefill_chunk_start = 0;  // chunked prefill progress
  int max_gen_len = 2048;

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
