// ============================================================================
// Token-consistency A/B driver (paged vs continuous KV layout gates)
//
// Runs ONE request (batch 1) through the Scheduler and prints the generated
// token ids + decoded text. Two builds (USE_PAGED_ATTENTION on/off) must
// produce bit-identical output — the paged kernels keep the same split-KV
// structure and fp32 arithmetic as the continuous ones, so any diff means a
// real addressing bug, not a numerics drift.
//
// 用法: ./build/benchmark/verify_tokens <checkpoint> <tokenizer> <max_gen> [prompt]
// ============================================================================

#include <base/alloc.h>
#include <base/base.h>
#include <glog/logging.h>
#include "model/qwen2.h"
#include "scheduler/scheduler.h"

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::GLOG_WARNING);
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <checkpoint> <tokenizer> <max_gen> [prompt]\n";
    return -1;
  }
  const char* checkpoint = argv[1];
  const char* tokenizer = argv[2];
  int max_gen = std::stoi(argv[3]);
  std::string prompt = (argc > 4) ? argv[4] : "What is AI?";

  auto model = std::make_shared<model::Qwen2Model>(
      base::TokenizerType::kEncodeBpe, tokenizer, checkpoint, false);
  auto st = model->init(base::DeviceType::kDeviceCUDA);
  if (!st) {
    LOG(FATAL) << "Init failed: " << st.get_err_code();
  }

  // Warmup through the same scheduler path (exercises CUDA-graph capture).
  {
    scheduler::Scheduler warm_sched(model, 1, 128, 8);
    auto t = model->encode("warm up");
    if (t.empty()) t = {1};
    warm_sched.add_request(t);
    while (!warm_sched.all_finished()) warm_sched.step();
  }
  cudaDeviceSynchronize();

  auto tokens = model->encode(prompt);
  if (tokens.empty()) tokens = {1};
  int max_seq_len = static_cast<int>(tokens.size()) + max_gen;
  scheduler::Scheduler sched(model, 1, max_seq_len, max_gen);
  sched.add_request(tokens);
  while (!sched.all_finished()) sched.step();
  cudaDeviceSynchronize();

  std::cout << "PROMPT: " << prompt << "\n";
  for (const auto& seq : sched.get_finished()) {
    std::cout << "SEQ id=" << seq.id
              << " prompt_tokens=" << seq.num_prompt_tokens
              << " generated_tokens=" << seq.num_generated_tokens << "\n";
    std::cout << "PROMPT_IDS:";
    for (int t : seq.prompt_tokens) std::cout << " " << t;
    std::cout << "\nGEN_IDS:";
    for (int t : seq.generated_tokens) std::cout << " " << t;
    std::cout << "\nTEXT: " << model->decode(seq.generated_tokens) << "\n";
  }
  return 0;
}
