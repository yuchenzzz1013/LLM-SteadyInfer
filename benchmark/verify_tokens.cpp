// ============================================================================
// Token-consistency A/B driver (paged vs continuous KV layout gates)
//
// Runs ONE request (batch 1) through the Scheduler and prints the generated
// token ids + decoded text. Two builds (USE_PAGED_ATTENTION on/off) must
// produce bit-identical output — the paged kernels keep the same split-KV
// structure and fp32 arithmetic as the continuous ones, so any diff means a
// real addressing bug, not a numerics drift.
//
// BF16 校验:加载 HF safetensors 模型目录(config.json + *.safetensors,BF16),
// CUDA 上全链路 BF16(权重 / 激活 / KV Cache / logits);CPU 设备运行时权重在
// 加载期转为 FP32 并走 FP32 内核(兼容后备,用于数值对照)。
//
// 用法: ./build/benchmark/verify_tokens <model-dir> <tokenizer> <max_gen>
//                                            [prompt] [--model-type qwen3]
//   --model-type: qwen3(默认) | qwen2 | llama
// ============================================================================

#include <base/alloc.h>
#include <base/base.h>
#include <glog/logging.h>
#include "model/llama3.h"
#include "model/qwen2.h"
#include "model/qwen3.h"
#include "scheduler/scheduler.h"

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::GLOG_WARNING);
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model-dir> <tokenizer> <max_gen> [prompt]"
                 " [--model-type qwen3|qwen2|llama]\n";
    return -1;
  }
  const std::string model_dir = argv[1];
  const std::string tokenizer = argv[2];
  int max_gen = std::stoi(argv[3]);
  std::string prompt = (argc > 4 && std::string(argv[4]).rfind("--", 0) != 0)
                           ? argv[4] : "What is AI?";

  std::string model_type = "qwen3";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--model-type") model_type = argv[i + 1];
  }

  base::TokenizerType tt;
  std::shared_ptr<model::Model> model;
  if (model_type == "qwen2") {
    tt = base::TokenizerType::kEncodeBpe;
    model = std::make_shared<model::Qwen2Model>(tt, tokenizer, model_dir, false);
  } else if (model_type == "qwen3") {
    tt = base::TokenizerType::kEncodeBpe;
    model = std::make_shared<model::Qwen3Model>(tt, tokenizer, model_dir, false);
  } else if (model_type == "llama" || model_type == "llama3") {
    tt = base::TokenizerType::kEncodeSpe;
    model = std::make_shared<model::LLamaModel>(tt, tokenizer, model_dir, false);
  } else {
    LOG(FATAL) << "未知 --model-type: " << model_type
               << " (支持: qwen3 | qwen2 | llama)";
  }

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

  std::cout << "MODEL: " << model_dir << " (" << model_type << ")\n";
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
