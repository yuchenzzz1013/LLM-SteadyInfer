#include <base/alloc.h>
#include <base/base.h>
#include <glog/logging.h>
#include "model/llama3.h"
#include "scheduler/scheduler.h"

#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>

// ========== Statistics ==========
double percentile(const std::vector<double>& data, double p) {
  if (data.empty()) return 0.0;
  std::vector<double> sorted = data;
  std::sort(sorted.begin(), sorted.end());
  size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * sorted.size()) - 1);
  if (idx >= sorted.size()) idx = sorted.size() - 1;
  return sorted[idx];
}

double std_dev(const std::vector<double>& data, double mean) {
  if (data.size() < 2) return 0.0;
  double sum_sq = 0.0;
  for (double v : data) {
    double d = v - mean;
    sum_sq += d * d;
  }
  return std::sqrt(sum_sq / (data.size() - 1));
}

struct BenchmarkResult {
  double p99_ttft_ms = 0;
  double tpot_mean_ms = 0;
  double tpot_std_ms = 0;
  double tpot_jitter = 0;        // std_dev / mean
  double tps = 0;                // tokens per second
  double qps = 0;                // queries per second
  double kv_fragmentation = 0;   // avg(1 - used/max_seq_len) across finished seqs
  double avg_batch_reconstruct_ms = 0;
  double p99_batch_reconstruct_ms = 0;
};

// ========== Continuous Batching Benchmark ==========
BenchmarkResult benchmark_batch(std::shared_ptr<model::Model> model,
                                 const std::vector<std::string>& prompts,
                                 int max_gen_len, int max_batch_size, int num_requests) {
  using Clock = std::chrono::steady_clock;
  // Compute KV cache slot size from actual prompt lengths + max_gen_len
  // instead of the model's full seq_len (e.g. 32768) to avoid >95% fragmentation.
  int max_prompt_len = 0;
  for (int i = 0; i < num_requests; ++i) {
    const std::string& prompt = prompts[i % prompts.size()];
    auto tokens = model->encode(prompt);
    if (!tokens.empty()) max_prompt_len = std::max(max_prompt_len, (int)tokens.size());
  }
  int max_total_seq_len = max_prompt_len + max_gen_len;
  LOG(INFO) << "[BENCH] KV slot size: max_prompt=" << max_prompt_len
            << " max_gen=" << max_gen_len << " => " << max_total_seq_len;

  // Shrink model's internal KV cache (used during prefill) from seq_len
  // (e.g. 32768 * 768 MB) down to max_prompt_len (e.g. ~100 * few MB).
  model->resize_internal_kv_cache(max_prompt_len);

  scheduler::Scheduler sched(model, max_batch_size, max_total_seq_len, max_gen_len);

  for (int i = 0; i < num_requests; ++i) {
    const std::string& prompt = prompts[i % prompts.size()];
    auto tokens = model->encode(prompt);
    if (tokens.empty()) tokens = {1};
    sched.add_request(tokens);
  }

  LOG(INFO) << "[BENCH] Starting benchmark: " << num_requests
            << " requests, batch=" << max_batch_size << " max_gen=" << max_gen_len;

  auto start = Clock::now();
  while (!sched.all_finished()) { sched.step(); }
  cudaDeviceSynchronize();  // ensure all GPU work is done before timing stop
  auto end = Clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
  double total_s = total_ms / 1000.0;

  const auto& finished = sched.get_finished();

  // Collect per-request TTFT and TPOT
  std::vector<double> ttfts, tpots;
  int total_prompt = 0, total_gen = 0;

  for (const auto& seq : finished) {
    if (seq.num_generated_tokens <= 0) continue;
    double admit_ms = std::chrono::duration<double, std::milli>(seq.admit_time - start).count();
    double first_ms = std::chrono::duration<double, std::milli>(seq.first_token_time - start).count();
    double finish_ms = std::chrono::duration<double, std::milli>(seq.finish_time - start).count();

    double ttft = first_ms - admit_ms;
    double e2e = finish_ms - admit_ms;
    ttfts.push_back(ttft);

    if (seq.num_generated_tokens > 1) {
      double tpot = (e2e - ttft) / (seq.num_generated_tokens - 1);
      tpots.push_back(tpot);
    }

    total_prompt += seq.num_prompt_tokens;
    total_gen += seq.num_generated_tokens;
  }

  int total_tokens = total_prompt + total_gen;
  int valid_requests = static_cast<int>(ttfts.size());

  BenchmarkResult result;

  // P99 TTFT
  result.p99_ttft_ms = percentile(ttfts, 99);

  // TPOT Jitter Rate
  if (!tpots.empty()) {
    double sum = std::accumulate(tpots.begin(), tpots.end(), 0.0);
    result.tpot_mean_ms = sum / tpots.size();
    result.tpot_std_ms = std_dev(tpots, result.tpot_mean_ms);
    result.tpot_jitter = (result.tpot_mean_ms > 0)
        ? result.tpot_std_ms / result.tpot_mean_ms : 0;
  }

  // Stable Throughput
  result.tps = (total_s > 0) ? total_tokens / total_s : 0;
  result.qps = (total_s > 0) ? valid_requests / total_s : 0;

  // KV Cache Fragmentation Rate
  int max_seq_len = sched.get_max_kv_seq_len();
  if (!finished.empty() && max_seq_len > 0) {
    double waste_sum = 0;
    int count = 0;
    for (const auto& seq : finished) {
      int used = seq.num_prompt_tokens + seq.num_generated_tokens;
      if (used > 0) {
        waste_sum += 1.0 - static_cast<double>(used) / max_seq_len;
        count++;
      }
    }
    result.kv_fragmentation = (count > 0) ? waste_sum / count : 0;
  }

  // Batch Reconstruction Latency
  const auto& recon_times = sched.get_batch_reconstruct_times_ms();
  if (!recon_times.empty()) {
    double sum = std::accumulate(recon_times.begin(), recon_times.end(), 0.0);
    result.avg_batch_reconstruct_ms = sum / recon_times.size();
    result.p99_batch_reconstruct_ms = percentile(recon_times, 99);
  }

  LOG(INFO) << "[BENCH] Benchmark finished: " << total_ms << " ms, "
            << total_tokens << " tokens, " << valid_requests << " valid requests";
  return result;
}

// ========== CSV Output ==========
void write_result_csv(const std::string& filename,
                      const std::vector<BenchmarkResult>& results) {
  std::ofstream file(filename);
  if (!file.is_open()) { LOG(ERROR) << "Cannot open: " << filename; return; }
  file << "run_id,p99_ttft_ms,tpot_mean_ms,tpot_jitter,tps,qps,"
       << "kv_cache_fragmentation,avg_batch_reconstruct_ms,p99_batch_reconstruct_ms\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    file << i << ","
         << std::fixed << std::setprecision(3)
         << r.p99_ttft_ms << ","
         << r.tpot_mean_ms << "," << r.tpot_jitter << ","
         << r.tps << "," << r.qps << ","
         << r.kv_fragmentation << ","
         << r.avg_batch_reconstruct_ms << "," << r.p99_batch_reconstruct_ms << "\n";
  }
  file.close();
  LOG(INFO) << "Results CSV saved to " << filename;
}

// ========== Main ==========
int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::GLOG_WARNING);
  if (argc < 7) {
    LOG(INFO) << "Usage: " << argv[0]
              << " <checkpoint> <tokenizer> <num_requests> <max_batch> <max_gen>"
              << " <output_csv> [prompts_comma] [iterations]";
    return -1;
  }

  const char* checkpoint = argv[1];
  const char* tokenizer = argv[2];

  // Accept both old (--batch) and new (direct) formats
  int arg_idx = 3;
  bool has_batch_flag = (std::string(argv[3]) == "--batch");
  if (has_batch_flag) {
    LOG(INFO) << "Detected legacy --batch flag, adapting...";
    arg_idx = 4;
  }

  if (argc < arg_idx + 3) {
    LOG(ERROR) << "Insufficient arguments. Need: num_requests max_batch max_gen output_csv";
    return -1;
  }

  int num_requests = std::stoi(argv[arg_idx]);
  int max_batch = std::stoi(argv[arg_idx + 1]);
  int max_gen = std::stoi(argv[arg_idx + 2]);
  std::string output_csv = argv[arg_idx + 3];

  // Parse prompts
  int prompts_idx = arg_idx + 4;
  std::vector<std::string> prompts;
  if (argc > prompts_idx) {
    std::stringstream ss(argv[prompts_idx]); std::string item;
    while (std::getline(ss, item, ',')) {
      size_t s = item.find_first_not_of(" \t"), e = item.find_last_not_of(" \t");
      if (s != std::string::npos) prompts.push_back(item.substr(s, e - s + 1));
    }
  }
  if (prompts.empty()) prompts.push_back("What is AI?");

  int iterations = (argc > prompts_idx + 1) ? std::stoi(argv[prompts_idx + 1]) : 1;

  LOG(INFO) << "Checkpoint: " << checkpoint << " Tokenizer: " << tokenizer;
  LOG(INFO) << "Config: " << num_requests << " requests, batch=" << max_batch
            << " max_gen=" << max_gen << " iterations=" << iterations;

  auto model = std::make_shared<model::LLamaModel>(
      base::TokenizerType::kEncodeSpe, tokenizer, checkpoint, false);
  auto st = model->init(base::DeviceType::kDeviceCUDA);
  if (!st) LOG(FATAL) << "Init failed: " << st.get_err_code();
  LOG(INFO) << "Model ready. kv_dim=" << model->kv_dim()
            << " layers=" << model->layer_num();

  // Warmup
  LOG(INFO) << "Warming up...";
  {
    scheduler::Scheduler warm_sched(model, max_batch, 128, std::min(max_gen, 8));
    auto tokens = model->encode("warm up");
    if (tokens.empty()) tokens = {1};
    warm_sched.add_request(tokens);
    warm_sched.add_request(tokens);
    while (!warm_sched.all_finished()) { warm_sched.step(); }
  }
  cudaDeviceSynchronize();
  base::CUDADeviceAllocatorFactory::get_instance()->free_idle();

  // Benchmark runs
  std::vector<BenchmarkResult> all_results;
  all_results.reserve(iterations);

  for (int i = 0; i < iterations; ++i) {
    LOG(INFO) << "=== Run " << (i + 1) << "/" << iterations << " ===";
    all_results.push_back(
        benchmark_batch(model, prompts, max_gen, max_batch, num_requests));
  }

  // CSV output
  write_result_csv(output_csv, all_results);

  // Console summary
  LOG(INFO) << "========== Benchmark Results ==========";
  for (size_t i = 0; i < all_results.size(); ++i) {
    const auto& r = all_results[i];
    printf("\nRun %zu:\n", i);
    printf("  P99 TTFT:                  %10.3f ms\n", r.p99_ttft_ms);
    printf("  TPOT Jitter Rate:          %10.3f  (mean=%.3f std=%.3f ms)\n",
           r.tpot_jitter, r.tpot_mean_ms, r.tpot_std_ms);
    printf("  Stable Throughput:         %10.1f TPS  /  %.1f QPS\n", r.tps, r.qps);
    printf("  KV Cache Fragmentation:    %10.1f%%\n", r.kv_fragmentation * 100.0);
    printf("  Batch Reconstruct Latency: avg=%.3f ms  p99=%.3f ms\n",
           r.avg_batch_reconstruct_ms, r.p99_batch_reconstruct_ms);
  }

  LOG(INFO) << "Benchmark finished.";
  return 0;
}
