// ============================================================================
// 离线批推理 Benchmark(参考 vLLM benchmark_throughput 的离线压测思路)
//
// 读取 ShareGPT_prompts.jsonl(每行 {"id":..., "prompt":...}),通过 LLM-SteadyInfer
// 的 scheduler 做 continuous batching 离线推理,输出指标:
//   - output_tps                 每秒输出 Token 数(核心吞吐)
//   - total_tps / qps            总 Token 吞吐 / 每秒完成请求数
//   - kv_cache_fragmentation     KV cache 碎片率(全局时间加权 + 每请求视角)
//   - avg/p99_batch_reconstruct_ms  每 decode step 的 batch 重建开销
//   - throughput_efficiency      吞吐效率 = 实际输出 TPS / 理论峰值 TPS
//                                (= decode 步骤平均 batch 占用率)
//   - hardware_utilization       硬件利用率:MFU(模型算力利用率)+ NVML 采样的
//                                GPU SM 利用率 / 显存利用率
//   - TTFT / TPOT / ITL / e2e    延迟分布(avg / p50 / p99)
//
// 用法:
//   ./build/benchmark/offline_batch_benchmark --dataset ShareGPT_prompts.jsonl \
//       --checkpoint Qwen2.5-0.5B.bin \
//       --tokenizer Qwen/Qwen2.5-0.5B/tokenizer.json \
//       --num-requests 512 --max-batch 32 --max-gen 256
// ============================================================================

#include <base/base.h>
#include <glog/logging.h>
#include "model/llama3.h"
#include "model/qwen2.h"
#include "model/qwen3.h"
#include "scheduler/scheduler.h"

#include "bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using namespace bench;  // get_arg/has_arg/resolve_default/load_dataset/percentile/mean/GpuUtilSampler/MFU

// ============================== 参数解析 ==============================

struct Args {
  std::string model_type = "qwen2";  // qwen2 | qwen3 | llama (引擎内三模型均已编译进 lib)
  std::string dataset = "ShareGPT_prompts.jsonl";
  std::string checkpoint = "Qwen2.5-0.5B.bin";
  std::string tokenizer = "Qwen/Qwen2.5-0.5B/tokenizer.json";
  std::string model_config = "Qwen/Qwen2.5-0.5B/config.json";
  std::string output_csv = "results/metrics.csv";
  int num_requests = 512;  // 0 = 使用全部样本
  int max_batch = 32;
  int max_gen = 256;
  int iterations = 1;
  int seed = 42;
};

static Args parse_args(int argc, char** argv) {
  Args a;
  // 兼容 benchmark 的位置参数风格(第一个参数不以 "--" 开头时生效):
  //   <checkpoint> <tokenizer> <num_requests> <max_batch> <max_gen>
  //   [output_csv] [iterations]
  bool positional = argc > 1 && std::string(argv[1]).rfind("--", 0) != 0;
  if (positional) {
    if (argc > 1) a.checkpoint = argv[1];
    if (argc > 2) a.tokenizer = argv[2];
    if (argc > 3) a.num_requests = std::stoi(argv[3]);
    if (argc > 4) a.max_batch = std::stoi(argv[4]);
    if (argc > 5) a.max_gen = std::stoi(argv[5]);
    if (argc > 6) a.output_csv = argv[6];
    if (argc > 7) a.iterations = std::stoi(argv[7]);
  }

  // --flag 覆盖(两种模式均生效;--flag 优先于位置参数)
  if (has_arg(argc, argv, "--model-type"))
    a.model_type = get_arg(argc, argv, "--model-type");
  if (has_arg(argc, argv, "--dataset"))
    a.dataset = get_arg(argc, argv, "--dataset");
  if (has_arg(argc, argv, "--checkpoint"))
    a.checkpoint = get_arg(argc, argv, "--checkpoint");
  if (has_arg(argc, argv, "--tokenizer"))
    a.tokenizer = get_arg(argc, argv, "--tokenizer");
  if (has_arg(argc, argv, "--model-config"))
    a.model_config = get_arg(argc, argv, "--model-config");
  if (has_arg(argc, argv, "--output-csv"))
    a.output_csv = get_arg(argc, argv, "--output-csv");
  if (has_arg(argc, argv, "--num-requests"))
    a.num_requests = std::stoi(get_arg(argc, argv, "--num-requests"));
  if (has_arg(argc, argv, "--max-batch"))
    a.max_batch = std::stoi(get_arg(argc, argv, "--max-batch"));
  if (has_arg(argc, argv, "--max-gen"))
    a.max_gen = std::stoi(get_arg(argc, argv, "--max-gen"));
  if (has_arg(argc, argv, "--iterations"))
    a.iterations = std::stoi(get_arg(argc, argv, "--iterations"));
  if (has_arg(argc, argv, "--seed"))
    a.seed = std::stoi(get_arg(argc, argv, "--seed"));
  return a;
}

// 默认路径解析(见 bench_common.h 的 resolve_default):
// 按 CWD 解释失败后,相对可执行文件推断出的项目根目录
// (build/benchmark/offline_batch_benchmark 与
//  benchmark/build/offline_batch_benchmark 两种布局均适用)
static void print_usage(const char* prog) {
  std::cout
      << "用法 1(与 benchmark 相同的调用方式,位置参数):\n"
      << "  " << prog << " <checkpoint> <tokenizer> <num_requests> <max_batch>"
      << " <max_gen> [output_csv] [iterations]\n"
      << "  例: ./build/benchmark/offline_batch_benchmark"
      << " Qwen2.5-0.5B.bin Qwen/Qwen2.5-0.5B/tokenizer.json"
      << " 512 32 256 results/metrics.csv\n"
      << "  选其他模型: 在位置参数后追加 --model-type llama|qwen3\n\n"
      << "用法 2(--flag 风格,默认值见括号):\n"
      << "  " << prog << " [选项]\n"
      << "  --model-type <type>    模型类型: qwen2(默认) | qwen3 | llama\n"
      << "                         (llama 等价 llama3;三模型均已编译进引擎,运行时选择)\n"
      << "  --dataset <path>       ShareGPT_prompts.jsonl 路径(默认自动探测)\n"
      << "  --checkpoint <path>    模型权重 .bin 路径(默认 Qwen2.5-0.5B.bin)\n"
      << "  --tokenizer <path>     tokenizer 路径(qwen2/qwen3 为 tokenizer.json,\n"
      << "                         llama 为 sentencepiece 模型;默认 Qwen/Qwen2.5-0.5B/tokenizer.json)\n"
      << "  --model-config <path>  config.json 路径,用于计算 MFU(默认 Qwen/Qwen2.5-0.5B/config.json)\n"
      << "  --num-requests <N>     压测请求数,0 表示全部(默认 512)\n"
      << "  --max-batch <N>        最大 batch size(默认 32)\n"
      << "  --max-gen <N>          每个请求最大生成 token 数(默认 256)\n"
      << "  --iterations <N>       重复轮数(默认 1)\n"
      << "  --seed <N>             请求抽样随机种子(默认 42)\n"
      << "  --output-csv <path>    结果 CSV 路径(默认 results/metrics.csv)\n";
}

// 数据集加载 / 统计工具 / NVML 采样 / MFU 估算均来自 bench_common.h

// ============================== 单轮 Benchmark ==============================

struct RunMetrics {
  int run_id = 0;
  int num_requests = 0, completed = 0, rejected = 0;
  long long prompt_tokens = 0, output_tokens = 0, total_tokens = 0;
  double wall_time_s = 0;

  // 吞吐
  double output_tps = 0;   // 每秒输出 Token 数
  double total_tps = 0;    // 每秒总 Token 数(prefill + decode)
  double qps = 0;

  // 延迟
  double ttft_avg_ms = 0, ttft_p50_ms = 0, ttft_p99_ms = 0;
  double tpot_avg_ms = 0, tpot_p50_ms = 0, tpot_p99_ms = 0;
  double itl_avg_ms = 0, itl_p99_ms = 0;
  double e2e_avg_ms = 0, e2e_p99_ms = 0;

  // KV cache
  double kv_cache_util_global = 0;   // 时间加权:已用 token 槽 / 已分配容量
  double kv_cache_frag_global = 0;   // 1 - util_global
  double kv_cache_frag_per_seq = 0;  // 每请求视角
  double avg_busy_kv_slots = 0, peak_busy_kv_slots = 0;

  // batch 重建开销
  double avg_batch_reconstruct_ms = 0, p99_batch_reconstruct_ms = 0;

  // 吞吐效率
  double avg_batch_size = 0;         // 平均运行中序列数
  double avg_decode_batch = 0;       // decode 步骤平均 batch
  double throughput_efficiency = 0;  // 实际输出 TPS / 理论峰值 TPS
  double theoretical_peak_tps = 0;   // max_batch 常满时的理论输出 TPS

  // 硬件利用率
  double peak_tflops = 0, mfu = 0;
  double gpu_sm_util_pct = 0, gpu_mem_util_pct = 0, gpu_mem_used_mb = 0;
};

static RunMetrics run_once(const std::shared_ptr<model::Model>& model,
                           const std::vector<std::vector<int>>& prompt_tokens,
                           int run_id, int max_batch, int max_gen_len,
                           double flops_per_tok, double peak_tflops) {
  using namespace scheduler;
  RunMetrics m;
  m.run_id = run_id;
  m.peak_tflops = peak_tflops;

  // 根据实际 prompt 长度确定 KV slot 大小,避免按模型全长分配导致碎片率虚高
  int max_prompt_len = 0;
  long long prompt_len_sum = 0;
  for (const auto& t : prompt_tokens) {
    max_prompt_len = std::max(max_prompt_len, static_cast<int>(t.size()));
    prompt_len_sum += t.size();
  }
  int max_total_seq_len =
      std::min(max_prompt_len + max_gen_len, static_cast<int>(model->seq_len()));
  // 收缩模型内部 KV cache —— 必须在任何 decode CUDA graph 捕获之前调用:
  // 捕获会把当时的 buffer 指针烘焙进图,捕获后再重分配会导致 replay 时
  // 访问已释放内存(illegal memory access)。
  model->resize_internal_kv_cache(max_prompt_len);

  // 分页块大小按工作负载平均 prompt 长度自适应(短文本 8/长文本 32/默认 16)。
  const long long avg_prompt_len =
      prompt_tokens.empty() ? 0 : prompt_len_sum / static_cast<long long>(prompt_tokens.size());
  Scheduler sched(model, max_batch, max_total_seq_len, max_gen_len,
                  Scheduler::resolve_block_size(avg_prompt_len));

  // ---- 预热:与正式压测共用同一 Scheduler ----
  // decode CUDA graph 在首次 decode 时捕获并烘焙当时 KV/logits 的设备指针;
  // 引擎已在 replay 前校验这些指针与形状,跨 Scheduler 时自动销毁旧图重捕获
  // (见 src/source/model/model.cpp 的 decode_step)。预热仍与压测共用同一
  // Scheduler:避免多余的一次重捕获,且预热请求在统计中跳过。
  // 注意:这里不能调用 free_idle(),它可能释放图引用的池内缓冲。
  std::set<int> warm_ids;
  {
    auto t = model->encode("warm up");
    if (t.empty()) t = {1};
    std::vector<int> warm_tokens(t.begin(), t.end());
    for (int i = 0; i < 2; ++i) {
      int id = sched.add_request(warm_tokens);
      if (id >= 0) warm_ids.insert(id);
    }
    while (!sched.all_finished()) {
      sched.step();
    }
    cudaDeviceSynchronize();
  }

  for (const auto& t : prompt_tokens) {
    if (sched.add_request(t) < 0) m.rejected++;
  }
  m.num_requests = static_cast<int>(prompt_tokens.size());

  GpuUtilSampler sampler;
  sampler.start();

  // 主循环:每步采样一次 KV 占用快照(纯 host 端读取,开销可忽略)
  auto start = Clock::now();
  long long used_cap_sum = 0, alloc_cap_sum = 0;
  long long busy_slots_sum = 0, running_sum = 0;
  int kv_samples = 0, peak_busy = 0;
  while (!sched.all_finished()) {
    const auto& running = sched.get_running();
    int busy = sched.get_busy_kv_slots();
    if (busy > 0) {
      long long used = 0;
      for (const auto& s : running) {
        used += static_cast<long long>(s.num_prompt_tokens) + s.num_generated_tokens;
      }
      used_cap_sum += used;
      {
        // Paged: truthful reservation = allocated blocks x block_size (lazy
        // growth); continuous layout falls back to busy slots x capacity.
        long long alloc_tokens = sched.get_allocated_kv_tokens();
        alloc_cap_sum += alloc_tokens > 0
                             ? alloc_tokens
                             : static_cast<long long>(busy) * max_total_seq_len;
      }
      busy_slots_sum += busy;
      running_sum += running.size();
      peak_busy = std::max(peak_busy, busy);
      kv_samples++;
    }
    sched.step();
  }
  cudaDeviceSynchronize();
  auto end = Clock::now();
  sampler.stop();

  double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
  double total_s = total_ms / 1000.0;
  m.wall_time_s = total_s;

  // ---- 逐请求统计 ----
  const auto& finished = sched.get_finished();
  std::vector<double> ttfts, tpots, e2es, itls, per_seq_frag;
  for (const auto& seq : finished) {
    if (warm_ids.count(seq.id)) continue;  // 跳过预热请求
    if (seq.num_generated_tokens <= 0) continue;
    double admit_ms =
        std::chrono::duration<double, std::milli>(seq.admit_time - start).count();
    double first_ms =
        std::chrono::duration<double, std::milli>(seq.first_token_time - start).count();
    double finish_ms =
        std::chrono::duration<double, std::milli>(seq.finish_time - start).count();

    double ttft = first_ms - admit_ms;
    double e2e = finish_ms - admit_ms;
    ttfts.push_back(ttft);
    e2es.push_back(e2e);
    if (seq.num_generated_tokens > 1) {
      tpots.push_back((e2e - ttft) / (seq.num_generated_tokens - 1));
    }
    for (double itl : seq.token_timestamps_ms) itls.push_back(itl);

    int used = seq.num_prompt_tokens + seq.num_generated_tokens;
    per_seq_frag.push_back(1.0 - static_cast<double>(used) / max_total_seq_len);

    m.prompt_tokens += seq.num_prompt_tokens;
    m.output_tokens += seq.num_generated_tokens;
  }
  m.completed = static_cast<int>(ttfts.size());
  m.total_tokens = m.prompt_tokens + m.output_tokens;

  // ---- 吞吐 ----
  m.output_tps = total_s > 0 ? m.output_tokens / total_s : 0;
  m.total_tps = total_s > 0 ? m.total_tokens / total_s : 0;
  m.qps = total_s > 0 ? m.completed / total_s : 0;

  // ---- 延迟 ----
  m.ttft_avg_ms = mean(ttfts);
  m.ttft_p50_ms = percentile(ttfts, 50);
  m.ttft_p99_ms = percentile(ttfts, 99);
  m.tpot_avg_ms = mean(tpots);
  m.tpot_p50_ms = percentile(tpots, 50);
  m.tpot_p99_ms = percentile(tpots, 99);
  m.itl_avg_ms = mean(itls);
  m.itl_p99_ms = percentile(itls, 99);
  m.e2e_avg_ms = mean(e2es);
  m.e2e_p99_ms = percentile(e2es, 99);

  // ---- KV cache 碎片率 ----
  // 全局(时间加权):已分配容量中真正存有 token 的比例
  m.kv_cache_util_global =
      alloc_cap_sum > 0 ? static_cast<double>(used_cap_sum) / alloc_cap_sum : 0;
  m.kv_cache_frag_global = 1.0 - m.kv_cache_util_global;
  // 每请求视角:单个请求对其 KV slot 序列维度的浪费
  m.kv_cache_frag_per_seq = mean(per_seq_frag);
  m.avg_busy_kv_slots =
      kv_samples > 0 ? static_cast<double>(busy_slots_sum) / kv_samples : 0;
  m.peak_busy_kv_slots = peak_busy;

  // ---- batch 重建开销(调度器在每次 decode step 记录的 host 端 tensor 组装耗时)----
  const auto& recon = sched.get_batch_reconstruct_times_ms();
  if (!recon.empty()) {
    m.avg_batch_reconstruct_ms = mean(recon);
    m.p99_batch_reconstruct_ms = percentile(recon, 99);
  }

  // ---- 吞吐效率 ----
  // decode 步骤数 = recon 记录数(每条记录对应一次 decode step)。
  // 每个 decode step 为 batch 内每条序列生成 1 个 token,因此
  //   平均 decode batch = output_tokens / decode 步数
  //   理论峰值 TPS     = max_batch * decode 步数 / 总时长(batch 常满)
  //   吞吐效率          = 实际输出 TPS / 理论峰值 TPS = 平均 decode batch 占用率
  long long n_decode = static_cast<long long>(recon.size());
  m.avg_decode_batch = n_decode > 0 ? static_cast<double>(m.output_tokens) / n_decode : 0;
  m.theoretical_peak_tps =
      total_s > 0 ? max_batch * static_cast<double>(n_decode) / total_s : 0;
  m.throughput_efficiency =
      m.theoretical_peak_tps > 0 ? m.output_tps / m.theoretical_peak_tps : 0;
  m.avg_batch_size =
      kv_samples > 0 ? static_cast<double>(running_sum) / kv_samples : 0;

  // ---- 硬件利用率 ----
  double flops = flops_per_tok * static_cast<double>(m.total_tokens);
  m.mfu = (peak_tflops > 0 && total_s > 0)
              ? flops / (peak_tflops * 1e12 * total_s) : 0;
  m.gpu_sm_util_pct = sampler.sm_util_pct();
  m.gpu_mem_util_pct = sampler.mem_util_pct();
  m.gpu_mem_used_mb = sampler.mem_used_mb();

  return m;
}

// ============================== CSV 输出 ==============================

static void write_csv(const std::string& path,
                      const std::vector<RunMetrics>& results) {
  std::ofstream f(path);
  if (!f.is_open()) {
    LOG(ERROR) << "无法打开输出文件: " << path;
    return;
  }
  f << "run_id,num_requests,completed,rejected,prompt_tokens,output_tokens,"
       "total_tokens,wall_time_s,output_tps,total_tps,qps,"
       "ttft_avg_ms,ttft_p50_ms,ttft_p99_ms,"
       "tpot_avg_ms,tpot_p50_ms,tpot_p99_ms,itl_avg_ms,itl_p99_ms,"
       "e2e_avg_ms,e2e_p99_ms,"
       "kv_cache_util_global,kv_cache_frag_global,kv_cache_frag_per_seq,"
       "avg_busy_kv_slots,peak_busy_kv_slots,"
       "avg_batch_reconstruct_ms,p99_batch_reconstruct_ms,"
       "avg_batch_size,avg_decode_batch,throughput_efficiency,theoretical_peak_tps,"
       "mfu,peak_fp32_tflops,gpu_sm_util_pct,gpu_mem_util_pct,gpu_mem_used_mb\n";
  f << std::fixed << std::setprecision(6);
  for (const auto& r : results) {
    f << r.run_id << "," << r.num_requests << "," << r.completed << ","
      << r.rejected << "," << r.prompt_tokens << "," << r.output_tokens << ","
      << r.total_tokens << "," << r.wall_time_s << "," << r.output_tps << ","
      << r.total_tps << "," << r.qps << "," << r.ttft_avg_ms << ","
      << r.ttft_p50_ms << "," << r.ttft_p99_ms << "," << r.tpot_avg_ms << ","
      << r.tpot_p50_ms << "," << r.tpot_p99_ms << "," << r.itl_avg_ms << ","
      << r.itl_p99_ms << "," << r.e2e_avg_ms << "," << r.e2e_p99_ms << ","
      << r.kv_cache_util_global << "," << r.kv_cache_frag_global << ","
      << r.kv_cache_frag_per_seq << "," << r.avg_busy_kv_slots << ","
      << r.peak_busy_kv_slots << "," << r.avg_batch_reconstruct_ms << ","
      << r.p99_batch_reconstruct_ms << "," << r.avg_batch_size << ","
      << r.avg_decode_batch << "," << r.throughput_efficiency << ","
      << r.theoretical_peak_tps << "," << r.mfu << "," << r.peak_tflops << ","
      << r.gpu_sm_util_pct << "," << r.gpu_mem_util_pct << ","
      << r.gpu_mem_used_mb << "\n";
  }
  f.close();
}

// ============================== 控制台报告 ==============================

static void print_report(const RunMetrics& r) {
  auto pct = [](double v) { return v * 100.0; };
  std::cout << "\n==================== Run " << r.run_id
            << " 离线批推理结果 ====================\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "请求: " << r.num_requests << " 条, 完成 " << r.completed
            << " 条, 拒绝 " << r.rejected << " 条\n";
  std::cout << "Token: prompt=" << r.prompt_tokens
            << " 输出=" << r.output_tokens
            << " 总计=" << r.total_tokens
            << "  耗时=" << r.wall_time_s << " s\n\n";

  std::cout << "--- 吞吐 ---\n";
  std::cout << std::setprecision(2);
  std::cout << "  output_tps (每秒输出Token数): " << r.output_tps << " tok/s\n";
  std::cout << "  total_tps (含prefill)        : " << r.total_tps << " tok/s\n";
  std::cout << "  qps (每秒完成请求数)         : " << r.qps << " req/s\n";
  std::cout << "  throughput_efficiency (吞吐效率): " << std::setprecision(2)
            << pct(r.throughput_efficiency) << "%"
            << "  (decode 平均 batch=" << r.avg_decode_batch
            << ", 运行平均 batch=" << r.avg_batch_size << ")\n";
  std::cout << "  theoretical_peak_tps         : " << r.theoretical_peak_tps
            << " tok/s (batch 常满)\n\n";

  std::cout << "--- KV cache ---\n";
  std::cout << "  kv_cache_fragmentation (全局时间加权): " << std::setprecision(2)
            << pct(r.kv_cache_frag_global) << "%"
            << "  (利用率 " << pct(r.kv_cache_util_global) << "%)\n";
  std::cout << "  kv_cache_fragmentation (每请求均值): " << std::setprecision(2)
            << pct(r.kv_cache_frag_per_seq) << "%\n";
  std::cout << "  平均/峰值占用 KV slot: " << r.avg_busy_kv_slots << " / "
            << r.peak_busy_kv_slots << "\n\n";

  std::cout << "--- batch 重建 ---\n";
  std::cout << std::setprecision(3);
  std::cout << "  avg_batch_reconstruct_ms: " << r.avg_batch_reconstruct_ms << " ms\n";
  std::cout << "  p99_batch_reconstruct_ms: " << r.p99_batch_reconstruct_ms << " ms\n\n";

  std::cout << "--- 硬件利用率 ---\n";
  std::cout << std::setprecision(2);
  std::cout << "  MFU (模型算力利用率): " << pct(r.mfu)
            << "%  (峰值 " << r.peak_tflops << " TFLOPS FP32)\n";
  std::cout << "  GPU SM 利用率 (NVML 采样均值): " << r.gpu_sm_util_pct << "%\n";
  std::cout << "  GPU 显存利用率 (NVML 采样均值): " << r.gpu_mem_util_pct
            << "%  (已用 " << r.gpu_mem_used_mb << " MB)\n\n";

  std::cout << "--- 延迟 ---\n";
  std::cout << std::setprecision(2);
  std::cout << "  TTFT: avg=" << r.ttft_avg_ms << " ms  p50=" << r.ttft_p50_ms
            << " ms  p99=" << r.ttft_p99_ms << " ms\n";
  std::cout << "  TPOT: avg=" << r.tpot_avg_ms << " ms  p50=" << r.tpot_p50_ms
            << " ms  p99=" << r.tpot_p99_ms << " ms\n";
  std::cout << "  ITL : avg=" << r.itl_avg_ms << " ms  p99=" << r.itl_p99_ms
            << " ms\n";
  std::cout << "  E2E : avg=" << r.e2e_avg_ms << " ms  p99=" << r.e2e_p99_ms
            << " ms\n";
  std::cout << "====================================================\n";
}

// ============================== Main ==============================

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::GLOG_WARNING);

  if (has_arg(argc, argv, "--help") || has_arg(argc, argv, "-h")) {
    print_usage(argv[0]);
    return 0;
  }
  Args args = parse_args(argc, argv);

  // ---- 从可执行文件路径推断项目根目录,用于解析默认相对路径 ----
  // 两种常见布局: <root>/build/benchmark/offline_batch_benchmark
  //             <root>/benchmark/build/offline_batch_benchmark
  // 上溯两级即为根目录;推断失败则保持默认值原样(CWD 相对)。
  std::string exe_root;
  {
    std::filesystem::path p(argv[0]);
    if (p.has_parent_path()) {
      std::error_code ec;
      auto root = std::filesystem::canonical(p.parent_path() / "../..", ec);
      if (ec) {
        ec.clear();
        root = std::filesystem::canonical(p.parent_path(), ec);
      }
      if (!ec) exe_root = root.string();
    }
  }
  args.dataset = resolve_default(args.dataset, exe_root);
  args.checkpoint = resolve_default(args.checkpoint, exe_root);
  args.tokenizer = resolve_default(args.tokenizer, exe_root);
  args.model_config = resolve_default(args.model_config, exe_root);

  // ---- 数据集 ----
  std::vector<Request> dataset = load_dataset(args.dataset);
  if (dataset.empty()) {
    LOG(ERROR) << "数据集为空: " << args.dataset;
    return -1;
  }

  // 按种子抽样(仅当请求数小于数据集规模时)
  if (args.num_requests > 0 && args.num_requests < static_cast<int>(dataset.size())) {
    std::mt19937 rng(args.seed);
    std::shuffle(dataset.begin(), dataset.end(), rng);
    dataset.resize(args.num_requests);
  }

  // ---- 模型加载:按 --model-type 运行时选择(三个模型均已编译进引擎库) ----
  auto load_model = [&args]() -> std::shared_ptr<model::Model> {
    std::string mt = args.model_type;
    base::TokenizerType tt;
    std::shared_ptr<model::Model> model;
    if (mt == "qwen2") {
      tt = base::TokenizerType::kEncodeBpe;
      model = std::make_shared<model::Qwen2Model>(tt, args.tokenizer,
                                                  args.checkpoint, false);
    } else if (mt == "qwen3") {
      tt = base::TokenizerType::kEncodeBpe;
      model = std::make_shared<model::Qwen3Model>(tt, args.tokenizer,
                                                  args.checkpoint, false);
    } else if (mt == "llama" || mt == "llama3") {
      tt = base::TokenizerType::kEncodeSpe;
      model = std::make_shared<model::LLamaModel>(tt, args.tokenizer,
                                                  args.checkpoint, false);
    } else {
      LOG(ERROR) << "未知 --model-type: " << mt
                 << " (支持: qwen2 | qwen3 | llama)";
      return nullptr;
    }
    auto st = model->init(base::DeviceType::kDeviceCUDA);
    if (!st) {
      LOG(FATAL) << "模型初始化失败: " << st.get_err_code();
    }
    return model;
  };

  auto model = load_model();
  if (!model) return -1;

  // ---- Tokenize(仅对抽样后的子集) ----
  std::vector<std::vector<int>> prompt_tokens;
  prompt_tokens.reserve(dataset.size());
  for (const auto& r : dataset) {
    auto t = model->encode(r.prompt);
    if (!t.empty()) prompt_tokens.push_back(std::vector<int>(t.begin(), t.end()));
  }
  if (prompt_tokens.empty()) {
    LOG(ERROR) << "所有 prompt 编码后均为空";
    return -1;
  }

  // ---- 模型结构 / 算力 ----
  ModelDims dims;
  double flops_per_tok = 0;
  if (load_model_dims(args.model_config, dims)) {
    flops_per_tok = flops_per_token(dims);
  } else {
    LOG(WARNING) << "无法读取 model config,MFU 将不可用: " << args.model_config;
  }
  double peak_tflops = peak_fp32_tflops();

  // ---- 正式压测 ----
  // 多轮迭代可安全复用同一 Model:引擎 decode_step 已在 replay 前校验图中烘焙的
  // KV/logits 指针与形状,新 Scheduler 缓冲地址变化时自动销毁旧图并重捕获
  // (见 src/source/model/model.cpp),无需每轮重载模型。
  std::vector<RunMetrics> results;
  results.reserve(args.iterations);
  for (int i = 0; i < args.iterations; ++i) {
    results.push_back(run_once(model, prompt_tokens, i, args.max_batch,
                               args.max_gen, flops_per_tok, peak_tflops));
    print_report(results.back());
  }

  // ---- CSV ----
  {
    size_t slash = args.output_csv.find_last_of('/');
    if (slash != std::string::npos) {
      std::error_code ec;
      std::filesystem::create_directories(args.output_csv.substr(0, slash), ec);
    }
    write_csv(args.output_csv, results);
  }

  return 0;
}
