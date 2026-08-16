// ============================================================================
// 在线 Serving Benchmark(模仿 vLLM benchmark_serving 的在线压测思路)
//
// 与 offline_batch_benchmark 的区别:请求不是一次性全部提交,而是按到达过程
// (泊松/伽马分布)陆续到达 Scheduler 的等待队列,测量在线负载下的指标:
//   - completed_rps / output_tps  完成请求速率 / 输出 Token 吞吐
//   - TTFT / TPOT / ITL / E2E     延迟分布(avg / p50 / p99;TTFT 含排队)
//   - queue_wait                  排队等待延迟(admit - arrival)
//   - avg_queue_len               时间加权平均排队长度
//   - goodput                     满足 TTFT/TPOT SLA 的完成请求速率
//   - kv_cache_fragmentation / batch reconstruct / MFU / NVML(与 offline 同口径)
//
// 用法:
//   ./build/benchmark/online_serving_benchmark \
//       --model-type qwen2 --checkpoint Qwen2.5-0.5B.bin \
//       --tokenizer Qwen/Qwen2.5-0.5B/tokenizer.json \
//       --dataset ShareGPT_prompts.jsonl \
//       --request-rate 8 --num-requests 256 --max-batch 32 --max-gen 256 \
//       --duration 60 --ttft-sla-ms 2000 --tpot-sla-ms 100
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
  std::string output_csv = "results/serving_metrics.csv";
  int num_requests = 512;  // 0 = 使用全部样本
  int max_batch = 32;
  int max_gen = 256;
  int iterations = 1;
  int seed = 42;

  // ---- 在线负载(与 vLLM benchmark_serving 对齐) ----
  double request_rate = -1;  // 请求到达速率 req/s;<0 = 无穷大(全部立即到达,退化为 offline)
  double burstiness = 1.0;   // 到达间隔伽马分布 shape:<1 突发,=1 泊松,>1 平稳
  double duration = 0;       // 请求提交窗口秒数;<=0 不限制(提交完即排空)
  double ttft_sla_ms = -1;   // TTFT SLA 阈值;<=0 禁用
  double tpot_sla_ms = -1;   // TPOT SLA 阈值;<=0 禁用
};

// 解析浮点参数,支持 "inf"(无穷大 → -1)
static double parse_double_arg(const std::string& s) {
  if (s == "inf") return -1;
  return std::stod(s);
}

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
  if (has_arg(argc, argv, "--request-rate"))
    a.request_rate = parse_double_arg(get_arg(argc, argv, "--request-rate"));
  if (has_arg(argc, argv, "--burstiness"))
    a.burstiness = std::stod(get_arg(argc, argv, "--burstiness"));
  if (has_arg(argc, argv, "--duration"))
    a.duration = std::stod(get_arg(argc, argv, "--duration"));
  if (has_arg(argc, argv, "--ttft-sla-ms"))
    a.ttft_sla_ms = std::stod(get_arg(argc, argv, "--ttft-sla-ms"));
  if (has_arg(argc, argv, "--tpot-sla-ms"))
    a.tpot_sla_ms = std::stod(get_arg(argc, argv, "--tpot-sla-ms"));
  return a;
}

// 默认路径解析(见 bench_common.h 的 resolve_default):
// 按 CWD 解释失败后,相对可执行文件推断出的项目根目录
// (build/benchmark/online_serving_benchmark 与
//  benchmark/build/online_serving_benchmark 两种布局均适用)
static void print_usage(const char* prog) {
  std::cout
      << "用法 1(与 offline_batch_benchmark 相同的调用方式,位置参数):\n"
      << "  " << prog << " <checkpoint> <tokenizer> <num_requests> <max_batch>"
      << " <max_gen> [output_csv] [iterations]\n"
      << "  例: ./build/benchmark/online_serving_benchmark"
      << " Qwen2.5-0.5B.bin Qwen/Qwen2.5-0.5B/tokenizer.json"
      << " 512 32 256 results/serving_metrics.csv\n"
      << "  在线负载参数(如 --request-rate)需用 --flag 方式追加\n\n"
      << "用法 2(--flag 风格,默认值见括号):\n"
      << "  " << prog << " [选项]\n"
      << "  --model-type <type>    模型类型: qwen2(默认) | qwen3 | llama\n"
      << "                         (llama 等价 llama3;三模型均已编译进引擎,运行时选择)\n"
      << "  --dataset <path>       ShareGPT_prompts.jsonl 路径(默认自动探测)\n"
      << "  --checkpoint <path>    模型权重 .bin 路径(默认 Qwen2.5-0.5B.bin)\n"
      << "  --tokenizer <path>     tokenizer 路径(qwen2/qwen3 为 tokenizer.json,\n"
      << "                         llama 为 sentencepiece 模型;默认 Qwen/Qwen2.5-0.5B/tokenizer.json)\n"
      << "  --model-config <path>  config.json 路径,用于计算 MFU(默认 Qwen/Qwen2.5-0.5B/config.json)\n"
      << "  --num-requests <N>     压测请求总数,0 表示全部(默认 512)\n"
      << "  --max-batch <N>        最大 batch size(默认 32)\n"
      << "  --max-gen <N>          每个请求最大生成 token 数(默认 256)\n"
      << "  --iterations <N>       重复轮数(默认 1)\n"
      << "  --seed <N>             数据集抽样与到达过程随机种子(默认 42)\n"
      << "  --request-rate <R>     请求到达速率 req/s;inf 或 -1 表示全部立即到达\n"
      << "                         (退化为 offline 模式,默认 inf)\n"
      << "  --burstiness <g>       到达间隔伽马分布 shape 参数(默认 1.0 = 泊松过程;\n"
      << "                         <1 更突发,>1 更平稳,对齐 vLLM --burstiness)\n"
      << "  --duration <s>         请求提交窗口秒数,到期后停止提交并排空\n"
      << "                         (默认 0 = 不限制)\n"
      << "  --ttft-sla-ms <ms>     TTFT SLA 阈值,用于 goodput 计算(默认关闭)\n"
      << "  --tpot-sla-ms <ms>     TPOT SLA 阈值,用于 goodput 计算(默认关闭)\n"
      << "  --output-csv <path>    结果 CSV 路径(默认 results/serving_metrics.csv)\n";
}

// ============================== 单轮压测 ==============================

struct ServingMetrics {
  int run_id = 0;
  int num_requests = 0, submitted = 0, completed = 0, rejected = 0;
  long long prompt_tokens = 0, output_tokens = 0, total_tokens = 0;
  double wall_time_s = 0;
  double request_rate = 0, burstiness = 1.0, duration_s = 0;

  // 吞吐
  double completed_rps = 0;  // 每秒完成请求数
  double output_tps = 0;     // 每秒输出 Token 数
  double total_tps = 0;      // 每秒总 Token 数(prefill + decode)

  // 延迟(TTFT 从请求提交算起,含排队,与 vLLM 口径一致)
  double ttft_avg_ms = 0, ttft_p50_ms = 0, ttft_p99_ms = 0;
  double tpot_avg_ms = 0, tpot_p50_ms = 0, tpot_p99_ms = 0;
  double itl_avg_ms = 0, itl_p99_ms = 0;
  double e2e_avg_ms = 0, e2e_p99_ms = 0;

  // 排队
  double queue_wait_avg_ms = 0, queue_wait_p50_ms = 0, queue_wait_p99_ms = 0;
  double avg_queue_len = 0;  // 时间加权平均排队长度

  // Goodput(SLA)
  double goodput_rps = 0;  // 满足 TTFT/TPOT SLA 的完成请求速率
  int goodput_count = 0;
  double goodput_pct = 0;  // goodput_count / completed

  // KV cache
  double kv_cache_util_global = 0;   // 时间加权:已用 token 槽 / 已分配容量
  double kv_cache_frag_global = 0;   // 1 - util_global
  double kv_cache_frag_per_seq = 0;  // 每请求视角(与 offline 口径一致)
  double avg_busy_kv_slots = 0, peak_busy_kv_slots = 0;

  // 系统
  double avg_batch_size = 0;  // 平均运行中序列数
  double avg_batch_reconstruct_ms = 0, p99_batch_reconstruct_ms = 0;
  double peak_tflops = 0, mfu = 0;
  double gpu_sm_util_pct = 0, gpu_mem_util_pct = 0, gpu_mem_used_mb = 0;
};

static ServingMetrics serve_once(const std::shared_ptr<model::Model>& model,
                                 const std::vector<std::vector<int>>& prompt_tokens,
                                 const Args& args, int run_id,
                                 double flops_per_tok, double peak_tflops) {
  using namespace scheduler;
  ServingMetrics m;
  m.run_id = run_id;
  m.peak_tflops = peak_tflops;
  m.request_rate = args.request_rate;
  m.burstiness = args.burstiness;
  m.duration_s = args.duration;
  m.num_requests = static_cast<int>(prompt_tokens.size());

  // 根据实际 prompt 长度确定 KV slot 大小(与 offline 同口径)
  int max_prompt_len = 0;
  long long prompt_len_sum = 0;
  for (const auto& t : prompt_tokens) {
    max_prompt_len = std::max(max_prompt_len, static_cast<int>(t.size()));
    prompt_len_sum += t.size();
  }
  int max_total_seq_len =
      std::min(max_prompt_len + args.max_gen, static_cast<int>(model->seq_len()));
  LOG(INFO) << "[SERVING] KV slot: max_prompt=" << max_prompt_len
            << " max_gen=" << args.max_gen << " => " << max_total_seq_len;

  // 收缩模型内部 KV cache(必须在任何 decode CUDA graph 捕获之前,同 offline)
  model->resize_internal_kv_cache(max_prompt_len);

  // 分页块大小按工作负载平均 prompt 长度自适应(短文本 8/长文本 32/默认 16)。
  const long long avg_prompt_len =
      prompt_tokens.empty() ? 0 : prompt_len_sum / static_cast<long long>(prompt_tokens.size());
  Scheduler sched(model, args.max_batch, max_total_seq_len, args.max_gen,
                  Scheduler::resolve_block_size(avg_prompt_len));

  // ---- 预热:与正式压测共用同一 Scheduler(同 offline,统计中跳过) ----
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

  // ---- 到达过程 ----
  // vLLM 语义:request_rate = inf 时全部请求在 t=0 提交(退化为 offline);
  // 否则按伽马分布生成到达间隔(burstiness=1 即泊松过程)。
  std::mt19937 rng(args.seed);
  const bool unlimited = args.request_rate < 0;
  auto next_interval = [&]() -> double {
    if (args.burstiness == 1.0) {
      std::exponential_distribution<double> d(args.request_rate);
      return d(rng);
    }
    // Gamma(shape=burstiness, scale=1/(rate*burstiness)) → 均值 1/rate
    std::gamma_distribution<double> d(
        args.burstiness, 1.0 / (args.request_rate * args.burstiness));
    return d(rng);
  };

  if (unlimited) {
    for (const auto& t : prompt_tokens) {
      if (sched.add_request(t) < 0) m.rejected++;
      else m.submitted++;
    }
  }

  GpuUtilSampler sampler;
  sampler.start();

  // 主循环:到达判断与调度步进交错;每步采样 KV 占用与排队长度快照
  auto start = Clock::now();
  double next_arrival_s = unlimited ? 0.0 : next_interval();
  bool arrival_open = true;
  long long used_cap_sum = 0, alloc_cap_sum = 0;
  long long busy_slots_sum = 0, running_sum = 0, queue_len_sum = 0;
  int kv_samples = 0, queue_samples = 0, peak_busy = 0;

  // 在线模式下请求按到达过程陆续提交,两次到达之间 scheduler 可能完全排空;
  // 因此退出条件必须同时考虑"还有未提交的请求",不能只看 all_finished。
  while (!sched.all_finished() ||
         (arrival_open &&
          m.submitted + m.rejected < static_cast<int>(prompt_tokens.size()))) {
    double now_s = std::chrono::duration<double>(Clock::now() - start).count();

    // 到达:提交窗口内,到点即提交下一条请求
    if (!unlimited && arrival_open) {
      if (args.duration > 0 && now_s >= args.duration) arrival_open = false;
      while (arrival_open &&
             m.submitted + m.rejected < static_cast<int>(prompt_tokens.size()) &&
             now_s >= next_arrival_s) {
        if (sched.add_request(prompt_tokens[m.submitted + m.rejected]) < 0) {
          m.rejected++;
        } else {
          m.submitted++;
        }
        next_arrival_s += next_interval();
      }
    }

    // 空闲等待下一次到达时避免空转烧 CPU(下一轮循环顶部会重新读时钟)
    if (!unlimited && arrival_open && sched.get_running().empty() &&
        sched.num_waiting() == 0 && next_arrival_s > now_s) {
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }

    // 采样(纯 host 端读取,开销可忽略)
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
    queue_len_sum += sched.num_waiting();
    queue_samples++;

    sched.step();
  }
  cudaDeviceSynchronize();
  auto end = Clock::now();
  sampler.stop();

  double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
  double total_s = total_ms / 1000.0;
  m.wall_time_s = total_s;

  // ---- 逐请求统计(与 vLLM 口径:TTFT/E2E 从请求提交算起,含排队) ----
  const auto& finished = sched.get_finished();
  std::vector<double> ttfts, tpots, e2es, itls, qwaits, per_seq_frag;
  for (const auto& seq : finished) {
    if (warm_ids.count(seq.id)) continue;  // 跳过预热请求
    if (seq.num_generated_tokens <= 0) continue;
    double arrival_ms =
        std::chrono::duration<double, std::milli>(seq.arrival_time - start).count();
    double admit_ms =
        std::chrono::duration<double, std::milli>(seq.admit_time - start).count();
    double first_ms =
        std::chrono::duration<double, std::milli>(seq.first_token_time - start).count();
    double finish_ms =
        std::chrono::duration<double, std::milli>(seq.finish_time - start).count();

    double ttft = first_ms - arrival_ms;
    double e2e = finish_ms - arrival_ms;
    double tpot =
        seq.num_generated_tokens > 1 ? (finish_ms - first_ms) / (seq.num_generated_tokens - 1) : 0.0;
    ttfts.push_back(ttft);
    e2es.push_back(e2e);
    tpots.push_back(tpot);
    qwaits.push_back(admit_ms - arrival_ms);
    for (double itl : seq.token_timestamps_ms) itls.push_back(itl);

    int used = seq.num_prompt_tokens + seq.num_generated_tokens;
    per_seq_frag.push_back(1.0 - static_cast<double>(used) / max_total_seq_len);

    // Goodput:SLA 未设置时该维度恒通过(与 vLLM 一致)
    bool ttft_ok = args.ttft_sla_ms <= 0 || ttft <= args.ttft_sla_ms;
    bool tpot_ok = args.tpot_sla_ms <= 0 || tpot <= args.tpot_sla_ms;
    if (ttft_ok && tpot_ok) m.goodput_count++;

    m.prompt_tokens += seq.num_prompt_tokens;
    m.output_tokens += seq.num_generated_tokens;
  }
  m.completed = static_cast<int>(ttfts.size());
  m.total_tokens = m.prompt_tokens + m.output_tokens;

  // ---- 吞吐 ----
  m.completed_rps = total_s > 0 ? m.completed / total_s : 0;
  m.output_tps = total_s > 0 ? m.output_tokens / total_s : 0;
  m.total_tps = total_s > 0 ? m.total_tokens / total_s : 0;
  m.goodput_rps = total_s > 0 ? m.goodput_count / total_s : 0;
  m.goodput_pct = m.completed > 0
                      ? static_cast<double>(m.goodput_count) / m.completed : 0;

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

  // ---- 排队 ----
  m.queue_wait_avg_ms = mean(qwaits);
  m.queue_wait_p50_ms = percentile(qwaits, 50);
  m.queue_wait_p99_ms = percentile(qwaits, 99);
  m.avg_queue_len =
      queue_samples > 0 ? static_cast<double>(queue_len_sum) / queue_samples : 0;

  // ---- KV cache 碎片率(与 offline 同口径) ----
  m.kv_cache_util_global =
      alloc_cap_sum > 0 ? static_cast<double>(used_cap_sum) / alloc_cap_sum : 0;
  m.kv_cache_frag_global = 1.0 - m.kv_cache_util_global;
  m.kv_cache_frag_per_seq = mean(per_seq_frag);
  m.avg_busy_kv_slots =
      kv_samples > 0 ? static_cast<double>(busy_slots_sum) / kv_samples : 0;
  m.peak_busy_kv_slots = peak_busy;

  // ---- batch 重建开销 ----
  const auto& recon = sched.get_batch_reconstruct_times_ms();
  if (!recon.empty()) {
    m.avg_batch_reconstruct_ms = mean(recon);
    m.p99_batch_reconstruct_ms = percentile(recon, 99);
  }

  // ---- 系统 ----
  m.avg_batch_size =
      kv_samples > 0 ? static_cast<double>(running_sum) / kv_samples : 0;
  double flops = flops_per_tok * static_cast<double>(m.total_tokens);
  m.mfu = (peak_tflops > 0 && total_s > 0)
              ? flops / (peak_tflops * 1e12 * total_s) : 0;
  m.gpu_sm_util_pct = sampler.sm_util_pct();
  m.gpu_mem_util_pct = sampler.mem_util_pct();
  m.gpu_mem_used_mb = sampler.mem_used_mb();

  LOG(INFO) << "[SERVING] run " << run_id << " finished: " << total_ms << " ms, "
            << m.completed << "/" << m.num_requests << " requests (rejected="
            << m.rejected << "), " << m.output_tokens << " output tokens, "
            << "nvml samples=" << sampler.samples();
  return m;
}

// ============================== CSV 输出 ==============================

static void write_csv(const std::string& path,
                      const std::vector<ServingMetrics>& results) {
  std::ofstream f(path);
  if (!f.is_open()) {
    LOG(ERROR) << "无法打开输出文件: " << path;
    return;
  }
  f << "run_id,request_rate,burstiness,duration_s,num_requests,submitted,"
       "completed,rejected,prompt_tokens,output_tokens,total_tokens,wall_time_s,"
       "completed_rps,output_tps,total_tps,"
       "ttft_avg_ms,ttft_p50_ms,ttft_p99_ms,"
       "tpot_avg_ms,tpot_p50_ms,tpot_p99_ms,itl_avg_ms,itl_p99_ms,"
       "e2e_avg_ms,e2e_p99_ms,"
       "queue_wait_avg_ms,queue_wait_p50_ms,queue_wait_p99_ms,avg_queue_len,"
       "goodput_rps,goodput_count,goodput_pct,"
       "kv_cache_util_global,kv_cache_frag_global,kv_cache_frag_per_seq,"
       "avg_busy_kv_slots,peak_busy_kv_slots,"
       "avg_batch_size,avg_batch_reconstruct_ms,p99_batch_reconstruct_ms,"
       "mfu,peak_fp32_tflops,gpu_sm_util_pct,gpu_mem_util_pct,gpu_mem_used_mb\n";
  f << std::fixed << std::setprecision(6);
  for (const auto& r : results) {
    f << r.run_id << "," << r.request_rate << "," << r.burstiness << ","
      << r.duration_s << "," << r.num_requests << "," << r.submitted << ","
      << r.completed << "," << r.rejected << "," << r.prompt_tokens << ","
      << r.output_tokens << "," << r.total_tokens << "," << r.wall_time_s << ","
      << r.completed_rps << "," << r.output_tps << "," << r.total_tps << ","
      << r.ttft_avg_ms << "," << r.ttft_p50_ms << "," << r.ttft_p99_ms << ","
      << r.tpot_avg_ms << "," << r.tpot_p50_ms << "," << r.tpot_p99_ms << ","
      << r.itl_avg_ms << "," << r.itl_p99_ms << "," << r.e2e_avg_ms << ","
      << r.e2e_p99_ms << "," << r.queue_wait_avg_ms << "," << r.queue_wait_p50_ms
      << "," << r.queue_wait_p99_ms << "," << r.avg_queue_len << ","
      << r.goodput_rps << "," << r.goodput_count << "," << r.goodput_pct << ","
      << r.kv_cache_util_global << "," << r.kv_cache_frag_global << ","
      << r.kv_cache_frag_per_seq << "," << r.avg_busy_kv_slots << ","
      << r.peak_busy_kv_slots << "," << r.avg_batch_size << ","
      << r.avg_batch_reconstruct_ms << "," << r.p99_batch_reconstruct_ms << ","
      << r.mfu << "," << r.peak_tflops << "," << r.gpu_sm_util_pct << ","
      << r.gpu_mem_util_pct << "," << r.gpu_mem_used_mb << "\n";
  }
  f.close();
  LOG(INFO) << "结果 CSV 已写出: " << path;
}

// ============================== 控制台报告 ==============================

static void print_report(const ServingMetrics& r, const Args& args) {
  auto pct = [](double v) { return v * 100.0; };
  std::cout << "\n============ Serving Benchmark Result (Run " << r.run_id
            << ") ============\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Successful requests:                    " << r.completed << "\n";
  std::cout << "Rejected requests:                      " << r.rejected << "\n";
  std::cout << "Benchmark duration (s):                 " << r.wall_time_s << "\n";
  std::cout << "Total input tokens:                     " << r.prompt_tokens << "\n";
  std::cout << "Total generated tokens:                 " << r.output_tokens << "\n";
  std::cout << "Request throughput (req/s):             " << r.completed_rps << "\n";
  std::cout << "Output token throughput (tok/s):        " << r.output_tps << "\n";
  std::cout << "Total Token throughput (tok/s):         " << r.total_tps << "\n";
  if (args.ttft_sla_ms > 0 || args.tpot_sla_ms > 0) {
    std::cout << "Goodput (req/s):                        " << r.goodput_rps
              << "  (" << r.goodput_count << "/" << r.completed << " = "
              << pct(r.goodput_pct) << "% within SLA:";
    if (args.ttft_sla_ms > 0) std::cout << " TTFT<=" << args.ttft_sla_ms << "ms";
    if (args.tpot_sla_ms > 0) std::cout << " TPOT<=" << args.tpot_sla_ms << "ms";
    std::cout << ")\n";
  } else {
    std::cout << "Goodput (req/s):                        " << r.goodput_rps
              << "  (SLA 未设置,等价于请求吞吐)\n";
  }

  std::cout << "\n---------------Time to First Token----------------\n";
  std::cout << "Mean TTFT (ms):                         " << r.ttft_avg_ms << "\n";
  std::cout << "Median TTFT (ms):                       " << r.ttft_p50_ms << "\n";
  std::cout << "P99 TTFT (ms):                          " << r.ttft_p99_ms << "\n";

  std::cout << "-----Time per Output Token (excl. 1st token)------\n";
  std::cout << "Mean TPOT (ms):                         " << r.tpot_avg_ms << "\n";
  std::cout << "Median TPOT (ms):                       " << r.tpot_p50_ms << "\n";
  std::cout << "P99 TPOT (ms):                          " << r.tpot_p99_ms << "\n";

  std::cout << "---------------Inter-token Latency----------------\n";
  std::cout << "Mean ITL (ms):                          " << r.itl_avg_ms << "\n";
  std::cout << "P99 ITL (ms):                           " << r.itl_p99_ms << "\n";

  std::cout << "---------------------End-to-End-------------------\n";
  std::cout << "Mean E2E (ms):                          " << r.e2e_avg_ms << "\n";
  std::cout << "P99 E2E (ms):                           " << r.e2e_p99_ms << "\n";

  std::cout << "---------------------Queue(排队)-------------------\n";
  std::cout << "Mean Queue Wait (ms):                   " << r.queue_wait_avg_ms << "\n";
  std::cout << "Median Queue Wait (ms):                 " << r.queue_wait_p50_ms << "\n";
  std::cout << "P99 Queue Wait (ms):                    " << r.queue_wait_p99_ms << "\n";
  std::cout << "Mean Queue Length:                      " << r.avg_queue_len << "\n";

  std::cout << "-----------------------System-----------------------\n";
  std::cout << "KV cache fragmentation (global):        " << pct(r.kv_cache_frag_global)
            << "%  (utilization " << pct(r.kv_cache_util_global) << "%)\n";
  std::cout << "KV cache fragmentation (per-seq):       " << pct(r.kv_cache_frag_per_seq)
            << "%\n";
  std::cout << "Busy KV slots: avg=" << r.avg_busy_kv_slots
            << "  peak=" << r.peak_busy_kv_slots << "\n";
  std::cout << "Average running batch:                  " << r.avg_batch_size << "\n";
  std::cout << std::setprecision(3);
  std::cout << "Batch reconstruct: avg=" << r.avg_batch_reconstruct_ms
            << " ms  p99=" << r.p99_batch_reconstruct_ms << " ms\n";
  std::cout << std::setprecision(2);
  std::cout << "MFU:                                    " << pct(r.mfu)
            << "%  (peak " << r.peak_tflops << " TFLOPS FP32)\n";
  std::cout << "GPU SM util (NVML):                     " << r.gpu_sm_util_pct << "%\n";
  std::cout << "GPU mem util (NVML):                    " << r.gpu_mem_util_pct
            << "%  (used " << r.gpu_mem_used_mb << " MB)\n";
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
  // 两种常见布局: <root>/build/benchmark/online_serving_benchmark
  //             <root>/benchmark/build/online_serving_benchmark
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
  LOG(INFO) << "dataset    : " << args.dataset
            << "\ncheckpoint : " << args.checkpoint
            << "\ntokenizer  : " << args.tokenizer
            << "\nconfig     : " << args.model_config
            << "\nrequest_rate="
            << (args.request_rate < 0 ? "inf" : std::to_string(args.request_rate))
            << " burstiness=" << args.burstiness
            << " duration=" << args.duration;

  // ---- 数据集 ----
  std::vector<Request> dataset = load_dataset(args.dataset);
  if (dataset.empty()) {
    LOG(ERROR) << "数据集为空: " << args.dataset;
    return -1;
  }
  LOG(INFO) << "数据集加载: " << dataset.size() << " 条 prompt";

  // 按种子抽样(仅当请求数小于数据集规模时)
  if (args.num_requests > 0 && args.num_requests < static_cast<int>(dataset.size())) {
    std::mt19937 rng(args.seed);
    std::shuffle(dataset.begin(), dataset.end(), rng);
    dataset.resize(args.num_requests);
    LOG(INFO) << "随机抽样(seed=" << args.seed << "): " << dataset.size() << " 条";
  }

  // ---- 模型加载:按 --model-type 运行时选择(与 offline 一致) ----
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
  LOG(INFO) << "模型就绪: layers=" << model->layer_num()
            << " hidden=" << model->hidden_dim()
            << " kv_dim=" << model->kv_dim()
            << " vocab=" << model->vocab_size()
            << " seq_len=" << model->seq_len();

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
  LOG(INFO) << "编码完成: " << prompt_tokens.size() << " 条有效 prompt";

  // ---- 模型结构 / 算力 ----
  ModelDims dims;
  double flops_per_tok = 0;
  if (load_model_dims(args.model_config, dims)) {
    flops_per_tok = flops_per_token(dims);
    LOG(INFO) << "模型结构: d=" << dims.d << " ffn=" << dims.ffn
              << " layers=" << dims.layers << " heads=" << dims.heads
              << " kv_heads=" << dims.kv_heads << " => " << flops_per_tok / 1e9
              << " GFLOPs/token";
  } else {
    LOG(WARNING) << "无法读取 model config,MFU 将不可用: " << args.model_config;
  }
  double peak_tflops = peak_fp32_tflops();
  LOG(INFO) << "GPU 峰值 FP32 算力: " << peak_tflops << " TFLOPS";

  // ---- 正式压测 ----
  // 多轮迭代可安全复用同一 Model(同 offline,decode_step 会校验并重捕获 CUDA graph)
  std::vector<ServingMetrics> results;
  results.reserve(args.iterations);
  for (int i = 0; i < args.iterations; ++i) {
    LOG(INFO) << "=== Run " << (i + 1) << "/" << args.iterations << " ===";
    results.push_back(serve_once(model, prompt_tokens, args, i, flops_per_tok,
                                 peak_tflops));
    print_report(results.back(), args);
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

  LOG(INFO) << "Serving benchmark 完成。";
  return 0;
}
