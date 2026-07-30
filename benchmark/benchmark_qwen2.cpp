#include <base/base.h>
#include <glog/logging.h>
#include "model/qwen2.h"

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

// ========== 统计数据结构（新增字段） ==========
struct GenerationStats {
    int prompt_len;
    int gen_len;
    double ttft_ms;                  // Time To First Token
    double tpot_ms;                  // Time Per Output Token (不含首token)
    double e2e_ms;                  // 端到端总时间
    double avg_itl_ms;              // 平均 token 间延迟
    double max_itl_ms;
    double min_itl_ms;
    double prefix_time_ms;          // ★ 新增：前缀处理时间（Prefill 阶段）
    double generation_time_ms;      // ★ 新增：生成阶段总时间（从第一个token到结束）
    double total_throughput_tps;    // ★ 新增：端到端吞吐量（总token/总时间）
    double gen_throughput_tps;      // 生成吞吐量（原 throughput_tps）
};

// ========== 核心推理函数（带精细计时，新增 prefix 记录） ==========
GenerationStats benchmark_generate(model::Qwen2Model& model,
                                   const std::string& sentence,
                                   int max_steps,
                                   bool need_output = false) {
    using Clock = std::chrono::steady_clock;
    auto start_time = Clock::now();

    auto tokens = model.encode(sentence);
    int prompt_len = tokens.size();
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    int pos = 0;
    int next = tokens.at(pos);
    bool is_prompt = true;
    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

    std::vector<int32_t> words;
    words.push_back(next);

    // 用于记录生成阶段的时间戳（ITL 计算）
    std::vector<double> token_timestamps_ms;
    bool first_token_generated = false;
    double first_token_time_ms = 0.0;
    double last_token_time_ms = 0.0;
    int generated_count = 0;

    // ★ 新增：用于记录前缀处理结束时间
    bool prefix_done = false;
    double prefix_time_ms = 0.0;

    while (pos < max_steps) {
        pos_tensor.index<int32_t>(0) = pos;

        if (pos < prompt_len - 1) {
            // ---------- 处理 prompt tokens ----------
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
            // 注意：这里不记录时间，但会在进入生成阶段时记录整体前缀耗时
        } else {
            // ---------- 进入生成阶段 ----------
            // ★ 第一次进入生成阶段时，记录前缀处理结束时间
            if (!prefix_done) {
                auto prefix_end = Clock::now();
                prefix_time_ms = std::chrono::duration<double, std::milli>(prefix_end - start_time).count();
                prefix_done = true;
            }

            // 执行生成
            is_prompt = false;
            tokens = std::vector<int32_t>{next};
            const auto& token_embedding = model.embedding(tokens);
            tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);

            model.predict(input, pos_tensor, is_prompt, next);
            cudaDeviceSynchronize();

            auto now = Clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();

            if (!first_token_generated) {
                first_token_generated = true;
                first_token_time_ms = elapsed_ms;
                last_token_time_ms = elapsed_ms;
            } else {
                double itl = elapsed_ms - last_token_time_ms;
                token_timestamps_ms.push_back(itl);
                last_token_time_ms = elapsed_ms;
            }

            words.push_back(next);
            generated_count++;

            if (model.is_sentence_ending(next)) {
                break;
            }
        }

        // 更新下一个 token（prompt 阶段从 tokens 取，生成阶段 next 已由 predict 更新）
        if (pos < prompt_len - 1) {
            next = tokens.at(pos + 1);
        }
        pos += 1;
    }

    // ========== 计算统计量 ==========
    GenerationStats stats;
    stats.prompt_len = prompt_len;
    stats.gen_len = generated_count;
    stats.e2e_ms = std::chrono::duration<double, std::milli>(Clock::now() - start_time).count();
    stats.prefix_time_ms = prefix_time_ms;   // 前缀处理时间
    stats.generation_time_ms = stats.e2e_ms - prefix_time_ms; // 生成阶段总时间

    if (generated_count > 0) {
        stats.ttft_ms = first_token_time_ms;
        if (generated_count > 1) {
            stats.tpot_ms = (stats.e2e_ms - first_token_time_ms) / (generated_count - 1);
        } else {
            stats.tpot_ms = 0.0;
        }

        if (!token_timestamps_ms.empty()) {
            double sum = std::accumulate(token_timestamps_ms.begin(), token_timestamps_ms.end(), 0.0);
            stats.avg_itl_ms = sum / token_timestamps_ms.size();
            stats.max_itl_ms = *std::max_element(token_timestamps_ms.begin(), token_timestamps_ms.end());
            stats.min_itl_ms = *std::min_element(token_timestamps_ms.begin(), token_timestamps_ms.end());
        } else {
            stats.avg_itl_ms = stats.max_itl_ms = stats.min_itl_ms = 0.0;
        }

        stats.gen_throughput_tps = generated_count / (stats.e2e_ms / 1000.0);
        // ★ 端到端吞吐量（总token数 / 总时间）
        double total_tokens = prompt_len + generated_count;
        stats.total_throughput_tps = total_tokens / (stats.e2e_ms / 1000.0);
    } else {
        stats.ttft_ms = stats.tpot_ms = stats.avg_itl_ms = stats.max_itl_ms = stats.min_itl_ms = 0.0;
        stats.gen_throughput_tps = stats.total_throughput_tps = 0.0;
    }

    if (need_output) {
        printf("%s ", model.decode(words).data());
        fflush(stdout);
    }

    return stats;
}

// ========== 写入 CSV（新增列） ==========
void write_csv(const std::string& filename,
               const std::vector<std::string>& prompts,
               const std::vector<std::vector<GenerationStats>>& all_stats) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Cannot open CSV file: " << filename;
        return;
    }

    // 表头（新增 prefix_time_ms, generation_time_ms, total_throughput_tps）
    file << "prompt,run_id,prompt_len,gen_len,ttft_ms,tpot_ms,e2e_ms,"
         << "prefix_time_ms,generation_time_ms,"
         << "avg_itl_ms,max_itl_ms,min_itl_ms,"
         << "gen_throughput_tps,total_throughput_tps\n";

    for (size_t p = 0; p < prompts.size(); ++p) {
        const auto& stats_vec = all_stats[p];
        for (size_t i = 0; i < stats_vec.size(); ++i) {
            const auto& s = stats_vec[i];
            file << "\"" << prompts[p] << "\"," << i << ","
                 << s.prompt_len << "," << s.gen_len << ","
                 << std::fixed << std::setprecision(3)
                 << s.ttft_ms << "," << s.tpot_ms << ","
                 << s.e2e_ms << ","
                 << s.prefix_time_ms << "," << s.generation_time_ms << ","
                 << s.avg_itl_ms << "," << s.max_itl_ms << "," << s.min_itl_ms << ","
                 << s.gen_throughput_tps << "," << s.total_throughput_tps << "\n";
        }
    }

    file.close();
    LOG(INFO) << "CSV results saved to " << filename;
}

// ========== 计算百分位数 ==========
double percentile(const std::vector<double>& data, double p) {
    if (data.empty()) return 0.0;
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * sorted.size()) - 1);
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return sorted[idx];
}

// ========== 主函数 ==========
int main(int argc, char* argv[]) {
    if (argc < 5) {
        LOG(INFO) << "Usage: " << argv[0]
                  << " <checkpoint_path> <tokenizer_path> <iterations> <output_csv> "
                     "[prompts_comma_separated] [max_steps]";
        LOG(INFO) << "  prompts_comma_separated : optional, default 'hi!', e.g. 'hello,world,how are you'";
        LOG(INFO) << "  max_steps               : optional, default 128";
        return -1;
    }

    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];
    int iterations = std::stoi(argv[3]);
    std::string output_csv = argv[4];

    // 解析提示词列表
    std::vector<std::string> prompts;
    if (argc >= 6) {
        std::string prompts_str = argv[5];
        std::stringstream ss(prompts_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            size_t start = item.find_first_not_of(" \t");
            if (start == std::string::npos) continue;
            size_t end = item.find_last_not_of(" \t");
            prompts.push_back(item.substr(start, end - start + 1));
        }
    }
    if (prompts.empty()) {
        prompts.push_back("hi!");
    }

    int max_steps = 128;
    if (argc >= 7) {
        max_steps = std::stoi(argv[6]);
    }

    LOG(INFO) << "Checkpoint: " << checkpoint_path;
    LOG(INFO) << "Tokenizer: " << tokenizer_path;
    LOG(INFO) << "Iterations per prompt: " << iterations;
    LOG(INFO) << "Max steps: " << max_steps;
    LOG(INFO) << "Prompts: " << prompts.size() << " prompts";
    for (const auto& p : prompts) LOG(INFO) << "  - " << p;

    // ---------- 加载模型 ----------
    model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path,
                            checkpoint_path, false);
    auto init_status = model.init(base::DeviceType::kDeviceCUDA);
    if (!init_status) {
        LOG(FATAL) << "Model init failed, error code: " << init_status.get_err_code();
    }

    // ---------- 预热 ----------
    LOG(INFO) << "Warming up (1 run) ...";
    benchmark_generate(model, "warm up", 32, false);

    // ---------- 正式测试 ----------
    std::vector<std::vector<GenerationStats>> all_stats;
    all_stats.reserve(prompts.size());

    for (size_t p = 0; p < prompts.size(); ++p) {
        const std::string& prompt = prompts[p];
        LOG(INFO) << "Benchmarking prompt [" << p << "]: " << prompt;

        std::vector<GenerationStats> stats_vec;
        stats_vec.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            auto stats = benchmark_generate(model, prompt, max_steps, false);
            stats_vec.push_back(stats);
            if ((i + 1) % 10 == 0) {
                LOG(INFO) << "  Completed " << (i + 1) << "/" << iterations << " runs";
            }
        }
        all_stats.push_back(stats_vec);
    }

    // ---------- 写入 CSV ----------
    write_csv(output_csv, prompts, all_stats);

    // ---------- 打印汇总统计（新增指标） ----------
    LOG(INFO) << "========== Summary ==========";
    for (size_t p = 0; p < prompts.size(); ++p) {
        const std::string& prompt = prompts[p];
        const auto& vec = all_stats[p];
        if (vec.empty()) continue;

        std::vector<double> ttfts, tpots, e2es, gen_tps, total_tps, prefix_times, gen_times;
        for (const auto& s : vec) {
            ttfts.push_back(s.ttft_ms);
            tpots.push_back(s.tpot_ms);
            e2es.push_back(s.e2e_ms);
            gen_tps.push_back(s.gen_throughput_tps);
            total_tps.push_back(s.total_throughput_tps);
            prefix_times.push_back(s.prefix_time_ms);
            gen_times.push_back(s.generation_time_ms);
        }

        auto print_stats = [](const std::string& name, const std::vector<double>& data, const std::string& unit = "") {
            if (data.empty()) return;
            double avg = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
            double p50 = percentile(data, 50);
            double p90 = percentile(data, 90);
            double p99 = percentile(data, 99);
            printf("  %-20s avg=%8.3f  p50=%8.3f  p90=%8.3f  p99=%8.3f %s\n",
                   name.c_str(), avg, p50, p90, p99, unit.c_str());
        };

        printf("\nPrompt: \"%s\" (runs=%zu)\n", prompt.c_str(), vec.size());
        print_stats("TTFT (ms)", ttfts, "ms");
        print_stats("TPOT (ms)", tpots, "ms");
        print_stats("E2E  (ms)", e2es, "ms");
        print_stats("Prefix time (ms)", prefix_times, "ms");
        print_stats("Generation time (ms)", gen_times, "ms");
        print_stats("Gen Throughput", gen_tps, "tok/s");
        print_stats("Total Throughput", total_tps, "tok/s");
    }

    LOG(INFO) << "Benchmark finished. Results saved to " << output_csv;
    return 0;
}