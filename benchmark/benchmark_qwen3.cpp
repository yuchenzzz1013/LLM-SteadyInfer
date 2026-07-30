#include <base/base.h>
#include <glog/logging.h>
#include "model/qwen3.h"

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

// ========== 与 demo 一致的模板填充函数 ==========
std::string fill_template(const std::string& content) {
    const std::string format =
        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
    std::string result = format;
    size_t pos = result.find("%s");
    if (pos != std::string::npos) {
        result.replace(pos, 2, content);
    }
    return result;
}

// ========== 统计数据结构 ==========
struct GenerationStats {
    int prompt_len;
    int gen_len;
    double ttft_ms;
    double tpot_ms;
    double e2e_ms;
    double avg_itl_ms;
    double max_itl_ms;
    double min_itl_ms;
    double prefix_time_ms;
    double generation_time_ms;
    double total_throughput_tps;
    double gen_throughput_tps;
};

// ========== 核心生成函数（适配 Qwen3） ==========
GenerationStats benchmark_generate(model::Qwen3Model& model,
                                   const std::string& user_content,
                                   int max_steps,
                                   bool need_output = false) {
    using Clock = std::chrono::steady_clock;
    auto start_time = Clock::now();

    // 应用模板
    std::string sentence = fill_template(user_content);

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

    std::vector<double> token_timestamps_ms;
    bool first_token_generated = false;
    double first_token_time_ms = 0.0;
    double last_token_time_ms = 0.0;
    int generated_count = 0;

    bool prefix_done = false;
    double prefix_time_ms = 0.0;

    while (pos < max_steps) {
        pos_tensor.index<int32_t>(0) = pos;

        if (pos < prompt_len - 1) {
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        } else {
            if (!prefix_done) {
                auto prefix_end = Clock::now();
                prefix_time_ms = std::chrono::duration<double, std::milli>(prefix_end - start_time).count();
                prefix_done = true;
            }

            is_prompt = false;
            tokens = std::vector<int32_t>{next};
            const auto& token_embedding = model.embedding(tokens);
            tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);

            model.predict(input, pos_tensor, is_prompt, next);
            cudaDeviceSynchronize();

            auto now = Clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();

            // 过滤特殊 token（与 demo 一致）
            if (next != 151645 && next != 151644) {
                words.push_back(next);
                generated_count++;

                if (!first_token_generated) {
                    first_token_generated = true;
                    first_token_time_ms = elapsed_ms;
                    last_token_time_ms = elapsed_ms;
                } else {
                    double itl = elapsed_ms - last_token_time_ms;
                    token_timestamps_ms.push_back(itl);
                    last_token_time_ms = elapsed_ms;
                }
            }

            if (model.is_sentence_ending(next)) {
                break;
            }
        }

        if (pos < prompt_len - 1) {
            next = tokens.at(pos + 1);
        }
        pos += 1;
    }

    GenerationStats stats;
    stats.prompt_len = prompt_len;
    stats.gen_len = generated_count;
    stats.e2e_ms = std::chrono::duration<double, std::milli>(Clock::now() - start_time).count();
    stats.prefix_time_ms = prefix_time_ms;
    stats.generation_time_ms = stats.e2e_ms - prefix_time_ms;

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

// ========== CSV 写入和百分位数函数（与之前相同） ==========
void write_csv(const std::string& filename,
               const std::vector<std::string>& prompts,
               const std::vector<std::vector<GenerationStats>>& all_stats) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Cannot open CSV file: " << filename;
        return;
    }
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
        LOG(INFO) << "  prompts_comma_separated : optional, default 'What is AI?'";
        LOG(INFO) << "  max_steps               : optional, default 128";
        return -1;
    }

    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];
    int iterations = std::stoi(argv[3]);
    std::string output_csv = argv[4];

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
    if (prompts.empty()) prompts.push_back("What is AI?");

    int max_steps = 128;
    if (argc >= 7) max_steps = std::stoi(argv[6]);

    LOG(INFO) << "Checkpoint: " << checkpoint_path;
    LOG(INFO) << "Tokenizer: " << tokenizer_path;
    LOG(INFO) << "Iterations per prompt: " << iterations;
    LOG(INFO) << "Max steps: " << max_steps;
    LOG(INFO) << "Prompts: " << prompts.size() << " prompts";

    model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path,
                            checkpoint_path, false);
    auto init_status = model.init(base::DeviceType::kDeviceCUDA);
    if (!init_status) {
        LOG(FATAL) << "Model init failed, error code: " << init_status.get_err_code();
    }

    LOG(INFO) << "Warming up (1 run) ...";
    benchmark_generate(model, "warm up", 32, false);

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

    write_csv(output_csv, prompts, all_stats);

    LOG(INFO) << "========== Summary ==========";
    for (size_t p = 0; p < prompts.size(); ++p) {
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
        printf("\nPrompt: \"%s\" (runs=%zu)\n", prompts[p].c_str(), vec.size());
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