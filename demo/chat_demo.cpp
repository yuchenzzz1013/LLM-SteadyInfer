// ============================================================================
// 连续对话(实时 Chat)最小示例 — LLM-SteadyInfer
//
// 一个命令行里的实时对话循环:
//   ChatML 模板拼 prompt → model->encode → scheduler.add_request
//   → step() 直到 all_finished → 逐 token 流式打印 → 回复写回记忆
//
// 特性:
//   - 记忆:每轮都把 system + 之前的问答 + 本轮提问重新拼成 prompt,模型据此回答;
//     历史超过上下文预算时从最旧的一轮开始整轮丢弃(按 token 数精确裁剪)。
//   - 上下文 / 显存:--max-seq-len 默认 0 = 自适应,取"模型窗口(Qwen3-4B 已按官方
//     值设为 40960)"和"显存放得下"的较小值,把空闲显存尽量变成 KV 池;启动时会
//     打印 KV 池大小与整卡占用。每轮生成长度由 --max-new-tokens 控制,两者共享
//     同一份 KV cache。
//   - 选模型:启动参数 --model-type / --model-dir / --tokenizer 选模型;
//     对话中也可以用 /model <type> <模型目录> [tokenizer] 热切换(会清空记忆)。
//
// 每轮重发完整历史是最直观的连续对话实现;Scheduler 的 prefix cache 会自动命中
// 上一轮已缓存的公共前缀(整块复用,跳过重复 prefill),重复部分基本不花算力。
//
// 用法:
//   ./build/demo/chat_demo                                   # 实时对话,exit 退出
//   ./build/demo/chat_demo --model-dir Qwen3-4B --max-seq-len 1024 --max-new-tokens 256
//   ./build/demo/chat_demo --questions "你好|讲个笑话"          # 非交互,依次提问
//   ./build/demo/chat_demo --help
// ============================================================================

#include <base/alloc.h>
#include <base/base.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <model/llama3.h>
#include <model/qwen2.h>
#include <model/qwen3.h>
#include <scheduler/scheduler.h>

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// ============================== 参数 ==============================

struct Args {
  std::string model_type = "qwen3";    // qwen3 | qwen2 | llama
  std::string model_dir = "Qwen3-4B";  // HF safetensors 目录(config.json + *.safetensors)
  std::string tokenizer = "Qwen3-4B/tokenizer.json";
  std::string device = "cuda";         // cuda | cpu
  std::string system_prompt = "你是一个乐于助人的 AI 助手,回答尽量简洁。";
  // 上下文自适应(--max-seq-len 0):取"模型窗口"与"显存放得下"中较小的那个,
  // 也就是把 A100 的显存尽量变成 KV 池(每 token KV = layers x 2 x kv_dim x dtype)。
  int max_seq_len = 0;
  int max_batch = 1;              // 调度器槽位数;KV 池 = max_batch x max_seq_len
  double gpu_mem_fraction = 0.9;  // KV 池最多占空闲显存的比例(留 10% 给激活/临时缓冲)
  int max_new_tokens = 1024;      // 每轮最多生成多少 token
  bool thinking = false;               // Qwen3 思考模式(默认关闭,回答更直接)
  bool stream = true;                  // 逐 token 流式输出
  std::vector<std::string> questions;  // 非空 = 非交互模式,依次提问
  std::string exe_root;                // 项目根目录(解析默认相对路径用)
};

void print_usage(const char* prog) {
  std::cout
      << "用法: " << prog << " [选项]\n"
      << "  --model-type <t>     模型类型: qwen3(默认) | qwen2 | llama\n"
      << "  --model-dir <path>   HF safetensors 模型目录(默认 Qwen3-4B)\n"
      << "  --tokenizer <path>   tokenizer 路径(qwen2/qwen3 为 tokenizer.json,\n"
      << "                       llama 为 sentencepiece 模型;默认 Qwen3-4B/tokenizer.json)\n"
      << "  --device <d>         推理设备: cuda(默认) | cpu\n"
      << "  --system <text>      系统提示词\n"
      << "  --max-seq-len <N>    上下文容量(KV cache),默认 0 = 自适应(取模型窗口与\n"
      << "                       显存放得下的较小值);写死数字则按该值,超过模型窗口\n"
      << "                       时截断。历史超出预算从最旧一轮开始丢弃\n"
      << "  --max-batch <N>      调度器槽位数(默认 1);KV 池按 max_batch x max_seq_len\n"
      << "                       整块预分配,想多占显存可以调大\n"
      << "  --gpu-mem-fraction <f>  KV 池最多占空闲显存的比例(默认 0.9)\n"
      << "  --max-new-tokens <N> 每轮最大生成长度(默认 1024)\n"
      << "  --questions <q1|q2>  非交互模式:用 | 分隔多个问题,依次提问后退出\n"
      << "  --thinking           Qwen3 思考模式(默认关闭:追加空 think 块,直出答案)\n"
      << "  --no-stream          关闭流式输出,整段回复一次打印\n"
      << "  -h, --help           显示本帮助\n\n"
      << "对话中的命令:\n"
      << "  /model                         查看当前模型\n"
      << "  /model <type> <模型目录> [tokenizer]   热切换模型(会清空记忆)\n"
      << "  /history                       查看记忆轮数\n"
      << "  /clear                         清空记忆\n"
      << "  exit | quit                    退出\n";
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "参数 " << name << " 缺少取值\n";
        std::exit(1);
      }
      return argv[++i];
    };
    if (k == "--model-type") {
      a.model_type = value("--model-type");
    } else if (k == "--model-dir") {
      a.model_dir = value("--model-dir");
    } else if (k == "--tokenizer") {
      a.tokenizer = value("--tokenizer");
    } else if (k == "--device") {
      a.device = value("--device");
    } else if (k == "--system") {
      a.system_prompt = value("--system");
    } else if (k == "--max-seq-len") {
      a.max_seq_len = std::stoi(value("--max-seq-len"));
    } else if (k == "--max-batch") {
      a.max_batch = std::stoi(value("--max-batch"));
    } else if (k == "--gpu-mem-fraction") {
      a.gpu_mem_fraction = std::stod(value("--gpu-mem-fraction"));
    } else if (k == "--max-new-tokens") {
      a.max_new_tokens = std::stoi(value("--max-new-tokens"));
    } else if (k == "--questions") {
      const std::string list = value("--questions");
      size_t pos = 0;
      while (pos <= list.size()) {
        size_t bar = list.find('|', pos);
        if (bar == std::string::npos) bar = list.size();
        std::string q = list.substr(pos, bar - pos);
        if (!q.empty()) a.questions.push_back(q);
        pos = bar + 1;
      }
    } else if (k == "--thinking") {
      a.thinking = true;
    } else if (k == "--no-stream") {
      a.stream = false;
    } else if (k == "-h" || k == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "未知参数: " << k << "\n";
      print_usage(argv[0]);
      std::exit(1);
    }
  }
  return a;
}

// 默认路径解析:相对 CWD 找不到时,相对项目根目录再找。
// 适配 <root>/build/demo/chat_demo 与 <root>/demo/build/chat_demo 两种布局。
std::string resolve_default(const std::string& path, const std::string& exe_root) {
  if (exe_root.empty()) return path;
  std::ifstream probe(path);
  if (probe.good()) return path;
  std::string alt = exe_root + "/" + path;
  probe.open(alt);
  if (probe.good()) return alt;
  return path;  // 保持原值,后续加载时给出明确报错
}

// 项目根目录下现成的模型目录(config.json + tokenizer),加载失败时提示用
void list_local_models(const std::string& root) {
  if (root.empty()) return;
  std::error_code ec;
  std::vector<std::string> found;
  for (const auto& e : std::filesystem::directory_iterator(root, ec)) {
    if (!e.is_directory()) continue;
    const auto dir = e.path();
    if (!std::filesystem::exists(dir / "config.json", ec)) continue;
    const bool bpe = std::filesystem::exists(dir / "tokenizer.json", ec);
    const bool spe = std::filesystem::exists(dir / "tokenizer.model", ec);
    if (!bpe && !spe) continue;
    found.push_back(dir.filename().string() + (bpe ? " (tokenizer.json)" : " (tokenizer.model)"));
  }
  if (found.empty()) return;
  std::cout << "可选的本地模型目录(项目根目录 " << root << "):\n";
  for (const auto& f : found) std::cout << "  " << f << "\n";
}

// ============================== ChatML 模板 ==============================

struct Turn {
  std::string role;  // "user" | "assistant"
  std::string content;
};

// Qwen 系列(ChatML):<|im_start|>role\n内容<|im_end|>\n,最后补 assistant 头等模型续写。
// Qwen3 非思考模式按官方模板在 assistant 头后追加空 think 块,模型跳过推理直接作答。
std::string render_prompt(const std::string& system, const std::vector<Turn>& turns,
                          bool thinking) {
  std::string prompt;
  if (!system.empty()) {
    prompt += "<|im_start|>system\n" + system + "<|im_end|>\n";
  }
  for (const auto& t : turns) {
    prompt += "<|im_start|>" + t.role + "\n" + t.content + "<|im_end|>\n";
  }
  prompt += "<|im_start|>assistant\n";
  if (!thinking) {
    prompt += " thinking\n\n\n\n";
  }
  return prompt;
}

// 思考模式:模型输出形如 " thinking...<｜end▁of▁thinking｜>答案",历史里只保留最终答案
// (Qwen3 官方模板对历史 assistant 轮次同样只保留 content)。
std::string strip_think(const std::string& text) {
  std::string out;
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t open = text.find(" thinking", pos);
    if (open == std::string::npos) {
      out += text.substr(pos);
      break;
    }
    out += text.substr(pos, open - pos);
    const size_t close = text.find("<｜end▁of▁thinking｜>", open);
    if (close == std::string::npos) break;  // 还没想完:后半段不属于答案
    pos = close + std::string("<｜end▁of▁thinking｜>").size();
  }
  return out;
}

std::string trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// ============================== UTF-8 流式输出 ==============================

// decode() 返回原始字节:一个汉字可能被拆在相邻两个 token 里,逐 token 打印会
// 出现乱码。这里只输出完整的 UTF-8 字符,把不完整的尾巴留到下一个 token。
class Utf8Stream {
 public:
  std::string push(const std::string& bytes) {
    pending_ += bytes;
    const size_t n = complete_len(pending_);
    std::string out = pending_.substr(0, n);
    pending_.erase(0, n);
    return out;
  }

  std::string flush() {
    std::string out = pending_;
    pending_.clear();
    return out;
  }

 private:
  static size_t complete_len(const std::string& s) {
    size_t i = 0, last_complete = 0;
    while (i < s.size()) {
      const unsigned char c = static_cast<unsigned char>(s[i]);
      size_t need = 1;
      if ((c & 0x80) == 0x00) {
        need = 1;
      } else if ((c & 0xE0) == 0xC0) {
        need = 2;
      } else if ((c & 0xF0) == 0xE0) {
        need = 3;
      } else if ((c & 0xF8) == 0xF0) {
        need = 4;
      } else {
        ++i;  // 非法首字节:原样放过,避免卡死
        last_complete = i;
        continue;
      }
      if (i + need > s.size()) break;  // 尾部字符不完整,留到下次
      i += need;
      last_complete = i;
    }
    return last_complete;
  }

  std::string pending_;
};

// ============================== 推理会话 ==============================

// 模型 + 调度器(对话期间复用同一个 Scheduler,KV cache / CUDA Graph 都建一次)
struct ChatEngine {
  std::shared_ptr<model::Model> model;
  std::unique_ptr<scheduler::Scheduler> sched;
  base::DeviceType device_type = base::DeviceType::kDeviceCUDA;
  int ctx = 0;             // KV cache 容量(prompt + 生成)
  int prompt_budget = 0;   // 留给 prompt 的 token 预算
  int max_batch = 1;       // 调度器槽位数
  long long kv_bytes = 0;  // KV 池实际占用(整块预分配)
  long long vram_free = 0;
  long long vram_total = 0;

  // 每个 token 的 KV 字节数: 层数 x (K+V) x kv_dim x 元素字节(BF16=2 / FP32=4)
  long long kv_bytes_per_token() const {
    const long long elem = (model->compute_dtype() == base::DataType::kDataTypeBF16) ? 2 : 4;
    return static_cast<long long>(model->layer_num()) * 2 * model->kv_dim() * elem;
  }

  // 按 args 加载模型并建调度器;失败返回 false(已打印原因)
  bool load(const Args& args) {
    device_type = (args.device == "cpu") ? base::DeviceType::kDeviceCPU
                                         : base::DeviceType::kDeviceCUDA;
    if (args.model_type == "qwen2") {
      model = std::make_shared<model::Qwen2Model>(base::TokenizerType::kEncodeBpe, args.tokenizer,
                                                  args.model_dir, false);
    } else if (args.model_type == "qwen3") {
      model = std::make_shared<model::Qwen3Model>(base::TokenizerType::kEncodeBpe, args.tokenizer,
                                                  args.model_dir, false);
    } else if (args.model_type == "llama" || args.model_type == "llama3") {
      model = std::make_shared<model::LLamaModel>(base::TokenizerType::kEncodeSpe, args.tokenizer,
                                                  args.model_dir, false);
    } else {
      std::cerr << "未知 --model-type: " << args.model_type << " (支持 qwen3 | qwen2 | llama)\n";
      return false;
    }

    std::cout << "正在加载模型 " << args.model_dir << " ..." << std::endl;
    base::Status st = model->init(device_type);
    if (!st) {
      std::cerr << "模型加载失败: " << st.get_err_code() << "\n";
      model.reset();
      list_local_models(args.exe_root);
      return false;
    }

    // 模型自带的内部 KV cache([layers, seq_len, kv_dim])只服务于旧的单序列
    // prefill 路径(predict/forward);本示例走 Scheduler + forward_batch,用的
    // 是 KVManager 的池子,这份内部缓存是死重。窗口调到 40960 时它要占约 6GB,
    // 这里缩到最小把它还给显存池。
    model->resize_internal_kv_cache(1024);

    max_batch = args.max_batch > 0 ? args.max_batch : 1;
    ctx = args.max_seq_len > 0 ? args.max_seq_len : model->seq_len();
    if (ctx > model->seq_len()) {
      std::cout << "[提示] --max-seq-len " << ctx << " 超过模型窗口 " << model->seq_len()
                << ",按模型窗口截断\n";
      ctx = model->seq_len();
    }

    // 显存预算:KV 池 = max_batch x max_seq_len 个位置整块预分配,不能超过
    // 空闲显存的 --gpu-mem-fraction,否则直接 OOM。
    if (device_type == base::DeviceType::kDeviceCUDA) {
      size_t free_b = 0, total_b = 0;
      if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        vram_free = static_cast<long long>(free_b);
        vram_total = static_cast<long long>(total_b);
        const double budget = static_cast<double>(free_b) * args.gpu_mem_fraction;
        const long long per_token = kv_bytes_per_token() * max_batch;
        const int fit = static_cast<int>(budget / static_cast<double>(per_token));
        if (fit < ctx) {
          std::cout << "[提示] 显存只放得下 " << fit << " tokens 的 KV(空闲 "
                    << (vram_free >> 20) << " MB x " << args.gpu_mem_fraction
                    << "),上下文从 " << ctx << " 缩到 " << fit << "\n";
          ctx = fit;
        }
      }
    }

    prompt_budget = ctx - args.max_new_tokens - 16;  // 16 = 模板与裁剪误差余量
    if (prompt_budget <= 0) {
      std::cerr << "上下文容量太小: max_seq_len=" << ctx
                << " max_new_tokens=" << args.max_new_tokens << "\n";
      model.reset();
      return false;
    }

    // 调度器:负责 KV cache 分配、prefill / decode 与采样。paged 布局下池子按
    // max_batch x ceil(ctx/block_size) 个块预分配,所以 max_batch 是"占多少显存"
    // 的直接旋钮。
    //
    // 连续对话每轮重发完整历史 → 与前一轮共享长前缀,开启 prefix cache:
    // 整块复用上一轮已算好的 KV,只 prefill 新增的尾巴,TTFT 与算力都省。
    sched = std::make_unique<scheduler::Scheduler>(model, max_batch, ctx, args.max_new_tokens,
                                                   /*block_size=*/0,
                                                   /*enable_prefix_cache=*/true);
    kv_bytes = static_cast<long long>(max_batch) * ctx * kv_bytes_per_token();
    return true;
  }

  // 卸载:先释放调度器(持有 KV cache)再释放模型,并把显存池归还给驱动,
  // 否则热切换模型时旧模型占着显存容易 OOM。
  void unload() {
    sched.reset();
    model.reset();
    if (device_type == base::DeviceType::kDeviceCUDA) {
      base::CUDADeviceAllocatorFactory::get_instance()->free_idle();
    }
  }
};

struct Generation {
  std::string text;     // 生成文本(已去掉停止符)
  int num_tokens = 0;
  double ttft_ms = 0;   // 首 token 延迟(含 prefill)
  double total_ms = 0;  // 端到端
  // 本轮从 prefix cache 复用的整块数:这些 prompt token 无需重新 prefill。
  int64_t shared_blocks = 0;
};

// 跑一次请求:scheduler 一步一个 decode token 地推进,边跑边把新增 token 打出来。
Generation generate(const ChatEngine& engine, const std::vector<int32_t>& prompt_tokens,
                    bool stream) {
  Generation gen;
  const auto& model = engine.model;
  scheduler::Scheduler& sched = *engine.sched;
  const int64_t matched_before = sched.prefix_cache_stats().matched_blocks;
  const int seq_id = sched.add_request(prompt_tokens);
  if (seq_id < 0) {
    std::cerr << "[错误] 请求被调度器拒绝(prompt 放不进 KV cache)\n";
    return gen;
  }

  auto lookup = [&]() -> const scheduler::Sequence* {
    for (const auto& s : sched.get_running()) {
      if (s.id == seq_id) return &s;
    }
    for (const auto& s : sched.get_finished()) {
      if (s.id == seq_id) return &s;
    }
    return nullptr;
  };

  size_t printed = 0;
  Utf8Stream utf8;
  auto pump = [&](const scheduler::Sequence& s) {
    for (size_t k = printed; k < s.generated_tokens.size(); ++k) {
      const int32_t tok = s.generated_tokens[k];
      if (model->is_sentence_ending(tok)) continue;  // <|im_end|> 等停止符不显示
      const std::string piece = model->decode(std::vector<int32_t>{tok});
      if (stream) std::cout << utf8.push(piece) << std::flush;
    }
    printed = s.generated_tokens.size();
  };

  while (!sched.all_finished()) {
    sched.step();
    if (const scheduler::Sequence* s = lookup()) pump(*s);
  }
  const scheduler::Sequence* done = lookup();
  if (done == nullptr) return gen;
  pump(*done);  // 收尾:最后一步的 token 在序列转入 finished 之后才可见
  if (stream) std::cout << utf8.flush() << std::flush;

  // 回复文本整段重解一遍:逐 token 拼接在字节层面等价,整段解码更稳妥
  std::vector<int32_t> ids;
  for (int32_t t : done->generated_tokens) {
    if (!model->is_sentence_ending(t)) ids.push_back(t);
  }
  gen.text = model->decode(ids);
  gen.num_tokens = static_cast<int>(ids.size());
  // 本 demo 一次只跑一个请求,所以统计增量就是本轮的命中块数。
  gen.shared_blocks = sched.prefix_cache_stats().matched_blocks - matched_before;
  // 延迟打点取自序列本身(提交 → 首个 token / 结束)
  if (done->num_generated_tokens > 0 && done->first_token_time > done->arrival_time) {
    gen.ttft_ms = ms_between(done->arrival_time, done->first_token_time);
    gen.total_ms = ms_between(done->arrival_time, done->finish_time);
  } else {
    gen.total_ms = ms_between(done->arrival_time, Clock::now());
  }
  return gen;
}

int sched_block_size(const ChatEngine& engine) {
  return engine.sched ? engine.sched->get_block_size() : 0;
}

void print_stats(const Generation& gen, int block_size) {
  const double tps = gen.total_ms > 0 ? gen.num_tokens * 1000.0 / gen.total_ms : 0.0;
  std::cout << "[TTFT " << static_cast<long long>(gen.ttft_ms) << " ms | 生成 "
            << gen.num_tokens << " tokens | " << static_cast<long long>(gen.total_ms)
            << " ms | " << static_cast<long long>(tps) << " tok/s";
  if (gen.shared_blocks > 0) {
    std::cout << " | 命中缓存 " << gen.shared_blocks << " 块(跳过 "
              << gen.shared_blocks * block_size << " tokens prefill)";
  }
  std::cout << "]\n";
}

// 会话结束时的 prefix cache 汇总:多轮对话的命中率即"这一轮有多少 prompt
// 是上一轮已经算过的"。
void print_prefix_summary(const scheduler::Scheduler& sched) {
  if (!sched.prefix_cache_enabled()) return;
  const scheduler::PrefixCacheStats& st = sched.prefix_cache_stats();
  std::printf("prefix cache: lookups=%lld hits=%lld hit_rate=%.1f%% matched_blocks=%lld "
              "inserts=%lld evictions=%lld (block=%d tokens)\n",
              static_cast<long long>(st.lookups), static_cast<long long>(st.hits),
              st.hit_rate() * 100.0, static_cast<long long>(st.matched_blocks),
              static_cast<long long>(st.inserts), static_cast<long long>(st.evictions),
              sched.get_block_size());
}

void print_commands() {
  std::cout << "[命令] /model 查看或切换模型 | /history 记忆轮数 | /clear 清空记忆 | "
               "exit 退出\n"
            << "输入问题回车发送(也可以从管道喂入多行,便于脚本化)\n"
            << "====================================================\n";
}

std::string gb(long long bytes) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(bytes) / (1024.0 * 1024 * 1024));
  return buf;
}

void print_banner(const ChatEngine& engine, const Args& args) {
  std::cout << "=========== 连续对话 Demo ===========\n"
            << "模型: " << args.model_dir << " (" << args.model_type << ")"
            << "  设备: " << (engine.device_type == base::DeviceType::kDeviceCUDA ? "cuda" : "cpu")
            << "\n上下文: " << engine.ctx << " tokens"
            << "(模型窗口 " << engine.model->seq_len() << ",每轮最多生成 "
            << args.max_new_tokens << ")" << "\n"
            << "KV 池: " << engine.max_batch << " 槽 x " << engine.ctx << " tokens x "
            << (engine.kv_bytes_per_token() / 1024) << " KB/token = " << gb(engine.kv_bytes)
            << "(整块预分配)\n"
            << "记忆: 开启(每轮携带全部历史,超出上下文时丢弃最旧的一轮)\n"
            << "前缀复用(prefix cache): 开启(每轮重发完整历史,"
               "复用的是上一轮已经算好的整块 KV)\n";
  if (engine.device_type == base::DeviceType::kDeviceCUDA) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
      const long long total = static_cast<long long>(total_b);
      const long long free_now = static_cast<long long>(free_b);
      std::cout << "显存: 已用 " << gb(total - free_now) << " / " << gb(total) << "(空闲 "
                << gb(free_now) << ")\n";
    }
  }
  print_commands();
}

// 预热:首次 forward 含 CUDA Graph 捕获与句柄初始化,不预热会算进首轮 TTFT
void warmup(const ChatEngine& engine, const Args& args) {
  Clock::time_point t0 = Clock::now();
  std::vector<Turn> warm = {{"user", "你好"}};
  generate(engine, engine.model->encode(render_prompt(args.system_prompt, warm, args.thinking)),
           /*stream=*/false);
  std::cout << "预热完成(" << static_cast<long long>(ms_between(t0, Clock::now())) << " ms)\n";
}

// ============================== 主流程 ==============================

int run(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  // ---- 从可执行文件路径推断项目根目录,用于解析默认相对路径 ----
  {
    std::filesystem::path p(argv[0]);
    if (p.has_parent_path()) {
      std::error_code ec;
      auto root = std::filesystem::canonical(p.parent_path() / "../..", ec);
      if (ec) {
        ec.clear();
        root = std::filesystem::canonical(p.parent_path(), ec);
      }
      if (!ec) args.exe_root = root.string();
    }
  }
  args.model_dir = resolve_default(args.model_dir, args.exe_root);
  args.tokenizer = resolve_default(args.tokenizer, args.exe_root);

  ChatEngine engine;
  if (!engine.load(args)) return 1;
  print_banner(engine, args);
  warmup(engine, args);

  std::vector<Turn> history;  // 交替保存的 [user, assistant, user, assistant, ...]
  size_t next_question = 0;

  while (true) {
    // ---- 取用户输入(实时交互读 stdin;--questions 模式依次取预设问题)----
    std::string user_input;
    if (!args.questions.empty()) {
      if (next_question >= args.questions.size()) break;
      user_input = args.questions[next_question++];
      std::cout << "\n你: " << user_input << std::endl;
    } else {
      std::cout << "\n你: " << std::flush;
      if (!std::getline(std::cin, user_input)) break;  // Ctrl-D 结束
      if (!isatty(fileno(stdin))) {
        std::cout << user_input << "\n";  // 管道/重定向时终端不回显,这里补上
      }
      user_input = trim(user_input);
      if (user_input.empty()) continue;
      if (user_input == "exit" || user_input == "quit" || user_input == "q") break;
      if (user_input == "/clear") {
        history.clear();
        std::cout << "[已清空记忆]\n";
        continue;
      }
      if (user_input == "/history") {
        std::cout << "[当前记忆 " << history.size() / 2 << " 轮对话]\n";
        continue;
      }
      if (user_input == "/help") {
        print_commands();
        continue;
      }
      if (user_input.rfind("/model", 0) == 0) {
        // /model 查看当前模型;/model <type> <模型目录> [tokenizer] 热切换
        std::istringstream iss(user_input);
        std::string cmd, type, dir, tokenizer;
        iss >> cmd >> type >> dir >> tokenizer;
        if (dir.empty()) {
          std::cout << "[当前模型 " << args.model_type << " @ " << args.model_dir
                    << ",上下文 " << engine.ctx << " tokens]\n"
                    << "切换用法: /model <qwen3|qwen2|llama> <模型目录> [tokenizer 路径]\n";
          continue;
        }
        Args want = args;
        want.model_type = type;
        want.model_dir = resolve_default(dir, args.exe_root);
        if (!tokenizer.empty()) {
          want.tokenizer = resolve_default(tokenizer, args.exe_root);
        } else if (want.model_dir != args.model_dir) {
          // 没给 tokenizer 时按模型目录里常见的文件名找一个
          const std::string bpe = want.model_dir + "/tokenizer.json";
          const std::string spe = want.model_dir + "/tokenizer.model";
          std::ifstream probe(bpe);
          want.tokenizer = probe.good() ? bpe : spe;
        }
        std::cout << "[切换模型 " << want.model_dir << " (" << want.model_type << "),清空记忆]\n";
        engine.unload();
        if (!engine.load(want)) {
          std::cerr << "[切换失败,回滚到 " << args.model_dir << "]\n";
          if (!engine.load(args)) return 1;  // 旧模型也加载不回来就退出
          continue;
        }
        args = want;
        history.clear();
        print_banner(engine, args);
        warmup(engine, args);
        continue;
      }
    }

    // ---- 拼本轮 prompt:记忆 + 本轮提问,按 token 数裁剪到预算内 ----
    std::vector<Turn> turns = history;
    turns.push_back({"user", user_input});

    std::vector<Turn> kept;
    kept.push_back(turns.back());  // 当前提问永远保留
    int dropped = 0;
    for (int i = static_cast<int>(turns.size()) - 2; i >= 1; i -= 2) {
      std::vector<Turn> candidate = {turns[i - 1], turns[i]};  // 一整轮 user + assistant
      candidate.insert(candidate.end(), kept.begin(), kept.end());
      if (static_cast<int>(engine.model->encode(render_prompt(args.system_prompt, candidate,
                                                              args.thinking)).size()) <=
          engine.prompt_budget) {
        kept = std::move(candidate);
      } else {
        dropped = (i + 1) / 2;
        break;
      }
    }
    if (dropped > 0) {
      std::cout << "[记忆已满:丢弃最早的 " << dropped << " 轮对话]\n";
    }

    // ---- 生成 ----
    const std::vector<int32_t> prompt_tokens =
        engine.model->encode(render_prompt(args.system_prompt, kept, args.thinking));
    if (static_cast<int>(prompt_tokens.size()) > engine.prompt_budget) {
      // 裁剪只丢历史,当前提问本身太长时只能让用户改写
      std::cout << "[本轮提问太长:prompt 需要 " << prompt_tokens.size() << " tokens,超出预算 "
                << engine.prompt_budget << " tokens,请缩短提问]\n";
      continue;
    }
    std::cout << "\nAI: " << std::flush;
    const Generation gen = generate(engine, prompt_tokens, args.stream);
    std::cout << "\n";
    std::cout << "[prompt " << prompt_tokens.size() << " tokens] ";
    print_stats(gen, sched_block_size(engine));

    // ---- 回复写回记忆,下一轮就是"连续对话" ----
    const std::string reply = trim(strip_think(gen.text));
    if (reply.empty()) {
      std::cout << "[本轮没有生成内容,跳过写入记忆]\n";
      continue;
    }
    history.push_back({"user", user_input});
    history.push_back({"assistant", reply});
  }

  std::cout << "\n";
  print_prefix_summary(*engine.sched);
  std::cout << "再见。\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::GLOG_WARNING);
  return run(argc, argv);
}
