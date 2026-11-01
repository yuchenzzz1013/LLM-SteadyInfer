// ============================================================================
// Weight-quantization fidelity eval (reference BF16 model vs AWQ W4A16 model)
// as specified by eval.md:
//
//   dataset : Salesforce/wikitext -> wikitext-2-raw-v1, split=test
//             (already downloaded as wikitext-2-raw-v1-test.raw)
//   preproc : keep blank lines, join lines with "\n", encode, then cut into
//             NON-OVERLAPPING `seq_len`-token chunks (512 by default). The
//             final partial chunk is dropped, exactly like the reference
//             protocol (565 chunks -> 288,715 scored positions = 565*511).
//   metrics : Mean KLD, 99.9% KLD, Same Top-1, PPL Change
//             KLD direction = KL(reference || quantized)
//
// Teacher forcing: for chunk s and position t (0 <= t < seq_len-1) the model
// is fed tokens s[0..t] and its logits must rank s[t+1]. This harness drives
// `Model::decode_step` one token at a time (one batch row per chunk), so every
// scored position gets a real logit vector; the final token of a chunk has no
// successor and is not scored. The four metrics therefore all come from the
// same set of positions, only the aggregation differs.
//
// Why two passes (and a logits cache):
//   Qwen3-14B BF16 is ~29.5 GB of weights and its AWQ checkpoint ~10 GB. Both
//   models plus KV cache / scratch / the [batch, vocab] logits buffer do not
//   fit on a 40 GB A100. Pass 1 therefore runs the reference model and streams
//   its logits per position to a cache file ([chunks, seq_len-1, vocab] bf16,
//   chunk-major); pass 2 runs the quantized model and compares each candidate
//   row against the cached reference row with wq_eval_kernel.cu. A 512-token
//   chunk x 152K vocab row is 304 KB, so per-step cache I/O is far below the
//   decode step time; the full WikiText-2 cache is ~88 GB (the harness prints
//   the size and takes --max-chunks for a cheaper smoke run).
//
// KV cache is always BF16 (KV quantization off) so this benchmark measures
// *weight* quantization only.
//
// Usage:
//   ./weight_quantization_eval <ref-model-dir> <quant-model-dir> [dataset.raw]
//       [--tokenizer tokenizer.json] [--model-type qwen3|qwen2|llama]
//       [--seq-len 512] [--batch 16] [--max-chunks N]
//       [--cache <path>] [--stage all|ref|quant] [--block-size 16]
//       [--csv out.csv] [--tag label]
//
//   --stage all   : reference pass, free the model, quant pass (default)
//   --stage ref   : only build the reference logits cache
//   --stage quant : only run the candidate pass against an existing cache
//   (ref/quant split lets a machine that cannot reload in-process use two
//    invocations with the same --cache.)
// ============================================================================

#include <base/alloc.h>
#include <base/base.h>
#include <glog/logging.h>
#include "model/llama3.h"
#include "model/qwen2.h"
#include "model/qwen3.h"
#include "scheduler/kv_manager.h"
#include "tensor/tensor.h"

#include "../bench_common.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// wq_eval_kernel.cu — device-side KLD / top-1 / NLL reduction (see that file)
void wq_eval_cu(const void* ref_logits_bf16, const void* cand_logits_bf16,
                const int32_t* targets, int32_t vocab, int32_t rows, float* out_kl,
                int32_t* same_top1, float* nll_ref, float* nll_cand, void* stream);

namespace {

// ---------------------------- cache file layout ----------------------------
// Header followed by `num_chunks` chunk records. A chunk record is
// (seq_len-1) consecutive [vocab] bf16 rows, position-major: position t at
// offset t*vocab*2. Chunk-major keeps both passes' I/O sequential.
struct CacheHeader {
  char magic[8];  // "WQEVAL1" + NUL
  uint32_t version;
  int32_t vocab;
  int32_t seq_len;
  int32_t num_chunks;
  int32_t scored_per_chunk;  // seq_len - 1
  int32_t dtype;             // 0 = bf16 logits
  int32_t reserved0;
  uint64_t tokens_hash;  // FNV-1a over the scored token prefix
  int64_t total_bytes;
  int64_t reserved[5];
};
static_assert(sizeof(CacheHeader) == 96, "cache header must stay 96 bytes");

constexpr uint32_t kCacheVersion = 1;
constexpr int32_t kCacheDtypeBf16 = 0;

struct Args {
  std::string ref_model;
  std::string quant_model;
  std::string dataset = "wikitext-2-raw-v1-test.raw";
  std::string tokenizer;
  std::string model_type = "qwen3";
  std::string cache = "/tmp/weight_quantization_eval_ref_logits.bin";
  std::string stage = "all";
  std::string csv;
  std::string tag = "weight_quant";
  int seq_len = 512;
  int batch = 16;
  int max_chunks = 0;  // 0 = all complete chunks
  int block_size = 16;
};

struct Metrics {
  double kl_sum = 0.0;
  int64_t count = 0;
  int64_t top1_match = 0;
  double nll_ref_sum = 0.0;
  double nll_cand_sum = 0.0;
  std::vector<float> kl_values;  // one per scored position (for 99.9%)

  // Per-chunk aggregation (CSV diagnostics).
  std::vector<double> chunk_kl_sum;
  std::vector<double> chunk_nll_ref;
  std::vector<double> chunk_nll_cand;
  std::vector<int64_t> chunk_count;
  std::vector<int64_t> chunk_match;
};

std::string join_path(const std::string& dir, const std::string& name) {
  return (std::filesystem::path(dir) / name).string();
}

bool file_exists(const std::string& path) {
  std::error_code ec;
  return std::filesystem::exists(path, ec);
}

// Read the whole raw WikiText file and normalize line endings. eval.md says
// "keep blank lines, join lines with \n"; the raw file already is newline
// separated, so preserving the bytes (minus CRLF/BOM) is equivalent.
std::string read_dataset_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    LOG(FATAL) << "无法打开数据集: " << path;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  std::string text = ss.str();
  // Strip a UTF-8 BOM if present.
  if (text.size() >= 3 && static_cast<uint8_t>(text[0]) == 0xEF &&
      static_cast<uint8_t>(text[1]) == 0xBB && static_cast<uint8_t>(text[2]) == 0xBF) {
    text.erase(0, 3);
  }
  // Normalize CRLF/CR to LF without touching blank lines.
  std::string out;
  out.reserve(text.size());
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\r') {
      if (i + 1 < text.size() && text[i + 1] == '\n') continue;  // drop CR of CRLF
      out.push_back('\n');                                       // lone CR -> LF
    } else {
      out.push_back(text[i]);
    }
  }
  return out;
}

uint64_t fnv1a_u64(const void* data, size_t bytes, uint64_t h = 1469598103934665603ULL) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

// Index model dir/config.json for the checkpoint's weight-quantization method.
std::string read_quant_method(const std::string& model_dir) {
  std::ifstream f(join_path(model_dir, "config.json"));
  if (!f.is_open()) return "";
  try {
    auto j = bench::json::parse(f);
    if (j.contains("quantization_config") && j["quantization_config"].is_object()) {
      const auto& q = j["quantization_config"];
      if (q.contains("quant_method")) return q.value("quant_method", "");
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "解析 " << model_dir << "/config.json 失败: " << e.what();
  }
  return "";
}

std::shared_ptr<model::Model> load_model(const std::string& model_type,
                                         const std::string& tokenizer,
                                         const std::string& model_dir, bool is_quant_model) {
  using base::TokenizerType;
  std::shared_ptr<model::Model> m;
  if (model_type == "qwen2") {
    m = std::make_shared<model::Qwen2Model>(TokenizerType::kEncodeBpe, tokenizer, model_dir,
                                            is_quant_model);
  } else if (model_type == "qwen3") {
    m = std::make_shared<model::Qwen3Model>(TokenizerType::kEncodeBpe, tokenizer, model_dir,
                                            is_quant_model);
  } else if (model_type == "llama" || model_type == "llama3") {
    m = std::make_shared<model::LLamaModel>(TokenizerType::kEncodeSpe, tokenizer, model_dir,
                                            is_quant_model);
  } else {
    LOG(FATAL) << "未知 --model-type: " << model_type;
  }
  auto st = m->init(base::DeviceType::kDeviceCUDA);
  if (!st) LOG(FATAL) << "模型初始化失败 (" << model_dir << "): " << st.get_err_code();
  if (m->compute_dtype() != base::DataType::kDataTypeBF16) {
    LOG(FATAL) << "只支持 BF16 计算路径(CUDA); " << model_dir << " 不是 BF16";
  }
  return m;
}

// Encode the dataset and cut it into non-overlapping `seq_len`-token chunks.
// A partial trailing chunk is dropped (see the file header).
void tokenize_chunks(const std::shared_ptr<model::Model>& m, const std::string& dataset,
                     int seq_len, int max_chunks,
                     std::vector<std::vector<int32_t>>& chunks, uint64_t& tokens_hash) {
  std::string text = read_dataset_text(dataset);
  std::vector<int32_t> tokens = m->encode(text);
  text.clear();
  text.shrink_to_fit();
  if (tokens.empty()) LOG(FATAL) << "语料编码后为空: " << dataset;

  const size_t complete = tokens.size() / static_cast<size_t>(seq_len);
  size_t use = complete;
  if (max_chunks > 0) use = std::min<size_t>(use, static_cast<size_t>(max_chunks));
  if (use == 0) {
    LOG(FATAL) << "语料只有 " << tokens.size() << " token,不足一个 " << seq_len
               << "-token chunk";
  }
  const size_t remainder = tokens.size() - use * static_cast<size_t>(seq_len);
  chunks.clear();
  chunks.reserve(use);
  for (size_t c = 0; c < use; ++c) {
    const int32_t* base = tokens.data() + c * static_cast<size_t>(seq_len);
    chunks.emplace_back(base, base + seq_len);
  }
  // Hash only the tokens that are actually scored, so a cache built with a
  // different --max-chunks is detected instead of silently misaligned.
  tokens_hash = fnv1a_u64(tokens.data(), use * static_cast<size_t>(seq_len) * sizeof(int32_t));
  std::cout << "CORPUS bytes=" << std::filesystem::file_size(dataset) << " tokens=" << tokens.size()
            << " complete_chunks=" << complete << " used_chunks=" << use
            << " remainder_tokens=" << remainder << " seq_len=" << seq_len << "\n";
}

// --------------------------- cache file helpers ----------------------------
void pwrite_all(int fd, const void* data, size_t bytes, int64_t off) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  while (bytes > 0) {
    const ssize_t n = ::pwrite(fd, p, bytes, static_cast<off_t>(off));
    if (n < 0) {
      if (errno == EINTR) continue;
      LOG(FATAL) << "pwrite 失败: " << std::strerror(errno);
    }
    p += n;
    bytes -= static_cast<size_t>(n);
    off += n;
  }
}

void pread_all(int fd, void* data, size_t bytes, int64_t off) {
  uint8_t* p = static_cast<uint8_t*>(data);
  while (bytes > 0) {
    const ssize_t n = ::pread(fd, p, bytes, static_cast<off_t>(off));
    if (n < 0) {
      if (errno == EINTR) continue;
      LOG(FATAL) << "pread 失败: " << std::strerror(errno);
    }
    if (n == 0) LOG(FATAL) << "缓存文件被截断 (offset=" << off << ")";
    p += n;
    bytes -= static_cast<size_t>(n);
    off += n;
  }
}

int64_t cache_data_bytes(const CacheHeader& h) {
  return static_cast<int64_t>(h.num_chunks) * h.scored_per_chunk * h.vocab * 2;
}

CacheHeader make_header(int32_t vocab, int32_t seq_len, int32_t num_chunks,
                        uint64_t tokens_hash) {
  CacheHeader h{};
  std::memcpy(h.magic, "WQEVAL1", 7);
  h.version = kCacheVersion;
  h.vocab = vocab;
  h.seq_len = seq_len;
  h.num_chunks = num_chunks;
  h.scored_per_chunk = seq_len - 1;
  h.dtype = kCacheDtypeBf16;
  h.tokens_hash = tokens_hash;
  h.total_bytes = static_cast<int64_t>(sizeof(CacheHeader)) + cache_data_bytes(h);
  return h;
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

double mib(double bytes) { return bytes / (1024.0 * 1024.0); }

void log_gpu_mem(const std::string& when) {
  size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
    cudaGetErrorString(cudaGetLastError());
    return;
  }
  std::cout << "GPU_MEM " << when << " free_mb=" << std::fixed << std::setprecision(0)
            << mib(free_b) << " total_mb=" << mib(total_b) << std::defaultfloat << "\n";
}

// Release everything the (destroyed) model allocated back to the driver. The
// runtime allocator is a caching pool, so dropping the last buffer alone only
// marks region idle; free_idle() hands fully idle cudaMalloc regions back.
void release_gpu_cache() {
  cudaDeviceSynchronize();
  base::CUDADeviceAllocatorFactory::get_instance()->free_idle();
  log_gpu_mem("after_release");
}

// ----------------------------- decode driver ------------------------------
// One batch row per chunk; each step advances every row by one token (teacher
// forcing, positions synchronized). `on_step` receives the freshly produced
// [rows, vocab] BF16 logits after the step's stream has been drained.
// `targets_dev` is the current step's [rows] int32 target tensor on the device,
// already uploaded before decode_step.
using StepFn = std::function<void(int chunk0, int rows, int t, const tensor::Tensor& logits,
                                  const int32_t* targets_dev, cudaStream_t stream)>;

void drive_chunks(const std::shared_ptr<model::Model>& m,
                  const std::vector<std::vector<int32_t>>& chunks, int batch, int seq_len,
                  int block_size, const StepFn& on_step) {
  const int vocab = m->vocab_size();
  const int layers = m->layer_num();
  const int kv_dim = m->kv_dim();
  const int num_chunks = static_cast<int>(chunks.size());

  auto kv_alloc = base::CUDADeviceAllocatorFactory::get_instance();
  auto host_alloc = std::shared_ptr<base::DeviceAllocator>(
      base::PinnedCPUDeviceAllocatorFactory::get_instance());

  // BF16 KV pool: KV quantization is deliberately off so only the weights
  // differ between the two passes.
  scheduler::KVManager kv(layers, batch, seq_len, kv_dim, m->kv_head_num(), kv_alloc,
                          base::DeviceType::kDeviceCUDA, block_size,
                          scheduler::kKvQuantNone);
  const int table_stride = kv.max_blocks_per_seq();
  scheduler::BlockAllocator* ba = kv.block_allocator();

  std::vector<int32_t> rows(batch, -1);
  for (int r = 0; r < batch; ++r) {
    rows[r] = ba->allocate_blocks(table_stride);
    if (rows[r] < 0) {
      LOG(FATAL) << "KV block 池不足: 第 " << r << " 行分配失败 (blocks=" << ba->num_blocks()
                 << ", stride=" << table_stride << ")";
    }
  }

  tensor::Tensor host_input_ids(base::DataType::kDataTypeInt32, batch, true, host_alloc);
  tensor::Tensor host_positions(base::DataType::kDataTypeInt32, batch, true, host_alloc);
  tensor::Tensor host_block_table(base::DataType::kDataTypeInt32, batch, table_stride, true,
                                  host_alloc);
  tensor::Tensor host_targets(base::DataType::kDataTypeInt32, batch, true, host_alloc);
  tensor::Tensor logits(base::DataType::kDataTypeBF16, batch, vocab, true, kv_alloc);
  tensor::Tensor targets_dev(base::DataType::kDataTypeInt32, batch, true, kv_alloc);
  if (host_input_ids.is_empty() || host_positions.is_empty() || host_block_table.is_empty() ||
      host_targets.is_empty() || logits.is_empty() || targets_dev.is_empty()) {
    LOG(FATAL) << "pinned/设备暂存分配失败";
  }

  cudaStream_t stream = m->cuda_stream();
  const double t0 = now_s();
  for (int g = 0; g < num_chunks; g += batch) {
    const int b = std::min(batch, num_chunks - g);

    // Fresh KV for this group (same allocation path for every group).
    for (int r = 0; r < b; ++r) {
      ba->free_all(rows[r]);
      rows[r] = ba->allocate_blocks(table_stride);
      if (rows[r] < 0) LOG(FATAL) << "第 " << g << " 组重新分配 KV block 失败";
      ba->copy_block_table_row(rows[r], host_block_table.ptr<int32_t>() + r * table_stride,
                               table_stride);
    }

    tensor::Tensor ids_view = host_input_ids;
    tensor::Tensor pos_view = host_positions;
    tensor::Tensor bt_view = host_block_table;
    tensor::Tensor tgt_view = host_targets;
    tensor::Tensor logits_view = logits;
    ids_view.reshape({b});
    pos_view.reshape({b});
    bt_view.reshape({b, table_stride});
    tgt_view.reshape({b});
    logits_view.reshape({b, vocab});

    for (int t = 0; t + 1 < seq_len; ++t) {
      for (int r = 0; r < b; ++r) {
        ids_view.index<int32_t>(r) = chunks[g + r][t];
        pos_view.index<int32_t>(r) = t;
        tgt_view.index<int32_t>(r) = chunks[g + r][t + 1];
      }
      cudaMemcpyAsync(targets_dev.ptr<int32_t>(), tgt_view.ptr<int32_t>(),
                      static_cast<size_t>(b) * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

      const int table_cols = std::min(t / block_size + 1, table_stride);
      auto st = m->decode_step(ids_view, pos_view, bt_view, kv.key_cache(), kv.value_cache(),
                               logits_view, table_cols, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr);
      if (!st) LOG(FATAL) << "decode_step 失败 @group=" << g << " t=" << t << ": "
                          << st.get_err_code();

      // decode_step stages host inputs into graph-captured H2D upload nodes.
      // Without draining here the host would overwrite the staging before the
      // device consumes it (the scheduler masks this by doing a D2H sample
      // copy every step; this harness deliberately does not).
      cudaStreamSynchronize(stream);

      on_step(g, b, t, logits_view, targets_dev.ptr<int32_t>(), stream);
    }

    const double elapsed = now_s() - t0;
    const int done = g + b;
    const double eta = done > 0 ? elapsed * (num_chunks - done) / done : 0.0;
    std::cout << "PASS_CHUNKS " << done << "/" << num_chunks << " elapsed_s=" << std::fixed
              << std::setprecision(1) << elapsed << " eta_s=" << eta << std::defaultfloat
              << std::endl;
  }
}

// ------------------------------ pass 1: ref -------------------------------
void run_ref_pass(const std::shared_ptr<model::Model>& m,
                  const std::vector<std::vector<int32_t>>& chunks, const Args& a,
                  const std::string& cache_path, uint64_t tokens_hash) {
  const int vocab = m->vocab_size();
  const int num_chunks = static_cast<int>(chunks.size());

  CacheHeader h = make_header(vocab, a.seq_len, num_chunks, tokens_hash);
  if (h.total_bytes < 0) LOG(FATAL) << "缓存大小溢出";
  const int fd = ::open(cache_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0) LOG(FATAL) << "无法创建缓存文件 " << cache_path << ": " << std::strerror(errno);
  pwrite_all(fd, &h, sizeof(h), 0);
  if (::ftruncate(fd, static_cast<off_t>(h.total_bytes)) != 0) {
    LOG(FATAL) << "无法预分配缓存文件 " << cache_path << " (" << mib(h.total_bytes)
               << " MiB): " << std::strerror(errno)
               << " —— 用 --max-chunks 缩小规模或把 --cache 放到更大的磁盘";
  }
  std::cout << "CACHE write=" << cache_path << " bytes=" << h.total_bytes << " ("
            << std::fixed << std::setprecision(2) << (h.total_bytes / (1024.0 * 1024.0 * 1024.0))
            << " GiB)\n"
            << std::defaultfloat;

  auto host_alloc = std::shared_ptr<base::DeviceAllocator>(
      base::PinnedCPUDeviceAllocatorFactory::get_instance());
  const int batch = a.batch;
  tensor::Tensor stage(base::DataType::kDataTypeBF16, batch, vocab, true, host_alloc);
  if (stage.is_empty()) LOG(FATAL) << "pinned logits 暂存分配失败";

  const int64_t chunk_bytes = static_cast<int64_t>(a.seq_len - 1) * vocab * 2;
  const int64_t data_off = sizeof(CacheHeader);

  drive_chunks(m, chunks, batch, a.seq_len, a.block_size,
               [&](int chunk0, int rows, int t, const tensor::Tensor& logits, const int32_t*,
                   cudaStream_t stream) {
                 const size_t bytes = static_cast<size_t>(rows) * vocab * sizeof(uint16_t);
                 cudaMemcpyAsync(stage.ptr<uint16_t>(), logits.ptr<uint16_t>(), bytes,
                                 cudaMemcpyDeviceToHost, stream);
                 if (cudaStreamSynchronize(stream) != cudaSuccess) {
                   LOG(FATAL) << "参考 logits D2H 失败: "
                              << cudaGetErrorString(cudaGetLastError());
                 }
                 for (int r = 0; r < rows; ++r) {
                   const int64_t off = data_off +
                                       static_cast<int64_t>(chunk0 + r) * chunk_bytes +
                                       static_cast<int64_t>(t) * vocab * 2;
                   pwrite_all(fd, stage.ptr<uint16_t>() + static_cast<int64_t>(r) * vocab,
                              static_cast<size_t>(vocab) * 2, off);
                 }
               });

  if (::fsync(fd) != 0) LOG(WARNING) << "fsync(" << cache_path << ") 失败: " << std::strerror(errno);
  ::close(fd);

  // The reference PPL needs a second pass (it is computed from the cached
  // logits together with the candidate); a ref-only run can report nothing
  // but the cache metadata.
  std::cout << "REF_PASS done cache=" << cache_path << " chunks=" << num_chunks
            << " positions=" << static_cast<int64_t>(num_chunks) * (a.seq_len - 1) << "\n";
}

// ----------------------------- pass 2: quant ------------------------------
void run_quant_pass(const std::shared_ptr<model::Model>& m,
                    const std::vector<std::vector<int32_t>>& chunks, const Args& a,
                    const std::string& cache_path, uint64_t tokens_hash) {
  const int vocab = m->vocab_size();
  const int num_chunks = static_cast<int>(chunks.size());

  const int fd = ::open(cache_path.c_str(), O_RDONLY);
  if (fd < 0) {
    LOG(FATAL) << "无法打开参考 logits 缓存 " << cache_path << ": " << std::strerror(errno)
               << " —— 先用 --stage ref 生成";
  }
  CacheHeader h{};
  pread_all(fd, &h, sizeof(h), 0);
  if (std::memcmp(h.magic, "WQEVAL1", 7) != 0 || h.version != kCacheVersion) {
    LOG(FATAL) << "缓存文件头无效: " << cache_path;
  }
  if (h.vocab != vocab || h.seq_len != a.seq_len || h.num_chunks != num_chunks ||
      h.scored_per_chunk != a.seq_len - 1 || h.dtype != kCacheDtypeBf16) {
    LOG(FATAL) << "缓存与当前配置不匹配: cache(vocab=" << h.vocab << " seq_len=" << h.seq_len
               << " chunks=" << h.num_chunks << ") vs run(vocab=" << vocab
               << " seq_len=" << a.seq_len << " chunks=" << num_chunks << ")";
  }
  if (h.tokens_hash != tokens_hash) {
    LOG(FATAL) << "缓存 token 序列与当前数据集/分词不一致 (hash mismatch); 用同一份数据集和"
                  " --max-chunks 重新生成 --stage ref";
  }
  std::cout << "CACHE read=" << cache_path << " chunks=" << h.num_chunks
            << " positions=" << static_cast<int64_t>(h.num_chunks) * h.scored_per_chunk << "\n";

  auto host_alloc = std::shared_ptr<base::DeviceAllocator>(
      base::PinnedCPUDeviceAllocatorFactory::get_instance());
  const int batch = a.batch;
  tensor::Tensor stage(base::DataType::kDataTypeBF16, batch, vocab, true, host_alloc);
  tensor::Tensor ref_dev(base::DataType::kDataTypeBF16, batch, vocab, true,
                         base::CUDADeviceAllocatorFactory::get_instance());
  tensor::Tensor kl_dev(base::DataType::kDataTypeFp32, batch, true,
                        base::CUDADeviceAllocatorFactory::get_instance());
  tensor::Tensor same_dev(base::DataType::kDataTypeInt32, batch, true,
                          base::CUDADeviceAllocatorFactory::get_instance());
  tensor::Tensor nll_ref_dev(base::DataType::kDataTypeFp32, batch, true,
                             base::CUDADeviceAllocatorFactory::get_instance());
  tensor::Tensor nll_cand_dev(base::DataType::kDataTypeFp32, batch, true,
                              base::CUDADeviceAllocatorFactory::get_instance());
  tensor::Tensor kl_host(base::DataType::kDataTypeFp32, batch, true, host_alloc);
  tensor::Tensor same_host(base::DataType::kDataTypeInt32, batch, true, host_alloc);
  tensor::Tensor nll_ref_host(base::DataType::kDataTypeFp32, batch, true, host_alloc);
  tensor::Tensor nll_cand_host(base::DataType::kDataTypeFp32, batch, true, host_alloc);
  if (stage.is_empty() || ref_dev.is_empty() || kl_dev.is_empty() || same_dev.is_empty() ||
      nll_ref_dev.is_empty() || nll_cand_dev.is_empty() || kl_host.is_empty() ||
      same_host.is_empty() || nll_ref_host.is_empty() || nll_cand_host.is_empty()) {
    LOG(FATAL) << "候选 pass 的 pinned/设备暂存分配失败";
  }

  Metrics met;
  met.kl_values.reserve(static_cast<size_t>(num_chunks) * (a.seq_len - 1));
  met.chunk_kl_sum.assign(num_chunks, 0.0);
  met.chunk_nll_ref.assign(num_chunks, 0.0);
  met.chunk_nll_cand.assign(num_chunks, 0.0);
  met.chunk_count.assign(num_chunks, 0);
  met.chunk_match.assign(num_chunks, 0);

  const int64_t chunk_bytes = static_cast<int64_t>(a.seq_len - 1) * vocab * 2;
  const int64_t data_off = sizeof(CacheHeader);

  drive_chunks(m, chunks, batch, a.seq_len, a.block_size,
               [&](int chunk0, int rows, int t, const tensor::Tensor& cand_logits,
                   const int32_t* targets_dev, cudaStream_t stream) {
                 // Reference row for each candidate row, straight from cache.
                 for (int r = 0; r < rows; ++r) {
                   const int64_t off = data_off +
                                       static_cast<int64_t>(chunk0 + r) * chunk_bytes +
                                       static_cast<int64_t>(t) * vocab * 2;
                   pread_all(fd, stage.ptr<uint16_t>() + static_cast<int64_t>(r) * vocab,
                             static_cast<size_t>(vocab) * 2, off);
                 }
                 cudaMemcpyAsync(ref_dev.ptr<uint16_t>(), stage.ptr<uint16_t>(),
                                 static_cast<size_t>(rows) * vocab * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice, stream);
                 wq_eval_cu(ref_dev.ptr<uint16_t>(), cand_logits.ptr<uint16_t>(), targets_dev,
                            vocab, rows, kl_dev.ptr<float>(), same_dev.ptr<int32_t>(),
                            nll_ref_dev.ptr<float>(), nll_cand_dev.ptr<float>(), stream);

                 cudaMemcpyAsync(kl_host.ptr<float>(), kl_dev.ptr<float>(),
                                 static_cast<size_t>(rows) * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream);
                 cudaMemcpyAsync(same_host.ptr<int32_t>(), same_dev.ptr<int32_t>(),
                                 static_cast<size_t>(rows) * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream);
                 cudaMemcpyAsync(nll_ref_host.ptr<float>(), nll_ref_dev.ptr<float>(),
                                 static_cast<size_t>(rows) * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream);
                 cudaMemcpyAsync(nll_cand_host.ptr<float>(), nll_cand_dev.ptr<float>(),
                                 static_cast<size_t>(rows) * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream);
                 if (cudaStreamSynchronize(stream) != cudaSuccess) {
                   LOG(FATAL) << "候选指标 D2H 失败: " << cudaGetErrorString(cudaGetLastError());
                 }

                 for (int r = 0; r < rows; ++r) {
                   const int chunk = chunk0 + r;
                   const double kl = kl_host.ptr<float>()[r];
                   const double nr = nll_ref_host.ptr<float>()[r];
                   const double nc = nll_cand_host.ptr<float>()[r];
                   met.kl_sum += kl;
                   met.nll_ref_sum += nr;
                   met.nll_cand_sum += nc;
                   met.count += 1;
                   met.kl_values.push_back(static_cast<float>(kl));
                   met.chunk_kl_sum[chunk] += kl;
                   met.chunk_nll_ref[chunk] += nr;
                   met.chunk_nll_cand[chunk] += nc;
                   met.chunk_count[chunk] += 1;
                   if (same_host.ptr<int32_t>()[r]) {
                     met.top1_match += 1;
                     met.chunk_match[chunk] += 1;
                   }
                 }
               });
  ::close(fd);

  if (met.count == 0) LOG(FATAL) << "没有打分位置";

  const double mean_kl = met.kl_sum / static_cast<double>(met.count);
  const double p999_kl = bench::percentile(
      std::vector<double>(met.kl_values.begin(), met.kl_values.end()), 99.9);
  const double same_top1 = static_cast<double>(met.top1_match) / static_cast<double>(met.count);
  const double mean_nll_ref = met.nll_ref_sum / static_cast<double>(met.count);
  const double mean_nll_cand = met.nll_cand_sum / static_cast<double>(met.count);
  const double ppl_ref = std::exp(mean_nll_ref);
  const double ppl_cand = std::exp(mean_nll_cand);
  const double ppl_change_pct = (ppl_cand / ppl_ref - 1.0) * 100.0;

  std::cout << "\n===== Weight Quantization Fidelity (WikiText-2) =====\n"
            << "TAG              " << a.tag << "\n"
            << "REF_MODEL        " << a.ref_model << "\n"
            << "QUANT_MODEL      " << a.quant_model << "\n"
            << "DATASET          " << a.dataset << "\n"
            << "CONFIG           seq_len=" << a.seq_len << " batch=" << a.batch
            << " chunks=" << num_chunks << " scored_positions=" << met.count
            << " block_size=" << a.block_size << "\n"
            << std::setprecision(6) << "Mean KLD         " << mean_kl << "\n"
            << "99.9% KLD        " << p999_kl << "\n"
            << std::setprecision(4) << "Same Top-1       " << (same_top1 * 100.0) << "% ("
            << met.top1_match << "/" << met.count << ")\n"
            << std::setprecision(6) << "PPL (reference)  " << ppl_ref << "\n"
            << "PPL (quantized)  " << ppl_cand << "\n"
            << std::setprecision(4) << "PPL Change       " << (ppl_change_pct >= 0 ? "+" : "")
            << ppl_change_pct << "%\n"
            << std::setprecision(6) << "NLL (reference)  " << mean_nll_ref << "\n"
            << "NLL (quantized)  " << mean_nll_cand << "\n"
            << "Tokens scored    " << met.count << "\n";
  std::cout << std::defaultfloat;

  if (!a.csv.empty()) {
    std::ofstream f(a.csv);
    if (!f) {
      LOG(ERROR) << "无法写出 CSV: " << a.csv;
    } else {
      f << "# tag=" << a.tag << " ref=" << a.ref_model << " quant=" << a.quant_model
        << " dataset=" << a.dataset << " seq_len=" << a.seq_len << " batch=" << a.batch
        << " chunks=" << num_chunks << " block_size=" << a.block_size << "\n";
      f << "metric,value\n";
      f << std::setprecision(10) << "mean_kld," << mean_kl << "\np999_kld," << p999_kl
        << "\nsame_top1," << same_top1 << "\nppl_ref," << ppl_ref << "\nppl_quant," << ppl_cand
        << "\nppl_change_pct," << ppl_change_pct << "\ntokens," << met.count << "\n";
      f << "# per chunk (each chunk has seq_len-1 scored positions)\n";
      f << "chunk,mean_kld,mean_nll_ref,mean_nll_quant,same_top1\n";
      for (int c = 0; c < num_chunks; ++c) {
        if (met.chunk_count[c] == 0) continue;
        const double n = static_cast<double>(met.chunk_count[c]);
        f << c << "," << std::setprecision(10) << (met.chunk_kl_sum[c] / n) << ","
          << (met.chunk_nll_ref[c] / n) << "," << (met.chunk_nll_cand[c] / n) << ","
          << (static_cast<double>(met.chunk_match[c]) / n) << "\n";
      }
      std::cout << "\nCSV " << a.csv << "\n";
    }
  }
}

std::string resolve_tokenizer(const Args& a) {
  if (!a.tokenizer.empty()) return a.tokenizer;
  const char* names[] = {"tokenizer.json", "tokenizer.model"};
  for (const char* name : names) {
    const std::string p = join_path(a.ref_model, name);
    if (file_exists(p)) return p;
  }
  for (const char* name : names) {
    const std::string p = join_path(a.quant_model, name);
    if (file_exists(p)) return p;
  }
  return "";
}

void usage(const char* argv0) {
  std::cerr
      << "用法: " << argv0 << " <ref-model-dir> <quant-model-dir> [dataset.raw]\n"
      << "       [--tokenizer tokenizer.json] [--model-type qwen3|qwen2|llama]\n"
      << "       [--seq-len 512] [--batch 16] [--max-chunks N] [--block-size 16]\n"
      << "       [--cache /tmp/weight_quantization_eval_ref_logits.bin]\n"
      << "       [--stage all|ref|quant] [--csv out.csv] [--tag label]\n"
      << "\n"
      << "  参考模型: 未量化的 BF16 模型目录 (Qwen3-14B)\n"
      << "  量化模型: AWQ W4A16 模型目录 (Qwen3-14B-AWQ)\n"
      << "  数据集  : WikiText-2 raw test 文件 (默认 wikitext-2-raw-v1-test.raw)\n"
      << "  指标    : Mean KLD / 99.9% KLD / Same Top-1 / PPL Change\n"
      << "            (KLD 方向 = KL(reference || quantized), 非重叠 512-token 块)\n";
}
}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::GLOG_WARNING);

  if (argc < 3) {
    usage(argv[0]);
    return -1;
  }

  using namespace bench;
  Args a;
  a.ref_model = argv[1];
  a.quant_model = argv[2];
  // Optional third positional: the dataset path (flags may follow it).
  if (argc >= 4 && argv[3][0] != '-') a.dataset = argv[3];

  if (has_arg(argc, argv, "--dataset")) {
    a.dataset = get_arg(argc, argv, "--dataset");
  }
  if (has_arg(argc, argv, "--tokenizer")) {
    a.tokenizer = get_arg(argc, argv, "--tokenizer");
  }
  if (has_arg(argc, argv, "--model-type")) {
    a.model_type = get_arg(argc, argv, "--model-type");
  }
  if (has_arg(argc, argv, "--seq-len")) {
    a.seq_len = to_int(get_arg(argc, argv, "--seq-len", "512"));
  }
  if (has_arg(argc, argv, "--batch")) {
    a.batch = to_int(get_arg(argc, argv, "--batch", "16"));
  }
  if (has_arg(argc, argv, "--max-chunks")) {
    a.max_chunks = to_int(get_arg(argc, argv, "--max-chunks", "0"));
  }
  if (has_arg(argc, argv, "--block-size")) {
    a.block_size = to_int(get_arg(argc, argv, "--block-size", "16"));
  }
  if (has_arg(argc, argv, "--cache")) {
    a.cache = get_arg(argc, argv, "--cache");
  }
  if (has_arg(argc, argv, "--stage")) {
    a.stage = get_arg(argc, argv, "--stage");
  }
  if (has_arg(argc, argv, "--csv")) {
    a.csv = get_arg(argc, argv, "--csv");
  }
  if (has_arg(argc, argv, "--tag")) {
    a.tag = get_arg(argc, argv, "--tag");
  }

  if (a.seq_len < 2) {
    std::cerr << "参数非法: --seq-len 必须 >= 2\n";
    return -1;
  }
  if (a.batch <= 0 || a.max_chunks < 0) {
    std::cerr << "参数非法: --batch 必须为正, --max-chunks 不能为负\n";
    return -1;
  }
  if (a.stage != "all" && a.stage != "ref" && a.stage != "quant") {
    std::cerr << "参数非法: --stage 只支持 all|ref|quant,收到 " << a.stage << "\n";
    return -1;
  }

  // Resolve defaults relative to the project root (build/benchmark/<bin>).
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
  a.dataset = resolve_default(a.dataset, exe_root);
  a.ref_model = resolve_default(a.ref_model, exe_root);
  a.quant_model = resolve_default(a.quant_model, exe_root);
  if (!a.tokenizer.empty()) a.tokenizer = resolve_default(a.tokenizer, exe_root);

  // Role validation: a swapped pair would produce a confident but meaningless
  // number, so reject it before spending minutes loading weights.
  const std::string ref_q = read_quant_method(a.ref_model);
  const std::string quant_q = read_quant_method(a.quant_model);
  if (!ref_q.empty()) {
    LOG(FATAL) << "参考模型 " << a.ref_model << " 带有 quantization_config(" << ref_q
               << "),不是未量化的 BF16 参考模型;两个参数是否写反了?";
  }
  if (quant_q.empty()) {
    LOG(FATAL) << "量化模型 " << a.quant_model << " 的 config.json 没有 quantization_config;"
                  "它看起来是未量化模型。两个参数是否写反了?";
  }
  if (quant_q != "awq") {
    LOG(FATAL) << "量化模型使用 " << quant_q << ",本运行时只支持 awq";
  }

  const std::string tokenizer = resolve_tokenizer(a);
  if (tokenizer.empty()) {
    std::cerr << "找不到 tokenizer: 请用 --tokenizer 指定 (Qwen: tokenizer.json;"
                 " Llama: tokenizer.model)\n";
    return -1;
  }

  std::cout << "REF_MODEL   " << a.ref_model << " (quant_method=none)\n"
            << "QUANT_MODEL " << a.quant_model << " (quant_method=" << quant_q << ")\n"
            << "DATASET     " << a.dataset << "\n"
            << "TOKENIZER   " << tokenizer << "\n"
            << "CONFIG      model_type=" << a.model_type << " seq_len=" << a.seq_len
            << " batch=" << a.batch << " max_chunks=" << a.max_chunks
            << " block_size=" << a.block_size << " stage=" << a.stage
            << " cache=" << a.cache << "\n";
  log_gpu_mem("start");

  std::vector<std::vector<int32_t>> chunks;
  uint64_t tokens_hash = 0;
  int ref_vocab = 0;

  if (a.stage != "quant") {
    auto ref_model = load_model(a.model_type, tokenizer, a.ref_model, /*is_quant_model=*/false);
    ref_vocab = ref_model->vocab_size();
    if (ref_model->seq_len() < a.seq_len) {
      LOG(FATAL) << "--seq-len " << a.seq_len << " 超过模型窗口 " << ref_model->seq_len();
    }
    std::cout << "REF_MODEL_LOADED vocab=" << ref_vocab << " layers=" << ref_model->layer_num()
              << " kv_dim=" << ref_model->kv_dim() << "\n";
    log_gpu_mem("after_ref_load");

    tokenize_chunks(ref_model, a.dataset, a.seq_len, a.max_chunks, chunks, tokens_hash);

    run_ref_pass(ref_model, chunks, a, a.cache, tokens_hash);

    ref_model.reset();
    release_gpu_cache();
  }

  if (a.stage != "ref") {
    auto quant_model = load_model(a.model_type, tokenizer, a.quant_model,
                                  /*is_quant_model=*/true);
    if (a.stage == "quant") {
      // No reference model was loaded: tokenize with the candidate's
      // (identical) tokenizer and derive the cache hash from it.
      tokenize_chunks(quant_model, a.dataset, a.seq_len, a.max_chunks, chunks, tokens_hash);
    }
    if (ref_vocab != 0 && quant_model->vocab_size() != ref_vocab) {
      LOG(FATAL) << "参考/量化模型 vocab 不一致: " << ref_vocab << " vs "
                 << quant_model->vocab_size();
    }
    if (quant_model->seq_len() < a.seq_len) {
      LOG(FATAL) << "--seq-len " << a.seq_len << " 超过模型窗口 " << quant_model->seq_len();
    }
    std::cout << "QUANT_MODEL_LOADED vocab=" << quant_model->vocab_size()
              << " layers=" << quant_model->layer_num()
              << " kv_dim=" << quant_model->kv_dim() << "\n";
    log_gpu_mem("after_quant_load");

    run_quant_pass(quant_model, chunks, a, a.cache, tokens_hash);
  }

  return 0;
}
