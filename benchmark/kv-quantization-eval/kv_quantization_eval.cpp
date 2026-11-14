// KV-cache quantization fidelity eval (WikiText-2 raw test).
// 直接调用 forward_batch,复用 KVManager / S4 元数据 / kv_slots。参考臂
// kv_quant=0 (BF16),候选臂 --kv-quant (1=S4),两个 KV 池在进程内 lockstep
// 推进,逐位置比较 logits。指标与权重量化 eval 一致:
// Mean KLD / 99.9% KLD / Same Top-1 / PPL Change。

#include <base/alloc.h>
#include <base/base.h>
#include <glog/logging.h>
#include "model/llama3.h"
#include "model/qwen2.h"
#include "model/qwen3.h"
#include "scheduler/block_allocator.h"
#include "scheduler/kv_manager.h"
#include "tensor/tensor.h"

#include "../bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// kv_eval_kernel.cu —— 设备端逐位置 KLD / top-1 / NLL 归约(见该文件)
void kv_eval_cu(const void* ref_logits_bf16, const void* cand_logits_bf16,
                const int32_t* targets, int32_t vocab, int32_t rows, float* out_kl,
                int32_t* same_top1, float* nll_ref, float* nll_cand, void* stream);

namespace {

struct Args {
  std::string model_dir;
  std::string tokenizer;
  std::string dataset = "wikitext-2-raw-v1-test.raw";
  std::string model_type = "qwen3";
  std::string csv;
  std::string tag = "kv_quant";
  int seq_len = 512;
  int batch = 16;
  int max_chunks = 0;  // 0 = all complete chunks
  int block_size = 16;
  int kv_quant = scheduler::kKvQuantS4;
  int group_size = 0;
  int channel_group = 0;
};

struct Pool {
  std::unique_ptr<scheduler::KVManager> kv;
  std::vector<int32_t> rows;
  tensor::Tensor host_bt;
  tensor::Tensor host_slots;
  int table_stride = 0;
};

struct Metrics {
  double kl_sum = 0.0;
  int64_t count = 0;
  int64_t top1_match = 0;
  double nll_ref_sum = 0.0;
  double nll_cand_sum = 0.0;
  std::vector<float> kl_values;

  std::vector<double> chunk_kl_sum;
  std::vector<double> chunk_nll_ref;
  std::vector<double> chunk_nll_cand;
  std::vector<int64_t> chunk_count;
  std::vector<int64_t> chunk_match;

  std::vector<double> pos_kl_sum;
  std::vector<int64_t> pos_count;
  std::vector<int64_t> pos_match;
};

std::string join_path(const std::string& dir, const std::string& name) {
  return (std::filesystem::path(dir) / name).string();
}

bool file_exists(const std::string& path) {
  std::error_code ec;
  return std::filesystem::exists(path, ec);
}

std::string read_dataset_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    LOG(FATAL) << "无法打开数据集: " << path;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  std::string text = ss.str();
  if (text.size() >= 3 && static_cast<uint8_t>(text[0]) == 0xEF &&
      static_cast<uint8_t>(text[1]) == 0xBB && static_cast<uint8_t>(text[2]) == 0xBF) {
    text.erase(0, 3);
  }
  std::string out;
  out.reserve(text.size());
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\r') {
      if (i + 1 < text.size() && text[i + 1] == '\n') continue;
      out.push_back('\n');
    } else {
      out.push_back(text[i]);
    }
  }
  return out;
}

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

void tokenize_chunks(const std::shared_ptr<model::Model>& m, const std::string& dataset,
                     int seq_len, int max_chunks, std::vector<std::vector<int32_t>>& chunks) {
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
  std::cout << "CORPUS bytes=" << std::filesystem::file_size(dataset) << " tokens=" << tokens.size()
            << " complete_chunks=" << complete << " used_chunks=" << use
            << " remainder_tokens=" << remainder << " seq_len=" << seq_len << "\n";
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void init_pool(Pool& p, const std::shared_ptr<model::Model>& m, int batch, int seq_len,
               int block_size, int kv_quant, int group_size, int channel_group,
               const std::shared_ptr<base::DeviceAllocator>& host_alloc) {
  auto kv_alloc = base::CUDADeviceAllocatorFactory::get_instance();
  p.kv = std::make_unique<scheduler::KVManager>(
      m->layer_num(), batch, seq_len, m->kv_dim(), m->kv_head_num(), kv_alloc,
      base::DeviceType::kDeviceCUDA, block_size, kv_quant, group_size, channel_group);
  p.table_stride = p.kv->max_blocks_per_seq();
  p.rows.assign(batch, -1);
  p.host_bt = tensor::Tensor(base::DataType::kDataTypeInt32, batch, p.table_stride, true,
                             host_alloc);
  p.host_slots = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, host_alloc);
  if (p.host_bt.is_empty() || p.host_slots.is_empty()) {
    LOG(FATAL) << "pinned block table / kv_slots 暂存分配失败";
  }
  scheduler::BlockAllocator* ba = p.kv->block_allocator();
  for (int r = 0; r < batch; ++r) {
    p.rows[r] = ba->allocate_blocks(p.table_stride);
    if (p.rows[r] < 0) {
      LOG(FATAL) << "KV block 池不足: 第 " << r << " 行分配失败 (blocks=" << ba->num_blocks()
                 << ", stride=" << p.table_stride << ")";
    }
    ba->copy_block_table_row(p.rows[r],
                             p.host_bt.ptr<int32_t>() + r * p.table_stride, p.table_stride);
    p.host_slots.ptr<int32_t>()[r] = p.rows[r];
  }
}

void refresh_pool(Pool& p, int rows) {
  scheduler::BlockAllocator* ba = p.kv->block_allocator();
  for (int r = 0; r < rows; ++r) {
    ba->free_all(p.rows[r]);
    p.rows[r] = ba->allocate_blocks(p.table_stride);
    if (p.rows[r] < 0) LOG(FATAL) << "重新分配 KV block 失败";
    ba->copy_block_table_row(p.rows[r],
                             p.host_bt.ptr<int32_t>() + r * p.table_stride, p.table_stride);
    p.host_slots.ptr<int32_t>()[r] = p.rows[r];
  }
}

void run_fidelity(const std::shared_ptr<model::Model>& m,
                  const std::vector<std::vector<int32_t>>& chunks, const Args& a,
                  Pool& ref_pool, Pool& cand_pool) {
  const int vocab = m->vocab_size();
  const int num_chunks = static_cast<int>(chunks.size());

  auto kv_alloc = base::CUDADeviceAllocatorFactory::get_instance();
  auto host_alloc = std::shared_ptr<base::DeviceAllocator>(
      base::PinnedCPUDeviceAllocatorFactory::get_instance());

  tensor::Tensor host_input_ids(base::DataType::kDataTypeInt32, a.batch, true, host_alloc);
  tensor::Tensor host_positions(base::DataType::kDataTypeInt32, a.batch, true, host_alloc);
  tensor::Tensor host_targets(base::DataType::kDataTypeInt32, a.batch, true, host_alloc);
  tensor::Tensor logits_ref(base::DataType::kDataTypeBF16, a.batch, vocab, true, kv_alloc);
  tensor::Tensor logits_cand(base::DataType::kDataTypeBF16, a.batch, vocab, true, kv_alloc);
  tensor::Tensor targets_dev(base::DataType::kDataTypeInt32, a.batch, true, kv_alloc);
  tensor::Tensor kl_dev(base::DataType::kDataTypeFp32, a.batch, true, kv_alloc);
  tensor::Tensor same_dev(base::DataType::kDataTypeInt32, a.batch, true, kv_alloc);
  tensor::Tensor nll_ref_dev(base::DataType::kDataTypeFp32, a.batch, true, kv_alloc);
  tensor::Tensor nll_cand_dev(base::DataType::kDataTypeFp32, a.batch, true, kv_alloc);
  tensor::Tensor kl_host(base::DataType::kDataTypeFp32, a.batch, true, host_alloc);
  tensor::Tensor same_host(base::DataType::kDataTypeInt32, a.batch, true, host_alloc);
  tensor::Tensor nll_ref_host(base::DataType::kDataTypeFp32, a.batch, true, host_alloc);
  tensor::Tensor nll_cand_host(base::DataType::kDataTypeFp32, a.batch, true, host_alloc);
  if (host_input_ids.is_empty() || host_positions.is_empty() || host_targets.is_empty() ||
      logits_ref.is_empty() || logits_cand.is_empty() || targets_dev.is_empty() ||
      kl_dev.is_empty() || same_dev.is_empty() || nll_ref_dev.is_empty() ||
      nll_cand_dev.is_empty() || kl_host.is_empty() || same_host.is_empty() ||
      nll_ref_host.is_empty() || nll_cand_host.is_empty()) {
    LOG(FATAL) << "pinned/设备暂存分配失败";
  }

  Metrics met;
  met.chunk_kl_sum.assign(num_chunks, 0.0);
  met.chunk_nll_ref.assign(num_chunks, 0.0);
  met.chunk_nll_cand.assign(num_chunks, 0.0);
  met.chunk_count.assign(num_chunks, 0);
  met.chunk_match.assign(num_chunks, 0);
  met.pos_kl_sum.assign(static_cast<size_t>(a.seq_len), 0.0);
  met.pos_count.assign(static_cast<size_t>(a.seq_len), 0);
  met.pos_match.assign(static_cast<size_t>(a.seq_len), 0);
  met.kl_values.reserve(static_cast<size_t>(num_chunks) * (a.seq_len - 1));

  cudaStream_t stream = m->cuda_stream();
  const bool cand_quantized = cand_pool.kv->quantized();
  const double t0 = now_s();

  for (int g = 0; g < num_chunks; g += a.batch) {
    const int b = std::min(a.batch, num_chunks - g);
    refresh_pool(ref_pool, b);
    refresh_pool(cand_pool, b);

    tensor::Tensor ids_view = host_input_ids;
    tensor::Tensor pos_view = host_positions;
    tensor::Tensor tgt_view = host_targets;
    tensor::Tensor ref_bt_view = ref_pool.host_bt;
    tensor::Tensor cand_bt_view = cand_pool.host_bt;
    tensor::Tensor ref_logits_view = logits_ref;
    tensor::Tensor cand_logits_view = logits_cand;
    tensor::Tensor cand_slots_view = cand_pool.host_slots;
    ids_view.reshape({b});
    pos_view.reshape({b});
    tgt_view.reshape({b});
    ref_bt_view.reshape({b, ref_pool.table_stride});
    cand_bt_view.reshape({b, cand_pool.table_stride});
    ref_logits_view.reshape({b, vocab});
    cand_logits_view.reshape({b, vocab});
    cand_slots_view.reshape({b});

    for (int t = 0; t + 1 < a.seq_len; ++t) {
      for (int r = 0; r < b; ++r) {
        host_input_ids.ptr<int32_t>()[r] = chunks[g + r][t];
        host_positions.ptr<int32_t>()[r] = t;
        host_targets.ptr<int32_t>()[r] = chunks[g + r][t + 1];
      }
      cudaMemcpyAsync(targets_dev.ptr<int32_t>(), host_targets.ptr<int32_t>(),
                      static_cast<size_t>(b) * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

      const int table_cols = std::min(t / a.block_size + 1, ref_pool.table_stride);

      auto st_ref = m->forward_batch(
          ids_view, pos_view, ref_bt_view, ref_pool.kv->key_cache(),
          ref_pool.kv->value_cache(), ref_logits_view, /*need_logits=*/true, /*num_decode_rows=*/b,
          /*seq_row_start=*/nullptr, /*num_prefill_seqs=*/0, /*scratch=*/nullptr, table_cols,
          /*key_scale=*/nullptr, /*value_scale=*/nullptr, /*key_zero=*/nullptr,
          /*value_zero=*/nullptr, /*key_residual=*/nullptr, /*kv_slots=*/nullptr);
      if (!st_ref) {
        LOG(FATAL) << "参考臂 forward_batch 失败 @group=" << g << " t=" << t << ": "
                   << st_ref.get_err_code();
      }

      auto st_cand = m->forward_batch(
          ids_view, pos_view, cand_bt_view, cand_pool.kv->key_cache(),
          cand_pool.kv->value_cache(), cand_logits_view, /*need_logits=*/true,
          /*num_decode_rows=*/b, /*seq_row_start=*/nullptr, /*num_prefill_seqs=*/0,
          /*scratch=*/nullptr, table_cols,
          cand_quantized ? &cand_pool.kv->key_scale_cache() : nullptr,
          cand_quantized ? &cand_pool.kv->value_scale_cache() : nullptr,
          cand_quantized ? &cand_pool.kv->key_zero_cache() : nullptr,
          cand_quantized ? &cand_pool.kv->value_zero_cache() : nullptr,
          cand_quantized ? &cand_pool.kv->key_residual_cache() : nullptr,
          cand_quantized ? &cand_slots_view : nullptr);
      if (!st_cand) {
        LOG(FATAL) << "候选臂 forward_batch 失败 @group=" << g << " t=" << t << ": "
                   << st_cand.get_err_code();
      }

      kv_eval_cu(logits_ref.ptr<uint16_t>(), logits_cand.ptr<uint16_t>(),
                 targets_dev.ptr<int32_t>(), vocab, b, kl_dev.ptr<float>(),
                 same_dev.ptr<int32_t>(), nll_ref_dev.ptr<float>(),
                 nll_cand_dev.ptr<float>(), reinterpret_cast<void*>(stream));

      cudaMemcpyAsync(kl_host.ptr<float>(), kl_dev.ptr<float>(),
                      static_cast<size_t>(b) * sizeof(float), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(same_host.ptr<int32_t>(), same_dev.ptr<int32_t>(),
                      static_cast<size_t>(b) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(nll_ref_host.ptr<float>(), nll_ref_dev.ptr<float>(),
                      static_cast<size_t>(b) * sizeof(float), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(nll_cand_host.ptr<float>(), nll_cand_dev.ptr<float>(),
                      static_cast<size_t>(b) * sizeof(float), cudaMemcpyDeviceToHost, stream);
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        LOG(FATAL) << "指标 D2H 失败: " << cudaGetErrorString(cudaGetLastError());
      }

      // 同步后才可覆写下一次的 host staging。
      for (int r = 0; r < b; ++r) {
        const int chunk = g + r;
        const double kl = kl_host.ptr<float>()[r];
        const double nr = nll_ref_host.ptr<float>()[r];
        const double nc = nll_cand_host.ptr<float>()[r];
        const bool same = same_host.ptr<int32_t>()[r] != 0;

        met.kl_sum += kl;
        met.nll_ref_sum += nr;
        met.nll_cand_sum += nc;
        met.count += 1;
        met.kl_values.push_back(static_cast<float>(kl));
        met.chunk_kl_sum[chunk] += kl;
        met.chunk_nll_ref[chunk] += nr;
        met.chunk_nll_cand[chunk] += nc;
        met.chunk_count[chunk] += 1;
        met.pos_kl_sum[t] += kl;
        met.pos_count[t] += 1;
        if (same) {
          met.top1_match += 1;
          met.chunk_match[chunk] += 1;
          met.pos_match[t] += 1;
        }
      }
    }

    const double elapsed = now_s() - t0;
    const int done = g + b;
    const double eta = done > 0 ? elapsed * (num_chunks - done) / done : 0.0;
    std::cout << "PASS_CHUNKS " << done << "/" << num_chunks << " elapsed_s=" << std::fixed
              << std::setprecision(1) << elapsed << " eta_s=" << eta << std::defaultfloat
              << std::endl;
  }

  if (met.count == 0) LOG(FATAL) << "没有打分位置";

  const double mean_kl = met.kl_sum / static_cast<double>(met.count);
  const double p999_kl =
      bench::percentile(std::vector<double>(met.kl_values.begin(), met.kl_values.end()), 99.9);
  const double same_top1 = static_cast<double>(met.top1_match) / static_cast<double>(met.count);
  const double mean_nll_ref = met.nll_ref_sum / static_cast<double>(met.count);
  const double mean_nll_cand = met.nll_cand_sum / static_cast<double>(met.count);
  const double ppl_ref = std::exp(mean_nll_ref);
  const double ppl_cand = std::exp(mean_nll_cand);
  const double ppl_change_pct = (ppl_cand / ppl_ref - 1.0) * 100.0;

  const double ref_pool_gb =
      static_cast<double>(ref_pool.kv->pool_bytes()) / (1024.0 * 1024.0 * 1024.0);
  const double cand_pool_gb =
      static_cast<double>(cand_pool.kv->pool_bytes()) / (1024.0 * 1024.0 * 1024.0);
  const double compression = static_cast<double>(ref_pool.kv->pool_bytes()) /
                             static_cast<double>(cand_pool.kv->pool_bytes());

  std::cout << "\n===== KV Quantization Fidelity (WikiText-2) =====\n"
            << "TAG              " << a.tag << "\n"
            << "MODEL            " << a.model_dir << "\n"
            << "DATASET          " << a.dataset << "\n"
            << "CONFIG           seq_len=" << a.seq_len << " batch=" << a.batch
            << " chunks=" << num_chunks << " scored_positions=" << met.count
            << " block_size=" << a.block_size << "\n"
            << "KV_REF           bf16 pool=" << std::fixed << std::setprecision(2) << ref_pool_gb
            << " GB\n"
            << "KV_QUANT         " << (cand_quantized ? "s4" : "bf16(self-check)")
            << (cand_quantized ? (" group_size=" + std::to_string(cand_pool.kv->group_size()) +
                                  " channel_group=" +
                                  std::to_string(cand_pool.kv->channel_group()))
                               : "")
            << " pool=" << cand_pool_gb << " GB compression=" << compression << "x\n"
            << std::setprecision(6) << "Mean KLD         " << mean_kl << "\n"
            << "99.9% KLD        " << p999_kl << "\n"
            << std::setprecision(4) << "Same Top-1       " << (same_top1 * 100.0) << "% ("
            << met.top1_match << "/" << met.count << ")\n"
            << std::setprecision(6) << "PPL (reference)  " << ppl_ref << "\n"
            << "PPL (kv-quant)   " << ppl_cand << "\n"
            << std::setprecision(4) << "PPL Change       " << (ppl_change_pct >= 0 ? "+" : "")
            << ppl_change_pct << "%\n"
            << std::setprecision(6) << "NLL (reference)  " << mean_nll_ref << "\n"
            << "NLL (kv-quant)   " << mean_nll_cand << "\n"
            << "Tokens scored    " << met.count << "\n";
  std::cout << std::defaultfloat;

  const int bucket_edges[] = {0, 128, 256, 384, 512};
  std::cout << "\nBUCKET  pos_lo  pos_hi   KLD_mean   top1\n";
  for (int i = 0; i + 1 < static_cast<int>(sizeof(bucket_edges) / sizeof(int)); ++i) {
    const int lo = bucket_edges[i];
    const int hi = std::min(bucket_edges[i + 1], a.seq_len - 1);
    if (lo >= a.seq_len - 1) break;
    double s = 0.0;
    int64_t n = 0, match = 0;
    for (int t = lo; t < hi; ++t) {
      if (met.pos_count[t] == 0) continue;
      s += met.pos_kl_sum[t];
      n += met.pos_count[t];
      match += met.pos_match[t];
    }
    if (n == 0) continue;
    std::cout << std::setw(6) << lo << std::setw(8) << hi << "  "
              << std::setprecision(5) << (s / static_cast<double>(n)) << "   "
              << std::setprecision(4) << (100.0 * static_cast<double>(match) / n) << "%\n";
  }
  std::cout << std::defaultfloat;

  if (!a.csv.empty()) {
    std::ofstream f(a.csv);
    if (!f) {
      LOG(ERROR) << "无法写出 CSV: " << a.csv;
    } else {
      f << "# tag=" << a.tag << " model=" << a.model_dir << " dataset=" << a.dataset
        << " seq_len=" << a.seq_len << " batch=" << a.batch << " chunks=" << num_chunks
        << " block_size=" << a.block_size << " kv_quant=" << a.kv_quant
        << " group_size=" << cand_pool.kv->group_size()
        << " channel_group=" << cand_pool.kv->channel_group()
        << " ref_pool_bytes=" << ref_pool.kv->pool_bytes()
        << " quant_pool_bytes=" << cand_pool.kv->pool_bytes() << "\n";
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
      f << "# per position (all chunks)\n";
      f << "pos,mean_kld,same_top1,tokens\n";
      for (int t = 0; t + 1 < a.seq_len; ++t) {
        if (met.pos_count[t] == 0) continue;
        f << t << "," << std::setprecision(10)
          << (met.pos_kl_sum[t] / static_cast<double>(met.pos_count[t])) << ","
          << (static_cast<double>(met.pos_match[t]) / static_cast<double>(met.pos_count[t])) << ","
          << met.pos_count[t] << "\n";
      }
      std::cout << "\nCSV " << a.csv << "\n";
    }
  }
}

std::string resolve_tokenizer(const Args& a) {
  if (!a.tokenizer.empty()) return a.tokenizer;
  const char* names[] = {"tokenizer.json", "tokenizer.model"};
  for (const char* name : names) {
    const std::string p = join_path(a.model_dir, name);
    if (file_exists(p)) return p;
  }
  return "";
}

void usage(const char* argv0) {
  std::cerr
      << "用法: " << argv0 << " <model-dir> [dataset.raw]\n"
      << "       [--tokenizer tokenizer.json] [--model-type qwen3|qwen2|llama]\n"
      << "       [--seq-len 512] [--batch 16] [--max-chunks N] [--block-size 16]\n"
      << "       [--kv-quant 1] [--group-size 0] [--channel-group 0]\n"
      << "       [--csv out.csv] [--tag label]\n"
      << "\n"
      << "  参考臂 : kv_quant=0, BF16 KV cache\n"
      << "  候选臂 : --kv-quant 1 = S4 (K int4 分组 + BF16 残差窗口, V int4);\n"
      << "           0 = 自检,两个 bf16 池,应得到 KLD=0 / top1=100%\n"
      << "  数据集 : WikiText-2 raw test 文件 (默认 wikitext-2-raw-v1-test.raw)\n"
      << "  指标   : Mean KLD / 99.9% KLD / Same Top-1 / PPL Change\n"
      << "           (KLD 方向 = KL(reference || kv-quantized), 非重叠 seq-len 块)\n";
}
}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::GLOG_WARNING);

  if (argc < 2) {
    usage(argv[0]);
    return -1;
  }

  using namespace bench;
  Args a;
  a.model_dir = argv[1];
  if (argc >= 3 && argv[2][0] != '-') a.dataset = argv[2];

  if (has_arg(argc, argv, "--dataset")) a.dataset = get_arg(argc, argv, "--dataset");
  if (has_arg(argc, argv, "--tokenizer")) a.tokenizer = get_arg(argc, argv, "--tokenizer");
  if (has_arg(argc, argv, "--model-type")) a.model_type = get_arg(argc, argv, "--model-type");
  if (has_arg(argc, argv, "--seq-len"))
    a.seq_len = to_int(get_arg(argc, argv, "--seq-len", "512"));
  if (has_arg(argc, argv, "--batch"))
    a.batch = to_int(get_arg(argc, argv, "--batch", "16"));
  if (has_arg(argc, argv, "--max-chunks"))
    a.max_chunks = to_int(get_arg(argc, argv, "--max-chunks", "0"));
  if (has_arg(argc, argv, "--block-size"))
    a.block_size = to_int(get_arg(argc, argv, "--block-size", "16"));
  if (has_arg(argc, argv, "--kv-quant"))
    a.kv_quant = to_int(get_arg(argc, argv, "--kv-quant", "1"));
  if (has_arg(argc, argv, "--group-size"))
    a.group_size = to_int(get_arg(argc, argv, "--group-size", "0"));
  if (has_arg(argc, argv, "--channel-group"))
    a.channel_group = to_int(get_arg(argc, argv, "--channel-group", "0"));
  if (has_arg(argc, argv, "--csv")) a.csv = get_arg(argc, argv, "--csv");
  if (has_arg(argc, argv, "--tag")) a.tag = get_arg(argc, argv, "--tag");

  if (a.seq_len < 2) {
    std::cerr << "参数非法: --seq-len 必须 >= 2\n";
    return -1;
  }
  if (a.batch <= 0 || a.max_chunks < 0) {
    std::cerr << "参数非法: --batch 必须为正, --max-chunks 不能为负\n";
    return -1;
  }
  if (a.kv_quant != scheduler::kKvQuantNone && a.kv_quant != scheduler::kKvQuantS4) {
    std::cerr << "参数非法: --kv-quant 必须是 0(off)或 1(S4);得到 " << a.kv_quant << "\n";
    return -1;
  }
  if (a.kv_quant == scheduler::kKvQuantNone && (a.group_size != 0 || a.channel_group != 0)) {
    std::cerr << "--group-size/--channel-group 只在 --kv-quant 1 时生效;当前候选臂为 bf16\n";
  }

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
  a.model_dir = resolve_default(a.model_dir, exe_root);
  if (!a.tokenizer.empty()) a.tokenizer = resolve_default(a.tokenizer, exe_root);

  const std::string quant_method = read_quant_method(a.model_dir);
  if (!quant_method.empty() && quant_method != "awq") {
    LOG(FATAL) << "模型使用 " << quant_method << ";本运行时只支持 awq(或不带量化配置的 BF16)";
  }
  const bool is_quant_model = quant_method == "awq";

  const std::string tokenizer = resolve_tokenizer(a);
  if (tokenizer.empty()) {
    std::cerr << "找不到 tokenizer: 请用 --tokenizer 指定 (Qwen: tokenizer.json;"
                 " Llama: tokenizer.model)\n";
    return -1;
  }

  std::cout << "MODEL       " << a.model_dir << " (base_weights="
            << (is_quant_model ? "awq" : "bf16") << ")\n"
            << "DATASET     " << a.dataset << "\n"
            << "TOKENIZER   " << tokenizer << "\n"
            << "CONFIG      model_type=" << a.model_type << " seq_len=" << a.seq_len
            << " batch=" << a.batch << " max_chunks=" << a.max_chunks
            << " block_size=" << a.block_size << " kv_quant=" << a.kv_quant
            << " group_size=" << a.group_size << " channel_group=" << a.channel_group << "\n";

  auto model = load_model(a.model_type, tokenizer, a.model_dir, is_quant_model);
  if (model->seq_len() < a.seq_len) {
    LOG(FATAL) << "--seq-len " << a.seq_len << " 超过模型窗口 " << model->seq_len();
  }
  std::cout << "MODEL_LOADED vocab=" << model->vocab_size()
            << " layers=" << model->layer_num() << " kv_dim=" << model->kv_dim()
            << " kv_heads=" << model->kv_head_num() << "\n";

  std::vector<std::vector<int32_t>> chunks;
  tokenize_chunks(model, a.dataset, a.seq_len, a.max_chunks, chunks);

  auto host_alloc = std::shared_ptr<base::DeviceAllocator>(
      base::PinnedCPUDeviceAllocatorFactory::get_instance());
  Pool ref_pool;
  Pool cand_pool;
  init_pool(ref_pool, model, a.batch, a.seq_len, a.block_size, scheduler::kKvQuantNone, 0, 0,
            host_alloc);
  init_pool(cand_pool, model, a.batch, a.seq_len, a.block_size, a.kv_quant, a.group_size,
            a.channel_group, host_alloc);
  if (ref_pool.table_stride != cand_pool.table_stride) {
    LOG(FATAL) << "参考/候选 block table 宽度不一致: " << ref_pool.table_stride << " vs "
               << cand_pool.table_stride;
  }

  std::cout << "KV_POOL     ref_bytes=" << ref_pool.kv->pool_bytes()
            << " quant_bytes=" << cand_pool.kv->pool_bytes()
            << " quantized=" << (cand_pool.kv->quantized() ? 1 : 0) << "\n";

  run_fidelity(model, chunks, a, ref_pool, cand_pool);
  return 0;
}
