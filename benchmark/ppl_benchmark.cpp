// ============================================================================
// PPL gate for KV-cache quantization (bf16 baseline vs int8 arm).
//
// 为什么直接驱动模型,而不是走 Scheduler:
//   * PPL 需要 teacher forcing —— 每一步必须喂入"真实的下一个 token"。
//     Scheduler 是 argmax 采样并把采样结果回灌,且 logits_ 是私有的,
//     结构上无法算 PPL;
//   * decode 模式驱动 decode_step(量化要改的那条 decode 读路径);chunked
//     模式驱动 forward_batch 的混合步,覆盖 split-KV decode 之外还要走一遍
//     prefill attention 读路径。两者都直接驱动,测量结果里没有调度器的
//     任何影响。
//
// 两种模式:
//
// [decode,默认] 一次运行的形态:`num_seq` 条长度为 `seq_len` 的序列,逐 token
//   推进(所有行步调一致),共 `passes` 遍(每遍取不同的文档片段)。
//   第 r 行第 t 步:喂入 seq[r][t],对 seq[r][t+1] 打分。
//   => 长度 L 的序列需要 L-1 次 decode,产出 L-1 个 NLL。
//   之所以不用 prefill 一次把 prompt 打进去:prefill 路径只为 decode 行产出
//   logits,而 NLL 需要在"每一个位置"都有值(canonical PPL 定义)。
//
// [chunked] 复现生产的 chunked-prefill 混合步形(生产一档就是
//   kPrefillRowReserve=512 个 token),直接驱动 forward_batch:每条序列按
//   `chunk_size` 个 token 一段推进,每步的 batch 行 =
//     [该 chunk 的最后一个 token 作 decode 行]
//   + [chunk 里其余 token 作 prefill 行]
//   (num_decode_rows=1,seq_row_start=[1,len],num_prefill_seqs=1;行序/行地图
//   按 Model::forward_batch 的混合步约定填)。
//   只有 decode 行产出 logits,所以 NLL 也只在 decode 行上算 —— 而 teacher
//   forcing 要的"下一个 token"的 logits 恰好就是它产出的:每个 chunk 边界
//   位置 p=e-1 得到一个 NLL(=seq[r][e] 在上下文 seq[r][:e] 下的负对数似然)。
//   最后一个 chunk 的末尾没有下一个 token,不打分,但它的 token 仍然进模型,
//   保证整条序列的 KV 覆盖完整。打分位置是 decode 模式位置集合的子集,分桶
//   与 CSV 口径完全一致,因此两条驱动路径可以逐位置直接对比(decode-PPL 在
//   相同位置的值就来自同一条序列、同一个上下文)。
//
// A/B 对比用同一批序列、同一构建下的两个开关。绝对 PPL 是语料特定的
// (ShareGPT 文档拼接出来的数不具可比性),有意义的是 DELTA:整体 / 逐序列 /
// 按上下文位置分桶。KV 量化的误差随 key 被读取时的上下文长度累积,
// 所以"位置曲线"才是把真实退化与开头几百个 token 的噪声区分开的判据。
//
// 用法:
//   ./ppl_benchmark <model-dir> <tokenizer> [--model-type qwen3|qwen2|llama]
//                   [--dataset ShareGPT_prompts.jsonl] [--num-seq 32]
//                   [--seq-len 4096] [--passes 1] [--block-size 16]
//                   [--mode decode|chunked] [--chunk-size 512]
//                   [--csv out.csv] [--tag label]
// ============================================================================

#include <base/alloc.h>
#include <base/base.h>
#include <glog/logging.h>
#include "model/llama3.h"
#include "model/qwen2.h"
#include "model/qwen3.h"
#include "scheduler/block_allocator.h"
#include "scheduler/kv_manager.h"
#include "tensor/tensor.h"

#include "bench_common.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ppl_nll_kernel.cu —— 设备端 NLL 归约(见该文件的头注释)
void ppl_nll_cu(const void* logits_bf16, const int32_t* targets, float* out, int32_t vocab,
                int32_t rows, int32_t target_stride, int32_t out_stride, void* stream);

namespace {

struct Args {
  std::string model_dir;
  std::string tokenizer;
  std::string dataset = "ShareGPT_prompts.jsonl";
  std::string model_type = "qwen3";
  std::string csv;
  std::string tag = "run";
  std::string mode = "decode";  // decode | chunked(见文件头)
  int num_seq = 32;
  int seq_len = 4096;
  int passes = 1;
  int block_size = 16;
  int chunk_size = 512;  // chunked 模式每步的 chunk token 数(= 生产 kPrefillRowReserve)
  int kv_quant = -1;  // scheduler::KvQuant 掩码,-1 = 从 LLAMA_KV_QUANT 取
  int debug_logits = 0;  // >0: dump top-N logits of row 0 at the final step
};

// 把编码后的文档按顺序拼成恰好 seq_len 个 token 的序列。
// 用全局游标(cursor)推进文档,所以各序列内容不重叠、也和遍数无关地确定;
// pass 只影响从哪一篇文档开始(不同遍看不同语料,从而增加统计量)。
std::vector<std::vector<int32_t>> build_sequences(
    const std::vector<std::vector<int32_t>>& docs, int num_seq, int seq_len, size_t& cursor) {
  std::vector<std::vector<int32_t>> out;
  out.reserve(num_seq);
  for (int b = 0; b < num_seq; ++b) {
    std::vector<int32_t> s;
    s.reserve(seq_len);
    while (static_cast<int>(s.size()) < seq_len && !docs.empty()) {
      const std::vector<int32_t>& d = docs[cursor % docs.size()];
      ++cursor;
      if (d.empty()) continue;
      const int take = std::min<int>(seq_len - static_cast<int>(s.size()),
                                     static_cast<int>(d.size()));
      s.insert(s.end(), d.begin(), d.begin() + take);
    }
    // 语料耗尽(文档太短):用最后一个 token 补齐,保证形状合法。
    while (static_cast<int>(s.size()) < seq_len) s.push_back(s.empty() ? 0 : s.back());
    out.push_back(std::move(s));
  }
  return out;
}

// 位置分桶边界。0-512 段单独看,是因为那段 NLL 由 RoPE 短上下文主导、
// 与量化误差无关,混进总均值会稀释信号(它同时也是不同构建间最先漂移的地方)。
const int kBucketEdges[] = {0, 512, 1024, 2048, 4096};

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::GLOG_WARNING);
  if (argc < 3) {
    std::cerr << "用法: " << argv[0]
              << " <model-dir> <tokenizer> [--dataset p.jsonl] [--num-seq 32]"
                 " [--seq-len 4096] [--passes 1] [--block-size 16]"
                 " [--mode decode|chunked] [--chunk-size 512]"
                 " [--kv-quant 0|1] [--model-type qwen3|qwen2|llama]"
                 " [--csv out.csv] [--tag label]\n"
                 "  --kv-quant: 0=off, 6=S4 (默认读环境变量 LLAMA_KV_QUANT)\n"
                 "  --mode decode: 逐 token decode 驱动(每位置都有 NLL);\n"
                 "         chunked: 生产混合步形,每 chunk 只对末位置的 decode 行算 NLL\n";
    return -1;
  }

  using namespace bench;
  Args a;
  a.model_dir = argv[1];
  a.tokenizer = argv[2];
  if (has_arg(argc, argv, "--dataset")) a.dataset = get_arg(argc, argv, "--dataset");
  if (has_arg(argc, argv, "--model-type")) a.model_type = get_arg(argc, argv, "--model-type");
  if (has_arg(argc, argv, "--num-seq"))
    a.num_seq = to_int(get_arg(argc, argv, "--num-seq", "32"));
  if (has_arg(argc, argv, "--seq-len"))
    a.seq_len = to_int(get_arg(argc, argv, "--seq-len", "4096"));
  if (has_arg(argc, argv, "--passes"))
    a.passes = to_int(get_arg(argc, argv, "--passes", "1"));
  if (has_arg(argc, argv, "--block-size"))
    a.block_size = to_int(get_arg(argc, argv, "--block-size", "16"));
  if (has_arg(argc, argv, "--mode")) a.mode = get_arg(argc, argv, "--mode");
  if (has_arg(argc, argv, "--chunk-size"))
    a.chunk_size = to_int(get_arg(argc, argv, "--chunk-size", "512"));
  if (has_arg(argc, argv, "--kv-quant"))
    a.kv_quant = to_int(get_arg(argc, argv, "--kv-quant", "-1"));
  if (has_arg(argc, argv, "--csv")) a.csv = get_arg(argc, argv, "--csv");
  if (has_arg(argc, argv, "--tag")) a.tag = get_arg(argc, argv, "--tag");
  if (has_arg(argc, argv, "--debug-logits"))
    a.debug_logits = to_int(get_arg(argc, argv, "--debug-logits", "10"));
  if (a.num_seq <= 0 || a.seq_len < 2 || a.passes <= 0) {
    std::cerr << "参数非法: num_seq/seq_len/passes 必须为正(seq_len >= 2)\n";
    return -1;
  }
  // 掩码在 KVManager 的构造里也会校验,但那要等模型加载完之后;一次 typo 就是
  // 一次白跑,所以在解析处先拒绝(与 online/offline 同一口径)。只认 0 和 1:
  // 2/3/4 三条臂已随掩码收敛删除,把它们当别名会跑出一个没人能归因的数。
  if (a.kv_quant != -1 && a.kv_quant != 0 && a.kv_quant != scheduler::kKvQuantS4) {
    std::cerr << "参数非法: --kv-quant 必须是 0(off)或 6(S4),"
                 "或 -1(= 读 LLAMA_KV_QUANT);得到 "
              << a.kv_quant << "\n";
    return -1;
  }
  const bool chunked = a.mode == "chunked";
  if (!chunked && a.mode != "decode") {
    std::cerr << "参数非法: --mode 只支持 decode|chunked,收到 " << a.mode << "\n";
    return -1;
  }
  if (chunked && (a.chunk_size < 2 || a.chunk_size >= a.seq_len)) {
    // chunk_size == 1 退化成纯 decode(没有 prefill 行,seq_row_start 也无从填),
    // chunk_size >= seq_len 则整条序列只有一个 chunk、没有任何可打分的位置。
    std::cerr << "参数非法: chunked 模式要求 2 <= chunk-size < seq-len\n";
    return -1;
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
  a.tokenizer = resolve_default(a.tokenizer, exe_root);

  std::shared_ptr<model::Model> model;
  if (a.model_type == "qwen2") {
    model = std::make_shared<model::Qwen2Model>(base::TokenizerType::kEncodeBpe, a.tokenizer,
                                                a.model_dir, false);
  } else if (a.model_type == "qwen3") {
    model = std::make_shared<model::Qwen3Model>(base::TokenizerType::kEncodeBpe, a.tokenizer,
                                                a.model_dir, false);
  } else if (a.model_type == "llama" || a.model_type == "llama3") {
    model = std::make_shared<model::LLamaModel>(base::TokenizerType::kEncodeSpe, a.tokenizer,
                                                a.model_dir, false);
  } else {
    LOG(FATAL) << "未知 --model-type: " << a.model_type;
  }
  auto st = model->init(base::DeviceType::kDeviceCUDA);
  if (!st) LOG(FATAL) << "Init failed: " << st.get_err_code();

  const int vocab = model->vocab_size();
  const int kv_dim = model->kv_dim();
  const int layers = model->layer_num();
  const int window = model->seq_len();
  const int seq_len = a.seq_len;
  const int num_seq = a.num_seq;
  if (seq_len > window) {
    LOG(FATAL) << "--seq-len " << seq_len << " 超过模型窗口 " << window
               << "(RoPE/score scratch 都按窗口分配)";
  }
  if (model->compute_dtype() != base::DataType::kDataTypeBF16) {
    LOG(FATAL) << "本基准只支持 BF16 计算设备路径(CUDA)";
  }

  // ---------------- 语料 ----------------
  std::vector<bench::Request> dataset = load_dataset(a.dataset);
  if (dataset.empty()) LOG(FATAL) << "数据集为空或不可读: " << a.dataset;
  std::vector<std::vector<int32_t>> docs;
  docs.reserve(dataset.size());
  int64_t corpus_tokens = 0;
  for (const auto& r : dataset) {
    auto t = model->encode(r.prompt);
    if (t.empty()) continue;
    corpus_tokens += static_cast<int64_t>(t.size());
    docs.emplace_back(t.begin(), t.end());
  }
  if (docs.empty()) LOG(FATAL) << "语料编码后为空";
  // 语料量必须够覆盖一遍的全部序列,否则不同序列会共享文档前缀(它们的
  // NLL 就不再独立,统计量被高估)。
  const int64_t need = static_cast<int64_t>(seq_len) * num_seq * a.passes;
  if (corpus_tokens < need + seq_len) {
    LOG(WARNING) << "语料偏小: 共 " << corpus_tokens << " token,一遍需要 " << need
                 << "。序列间会复用文档,统计独立性下降(两臂仍然可比)。";
  }

  // ---------------- KV 池 ----------------
  auto kv_alloc = base::CUDADeviceAllocatorFactory::get_instance();
  // KV 量化掩码:CLI > env。两臂必须来自同一个二进制,否则不可比。
  const int kv_quant =
      a.kv_quant >= 0 ? a.kv_quant : scheduler::kv_quant_from_env();
  scheduler::KVManager kv(layers, num_seq, seq_len, kv_dim, model->kv_head_num(), kv_alloc,
                          base::DeviceType::kDeviceCUDA, a.block_size, kv_quant);
  const int table_stride = kv.max_blocks_per_seq();
  std::cout << "TAG " << a.tag << "\n"
            << "MODEL " << a.model_dir << "  layers=" << layers << " kv_dim=" << kv_dim
            << " vocab=" << vocab << "\n"
            << "CONFIG num_seq=" << num_seq << " seq_len=" << seq_len << " passes=" << a.passes
            << " block_size=" << a.block_size << " table_stride=" << table_stride
            << " mode=" << a.mode << (chunked ? (" chunk_size=" + std::to_string(a.chunk_size)) : "")
            << " kv_quant=" << kv_quant
            << " pool_gb=" << std::fixed << std::setprecision(2)
            << static_cast<double>(kv.pool_bytes()) / (1024.0 * 1024.0 * 1024.0) << "\n"
            << "CORPUS docs=" << docs.size() << " tokens=" << corpus_tokens << "\n";

  // 行(block table row)+ 每序列满额 block。整段序列长度已知,所以一次性
  // 申请到底,不需要 append_blocks 的惰性增长路径。
  scheduler::BlockAllocator* ba = kv.block_allocator();
  std::vector<int32_t> rows(num_seq, -1);
  for (int r = 0; r < num_seq; ++r) {
    rows[r] = ba->allocate_blocks(table_stride);
    if (rows[r] < 0) {
      LOG(FATAL) << "block 池不足: 第 " << r << " 行分配失败。pool=" << ba->num_blocks()
                 << " blocks max_batch=" << num_seq << " 每序列 " << table_stride;
    }
  }

  // ---------------- 张量 ----------------
  auto host_alloc = std::shared_ptr<base::DeviceAllocator>(
      base::PinnedCPUDeviceAllocatorFactory::get_instance());
  // 逐行独立:每行有自己的一维输入张量和 block table 行。
  // decode_step 读的是这两个 host tensor 的前 num_seq 个元素 / 前
  // num_seq*table_stride 个元素,所以这里按行连续排布。  // S4 residual-window index: one sequence slot per batch row. Filled per
  // step from the block-table row ids this driver owns.
  tensor::Tensor host_kv_slots =
      tensor::Tensor(base::DataType::kDataTypeInt32, std::max(num_seq, a.chunk_size), true,
                     host_alloc);
  if (host_kv_slots.is_empty()) LOG(FATAL) << "S4 kv_slots staging allocation failed";

  tensor::Tensor host_input_ids(base::DataType::kDataTypeInt32, num_seq, true, host_alloc);
  tensor::Tensor host_positions(base::DataType::kDataTypeInt32, num_seq, true, host_alloc);
  tensor::Tensor host_block_table(base::DataType::kDataTypeInt32, num_seq, table_stride, true,
                                  host_alloc);
  if (host_input_ids.is_empty() || host_positions.is_empty() || host_block_table.is_empty()) {
    LOG(FATAL) << "pinned 暂存分配失败";
  }
  for (int r = 0; r < num_seq; ++r) {
    ba->copy_block_table_row(rows[r], host_block_table.ptr<int32_t>() + r * table_stride,
                             table_stride);
    host_positions.ptr<int32_t>()[r] = 0;
    host_input_ids.ptr<int32_t>()[r] = 0;
  }

  tensor::Tensor logits(base::DataType::kDataTypeBF16, num_seq, vocab, true, kv_alloc);
  // targets[r * L + t] = 第 r 行第 t 步的正确答案(即 seq[r][t+1])。
  // 一次上传,之后每步只把指针 +t 传下去(见 ppl_nll_cu 的 stride 约定)。
  tensor::Tensor targets_dev(base::DataType::kDataTypeInt32, num_seq, seq_len, true, kv_alloc);
  tensor::Tensor nll_dev(base::DataType::kDataTypeFp32, num_seq, seq_len, true, kv_alloc);
  if (logits.is_empty() || targets_dev.is_empty() || nll_dev.is_empty()) {
    LOG(FATAL) << "logits/targets/nll 设备张量分配失败";
  }
  // chunked 模式的暂存:一步最多 chunk_size 行(行 0 是 decode 行,logits 也只
  // 为它产出,所以单开一个 [1, vocab] 的设备张量);seq_row_start 是长度
  // num_prefill_seqs+1 的行地图。decode 模式的张量形状约定是 dim0=num_seq
  // (decode_step 直接按它推 batch),两套分开、互不影响。
  tensor::Tensor c_input_ids, c_positions, c_block_table, c_logits_row;
  tensor::Tensor c_seq_row_start_dev, c_seq_row_start_host;
  if (chunked) {
    c_input_ids = tensor::Tensor(base::DataType::kDataTypeInt32, a.chunk_size, true, host_alloc);
    c_positions = tensor::Tensor(base::DataType::kDataTypeInt32, a.chunk_size, true, host_alloc);
    c_block_table = tensor::Tensor(base::DataType::kDataTypeInt32, a.chunk_size, table_stride,
                                   true, host_alloc);
    c_logits_row = tensor::Tensor(base::DataType::kDataTypeBF16, 1, vocab, true, kv_alloc);
    c_seq_row_start_dev = tensor::Tensor(base::DataType::kDataTypeInt32, 2, true, kv_alloc);
    c_seq_row_start_host = tensor::Tensor(base::DataType::kDataTypeInt32, 2, true, host_alloc);
    if (c_input_ids.is_empty() || c_positions.is_empty() || c_block_table.is_empty() ||
        c_logits_row.is_empty() || c_seq_row_start_dev.is_empty() ||
        c_seq_row_start_host.is_empty()) {
      LOG(FATAL) << "chunked 模式的 pinned/设备暂存分配失败";
    }
  }
  std::vector<int32_t> tgt_host(static_cast<size_t>(num_seq) * seq_len, 0);
  // 打分位置掩码:decode 模式覆盖每个位置(0..seq_len-2),chunked 模式只覆盖
  // chunk 边界位置;聚合/分桶只走掩码为 1 的位置,于是两种模式共用同一套统计。
  std::vector<uint8_t> scored(static_cast<size_t>(num_seq) * seq_len, 0);
  // 主机侧累加器:[row][pos] 的 NLL 之和(跨遍累加),以及分母计数。
  std::vector<double> nll_sum(static_cast<size_t>(num_seq) * seq_len, 0.0);
  std::vector<double> pos_sum(seq_len, 0.0);   // 按位置聚合(所有行/遍)
  std::vector<int64_t> pos_cnt(seq_len, 0);
  std::vector<double> row_sum(num_seq, 0.0);   // 按序列聚合
  std::vector<int64_t> row_cnt(num_seq, 0);
  double total_sum = 0.0;
  int64_t total_cnt = 0;

  cudaStream_t stream = model->cuda_stream();
  std::vector<float> nll_host(static_cast<size_t>(num_seq) * seq_len, 0.f);

  size_t cursor = 0;
  const auto t_start = std::chrono::steady_clock::now();
  for (int pass = 0; pass < a.passes; ++pass) {
    std::vector<std::vector<int32_t>> seqs = build_sequences(docs, num_seq, seq_len, cursor);
    // 目标 token 一次填好
    for (int r = 0; r < num_seq; ++r) {
      for (int t = 0; t + 1 < seq_len; ++t) {
        tgt_host[static_cast<size_t>(r) * seq_len + t] = seqs[r][t + 1];
      }
      tgt_host[static_cast<size_t>(r) * seq_len + seq_len - 1] = seqs[r][seq_len - 1];
    }
    cudaMemcpyAsync(targets_dev.ptr<int32_t>(), tgt_host.data(),
                    static_cast<size_t>(num_seq) * seq_len * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);

    // 每个序列重头开始:释放旧 block 表、重新申请满额 block。
    // (重申请而不是就地复用,是为了让所有遍走完全相同的分配路径。)
    for (int r = 0; r < num_seq; ++r) {
      ba->free_all(rows[r]);
      rows[r] = ba->allocate_blocks(table_stride);
      if (rows[r] < 0) LOG(FATAL) << "第 " << pass << " 遍重新分配 block 失败";
      ba->copy_block_table_row(rows[r], host_block_table.ptr<int32_t>() + r * table_stride,
                               table_stride);
    }

    if (chunked) {
      // ---- chunked-prefill 驱动:每步推一个 chunk ----
      // 行 0 = chunk 的最后一个 token(位置 e-1),以 decode 行喂入 —— teacher
      // forcing 要的"下一个 token"的 logits 恰好由它产出;行 1..len-1 = chunk
      // 里其余 token,作为 prefill 行推进 KV。行序/行地图按 Model::forward_batch
      // 的混合步约定:decode 行在 [0,1),prefill 行是同一个序列的一整段,
      // seq_row_start = [1, len](len 是哨兵)。
      const int chunk_steps = (seq_len + a.chunk_size - 1) / a.chunk_size;
      const size_t words_per_row = static_cast<size_t>(table_stride) * sizeof(int32_t);
      for (int r = 0; r < num_seq; ++r) {
        for (int ci = 0; ci < chunk_steps; ++ci) {
          const int s = ci * a.chunk_size;
          const int e = std::min(s + a.chunk_size, seq_len);
          const int len = e - s;  // 本步的 batch 行数
          int32_t* in_ids = c_input_ids.ptr<int32_t>();
          int32_t* pos = c_positions.ptr<int32_t>();
          in_ids[0] = seqs[r][e - 1];
          pos[0] = e - 1;
          for (int i = 1; i < len; ++i) {
            in_ids[i] = seqs[r][s + i - 1];
            pos[i] = s + i - 1;
          }
          // 一步里所有行都属于序列 r:block table 的每一行都指向它的 block 行
          //(与调度器逐行 copy_block_table_row 等价,只是这里每行同一份数据)。
          int32_t* bt = c_block_table.ptr<int32_t>();
          ba->copy_block_table_row(rows[r], bt, table_stride);
          for (int i = 1; i < len; ++i) {
            std::memcpy(bt + static_cast<size_t>(i) * table_stride, bt, words_per_row);
          }
          // 活跃 block 表列数:最远的行是 decode 行(位置 e-1)。
          const int table_cols = std::min((e - 1) / a.block_size + 1, table_stride);
          int32_t* rs = c_seq_row_start_host.ptr<int32_t>();
          rs[0] = 1;
          rs[1] = len;
          cudaMemcpyAsync(c_seq_row_start_dev.ptr<int32_t>(), rs, 2 * sizeof(int32_t),
                          cudaMemcpyHostToDevice, stream);
          tensor::Tensor in_view = c_input_ids;
          tensor::Tensor pos_view = c_positions;
          tensor::Tensor bt_view = c_block_table;
          in_view.reshape({len});
          pos_view.reshape({len});
          bt_view.reshape({len, table_stride});
          const bool s4 = kv.quantized();
          tensor::Tensor slots_view = host_kv_slots;
          slots_view.reshape({len});
          for (int i = 0; i < len; ++i) slots_view.index<int32_t>(i) = rows[r];
          auto dst = model->forward_batch(in_view, pos_view, bt_view, kv.key_cache(),
                                          kv.value_cache(), c_logits_row,
                                          /*need_logits=*/true, /*num_decode_rows=*/1,
                                          &c_seq_row_start_dev, /*num_prefill_seqs=*/1, nullptr,
                                          table_cols, &kv.key_scale_cache(),
                                          &kv.value_scale_cache(),
                                          s4 ? &kv.key_zero_cache() : nullptr,
                                          s4 ? &kv.value_zero_cache() : nullptr,
                                          s4 ? &kv.key_residual_cache() : nullptr,
                                          s4 ? &slots_view : nullptr);
          if (!dst) {
            LOG(FATAL) << "chunked forward_batch 失败 @seq=" << r << " chunk=" << ci << ": "
                       << dst.get_err_code();
          }
          // 与 decode 循环同理(那条注释同样适用):这一步的 H2D 上传读的是
          // 上面的主机暂存,主机重写它之前必须把流排空。混合步不走 CUDA graph
          // (chunk 形状逐段可变),没有捕获节点,排空的理由同样是 staging 覆盖。
          cudaStreamSynchronize(stream);
          // 只对 decode 行打分。最后一个 chunk 的末尾没有下一个 token(序列到
          // 此为止),它的 token 仍然进模型,只是不出 NLL —— 整条序列的 KV
          // 覆盖保持完整。
          if (e < seq_len) {
            const size_t idx = static_cast<size_t>(r) * seq_len + (e - 1);
            ppl_nll_cu(c_logits_row.ptr<uint16_t>(), targets_dev.ptr<int32_t>() + idx,
                       nll_dev.ptr<float>() + idx, vocab, /*rows=*/1, seq_len, seq_len,
                       reinterpret_cast<void*>(stream));
            scored[idx] = 1;
          }
        }
        std::cout << "CHUNKED seq " << r << " done  elapsed_s=" << std::fixed
                  << std::setprecision(1)
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
                         .count()
                  << std::endl;
      }
    } else {
      for (int t = 0; t + 1 < seq_len; ++t) {
        for (int r = 0; r < num_seq; ++r) {
          host_input_ids.ptr<int32_t>()[r] = seqs[r][t];
          host_positions.ptr<int32_t>()[r] = t;
          scored[static_cast<size_t>(r) * seq_len + t] = 1;
        }
        // 活跃 block 表列数:所有行都在位置 t,列 = t / block_size + 1。
        const int table_cols = std::min(t / a.block_size + 1, table_stride);
        const bool s4 = kv.quantized();
        tensor::Tensor slots_view = host_kv_slots;
        slots_view.reshape({num_seq});
        for (int r = 0; r < num_seq; ++r) slots_view.index<int32_t>(r) = rows[r];
        auto dst = model->decode_step(host_input_ids, host_positions, host_block_table,
                                      kv.key_cache(), kv.value_cache(), logits, table_cols,
                                      &kv.key_scale_cache(), &kv.value_scale_cache(),
                                      s4 ? &kv.key_zero_cache() : nullptr,
                                      s4 ? &kv.value_zero_cache() : nullptr,
                                      s4 ? &kv.key_residual_cache() : nullptr,
                                      s4 ? &slots_view : nullptr);
        if (!dst) LOG(FATAL) << "decode_step 失败 @" << t << ": " << dst.get_err_code();

        // 这一步的同步是正确性要求,不是保守起见:
        //   decode_step 在"主机时间"把 host input_ids/positions 拷进 scratch,而
        //   读 scratch 的是被捕获进 CUDA graph 的 H2D 上传节点 —— 它要等到流里
        //   排到才执行。主机比设备快得多,没有同步就会在设备消费上一份 staging
        //   之前把它覆盖掉,于是每一步实际喂进去的是"之后某一步"的 token。
        //   调度器天然没有这个问题:它每步都要把采样结果 D2H 回主机,那次拷贝
        //   顺带把流排空了。本基准刻意不做逐步 D2H(NLL 在设备端归约),所以必须
        //   显式排空。
        //   症状:上下文被忽略(输出接近均匀分布,NLL ≈ ln(vocab)),而最后一步
        //   恰好正确 —— 循环结束后没人再覆盖 staging。
        cudaStreamSynchronize(stream);
        ppl_nll_cu(logits.ptr<uint16_t>(), targets_dev.ptr<int32_t>() + t,
                   nll_dev.ptr<float>() + t, vocab, num_seq, seq_len, seq_len,
                   reinterpret_cast<void*>(stream));
      }
    }
    // 诊断:把最后一步 row 0 的 logits 拉回来，打印 top-N 与正确答案的排名。
    // 模型正常工作时 top-1 应当（在重复语料上必然）命中 target。
    // (只对 decode 模式有意义:chunked 模式每步都会覆盖 logits 行,最后一步
    //  的 logits 不再对应某个固定位置。)
    if (!chunked && a.debug_logits > 0 && pass == 0) {
      std::vector<uint16_t> lg(static_cast<size_t>(num_seq) * vocab);
      cudaMemcpyAsync(lg.data(), logits.ptr<uint16_t>(), lg.size() * sizeof(uint16_t),
                      cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      auto to_f = [](uint16_t h) {
        const uint32_t u = static_cast<uint32_t>(h) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
      };
      const int t_last = seq_len - 2;
      const int target = seqs[0][t_last + 1];
      std::vector<std::pair<float, int>> top;
      top.reserve(vocab);
      for (int v = 0; v < vocab; ++v) top.emplace_back(to_f(lg[v]), v);
      const int n = std::min<int>(a.debug_logits, vocab);
      std::partial_sort(top.begin(), top.begin() + n, top.end(),
                        [](const std::pair<float, int>& x, const std::pair<float, int>& y) {
                          return x.first > y.first;
                        });
      int target_rank = -1;
      for (int v = 0; v < vocab; ++v) {
        if (top[v].second == target) {
          target_rank = v;
          break;
        }
      }
      std::cout << "DEBUG_LOGITS row0 t=" << t_last << " input_tok=" << seqs[0][t_last]
                << " target_tok=" << target << " target_logit=" << to_f(lg[target])
                << " target_rank(in_top" << n << ")=" << target_rank << "\n";
      for (int i = 0; i < n; ++i) {
        std::cout << "    #" << i << " tok=" << top[i].second << " logit=" << top[i].first << "\n";
      }
    }

    cudaMemcpyAsync(nll_host.data(), nll_dev.ptr<float>(),
                    static_cast<size_t>(num_seq) * seq_len * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      LOG(FATAL) << "第 " << pass << " 遍 NLL 回传失败: " << cudaGetErrorString(cudaGetLastError());
    }

    // 只聚合打分位置(scored 掩码):decode 模式 = 每个位置,chunked 模式 =
    // chunk 边界位置。分母跟着掩码走,所以两种模式的 PPL/分桶口径一致。
    for (int r = 0; r < num_seq; ++r) {
      for (int t = 0; t + 1 < seq_len; ++t) {
        if (!scored[static_cast<size_t>(r) * seq_len + t]) continue;
        const double v = nll_host[static_cast<size_t>(r) * seq_len + t];
        nll_sum[static_cast<size_t>(r) * seq_len + t] += v;
        pos_sum[t] += v;
        pos_cnt[t] += 1;
        row_sum[r] += v;
        row_cnt[r] += 1;
        total_sum += v;
        total_cnt += 1;
      }
    }
    std::cout << "PASS " << pass << " done  elapsed_s=" << std::fixed << std::setprecision(1)
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count()
              << std::endl;
  }

  // ---------------- 统计 ----------------
  auto ppl_of = [](double sum, int64_t n) { return std::exp(sum / static_cast<double>(n)); };
  const double ppl_total = ppl_of(total_sum, total_cnt);

  // 序列间标准差:各序列自己的 PPL,再看它们在均值附近的散布。这是判断
  // 两臂差异是否超出语料噪声的主要依据(位置分桶内的 token 高度相关,
  // 不能按 token 数当独立样本算 SE)。
  std::vector<double> row_ppl(num_seq, 0.0);
  for (int r = 0; r < num_seq; ++r) {
    row_ppl[r] = row_cnt[r] > 0 ? ppl_of(row_sum[r], row_cnt[r]) : 0.0;
  }
  double m = 0.0;
  for (int r = 0; r < num_seq; ++r) m += row_ppl[r];
  m /= num_seq;
  double var = 0.0;
  for (int r = 0; r < num_seq; ++r) var += (row_ppl[r] - m) * (row_ppl[r] - m);
  var /= (num_seq > 1 ? num_seq - 1 : 1);
  const double se = std::sqrt(var / num_seq);

  std::cout << "\n===== PPL " << a.tag << " =====\n"
            << std::setprecision(6) << "PPL_total        " << ppl_total << "  (tokens="
            << total_cnt << ")\n"
            << "PPL_seq_mean     " << m << "  +/- " << se << " (SE over " << num_seq
            << " sequences, sd=" << std::sqrt(var) << ")\n"
            << "NLL_mean         " << (total_sum / static_cast<double>(total_cnt)) << "\n";
  if (chunked) {
    std::cout << "NOTE chunked 模式: 每序列只对 chunk 边界位置打分("
              << (seq_len - 1) / a.chunk_size << " 个 / 序列),与 decode-PPL 在同一位置逐点可比\n";
  }

  std::cout << "\nBUCKET  pos_lo  pos_hi   NLL_mean   tokens\n";
  for (int i = 0; i + 1 < static_cast<int>(sizeof(kBucketEdges) / sizeof(int)); ++i) {
    const int lo = kBucketEdges[i];
    const int hi = kBucketEdges[i + 1];
    if (lo >= seq_len) break;
    double s = 0.0;
    int64_t n = 0;
    for (int t = lo; t < std::min(hi, seq_len) && t + 1 < seq_len; ++t) {
      s += pos_sum[t];
      n += pos_cnt[t];
    }
    if (n == 0) continue;
    std::cout << std::setw(6) << lo << std::setw(8) << hi << "  " << std::setprecision(5)
              << (s / static_cast<double>(n)) << "   " << n << "\n";
  }

  if (!a.csv.empty()) {
    std::ofstream f(a.csv);
    if (!f) {
      LOG(ERROR) << "无法写出 CSV: " << a.csv;
    } else {
      f << "# tag=" << a.tag << " num_seq=" << num_seq << " seq_len=" << seq_len
        << " passes=" << a.passes << " block_size=" << a.block_size
        << " mode=" << a.mode << (chunked ? (" chunk_size=" + std::to_string(a.chunk_size)) : "")
        << " kv_quant=" << kv_quant << "\n";
      f << "# ppl_total=" << std::setprecision(8) << ppl_total << "\n";
      f << "pos,nll_mean,tokens\n";
      for (int t = 0; t + 1 < seq_len; ++t) {
        if (pos_cnt[t] == 0) continue;
        f << t << "," << std::setprecision(8) << (pos_sum[t] / static_cast<double>(pos_cnt[t]))
          << "," << pos_cnt[t] << "\n";
      }
      f << "# rows\nrow,ppl,tokens\n";
      for (int r = 0; r < num_seq; ++r) {
        f << r << "," << std::setprecision(8) << row_ppl[r] << "," << row_cnt[r] << "\n";
      }
      std::cout << "\nCSV " << a.csv << "\n";
    }
  }
  return 0;
}
