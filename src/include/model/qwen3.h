#ifndef SRC_INCLUDE_MODEL_QWEN3_H_
#define SRC_INCLUDE_MODEL_QWEN3_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {
struct Qwen3Layers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  // Fused QKV projection (M3): one GEMM with the stacked weight
  // [dim + 2*kv_dim, hidden_dim] instead of three (Qwen3 has no QKV bias).
  // The fused layer owns its stacked storage; on CUDA the wq/wk/wv layers
  // view slices of it, so the projections have a single device residence.
  bool fused_qkv_enabled_ = false;
  std::vector<std::shared_ptr<op::Layer>> fused_qkv_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;

  // Fused gate/up projection: one GEMM with the stacked weight
  // [2 * immediate_dim (FFN width), hidden_dim (FFN input)] instead of w1 and
  // w3 separately, paired with kernel::swiglu_kernel_cu_fused (which reads
  // gate and up from the two halves of each output row). The fused layer owns
  // its stacked storage; on CUDA w1/w3 view the two halves of it, so the pair
  // has a single device residence.
  bool fused_w13_enabled_ = false;
  std::vector<std::shared_ptr<op::Layer>> fused_w13_layers_;
  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class Qwen3Model : public Model {
 public:
  explicit Qwen3Model(base::TokenizerType tokenizer_type, std::string token_path,
                      std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  base::Status forward_batch(
      const tensor::Tensor& input_ids,
      const tensor::Tensor& positions,
      const tensor::Tensor& block_table,
      tensor::Tensor& key_cache,
      tensor::Tensor& value_cache,
      tensor::Tensor& logits,
      bool need_logits = true,
      int32_t num_decode_rows = 0,
      const tensor::Tensor* seq_row_start = nullptr,
      int32_t num_prefill_seqs = 0,
      BatchScratch* scratch = nullptr,
      int32_t table_cols = 0,
      const tensor::Tensor* key_scale = nullptr,
      const tensor::Tensor* value_scale = nullptr,
      const tensor::Tensor* key_zero = nullptr,
      const tensor::Tensor* value_zero = nullptr,
      const tensor::Tensor* key_residual = nullptr,
      const tensor::Tensor* kv_slots = nullptr) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  void sync_stream() const override;

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;


  // Build the fused QKV MatmulLayers from the host weight map (see
  // weight_map_) and, on CUDA, re-point wq/wk/wv at slices of the stacked
  // buffers so the projections have a single device residence. Must run
  // BEFORE Qwen3Layers::to_cuda in init_mem: the stacked buffer is built from
  // the host weights, and the re-pointed layers make to_cuda() a no-op for
  // them (a layer weight already tagged CUDA is not copied again).
  void build_fused_qkv_layers();

  // Stack w1/w3 into one [2 * immediate_dim, hidden_dim] MatmulLayer (gate
  // rows first, then up) and re-point w1/w3 at its two halves. Must run
  // BEFORE Qwen3Layers::to_cuda in init_mem, for the reason above.
  void build_fused_w13_layers();

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

 private:
  std::unique_ptr<Qwen3Layers> qwen_layers_;
};
}  // namespace model

#endif