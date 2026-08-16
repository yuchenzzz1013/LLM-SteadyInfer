#ifndef SRC_INCLUDE_MODEL_LLAMA3_H_
#define SRC_INCLUDE_MODEL_LLAMA3_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {

struct LLamaLayers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  // Fused QKV projection (M3): one GEMM with the stacked weight
  // [dim + 2*kv_dim, dim] instead of three (LLaMA has no QKV bias).
  // The *_src_ tensors keep the fused weight buffers alive (set_weight
  // stores a non-owning external view into them).
  bool fused_qkv_enabled_ = false;
  std::vector<std::shared_ptr<op::Layer>> fused_qkv_layers_;
  std::vector<tensor::Tensor> fused_qkv_weight_src_;
  std::vector<tensor::Tensor> fused_qkv_bias_src_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class LLamaModel : public Model {
 public:
  explicit LLamaModel(base::TokenizerType tokenizer_type, std::string token_path,
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
      BatchScratch* scratch = nullptr) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  void sync_stream() const override;

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  // Build the fused QKV MatmulLayers from the (already device-resident)
  // wq/wk/wv weights. Must run after LLamaLayers::to_cuda in init_mem.
  void build_fused_qkv_layers();

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

 private:
  std::unique_ptr<LLamaLayers> llama_layers_;
};
}  // namespace model

#endif