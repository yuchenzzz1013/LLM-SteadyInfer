#ifndef SRC_INCLUDE_MODEL_CONFIG_H_
#define SRC_INCLUDE_MODEL_CONFIG_H_
#include <cstdint>
#include <string>
#include <vector>
namespace model {
// Weight-only quantization schemes understood by the runtime. The scheme is
// orthogonal to the KV-cache quantization chosen by the scheduler: this one
// describes how the checkpoint's linear weights are stored.
enum class WeightQuantType : uint8_t {
  kNone = 0,
  // AWQ (Activation-aware Weight Quantization), W4A16:
  //   qweight [K, N/8] int32 (8 signed 4-bit codes per int32)
  //   qzeros  [K/group_size, N/8] int32
  //   scales  [K/group_size, N] (checkpoint float dtype)
  kAwqInt4 = 1,
};

// Fields consumed from an HF config.json (Qwen2/Qwen3/Llama families).
struct HfConfig {
  int64_t hidden_size = 0;
  int64_t intermediate_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;  // absent in HF -> defaults to num_attention_heads
  int64_t vocab_size = 0;
  int64_t max_position_embeddings = 0;
  int64_t head_dim = 0;  // absent -> hidden_size / num_attention_heads
  float rope_theta = 1000000.0f;
  bool tie_word_embeddings = false;

  // quant_method is empty when the checkpoint is unquantized (the common
  // case). Everything below is only meaningful for a quantized checkpoint and
  // is parsed from config.json's "quantization_config" block, so loading a
  // checkpoint is driven by the checkpoint itself rather than a CLI flag.
  std::string quant_method;   // "awq" | "" | unsupported
  std::string quant_version;  // AWQ packing layout ("gemm" for the supported one)
  int32_t quant_bits = 4;
  int32_t quant_group_size = 128;
  bool quant_zero_point = true;
  // Modules listed here keep full-precision ".weight" tensors in the
  // checkpoint. The layer builder also treats the absence of a ".qweight"
  // tensor as authoritative, so a checkpoint that omits the list still works.
  std::vector<std::string> modules_to_not_convert;
};

struct TransformerConfig {
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  int32_t head_size_ = 0;
  int32_t vocab_size_ = 0;

  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;
  bool is_shared_weight_ = false;
  int32_t immediate_dim_ = 0;  // Qwen3 intermediate FFN dimension (always present for ABI stability)
  float rope_theta_ = 1000000.0f;  // RoPE base frequency (1e6 for Qwen, 5e5 for Llama)

  // Resolution of the checkpoint's weight storage. AWQ modules keep their
  // int4 codes packed on the device and dequantize inside the GEMM kernel.
  WeightQuantType weight_quant_ = WeightQuantType::kNone;
  int32_t weight_quant_group_size_ = 0;
  bool weight_quant_zero_point_ = true;
  // Modules the checkpoint explicitly keeps full precision. The presence of a
  // packed "<module>.qweight" is the authoritative signal; an entry here wins
  // when a checkpoint happens to ship both forms.
  std::vector<std::string> weight_quant_exclude;
};
}  // namespace model
#endif
