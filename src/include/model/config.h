#ifndef SRC_INCLUDE_MODEL_CONFIG_H_
#define SRC_INCLUDE_MODEL_CONFIG_H_
namespace model {
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
};

struct ModelConfig {
  int32_t dim = 0;
  int32_t hidden_dim = 0;
  int32_t layer_num = 0;
  int32_t head_num = 0;
  int32_t kv_head_num = 0;
  int32_t vocab_size = 0;
  int32_t seq_len = 0;
  int32_t immediate_dim_ = 0;  // Qwen3 intermediate FFN dimension (always present for ABI stability)
  float rope_theta_ = 1000000.0f;  // RoPE base frequency (1e6 for Qwen, 5e5 for Llama)
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
};
}  // namespace model
#endif  
