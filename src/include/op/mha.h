#ifndef SRC_INLCUDE_OP_MHA_H
#define SRC_INLCUDE_OP_MHA_H
#include <base/cuda_config.h>
#include "layer.h"
namespace op {
class MultiHeadAttention : public op::Layer {
 public:
  // kv_head_num is the number of GQA KV heads. The KV head for attention head
  // h is h * kv_head_num / head_num, which handles uneven GQA splits (e.g.
  // Qwen3-4B: 32 query heads / 8 KV heads) correctly — unlike head / (head_num
  // / kv_head_num), which mis-maps heads and reads out of bounds.
  // No seq_len/context-window parameter: forward() derives the KV length from
  // the cache tensor it is handed (key_cache dims_size-2), which is correct for
  // both the legacy internal cache and any external cache. The old seq_len
  // argument was stored and never read — a stale duplicate of the window that
  // could only ever drift from the tensor actually used.
  explicit MultiHeadAttention(base::DeviceType device_type, int32_t layer_index,
                              int32_t kv_head_num, int32_t kv_dim, int32_t head_num,
                              int32_t head_size);

  base::Status check() const override;

  void set_pos(int32_t pos);
  void set_layer_idx(int32_t layer_idx);

  base::Status forward() override;

 private:
  int32_t layer_index_ = 0;
  int32_t pos_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t kv_dim_ = 0;
  int32_t head_num_ = 0;
  int32_t head_size_ = 0;
};
}  // namespace op
#endif
