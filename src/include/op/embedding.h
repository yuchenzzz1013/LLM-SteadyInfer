#ifndef SRC_INCLUDE_OP_EMBEDDING_H_
#define SRC_INCLUDE_OP_EMBEDDING_H_
#include <utility>
#include "layer.h"
namespace op {
struct EmbeddingOutput {
  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;
  explicit EmbeddingOutput(tensor::Tensor input_tokens, tensor::Tensor input_embeddings,
                           tensor::Tensor input_token_num)
      : input_tokens(std::move(input_tokens)),
        input_embeddings(std::move(input_embeddings)),
        input_token_num(std::move(input_token_num)) {}
};

class EmbeddingLayer : public LayerParam {
 public:
  explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                          int32_t vocab_size);

  base::Status check() const override;

  base::Status forward() override;

  void set_batch_size(int32_t batch) { batch_size_ = batch; }
  int32_t batch_size() const { return batch_size_; }

 private:
  int32_t dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t vocab_size_ = 0;
  int32_t batch_size_ = 1;
};
}  // namespace op
#endif
