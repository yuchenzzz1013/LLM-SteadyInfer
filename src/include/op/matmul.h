#ifndef SRC_INCLUDE_OP_MATMUL_H_
#define SRC_INCLUDE_OP_MATMUL_H_
#include <base/cuda_config.h>
#include "layer.h"
namespace op {
class MatmulLayer : public LayerParam {
 public:
  explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                       bool is_quant_layer = false, bool has_bias = false);

  base::Status check() const override;

  base::Status forward() override;

  base::Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,
                          base::DeviceType device_type);

  tensor::Tensor& get_bias(int32_t idx);

  const tensor::Tensor& get_bias(int32_t idx) const;

  void to_cuda() override;

  void set_batch_size(int32_t batch) { batch_size_ = batch; }
  int32_t batch_size() const { return batch_size_; }

 private:
  int32_t dim0_ = 0;
  int32_t dim1_ = 0;
  int32_t batch_size_ = 1;
  bool has_bias_ = false;
  std::vector<tensor::Tensor> bias_;
};
}  // namespace op
#endif
