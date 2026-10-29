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

  base::Status set_bias(int32_t idx, int32_t dims, const void* bias_ptr,
                        base::DeviceType device_type,
                        base::DataType data_type = base::DataType::kDataTypeFp32);

  // Owning form: keeps the tensor's storage alive with the layer (used by the
  // fused projections, whose stacked bias is built by the model).
  base::Status set_bias(int32_t idx, const tensor::Tensor& bias);

  tensor::Tensor& get_bias(int32_t idx);

  const tensor::Tensor& get_bias(int32_t idx) const;

  // AWQ W4A16 weights: the int4 codes stay packed in memory and the GEMM
  // dequantizes them on the fly. qweight is int32 [in_features, packed_cols],
  // qzeros int32 [groups, packed_cols] (empty for symmetric quantization) and
  // scales the compute dtype [groups, out_features].
  //
  // A fused projection owns one stacked copy of the packed tensors; the
  // per-projection layers reuse it with `out_offset` selecting their output
  // rows, so no second weight copy is made for the fused case.
  base::Status set_awq_weight(const tensor::Tensor& qweight, const tensor::Tensor& qzeros,
                              const tensor::Tensor& scales, int32_t group_size,
                              bool zero_point, int32_t out_offset, int32_t out_features);

  bool is_awq() const { return awq_; }

  const tensor::Tensor& awq_qweight() const { return awq_qweight_; }

  const tensor::Tensor& awq_qzeros() const { return awq_qzeros_; }

  const tensor::Tensor& awq_scales() const { return awq_scales_; }

  int32_t awq_group_size() const { return awq_group_size_; }

  bool awq_zero_point() const { return awq_zero_point_; }

  int32_t awq_out_offset() const { return awq_out_offset_; }

  bool has_bias() const { return has_bias_; }

  int32_t out_features() const { return dim0_; }
  int32_t in_features() const { return dim1_; }

  void to_cuda() override;

  void set_batch_size(int32_t batch) { batch_size_ = batch; }
  int32_t batch_size() const { return batch_size_; }

 private:
  int32_t dim0_ = 0;
  int32_t dim1_ = 0;
  int32_t batch_size_ = 1;
  bool has_bias_ = false;
  std::vector<tensor::Tensor> bias_;

  // Packed AWQ weight state (see set_awq_weight).
  bool awq_ = false;
  int32_t awq_group_size_ = 0;
  bool awq_zero_point_ = true;
  int32_t awq_out_offset_ = 0;
  tensor::Tensor awq_qweight_;
  tensor::Tensor awq_qzeros_;
  tensor::Tensor awq_scales_;
};
}  // namespace op
#endif
