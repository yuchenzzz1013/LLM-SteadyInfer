#include "op/embedding.h"
#include "kernels/cpu/emb_kernel.h"
#include "kernels/kernels_interface.h"
#include "op/layer.h"
namespace op {
EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                               int32_t vocab_size)
    : dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size),
      LayerParam(device_type, LayerType::kLayerEmbedding, false, "Embedding") {
  reset_weight_size(1);
  reset_input_size(2);
  reset_output_size(1);
}

base::Status EmbeddingLayer::check() const {
  const auto& input_tensor = get_input(0);
  const int32_t total_tokens = get_input(1).size();
  if (total_tokens > input_tensor.size()) {
    return base::error::InvalidArgument("The number of input tensor is greater than seq len.");
  }

  // Auto-detect batch from input dimensions. Tokens may live on the host
  // (normal path) or on the device (CUDA-Graph path: the staged tokens are
  // uploaded once and the embedding kernel reads them directly — accepting
  // device input avoids a per-call clone inside the captured region).
  const base::DeviceType token_device =
      (device_type_ == base::DeviceType::kDeviceCUDA) ? input_tensor.device_type()
                                                      : base::DeviceType::kDeviceCPU;
  int input_dims = input_tensor.dims_size();
  base::Status status;
  if (input_dims == 2) {
    status = check_tensor_with_dim(input_tensor, token_device,
                                    base::DataType::kDataTypeInt32,
                                    input_tensor.get_dim(0), input_tensor.get_dim(1));
  } else {
    status = check_tensor_with_dim(input_tensor, token_device,
                                    base::DataType::kDataTypeInt32, total_tokens);
  }
  if (!status) {
    LOG(ERROR) << "The input tensor error in the embedding layer.";
    return status;
  }

  status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, vocab_size_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the embedding layer.";
    return status;
  }

  // Output check: match input shape + dim_
  const auto& output = get_output(0);
  if (input_dims == 2) {
    status = check_tensor_with_dim(output, device_type_, data_type_,
                                    input_tensor.get_dim(0), input_tensor.get_dim(1), dim_);
  } else {
    status = check_tensor_with_dim(output, device_type_, data_type_, total_tokens, dim_);
  }
  if (!status) {
    LOG(ERROR) << "The output tensor error in the embedding layer.";
    return status;
  }
  return base::error::Success();
}

base::Status EmbeddingLayer::forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_emb_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), vocab_size_,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::StatusCode::kSuccess;
}
}  // namespace op