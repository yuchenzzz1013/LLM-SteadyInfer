#include "op/matmul.h"
#include "kernels/cpu/matmul_kernel.h"
#include "kernels/kernels_interface.h"
namespace op {
MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                         bool is_quant_layer, bool has_bias)
    : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul"),
      dim0_(dim0),
      dim1_(dim1),
      has_bias_(has_bias) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
  if (has_bias_) {
    bias_.resize(1);
  }
}

base::Status MatmulLayer::check() const {
  base::Status status;
  const auto& input = get_input(0);
  int input_dims = input.dims_size();

  if (input_dims == 2) {
    // Batch mode: [batch, M]
    status = check_tensor_with_dim(input, device_type_, data_type_, input.get_dim(0), dim1_);
    if (!status) {
      LOG(ERROR) << "The input tensor error in the matmul layer (batch mode).";
      return status;
    }
  } else {
    status = check_tensor_with_dim(input, device_type_, data_type_, dim1_);
    if (!status) {
      LOG(ERROR) << "The input tensor error in the matmul layer.";
      return status;
    }
  }

  if (!is_quant_layer_) {
    status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim0_, dim1_);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";
      return status;
    }
  } else {
    status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeInt8,
                                   dim0_, dim1_);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";
      return status;
    }
  }

  if (is_quant_layer_) {
    status = check_tensor_with_dim(scales_, device_type_, base::DataType::kDataTypeFp32, scales_.size());
    if (!status) {
      LOG(ERROR) << "The scale tensor error in the matmul layer.";
      return status;
    }
  }

  const auto& output = get_output(0);
  if (input_dims == 2) {
    status = check_tensor_with_dim(output, device_type_, data_type_, input.get_dim(0), dim0_);
    if (!status) {
      LOG(ERROR) << "The output tensor error in the matmul layer (batch mode).";
      return status;
    }
  } else {
    status = check_tensor_with_dim(output, device_type_, data_type_, dim0_);
    if (!status) {
      LOG(ERROR) << "The output tensor error in the matmul layer.";
      return status;
    }
  }
  return base::error::Success();
}

base::Status MatmulLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  const auto& input = get_input(0);
  // Auto-detect batch from input dimensions: [M] → batch=1, [batch, M] → batch>1
  int32_t batch = 1;
  if (input.dims_size() == 2) {
    batch = input.get_dim(0);
  }

  const auto& weight = get_weight(0);
  auto& output = get_output(0);
  int32_t M = dim1_;  // input dim per sample
  int32_t K = dim0_;  // output dim per sample
  int32_t total_input = batch * M;
  int32_t total_output = batch * K;

  // Always flatten to 1D for the kernel: [batch*M] x [K, M] -> [batch*K]
  tensor::Tensor input_flat(base::DataType::kDataTypeFp32, total_input, false, nullptr,
                            const_cast<float*>(input.ptr<float>()));
  input_flat.set_device_type(device_type_);

  tensor::Tensor output_flat(base::DataType::kDataTypeFp32, total_output, false, nullptr,
                             output.ptr<float>());
  output_flat.set_device_type(device_type_);

  if (is_quant_layer_) {
    kernel::get_matmul_kernel_quant8(device_type_)(input_flat, weight, output_flat,
                                                   group_size_, scales_,
                                                   cuda_config_ ? cuda_config_.get() : nullptr);
  } else {
    kernel::get_matmul_kernel(device_type_)(input_flat, weight, output_flat, 1.f,
                                            cuda_config_ ? cuda_config_.get() : nullptr);
  }

  if (has_bias_) {
    if (batch > 1) {
      for (int b = 0; b < batch; ++b) {
        tensor::Tensor output_view(base::DataType::kDataTypeFp32, K, false, nullptr,
                                   output.ptr<float>(b * K));
        output_view.set_device_type(device_type_);
        kernel::get_add_kernel(device_type_)(output_view, get_bias(0), output_view,
                                             cuda_config_ ? cuda_config_->stream : nullptr);
      }
    } else {
      kernel::get_add_kernel(device_type_)(output_flat, get_bias(0), output_flat,
                                           cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }

  return base::error::Success();
}

base::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr,
                                   base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  CHECK_NE(bias_ptr, nullptr);

  size_t size = dim * sizeof(float);
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  if (!is_quant_layer_) {
    tensor::Tensor bias(base::DataType::kDataTypeFp32, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    bias_.at(idx) = bias;
  } else {
    // is quant layer
    tensor::Tensor bias(base::DataType::kDataTypeInt8, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    bias_.at(idx) = bias;

    const int32_t bias_size = static_cast<int32_t>(bias.size());
    CHECK(bias_size % group_size_ == 0);

    int32_t scale_nums = bias_size / group_size_;
    scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size)};
    scales_.set_device_type(device_type);
  }

  return base::error::Success();
}

tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  return bias_.at(idx);
}

const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  return bias_.at(idx);
}

void MatmulLayer::to_cuda() {
  LayerParam::to_cuda();
  if (has_bias_) {
    for (auto& bias : bias_) {
      bias.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

}  // namespace op