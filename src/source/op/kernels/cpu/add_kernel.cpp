#include "add_kernel.h"
#include <armadillo>
#include "base/base.h"
namespace kernel {
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  CHECK_EQ(input1.size(), input2.size());
  CHECK_EQ(input1.size(), output.size());

  arma::fvec input_vec1(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
  arma::fvec input_vec2(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
  arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);
  output_vec = input_vec1 + input_vec2;
}

void add_bias_kernel_cpu(const tensor::Tensor& output, const tensor::Tensor& bias, int32_t rows,
                         int32_t cols, void* stream) {
  UNUSED(stream);
  CHECK_EQ(output.is_empty(), false);
  CHECK_EQ(bias.is_empty(), false);
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  CHECK_EQ(bias.size(), static_cast<int64_t>(cols));
  CHECK_EQ(output.size(), static_cast<int64_t>(rows) * cols);
  float* out = const_cast<float*>(output.ptr<float>());
  const float* b = bias.ptr<float>();
  for (int32_t r = 0; r < rows; ++r) {
    float* row = out + static_cast<int64_t>(r) * cols;
    for (int32_t c = 0; c < cols; ++c) {
      row[c] += b[c];
    }
  }
}

}  // namespace kernel
