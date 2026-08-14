#ifndef SRC_SOURCE_OP_KERNELS_CPU_MHA_KERNEL_H
#define SRC_SOURCE_OP_KERNELS_CPU_MHA_KERNEL_H
#include <base/cuda_config.h>
#include "base/base.h"
#include "tensor/tensor.h"
namespace kernel {
void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_head_num, int32_t head_size, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type, CudaConfig* config);

// Batched CPU MHA (OpenMP over (batch, head)) for the external head-dim
// contiguous cache [num_layers, num_slots, kv_dim, max_seq_len]:
//   positions / kv_offsets   [batch] host int32
//   query_batch              [batch, dim]
//   mha_out_batch            [batch, dim]
void mha_kernel_cpu_batch(int32_t head_num, int32_t layer_idx, int32_t num_slots,
                          int32_t max_seq_len, int32_t kv_dim, int32_t kv_head_num,
                          int32_t head_size, const tensor::Tensor& positions,
                          const tensor::Tensor& kv_offsets, const tensor::Tensor& query_batch,
                          tensor::Tensor& mha_out_batch, const tensor::Tensor& key_cache,
                          const tensor::Tensor& value_cache);
}  // namespace kernel
#endif 
