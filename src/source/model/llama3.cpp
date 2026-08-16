#include "model/llama3.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <sentencepiece_processor.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include "../op/kernels/cpu/mha_kernel.h"
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/mha_kernel.cuh"
#include "../op/kernels/cuda/paged_kernels.cuh"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "base/tick.h"
namespace model {

void LLamaLayers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  if (add_layer_) {
    add_layer_->set_cuda_config(config);
    add_layer_->to_cuda();
  }

  if (rope_layer_) {
    rope_layer_->set_cuda_config(config);
    rope_layer_->to_cuda();
  }

  if (swiglu_layer_) {
    swiglu_layer_->set_cuda_config(config);
    swiglu_layer_->to_cuda();
  }

  if (cls_layer_) {
    cls_layer_->set_cuda_config(config);
    cls_layer_->to_cuda();
  }

  if (embedding_layer_) {
    embedding_layer_->set_cuda_config(config);
    embedding_layer_->to_cuda();
  }

  if (mha_layer_) {
    mha_layer_->set_cuda_config(config);
    mha_layer_->to_cuda();
  }

  for (auto& weight_layer : wq_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wk_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wv_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wo_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w1_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w2_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w3_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    if (rms_norm_layer) {
      rms_norm_layer->to_cuda();
      rms_norm_layer->set_cuda_config(config);
    }
  }
}

LLamaModel::LLamaModel(base::TokenizerType tokenizer_type, std::string token_path,
                         std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama, std::move(token_path),
            std::move(model_path), is_quant_model) {}

base::Status LLamaModel::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return error::InternalError("The cpu device do not support int8 quant model.");
  }

  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return error::InternalError("The cuda hanle create failed.");
    }
  }

  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }
  init_mem();
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>(),
                                   config_->rope_theta_);
  } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), cuda_config_->stream,
                                  config_->rope_theta_);
  }

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

base::Status LLamaModel::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  }

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    // attention (wq wk wv @ input)
    attention_qkv(layer_idx, pos_tensor);
    // multi-head attention
    attention_mha(layer_idx, pos_tensor);
    // feed forward
    feed_forward(layer_idx, input);
  }
  // Logits are only needed for the token that feeds sampling (the last
  // prompt token); predict() calls cls_logits() explicitly.
  return base::error::Success();
}

void LLamaModel::create_nonparam_layers() {
  CHECK(llama_layers_ != nullptr);
  llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

  llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_head_num_, config_->kv_dim_, config_->seq_len_,
      config_->head_num_, config_->head_size_);

  llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  llama_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
}

void LLamaModel::create_param_quant_layers() {
  CHECK(is_quant_model_);
  CHECK(llama_layers_ != nullptr);

  size_t pos = 0;
  int32_t dim = config_->dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wq->set_group_size(group_size_);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wk->set_group_size(group_size_);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wv->set_group_size(group_size_);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wo->set_group_size(group_size_);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
  }

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w1->set_group_size(group_size_);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
    w2->set_group_size(group_size_);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w3->set_group_size(group_size_);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
  }

  // wcls layer
  auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
  cls_layer->set_group_size(group_size_);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
  } else {
    // no shared
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
    pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
  }
  llama_layers_->cls_layer_ = cls_layer;

  // embedding layer
  float* weight_ptr = (float*)raw_model_data_->weight(pos);
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, weight_ptr,
                                              cpu_device_type);
  weight_ptr += config_->vocab_size_ * dim;

  // rmsnorm attention attention,ffn,final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim);

    rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += dim;
  }
}

void LLamaModel::create_param_layers() {
  CHECK(!is_quant_model_);
  CHECK(llama_layers_ != nullptr);
  // The embedding layer
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));

  const void* weight_embedding = raw_model_data_->weight(0);
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), config_->dim_},
                                              weight_embedding, cpu_device_type);

  // create all matmul layer
  int32_t dim = config_->dim_;
  size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;
  // create weight matrix for query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos += dim * dim;
  }

  // create weight matrix for key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // skip ffn rmsnorm
  pos += config_->layer_num_ * dim;

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos += dim * hidden_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos += dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos += dim * hidden_dim;
  }

  // skip final rms weight
  pos += dim;
  // skip freqs_cos and freqs_sin weight
  pos += config_->seq_len_ * config_->head_size_;

  llama_layers_->cls_layer_ =
      std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          this->raw_model_data_->weight(0), cpu_device_type);
  } else {
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          this->raw_model_data_->weight(pos), cpu_device_type);
  }

  // create rmsnorm layer
  size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    rmsnorm_pos += config_->dim_;
  }

  // skip attention.wq attention.wk attention.wv attention.wo
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
  rmsnorm_pos +=
      config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos +=
      config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += config_->dim_;
  }

  // skip ffn.w1 ffn.w2 ffn.w3
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

  std::shared_ptr<op::RmsNormLayer> rms_final_layer =
      std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

  const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
  rms_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
  llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}

void LLamaModel::build_fused_qkv_layers() {
  // Escape hatch: fall back to the three-GEMM path (numerics A/B).
  const char* disable = std::getenv("LLAMA_DISABLE_FUSED_QKV");
  if (disable && std::string(disable) == "1") {
    LOG(INFO) << "[MODEL] Fused QKV disabled via LLAMA_DISABLE_FUSED_QKV";
    return;
  }
  if (is_quant_model_) {
    LOG(INFO) << "[MODEL] Fused QKV skipped for int8 quant model";
    return;
  }

  const int32_t dim = config_->dim_;
  const int32_t kv_dim = config_->kv_dim_;
  const int32_t fused_dim = dim + 2 * kv_dim;
  const int32_t num_layers = config_->layer_num_;
  const size_t w_block_q = static_cast<size_t>(dim) * dim * 4;
  const size_t w_block_kv = static_cast<size_t>(kv_dim) * dim * 4;

  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  }

  llama_layers_->fused_qkv_layers_.clear();
  llama_layers_->fused_qkv_weight_src_.clear();
  llama_layers_->fused_qkv_bias_src_.clear();
  for (int32_t i = 0; i < num_layers; ++i) {
    const tensor::Tensor& wq_w =
        std::dynamic_pointer_cast<op::MatmulLayer>(llama_layers_->wq_layers_.at(i))->get_weight(0);
    const tensor::Tensor& wk_w =
        std::dynamic_pointer_cast<op::MatmulLayer>(llama_layers_->wk_layers_.at(i))->get_weight(0);
    const tensor::Tensor& wv_w =
        std::dynamic_pointer_cast<op::MatmulLayer>(llama_layers_->wv_layers_.at(i))->get_weight(0);

    // Stacked weight [q | k | v]: K-major blocks, one contiguous memcpy each.
    tensor::Tensor fused_w(base::DataType::kDataTypeFp32, fused_dim, dim, true, alloc);
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
      cudaStream_t stream = cuda_config_->stream;
      cudaMemcpyAsync(fused_w.ptr<float>(), wq_w.ptr<float>(), w_block_q,
                      cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(fused_w.ptr<float>(static_cast<int64_t>(dim) * dim), wk_w.ptr<float>(),
                      w_block_kv, cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(fused_w.ptr<float>(static_cast<int64_t>(dim + kv_dim) * dim),
                      wv_w.ptr<float>(), w_block_kv, cudaMemcpyDeviceToDevice, stream);
      cudaStreamSynchronize(stream);  // init-time only: weights must be ready pre-forward
    } else {
      std::memcpy(fused_w.ptr<float>(), wq_w.ptr<float>(), w_block_q);
      std::memcpy(fused_w.ptr<float>(static_cast<int64_t>(dim) * dim), wk_w.ptr<float>(),
                  w_block_kv);
      std::memcpy(fused_w.ptr<float>(static_cast<int64_t>(dim + kv_dim) * dim),
                  wv_w.ptr<float>(), w_block_kv);
    }

    // LLaMA has no QKV bias (unlike Qwen2).
    auto fused = std::make_shared<op::MatmulLayer>(device_type_, fused_dim, dim, false, false);
    CHECK(fused->set_weight(0, fused_w));
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
      fused->set_cuda_config(cuda_config_);
    }
    llama_layers_->fused_qkv_layers_.push_back(fused);
    llama_layers_->fused_qkv_weight_src_.push_back(fused_w);
  }
  llama_layers_->fused_qkv_enabled_ = true;
  LOG(INFO) << "[MODEL] Fused QKV ready: " << num_layers << " layers of ["
            << fused_dim << ", " << dim << "] (LLAMA_DISABLE_FUSED_QKV=1 to disable)";
}

void LLamaModel::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    llama_layers_->to_cuda(cuda_config_);
  }

  // Fused QKV (M3): the wq/wk/wv weights are on the target device now.
  // Gated on the paged-attention build so the A/B baseline (macro OFF)
  // reproduces the original three-GEMM path exactly.
#ifdef USE_PAGED_ATTENTION
  build_fused_qkv_layers();
#endif

  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);

  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // kv cache
  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // Wq query output
  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Attention output
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

  // final forward output
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }

  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

base::Status LLamaModel::create_layers() {
  using namespace base;
  if (!llama_layers_) {
    llama_layers_ = std::make_unique<LLamaLayers>();
  }

  if (!is_quant_model_) {
    create_param_layers();
  } else {
    create_param_quant_layers();
  }
  create_nonparam_layers();

  if (!llama_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  if (llama_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
    return error::InternalError("Create the rmsnorm layers for the llama model failed!");
  }

  if (llama_layers_->wq_layers_.size() != config_->layer_num_ ||
      llama_layers_->wk_layers_.size() != config_->layer_num_ ||
      llama_layers_->wv_layers_.size() != config_->layer_num_ ||
      llama_layers_->wo_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for "
        "the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->wq_layers_.at(i) || !llama_layers_->wk_layers_.at(i) ||
        !llama_layers_->wv_layers_.at(i) || !llama_layers_->wo_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }
  }

  if (llama_layers_->w1_layers_.size() != config_->layer_num_ ||
      llama_layers_->w2_layers_.size() != config_->layer_num_ ||
      llama_layers_->w3_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->w1_layers_.at(i) || !llama_layers_->w2_layers_.at(i) ||
        !llama_layers_->w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }
  }

  if (!llama_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for the llama model failed!");
  }

  if (!llama_layers_->add_layer_) {
    return error::InternalError("Create the add layer for the llama model failed!");
  }

  if (!llama_layers_->mha_layer_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }

  if (!llama_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  return error::Success();
}

op::EmbeddingOutput LLamaModel::embedding(const std::vector<int>& tokens) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !llama_layers_->embedding_layer_)
      << "The embedding layer in the llama model is null pointer.";
  STATUS_CHECK(
      llama_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}

void LLamaModel::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  // attn rmsnorm
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama model";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void LLamaModel::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);
  // query
  const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // key
  const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
  // value
  const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(llama_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(llama_layers_->rope_layer_->forward(
      query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}

base::Status LLamaModel::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  if (!is_prompt) {
    // Only the token that feeds sampling needs logits.
    cls_logits(input);
    next = post_processing(pos_tensor, is_prompt);
  } else {
    next = -1;
  }
  return base::error::Success();
}

void LLamaModel::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // mha
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  // VAL = [val1,val2,...val t]
  // output @ VAL = 最终的结果
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

  const auto& mha_layer = llama_layers_->mha_layer_;
  CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
  int pos = pos_tensor.index<int32_t>(0);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void LLamaModel::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      llama_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // SwiGLU
  CHECK_NE(llama_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
}

void LLamaModel::cls_logits(const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(llama_layers_->cls_layer_, nullptr);
  STATUS_CHECK(llama_layers_->cls_layer_->forward(input, forward_output));
}

int32_t LLamaModel::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    // Restrict sampling to the tokenizer's vocab range (see
    // post_processing_batch): header vocab may be larger than the tokenizer's.
    size_t sample_size = static_cast<size_t>(forward_output.size());
    if (encode_layer_ && encode_layer_->vocab_size() > 0) {
      sample_size = std::min(sample_size, static_cast<size_t>(encode_layer_->vocab_size()));
    }
    next = static_cast<int32_t>(sampler_->sample(
        forward_logits, sample_size, cuda_config_ ? cuda_config_->stream : nullptr));
  }
  return next;
}

void LLamaModel::sync_stream() const {
  if (device_type_ == base::DeviceType::kDeviceCUDA && cuda_config_ &&
      cuda_config_->stream) {
    cudaStreamSynchronize(cuda_config_->stream);
  }
}

// ========== Continuous Batching forward ==========
base::Status LLamaModel::forward_batch(
    const tensor::Tensor& input_ids,
    const tensor::Tensor& positions,
    const tensor::Tensor& block_table,
    tensor::Tensor& key_cache,
    tensor::Tensor& value_cache,
    tensor::Tensor& logits,
    bool need_logits,
    BatchScratch* scratch) const {
  if (input_ids.is_empty()) {
    return base::error::InvalidArgument("The input_ids tensor is empty.");
  }

  int32_t batch = input_ids.get_dim(0);
  int32_t hidden_dim = config_->dim_;
  int32_t kv_dim = config_->kv_dim_;
  // KV cache geometry from the tensors themselves: paged
  // [layers, num_blocks, block_size, kv_dim] (CUDA + USE_PAGED_ATTENTION) or
  // head-dim-contiguous [layers, slots, kv_dim, max_seq_len] (CPU / baseline).
  const KVCacheDims cache_dims = resolve_kv_cache_dims(key_cache, block_table, device_type_);
  int32_t num_blocks = cache_dims.num_blocks;
  int32_t block_size = cache_dims.block_size;
  int32_t table_stride = cache_dims.table_stride;
  int32_t num_slots = cache_dims.paged ? 0 : num_blocks;
  int32_t max_seq_len = cache_dims.max_seq_len;
  CHECK_EQ(key_cache.get_dim(cache_dims.paged ? 3 : 2), kv_dim);

  // Allocate intermediate batch tensors
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  // Persistent scratch (CUDA-Graph decode path) or pooled allocation (prefill
  // path). The scratch keeps stable addresses across steps so a captured
  // graph always replays against the same memory.
  tensor::Tensor hidden, rms_out, q_batch, key_batch, val_batch, mha_out_batch, attn_out,
      ffn_norm_out, w1_out, w3_out, w2_out, partial_batch, qkv_out;
  if (scratch) {
    scratch->ensure(batch, hidden_dim, config_->dim_, kv_dim, config_->hidden_dim_,
                    config_->head_num_, config_->head_size_, max_seq_len, table_stride,
                    device_type_, alloc);
    hidden = scratch->hidden;
    rms_out = scratch->rms_out;
    q_batch = scratch->q_batch;
    key_batch = scratch->key_batch;
    val_batch = scratch->val_batch;
    mha_out_batch = scratch->mha_out_batch;
    attn_out = scratch->attn_out;
    ffn_norm_out = scratch->ffn_norm_out;
    w1_out = scratch->w1_out;
    w3_out = scratch->w3_out;
    w2_out = scratch->w2_out;
    partial_batch = scratch->partial_batch;
    qkv_out = scratch->qkv_out;
  } else {
    hidden = tensor::Tensor(base::DataType::kDataTypeFp32, batch, hidden_dim, true, alloc);
    rms_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, hidden_dim, true, alloc);
    q_batch = tensor::Tensor(base::DataType::kDataTypeFp32, batch, config_->dim_, true, alloc);
    mha_out_batch = tensor::Tensor(base::DataType::kDataTypeFp32, batch, config_->dim_, true, alloc);
    attn_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, hidden_dim, true, alloc);
    ffn_norm_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, hidden_dim, true, alloc);
    w1_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, config_->hidden_dim_, true, alloc);
    w3_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, config_->hidden_dim_, true, alloc);
    w2_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch, hidden_dim, true, alloc);
    key_batch = tensor::Tensor(base::DataType::kDataTypeFp32, batch, kv_dim, true, alloc);
    val_batch = tensor::Tensor(base::DataType::kDataTypeFp32, batch, kv_dim, true, alloc);
    qkv_out = tensor::Tensor(base::DataType::kDataTypeFp32, batch,
                             config_->dim_ + 2 * kv_dim, true, alloc);
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
      int32_t num_splits = kernel::flash_decoding_num_splits(max_seq_len);
      partial_batch = tensor::Tensor(
          base::DataType::kDataTypeFp32,
          static_cast<int64_t>(batch) * config_->head_num_ * num_splits *
              (config_->head_size_ + 2),
          true, alloc);
    }
  }

  // 1. Embedding: [batch] -> [batch, hidden_dim]
  {
    const auto& emb_layer = llama_layers_->embedding_layer_;
    tensor::Tensor input_token_num;
    if (scratch) {
      input_token_num = scratch->input_token_num;
    } else {
      input_token_num = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true,
                                       base::CPUDeviceAllocatorFactory::get_instance());
    }
    // Set batch size on embedding layer
    std::dynamic_pointer_cast<op::EmbeddingLayer>(emb_layer)->set_batch_size(batch);
    tensor::Tensor tokens = input_ids;
    if (device_type_ == base::DeviceType::kDeviceCUDA && scratch) {
      // Upload the stable host staging buffer into the device staging tensor.
      // Captured as a graph upload node on the first call, re-executed with
      // the fresh host data on every replay.
      cudaMemcpyAsync(scratch->tokens_cu.ptr<int32_t>(), input_ids.ptr<int32_t>(),
                      batch * sizeof(int32_t), cudaMemcpyHostToDevice, cuda_config_->stream);
      tokens = scratch->tokens_cu;
    }
    STATUS_CHECK(emb_layer->forward(tokens, input_token_num, hidden));
  }

  // 2. Per-layer processing
  // Device copies of positions/block_table for the CUDA kernels. In scratch
  // mode these uploads are captured into the graph too.
  tensor::Tensor positions_cu;
  tensor::Tensor block_table_cu;
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    if (scratch) {
      cudaMemcpyAsync(scratch->positions_cu.ptr<int32_t>(), positions.ptr<int32_t>(),
                      batch * sizeof(int32_t), cudaMemcpyHostToDevice, cuda_config_->stream);
      cudaMemcpyAsync(scratch->block_table_cu.ptr<int32_t>(), block_table.ptr<int32_t>(),
                      static_cast<size_t>(batch) * table_stride * sizeof(int32_t),
                      cudaMemcpyHostToDevice, cuda_config_->stream);
      positions_cu = scratch->positions_cu;
      block_table_cu = scratch->block_table_cu;
    } else {
      positions_cu = tensor::Tensor(base::DataType::kDataTypeInt32, batch, true, alloc);
      block_table_cu = tensor::Tensor(base::DataType::kDataTypeInt32, batch, table_stride,
                                      true, alloc);
      cudaMemcpyAsync(const_cast<int32_t*>(positions_cu.ptr<int32_t>()),
                      positions.ptr<int32_t>(), batch * sizeof(int32_t), cudaMemcpyHostToDevice,
                      cuda_config_->stream);
      cudaMemcpyAsync(const_cast<int32_t*>(block_table_cu.ptr<int32_t>()),
                      block_table.ptr<int32_t>(),
                      static_cast<size_t>(batch) * table_stride * sizeof(int32_t),
                      cudaMemcpyHostToDevice, cuda_config_->stream);
    }
  }

  // Bounds check once for the whole batch (host tensors only — the graph path
  // validates its staged inputs in decode_step).
  if (positions.device_type() == base::DeviceType::kDeviceCPU) {
    for (int32_t b = 0; b < batch; ++b) {
      if (positions.ptr<int32_t>()[b] >= max_seq_len) {
        LOG(ERROR) << "forward_batch: position " << positions.ptr<int32_t>()[b]
                   << " exceeds max_seq_len " << max_seq_len
                   << " for table row " << block_table.ptr<int32_t>()[b];
        return base::error::InvalidArgument(
            "Position exceeds KV cache capacity. Increase max_seq_len.");
      }
    }
  }

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    // a. RMSNorm: [batch, hidden_dim]
    {
      auto& rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx);
      STATUS_CHECK(rmsnorm->forward(hidden, rms_out));
    }

    // b. QKV projection: one fused GEMM (M3) with zero-copy q/k/v views, or
    // the classic three GEMMs (escape hatch LLAMA_DISABLE_FUSED_QKV / quant).
    const bool fused_qkv = llama_layers_->fused_qkv_enabled_;
    if (fused_qkv) {
      auto& fused = llama_layers_->fused_qkv_layers_.at(layer_idx);
      std::dynamic_pointer_cast<op::MatmulLayer>(fused)->set_batch_size(batch);
      STATUS_CHECK(fused->forward(rms_out, qkv_out));
      // Views into the fused output row blocks: [q | k | v].
      q_batch = tensor::Tensor(base::DataType::kDataTypeFp32, batch, config_->dim_,
                               false, nullptr, qkv_out.ptr<float>());
      key_batch = tensor::Tensor(base::DataType::kDataTypeFp32, batch, kv_dim,
                                 false, nullptr, qkv_out.ptr<float>(config_->dim_));
      val_batch = tensor::Tensor(base::DataType::kDataTypeFp32, batch, kv_dim,
                                 false, nullptr, qkv_out.ptr<float>(config_->dim_ + kv_dim));
      q_batch.set_device_type(device_type_);
      key_batch.set_device_type(device_type_);
      val_batch.set_device_type(device_type_);
    } else {
      // b. Q projection: [batch, hidden_dim] x [dim, hidden_dim] -> [batch, dim]
      {
        auto& wq = llama_layers_->wq_layers_.at(layer_idx);
        std::dynamic_pointer_cast<op::MatmulLayer>(wq)->set_batch_size(batch);
        STATUS_CHECK(wq->forward(rms_out, q_batch));
      }
    }

    // c/d/e: K, V projections (skipped under fused QKV) + RoPE + cache write
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
      if (!fused_qkv) {
        // Batched K projection: [batch, hidden_dim] -> [batch, kv_dim]
        {
          auto& wk = llama_layers_->wk_layers_.at(layer_idx);
          std::dynamic_pointer_cast<op::MatmulLayer>(wk)->set_batch_size(batch);
          STATUS_CHECK(wk->forward(rms_out, key_batch));
        }

        // Batched V projection
        {
          auto& wv = llama_layers_->wv_layers_.at(layer_idx);
          std::dynamic_pointer_cast<op::MatmulLayer>(wv)->set_batch_size(batch);
          STATUS_CHECK(wv->forward(rms_out, val_batch));
        }
      }

      // RoPE over the whole batch (single kernel, in place on q_batch/key_batch)
      kernel::rope_kernel_cu_batch(config_->dim_, kv_dim, config_->head_size_, q_batch, key_batch,
                                   positions_cu, get_buffer(ModelBufferType::kSinCache),
                                   get_buffer(ModelBufferType::kCosCache),
                                   cuda_config_->stream);

#ifdef USE_PAGED_ATTENTION
      // Write this layer's K/V rows into each sequence's paged cache blocks
      // (layout [layers, num_blocks, block_size, kv_dim], addressed through
      // the block table).
      kernel::paged_kv_scatter_cu(key_batch, key_cache, block_table_cu, positions_cu, kv_dim,
                                  num_blocks, block_size, layer_idx, cuda_config_->stream);
      kernel::paged_kv_scatter_cu(val_batch, value_cache, block_table_cu, positions_cu, kv_dim,
                                  num_blocks, block_size, layer_idx, cuda_config_->stream);

      // f. MHA: Paged Flash Decoding — split-KV pass + reduce pass, one
      // launch pair per layer for the whole batch.
      kernel::paged_attention_cu_batch(config_->head_num_, layer_idx, num_blocks, block_size,
                                       kv_dim, config_->kv_head_num_, config_->head_size_,
                                       positions_cu, block_table_cu, q_batch, partial_batch,
                                       mha_out_batch, key_cache, value_cache, cuda_config_.get());
#else
      // Write this layer's K/V rows into each sequence's KV cache slot
      // (head-dim-contiguous layout: cache[layer][slot][d][pos]).
      kernel::kv_scatter_cu(key_batch, key_cache, block_table_cu, positions_cu, kv_dim, num_slots,
                            max_seq_len, layer_idx, cuda_config_->stream);
      kernel::kv_scatter_cu(val_batch, value_cache, block_table_cu, positions_cu, kv_dim, num_slots,
                            max_seq_len, layer_idx, cuda_config_->stream);

      // f. MHA: Flash Decoding — split-KV pass + reduce pass, one launch pair
      // per layer for the whole batch.
      kernel::mha_kernel_cu_batch(config_->head_num_, layer_idx, num_slots, max_seq_len, kv_dim,
                                  config_->kv_head_num_, config_->head_size_, positions_cu,
                                  block_table_cu, q_batch, partial_batch, mha_out_batch, key_cache,
                                  value_cache, cuda_config_.get());
#endif
    } else {
      // CPU path: batched K/V GEMM (one GEMM per weight matrix, NOT a
      // per-sequence loop), batched RoPE, then scatter into the
      // head-dim-contiguous cache and a batched OpenMP attention pass.
      if (!fused_qkv) {
        {
          auto& wk = llama_layers_->wk_layers_.at(layer_idx);
          std::dynamic_pointer_cast<op::MatmulLayer>(wk)->set_batch_size(batch);
          STATUS_CHECK(wk->forward(rms_out, key_batch));
        }
        {
          auto& wv = llama_layers_->wv_layers_.at(layer_idx);
          std::dynamic_pointer_cast<op::MatmulLayer>(wv)->set_batch_size(batch);
          STATUS_CHECK(wv->forward(rms_out, val_batch));
        }
      }

      kernel::rope_kernel_cpu_batch(config_->dim_, kv_dim, config_->head_size_, q_batch,
                                    key_batch, positions, get_buffer(ModelBufferType::kSinCache),
                                    get_buffer(ModelBufferType::kCosCache), nullptr);

      // CPU kernels expect [batch] slot ids; the paged block table stores
      // them in column 0 (slot-mode pool on CPU devices).
      const tensor::Tensor& cpu_offsets = cpu_slot_offsets(block_table, batch);

      // Scatter K/V rows into the external cache:
      //   cache[layer][slot][d][pos] = key_batch[b][d]
      {
        const float* k_ptr = key_batch.ptr<float>();
        const float* v_ptr = val_batch.ptr<float>();
        float* kcache = const_cast<float*>(key_cache.ptr<float>());
        float* vcache = const_cast<float*>(value_cache.ptr<float>());
        const int64_t layer_base = static_cast<int64_t>(layer_idx) * num_slots * kv_dim * max_seq_len;
#pragma omp parallel for schedule(static)
        for (int32_t b = 0; b < batch; ++b) {
          const int32_t pos = positions.ptr<int32_t>()[b];
          const int32_t slot = cpu_offsets.ptr<int32_t>()[b];
          const int64_t slot_base = layer_base + static_cast<int64_t>(slot) * kv_dim * max_seq_len;
          for (int32_t d = 0; d < kv_dim; ++d) {
            const int64_t dst = slot_base + static_cast<int64_t>(d) * max_seq_len + pos;
            kcache[dst] = k_ptr[static_cast<int64_t>(b) * kv_dim + d];
            vcache[dst] = v_ptr[static_cast<int64_t>(b) * kv_dim + d];
          }
        }
      }

      // f. MHA: batched OpenMP attention (one task per (batch, head)).
      kernel::mha_kernel_cpu_batch(config_->head_num_, layer_idx, num_slots, max_seq_len, kv_dim,
                                   config_->kv_head_num_, config_->head_size_, positions,
                                   cpu_offsets, q_batch, mha_out_batch, key_cache, value_cache);
    }

    // g. WO projection: [batch, dim] -> [batch, hidden_dim]
    {
      auto& wo = llama_layers_->wo_layers_.at(layer_idx);
      std::dynamic_pointer_cast<op::MatmulLayer>(wo)->set_batch_size(batch);
      STATUS_CHECK(wo->forward(mha_out_batch, attn_out));
    }

    // h. Residual add: hidden = hidden + attn_out (element-wise)
    {
      auto& add_layer = llama_layers_->add_layer_;
      STATUS_CHECK(add_layer->forward(hidden, attn_out, hidden));
    }

    // i. FFN: RMSNorm -> W1/W3 -> SwiGLU -> W2 -> residual add
    {
      // FFN RMSNorm
      auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
      STATUS_CHECK(ffn_rmsnorm->forward(hidden, ffn_norm_out));

      // W1: [batch, hidden_dim] -> [batch, hidden_dim_ffn]
      {
        auto& w1 = llama_layers_->w1_layers_.at(layer_idx);
        std::dynamic_pointer_cast<op::MatmulLayer>(w1)->set_batch_size(batch);
        STATUS_CHECK(w1->forward(ffn_norm_out, w1_out));
      }

      // W3: [batch, hidden_dim] -> [batch, hidden_dim_ffn]
      {
        auto& w3 = llama_layers_->w3_layers_.at(layer_idx);
        std::dynamic_pointer_cast<op::MatmulLayer>(w3)->set_batch_size(batch);
        STATUS_CHECK(w3->forward(ffn_norm_out, w3_out));
      }

      // SwiGLU: w1_out = w1_out * sigmoid(w1_out) * w3_out (element-wise via kernel)
      STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_out, w3_out, w1_out));

      // W2: [batch, hidden_dim_ffn] -> [batch, hidden_dim]
      {
        auto& w2 = llama_layers_->w2_layers_.at(layer_idx);
        std::dynamic_pointer_cast<op::MatmulLayer>(w2)->set_batch_size(batch);
        STATUS_CHECK(w2->forward(w1_out, w2_out));
      }

      // Residual add
      STATUS_CHECK(llama_layers_->add_layer_->forward(hidden, w2_out, hidden));
    }
  }

  // 3. Final RMSNorm + LM Head (skipped for prefill chunks: only the first
  // generated token, produced by a decode step, needs logits)
  if (need_logits) {
    auto& final_norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
    STATUS_CHECK(final_norm->forward(hidden, hidden));

    // LM Head: [batch, hidden_dim] -> [batch, vocab_size]
    logits.reshape({batch, config_->vocab_size_});
    auto& cls = llama_layers_->cls_layer_;
    std::dynamic_pointer_cast<op::MatmulLayer>(cls)->set_batch_size(batch);
    STATUS_CHECK(cls->forward(hidden, logits));
  }

  // No stream sync here: the caller (Scheduler) syncs via sync_stream()
  // right before post_processing_batch reads the logits.
  return base::error::Success();
}

}  // namespace model