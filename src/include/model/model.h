#ifndef SRC_INCLUDE_MODEL_MODEL_H_
#define SRC_INCLUDE_MODEL_MODEL_H_
#include <op/embedding.h>
#include <cuda_runtime_api.h>
#include <map>
#include <memory>
#include <string>
#include "config.h"
#include "op/encode.h"
#include "op/layer.h"
#include "raw_model_data.h"
#include "sampler/argmax_sampler.h"
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"

namespace model {

// Persistent per-batch-size scratch buffers for the batched forward paths.
// Pointers stay stable across steps, which makes them CUDA-Graph-capturable
// (a graph replays against the same memory every launch) and also removes
// per-step allocator churn for the decode path (red line: pre-allocate and
// reuse all temporary tensors instead of allocating per iteration).
struct BatchScratch {
  int32_t batch = 0;

  // Device buffers (CUDA) or CPU buffers (CPU models), sized [batch, ...].
  tensor::Tensor hidden, rms_out, q_batch, key_batch, val_batch, mha_out_batch, attn_out,
      ffn_norm_out, w1_out, w3_out, w2_out;
  // Flash-Decoding partials (CUDA only):
  //   [batch * head_num * num_splits * (head_size + 2)]
  tensor::Tensor partial_batch;
  // Device copies of the per-token inputs (CUDA graph path reads these).
  tensor::Tensor tokens_cu, positions_cu, kv_offsets_cu;

  // Host staging at stable addresses (CUDA graph path writes new values here
  // before each launch; the H2D uploads happen outside the captured graph).
  tensor::Tensor input_ids, positions, kv_offsets, input_token_num;

  // Allocate (or keep, when sizes already match) every buffer above.
  void ensure(int32_t batch, int32_t hidden_dim, int32_t dim, int32_t kv_dim, int32_t ffn_dim,
              int32_t head_num, int32_t head_size, int32_t max_seq_len,
              base::DeviceType device, const std::shared_ptr<base::DeviceAllocator>& alloc);
};

// One entry of the decode CUDA-Graph pool (indexed by batch size).
struct CudaGraphDecodeEntry {
  int32_t batch = 0;
  cudaGraphExec_t exec = nullptr;  // instantiated graph; nullptr until captured
  bool capture_failed = false;     // fall back to the direct path forever
  int64_t last_used = 0;
  std::unique_ptr<BatchScratch> scratch;
  tensor::Tensor logits_view;  // [batch, vocab] view into the scheduler's logits buffer

  ~CudaGraphDecodeEntry() {
    if (exec) {
      cudaGraphExecDestroy(exec);
    }
  }
};

class Model {
 public:
  explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                 std::string token_path, std::string model_path, bool is_quant_model);

  virtual base::Status init(base::DeviceType device_type) = 0;

  virtual base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               bool is_prompt, int& next) const = 0;

  virtual base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               int& next) const = 0;

  // ========== Continuous Batching Interface ==========
  // Batched forward: each row is one token with its own position and KV slot,
  // so the same call serves both decode (1 token per sequence) and chunked
  // prefill (many prompt tokens per sequence, need_logits=false skips the
  // final RMSNorm + LM head).
  // key_cache/value_cache use the head-dim-contiguous external layout
  // [num_layers, num_slots, kv_dim, max_seq_len].
  // When scratch != nullptr the call reuses its persistent buffers (stable
  // addresses — CUDA-Graph capturable) and reads the staged device inputs
  // (input_ids/positions/kv_offsets must already be CUDA tensors).
  virtual base::Status forward_batch(
      const tensor::Tensor& input_ids,
      const tensor::Tensor& positions,
      const tensor::Tensor& kv_offsets,
      tensor::Tensor& key_cache,
      tensor::Tensor& value_cache,
      tensor::Tensor& logits,
      bool need_logits = true,
      BatchScratch* scratch = nullptr) const = 0;

  // Decode step with a CUDA-Graph pool: the first call for a given batch size
  // captures the decode kernels into a graph; subsequent calls only stage the
  // new inputs (H2D) and replay the graph, cutting the per-step kernel-launch
  // overhead from ~50us to ~5us. Falls back to forward_batch when graphs are
  // unsupported/disabled. Prefill keeps using forward_batch directly (its
  // chunk sizes vary per step).
  virtual base::Status decode_step(const tensor::Tensor& input_ids,
                                   const tensor::Tensor& positions,
                                   const tensor::Tensor& kv_offsets,
                                   tensor::Tensor& key_cache,
                                   tensor::Tensor& value_cache,
                                   tensor::Tensor& logits);

  base::ModelType model_type() const;

  base::DeviceType device_type() const { return device_type_; }

  int32_t layer_num() const { return config_ ? config_->layer_num_ : 0; }
  int32_t kv_dim() const { return config_ ? config_->kv_dim_ : 0; }
  int32_t hidden_dim() const { return config_ ? config_->dim_ : 0; }
  int32_t vocab_size() const { return config_ ? config_->vocab_size_ : 0; }
  int32_t seq_len() const { return config_ ? config_->seq_len_ : 0; }

  const std::string& token_path() const;

  const std::string& model_path() const;

  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;

  virtual bool is_sentence_ending(int32_t token_idx) const;

  virtual std::string decode(int32_t token_idx) const;

  virtual std::string decode(std::vector<int32_t> token_idxs) const;

  /////////////////////////////////////////////////////
  /////////////////////////////////////////////////////
  virtual std::vector<int32_t> encode(const std::string& sentence) const;

  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
                                                                   int32_t token_pos) const;

  virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) const = 0;

  virtual tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
                                    const op::EmbeddingOutput& embedding_output,
                                    bool is_prompt) const;

  // Batch post-processing: argmax sample next token per sequence
  virtual std::vector<int32_t> post_processing_batch(const tensor::Tensor& logits) const;

  // Free large internal KV cache and re-allocate at a smaller size.
  // The internal cache is only used during single-sequence prefill; its
  // sequence-length dimension can be reduced to the longest prompt the
  // caller expects to process, reclaiming hundreds of MB on GPU.
  void resize_internal_kv_cache(int32_t max_seq_len);

  // Point the internal KV buffers at one slot of an external KV cache
  // (e.g. KVManager) so prefill writes land directly in the external cache
  // and no internal->external copy is needed.
  base::Status bind_external_kv_cache(const tensor::Tensor& key_cache,
                                      const tensor::Tensor& value_cache,
                                      int32_t slot_id);

  // Restore the model's own internal KV buffers (no-op if not bound).
  void unbind_external_kv_cache();

  // Synchronize the model's CUDA stream (no-op on CPU or when no stream is set).
  virtual void sync_stream() const {}

 protected:
  virtual base::Status insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor);

  virtual base::Status read_model_file();

  virtual base::Status create_encode_layer();

  virtual base::Status gen_model_from_file();

  virtual base::Status generate_model_infos(const ModelConfig& config) const;

  virtual int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const = 0;

 private:
  virtual void init_mem() = 0;

  virtual base::Status create_layers() = 0;

  virtual void create_param_layers() = 0;

  virtual void create_nonparam_layers() = 0;

  virtual void create_param_quant_layers() = 0;

 protected:
  int32_t group_size_ = 1;
  bool is_quant_model_ = false;
  std::unique_ptr<TransformerConfig> config_;

  std::string token_path_;
  std::string model_path_;
  std::unique_ptr<op::EncodeLayerBase> encode_layer_;
  std::map<ModelBufferType, tensor::Tensor> buffers_;
  bool kv_bound_ = false;
  tensor::Tensor kv_key_backup_;
  tensor::Tensor kv_value_backup_;
  std::unique_ptr<sampler::Sampler> sampler_;
  std::shared_ptr<RawModelData> raw_model_data_;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
  base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;

  // CUDA stream + cuBLAS handle, owned by the model (created in the derived
  // init() for CUDA devices). Lives in the base so decode_step / graph
  // capture can use it; derived classes share it through inheritance.
  std::shared_ptr<kernel::CudaConfig> cuda_config_;

  // CUDA-Graph pool for the decode path, keyed by batch size (see
  // decode_step). Mutable so const forward paths can manage it.
  mutable std::map<int32_t, std::unique_ptr<CudaGraphDecodeEntry>> decode_graph_pool_;
  mutable int64_t decode_graph_clock_ = 0;  // LRU stamp for pool eviction
  bool use_cuda_graphs_ = true;
};
}  // namespace model
#endif
