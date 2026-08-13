#include "model/model.h"
#include <base/alloc.h>
#include <glog/logging.h>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
namespace model {
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

base::Status Model::read_model_file() {
  using namespace base;
  if (model_path_.empty()) {
    return error::PathNotValid("Failed to open the weight file, the model path is empty!");
  }
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }

  auto config = ModelConfig{};
  if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    return error::ModelParseError(
        "Failed to retrieve the configuration information from the model "
        "file.");
  }
  if (is_quant_model_) {
    if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
      return error::ModelParseError(
          "Failed to retrieve the group size information from the model "
          "file.");
    }
  }

  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    return gen_status;
  }

  if (!is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  } else {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return error::ModelParseError(
        "Failed to retrieve the file size information from the model "
        "file.");
  }
  raw_model_data_->file_size = sb.st_size;

  raw_model_data_->fd = fd;
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory.");
  }
  if (!is_quant_model_) {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
  } else {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig) + sizeof(group_size_);
  }
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                  " into memory, the pointer to weight start address is null");
  }
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;
  config_->immediate_dim_ = config.immediate_dim_;
  // Set RoPE theta based on model type
  config_->rope_theta_ = (model_type_ == base::ModelType::kModelTypeLLama) ? 500000.0f : 1000000.0f;
  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  // Qwen tokenizer size and embedding size is mismatched
  // refer: https://github.com/QwenLM/Qwen2.5/issues/29
  // if (std::abs(config.vocab_size) != config_->vocab_size_) {
  //   return base::error::ModelParseError(
  //       "Vocabulary size mismatch between the model file and the token list.");
  // }
  config_->vocab_size_ = std::abs(config.vocab_size);
  return base::error::Success();
}

base::Status Model::create_encode_layer() {
  using namespace base;

  // create token encode decode layer
  // Use runtime model_type_ instead of compile-time #ifdef to avoid
  // incorrect encode layer when multiple model support flags are enabled.
  if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
  } else if (model_type_ == ModelType::kModelTypeLLama) {
    encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
  } else if (model_type_ == ModelType::kModelTypeQwen2 ||
             model_type_ == ModelType::kModelTypeQwen3) {
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
  } else {
    return error::InternalError("Unsupported model type for tokenizer creation.");
  }
  if (!encode_layer_) {
    return error::InternalError("Create the encode layer failed.");
  }

  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  // init sentence piece processor
  // google sentence piece
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }
  // Capture the tokenizer-derived vocab size before read_model_file()
  // overwrites config_->vocab_size_ with the .bin header value.
  const int32_t tokenizer_vocab_size = encode_layer_->vocab_size();
  // mmap
  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
    return mmap_status;
  }
  // Qwen3-style headers store vocab_size=0 (no layout info); restore the
  // tokenizer value or every vocab-sized tensor is allocated with 0 rows.
  // Headers with vocab_size>0 (Qwen2/Llama3) define the on-disk embedding
  // table size that all weight offsets depend on, so keep the header value
  // there and only warn on mismatch.
  if (config_->vocab_size_ <= 0) {
    config_->vocab_size_ = tokenizer_vocab_size;
  } else if (config_->vocab_size_ != tokenizer_vocab_size) {
    LOG(WARNING) << "[MODEL] header vocab_size=" << config_->vocab_size_
                 << " differs from tokenizer vocab_size=" << tokenizer_vocab_size
                 << "; keeping header value for weight layout";
  }
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
    return layer_create_status;
  }

  return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                                int32_t token_pos) const {
  // Use the tensor's actual sequence dimension (dim 1) for stride, not config_->seq_len_,
  // so the internal KV cache can be resized independently of the model's max context length.
  int32_t cache_seq_len = get_buffer(ModelBufferType::kKeyCache).get_dim(1);
  int32_t layer_offset = layer_idx * cache_seq_len * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}

base::Status Model::bind_external_kv_cache(const tensor::Tensor& key_cache,
                                           const tensor::Tensor& value_cache,
                                           int32_t slot_id) {
  using namespace base;
  if (kv_bound_) {
    unbind_external_kv_cache();
  }
  const int32_t num_layers = config_->layer_num_;
  const int32_t num_slots = key_cache.get_dim(1);
  const int32_t max_seq_len = key_cache.get_dim(2);
  const int32_t kv_dim = key_cache.get_dim(3);
  if (num_layers != key_cache.get_dim(0) || kv_dim != config_->kv_dim_ ||
      slot_id < 0 || slot_id >= num_slots ||
      value_cache.get_dim(0) != num_layers || value_cache.get_dim(1) != num_slots ||
      value_cache.get_dim(2) != max_seq_len || value_cache.get_dim(3) != kv_dim) {
    return error::InvalidArgument("External KV cache shape or slot id out of range");
  }

  // View shaped [num_layers, num_slots*max_seq_len, kv_dim] whose base pointer
  // is the slot's first position. slice_kv_cache() and MHA derive their
  // per-layer stride from get_dim(1), which then matches the external cache
  // layout [layers, slots, seq_len, kv_dim]; within one prefill only rows
  // [0, pos) of each layer are ever touched.
  const int32_t slot_stride = max_seq_len * kv_dim;
  float* key_base = const_cast<float*>(key_cache.ptr<float>(slot_id * slot_stride));
  float* val_base = const_cast<float*>(value_cache.ptr<float>(slot_id * slot_stride));

  tensor::Tensor key_view(DataType::kDataTypeFp32, num_layers, num_slots * max_seq_len,
                          kv_dim, false, nullptr, key_base);
  tensor::Tensor val_view(DataType::kDataTypeFp32, num_layers, num_slots * max_seq_len,
                          kv_dim, false, nullptr, val_base);
  key_view.set_device_type(device_type_);
  val_view.set_device_type(device_type_);

  // Prefill MHA now uses a per-head score stride of num_slots*max_seq_len;
  // grow the shared score buffer if it no longer fits.
  int64_t needed = static_cast<int64_t>(config_->head_num_) * num_slots * max_seq_len;
  tensor::Tensor& score = buffers_.at(ModelBufferType::kScoreStorage);
  if (static_cast<int64_t>(score.size()) < needed) {
    score.reshape({config_->head_num_, static_cast<int32_t>(num_slots * max_seq_len)});
  }

  kv_key_backup_ = std::move(buffers_.at(ModelBufferType::kKeyCache));
  kv_value_backup_ = std::move(buffers_.at(ModelBufferType::kValueCache));
  buffers_.at(ModelBufferType::kKeyCache) = key_view;
  buffers_.at(ModelBufferType::kValueCache) = val_view;
  kv_bound_ = true;
  return error::Success();
}

void Model::unbind_external_kv_cache() {
  if (!kv_bound_) {
    return;
  }
  buffers_.at(ModelBufferType::kKeyCache) = std::move(kv_key_backup_);
  buffers_.at(ModelBufferType::kValueCache) = std::move(kv_value_backup_);
  kv_bound_ = false;
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }

  // Qwen3 uses hidden_dim_ as the model dimension (≠ dim_ for attention),
  // while Qwen2 and Llama use dim_ (== hidden_dim_).
  // Use runtime model_type_ instead of compile-time #ifdef.
  int32_t model_dim = (model_type_ == base::ModelType::kModelTypeQwen3)
                          ? config_->hidden_dim_
                          : config_->dim_;

  std::shared_ptr<base::Buffer> input_emb_buffer =
      std::make_shared<base::Buffer>(model_dim * sizeof(float), nullptr,
                                     input_embeddings.ptr<float>(index * model_dim), true);
  tensor::Tensor input(base::DataType::kDataTypeFp32, model_dim);
  input.assign(input_emb_buffer);
  input.set_device_type(device_type_);
  return input;
}

std::vector<int32_t> Model::post_processing_batch(const tensor::Tensor& logits) const {
  int32_t batch = logits.get_dim(0);
  int32_t vocab_size = logits.get_dim(1);
  std::vector<int32_t> tokens(batch);
  // Restrict sampling to the tokenizer's vocab range: the .bin header vocab
  // can be larger than the tokenizer's (Qwen3: 151936 vs 151665), and logits
  // beyond the tokenizer vocab correspond to reserved rows that cannot be
  // decoded. One batched kernel + one device-to-host copy (was: one kernel,
  // one copy and one stream sync per row).
  int32_t sample_vocab = vocab_size;
  if (encode_layer_ && encode_layer_->vocab_size() > 0) {
    sample_vocab = std::min(sample_vocab, encode_layer_->vocab_size());
  }
  sampler_->sample_batch(logits.ptr<float>(), vocab_size, sample_vocab, batch, tokens.data(),
                         nullptr);
  return tokens;
}

void Model::resize_internal_kv_cache(int32_t max_seq_len) {
  if (max_seq_len <= 0) return;
  // Drop any stale external-slot views before touching the real internal cache.
  unbind_external_kv_cache();

  auto alloc = (device_type_ == base::DeviceType::kDeviceCUDA)
      ? std::shared_ptr<base::DeviceAllocator>(base::CUDADeviceAllocatorFactory::get_instance())
      : base::CPUDeviceAllocatorFactory::get_instance();

  auto it_key = buffers_.find(ModelBufferType::kKeyCache);
  auto it_val = buffers_.find(ModelBufferType::kValueCache);

  if (it_key != buffers_.end() && it_val != buffers_.end()) {
    size_t old_bytes = it_key->second.byte_size() + it_val->second.byte_size();

    int32_t num_layers = it_key->second.get_dim(0);
    int32_t kv_dim = it_key->second.get_dim(2);

    it_key->second = tensor::Tensor(base::DataType::kDataTypeFp32,
                                     num_layers, max_seq_len, kv_dim, true, alloc);
    it_val->second = tensor::Tensor(base::DataType::kDataTypeFp32,
                                     num_layers, max_seq_len, kv_dim, true, alloc);

    size_t new_bytes = it_key->second.byte_size() + it_val->second.byte_size();
    LOG(INFO) << "[MODEL] Internal KV cache resized: "
              << (old_bytes >> 20) << "MB -> " << (new_bytes >> 20) << "MB"
              << " (seq_len " << max_seq_len << ")";
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    base::CUDADeviceAllocatorFactory::get_instance()->free_idle();
  }
}

}  // namespace model