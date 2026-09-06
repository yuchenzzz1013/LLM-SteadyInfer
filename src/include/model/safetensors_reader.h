#ifndef SRC_INCLUDE_MODEL_SAFETENSORS_READER_H_
#define SRC_INCLUDE_MODEL_SAFETENSORS_READER_H_
// Reader for Hugging Face safetensors model directories.
//
// Layout expected (as written by HF transformers):
//   <model_dir>/config.json
//   <model_dir>/model.safetensors               (single shard, no index file)
//   <model_dir>/model-00001-of-000NN.safetensors ... + model.safetensors.index.json
//
// Every safetensors file is:  <uint64 LE header_len> <JSON header> <raw data>.
// The JSON header maps tensor name -> {dtype, shape, data_offsets:[b,e]}, and
// the optional index file maps tensor name -> shard file name. dtype is
// restricted to "BF16"/"F32": both are delivered to the caller as BF16 bits
// (uint16_t), converting F32 on the fly so the in-memory model is uniformly
// BF16 (see base/bf16_utils.h).
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "base/base.h"

namespace model {

struct SafetensorMeta {
  std::string name;
  std::string dtype;  // "BF16" | "F32" | "F16" ...
  std::vector<int64_t> shape;
  int64_t begin = 0;   // byte offset of the tensor inside the shard data block
  int64_t end = 0;     // (exclusive)
  std::string shard;   // shard file name, relative to the model dir
};

class SafetensorsReader {
 public:
  // Parse <dir>/model.safetensors.index.json (when present) plus the header
  // JSON of every shard it references; a lone *.safetensors file is treated
  // as a single shard. Returns kModelParseError with a message on failure.
  base::Status load(const std::string& model_dir);

  // List of tensor names in the index (unsorted, shard order).
  std::vector<std::string> tensor_names() const;

  bool has(const std::string& name) const { return metas_.count(name) > 0; }

  const SafetensorMeta& meta(const std::string& name) const;

  int64_t numel(const std::string& name) const;

  // Read one tensor into `dst` as BF16 raw bits (little-endian uint16_t per
  // element). `dst` must hold numel(name) elements. F32 tensors are converted
  // during the read; "BF16" is copied verbatim. Returns numel on success and
  // -1 on failure (error details are logged).
  int64_t read_bf16(const std::string& name, uint16_t* dst) const;

 private:
  base::Status parse_shard(const std::string& file_path, const std::string& shard_name,
                           std::unordered_map<std::string, SafetensorMeta>* metas);

  std::string dir_;
  std::unordered_map<std::string, SafetensorMeta> metas_;
};

}  // namespace model
#endif  // SRC_INCLUDE_MODEL_SAFETENSORS_READER_H_
