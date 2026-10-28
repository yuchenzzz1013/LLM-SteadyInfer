#include "model/safetensors_reader.h"
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include "base/bf16_utils.h"

namespace model {

namespace {
constexpr int64_t kSafetensorHeaderPrefix = 8;  // uint64 LE header length

// IEEE-754 binary16 -> binary32 (host-side; only used at load time for
// checkpoints that store scales/biases as F16). Handles subnormals, inf and
// NaN via the standard bit-field expansion.
float fp16_to_fp32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +/- 0
    } else {
      // Subnormal: normalize the mantissa.
      int32_t e = -1;
      do {
        ++e;
        mant <<= 1;
      } while ((mant & 0x400u) == 0);
      mant &= 0x3FFu;
      bits = sign | static_cast<uint32_t>(127 - 15 - e) << 23 | mant << 13;
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | mant << 13;  // inf / NaN
  } else {
    bits = sign | (exp - 15 + 127) << 23 | mant << 13;
  }
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// Read the header JSON of one safetensors file: [8-byte len][JSON][data].
bool parse_shard_header(const std::string& file_path,
                        std::unordered_map<std::string, SafetensorMeta>* metas,
                        std::string* shard_name_out) {
  FILE* file = fopen(file_path.c_str(), "rb");
  if (!file) {
    LOG(ERROR) << "[safetensors] cannot open shard file " << file_path;
    return false;
  }

  uint64_t header_len = 0;
  if (fread(&header_len, sizeof(header_len), 1, file) != 1) {
    LOG(ERROR) << "[safetensors] failed to read header length from " << file_path;
    fclose(file);
    return false;
  }
  if (header_len <= 0 || header_len > (1ull << 30)) {
    LOG(ERROR) << "[safetensors] implausible header length " << header_len << " in "
               << file_path;
    fclose(file);
    return false;
  }

  std::string header(static_cast<size_t>(header_len), '\0');
  if (fread(header.data(), 1, header.size(), file) != header.size()) {
    LOG(ERROR) << "[safetensors] failed to read header from " << file_path;
    fclose(file);
    return false;
  }
  const int64_t data_begin = static_cast<int64_t>(header_len) + kSafetensorHeaderPrefix;
  fclose(file);

  nlohmann::json json;
  try {
    json = nlohmann::json::parse(header);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[safetensors] header JSON parse failed in " << file_path << ": " << e.what();
    return false;
  }

  *shard_name_out = std::filesystem::path(file_path).filename().string();
  for (auto it = json.begin(); it != json.end(); ++it) {
    if (it.key() == "__metadata__") {
      continue;
    }
    const auto& entry = it.value();
    try {
      SafetensorMeta meta;
      meta.name = it.key();
      meta.dtype = entry.at("dtype").get<std::string>();
      const auto& offsets = entry.at("data_offsets");
      meta.begin = offsets.at(0).get<int64_t>();
      meta.end = offsets.at(1).get<int64_t>();
      for (const auto& dim : entry.at("shape")) {
        meta.shape.push_back(dim.get<int64_t>());
      }
      meta.shard = *shard_name_out;
      metas->emplace(meta.name, std::move(meta));
    } catch (const std::exception& e) {
      LOG(ERROR) << "[safetensors] malformed tensor entry '" << it.key() << "' in "
                 << file_path << ": " << e.what();
      return false;
    }
  }
  return true;
}
}  // namespace

base::Status SafetensorsReader::load(const std::string& model_dir) {
  using namespace base;
  dir_ = model_dir;

  // 1. Shard list: the index file (sharded) or a lone *.safetensors file.
  std::vector<std::string> shard_files;  // absolute paths
  std::vector<std::string> shard_names;  // matching file names (index keys)
  const std::string index_path = model_dir + "/model.safetensors.index.json";
  std::error_code ec;
  if (std::filesystem::exists(index_path, ec)) {
    std::ifstream in(index_path);
    if (!in.is_open()) {
      return error::ModelParseError("Failed to open the safetensors index file " + index_path);
    }
    nlohmann::json index;
    try {
      in >> index;
    } catch (const std::exception& e) {
      return error::ModelParseError(std::string("Failed to parse the safetensors index file ") +
                                    index_path + ": " + e.what());
    }
    const auto& weight_map = index.at("weight_map");
    std::unordered_map<std::string, std::string> name_to_shard;
    for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
      name_to_shard[it.key()] = it.value().get<std::string>();
    }
    std::vector<std::string> ordered_shards;
    std::unordered_map<std::string, int32_t> shard_idx;
    for (const auto& [name, shard] : name_to_shard) {
      if (!shard_idx.count(shard)) {
        shard_idx[shard] = static_cast<int32_t>(ordered_shards.size());
        ordered_shards.push_back(shard);
      }
    }
    for (const auto& shard : ordered_shards) {
      shard_names.push_back(shard);
      shard_files.push_back(model_dir + "/" + shard);
    }
  } else {
    for (const auto& entry : std::filesystem::directory_iterator(model_dir, ec)) {
      if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
        shard_files.push_back(entry.path().string());
        shard_names.push_back(entry.path().filename().string());
      }
    }
    std::sort(shard_files.begin(), shard_files.end());
    std::sort(shard_names.begin(), shard_names.end());
  }

  if (shard_files.empty()) {
    return error::PathNotValid("No .safetensors file found under the model dir " + model_dir);
  }

  // 2. Parse every shard header and merge the per-shard name maps.
  std::unordered_map<std::string, SafetensorMeta> all_metas;
  for (size_t i = 0; i < shard_files.size(); ++i) {
    std::unordered_map<std::string, SafetensorMeta> shard_metas;
    std::string parsed_name;
    if (!parse_shard_header(shard_files[i], &shard_metas, &parsed_name)) {
      return error::ModelParseError("Failed to parse the safetensors shard " + shard_files[i]);
    }
    if (!parsed_name.empty()) {
      shard_names[i] = parsed_name;
    }
    for (auto& [name, meta] : shard_metas) {
      meta.shard = shard_names[i];
      if (!all_metas.emplace(name, meta).second) {
        LOG(WARNING) << "[safetensors] duplicated tensor name '" << name
                     << "' across shards; keeping the first occurrence.";
      }
    }
  }
  metas_ = std::move(all_metas);
  LOG(INFO) << "[safetensors] loaded " << metas_.size() << " tensors from "
            << shard_files.size() << " shard file(s) under " << model_dir;
  return error::Success();
}

std::vector<std::string> SafetensorsReader::tensor_names() const {
  std::vector<std::string> names;
  names.reserve(metas_.size());
  for (const auto& [name, meta] : metas_) {
    UNUSED(meta);
    names.push_back(name);
  }
  return names;
}

const SafetensorMeta& SafetensorsReader::meta(const std::string& name) const {
  auto it = metas_.find(name);
  CHECK(it != metas_.end()) << "[safetensors] unknown tensor '" << name << "'";
  return it->second;
}

int64_t SafetensorsReader::numel(const std::string& name) const {
  const auto& m = meta(name);
  if (m.shape.empty()) {
    return 0;
  }
  return std::accumulate(m.shape.begin(), m.shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

int64_t SafetensorsReader::read_tensor(const std::string& name, base::DataType dst_dtype,
                                       void* dst) const {
  if (!has(name) || !dst) {
    return -1;
  }
  const bool dst_bf16 = dst_dtype == base::DataType::kDataTypeBF16;
  const bool dst_fp32 = dst_dtype == base::DataType::kDataTypeFp32;
  const bool dst_i32 = dst_dtype == base::DataType::kDataTypeInt32;
  CHECK(dst_bf16 || dst_fp32 || dst_i32)
      << "[safetensors] unsupported target dtype " << static_cast<int>(dst_dtype);
  const SafetensorMeta& m = meta(name);
  const int64_t n = numel(name);
  const bool src_bf16 = m.dtype == "BF16";
  const bool src_fp32 = m.dtype == "F32";
  const bool src_fp16 = m.dtype == "F16";
  const bool src_i32 = m.dtype == "I32";
  if (!src_bf16 && !src_fp32 && !src_fp16 && !src_i32) {
    LOG(ERROR) << "[safetensors] unsupported dtype '" << m.dtype << "' for tensor '" << name
               << "' (expected BF16/F16/F32/I32)";
    return -1;
  }
  if (src_i32 != dst_i32) {
    LOG(ERROR) << "[safetensors] tensor '" << name << "' is " << m.dtype
               << " but the destination dtype is " << static_cast<int>(dst_dtype)
               << " (integer tensors must stay int32)";
    return -1;
  }
  const std::string file_path = dir_ + "/" + m.shard;

  FILE* file = fopen(file_path.c_str(), "rb");
  if (!file) {
    LOG(ERROR) << "[safetensors] cannot open shard file " << file_path;
    return -1;
  }

  // Data block starts right after the 8-byte length + JSON header.
  uint64_t header_len = 0;
  if (fread(&header_len, sizeof(header_len), 1, file) != 1) {
    fclose(file);
    return -1;
  }
  const int64_t begin = static_cast<int64_t>(header_len) + kSafetensorHeaderPrefix + m.begin;
  if (fseek(file, begin, SEEK_SET) != 0) {
    LOG(ERROR) << "[safetensors] seek failed for '" << name << "' in " << file_path;
    fclose(file);
    return -1;
  }

  int64_t rc = -1;
  const size_t count = static_cast<size_t>(n);
  if (src_i32) {
    // Packed integer payload (AWQ qweight/qzeros): verbatim into int32.
    if (fread(dst, sizeof(int32_t), count, file) == count) {
      rc = n;
    }
  } else if (src_bf16 && dst_bf16) {
    // Same element layout (BF16 bits as uint16_t): copy verbatim.
    if (fread(dst, sizeof(uint16_t), count, file) == count) {
      rc = n;
    }
  } else if (src_fp32 && dst_fp32) {
    if (fread(dst, sizeof(float), count, file) == count) {
      rc = n;
    }
  } else if (src_fp32) {  // F32 -> BF16 (CUDA models downcast at load)
    std::vector<float> tmp(count);
    if (fread(tmp.data(), sizeof(float), count, file) == count) {
      base::fp32_to_bf16_batch(tmp.data(), static_cast<base::bf16_t*>(dst), count);
      rc = n;
    }
  } else if (src_bf16) {  // BF16 -> F32 (CPU models upconvert once at load)
    std::vector<base::bf16_t> tmp(count);
    if (fread(tmp.data(), sizeof(uint16_t), count, file) == count) {
      base::bf16_to_fp32_batch(tmp.data(), static_cast<float*>(dst), count);
      rc = n;
    }
  } else {  // F16 -> F32 / BF16 (AWQ checkpoints written with float16 scales)
    std::vector<uint16_t> tmp(count);
    if (fread(tmp.data(), sizeof(uint16_t), count, file) == count) {
      if (dst_fp32) {
        float* out = static_cast<float*>(dst);
        for (size_t i = 0; i < count; ++i) out[i] = fp16_to_fp32(tmp[i]);
      } else {
        base::bf16_t* out = static_cast<base::bf16_t*>(dst);
        for (size_t i = 0; i < count; ++i) {
          out[i] = base::fp32_to_bf16(fp16_to_fp32(tmp[i]));
        }
      }
      rc = n;
    }
  }

  if (rc < 0) {
    LOG(ERROR) << "[safetensors] failed to read '" << name << "' from " << file_path;
  }
  fclose(file);
  return rc;
}

}  // namespace model
