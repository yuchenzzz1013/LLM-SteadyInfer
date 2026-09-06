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

int64_t SafetensorsReader::read_bf16(const std::string& name, uint16_t* dst) const {
  if (!has(name) || !dst) {
    return -1;
  }
  const SafetensorMeta& m = meta(name);
  const int64_t n = numel(name);
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
  if (m.dtype == "BF16") {
    if (fread(dst, sizeof(uint16_t), static_cast<size_t>(n), file) ==
        static_cast<size_t>(n)) {
      rc = n;
    }
  } else if (m.dtype == "F32") {
    std::vector<float> tmp(static_cast<size_t>(n));
    if (fread(tmp.data(), sizeof(float), static_cast<size_t>(n), file) ==
        static_cast<size_t>(n)) {
      base::fp32_to_bf16_batch(tmp.data(), dst, static_cast<size_t>(n));
      rc = n;
    }
  } else {
    LOG(ERROR) << "[safetensors] unsupported dtype '" << m.dtype << "' for tensor '"
               << name << "' (expected BF16 or F32)";
  }

  if (rc < 0) {
    LOG(ERROR) << "[safetensors] failed to read '" << name << "' from " << file_path;
  }
  fclose(file);
  return rc;
}

}  // namespace model
