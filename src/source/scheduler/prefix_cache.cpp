#include "scheduler/prefix_cache.h"
#include <algorithm>
#include <glog/logging.h>

namespace scheduler {

// FNV-1a, 64-bit. Tokens are hashed byte-wise (int32 tokens: 4 bytes each).
uint64_t PrefixCache::block_hash(const int* tokens, int n) {
  uint64_t h = 14695981039346656037ULL;
  for (int i = 0; i < n; ++i) {
    const uint32_t t = static_cast<uint32_t>(tokens[i]);
    for (int shift = 0; shift < 32; shift += 8) {
      h ^= (t >> shift) & 0xFF;
      h *= 1099511628211ULL;
    }
  }
  return h;
}

int PrefixCache::lookup(const std::vector<int>& prompt_tokens,
                        const BlockAllocator& allocator,
                        std::vector<int32_t>* shared_blocks) const {
  shared_blocks->clear();
  if (prompt_tokens.empty()) return 0;
  const int full_blocks = static_cast<int>(prompt_tokens.size()) / block_size_;

  for (int b = 0; b < full_blocks; ++b) {
    const uint64_t h = block_hash(prompt_tokens.data() + b * block_size_, block_size_);
    const auto it = map_.find(h);
    if (it == map_.end()) break;
    bool found = false;
    for (const Entry& e : it->second) {
      // Guard: the block must still be referenced (owner alive or another
      // sharer holds it) AND still mounted at the recorded (row, index) —
      // i.e. its content is still the KV the hash was recorded for. Recycled
      // (refcount 0) or truncated (table moved on) blocks are rejected.
      if (allocator.ref_count(e.block_id) <= 0) continue;
      const int32_t* row = allocator.block_table_row(e.row);
      if (row[e.block_index] != e.block_id) continue;
      shared_blocks->push_back(e.block_id);
      found = true;
      break;
    }
    if (!found) break;
  }
  return static_cast<int>(shared_blocks->size());
}

void PrefixCache::insert(const std::vector<int>& prompt_tokens,
                         const BlockAllocator& allocator, int row) {
  if (prompt_tokens.empty()) return;
  const int full_blocks = static_cast<int>(prompt_tokens.size()) / block_size_;
  if (full_blocks <= 0) return;

  // Coarse cap: a full flush beats unbounded growth. Entries are cheap to
  // rebuild (the next identical-prefix request re-records them).
  constexpr size_t kMaxEntries = 1u << 16;
  if (map_.size() >= kMaxEntries) {
#ifndef NDEBUG
    VLOG(1) << "[PREFIX] entry cap reached (" << kMaxEntries << "); flushing";
#endif
    map_.clear();
  }

  const int32_t* table = allocator.block_table_row(row);
  for (int b = 0; b < full_blocks; ++b) {
    const uint64_t h = block_hash(prompt_tokens.data() + b * block_size_, block_size_);
    map_[h].push_back(Entry{row, b, table[b]});
  }
}

}  // namespace scheduler
