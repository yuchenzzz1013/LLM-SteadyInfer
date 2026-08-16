#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "block_allocator.h"

namespace scheduler {

// Content-hash prefix cache (vLLM-style, share-without-CoW design):
// maps the 64-bit FNV-1a hash of a whole token block to the physical blocks
// holding that block's KV. A new request whose prompt starts with matching
// whole blocks shares them read-only (refcounted via the BlockAllocator) and
// skips prefilling the shared region — the TTFT win. Positions the request
// itself writes always land in private blocks (the shared prefix is never
// written again, and partial boundary blocks are never shared), so no
// copy-on-write is needed anywhere.
//
// Entries are only recorded once the owner's prefill has completed (the KV
// is committed); lookups validate each candidate against the allocator — the
// block must still be referenced and still mounted at the recorded (row,
// index) — so recycled or truncated blocks never leak wrong KV. Hash
// collisions (64-bit) are treated as a match, which is correct in practice.
class PrefixCache {
 public:
  explicit PrefixCache(int block_size) : block_size_(block_size) {}

  // Longest whole-block prefix of `prompt_tokens` that can be shared. Fills
  // `shared_blocks` with the physical block ids (one per matched block) and
  // returns the matched count.
  int lookup(const std::vector<int>& prompt_tokens, const BlockAllocator& allocator,
             std::vector<int32_t>* shared_blocks) const;

  // Record the full prompt blocks of `row`. Call when the sequence's prefill
  // completes — only then is the KV content committed and shareable.
  void insert(const std::vector<int>& prompt_tokens, const BlockAllocator& allocator, int row);

  size_t entry_count() const { return map_.size(); }

 private:
  struct Entry {
    int row = -1;         // block-table row that owns the block
    int block_index = 0;  // index within that row's block table
    int32_t block_id = -1;
  };

  static uint64_t block_hash(const int* tokens, int n);

  int block_size_ = 16;
  std::unordered_map<uint64_t, std::vector<Entry>> map_;
};

}  // namespace scheduler
