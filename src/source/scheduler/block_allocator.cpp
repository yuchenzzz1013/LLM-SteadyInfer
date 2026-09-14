#include "scheduler/block_allocator.h"
#include <algorithm>
#include <glog/logging.h>

namespace scheduler {

BlockAllocator::BlockAllocator(int num_blocks, int num_rows, int max_blocks_per_seq, int block_size)
    : num_blocks_(num_blocks),
      num_rows_(num_rows),
      max_blocks_per_seq_(max_blocks_per_seq),
      block_size_(block_size),
      free_blocks_(num_blocks),
      ref_counts_(num_blocks, 0),
      cached_(num_blocks, 0),
      block_tables_(static_cast<size_t>(num_rows) * max_blocks_per_seq, -1) {
  // Start with the whole pool free. Fill back-to-front so the first
  // allocation pops block 0 and allocation locality stays ascending.
  for (int i = 0; i < num_blocks_; ++i) {
    free_blocks_[num_blocks_ - 1 - i] = i;
  }
  CHECK_GT(block_size_, 0) << "block_size must be positive";
  CHECK_GT(num_blocks_, 0) << "num_blocks must be positive";
}

void BlockAllocator::reclaim_for(size_t n) {
  if (free_blocks_.size() >= n || !reclaim_fn_) return;
  reclaim_fn_(static_cast<int>(n - free_blocks_.size()));
}

int BlockAllocator::append_blocks(int row, int n) {
  if (row < 0 || row >= num_rows_ || n <= 0) {
    return -1;
  }
  const int used = num_used_blocks(row);
  if (used < 0 || used + n > max_blocks_per_seq_) {
    return -1;
  }
  // Free list first, cached blocks second (prefix-cache LRU eviction).
  reclaim_for(static_cast<size_t>(n));
  if (static_cast<int>(free_blocks_.size()) < n) {
    return -1;
  }
  for (int i = 0; i < n; ++i) {
    int block_idx = free_blocks_.back();
    free_blocks_.pop_back();
    CHECK_EQ(ref_counts_[block_idx], 0) << "block " << block_idx
                                        << " in free list with refcount > 0";
    CHECK_EQ(cached_[block_idx], 0) << "block " << block_idx
                                    << " in free list but still cached";
    ref_counts_[block_idx] = 1;
    block_tables_[static_cast<size_t>(row) * max_blocks_per_seq_ + used + i] = block_idx;
  }
  return used + n;
}

int BlockAllocator::allocate_blocks(int n) {
  if (n > max_blocks_per_seq_) {
    LOG(ERROR) << "[BLOCK] allocate_blocks: " << n << " exceeds table width "
               << max_blocks_per_seq_;
    return -1;
  }
  if (n < 0) {
    return -1;
  }
  reclaim_for(static_cast<size_t>(n));
  if (static_cast<int>(free_blocks_.size()) < n) {
    return -1;
  }

  // Find a free row (first-fit, same policy as the old slot allocator).
  int row = -1;
  for (int r = 0; r < num_rows_; ++r) {
    if (block_tables_[static_cast<size_t>(r) * max_blocks_per_seq_] == -1) {
      row = r;
      break;
    }
  }
  if (row < 0) {
    return -1;
  }

  for (int i = 0; i < n; ++i) {
    int block_idx = free_blocks_.back();
    free_blocks_.pop_back();
    CHECK_EQ(ref_counts_[block_idx], 0) << "block " << block_idx
                                        << " in free list with refcount > 0";
    CHECK_EQ(cached_[block_idx], 0) << "block " << block_idx
                                    << " in free list but still cached";
    ref_counts_[block_idx] = 1;
    block_tables_[static_cast<size_t>(row) * max_blocks_per_seq_ + i] = block_idx;
  }
  return row;
}

bool BlockAllocator::reserve_shared_prefix(int row, const std::vector<int32_t>& blocks) {
  if (row < 0 || row >= num_rows_ || blocks.empty()) return false;
  const int used = num_used_blocks(row);
  const int n = static_cast<int>(blocks.size());
  if (used + n > max_blocks_per_seq_) return false;

  // Validate every block before mutating anything (same contract as
  // allocate_blocks: a failed call leaves the pool untouched). A CACHED block
  // has refcount 0 but is mountable — the cache pins its KV; a FREE block is
  // not (the pool may hand it to a writer at any moment).
  for (int i = 0; i < n; ++i) {
    const int b = blocks[i];
    if (b < 0 || b >= num_blocks_) return false;
    if (ref_counts_[b] == 0 && !cached_[b]) {
      LOG(ERROR) << "[BLOCK] reserve_shared_prefix: shared prefix block " << b
                 << " is not allocated (refcount 0, not cached)";
      return false;
    }
  }

  int32_t* table = block_tables_.data() + static_cast<size_t>(row) * max_blocks_per_seq_;
  // Shift the row's existing (private) entries right by n; the shared prefix
  // takes the front. std::copy_backward handles the overlap.
  std::copy_backward(table, table + used, table + used + n);
  for (int i = 0; i < n; ++i) {
    table[i] = blocks[i];
    // 0 -> 1 revives a CACHED block (it stays cached: when this row lets go it
    // parks back in the cache rather than the free list); >0 shares a block
    // another row (or several) already holds.
    ++ref_counts_[blocks[i]];  // shared read-only reference
  }
  return true;
}

void BlockAllocator::free_blocks_from(int row, int start_block_idx) {
  if (row < 0 || row >= num_rows_) return;
  for (int i = start_block_idx; i < max_blocks_per_seq_; ++i) {
    int32_t& slot = block_tables_[static_cast<size_t>(row) * max_blocks_per_seq_ + i];
    if (slot < 0) continue;
    const int block_idx = slot;
    slot = -1;
    CHECK_GT(ref_counts_[block_idx], 0) << "freeing block " << block_idx
                                        << " with zero refcount (row=" << row << ")";
    if (--ref_counts_[block_idx] == 0) {
      release_block(block_idx);
    }
  }
}

void BlockAllocator::free_all(int row) { free_blocks_from(row, 0); }

int BlockAllocator::num_used_blocks(int row) const {
  if (row < 0 || row >= num_rows_) return 0;
  const int32_t* table = block_table_row(row);
  int used = 0;
  for (int i = 0; i < max_blocks_per_seq_; ++i) {
    if (table[i] >= 0) ++used;
  }
  return used;
}

const int32_t* BlockAllocator::block_table_row(int row) const {
  CHECK_GE(row, 0) << "block_table_row: negative row";
  CHECK_LT(row, num_rows_) << "block_table_row: row out of range";
  return block_tables_.data() + static_cast<size_t>(row) * max_blocks_per_seq_;
}

void BlockAllocator::copy_block_table_row(int row, int32_t* dst) const {
  const int32_t* src = block_table_row(row);
  for (int i = 0; i < max_blocks_per_seq_; ++i) {
    dst[i] = src[i];
  }
}

int BlockAllocator::free_block_count() const {
  return static_cast<int>(free_blocks_.size());
}

void BlockAllocator::push_free_block(int block_idx) {
  free_blocks_.push_back(block_idx);
}

void BlockAllocator::release_block(int block_idx) {
  if (cached_[block_idx]) {
    // The prefix cache still points at this block's KV: park it in the CACHED
    // state (out of the free list) so it stays shareable. The cache releases
    // it via evict_cached() when the pool needs it back.
    return;
  }
  push_free_block(block_idx);
}

void BlockAllocator::inc_ref(int block_idx) {
  CHECK_GE(block_idx, 0);
  CHECK_LT(block_idx, num_blocks_);
  CHECK_GT(ref_counts_[block_idx], 0) << "inc_ref on a free block " << block_idx;
  ++ref_counts_[block_idx];
}

void BlockAllocator::dec_ref(int block_idx) {
  CHECK_GE(block_idx, 0);
  CHECK_LT(block_idx, num_blocks_);
  CHECK_GT(ref_counts_[block_idx], 0) << "dec_ref on a free block " << block_idx;
  if (--ref_counts_[block_idx] == 0) {
    release_block(block_idx);
  }
}

int BlockAllocator::ref_count(int block_idx) const {
  if (block_idx < 0 || block_idx >= num_blocks_) return 0;
  return ref_counts_[block_idx];
}

void BlockAllocator::mark_cached(int block_idx) {
  CHECK_GE(block_idx, 0);
  CHECK_LT(block_idx, num_blocks_);
  CHECK_GT(ref_counts_[block_idx], 0)
      << "mark_cached on an unreferenced block " << block_idx
      << ": its KV is not pinned by any row";
  cached_[block_idx] = 1;
}

void BlockAllocator::evict_cached(int block_idx) {
  CHECK_GE(block_idx, 0);
  CHECK_LT(block_idx, num_blocks_);
  CHECK_EQ(ref_counts_[block_idx], 0)
      << "evict_cached on referenced block " << block_idx << " (refcount "
      << ref_counts_[block_idx] << "): a row still reads it";
  if (!cached_[block_idx]) return;
  cached_[block_idx] = 0;
  push_free_block(block_idx);
}

bool BlockAllocator::is_cached(int block_idx) const {
  if (block_idx < 0 || block_idx >= num_blocks_) return false;
  return cached_[block_idx] != 0;
}

int BlockAllocator::cached_block_count() const {
  int n = 0;
  for (int i = 0; i < num_blocks_; ++i) {
    if (cached_[i]) ++n;
  }
  return n;
}

bool BlockAllocator::invariant_holds() const {
  int refcounted = 0;
  int cached_unreferenced = 0;
  for (int i = 0; i < num_blocks_; ++i) {
    if (ref_counts_[i] > 0) {
      ++refcounted;
    } else if (cached_[i]) {
      ++cached_unreferenced;
    }
  }
  return static_cast<int>(free_blocks_.size()) + refcounted + cached_unreferenced == num_blocks_;
}

std::string BlockAllocator::debug_string() const {
  int refcounted = 0;
  int cached_unreferenced = 0;
  for (int i = 0; i < num_blocks_; ++i) {
    if (ref_counts_[i] > 0) {
      ++refcounted;
    } else if (cached_[i]) {
      ++cached_unreferenced;
    }
  }
  return "blocks=" + std::to_string(num_blocks_) + " free=" +
         std::to_string(free_blocks_.size()) + " referenced=" + std::to_string(refcounted) +
         " cached=" + std::to_string(cached_unreferenced);
}

}  // namespace scheduler
