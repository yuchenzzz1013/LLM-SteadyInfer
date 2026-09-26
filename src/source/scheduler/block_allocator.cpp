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
      block_tables_(static_cast<size_t>(num_rows) * max_blocks_per_seq, -1),
      row_busy_(num_rows, 0) {
  // Start with the whole pool free. Fill back-to-front so the first
  // allocation pops block 0 and allocation locality stays ascending.
  for (int i = 0; i < num_blocks_; ++i) {
    free_blocks_[num_blocks_ - 1 - i] = i;
  }
  CHECK_GT(block_size_, 0) << "block_size must be positive";
  CHECK_GT(num_blocks_, 0) << "num_blocks must be positive";
}

void BlockAllocator::reclaim_for(int n) {
  if (n <= 0 || !reclaim_fn_) return;
  const int short_by = n - static_cast<int>(free_blocks_.size());
  if (short_by <= 0) return;
  reclaim_fn_(short_by);
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
  reclaim_for(n);
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
  if (n < 0) {
    return -1;
  }
  if (n > max_blocks_per_seq_) {
    // Log once: this fires from the admission loop, so a persistent mismatch
    // (e.g. a block size that disagrees with the pool's) would otherwise flood
    // the log at scheduler-step rate — it once produced tens of GB of /tmp.
    static bool warned = false;
    if (!warned) {
      warned = true;
      LOG(ERROR) << "[BLOCK] allocate_blocks: " << n << " exceeds table width "
                 << max_blocks_per_seq_ << " (further occurrences not logged)";
    }
    return -1;
  }

  // Find a free row first (first-fit, same policy as the old slot allocator):
  // reclaiming cached blocks is pointless when the limit is rows, not blocks.
  // row_busy_ (not "table[0] == -1") is what marks a row as taken, so an n == 0
  // row — prefix-cache-only, empty table — cannot be handed out twice.
  int row = -1;
  for (int r = 0; r < num_rows_; ++r) {
    if (!row_busy_[r]) {
      row = r;
      break;
    }
  }
  if (row < 0) {
    return -1;
  }

  reclaim_for(n);
  if (static_cast<int>(free_blocks_.size()) < n) {
    return -1;
  }

  row_busy_[row] = 1;
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
  bool any_left = false;
  for (int i = 0; i < max_blocks_per_seq_; ++i) {
    int32_t& slot = block_tables_[static_cast<size_t>(row) * max_blocks_per_seq_ + i];
    if (slot < 0) continue;
    if (i >= start_block_idx) {
      const int block_idx = slot;
      slot = -1;
      CHECK_GT(ref_counts_[block_idx], 0) << "freeing block " << block_idx
                                          << " with zero refcount (row=" << row << ")";
      if (--ref_counts_[block_idx] == 0) {
        release_block(block_idx);
      }
    } else {
      any_left = true;
    }
  }
  // An empty row is free again; a row keeping its prefix (truncation) is not.
  if (!any_left) row_busy_[row] = 0;
}

int BlockAllocator::reclaimable_blocks_from(int row, int start_block_idx) const {
  if (row < 0 || row >= num_rows_) return 0;
  const int32_t* table = block_table_row(row);
  int n = 0;
  for (int i = std::max(0, start_block_idx); i < max_blocks_per_seq_; ++i) {
    const int b = table[i];
    // Only blocks this row alone owns come back to the free list: a shared
    // block (refcount > 1) stays mounted in the other rows, and a cached block
    // is parked for the prefix cache instead of being handed out again.
    if (b >= 0 && ref_counts_[b] == 1 && !cached_[b]) ++n;
  }
  return n;
}

int BlockAllocator::num_busy_rows() const {
  int n = 0;
  for (uint8_t busy : row_busy_) {
    if (busy) ++n;
  }
  return n;
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
