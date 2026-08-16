#pragma once

#include <cstdint>
#include <vector>

namespace scheduler {

// vLLM-style physical block pool (one pool serves every layer: block i of
// layer l lives at key_cache[l][i][*][*]). Blocks are allocated in whole
// rows — one row == one sequence's block table — and every row is
// max_blocks_per_seq entries wide (the per-sequence KV capacity, in blocks).
//
// The host-side block table mirror is the single source of truth; the model
// forward paths receive it as a [batch, max_blocks_per_seq] tensor (H2D
// upload per step, captured into the decode CUDA graphs like the other
// per-token inputs). -1 marks an unused table entry.
//
// Refcounting (inc_ref/dec_ref) enables prefix-cache sharing: several rows
// may reference the same physical blocks read-only. A block returns to the
// free list only when its refcount drops to zero, so a shared block is never
// handed to a writer while another sequence still reads it.
class BlockAllocator {
 public:
  // num_blocks: total physical blocks; num_rows: max concurrent sequences
  // (block table count); max_blocks_per_seq: block table width.
  BlockAllocator(int num_blocks, int num_rows, int max_blocks_per_seq, int block_size);

  // Allocate `n` blocks for a free row (n == 0 returns a row with an empty
  // table — prefix-cache-only rows). Returns the row index, -1 on failure
  // (all failure conditions are checked before any state is mutated, so a
  // failed call leaves the pool untouched).
  int allocate_blocks(int n);

  // Mount `blocks` at the front of `row`'s block table (existing entries
  // shift right) and bump their refcounts — prefix-cache read-only sharing.
  // Returns false when the row does not exist or overflows the table width.
  bool reserve_shared_prefix(int row, const std::vector<int32_t>& blocks);

  // Release every block of a row (refcount--; a block shared by other rows
  // via the prefix cache stays allocated until its refcount hits zero).
  void free_all(int row);

  // Grow a row by `n` additional blocks (appended after the current ones).
  // Returns the new used count, or -1 without mutating anything when the
  // pool cannot satisfy the request (lazy generation growth).
  int append_blocks(int row, int n);

  // Release blocks [start_block_idx, num_used) of a row — preemption
  // truncation (the retained prefix keeps its blocks).
  void free_blocks_from(int row, int start_block_idx);

  // Number of allocated (non -1) entries in a row's block table.
  int num_used_blocks(int row) const;

  // Copy a row's block table into dst (max_blocks_per_seq entries).
  void copy_block_table_row(int row, int32_t* dst) const;
  const int32_t* block_table_row(int row) const;

  int free_block_count() const;
  bool has_free_blocks(int n) const { return free_block_count() >= n; }
  int num_blocks() const { return num_blocks_; }
  int block_size() const { return block_size_; }
  int max_blocks_per_seq() const { return max_blocks_per_seq_; }
  int num_rows() const { return num_rows_; }

  // Prefix-cache sharing (M3): reference-count a physical block.
  void inc_ref(int block_idx);
  void dec_ref(int block_idx);
  int ref_count(int block_idx) const;

  // Invariant check: free list + refcounted == num_blocks. LOG(FATAL)s on
  // violation — guards against refcount leaks silently stranding blocks.
  bool invariant_holds() const;

 private:
  void push_free_block(int block_idx);

  int num_blocks_ = 0;
  int num_rows_ = 0;
  int max_blocks_per_seq_ = 0;
  int block_size_ = 0;
  std::vector<int> free_blocks_;                  // LIFO free list
  std::vector<int> ref_counts_;                   // per physical block
  std::vector<int32_t> block_tables_;             // flat [num_rows * max_blocks_per_seq]
};

}  // namespace scheduler
