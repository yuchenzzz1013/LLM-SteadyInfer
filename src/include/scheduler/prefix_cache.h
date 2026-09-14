#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "block_allocator.h"

namespace scheduler {

// Prefix-cache effectivenes counters (Scheduler::prefix_cache_stats).
struct PrefixCacheStats {
  int64_t lookups = 0;         // lookup() calls
  int64_t hits = 0;            // lookups that matched >= 1 whole block
  int64_t matched_blocks = 0;  // cumulative matched blocks over all lookups
  int64_t inserts = 0;         // entries inserted
  int64_t evictions = 0;       // entries evicted (LRU)
  double hit_rate() const {
    return lookups > 0 ? static_cast<double>(hits) / static_cast<double>(lookups) : 0.0;
  }
};

// Content-hash prefix cache (vLLM-style, share-without-CoW design): maps the
// chained 64-bit FNV-1a hash of a token block to the physical block holding
// that block's KV. A new request whose prompt starts with matching whole
// blocks shares them read-only (refcounted via the BlockAllocator) and skips
// prefilling the shared region — the TTFT win.
//
// Two properties make the sharing safe without copy-on-write:
//
//  1. Prefix-dependent hashing: hash(b) = f(hash(b-1), tokens_b), so a block is
//     only ever matched against a block reached through the identical prefix
//     chain. Identical content under different prefixes gets different hashes
//     (and a prompt repeating one block gets one hash per position), so the
//     cache can never hand a caller KV that was computed from another context.
//  2. A cached block is pinned: the allocator's CACHED state keeps it out of
//     the free list, so a writer can never be handed a block that a cached
//     entry points at. Blocks are released only by LRU eviction (when no
//     sequence references them) or by the pool reclaiming them under pressure.
//
// The last prompt block is deliberately never cached/shared: the first decode
// step re-runs the last prompt token at its own position (prefill produces no
// logits), which rewrites that token's KV slot. Landing that write in a shared
// block would violate the read-only invariant (and make output depend on which
// sequence wrote last), so cacheable_blocks() stops one block short.
//
// Entries store the block's token ids and validate them on every candidate
// match: a 64-bit hash collision is then a miss, not silent KV corruption.
class PrefixCache {
 public:
  static constexpr size_t kDefaultMaxEntries = 4096;

  // block_size: tokens per physical block (must match the KV pool's).
  // max_entries: LRU capacity in blocks; the cache may grow to 2x this while
  // every entry is still referenced by a running sequence, then stops
  // inserting (blocks of the losing inserts stay live but are not cached).
  explicit PrefixCache(int block_size, size_t max_entries = kDefaultMaxEntries);
  ~PrefixCache();

  // Longest whole-block prefix of `prompt_tokens` that can be shared. Fills
  // `shared_blocks` with the physical block ids (one per matched block, in
  // order) and returns the matched count. Touches the matched entries (LRU).
  int lookup(const std::vector<int>& prompt_tokens, BlockAllocator& allocator,
             std::vector<int32_t>* shared_blocks);

  // Record every committed full prompt block of a sequence (idempotent and
  // incremental — call it after each chunked-prefill step).
  //
  //   completed_tokens: prompt tokens whose KV is committed (== the sequence's
  //     next_prefill_chunk_start)
  //   row_table:        the owning row's block table (block index b of the
  //     prompt == table[b] for every caller: shared prefixes keep prompt block
  //     indices, private blocks follow them)
  //   hash_chain:       per-sequence progress cursor. hash_chain[b] is the
  //     chained hash of prompt block b; the chain is filled for blocks
  //     [0, hash_chain->size()) and doubles as "how far this sequence has
  //     already been recorded", so a block is never hashed or inserted twice.
  //     Clearing the sequence's token history (never done today) would require
  //     clearing this too.
  //
  // Returns the number of blocks newly inserted. Blocks whose hash is already
  // in the cache are skipped: the resident entry already pins an equivalent
  // block (vLLM: first insert wins), which also keeps a shared prefix from
  // pinning the sharer's private copy.
  int insert_committed(const int* prompt_tokens, int num_prompt_tokens,
                       int completed_tokens, const int32_t* row_table,
                       std::vector<uint64_t>* hash_chain, BlockAllocator& allocator);

  // Evict up to `n` LRU entries (fewer when the tail entries are still in use),
  // returning the number evicted. Entries whose block is still referenced by a
  // sequence are skipped — releasing them would unpin KV a running row reads.
  // This is the BlockAllocator's reclaim source when the free list runs dry.
  int evict(int n, BlockAllocator& allocator);

  // Whole-block prefix of `num_prompt_tokens` that may be cached/shared (see
  // the class comment: the block holding the last prompt token is excluded).
  int cacheable_blocks(int num_prompt_tokens) const {
    return num_prompt_tokens <= 0 ? 0 : (num_prompt_tokens - 1) / block_size_;
  }

  size_t entry_count() const { return entries_.size(); }
  size_t max_entries() const { return max_entries_; }
  int block_size() const { return block_size_; }
  const PrefixCacheStats& stats() const { return stats_; }

  // Chained block hash: FNV-1a continued from `prev_hash` over one block of
  // token ids (0 for a prompt's first block). Bijective in prev_hash, so
  // different prefixes never collide on the same block content.
  static uint64_t block_hash(uint64_t prev_hash, const int* tokens, int n);

 private:
  // Hash -> cached block. One entry per hash: a duplicate insert means the
  // same (prefix, content) is already cached, so the resident entry wins.
  struct Entry {
    uint64_t hash = 0;
    uint64_t prev_hash = 0;
    int32_t block_id = -1;         // physical block holding this block's KV
    std::vector<int32_t> tokens;   // this block's token ids (collision check)
    Entry* prev = nullptr;         // LRU list: head = most recent
    Entry* next = nullptr;
  };

  // LRU list primitives (intrusive, O(1)).
  void unlink(Entry* e);
  void push_front(Entry* e);
  void touch(Entry* e);

  // Evict up to `n` unreferenced entries from the tail; returns the count.
  int evict_lru(int n, BlockAllocator& allocator);

  bool tokens_match(const Entry& e, const int* tokens) const;

  int block_size_ = 16;
  size_t max_entries_ = kDefaultMaxEntries;
  // Insert is refused past this (only reachable while every entry is
  // referenced): bounds the entry map when the pool is too small for the
  // working set. Blocks past the cap are simply not cached.
  size_t hard_max_entries_ = 2 * kDefaultMaxEntries;
  std::unordered_map<uint64_t, Entry*> entries_;  // hash -> Entry
  Entry* lru_head_ = nullptr;                     // most recently used
  Entry* lru_tail_ = nullptr;                     // eviction candidate
  PrefixCacheStats stats_;
};

}  // namespace scheduler
