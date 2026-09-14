#include "scheduler/prefix_cache.h"
#include <algorithm>
#include <glog/logging.h>

namespace scheduler {

namespace {
// Per-call cap on the LRU walk in evict_lru: a pool under pressure may have
// most entries still referenced, and walking a huge cache on every failed
// allocation would cost more than the eviction is worth. Anything left is
// retried on the next allocation attempt.
constexpr int kEvictScanCap = 1024;
}  // namespace

// FNV-1a, 64-bit, continued from the previous block's hash: the seed is the
// running FNV state, so a block's hash depends on the whole prefix chain in
// front of it. Both xoring and multiplying by the (odd) FNV prime are
// bijections on the 64-bit state, so distinct prev_hash values produce
// distinct hashes for the same content — identical blocks under different
// prefixes (or repeated blocks inside one prompt) never alias.
// Tokens are folded byte-wise (int32 little-endian).
uint64_t PrefixCache::block_hash(uint64_t prev_hash, const int* tokens, int n) {
  uint64_t h = 14695981039346656037ULL ^ prev_hash;
  for (int i = 0; i < n; ++i) {
    const uint32_t t = static_cast<uint32_t>(tokens[i]);
    for (int shift = 0; shift < 32; shift += 8) {
      h ^= (t >> shift) & 0xFF;
      h *= 1099511628211ULL;
    }
  }
  return h;
}

PrefixCache::PrefixCache(int block_size, size_t max_entries)
    : block_size_(block_size),
      max_entries_(std::max<size_t>(1, max_entries)),
      hard_max_entries_(2 * std::max<size_t>(1, max_entries)) {
  CHECK_GT(block_size_, 0) << "PrefixCache: block_size must be positive";
}

PrefixCache::~PrefixCache() {
  // The entries own their LRU nodes. The blocks they pin stay marked CACHED in
  // the allocator: the pool is destroyed with the scheduler and nothing
  // allocates in between, so there is nothing to unwind (and no allocator to
  // unwind through — the cache is allocator-agnostic by design).
  Entry* e = lru_head_;
  while (e != nullptr) {
    Entry* next = e->next;
    delete e;
    e = next;
  }
}

bool PrefixCache::tokens_match(const Entry& e, const int* tokens) const {
  if (static_cast<int>(e.tokens.size()) != block_size_) return false;
  for (int i = 0; i < block_size_; ++i) {
    if (e.tokens[static_cast<size_t>(i)] != static_cast<int32_t>(tokens[i])) return false;
  }
  return true;
}

void PrefixCache::unlink(Entry* e) {
  if (e->prev) e->prev->next = e->next;
  else lru_head_ = e->next;
  if (e->next) e->next->prev = e->prev;
  else lru_tail_ = e->prev;
  e->prev = e->next = nullptr;
}

void PrefixCache::push_front(Entry* e) {
  e->prev = nullptr;
  e->next = lru_head_;
  if (lru_head_) lru_head_->prev = e;
  lru_head_ = e;
  if (lru_tail_ == nullptr) lru_tail_ = e;
}

void PrefixCache::touch(Entry* e) {
  if (e == lru_head_) return;
  unlink(e);
  push_front(e);
}

int PrefixCache::lookup(const std::vector<int>& prompt_tokens, BlockAllocator& allocator,
                        std::vector<int32_t>* shared_blocks) {
  shared_blocks->clear();
  ++stats_.lookups;
  const int max_blocks = cacheable_blocks(static_cast<int>(prompt_tokens.size()));

  uint64_t prev_hash = 0;
  for (int b = 0; b < max_blocks; ++b) {
    const int* block_tokens = prompt_tokens.data() + static_cast<size_t>(b) * block_size_;
    const uint64_t h = block_hash(prev_hash, block_tokens, block_size_);
    const auto it = entries_.find(h);
    if (it == entries_.end()) break;
    Entry* e = it->second;
    // Candidate validation. The chain hash already pins the prefix, but a
    // 64-bit collision (or an entry whose block was released behind the
    // cache's back) must be a miss, never a share of the wrong KV.
    if (e->prev_hash != prev_hash || !tokens_match(*e, block_tokens)) {
#ifndef NDEBUG
      VLOG(1) << "[PREFIX] hash collision or stale entry at block " << b << "; miss";
#endif
      break;
    }
    if (!allocator.is_cached(e->block_id) && allocator.ref_count(e->block_id) <= 0) {
#ifndef NDEBUG
      VLOG(1) << "[PREFIX] entry block " << e->block_id << " no longer held; miss";
#endif
      break;
    }
    shared_blocks->push_back(e->block_id);
    touch(e);  // matched entries are the most recently used: evict them last
    prev_hash = h;  // only a matched block lets the chain advance
  }

  const int matched = static_cast<int>(shared_blocks->size());
  if (matched > 0) {
    ++stats_.hits;
    stats_.matched_blocks += matched;
  }
  return matched;
}

int PrefixCache::insert_committed(const int* prompt_tokens, int num_prompt_tokens,
                                  int completed_tokens, const int32_t* row_table,
                                  std::vector<uint64_t>* hash_chain,
                                  BlockAllocator& allocator) {
  if (prompt_tokens == nullptr || row_table == nullptr || hash_chain == nullptr) return 0;
  const int cacheable = cacheable_blocks(num_prompt_tokens);
  int committed_blocks = completed_tokens / block_size_;
  if (committed_blocks > cacheable) committed_blocks = cacheable;

  int inserted = 0;
  for (int b = static_cast<int>(hash_chain->size()); b < committed_blocks; ++b) {
    const int32_t block_id = row_table[b];
    if (block_id < 0) break;  // table not grown this far (defensive)
    const uint64_t prev_hash = hash_chain->empty() ? 0 : hash_chain->back();
    const uint64_t h =
        block_hash(prev_hash, prompt_tokens + static_cast<size_t>(b) * block_size_, block_size_);
    // Advance the sequence's chain before any insert decision: the cursor must
    // stay in lockstep with the blocks the sequence has committed, whether or
    // not this block ends up cached.
    hash_chain->push_back(h);

    if (entries_.find(h) != entries_.end()) {
      // Same chained hash == same (prefix, content) already cached: the
      // resident entry wins (vLLM's first-insert-wins), so the caller's block
      // stays private instead of pinning a second copy.
      continue;
    }
    if (entries_.size() >= max_entries_) {
      evict_lru(1, allocator);  // no-op while every entry is still referenced
    }
    if (entries_.size() >= hard_max_entries_) {
      // Cache full of in-use entries: give up caching this block rather than
      // growing without bound. The block stays live, just not shareable.
#ifndef NDEBUG
      VLOG(1) << "[PREFIX] entry cap " << hard_max_entries_ << " reached; block not cached";
#endif
      continue;
    }

    Entry* e = new Entry();
    e->hash = h;
    e->prev_hash = prev_hash;
    e->block_id = block_id;
    e->tokens.resize(block_size_);
    for (int i = 0; i < block_size_; ++i) {
      e->tokens[static_cast<size_t>(i)] =
          static_cast<int32_t>(prompt_tokens[static_cast<size_t>(b) * block_size_ + i]);
    }
    // Pin the block: it now has a cache entry, so freeing the owning row's
    // reference must not return it to the pool (mark_cached requires a live
    // row reference, which the caller's row holds for this block).
    allocator.mark_cached(block_id);
    entries_.emplace(h, e);
    push_front(e);
    ++inserted;
    ++stats_.inserts;
  }
  return inserted;
}

int PrefixCache::evict(int n, BlockAllocator& allocator) {
  if (n <= 0) return 0;
  return evict_lru(n, allocator);
}

int PrefixCache::evict_lru(int n, BlockAllocator& allocator) {
  int evicted = 0;
  int scan_budget = kEvictScanCap;
  Entry* e = lru_tail_;
  while (e != nullptr && evicted < n && scan_budget-- > 0) {
    Entry* prev = e->prev;  // unlink() rewrites the list around e
    // Only blocks no sequence references may be released: dropping the pin on
    // a block a row still reads would let the pool hand its KV to a writer.
    // Entries still in use keep their LRU position and are skipped.
    if (allocator.ref_count(e->block_id) == 0) {
      unlink(e);
      entries_.erase(e->hash);
      allocator.evict_cached(e->block_id);
      delete e;
      ++evicted;
      ++stats_.evictions;
    }
    e = prev;
  }
  return evicted;
}

}  // namespace scheduler
