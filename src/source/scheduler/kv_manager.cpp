#include "scheduler/kv_manager.h"

#include <cctype>
#include <cstdlib>
#include <string>

namespace scheduler {

int kv_quant_from_env() {
  const char* raw = std::getenv("LLAMA_KV_QUANT");
  if (raw == nullptr || raw[0] == '\0') return kKvQuantNone;
  // Case-insensitive: the value is typed by hand on a command line.
  std::string v(raw);
  for (char& c : v) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
  // "1" is the switch itself, identical to the Scheduler constructor argument
  // and the benchmarks' --kv-quant, so the two entry points cannot be read
  // against each other. "5" is accepted as the same thing because that is what
  // the arm used to be called while the mask existed and what every reproduce
  // command in results/ says; both spellings select one arm, so accepting both
  // cannot be ambiguous. Every other digit is a LOG(FATAL) rather than an alias,
  // because 2/3/4 named precisions this build no longer has and quietly handing
  // back a different one produces a number nobody can attribute.
  if (v == "0" || v == "off" || v == "none" || v == "false") return kKvQuantNone;
  if (v == "1" || v == "kv4" || v == "k+v4" || v == "5") return kKvQuantK8V4;
  LOG(FATAL) << "LLAMA_KV_QUANT: unrecognized value '" << raw
             << "' (expected 0/off or 1/kv4 — K int8 + V int4; the arms 2 (V int8), "
                "3 (K+V int8) and 4 (V int4 over a bf16 K) were removed, and 1 used to "
                "mean \"K only\" while the mask existed)";
  return kKvQuantNone;
}

KVManager::KVManager(int num_layers, int max_batch, int max_seq_len, int kv_dim,
                     int kv_head_num, std::shared_ptr<base::DeviceAllocator> alloc,
                     base::DeviceType device, int block_size, int kv_quant)
    : num_layers_(num_layers),
      max_batch_(max_batch),
      max_seq_len_(max_seq_len),
      kv_dim_(kv_dim),
      kv_head_num_(kv_head_num) {
#ifdef USE_PAGED_ATTENTION
  paged_ = (device == base::DeviceType::kDeviceCUDA);
#else
  paged_ = false;
#endif

  // Quantization requires the paged layout (see the header). Reject rather
  // than downgrade: a caller that asked for quantization and silently got bf16
  // would measure no speedup, no PPL change, and no error.
  CHECK(kv_quant == kKvQuantNone || paged_)
      << "KV quantization (LLAMA_KV_QUANT) is only implemented for the paged CUDA layout "
         "(USE_PAGED_ATTENTION + device=kDeviceCUDA); requested mask="
      << kv_quant << ", paged=" << paged_;
  // Mask validation: exactly two values, and the removed ones are an error
  // rather than an alias for 5. A driver or a script that still passes 1/2/3/4
  // is asking for a precision this build does not have; resolving it to "K int8
  // + V int4" would produce a PPL number attributed to an arm that never ran.
  CHECK(kv_quant == kKvQuantNone || kv_quant == kKvQuantK8V4)
      << "Invalid KV quantization mask " << kv_quant
      << " (expected 0 for off or 1 for K int8 + V int4; the arms 2 (V int8), 3 (K+V int8) "
         "and 4 (V int4 over a bf16 K) were removed)";
  kv_quant_ = kv_quant;

  // The 4-bit V residual window is gone. Every script and reproduce command in
  // results/ exported LLAMA_KV_WINDOW, and a run that still sets it would
  // silently measure the windowless regime while its own logs claim otherwise,
  // so it is a hard error rather than an ignored variable.
  const char* window_env = std::getenv("LLAMA_KV_WINDOW");
  CHECK(window_env == nullptr)
      << "LLAMA_KV_WINDOW=" << window_env
      << " is no longer supported: the 4-bit V residual window was removed, and the KV cache "
         "is now the quantized pool alone (kv_quant 5 stores K int8 and V int4). Unset it — "
         "any value here, including 0, is a leftover.";

  if (paged_) {
    // Paged layout: [num_layers, num_blocks, block_size, kv_dim]. One block
    // pool serves all layers (block i of layer l is key_cache[l][i]).
    // Pool capacity matches the continuous layout within one block per seq:
    //   num_blocks = max_batch * ceil(max_seq_len / block_size)
    //
    // Validate the block size here, where the value and its origin (Scheduler
    // argument or LLAMA_BLOCK_SIZE) are still attributable: the kernels assume
    // a power-of-two page that fits a warp's 4-position group
    // (paged_decode_geometry_ok requires block_size >= 4) and the pool caps a
    // page at 256 tokens. Anything else would be rejected by the kernels'
    // geometry CHECK or mis-address the pool.
    CHECK(block_size >= 4 && (block_size & (block_size - 1)) == 0 && 256 % block_size == 0)
        << "Invalid paged KV block_size=" << block_size
        << " (must be a power of two >= 4 dividing 256: 8, 16, 32, ...); "
        << "set by the Scheduler block_size argument or LLAMA_BLOCK_SIZE";
    block_size_ = block_size;
    max_blocks_per_seq_ = (max_seq_len_ + block_size_ - 1) / block_size_;
    num_blocks_ = max_batch_ * max_blocks_per_seq_;
    block_allocator_ = std::make_unique<BlockAllocator>(num_blocks_, max_batch_,
                                                        max_blocks_per_seq_, block_size_);
  } else {
    // Continuous layout: [num_layers, max_batch, kv_dim, max_seq_len].
    block_size_ = max_seq_len_;
    max_blocks_per_seq_ = 1;
    slot_busy_.assign(max_batch_, false);
#ifdef USE_PAGED_ATTENTION
    // CPU device (or paging disabled): degenerate slot-mode pool so the
    // scheduler's block-table code path stays uniform — one block == one
    // whole slot, block table width 1.
    block_allocator_ = std::make_unique<BlockAllocator>(max_batch_, max_batch_, 1, max_seq_len_);
#endif
  }

  // All CUDA kernels (scatter / attention / decode) address the cache as
  // raw bf16, so allocate BF16 on CUDA and keep FP32 only on the CPU path
  // (whose FP32 fallback kernels read float pointers). This halves KV
  // memory on device.
  const base::DataType kv_dtype =
      (device == base::DeviceType::kDeviceCUDA) ? base::DataType::kDataTypeBF16
                                                : base::DataType::kDataTypeFp32;

  if (paged_) {
    // Element (layer, block, pos_in_block, d) at
    //   layer * (num_blocks * block_size * row_bytes)
    // + block * (block_size * row_bytes)
    // + pos_in_block * row_bytes + d
    // d-innermost within a page: the flash-decoding kernels walk consecutive
    // positions contiguously, crossing page boundaries through block_table.
    // Quantized, the same shape in int8 (half the bytes) — the addressing
    // arithmetic is unchanged, only the element size and the load width are.
    //
    // The V pool is the same int8 dtype with row_bytes = kv_dim/2: dim d lives
    // in byte d/2, low nibble for even d. The last dim is the discriminator the
    // kernels read the width from, so it is set here and nowhere else.
    const base::DataType cached_dtype =
        quantized() ? base::DataType::kDataTypeInt8 : kv_dtype;
    const int value_row_bytes = quantized() ? kv_dim_ / 2 : kv_dim_;
    if (quantized()) {
      CHECK(kv_dim_ % 2 == 0)
          << "4-bit V needs an even kv_dim (two values per byte), got " << kv_dim_;
    }
    key_cache_ = tensor::Tensor(cached_dtype, num_layers_, num_blocks_, block_size_, kv_dim_,
                                true, alloc);
    value_cache_ = tensor::Tensor(cached_dtype, num_layers_, num_blocks_, block_size_,
                                  value_row_bytes, true, alloc);

    // Scale pools, one fp32 per (layer, block, kv_head, pos_in_block). Only
    // allocated when quantized; the addressing (and why pos is the innermost
    // dim) is documented on the members in the header.
    if (quantized()) {
      key_scale_cache_ = tensor::Tensor(base::DataType::kDataTypeFp32, num_layers_, num_blocks_,
                                        kv_head_num_, block_size_, true, alloc);
      value_scale_cache_ = tensor::Tensor(base::DataType::kDataTypeFp32, num_layers_,
                                          num_blocks_, kv_head_num_, block_size_, true, alloc);
    }
  } else {
    // Head-dim-contiguous layout: cache[layer][slot][d][pos], position
    // innermost so decode MHA reads coalesce across consecutive positions.
    key_cache_ = tensor::Tensor(kv_dtype, num_layers_, max_batch_, kv_dim_, max_seq_len_,
                                true, alloc);
    value_cache_ = tensor::Tensor(kv_dtype, num_layers_, max_batch_, kv_dim_, max_seq_len_,
                                  true, alloc);
  }

  // The KV pool is by far the largest allocation in the process and it is
  // requested here, once, up front. If the device is out of memory the
  // allocator returns an empty tensor; without this check that surfaces much
  // later as a CUDA illegal address (or a null-pointer memcpy on CPU), with
  // nothing pointing at the pool size or the knobs that control it. Fail at
  // construction instead, with the numbers needed to size it down.
  const bool scales_ok =
      (!quantized() || (!key_scale_cache_.is_empty() && !value_scale_cache_.is_empty()));
  if (key_cache_.is_empty() || value_cache_.is_empty() || !scales_ok) {
    const double pool_gb =
        static_cast<double>(pool_bytes()) / (1024.0 * 1024.0 * 1024.0);
    LOG(FATAL) << "KV pool allocation failed: tried to allocate " << pool_gb << " GB ("
               << (paged_ ? "paged" : "continuous") << " layout, layers=" << num_layers_
               << " max_batch=" << max_batch_ << " max_seq_len=" << max_seq_len_
               << " kv_dim=" << kv_dim_
               << (paged_ ? " block_size=" + std::to_string(block_size_) : std::string())
               << (quantized() ? " kv_quant=" + std::to_string(kv_quant_) +
                                         " (K int8 + V int4, V row packed to kv_dim/2 bytes, "
                                         "plus the two scale pools)"
                               : std::string())
               << "). Lower max_batch / max_seq_len (or the driver's --gpu-mem-fraction) "
                  "and retry.";
  }
}

long long KVManager::pool_bytes() const {
  // Derived from the dims, so a failed (empty) allocation still reports the
  // size that was attempted — see the constructor's failure message. The scale
  // pools are included: they are real cache memory (~1.5% of the int8 pool at
  // block_size=16, and a driver sizing the pool from the KV dtype alone would
  // be wrong by exactly that much).
  return static_cast<long long>(key_cache_.byte_size()) +
         static_cast<long long>(value_cache_.byte_size()) +
         static_cast<long long>(key_scale_cache_.byte_size()) +
         static_cast<long long>(value_scale_cache_.byte_size());
}

int KVManager::allocate() {
#ifdef USE_PAGED_ATTENTION
  return block_allocator_->allocate_blocks(max_blocks_per_seq_);
#else
  for (int i = 0; i < max_batch_; ++i) {
    if (!slot_busy_[i]) {
      slot_busy_[i] = true;
      return i;
    }
  }
  return -1;
#endif
}

void KVManager::deallocate(int slot_or_row) {
#ifdef USE_PAGED_ATTENTION
  block_allocator_->free_all(slot_or_row);
#else
  if (slot_or_row >= 0 && slot_or_row < max_batch_) {
    slot_busy_[slot_or_row] = false;
  }
#endif
}

bool KVManager::has_free_slot() const {
  return free_slot_count() > 0;
}

int KVManager::busy_slot_count() const {
#ifdef USE_PAGED_ATTENTION
  // Rows owned by a sequence — NOT "rows with at least one block": a fully
  // prefix-shared prompt owns an empty (or shared-only) table but is still in
  // flight, and counting it would understate the KV cache in use.
  return block_allocator_->num_busy_rows();
#else
  int count = 0;
  for (bool busy : slot_busy_) {
    if (busy) ++count;
  }
  return count;
#endif
}

int KVManager::free_slot_count() const {
#ifdef USE_PAGED_ATTENTION
  // Rows, not blocks. One sequence owns one block-table row for its whole
  // lifetime, so the pool accepts exactly (num_rows - busy_rows) more
  // sequences — that is what "free slots" means in the continuous layout this
  // mirrors, and what an admission check needs. The old free_blocks /
  // max_blocks_per_seq counted full-window sequences' worth of blocks instead:
  // it read 0 with rows still free (blocks held by running rows or pinned by
  // the prefix cache), and > 0 with every row taken by sequences that had used
  // little of their capacity. Block pressure is a different question, and one
  // the allocator answers per allocation (append_blocks reclaims before it
  // fails), not from a precomputed count.
  return block_allocator_->num_rows() - block_allocator_->num_busy_rows();
#else
  int count = 0;
  for (bool busy : slot_busy_) {
    if (!busy) ++count;
  }
  return count;
#endif
}

}  // namespace scheduler
