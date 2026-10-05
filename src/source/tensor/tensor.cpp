#include "tensor/tensor.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <numeric>
#include <stdexcept>

namespace tensor {
template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init) {
  // NOTE: Tp must be 64-bit; dims_ stores int32 and element counts can
  // exceed 2^31 (e.g. the paged KV cache at max_batch >= 128).
  if (begin >= end) {
    return 0;
  }
  size_t size = std::accumulate(begin, end, init, std::multiplies<>());
  return size;
}

static size_t data_type_size(base::DataType data_type) {
  switch (data_type) {
    case base::DataType::kDataTypeFp32: {
      return 4;
    }
    case base::DataType::kDataTypeInt8: {
      return 1;
    }
    case base::DataType::kDataTypeInt32: {
      return 4;
    }
    case base::DataType::kDataTypeBF16: {
      return 2;
    }
    default: {
      LOG(FATAL) << "Unknown data type size for " << int(data_type);
      return 0;
    }
  }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  size_ = dim0;
  if (need_alloc && alloc) {
    allocate_or_throw(alloc);
  } else {
    if (ptr != nullptr) {
      CHECK(need_alloc == false)
          << "The need_alloc is is true when ptr parameter is not a null pointer.";
      init_buffer(alloc, data_type_, need_alloc, ptr);
    }
  }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = static_cast<int64_t>(dim0) * dim1;
  if (need_alloc && alloc) {
    allocate_or_throw(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = static_cast<int64_t>(dim0) * dim1 * dim2;
  if (need_alloc && alloc) {
    allocate_or_throw(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = static_cast<int64_t>(dim0) * dim1 * dim2 * dim3;
  if (need_alloc && alloc) {
    allocate_or_throw(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : dims_(std::move(dims)), data_type_(data_type) {
  size_ = reduce_dimension(dims_.begin(), dims_.end(), 1LL);
  if (need_alloc && alloc) {
    allocate_or_throw(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

void Tensor::to_cuda(cudaStream_t stream) {
  CHECK_NE(buffer_, nullptr) << "to_cuda: the tensor has no buffer.";
  const base::DeviceType device_type = this->device_type();
  if (device_type == base::DeviceType::kDeviceUnknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
    throw std::runtime_error("Tensor::to_cuda: unknown device type");
  } else if (device_type == base::DeviceType::kDeviceCPU) {
    if (buffer_->ptr() == nullptr) {
      LOG(ERROR) << "Tensor::to_cuda: the CPU source buffer has a null data pointer.";
      throw std::runtime_error("Tensor::to_cuda: null source pointer");
    }
    size_t byte_size = this->byte_size();
    auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    auto cu_buffer = std::make_shared<base::Buffer>(byte_size, cu_alloc);
    if (cu_buffer->ptr() == nullptr) {
      LOG(ERROR) << "Tensor::to_cuda: failed to allocate " << byte_size
                 << " bytes on the CUDA device.";
      throw std::runtime_error("Tensor::to_cuda: CUDA allocation failed");
    }
    cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size, base::MemcpyKind::kMemcpyCPU2CUDA,
                     stream);
    this->buffer_ = cu_buffer;
  } else {
#ifndef NDEBUG
    VLOG(1) << "The device type of the tensor is already cuda.";
#endif
  }
}

void Tensor::to_cpu() {
  CHECK_NE(buffer_, nullptr);
  const base::DeviceType device_type = this->device_type();

  if (device_type == base::DeviceType::kDeviceUnknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
    throw std::runtime_error("Tensor::to_cpu: unknown device type");
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    if (buffer_->ptr() == nullptr) {
      LOG(ERROR) << "Tensor::to_cpu: the CUDA source buffer has a null data pointer.";
      throw std::runtime_error("Tensor::to_cpu: null source pointer");
    }
    size_t byte_size = this->byte_size();
    auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
    auto cpu_buffer = std::make_shared<base::Buffer>(byte_size, cpu_alloc);
    if (cpu_buffer->ptr() == nullptr) {
      LOG(ERROR) << "Tensor::to_cpu: failed to allocate " << byte_size << " bytes on the host.";
      throw std::runtime_error("Tensor::to_cpu: host allocation failed");
    }
    cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size,
                      base::MemcpyKind::kMemcpyCUDA2CPU);
    this->buffer_ = cpu_buffer;
  } else {
#ifndef NDEBUG
    VLOG(1) << "The device type of the tensor is already cpu.";
#endif
  }
}

size_t Tensor::size() const { return this->size_; }

int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}

base::DeviceType Tensor::device_type() const {
  if (!buffer_) {
    return base::DeviceType::kDeviceUnknown;
  }
  return buffer_->device_type();
}

bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
  if (!buffer) {
    LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
    return false;
  }
  if (buffer_ && buffer_->device_type() != base::DeviceType::kDeviceUnknown &&
      buffer->device_type() != base::DeviceType::kDeviceUnknown &&
      buffer_->device_type() != buffer->device_type()) {
    // Continuing here would leave the tensor tagged with a device its data is
    // not on: every later kernel dispatch would read the wrong memory. An
    // unknown device on either side means "not tagged yet", which is safe.
    LOG(ERROR) << "The device type of the new buffer ("
               << static_cast<int>(buffer->device_type())
               << ") is different from the original one ("
               << static_cast<int>(buffer_->device_type()) << ").";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (byte_size > buffer->byte_size()) {
    LOG(ERROR) << "The size of buffer is too small for the tensor!";
    return false;
  }
  buffer_ = buffer;
  return true;
}

bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
  if (!allocator) {
    LOG(ERROR) << "The allocator parameter in the allocate function is null "
                  "pointer!";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (!byte_size) {
    LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
    return false;
  }

  if (buffer_ && buffer_->ptr() && byte_size <= buffer_->byte_size()) {
    if (!need_realloc) {
      return true;
    }
  }

  auto new_buffer = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
  if (!new_buffer->ptr()) {
    // Drop the buffer instead of keeping a null-backed one: is_empty() would
    // report the tensor as empty, yet a later allocate() would see buffer_ &&
    // byte_size <= byte_size and "succeed" without ever allocating.
    LOG(ERROR) << "The memory allocated is a null pointer! (byte_size=" << byte_size
               << ", device=" << int(allocator->device_type()) << ")";
    buffer_ = nullptr;
    return false;
  }
  buffer_ = new_buffer;
  return true;
}

void Tensor::allocate_or_throw(const std::shared_ptr<base::DeviceAllocator>& allocator) {
  if (!allocate(allocator, /*need_realloc=*/true)) {
    throw std::runtime_error("Tensor allocation failed: " + std::to_string(byte_size()) +
                             " bytes, dtype=" + std::to_string(static_cast<int>(data_type_)));
  }
}

const std::vector<int32_t>& Tensor::dims() const { return this->dims_; }

void Tensor::set_device_type(base::DeviceType device_type) const {
  // Retagging mutates the Buffer, which other tensors may share: a copy of
  // this tensor, a buffer handed out by assign(), a view. Silently changing
  // the device type they observe would make them dispatch kernels against the
  // wrong device. Only a buffer this tensor owns exclusively may be retagged;
  // shared views must tag the Buffer directly at creation time.
  CHECK(buffer_ != nullptr) << "set_device_type: the tensor has no buffer to tag.";
  CHECK_EQ(buffer_.use_count(), 1)
      << "set_device_type: the buffer is shared with another tensor; tag the Buffer "
         "directly instead of retagging it through one of its views.";
  buffer_->set_device_type(device_type);
}

void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) {
  this->data_type_ = data_type;
  this->dims_ = dims;
  this->size_ = reduce_dimension(dims.begin(), dims.end(), 1LL);
  this->buffer_ = nullptr;
}

int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

base::DataType Tensor::data_type() const { return data_type_; }

void Tensor::reshape(const std::vector<int32_t>& dims) {
  size_t size = reduce_dimension(dims.begin(), dims.end(), 1LL);
  if (buffer_) {
    // Reshape only changes the shape metadata; the storage stays as
    // allocated. It used to silently allocate and copy a new buffer whenever
    // the requested element count exceeded the *current* size, which replaced
    // the storage under every other view of it (and hid capacity bugs). The
    // only real limit is the buffer's capacity.
    const size_t elem_size = base::DataTypeSize(data_type_);
    CHECK_NE(elem_size, 0) << "reshape: the tensor has an unknown data type.";
    const size_t capacity = buffer_->byte_size() / elem_size;
    CHECK_LE(size, capacity) << "reshape would grow the tensor from " << size_ << " to " << size
                             << " elements, beyond its buffer capacity of " << capacity
                             << "; allocate a larger buffer first.";
  }
  this->dims_ = dims;
  this->size_ = size;
}

std::shared_ptr<base::Buffer> Tensor::get_buffer() const { return buffer_; }

Tensor Tensor::clone() const {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "clone: the tensor has no data to clone.";
  Tensor new_tensor = *this;
  size_t byte_size = this->byte_size();

  auto allocator = buffer_->allocator();
  CHECK(allocator != nullptr)
      << "clone: the source buffer has no allocator, cannot allocate a copy.";
  new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
  CHECK(new_tensor.buffer_->ptr() != nullptr)
      << "clone: failed to allocate " << byte_size << " bytes for the copy.";
  new_tensor.buffer_->copy_from(buffer_.get());
  return new_tensor;
}

size_t Tensor::byte_size() const { return this->size() * DataTypeSize(data_type_); }

std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides;
  if (!dims_.empty()) {
    for (int32_t i = 0; i < dims_.size() - 1; ++i) {
      size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1LL);
      strides.push_back(stride);
    }
    strides.push_back(1);
  }
  return strides;
}

bool Tensor::is_empty() const {
  return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}

void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                         bool need_alloc, void* ptr) {
  if (!alloc && !need_alloc) {
    std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
    this->buffer_ = buffer;
  } else {
    allocate_or_throw(alloc);
  }
}
}  // namespace tensor