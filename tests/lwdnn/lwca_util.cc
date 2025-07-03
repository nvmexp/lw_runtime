/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "lwda_util.h"
#include "lwca/include/lwda_runtime.h"

#include <atomic>

namespace lwidia_libs_test {

Status GetStatus(LWresult result) {
  if (result == LWDA_SUCCESS) {
    return OkStatus();
  }
  const char* str = nullptr;
  CHECK(lwGetErrorString(result, &str) == LWDA_SUCCESS);
  return ErrorStatus("LWCA Driver API error '") << str << "'";
}

Status GetStatus(lwdaError_t error) {
  if (error == lwdaSuccess) {
    return OkStatus();
  }
  // Reset LWCA runtime status because we can expect the user to handle the
  // returned error.
  lwdaGetLastError();
  const char* str = lwdaGetErrorString(error);
  return ErrorStatus("LWCA Runtime API error '") << str << "'";
}

Status GetStatus(LWptiResult result) {
  if (result == LWPTI_SUCCESS) {
    return OkStatus();
  }
  const char* str = nullptr;
  CHECK(lwptiGetResultString(result, &str) == LWPTI_SUCCESS);
  return ErrorStatus("LWPTI error '") << str << "'";
}

RandomGenerator::RandomGenerator(size_t seed)
    : state_(std::move(
          AllocateDeviceMemory(detail::GetLwrandStateSize()).ValueOrDie())) {
  detail::InitializeLwrandState(state_.get(), seed);
}

namespace detail {
void HostMemoryDeleter::operator()(void* ptr) const {
  CHECK_OK_STATUS(GetStatus(lwdaFreeHost(ptr)));
}
}  // namespace detail

DeviceMemory::DeviceMemory(std::nullptr_t) : ptr_(nullptr), size_(0) {}

DeviceMemory::DeviceMemory(DeviceMemory&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
  other.ptr_ = nullptr;
  other.size_ = 0;
}

DeviceMemory& DeviceMemory::operator=(DeviceMemory&& other) {
  if (this != &other) {
    CHECK_EQ(ptr_ == nullptr || ptr_ != other.ptr_, true);
    CHECK_OK_STATUS(GetStatus(lwdaFree(ptr_)));
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

namespace {
std::atomic<std::size_t> allocated_device_memory_bytes{0};
}  // namespace

DeviceMemory::~DeviceMemory() {
  CHECK_GE(allocated_device_memory_bytes, size_);
  allocated_device_memory_bytes -= size_;
  CHECK_OK_STATUS(GetStatus(lwdaFree(ptr_)));
}

StatusOr<HostMemory> AllocateHostMemory(size_t size) {
  void* result;
  RETURN_IF_ERROR_STATUS(GetStatus(lwdaMallocHost(&result, size)));
  return HostMemory(result);
}

void* GetDevicePointer(const HostMemory& host_ptr) {
  void* dev_ptr;
  CHECK_OK_STATUS(
      GetStatus(lwdaHostGetDevicePointer(&dev_ptr, host_ptr.get(), 0)));
  return dev_ptr;
}

StatusOr<DeviceMemory> AllocateDeviceMemory(size_t size) {
  DeviceMemory result(nullptr);
  lwdaError_t error = lwdaMalloc(&result.ptr_, size);
  auto status = GetStatus(error);
  if (error == lwdaErrorMemoryAllocation) {
    size_t free = 0;
    size_t total = 0;
    CHECK_OK_STATUS(GetStatus(lwdaMemGetInfo(&free, &total)));
    status << "\nbytes requested: " << size
           << "\nbytes allocated: " << allocated_device_memory_bytes
           << "\nbytes free: " << free << "\nbytes total: " << total;
  }
  RETURN_IF_ERROR_STATUS(status);
  result.size_ = size;
  allocated_device_memory_bytes += size;
  return std::move(result);
}

void FillWithNaNs(const DeviceMemory& mem) {
  CHECK_OK_STATUS(GetStatus(lwdaMemset(mem.get(), 0xff, mem.size())));
}

size_t GetAllocatedDeviceMemoryBytes() { return allocated_device_memory_bytes; }

Status CopyDeviceMemory(const DeviceMemory& dst, const DeviceMemory& src,
                        size_t size) {
  RETURN_IF_ERROR_STATUS(GetStatus(
      lwdaMemcpy(dst.get(), src.get(), size, lwdaMemcpyDeviceToDevice)));
  return OkStatus();
}

void ResetDevice() {
  lwdaGetLastError();  // Reset LWCA runtime status.
  CHECK_OK_STATUS(GetStatus(lwdaDeviceReset()));
}

namespace {
lwdaDeviceProp GetDeviceProperties() {
  int device = 0;
  CHECK_OK_STATUS(GetStatus(lwdaGetDevice(&device)));
  lwdaDeviceProp props;
  CHECK_OK_STATUS(GetStatus(lwdaGetDeviceProperties(&props, device)));
  return props;
}
}  // namespace

bool DeviceHasAtLeastComputeCapability(int major, int minor) {
  static lwdaDeviceProp props = GetDeviceProperties();
  return props.major > major || (props.major == major && props.minor >= minor);
}

bool DeviceSupportsReducedPrecision() {
  return DeviceHasAtLeastComputeCapability(5, 3);
}

bool DeviceSupportsTensorOpMath() {
  return DeviceHasAtLeastComputeCapability(7, 0);
}
}  // namespace lwidia_libs_test
