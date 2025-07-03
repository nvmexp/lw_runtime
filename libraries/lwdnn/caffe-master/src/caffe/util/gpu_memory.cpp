#include <algorithm>
#include <sstream>
#include "caffe/common.hpp"
#include "caffe/util/gpu_memory.hpp"

#include "lwb/util_allocator.lwh"

namespace caffe {
using std::vector;

const int GPUMemory::ILWALID_DEVICE = lwb::CachingDeviceAllocator::ILWALID_DEVICE_ORDINAL;
const unsigned int GPUMemory::Manager::BIN_GROWTH = 2;
const unsigned int GPUMemory::Manager::MIN_BIN = 6;
const unsigned int GPUMemory::Manager::MAX_BIN = 22;
const size_t GPUMemory::Manager::MAX_CACHED_BYTES = (size_t) -1;
const size_t GPUMemory::Manager::MAX_CACHED_SIZE = (1 << GPUMemory::Manager::MAX_BIN);  // 4M
std::mutex GPUMemory::dev_info_mutex_;

GPUMemory::Manager GPUMemory::mgr_;

// If there is a room to grow it tries
// It keeps what it has otherwise
bool GPUMemory::Workspace::safe_reserve(size_t size, int device) {
  if (size <= size_) {
    return false;
  }
  size_t gpu_bytes_left, total_memory;
  GPUMemory::GetInfo(&gpu_bytes_left, &total_memory, true);
  if (size > size_ + align_down<8>(gpu_bytes_left)) {
    LOG(FATAL) << "Out of memory in safe_reserve: "
        << size << " > " << size_ << " + " << align_down<8>(gpu_bytes_left)
        << " on device " << device;
    return false;
  }
  release();
  reserve(size, device);  // might fail here
  return true;
}

bool GPUMemory::Workspace::try_reserve(size_t size, int device) {
  bool status = true;
  if (size > 0UL && (size > size_ || ptr_ == nullptr)) {
    release();
    if (device != ILWALID_DEVICE) {
      device_ = device;  // switch from default to specific one
    }
    pstream_ = Caffe::thread_pstream(0);
    status = mgr_.try_allocate(&ptr_, size, device_, pstream_);
    if (status) {
      CHECK_NOTNULL(ptr_);
      size_ = size;
    }
  }
  return status;
}

GPUMemory::Manager::Manager() : debug_(false), initialized_(false) {
  const int count = Caffe::device_count();
  dev_info_.resize(count);
  update_thresholds_.resize(count);
}


void GPUMemory::Manager::init(const vector<int>& gpus, bool debug) {
  if (initialized_) {
    return;
  }
  bool debug_elw = getelw("DEBUG_GPU_MEM") != 0;
  debug_ = debug || debug_elw;
  try {
    // Just in case someone installed 'no cleanup' arena before
    lwb_allocator_.reset(new lwb::CachingDeviceAllocator(BIN_GROWTH, MIN_BIN, MAX_BIN,
        MAX_CACHED_BYTES, true, debug_));
  } catch (...) {
  }
  CHECK(lwb_allocator_);
  for (int i = 0; i < gpus.size(); ++i) {
    update_dev_info(gpus[i]);
    update_thresholds_[gpus[i]] = dev_info_[gpus[i]].total_ * 9UL / 10UL;
  }
  initialized_ = true;
  LOG(INFO) << "GPUMemory::Manager initialized";
  for (int i = 0; i < gpus.size(); ++i) {
    LOG(INFO) << report_dev_info(gpus[i]);
  }
}

void GPUMemory::Manager::reset() {
  if (!initialized_) {
    return;
  }
  lwb_allocator_.reset();
  initialized_ = false;
}

GPUMemory::Manager::~Manager() {
}

void GPUMemory::Manager::lazy_init(int device) {
  if (initialized_) {
    return;
  }
  if (device < 0) {
    LWDA_CHECK(lwdaGetDevice(&device));
  }
  LOG(WARNING) << "Lazily initializing GPU Memory Manager Scope on device " << device
               << ". Note: it's recommended to do this explicitly in your "
                   "main() function.";
  vector<int> gpus(1, device);
  static Scope gpu_memory_scope(gpus);
}

bool GPUMemory::Manager::try_allocate(void** ptr, size_t size, int device,
                                      const shared_ptr<LwdaStream>& pstream) {
  if (!initialized_) {
    lazy_init(device);
  }
  CHECK_NOTNULL(ptr);
  CHECK_EQ(Caffe::lwrrent_device(), device);
  CHECK_EQ(Caffe::device(), device);
  lwdaError_t status = lwdaSuccess, last_err = lwdaSuccess;
  {
    size_t size_allocated = 0;
    // Clean Cache & Retry logic is inside now
    status = lwb_allocator_->DeviceAllocate(device, ptr, size, pstream->get(), size_allocated);
    if (status == lwdaSuccess && device > ILWALID_DEVICE) {
//      if (device == 0) {
//        DevInfo dev_info;
//        LWDA_CHECK(lwdaMemGetInfo(&dev_info.free_, &dev_info.total_));
//        size_t allocated = dev_info.total_ - dev_info.free_;
//        size_t pcent = 100UL* allocated / dev_info.total_;
//        std::string bar(pcent, '*');
//        std::cout << bar << " " << pcent << "%" << std::endl;
//      }
      if (size_allocated > 0) {
        if (dev_info_[device].free_ < update_thresholds_[device]) {
          update_dev_info(device);
          update_thresholds_[device] =
              update_thresholds_[device] * 9UL / 10UL;  // every 10% decrease
        } else if (dev_info_[device].free_ < size_allocated) {
          update_dev_info(device);
        } else {
          dev_info_[device].free_ -= size_allocated;
        }
      }
    }
  }
  // If there was a retry and it succeeded we get good status here but
  // we need to clean up last error...
  last_err = lwdaGetLastError();
  // ...and update the dev info if something was wrong
  if (status != lwdaSuccess || last_err != lwdaSuccess) {
    // If we know what particular device failed we update its info only
    if (device > ILWALID_DEVICE && device < dev_info_.size()) {
      // only query devices that were initialized
      if (dev_info_[device].total_) {
        update_dev_info(device);
        dev_info_[device].flush_count_++;
        DLOG(INFO) << "Updated info for device " << device << ": " << report_dev_info(device);
      }
    } else {
      // Update them all otherwise
      int lwr_device;
      LWDA_CHECK(lwdaGetDevice(&lwr_device));
      // Refresh per-device saved values.
      for (int i = 0; i < dev_info_.size(); ++i) {
        // only query devices that were initialized
        if (dev_info_[i].total_) {
          update_dev_info(i);
          // record which device caused cache flush
          if (i == lwr_device) {
            dev_info_[i].flush_count_++;
          }
          DLOG(INFO) << "Updated info for device " << i << ": " << report_dev_info(i);
        }
      }
    }
  }
  return status == lwdaSuccess;
}

void GPUMemory::Manager::deallocate(void* ptr, int device) {
  // allow for null pointer deallocation
  if (ptr == nullptr || lwb_allocator_ == nullptr) {
    return;
  }
  int lwrrent_device;  // Just to check LWCA status:
  lwdaError_t status = lwdaGetDevice(&lwrrent_device);
  // Preventing dead lock while Caffe shutting down.
  if (status != lwdaErrorLwdartUnloading) {
    size_t size_deallocated = 0;
    status = lwb_allocator_->DeviceFree(device, ptr, size_deallocated);
    if (status == lwdaSuccess && size_deallocated > 0) {
      dev_info_[device].free_ += size_deallocated;
    }
  }
}

void GPUMemory::Manager::update_dev_info(int device) {
  std::lock_guard<std::mutex> lock(dev_info_mutex_);
  if (device + 1 > dev_info_.size()) {
    dev_info_.resize(device + 1);
  }
  lwdaDeviceProp props;
  LWDA_CHECK(lwdaGetDeviceProperties(&props, device));
  LWDA_CHECK(lwdaMemGetInfo(&dev_info_[device].free_, &dev_info_[device].total_));
  // Make sure we don't have more than total device memory.
  dev_info_[device].total_ = std::min(props.totalGlobalMem, dev_info_[device].total_);
  dev_info_[device].free_ = std::min(dev_info_[device].total_, dev_info_[device].free_);
}

std::string GPUMemory::Manager::report_dev_info(int device) {
  lwdaDeviceProp props;
  LWDA_CHECK(lwdaGetDeviceProperties(&props, device));
  DevInfo dev_info;
  LWDA_CHECK(lwdaMemGetInfo(&dev_info.free_, &dev_info.total_));
  std::ostringstream os;
  os << "Total memory: " << props.totalGlobalMem << ", Free: " << dev_info.free_ << ", dev_info["
     << device << "]: total=" << dev_info_[device].total_ << " free=" << dev_info_[device].free_;
  return os.str();
}

void GPUMemory::Manager::GetInfo(size_t* free_mem, size_t* total_mem, bool with_update) {
  CHECK(lwb_allocator_) <<
      "Forgot to add 'caffe::GPUMemory::Scope gpu_memory_scope(gpus);' to your main()?";
  int lwr_device;
  LWDA_CHECK(lwdaGetDevice(&lwr_device));
  if (with_update) {
    update_dev_info(lwr_device);
  }
  *total_mem = dev_info_[lwr_device].total_;
  // Free memory is free GPU memory plus free cached memory in the pool.
  *free_mem = dev_info_[lwr_device].free_ + lwb_allocator_->cached_bytes[lwr_device].free;
  if (*free_mem > *total_mem) {  // sanity check
    *free_mem = *total_mem;
  }
}

GPUMemory::PinnedBuffer::PinnedBuffer(size_t size) {
  CHECK_GT(size, 0);
  LWDA_CHECK(lwdaHostAlloc(&hptr_, size, lwdaHostAllocMapped));
  LWDA_CHECK(lwdaHostGetDevicePointer(&dptr_, hptr_, 0));
}

GPUMemory::PinnedBuffer::~PinnedBuffer() {
  LWDA_CHECK(lwdaFreeHost(hptr_));
}

}  // namespace caffe
