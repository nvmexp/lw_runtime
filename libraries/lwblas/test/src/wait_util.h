#pragma once

#include <cstdint>

#include <lwda_runtime.h>

namespace wait_util
{

inline lwdaError_t
gpuAllocPinnedAndMap(size_t sizeInbytes, void **HostMemPtr, void **GpuMemPtr) {
    lwdaError_t err = lwdaHostAlloc(HostMemPtr, sizeInbytes, lwdaHostAllocMapped);
    if (err != lwdaSuccess) {
        return err;
    }
    return (lwdaHostGetDevicePointer(GpuMemPtr, *HostMemPtr, 0));
}

__global__ void
wait_kernel(volatile int32_t *counter, const int32_t threshold) {
    static const int64_t WAIT_CYCLES = 1024;
    while (*counter < threshold) {
        int64_t elapsed = 0, t0 = clock64();
        do {
            elapsed = clock64() - t0;
        } while (elapsed < WAIT_CYCLES);
    }
}

struct WaitKernel
{
    int32_t *wait_ctr_host_ptr = nullptr;
    int32_t *wait_ctr_device_ptr = nullptr;

    lwdaError_t init()
    {

        // prepare a semaphore for wait kernel to make sure events are aligned to actual kernels
        return gpuAllocPinnedAndMap(sizeof(*wait_ctr_host_ptr),
                             &reinterpret_cast<void *&>(wait_ctr_host_ptr),
                             &reinterpret_cast<void *&>(wait_ctr_device_ptr));
    }

    void reset()
    {
        *reinterpret_cast<volatile int32_t *>(wait_ctr_host_ptr) = 0;
    }

    void launch()
    {
        reset();
        wait_kernel<<<1, 1>>>(wait_ctr_device_ptr, 1);
    }

    void set()
    {
        *reinterpret_cast<volatile int32_t *>(wait_ctr_host_ptr) = 1;
    }

    WaitKernel() = default;
    WaitKernel(const WaitKernel& copy) = delete;
    WaitKernel& operator=(const WaitKernel& assign) = delete;
    ~WaitKernel()
    {
        if (wait_ctr_host_ptr)
            lwdaFreeHost(wait_ctr_host_ptr);
    }

};

}  // namespace wait_util
