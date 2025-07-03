/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <stdarg.h>
#include <sys/types.h>
#include <unistd.h>
#include <map>
#include <lwca.h>
#include "common.hpp"

#define ROUND_UP(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))

namespace gdrcopy {
    namespace test {
        bool print_dbg_msg = false;
        const char *testname = "";

        void print_dbg(const char* fmt, ...)
        {
            if (print_dbg_msg) {
                va_list ap;
                va_start(ap, fmt);
                vfprintf(stderr, fmt, ap);
            }
        }

        LWresult gpu_mem_alloc(gpu_mem_handle_t *handle, const size_t size, bool aligned_mapping, bool set_sync_memops)
        {
            LWresult ret = LWDA_SUCCESS;
            LWdeviceptr ptr, out_ptr;
            size_t allocated_size;

            if (aligned_mapping)
                allocated_size = size + GPU_PAGE_SIZE - 1;
            else
                allocated_size = size;

            ret = lwMemAlloc(&ptr, allocated_size);
            if (ret != LWDA_SUCCESS)
                return ret;

            if (set_sync_memops) {
                unsigned int flag = 1;
                ret = lwPointerSetAttribute(&flag, LW_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr);
                if (ret != LWDA_SUCCESS) {
                    lwMemFree(ptr);
                    return ret;
                }
            }

            if (aligned_mapping)
                out_ptr = ROUND_UP(ptr, GPU_PAGE_SIZE);
            else
                out_ptr = ptr;

            handle->ptr = out_ptr;
            handle->unaligned_ptr = ptr;
            handle->size = size;
            handle->allocated_size = allocated_size;

            return LWDA_SUCCESS;
        }

        LWresult gpu_mem_free(gpu_mem_handle_t *handle)
        {
            LWresult ret = LWDA_SUCCESS;
            LWdeviceptr ptr;

            ret = lwMemFree(handle->unaligned_ptr);
            if (ret == LWDA_SUCCESS)
                memset(handle, 0, sizeof(gpu_mem_handle_t));

            return ret;
        }

#if LWDA_VERSION >= 11000
        /**
         * Allocating GPU memory using VMM API.
         * VMM API is available since LWCA 10.2. However, the RDMA support is added in LWCA 11.0.
         * Our tests are not useful without RDMA support. So, we enable this VMM allocation from LWCA 11.0.
         */
        LWresult gpu_vmm_alloc(gpu_mem_handle_t *handle, const size_t size, bool aligned_mapping, bool set_sync_memops)
        {
            LWresult ret = LWDA_SUCCESS;

            size_t granularity, gran;
            LWmemAllocationProp mprop;
            LWdevice gpu_dev;
            size_t rounded_size;
            LWdeviceptr ptr = 0;
            LWmemGenericAllocationHandle mem_handle = 0;
            bool is_mapped = false;

            int RDMASupported = 0;

            int version;

            ret = lwDriverGetVersion(&version);
            if (ret != LWDA_SUCCESS) {
                print_dbg("error in lwDriverGetVersion\n");
                goto out;
            }

            if (version < 11000) {
                print_dbg("VMM with RDMA is not supported in this LWCA version.\n");
                ret = LWDA_ERROR_NOT_SUPPORTED;
                goto out;
            }

            ret = lwCtxGetDevice(&gpu_dev);
            if (ret != LWDA_SUCCESS) {
                print_dbg("error in lwCtxGetDevice\n");
                goto out;
            }

            ret = lwDeviceGetAttribute(&RDMASupported, LW_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_LWDA_VMM_SUPPORTED, gpu_dev);
            if (ret != LWDA_SUCCESS) {
                print_dbg("error in lwDeviceGetAttribute\n");
                goto out;
            }

            if (!RDMASupported) {
                print_dbg("GPUDirect RDMA is not supported on this GPU.\n");
                ret = LWDA_ERROR_NOT_SUPPORTED;
                goto out;
            }

            memset(&mprop, 0, sizeof(LWmemAllocationProp));
            mprop.type = LW_MEM_ALLOCATION_TYPE_PINNED;
            mprop.location.type = LW_MEM_LOCATION_TYPE_DEVICE;
            mprop.location.id = gpu_dev;
            mprop.allocFlags.gpuDirectRDMACapable = 1;

            ret = lwMemGetAllocationGranularity(&gran, &mprop, LW_MEM_ALLOC_GRANULARITY_RECOMMENDED);
            if (ret != LWDA_SUCCESS) {
                print_dbg("error in lwMemGetAllocationGranularity\n");
                goto out;
            }

            // In case gran is smaller than GPU_PAGE_SIZE
            granularity = ROUND_UP(gran, GPU_PAGE_SIZE);

            rounded_size = ROUND_UP(size, granularity);
            ret = lwMemAddressReserve(&ptr, rounded_size, granularity, 0, 0);
            if (ret != LWDA_SUCCESS) {
                print_dbg("error in lwMemAddressReserve\n");
                goto out;
            }

            ret = lwMemCreate(&mem_handle, rounded_size, &mprop, 0);
            if (ret != LWDA_SUCCESS) {
                print_dbg("error in lwMemCreate\n");
                goto out;
            }

            ret = lwMemMap(ptr, rounded_size, 0, mem_handle, 0);
            if (ret != LWDA_SUCCESS) {
                print_dbg("error in lwMemMap\n");
                goto out;
            }
            is_mapped = true;

            LWmemAccessDesc access;
            access.location.type = LW_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = gpu_dev;
            access.flags = LW_MEM_ACCESS_FLAGS_PROT_READWRITE;

            ret = lwMemSetAccess(ptr, rounded_size, &access, 1);
            if (ret != LWDA_SUCCESS) {
                print_dbg("error in lwMemSetAccess\n");
                goto out;
            }

            // lwMemAddressReserve always returns aligned ptr
            handle->ptr = ptr;
            handle->handle = mem_handle;
            handle->size = size;
            handle->allocated_size = rounded_size;

out:
            if (ret != LWDA_SUCCESS) {
                if (is_mapped)
                    lwMemUnmap(ptr, rounded_size);
                
                if (mem_handle)
                    lwMemRelease(mem_handle);
                
                if (ptr)
                    lwMemAddressFree(ptr, rounded_size);
            }
            return ret;
        }

        LWresult gpu_vmm_free(gpu_mem_handle_t *handle)
        {
            LWresult ret;

            if (!handle || !handle->ptr)
                return LWDA_ERROR_ILWALID_VALUE;

            ret = lwMemUnmap(handle->ptr, handle->allocated_size);
            if (ret != LWDA_SUCCESS) {
                print_dbg("error in lwMemUnmap\n");
                return ret;
            }

            ret = lwMemRelease(handle->handle);
            if (ret != LWDA_SUCCESS) {
                print_dbg("error in lwMemRelease\n");
                return ret;
            }

            ret = lwMemAddressFree(handle->ptr, handle->allocated_size);
            if (ret != LWDA_SUCCESS) {
                print_dbg("error in lwMemAddressFree\n");
                return ret;
            }

            memset(handle, 0, sizeof(gpu_mem_handle_t));

            return LWDA_SUCCESS;
        }
#else
        /* VMM with RDMA is not available before LWCA 11.0 */
        LWresult gpu_vmm_alloc(gpu_mem_handle_t *handle, const size_t size, bool aligned_mapping, bool set_sync_memops)
        {
            return LWDA_ERROR_NOT_SUPPORTED;
        }

        LWresult gpu_vmm_free(gpu_mem_handle_t *handle)
        {
            return LWDA_ERROR_NOT_SUPPORTED;
        }
#endif

        int compare_buf(uint32_t *ref_buf, uint32_t *buf, size_t size)
        {
            int diff = 0;
            if (size % 4 != 0U) {
                print_dbg("warning: buffer size %zu is not dword aligned, ignoring trailing bytes\n", size);
                size -= (size % 4);
            }
            unsigned ndwords = size/sizeof(uint32_t);
            for(unsigned  w = 0; w < ndwords; ++w) {
                if (ref_buf[w] != buf[w]) {
                    if (!diff) {
                        printf("%10.10s %8.8s %8.8s\n", "word", "content", "expected");
                    }
                    if (diff < 10) {
                        printf("%10d %08x %08x\n", w, buf[w], ref_buf[w]);
                    }
                    ++diff;
                }
            }
            if (diff) {
                print_dbg("check error: %d different dwords out of %d\n", diff, ndwords);
            }
            return diff;
        }

        void init_hbuf_walking_bit(uint32_t *h_buf, size_t size)
        {
            uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
            unsigned w;
            ASSERT_NEQ(h_buf, (void*)0);
            ASSERT_EQ(size % 4, 0U);
            //OUT << "filling mem with walking bit " << endl;
            for(w = 0; w<size/sizeof(uint32_t); ++w)
                h_buf[w] = base_value ^ (1<< (w%32));
        }

        void init_hbuf_linear_ramp(uint32_t *h_buf, size_t size)
        {
            uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
            unsigned w;
            ASSERT_NEQ(h_buf, (void*)0);
            ASSERT_EQ(size % 4, 0U);
            //OUT << "filling mem with walking bit " << endl;
            for(w = 0; w<size/sizeof(uint32_t); ++w)
                h_buf[w] = w;
        }

        bool check_gdr_support(LWdevice dev)
        {
            #if LWDA_VERSION >= 11030
            int drv_version;
            ASSERTDRV(lwDriverGetVersion(&drv_version));

            // Starting from LWCA 11.3, LWCA provides an ability to check GPUDirect RDMA support.
            if (drv_version >= 11030) {
                int gdr_support = 0;
                ASSERTDRV(lwDeviceGetAttribute(&gdr_support, LW_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev));

                if (!gdr_support)
                    print_dbg("This GPU does not support GPUDirect RDMA.\n");

                return !!gdr_support;
            }
            #endif

            // For older versions, we fall back to detect this support with gdr_pin_buffer.
            const size_t size = GPU_PAGE_SIZE;
            LWdeviceptr d_A;
            gpu_mem_handle_t mhandle;
            ASSERTDRV(gpu_mem_alloc(&mhandle, size, true, true));
            d_A = mhandle.ptr;

            gdr_t g = gdr_open_safe();

            gdr_mh_t mh;
            int status = gdr_pin_buffer(g, d_A, size, 0, 0, &mh);
            if (status != 0) {
                print_dbg("error in gdr_pin_buffer with code=%d\n", status);
                print_dbg("Your GPU might not support GPUDirect RDMA\n");
            }
            else
                ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);

            ASSERT_EQ(gdr_close(g), 0);

            ASSERTDRV(gpu_mem_free(&mhandle));

            return status == 0;
        }

        void print_histogram(double *lat_arr, int count, int *bin_arr, int num_bins, double min, double max)
        {
            int den = (max - min) / num_bins;
            den = den > 0 ? den : 1;
            for (int j = 0; j < num_bins; j++) 
                bin_arr[j] = 0;
            for (int i = 0; i < count; i++) {
                bin_arr[(int) ((lat_arr[i] - min) / den)]++;
            }
            for (int j = 0; j < num_bins; j++) {
                printf("[%lf\t-\t%lf]\t%d\n", (min * (j + 1)), (min * (j + 2)), bin_arr[j]);
            }
        }
    }
}
