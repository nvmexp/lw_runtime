/***************************************************************************************************
 * Copyright (c) 2011-2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_SPLIT_K_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_SPLIT_K_H

#pragma once

#include "utils.h"
#include <cstdint>
#include <lwda_runtime.h>

namespace xmma
{
namespace ext
{
namespace depthwise_colwolution
{

template <typename Gmem_tile_c_t> struct Split_k {
    public:
    __device__ inline Split_k(Gmem_tile_c_t &gmem_tile_c)
    {
        index_of_the_split_k_slice_ = gmem_tile_c.get_index_of_the_split_k_slice();
        split_k_slices_ = gmem_tile_c.get_split_k_slices();
        split_k_buffers_ = gmem_tile_c.get_split_k_buffers();
        ptr_split_k_buffer_counter_ = gmem_tile_c.get_ptr_split_k_buffer_counter();
        ptr_split_k_final_counter_ = gmem_tile_c.get_ptr_split_k_final_counter();
    }

    __device__ inline bool is_the_first_batch()
    {
        return index_of_the_split_k_slice_ < split_k_buffers_;
    }

    __device__ inline void wait(void *address, int32_t target_value)
    {
        while (!(target_value ==
                 atomicCAS(static_cast<int32_t *>(address), target_value, target_value)))
            ;
    }

    __device__ inline void wait_for_the_former_slice_done()
    {
        wait(ptr_split_k_buffer_counter_,
             (index_of_the_split_k_slice_ / split_k_buffers_) * split_k_buffers_);
    }

    __device__ inline void wait_for_all_other_slices_done()
    {
        wait(ptr_split_k_final_counter_, split_k_slices_ - 1);
    }

    __device__ inline bool is_the_last_slice()
    {
        return index_of_the_split_k_slice_ == split_k_slices_ - 1;
    }

    __device__ inline void mark_the_lwrrent_slice_done()
    {
        if (threadIdx_x() + threadIdx_y() + threadIdx_z() == 0) {
            int32_t former_counter =
                atomicAdd(static_cast<int32_t *>(ptr_split_k_buffer_counter_), split_k_buffers_);
            int32_t final_counter =
                atomicAdd(static_cast<int32_t *>(ptr_split_k_final_counter_), 1);
        }
    }

    __device__ inline void exlwte(Gmem_tile_c_t &gmem_tile_c)
    {
        if (split_k_slices_ == 1) {
            return;
        }
        if (is_the_first_batch()) {
            gmem_tile_c.store_to_split_k_buffer();
            __syncthreads();
            __threadfence();
            mark_the_lwrrent_slice_done();
            return;
        } else {
            wait_for_the_former_slice_done();
            if (!is_the_last_slice()) {
                // Parallel reduction
                gmem_tile_c.atomic_add_in_the_split_k_buffer();
                __syncthreads();
                __threadfence();
                mark_the_lwrrent_slice_done();
                return;
            } else {
                wait_for_all_other_slices_done();
                gmem_tile_c.set_ptr_base_split_k();
                // Serial reduction
                for (int32_t index_split_k_buffer = 0; index_split_k_buffer < split_k_buffers_;
                     ++index_split_k_buffer) {
                    gmem_tile_c.set_ptrs_split_k();
                    gmem_tile_c.load_from_split_k_buffer(index_split_k_buffer);
                    gmem_tile_c.add_the_data_from_the_split_k_buffer();
                    gmem_tile_c.update_ptr_base_split_k();
                }
            }
        }
    }

    int32_t index_of_the_split_k_slice_;
    int32_t split_k_slices_;
    int32_t split_k_buffers_;
    void *ptr_split_k_buffer_counter_;
    void *ptr_split_k_final_counter_;
};

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
