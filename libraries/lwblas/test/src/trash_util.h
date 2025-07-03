#pragma once

#include <lwda_runtime.h>

#include "gtest/gtest.h"

namespace trash_util
{

__global__ void trashL2_kernel(float* junk1, float* junk2)
{
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   junk1[idx] = 1.3 * junk2[idx];
}

class TrashCache
{

    public:
        TrashCache() : trashSize_(1024 * 1024 * 100 * sizeof(float)), isInitialized_(false) {}

        ~TrashCache()
        {
            if (isInitialized_)
            {
                lwdaFree(junkBuff1_);
                lwdaFree(junkBuff2_);
            }
        }

        void init(lwdaStream_t stream)
        {
            ASSERT_EQ( lwdaMalloc((void**)&junkBuff1_, trashSize_), lwdaSuccess );
            ASSERT_EQ( lwdaMalloc((void**)&junkBuff2_, trashSize_), lwdaSuccess );
            ASSERT_EQ( lwdaMemsetAsync(junkBuff1_, 1, trashSize_, stream), lwdaSuccess );
            ASSERT_EQ( lwdaMemsetAsync(junkBuff2_, 2, trashSize_, stream), lwdaSuccess );
            ASSERT_EQ( lwdaStreamSynchronize(stream), lwdaSuccess );
            ASSERT_EQ( lwdaGetLastError(), lwdaSuccess );

            isInitialized_ = true;
        }

        void trashL2(lwdaStream_t stream)
        {
           int nthreads = 256;
           int nblocks = trashSize_ / (sizeof(float) * nthreads);
           trashL2_kernel<<<nblocks, nthreads, 0, stream>>>((float*)junkBuff1_, (float*)junkBuff2_);
           ASSERT_EQ( lwdaStreamSynchronize(stream), lwdaSuccess );
        }

        size_t getTrashSize() const {return 2 * trashSize_;}

    private:
        void*  junkBuff1_;
        void*  junkBuff2_;
        size_t trashSize_;
        bool   isInitialized_;
};

}  // namespace trash_util
