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
#pragma once

#include <xmma/xmma.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Named_barrier {
public:

    inline __device__ Named_barrier() {
        bar_ = -1;
        num_threads_ = -1;
    }

    inline __device__ Named_barrier(int bar, int num_threads) {
        bar_ = bar;
        num_threads_ = num_threads;
    }

    inline __device__ bool invalid() const {
        return num_threads_ == -1;
    }

    inline __device__ void wait(int i = 0) const {
        asm volatile("bar.sync %0, %1;" : : "r"(bar_+i), "r"(num_threads_));
    }

    inline __device__ void arrive(int i = 0) const {
        asm volatile("bar.arrive %0, %1;" : : "r"(bar_+i), "r"(num_threads_));
    }

private:
    // The idx of the named barrier.
    int bar_;
    // The number of threads (producers and consumers) that we expect to arrive at current barrier.
    int num_threads_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int N_>
struct Specialized_group {
public:

    // The number of threads in current specialized group.
    enum { N = N_ };

    inline __device__ Specialized_group(int start_rank) {
        static_assert( (N & (N - 1)) == 0, 
            "Number of threads in a specialized group must be pow-of-2 to avoid modulo.");
        rank_ = ((threadIdx.x - start_rank) & (N - 1));
    }

    inline __device__ int rank() const { return rank_; }

    inline __device__ int size() const { return N; }

protected:
    // Thread idx inside current specialize group.
    int rank_ = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Scheduler {
public:
    template< int SIZE, typename Op, typename ... Args >
    inline __device__ void launch(Op op, Args ... args) {
        // TBD: Lwrrently only support 1D thread allocation within a CTA.
        int start_rank = lwrrent_tid_;
        int end_rank = lwrrent_tid_ + SIZE;
        if( start_rank <= threadIdx.x && end_rank > threadIdx.x ) {
            Specialized_group<SIZE> sg(start_rank);
            op(sg, args ...);
        }
        lwrrent_tid_ = end_rank;
    }

    inline __device__ void sync() {
        __syncthreads();
    }

private:
    // The number of threads already used in previous specialized groups.
    int lwrrent_tid_ = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

