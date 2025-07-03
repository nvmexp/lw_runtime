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

#include <xmma/volta/traits.h>
#include <xmma/fragment.h>
#include <xmma/utils.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 1 6
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Volta_hmma_fp16_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Volta_hmma_fp16_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Volta_hmma_fp16_traits> : public Fragment<lwtlass::half_t, 8> {

    // The base class.
    using Base = Fragment<lwtlass::half_t, 8>;

    // Row fragment for A.
    using Fragment_row_a = Fragment_a<Volta_hmma_fp16_traits, Row>;
    // Col fragment for A.
    using Fragment_col_a = Fragment_a<Volta_hmma_fp16_traits, Col>;
    // Row fragment for B.
    using Fragment_row_b = Fragment_b<Volta_hmma_fp16_traits, Row>;
    // Col fragment for B.
    using Fragment_col_b = Fragment_b<Volta_hmma_fp16_traits, Col>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_REGS; ++ii ) {
            this->reg(ii) = hadd2(this->reg(ii), other.reg(ii));
        }
    }

    // HMMA.F16.
    inline __device__ void mma(const Fragment_row_a &a, const Fragment_row_b &b) {
#if (LWDA_ENABLE_EXTENDED_PTX)
        asm volatile( \
            "_mma.m8n8k4.row.row.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "_mma.m8n8k4.row.row.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#else
        asm volatile( \
            "mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#endif
    }

    // HMMA.F16.
    inline __device__ void mma(const Fragment_row_a &a, const Fragment_col_b &b) {
#if (LWDA_ENABLE_EXTENDED_PTX)
        asm volatile( \
            "_mma.m8n8k4.row.col.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "_mma.m8n8k4.row.col.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#else
        asm volatile( \
            "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#endif
    }

    // HMMA.F16.
    inline __device__ void mma(const Fragment_col_a &a, const Fragment_row_b &b) {
#if (LWDA_ENABLE_EXTENDED_PTX)
        asm volatile( \
            "_mma.m8n8k4.col.row.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "_mma.m8n8k4.col.row.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#else
        asm volatile( \
            "mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#endif
    }

    // HMMA.F16.
    inline __device__ void mma(const Fragment_col_a &a, const Fragment_col_b &b) {
#if (LWDA_ENABLE_EXTENDED_PTX)
        asm volatile( \
            "_mma.m8n8k4.col.col.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "_mma.m8n8k4.col.col.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#else
        asm volatile( \
            "mma.sync.aligned.m8n8k4.col.col.f16.f16.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "mma.sync.aligned.m8n8k4.col.col.f16.f16.f16.f16 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1)), "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#endif
    }

    // Load the data from global memory.
    template <int BYTES_PER_LDG = 16>
    inline __device__ void deserialize( const void *ptr, 
                                        int tidx_in_slice, 
                                        int threads_per_slice, 
                                        uint64_t mem_desc = MEM_DESC_DEFAULT ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Volta_hmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_hmma_fp16_epilogue_pre_swizzle<Volta_hmma_fp16_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<Volta_hmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_hmma_fp16_epilogue_post_swizzle<Volta_hmma_fp16_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Volta_hmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_hmma_fp16_c<Volta_hmma_fp16_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Volta_hmma_fp32_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Volta_hmma_fp32_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Volta_hmma_fp32_interleaved_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Volta_hmma_fp32_interleaved_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Traits>
struct Fragment_aclwmulator_hmma_fp32_base: public Fragment<float, 8> {

    // The base class.
    using Base = Fragment<float, 8>;

    // Col fragment for A.
    using Fragment_col_a = Fragment_a<Traits, Col>;
    // Row fragment for A.
    using Fragment_row_a = Fragment_a<Traits, Row>;
    // Col fragment for B.
    using Fragment_col_b = Fragment_b<Traits, Col>;
    // Row fragment for B.
    using Fragment_row_b = Fragment_b<Traits, Row>;


    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    // HMMA.COL.COL.F32.
    inline __device__ void mma(const Fragment_col_a &a, const Fragment_col_b &b) {
#if (LWDA_ENABLE_EXTENDED_PTX)
        asm volatile( \
            "_mma.m8n8k4.col.col.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "_mma.m8n8k4.col.col.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#else
        asm volatile( \
            "mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#endif
    }

    // HMMA.COL.ROW.F32.
    inline __device__ void mma(const Fragment_col_a &a, const Fragment_row_b &b) {
#if (LWDA_ENABLE_EXTENDED_PTX)
        asm volatile( \
            "_mma.m8n8k4.col.row.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "_mma.m8n8k4.col.row.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#else
        asm volatile( \
            "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#endif
    }

    // HMMA.ROW.COL.F32.
    inline __device__ void mma(const Fragment_row_a &a, const Fragment_col_b &b) {
#if (LWDA_ENABLE_EXTENDED_PTX)
        asm volatile( \
            "_mma.m8n8k4.row.col.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "_mma.m8n8k4.row.col.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#else
        asm volatile( \
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#endif
    }

    // HMMA.ROW.ROW.F32.
    inline __device__ void mma(const Fragment_row_a &a, const Fragment_row_b &b) {
#if (LWDA_ENABLE_EXTENDED_PTX)
        asm volatile( \
            "_mma.m8n8k4.row.row.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "_mma.m8n8k4.row.row.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#else
        asm volatile( \
            "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))); 
        asm volatile( \
            "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#endif
    }

    // HMMA.COL.COL.F32 with a predicate.
    inline __device__ void mma(const Fragment_col_a &a, const Fragment_col_b &b, int predicate) {
#if (LWDA_ENABLE_EXTENDED_PTX)
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %12, 0;\n" \
            "@p0 _mma.m8n8k4.col.col.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)), "r"(predicate)); 
        asm volatile( \
            "@p0 _mma.m8n8k4.col.col.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
            "}\n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#else
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %12, 0;\n" \
            "@p0 mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)), "r"(predicate)); 
        asm volatile( \
            "@p0 mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
            "}\n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#endif
    }

    // HMMA.COL.ROW.F32 with a predicate.
    inline __device__ void mma(const Fragment_col_a &a, const Fragment_row_b &b, int predicate) {
#if (LWDA_ENABLE_EXTENDED_PTX)
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %12, 0;\n" \
            "@p0 _mma.m8n8k4.col.row.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)), "r"(predicate)); 
        asm volatile( \
            "@p0 _mma.m8n8k4.col.row.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
            "}\n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#else
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %12, 0;\n" \
            "@p0 mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)), "r"(predicate)); 
        asm volatile( \
            "@p0 mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
            "}\n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#endif
    }

    // HMMA.ROW.COL.F32 with a predicate.
    inline __device__ void mma(const Fragment_row_a &a, const Fragment_col_b &b, int predicate) {
#if (LWDA_ENABLE_EXTENDED_PTX)
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %12, 0;\n" \
            "@p0 _mma.m8n8k4.row.col.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)), "r"(predicate)); 
        asm volatile( \
            "@p0 _mma.m8n8k4.row.col.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
            "}\n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#else
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %12, 0;\n" \
            "@p0 mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)), "r"(predicate)); 
        asm volatile( \
            "@p0 mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
            "}\n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#endif
    }

    // HMMA.ROW.ROW.F32 with a predicate.
    inline __device__ void mma(const Fragment_row_a &a, const Fragment_row_b &b, int predicate) {
#if (LWDA_ENABLE_EXTENDED_PTX)
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %12, 0;\n" \
            "@p0 _mma.m8n8k4.row.row.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)), "r"(predicate)); 
        asm volatile( \
            "@p0 _mma.m8n8k4.row.row.f32.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
            "}\n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3))); 
#else
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %12, 0;\n" \
            "@p0 mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)), "r"(predicate)); 
        asm volatile( \
            "@p0 mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%10, %11}, \n" \
            "    {%0, %1, %2, %3, %4, %5, %6, %7}; \n" \
            "}\n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    , "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#endif
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Volta_hmma_fp32_traits> : public Fragment_aclwmulator_hmma_fp32_base<Volta_hmma_fp32_traits> {

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Volta_hmma_fp32_interleaved_traits> :public Fragment_aclwmulator_hmma_fp32_base<Volta_hmma_fp32_interleaved_traits> {

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Volta_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_epilogue_pre_swizzle<Volta_hmma_fp32_traits, Cta_tile> {

    // The traits.
    using Traits = Volta_hmma_fp32_traits;
    // The aclwmulators from the main loop.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    // Quantize the aclwmulators -- do a scaled copy.
    inline __device__ void scaled_colwert(float alpha, const Aclwmulators &acc) {
        this->elt(0) = acc.elt(0) * alpha;
        this->elt(1) = acc.elt(1) * alpha;
        this->elt(2) = acc.elt(4) * alpha;
        this->elt(3) = acc.elt(5) * alpha;
        this->elt(4) = acc.elt(2) * alpha;
        this->elt(5) = acc.elt(3) * alpha;
        this->elt(6) = acc.elt(6) * alpha;
        this->elt(7) = acc.elt(7) * alpha;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_pre_swizzle<Volta_hmma_fp32_interleaved_traits, Cta_tile, true>
    : public Fragment_hmma_fp32_epilogue_pre_swizzle<Volta_hmma_fp32_interleaved_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_fp32_epilogue_pre_swizzle_base
    : public Fragment_hmma_fp16_epilogue_pre_swizzle<Traits, Cta_tile> {

    // The base class.
    using Base = Fragment_hmma_fp16_epilogue_pre_swizzle<Traits, Cta_tile>; 
    // The aclwmulators.
    using Aclwmulators = typename Base::Aclwmulators;

    // Colwert from fp16 aclwmulators to fp16 outputs.
    inline __device__ void colwert(float alpha, const Aclwmulators &acc) {
        this->reg(0) = float2_to_half2(alpha * acc.elt(0), alpha * acc.elt(1));
        this->reg(1) = float2_to_half2(alpha * acc.elt(4), alpha * acc.elt(5));
        this->reg(2) = float2_to_half2(alpha * acc.elt(2), alpha * acc.elt(3));
        this->reg(3) = float2_to_half2(alpha * acc.elt(6), alpha * acc.elt(7));
    }

    // Colwert from fp16 aclwmulators to fp16 outputs.
    inline __device__ void scaled_colwert(float alpha, const Aclwmulators &acc) {
        this->reg(0) = float2_to_half2(alpha * acc.elt(0), alpha * acc.elt(1));
        this->reg(1) = float2_to_half2(alpha * acc.elt(4), alpha * acc.elt(5));
        this->reg(2) = float2_to_half2(alpha * acc.elt(2), alpha * acc.elt(3));
        this->reg(3) = float2_to_half2(alpha * acc.elt(6), alpha * acc.elt(7));
    }

    // Colwert from fp16 aclwmulators to fp16 outputs.
    inline __device__ void colwert(Fragment<float, 8, 0> alpha, const Aclwmulators &acc) {
        this->reg(0) = float2_to_half2(alpha.elt(0) * acc.elt(0), alpha.elt(1) * acc.elt(1));
        this->reg(1) = float2_to_half2(alpha.elt(4) * acc.elt(4), alpha.elt(5) * acc.elt(5));
        this->reg(2) = float2_to_half2(alpha.elt(2) * acc.elt(2), alpha.elt(3) * acc.elt(3));
        this->reg(3) = float2_to_half2(alpha.elt(6) * acc.elt(6), alpha.elt(7) * acc.elt(7));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_pre_swizzle<Volta_hmma_fp32_interleaved_traits, Cta_tile, false> 
    : public Fragment_hmma_fp32_epilogue_pre_swizzle_base<Volta_hmma_fp32_interleaved_traits, Cta_tile> {
 
};


////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename Cta_tile, bool IN_CTA_SPLIT_K>
struct Fragment_epilogue_post_swizzle<Volta_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_epilogue_post_swizzle<Volta_hmma_fp32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_post_swizzle<Volta_hmma_fp32_interleaved_traits, Cta_tile, true>
    : public Fragment_hmma_fp32_epilogue_post_swizzle<Volta_hmma_fp32_interleaved_traits, Cta_tile> {

    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &, Fragment<float, 2> beta) {
    }

        template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias) {
         this->reg(0) = hadd2(this->reg(0), bias.reg(0));
    }

    template< typename Fragment_scale >
    inline __device__ void scale(const Fragment_scale &scale) {
        this->reg(0) = hmul2(this->reg(0), scale.reg(0));
    }

    inline __device__ void reduce(Fragment<float, 2> alpha) {
    }

    // RELU activation
    inline __device__ void relu(float relu_lb) {
        this->reg(0) = relu_fp16x2(this->reg(0), relu_lb);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_interleaved_post_swizzle<Volta_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_hmma_fp32_epilogue_interleaved_post_swizzle<
                 Volta_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_interleaved_post_swizzle<Volta_hmma_fp32_interleaved_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_hmma_fp32_epilogue_interleaved_post_swizzle<
                 Volta_hmma_fp32_interleaved_traits, Cta_tile, IN_CTA_SPLIT_K> {
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c & res, Fragment<lwtlass::half_t, 2> beta) {
        this->reg(0) = hfma2(beta.u32(0), res.to_int(), this->reg(0));
    }

    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias) {
        this->reg(0) = hadd2(this->reg(0), bias.reg(0));
    }
    
    template< typename Fragment_scale >
    inline __device__ void scale(const Fragment_scale &scale) {
        this->reg(0) = hmul2(this->reg(0), scale.reg(0));
    }

    // Do the parallel reduction.
    inline __device__ void reduce(float) {
    }


    inline __device__ void reduce(Fragment<float, 2> alpha) {
    }

    // RELU activation
    inline __device__ void relu(float relu_lb) {
        half2 relu_lb_ = xmma::colwert<half2>(relu_lb);
        this->reg(0) = relu_fp16x2(this->reg(0), reinterpret_cast<uint32_t&>(relu_lb_));
    }
    
    // Clip-ReLu.
    inline __device__ void relu_ub(float relu_ub) {
        half2 relu_ub_ = xmma::colwert<half2>(relu_ub);
        this->reg(0) = relu_ub_fp16x2(this->reg(0), reinterpret_cast<uint32_t&>(relu_ub_));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Volta_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_hmma_fp32_c<Volta_hmma_fp32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_interleaved_c<Volta_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_hmma_fp32_interleaved_c<Volta_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K> {
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_interleaved_c<Volta_hmma_fp32_interleaved_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_hmma_base_c<Volta_hmma_fp32_interleaved_traits, Cta_tile, 2> {

    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, float beta) {
    }

    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, Fragment<lwtlass::half_t, 2> beta) {   
    }

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(float, const Fragment_post_swizzle &frag) {
        this->reg(0) = frag.reg(0);
    }

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(Fragment<float, 2>, const Fragment_post_swizzle &frag) {
        this->reg(0) = frag.reg(0);
    }

    // The bias.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias_) {
        this->reg(0) = hadd2(bias_.reg(0), this->reg(0));
    }

    // The bias+relu.
    template< typename Fragment_bias >
    inline __device__ void add_bias_relu(const Fragment_bias &bias_,
                                         int32_t with_relu,
                                         float relu_lb,
                                         float one) {
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // ReLu.
    inline __device__ void relu(float relu_lb) {
        half2 relu_lb_ = xmma::colwert<half2>(relu_lb);
        this->reg(0) = relu_fp16x2(this->reg(0), reinterpret_cast<uint32_t&>(relu_lb_));
    }

    // Clip-ReLu.
    inline __device__ void relu_ub(float relu_ub) {
        half2 relu_ub_ = xmma::colwert<half2>(relu_ub);
        this->reg(0) = relu_ub_fp16x2(this->reg(0), reinterpret_cast<uint32_t&>(relu_ub_));
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8 NHWC/TN layout
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout, bool IS_GELU_ERF >
struct Fragment_a<Volta_imma_int8_int32_traits<IS_GELU_ERF>, Layout> : public Fragment<int8_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout, bool IS_GELU_ERF >
struct Fragment_b<Volta_imma_int8_int32_traits<IS_GELU_ERF>, Layout> : public Fragment<int8_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< bool IS_GELU_ERF >
struct Fragment_aclwmulator<Volta_imma_int8_int32_traits<IS_GELU_ERF>> 
    : public Fragment<int32_t, 8> {

    // The base class.
    using Base = Fragment<int32_t, 8>;

    // The fragments.
    using Fragment_a = xmma::Fragment_a<Volta_imma_int8_int32_traits<IS_GELU_ERF>, Row>;
    using Fragment_b = xmma::Fragment_b<Volta_imma_int8_int32_traits<IS_GELU_ERF>, Col>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    // IMMA.
    inline __device__ void mma(const Fragment_a &a, const Fragment_b &b) {
        #pragma unroll
        for( int i = 0; i < 4; ++i ) {
#if (XMMA_EXTENDED_PTX_ENABLED) || defined(__LWDA_ARCH__) && (__LWDA_ARCH__ <= 720)
            asm volatile( \
                "_mma.m8n8k16.row.col.s8.s8 \n" \
                "    {%0, %1}, \n" \
                "    {%2}, \n" \
                "    {%3}, \n" \
                "    {%0, %1}; \n" \
                : "+r"(this->reg(2*i+0))
                , "+r"(this->reg(2*i+1))
                :  "r"(    a.reg(i/2))
                ,  "r"(    b.reg(i%2)));
#else
            asm volatile( \
                "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 \n" \
                "    {%0, %1}, \n" \
                "    {%2}, \n" \
                "    {%3}, \n" \
                "    {%0, %1}; \n" \
                : "+r"(this->reg(2*i+0))
                , "+r"(this->reg(2*i+1))
                :  "r"(    a.reg(i/2))
                ,  "r"(    b.reg(i%2)));
#endif
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K, bool IS_GELU_ERF >
struct Fragment_epilogue_pre_swizzle<Volta_imma_int8_int32_traits<IS_GELU_ERF>, 
    Cta_tile, 
    IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_epilogue_pre_swizzle<
        Volta_imma_int8_int32_traits<IS_GELU_ERF>, 
        Cta_tile> {

    // The base class.
    using Base = Fragment_imma_int32_epilogue_pre_swizzle<Volta_imma_int8_int32_traits<IS_GELU_ERF>, 
        Cta_tile>;
    // The aclwmulators.
    using Aclwmulators = typename Base::Aclwmulators;

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void scaled_colwert(float &alpha, const Aclwmulators &acc) {

        // This is for per-tensor scaling.
        #pragma unroll
        for( int ii = 0; ii < Aclwmulators::NUM_REGS; ++ii ) {
            asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(this->elt(ii)) : "r"(acc.elt(ii)));
            this->elt(ii) = alpha * this->elt(ii);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K, bool IS_GELU_ERF >
struct Fragment_epilogue_post_swizzle<Volta_imma_int8_int32_traits<IS_GELU_ERF>, 
    Cta_tile, 
    IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_epilogue_post_swizzle<Volta_imma_int8_int32_traits<IS_GELU_ERF>, 
        Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K, bool IS_GELU_ERF >
struct Fragment_c<Volta_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_nhwc_int8_c<Volta_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8 NC/32HW32 layout
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Volta_imma_interleaved_int8_int32_traits, Layout> : public Fragment<int8_t, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Volta_imma_interleaved_int8_int32_traits, Layout> : public Fragment<int8_t, 32> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Volta_imma_interleaved_int8_int32_traits>
    : public Fragment<int32_t, 16> {

    // The base class.
    using Base = Fragment<int32_t, 16>;

    // The fragments.
    using Fragment_a = xmma::Fragment_a<Volta_imma_interleaved_int8_int32_traits, Row>;
    using Fragment_b = xmma::Fragment_b<Volta_imma_interleaved_int8_int32_traits, Col>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    // IMMA.
    inline __device__ void mma(const Fragment_a &a, const Fragment_b &b) {
        #pragma unroll
        for( int k = 0; k < 2; ++k ) {
            #pragma unroll
            for( int i = 0; i < 2; ++i ) {
                #pragma unroll
                for( int j = 0; j < 4; ++j ) {
#if (XMMA_EXTENDED_PTX_ENABLED) || defined(__LWDA_ARCH__) && (__LWDA_ARCH__ <= 720)
                    asm volatile( \
                        "_mma.m8n8k16.row.col.s8.s8.sat \n" \
                        "    {%0, %1}, \n" \
                        "    {%2}, \n" \
                        "    {%3}, \n" \
                        "    {%0, %1}; \n" \
                        : "+r"(this->reg(8*i+2*j+0))
                        , "+r"(this->reg(8*i+2*j+1))
                        :  "r"(    a.reg(2*i    +k))
                        ,  "r"(    b.reg(    2*j+k)));
#else
                    asm volatile( \
                        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 \n" \
                        "    {%0, %1}, \n" \
                        "    {%2}, \n" \
                        "    {%3}, \n" \
                        "    {%0, %1}; \n" \
                        : "+r"(this->reg(8*i+2*j+0))
                        , "+r"(this->reg(8*i+2*j+1))
                        :  "r"(    a.reg(2*i    +k))
                        ,  "r"(    b.reg(    2*j+k)));

#endif
                } // end j
            } // end i
        } // end k
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<
    Volta_imma_interleaved_int8_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_interleaved_int32_epilogue_pre_swizzle<
        Volta_imma_interleaved_int8_int32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_interleaved_post_swizzle<
    Volta_imma_interleaved_int8_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_fp32_epilogue_interleaved_post_swizzle<
        Volta_imma_interleaved_int8_int32_traits, Cta_tile, IN_CTA_SPLIT_K> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_interleaved_c<Volta_imma_interleaved_int8_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_interleaved_c<Volta_imma_interleaved_int8_int32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

