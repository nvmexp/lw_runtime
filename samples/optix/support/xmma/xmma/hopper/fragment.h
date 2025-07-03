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

#include <xmma/hopper/traits.h>
#include <xmma/hopper/instructions.h>
#include <xmma/fragment.h>
#include <xmma/utils.h>
#include <xmma/hopper/gmma_descriptor.h>
#include <xmma/ampere/fragment.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 1 6
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Hopper_hmma_fp16_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Hopper_hmma_fp16_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Hopper_hmma_fp16_traits> : public Fragment<lwtlass::half_t, 8> {

    // The base class.
    using Base = Fragment<lwtlass::half_t, 8>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_REGS; ++ii ) {
            this->reg(ii) = hadd2(this->reg(ii), other.reg(ii));
        }
    }

    // Do the HMMA.
    template< typename Layout_a, typename Layout_b >
    inline __device__
        void mma(const Fragment_a<Hopper_hmma_fp16_traits, Layout_a> &a,
                 const Fragment_b<Hopper_hmma_fp16_traits, Layout_b> &b) {
#if (XMMA_EXTENDED_PTX_ENABLED)
        asm volatile( \
            "_mma.m16n8k16.row.col.f16.f16 \n" \
            "    {%0, %1}, \n" \
            "    {%2, %3, %4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)));
        asm volatile( \
            "_mma.m16n8k16.row.col.f16.f16 \n" \
            "    {%0, %1}, \n" \
            "    {%2, %3, %4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1}; \n" \
                    : "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#else
        asm volatile( \
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \n" \
            "    {%0, %1}, \n" \
            "    {%2, %3, %4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)));
        asm volatile( \
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \n" \
            "    {%0, %1}, \n" \
            "    {%2, %3, %4, %5}, \n" \
            "    {%6, %7}, \n" \
            "    {%0, %1}; \n" \
                    : "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));

#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Hopper_hmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp16_epilogue_pre_swizzle<Hopper_hmma_fp16_traits, Cta_tile> {

    // The traits.
    using Traits = Hopper_hmma_fp16_traits;
    // The aclwmulators from the main loop.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    // Colwert from fp16 aclwmulators to fp16 outputs.
    inline __device__ void colwert(lwtlass::half_t alpha, const Aclwmulators &acc) {
        uint32_t alpha_v2;
        asm volatile("mov.b32 %0, {%1, %1};\n" : "=r"(alpha_v2) : "h"(alpha));

        this->reg(0) = hmul2(alpha_v2, acc.reg(0));
        this->reg(1) = hmul2(alpha_v2, acc.reg(2));
        this->reg(2) = hmul2(alpha_v2, acc.reg(1));
        this->reg(3) = hmul2(alpha_v2, acc.reg(3));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<Hopper_hmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp16_epilogue_post_swizzle<Hopper_hmma_fp16_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Hopper_hmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp16_c<Hopper_hmma_fp16_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Hopper_hmma_fp32_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Hopper_hmma_fp32_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Hopper_hmma_fp32_traits> : public Fragment<float, 8> {

    // The base class.
    using Base = Fragment<float, 8>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    // Do the HMMA.
    template< typename Layout_a, typename Layout_b >
    inline __device__
        void mma(const Fragment_a<Hopper_hmma_fp32_traits, Layout_a> &a,
                 const Fragment_b<Hopper_hmma_fp32_traits, Layout_b> &b) {
#if (XMMA_EXTENDED_PTX_ENABLED)
        asm volatile( \
            "_mma.m16n8k16.row.col.f32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)));
        asm volatile( \
            "_mma.m16n8k16.row.col.f32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#else
        asm volatile( \
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)));
        asm volatile( \
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#endif
    }

    template< typename Layout_a, typename Layout_b >
    inline __device__
        void mma(const Fragment_a<Hopper_hmma_fp32_traits, Layout_a> &a,
                 const Fragment_b<Hopper_hmma_fp32_traits, Layout_b> &b,
                 int predicate) {
#if (XMMA_EXTENDED_PTX_ENABLED)
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %10, 0;\n" \
            "@p0 _mma.m16n8k16.row.col.f32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))
                    ,  "r"(predicate));
        asm volatile( \
            "@p0 _mma.m16n8k16.row.col.f32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
            "}\n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#else
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %10, 0;\n" \
            "@p0 mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))
                    ,  "r"(predicate));
        asm volatile( \
            "@p0 mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
            "}\n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));

#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Hopper_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_epilogue_pre_swizzle<Hopper_hmma_fp32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<Hopper_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp16_epilogue_post_swizzle<Hopper_hmma_fp32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Hopper_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_c<Hopper_hmma_fp32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Layout>
struct Fragment_gmma_epilogue_pre_swizzle {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Layout>
struct Fragment_gmma_epilogue_post_swizzle {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Layout>
struct Fragment_gmma_c {};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout of epilogue
    typename Layout,
    // The number of bytes per LDS
    int BYTES_PER_LDS = 16>
struct Gmma_epilogue_regs_num_per_stage {};

template <typename Traits, typename Cta_tile>
struct Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, xmma::Row, 16> {

    // The number of bytes per element
    enum { BYTES_PER_ELEMENT = sizeof( typename Traits::Epilogue_type ) };
    // The number of bytes per element C
    enum { BYTES_PER_ELEMENT_C = sizeof( typename Traits::C_type ) };
    // The number of bits per element
    enum { BITS_PER_ELEMENT = BYTES_PER_ELEMENT * 8 };
    // CTA tile size
    enum { CTA_M = Cta_tile::M, CTA_N = Cta_tile::N };

    // WARP GROUP distribution
    enum { WARP_GROUP_M = Cta_tile::WARP_GROUP_M, WARP_GROUP_N = Cta_tile::WARP_GROUP_N };
    // GMMA shape
    enum { GMMA_M = Traits::GMMA_M, GMMA_N = Traits::GMMA_N };
    // The number of threads
    enum { THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA };

    // reason about the min tile N
    // the size for each LDS.128
    enum { BYTES_PER_LDS = 16 };

    // tile_n if we can have 8 threads doing lds.128 unless cta_n is smaller than that
    enum { COLUMNS_PER_LDS = BYTES_PER_LDS * 8 / BYTES_PER_ELEMENT_C };
    enum { MIN_TILE_N = CTA_N < COLUMNS_PER_LDS ? CTA_N : COLUMNS_PER_LDS };

    // tile_m is limited such that every thread can participate, 8 rows per warp
    enum { TILE_M = 8 * THREADS_PER_CTA / 32, TILE_N = MIN_TILE_N };

    enum { ELEMENT_PER_32bit = 4 / BYTES_PER_ELEMENT };
    //
    // the number of 32 bit register held by each thread per tile before sts
    enum {
        NUM_REGS_PER_TILE_PRE_STORE =
            ( TILE_M * TILE_N ) / ( Cta_tile::WARP_GROUP_M * Cta_tile::THREADS_PER_WARP_GROUP ) /
            ELEMENT_PER_32bit
    };
    // the number of threads per lds needed by a row
    enum { LDS_THREADS_PER_ROW = TILE_N * BYTES_PER_ELEMENT_C / BYTES_PER_LDS };
    // the number of rows per LDS instructions by all threads
    enum { ROWS_PER_LDS = THREADS_PER_CTA / LDS_THREADS_PER_ROW };
    // the number of inner iterations
    enum { LDS_ITERATIONS_PER_TILE = TILE_M / ROWS_PER_LDS };
    // the number of 32 bit register held by each thread per tile after lds
    enum {
        NUM_REGS_PER_TILE_POST_LOAD =
            ( BYTES_PER_LDS * LDS_ITERATIONS_PER_TILE / BYTES_PER_ELEMENT_C ) / ELEMENT_PER_32bit
    };

    // The number of C regs held by each thread per tile before STG
    enum {
        NUM_REGS_PER_TILE_C =
            NUM_REGS_PER_TILE_POST_LOAD / ( BYTES_PER_ELEMENT / BYTES_PER_ELEMENT_C )
    };
};

template <typename Traits, typename Cta_tile>
struct Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, xmma::Col, 16> {

    // The number of bytes per element
    enum { BYTES_PER_ELEMENT = sizeof( typename Traits::Epilogue_type ) };
    // The number of bytes per element C
    enum { BYTES_PER_ELEMENT_C = sizeof( typename Traits::C_type ) };
    // The number of bits per element
    enum { BITS_PER_ELEMENT = BYTES_PER_ELEMENT * 8 };
    // CTA tile size
    enum { CTA_M = Cta_tile::M, CTA_N = Cta_tile::N };

    // WARP GROUP distribution
    enum { WARP_GROUP_M = Cta_tile::WARP_GROUP_M, WARP_GROUP_N = Cta_tile::WARP_GROUP_N };
    // GMMA shape
    enum { GMMA_M = Traits::GMMA_M, GMMA_N = Traits::GMMA_N };
    // The number of threads
    enum { THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA };

    // reason about the min tile N
    // the size for each LDS.128
    enum { BYTES_PER_LDS = 16 };

    // Threads for LDS per column
    enum { LDS_THREADS_PER_COLUMN = CTA_M * BYTES_PER_ELEMENT_C / BYTES_PER_LDS };
    static_assert( LDS_THREADS_PER_COLUMN >= 8,
                   "LDS_THREADS_PER_COLUMN should be larger than 8\n" );
    // the number of columns can be loaded by all threads per LDS instruction
    enum { COLUMNS_PER_LDS = THREADS_PER_CTA / LDS_THREADS_PER_COLUMN };

    // the min tile in N dim is 8 such that every thread can participate in sts
    enum { MIN_TILE_N = COLUMNS_PER_LDS < 8 ? 8 : COLUMNS_PER_LDS };
    static_assert( MIN_TILE_N % 8 == 0, "MIN_TILE_N should be multiple of 8" );

    // we can probably reduce the tile M to MIN_TILE_M, but for simplicity we set tile_M = cta_M
    enum { TILE_M = CTA_M, TILE_N = MIN_TILE_N };

    enum { ELEMENT_PER_32bit = 4 / BYTES_PER_ELEMENT };
    //
    // the number of 32 bit register held by each thread per tile before sts
    enum {
        NUM_REGS_PER_TILE_PRE_STORE =
            ( GMMA_M * TILE_N ) / ( Cta_tile::WARP_GROUP_M * Cta_tile::THREADS_PER_WARP_GROUP ) /
            ELEMENT_PER_32bit
    };

    // the number of inner iterations
    enum { LDS_ITERATIONS_PER_TILE = TILE_N / COLUMNS_PER_LDS };
    // the number of 32 bit register held by each thread per tile after lds
    enum {
        NUM_REGS_PER_TILE_POST_LOAD =
            ( BYTES_PER_LDS * LDS_ITERATIONS_PER_TILE / BYTES_PER_ELEMENT_C ) / ELEMENT_PER_32bit
    };

    // The number of C regs held by each thread per tile before STG
    enum {
        NUM_REGS_PER_TILE_C =
            NUM_REGS_PER_TILE_POST_LOAD / ( BYTES_PER_ELEMENT / BYTES_PER_ELEMENT_C )
    };
};

template <typename Traits, typename Cta_tile, typename Layout>
struct Fragment_hgmma_base_c
    : public Fragment<
          lwtlass::half_t,
          Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, Layout>::NUM_REGS_PER_TILE_C * 2> {

    // The base class.
    using Base =
        Fragment<lwtlass::half_t,
                 Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, Layout>::NUM_REGS_PER_TILE_C *
                     2>;

    enum { NUM_REGS = Base::NUM_REGS };

    // Compute the sum between two fragments.
    inline __device__ void add( const Fragment_hgmma_base_c &other ) {
        #pragma unroll
        for( int i = 0; i < NUM_REGS; ++i ) {
            this->reg( i ) = hadd2( this->reg( i ), other.reg( i ) );
        }
    }

    // Extract from an int2.
    inline __device__ void from_int2( const uint2 &x ) {
        this->reg( 0 ) = x.x;
        this->reg( 1 ) = x.y;
    }

    // Extract from an int4.
    inline __device__ void from_int4( const uint4 &x ) {
        this->reg( 0 ) = x.x;
        this->reg( 1 ) = x.y;
        this->reg( 2 ) = x.z;
        this->reg( 3 ) = x.w;
    }

    // Get an int2 from it.
    inline __device__ uint2 to_int2() const {
        return make_uint2( this->reg( 0 ), this->reg( 1 ) );
    }

    // Get an int4 from it.
    inline __device__ uint4 to_int4() const {
        return make_uint4( this->reg( 0 ), this->reg( 1 ), this->reg( 2 ), this->reg( 3 ) );
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H G M M A . F 1 6
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// both operands are coming from SMEM
template<int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_aclwmulator<Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, false, false> > 
  : public Fragment<lwtlass::half_t, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)> {

    // The base class.
    using Base = Fragment<lwtlass::half_t, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_REGS; ++ii ) {
            this->reg( ii ) = hadd2( this->reg( ii ), other.reg( ii ) );
        }
    }


    // Do the GMMA.
    template< typename Gmma_single_desc_a, typename Gmma_single_desc_b>
    inline __device__ void hgmma(const Gmma_single_desc_a &single_desc_a, 
                                 const Gmma_single_desc_b &single_desc_b, 
                                 bool increment_score_board) {
        // call hgmma
        if( increment_score_board == true ) {
            xmma::hgmma_fp16<
              Gmma_single_desc_a::TRANS_MODE == xmma::Gmma_descriptor_transpose::TRANS 
                  ? true : false, 
              Gmma_single_desc_b::TRANS_MODE == xmma::Gmma_descriptor_transpose::TRANS 
                  ? true : false,
              GMMA_N,
              true>(single_desc_a.get(), single_desc_b.get(), this->regs_);
        } else {
            xmma::hgmma_fp16<
              Gmma_single_desc_a::TRANS_MODE == xmma::Gmma_descriptor_transpose::TRANS 
                  ? true : false, 
              Gmma_single_desc_b::TRANS_MODE == xmma::Gmma_descriptor_transpose::TRANS 
                  ? true : false,
              GMMA_N,
              false>(single_desc_a.get(), single_desc_b.get(), this->regs_);          
        }
    }    
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// A is coming from RF; B is coming from SMEM
template<int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_aclwmulator<Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, true, false> > 
  : public Fragment<lwtlass::half_t, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)> {

    // The base class.
    using Base = Fragment<lwtlass::half_t, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)>;
    
    // The Traits
    using Traits = Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, true, false>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_REGS; ++ii ) {
            this->reg( ii ) = hadd2( this->reg( ii ), other.reg( ii ) );
        }
    }

    // Do the GMMA.
    template<typename Layout_a, typename Gmma_single_desc_b>
    inline __device__ void hgmma(const Fragment_a<Traits, Layout_a> &a, 
                                 const Gmma_single_desc_b &single_desc_b, 
                                 bool increment_score_board) {
        // call hgmma
        if( increment_score_board == true ) {
            xmma::hgmma_rfa_fp16<
              Gmma_single_desc_b::TRANS_MODE == xmma::Gmma_descriptor_transpose::TRANS 
                  ? true : false,
              GMMA_N,
              true>(a.regs_, single_desc_b.get(), this->regs_);
        } else {
            xmma::hgmma_rfa_fp16<
              Gmma_single_desc_b::TRANS_MODE == xmma::Gmma_descriptor_transpose::TRANS 
                  ? true : false,
              GMMA_N,
              false>(a.regs_, single_desc_b.get(), this->regs_);      
        }
    }    
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Layout>
struct Fragment_hgmma_fp16_epilogue_pre_swizzle
    : public Fragment<
          lwtlass::half_t,
          Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, Layout>::NUM_REGS_PER_TILE_PRE_STORE *
              2> {

    // The base class
    using Base = Fragment<
        lwtlass::half_t,
        Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, Layout>::NUM_REGS_PER_TILE_PRE_STORE *
            2>;

    // The aclwmulators from the main loop.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    enum { NUM_REGS = Base::NUM_REGS };

    // Compute the sum between two fragments.
    inline __device__ void add( const Fragment_hgmma_fp16_epilogue_pre_swizzle &other ) {
        #pragma unroll
        for( int i = 0; i < NUM_REGS; ++i ) {
            this->reg( i ) = hadd2( this->reg( i ), other.reg( i ) );
        }
    }

    // Colwert from fp16 aclwmulators to fp16 outputs.
    inline __device__ void colwert( int offset, lwtlass::half_t alpha, const Aclwmulators &acc ) {
        uint32_t alpha_v2;
        asm volatile( "mov.b32 %0, {%1, %1};\n" : "=r"( alpha_v2 ) : "h"( alpha ) );

        #pragma unroll
        for( int i = 0; i < NUM_REGS; ++i ) {
            this->reg( i ) = hmul2( alpha_v2, acc.reg( offset + i * 2 ) );
        }
    }

    inline __device__ void shuffle_groups( Aclwmulators &acc ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Fragment_hgmma_fp16_epilogue_pre_swizzle<Traits, Cta_tile, xmma::Col>
    : public Fragment<lwtlass::half_t,
                      Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, xmma::Col>::
                              NUM_REGS_PER_TILE_PRE_STORE *
                          2> {

    // The base class
    using Base = Fragment<
        lwtlass::half_t,
        Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, xmma::Col>::NUM_REGS_PER_TILE_PRE_STORE *
            2>;

    // The aclwmulators from the main loop.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    using Gmma_reg = Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, xmma::Col>;

    enum { NUM_REGS = Base::NUM_REGS };

    // Compute the sum between two fragments.
    inline __device__ void add( const Fragment_hgmma_fp16_epilogue_pre_swizzle &other ) {
        #pragma unroll
        for( int i = 0; i < NUM_REGS; ++i ) {
            this->reg( i ) = hadd2( this->reg( i ), other.reg( i ) );
        }
    }

    // Colwert from fp16 aclwmulators to fp16 outputs.
    inline __device__ void colwert( int offset, lwtlass::half_t alpha, const Aclwmulators &acc ) {
        uint32_t alpha_v2;
        asm volatile( "mov.b32 %0, {%1, %1};\n" : "=r"( alpha_v2 ) : "h"( alpha ) );

        #pragma unroll
        for( int i = 0; i < NUM_REGS; ++i ) {
            this->reg( i ) = hmul2( alpha_v2, acc.reg( offset + i ) );
        }
    }

    inline __device__ void shuffle_groups( Aclwmulators &acc ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Layout>
struct Fragment_hgmma_fp16_epilogue_post_swizzle
    : public Fragment<lwtlass::half_t,
                      Cta_tile::WARPS_K *
                          Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, Layout>::
                              NUM_REGS_PER_TILE_POST_LOAD *
                          2> {

    // The base class.
    using Base = Fragment<lwtlass::half_t,
                          Cta_tile::WARPS_K *
                              Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, Layout>::
                                  NUM_REGS_PER_TILE_POST_LOAD *
                              2>;

    // The number of registers after reduction.
    enum {
        NUM_REGS_AFTER_REDUCTION =
            Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, Layout>::NUM_REGS_PER_TILE_POST_LOAD
    };
    // Make sure the fragment oclwpies 4 registers after reduction.
    static_assert( Base::NUM_REGS == NUM_REGS_AFTER_REDUCTION * Cta_tile::WARPS_K, "" );
    // The number of bytes for load/store -- we only load/store the 1st 16 bytes.
    enum { BYTES_PER_LOAD_STORE = NUM_REGS_AFTER_REDUCTION * sizeof( uint32_t ) };

    // Add two fragments together.
    template <typename Other_fragment> inline __device__ void add( const Other_fragment &other ) {
        #pragma unroll
        for( int ii = 0; ii < NUM_REGS_AFTER_REDUCTION; ++ii ) {
            this->reg( ii ) = hadd2( this->reg( ii ), other.reg( ii ) );
        }
    }

    // The residual is added later.
    template <typename Fragment_c>
    inline __device__ void add_residual( const Fragment_c &res, lwtlass::half_t beta ) {
    }

    // The bias is added later.
    template <typename Fragment_bias> inline __device__ void add_bias( const Fragment_bias &bias ) {
    }

    // The bias is added later.
    template <typename Fragment_bias>
    inline __device__ void add_bias_nchw( const Fragment_bias &bias, int i ) {
    }

    // ReLu.
    inline __device__ void relu( lwtlass::half_t relu_lb ) {
    }

    // Clip-ReLu.
    inline __device__ void relu_ub( lwtlass::half_t relu_ub ) {
    }

    // Load from global memory (for inter-CTA split-k).
    template <int BYTES_PER_LDG = 16>
    inline __device__ void deserialize( const void *ptr, int tidx, int threads ) {
        xmma::deserialize_<BYTES_PER_LDG, NUM_REGS_AFTER_REDUCTION>(
            this->regs_, ptr, tidx, threads );
    }

    // Do the reduction for in-CTA split-K.
    inline __device__ void reduce( lwtlass::half_t ) {
        #pragma unroll
        for( int ki = 1; ki < Cta_tile::WARPS_K; ++ki ) {
            #pragma unroll
            for( int ii = 0; ii < NUM_REGS_AFTER_REDUCTION; ++ii ) {
                this->reg( ii ) = hadd2( this->reg( ii ), this->reg( 4 * ki + ii ) );
            }
        }
    }

    // Store to global memory (for inter-CTA split-k).
    template <int BYTES_PER_STG = 16>
    inline __device__ void serialize( void *ptr, int tidx, int threads ) const {
        xmma::serialize_<BYTES_PER_STG, NUM_REGS_AFTER_REDUCTION>(
            ptr, this->regs_, tidx, threads );
    }

    // Atomic add to global memory (for inter-CTA split-k).
    template <int BYTES_PER_STG = 16>
    inline __device__ void serialize_atomic_add( void *ptr, int tidx, int threads ) const {
        xmma::serialize_atomic_add_<BYTES_PER_STG, NUM_REGS_AFTER_REDUCTION>(
            ptr, this->regs_, tidx, threads );
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Layout>
struct Fragment_hgmma_fp16_c : public Fragment_hgmma_base_c<Traits, Cta_tile, Layout> {

    // The Base class.
    using Base = Fragment_hgmma_base_c<Traits, Cta_tile, Layout>;

    // The number of regs.
    enum { NUM_REGS = Base::NUM_REGS };

    // The residual.
    template <typename Fragment_c>
    inline __device__ void add_residual( const Fragment_c &res_, lwtlass::half_t beta ) {
        // uint4 res = res_.to_int4();
        ushort2 beta_ = make_ushort2( beta, beta );

        #pragma unroll
        for( int i = 0; i < Base::NUM_REGS; ++i ) {
            this->reg( i ) =
                hfma2( reinterpret_cast<const uint32_t &>( beta_ ), res_.reg( i ), this->reg( i ) );
        }
    }

    // Compute the sum between two fragments.
    template <typename Fragment_post_swizzle>
    inline __device__ void pack( lwtlass::half_t, const Fragment_post_swizzle &frag ) {

        #pragma unroll
        for( int i = 0; i < Base::NUM_REGS; ++i ) {
            this->reg( i ) = frag.reg( i );
        }
    }

    // The bias.
    template <typename Fragment_bias>
    inline __device__ void add_bias( const Fragment_bias &bias_ ) {

        #pragma unroll
        for( int i = 0; i < Base::NUM_REGS; ++i ) {
            this->reg( i ) = hadd2( bias_.reg( i ), this->reg( i ) );
        }
    }

    template <typename Fragment_bias>
    inline __device__ void add_bias_relu( const Fragment_bias &bias_,
                                          int32_t with_relu,
                                          lwtlass::half_t relu_lb,
                                          float one ) {
        uint32_t one2 = float2_to_half2( one, one );

        ushort2 tmp = make_ushort2( relu_lb, relu_lb );
        uint32_t relu_lb_ = reinterpret_cast<uint32_t &>( tmp );

        #pragma unroll
        for( int ii = 0; ii < Base::NUM_REGS; ++ii ) {
            this->reg( ii ) =
                xmma::hfma2_relu( this->reg( ii ), one2, bias_.reg( ii ), with_relu, relu_lb_ );
        }
    }

    // The bias is added later.
    template <typename Fragment_bias>
    inline __device__ void add_bias_nchw( const Fragment_bias &bias, int i ) {
    }

    // ReLu.
    inline __device__ void relu( lwtlass::half_t relu_lb ) {
        ushort2 relu_lb_ = make_ushort2( relu_lb, relu_lb );
        #pragma unroll
        for( int i = 0; i < Base::NUM_REGS; ++i ) {
            this->reg( i ) =
                relu_fp16x2( this->reg( i ), reinterpret_cast<const uint32_t &>( relu_lb_ ) );
        }
    }

    // Clip-ReLu.
    inline __device__ void relu_ub( lwtlass::half_t relu_ub ) {
        ushort2 relu_ub_ = make_ushort2( relu_ub, relu_ub );
        #pragma unroll
        for( int i = 0; i < Base::NUM_REGS; ++i ) {
            this->reg( i ) =
                relu_ub_fp16x2( this->reg( i ), reinterpret_cast<const uint32_t &>( relu_ub_ ) );
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          typename Layout>
struct Fragment_gmma_epilogue_pre_swizzle<
    Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
    Cta_tile,
    Layout>
    : public Fragment_hgmma_fp16_epilogue_pre_swizzle<
          Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile,
          Layout> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          typename Layout>
struct Fragment_gmma_epilogue_post_swizzle<
    Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
    Cta_tile,
    Layout>
    : public Fragment_hgmma_fp16_epilogue_post_swizzle<
          Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile,
          Layout> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          typename Layout>
struct Fragment_gmma_c<Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
                       Cta_tile,
                       Layout>
    : public Fragment_hgmma_fp16_c<
          Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile,
          Layout> {};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H G M M A . F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////
// both operands are coming from SMEM
template<int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_aclwmulator<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, false, false> >   
: public Fragment<float, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)> {

    // The base class.
    using Base = Fragment<float,
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt( ii ) = this->elt( ii ) + other.elt( ii );
        }
    }

    
    // Do the GMMA.
    template< typename Gmma_single_desc_a, typename Gmma_single_desc_b>
    inline __device__ void hgmma(const Gmma_single_desc_a &single_desc_a, 
                                 const Gmma_single_desc_b &single_desc_b, 
                                 bool increment_score_board) {
        // call hgmma
        if( increment_score_board == true ) {
            xmma::hgmma_fp32<
              Gmma_single_desc_a::TRANS_MODE == xmma::Gmma_descriptor_transpose::TRANS 
                  ? true : false, 
              Gmma_single_desc_b::TRANS_MODE == xmma::Gmma_descriptor_transpose::TRANS 
                  ? true : false,
              GMMA_N,
              true>(single_desc_a.get(), single_desc_b.get(), this->regs_);
        } else {
            xmma::hgmma_fp32<
              Gmma_single_desc_a::TRANS_MODE == xmma::Gmma_descriptor_transpose::TRANS 
                  ? true : false, 
              Gmma_single_desc_b::TRANS_MODE == xmma::Gmma_descriptor_transpose::TRANS 
                  ? true : false,
              GMMA_N,
              false>(single_desc_a.get(), single_desc_b.get(), this->regs_);          
        }
    }   
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// A is coming from RF; B is coming from SMEM
template<int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_aclwmulator<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, true, false> >   
: public Fragment<float, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)> {

    // The base class.
    using Base = Fragment<float, 
    (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper:: WARPS_PER_WARP_GROUP)>;
    
    // The Traits
    using Traits = Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, true, false>;


    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt( ii ) = this->elt( ii ) + other.elt( ii );
        }
    }

    
    // Do the GMMA.
    template<typename Layout_a, typename Gmma_single_desc_b>
    inline __device__ void hgmma(const Fragment_a<Traits, Layout_a> &a, 
                                 const Gmma_single_desc_b &single_desc_b, 
                                 bool increment_score_board) {
        // call hgmma
        if( increment_score_board == true ) {
            xmma::hgmma_rfa_fp32<
              Gmma_single_desc_b::TRANS_MODE == xmma::Gmma_descriptor_transpose::TRANS 
                  ? true : false,
              GMMA_N,
              true>(a.regs_, single_desc_b.get(), this->regs_);
        } else {
            xmma::hgmma_rfa_fp32<
              Gmma_single_desc_b::TRANS_MODE == xmma::Gmma_descriptor_transpose::TRANS 
                  ? true : false,
              GMMA_N,
              false>(a.regs_, single_desc_b.get(), this->regs_);      
        }
    } 
  
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Layout>
struct Fragment_hgmma_fp32_epilogue_pre_swizzle
    : public Fragment<
          float,
          Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, Layout>::NUM_REGS_PER_TILE_PRE_STORE> {

    // The base class.
    using Base = Fragment<
        float,
        Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, Layout>::NUM_REGS_PER_TILE_PRE_STORE>;

    // The aclwmulators.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    enum { NUM_REGS = Base::NUM_REGS };

    // Compute the sum between two fragments.
    inline __device__ void add( const Fragment_hgmma_fp32_epilogue_pre_swizzle &other ) {
        // Not needed as it happens after the swizzle!
    }

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void colwert( int offset, float, const Aclwmulators &acc ) {
        #pragma unroll
        for( int i = 0; i < NUM_REGS; ++i ) {
            this->elt( i ) = acc.elt( offset * 2 + ( i / 2 ) * 4 + i % 2 );
        }
    }

    inline __device__ void shuffle_groups( Aclwmulators &acc ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Fragment_hgmma_fp32_epilogue_pre_swizzle<Traits, Cta_tile, xmma::Col>
    : public Fragment<float,
                      Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, xmma::Col>::
                          NUM_REGS_PER_TILE_PRE_STORE> {

    // The base class.
    using Base = Fragment<
        float,
        Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, xmma::Col>::NUM_REGS_PER_TILE_PRE_STORE>;

    // The aclwmulators.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    enum { NUM_REGS = Base::NUM_REGS };

    // Compute the sum between two fragments.
    inline __device__ void add( const Fragment_hgmma_fp32_epilogue_pre_swizzle &other ) {
        // Not needed as it happens after the swizzle!
    }

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void colwert( int offset, float, const Aclwmulators &acc ) {
        #pragma unroll
        for( int i = 0; i < NUM_REGS; ++i ) {
            this->elt( i ) = acc.elt( offset * 2 + i );
        }
    }

    inline __device__ void shuffle_groups( Aclwmulators &acc ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Layout>
struct Fragment_hgmma_fp32_epilogue_post_swizzle
    : public Fragment<float,
                      Cta_tile::WARPS_K *
                          Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, Layout>::
                              NUM_REGS_PER_TILE_POST_LOAD> {

    using Base =
        Fragment<float,
                 Cta_tile::WARPS_K * Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, Layout>::
                                         NUM_REGS_PER_TILE_POST_LOAD>;

    // The number of registers after reduction.
    enum {
        NUM_REGS_AFTER_REDUCTION =
            Gmma_epilogue_regs_num_per_stage<Traits, Cta_tile, Layout>::NUM_REGS_PER_TILE_POST_LOAD
    };
    // Make sure the fragment oclwpies ELTS_PER_THREAD registers after reduction.
    static_assert( Base::NUM_REGS == NUM_REGS_AFTER_REDUCTION * Cta_tile::WARPS_K, "" );
    // The number of bytes for load/store -- we only load/store the 1st 16 bytes.
    enum { BYTES_PER_LOAD_STORE = NUM_REGS_AFTER_REDUCTION * sizeof( uint32_t ) };

    // Add two fragments together.
    template <typename Other_fragment> inline __device__ void add( const Other_fragment &other ) {
        #pragma unroll
        for( int ii = 0; ii < NUM_REGS_AFTER_REDUCTION; ++ii ) {
            this->elt( ii ) = this->elt( ii ) + other.elt( ii );
        }
    }

    // The residual is added later.
    template <typename Fragment_c>
    inline __device__ void add_residual( const Fragment_c &, float ) {
    }

    // The bias is added later.
    template <typename Fragment_bias> inline __device__ void add_bias( const Fragment_bias &bias ) {
    }

    // The bias is added later.
    template <typename Fragment_bias>
    inline __device__ void add_bias_nchw( const Fragment_bias &bias, int i ) {
    }

    // ReLu.
    inline __device__ void relu( float ) {
    }

    // Clip-ReLu.
    inline __device__ void relu_ub( float ) {
    }

    // Load from global memory (for inter-CTA split-k).
    template <int BYTES_PER_LDG = 16>
    inline __device__ void deserialize( const void *ptr, int tidx, int threads ) {
        xmma::deserialize_<BYTES_PER_LDG, NUM_REGS_AFTER_REDUCTION>(
            this->regs_, ptr, tidx, threads );
    }

    // Do the parallel reduction.
    inline __device__ void reduce( float alpha ) {
        #pragma unroll
        for( int ni = 0; ni < NUM_REGS_AFTER_REDUCTION; ++ni ) {
            #pragma unroll
            for( int ki = 1; ki < Cta_tile::WARPS_K; ++ki ) {
                this->elt( ni ) += this->elt( ki * 8 + ni );
            }
            this->elt( ni ) *= alpha;
        }
    }

    // Store to global memory (for inter-CTA split-k).
    template <int BYTES_PER_STG = 16>
    inline __device__ void serialize( void *ptr, int tidx, int threads ) const {
        xmma::serialize_<BYTES_PER_STG, NUM_REGS_AFTER_REDUCTION>(
            ptr, this->regs_, tidx, threads );
    }

    // Atomic add to global memory (for inter-CTA split-k).
    template <int BYTES_PER_STG = 16>
    inline __device__ void serialize_atomic_add( void *ptr, int tidx, int threads ) const {
        xmma::serialize_atomic_add_<BYTES_PER_STG, NUM_REGS_AFTER_REDUCTION>(
            ptr, this->regs_, tidx, threads );
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Layout>
struct Fragment_hgmma_fp32_c : public Fragment_hgmma_base_c<Traits, Cta_tile, Layout> {

    // Base class
    using Base = Fragment_hgmma_base_c<Traits, Cta_tile, Layout>;

    enum { NUM_REGS = Base::NUM_REGS };

    // The residual.
    template <typename Fragment_c>
    inline __device__ void add_residual( const Fragment_c &res_, float beta ) {

        uint32_t beta_beta = float2_to_half2( beta, beta );
        #pragma unroll
        for( int i = 0; i < NUM_REGS; ++i ) {
            this->reg( i ) = hfma2( beta_beta, res_.reg( i ), this->reg( i ) );
        }
    }

    // Compute the sum between two fragments.
    template <typename Fragment_post_swizzle>
    inline __device__ void pack( float, const Fragment_post_swizzle &frag ) {

        #pragma unroll
        for( int i = 0; i < NUM_REGS; ++i ) {
            this->reg( i ) = float2_to_half2( frag.elt( 2 * i ), frag.elt( 2 * i + 1 ) );
        }
    }

    // The bias.
    template <typename Fragment_bias>
    inline __device__ void add_bias( const Fragment_bias &bias_ ) {

        #pragma unroll
        for( int i = 0; i < NUM_REGS; ++i ) {
            this->reg( i ) = hadd2( bias_.reg( i ), this->reg( i ) );
        }
    }

    template <typename Fragment_bias>
    inline __device__ void
    add_bias_relu( const Fragment_bias &bias_, uint32_t with_relu, float relu_lb, float one ) {
        uint32_t one2 = float2_to_half2( one, one );

        half2 tmp = xmma::colwert<half2>( relu_lb );
        uint32_t relu_lb_ = reinterpret_cast<uint32_t &>( tmp );

        #pragma unroll
        for( int ii = 0; ii < NUM_REGS; ++ii ) {
            this->reg( ii ) =
                xmma::hfma2_relu( this->reg( ii ), one2, bias_.reg( ii ), with_relu, relu_lb_ );
        }
    }

    // The bias is added later.
    template <typename Fragment_bias>
    inline __device__ void add_bias_nchw( const Fragment_bias &bias, int i ) {
    }

    // ReLu.
    inline __device__ void relu( float relu_lb ) {
        half2 tmp = xmma::colwert<half2>( relu_lb );
        uint32_t relu_lb_ = reinterpret_cast<uint32_t &>( tmp );

        #pragma unroll
        for( int i = 0; i < NUM_REGS; ++i ) {
            this->reg( i ) = relu_fp16x2( this->reg( i ), relu_lb_ );
        }
    }

    // Clip-ReLu.
    inline __device__ void relu_ub( float relu_ub ) {
        half2 tmp = xmma::colwert<half2>( relu_ub );
        uint32_t relu_ub_ = reinterpret_cast<uint32_t &>( tmp );

        #pragma unroll
        for( int i = 0; i < NUM_REGS; ++i ) {
            this->reg( i ) = relu_ub_fp16x2( this->reg( i ), relu_ub_ );
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          typename Layout>
struct Fragment_gmma_epilogue_pre_swizzle<
    Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
    Cta_tile,
    Layout>
    : public Fragment_hgmma_fp32_epilogue_pre_swizzle<
          Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile,
          Layout> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          typename Layout>
struct Fragment_gmma_epilogue_post_swizzle<
    Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
    Cta_tile,
    Layout>
    : public Fragment_hgmma_fp32_epilogue_post_swizzle<
          Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile,
          Layout> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          typename Layout>
struct Fragment_gmma_c<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
                       Cta_tile,
                       Layout>
    : public Fragment_hgmma_fp32_c<
          Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile,
          Layout> {};

//////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// F R A G M E N T  (A)  
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// Only needed if Operand A is coming from RF.
template<int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Layout>
struct Fragment_a<Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Layout> 
  : public Fragment<lwtlass::half_t, 
                    (GMMA_M * GMMA_K) / (Hopper::WARPS_PER_WARP_GROUP * Hopper::THREADS_PER_WARP) > {
    // A should be coming from RF.
    static_assert(GMMA_A_RF == true, "GMMA_A_RF must be true to allocate RF for Operand A.\n");
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Only needed if Operand A is coming from RF.
template<int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Layout>
struct Fragment_a<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Layout> 
  : public Fragment<lwtlass::half_t, 
                    (GMMA_M * GMMA_K) / (Hopper::WARPS_PER_WARP_GROUP * Hopper::THREADS_PER_WARP) > {
    // A should be coming from RF.
    static_assert(GMMA_A_RF == true, "GMMA_A_RF must be true to allocate RF for Operand A.\n");
};

////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
//
// D M M A . F 6 4
//
////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename Layout >
struct Fragment_a<Hopper_dmma_fp64_traits, Layout> : public Fragment<double, 8> {
    //8 elements for one basic compute block 16x16x16
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Hopper_dmma_fp64_traits, Layout> : public Fragment<double, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Hopper_dmma_fp64_traits> : public Fragment<double, 8> {

    // The base class.
    using Base = Fragment<double, 8>;

    template< typename Layout_a, typename Layout_b >
    inline __device__ 
            void mma(const Fragment_a<Hopper_dmma_fp64_traits, Layout_a> &a,
                     const Fragment_b<Hopper_dmma_fp64_traits, Layout_b> &b) {  
        #pragma unroll
        for (int i = 0; i < 2; ++i){
            asm volatile(
            "_mma.m16n8k16.row.col.f64.f64.f64.f64 {%0, %1, %2, %3}, {%4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, {%0, %1, %2, %3};\n" 
            : "+d"(elt(i*4)), "+d"(elt(i*4+1)), "+d"(elt(i*4+2)), "+d"(elt(i*4+3))
            : "d"(a.elt(0)), "d"(a.elt(1)), "d"(a.elt(2)), "d"(a.elt(3)), "d"(a.elt(4)), "d"(a.elt(5)), "d"(a.elt(6)), "d"(a.elt(7)) 
                ,"d"(b.elt(i*4)), "d"(b.elt(i*4+1)), "d"(b.elt(i*4+2)), "d"(b.elt(i*4+3)));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Hopper_dmma_fp64_traits, Cta_tile, IN_CTA_SPLIT_K> : public Fragment<double, 8> {

    // The traits.
    using Traits = Hopper_dmma_fp64_traits;
    // The aclwmulators from the main loop.
    using Aclwmulators = Fragment_aclwmulator<Traits>;


    inline __device__ void colwert(double alpha, const Aclwmulators &acc) {
        // keep it as is.
        this->elt(0) = acc.elt(0);
        this->elt(1) = acc.elt(1);
        this->elt(2) = acc.elt(2);
        this->elt(3) = acc.elt(3);
        this->elt(4) = acc.elt(4);
        this->elt(5) = acc.elt(5);
        this->elt(6) = acc.elt(6);
        this->elt(7) = acc.elt(7);
    }

    inline __device__ void shuffle_groups(Aclwmulators &acc) {
        // Do nothing.
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<Hopper_dmma_fp64_traits, Cta_tile, IN_CTA_SPLIT_K>  : public Fragment<double, 8> {
    // Do the parallel reduction.
    inline __device__ void reduce(double &alpha) {
        //do nothing
    }

    // Add residual before packing
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, double beta) {
        // Do nothing
    }

     // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias) {
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // ReLu.
    inline __device__ void relu(double relu_lb) {
    }

    // Clip-ReLu.
    inline __device__ void relu_ub(double relu_ub) {
    }

    // Gelu_erf activation.
    inline __device__ void gelu_erf(double gelu_scale) {
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Hopper_dmma_fp64_traits, Cta_tile, IN_CTA_SPLIT_K> :  public Fragment<double, 8> {

    // Extract from an int4.
    inline __device__ void from_int4(const uint4 &x) {
        this->reg(0) = x.x;
        this->reg(1) = x.y;
        this->reg(2) = x.z;
        this->reg(3) = x.w;
    }

    // Colwert from a post-swizzle fragment.
    template< typename Fragment_post_swizzle >
    inline __device__ void pack(double alpha, const Fragment_post_swizzle &frag) {
        //do alpha operation for both DMMA w/o smem and normal DMMA
        this->elt(0) = frag.elt(0) * alpha;
        this->elt(1) = frag.elt(1) * alpha;
        this->elt(2) = frag.elt(2) * alpha;
        this->elt(3) = frag.elt(3) * alpha;
        this->elt(4) = frag.elt(4) * alpha;
        this->elt(5) = frag.elt(5) * alpha;
        this->elt(6) = frag.elt(6) * alpha;
        this->elt(7) = frag.elt(7) * alpha;
    }

    // The residual is added before store.
    inline __device__ void add_residual(const Fragment_c& res, double beta) {
        //Do beta
        this->elt(0) += beta * res.elt(0);
        this->elt(1) += beta * res.elt(1);
        this->elt(2) += beta * res.elt(2);
        this->elt(3) += beta * res.elt(3);
        this->elt(4) += beta * res.elt(4);
        this->elt(5) += beta * res.elt(5);
        this->elt(6) += beta * res.elt(6);
        this->elt(7) += beta * res.elt(7);
    
    }

    // Get an int4 from it.
    inline __device__ uint4 to_int4() const {
        return make_uint4(this->reg(0), this->reg(1), this->reg(2), this->reg(3));
    }

    // bias is added in pre_store()
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias) {
    }

    // bias_relu is added in pre_store()
    template< typename Fragment_bias >
    inline __device__ void add_bias_relu(const Fragment_bias &bias_, int32_t with_relu,
                                         double relu_lb, float one) {
    }

    // bias is added in pre_store()
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // ReLu.
    inline __device__ void relu(double relu_lb=0.f) {
    }

    // Clip-ReLu.
    inline __device__ void relu_ub(double relu_ub) {
    }

};


} // namespace xmma


