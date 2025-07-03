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

#include <xmma/ampere/traits.h>
#include <xmma/fragment.h>
#include <xmma/utils.h>

#ifndef XMMA_PTX_MMA_FP64_ENABLED
#define XMMA_PTX_MMA_FP64_ENABLED 0
#endif
namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 1 6
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Ampere_hmma_fp16_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Ampere_hmma_fp16_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Ampere_hmma_fp16_traits> : public Fragment<lwtlass::half_t, 8> {

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
        void mma(const Fragment_a<Ampere_hmma_fp16_traits, Layout_a> &a,
                 const Fragment_b<Ampere_hmma_fp16_traits, Layout_b> &b) {
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
struct Fragment_epilogue_pre_swizzle<Ampere_hmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp16_epilogue_pre_swizzle<Ampere_hmma_fp16_traits, Cta_tile> {

    // The traits.
    using Traits = Ampere_hmma_fp16_traits;
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

    // Colwert from fp16 aclwmulators to fp16 outputs.
    inline __device__ void scaled_colwert(lwtlass::half_t alpha, const Aclwmulators &acc) {
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
struct Fragment_epilogue_post_swizzle<Ampere_hmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp16_epilogue_post_swizzle<Ampere_hmma_fp16_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_hmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp16_c<Ampere_hmma_fp16_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Ampere_hmma_fp32_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Ampere_hmma_fp32_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Ampere_hmma_fp32_traits> : public Fragment<float, 8> {

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
        void mma(const Fragment_a<Ampere_hmma_fp32_traits, Layout_a> &a,
                 const Fragment_b<Ampere_hmma_fp32_traits, Layout_b> &b) {
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
        void mma(const Fragment_a<Ampere_hmma_fp32_traits, Layout_a> &a,
                 const Fragment_b<Ampere_hmma_fp32_traits, Layout_b> &b,
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
struct Fragment_epilogue_pre_swizzle<Ampere_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_epilogue_pre_swizzle<Ampere_hmma_fp32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<Ampere_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_epilogue_post_swizzle<Ampere_hmma_fp32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_c<Ampere_hmma_fp32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . E 8 M 1 0
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type, typename Layout >
struct Fragment_a<Ampere_hmma_tf32_traits<Input_type, Output_type>, Layout>
    : public Fragment<uint32_t, 4> {
    public:
        using Input_type_ = Input_type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type, typename Layout >
struct Fragment_b<Ampere_hmma_tf32_traits<Input_type, Output_type>, Layout>
    : public Fragment<uint32_t, 4> {
    public:
        using Input_type_ = Input_type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type >
struct Fragment_aclwmulator<Ampere_hmma_tf32_traits<Input_type, Output_type>>
    : public Fragment<float, 8> {

    // The traits.
    using Traits = Ampere_hmma_tf32_traits<Input_type, Output_type>;
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
    inline __device__ void mma(const Fragment_a<Traits, Layout_a> &a,
                               const Fragment_b<Traits, Layout_b> &b) {
#if (XMMA_EXTENDED_PTX_ENABLED)
        asm volatile( \
            "_mma.m16n8k8.row.col.f32.tf32.tf32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)));
        asm volatile( \
            "_mma.m16n8k8.row.col.f32.tf32.tf32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#else
        asm volatile( \
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)));
        asm volatile( \
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 \n" \
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
    inline __device__ void mma(const Fragment_a<Traits, Layout_a> &a,
                               const Fragment_b<Traits, Layout_b> &b,
                 int predicate) {
#if (XMMA_EXTENDED_PTX_ENABLED)
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %10, 0;\n" \
            "@p0 _mma.m16n8k8.row.col.f32.tf32.tf32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))
                    ,  "r"(predicate));
        asm volatile( \
            "@p0 _mma.m16n8k8.row.col.f32.tf32.tf32.f32 \n" \
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
            "@p0 mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))
                    ,  "r"(predicate));
        asm volatile( \
            "@p0 mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 \n" \
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

template< typename Input_type, typename Output_type, typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Ampere_hmma_tf32_traits<Input_type, Output_type>,
                                     Cta_tile,
                                     IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_epilogue_pre_swizzle<Ampere_hmma_tf32_traits<Input_type,
                                                                              Output_type>,
                                                     Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_tf32_epilogue_post_swizzle: public Fragment<float, Cta_tile::WARPS_K*4> {

    // The base class.
    using Base = Fragment<float, Cta_tile::WARPS_K*4>;

    // The number of registers after reduction.
    enum { NUM_REGS_AFTER_REDUCTION = 4 };
    // Make sure the fragment oclwpies 8 registers after reduction.
    static_assert(Base::NUM_REGS == NUM_REGS_AFTER_REDUCTION*Cta_tile::WARPS_K, "");
    // The number of bytes for load/store -- we only load/store the 1st 16 bytes.
    enum { BYTES_PER_LOAD_STORE = NUM_REGS_AFTER_REDUCTION*sizeof(uint32_t) };

    // Add two fragments together.
    template< typename Other_fragment >
    inline __device__ void add(const Other_fragment &other) {
        #pragma unroll
        for( int ii = 0; ii < 4; ++ii ) {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    // The residual is added later.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &, float) {
    }

    // The bias is to handle nhwc layout.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &) {
    }

    // The bias is to handle nchw layout.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &, int) {
    }

    // RELU activation.
    inline __device__ void relu(float) {
    }

    // Clip-RELU activation.
    inline __device__ void relu_ub(float) {
    }

    // Gelu_erf activation.
    inline __device__ void gelu_erf(float gelu_scale) {
    }

    // Load from global memory (for inter-CTA split-k).
    template <int BYTES_PER_LDG = 16>
    inline __device__ void deserialize( const void *ptr,
                                        int tidx,
                                        int threads,
                                        uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        xmma::deserialize_<BYTES_PER_LDG, NUM_REGS_AFTER_REDUCTION>(
            this->regs_, ptr, tidx, threads, mem_desc );
    }

    // Do the parallel reduction.
    inline __device__ void reduce(float alpha) {
        #pragma unroll
        for( int ni = 0; ni < 4; ++ni ) {
            #pragma unroll
            for( int ki = 1; ki < Cta_tile::WARPS_K; ++ki ) {
              this->elt(ni) += this->elt(ki*4 + ni);
            }
            this->elt(ni) *= alpha;
        }
    }

    // Store to global memory (for inter-CTA split-k).
    template <int BYTES_PER_STG = 16>
    inline __device__ void serialize(void *ptr, int tidx, int threads) const {
        xmma::serialize_<BYTES_PER_STG, NUM_REGS_AFTER_REDUCTION>(ptr, this->regs_, tidx, threads);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type, typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<Ampere_hmma_tf32_traits<Input_type, Output_type>,
                                      Cta_tile,
                                      IN_CTA_SPLIT_K>
    : public Fragment_hmma_tf32_epilogue_post_swizzle<
        Ampere_hmma_tf32_traits<Input_type, Output_type>, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits >
struct Fragment_tf32_c : public Fragment<float, 4> {

    // The base class.
    using Base = Fragment<float, 4>;
    // Make sure the size of the fragment is what we expect.
    static_assert(sizeof(Base) == 16 && Base::SIZE_IN_BYTES == 16 && Base::NUM_REGS == 4, "");

    // Compute the sum between two fragments.
    inline __device__ void add(const Fragment_tf32_c &other) {
        this->elt(0) = this->elt(0) + other.elt(0);
        this->elt(1) = this->elt(1) + other.elt(1);
        this->elt(2) = this->elt(2) + other.elt(2);
        this->elt(3) = this->elt(3) + other.elt(3);
    }

    // The residual is added before packing (on floats).
    inline __device__ void add_residual(const Fragment_tf32_c &res_, float beta) {
        #pragma unroll
        for( int ii = 0; ii < 4; ++ii ) {
            this->elt(ii) = this->elt(ii) + beta * res_.elt(ii);
        }
    }

    // The bias is added before packing (on floats).
    template< typename Fragment_tf32_bias >
    inline __device__ void add_bias(const Fragment_tf32_bias &bias_) {
        #pragma unroll
        for( int ii = 0; ii < 4; ++ii ) {
            // bias is tf32 stored in uint32_t. So we need to reinterpret it as float
            this->elt(ii) = this->elt(ii) + reinterpret_cast<const float&>(bias_.elt(ii));
        }
    }

    // The bias and relu.
    template< typename Fragment_bias >
    inline __device__ void add_bias_relu(const Fragment_bias &bias_, int32_t with_relu,
                                         float relu_lb, float one) {
        add_bias(bias_);
        relu(relu_lb);
    }

    // The bias is added before packing (on floats).
    template< typename Fragment_tf32_bias >
    inline __device__ void add_bias_nchw(const Fragment_tf32_bias& bias_, int i) {
        #pragma unroll
        for( int ii = 0; ii < 4; ++ii ) {
            this->elt(ii) = this->elt(ii) + bias_.elt(i);
        }
    }

    // RELU activation.
    inline __device__ void relu(float relu_lb=0.f) {
        #pragma unroll
        for ( int ii = 0; ii < 4; ++ii ) {
            this->elt(ii) = xmma::relu_fp32(this->elt(ii), relu_lb);
        }
    }

    // Clip-RELU activation.
    inline __device__ void relu_ub(float relu_ub) {
        #pragma unroll
        for ( int ii = 0; ii < 4; ++ii ) {
            this->elt(ii) = xmma::relu_ub_fp32(this->elt(ii), relu_ub);
        }
    }

    // Extract from an int2.
    inline __device__ void from_int2(const uint2 &x) {
        this->reg(0) = x.x;
        this->reg(1) = x.y;
    }

    // Extract from an int4.
    inline __device__ void from_int4(const uint4 &x) {
        this->reg(0) = x.x;
        this->reg(1) = x.y;
        this->reg(2) = x.z;
        this->reg(3) = x.w;
    }

    // Get an int2 from it.
    inline __device__ uint2 to_int2() const {
        return make_uint2(this->reg(0), this->reg(1));
    }

    // Get an int4 from it.
    inline __device__ uint4 to_int4() const {
        return make_uint4(this->reg(0), this->reg(1), this->reg(2), this->reg(3));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_hmma_tf32_traits<Input_type, lwtlass::float_tf32_t>,
                  Cta_tile,
                  IN_CTA_SPLIT_K>
    : public Fragment_tf32_c<Ampere_hmma_tf32_traits<Input_type, lwtlass::float_tf32_t>> {

    // Colwert from a post-swizzle fragment.
    template< typename Fragment_post_swizzle >
    inline __device__ void pack(float, const Fragment_post_swizzle &frag) {
        this->reg(0) = float_to_tf32_rn(frag.elt(0));
        this->reg(1) = float_to_tf32_rn(frag.elt(1));
        this->reg(2) = float_to_tf32_rn(frag.elt(2));
        this->reg(3) = float_to_tf32_rn(frag.elt(3));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_hmma_tf32_traits<Input_type, float>, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_tf32_c<Ampere_hmma_tf32_traits<Input_type, float>> {

    // Colwert from a post-swizzle fragment.
    template< typename Fragment_post_swizzle >
    inline __device__ void pack(float, const Fragment_post_swizzle &frag) {
        this->reg(0) = frag.reg(0);
        this->reg(1) = frag.reg(1);
        this->reg(2) = frag.reg(2);
        this->reg(3) = frag.reg(3);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . E 8 M 7
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Output_type, typename Layout >
struct Fragment_a<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>, Layout>
    : public Fragment<lwtlass::float_bf16_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Output_type, typename Layout >
struct Fragment_b<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>, Layout>
    : public Fragment<lwtlass::float_bf16_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename Output_type >
struct Fragment_aclwmulator<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>>
    : public Fragment<float, 8> {

    // The traits.
    using Traits = Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>;
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
    inline __device__ void mma(const Fragment_a<Traits, Layout_a> &a,
                               const Fragment_b<Traits, Layout_b> &b) {
#if (XMMA_EXTENDED_PTX_ENABLED)
        asm volatile( \
            "_mma.m16n8k16.row.col.f32.bf16.bf16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)));
        asm volatile( \
            "_mma.m16n8k16.row.col.f32.bf16.bf16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
#else
        asm volatile( \
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)));
        asm volatile( \
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n" \
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
    inline __device__ void mma(const Fragment_a<Traits, Layout_a> &a,
                               const Fragment_b<Traits, Layout_b> &b,
                 int predicate) {
#if (XMMA_EXTENDED_PTX_ENABLED)
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %10, 0;\n" \
            "@p0 _mma.m16n8k16.row.col.f32.bf16.bf16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))
                    ,  "r"(predicate));
        asm volatile( \
            "@p0 _mma.m16n8k16.row.col.f32.bf16.bf16.f32 \n" \
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
            "@p0 mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1))
                    ,  "r"(predicate));
        asm volatile( \
            "@p0 mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n" \
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

template< typename Input_type, typename Output_type, typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Ampere_hmma_bf16_traits<Input_type, Output_type>,
                                     Cta_tile,
                                     IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_epilogue_pre_swizzle<Ampere_hmma_bf16_traits<Input_type,
                                                                             Output_type>,
                                                     Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// If bf16 kernel output type is bf16, epilogue math will do in fp32 in post_swizzle fragment.
// So here post_swizzle fragment is inherited from fp32 post_swizzle fragment.
template< typename Input_type, typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<Ampere_hmma_bf16_traits<Input_type, lwtlass::float_bf16_t>,
                                      Cta_tile,
                                      IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_epilogue_post_swizzle<Ampere_hmma_bf16_traits<
                                                            Input_type,
                                                            lwtlass::float_bf16_t>,
                                                      Cta_tile> {
    // The residual is added.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, float beta) {
        for ( int ii = 0; ii < Fragment_c::NUM_REGS; ii++ ) {
            float2 tmp = bf16_2_to_float2(res.reg(ii));
            this->elt(ii * 2 + 0) += beta * tmp.x;
            this->elt(ii * 2 + 1) += beta * tmp.y;
        }
    }

    // The bias is added.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias) {
        #pragma unroll
        for (int ii = 0; ii < Fragment_bias::NUM_REGS; ii++) {
            float2 tmp = bf16_2_to_float2(bias.reg(ii));
            this->elt(ii * 2 + 0) += tmp.x;
            this->elt(ii * 2 + 1) += tmp.y;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// If bf16 kernel output type is fp32, epilogue math will do in fp32 in c fragment.
// So here post_swizzle fragment is inherited from tf32 post_swizzle fragment which does nothing.
template< typename Input_type, typename Cta_tile, bool IN_CTA_SPLIT_K>
struct Fragment_epilogue_post_swizzle<Ampere_hmma_bf16_traits<Input_type, float>,
                                      Cta_tile,
                                      IN_CTA_SPLIT_K>
    : public Fragment_hmma_tf32_epilogue_post_swizzle<
        Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, float>, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// If bf16 kernel output type is bf16, epilogue math will do in fp32 in post_swizzle fragment.
// So here c fragment is inherited from fp32 c fragment with pack() colwerting fp32 to bf16.
template< typename Input_type, typename Cta_tile, bool IN_CTA_SPLIT_K>
struct Fragment_c<Ampere_hmma_bf16_traits<Input_type, lwtlass::float_bf16_t>,
                  Cta_tile,
                  IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_c<Ampere_hmma_bf16_traits<Input_type, lwtlass::float_bf16_t>,
                                  Cta_tile> {

    template< typename Fragment_post_swizzle >
    inline __device__ void pack(float, const Fragment_post_swizzle &frag) {
        this->reg(0) = float2_to_bf16_2(frag.elt(0), frag.elt(1));
        this->reg(1) = float2_to_bf16_2(frag.elt(2), frag.elt(3));
        this->reg(2) = float2_to_bf16_2(frag.elt(4), frag.elt(5));
        this->reg(3) = float2_to_bf16_2(frag.elt(6), frag.elt(7));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// If bf16 kernel output type is fp32, epilogue math will do in fp32 in c fragment.
// So here c fragment is inherited from tf32 c fragment with pack() not colwerting fp32 to tf32.
template< typename Input_type, typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_hmma_bf16_traits<Input_type, float>,
                  Cta_tile,
                  IN_CTA_SPLIT_K>
    : public Fragment_tf32_c<Ampere_hmma_bf16_traits<Input_type, float> > {

    // Colwert from a post-swizzle fragment.
    template< typename Fragment_post_swizzle >
    inline __device__ void pack(float, const Fragment_post_swizzle &frag) {
        this->reg(0) = frag.reg(0);
        this->reg(1) = frag.reg(1);
        this->reg(2) = frag.reg(2);
        this->reg(3) = frag.reg(3);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout, bool IS_GELU_ERF >
struct Fragment_a<Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Layout>
    : public Fragment<int8_t, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout, bool IS_GELU_ERF >
struct Fragment_b<Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Layout>
    : public Fragment<int8_t, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< bool IS_GELU_ERF >
struct Fragment_aclwmulator<Ampere_imma_int8_int32_traits<IS_GELU_ERF>> 
    : public Fragment<int32_t, 8> {

    // The base class.
    using Base = Fragment<int32_t, 8>;

    // The fragments.
    using Fragment_a = xmma::Fragment_a<Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Row>;
    using Fragment_b = xmma::Fragment_b<Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Col>;

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
        for( int i = 0; i < 2; ++i ) {
#if (XMMA_EXTENDED_PTX_ENABLED)
	    asm volatile( \
	        "_mma.m16n8k32.row.col.s32.s8.s8.s32 \n" \
		"    {%0, %1, %2, %3}, \n" \
		"    {%4, %5, %6, %7}, \n" \
		"    {%8, %9}, \n" \
		"    {%0, %1, %2, %3}; \n" \
		: "+r"(  reg(i * 4))
                , "+r"(  reg(i * 4 + 1))
                , "+r"(  reg(i * 4 + 2))
                , "+r"(  reg(i * 4 + 3))
		:  "r"(a.reg(0))
                ,  "r"(a.reg(1))
                ,  "r"(a.reg(2))
                ,  "r"(a.reg(3))
                ,  "r"(b.reg(i * 2))
                ,  "r"(b.reg(i * 2 + 1)));
#else
	    asm volatile( \
	        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 \n" \
		"    {%0, %1, %2, %3}, \n" \
		"    {%4, %5, %6, %7}, \n" \
		"    {%8, %9}, \n" \
		"    {%0, %1, %2, %3}; \n" \
		: "+r"(  reg(i * 4))
                , "+r"(  reg(i * 4 + 1))
                , "+r"(  reg(i * 4 + 2))
                , "+r"(  reg(i * 4 + 3))
		:  "r"(a.reg(0))
                ,  "r"(a.reg(1))
                ,  "r"(a.reg(2))
                ,  "r"(a.reg(3))
                ,  "r"(b.reg(i * 2))
                ,  "r"(b.reg(i * 2 + 1)));

#endif
        }// end i
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K, bool IS_GELU_ERF >
struct Fragment_epilogue_pre_swizzle<
    Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_epilogue_pre_swizzle<
        Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile> {
    // The base class.
    using Base = Fragment_imma_int32_epilogue_pre_swizzle<
        Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile>;
    // The aclwmulators.
    using Aclwmulators = typename Base::Aclwmulators;

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void scaled_colwert(float &alpha, const Aclwmulators &acc) {
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

template< typename Cta_tile, bool IN_CTA_SPLIT_K, bool IS_GELU_ERF >
struct Fragment_epilogue_post_swizzle<
    Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_epilogue_post_swizzle<
        Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K, bool IS_GELU_ERF >
struct Fragment_c<Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_nhwc_int8_c<Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8 NHWC w/o Swizzle Epilogue
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Ampere_imma_wo_epi_swizzle_int8_int32_traits, Layout>
    : public Fragment_a<Ampere_imma_int8_int32_traits<false>, Layout> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Ampere_imma_wo_epi_swizzle_int8_int32_traits, Layout>
    : public Fragment_b<Ampere_imma_int8_int32_traits<false>, Layout> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< >
struct Fragment_aclwmulator<Ampere_imma_wo_epi_swizzle_int8_int32_traits>
    : public Fragment_aclwmulator<Ampere_imma_int8_int32_traits<false>> {

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<
    Ampere_imma_wo_epi_swizzle_int8_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_epilogue_pre_swizzle<
        Ampere_imma_int8_int32_traits<false>, Cta_tile, IN_CTA_SPLIT_K> {

    // The aclwmulators.
    using Aclwmulators = Fragment_aclwmulator<Ampere_imma_wo_epi_swizzle_int8_int32_traits>;

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void colwert(Fragment<float, Aclwmulators::NUM_REGS> &alpha, const Aclwmulators &acc) {
        #pragma unroll
        for( int ii = 0; ii < Aclwmulators::NUM_REGS; ++ii ) {
            asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(this->elt(ii)) : "r"(acc.elt(ii)));
            this->elt(ii) = alpha.elt(ii) * this->elt(ii);
        }
    }

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void scaled_colwert(float &alpha, const Aclwmulators &acc) {
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

template<
    typename Traits,
    typename Cta_tile,
    int REG_COUNT
>
struct Fragment_imma_int32_pack_epilogue_post_swizzle_base
    : public Fragment<float, REG_COUNT> {

    // The base class.
    using Base = Fragment<float, REG_COUNT>;

    // Do the parallel reduction.
    inline __device__ void reduce(float &alpha) {
    }

    // Do the parallel reduction.
    inline __device__ void reduce(Fragment<float, REG_COUNT> &alpha) {
    }

    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, float beta) {
    }

    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, Fragment<float, REG_COUNT> &beta) {
        #pragma unroll
        for( int ii = 0; ii < (REG_COUNT / Fragment_c::NUM_REGS) ; ++ii ) {
            float4 tmp = s8x4_to_float4(res.reg(ii));
            this->elt(ii * 4 + 0) += beta.elt(ii * 4 + 0) * tmp.x;
            this->elt(ii * 4 + 1) += beta.elt(ii * 4 + 1) * tmp.y;
            this->elt(ii * 4 + 2) += beta.elt(ii * 4 + 2) * tmp.z;
            this->elt(ii * 4 + 3) += beta.elt(ii * 4 + 3) * tmp.w;
        }
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias) {
        #pragma unroll
        for ( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) += bias.elt(ii);
        }
    }

    // RELU activation.
    inline __device__ void relu(float relu_lb=0.f) {
    }

    // Clip-RELU activation.
    inline __device__ void relu_ub(float relu_ub) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    int CTA_N
>
struct Fragment_imma_int32_pack_epilogue_post_swizzle {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile
>
struct Fragment_imma_int32_pack_epilogue_post_swizzle <Traits, Cta_tile, 128>
    : public Fragment_imma_int32_pack_epilogue_post_swizzle_base<Traits, Cta_tile, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile
>
struct Fragment_imma_int32_pack_epilogue_post_swizzle <Traits, Cta_tile, 64>
    : public Fragment_imma_int32_pack_epilogue_post_swizzle_base<Traits, Cta_tile, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<
    Ampere_imma_wo_epi_swizzle_int8_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_pack_epilogue_post_swizzle<
        Ampere_imma_int8_int32_traits<false>, Cta_tile, Cta_tile::N> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_imma_wo_epi_swizzle_int8_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_c<Ampere_imma_int8_int32_traits<false>, Cta_tile, IN_CTA_SPLIT_K> {

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(Fragment<float, Fragment_post_swizzle::NUM_REGS>,
                                const Fragment_post_swizzle &frag) {

        int32_t tmp[4];
        #pragma unroll
        for( int ii = 0; ii < Fragment_post_swizzle::NUM_REGS/4; ++ii ) {
            asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(tmp[0]) : "f"(frag.elt(4*ii  )));
            asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(tmp[1]) : "f"(frag.elt(4*ii+1)));
            asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(tmp[2]) : "f"(frag.elt(4*ii+2)));
            asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(tmp[3]) : "f"(frag.elt(4*ii+3)));

            asm volatile(
                "{ .reg .u32 r4;"
                "cvt.pack.sat.s8.s32.b32   r4, %4, %3, 0;\n"
                "cvt.pack.sat.s8.s32.b32   %0, %2, %1, r4;\n"
                "}"
                : "=r"(this->reg(ii)) : "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]), "r"(tmp[3]));
        }

    }

    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, Fragment<float, 8> beta) {
    }

    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, Fragment<float, 16> beta) {
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8 INTERLEAVED NC/32HW32
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Input_type,
         typename Output_type,
         bool IS_GELU,
         bool IS_EPIFADD,
         bool IS_SWISH,
         bool IS_RT_FUSE,
         typename Layout>
struct Fragment_a<Ampere_imma_interleaved_traits<Input_type,
                                                 Output_type,
                                                 IS_GELU,
                                                 IS_EPIFADD,
                                                 IS_SWISH,
                                                 IS_RT_FUSE>,
                  Layout>
    : public Fragment<Input_type, 16> {
};

template<typename Input_type,
         typename Output_type,
         bool IS_GELU,
         bool IS_EPIFADD,
         bool IS_SWISH,
         bool IS_RT_FUSE,
         typename Layout>
struct Fragment_b<Ampere_imma_interleaved_traits<Input_type,
                                                 Output_type,
                                                 IS_GELU,
                                                 IS_EPIFADD,
                                                 IS_SWISH,
                                                 IS_RT_FUSE>,
                  Layout>
    : public Fragment<Input_type, 32> {
};

template< typename Input_type, typename Output_type, bool IS_GELU, bool IS_EPIFADD, bool IS_SWISH,
        bool IS_RT_FUSE >
struct Fragment_aclwmulator<Ampere_imma_interleaved_traits<Input_type,
                                                           Output_type,
                                                           IS_GELU,
                                                           IS_EPIFADD,
                                                           IS_SWISH,
                                                           IS_RT_FUSE>>
    : public Fragment<int32_t, 16> {

    // The base class.
    using Base = Fragment<int32_t, 16>;

    // The fragments.
    using Fragment_a = Fragment_a<Ampere_imma_interleaved_traits<Input_type,
                                                                 Output_type,
                                                                 IS_GELU,
                                                                 IS_EPIFADD,
                                                                 IS_SWISH,
                                                                 IS_RT_FUSE>,
                                  Row>;
    using Fragment_b = Fragment_b<Ampere_imma_interleaved_traits<Input_type,
                                                                 Output_type,
                                                                 IS_GELU,
                                                                 IS_EPIFADD,
                                                                 IS_SWISH,
                                                                 IS_RT_FUSE>,
                                  Col>;


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
#if (XMMA_EXTENDED_PTX_ENABLED)
	    asm volatile( \
	        "_mma.m16n8k32.row.col.s32.s8.s8.s32 \n" \
		"    {%0, %1, %2, %3}, \n" \
		"    {%4, %5, %6, %7}, \n" \
		"    {%8, %9}, \n" \
		"    {%0, %1, %2, %3}; \n" \
		: "+r"(  reg(i * 4))
                , "+r"(  reg(i * 4 + 1))
                , "+r"(  reg(i * 4 + 2))
                , "+r"(  reg(i * 4 + 3))
		:  "r"(a.reg(0))
                ,  "r"(a.reg(1))
                ,  "r"(a.reg(2))
                ,  "r"(a.reg(3))
                ,  "r"(b.reg(i * 2))
                ,  "r"(b.reg(i * 2 + 1)));
#else
	    asm volatile( \
	        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 \n" \
		"    {%0, %1, %2, %3}, \n" \
		"    {%4, %5, %6, %7}, \n" \
		"    {%8, %9}, \n" \
		"    {%0, %1, %2, %3}; \n" \
		: "+r"(  reg(i * 4))
                , "+r"(  reg(i * 4 + 1))
                , "+r"(  reg(i * 4 + 2))
                , "+r"(  reg(i * 4 + 3))
		:  "r"(a.reg(0))
                ,  "r"(a.reg(1))
                ,  "r"(a.reg(2))
                ,  "r"(a.reg(3))
                ,  "r"(b.reg(i * 2))
                ,  "r"(b.reg(i * 2 + 1)));
#endif
        }// end i
    }

    inline __device__ void clear() {
        if(IS_EPIFADD) {
            #pragma unroll
            for( int ii = 0; ii < NUM_REGS; ++ii ) {
                reg(ii) = 0x4B400000;
            }
        } else {
            #pragma unroll
            for( int ii = 0; ii < NUM_REGS; ++ii ) {
                asm volatile ("mov.u32 %0, 0; \n" : "=r"(this->reg(ii)) : );
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename Cta_tile,
          typename Input_type,
          typename Output_type,
          bool IS_GELU,
          bool IS_EPIFADD,
          bool IS_SWISH,
          bool IS_RT_FUSE,
          bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Ampere_imma_interleaved_traits<Input_type,
                                                                    Output_type,
                                                                    IS_GELU,
                                                                    IS_EPIFADD,
                                                                    IS_SWISH,
                                                                    IS_RT_FUSE>,
                                     Cta_tile,
                                     IN_CTA_SPLIT_K>
    : public Fragment_ampere_imma_interleaved_int32_epilogue_pre_swizzle<
        Ampere_imma_interleaved_traits<Input_type,
                                       Output_type,
                                       IS_GELU,
                                       IS_EPIFADD,
                                       IS_SWISH,
                                       IS_RT_FUSE>,
                                       Cta_tile> {
};

template< typename Cta_tile,
          typename Input_type,
          typename Output_type,
          bool IS_GELU,
          bool IS_SWISH,
          bool IS_RT_FUSE,
          bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Ampere_imma_interleaved_traits<Input_type,
                                                                    Output_type,
                                                                    IS_GELU,
                                                                    true,
                                                                    IS_SWISH,
                                                                    IS_RT_FUSE>,
                                     Cta_tile,
                                     IN_CTA_SPLIT_K>
    : public Fragment_ampere_imma_interleaved_int32_epilogue_fadd_pre_swizzle<
        Ampere_imma_interleaved_traits<Input_type,
                                       Output_type,
                                       IS_GELU,
                                       true,
                                       IS_SWISH,
                                       IS_RT_FUSE>,
                                       Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile,
          typename Input_type,
          typename Output_type,
          bool IS_GELU,
          bool IS_EPIFADD,
          bool IS_SWISH,
          bool IS_RT_FUSE,
          bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_interleaved_post_swizzle<Ampere_imma_interleaved_traits<Input_type,
                                                                                 Output_type,
                                                                                 IS_GELU,
                                                                                 IS_EPIFADD,
                                                                                 IS_SWISH,
                                                                                 IS_RT_FUSE>,
                                                  Cta_tile,
                                                  IN_CTA_SPLIT_K>
    : public Fragment_imma_fp32_epilogue_interleaved_post_swizzle<
        Ampere_imma_interleaved_traits<Input_type, Output_type, IS_GELU, IS_EPIFADD, IS_SWISH,
        IS_RT_FUSE>,
        Cta_tile,
        IN_CTA_SPLIT_K> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Cta_tile, bool IS_GELU, bool IS_EPIFADD, bool IS_SWISH, bool IS_RT_FUSE,
        bool IN_CTA_SPLIT_K>
struct Fragment_epilogue_interleaved_post_swizzle<Ampere_imma_interleaved_traits<int8_t,
                                                                                 lwtlass::half_t,
                                                                                 IS_GELU,
                                                                                 IS_EPIFADD,
                                                                                 IS_SWISH,
                                                                                 IS_RT_FUSE>,
                                                  Cta_tile,
                                                  IN_CTA_SPLIT_K>
    : public Fragment_imma_fp32_epilogue_interleaved_post_swizzle<
        Ampere_imma_interleaved_traits<int8_t, lwtlass::half_t, IS_GELU, IS_EPIFADD, IS_SWISH,
        IS_RT_FUSE>,
        Cta_tile,
        IN_CTA_SPLIT_K> {

    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, Fragment<float, 8> &beta) {
        #pragma unroll
        for ( int ri = 0; ri < Fragment_c::NUM_REGS; ++ri ) {
            float2 tmp;
            tmp = half2_to_float2(res.reg(ri));
            this->elt(ri * 2 + 0) += tmp.x * beta.elt((ri * 2 + 0));
            this->elt(ri * 2 + 1) += tmp.y * beta.elt((ri * 2 + 1));
        }
    }
};

template<typename Cta_tile, bool IS_GELU, bool IS_EPIFADD, bool IS_SWISH, bool IS_RT_FUSE, bool IN_CTA_SPLIT_K>
struct Fragment_epilogue_interleaved_post_swizzle<Ampere_imma_interleaved_traits<int8_t,
                                                                                 float,
                                                                                 IS_GELU,
                                                                                 IS_EPIFADD,
                                                                                 IS_SWISH,
                                                                                 IS_RT_FUSE>,
                                                  Cta_tile,
                                                  IN_CTA_SPLIT_K>
    : public Fragment_imma_fp32_epilogue_interleaved_post_swizzle<
        Ampere_imma_interleaved_traits<int8_t, float, IS_GELU, IS_EPIFADD, IS_SWISH, IS_RT_FUSE>,
        Cta_tile,
        IN_CTA_SPLIT_K> {

    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, Fragment<float, 8> &beta) {
        #pragma unroll
        for ( int ri = 0; ri < Fragment_c::NUM_REGS; ++ri ) {
            this->elt(ri) += res.elt(ri) * beta.elt(ri);
        }

    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile,
          typename Input_type,
          typename Output_type,
          bool IS_GELU,
          bool IS_EPIFADD,
          bool IS_SWISH,
          bool IS_RT_FUSE,
          bool IN_CTA_SPLIT_K >
struct Fragment_interleaved_c<Ampere_imma_interleaved_traits<Input_type,
                                                             Output_type,
                                                             IS_GELU,
                                                             IS_EPIFADD,
                                                             IS_SWISH,
                                                             IS_RT_FUSE>,
                              Cta_tile,
                              IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_interleaved_c<
        Ampere_imma_interleaved_traits<Input_type,
                                       Output_type,
                                       IS_GELU,
                                       IS_EPIFADD,
                                       IS_SWISH,
                                       IS_RT_FUSE>,
        Cta_tile> {
};

// Specilized for half output
template< typename Cta_tile, bool IS_GELU, bool IS_EPIFADD, bool IS_SWISH, bool IS_RT_FUSE,
        bool IN_CTA_SPLIT_K >
struct Fragment_interleaved_c<Ampere_imma_interleaved_traits<int8_t,
                                                             lwtlass::half_t,
                                                             IS_GELU,
                                                             IS_EPIFADD,
                                                             IS_SWISH,
                                                             IS_RT_FUSE>,
                              Cta_tile,
                              IN_CTA_SPLIT_K>
    : public Fragment<lwtlass::half_t, 8> {

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(Fragment<float, Fragment_post_swizzle::NUM_REGS>,
                                const Fragment_post_swizzle &frag) {

        this->reg(0) = float2_to_half2(frag.elt(0), frag.elt(1));
        this->reg(1) = float2_to_half2(frag.elt(2), frag.elt(3));
        this->reg(2) = float2_to_half2(frag.elt(4), frag.elt(5));
        this->reg(3) = float2_to_half2(frag.elt(6), frag.elt(7));
    }

    // Add the residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, Fragment<float, 8> beta) {
    }
};

// Specilized for float output
template< typename Cta_tile, bool IS_GELU, bool IS_EPIFADD, bool IS_SWISH, bool IS_RT_FUSE,
        bool IN_CTA_SPLIT_K >
struct Fragment_interleaved_c<Ampere_imma_interleaved_traits<int8_t,
                                                             float,
                                                             IS_GELU,
                                                             IS_EPIFADD,
                                                             IS_SWISH,
                                                             IS_RT_FUSE>,
                              Cta_tile,
                              IN_CTA_SPLIT_K>
    : public Fragment<float, 8> {

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(Fragment<float, Fragment_post_swizzle::NUM_REGS>,
                                const Fragment_post_swizzle &frag) {
        this->reg(0) = frag.reg(0);
        this->reg(1) = frag.reg(1);
        this->reg(2) = frag.reg(2);
        this->reg(3) = frag.reg(3);
        this->reg(4) = frag.reg(4);
        this->reg(5) = frag.reg(5);
        this->reg(6) = frag.reg(6);
        this->reg(7) = frag.reg(7);
    }

    // Add the residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, Fragment<float, 8> beta) {
    }
};

// Specilized for epilogue FADD trick
template< typename Cta_tile,
          typename Input_type,
          typename Output_type,
          bool IS_GELU,
          bool IS_SWISH,
          bool IS_RT_FUSE,
          bool IN_CTA_SPLIT_K >
struct Fragment_interleaved_c<Ampere_imma_interleaved_traits<Input_type,
                                                             Output_type,
                                                             IS_GELU,
                                                             true,
                                                             IS_SWISH,
                                                             IS_RT_FUSE>,
                              Cta_tile,
                              IN_CTA_SPLIT_K>
    : public Fragment_imma_epilogue_fadd_int32_interleaved_c<
        Ampere_imma_interleaved_traits<Input_type,
                                       Output_type,
                                       IS_GELU,
                                       true,
                                       IS_SWISH,
                                       IS_RT_FUSE>,
        Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// D M M A . F 6 4
//
////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename Layout >
struct Fragment_a<Ampere_dmma_fp64_traits, Layout> : public Fragment<double, 1> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Ampere_dmma_fp64_traits, Layout> : public Fragment<double, 1> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Ampere_dmma_fp64_traits> : public Fragment<double, 2> {

    // The base class.
    using Base = Fragment<double, 2>;

    template< typename Layout_a, typename Layout_b >
    inline __device__
        void mma(const Fragment_a<Ampere_dmma_fp64_traits, Layout_a> &a,
                 const Fragment_b<Ampere_dmma_fp64_traits, Layout_b> &b) {
#if (XMMA_PTX_MMA_FP64_ENABLED == 0)
            asm volatile( \
            "_mma.m8n8k4.row.col.f64.f64.f64.f64 \n" \
            "{%0,   %1},    %2,   %3,   {%0,    %1}; \n" \
            : "+d"(elt(0)), "+d"(elt(1))
            : "d"(a.elt(0)), "d"(b.elt(0)));
#else
            asm volatile( \
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 \n" \
            "{%0,   %1},    {%2},   {%3},   {%0,    %1}; \n" \
            : "+d"(elt(0)), "+d"(elt(1))
            : "d"(a.elt(0)), "d"(b.elt(0)));
#endif
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Ampere_dmma_fp64_traits, Cta_tile, IN_CTA_SPLIT_K> : public Fragment<double, 2> {

    // The traits.
    using Traits = Ampere_dmma_fp64_traits;
    // The aclwmulators from the main loop.
    using Aclwmulators = Fragment_aclwmulator<Traits>;


    inline __device__ void colwert(double alpha, const Aclwmulators &acc) {
        // keep it as is.
        this->elt(0) = acc.elt(0);
        this->elt(1) = acc.elt(1);
    }

    inline __device__ void shuffle_groups(Aclwmulators &acc) {
        // Do nothing.
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<Ampere_dmma_fp64_traits, Cta_tile, IN_CTA_SPLIT_K>  : public Fragment<double, 2> {
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
    inline __device__ void gelu_erf(float gelu_scale) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_dmma_fp64_traits, Cta_tile, IN_CTA_SPLIT_K> :  public Fragment<double, 2> {

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
    }

    // The residual is added before store.
    inline __device__ void add_residual(const Fragment_c& res, double beta) {
        //Do beta
        this->elt(0) += beta * res.elt(0);
        this->elt(1) += beta * res.elt(1);
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

#include <xmma/helpers/gemm_tf32.h>



