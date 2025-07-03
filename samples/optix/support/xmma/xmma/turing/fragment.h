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

#include <xmma/turing/traits.h>
#include <xmma/fragment.h>
#include <xmma/utils.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 1 6
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Turing_hmma_fp16_traits, Layout> : public Fragment<lwtlass::half_t, 4> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Turing_hmma_fp16_traits, Layout> : public Fragment<lwtlass::half_t, 4> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Turing_hmma_fp16_traits> : public Fragment<lwtlass::half_t, 8> {

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
        void mma(const Fragment_a<Turing_hmma_fp16_traits, Layout_a> &a,
                 const Fragment_b<Turing_hmma_fp16_traits, Layout_b> &b) {
#if (XMMA_EXTENDED_PTX_ENABLED)
        asm volatile( \
            "_mma.m16n8k8.row.col.f16.f16 \n" \
            "    {%0, %1}, \n" \
            "    {%2, %3}, \n" \
            "    {%4}, \n" \
            "    {%0, %1}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)));
        asm volatile( \
            "_mma.m16n8k8.row.col.f16.f16 \n" \
            "    {%0, %1}, \n" \
            "    {%2, %3}, \n" \
            "    {%4}, \n" \
            "    {%0, %1}; \n" \
                    : "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(1)));
#else
        asm volatile( \
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \n" \
            "    {%0, %1}, \n" \
            "    {%2, %3}, \n" \
            "    {%4}, \n" \
            "    {%0, %1}; \n" \
                    : "+r"(  reg(0)), "+r"(  reg(1))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)));
        asm volatile( \
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \n" \
            "    {%0, %1}, \n" \
            "    {%2, %3}, \n" \
            "    {%4}, \n" \
            "    {%0, %1}; \n" \
                    : "+r"(  reg(2)), "+r"(  reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(1)));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Turing_hmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp16_epilogue_pre_swizzle<Turing_hmma_fp16_traits, Cta_tile> {

    // The traits.
    using Traits = Turing_hmma_fp16_traits;
    // The aclwmulators from the main loop.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    // Colwert from fp16 aclwmulators to fp16 outputs.
    inline __device__ void colwert(lwtlass::half_t alpha, const Aclwmulators &acc) {
        ushort2 alpha_ = make_ushort2(alpha, alpha);

        this->reg(0) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), acc.reg(0));
        this->reg(1) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), acc.reg(2));
        this->reg(2) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), acc.reg(1));
        this->reg(3) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), acc.reg(3));
    }

    // Colwert from fp16 aclwmulators to fp16 outputs.
    inline __device__ void scaled_colwert(lwtlass::half_t alpha, const Aclwmulators &acc) {
        ushort2 alpha_ = make_ushort2(alpha, alpha);

        this->reg(0) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), acc.reg(0));
        this->reg(1) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), acc.reg(2));
        this->reg(2) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), acc.reg(1));
        this->reg(3) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), acc.reg(3));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<Turing_hmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp16_epilogue_post_swizzle<Turing_hmma_fp16_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Turing_hmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp16_c<Turing_hmma_fp16_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Turing_hmma_fp32_traits, Layout> : public Fragment<lwtlass::half_t, 4> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Turing_hmma_fp32_traits, Layout> : public Fragment<lwtlass::half_t, 4> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Turing_hmma_fp32_traits> : public Fragment<float, 8> {

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
        void mma(const Fragment_a<Turing_hmma_fp32_traits, Layout_a> &a,
                 const Fragment_b<Turing_hmma_fp32_traits, Layout_b> &b) {
#if (XMMA_EXTENDED_PTX_ENABLED)
        asm volatile( \
            "_mma.m16n8k8.row.col.f32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)));
        asm volatile( \
            "_mma.m16n8k8.row.col.f32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(1)));
#else
        asm volatile( \
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0)));
        asm volatile( \
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(1)));
#endif
    }

    template< typename Layout_a, typename Layout_b >
    inline __device__
        void mma(const Fragment_a<Turing_hmma_fp32_traits, Layout_a> &a,
                 const Fragment_b<Turing_hmma_fp32_traits, Layout_b> &b,
                 int predicate) {
#if (XMMA_EXTENDED_PTX_ENABLED)
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %7, 0;\n" \
            "@p0 _mma.m16n8k8.row.col.f32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6}, \n" \
            "    {%0, %1, %2, %3}; \n"
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0))
                    ,  "r"(predicate));
        asm volatile( \
            "@p0 _mma.m16n8k8.row.col.f32.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6}, \n" \
            "    {%0, %1, %2, %3}; \n" \
            "}\n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(1)));
        }
#else
        asm volatile( \
            "{\n" \
            ".reg .pred p<1>;\n"
            "setp.gt.s32 p0, %7, 0;\n" \
            "@p0 mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6}, \n" \
            "    {%0, %1, %2, %3}; \n"
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(0))
                    ,  "r"(predicate));
        asm volatile( \
            "@p0 mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5}, \n" \
            "    {%6}, \n" \
            "    {%0, %1, %2, %3}; \n" \
            "}\n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1))
                    ,  "r"(b.reg(1)));
        }

#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Turing_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_epilogue_pre_swizzle<Turing_hmma_fp32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<Turing_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_epilogue_post_swizzle<Turing_hmma_fp32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Turing_hmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_c<Turing_hmma_fp32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout, bool IS_GELU_ERF >
struct Fragment_a<Turing_imma_int8_int32_traits<IS_GELU_ERF>, Layout> : public Fragment<int8_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout, bool IS_GELU_ERF  >
struct Fragment_b<Turing_imma_int8_int32_traits<IS_GELU_ERF>, Layout> : public Fragment<int8_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< bool IS_GELU_ERF >
struct Fragment_aclwmulator<Turing_imma_int8_int32_traits<IS_GELU_ERF>> 
    : public Fragment<int32_t, 8> {

    // The base class.
    using Base = Fragment<int32_t, 8>;

    // The fragments.
    using Fragment_a = xmma::Fragment_a<Turing_imma_int8_int32_traits<IS_GELU_ERF>, Row>;
    using Fragment_b = xmma::Fragment_b<Turing_imma_int8_int32_traits<IS_GELU_ERF>, Col>;

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
struct Fragment_epilogue_pre_swizzle<Turing_imma_int8_int32_traits<IS_GELU_ERF>, 
    Cta_tile, 
    IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_epilogue_pre_swizzle<
        Turing_imma_int8_int32_traits<IS_GELU_ERF>, 
        Cta_tile> {
    // The base class.
    using Base = Fragment_imma_int32_epilogue_pre_swizzle<
        Turing_imma_int8_int32_traits<IS_GELU_ERF>, 
        Cta_tile>;
    // The aclwmulators.
    using Aclwmulators = typename Base::Aclwmulators;

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void scaled_colwert(float &alpha, const Aclwmulators &acc) {
        this->elt(0) = acc.elt(0) * alpha;
        this->elt(1) = acc.elt(1) * alpha;
        this->elt(2) = acc.elt(2) * alpha;
        this->elt(3) = acc.elt(3) * alpha;
        this->elt(4) = acc.elt(4) * alpha;
        this->elt(5) = acc.elt(5) * alpha;
        this->elt(6) = acc.elt(6) * alpha;
        this->elt(7) = acc.elt(7) * alpha;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K, bool IS_GELU_ERF >
struct Fragment_epilogue_post_swizzle<Turing_imma_int8_int32_traits<IS_GELU_ERF>, 
    Cta_tile, 
    IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_epilogue_post_swizzle<Turing_imma_int8_int32_traits<IS_GELU_ERF>, 
    Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K, bool IS_GELU_ERF >
struct Fragment_c<Turing_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_nhwc_int8_c<Turing_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Turing_imma_interleaved_int8_int32_traits, Layout> : public Fragment<int8_t, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Turing_imma_interleaved_int8_int32_traits, Layout> : public Fragment<int8_t, 32> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Turing_imma_interleaved_int8_int32_traits>
    : public Fragment<int32_t, 16> {

    // The base class.
    using Base = Fragment<int32_t, 16>;

    // The fragments.
    using Fragment_a = xmma::Fragment_a<Turing_imma_interleaved_int8_int32_traits, Row>;
    using Fragment_b = xmma::Fragment_b<Turing_imma_interleaved_int8_int32_traits, Col>;

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
    Turing_imma_interleaved_int8_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_interleaved_int32_epilogue_pre_swizzle<
        Turing_imma_interleaved_int8_int32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_interleaved_post_swizzle<
    Turing_imma_interleaved_int8_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_fp32_epilogue_interleaved_post_swizzle<
        Turing_imma_interleaved_int8_int32_traits, Cta_tile, IN_CTA_SPLIT_K> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_interleaved_c<Turing_imma_interleaved_int8_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_interleaved_c<Turing_imma_interleaved_int8_int32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 4
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Turing_imma_int4_int32_traits, Layout> : public Fragment<lwtlass::int4_t, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Turing_imma_int4_int32_traits, Layout> : public Fragment<lwtlass::int4_t, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Turing_imma_int4_int32_traits> : public Fragment<int32_t, 8> {

    // The base class.
    using Base = Fragment<int32_t, 8>;

    // The fragments.
    using Fragment_a = xmma::Fragment_a<Turing_imma_int4_int32_traits, Row>;
    using Fragment_b = xmma::Fragment_b<Turing_imma_int4_int32_traits, Col>;

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
                "_mma.m8n8k32.row.col.s4.s4 \n" \
                "    {%0, %1}, \n" \
                "     %2, \n" \
                "     %3, \n" \
                "    {%0, %1}; \n" \
                    : "+r"(this->reg(2*i+0))
                    , "+r"(this->reg(2*i+1))
                    :  "r"(    a.reg(i/2))
                    ,  "r"(    b.reg(i%2)));
#else
            asm volatile( \
                "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 \n" \
                "    {%0, %1}, \n" \
                "     %2, \n" \
                "     %3, \n" \
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

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Turing_imma_int4_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_epilogue_pre_swizzle<Turing_imma_int4_int32_traits, Cta_tile> {

    using Aclwmulators = Fragment_aclwmulator<Turing_imma_int4_int32_traits>;

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    // Covert s32 to f32
    // FIXME: alpha should be float
    inline __device__ void colwert(int32_t &alpha, const Aclwmulators &acc) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<Turing_imma_int4_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_epilogue_post_swizzle<Turing_imma_int4_int32_traits, Cta_tile> {

    // FIXME: alpha should be float
    inline __device__ void reduce(int32_t &alpha) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Turing_imma_int4_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_c<Turing_imma_int4_int32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// B M M A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Turing_bmma_int32_traits, Layout> : public Fragment<bool, 64> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Turing_bmma_int32_traits, Layout> : public Fragment<bool, 64> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Turing_bmma_int32_traits> : public Fragment<int32_t, 8> {

    // The base class.
    using Base = Fragment<int32_t, 8>;

    // The fragments.
    using Fragment_a = xmma::Fragment_a<Turing_bmma_int32_traits, Row>;
    using Fragment_b = xmma::Fragment_b<Turing_bmma_int32_traits, Col>;

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
                "_mma.m8n8k128.row.col.s32.b1.b1.s32.xor.popc \n" \
                "    {%0, %1}, \n" \
                "     %2, \n" \
                "     %3, \n" \
                "    {%0, %1}; \n" \
                    : "+r"(this->reg(2*i+0))
                    , "+r"(this->reg(2*i+1))
                    :  "r"(    a.reg(i/2))
                    ,  "r"(    b.reg(i%2)));
#else
            asm volatile( \
                "wmma.mma.xor.popc.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32 \n" \
                "    {%0, %1}, \n" \
                "     %2, \n" \
                "     %3, \n" \
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

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_pre_swizzle<Turing_bmma_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_epilogue_pre_swizzle<Turing_bmma_int32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_epilogue_post_swizzle<Turing_bmma_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_epilogue_post_swizzle<Turing_bmma_int32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Turing_bmma_int32_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_imma_int32_c<Turing_bmma_int32_traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

