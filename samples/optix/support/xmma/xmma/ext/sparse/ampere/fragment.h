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

#include <xmma/ampere/fragment.h>
#include <xmma/ext/sparse/ampere/traits.h>
#include <xmma/ext/sparse/fragment.h>

#ifndef XMMA_PTX_MMA_SPARSE_ENABLED
#define XMMA_PTX_MMA_SPARSE_ENABLED 0
#endif

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  H M M A . F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Ampere_sphmma_fp32_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Ampere_sphmma_fp32_traits, Layout> : public Fragment<lwtlass::half_t, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_e<Ampere_sphmma_fp32_traits, Layout> : public Fragment<lwtlass::half_t, 4> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Ampere_sphmma_fp32_traits> : public Fragment<float, 8> {
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
        void mma(const Fragment_a<Ampere_sphmma_fp32_traits, Layout_a> &a,
                 const Fragment_b<Ampere_sphmma_fp32_traits, Layout_b> &b,
                 const uint32_t e, int thread) {
        if (thread == 0) {
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
        asm volatile( \
            "_mma.sp.thread.m16n8k32.row.col.f32.f16.f16.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+f"(   elt(0)), "+f"(   elt(1)), "+f"(   elt(2)), "+f"(   elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "_mma.sp.thread.m16n8k32.row.col.f32.f16.f16.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+f"(   elt(4)), "+f"(   elt(5)), "+f"(   elt(6)), "+f"(   elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#else
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+f"(   elt(0)), "+f"(   elt(1)), "+f"(   elt(2)), "+f"(   elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+f"(   elt(4)), "+f"(   elt(5)), "+f"(   elt(6)), "+f"(   elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#endif
        } else {
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
        asm volatile( \
            "_mma.sp.thread.m16n8k32.row.col.f32.f16.f16.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 1;" \
                    : "+f"(   elt(0)), "+f"(   elt(1)), "+f"(   elt(2)), "+f"(   elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "_mma.sp.thread.m16n8k32.row.col.f32.f16.f16.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 1;" \
                    : "+f"(   elt(4)), "+f"(   elt(5)), "+f"(   elt(6)), "+f"(   elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#else
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 1;" \
                    : "+f"(   elt(0)), "+f"(   elt(1)), "+f"(   elt(2)), "+f"(   elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 1;" \
                    : "+f"(   elt(4)), "+f"(   elt(5)), "+f"(   elt(6)), "+f"(   elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#endif
        }
    }

    template< typename Layout_a, typename Layout_b, typename Layout_e>
    //template< typename Layout_a, typename Layout_b>
    inline __device__ 
        void spmma_s0(const Fragment_a<Ampere_sphmma_fp32_traits, Layout_a> &a, 
                      const Fragment_b<Ampere_sphmma_fp32_traits, Layout_b> &b,
                    const Fragment_e<Ampere_sphmma_fp32_traits, Layout_e> &e){
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(0)));
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(0)));
#else
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(0)));
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(0)));
#endif
    }                   
    template< typename Layout_a, typename Layout_b, typename Layout_e>
    //template< typename Layout_a, typename Layout_b>
    inline __device__ 
        void spmma_s1(const Fragment_a<Ampere_sphmma_fp32_traits, Layout_a> &a, 
                      const Fragment_b<Ampere_sphmma_fp32_traits, Layout_b> &b,
                    const Fragment_e<Ampere_sphmma_fp32_traits, Layout_e> &e){
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0) 
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(0)));
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(0)));
#else
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(0)));
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(0)));
#endif
    }
    template< typename Layout_a, typename Layout_b, typename Layout_e>
    //template< typename Layout_a, typename Layout_b>
    inline __device__ 
        void spmma_s2(const Fragment_a<Ampere_sphmma_fp32_traits, Layout_a> &a, 
                      const Fragment_b<Ampere_sphmma_fp32_traits, Layout_b> &b,
                    const Fragment_e<Ampere_sphmma_fp32_traits, Layout_e> &e){
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0) 
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(1)));
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(1)));
#else
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(1)));
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(1)));
#endif
    }

    template< typename Layout_a, typename Layout_b, typename Layout_e>
    //template< typename Layout_a, typename Layout_b>
    inline __device__ 
        void spmma_s3(const Fragment_a<Ampere_sphmma_fp32_traits, Layout_a> &a, 
                      const Fragment_b<Ampere_sphmma_fp32_traits, Layout_b> &b,
                    const Fragment_e<Ampere_sphmma_fp32_traits, Layout_e> &e){
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)), 
                    "r"(e.reg(1)));
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(1)));
#else
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)), 
                    "r"(e.reg(1)));
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(1)));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_pre_swizzle<Ampere_sphmma_fp32_traits, Cta_tile, true>
    : public Fragment_hmma_fp32_epilogue_pre_swizzle<Ampere_sphmma_fp32_traits, Cta_tile> {

    // The traits.
    using Traits = Ampere_sphmma_fp32_traits;
    // The aclwmulators from the main loop.
    using Aclwmulators = Fragment_aclwmulator<Traits>;
    
    inline __device__ void shuffle_groups(Aclwmulators &acc) {
        // C=K=8
        if ( Cta_tile::GROUPS ==  8 ){
            acc.reg(2) = acc.reg(6);
            acc.reg(3) = acc.reg(7);
        }
        // C=K=4
        if ( Cta_tile::GROUPS == 16 ){
            acc.reg(2) = acc.reg(6);
            acc.reg(3) = acc.reg(7);
            const unsigned mask = 0xFFFFFFFF;
            int lane = threadIdx.x % 32;
            int src_lane = lane;
            if ( lane >= 16 && lane % 4 < 2 ) {
                src_lane = lane + 2;
            }
            #pragma unroll
            for ( int k = 0; k < 4; ++k) {
                acc.reg(k) = __shfl_sync(mask, acc.reg(k), src_lane);
            }
        }
        // C=K=16 and C=K=32 and non-group colwolution cases, do nothing.
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_pre_swizzle<Ampere_sphmma_fp32_traits, Cta_tile, false> 
    : public Fragment_hmma_fp32_epilogue_pre_swizzle<Ampere_sphmma_fp32_traits, Cta_tile> {
    //: public Fragment_hmma_fp16_epilogue_pre_swizzle<Ampere_sphmma_fp32_traits, Cta_tile> {

    // The traits.
    using Traits = Ampere_sphmma_fp32_traits;
    // The aclwmulators from the main loop.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    inline __device__ void colwert(float alpha, const Aclwmulators &acc) {
        this->reg(0) = acc.reg(0);
        this->reg(1) = acc.reg(1);
        this->reg(2) = acc.reg(4);
        this->reg(3) = acc.reg(5);
        this->reg(4) = acc.reg(2);
        this->reg(5) = acc.reg(3);
        this->reg(6) = acc.reg(6);
        this->reg(7) = acc.reg(7);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_post_swizzle<Ampere_sphmma_fp32_traits, Cta_tile, true> 
    : public Fragment_hmma_fp32_epilogue_post_swizzle<Ampere_sphmma_fp32_traits, Cta_tile> {

    template< typename Fragment_alpha >
    inline __device__ void scale(float alpha, Fragment_alpha alpha_frag) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            this->elt(i) *= alpha;
        }
    }

    template< typename Fragment_res, typename Fragment_beta >
    inline __device__ void residual(float beta, Fragment_res frag, Fragment_beta beta_frag) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 tmp = xmma::half2_to_float2(frag.reg(i));
            this->elt(i*2) += tmp.x * beta;
            this->elt(i*2+1) += tmp.y * beta;
        }
    }

    template< typename Fragment_ >
    inline __device__ void add_bias(Fragment_ frag) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 tmp = xmma::half2_to_float2(frag.reg(i));
            this->elt(i*2) += tmp.x;
            this->elt(i*2+1) += tmp.y;
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_post_swizzle<Ampere_sphmma_fp32_traits, Cta_tile, false> 
    : public Fragment_hmma_fp32_epilogue_post_swizzle<Ampere_sphmma_fp32_traits, Cta_tile> {

    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, float beta) {
        #pragma unroll
        for( int i = 0; i < 4; ++i ) {
            float2 tmp = half2_to_float2(res_.reg(i));
            this->elt(2 * i) = this->elt(2 * i) + beta * tmp.x;
            this->elt(2 * i + 1) = this->elt(2 * i + 1) + beta * tmp.y;
        }
    }

    // The bias.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias_) {
        #pragma unroll
        for( int i = 0; i < 4; ++i ) {
            float2 tmp = half2_to_float2(bias_.reg(i));
            this->elt(2 * i) = this->elt(2 * i) + tmp.x;
            this->elt(2 * i + 1) = this->elt(2 * i + 1) + tmp.y;
        }
    }

    inline __device__ void add_single_bias(uint16_t bias) {
        float f32_bias = half_to_float(bias);
        #pragma unroll
        for( int i = 0; i < Fragment_epilogue_post_swizzle::NUM_REGS; ++i ) {
            this->elt(i) = this->elt(i) + f32_bias;
        }
    }

    // fp32 ReLu.
    inline __device__ void relu(float relu_lb) {
        #pragma unroll
        for( int i = 0; i < Fragment_epilogue_post_swizzle::NUM_REGS; ++i ) {
            this->elt(i) = reluTh_fp32( this->elt(i), relu_lb );
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_sphmma_fp32_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_hmma_fp32_c<Ampere_sphmma_fp32_traits, Cta_tile> {

    // Pack two fragments.
    template< typename Fragment_post_swizzle >
    inline __device__ void pack_sparse(float alpha,
            Fragment_post_swizzle frag) {
        this->reg(0) = float2_to_half2(frag.elt(0), frag.elt(1));
        this->reg(1) = float2_to_half2(frag.elt(2), frag.elt(3));
        this->reg(2) = float2_to_half2(frag.elt(4), frag.elt(5));
        this->reg(3) = float2_to_half2(frag.elt(6), frag.elt(7));
    }

    // ReLu.
    inline __device__ void relu(float relu_lb) {
        half2 tmp = xmma::colwert<half2>(relu_lb);
        uint32_t relu_lb_ = reinterpret_cast<uint32_t&>(tmp);

        this->reg(0) = xmma::reluTh_fp16x2(this->reg(0), relu_lb_);
        this->reg(1) = xmma::reluTh_fp16x2(this->reg(1), relu_lb_);
        this->reg(2) = xmma::reluTh_fp16x2(this->reg(2), relu_lb_);
        this->reg(3) = xmma::reluTh_fp16x2(this->reg(3), relu_lb_);
    }

    inline __device__ void add_bias_v2(uint16_t bias) {
        #pragma unroll
        for( int i = 0; i < Fragment_c::NUM_REGS; ++i ) {
            uint32_t bias2 = half_to_half2(bias);
            this->reg(i) = hadd2(this->reg(i), bias2);
        }
    }

    template< typename Fragment_post_swizzle >
    inline __device__ void pack(float, const Fragment_post_swizzle &frag) {
        this->reg(0) = float2_to_half2(frag.elt(0), frag.elt(1));
        this->reg(1) = float2_to_half2(frag.elt(2), frag.elt(3));
        this->reg(2) = float2_to_half2(frag.elt(4), frag.elt(5));
        this->reg(3) = float2_to_half2(frag.elt(6), frag.elt(7));
    }

    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, float beta) {
        // Add residual in fp32 post swizzle
        // Need a empty function to bypass
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  H M M A . B F 1 6 . F 3 2 . B F 1 6
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Ampere_sphmma_bf16_fp32_bf16_traits, Layout> : public Fragment<lwtlass::float_bf16_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Ampere_sphmma_bf16_fp32_bf16_traits, Layout> : public Fragment<lwtlass::float_bf16_t, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_e<Ampere_sphmma_bf16_fp32_bf16_traits, Layout> : public Fragment<lwtlass::half_t, 4> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Ampere_sphmma_bf16_fp32_bf16_traits> : public Fragment<float, 8> {
    // The base class.
    using Base = Fragment<float, 8>;

    // The traits.
    using Traits = Ampere_sphmma_bf16_fp32_bf16_traits;


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
        void mma(const Fragment_a<Traits, Layout_a> &a,
                 const Fragment_b<Traits, Layout_b> &b,
                 const uint32_t e, int thread) {
    }

    template< typename Layout_a, typename Layout_b, typename Layout_e>
    //template< typename Layout_a, typename Layout_b>
    inline __device__ 
        void spmma_s0(const Fragment_a<Traits, Layout_a> &a, 
                      const Fragment_b<Traits, Layout_b> &b,
                      const Fragment_e<Traits, Layout_e> &e){
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(0)));
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(0)));
#else
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(0)));
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32  \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(0)));
#endif
    }

    template< typename Layout_a, typename Layout_b, typename Layout_e>
    //template< typename Layout_a, typename Layout_b>
    inline __device__ 
        void spmma_s1(const Fragment_a<Traits, Layout_a> &a, 
                      const Fragment_b<Traits, Layout_b> &b,
                      const Fragment_e<Traits, Layout_e> &e){
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(0)));
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(0)));
#else
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(0)));
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(0)));
#endif
    }

    template< typename Layout_a, typename Layout_b, typename Layout_e>
    //template< typename Layout_a, typename Layout_b>
    inline __device__ 
        void spmma_s2(const Fragment_a<Traits, Layout_a> &a, 
                      const Fragment_b<Traits, Layout_b> &b,
                      const Fragment_e<Traits, Layout_e> &e){
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(1)));
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(1)));
#else
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(1)));
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(1)));
#endif
    }

    template< typename Layout_a, typename Layout_b, typename Layout_e>
    //template< typename Layout_a, typename Layout_b>
    inline __device__ 
        void spmma_s3(const Fragment_a<Traits, Layout_a> &a, 
                      const Fragment_b<Traits, Layout_b> &b,
                      const Fragment_e<Traits, Layout_e> &e){
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)), 
                    "r"(e.reg(1)));
            asm volatile("_mma.sp.thread.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(1)));
#else
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)), 
                    "r"(e.reg(1)));
            asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(1)));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_pre_swizzle<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile, true>
    : public Fragment_hmma_fp32_epilogue_pre_swizzle<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile> {

    // The traits.
    using Traits = Ampere_sphmma_bf16_fp32_bf16_traits;
    // The aclwmulators from the main loop.
    using Aclwmulators = Fragment_aclwmulator<Traits>;
    
    inline __device__ void shuffle_groups(Aclwmulators &acc) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_pre_swizzle<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile, false> 
    : public Fragment_hmma_fp32_epilogue_pre_swizzle<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile> {
    //: public Fragment_hmma_fp16_epilogue_pre_swizzle<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile> {

    // The traits.
    using Traits = Ampere_sphmma_bf16_fp32_bf16_traits;
    // The aclwmulators from the main loop.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    // Colwert from fp16 aclwmulators to fp16 outputs.
    inline __device__ void colwert(float alpha, const Aclwmulators &acc) {
        this->reg(0) = acc.reg(0);
        this->reg(1) = acc.reg(1);
        this->reg(2) = acc.reg(4);
        this->reg(3) = acc.reg(5);
        this->reg(4) = acc.reg(2);
        this->reg(5) = acc.reg(3);
        this->reg(6) = acc.reg(6);
        this->reg(7) = acc.reg(7);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_post_swizzle<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile, true> 
    : public Fragment_hmma_fp32_epilogue_post_swizzle<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile> {
    
    // Empty add_bias.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias_) { }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_post_swizzle<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile, false> 
    : public Fragment_hmma_fp32_epilogue_post_swizzle<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile> {

    // Empty add_bias.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias_) { 
        #pragma unroll
        for( int i = 0; i < 4; ++i ) {
            float2 tmp = bf16_2_to_float2(bias_.reg(i));
            this->elt(2 * i) = this->elt(2 * i) + tmp.x;
            this->elt(2 * i + 1) = this->elt(2 * i + 1) + tmp.y;
        }
    }

    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, float beta) {
        #pragma unroll
        for( int i = 0; i < Fragment_c::NUM_REGS ; ++i ) {
            float2 tmp = bf16_2_to_float2(res_.reg(i));
            this->elt(2 * i) = this->elt(2 * i) + beta * tmp.x;
            this->elt(2 * i + 1) = this->elt(2 * i + 1) + beta * tmp.y;
        }
    }

    inline __device__ void add_single_bias(uint16_t bias) {
        float f32_bias = bf16_to_float(bias);
        #pragma unroll
        for( int i = 0; i < Fragment_epilogue_post_swizzle::NUM_REGS; ++i ) {
            this->elt(i) = this->elt(i) + f32_bias;
        }
    }

    // fp32 ReLu.
    inline __device__ void relu(float relu_lb) {
        #pragma unroll
        for( int i = 0; i < Fragment_epilogue_post_swizzle::NUM_REGS; ++i ) {
            this->elt(i) = reluTh_fp32( this->elt(i), relu_lb );
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_hmma_fp32_c<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile> {

    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, float beta) {
        // Keep this in case we need bf16
        //uint4 res = res_.to_int4();
        //uint32_t beta_beta = float2_to_bf16_2(beta, beta);
        //this->reg(0) = bfma2(beta_beta, res.x, this->reg(0));
        //this->reg(1) = bfma2(beta_beta, res.y, this->reg(1));
        //this->reg(2) = bfma2(beta_beta, res.z, this->reg(2));
        //this->reg(3) = bfma2(beta_beta, res.w, this->reg(3));
    }

    // The bias.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias_) {
        uint32_t one_one = float2_to_bf16_2(1.0f, 1.0f);
        this->reg(0) = bfma2(one_one, bias_.reg(0), this->reg(0));
        this->reg(1) = bfma2(one_one, bias_.reg(1), this->reg(1));
        this->reg(2) = bfma2(one_one, bias_.reg(2), this->reg(2));
        this->reg(3) = bfma2(one_one, bias_.reg(3), this->reg(3));
    }

    // ReLu.
    inline __device__ void relu(float relu_lb) {
        uint32_t relu_lb_ = float2_to_bf16_2(relu_lb, relu_lb);
        this->reg(0) = relu_bf16x2(this->reg(0), relu_lb_);
        this->reg(1) = relu_bf16x2(this->reg(1), relu_lb_);
        this->reg(2) = relu_bf16x2(this->reg(2), relu_lb_);
        this->reg(3) = relu_bf16x2(this->reg(3), relu_lb_);
    }

    inline __device__ void add_bias_v2(uint16_t bias) {
        #pragma unroll
        for( int i = 0; i < Fragment_c::NUM_REGS; ++i ) {
            uint32_t bias2 = half_to_half2(bias);
            this->reg(i) = bfadd2(this->reg(i), bias2);
    }
    }

    template< typename Fragment_post_swizzle >
    inline __device__ void pack(float, const Fragment_post_swizzle &frag) {
        this->reg(0) = float2_to_bf16_2(reinterpret_cast<float const &>(frag.reg(0)), 
                                        reinterpret_cast<float const &>(frag.reg(1)));

        this->reg(1) = float2_to_bf16_2(reinterpret_cast<float const &>(frag.reg(2)), 
                                        reinterpret_cast<float const &>(frag.reg(3)));

        this->reg(2) = float2_to_bf16_2(reinterpret_cast<float const &>(frag.reg(4)), 
                                        reinterpret_cast<float const &>(frag.reg(5)));

        this->reg(3) = float2_to_bf16_2(reinterpret_cast<float const &>(frag.reg(6)), 
                                        reinterpret_cast<float const &>(frag.reg(7)));

    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  H M M A . F 1 6
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Ampere_sphmma_fp16_traits, Layout> : public Fragment<lwtlass::half_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Ampere_sphmma_fp16_traits, Layout> : public Fragment<lwtlass::half_t, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_e<Ampere_sphmma_fp16_traits, Layout> : public Fragment<lwtlass::half_t, 4> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Ampere_sphmma_fp16_traits> : public Fragment<lwtlass::half_t, 8> {
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
        void mma(const Fragment_a<Ampere_sphmma_fp16_traits, Layout_a> &a,
                 const Fragment_b<Ampere_sphmma_fp16_traits, Layout_b> &b,
                 const uint32_t e, int thread) {
        if (thread == 0) {
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
        asm volatile( \
            "_mma.sp.thread.m16n8k32.row.col.f16.f16.f16.f16" \
            "    {%0, %1}," \
            "    {%2, %3, %4, %5}," \
            "    {%6, %7, %8, %9}," \
            "    {%0, %1}, %10, 0;" \
                    : "+r"(   reg(0)), "+r"(   reg(1))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "_mma.sp.thread.m16n8k32.row.col.f16.f16.f16.f16" \
            "    {%0, %1}," \
            "    {%2, %3, %4, %5}," \
            "    {%6, %7, %8, %9}," \
            "    {%0, %1}, %10, 0;" \
                    : "+r"(   reg(2)), "+r"(   reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#else
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16" \
            "    {%0, %1}," \
            "    {%2, %3, %4, %5}," \
            "    {%6, %7, %8, %9}," \
            "    {%0, %1}, %10, 0;" \
                    : "+r"(   reg(0)), "+r"(   reg(1))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16" \
            "    {%0, %1}," \
            "    {%2, %3, %4, %5}," \
            "    {%6, %7, %8, %9}," \
            "    {%0, %1}, %10, 0;" \
                    : "+r"(   reg(2)), "+r"(   reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#endif
        } else {
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
        asm volatile( \
            "_mma.sp.thread.m16n8k32.row.col.f16.f16.f16.f16" \
            "    {%0, %1}," \
            "    {%2, %3, %4, %5}," \
            "    {%6, %7, %8, %9}," \
            "    {%0, %1}, %10, 1;" \
                    : "+r"(   reg(0)), "+r"(   reg(1))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "_mma.sp.thread.m16n8k32.row.col.f16.f16.f16.f16" \
            "    {%0, %1}," \
            "    {%2, %3, %4, %5}," \
            "    {%6, %7, %8, %9}," \
            "    {%0, %1}, %10, 1;" \
                    : "+r"(   reg(2)), "+r"(   reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#else
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16" \
            "    {%0, %1}," \
            "    {%2, %3, %4, %5}," \
            "    {%6, %7, %8, %9}," \
            "    {%0, %1}, %10, 1;" \
                    : "+r"(   reg(0)), "+r"(   reg(1))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16" \
            "    {%0, %1}," \
            "    {%2, %3, %4, %5}," \
            "    {%6, %7, %8, %9}," \
            "    {%0, %1}, %10, 1;" \
                    : "+r"(   reg(2)), "+r"(   reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#endif
        }
    }
};

template< typename Cta_tile >
struct Fragment_epilogue_post_swizzle<Ampere_sphmma_fp16_traits, Cta_tile, true> 
    : public Fragment_hmma_fp16_epilogue_post_swizzle<Ampere_sphmma_fp16_traits, Cta_tile> {

    template< typename Fragment_alpha >
    inline __device__ void scale(lwtlass::half_t alpha, Fragment_alpha alpha_frag) {
        uint32_t alpha_v2;
        asm volatile("mov.b32 %0, {%1, %1};\n" : "=r"(alpha_v2) : "h"(alpha));

        this->reg(0) = hmul2(alpha_v2, this->reg(0));
        this->reg(1) = hmul2(alpha_v2, this->reg(1));
        this->reg(2) = hmul2(alpha_v2, this->reg(2));
        this->reg(3) = hmul2(alpha_v2, this->reg(3));
    }

    template< typename Fragment_res, typename Fragment_beta >
    inline __device__ void residual(lwtlass::half_t beta, Fragment_res frag,
        Fragment_beta beta_frag) {
        uint32_t beta_v2;
        asm volatile("mov.b32 %0, {%1, %1};\n" : "=r"(beta_v2) : "h"(beta));

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            this->reg(i) = hfma2(beta_v2, frag.reg(i), this->reg(i));
        }
    }

    template< typename Fragment_ >
    inline __device__ void add_bias(Fragment_ frag) {

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            this->reg(i) = hadd2(frag.reg(i), this->reg(i));
        }
    }

};

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_sphmma_fp16_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_hmma_fp16_c<Ampere_sphmma_fp16_traits, Cta_tile> {

    // Pack two fragments.
    template< typename Fragment_post_swizzle >
    inline __device__ void pack_sparse(float alpha,
            Fragment_post_swizzle &frag) {
        this->reg(0) = frag.reg(0);
        this->reg(1) = frag.reg(1);
        this->reg(2) = frag.reg(2);
        this->reg(3) = frag.reg(3);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  H M M A . T F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type, typename Layout >
struct Fragment_a<Ampere_sphmma_tf32_traits<Input_type, Output_type>, Layout>
    : public Fragment<uint32_t, 4> {
    public:
        using Input_type_ = Input_type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type, typename Layout >
struct Fragment_b<Ampere_sphmma_tf32_traits<Input_type, Output_type>, Layout>
    : public Fragment<uint32_t, 8> {
    public:
        using Input_type_ = Input_type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type, typename Layout >
struct Fragment_e<Ampere_sphmma_tf32_traits<Input_type, Output_type>, Layout>
    : public Fragment<lwtlass::half_t, 4> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type >
struct Fragment_aclwmulator<Ampere_sphmma_tf32_traits<Input_type, Output_type>>
    : public Fragment<float, 8> {
    // The traits.
    using Traits = Ampere_sphmma_tf32_traits<Input_type, Output_type>;

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
        void mma(const Fragment_a<Traits, Layout_a> &a,
                 const Fragment_b<Traits, Layout_b> &b,
                 const uint32_t e, int thread) {
        if (thread == 0) {
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
        asm volatile( \
            "_mma.sp.thread.m16n8k16.row.col.f32.tf32.tf32.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+f"(   elt(0)), "+f"(   elt(1)), "+f"(   elt(2)), "+f"(   elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "_mma.sp.thread.m16n8k16.row.col.f32.tf32.tf32.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+f"(   elt(4)), "+f"(   elt(5)), "+f"(   elt(6)), "+f"(   elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#else
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+f"(   elt(0)), "+f"(   elt(1)), "+f"(   elt(2)), "+f"(   elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+f"(   elt(4)), "+f"(   elt(5)), "+f"(   elt(6)), "+f"(   elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#endif
        } else {
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
        asm volatile( \
            "_mma.sp.thread.m16n8k16.row.col.f32.tf32.tf32.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 1;" \
                    : "+f"(   elt(0)), "+f"(   elt(1)), "+f"(   elt(2)), "+f"(   elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "_mma.sp.thread.m16n8k16.row.col.f32.tf32.tf32.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 1;" \
                    : "+f"(   elt(4)), "+f"(   elt(5)), "+f"(   elt(6)), "+f"(   elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#else
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 1;" \
                    : "+f"(   elt(0)), "+f"(   elt(1)), "+f"(   elt(2)), "+f"(   elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 1;" \
                    : "+f"(   elt(4)), "+f"(   elt(5)), "+f"(   elt(6)), "+f"(   elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#endif
        }
    }

    template< typename Layout_a, typename Layout_b, typename Layout_e>
    inline __device__ 
        void spmma_s0(const Fragment_a<Traits, Layout_a> &a, 
                      const Fragment_b<Traits, Layout_b> &b,
                      const Fragment_e<Traits, Layout_e> &e){
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
            asm volatile("_mma.sp.thread.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(0)));
            asm volatile("_mma.sp.thread.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(0)));
#else
            asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(0)));
            asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(0)));     
#endif    

    }
    template< typename Layout_a, typename Layout_b, typename Layout_e>
    inline __device__ 
        void spmma_s1(const Fragment_a<Traits, Layout_a> &a, 
                      const Fragment_b<Traits, Layout_b> &b,
                      const Fragment_e<Traits, Layout_e> &e){
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0) 
            asm volatile("_mma.sp.thread.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(0)));
            asm volatile("_mma.sp.thread.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(0)));
#else
            asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(0)));
            asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(0)));   
#endif       
    }
    template< typename Layout_a, typename Layout_b, typename Layout_e>
    inline __device__ 
        void spmma_s2(const Fragment_a<Traits, Layout_a> &a, 
                      const Fragment_b<Traits, Layout_b> &b,
                      const Fragment_e<Traits, Layout_e> &e){
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
            asm volatile("_mma.sp.thread.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(1)));
            asm volatile("_mma.sp.thread.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(1)));
#else
            asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)),
                    "r"(e.reg(1)));
            asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x0; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(1)));    
#endif     
    }
    template< typename Layout_a, typename Layout_b, typename Layout_e>
    inline __device__ 
        void spmma_s3(const Fragment_a<Traits, Layout_a> &a, 
                      const Fragment_b<Traits, Layout_b> &b,
                      const Fragment_e<Traits, Layout_e> &e){
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)
            asm volatile("_mma.sp.thread.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)), 
                    "r"(e.reg(1)));
            asm volatile("_mma.sp.thread.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(1)));
#else
            asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3)), 
                    "r"(e.reg(1)));
            asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 \n" \
                " {%0, %1, %2, %3}, \n" \
                " {%4, %5, %6, %7}, \n" \
                " {%8, %9, %10, %11}, \n" \
                " {%0, %1, %2, %3}, %12, 0x1; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    : "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3)),  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7)), 
                    "r"(e.reg(1))); 
#endif         
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type, typename Cta_tile, bool IN_CTA_SPLIT_K>
struct Fragment_epilogue_pre_swizzle<Ampere_sphmma_tf32_traits<Input_type, Output_type>,
    Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_hmma_fp32_epilogue_pre_swizzle<
    Ampere_sphmma_tf32_traits<Input_type, Output_type>, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type, typename Cta_tile >
struct Fragment_epilogue_post_swizzle<Ampere_sphmma_tf32_traits<Input_type, Output_type>,
    Cta_tile, true> 
    : public Fragment_hmma_fp32_epilogue_post_swizzle<
        Ampere_sphmma_tf32_traits<Input_type, Output_type>, Cta_tile> {

    template< typename Fragment_alpha >
    inline __device__ void scale(float alpha, Fragment_alpha alpha_frag) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            this->elt(i) *= alpha;
        }
    }

    template< typename Fragment_res, typename Fragment_beta >
    inline __device__ void residual(float beta, Fragment_res frag, Fragment_beta beta_frag) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            this->elt(i) += frag.elt(i) * beta;
        }
    }

    template< typename Fragment_ >
    inline __device__ void add_bias(Fragment_ frag) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            this->elt(i) += frag.elt(i);
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type, typename Cta_tile >
struct Fragment_epilogue_post_swizzle<Ampere_sphmma_tf32_traits<Input_type, Output_type>,
    Cta_tile, false> 
    : public Fragment_hmma_fp32_epilogue_post_swizzle<
        Ampere_sphmma_tf32_traits<Input_type, Output_type>, Cta_tile> {

    // The residual is added later.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, float beta) {
    }

    // The bias is added later.
    template< typename Fragment_ >
    inline __device__ void add_bias(Fragment_ frag) {
    }

    // The relu is done later.
    inline __device__ void relu(float relu_lb=0.f) {
    }

    // The clip relu is done later.
    inline __device__ void relu_ub(float relu_ub) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits >
struct Fragment_sparse_tf32_c : public Fragment<float, 8> {

    // The base class.
    using Base = Fragment<float, 8>;

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

template< typename Input_type, typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_sphmma_tf32_traits<Input_type, lwtlass::float_tf32_t>,
    Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_sparse_tf32_c<
        Ampere_sphmma_tf32_traits<Input_type, lwtlass::float_tf32_t>> {

    // Pack two fragments.
    template< typename Fragment_post_swizzle >
    inline __device__ void pack_sparse(float alpha,
            Fragment_post_swizzle frag) {

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float x = reinterpret_cast<float const &>(frag.reg(i)); 
            this->reg(i) = xmma::colwert_tf32(x);
        }
    }

    // The residual is added later.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, float beta) { 
        #pragma unroll
        for( int ii = 0; ii < Fragment_c::NUM_ELTS;  ++ii ) {
            this->elt(ii) = this->elt(ii) + beta * res_.elt(ii);
        }
    }

    // Add two fragments together.
    template< typename Other_fragment >
    inline __device__ void add(const Other_fragment &other) {
    }

    // Colwert from a post-swizzle fragment.
    template< typename Fragment_post_swizzle >
    inline __device__ void pack(float, const Fragment_post_swizzle &frag) {
        #pragma unroll
        for (int i = 0; i < Fragment_post_swizzle::NUM_ELTS; i++) {
            this->reg(i) = frag.reg(i);
        }
    }

    template< typename Fragment_bias >
    inline __device__ void add_bias(Fragment_bias frag) {
        #pragma unroll
        for (int i = 0; i < Fragment_bias::NUM_ELTS; i++) {
            this->elt(i) += frag.elt(i);
        }
    }

    inline __device__ void add_single_bias(uint32_t bias) {
        float x = reinterpret_cast<float const &>(bias); 
        #pragma unroll
        for (int i = 0; i < Fragment_c::NUM_REGS; i++) {
            this->elt(i) += x;
        }
    }

    inline __device__ void relu(float relu_lb) {
        #pragma unroll
        for (int i = 0; i < Fragment_c::NUM_REGS; i++) {
            this->elt(i) = xmma::reluTh_fp32(this->elt(i), relu_lb);
        }
    }

    inline __device__ void output_colwert() {
        #pragma unroll
        for (int i = 0; i < Fragment_c::NUM_REGS; i++) {
            this->reg(i) = xmma::colwert_tf32(this->elt(i));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_sphmma_tf32_traits<Input_type, float>,
    Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_sparse_tf32_c<
        Ampere_sphmma_tf32_traits<Input_type, float>> {

    // Pack two fragments.
    template< typename Fragment_post_swizzle >
    inline __device__ void pack_sparse(float alpha,
            Fragment_post_swizzle frag) {

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            this->reg(i) = frag.reg(i);
        }
    }

    // The residual is added later.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, float beta) {
        #pragma unroll
        for( int ii = 0; ii < Fragment_c::NUM_ELTS; ++ii ) {
            this->elt(ii) = this->elt(ii) + beta * res_.elt(ii);
        }
    }

    // Add two fragments together.
    template< typename Other_fragment >
    inline __device__ void add(const Other_fragment &other) {
    }

    // Colwert from a post-swizzle fragment.
    template< typename Fragment_post_swizzle >
    inline __device__ void pack(float, const Fragment_post_swizzle &frag) {
        #pragma unroll
        for (int i = 0; i < Fragment_post_swizzle::NUM_ELTS; i++) {
            this->reg(i) = frag.reg(i);
        }
    }

    template< typename Fragment_bias >
    inline __device__ void add_bias(Fragment_bias frag) {
        #pragma unroll
        for (int i = 0; i < Fragment_bias::NUM_ELTS; i++) {
            this->elt(i) += frag.elt(i);
        }
    }

    inline __device__ void add_single_bias(uint32_t bias) {
        float x = reinterpret_cast<float const &>(bias); 
        #pragma unroll
        for (int i = 0; i < Fragment_c::NUM_REGS; i++) {
            this->elt(i) += x;
        }
    }

    inline __device__ void relu(float relu_lb) {
        #pragma unroll
        for (int i = 0; i < Fragment_c::NUM_REGS; i++) {
            this->elt(i) = xmma::reluTh_fp32(this->elt(i), relu_lb);
        }
    }

    inline __device__ void output_colwert() {
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  I M M A . 8
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Ampere_spimma_int8_traits, Layout> : public Fragment<int8_t, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Ampere_spimma_int8_traits, Layout> : public Fragment<int8_t, 32> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_e<Ampere_spimma_int8_traits, Layout> : public Fragment<uint32_t, 4> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Ampere_spimma_int8_traits> : public Fragment<int32_t, 8> {
    // The base class.
    using Base = Fragment<int32_t, 8>;

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
        void mma(const Fragment_a<Ampere_spimma_int8_traits, Layout_a> &a,
                 const Fragment_b<Ampere_spimma_int8_traits, Layout_b> &b,
                 const uint32_t e, int thread) {
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)       
        asm volatile( \
            "_mma.sp.thread.m16n8k64.row.col.s32.s8.s8.s32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+r"(   reg(0)), "+r"(   reg(1)), "+r"(   reg(2)), "+r"(   reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "_mma.sp.thread.m16n8k64.row.col.s32.s8.s8.s32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+r"(   reg(4)), "+r"(   reg(5)), "+r"(   reg(6)), "+r"(   reg(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#else
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+r"(   reg(0)), "+r"(   reg(1)), "+r"(   reg(2)), "+r"(   reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+r"(   reg(4)), "+r"(   reg(5)), "+r"(   reg(6)), "+r"(   reg(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_pre_swizzle<Ampere_spimma_int8_traits, Cta_tile, true>
    : public Fragment_imma_int32_epilogue_pre_swizzle<Ampere_spimma_int8_traits, Cta_tile> {

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_post_swizzle<Ampere_spimma_int8_traits, Cta_tile, true> 
    : public Fragment_imma_int32_epilogue_post_swizzle<Ampere_spimma_int8_traits, Cta_tile> {

    template< typename Fragment_alpha >
    inline __device__ void scale(float alpha, Fragment_alpha alpha_frag) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            this->elt(i) *= alpha_frag.elt(i);
        }
    }

    template< typename Fragment_res, typename Fragment_beta >
    inline __device__ void residual(float beta, Fragment_res frag, Fragment_beta beta_frag) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            char4 tmp;
            tmp = reinterpret_cast<const char4&>(frag.reg(i));
            this->elt(i * 4 + 0) += beta_frag.elt(i*4+0) * float(tmp.x);
            this->elt(i * 4 + 1) += beta_frag.elt(i*4+1) * float(tmp.y);
            this->elt(i * 4 + 2) += beta_frag.elt(i*4+2) * float(tmp.z);
            this->elt(i * 4 + 3) += beta_frag.elt(i*4+3) * float(tmp.w);
        }
    }

    template< typename Fragment_ >
    inline __device__ void add_bias(Fragment_ frag) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            this->elt(i) += frag.elt(i);
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_spimma_int8_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_imma_int32_c<Ampere_spimma_int8_traits, Cta_tile> {

    // Pack two fragments.
    template <typename Fragment_post_swizzle>
    inline __device__ void pack_sparse(float alpha,
            Fragment_post_swizzle frag) {

        int32_t tmp[4];
        #pragma unroll
        for( int ii = 0; ii < 2; ++ii ) {
            tmp[0] = xmma::f2i(frag.elt(4 * ii    ));
            tmp[1] = xmma::f2i(frag.elt(4 * ii + 1));
            tmp[2] = xmma::f2i(frag.elt(4 * ii + 2));
            tmp[3] = xmma::f2i(frag.elt(4 * ii + 3));
            this->reg(ii) = xmma::pack_int8x4(tmp);
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Ampere_spimma_gelu_int8_traits, Layout>
    : public Fragment_a<Ampere_spimma_int8_traits, Layout> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Ampere_spimma_gelu_int8_traits, Layout>
    : public Fragment_b<Ampere_spimma_int8_traits, Layout> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_e<Ampere_spimma_gelu_int8_traits, Layout>
    : public Fragment_e<Ampere_spimma_int8_traits, Layout> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Ampere_spimma_gelu_int8_traits>
    : public Fragment_aclwmulator<Ampere_spimma_int8_traits> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_spimma_gelu_int8_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_c<Ampere_spimma_int8_traits, Cta_tile, IN_CTA_SPLIT_K>  {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  I M M A . 8   I N T E R L E A V E D
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Ampere_spimma_interleaved_int8_traits, Layout> : public Fragment<int8_t, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Ampere_spimma_interleaved_int8_traits, Layout> : public Fragment<int8_t, 32> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_e<Ampere_spimma_interleaved_int8_traits, Layout> : public Fragment<uint32_t, 4> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Ampere_spimma_interleaved_int8_traits> : public Fragment<int32_t, 8> {
    // The base class.
    using Base = Fragment<int32_t, 8>;

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
        void mma(const Fragment_a<Ampere_spimma_interleaved_int8_traits, Layout_a> &a,
                 const Fragment_b<Ampere_spimma_interleaved_int8_traits, Layout_b> &b,
                 const uint32_t e, int thread) {
#if (XMMA_PTX_MMA_SPARSE_ENABLED == 0)       
        asm volatile( \
            "_mma.sp.thread.m16n8k64.row.col.s32.s8.s8.s32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+r"(   reg(0)), "+r"(   reg(1)), "+r"(   reg(2)), "+r"(   reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "_mma.sp.thread.m16n8k64.row.col.s32.s8.s8.s32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+r"(   reg(4)), "+r"(   reg(5)), "+r"(   reg(6)), "+r"(   reg(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#else
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+r"(   reg(0)), "+r"(   reg(1)), "+r"(   reg(2)), "+r"(   reg(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)),  "r"(b.reg(2)),  "r"(b.reg(3))
                    ,  "r"(e));
        asm volatile( \
            "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32" \
            "    {%0, %1, %2, %3}," \
            "    {%4, %5, %6, %7}," \
            "    {%8, %9, %10, %11}," \
            "    {%0, %1, %2, %3}, %12, 0;" \
                    : "+r"(   reg(4)), "+r"(   reg(5)), "+r"(   reg(6)), "+r"(   reg(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(4)),  "r"(b.reg(5)),  "r"(b.reg(6)),  "r"(b.reg(7))
                    ,  "r"(e));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_pre_swizzle<Ampere_spimma_interleaved_int8_traits, Cta_tile, true>
    : public Fragment_imma_int32_epilogue_pre_swizzle<Ampere_spimma_interleaved_int8_traits, Cta_tile> {

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Fragment_epilogue_post_swizzle<Ampere_spimma_interleaved_int8_traits, Cta_tile, true> 
    : public Fragment_imma_int32_epilogue_post_swizzle<Ampere_spimma_interleaved_int8_traits, Cta_tile> {

    template< typename Fragment_alpha >
    inline __device__ void scale(float alpha, Fragment_alpha alpha_frag) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            this->elt(i) *= alpha_frag.elt(i);
        }
    }

    template< typename Fragment_res, typename Fragment_beta >
    inline __device__ void residual(float beta, Fragment_res frag, Fragment_beta beta_frag) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            char4 tmp;
            tmp = reinterpret_cast<const char4&>(frag.reg(i));
            this->elt(i * 4 + 0) += beta_frag.elt(i*4+0) * float(tmp.x);
            this->elt(i * 4 + 1) += beta_frag.elt(i*4+1) * float(tmp.y);
            this->elt(i * 4 + 2) += beta_frag.elt(i*4+2) * float(tmp.z);
            this->elt(i * 4 + 3) += beta_frag.elt(i*4+3) * float(tmp.w);
        }
    }

    template< typename Fragment_ >
    inline __device__ void add_bias(Fragment_ frag) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            this->elt(i) += frag.elt(i);
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_spimma_interleaved_int8_traits, Cta_tile, IN_CTA_SPLIT_K> 
    : public Fragment_imma_int32_c<Ampere_spimma_interleaved_int8_traits, Cta_tile> {

    // Pack two fragments.
    template <typename Fragment_post_swizzle>
    inline __device__ void pack_sparse(float alpha,
            Fragment_post_swizzle frag) {

        int32_t tmp[4];
        #pragma unroll
        for( int ii = 0; ii < 2; ++ii ) {
            tmp[0] = xmma::f2i(frag.elt(4 * ii    ));
            tmp[1] = xmma::f2i(frag.elt(4 * ii + 1));
            tmp[2] = xmma::f2i(frag.elt(4 * ii + 2));
            tmp[3] = xmma::f2i(frag.elt(4 * ii + 3));

            this->reg(ii) = xmma::pack_int8x4(tmp);
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Ampere_spimma_interleaved_gelu_int8_traits, Layout>
    : public Fragment_a<Ampere_spimma_interleaved_int8_traits, Layout> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Ampere_spimma_interleaved_gelu_int8_traits, Layout>
    : public Fragment_b<Ampere_spimma_interleaved_int8_traits, Layout> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_e<Ampere_spimma_interleaved_gelu_int8_traits, Layout>
    : public Fragment_e<Ampere_spimma_interleaved_int8_traits, Layout> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Ampere_spimma_interleaved_gelu_int8_traits>
    : public Fragment_aclwmulator<Ampere_spimma_interleaved_int8_traits> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_spimma_interleaved_gelu_int8_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_c<Ampere_spimma_interleaved_int8_traits, Cta_tile, IN_CTA_SPLIT_K>  {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a<Ampere_spimma_int8_rt_fuse_traits, Layout>
    : public Fragment_a<Ampere_spimma_int8_traits, Layout> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b<Ampere_spimma_int8_rt_fuse_traits, Layout>
    : public Fragment_b<Ampere_spimma_int8_traits, Layout> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_e<Ampere_spimma_int8_rt_fuse_traits, Layout>
    : public Fragment_e<Ampere_spimma_int8_traits, Layout> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_aclwmulator<Ampere_spimma_int8_rt_fuse_traits>
    : public Fragment_aclwmulator<Ampere_spimma_int8_traits> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Fragment_c<Ampere_spimma_int8_rt_fuse_traits, Cta_tile, IN_CTA_SPLIT_K>
    : public Fragment_c<Ampere_spimma_int8_traits, Cta_tile, IN_CTA_SPLIT_K>  {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace xmma


