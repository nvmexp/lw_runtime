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

#include <stdint.h>
#include <xmma/ext/sparse/utils.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace implicit_gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int FLT_T_, int FLT_R_, int FLT_S_, bool IS_SIMPLE_1x1x1_, bool USE_PREDICATES_> struct Input_related {
    enum { FLT_T = FLT_T_ };
    enum { FLT_R = FLT_R_ };
    enum { FLT_S = FLT_S_ };
    enum { IS_SIMPLE_1x1x1 = IS_SIMPLE_1x1x1_ };
    enum { STATIC_FILTER_SIZE = ( ( FLT_T > 0 && FLT_R > 0 && FLT_S > 0 ) ? 1 : 0 ) };
    enum { USE_PREDICATES = USE_PREDICATES_ };

    static_assert( ( FLT_T >= 0 && FLT_R >= 0 && FLT_S >= 0 ),
                   "FLT_T >= 0 && FLT_R >= 0 && FLT_S >= 0" );
    static_assert( !( IS_SIMPLE_1x1x1 && ( FLT_T != 1 || FLT_R != 1 || FLT_S != 1 ) ),
                   "!(IS_SIMPLE_1x1x1 && (FLT_T!=1 || FLT_R!=1 || FLT_S!=1))" );
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int FLT_T, int FLT_R, int FLT_S> struct Build_mask_t {
    enum { VALUE = ( 1u << ( FLT_R * FLT_S ) ) - 1 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int FLT_R, int FLT_S> struct Build_mask_r_ {
    enum { VALUE = ( 1u << FLT_R ) - 1 };
};

template <int FLT_T, int FLT_R, int FLT_S> struct Build_mask_r {
    enum {
        VALUE = Build_mask_r_<FLT_R, FLT_S>::VALUE |
                ( Build_mask_r<FLT_T - 1, FLT_R, FLT_S>::VALUE << ( FLT_R * FLT_S ) )
    };
};

template <int FLT_R, int FLT_S> struct Build_mask_r<1, FLT_R, FLT_S> {
    enum { VALUE = Build_mask_r_<FLT_R, FLT_S>::VALUE };
};

template <> struct Build_mask_r<0, 0, 0> {
    enum { VALUE = 0 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int FLT_R, int FLT_S> struct Build_mask_s_ {
    enum { VALUE = 0x1 | ( Build_mask_s_<FLT_R - 1, FLT_S>::VALUE << FLT_S ) };
};

template <int FLT_S> struct Build_mask_s_<1, FLT_S> {
    enum { VALUE = 0x1 };
};

template <int FLT_T, int FLT_R, int FLT_S> struct Build_mask_s {
    enum {
        VALUE = Build_mask_s_<FLT_R, FLT_S>::VALUE |
                ( Build_mask_s<FLT_T - 1, FLT_R, FLT_S>::VALUE << ( FLT_R * FLT_S ) )
    };
};

template <int FLT_R, int FLT_S> struct Build_mask_s<1, FLT_R, FLT_S> {
    enum { VALUE = Build_mask_s_<FLT_R, FLT_S>::VALUE };
};

template <> struct Build_mask_s<0, 0, 0> {
    enum { VALUE = 0 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int M, int N >
inline __device__ void pack_predicates(uint32_t (&preds)[M], const uint32_t (&masks)[N], uint32_t bit_mask) {
    uint32_t lop[N];
#pragma unroll
    for( int i = 0; i < N; ++i ) {
        lop[i] = masks[i] & bit_mask;
    }
    return xmma::pack_predicates( preds, lop );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace implicit_gemm
}  // namespace ext
} // namespace xmma
