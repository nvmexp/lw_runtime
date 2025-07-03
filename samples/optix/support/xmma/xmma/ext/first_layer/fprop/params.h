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

namespace xmma {
namespace ext {
namespace first_layer {
namespace fprop {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Params {
    // The pointer to the image.
    const void *img_gmem;
    // The pointer to the filter.
    const void *flt_gmem;
    // The pointer to the output.
    void *out_gmem;

    // The dimensions of the problem.
    int n, d, h, w, k, t, r, s, o, p, q;
    // The image strides.
    int img_stride_n, img_stride_h;
    // The output strides.
    int out_stride_n, out_stride_h, out_stride_w;
    // The number of output rows computed per CTA.
    int out_rows_per_cta;

    // The values for alpha/beta.
    typename Traits::Epilogue_type alpha, beta;

    // The number of CTAs.
    int ctas_pq, ctas_q;
    // The precomputed values for the fast division.
    uint32_t mul_ctas_pq, shr_ctas_pq, mul_ctas_q, shr_ctas_q;

    // The memory descriptors.
    xmma::Memory_descriptors mem_descriptors;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fprop
}  // namespace first_layer
}  // namespace ext
} // namespace xmma

