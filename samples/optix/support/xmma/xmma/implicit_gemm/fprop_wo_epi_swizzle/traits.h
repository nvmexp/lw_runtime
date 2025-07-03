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
#include <xmma/turing/traits.h>
#include <xmma/ampere/traits.h>
#include <xmma/ampere/smem_tile.h>

#include <xmma/helpers/epilogue.h>
#include <xmma/helpers/callbacks.h>

#include <xmma/params.h>
#include <xmma/tile_distribution.h>
#include <xmma/compute_tile.h>

#include <xmma/implicit_gemm/fprop/params.h>
#include <xmma/implicit_gemm/fprop/traits.h>
#include <xmma/implicit_gemm/fprop_wo_epi_swizzle/gmem_tile.h>
#include <xmma/implicit_gemm/fprop_wo_epi_swizzle/callbacks.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace implicit_gemm {
namespace fprop_wo_epi_swizzle {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The CTA tile descriptor.
    typename Cta_tile_,
    // The global memory tile for A (transposed or not).
    typename Gmem_tile_a_,
    // The global memory tile for Epilogue (transposed or not).
    typename Gmem_tile_epilogue_,
    // Input related params
    typename Input_related_,
    // The number of stages in the prefetch pipeline.
    int STAGES_ = 1>
struct Kernel_traits : public fprop::Kernel_traits<Traits_,
                                                   Cta_tile_,
                                                   Gmem_tile_a_,
                                                   Gmem_tile_epilogue_,
                                                   Input_related_,
                                                   STAGES_> {
    using Base = fprop::Kernel_traits<Traits_, Cta_tile_, Gmem_tile_a_, Gmem_tile_epilogue_, Input_related_, STAGES_>;

    // The global memory loader for B.
    using Gmem_tile_b = fprop_wo_epi_swizzle::template Gmem_tile_b< typename Base::Traits,
                                                                    typename Base::Cta_tile,
                                                                    typename Base::Input_related>;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    static const int BUFFERS_PER_SMEM_TILE_B = Gmem_tile_b::USE_LDGSTS ? Max<2, STAGES_>::VALUE : STAGES_;
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<typename Base::Traits, typename Base::Cta_tile, xmma::Col,
                                          Gmem_tile_b::BYTES_PER_LDG, BUFFERS_PER_SMEM_TILE_B>;

    // The compute tile.
    using Compute_tile = typename Compute_tile_selector<typename Base::Traits,
                                                        typename Base::Cta_tile,
                                                        typename Base::Smem_tile_a,
                                                        Smem_tile_b,
                                                        Base::OPERATION_TYPE>::Class;

    // Empty swizzle epilogue.
    using Swizzle_epilogue = xmma::Swizzle_epilogue_bypass< typename Base::Traits, typename Base::Cta_tile, xmma::Row >;
    // The callbacks.
    using Callbacks_epilogue =
        fprop_wo_epi_swizzle::template Callbacks_epilogue< typename Base::Traits,
                                                           typename Base::Cta_tile,
                                                           typename Base::Gmem_tile_epilogue >;
    // The epilogue.
    using Epilogue =
        xmma::helpers::template Epilogue<
            typename Base::Traits,
            typename Base::Cta_tile,
            typename Base::Gmem_tile_epilogue::Layout,
            typename Base::Gmem_tile_epilogue,
            Callbacks_epilogue,
            Swizzle_epilogue,
            typename Callbacks_epilogue::Fragment_pre_swizzle,
            typename Callbacks_epilogue::Fragment_post_swizzle,
            typename Callbacks_epilogue::Fragment_c>;

    // The global memory loader for A without smem.
    using Gmem_wo_smem_tile_b = xmma::Gmem_wo_smem_tile_b< typename Base::Traits,
                                                           typename Base::Cta_tile,
                                                           typename Gmem_tile_b::Smem_layout,
                                                           Gmem_tile_b::BYTES_PER_LDG>;

    using Callbacks_wo_smem_epilogue = Callbacks_epilogue;
    using Epilogue_wo_smem = Epilogue;

    static int dynamic_smem_size_per_cta() {
        // The amount of shared memory needed for the main loop.
        const int LOOP_SIZE_IN_BYTES = Base::Smem_tile_a::BYTES_PER_TILE + Base::Smem_tile_b::BYTES_PER_TILE;

        return LOOP_SIZE_IN_BYTES;
    }

#if !defined( __LWDACC_RTC__ )
    using Params = fprop::template Params<typename Base::Traits,
                                                  typename Base::Cta_tile,
                                                  Base::STAGES>;

    typedef void ( *Kernel_type )( Params params );

    // Return device kernel function pointer.
    static XMMA_HOST Kernel_type kernel_ptr( const Params params = Params() ) {
        return &::xmma::gemm::kernel<Kernel_traits>;
    }

    // Return split k kernel function pointer.
    static XMMA_HOST Kernel_type split_k_kernel_ptr() {
        return &::xmma::gemm::split_k_kernel<Kernel_traits>;
    }
#endif

};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fprop_wo_epi_swizzle
} // namespace implicit_gemm
} // namespace xmma

