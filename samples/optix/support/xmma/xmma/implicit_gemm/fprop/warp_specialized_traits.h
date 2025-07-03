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
#include <xmma/turing/traits.h>
#include <xmma/volta/traits.h>

#include <xmma/compute_tile.h>
#include <xmma/params.h>
#include <xmma/tile_distribution.h>
#include <xmma/warp_specialized_traits.h>

#include <xmma/helpers/epilogue_with_split_k.h>

#include <xmma/implicit_gemm/fprop/gmem_tile.h>
#include <xmma/implicit_gemm/fprop/traits.h>
#include <xmma/implicit_gemm/fprop/warp_specialized_params.h>

#include <xmma/gemm/warp_specialized_kernel.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace implicit_gemm {
namespace fprop {
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits
    typename Traits_,
    // The CTA tile descriptor
    typename Cta_tile_,
    // The global memory tile for Epilogue (transpose or not)
    typename Gmem_tile_epilogue_,
    // Input realted params
    typename Input_related_,
    // The arch being compiled for this warp specialized kernel.
    int32_t ARCH_ = 80> // int STAGES_ = 1>//, int SMEM_BYTES_PER_SM_ = // 167936 >
struct Warp_specialized_kernel_traits
    : public xmma::implicit_gemm::fprop::Kernel_traits<
            Traits_, Cta_tile_, xmma::implicit_gemm::fprop::Gmem_tile_a_t<Traits_, Cta_tile_, Input_related_>,
            Gmem_tile_epilogue_, Input_related_, 1> {

    using Base =
        xmma::implicit_gemm::fprop::Kernel_traits<
            Traits_, Cta_tile_,
            xmma::implicit_gemm::fprop::Gmem_tile_a_t<Traits_, Cta_tile_, Input_related_>,
            Gmem_tile_epilogue_, Input_related_, 1>;

    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::FPROP;
    static const bool USE_WARP_SPECIALIZATION = true;

    // The warp specialized kernel traits type.
    enum { ARRIVE_WAIT = 0, NAMED_BARRIER = 1 };
    // The arch for smem allocation.
    enum { ARCH = ARCH_ };

    // The instruction traits.
    using Traits = Traits_;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // Template parameters which are related to input parameters like filter size.
    using Input_related = Input_related_;
    // The kernel parameters.
    using Params =
        xmma::implicit_gemm::fprop::Warp_specialized_params<Traits, Cta_tile>;
    // The global memory loader for A.
    using Gmem_tile_a = typename Base::Gmem_tile_a;
    // The global memory loader for B.
    using Gmem_tile_b = typename Base::Gmem_tile_b;
    // The warp specialized kernel traits.
    using Warp_specialized_traits =
        typename xmma::Warp_specialized_traits_selector<
            Traits, Cta_tile, xmma::Row, xmma::Col, xmma::Row,
            Gmem_tile_a::BYTES_PER_LDG, Gmem_tile_b::BYTES_PER_LDG,
            ARCH, ARRIVE_WAIT>::Class;

    // The global memory loader for epilogue
    using Gmem_tile_epilogue = typename Base::Gmem_tile_epilogue;
    // The callback for epilgoue.
    using Callbacks_epilogue = typename Base::Callbacks_epilogue;

    // The Epilogue with splik
    using Epilogue_withsplitk =
        xmma::helpers::Epilogue_with_split_k<Traits, Cta_tile,
                                             typename Gmem_tile_epilogue::Layout,
                                             Gmem_tile_epilogue, Callbacks_epilogue>;

    // The Epilogue without splik
    using Epilogue_wosplitk =
        xmma::helpers::Epilogue<Traits, Cta_tile,
                                typename Gmem_tile_epilogue::Layout,
                                Gmem_tile_epilogue, Callbacks_epilogue>;

    // Tile distribution_persisitent
    using Tile_distribution_persistent =
        typename Warp_specialized_traits::Tile_distribution_persistent;

    enum { WARP_SPECIALIZED_CONFIG = Warp_specialized_traits::WARP_SPECIALIZED_CONFIG };
    enum { BUFFERS_PER_SMEM_TILE_A = Warp_specialized_traits::BUFFERS_PER_SMEM_TILE_A };
    enum { BUFFERS_PER_SMEM_TILE_B = Warp_specialized_traits::BUFFERS_PER_SMEM_TILE_B };

    // The shared memory loader for A.
    using Smem_tile_a =
        xmma::Smem_tile_a<Traits, Cta_tile, xmma::Row, Gmem_tile_a::BYTES_PER_LDG,
                          BUFFERS_PER_SMEM_TILE_A>;
    // The shared memory loader for B.
    using Smem_tile_b =
        xmma::Smem_tile_b<Traits, Cta_tile, xmma::Col, Gmem_tile_b::BYTES_PER_LDG,
                          BUFFERS_PER_SMEM_TILE_B>;
    // The compute tile.
    using Compute_tile =
        typename xmma::Compute_tile_selector<Traits, Cta_tile, Smem_tile_a,
                                             Smem_tile_b, OPERATION_TYPE>::Class;
    enum {
        SMEM_BYTES_PER_CTA = Smem_tile_a::BYTES_PER_TILE +
                             Smem_tile_b::BYTES_PER_TILE +
                             Warp_specialized_traits::EPILOGUE_SIZE_IN_BYTES +
                             Warp_specialized_traits::ARRIVE_WAIT_SMEM_SIZE
    };

    static_assert((int)SMEM_BYTES_PER_CTA <= (int)Warp_specialized_traits::SMEM_BYTES_PER_SM,
                  "error: Shared memory needed exceeds capacity");

#if !defined( __LWDACC_RTC__ )
    typedef void ( *Kernel_type )( Params params );

    static XMMA_HOST Kernel_type kernel_ptr( const Params params = Params() ) {
        if( params.specialize == xmma::CONFIG_1DMA_1MATH ) {
            return &xmma::gemm::xmma_implicit_gemm_specialize_1math_1dma_arrive_wait_kernel<
                Warp_specialized_kernel_traits>;
        } else if( params.specialize == xmma::CONFIG_1DMA_2MATH ) {
            return &xmma::gemm::xmma_implicit_gemm_specialize_2math_1dma_arrive_wait_kernel<
                Warp_specialized_kernel_traits>;
        }

        return &xmma::gemm::xmma_implicit_gemm_specialize_1math_1dma_arrive_wait_kernel<
            Warp_specialized_kernel_traits>;
    }

    static XMMA_HOST Kernel_type split_k_kernel_ptr() {
        return &::xmma::gemm::split_k_kernel<Warp_specialized_kernel_traits>;
    }
#endif

    // The number of threads in the CTA.
    static int32_t threads_per_cta( const Params params = Params() ) {
        if( params.specialize == xmma::CONFIG_1DMA_2MATH ) {
            return Cta_tile::THREADS_PER_CTA * 3;
        }
        else if( params.specialize == xmma::CONFIG_1DMA_1MATH ) {
            return Cta_tile::THREADS_PER_CTA * 2;
        }
        else {
            return Cta_tile::THREADS_PER_CTA;
        }
    }

    // The amount of shared memory.
    static int dynamic_smem_size_per_cta() {
        // The amount of shared memory to launch the kernel.
        return SMEM_BYTES_PER_CTA;
    }

    static int32_t epilogue_smem_size_per_cta() {
        // The amount of shared memory needed in epilogue.
        return Warp_specialized_traits::EPILOGUE_SIZE_IN_BYTES;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fprop
} // namespace implicit_gemm
} // namespace xmma
