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
#include <xmma/helpers/epilogue_with_split_k.h>
#include <xmma/params.h>
#include <xmma/tile_distribution.h>
#include <xmma/compute_tile.h>
#include <xmma/ext/implicit_gemm/indexed_wo_smem/wgrad/params.h>
#include <xmma/ext/implicit_gemm/indexed_wo_smem/wgrad/gmem_tile.h>
#include <xmma/helpers/callbacks.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace implicit_gemm {
namespace indexed_wo_smem {
namespace wgrad {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The CTA tile descriptor.
    typename Cta_tile_,
    // ELEMENT_PER_LDG
    int ELTS_PER_LDG_,
    // ELEMENT_PER_STG
    int ELTS_PER_STG_>
struct Kernel_traits : public Traits_ {

    // Colwolution type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::WGRAD;
    // Colwolution algorithm.
    static const xmma::Colwolution_algorithm COLW_ALGO =
        xmma::Colwolution_algorithm::INDEX;
    // Colwolution layout.
    static const xmma::Colwolution_layout COLW_LAYOUT = xmma::Colwolution_layout::NHWC;
    // Whether use warp specialized
    static const bool USE_WARP_SPECIALIZATION = false;

    // The instruction traits.
    using Traits = Traits_;
    // The Cta tile.
    using Cta_tile = Cta_tile_;

    // The number of elements per LDG.
    enum { ELTS_PER_LDG = ELTS_PER_LDG_ };
    // The number of elements per STG.
    enum { ELTS_PER_STG = ELTS_PER_STG_ };

    // The number of stages.
    enum { STAGES = 1 };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = xmma::ext::implicit_gemm::indexed_wo_smem::wgrad::Params<Traits, Cta_tile, STAGES>;
    // Tile distribution
    using Tile_distribution = xmma::Tile_distribution;

    // The global memory loader for A.
    using Gmem_tile_a = xmma::ext::implicit_gemm::indexed_wo_smem::wgrad::Gmem_tile_a<Traits, Cta_tile, ELTS_PER_LDG, ELTS_PER_STG>;

    // The global memory loader for B.
    using Gmem_tile_b = xmma::ext::implicit_gemm::indexed_wo_smem::wgrad::Gmem_tile_b<Traits, Cta_tile, ELTS_PER_LDG, ELTS_PER_STG>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = Gmem_tile_c<Traits, Cta_tile, ELTS_PER_LDG, ELTS_PER_STG>;

    using Fragment_layout_a = typename Fragment_layout<Traits::Gpu_arch::HAS_SUPER_HMMA>::Layout_a;
    using Fragment_layout_b = typename Fragment_layout<Traits::Gpu_arch::HAS_SUPER_HMMA>::Layout_b;
    using Fragment_a = xmma::Fragment_a<Traits, Fragment_layout_a>;
    using Fragment_b = xmma::Fragment_b<Traits, Fragment_layout_b>;

    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue_empty<Traits, Cta_tile,
        xmma::Row>;
    using Callbacks_epilogue = xmma::ext::helpers::Callbacks_epilogue<Traits, Cta_tile, ELTS_PER_STG>;
    using Epilogue = xmma::ext::helpers::Epilogue_with_split_k<Traits, Cta_tile,
        xmma::Row, Gmem_tile_epilogue, Callbacks_epilogue, Swizzle_epilogue>;
    // The number of threads in the CTA.
    static int threads_per_cta() {
        return Cta_tile::THREADS_PER_CTA;
    }

    // The amount of shared memory.
    static int dynamic_smem_size_per_cta() {
        // The amount of shared memory needed for the main loop.
        const int LOOP_SIZE_IN_BYTES =
              Cta_tile::THREADS_PER_CTA * 4 * 4;

        // The amount of shared memory needed by the epilogue.
        const int EPILOGUE_SIZE_IN_BYTES = 0;

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES );
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace wgrad
} // namespace indexed_wo_smem
} // namespace implicit_gemm
} // namespace ext
} // namespace xmma

