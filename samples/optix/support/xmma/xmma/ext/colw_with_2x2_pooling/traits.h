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

#include <xmma/ext/colw_with_2x2_pooling/gmem_tile.h>
#include <xmma/ext/colw_with_2x2_pooling/smem_tile.h>
#include <xmma/ext/colw_with_2x2_pooling/callbacks.h>
#include <xmma/smem_tile_with_halo.h>
#include <xmma/implicit_gemm/fprop/gmem_tile.h>

namespace xmma {
namespace ext {
namespace colw_with_2x2_pooling {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits_, 
    // The dimensions of the CTA.
    typename Cta_tile_, 
    // The distribution of CTAs in the grid.
    typename Cta_distribution_,
    // The dimensions of the tile computed by the colwolution.
    typename Colw_tile_, 
    // The dimensions of the filter.
    typename Colw_filter_, 
    // The pooling functor.
    typename Pooling_functor_
>
struct Kernel_traits : public Traits_ {

    // The traits class.
    using Traits = Traits_;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The Cta distribution in the grid.
    using Cta_distribution = Cta_distribution_;
    // That kernel uses a single stage for the pipeline.
    enum { STAGES = 1 };

    // The dimensions of the tile computed by the colwolution.
    using Colw_tile = Colw_tile_; 
    // The dimensions of the filter.
    using Colw_filter = Colw_filter_; 
    // The halo for the filter.
    using Colw_halo = xmma::Tile_nhw<0, Colw_filter::FLT_R-1, Colw_filter::FLT_S-1>;

    // The pooling functor.
    using Pooling_functor = Pooling_functor_;

    // The global memory loader for A.
    using Gmem_tile_a = xmma::ext::colw_with_2x2_pooling::Gmem_tile_a<Traits, 
                                                                     Cta_tile, 
                                                                     Colw_tile, 
                                                                     Colw_filter>;
    // Does the Gmem tile for A use LDGSTS?
    enum { USE_LDGSTS_A = Gmem_tile_a::USE_LDGSTS };
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_A = USE_LDGSTS_A ? xmma::Max<2, STAGES>::VALUE : STAGES };

    // The shared memory loader for A.
    using Smem_tile_a = xmma::Smem_tile_with_halo_a<Traits, 
                                                        Cta_tile, 
                                                        xmma::Row, 
                                                        Colw_tile, 
                                                        Colw_halo,
                                                        BUFFERS_PER_SMEM_TILE_A>; 

    // The size of each LDG for B.
    enum { BYTES_PER_LDG_B = 16 };
    // The global memory loader for B.
    using Gmem_tile_b = xmma::implicit_gemm::fprop::Gmem_tile_b<Traits, 
                                                                    Cta_tile,
                                                                    Colw_filter,
                                                                    BYTES_PER_LDG_B,
                                                                    true>;
    // Does the Gmem tile for B use LDGSTS?
    enum { USE_LDGSTS_B = Gmem_tile_a::USE_LDGSTS };
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_B = USE_LDGSTS_B ? xmma::Max<2, STAGES>::VALUE : STAGES };

    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<Traits, 
                                              Cta_tile, 
                                              xmma::Col, 
                                              Gmem_tile_b::BYTES_PER_LDG, 
                                              BUFFERS_PER_SMEM_TILE_B>;
    // Assemble the global memory tile for the output.
    using Gmem_tile_epilogue = xmma::ext::colw_with_2x2_pooling::Gmem_tile_c<Traits, 
                                                                            Cta_tile, 
                                                                            Colw_tile>;
    // The fragment stored by the tile.
    using Fragment_c = typename Gmem_tile_epilogue::Fragment_c;
    // The shared memory tile.
    using Swizzle_epilogue = xmma::ext::colw_with_2x2_pooling::Smem_tile_c<Traits, 
                                                                          Cta_tile, 
                                                                          Gmem_tile_epilogue>;

    // The fragment before swizzle.
    using Fragment_pre_swizzle = typename Swizzle_epilogue::Fragment_pre_swizzle;
    // The fragment after swizzle.
    using Fragment_post_swizzle = typename Swizzle_epilogue::Fragment_post_swizzle;

    // The callbacks.
    using Callbacks_epilogue = xmma::ext::colw_with_2x2_pooling::Callbacks<Traits, 
                                                                          Cta_tile,
                                                                          Fragment_pre_swizzle,
                                                                          Fragment_post_swizzle,
                                                                          Fragment_c,
                                                                          Pooling_functor>;
    // The epilogue.
    using Epilogue = xmma::helpers::Epilogue<Traits, 
                                                 Cta_tile, 
                                                 xmma::Row, 
                                                 Gmem_tile_epilogue,
                                                 Callbacks_epilogue,
                                                 Swizzle_epilogue>;


    // The number of threads in the CTA.
    static int threads_per_cta() {
        return Cta_tile::THREADS_PER_CTA;
    }

    // The amount of shared memory.
    static int dynamic_smem_size_per_cta() {

        // The amount of shared memory needed for the main loop.
        enum { LOOP_SIZE_IN_BYTES = Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE };
        // The amount of shared memory needed by the epilogue.
        enum { EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE };
        // The amount of shared memory to launch the kernel.
        return (int) xmma::Max<LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES>::VALUE;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace colw_with_2x2_pooling
} // namespace ext
} // namespace xmma

