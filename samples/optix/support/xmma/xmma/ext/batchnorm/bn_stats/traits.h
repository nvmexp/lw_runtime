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

#include <xmma/implicit_gemm/fprop/traits.h>
#include <xmma/ext/batchnorm/bn_apply/fprop/params.h>
#include <xmma/ext/batchnorm/bn_stats/callbacks_epilogue.h>
#include <xmma/ext/batchnorm/bn_stats/volta/callbacks_epilogue.h>
#include <xmma/ext/batchnorm/bn_stats/turing/callbacks_epilogue.h>
#include <xmma/ext/batchnorm/bn_stats/ampere/callbacks_epilogue.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_stats {

///////////////////////////////////////////////////////////////////////////////////////////////////

// The only thing extra here compared to implicit gemm fprop is the epilogue callback
template< typename Traits_, typename Cta_tile_, typename Input_related_, int STAGES_ = 1>
struct Kernel_traits : 
        public xmma::implicit_gemm::fprop::Kernel_traits<Traits_, Cta_tile_,
        xmma::implicit_gemm::fprop::Gmem_tile_a_t<Traits_, Cta_tile_, Input_related_>,
        xmma::implicit_gemm::fprop::Gmem_tile_c_t<Traits_, Cta_tile_, 16>,
        Input_related_, STAGES_> {

    // The traits class.
    using Traits = Traits_;

    // The Cta tile.
    using Cta_tile = Cta_tile_;

    // The kernel parameters.
    //using Params = Fprop_bn_stats_params<Traits, Cta_tile, STAGES_>;
    using Params = bn_apply::fprop::Bn_apply_with_scale_bias_relu_fprop_params<Traits, Cta_tile,
                                                                               STAGES_>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = xmma::implicit_gemm::fprop::Gmem_tile_c_t<Traits, Cta_tile, 16>;

    // The callbacks.
    using Callbacks_epilogue = Batch_norm_fprop_callbacks_epilogue<Traits, Cta_tile>;
    //using Callbacks_epilogue = xmma::helpers::Empty_callbacks_epilogue<Traits, Cta_tile>;

    using Epilogue = xmma::helpers::Epilogue_with_split_k<Traits,
                                                              Cta_tile,
                                                              xmma::Row,
                                                              Gmem_tile_epilogue,
                                                              Callbacks_epilogue>;

    /* NOTE: Only FP64 GEMM supports gmem_wo_smem kernel */
    using Gmem_tile_wo_smem_epilogue = Gmem_tile_epilogue;
    // The callbacks.
    using Callbacks_wo_smem_epilogue = Callbacks_epilogue;
    // The epilogue.
    using Epilogue_wo_smem = Epilogue;
    /* NOTE: end. */

    // The amount of shared memory.
    static int dynamic_smem_size_per_cta() {

        // The amount of shared memory needed to compute the BN stats.
        const int BN_SIZE_IN_BYTES = Callbacks_epilogue::BYTES_PER_TILE;

        // Max(main loop, epilogue) used in the FPROP portion
        const int FPROP_SIZE_IN_BYTES = xmma::implicit_gemm::fprop::Kernel_traits<Traits_, Cta_tile_,
                                        xmma::implicit_gemm::fprop::Gmem_tile_a_t<Traits_, Cta_tile_, Input_related_>,
                                        xmma::implicit_gemm::fprop::Gmem_tile_c_t<Traits_, Cta_tile_, 16>,
                                        Input_related_, STAGES_>
                                        ::dynamic_smem_size_per_cta();

        // The amount of shared memory to launch the kernel.
        return max(FPROP_SIZE_IN_BYTES, BN_SIZE_IN_BYTES);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
} // bn_stats
} // batchnorm
} // ext
} // xmma
///////////////////////////////////////////////////////////////////////////////////////////////////
