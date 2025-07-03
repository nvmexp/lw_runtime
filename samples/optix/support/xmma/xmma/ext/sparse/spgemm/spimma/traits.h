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

#include <xmma/tile_distribution.h>
#include <xmma/ext/sparse/helpers/gemm.h>
#include <xmma/ext/sparse/spgemm/spimma/gmem_tile.h>
#include <xmma/ext/sparse/spgemm/spimma/params.h>
#include <xmma/ext/sparse/spgemm/spimma/epilogue_light.h>
#include <xmma/ext/sparse/smem_tile.h>
#include <xmma/ext/sparse/ampere/traits.h>
#include <xmma/ext/sparse/spgemm/spimma/kernel.h>
//#include <xmma/smem_tile.h>


////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace gemm {
namespace sparse_imma_gemm {
////////////////////////////////////////////////////////////////////////////////////////////////////
template<
    // The instruction traits.
    typename Traits_,
    // The CTA tile descriptor.
    typename Cta_tile_,
    // The global memory tile for A (transposed or not).
    typename Gmem_tile_a_,
    // The global memory tile for B (transposed or not).
    typename Gmem_tile_b_,
    // The global memory tile for C (transposed or not).
    typename Gmem_tile_c_,
    // The global memory tile for the metadata. ???
    typename Gmem_tile_e_,
    // The number of stages in the prefetch pipeline.
    int STAGES_ = 1,
    // Choose RELU or GELU
    int ELTWISE = xmma::RELU
>
struct Sparse_kernel_traits  : public Traits_ {

    static const bool USE_SPARSE_IMMA = true;

    // The traits class.
    using Traits = Traits_;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // The number of STAGES.
    enum { STAGES = STAGES_ };

    // The output layout
    using Layout = typename Gmem_tile_c_::output_layout;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = Xmma_sparse_gemm_params<Traits, Cta_tile, STAGES>;
    // Tile distribution
    using Tile_distribution = xmma::Tile_distribution;
    // The global memory loader for B.
    using Gmem_tile_b = Gmem_tile_b_;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_B = xmma::Max<2, STAGES>::VALUE };
    // The shared memory layout for B.
    using Smem_layout_b = typename Gmem_tile_b::Smem_layout;
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<Traits,
                                              Cta_tile,
                                              Smem_layout_b,
                                              16,
                                              BUFFERS_PER_SMEM_TILE_B>;

    // The global memory loader for A.
    using Gmem_tile_a = Gmem_tile_a_;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_A = xmma::Max<2, STAGES>::VALUE };
    // The shared memory layout for A.
    using Smem_layout_a = typename Gmem_tile_a::Smem_layout;
    // The shared memory loader for A.
    using Smem_tile_a = xmma::Smem_tile_a<Traits,
                                              Cta_tile,
                                              Smem_layout_a,
                                              16,
                                              BUFFERS_PER_SMEM_TILE_A>;

    // The global memory loader for E.
    using Gmem_tile_e = Gmem_tile_e_;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_E = xmma::Max<2, STAGES>::VALUE };
    // The shared memory layout for E.
    using Smem_layout_e = typename Gmem_tile_e::Smem_layout;
    // The shared memory loader for E.
    using Smem_tile_e = xmma::Smem_tile_e<Traits,
                                              Cta_tile,
                                              Smem_layout_e,
                                              16,
                                              BUFFERS_PER_SMEM_TILE_E>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = Gmem_tile_c_;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue<Traits, Cta_tile, Layout>;
    // The callbacks.
    using Callbacks_epilogue = xmma::helpers::Callbacks_epilogue_light<Traits, Cta_tile, Layout, ELTWISE>;
    // The epilogue.
    using Epilogue = xmma::helpers::Epilogue_light<Traits,
                                                       Cta_tile,
                                                       //xmma::Col,
                                                       Layout,
                                                       Gmem_tile_epilogue,
                                                       Callbacks_epilogue,
                                                       Swizzle_epilogue>;

    // The number of threads per CTA.
    static int threads_per_cta() {
        return Cta_tile::THREADS_PER_CTA;
    }

    // The amount of dynamic shared memory per CTA.
    static int dynamic_smem_size_per_cta() {

        // The amount of shared memory needed for the main loop.
        const int LOOP_SIZE_IN_BYTES = Smem_tile_a::BYTES_PER_TILE +
                                       Smem_tile_b::BYTES_PER_TILE +
                                       Smem_tile_e::BYTES_PER_TILE ;

        // The amount of shared memory needed by the epilogue.
        //const int EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory to launch the kernel.
        //return max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES );
        return LOOP_SIZE_IN_BYTES;
    }

    // Gemm.
    template<
        typename Fragment_aclwmulators,
        typename Fragment_a,
        typename Fragment_b,
        typename Fragment_e,
        int M,
        int N,
        int N_PER_GROUP
    >
    static inline __device__ void spigemm(Fragment_aclwmulators (&acc)[M][N_PER_GROUP],
              const Fragment_a (&a)[M],
              const Fragment_b (&b)[N],
              const Fragment_e (&e)[1],
              int ki=0) {
        xmma::helpers::sparse_igemm_unroll(acc, a, b, e);
    }


    template<
        typename Fragment_aclwmulators,
        typename Fragment_a,
        typename Fragment_b,
        typename Fragment_e,
        int M,
        int N,
        int N_PER_GROUP
    >
    static inline __device__ void spigemm_pipeline(Fragment_aclwmulators (&acc)[M][N_PER_GROUP],
        const Fragment_a (&a)[M],
        const Fragment_b (&b)[N],
        const Fragment_e (&e)[1],
        int pipe_stage
        ) {
        xmma::helpers::sparse_igemm_pipeline(acc, a, b, e, pipe_stage);
    }

#if !defined(__LWDACC_RTC__)
    typedef void (*Kernel_type)(Params params);

    // Return device kernel function pointer.    
    static XMMA_HOST Kernel_type kernel_ptr(const Params params = Params()) {
        return &::xmma::ext::gemm::sparse_imma_gemm::kernel<Sparse_kernel_traits>;
    }

    // Return split k kernel function pointer.
    static XMMA_HOST Kernel_type split_k_kernel_ptr() {
        return &::xmma::ext::gemm::sparse_imma_gemm::split_k_kernel<Sparse_kernel_traits>;
    }
#endif
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    typename Traits,
    typename Cta_tile,
    typename Gmem_tile_a,
    typename Gmem_tile_b,
    typename Gmem_tile_c,
    typename Gmem_tile_e,
    int STAGES
>
struct Sparse_kernel_traits_adapter {
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Cta_tile,
    typename Gmem_tile_a,
    typename Gmem_tile_b,
    typename Gmem_tile_c,
    typename Gmem_tile_e,
    int STAGES >
struct Sparse_kernel_traits_adapter<xmma::Ampere_spimma_int8_traits,
                                    Cta_tile,
                                    Gmem_tile_a,
                                    Gmem_tile_b,
                                    Gmem_tile_c,
                                    Gmem_tile_e,
                                    STAGES > {
    using Type = Sparse_kernel_traits<xmma::Ampere_spimma_int8_traits,
                                      Cta_tile,
                                      Gmem_tile_a,
                                      Gmem_tile_b,
                                      Gmem_tile_c,
                                      Gmem_tile_e,
                                      STAGES,
                                      xmma::RELU>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Cta_tile,
    typename Gmem_tile_a,
    typename Gmem_tile_b,
    typename Gmem_tile_c,
    typename Gmem_tile_e,
    int STAGES >
struct Sparse_kernel_traits_adapter<xmma::Ampere_spimma_gelu_int8_traits,
                                    Cta_tile,
                                    Gmem_tile_a,
                                    Gmem_tile_b,
                                    Gmem_tile_c,
                                    Gmem_tile_e,
                                    STAGES > {
    using Type = Sparse_kernel_traits<xmma::Ampere_spimma_gelu_int8_traits,
                                      Cta_tile,
                                      Gmem_tile_a,
                                      Gmem_tile_b,
                                      Gmem_tile_c,
                                      Gmem_tile_e,
                                      STAGES,
                                      xmma::GELU>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
template<
    typename Cta_tile,
    typename Gmem_tile_a,
    typename Gmem_tile_b,
    typename Gmem_tile_c,
    typename Gmem_tile_e,
    int STAGES >
struct Sparse_kernel_traits_adapter<xmma::Ampere_spimma_int8_rt_fuse_traits,
                                    Cta_tile,
                                    Gmem_tile_a,
                                    Gmem_tile_b,
                                    Gmem_tile_c,
                                    Gmem_tile_e,
                                    STAGES > {
    using Type = Sparse_kernel_traits<xmma::Ampere_spimma_int8_rt_fuse_traits,
                                      Cta_tile,
                                      Gmem_tile_a,
                                      Gmem_tile_b,
                                      Gmem_tile_c,
                                      Gmem_tile_e,
                                      STAGES,
                                      xmma::RT_ACT>;
};

template<
    typename Traits,
    typename Cta_tile,
    typename Gmem_tile_a,
    typename Gmem_tile_b,
    typename Gmem_tile_c,
    typename Gmem_tile_e,
    int STAGES
>
using Kernel_traits = typename Sparse_kernel_traits_adapter<Traits,
                                                            Cta_tile,
                                                            Gmem_tile_a,
                                                            Gmem_tile_b,
                                                            Gmem_tile_c,
                                                            Gmem_tile_e,
                                                            STAGES>::Type;

///////////////////////////////////////////////////////////////////////////////////////////////////

}
} // namespace gemm
} // namespace ext
} // namespace xmma

