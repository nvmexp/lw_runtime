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

#include <xmma/helpers/epilogue_with_split_k.h>
#include <xmma/ext/sparse/helpers/gemm.h>
#include <xmma/ext/sparse/spgemm/sphmma/gmem_tile.h>
#include <xmma/ext/sparse/spgemm/sphmma/params.h>
#include <xmma/ext/sparse/smem_tile.h>
#include <xmma/ext/sparse/spgemm/sphmma/callbacks.h>
#include <xmma/ext/sparse/spgemm/sphmma/kernel.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace gemm {
namespace sparse_hmma_gemm {

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
    int STAGES_
>
struct Kernel_traits {
};
///////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    // typename Traits_, 
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
    int STAGES_
>
struct Kernel_traits <xmma::Ampere_sphmma_tf32_traits<float, float>,
                      Cta_tile_,
                      Gmem_tile_a_,
                      Gmem_tile_b_,
                      Gmem_tile_c_,
                      Gmem_tile_e_,
                      STAGES_>: public xmma::Ampere_sphmma_tf32_traits<float, float> {

    static const bool USE_SPARSE_IMMA = false;
    static const bool HAS_2_KERNEL_SPLITK = true;

    // The traits class.
    using Traits = xmma::Ampere_sphmma_tf32_traits<float, float>;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // The number of STAGES.
    enum { STAGES = STAGES_ };

    // The XMMA tile.
    using Xmma_tile = typename Traits::Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = Xmma_sparse_gemm_params<Traits, Cta_tile, STAGES>;

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
                                              4,
                                              BUFFERS_PER_SMEM_TILE_E>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = Gmem_tile_c_;
    // The output layout
    using Layout = typename Gmem_tile_epilogue::output_layout;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue<Traits, Cta_tile, Layout>;
    // The callbacks.
    using Callbacks_epilogue = xmma::ext::gemm::sparse_hmma_gemm::Callbacks_epilogue_with_bias_and_relu<float, 
                                                                                                        float, 
                                                                                                        Traits, 
                                                                                                        Layout, 
                                                                                                        Cta_tile>;
    // The epilogue.                                              
    using Epilogue = xmma::helpers::Epilogue_with_split_k<Traits, 
                                                              Cta_tile, 
                                                              Layout,
                                                              Gmem_tile_epilogue,
                                                              Callbacks_epilogue>;

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
        const int EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES );
    }
#if !defined(__LWDACC_RTC__)
    typedef void (*Kernel_type)(Params params);

    // Return device kernel function pointer.    
    static XMMA_HOST Kernel_type kernel_ptr(const Params params = Params()) {
        return &::xmma::ext::gemm::sparse_hmma_gemm::kernel<Kernel_traits>;
    }

    // Return split k kernel function pointer.
    static XMMA_HOST Kernel_type split_k_kernel_ptr() {
        return &::xmma::ext::gemm::sparse_hmma_gemm::split_k_kernel<Kernel_traits>;
    }
#endif
};


///////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    // typename Traits_, 
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
    int STAGES_
>
struct Kernel_traits <xmma::Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t>,
                      Cta_tile_,
                      Gmem_tile_a_,
                      Gmem_tile_b_,
                      Gmem_tile_c_,
                      Gmem_tile_e_,
                      STAGES_>: public xmma::Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t> {

    static const bool USE_SPARSE_IMMA = false;
    static const bool HAS_2_KERNEL_SPLITK = true;

    // The traits class.
    using Traits = xmma::Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t>;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // The number of STAGES.
    enum { STAGES = STAGES_ };

    // The XMMA tile.
    using Xmma_tile = typename Traits::Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = Xmma_sparse_gemm_params<Traits, Cta_tile, STAGES>;

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
                                              4,
                                              BUFFERS_PER_SMEM_TILE_E>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = Gmem_tile_c_;
    // The output layout
    using Layout = typename Gmem_tile_epilogue::output_layout;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue<Traits, Cta_tile, Layout>;
    // The callbacks.
    using Callbacks_epilogue = xmma::ext::gemm::sparse_hmma_gemm::Callbacks_epilogue_with_bias_and_relu<lwtlass::float_tf32_t, 
                                                                                                        lwtlass::float_tf32_t, 
                                                                                                        Traits, 
                                                                                                        Layout, 
                                                                                                        Cta_tile>;
    // The epilogue.                                              
    using Epilogue = xmma::helpers::Epilogue_with_split_k<Traits, 
                                                              Cta_tile, 
                                                              Layout,
                                                              Gmem_tile_epilogue,
                                                              Callbacks_epilogue>;

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
        const int EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES );
    }
#if !defined(__LWDACC_RTC__)
    typedef void (*Kernel_type)(Params params);

    // Return device kernel function pointer.    
    static XMMA_HOST Kernel_type kernel_ptr(const Params params = Params()) {
        return &::xmma::ext::gemm::sparse_hmma_gemm::kernel<Kernel_traits>;
    }

    // Return split k kernel function pointer.
    static XMMA_HOST Kernel_type split_k_kernel_ptr() {
        return &::xmma::ext::gemm::sparse_hmma_gemm::split_k_kernel<Kernel_traits>;
    }
#endif
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    // typename Traits_, 
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
    int STAGES_
>
struct Kernel_traits <xmma::Ampere_sphmma_fp32_traits,
                      Cta_tile_,
                      Gmem_tile_a_,
                      Gmem_tile_b_,
                      Gmem_tile_c_,
                      Gmem_tile_e_,
                      STAGES_>: public xmma::Ampere_sphmma_fp32_traits {

    static const bool USE_SPARSE_IMMA = false;
    static const bool HAS_2_KERNEL_SPLITK = true;

    // The traits class.
    using Traits = xmma::Ampere_sphmma_fp32_traits;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // The number of STAGES.
    enum { STAGES = STAGES_ };

    // The XMMA tile.
    using Xmma_tile = typename Traits::Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = Xmma_sparse_gemm_params<Traits, Cta_tile, STAGES>;

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
                                              4,
                                              BUFFERS_PER_SMEM_TILE_E>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = Gmem_tile_c_;
    // The output layout
    using Layout = typename Gmem_tile_epilogue::output_layout;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue<Traits, Cta_tile, Layout>;
    // The callbacks.
    using Callbacks_epilogue = xmma::ext::gemm::sparse_hmma_gemm::Callbacks_epilogue_with_bias_and_relu<lwtlass::half_t, 
                                                                                                        lwtlass::half_t, 
                                                                                                        Traits, 
                                                                                                        Layout, 
                                                                                                        Cta_tile>;
    // The epilogue.                                              
    using Epilogue = xmma::helpers::Epilogue_with_split_k<Traits, 
                                                              Cta_tile, 
                                                              Layout,
                                                              Gmem_tile_epilogue,
                                                              Callbacks_epilogue>;

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
        const int EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES );
    }
#if !defined(__LWDACC_RTC__)
    typedef void (*Kernel_type)(Params params);

    // Return device kernel function pointer.    
    static XMMA_HOST Kernel_type kernel_ptr(const Params params = Params()) {
        return &::xmma::ext::gemm::sparse_hmma_gemm::kernel<Kernel_traits>;
    }

    // Return split k kernel function pointer.
    static XMMA_HOST Kernel_type split_k_kernel_ptr() {
        return &::xmma::ext::gemm::sparse_hmma_gemm::split_k_kernel<Kernel_traits>;
    }
#endif
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    // typename Traits_, 
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
    int STAGES_
>
struct Kernel_traits <xmma::Ampere_sphmma_bf16_fp32_bf16_traits,
                      Cta_tile_,
                      Gmem_tile_a_,
                      Gmem_tile_b_,
                      Gmem_tile_c_,
                      Gmem_tile_e_,
                      STAGES_>: public xmma::Ampere_sphmma_bf16_fp32_bf16_traits {

    static const bool USE_SPARSE_IMMA = false;
    static const bool HAS_2_KERNEL_SPLITK = true;

    // The traits class.
    using Traits = xmma::Ampere_sphmma_bf16_fp32_bf16_traits;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // The number of STAGES.
    enum { STAGES = STAGES_ };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = Xmma_sparse_gemm_params<Traits, Cta_tile, STAGES>;

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
                                              4,
                                              BUFFERS_PER_SMEM_TILE_E>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = Gmem_tile_c_;
    // The output layout
    using Layout = typename Gmem_tile_epilogue::output_layout;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue<Traits, Cta_tile, Layout>;
    // The callbacks.
    using Callbacks_epilogue = xmma::ext::gemm::sparse_hmma_gemm::Callbacks_epilogue_with_bias_and_relu<lwtlass::float_bf16_t, 
                                                                                                        lwtlass::float_bf16_t, 
                                                                                                        Traits, 
                                                                                                        Layout, 
                                                                                                        Cta_tile>;
    // The epilogue.
    using Epilogue = xmma::helpers::Epilogue_with_split_k<Traits,
                                                              Cta_tile,
                                                              Layout,
                                                              Gmem_tile_epilogue,
                                                              Callbacks_epilogue>;

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
        const int EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES );
    }
#if !defined(__LWDACC_RTC__)
    typedef void (*Kernel_type)(Params params);

    // Return device kernel function pointer.    
    static XMMA_HOST Kernel_type kernel_ptr(const Params params = Params()) {
        return &::xmma::ext::gemm::sparse_hmma_gemm::kernel<Kernel_traits>;
    }

    // Return split k kernel function pointer.
    static XMMA_HOST Kernel_type split_k_kernel_ptr() {
        return &::xmma::ext::gemm::sparse_hmma_gemm::split_k_kernel<Kernel_traits>;
    }
#endif
};

///////////////////////////////////////////////////////////////////////////////////////////////////
}
} // namespace gemm
} // namespace ext
} // namespace xmma

