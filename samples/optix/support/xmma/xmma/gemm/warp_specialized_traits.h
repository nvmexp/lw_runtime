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
#include <xmma/compute_tile.h>
#include <xmma/params.h>
#include <xmma/warp_specialized_traits.h>

#include <xmma/volta/traits.h>
#include <xmma/turing/traits.h>
#include <xmma/ampere/traits.h>
#include <xmma/hopper/traits.h>

#include <xmma/helpers/epilogue_with_split_k.h>

#include <xmma/gemm/gmem_tile.h>
#include <xmma/gemm/traits.h>
#include <xmma/gemm/kernel.h>
#include <xmma/gemm/warp_specialized_params.h>
#include <xmma/gemm/warp_specialized_kernel.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace gemm {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits
    typename Traits_,
    // The CTA tile descriptor
    typename Cta_tile_,
    // The global memory tile for A (transpose or not)
    typename Gmem_tile_a_,
    // The global memory tile for B (transpose or not)
    typename Gmem_tile_b_,
    // The arch being compiled for this warp specialized kernel.
    int32_t ARCH_ = 80>
struct Warp_specialized_kernel_traits
    : public xmma::gemm::Kernel_traits<Traits_, Cta_tile_, Gmem_tile_a_, Gmem_tile_b_, 1> {
    // The base class
    using Base = xmma::gemm::Kernel_traits<Traits_, Cta_tile_, Gmem_tile_a_, Gmem_tile_b_, 1>;
    // Operation type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::GEMM;
    // Whether use warp specialized.
    static const bool USE_WARP_SPECIALIZATION = true;
    // The warp specialized kernel traits type.
    enum { ARRIVE_WAIT = 0, NAMED_BARRIER = 1 };
    // The arch for smem allocation.
    enum { ARCH = ARCH_ };

    // The instruction traits.
    using Traits = Traits_;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // The kernel parameters.
    using Params = xmma::gemm::Warp_specialized_params<Traits, Cta_tile>;
    // The global memory loader for A.
    using Gmem_tile_a = typename Base::Gmem_tile_a;
    // The global memory loader for B.
    using Gmem_tile_b = typename Base::Gmem_tile_b;
    // The shared memory layout for A.
    using Smem_layout_a = typename Gmem_tile_a::Smem_layout;
    // The shared memory layout for B.
    using Smem_layout_b = typename Gmem_tile_b::Smem_layout;
    // The warp specialized kernel traits.
    using Warp_specialized_traits = typename xmma::Warp_specialized_traits_selector<
        Traits,
        Cta_tile,
        Smem_layout_a,  // These are A/B gmem layouts, but smem layouts are the same so
        Smem_layout_b,  // use these for now. Maybe add "Gmem_layout" to Gmem_tile later?
        xmma::Row,
        Gmem_tile_a::BYTES_PER_LDG,
        Gmem_tile_b::BYTES_PER_LDG,
        ARCH,
        ARRIVE_WAIT>::Class;

    // The global memory loader for epilogue
    using Gmem_tile_epilogue = typename Base::Gmem_tile_epilogue;   
    // The callback for epilgoue.
    using Callbacks_epilogue = typename Base::Callbacks_epilogue;
    // The Epilogue with splik
    using Epilogue_withsplitk = xmma::helpers::
        Epilogue_with_split_k<Traits, Cta_tile, xmma::Row, Gmem_tile_epilogue, Callbacks_epilogue>;
    // The Epilogue without splik
    using Epilogue_wosplitk = xmma::helpers::
        Epilogue<Traits, Cta_tile, xmma::Row, Gmem_tile_epilogue, Callbacks_epilogue>;

    // Tile distribution_persisitent
    using Tile_distribution_persistent =
        typename Warp_specialized_traits::Tile_distribution_persistent;

    enum { WARP_SPECIALIZED_CONFIG = Warp_specialized_traits::WARP_SPECIALIZED_CONFIG };

    enum { BUFFERS_PER_SMEM_TILE_A = Warp_specialized_traits::BUFFERS_PER_SMEM_TILE_A };
    enum { BUFFERS_PER_SMEM_TILE_B = Warp_specialized_traits::BUFFERS_PER_SMEM_TILE_B };

    // The shared memory loader for A.
    using Smem_tile_a = xmma::Smem_tile_a<Traits,
                                          Cta_tile,
                                          Smem_layout_a,
                                          Gmem_tile_a::BYTES_PER_LDG,
                                          BUFFERS_PER_SMEM_TILE_A, true, 
                                          Gmem_tile_a::COPY_ENGINE == Copy_engine::CE_UTMALDG>;
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<Traits,
                                          Cta_tile,
                                          Smem_layout_b,
                                          Gmem_tile_b::BYTES_PER_LDG,
                                          BUFFERS_PER_SMEM_TILE_B, true, 
                                          Gmem_tile_b::COPY_ENGINE == Copy_engine::CE_UTMALDG>;
    // The compute tile.
    using Compute_tile = xmma::Compute_tile<Traits, Cta_tile, Smem_tile_a, Smem_tile_b>;

    enum {
        SMEM_BYTES_PER_CTA = Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE +
                             Warp_specialized_traits::EPILOGUE_SIZE_IN_BYTES +
                             Warp_specialized_traits::ARRIVE_WAIT_SMEM_SIZE
    };

    static_assert( (int)SMEM_BYTES_PER_CTA <= (int)Warp_specialized_traits::SMEM_BYTES_PER_SM,
                   "error: Shared memory needed exceeds capacity" );

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

    static int32_t threads_per_cta( const Params params = Params() ) {
        if( params.specialize == xmma::CONFIG_1DMA_2MATH ) {
            return Cta_tile::THREADS_PER_CTA * 3;
        }
        // FixMe: need to refactor Cta_tile::THREADS_PER_CTA to Cta_tile::THREADS_PER_WARPGROUP
        else if( params.specialize == xmma::CONFIG_1DMA_1MATH ) {
            return Cta_tile::THREADS_PER_CTA * 2;
        }
        else {
            return Cta_tile::THREADS_PER_CTA;
        }
    }

    static int32_t dynamic_smem_size_per_cta() {
        // The amount of shared memory to launch the kernel.
        return SMEM_BYTES_PER_CTA;
    }

    static int32_t epilogue_smem_size_per_cta() {
        // The amount of shared memory needed in epilogue.
        return Warp_specialized_traits::EPILOGUE_SIZE_IN_BYTES;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef USE_GMMA

template <
    // The instruction traits
    typename Traits_,
    // The CTA tile descriptor
    typename Cta_tile_,
    // The global memory tile for A (transpose or not)
    typename Gmem_tile_a_,
    // The global memory tile for B (transpose or not)
    typename Gmem_tile_b_,
    // The layout of C/D, could be row major or column major.
    typename Layout_c_,
    // The number of GMMA stages to be issued in the prologue.
    int STAGES_ = 1,
    // The number of GMMA Stages to be issued in the prologoue.
    int GMMA_STAGES_ = 1>
struct Warp_specialized_gmma_kernel_traits 
    : public xmma::gemm::Gmma_kernel_traits<Traits_, Cta_tile_, Gmem_tile_a_, 
    					    Gmem_tile_b_, Layout_c_, 1, GMMA_STAGES_> {
    // The base class
    using Base = xmma::gemm::Gmma_kernel_traits<Traits_, Cta_tile_, Gmem_tile_a_, 
    						Gmem_tile_b_, Layout_c_, 1, GMMA_STAGES_>;
    // Operation type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::GEMM;
    // Whether use warp specialized.
    static const bool USE_WARP_SPECIALIZATION = true;

    // The instruction traits.
    using Traits = Traits_;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // The kernel parameters.
    using Params = xmma::gemm::Warp_specialized_params<Traits, Cta_tile>;
    // The global memory loader for A.
    using Gmem_tile_a = typename Base::Gmem_tile_a;
    // The global memory loader for B.
    using Gmem_tile_b = typename Base::Gmem_tile_b;
    // The shared memory layout for A.
    using Smem_layout_a = typename Gmem_tile_a::Smem_layout;
    // The shared memory layout for B.
    using Smem_layout_b = typename Gmem_tile_b::Smem_layout;
    
    
    // The global memory loader for epilogue
    using Gmem_tile_epilogue = typename Base::Gmem_tile_epilogue;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = typename Base::Swizzle_epilogue;
    // The callback for epilgoue.
    using Callbacks_epilogue = typename Base::Callbacks_epilogue;
    // The Epilogue with splik
    using Epilogue = typename Base::Epilogue;//xmma::helpers::
    
    
    
    // The shared memory tile for A.
    // Only 1 buffer in the smem tile.
    // Only used this to get the A buffer size
    using Smem_tile_a_temporary = xmma::Smem_tile_hopper_a<Traits,
                                          Cta_tile,
                                          Smem_layout_a,
                                          1,
					  Gmem_tile_a::GMMA_DESC_MODE>;
    // The shared memory tile for B.
    // Only 1 buffer in the smem tile.
    // Only used this to get the B buffer size
    using Smem_tile_b_temporary = xmma::Smem_tile_hopper_b<Traits,
                                          Cta_tile,
                                          Smem_layout_b,
                                          1,
					  Gmem_tile_b::GMMA_DESC_MODE>;
    // SMEM BUffer A and B size.
    // In Timmy's hopper smem A/B tile, BYTES_PER_BUFFER != BYTES_PER_BUFFER_TILE here, 
    // as it adds BYTES_FOR_ALIGNMENT at the end of tile.
    enum { A_SIZE_IN_BYTES_PER_BUFFER = Smem_tile_a_temporary::BYTES_PER_BUFFER };  
    // In Timmy's hopper smem A/B tile, BYTES_PER_BUFFER != BYTES_PER_BUFFER_TILE here,
    // as it adds BYTES_FOR_ALIGNMENT at the end of tile.
    enum { B_SIZE_IN_BYTES_PER_BUFFER = Smem_tile_b_temporary::BYTES_PER_BUFFER };  
    // Epilogue SMEM buffer size.
    enum { EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE };

    // Hopper SMEM capacity:  228KB.
    // FixMe: need to confirm the final decision from lwca driver on the reserved size.
    //enum { SMEM_BYTES_PER_SM = 233472  };
    enum { SMEM_BYTES_PER_SM = 229376  };
    // The shared memory size (Bytes) reserved for barriers
    enum { BARRIER_RESERVE_BYTES = 512 };
    // Added in Timmy's hopper smem_tile A/B, can we remove it???
    // The size in bytes of total buffers.
    // +128 byte to guarantee that the base address can be aligned to 128B
    enum { BYTES_FOR_ALIGNMENT = 128 };

    // The number of buffer B
    enum { 
       BUFFERS_PER_SMEM_TILE_B = (
                      SMEM_BYTES_PER_SM -
		      EPILOGUE_SIZE_IN_BYTES-
		      BARRIER_RESERVE_BYTES - 
		      BYTES_FOR_ALIGNMENT -
		      BYTES_FOR_ALIGNMENT) /
		      (A_SIZE_IN_BYTES_PER_BUFFER + 
		      B_SIZE_IN_BYTES_PER_BUFFER)
		};
    enum { BUFFERS_PER_SMEM_TILE_A = BUFFERS_PER_SMEM_TILE_B };

    // Tile distribution_persisitent
    //using Tile_distribution_persistent =  xmma::Tile_distribution_persistent;
    using Tile_distribution_persistent =  xmma::Tile_distribution_persistent_hopper;

    enum { WARP_SPECIALIZED_CONFIG = xmma::CONFIG_1DMA_2MATH };

    static constexpr bool GMMA_A_RF = Traits::GMMA_A_RF;

    static constexpr bool GMMA_B_RF = Traits::GMMA_B_RF;

    enum {
        TRAITS_USE_UTMALDG = Gmem_tile_a::COPY_ENGINE == Copy_engine::CE_UTMALDG ||
                             Gmem_tile_b::COPY_ENGINE == Copy_engine::CE_UTMALDG
    };

    // The shared memory loader for A.
    using Smem_tile_a = xmma::Smem_tile_hopper_a<Traits,
                                          Cta_tile,
                                          Smem_layout_a,
                                          BUFFERS_PER_SMEM_TILE_A,
					  Gmem_tile_a::GMMA_DESC_MODE, GMMA_A_RF, xmma::Gmma_fusion_mode::NO_FUSION, Gmem_tile_a::USE_UTMALDG>;
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_hopper_b<Traits,
                                          Cta_tile,
                                          Smem_layout_b,
                                          BUFFERS_PER_SMEM_TILE_B,
					  Gmem_tile_b::GMMA_DESC_MODE, Gmem_tile_b::USE_UTMALDG>;
    // The compute tile.
    using Compute_tile = wip_do_not_use::Compute_tile_with_gmma<Traits, 
                                                                Cta_tile, 
                                                                Smem_tile_a, 
                                                                Smem_tile_b, 
                                                                Traits::GMMA_A_RF,
                                                                Traits::GMMA_B_RF,
                                                                STAGES_>;
    // The size used for shared memory barriers, 
    // each smem barrier is 8 bytes, each buffer has 2 barriers
    // we have  BUFFERS_PER_SMEM_TILE_A*2 barriers
    enum { SMEM_BARRIER_SIZE_IN_BYTES = BUFFERS_PER_SMEM_TILE_A * 2 * 8 };


    // Total shared memory size used by the kernel
    // Smem_tile_a::BYTES_PER_TILE  is the size for holding A buffers.
    // Smem_tile_b::BYTES_PER_TILE  is the size for holding B fuffers.
    // EPILOGUE_SIZE_IN_BYTES  the size used in epilogue
    // SMEM_BARRIER_SIZE_IN_BYTES  the size used in shared memory barriers.
    enum {
        SMEM_BYTES_PER_CTA =
            Smem_tile_a::BYTES_PER_TILE +
            Smem_tile_b::BYTES_PER_TILE  +
            EPILOGUE_SIZE_IN_BYTES +
	    SMEM_BARRIER_SIZE_IN_BYTES
    };

    static_assert( (int)SMEM_BYTES_PER_CTA <= (int)SMEM_BYTES_PER_SM,
                   "error: Shared memory needed exceeds capacity" );

    // The number of threads in the CTA.
    // FIXME : need to refactor Cta_tile::THREADS_PER_CTA to Cta_tile::THREADS_PER_WARPGROUP
    static int threads_per_cta() {
        return Cta_tile::THREADS_PER_CTA * 3;
    }

    // The amount of shared memory used per CTA.
    static int dynamic_smem_size_per_cta() {
	//printf("SMEM_BYTES_PER_CTA %d EPILOGUE_SIZE_IN_BYTES  %d  A_BUFFER_SIZE %d 
	// B_BUFFER_SIZE %d A_B_Buffer_Num %d  SMEM_BARRIER_SIZE_IN_BYTES %d \n",
	// SMEM_BYTES_PER_CTA,  EPILOGUE_SIZE_IN_BYTES, A_SIZE_IN_BYTES_PER_BUFFER, 
	// B_SIZE_IN_BYTES_PER_BUFFER, BUFFERS_PER_SMEM_TILE_A, SMEM_BARRIER_SIZE_IN_BYTES);
        // The amount of shared memory to launch the kernel.
        return SMEM_BYTES_PER_CTA;
    }
 
};
#endif
}  // namespace gemm
}  // namespace xmma
