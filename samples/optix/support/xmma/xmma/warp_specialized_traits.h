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

#include <xmma/params.h>
#include <xmma/tile_distribution.h>
#include <xmma/compute_tile.h>

#include <xmma/volta/traits.h>
#include <xmma/turing/traits.h>
#include <xmma/ampere/traits.h>

#include <xmma/helpers/epilogue_with_split_k.h>

#include <xmma/implicit_gemm/fprop/params.h>
#include <xmma/implicit_gemm/fprop/gmem_tile.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {

// The two warp_specialized kernel configs:
// 1DMA_1MATH means 1 warpgroup  for doing buffer loading sequence from global memory to shared memory (DMA ops) , 1 warpgroup for math and epilog sequence.   Each warpgroup consists of 4 warps.
// 1DMA_2MATH means 1 warpgroup  for doing buffer loading sequence from global memory to shared memory (DMA ops) , 2 warpgroups for math and epilog sequence  where  2 warp group exelwtes in a ping-pong manner. Each warpgroup consists of 4 warps.
enum {
    CONFIG_1DMA_1MATH = 1,
    CONFIG_1DMA_2MATH = 2
};

template< typename Traits_,
          typename Cta_tile_,
          typename Layout_a,
          typename Layout_b,
          typename Layout_epilogue,
          int Byte_per_ldg_a,
          int Byte_per_ldg_b,
          int Arch,
          int Extra_bytes_dma,
          int Extra_bytes_epi > //, int SMEM_BYTES_PER_SM_ = 167936 >
struct Arrive_wait_warp_specialized_traits{
    // Warp specialized config.
    enum { WARP_SPECIALIZED_CONFIG = CONFIG_1DMA_1MATH };
    // Shared memory capacity.
    // Update according to LWCA 11 Spec,  Noted LWCA 11.1 will make changes,
    // likely 99KB (lwda_arch=sm_86) 183 kb (lwda_arch=sm_80/87)
    enum { SMEM_BYTES_PER_SM = Arch == 86 ? 98304 : 163840 };

    enum { INTERLEAVED = 0 };

    // The instruction traits.
    using Traits = Traits_;
    // The Cta tile.
    using Cta_tile = Cta_tile_;

    //The tile distribution
    using Tile_distribution_persistent = xmma::Tile_distribution_persistent;
    // The shared memory loader for A.
    using Smem_tile_a_temporary = xmma::Smem_tile_a<Traits, Cta_tile, Layout_a, Byte_per_ldg_a>;
    // The shared memory loader for A.
    using Smem_tile_b_temporary = xmma::Smem_tile_b<Traits, Cta_tile, Layout_b, Byte_per_ldg_b>;
    // The shared memory epilogue tile.
    using Swizzle_epilogue_temporary = xmma::Swizzle_epilogue<Traits, Cta_tile, Layout_epilogue>;
    // Setup SHM buffer size for A/B/EPILOG.
    enum { A_SIZE_IN_BYTES_PER_BUFFER = Smem_tile_a_temporary::BYTES_PER_TILE };
    enum { B_SIZE_IN_BYTES_PER_BUFFER = Smem_tile_b_temporary::BYTES_PER_TILE };
    enum { EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue_temporary::BYTES_PER_TILE + Extra_bytes_epi };
    // The group of dma warps.
    enum { DMA_TILE_PER_CTA = 1 } ;
    // The shared memory size for epilogue
    enum { SMEM_SIZE_EPILOGUE = EPILOGUE_SIZE_IN_BYTES};
    // The shared memory size for B (to do MATH_TILE_PER_CTA)
    enum { SMEM_SIZE_PER_B = B_SIZE_IN_BYTES_PER_BUFFER };
    // The shared memory size (Bytes) reserved for barriers
    enum { BARRIER_RESERVE_BYTES = 512};
    // The buffers in shared memroy tile B
    enum { BUFFERS_PER_SMEM_TILE_B = (SMEM_BYTES_PER_SM - SMEM_SIZE_EPILOGUE - BARRIER_RESERVE_BYTES
           - Extra_bytes_dma * DMA_TILE_PER_CTA) / (A_SIZE_IN_BYTES_PER_BUFFER + SMEM_SIZE_PER_B) };
    // The buffers in shared memory tile A
    enum { BUFFERS_PER_SMEM_TILE_A = BUFFERS_PER_SMEM_TILE_B };
    // The shared memory needed for arrive wait barrier state. 4 for ping-pong usage.
    enum { ARRIVE_WAIT_SMEM_SIZE = (BUFFERS_PER_SMEM_TILE_A + BUFFERS_PER_SMEM_TILE_B) * 8 + 4};
};


////////////////////////////// Interleaved ////////////////////////////////////////////

template< typename Traits_,
          typename Cta_tile_,
          typename Layout_a,
          typename Layout_b,
          typename Layout_epilogue,
          int Byte_per_ldg_a,
          int Byte_per_ldg_b,
          int Arch,
          int Extra_bytes_dma,
          int Extra_bytes_epi > //, int SMEM_BYTES_PER_SM_ = 167936 >
struct Arrive_wait_warp_specialized_interleaved_traits{
    // Warp specialized config.
    enum { WARP_SPECIALIZED_CONFIG = CONFIG_1DMA_1MATH };
    // Shared memory capacity.
    // Update according to LWCA 11 Spec,  Noted LWCA 11.1 will make changes,
    // likely 99KB (lwda_arch=sm_86) 183 kb (lwda_arch=sm_80/87)
    enum { SMEM_BYTES_PER_SM = Arch == 86 ? 98304 : 163840 };

    enum { INTERLEAVED = 1 };

    // The instruction traits.
    using Traits = Traits_;
    // The Cta tile.
    using Cta_tile = Cta_tile_;

    //The tile distribution
    using Tile_distribution_persistent = xmma::Tile_distribution_persistent;
    // The shared memory loader for A.
    using Smem_tile_a_temporary = xmma::Smem_tile_a<Traits, Cta_tile, Layout_a, Byte_per_ldg_a>;
    // The shared memory loader for A.
    using Smem_tile_b_temporary = xmma::Smem_tile_b<Traits, Cta_tile, Layout_b, Byte_per_ldg_b>;
    // The shared memory epilogue tile.
    using Swizzle_epilogue_temporary =
        xmma::Swizzle_epilogue_interleaved<Traits, Cta_tile, Layout_epilogue>;

    // Setup SHM buffer size for A/B/EPILOG.
    enum { A_SIZE_IN_BYTES_PER_BUFFER = Smem_tile_a_temporary::BYTES_PER_TILE };
    enum { B_SIZE_IN_BYTES_PER_BUFFER = Smem_tile_b_temporary::BYTES_PER_TILE };
    enum { EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue_temporary::BYTES_PER_TILE + Extra_bytes_epi };
    // The group of math warps that could be in epilog at the same time.
    enum { MATH_TILE_PER_CTA = (
           WARP_SPECIALIZED_CONFIG == CONFIG_1DMA_2MATH) ? 2 : 1 };
    // The group of dma warps.
    enum { DMA_TILE_PER_CTA = 1 } ;
    // The shared memory size for epilogue
    enum { SMEM_SIZE_EPILOGUE = EPILOGUE_SIZE_IN_BYTES * MATH_TILE_PER_CTA };
    // The shared memory size for B (to do MATH_TILE_PER_CTA)
    enum { SMEM_SIZE_PER_B = B_SIZE_IN_BYTES_PER_BUFFER * MATH_TILE_PER_CTA };
    // The buffers in shared memroy tile B
    enum { BUFFERS_PER_SMEM_TILE_B = (SMEM_BYTES_PER_SM - SMEM_SIZE_EPILOGUE - 256
           - Extra_bytes_dma * DMA_TILE_PER_CTA)
           / (A_SIZE_IN_BYTES_PER_BUFFER + SMEM_SIZE_PER_B) };
    // The buffers in shared memory tile A
    enum { BUFFERS_PER_SMEM_TILE_A = BUFFERS_PER_SMEM_TILE_B };
    // The shared memory needed for arrive wait barrier state
    enum { ARRIVE_WAIT_SMEM_SIZE = (BUFFERS_PER_SMEM_TILE_A + BUFFERS_PER_SMEM_TILE_B) * 8 };
};

///////////////////////////////////////////////////////////////////////////////////////


template< typename Traits_,
          typename Cta_tile_,
          typename Layout_a,
          typename Layout_b,
          typename Layout_epilogue,
          int Byte_per_ldg_a,
          int Byte_per_ldg_b,
          int Arch,
          int Synchronization_type,
          int Extra_bytes_dma = 0,
          int Extra_bytes_epi = 0,
          bool interleaved = false > //, int SMEM_BYTES_PER_SM_ = 167936 >
struct Warp_specialized_traits_selector{
};

template< typename Traits_,
          typename Cta_tile_,
          typename Layout_a,
          typename Layout_b,
          typename Layout_epilogue,
          int Byte_per_ldg_a,
          int Byte_per_ldg_b,
          int Arch,
          int Extra_bytes_dma,
          int Extra_bytes_epi> //, int SMEM_BYTES_PER_SM_ = 167936 >
struct Warp_specialized_traits_selector<Traits_,
                                        Cta_tile_,
                                        Layout_a,
                                        Layout_b,
                                        Layout_epilogue,
                                        Byte_per_ldg_a,
                                        Byte_per_ldg_b,
                                        Arch,
                                        0,
                                        Extra_bytes_dma,
                                        Extra_bytes_epi,
                                        false> {
    using Class = Arrive_wait_warp_specialized_traits<Traits_,
                                                      Cta_tile_,
                                                      Layout_a,
                                                      Layout_b,
                                                      Layout_epilogue,
                                                      Byte_per_ldg_a,
                                                      Byte_per_ldg_b,
                                                      Arch,
                                                      Extra_bytes_dma,
                                                      Extra_bytes_epi>;
};

template< typename Traits_,
          typename Cta_tile_,
          typename Layout_a,
          typename Layout_b,
          typename Layout_epilogue,
          int Byte_per_ldg_a,
          int Byte_per_ldg_b,
          int Arch,
          int Extra_bytes_dma,
          int Extra_bytes_epi > //, int SMEM_BYTES_PER_SM_ = 167936 >
struct Warp_specialized_traits_selector<Traits_,
                                        Cta_tile_,
                                        Layout_a,
                                        Layout_b,
                                        Layout_epilogue,
                                        Byte_per_ldg_a, Byte_per_ldg_b,
                                        Arch,
                                        0,
                                        Extra_bytes_dma,
                                        Extra_bytes_epi,
                                        true> {
    using Class = Arrive_wait_warp_specialized_interleaved_traits<Traits_,
                                                                  Cta_tile_,
                                                                  Layout_a,
                                                                  Layout_b,
                                                                  Layout_epilogue,
                                                                  Byte_per_ldg_a,
                                                                  Byte_per_ldg_b,
                                                                  Arch,
                                                                  Extra_bytes_dma,
                                                                  Extra_bytes_epi>;
};

} // namespace xmma
