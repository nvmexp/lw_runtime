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

#include <xmma/jetfire/jetfire.h>
#include <xmma/ext/sparse/utils.h>

#include <xmma/helpers/fragment.h>
#include <xmma/ext/sparse/helpers/gemm.h>
#include <xmma/ext/sparse/implicit_spgemm/fprop/gmem_tile.h>
#include <xmma/ext/sparse/ampere/smem_tile_sparse.h>
#include <xmma/ext/sparse/helpers/epilogue_sparse.h>

// #define SPLIT_2_ITER

namespace xmma {
namespace ext {
namespace implicit_gemm {

template< typename Kernel_traits, bool WITH_RESIDUAL = true>
static __global__ __launch_bounds__(Kernel_traits::Cta_tile::THREADS_PER_CTA)
void kernel(typename Kernel_traits::Params params) {

    // The traits class.
    using Traits = typename Kernel_traits::Traits;
    // The CTA tile.
    using Cta_tile = typename Kernel_traits::Cta_tile;
    // The XMMA tile.
    using Xmma_tile = typename Kernel_traits::Xmma_tile;
    // The tile distribution manager.
    using Tile_distribution_traits = typename Kernel_traits::Tile_distribution;

    // The number of XMMAs.
    const int XMMAS_M = Xmma_tile::XMMAS_M;
    const int XMMAS_N = Xmma_tile::XMMAS_N;

    // The block indices.
    int bidm, bidn, bidz;

    Tile_distribution_traits tile(params, blockIdx);
    bidm = tile.bidm();
    bidn = tile.bidn();
    bidz = tile.bidz();

    // The thread index.
    const int tidx = threadIdx.x;

    // Ampere memory descritors.
    const uint64_t mem_desc_a = params.mem_descriptors.descriptor_a;
    const uint64_t mem_desc_b = params.mem_descriptors.descriptor_b;

    // The tiles in global memory for the images.
    using Gmem_tile_a = typename Kernel_traits::Gmem_tile_a;
    // The tiles to store the data in shared memory for the images.
    using Smem_tile_a = typename Kernel_traits::Smem_tile_a;
    // The fragment.
    using Fragment_a = typename Smem_tile_a::Fragment;

    // The tiles in global memory for the filters.
    using Gmem_tile_b = typename Kernel_traits::Gmem_tile_b;
    // The tiles to store the data in shared memory for the filters.
    using Smem_tile_b = typename Kernel_traits::Smem_tile_b;
    // The fragment.
    using Fragment_b = typename Smem_tile_b::Fragment;

    // The tiles in global memory for the metadata.
    using Gmem_tile_e = typename Kernel_traits::Gmem_tile_e;
    // The tiles to store the data in shared memory for the filters.
    using Smem_tile_e = typename Kernel_traits::Smem_tile_e;
    // The fragment.
    using Fragment_e = typename Smem_tile_e::Fragment;

    // The tile in global memory.
    using Gmem_tile_epilogue = typename Kernel_traits::Gmem_tile_epilogue;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = typename Kernel_traits::Swizzle_epilogue;
    // The callbacks.
    using Callbacks_epilogue = typename Kernel_traits::Callbacks_epilogue;
    // The epilogue.
    using Epilogue = typename Kernel_traits::Epilogue;

    // The shared memory. It is allocated at launch time.
    extern __shared__ char smem_[];

    // The shared memory pointers.
    char *a_smem_ = &smem_[0];
    char *b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];
    char *e_smem_ =
        &smem_[Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE];
    char *i_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE
        + Smem_tile_b::BYTES_PER_TILE
        + Smem_tile_e::BYTES_PER_TILE];

    Smem_tile_a a_smem(a_smem_, tidx);
    Smem_tile_b b_smem(b_smem_, tidx);
    Smem_tile_e e_smem(e_smem_, tidx);

    Gmem_tile_a a_gmem(params, NULL, tile.bidx(), tidx);
    Gmem_tile_b b_gmem(params, i_smem_, tile.bidx(), tidx);
    Gmem_tile_e e_gmem(params, NULL, tile.bidx(), tidx);

    xmma::Fragment_aclwmulator<Traits> acc[XMMAS_M][XMMAS_N];
    Fragment_a a[2][XMMAS_M];
    Fragment_b b[2][XMMAS_N];
    Fragment_e e[2][1];

    // The filter index.
    int64_t a_delta, b_delta;
    int trsi = Kernel_traits::initialize_filter_position( params );

    enum { SPLIT_XMMAS_N = Cta_tile::N != 160 ? 0 : 2};

    // Do we use LDGSTS in that kernel?
    enum { USE_LDGSTS = Gmem_tile_a::USE_LDGSTS || Gmem_tile_b::USE_LDGSTS };
    enum { PREFETCH_STAGES = Kernel_traits::STAGES - 1 };
    const int loop_count = params.loop_start + 1;
    const int loop_to_enter_residue = params.loop_start - (params.loop_residue - PREFETCH_STAGES);

    // Clear the aclwmulators.
    xmma::helpers::clear(acc);

    // Only enable fencing if it looks like our desired stage overlap will fit in register budget.
    constexpr bool JETFIRE_FENCING_ENABLED = 1;

    for( int stage = 0, prefetch = min( loop_count, PREFETCH_STAGES ); stage < prefetch; ++stage ) {        
        /* COPY FROM GLOBAL TO SMEM */
        e_gmem.load(e_smem);
        b_gmem.load(b_smem, mem_desc_b);
        a_gmem.load(a_smem, mem_desc_a);

        xmma::ldgdepbar<USE_LDGSTS>();

        b_smem.move_next_write_buffer();
        a_smem.move_next_write_buffer();
        e_smem.move_next_write_buffer();

        trsi = Kernel_traits::load_deltas_and_move_filter_position( a_delta,
            b_delta, params, trsi );

        a_gmem.move(0, 0);
        b_gmem.move_b(trsi, b_delta, params);
        
        // Decide whether we should move the pointer, to avoid out-of-bound GMEM access
        if (stage + 1 <= params.loop_start) {
            e_gmem.move();
        }

        if (params.loop_start == 0) {
            //a_gmem.disable_loads();
            b_gmem.disable_loads();
            continue;
        }

        //if (stage == loop_count - PREFETCH_STAGES) {
        if (stage == loop_to_enter_residue) {
            b_gmem.residue();
            //a_gmem.residue();
        }
    }

#if !defined(__LWDACC_RTC__)
    #pragma unroll 1
#endif
    for( int ii = loop_count; ii < PREFETCH_STAGES; ++ii ) {
        xmma::ldgdepbar<USE_LDGSTS>();
    }

    /* SYNC SMEM CHUNK 0*/
    xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>(); __syncthreads();

    // Residue prefetch
    using Gmem_tile_epilogue_prefetch = typename Kernel_traits::Gmem_tile_epilogue_prefetch;
    if (params.with_residual) {
        Gmem_tile_epilogue_prefetch prefetch(params, bidm, bidn, bidz, tidx);
        prefetch.prefetch();
    }

    /* LOAD CHUNK 0*/
    a_smem.load(a[0], 0);
    b_smem.load(b[0], 0);
    e_smem.load(e[0], 0);

#ifdef XMMA_ENABLED_PMTRIG
    __prof_trigger(15);
#endif

#ifdef SPLIT_2_ITER
    // int main_iter = min(PREFETCH_STAGES - 1, 1);
    enum { MAIN_ITER = Min<PREFETCH_STAGES - 1, 1>::VALUE };
    JETFIRE_MAC_LOOP_PRAGMA
    #pragma unroll (1)
    for (int c = params.loop_start; c > MAIN_ITER; c--) { // Extract 2 iterations
#else
    JETFIRE_MAC_LOOP_PRAGMA
    #pragma unroll (1)    
    for (int c = params.loop_start; c > (PREFETCH_STAGES - 1); c--) { // Extract PREFETCH_STAGES iterations
#endif
        JETFIRE_MAC_LOOP_HEADER

        #if JETFIRE_ENABLED
        asm volatile (".pragma \"set knob BarDeferBlockingSuppLat=16\";\n" : : : "memory");
        asm volatile (".pragma \"set knob SchedReadSBBaseLatency=32\";\n" : : : "memory");
        asm volatile (".pragma \"set knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");
        #endif

#ifdef SPLIT_2_ITER   
        if(Kernel_traits::STAGES > 3) {
            if(c < PREFETCH_STAGES) {
                b_gmem.disable_loads();
            }

            // Use iteration count to decide if we want to load metadata
            if(c >= PREFETCH_STAGES) {
                e_gmem.load(e_smem);
            }
        } else {
            e_gmem.load(e_smem);
        }
#else
        e_gmem.load(e_smem);
#endif

#ifdef SPLIT_2_ITER   // Current fix for no_pred kernel
        if(Kernel_traits::USE_PREDICATES) {
            b_gmem.load(b_smem, mem_desc_b);    
        } else {
            if(Kernel_traits::STAGES > 3) {
                if(c >= PREFETCH_STAGES) {
                    b_gmem.load(b_smem, mem_desc_b);
                }
            } else {
                b_gmem.load(b_smem, mem_desc_b);
            }
        }
#else
        b_gmem.load(b_smem, mem_desc_b);
#endif        
        //b_gmem.load(b_smem, mem_desc_b); Original here we load b
        a_gmem.load(a_smem, mem_desc_a);
        xmma::ldgdepbar<USE_LDGSTS>();
        a_smem.move_next_write_buffer();
        b_smem.move_next_write_buffer();

        e_smem.load(e[1], 1);

        if (Traits::ACLWMULATOR_32BIT && Cta_tile::N != 160) {
            if(Cta_tile::M==64 && Cta_tile::N==256) {
                jetfire::ifence(JETFIRE_FENCING_ENABLED);
            }
        }

        a_smem.load(a[1], 0);

        if (Traits::ACLWMULATOR_32BIT && 
            Cta_tile::N != 160 &&
            !(Cta_tile::M==64 && Cta_tile::N==256)
            ) {
            if (( (Cta_tile::M==128 && Cta_tile::N==128) || 
                  (Cta_tile::M==256 && Cta_tile::N==128) ||
                  (Cta_tile::M==256 && Cta_tile::N==64)
            )) {
                jetfire::ifence(JETFIRE_FENCING_ENABLED);
            }
        }

        b_smem.load(b[1], 1);

        if (Cta_tile::N != 160 && !(Cta_tile::M==64 && Cta_tile::N==256) &&
            (Traits::ACLWMULATOR_32BIT == 0 ||
            !( (Cta_tile::M==128 && Cta_tile::N==128) ||
               (Cta_tile::M==256 && Cta_tile::N==128) ||
               (Cta_tile::M==256 && Cta_tile::N==64)
            ))
            ) {
            jetfire::ifence(JETFIRE_FENCING_ENABLED);
        }

        /* GEMM*/
        Kernel_traits::template gemm<0, XMMAS_M, 0, XMMAS_N>(acc, a[0], b[0], e[0]);

        xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>(); __syncthreads();
        #if JETFIRE_ENABLED
        asm volatile (".pragma \"next knob FenceCode\";\n" : : : "memory");
        #endif
        b_smem.move_next_read_buffer();
        a_smem.move_next_read_buffer();
        e_smem.move_next_read_buffer();

        Kernel_traits::template gemm<0, XMMAS_M, 0, SPLIT_XMMAS_N>(acc, a[1], b[1], e[1]);
        if (SPLIT_XMMAS_N != 0) {
            #if JETFIRE_ENABLED
            asm volatile (".pragma \"next knob FenceCode\";\n" : : : "memory");
            #endif
        }

        JETFIRE_MAC_LOOP_HEADER

        e_smem.load(e[0], 0);

        if (Traits::ACLWMULATOR_32BIT && Cta_tile::N != 160) {
            if(Cta_tile::M==64 && Cta_tile::N==256) {
                jetfire::ifence(JETFIRE_FENCING_ENABLED);
            }
        }

        a_smem.load(a[0], 0);

        if (Traits::ACLWMULATOR_32BIT && 
            Cta_tile::N != 160 &&
            !(Cta_tile::M==64 && Cta_tile::N==256)
            ) {
            if (( (Cta_tile::M==128 && Cta_tile::N==128) ||
                  (Cta_tile::M==256 && Cta_tile::N==128) ||
                  (Cta_tile::M==256 && Cta_tile::N==64)
            )) {
                jetfire::ifence(JETFIRE_FENCING_ENABLED);
            }
        }

        b_smem.load(b[0], 0);

        if (Cta_tile::N != 160 && !(Cta_tile::M==64 && Cta_tile::N==256) &&
            (Traits::ACLWMULATOR_32BIT == 0 ||
            !( (Cta_tile::M==128 && Cta_tile::N==128) ||
               (Cta_tile::M==256 && Cta_tile::N==128) ||
               (Cta_tile::M==256 && Cta_tile::N==64)
            ))
            ) {
            jetfire::ifence(JETFIRE_FENCING_ENABLED);
        }

        /* GEMM*/
        Kernel_traits::template gemm<0, XMMAS_M, SPLIT_XMMAS_N, XMMAS_N>(acc, a[1], b[1], e[1]);

        /*move global mem*/
        trsi = Kernel_traits::load_deltas_and_move_filter_position( a_delta,
            b_delta, params, trsi );

        a_gmem.move(0, 0);
        b_gmem.move_b(trsi, b_delta, params);
        if (SPLIT_XMMAS_N == 0) {
            #if JETFIRE_ENABLED
            asm volatile (".pragma \"next knob FenceCode\";\n" : : : "memory");
            #endif
        }
        e_gmem.move();
        e_smem.move_next_write_buffer();

        if (c <= params.loop_residue) {
            b_gmem.residue();
        }

        #if JETFIRE_ENABLED
        asm volatile (".pragma \"reset knob BarDeferBlockingSuppLat=16\";\n" : : : "memory");
        asm volatile (".pragma \"reset knob SchedReadSBBaseLatency=32\";\n" : : : "memory");
        asm volatile (".pragma \"reset knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");
        #endif
    }

#ifdef XMMA_ENABLED_PMTRIG
    __prof_trigger(15);
#endif
    // Do allocate the tile to output in the epilogue.
    Gmem_tile_epilogue gmem_epilogue(params, bidm, bidn, bidz, tidx);
    // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
    Swizzle_epilogue swizzle_epilogue( smem_, tidx );
    // The callbacks.
    Callbacks_epilogue callbacks( params, smem_, bidm, bidn, 0, tidx );

    Epilogue epilogue(params, gmem_epilogue, swizzle_epilogue, callbacks);
    using Fragment_post_swizzle = typename Callbacks_epilogue::Fragment_post_swizzle;
    Fragment_post_swizzle bias_regs;
    callbacks.load_bias(bias_regs);

#ifdef SPLIT_2_ITER
    // int rest_iter = min(PREFETCH_STAGES, 2);
    enum { REST_ITER = Min<PREFETCH_STAGES, 2>::VALUE };
    int remaining_iter = min(loop_count, REST_ITER);
 
    // GEMM LAST CHUNK -- 2 iterations only
    JETFIRE_MAC_LOOP_PRAGMA
    #pragma unroll (1)
    for (int loop = 0; loop < remaining_iter; loop++) {
        JETFIRE_MAC_LOOP_HEADER

        xmma::ldgdepbar<USE_LDGSTS>();

        // LOAD CHUNK 1
        #pragma unroll
        for (int ki = 0; ki < 2; ki++) {

            if (loop == 0 && ki == 1) {
                xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>(); __syncthreads();
            }
            if (loop == 0 || ki == 0) {
                a_smem.load(a[(ki+1)%2], 0);
                b_smem.load(b[(ki+1)%2], (ki+1)%2);
                e_smem.load(e[(ki+1)%2], (ki+1)%2);
                
                if(loop == 0 && ki == 0) {
                    b_smem.move_next_read_buffer();
                    a_smem.move_next_read_buffer();
                    e_smem.move_next_read_buffer();
                }
            }
            // GEMM
            jetfire::ifence(JETFIRE_FENCING_ENABLED);
            Kernel_traits::template gemm<0, XMMAS_M, 0, XMMAS_N>(acc, a[ki], b[ki], e[ki]);
        }
    }
#else
    int remaining_iter = min(loop_count, PREFETCH_STAGES);
    // GEMM LAST CHUNK -- PREFETCH_STAGES or loop_count iterations
    JETFIRE_MAC_LOOP_PRAGMA
    #pragma unroll (1)
    for (int loop = 0; loop < remaining_iter; loop++) {
        JETFIRE_MAC_LOOP_HEADER

        #if JETFIRE_ENABLED
        asm volatile (".pragma \"set knob BarDeferBlockingSuppLat=16\";\n" : : : "memory");
        asm volatile (".pragma \"set knob SchedReadSBBaseLatency=32\";\n" : : : "memory");
        asm volatile (".pragma \"set knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");
        #endif

        xmma::ldgdepbar<USE_LDGSTS>();

        #pragma unroll
        for (int ki = 0; ki < 2; ki++) {

            if (ki == 1) {
                xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>(); __syncthreads();
                b_smem.move_next_read_buffer();
                a_smem.move_next_read_buffer();
                e_smem.move_next_read_buffer();
            }
            a_smem.load(a[(ki+1)%2], 0);
            b_smem.load(b[(ki+1)%2], (ki+1)%2);
            e_smem.load(e[(ki+1)%2], (ki+1)%2);
            jetfire::ifence(JETFIRE_FENCING_ENABLED);
            Kernel_traits::template gemm<0, XMMAS_M, 0, XMMAS_N>(acc, a[ki], b[ki], e[ki]);
        }

        #if JETFIRE_ENABLED
        asm volatile (".pragma \"reset knob BarDeferBlockingSuppLat=16\";\n" : : : "memory");
        asm volatile (".pragma \"reset knob SchedReadSBBaseLatency=32\";\n" : : : "memory");
        asm volatile (".pragma \"reset knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");
        #endif
    }
#endif
    xmma::depbar<USE_LDGSTS, 2>();
    if(WITH_RESIDUAL) {
        epilogue.template execute<true>(acc, bias_regs);
    } else {
        epilogue.template execute<false>(acc, bias_regs);
    }
}

template <typename Kernel_traits>
static __global__ __launch_bounds__( Kernel_traits::Cta_tile::THREADS_PER_CTA )
void split_k_kernel( typename Kernel_traits::Params params ) {
}

} // namespace implicit_gemm
} // namespace ext
} // namespace xmma
