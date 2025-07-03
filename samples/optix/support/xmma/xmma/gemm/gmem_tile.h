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

#include <xmma/utils.h>
#include <xmma/ampere/traits.h>
#include <xmma/hopper/traits.h>
#include <xmma/helpers/epilogue.h>

#define MIN(m, n) ((m < n) ? m : n)
namespace xmma {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits_,
    // The CTA descriptor.
    typename Cta_tile_,
    // The M dimension of the tile (depends on A/B and whether it is transposed or not).
    int M_,
    // The N dimension of the tile (depends on A/B and whether it is transposed or not).
    int N_,
    // The size if bits of each element.
    int BITS_PER_ELT_,
    // The size in bytes of the LDG.
    int BYTES_PER_LDG_
>
struct Gmem_tile_base {

    // The traits class.
    using Traits = Traits_;
    // The CTA tile descriptor.
    using Cta_tile = Cta_tile_;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The dimensions of the tile.
    enum { M = M_, N = N_, M_WITH_PADDING = Next_power_of_two<M>::VALUE };
    // The size in bits of each element.
    enum { BITS_PER_ELT = BITS_PER_ELT_ };
    // The size in bytes of each element.
    enum { BYTES_PER_ELT = BITS_PER_ELT / 8 };
    // The unroll factor for LDGS
    enum { LDGS_UNROLL = 16 / BYTES_PER_LDG_ };
    // The size in bytes of each LDG.
    enum { BYTES_PER_LDG = LDGS_UNROLL * BYTES_PER_LDG_ };

    // The number of elements per LDG.
    enum { ELTS_PER_LDG = BYTES_PER_LDG * 8 / BITS_PER_ELT };

    // The number of threads per CTA.
    enum { THREADS_PER_CTA_ = Cta_tile::THREADS_PER_CTA };
    // Make sure we have a "nice" number of elements per LDG.
    static_assert(M_WITH_PADDING % ELTS_PER_LDG == 0, "");
    // The number of threads needed to load a column. Each thread does LDG.128.
    enum { THREADS_PER_COLUMN = Min<THREADS_PER_CTA_, M_WITH_PADDING / ELTS_PER_LDG>::VALUE };
    // Make sure we have a "nice" number of pixels.
    static_assert(THREADS_PER_COLUMN > 0, "");

    // The number of rows loaded per LDG.
    enum { ROWS_PER_LDG = THREADS_PER_COLUMN * ELTS_PER_LDG };
    // Make sure we do not have more rows loaded per LDG than the CTA tile size.
    static_assert(M_WITH_PADDING % ROWS_PER_LDG == 0, "");
    // The number of LDGS needed to load a complete column.
    enum { LDGS_PER_COLUMN = M_WITH_PADDING / ROWS_PER_LDG };
    // Make sure we have at least one LDG per column.
    static_assert(LDGS_PER_COLUMN > 0, "");

    // The number of columns loaded per LDG.
    enum { COLUMNS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_COLUMN };
    // The number of steps needed to load the rows.
    enum { LDGS_PER_ROW = Div_up<N, COLUMNS_PER_LDG>::VALUE };

    // The total number of LDGs.
    enum { LDGS = LDGS_PER_COLUMN * LDGS_PER_ROW * LDGS_UNROLL };
    // Make sure we have at least one LDG.
    static_assert(LDGS > 0, "");
    // The number of phases per tile.
    enum { LDG_PHASES = Xmma_tile::XMMAS_K };
    // The number of LDGs per phase.
    enum { LDGS_PER_PHASE = (LDGS + LDG_PHASES - 1) / LDG_PHASES };

    // The number of predicate registers.
    enum { PRED_REGS = Compute_number_of_pred_regs<LDGS>::VALUE };

    // The amount of extra shared memory needed by the tile.
    enum { BYTES_PER_EXTRA_SMEM = 0 };

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_base( const Params &params,
                                      void *,
                                      int k,
                                      const void *ptr,
                                      int bidz = blockIdx.z )
        : params_k_(k)
        , params_residue_k_(params.loop_residue_k)
        , ptr_(reinterpret_cast<const char*>(ptr))
        , bidz_(bidz) {
    }

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_base( const Params &params,
                                      void *,
                                      int m,
                                      int n,
                                      int ld,
                                      const void *ptr,
                                      int bidm,
                                      int bidn,
                                      int tidx,
                                      int bidz = blockIdx.z )
        : Gmem_tile_base(params, nullptr, params.k, ptr, bidz) {

        // // DEBUG.
        // static_assert(LDGS_PER_COLUMN == 1 && LDGS_PER_ROW == 32, "");

        // The 1st row loaded by this thread.
        int row[LDGS_PER_COLUMN];
        if( THREADS_PER_COLUMN == Cta_tile::THREADS_PER_CTA ) {
            row[0] = bidm * M + tidx * ELTS_PER_LDG;
        } else {
            row[0] = bidm * M + tidx % THREADS_PER_COLUMN * ELTS_PER_LDG;
        }

        // Compute the rows loaded by this thread.
        #pragma unroll
        for( int ii = 1; ii < LDGS_PER_COLUMN; ++ii ) {
            row[ii] = row[0] + ii * ROWS_PER_LDG;
        }

        // The 1st column loaded by this thread.
        int col[LDGS_PER_ROW];
        if( THREADS_PER_COLUMN == Cta_tile::THREADS_PER_CTA ) {
            col[0] = bidn * N;
        } else {
            col[0] = bidn * N + tidx / THREADS_PER_COLUMN;
        }

        // Compute the columns loaded by this thread.
        #pragma unroll
        for( int ii = 1; ii < LDGS_PER_ROW; ++ii ) {
            col[ii] = col[0] + ii * COLUMNS_PER_LDG;
        }

        // Compute the offsets. TODO: Do we want to store 2D offsets?
        #pragma unroll
        for( int ii = 0; ii < LDGS_PER_ROW; ++ii ) {
            #pragma unroll
            for( int jj = 0; jj < LDGS_PER_COLUMN; ++jj ) {
                this->offsets_[ii * LDGS_PER_COLUMN + jj] = col[ii] * ld + row[jj];
            }
        }
        // Compute the predicates. TODO: Should we keep two arrays?
        uint32_t preds[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS_PER_ROW; ++ii ) {
            #pragma unroll
            for( int jj = 0; jj < LDGS_PER_COLUMN; ++jj ) {
                preds[ii * LDGS_PER_COLUMN + jj] = row[jj] < m && col[ii] < n;
            }
        }
        // Pack the predicates.
        xmma::pack_predicates(this->preds_, preds);
    }

    // Compute the global memory pointers.
    inline __device__ void compute_load_pointers(const void* (&ptrs)[LDGS]) const {
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = this->ptr_ + ((int64_t)this->offsets_[ii] * BYTES_PER_ELT);
        }
    }

    // Compute the global memory pointers.
    template< int PHASE >
    inline __device__ void compute_load_pointers_per_phase(const void* (&ptrs)[LDGS_PER_PHASE]) const {
        #pragma unroll
        for( int ii = PHASE * LDGS_PER_PHASE; ii < MIN((PHASE + 1) * LDGS_PER_PHASE, LDGS); ++ii ) {
            ptrs[ii - PHASE*LDGS_PER_PHASE] = this->ptr_ + ((int64_t)this->offsets_[ii] * BYTES_PER_ELT);
        }
    }

    // Disable the loads.
    inline __device__ void disable_loads() {
        #pragma unroll
        for( int ii = 0; ii < PRED_REGS; ++ii ) {
            this->preds_[ii] = 0u;
        }
    }

    // Move the pointers and update the predicates for R2P/P2R (if needed).
    inline __device__ void move(int, int64_t delta) {
        ptr_ += delta;
    }

    inline __device__ void precompute_residue_predicates_a_n_b_t(int split_k = Cta_tile::K) {
        // The thread id -- use inline PTX to avoid LWVM's rematerialization.
        int tidx;
        asm volatile("mov.b32 %0, %%tid.x;" : "=r"(tidx));

        // The 1st column loaded by this thread.
        int col;
        if( THREADS_PER_COLUMN == Cta_tile::THREADS_PER_CTA ) {
            col = this->params_residue_k_ + bidz_ * split_k;
        } else {
            col = this->params_residue_k_ + bidz_ * split_k + tidx / THREADS_PER_COLUMN;
        }

        // Populate the other columns.
        #pragma unroll
        for( int ii = 0; ii < LDGS_PER_ROW; ++ii ) {
            preds_residue_[ii] = (col + ii * COLUMNS_PER_LDG) < params_k_;
        }
    }

    // The residue to "fix" the predicates.
    inline __device__ int residue_a_n_b_t(int split_k = Cta_tile::K) {

        // The predicates. TODO: It'd be nice to update only the column preds.
        uint32_t preds[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS_PER_ROW; ++ii ) {
            #pragma unroll
            for( int jj = 0; jj < LDGS_PER_COLUMN * LDGS_UNROLL; ++jj ) {
                preds[ii * LDGS_PER_COLUMN * LDGS_UNROLL + jj] = preds_residue_[ii];
            }
        }

        // Update the predicates.
        uint32_t tmp[PRED_REGS];
        xmma::pack_predicates(tmp, preds);

        #pragma unroll
        for( int ii = 0; ii < PRED_REGS; ++ii ) {
            this->preds_[ii] &= tmp[ii];
        }

        // We did not branch back to the main loop.
        return (LDGS_PER_COLUMN * LDGS_UNROLL == 1 && preds_residue_[0]);
    }

    inline __device__ void precompute_residue_predicates_a_t_b_n(int split_k = Cta_tile::K) {
        // The coordinates -- use inline PTX to avoid LWVM's rematerialization.
        // The thread id -- use inline PTX to avoid LWVM's rematerialization.
        int tidx;
        asm volatile("mov.b32 %0, %%tid.x;" : "=r"(tidx));

        // The 1st row loaded by this thread.
        int row;
        if( THREADS_PER_COLUMN == Cta_tile::THREADS_PER_CTA ) {
            row = bidz_ * split_k + tidx * ELTS_PER_LDG;
        } else {
            row = bidz_ * split_k + tidx % THREADS_PER_COLUMN * ELTS_PER_LDG;
        }

        // Take into account the number of elements already read.
        row += this->params_residue_k_;

        // The rows loaded by this thread.
        #pragma unroll
        for( int ii = 0; ii < LDGS_PER_COLUMN; ++ii ) {
            preds_residue_[ii] = (row + ii * ROWS_PER_LDG) < params_k_;
        }
    }

    // The residue to "fix" the predicates.
    inline __device__ int residue_a_t_b_n(int split_k = Cta_tile::K) {

        // Do we jump back to the loop as we are done?
        if( LDGS_PER_COLUMN == 1 && preds_residue_[0] > 0 ) {
            return 1;
        }

        // Disable the predicates.
        if( LDGS_PER_COLUMN == 1 ) {
            #pragma unroll
            for( int ii = 0; ii < PRED_REGS; ++ii ) {
                this->preds_[ii] = 0u;
            }
        } else {
            // Compute the predicates.
            uint32_t preds[LDGS];
            #pragma unroll
            for( int ii = 0; ii < LDGS_PER_ROW; ++ii ) {
                #pragma unroll
                for( int jj = 0; jj < LDGS_PER_COLUMN; ++jj ) {
                    preds[ii * LDGS_PER_COLUMN + jj] = preds_residue_[jj];
                }
            }

            // Pack.
            uint32_t tmp[PRED_REGS];
            xmma::pack_predicates(tmp, preds);
            #pragma unroll
            for( int ii = 0; ii < PRED_REGS; ++ii ) {
                this->preds_[ii] &= tmp[ii];
            }
        }

        // We did not branch back to the main loop.
        return 0;
    }

    /////////////////////////////////////////////////////////////////
    // Main loop fusion support
    //   d = matmul(f(a, x0, ...), g(b, y0, ...)) + c
    /////////////////////////////////////////////////////////////////
    template < typename Callback_fuse >
    inline __device__ void apply_fuse( Callback_fuse &fuse ) {  }

    template < typename Callback_fuse >
    inline __device__ void load_vectors_m( Callback_fuse &fuse ) {  }

    template < typename Callback_fuse >
    inline __device__ void load_vectors_n( Callback_fuse &fuse ) {  }

    // The K dimension.
    const int params_k_, params_residue_k_;
    // The pointer.
    const char *ptr_;
    // The associated offsets.
    int offsets_[LDGS];
    // The predicates.
    uint32_t preds_[PRED_REGS];
    // The predicates for residue.
    uint32_t preds_residue_[Max<LDGS_PER_COLUMN, LDGS_PER_ROW>::VALUE];
    // The blockIdx.z passed in from tile distribution.
    int bidz_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <
    // The instruction traits.
    typename Traits,
    // The CTA descriptor.
    typename Cta_tile,
    // The M dimension of the tile (depends on A/B and whether it is transposed
    // or not).
    int M,
    // The N dimension of the tile (depends on A/B and whether it is transposed
    // or not).
    int N,
    // The size if bits of each element.
    int BITS_PER_ELT,
    // The size in bytes of the LDG.
    int BYTES_PER_LDG,
    // The base class.
    typename Base = Gmem_tile_base<Traits, Cta_tile, M, N, BITS_PER_ELT, BYTES_PER_LDG>>
struct Gmem_tile_with_ldg_and_sts : public Base {

    enum { USE_LDGSTS = 0 };
    enum { USE_UTMALDG = 0 };
    // Use LDG + STS
    static const Copy_engine COPY_ENGINE = Copy_engine::CE_DEFAULT;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_with_ldg_and_sts( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params, smem, bidx, tidx ) {
    }

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_with_ldg_and_sts( const Params &params,
                                                  void *smem,
                                                  int m,
                                                  int n,
                                                  int ld,
                                                  const void *ptr,
                                                  int bidm,
                                                  int bidn,
                                                  int tidx,
                                                  int bidz = blockIdx.z )
        : Base( params, smem, m, n, ld, ptr, bidm, bidn, tidx, bidz ) {
    }

    // Load a tile from global memory.
    template <typename Smem_tile>
    inline __device__ void load( Smem_tile &, uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const void *ptrs[Base::LDGS];
        this->compute_load_pointers( ptrs );
        xmma::ldg( this->fetch_, ptrs, this->preds_, mem_desc );
    }

    // Store the pixels to shared memory.
    template <typename Smem_tile> inline __device__ void commit( Smem_tile &smem_tile ) {
        smem_tile.store( this->fetch_ );
    }

    // The fetch registers.
    typename Uint_from_size_in_bytes<Base::BYTES_PER_LDG>::Type fetch_[Base::LDGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Base>
using Rebase_gmem_tile_with_ldg_and_sts = Gmem_tile_with_ldg_and_sts<typename Base::Traits,
                                                                     typename Base::Cta_tile,
                                                                     Base::M,
                                                                     Base::N,
                                                                     Base::BITS_PER_ELT,
                                                                     Base::BYTES_PER_LDG,
                                                                     Base>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA descriptor.
    typename Cta_tile,
    // The M dimension of the tile (depends on A/B and whether it is transposed
    // or not).
    int M,
    // The N dimension of the tile (depends on A/B and whether it is transposed
    // or not).
    int N,
    // The size if bits of each element.
    int BITS_PER_ELT,
    // The base class.
    typename Base = Gmem_tile_base<Traits, Cta_tile, M, N, BITS_PER_ELT, 16>,
    // The LDGSTS configuration.
    typename LDGSTS_CFG = xmma::Ldgsts_config<true>>
struct Gmem_tile_with_ldgsts : public Base {

    enum { USE_LDGSTS = 1 };
    enum { USE_UTMALDG = 0 };
    // Use LDGSTS
    static const Copy_engine COPY_ENGINE = Copy_engine::CE_LDGSTS;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_with_ldgsts( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params, smem, bidx, tidx ) {
    }

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_with_ldgsts( const Params &params,
                                             void *smem,
                                             int m,
                                             int n,
                                             int ld,
                                             const void *ptr,
                                             int bidm,
                                             int bidn,
                                             int tidx,
                                             int bidz = blockIdx.z )
        : Base( params, smem, m, n, ld, ptr, bidm, bidn, tidx, bidz ) {
    }

    // Trigger the different LDGSTS.
    template <typename Smem_tile>
    inline __device__ void load( Smem_tile &smem_tile, uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const void *ptrs[Base::LDGS];
        this->compute_load_pointers( ptrs );
        smem_tile.template store<Base::LDGS, Base::PRED_REGS, Base::LDGS_UNROLL, LDGSTS_CFG>(
            ptrs, this->preds_, mem_desc );
    }

    // Trigger the different LDGSTS.
    template< typename Smem_tile, int PHASE >
    inline __device__ void load_per_phase(Smem_tile &smem_tile, uint64_t mem_desc = MEM_DESC_DEFAULT) {
            if( (PHASE + 1) * Base::LDGS_PER_PHASE <= Base::LDGS ) {
                const void *ptrs[Base::LDGS_PER_PHASE];
                this->compute_load_pointers_per_phase<PHASE>(ptrs);
                smem_tile.template store_per_phase<Base::LDGS_PER_PHASE, Base::PRED_REGS, PHASE, Base::LDGS_UNROLL, LDGSTS_CFG>(
                    ptrs, this->preds_, mem_desc);
            }
    }

    // It does nothing.
    template <typename Smem_tile> inline __device__ void commit( Smem_tile & ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Base, typename LDGSTS_CFG = xmma::Ldgsts_config<true>>
using Rebase_gmem_tile_with_ldgsts = Gmem_tile_with_ldgsts<typename Base::Traits,
                                                           typename Base::Cta_tile,
                                                           Base::M,
                                                           Base::N,
                                                           Base::BITS_PER_ELT,
                                                           Base,
                                                           LDGSTS_CFG>;

template <typename Traits_, typename Cta_tile_, int M_, int N_, int BITS_PER_ELEMENT_>
struct Gmem_tile_with_utmaldg {

    using Tile_traits = Cta_tile_;

    enum { BYTES_PER_LDG = 16 };
    enum { BYTES_PER_EXTRA_SMEM = 0 };

    enum { USE_LDGSTS = 0 };
    enum { USE_UTMALDG = 1 };
    enum { USE_TMA_MULTICAST = Tile_traits::USE_TMA_MULTICAST };

    // Multicast-mask is just 16 bits, so if it happens to be bigger, 
    // We will not have functionally behaviour, per kernel.h expectation
    static_assert( USE_TMA_MULTICAST * Tile_traits::CLUSTER_COL * Tile_traits::CLUSTER_ROW <= 16,
                    "Cluster Size too big for TMA Multicasting");

    // Use UTMALDG
    static const Copy_engine COPY_ENGINE = Copy_engine::CE_UTMALDG;

    enum { NUM_TILE_BLOCKS_ROW_ = xmma::tma::kNumTileBlocksRow( M_, BITS_PER_ELEMENT_ ) };
    enum { NUM_TILE_BLOCKS_COL_ = xmma::tma::kNumTileBlocksCol( N_ ) };
    enum { TILE_BLOCK_ROW_ = xmma::tma::kTileBlockRow( M_, BITS_PER_ELEMENT_ ) };
    enum { TILE_BLOCK_COL_ = xmma::tma::kTileBlockCol( N_ ) };
    enum {
        TILE_BLOCK_ROW_ALIGNED_OFFSET_BYTES_ =
            xmma::tma::kTileBlockRowAlignedOffsetBytes( M_, N_, BITS_PER_ELEMENT_ )
    };
    enum {
        TILE_BLOCK_COL_ALIGNED_OFFSET_BYTES_ =
            xmma::tma::kTileBlockColAlignedOffsetBytes( M_, N_, BITS_PER_ELEMENT_ )
    };
    enum { BYTES_PER_UTMALDG = ( TILE_BLOCK_ROW_ * TILE_BLOCK_COL_ * BITS_PER_ELEMENT_ ) / 8 };
    enum { NUM_UTMALDG_OPS = NUM_TILE_BLOCKS_ROW_ * NUM_TILE_BLOCKS_COL_ };
    enum { COPY_BYTES = BYTES_PER_UTMALDG * NUM_UTMALDG_OPS };

#if USE_GMMA

    static constexpr xmma::Gmma_descriptor_mode GMMA_DESC_MODE =
        xmma::tma::kTileBlockRowSwizzle(M_, BITS_PER_ELEMENT_) == SWIZZLE_128B
            ? xmma::Gmma_descriptor_mode::SWIZZLE_128B
            : xmma::Gmma_descriptor_mode::SWIZZLE_64B;

#endif

    using Traits = Traits_;
    //using Cta_tile = Cta_tile_;
    using Cta_tile = typename xmma::lwca::conditional< xmma::lwca::is_same<
                        typename Cta_tile_::Gpu_arch, xmma::Hopper >::value,
                        typename Cta_tile_::Cta_tile,
                        Cta_tile_>::type;

    bool dont_issue_utmaldg;

    template <typename Params>
    inline __device__ Gmem_tile_with_utmaldg( const Params &params,
                                              const lwdaTmaDescv2 *p_desc,
                                              char *extra_smem,
                                              int bidm,
                                              int bidn )
        : p_desc_( p_desc ), bidm_( bidm ), bidn_( bidn ), dont_issue_utmaldg( false ) {
        // Uniform across CTA
        if( threadIdx.x == 0 ) {
            utmapf2<2, TILED>( p_desc_, bidm_, bidn_, 0, 0, 0 );
        }
    }

    /**
     * // Use this code when tma ptx instruction is available
     *
     * unsigned multicast_mask = MULTICAST_MASK;
     *
     * template<typename Smem_tile>
     * inline __device__ void load(Smem_tile &smem_tile, uint64_t) {
     *   #pragma unroll
     *   for(unsigned c = 0; c < NUM_TILE_BLOCKS_COL_; c++) {
     *     for(unsigned r = 0; r < NUM_TILE_BLOCKS_COL_; r++) {
     *       unsigned x0 = bidm_ + r * TILE_BLOCK_ROW_;
     *       unsigned x1 = bidn_ + c * TILE_BLOCK_COL_;
     *       unsigned smem_offset = r * TILE_BLOCK_ROW_ALIGNED_OFFSET_BYTES_ + \
     *          c * TILE_BLOCK_COL_ALIGNED_OFFSET_BYTES_;
     *       smem_tile.store<2, TILED>(desc, smem_offset, 0, x0, x1, 0, 0, 0,
     * multicast_mask);
     *     }
     *   }
     * }
     */
    template <typename Smem_tile> inline __device__ void load( Smem_tile &smem_tile, uint64_t mem_desc, uint16_t mcast_cta_mask ) {
#pragma unroll
        for( unsigned c = 0; c < NUM_TILE_BLOCKS_COL_; c++ ) {
#pragma unroll
            for( unsigned r = 0; r < NUM_TILE_BLOCKS_ROW_; r++ ) {
                unsigned smem_offset = r * TILE_BLOCK_ROW_ALIGNED_OFFSET_BYTES_ +
                                       c * TILE_BLOCK_COL_ALIGNED_OFFSET_BYTES_;
                int bidm__ = bidm_ + r * TILE_BLOCK_ROW_;
                int bidn__ = bidn_ + c * TILE_BLOCK_COL_;
                
                smem_tile.template store<2, TILED, BYTES_PER_UTMALDG, USE_TMA_MULTICAST>(
                    reinterpret_cast<const void *>( p_desc_ ),
                    smem_offset,
                    bidm__,
                    bidn__,
                    0,
                    0,
                    0,
                    0u,
                    0u,
                    0u,
                    mcast_cta_mask);
                    
            }
        }
    }

    template <typename Smem_tile> inline __device__ void commit( Smem_tile &smem_tile_ ) {
    }

    /////////////////////////////////////////////////////////////////
    // Main loop fusion support
    //   d = matmul(f(a, x0, ...), g(b, y0, ...)) + c
    /////////////////////////////////////////////////////////////////
    template < typename Callback_fuse >
    inline __device__ void apply_fuse( Callback_fuse &fuse ) {  }

    template < typename Callback_fuse >
    inline __device__ void load_vectors_m( Callback_fuse &fuse ) {  }

    template < typename Callback_fuse >
    inline __device__ void load_vectors_n( Callback_fuse &fuse ) {  }

    // TODO: Don't know if we want it
    inline __device__ void residue() {
    }

    // TODO: Don't know if we want it
    inline __device__ void disable_loads() {
        // dont_issue_utmaldg = true;
    }

    const lwdaTmaDescv2 *p_desc_;

    int bidm_, bidn_;
};

template <typename Base, typename LDGSTS_CFG = xmma::Ldgsts_config<true>>
using Rebase_gmem_tile_with_utmaldg = Gmem_tile_with_utmaldg<typename Base::Traits,
                                                             typename Base::Cta_tile,
                                                             Base::M,
                                                             Base::N,
                                                             Base::BITS_PER_ELT>;

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S E L E C T O R   T O   P I C K   T H E   C O R R E C T   B A S E   C L A S S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile descriptor.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of each LDG.
    int BYTES_PER_LDG,
    // Do we disable LDGSTS on an architecture that supports it?
    bool DISABLE_LDGSTS>
struct Gmem_ldgsts_selector
    : public Ldgsts_selector<
          Traits,
          Gmem_tile_with_ldgsts<Traits, Cta_tile, M, N, BITS_PER_ELT>,
          Gmem_tile_with_ldg_and_sts<Traits, Cta_tile, M, N, BITS_PER_ELT, BYTES_PER_LDG>,
          DISABLE_LDGSTS> {};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// G M E M   T I L E S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Base = Gmem_tile_with_ldg_and_sts<Traits, Cta_tile, M, N, BITS_PER_ELT, BYTES_PER_LDG>>
struct Gmem_tile_with_ldg_and_sts_a_n : public Base {

    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_with_ldg_and_sts_a_n( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params,
                smem,
                params.m,
                params.k,
                params.lda,
                params.a_gmem,
                bidx.x,
                params.batch.is_batched ? 0 : bidx.z,
                tidx,
                params.batch.is_batched ? 0 : bidx.z ) {
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_a(
                static_cast<int64_t>( bidx.z ) * static_cast<int64_t>( params.a_stride_batches ) );
        }
        precompute_residue_predicates();
    }

    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates() {
        this->precompute_residue_predicates_a_n_b_t();
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_n_b_t();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Base = Gmem_tile_with_ldg_and_sts<Traits, Cta_tile, M, N, BITS_PER_ELT, BYTES_PER_LDG>>
struct Gmem_tile_with_ldg_and_sts_a_t : public Base {

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_with_ldg_and_sts_a_t( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params,
                smem,
                params.k,
                params.m,
                params.lda,
                params.a_gmem,
                params.batch.is_batched ? 0 : bidx.z,
                bidx.x,
                tidx,
                params.batch.is_batched ? 0 : bidx.z ) {
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_a(
                static_cast<int64_t>( bidx.z ) * static_cast<int64_t>( params.a_stride_batches ) );
        }
        precompute_residue_predicates();
    }

    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates() {
        this->precompute_residue_predicates_a_t_b_n();
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_t_b_n();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Base = Gmem_tile_with_ldg_and_sts<Traits, Cta_tile, M, N, BITS_PER_ELT, BYTES_PER_LDG>>
struct Gmem_tile_with_ldg_and_sts_b_n : public Base {

    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_with_ldg_and_sts_b_n( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params,
                smem,
                params.k,
                params.n,
                params.ldb,
                params.b_gmem,
                params.batch.is_batched ? 0 : bidx.z,
                bidx.y,
                tidx,
                params.batch.is_batched ? 0 : bidx.z ) {
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_b(
                static_cast<int64_t>( bidx.z ) * static_cast<int64_t>( params.b_stride_batches ) );
        }
        precompute_residue_predicates();
    }

    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates() {
        this->precompute_residue_predicates_a_t_b_n();
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_t_b_n();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    // bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Base = Gmem_tile_with_ldg_and_sts<Traits, Cta_tile, M, N, BITS_PER_ELT, BYTES_PER_LDG>>
struct Gmem_tile_with_ldg_and_sts_b_t : public Base {

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_with_ldg_and_sts_b_t( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params,
                smem,
                params.n,
                params.k,
                params.ldb,
                params.b_gmem,
                bidx.y,
                params.batch.is_batched ? 0 : bidx.z,
                tidx,
                params.batch.is_batched ? 0 : bidx.z ) {
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_b(
                static_cast<int64_t>( bidx.z ) * static_cast<int64_t>( params.b_stride_batches ) );
        }
        precompute_residue_predicates();
    }

    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates() {
        this->precompute_residue_predicates_a_n_b_t();
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_n_b_t();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// LDGSTS

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Base = Gmem_tile_with_ldgsts<Traits, Cta_tile, M, N, BITS_PER_ELT>>
struct Gmem_tile_with_ldgsts_a_n : public Base {

    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_with_ldgsts_a_n( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params,
                smem,
                params.m,
                params.k,
                params.lda,
                params.a_gmem,
                bidx.x,
                params.batch.is_batched ? 0 : bidx.z,
                tidx,
                params.batch.is_batched ? 0 : bidx.z ) {
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_a(
                static_cast<int64_t>( bidx.z ) * static_cast<int64_t>( params.a_stride_batches ) );
        }
        precompute_residue_predicates();
    }

    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates() {
        this->precompute_residue_predicates_a_n_b_t();
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_n_b_t();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Base = Gmem_tile_with_ldgsts<Traits, Cta_tile, M, N, BITS_PER_ELT>>
struct Gmem_tile_with_ldgsts_a_t : public Base {

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_with_ldgsts_a_t( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params,
                smem,
                params.k,
                params.m,
                params.lda,
                params.a_gmem,
                params.batch.is_batched ? 0 : bidx.z,
                bidx.x,
                tidx,
                params.batch.is_batched ? 0 : bidx.z ) {
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_a(
                static_cast<int64_t>( bidx.z ) * static_cast<int64_t>( params.a_stride_batches ) );
        }
        precompute_residue_predicates();
    }

    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates() {
        this->precompute_residue_predicates_a_t_b_n();
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_t_b_n();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Base = Gmem_tile_with_ldgsts<Traits, Cta_tile, M, N, BITS_PER_ELT>>
struct Gmem_tile_with_ldgsts_b_n : public Base {

    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_with_ldgsts_b_n( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params,
                smem,
                params.k,
                params.n,
                params.ldb,
                params.b_gmem,
                params.batch.is_batched ? 0 : bidx.z,
                bidx.y,
                tidx,
                params.batch.is_batched ? 0 : bidx.z ) {
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_b(
                static_cast<int64_t>( bidx.z ) * static_cast<int64_t>( params.b_stride_batches ) );
        }
        precompute_residue_predicates();
    }

    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates() {
        this->precompute_residue_predicates_a_t_b_n();
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_t_b_n();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Base = Gmem_tile_with_ldgsts<Traits, Cta_tile, M, N, BITS_PER_ELT>>
struct Gmem_tile_with_ldgsts_b_t : public Base {

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_with_ldgsts_b_t( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params,
                smem,
                params.n,
                params.k,
                params.ldb,
                params.b_gmem,
                bidx.y,
                params.batch.is_batched ? 0 : bidx.z,
                tidx,
                params.batch.is_batched ? 0 : bidx.z ) {
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_b(
                static_cast<int64_t>( bidx.z ) * static_cast<int64_t>( params.b_stride_batches ) );
        }
        precompute_residue_predicates();
    }

    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates() {
        this->precompute_residue_predicates_a_n_b_t();
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_n_b_t();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// UTMALDG

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // Base class
    typename Base = Gmem_tile_with_utmaldg<Traits, Cta_tile, M, N, BITS_PER_ELT>>
struct Gmem_tile_with_utmaldg_a_n : public Base {

    using Tile_traits = Cta_tile;

    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor
    template <typename Params>
    inline __device__
    Gmem_tile_with_utmaldg_a_n( const Params &params, void *extra_smem, const dim3 &bidx, int )
        : Base( params,
                reinterpret_cast<const lwdaTmaDescv2 *>( params.a_desc ),
                reinterpret_cast<char *>( extra_smem ),
                bidx.x * M,
                bidx.z * N ) {

        if( Tile_traits::USE_TMA_MULTICAST ) {
            tma_multicast_mask = 1;
            #pragma unroll
            for( int i = 0 ; i < Tile_traits::CLUSTER_COL; i++ ) {
                tma_multicast_mask |= 1 << i;
            }

            // Get CGA Row ID
            // Eg. For a 4x4 CGA, it is in the range(0-3)
            int cga_row_id = bidx.y % Tile_traits::CLUSTER_ROW;
            tma_multicast_mask <<= cga_row_id;
        }
    }

    // Move to next kBlock
    inline __device__ void move( int, uint64_t ) {
        // Uniform across CTA
        this->bidn_ += N;
        xmma::utmapf2<2, TILED>( this->p_desc_, this->bidm_, this->bidn_, 0, 0, 0 );
    }

    template <typename Smem_tile> inline __device__ 
    void load( Smem_tile &smem_tile, uint64_t mem_desc)
    {
        Base::load(smem_tile, mem_desc, tma_multicast_mask);
    }

    uint16_t tma_multicast_mask;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // Base class
    typename Base = Gmem_tile_with_utmaldg<Traits, Cta_tile, M, N, BITS_PER_ELT>>
struct Gmem_tile_with_utmaldg_a_t : public Base {

    using Tile_traits = Cta_tile;

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor
    template <typename Params>
    inline __device__
    Gmem_tile_with_utmaldg_a_t( const Params &params, void *extra_smem, const dim3 &bidx, int )
        : Base( params,
                reinterpret_cast<const lwdaTmaDescv2 *>( params.a_desc ),
                reinterpret_cast<char *>( extra_smem ),
                bidx.z * M,
                bidx.x * N ) {

        if( Tile_traits::USE_TMA_MULTICAST ) {
            tma_multicast_mask = 1;
            #pragma unroll
            for( int i = 0 ; i < Tile_traits::CLUSTER_COL; i++ ) {
                tma_multicast_mask |= 1 << i;
            }

            // Get CGA Row ID
            // Eg. For a 4x4 CGA, it is in the range(0-3)
            int cga_row_id = bidx.y % Tile_traits::CLUSTER_ROW;
            tma_multicast_mask <<= cga_row_id;
        }
    }

    // Move to next kBlock
    inline __device__ void move( int, uint64_t ) {
        // Uniform across CTA
            this->bidm_ += M;
            xmma::utmapf2<2, TILED>( this->p_desc_, this->bidm_, this->bidn_, 0, 0, 0 );
    }

    template <typename Smem_tile> inline __device__ 
    void load( Smem_tile &smem_tile, uint64_t mem_desc)
    {
        Base::load(smem_tile, mem_desc, tma_multicast_mask);
    }

    uint16_t tma_multicast_mask;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // Base class
    typename Base = Gmem_tile_with_utmaldg<Traits, Cta_tile, M, N, BITS_PER_ELT>>
struct Gmem_tile_with_utmaldg_b_n : public Base {

    using Tile_traits = Cta_tile;
    
    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor
    template <typename Params>
    inline __device__
    Gmem_tile_with_utmaldg_b_n( const Params &params, void *extra_smem, const dim3 &bidx, int )
        : Base( params,
                reinterpret_cast<const lwdaTmaDescv2 *>( params.b_desc ),
                reinterpret_cast<char *>( extra_smem ),
                bidx.z * M,
                bidx.y * N ) {

        if( Tile_traits::USE_TMA_MULTICAST ) {
            tma_multicast_mask = 1;
            #pragma unroll
            for( int i = 0 ; i < Tile_traits::CLUSTER_ROW; i++ ) {
                tma_multicast_mask |= 1 << Tile_traits::CLUSTER_COL * i;
            }

            // Get CGA Col ID - B matrix is always shared among columns of CTAs in a CGA
            // Eg. For a 4x4 CGA, it is in the range(0-3)
            int cga_col_id = bidx.x % Tile_traits::CLUSTER_COL;
            tma_multicast_mask <<= cga_col_id;
        }
    }

    // Move to next kBlock
    inline __device__ void move( int, uint64_t ) {
        // Uniform across CTA
            this->bidm_ += M;
            xmma::utmapf2<2, TILED>( this->p_desc_, this->bidm_, this->bidn_, 0, 0, 0 );
    }

    template <typename Smem_tile> inline __device__ 
    void load( Smem_tile &smem_tile, uint64_t mem_desc)
    {
        Base::load(smem_tile, mem_desc, tma_multicast_mask);
    }

    uint16_t tma_multicast_mask;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // Base class
    typename Base = Gmem_tile_with_utmaldg<Traits, Cta_tile, M, N, BITS_PER_ELT>>
struct Gmem_tile_with_utmaldg_b_t : public Base {

    using Tile_traits = Cta_tile;

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor
    template <typename Params>
    inline __device__
    Gmem_tile_with_utmaldg_b_t( const Params &params, void *extra_smem, const dim3 &bidx, int )
        : Base( params,
                reinterpret_cast<const lwdaTmaDescv2 *>( params.b_desc ),
                reinterpret_cast<char *>( extra_smem ),
                bidx.y * M,
                bidx.z * N ) {

        if( Tile_traits::USE_TMA_MULTICAST ) {
            tma_multicast_mask = 1;
            #pragma unroll
            for( int i = 0 ; i < Tile_traits::CLUSTER_ROW; i++ ) {
                tma_multicast_mask |= 1 << Tile_traits::CLUSTER_COL * i;
            }

            // Get CGA Col ID - B matrix is always shared among columns of CTAs in a CGA
            // Eg. For a 4x4 CGA, it is in the range(0-3)
            int cga_col_id = bidx.x % Tile_traits::CLUSTER_COL;
            tma_multicast_mask <<= cga_col_id;
        }
    }

    // Move to next kBlock
    inline __device__ void move( int, uint64_t ) {
        // Uniform across CTA
            this->bidn_ += N;
            xmma::utmapf2<2, TILED>( this->p_desc_, this->bidm_, this->bidn_, 0, 0, 0 );
    }

    template <typename Smem_tile> inline __device__ 
    void load( Smem_tile &smem_tile, uint64_t mem_desc)
    {
        Base::load(smem_tile, mem_desc, tma_multicast_mask);
    }

    uint16_t tma_multicast_mask;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile descriptor.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of each LDG.
    int BYTES_PER_LDG,
    // Select copy engine
    Copy_engine COPY_ENGINE>
struct Gmem_copy_engine_selector_a_n
    : public Copy_engine_selector<
          Traits,
          Gmem_tile_with_ldg_and_sts_a_n<Traits, Cta_tile, M, N, BITS_PER_ELT, BYTES_PER_LDG>,
          Gmem_tile_with_ldgsts_a_n<Traits, Cta_tile, M, N, BITS_PER_ELT>,
          Gmem_tile_with_utmaldg_a_n<Traits, Cta_tile, M, N, BITS_PER_ELT>,
          COPY_ENGINE> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile descriptor.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of each LDG.
    int BYTES_PER_LDG,
    // Select copy engine
    Copy_engine COPY_ENGINE>
struct Gmem_copy_engine_selector_a_t
    : public Copy_engine_selector<
          Traits,
          Gmem_tile_with_ldg_and_sts_a_t<Traits, Cta_tile, M, N, BITS_PER_ELT, BYTES_PER_LDG>,
          Gmem_tile_with_ldgsts_a_t<Traits, Cta_tile, M, N, BITS_PER_ELT>,
          Gmem_tile_with_utmaldg_a_t<Traits, Cta_tile, M, N, BITS_PER_ELT>,
          COPY_ENGINE> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile descriptor.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of each LDG.
    int BYTES_PER_LDG,
    // Select copy engine
    Copy_engine COPY_ENGINE>
struct Gmem_copy_engine_selector_b_n
    : public Copy_engine_selector<
          Traits,
          Gmem_tile_with_ldg_and_sts_b_n<Traits, Cta_tile, M, N, BITS_PER_ELT, BYTES_PER_LDG>,
          Gmem_tile_with_ldgsts_b_n<Traits, Cta_tile, M, N, BITS_PER_ELT>,
          Gmem_tile_with_utmaldg_b_n<Traits, Cta_tile, M, N, BITS_PER_ELT>,
          COPY_ENGINE> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile descriptor.
    typename Cta_tile,
    // The M dimension of the GEMM tile.
    int M,
    // The N dimension of the GEMM tile.
    int N,
    // The number of bits per element.
    int BITS_PER_ELT,
    // The size in bytes of each LDG.
    int BYTES_PER_LDG,
    // Select copy engine
    Copy_engine COPY_ENGINE>
struct Gmem_copy_engine_selector_b_t
    : public Copy_engine_selector<
          Traits,
          Gmem_tile_with_ldg_and_sts_b_t<Traits, Cta_tile, M, N, BITS_PER_ELT, BYTES_PER_LDG>,
          Gmem_tile_with_ldgsts_b_t<Traits, Cta_tile, M, N, BITS_PER_ELT>,
          Gmem_tile_with_utmaldg_b_t<Traits, Cta_tile, M, N, BITS_PER_ELT>,
          COPY_ENGINE> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // Select copy engine, default is LDGSTS
    Copy_engine COPY_ENGINE = Copy_engine::CE_LDGSTS,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Base = typename Gmem_copy_engine_selector_a_n<Traits,
                                                           Cta_tile,
                                                           Cta_tile::M,
                                                           Cta_tile::K,
                                                           Traits::BITS_PER_ELEMENT_A,
                                                           BYTES_PER_LDG,
                                                           COPY_ENGINE>::Class>
struct Gmem_tile_a_n : public Base {

    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a_n( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params, smem, bidx, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // Select copy engine, default is LDGSTS
    Copy_engine COPY_ENGINE = Copy_engine::CE_LDGSTS,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Base = typename Gmem_copy_engine_selector_a_t<Traits,
                                                           Cta_tile,
                                                           Cta_tile::K,
                                                           Cta_tile::M,
                                                           Traits::BITS_PER_ELEMENT_A,
                                                           BYTES_PER_LDG,
                                                           COPY_ENGINE>::Class>
struct Gmem_tile_a_t : public Base {

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a_t( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params, smem, bidx, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // Select copy engine, default is LDGSTS
    Copy_engine COPY_ENGINE = Copy_engine::CE_LDGSTS,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Base = typename Gmem_copy_engine_selector_b_n<Traits,
                                                           Cta_tile,
                                                           Cta_tile::K,
                                                           Cta_tile::N,
                                                           Traits::BITS_PER_ELEMENT_B,
                                                           BYTES_PER_LDG,
                                                           COPY_ENGINE>::Class>
struct Gmem_tile_b_n : public Base {

    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b_n( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params, smem, bidx, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // Select copy engine, default is LDGSTS
    Copy_engine COPY_ENGINE = Copy_engine::CE_LDGSTS,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Base = typename Gmem_copy_engine_selector_b_t<Traits,
                                                           Cta_tile,
                                                           Cta_tile::N,
                                                           Cta_tile::K,
                                                           Traits::BITS_PER_ELEMENT_B,
                                                           BYTES_PER_LDG,
                                                           COPY_ENGINE>::Class>
struct Gmem_tile_b_t : public Base {

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b_t( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params, smem, bidx, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>>
struct Gmem_tile_epilogue
    : public xmma::helpers::Gmem_tile_epilogue<Traits, Cta_tile, xmma::Row, Fragment_c> {

    using Layout = xmma::Row;
    // The base class.
    using Base = xmma::helpers::Gmem_tile_epilogue<Traits, Cta_tile, xmma::Row, Fragment_c>;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_epilogue( const Params &params, int bidm, int bidn, int bidz, int tidx )
        : Base( params.m,
                params.n,
                params.ldd,
                reinterpret_cast<char *>( params.d_gmem ),
                reinterpret_cast<const char *>( params.c_gmem ),
                bidm,
                bidn,
                bidz,
                tidx ) {
        if( params.batch.is_batched ) {
            const int64_t batch_offset =
                static_cast<int64_t>( bidz ) * static_cast<int64_t>( params.c_stride_batches );
            this->out_ptr_ += Traits::offset_in_bytes_c( batch_offset );
            this->res_ptr_ += Traits::offset_in_bytes_c( batch_offset );
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>>
struct Gmem_tile_wo_smem_epilogue
    : public xmma::helpers::Gmem_tile_wo_smem_epilogue<Traits, Cta_tile, xmma::Row, Fragment_c> {

    using Layout = xmma::Row;
    // The base class.
    using Base = xmma::helpers::Gmem_tile_wo_smem_epilogue<Traits, Cta_tile, xmma::Row, Fragment_c>;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_wo_smem_epilogue( const Params &params, int bidm, int bidn, int bidz, int tidx )
        : Base( params.m,
                params.n,
                params.ldd,
                reinterpret_cast<char *>( params.d_gmem ),
                reinterpret_cast<const char *>( params.c_gmem ),
                bidm,
                bidn,
                bidz,
                tidx ) {
        if( params.batch.is_batched ) {
            const int64_t batch_offset =
                static_cast<int64_t>( bidz ) * static_cast<int64_t>( params.c_stride_batches );
            this->out_ptr_ += Traits::offset_in_bytes_c( batch_offset );
            this->res_ptr_ += Traits::offset_in_bytes_c( batch_offset );
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace xmma
