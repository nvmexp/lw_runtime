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

#include <xmma/implicit_gemm/utils.h>
#include <xmma/gemm/gmem_tile.h>
#include <xmma/warp_masks.h>

namespace xmma {
namespace implicit_gemm {
namespace fprop_wo_epi_swizzle {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   B
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Input_related, int BYTES_PER_LDG>
struct Gmem_tile_base_b : public gemm::Gmem_tile_base<Traits, Cta_tile, Cta_tile::K, Cta_tile::N,
                                                      Traits::BITS_PER_ELEMENT_B, BYTES_PER_LDG> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The base class.
    using Base_ = gemm::Gmem_tile_base<Traits, Cta_tile, Cta_tile::K, Cta_tile::N,
                                       Traits::BITS_PER_ELEMENT_B, BYTES_PER_LDG>;

    // The number of threads needed to load a pixel.
    enum { THREADS_PER_FILTER = Base_::THREADS_PER_COLUMN };
    // The number of images loaded per LDG.
    enum { FILTERS_PER_LDG = Base_::COLUMNS_PER_LDG };

    enum { FLT_T = Input_related::FLT_T };
    enum { FLT_R = Input_related::FLT_R };
    enum { FLT_S = Input_related::FLT_S };
    enum { STATIC_FILTER_SIZE = Input_related::STATIC_FILTER_SIZE };

    enum { CONTIGUOUS_BYTES = ( Cta_tile::N >= 128 ) ? 16 : 8 };
    //enum { GRP_ELEMENTS = BYTES_PER_LDG * 8 / Traits::BITS_PER_ELEMENT_B };
    enum { GRP_ELEMENTS = CONTIGUOUS_BYTES * 8 / Traits::BITS_PER_ELEMENT_B };
    enum { GRP_ACC = 2 };
    enum { ACC_STRIDE = 8 };
    enum { PERMUTE_OFFSET = ( GRP_ELEMENTS / GRP_ACC) * 8 };    // 8 -- 16x8x32, the 8
    enum { PERMUTE_REGION = Xmma_tile::N_PER_XMMA_PER_CTA };    // 32
    enum { PERMUTE_SUB_REGION = Xmma_tile::N_PER_XMMA };        // 16

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_b( const Params& params,
                                        void* smem,
                                        const dim3& bidx,
                                        int tidx )
        : Base_( params, smem, params.c * params.g, params.flt_gmem, bidx.z )
        , params_delta_k_( FILTERS_PER_LDG * params.trsc )
        , params_split_k_c_( params.split_k_c ) {

        // The coordinate in the C dimension.
        int c = bidx.z * params_split_k_c_ + tidx % THREADS_PER_FILTER * Base_::ELTS_PER_LDG;
        
        // Restrict C to the index in the group if we have more than 1 group.
        int c_in_group = c;
        if( Cta_tile::GROUPS > 1 ) {
            c_in_group = c & ( Cta_tile::K / Cta_tile::GROUPS - 1 );
        }

///////////////////////////////////////////////////////////////////////////////////////////////////

        // Assemble the offset and compute the predicates.
        uint32_t preds[Base_::LDGS];
        #pragma unroll
        for( int ii = 0; ii < Base_::LDGS; ++ii ) {
            int k_base = ii * FILTERS_PER_LDG + tidx / THREADS_PER_FILTER;
            int k_permute_region = k_base / PERMUTE_REGION;
            int k_permute_sub_region = (k_base / PERMUTE_SUB_REGION) % 2;
            int k = ((k_permute_region * PERMUTE_SUB_REGION + k_base % PERMUTE_SUB_REGION) / 2) * 2 / ACC_STRIDE * GRP_ACC +
                             k_permute_sub_region * PERMUTE_OFFSET +
                             ( (k_base % ACC_STRIDE) / 2) * GRP_ELEMENTS + 
                             (k_base % 2);

            this->offsets_[ii] = (bidx.y * Cta_tile::N + k) * params.trsc + c_in_group;

            // The image tile does a cross-correlation, and we modify the filter to implement the colw.
            if( !params.cross_correlation ) {
                int filter_trs_per_cta;
                if( STATIC_FILTER_SIZE ) {
                    filter_trs_per_cta = FLT_T * FLT_R * FLT_S;
                } else {
                    filter_trs_per_cta = params.filter_trs_per_cta;
                }
                this->offsets_[ii] += (filter_trs_per_cta - 1) * params.c;
            }
            preds[ii] = (bidx.y * Cta_tile::N + k) < params.k * params.g;
        }

///////////////////////////////////////////////////////////////////////////////////////////////////

        // Finalize the predicates.
        if ( Cta_tile::GROUPS == 1 ) {
            asm volatile( "set.lt.u32.u32 %0, %1, %2;"
                          : "=r"( this->preds_[0] )
                          : "r"( c ), "r"( params.c ) );
        } else {
            asm volatile( "set.lt.u32.u32 %0, %1, %2;"
                          : "=r"( this->preds_[0] )
                          : "r"( c ), "r"( Cta_tile::K / Xmma_tile::XMMAS_GROUPS ) );
        }
        this->preds_[0] &= xmma::pack_predicates( preds );

        precompute_residue_predicates( this->params_split_k_c_ );
    }

    // Compute the global memory pointers.
    inline __device__ void compute_load_pointers( const void* ( &ptrs )[Base_::LDGS] ) const {
        #pragma unroll
        for( int ii = 0; ii < Base_::LDGS; ++ii ) {
            ptrs[ii] =
                &this->ptr_[Traits::offset_in_bytes_b( this->offsets_[ii] )];
                //&this->ptr_[Traits::offset_in_bytes_b( this->offsets_[0] + ii * params_delta_k_ )];
        }
    }

    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates( int params_split_k_c_ ) {
        this->precompute_residue_predicates_a_t_b_n( params_split_k_c_ );
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_t_b_n( params_split_k_c_ );
    }

    // The constant C dimension and the delta in the k dimension.
    const int params_delta_k_, params_split_k_c_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA descriptor.
    typename Cta_tile,
    // Template parameters which are related to input parameters like filter size.
    typename Input_related,
    // The size in bytes of the LDG.
    int BYTES_PER_LDG,
    // The base class.
    typename Base = Gmem_tile_base_b<Traits, Cta_tile, Input_related, BYTES_PER_LDG>>
struct Gmem_tile_with_ldg_and_sts_b : public Base {

    // It does not use LDGSTS.
    enum { USE_LDGSTS = 0 };
    enum { USE_UTMALDG = 0 };

    // DEBUG: The group implementation on Volta and Turing assumes LDG.128 
    // and K == 64, WARPS_N == 2 or 4 and WARPS_K == 1.
    static_assert(Cta_tile::GROUPS == 1 || (BYTES_PER_LDG == 16 && Cta_tile::K == 64 \
          && (Cta_tile::WARPS_N == 2 || (Cta_tile::WARPS_N == 4 && Cta_tile::GROUPS > 2)) \
          && Cta_tile::WARPS_K == 1), "");
    // END OF DEBUG.

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_with_ldg_and_sts_b( const Params& params,
                                                    void* smem,
                                                    const dim3& bidx,
                                                    int tidx )
        : Base( params, smem, bidx, tidx ) {

        // C=K=4 or 8. Each character is a 8x8 block.
        //
        // x x -> x 0
        // x x -> 0 x
        //
        if( Cta_tile::GROUPS == 8  || Cta_tile::GROUPS == 16 ) {
            if ( (threadIdx.x % Base::THREADS_PER_FILTER & 1) ^
                    (threadIdx.x / Base::THREADS_PER_FILTER / 8 & 1) ) {
                this->disable_loads();
                #pragma unroll
                for ( int ii = 0; ii < Base::LDGS; ++ii) {
                    fetch_[ii] = make_uint4(0, 0, 0, 0);
                }
            }
        }
    }

    // Load a tile from global memory.
    template <typename Smem_tile>
    inline __device__ void load(Smem_tile&, uint64_t mem_desc = MEM_DESC_DEFAULT) {
        // Prepare the pointers.
        const void* ptrs[Base::LDGS];
        this->compute_load_pointers( ptrs );

        // Issue the loads.
        if( Cta_tile::GROUPS == 16 ) {
            xmma::ldg_force_64( this->fetch_, ptrs, this->preds_[0] );
        } else {
            xmma::ldg( this->fetch_, ptrs, this->preds_ );
        }
    }

    // Store the pixels to shared memory.
    template <typename Xmma_smem_tile> inline __device__ void commit( Xmma_smem_tile& smem ) {
        // Remove redundant elements for group colwolution.
        if( Cta_tile::GROUPS > 1 ) {
            remove_redundant_group_elements();
        }

        // Store to shared memory.
        if( Cta_tile::N % Base::FILTERS_PER_LDG != 0 ) {  // Incomplete stores.
            smem.store( this->fetch_, this->preds_ );
        } else {
            smem.store( this->fetch_ );
        }
    }

    // Set redundant data to zero.
    inline __device__ void remove_redundant_group_elements() {

        // C=K=4. Each character is a 4x4 block.
        //
        // a b 0 0 -> a 0 0 0
        // x x 0 0 -> 0 b 0 0
        // 0 0 c d -> 0 0 c 0
        // 0 0 x x -> 0 0 0 d
        //
        if( Cta_tile::GROUPS == 16 ) {
            if ( (threadIdx.x / Cta_tile::THREADS_PER_WARP & 1) == 0 ) {
                #pragma unroll
                for ( int ii = 0; ii < Base::LDGS; ++ii ) {
                    fetch_[ii].z = 0;
                    fetch_[ii].w = 0;
                }
            }
            else {
                #pragma unroll
                for ( int ii = 0; ii < Base::LDGS; ++ii ) {
                    fetch_[ii].z = fetch_[ii].x;
                    fetch_[ii].w = fetch_[ii].y;
                    fetch_[ii].x = 0;
                    fetch_[ii].y = 0;
                }
            }
        }
    }

    // The fetch registers.
    typename Uint_from_size_in_bytes<Base::BYTES_PER_LDG>::Type fetch_[Base::LDGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Base>
using Rebase_gmem_tile_with_ldg_and_sts_b =
    Gmem_tile_with_ldg_and_sts_b<typename Base::Traits, typename Base::Cta_tile,
                                 typename Base::Input_related, Base::BYTES_PER_LDG, Base>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA descriptor.
    typename Cta_tile,
    // Template parameters which are related to input parameters like filter size.
    typename Input_related,
    // The size in bytes of the LDG.
    int BYTES_PER_LDG,
    // The base class.
    typename Base = Gmem_tile_base_b<Traits, Cta_tile, Input_related, BYTES_PER_LDG>
>
struct Gmem_tile_with_ldgsts_b : public Base {

    // It uses LDGSTS.
    enum { USE_LDGSTS = 1 };
    enum { USE_UTMALDG = 0 };

    // DEBUG: The group implementation on Volta and Ampere assumes LDG.128 
    // and K == 64, WARPS_N == 2 or 4 and WARPS_K == 1.
    static_assert(Cta_tile::GROUPS == 1 || (BYTES_PER_LDG == 16 && Cta_tile::K == 64 \
          && (Cta_tile::WARPS_N == 2 || (Cta_tile::WARPS_N == 4 && Cta_tile::GROUPS > 2)) \
          && Cta_tile::WARPS_K == 1), "");
    // END OF DEBUG.

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_with_ldgsts_b( const Params &params,
                                               void *smem,
                                               const dim3 &bidx,
                                               int tidx )
        : Base(params, smem, bidx, tidx) {

        // For group colw.
        uint32_t masks[Base::LDGS];
        for( int ii = 0; ii < Base::LDGS; ++ii ) {
            masks[ii] = uint32_t( -1 );
        }
        // C=K=8 or C=K=4 - Each character is a 8x8 block.
        //
        // a 0 c 0 e 0 g 0
        // 0 b 0 d 0 f 0 h
        // 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0
        //
        if ( Traits::BITS_PER_ELEMENT_B == 16 ) {
            if ( Cta_tile::GROUPS == 8 || Cta_tile::GROUPS == 16 ) {
                int row_idx = threadIdx.x % Base::THREADS_PER_FILTER;
                int col_idx = threadIdx.x / Base::THREADS_PER_FILTER / 8;
                if ( (row_idx & 1) ^ (col_idx & 1) ) {
                    // Disable loads.
                    #pragma unroll
                    for( int ii = 0; ii < Base::LDGS; ++ii ) {
                        masks[ii] = 0u;
                    }
                }
            }
        } else if ( Traits::BITS_PER_ELEMENT_B == 32 ) {
            int row_idx = threadIdx.x % Base::THREADS_PER_FILTER;
            if ( Cta_tile::GROUPS == 8 ) {
                #pragma unroll
                for( int ii = 0; ii < Base::LDGS; ++ii ) {
                    masks[ii] = 0u;
                    if ( row_idx < 2 ) {
                        masks[ii] = ii % 2 == 0;
                    } 
                    else if (row_idx < 4) {
                        masks[ii] = ii % 2 == 1;
                    }
                }
            } else if ( Cta_tile::GROUPS == 16 ) {
                // C=K=4 - Each character is a 4x4 block.
                // We use the format directly.
                // a 0 0 0
                // 0 b 0 0
                // 0 0 c 0
                // 0 0 0 d
                //
                int col_idx = threadIdx.x / Base::THREADS_PER_FILTER / 4;
                #pragma unroll
                for( int ii = 0; ii < Base::LDGS; ++ii ) {
                    masks[ii] = 0u;
                    if ((row_idx == 0 && col_idx == 0) || (row_idx == 1  && col_idx == 1)) {
                        masks[ii] = ii % 2 == 0;
                    } 
                    else if ((row_idx == 2 && col_idx == 0) || (row_idx == 3 && col_idx == 1)) {
                        masks[ii] = ii % 2 == 1;
                    }
                }
            }
        }

        // Packing masks.
        xmma::pack_predicates(preds_masks_, masks);
    }

    // Load a tile from global memory.
    template< typename Smem_tile >
    inline __device__ void load(Smem_tile &smem_tile, uint64_t mem_desc = MEM_DESC_DEFAULT) {
        // Prepare the pointers.
        const void *ptrs[Base::LDGS];
        this->compute_load_pointers(ptrs);

        if ( Cta_tile::GROUPS == 16 &&
             Traits::BITS_PER_ELEMENT_B == 16 ) {
            // C=K=4 - Each character is a 4x4 block.
            //
            // a b 0 0     a 0 0 0
            // 0 0 0 0 --> 0 b 0 0
            // 0 0 c d     0 0 c 0
            // 0 0 0 0     0 0 0 d
            //
            uint32_t smem_str_ptrs[Base::LDGS];
            for( int ii = 0; ii < Base::PRED_REGS; ++ii ) {
                this->preds_[ii] &= preds_masks_[ii];
            }
            smem_tile.compute_store_pointers(smem_str_ptrs);
            int is_odd_group = threadIdx.x / Cta_tile::THREADS_PER_WARP % 2 == 1;
            #pragma unroll
            for( int ii = 0; ii < Base::LDGS; ++ii ) {
                smem_str_ptrs[ii] = smem_str_ptrs[ii] + (is_odd_group ? 8 : 0);
            }
            // Use @Px LDGSTS.E.64 to save MIO traffic
            ldgsts<Base::LDGS, Base::PRED_REGS, 8, false>(smem_str_ptrs, ptrs, this->preds_, mem_desc);
        } else {
            if ( Cta_tile::GROUPS >= 8 ) {
                for( int ii = 0; ii < Base::PRED_REGS; ++ii ) {
                    this->preds_[ii] &= preds_masks_[ii];
                }
            }
            smem_tile.store(ptrs, this->preds_, mem_desc);
        }
    }
    
    // Load a tile from global memory.
    template< typename Smem_tile, int PHASE >
    inline __device__ void load_per_phase(Smem_tile &smem_tile, uint64_t mem_desc = MEM_DESC_DEFAULT) {
        // Prepare the pointers.
        const void *ptrs[Base::LDGS_PER_PHASE];
        this->compute_load_pointers_per_phase<PHASE>(ptrs);

        if ( Cta_tile::GROUPS == 16 &&
             Traits::BITS_PER_ELEMENT_B == 16 ) {
            // C=K=4 - Each character is a 4x4 block.
            //
            // a b 0 0     a 0 0 0
            // 0 0 0 0 --> 0 b 0 0
            // 0 0 c d     0 0 c 0
            // 0 0 0 0     0 0 0 d
            //
            uint32_t smem_str_ptrs[Base::LDGS_PER_PHASE];
            for( int ii = 0; ii < Base::PRED_REGS; ++ii ) {
                this->preds_[ii] &= preds_masks_[ii];
            }
            smem_tile.compute_store_pointers_per_phase<Base::LDGS_PER_PHASE, PHASE, Base::LDGS_UNROLL>(smem_str_ptrs);
            int is_odd_group = threadIdx.x / Cta_tile::THREADS_PER_WARP % 2 == 1;
            #pragma unroll
            for( int ii = 0; ii < Base::LDGS_PER_PHASE; ++ii ) {
                smem_str_ptrs[ii] = smem_str_ptrs[ii] + (is_odd_group ? 8 : 0);
            }
            // Use @Px LDGSTS.E.64 to save MIO traffic
            ldgsts_per_phase<Base::LDGS_PER_PHASE, Base::PRED_REGS, PHASE, 8, false>(smem_str_ptrs, ptrs, this->preds_, mem_desc);
        } else {
            if( (PHASE + 1) * Base::LDGS_PER_PHASE <= Base::LDGS ){ 
                if ( Cta_tile::GROUPS >= 8 ) {
                    for( int ii = 0; ii < Base::PRED_REGS; ++ii ) {
                        this->preds_[ii] &= preds_masks_[ii];
                    }
                }
            smem_tile.template store_per_phase<Base::LDGS_PER_PHASE, Base::PRED_REGS, PHASE, Base::LDGS_UNROLL, xmma::Ldgsts_config<true>>(ptrs, this->preds_, mem_desc);
            }
        }
    }




    // It does nothing.
    template< typename Smem_tile >
    inline __device__ void commit(Smem_tile &) {
    }
    uint32_t preds_masks_[Base::PRED_REGS];

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Base >
using Rebase_gmem_tile_with_ldgsts_b = Gmem_tile_with_ldgsts_b<typename Base::Traits,
                                                               typename Base::Cta_tile,
                                                               typename Base::Input_related,
                                                               Base::BYTES_PER_LDG,
                                                               Base>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits, 
    // The CTA tile description.
    typename Cta_tile,
    // Template parameters which are related to input parameters like filter size.
    typename Input_related,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_b<Traits, Cta_tile, Input_related, BYTES_PER_LDG>
>
struct Gmem_tile_b 
    : public Ldgsts_selector<Traits,
                             Gmem_tile_with_ldgsts_b<Traits, 
                                                     Cta_tile, 
                                                     Input_related, 
                                                     BYTES_PER_LDG>,
                             Gmem_tile_with_ldg_and_sts_b<Traits, 
                                                          Cta_tile, 
                                                          Input_related, 
                                                          BYTES_PER_LDG>,
                             DISABLE_LDGSTS>::Class {

    // The base class.
    using Base_ = 
        typename Ldgsts_selector<Traits,
                                 Gmem_tile_with_ldgsts_b<Traits, 
                                                         Cta_tile, 
                                                         Input_related, 
                                                         BYTES_PER_LDG>,
                                 Gmem_tile_with_ldg_and_sts_b<Traits, 
                                                              Cta_tile, 
                                                              Input_related, 
                                                              BYTES_PER_LDG>,
                                 DISABLE_LDGSTS>::Class;

    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b( const Params& params,
                                   void* smem,
                                   const dim3& bidx,
                                   int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   C
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Layout_C
    typename Layout_,
    // The number of bytes per STG.
    int BYTES_PER_STG_,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>,
    // Assuming tensor is packed and tensor strides won't be used.
    bool DISABLE_STRIDES = false
>
struct Gmem_tile_epilogue_base {
    
    // STG Sizing Scale
    enum { STG_SIZING = (Cta_tile::N == 128) ? 1 : 2 };
    // Resize STG
    enum { BYTES_PER_STG = BYTES_PER_STG_ / STG_SIZING };
    
    // Output layout
    using Layout = Layout_;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The helper class to compute the row offset (in the tile).
    using Tile_distribution = xmma::helpers::Gmem_tile_epilogue_distribution<Traits, Cta_tile, Layout, BYTES_PER_STG>;

    // The number of bytes per STG.
    // enum { BYTES_PER_STG = BYTES_PER_STG_ };

    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    // The number of elements per row.
    enum { ELEMENTS_PER_ROW = Layout::ROW ? Cta_tile::N : Cta_tile::M };
    // The number of threads needed to store a row.
    enum { THREADS_PER_ROW =
        Min<Cta_tile::THREADS_PER_CTA, ELEMENTS_PER_ROW / ELEMENTS_PER_STG>::VALUE };

    // The number of column loaded per STG
    enum { COLUMNS_PER_STG = THREADS_PER_ROW * ELEMENTS_PER_STG };

    // The number of rows that are written with a single STG (accross the CTA).
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // The number of STGS needed to load a complete row.
    enum { STGS_PER_ROW = ELEMENTS_PER_ROW / COLUMNS_PER_STG };
    static_assert(STGS_PER_ROW > 0, "");

    // The number of rows to store per XMMA per CTA.
    enum { ROWS_PER_XMMA_PER_CTA = Layout::ROW ? Xmma_tile::M_PER_XMMA_PER_CTA
                                               : Xmma_tile::N_PER_XMMA_PER_CTA };
    // The number of steps needed to load the columns.
    enum { STGS_PER_COLUMN = ROWS_PER_XMMA_PER_CTA / ROWS_PER_STG };

    enum { ACC_STRIDE = 8 };

    // The number of STGs needed to store the elements per iteration.
    enum { STGS = STGS_PER_COLUMN * STGS_PER_ROW };
    static_assert(STGS > 0, "");

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_epilogue_base( const Params& params,
                                               int bidm,
                                               int bidn,
                                               int bidz,
                                               int tidx)
        : Gmem_tile_epilogue_base(params.out_gmem,
                             params.res_gmem,
                             params.n,
                             params.o,
                             params.p,
                             params.q,
                             params.k * params.g,
                             params.nopq,
                             params.out_stride_n,
                             params.out_stride_d,
                             params.out_stride_h,
                             params.out_stride_w,
                             params.out_stride_c,
                             params.q,
                             params.mul_q,
                             params.shr_q,
                             params.pq,
                             params.mul_pq,
                             params.shr_pq,
                             params.opq,
                             params.mul_opq,
                             params.shr_opq,
                             bidm,
                             bidn,
                             bidz,
                             tidx ) {
    }

    // Ctor.
    inline __device__ Gmem_tile_epilogue_base(void *out_ptr,
                                         const void *res_ptr,
                                         int n,
                                         int d,
                                         int h,
                                         int w,
                                         int c,
                                         int ndhw,
                                         int stride_n,
                                         int stride_d,
                                         int stride_h,
                                         int stride_w,
                                         int stride_c,
                                         int div_w,
                                         int mul_w,
                                         int shr_w,
                                         int div_hw,
                                         int mul_hw,
                                         int shr_hw,
                                         int div_dhw,
                                         int mul_dhw,
                                         int shr_dhw,
                                         int bidm,
                                         int bidn,
                                         int bidz,
                                         int tidx)
        : params_out_ptr_(out_ptr)
        ,out_ptr_(out_ptr)
        , params_res_ptr_(res_ptr)
        ,res_ptr_(res_ptr)
        , params_n_(n)
        , params_d_(d)
        , params_h_(h)
        , params_w_(w)
        , params_c_(c)
        , params_ndhw_(ndhw)
        , params_stride_n_(stride_n)
        , params_stride_d_(stride_d)
        , params_stride_h_(stride_h)
        , params_stride_w_(stride_w)
        , params_stride_c_(stride_c)
        , params_div_w_(div_w)
        , params_mul_w_(mul_w)
        , params_shr_w_(shr_w)
        , params_div_hw_(div_hw)
        , params_mul_hw_(mul_hw)
        , params_shr_hw_(shr_hw)
        , params_div_dhw_(div_dhw)
        , params_mul_dhw_(mul_dhw)
        , params_shr_dhw_(shr_dhw) {

        if (Layout::ROW) {
            // The number of warps.
            const int WARPS_M = Cta_tile::WARPS_M;
            const int WARPS_N = Cta_tile::WARPS_N;
            const int WARPS_K = Cta_tile::WARPS_K;

            // The masks to select the warps.
            const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
            const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

            // The divisor for the warps.
            const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
            const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

            int warp_m = (tidx & WARP_MASK_M) / WARP_DIV_M;
            int warp_n = (tidx & WARP_MASK_N) / WARP_DIV_N;

            int row = (tidx % Cta_tile::THREADS_PER_WARP) / (THREADS_PER_ROW / 2) + warp_m * Xmma_tile::M_PER_XMMA ; // 16
            int col = tidx % (THREADS_PER_ROW / 2) ;
            
            ndhw_ = bidm * Cta_tile::M + row;
            c_ = bidn * Cta_tile::N + col * ELEMENTS_PER_STG + warp_n * (Cta_tile::N / 2);
        } else {
            assert(0);
        }
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask(int mi, int ni) {
        // Compute the new offsets for that iteration.
        if( ni == 0 ) {
            #pragma unroll
            for( int ii = 0; ii < STGS_PER_ROW; ++ii )
            #pragma unroll
            for( int jj = 0; jj < STGS_PER_COLUMN; ++jj ) {
                int ndhw = ndhw_;
                int c = c_;
                if (Layout::ROW) {
                    // The row index..
                    //ndhw += Tile_distribution::compute_offset(mi, jj);
                    // M_PER_XMMA_PER_CTA -- 32, ACC_STRIDE -- 8
                    ndhw += mi * Xmma_tile::M_PER_XMMA_PER_CTA + jj * ACC_STRIDE;
                    c += ii * COLUMNS_PER_STG;
                } else {
                    assert(0);
                }

                // Finally assemble the offset.
                int offset = -1;

                if( DISABLE_STRIDES ) {
                    if(Layout::ROW) {
                        if(ndhw < params_ndhw_ && c < params_c_) {
                            offset = ndhw * params_c_ + c;
                        }
                    } else {
                        assert(0);
                    }
                } else {
                    // Decompose the position into N and OPQ.
                    int n, dhw;
                    xmma::fast_divmod(n, dhw, ndhw, params_div_dhw_, 
                                                        params_mul_dhw_, 
                                                        params_shr_dhw_);

                    // Decompose the position into D and HW.
                    int d, hw;
                    xmma::fast_divmod(d, hw, dhw, params_div_hw_, 
                                                    params_mul_hw_, 
                                                    params_shr_hw_);

                    // Decompose the position into H and W.
                    int h, w;
                    xmma::fast_divmod(h, w, hw, params_div_w_, 
                                                    params_mul_w_, 
                                                    params_shr_w_);

                    if( ndhw < params_ndhw_ && c < params_c_ ) {
                        offset = n * params_stride_n_ + 
                                 d * params_stride_d_ + 
                                 h * params_stride_h_ + 
                                 w * params_stride_w_ + 
                                 c * params_stride_c_;
                    }
                }
                offsets_[ii * STGS_PER_COLUMN + jj] = offset;
            }
        }

        // Is it a valid mask?
        return  offsets_[ni] >= 0;
    }

    // Load the data from global memory.
    inline __device__ void load(Fragment_c &data, 
                                int mi,
                                int ii,
                                int mask,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {
        const char *ptr = reinterpret_cast<const char*>(params_res_ptr_);
        if( mask ) {
            if (BYTES_PER_STG == 16) {
                uint4 tmp;
                xmma::ldg(tmp, ptr + Traits::offset_in_bytes_c(offsets_[ii]), mem_desc);
                data.from_int4( tmp );
            } else if (BYTES_PER_STG == 8) {
                uint2 tmp;
                xmma::ldg(tmp, ptr + Traits::offset_in_bytes_c(offsets_[ii]), mem_desc);
                data.from_int2( tmp );
            } else {
                uint32_t tmp;
                xmma::ldg(tmp, ptr + Traits::offset_in_bytes_c(offsets_[ii]), mem_desc);
                data.reg(0) = tmp;
            }
        }
    }
    
    // Load residual from gmem to smem buffers.
    template<typename Smem_tile_a, typename Smem_tile_b>
    inline __device__ void load_residual_to_smem(
                                Smem_tile_a &smem_tile_a,
                                Smem_tile_b &smem_tile_b,
                                int mi, 
                                int ii, 
                                int mask,
                                int xmma_tiles_per_a,
                                int xmma_tiles_per_b,
                                int tidx,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {                     
        const char *ptr = reinterpret_cast<const char*>(params_res_ptr_);
        if( mask ) {
          
          int xmma_tile_idx = (mi * STGS + ii) % (xmma_tiles_per_a + xmma_tiles_per_b);
          uint32_t smem_ptr;
          
          if (xmma_tile_idx < xmma_tiles_per_a) 
            smem_ptr = smem_tile_a.smem_ + smem_tile_a.smem_write_buffer_ + 
            xmma_tile_idx *  Cta_tile::THREADS_PER_CTA * BYTES_PER_STG + tidx *BYTES_PER_STG;
          else
            smem_ptr = smem_tile_b.smem_ + smem_tile_b.smem_write_buffer_ + 
            (xmma_tile_idx - xmma_tiles_per_a) *  Cta_tile::THREADS_PER_CTA * BYTES_PER_STG + 
            tidx * BYTES_PER_STG;

            if (BYTES_PER_STG == 16) {
              ldgsts128(smem_ptr, ptr + Traits::offset_in_bytes_c(offsets_[ii]), true, mem_desc);

            } else if (BYTES_PER_STG == 8) {
              ldgsts64(smem_ptr, ptr + Traits::offset_in_bytes_c(offsets_[ii]), true, mem_desc);

            } else {
              ldgsts32(smem_ptr, ptr + Traits::offset_in_bytes_c(offsets_[ii]), true, mem_desc);
            }
        }      
    }

    // Store the data to global memory.
    inline __device__ void store(int mi,
                                 int ii,
                                 const Fragment_c &data, int mask,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        char *ptr = reinterpret_cast<char*>(params_out_ptr_);
        if( mask ) {
            if (BYTES_PER_STG == 16) {
                xmma::stg(ptr + Traits::offset_in_bytes_c(offsets_[ii]),
                    data.to_int4(), mem_desc);
            } else if (BYTES_PER_STG == 8) {
                xmma::stg(ptr + Traits::offset_in_bytes_c(offsets_[ii]),
                    make_uint2(data.reg(0),data.reg(1)), mem_desc);
            } else {
                xmma::stg(ptr + Traits::offset_in_bytes_c(offsets_[ii]),
                    data.reg(0), mem_desc);
            }
        }
    }
    
    // The pointer to the output buffer.
    void *const out_ptr_;
    // The pointer to the output buffer. 
    //TODO: Change params_out_ptr_ to out_ptr_ to be conssitent with gemm.
    void *const params_out_ptr_;

    // The pointer to the input residual buffer.
    const void *const res_ptr_;
    // The pointer to the input residual buffer.
    //TODO: Change params_res_ptr_ to res_ptr_ to be conssitent with gemm.
    const void *const params_res_ptr_;
    // The dimensions of the output.
    const int params_n_, params_d_, params_h_, params_w_, params_c_;
    // The strides.
    const int params_stride_n_, params_stride_d_, params_stride_h_,
        params_stride_w_, params_stride_c_;
    // The constants to help with faster division.
    const int params_div_w_; const unsigned params_mul_w_, params_shr_w_;
    // The constants to help with faster division.
    const int params_div_hw_; const unsigned params_mul_hw_, params_shr_hw_;
    // The constants to help with faster division.
    const int params_div_dhw_; const unsigned params_mul_dhw_, params_shr_dhw_;
    // The position of the thread in the 2D output matrix.
    int ndhw_, c_, params_ndhw_;
    // Offsets.
    int offsets_[STGS];
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bytes per STG.
    int BYTES_PER_STG,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>,
    bool DISABLE_STRIDES = false
>
struct Gmem_tile_c_t : public Gmem_tile_epilogue_base<Traits, Cta_tile,
    xmma::Row, BYTES_PER_STG, Fragment_c, DISABLE_STRIDES> {

    template <typename Params>
    inline __device__ Gmem_tile_c_t( const Params& params,
                                     int bidm,
                                     int bidn,
                                     int bidz,
                                     int tidx )
       : Gmem_tile_epilogue_base<Traits,
                                 Cta_tile,
                                 xmma::Row,
                                 BYTES_PER_STG,
                                 Fragment_c,
                                 DISABLE_STRIDES> ( params, bidm, bidn, bidz, tidx ) {
    }
    template <typename Params>
    inline __device__ Gmem_tile_c_t(const Params& params, int bidm,
        int bidn, int tidx, void* lwstom_residual)
        : Gmem_tile_epilogue_base<Traits,
                                  Cta_tile,
                                  xmma::Row,
                                  BYTES_PER_STG,
                                  Fragment_c,
                                  DISABLE_STRIDES>(params.out_gmem,
                                                   lwstom_residual,
                                                   params.n,
                                                   params.o,
                                                   params.p,
                                                   params.q,
                                                   params.k * params.g,
                                                   params.nopq,
                                                   params.out_stride_n,
                                                   params.out_stride_d,
                                                   params.out_stride_h,
                                                   params.out_stride_w,
                                                   params.out_stride_c,
                                                   params.q,
                                                   params.mul_q,
                                                   params.shr_q,
                                                   params.pq,
                                                   params.mul_pq,
                                                   params.shr_pq,
                                                   params.opq,
                                                   params.mul_opq,
                                                   params.shr_opq,
                                                   bidm,
                                                   bidn,
                                                   tidx ) {
    }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fprop_wo_epi_swizzle
}  // namespace implicit_gemm
}  // namespace xmma
