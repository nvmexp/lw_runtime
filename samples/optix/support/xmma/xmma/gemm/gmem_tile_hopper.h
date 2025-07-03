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
#include <xmma/helpers/epilogue.h>
#include "xmma/hopper/gmma_descriptor.h"

namespace xmma {
namespace gemm {

/////////////////////////////////////////////////////////////////////////////////////////////////////
// There is a separate excel spreadsheet that dolwments the ldgsts pattern. Should add link
// M is number of rows, N is number of columns, if column major view. Use M, N here to be consistent
// with the XMMA style
// Gmem tile for GMMA
template <typename Traits,
          typename Cta_tile,
          int M,
          int N,
          int BITS_PER_ELEMENT,
          xmma::Gmma_descriptor_mode desc_mode>
struct Gmem_tile_gmma_base {
    // Lwrrently Interleaved Mode is not implemented. 
    static_assert(desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_NONE, 
      "Lwrrently, Interleaved Mode is not implemented.\n");
    
    // Hopper GMMA lwrrently uses ldgsts.
    enum { USE_LDGSTS = 1 };

    // Only ldgsts.128 is used lwrrently.
    enum { BYTES_PER_LDG = 16 };

    // The amount of extra shared memory needed by the tile.
    enum { BYTES_PER_EXTRA_SMEM = 0 };

    enum { USE_UTMALDG = 0 };

    // SWIZZLE_128B mode.
    static constexpr xmma::Gmma_descriptor_mode GMMA_DESC_MODE = desc_mode;

    // The number of elements per LDG.128.
    enum { ELTS_PER_LDG = BYTES_PER_LDG * 8 / BITS_PER_ELEMENT };
    // Make sure we have a "nice" number of elements per LDG.
    static_assert( ELTS_PER_LDG > 0, "" );

    // With SWIZZLE_128B mode, 
    // each strip (128B) is designed to be consumed by 8 threads, 
    // with each thread calling ldgsts.128
    // With SWIZZLE_64B mode, 
    // each strip (64B) is designed to be consumed by 4 threads, 
    // with each thread calling ldgsts.128
    enum {
        ELEMENT_PER_STRIP = ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B ? 128 : 64 ) /
                            ( BITS_PER_ELEMENT / 8 )
    };

    // Number of strips along the leading dimension
    enum { LDGS_M = ( M + ELEMENT_PER_STRIP - 1 ) / ELEMENT_PER_STRIP };
    // LDGS_M must be 1, otherwise, SWIZZLE_128B mode should have been chosen
    static_assert( ( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_64B ) || LDGS_M == 1,
                   "LDGS_M should have been 1 for SWIZZLE_64B mode\n" );

    // The number of threads needed to load a column. Each thread does LDG.128.
    enum { THREADS_PER_STRIP = ELEMENT_PER_STRIP / ELTS_PER_LDG };

    // Make sure we have a "nice" number of pixels.
    static_assert( THREADS_PER_STRIP > 0, "" );

    // Threads per strip is either 4 or 8
    static_assert( ( THREADS_PER_STRIP == 4 || THREADS_PER_STRIP == 8 ),
                   "Threads per strip is either 4 or 8" );

    // The number of strips loaded per LDG per CTA
    enum { STRIPS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_STRIP };
    // Make sure we have a "nice" number of columns.
    static_assert( N % STRIPS_PER_LDG == 0, "" );

    // The number of steps needed to load the columns.
#ifdef GMMA_QUARTER_LDGSTS
    // temperary hack, issue 1/4 ldgsts for perf build.
    enum { LDGS_N = N / STRIPS_PER_LDG / 4 };
#else
    enum { LDGS_N = N / STRIPS_PER_LDG };
#endif
    //
    enum { LDGS = LDGS_M * LDGS_N };
    // Make sure we have a "nice" number of LDGs.
    static_assert( LDGS > 0, "" );

    // The number of predicate registers.
    enum { PRED_REGS = Compute_number_of_pred_regs<LDGS>::VALUE };

    // The number of predicates that we store per register.
    enum { PREDS_PER_REG = 4 };

    // Below enums are used by implcit gemm Gmem_tile_base_a/b.
    // The unroll factor for LDGS.
    enum { LDGS_UNROLL = 16 / BYTES_PER_LDG };
    // Number of threads per column
    enum { THREADS_PER_COLUMN = THREADS_PER_STRIP };
    // The number of columns loaded per LDG per CTA
    enum { COLUMNS_PER_LDG = STRIPS_PER_LDG };

    static const xmma::Copy_engine COPY_ENGINE = xmma::Copy_engine::CE_LDGSTS;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_gmma_base( const Params &params,
                                           const void *ptr )
        : params_k_( params.k ), params_residue_k_( params.loop_residue_k ),
          ptr_( reinterpret_cast<const char *>( ptr ) ) {
    }

    // Ctor.
    // Used by implcit gemm Gmem_tile_base_a/b.
    template <typename Params>
    inline __device__ Gmem_tile_gmma_base( const Params &params,
                                           void *,
                                           int k,
                                           const void *ptr,
                                           int bidz = blockIdx.z )
        : params_k_( k )
        , params_residue_k_( params.loop_residue_k )
        , ptr_( reinterpret_cast<const char *>( ptr ) )
        , bidz_( bidz ) {
    }

    // Load a tile from global memory.
    template <typename Xmma_smem_tile> inline __device__ void load( Xmma_smem_tile &smem ) {
        const char *ptrs[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = this->ptr_ + Traits::offset_in_bytes_a( this->offsets_[ii] );
        }
        // Issue the ldgsts.
        smem.store<LDGS_M, LDGS_N>( ptrs, this->preds_ );
    }

    template <typename Smem_tile>
    inline __device__ void load( Smem_tile &smem_tile, uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const void *ptrs[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = this->ptr_ + Traits::offset_in_bytes_a( this->offsets_[ii] );
        }
        smem_tile.store<LDGS_M, LDGS_N>( ptrs, this->preds_, mem_desc );
    }

    // Store the pixels to shared memory.
    template <typename Xmma_smem_tile> inline __device__ void commit( Xmma_smem_tile &smem ) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ < 800
        smem.store( fetch_ );
#endif
    }

    // Disable the loads.
    inline __device__ void disable_loads() {
        #pragma unroll
        for( int ii = 0; ii < PRED_REGS; ++ii ) {
            this->preds_[ii] = 0u;    
        }
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move( int, int64_t delta ) {
        ptr_ += delta;
    }

    inline __device__ void precompute_residue_predicates_a_n_b_t(int split_k = Cta_tile::K) {
    }

    // The residue to "fix" the predicates.
    inline __device__ int residue_a_n_b_t(int split_k = Cta_tile::K) {
        return 0;
    }

    inline __device__ void precompute_residue_predicates_a_t_b_n(int split_k = Cta_tile::K) {
    }

    // The residue to "fix" the predicates.
    inline __device__ int residue_a_t_b_n(int split_k = Cta_tile::K) {
        return 0;
    }

    // The K dimension.
    const int params_k_, params_residue_k_;
    // The pointer.
    const char *ptr_;
    // The associated offsets.
    int offsets_[LDGS];
    // The predicates.
    uint32_t preds_[PRED_REGS];
    // The fetch registers.
    int4 fetch_[LDGS];
    // The blockIdx.z passed in from tile distribution.
    int bidz_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// GMEM Tile for A
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int M, int N, xmma::Gmma_descriptor_mode desc_mode>
struct Gmem_tile_gmma_a
    : public Gmem_tile_gmma_base<Traits, Cta_tile, M, N, Traits::BITS_PER_ELEMENT_A, desc_mode> {

    // The base class.
    using Base = Gmem_tile_gmma_base<Traits, Cta_tile, M, N, Traits::BITS_PER_ELEMENT_A, desc_mode>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_gmma_a( const Params &params ) : Base( params, params.a_gmem ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, xmma::Gmma_descriptor_mode desc_mode>
struct Gmem_tile_gmma_a_n
    : public Gmem_tile_gmma_a<Traits, Cta_tile, Cta_tile::M, Cta_tile::K, desc_mode> {

    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_64B,
                   "Lwrrently, for SWIZZLE_64B mode, a_n is not needed/implemented" );

    // The base class.
    using Base = Gmem_tile_gmma_a<Traits, Cta_tile, Cta_tile::M, Cta_tile::K, desc_mode>;
    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_gmma_a_n( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params ) {
        // The position in the M dimension.
        int m[Base::LDGS_M];
        m[0] = bidx.x * Cta_tile::M + ( tidx % Base::THREADS_PER_STRIP ) * Base::ELTS_PER_LDG;
        #pragma unroll
        for( int mi = 1; mi < Base::LDGS_M; ++mi ) {
            m[mi] = m[0] + mi * Base::ELEMENT_PER_STRIP;
        }

        // The K position for the thread.
        int k[Base::LDGS_N];
        k[0] = bidx.z * Cta_tile::K + tidx / Base::THREADS_PER_STRIP;
        #pragma unroll
        for( int ki = 1; ki < Base::LDGS_N; ++ki ) {
            k[ki] = k[0] + ki * Base::STRIPS_PER_LDG;
        }

        // Compute the offsets.
        #pragma unroll
        for( int ki = 0; ki < Base::LDGS_N; ++ki ) {
            #pragma unroll
            for( int mi = 0; mi < Base::LDGS_M; ++mi ) {
                this->offsets_[ki * Base::LDGS_M + mi] = k[ki] * params.lda + m[mi];
            }
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        #pragma unroll
        for( int ki = 0; ki < Base::LDGS_N; ++ki ) {
            #pragma unroll
            for( int mi = 0; mi < Base::LDGS_M; ++mi ) {
                if( ( k[ki] < params.k ) && ( m[mi] < params.m ) ) {
                    preds[ki * Base::LDGS_M + mi] = 1;
                } else {
                    preds[ki * Base::LDGS_M + mi] = 0;
                }
            }
        }

        // Finalize the predicates.
        // asm volatile("set.lt.u32.u32 %0, %1, %2;" : "=r"(this->preds_) : "r"(m), "r"(params.m));
        xmma::pack_predicates( this->preds_, preds );
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_n_b_t();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, xmma::Gmma_descriptor_mode desc_mode>
struct Gmem_tile_gmma_a_t
    : public Gmem_tile_gmma_a<Traits, Cta_tile, Cta_tile::K, Cta_tile::M, desc_mode> {

    // The base class.
    using Base = Gmem_tile_gmma_a<Traits, Cta_tile, Cta_tile::K, Cta_tile::M, desc_mode>;
    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_gmma_a_t( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params ) {
        // The position in the K dimension.
        int k[Base::LDGS_M];
        k[0] = bidx.z * Cta_tile::K + tidx % Base::THREADS_PER_STRIP * Base::ELTS_PER_LDG;
        #pragma unroll
        for( int ki = 1; ki < Base::LDGS_M; ++ki ) {
            k[ki] = k[0] + ki * Base::ELEMENT_PER_STRIP;
        }

        // For each LDG, compute the M position.
        int m[Base::LDGS_N];
        m[0] = bidx.x * Cta_tile::M + tidx / Base::THREADS_PER_STRIP;
        #pragma unroll
        for( int mi = 1; mi < Base::LDGS_N; ++mi ) {
            m[mi] = m[0] + mi * Base::STRIPS_PER_LDG;
        }

        // Compute the offsets.
        #pragma unroll
        for( int mi = 0; mi < Base::LDGS_N; ++mi ) {
            #pragma unroll
            for( int ki = 0; ki < Base::LDGS_M; ++ki ) {
                this->offsets_[mi * Base::LDGS_M + ki] = m[mi] * params.lda + k[ki];
            }
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        #pragma unroll
        for( int mi = 0; mi < Base::LDGS_N; ++mi ) {
            #pragma unroll
            for( int ki = 0; ki < Base::LDGS_M; ++ki ) {
                if( ( k[ki] < params.k ) && ( m[mi] < params.m ) ) {
                    preds[mi * Base::LDGS_M + ki] = 1;
                } else {
                    preds[mi * Base::LDGS_M + ki] = 0;
                }
            }
        }
        // Finalize the predicates.
        // asm volatile("set.lt.u32.u32 %0, %1, %2;" : "=r"(this->preds_) : "r"(k), "r"(params.k));
        xmma::pack_predicates( this->preds_, preds );
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_t_b_n();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, xmma::Gmma_descriptor_mode desc_mode>
struct Gmem_tile_gmma_a_t_bn_apply
    : public Gmem_tile_gmma_a_t<Traits, Cta_tile, desc_mode> {
    // The base class.
    using Base = Gmem_tile_gmma_a_t<Traits, Cta_tile, desc_mode>;
    // should probably add only SWIZZLE_128B is supported    
    static_assert(desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B, 
      "Lwrrently, bn apply kernels only support SWIZZLE_128B mode. \n");    
    // scale and bias are also in fp16.
    enum { BYTES_PER_SCALE_BIAS_ELEMENT = Traits::BITS_PER_ELEMENT_A / 8 };
    // make sure to use ldgsts.128 to load scale and bias.
    enum { BYTES_PER_LDGSTS_SCALE_BIAS = 16 };
    
    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_gmma_a_t_bn_apply( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base(params, smem, bidx, tidx)
        , gmem_scale_bias_ptr_(reinterpret_cast<const char*>(params.scale_bias_gmem)) {
        // each thread will load one ldgsts.128 for scale or bias
        // assuming kblock = 64, only 16 threads are needed to load all 256B of data
        // each thread will do ldgsts.128
        // some magic numbers maybe not great. 
        if(tidx < 16) {
          if(tidx < 8) {
            gmem_scale_bias_ptr_ += tidx * BYTES_PER_LDGSTS_SCALE_BIAS;
          } else {
            gmem_scale_bias_ptr_ += (tidx % 8) * BYTES_PER_LDGSTS_SCALE_BIAS + params.k * BYTES_PER_SCALE_BIAS_ELEMENT;
          }
        }
        
    }
    
    // Load a tile from global memory.
    // Load scale and bias from global memory. 
    template< typename Smem_tile >
    inline __device__ void load(Smem_tile &smem_tile, uint64_t mem_desc = MEM_DESC_DEFAULT) {
        // Issue the LDGSTS for the Operand.
        Base::load(smem_tile, mem_desc);        
        // Issue the LDGSTS for scale and bias.
        smem_tile.store_scale_bias(gmem_scale_bias_ptr_);
    }
    
    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move(int trsi, int64_t delta) {
        Base::move(trsi, delta);
        // be careful to not access out of bound. 
        gmem_scale_bias_ptr_ += delta;
    }
    
    // The scale and bias pointer
    const char *gmem_scale_bias_ptr_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// GMEM Tile for B
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int M, int N, xmma::Gmma_descriptor_mode desc_mode>
struct Gmem_tile_gmma_b
    : public Gmem_tile_gmma_base<Traits, Cta_tile, M, N, Traits::BITS_PER_ELEMENT_B, desc_mode> {

    // The base class.
    using Base = Gmem_tile_gmma_base<Traits, Cta_tile, M, N, Traits::BITS_PER_ELEMENT_B, desc_mode>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_gmma_b( const Params &params ) : Base( params, params.b_gmem ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, xmma::Gmma_descriptor_mode desc_mode>
struct Gmem_tile_gmma_b_n
    : public Gmem_tile_gmma_b<Traits, Cta_tile, Cta_tile::K, Cta_tile::N, desc_mode> {

    // The base class.
    using Base = Gmem_tile_gmma_b<Traits, Cta_tile, Cta_tile::K, Cta_tile::N, desc_mode>;
    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_gmma_b_n( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params ) {
        // The position in the K dimension.
        int k[Base::LDGS_M];
        k[0] = bidx.z * Cta_tile::K + tidx % Base::THREADS_PER_STRIP * Base::ELTS_PER_LDG;
        #pragma unroll
        for( int ki = 1; ki < Base::LDGS_M; ++ki ) {
            k[ki] = k[0] + ki * Base::ELEMENT_PER_STRIP;
        }

        // For each LDG, compute the M position.
        int n[Base::LDGS_N];
        n[0] = bidx.y * Cta_tile::N + tidx / Base::THREADS_PER_STRIP;
        #pragma unroll
        for( int ni = 1; ni < Base::LDGS_N; ++ni ) {
            n[ni] = n[0] + ni * Base::STRIPS_PER_LDG;
        }

        // Compute the offsets.
        #pragma unroll
        for( int ni = 0; ni < Base::LDGS_N; ++ni ) {
            #pragma unroll
            for( int ki = 0; ki < Base::LDGS_M; ++ki ) {
                this->offsets_[ni * Base::LDGS_M + ki] = n[ni] * params.ldb + k[ki];
            }
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        #pragma unroll
        for( int ni = 0; ni < Base::LDGS_N; ++ni ) {
            #pragma unroll
            for( int ki = 0; ki < Base::LDGS_M; ++ki ) {
                if( ( k[ki] < params.k ) && ( n[ni] < params.n ) ) {
                    preds[ni * Base::LDGS_M + ki] = 1;
                } else {
                    preds[ni * Base::LDGS_M + ki] = 0;
                }
            }
        }
        // Finalize the predicates.
        // asm volatile("set.lt.u32.u32 %0, %1, %2;" : "=r"(this->preds_) : "r"(k), "r"(params.k));
        xmma::pack_predicates( this->preds_, preds );
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_t_b_n();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, xmma::Gmma_descriptor_mode desc_mode>
struct Gmem_tile_gmma_b_t
    : public Gmem_tile_gmma_b<Traits, Cta_tile, Cta_tile::N, Cta_tile::K, desc_mode> {

    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_64B,
                   "Lwrrently, for SWIZZLE_64B mode, b_t is not needed/implemented" );

    // The base class.
    using Base = Gmem_tile_gmma_b<Traits, Cta_tile, Cta_tile::N, Cta_tile::K, desc_mode>;
    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_gmma_b_t( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params ) {
        // The position in the N dimension.
        int n[Base::LDGS_M];
        n[0] = bidx.y * Cta_tile::N + tidx % Base::THREADS_PER_STRIP * Base::ELTS_PER_LDG;
        #pragma unroll
        for( int ni = 1; ni < Base::LDGS_M; ++ni ) {
            n[ni] = n[0] + ni * Base::ELEMENT_PER_STRIP;
        }

        // The K position for the thread.
        int k[Base::LDGS_N];
        k[0] = bidx.z * Cta_tile::K + tidx / Base::THREADS_PER_STRIP;
        #pragma unroll
        for( int ki = 1; ki < Base::LDGS_N; ++ki ) {
            k[ki] = k[0] + ki * Base::STRIPS_PER_LDG;
        }

        // Compute the offsets.
        #pragma unroll
        for( int ki = 0; ki < Base::LDGS_N; ++ki ) {
            #pragma unroll
            for( int ni = 0; ni < Base::LDGS_M; ++ni ) {
                this->offsets_[ki * Base::LDGS_M + ni] = k[ki] * params.ldb + n[ni];
            }
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        #pragma unroll
        for( int ki = 0; ki < Base::LDGS_N; ++ki ) {
            #pragma unroll
            for( int ni = 0; ni < Base::LDGS_M; ++ni ) {
                if( ( k[ki] < params.k ) && ( n[ni] < params.n ) ) {
                    preds[ki * Base::LDGS_M + ni] = 1;
                } else {
                    preds[ki * Base::LDGS_M + ni] = 0;
                }
            }
        }

        // Finalize the predicates.
        // asm volatile("set.lt.u32.u32 %0, %1, %2;" : "=r"(this->preds_) : "r"(n), "r"(params.n));
        xmma::pack_predicates( this->preds_, preds );
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_n_b_t();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          typename Layout_c,
          typename Fragment_c = xmma::Fragment_gmma_c<Traits, Cta_tile, Layout_c>>
struct Gmem_tile_gmma_epilogue {};

///////////////////////////////////////////////////////////////////////////////////////////////////
// column major output
template <typename Traits, typename Cta_tile, typename Fragment_c>
struct Gmem_tile_gmma_epilogue<Traits, Cta_tile, xmma::Col, Fragment_c> {
    // HGMMA operation, where A and B should be in fp16. 
    static_assert(sizeof(typename Traits::A_type) == 2 && sizeof(typename Traits::B_type) == 2,
      "HGMMA operation, where A and B should be in fp16 is required.\n");

    // The helper class to compute the row offset (in the tile).
    using Tile_distribution =
        xmma::helpers::Gmem_tile_gmma_epilogue_distribution<Traits, Cta_tile, xmma::Col, 16>;

    //
    enum { BYTES_PER_ELEMENT = 2 };
    //
    enum { BITS_PER_ELEMENT = BYTES_PER_ELEMENT * 8 };
    // CTA tile size
    enum { CTA_M = Cta_tile::M, CTA_N = Cta_tile::N };
    // The number of threads
    enum { THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA };
    // the size for each STG.128
    enum { BYTES_PER_STG = 16 };
    // the number of elements for each stg.128
    enum { ELEMENTS_PER_STG = BYTES_PER_STG / BYTES_PER_ELEMENT };
    // Threads for STG per column
    enum { STG_THREADS_PER_COLUMN = CTA_M * BYTES_PER_ELEMENT / BYTES_PER_STG };
    static_assert( STG_THREADS_PER_COLUMN >= 8,
                   "STG_THREADS_PER_COLUMN should be larger than 8\n" );
    // the number of columns can be store by all threads per STG instruction
    enum { COLUMNS_PER_STG = THREADS_PER_CTA / STG_THREADS_PER_COLUMN };
    // we can probably reduce the tile M to MIN_TILE_M, but for simplicity we set tile_M = cta_M
    enum { TILE_M = CTA_M, TILE_N = COLUMNS_PER_STG < 8 ? 8 : COLUMNS_PER_STG };
    // the min tile in N dim is 8 such that every thread can participate in sts
    static_assert( TILE_N % 8 == 0, "TILE_N should be multiple of 8" );
    // the number of inner iterations
    enum { STG_ITERATIONS_PER_TILE = TILE_N / COLUMNS_PER_STG };

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_gmma_epilogue( const Params &params, int bidm, int bidn, int bidz, int tidx )
        : params_m_( params.m ), params_n_( params.n ), params_stride_m_( params.ldd ) {

        // Compute the output position for each thread.
        int row = Tile_distribution::compute_row( tidx );
        int col = Tile_distribution::compute_col( tidx );

        m_ = bidm * Cta_tile::M + row;
        n_ = bidn * Cta_tile::N + col;

        // The pointer.
        const int64_t offset = Traits::offset_in_bytes_c( m_ + n_ * params_stride_m_ );
        out_ptr_ = &reinterpret_cast<char *>( params.d_gmem )[offset];
        res_ptr_ = &reinterpret_cast<const char *>( params.c_gmem )[offset];
    }

    // store to gmem
    template <typename Fragment_pre_stg>
    inline __device__ void store( int ni, const Fragment_pre_stg &acc_pre_stg ) {
        // ptr_ += ni * TILE_N * params_stride_m_ * BYTES_PER_ELEMENT;
        const int offset = ni * TILE_N * params_stride_m_;
        char *ptr = &out_ptr_[Traits::offset_in_bytes_c( offset )];
        #pragma unroll
        for( int stg_idx = 0; stg_idx < STG_ITERATIONS_PER_TILE; ++stg_idx ) {
            ptr += stg_idx * COLUMNS_PER_STG * params_stride_m_ * BYTES_PER_ELEMENT;
            int acc_idx = stg_idx * 4;
            uint4 tmp = make_uint4( acc_pre_stg.regs_[acc_idx],
                                    acc_pre_stg.regs_[acc_idx + 1],
                                    acc_pre_stg.regs_[acc_idx + 2],
                                    acc_pre_stg.regs_[acc_idx + 3] );
            // if( mask ) {
            xmma::stg( ptr, tmp );
            //}
        }
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask( int xmmas_ni, int mi, int ni, int ii ) {
        const int offset = Tile_distribution::compute_offset( xmmas_ni, ni, ii );
        return m_ < params_m_ && ( n_ + offset ) < params_n_;
    }

    // Load the data from global memory.
    inline __device__ void load( Fragment_c &data,
                                 int xmmas_ni,
                                 int mi,
                                 int ni,
                                 int ii,
                                 int mask,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const int offset = Tile_distribution::compute_offset( xmmas_ni, ni, ii );
        const char *ptr = res_ptr_ + Traits::offset_in_bytes_c( offset * params_stride_m_ );

        if( mask ) {
            if( BYTES_PER_STG == 16 ) {
                uint4 tmp;
                xmma::ldg( tmp, ptr, mem_desc );
                data.from_int4( tmp );
            } else if( BYTES_PER_STG == 8 ) {
                uint2 tmp;
                xmma::ldg( tmp, ptr, mem_desc );
                data.from_int2( tmp );
            } else {
                uint32_t tmp;
                xmma::ldg( tmp, ptr, mem_desc );
                data.reg( 0 ) = tmp;
            }
        }
    }

    // store to gmem
    inline __device__ void store( int xmmas_ni,
                                  int mi,
                                  int ni,
                                  int ii,
                                  const Fragment_c &data,
                                  int mask,
                                  uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const int offset = Tile_distribution::compute_offset( xmmas_ni, ni, ii );
        char *ptr = out_ptr_ + Traits::offset_in_bytes_c( offset * params_stride_m_ );

        if( mask ) {
            if( BYTES_PER_STG == 16 ) {
                xmma::stg( ptr, data.to_int4(), mem_desc );
            } else if( BYTES_PER_STG == 8 ) {
                xmma::stg( ptr, make_uint2( data.reg( 0 ), data.reg( 1 ) ), mem_desc );
            } else {
                xmma::stg( ptr, data.reg( 0 ), mem_desc );
            }
        }
    }

    // The dimension of the matrix.
    const int params_m_, params_n_, params_stride_m_;
    // The position of the tile.
    int m_, n_;
    // The pointer to matrix D in global memory.
    char *out_ptr_;
    // The pointer to matrix C in global memory.
    const char *res_ptr_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// row major output
template <typename Traits, typename Cta_tile, typename Fragment_c>
struct Gmem_tile_gmma_epilogue<Traits, Cta_tile, xmma::Row, Fragment_c> {
    // HGMMA operation, where A and B should be in fp16. 
    static_assert(sizeof(typename Traits::A_type) == 2 && sizeof(typename Traits::B_type) == 2,
      "HGMMA operation, where A and B should be in fp16 is required.\n");

    // The helper class to compute the row offset (in the tile).
    using Tile_distribution =
        xmma::helpers::Gmem_tile_gmma_epilogue_distribution<Traits, Cta_tile, xmma::Row, 16>;

    // Bytes per element
    enum { BYTES_PER_ELEMENT = 2 };
    // Bits per element
    enum { BITS_PER_ELEMENT = BYTES_PER_ELEMENT * 8 };
    // CTA tile size
    enum { CTA_M = Cta_tile::M, CTA_N = Cta_tile::N };
    // The number of threads
    enum { THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA };
    // the size for each STG.128
    enum { BYTES_PER_STG = 16 };
    // the number of elements for each stg.128
    enum { ELEMENTS_PER_STG = BYTES_PER_STG / BYTES_PER_ELEMENT };

    // tile_n if we can have 8 threads doing stg.128 unless cta_n is smaller than that
    enum { COLUMNS_PER_STG = BYTES_PER_STG * 8 / BYTES_PER_ELEMENT };
    enum { MIN_TILE_N = CTA_N < COLUMNS_PER_STG ? CTA_N : COLUMNS_PER_STG };
    // tile_m is limited such that every thread can participate, 8 rows per warp
    enum { TILE_M = 8 * THREADS_PER_CTA / 32, TILE_N = MIN_TILE_N };

    // Threads for STG per row
    enum { STG_THREADS_PER_ROW = TILE_N * BYTES_PER_ELEMENT / BYTES_PER_STG };
    static_assert( STG_THREADS_PER_ROW >= 8, "STG_THREADS_PER_ROW should be larger than 8\n" );
    // the number of rows can be store by all threads per STG instruction
    enum { ROWS_PER_STG = THREADS_PER_CTA / STG_THREADS_PER_ROW };
    // the number of rows per STG instruction by one warp.
    enum { ROWS_PER_STG_PER_WARP = Cta_tile::THREADS_PER_WARP / STG_THREADS_PER_ROW };
    // the min tile in N dim is 8 such that every thread can participate in sts
    static_assert( TILE_N % 8 == 0, "TILE_N should be multiple of 8" );
    // the number of inner iterations
    enum { STG_ITERATIONS_PER_TILE = TILE_M / ROWS_PER_STG };
    //
    enum { M_PER_WARP = 8 };
    // the number of inner iteration to cover a GMMA M. should always be 2.
    enum { STG_ITERATIONS_PER_GMMA_M = 16 / M_PER_WARP };

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_gmma_epilogue( const Params &params, int bidm, int bidn, int bidz, int tidx )
        : params_m_( params.m ), params_n_( params.n ), params_stride_n_( params.ldd ) {

        // Compute the output position for each thread.
        int row = Tile_distribution::compute_row( tidx );
        int col = Tile_distribution::compute_col( tidx );

        // Compute the output position for each thread.
        m_ = bidm * Cta_tile::M + row;
        n_ = bidn * Cta_tile::N + col * ELEMENTS_PER_STG;

        // The pointer.
        const int64_t offset = Traits::offset_in_bytes_c( m_ * params_stride_n_ + n_ );
        out_ptr_ = &reinterpret_cast<char *>( params.d_gmem )[offset];
        res_ptr_ = &reinterpret_cast<const char *>( params.c_gmem )[offset];
    }

    // store to gmem
    template <typename Fragment_pre_stg>
    inline __device__ void store( int mi, int ni, const Fragment_pre_stg &acc_pre_stg ) {
        const int offset = ni * TILE_N +
                           ( mi % STG_ITERATIONS_PER_GMMA_M ) * M_PER_WARP * params_stride_n_ +
                           ( mi / STG_ITERATIONS_PER_GMMA_M ) *
                               ( TILE_M * STG_ITERATIONS_PER_GMMA_M ) * params_stride_n_;

        char *ptr = &out_ptr_[Traits::offset_in_bytes_c( offset )];

        #pragma unroll
        for( int stg_idx = 0; stg_idx < STG_ITERATIONS_PER_TILE; ++stg_idx ) {
            ptr += stg_idx * ROWS_PER_STG_PER_WARP * params_stride_n_ * BYTES_PER_ELEMENT;
            int acc_idx = stg_idx * 4;
            uint4 tmp = make_uint4( acc_pre_stg.regs_[acc_idx],
                                    acc_pre_stg.regs_[acc_idx + 1],
                                    acc_pre_stg.regs_[acc_idx + 2],
                                    acc_pre_stg.regs_[acc_idx + 3] );
            // if( mask ) {
            xmma::stg( ptr, tmp );
            //}
        }
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask( int xmmas_mi, int mi, int ni, int ii ) {
        const int offset = Tile_distribution::compute_offset( xmmas_mi, mi, ii );
        return ( m_ + offset ) < params_m_ && ( n_ + ni * TILE_N ) < params_n_;
    }

    // Load the data from global memory.
    inline __device__ void load( Fragment_c &data,
                                 int xmmas_mi,
                                 int mi,
                                 int ni,
                                 int ii,
                                 int mask,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const int offset = Tile_distribution::compute_offset( xmmas_mi, mi, ii );
        const char *ptr =
            res_ptr_ + Traits::offset_in_bytes_c( offset * params_stride_n_ + ni * TILE_N );
        if( mask ) {
            if( BYTES_PER_STG == 16 ) {
                uint4 tmp;
                xmma::ldg( tmp, ptr, mem_desc );
                data.from_int4( tmp );
            } else if( BYTES_PER_STG == 8 ) {
                uint2 tmp;
                xmma::ldg( tmp, ptr, mem_desc );
                data.from_int2( tmp );
            } else {
                uint32_t tmp;
                xmma::ldg( tmp, ptr, mem_desc );
                data.reg( 0 ) = tmp;
            }
        }
    }

    // store to gmem
    inline __device__ void store( int xmmas_mi,
                                  int mi,
                                  int ni,
                                  int ii,
                                  const Fragment_c &data,
                                  int mask,
                                  uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const int offset = Tile_distribution::compute_offset( xmmas_mi, mi, ii );
        char *ptr = out_ptr_ + Traits::offset_in_bytes_c( offset * params_stride_n_ + ni * TILE_N );

        if( mask ) {
            if( BYTES_PER_STG == 16 ) {
                xmma::stg( ptr, data.to_int4(), mem_desc );
            } else if( BYTES_PER_STG == 8 ) {
                xmma::stg( ptr, make_uint2( data.reg( 0 ), data.reg( 1 ) ), mem_desc );
            } else {
                xmma::stg( ptr, data.reg( 0 ), mem_desc );
            }
        }
    }

    // The dimension of the matrix.
    const int params_m_, params_n_, params_stride_n_;
    // The position of the tile.
    int m_, n_;
    // The pointer to matrix D in global memory.
    char *out_ptr_;
    // The pointer to matrix C in global memory.
    const char *res_ptr_;
};

}  // namespace gemm
}  // namespace xmma
