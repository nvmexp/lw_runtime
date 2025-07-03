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
#include <xmma/implicit_gemm/dgrad/gmem_tile.h>

namespace xmma {
namespace implicit_gemm {
namespace dgrad_indexed {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   A
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Input_related, int BYTES_PER_LDG_,
    typename Layout, int M, int N>
struct Gmem_tile_base_a
    : public dgrad::Gmem_tile_base_a<Traits, Cta_tile, Input_related, BYTES_PER_LDG_,
        Layout, M, N> {

    // Idx kernels don't need the Input_related
    static_assert( Input_related::STATIC_FILTER_SIZE == 0, "Input_related::STATIC_FILTER_SIZE==0" );
    using Gmem_layout = Layout;

    // The base class.
    using Base_ = dgrad::Gmem_tile_base_a<Traits, Cta_tile, Input_related, BYTES_PER_LDG_,
        Layout, M, N>;

    // We store the coordinates of the pixels loaded by that tile in shared memory.
    enum { BYTES_PER_EXTRA_SMEM = Cta_tile::M * sizeof( int4 ) };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_a( const Params& params,
                                        void* smem,
                                        int bidz )
        : Base_( params, nullptr, bidz ), params_o_( params.o ), params_p_( params.p ),
          params_q_( params.q ), params_rs_( params.rs ), params_s_( params.s ),
          params_dilation_t_( params.dilation[0] ), params_dilation_r_( params.dilation[1] ),
          params_dilation_s_( params.dilation[2] ), params_delta_( &params.a_delta[0] ),
          params_mul_rs_( params.mul_rs ), params_shr_rs_( params.shr_rs ),
          params_mul_s_( params.mul_s ), params_shr_s_( params.shr_s ),
          smem_( get_smem_pointer( smem ) ) {
    }

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_a( const Params& params,
                                        void* smem,
                                        const dim3& bidx,
                                        int tidx )
        : Gmem_tile_base_a( params, smem, bidx.z ) {

        // We first compute the coordinates of all the pixels loaded by this CTA.
        const int STEPS = Div_up<Cta_tile::M, Cta_tile::THREADS_PER_CTA>::VALUE;
        for( int ii = 0; ii < STEPS; ++ii ) {

            // The linear index of the pixel computed by that thread inside the CTA.
            int idx_in_cta = tidx + ii * Cta_tile::THREADS_PER_CTA;

            // Decompose the linear index into its corresponding 4D position.
            int n, dhw;
            xmma::fast_divmod( n, dhw, bidx.x * Cta_tile::M + idx_in_cta, params.dhw,
                                   params.mul_dhw, params.shr_dhw );
            int d, hw;
            xmma::fast_divmod( d, hw, dhw, params.hw, params.mul_hw, params.shr_hw );
            int h, w;
            xmma::fast_divmod( h, w, hw, params.w, params.mul_w, params.shr_w );

            // Apply strides and padding.
            int o = d + params.pad[0][0];
            int p = h + params.pad[1][0];
            int q = w + params.pad[2][0];

            // Store the valid pixels.
            if( tidx + ii * Cta_tile::THREADS_PER_CTA < Cta_tile::M ) {
                uint32_t ptr = this->smem_ + idx_in_cta * sizeof( int4 );
                int4 indices = make_int4( n, o, p, q );
                sts( ptr, reinterpret_cast<const uint4&>( indices ) );
            }
        }

        // Make sure the values are in shared memory.
        __syncthreads();

        // The position in the K dimension.
        int first_k_in_cta = bidx.z * this->params_split_k_k_;
        int k, k_base = 0;

        // Determine the 1st pixel loaded by this thread inside the CTA.
        if (Gmem_layout::ROW) {
            first_k_in_cta += tidx % Base_::THREADS_PER_PIXEL * Base_::ELTS_PER_LDG;
            first_idx_in_cta_ = tidx / Base_::THREADS_PER_PIXEL;
        } else {
            first_k_in_cta += tidx / Base_::THREADS_PER_PIXEL;
            first_idx_in_cta_ = tidx % Base_::THREADS_PER_PIXEL * Base_::ELTS_PER_LDG;
        }

        if( params.g > 1 ) {
            k_base = bidx.y * Cta_tile::N;
        }

        // Initialize the offsets and predicates. Since N/K do not change, we pack them.
        uint32_t preds_nk[Base_::LDGS], preds_opq[Base_::LDGS];
#pragma unroll
        for( int mi = 0; mi < Base_::LDGS; ++mi ) {

            // Read the coordinates from shared memory.
            int idx_in_cta = first_idx_in_cta_;
            int k_in_cta;
            if (Gmem_layout::ROW) {
                idx_in_cta += (mi / Base_::LDGS_UNROLL) * Base_::PIXELS_PER_LDG;
                k_in_cta = first_k_in_cta + mi % Base_::LDGS_UNROLL;   
            } else {
                idx_in_cta += mi % Base_::LDGS_UNROLL;
                k_in_cta = first_k_in_cta + (mi / Base_::LDGS_UNROLL) * Base_::PIXELS_PER_LDG;
            }

            k = k_base + k_in_cta;

            // Load the coordinates of the pixel loaded by this thread with the mi-th ldg.
            uint4 pix;
            lds( pix, this->smem_ + idx_in_cta * sizeof( uint4 ) );

            // Extract the coordinates of that pixel.
            int n = reinterpret_cast<const int&>( pix.x );
            int o = reinterpret_cast<const int&>( pix.y );
            int p = reinterpret_cast<const int&>( pix.z );
            int q = reinterpret_cast<const int&>( pix.w );

            // Compute the offsets for the load.
            this->offsets_[mi] = n * params.out_stride_n +
                                 o * params.out_stride_d +
                                 p * params.out_stride_h +
                                 q * params.out_stride_w +
                                 k * params.out_stride_c;

            // Compute the mask for N/K.
            preds_nk[mi] = n < params.n && k < this->params_k_;

            if ( Cta_tile::N < Cta_tile::K && params.g > 1 ) {
                preds_nk[mi] &= ( k_in_cta < Cta_tile::N );
            }

            // Compute the predicate for O/P/Q. They change at each iteration.
            preds_opq[mi] =
                (unsigned)o < params.o && (unsigned)p < params.p && (unsigned)q < params.q;
        }

        // Pack the predicates for N/C. They are constant.
        uint32_t tmp[Base_::PRED_REGS];
        xmma::pack_predicates( tmp, preds_nk );
        #pragma unroll
        for (int ii = 0; ii < Base_::PRED_REGS; ii++) {
            this->masks_[ii] = tmp[ii];
        }

        // Compute the real predicates for the load.
        xmma::pack_predicates( this->preds_, preds_opq );
        #pragma unroll
        for (int ii = 0; ii < Base_::PRED_REGS; ii++) {
            this->preds_[ii] &= this->masks_[ii];
        }
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move( int next_trsi, int64_t ) {

        // Extract T, R and S from the linear filter position.
        int t, rs;
        xmma::fast_divmod( t, rs, next_trsi, params_rs_, params_mul_rs_, params_shr_rs_ );
        int r, s;
        xmma::fast_divmod( r, s, rs, params_s_, params_mul_s_, params_shr_s_ );

        // Determine by how much we are moving. It depends on the position of the filter.
        int delta_offset = 0;

        // If we move to a new channel/feature map...
        if( next_trsi == 0 ) {
            delta_offset = 3;
        }
        // If we move to a new slice (in a 3D lwbe)...
        else if( rs == 0 ) {
            delta_offset = 2;
        }
        // If we move to a new row...
        else if( s == 0 ) {
            delta_offset = 1;
        }

        // Update the pointer.
        this->ptr_ += params_delta_[delta_offset];

        // The dilated filter.
        int dilated_t = t * params_dilation_t_;
        int dilated_r = r * params_dilation_r_;
        int dilated_s = s * params_dilation_s_;

        // Update the predicates for the different loads.
        uint32_t preds[Base_::LDGS];
#pragma unroll
        for( int mi = 0; mi < Base_::LDGS; ++mi ) {

            // Read the coordinates from shared memory.
            int idx_in_cta = first_idx_in_cta_;
            if (Gmem_layout::ROW) {
                idx_in_cta += (mi / Base_::LDGS_UNROLL) * Base_::PIXELS_PER_LDG;
            } else {
                idx_in_cta += mi % Base_::LDGS_UNROLL;
            }

            // Load the coordinates of the pixel loaded by this thread with the mi-th ldg.
            uint4 pix;
            lds( pix, this->smem_ + idx_in_cta * sizeof( uint4 ) );

            // Extract the coordinates of that pixel.
            int o = reinterpret_cast<const int&>( pix.y ) - dilated_t;
            int p = reinterpret_cast<const int&>( pix.z ) - dilated_r;
            int q = reinterpret_cast<const int&>( pix.w ) - dilated_s;

            // Compute the predicate for D/H/W. They change at each iteration.
            preds[mi] =
                (unsigned)o < params_o_ && (unsigned)p < params_p_ && (unsigned)q < params_q_;
        }

        // Pack the predicates.
        xmma::pack_predicates( this->preds_, preds );
        #pragma unroll
        for (int ii = 0; ii < Base_::PRED_REGS; ii++) {
            this->preds_[ii] &= this->masks_[ii];
        }
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        Base_::residue( Base_::PRED_REGS );
    }

    // The dimensions of the image.
    const int params_o_, params_p_, params_q_;
    // The dimensions of the filter.
    const int params_rs_, params_s_;
    // The dilation factors.
    const int params_dilation_t_, params_dilation_r_, params_dilation_s_;
    // The delta table in constant memory.
    const int64_t* params_delta_;
    // The precomputed values for faster divmod.
    const uint32_t params_mul_rs_, params_shr_rs_, params_mul_s_, params_shr_s_;
    // The shared memory pointer.
    uint32_t smem_;
    // The position of the 1st pixel loaded by this thread in the CTA.
    int first_idx_in_cta_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // Template parameters which are related to input parameters like filter size.
    typename Input_related,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG_ = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_a<Traits, Cta_tile, Input_related, BYTES_PER_LDG_,
        xmma::Row, Cta_tile::K, Cta_tile::M>>
struct Gmem_tile_a_t : public dgrad::Gmem_tile_a_t<Traits, Cta_tile, Input_related, BYTES_PER_LDG_,
                                               DISABLE_LDGSTS, Ancestor> {

    // Idx kernels don't need the Input_related
    static_assert( Input_related::STATIC_FILTER_SIZE == 0, "Input_related::STATIC_FILTER_SIZE==0" );

    // The base class.
    using Base_ = dgrad::Gmem_tile_a_t<Traits, Cta_tile, Input_related, BYTES_PER_LDG_, DISABLE_LDGSTS,
                                     Ancestor>;

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a_t( const Params& params,
                                     void* smem,
                                     const dim3& bidx,
                                     int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // Template parameters which are related to input parameters like filter size.
    typename Input_related,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG_ = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_a<Traits, Cta_tile, Input_related, BYTES_PER_LDG_,
        xmma::Col, Cta_tile::M, Cta_tile::K>>
struct Gmem_tile_a_n : public dgrad::Gmem_tile_a_n<Traits, Cta_tile, Input_related, BYTES_PER_LDG_,
                                               DISABLE_LDGSTS, Ancestor> {

    // Idx kernels don't need the Input_related
    static_assert( Input_related::STATIC_FILTER_SIZE == 0, "Input_related::STATIC_FILTER_SIZE==0" );

    // The base class.
    using Base_ = dgrad::Gmem_tile_a_n<Traits, Cta_tile, Input_related, BYTES_PER_LDG_, DISABLE_LDGSTS,
                                     Ancestor>;

    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a_n( const Params& params,
                                     void* smem,
                                     const dim3& bidx,
                                     int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   C
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bytes per STG.
    int BYTES_PER_STG,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>
>
struct Gmem_tile_c_t
    : public xmma::implicit_gemm::dgrad::Gmem_tile_epilogue_base<Traits,
        Cta_tile, xmma::Row, BYTES_PER_STG, Fragment_c> {
    using Base = xmma::implicit_gemm::dgrad::Gmem_tile_epilogue_base<Traits,
        Cta_tile, xmma::Row, BYTES_PER_STG, Fragment_c>;

    template <typename Params>
    inline __device__ Gmem_tile_c_t( const Params& params,
                                     int bidm,
                                     int bidn,
                                     int bidz,
                                     int tidx )
       : Base ( params, bidm, bidn, bidz, tidx ) {
    }

};

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bytes per STG.
    int BYTES_PER_STG,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>
>
struct Gmem_tile_c_n
    : public xmma::implicit_gemm::dgrad::Gmem_tile_epilogue_base<Traits,
        Cta_tile, xmma::Col, BYTES_PER_STG, Fragment_c> {
    using Base = xmma::implicit_gemm::dgrad::Gmem_tile_epilogue_base<Traits,
        Cta_tile, xmma::Col, BYTES_PER_STG, Fragment_c>;

    template <typename Params>
    inline __device__ Gmem_tile_c_n( const Params& params,
                                     int bidm,
                                     int bidn,
                                     int bidz,
                                     int tidx )
       : Base ( params, bidm, bidn, bidz, tidx ) {
    }

};
///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace dgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma
