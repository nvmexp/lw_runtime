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

#include <xmma/layout.h>
#include <xmma/utils.h>
#include <xmma/fragment.h>
#include <xmma/params.h>
#include <xmma/helpers/fragment.h>
#include <xmma/helpers/gemm.h>
#include <xmma/numeric_types.h>

namespace xmma {
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef USE_GMMA
namespace wip_do_not_use {
template <typename Traits,
          typename Cta_tile,
          typename Smem_tile_a,
          typename Smem_tile_b,
          bool GMMA_A_RF_,  // GMMA A operand coming from RF?
          bool GMMA_B_RF_,  // GMMA B operand coming from RF?
          int STAGES>
struct Compute_tile_with_gmma {};

/*
compute tile used when both operands are coming from SMEM
*/
template <typename Traits,
          typename Cta_tile,
          typename Smem_tile_a,
          typename Smem_tile_b,
          int STAGES>
struct Compute_tile_with_gmma<Traits,
                              Cta_tile,
                              Smem_tile_a,
                              Smem_tile_b,
                              false,  // GMMA A operand coming from SMEM
                              false,  // GMMA B operand coming from SMEM
                              STAGES> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // desc for A and B should have the same strategy
    static_assert( Smem_tile_a::Gmma_descriptor::GMMA_DESC_SIZE_PER_GROUP ==
                       Smem_tile_b::Gmma_descriptor::GMMA_DESC_SIZE_PER_GROUP,
                   "GMMA desc for A and B should have the same strategy." );

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M };
    enum { XMMAS_N = Xmma_tile::XMMAS_N };
    enum { XMMAS_K = Xmma_tile::XMMAS_K };

    // the number of GMMAs along N dim should be 1, we have multiple GMMAs along M and K dim.
    static_assert( XMMAS_N == 1,
                   "the number of GMMAs along N dim should be 1, at least for now, "
                   "all cases we test meet this requirement" );
    // Ctor.
    inline __device__ Compute_tile_with_gmma() {
    }

    // Ctor, that helps set the gmma descs
    inline __device__ Compute_tile_with_gmma( void *a_smem_, void *b_smem_ ) {
        uint32_t a_smem_lwvm_pointer = get_smem_pointer( a_smem_ );
        uint32_t b_smem_lwvm_pointer = get_smem_pointer( b_smem_ );

        #pragma unroll
        for( int xmma_m_idx = 0; xmma_m_idx < XMMAS_M; ++xmma_m_idx ) {
            gmma_desc_a_[xmma_m_idx].set_smem_pointer(
                a_smem_lwvm_pointer + xmma_m_idx * Smem_tile_a::GMMA_GROUP_SMEM_DISTANCE );
            gmma_desc_a_[xmma_m_idx].set_max_descriptor_0(
                Smem_tile_a::BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB );
        }

        #pragma unroll
        for( int xmma_n_idx = 0; xmma_n_idx < XMMAS_N; ++xmma_n_idx ) {
            gmma_desc_b_[xmma_n_idx].set_smem_pointer(
                b_smem_lwvm_pointer + xmma_n_idx * Smem_tile_b::GMMA_GROUP_SMEM_DISTANCE );
            gmma_desc_b_[xmma_n_idx].set_max_descriptor_0(
                Smem_tile_b::BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB );
        }
    }
    // Ctor, that helps set the gmma descs to support different buffer index as the start address.
    // The ctor without buffer index will assume the start address pointing to buffer_index = 0
    inline __device__ Compute_tile_with_gmma(void *a_smem_, void *b_smem_, int buffer_index, int BUFFERS_PER_TILE_) {
        uint32_t a_smem_lwvm_pointer = get_smem_pointer(a_smem_);
        uint32_t b_smem_lwvm_pointer = get_smem_pointer(b_smem_);
        
        #pragma unroll
        for( int xmma_m_idx = 0; xmma_m_idx < XMMAS_M; ++xmma_m_idx ) {
            gmma_desc_a_[xmma_m_idx].set_smem_pointer(a_smem_lwvm_pointer 
                + xmma_m_idx * Smem_tile_a::GMMA_GROUP_SMEM_DISTANCE);   
            gmma_desc_a_[xmma_m_idx].set_max_descriptor_0(
	      Smem_tile_a::BYTES_PER_BUFFER_NO_4LSB * (BUFFERS_PER_TILE_ - buffer_index -1));
        }
        
        #pragma unroll
        for( int xmma_n_idx = 0; xmma_n_idx < XMMAS_N; ++xmma_n_idx ) {
            gmma_desc_b_[xmma_n_idx].set_smem_pointer(b_smem_lwvm_pointer 
                + xmma_n_idx * Smem_tile_b::GMMA_GROUP_SMEM_DISTANCE);    
            gmma_desc_b_[xmma_n_idx].set_max_descriptor_0(
	      Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB * (BUFFERS_PER_TILE_ - buffer_index -1));
        }
    }
    
    //move the gmme desc by N buffers.
    inline __device__  void increment_N_gmma_desc_group(int N)
    {
        // #pragma unroll
        // for (int i=0; i< N;i++)
        // 	   increment_gmma_desc_group(); 
	//More efficient way of increment N group of gmma desc.
	//Instead of using a loop on incrementing 1 group for N time (N = the number of kblocks) 
	//by directly updating the descriptor once.
        #pragma unroll
        for( int idx = 0; idx < Smem_tile_a::Gmma_descriptor::NUM_DESCRIPTORS; ++idx ) {
        #pragma unroll
            for( int xmma_m_idx = 0; xmma_m_idx < XMMAS_M; ++xmma_m_idx ) {
                uint64_t temp_desc = gmma_desc_a_[xmma_m_idx].get_descriptor( idx );
	        uint64_t max_desc_a = gmma_desc_a_[xmma_m_idx].get_max_descriptor_0();
		uint64_t n_buffer_bytes = N * Smem_tile_a::BYTES_PER_BUFFER_NO_4LSB;
		uint64_t tot_buffer_size = Smem_tile_a::BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB
						 + Smem_tile_a::BYTES_PER_BUFFER_NO_4LSB;
		uint64_t size_in_tot_buffer= n_buffer_bytes % tot_buffer_size;
                temp_desc = (temp_desc + size_in_tot_buffer) < (max_desc_a + Smem_tile_a::BYTES_PER_BUFFER_NO_4LSB) 
			       ? (temp_desc + size_in_tot_buffer) 
			       : (temp_desc + size_in_tot_buffer - tot_buffer_size);
                gmma_desc_a_[xmma_m_idx].set_descriptor( idx, temp_desc );
            }

        #pragma unroll
            for( int xmma_n_idx = 0; xmma_n_idx < XMMAS_N; ++xmma_n_idx ) {
                uint64_t temp_desc = gmma_desc_b_[xmma_n_idx].get_descriptor( idx );
	        uint64_t max_desc_b = gmma_desc_b_[xmma_n_idx].get_max_descriptor_0();
		uint64_t n_buffer_bytes = N * Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
		uint64_t tot_buffer_size = Smem_tile_b::BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB 
					    + Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
		uint64_t size_in_tot_buffer= n_buffer_bytes % tot_buffer_size;
                temp_desc = (temp_desc + size_in_tot_buffer) < (max_desc_b + Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB) 
			       ? (temp_desc + size_in_tot_buffer) 
			       : (temp_desc + size_in_tot_buffer - tot_buffer_size);
                gmma_desc_b_[xmma_n_idx].set_descriptor( idx, temp_desc );
            }
        }
         
    }

    // Clear the aclwmulators. It does nothing as we have a special flag for GMMA.
    inline __device__ void clear() {
        helpers::clear( acc_ );
    }

    // smarter way of increment a group of gmma desc.
    // if one of them need to be reset to the first ldgsts buffer
    // it is very likely (lwrrently guaranteed) that all of them need to be reset to the first
    // ldgsts buffer.
    // we do this to save the usage of uniform register. Otherwise, kernel with larger M could not
    // achieve sol.
    inline __device__ void increment_gmma_desc_group() {
        bool reset_buffer = false;
        if( gmma_desc_a_[0].get_descriptor( 0 ) >= gmma_desc_a_[0].get_max_descriptor_0() ) {
            reset_buffer = true;
        }

        #pragma unroll
        for( int idx = 0; idx < Smem_tile_a::Gmma_descriptor::NUM_DESCRIPTORS; ++idx ) {
        #pragma unroll
            for( int xmma_m_idx = 0; xmma_m_idx < XMMAS_M; ++xmma_m_idx ) {
                uint64_t temp_desc = gmma_desc_a_[xmma_m_idx].get_descriptor( idx );
                // temp_desc += Smem_tile_a::BYTES_PER_BUFFER_NO_4LSB;
                temp_desc += ( reset_buffer == true )
                                 ? -Smem_tile_a::BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB
                                 : Smem_tile_a::BYTES_PER_BUFFER_NO_4LSB;
                gmma_desc_a_[xmma_m_idx].set_descriptor( idx, temp_desc );
            }

        #pragma unroll
            for( int xmma_n_idx = 0; xmma_n_idx < XMMAS_N; ++xmma_n_idx ) {
                uint64_t temp_desc = gmma_desc_b_[xmma_n_idx].get_descriptor( idx );
                // temp_desc += Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
                temp_desc += ( reset_buffer == true )
                                 ? -Smem_tile_b::BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB
                                 : Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
                gmma_desc_b_[xmma_n_idx].set_descriptor( idx, temp_desc );
            }
        }
    }

    // Compute.
    // last of group indicates it is the last GMMA with a GMMA group. So the GSB should be updated
    // last of kblock indicates it is the last GMMA with kblock. so desc will be updated accordingly
    inline __device__ void
    compute( int ki, bool last_of_group = false, bool last_of_kblock = false ) {
        #pragma unroll
        for( int xmmas_m_idx = 0; xmmas_m_idx < XMMAS_M; ++xmmas_m_idx ) {
        #pragma unroll
            for( int xmmas_n_idx = 0; xmmas_n_idx < XMMAS_N; ++xmmas_n_idx ) {
                // weird code to use SEL to avoid reg spill
                typename Smem_tile_a::Gmma_descriptor::Single_desc single_desc_a;
                typename Smem_tile_b::Gmma_descriptor::Single_desc single_desc_b;

                single_desc_a.set( gmma_desc_a_[xmmas_m_idx].get_descriptor( ki ) );
                single_desc_b.set( gmma_desc_b_[xmmas_n_idx].get_descriptor( ki ) );

                if( xmmas_n_idx == ( XMMAS_N - 1 ) ) {
                    // update desc for A
                    gmma_desc_a_[xmmas_m_idx].increment_single_descriptor( last_of_kblock );
                }
                if( xmmas_m_idx == ( XMMAS_M - 1 ) ) {
                    // update desc for B
                    gmma_desc_b_[xmmas_n_idx].increment_single_descriptor( last_of_kblock );
                }

                if( ( last_of_group == true ) && ( xmmas_m_idx == ( XMMAS_M - 1 ) ) &&
                    ( xmmas_n_idx == ( XMMAS_N - 1 ) ) ) {
                    // increment the scoreboard
                    acc_[xmmas_m_idx][xmmas_n_idx].hgmma( single_desc_a, single_desc_b, true );
                } else {
                    acc_[xmmas_m_idx][xmmas_n_idx].hgmma( single_desc_a, single_desc_b, false );
                }
            }  // for (xmmas_n_idx)
        }      // for (xmmas_m_idx)
    }

    // Load from shared memory. For GMMA where both operand comes from SMEM, this does nothing
    inline __device__ void
    load( Smem_tile_a &smem_a, Smem_tile_b &smem_b, int ki, bool first = false ) {
    }

    // The aclwmulators.
    Fragment_aclwmulator<Traits> acc_[XMMAS_M][XMMAS_N];

    // one descriptor group per stage, different GMMAs may or maynot share descriptor group
    // each descriptor group holds all the descriptors for the entire kblock

    // The descriptor to load A.
    typename Smem_tile_a::Gmma_descriptor gmma_desc_a_[XMMAS_M];
    // The descriptor to load B.
    typename Smem_tile_b::Gmma_descriptor gmma_desc_b_[XMMAS_N];
};

/*
compute tile used when A is from RF, B is from SMEM
*/
template <typename Traits,
          typename Cta_tile,
          typename Smem_tile_a,
          typename Smem_tile_b,
          int STAGES>
struct Compute_tile_with_gmma<Traits,
                              Cta_tile,
                              Smem_tile_a,
                              Smem_tile_b,
                              true,   // GMMA A operand coming from RF
                              false,  // GMMA A operand coming from SMEM
                              STAGES> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // desc for A and B should have the same strategy
    static_assert( Smem_tile_a::Gmma_descriptor::GMMA_DESC_SIZE_PER_GROUP ==
                       Smem_tile_b::Gmma_descriptor::GMMA_DESC_SIZE_PER_GROUP,
                   "GMMA desc for A and B should have the same strategy." );

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M };
    enum { XMMAS_N = Xmma_tile::XMMAS_N };
    enum { XMMAS_K = Xmma_tile::XMMAS_K };

    // the number of GMMAs along N dim should be 1, we have multiple GMMAs along M and K dim.
    static_assert( XMMAS_N == 1,
                   "the number of GMMAs along N dim should be 1, at least for now, "
                   "all cases we test meet this requirement" );
    // Ctor.
    inline __device__ Compute_tile_with_gmma() {
    }

    // Ctor, that helps set the gmma descs
    inline __device__ Compute_tile_with_gmma( void *a_smem_, void *b_smem_ ) {
        uint32_t a_smem_lwvm_pointer = get_smem_pointer( a_smem_ );
        uint32_t b_smem_lwvm_pointer = get_smem_pointer( b_smem_ );

        #pragma unroll
        for( int xmma_n_idx = 0; xmma_n_idx < XMMAS_N; ++xmma_n_idx ) {
            gmma_desc_b_[xmma_n_idx].set_smem_pointer(
                b_smem_lwvm_pointer + xmma_n_idx * Smem_tile_b::GMMA_GROUP_SMEM_DISTANCE );
            gmma_desc_b_[xmma_n_idx].set_max_descriptor_0(
                Smem_tile_b::BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB );
        }
    }

    // Clear the aclwmulators. It does nothing as we have a special flag for GMMA.
    inline __device__ void clear() {
        helpers::clear( acc_ );
    }

    // smarter way of increment a group of gmma desc.
    // if one of them need to be reset to the first ldgsts buffer
    // it is very likely (lwrrently guaranteed) that all of them need to be reset to the first
    // ldgsts buffer.
    // we do this to save the usage of uniform register. Otherwise, kernel with larger M could not
    // achieve sol.
    inline __device__ void increment_gmma_desc_group() {
        bool reset_buffer = false;
        if( gmma_desc_b_[0].get_descriptor( 0 ) >= gmma_desc_b_[0].get_max_descriptor_0() ) {
            reset_buffer = true;
        }

        #pragma unroll
        for( int idx = 0; idx < Smem_tile_a::Gmma_descriptor::NUM_DESCRIPTORS; ++idx ) {
        #pragma unroll
            for( int xmma_n_idx = 0; xmma_n_idx < XMMAS_N; ++xmma_n_idx ) {
                uint64_t temp_desc = gmma_desc_b_[xmma_n_idx].get_descriptor( idx );
                // temp_desc += Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
                temp_desc += ( reset_buffer == true )
                                 ? -Smem_tile_b::BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB
                                 : Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
                gmma_desc_b_[xmma_n_idx].set_descriptor( idx, temp_desc );
            }
        }
    }

    // Compute.
    // last of group indicates it is the last GMMA with a GMMA group. So the GSB should be updated
    // last of kblock indicates it is the last GMMA with kblock. so desc will be updated accordingly
    inline __device__ void
    compute( int ki, bool last_of_group = false, bool last_of_kblock = false ) {

        #pragma unroll
        for( int xmmas_m_idx = 0; xmmas_m_idx < XMMAS_M; ++xmmas_m_idx ) {
        #pragma unroll
            for( int xmmas_n_idx = 0; xmmas_n_idx < XMMAS_N; ++xmmas_n_idx ) {
                // weird code to use SEL to avoid reg spill
                typename Smem_tile_b::Gmma_descriptor::Single_desc single_desc_b;

                single_desc_b.set( gmma_desc_b_[xmmas_n_idx].get_descriptor( ki ) );

                if( xmmas_m_idx == ( XMMAS_M - 1 ) ) {
                    // update desc for B
                    gmma_desc_b_[xmmas_n_idx].increment_single_descriptor( last_of_kblock );
                }

                if( ( last_of_group == true ) && ( xmmas_m_idx == ( XMMAS_M - 1 ) ) &&
                    ( xmmas_n_idx == ( XMMAS_N - 1 ) ) ) {
                    // increment the scoreboard
                    acc_[xmmas_m_idx][xmmas_n_idx].hgmma(
                        a_[ki][xmmas_m_idx], single_desc_b, true );
                } else {
                    acc_[xmmas_m_idx][xmmas_n_idx].hgmma(
                        a_[ki][xmmas_m_idx], single_desc_b, false );
                }
            }  // for (xmmas_n_idx)
        }      // for (xmmas_m_idx)
    }

    // Load from shared memory.
    inline __device__ void
    load( Smem_tile_a &smem_a, Smem_tile_b &smem_b, int ki ) {
        smem_a.load( a_[ki], ki );
    }

    // The aclwmulators.
    Fragment_aclwmulator<Traits> acc_[XMMAS_M][XMMAS_N];

    // The fragments to load A.
    typename Smem_tile_a::Fragment a_[4][XMMAS_M];

    // one descriptor group per stage, different GMMAs may or maynot share descriptor group
    // each descriptor group holds all the descriptors for the entire kblock

    // The descriptor to load B.
    typename Smem_tile_b::Gmma_descriptor gmma_desc_b_[XMMAS_N];
};

}  // namespace wip_do_not_use
#endif  // USE_GMMA
////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace xmma
