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

//#include <xmma/fragment_ampere.h>
//#include <xmma/warp_masks.h>
#include <xmma/smem_tile.h>
#include <xmma/hopper/gmma_descriptor.h>
#include <xmma/turing/smem_tile.h>
#include <xmma/ampere/smem_tile.h>
#include <xmma/arrive_wait.h>

namespace xmma {
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// x M M A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N_IN_BITS> struct Rows_per_xor_pattern_hopper {
    enum { VALUE = N_IN_BITS == 256 ? 2 : ( N_IN_BITS == 512 ? 4 : 8 ) };
};

template <typename Traits, int N> struct Rows_per_xor_pattern_hopper_a {
    enum { N_IN_BITS = N * Traits::BITS_PER_ELEMENT_A };
    enum { VALUE = N_IN_BITS <= 256 ? 2 : ( N_IN_BITS <= 512 ? 4 : 8 ) };
};

template <typename Traits> struct Cols_per_xor_pattern_hopper {
    enum { VALUE = 1 };
};

template <typename Input_type, typename Output_type>
struct Cols_per_xor_pattern_hopper<Hopper_hmma_tf32_traits<Input_type, Output_type>> {
    enum { VALUE = 2 };
};

template <typename Traits, int N>
struct Rows_per_xor_pattern_hopper_col_a : public Rows_per_xor_pattern_hopper_a<Traits, N> {};

template <typename Input_type, typename Output_type, int N>
struct Rows_per_xor_pattern_hopper_col_a<Hopper_hmma_tf32_traits<Input_type, Output_type>, N> {
    enum { VALUE = 4 };
};

template< 
    // The description of the tile computed by this CTA.
    typename Cta_tile, 
    // The number of rows in the 2D shared memory buffer.
    int M_, 
    // The number of cols.
    int N_, 
    // The size in bits of each element.
    int BITS_PER_ELEMENT_, 
    // The number of bytes per STS.
    int BYTES_PER_STS_ = 16,
    // The number of buffers. (Used in multistage and double buffer cases.)
    int BUFFERS_PER_TILE_ = 1,
    // Do we enable the fast path for LDS.128 and friends.
    int ENABLE_LDS_FAST_PATH_ = 0, 
    // The number of rows that are used for the XOR swizzling to allow fast STS/LDS. 
    int ROWS_PER_XOR_PATTERN_ = 8,
    // The number of cols that are used for the XOR swizzling to allow fast STS/LDS. 
    int COLS_PER_XOR_PATTERN_ = 1,
    // Use or not predicates
    bool USE_PREDICATES_ = true
>
struct Smem_tile_hopper_tma_base {
    // The size in bits of each element.
    enum { BITS_PER_ELEMENT = BITS_PER_ELEMENT_ };
    // The size in bytes of a single STS.
    enum { BYTES_PER_STS = BYTES_PER_STS_ };
    // The number of elements per STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // To support arbitrary N, we pad some values to a power-of-2.
    enum { N_WITH_PADDING = Next_power_of_two<N_>::VALUE }; 
    // The number of bytes per row without packing of rows.
    enum { BYTES_PER_ROW_BEFORE_PACKING = N_WITH_PADDING * BITS_PER_ELEMENT / 8 };
    // The number of bytes per row -- we want at least 128B per row.
    enum { BYTES_PER_ROW = Max<BYTES_PER_ROW_BEFORE_PACKING, 128>::VALUE };
    // The number of rows in shared memory (two rows may be packed into a single one).
    enum { ROWS = M_ * BYTES_PER_ROW_BEFORE_PACKING / BYTES_PER_ROW };

    // The number of threads per row.
    enum { THREADS_PER_ROW_UNBOUNDED = BYTES_PER_ROW / BYTES_PER_STS };
    // The number of threads per row.
    enum { THREADS_PER_ROW = Min<Cta_tile::THREADS_PER_CTA, THREADS_PER_ROW_UNBOUNDED>::VALUE };

    // The number of STS per row.
    enum { STS_PER_ROW = BYTES_PER_ROW / THREADS_PER_ROW / BYTES_PER_STS };
    // It must be at least one.
    static_assert(STS_PER_ROW >= 1, "");
    // The number of rows written with a single STS.
    enum { ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // Make sure we write to at least one row per STS. Thanks Dr. Obvious ;)
    static_assert(ROWS_PER_STS >= 1, "");
    // The number of STS needed to store all rows.
    enum { STS_PER_COL = Div_up<ROWS, ROWS_PER_STS>::VALUE };
    // The number of STS in total.
    enum { STS = STS_PER_COL * STS_PER_ROW };

    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = STS * BYTES_PER_STS * Cta_tile::THREADS_PER_CTA };
    // The number of buffers. 
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // The size in bytes of total buffers.
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE };
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER };

    // Do we enable the LDS.128 fast path?
    enum { ENABLE_LDS_FAST_PATH = ENABLE_LDS_FAST_PATH_ };
    // The number of rows that are used for the XOR swizzling to allow fast STS/LDS. 
    enum { ROWS_PER_XOR_PATTERN = ROWS_PER_XOR_PATTERN_ };
    // The number of cols that are used for the XOR swizzling to allow fast STS/LDS. 
    enum { COLS_PER_XOR_PATTERN = COLS_PER_XOR_PATTERN_ * 16 / BYTES_PER_STS };
    // Use or not predicates
    enum { USE_PREDICATES = USE_PREDICATES_ };

    // The type of elements that are stored in shared memory by each thread.
    using Store_type = typename Uint_from_size_in_bytes<BYTES_PER_STS>::Type;

    inline __device__ Smem_tile_hopper_tma_base( void *smem, int tidx )
        : smem_( get_smem_pointer( smem ) ) {

        // The row written by a thread. See doc/xmma_smem_layout.xlsx.
        int smem_write_row = tidx / THREADS_PER_ROW;

        // Swizzle the rows 4..7 and 8..11 if we use the LDS fast path for that tile.
        if( ENABLE_LDS_FAST_PATH ) {
            smem_write_row = (smem_write_row & 0x08)/2 + (smem_write_row & 0x04)*2;
        }
        
        // The XOR pattern.
        int smem_write_xor = smem_write_row % ROWS_PER_XOR_PATTERN * COLS_PER_XOR_PATTERN;
        // Compute the column and apply the XOR pattern.
        int smem_write_col = (tidx % THREADS_PER_ROW) ^ smem_write_xor;

        // The offset.
        this->smem_write_offset_ = smem_write_row*BYTES_PER_ROW + smem_write_col*BYTES_PER_STS;

        // TODO: Why not merge it with the read offset?
        this->smem_read_buffer_ = __shfl_sync(0xffffffff, 0, 0);
        this->smem_write_buffer_ = __shfl_sync(0xffffffff, 0, 0);
        this->smem_barrier_offset_ = 0;
    }

    inline __device__ void add_smem_barrier_base( uint64_t *smem_barrier ) {
        this->smem_barrier_ = static_cast<uint32_t>( __cvta_generic_to_shared( smem_barrier ) );
    }

    // Compute the store pointers.
    template< int N, int K = 1 >
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int ii = 0; ii < N / K; ++ii ) {
            // Decompose the STS into row/col.
            int row = ii / STS_PER_ROW;
            int col = ii % STS_PER_ROW;

            // Compute the immediate.
            int imm;
            if( ENABLE_LDS_FAST_PATH && ROWS_PER_STS == 1 ) {
                // i =  0 -> row  0 || i =  4 -> row  8 || i =  8 -> row  4 || i = 12 -> row 12
                // i =  1 -> row  1 || i =  5 -> row  9 || i =  9 -> row  5 || i = 13 -> row 13
                // i =  2 -> row  2 || i =  6 -> row 10 || i = 10 -> row  6 || i = 14 -> row 14
                // i =  3 -> row  3 || i =  7 -> row 11 || i = 11 -> row  7 || i = 15 -> row 15
                //
                // i = 16 -> row 16 || ...
                imm = (row & 0xf3) + (row & 0x08)/2 + (row & 0x04)*2;
            } else if( ENABLE_LDS_FAST_PATH && ROWS_PER_STS == 2 ) {
                // i =  0 -> row  0 || i =  2 -> row  8 || i =  4 -> row  4 || i =  6 -> row 12 
                // i =  1 -> row  2 || i =  3 -> row 10 || i =  5 -> row  6 || i =  7 -> row 14 
                //
                // i =  8 -> row 16 || ...
                imm = (row & 0xf9)*2 + (row & 0x04) + (row & 0x02)*4;
            } else if( ENABLE_LDS_FAST_PATH && ROWS_PER_STS == 4 ) {
                // i =  0 -> row  0 || i =  1 -> row  8 || i =  2 -> row  4 || i =  3 -> row 12 
                //
                // i =  4 -> row 16 || ...
                imm = (row & 0xfc)*4 + (row & 0x02)*2 + (row & 0x01)*8;
            } else if( ENABLE_LDS_FAST_PATH && ROWS_PER_STS == 8 ) {
                // i =  0 -> row  0 || i =  1 -> row  4  
                //
                // i =  2 -> row 16 || ...
                imm = (row & 0xfe)*8 + (row & 0x01)*4;
            } else {
                imm = (row);
            }

            // Assemble the offset.
            int offset = smem_write_offset_ + imm*ROWS_PER_STS*BYTES_PER_ROW;

            // Take the column into account.
            if( STS_PER_ROW > 1 ) {
                offset += col*THREADS_PER_ROW*BYTES_PER_STS; 
            }

            // Apply the XOR pattern if needed.
            if( ROWS_PER_STS < ROWS_PER_XOR_PATTERN ) {
                const int m = row * ROWS_PER_STS % ROWS_PER_XOR_PATTERN;
                offset ^= m * COLS_PER_XOR_PATTERN * BYTES_PER_STS;
            }

            // Assemble the final pointer :)
            #pragma unroll
            for (int k = 0; k < K; k++) {
                ptrs[ii*K + k] = smem_ + offset + k * 4 + smem_write_buffer_;
            }
        }
    }

    inline __device__ void debug_reset() {
        for( int buffer = 0; buffer < BYTES_PER_TILE; buffer += BYTES_PER_BUFFER) {
        for( int row = 0; row < ROWS; ++row ) {
            for( int col = 0; col < BYTES_PER_ROW; col += 4 ) {
                if( threadIdx.x == 0 ) {
                    uint32_t val = 0x0;
                    sts(val, smem_ + row*BYTES_PER_ROW + col + buffer);
                }
            }
        }
        }
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int buffer = 0; buffer < BYTES_PER_TILE; buffer += BYTES_PER_BUFFER) {
        for( int row = 0; row < ROWS; ++row ) {
            for( int col = 0; col < BYTES_PER_ROW; col += 4 ) {
                if( threadIdx.x == 0 ) {
                    uint32_t val;
                    lds(val, smem_ + row*BYTES_PER_ROW + col + buffer);
                    printf("block=(x=%2d, y=%2d, z=%2d) (smem_=%2d, buffer=%2d, row=%2d, byte=%4d)=0x%08x\n",
                        blockIdx.x,
                        blockIdx.y,
                        blockIdx.z,
                        smem_,
                        buffer,
                        row,
                        col,
                        val);
                }
            }
        }
        }
    }

    // Move the read offset to next buffer.
    inline __device__ void move_to_next_read_buffer() {
        if( BUFFERS_PER_TILE > 1 && smem_read_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
            this->smem_read_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_read_buffer_ += BYTES_PER_BUFFER;
        }
    }

    // Move the read offset to next buffer. TODO: Remove this member function!!!
    inline __device__ void move_next_read_buffer() {
        this->move_to_next_read_buffer();
    }

    // Move the read offset to next N buffer (cirlwlar-buffer).
    inline __device__ void move_to_next_read_buffer(int N) {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_read_buffer_ += N * BYTES_PER_BUFFER;
            this->smem_read_buffer_ -= smem_read_buffer_ >= BYTES_PER_TILE ? BYTES_PER_TILE : 0;
        }
    }

    // Move the read offset to next N buffer (cirlwlar-buffer). TODO: Remove this member function!!!
    inline __device__ void move_next_read_buffer(int N) {
        this->move_to_next_read_buffer(N);
    }

    // Move the write offset to next buffer.
    inline __device__ void move_to_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 && smem_write_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
            this->smem_write_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_buffer_ += BYTES_PER_BUFFER;
        }
        this->smem_barrier_offset_ +=
            this->smem_barrier_offset_ >= BUFFERS_PER_TILE * 8 ? -BUFFERS_PER_TILE * 8 : 8;
    }

    // Move the write offset to next buffer. TODO: Remove that member function!
    inline __device__ void move_next_write_buffer() {
        this->move_to_next_write_buffer();
    }

    inline __device__ void move_next_write_buffer( int buffer_id ) {
        this->move_next_write_buffer( buffer_id );
    }

    // Move the read offset.
    inline __device__ void move_read_offset(int delta) {
        this->smem_read_offset_ += delta;
    }

    /**
     * \brief load tensor blocks from global memory and stores to shared memory using tma instructions
     * 
     * \param p_desc pointer to tma descriptor masked as const void* pointer
     * \param smem_offset shared memory offset in bytes relative to smem_write_buffer_
     * \param coord0 tensor access coordinate in dimension 1, used by tma load
     * \param coord1 tensor access coordinate in dimension 2, used by tma load
     * \param coord2 tensor access coordinate in dimension 3, used by tma load
     * \param coord3 tensor access coordinate in dimension 4, used by tma load
     * \param coord4 tensor access coordinate in dimension 5, used by tma load
     * \param filter_offsets encodes multicast cta id and filter offsets
     */
    template <uint32_t DIM, lwdaTmaDescType DESC_TYPE, unsigned COPY_BYTES, int USE_TMA_MULTICAST = 0>
    inline __device__ void store( const void *p_desc,
                                  const unsigned &smem_offset,
                                  int32_t coord0,
                                  int32_t coord1,
                                  int32_t coord2,
                                  int32_t coord3,
                                  int32_t coord4,
                                  uint32_t off_w = 0,
                                  uint32_t off_h = 0,
                                  uint32_t off_d = 0,
                                  uint16_t mcast_bitmask = 0 ) {
        if( threadIdx.x == 0 ) {
            unsigned smem = xmma::hopper::emu::set_shared_data_address(
                static_cast<uint32_t>( __cvta_generic_to_shared( this->smem_ ) ) +
                this->smem_write_offset_ + smem_offset );

            xmma::utmaldg<DIM, DESC_TYPE, USE_TMA_MULTICAST>( reinterpret_cast<const lwdaTmaDescv2 *>( p_desc ),
                                           smem,
                                           unsigned( this->smem_barrier_ + this->smem_barrier_offset_ ),
                                           coord0,
                                           coord1,
                                           coord2,
                                           coord3,
                                           coord4,
                                           off_w,
                                           off_h,
                                           off_d,
                                           mcast_bitmask );
        }
    }

    unsigned smem_;
    int smem_read_offset_;
    int smem_read_buffer_;
    // Needed for zeroing smem for group colwolutions
    int smem_write_offset_;
    int smem_write_buffer_;
    uint32_t smem_barrier_;
    uint32_t smem_barrier_offset_;
};

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_hopper_col_a<Traits, Cta_tile::M>::VALUE,
    // How many cols to use for the XOR pattern to avoid bank conflicts?
    int COLS_PER_XOR_PATTERN_ = Cols_per_xor_pattern_hopper<Traits>::VALUE>
struct Smem_tile_hopper_tma_col_a : public Smem_tile_hopper_tma_base<Cta_tile,
                                         Cta_tile::K,
                                         Cta_tile::M,
                                         Traits::BITS_PER_ELEMENT_A,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         COLS_PER_XOR_PATTERN_>
 {
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_hopper_tma_base<Cta_tile,
                                         Cta_tile::K,
                                         Cta_tile::M,
                                         Traits::BITS_PER_ELEMENT_A,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         COLS_PER_XOR_PATTERN_>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Col>;

    // Can we use LDSM? No if the data type is 32-bit large.
    enum { USE_LDSMT = Traits::BITS_PER_ELEMENT_A == 16 };
    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = USE_LDSMT ? 16 : 4 };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_A };

    enum { BYTES_CACHELINE = 128 };

    // Ctor.
    inline __device__ Smem_tile_hopper_tma_col_a( void *smem, int tidx ) : Base( smem, tidx ) {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M = 1 * 1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row/col read by the thread.
        int smem_read_row, smem_read_col;

        static_assert( ( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 8 ) ||
                           Base::ROWS_PER_XOR_PATTERN == 4 || Base::ROWS_PER_XOR_PATTERN == 2,
                       "" );

        if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 8 ) {
            smem_read_row = ( tidx & WARP_MASK_K ) / WARP_DIV_K * Xmma_tile::XMMAS_K * 16 +
                            ( tidx & 0x10 ) / 2 + ( tidx & 0x07 );
            smem_read_col = ( tidx & 0x07 );
        } else if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = ( tidx & WARP_MASK_K ) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 +
                            ( tidx & 0x10 ) / 4 + ( tidx & 0x06 ) / 2;
            smem_read_col = ( tidx & 0x01 ) * 4 + ( tidx & 0x06 ) / 2;
        } else if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = ( tidx & WARP_MASK_K ) / WARP_DIV_K * Xmma_tile::XMMAS_K * 4 +
                            ( tidx & 0x10 ) / 8 + ( tidx & 0x04 ) / 4;
            smem_read_col = ( tidx & 0x03 ) * 2 + ( tidx & 0x04 ) / 4;
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 && Base::COLS_PER_XOR_PATTERN == 2 ) {
            smem_read_row =
                ( tidx & WARP_MASK_K ) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 + ( tidx & 0x03 );
            smem_read_col = ( tidx & 0x1c ) / 4 + ( tidx & 0x03 ) * 8;
        }

        // Swizzle the column for other warps.
        if( USE_LDSMT ) {
            smem_read_col ^= ( tidx & WARP_MASK_M ) / WARP_DIV_M * 2 + ( tidx & 0x08 ) / 8;
        } else {
            smem_read_col ^= ( tidx & WARP_MASK_M ) / WARP_DIV_M * 16;
        }

        // The shared memory offset.
        // this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW +
        // smem_read_col*BYTES_PER_LDS;
        this->smem_read_offset_ = smem_read_row * BYTES_CACHELINE + smem_read_col * BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset( int ki = 0 ) {

        static_assert( Cta_tile::WARPS_M == 4 || Cta_tile::WARPS_M == 2 ||
                           Xmma_tile::XMMAS_M == 4 || Xmma_tile::XMMAS_M == 2 ||
                           Xmma_tile::XMMAS_M == 1,
                       "" );

        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( Cta_tile::WARPS_M == 4 ) {
                // Nothing to do!
            } else if( Cta_tile::WARPS_M == 2 && Xmma_tile::XMMAS_M > 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * ( 4 );
            } else if( Cta_tile::WARPS_M == 2 ) {
                // Nothing to do!
            } else if( Xmma_tile::XMMAS_M == 4 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * ( mi % 2 == 0 ? 2 : 6 );
            } else if( Xmma_tile::XMMAS_M == 2 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            }
        }
    }

    // Load from shared memory.
    inline __device__ void load( Fragment ( &a )[Xmma_tile::XMMAS_M], int ki ) {
        // The size of each element in bits.
        const int BITS_PER_ELT = Traits::BITS_PER_ELEMENT_A;
        // The size in bytes of the data needed to compute an XMMA per CTA.
        const int BYTES_PER_XMMA_PER_CTA = Xmma_tile::M_PER_XMMA_PER_CTA * BITS_PER_ELT / 8;

        // Perform the different loads.
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Prepare the offset.
            int offset =
                ki * Base::ROWS_PER_XOR_PATTERN * 2 * BYTES_CACHELINE;  // Base::BYTES_PER_ROW;
            if( BYTES_PER_XMMA_PER_CTA == 32 ) {
                offset += this->smem_read_offset_;
            } else if( BYTES_PER_XMMA_PER_CTA == 64 ) {
                offset +=
                    this->smem_read_offset_ +
                    ( mi / 2 ) * Cta_tile::K * BYTES_CACHELINE;  // BYTES_PER_XMMA_PER_CTA * 2;
            } else if( BYTES_PER_XMMA_PER_CTA >= 128 ) {
                offset += this->smem_read_offset_ +
                          (mi)*Cta_tile::K * BYTES_CACHELINE;  // BYTES_PER_XMMA_PER_CTA;
            } else {
                assert( false );
            }

            // Load the data using LDSM.MT88.4 or 4x LDS.32.
            uint32_t ptr = this->smem_ + this->smem_read_buffer_ + offset;
            uint4 tmp;
            if( USE_LDSMT ) {
                ldsmt( tmp, ptr );
            } else {
                lds( tmp.x, ( ptr ) + 0 * Base::BYTES_PER_ROW );
                lds( tmp.y, ( ptr ^ 32 ) + 0 * Base::BYTES_PER_ROW );
                lds( tmp.z, ( ptr ) + 4 * Base::BYTES_PER_ROW );
                lds( tmp.w, ( ptr ^ 32 ) + 4 * Base::BYTES_PER_ROW );
            }

            // Store those values in the fragment.
            a[mi].reg( 0 ) = tmp.x;
            a[mi].reg( 1 ) = tmp.y;
            a[mi].reg( 2 ) = tmp.z;
            a[mi].reg( 3 ) = tmp.w;

            static_assert( BYTES_PER_XMMA_PER_CTA >= 128 || BYTES_PER_XMMA_PER_CTA == 64 ||
                               ( BYTES_PER_XMMA_PER_CTA == 32 &&
                                 ( Xmma_tile::XMMAS_M == 4 || Xmma_tile::XMMAS_M == 2 ||
                                   Xmma_tile::XMMAS_M == 1 ) ),
                           "" );

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( BYTES_PER_XMMA_PER_CTA >= 128 ) {
                // Nothing to do!
            } else if( BYTES_PER_XMMA_PER_CTA == 64 && Xmma_tile::XMMAS_M > 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_XMMA_PER_CTA;
            } else if( BYTES_PER_XMMA_PER_CTA == 64 ) {
                // Nothing to do!
            } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_M == 4 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * ( mi % 2 == 0 ? 2 : 6 );
            } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_M == 2 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N>
struct Rows_per_xor_pattern_hopper_row_a : public Rows_per_xor_pattern_hopper_a<Traits, N> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_hopper_row_a<Traits, Cta_tile::K>::VALUE>
struct Smem_tile_hopper_tma_row_a : public Smem_tile_hopper_tma_base<Cta_tile,
                                         Cta_tile::M,
                                         Cta_tile::K,
                                         Traits::BITS_PER_ELEMENT_A,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         1> 
 {
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_hopper_tma_base<Cta_tile,
                                         Cta_tile::M,
                                         Cta_tile::K,
                                         Traits::BITS_PER_ELEMENT_A,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         1>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>;
    // The number of XMMAs.
    using Xmma_tile_with_padding = typename Traits::template Xmma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_hopper_tma_row_a( void *smem, int tidx ) : Base( smem, tidx ) {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M = 1 * 1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert( Base::ROWS_PER_XOR_PATTERN == 8 || Base::ROWS_PER_XOR_PATTERN == 4 ||
                           Base::ROWS_PER_XOR_PATTERN == 2,
                       "" );

        if( Base::ROWS_PER_XOR_PATTERN == 8 ) {
            smem_read_row =
                ( tidx & WARP_MASK_M ) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 1 + ( tidx & 0x0f );
            smem_read_col = ( tidx & 0x07 );
            smem_read_col ^= ( tidx & 0x10 ) / 16;
            // For group fprop/dgrd. A is divided into 2 halves along K dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            if( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_col ^= ( tidx & WARP_MASK_N ) / WARP_DIV_N * ( Cta_tile::K / WARPS_N ) /
                                 ( BYTES_PER_LDS * 8 / Base::BITS_PER_ELEMENT );
            }
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = ( tidx & WARP_MASK_M ) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 2 +
                            ( tidx & 0x0e ) / 2;
            smem_read_col = ( tidx & 0x06 ) / 2 + ( tidx & 0x01 ) * 4;
            smem_read_col ^= ( tidx & 0x10 ) / 16;
        } else if( Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = ( tidx & WARP_MASK_M ) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 4 +
                            ( tidx & 0x0c ) / 4;
            smem_read_col = ( tidx & 0x04 ) / 4 + ( tidx & 0x03 ) * 2;
            smem_read_col ^= ( tidx & 0x10 ) / 16;
        }

        static_assert( WARPS_K <= 2, "" );
        static_assert( WARPS_K != 2 || Base::ROWS_PER_XOR_PATTERN != 2, "" );

        // We "swap" the block for the second warp working on the same outputs in-CTA split-K.
        if( WARPS_K == 2 ) {
            smem_read_col ^=
                ( tidx & WARP_MASK_K ) / WARP_DIV_K * Xmma_tile_with_padding::XMMAS_K * 2;
        }

        // The shared memory offset.
        this->smem_read_offset_ =
            smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset( int ki = 0 ) {

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        if( Base::ROWS_PER_XOR_PATTERN == 8 ) {
            if( Xmma_tile_with_padding::XMMAS_K == 4 ) {
                this->smem_read_offset_ ^= ( ( ki % 2 == 0 ) ? 1 : 3 ) * 2 * BYTES_PER_LDS;
            } else if( Xmma_tile_with_padding::XMMAS_K == 2 ) {
                this->smem_read_offset_ ^= 2 * BYTES_PER_LDS;
            }
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            this->smem_read_offset_ ^= 2 * BYTES_PER_LDS;
        }
    }

    // Load from shared memory.
    inline __device__ void load( Fragment ( &a )[Xmma_tile::XMMAS_M], int ki ) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // Load using LDSM.M88.4.
            uint4 tmp;
            ldsm( tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset );

            // Store the value into the fragment.
            a[mi].reg( 0 ) = tmp.x;
            a[mi].reg( 1 ) = tmp.y;
            a[mi].reg( 2 ) = tmp.z;
            a[mi].reg( 3 ) = tmp.w;
        }

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        static_assert( Xmma_tile_with_padding::XMMAS_K < 64, "Not implemented" );
        if( Xmma_tile_with_padding::XMMAS_K >= 32 && ki % 16 == 15 ) {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 16 && ki % 8 == 7 ) {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 8 && ki % 4 == 3 ) {
            this->smem_read_offset_ ^= 7 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 4 && ki % 2 == 1 ) {
            this->smem_read_offset_ ^= 3 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 2 ) {
            this->smem_read_offset_ ^= 1 * BYTES_PER_LDS * 2;
        }
    }

    // Reset the read offset.
    inline __device__ void reset_read_offset() {
        // The number of XMMAs in the K dimension.
        enum { XMMAS_K = Xmma_tile::XMMAS_K };
        // The number of XMMAs in the K dimension when we include padding.
        enum { XMMAS_K_WITH_PADDING = Xmma_tile_with_padding::XMMAS_K };
        // Assemble the mask.
        enum { MASK = Compute_reset_mask<XMMAS_K, XMMAS_K_WITH_PADDING>::VALUE };

        // Reset the read offset.
        this->smem_read_offset_ ^= MASK * BYTES_PER_LDS * 2;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N> struct Rows_per_xor_pattern_hopper_b {
    // The size in bits.
    enum { N_IN_BITS = N * Traits::BITS_PER_ELEMENT_B };
    // The number of rows.
    enum { VALUE = N_IN_BITS <= 256 ? 2 : ( N_IN_BITS <= 512 ? 4 : 8 ) };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N>
struct Rows_per_xor_pattern_hopper_col_b : public Rows_per_xor_pattern_hopper_b<Traits, N> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_hopper_col_b<Traits, Cta_tile::K>::VALUE>
struct Smem_tile_hopper_tma_col_b : public Smem_tile_hopper_tma_base<Cta_tile,
                                         Cta_tile::N,
                                         Cta_tile::K,
                                         Traits::BITS_PER_ELEMENT_B,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         1>
 {
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_hopper_tma_base<Cta_tile,
                                         Cta_tile::N,
                                         Cta_tile::K,
                                         Traits::BITS_PER_ELEMENT_B,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         1>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>;
    // The number of XMMAs.
    using Xmma_tile_with_padding = typename Traits::template Xmma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // The number of STS per thread
    enum { STS_PER_THREAD_ = Base::ROWS * Base::THREADS_PER_ROW / Cta_tile::THREADS_PER_CTA };
    // The number of STS per thread must be at least 1.
    enum { STS_PER_THREAD = Max<1, STS_PER_THREAD_>::VALUE };

    // Ctor.
    inline __device__ Smem_tile_hopper_tma_col_b( void *smem, int tidx ) : Base( smem, tidx ) {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert( Base::ROWS_PER_XOR_PATTERN == 8 || Base::ROWS_PER_XOR_PATTERN == 4 ||
                           Base::ROWS_PER_XOR_PATTERN == 2,
                       "" );

        if( Base::ROWS_PER_XOR_PATTERN == 8 ) {
            // For group fprop. B is divided into 2 halves along N dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            if( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_row =
                    ( tidx & WARP_MASK_N ) / WARP_DIV_N * ( Cta_tile::N / WARPS_N ) / 1 +
                    ( tidx & 0x07 ) + ( tidx & 0x10 ) / 2;
            } else {
                smem_read_row = ( tidx & WARP_MASK_N ) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 1 +
                                ( tidx & 0x07 ) + ( tidx & 0x10 ) / 2;
            }
            smem_read_col = ( tidx & 0x07 );
            smem_read_col ^= ( tidx & 0x08 ) / 8;
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = ( tidx & WARP_MASK_N ) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 2 +
                            ( tidx & 0x06 ) / 2 + ( tidx & 0x10 ) / 4;
            smem_read_col = ( tidx & 0x06 ) / 2 + ( tidx & 0x01 ) * 4;
            smem_read_col ^= ( tidx & 0x08 ) / 8;
        } else if( Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = ( tidx & WARP_MASK_N ) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 4 +
                            ( tidx & 0x04 ) / 4 + ( tidx & 0x10 ) / 8;
            smem_read_col = ( tidx & 0x04 ) / 4 + ( tidx & 0x03 ) * 2;
            smem_read_col ^= ( tidx & 0x08 ) / 8;
        }

        static_assert( WARPS_K <= 2, "" );
        static_assert( WARPS_K != 2 || Base::ROWS_PER_XOR_PATTERN != 2, "" );

        // We "swap" the block for the second warp working on the in-CTA split-K.
        if( WARPS_K == 2 ) {
            smem_read_col ^=
                ( tidx & WARP_MASK_K ) / WARP_DIV_K * Xmma_tile_with_padding::XMMAS_K * 2;
        }

        // The shared memory offset.
        this->smem_read_offset_ =
            smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_LDS;

        // Fill zeroes for group colw
        if( Base::BITS_PER_ELEMENT == 16 && Cta_tile::GROUPS == 16 ) {
            int row_idx = threadIdx.x & ( Base::THREADS_PER_ROW - 1 );
            if( row_idx < 2 ) {
                uint32_t smem_ptrs[STS_PER_THREAD];
                #pragma unroll
                for( int i = 0; i < BUFFERS_PER_TILE; ++i ) {
                    this->compute_store_pointers( smem_ptrs );
                    uint4 zero = make_uint4( 0, 0, 0, 0 );
                    #pragma unroll
                    for( int ii = 0; ii < STS_PER_THREAD; ++ii ) {
                        sts( smem_ptrs[ii], zero );
                    }
                    this->move_next_write_buffer();
                }
            }
        }
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset( int ki = 0 ) {

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        if( Base::ROWS_PER_XOR_PATTERN == 8 ) {
            if( Xmma_tile_with_padding::XMMAS_K == 4 ) {
                this->smem_read_offset_ ^= ( ( ki % 2 == 0 ) ? 1 : 3 ) * 2 * BYTES_PER_LDS;
            } else if( Xmma_tile_with_padding::XMMAS_K == 2 ) {
                this->smem_read_offset_ ^= 2 * BYTES_PER_LDS;
            }
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            this->smem_read_offset_ ^= 2 * BYTES_PER_LDS;
        }
    }

    // Load from shared memory.
    inline __device__ void load( Fragment ( &b )[Xmma_tile::XMMAS_N], int ki ) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = ni * ( Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA
                                                     : Xmma_tile::N_PER_XMMA_PER_CTA ) *
                         Base::BYTES_PER_ROW_BEFORE_PACKING;

            // Load using LDSM.M88.4.
            uint4 tmp;
            ldsm( tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset );

            // Store the value into the fragment.
            b[ni].reg( 0 ) = tmp.x;
            b[ni].reg( 1 ) = tmp.y;
            b[ni].reg( 2 ) = tmp.z;
            b[ni].reg( 3 ) = tmp.w;
        }

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        static_assert( Xmma_tile_with_padding::XMMAS_K < 64, "Not implemented" );
        if( Xmma_tile_with_padding::XMMAS_K >= 32 && ki % 16 == 15 ) {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 16 && ki % 8 == 7 ) {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 8 && ki % 4 == 3 ) {
            this->smem_read_offset_ ^= 7 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 4 && ki % 2 == 1 ) {
            this->smem_read_offset_ ^= 3 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 2 ) {
            this->smem_read_offset_ ^= 1 * BYTES_PER_LDS * 2;
        }
    }

    // Reset the read offset.
    inline __device__ void reset_read_offset() {
        // The number of XMMAs in the K dimension.
        enum { XMMAS_K = Xmma_tile::XMMAS_K };
        // The number of XMMAs in the K dimension when we include padding.
        enum { XMMAS_K_WITH_PADDING = Xmma_tile_with_padding::XMMAS_K };
        // Assemble the mask.
        enum { MASK = Compute_reset_mask<XMMAS_K, XMMAS_K_WITH_PADDING>::VALUE };

        // Reset the read offset.
        this->smem_read_offset_ ^= MASK * BYTES_PER_LDS * 2;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N>
struct Rows_per_xor_pattern_hopper_row_b : public Rows_per_xor_pattern_hopper_b<Traits, N> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Input_type, typename Output_type, int N>
struct Rows_per_xor_pattern_hopper_row_b<Hopper_hmma_tf32_traits<Input_type, Output_type>, N> {
    // The number of rows.
    enum { VALUE = 4 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_hopper_row_b<Traits, Cta_tile::N>::VALUE,
    // How many cols to use for the XOR pattern to avoid bank conflicts?
    int COLS_PER_XOR_PATTERN_ = Cols_per_xor_pattern_hopper<Traits>::VALUE>
struct Smem_tile_hopper_tma_row_b : public Smem_tile_hopper_tma_base<Cta_tile,
                                         Cta_tile::K,
                                         Cta_tile::N,
                                         Traits::BITS_PER_ELEMENT_B,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         COLS_PER_XOR_PATTERN_> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_hopper_tma_base<Cta_tile,
                                         Cta_tile::K,
                                         Cta_tile::N,
                                         Traits::BITS_PER_ELEMENT_B,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         COLS_PER_XOR_PATTERN_>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Row>;

    // Can we use LDSM? No if the data type is 32-bit large.
    enum { USE_LDSMT = Traits::BITS_PER_ELEMENT_B == 16 };
    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = USE_LDSMT ? 16 : 4 };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_B };

    // The number of STS per thread
    enum { STS_PER_THREAD_ = Base::ROWS * Base::THREADS_PER_ROW / Cta_tile::THREADS_PER_CTA };
    // The number of STS per thread must be at least 1.
    enum { STS_PER_THREAD = Max<1, STS_PER_THREAD_>::VALUE };

    enum { BYTES_CACHELINE = 128 };

    // Ctor.
    inline __device__ Smem_tile_hopper_tma_row_b( void *smem, int tidx ) : Base( smem, tidx ) {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row/col read by the thread.
        int smem_read_row, smem_read_col;

        static_assert( ( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 8 ) ||
                           Base::ROWS_PER_XOR_PATTERN == 4 || Base::ROWS_PER_XOR_PATTERN == 2,
                       "" );

        if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 8 ) {
            // For group dgrad. B is divided into 2 halves along K dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            if( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_row = ( tidx & WARP_MASK_N ) / WARP_DIV_N * ( Cta_tile::N / WARPS_N ) +
                                ( tidx & 0x07 ) + ( tidx & 0x08 );
            } else {
                smem_read_row = ( tidx & WARP_MASK_K ) / WARP_DIV_K * Xmma_tile::XMMAS_K * 16 +
                                ( tidx & 0x07 ) + ( tidx & 0x08 );
            }
            smem_read_col = ( tidx & 0x07 );
        } else if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = ( tidx & WARP_MASK_K ) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 +
                            ( tidx & 0x06 ) / 2 + ( tidx & 0x08 ) / 2;
            smem_read_col = ( tidx & 0x01 ) * 4 + ( tidx & 0x06 ) / 2;
        } else if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = ( tidx & WARP_MASK_K ) / WARP_DIV_K * Xmma_tile::XMMAS_K * 4 +
                            ( tidx & 0x04 ) / 4 + ( tidx & 0x08 ) / 4;
            smem_read_col = ( tidx & 0x03 ) * 2 + ( tidx & 0x04 ) / 4;
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 && Base::COLS_PER_XOR_PATTERN == 2 ) {
            if( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_row = ( tidx & WARP_MASK_N ) / WARP_DIV_N * ( Cta_tile::N / WARPS_N ) +
                                ( tidx & 0x03 );
            } else {
                smem_read_row =
                    ( tidx & WARP_MASK_K ) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 + ( tidx & 0x03 );
            }
            smem_read_col = ( tidx & 0x1c ) / 4 + ( tidx & 0x03 ) * 8;
        }

        // Each half-warp applies a different XOR pattern -- see the Excel document.
        if( USE_LDSMT ) {
            if( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 )
                smem_read_col ^= ( tidx & 0x10 ) / 16;
            else
                smem_read_col ^= ( tidx & WARP_MASK_N ) / WARP_DIV_N * 2 + ( tidx & 0x10 ) / 16;
        } else {
            // Only for non-group.
            if( Cta_tile::GROUPS == 1 ) {
                smem_read_col ^= ( tidx & WARP_MASK_N ) / WARP_DIV_N * 16;
            }
        }

        // The shared memory offset.
        // this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW +
        // smem_read_col*BYTES_PER_LDS;
        this->smem_read_offset_ = smem_read_row * BYTES_CACHELINE + smem_read_col * BYTES_PER_LDS;

        // Fill zeroes for group colw
        if( Base::BITS_PER_ELEMENT == 16 && Cta_tile::GROUPS == 16 && Cta_tile::WARPS_N > 1 ) {
            int row_idx = threadIdx.x & ( Base::THREADS_PER_ROW - 1 );
            if( row_idx < 2 ) {
                uint32_t smem_ptrs[STS_PER_THREAD];
                #pragma unroll
                for( int i = 0; i < BUFFERS_PER_TILE; ++i ) {
                    this->compute_store_pointers( smem_ptrs );
                    uint4 zero = make_uint4( 0, 0, 0, 0 );
                    #pragma unroll
                    for( int ii = 0; ii < STS_PER_THREAD; ++ii ) {
                        sts( smem_ptrs[ii], zero );
                    }
                    this->move_next_write_buffer();
                }
            }
        }
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset( int ki = 0 ) {
        static_assert( Cta_tile::WARPS_M == 4 || Cta_tile::WARPS_M == 2 ||
                           Xmma_tile::XMMAS_M == 4 || Xmma_tile::XMMAS_M == 2 ||
                           Xmma_tile::XMMAS_M == 1,
                       "" );

        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( Cta_tile::WARPS_N == 4 ) {
                // Nothing to do!
            } else if( Cta_tile::WARPS_N == 2 ) {
                if( Xmma_tile::XMMAS_N > 1 ) {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * ( 4 );
                }
            } else if( Xmma_tile::XMMAS_N == 4 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * ( ni % 2 == 0 ? 2 : 6 );
            } else if( Xmma_tile::XMMAS_N == 2 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            }
        }
    }

    // Load from shared memory.
    inline __device__ void load( Fragment ( &b )[Xmma_tile::XMMAS_N], int ki ) {
        // The size of each element in bits.
        const int BITS_PER_ELT = Traits::BITS_PER_ELEMENT_B;
        // The size in bytes of the data needed to compute an XMMA per CTA.
        const int BYTES_PER_XMMA_PER_CTA = Xmma_tile::N_PER_XMMA_PER_CTA * BITS_PER_ELT / 8;

        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Prepare the offset.
            int offset =
                ki * Base::ROWS_PER_XOR_PATTERN * 2 * BYTES_CACHELINE;  // Base::BYTES_PER_ROW;
            if( Cta_tile::GROUPS > 1 && Cta_tile::WARPS_K == 1 && Cta_tile::WARPS_N > 1 ) {
                offset += this->smem_read_offset_;
            } else {
                if( BYTES_PER_XMMA_PER_CTA == 32 ) {
                    offset += this->smem_read_offset_;
                } else if( BYTES_PER_XMMA_PER_CTA == 64 ) {
                    offset +=
                        this->smem_read_offset_ +
                        ( ni / 2 ) * Cta_tile::K * BYTES_CACHELINE;  // BYTES_PER_XMMA_PER_CTA * 2;
                } else {
                    offset += this->smem_read_offset_ +
                              (ni)*Cta_tile::K * BYTES_CACHELINE;  // BYTES_PER_XMMA_PER_CTA;
                }
            }

            // Load the data using LDSM.MT88.2.
            uint32_t ptr = this->smem_ + this->smem_read_buffer_ + offset;
            uint4 tmp;
            if( USE_LDSMT ) {
                ldsmt( tmp, ptr );
            } else {
                lds( tmp.x, ( ptr ) + 0 * Base::BYTES_PER_ROW );
                lds( tmp.y, ( ptr ) + 4 * Base::BYTES_PER_ROW );
                lds( tmp.z, ( ptr ^ 32 ) + 0 * Base::BYTES_PER_ROW );
                lds( tmp.w, ( ptr ^ 32 ) + 4 * Base::BYTES_PER_ROW );
            }

            // Store those values in the fragment.
            b[ni].reg( 0 ) = tmp.x;
            b[ni].reg( 1 ) = tmp.y;
            b[ni].reg( 2 ) = tmp.z;
            b[ni].reg( 3 ) = tmp.w;

            static_assert( BYTES_PER_XMMA_PER_CTA >= 128 || BYTES_PER_XMMA_PER_CTA == 64 ||
                               ( BYTES_PER_XMMA_PER_CTA == 32 &&
                                 ( Xmma_tile::XMMAS_M == 4 || Xmma_tile::XMMAS_M == 2 ||
                                   Xmma_tile::XMMAS_M == 1 ) ),
                           "" );

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( Cta_tile::GROUPS > 1 && Cta_tile::WARPS_N > 1 && Xmma_tile::XMMAS_N > 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_XMMA_PER_CTA / 2;
            } else {
                if( BYTES_PER_XMMA_PER_CTA >= 128 ) {
                    // Nothing to do!
                } else if( BYTES_PER_XMMA_PER_CTA == 64 && Xmma_tile::XMMAS_N > 1 ) {
                    this->smem_read_offset_ ^= BYTES_PER_XMMA_PER_CTA;
                } else if( BYTES_PER_XMMA_PER_CTA == 64 ) {
                    // Nothing to do!
                } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_N == 4 ) {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * ( ni % 2 == 0 ? 2 : 6 );
                } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_N == 2 ) {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
                }
            }
        }
    }
};

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Hopper_hmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE, true, true>
    : public Smem_tile_hopper_tma_row_a<Hopper_hmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Hopper_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_hopper_tma_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Hopper_hmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE, true, true>
    : public Smem_tile_hopper_tma_col_a<Hopper_hmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Hopper_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_hopper_tma_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Hopper_hmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE, true, true>
    : public Smem_tile_hopper_tma_col_b<Hopper_hmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Hopper_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_hopper_tma_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Hopper_hmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE, true, true>
    : public Smem_tile_hopper_tma_row_b<Hopper_hmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Hopper_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_hopper_tma_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Hopper_hmma_fp16_traits,
                   Cta_tile,
                   Row,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE,
                   true,
                   false> : public Smem_tile_ampere_row_a<Hopper_hmma_fp16_traits,
                                                          Cta_tile,
                                                          BYTES_PER_STS,
                                                          BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Hopper_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Hopper_hmma_fp16_traits,
                   Cta_tile,
                   Col,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE,
                   true,
                   false> : public Smem_tile_ampere_col_a<Hopper_hmma_fp16_traits,
                                                          Cta_tile,
                                                          BYTES_PER_STS,
                                                          BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Hopper_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Hopper_hmma_fp16_traits,
                   Cta_tile,
                   Col,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE,
                   true,
                   false> : public Smem_tile_ampere_col_b<Hopper_hmma_fp16_traits,
                                                          Cta_tile,
                                                          BYTES_PER_STS,
                                                          BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Hopper_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Hopper_hmma_fp16_traits,
                   Cta_tile,
                   Row,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE,
                   true,
                   false> : public Smem_tile_ampere_row_b<Hopper_hmma_fp16_traits,
                                                          Cta_tile,
                                                          BYTES_PER_STS,
                                                          BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Hopper_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

template <typename Cta_tile, bool IN_CTA_SPLIT_K>
struct Swizzle_epilogue<Hopper_hmma_fp16_traits, Cta_tile, Row, 16, IN_CTA_SPLIT_K>
    : public Swizzle_turing_hmma_fp16_epilogue<Hopper_hmma_fp16_traits, Cta_tile> {

    // The traits.
    using Traits = Hopper_hmma_fp16_traits;
    // The base class.
    using Base = Swizzle_turing_hmma_fp16_epilogue<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Hopper_hmma_fp32_traits,
                   Cta_tile,
                   Row,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE,
                   true,
                   true> : public Smem_tile_hopper_tma_row_a<Hopper_hmma_fp32_traits,
                                                             Cta_tile,
                                                             BYTES_PER_STS,
                                                             BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Hopper_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_hopper_tma_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Hopper_hmma_fp32_traits,
                   Cta_tile,
                   Col,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE,
                   true,
                   true> : public Smem_tile_hopper_tma_col_a<Hopper_hmma_fp32_traits,
                                                             Cta_tile,
                                                             BYTES_PER_STS,
                                                             BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Hopper_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_hopper_tma_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Hopper_hmma_fp32_traits,
                   Cta_tile,
                   Col,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE,
                   true,
                   true> : public Smem_tile_hopper_tma_col_b<Hopper_hmma_fp32_traits,
                                                             Cta_tile,
                                                             BYTES_PER_STS,
                                                             BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Hopper_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_hopper_tma_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Hopper_hmma_fp32_traits,
                   Cta_tile,
                   Row,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE,
                   true,
                   true> : public Smem_tile_hopper_tma_row_b<Hopper_hmma_fp32_traits,
                                                             Cta_tile,
                                                             BYTES_PER_STS,
                                                             BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Hopper_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_hopper_tma_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Hopper_hmma_fp32_traits,
                   Cta_tile,
                   Row,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE,
                   true,
                   false> : public Smem_tile_ampere_row_a<Hopper_hmma_fp32_traits,
                                                          Cta_tile,
                                                          BYTES_PER_STS,
                                                          BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Hopper_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Hopper_hmma_fp32_traits,
                   Cta_tile,
                   Col,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE,
                   true,
                   false> : public Smem_tile_ampere_col_a<Hopper_hmma_fp32_traits,
                                                          Cta_tile,
                                                          BYTES_PER_STS,
                                                          BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Hopper_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Hopper_hmma_fp32_traits,
                   Cta_tile,
                   Col,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE,
                   true,
                   false> : public Smem_tile_ampere_col_b<Hopper_hmma_fp32_traits,
                                                          Cta_tile,
                                                          BYTES_PER_STS,
                                                          BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Hopper_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Hopper_hmma_fp32_traits,
                   Cta_tile,
                   Row,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE,
                   true,
                   false> : public Smem_tile_ampere_row_b<Hopper_hmma_fp32_traits,
                                                          Cta_tile,
                                                          BYTES_PER_STS,
                                                          BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Hopper_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, bool IN_CTA_SPLIT_K>
struct Swizzle_epilogue<Hopper_hmma_fp32_traits, Cta_tile, Row, 16, IN_CTA_SPLIT_K>
    : public Swizzle_turing_hmma_fp32_epilogue<Hopper_hmma_fp32_traits,
                                               Cta_tile,
                                               Row> {
    // The traits class.
    using Traits = Hopper_hmma_fp32_traits;
    // The base class.
    using Base = Swizzle_turing_hmma_fp32_epilogue<Traits, Cta_tile, Row>;

    // Ctor.
    inline __device__ Swizzle_epilogue( void *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// D M M A . F P 6 4
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Hopper_dmma_fp64_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_without_skews<Cta_tile, Cta_tile::M, Cta_tile::K,
                                    Hopper_dmma_fp64_traits::BITS_PER_ELEMENT_A,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    0, //ENABLE_LDS_FAST_PATH_
                                    4, //ROWS_PER_XOR_PATTERN_
                                    2> {
    // The traits class.
    using Traits = Hopper_dmma_fp64_traits;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::M, Cta_tile::K,
                                         Traits::BITS_PER_ELEMENT_A,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         4, //?
                                         2>; //?
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>;
    // The number of XMMAs.
    using Xmma_tile_with_padding = typename Traits::template Xmma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 8 };

    enum { BYTES_PER_ELEMENT = 8 };


    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert( Base::ROWS_PER_XOR_PATTERN == 4, "");

        if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            // warp_id * M_PER_XMMA + tidx % warp_thread_num / thread_pre_row
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA + (tidx & 0x1f ) / 4; 
            smem_read_col = tidx & 0x0f; // tidx % 16
        }

        static_assert(WARPS_K <= 1, "");

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_ELEMENT;
    }

    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset1 = mi * Xmma_tile::M_PER_XMMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING 
                             + ki * BYTES_PER_ELEMENT * Xmma_tile::K_PER_XMMA;
            int offset2 = offset1 + Xmma_tile::M_PER_XMMA * Base::BYTES_PER_ROW_BEFORE_PACKING /2 ;
            
            int base = this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_; 
            int base1 = this->smem_ + (this->smem_read_offset_ ^ (4 * BYTES_PER_ELEMENT)) + this->smem_read_buffer_;
            int base2 = this->smem_ + (this->smem_read_offset_ ^ (8 * BYTES_PER_ELEMENT)) + this->smem_read_buffer_;
            int base3 =  this->smem_ + (this->smem_read_offset_ ^ (12 * BYTES_PER_ELEMENT)) + this->smem_read_buffer_;

            // Load using LDS.64
            uint2 tmp;
            lds(tmp, base + offset1);
            // Store the value into the fragment.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;

            uint2 tmp1;
            lds(tmp1, base + offset2);
            a[mi].reg(2) = tmp1.x; 
            a[mi].reg(3) = tmp1.y;

            //--
            lds(tmp, base1 + offset1);
            // Store the value into the fragment.
            a[mi].reg(4) = tmp.x;
            a[mi].reg(5) = tmp.y;

            lds(tmp1, base1 + offset2);
            a[mi].reg(6) = tmp1.x; 
            a[mi].reg(7) = tmp1.y;

            //--
            lds(tmp, base2 + offset1);
            // Store the value into the fragment.
            a[mi].reg(8) = tmp.x;
            a[mi].reg(9) = tmp.y;

            lds(tmp1, base2 + offset2);
            a[mi].reg(10) = tmp1.x; 
            a[mi].reg(11) = tmp1.y;

            //--
            lds(tmp, base3 + offset1);
            // Store the value into the fragment.
            a[mi].reg(12) = tmp.x;
            a[mi].reg(13) = tmp.y;

            lds(tmp1, base3 + offset2);
            a[mi].reg(14) = tmp1.x; 
            a[mi].reg(15) = tmp1.y;
        }
        //smem_read_offset is idential for different K.
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Hopper_dmma_fp64_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_without_skews<Cta_tile,
                                    Cta_tile::K,
                                    Cta_tile::M,
                                    Ampere_dmma_fp64_traits::BITS_PER_ELEMENT_A,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    0,
                                    4,
                                    2> {
    // The traits class.
    using Traits = Hopper_dmma_fp64_traits;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile,
                                         Cta_tile::K,
                                         Cta_tile::M,
                                         Traits::BITS_PER_ELEMENT_A,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         4,
                                         2>;

    // The fragment.
    using Fragment = Fragment_a<Traits, Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 8 };

    enum { BYTES_PER_ELEMENT = 8};

    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_A };

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert( Base::ROWS_PER_XOR_PATTERN == 4, "");

        if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = tidx & 0x03;

            // warp_id * M_PER_XMMA + tidx % warp_thread_num / thread_pre_row
            smem_read_col = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA + (tidx & 0x1f) / 4;  
            smem_read_col ^= smem_read_row * 4;
        }

        static_assert(WARPS_K <= 1, "");

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_ELEMENT;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {

        //Need load 16x16 for a 
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Jump by as many matrix rows as needed.
            int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * BYTES_PER_ELEMENT 
                            + ki * Base::BYTES_PER_ROW *Xmma_tile::K_PER_XMMA;
            int base = this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_;
            int xor_pattern =  Xmma_tile::M_PER_XMMA / 2 * BYTES_PER_ELEMENT;
            
            assert(Xmma_tile::K_PER_XMMA == 16);

            // Load using LDS.64
            uint2 tmp;
            lds(tmp, base + offset);
            // Store the value into the fragment.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;

            uint2 tmp1;
            lds(tmp1, (base + offset) ^ xor_pattern);
            a[mi].reg(2) = tmp1.x; 
            a[mi].reg(3) = tmp1.y;

            //---
            int offset1 = offset + Base::BYTES_PER_ROW * 4;
            lds(tmp, base + offset1);
            a[mi].reg(4) = tmp.x; 
            a[mi].reg(5) = tmp.y;

            lds(tmp1, (base + offset1) ^ xor_pattern);
            a[mi].reg(6) = tmp1.x; 
            a[mi].reg(7) = tmp1.y;
            //---
            int offset2 = offset + Base::BYTES_PER_ROW * 8;
            lds(tmp, base + offset2);
            a[mi].reg(8) = tmp.x; 
            a[mi].reg(9) = tmp.y;

            lds(tmp1, (base + offset2) ^ xor_pattern);
            a[mi].reg(10) = tmp1.x; 
            a[mi].reg(11) = tmp1.y;
            //---
            int offset3 = offset + Base::BYTES_PER_ROW * 12;
            lds(tmp, base + offset3);
            a[mi].reg(12) = tmp.x; 
            a[mi].reg(13) = tmp.y;

            lds(tmp1, (base + offset3) ^ xor_pattern);
            a[mi].reg(14) = tmp1.x; 
            a[mi].reg(15) = tmp1.y;
        }
        //smem_read_offset is idential for different K.
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Hopper_dmma_fp64_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE >
    : public Smem_tile_without_skews<Cta_tile,
                                    Cta_tile::K,
                                    Cta_tile::N,
                                    Ampere_dmma_fp64_traits::BITS_PER_ELEMENT_B,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    0,
                                    4,
                                    2> {

    // The traits class.
    using Traits = Hopper_dmma_fp64_traits;
        // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile,
                                         Cta_tile::K,
                                         Cta_tile::N,
                                         Traits::BITS_PER_ELEMENT_B,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         4,
                                         2>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Row>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 8 };

    enum { BYTES_PER_ELEMENT = 8};

    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_B };

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert( Base::ROWS_PER_XOR_PATTERN == 4, "");

        if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = tidx & 0x03;
            smem_read_col = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA + (tidx & 0x1c) / 4;
            smem_read_col ^= smem_read_row * 4;
        }

        static_assert(WARPS_K <= 1, "");

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_ELEMENT;

    }


    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        //Need load 8*16 for b
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Jump by as many matrix rows as needed.
            int offset = ni * Xmma_tile::N_PER_XMMA_PER_CTA * BYTES_PER_ELEMENT 
                            + ki * Base::BYTES_PER_ROW * Xmma_tile::K_PER_XMMA;
            int base = this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ ;
            int xor_pattern =  Xmma_tile::N_PER_XMMA / 2 * BYTES_PER_ELEMENT;

            uint2 tmp;
            lds(tmp, base + offset);
            // Store the value into the fragment.
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            
            uint2 tmp1;
            lds(tmp1, (base + offset) ^ xor_pattern);
            b[ni].reg(8) = tmp1.x; 
            b[ni].reg(9) = tmp1.y;

            //---
            int offset1 = offset + Base::BYTES_PER_ROW * 4;
            lds(tmp, base + offset1);
            b[ni].reg(2) = tmp.x; 
            b[ni].reg(3) = tmp.y;

            lds(tmp1, (base + offset1) ^ xor_pattern);
            b[ni].reg(10) = tmp1.x; 
            b[ni].reg(11) = tmp1.y;
            //---
            int offset2 = offset + Base::BYTES_PER_ROW * 8;
            lds(tmp, base + offset2);
            b[ni].reg(4) = tmp.x; 
            b[ni].reg(5) = tmp.y;

            lds(tmp1, (base + offset2) ^ xor_pattern);
            b[ni].reg(12) = tmp1.x; 
            b[ni].reg(13) = tmp1.y;
            //---
            int offset3 = offset + Base::BYTES_PER_ROW * 12;
            lds(tmp, base + offset3);
            b[ni].reg(6) = tmp.x; 
            b[ni].reg(7) = tmp.y;

            lds(tmp1, (base + offset3) ^ xor_pattern);
            b[ni].reg(14) = tmp1.x; 
            b[ni].reg(15) = tmp1.y;
        }
        //smem_read_offset is idential for different K.
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Hopper_dmma_fp64_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE >
    : public Smem_tile_without_skews<Cta_tile,
                                    Cta_tile::N,
                                    Cta_tile::K,
                                    Ampere_dmma_fp64_traits::BITS_PER_ELEMENT_B,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    0,
                                    4,
                                    2>{

    // The traits class.
    using Traits = Hopper_dmma_fp64_traits;

  // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile,
                                         Cta_tile::N,
                                         Cta_tile::K,
                                         Traits::BITS_PER_ELEMENT_B,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         4,
                                         2>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>;

    // The number of XMMAs.
    using Xmma_tile_with_padding = typename Traits::template Xmma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 8 };

    enum { BYTES_PER_ELEMENT = 8 };

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert(Base::ROWS_PER_XOR_PATTERN == 4, "");

        if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA +  (tidx & 0x1c ) / 4;
            smem_read_col = tidx & 0x0f;
        }

        static_assert(WARPS_K <= 1, "");
    
        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_ELEMENT;

    }


    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset1 = ni * Xmma_tile::N_PER_XMMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING 
                            + ki * BYTES_PER_ELEMENT * Xmma_tile::K_PER_XMMA;
            int offset2 = offset1 + Xmma_tile::N_PER_XMMA * Base::BYTES_PER_ROW_BEFORE_PACKING /2 ;
        
            int base = this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_;
            int base1 = this->smem_ + (this->smem_read_offset_ ^ (4 * BYTES_PER_ELEMENT)) + this->smem_read_buffer_;
            int base2 = this->smem_ + (this->smem_read_offset_ ^ (8 * BYTES_PER_ELEMENT)) + this->smem_read_buffer_;
            int base3 =  this->smem_ + (this->smem_read_offset_ ^ (12 * BYTES_PER_ELEMENT)) + this->smem_read_buffer_;

            // Load using LDS.64
            uint2 tmp;
            uint2 tmp1;
            lds(tmp, base + offset1);
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;

            lds(tmp1, base + offset2);
            b[ni].reg(8) = tmp1.x;
            b[ni].reg(9) = tmp1.y;

            //----
            lds(tmp, base1 + offset1);
            b[ni].reg(2) = tmp.x;
            b[ni].reg(3) = tmp.y;

            lds(tmp1, base1 + offset2);
            b[ni].reg(10) = tmp1.x;
            b[ni].reg(11) = tmp1.y;

            //----
            lds(tmp, base2 + offset1);
            b[ni].reg(4) = tmp.x;
            b[ni].reg(5) = tmp.y;

            lds(tmp1, base2 + offset2);
            b[ni].reg(12) = tmp1.x;
            b[ni].reg(13) = tmp1.y;

            //----
            lds(tmp, base3 + offset1);
            b[ni].reg(6) = tmp.x;
            b[ni].reg(7) = tmp.y;

            lds(tmp1, base3 + offset2);
            b[ni].reg(14) = tmp1.x;
            b[ni].reg(15) = tmp1.y;
        }
        //smem_read_offset is idential for different K.
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K>
struct Swizzle_epilogue<Hopper_dmma_fp64_traits, Cta_tile, Row, 16, IN_CTA_SPLIT_K>
    : public Swizzle_turing_epilogue<Hopper_dmma_fp64_traits, Cta_tile, double> {

    // The traits.
    using Traits = Hopper_dmma_fp64_traits;
    // The base class.
    using Base =  Swizzle_turing_epilogue<Hopper_dmma_fp64_traits, Cta_tile, double>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {   
        for( int mi = 0; mi < Base::M_PER_XMMA_PER_THREAD; ++mi ) { 
            int offset = mi * Base::THREADS_PER_XMMA_M * Base::BYTES_PER_ROW_WITH_SKEW +
                         ni * Xmma_tile::N_PER_XMMA_PER_CTA * sizeof(double);

            uint32_t ptr = this->smem_ + this->smem_write_offset_ + offset; 

            sts(ptr +  0, 
                make_uint4( c.reg(4*mi + 0), 
                            c.reg(4*mi + 1),
                            c.reg(4*mi + 2),
                            c.reg(4*mi + 3)));
            
            sts(ptr + sizeof(double) * 2 * Base::THREADS_PER_XMMA_N,  
                make_uint4( c.reg(4*mi + 8), 
                            c.reg(4*mi + 9), 
                            c.reg(4*mi + 10),
                            c.reg(4*mi + 11)));
        }

    }


    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        int offset = oi * Base::PIXELS_PER_STG * Base::BYTES_PER_ROW_WITH_SKEW;

        uint4 tmp;
        lds(tmp, this->smem_ + this->smem_read_offset_ + offset);
        dst.reg(0) = tmp.x;
        dst.reg(1) = tmp.y;
        dst.reg(2) = tmp.z;
        dst.reg(3) = tmp.w;
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
/// @brief Interface to Smem tiles for a operator
//  HGMMA
//
////////////////////////////////////////////////////////////////////////////////////////////////////
enum class Gmma_fusion_mode {NO_FUSION, BN_APPLY};

template <typename Traits, 
          typename Cta_tile,
          typename Layout,
          int BUFFERS_PER_TILE = 1,
          xmma::Gmma_descriptor_mode desc_mode = xmma::Gmma_descriptor_mode::SWIZZLE_128B,
          bool GMMA_A_RF = Traits::GMMA_A_RF,
          // by default, there is no fusion.
          xmma::Gmma_fusion_mode fusion_mode = xmma::Gmma_fusion_mode::NO_FUSION,
          bool USE_TMA = false>
struct Smem_tile_hopper_a {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, 
          typename Cta_tile, 
          typename Layout,
          int BUFFERS_PER_TILE = 1,
          xmma::Gmma_descriptor_mode desc_mode = xmma::Gmma_descriptor_mode::SWIZZLE_128B,
          bool USE_TMA = false>
struct Smem_tile_hopper_b {};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Col Major. For GMMA, A is from SMEM directly.
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_col_a {
  
    // Lwrrently Interleaved Mode is not implemented. 
    static_assert(desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_NONE, 
      "Lwrrently, Interleaved Mode is not implemented.\n");
      
    // HGMMA operation, where A and B should be in fp16. 
    static_assert(sizeof(typename Traits::A_type) == 2 && sizeof(typename Traits::B_type) == 2,
      "HGMMA operation, where A and B should be in fp16 is required.\n");
      
    // For SWIZZLE_64B, col a is not needed/implemented
    static_assert(desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_64B,
      "Lwrrently, for SWIZZLE_64B mode, col_a is not needed/implemented. \n");

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The fragment.
    using Fragment = Fragment_a<Traits, Col>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr xmma::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP =
        xmma::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = xmma::Gmma_descriptor_a<xmma::Gmma_descriptor_transpose::TRANS,
                                                    desc_mode,
                                                    Cta_tile,
                                                    Traits::BITS_PER_ELEMENT_A,
                                                    Traits::GMMA_M,
                                                    Traits::GMMA_N,
                                                    Traits::GMMA_K,
                                                    GMMA_DESC_SIZE_PER_GROUP>;

    // the size in bits of each element
    enum { BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_A };
    // the size of bytes of each element
    enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8 };
    // The size in bytes of a single LDGSTS/STS.
    enum { BYTES_PER_STS = 16 };
    // The number of elements per LDGSTS/STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for SWIZZLE_128B and
    // SWIZZLE_64B format
    enum { BYTES_PER_COLUMN = 128 };
    // the number of columns per one column of M due the the limiation of leading dim size
    enum {
        NUM_COLS_PER_M =
            ( Cta_tile::M * BYTES_PER_ELEMENT + BYTES_PER_COLUMN - 1 ) / BYTES_PER_COLUMN
    };
    // the number of columns per one column of M_PER_GMMA_GROUP
    enum {
        NUM_COLS_PER_GMMA_GROUP_M =
            ( Xmma_tile::M_PER_GMMA_GROUP * BYTES_PER_ELEMENT + BYTES_PER_COLUMN - 1 ) /
            BYTES_PER_COLUMN
    };
    // for 64xNx16 GMMA shape, NUM_ROWS_PER_GMMA_GROUP_M must be 1.
    static_assert( NUM_COLS_PER_GMMA_GROUP_M == 1,
                   "for 64xNx16 GMMA shape, NUM_ROWS_PER_GMMA_GROUP_M must be 1.\n" );
    // Number of SMEM columns
    enum { NUM_COLUMNS = Cta_tile::K * NUM_COLS_PER_M };
    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = NUM_COLUMNS * BYTES_PER_COLUMN };
    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum { BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16 };
    // this is needed to decrement GMMA desc
    enum {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB =
            BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };
    // The number of buffers.
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // The size in bytes of total buffers.
    // +128 byte to guarantee that the base address can be aligned to 128B
    enum { BYTES_FOR_ALIGNMENT = 128 };
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE + BYTES_FOR_ALIGNMENT };
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_FOR_ALIGNMENT - BYTES_PER_BUFFER };
    // The number of threads needed to store a column
    enum { THREADS_PER_COLUMN = BYTES_PER_COLUMN / BYTES_PER_STS };
    // The number of columns written with a single STS.
    enum { COLUMNS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_COLUMN };
    // for swizzle_128B the xor factor is 8
    enum { COLUMNS_PER_XOR_PATTERN = 8 };
    // The distance in byte between different GMMA groups (might need multiple due to cta tile size)
    // each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum {
        GMMA_GROUP_SMEM_DISTANCE =
            Xmma_tile::K_PER_GMMA_GROUP * NUM_COLS_PER_GMMA_GROUP_M * BYTES_PER_COLUMN
    };

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_col_a( char *smem, int tidx ) : smem_( smem ) {
        int smem_write_col = tidx / THREADS_PER_COLUMN;
        int smem_write_xor = smem_write_col % COLUMNS_PER_XOR_PATTERN;
        int smem_write_row = ( tidx % THREADS_PER_COLUMN ) ^ smem_write_xor;
        this->smem_write_offset_ =
            smem_write_col * BYTES_PER_COLUMN + smem_write_row * BYTES_PER_STS;
    }

    inline __device__ void add_smem_barrier_base( uint64_t * ) {
    }

    // set the scale and bias smem pointer
    // do nothing. 
    inline __device__ void set_scale_bias_smem_ptr(char *scale_bias_smem_ptr, 
                                                   int tidx, 
                                                   int k) { }
    // Load from shared memory.
    // LDSM is not required is both operands coming from SMEM for GMMA
    // however, for some cases it is needed
    inline __device__ void load( Fragment ( &a )[Xmma_tile::XMMAS_M], int ki ) {
    }

    // Compute the store pointers.
    template <int LDGS_M, int LDGS_N>
    inline __device__ void compute_store_pointers( uint32_t ( &ptrs )[LDGS_M * LDGS_N] ) {
        #pragma unroll
        for( int ni = 0; ni < LDGS_N; ++ni ) {
            int offset_n = smem_write_offset_ + ni * COLUMNS_PER_STS * BYTES_PER_COLUMN;
            #pragma unroll
            for( int mi = 0; mi < LDGS_M; ++mi ) {
                // LDGS_M > 1 means there is reshaping of the smem going on
                // for example a column major 128x64 block in smem
                // is now reshaped into a column major 64x128 block (not transpose)
                // see spreadsheet for more details
                int offset_m = mi * BYTES_PER_COLUMN * Cta_tile::K;
                int offset = offset_m + offset_n;
                ptrs[ni * LDGS_M + mi] = get_smem_pointer( &smem_[offset] );
            }
        }
    }

    // Store to the tile in shared memory.
    template< int LDGS_M, int LDGS_N, int M, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true> >
    inline __device__ void store(const void* (&gmem_ptrs)[LDGS_M * LDGS_N],
                                 uint32_t (&preds)[M],
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[LDGS_M * LDGS_N];
        this->compute_store_pointers<LDGS_M, LDGS_N>(smem_ptrs);
        ldgsts<LDGS_M * LDGS_N, M, 16/K, LDGSTS_CFG>(smem_ptrs, gmem_ptrs, preds, mem_desc);
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                                            ? -BYTES_PER_TILE_INC_BOUNDARY
                                            : BYTES_PER_BUFFER;
        }
    }

    inline __device__ void move_next_write_buffer( int ) {
    }

    // Move the read offset to next buffer.
    // do nothing, as it is controled by gmma desc
    inline __device__ void move_next_read_buffer() {
    }

    // The shared memory pointer.
    char *smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Col Major. For GMMA, A is from RF.
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_col_a_rf
    : public Smem_tile_hopper_gmma_col_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode> {

    // Base struct
    using Base = Smem_tile_hopper_gmma_col_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // A is loaded from RF
    static_assert( Traits::GMMA_A_RF == true,
                   "A should be loaded from RF for class Smem_tile_hopper_gmma_col_a_rf" );

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_col_a_rf( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Row Major. For GMMA, A is from SMEM directly.
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_row_a {
  
    // Lwrrently Interleaved Mode is not implemented. 
    static_assert(desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_NONE, 
      "Lwrrently, Interleaved Mode is not implemented.\n");
      
    // HGMMA operation, where A and B should be in fp16. 
    static_assert(sizeof(typename Traits::A_type) == 2 && sizeof(typename Traits::B_type) == 2,
      "HGMMA operation, where A and B should be in fp16 is required.\n");
      
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr xmma::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP =
        xmma::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = xmma::Gmma_descriptor_a<xmma::Gmma_descriptor_transpose::NOTRANS,
                                                    desc_mode,
                                                    Cta_tile,
                                                    Traits::BITS_PER_ELEMENT_A,
                                                    Traits::GMMA_M,
                                                    Traits::GMMA_N,
                                                    Traits::GMMA_K,
                                                    GMMA_DESC_SIZE_PER_GROUP>;

    // the size in bits of each element
    enum { BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_A };
    // the size of bytes of each element
    enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8 };
    // The size in bytes of a single LDGSTS/STS.
    enum { BYTES_PER_STS = 16 };
    // The number of elements per LDGSTS/STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for SWIZZLE_128B
    // and SWIZZLE_64B format
    enum { BYTES_PER_ROW = 128 };
    // the number of rows per one row of K due the the limiation of leading dim size
    enum {
        NUM_ROWS_PER_K = ( Cta_tile::K * BYTES_PER_ELEMENT + BYTES_PER_ROW - 1 ) / BYTES_PER_ROW
    };

    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_64B || Cta_tile::K == 32,
                   "swizzle_64B row_a is valid if kblock=32\n" );
    // Number of SMEM rows
    enum {
        NUM_ROWS = ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B )
                       ? ( Cta_tile::M * NUM_ROWS_PER_K )
                       : ( Cta_tile::M / 2 )
    };
    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = NUM_ROWS * BYTES_PER_ROW };
    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum { BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16 };
    // this is needed to decrement GMMA desc
    enum {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB =
            BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };
    // The number of buffers.
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // The size in bytes of total buffers.
    enum { BYTES_FOR_ALIGNMENT = 128 };
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE + BYTES_FOR_ALIGNMENT };
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_FOR_ALIGNMENT - BYTES_PER_BUFFER };
    // The number of threads needed to store a row
    enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STS };
    // The number of rows written with a single STS.
    enum { ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // for swizzle_128B the xor factor is 8
    enum {
        ROWS_PER_XOR_PATTERN = ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B ) ? 8 : 4
    };
    // The distance in byte between different GMMA groups (might need multiple due to cta tile size)
    // each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum {
        GMMA_GROUP_SMEM_DISTANCE =
            Xmma_tile::M_PER_GMMA_GROUP /
            ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B ? 1 : 2 ) *
            BYTES_PER_ROW
    };

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_row_a( char *smem, int tidx ) : smem_( smem ) {
        int smem_write_row = tidx / THREADS_PER_ROW;
        int smem_write_xor = smem_write_row % ROWS_PER_XOR_PATTERN;
        int smem_write_col = 0;

        if( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B ) {
            smem_write_col = ( tidx % THREADS_PER_ROW ) ^ smem_write_xor;
        } else if( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_64B ) {
            smem_write_col =
                ( tidx % ( THREADS_PER_ROW / 2 ) ) ^
                smem_write_xor + ( ( tidx % THREADS_PER_ROW ) / ( THREADS_PER_ROW / 2 ) ) * 4;
        }

        this->smem_write_offset_ = smem_write_row * BYTES_PER_ROW + smem_write_col * BYTES_PER_STS;
    }

    inline __device__ void add_smem_barrier_base( uint64_t * ) {
    }

    // set the scale and bias smem pointer
    // do nothing. 
    inline __device__ void set_scale_bias_smem_ptr(char *scale_bias_smem_ptr, 
                                                   int tidx, 
                                                   int k) { }

    // Load from shared memory.
    inline __device__ void load( Fragment ( &a )[Xmma_tile::XMMAS_M], int ki ) {
    }
    // Compute the store pointers.
    template <int LDGS_M, int LDGS_N>
    inline __device__ void compute_store_pointers( uint32_t ( &ptrs )[LDGS_M * LDGS_N] ) {
        #pragma unroll
        for( int ni = 0; ni < LDGS_N; ++ni ) {
            int offset_n = smem_write_offset_ + ni * ROWS_PER_STS * BYTES_PER_ROW;
            #pragma unroll
            for( int mi = 0; mi < LDGS_M; ++mi ) {
                int offset_m = mi * BYTES_PER_ROW * Cta_tile::M;
                int offset = offset_m + offset_n;
                ptrs[ni * LDGS_M + mi] = get_smem_pointer( &smem_[offset] );
            }
        }
    }

    // Store to the tile in shared memory.
    template< int LDGS_M, int LDGS_N, int M, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true> >
    inline __device__ void store(const void* (&gmem_ptrs)[LDGS_M * LDGS_N],
                                 uint32_t (&preds)[M],
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[LDGS_M * LDGS_N];
        this->compute_store_pointers<LDGS_M, LDGS_N>(smem_ptrs);
        ldgsts<LDGS_M * LDGS_N, M, 16/K, LDGSTS_CFG>(smem_ptrs, gmem_ptrs, preds, mem_desc);
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                                            ? -BYTES_PER_TILE_INC_BOUNDARY
                                            : BYTES_PER_BUFFER;
        }
    }

    inline __device__ void move_next_write_buffer( int ) {
    }

    // Move the read offset to next buffer.
    // do nothing, as it is controled by gmma desc
    inline __device__ void move_next_read_buffer() {
    }

    // The shared memory pointer.
    char *smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Row Major. For GMMA, A is from RF.
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_row_a_rf
    : public Smem_tile_hopper_gmma_row_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode> {
    // Base struct
    using Base = Smem_tile_hopper_gmma_row_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // A is loaded from RF
    static_assert( Traits::GMMA_A_RF == true,
                   "A should be loaded from RF for class Smem_tile_hopper_gmma_row_a_rf" );

    // The size in bytes of a single LDS.
    enum { BYTES_PER_LDS = 16 };
    // The number of LDSM.M88.4 that is needed per k=16.
    // if Mblock = 64, one LDSM is needed. if Mblock = 128, two LDSM are needed, assuming 1 WG.
    enum {
        NUM_LDSM =
            ( ( Cta_tile::M / 8 ) * ( Traits::GMMA_K / 8 ) / 4 / ( Cta_tile::WARPS_PER_CTA ) )
    };
    // for now, the number of LDSM is 1, 2, 4
    static_assert( NUM_LDSM == 1 || NUM_LDSM == 2 || NUM_LDSM == 4,
                   "NUM_LDSM should be 1, 2 or 4\n" );
    // for now, the number of warps is assumed to be 4
    static_assert( Cta_tile::WARPS_PER_CTA == 4,
                   "for now the number of warps per cta is assumed to be 4\n" );
    // The distance between neighboring LDSM (along M dimension)
    enum { LDSM_DISTANCE_IN_BYTE = Traits::GMMA_M * Base::BYTES_PER_ROW };
    // The number of elements per LDSM per thread.
    enum { ELEMENTS_PER_LDSM_PER_THREAD = 8 };
    // XMMAS_K should be only 4.
    static_assert( Base::Xmma_tile::XMMAS_K == 4, "Xmma_tile::XMMAS_K must be 4\n" );

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_row_a_rf( char *smem, int tidx ) : Base( smem, tidx ) {
        // for LDSM 884, each warp reads 16x16 block. 4 warps together read 64x16 block
        int warp_m_idx = tidx / Cta_tile::THREADS_PER_WARP;
        // within a warp, 16x16 block is splited into 4 8x8 block.
        // the first 8 threads hold the address to the 8x8 block on the upper left corner
        // the seond 8 threads hold the address to the 8x8 block on the lower left corner
        // the thrid 8 threads hold the address to the 8x8 block on the upper right corner
        // the fourth 8 threads hold the address to the 8x8 block on the lower right corner
        int quad_pair_idx = ( tidx % Cta_tile::THREADS_PER_WARP ) / 8;
        int quad_pair_m_idx = quad_pair_idx % 2;
        int quad_pair_n_idx = quad_pair_idx / 2;

        int tidx_in_quad_pair = tidx % 8;
        this->smem_read_offset_ = tidx_in_quad_pair * Base::BYTES_PER_ROW +
                                  ( quad_pair_n_idx ^ tidx_in_quad_pair ) *
                                      Base::BYTES_PER_ELEMENT * ELEMENTS_PER_LDSM_PER_THREAD;
        this->smem_read_offset_ += quad_pair_m_idx * 8 * Base::BYTES_PER_ROW;
        this->smem_read_offset_ += warp_m_idx * 16 * Base::BYTES_PER_ROW;
        // I think we can get rid of this
        this->smem_read_buffer_ = __shfl_sync( 0xffffffff, 0, 0 );
        ldsm_xor_factor_ = 2 * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load( typename Base::Fragment ( &a )[Base::Xmma_tile::XMMAS_M],
                                 int ki ) {
        #pragma unroll
        for( int mi = 0; mi < Base::Xmma_tile::XMMAS_M; ++mi ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = mi * LDSM_DISTANCE_IN_BYTE;

            // Load using LDSM.M88.4.
            uint4 tmp;
            uint32_t ptr = get_smem_pointer( this->smem_ + this->smem_read_offset_ +
                                             this->smem_read_buffer_ + offset );
            ldsm( tmp, ptr );

            // Store the value into the fragment.
            a[mi].reg( 0 ) = tmp.x;
            a[mi].reg( 1 ) = tmp.y;
            a[mi].reg( 2 ) = tmp.z;
            a[mi].reg( 3 ) = tmp.w;
        }

        // move the offset to next position, within a buffer
        this->smem_read_offset_ ^= ( ldsm_xor_factor_ );
        // ldsm_xor_factor order should be 2(ki=0)->6(ki=1)->2(ki=2)->6(ki=3) * 16.
        // I am not proud of the fomula listed below,
        // note we are callwlating the ldsm_xor_factor for the next ki.
        ldsm_xor_factor_ = BYTES_PER_LDS * ( ( ( ki + 1 ) % 2 ) * 4 + 2 );
    }

    // Move the read offset to next buffer.
    inline __device__ void move_next_read_buffer() {
        if( Base::BUFFERS_PER_TILE > 1 && smem_read_buffer_ >= Base::BYTES_PER_TILE_INC_BOUNDARY ) {
            this->smem_read_buffer_ -= Base::BYTES_PER_TILE_INC_BOUNDARY;
        } else if( Base::BUFFERS_PER_TILE > 1 ) {
            this->smem_read_buffer_ += Base::BYTES_PER_BUFFER;
        }
        //smem_read_offset is idential for different K.
    }

    // The buffer base offset for read. I think we can get rid of this
    int smem_read_buffer_;
    //
    int ldsm_xor_factor_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Row Major. For GMMA, A is from RF. Fusion also applied to A
// there is good amount concepts in this class that could be removed.
// should inherit from a_rf class
// think about that before make an MR [Timmy]
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_row_a_rf_bn_apply 
    : public Smem_tile_hopper_gmma_row_a_rf<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>{
    // Base struct
    using Base = Smem_tile_hopper_gmma_row_a_rf<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;
    
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;
      
    // A is loaded from RF
    static_assert(Traits::GMMA_A_RF == true, 
      "A should be loaded from RF for class Smem_tile_hopper_gmma_row_a_rf_bn_apply");
    
    // SCALE and BIAS in SMEM are located after A and B. 
    
    // BYTES per scale and bias element. 
    enum { BYTES_PER_SCALE_BIAS_ELEMENT = 2 };
    // per stage SMEM for scale and bias.
    enum { BYTES_PER_SCALE_BIAS_BUFFER = Cta_tile::K * BYTES_PER_SCALE_BIAS_ELEMENT * 2 };
    // scale and bias have the same number of buffers with the Operand.
    enum { BUFFERS_PER_TILE = Base::BUFFERS_PER_TILE };
    // total SMEM for scale and bias.
    enum { BYTES_PER_SCALE_BIAS_PER_TILE = BUFFERS_PER_TILE * BYTES_PER_SCALE_BIAS_BUFFER };
    // the boundary for scale and bias SMEM buffer.
    enum { BYTES_PER_SCALE_BIAS_TILE_INC_BOUNDARY = BYTES_PER_SCALE_BIAS_PER_TILE - BYTES_PER_SCALE_BIAS_BUFFER };  

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_row_a_rf_bn_apply(char *smem, int tidx) :
    Base( smem, tidx ) { }
    
    // should be called along the ctor.
    // set the scale and bias smem pointer
    inline __device__ void set_scale_bias_smem_ptr(char *scale_bias_smem_ptr, 
                                                   int tidx, 
                                                   int k) { 
        // smem pointer for ldgsts of scale and bias
        smem_scale_bias_write_ptr_ = get_smem_pointer(scale_bias_smem_ptr);
        smem_scale_bias_write_offset_ = tidx * 16;
      
        // variables to help predicate off ldsm if needed.
        scale_bias_limit = k * BYTES_PER_SCALE_BIAS_ELEMENT;
        // callwlate the index needed to load scale and bias
        scale_bias_offset = 0;
        
        // smem pointer for ldsm of scale and bias
        smem_scale_bias_ldsm_ptr_ = get_smem_pointer(scale_bias_smem_ptr);
        int quad_pair_idx = (tidx % Cta_tile::THREADS_PER_WARP) / 8;
        smem_scale_bias_ldsm_offset_ = (quad_pair_idx / 2) * Cta_tile::K * BYTES_PER_SCALE_BIAS_ELEMENT
                                     + (quad_pair_idx % 2) * 8 * BYTES_PER_SCALE_BIAS_ELEMENT;
    }
    
    // Store LDGSTS for scale and bias
    // only 16 threads need to participate (k =64)
    inline __device__ void store_scale_bias(const char *gmem_scale_bias_ptr_) {  
        if(threadIdx.x < 16) {
            // store scale and bias
            xmma::ldgsts128_nopreds(smem_scale_bias_write_ptr_ + smem_scale_bias_write_offset_, 
                                    gmem_scale_bias_ptr_);          
        }        
    }

    // Load from shared memory.
    // also load scale and bias.
    // also apply scale and bias. 
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        // each call to load will ldsm mblock x 16 into register. 
        // each thread owns 4 fp16 elements along k-dim
        // and XMMAS_M x 2 elements along m-dim
        
        // load scale and bias, 2 ldg.32 for scale and 2 ldg.32 for bias
        uint2 scale = make_uint2(0, 0);
        uint2 bias = make_uint2(0, 0);
        
        // one ldsm88x4 to load all scale and bias 
        if(scale_bias_offset < scale_bias_limit) {
          uint4 tmp;
          ldsm(tmp, smem_scale_bias_ldsm_ptr_ + smem_scale_bias_ldsm_offset_);
          scale.x = tmp.x;
          scale.y = tmp.y;
          bias.x = tmp.z;
          bias.y = tmp.w;
        }  
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = mi * Base::LDSM_DISTANCE_IN_BYTE;

            // Load using LDSM.M88.4.
            uint4 tmp;
            uint32_t ptr = 
              get_smem_pointer(this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            ldsm(tmp, ptr);
            
            // apply scale and bias;
            // tmp.x and tmp.y are of the same scale and bias;
            // tmp.z and tmp.w are of the same scale and bias.
            tmp.x = xmma::scale_bias_relu<true>(tmp.x, scale.x, bias.x);
            tmp.y = xmma::scale_bias_relu<true>(tmp.y, scale.x, bias.x);
            tmp.z = xmma::scale_bias_relu<true>(tmp.z, scale.y, bias.y);
            tmp.w = xmma::scale_bias_relu<true>(tmp.w, scale.y, bias.y);
            // // check OOB?
            // tmp.x = xmma::guarded_scale_bias_relu_a<true>(tmp.x, scale.x, bias.x);
            // tmp.y = xmma::guarded_scale_bias_relu_a<true>(tmp.y, scale.x, bias.x);
            // tmp.z = xmma::guarded_scale_bias_relu_a<true>(tmp.z, scale.y, bias.y);
            // tmp.w = xmma::guarded_scale_bias_relu_a<true>(tmp.w, scale.y, bias.y);  

            // Store the value into the fragment.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
        }

        // move the offset to next position, within a buffer
        this->smem_read_offset_ ^= (this->ldsm_xor_factor_);
        // ldsm_xor_factor order should be 2(ki=0)->6(ki=1)->2(ki=2)->6(ki=3) * 16. 
        // I am not proud of the fomula listed below, 
        // note we are callwlating the ldsm_xor_factor for the next ki. 
        this->ldsm_xor_factor_ = Base::BYTES_PER_LDS * (((ki+1)%2) * 4 + 2);

        // update scale_bias_offset
        scale_bias_offset += 16 * BYTES_PER_SCALE_BIAS_ELEMENT;
        
        // update the ldsm scale bias offset 
        smem_scale_bias_ldsm_offset_ += 16 * BYTES_PER_SCALE_BIAS_ELEMENT;
    }
        

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer() {
        // move next write buffer for the Operand.
        Base::move_next_write_buffer();
        // move next write buffer for scale and bias.
        if( BUFFERS_PER_TILE > 1 ) {  
            this->smem_scale_bias_write_offset_ +=
                (smem_scale_bias_write_offset_ >= BYTES_PER_SCALE_BIAS_TILE_INC_BOUNDARY)
                ? -BYTES_PER_SCALE_BIAS_TILE_INC_BOUNDARY
                : BYTES_PER_SCALE_BIAS_BUFFER ;
        } 
    }
    
    // Move the read offset to next buffer. 
    inline __device__ void move_next_read_buffer() {
        // Move the read offset to next buffer for the Operand.
        Base::move_next_read_buffer();
        // update LDSM scale bias offset to the next buffer
        if( BUFFERS_PER_TILE > 1 && 
            smem_scale_bias_ldsm_offset_ >= (BYTES_PER_SCALE_BIAS_TILE_INC_BOUNDARY + 16 * BYTES_PER_SCALE_BIAS_ELEMENT * Xmma_tile::XMMAS_K )) {
            this->smem_scale_bias_ldsm_offset_ -= BYTES_PER_SCALE_BIAS_TILE_INC_BOUNDARY + 16 * BYTES_PER_SCALE_BIAS_ELEMENT * Xmma_tile::XMMAS_K;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_scale_bias_ldsm_offset_ += BYTES_PER_SCALE_BIAS_BUFFER - 16 * BYTES_PER_SCALE_BIAS_ELEMENT * Xmma_tile::XMMAS_K;
        }           
    }
    
    // scale and bias offset in byte in GMEM
    int scale_bias_offset;
    // scale and bias limit. Along with scale_bias_offset to control when not to lds. 
    int scale_bias_limit;
    // smem pointer for ldgsts of scale and bias
    uint32_t smem_scale_bias_write_ptr_;
    // smem pointer offset for ldgsts of scale and bias
    int smem_scale_bias_write_offset_;
    
    // smem pointer for LDSM of scale and bias
    // each thread needs to load 2 elements for scale and 2 elements for bias per k=8
    // since each kgroup = 16, each thread needs to load 4 elements for scale and 4 elements for bias
    // we can use LDSM.88x4 to do this. the trick is each 8x8 matrix is actually just 8x1. 
    // 8 threads will hold the same address
    uint32_t smem_scale_bias_ldsm_ptr_;
    // per thread offset for ldsm of scale and bias.
    int smem_scale_bias_ldsm_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// B Col Major. For GMMA, B is from SMEM directly.
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_col_b {
  
    // Lwrrently Interleaved Mode is not implemented. 
    static_assert(desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_NONE, 
      "Lwrrently, Interleaved Mode is not implemented.\n");
      
    // HGMMA operation, where A and B should be in fp16. 
    static_assert(sizeof(typename Traits::A_type) == 2 && sizeof(typename Traits::B_type) == 2,
      "HGMMA operation, where A and B should be in fp16 is required.\n");
      
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr xmma::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP =
        xmma::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = xmma::Gmma_descriptor_b<xmma::Gmma_descriptor_transpose::NOTRANS,
                                                    desc_mode,
                                                    Cta_tile,
                                                    Traits::BITS_PER_ELEMENT_B,
                                                    Traits::GMMA_M,
                                                    Traits::GMMA_N,
                                                    Traits::GMMA_K,
                                                    GMMA_DESC_SIZE_PER_GROUP>;

    // the size in bits of each element
    enum { BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_B };
    // the size of bytes of each element
    enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8 };
    // The size in bytes of a single LDGSTS/STS.
    enum { BYTES_PER_STS = 16 };
    // The number of elements per LDGSTS/STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for SWIZZLE_128B and
    // SWIZZLE_64B format
    enum { BYTES_PER_COLUMN = 128 };

    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_64B || Cta_tile::K == 32,
                   "swizzle_64B col_b is valid if kblock=32\n" );
    // the number of columns per one column of K due the the limiation of leading dim size
    enum {
        NUM_COLS_PER_K =
            ( Cta_tile::K * BYTES_PER_ELEMENT + BYTES_PER_COLUMN - 1 ) / BYTES_PER_COLUMN
    };
    // Number of SMEM columns
    enum {
        NUM_COLUMNS = ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B )
                          ? Cta_tile::N * NUM_COLS_PER_K
                          : Cta_tile::N / 2
    };
    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = NUM_COLUMNS * BYTES_PER_COLUMN };
    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum { BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16 };
    // this is needed to decrement GMMA desc
    enum {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB =
            BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };
    // The number of buffers.
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // The size in bytes of total buffers.
    enum { BYTES_FOR_ALIGNMENT = 128 };
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE + BYTES_FOR_ALIGNMENT };
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_FOR_ALIGNMENT - BYTES_PER_BUFFER };
    // The number of threads needed to store a column
    enum { THREADS_PER_COLUMN = BYTES_PER_COLUMN / BYTES_PER_STS };
    // The number of columns written with a single STS.
    enum { COLUMNS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_COLUMN };
    // for swizzle_128B the xor factor is 8
    enum {
        COLUMNS_PER_XOR_PATTERN = ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B ) ? 8 : 4
    };
    // The distance in byte between different GMMA groups (might need multiple due to cta tile size)
    // each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum {
        GMMA_GROUP_SMEM_DISTANCE =
            Xmma_tile::N_PER_GMMA_GROUP /
            ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B ? 1 : 2 ) *
            BYTES_PER_COLUMN
    };

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_col_b( char *smem, int tidx ) : smem_( smem ) {
        int smem_write_col = tidx / THREADS_PER_COLUMN;
        int smem_write_xor = smem_write_col % COLUMNS_PER_XOR_PATTERN;
        int smem_write_row = 0;

        if( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B ) {
            smem_write_row = ( tidx % THREADS_PER_COLUMN ) ^ smem_write_xor;
        } else if( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_64B ) {
            smem_write_row =
                ( tidx % ( THREADS_PER_COLUMN / 2 ) ) ^
                smem_write_xor + ( ( tidx % THREADS_PER_COLUMN ) / ( THREADS_PER_COLUMN / 2 ) ) * 4;
        }

        this->smem_write_offset_ =
            smem_write_col * BYTES_PER_COLUMN + smem_write_row * BYTES_PER_STS;
    }

    inline __device__ void add_smem_barrier_base( uint64_t * ) {
    }

    // Load from shared memory.
    inline __device__ void load( Fragment ( &b )[Xmma_tile::XMMAS_N], int ki ) {
    }

    // Compute the store pointers.
    template <int LDGS_M, int LDGS_N>
    inline __device__ void compute_store_pointers( uint32_t ( &ptrs )[LDGS_M * LDGS_N] ) {
        #pragma unroll
        for( int ni = 0; ni < LDGS_N; ++ni ) {
            int offset_n = smem_write_offset_ + ni * COLUMNS_PER_STS * BYTES_PER_COLUMN;
            #pragma unroll
            for( int mi = 0; mi < LDGS_M; ++mi ) {
                int offset_m = mi * BYTES_PER_COLUMN * Cta_tile::N;
                int offset = offset_m + offset_n;
                ptrs[ni * LDGS_M + mi] = get_smem_pointer( &smem_[offset] );
            }
        }
    }

    // Store to the tile in shared memory.
    template< int LDGS_M, int LDGS_N, int M, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true> >
    inline __device__ void store(const void* (&gmem_ptrs)[LDGS_M * LDGS_N],
                                 uint32_t (&preds)[M],
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[LDGS_M * LDGS_N];
        this->compute_store_pointers<LDGS_M, LDGS_N>(smem_ptrs);
        ldgsts<LDGS_M * LDGS_N, M, 16/K, LDGSTS_CFG>(smem_ptrs, gmem_ptrs, preds, mem_desc);
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                                            ? -BYTES_PER_TILE_INC_BOUNDARY
                                            : BYTES_PER_BUFFER;
        }
    }

    inline __device__ void move_next_write_buffer( int ) {
    }

    // Move the read offset to next buffer.
    // do nothing, as it is controled by gmma desc
    inline __device__ void move_next_read_buffer() {
    }

    // The shared memory pointer.
    char *smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// B Row Major. For GMMA, A is from SMEM directly.
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode,
          int ROWS_PER_XOR_PATTERN_ =
              Rows_per_xor_pattern_hopper<Cta_tile::N * Traits::BITS_PER_ELEMENT_B>::VALUE>
struct Smem_tile_hopper_gmma_row_b {
  
    // Lwrrently Interleaved Mode is not implemented. 
    static_assert(desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_NONE, 
      "Lwrrently, Interleaved Mode is not implemented.\n");
      
    // HGMMA operation, where A and B should be in fp16. 
    static_assert(sizeof(typename Traits::A_type) == 2 && sizeof(typename Traits::B_type) == 2,
      "HGMMA operation, where A and B should be in fp16 is required.\n");

    // For SWIZZLE_64B, row b is not needed/implemented
    static_assert(desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_64B,
      "Lwrrently, for SWIZZLE_64B mode, row_b is not needed/implemented. \n");
      
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The fragment.
    using Fragment = Fragment_b<Traits, Row>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr xmma::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP =
        xmma::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = xmma::Gmma_descriptor_b<xmma::Gmma_descriptor_transpose::TRANS,
                                                    desc_mode,
                                                    Cta_tile,
                                                    Traits::BITS_PER_ELEMENT_B,
                                                    Traits::GMMA_M,
                                                    Traits::GMMA_N,
                                                    Traits::GMMA_K,
                                                    GMMA_DESC_SIZE_PER_GROUP>;

    // the size in bits of each element
    enum { BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_B };
    // the size of bytes of each element
    enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8 };
    // The size in bytes of a single LDGSTS/STS.
    enum { BYTES_PER_STS = 16 };
    // The number of elements per LDGSTS/STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for SWIZZLE_128B and
    // SWIZZLE_64B format
    enum { BYTES_PER_ROW = 128 };
    // the number of rows per one row of N due the the limiation of leading dim size
    enum {
        NUM_ROWS_PER_N = ( Cta_tile::N * BYTES_PER_ELEMENT + BYTES_PER_ROW - 1 ) / BYTES_PER_ROW
    };
    // the number of rows per one row of N_PER_GMMA_GROUP
    enum {
        NUM_ROWS_PER_GMMA_GROUP_N =
            ( Xmma_tile::N_PER_GMMA_GROUP * BYTES_PER_ELEMENT + BYTES_PER_ROW - 1 ) / BYTES_PER_ROW
    };
    // Number of SMEM rows
    enum { NUM_ROWS = Cta_tile::K * NUM_ROWS_PER_N };
    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = NUM_ROWS * BYTES_PER_ROW };
    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum { BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16 };
    // this is needed to decrement GMMA desc
    enum {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB =
            BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };
    // The number of buffers.
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // The size in bytes of total buffers.
    enum { BYTES_FOR_ALIGNMENT = 128 };
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE + BYTES_FOR_ALIGNMENT };
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_FOR_ALIGNMENT - BYTES_PER_BUFFER };
    // The number of threads needed to store a row
    enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STS };
    // The number of rows written with a single STS.
    enum { ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // for swizzle_128B the xor factor is 8
    enum { ROWS_PER_XOR_PATTERN = 8 };
    // The distance in byte between different GMMA groups (might need multiple due to cta tile size)
    // each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum {
        GMMA_GROUP_SMEM_DISTANCE =
            Xmma_tile::K_PER_GMMA_GROUP * NUM_ROWS_PER_GMMA_GROUP_N * BYTES_PER_ROW
    };

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_row_b( char *smem, int tidx ) : smem_( smem ) {
        int smem_write_row = tidx / THREADS_PER_ROW;
        int smem_write_xor = smem_write_row % ROWS_PER_XOR_PATTERN;
        int smem_write_col = ( tidx % THREADS_PER_ROW ) ^ smem_write_xor;
        this->smem_write_offset_ = smem_write_row * BYTES_PER_ROW + smem_write_col * BYTES_PER_STS;
    }

    inline __device__ void add_smem_barrier_base( uint64_t * ) {
    }

    // Load from shared memory.
    inline __device__ void load( Fragment ( &b )[Xmma_tile::XMMAS_N], int ki ) {
    }

    // Compute the store pointers.
    template <int LDGS_M, int LDGS_N>
    inline __device__ void compute_store_pointers( uint32_t ( &ptrs )[LDGS_M * LDGS_N] ) {
        #pragma unroll
        for( int ni = 0; ni < LDGS_N; ++ni ) {
            int offset_n = smem_write_offset_ + ni * ROWS_PER_STS * BYTES_PER_ROW;
            #pragma unroll
            for( int mi = 0; mi < LDGS_M; ++mi ) {
                int offset_m = mi * BYTES_PER_ROW * Cta_tile::K;
                int offset = offset_m + offset_n;
                ptrs[ni * LDGS_M + mi] = get_smem_pointer( &smem_[offset] );
            }
        }
    }

    // Store to the tile in shared memory.
    template< int LDGS_M, int LDGS_N, int M, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true> >
    inline __device__ void store(const void* (&gmem_ptrs)[LDGS_M * LDGS_N],
                                 uint32_t (&preds)[M],
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[LDGS_M * LDGS_N];
        this->compute_store_pointers<LDGS_M, LDGS_N>(smem_ptrs);
        ldgsts<LDGS_M * LDGS_N, M, 16/K, LDGSTS_CFG>(smem_ptrs, gmem_ptrs, preds, mem_desc);
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                                            ? -BYTES_PER_TILE_INC_BOUNDARY
                                            : BYTES_PER_BUFFER;
        }
    }

    inline __device__ void move_next_write_buffer( int ) {
    }

    // Move the read offset to next buffer.
    // do nothing, as it is controled by gmma desc
    inline __device__ void move_next_read_buffer() {
    }

    // The shared memory pointer.
    char *smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H G M M A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Col Major with fp16/fp32 acc, A coming from SMEM
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_a<Traits,
                          Cta_tile,
                          Col,
                          BUFFERS_PER_TILE_,
                          desc_mode,
                          false,
                          xmma::Gmma_fusion_mode::NO_FUSION,
                          false>
    : public Smem_tile_hopper_gmma_col_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode> {

    // The base class.
    using Base = Smem_tile_hopper_gmma_col_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_a( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Col Major with fp16/fp32 acc, A coming from RF, B coming SMEM
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_a<Traits,
                          Cta_tile,
                          Col,
                          BUFFERS_PER_TILE_,
                          desc_mode,
                          true,
                          xmma::Gmma_fusion_mode::NO_FUSION,
                          false>
    : public Smem_tile_hopper_gmma_col_a_rf<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode> {

    // The base class.
    using Base = Smem_tile_hopper_gmma_col_a_rf<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_a( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Row Major with fp16/fp32 acc, A coming from SMEM
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_a<Traits,
                          Cta_tile,
                          Row,
                          BUFFERS_PER_TILE_,
                          desc_mode,
                          false,
                          xmma::Gmma_fusion_mode::NO_FUSION,
                          false>
    : public Smem_tile_hopper_gmma_row_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode> {

    // The base class.
    using Base = Smem_tile_hopper_gmma_row_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_a( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// A Row Major with fp16/fp32 acc, A coming from RF, B coming from SMEM
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_a<Traits, Cta_tile, Row, BUFFERS_PER_TILE_, desc_mode, true>
    : public Smem_tile_hopper_gmma_row_a_rf<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode> {

    // The base class.
    using Base = Smem_tile_hopper_gmma_row_a_rf<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_a( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// A Row Major with fp16/fp32 acc, A coming from RF, Fuse A, B coming from SMEM
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_a<Traits,
                          Cta_tile,
                          Row,
                          BUFFERS_PER_TILE_,
                          desc_mode,
                          true,
                          xmma::Gmma_fusion_mode::BN_APPLY>
    : public Smem_tile_hopper_gmma_row_a_rf_bn_apply<Traits,
                                                     Cta_tile,
                                                     BUFFERS_PER_TILE_,
                                                     desc_mode> {

    // The base class.
    using Base =
        Smem_tile_hopper_gmma_row_a_rf_bn_apply<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_a( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// B Col Major with fp16/fp32 acc, B coming from SMEM
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_b<Traits, Cta_tile, Col, BUFFERS_PER_TILE_, desc_mode, false>
    : public Smem_tile_hopper_gmma_col_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode> {

    // The base class.
    using Base = Smem_tile_hopper_gmma_col_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_b( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// B Row Major with fp16/fp32 acc, B coming from SMEM
template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_b<Traits, Cta_tile, Row, BUFFERS_PER_TILE_, desc_mode, false>
    : public Smem_tile_hopper_gmma_row_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode> {

    // The base class.
    using Base = Smem_tile_hopper_gmma_row_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    enum { BYTES_PER_ELEMENT = 8 };

    // Ctor.
    inline __device__ Smem_tile_hopper_b( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_tma_col_a {
    // Lwrrently Interleaved Mode is not implemented.
    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_NONE,
                   "Lwrrently, Interleaved Mode is not implemented.\n" );

    // HGMMA operation, where A and B should be in fp16.
    static_assert( sizeof( typename Traits::A_type ) == 2 && sizeof( typename Traits::B_type ) == 2,
                   "HGMMA operation, where A and B should be in fp16 is required.\n" );

    // For SWIZZLE_64B, col a is not needed/implemented
    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_64B,
                   "Lwrrently, for SWIZZLE_64B mode, col_a is not needed/implemented. \n" );

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The fragment.
    using Fragment = Fragment_a<Traits, Col>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr xmma::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP =
        xmma::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = xmma::Gmma_descriptor_a<xmma::Gmma_descriptor_transpose::TRANS,
                                                    desc_mode,
                                                    Cta_tile,
                                                    Traits::BITS_PER_ELEMENT_A,
                                                    Traits::GMMA_M,
                                                    Traits::GMMA_N,
                                                    Traits::GMMA_K,
                                                    GMMA_DESC_SIZE_PER_GROUP>;

    // the size in bits of each element
    enum { BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_A };
    // the size of bytes of each element
    enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8 };
    // The size in bytes of a single LDGSTS/STS.
    enum { BYTES_PER_STS = 16 };
    // The number of elements per LDGSTS/STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for
    // SWIZZLE_128B and SWIZZLE_64B format
    enum { BYTES_PER_COLUMN = 128 };
    // the number of columns per one column of M due the the limiation of leading
    // dim size
    enum {
        NUM_COLS_PER_M =
            ( Cta_tile::M * BYTES_PER_ELEMENT + BYTES_PER_COLUMN - 1 ) / BYTES_PER_COLUMN
    };
    // the number of columns per one column of M_PER_GMMA_GROUP
    enum {
        NUM_COLS_PER_GMMA_GROUP_M =
            ( Xmma_tile::M_PER_GMMA_GROUP * BYTES_PER_ELEMENT + BYTES_PER_COLUMN - 1 ) /
            BYTES_PER_COLUMN
    };
    // for 64xNx16 GMMA shape, NUM_ROWS_PER_GMMA_GROUP_M must be 1.
    static_assert( NUM_COLS_PER_GMMA_GROUP_M == 1,
                   "for 64xNx16 GMMA shape, NUM_ROWS_PER_GMMA_GROUP_M must be 1.\n" );
    // Number of SMEM columns
    enum { NUM_COLUMNS = Cta_tile::K * NUM_COLS_PER_M };
    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = NUM_COLUMNS * BYTES_PER_COLUMN };
    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum { BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16 };
    // this is needed to decrement GMMA desc
    enum {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB =
            BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };
    // The number of buffers.
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // The size in bytes of total buffers.
    // +128 byte to guarantee that the base address can be aligned to 128B
    enum { BYTES_FOR_ALIGNMENT = 128 };
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE };
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER };
    // The number of threads needed to store a column
    enum { THREADS_PER_COLUMN = BYTES_PER_COLUMN / BYTES_PER_STS };
    // The number of columns written with a single STS.
    enum { COLUMNS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_COLUMN };
    // for swizzle_128B the xor factor is 8
    enum { COLUMNS_PER_XOR_PATTERN = 8 };
    // The distance in byte between different GMMA groups (might need multiple due
    // to cta tile size) each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum {
        GMMA_GROUP_SMEM_DISTANCE =
            Xmma_tile::K_PER_GMMA_GROUP * NUM_COLS_PER_GMMA_GROUP_M * BYTES_PER_COLUMN
    };

    enum { M_ = Cta_tile::K };

    enum { N_ = Cta_tile::M };

    enum { N_WITH_PADDING = Next_power_of_two<N_>::VALUE };
    // The number of bytes per row without packing of rows.
    enum { BYTES_PER_ROW_BEFORE_PACKING = N_WITH_PADDING * BITS_PER_ELEMENT / 8 };
    // The number of bytes per row -- we want at least 128B per row.
    enum { BYTES_PER_ROW_ = Max<BYTES_PER_ROW_BEFORE_PACKING, 128>::VALUE };
    // The number of rows in shared memory (two rows may be packed into a single one).
    enum { ROWS = M_ * BYTES_PER_ROW_BEFORE_PACKING / BYTES_PER_ROW_ };

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_tma_col_a( char *smem, int tidx ) : smem_( smem ) {
        this->smem_write_offset_ = 0;
        this->smem_barrier_offset_ = 0;
    }

    inline __device__ void add_smem_barrier_base( uint64_t *smem_barrier ) {
        this->smem_barrier_ = smem_barrier;
        this->smem_barrier_offset_ = __cvta_generic_to_shared( this->smem_barrier_ );
    }

    // set the scale and bias smem pointer
    // do nothing.
    inline __device__ void set_scale_bias_smem_ptr( char *scale_bias_smem_ptr, int tidx, int k ) {
    }
    // Load from shared memory.
    // LDSM is not required is both operands coming from SMEM for GMMA
    // however, for some cases it is needed
    inline __device__ void load( Fragment ( &a )[Xmma_tile::XMMAS_M], int ki ) {
    }

    // Compute the store pointers.
    template <int LDGS_M, int LDGS_N>
    inline __device__ void compute_store_pointers( uint32_t ( &ptrs )[LDGS_M * LDGS_N] ) {
#pragma unroll
        for( int ni = 0; ni < LDGS_N; ++ni ) {
            int offset_n = smem_write_offset_ + ni * COLUMNS_PER_STS * BYTES_PER_COLUMN;
#pragma unroll
            for( int mi = 0; mi < LDGS_M; ++mi ) {
                // LDGS_M > 1 means there is reshaping of the smem going on
                // for example a column major 128x64 block in smem
                // is now reshaped into a column major 64x128 block (not transpose)
                // see spreadsheet for more details
                int offset_m = mi * BYTES_PER_COLUMN * Cta_tile::K;
                int offset = offset_m + offset_n;
                ptrs[ni * LDGS_M + mi] = get_smem_pointer( &smem_[offset] );
            }
        }
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int buffer = 0; buffer < BYTES_PER_TILE; buffer += BYTES_PER_BUFFER ) {
            for( int row = 0; row < ROWS; ++row ) {
                for( int col = 0; col < BYTES_PER_ROW_; col += 4 ) {
                    if( threadIdx.x == 0 ) {
                        uint32_t val;
                        lds( val,
                             static_cast<unsigned>( __cvta_generic_to_shared( smem_ ) ) +
                                 row * BYTES_PER_ROW_ + col + buffer );
                        printf( "block=(x=%2d, y=%2d, z=%2d) (smem_=%2d, buffer=%2d, row=%2d, "
                                "byte=%4d)=0x%08x\n",
                                blockIdx.x,
                                blockIdx.y,
                                blockIdx.z,
                                static_cast<unsigned>( __cvta_generic_to_shared( smem_ ) ),
                                buffer,
                                row,
                                col,
                                val );
                    }
                }
            }
        }
    }

    // Store to the tile in shared memory.
    template <int LDGS_M,
              int LDGS_N,
              int M,
              int K = 1,
              typename LDGSTS_CFG = xmma::Ldgsts_config<true>>
    inline __device__ void store( const void *( &gmem_ptrs )[LDGS_M * LDGS_N],
                                  uint32_t ( &preds )[M],
                                  uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        uint32_t smem_ptrs[LDGS_M * LDGS_N];
        this->compute_store_pointers<LDGS_M, LDGS_N>( smem_ptrs );
        ldgsts<LDGS_M * LDGS_N, M, 16 / K, LDGSTS_CFG>( smem_ptrs, gmem_ptrs, preds, mem_desc );
    }

    template <uint32_t DIM, lwdaTmaDescType DESC_TYPE, unsigned COPY_BYTES, int USE_TMA_MULTICAST= 0>
    inline __device__ void store( const void *p_desc,
                                  const unsigned &smem_offset,
                                  int32_t coord0,
                                  int32_t coord1,
                                  int32_t coord2,
                                  int32_t coord3,
                                  int32_t coord4,
                                  uint32_t off_w = 0,
                                  uint32_t off_h = 0,
                                  uint32_t off_d = 0,
                                  uint16_t mcast_cta_mask = 0 ) {
            unsigned smem = xmma::hopper::emu::set_shared_data_address(
                static_cast<uint32_t>( __cvta_generic_to_shared( this->smem_ ) ) +
                this->smem_write_offset_ + smem_offset );

            xmma::utmaldg<DIM, DESC_TYPE, USE_TMA_MULTICAST>( reinterpret_cast<const lwdaTmaDescv2 *>( p_desc ),
                                           smem,
                                           unsigned( this->smem_barrier_offset_ ),
                                           coord0,
                                           coord1,
                                           coord2,
                                           coord3,
                                           coord4,
                                           off_w,
                                           off_h,
                                           off_d,
                                           mcast_cta_mask );
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                                            ? -BYTES_PER_TILE_INC_BOUNDARY
                                            : BYTES_PER_BUFFER;
            this->smem_barrier_offset_ +=
                ( this->smem_barrier_offset_ >= BUFFERS_PER_TILE * 8 ) ? -BUFFERS_PER_TILE * 8 : 8;
        }
    }

    inline __device__ void move_next_write_buffer( int buffer_id ) {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                                            ? -BYTES_PER_TILE_INC_BOUNDARY
                                            : BYTES_PER_BUFFER;
        }
        this->smem_barrier_offset_ = __cvta_generic_to_shared( this->smem_barrier_ + buffer_id );
    }

    // Move the read offset to next buffer.
    // do nothing, as it is controled by gmma desc
    inline __device__ void move_next_read_buffer() {
    }

    // The shared memory pointer.
    char *smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The shared memory barrier base pointer
    uint64_t *smem_barrier_;
    // The shared memory barrier offset.
    uint32_t smem_barrier_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_a<Traits,
                          Cta_tile,
                          Col,
                          BUFFERS_PER_TILE_,
                          desc_mode,
                          false,
                          xmma::Gmma_fusion_mode::NO_FUSION,
                          true>
    : public Smem_tile_hopper_gmma_tma_col_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode> {

    // The base class.
    using Base = Smem_tile_hopper_gmma_tma_col_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_a( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_tma_row_a {

    // Lwrrently Interleaved Mode is not implemented.
    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_NONE,
                   "Lwrrently, Interleaved Mode is not implemented.\n" );

    // HGMMA operation, where A and B should be in fp16.
    static_assert( sizeof( typename Traits::A_type ) == 2 && sizeof( typename Traits::B_type ) == 2,
                   "HGMMA operation, where A and B should be in fp16 is required.\n" );

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr xmma::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP =
        xmma::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = xmma::Gmma_descriptor_a<xmma::Gmma_descriptor_transpose::NOTRANS,
                                                    desc_mode,
                                                    Cta_tile,
                                                    Traits::BITS_PER_ELEMENT_A,
                                                    Traits::GMMA_M,
                                                    Traits::GMMA_N,
                                                    Traits::GMMA_K,
                                                    GMMA_DESC_SIZE_PER_GROUP>;

    // the size in bits of each element
    enum { BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_A };
    // the size of bytes of each element
    enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8 };
    // The size in bytes of a single LDGSTS/STS.
    enum { BYTES_PER_STS = 16 };
    // The number of elements per LDGSTS/STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for
    // SWIZZLE_128B and SWIZZLE_64B format
    enum { BYTES_PER_ROW = 128 };
    // the number of rows per one row of K due the the limiation of leading dim
    // size

    enum { M_ = Cta_tile::K };

    enum { N_ = Cta_tile::M };

    enum { N_WITH_PADDING = Next_power_of_two<N_>::VALUE };
    // The number of bytes per row without packing of rows.
    enum { BYTES_PER_ROW_BEFORE_PACKING = N_WITH_PADDING * BITS_PER_ELEMENT / 8 };
    // The number of bytes per row -- we want at least 128B per row.
    enum { BYTES_PER_ROW_ = Max<BYTES_PER_ROW_BEFORE_PACKING, 128>::VALUE };
    // The number of rows in shared memory (two rows may be packed into a single one).
    enum { ROWS = M_ * BYTES_PER_ROW_BEFORE_PACKING / BYTES_PER_ROW_ };

    enum {
        NUM_ROWS_PER_K = ( Cta_tile::K * BYTES_PER_ELEMENT + BYTES_PER_ROW - 1 ) / BYTES_PER_ROW
    };

    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_64B || Cta_tile::K == 32,
                   "swizzle_64B row_a is valid if kblock=32\n" );
    // Number of SMEM rows
    enum {
        NUM_ROWS = ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B )
                       ? ( Cta_tile::M * NUM_ROWS_PER_K )
                       : ( Cta_tile::M / 2 )
    };
    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = NUM_ROWS * BYTES_PER_ROW };
    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum { BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16 };
    // this is needed to decrement GMMA desc
    enum {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB =
            BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };
    // The number of buffers.
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // The size in bytes of total buffers.
    enum { BYTES_FOR_ALIGNMENT = 128 };
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE };
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER };
    // The number of threads needed to store a row
    enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STS };
    // The number of rows written with a single STS.
    enum { ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // for swizzle_128B the xor factor is 8
    enum {
        ROWS_PER_XOR_PATTERN = ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B ) ? 8 : 4
    };
    // The distance in byte between different GMMA groups (might need multiple due
    // to cta tile size) each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum {
        GMMA_GROUP_SMEM_DISTANCE =
            Xmma_tile::M_PER_GMMA_GROUP /
            ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B ? 1 : 2 ) * BYTES_PER_ROW
    };

    // Ctor.
    // colwert the pointer from char* to uint32_t
    inline __device__ Smem_tile_hopper_gmma_tma_row_a( char *smem, int tidx ) : smem_( smem ) {
        this->smem_write_offset_ = 0;
        this->smem_barrier_offset_ = 0;
    }

    inline __device__ void add_smem_barrier_base( uint64_t *smem_barrier ) {
        this->smem_barrier_ = smem_barrier;
        this->smem_barrier_offset_ = __cvta_generic_to_shared( this->smem_barrier_ );
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int buffer = 0; buffer < BYTES_PER_TILE; buffer += BYTES_PER_BUFFER ) {
            for( int row = 0; row < ROWS; ++row ) {
                for( int col = 0; col < BYTES_PER_ROW_; col += 4 ) {
                    if( threadIdx.x == 0 ) {
                        uint32_t val;
                        lds( val,
                             static_cast<unsigned>( __cvta_generic_to_shared( smem_ ) ) +
                                 row * BYTES_PER_ROW_ + col + buffer );
                        printf( "block=(x=%2d, y=%2d, z=%2d) (smem_=%2d, buffer=%2d, row=%2d, "
                                "byte=%4d)=0x%08x\n",
                                blockIdx.x,
                                blockIdx.y,
                                blockIdx.z,
                                static_cast<unsigned>( __cvta_generic_to_shared( smem_ ) ),
                                buffer,
                                row,
                                col,
                                val );
                    }
                }
            }
        }
    }

    // set the scale and bias smem pointer
    // do nothing.
    inline __device__ void set_scale_bias_smem_ptr( char *scale_bias_smem_ptr, int tidx, int k ) {
    }

    // Load from shared memory.
    inline __device__ void load( Fragment ( &a )[Xmma_tile::XMMAS_M], int ki ) {
    }
    // Compute the store pointers.
    template <int LDGS_M, int LDGS_N>
    inline __device__ void compute_store_pointers( uint32_t ( &ptrs )[LDGS_M * LDGS_N] ) {
#pragma unroll
        for( int ni = 0; ni < LDGS_N; ++ni ) {
            int offset_n = smem_write_offset_ + ni * ROWS_PER_STS * BYTES_PER_ROW;
#pragma unroll
            for( int mi = 0; mi < LDGS_M; ++mi ) {
                int offset_m = mi * BYTES_PER_ROW * Cta_tile::M;
                int offset = offset_m + offset_n;
                ptrs[ni * LDGS_M + mi] = get_smem_pointer( &smem_[offset] );
            }
        }
    }

    // Store to the tile in shared memory.
    template <int LDGS_M,
              int LDGS_N,
              int M,
              int K = 1,
              typename LDGSTS_CFG = xmma::Ldgsts_config<true>>
    inline __device__ void store( const void *( &gmem_ptrs )[LDGS_M * LDGS_N],
                                  uint32_t ( &preds )[M],
                                  uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        uint32_t smem_ptrs[LDGS_M * LDGS_N];
        this->compute_store_pointers<LDGS_M, LDGS_N>( smem_ptrs );
        ldgsts<LDGS_M * LDGS_N, M, 16 / K, LDGSTS_CFG>( smem_ptrs, gmem_ptrs, preds, mem_desc );
    }

    template <uint32_t DIM, lwdaTmaDescType DESC_TYPE, unsigned COPY_BYTES, int USE_TMA_MULTICAST = 0>
    inline __device__ void store( const void *p_desc,
                                  const unsigned &smem_offset,
                                  int32_t coord0,
                                  int32_t coord1,
                                  int32_t coord2,
                                  int32_t coord3,
                                  int32_t coord4,
                                  uint32_t off_w = 0,
                                  uint32_t off_h = 0,
                                  uint32_t off_d = 0,
                                  uint16_t mcast_cta_mask = 0 ) {
            unsigned smem = xmma::hopper::emu::set_shared_data_address(
                static_cast<uint32_t>( __cvta_generic_to_shared( this->smem_ ) ) +
                this->smem_write_offset_ + smem_offset );

            xmma::utmaldg<DIM, DESC_TYPE, USE_TMA_MULTICAST>( reinterpret_cast<const lwdaTmaDescv2 *>( p_desc ),
                                           smem,
                                           unsigned( this->smem_barrier_offset_ ),
                                           coord0,
                                           coord1,
                                           coord2,
                                           coord3,
                                           coord4,
                                           off_w,
                                           off_h,
                                           off_d,
                                           mcast_cta_mask );
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                                            ? -BYTES_PER_TILE_INC_BOUNDARY
                                            : BYTES_PER_BUFFER;
            this->smem_barrier_offset_ +=
                ( this->smem_barrier_offset_ >= BUFFERS_PER_TILE * 8 ) ? -BUFFERS_PER_TILE * 8 : 8;
        }
    }

    inline __device__ void move_next_write_buffer( int buffer_id ) {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                                            ? -BYTES_PER_TILE_INC_BOUNDARY
                                            : BYTES_PER_BUFFER;
        }
        this->smem_barrier_offset_ = __cvta_generic_to_shared( this->smem_barrier_ + buffer_id );
    }

    // Move the read offset to next buffer.
    // do nothing, as it is controled by gmma desc
    inline __device__ void move_next_read_buffer() {
    }

    // The shared memory pointer.
    char *smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The shared memory barrier base pointer.
    uint64_t *smem_barrier_;
    // The shared memory barrier offset.
    uint32_t smem_barrier_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_a<Traits,
                          Cta_tile,
                          Row,
                          BUFFERS_PER_TILE_,
                          desc_mode,
                          false,
                          xmma::Gmma_fusion_mode::NO_FUSION,
                          true>
    : public Smem_tile_hopper_gmma_tma_row_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode> {

    // The base class.
    using Base = Smem_tile_hopper_gmma_tma_row_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_a( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_tma_row_b {

    // Lwrrently Interleaved Mode is not implemented.
    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_NONE,
                   "Lwrrently, Interleaved Mode is not implemented.\n" );

    // HGMMA operation, where A and B should be in fp16.
    static_assert( sizeof( typename Traits::A_type ) == 2 && sizeof( typename Traits::B_type ) == 2,
                   "HGMMA operation, where A and B should be in fp16 is required.\n" );

    // For SWIZZLE_64B, row b is not needed/implemented
    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_64B,
                   "Lwrrently, for SWIZZLE_64B mode, row_b is not needed/implemented. \n" );

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The fragment.
    using Fragment = Fragment_b<Traits, Row>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr xmma::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP =
        xmma::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = xmma::Gmma_descriptor_b<xmma::Gmma_descriptor_transpose::TRANS,
                                                    desc_mode,
                                                    Cta_tile,
                                                    Traits::BITS_PER_ELEMENT_B,
                                                    Traits::GMMA_M,
                                                    Traits::GMMA_N,
                                                    Traits::GMMA_K,
                                                    GMMA_DESC_SIZE_PER_GROUP>;

    // the size in bits of each element
    enum { BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_B };
    // the size of bytes of each element
    enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8 };
    // The size in bytes of a single LDGSTS/STS.
    enum { BYTES_PER_STS = 16 };
    // The number of elements per LDGSTS/STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for
    // SWIZZLE_128B and SWIZZLE_64B format
    enum { BYTES_PER_ROW = 128 };
    // the number of rows per one row of N due the the limiation of leading dim
    // size
    enum {
        NUM_ROWS_PER_N = ( Cta_tile::N * BYTES_PER_ELEMENT + BYTES_PER_ROW - 1 ) / BYTES_PER_ROW
    };
    // the number of rows per one row of N_PER_GMMA_GROUP
    enum {
        NUM_ROWS_PER_GMMA_GROUP_N =
            ( Xmma_tile::N_PER_GMMA_GROUP * BYTES_PER_ELEMENT + BYTES_PER_ROW - 1 ) / BYTES_PER_ROW
    };
    // Number of SMEM rows
    enum { NUM_ROWS = Cta_tile::K * NUM_ROWS_PER_N };
    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = NUM_ROWS * BYTES_PER_ROW };
    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum { BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16 };
    // this is needed to decrement GMMA desc
    enum {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB =
            BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };
    // The number of buffers.
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // The size in bytes of total buffers.
    enum { BYTES_FOR_ALIGNMENT = 128 };
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE };
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER };
    // The number of threads needed to store a row
    enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STS };
    // The number of rows written with a single STS.
    enum { ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // for swizzle_128B the xor factor is 8
    enum { ROWS_PER_XOR_PATTERN = 8 };
    // The distance in byte between different GMMA groups (might need multiple due
    // to cta tile size) each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum {
        GMMA_GROUP_SMEM_DISTANCE =
            Xmma_tile::K_PER_GMMA_GROUP * NUM_ROWS_PER_GMMA_GROUP_N * BYTES_PER_ROW
    };

    enum { M_ = Cta_tile::N };

    enum { N_ = Cta_tile::K };

    enum { N_WITH_PADDING = Next_power_of_two<N_>::VALUE };
    // The number of bytes per row without packing of rows.
    enum { BYTES_PER_ROW_BEFORE_PACKING = N_WITH_PADDING * BITS_PER_ELEMENT / 8 };
    // The number of bytes per row -- we want at least 128B per row.
    enum { BYTES_PER_ROW_ = Max<BYTES_PER_ROW_BEFORE_PACKING, 128>::VALUE };
    // The number of rows in shared memory (two rows may be packed into a single one).
    enum { ROWS = M_ * BYTES_PER_ROW_BEFORE_PACKING / BYTES_PER_ROW_ };

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_tma_row_b( char *smem, int tidx ) : smem_( smem ) {
        this->smem_write_offset_ = 0;
        this->smem_barrier_offset_ = 0;
    }

    inline __device__ void add_smem_barrier_base( uint64_t *smem_barrier ) {
        this->smem_barrier_ = smem_barrier;
        this->smem_barrier_offset_ = __cvta_generic_to_shared( this->smem_barrier_ );
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int buffer = 0; buffer < BYTES_PER_TILE; buffer += BYTES_PER_BUFFER ) {
            for( int row = 0; row < ROWS; ++row ) {
                for( int col = 0; col < BYTES_PER_ROW_; col += 4 ) {
                    if( threadIdx.x == 0 ) {
                        uint32_t val;
                        lds( val,
                             static_cast<unsigned>( __cvta_generic_to_shared( smem_ ) ) +
                                 row * BYTES_PER_ROW_ + col + buffer );
                        printf( "block=(x=%2d, y=%2d, z=%2d) (smem_=%2d, buffer=%2d, row=%2d, "
                                "byte=%4d)=0x%08x\n",
                                blockIdx.x,
                                blockIdx.y,
                                blockIdx.z,
                                static_cast<unsigned>( __cvta_generic_to_shared( smem_ ) ),
                                buffer,
                                row,
                                col,
                                val );
                    }
                }
            }
        }
    }

    // Load from shared memory.
    inline __device__ void load( Fragment ( &b )[Xmma_tile::XMMAS_N], int ki ) {
    }

    // Compute the store pointers.
    template <int LDGS_M, int LDGS_N>
    inline __device__ void compute_store_pointers( uint32_t ( &ptrs )[LDGS_M * LDGS_N] ) {
#pragma unroll
        for( int ni = 0; ni < LDGS_N; ++ni ) {
            int offset_n = smem_write_offset_ + ni * ROWS_PER_STS * BYTES_PER_ROW;
#pragma unroll
            for( int mi = 0; mi < LDGS_M; ++mi ) {
                int offset_m = mi * BYTES_PER_ROW * Cta_tile::K;
                int offset = offset_m + offset_n;
                ptrs[ni * LDGS_M + mi] = get_smem_pointer( &smem_[offset] );
            }
        }
    }

    // Store to the tile in shared memory.
    template <int LDGS_M,
              int LDGS_N,
              int M,
              int K = 1,
              typename LDGSTS_CFG = xmma::Ldgsts_config<true>>
    inline __device__ void store( const void *( &gmem_ptrs )[LDGS_M * LDGS_N],
                                  uint32_t ( &preds )[M],
                                  uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        uint32_t smem_ptrs[LDGS_M * LDGS_N];
        this->compute_store_pointers<LDGS_M, LDGS_N>( smem_ptrs );
        ldgsts<LDGS_M * LDGS_N, M, 16 / K, LDGSTS_CFG>( smem_ptrs, gmem_ptrs, preds, mem_desc );
    }

    template <uint32_t DIM, lwdaTmaDescType DESC_TYPE, unsigned COPY_BYTES, int USE_TMA_MULTICAST = 0>
    inline __device__ void store( const void *p_desc,
                                  const unsigned &smem_offset,
                                  int32_t coord0,
                                  int32_t coord1,
                                  int32_t coord2,
                                  int32_t coord3,
                                  int32_t coord4,
                                  uint32_t off_w = 0,
                                  uint32_t off_h = 0,
                                  uint32_t off_d = 0,
                                  uint16_t mcast_cta_mask = 0 ) {
            unsigned smem = xmma::hopper::emu::set_shared_data_address(
                static_cast<uint32_t>( __cvta_generic_to_shared( this->smem_ ) ) +
                this->smem_write_offset_ + smem_offset );
    
            xmma::utmaldg<DIM, DESC_TYPE, USE_TMA_MULTICAST>( reinterpret_cast<const lwdaTmaDescv2 *>( p_desc ),
                                           smem,
                                           unsigned( this->smem_barrier_offset_ ),
                                           coord0,
                                           coord1,
                                           coord2,
                                           coord3,
                                           coord4,
                                           off_w,
                                           off_h,
                                           off_d,
                                           mcast_cta_mask );
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                                            ? -BYTES_PER_TILE_INC_BOUNDARY
                                            : BYTES_PER_BUFFER;
            this->smem_barrier_offset_ +=
                ( this->smem_barrier_offset_ >= BUFFERS_PER_TILE * 8 ) ? -BUFFERS_PER_TILE * 8 : 8;
        }
    }

    inline __device__ void move_next_write_buffer( int buffer_id ) {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                                            ? -BYTES_PER_TILE_INC_BOUNDARY
                                            : BYTES_PER_BUFFER;
        }
        this->smem_barrier_offset_ = __cvta_generic_to_shared( this->smem_barrier_ + buffer_id );
    }

    // Move the read offset to next buffer.
    // do nothing, as it is controled by gmma desc
    inline __device__ void move_next_read_buffer() {
    }

    // The shared memory pointer.
    char *smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The shared memory barrier base pointer.
    uint64_t *smem_barrier_;
    // The shared memory barrier offset.
    uint32_t smem_barrier_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_b<Traits, Cta_tile, Row, BUFFERS_PER_TILE_, desc_mode, true>
    : public Smem_tile_hopper_gmma_tma_row_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode> {

    // The base class.
    using Base = Smem_tile_hopper_gmma_tma_row_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_b( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_tma_col_b {

    // Lwrrently Interleaved Mode is not implemented.
    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_NONE,
                   "Lwrrently, Interleaved Mode is not implemented.\n" );

    // HGMMA operation, where A and B should be in fp16.
    static_assert( sizeof( typename Traits::A_type ) == 2 && sizeof( typename Traits::B_type ) == 2,
                   "HGMMA operation, where A and B should be in fp16 is required.\n" );

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr xmma::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP =
        xmma::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = xmma::Gmma_descriptor_b<xmma::Gmma_descriptor_transpose::NOTRANS,
                                                    desc_mode,
                                                    Cta_tile,
                                                    Traits::BITS_PER_ELEMENT_B,
                                                    Traits::GMMA_M,
                                                    Traits::GMMA_N,
                                                    Traits::GMMA_K,
                                                    GMMA_DESC_SIZE_PER_GROUP>;

    // the size in bits of each element
    enum { BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_B };
    // the size of bytes of each element
    enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8 };
    // The size in bytes of a single LDGSTS/STS.
    enum { BYTES_PER_STS = 16 };
    // The number of elements per LDGSTS/STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for
    // SWIZZLE_128B and SWIZZLE_64B format
    enum { BYTES_PER_COLUMN = 128 };

    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_64B || Cta_tile::K == 32,
                   "swizzle_64B col_b is valid if kblock=32\n" );
    // the number of columns per one column of K due the the limiation of leading
    // dim size
    enum {
        NUM_COLS_PER_K =
            ( Cta_tile::K * BYTES_PER_ELEMENT + BYTES_PER_COLUMN - 1 ) / BYTES_PER_COLUMN
    };
    // Number of SMEM columns
    enum {
        NUM_COLUMNS = ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B )
                          ? Cta_tile::N * NUM_COLS_PER_K
                          : Cta_tile::N / 2
    };
    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = NUM_COLUMNS * BYTES_PER_COLUMN };
    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum { BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16 };
    // this is needed to decrement GMMA desc
    enum {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB =
            BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };
    // The number of buffers.
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // The size in bytes of total buffers.
    enum { BYTES_FOR_ALIGNMENT = 128 };
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE };
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER };
    // The number of threads needed to store a column
    enum { THREADS_PER_COLUMN = BYTES_PER_COLUMN / BYTES_PER_STS };
    // The number of columns written with a single STS.
    enum { COLUMNS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_COLUMN };
    // for swizzle_128B the xor factor is 8
    enum {
        COLUMNS_PER_XOR_PATTERN = ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B ) ? 8 : 4
    };
    // The distance in byte between different GMMA groups (might need multiple due
    // to cta tile size) each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum {
        GMMA_GROUP_SMEM_DISTANCE =
            Xmma_tile::N_PER_GMMA_GROUP /
            ( desc_mode == xmma::Gmma_descriptor_mode::SWIZZLE_128B ? 1 : 2 ) * BYTES_PER_COLUMN
    };

    enum { M_ = Cta_tile::K };

    enum { N_ = Cta_tile::N };

    enum { N_WITH_PADDING = Next_power_of_two<N_>::VALUE };
    // The number of bytes per row without packing of rows.
    enum { BYTES_PER_ROW_BEFORE_PACKING = N_WITH_PADDING * BITS_PER_ELEMENT / 8 };
    // The number of bytes per row -- we want at least 128B per row.
    enum { BYTES_PER_ROW_ = Max<BYTES_PER_ROW_BEFORE_PACKING, 128>::VALUE };
    // The number of rows in shared memory (two rows may be packed into a single one).
    enum { ROWS = M_ * BYTES_PER_ROW_BEFORE_PACKING / BYTES_PER_ROW_ };

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_tma_col_b( char *smem, int tidx ) : smem_( smem ) {
        this->smem_write_offset_ = 0;
        this->smem_barrier_offset_ = 0;
    }

    inline __device__ void add_smem_barrier_base( uint64_t *smem_barrier ) {
        this->smem_barrier_ = smem_barrier;
        this->smem_barrier_offset_ = __cvta_generic_to_shared( this->smem_barrier_ );
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int buffer = 0; buffer < BYTES_PER_TILE; buffer += BYTES_PER_BUFFER ) {
            for( int row = 0; row < ROWS; ++row ) {
                for( int col = 0; col < BYTES_PER_ROW_; col += 4 ) {
                    if( threadIdx.x == 0 ) {
                        uint32_t val;
                        lds( val,
                             static_cast<unsigned>( __cvta_generic_to_shared( smem_ ) ) +
                                 row * BYTES_PER_ROW_ + col + buffer );
                        printf( "block=(x=%2d, y=%2d, z=%2d) (smem_=%2d, buffer=%2d, row=%2d, "
                                "byte=%4d)=0x%08x\n",
                                blockIdx.x,
                                blockIdx.y,
                                blockIdx.z,
                                static_cast<unsigned>( __cvta_generic_to_shared( smem_ ) ),
                                buffer,
                                row,
                                col,
                                val );
                    }
                }
            }
        }
    }

    // Load from shared memory.
    inline __device__ void load( Fragment ( &b )[Xmma_tile::XMMAS_N], int ki ) {
    }

    // Compute the store pointers.
    template <int LDGS_M, int LDGS_N>
    inline __device__ void compute_store_pointers( uint32_t ( &ptrs )[LDGS_M * LDGS_N] ) {
#pragma unroll
        for( int ni = 0; ni < LDGS_N; ++ni ) {
            int offset_n = smem_write_offset_ + ni * COLUMNS_PER_STS * BYTES_PER_COLUMN;
#pragma unroll
            for( int mi = 0; mi < LDGS_M; ++mi ) {
                int offset_m = mi * BYTES_PER_COLUMN * Cta_tile::N;
                int offset = offset_m + offset_n;
                ptrs[ni * LDGS_M + mi] = get_smem_pointer( &smem_[offset] );
            }
        }
    }

    // Store to the tile in shared memory.
    template <int LDGS_M,
              int LDGS_N,
              int M,
              int K = 1,
              typename LDGSTS_CFG = xmma::Ldgsts_config<true>>
    inline __device__ void store( const void *( &gmem_ptrs )[LDGS_M * LDGS_N],
                                  uint32_t ( &preds )[M],
                                  uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        uint32_t smem_ptrs[LDGS_M * LDGS_N];
        this->compute_store_pointers<LDGS_M, LDGS_N>( smem_ptrs );
        ldgsts<LDGS_M * LDGS_N, M, 16 / K, LDGSTS_CFG>( smem_ptrs, gmem_ptrs, preds, mem_desc );
    }

    template <uint32_t DIM, lwdaTmaDescType DESC_TYPE, unsigned COPY_BYTES, int USE_TMA_MULTICAST = 0>
    inline __device__ void store( const void *p_desc,
                                  const unsigned &smem_offset,
                                  int32_t coord0,
                                  int32_t coord1,
                                  int32_t coord2,
                                  int32_t coord3,
                                  int32_t coord4,
                                  uint32_t off_w = 0,
                                  uint32_t off_h = 0,
                                  uint32_t off_d = 0,
                                  uint16_t mcast_cta_mask = 0 ) {
            unsigned smem = xmma::hopper::emu::set_shared_data_address(
                static_cast<uint32_t>( __cvta_generic_to_shared( this->smem_ ) ) +
                this->smem_write_offset_ + smem_offset );

            xmma::utmaldg<DIM, DESC_TYPE, USE_TMA_MULTICAST>( reinterpret_cast<const lwdaTmaDescv2 *>( p_desc ),
                                           smem,
                                           unsigned( this->smem_barrier_offset_ ),
                                           coord0,
                                           coord1,
                                           coord2,
                                           coord3,
                                           coord4,
                                           off_w,
                                           off_h,
                                           off_d,
                                           mcast_cta_mask );
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                                            ? -BYTES_PER_TILE_INC_BOUNDARY
                                            : BYTES_PER_BUFFER;
            this->smem_barrier_offset_ +=
                ( this->smem_barrier_offset_ >= BUFFERS_PER_TILE * 8 ) ? -BUFFERS_PER_TILE * 8 : 8;
        }
    }

    inline __device__ void move_next_write_buffer( int buffer_id ) {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                                            ? -BYTES_PER_TILE_INC_BOUNDARY
                                            : BYTES_PER_BUFFER;
        }
        this->smem_barrier_offset_ = __cvta_generic_to_shared( this->smem_barrier_ + buffer_id );
    }

    // Move the read offset to next buffer.
    // do nothing, as it is controled by gmma desc
    inline __device__ void move_next_read_buffer() {
    }

    // The shared memory pointer.
    char *smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The shared memory barrier base pointer.
    uint64_t *smem_barrier_;
    // The shared memory barrier offset.
    uint32_t smem_barrier_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          int BUFFERS_PER_TILE_,
          xmma::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_b<Traits, Cta_tile, Col, BUFFERS_PER_TILE_, desc_mode, true>
    : public Smem_tile_hopper_gmma_tma_col_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode> {

    // The base class.
    using Base = Smem_tile_hopper_gmma_tma_col_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_b( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// SMEM layout for Epilogue with HGMMA
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// column major output
// When Aclwmulator is in fp32, the data is casted to fp16 before storing to SMEM
template <typename Traits, typename Cta_tile, bool IN_CTA_SPLIT_K>
struct Swizzle_epilogue<Traits, Cta_tile, Col, 16, IN_CTA_SPLIT_K> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes per element
    enum { BYTES_PER_ELEMENT = 2 };
    // The number of bits per element
    enum { BITS_PER_ELEMENT = BYTES_PER_ELEMENT * 8 };
    // CTA tile size
    enum { CTA_M = Cta_tile::M, CTA_N = Cta_tile::N };
    // WARP GROUP distribution
    enum { WARP_GROUP_M = Cta_tile::WARP_GROUP_M, WARP_GROUP_N = Cta_tile::WARP_GROUP_N };
    // GMMA shape
    enum { GMMA_M = Traits::GMMA_M, GMMA_N = Traits::GMMA_N };
    // The number of threads
    enum { THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA };

    // reason about the min tile N
    // the size for each LDS.128
    enum { BYTES_PER_LDS = 16 };
    // Threads for LDS per column
    enum { LDS_THREADS_PER_COLUMN = CTA_M * BYTES_PER_ELEMENT / BYTES_PER_LDS };
    static_assert( LDS_THREADS_PER_COLUMN >= 8,
                   "LDS_THREADS_PER_COLUMN should be larger than 8\n" );
    // the number of columns can be loaded by all threads per LDS instruction
    enum { COLUMNS_PER_LDS = THREADS_PER_CTA / LDS_THREADS_PER_COLUMN };

    // the min tile in N dim is 8 such that every thread can participate in sts
    enum { MIN_TILE_N = COLUMNS_PER_LDS < 8 ? 8 : COLUMNS_PER_LDS };
    static_assert( MIN_TILE_N % 8 == 0, "MIN_TILE_N should be multiple of 8" );

    // we can probably reduce the tile M to MIN_TILE_M, but for simplicity we set tile_M = cta_M
    enum { TILE_M = CTA_M, TILE_N = MIN_TILE_N };
    //
    enum { BYTES_PER_COLUMN = TILE_M * BYTES_PER_ELEMENT };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = TILE_M * TILE_N * BYTES_PER_ELEMENT };

    enum { ELEMENT_PER_32bit = 4 / BYTES_PER_ELEMENT };

    // the number of 32 bit register held by each thread per tile before sts
    enum {
        NUM_REGS_PER_TILE_PRE_STORE =
            ( GMMA_M * TILE_N ) / ( Cta_tile::WARP_GROUP_M * Cta_tile::THREADS_PER_WARP_GROUP ) /
            ELEMENT_PER_32bit
    };

    //
    enum { M_PER_WARP = 16 };
    //
    enum { OFFSET_HI = BYTES_PER_COLUMN };

    // the number of inner iterations
    enum { LDS_ITERATIONS_PER_TILE = TILE_N / COLUMNS_PER_LDS };
    // the number of inner iteration to cover a GMMA M.
    enum { LDS_ITERATIONS_PER_GMMA_M = 16 / M_PER_WARP };
    // the number of inner iteration to cover a GMMA N.
    enum { LDS_ITERATIONS_PER_GMMA_N = Xmma_tile::N_PER_XMMA_PER_CTA / TILE_N };
    // the number of 32 bit register held by each thread per tile after lds
    enum {
        NUM_REGS_PER_TILE_POST_LOAD =
            ( BYTES_PER_LDS * LDS_ITERATIONS_PER_TILE / BYTES_PER_ELEMENT ) / ELEMENT_PER_32bit
    };

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_epilogue( char *smem, int tidx ) : smem_( smem ) {
        int lane_idx = tidx % 32;
        int warp_m_idx = tidx / 32;
        smem_write_offset_ = ( warp_m_idx * M_PER_WARP + ( lane_idx / 4 ) ) * BYTES_PER_ELEMENT;
        smem_write_offset_ += ( lane_idx % 4 ) * 2 * BYTES_PER_COLUMN;

        smem_read_offset_ = ( tidx % LDS_THREADS_PER_COLUMN ) * BYTES_PER_LDS +
                            ( tidx / LDS_THREADS_PER_COLUMN ) * BYTES_PER_COLUMN;
    }

    // Load from the tile in shared memory.
    template <typename Fragment_post_swizzle>
    inline __device__ void load( Fragment_post_swizzle &dst, int oi ) {
        const int offset = smem_read_offset_ + oi * COLUMNS_PER_LDS * BYTES_PER_COLUMN;

        int4 tmp = reinterpret_cast<const int4 *>( &this->smem_[offset] )[0];

        dst.reg( 0 ) = tmp.x;
        dst.reg( 1 ) = tmp.y;
        dst.reg( 2 ) = tmp.z;
        dst.reg( 3 ) = tmp.w;
    }

    // Store to the tile in shared memory.
    template <typename Fragment_pre_swizzle>
    inline __device__ void store( int xmmas_mi, const Fragment_pre_swizzle &c ) {
        #pragma unroll
        for( int reg_idx = 0; reg_idx < Fragment_pre_swizzle::NUM_REGS; ++reg_idx ) {
            // each reg_idx contains 2 fp16 elements
            uint16_t reg_lo, reg_hi;
            xmma::unpack_half2( c.regs_[reg_idx], reg_lo, reg_hi );
            int offset = smem_write_offset_
                         // 8 element in distance
                         + ( reg_idx % 2 ) * 8 * BYTES_PER_ELEMENT
                         // 8 columns in distance
                         + ( reg_idx / 2 ) * 8 * BYTES_PER_COLUMN
                         // GMMA_M element in distance
                         + xmmas_mi * GMMA_M * BYTES_PER_ELEMENT;
            reinterpret_cast<uint16_t *>( &smem_[offset] )[0] = reg_lo;
            reinterpret_cast<uint16_t *>( &smem_[offset + OFFSET_HI] )[0] = reg_hi;
        }
    }

    // The shared memory pointer in bytes.
    char *smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// row major output
// When Aclwmulator is in fp32, the data is casted to fp16 before storing to SMEM
template <typename Traits, typename Cta_tile, bool IN_CTA_SPLIT_K>
struct Swizzle_epilogue<Traits, Cta_tile, Row, 16, IN_CTA_SPLIT_K> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes per element
    enum { BYTES_PER_ELEMENT = 2 };
    // The number of bits per element
    enum { BITS_PER_ELEMENT = BYTES_PER_ELEMENT * 8 };
    // CTA tile size
    enum { CTA_M = Cta_tile::M, CTA_N = Cta_tile::N };

    // when CTA_N <= 32 we want to do something a little different
    static_assert( CTA_N > 32, "CTA_N <= 32 is not yet implemented. " );
    // when CTA_N < 64 but > 32, we will predicate out some threads.
    static_assert( CTA_N >= 64, "CTA_N < 64 is not yet implemented. " );
    // WARP GROUP distribution
    enum { WARP_GROUP_M = Cta_tile::WARP_GROUP_M, WARP_GROUP_N = Cta_tile::WARP_GROUP_N };
    // GMMA shape
    enum { GMMA_M = Traits::GMMA_M, GMMA_N = Traits::GMMA_N };
    // The number of threads
    enum { THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA };

    // reason about the min tile N
    // the size for each LDS.128
    enum { BYTES_PER_LDS = 16 };

    // tile_n if we can have 8 threads doing lds.128 unless cta_n is smaller than that
    enum { COLUMNS_PER_LDS = BYTES_PER_LDS * 8 / BYTES_PER_ELEMENT };
    enum { MIN_TILE_N = CTA_N < COLUMNS_PER_LDS ? CTA_N : COLUMNS_PER_LDS };

    // tile_m is limited such that every thread can participate, 8 rows per warp
    enum { TILE_M = 8 * THREADS_PER_CTA / 32, TILE_N = MIN_TILE_N };
    //
    enum { BYTES_PER_ROW = TILE_N * BYTES_PER_ELEMENT };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = TILE_M * TILE_N * BYTES_PER_ELEMENT };

    enum { ELEMENT_PER_32bit = 4 / BYTES_PER_ELEMENT };
    //
    // the number of 32 bit register held by each thread per tile before sts
    enum {
        NUM_REGS_PER_TILE_PRE_STORE =
            ( TILE_M * TILE_N ) / ( Cta_tile::WARP_GROUP_M * Cta_tile::THREADS_PER_WARP_GROUP ) /
            ELEMENT_PER_32bit
    };

    //
    enum { M_PER_WARP = 8 };
    ////
    // enum { OFFSET_HI = BYTES_PER_COLUMN };
    //
    // the number of threads per lds needed by a row
    enum { LDS_THREADS_PER_ROW = TILE_N * BYTES_PER_ELEMENT / BYTES_PER_LDS };
    // the number of rows per LDS instructions by all threads
    enum { ROWS_PER_LDS = THREADS_PER_CTA / LDS_THREADS_PER_ROW };
    // the number of rows per LDS instruction by one warp.
    enum { ROWS_PER_LDS_PER_WARP = Cta_tile::THREADS_PER_WARP / LDS_THREADS_PER_ROW };
    // the number of inner iterations
    enum { LDS_ITERATIONS_PER_TILE = TILE_M / ROWS_PER_LDS };
    // the number of inner iteration to cover a GMMA M. should always be 2.
    enum { LDS_ITERATIONS_PER_GMMA_M = 16 / M_PER_WARP };
    // the number of inner iteration to cover a GMMA N.
    enum { LDS_ITERATIONS_PER_GMMA_N = Xmma_tile::N_PER_XMMA_PER_CTA / TILE_N };
    // the number of 32 bit register held by each thread per tile after lds
    enum {
        NUM_REGS_PER_TILE_POST_LOAD =
            ( BYTES_PER_LDS * LDS_ITERATIONS_PER_TILE / BYTES_PER_ELEMENT ) / ELEMENT_PER_32bit
    };
    //
    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_epilogue( char *smem, int tidx ) : smem_( smem ) {
        int lane_idx = tidx % 32;
        int warp_m_idx = tidx / 32;
        smem_write_offset_ = ( warp_m_idx * M_PER_WARP + ( lane_idx / 4 ) ) * BYTES_PER_ROW;
        smem_write_offset_ += ( lane_idx % 4 ) * 2 * BYTES_PER_ELEMENT;

        smem_read_offset_ = ( lane_idx % LDS_THREADS_PER_ROW ) * BYTES_PER_LDS +
                            ( lane_idx / LDS_THREADS_PER_ROW ) * BYTES_PER_ROW +
                            ( warp_m_idx * M_PER_WARP ) * BYTES_PER_ROW;
    }

    // Load from the tile in shared memory.
    template <typename Fragment_post_swizzle>
    inline __device__ void load( Fragment_post_swizzle &dst, int oi ) {
        const int offset = smem_read_offset_ + oi * ROWS_PER_LDS_PER_WARP * BYTES_PER_ROW;
        int4 tmp = reinterpret_cast<const int4 *>( &this->smem_[offset] )[0];

        dst.reg( 0 ) = tmp.x;
        dst.reg( 1 ) = tmp.y;
        dst.reg( 2 ) = tmp.z;
        dst.reg( 3 ) = tmp.w;
    }

    // Store to the tile in shared memory.
    template <typename Fragment_pre_swizzle>
    inline __device__ void store( int, const Fragment_pre_swizzle &c ) {
        #pragma unroll
        for( int reg_idx = 0; reg_idx < Fragment_pre_swizzle::NUM_REGS; ++reg_idx ) {
            // each reg_idx contains 2 fp16 elements
            int offset = smem_write_offset_ + reg_idx * 8 * BYTES_PER_ELEMENT;
            reinterpret_cast<uint32_t *>( &smem_[offset] )[0] = c.regs_[reg_idx];
        }
    }
    // The shared memory pointer in bytes.
    char *smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Layout>
struct Swizzle_hopper_hgmma_fp32_epilogue {};

////////////////////////////////////////////////////////////////////////////////////////////////////
// col major output for fp32 epilogue type
template <typename Traits, typename Cta_tile>
struct Swizzle_hopper_hgmma_fp32_epilogue<Traits, Cta_tile, Col> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes per epilogue element
    enum { BYTES_PER_ELEMENT = sizeof( typename Traits::Epilogue_type ) };
    // The number of bits per element
    enum { BITS_PER_ELEMENT = BYTES_PER_ELEMENT * 8 };
    // The number of bytes per element C
    enum { BYTES_PER_ELEMENT_C = sizeof( typename Traits::C_type ) };
    // CTA tile size
    enum { CTA_M = Cta_tile::M, CTA_N = Cta_tile::N };
    // WARP GROUP distribution
    enum { WARP_GROUP_M = Cta_tile::WARP_GROUP_M, WARP_GROUP_N = Cta_tile::WARP_GROUP_N };
    // GMMA shape
    enum { GMMA_M = Traits::GMMA_M, GMMA_N = Traits::GMMA_N };
    // The number of threads
    enum { THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA };

    // we can probably reduce the tile M to MIN_TILE_M, but for simplicity we set tile_M = cta_M
    enum { TILE_M = CTA_M };
    // we break one tile columns into two smem cols, i.e. reshaping the MxN tile into (M/2)x(N*2)
    // the left half is for lane0-lane15, and the right half is for lane16-lane31
    // the number of rows of smem
    enum { ROWS_PER_SMEM = TILE_M / 2 };

    // reason about the min tile N
    // the size for each LDS.128
    enum { BYTES_PER_LDS = 16 };
    // Threads for LDS per column
    enum { LDS_THREADS_PER_COLUMN = ROWS_PER_SMEM * BYTES_PER_ELEMENT / BYTES_PER_LDS };
    static_assert( LDS_THREADS_PER_COLUMN >= 8,
                   "LDS_THREADS_PER_COLUMN should be larger than 8\n" );
    // the number of columns can be loaded by all threads per LDS instruction
    enum { COLUMNS_PER_LDS = THREADS_PER_CTA / LDS_THREADS_PER_COLUMN };

    // the min tile in N dim is 8 such that every thread can participate in sts
    enum { MIN_TILE_N = COLUMNS_PER_LDS < 8 ? 8 : COLUMNS_PER_LDS };
    static_assert( MIN_TILE_N % 8 == 0, "MIN_TILE_N should be multiple of 8" );

    // the size of tile N
    enum { TILE_N = MIN_TILE_N };
    // the number of columns of smem
    enum { COLS_PER_SMEM = TILE_N * 2 };
    //
    enum { BYTES_PER_COLUMN = ROWS_PER_SMEM * BYTES_PER_ELEMENT };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS_PER_SMEM * COLS_PER_SMEM * BYTES_PER_ELEMENT };

    enum { ELEMENT_PER_32bit = 4 / BYTES_PER_ELEMENT };

    // the number of 32 bit register held by each thread per tile before sts
    enum {
        NUM_REGS_PER_TILE_PRE_STORE =
            ( GMMA_M * TILE_N ) / ( Cta_tile::WARP_GROUP_M * Cta_tile::THREADS_PER_WARP_GROUP ) /
            ELEMENT_PER_32bit
    };

    //
    enum { M_PER_WARP = 16 };
    //
    enum { OFFSET_HI = BYTES_PER_COLUMN };

    // the number of inner iterations
    // there are two LDS.128 in each iteration to load 8xfp32 elements
    enum { LDS_ITERATIONS_PER_TILE = COLS_PER_SMEM / COLUMNS_PER_LDS / 2 };
    // the number of inner iteration to cover a GMMA M.
    enum { LDS_ITERATIONS_PER_GMMA_M = 16 / M_PER_WARP };
    // the number of inner iteration to cover a GMMA N.
    enum { LDS_ITERATIONS_PER_GMMA_N = Xmma_tile::N_PER_XMMA_PER_CTA / TILE_N };
    // the number of 32 bit register held by each thread per tile after lds
    enum {
        NUM_REGS_PER_TILE_POST_LOAD =
            ( BYTES_PER_LDS * LDS_ITERATIONS_PER_TILE * 2 / BYTES_PER_ELEMENT ) / ELEMENT_PER_32bit
    };

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_hopper_hgmma_fp32_epilogue( char *smem, int tidx ) : smem_( smem ) {
        int lane_idx = tidx % 32;
        int warp_m_idx = tidx / 32;
        smem_write_offset_ =
            ( warp_m_idx * ( M_PER_WARP / 2 ) + ( ( lane_idx & 0xf ) >> 2 ) ) * BYTES_PER_ELEMENT;
        smem_write_offset_ +=
            ( ( lane_idx & 0x3 ) * 2 + ( lane_idx >> 4 ) * TILE_N ) * BYTES_PER_COLUMN;

        smem_read_offset_ = ( tidx % LDS_THREADS_PER_COLUMN ) * BYTES_PER_LDS +
                            ( tidx / LDS_THREADS_PER_COLUMN ) * BYTES_PER_COLUMN;
    }

    // Load from the tile in shared memory.
    template <typename Fragment_post_swizzle>
    inline __device__ void load( Fragment_post_swizzle &dst, int oi ) {
        const int offset_0 = smem_read_offset_ + oi * COLUMNS_PER_LDS * BYTES_PER_COLUMN;
        int4 tmp_0 = reinterpret_cast<const int4 *>( &this->smem_[offset_0] )[0];

        dst.reg( 0 ) = tmp_0.x;
        dst.reg( 1 ) = tmp_0.y;
        dst.reg( 2 ) = tmp_0.z;
        dst.reg( 3 ) = tmp_0.w;

        const int offset_1 = offset_0 + TILE_N * BYTES_PER_COLUMN;
        int4 tmp_1 = reinterpret_cast<const int4 *>( &this->smem_[offset_1] )[0];

        dst.reg( 4 ) = tmp_1.x;
        dst.reg( 5 ) = tmp_1.y;
        dst.reg( 6 ) = tmp_1.z;
        dst.reg( 7 ) = tmp_1.w;
    }

    // Store to the tile in shared memory.
    template <typename Fragment_pre_swizzle>
    inline __device__ void store( int xmmas_mi, const Fragment_pre_swizzle &c ) {
        #pragma unroll
        for( int reg_idx = 0; reg_idx < Fragment_pre_swizzle::NUM_REGS; ++reg_idx ) {
            // each reg_idx contains 1 fp32 elements
            int offset = smem_write_offset_
                         // 8 element in distance
                         + ( reg_idx % 4 ) / 2 * 4 * BYTES_PER_ELEMENT
                         // 8 columns in distance
                         + ( reg_idx / 4 ) * 8 * BYTES_PER_COLUMN
                         // 1 column in distance
                         + ( reg_idx % 2 ) * BYTES_PER_COLUMN
                         // GMMA_M element in distance
                         + xmmas_mi * ( GMMA_M / 2 ) * BYTES_PER_ELEMENT;
            reinterpret_cast<uint32_t *>( &smem_[offset] )[0] = c.regs_[reg_idx];
        }
    }

    // The shared memory pointer in bytes.
    char *smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// row major output for fp32 epilogue type
template <typename Traits, typename Cta_tile>
struct Swizzle_hopper_hgmma_fp32_epilogue<Traits, Cta_tile, Row> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes per element
    enum { BYTES_PER_ELEMENT = sizeof( typename Traits::Epilogue_type ) };
    // The number of bytes per element C
    enum { BYTES_PER_ELEMENT_C = sizeof( typename Traits::C_type ) };
    // The number of bits per element
    enum { BITS_PER_ELEMENT = BYTES_PER_ELEMENT * 8 };
    // CTA tile size
    enum { CTA_M = Cta_tile::M, CTA_N = Cta_tile::N };

    // when CTA_N <= 32 we want to do something a little different
    static_assert( CTA_N > 32, "CTA_N <= 32 is not yet implemented. " );
    // when CTA_N < 64 but > 32, we will predicate out some threads.
    static_assert( CTA_N >= 64, "CTA_N < 64 is not yet implemented. " );
    // WARP GROUP distribution
    enum { WARP_GROUP_M = Cta_tile::WARP_GROUP_M, WARP_GROUP_N = Cta_tile::WARP_GROUP_N };
    // GMMA shape
    enum { GMMA_M = Traits::GMMA_M, GMMA_N = Traits::GMMA_N };
    // The number of threads
    enum { THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA };

    // reason about the min tile N
    // the size for each LDS.128
    enum { BYTES_PER_LDS = 16 };

    // tile_n if we can have 8 threads doing lds.128 unless cta_n is smaller than that
    enum { COLUMNS_PER_LDS = BYTES_PER_LDS * 8 / BYTES_PER_ELEMENT_C };
    enum { MIN_TILE_N = CTA_N < COLUMNS_PER_LDS ? CTA_N : COLUMNS_PER_LDS };

    // tile_m is limited such that every thread can participate, 8 rows per warp
    enum { TILE_M = 8 * THREADS_PER_CTA / 32, TILE_N = MIN_TILE_N };
    // we break one tile row into two rows in smem, reshaping the MxN tile into (M*2)x(N/2)
    // the number of columns of smem
    enum { COLS_PER_SMEM = TILE_N / 2 };
    // the number of rows of smem
    enum { ROWS_PER_SMEM = TILE_M * 2 };
    // the bytes per smem row
    enum { BYTES_PER_ROW = COLS_PER_SMEM * BYTES_PER_ELEMENT };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS_PER_SMEM * COLS_PER_SMEM * BYTES_PER_ELEMENT };

    enum { ELEMENT_PER_32bit = 4 / BYTES_PER_ELEMENT };
    //
    // the number of 32 bit register held by each thread per tile before sts
    enum {
        NUM_REGS_PER_TILE_PRE_STORE =
            ( TILE_M * TILE_N ) / ( Cta_tile::WARP_GROUP_M * Cta_tile::THREADS_PER_WARP_GROUP ) /
            ELEMENT_PER_32bit
    };

    //
    enum { M_PER_WARP = 8 };
    ////
    // enum { OFFSET_HI = BYTES_PER_COLUMN };
    //
    // the number of threads per lds needed by a row
    enum { LDS_THREADS_PER_ROW = COLS_PER_SMEM * BYTES_PER_ELEMENT / BYTES_PER_LDS };
    // the number of rows per LDS instructions by all threads
    enum { ROWS_PER_LDS = THREADS_PER_CTA / LDS_THREADS_PER_ROW };
    // the number of rows per LDS instruction by one warp.
    enum { ROWS_PER_LDS_PER_WARP = Cta_tile::THREADS_PER_WARP / LDS_THREADS_PER_ROW };
    // the number of inner iterations
    // there are two LDS issued in one iteration
    enum { LDS_ITERATIONS_PER_TILE = ROWS_PER_SMEM / ( ROWS_PER_LDS * 2 ) };
    // the number of inner iteration to cover a GMMA M. should always be 2.
    enum { LDS_ITERATIONS_PER_GMMA_M = 16 / M_PER_WARP };
    // the number of inner iteration to cover a GMMA N.
    enum { LDS_ITERATIONS_PER_GMMA_N = Xmma_tile::N_PER_XMMA_PER_CTA / TILE_N };
    // the number of 32 bit register held by each thread per tile after lds
    enum {
        NUM_REGS_PER_TILE_POST_LOAD =
            ( BYTES_PER_LDS * LDS_ITERATIONS_PER_TILE / BYTES_PER_ELEMENT ) / ELEMENT_PER_32bit
    };
    // The number of C regs held by each thread per tile before STG
    enum {
        NUM_REGS_PER_TILE_C =
            NUM_REGS_PER_TILE_POST_LOAD / ( BYTES_PER_ELEMENT / BYTES_PER_ELEMENT_C )
    };
    //
    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_hopper_hgmma_fp32_epilogue( char *smem, int tidx ) : smem_( smem ) {
        int lane_idx = tidx % 32;
        int warp_m_idx = tidx / 32;
        smem_write_offset_ = ( warp_m_idx * ROWS_PER_LDS_PER_WARP * 2 * LDS_ITERATIONS_PER_TILE +
                               ( lane_idx & 0x1e ) / 2 ) *
                             BYTES_PER_ROW;
        smem_write_offset_ += ( lane_idx & 0x01 ) * 2 * BYTES_PER_ELEMENT;

        smem_read_offset_ =
            ( lane_idx % LDS_THREADS_PER_ROW ) * BYTES_PER_LDS +
            ( lane_idx / LDS_THREADS_PER_ROW ) * 2 * BYTES_PER_ROW +
            ( warp_m_idx * ROWS_PER_LDS_PER_WARP * 2 * LDS_ITERATIONS_PER_TILE ) * BYTES_PER_ROW;
    }

    // Load from the tile in shared memory.
    template <typename Fragment_post_swizzle>
    inline __device__ void load( Fragment_post_swizzle &dst, int oi ) {
        const int offset_0 = smem_read_offset_ + oi * ROWS_PER_LDS_PER_WARP * 2 * BYTES_PER_ROW;
        int4 tmp_0 = reinterpret_cast<const int4 *>( &this->smem_[offset_0] )[0];

        dst.reg( 0 ) = tmp_0.x;
        dst.reg( 1 ) = tmp_0.y;
        dst.reg( 2 ) = tmp_0.z;
        dst.reg( 3 ) = tmp_0.w;

        const int offset_1 = offset_0 + BYTES_PER_ROW;
        int4 tmp_1 = reinterpret_cast<const int4 *>( &this->smem_[offset_1] )[0];

        dst.reg( 4 ) = tmp_1.x;
        dst.reg( 5 ) = tmp_1.y;
        dst.reg( 6 ) = tmp_1.z;
        dst.reg( 7 ) = tmp_1.w;
    }

    // Store to the tile in shared memory.
    template <typename Fragment_pre_swizzle>
    inline __device__ void store( int, const Fragment_pre_swizzle &c ) {
        #pragma unroll
        for( int reg_idx = 0; reg_idx < Fragment_pre_swizzle::NUM_REGS; ++reg_idx ) {
            // each reg_idx contains 1 fp32 elements
            int offset = smem_write_offset_ + ( reg_idx >> 1 ) * 4 * BYTES_PER_ELEMENT +
                         reg_idx % 2 * BYTES_PER_ELEMENT;
            reinterpret_cast<uint32_t *>( &smem_[offset] )[0] = c.regs_[reg_idx];
        }
    }

    // The shared memory pointer in bytes.
    char *smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          bool IN_CTA_SPLIT_K>
struct Swizzle_epilogue<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
                        Cta_tile,
                        xmma::Row,
                        16,
                        IN_CTA_SPLIT_K>
    : public Swizzle_hopper_hgmma_fp32_epilogue<
          Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile,
          xmma::Row> {

    // The traits class.
    using Traits = Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // The base class.
    using Base = Swizzle_hopper_hgmma_fp32_epilogue<Traits, Cta_tile, xmma::Row>;

    // Ctor.
    inline __device__ Swizzle_epilogue( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M,
          int GMMA_N,
          int GMMA_K,
          bool GMMA_A_RF,
          bool GMMA_B_RF,
          typename Cta_tile,
          bool IN_CTA_SPLIT_K>
struct Swizzle_epilogue<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
                        Cta_tile,
                        xmma::Col,
                        16,
                        IN_CTA_SPLIT_K>
    : public Swizzle_hopper_hgmma_fp32_epilogue<
          Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile,
          xmma::Col> {

    // The traits class.
    using Traits = Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;
    // The base class.
    using Base = Swizzle_hopper_hgmma_fp32_epilogue<Traits, Cta_tile, xmma::Col>;

    // Ctor.
    inline __device__ Swizzle_epilogue( char *smem, int tidx ) : Base( smem, tidx ) {
    }
};
}  // namespace xmma
