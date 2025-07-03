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

#include <xmma/xmma.h>

#define MIN(m, n) ((m < n) ? m : n)
namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The layout of the tile.
    typename Layout, 
    // The size of the STS.
    int BYTES_PER_STS = 16,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE = 1,
    // Use or not predicates
    bool USE_PREDICATES = true,
    // Use tma ?
    bool USE_TMA = false
>
struct Smem_tile_a {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The layout of the tile.
    typename Layout, 
    // The size of the STS.
    int BYTES_PER_STS = 16,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE = 1,
    // Use or not predicates
    bool USE_PREDICATES = true,
    // Use tma ?
    bool USE_TMA = false
>
struct Smem_tile_b {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The layout of the tile.
    typename Layout, 
    // The number of bytes per LDS
    int BYTES_PER_LDS = 16,
    // Do we use the special split-k trick inside the CTA?
    bool = (Cta_tile::WARPS_K > 1)
>
struct Swizzle_epilogue {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout of the tile.
    typename Layout,
    // Do we use the special split-k trick inside the CTA?
    bool = (Cta_tile::WARPS_K > 1)
>
struct Swizzle_epilogue_empty {

    // The size in bytes of total buffers.
    enum { BYTES_PER_TILE = 0 };

    // The traits.
    using Traits = Traits_;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    inline __device__ Swizzle_epilogue_empty(void *smem, int tidx) {}

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {}

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The layout of the tile.
    typename Layout, 
    // Do we use the special split-k trick inside the CTA?
    bool = (Cta_tile::WARPS_K > 1)
>
struct Swizzle_epilogue_interleaved {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The pixels.
    typename Pixel_tile, 
    // The layout of the tile.
    typename Layout, 
    // Do we use the special split-k trick inside the CTA?
    bool IN_CTA_SPLIT_K = (Cta_tile::WARPS_K > 1)
>
struct Swizzle_epilogue_interleaved_1x1_with_3x3 {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

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
struct Smem_tile_without_skews {

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

    // Ctor.
    inline __device__ Smem_tile_without_skews(void *smem, int tidx) 
        : smem_(get_smem_pointer(smem)) {

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
    }

    // Compute the store pointers.
    template< int N, int PHASE, int K = 1 >
    inline __device__ void compute_store_pointers_per_phase(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int ii = N * PHASE; ii < N * (PHASE + 1) ; ii += MIN(N, K) ) {
            // Decompose the STS into row/col.
            int row = (ii / K) / STS_PER_ROW;
            int col = (ii / K) % STS_PER_ROW;

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
            for (int k = 0; k < MIN(N, K); k++) {
                ptrs[ii + k - N * PHASE] = smem_ + offset + ((N*PHASE) % K + k) * 4 + smem_write_buffer_;
            }
        }
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
    }

    // Move the write offset to next buffer. TODO: Remove that member function!
    inline __device__ void move_next_write_buffer() {
        this->move_to_next_write_buffer();
    }

    // Move the read offset.
    inline __device__ void move_read_offset(int delta) {
        this->smem_read_offset_ += delta;
    }

    // Move the write offset.
    inline __device__ void move_write_offset(int delta) {
        this->smem_write_offset_ += delta;
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const Store_type (&data)[N]) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        sts(smem_ptrs, data);
    }

    // Store to the tile in shared memory.
    template< int N, int M >
    inline __device__ void store(const Store_type (&data)[N], uint32_t (&preds)[M]) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        sts(smem_ptrs, data, preds);
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const Store_type (&data)[N], 
                                 uint32_t preds, 
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        this->store(data, preds, mem_desc);
    }

    // Store to the tile in shared memory.
    template< int N, int M, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true> >
    inline __device__ void store(const void* (&gmem_ptrs)[N], 
                                 uint32_t (&preds)[M], 
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers<N, K>(smem_ptrs);
        if (USE_PREDICATES) {
            ldgsts< N, M, 16/K, LDGSTS_CFG >(smem_ptrs, gmem_ptrs, preds, mem_desc);
        } else {
            #pragma unroll
            for (int ii = 0; ii < N; ii++)
            ldgsts128_nopreds(smem_ptrs[ii], gmem_ptrs[ii]);
        }
    }

    // Store to the tile in shared memory.
    template< int N, int M, int PHASE, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true> >
    inline __device__ void store_per_phase(const void* (&gmem_ptrs)[N], 
                                 uint32_t (&preds)[M], 
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[N];
        static_assert(USE_PREDICATES == true, "");
        this->compute_store_pointers_per_phase<N, PHASE, K>(smem_ptrs);
        ldgsts_per_phase<N, M, PHASE, 16/K, LDGSTS_CFG>(smem_ptrs, gmem_ptrs, preds, mem_desc);
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const void* (&gmem_ptrs)[N], 
                                 uint32_t preds, 
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t tmp[1] = { preds };
        this->store(gmem_ptrs, tmp, mem_desc);
    }

    inline __device__ void add_smem_barrier_base( uint64_t * ) {
    }

    // Store to the tile in shared memory.
    template< int N, int phase >
    inline __device__ void store_per_phase(const void* (&gmem_ptrs)[N], 
                                 uint32_t preds, 
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t tmp[1] = { preds };
        this->store_per_phase<N, 1, phase>(gmem_ptrs, tmp, mem_desc);
    }

    // The shared memory pointer.
    uint32_t smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The buffer base offset for read.
    int smem_read_buffer_;
    // The buffer base offset for write.
    int smem_write_buffer_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

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
struct Smem_tile_interleaved {

    // The number of words in 128b.
    enum { BITS_PER_ELEMENT = BITS_PER_ELEMENT_, BYTES_PER_STS = BYTES_PER_STS_ };
    // The number of elements that are stored with a single STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // The number of rows that are needed.
    enum { ROWS = M_ / ELEMENTS_PER_STS };
    // The number of threads per row.
    enum { THREADS_PER_ROW = Cta_tile::THREADS_PER_CTA / ROWS };
    // The number of bytes per row.
    enum { BYTES_PER_ROW = N_ * BYTES_PER_STS };
    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = ROWS * BYTES_PER_ROW };
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
    // The number of STS per row.
    enum { STS_PER_ROW = BYTES_PER_ROW / THREADS_PER_ROW / BYTES_PER_STS };
    // It must be at least one.
    static_assert(STS_PER_ROW >= 1, "");
    // The number of rows written with a single STS.
    enum { ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // Make sure we write to at least one row per STS. Thanks Dr. Obvious ;)
    static_assert(ROWS_PER_STS >= 1, "");
    
    // The type of elements that are stored in shared memory by each thread.
    using Store_type = typename Uint_from_size_in_bytes<BYTES_PER_STS>::Type;

    // Ctor.
    inline __device__ Smem_tile_interleaved(void *smem, int tidx) 
        : smem_(get_smem_pointer(smem)) {
        if (BITS_PER_ELEMENT == 16) {
            // The row/col written by the thread.
            int smem_write_row = tidx / THREADS_PER_ROW;
            int smem_write_col = tidx % THREADS_PER_ROW;

            // The location where the thread writes its elements.
            smem_write_offset_ = smem_write_row*BYTES_PER_ROW + smem_write_col*BYTES_PER_STS;
        }else {
            // The location where the thread writes its elements.
            smem_write_offset_ = tidx * BYTES_PER_STS;
        }

        // Initialize the URF.
        smem_read_buffer_ = smem_write_buffer_ = __shfl_sync(0xffffffff, 0, 0);
    }

    // Compute the store pointers.
    template< int N >
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int ii = 0; ii < N; ++ii ) {
            if (BITS_PER_ELEMENT == 16) {
                int offset = smem_write_offset_ + ii * THREADS_PER_ROW * BYTES_PER_STS;
                ptrs[ii] = smem_ + offset;
            }else {
                int imm = ii * Cta_tile::THREADS_PER_CTA * BYTES_PER_STS;
                ptrs[ii] = smem_ + smem_write_buffer_ + smem_write_offset_ + imm;
            }
        }
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int row = 0; row < ROWS; ++row ) {
            for( int col = 0; col < BYTES_PER_ROW; col += 4 ) {
                if( threadIdx.x == 0 ) {
                    uint32_t val;
                    lds(val, smem_ + row*BYTES_PER_ROW + col);
                    printf("block=(x=%2d, y=%2d, z=%2d) (row=%2d, byte=%4d)=0x%08x\n",
                        blockIdx.x,
                        blockIdx.y,
                        blockIdx.z,
                        row,
                        col,
                        val);
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
        if (BITS_PER_ELEMENT == 16) {
             this->smem_write_offset_ += 
                (smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                ? -BYTES_PER_TILE_INC_BOUNDARY
                : BYTES_PER_BUFFER ;    
        }else{
            if( BUFFERS_PER_TILE > 1 && smem_write_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
                this->smem_write_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
            } else if( BUFFERS_PER_TILE > 1 ) {
                this->smem_write_buffer_ += BYTES_PER_BUFFER;
            }
        }
    }

    // Move the write offset to next buffer. TODO: Remove that member function!
    inline __device__ void move_next_write_buffer() {
        this->move_to_next_write_buffer();
    }

    // Move the read offset.
    inline __device__ void move_read_offset(int delta) {
        this->smem_read_offset_ += delta;
    }

    // Move the write offset.
    inline __device__ void move_write_offset(int delta) {
        this->smem_write_offset_ += delta;
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const Store_type (&data)[N]) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        sts(smem_ptrs, data);
    }

    // Store to the tile in shared memory.
    template< int N, int M, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true> >
    inline __device__ void store(const void* (&gmem_ptrs)[N],
                                 uint32_t (&preds)[M],
                                 const uint64_t mem_desc) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        ldgsts<N, M, 16/K>(smem_ptrs, gmem_ptrs, preds, mem_desc);
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const void* (&gmem_ptrs)[N], 
                                 uint32_t preds, 
                                 const uint64_t mem_desc) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        ldgsts(smem_ptrs, gmem_ptrs, preds, mem_desc);
    }
    
    // Store to the tile in shared memory.
    template< int N, int M, int PHASE, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true> >
    inline __device__ void store_per_phase(const void* (&gmem_ptrs)[N], 
                                 uint32_t (&preds)[M], 
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[N];
        static_assert(USE_PREDICATES == true, "");
        this->compute_store_pointers_per_phase<N, PHASE, K>(smem_ptrs);
        ldgsts_per_phase<N, M, PHASE, 16/K, LDGSTS_CFG>(smem_ptrs, gmem_ptrs, preds, mem_desc);
    }

    // Store to the tile in shared memory.
    template< int N, int phase >
    inline __device__ void store_per_phase(const void* (&gmem_ptrs)[N], 
                                 uint32_t preds, 
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t tmp[1] = { preds };
        this->store_per_phase<N, 1, phase>(gmem_ptrs, tmp, mem_desc);
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store_with_guard(const Store_type (&data)[N]) {
        // The shared memory pointers.
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);

        // Store the ungarded elements.
        #pragma unroll
        for( int ii = 0; ii < N-1; ++ii ) {
            sts(smem_ptrs[ii], data[ii]);
        }

        // Remainder.
        enum { REMAINDER = (M_ * N_) % (Cta_tile::THREADS_PER_CTA * ELEMENTS_PER_STS) };
        if( REMAINDER == 0 || threadIdx.x * ELEMENTS_PER_STS < REMAINDER ) {
            sts(smem_ptrs[N-1], data[N-1]);
        }
    }

    // Set barrier base.
    inline __device__ void add_smem_barrier_base( uint64_t * ) {
    }

    // The shared memory pointer.
    uint32_t smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The buffer base offset for read.
    int smem_read_buffer_;
    // The buffer base offset for write.
    int smem_write_buffer_;
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template< int XMMAS_K, int XMMAS_K_WITH_PADDING >
struct Compute_reset_mask {
    // The potential mask.
    enum { HALF = XMMAS_K_WITH_PADDING / 2 };
    // The remainder.
    enum { MOD = XMMAS_K % HALF };
    // The final value.
    enum { VALUE = (XMMAS_K == MOD ? 0 : HALF) | Compute_reset_mask<MOD, HALF>::VALUE };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int XMMAS_K_WITH_PADDING >
struct Compute_reset_mask<0, XMMAS_K_WITH_PADDING> {
    enum { VALUE = 0 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int XMMAS_K >
struct Compute_reset_mask<XMMAS_K, XMMAS_K> {
    enum { VALUE = XMMAS_K - 1 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

