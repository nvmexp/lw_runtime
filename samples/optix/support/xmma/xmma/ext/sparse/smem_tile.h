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

#include <xmma/smem_tile.h>

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
    int BUFFERS_PER_TILE = 1
>
struct Smem_tile_interleaved_a {
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
    int BUFFERS_PER_TILE = 1
>
struct Smem_tile_e {
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
    int BUFFERS_PER_TILE = 1
>
struct Smem_tile_lds_e {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    typename Cta_tile, 
    // The number of rows in the 2D shared memory buffer.
    int M_, 
    // The number of cols.
    int N_, 
    // The size in bits of each element.
    int BITS_PER_ELEMENT_, 
    // The number of bytes per STS.
    int BYTES_PER_STS_ = 4,
    // The number of buffers. (Used in multistage and double buffer cases.)
    int BUFFERS_PER_TILE_ = 1,
    // Do we enable the fast path for LDS.128 and friends.
    int ENABLE_LDS_FAST_PATH_ = 0, 
    // The number of rows that are used for the XOR swizzling to allow fast STS/LDS. 
    int ROWS_PER_XOR_PATTERN_ = 8,
    // The number of cols that are used for the XOR swizzling to allow fast STS/LDS. 
    int COLS_PER_XOR_PATTERN_ = 1
>
struct Smem_metadata_tile_linear {
    // The size in bits of each element.
    enum { BITS_PER_ELEMENT = BITS_PER_ELEMENT_ };
    // M
    enum { M = M_ };
    // N
    enum { N = N_ };
    // LDGSTS_PER_WARP
    enum { LDGSTS_PER_WARP = Cta_tile::WARPS_N == 1 ? 2 : 1 };
    // The size in bits of a LDGSTS type.
    enum { LDGSTS_SIZE = (M_ * N_ * BITS_PER_ELEMENT /
        Cta_tile::THREADS_PER_CTA) / LDGSTS_PER_WARP };
    // The size in bytes of a single STS.
    enum { BYTES_PER_STS = BYTES_PER_STS_ };
    // The size in bits of a LDGSTS type.
    enum { STSS = LDGSTS_SIZE / (BYTES_PER_STS * 8) };
    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = M_ * N_ * BITS_PER_ELEMENT / 8 };
    // The number of buffers. 
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // The size in bytes of total buffers.
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE };
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER };

    // To support arbitrary N, we pad some values to a power-of-2.
    enum { N_WITH_PADDING = xmma::Next_power_of_two<N_>::VALUE }; 
    // The number of bytes per row without packing of rows.
    enum { BYTES_PER_ROW_BEFORE_PACKING = N_WITH_PADDING * BITS_PER_ELEMENT / 8 };
    // The number of bytes per row -- we want at least 128B per row.
    enum { BYTES_PER_ROW = Max<BYTES_PER_ROW_BEFORE_PACKING, 128>::VALUE };
    // The number of rows that are used for the XOR swizzling to allow fast STS/LDS. 
    enum { ROWS_PER_XOR_PATTERN = ROWS_PER_XOR_PATTERN_ };
    
    // Ctor.
    inline __device__ Smem_metadata_tile_linear(char *smem, int tidx)
        : smem_(xmma::get_smem_pointer(smem)) {

        // The offset.
        if(Cta_tile::WARPS_M == 4 && Cta_tile::WARPS_N == 1){
            if (Cta_tile::K == 64 || Cta_tile::K == 32) {
                this->smem_write_offset_ = (tidx % 32) * (LDGSTS_SIZE / 8) + (tidx / 32) * 512;
            } else {
                this->smem_write_offset_ = (tidx % 32) * (LDGSTS_SIZE / 8) + (tidx / 32) * 1024;
            }
        } else {
            this->smem_write_offset_ = tidx * (LDGSTS_SIZE / 8);
        }
        
        this->smem_read_buffer_  = __shfl_sync(0xffffffff, 0, 0);
    }

    // Compute the store pointers.
    template< int N >
    inline __device__ void compute_store_pointers_e(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int i = 0; i < N; ++i ) {
            int offset = smem_write_offset_ + i * 4;    // 4 represents ldgsts32
            ptrs[i] = smem_ + offset;
        }
    }

    // Compute the store pointers.
    template< int N >
    inline __device__ void compute_store_pointers_e(uint32_t (&ptrs)[N], int sel_64) {
        #pragma unroll
        for( int i = 0; i < N; ++i ) {
            int offset = 0;
            if(sel_64) {
                offset = smem_write_offset_ + i * 8;    // 8 represents ldgsts64
            } else {
                offset = smem_write_offset_ + i * 4;    // 4 represents ldgsts32
            }
            ptrs[i] = smem_ + offset;
        }
    }
    
    // Move the read offset.
    inline __device__ void move_read_offset(int delta) {
        this->smem_read_offset_ += delta;
    }

    // Move the write offset.
    inline __device__ void move_write_offset(int delta) {
        this->smem_write_offset_ += delta;
    }

    inline __device__ void store(const char* gmem_ptr, int i = 0) {
        uint32_t smem_ptrs = smem_ + smem_write_offset_ + i * 32 * (LDGSTS_SIZE / 8);

        if (Cta_tile::WARPS_N == 4) {
            if (Cta_tile::K == 64 || Cta_tile::K == 32) {
                ldgsts32(smem_ptrs, gmem_ptr);
            } else {
                ldgsts64_nopreds(smem_ptrs, gmem_ptr);
            }
        } else {
            if (Cta_tile::K == 64 || Cta_tile::K == 32) {
                ldgsts64_nopreds(smem_ptrs, gmem_ptr);
            } else {
                ldgsts128_nopreds(smem_ptrs, gmem_ptr);
            }
        }
    }

    // Store to the metadata tile in shared memory.
    template< int N >
    inline __device__ void store_e(const char* (&gmem_ptrs)[N], uint32_t preds) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers_e(smem_ptrs);
        ldgsts32(smem_ptrs, gmem_ptrs, preds);
    }
    
    // Store to the metadata tile in shared memory.
    template< int N >
    inline __device__ void store_e(const char* (&gmem_ptrs)[N], uint32_t preds, int sel_64) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers_e(smem_ptrs, sel_64);
        if(sel_64)
            ldgsts64(smem_ptrs, gmem_ptrs, preds);
        else
            ldgsts32(smem_ptrs, gmem_ptrs, preds);
    }
    // Store to the metadata tile in shared memory.
    template< int N >
    inline __device__ void store_ldgsts128(const char* (&gmem_ptrs)[N], 
                                           uint32_t preds) {
        uint32_t smem_ptrs[N];
        smem_ptrs[0] = smem_ + smem_write_offset_;
        ldgsts128(smem_ptrs, gmem_ptrs, preds);
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += 
                (smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                ? -BYTES_PER_TILE_INC_BOUNDARY
                : BYTES_PER_BUFFER ;
        }
    }

    // Move the read offset to next buffer.
    inline __device__ void move_next_read_buffer() {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_read_buffer_ += 
                (smem_read_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY )
                ? -BYTES_PER_TILE_INC_BOUNDARY
                : BYTES_PER_BUFFER ;
        }
    }

    // The shared memory pointer.
    uint32_t smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The buffer base offset for read.
    int smem_read_buffer_;
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
struct Smem_tile_sparse_interleaved {

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

    //enum { LDGS_PER_XOR = Cta_tile::WARPS_N == 1 ? 2 : 1 };
    enum { LDGS_PER_XOR = (Cta_tile::WARPS_N == 1 && Cta_tile::N <=32) ? 2 : 1 };

    // The type of elements that are stored in shared memory by each thread.
    using Store_type = typename Uint_from_size_in_bytes<BYTES_PER_STS>::Type;

    // Ctor.
    inline __device__ Smem_tile_sparse_interleaved(void *smem, int tidx) 
        : smem_(xmma::get_smem_pointer(smem)) {

        // The row written by a thread. See doc/xmma_smem_layout.xlsx.
        int smem_write_row = (tidx / THREADS_PER_ROW) * LDGS_PER_XOR;

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
    template< int N, int K = 1 >
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int ii = 0; ii < N / K; ++ii ) {
            // Decompose the STS into row/col.
            int row = ii / LDGS_PER_XOR;
            int col = ii % LDGS_PER_XOR;

            // Assemble the offset.
            int offset = smem_write_offset_ +
                row*ROWS_PER_STS*BYTES_PER_ROW*LDGS_PER_XOR + col * BYTES_PER_ROW;

            offset ^= col * COLS_PER_XOR_PATTERN * BYTES_PER_STS;

            // Assemble the final pointer :)
            #pragma unroll
            for (int k = 0; k < K; k++) {
                ptrs[ii*K + k] = smem_ + offset + k * 4 + smem_write_buffer_;
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
            ldgsts<N, M, 16/K >(smem_ptrs, gmem_ptrs, preds, mem_desc);
        } else {
            #pragma unroll
            for (int ii = 0; ii < N; ii++)
            ldgsts128_nopreds(smem_ptrs[ii], gmem_ptrs[ii]);
        }
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const void* (&gmem_ptrs)[N], 
                                 uint32_t preds, 
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t tmp[1] = { preds };
        this->store(gmem_ptrs, tmp, mem_desc);
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

} // namespace xmma
