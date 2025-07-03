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

#include <xmma/helpers/epilogue.h>

namespace xmma {
namespace helpers {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Gmem_row_stride_tile_epilogue_distribution {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes per STG.
    enum { BYTES_PER_STG = 16 };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    // The number of threads per output row (row-major).
    enum { THREADS_PER_ROW = Cta_tile::N / ELEMENTS_PER_STG };
    // The number of rows that are written with a single STG (accross the CTA).
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // The iterations to fill 1 XMMA tile
    enum { SPLIT_FACTOR = (Traits::ROW_STRIDE_GROUP / Xmma_tile::XMMAS_M) / ROWS_PER_STG };

    // Compute the row.
    static inline __device__ int compute_col(int tidx) {
        return tidx % THREADS_PER_ROW;
    }

    // Compute the row.
    static inline __device__ int compute_row(int tidx) {
        return tidx / THREADS_PER_ROW;
    }

    // Compute the row offset.
    // mi -- Determined the mi iteration within a 64x16 XMMA tile
    // Split factor -- The iterations to fill a row = 16 tile region within a 64x16 XMMA tile
    static inline __device__ constexpr int compute_offset(int mi, int ii) {
        return ( mi * Xmma_tile::M_PER_XMMA +
                (ii / SPLIT_FACTOR) * Traits::ROW_STRIDE_GROUP +
                (ii % SPLIT_FACTOR) * ROWS_PER_STG );

    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>
>
struct Gmem_row_stride_tile_epilogue {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The helper class to compute the row offset (in the tile).
    using Tile_distribution = Gmem_row_stride_tile_epilogue_distribution<Traits, Cta_tile>;

    // The number of bytes per STG.
    enum { BYTES_PER_STG = 16 };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    // The number of threads per output row (row-major).
    enum { THREADS_PER_ROW = Cta_tile::N / ELEMENTS_PER_STG };
    // The number of rows that are written with a single STG (accross the CTA).
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // The number of STGS needed to output the rows produced by a CTA-wide XMMA.
    enum { STGS = Xmma_tile::M_PER_XMMA_PER_CTA / ROWS_PER_STG };

    // Ctor.
    inline __device__ Gmem_row_stride_tile_epilogue(int m, int n, int stride_n)
        : params_m_(m)
        , params_n_(n)
        , params_stride_n_(stride_n) {
    }

    // Ctor.
    inline __device__ Gmem_row_stride_tile_epilogue(
        int m, int n, int stride_n,
        char *out_ptr,
        const char* res_ptr,
        int bidm, int bidn, int bidz, int tidx)
        : params_m_(m)
        , params_n_(n)
        , params_stride_n_(stride_n) {

        // The location of the tile.
        int row = Tile_distribution::compute_row(tidx);
        int col = Tile_distribution::compute_col(tidx);

        // Compute the output position for each thread.
        m_ = bidm * Cta_tile::M + row;
        n_ = bidn * Cta_tile::N + col * ELEMENTS_PER_STG;

        // The pointer.
        int64_t offset = Traits::offset_in_bytes_c(m_*params_stride_n_ + n_);

        out_ptr_ = &out_ptr[offset];
        res_ptr_ = &res_ptr[offset];
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask(int mi, int ii) const {
        const int offset = Tile_distribution::compute_offset(mi, ii);
        return (m_ + offset) < params_m_ && n_ < params_n_;
    }

    // Load the data from global memory.
    inline __device__ void load(Fragment_c &data, int mi, int ii, int mask, const uint64_t mem_desc) {
        const int offset = Tile_distribution::compute_offset(mi, ii);
        const char *ptr = &res_ptr_[Traits::offset_in_bytes_c(offset*params_stride_n_)];
        if( mask ) {
            uint4 tmp;
            xmma::ldg(tmp, ptr, mem_desc);
            data.from_int4(tmp);
            //data.from_int4(xmma::ldg128(ptr));
        }
    }

    // Store the data to global memory.
    inline __device__ void store(int mi, int ii, const Fragment_c &data, int mask, const uint64_t mem_desc) {
        const int offset = Tile_distribution::compute_offset(mi, ii);
        char *ptr = &out_ptr_[Traits::offset_in_bytes_c(offset*params_stride_n_)];
        if( mask ) {
            xmma::stg(ptr, data.to_int4(), mem_desc);
        }
    }

    // The dimensions of the matrix.
    const int params_m_, params_n_, params_stride_n_;
    // The position of the tile.
    int m_, n_;
    // The pointer to global memory.
    char *out_ptr_;
    const char* res_ptr_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace helpers
} // namespace xmma

