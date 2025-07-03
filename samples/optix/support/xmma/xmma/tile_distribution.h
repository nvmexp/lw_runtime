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

#include <xmma/utils.h>

namespace xmma {

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Tile_distribution {

    // Ctor.
    template< typename Params >
    inline __device__ Tile_distribution(const Params &params)
        : params_tiles_m_(params.tiles_m)
        , params_tiles_n_(params.tiles_n)
        , params_use_horizontal_cta_rasterization_(params.use_horizontal_cta_rasterization)
        , params_ctas_per_wave_(params.ctas_per_wave)
        , params_best_log2_group_cols_(params.best_log2_group_cols)
        , params_tiles_x_(params.use_horizontal_cta_rasterization
                          ? params_tiles_n_ : params_tiles_m_)
        , params_tiles_y_(params.use_horizontal_cta_rasterization
                          ? params_tiles_m_ : params_tiles_n_)
        , params_cluster_height_(params.cluster_height)
        , params_cluster_width_(params.cluster_width) {
    }

    // Ctor.
    template< typename Params >
    inline __device__ Tile_distribution(const Params &params, const dim3 &bid)
        : Tile_distribution(params) {

        int bid_x = bid.x;
        int bid_y = bid.y;
        int bid_z = bid.z;

        // Apply L2 swizzling.
        if( params.ctas_per_wave > 0 ) {
            swizzle(bid_x, bid_y, params_best_log2_group_cols_);
        }

        // Define the row/col.
        if( params_use_horizontal_cta_rasterization_ ) {
            tile_m_ = bid_y;
            tile_n_ = bid_x;
        } else {
            tile_m_ = bid_x;
            tile_n_ = bid_y;
        }
        tile_z_ = bid_z;
    }


    // Apply CTA swizzling.
    inline __device__ void swizzle(int &bid_x, int &bid_y, int log2_group_cols) {

        // The corresponding mask.
        unsigned mask = (1u << log2_group_cols) - 1;
        // Is the code going bottom-to-top or top-to-bottom? Be prepared to change your mind.
        unsigned bottom_to_top = 0;

        // The row/col. Start with the CTA position.
        unsigned row = bid_x;
        unsigned col = bid_y;

        // Loop until we find the correct swizzling.
        for( ;; mask /= 2, --log2_group_cols ) {
            // test_col is the last column in group.
            unsigned test_col = col | mask;

            // bottom_to_top ^= (pos.col & group_cols) != 0
            // For starting group_cols (which we know at least one group uses, due to rude assertion
            // above), this sets
            // bottom_to_top = (pos.col / group_cols) % 2 == 1
            // That is, each odd group_cols goes bottom to top.
            // After that, it ilwerts bottom_to_top for each reduced-size group_cols that gets used
            // by some group.  It's almost magic.
            if( col & (1u << log2_group_cols) ) {
                bottom_to_top = !bottom_to_top;
            }

            if( test_col < params_tiles_y_ ) {
                // Linear CTA index within current group.
                unsigned linear_local = (col & mask) * params_tiles_x_ + row;
                // row = linear_local / group_cols.
                row = linear_local >> log2_group_cols;
                // col = col_base + linear_local % group_cols.
                col = (col & ~mask) | (linear_local & mask);
                // We are done!
                break;
            }

            // If we reduce rightmost group_cols to 3 is the group inside?
            // colInGroup3 = group width == 4 && in rightmost group && gridDim.col mod 4 == 3
            if( log2_group_cols == 2 && test_col == params_tiles_y_ ) {
                // Linear CTA index within current group
                unsigned linear_local = (col & mask) * params_tiles_x_ + row;
                // row = linearLocal / 3
                row = (uint64_t(linear_local) * 0x55555556ULL) >> 32;
                // col = colBase + linearLocal % 3
                col = (col & ~mask) - row * 3 + linear_local;
                // We are done!
                break;
            }
        } // end for

        // If we go bottom-to-top, we switch the row.
        if( bottom_to_top ) {
            row = params_tiles_x_ - 1 - row;
        }

        // The final position.
        bid_x = row;
        bid_y = col;
    }

    // Apply CGA swizzling.
    inline __device__ void swizzle_cga(int &bid_x, int &bid_y, int log2_group_cols) {

        // TODO : Verify
        // The corresponding mask.
        unsigned mask = (1u << log2_group_cols) - 1;
        // Is the code going bottom-to-top or top-to-bottom? Be prepared to change your mind.
        unsigned bottom_to_top = 0;

        // The row/col. Start with the CTA position.
        unsigned row = bid_x;
        unsigned col = bid_y;

        unsigned group_col_width = 1 << log2_group_cols;
        unsigned cga_y_in_group_col = group_col_width / params_cluster_height_;
        unsigned cga_x_in_group_col = params_tiles_x_ / params_cluster_width_;

        // Simpler names
        const unsigned &cga_height = params_cluster_height_;
        const unsigned &cga_width = params_cluster_width_;

        // Loop until we find the correct swizzling.
        for( ;; mask /= 2, --log2_group_cols ) {
            // test_col is the last column in group.
            unsigned test_col = col | mask;

            // bottom_to_top ^= (pos.col & group_cols) != 0
            // For starting group_cols (which we know at least one group uses, due to rude assertion
            // above), this sets
            // bottom_to_top = (pos.col / group_cols) % 2 == 1
            // That is, each odd group_cols goes bottom to top.
            // After that, it ilwerts bottom_to_top for each reduced-size group_cols that gets used
            // by some group.  It's almost magic.
            if( col & (1u << log2_group_cols) ) {
                bottom_to_top = !bottom_to_top;
            }

            if( test_col < params_tiles_y_ ) {
                // Linear CGA index within current group.
                unsigned linear_cga_local = ((col & mask) / params_cluster_height_) * cga_x_in_group_col 
                                            + (row / cga_height);
                unsigned cga_row = linear_cga_local / cga_y_in_group_col;
                unsigned cga_col = linear_cga_local % cga_y_in_group_col 
                                   + ((col & ~mask) / cga_height);

                // row, col = cga offset + offset within cga
                row = cga_row * cga_width + row % cga_width;
                col = cga_col * cga_height + col % cga_height;
                // We are done!
                break;
            }
        } // end for

        // If we go bottom-to-top, we switch the row.
        if( bottom_to_top ) {
            row = params_tiles_x_ - 1 - row;
        }

        // The final position.
        bid_x = row;
        bid_y = col;
    }

    // The tile index in M.
    inline __device__ int bidm() const {
        return tile_m_;
    }

    // The tile index in N.
    inline __device__ int bidn() const {
        return tile_n_;
    }

    // Pack the block indices.
    inline __device__ dim3 bidx() const {
        return dim3(this->bidm(), this->bidn(), this->bidz());
    }

    // The tile index in Z. Often used for either split-k or batching.
    inline __device__ int bidz() const {
        return tile_z_;
    }

protected:
    // Do we use horizontal rasterization.
    const int params_use_horizontal_cta_rasterization_;
    // The number of tiles in the M/N dimensions. Those are kernel parameters.
    const int params_tiles_m_, params_tiles_n_;
    // The number of cols/rows of CTAs in grid. Those are kernel parameters.
    const int params_tiles_x_, params_tiles_y_;
    // The number of ctas per wave if we want to trigger L2-friendly rasterization.
    const int params_ctas_per_wave_;
    // The number of columns per group for L2-friendly rasterization.
    const int params_best_log2_group_cols_;
    // The index for the current tile in the M/N/Z dimension of the grid.
    int tile_m_, tile_n_, tile_z_;
    // CGA Dimensions
    const uint32_t params_cluster_height_, params_cluster_width_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Tile_distribution_persistent : public Tile_distribution {

    using Base = Tile_distribution;

    template< typename Params >
    inline __device__ Tile_distribution_persistent(const Params &params, int tile)
        : Base(params)
        , params_tiles_(params.tiles_all)
        , params_tiles_yx_(params.tiles_mn)
        , params_tile_move_step_(params.tile_move_step)
        , params_mul_tiles_yx_(params.mul_grid_yx)
        , params_shr_tiles_yx_(params.shr_grid_yx)
        , params_mul_tiles_x_(params.mul_grid_x)
        , params_shr_tiles_x_(params.shr_grid_x)
        , tile_(tile) {

        // Compute the grid position.
        this->move_();
    }

    // Move the tile to its next position.
    inline __device__ void move() {
        this->move_(params_tile_move_step_);
    }

    // Is it the last tile?
    inline __device__ bool is_last() {
        return tile_ >= params_tiles_;
    }

private:
    // Move the tile.
    inline __device__ void move_(int move_step = 0) {
        // Update the linear index.
        tile_ += move_step;

        // Compute the 3D grid position from the linear index.
        int tile_z, tile_yx;
        xmma::fast_divmod(tile_z, tile_yx, tile_, params_tiles_yx_,
                                                      params_mul_tiles_yx_,
                                                      params_shr_tiles_yx_);

        // We use X/Y instead of M/N as it depends on whether we use horiz/vertical rasterization.
        int tile_y, tile_x;
        xmma::fast_divmod(tile_y, tile_x, tile_yx, params_tiles_x_,
                                                       params_mul_tiles_x_,
                                                       params_shr_tiles_x_);

        // Initialize the tile indices considering horizontal rasterization.
        if( params_use_horizontal_cta_rasterization_ ) {
            this->tile_m_ = tile_y;
            this->tile_n_ = tile_x;
        } else {
            this->tile_m_ = tile_x;
            this->tile_n_ = tile_y;
        }

        // Store Z as well.
        this->tile_z_ = tile_z;
    }

    // The number of total tiles and the number of tiles in the MxN plane.
    const int params_tiles_, params_tiles_yx_;
    // The step to move from the current linear tile position to the next one.
    const int params_tile_move_step_;
    // Precomputed values for fast_divmod for M/N.
    const uint32_t params_mul_tiles_yx_, params_shr_tiles_yx_;
    // Precomputed values for fast_divmod for M/N.
    const uint32_t params_mul_tiles_x_, params_shr_tiles_x_;
    // The 1D linear tile index.
    int tile_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// In Hopper, we need to have 2D persistent CTAs - since we try to launch CGAs
// This means that we need to have a multiple of CGAs launched persistently
struct Tile_distribution_persistent_hopper : public Tile_distribution {

    using Base = Tile_distribution;

    template< typename Params >
    inline __device__ Tile_distribution_persistent_hopper(const Params &params, int tile_x)
        : Base(params)
        // Number of CTA Tiles (original grid)
        , params_tiles_(params.tiles_all)
        // Number of CGA Tiles almong N
        , params_cga_x_(params.num_cga_tile_n)
        // Number of CGA Tiles almong M
        , params_cga_y_(params.num_cga_tile_m)
        // Number of CGA tiles "launched"
        , params_tile_move_step_cga_(params.tile_move_step_cga)
        // CGA Height
        , cluster_height(params.cluster_height)
        // CGA Width
        , cluster_width(params.cluster_width)
        // BlockIdx.x in the current launched grid
        , tile_x(tile_x)
        // BlockIdx.y in the current launched grid
        , tile_y(blockIdx.y) {

        if( params_use_horizontal_cta_rasterization_ ) {
            // Idea here is to find the CGA Linear ID Local from the block X
            // Since it is a 1D CGA launch - Linear id = bid.x / cga_width
            cga_linear_id_ = tile_x / cluster_width;
        } else {
            // TODO : Vertical Raster
        }

        this->move_();

    }

    // Move the tile to its next position. (in terms of CGAs)
    inline __device__ void move() {
        this->move_(params_tile_move_step_cga_);
    }

    // Is it the last tile?
    inline __device__ bool is_last() {
        return cga_linear_id_ >= (params_tiles_ / (cluster_width * cluster_height));
    }

private:
    // Move the tile 1 step
    inline __device__ void move_(uint32_t val = 0) {

        cga_linear_id_ += val; 

        // Initialize the tile indices considering horizontal rasterization.
        if( params_use_horizontal_cta_rasterization_ ) {
            uint32_t cga_final_x = cga_linear_id_ % params_cga_x_;
            uint32_t cga_final_y = cga_linear_id_ / params_cga_x_;

            this->tile_m_ = tile_y % cluster_height + cga_final_y * cluster_height;
            this->tile_n_ = tile_x % cluster_width  + cga_final_x * cluster_width;
        } else {
            // TODO : Vertical Raster
        }

        // Store Z as well.
        this->tile_z_ = 0;
    }

    // The number of total tiles and the number of tiles in the MxN plane.
    const int params_tiles_;
    const int params_cga_x_, params_cga_y_;
    // The step to move from the current tile position to the next one.
    const int params_tile_move_step_cga_;
    // The 2D tile index in the "launched" grid. (X is horizontal, Y is vertical)
    const int tile_y, tile_x;
    // CGA params
    const int cluster_width, cluster_height;
    uint32_t cga_linear_id_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

