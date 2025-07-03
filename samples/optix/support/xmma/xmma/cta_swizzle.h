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

/*****************************************************************************

Note well: I spent two days tuning this code, so please run the performance
tuning code in lw/cta_swizzle_test.cpp before checking in any changes to it.

Class CtaMap mimics SASS code that remaps HW's linear CTA rasterization
into something that reduces total accesses to A and B matrix data.  The
strategy is to rasterize across a specified column width of CTAs, then
move down to the next row.  Each column group must be a power of 2 wide
to keep SASS implementation, and the search space for choose_best_log2_group_cols,
reasonable.  The column group size is automatically reduced by SASS near the
right edge of the grid.  The last column group can be 3 wide because 2 then 1
is inefficient and allowing 3 wasn't too bad.)

Traversal from one column group to the next optionally serpentines, in
which case always changes direction from top to bottom, to bottom to top,
or vice-versa, even when the width of a column group has been automatically
reduced.

But the real point of the class is to then compute an *optimal* starting
column width for a specified grid and CTA wave size, which results in
the minimal amount of data fetched from A and B for the entire grid.

Since a CTA isn't always square, the CTA tile size is taken into account
when computing how much data is fetched from A and B.  We compensate for
different data sizes for A and B data.  Since LWDNN's implicit GEMM
replicates A matrix data, we account for that too, which results in waves
that appear to be tallish and narrow compared to a GEMM wave shape.

A simple strategy that tries to choose a column width that minimizes A
and B data fetched for a single CTA wave is not sufficient.  When a
CTA wave wraps around from one column group to another, it fetches more
B data.  This can make "squarish" rasterization of a wave worse than no
remapping at all!  Instead, we exhaustively try all possible column
widths, sum how much data all CTA waves fetch, and choose the best.  Here
again, a simple strategy to stop increasing the column width once we've
passed the "best" width is not sufficient...we are not looking at a
parabola-ish function, but one that bounces up and down jerkily.  I hope
the code here is fast enough that this doesn't significantly affect
kernel launch time, as the payoffs in bandwidth reduction can be quite large.

****************************************************************************/

#pragma once

#if !defined(__LWDACC_RTC__)
#include <bitset>
#include <algorithm>

#include <xmma/xmma.h>

namespace xmma {

///////////////////////////////////////////////////////////////////////////////////////////////////

// DIV can be 'H' (hardware); 'S' (software constant integer optimization),
// which is faster; or 'B' (both to check that they get the same answer)
#define DIV 'S'
#ifdef _MSC_VER
#include <intrin.h>
#endif
class Cta_swizzle {
protected:
    typedef unsigned long long uint64;


    // clz - "count leading zeros." In most modern architectures this
    // should be handled by a single instruction (e.g., clz or bsr), but C
    // offers no clean why to express it, so we must rely on
    // compiler-specific intrinsics.
    static inline unsigned clz(unsigned value) {
#if defined __GNUC__
        return __builtin_clz(value);
#elif defined _MSC_VER
        unsigned long first_zero = 0;
        if ( _BitScanReverse( &first_zero, value ) ) {
            return 31 - first_zero;
        }
        else {
            return 32;
        }
#else
#error "Unknown platform for clz"
#endif
    }

    // Protected so that it can't be used directly
    Cta_swizzle() = default;

public:
    class Pos2 {
    public:
        Pos2() = default;
        Pos2(unsigned row, unsigned col) : row(row), col(col) {}

        unsigned row, col;

        Pos2 &operator +=(const Pos2 &rhs) {
            row += rhs.row;
            col += rhs.col;
            return *this;
        }
    }; // class Pos2


    // Batched GEMM and split-k-across-CTAs must know all 3 dimensions of the
    // grid and a CTA's position. The batch CTA dimension (a.k.a. grid.Z)
    // indicates which group of A, B, C matrices to use in batched mode, or
    // which group of k indices to use in split-k-across-CTAs.  In those cases,
    // we only share data among CTAs in the same "batch".

    class Pos3 {
    public:
        Pos3() = default;
        Pos3(unsigned row, unsigned col, unsigned batch)
            : row(row), col(col), batch(batch) {}

        unsigned row, col, batch;
    }; // class Pos


    // Pos3 + modified log2_group_cols + up/down direction in group.
    // We always serpentine between adjacent columns, no matter how the
    // right edge loops through halving group_cols.
    class  Pos3X: public Pos3 {
    public:
        Pos3X(Pos3 pos, unsigned group_cols, unsigned log2_group_cols,
              bool bottom_to_top)
            : Pos3(pos), group_cols(group_cols), log2_group_cols(log2_group_cols),
              bottom_to_top(bottom_to_top) {}

        unsigned group_cols;
        unsigned log2_group_cols;
        bool     bottom_to_top;
    }; // Pos3X


    // Colwert an integer division into faster sequence of multiplication and
    // shift.  Guaranteed for all unsigned divisors except 0 (duh), and 1.
    // div by 1 returns the numerator, so we don't care if the parameters we
    // compute for it are wrong. See "Division by Ilwariant Integers using
    // Multiplication" by Torbjorn Granlund and Peter L. Montgomery at
    // https://gmplib.org/~tege/divcnst-pldi94.pdf
    class Const_div {
    public:
        Const_div(unsigned divisor = 1U) : divisor(divisor) {
            assert(divisor != 0 && "Const_div cannot divide by 0");
            ceil_log2 = 32 - clz(divisor - 1U);

            m = unsigned(((1ULL << (32+ceil_log2)) - (uint64(divisor) << 32ULL))
                         / divisor) + 1;
        } // Const_div()

        inline unsigned div(unsigned n) const {
            if (divisor == 1) return n;
            unsigned t1 = static_cast<unsigned>((uint64(m) * n) >> 32);  // I.e., IMUL.HI
            // 64-bit add, shift is faster than their 2-shift version
            unsigned quotient = static_cast<unsigned>((uint64(t1) + n) >> ceil_log2);
            return quotient;
        } // unsigned div()

    private:
        unsigned divisor, ceil_log2, m;
    }; // class Const_div


    // Max log2_group_cols is really 15, but we limit search to group_cols = 256.
    static const unsigned MAX_LOG2_GROUP_COLS = 8;

    Pos3 grid_dim;       // (row, col, batch) size of grid in CTAs
    unsigned ctas_per_batch;
    unsigned ctas_per_grid;
    Const_div const_div_grid_dim_row;
    Const_div const_div_ctas_per_batch;
    Pos2 cta_tile;       // (row, col) size of CTA in elements
    // Below parameters are only used by implicit GEMMs.
    Pos2 filter;        // (row, col) size of implicit GEMM filter.
    Pos2 colw_stride;    // (row, col) size of implicit GEMM filter's stride.
    Pos2 output_size;    // (row, col) size of implicit GEMM output.
    unsigned default_a_replication;
    int use_horizontal_cta_rasterization;

    // Construct finction for explicit GEMMs.
    Cta_swizzle(
        const Pos3 &grid_dim,
        const Pos2 &cta_tile)
      : Cta_swizzle(grid_dim, cta_tile, Pos2(1U, 1U), Pos2(1U, 1U), Pos2(1U, 1U), 0) {}

    Cta_swizzle(
        const Pos3 &grid_dim,
        const Pos2 &cta_tile,
        const int &use_horizontal_cta_rasterization)
      : Cta_swizzle(grid_dim, cta_tile, Pos2(1U, 1U), Pos2(1U, 1U), Pos2(1U, 1U),
                    use_horizontal_cta_rasterization) {}

    // Construct function for implicit GEMMs.
    Cta_swizzle(
        const Pos3 &grid_dim,
        const Pos2 &cta_tile,
        const Pos2 &filter,
        const Pos2 &colw_stride,
        const Pos2 &output_size,
        const int &use_horizontal_cta_rasterization) : grid_dim(grid_dim), cta_tile(cta_tile), filter(filter),
            colw_stride(colw_stride), output_size(output_size),
            use_horizontal_cta_rasterization(use_horizontal_cta_rasterization)
    {
#if 0
        printf("Cta_swizzle created with grid_dim(%d, %d, %d), cta_tile(%d, %d)\n",
                grid_dim.row, grid_dim.col, grid_dim.batch, cta_tile.row, cta_tile.col);
#endif
        ctas_per_batch = grid_dim.row * grid_dim.col;
        ctas_per_grid  = ctas_per_batch * grid_dim.batch;

        // For colwolutions, filter.row and filter.col can be > 1.  If the
        // input width of the activation is small enough, we'll replicate
        // data nearly filter.col / colw_stride.col times within a CTA as we
        // sweep the filter from left to right.  The activation data at the
        // bottommost row of the filter is then replicated by another factor
        // of filter.row, as we move the filter down each activation row and
        // sweep left to right again and again.  The cache should avoid
        // refetching this from the FB.  This reduces the effective "height"
        // of the CTA as seen by the FB by "nearly"
        // (filter.row / colw_stride.row) * (filter.col / colw_stride.col)

        // If the activation width is sufficiently large compared to the
        // CTA height (and it often is), then we only replicate
        // activations *within a CTA* as we create new A matrix rows by
        // sweeping the filter to the right, so replicate approximately
        // filter.col times.  By the time we reach the right edge of the
        // input data, and move the filter back to the left edge and down
        // one row, we're in another CTA below us.

        // Unfortunately, we do not know the # of CTAs per wave now.
        // So we'll just play it safe and use filter.col / colw_stride_w for
        // the degree of replication that is captured by the L1 and sometimes
        // L2 if L1 is not big enough. We'll check if there's more replication
        // later at choose_best_log2_group_cols().

        // See BUG 200497373 for more information.

        default_a_replication = std::max(1U, filter.col / colw_stride.col);
        const_div_grid_dim_row = Const_div(grid_dim.row);
        if (grid_dim.batch > 1) {
            const_div_ctas_per_batch = Const_div(ctas_per_batch);
        }
    }
    // Map a 2D position that was column-major scanline rasterized to a
    // new 2D position that is row-major scanline rasterized within
    // column groups.  This is the same mapping that cta_swizzle.py generates.
    const Pos3X map_pos3_to_swizzled(
        const Pos3     pos,
              unsigned log2_group_cols,
              bool     serpentine) const
    {
        unsigned group_cols = (1 << log2_group_cols);
#ifdef DEBUG
        if (group_cols > grid_dim.col) {
            printf("group_cols=%d > grid_dim.col=%d, which not only causes SASS to waste cycles reducing to a reasonable value, but also results in incorrect computation of up/down direction.  The client should ensure the stated condition.", group_cols, grid_dim.col);
            assert(false);
        }
        if ((pos.row >= grid_dim.row) || (pos.col >= grid_dim.col)
                || (pos.batch >= grid_dim.batch)) {
            printf("pos(%d, %d, %d) is outside of grid_dim(%d, %d, %d)\n",
                pos.row, pos.col, pos.batch,
                grid_dim.row, grid_dim.col, grid_dim.batch);
                assert(false);
        }
#endif
        unsigned mask = group_cols - 1;
        bool bottom_to_top = false;
        unsigned swiz_row, swiz_col;
        for (;;) {
            // test_col = last CTA column in group
            unsigned test_col  = pos.col | mask;

            // Are all CTA's in column group inside right edge of grid?
            bool col_in_group  = test_col < grid_dim.col;

            // bottom_to_top ^= (pos.col & group_cols) != 0
            // For starting group_cols (which we know at least one group uses,
            // due to rude assertion above), this sets
            // bottom_to_top = (pos.col / group_cols) % 2 == 1
            // That is, each odd group_cols goes bottom to top.
            // After that, it ilwerts bottom_to_top for each reduced-size
            // group_cols that gets used by some group.  It's almost magic.
            bottom_to_top = bottom_to_top != ((pos.col & group_cols) != 0);
#if 0
            printf("test_col=%d, col_in_group=%d\n", test_col, col_in_group);
#endif
            // These are not yet used if we loop, but allows better scheduling
            unsigned col_mod  = pos.col & mask;
            unsigned col_base = pos.col & ~mask;
#if 0
            printf("col_mod=%d, col_base=%d\n", col_mod, col_base);
#endif

            if (col_in_group) {
                // Linear CTA index within current group
                unsigned linear_local = col_mod * grid_dim.row + pos.row;
                // swiz_row = linear_local / group_cols
                swiz_row = linear_local >> log2_group_cols;
                // swiz_col = col_base + linear_local % group_cols
                swiz_col = col_base | (linear_local & mask);
                break;
            }

            // If we reduce rightmost group_cols to 3 is the group inside?
            // col_in_group3 = group width == 4 && in rightmost group
            //               && grid_dim.col mod 4 == 3
            bool col_in_group3 = (group_cols == 4) && (test_col == grid_dim.col);

            if (col_in_group3) {
                // Linear CTA index within current group
                unsigned linear_local = col_mod * grid_dim.row + pos.row;
                // swiz_row = linear_local / 3
                swiz_row = static_cast<unsigned>((uint64(linear_local) * 0x55555556) >> 32ULL);
                // swiz_col = col_base + linear_local % 3
                swiz_col = col_base - swiz_row * 3 + linear_local;
                break;
            }

            // Next iteration.
            group_cols >>= 1;
            mask >>= 1;
            log2_group_cols -= 1;
        } // end for

#if 0
        printf("linear_local=%d, col_in_group3=%d\n", linear_local, col_in_group3);
#endif

        bottom_to_top &= serpentine;
        if (bottom_to_top) {
            swiz_row = grid_dim.row - 1 - swiz_row;
        }
        return Pos3X(Pos3(swiz_row, swiz_col, pos.batch),
                    group_cols, log2_group_cols, bottom_to_top);
    } // map_pos3_to_swizzled


    // Map column-major linearized CTA index to a swizzled 3D position
    // This is useful for mapping the first and last CTA in a wave.
    const Pos3X map_linear_to_swizzled(
        const unsigned index,
        const unsigned log2_group_cols,
        const bool     serpentine) const
    {
        // First map to a 3D position.  Rasterization proceeds row by row
        // down a column, then across column by column, then batch by batch.
        // (Remember, row = x and col = y.)
        unsigned batch    = 0;
        unsigned batch_mod = index;
        if (index >= ctas_per_batch) {
            // Only needed for batched GEMM/split-k-across-CTAs
#if DIV == 'H'
            batch    = index / ctas_per_batch;
            batch_mod = index % ctas_per_batch;
#elif DIV == 'S'
            batch    = const_div_ctas_per_batch.div(index);
            batch_mod = index - batch * ctas_per_batch;
#elif DIV == 'B'
            batch    = index / ctas_per_batch;
            batch_mod = index % ctas_per_batch;
            unsigned batch2    = const_div_ctas_per_batch.div(index);
            unsigned batch_mod2 = index - batch * ctas_per_batch;
            if ((batch != batch2) || (batch_mod != batch_mod2)) {
                printf("Const div failed: batch = %d, batch2 = %d, batch_mod = %d, batch_mod2 = %d\n",
                    batch, batch2, batch_mod, batch_mod2);
                assert(false);
            }
#endif
        }
#if DIV == 'H'
        unsigned col = batch_mod / grid_dim.row;
        unsigned row = batch_mod % grid_dim.row;
#elif DIV == 'S'
        unsigned col = const_div_grid_dim_row.div(batch_mod);
        unsigned row = batch_mod - col * grid_dim.row;
#elif DIV == 'B'
        unsigned col = batch_mod / grid_dim.row;
        unsigned row = batch_mod % grid_dim.row;
        unsigned col2 = const_div_grid_dim_row.div(batch_mod);
        unsigned row2 = batch_mod - col * grid_dim.row;
        if ((col != col2) || (row != row2)) {
            printf("Const div failed: col = %d, col2 = %d, row = %d, row2 = %d\n",
                col, col2, row, row2);
            assert(false);
        }
#endif

        return map_pos3_to_swizzled(Pos3(row, col, batch), log2_group_cols,
            serpentine);
    }


    // Compute # of CTAs high in A and wide in B, fetched by CTAs in same batch
    Pos2 batch_cta_fetches(
        const Pos3X &first,
        const Pos3X &last,
        const bool serpentine) const
    {
        if (first.batch != last.batch) {
            printf("first.batch=%d != last.batch=%d\n",
                first.batch, last.batch);
            assert(false);
        }
        unsigned first_mask = first.group_cols - 1;
        // First column in the current group
        unsigned first_b_col_base = first.col & ~first_mask;

        // The last CTA in the wave is farther to the right, and so may be
        // in an automatically narrowed column group smaller than first.
        unsigned last_mask = last.group_cols - 1;
        unsigned last_b_col_base = last.col & ~last_mask;

        // How many columns of B data?  From the above assertion, we know
        // that we'll always span an entire B column group with the wave,
        // unless we're in the "last" row of the column group.
        bool first_spans_col_group = (first.bottom_to_top && (first.row > 0))
            || (!first.bottom_to_top && (first.row < (grid_dim.row-1)));
        unsigned first_b_col = first_spans_col_group ? first_b_col_base : first.col;

        // Similar logic applies here.  We'll always span an entire B
        // column group, unless we're in the "first" row of the column.
        bool last_spans_col_group =
               (last.bottom_to_top && (last.row < (grid_dim.row-1)))
            || (!last.bottom_to_top && (last.row > 0));
        unsigned last_b_col = last_spans_col_group ? (last.col | last_mask)
                                              : last.col;
        // Compensate for possible group_cols == 3
        if (last_b_col == grid_dim.col)  last_b_col--;
        unsigned b_cols = last_b_col - first_b_col + 1;

        // A is trickier.
        unsigned a_rows;
        if (first_b_col_base == last_b_col_base) {
            // Same column group.  Span from one position to the other.
            a_rows = std::abs(int(last.row) - int(first.row)) + 1;
        } else if ((first_b_col_base + first.group_cols) == last_b_col_base){
            // Adjacent column groups.
            if (serpentine) {
                // Use the larger of the span of the first CTA to the "last"
                // row in its column, and "first" row of the next column to
                // the last CTA.
#ifdef DEBUG
                assert((first.bottom_to_top != last.bottom_to_top)
                     && "Somehow serpentine code got confused");
#endif
                if (first.bottom_to_top) {
                    a_rows = std::max(first.row, last.row) + 1;
                } else {
                    a_rows = std::max(grid_dim.row - first.row,
                                     grid_dim.row - last.row);
                }
            } else {
                // Both columns groups going top to bottom, so sum of first
                // CTA to bottom of column and top of column to last CTA.
#ifdef DEBUG
                assert(!first.bottom_to_top && !last.bottom_to_top
                     && "Somehow serpentine code got confused");
#endif
                a_rows = std::min(grid_dim.row,
                                 (grid_dim.row - first.row + 1) + last.row);
            } // if serpentine ... else ...
        } else {
            // At least one complete column in middle of wave, so span all
            // of grid height.
            a_rows = grid_dim.row;
        } // if various A matrix cases

        return Pos2(a_rows, b_cols);
    } // batch_cta_fetches


    // Compute # of CTAs high in A, and wide in B, fetched by all CTA waves
    // in the grid.  We don't share information between batches, so we must
    // compute this independently for each batch index.
    Pos2 grid_cta_fetches(
        const unsigned ctas_per_wave,
        const unsigned log2_group_cols,
        const bool     serpentine)
    {
#ifdef DEBUG
        unsigned group_cols = 1 << log2_group_cols;
        if ((ctas_per_wave <= group_cols) && (group_cols > 1)) {
            printf("It makes no sense to have ctas_per_wave=%d <= group_cols=%d, as the whole point is to share *both* A and B data.  Further, this code depends upon this condition.", ctas_per_wave, group_cols);
        }
#endif
        Pos2 total = Pos2(0, 0);
        for (unsigned first_linear = 0;
             first_linear < ctas_per_grid;
             first_linear += ctas_per_wave) {

            Pos3X first = map_linear_to_swizzled(first_linear, log2_group_cols,
                serpentine);
            unsigned last_linear =
                std::min(first_linear+ctas_per_wave, ctas_per_grid) - 1;
            Pos3X last = map_linear_to_swizzled(last_linear, log2_group_cols,
                serpentine);

            Pos2 wave_total = Pos2(0, 0);
            if (first.batch == last.batch) {
                wave_total = batch_cta_fetches(first, last, serpentine);
            } else {
                // Split work into 3 parts
                // From first to the last CTA in the first batch
                unsigned last_linear_in_batch =
                    first.batch * ctas_per_batch + ctas_per_batch - 1;
                Pos3X last_in_batch = map_linear_to_swizzled(last_linear_in_batch,
                    log2_group_cols, serpentine);
                wave_total = batch_cta_fetches(first, last_in_batch, serpentine);
                // Completely covered batches, if any, between first and last
                unsigned factor = last.batch - first.batch - 1;
                wave_total += Pos2(factor * grid_dim.row, factor * grid_dim.col);
                // From the first CTA in the last batch to last
                unsigned first_linear_in_batch = last.batch * ctas_per_batch;
                Pos3X first_in_batch = map_linear_to_swizzled(first_linear_in_batch,
                    log2_group_cols, serpentine);
                wave_total += batch_cta_fetches(first_in_batch, last, serpentine);
            }

#if defined(DEBUG)
           printf("log2 = %d, ctas %3d..%3d, batch = %d..%d: A height = %d, B width = %d\n",
               log2_group_cols, first_linear, last_linear, first.batch, last.batch,
               wave_total.row, wave_total.col);
#endif
            total += wave_total;
        } // end for first_cta

        return total;
    } // grid_cta_fetches

    // Find the best value for log2_group_cols for this grid and CTA wave
    // size.  At least theoretically, serpentining is never worse then
    // always going top to bottom, so that's hardwired here.
    unsigned choose_best_log2_group_cols(unsigned ctas_per_wave) {
        if (ctas_per_wave == 0) {
          // CTA swizzle is disabled.
          return 0;
        }
        if (ctas_per_wave >= (grid_dim.batch * ctas_per_batch - 1)) {
            // No possible way to help by swizzling ids.
            return 0;
        }
        unsigned best_log2_group_cols = 0;
        unsigned best_fetches = ~0;
        unsigned waves_per_grid = (ctas_per_grid + ctas_per_wave / 2 ) / ctas_per_wave;
        for (unsigned log2_group_cols = 0;
             log2_group_cols <= MAX_LOG2_GROUP_COLS;
             log2_group_cols++) {
            unsigned group_cols = 1 << log2_group_cols;
            if (grid_dim.col != 3 && group_cols > grid_dim.col) break;
            // For grid width = 3, we will evaluate group_cols = 4 to see
            // if we can fetch less tiles
            // See https://jirasw.lwpu.com/browse/CFK-257
            if (grid_dim.col == 3 && group_cols > 4) break;
            if (group_cols >= ctas_per_wave) break;
#if 0
            // This is blindingly faster than the exhaustive code, but doesn't
            // account for a wave wrapping around to the next column group at
            // the bottom or top of the grid, and so can choose too wide.
            Pos2 cta_fetches((ctas_per_wave + group_cols - 1) / group_cols,
                            group_cols);
            if (cta_fetches.row > grid_dim.row) {
                // CTA wave oclwpies multiple col groups, can even be one more
                // col group than we compute here
                cta_fetches.col = (ctas_per_wave + grid_dim.row + 1) / grid_dim.row;
                cta_fetches.row = grid_dim.row;
            }
#else
            Pos2 cta_fetches = grid_cta_fetches(ctas_per_wave, log2_group_cols, true);
#endif
            unsigned a_replication = default_a_replication;
            if (filter.col != 1 || filter.row != 1) {
                unsigned row_fetched_per_wave(cta_fetches.row * cta_tile.row / waves_per_grid);
                if (output_size.col < row_fetched_per_wave) {
                    a_replication *= std::max(1U, std::min(row_fetched_per_wave / output_size.col, filter.row) / colw_stride.row);
                }
            }
            // Scale fetches up by CTA tile dimensions, and scale A fetches
            // down by degree of filter replication (for implicit GEMM).
            unsigned fetches = 0;
            if ( !use_horizontal_cta_rasterization ) {
                fetches = cta_fetches.row * cta_tile.row / a_replication
                        + cta_fetches.col * cta_tile.col;
            } else {
                fetches = cta_fetches.row * cta_tile.row
                        + cta_fetches.col * cta_tile.col / a_replication;
            }
#if defined(DEBUG)
            printf("A total = %d, B total = %d, scaled fetches = %d\n",
                   cta_fetches.row, cta_fetches.col, fetches);
#endif
            if (fetches < best_fetches) {
                best_fetches = fetches;
                best_log2_group_cols = log2_group_cols;
            } else if (fetches > (best_fetches * 5 / 4)) {
                // Unlikely that we'll manage to get better from here
                break;
            }
        } // end for log2_group_cols;
        return best_log2_group_cols;
    } // chose_best_log2_group_cols
}; // class Cta_swizzle


// Hopper Specific CGA Rasterization based on Group-Cols
class Cga_swizzle : public Cta_swizzle {

public :

    using Base = Cta_swizzle;

    Pos2 cga_tile;       // (row, col) size of CGA in elements

    Cga_swizzle() {};

    Cga_swizzle(
        const Pos3 &grid_dim,
        const Pos2 &cta_tile,
        const Pos2 &cga_tile,
        const int &use_horizontal_cta_rasterization)
        : Base(grid_dim, cta_tile, use_horizontal_cta_rasterization) {}

    // Find the best value for log2_group_cols for this CGA grid and CTA wave
    // size.  At least theoretically, serpentining is never worse then
    // always going top to bottom, so that's hardwired here.

    // For hopper with CGAs enabled, The Group Cols CTA width needs to be at-least same
    // as the CGA width(in terms of CTAs), since we are launching CGAs and not CTAs.
    // One benefit in hopper is that CGA width divides the grid width perfectly.
    // So there are no "left-overs" / corner cases like the regular CTA launch.
    unsigned choose_best_log2_group_cols(unsigned ctas_per_wave) {

        // we atleast need to launch 1 CGA Col wide CTAs
        uint32_t group_col_width = cga_tile.col;
        uint32_t log2_cga_col = static_cast<uint32_t>(log2(static_cast<float>(group_col_width)));

        // Something is wrong - Grid width needs to be atleast 1 CGA wide
        if( group_col_width < grid_dim.col ) {
            assert(false);
        }

        if ( ctas_per_wave == 0 ) {
          // CTA swizzle is disabled.
          return group_col_width;
        }

        unsigned best_log2_group_cols = log2_cga_col;
        unsigned best_fetches = ~0;
        unsigned waves_per_grid = (ctas_per_grid + ctas_per_wave / 2 ) / ctas_per_wave;

        for (unsigned log2_group_cols = log2_cga_col;
             log2_group_cols <= MAX_LOG2_GROUP_COLS;
             log2_group_cols++) {

            unsigned group_cols = 1 << log2_group_cols;
            if (grid_dim.col != 3 && group_cols > grid_dim.col) break;
            if (grid_dim.col == 3 && group_cols > 4) break;
            if (group_cols >= ctas_per_wave) break;

            Pos2 cta_fetches = grid_cta_fetches(ctas_per_wave, log2_group_cols, true);

            unsigned a_replication = default_a_replication;
            if (filter.col != 1 || filter.row != 1) {
                unsigned row_fetched_per_wave(cta_fetches.row * cta_tile.row / waves_per_grid);
                if (output_size.col < row_fetched_per_wave) {
                    a_replication *= std::max(1U, std::min(row_fetched_per_wave / output_size.col, filter.row) / colw_stride.row);
                }
            }

            // Scale fetches up by CTA tile dimensions, and scale A fetches
            // down by degree of filter replication (for implicit GEMM).
            unsigned fetches = 0;
            if ( !use_horizontal_cta_rasterization ) {
                fetches = cta_fetches.row * cta_tile.row / a_replication
                        + cta_fetches.col * cta_tile.col;
            } else {
                fetches = cta_fetches.row * cta_tile.row
                        + cta_fetches.col * cta_tile.col / a_replication;
            }

#if defined(DEBUG)
            printf("A total = %d, B total = %d, scaled fetches = %d\n",
                   cta_fetches.row, cta_fetches.col, fetches);
#endif

            if (fetches < best_fetches) {
                best_fetches = fetches;
                best_log2_group_cols = log2_group_cols;
            } else if (fetches > (best_fetches * 5 / 4)) {
                // Unlikely that we'll manage to get better from here
                break;
            }
        } // end for log2_group_cols;
        return best_log2_group_cols;
    } // chose_best_log2_group_cols
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma
#endif
