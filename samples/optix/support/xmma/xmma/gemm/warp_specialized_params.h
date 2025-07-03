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
#include <xmma/params.h>
#include <xmma/cta_swizzle.h>
#include <xmma/warp_specialized_traits.h>

#include <xmma/gemm/params.h>

namespace xmma {
namespace gemm {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Warp_specialized_params : public xmma::gemm::Params<Traits, Cta_tile, 1> {
    using Base = xmma::gemm::Params<Traits, Cta_tile, 1>;

    // lic+
    // warp specialized parameters
    // which warp specialization mode
    int32_t specialize;
    // used in ping-pong mode
    int32_t buffers_img, buffers_flt, buffers_epilog;
    int32_t delta_flt_head, delta_img_head;
    // steps for tile move
    int32_t tile_move_step, tile_move_step_x, tile_move_step_cga;
    // sm id used for shared memory capacity.
    int32_t sm;
    // The number of CTA tiles in each dimension.
    int32_t tiles_mn, tiles_all;
    // precomputed values for fast_divmod
    uint32_t mul_grid_yx, shr_grid_yx, mul_grid_x, shr_grid_x;
    // CGA Specific for hopper
    uint32_t num_cga_tile_m, num_cga_tile_n;

#if !defined(__LWDACC_RTC__)
    template<typename Kernel_traits>
    XMMA_HOST xmma::Error finalize(const dim3 &grid_) {
        this->ampere = std::is_same<typename Traits::Traits::Gpu_arch, xmma::Ampere>::value;
        this->hopper = std::is_same<typename Traits::Traits::Gpu_arch, xmma::Hopper>::value;

        if(this->finalize_problem(grid_) != xmma::Error::SUCCESS) {
            return xmma::Error::ERROR_ILWALID_PARAMS;
        }

        if(this->finalize_performance(grid_) != xmma::Error::SUCCESS) {
            return xmma::Error::ERROR_ILWALID_PARAMS;
        }

        this->specialize = Kernel_traits::WARP_SPECIALIZED_CONFIG;
        this->loop_residue = 1;

        // 2math config
        if( this->specialize == xmma::CONFIG_1DMA_2MATH ) {
            int32_t buffers_a = Kernel_traits::BUFFERS_PER_SMEM_TILE_A;
            int32_t buffers_b = Kernel_traits::BUFFERS_PER_SMEM_TILE_B;
            this->delta_img_head = buffers_a > 0 ? ( this->loop_start + 1 ) % buffers_a : 0;
            this->delta_flt_head = buffers_b > 0 ? ( this->loop_start + 1 ) % buffers_b : 0;
        }

        return xmma::Error::SUCCESS;
    }

    // FIXME: depcreated
    template<typename Kernel_traits>
    XMMA_HOST xmma::Error initialize( const Host_workspace<Kernel_traits> *workspace ) {
        return this->finalize<Kernel_traits>(workspace->grid);
    }

    XMMA_HOST xmma::Error callwlate_grid_dimensions( dim3 &grid, dim3 &split_k_grid, int32_t xmmas_m ) {

        xmma::Error err = xmma::Error::SUCCESS;

        // What if this is called before initialize ?
        this->hopper = std::is_same<typename Traits::Traits::Gpu_arch, xmma::Hopper>::value;

        // All hopper WS kernels will use CGAs
        if( this->hopper ) {

            err = callwlate_grid_dimensions_hopper( grid, split_k_grid, xmmas_m );  

        } else {
            lwdaDeviceProp props;
            XMMA_LWDA_CALL( lwdaGetDeviceProperties( &props, 0 ) );
            // persistent CTA:  1CTA/SM.
            this->tile_move_step = props.multiProcessorCount;
            this->sm = props.major * 10 + props.minor;

            // Compute tiles along each dimension.
            this->tiles_m = xmma::div_round_up( this->m, Cta_tile::M );
            this->tiles_n = xmma::div_round_up( this->n, Cta_tile::N );
            this->tiles_k = this->split_k.slices;

            tiles_mn = this->tiles_m * this->tiles_n;

            if( this->batch.is_batched ) {
                tiles_all = tiles_mn * this->batch.batches;
            } else {
                tiles_all = tiles_mn * this->split_k.slices;
            }

            // Precomputed values for fast_div in persistent tile distributrion
            xmma::find_divisor( mul_grid_yx, shr_grid_yx, tiles_mn );
            xmma::find_divisor( mul_grid_x, shr_grid_x,
                                this->use_horizontal_cta_rasterization ? this->tiles_n : this->tiles_m );

            // Warp specialization allocates 1 CTA/SM.
            grid.x = tile_move_step < tiles_all ? tile_move_step : tiles_all;
            grid.y = grid.z = 1;

            if( this->use_horizontal_cta_rasterization ) {
                this->tiles_y = this->tiles_m;
                this->tiles_x = this->tiles_n;
                split_k_grid.y = this->tiles_m;
                split_k_grid.x = this->tiles_n;
            } else {
                this->tiles_x = this->tiles_m;
                this->tiles_y = this->tiles_n;
                split_k_grid.x = this->tiles_m;
                split_k_grid.y = this->tiles_n;
            }
            split_k_grid.z = xmmas_m;
        }

        return err;
    }

    // For hopper - we can atmost launch a multiple of CGA shape number of CTAs
    // Hence we use a different grid callwlation mechanism
    XMMA_HOST xmma::Error callwlate_grid_dimensions_hopper( dim3 &grid, dim3 &split_k_grid, 
                                                            int32_t xmmas_m ) {

        lwdaDeviceProp props;
        XMMA_LWDA_CALL( lwdaGetDeviceProperties( &props, 0 ) );

        uint32_t cta_per_cga = this->cluster_height * this->cluster_width;
        uint32_t num_sm = props.multiProcessorCount; 

        // persistent CTA:  1CTA/SM.
        // Move is in terms of CGAs now
        // NOTE :: We assume each GPC has 18 fully functional SMs to callwlate grid size
        constexpr uint32_t MIN_SM_PER_GPC = 18;
        uint32_t cta_oclwpancy_per_gpc = MIN_SM_PER_GPC - MIN_SM_PER_GPC % cta_per_cga;
        uint32_t num_gpc_per_device = num_sm / MIN_SM_PER_GPC;

        // TODO : Dislwssion ongoing with the driver team on this - for a fix in the long-run
        if( ((num_sm % MIN_SM_PER_GPC) != 0) &&
            ((this->cluster_height * this->cluster_width) > 1) ) {
            printf("WARNING : Hopper WS CGA Kernels assume for now non-floorswept GH100 config.\n"
                   "Using CGAs in a Floor-swept config can lead to performance degradations\n");
        }

        this->tile_move_step = num_gpc_per_device * cta_oclwpancy_per_gpc;
        this->tile_move_step_x = this->tile_move_step / this->cluster_height;
        this->tile_move_step_cga = this->tile_move_step /
                                        (this->cluster_height * this->cluster_width);
        this->sm = props.major * 10 + props.minor;

        // Compute tiles along each dimension.
        this->tiles_m = xmma::div_round_up( this->m, Cta_tile::M );
        this->tiles_n = xmma::div_round_up( this->n, Cta_tile::N );
        this->tiles_k = this->split_k.slices;

        // Num CGAs Tiles in the orignal Tile grid (tiles_m x tiles_n)
        this->num_cga_tile_m = this->tiles_m / this->cluster_height;
        this->num_cga_tile_n = this->tiles_n / this->cluster_width;

        tiles_mn = this->tiles_m * this->tiles_n;

        // For now not supporting Split-K, Batched WS kernel
        if( this->batch.is_batched ) {
            tiles_all = tiles_mn * this->batch.batches;
            assert(this->batch.batches == 1);
        } else {
            tiles_all = tiles_mn * this->split_k.slices;
            assert(this->split_k.slices == 1);
        }

        // Warp specialization allocates 1 CTA/SM.
        // In the case of launching CGAs, we launch (K x 1) CGAs worth CTAs
        // Enough to maximally occupy the GPU
        grid.y = this->cluster_height;
        grid.x = (tile_move_step < tiles_all) ? tile_move_step_x : (tiles_all / this->cluster_height);
        grid.z = 1;

        // This is used in the kernel to setup the DMA warps - so needs to be modified for CGAs
        this->tile_move_step = tile_move_step_x;

        if( this->use_horizontal_cta_rasterization ) {
            this->tiles_y = this->tiles_m;
            this->tiles_x = this->tiles_n;
            split_k_grid.y = this->tiles_m;
            split_k_grid.x = this->tiles_n;
        } else {
            // TODO : Verical Raster
        }
        split_k_grid.z = xmmas_m;

        return xmma::Error::SUCCESS;
    }
#endif // __LWDACC_RTC__

};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace xmma
