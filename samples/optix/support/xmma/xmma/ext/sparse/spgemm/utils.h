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

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace gemm {

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Params >
int compute_grid_dimensions(dim3 &grid, Params &params, int tile_m, int tile_n) {
    params.tiles_m = xmma::div_up(params.m, tile_m);
    params.tiles_n = xmma::div_up(params.n, tile_n);
    if( params.use_horizontal_cta_rasterization ) {
        grid.y = params.tiles_m;
        grid.x = params.tiles_n;
        params.tiles_y = params.tiles_m;
        params.tiles_x = params.tiles_n;
    } else {
        grid.x = params.tiles_m;
        grid.y = params.tiles_n;
        params.tiles_x = params.tiles_m;
        params.tiles_y = params.tiles_n;
    }
    if( params.batch.is_batched ) {
        grid.z = params.batch.batches;
    } else {
        grid.z = params.split_k.slices;
    }
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static void meta_reorder (void *dst, const void *src, const int ld_dim, const int k,
    size_t e_sz, bool transpose, const int cta_m, const int cta_k) {
    // ld_dim -- size of M;
    // k -- equivalent size of K in 16bits element;
    // cta_m -- size of cta M; cta_k -- equivalent size of cta K in 16bits element

    const uint16_t *src_ptr = reinterpret_cast<const uint16_t *>(src);
    uint16_t *dst_ptr = reinterpret_cast<uint16_t *>(dst);
    void *tmp = malloc(e_sz);
    uint16_t *tmp_ptr = reinterpret_cast<uint16_t *>(tmp);

    const int reorder_bound = 32;
    if(transpose) {
        for(int group = 0 ; group < ld_dim ; group = group + reorder_bound){
            for( int i = 0; i < reorder_bound; ++i ) {
                for( int ki = 0; ki < k; ++ki ) {
                    int src_idx = (group + i) + ki * ld_dim;
                    int dst_row_idx = group + ((i % 8) * 4 + (i / 8));
                    int dst_idx = dst_row_idx + ki * ld_dim;
                    tmp_ptr[dst_idx] = src_ptr[src_idx];
                }
            }
        }
    }
    else{
        for(int group = 0 ; group < ld_dim ; group = group + reorder_bound){
            for( int i = 0; i < reorder_bound; ++i ) {
                for( int ki = 0; ki < k; ++ki ) {
                    int src_idx = (group + i) * k + ki;
                    int dst_row_idx = group + ((i % 8) * 4 + (i / 8));
                    int dst_idx = dst_row_idx * k + ki;
                    tmp_ptr[dst_idx] = src_ptr[src_idx];
                }
            }
        }
    }

    uint16_t *reorder = (uint16_t*)malloc(sizeof(uint16_t) * ld_dim * k);
    // Reorder metadata storing direction in vertical rasterization order
    if(transpose) {
        int elemenet_offset = 0;

        for(int ki = 0 ; ki < (k / 2) ; ki+=(cta_k / 2)){
            for(int mi = 0 ; mi < ld_dim ; mi+=cta_m){
                // Looping over metadata tile and compute offset to access metadata
                // Due to vertical rasterization, we loop M dimension first then K dimension

                int index = 0;
                int offset = elemenet_offset; // Storing offset for device memory
                for(int i = ki ; i < ki + (cta_k / 2) ; i++){
                    for(int j = mi ; (j < mi + cta_m) && (j < ld_dim) ; j++){
                        int idx0 = i * 2 * ld_dim + j;
                        int idx1 = i * 2 * ld_dim + ld_dim + j;
                        uint16_t a = tmp_ptr[idx0];
                        uint16_t b = tmp_ptr[idx1];

                        reorder[index + offset] = a;
                        index = index + 1;
                        reorder[index + offset] = b;
                        index = index + 1;

                        elemenet_offset = elemenet_offset + 2;
                    }
                }//Inside CTA

            }
        }
    }
    else{    // Change ld_dim
        uint32_t *pack_ptr = (uint32_t *)tmp_ptr;
        int elemenet_offset = 0;
        for(int ki = 0 ; ki < (k / 2) ; ki+=(cta_k / 2)){
            for(int mi = 0 ; mi < ld_dim ; mi+=cta_m){
                // Looping over metadata tile and compute offset to access metadata
                // Due to vertical rasterization, we loop M dimension first then K dimension

                int in_cta_residue = 0;
                int next_mi = (mi / cta_m + 1) * cta_m;
                if(ld_dim < next_mi){
                    in_cta_residue = ld_dim - mi;
                }
                else{
                    in_cta_residue = cta_m;
                }

                int offset = elemenet_offset; // Storing offset for device memory

                for(int j = mi ; (j < mi + cta_m) && (j < ld_dim) ; j++){
                    for(int i = ki ; i < ki + (cta_k / 2) ; i++){
                        //printf("j %d i %d\n", j, i);
                        int idx = j * (k / 2) + i;
                        uint32_t a = pack_ptr[idx];
                        uint32_t Mask = 0xFFFF;
                        uint16_t x0 = static_cast<uint16_t>(a & Mask);
                        uint16_t x1 = static_cast<uint16_t>((a>>16) & Mask);

                        int dst_cnt0 = (i % (cta_k / 2)) * in_cta_residue * 2 +
                                        ((j % cta_m) % in_cta_residue) * 2 + offset;

                        int dst_cnt1 = (i % (cta_k / 2)) * in_cta_residue * 2 +
                                        ((j % cta_m) % in_cta_residue) * 2 + 1 + offset;

                        reorder[dst_cnt0] = x0;
                        reorder[dst_cnt1] = x1;
                        elemenet_offset = elemenet_offset + 2;
                    }
                }//Inside CTA

            }
        }
    }

    if(transpose) {
        for(int i = 0 ; i < ld_dim * k / 4 ; i++){
            dst_ptr[i * 4] = reorder[i * 4];
            dst_ptr[i * 4 + 1] = reorder[i * 4 + 2];
            dst_ptr[i * 4 + 2] = reorder[i * 4 + 1];
            dst_ptr[i * 4 + 3] = reorder[i * 4 + 3];

        }
    }
    else{
        for(int i = 0 ; i < ld_dim * k / 4 ; i++){
            dst_ptr[i * 4] = reorder[i * 4];
            dst_ptr[i * 4 + 1] = reorder[i * 4 + 2];
            dst_ptr[i * 4 + 2] = reorder[i * 4 + 1];
            dst_ptr[i * 4 + 3] = reorder[i * 4 + 3];
        }

    }

    free(tmp);
    free(reorder);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace ext
} // namespace xmma
///////////////////////////////////////////////////////////////////////////////////////////////////
