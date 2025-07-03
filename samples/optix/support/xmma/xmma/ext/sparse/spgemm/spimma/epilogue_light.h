/**************************************************************************************************
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

//#include <xmma/helpers/epilogue.h>
#include <xmma/warp_masks.h>
#include <xmma/device_call.h>
namespace xmma {
namespace helpers {

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef LINK
extern "C"  __device__ ResultPack<float, 8> activation(float reg0, float reg1, float reg2, float reg3,float reg4, float reg5, float reg6, float reg7);
//extern  __device__ ResultPack<float, 16> activation(float reg0, float reg1, float reg2, float reg3,float reg4, float reg5, float reg6, float reg7,
 //               float reg8, float reg9, float reg10, float reg11, float reg12, float reg13,float reg14, float reg15);
#endif
template<
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The callbacks to lwstomize the behaviour of the epilogue.
    typename Callbacks_,
    // The class to swizzle the data.
    typename Swizzle_,
    // The fragment class before the swizzling.
    typename Fragment_pre_swizzle_ = typename Callbacks_::Fragment_pre_swizzle,
    // The fragment class after the swizzling.
    typename Fragment_post_swizzle_ = typename Callbacks_::Fragment_post_swizzle,
    // The output fragment.
    typename Fragment_c_ = typename Callbacks_::Fragment_c
>
struct Epilogue_light {

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    //typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The callbacks to lwstomize the behaviour of the epilogue.
    typename Callbacks_,
    // The class to swizzle the data.
    typename Swizzle_,
    // The fragment class before the swizzling.
    typename Fragment_pre_swizzle_,
    // The fragment class after the swizzling.
    typename Fragment_post_swizzle_,
    // The output fragment.
    typename Fragment_c_
>
struct Epilogue_light <Traits_,
                       Cta_tile_,
                       xmma::Col,
                       Gmem_tile_,
                       Callbacks_,
                       Swizzle_,
                       Fragment_pre_swizzle_,
                       Fragment_post_swizzle_,
                       Fragment_c_>{

    // The instruction traits.
    using Traits = Traits_;
    // The dimensions of the tile computed by the CTA.
    using Cta_tile = Cta_tile_;
    // The layout of the tile.
    using Layout = xmma::Col;
    // The global memory tile to store the output.
    using Gmem_tile = Gmem_tile_;
    // The callbacks.
    using Callbacks = Callbacks_;
    // The class to swizzle the data.
    using Swizzle = Swizzle_;
    // The fragment class before the swizzling.
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    // The fragment class after the swizzling.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    // The output fragment.
    using Fragment_c = Fragment_c_;
    // The fragment for bias.
    using Fragment_bias = typename Callbacks::Fragment_bias;
    // The fragment for alpha (used before swizzling).
    using Fragment_alpha_pre_swizzle = typename Callbacks::Fragment_alpha_pre_swizzle;
    // The fragment for alpha (used after swizzling).
    using Fragment_alpha_post_swizzle = typename Callbacks::Fragment_alpha_post_swizzle;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment for epilogue.
    using Fragment_epilogue = typename Callbacks::Fragment_epilogue;


    // Ctor.
    inline __device__ Epilogue_light() {
    }

    // Ctor.
    template< typename Params >
    inline __device__ Epilogue_light(const Params &params,
                                     Gmem_tile &gmem_tile,
                                     Callbacks &callbacks,
                                     Swizzle &swizzle)
                : gmem_tile_(gmem_tile)
                , callbacks_(callbacks)
                , swizzle_(swizzle)
                , mem_desc_c_(params.mem_descriptors.descriptor_c)
                , mem_desc_d_(params.mem_descriptors.descriptor_d) {
    }

    template< typename Fragment_aclwmulator, int STRIDED, int CONTIGUOUS >
    inline __device__ void execute(Fragment_aclwmulator (&acc)[CONTIGUOUS][STRIDED]) {

        ///////////////////////////////////////////////////////////////////////////////////////
        //   STGS = 2 cause each thread holds 2 row-major regs
        //   STRIDED = 4 cause each thread loop 4 times XMMAS_N
        //   0 0 1 1 2 2 3 3 0 0 1 1 2 2 3 3 | 64 64 65 65 66 66 67 67 64 64 65 65 66 66 67 67
        //   0 0 1 1 2 2 3 3 0 0 1 1 2 2 3 3 | 64 64 65 65 66 66 67 67 64 64 65 65 66 66 67 67
        //   0 0 1 1 2 2 3 3 0 0 1 1 2 2 3 3 | 64 64 65 65 66 66 67 67 64 64 65 65 66 66 67 67
        //   0 0 1 1 2 2 3 3 0 0 1 1 2 2 3 3 | 64 64 65 65 66 66 67 67 64 64 65 65 66 66 67 67
        //   0 0 1 1 2 2 3 3 0 0 1 1 2 2 3 3 | 64 64 65 65 66 66 67 67 64 64 65 65 66 66 67 67
        //   0 0 1 1 2 2 3 3 0 0 1 1 2 2 3 3 | 64 64 65 65 66 66 67 67 64 64 65 65 66 66 67 67
        //   0 0 1 1 2 2 3 3 0 0 1 1 2 2 3 3 | 64 64 65 65 66 66 67 67 64 64 65 65 66 66 67 67
        //   0 0 1 1 2 2 3 3 0 0 1 1 2 2 3 3 | 64 64 65 65 66 66 67 67 64 64 65 65 66 66 67 67
        //   4 4 5 5 6 6 7 7 4 4 5 5 6 6 7 7 | 68 68 69 69 70 70 71 71 68 68 69 69 70 70 71 71
        //   4 4 5 5 6 6 7 7 4 4 5 5 6 6 7 7 | 68 68 69 69 70 70 71 71 68 68 69 69 70 70 71 71
        //   4 4 5 5 6 6 7 7 4 4 5 5 6 6 7 7 | 68 68 69 69 70 70 71 71 68 68 69 69 70 70 71 71
        //   4 4 5 5 6 6 7 7 4 4 5 5 6 6 7 7 | 68 68 69 69 70 70 71 71 68 68 69 69 70 70 71 71
        //   4 4 5 5 6 6 7 7 4 4 5 5 6 6 7 7 | 68 68 69 69 70 70 71 71 68 68 69 69 70 70 71 71
        //   4 4 5 5 6 6 7 7 4 4 5 5 6 6 7 7 | 68 68 69 69 70 70 71 71 68 68 69 69 70 70 71 71
        //   4 4 5 5 6 6 7 7 4 4 5 5 6 6 7 7 | 68 68 69 69 70 70 71 71 68 68 69 69 70 70 71 71
        //   4 4 5 5 6 6 7 7 4 4 5 5 6 6 7 7 | 68 68 69 69 70 70 71 71 68 68 69 69 70 70 71 71
        ///////////////////////////////////////////////////////////////////////////////////////

        Fragment_bias bias_regs;
        // Load bias
        callbacks_.load_bias(bias_regs);

        #pragma unroll
        for (int si = 0; si < Gmem_tile::STGS * STRIDED ; si++) {
            Fragment_c out_regs[Gmem_tile::STGS];
            Fragment_epilogue epi_reg[Gmem_tile::STGS];

            //Colwert ACC to FP32
            #pragma unroll
            for (int i = 0; i < Gmem_tile::STGS; i++) {
                callbacks_.colwert<STRIDED,CONTIGUOUS>(acc, epi_reg[i], si, i);
            }

            // Do alpha
            #pragma unroll
            for (int i = 0; i < Gmem_tile::STGS; i++) {
                callbacks_.scale(epi_reg[i], callbacks_.alpha_);
            }

            // Do beta load and add
            if (callbacks_.beta_ != 0.0) {
                #pragma unroll
                for (int i = 0; i < Gmem_tile::STGS; i++) {
                    gmem_tile_.load(epi_reg[i], callbacks_.beta_, si, i, mem_desc_c_);
                }
            }

            // Add bias
            if (callbacks_.with_bias_) {
                #pragma unroll
                for (int i = 0; i < Gmem_tile::STGS; i++) {
                    callbacks_.add_bias(epi_reg[i], bias_regs);
                }
            }

            // Do Relu
            // Should we do fp32 relu? or int8 relu? Anton seems apply int8 relu.
            // Here I use fp32 relu.
            #pragma unroll
            for (int i = 0; i < Gmem_tile::STGS; i++) {
                callbacks_.pre_pack(si, i, epi_reg[i]);
            }

            // Colwert aclwmulator to ouput (float to int8) and pack
            #pragma unroll
            for (int i = 0; i < Gmem_tile::STGS; i++) {
                callbacks_.pack(out_regs[i], epi_reg[i]);
            }

            // Store results in STG.64
            #pragma unroll
            for (int i = 0; i < Gmem_tile::STGS; i++) {
                gmem_tile_.store(si, i, out_regs[i], mem_desc_d_);
            }
        }
    }

    // The output tile.
    Gmem_tile &gmem_tile_;
    // The shared memory tile.
    Swizzle &swizzle_;
    // The callbacks.
    Callbacks &callbacks_;
    // Ampere memory descriptors
    const uint64_t mem_desc_c_, mem_desc_d_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    //typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The callbacks to lwstomize the behaviour of the epilogue.
    typename Callbacks_,
    // The class to swizzle the data.
    typename Swizzle_,
    // The fragment class before the swizzling.
    typename Fragment_pre_swizzle_,
    // The fragment class after the swizzling.
    typename Fragment_post_swizzle_,
    // The output fragment.
    typename Fragment_c_
>
struct Epilogue_light <Traits_,
                       Cta_tile_,
                       xmma::Row,
                       Gmem_tile_,
                       Callbacks_,
                       Swizzle_,
                       Fragment_pre_swizzle_,
                       Fragment_post_swizzle_,
                       Fragment_c_>{
    // The instruction traits.
    using Traits = Traits_;
    // The dimensions of the tile computed by the CTA.
    using Cta_tile = Cta_tile_;
    // The layout of the tile.
    using Layout = xmma::Row;
    // The global memory tile to store the output.
    using Gmem_tile = Gmem_tile_;
    // The callbacks.
    using Callbacks = Callbacks_;
    // The class to swizzle the data.
    using Swizzle = Swizzle_;
    // The fragment class before the swizzling.
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    // The fragment class after the swizzling.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    // The output fragment.
    using Fragment_c = Fragment_c_;
    // The fragment for bias.
    using Fragment_bias = typename Callbacks::Fragment_bias;
    // The fragment for alpha (used before swizzling).
    using Fragment_alpha_pre_swizzle = typename Callbacks::Fragment_alpha_pre_swizzle;
    // The fragment for alpha (used after swizzling).
    using Fragment_alpha_post_swizzle = typename Callbacks::Fragment_alpha_post_swizzle;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment for epilogue.
    using Fragment_epilogue = typename Callbacks::Fragment_epilogue;

    // Ctor.
    inline __device__ Epilogue_light() {
    }

    // Ctor.
    template< typename Params >
    inline __device__ Epilogue_light(const Params &params,
                                     Gmem_tile &gmem_tile,
                                     Callbacks &callbacks,
                                     Swizzle &swizzle)
                : gmem_tile_(gmem_tile)
                , callbacks_(callbacks)
                , swizzle_(swizzle)
                , mem_desc_c_(params.mem_descriptors.descriptor_c)
                , mem_desc_d_(params.mem_descriptors.descriptor_d) {
    }

    template< typename Fragment_aclwmulator, int STRIDED, int CONTIGUOUS >
    inline __device__ void execute(Fragment_aclwmulator (&acc)[STRIDED][CONTIGUOUS]) {

        Fragment_bias bias_regs;

        #pragma unroll
        for (int si = 0; si < STRIDED ; si++) {
            Fragment_c out_regs[Gmem_tile::STGS];
            Fragment_epilogue epi_reg[Gmem_tile::STGS];

            //Colwert ACC to FP32
            #pragma unroll
            for (int i = 0; i < Gmem_tile::STGS; i++) {
                callbacks_.colwert<STRIDED, CONTIGUOUS>(acc, epi_reg[i], si, i);
            }

            // Do alpha
            #pragma unroll
            for (int i = 0; i < Gmem_tile::STGS; i++) {
                callbacks_.scale(epi_reg[i], callbacks_.alpha_);
            }

            // Do beta load and add
            if (callbacks_.beta_ != 0.0) {
                #pragma unroll
                for (int i = 0; i < Gmem_tile::STGS; i++) {
                    gmem_tile_.load(epi_reg[i], callbacks_.beta_, si, i, mem_desc_c_);
                }
            }

            // Add bias
            if (callbacks_.with_bias_) {
                #pragma unroll
                for (int i = 0; i < Gmem_tile::STGS; i++) {
                    callbacks_.load_bias(si, i, bias_regs);
                    callbacks_.add_bias(epi_reg[i], bias_regs);
                }
            }

            // Do Relu
            // Should we do fp32 relu? or int8 relu? Anton seems apply int8 relu.
            // Here I use fp32 relu.
            #pragma unroll
            for (int i = 0; i < Gmem_tile::STGS; i++) {
                callbacks_.pre_pack(si, i, epi_reg[i]);
            }

            // Colwert aclwmulator to ouput (float to int8) and pack
            #pragma unroll
            for (int i = 0; i < Gmem_tile::STGS; i++) {
                callbacks_.pack(out_regs[i], epi_reg[i]);
            }

            // Store results in STG.64
            #pragma unroll
            for (int i = 0; i < Gmem_tile::STGS; i++) {
                gmem_tile_.store(si, i, out_regs[i], mem_desc_d_);
            }
        }
    }

    // The output tile.
    Gmem_tile &gmem_tile_;
    // The shared memory tile.
    Swizzle &swizzle_;
    // The callbacks.
    Callbacks &callbacks_;
    // Ampere memory descriptors
    const uint64_t mem_desc_c_, mem_desc_d_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout.
    typename Layout,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>
>
struct Gmem_tile_epilogue_light {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout.
    // typename Layout = xmma::Col,
    // The fragment class before writing data to global memory.
    typename Fragment_c
>
struct Gmem_tile_epilogue_light <Traits,
                                 Cta_tile,
                                 xmma::Col,
                                 Fragment_c>{


    // Because we do transposition to get col-major layout
    enum { STGS = 2 };
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;
    using Fragment_epilogue = xmma::Fragment<float, 8>;


    // Ctor.
    inline __device__ Gmem_tile_epilogue_light(int m, int n, int stride_n)
        : params_m_(m)
        , params_n_(n)
        , params_stride_n_(stride_n) {
    }

    // Ctor.
    inline __device__ Gmem_tile_epilogue_light(
        int m, int n, int stride_n, char *out_ptr, const char* res_ptr,
        int bidm, int bidn, int bidz, int tidx)
        : params_m_(m)
        , params_n_(n)
        , params_stride_n_(stride_n) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        // The location of the tile.
        int row = ((tidx & WARP_MASK_N) / WARP_DIV_N) * Xmma_tile::N_PER_XMMA
            + (tidx % 4) * 2;

        int col = ((tidx & WARP_MASK_M) / WARP_DIV_M) * Xmma_tile::M_PER_WARP
            + ((tidx % Cta_tile::THREADS_PER_WARP)/4) * 8; //ELEMENTS_PER_STG = 8

        m_ = bidn * Cta_tile::N + row;
        n_ = bidm * Cta_tile::M + col;

        // The pointer.
        const int64_t offset = Traits::offset_in_bytes_c(m_*params_stride_n_ + n_);
        out_ptr_ = &out_ptr[offset];
        res_ptr_ = &res_ptr[offset];
    }

    // Compute the row offset.
    static inline __device__ constexpr int compute_offset(int si, int i) {
        return (si % 2) * Xmma_tile::N_PER_XMMA / 2 +
               (si / 2) * Xmma_tile::N_PER_XMMA_PER_CTA + i;
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask(const int offset) const {
        return (offset + m_) < params_m_ && n_ < params_n_;
    }

    // Store the data to global memory.
    inline __device__ void store(int si,
                                 int ii,
                                 Fragment_c &data,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {

        int offset = compute_offset(si, ii);
        //char *ptr = &out_ptr_[Traits::offset_in_bytes_c(offset*params_stride_n_)];
        char *ptr = out_ptr_ + Traits::offset_in_bytes_c(offset*params_stride_n_);

        int mask = compute_output_mask(offset);

        if (mask) {
            xmma::stg(ptr, data.to_int2(), mem_desc);
        }
    }

    // Load the data from global memory.
    inline __device__ void load(Fragment_epilogue &data,
                                float beta,
                                int si,
                                int i,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {

        const int offset = compute_offset(si, i);
        //const char *ptr = &res_ptr_[Traits::offset_in_bytes_c(offset*params_stride_n_)];
        const char *ptr = res_ptr_ + Traits::offset_in_bytes_c(offset*params_stride_n_);

        int mask = compute_output_mask(offset);

        if (mask) {
            uint2 tmp;
            xmma::ldg(tmp, ptr, mem_desc);
            
            float4 tmp_0 = s8x4_to_float4(tmp.x);
            data.elt(0) += tmp_0.x * beta;
            data.elt(1) += tmp_0.y * beta;
            data.elt(2) += tmp_0.z * beta;
            data.elt(3) += tmp_0.w * beta;

            float4 tmp_1 = s8x4_to_float4(tmp.y);
            data.elt(4) += tmp_1.x * beta;
            data.elt(5) += tmp_1.y * beta;
            data.elt(6) += tmp_1.z * beta;
            data.elt(7) += tmp_1.w * beta;
        }
    }

    // The dimensions of the matrix.
    const int params_m_, params_n_, params_stride_n_;
    // The position of the tile.
    int m_, n_;
    // The pointer to global memory.
    char *out_ptr_;
    const char *res_ptr_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout.
    // typename Layout = xmma::Row,
    // The fragment class before writing data to global memory.
    typename Fragment_c
>
struct Gmem_tile_epilogue_light <Traits,
                                 Cta_tile,
                                 xmma::Row,
                                 Fragment_c>{

    // Because we do transposition to get col-major layout
    enum { STGS = 2 };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;

    // STG Sizing Scale
    enum { STG_SIZING = (Cta_tile::N == 128) ? 1 : 2 };
    enum { REG_COUNT = (Cta_tile::N == 128) ? 16 : 8 };

    using Fragment_epilogue = xmma::Fragment<float, REG_COUNT>;

    // Resize STG
    enum { BYTES_PER_STG = 16 / STG_SIZING };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    // The number of elements per row.
    enum { ELEMENTS_PER_ROW = Cta_tile::N };
    // The number of threads needed to store a row.
    enum { THREADS_PER_ROW = ELEMENTS_PER_ROW / ELEMENTS_PER_STG };

    enum { ACC_STRIDE = 8 };

    // Ctor.
    inline __device__ Gmem_tile_epilogue_light(int m, int n, int stride_n)
        : params_m_(m)
        , params_n_(n)
        , params_stride_n_(stride_n) {
    }

    // Ctor.
    inline __device__ Gmem_tile_epilogue_light(
        int m, int n, int stride_n, char *out_ptr, const char* res_ptr,
        int bidm, int bidn, int bidz, int tidx)
        : params_m_(m)
        , params_n_(n)
        , params_stride_n_(stride_n) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        int warp_m = (tidx & WARP_MASK_M) / WARP_DIV_M;
        int warp_n = (tidx & WARP_MASK_N) / WARP_DIV_N;

        int row = (tidx % Cta_tile::THREADS_PER_WARP) / (THREADS_PER_ROW / 2) + warp_m * 64;
        int col = tidx % (THREADS_PER_ROW / 2) ;

        m_ = bidm * Cta_tile::M + row;
        n_ = bidn * Cta_tile::N + col * ELEMENTS_PER_STG + warp_n * (Cta_tile::N / 2);

        // The pointer.
        const int64_t offset = Traits::offset_in_bytes_c(m_*params_stride_n_ + n_);
        out_ptr_ = &out_ptr[offset];
        res_ptr_ = &res_ptr[offset];

    }

    // Compute the row offset.
    static inline __device__ constexpr int compute_offset(int si, int i) {
        return si * Xmma_tile::M_PER_XMMA + i * ACC_STRIDE;
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask(const int offset) const {
        return (offset + m_) < params_m_ && n_ < params_n_;
    }

    // Store the data to global memory.
    inline __device__ void store(int si,
                                 int ii,
                                 Fragment_c &data,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {

        int offset = compute_offset(si, ii);
        char *ptr = out_ptr_ + Traits::offset_in_bytes_c(offset*params_stride_n_);

        int mask = compute_output_mask(offset);

        if (mask) {
            // xmma::stg(ptr, data.to_int2(), mem_desc);
            if (BYTES_PER_STG == 16) {
                xmma::stg(ptr, data.to_int4(), mem_desc);
            } else if (BYTES_PER_STG == 8) {
                xmma::stg(ptr, make_uint2(data.reg(0),data.reg(1)), mem_desc);
            } else {
                xmma::stg(ptr, data.reg(0), mem_desc);
            }
        }
    }

    // Load the data from global memory.
    inline __device__ void load(Fragment_epilogue &data,
                                float beta,
                                int si,
                                int i,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {

        const int offset = compute_offset(si, i);
        const char *ptr = res_ptr_ + Traits::offset_in_bytes_c(offset*params_stride_n_);

        int mask = compute_output_mask(offset);

        if (mask) {

            if (BYTES_PER_STG == 16) {
                uint4 tmp;
                xmma::ldg(tmp, ptr, mem_desc);

                float4 tmp_0 = s8x4_to_float4(tmp.x);
                data.elt(0) += tmp_0.x * beta;
                data.elt(1) += tmp_0.y * beta;
                data.elt(2) += tmp_0.z * beta;
                data.elt(3) += tmp_0.w * beta;

                float4 tmp_1 = s8x4_to_float4(tmp.y);
                data.elt(4) += tmp_1.x * beta;
                data.elt(5) += tmp_1.y * beta;
                data.elt(6) += tmp_1.z * beta;
                data.elt(7) += tmp_1.w * beta;

                float4 tmp_2 = s8x4_to_float4(tmp.z);
                data.elt(8) += tmp_2.x * beta;
                data.elt(9) += tmp_2.y * beta;
                data.elt(10) += tmp_2.z * beta;
                data.elt(11) += tmp_2.w * beta;

                float4 tmp_3 = s8x4_to_float4(tmp.w);
                data.elt(12) += tmp_3.x * beta;
                data.elt(13) += tmp_3.y * beta;
                data.elt(14) += tmp_3.z * beta;
                data.elt(15) += tmp_3.w * beta;

            } else if (BYTES_PER_STG == 8) {
                uint2 tmp;
                xmma::ldg(tmp, ptr, mem_desc);

                float4 tmp_0 = s8x4_to_float4(tmp.x);
                data.elt(0) += tmp_0.x * beta;
                data.elt(1) += tmp_0.y * beta;
                data.elt(2) += tmp_0.z * beta;
                data.elt(3) += tmp_0.w * beta;

                float4 tmp_1 = s8x4_to_float4(tmp.y);
                data.elt(4) += tmp_1.x * beta;
                data.elt(5) += tmp_1.y * beta;
                data.elt(6) += tmp_1.z * beta;
                data.elt(7) += tmp_1.w * beta;
            } else {
                uint32_t tmp;
                xmma::ldg(tmp, ptr, mem_desc);

                float4 tmp_0 = s8x4_to_float4(tmp);
                data.elt(0) += tmp_0.x * beta;
                data.elt(1) += tmp_0.y * beta;
                data.elt(2) += tmp_0.z * beta;
                data.elt(3) += tmp_0.w * beta;
            }
        }
    }

    // The dimensions of the matrix.
    const int params_m_, params_n_, params_stride_n_;
    // The position of the tile.
    int m_, n_;
    // The pointer to global memory.
    char *out_ptr_;
    const char *res_ptr_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout.
    typename Layout,
    // Elementwise operation
    int ELTWISE,
    typename Fragment_pre_swizzle_ = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
    typename Fragment_post_swizzle_ = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
    typename Fragment_c_ = xmma::Fragment_c<Traits, Cta_tile>
>
struct Callbacks_epilogue_light {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    // The layout.
    // typename Layout,
    int ELTWISE,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_
>
struct Callbacks_epilogue_light <Traits,
                                 Cta_tile,
                                 xmma::Col,
                                 ELTWISE,
                                 Fragment_pre_swizzle_,
                                 Fragment_post_swizzle_,
                                 Fragment_c_>
    : public xmma::helpers::Empty_callbacks_epilogue<Traits,
                                                         Cta_tile,
                                                         Fragment_pre_swizzle_,
                                                         Fragment_post_swizzle_,
                                                         Fragment_c_> {

    // The base class.
    using Base = xmma::helpers::Empty_callbacks_epilogue<Traits,
                                                             Cta_tile,
                                                             Fragment_pre_swizzle_,
                                                             Fragment_post_swizzle_,
                                                             Fragment_c_>;
    // To make contiguous elements for STG.64
    enum { ELEMENTS_PER_LDG = 8 };

    // unified field for RT_ACT acc numbers
    using Fragment_epilogue = xmma::Fragment<float, ELEMENTS_PER_LDG>;
    enum { REGS_PER_CALL = Fragment_epilogue::NUM_REGS };
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    using Fragment_c = Fragment_c_;
    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;
    using Fragment_bias = xmma::Fragment<float, ELEMENTS_PER_LDG>;
    using C_type = typename Traits::C_type;


    template< typename Params >
    inline __device__ Callbacks_epilogue_light(const Params &params,
                                               char *smem,
                                               int bidm,
                                               int bidn,
                                               int bidz,
                                               int tidx)
                                        : Base(params, smem, bidm, bidn, bidz, tidx)
                                        , relu_lb_(params.relu_lb)
                                        , relu_ub_(params.relu_ub)
                                        , beta_(params.beta)
                                        , alpha_(params.alpha)
                                        , params_m_(params.m)
                                        , with_bias_(params.with_bias)
                                        , bias_ptr_(reinterpret_cast<const char*>(params.bias_gmem))  {
        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;

        // The location of the tile.
        const int row =
            ((tidx & WARP_MASK_M) / WARP_DIV_M) * Xmma_tile::M_PER_WARP
           + ((tidx % Cta_tile::THREADS_PER_WARP)/4) * ELEMENTS_PER_LDG;

        // Compute the output position for each thread.
        bias_m_ = bidm * Cta_tile::M + row;
        //printf("tidx %d row %d\n", threadIdx.x, row);
        // The pointer.
        bias_ptr_ += bias_m_ * sizeof(float);

        memcpy(&gelu_scale_, &params.runtime_params.runtime_param0, sizeof(float));
    }

    template <int STRIDED, int CONTIGUOUS>
    inline __device__ void colwert(
        Fragment_aclwmulator (&acc)[CONTIGUOUS][STRIDED],
        Fragment_epilogue &out,
        const int si, const int i) {

        out.elt(0) = acc[0][si/2].elt(0 + i + (si%2) * 4);
        out.elt(1) = acc[0][si/2].elt(2 + i + (si%2) * 4);
        out.elt(2) = acc[1][si/2].elt(0 + i + (si%2) * 4);
        out.elt(3) = acc[1][si/2].elt(2 + i + (si%2) * 4);
        out.elt(4) = acc[2][si/2].elt(0 + i + (si%2) * 4);
        out.elt(5) = acc[2][si/2].elt(2 + i + (si%2) * 4);
        out.elt(6) = acc[3][si/2].elt(0 + i + (si%2) * 4);
        out.elt(7) = acc[3][si/2].elt(2 + i + (si%2) * 4);

    }

    inline __device__ void scale(
        Fragment_epilogue &data, float alpha) {
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; i++) {
            data.elt(i) *= alpha;
        }
    }

    inline __device__ void pack(
        Fragment_c &out,
        Fragment_epilogue &frag) {

        if (ELTWISE == xmma::RELU) {
            #pragma unroll
            for( int ii = 0; ii < ELEMENTS_PER_LDG/4; ++ii ) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    frag.elt(4 * ii + j) = fmax(frag.elt(4 * ii + j), -128.0f);
                }
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    frag.elt(4 * ii + j) += 12582912.0f;
                }
                int32_t result[4];
                memcpy(result, &frag.reg(4 * ii), 16);
                asm volatile("prmt.b32 %0, %0, %1, 0x40;\n"
                    : "+r"(result[0]) : "r"(result[1]));
                asm volatile("prmt.b32 %0, %0, %1, 0x40;\n"
                    : "+r"(result[2]) : "r"(result[3]));
                asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n"
                    : "=r"(out.reg(ii)) : "r"(result[0]), "r"(result[2]));
            }
        }
        if (ELTWISE == xmma::GELU) {
            int32_t tmp[4];
            #pragma unroll
            for( int ii = 0; ii < ELEMENTS_PER_LDG/4; ++ii ) {
                asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(tmp[0]) : "f"(frag.elt(4*ii    )));
                asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(tmp[1]) : "f"(frag.elt(4*ii + 1)));
                asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(tmp[2]) : "f"(frag.elt(4*ii + 2)));
                asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(tmp[3]) : "f"(frag.elt(4*ii + 3)));
                out.reg(ii) = xmma::pack_int8x4(tmp);
            }
        }
    }

    // We do ReLU here.
    inline __device__ void pre_pack(int si, int i,
                                    Fragment_epilogue &frag) {
        if (ELTWISE == xmma::RELU) {
            #pragma unroll
            for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
                frag.elt(i) = fmin(relu_ub_, reluTh_fp32( frag.elt(i), relu_lb_));
            }
        }
        if (ELTWISE == xmma::RT_ACT){
#ifdef LINK
            //ResultPack res;
            if (REGS_PER_CALL == 8){
              ResultPack<float, 8> res;
              asm volatile (".pragma \"call_abi_param_reg 8\";");
              res = activation(frag.elt(0),frag.elt(1),frag.elt(2),frag.elt(3),
                              frag.elt(4),frag.elt(5),frag.elt(6),frag.elt(7));
              res.setFrag(frag);
            }
#endif
        }
        if (ELTWISE == xmma::GELU) {
            // origin: 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
            // new: 0.5f * (x + x * tanh(x * (0.797885f + 0.0356774f * x * x)));
            // reduce two op
            constexpr auto literal0 = 0.044715f * 0.797885f;
            constexpr auto literal1 = 0.797885f;
            constexpr auto literal2 = 0.500000f;
            #pragma unroll
            for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
                frag.elt(i) = xmma::gelu(frag.elt(i), literal0, literal1, literal2, gelu_scale_);
            }
        }

    }

    inline __device__ void load_bias(Fragment_bias &data) {

        if (with_bias_ && (bias_m_ < params_m_) ) {
            uint4 tmp;

            xmma::ldg(tmp, bias_ptr_);
            data.reg(0) = tmp.x;
            data.reg(1) = tmp.y;
            data.reg(2) = tmp.z;
            data.reg(3) = tmp.w;

            xmma::ldg(tmp, bias_ptr_ + sizeof(float) * 4);
            data.reg(4) = tmp.x;
            data.reg(5) = tmp.y;
            data.reg(6) = tmp.z;
            data.reg(7) = tmp.w;
        }
    }

    inline __device__ void add_bias(
        Fragment_epilogue &out,
        Fragment_bias bias) {
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; i++) {
            out.elt(i) += bias.elt(i);
        }
    }

    float relu_lb_, relu_ub_, alpha_, beta_;
    bool with_bias_;
    int bias_m_;
    const int params_m_;
    const char *bias_ptr_;
    float gelu_scale_;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    // The layout.
    // typename Layout,
    int ELTWISE,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_
>
struct Callbacks_epilogue_light <Traits,
                                 Cta_tile,
                                 xmma::Row,
                                 ELTWISE,
                                 Fragment_pre_swizzle_,
                                 Fragment_post_swizzle_,
                                 Fragment_c_>
    : public xmma::helpers::Empty_callbacks_epilogue<Traits,
                                                         Cta_tile,
                                                         Fragment_pre_swizzle_,
                                                         Fragment_post_swizzle_,
                                                         Fragment_c_> {
    // The base class.
    using Base = xmma::helpers::Empty_callbacks_epilogue<Traits,
                                                             Cta_tile,
                                                             Fragment_pre_swizzle_,
                                                             Fragment_post_swizzle_,
                                                             Fragment_c_>;

    enum { ELEMENTS_PER_LOAD = (Cta_tile::N == 128) ? 16 : 8 };
    enum { PER_CHANNEL_ELEMENT_BITS = 32 };
    enum { THREADS_PER_ROW_PER_WARP = 4 };
    enum { LDGS = (ELEMENTS_PER_LOAD * PER_CHANNEL_ELEMENT_BITS) / 128 };
    enum { ELEMENTS_PER_LDG = 128 / PER_CHANNEL_ELEMENT_BITS };
    enum { OUT_ELEMENTS = ELEMENTS_PER_LOAD / 4 };

    using Fragment_epilogue = xmma::Fragment<float, ELEMENTS_PER_LOAD>;
    //using Fragment_c = xmma::Fragment<int32_t, OUT_ELEMENTS>;
    using Fragment_c = Fragment_c_;
    // using Fragment_epilogue = xmma::Fragment<float, ELEMENTS_PER_LDG>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;
    // using Fragment_bias = xmma::Fragment<float, ELEMENTS_PER_LDG>;
    using Fragment_bias = xmma::Fragment<float, 1>;
    using C_type = typename Traits::C_type;

    // STG Sizing Scale
    enum { STG_SIZING = (Cta_tile::N == 128) ? 1 : 2 };
    // Resize STG
    enum { BYTES_PER_STG = 16 / STG_SIZING };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    // The number of elements per row.
    enum { ELEMENTS_PER_ROW = Cta_tile::N };
    // The number of threads needed to store a row.
    enum { THREADS_PER_ROW = ELEMENTS_PER_ROW / ELEMENTS_PER_STG };
    enum { ACC_STRIDE = 8 };


    template< typename Params >
    inline __device__ Callbacks_epilogue_light(const Params &params,
                                               char *smem,
                                               int bidm,
                                               int bidn,
                                               int bidz,
                                               int tidx)
                                        : Base(params, smem, bidm, bidn, bidz, tidx)
                                        , relu_lb_(params.relu_lb)
                                        , relu_ub_(params.relu_ub)
                                        , beta_(params.beta)
                                        , alpha_(params.alpha)
                                        , params_m_(params.m)
                                        , with_bias_(params.with_bias)
                                        , bias_ptr_(reinterpret_cast<const char*>(params.bias_gmem))  {
        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        int warp_m = (tidx & WARP_MASK_M) / WARP_DIV_M;

        int row = (tidx % Cta_tile::THREADS_PER_WARP) / (THREADS_PER_ROW / 2) + warp_m * 64;

        m_ = bidm * Cta_tile::M + row;

        memcpy(&gelu_scale_, &params.runtime_params.runtime_param0, sizeof(float));
    }

    // Compute the row offset.
    static inline __device__ constexpr int compute_offset(int si, int i) {
        return si * Xmma_tile::M_PER_XMMA + i * ACC_STRIDE;
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask(const int offset) const {
        return (offset + m_) < params_m_;
    }

    template <int STRIDED, int CONTIGUOUS>
    inline __device__ void colwert(
        Fragment_aclwmulator (&acc)[STRIDED][CONTIGUOUS],
        Fragment_epilogue &out,
        const int si, const int i) {

        #pragma unroll
        for( int ii = 0; ii < CONTIGUOUS ; ++ii ) {
        //for( int ii = 0; ii < Fragment_epilogue:::NUM_REGS ; ++ii ) {
            out.elt(0 + ii * 4) = acc[si][ii].elt(0 + i * 2);
            out.elt(1 + ii * 4) = acc[si][ii].elt(1 + i * 2);
            out.elt(2 + ii * 4) = acc[si][ii].elt(4 + i * 2);
            out.elt(3 + ii * 4) = acc[si][ii].elt(5 + i * 2);
        }
    }

    inline __device__ void scale(
        Fragment_epilogue &data, float alpha) {
        #pragma unroll
        for (int i = 0; i < Fragment_epilogue::NUM_REGS ; i++) {
            data.elt(i) *= alpha;
        }
    }

    inline __device__ void pack(
        Fragment_c &out,
        Fragment_epilogue &frag) {
        if (ELTWISE == xmma::RELU) {
            #pragma unroll
            for( int ii = 0; ii < OUT_ELEMENTS ; ++ii ) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    frag.elt(4 * ii + j) = fmax(frag.elt(4 * ii + j), -128.0f);
                }
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    frag.elt(4 * ii + j) += 12582912.0f;
                }
                int32_t result[4];
                memcpy(result, &frag.reg(4 * ii), 16);
                asm volatile("prmt.b32 %0, %0, %1, 0x40;\n"
                    : "+r"(result[0]) : "r"(result[1]));
                asm volatile("prmt.b32 %0, %0, %1, 0x40;\n"
                    : "+r"(result[2]) : "r"(result[3]));
                asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n"
                    : "=r"(out.reg(ii)) : "r"(result[0]), "r"(result[2]));
            }
        }
        if (ELTWISE == xmma::GELU) {
            int32_t tmp[4];
            #pragma unroll
            for( int ii = 0; ii < OUT_ELEMENTS ; ++ii ) {
                asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(tmp[0]) : "f"(frag.elt(4*ii  )));
                asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(tmp[1]) : "f"(frag.elt(4*ii+1)));
                asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(tmp[2]) : "f"(frag.elt(4*ii+2)));
                asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(tmp[3]) : "f"(frag.elt(4*ii+3)));
                out.reg(ii) = xmma::pack_int8x4(tmp);
            }
        }
    }

    // We do ReLU here.
    inline __device__ void pre_pack(int si, int i,
                                    Fragment_epilogue &frag) {

        if (ELTWISE == xmma::RELU) {
            #pragma unroll
            for( int i = 0; i < Fragment_epilogue::NUM_ELTS ; ++i ) {
                frag.elt(i) = fmin(relu_ub_, reluTh_fp32( frag.elt(i), relu_lb_));
            }
        }
        if (ELTWISE == xmma::RT_ACT){
#ifdef LINK
            //ResultPack res;
            if (REGS_PER_CALL == 8){
              ResultPack<float, 8> res;
              asm volatile (".pragma \"call_abi_param_reg 8\";");
              res = activation(frag.elt(0),frag.elt(1),frag.elt(2),frag.elt(3),
                              frag.elt(4),frag.elt(5),frag.elt(6),frag.elt(7));
              res.setFrag(frag);
            }
#endif
        }

        if (ELTWISE == xmma::GELU) {
            // origin: 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
            // new: 0.5f * (x + x * tanh(x * (0.797885f + 0.0356774f * x * x)));
            // reduce two op
            constexpr auto literal0 = 0.044715f * 0.797885f;
            constexpr auto literal1 = 0.797885f;
            constexpr auto literal2 = 0.500000f;
            #pragma unroll
            for( int i = 0; i < Fragment_epilogue::NUM_ELTS; ++i ) {
                frag.elt(i) = xmma::gelu(frag.elt(i), literal0, literal1, literal2, gelu_scale_);
            }
        }
    }

    inline __device__ void add_bias(
        Fragment_epilogue &out,
        Fragment_bias bias) {

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_STG; i++) {
            out.elt(i) += bias.elt(0);
        }

    }

    inline __device__ void load_bias(int si,
                                     int ii, Fragment_bias &data) {

        int offset = compute_offset(si, ii);
        int tmp_row = m_ + offset;
        const char *bias_ptr_tmp = bias_ptr_;
        bias_ptr_tmp += tmp_row * sizeof(float);

        if(tmp_row < params_m_){

            uint32_t tmp;
            xmma::ldg(tmp, bias_ptr_tmp);
            data.reg(0) = tmp;
        }

    }

    float relu_lb_, relu_ub_, alpha_, beta_;
    bool with_bias_;
    int bias_m_;
    const int params_m_;
    const char *bias_ptr_;
    float gelu_scale_;

    // The position of the tile.
    int m_;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
