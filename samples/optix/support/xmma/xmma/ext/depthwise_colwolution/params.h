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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_PARAMS_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_PARAMS_H

#pragma once

#include "utils.h"
#include "xmma/utils.h"
#include "xmma/xmma.h"

namespace xmma
{
namespace ext
{
namespace depthwise_colwolution
{

struct Depthwise_colwolution_parameter_base {
    void *gmem[Tensor_type::COUNT];
    void *tmp_gmem_trsg;
    void *final_gmem_trsg;
    int32_t g, n, d, h, w, t, r, s, o, p, q;
    int32_t pad[3][2];
    int32_t stride[3];
    int32_t dilation[3];

    uint32_t img_stride_n, img_stride_d, img_stride_h, img_stride_w;
    uint32_t out_stride_n, out_stride_o, out_stride_p, out_stride_q;
    uint32_t flt_stride_t, flt_stride_r, flt_stride_s;

    uint32_t alpha, beta;
    bool is_colwolution;
    bool is_dgrad;

    XMMA_HOST Depthwise_colwolution_parameter_base() {}

    XMMA_HOST void set_a_gmem(void *in) { gmem[Tensor_type::A] = in; }

    XMMA_HOST void set_b_gmem(void *in) { gmem[Tensor_type::B] = in; }

    XMMA_HOST void set_c_gmem(void *in) { gmem[Tensor_type::C] = in; }

    XMMA_HOST void set_d_gmem(void *in) { gmem[Tensor_type::D] = in; }

    XMMA_HOST void set_tmp_gmem_trsg(void *in) { tmp_gmem_trsg = in; }

    XMMA_HOST void set_final_gmem_trsg(void *in) { final_gmem_trsg = in; }

    XMMA_HOST void set_g(const int32_t in)
    {
        g = in;
        return;
    }

    XMMA_HOST void set_n(const int32_t in)
    {
        n = in;
        return;
    }

    XMMA_HOST void set_d(const int32_t in)
    {
        d = in;
        return;
    }

    XMMA_HOST void set_h(const int32_t in)
    {
        h = in;
        return;
    }

    XMMA_HOST void set_w(const int32_t in)
    {
        w = in;
        return;
    }

    XMMA_HOST void set_t(const int32_t in)
    {
        t = in;
        return;
    }

    XMMA_HOST void set_r(const int32_t in)
    {
        r = in;
        return;
    }

    XMMA_HOST void set_s(const int32_t in)
    {
        s = in;
        return;
    }

    XMMA_HOST void set_o(const int32_t in)
    {
        o = in;
        return;
    }

    XMMA_HOST void set_p(const int32_t in)
    {
        p = in;
        return;
    }

    XMMA_HOST void set_q(const int32_t in)
    {
        q = in;
        return;
    }

    XMMA_HOST void set_pad_front(const int32_t in)
    {
        pad[0][0] = in;
        return;
    }

    XMMA_HOST void set_pad_back(const int32_t in)
    {
        pad[0][1] = in;
        return;
    }

    XMMA_HOST void set_pad_top(const int32_t in)
    {
        pad[1][0] = in;
        return;
    }

    XMMA_HOST void set_pad_bottom(const int32_t in)
    {
        pad[1][1] = in;
        return;
    }

    XMMA_HOST void set_pad_left(const int32_t in)
    {
        pad[2][0] = in;
        return;
    }

    XMMA_HOST void set_pad_right(const int32_t in)
    {
        pad[2][1] = in;
        return;
    }

    XMMA_HOST void set_stride_depth(const int32_t in)
    {
        stride[0] = in;
        return;
    }

    XMMA_HOST void set_stride_height(const int32_t in)
    {
        stride[1] = in;
        return;
    }

    XMMA_HOST void set_stride_width(const int32_t in)
    {
        stride[2] = in;
        return;
    }

    XMMA_HOST void set_dilation_depth(const int32_t in)
    {
        dilation[0] = in;
        return;
    }

    XMMA_HOST void set_dilation_height(const int32_t in)
    {
        dilation[1] = in;
        return;
    }

    XMMA_HOST void set_dilation_width(const int32_t in)
    {
        dilation[2] = in;
        return;
    }

    XMMA_HOST void set_alpha(const int32_t in)
    {
        alpha = int32_as_uint32(in);
        return;
    }

    XMMA_HOST void set_beta(const int32_t in)
    {
        beta = int32_as_uint32(in);
        return;
    }

    XMMA_HOST void set_is_colwolution(const bool in) { is_colwolution = in; }

    XMMA_HOST void set_is_dgrad(const bool in) { is_dgrad = in; }

    XMMA_HOST xmma::Error finalize()
    {
#define SWAP(a, b)                                                                                 \
    {                                                                                              \
        int32_t tmp = a;                                                                           \
        a = b;                                                                                     \
        b = tmp;                                                                                   \
    }
        if (is_dgrad) {
            SWAP(d, o);
            SWAP(h, p);
            SWAP(w, q);
#undef SWAP
            pad[0][0] = (t - 1) * dilation[0] - pad[0][0];
            pad[1][0] = (r - 1) * dilation[1] - pad[1][0];
            pad[2][0] = (s - 1) * dilation[2] - pad[2][0];
            is_colwolution = !is_colwolution;
        }

        img_stride_w = g;
        img_stride_h = w * img_stride_w;
        img_stride_d = h * img_stride_h;
        img_stride_n = d * img_stride_d;

        out_stride_q = g;
        out_stride_p = q * out_stride_q;
        out_stride_o = p * out_stride_p;
        out_stride_n = o * out_stride_o;

        flt_stride_s = g;
        flt_stride_r = s * flt_stride_s;
        flt_stride_t = r * flt_stride_r;
        return xmma::Error::SUCCESS;
    }
};

struct Depthwise_colwolution_parameter_middle : public Depthwise_colwolution_parameter_base {
    public:
    int32_t tiles_opq, tiles_pq, tiles_q, tiles_o, tiles_p;
    uint32_t mul_tiles_opq, mul_tiles_pq, mul_tiles_q;
    uint32_t shr_tiles_opq, shr_tiles_pq, shr_tiles_q;
    int32_t delta_tiles_n, delta_tiles_o, delta_tiles_p, delta_tiles_q;
    int32_t tiles_trs, tiles_rs, tiles_s;
    uint32_t mul_tiles_trs, mul_tiles_rs, mul_tiles_s;
    uint32_t shr_tiles_trs, shr_tiles_rs, shr_tiles_s;
    int32_t count_main_loop;

    int32_t split_m_slices;
    int32_t split_k_slices;
    int32_t split_k_buffers;
    int32_t single_split_k_buffer_size_in_bytes;
    int32_t split_k_counter_size_in_bytes;
    void *split_k_gmem;
    void *split_k_gmem_buffer_counter;
    void *split_k_gmem_final_counter;

    XMMA_HOST xmma::Error finalize(int32_t TILE_O,
                                   int32_t TILE_P,
                                   int32_t TILE_Q,
                                   int32_t TILE_T,
                                   int32_t TILE_R,
                                   int32_t TILE_S)
    {
        XMMA_CALL(Depthwise_colwolution_parameter_base::finalize());
        tiles_q = (q + TILE_Q - 1) / TILE_Q;
        tiles_p = (p + TILE_P - 1) / TILE_P;
        tiles_o = (o + TILE_O - 1) / TILE_O;
        tiles_pq = tiles_p * tiles_q;
        tiles_opq = tiles_o * tiles_pq;
        count_main_loop = (n * tiles_opq + split_m_slices - 1) / split_m_slices;
        count_main_loop = (count_main_loop + split_k_slices - 1) / split_k_slices;
        delta_tiles_q = split_m_slices % tiles_q;
        delta_tiles_p = (split_m_slices / tiles_q) % tiles_p;
        delta_tiles_o = ((split_m_slices / tiles_q) / tiles_p) % tiles_o;
        delta_tiles_n = (((split_m_slices / tiles_q) / tiles_p) / tiles_o) % n;
        xmma::find_divisor(mul_tiles_q, shr_tiles_q, tiles_q);
        xmma::find_divisor(mul_tiles_pq, shr_tiles_pq, tiles_pq);
        xmma::find_divisor(mul_tiles_opq, shr_tiles_opq, tiles_opq);

        tiles_s = (s + TILE_S - 1) / TILE_S;
        tiles_rs = ((r + TILE_R - 1) / TILE_R) * tiles_s;
        tiles_trs = ((t + TILE_T - 1) / TILE_T) * tiles_rs;
        xmma::find_divisor(mul_tiles_s, shr_tiles_s, tiles_s);
        xmma::find_divisor(mul_tiles_rs, shr_tiles_rs, tiles_rs);
        xmma::find_divisor(mul_tiles_trs, shr_tiles_trs, tiles_trs);

        return xmma::Error::SUCCESS;
        ;
    }

    XMMA_HOST void set_split_m_slices(int32_t in) { split_m_slices = in; }

    XMMA_HOST void set_split_k_slices(int32_t in) { split_k_slices = in; }

    XMMA_HOST void set_split_k_buffers(int32_t in) { split_k_buffers = in; }

    XMMA_HOST bool set_split_k_gmem(void *in)
    {
        if (in != nullptr) {
            split_k_gmem = in;
            return true;
        } else {
            split_k_gmem = nullptr;
            return false;
        }
    }

    XMMA_HOST bool set_split_k_gmem_buffer_counter(const int32_t offset)
    {

        if (split_k_slices > 1) {
            if (split_k_gmem != nullptr) {
                split_k_gmem_buffer_counter = move_pointer(split_k_gmem, offset);
                return true;
            } else {
                split_k_gmem_buffer_counter = nullptr;
                return false;
            }
        } else {
            return true;
        }
    }

    XMMA_HOST bool set_split_k_gmem_final_counter(const int32_t offset)
    {
        if (split_k_slices > 1) {
            if (split_k_gmem_buffer_counter != nullptr) {
                split_k_gmem_final_counter = move_pointer(split_k_gmem_buffer_counter, offset);
                return true;
            } else {
                split_k_gmem_final_counter = nullptr;
                return false;
            }
        } else {
            return true;
        }
    }

    XMMA_HOST void set_single_split_k_buffer_size_in_bytes(const int32_t in)
    {
        single_split_k_buffer_size_in_bytes = in;
    }

    XMMA_HOST void set_split_k_counter_size_in_bytes(const int32_t in)
    {
        split_k_counter_size_in_bytes = in;
    }
};

template <typename Cta_tile_>
struct Depthwise_colwolution_parameter : public Depthwise_colwolution_parameter_middle {
    public:
    // using Cta_tile_t = typename Kernel_traits::Cta_tile_t;
    // using Tile_memory_per_cta_t = typename Cta_tile_t::Tile_memory_per_cta_t;
    // using Tile_opq_t = typename Tile_memory_per_cta_t::Tile_opq_t;
    // using Gmem_tile_a_t = typename Kernel_traits::Gmem_tile_a_t;
    // using Gmem_tile_b_t = typename Kernel_traits::Gmem_tile_b_t;
    // static const int32_t LDGS_PER_THREAD_A = Gmem_tile_a_t::LDGS_PER_THREAD;
    // static const int32_t LDGS_PER_THREAD_B = Gmem_tile_b_t::LDGS_PER_THREAD;

    using Cta_tile_t = Cta_tile_;
    using Tile_memory_per_cta_t = typename Cta_tile_t::Tile_memory_per_cta_t;
    using Tile_opq_t = typename Tile_memory_per_cta_t::Tile_opq_t;
    using Tile_trs_t = typename Tile_memory_per_cta_t::Tile_trs_t;

    XMMA_HOST xmma::Error finalize()
    {
        return Depthwise_colwolution_parameter_middle::finalize(Tile_opq_t::DEPTH,
                                                                Tile_opq_t::HEIGHT,
                                                                Tile_opq_t::WIDTH,
                                                                Tile_trs_t::DEPTH,
                                                                Tile_trs_t::HEIGHT,
                                                                Tile_trs_t::WIDTH);
    }

    XMMA_HOST void print(bool enable_print)
    {
        if (enable_print) {
#define PRINT_POINTER(name, value) printf(#name " = %p\n", value)
#define PRINT_INT(name, value) printf(#name " = %d\n", value)
#define PRINT_UINT(name, value) printf(#name " = %u\n", value)
#define PRINT_FLOAT(name, value)                                                                   \
    float *ptr_##name = reinterpret_cast<float *>(&value);                                         \
    printf(#name " = %f\n", *ptr_##name)
#define PRINT_BOOL(name, value) printf(#name " = %s\n", (value ? "true" : "false"))
            PRINT_POINTER(a_gmem, this->gmem[Tensor_type::A]);
            PRINT_POINTER(b_gmem, this->gmem[Tensor_type::B]);
            PRINT_POINTER(c_gmem, this->gmem[Tensor_type::C]);
            PRINT_POINTER(d_gmem, this->gmem[Tensor_type::D]);
            PRINT_POINTER(tmp_gmem_trsg, this->tmp_gmem_trsg);
            PRINT_POINTER(final_gmem_trsg, this->final_gmem_trsg);
            PRINT_INT(g, this->g);
            PRINT_INT(n, this->n);
            PRINT_INT(d, this->d);
            PRINT_INT(h, this->h);
            PRINT_INT(w, this->w);
            PRINT_INT(t, this->t);
            PRINT_INT(r, this->r);
            PRINT_INT(s, this->s);
            PRINT_INT(o, this->o);
            PRINT_INT(p, this->p);
            PRINT_INT(q, this->q);
            PRINT_INT(pad_front, this->pad[0][0]);
            PRINT_INT(pad_back, this->pad[0][1]);
            PRINT_INT(pad_top, this->pad[1][0]);
            PRINT_INT(pad_bottom, this->pad[1][1]);
            PRINT_INT(pad_left, this->pad[2][0]);
            PRINT_INT(pad_right, this->pad[2][1]);
            PRINT_INT(stride_depth, this->stride[0]);
            PRINT_INT(stride_height, this->stride[1]);
            PRINT_INT(stride_width, this->stride[2]);
            PRINT_INT(dilation_depth, this->dilation[0]);
            PRINT_INT(dilation_height, this->dilation[1]);
            PRINT_INT(dilation_width, this->dilation[2]);
            PRINT_FLOAT(alpha, this->alpha);
            PRINT_FLOAT(beta, this->beta);
            PRINT_BOOL(is_colwolution, this->is_colwolution);
            PRINT_BOOL(is_dgrad, this->is_dgrad);

            PRINT_UINT(img_stride_n, this->img_stride_n);
            PRINT_UINT(img_stride_d, this->img_stride_d);
            PRINT_UINT(img_stride_h, this->img_stride_h);
            PRINT_UINT(img_stride_w, this->img_stride_w);
            PRINT_UINT(out_stride_n, this->out_stride_n);
            PRINT_UINT(out_stride_o, this->out_stride_o);
            PRINT_UINT(out_stride_p, this->out_stride_p);
            PRINT_UINT(out_stride_q, this->out_stride_q);
            PRINT_UINT(flt_stride_t, this->flt_stride_t);
            PRINT_UINT(flt_stride_r, this->flt_stride_r);
            PRINT_UINT(flt_stride_s, this->flt_stride_s);
            PRINT_INT(count_main_loop, this->count_main_loop);
            PRINT_INT(tiles_opq, this->tiles_opq);
            PRINT_INT(tiles_pq, this->tiles_pq);
            PRINT_INT(tiles_q, this->tiles_q);
            PRINT_UINT(mul_tiles_opq, this->mul_tiles_opq);
            PRINT_UINT(mul_tiles_pq, this->mul_tiles_pq);
            PRINT_UINT(mul_tiles_q, this->mul_tiles_q);
            PRINT_UINT(shr_tiles_opq, this->shr_tiles_opq);
            PRINT_UINT(shr_tiles_pq, this->shr_tiles_pq);
            PRINT_UINT(shr_tiles_q, this->shr_tiles_q);
            PRINT_INT(delta_tiles_n, this->delta_tiles_n);
            PRINT_INT(delta_tiles_o, this->delta_tiles_o);
            PRINT_INT(delta_tiles_p, this->delta_tiles_p);
            PRINT_INT(delta_tiles_q, this->delta_tiles_q);
            PRINT_INT(tiles_trs, this->tiles_trs);
            PRINT_INT(tiles_rs, this->tiles_rs);
            PRINT_INT(tiles_s, this->tiles_s);
            PRINT_UINT(mul_tiles_trs, this->mul_tiles_trs);
            PRINT_UINT(mul_tiles_rs, this->mul_tiles_rs);
            PRINT_UINT(mul_tiles_s, this->mul_tiles_s);
            PRINT_UINT(shr_tiles_trs, this->shr_tiles_trs);
            PRINT_UINT(shr_tiles_rs, this->shr_tiles_rs);
            PRINT_UINT(shr_tiles_s, this->shr_tiles_s);

            PRINT_POINTER(split_k_gmem, this->split_k_gmem);
            PRINT_POINTER(split_k_gmem_buffer_counter, this->split_k_gmem_buffer_counter);
            PRINT_POINTER(split_k_gmem_final_counter, this->split_k_gmem_final_counter);
            PRINT_INT(split_m_slices, this->split_m_slices);
            PRINT_INT(split_k_slices, this->split_k_slices);
            PRINT_INT(split_k_buffers, this->split_k_buffers);
            PRINT_INT(single_split_k_buffer_size_in_bytes,
                      this->single_split_k_buffer_size_in_bytes);
            PRINT_INT(split_k_counter_size_in_bytes, this->split_k_counter_size_in_bytes);

#undef PRINT_POINTER
#undef PRINT_INT
#undef PRINT_UINT
#undef PRINT_FLOAT
#undef PRINT_BOOL
        }
        return;
    }
};

} // namespace depthwise
} // namespace ext
} // namespace xmma

#endif
