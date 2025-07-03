//
// Copyright 2020 LWPU Corporation. All rights reserved.
//

#include <xmma/implicit_gemm/fprop/traits.h>
#include <xmma/implicit_gemm/fprop/params.h>
#include <xmma/implicit_gemm/fprop/utils.h>

#include <xmma/gemm/kernel.h>
#include <xmma/utils.h>

#include <xmma/ext/colw_with_2x2_pooling/kernel.h>
#include <xmma/ext/colw_with_2x2_pooling/traits.h>

#include <layers.h>

namespace optix_exp {

struct Mem_desc_params {
  /* ii= 0 offset=   0 */ uint64_t descriptor_a;
  /* ii= 1 offset=   8 */ uint64_t descriptor_b;
  /* ii= 2 offset=  16 */ uint64_t descriptor_c;
  /* ii= 3 offset=  24 */ uint64_t descriptor_d;
};

struct Bias_params {
  /* ii= 0 offset=   0 */ void* bias;
};

struct Relu_params {
  /* ii= 0 offset=   0 */ uint32_t leak_minus_one;
};

struct Colw_params {
  /* ii= 0 offset=   0 */ void* img_gmem;
  /* ii= 1 offset=   8 */ void* flt_gmem;
  /* ii= 2 offset=  16 */ void* out_gmem;
  /* ii= 3 offset=  24 */ void* res_gmem;
  /* ii= 4 offset=  32 */ int32_t n;
  /* ii= 5 offset=  36 */ int32_t d;
  /* ii= 6 offset=  40 */ int32_t h;
  /* ii= 7 offset=  44 */ int32_t w;
  /* ii= 8 offset=  48 */ int32_t g;
  /* ii= 9 offset=  52 */ int32_t c;
  /* ii=10 offset=  56 */ int32_t k;
  /* ii=11 offset=  60 */ int32_t t;
  /* ii=12 offset=  64 */ int32_t r;
  /* ii=13 offset=  68 */ int32_t s;
  /* ii=14 offset=  72 */ int32_t o;
  /* ii=15 offset=  76 */ int32_t p;
  /* ii=16 offset=  80 */ int32_t q;
  /* ii=17 offset=  84 */ int32_t pad[3][2];
  /* ii=18 offset= 108 */ int32_t stride[3];
  /* ii=19 offset= 120 */ int32_t dilation[3];
  /* ii=20 offset= 132 */ int32_t cross_correlation;
  /* ii=21 offset= 136 */ uint16_t alpha;
  /* ii=22 offset= 138 */ uint16_t beta;
  /* ii=23 offset= 140 */ int32_t with_residual;
  /* ii=24 offset= 144 */ Mem_desc_params mem_descriptors;
  /* ii=25 offset= 176 */ uint32_t img_stride_n;
  /* ii=26 offset= 180 */ uint32_t img_stride_d;
  /* ii=27 offset= 184 */ uint32_t img_stride_h;
  /* ii=28 offset= 188 */ uint32_t img_stride_w;
  /* ii=29 offset= 192 */ uint32_t img_stride_c;
  /* ii=30 offset= 196 */ uint32_t out_stride_n;
  /* ii=31 offset= 200 */ uint32_t out_stride_d;
  /* ii=32 offset= 204 */ uint32_t out_stride_h;
  /* ii=33 offset= 208 */ uint32_t out_stride_w;
  /* ii=34 offset= 212 */ uint32_t out_stride_c;
  /* ii=35 offset= 216 */ int32_t filter_t_per_cta;
  /* ii=36 offset= 220 */ int32_t filter_r_per_cta;
  /* ii=37 offset= 224 */ int32_t filter_s_per_cta;
  /* ii=38 offset= 228 */ int32_t filter_trs_per_cta;
  /* ii=39 offset= 232 */ int32_t filter_rs_per_cta;
  /* ii=40 offset= 236 */ uint32_t mask_t;
  /* ii=41 offset= 240 */ uint32_t mask_r;
  /* ii=42 offset= 244 */ uint32_t mask_s;
  /* ii=43 offset= 248 */ int64_t a_delta[32];
  /* ii=44 offset= 504 */ int64_t b_delta[32];
  /* ii=45 offset= 760 */ uint32_t loop_start;
  /* ii=46 offset= 764 */ uint32_t loop_residue;
  /* ii=47 offset= 768 */ uint32_t loop_residue_k;
  /* ii=48 offset= 772 */ int32_t split_k_t;
  /* ii=49 offset= 776 */ int32_t split_k_r;
  /* ii=50 offset= 780 */ int32_t split_k_c;
  /* ii=51 offset= 784 */ int32_t split_k_trs;
  /* ii=52 offset= 788 */ int32_t split_k_rs;
  /* ii=53 offset= 792 */ uint32_t wc;
  /* ii=54 offset= 796 */ uint32_t nopq;
  /* ii=55 offset= 800 */ uint32_t opq;
  /* ii=56 offset= 804 */ uint32_t pq;
  /* ii=57 offset= 808 */ uint32_t trsc;
  /* ii=58 offset= 812 */ uint32_t trs;
  /* ii=59 offset= 816 */ uint32_t mul_opq;
  /* ii=60 offset= 820 */ uint32_t shr_opq;
  /* ii=61 offset= 824 */ uint32_t mul_pq;
  /* ii=62 offset= 828 */ uint32_t shr_pq;
  /* ii=63 offset= 832 */ uint32_t mul_q;
  /* ii=64 offset= 836 */ uint32_t shr_q;
  /* ii=65 offset= 840 */ int32_t ctas_p;
  /* ii=66 offset= 844 */ int32_t ctas_q;
  /* ii=67 offset= 848 */ uint32_t mul_ctas_q;
  /* ii=68 offset= 852 */ uint32_t shr_ctas_q;
  /* ii=69 offset= 856 */ Bias_params fusion0;
  /* ii=70 offset= 864 */ Relu_params fusion1;
};

struct Simple_tile_distribution {
    // Ctor.
    template< typename Params >
    inline __device__ Simple_tile_distribution(const Params&, const dim3 &bid)
        : tile_m_(bid.x)
        , tile_n_(bid.y)
        , tile_z_(bid.z) {
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

    // The index for the current tile in the M/N/Z dimension of the grid.
    int tile_m_, tile_n_, tile_z_;
};

struct Xmma_colw {

    Xmma_colw() {}
    ~Xmma_colw() {}

    void setup( int arch, bool fusedPooling, int M, int N, int K, int c, int k, int w, int h, unsigned int act, float actAlpha );

    OptixResult run( void* img_gmem, void* out_gmem, void* flt_gmem, void* bias, lwdaStream_t stream, ErrorDetails& errDetails );

private:
    Colw_params   m_params;
    int           m_arch;
    int           m_grid[3];
    int           m_ctaThreads;
    size_t        m_smemSize;
    int           m_M;
    int           m_N;
    int           m_K;
    bool          m_fusedPooling;
};

};
