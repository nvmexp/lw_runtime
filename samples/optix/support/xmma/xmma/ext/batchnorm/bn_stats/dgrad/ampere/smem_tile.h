/***************************************************************************************************
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once
#include <xmma/smem_tile.h>
#include <xmma/turing/smem_tile.h>

namespace xmma {

    template< typename Traits, typename Cta_tile, typename Layout>
    struct Swizzle_turing_hmma_fp32_epilogue_bn_stats {
    };

    template< typename Traits, typename Cta_tile >
    struct Swizzle_turing_hmma_fp32_epilogue_bn_stats<Traits, Cta_tile, Row>
        : public Swizzle_turing_epilogue<Traits, Cta_tile, cutlass::half_t> {

        // The base class.
        using Base = Swizzle_turing_epilogue<Traits, Cta_tile, cutlass::half_t>;
        // The XMMA tile.
        using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

        // Ctor.
        inline __device__ Swizzle_turing_hmma_fp32_epilogue_bn_stats(void *smem,
                                                                     int tidx) : Base(smem, tidx) {
        }

        // Load from the tile in shared memory.
        template<typename Fragment_post_swizzle>
        inline __device__
        void load(Fragment_post_swizzle &dst, int oi) const {
            const int offset = oi * Base::PIXELS_PER_STG * Base::BYTES_PER_ROW_WITH_SKEW;
            uint4 tmp;
            lds(tmp, this->smem_ + this->smem_read_offset_ + offset);
            dst.reg(0) = tmp.x;
            dst.reg(1) = tmp.y;
            dst.reg(2) = tmp.z;
            dst.reg(3) = tmp.w;
        }

        // Store to the tile in shared memory.
        template<typename Fragment_pre_swizzle>
        inline __device__
        void store(int ni, const Fragment_pre_swizzle &c) {
            #pragma unroll
            for( int mi = 0; mi < Base::M_PER_XMMA_PER_THREAD; ++mi ) {
                int offset = mi * Base::THREADS_PER_XMMA_M * Base::BYTES_PER_ROW_WITH_SKEW +
                            ni * (Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA :
                            Xmma_tile::N_PER_XMMA_PER_CTA) * sizeof(cutlass::half_t);

                uint32_t ptr = this->smem_ + this->smem_write_offset_ + offset;
                sts(ptr +  0, c.reg(2*mi + 0));
                sts(ptr + 16, c.reg(2*mi + 1));
            }
        }
    };

} //namespace xmma