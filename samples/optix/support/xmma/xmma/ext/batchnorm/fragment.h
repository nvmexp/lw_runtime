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
#include <xmma/fragment.h>
#include <xmma/ampere/fragment.h>

namespace xmma {

template< typename Traits, typename Cta_tile>
struct Fragment_c_bn_stats {
};

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_fp32_c_bn_stats
    : public Fragment_hmma_base_c<Traits, Cta_tile> {

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(float, const Fragment_post_swizzle &frag) {
        this->reg(0) = frag.reg(0);
        this->reg(1) = frag.reg(1);
        this->reg(2) = frag.reg(2);
        this->reg(3) = frag.reg(3);
    }
};

template< typename Cta_tile>
struct Fragment_c_bn_stats<Ampere_hmma_fp32_traits, Cta_tile>
    : public Fragment_hmma_fp32_c_bn_stats<Ampere_hmma_fp32_traits, Cta_tile> {
};

template< typename Traits, typename Cta_tile, bool = (Cta_tile::WARPS_K > 1) >
struct Fragment_epilogue_post_swizzle_bn_stats {
};

template< typename Cta_tile >
    struct Fragment_epilogue_post_swizzle_bn_stats<Ampere_hmma_fp32_traits, Cta_tile, true>
        : public Fragment_hmma_fp32_epilogue_post_swizzle<Ampere_hmma_fp32_traits, Cta_tile> {
        };

template< typename Cta_tile >
struct Fragment_epilogue_post_swizzle_bn_stats<Ampere_hmma_fp32_traits, Cta_tile, false>
    : public Fragment_hmma_fp16_epilogue_post_swizzle<Ampere_hmma_fp32_traits, Cta_tile> {
};


template< typename Traits, typename Cta_tile, bool = (Cta_tile::WARPS_K > 1) >
struct Fragment_epilogue_pre_swizzle_bn_stats {
};

template< typename Cta_tile >
struct Fragment_epilogue_pre_swizzle_bn_stats<Ampere_hmma_fp32_traits, Cta_tile, false>
    : public Fragment_hmma_fp16_epilogue_pre_swizzle<Ampere_hmma_fp32_traits, Cta_tile> {

            // The traits.
            using Traits = Ampere_hmma_fp32_traits;
            // The aclwmulators from the main loop.
            using Aclwmulators = Fragment_aclwmulator<Traits>;

            //Colwert from fp16 aclwmulators to fp16 outputs.
            inline __device__ void colwert(float alpha, const Aclwmulators &acc) {
                this->reg(0) = float2_to_half2(alpha * acc.elt(0), alpha * acc.elt(1));
                this->reg(1) = float2_to_half2(alpha * acc.elt(4), alpha * acc.elt(5));
                this->reg(2) = float2_to_half2(alpha * acc.elt(2), alpha * acc.elt(3));
                this->reg(3) = float2_to_half2(alpha * acc.elt(6), alpha * acc.elt(7));
            }
    };
}
