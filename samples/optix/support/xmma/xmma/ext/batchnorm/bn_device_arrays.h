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

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Bn_Device_Arrays {
    // Holds the reduced sum
    void *bn_sum_gmem, *dual_bn_sum_gmem;

    // Holds the reduced mean
    void *bn_mean_gmem, *dual_bn_mean_gmem;

    // Holds the reduced sum of squares
    // x --> COLW --> y
    // dx <-- DGRAD <-- dy
    // for fprop it holds y * y
    // for dgrad it holds x * dy
    void *bn_sum_of_squares_gmem, *dual_bn_sum_of_squares_gmem;

    // Ilwerse of standard deviation
    void *bn_ilw_stddev_gmem, *dual_bn_ilw_stddev_gmem;

    // Holds partial sum output from colwolution
    void *bn_partial_sums_gmem;

    // Effective scale and bias
    void *bn_scale_gmem;
    void *bn_bias_gmem;

    // Effective scale and bias for residual add tensor
    void *bn_res_scale_gmem;
    void *bn_res_bias_gmem;

    // Used for dgrad bn(s)
    // Mean and ilwerse stdandar deviataion
    void *dual_bn_fprop_mean_gmem;
    void *dual_bn_fprop_ilw_stddev_gmem;
    void *dual_bn_fprop_tensor_gmem;
    void *dual_bn_fprop_alpha_gmem;

    void *bn_fprop_mean_gmem;
    void *bn_fprop_ilw_stddev_gmem;
    void *bn_fprop_tensor_gmem;
    void *bn_fprop_alpha_gmem;

    // Used for dgrad bn(a)
    //  dbias  = sum_per_channel(gradient) / NPQ
    //  dscale = sumproduct( (bna_fprop_tensor - bna_fprop_mean), gradient) / NPQ
    //  dgrad  = bna_fprop_alpha * bna_fprop_ilw_stdev ( (gradient - dbias) - dscale *
    //  bna_fprop_ilw_stdev^2 (bna_fprop_tensor-bna_fprop_mean))
    //         = grad-scale * gradient + bna_fprop_tensor-scale * bna_fprop_tensor + bias
    //  grad-scale             = bna_fprop_alpha * bna_fprop_ilw_stdev
    //  bna_fprop_tensor-scale = -1 * bna_fprop_alpha * bna_fprop_ilw_stdev * dscale *
    //  bna_fprop_ilw_stdev^2 bias                   = -1 * bna_fprop_alpha * bna_fprop_ilw_stdev *
    //  dbias + dscale * bna_fprop_ilw_stdev^2 * bna_fprop_mean
    void *bna_bias_gmem;
    void *bna_grad_scale_gmem;
    void *bna_fprop_tensor_gmem;
    void *bna_fprop_tensor_scale_gmem;

    // Batchnorm output
    void *bn_out_gmem;

    // Residual add input
    void *bn_residual_gmem;

    // Residual add output
    void *bn_res_add_relu_out_gmem;

    // Bitmask RELU output
    void *bn_bitmask_relu_out_gmem;

    // Input bitmask for dRelu
    void *bn_drelu_bitmask;

    // Standalone dbn(a) output
    void *standalone_dbna_output;
    void *dual_standalone_dbna_output;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace batchnorm
}  // namespace ext
}  // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
