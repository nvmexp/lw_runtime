/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(LDPC2_SHARED_CLUSTER_HPP_INCLUDED__)
#define LDPC2_SHARED_CLUSTER_HPP_INCLUDED__

#include "ldpc2.hpp"

namespace ldpc2
{

lwphyStatus_t decode_ldpc2_shared_cluster_index(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);

lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG1_Z64 (const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG1_Z96 (const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG1_Z128(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG1_Z160(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG1_Z192(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG1_Z224(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG1_Z256(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG1_Z288(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG1_Z320(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG1_Z352(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG1_Z384(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);

lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG2_Z64 (const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG2_Z96 (const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG2_Z128(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG2_Z160(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG2_Z192(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG2_Z224(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG2_Z256(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG2_Z288(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG2_Z320(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG2_Z352(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);
lwphyStatus_t decode_ldpc2_shared_cluster_index_half_BG2_Z384(const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, lwdaStream_t strm);

} // namespace ldpc2

#endif // !defined(LDPC2_SHARED_CLUSTER_HPP_INCLUDED__)
