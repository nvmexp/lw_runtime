/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(LDPC_MS_MKA_FLOODING_FLAT_HPP_INCLUDED_)
#define LDPC_MS_MKA_FLOODING_FLAT_HPP_INCLUDED_

// Min-sum, Multi-kernel w/Atomic, Flooding , Flat BG Table LDPC Implementation

#include "ldpc.hpp"

////////////////////////////////////////////////////////////////////////
// ldpc
namespace ldpc
{
////////////////////////////////////////////////////////////////////////
// decode_multi_kernel_atomic_flat()
lwphyStatus_t decode_multi_kernel_atomic_flat(LDPC_output_t&      tDst,
                                              const_tensor_pair&  tLLR,
                                              const LDPC_config&  config,
                                              float               normalization,
                                              lwphyLDPCResults_t* results,
                                              void*               workspace,
                                              lwdaStream_t        strm);

////////////////////////////////////////////////////////////////////////
// decode_multi_kernel_atomic_flat_workspace_size()
std::pair<bool, size_t> decode_multi_kernel_atomic_flat_workspace_size(const LDPC_config& config);

} // namespace ldpc

#endif // !defined(LDPC_MS_MKA_FLOODING_FLAT_HPP_INCLUDED_)
