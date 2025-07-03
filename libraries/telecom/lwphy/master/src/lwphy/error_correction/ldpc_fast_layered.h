/***************************************************************************************************
 * Copyright (c) 2011-2020, LWPU CORPORATION.  All rights reserved.
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

#include "ldpc.hpp"

////////////////////////////////////////////////////////////////////////
// ldpc
namespace ldpc
{
////////////////////////////////////////////////////////////////////////
// decode_fast_layered()
lwphyStatus_t decode_fast_layered(LDPC_output_t&         tDst,
                                  const_tensor_pair&     tLLR,
                                  const LDPC_config&     config,
                                  float                  normalization,
                                  lwphyLDPCResults_t*    results,
                                  void*                  workspace,
                                  lwphyLDPCDiagnostic_t* diag,
                                  lwdaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_fast_layered_workspace_size()
std::pair<bool, size_t> decode_fast_layered_workspace_size(const LDPC_config& cfg);

} // namespace ldpc


////////////////////////////////////////////////////////////////////////////////////////////////////

