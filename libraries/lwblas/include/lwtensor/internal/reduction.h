#pragma once

#include <lwda_runtime.h>

#include <cstdint>
#include <lwtensor/types.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{
    struct ReductionParams;
    class Context;
/**
  * Implements D = alpha opReduce(opAB(opA(A), opB(B))) + beta * opC(C)
  *
  * This funciton implements GEMV- and DOT-like tensor contractions.
  */
lwtensorStatus_t tensorReductionDispatch(const Context* ctx, const void* alpha, const void* A, const void* B,
                                           const void* beta,  const void* C,       void* D,
                                           const ReductionParams& params,
                                           void* workspace, uint64_t workspaceSize,
                                           lwdaStream_t stream, bool dispatch);

}

