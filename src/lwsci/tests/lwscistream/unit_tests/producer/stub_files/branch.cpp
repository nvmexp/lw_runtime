//! \file
//! \brief LwSciStream branch tracking for multicast.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include <limits>
#include "lwscistream_common.h"
#include "branch.h"

namespace LwSciStream {

// Constant expression is defined in header but still requires instantiation
constexpr BranchMap::Range BranchMap::emptyRange;

//! <b>Sequence of operations</b>
//! - Initialize range vector to size equal to number of connections, with
//!   empty range values.
BranchMap::BranchMap(
    size_t const paramConnCount) noexcept :
        connTrack(paramConnCount),
        connRange(),
        initError(LwSciError_Success)
{
    // Allocate vector
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        connRange.resize(paramConnCount, emptyRange);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
    } catch (...) {
        initError = LwSciError_InsufficientMemory;
    }
}

//! <b>Sequence of operations</b>
//! - Call BranchTrack::set() to validate index and make sure its value is
//!   only set once.
LwSciError
BranchMap::set(
    size_t const index,
    size_t const count) noexcept
{
        // CERT-mandated check that should never fail in practice
    if (MAX_INT_SIZE <= count) {
        return LwSciError_Overflow;
    }

    // Check whether indexed connection has been set, and if not take
    //   responsibility for doing so
    LwSciError const err { connTrack.set(index) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Set the connection's count and update the starting point for all
    //   connections after it
    connRange[index].count = count;
    for (size_t i { index+1U }; connRange.size() > i; ++i) {
        // CERT-mandated check that should never fail in practice
        if (MAX_INT_SIZE <= connRange[i].start) {
            return LwSciError_Overflow;
        }
        connRange[i].start += count;
    }

    return LwSciError_Success;
}

} // namespace LwSciStream
