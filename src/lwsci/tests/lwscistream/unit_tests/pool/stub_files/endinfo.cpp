//! \file
//! \brief LwSciStream endpoint information.
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
#include "endinfo.h"
#include "ipcbuffer.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Add type/attr pair to the list.
LwSciError
EndInfo::infoSet(
    uint32_t const userType,
    InfoPtr const& info) noexcept
{
    if (LwSciError_Success != initErr) {
        return initErr;
    }

    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        auto const insertion{ infoMap.emplace(userType, info) };
        if (!insertion.second) {
            return LwSciError_AlreadyInUse;
        }
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Look up the info for @a userType and return if found.
LwSciError
EndInfo::infoGet(
    uint32_t const userType,
    InfoPtr& info) const noexcept
{
    if (LwSciError_Success != initErr) {
        return initErr;
    }

    // Search for matching entry in consolidated list
    auto const mapEntry{ infoMap.find(userType) };

    if (infoMap.cend() == mapEntry) {
        return LwSciError_StreamInfoNotProvided;
    }

    info = mapEntry->second;
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call ipcBufferPack() to pack the endpoint info map.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
LwSciError
EndInfo::pack(
    IpcBuffer& buf) const noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
{
    if (LwSciError_Success != initErr) {
        return initErr;
    }

    return ipcBufferPack(buf, infoMap);
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! <b>Sequence of operations</b>
//! - Call ipcBufferUnpack() to unpack the endpoint info map.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
LwSciError
EndInfo::unpack(
    IpcBuffer& buf) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
{
    if (LwSciError_Success != initErr) {
        return initErr;
    }

    return ipcBufferUnpack(buf, infoMap);
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

} // namespace LwSciStream
