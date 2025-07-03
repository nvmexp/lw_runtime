//! \file
//! \brief LwSciStream endpoint information.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef ENDINFO_H
#define ENDINFO_H

#include <cstdint>
#include <atomic>
#include <vector>
#include <map>
#include "lwscierror.h"
#include "lwscistream_common.h"
#include "ipcbuffer.h"

namespace LwSciStream {

//! \brief Data structure to track information about an endpoint that is set
//!        before the endpoint is connected and then made available to other
//!        blocks in various forms.
class EndInfo final
{
public:
    // Will be manipulating vectors of these, so all basic operators
    //   are required. For now, the defaults are fine. Might need to
    //   specifically implement these depending on what kind of data
    //   is added.

    //! \brief Default void constructor.
    EndInfo(void) noexcept { /* initialize with defaults */ };

    //! \brief Copy constructor.
    //!   Set the init error if there's any failure.
    EndInfo(EndInfo const& paramEndInfo) noexcept
    {
        try {
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            infoMap = paramEndInfo.infoMap;
        } catch (...) {
            initErr = LwSciError_InsufficientMemory;
        }
    };

    //! \brief Copy operator.
    //!   Set the init error if there's any failure.
    EndInfo& operator=(EndInfo const& paramEndInfo) noexcept
    {
        if (this == &paramEndInfo) {
            return *this;
        }

        try {
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            infoMap = paramEndInfo.infoMap;
        } catch (...) {
            initErr = LwSciError_InsufficientMemory;
        }
        return *this;
    };

    //! \brief Default move constructor.
    EndInfo(EndInfo&&) noexcept = default;
    //! \brief Default move operator.
    EndInfo& operator=(EndInfo&&) & noexcept      = default;
    //! \brief Default destructor.
    ~EndInfo(void) noexcept                       = default;

    //! \brief Adds a new info to the list.
    //!
    //! \param [in] userType: Unique user-defined type to identify the info.
    //! \param [in] info: Endpoint info referenced by the shared pointer.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The new info was successfully added to the list.
    //! * LwSciError_AlreadyInUse: An endpoint info with the specified
    //!   @a userType already exists in the list.
    //! * LwSciError_InsufficientMemory: Unable to grow storage for additional
    //!   endpoint info.
    //! * Any error at object construction.
    LwSciError infoSet(
        uint32_t const userType,
        InfoPtr const& info) noexcept;

    //! \brief Query the endpoint info for an unique user-defined type.
    //!
    //! \param [in] userType: Unique user-defined type to identify the info.
    //! \param [in,out] info: Location in which to store the InfoPtr.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The attribute list was successfully queried.
    //! * LwSciError_StreamInfoNotProvided: Endpoint info for @a userType
    //!   not exist.
    //! * Any error at object construction.
    LwSciError infoGet(
        uint32_t const userType,
        InfoPtr& info) const noexcept;

    //! \brief Pack info to IPC buffer.
    //!
    //! \param [in] buf: Buffer to pack info into.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: Info packed successfully
    //! * Any error returned by IpcBuffer::packVal().
    //! * Any error at object construction.
    LwSciError pack(IpcBuffer& buf) const noexcept;

    //! \brief Unpack info from IPC buffer.
    //!
    //! \param [in] buf: Buffer to unpack info from.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: Info unpacked successfully
    //! * Any error returned by IpcBuffer::unpackVal().
    //! * Any error at object construction.
    LwSciError unpack(IpcBuffer& buf) noexcept;

private:
    //! \brief Error encountered during construction.
    //!   Initialized to LwSciError_Success and then overwritten by the
    //!   first failure, if any, during the constructor.
    LwSciError initErr{ LwSciError_Success };

    //! \brief Map in which list is compiled from user input as a set of
    //!   (userType, InfoPtr) entries. Initialized to empty at creation.
    std::map<uint32_t, InfoPtr> infoMap{ };
};

//! \brief Wrapper to allow use of EndInfo::pack with generic templates.
//!
//! \param [in,out] buf: Buffer in which the object will be packed.
//! \param [in] val: Value to be packed.
//!
//! \return LwSciError
//! - LwSciError_Success - Packing succeeded
//! - Any error returned by EndInfo::pack().
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
inline LwSciError ipcBufferPack(IpcBuffer& buf, EndInfo const& val) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    return val.pack(buf);
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! \brief Wrapper to allow use of EndInfo::unpack with generic templates.
//!
//! \param [in,out] buf: Buffer from which the object will be unpacked.
//! \param [in,out] val: Location in which to store the value.
//!
//! \return LwSciError
//! - LwSciError_Success - Unpacking succeeded
//! - Any error returned by EndInfo::unpack().
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
inline LwSciError ipcBufferUnpack(IpcBuffer& buf, EndInfo& val) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    return val.unpack(buf);
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! \brief Vector of endpoint information objects
using EndInfoVector = std::vector<EndInfo>;

} // namespace LwSciStream
#endif // ENDINFO_H
