//! \file
//! \brief LwSciStream ipc communication implementation.
//!
//! \copyright
//! Copyright (c) 2018-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.
#include <cstdint>
#include <sys/eventfd.h>
#include <utility>
#include <cstddef>
#include <iostream>
#include <array>
#include <unistd.h>
#include "covanalysis.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "ipccomm_common.h"
#include "fdutils.h"
#include "ipccomm.h"

namespace LwSciStream {

//! Flags used for creating event FDs
static constexpr int32_t IPC_EFD_FLAGS {
    static_cast<int32_t>(static_cast<uint32_t>(EFD_CLOEXEC) |
                         static_cast<uint32_t>(EFD_SEMAPHORE))
};

//! \brief IpcComm constructor
//!        on failure initSuccess is set to false
//!
//! \param [in] ipcHandle: handle to lwsciipc endpoint set by Application.
//!
IpcComm::IpcComm(LwSciIpcEndpoint const ipcHandle) noexcept :
    handle(ipcHandle),
    channelInfo{},
    initSuccess(false),
    ipcEventNotify(ILWALID_FD),
    internalWriteNotify(eventfd(0U, IPC_EFD_FLAGS)),
    internalDisconnNotify(eventfd(0U, IPC_EFD_FLAGS)),
    internalWritePending(false)
{

    // initialize ipc event notifier
    LwSciError err = LwSciIpcGetLinuxEventFd(handle, &ipcEventNotify);

    // Make sure file descriptors were successfully allocated
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if ((LwSciError_Success != err) ||
        ((ipcEventNotify < 0) || (ipcEventNotify >= FD_SETSIZE))) {
        return;
    }
    if ((internalWriteNotify < 0) || (internalWriteNotify >= FD_SETSIZE)) {
        return;
    }
    if ((internalDisconnNotify < 0) ||
        (internalDisconnNotify >= FD_SETSIZE)) {
        return;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // Query endpoint info
    if (LwSciError_Success == LwSciIpcGetEndpointInfo(handle, &channelInfo)) {
        // ipc / ivc frame is big enough to contain a frame
        if (channelInfo.frame_size > 0U) {
            // mark successfully created
            initSuccess = true;
        }
    }
}

//! \brief IpcComm destructor
//!
IpcComm::~IpcComm(void) noexcept
{
    if (ILWALID_FD != internalWriteNotify) {
        static_cast<void>(close(internalWriteNotify));
        internalWriteNotify = ILWALID_FD;
    }

    if (ILWALID_FD != internalDisconnNotify) {
        static_cast<void>(close(internalDisconnNotify));
        internalDisconnNotify = ILWALID_FD;
    }

    LwSciIpcCloseEndpoint(handle);
}

//! \brief Accessor to check if constructor encountered any errors
//!
//! \return if IpcComm was created successfully.
bool IpcComm::isInitSuccess(void) const noexcept
{
    return initSuccess;
}

//! \brief Accessor to get channel frame size
//!
//! \return lwsciipc endpoint framesize
uint32_t IpcComm::getFrameSize(void) const noexcept
{
    return channelInfo.frame_size;
}

//! \brief Fetch all lwsciipc events available since last wait
//!        and mark all relevant events
//!
//! \return IpcQueryFlags
//! - readReady: set if ipc read frame is available
//! - writeReady: set if ipc write frame is available
//! - connectReady: set if connection_est event is received
//! - LwSciError_Success: some event is available and corresponding operations
//!                       can be successfully exelwted on next frame.
//! - err: LwSciError encountered while fetching next event
//!
IpcQueryFlags IpcComm::waitForEvent(void) noexcept
{
    IpcQueryFlags returnFlags {false, false, LwSciError_Success};
    bool disconnectDetected { false };

    // Loop until one of:
    //   An error oclwrs
    //   There is a message to receive
    //   There is a message to send, and room for it
    //   Disconnect is triggered and there are no more messages to send
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    do {

        // Check IPC for read or write events
        uint32_t receivedEvents { 0U };
        returnFlags.err = LwSciIpcGetEvent(handle, &receivedEvents);
        if (LwSciError_Success != returnFlags.err) {
            break;
        }
        if (0U != (receivedEvents & LW_SCI_IPC_EVENT_READ)) {
            returnFlags.readReady = true;
        }
        if ((0U != (receivedEvents & LW_SCI_IPC_EVENT_WRITE)) &&
            internalWritePending) {
            returnFlags.writeReady = true;
            internalWritePending = false;
        }

        // If a read or write message is ready, break out and return
        if (returnFlags.readReady || returnFlags.writeReady) {
            break;
        }

        // Set up FD set to wait for IPC event or new event FD signal
        //   Once the write or disconnect event FD is detected, we don't
        //     check it again, since select will just return immediately.
        //   Note: The only way we can reach this loop after one or both
        //     of those FDs have been detected is if there is a write
        //     message pending and there wasn't previously room for it,
        //     so waiting for the IPC notification should be sufficient.
        fd_set rfds;
        fdZEROWrap(rfds);
        int32_t maxFd;
        maxFd = fdSETWrap(ipcEventNotify, rfds, 0);
        if (!internalWritePending) {
            maxFd = fdSETWrap(internalWriteNotify, rfds, maxFd);
        }
        if (!disconnectDetected) {
            maxFd = fdSETWrap(internalDisconnNotify, rfds, maxFd);
        }

        // Wait until event is received
        if (0 <= select(maxFd + 1, &rfds, nullptr, nullptr, nullptr)) {

            // If write FD triggered, mark write as pending
            if ((internalWriteNotify <= maxFd) &&
                fdISSETWrap(internalWriteNotify, rfds)) {
                internalWritePending = true;
            }

            // If disconnect FD triggered, mark disonnect as pending
            if ((internalDisconnNotify <= maxFd) &&
                fdISSETWrap(internalDisconnNotify, rfds)) {
                disconnectDetected = true;
            }

        } else {
            // select failed
            returnFlags.err = LwSciError_StreamInternalError;
        }

    } while ((LwSciError_Success == returnFlags.err) &&
             (!disconnectDetected || internalWritePending));
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    return returnFlags;
}

//! \brief Wait for connection-establishment signal from other end
//!        of Ipc endpoint
//! \return error code returned by waitForEvent
//! - LwSciError_Success: message was sent successfully
//! - LwSciError_IlwalidState: waitForEvent failed
//!
// Note: Since this doesn't modify any members, it could be const. But the
//       QNX version does modify members, so we need the prototype to be
//       non-const. This introduces autosar violations.
// If/when we care about violations in the linux-specific code, we may need
// to virtualize this class.
LwSciError IpcComm::waitForConnection(void) noexcept
{
    // Per LwSciIpc specification, a LwSciIpcGetEvent or LwSciIpcWrite
    // is required to trigger state change on the other end of the connection.
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    while (true) {

        // Check IPC for connection event
        uint32_t receivedEvents { 0U };
        LwSciError const err { LwSciIpcGetEvent(handle, &receivedEvents) };
        if (LwSciError_Success != err) {
            return err;
        }
        if (0U != (receivedEvents & LW_SCI_IPC_EVENT_CONN_EST_ALL)) {
            return LwSciError_Success;
        }

        // Set up FD set to wait for IPC event or disconnect
        fd_set rfds;
        fdZEROWrap(rfds);
        int32_t maxFd;
        maxFd = fdSETWrap(ipcEventNotify, rfds, 0);
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A0_1_1), "Bug 2738197")
        maxFd = fdSETWrap(internalDisconnNotify, rfds, maxFd);

        // Wait until event is received
        if (0 <= select(maxFd + 1, &rfds, nullptr, nullptr, nullptr)) {
            if (!fdISSETWrap(ipcEventNotify, rfds)) {
                return LwSciError_StreamInternalError;
            }
        } else {
            // select failed
            return LwSciError_StreamInternalError;
        }
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

// Note: Since this doesn't modify any members, it could be const. But the
//       QNX version does modify members, so we need the prototype to be
//       non-const. This introduces autosar violations.
// If/when we care about violations in the linux-specific code, we may need
// to virtualize this class.
LwSciError IpcComm::signalWrite(void) noexcept
{
    uint64_t sig { 1ULL };
    if (-1 != write(internalWriteNotify, &sig, sizeof(sig))) {
        return LwSciError_Success;
    } else {
        // Get errno and return the corresponding LwSciError
        return LwSciError_StreamInternalError;
    }
}

// Note this function only exists to support the QNX version.
// But having this function which does nothing introduces autosar violations.
// If/when we care about violations in the linux-specific code, we may need
// to virtualize this class.
LwSciError IpcComm::flushWriteSignals(void) noexcept
{
    // Nothing to do.
    return LwSciError_Success;
}

LwSciError IpcComm::signalDisconnect(void) const noexcept
{
    uint64_t sig { 1ULL };
    if (-1 != write(internalDisconnNotify, &sig, sizeof(sig))) {
        return LwSciError_Success;
    } else {
        // Get errno and return the corresponding LwSciError
        return LwSciError_StreamInternalError;
    }
}

//! \brief send new frame across ipc channel.
//!
//! \param [in] buffer: Buffer containing packed data to send
//!
//! \return error code
//! - LwSciError_Success: message was sent successfully
//! - LwSciError_IlwalidState: waitForEvent failed
//!
// Note: Since this doesn't modify any members, it could be const. But the
//       QNX version does modify members, so we need the prototype to be
//       non-const. This introduces autosar violations.
// If/when we care about violations in the linux-specific code, we may need
// to virtualize this class.
LwSciError IpcComm::sendFrame(IpcBuffer& buffer) noexcept
{
    // Switch to send mode if not already done
    LwSciError err { buffer.changeMode(IpcBuffer::UserMode::Send) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Retrieve next block of data to try to send
    IpcBuffer::CBlob blob;
    err = buffer.sendInfoGet(blob);
    if (LwSciError_Success != err) {
        return err;
    }
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    // This shouldn't be possible if the caller did everything right
    if ((nullptr == blob.data) ||
        ((0U == blob.size) ||
         (static_cast<size_t>(channelInfo.frame_size) < blob.size))) {
        return LwSciError_StreamInternalError;
    }

    // Send data
    int32_t bytesWritten;
    err = LwSciIpcWrite(handle, blob.data, blob.size, &bytesWritten);

    // If successful, advance buffer's send offset
    if (LwSciError_Success == err) {
        err = buffer.sendSizeAdd(static_cast<size_t>(bytesWritten));
    }

    // If succesful and no more data to send update write even fd semaphore
    //   Effectively this decrements number of pending write requests by 1
    if ((LwSciError_Success == err) && buffer.sendDone()) {
        // update write event fd semaphore
        uint64_t readVal { 0ULL };
        ssize_t const readRv
            { read(internalWriteNotify, &readVal, sizeof(readVal)) };
        err = ((readRv >= static_cast<ssize_t>(0L)) && (LONG_ONE == readVal))
            ? LwSciError_Success
            : LwSciError_StreamInternalError;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
    return err;
}

//! \brief read single frame containing lwscistream message from ipc channel
//!
//! \param [in] buffer: destination buffer to read ipc frame into
//!                     checks must be made to ensure destination is
//!                     large enough to hold frames worth of data
//!
//! \return error code
//! \todo other errors
//!
LwSciError IpcComm::readFrame(IpcBuffer& buffer) const noexcept
{
    // Switch to receive mode if not already done
    LwSciError err { buffer.changeMode(IpcBuffer::UserMode::Recv) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Retrieve next block of data to try to read into
    IpcBuffer::VBlob blob;
    err = buffer.recvInfoGet(blob);
    if (LwSciError_Success != err) {
        return err;
    }

    // Receive data
    int32_t bytesRead {0};
    err = LwSciIpcRead(handle, blob.data, blob.size, &bytesRead);

    // If successful, advance buffer's receive offset
    if (LwSciError_Success == err) {
        err = buffer.recvSizeAdd(static_cast<size_t>(bytesRead));
    }

    return err;
}

} // namespace LwSciStream
