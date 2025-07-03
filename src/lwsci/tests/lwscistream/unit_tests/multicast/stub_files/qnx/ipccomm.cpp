//! \file
//! \brief LwSciStream ipc communication implementation on QNX.
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
#include <cstddef>
#include <cassert>
#include <unistd.h>
#include <atomic>
#include <iostream>
#include <array>
#include <utility>
#include <sys/neutrino.h>
#include <lwsciipc_internal.h>
#include "covanalysis.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "lwscistream_common.h"
#include "ipccomm_common.h"
#include "ipccomm.h"

namespace LwSciStream {

// QNX pulse code used by LwSciIpc to monitor LwSciIpc events
constexpr int8_t DELTA {1};
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
constexpr int8_t ipcCode {_PULSE_CODE_MINAVAIL + DELTA};
// QNX pulse code used by internal interrupt.
constexpr int8_t interruptCode {ipcCode + DELTA};
constexpr int32_t valWriteReq {static_cast<int32_t>(0xDABBAD00)};
constexpr int32_t valDisconnReq { static_cast<int32_t>(0xD15C099) };
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
constexpr int16_t pulsePriority {SIGEV_PULSE_PRIO_INHERIT};

//! <b>Sequence of operations</b>
//!  - Creates a QNX channel by calling ChannelCreate_r() with flags set to
//!    _NTO_CHF_UNBLOCK and _NTO_CHF_PRIVATE.
//!  - Upon successful creation of channel, establishes the connection between
//!    process and channel by calling ConnectAttach_r().
//!  - Upon successful connection, calls LwSciIpcSetQnxPulseParam() to set the
//!    event pulse parameters.
//!  - If successful, calls LwSciIpcGetEndpointInfo() to query the IPC endpoint
//!    information.
//!  - Resets the endpoint by calling LwSciIpcResetEndpoint() if frame size is
//!    not 0U.
//!
//! \implements{19617612}
//!
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A12_1_5), "Bug 2989775")
IpcComm::IpcComm(LwSciIpcEndpoint const ipcHandle) noexcept :
    handle(ipcHandle),
    channelInfo{},
    initSuccess(false),
    qnxChannelId(0),
    ipcEventConnId(0),
    internalInterruptId(0),
    connEstablished(false),
    numUnsentSignals(0U),
    numWritePending(0U)
{
    constexpr int32_t milwalid {0};
    qnxChannelId = ChannelCreate_r(_NTO_CHF_UNBLOCK | _NTO_CHF_PRIVATE);
    if (qnxChannelId >= milwalid) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        ipcEventConnId = ConnectAttach_r(0U,
                                         static_cast<pid_t>(0),
                                         qnxChannelId,
                                         static_cast<uint32_t>(_NTO_SIDE_CHANNEL),
                                         _NTO_COF_CLOEXEC);
        internalInterruptId = ConnectAttach_r(0U,
                                          static_cast<pid_t>(0),
                                          qnxChannelId,
                                          static_cast<uint32_t>(_NTO_SIDE_CHANNEL) + ONE,
                                          _NTO_COF_CLOEXEC);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
        if ((ipcEventConnId >= milwalid) &&
            (internalInterruptId >= milwalid)) {
            // Connect coid to endpoint and set pulse parameters so that
            // lwscistream can receive peer notifications from LwSciIpc
            // library.
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            if (LwSciError_Success == LwSciIpcSetQnxPulseParam(handle,
                                         ipcEventConnId,
                                         pulsePriority,
                                         ipcCode,
                                         nullptr)) {
                // Query endpoint info
                if (LwSciError_Success == LwSciIpcGetEndpointInfo(handle, &channelInfo)) {
                    // ipc / ivc frame is big enough to contain a frame
                    if (channelInfo.frame_size > 0U) {
                        // mark successfully created
                        initSuccess = true;
                    }
                }
            }
            LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
        }
    }
}

// Destructor detaches connection for receiving LwSciIpc signals
// (pulses) and for receiving internal signals (pulses).
IpcComm::~IpcComm(void) noexcept
{
    static_cast<void>(ConnectDetach(ipcEventConnId));
    static_cast<void>(ConnectDetach(internalInterruptId));
    static_cast<void>(ChannelDestroy(qnxChannelId));
    LwSciIpcCloseEndpoint(handle);
}

// Accessor to get IPC channel frame size from LwSciIpcEndpointInfo.
uint32_t IpcComm::getFrameSize(void) const noexcept
{
    return channelInfo.frame_size;
}

//! <b>Sequence of operations</b>
//!  - The function does the following in a loop:
//!  - The function gets LwSciIpc events by calling LwSciIpcGetEvent().
//!    If event LW_SCI_IPC_EVENT_READ is available, member readReady of
//!    IpcQueryFlags will be set. If event LW_SCI_IPC_EVENT_WRITE is available and
//!    there is pending write request, member writeReady of IpcQueryFlags will be
//!    set. If either readReady or writeReady is set, the function returns
//!    IpcQueryFlags.
//!  - If neither readReady nor writeReady is set MsgReceivePulse_r() will be
//!    called to wait for pulses.
//!  - If the pulse waking up the calling thread indicates a LwSciIpc event,
//!    the function calls LwSciIpcGetEvent() again.
//!  - If the pulse indicates a write request, the function increments
//!    numWritePending and calls LwSciIpcGetEvent() again.
//!  - If the pulse indicates a disconnect request, the function aborts the
//!    loop if numWritePending is zero, otherwise it calls LwSciIpcGetEvent()
//!    again.
//!
//! \implements{19652136}
IpcQueryFlags IpcComm::waitForEvent(void) noexcept
{
    IpcQueryFlags returnFlags {false, false, LwSciError_Success};
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A0_1_1), "Bug 2738197")
    bool disconnectDetected { false };

    // Loop until one of:
    //   An error oclwrs
    //   There is a message to receive
    //   There is a message to send, and room for it
    //   Disconnect is triggered and there are no more messages to send
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    do {
        // Check IPC for read or write events
        uint32_t receivedEvents{ 0U };
        returnFlags.err = LwSciIpcGetEvent(handle, &receivedEvents);
        if (LwSciError_Success != returnFlags.err) {
            break;
        }
        if (0U != (receivedEvents & LW_SCI_IPC_EVENT_READ)) {
            returnFlags.readReady = true;
        }
        if ((0U != (receivedEvents & LW_SCI_IPC_EVENT_WRITE)) &&
            (numWritePending > 0U)) {
            returnFlags.writeReady = true;
        }

        // If a read or write message is ready, break out and return
        if (returnFlags.readReady || returnFlags.writeReady) {
            break;
        }

        // Wait for message pulse of IPC event or write request
        struct _pulse eventPulse;
        returnFlags.err = LwSciIpcErrnoToLwSciErr(
                            MsgReceivePulse_r(qnxChannelId,
                                                &eventPulse,
                                                sizeof(eventPulse),
                                                nullptr));

        if (LwSciError_Success != returnFlags.err) {
            break;
        }

        if (interruptCode == eventPulse.code) {
            // internal write requested
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
            if (eventPulse.value.sival_int == valWriteReq) {
                if (UINT32_MAX == numWritePending) {
                    returnFlags.err = LwSciError_InsufficientResource;
                    break;
                }
                ++numWritePending;
            }

            // disconnect request
            if (!disconnectDetected &&
                (eventPulse.value.sival_int == valDisconnReq)) {
                disconnectDetected = true;
            }
            LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
        }
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A6_5_3), "LwSciStream-ADV-AUTOSARC++14-004")
    } while (!disconnectDetected || (numWritePending > 0U));
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A0_1_1))
    return returnFlags;
}

//! <b>Sequence of operations</b>
//!  - This function fetches LwSciIpc events by calling LwSciIpcGetEvent().
//!  - If event LW_SCI_IPC_EVENT_CONN_EST_ALL is available, connEstablished
//!  will be set, and the function will return. Otherwise it calls
//!  MsgReceivePulse_r() to wait for pulses.
//!  - After waking up by an arrival pulse, the function calls
//!  LwSciIpcGetEvent() again.
//!  - The loop repeats until event LW_SCI_IPC_EVENT_CONN_EST_ALL is available,
//!  or an error oclwrs.
//!
//! \implements{19652145}
LwSciError IpcComm::waitForConnection(void) noexcept
{
    LwSciError err;
    // event pulse
    struct _pulse eventPulse;
    eventPulse.code = ipcCode;

    // Per LwSciIpc specification, a LwSciIpcGetEvent or LwSciIpcWrite
    // is required to trigger state change on the other end of the connection.
    do {
        if (ipcCode == eventPulse.code) {
            uint32_t receivedEvents;
            err = LwSciIpcGetEvent(handle, &receivedEvents);
            if (LwSciError_Success != err) {
                break;
            }
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            if (0U != (receivedEvents & LW_SCI_IPC_EVENT_CONN_EST_ALL)) {
                connEstablished.store(true);
                break;
            }
            LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
        }
        // Wait for the connection signal
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        err = LwSciIpcErrnoToLwSciErr(
                MsgReceivePulse_r(qnxChannelId,
                                    &eventPulse,
                                    sizeof(eventPulse),
                                    nullptr));
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A6_5_3), "LwSciStream-ADV-AUTOSARC++14-004")
    } while (LwSciError_Success == err);
    return err;
}

//! <b>Sequence of operations</b>
//!  - Signals write request by calling MsgSendPulse_r(). A pulse will be sent
//!   through the QNX Connection identified by internalInterruptId, notifying
//!   the thread blocked on the call to IpcComm::waitForEvent().
//!
//! \implements{19652163}
LwSciError IpcComm::signalWrite(void) noexcept
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A0_1_1), "Bug 2738197")
    LwSciError err {LwSciError_Success};
    if (connEstablished.load()) {
        // signal write request if IPC connection has been
        // established.
        err = LwSciIpcErrnoToLwSciErr(MsgSendPulse_r(
                                        internalInterruptId,
                                        pulsePriority,
                                        interruptCode,
                                        valWriteReq));
    } else {
        // IPC connection not is not ready yet,
        // queue up the write signal.
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
        if (numUnsentSignals < UINT32_MAX) {
            numUnsentSignals++;
        }
    }
    return err;
}

//! <b>Sequence of operations</b>
//!  - Sends a signal to write the message over the IPC channel, by calling
//!    MsgSendPulse_r(). Pulses will be sent through the QNX Connection
//!    identified by internalInterruptId, notifying the thread blocked on the
//!    call to IpcComm::waitForEvent(). A pulse means a write request queued up
//!    before IPC connection is established.
//!
//! \implements{19652184}
LwSciError IpcComm::flushWriteSignals(void) noexcept
{
    // Flush the write signals that are queued up before IPC
    // connection is established.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    while (numUnsentSignals > 0U) {
        LwSciError const err { LwSciIpcErrnoToLwSciErr(MsgSendPulse_r(
                                        internalInterruptId,
                                        pulsePriority,
                                        interruptCode,
                                        valWriteReq)) };
        if (LwSciError_Success != err) {
            return err;
        }
        numUnsentSignals--;
    }
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//!  - Sends a disconnect request signal by calling MsgSendPulse_r(). A pulse
//!   will be sent through the QNX Connection identified by internalInterruptId,
//!   notifying the thread blocked on the call to IpcComm::waitForEvent().
//!
//! \implements{19652160}
LwSciError IpcComm::signalDisconnect(void) const noexcept
{
    return LwSciIpcErrnoToLwSciErr(
                MsgSendPulse_r(
                    internalInterruptId,
                    pulsePriority,
                    interruptCode,
                    valDisconnReq));
}

//! <b>Sequence of operations</b>
//!   - Calls IpcBuffer::changeMode to put buffer in Send mode.
//!   - Calls IpcBuffer::sendInfoGet() to retrieve pointer to and size of
//!     packed data waiting to be sent.
//!   - Calls LwSciIpcWrite() to send a message to the other side of the IPC
//!   channel.
//!   - Calls IpcBuffer::sendSizeAdd() to inform buffer of amount of data sent.
//!   - Calls IpcBuffer::sendDone() to check whether full message is sent. If yes,
//!     decrements numWritePending.
//!
//! \implements{19652169}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
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

    // If succesful and no more data to send, decrement numWritePending
    if ((LwSciError_Success == err) && buffer.sendDone()) {
        if (numWritePending > 0U) {
            numWritePending--;
        } else {
            err = LwSciError_StreamInternalError;
        }
    }

    return err;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! <b>Sequence of operations</b>
//!  - Calls IpcBuffer::changeMode to put buffer in Recv mode.
//!  - Calls IpcBuffer::recvInfoGet to retrieve pointer to and size of
//!    array into which data can be received.
//!  - Calls LwSciIpcRead() to read a single frame.
//!  - Calls IpcBuffer::recvSizeAdd to inform buffer of amount of data read.
//!
//! \implements{19652175}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
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
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))

} // namespace LwSciStream
