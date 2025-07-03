//! \file
//! \brief LwSciStream IPC source block reception.
//!
//! \copyright
//! Copyright (c) 2020-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.
#include <cstddef>
#include <iostream>
#include <array>
#include <mutex>
#include <cstdint>
#include <array>
#include <utility>
#include <vector>
#include <functional>
#include <atomic>
#include <unordered_map>
#include <memory>
#include <cmath>
#include "covanalysis.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "lwscistream_common.h"
#include "sciwrap.h"
#include "lwscistream_types.h"
#include "lwscicommon_os.h"
#include "safeconnection.h"
#include "trackarray.h"
#include "packet.h"
#include "ipccomm_common.h"
#include "ipccomm.h"
#include "ipcbuffer.h"
#include "ipcbuffer_sciwrap.h"
#include "ipcsrc.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Calls appropriate message handler based on type:
//! -- SupportedElements: recvSupportedElements().
//! -- IPCPacketStatus: ipcRecvPacketStatus().
//! -- WaiterAttr: ipcRecvSyncWaiterAttr().
//! -- SignalObj: ipcRecvSyncSignalObj().
//! -- Runtime: recvRuntimeReady().
//! -- IPCPayload: ipcRecvPayload().
//! -- Disconnect: recvDisconnect().
LwSciError
IpcSrc::recvMessage(
    IpcBuffer& recvBuf,
    IpcMsg const msgType) noexcept
{
    switch (msgType) {

    case IpcMsg::SupportedElements:
        return recvSupportedElements(recvBuf);

    case IpcMsg::IPCPacketStatus:
        assert(!isC2CBlock);
        return ipcRecvPacketStatus(recvBuf);

    case IpcMsg::IPCWaiterAttr:
        return ipcRecvSyncWaiterAttr(recvBuf);

    case IpcMsg::SignalObj:
        assert(!isC2CBlock);
        return ipcRecvSyncSignalObj(recvBuf);

    case IpcMsg::Runtime:
        return recvRuntimeReady(recvBuf);

    case IpcMsg::IPCPayload:
        assert(!isC2CBlock);
        return ipcRecvPayload(recvBuf);

    case IpcMsg::Disconnect:
        return recvDisconnect(recvBuf);

    default:
        break;
    }

    return LwSciError_IlwalidOperation;
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::changeMode() to initiate unpacking of recvBuffer.
//! - Call ipcBuffer:unpackVal() to unpack the message type from recvBuffer.
LwSciError
IpcSrc::recvHeader(
    IpcMsg& msgType) noexcept
{
    // Put sendBuffer into pack mode
    LwSciError err { recvBuffer.changeMode(IpcBuffer::UserMode::Unpack) };
    if (LwSciError_Success == err) {
        // Unack message type from buffer
        err = recvBuffer.unpackVal(msgType);
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call ipcBufferUnpack() on recvBuffer to unpack the info vector.
//! - Call Block::consInfoSet() to save the incoming info.
//! - Call Block::consInfoFlow() to send the info upstream.
LwSciError
IpcSrc::ipcRecvConnect(
    IpcBuffer& recvBuf) noexcept
{
    // Unpack the vector
    EndInfoVector info {};
    LwSciError err { ipcBufferUnpack(recvBuf, info) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Save the incoming info
    err = consInfoSet(info);
    if (LwSciError_Success != err) {
        return err;
    }

    // Mark receipt of connection readiness done
    connectReadyDone = true;

    // Send upstream
    consInfoFlow();

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Elements::dataUnpack() to unpack the supported element information
//!   into temporary storage.
//! - Call Elements::dataSend() to ilwoke dstRecvSupportedElements() on the
//!   upstream connection to pass on the element list.
LwSciError
IpcSrc::recvSupportedElements(
    IpcBuffer& recvBuf) noexcept
{
    // Unpack the elements into a temporary object
    Elements supportedElements {FillMode::IPC, FillMode::None};
    LwSciError const err { supportedElements.dataUnpack(recvBuf) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Define send operation
    Elements::Action const sendAction {
        [this](Elements const& elements) noexcept -> void {
            LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
            getSrc().dstRecvSupportedElements(elements);
        }
    };

    // Send upstream
    return supportedElements.dataSend(sendAction, false);
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::unpackVal() to unpack the packet handle.
//! - Call Block::pktFindByHandle to retrieve the corresponding local Packet.
//! - Call Packet::statusConsUnpack() to unpack the consumer status.
//! - Call dstRecvPacketStatus() on source block to send status upstream.
LwSciError
IpcSrc::ipcRecvPacketStatus(
    IpcBuffer& recvBuf) noexcept
{
    // Unpack the handle
    LwSciStreamPacket pktHandle;
    LwSciError err { recvBuf.unpackVal(pktHandle) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Retrieve the packet
    PacketPtr const pkt { pktFindByHandle(pktHandle) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }

    // Unpack the status into the packet
    err = pkt->statusConsUnpack(recvBuf);
    if (LwSciError_Success != err) {
        return err;
    }

    // Send the status upstream
    getSrc().dstRecvPacketStatus(*pkt);

    // Free status resources
    pkt->statusClear();

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Waiters::unpack() to unpack the waiter information into the
//!   the consumer info tracker.
//! - Call dstRecvSyncWaiter() on the upstream connection to pass on the
//!   information.
LwSciError
IpcSrc::ipcRecvSyncWaiterAttr(
    IpcBuffer& recvBuf) noexcept
{
    // Unpack the message
    LwSciError const err { ipcSyncWaiter.unpack(recvBuf) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Send the info upstream
    getSrc().dstRecvSyncWaiter(ipcSyncWaiter);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Signals::unpack() to unpack the signal information into the
//!   the consumer info tracker.
//! - Call dstRecvSyncSignal() on the upstream connection to pass on the
//!   information.
//! - Call phaseConsSyncDoneSet() to advance setup phase.
LwSciError
IpcSrc::ipcRecvSyncSignalObj(
    IpcBuffer& recvBuf) noexcept
{
    // Unpack the message
    LwSciError const err { ipcSyncSignal.unpack(recvBuf) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Send the info upstream
    getSrc().dstRecvSyncSignal(ipcSyncSignal);

    // Advance setup phase
    phaseConsSyncDoneSet();

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call phaseDstDoneSet() to indicate the downstream block is ready to
//!   change phase and potentially trigger the upstream flow.
LwSciError
IpcSrc::recvRuntimeReady(
    IpcBuffer& recvBuf) noexcept
{
    // No additional data needed
    static_cast<void>(recvBuf);

    // Chnage phase and transmit downstream
    phaseDstDoneSet(0U);
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::unpackVal() to unpacks the LwSciStreamPacket handle.
//! - Call Block::pktFindByHandle() to find the packet for this handle.
//! - Call Packet::locationUpdate() to update the location from dowstream
//!   to upstream.
//! - Call Packet::fenceConsUnpack() to unpack the consumer fences, and
//!   Packet::fenceProdReset() to reset the producer fences.
//! - Call the dstRecvPayload() interface of the source block to send
//!   the payload upstream.
LwSciError
IpcSrc::ipcRecvPayload(
    IpcBuffer& recvBuf) noexcept
{
    // Unpack the handle
    LwSciStreamPacket pktHandle;
    LwSciError err { recvBuf.unpackVal(pktHandle) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Find the packet for this handle
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    PacketPtr const pkt { pktFindByHandle(pktHandle) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // Make sure packet was downstream and shift it upstream
    if (!pkt->locationUpdate(Packet::Location::Downstream,
                             Packet::Location::Upstream)) {
        return LwSciError_StreamInternalError;
    }

    // Unpack the consumer fences and reset the producer fences
    err = pkt->fenceConsUnpack(recvBuf, ipcSyncSignal);
    if (LwSciError_Success != err) {
        return err;
    }
    pkt->fenceProdReset();

    // Send the payload upstream
    getSrc().dstRecvPayload(*pkt);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call disconnectSrc() to send the disconnection upstream.
//! - Call disconnectEvent() to queue a disconnection event.
//! - Call destroyIOLoop() to shut down message handling.
LwSciError
IpcSrc::recvDisconnect(
    IpcBuffer& recvBuf) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // No additional data needed
    static_cast<void>(recvBuf);

    // Signal disconnect upstream
    disconnectSrc();

    // Signal disconnect event if not already done
    disconnectEvent();

    // Signal message handling can shut down
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    destroyIOLoop();

    return LwSciError_Success;

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
}

//! <b>Sequence of operations</b>
//! - Call recvHeader() to initiate unpacking and retrieve the message type.
//! - If the message type is Connect, call ipcRecvConnect() and return.
//! - Call recvMessage to unpack message based on type.
//!
//! \implements{19839639}
LwSciError IpcSrc::processReadMsg(void) noexcept
{
    // Initiate unpacking of message
    IpcMsg msgType { IpcMsg::None };
    LwSciError err { recvHeader(msgType) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Shouldn't receive any messages before they're allowed, except disconnect
    //   which can happen at anytime
    if ((IpcMsg::Disconnect != msgType) &&
        !recvEnabled(IpcMsg::Connect == msgType)) {
        return LwSciError_NotInitialized;
    }

    // Connection is handled before any block-specific messages.
    if (IpcMsg::Connect == msgType) {
        return ipcRecvConnect(recvBuffer);
    }

    // Unpack any block-specific message
    return recvMessage(recvBuffer, msgType);
}

//! <b>Sequence of operations</b>
//! - Checks whether current state allows message reception.
bool
IpcSrc::recvEnabled(
    bool const connect) const noexcept
{
    // No messages can be received by Src block before connection readiness
    //   except the readiness message itself
    if (!connectReadyDone) {
        return connect;
    }
    // No other messages can be received by Src block before connection is
    //   complete
    else {
        return connectStartDone;
    }
}

} // namespace LwSciStream
