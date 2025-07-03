//! \file
//! \brief LwSciStream IPC destination block reception.
//!
//! \copyright
//! Copyright (c) 2020-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <array>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <cmath>
#include <vector>
#include <functional>
#include <utility>
#include <cstddef>
#include <iostream>
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "covanalysis.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "lwscistream_common.h"
#include "lwscistream_types.h"
#include "lwscicommon_os.h"
#include "sciwrap.h"
#include "safeconnection.h"
#include "packet.h"
#include "trackarray.h"
#include "ipccomm_common.h"
#include "ipcbuffer.h"
#include "ipcbuffer_sciwrap.h"
#include "ipccomm.h"
#include "ipcdst.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Calls appropriate message handler based on type:
//! -- AllocatedElements: recvAllocatedElements().
//! -- IPCPacketCreate: ipcRecvPacketCreate().
//! -- IPCPacketsComplete: ipcRecvPacketsComplete().
//! -- IPCPacketDelete: ipcRecvPacketDelete().
//! -- WaiterAttr: ipcRecvSyncWaiterAttr().
//! -- SignalObj: ipcRecvSyncSignalObj().
//! -- Runtime: recvRuntimeBegin().
//! -- IPCPayload: ipcRecvPayload().
//! -- Disconnect: recvDisconnect().
LwSciError
IpcDst::recvMessage(
    IpcBuffer& recvBuf,
    IpcMsg const msgType) noexcept
{
    switch (msgType) {

    case IpcMsg::AllocatedElements:
        return recvAllocatedElements(recvBuf);

    case IpcMsg::IPCPacketCreate:
        assert(!isC2CBlock);
        return ipcRecvPacketCreate(recvBuf);

    case IpcMsg::IPCPacketsComplete:
        assert(!isC2CBlock);
        return ipcRecvPacketsComplete(recvBuf);

    case IpcMsg::IPCPacketDelete:
        assert(!isC2CBlock);
        return ipcRecvPacketDelete(recvBuf);

    case IpcMsg::IPCWaiterAttr:
        return ipcRecvSyncWaiterAttr(recvBuf);

    case IpcMsg::SignalObj:
        assert(!isC2CBlock);
        return ipcRecvSyncSignalObj(recvBuf);

    case IpcMsg::Runtime:
        return recvRuntimeBegin(recvBuf);

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
IpcDst::recvHeader(
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
//! - Call Block::prodInfoSet() to save the incoming info.
//! - Call Block::prodInfoFlow() to send the info downstream.
LwSciError
IpcDst::ipcRecvConnect(
    IpcBuffer& recvBuf) noexcept
{
    // Unpack the vector
    EndInfoVector info {};
    LwSciError err { ipcBufferUnpack(recvBuf, info) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Save the incoming info
    err = prodInfoSet(info);
    if (LwSciError_Success != err) {
        return err;
    }

    // Mark receipt of connection completion done
    connectStartDone = true;

    // Send downstream
    prodInfoFlow();

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Elements::dataUnpack() to unpack and save the allocated element
//!   information.
//! - Call Elements::sizePeek() to retrieve the number of elements.
//! - Call Block::elementCountSet() to save the number of elements.
//! - Call Waiters::sizeInit() to initialize the sync attribute vectors.
//! - Call Elements::dataSend() to ilwoke srcRecvAllocatedElements() on the
//!   downstream connection to pass on the element list.
LwSciError
IpcDst::recvAllocatedElements(
    IpcBuffer& recvBuf) noexcept
{
    // Unpack the elements
    LwSciError err { allocatedElements.dataUnpack(recvBuf) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Retrieve and save number of elements
    size_t const elemCount { allocatedElements.sizePeek() };
    elementCountSet(elemCount);

    // Initialize waiter sync attribute trackers if this is regular IPC
    if (!isC2CBlock) {
        err = ipcSyncWaiter.sizeInit(elemCount);
        if (LwSciError_Success != err) {
            return err;
        }
        err = endSyncWaiter.sizeInit(elemCount);
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // Define send operation
    Elements::Action const sendAction {
        [this](Elements const& elements) noexcept -> void {
            LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
            getDst().srcRecvAllocatedElements(elements);
        }
    };

    // Send downstream, but do not clear the data at this time
    return allocatedElements.dataSend(sendAction, false);
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::unpackVal() to unpack the LwSciStreamPacket handle.
//! - Call Block::pktCreate() to create a new Packet instance and insert
//!   itin the map.
//! - Call Block::pktFindByHandle() to retrieve the Packet.
//! - Call Packet::defineUnpack() to unpack the rest of the data from
//!   the IpcBuffer.
//! - Call srcRecvPacketCreate() interface of the destination block to pass
//!   the Packet downstream.
LwSciError
IpcDst::ipcRecvPacketCreate(
    IpcBuffer& recvBuf) noexcept
{
    // Unpack the handle from IPC
    LwSciStreamPacket pktHandle;
    LwSciError err { recvBuf.unpackVal(pktHandle) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Create the new packet and insert in the map
    err = pktCreate(pktHandle);
    if (LwSciError_Success != err) {
        return err;
    }

    // Retrieve the packet
    PacketPtr const pkt { pktFindByHandle(pktHandle) };
    if (nullptr == pkt) {
        // Something serious went wrong above if we end up here.
        return LwSciError_StreamBadPacket;
    }

    // Unpack the definition into the packet
    err = pkt->defineUnpack(recvBuf, allocatedElements);
    if (LwSciError_Success != err) {
        return err;
    }

    // Send the packet downstream
    getDst().srcRecvPacketCreate(*pkt);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call phasePacketsDoneSet() to advance setup phase.
//! - Call srcRecvPacketsComplete() interface of destination block to send
//!   the notice downstream.
LwSciError
IpcDst::ipcRecvPacketsComplete(
    IpcBuffer& recvBuf) noexcept
{
    // No additional data needed
    static_cast<void>(recvBuf);

    // Advance setup phase
    // TODO: May need special handling for C2C
    phasePacketsDoneSet();

    // Send downstream
    getDst().srcRecvPacketsComplete();

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::unpackVal() to unpack the LwSciStreamPacket handle.
//! - Call Block::pktRemove() to remove the Packet from the map.
//! - Call srcRecvPacketDelete() interface of destination block to send
//!   the notice downstream.
LwSciError
IpcDst::ipcRecvPacketDelete(
    IpcBuffer& recvBuf) noexcept
{
    // Unpack the handle
    LwSciStreamPacket pktHandle;
    LwSciError const err { recvBuf.unpackVal(pktHandle) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Look up the packet
    PacketPtr const pkt { pktFindByHandle(pktHandle) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }

    // Delete message should only be received when packet is upstream
    if (!pkt->locationCheck(Packet::Location::Upstream)) {
        return LwSciError_StreamPacketInaccessible;
    }

    // Delete packet
    pktRemove(pktHandle);

    // Send downstream
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    getDst().srcRecvPacketDelete(pktHandle);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Waiters::unpack() to unpack the waiter information into the
//!   the producer info tracker.
//! - Call srcRecvSyncWaiter() on the downstream connection to pass on the
//!   information.
LwSciError
IpcDst::ipcRecvSyncWaiterAttr(
    IpcBuffer& recvBuf) noexcept
{
    // Unpack the message
    LwSciError const err { ipcSyncWaiter.unpack(recvBuf) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Send the info downstream
    getDst().srcRecvSyncWaiter(ipcSyncWaiter);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Signals::unpack() to unpack the signal information into the
//!   the consumer info tracker.
//! - Call srcRecvSyncSignal() on the downstream connection to pass on the
//!   information.
//! - Call phaseProdSyncDoneSet() to advance setup phase.
LwSciError
IpcDst::ipcRecvSyncSignalObj(
    IpcBuffer& recvBuf) noexcept
{
    // Unpack the message
    LwSciError const err { ipcSyncSignal.unpack(recvBuf) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Send the info upstream
    getDst().srcRecvSyncSignal(ipcSyncSignal);

    // Advance setup phase
    phaseProdSyncDoneSet();

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call phaseSrcDoneSet() to indicate the upstream block has changed phase
//!   and trigger the downstream flow.
LwSciError
IpcDst::recvRuntimeBegin(
    IpcBuffer& recvBuf) noexcept
{
    // No additional data needed
    static_cast<void>(recvBuf);

    // Chnage phase and transmit downstream
    phaseSrcDoneSet(0U);
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::unpackVal() to unpacks the LwSciStreamPacket handle.
//! - Call Block::pktFindByHandle() to find the packet for this handle.
//! - Call Packet::locationUpdate() to update the location from upstream
//!   to downstream.
//! - Call Packet::fenceProdUnpack() to unpack the producer fences, and
//!   Packet::fenceConsReset() to reset the consumer fences.
//! - Call the srcRecvPayload() interface of the destination block to send
//!   the payload downstream.
LwSciError
IpcDst::ipcRecvPayload(
    IpcBuffer& recvBuf) noexcept
{
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

    // Make sure packet was upstream and shift it downstream
    if (!pkt->locationUpdate(Packet::Location::Upstream,
                             Packet::Location::Downstream)) {
        return LwSciError_StreamPacketInaccessible;
    }

    // Unpack the producer fences and reset the consumer fences
    err = pkt->fenceProdUnpack(recvBuf, ipcSyncSignal);
    if (LwSciError_Success != err) {
        return err;
    }
    pkt->fenceConsReset();

    // Send the payload downstream
    getDst().srcRecvPayload(*pkt);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call disconnectDst() to send the disconnection downstream.
//! - Call disconnectEvent() to queue a disconnection event.
//! - Call destroyIOLoop() to shut down message handling.
LwSciError
IpcDst::recvDisconnect(
    IpcBuffer& recvBuf) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // No additional data needed
    static_cast<void>(recvBuf);

    // signal disconnect downstream
    disconnectDst();

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
//! \implements{19840539}
LwSciError IpcDst::processReadMsg(void) noexcept
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
IpcDst::recvEnabled(
    bool const connect) const noexcept
{
    // No messages can be received by Dst block before it has connection
    //   readiness
    if (!connectReadyDone) {
        return false;
    }
    // No message can be received by Dst block before connection is complete
    //   except the connection message itself
    else if (!connectStartDone) {
        return connect;
    }
    // After both are done, any message is allowed
    else {
        return true;
    }
}

} // namespace LwSciStream
