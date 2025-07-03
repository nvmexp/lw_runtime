//! \file
//! \brief LwSciStream C2C destination block reception.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
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
#include "c2cdst.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Calls appropriate message handler based on type:
//! -- C2CPacketStatus: c2cRecvPacketStatus().
//! -- SignalAttr: c2cRecvSyncSignalAttr().
//! -- C2CWaiterAttr: c2cRecvSyncWaiterAttr().
//! -- C2CPayload:: c2cRecvPayload().
//! - If none of the above, calls IpcDst::recvMessage().
LwSciError
C2CDst::recvMessage(
    IpcBuffer& recvBuf,
    IpcMsg const msgType) noexcept
{
    switch (msgType) {

    // Handle C2C packet status
    case IpcMsg::C2CPacketStatus:
        return c2cRecvPacketStatus(recvBuf);

    // Handle C2CSrc's copy signal attributes
    case IpcMsg::SignalAttr:
        return c2cRecvSyncSignalAttr(recvBuf);

    // Handle C2CDst's waiter attribute flow
    case IpcMsg::C2CWaiterAttr:
        return c2cRecvSyncWaiterAttr(recvBuf);

    // Handle C2C copy payload
    case IpcMsg::C2CPayload:
        return c2cRecvPayload(recvBuf);

    default:
        break;

    }

    // Fall back to common IPC/C2C message processing
    return IpcDst::recvMessage(recvBuf, msgType);
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::unpackVal() to unpack the packet handle.
//! - Call Block::pktFindByHandle to retrieve the corresponding local Packet.
//! - Call IpcBuffer::unpackVal() to unpack the status.
//! - Call Packet::statusProdSet() to set the status.
//! - Call srcRecvPacketStatus() on destination block to send status downstream
LwSciError
C2CDst::c2cRecvPacketStatus(IpcBuffer& recvBuf) noexcept
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

    // Unpack the status
    LwSciError status {};
    err = recvBuf.unpackVal(status);
    if (LwSciError_Success != err) {
        return err;
    }

    // Set the packet status
    //   The handle is used as the cookie, to facilitate deletion.
    err = pkt->statusProdSet(status, pktHandle);
    if (LwSciError_Success != err) {
        return err;
    }

    // Send the status downstream
    getDst().srcRecvPacketStatus(*pkt);

    // Free status resources
    pkt->statusClear();

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Wrapper::viewVal() to make sure not already done.
//! - Call ipcBufferUnpack() to unpack the sync attributes from recvBuffer.
//! - Call Wrapper::getErr() to make sure import worked.
LwSciError
C2CDst::c2cRecvSyncSignalAttr(
    IpcBuffer& recvBuf) noexcept
{
    // Make sure not already done
    if (nullptr != copyDoneSignalAttr.viewVal()) {
        return LwSciError_AlreadyDone;
    }

    // Unpack the object
    LwSciWrap::SyncObj syncObj {};
    LwSciError err { ipcBufferUnpack(recvBuf, copyDoneSignalAttr) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Make sure import worked
    return copyDoneSignalAttr.getErr();
}

//! <b>Sequence of operations</b>
//! - Call Block::elementCountGet() to retrieve count for number of syncs.
//! - Create a new Waiters instance and call Waiters::sizeInit() and
//!   Waiters::entryFill() to populate it with copies of the waiter attributes.
//! - Call the dstRecvSyncWaiter() interface of the source block to send the
//!   waiter attributes upstream.
LwSciError
C2CDst::c2cRecvSyncWaiterAttr(
    IpcBuffer& recvBuf) noexcept
{
    // TODO: Check if already done

    // No data to unpack
    static_cast<void>(recvBuf);

    // Fill a temporary Waiters object with the block's waiting requirements
    Waiters tmpWaiter { FillMode::User, false };
    LwSciError err { tmpWaiter.sizeInit(elementCountGet()) };
    if (LwSciError_Success == err) {
        err = tmpWaiter.entryFill(engineDoneWaiterAttr);
    }
    if (LwSciError_Success != err) {
        return err;
    }

    // Send the attributes downstream
    getDst().srcRecvSyncWaiter(tmpWaiter);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::unpackVal() to unpack the handle.
//! - Call Block::pktFindByHandle() to find the corresponding local packet.
//! - Call Packet::locationUpdate() to validate and update the location.
//! - Call Packet::fenceConsReset() to reset the packet's consumer fences.
//! - Call ipcBufferUnpack() to unpack the C2C copy fence.
//! - If the consumer supports synchronization, call Packet::fenceProdFill()
//!   to fill the packet's producer fences with copies of the C2C copy fence.
//! - Otherwise, call LwSciSyncFenceWait() to do a CPU wait for the C2C
//!   copy fence. And call Packet::fenceProdDone() to mark the producer fences
//!   setup done.
//! - Call srcRecvPayload() interface of destination blocks to send the payload
//!   downstream.
//! - Call Packet::fenceProdReset() to reset the producer fences
LwSciError
C2CDst::c2cRecvPayload(IpcBuffer& recvBuf) noexcept
{
    // Unpack the packet handle
    LwSciStreamPacket handle;
    LwSciError err { recvBuf.unpackVal(handle) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Find the local packet for this handle
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    PacketPtr const pkt { pktFindByHandle(handle) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // Make sure packet was upstream and shift it downstream
    if (!pkt->locationUpdate(Packet::Location::Upstream,
                             Packet::Location::Downstream)) {
        return LwSciError_StreamPacketInaccessible;
    }

    // Clear packet's consumer fences
    pkt->fenceConsReset();

    // Unpack the fence that indicates when C2C copy is done
    LwSciWrap::SyncFence c2cFence {};
    err = ipcBufferUnpack(recvBuf, c2cFence, c2cCopySyncObj);
    if (LwSciError_Success != err) {
        return err;
    }

    // If consumer does not support synchronization for any element,
    //   need to perform a CPU wait here.
    // TODO: Offload this using LwSciEvent.
    if (cpuWaitAfterCopy) {
        err = c2cFence.getErr();
        if (LwSciError_Success == err) {
            // Take lock and wait for fences
            {
                Lock const blockLock{ blkMutexLock() };
                err = LwSciSyncFenceWait(&c2cFence.viewVal(),
                                         waitContext,
                                         INFINITE_TIMEOUT);
            }
        }
        if (LwSciError_Success != err) {
            return err;
        }

        // Mark fence setup done
        pkt->fenceProdDone();
    }

    // Otherwise, fill packet with this fence
    else {
        err = pkt->fenceProdFill(c2cFence);
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // Send the payload downstream
    getDst().srcRecvPayload(*pkt);

    return LwSciError_Success;
}

} // namespace LwSciStream
