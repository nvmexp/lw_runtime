//! \file
//! \brief LwSciStream C2C source block reception.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
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
#include "c2csrc.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Calls appropriate message handler based on type:
//! -- C2CPacketCreate: c2cRecvPacketCreate().
//! -- C2CPacketsComplete: No action required.
//! -- C2CPacketDelete: c2cRecvPacketDelete().
//! -- SignalAttr: c2cRecvSyncSignalAttr().
//! -- C2CWaiterAttr: c2cRecvSyncWaiterAttr().
//! -- WaiterObj: c2cRecvSyncWaiterObj().
//! -- C2CPayload:: c2cRecvPayload().
//! - If none of the above, calls IpcDst::recvMessage().
LwSciError
C2CSrc::recvMessage(
    IpcBuffer& recvBuf,
    IpcMsg const msgType) noexcept
{
    switch (msgType) {

    // Handle C2C packet creation
    case IpcMsg::C2CPacketCreate:
        return c2cRecvPacketCreate(recvBuf);

    // Handle C2C packet list completion
    case IpcMsg::C2CPacketsComplete:
        // Lwrrently no action required
        return LwSciError_Success;

    // Handle C2C packet deletion
    case IpcMsg::C2CPacketDelete:
        return c2cRecvPacketDelete(recvBuf);

    // Handle C2CDst's read signal attributes
    case IpcMsg::SignalAttr:
        return c2cRecvSyncSignalAttr(recvBuf);

    // Handle C2CDst's waiter attribute flow
    case IpcMsg::C2CWaiterAttr:
        return c2cRecvSyncWaiterAttr(recvBuf);

    // Handle C2CDst's C2C copy waiter object (C2CSrc's signaller object)
    case IpcMsg::WaiterObj:
        return c2cRecvSyncWaiterObj(recvBuf);

    // Handle C2C payload for reuse
    case IpcMsg::C2CPayload:
        return c2cRecvPayload(recvBuf);

    default:
        break;

    }

    // Fall back to common IPC/C2C message processing
    return IpcSrc::recvMessage(recvBuf, msgType);
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::unpackVal() to unpack the new packet handle.
//! - Call c2cPktCreate() to create a new packet, unpacking the data.
//! - Call c2cPktFind() to retrieve the newly created packet.
//! - Call C2CPacket::registerC2CBufTargetHandles() to register the packet's
//!   buffers with the C2C channel.
//! - Call enqueueIpcWrite() to inform the dispatch thread the packet status is
//!   ready to send back downstream.
LwSciError
C2CSrc::c2cRecvPacketCreate(
    IpcBuffer &recvBuf) noexcept
{
    LwSciStreamPacket pktHandle;
    LwSciError err { recvBuf.unpackVal(pktHandle) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Create a record of the C2C packet, queuing status event
    err = c2cPktCreate(pktHandle, recvBuf);
    if (LwSciError_Success != err) {
        return err;
    }

    // Retrieve the newly created packet
    C2CPacketPtr const pkt{ c2cPktFind(pktHandle) };
    if (nullptr == pkt) {
        return LwSciError_StreamInternalError;
    }

    // Register the buffer with C2C service.
    err = pkt->registerC2CBufTargetHandles(c2cChannel);
    if (LwSciError_Success != err) {
        return err;
    }

    // Signal status event ready to send downstream
    err = enqueueIpcWrite();
    if (LwSciError_Success != err) {
        return err;
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::unpackVal() to unpack the deleted packet's handle.
//! - Call c2cPktRemove() to delete the packet.
LwSciError
C2CSrc::c2cRecvPacketDelete(
    IpcBuffer &recvBuf) noexcept
{
    LwSciStreamPacket pktHandle;
    LwSciError const err { recvBuf.unpackVal(pktHandle) };
    if (LwSciError_Success != err) {
        return err;
    }

    // TODO: Need to deal with removal from queue and reducing count of
    //   available packets. Producer has related issues.

    c2cPktRemove(pktHandle);

    return err;
}

//! <b>Sequence of operations</b>
//! - Call Wrapper::viewVal() to make sure not already done.
//! - Call ipcBufferUnpack() to unpack the sync attributes from recvBuffer.
//! - Call Wrapper::getErr() to make sure import worked.
LwSciError
C2CSrc::c2cRecvSyncSignalAttr(
    IpcBuffer& recvBuf) noexcept
{
    // Make sure not already done
    if (nullptr != readDoneSignalAttr.viewVal()) {
        return LwSciError_AlreadyDone;
    }

    // Unpack the object
    LwSciWrap::SyncObj syncObj {};
    LwSciError err { ipcBufferUnpack(recvBuf, readDoneSignalAttr) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Make sure import worked
    return readDoneSignalAttr.getErr();
}

//! <b>Sequence of operations</b>
//! - Call Block::elementCountGet() to retrieve count for number of syncs.
//! - Create a new Waiters instance and call Waiters::sizeInit() and
//!   Waiters::entryFill() to populate it with copies of the waiter attributes.
//! - Call the dstRecvSyncWaiter() interface of the source block to send the
//!   waiter attributes upstream.
LwSciError
C2CSrc::c2cRecvSyncWaiterAttr(
    IpcBuffer& recvBuf) noexcept
{
    // TODO: Check if already done

    // No data to unpack
    static_cast<void>(recvBuf);

    // Fill a temporary Waiters object with the block's waiting requirements
    Waiters tmpWaiter { FillMode::User, true };
    LwSciError err { tmpWaiter.sizeInit(elementCountGet()) };
    if (LwSciError_Success == err) {
        err = tmpWaiter.entryFill(engineDoneWaiterAttr);
    }
    if (LwSciError_Success != err) {
        return err;
    }

    // Send the attributes upstream
    getSrc().dstRecvSyncWaiter(tmpWaiter);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call ipcBufferUnpack() to unpack the sync object from recvBuffer.
//! - Call LwSciSyncRegisterSignalObjIndirectChannelC2c() to register
//!   the sync object so the C2C-engine can signal it.
LwSciError
C2CSrc::c2cRecvSyncWaiterObj(
    IpcBuffer& recvBuf) noexcept
{
    // Make sure not already done
    if (nullptr != c2cSignalConsHandle) {
        return LwSciError_AlreadyDone;
    }

    // Unpack the object
    LwSciWrap::SyncObj syncObj {};
    LwSciError err { ipcBufferUnpack(recvBuf, syncObj, c2cSignalAttr) };
    if (LwSciError_Success != err) {
        return err;
    }

    if (LwSciError_Success != syncObj.getErr()) {
        return err;
    }

    // Register the sync object with C2C
    return LwSciSyncRegisterSignalObjIndirectChannelC2c(c2cChannel,
                                                        syncObj.viewVal(),
                                                        &c2cSignalConsHandle);
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::unpackVal() to unpack the packet handle.
//! - Call Block::blkMutexLock() to prevent interference while manipulating
//!   the packet map and queue:
//! -- Call c2cPktFind() to look up the packet.
//! -- Call C2CPacket::Payload::enqueue() to enqueue it for reuse.
//! - Call ipcEnqueueWrite() to alert the dispatch thread if it is waiting
//!   to send a packet.
LwSciError
C2CSrc::c2cRecvPayload(IpcBuffer& recvBuf) noexcept
{
    // Unpack the handle
    LwSciStreamPacket pktHandle;
    LwSciError err{ recvBuf.unpackVal(pktHandle) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Scoped lock while manipulating queue and packet
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
        Lock const blockLock { blkMutexLock() };

        // Look up the packet in the map for C2C packets
        C2CPacketPtr const pkt { c2cPktFind(pktHandle, true) };
        if (nullptr != pkt) {
            // Add the packet to the queue of available C2C packets
            c2cPayloadQueue.enqueue(pkt);
            ++numC2CPktsAvailable;
        } else {
            return LwSciError_StreamBadPacket;
        }

        if (0U == numPayloadsPendingWriteSignal) {
            if (UINT32_MAX == numC2CPktsForWriteSignal) {
                return LwSciError_InsufficientResource;
            }
            // If there is no payload pending for IPC write, increment the
            // number of available C2C packets for adding payloads to the
            // IPC write queue.
            ++numC2CPktsForWriteSignal;
            return LwSciError_Success;
        }
        --numPayloadsPendingWriteSignal;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // Reaching this point means there is payload pending to be signal for
    // IPC write.
    // Trigger the notifier that will wake the IPC thread again,
    // so it will recheck whether is needs to send a packet downstream.
    return enqueueIpcWrite();
}

} // namespace LwSciStream
