//! \file
//! \brief LwSciStream C2C destination block transmission.
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
#include "c2cdst.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Call Block::pktCreate() to create a new packet, copy the definition,
//!   and insert it in the map.
//! - Call enqueueIpcWrite() to signal available message.
void C2CDst::dstRecvPacketCreate(
    uint32_t const dstIndex,
    Packet const& origPacket) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation { };
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Create new packet, copy definition, and insert in map.
    //   This will automatically queue up creation event.
    LwSciError err { pktCreate(origPacket.handleGet(), &origPacket) };

    // Set any error and wake waiting threads
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(M0_1_2), "Bug 3003226")
    if (LwSciError_Success == err) {
        err = enqueueIpcWrite();
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(M0_1_2), "Bug 3003226")
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        setErrorEvent(err);
        LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    }
}

//! <b>Sequence of operations</b>
//! - Check whether packets already marked complete, and if not prepare event.
//! - Call enqueueIpcWrite() to signal message ready to send.
void C2CDst::dstRecvPacketsComplete(
    uint32_t const dstIndex) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation { };
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // If packets not already completed, mark done
    bool expected { false };
    if (!secondaryPacketExportDone.compare_exchange_strong(expected, true)) {
        setErrorEvent(LwSciError_AlreadyDone);
        return;
    }

    // Set flag to send message
    secondaryPacketExportEvent.store(true);

    // Wake any waiting threads
    LwSciError const err{ enqueueIpcWrite() };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
    }
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindByHandle() to retrieve the Packet instance
//!   for @a handle.
//! - Call Packet::locationCheck() to ensure the Packet is downstream.
//! - Call Packet::deleteSet() to mark the packet for deletion.
//! - Call enqueueIpcWrite() to signal message ready to send.
void C2CDst::dstRecvPacketDelete(
    uint32_t const dstIndex,
    LwSciStreamPacket const handle) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation{ };
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Find the packet for this handle
    PacketPtr const pkt { pktFindByHandle(handle) };
    if (nullptr == pkt) {
        setErrorEvent(LwSciError_StreamBadPacket);
        return;
    }

    // Should only receive deletion message when the packet is downstream
    if (!pkt->locationCheck(Packet::Location::Downstream)) {
        setErrorEvent(LwSciError_StreamPacketInaccessible);
        return;
    }

    // Mark the packet for deletion
    if (!pkt->deleteSet()) {
        setErrorEvent(LwSciError_StreamPacketDeleted);
        return;
    }

    // Wake any waiting threads
    LwSciError const err { enqueueIpcWrite() };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Block::elementCountGet() to retrieve count for number of syncs.
//! - For all elements, call Waiters::attrPeek() and Wrapper::viewVal() to
//!   retrieve the attributes lists from the consumer and combine into a
//!   vector.
//! - If any of the attributes are NULL, set the flag indicating that we
//!   need to do CPU waits after copies, clear out the vector and instead
//!   call Wrapper::viewVal() to retrieve the CPU waiter attributes and
//!   insert it in the vector.
//! - Call Wrapper::viewVal() to retrieve the copy done signaller attribute
//!   list and add it to the vector.
//! - Call LwSciSyncAttrListReconcile() to reconcile the attributes.
//! - Call enqueueIpcWrite to inform dispatch thread of attribute message.
void C2CDst::dstRecvSyncWaiter(
    uint32_t const dstIndex,
    Waiters const& syncWaiter) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation { };
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Check whether this has already been done
    if (nullptr != copyDoneAttr.viewVal()) {
        setErrorEvent(LwSciError_AlreadyDone);
        return;
    }

    // Retrieve element count
    size_t const elemCount { elementCountGet() };

    // Construct a list of all necessary attributes for a sync object
    //   to signal from C2CSrc to consumers that C2C copy is done
    std::vector<LwSciSyncAttrList> unreconciledList;
    try {
        // Try to combine all of the consumer's waiter requirements,
        //   aborting if any are NULL, requiring CPU waiting
        for (size_t i {0U}; elemCount > i; ++i) {
            if (syncWaiter.usedPeek(i)) {
                LwSciSyncAttrList const attr { syncWaiter.attrPeek(i) };
                if (nullptr != attr) {
                    unreconciledList.push_back(attr);
                } else {
                    cpuWaitAfterCopy = true;
                    break;
                }
            }
        }

        // If any attribute was NULL, clear out the list and instead
        //   insert CPU waiter attributes
        if (cpuWaitAfterCopy) {
            unreconciledList.resize(0);
            unreconciledList.push_back(cpuWaiterAttr.viewVal());
        }

        // Add the copy done signal attributes
        unreconciledList.push_back(copyDoneSignalAttr.viewVal());
    } catch (...) {
        setErrorEvent(LwSciError_InsufficientMemory);
        return;
    }

    // Reconcile the attributes and wrap the resulting lists
    LwSciSyncAttrList reconciledList { nullptr };
    LwSciSyncAttrList newConflictList { nullptr };
    LwSciError err {
        LwSciSyncAttrListReconcile(
                unreconciledList.data(),
                unreconciledList.size(),
                &reconciledList,
                &newConflictList)
    };
    LwSciWrap::SyncAttr conflictAttr { newConflictList, true };
    copyDoneAttr = LwSciWrap::SyncAttr(reconciledList, true);
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Trigger flow of waiter attributes upstream
    waiterAttrMsg = true;
    err = enqueueIpcWrite();
    if (LwSciError_Success != err) {
        setErrorEvent(err);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Block::elementCountGet() to retrieve count for number of syncs.
//! - Call LwSciSyncObjAlloc() to create a C2C LwSciSyncObj from the
//!   reconciled attributes.
//! - Create a new LwSciWrap::SyncObj that owns the C2C LwSciSyncObj.
//! - Create a temporary Signals instance and call Signals::sizeInit()
//!   and either Signals::syncFill() or Signals::doneSet() to fill it with
//!   copies of the sync object or leave it empty.
//! - Call dstRecvSyncSignal() interface of the source block to send the
//!   sync object info downstream.
//! - Call phaseProdSyncDoneSet() and phaseConsSyncDoneSet() to advance setup.
void C2CDst::dstRecvSyncSignal(
    uint32_t const dstIndex,
    Signals const& syncSignal) noexcept
{
    // Consumer's signalling sync objects are not transmitted or used directly
    //   (The fences from them are used, but the sync objects aren't needed)
    static_cast<void>(syncSignal);

    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation { };
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Retrieve element count
    size_t const elemCount { elementCountGet() };

    // Create a new sync object and wrap it
    LwSciSyncObj c2cSyncObj;
    LwSciError err { LwSciSyncObjAlloc(copyDoneAttr.viewVal(), &c2cSyncObj) };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A7_1_1), "Bug 3258479")
    c2cCopySyncObj = LwSciWrap::SyncObj(c2cSyncObj, true);

    // Fill a Signals object with the sync object or empty wrappers, depending
    //   on whether the consumer or the C2C object will handle the wait
    Signals c2cSyncSignal(FillMode::User, nullptr);
    err = c2cSyncSignal.sizeInit(1U, elemCount);
    if (LwSciError_Success == err) {
        if (cpuWaitAfterCopy) {
            err = c2cSyncSignal.doneSet();
        } else {
            err = c2cSyncSignal.syncFill(c2cCopySyncObj);
        }
    }
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Send C2C signal objects downstream
    getDst().srcRecvSyncSignal(c2cSyncSignal);

    // Prepare event for sending the object back to C2CSrc
    c2cCopySyncObjSend = true;
    err = enqueueIpcWrite();
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        setErrorEvent(err);
        LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    }

    // Advance setup phase
    phaseProdSyncDoneSet();
    phaseConsSyncDoneSet();
}

//! <b>Sequence of operations</b>
//! - Increase the number of available payloads from downstream.
//! - Call enqueueIpcWrite() to wake dispatch thread.
void C2CDst::dstRecvPayload(
    uint32_t const dstIndex,
    Packet const& consPayload) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Input value is ignored
    static_cast<void>(consPayload);

    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Index must be valid
    ValidateBits validation{};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // If disconnect has oclwrred, return but don't report an error
    // TODO: Need to distinguish various disconnect cases and handle
    //       in the validate function above.
    if (!connComplete()) {
        return;
    }

    numPayloadsAvailable++;

    // Wake waiting dispatch thread
    LwSciError const err{ enqueueIpcWrite() };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
    }

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - To avoid unncessary function calls, call runtimeEnabled() to
//!   determine whether in init or runtime phase.
//! - If init phase:
//! -- If there is a pending sync signal attribute message, call
//!    c2cSendSyncSignalAttr() and return.
//! -- Call Block::pktPendingEvent() to check if there is a pending packet
//!    creation message, and if so call c2cSendPacketCreate() to pack it.
//! -- If there is a pending packet list completion message, call
//!    c2cSendPacketsComplete() to pack it.
//! -- If there is a pending sync waiter attribute message, call
//!    c2cSendSyncWaiterAttr() and return.
//! -- If there is a pending C2C copy waiter object message, call
//!    c2cSendSyncWaiterObj() to pack it.
//! -- Call Block::pktPendingEvent() to check if there is a pending packet
//!    deletion message, and if so call c2cSendPacketDelete() to pack it.
//! - If runtime phase:
//! -- Check for a pending payload to return for reuse, and if so call
//!    c2cSendPayload() to pack it.
//! - Call IpcSrc::sendMessage() for any common IPC/C2C messages.
LwSciError
C2CDst::sendMessage(
    IpcBuffer& sendBuf,
    Lock& blockLock) noexcept
{
    // Init phase messages
    if (!runtimeEnabled()) {

        // Check for signal attribute message
        if (readDoneSignalAttrMsg) {
            readDoneSignalAttrMsg = false;
            return c2cSendSyncSignalAttr(sendBuf);
        }

        // Check for pending C2C packet creation event
        PacketPtr pkt { pktPendingEvent(&Packet::definePending, true) };
        if (nullptr != pkt) {
            return c2cSendPacketCreate(sendBuf, pkt);
        }

        // Check for pending C2C packet list completion message
        if (secondaryPacketExportEvent) {
            secondaryPacketExportEvent.store(false);
            return c2cSendPacketsComplete(sendBuf);
        }

        // Check for signal attribute message
        if (waiterAttrMsg) {
            waiterAttrMsg = false;
            return c2cSendSyncWaiterAttr(sendBuf);
        }

        // Check for pending C2C-copy sync message
        if (c2cCopySyncObjSend) {
            c2cCopySyncObjSend = false;
            return c2cSendSyncWaiterObj(sendBuf);
        }

        // Check for pending C2C packet deletion event
        pkt = pktPendingEvent(&Packet::deletePending, true);
        if (nullptr != pkt) {
            return c2cSendPacketDelete(sendBuf, pkt);
        }
    }

    // Runtime phase messages
    else {
        if (0U < numPayloadsAvailable) {
            return c2cSendPayload(sendBuf, blockLock);
        }
    }

    // Call IpcDst function for common IPC/C2C messages
    return IpcDst::sendMessage(sendBuf, blockLock);
}

//! <b>Sequence of operations</b>
//! - Call Packet::defineHandlePending() to clear the event indicating the
//!   packet's handle is available.
//! - Call IpcSrc::sendHeader() to initiate the message.
//! - Call Packet::handleGet() to retrieve the packet's handle.
//! - Call IpcBuffer::packval() to pack the handle into sendBuffer.
//! - Call Packet::definePack() to pack the Packet's definition into
//!   sendBuffer.
LwSciError
C2CDst::c2cSendPacketCreate(
    IpcBuffer& sendBuf,
    PacketPtr const& pkt) noexcept
{
    // Clear the packet's handle event
    static_cast<void>(pkt->defineHandlePending());

    // Pack message header
    LwSciError err { sendHeader(IpcMsg::C2CPacketCreate) };
    if (LwSciError_Success == err) {
        // Pack the packet handle
        err = sendBuf.packVal(pkt->handleGet());
        if (LwSciError_Success == err) {
            // Pack the packet definitions
            err = pkt->definePack(sendBuf);
        }
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call phasePacketsDoneSet() to advance setup.
//! - Call IpcSrc::sendHeader() to initiate the message.
LwSciError
C2CDst::c2cSendPacketsComplete(
    IpcBuffer& sendBuf) noexcept
{
    // Lwrrently no data to send
    static_cast<void>(sendBuf);

    // Advance setup phase
    phasePacketsDoneSet();

    // Pack message header
    return sendHeader(IpcMsg::C2CPacketsComplete);
}

//! <b>Sequence of operations</b>
//! - Call Packet::deleteCookiePending() to clear the event indicating the
//!   packet's cookie is available.
//! - Call Packet::handleGet() to retrieve the packet's handle.
//! - Call Block::pktRemove() to remove the packet from the map.
//! - Call IpcSrc::sendHeader() to initiate the message.
//! - Call IpcBuffer::packval() to pack the handle into sendBuffer.
LwSciError
C2CDst::c2cSendPacketDelete(
    IpcBuffer& sendBuf,
    PacketPtr const& pkt) noexcept
{
    // Clear the packet's handle event
    static_cast<void>(pkt->deleteCookiePending());

    // Remove the packet from the map
    LwSciStreamPacket const handle { pkt->handleGet() };
    pktRemove(handle, true);

    // Pack message header
    LwSciError err { sendHeader(IpcMsg::C2CPacketDelete) };
    if (LwSciError_Success == err) {
        // Pack the packet handle
        err = sendBuf.packVal(handle);
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call IpcSrc::sendHeader() to initiate the message.
//! - Call ipcBufferPack(LwSciWrap::SyncAttr) to pack the sync attributes into
//!   sendBuffer.
LwSciError
C2CDst::c2cSendSyncSignalAttr(
    IpcBuffer& sendBuf) noexcept
{
    // Pack message header
    LwSciError err { sendHeader(IpcMsg::SignalAttr) };
    if (LwSciError_Success == err) {
        err = ipcBufferPack(sendBuf, readDoneSignalAttr);
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call sendHeader() to initiate the message.
LwSciError
C2CDst::c2cSendSyncWaiterAttr(
    IpcBuffer& sendBuf) noexcept
{
    // Note: This message ensures flow of waiter attributes from one endpoint
    //   to the other in C2C matches IPC, even though there is no data.

    // No data to send
    static_cast<void>(sendBuf);

    // Initiate the message
    return sendHeader(IpcMsg::C2CWaiterAttr);
}

//! <b>Sequence of operations</b>
//! - Call IpcSrc::sendHeader() to initiate the message.
//! - Call ipcBufferPack(LwSciWrap::SyncObj) to pack the sync object into
//!   sendBuffer.
LwSciError
C2CDst::c2cSendSyncWaiterObj(
    IpcBuffer& sendBuf) noexcept
{
    // Pack message header
    LwSciError err { sendHeader(IpcMsg::WaiterObj) };
    if (LwSciError_Success == err) {
        err = ipcBufferPack(sendBuf, c2cCopySyncObj);
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call srcDequeuePayload() interface of destination block to obtain next
//!   available payload.
//! - Call Packet::handleGet() to retrieve the handle for the payload and call
//!   Block::pktFindByHandle() to find the corresponding local packet.
//! - Call Packet::locationUpdate() to move the packet from downstream to
//!   queued for transmission.
//! - Call Packet::fenceProdReset() to clear the old producer fences.
//! - Call Packet::fenceConsCopy() to copy the incoming fences and
//!   Packet::fenceConsWait() to wait for them.
//! - Call IpcSrc::sendHeader() to initiate the message.
//! - Call IpcBuffer::packval() to pack the handle into sendBuffer.
LwSciError
C2CDst::c2cSendPayload(
    IpcBuffer& sendBuf,
    Lock& blockLock) noexcept
{
    if (0U == numPayloadsAvailable) {
        return LwSciError_StreamInternalError;
    }

    // Decrement counter and dequeue available payload
    numPayloadsAvailable--;

    // Unlock before obtaining payload from downstream.
    blockLock.unlock();
    PacketPtr const usePayload{ getDst().srcDequeuePayload() };

    // Return if the stream is disconnected
    if (!connComplete()) {
        return LwSciError_Success;
    }

    if (nullptr == usePayload) {
        return LwSciError_NoStreamPacket;
    }

    // Retake the lock
    blockLock.lock();

    // Look up corresponding local packet
    PacketPtr const pkt{ pktFindByHandle(usePayload->handleGet(), true) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }

    // Validate and update location
    if (!pkt->locationUpdate(Packet::Location::Downstream,
                             Packet::Location::Upstream)) {
        return LwSciError_StreamPacketInaccessible;
    }

    // Copy consumer fences into packet and wait for them
    pkt->fenceProdReset();
    LwSciError err{ pkt->fenceConsCopy(*usePayload) };
    if (LwSciError_Success == err) {
        // Do cpu wait
        err = pkt->fenceConsWait(waitContext);
    }
    if (LwSciError_Success != err) {
        return err;
    }

    // Pack message header
    err = sendHeader(IpcMsg::C2CPayload);
    if (LwSciError_Success == err) {
        // Pack the packet handle
        err = sendBuf.packVal(pkt->handleGet());
    }

    return err;
}

} // namespace LwSciStream
