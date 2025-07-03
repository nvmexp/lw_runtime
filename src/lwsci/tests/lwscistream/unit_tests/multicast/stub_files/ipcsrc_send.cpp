//! \file
//! \brief LwSciStream IPC source block transmission.
//!
//! \copyright
//! Copyright (c) 2020-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.
#include <cstdint>
#include <array>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <mutex>
#include <vector>
#include <functional>
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
#include "packet.h"
#include "trackarray.h"
#include "enumbitset.h"
#include "block.h"
#include "ipccomm_common.h"
#include "ipcbuffer.h"
#include "ipcbuffer_sciwrap.h"
#include "ipccomm.h"
#include "ipcsrc.h"
#include "endinfo.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Call Block::blkMutexLock() to lock the block mutex and call
//!   Block::prodInfoSet() to save the incoming producer info.
//! - Call Block::prodInfoFlow() to trigger event.
//! - Call IpcComm::signalWrite() to wake dispatch thread.
void IpcSrc::srcRecvProdInfo(
    uint32_t const srcIndex,
    EndInfoVector const& info) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Lock and save the producer info
    {
        Lock const blockLock { blkMutexLock() };
        LwSciError const err { prodInfoSet(info) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }
    }

    // Trigger event
    prodInfoFlow();

    // Mark pending connection message
    connectMsg = true;
    LwSciError const err2 { comm.signalWrite() };
    if (LwSciError_Success != err2) {
        setErrorEvent(err2);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Elements::dataCopy() to save a copy of the element list for
//!   transmission over IPC.
//! - Call Elements::sizePeek() to retrieve the number of elements.
//! - Call Block::elementCountSet() to save the number of elements.
//! - Call Waiters::sizeInit() to initialize the sync attribute vectors.
//! - Call IpcComm::signalWrite() to signal data is available for sending.
void IpcSrc::srcRecvAllocatedElements(
    uint32_t const dstIndex,
    Elements const& inElements) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Copy incoming data
    LwSciError err { allocatedElements.dataCopy(inElements, false) };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Retrieve and save number of elements
    size_t const elemCount { inElements.sizePeek() };
    elementCountSet(elemCount);

    // Initialize waiter sync attribute trackers if this is regular IPC
    if (!isC2CBlock) {
        err = endSyncWaiter.sizeInit(elemCount);
        if (LwSciError_Success != err) {
            setErrorEvent(err);
            return;
        }
        err = ipcSyncWaiter.sizeInit(elemCount);
        if (LwSciError_Success != err) {
            setErrorEvent(err);
            return;
        }
    }

    // Signal availability of message
    err = comm.signalWrite();
    if (LwSciError_Success != err) {
        setErrorEvent(err);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Block::pktCreate() to create a new packet, copy the definition,
//!   and insert it in the map.
//! - Call IpcComm::signalWrite() on comm.
void IpcSrc::srcRecvPacketCreate(
    uint32_t const srcIndex,
    Packet const& origPacket) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation { };
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Create new packet, copy definition, and insert in map.
    //   The handle is used as the cookie, to facilitate deletion.
    //   This will automatically queue up creation event.
    LwSciError err {
        pktCreate(origPacket.handleGet(), &origPacket, origPacket.handleGet())
    };

    // Set any error and wake waiting threads
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(M0_1_2), "Bug 3003226")
    if (LwSciError_Success == err) {
        err = comm.signalWrite();
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
//! - Call Block::pktFindByHandle() to retrieve the Packet instance
//!   for @a handle.
//! - Call Packet::locationCheck() to ensure the Packet is upstream.
//! - Call Packet::deleteSet() to mark the packet for deletion.
//! - Call IpcComm::signalWrite() on comm.
void IpcSrc::srcRecvPacketDelete(
    uint32_t const srcIndex,
    LwSciStreamPacket const handle) noexcept
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
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Find the packet for this handle
    PacketPtr const pkt { pktFindByHandle(handle) };
    if (nullptr == pkt) {
        setErrorEvent(LwSciError_StreamBadPacket);
        return;
    }

    // Should only receive deletion message when the packet is upstream
    if (!pkt->locationCheck(Packet::Location::Upstream)) {
        setErrorEvent(LwSciError_StreamPacketInaccessible);
        return;
    }

    // Mark the packet for deletion
    if (!pkt->deleteSet()) {
        setErrorEvent(LwSciError_StreamPacketDeleted);
        return;
    }

    // Wake any waiting threads
    LwSciError const err { comm.signalWrite() };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Mark packet completion as done if not already done.
//! - Call phasePacketsDoneSet() to advance setup phase.
//! - Set flag indicating pending packet completion event.
//! - Call IpcComm::signalWrite() on comm.
void IpcSrc::srcRecvPacketsComplete(
    uint32_t const srcIndex) noexcept
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
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Mark packets completed if not already done
    bool expected { false };
    if (!allocatedPacketExportDone.compare_exchange_strong(expected, true)) {
        setErrorEvent(LwSciError_AlreadyDone);
        return;
    }

    // Advance setup phase
    // TODO: May need special handling for C2C
    phasePacketsDoneSet();

    // Set flag for event
    allocatedPacketExportEvent.store(true);

    // Wake any waiting threads if not C2C block.
    // C2C does not send this event.
    if (!isC2CBlock) {
        LwSciError const err { comm.signalWrite() };
        if (LwSciError_Success != err) {
            setErrorEvent(err);
        }
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Waiters::copy() to save a copy of the waiter information for
//!   transmission over IPC.
//! - Call IpcComm::signalWrite() to signal data is available for sending.
void IpcSrc::srcRecvSyncWaiter(
    uint32_t const srcIndex,
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
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Take the mutex lock and save a copy of the incoming information
    {
        Lock const blockLock { blkMutexLock() };
        LwSciError const err { endSyncWaiter.copy(syncWaiter) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }
    }

    // Signal available message or set error
    LwSciError const err2 { comm.signalWrite() };
    if (LwSciError_Success != err2) {
        setErrorEvent(err2);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Signals::copy() to save a copy of the signal information for
//!   transmission over IPC.
//! - Call IpcComm::signalWrite() to signal data is available for sending.
//! - Call phaseProdSyncDoneSet() to advance setup phase.
void IpcSrc::srcRecvSyncSignal(
    uint32_t const srcIndex,
    Signals const& syncSignal) noexcept
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
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Take the mutex lock and save a copy of the incoming information
    {
        Lock const blockLock { blkMutexLock() };
        LwSciError const err { endSyncSignal.copy(syncSignal) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }
    }

    // Signal available message or set error
    LwSciError const err2 { comm.signalWrite() };
    if (LwSciError_Success != err2) {
        setErrorEvent(err2);
    }

    // Advance setup phase
    phaseProdSyncDoneSet();

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Packet::handleGet() to retrieve the packet handle, and then call
//!   Block::pktFindByHandle() to find the corresponding local packet.
//! - Call Packet::locationUpdate() to move the packet from upstream to queued
//!   for transmission.
//! - Call Packet::fenceProdCopy() to copy the incoming fences and
//!   Packet::fenceConsReset() to clear the old consumer fences.
//! - Call Packet::PayloadQ::enqueue() to add to the queue of packets
//!   ready to transmit.
//! - Call IpcComm::signalWrite() on comm.
//!
//! \implements{19675914}
void IpcSrc::srcRecvPayload(
    uint32_t const srcIndex,
    Packet const& prodPayload) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation { };
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Find the local packet for this handle
    PacketPtr const pkt { pktFindByHandle(prodPayload.handleGet()) };
    if (nullptr == pkt) {
        setErrorEvent(LwSciError_StreamBadPacket);
        return;
    }

    // Validate and update location
    if (!pkt->locationUpdate(Packet::Location::Upstream,
                             Packet::Location::Queued)) {
        setErrorEvent(LwSciError_StreamPacketInaccessible);
        return;
    }

    // Copy the producer fences, and reset the consumer fences
    LwSciError err { pkt->fenceProdCopy(prodPayload) };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }
    pkt->fenceConsReset();

    // Take lock and add to queue
    {
        Lock const blockLock { blkMutexLock() };
        readyPayloadQueue.enqueue(pkt);
    }

    // Wake waiting dispatch thread
    err = comm.signalWrite();
    if (LwSciError_Success != err) {
        setErrorEvent(err);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Set pending event flag.
//! - Call IpcComm:signalWrite().
void
IpcSrc::phaseSendChange(void) noexcept
{
    // Set flag for pending event
    runtimeBeginMsg = true;

    // Wake waiting dispatch thread
    LwSciError const err { comm.signalWrite() };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
    }
}

//! <b>Sequence of operations</b>
//! - Calls Block::disconnectSrc(), Block::disconnectEvent() on itself.
//! - Calls IpcComm::signalWrite() on comm.
//! - Checks if the disconnect has already been requested by instrumenting
//!   disconnectRequested with std::atomic<bool>::load().
//! - If yes, exits the dispatched I/O thread, if it is still running,
//!   by setting disconnectRequested to true with std::atomic<bool>::store()
//!   and calling IpcComm::signalDisconnect() on comm.
//!
//! \implements{19675917}
void IpcSrc::srcDisconnect(
    uint32_t const srcIndex) noexcept
{
    // Validate block/input state
    //   Index must be valid
    ValidateBits validation { };
    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }
    // signal disconnect upstream
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectSrc();

    // Signal disconnect event if not already done
    disconnectEvent();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))

    // also enqueue event onto write queue
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    disconnectMsg = true;
    LwSciError const err { comm.signalWrite() };
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        setErrorEvent(err);
        LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    }

    // takedown io loop
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    destroyIOLoop();
}

//! <b>Sequence of operations</b>
//! - Creates a new Block::Lock object with blkMutexLock() to protect
//!   against new messages being sheduled while checking for messages
//!   to send, which could cause messages to flow out of order.
//! - If there is a pending connection message, call ipcSendConnect()
//!   to pack it and return.
//! - Call sendMessage() to check for any other messages as prioritized
//!   by the block type. If one is found, return.
//!
//! \implements{19977678}
LwSciError IpcSrc::processWriteMsg() noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Take lock.
    // TODO: The lock is needed while checking for all pending events, to
    //       ensure nothing comes along in the middle of the check. But
    //       once we have the data to send, we could potentially unlock
    //       before this function exits, allowing more time for new info
    //       to flow in from other threads.
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    Lock blockLock { blkMutexLock() };
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))

    // Return immediately if messages are not allowed yet, unless there's
    //  a disconnect message, which can happen at any time
    if (!disconnectMsg && !sendEnabled(connectMsg)) {
        return LwSciError_Success;
    }

    // Connection is handled first, before any block-specific messages.
    if (connectMsg) {
        connectMsg = false;
        return ipcSendConnect(sendBuffer);
    }

    // Pack any block-specific messages
    LwSciError err { sendMessage(sendBuffer, blockLock) };
    if (LwSciError_Success != err) {
        // On failure, clear flag indicating the buffer has a message
        sendBufferPacked = false;
    }

    return err;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Checks whether current state allows message sending.
bool
IpcSrc::sendEnabled(
    bool const connect) const noexcept
{
    // No messages can be sent from Src block before connection readiness
    //   is received from Dst
    if (!connectReadyDone) {
        return false;
    }
    // No message can be sent from Src block before connection is complete
    //   except the connection message itself
    else if (!connectStartDone) {
        return connect;
    }
    // After both are done, any message is allowed
    else {
        return true;
    }
}

//! <b>Sequence of operations</b>
//! - To avoid unncessary function calls, call runtimeEnabled() to
//!   determine whether in init or runtime phase.
//! - If init phase:
//! -- Call Elements::eventGet() to check for pending allocated elements
//!    message, and if found call sendAllocatedElements() and return.
//! -- Call Block::pktPendingEvent() to check for pending packet creation
//!    message, and if found call ipcSendPacketCreate() and return.
//! -- Check for pending packet list completion message, and if found call
//!    ipcSendPacketsComplete) and return.
//! -- Call Waiters::pendingEvent() to check for pending waiter sying attribute
//!    message, and if found call ipcSendSyncWaiterAttr() and return.
//! -- Call Signals::pendingEvent() to check for pending signal sync object
//!    message, and if found call ipcSendSyncSignalObj() and return.
//! - If runtime phase:
//! -- Check for pending setup completion message and if found call
//!    sendRuntimeBegin() and return.
//! -- Call Packet::PayloadQ::dequeue() to check for pending payload
//!    message, and if found call ipcSendPayload() and return.
//! - Call Block::pktPendingEvent() to checkk for pending packet deletion
//!   message, and if found call ipcSendPacketDelete().
//! - Check for pending disconnection message, and if found call
//!   sendDisconnect() and return.
LwSciError
IpcSrc::sendMessage(
    IpcBuffer& sendBuf,
    Lock& blockLock) noexcept
{
    static_cast<void>(blockLock);

    // Init phase messages
    if (!runtimeEnabled()) {

        // Check for pending allocated elements message
        if (allocatedElements.eventGet()) {
            return sendAllocatedElements(sendBuf);
        }

        // Check for pending packet creation messages
        // TODO: Make sure the flag is cleared for C2C so don't need condition
        if (!isC2CBlock) {
            PacketPtr const pkt
                { pktPendingEvent(&Packet::definePending, true) };
            if (nullptr != pkt) {
                return ipcSendPacketCreate(sendBuf, pkt);
            }
        }

        // Check for pending packet list completion event
        // TODO: Make sure the flag is cleared for C2C so don't need condition
        if (!isC2CBlock && allocatedPacketExportEvent) {
            allocatedPacketExportEvent.store(false);
            return ipcSendPacketsComplete(sendBuf);
        }

        // Check for pending sync waiter attribute message
        // TODO: Make sure the flag is cleared for C2C so don't need condition
        if (!isC2CBlock && endSyncWaiter.pendingEvent()) {
            return ipcSendSyncWaiterAttr(sendBuf);
        }

        // Check for pending sync signal object message
        // TODO: Refine the IPC/C2C difference
        if (!isC2CBlock && endSyncSignal.pendingEvent()) {
            return ipcSendSyncSignalObj(sendBuf);
        }

        // Check for setup completion event
        if (runtimeBeginMsg) {
            runtimeBeginMsg = false;
            return sendRuntimeBegin(sendBuf);
        }
    }

    // Runtime phase messages
    else {

        // Check for pending payload messages
        PacketPtr const pkt { readyPayloadQueue.dequeue() };
        if (nullptr != pkt) {
            return ipcSendPayload(sendBuf, pkt);
        }
    }

    // Check for pending packet deletion events
    // TODO: Make sure the flag is cleared for C2C so don't need condition
    if (!isC2CBlock) {
        PacketPtr const pkt { pktPendingEvent(&Packet::deletePending, true) };
        if (nullptr != pkt) {
            return ipcSendPacketDelete(sendBuf, pkt);
        }
    }

    // Disconnection events are handled last
    if (disconnectMsg) {
        disconnectMsg = false;
        return sendDisconnect(sendBuf);
    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::changeMode() to initiate packing of sendBuffer.
//! - Call ipcBuffer:packVal() to pack the message type into sendBuffer.
LwSciError
IpcSrc::sendHeader(
    IpcMsg const msgType) noexcept
{
    // Shouldn't be here if a message is already packed
    assert(!sendBufferPacked);

    // Put sendBuffer into pack mode
    LwSciError err { sendBuffer.changeMode(IpcBuffer::UserMode::Pack) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Pack message type into buffer
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP37_C), "LwSciStream-ADV-CERTC-003")
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    err = sendBuffer.packVal(msgType);
    if (LwSciError_Success != err) {
        return err;
    }
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP37_C))

    // Indicate message in buffer
    //   The message may not be done yet, but but it will be before control
    //   returns to the point where the message will be sent.
    sendBufferPacked = true;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call sendHeader() to initiate the message.
//! - Call ipcBufferPack() to pack the producer endpoint info.
LwSciError
IpcSrc::ipcSendConnect(
    IpcBuffer& sendBuf) noexcept
{
    // Initiate the message
    LwSciError err { sendHeader(IpcMsg::Connect) };
    if (LwSciError_Success == err) {
        // Pack the producer endpoint info
        err = ipcBufferPack(sendBuf, prodInfoGet());
        if (LwSciError_Success != err) {
            sendBufferPacked = false;
        }
    }

    // Mark sending of connection completion done
    connectStartDone = true;

    return err;
}

//! <b>Sequence of operations</b>
//! - Call sendHeader() to initiate the message.
//! - Call Elements::dataPack() to pack the elements.
LwSciError
IpcSrc::sendAllocatedElements(
    IpcBuffer& sendBuf) noexcept
{
    // Initiate the message
    LwSciError err { sendHeader(IpcMsg::AllocatedElements) };
    if (LwSciError_Success == err) {
        // Pack element information
        err = allocatedElements.dataPack(sendBuf);
    }

    return err;
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
IpcSrc::ipcSendPacketCreate(
    IpcBuffer& sendBuf,
    PacketPtr const& pkt) noexcept
{
    // Clear the packet's handle event
    static_cast<void>(pkt->defineHandlePending());

    // Pack message header
    LwSciError err { sendHeader(IpcMsg::IPCPacketCreate) };
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
//! - Call IpcSrc::sendHeader() to initiate the message.
LwSciError
IpcSrc::ipcSendPacketsComplete(
    IpcBuffer& sendBuf) noexcept
{
    // Lwrrently no data to send
    static_cast<void>(sendBuf);

    // Pack message header
    return sendHeader(IpcMsg::IPCPacketsComplete);
}

//! <b>Sequence of operations</b>
//! - Call Packet::deleteCookiePending() to clear the event indicating the
//!   packet's cookie is available.
//! - Call Packet::handleGet() to retrieve the packet's handle.
//! - Call Block::pktRemove() to remove the packet from the map.
//! - Call IpcSrc::sendHeader() to initiate the message.
//! - Call IpcBuffer::packval() to pack the handle into sendBuffer.
LwSciError
IpcSrc::ipcSendPacketDelete(
    IpcBuffer& sendBuf,
    PacketPtr const& pkt) noexcept
{
    // Clear the packet's handle event
    static_cast<void>(pkt->deleteCookiePending());

    // Remove the packet from the map
    LwSciStreamPacket const handle { pkt->handleGet() };
    pktRemove(handle, true);

    // Pack message header
    LwSciError err { sendHeader(IpcMsg::IPCPacketDelete) };
    if (LwSciError_Success == err) {
        // Pack the packet handle
        err = sendBuf.packVal(handle);
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call sendHeader() to initiate the message.
//! - Call Waiters::pack() to pack the sync attributes.
LwSciError
IpcSrc::ipcSendSyncWaiterAttr(
    IpcBuffer& sendBuf) noexcept
{
    // Initiate the message
    LwSciError err { sendHeader(IpcMsg::IPCWaiterAttr) };
    if (LwSciError_Success == err) {
        // Pack waiter attr information
        err = endSyncWaiter.pack(sendBuf);
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call sendHeader() to initiate the message.
//! - Call Signals::pack() to pack the sync objects.
LwSciError
IpcSrc::ipcSendSyncSignalObj(
    IpcBuffer& sendBuf) noexcept
{
    // Initiate the message
    LwSciError err { sendHeader(IpcMsg::SignalObj) };
    if (LwSciError_Success == err) {
        // Pack element information
        err = endSyncSignal.pack(sendBuf);
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call sendHeader() to initiate the message.
LwSciError
IpcSrc::sendRuntimeBegin(
    IpcBuffer& sendBuf) noexcept
{
    // No data to send
    static_cast<void>(sendBuf);

    // Switch from sending init messages to runtime
    runtimeBeginDone = true;

    // Initiate the message
    return sendHeader(IpcMsg::Runtime);
}

//! <b>Sequence of operations</b>
//! - Call Packet::locationUpdate() to update the packet location.
//! - Call sendHeader() to initiate the message.
//! - Call Packet::handleGet() to retrieve the packet's handle.
//! - Call IpcBuffer::packval() to pack the handle into sendBuffer.
//! - Call Packet::fenceProdPack() to pack the packet fences.
LwSciError
IpcSrc::ipcSendPayload(
    IpcBuffer& sendBuf,
    PacketPtr const& pkt) noexcept
{
    // Move the packet from queued to downstream
    static_cast<void>(
        pkt->locationUpdate(Packet::Location::Queued,
                            Packet::Location::Downstream));

    // Initiate the message
    LwSciError err { sendHeader(IpcMsg::IPCPayload) };
    if (LwSciError_Success == err) {
        err = sendBuf.packVal(pkt->handleGet());
        if (LwSciError_Success == err) {
            // Pack the producer fences
            err = pkt->fenceProdPack(sendBuf);
        }
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call sendHeader() to initiate the message.
LwSciError
IpcSrc::sendDisconnect(
    IpcBuffer& sendBuf) noexcept
{
    // No data to send
    static_cast<void>(sendBuf);

    // Initiate the message. No other data is needed
    return sendHeader(IpcMsg::Disconnect);
}

} // namespace LwSciStream
