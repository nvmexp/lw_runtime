//! \file
//! \brief LwSciStream IPC destination block transmission.
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
#include <array>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <vector>
#include <functional>
#include <bitset>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include "covanalysis.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "lwscistream_common.h"
#include "lwscistream_types.h"
#include "lwscicommon_os.h"
#include "sciwrap.h"
#include "trackarray.h"
#include "safeconnection.h"
#include "block.h"
#include "packet.h"
#include "enumbitset.h"
#include "ipccomm_common.h"
#include "ipccomm.h"
#include "ipcbuffer.h"
#include "ipcbuffer_sciwrap.h"
#include "ipcdst.h"
#include "endinfo.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Call Block::blkMutexLock() to lock the block mutex and call
//!   Block::consInfoSet() to save the incoming consumer info.
//! - Call Block::consInfoFlow() to update flow state.
//! - Call IpcComm::signalWrite() to wake dispatch thread.
void IpcDst::dstRecvConsInfo(
    uint32_t const dstIndex,
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
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Lock and save the consumer info
    {
        Lock const blockLock { blkMutexLock() };
        LwSciError const err { consInfoSet(info) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }
    }

    // Update flow state
    //   For this block, this lwrrently doesn't do anything but clear the flag
    //   indicating the info is pending. But we call it for symmetry with the
    //   downstream flow, and in case any other operations are added to the
    //   function in the future.
    consInfoFlow();

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
//! - Call IpcComm::signalWrite() to signal data is available for sending.
void IpcDst::dstRecvSupportedElements(
    uint32_t const dstIndex,
    Elements const& inElements) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Copy incoming data and signal data available to write
    LwSciError err { supportedElements.dataCopy(inElements, false) };
    if (LwSciError_Success == err) {
        err = comm.signalWrite();
    }

    // Set error event on failure
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(M0_1_2), "Bug 3003226")
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        setErrorEvent(err);
        LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    }
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindHandle() to find the local Packet instance
//!   corresponding to the incoming packet.
//! - Call Block::blkMutexLock() to take the lock and then call
//!   Packet::statusConsCopy() to copy the consumer status information
//!   into the local Packet.
//! - Call IpcComm::signalWrite() to signal an available message.
void IpcDst::dstRecvPacketStatus(
    uint32_t const dstIndex,
    Packet const& origPacket) noexcept
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

    // Find the packet for this handle
    PacketPtr const pkt { pktFindByHandle(origPacket.handleGet()) };
    if (nullptr == pkt) {
        setErrorEvent(LwSciError_StreamBadPacket);
        return;
    }

    // Take the lock, copy the status and signal message to write
    LwSciError err { };
    {
        Lock const blockLock { blkMutexLock() };
        err = pkt->statusConsCopy(origPacket);
        if (LwSciError_Success == err) {
            err = comm.signalWrite();
        }
    }
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Waiters::copy() to save a copy of the waiter information for
//!   transmission over IPC.
//! - Call IpcComm::signalWrite() to signal data is available for sending.
void IpcDst::dstRecvSyncWaiter(
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
//! - Call phaseConsSyncDoneSet() to advance setup phase.
void IpcDst::dstRecvSyncSignal(
    uint32_t const dstIndex,
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
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
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
    phaseConsSyncDoneSet();

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Packet::handleGet() to retrieve the packet handle, and then call
//!   Block::pktFindByHandle() to find the corresponding local packet.
//! - Call Packet::locationUpdate() to move the packet from downstream to
//!   queued for transmission.
//! - Call Packet::fenceConsCopy() to copy the incoming fences and
//!   Packet::fenceProdReset() to clear the old producer fences.
//! - Call Packet::PayloadQ::enqueue() to add to the queue of packets
//!   ready to transmit.
//! - Call IpcComm::signalWrite() on comm.
//!
//! \implements{19791603}
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
void IpcDst::dstRecvPayload(
    uint32_t const dstIndex,
    Packet const& consPayload) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Index must be valid
    ValidateBits validation { };
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // If disconnect has oclwrred, return but don't report an error
    // TODO: Need to distinguish various disconnect cases and handle
    //       in the validate function above.
    if (!connComplete()) {
        return;
    }

    // Find the local packet for this handle
    PacketPtr const pkt { pktFindByHandle(consPayload.handleGet()) };
    if (nullptr == pkt) {
        setErrorEvent(LwSciError_StreamBadPacket);
        return;
    }

    // Validate and update location
    if (!pkt->locationUpdate(Packet::Location::Downstream,
                             Packet::Location::Queued)) {
        setErrorEvent(LwSciError_StreamPacketInaccessible);
        return;
    }

    // Copy the consumer fences, and reset the producer fences
    LwSciError err { pkt->fenceConsCopy(consPayload) };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }
    pkt->fenceProdReset();

    // Take lock and add to queue
    {
        Lock const blockLock { blkMutexLock() };
        reusePayloadQueue.enqueue(pkt);
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
IpcDst::phaseSendReady(void) noexcept
{
    // Set flag for pending event
    runtimeReadyMsg = true;

    // Wake waiting dispatch thread
    LwSciError const err { comm.signalWrite() };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
    }
}

//! <b>Sequence of operations</b>
//! - Calls Block::disconnectDst().
//! - Calls Block::disconnectEvent().
//! - Calls IpcComm::signalWrite() on comm.
//! - Checks if the disconnect has already been
//!   requested by instrumenting disconnectRequested with
//!   std::atomic<bool>::load().
//! - If yes, exits the dispatched I/O thread,
//!   if it is still running, by setting disconnectRequested to true with
//!   std::atomic<bool>::store() and calling
//!   IpcComm::signalDisconnect() on comm.
//!
//! \implements{19791606}
void IpcDst::dstDisconnect(
    uint32_t const dstIndex) noexcept
{
    // Validate block/input state
    //   Index must be valid
    ValidateBits validation { };
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectDst();

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

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    destroyIOLoop();
}

//! <b>Sequence of operations</b>
//! - Creates a new Block::Lock object with blkMutexLock() to protect
//!   against new messages being sheduled while checking for messages
//!   to send, which could cause messages to flow out of order.
//! - If there is a pending connection message, call sendHeader() and
//!   ipcBufferPackt to pack it and return.
//! - Call sendMessage() to check for any other messages as prioritized
//!   by the block type. If one is found, return.
//!
//! \implements{19839633}
LwSciError IpcDst::processWriteMsg() noexcept
{
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
}

//! <b>Sequence of operations</b>
//! - Checks whether current state allows message sending.
bool
IpcDst::sendEnabled(
    bool const connect) const noexcept
{
    // No messages can be sent from Dst block before connection readiness
    //   except the readiness message itself
    if (!connectReadyDone) {
        return connect;
    }
    // No other messages can be sent from Dst block before connection is
    //   complete
    else {
        return connectStartDone;
    }
}

//! <b>Sequence of operations</b>
//! - To avoid unncessary function calls, call runtimeEnabled() to
//!   determine whether in init or runtime phase.
//! - If init phase:
//! -- Call Elements::eventGet() to check for pending supported elements
//!    message, and if found call sendSupportedElements() and return.
//! -- Call Block::pktPendingEvent() to check for pending packet status
//!    message, and if found call ipcSendPacketStatus() and return.
//! -- Call Waiters::pendingEvent() to check for pending waiter sying attribute
//!    message, and if found call ipcSendSyncWaiterAttr() and return.
//! -- Call Signals::pendingEvent() to check for pending signal sync object
//!    message, and if found call ipcSendSyncSignalObj() and return.
//! - If runtime phase:
//! -- Check for pending setup completion message and if found call
//!    sendRuntimeReady() and return.
//! -- Call Packet::PayloadQ::dequeue() to check for pending payload
//!    message, and if found call ipcSendPayload() and return.
//! - Check for pending disconnection message, and if found call
//!   sendDisconnect() and return.
LwSciError
IpcDst::sendMessage(
    IpcBuffer& sendBuf,
    Lock& blockLock) noexcept
{
    static_cast<void>(blockLock);

    // Init phase messages
    if (!runtimeEnabled()) {

        // Check for pending allocated elements message
        if (supportedElements.eventGet()) {
            return sendSupportedElements(sendBuf);
        }

        // Check for pending packet status messages
        // TODO: Make sure the flag is cleared for C2C so don't need condition
        if (!isC2CBlock) {
            PacketPtr const pkt
                { pktPendingEvent(&Packet::statusPending, true) };
            if (nullptr != pkt) {
                return ipcSendPacketStatus(sendBuf, pkt);
            }
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
        if (runtimeReadyMsg) {
            runtimeReadyMsg = false;
            return sendRuntimeReady(sendBuf);
        }
    }

    // Runtime phase messages
    else {

        // Check for pending payload messages
        PacketPtr const pkt { reusePayloadQueue.dequeue() };
        if (nullptr != pkt) {
            return ipcSendPayload(sendBuf, pkt);
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
IpcDst::sendHeader(
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
//! - Call ipcBufferPack() to pack the consumer endpoint info.
LwSciError
IpcDst::ipcSendConnect(
    IpcBuffer& sendBuf) noexcept
{
    // Initiate the message
    LwSciError err { sendHeader(IpcMsg::Connect) };
    if (LwSciError_Success == err) {
        // Pack the consumer endpoint info
        err = ipcBufferPack(sendBuf, consInfoGet());
        if (LwSciError_Success != err) {
            sendBufferPacked = false;
        }
    }

    // Mark sending of connection readiness done
    connectReadyDone = true;

    return err;
}

//! <b>Sequence of operations</b>
//! - Call sendHeader() to initiate the message.
//! - Call Elements::dataPack() to pack the elements.
LwSciError
IpcDst::sendSupportedElements(
    IpcBuffer& sendBuf) noexcept
{
    // Initiate the message
    LwSciError err { sendHeader(IpcMsg::SupportedElements) };
    if (LwSciError_Success == err) {
        // Pack element information
        err = supportedElements.dataPack(sendBuf);
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call IpcDst::sendHeader() to initiate the message.
//! - Call Packet::handleGet() to retrieve the packet's handle.
//! - Call IpcBuffer::packval() to pack the handle into sendBuffer.
//! - Call Packet::statusCondPack() to pack the Packet's consumer status
//!   into sendBuffer.
LwSciError
IpcDst::ipcSendPacketStatus(
    IpcBuffer& sendBuf,
    PacketPtr const& pkt) noexcept
{
    // Pack message header
    LwSciError err { sendHeader(IpcMsg::IPCPacketStatus) };
    if (LwSciError_Success == err) {
        // Pack the packet handle
        err = sendBuf.packVal(pkt->handleGet());
        if (LwSciError_Success == err) {
            // Pack the packet definitions
            err = pkt->statusConsPack(sendBuf);
        }
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call sendHeader() to initiate the message.
//! - Call Waiters::pack() to pack the sync attributes.
LwSciError
IpcDst::ipcSendSyncWaiterAttr(
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
IpcDst::ipcSendSyncSignalObj(
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
IpcDst::sendRuntimeReady(
    IpcBuffer& sendBuf) noexcept
{
    // No data to send
    static_cast<void>(sendBuf);

    // Switch from sending init messages to runtime
    runtimeReadyDone = true;

    // Initiate the message
    return sendHeader(IpcMsg::Runtime);
}

//! <b>Sequence of operations</b>
//! - Call Packet::locationUpdate() to update the packet location.
//! - Call sendHeader() to initiate the message.
//! - Call Packet::handleGet() to retrieve the packet's handle.
//! - Call IpcBuffer::packval() to pack the handle into sendBuffer.
//! - Call Packet::fenceConsPack() to pack the packet fences.
LwSciError
IpcDst::ipcSendPayload(
    IpcBuffer& sendBuf,
    PacketPtr const& pkt) noexcept
{
    // Move the packet from queued to upstream
    static_cast<void>(
        pkt->locationUpdate(Packet::Location::Queued,
                            Packet::Location::Upstream));

    // Initiate the message
    LwSciError err { sendHeader(IpcMsg::IPCPayload) };
    if (LwSciError_Success == err) {
        err = sendBuf.packVal(pkt->handleGet());
        if (LwSciError_Success == err) {
            // Pack the consumer fences
            err = pkt->fenceConsPack(sendBuf);
        }
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call sendHeader() to initiate the message.
LwSciError
IpcDst::sendDisconnect(
    IpcBuffer& sendBuf) noexcept
{
    // No data to send
    static_cast<void>(sendBuf);

    // Initiate the message. No other data is needed
    return sendHeader(IpcMsg::Disconnect);
}

} // namespace LwSciStream
