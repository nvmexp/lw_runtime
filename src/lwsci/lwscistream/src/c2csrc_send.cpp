//! \file
//! \brief LwSciStream C2C source block transmission.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistream_types.h"
#include "c2csrc.h"
#include "ipcbuffer.h"
#include "ipcbuffer_sciwrap.h"
#include "lwscibuf_c2c_internal.h"
#include "lwscievent.h"
#include "sciwrap.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Call Elements::sizePeek() to retrieve the number of elements.
//! - Call Elements::attrPeek() to retrieve the LwSciBuf attribute list
//!   of each element.
//! - Call LwSciBufAttrListGetAttrs() to query the buffer type and buffer
//!   size from the LwSciBuf attribute list and saves the buffer size for
//!   each element.
//! - Call LwSciBufOpenIndirectChannelC2c() to open the C2C channel.
//! - Call IpcSrc::srcRecvAllocatedElements() to send the allocated elements
//!   downstream.
void C2CSrc::srcRecvAllocatedElements(
    uint32_t const dstIndex,
    Elements const& inElements) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation{};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Retrieve the number of elements
    size_t const elemCount { inElements.sizePeek() };

    // Query and save the buffer size in each element,
    // which will be used when triggering c2c copy
    try {
        bufSize.resize(elemCount, 0UL);
    } catch(...){
        setErrorEvent(LwSciError_InsufficientMemory);
        return;
    }

    for (uint32_t i{ 0U }; elemCount > i; i++) {
        auto const attrEntry{ inElements.attrPeek(i) };
        if (LwSciError_Success != attrEntry.first) {
            setErrorEvent(attrEntry.first);
            return;
        }

        // Query buffer type
        std::array<LwSciBufAttrKeyValuePair, 2U> bufAttr;
        bufAttr[0U] = {LwSciBufGeneralAttrKey_Types, nullptr, 0U };

        LwSciError bufErr{
            LwSciBufAttrListGetAttrs(attrEntry.second, bufAttr.data(), 1U)
        };
        if (LwSciError_Success != bufErr) {
            setErrorEvent(bufErr);
            return;
        }

        LwSciBufType const bufType{
            *(static_cast<const LwSciBufType *>(bufAttr[0U].value))
        };

        // Query buffer size according to bufer type
        size_t const pairCount{ (bufType == LwSciBufType_Pyramid) ? 2U : 1U };
        switch (bufType)
        {
        case LwSciBufType_RawBuffer:
            bufAttr[0U] = { LwSciBufRawBufferAttrKey_Size, nullptr, 0U};
            break;
        case LwSciBufType_Image:
            bufAttr[0U] = { LwSciBufImageAttrKey_Size, nullptr, 0U };
            break;
        case LwSciBufType_Tensor:
            bufAttr[0U] = { LwSciBufTensorAttrKey_Size, nullptr, 0U };
            break;
        case LwSciBufType_Array:
            bufAttr[0U] = { LwSciBufArrayAttrKey_Size, nullptr, 0U };
            break;
        case LwSciBufType_Pyramid:
            bufAttr[0U] = { LwSciBufPyramidAttrKey_NumLevels, nullptr, 0U };
            bufAttr[1U] = { LwSciBufPyramidAttrKey_LevelSize, nullptr, 0U };
            break;

        default:
            setErrorEvent(LwSciError_NotSupported);
            return;
        }

        bufErr = LwSciBufAttrListGetAttrs(attrEntry.second,
                                          bufAttr.data(),
                                          pairCount);
        if (LwSciError_Success != bufErr) {
            setErrorEvent(bufErr);
            return;
        }

        if (bufType == LwSciBufType_Pyramid) {
            uint32_t const numLevels{
                *(static_cast<const uint32_t *>(bufAttr[0U].value)) };
            uint64_t const * const levelSize{
                static_cast<const uint64_t*>(bufAttr[1U].value) };

            bufSize[i] = 0U;
            for (uint32_t j{ 0U }; numLevels > j; j++) {
                if (bufSize[i] >
                    (std::numeric_limits<uint64_t>::max() - levelSize[j])) {
                    setErrorEvent(LwSciError_Overflow);
                    return;
                }
                bufSize[i] += levelSize[j];
            }
        } else {
            bufSize[i] = *(static_cast<const uint64_t *>(bufAttr[0U].value));
        }
    }

    // Init c2cChannel for C2C operations
    LwSciError const err{
        LwSciBufOpenIndirectChannelC2c(
                getIpcEndpoint(),
                &service->EventService,
                elemCount, // numRequests
                elemCount, // # flush ranges per request
#if (C2C_EVENT_SERVICE)
                elemCount, // # preFences per request
#else // (C2C_EVENT_SERVICE)
                0U,        // # preFences per request
#endif // (C2C_EVENT_SERVICE)
                2U,        // # postFences per request
                &c2cChannel)
    };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Call base function to save and transmit message
    IpcSrc::srcRecvAllocatedElements(dstIndex, inElements);

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Block::pktCreate() to create a new packet, copy the definition,
//!   and insert it in the map.
//! - Call Block::pktFindByHandle() to retrieve the newly created packet.
//! - Call Packet::statusConsSet() to set the status.
//! - Call dstRecvPacketStatus() on the source connection to send the status
//!   upstream.
void C2CSrc::srcRecvPacketCreate(
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

    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    // Create new packet, copy definition, and insert in map.
    LwSciError err { pktCreate(origPacket.handleGet(), &origPacket) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(M0_1_2), "Bug 3003226")
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        setErrorEvent(err);
        return;
    }

    // Retrieve the newly created packet
    PacketPtr const pkt { pktFindByHandle(origPacket.handleGet()) };
    if (nullptr == pkt) {
        setErrorEvent(LwSciError_StreamInternalError);
        return;
    }

    // Register the buffer with C2C service.
    err = pkt->registerC2CBufSourceHandles(c2cChannel);
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Set the status
    //   The handle is used as the cookie, to facilitate deletion.
    err = pkt->statusConsSet(err, origPacket.handleGet());
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))

    // Send the status upstream
    getSrc().dstRecvPacketStatus(*pkt);
}

//! <b>Sequence of operations</b>
//! - Call Block::elementCountGet() to retrieve count for number of syncs.
//! - For all elements, call Waiters::attrPeek() and Wrapper::viewVal() to
//!   retrieve the attributes lists from the producer and combine into a
//!   vector.
//! - If any of the attributes are NULL, set the flag indicating that we
//!   need to do CPU waits after copies, clear out the vector and instead
//!   call Wrapper::viewVal() to retrieve the CPU waiter attributes and
//!   insert it in the vector.
//! - Call Wrapper::viewVal() to retrieve the copy done signaller attribute
//!   list and add it to the vector.
//! - Call LwSciSyncAttrListReconcile() to reconcile the attributes.
//! - Call enqueueIpcWrite to inform dispatch thread of attribute message.
void C2CSrc::srcRecvSyncWaiter(
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

    // Check whether this has already been done
    if (nullptr != copyDoneAttr.viewVal()) {
        setErrorEvent(LwSciError_AlreadyDone);
        return;
    }

    // Retrieve element count
    size_t const elemCount { elementCountGet() };

    // Construct a list of all necessary attributes for a sync object
    //   to signal locally that C2C copy is done
    std::vector<LwSciSyncAttrList> unreconciledList;
    try {
        // Try to combine all of the producer's waiter requirements,
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

    // Trigger flow of waiter attributes downstream
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
//!   recnciled attributes.
//! - Create a new LwSciWrap::SyncObj that owns the C2C LwSciSyncObj.
//! - Registers the C2C LwSciSyncObj by calling
//!   LwSciSyncRegisterSignalObjIndirectChannelC2c().
//! - Registers the producer LwSciSyncObj by calling
//!   LwSciSyncRegisterWaitObjIndirectChannelC2c().
//! - Create a temporary Signals instance and call Signals::sizeInit()
//!   and either Signals::syncFill() or Signals::doneSet() to fill it with
//!   copies of the sync object or leave it empty.
//! - Call srcRecvSyncSignal() interface of the source block to send the
//!   C2C sync object info upstream.
//! - Call phaseProdSyncDoneSet() and phaseConsSyncDoneSet() to advance setup.
void C2CSrc::srcRecvSyncSignal(
    uint32_t const srcIndex,
    Signals const& syncSignal) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Producer's signalling sync objects are not transmitted or used directly
    //   (The fences from them are used, but the sync objects aren't needed)
    static_cast<void>(syncSignal);

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

    // Retrieve consumer and element counts
    size_t const consCount { consumerCountGet() };
    size_t const elemCount { elementCountGet() };

    // Create a new sync object and wrap it
    LwSciSyncObj c2cSyncObj;
    LwSciError err { LwSciSyncObjAlloc(copyDoneAttr.viewVal(), &c2cSyncObj) };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A7_1_1), "Bug 3258479")
    LwSciWrap::SyncObj c2cSyncObjWrap { c2cSyncObj, true };

    // Register the sync object with C2C
    err = LwSciSyncRegisterSignalObjIndirectChannelC2c(c2cChannel,
                                                       c2cSyncObjWrap.viewVal(),
                                                       &c2cSignalProdHandle);
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

#if (C2C_EVENT_SERVICE)
    // Allocate the c2cWaitProdEngineHandle array
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        c2cWaitProdEngineHandle.resize(elemCount, nullptr);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
    } catch (...) {
        setErrorEvent(LwSciError_InsufficientMemory);
        return;
    }

    //Register the sync objects received from the producer with the C2C
    for (size_t i {0U}; elemCount > i; ++i) {
        if (nullptr != syncSignal.syncPeek(i)) {

            err = LwSciSyncRegisterWaitObjIndirectChannelC2c(c2cChannel,
                                                         syncSignal.syncPeek(i),
                                                         &c2cWaitProdEngineHandle[i]);
            if (LwSciError_Success != err) {
                setErrorEvent(err);
                return;
            }
        }
    }
#endif // (C2C_EVENT_SERVICE)

    // Fill a Signals object with the sync object or empty wrappers, depending
    //   on whether the producer or the C2C object will handle the wait
    Signals c2cSyncSignal(FillMode::User, nullptr);
    err = c2cSyncSignal.sizeInit(consCount, elemCount);
    if (LwSciError_Success == err) {
        if (cpuWaitAfterCopy) {
            err = c2cSyncSignal.doneSet();
        } else {
            err = c2cSyncSignal.syncFill(c2cSyncObjWrap);
        }
    }
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Send C2C signal objects upstream
    getSrc().dstRecvSyncSignal(c2cSyncSignal);

    // Advance setup phase
    phaseProdSyncDoneSet();
    phaseConsSyncDoneSet();

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Increment count of available payloads.
//! - Call enqueueIpcWrite() to wake dispatch thread.
void C2CSrc::srcRecvPayload(
    uint32_t const srcIndex,
    Packet const& prodPayload) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Input value is ignored
    static_cast<void>(prodPayload);

    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation { };
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Increment the number of payloads available
    numPayloadsAvailable++;

    // Need to take a lock to synchronizae the action
    // of C2CSrc::c2cRecvPayload
    {
        Lock const blockLock { blkMutexLock() };

        if (0U == numC2CPktsForWriteSignal) {
            if (UINT32_MAX == numPayloadsPendingWriteSignal) {
                setErrorEvent(LwSciError_InsufficientResource, true);
                return;
            }
            ++numPayloadsPendingWriteSignal;
            return;
        }
        --numC2CPktsForWriteSignal;
    }

    // Reaching this point means there is payload pending to be signal for
    // IPC write.
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
//! -- Loop over all C2C packets, calling Packet::pendingStatusEvent().
//!    If one returns true, call c2cPackPacketStatus() and return.
//! -- If there is a pending sync waiter attribute message, call
//!    c2cSendSyncWaiterAttr() and return.
//! - If runtime phase:
//! -- If there is both a payload pending and an available C2C packet,
//!    call c2cSendPayload() and return.
//! - Call IpcSrc::sendMessage() for any common IPC/C2C messages.
LwSciError
C2CSrc::sendMessage(
    IpcBuffer& sendBuf,
    Lock& blockLock) noexcept
{
    // Init phase messages
    if (!runtimeEnabled()) {

        // Check for signal attribute message
        if (copyDoneSignalAttrMsg) {
            copyDoneSignalAttrMsg = false;
            return c2cSendSyncSignalAttr(sendBuf);
        }

        // Check all C2C packets for status event
        // TODO: Have a flag to indicate status has been sent for all packets
        //       so we can do a quick check instead of looping every time
        for (C2CPacketMap::iterator x { c2cPktMap.begin() };
             c2cPktMap.end() != x; ++x) {
            C2CPacketPtr const pkt { x->second };
            if (pkt->pendingStatusEvent()) {
                return c2cSendPacketStatus(sendBuf, pkt);
            }
        }

        // Check for signal attribute message
        if (waiterAttrMsg) {
            waiterAttrMsg = false;
            return c2cSendSyncWaiterAttr(sendBuf);
        }
    }

    // Runtime phase messages
    else {

        // If there is both at least one payload waiting to send and at least
        //   one C2C packet to hold it, send the payload
        if ((0U < numPayloadsAvailable) && (0U < numC2CPktsAvailable)) {
            return c2cSendPayload(sendBuf, blockLock);
        }

    }

    // Call IpcSrc function for common IPC/C2C messages
    return IpcSrc::sendMessage(sendBuf, blockLock);
}

//! <b>Sequence of operations</b>
//! - Call IpcSrc::sendHeader() to initiate the message.
//! - Call C2CPacket::handleGet() to retrieve the packet's handle.
//! - Call IpcBuffer::packVal() to pack the handle.
//! - Call IpcBuffer::packVal() to pack the status.
LwSciError
C2CSrc::c2cSendPacketStatus(
    IpcBuffer& sendBuf,
    C2CPacketPtr const& pkt) noexcept
{
    // Pack message header
    LwSciError err { sendHeader(IpcMsg::C2CPacketStatus) };
    if (LwSciError_Success == err) {
        // Pack handle
        err = sendBuf.packVal(pkt->handleGet());
        if (LwSciError_Success == err) {
            // Pack success status
            // TODO: We should track and return any errors in
            //       importing or registering packet.
            err = sendBuf.packVal(LwSciError_Success);
        }
    }
    return err;
}

//! <b>Sequence of operations</b>
//! - Call IpcSrc::sendHeader() to initiate the message.
//! - Call ipcBufferPack(LwSciWrap::SyncAttr) to pack the sync attributes into
//!   sendBuffer.
LwSciError
C2CSrc::c2cSendSyncSignalAttr(
    IpcBuffer& sendBuf) noexcept
{
    // Pack message header
    LwSciError err { sendHeader(IpcMsg::SignalAttr) };
    if (LwSciError_Success == err) {
        // The C2CSrc side can provide a signal when the copy finishes.
        err = ipcBufferPack(sendBuf, copyDoneSignalAttr);
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Call sendHeader() to initiate the message.
LwSciError
C2CSrc::c2cSendSyncWaiterAttr(
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
//! - Call C2CPayloadQ::dequeue() to obtain an available C2C packet.
//! - Call the dstDequeuePayload() interface of the upstream (queue) block
//!   to obtain a waiting payload.
//! - Call Packet::handleGet() to obtain the handle for the payload.
//! - Call Block::pktFindByHandle() to look up the local packet for the handle.
//! - Call Packet::locationUpdate() to validate and update the packet location.
//! - Call Packet::fenceProdCopy() to copy the fences into the local packet.
//! - For each element in the packet:
//! -- If the producer provides a sync object for the element:
//! --- Call Packet::fenceProdGet() and LwSciWrap::SyncFence::viewVal() to
//!     extract the producer fences for the element.
//! --- Call LwSciBufPushWaitIndirectChannelC2c() to schedule a C2C wait
//!     for the fence.
//! -- Call LwSciBufPushCopyIndirectChannelC2c() to scedule a C2C copy of
//!    the buffer.
//! - Call Packet::fenceProdReset() to reset the producer fences.
//! - Call LwSciBufPushSignalIndirectChannelC2c() to issue a
//!   C2C-signaling-consumer(s) instruction, and gets a LwSciSyncFence.
//! - Call LwSciBufPushSignalIndirectChannelC2c() to issue a
//!   C2C-signaling-producer instruction, and gets a LwSciSyncFence.
//! - Call LwSciBufPushSubmitIndirectChannelC2c() to initiate the C2C
//!   operations.
//! - Call IpcSrc::sendHeader() to initiate the payload message.
//! - Call C2CPacket::handleGet() to retrieve the C2C packet handle.
//! - Call IpcBuffer::packVal() to pack the handle.
//! - Call ipcBufferPacket() to pack the consumer-signalling C2C fence.
//! - If cpuWaitAfterCopy is set, call LwSciSyncFenceWait() to wait for the
//!   producer-signalling fence. And call Packet::fenceConsDone() to finalize
//!   the consumer fences.
//! - Otherwise, call Packet::fenceConsFill() to fill the packet's consumer
//!   fence list with copies of the producer-signalling fence.
//! - Call the dstRecvPayload() interface of source block to send
//!   packet upstream.
//! - Call Packet::fenceConsReset() to reset consumer fences.
LwSciError
C2CSrc::c2cSendPayload(
    IpcBuffer& sendBuf,
    Lock& blockLock) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Caller made sure both counts were positive, but double check so
    //   CERT doesn't complain
    if ((0U == numC2CPktsAvailable) || (0U == numPayloadsAvailable)) {
        return LwSciError_StreamInternalError;
    }

    // Decrement counter and dequeue available C2C packet
    numC2CPktsAvailable--;
    C2CPacketPtr const c2cPkt{ c2cPayloadQueue.dequeue() };
    if (nullptr == c2cPkt) {
        return LwSciError_NoStreamPacket;
    }

    // Decrement counter and dequeue available payload
    numPayloadsAvailable--;

    // Unlock before obtaining payload from upstream.
    blockLock.unlock();
    PacketPtr const usePayload { getSrc().dstDequeuePayload() };
    if (nullptr == usePayload) {
        return LwSciError_NoStreamPacket;
    }

    // Retake the lock
    blockLock.lock();

    // Look up corresponding local packet
    PacketPtr const pkt { pktFindByHandle(usePayload->handleGet(), true) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }

    // Update the location of local packet
    if (!pkt->locationUpdate(Packet::Location::Upstream,
                             Packet::Location::Downstream)) {
        return LwSciError_StreamPacketInaccessible;
    }

    // Reset consumer fences
    pkt->fenceConsReset();

    // Copy producer fences into packet
    LwSciError err{ pkt->fenceProdCopy(*usePayload) };
#if (!C2C_EVENT_SERVICE)
    if (LwSciError_Success == err) {
        // Do CPU wait
        err = pkt->fenceProdWait(waitContext);
    }
#endif // (!C2C_EVENT_SERVICE)
    if (LwSciError_Success != err) {
        return err;
    }

    // For each buffer, issue a C2C PushWait and PushCopy instruction.
    // TODO: Maybe implement a function in Packet for this
    size_t const elemCount { elementCountGet() };
    for (size_t i {0U}; elemCount > i; ++i) {

        // TODO: Do CPU wait for now until switching to event service,
        // which can handle c2c callback.
#if (C2C_EVENT_SERVICE)
        // Issue PushWait for non-synchronous elements.
        if (nullptr != c2cWaitProdEngineHandle[i]) {
            LwSciWrap::SyncFence prodEngineFenceWrap;
            err = pkt->fenceProdGet(i, prodEngineFenceWrap);
            if (LwSciError_Success != err) {
                return err;
            }

            // Note: Need to copy out the fence because the function takes
            //       a non-const pointer.
            LwSciSyncFence prodEngineFence { prodEngineFenceWrap.viewVal() };
            err = LwSciBufPushWaitIndirectChannelC2c(
                                              c2cChannel,
                                              c2cWaitProdEngineHandle[i],
                                              &prodEngineFence);
            if (LwSciError_Success != err) {
                return err;
            }
        }
#endif // (C2C_EVENT_SERVICE)

        // Issue PushCopy
        LwSciBufFlushRanges const flushRanges{ 0UL, bufSize[i] };
        size_t const numFlushRanges{ 1U };
        err = LwSciBufPushCopyIndirectChannelC2c(
                c2cChannel,
                pkt->c2cBufSourceHandleGet(i),
                c2cPkt->c2cBufTargetHandleGet(i),
                &flushRanges,
                numFlushRanges);
        if (LwSciError_Success != err) {
            return err;
        }
    }

    //Reset prod fences
    pkt->fenceProdReset();

    // Add a signaling instruction informing remote endpoints c2c copy is done.
    LwSciSyncFence c2cDoneConsFence { LwSciSyncFenceInitializer };
    err = LwSciBufPushSignalIndirectChannelC2c(
            c2cChannel,
            c2cSignalConsHandle,
            &c2cDoneConsFence);
    if (LwSciError_Success != err) {
        return err;
    }

    // Wrap the fence for management
    LwSciWrap::SyncFence const consFenceWrap { c2cDoneConsFence, true };

    // Add a signaling instruction informing local endpoint c2c copy is done.
    LwSciSyncFence c2cDoneProdFence { LwSciSyncFenceInitializer };
    err = LwSciBufPushSignalIndirectChannelC2c(
            c2cChannel,
            c2cSignalProdHandle,
            &c2cDoneProdFence);
    if (LwSciError_Success != err) {
        return err;
    }

    // Wrap the fence for management
    LwSciWrap::SyncFence const prodFenceWrap { c2cDoneProdFence, true };

    // Add a C2C PushSubmit instruction to initiate the copies
    err = LwSciBufPushSubmitIndirectChannelC2c(c2cChannel);
    if (LwSciError_Success != err) {
        return err;
    }

    // Initiate the message
    err = sendHeader(IpcMsg::C2CPayload);
    if (LwSciError_Success != err) {
        return err;
    }

    // Pack the c2c packet handle
    err = sendBuf.packVal(c2cPkt->handleGet());
    if (LwSciError_Success != err) {
        return err;
    }

    // Pack the C2C copy fence
    err = ipcBufferPack(sendBuf, consFenceWrap);
    if (LwSciError_Success != err) {
        return err;
    }

    // If producer does not support sync objects, do a CPU wait
    if (cpuWaitAfterCopy) {
        // TODO: Use LwSciEvent to defer this, waking again and returning
        //       the packet upstream when the copy is done
        err = LwSciSyncFenceWait(&c2cDoneProdFence, waitContext,
                                 INFINITE_TIMEOUT);
        if (LwSciError_Success != err) {
            return err;
        }

        // Mark fence setup done
        pkt->fenceConsDone();
    }

    // Otherwise, fill the packet with duplicates of the fence and return
    else {
        err = pkt->fenceConsFill(prodFenceWrap);
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // Update the location of local packet
    if (!pkt->locationUpdate(Packet::Location::Downstream,
                             Packet::Location::Upstream)) {
        return LwSciError_StreamPacketInaccessible;
    }

    // Unlock before sending payload upstream
    blockLock.unlock();
    getSrc().dstRecvPayload(*pkt);

    return LwSciError_Success;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

} // namespace LwSciStream
