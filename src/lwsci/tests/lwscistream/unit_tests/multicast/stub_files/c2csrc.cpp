//! \file
//! \brief LwSciStream C2C source block definition.
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
#include <cmath>
#include <functional>
#include <iostream>
#include <utility>
#include <array>
#include <unordered_map>
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
#include "elements.h"
#include "packet.h"
#include "enumbitset.h"
#include "block.h"
#include "ipccomm_common.h"
#include "ipcbuffer.h"
#include "ipcbuffer_sciwrap.h"
#include "ipccomm.h"
#include "ipcsrc.h"
#include "c2csrc.h"
#include "lwscibuf_c2c_internal.h"

namespace LwSciStream {

//
// Packet payload queue management
//
C2CPacket::C2CPacket(
    LwSciStreamPacket paramHandle,
    Elements const& paramElements,
    IpcBuffer& recvBuf) noexcept :
    initSuccess(false),
    handle(paramHandle),
    elemCount(paramElements.sizePeek()),
    elemBuffers(FillMode::IPC, false),
    statusEvent(true)
{
    if (LwSciError_Success != elemBuffers.sizeInit(elemCount)) {
        return;
    }

    if (LwSciError_Success !=
        elemBuffers.unpack(recvBuf, paramElements.elemArrayGet())) {
        return;
    }

    // Initialize C2C buffer handles list
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        c2cBufHandles.resize(elemCount, nullptr);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
    } catch (...) {
        return;
    }

    initSuccess = true;
}

//! <b>Sequence of operations</b>
//! - Call LwSciBufFreeTargetObjIndirectChannelC2c() to free the target handle
//!   of the C2C buffer.
C2CPacket::~C2CPacket(void) noexcept
{
    // Free c2c buf dest handles
    for (size_t i{ 0U }; c2cBufHandles.size() > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != c2cBufHandles[i]) {
            static_cast<void>(
                LwSciBufFreeTargetObjIndirectChannelC2c(c2cBufHandles[i]));
        }
    }
}

bool C2CPacket::pendingStatusEvent(void) noexcept
{
    // Check for pending packet status event
    if (statusEvent) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        statusEvent = false;
        return true;
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1));
    }

    return false;
}

//
// C2C Buffer functions
//

//! <b>Sequence of operations</b>
//! - For all buffer elements, call TrackArray::peek() to get
//!   the buffer element and call Wrapper::viewVal() to retrieve the
//!   LwSciBufObj.
//! - Call LwSciBufRegisterTargetObjIndirectChannelC2c() to register
//!   the LwSciBufObj with the C2C service and saves the target handle.
LwSciError
C2CPacket::registerC2CBufTargetHandles(
    LwSciC2cHandle const channelHandle) noexcept
{
    assert(elemBuffers.sizeGet() == c2cBufHandles.size());

    for (size_t i{ 0U }; c2cBufHandles.size() > i; ++i) {
        auto const bufObj{ elemBuffers.peek(i) };
        if (LwSciError_Success != bufObj.first) {
            return bufObj.first;
        }

        // Register with C2C service
        LwSciC2cBufTargetHandle targetHandle;
        LwSciError const err{
            LwSciBufRegisterTargetObjIndirectChannelC2c(
                channelHandle,
                bufObj.second->viewVal(),
                &targetHandle)
        };
        if (LwSciError_Success != err) {
            return err;
        }

        // Save the C2C buffer source handle
        c2cBufHandles[i] = targetHandle;
    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//!   - Marks the new packet instance enqueued by calling the swapQueued()
//!     interface of the packet instance.
//!   - Swaps the next pointer of the new packet instance with the tail of
//!     PayloadQ.
//!   - Swaps the previous pointer of the packet instance at tail if any with
//!     the pointer to the new packet instance.
//!   - Makes the new packet instance the new tail of the PayloadQ. If the head
//!     of PayloadQ is not set, sets it to the new packet instance as well.
void
C2CPacket::PayloadQ::enqueue(
    C2CPacketPtr const&  newPacket) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Mark as in queue
    static_cast<void>(newPacket->swapQueued(true));

    // Packet points to previous tail and vice versa
    static_cast<void>(newPacket->swapNext(tail));
    if (nullptr != tail) {
        static_cast<void>(tail->swapPrev(newPacket));
    }

    // Set tail and maybe head
    tail = newPacket;
    if (nullptr == head) {
        head = newPacket;
    }

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//!   - Marks the packet instance at head dequeued by calling the swapQueued()
//!     interface of the packet instance.
//!   - Sets the new head to the previous pointer of the removed packet
//!     instance and clears its previous pointer by calling the swapPrev()
//!     interface of the removed packet instance.
//!   - Sets the previous pointer of the new head with
//!     the new packet instance.
//!   - If PayloadQ is not empty, clears the next pointer of the new head by
//!     calling the swapNext() interface of this packet instance. Otherwise,
//!     sets the tail of PayloadQ to NULL.
C2CPacketPtr
C2CPacket::PayloadQ::dequeue(void) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Get head, and update queue if not null
    C2CPacketPtr const getPacket {
        head
    };
    if (nullptr != getPacket) {

        // Mark packet as dequeued
        static_cast<void>(getPacket->swapQueued(false));

        // Update head pointer and clear retrieved packet's previous pointer
        head = getPacket->swapPrev(C2CPacketPtr(nullptr));

        // Either clear new head's next pointer or clear tail pointer
        if (nullptr != head) {
            static_cast<void>(head->swapNext(C2CPacketPtr(nullptr)));
        } else {
            tail = nullptr;
        }
    }

    return getPacket;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! \brief Swaps payloadPrev with the @a newPrev.
//!
//! \param [in] newPrev: New smart pointer to a packet instance which needs to
//!   be swapped.
//!
//! \return Old value of the previous pointer.
C2CPacketPtr
C2CPacket::swapPrev(
    C2CPacketPtr const& newPrev) noexcept
{
    C2CPacketPtr const oldPrev{ payloadPrev };
    payloadPrev = newPrev;
    return oldPrev;
}

//! \brief Swaps payloadNext with the @a newNext.
//!
//! \param [in] newNext: New smart pointer to a packet instance which needs to
//!   be swapped.
//!
//! \return Old value of the next pointer.
C2CPacketPtr
C2CPacket::swapNext(
    C2CPacketPtr const& newNext) noexcept
{
    C2CPacketPtr const oldNext{ payloadNext };
    payloadNext = newNext;
    return oldNext;
}

//! \brief Sets payloadQueued with the @a newQueued to indicate whether
//!   the packet instance is in the PayloadQ or not.
//!
//! \param [in] newQueued: New value of queued flag.
//!
//! \return Old value of queued flag.
bool
C2CPacket::swapQueued(
    bool const newQueued) noexcept
{
    bool const oldQueued{ payloadQueued };
    payloadQueued = newQueued;
    return oldQueued;
}

//
// C2CSrc Block
//

//! <b>Sequence of operations</b>
//! - Initialize the base IpcSrc class and all members.
//! - Call LwSciSyncAttrListCreate() to create attributes lists for cpu
//!   and c2s waiting and signalling, and use them to create instances
//!   of LwSciWrap::SyncAttr.
//! - Call Wrapper::getErr() to check for any failures in allocation.
//! - Call Wrapper::viewVal() to retrieve the attribute lists and
//!   LwSciSyncAttrListSetAttrs() to populate their contents.
//! - Call LwSciSyncCpuWaitContextAlloc() to allocate a conext for CPU waits.
//! - Call IpcSrc::enqueueIpcWrite() to signal waiting attribute message.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A12_1_5), "Bug 2989775")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A3_1_1), "Bug 2758535")
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
C2CSrc::C2CSrc(LwSciIpcEndpoint const ipc,
               LwSciSyncModule const syncModule,
               LwSciBufModule const bufModule) noexcept :
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A3_1_1))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A12_1_5))
LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    IpcSrc(ipc, syncModule, bufModule, true),
    c2cChannel(nullptr),
    service(nullptr),
    cpuWaiterAttr(),
    cpuSignalAttr(),
    c2cWaiterAttr(),
    c2cSignalAttr(),
    engineDoneWaiterAttr(),
    copyDoneSignalAttr(),
    readDoneSignalAttr(),
    copyDoneAttr(),
    copyDoneSignalAttrMsg(false),
    cpuWaitAfterCopy(false),
    waiterAttrMsg(false),
    c2cSignalConsHandle(nullptr),
    c2cSignalProdHandle(nullptr),
    waitContext(nullptr),
    c2cPktMap(),
    c2cPayloadQueue(),
    numPayloadsAvailable(0U),
    numC2CPktsAvailable(0U),
    numPayloadsPendingWriteSignal(0U),
    numC2CPktsForWriteSignal(0U)
{
    // TODO: When C2C is complete, we may not need all of these attribute
    //       lists on both Src and Dst sides. We can trim some out if
    //       needed. For now, it is easiest to provide all of them for
    //       symmetry, and doing so will make it easier to consolidate
    //       duplicate Src/Dst code.

    // Create and wrap a CPU waiting attribute list.
    LwSciSyncAttrList attrList;
    LwSciError err { LwSciSyncAttrListCreate(syncModule, &attrList) };
    cpuWaiterAttr = LwSciWrap::SyncAttr(attrList, true, false, err);

    // Create and wrap a CPU signalling attribute list.
    err = LwSciSyncAttrListCreate(syncModule, &attrList);
    cpuSignalAttr = LwSciWrap::SyncAttr(attrList, true, false, err);

    // Create and wrap a C2C waiting attribute list.
    err = LwSciSyncAttrListCreate(syncModule, &attrList);
    c2cWaiterAttr = LwSciWrap::SyncAttr(attrList, true, false, err);

    // Create and wrap a C2C signalling attribute list.
    err = LwSciSyncAttrListCreate(syncModule, &attrList);
    c2cSignalAttr = LwSciWrap::SyncAttr(attrList, true, false, err);

    // Abort if any allocations failed
    if (((LwSciError_Success != cpuWaiterAttr.getErr()) ||
         (LwSciError_Success != cpuSignalAttr.getErr())) ||
        ((LwSciError_Success != c2cWaiterAttr.getErr()) ||
         (LwSciError_Success != c2cSignalAttr.getErr()))) {
        setInitFail();
        return;
    }

    // Attributes values
    bool const cpuAccess { true };
    LwSciSyncAccessPerm const waiterPerm { LwSciSyncAccessPerm_WaitOnly };
    LwSciSyncAccessPerm const signalPerm { LwSciSyncAccessPerm_SignalOnly };
    LwSciSyncAttrKeyValuePair pairCpuAccess
        { LwSciSyncAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess) };
    LwSciSyncAttrKeyValuePair pairWaiterPerm
        { LwSciSyncAttrKey_RequiredPerm, &waiterPerm, sizeof(waiterPerm) };
    LwSciSyncAttrKeyValuePair pairSignalPerm
        { LwSciSyncAttrKey_RequiredPerm, &signalPerm, sizeof(signalPerm) };
    std::array<LwSciSyncAttrKeyValuePair, 2U> pairs;

    // Fill CPU waiter attribute list
    pairs[0] = pairCpuAccess;
    pairs[1] = pairWaiterPerm;
    if (LwSciError_Success !=
        LwSciSyncAttrListSetAttrs(cpuWaiterAttr.viewVal(),
                                  pairs.data(), pairs.size())) {
        setInitFail();
    }

    // Fill CPU signaller attribute list
    pairs[0] = pairCpuAccess;
    pairs[1] = pairSignalPerm;
    if (LwSciError_Success !=
        LwSciSyncAttrListSetAttrs(cpuSignalAttr.viewVal(),
                                  pairs.data(), pairs.size())) {
        setInitFail();
    }

    // Fill C2C waiter attribute list
    if (LwSciError_Success !=
        LwSciSyncFillAttrsIndirectChannelC2c(ipc,
                                             c2cWaiterAttr.viewVal(),
                                             LwSciSyncAccessPerm_WaitOnly)) {
        setInitFail();
    }

    // Fill C2C signaller attribute list
    if (LwSciError_Success !=
        LwSciSyncFillAttrsIndirectChannelC2c(ipc,
                                             c2cSignalAttr.viewVal(),
                                             LwSciSyncAccessPerm_SignalOnly)) {
        setInitFail();
    }

    // Allocate context for CPU waiting
    if (LwSciError_Success !=
        LwSciSyncCpuWaitContextAlloc(syncModule, &waitContext)) {
        setInitFail();
    }

    // C2CSrc uses CPU waits for producer to finish writing
    // TODO: This is temporary until LwSciEvent support is in place.
    //       Then C2C will handle the waiting.
#if !C2C_EVENT_SERVICE
    engineDoneWaiterAttr = cpuWaiterAttr;
#else
    engineDoneWaiterAttr = c2cWaiterAttr;
#endif

    // C2CSrc uses C2C to signal to C2CDst that copy is done
    copyDoneSignalAttr = c2cSignalAttr;
    copyDoneSignalAttrMsg = true;

    // Signal attributes will be sent ASAP after connection
    if (LwSciError_Success != enqueueIpcWrite()) {
        setInitFail();
    }
}

//! <b>Sequence of operations</b>
//!  - Call IpcSrc::dequeueAll() and Block::clearPacketMap() to remove all
//!    packetsf from readyPayloadQueue.
//!  - Removes all packets from c2cPayloadQueue by calling
//!    C2CPacket::PayloadQ::dequeue() interface of the c2cPayloadQueue object.
//!  - Call LwSciSyncFreeObjIndirectChannelC2c() to free the
//!    LwSciC2cSyncHandle(s).
//!  - Call LwSciBufCloseIndirectChannelC2c() to close the C2C channel.
//!  - Call LwSciEventService::Delete() to releases resources associated with
//!    the event service.
C2CSrc::~C2CSrc(void) noexcept
{
    // Remove packets from readyPayloadQueue and free source buffer handles
    dequeueAll();
    clearPacketMap();

    // Remove packets from c2cPayloadQueue and free destination buffer handles
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A6_5_3), "LwSciStream-ADV-AUTOSARC++14-004")
    C2CPacketPtr oldPacket {};
    do {
        oldPacket = c2cPayloadQueue.dequeue();
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    } while (nullptr != oldPacket);
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A6_5_3))

    c2cPktMap.clear();

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    // Free the LwSciC2cSyncHandle(s).
    // TODO: when creating the c2c sync handles, wrap them
    //       in some management wrapper.
    if (nullptr != c2cSignalConsHandle) {
        static_cast<void>(
            LwSciSyncFreeObjIndirectChannelC2c(c2cSignalConsHandle));
    }
    if (nullptr != c2cSignalProdHandle) {
        static_cast<void>(
            LwSciSyncFreeObjIndirectChannelC2c(c2cSignalProdHandle));
    }
    for (size_t i{ 0U }; c2cWaitProdEngineHandle.size() > i; i++) {
        if (nullptr != c2cWaitProdEngineHandle[i]) {
            static_cast<void>(
                LwSciSyncFreeObjIndirectChannelC2c(c2cWaitProdEngineHandle[i]));
        }
    }

    // Close c2c channel
    if (nullptr != c2cChannel) {
        static_cast<void>(
            LwSciBufCloseIndirectChannelC2c(c2cChannel));
    }

    // Releases resources associated with the event service
    if (nullptr != service) {
        service->EventService.Delete(&service->EventService);
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//!   - Retrieves the BlockType of the given block
//!     instance by calling Block::getBlockType() interface
//!     and validates whether the returned BlockType is
//!     BlockType::QUEUE.
//!   - Retrieves the handle to C2CSrc block by calling
//!     Block::getHandle() interface and retrieves the C2CSrc
//!     block instance by calling Block::getRegisteredBlock()
//!     interface.
//!   - Initializes the destination connection of the queue block instance
//!     by calling Block::connDstInitiate().
//!   - If successful, initializes the source connection of the
//!     C2CSrc block instance by calling Block::connSrcInitiate().
//!   - If initialization of source connection is not successful
//!     then cancels the destination connection of the queue block instance
//!     by calling Block::connDstCancel(). Otherwise, completes the
//!     destination connection of queue block instance and source
//!     connection of the C2CSrc block instance by calling the
//!     Block::connDstComplete() and Block::connSrcComplete() interfaces
//!     respectively.
//!
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_11), "Bug 2738197")
LwSciError
C2CSrc::BindQueue(BlockPtr const& paramQueue) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_11))
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if ((nullptr == paramQueue) ||
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_5_1), "Bug 2807673")
        (BlockType::QUEUE != paramQueue->getBlockType())) {
        return LwSciError_BadParameter;
    }

    // Note: We get the shared_ptr from the registry rather than
    //       creating a new one from the this pointer
    BlockPtr const thisPtr {Block::getRegisteredBlock(getHandle())};

    // Reserve connections
    IndexRet const dstReserved { paramQueue->connDstInitiate(thisPtr) };
    if (LwSciError_Success != dstReserved.error) {
        return dstReserved.error;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    IndexRet const srcReserved { connSrcInitiate(paramQueue) };
    if (LwSciError_Success != srcReserved.error) {
        paramQueue->connDstCancel(dstReserved.index);
        return srcReserved.error;
    }

    // Finalize connections
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    paramQueue->connDstComplete(dstReserved.index, srcReserved.index);
    connSrcComplete(srcReserved.index, dstReserved.index);
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    queue = paramQueue;

    return LwSciError_Success;
}

// C2CSrc block retrieves the associated queue block object.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError C2CSrc::getInputConnectPoint(
    BlockPtr& paramBlock) const noexcept
{
    paramBlock = queue;
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    return isInitSuccess() ? LwSciError_Success : LwSciError_NotInitialized;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))


//! <b>Sequence of operations</b>
//! - Call LwSciEventLoopServiceCreate() to create an event service
//! - Call Block::finalizeConfigOptions() to lock other options
void
C2CSrc::finalizeConfigOptions(void) noexcept
{
    // Create the event service used by c2c channel
    // if not provided by the application.
    // TODO: config it to use event service
    LwSciError const err{ LwSciEventLoopServiceCreate(1U, &service) };

    // Call the base function to finalize other options
    if (LwSciError_Success == err) {
        Block::finalizeConfigOptions();
    } else {
        setErrorEvent(err);
    }
}

// Searches C2CPacketMap for a packet instance.
C2CPacketPtr
C2CSrc::c2cPktFind(
    LwSciStreamPacket const paramHandle,
    bool const locked) noexcept
{
    // If lock is not already held, take it
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    Lock const blockLock { blkMutexLock(locked) };

    C2CPacketPtr pkt {};
    C2CPacketMap::iterator const iter { c2cPktMap.find(paramHandle) };
    if (c2cPktMap.cend() != iter) {
        pkt = iter->second;
    }
    return pkt;
}

LwSciError
C2CSrc::c2cPktCreate(
     LwSciStreamPacket const paramHandle,
     IpcBuffer& recvBuf) noexcept
{
    // Get the lock
    Lock const blockLock { blkMutexLock() };

    if (nullptr != c2cPktFind(paramHandle, true)) {
        // TODO: Better error
        return LwSciError_BadParameter;
    }

    try {
        C2CPacketPtr const pkt {
            std::make_shared<C2CPacket>(paramHandle,
                                        allocatedElemGet(),
                                        recvBuf)
        };

        if (!pkt->isInitSuccess()) {
            return LwSciError_StreamInternalError;
        }

        auto const result {
            c2cPktMap.emplace(paramHandle, pkt) };
        if (result.second) {
            return LwSciError_Success;
        }

        // See if failure is because handle already existed in map.
        //   This means we did something wrong.
        if (nullptr != c2cPktFind(paramHandle, true)) {
            return LwSciError_StreamInternalError;
        }
    } catch (std::bad_alloc& e) {
        // Packet creation may throw bad_alloc.
        static_cast<void>(e);
    }

    return LwSciError_InsufficientMemory;;
}

void
C2CSrc::c2cPktRemove(LwSciStreamPacket const paramHandle) noexcept
{
    // Take lock
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    Lock const blockLock { blkMutexLock() };

    // Erase from map
    size_t const size { c2cPktMap.erase(paramHandle) };
    static_cast<void>(size);
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
    assert(1ULL == size);
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
}


} // namespace LwSciStream
