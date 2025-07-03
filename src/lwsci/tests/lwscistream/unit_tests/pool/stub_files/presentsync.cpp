//! \file
//! \brief LwSciStream PresentSync class declaration.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.
#include <cstdint>
#include <iostream>
#include <array>
#include <unordered_map>
#include <memory>
#include <atomic>
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
#include "enumbitset.h"
#include "block.h"
#include "packet.h"
#include "presentsync.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//!   - Calls the constructor of the Block base class with BlockType::PRESENTSYNC
//!
//! \implements{}
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A3_1_1), "Bug 2758535")
PresentSync::PresentSync(LwSciSyncModule const syncModule) noexcept :
LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    Block(BlockType::PRESENTSYNC),
    syncModule(syncModule),
    waiterDone(false),
    fenceWaitQueue(),
    waitContext(nullptr),
    dispatchThread(),
    teardown(false)
{
    // Set up packet description
    Packet::Desc desc { };
    desc.initialLocation = Packet::Location::Upstream;
    desc.fenceProdFillMode = FillMode::Copy;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    pktDescSet(std::move(desc));
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))

    // Launch thread.
    //   The conditional should never fail. It exist only to satisfy the
    //   deviation, guaranteeing that we never reach the path in
    //   std::thread that could abort.
    if (!dispatchThread.joinable()) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR50_CPP), "LwSciStream-ADV-CERTCPP-002")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_5_2), "Approved TID-482")
        dispatchThread = std::thread(&PresentSync::waitForFenceThreadFunc, this);
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR50_CPP))
    }
};

//! \brief I/O Thread loop for handling fence waiting opeations.
//!
//! <b>Sequence of operations</b>
//! - The function checks if there is a packet available to process.
//!   Otherwise, just waits for the notification.
//! - Upon receiving the wake up signal, it checks if the block is going
//!   to be destroyed, then immediately returns from the function. Otherwise,
//!   it dequeues the packet from queue by calling Packet::PayloadQ::dequeue().
//! - Call Packet::fenceProdWait() to CPU wait for all fences in the payload.
//! - Updates the location of the packet to Packet::Location::Downstream
//!   by calling Packet::locationUpdate().
//! - Forwards the payload downstream by calling the srcRecvPayload()
//!   interface of the destination block through the destination
//!   SafeConnection.
//! - Proceeds with next iteration if packets available in queue.
//!
//! \return void, Triggers the following error events:
//! - Any error returned by Packet::fenceProdWait().
//!
//! \implements{}
void
PresentSync::waitForFenceThreadFunc(void) noexcept
{
    LwSciError err { LwSciError_Success };

    while (LwSciError_Success == err)
    {
        // Loop until there is a packet to wait for or teardown starts
        PacketPtr pkt { };
        {
            Lock blockLock { blkMutexLock() };
            pkt = fenceWaitQueue.dequeue();
            while (!teardown && (nullptr == pkt.get())) {
                packetCond.wait(blockLock);
                pkt = fenceWaitQueue.dequeue();
            }
            if (teardown) {
                return;
            }
        }

        // CPU wait for all fences
        err = pkt->fenceProdWait(waitContext);

        // On success, pass on the packet
        if (LwSciError_Success == err) {
            // Update location
            static_cast<void>(
                    pkt->locationUpdate(Packet::Location::Queued,
                                        Packet::Location::Downstream));

            // Send downstream
            getDst().srcRecvPayload(*pkt);
        }
    }

    // Report error if any and exit
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        setErrorEvent(err);
        LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    }
}

//! <b>Sequence of operations</b>
//!  - Under thread protection provided by Block::blkMutexLock(), it
//!    notifies the dispatch thread for block destruction.
//!  - Empties the fenceWaitQueue by dequeueing the Packet instances from it,
//!    discarding any unused payloads.
//!  - Scope of the lock expires.
//!  - Waits for the dispatchThread to finish with std::thread::join().
//!
//! \implements{}
LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
PresentSync::~PresentSync(void) noexcept
{
    // Remove all packets from the fenceWaitQueue so their pointers to
    //   each other don't keep them alive.
    {
        Lock blockLock { blkMutexLock() };
        teardown = true;
        packetCond.notify_one();
    }

    PacketPtr pkt {};
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A6_5_3), "LwSciStream-ADV-AUTOSARC++14-004")
    do {
        pkt = fenceWaitQueue.dequeue();
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    } while (nullptr != pkt);
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A6_5_3))

    dispatchThread.join();
}
LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))


//! <b>Sequence of operations</b>
//!   - Disconnects the source block by calling the Block::disconnectSrc()
//!     interface.
//!   - Disconnects the destination block by calling the Block::disconnectDst()
//!     interface.
//!
//! \implements{}
LwSciError PresentSync::disconnect(void) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectSrc();
    disconnectDst();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    // TODO: Does this function need a return value?
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Packet::handleGet() to retrieve the handle for the payload and
//!   call Block::pktFindByHandle() to find the corresponding local packet.
//! - Call Packet::locationUpdate() to move the packet from upstream to queued.
//! - Call Packet::fenceProdCopy() to save a copy of the producer fences.
//! - Call Packet::PayloadQ::enqueue() to add to the queue of pending packets
//!   and then notify the dispatch thread a packet is available.
//!
//! \implements{}
void PresentSync::srcRecvPayload(
    uint32_t const srcIndex,
    Packet const& prodPayload) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation{};
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

    // Copy producer fences into the packet
    LwSciError const err { pkt->fenceProdCopy(prodPayload) };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Add to queue of payloads to send and notify dispatch thread
    Lock blockLock { blkMutexLock() };
    fenceWaitQueue.enqueue(pkt);
    packetCond.notify_one();

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Disconnects the source block by calling the Block::disconnectSrc()
//!   interface.
//! - Triggers the LwSciStreamEventType_Disconnected event by calling the
//!   Block::disconnectEvent() interface.
//! - Disconnects the destination block by calling the Block::disconnectDst()
//!   interface.
//!
//! \implements{}
void PresentSync::srcDisconnect(
    uint32_t const srcIndex) noexcept
{
    // Validate block/input state
    //   Index must be valid
    ValidateBits validation{};

    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectSrc();
    disconnectEvent();
    disconnectDst();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
}

//! <b>Sequence of operations</b>
//! - Checks and sets flag indicating already done.
//! - Discards the consumer waiter requirements.
//! - Creates a waiter LwSciSyncAttrList by calling LwSciSyncAttrListCreate().
//! - Creates a new instance of LwSciWrap::SyncAttr which owns the waiter
//!   LwSciSyncAttrList.
//! - Fills the waiter LwSciSyncAttrList with CPU waiting attributes by
//!   calling LwSciSyncAttrListSetAttrs().
//! - Creates the CPU context used for doing CPU waits by calling
//!   LwSciSyncCpuWaitContextAlloc().
//! - Retrieves number of elements by calling Block::elementCountGet().
//! - Creates a new temporary Waiters instance and populates it with
//!   the LwSciSyncAttrList be calling Waiters::sizeInit(), Waiters::attrSet(),
//!   and Waiters::doneSet().
//! - Sends the waiter info upstream by calling dstRecvSyncWaiter()
//!   interface of the destination block through the SafeConnection.
//!
//! \implements{}
void PresentSync::dstRecvSyncWaiter(
    uint32_t const dstIndex,
    Waiters const& syncWaiter) noexcept
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

    // Discard the consumer waiter info
    static_cast<void>(syncWaiter);

    // Make sure not already done
    bool expected { false };
    if (!waiterDone.compare_exchange_strong(expected, true)) {
        setErrorEvent(LwSciError_AlreadyDone);
        return;
    }

    // Create and wrap CPU waiting attributes
    LwSciSyncAttrList waiterAttrList;
    LwSciError err { LwSciSyncAttrListCreate(syncModule, &waiterAttrList) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A7_1_1), "Bug 3258479")
    LwSciWrap::SyncAttr waiterAttrWrap { waiterAttrList, true };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Set the CPU waiting attributes
    // TODO: Add LwSciEvent access support, if it requires more than
    //   just CPU access.
    bool const cpuAccess { true };
    LwSciSyncAccessPerm const cpuPerm { LwSciSyncAccessPerm_WaitOnly };
    LwSciSyncAttrKeyValuePair const cpuKeyVals[2] {
        { LwSciSyncAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess) },
        { LwSciSyncAttrKey_RequiredPerm,  &cpuPerm,   sizeof(cpuPerm) }
    };
    err = LwSciSyncAttrListSetAttrs(waiterAttrList, cpuKeyVals, 2U);
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Create CPU context
    err = LwSciSyncCpuWaitContextAlloc(syncModule, &waitContext);
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Fill temporary waiter info object with attribute list for each element
    size_t const elemCount { elementCountGet() };
    Waiters tmpWaiter(FillMode::User, true);
    err = tmpWaiter.sizeInit(elemCount);
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }
    for (size_t i {0U}; elemCount > i; ++i) {
        err = tmpWaiter.attrSet(i, waiterAttrWrap);
        if (LwSciError_Success != err) {
            setErrorEvent(err);
            return;
        }
    }
    err = tmpWaiter.doneSet();
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Send the waiter info upstream
    getSrc().dstRecvSyncWaiter(tmpWaiter);

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Discard the incoming signal info.
//! - Creaate a new Signals instance.
//! - Call Signals::sizeInit() and Signals::doneSet() to set up the
//!   object with a list of the appropriate size but no objects.
//! - Call srcRecvSyncSignal() interface of destination connection to send
//!   the new object downstream.
//! - Call phaseProdSyncDoneSet() to advance setup phase.
void PresentSync::srcRecvSyncSignal(
    uint32_t const srcIndex,
    Signals const& syncSignal) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation{};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Discard incoming info
    static_cast<void>(syncSignal);

    // Create new Signals instance and initialize with empty array
    Signals tmpSignal{FillMode::User, nullptr};
    LwSciError err { tmpSignal.sizeInit(ONE, elementCountGet()) };
    if (LwSciError_Success == err) {
        err = tmpSignal.doneSet();
    }
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Send the signal info downstream
    getDst().srcRecvSyncSignal(tmpSignal);

    // Advance setup phase
    phaseProdSyncDoneSet();
}

//! <b>Sequence of operations</b>
//! - Disconnects the destination block by calling the Block::disconnectDst()
//!   interface.
//! - Triggers the LwSciStreamEventType_Disconnected event by calling the
//!   Block::disconnectEvent() interface.
//! - Disconnects the source block by calling the Block::disconnectSrc()
//!   interface.
//!
//! \implements{}
void PresentSync::dstDisconnect(
    uint32_t const dstIndex) noexcept
{
    // Validate block/input state
    //   Index must be valid
    ValidateBits validation{};

    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectDst();
    disconnectEvent();
    disconnectSrc();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
}

} // namespace LwSciStream
