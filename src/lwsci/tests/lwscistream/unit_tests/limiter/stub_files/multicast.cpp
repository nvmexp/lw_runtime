//! \file
//! \brief LwSciStream MultiCast declaration.
//!
//! \copyright
//! Copyright (c) 2019-2022 LWPU Corporation. All rights reserved.
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
#include <atomic>
#include <cmath>
#include <vector>
#include <functional>
#include <unordered_map>
#include <memory>
#include <limits>
#include <iterator>
#include <utility>
#include <vector>
#include <array>
#include <cassert>
#include "covanalysis.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "sciwrap.h"
#include "trackarray.h"
#include "lwscistream_common.h"
#include "lwscistream_types.h"
#include "lwscicommon_os.h"
#include "safeconnection.h"
#include "packet.h"
#include "syncwait.h"
#include "syncsignal.h"
#include "enumbitset.h"
#include "block.h"
#include "multicast.h"

namespace LwSciStream
{
// Constructs an instance of the MultiCast class
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A12_1_3), "Bug 2806643")
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A3_1_1), "Bug 2758535")
MultiCast::MultiCast(uint32_t const dstTotal) noexcept :
LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    Block(BlockType::MULTICAST, 1U /*srcTotal*/, dstTotal),
    branchCount(dstTotal),
    branchMap(dstTotal),
    aclwmulatedConsInfo(),
    consumerElementsBranch(dstTotal),
    consumerElements(FillMode::Merge, FillMode::None),
    consSyncWaiterBranch(dstTotal),
    consSyncWaiter(FillMode::Merge, true),
    consSyncSignalBranch(dstTotal),
    consSyncSignal(FillMode::Collate, nullptr),
    disconnectBranch(dstTotal)
{
    // Check for setup errors
    if (LwSciError_Success != branchMap.initErrorGet()) {
        setInitFail();
        return;
    }

    // Set up packet description
    Packet::Desc desc { };
    desc.initialLocation = Packet::Location::Upstream;
    desc.statusConsFillMode = FillMode::Collate;
    desc.fenceConsFillMode = FillMode::Collate;
    desc.branchCount = branchCount;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    pktDescSet(std::move(desc));
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A12_1_3))

//! <b>Sequence of operations</b>
//!   - Disconnects the source block by calling the Block::disconnectSrc()
//!     interface.
//!   - Disconnects all destination blocks by calling the
//!     Block::disconnectDst() interface for each destination block.
//!
//! \implements{19780875}
LwSciError MultiCast::disconnect(void) noexcept
{
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectSrc();
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M2_10_1), "TID-547")
    for (uint32_t i {0U}; branchCount > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        disconnectDst(i);
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M2_10_1))
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Block::blkMutexLock() to lock the block during the insertion calls.
//! -- Call BranchMap::set() to set the consumer count for the branch.
//! -- Call BranchMap::get() to retrieve range for branch's info.
//! -- Insert the incoming data into the aclwmulated list.
//! -- Call BranchMap::done() to check whether all branches have provided
//!    the info, and if so take responsibility for passing it on.
//! - After releasing the mutex, if this thread is responsible for the data:
//! -- Call Block::consInfoSet() to save the info.
//! -- Call Block::consInfoFlow() to pass the info upstream.
void MultiCast::dstRecvConsInfo(
    uint32_t const dstIndex,
    EndInfoVector const& info) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Take lock and handle insertion
    bool responsible;
    {
        // Lock mutex
        Lock const blockLock { blkMutexLock() };

        // Update map with output's consumer count
        LwSciError err { branchMap.set(dstIndex, info.size()) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }

        // Retrieve range for output's info
        BranchMap::Range const range { branchMap.get(dstIndex) };
        assert(info.size() == range.count);

        // Insert the incoming info into the aclwmulated list
        try {
            static_cast<void>(
                aclwmulatedConsInfo.insert(
                    aclwmulatedConsInfo.begin() + range.start,
                    info.cbegin(),
                    info.cend()));
        } catch (...) {
            setErrorEvent(LwSciError_InsufficientMemory, true);
            return;
        }

        // Check if responsible for saving and sending upstream
        responsible = branchMap.done();
    }

    // When all outputs have provided info, first caller processes results
    if (responsible) {

        // Save the info in the base block
        LwSciError const err { consInfoSet(aclwmulatedConsInfo) };
        if (LwSciError_Success != err) {
            setErrorEvent(err);
            return;
        }

        // Send upstream
        consInfoFlow();

        // Aclwmlated copy is no longer needed
        aclwmulatedConsInfo.clear();
        aclwmulatedConsInfo.shrink_to_fit();
    }
}

//! <b>Sequence of operations</b>
//! - Call Block::blkMutexLock() to lock the block during the merge calls.
//! -- Call BranchTrack::set() to take ensure branch's data is only set once.
//! -- Call Elements::mapMerge() to merge the incoming elements into the
//!    aclwmulated list.
//! -- Call BranchTrack::done() to check whether all branches have provided
//!    the info, and if so take responsibility for passing it on.
//! - After releasing the mutex, if this thread is responsible for the data:
//! -- Call Elements::mapDone() to finalize the consolidated list.
//! -- Call Elements::dataSend() to ilwoke dstRecvSupportedElements() on the
//!    upstream connection to pass on the list, and then free the element
//!    memory.
void MultiCast::dstRecvSupportedElements(
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

    // Take lock and handle merge
    bool responsible;
    {
        // Lock mutex
        Lock const blockLock { blkMutexLock() };

        // Make sure branch's info is only set once
        LwSciError err { consumerElementsBranch.set(dstIndex) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }

        // Merge in the branch's info
        err = consumerElements.mapMerge(inElements);
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }

        // Check if responsible for sending upstream
        responsible = consumerElementsBranch.done();
    }

    // When all outputs have provided info, first caller sends upstream
    if (responsible) {

        // Indicate consolidated elements are complete
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        LwSciError err { consumerElements.mapDone(false) };
        if (LwSciError_Success != err) {
            setErrorEvent(err);
            return;
        }

        // Define send operation
        Elements::Action const sendAction {
            [this](Elements const& elements) noexcept -> void {
                LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
                getSrc().dstRecvSupportedElements(elements);
            }
        };

        // Send upstream and then clear the data
        err = consumerElements.dataSend(sendAction, true);
        if (LwSciError_Success != err) {
            setErrorEvent(err);
        }
    }
}

//! <b>Sequence of operations</b>
//! - Call Elements::sizePeek() to retrieve the number of elements.
//! - Call Waiters::sizeInit() to initialize the sync attribute vectors.
//! - Call Signals::sizeInit() to initialize the sync object vectors.
//! - Call Block::srcRecvAllocatedElements() to pass the elements downstream.
void MultiCast::srcRecvAllocatedElements(
    uint32_t const srcIndex,
    Elements const& inElements) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Retrieve the number of elements
    size_t const elemCount { inElements.sizePeek() };

    // Initialize waiter sync attribute tracker
    LwSciError err { consSyncWaiter.sizeInit(elemCount) };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Initialize signal sync object tracker
    err = consSyncSignal.sizeInit(consumerCountGet(), elemCount);
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Ilwoke base function to send elements downstream
    Block::srcRecvAllocatedElements(srcIndex, inElements);
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindHandle() to find the local Packet instance
//!   corresponding to the incoming packet.
//! - Call BranchMap::get() to retrieve the output's consumer range within
//!   the block's list.
//! - Call Block::blkMutexLock() to lock the block during the collation calls.
//! -- Call Packet::statusConsCollate() to collate the incoming dsta into
//!    the local packet.
//! -- Call Packet::statusConsCollateDone() to check whether all branches have
//!    returned the data, and if so take responsibility for passing it on.
//! - After releasing the mutex, if this thread is responsible for the data:
//! -- Call dstRecvPacketStatus() on source connection to send the status
//!    upstream.
//! -- Call Packet::statusClear() to free status resources.
void MultiCast::dstRecvPacketStatus(
    uint32_t const dstIndex,
    Packet const& origPacket) noexcept
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

    // Find the packet for this handle
    PacketPtr const pkt { pktFindByHandle(origPacket.handleGet()) };
    if (nullptr == pkt) {
        setErrorEvent(LwSciError_StreamBadPacket);
        return;
    }

    // Retrieve the output's offset and size within the overall list
    BranchMap::Range const range { branchMap.get(dstIndex) };

    // Do the copy
    bool responsible;
    {
        // Lock mutex
        Lock const blockLock { blkMutexLock() };

        // Collate if not already done for this branch
        LwSciError const err {
            pkt->statusConsCollate(origPacket, dstIndex,
                                   range.start, range.count)
        };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }

        // Check if responsible for sending upstream
        responsible = pkt->statusConsCollateDone();
    }

    // When all outputs have provided info, first caller sends upstream
    if (responsible) {

        // Send the packet upstream
        getSrc().dstRecvPacketStatus(*pkt);

        // Release status vector
        pkt->statusClear();
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Block::blkMutexLock() to lock the block during the merge calls.
//! -- Call BranchTrack::set() to take ensure branch's data is only set once.
//! -- Call Waiters::merge() to merge the incoming waiter information into
//!    the aclwmulated list.
//! -- Call BranchTrack::done() to check whether all branches have provided
//!    the info, and if so take responsibility for passing it on.
//! - After releasing the mutex, if this thread is responsible for the data:
//! -- Call Waiters::doneSet() to finalize the consolidated list.
//! -- Call dstRecvSyncWaiter() on the upstream connection to send the
//!    list upstream.
//! -- Call Waiters::clear() to reclaim resources associated with the list.
void MultiCast::dstRecvSyncWaiter(
    uint32_t const dstIndex,
    Waiters const& syncWaiter) noexcept
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

    // Take lock and handle merge
    bool responsible;
    {
        // Lock mutex
        Lock const blockLock { blkMutexLock() };

        // Make sure branch's info is only set once
        LwSciError err { consSyncWaiterBranch.set(dstIndex) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }

        // Merge in the branch's info
        err = consSyncWaiter.merge(syncWaiter);
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }

        // Check if responsible for sending upstream
        responsible = consSyncWaiterBranch.done();
    }

    // When all outputs have provided info, first caller sends upstream
    if (responsible) {

        // Indicate consolidated waiter information is complete
        LwSciError const err { consSyncWaiter.doneSet() };
        if (LwSciError_Success != err) {
            setErrorEvent(err);
            return;
        }

        // Send the consolidated information upstream
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getSrc().dstRecvSyncWaiter(consSyncWaiter);

        // Reclaim no longer needed resources
        consSyncWaiter.clear();
    }
}

//! <b>Sequence of operations</b>
//! - Call BranchMap::get() to retrieve the output's consumer range within
//!   the block's list.
//! - Call Block::blkMutexLock() to lock the block during the collate calls.
//! -- Call BranchTrack::set() to take ensure branch's data is only set once.
//! -- Call Signals::collate() to collate the incoming signal information into
//!    the aclwmulated list.
//! -- Call BranchTrack::done() to check whether all branches have provided
//!    the info, and if so take responsibility for passing it on.
//! - After releasing the mutex, if this thread is responsible for the data:
//! -- Call Signals::doneSet() to finalize the consolidated list.
//! -- Call dstRecvSyncSignal() on the upstream connection to send the
//!    list upstream.
//! -- Call Signals::clear() to reclaim resources associated with the list.
//! -- Call phaseConsSyncDoneSet() to advance setup phase.
void MultiCast::dstRecvSyncSignal(
    uint32_t const dstIndex,
    Signals const& syncSignal) noexcept
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

    // Retrieve the output's offset and size within the overall list
    BranchMap::Range const range { branchMap.get(dstIndex) };

    // Take lock and handle collation
    bool responsible;
    {
        // Lock mutex
        Lock const blockLock { blkMutexLock() };

        // Make sure branch's info is only set once
        LwSciError err { consSyncSignalBranch.set(dstIndex) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }

        // Collate in the branch's info
        err = consSyncSignal.collate(syncSignal, range.start, range.count);
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }

        // Check if responsible for sending upstream
        responsible = consSyncSignalBranch.done();
    }

    // When all outputs have provided info, first caller sends upstream
    if (responsible) {

        // Indicate consolidated signal information is complete
        LwSciError const err { consSyncSignal.doneSet() };
        if (LwSciError_Success != err) {
            setErrorEvent(err);
            return;
        }

        // Send the consolidated information upstream
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getSrc().dstRecvSyncSignal(consSyncSignal);

        // Reclaim no longer needed resources
        consSyncSignal.clear();

        // Advance setup phase
        phaseConsSyncDoneSet();
    }
}

//! <b>Sequence of operations</b>
//! - Call BranchMap::get() to retrieve the output's consumer range within
//!   the block's list.
//! - Call Packet::handleGet() to retrieve the handle from the payload and
//!   call Block::pktFindByHandle() to retrieve the corresponding local packet.
//! - Call Packet::locationCheck() to make sure the packet is downstream.
//! - Call Block::blkMutexLock() to lock the block during the collation calls.
//! -- Call Packet::fenceConsCollate() to collate the incoming fences into
//!    the local packet.
//! -- Call BranchTrack::getAll() on disconnectBranch and Call
//!    Packet::fenceConsCollateDone() to check whether all branches have
//!    returned the packet, and if so take responsibility for passing it on.
//! - After releasing the mutex, if this thread is responsible for the data:
//! -- Call Packet::locationUpdate() to change the location to upstream.
//! -- Call dstRecvPayload() interface of source block to send the complete
//!    payload upstream.
//!
//! \implements{}
void MultiCast::dstRecvPayload(
    uint32_t const dstIndex,
    Packet const& consPayload) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // If disconnect has oclwrred, return but don't report an error
    // TODO: Need to distinguish various disconnect cases and handle
    //       in the validate function above.
    if (disconnectBranch.get(dstIndex) || !connComplete()) {
        return;
    }

    // Retrieve the output's offset and size within the overall list
    BranchMap::Range const range { branchMap.get(dstIndex) };

    // Retrieve local packet for this payload
    PacketPtr const pkt { pktFindByHandle(consPayload.handleGet()) };
    if (nullptr == pkt) {
        setErrorEvent(LwSciError_StreamBadPacket);
        return;
    }

    // Packet must be downstream for it to be returned
    if (!pkt->locationCheck(Packet::Location::Downstream)) {
        setErrorEvent(LwSciError_StreamPacketInaccessible);
        return;
    }

    // Take lock and handle collation
    bool responsible;
    {
        // Lock mutex
        Lock const blockLock { blkMutexLock() };

        // Collate if not already done for this branch
        LwSciError const err {
            pkt->fenceConsCollate(consPayload, dstIndex,
                                  range.start, range.count)
        };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }

        // Check if responsible for sending upstream
        responsible = pkt->fenceConsCollateDone(disconnectBranch.getAll());
    }

    // When all outputs have provided info, first caller sends upstream
    if (responsible) {

        // Update location
        static_cast<void>(pkt->locationUpdate(Packet::Location::Downstream,
                                              Packet::Location::Upstream));

        // Send the consolidated packet upstream
        getSrc().dstRecvPayload(*pkt);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Block::blkMutexLock() to lock the block during the tracking.
//! -- Call BranchTrack::set() to take ensure branch only disconnects once.
//! -- Call BranchTrack::done() to check whether all branches have
//!    disconnected, and if so take responsibility for passing it on.
//! - After releasing the mutex:
//! -- Call Block::getPacketMap() to get the PacketMap.
//! -- Iterate through all the packets in the map. If the location of the packet
//!    is Packet::Location::Downstream then acquires the lock and
//!    calls Packet::fenceConsCollateDone().
//! -- After releasing the mutex, if this thread is responsible for the data:
//! -- Call Packet::locationUpdate() to change the location to upstream.
//! -- Call dstRecvPayload() interface of source block to send the complete
//!    payload upstream.
//! -- Call Block::disconnectDst() to clear the downstream connection.
//! -- If this thread is responsible for the data:
//! --- Call Block::disconnectSrc() to clear the upstream connection.
//! --- Call disconnectEvent() to post the disconnect event.
//!
//! \implements{19780902}
void MultiCast::dstDisconnect(
    uint32_t const dstIndex) noexcept
{
    // Validate block/input state
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Take lock and handle tracking
    bool responsible;
    {
        // Lock mutex
        Lock const blockLock { blkMutexLock() };

        // Make sure branch only disconnects once
        LwSciError err { disconnectBranch.set(dstIndex) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }

        // Check if responsible for sending upstream
        responsible = disconnectBranch.done();
    }

    // Mark any packets still held by the branch as returned
    PacketMap pktMap { getPacketMap() };

    for (PacketIter x { pktMap.begin() }; pktMap.end() != x; ++x) {
        PacketPtr const pkt { x->second };
        // Packet must be downstream for it to be returned
        if (pkt->locationCheck(Packet::Location::Downstream)) {
            bool collateDone{ false };
            {
                // take the lock
                Lock blockLock { blkMutexLock() };

                // Check if responsible for sending upstream
                collateDone = pkt->fenceConsCollateDone(disconnectBranch.getAll());
            }

            // When all outputs have provided info, first caller sends upstream
            if (collateDone) {

                // Update location
                static_cast<void>(pkt->locationUpdate(Packet::Location::Downstream,
                                                      Packet::Location::Upstream));

                // Send the consolidated packet upstream
                getSrc().dstRecvPayload(*pkt);
            }
        }
    }

    // Disconnect destination connection
    disconnectDst(dstIndex, responsible);

    // When all outputs have disconnected, first caller sends upstream and
    //   signals event.
    if (responsible) {
        disconnectSrc();
        disconnectEvent();
    }
}

//! <b>Sequence of operations</b>
//!   - Disconnects all the destination blocks by calling the
//!     Block::disconnectDst() interface for each destination block.
//!   - Disconnects the source block by calling the Block::disconnectSrc()
//!     interface.
//!   - Sets LwSciStreamEventType_Disconnected by calling the
//!     Block::disconnectEvent() interface.
//!
//! \implements{19780935}
void MultiCast::srcDisconnect(
    uint32_t const srcIndex) noexcept
{
    // Validate block/input state
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M2_10_1), "TID-547")
    for (uint32_t i {0U}; branchCount > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        disconnectDst(i);
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M2_10_1))
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectSrc();
    disconnectEvent();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
}

} // namespace LwSciStream
