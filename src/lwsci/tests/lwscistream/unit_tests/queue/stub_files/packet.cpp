//! \file
//! \brief LwSciStream packet class definition.
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
#include <array>
#include <cassert>
#include <iterator>
#include <unordered_map>
#include <memory>
#include <vector>
#include <atomic>
#include <vector>
#include <functional>
#include "covanalysis.h"
#include "lwscibuf_internal.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "sciwrap.h"
#include "lwscistream_types.h"
#include "lwscicommon_os.h"
#include "lwscistream_common.h"
#include "branch.h"
#include "trackarray.h"
#include "elements.h"
#include "syncsignal.h"
#include "packet.h"
#include "ipcbuffer.h"
#include "ipcbuffer_sciwrap.h"

namespace LwSciStream {

// Constructs an instance of the Packet class
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A12_1_5), "Bug 2989775")
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A3_1_1), "Bug 2758535")
Packet::Packet(
    Packet::Desc const& paramPktDesc,
    LwSciStreamPacket const paramHandle,
    LwSciStreamCookie const paramCookie) noexcept :
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A12_1_5))
        initError(LwSciError_Success),
        pktDesc(paramPktDesc),
        handle(paramHandle),
        cookie(paramCookie),
        lwrrLocation(paramPktDesc.initialLocation),
        buffers(paramPktDesc.defineFillMode, false),
        defineCompleted(false),
        defineEvent(false),
        defineHandleEvent(false),
        statusCons(paramPktDesc.statusConsFillMode, false),
        statusProd(LwSciError_StreamInternalError),
        statusConsCompleted(false),
        statusProdCompleted(false),
        statusConsBranch(paramPktDesc.branchCount),
        statusEvent(0U),
        rejected(false),
        fenceProd(paramPktDesc.fenceProdFillMode, true),
        fenceCons(paramPktDesc.fenceConsFillMode, true),
        fenceConsBranch(paramPktDesc.branchCount),
        deleteEvent(false),
        deleteCookieEvent(false),
        zombie(false),
        payloadPrev(),
        payloadNext(),
        payloadQueued(false)
{
    // Initialize buffer attribute list if used by this block
    if (pktDesc.needBuffers) {
        initError = buffers.sizeInit(pktDesc.elementCount);
        if (LwSciError_Success != initError) {
            return;
        }

        // Initialize C2C buffer handles list
        if (pktDesc.useC2CBuffer) {
            try {
                LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
                LWCOV_ALLOWLIST_LINE(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
                c2cBufHandles.resize(pktDesc.elementCount, nullptr);
                LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
            } catch (...) {
                initError = LwSciError_InsufficientMemory;
                return;
            }
        }
    }

    // Initialize consumer status vector if used by this block
    //   The StreamInternalError value is used to indicate the user has
    //   not yet set the status.
    if (FillMode::None != pktDesc.statusConsFillMode) {
        initError = statusCons.sizeInit(pktDesc.numConsumer,
                                        LwSciError_StreamInternalError);
        if (LwSciError_Success != initError) {
            return;
        }
    }

    // Initialize producer and consumer fence arrays
    if (FillMode::None != pktDesc.fenceProdFillMode) {
        initError = fenceProd.sizeInit(pktDesc.elementCount);
        if (LwSciError_Success != initError) {
            return;
        }
    }
    if (FillMode::None != pktDesc.fenceConsFillMode) {
        initError = fenceCons.sizeInit(pktDesc.numConsumer *
                                       pktDesc.elementCount);
        if (LwSciError_Success != initError) {
            return;
        }
    }
}

//! <b>Sequence of operations</b>
//! - Call LwSciBufFreeSourceObjIndirectChannelC2c() to free the source handle
//!   of the C2C buffer.
Packet::~Packet(void) noexcept
{
    // Free c2c buf source handles
    for (size_t i{ 0U }; c2cBufHandles.size() > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != c2cBufHandles[i]) {
            static_cast<void>(
                LwSciBufFreeSourceObjIndirectChannelC2c(c2cBufHandles[i]));
        }
    }
}

bool
Packet::locationCheck(
    Packet::Location const expectLocation) const noexcept
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A13_5_3), "LwSciStream-ADV-AUTOSARC++14-001")
    return (expectLocation == lwrrLocation);
}

// Atomically validates current packet Location and replaces it
// with the new one.
bool
Packet::locationUpdate(
    Packet::Location const oldLocation,
    Packet::Location const newLocation) noexcept
{
    // Try to update the location
    Packet::Location tmpLocation { oldLocation };
    return lwrrLocation.compare_exchange_strong(tmpLocation, newLocation);
}

//
// Packet definition functions
//

//! <b>Sequence of operations</b>
//! - Call TrackArray::get() to retrieve a copy of the indexed entry in the
//!   buffer list.
LwSciError
Packet::bufferGet(
    size_t const elemIndex,
    LwSciWrap::BufObj& elemBufObj) const noexcept
{
    return buffers.get(elemIndex, elemBufObj);
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::set() to set the indexed entry in the buffer list.
//! - Call TrackArray::peek() to see the resulting entry.
//! - Call LwSciWrap::BufObj::getErr() to check for any duplication errors.
LwSciError
Packet::bufferSet(
    size_t const elemIndex,
    LwSciWrap::BufObj const& elemBufObj) noexcept
{
    LwSciError err { buffers.set(elemIndex, elemBufObj) };
    if (LwSciError_Success == err) {
        auto const entry { buffers.peek(elemIndex) };
        err = entry.first;
        if (LwSciError_Success == err) {
            err = entry.second->getErr();
        }
    }
    return err;
}

//! <b>Sequence of operations</b>
//! - Ensure definition not already complete.
//! - Create a functional which calls LwSciWrap::BufObj::viewVal() and
//!   returns true if the value is not nullptr.
//! - Call TrackArray::test() with the functional to determine if all
//!   buffers in the array have been set.
//! - Call TrackArray::finalize() to lock the array.
//! - Mark definition complete.
LwSciError
Packet::defineDone(void) noexcept
{
    // Make sure not already done
    if (defineCompleted.load()) {
        return LwSciError_AlreadyDone;
    }

    // Functional to verify a member of the buffer list is set
    auto const criteria {
        [](LwSciWrap::BufObj const& val) noexcept -> bool {
            return (nullptr != val.viewVal());
        }
    };

    // Check if all entries in the buffer list are set
    if (!buffers.test(criteria)) {
        return LwSciError_InsufficientData;
    }

    // Finalize the array
    buffers.finalize();

    // Mark completed
    defineCompleted.store(true);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Ensure definition not already complete.
//! - If this packet uses the buffers, call TrackArray::copy() to copy in
//!   the contents of @a origPacket's buffer array.
//! - If @a setEvent is true, set the defineEvent flag.
//! - Mark definition complete.
LwSciError
Packet::defineCopy(
    Packet const& origPacket,
    bool const    setEvent) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Make sure not already done
    if (defineCompleted.load()) {
        return LwSciError_AlreadyDone;
    }
    assert(origPacket.defineCompleted.load());

    // If this block needs buffer objects, copy the array
    if (pktDesc.needBuffers) {
        LwSciError const err { buffers.copy(origPacket.buffers) };
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // If required, enable event
    if (setEvent) {
        defineEvent.store(true);
    }

    // Mark completed
    defineCompleted.store(true);

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Ensure definition is complete.
//! - Call TrackArray::pack() to pack the buffer array.
LwSciError
Packet::definePack(
    IpcBuffer& buf) const noexcept
{
    // Note: Handle was already sent ahead because it was needed to create
    //       the Packet on the far side so the data could be unpacked.

    // Make sure data is completed
    if (!defineCompleted.load()) {
        return LwSciError_InsufficientData;
    }

    // Pack the array of buffer objects
    return buffers.pack(buf);
}

//! <b>Sequence of operations</b>
//! - Ensure definition not already complete.
//! - Call TrackArray::unpack() to unpack the buffer array.
//! - Mark definition complete.
LwSciError
Packet::defineUnpack(
    IpcBuffer& buf,
    Elements const& aux) noexcept
{
    // Note: Handle was already sent ahead because it was needed to create
    //       this Packet so the data could be unpacked.

    // Make sure not already done
    if (defineCompleted.load()) {
        return LwSciError_AlreadyDone;
    }

    // Pack the array of buffer objects
    LwSciError const err { buffers.unpack(buf, aux.elemArrayGet()) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Mark completed
    defineCompleted.store(true);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::clear() to free the buffer vector resources.
void
Packet::defineClear(void) noexcept
{
    buffers.clear();
}

//
// C2C Buffer functions
//

//! <b>Sequence of operations</b>
//! - For all buffer elements, call TrackArray::peek() to get
//!   the buffer element and call Wrapper::viewVal() to retrieve the
//!   LwSciBufObj.
//! - Call LwSciBufRegisterSourceObjIndirectChannelC2c() to register
//!   the LwSciBufObj with the C2C service and saves the source handle.
LwSciError
Packet::registerC2CBufSourceHandles(
    LwSciC2cHandle const channelHandle) noexcept
{
    // Status must be complete
    if (!defineCompleted.load()) {
        return LwSciError_NotYetAvailable;
    }

    assert(buffers.sizeGet() == c2cBufHandles.size());

    for (size_t i{ 0U }; c2cBufHandles.size() > i; ++i) {
        auto const bufObj{ buffers.peek(i) };
        if (LwSciError_Success != bufObj.first) {
            return bufObj.first;
        }

        // Register with C2C service
        LwSciC2cBufSourceHandle sourceHandle;
        LwSciError const err{
            LwSciBufRegisterSourceObjIndirectChannelC2c(
                channelHandle,
                bufObj.second->viewVal(),
                &sourceHandle)
        };
        if (LwSciError_Success != err) {
            return err;
        }

        // Save the C2C buffer source handle
        c2cBufHandles[i] = sourceHandle;
    }

    return LwSciError_Success;
}

//
// Packet status functions
//

//! <b>Sequence of operations</b>
//! - Validate state.
//! - Retrieve status value.
LwSciError
Packet::statusProdGet(
    LwSciError& outStatus) const noexcept
{
    // Status must be complete
    if (!statusProdCompleted.load()) {
        return LwSciError_NotYetAvailable;
    }

    // Return value
    outStatus = statusProd;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Validate state.
//! - Call TrackArray::get to retrieve status value.
LwSciError
Packet::statusConsGet(
    size_t const consIndex,
    LwSciError& outStatus) const noexcept
{
    // Status must be complete
    if (!statusConsCompleted.load()) {
        return LwSciError_NotYetAvailable;
    }

    // Return value
    return statusCons.get(consIndex, outStatus);
}

//! <b>Sequence of operations</b>
//! - Validate parameters and current state.
//! - Set status, cookie, and completion flag.
LwSciError
Packet::statusProdSet(
    LwSciError const paramStatus,
    LwSciStreamCookie const paramCookie) noexcept
{
    // Forbid callers to use StreamInternalError, because that's what
    //   we use as the initializer
    if (LwSciError_StreamInternalError == paramStatus) {
        return LwSciError_BadParameter;
    }

    // If status is Success, a cookie must be provided
    if ((LwSciError_Success == paramStatus) &&
        (LwSciStreamCookie_Ilwalid == paramCookie)) {
        return LwSciError_StreamBadCookie;
    }

    // Check if already set
    if (statusProdCompleted.load()) {
        return LwSciError_AlreadyDone;
    }

    // Fill mode must be User
    if (FillMode::User != pktDesc.statusProdFillMode) {
        return LwSciError_IlwalidOperation;
    }

    // Save status and cookie, and mark completed
    // Note: We don't set the event for the status source
    statusProd = paramStatus;
    if (LwSciError_Success == paramStatus) {
        cookie = paramCookie;
    } else {
        rejected = true;
    }
    statusProdCompleted.store(true);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Validate parameters and current state.
//! - Call TrackArray::sizeGet() to get the consumer status array size.
//! - For all elements in consumer status array, call TrackArray::set() to
//!   set the status.
//! - Set cookie and completion flag.
//! - Call TrackArray::finalize() to lock the consumer status array.
LwSciError
Packet::statusConsSet(
    LwSciError const paramStatus,
    LwSciStreamCookie const paramCookie) noexcept
{
    // Forbid callers to use StreamInternalError, because that's what
    //   we use as the initializer
    if (LwSciError_StreamInternalError == paramStatus) {
        return LwSciError_BadParameter;
    }

    // If status is Success, a cookie must be provided
    if ((LwSciError_Success == paramStatus) &&
        (LwSciStreamCookie_Ilwalid == paramCookie)) {
        return LwSciError_StreamBadCookie;
    }

    // Check if already set
    if (statusConsCompleted.load()) {
        return LwSciError_AlreadyDone;
    }

    // Save status and cookie, and mark completed
    // Note: We don't set the event for the status source
    size_t const size { statusCons.sizeGet() };
    for (size_t i {0U}; size > i; ++i) {
        LwSciError const err { statusCons.set(i, paramStatus) };
        if (LwSciError_Success != err) {
            return err;
        }
    }
    if (LwSciError_Success == paramStatus) {
        cookie = paramCookie;
    } else {
        rejected = true;
    }
    statusCons.finalize();
    statusConsCompleted.store(true);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Validate parameters and current state.
//! - Set status and completion flag.
LwSciError
Packet::statusProdCopy(
    Packet const& origPacket) noexcept
{
    // Check if already set
    assert(origPacket.statusProdCompleted.load());
    if (statusProdCompleted.load()) {
        return LwSciError_AlreadyDone;
    }

    // Fill mode must be Copy
    if (FillMode::Copy != pktDesc.statusProdFillMode) {
        return LwSciError_IlwalidOperation;
    }

    // Copy status, mark completed, and increment event flag
    statusProd = origPacket.statusProd;
    rejected = origPacket.rejected;
    statusProdCompleted.store(true);
    statusEvent++;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Validate parameters and current state.
//! - Call TrackArray::copy to copy the status.
//! - Set completion flag.
LwSciError
Packet::statusConsCopy(
    Packet const& origPacket) noexcept
{
    // Check if already set
    assert(origPacket.statusConsCompleted.load());
    if (statusConsCompleted.load()) {
        return LwSciError_AlreadyDone;
    }

    // Copy status, mark completed, and increment event flag
    LwSciError const err { statusCons.copy(origPacket.statusCons) };
    if (LwSciError_Success != err) {
        return err;
    }
    rejected = origPacket.rejected;
    statusConsCompleted.store(true);
    statusEvent++;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call BranchTrack::set() to check whether the branch is already set,
//!   and if not take responsibility for doing so.
//! - Call TrackArray::collate() to copy in the status values.
LwSciError
Packet::statusConsCollate(
    Packet const& origPacket,
    size_t const branchIndex,
    size_t const rangeStart,
    size_t const rangeCount) noexcept
{
    // Indicate that values for the branch are being set
    LwSciError err { statusConsBranch.set(branchIndex) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Copy range of data
    err = statusCons.collate(origPacket.statusCons, rangeStart, rangeCount);
    if (LwSciError_Success != err) {
        return err;
    }

    // Copy any rejection
    if (origPacket.rejected) {
        rejected = true;
    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call BranchTrack::done() to take responsibility for passing on the
//!   data if it is all set.
//! - If so, call TrackArray::finalize() to lock the array.
bool
Packet::statusConsCollateDone(void) noexcept
{
    if (statusConsBranch.done()) {
        statusCons.finalize();
        statusConsCompleted.store(true);
        return true;
    }

    return false;
}

//! <b>Sequence of operations</b>
//! - Ensure status list is complete.
//! - Call IpcBuffer::packVal() to pack the rejection flag.
//! - Call TrackArray::pack() to pack the status array.
LwSciError
Packet::statusConsPack(
    IpcBuffer& buf) const noexcept
{
    // Make sure status is completed
    if (!statusConsCompleted.load()) {
        return LwSciError_InsufficientData;
    }

    // Pack the rejection flag
    LwSciError const err { buf.packVal(rejected) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Pack the status array
    return statusCons.pack(buf);
}

//! <b>Sequence of operations</b>
//! - Ensure status list is not already complete.
//! - Call IpcBuffer::packVal() to pack the rejection flag.
//! - Call TrackArray::unpack() to unpack the status array.
//! - Mark status complete.
LwSciError
Packet::statusConsUnpack(
    IpcBuffer& buf) noexcept
{
    // Make sure not already done
    if (statusConsCompleted.load()) {
        return LwSciError_AlreadyDone;
    }

    // Unpack the rejection flag
    LwSciError err { buf.unpackVal(rejected) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Unpack the array of buffer objects
    err = statusCons.unpack(buf);
    if (LwSciError_Success != err) {
        return err;
    }

    // Mark completed
    // Note: We don't set the event flag for IPC transfer
    statusConsCompleted.store(true);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Compute the number of completed status lists needed to trigger event.
//! - Check whether the event flag meets the expected number, clearing if so.
bool
Packet::statusPending(void) noexcept
{
    // The count to expect for the event flag depends on whether one or
    //   both lists are used
    uint32_t expectedCount {
        ((FillMode::None != pktDesc.statusProdFillMode) ? 1U : 0U) +
        ((FillMode::None != pktDesc.statusConsFillMode) ? 1U : 0U)
    };
    if (0U == expectedCount) {
        return false;
    }

    // Check if the number of lists ready matches the count, and if so clear
    return statusEvent.compare_exchange_strong(expectedCount, 0U);
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::clear() to free the consumer status vector resources.
void
Packet::statusClear(void) noexcept
{
    statusCons.clear();
}

//
// Packet fence functions
//

//! <b>Sequence of operations</b>
//! - Call TrackArray::get() to retrieve the indexed fence.
LwSciError
Packet::fenceProdGet(
    size_t const elemIndex,
    LwSciWrap::SyncFence& fence) const noexcept
{
    return fenceProd.get(elemIndex, fence);
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::get() to retrieve the indexed fence.
LwSciError
Packet::fenceConsGet(
    size_t const consIndex,
    size_t const elemIndex,
    LwSciWrap::SyncFence& fence) const noexcept
{
    // TODO: Indvidual limit checks and CERT check
    return fenceCons.get(consIndex * pktDesc.elementCount + elemIndex, fence);
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::set() to set the indexed fence.
LwSciError
Packet::fenceProdSet(
    size_t const elemIndex,
    LwSciWrap::SyncFence const& fence) noexcept
{
    return fenceProd.set(elemIndex, fence);
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::set() to set the indexed fence.
LwSciError
Packet::fenceConsSet(
    size_t const elemIndex,
    LwSciWrap::SyncFence const& fence) noexcept
{
    return fenceCons.set(elemIndex, fence);
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::finalize() to mark the fence list as complete.
void
Packet::fenceProdDone(void) noexcept
{
    fenceProd.finalize();
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::finalize() to mark the fence list as complete.
void
Packet::fenceConsDone(void) noexcept
{
    fenceCons.finalize();
}

//! <b>Sequence of operations</b>
//! - For all fences, call TrackArray::setAll() to set the fence.
LwSciError
Packet::fenceProdFill(
    LwSciWrap::SyncFence const& fence) noexcept
{
    return fenceProd.setAll(fence);
}

//! <b>Sequence of operations</b>
//! - For all fences, call TrackArray::setAll() to set the fence.
LwSciError
Packet::fenceConsFill(
    LwSciWrap::SyncFence const& fence) noexcept
{
    return fenceCons.setAll(fence);
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::sizeGet() to check whether the incoming packet
//!   provides fences.
//! -- If not, call TrackArray::finalize() to mark the current set of
//!    empty fences as complete.
//! -- Otherwise, call TrackArray::copy() to copy the incoming fences.
LwSciError
Packet::fenceProdCopy(
    Packet const& origPacket) noexcept
{
    // It's possible that the packet came from an intermediate block that
    //   makes the fences moot, so the packet doesn't provide any. In that
    //   just return success, but finalize the array so the data is
    //   considered complete.
    if (0U == origPacket.fenceProd.sizeGet()) {
        fenceProd.finalize();
        return LwSciError_Success;
    }

    // Do a normal array copy
    return fenceProd.copy(origPacket.fenceProd);
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::sizeGet() to check whether the incoming packet
//!   provides fences.
//! -- If not, call TrackArray::finalize() to mark the current set of
//!    empty fences as complete.
//! -- Otherwise, call TrackArray::copy() to copy the incoming fences.
LwSciError
Packet::fenceConsCopy(
    Packet const& origPacket) noexcept
{
    // It's possible that the packet came from an intermediate block that
    //   makes the fences moot, so the packet doesn't provide any. In that
    //   just return success, but finalize the array so the data is
    //   considered complete.
    if (0U == origPacket.fenceCons.sizeGet()) {
        fenceCons.finalize();
        return LwSciError_Success;
    }

    // Do a normal array copy
    return fenceCons.copy(origPacket.fenceCons);
}

//! <b>Sequence of operations</b>
//! - Call BranchTrack::set() to check whether the branch is already set,
//!   and if not take responsibility for doing so.
//! - Call TrackArray::collate() to copy in the fences.
LwSciError
Packet::fenceConsCollate(
    Packet const& origPacket,
    size_t const branchIndex,
    size_t const endRangeStart,
    size_t const endRangeCount) noexcept
{
    // Indicate that fences for the branch are being set
    LwSciError const err { fenceConsBranch.set(branchIndex) };
    if (LwSciError_Success != err) {
        return err;
    }

    // If incoming packet doesn't contain fences, they're assumed to be
    //   empty and there's nothing to do
    if (0U == origPacket.fenceCons.sizeGet()) {
        return LwSciError_Success;
    }

    // Copy fences to the appropriate subrange
    // TODO: CERT checks
    size_t const rangeStart { endRangeStart * pktDesc.elementCount };
    size_t const rangeCount { endRangeCount * pktDesc.elementCount };
    return fenceCons.collate(origPacket.fenceCons, rangeStart, rangeCount);
}

//! <b>Sequence of operations</b>
//! - Call BranchTrack::done() to take responsibility for passing on the
//!   fences if they are all set.
//! - If so, call TrackArray::finalize() to lock the array.
bool
Packet::fenceConsCollateDone(std::bitset<MAX_DST_CONNECTIONS> const mask) noexcept
{
    if (fenceConsBranch.done(mask)) {
        fenceCons.finalize();
        return true;
    }
    return false;
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::pack() to pack the fence array.
LwSciError
Packet::fenceProdPack(
    IpcBuffer& buf) const noexcept
{
    return fenceProd.pack(buf);
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::pack() to pack the fence array.
LwSciError
Packet::fenceConsPack(
    IpcBuffer& buf) const noexcept
{
    return fenceCons.pack(buf);
}

//! <b>Sequence of operations</b>
//! - Call Signals::syncsArrayGet() to retrieve the array of sync objects.
//! - Call TrackArray::unpack() to unpack the fence array.
// TODO: Maybe make Signals available through pktDesc
LwSciError
Packet::fenceProdUnpack(
    IpcBuffer& buf,
    Signals const& signal) noexcept
{
    return fenceProd.unpack(buf, signal.syncsArrayGet());
}

//! <b>Sequence of operations</b>
//! - Call Signals::syncsArrayGet() to retrieve the array of sync objects.
//! - Call TrackArray::unpack() to unpack the fence array.
// TODO: Maybe make Signals available through pktDesc
LwSciError
Packet::fenceConsUnpack(
    IpcBuffer& buf,
    Signals const& signal) noexcept
{
    return fenceCons.unpack(buf, signal.syncsArrayGet());
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::sizeGet() to obtain the size of the producer fence array
//! - For each fence:
//! -- Call TrackArray::peek() to look up the fence.
//! -- Call LwSciSyncFenceWait() to wait for the fence.
//! - Call TrackArray::reset() to clear out the fences.
LwSciError
Packet::fenceProdWait(
    LwSciSyncCpuWaitContext const ctx,
    uint64_t const timeout) noexcept
{
    size_t const fenceCount { fenceProd.sizeGet() };
    for (size_t i {0U}; fenceCount > i; ++i) {
        auto const fence { fenceProd.peek(i) };
        if (LwSciError_Success != fence.first) {
            return fence.first;
        }
        LwSciError const err
            { LwSciSyncFenceWait(&(fence.second->viewVal()), ctx, timeout) };
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // Clear the fences since they're expired, but keep the lock
    fenceProd.reset(true);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::sizeGet() to obtain the size of the consumer fence array
//! - For each fence:
//! -- Call TrackArray::peek() to look up the fence.
//! -- Call LwSciSyncFenceWait() to wait for the fence.
//! - Call TrackArray::reset() to clear out the fences.
LwSciError
Packet::fenceConsWait(
    LwSciSyncCpuWaitContext const ctx,
    uint64_t const timeout) noexcept
{
    size_t const fenceCount { fenceCons.sizeGet() };
    for (size_t i {0U}; fenceCount > i; ++i) {
        auto const fence { fenceCons.peek(i) };
        if (LwSciError_Success != fence.first) {
            return fence.first;
        }
        LwSciError const err
            { LwSciSyncFenceWait(&(fence.second->viewVal()), ctx, timeout) };
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // Clear the fences since they're expired, but keep the lock
    fenceCons.reset(true);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::reset() to clear out the fences and unlock the array.
void
Packet::fenceProdReset(void) noexcept
{
    // Note: This is safe to call even if the block doesn't track producer
    //   fences, which may happen when ilwoked from a generic function
    fenceProd.reset(false);
}

//! <b>Sequence of operations</b>
//! - Call BranchTrack::reset() to clear tracking of fence collation.
//! - Call TrackArray::reset() to clear out the fences and unlock the array.
void
Packet::fenceConsReset(void) noexcept
{
    // Note: This is safe to call even if the block doesn't track consumer
    //   fences, which may happen when ilwoked from a generic function
    fenceConsBranch.reset();
    fenceCons.reset(false);
}


//
// Packet teardown functions
//

// TODO: Pool lwrrently only checks whether the Used bits are set. That
//       is, the events have been received, but not yet passed off to the
//       application. Should we change that and not make packets available
//       until all events are processed?

//! <b>Sequence of operations</b>
//! - Atomically mark packet as deleted if not already done.
//! - If successful, set flag indicating pending deletion event.
bool
Packet::deleteSet(void) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Try to flip flag from false to true
    bool expected { false };
    if (zombie.compare_exchange_strong(expected, true)) {
        deleteEvent.store(true);
        return true;
    }
    return false;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

// Checks whether the packet instance is marked for deletion.
bool
Packet::deleteGet(void) const noexcept
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A13_5_3), "LwSciStream-ADV-AUTOSARC++14-001")
    return zombie;
}

//
// Packet payload queue management
//

//! <b>Sequence of operations</b>
//!   - Marks the new packet instance enqueued by calling the swapQueued()
//!     interface of the packet instance.
//!   - Swaps the next pointer of the new packet instance with the tail of
//!     PayloadQ.
//!   - Swaps the previous pointer of the packet instance at tail if any with
//!     the pointer to the new packet instance.
//!   - Makes the new packet instance the new tail of the PayloadQ. If the head
//!     of PayloadQ is not set, sets it to the new packet instance as well.
//!
//! \implements{20108010}
void
Packet::PayloadQ::enqueue(
    PacketPtr const&  newPacket) noexcept
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
//!
//! \implements{20108016}
PacketPtr
Packet::PayloadQ::dequeue(void) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Get head, and update queue if not null
    PacketPtr const getPacket { head };
    if (nullptr != getPacket) {

        // Mark packet as dequeued
        static_cast<void>(getPacket->swapQueued(false));

        // Update head pointer and clear retrieved packet's previous pointer
        head = getPacket->swapPrev(PacketPtr(nullptr));

        // Either clear new head's next pointer or clear tail pointer
        if (nullptr != head) {
            static_cast<void>(head->swapNext(PacketPtr(nullptr)));
        } else {
            tail = nullptr;
        }
    }

    return getPacket;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//!   - Validates the packet instance is enqueued and marks it dequeued by
//!     calling the swapQueued() interface of the packet instance.
//!   - If the extracted one has the previous neighbor, swaps this previous
//!     packet instance's next pointer with the extracted one's next pointer
//!     by calling the swapNext() interface.
//!   - If the extracted one has the next neighbor, swaps this next packet
//!     instance's previous pointer with the extracted one's previous pointer
//!     by calling the swapPrev() interface.
//!   - Clears the extracted packet instance's previous pointer and next
//!     pointer by calling the swapPrev() and swapNext() interfaces
//!     respectively.
//!
//! \implements{20108019}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
bool
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
Packet::PayloadQ::extract(
    Packet& oldPacket) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Unset queued flag, and avoid race condition with a dequeue operation
    if (!oldPacket.swapQueued(false)) {
        return false;
    }

    // Get packet's neighbors, clearing prev/next pointers
    PacketPtr const oldPrev { oldPacket.swapPrev(PacketPtr(nullptr)) };
    PacketPtr const oldNext { oldPacket.swapNext(PacketPtr(nullptr)) };

    // Make neighbors point to each other or update head/tail
    if (nullptr != oldPrev) {
        static_cast<void>(oldPrev->swapNext(oldNext));
    } else {
        tail = oldNext;
    }
    if (nullptr != oldNext) {
        static_cast<void>(oldNext->swapPrev(oldPrev));
    } else {
        head = oldPrev;
    }

    return true;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))

//! \brief Swaps payloadPrev with the @a newPrev.
//!
//! \param [in] newPrev: New smart pointer to a packet instance which needs to
//!   be swapped.
//!
//! \return Old value of the previous pointer.
//!
//! \implements{20040276}
PacketPtr
Packet::swapPrev(
    PacketPtr const& newPrev) noexcept
{
    PacketPtr const oldPrev { payloadPrev };
    payloadPrev = newPrev;
    return oldPrev;
}

//! \brief Swaps payloadNext with the @a newNext.
//!
//! \param [in] newNext: New smart pointer to a packet instance which needs to
//!   be swapped.
//!
//! \return Old value of the next pointer.
//!
//! \implements{20040279}
PacketPtr
Packet::swapNext(
    PacketPtr const& newNext) noexcept
{
    PacketPtr const oldNext { payloadNext };
    payloadNext = newNext;
    return oldNext;
}

//! \brief Sets payloadQueued with the @a newQueued to indicate whether
//!   the packet instance is in the PayloadQ or not.
//!
//! \param [in] newQueued: New value of queued flag.
//!
//! \return Old value of queued flag.
//!
//! \implements{20040282}
bool
Packet::swapQueued(
    bool const newQueued) noexcept
{
    bool const oldQueued { payloadQueued };
    payloadQueued = newQueued;
    return oldQueued;
}

LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))

} // namespace LwSciStream
