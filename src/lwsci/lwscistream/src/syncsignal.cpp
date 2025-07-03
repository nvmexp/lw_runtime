//! \file
//! \brief LwSciStream tracking of signaller sync setup.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistream_common.h"
#include "sciwrap.h"
#include "ipcbuffer.h"
#include "trackarray.h"
#include "syncwait.h"
#include "syncsignal.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Save sizes.
//! - Call TrackArray::sizeInit() to initialize the array size.
LwSciError
Signals::sizeInit(
    size_t const paramEndpointCount,
    size_t const paramElementCount) noexcept
{
    // Save sizes
    endpointCount = paramEndpointCount;
    elementCount = paramElementCount;

    // Initialize array
    return syncs.sizeInit(endpointCount * elementCount);
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::get() to query the entry.
//! - Call Wrapper::viewVal() to retrieve the sync object.
LwSciSyncObj
Signals::syncPeek(
    size_t const endIndex,
    size_t const elemIndex) const noexcept
{
    // TODO: Pointless CERT overflow check?
    size_t const index { (endIndex * elementCount) + elemIndex };

    // Peek at the entry
    auto const entryVal { syncs.peek(index) };
    return (LwSciError_Success != entryVal.first)
           ? nullptr
           : entryVal.second->viewVal();
}

// Temporary function until new fence APIs are in place
LwSciSyncObj
Signals::syncPeek(
    size_t const index) const noexcept
{
    // Peek at the entry
    auto const entryVal { syncs.peek(index) };
    return (LwSciError_Success != entryVal.first)
           ? nullptr
           : entryVal.second->viewVal();
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::get() to query the entry.
LwSciError
Signals::syncGet(
    size_t const endIndex,
    size_t const elemIndex,
    LwSciWrap::SyncObj& val) const noexcept
{
    // Validate indices
    if ((endIndex >= endpointCount) || (elemIndex >= elementCount)) {
        return LwSciError_IndexOutOfRange;
    }

    // TODO: Pointless CERT overflow check?
    size_t const index { (endIndex * elementCount) + elemIndex };

    // Retrieve the entry
    return syncs.get(index, val);
}

//! <b>Sequence of operations</b>
//! - If attributes are provided, call validate() to validate against them.
//! - Call TrackArray::set() to set the entry.
//! - Call TrackArray::peek() to check the entry.
//! - Call Wrapper::getErr() to check for duplication errors.
LwSciError
Signals::syncSet(
    size_t const endIndex,
    size_t const elemIndex,
    LwSciWrap::SyncObj const& val) noexcept
{
    // Validate indices
    if ((endIndex >= endpointCount) || (elemIndex >= elementCount)) {
        return LwSciError_IndexOutOfRange;
    }

    // TODO: Pointless CERT overflow check?
    size_t const index { (endIndex * elementCount) + elemIndex };

    // Validate against waiter attributes
    if (nullptr != waiter) {
        LwSciError const verr { validate(val, elemIndex) };
        if (LwSciError_Success != verr) {
            return verr;
        }
    }

    // Update the indexed entry
    LwSciError const err { syncs.set(index, val) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Retrieve the new indexed entry
    auto const newVal { syncs.peek(index) };
    if (LwSciError_Success != newVal.first) {
        return newVal.first;
    }

    // Return any duplication error
    return newVal.second->getErr();
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::finalize() to lock the array.
LwSciError
Signals::doneSet(void) noexcept
{
    // This function only allowed with FillMode::User and ::Collate
    if ((FillMode::User != fill) && (FillMode::Collate != fill)) {
        return LwSciError_AccessDenied;
    }

    // Make sure not already done
    if (completed) {
        return LwSciError_AlreadyDone;
    }

    // Lock the array
    syncs.finalize();

    // Mark completed
    completed = true;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::sizeGet() to get the array size.
//! - Loop over all entries, calling TrackArray::set() to set the value,
//!   followed by TrackArray::peek() and Wrapper::getErr() to check for
//!   duplication errors.
//! - Call TrackArray::finalize() to lock the array.
LwSciError
Signals::syncFill(
    LwSciWrap::SyncObj const& sync) noexcept
{
    // This function only allowed with FillMode::User
    if (FillMode::User != fill) {
        return LwSciError_AccessDenied;
    }

    // Get array size
    size_t const size { syncs.sizeGet() };

    // Set all entries and check for duplication errors
    for (size_t i {0U}; size > i; ++i) {
        // Set entry
        LwSciError const err { syncs.set(i, sync) };
        if (LwSciError_Success != err) {
            return err;
        }

        // Retrieve entry
        auto const check { syncs.peek(i) };
        if (LwSciError_Success != check.first) {
            return check.first;
        }

        // Check for duplication error
        if (LwSciError_Success != check.second->getErr()) {
            return check.second->getErr();
        }
    }

    // Lock the array
    syncs.finalize();

    // Mark completed
    completed = true;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - If attributes are provided, loop over all entries in incoming data.
//! -- Call TrackArray::peek() to look up the sync object.
//! -- Call validate() to validate against the attributes.
//! - Call TrackArray::copy() to copy the incoming attributes.
LwSciError
Signals::copy(
    Signals const& orig) noexcept
{
    // Validate all entries against waiter attributes
    if (nullptr != waiter) {
        for (size_t j {0U}; endpointCount > j; ++j) {
            for (size_t i {0U}; elementCount > i; ++i) {

                // TODO: Pointless CERT overflow check?
                size_t const index { (j * elementCount) + i };

                // Peek at incoming object
                auto const entryVal { orig.syncs.peek(index) };
                if (LwSciError_Success != entryVal.first) {
                    return entryVal.first;
                }

                // Validate with attributes
                LwSciError const verr { validate(*(entryVal.second), i) };
                if (LwSciError_Success != verr) {
                    return verr;
                }
            }
        }
    }

    // Copy the incoming data
    LwSciError const err { syncs.copy(orig.syncs) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Mark completed and signal event
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    completed = true;
    event.store(true);
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::collate() to copy the data.
LwSciError
Signals::collate(
    Signals const& orig,
    size_t const endRangeStart,
    size_t const endRangeCount) noexcept
{
    // TODO: Pointless CERT overflow check?
    size_t const rangeStart { endRangeStart * elementCount };
    size_t const rangeCount { endRangeCount * elementCount };

    // Copy incoming data
    return syncs.collate(orig.syncs, rangeStart, rangeCount);
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::pack() to pack the array.
LwSciError
Signals::pack(
    IpcBuffer& buf) const noexcept
{
    // Pack the array
    return syncs.pack(buf);
}

//! <b>Sequence of operations</b>
//! - Call Waiters::attrsArrayGet() to retrieve the attribute array.
//! - Call TrackArray::unpack() to unpack the array.
LwSciError
Signals::unpack(
    IpcBuffer& buf) noexcept
{
    // Waiters info must be provided
    if (nullptr == waiter) {
        return LwSciError_InsufficientData;
    }

    // Unpack the array
    LwSciError const err { syncs.unpack(buf, waiter->attrsArrayGet()) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Mark completed
    completed = true;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::clear() to free the vector contents.
void
Signals::clear(void) noexcept
{
    syncs.clear();
}

//! <b>Sequence of operations</b>
//! - Call Wrapper::viewVal() to extract sync object from wrapper.
//! - Call Waiters::peekAttr() to extract the corresponding waiter attributes.
//! - Call LwSciSyncObjGetAttrList() to extrat the object's attributes.
//! - Call LwSciSyncAttrListValidateReconciled() to validate the attributes.
LwSciError
Signals::validate(
    LwSciWrap::SyncObj const& syncWrap,
    size_t const elemIndex) noexcept
{
    // Get sync object
    //   If NULL, always compatible
    LwSciSyncObj const syncObj { syncWrap.viewVal() };
    if (nullptr == syncObj) {
        return LwSciError_Success;
    }

    // Retrieve corresponding attribute list
    //   If NULL, incompatible
    LwSciSyncAttrList const syncWaitAttr { waiter->attrPeek(elemIndex) };
    if (nullptr == syncWaitAttr) {
        return LwSciError_InconsistentData;
    }

    // Obtain the objects attribute list
    LwSciSyncAttrList syncObjAttr {};
    LwSciError err { LwSciSyncObjGetAttrList(syncObj, &syncObjAttr) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Validate the attribute lists
    bool isValid;
    err = LwSciSyncAttrListValidateReconciled(syncObjAttr,
                                              &syncWaitAttr,
                                              ONE,
                                              &isValid);
    if (LwSciError_Success != err) {
        return err;
    }
    if (!isValid) {
        return LwSciError_InconsistentData;
    }

    return LwSciError_Success;
}

} // namespace LwSciStream
