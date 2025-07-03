//! \file
//! \brief LwSciStream tracking of element setup.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include <limits>
#include <utility>
#include <map>
#include "lwscistream_common.h"
#include "ipcbuffer.h"
#include "trackarray.h"
#include "elements.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Make sure fill mode is compatible.
//! - Lock the data for fill operation.
//! - Create ScopeExitSet object to unlock the data on any return.
//! - Make sure map can fit another element.
//! - Add type/attr pair to the list.
//! - Call LwSciWrap::BufAttr::getErr() to check for any errors in
//!   duplicating the attribute list, and remove the entry from the
//!   list if one oclwrred.
LwSciError
Elements::mapAdd(
    uint32_t const elemType,
    LwSciWrap::BufAttr const& elemBufAttr) noexcept
{
    // This function only allowed with FillMode::User
    if (FillMode::User != elemFill) {
        return LwSciError_AccessDenied;
    }

    // TODO :Could have a flag set by the user which signals when the
    //       event that allows mapAdd() has started, so it can be checked
    //       here, rather than in the calling function.

    // Try to change status from Ready to Filling
    DataStatus expected { DataStatus::Ready };
    if (!status.compare_exchange_strong(expected, DataStatus::Filling)) {
        if (DataStatus::Complete <= status.load()) {
            return LwSciError_NoLongerAvailable;
        } else {
            return LwSciError_Busy;
        }
    }

    // Restore status to Ready on return
    ScopeExitSet<DataStatus> restoreStatus(status, DataStatus::Ready);

    // Attempt to insert element in map, handling any exceptions
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        // Check whether another element can fit.
        if (MAX_INT_SIZE <= elemMap.size()) {
            return LwSciError_Overflow;
        }

        // Do the insertion
        // TODO: Decomposed declaration (auto const [entry, success]) would
        //   be cleaner but is only available with -std=c++1z or -std=gnu++1z
        //   (or a C++17 compiler, which is where it was officially added).
        auto const insertion
            { elemMap.emplace(elemType, Entry(elemMap.size(), elemBufAttr)) };

        // Any failure that doesn't throw an exception should be because
        //   entry already exists
        if (!insertion.second) {
            return LwSciError_AlreadyInUse;
        }

        // Check for any error in duplicating the attributes
        LwSciError const err { insertion.first->second.second.getErr() };
        if (LwSciError_Success != err) {
            static_cast<void>(elemMap.erase(insertion.first));
            return err;
        }
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
    } catch (...) {
        // Assume any exception is due to allocation failure
        return LwSciError_InsufficientMemory;
    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Make sure fill mode is compatible.
//! - Lock the data for fill operation.
//! - Create ScopeExitSet object to unlock the data on any return.
//! - For each entry in incoming lists, check whether the type already
//!   exists in this list.
//! -- If so, call LwSciWrap::BufAttr::merge() to merge the new attribute list
//!    into the existing one.
//! -- If not, add the new Entry if there is room.
LwSciError
Elements::mapMerge(
    Elements const& inElements) noexcept
{
    // This function only allowed with FillMode::Merge
    if (FillMode::Merge != elemFill) {
        return LwSciError_AccessDenied;
    }

    // Incoming list should be locked and not parsed
    assert((DataStatus::Locked == inElements.status) && !inElements.parsed);

    // Try to change status from Ready to Filling
    DataStatus expected { DataStatus::Ready };
    if (!status.compare_exchange_strong(expected, DataStatus::Filling)) {
        if (DataStatus::Complete <= status.load()) {
            return LwSciError_NoLongerAvailable;
        } else {
            return LwSciError_Busy;
        }
    }

    // Restore status to Ready on return
    ScopeExitSet<DataStatus> restoreStatus(status, DataStatus::Ready);

    // Loop over all entries in incoming list
    for (auto const oldEntry : inElements.elemMap) {

        // Search for matching entry in consolidated list
        auto newEntry { elemMap.find(oldEntry.first) };

        // If found, merge the incoming attributes, with NULL values ignored
        if (elemMap.end() != newEntry) {
            // Note: Any merge errors are preserved in the wrapper and will
            //   propagate to the eventual recipient. We lwrrently don't
            //   set an error on the block where they occur because the
            //   user has no way of knowing what went wrong.
            newEntry->second.second.merge(oldEntry.second.second, true);
        }

        // Otherwise insert a new entry
        else {
            // Attempt to insert element in map, handling any exceptions
            // Note: We don't check for errors in duplicating attribute lists
            //   at this time. Such errors are preserved in the map entries
            //   and ultimately make their way to the end user when queried,
            //   where they have to be duped again anyways.
            try {
                LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
                // Check whether another element can fit.
                if (MAX_INT_SIZE <= elemMap.size()) {
                    return LwSciError_Overflow;
                }

                // Do the insertion
                // TODO: See comment in mapAdd()
                auto const insertion {
                    elemMap.emplace(oldEntry.first,
                                    Entry(elemMap.size(),
                                          oldEntry.second.second))
                };

                // Any failure that doesn't throw an exception should be
                //  because entry already exists, but that is impossible
                //  since we only got here after finding that it didn't.
                assert(insertion.second);
                static_cast<void>(insertion);
                LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
            } catch (...) {
                // Assume any exception is due to allocation failure
                return LwSciError_InsufficientMemory;
            }
        }
    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Make sure fill mode is compatible.
//! - Change status from Ready to Complete.
//! - If requested, call dataParse() to transfer the data from map to array.
//! - Set flag indicating block can signal an LwSciStreamEventType_Elements
//!   event, if appropriate.
LwSciError
Elements::mapDone(
    bool const parse) noexcept
{
    // This function only allowed with FillMode::User and ::Merge
    if ((FillMode::User != elemFill) && (FillMode::Merge != elemFill)) {
        return LwSciError_AccessDenied;
    }

    // Try to mark as complete
    DataStatus expected { DataStatus::Ready };
    if (!status.compare_exchange_strong(expected, DataStatus::Complete)) {
        if (DataStatus::Complete <= status.load()) {
            return LwSciError_AlreadyDone;
        } else {
            return LwSciError_Busy;
        }
    }

    // Optional parsing
    if (parse) {
        LwSciError const err { dataParse() };
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // Set event flag
    eventReady.store(true);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Make sure map is complete.
//! - Lock the data so no other threads will interfere.
//! - Create ScopeExitSet object to unlock the data on any return.
//! - Make sure data is not already parsed.
//! - Initialize array to size of map by calling TrackArray::sizeInit().
//! - Move entries one at a time from map to array by calling
//!   TrackArray::set()
//! - Mark array as complete by calling TrackArray::finalize().
//! - Set parsed flag
//! - Free any remaining map resources.
LwSciError
Elements::dataParse(void) noexcept
{
    // Data must be completed
    if (DataStatus::Complete > status.load()) {
        return LwSciError_NotYetAvailable;
    }

    // Try to lock data
    DataStatus expected { DataStatus::Complete };
    if (!status.compare_exchange_strong(expected, DataStatus::Locked)) {
        if (DataStatus::Cleared == status.load()) {
            return LwSciError_NoLongerAvailable;
        } else {
            return LwSciError_Busy;
        }
    }

    // Restore status to Complete on return
    ScopeExitSet<DataStatus> restoreStatus(status, DataStatus::Complete);

    // Check whether data is already parsed
    if (parsed) {
        return LwSciError_AlreadyDone;
    }

    // Make sure map size is within limits
    //   The mapAdd function already prevents this, so this error will
    //   never be reached. But AUTOSAR/CERT will complain if we don't do it.
    if (elemMap.size() >
        static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return LwSciError_Overflow;
    }
    // Initialize size of array
    LwSciError err
        { elemArray.sizeInit(static_cast<uint32_t>(elemMap.size())) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Loop over all map entries, moving to array
    for (auto it { elemMap.begin() }; it != elemMap.end(); /*nop*/) {

        // Extract the info from the map entry
        uint32_t const elemType { it->first };
        uint32_t const index { it->second.first };
        LwSciWrap::BufAttr elemAttr { std::move(it->second.second) };

        // Delete the map entry, advancing the iterator
        it = elemMap.erase(it);

        // Insert in the array. There should be no failures, because we're
        //   just moving objects without dups, and the array is already
        //   allocated. But just in case, if a failure oclwrs, we continue
        //   to parse the whole thing. This way at least the object is in
        //   a somewhat consistent state.
        LwSciError const setErr
            { elemArray.set(index, Entry(elemType, std::move(elemAttr))) };
        if (LwSciError_Success != setErr) {
            err = setErr;
        }
    }

    // Finalize array
    elemArray.finalize();

    // Reclaim any remaining allocations in the map
    elemMap.clear();

    // Marked parsed
    parsed = true;

    return err;
}

//! <b>Sequence of operations</b>
//! - Make sure fill mode is compatible.
//! - Lock the data so no other threads will interfere.
//! - Copy the map or array, depending on whether incoming object is parsed.
//! - Copy the parsed flag and mark the data as complete.
//! - Unlock the data.
//! - If requested, transfer data from map to array.
//! - Set flag indicating block can signal an LwSciStreamEventType_Elements
//!   event, if appropriate.
LwSciError
Elements::dataCopy(
    Elements const& orig,
    bool const parse) noexcept
{
    // This function only allowed with FillMode::Copy
    if ((FillMode::Copy != elemFill) && (FillMode::User != elemFill)) {
        return LwSciError_AccessDenied;
    }

    // Try to change status from Ready to Filling
    DataStatus expected { DataStatus::Ready };
    if (!status.compare_exchange_strong(expected, DataStatus::Filling)) {
        if (DataStatus::Complete <= status.load()) {
            return LwSciError_AlreadyDone;
        } else {
            return LwSciError_Busy;
        }
    }

    // Incoming data should be locked
    assert(DataStatus::Locked == orig.status);

    // Copy the map or array data
    // Note: In addition to possible allocation failures when the map or
    //   array itself is copied, which would return an immediate error,
    //   there is also a possibility that LwSciBuf will fail to dup an
    //   attribute list. The error will be tracked by the wrapper in the
    //   object. We don't check it now, but will report an error if and
    //   when the attribute list is queried. We'll need to do another dup
    //   at that time anyways, giving it another opportunity to fail.
    LwSciError err { LwSciError_Success };
    if (orig.parsed) {
        err = elemArray.copy(orig.elemArray);
    } else {
        try {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            elemMap = orig.elemMap;
            LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
        } catch (...) {
            err = LwSciError_InsufficientMemory;
        }
    }

    // On failure, revert to ready status
    if (LwSciError_Success != err) {
        status.store(DataStatus::Ready);
        return err;
    }

    // Copy parsed flag and mark data completed
    parsed = orig.parsed;
    status.store(DataStatus::Complete);

    // If requested, parse the data
    if (parse && !parsed) {
        err = dataParse();
    }
    if (LwSciError_Success != err) {
        return err;
    }

    // Set event flag
    eventReady.store(true);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Make sure map is complete.
//! - Lock the data so no other threads will interfere.
//! - Create ScopeExitSet object to unlock the data on any return.
//! - Call IpcBuffer::packVal() to pack the parsed flag.
//! - Depending on whether the data has been parsed, call either
//!   TrackArray::pack() to pack the array or ipcBufferPack()
//!   to pack the map.
LwSciError
Elements::dataPack(
    IpcBuffer& buf) noexcept
{
    // Data must be completed
    if (DataStatus::Complete > status.load()) {
        return LwSciError_NotYetAvailable;
    }

    // Try to lock data
    DataStatus expected { DataStatus::Complete };
    if (!status.compare_exchange_strong(expected, DataStatus::Locked)) {
        if (DataStatus::Cleared == status.load()) {
            return LwSciError_NoLongerAvailable;
        } else {
            return LwSciError_Busy;
        }
    }

    // Restore status to Complete on return
    ScopeExitSet<DataStatus> restoreStatus(status, DataStatus::Complete);

    // Pack parsed flag
    LwSciError err { buf.packVal(parsed) };

    // If successful, pack either the map or array, as appropriate
    if (LwSciError_Success == err) {
        err = parsed ? elemArray.pack(buf) : ipcBufferPack(buf, elemMap);
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Make sure fill mode is compatible.
//! - Lock the data so no other threads will interfere.
//! - Call IpcBuffer::unpackVal() to unpack the parsed flag.
//! - Depending on whether the data has been parsed, call either
//!   TrackArray::unpack() to unpack the array or ipcBufferUnpack()
//!   to unpack the map.
//! - Mark data complete and set flag indicating block can signal an
//!   LwSciStreamEventType_Elements event, if appropriate.
LwSciError
Elements::dataUnpack(
    IpcBuffer& buf) noexcept
{
    // This function only allowed with FillMode::IPC
    if (FillMode::IPC != elemFill) {
        return LwSciError_AccessDenied;
    }

    // Try to change status from Ready to Filling
    DataStatus expected { DataStatus::Ready };
    if (!status.compare_exchange_strong(expected, DataStatus::Filling)) {
        return LwSciError_AlreadyDone;
    }

    // Unpack parsed flag
    LwSciError err { buf.unpackVal(parsed) };

    // If successful, unpack either the map or array, as appropriate
    if (LwSciError_Success == err) {
        err = parsed ? elemArray.unpack(buf) : ipcBufferUnpack(buf, elemMap);
    }

    // Update stats based on success or failure
    status.store((LwSciError_Success == err) ? DataStatus::Complete
                                             : DataStatus::Ready);

    // Set event flag
    if (DataStatus::Complete == status) {
        eventReady.store(true);
    }

    return err;
}

//! <b>Sequence of operations</b>
//! - Check status is complete.
//! - Lock the data so no other threads will interfere.
//! - Perform the provided action.
//! - If requested, clear the data.
//! - Unlock the data, marking status appropriately.
LwSciError
Elements::dataSend(
    Action const& sendAction,
    bool const clear) noexcept
{
    // Data must be completed
    if (DataStatus::Complete > status.load()) {
        return LwSciError_NotYetAvailable;
    }

    // Try to lock data
    DataStatus expected { DataStatus::Complete };
    if (!status.compare_exchange_strong(expected, DataStatus::Locked)) {
        if (DataStatus::Cleared == status.load()) {
            return LwSciError_NoLongerAvailable;
        } else {
            return LwSciError_Busy;
        }
    }

    // Peform specified action
    sendAction(*this);

    // If clear is requested, free the data
    if (clear) {
        elemMap.clear();
        elemArray.clear();
    }

    // Unlock the data
    status.store(clear ? DataStatus::Cleared : DataStatus::Complete);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Makes sure map is complete.
//! - Lock the data so no other threads will interfere.
//! - Free contents of map and array.
//! - Set status to Cleared.
LwSciError
Elements::dataClear(void) noexcept
{
    // Data must be completed
    if (DataStatus::Complete > status.load()) {
        return LwSciError_NotYetAvailable;
    }

    // Try to lock data
    // Note: Unlike other functions, being already cleared is not treated
    //       as a failure, but we can skip the rest if that's the case.
    DataStatus expected { DataStatus::Complete };
    if (!status.compare_exchange_strong(expected, DataStatus::Locked)) {
        if (DataStatus::Cleared == status.load()) {
            return LwSciError_Success;
        } else {
            return LwSciError_Busy;
        }
    }

    // Free the data
    elemMap.clear();
    elemArray.clear();

    // Set status to Cleared
    status.store(DataStatus::Cleared);
    eventReady.store(false);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Makes sure map is complete and parsed.
//! - Lock the data so no other threads will interfere.
//! - Create ScopeExitSet object to unlock the data on any return.
//! - Query the size from the array.
LwSciError
Elements::sizeGet(
    uint32_t& size) noexcept
{
    // Data must be completed and parsed
    if ((DataStatus::Complete > status.load()) || !parsed) {
        return LwSciError_NotYetAvailable;
    }

    // Try to lock data
    DataStatus expected { DataStatus::Complete };
    if (!status.compare_exchange_strong(expected, DataStatus::Locked)) {
        if (DataStatus::Cleared == status.load()) {
            return LwSciError_NoLongerAvailable;
        } else {
            return LwSciError_Busy;
        }
    }

    // Restore status to Complete on return
    ScopeExitSet<DataStatus> restoreStatus(status, DataStatus::Complete);

    // Query the size
    // TODO: We've taken care to make sure it never exceeds 31-bits, so
    //       hopefully can RFD any coverity violations.
    size = static_cast<uint32_t>(elemArray.sizeGet());

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Makes sure map is complete and parsed.
//! - Lock the data so no other threads will interfere.
//! - Create ScopeExitSet object to unlock the data on any return.
//! - Query the type from the array.
LwSciError
Elements::typeGet(
    size_t const index,
    uint32_t& elemType) noexcept
{
    // Data must be completed and parsed
    if ((DataStatus::Complete > status.load()) || !parsed) {
        return LwSciError_NotYetAvailable;
    }

    // Try to lock data
    DataStatus expected { DataStatus::Complete };
    if (!status.compare_exchange_strong(expected, DataStatus::Locked)) {
        if (DataStatus::Cleared == status.load()) {
            return LwSciError_NoLongerAvailable;
        } else {
            return LwSciError_Busy;
        }
    }

    // Restore status to Complete on return
    ScopeExitSet<DataStatus> restoreStatus(status, DataStatus::Complete);

    // Query entry
    auto const val { elemArray.peek(index) };
    if (LwSciError_Success != val.first) {
        return val.first;
    }
    elemType = val.second->first;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Makes sure map is complete and parsed.
//! - Lock the data so no other threads will interfere.
//! - Create ScopeExitSet object to unlock the data on any return.
//! - Query the attribute list from the array, duplicating the entry.
LwSciError
Elements::attrGet(
    size_t const index,
    LwSciBufAttrList& elemAttr) noexcept
{
    // Data must be completed and parsed
    if ((DataStatus::Complete > status.load()) || !parsed) {
        return LwSciError_NotYetAvailable;
    }

    // Try to lock data
    DataStatus expected { DataStatus::Complete };
    if (!status.compare_exchange_strong(expected, DataStatus::Locked)) {
        if (DataStatus::Cleared == status.load()) {
            return LwSciError_NoLongerAvailable;
        } else {
            return LwSciError_Busy;
        }
    }

    // Restore status to Complete on return
    ScopeExitSet<DataStatus> restoreStatus(status, DataStatus::Complete);

    // Query and duplicate entry
    Entry val { };
    LwSciError const err { elemArray.get(index, val) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Extract attribute list
    return val.second.takeVal(elemAttr);
}

//! <b>Sequence of operations</b>
//! - Try to flip event flag from true to false.
bool
Elements::eventGet(void) noexcept
{
    bool expected { true };
    return eventReady.compare_exchange_strong(expected, false);
}

} // namespace LwSciStream
