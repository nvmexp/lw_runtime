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

#include "lwscistream_common.h"
#include "sciwrap.h"
#include "ipcbuffer.h"
#include "trackarray.h"
#include "syncwait.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Call TrackArray::sizeInit() to initialize the array size.
LwSciError
Waiters::sizeInit(
    size_t const size) noexcept
{
    // Initialize array
    //   By default, all entries have NULL attributes.
    //   The used flag defaults to true at the endpoints, and false otherwise.
    return attrs.sizeInit(size, Entry(FillMode::User == fill,
                                      LwSciWrap::SyncAttr()));
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::peek() to query the entry.
bool
Waiters::usedPeek(
    size_t const index) const noexcept
{
    auto const entryVal { attrs.peek(index) };
    return (LwSciError_Success != entryVal.first)
           ? false
           : entryVal.second->first;
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::peek() to query the entry.
//! - Copy out the used flag.
LwSciError
Waiters::usedGet(
    size_t const index,
    bool& used) const noexcept
{
    // Retrieve the entry
    auto const val { attrs.peek(index) };
    if (LwSciError_Success != val.first) {
        return val.first;
    }

    // Output the value
    used = val.second->first;
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::peek() to query the entry.
//! - Call Wrapper::viewVal to retrieve the sync attributes.
LwSciSyncAttrList
Waiters::attrPeek(
    size_t const index) const noexcept
{
    auto const entryVal { attrs.peek(index) };
    return (LwSciError_Success != entryVal.first)
           ? nullptr
           : entryVal.second->second.viewVal();
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::get() to query the entry.
//! - Move the attibute to the output location.
LwSciError
Waiters::attrGet(
    size_t const index,
    LwSciWrap::SyncAttr& val) const noexcept
{
    // Retrieve the entry
    Entry entryVal {};
    LwSciError const err { attrs.get(index, entryVal) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Output the value
    val = std::move(entryVal.second);
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::peek() to view the current entry.
//! - Call TrackArray::set() to set the entry with the new flag.
//! - Call TrackArray::peek() again, and then
//!   LwSciWrap::SyncAttr::getErr() to check for duplication errors.
LwSciError
Waiters::usedSet(
    size_t const index,
    bool const used) noexcept
{
    // This function only allowed with FillMode::User
    if (FillMode::User != fill) {
        return LwSciError_AccessDenied;
    }

    // Only allowed if usage can be changed (i.e. in consumer)
    if (!usage) {
        return LwSciError_IlwalidOperation;
    }

    // Retrieve the old indexed entry
    auto const oldVal { attrs.peek(index) };
    if (LwSciError_Success != oldVal.first) {
        return oldVal.first;
    }

    // TODO: This is a bit awkward, since we end up unnecessarily
    //       duplicating any attribute list. Perhaps we should have
    //       a non-const version of TrackArray::peek() that lets us
    //       modify the entry directly if the array isn't locked.

    // Update the indexed entry
    LwSciError const err
    { attrs.set(index, Entry(used, oldVal.second->second)) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Retrieve the new indexed entry
    auto const newVal { attrs.peek(index) };
    if (LwSciError_Success != newVal.first) {
        return newVal.first;
    }

    // Return any duplication error
    return newVal.second->second.getErr();
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::peek() to view the current entry.
//! - Call TrackArray::set() to set the entry with the new attributes.
//! - Call TrackArray::peek() again, and then
//!   LwSciWrap::SyncAttr::getErr() to check for duplication errors.
LwSciError
Waiters::attrSet(
    size_t const index,
    LwSciWrap::SyncAttr const& attr) noexcept
{
    // This function only allowed with FillMode::User
    if (FillMode::User != fill) {
        return LwSciError_AccessDenied;
    }

    // Retrieve the old indexed entry
    auto const oldVal { attrs.peek(index) };
    if (LwSciError_Success != oldVal.first) {
        return oldVal.first;
    }

    // Update the indexed entry
    LwSciError const err
        { attrs.set(index, Entry(oldVal.second->first, attr)) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Retrieve the new indexed entry
    auto const newVal { attrs.peek(index) };
    if (LwSciError_Success != newVal.first) {
        return newVal.first;
    }

    // Return any duplication error
    return newVal.second->second.getErr();
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::finalize() to lock the array.
LwSciError
Waiters::doneSet(void) noexcept
{
    // This function only allowed with FillMode::User and ::Merge
    if ((FillMode::User != fill) && (FillMode::Merge != fill)) {
        return LwSciError_AccessDenied;
    }

    // Make sure not already done
    if (completed) {
        return LwSciError_AlreadyDone;
    }

    // Lock the array
    attrs.finalize();

    // Mark completed
    completed = true;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Create a new instance of Entry with the values to copy.
//! - Call TrackArray::sizeGet() to get the array size.
//! - Loop over all entries, calling TrackArray::set() to set the value,
//!   followed by TrackArray::peek() and Wrapper::getErr() to check for
//!   duplication errors.
//! - Call TrackArray::finalize() to lock the array.
LwSciError
Waiters::entryFill(
    LwSciWrap::SyncAttr const& attr) noexcept
{
    // This function only allowed with FillMode::User
    if (FillMode::User != fill) {
        return LwSciError_AccessDenied;
    }

    // Create Entry to duplicate
    Entry const val { true, attr };

    // Get array size
    size_t const size { attrs.sizeGet() };

    // Set all entries and check for duplication errors
    for (size_t i {0U}; size > i; ++i) {
        // Set entry
        LwSciError const err { attrs.set(i, val) };
        if (LwSciError_Success != err) {
            return err;
        }

        // Retrieve entry
        auto const check { attrs.peek(i) };
        if (LwSciError_Success != check.first) {
            return check.first;
        }

        // Check for duplication error
        if (LwSciError_Success != check.second->second.getErr()) {
            return check.second->second.getErr();
        }
    }

    // Lock the array
    attrs.finalize();

    // Mark completed
    completed = true;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::copy() to copy the incoming attributes.
LwSciError
Waiters::copy(
    Waiters const& orig) noexcept
{
    // This function only allowed with FillMode::Copy
    if (FillMode::Copy != fill) {
        return LwSciError_AccessDenied;
    }

    // Usage flags must match
    if (orig.usage != usage) {
        return LwSciError_InconsistentData;
    }

    // Copy the incoming data
    LwSciError const err { attrs.copy(orig.attrs) };
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
//! - Call TrackArray::sizeGet() to retrieve and validate array sizes.
//! - For each entry:
//! -- Call TrackArray::get() to retrieve incoming and current entries.
//! -- If both are used, call LwSciWrap::SyncAttr::merge() to merge the
//!    the attribute lists.
//! -- If incoming entry was used, call TrackArray::set() to replace
//!    the current entry.
LwSciError
Waiters::merge(
    Waiters const& orig) noexcept
{
    // This function only allowed with FillMode::Merge
    if (FillMode::Merge != fill) {
        return LwSciError_AccessDenied;
    }

    // Usage flags and list sizes must match
    if ((orig.usage != usage) || (orig.attrs.sizeGet() != attrs.sizeGet())) {
        return LwSciError_InconsistentData;
    }

    // Merge all entries in the list
    size_t const size { orig.attrs.sizeGet() };
    for (size_t i {0U}; size > i; ++i) {

        // Copy incoming entry
        Entry newEntry {};
        LwSciError err { orig.attrs.get(i, newEntry) };
        if (LwSciError_Success != err) {
            return err;
        }

        // Only need to merge if incoming entry is used
        if (newEntry.first) {

            // Retrieve current entry
            auto lwrrEntry { attrs.peek(i) };
            if (LwSciError_Success != lwrrEntry.first) {
                return lwrrEntry.first;
            }

            // If current entry is used, merge with incoming, producing
            //   NULL if either are NULL
            if (lwrrEntry.second->first) {
                newEntry.second.merge(lwrrEntry.second->second, false);
            }

            // Replace the entry
            err = attrs.set(i, newEntry);
            if (LwSciError_Success != err) {
                return err;
            }
        }
    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::pack() to pack the array.
LwSciError
Waiters::pack(
    IpcBuffer& buf) const noexcept
{
    assert(!buf.isC2CGet());
    return attrs.pack(buf);
}

//! <b>Sequence of operations</b>
//! - Call TrackArray::unpack() to unpack the array.
LwSciError
Waiters::unpack(
    IpcBuffer& buf) noexcept
{
    assert(!buf.isC2CGet());

    // Unpack attribute array
    LwSciError const err { attrs.unpack(buf) };
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
Waiters::clear(void) noexcept
{
    attrs.clear();
}

} // namespace LwSciStream
