//! \file
//! \brief Utility to track setting/sending/receiving object arrays
//!
//! \copyright
//! Copyright (c) 2019-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef TRACK_ARRAY_H
#define TRACK_ARRAY_H
#include <cstdint>
#include <cassert>
#include <atomic>
#include <cmath>
#include <vector>
#include <utility>
#include <cstddef>
#include <iostream>
#include <array>
#include <limits>
#include <functional>
#include <lwscierror.h>
#include "covanalysis.h"
#include "lwscistream_common.h"
#include "ipcbuffer.h"
#include "ipcbuffer_sciwrap.h"

namespace LwSciStream {

//! \brief Utility to track an array of objects.
//!
//! - During both init and runtime, Blocks accumulate sets of various objects
//!   which are passed up and/or down stream when the user indicates the set
//!   is ready. They are then available to query when they arrive at their
//!   destination(s). Some sets of objects are sparse, while others must be
//!   completely filled. This object manages gathering the data and ensures
//!   that all appropriate error checking is done in a uniform fashion for
//!   all such sets.
//!
//! - The size of the set is either specified before any of the entries are
//!   provided or determined after the list has been created based on the
//!   number of entries, depending on the fill mode.
//!
//! - This object is not thread-safe on its own. For non-const operations,
//!   the owning object to use a mutex or otherwise protect against
//!   simultaneous modificationi in multiple threads. The owner is also
//!   expected to finalize the tracker before passing it to other objects,
//!   so that it will not be modified while being read by others.
//!
//! \tparam  T:
//    TODO: Fill this in as uses of the object are added. Will probably be
//          the set of all packable objects as defined for the IpcBuffer
//          functions, but that is not certain yet.
//
// TODO: This template lwrrently omits checking whether an indexed entry
//   has already been set, and simply allows it to be overridden, as long
//   as the tracker has not been locked yet. Given the new paradigm for
//   entering data, it is not clear if we need to do such a check any
//   more. We can add the check later if we deem it necessary, or
//   remove this comment if not.
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A14_1_1), "LwSciStream-ADV-AUTOSARC++14-002")
template <class T>
class TrackArray final
{
public:
    // Allow classes with the same template to access each others data.
    // TODO: This will require an RFD. Otherwise we have to add more functions.
    template <class A> friend class TrackArray;

    // Constructor
    TrackArray(FillMode const paramFill, bool const paramRuntime) :
        fill(paramFill),
        runtime(paramRuntime),
        locked(false),
        data()
    {
        // TODO: If, as expected, the allowed template types become
        //       the set of all packable types, add a static_assert
        //       here to ensure only compatible types are used. This
        //       will eliminate the need for the A14-1-1 deviation.
    }

    // Use the default destructor and disallow other core operations
    ~TrackArray(void) noexcept                                = default;
    TrackArray(void) noexcept                                 = delete;
    TrackArray(const TrackArray&) noexcept                    = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    TrackArray(TrackArray&&) noexcept                         = delete;
    auto operator=(const TrackArray&) noexcept -> TrackArray& = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    auto operator=(TrackArray&&) noexcept -> TrackArray&      = delete;

    //! \brief Initializes tracker to begin receiving data from user.
    //!   Must be called before using set().
    //!   For runtime arrays, also must be used before copy() or unpack().
    //!
    //! \param [in] size: The array size.
    //! \param [in] initValue: Value with which to initialize vector.
    //!
    //! <b>Sequence of operations</b>
    //! - Checks whether the size is already set.
    //! - Allocates memory for the vector, initializing with @a initValue.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: Initialization succeeded.
    //! * LwSciError_AlreadyDone: Init already done.
    //! * LwSciError_InsufficientMemory: Failed to allocate vector memory.
    LwSciError sizeInit(
        size_t const size,
        T const& initValue=T()) noexcept
    {
        // Only initialize once
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (0UL != data.size()) {
            return LwSciError_AlreadyDone;
        }

        // Allocate the vector
        try {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            data.resize(size, initValue);
            LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
        } catch (...) {
            return LwSciError_InsufficientMemory;
        }

        return LwSciError_Success;
    };

    //! \brief Query the current size.
    //!
    //! \return size_t, The size of the array.
    size_t sizeGet(void) const noexcept
    {
        return data.size();
    };

    //! \brief Query entry corresponding to @a index and return a pointer
    //!   to the entry. This is used when the caller is sure the array
    //!   data will persist for the duration needed, and wants to avoid
    //!   the cost of a copy.
    //!
    //! \param [in] index: Index of the entry to be used.
    //!   Valid values: [0, data().size - 1]
    //!
    //! <b>Sequence of operations</b>
    //! - Makes sure index is valid.
    //! - Retrieves the data entry
    //!
    //! \return (LwSciError, T const*) pair
    //! first: contains completion code of this operation
    //! * LwSciError_Success: The entry was retrieved successfully.
    //! * LwSciError_IlwalidState: The tracker is not locked.
    //! * LwSciError_IndexOutOfRange: The index value is invalid.
    //! second: pointer to the entry in the array, or nullptr on failure
    auto peek(size_t const index) const noexcept
    {
        using rvType = std::pair<LwSciError, T const*>;

        // Make sure index is in bounds
        if (index >= data.size()) {
            return rvType(LwSciError_IndexOutOfRange, nullptr);
        }

        // Retrieve data
        return rvType(LwSciError_Success, &data[index]);
    };

    //! \brief Query entry corresponding to @a index and stores a copy
    //!   in @a val. This is used when the array data may not persist
    //!   after the call.
    //!
    //! \param [in] index: Index of the entry to be used.
    //!   Valid values: [0, data().size - 1]
    //! \param [in,out] val: Reference to location to store value.
    //!
    //! <b>Sequence of operations</b>
    //! - Makes sure tracker is in state to query values and index is valid.
    //! - Copies the data to the provided location.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The entry was retrieved successfully.
    //! * LwSciError_NotYetAvailable: The tracker is not locked.
    //! * LwSciError_IndexOutOfRange: The index value is invalid.
    LwSciError get(size_t const index, T& val) const noexcept
    {
        // Must be locked
        if (!locked) {
            return LwSciError_NotYetAvailable;
        }

        // Make sure index is in bounds
        if (index >= data.size()) {
            return LwSciError_IndexOutOfRange;
        }

        // Retrieve data
        val = data[index];

        return LwSciError_Success;
    };

    //! \brief Store @a val in entry corresponding to @a index.
    //!
    //! \param [in] index: Index of the entry to be used.
    //!   Valid values: [0, data().size - 1]
    //! \param [in] val: Value to be stored.
    //!
    //! <b>Sequence of operations</b>
    //! - Makes sure tracker is in state to insert values and index is valid.
    //! - Copies the value to the vector.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The entry was stored successfully.
    //! * LwSciError_NoLongerAvailable: The tracker is locked.
    //! * LwSciError_IlwalidOperation: The FillMode is not User or Merge.
    //! * LwSciError_IndexOutOfRange: The index value is invalid.
    LwSciError set(size_t const index, T const& val) noexcept
    {
        // Fill mode must be User or Merge
        if ((FillMode::User != fill) && (FillMode::Merge != fill)) {
            return LwSciError_IlwalidOperation;
        }

        // Must not be locked
        if (locked) {
            return LwSciError_NoLongerAvailable;
        }

        // Make sure index is in bounds
        if (index >= data.size()) {
            return LwSciError_IndexOutOfRange;
        }

        // Store data
        data[index] = val;

        return LwSciError_Success;
    };

    //! \brief Store @a val in all entries.
    //!
    //! \param [in] val: Value to be stored.
    //!
    //! <b>Sequence of operations</b>
    //! - Makes sure tracker is in state to insert values.
    //! - Loops over all vector entries, copying the value.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The entry was stored successfully.
    //! * LwSciError_NoLongerAvailable: The tracker is locked.
    //! * LwSciError_IlwalidOperation: The FillMode is not User.
    LwSciError setAll(T const& val) noexcept
    {
        // Fill mode must be User
        if (FillMode::User != fill) {
            return LwSciError_IlwalidOperation;
        }

        // Must not be locked
        if (locked) {
            return LwSciError_NoLongerAvailable;
        }

        // Store data
        for (size_t i {0U}; data.size() > i; ++i) {
            data[i] = val;
        }

        // Mark complete
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        locked = true;

        return LwSciError_Success;
    };

    //! \brief Copy data from another array.
    //   Note: We use a named function rather than operator= because
    //         it allows us to add error handling.
    //!
    //! \param [in] orig: Array to copy data from.
    //!
    //! <b>Sequence of operations</b>
    //! - Makes sure fill mode is Copy.
    //! - Makes sure tracker is unlocked.
    //! - If runtime array, make sure data is allocated.
    //! - Use default vector operator= to copy, catching exceptions.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The entry was unpacked successfully.
    //! * LwSciError_IlwalidOperation: The FillMode is not Copy or User.
    //! * LwSciError_AlreadyDone: The data has already been filled in.
    //! * LwSciError_NotInitialized: Memory not pre-allocated.
    //! * LwSciError_InconsistentData: Pre-allocated memory size does not
    //!   match.
    //! * LwSciError_InsufficientMemory: Unable to allocate memory for the
    //!   the data.
    LwSciError copy(TrackArray<T> const& orig) noexcept
    {
        // Fill mode must be Copy
        if ((FillMode::Copy != fill) && (FillMode::User != fill)) {
            return LwSciError_IlwalidOperation;
        }

        // If locked, copy was already done
        assert(orig.locked);
        if (locked) {
            return LwSciError_AlreadyDone;
        }

        // If runtime array, must have pre-allocated memory
        if (runtime && (0U == data.size())) {
            return LwSciError_NotInitialized;
        }

        // Size of any pre-allocated memory must match
        if ((0U != data.size()) && (data.size() != orig.data.size())) {
            return LwSciError_InconsistentData;
        }

        // Copy entire data with operator=()
        try {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            data = orig.data;
            LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
        } catch (...) {
            return LwSciError_InsufficientMemory;
        }

        // Lock
        locked = true;

        return LwSciError_Success;
    }

    //! \brief Copy data from another array into a subrange.
    //!
    //! \param [in] orig: Array to copy data from.
    //! \param [in] rangeStart: First index in range to write.
    //! \param [in] rangeCount: Number of entries in range to write.
    //!
    //! <b>Sequence of operations</b>
    //! - Makes sure fill mode is Collate.
    //! - Makes sure tracker is unlocked.
    //! - Validates size of range and data.
    //! - Copy incoming data to subrange.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_IlwalidOperation: The FillMode is not Collate.
    //! * LwSciError_AlreadyDone: Array was already filled in.
    //! * LwSciError_InconsistentData: Incoming size does not match range.
    //! * LwSciError_IndexOutOfRange: The data does not have preallocated
    //!   size sufficient to fit the range.
    LwSciError collate(
        TrackArray<T> const& orig,
        size_t const rangeStart,
        size_t const rangeCount) noexcept
    {
        // Fill mode must be Collate
        if (FillMode::Collate != fill) {
            return LwSciError_IlwalidOperation;
        }

        // If locked, copy was already done.
        assert(orig.locked);
        if (locked) {
            return LwSciError_AlreadyDone;
        }

        // Size of range must match the incoming size
        if (orig.data.size() != rangeCount) {
            return LwSciError_InconsistentData;
        }

        // Must have preallocated memory large enough to fit the range
        if ((data.size() <= rangeStart) ||
            (data.size() <  rangeCount) ||
            ((data.size() - rangeCount) < rangeStart)) {
            return LwSciError_IndexOutOfRange;
        }

        // Copy each entry in range
        for (size_t i { 0U }; rangeCount > i; i++) {
            data[rangeStart + i] = orig.data[i];
        }

        return LwSciError_Success;
    }

    //! \brief Pack the array for sending over IPC.
    //!
    //! <b>Sequence of operations</b>
    //! - Makes sure tracker is locked.
    //! - Packs the data to the buffer.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The entry was stored successfully.
    //! * LwSciError_InsufficientData: The array was not completely filled in.
    //! * Any error returned by ipcBufferPack().
    LwSciError pack(IpcBuffer& buf) const noexcept
    {
        // Can only pack the data after its locked
        if (!locked) {
            return LwSciError_InsufficientData;
        }

        // Use the templated IpcBuffer vector packing function
        return ipcBufferPack(buf, data);
    }

    //! \brief Unpack the array from data received over IPC.
    //!
    //! <b>Sequence of operations</b>
    //! - Makes sure fill mode is IPC.
    //! - Makes sure tracker is unlocked.
    //! - If runtime array, make sure data is allocated.
    //! - Unpacks the data from the buffer.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The entry was unpacked successfully.
    //! * LwSciError_IlwalidOperation: The FillMode is not IPC.
    //! * LwSciError_AlreadyDone: The data was already unpacked.
    //! * LwSciError_NotInitialized: Memory not pre-allocated.
    //! * Any error returned by ipcBufferUnpack().
    LwSciError unpack(IpcBuffer& buf) noexcept
    {
        // Fill mode must be IPC
        if (FillMode::IPC != fill) {
            return LwSciError_IlwalidOperation;
        }

        // If locked, data was already unpcked
        if (locked) {
            return LwSciError_AlreadyDone;
        }

        // If runtime array, must have pre-allocated memory
        if (runtime && (0U == data.size())) {
            return LwSciError_NotInitialized;
        }

        // On success or failure, lock
        locked = true;

        // Use the templated IpcBuffer vector unpacking function
        return ipcBufferUnpack(buf, data);
    }

    //! \brief Unpack the array from data received over IPC, using
    //!   auxiliary data.
    //!
    //! <b>Sequence of operations</b>
    //! - Makes sure fill mode is IPC.
    //! - Makes sure tracker is unlocked and aux tracker is locked.
    //! - If runtime array, make sure data is allocated.
    //! - Unpacks the data from the buffer using the aux array.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The entry was unpacked successfully.
    //! * LwSciError_IlwalidOperation: The FillMode is not IPC.
    //! * LwSciError_AlreadyDone: The data was already unpacked.
    //! * LwSciError_NotInitialized: Memory not pre-allocated.
    //! * LwSciError_InconsistentData: Provide aux array isn't set properly.
    //! * Any error returned by ipcBufferUnpack().
    template <typename A>
    LwSciError unpack(IpcBuffer& buf, TrackArray<A> const& aux) noexcept
    {
        // Fill mode must be IPC
        if (FillMode::IPC != fill) {
            return LwSciError_IlwalidOperation;
        }

        // If locked, data was already unpacked
        if (locked) {
            return LwSciError_AlreadyDone;
        }

        // If aux is not locked, it can't be used
        if (!aux.locked) {
            return LwSciError_InconsistentData;
        }

        // If runtime array, must have pre-allocated memory
        if (runtime && (0U == data.size())) {
            return LwSciError_NotInitialized;
        }

        // On success or failure, lock
        locked = true;

        // Use the templated IpcBuffer vector unpacking function
        return ipcBufferUnpack(buf, data, aux.data);
    }

    //! \brief Mark the data as completed so no further changes can be made.
    //!
    //! <b>Sequence of operations</b>
    //! - Sets locked flag.
    //!
    //! \return void
    void finalize(void) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        locked = true;
    };

    //! \brief Reset vector for reuse.
    //!
    //! \param [in] setLock: Flag indicating whether the array is considered
    //!   complete after the reset
    //!
    //! <b>Sequence of operations</b>
    //! - Resets all entries in the vector.
    //! - Set locked flag.
    //!
    //! \return void
    void reset(bool const setLock) noexcept
    {
        // Lwrrently only runtime arrays are reused, but someday this may
        //   change if we support dynamically reconfiguring streams. An
        //   assert should be sufficient to catch misuse for now.
        assert(runtime);

        // Reset all entries in the vector, releasing any resources.
        for (size_t i {0U}; data.size() > i; ++i) {
            data[i] = T();
        }

        // Clear locked flag.
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        locked = setLock;
    };

    //! \brief Reclaim resources when array is no longer needed.
    //!
    //! <b>Sequence of operations</b>
    //! - Calls std::vector::clear() to clear all the data.
    //! - Calls std::vector::shrink_to_fit() to free the array.
    //! - Mark as locked so array won't be reused.
    //!
    //! \return void
    void clear(void) noexcept
    {
        // Free the underlying array, releasing any resources.
        try {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            data.clear();
            data.shrink_to_fit();
            LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
        } catch (...) {
            // Do nothing. The shrink_to_fit() function is capable of
            //   throwing an exception, so we need this catch here.
            //   But we know that it won't fail, because we're only
            //   freeing the underlying array, and not trying to
            //   allocate a new one in its place. Even if the free did
            //   fail, it does no harm to the application's ability to
            //   proceed. This array will never be used again.
        }

        // Mark as locked.
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        locked = true;
    };

    //! \brief Check all members of an array against some criteria.
    //!
    //! \param [in] criteria: Function to test entry.
    //!
    //! \return bool
    //! - true if all members of the array pass the criteria.
    //! - false otherwise.
    bool test(
        std::function<bool(T const&)> const& criteria) const noexcept
    {
        for (auto const& it : data) {
            if (!criteria(it)) {
                return false;
            }
        }
        return true;
    };

private:

    //! \brief Fill mode for this object. It is initialized at construction
    //!  to the user-specified value.
    FillMode const      fill;

    //! \brief Flag indicating this array is used for runtime data, so the
    //!  vector must be pre-allocated before data is filled in. It is
    //!  initialized at construction to the user-specified value.
    bool const          runtime;

    //! \brief Flag indicating whether all array data has been specified.
    //!  It is initialized to false during construction.
    bool                locked;

    //! \brief Vector storing the tracked data.
    std::vector<T>      data;
};

} //namespace LwSciStream

#endif // TRACK_ARRAY_H
