//! \file
//! \brief LwSciStream tracking of waiter sync setup.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef SYNCWAIT_H
#define SYNCWAIT_H

#include <atomic>
#include "lwscistream_common.h"
#include "sciwrap.h"
#include "ipcbuffer.h"
#include "trackarray.h"

namespace LwSciStream {

//! \brief Utility object to gather up all information related to the
//!   setup of waiter requirements for synchronization.
class Waiters final
{
public:
    //! \brief Type used for storing (used, syncAttr) pair indicating
    //!   whether an element is used and if so any attribute list with
    //!   the requirements to wait for operations on the element.
    using Entry = std::pair<bool, LwSciWrap::SyncAttr>;

    //! \brief Constructor
    //!
    //! \param [in] paramFill: Fill mode to use for setting up contents.
    //! \param [in] paramUsage: Indicates usage can be modified.
    Waiters(
        FillMode const paramFill,
        bool const paramUsage) noexcept :
            fill(paramFill),
            usage(paramUsage),
            attrs(paramFill, false),
            event(false),
            completed(false)
    { /* Nothing else to do */ };

    //! \brief Initialize size of array.
    //!
    //! \param [in] size: The number of elements.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray<Entry>::sizeInit().
    LwSciError sizeInit(
        size_t const size) noexcept;

    //! \brief Retrieve indexed usage flag, without an error code.
    //!   On failure, false is returned. Used internally when Waiter
    //!   object's data will persist during the period when the info
    //!   will be used, and the state is such that either errors
    //!   can't happen, or the specific error code is not needed.
    //!
    //! \param [in] index: Index of the entry to query.
    //!   Valid values: [0, size-1]
    //!
    //! \return bool, the indexed usage flag, or false on failure.
    bool usedPeek(
        size_t const index) const noexcept;

    //! \brief Retrieve original indexed attibrute list, without an error
    //!   code. On failure, NULL is returned. Used internally when Waiter
    //!   object's data will persist during the period when the attribute
    //!   list will be used, and the state is such that either errors
    //!   can't happen, or the specific error code is not needed.
    //!
    //! \param [in] index: Index of the entry to query.
    //!   Valid values: [0, size-1]
    //!
    //! \return LwSciSyncAttrList, the indexed attribute list, or NULL
    //!    on failure.
    LwSciSyncAttrList attrPeek(
        size_t const index) const noexcept;

    //! \brief Retrieve indexed usage flag, with full error code.
    //!
    //! \param [in] index: Index of the entry to query.
    //!   Valid values: [0, size-1]
    //! \param [in,out] used: Reference to location to store value.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray<Entry>::peek().
    LwSciError usedGet(
        size_t const index,
        bool& val) const noexcept;

    //! \brief Retrieve copy of indexed attribute list, with full error code.
    //!
    //! \param [in] index: Index of the entry to query.
    //!   Valid values: [0, size-1]
    //! \param [in,out] val: Reference to location to store value.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray<Entry>::get().
    LwSciError attrGet(
        size_t const index,
        LwSciWrap::SyncAttr& val) const noexcept;

    //! \brief Set the usage flag for an indexed entry.
    //!
    //! \param [in] index: Index of the entry to set.
    //!   Valid values: [0, size-1]
    //! \param [in] used: Flag indicating entry is used.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray<Entry>::peek().
    //! * Any error returned by TrackArray<Entry>::set().
    //! * Any error encountered by LwSciWrap::BufAttr during duplication.
    //! Note: Errors below are for internal bug detection and should never
    //! reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_AccessDenied: Operation incompatible with fill mode.
    //! * LwSciError_IlwalidOperation: The object does not support changing
    //!   the usage flag.
    // TODO: When we have a formal Usage object, we could eliminate this
    //       function and just pass the Usage as an optional parameter
    //       to setDone, which would copy all the values at once.
    LwSciError usedSet(
        size_t const index,
        bool const used) noexcept;

    //! \brief Set the attribute list for an indexed entry.
    //!
    //! \param [in] index: Index of the entry to set.
    //!   Valid values: [0, size-1]
    //! \param [in] attr: Wrapped attribute list for the entry.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray<Entry>::peek().
    //! * Any error returned by TrackArray<Entry>::set().
    //! * Any error encountered by LwSciWrap::BufAttr during duplication.
    //! Note: Errors below are for internal bug detection and should never
    //! reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_AccessDenied: Operation incompatible with fill mode.
    LwSciError attrSet(
        size_t const index,
        LwSciWrap::SyncAttr const& attr) noexcept;

    //! \brief Finalize the waiter attributes.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_AlreadyDone: The object setup was already completed.
    //! Note: Errors below are for internal bug detection and should never
    //! reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_AccessDenied: Operation incompatible with fill mode.
    LwSciError doneSet(void) noexcept;

    //! \brief Fill the attribute list with copies of the same value, marking
    //!   all as used.
    //!
    //! \param [in] attr: Wrapped attribute list for the entries.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray<Entry>::set().
    //! * Any error returned by TrackArray<Entry>::peek().
    //! * Any error encountered by LwSciWrap::SyncAttr during duplication.
    //! Note: Errors below are for internal bug detection and should never
    //! reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_AccessDenied: Operation incompatible with fill mode.
    LwSciError entryFill(
        LwSciWrap::SyncAttr const& attr) noexcept;

    //! \brief Copy the waiter info from another object.
    //!
    //! \param [in] orig: Object from which to copy information.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray<Entry>::copy().
    //! Note: Errors below are for internal bug detection and should never
    //! reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_AccessDenied: Operation incompatible with fill mode.
    //! * LwSciError_InconsistentData: The usage flag doesn't match.
    LwSciError copy(
        Waiters const& orig) noexcept;

    //! \brief Merge in the waiter info from another object.
    //!
    //! \param [in] orig: Object from which to merge information.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray<Entry>::get().
    //! * Any error returned by TrackArray<Entry>::set().
    //! Note: Errors below are for internal bug detection and should never
    //! reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_AccessDenied: Operation incompatible with fill mode.
    //! * LwSciError_InconsistentData: The usage flag or size doesn't match.
    LwSciError merge(
        Waiters const& orig) noexcept;

    //! \brief Packs waiter info into an IpcBuffer.
    //!
    //! \param [in,out] buf: The buffer into which to pack the data.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray<Entry>::pack().
    //! * Any error returned by ipcBufferPack(LwSciWrap::SyncAttr).
    LwSciError pack(
        IpcBuffer& buf) const noexcept;

    //! \brief Unpacks waiter info from an IpcBuffer.
    //!
    //! \param [in,out] buf: The buffer from which to unpack the data.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The data was successfully unpacked.
    //! * Any error returned by TrackArray<Entry>::unpack().
    LwSciError unpack(
        IpcBuffer& buf) noexcept;

    //! \brief Free no longer needed sync resources
    //!
    //! \return void
    void clear(void) noexcept;

    //! \brief Queries and clears the flag indicating an event is pending
    //!   after the data has arrived.
    //!
    //! \return bool, whether or not an event is pending.
    bool pendingEvent(void) noexcept
    {
        bool expected { true };
        return event.compare_exchange_strong(expected, false);
    };

    //! \brief Queries whether the data is complete.
    //!
    //! \return bool, whether or not data is complete.
    bool isComplete(void) const noexcept
    {
        return completed;
    };

    //! \brief Retrieve constant reference to attribute array.
    //!   Caller is expected to make sure it is complete before using it.
    //!
    //! \return TrackArray<Entry> reference, the attribute list
    TrackArray<Entry> const& attrsArrayGet(void) const noexcept
    {
        return attrs;
    };

private:
    //! \brief Fill mode for the object.
    //!   Initalized at construction to user provided value.
    FillMode const          fill;

    //! \brief Flag indicating usage flags can be modified. Producers
    //!   are expected to use all elements, but consumers are not.
    //!   Initialized at construction to user provided value.
    bool const              usage;

    //! \brief Array of usage/attribute information.
    //!   Its size is initialized with the number of elements when
    //!   that information is provided by the pool, and then it is
    //!   populated based on the fill mode.
    TrackArray<Entry>       attrs;

    //! \brief Flag indicating pending WaiterAttr event.
    //!   It is initialized to false at creation.
    std::atomic<bool>       event;

    //! \brief Flag indicating data is complete.
    //!   It is initialized to false at creation.
    bool                    completed;
};

//! \brief Simple wrapper to unpack a sync object using a Waiters::Entry
//!   for the auxiliary value.
inline LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciWrap::SyncObj& val,
    Waiters::Entry const& aux) noexcept
{
    return ipcBufferUnpack(buf, val, aux.second, !aux.first);
}

} // namespace LwSciStream

#endif // SYNCWAIT_H
