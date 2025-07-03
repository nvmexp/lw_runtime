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

#ifndef SYNCSIGNAL_H
#define SYNCSIGNAL_H

#include <atomic>
#include "lwscistream_common.h"
#include "sciwrap.h"
#include "ipcbuffer.h"
#include "trackarray.h"
#include "syncwait.h"

namespace LwSciStream {

//! \brief Utility object to gather up all information related to the
//!   setup of signalling sync objects.
class Signals final
{
public:
    //! \brief Constructor
    //!
    //! \param [in] paramFill: Fill mode to use for setting up contents.
    //! \param [in] paramWaiter: Waiters with which signaller must
    //!   be compatible.
    Signals(
        FillMode const paramFill,
        Waiters const* const paramWaiter) noexcept :
            fill(paramFill),
            waiter(paramWaiter),
            endpointCount(0U),
            elementCount(0U),
            syncs(paramFill, false),
            event(false),
            completed(false)
    { /* Nothing else to do */ };

    //! \brief Initialize size of array.
    //!
    //! \param [in] paramEndpointCount: The number of endpoints.
    //! \param [in] paramElementCount: The number of elements.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::sizeInit().
    LwSciError sizeInit(
        size_t const paramEndpointCount,
        size_t const paramElementCount) noexcept;

    //! \brief Retrieve original indexed sync object, without an error
    //!   code. On failure, NULL is returned. Used internally when Signal
    //!   object's data will persist during the period when the sync
    //!   object will be used, and the state is such that either errors
    //!   can't happen, or the specific error code is not needed.
    //!
    //! \param [in] endIndex: Index of the endpoint to query.
    //!   Valid values: [0, endpointCount-1]
    //! \param [in] elemIndex: Index of the element to query.
    //!   Valid values: [0, elementCount-1]
    //! \param [in,out] val: Reference to location to store value.
    //!
    //! \return LwSciSyncObj, the indexed sync object, or NULL on failure.
    LwSciSyncObj syncPeek(
        size_t const endIndex,
        size_t const elemIndex) const noexcept;

    // Temporary function until new fence APIs are in place
    LwSciSyncObj syncPeek(
        size_t const index) const noexcept;

    //! \brief Retrieve indexed sync object.
    //!
    //! \param [in] endIndex: Index of the endpoint to query.
    //!   Valid values: [0, endpointCount-1]
    //! \param [in] elemIndex: Index of the element to query.
    //!   Valid values: [0, elementCount-1]
    //! \param [in,out] val: Reference to location to store value.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_IndexOutOfRange: @a endIndex or @a elemIndex is
    //!   out of range.
    //! * Any error returned by TrackArray::get().
    LwSciError syncGet(
        size_t const endIndex,
        size_t const elemIndex,
        LwSciWrap::SyncObj& val) const noexcept;

    //! \brief Set indexed sync object.
    //!
    //! \param [in] endIndex: Index of the endpoint to set.
    //!   Valid values: [0, endpointCount-1]
    //! \param [in] elemIndex: Index of the element to set.
    //!   Valid values: [0, elementCount-1]
    //! \param [in] sync: Wrapped sync object for the entry.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_IndexOutOfRange: @a endIndex or @a elemIndex is
    //!   out of range.
    //! * Any error returned by TrackArray::set().
    //! * Any error returned by TrackArray::peek().
    //! * Any error encountered during duplication of sync objects.
    LwSciError syncSet(
        size_t const endIndex,
        size_t const elemIndex,
        LwSciWrap::SyncObj const& sync) noexcept;

    //! \brief Finalize the signaller objects.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_AlreadyDone: The object was already marked completed.
    //! Note: Errors below are for internal bug detection and should never
    //! reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_AccessDenied: Operation incompatible with fill mode.
    LwSciError doneSet(void) noexcept;


    //! \brief Fill the object list with copies of the same value.
    //!
    //! \param [in] sync: Wrapped sync object for the entries.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray:set().
    //! * Any error returned by TrackArray::peek().
    //! * Any error encountered by LwSciWrap::SyncObj during duplication.
    //! Note: Errors below are for internal bug detection and should never
    //! reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_AccessDenied: Operation incompatible with fill mode.
    LwSciError syncFill(
        LwSciWrap::SyncObj const& sync) noexcept;

    //! \brief Copy the signaller info from another object.
    //!
    //! \param [in] orig: Object from which to copy information.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::peek().
    //! * Any error returned by TrackArray::copy().
    LwSciError copy(
        Signals const& orig) noexcept;

    //! \brief Collate the signaller info from another object into a
    //!   subrange of this object.
    //!
    //! \param [in] orig: Object from which to collate information.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::collate().
    LwSciError collate(
        Signals const& orig,
        size_t const endRangeStart,
        size_t const endRangeCount) noexcept;

    //! \brief Packs signaller info into an IpcBuffer.
    //!
    //! \param [in,out] buf: The buffer into which to pack the data.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::pack().
    LwSciError pack(
        IpcBuffer& buf) const noexcept;

    //! \brief Unpacks signaller info from an IpcBuffer.
    //!
    //! \param [in,out] buf: The buffer from which to unpack the data.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The data was successfully unpacked.
    //! * LwSciError_InsufficientData: No Waiters object was provided.
    //! * Any error returned by TrackArray::unpack().
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

    //! \brief Retrieve constant reference to sync object array.
    //!   Caller is expected to make sure it is complete before using it.
    //!
    //! \return TrackArray<LwSciWrap::SyncObj> reference, the sync object list
    TrackArray<LwSciWrap::SyncObj> const& syncsArrayGet(void) const noexcept
    {
        return syncs;
    };

private:
    //! \brief Validates incoming sync object against waiter attributes.
    //!
    //! \param [in] syncWrap: Wrapped sync object.
    //! \param [in] elemIndex: Index within element list.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The object and attributes are compatible.
    //! * LwSciError_InconsistentData: The object and attributes are not
    //!   compatible.
    //! * Any error or panic returned by LwSciSyncObjGetAttrList() or
    //!   LwSciSyncAttrListValidateReconciled().
    LwSciError validate(
        LwSciWrap::SyncObj const& syncWrap,
        size_t const elemIndex) noexcept;

private:
    //! \brief Fill mode for the object.
    //!   Initalized at construction to user provided value.
    FillMode const                  fill;

    //! \brief Pointer to the waiter info, if any, with which the signalling
    //!   must be compatible.
    //!   Initialized at construction to user provided value.
    Waiters const* const            waiter;

    //! \brief Number of endpoints referenced by this object.
    //!   For producer sync objects, it is lwrrently always 1.
    //!   For consumer sync objects, it is the number of consumers
    //!     in the owning block's subtree.
    //!   Initialized when element count is known.
    size_t                          endpointCount;

    //! \brief Number of elements.
    //!   Initialized when element count is known.
    size_t                          elementCount;

    //! \brief Array of sync objects. It is treated as a 2D array
    //!   of endpointCount sets of elementCount objects.
    //!   Its size is initialized when the counts are set.
    TrackArray<LwSciWrap::SyncObj>  syncs;

    //! \brief Flag indicating pending SignalObj event.
    //!   It is initialized to false at creation.
    std::atomic<bool>               event;

    //! \brief Flag indicating data is complete.
    //!   It is initialized to false at creation.
    bool                            completed;

    // TODO: Some extra fields may be required for C2C to deal with
    //       the fact that everything gets squeezed down to a single
    //       sync object by the cross-chip copy step.

};

} // namespace LwSciStream

#endif // SYNCWAIT_H
