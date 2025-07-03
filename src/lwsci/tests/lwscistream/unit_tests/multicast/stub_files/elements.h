//! \file
//! \brief LwSciStream tracking of element-phase setup.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef ELEMENTS_H
#define ELEMENTS_H

#include <utility>
#include <atomic>
#include <map>
#include "lwscistream_common.h"
#include "sciwrap.h"
#include "ipcbuffer.h"
#include "trackarray.h"

namespace LwSciStream {

//! \brief Utility object to gather up all element-related information from
//!   a block, send it through the stream all at once, and make it queriable
//!   on blocks that require it.
//!   Lwrrently this just consists of the list of element names and their
//!   buffer attributes, but additional information may be added by future
//!   optional features.
class Elements final
{
 public:
    //! \brief Caller-specified action to be performed using this object.
    using Action = std::function<void(Elements const&)>;

    //! \brief Type used for storing both (index, elemAttr) pairs in the
    //!   map and (elemType, elemAttr) pairs in the array.
    using Entry = std::pair<uint32_t, LwSciWrap::BufAttr>;

    //! \brief Constructor
    //!
    //! \param [in] paramElemFill: Value to use for elemFill constant.
    //! \param [in] paramArrayFill: Value to pass to elemArray constructor.
    Elements(
        FillMode const paramElemFill,
        FillMode const paramArrayFill) noexcept :
            elemFill(paramElemFill),
            status(DataStatus::Ready),
            elemMap(),
            elemArray(paramArrayFill, false),
            eventReady(false),
            parsed(false)
    { /* Nothing else to do */ };

    //! \brief Default destructor
    ~Elements(void) noexcept                        = default;
    // Other constructors/operators not supported
    Elements(void) noexcept                         = delete;
    Elements(const Elements&) noexcept              = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Elements(Elements&&) noexcept                   = delete;
    Elements& operator=(const Elements&) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Elements& operator=(Elements&&) & noexcept      = delete;

    //! \brief Adds an element to the list.
    //!
    //! \param [in] elemType: Unique user-defined type to identify the element.
    //! \param [in] elemBufAttr: Wrapper containing an LwSciBufAttrList.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The element was successfully added to the list.
    //! * LwSciError_NoLongerAvailable: Block's element specification phase
    //!   has completed.
    //! * LwSciError_Busy: Another thread is lwrrently accessing the data.
    //! * LwSciError_Overflow: List contains maximum number of elements.
    //! * LwSciError_AlreadyInUse: An element with the specified @a elemType
    //!   already exists in the list.
    //! * LwSciError_InsufficientMemory: Unable to grow storage for additional
    //!   element.
    //! * Any error encountered by LwSciWrap::BufAttr during duplication.
    //! Note: Errors below are for internal bug detection and should never
    //!   reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_AccessDenied: This instance is not intended for end
    //!   user element insertion.
    LwSciError mapAdd(
        uint32_t const elemType,
        LwSciWrap::BufAttr const& elemBufAttr) noexcept;

    //! \brief Merges another Elements object into this one.
    //!
    //! \param [in] inElements: The other Elements object to merge.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The merge completed succssfully.
    //! * LwSciError_NoLongerAvailable: Block's element specification phase
    //!   has completed.
    //! * LwSciError_Busy: Another thread is lwrrently accessing the data.
    //! * LwSciError_Overflow: List contains maximum number of elements.
    //! * LwSciError_InsufficientMemory: Unable to grow storage for additional
    //!   element.
    //! Note: Errors below are for internal bug detection and should never
    //!   reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_AccessDenied: This instance is not intended for merge
    //!   operations.
    LwSciError mapMerge(
        Elements const& inElements) noexcept;

    //! \brief Mark the map data as completely populated. No further
    //!   additions are allowed. If requested, transfer map data to array.
    //!
    //! \param [in] parse: If true, transfer data from map to array.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The data was successfully marked done.
    //! * LwSciError_AlreadyDone: The data was already marked complete.
    //! * LwSciError_Busy: Another thread is lwrrently accessing the data.
    //! * Any error returned by mapParse().
    LwSciError mapDone(
        bool const parse) noexcept;

    //! \brief Parse the data, transfering from map to array form.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The data was successfully transferred.
    //! * LwSciError_Busy: Another thread is lwrrently accessing the data.
    //! * Any errors returned by TrackArray::sizeInit(), TrackArray::set(),
    //!   or TrackArray::finalize().
    //! Note: Errors below are for internal bug detection and should never
    //!   reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_NotYetAvailable: The map data is not yet complete.
    //! * LwSciError_NoLongerAvailable: Block's data has been cleared.
    //! * LwSciError_AlreadyDone: The data was already parsed.
    //! * LwSciError_Overflow: Map contains too many entries.
    LwSciError dataParse(void) noexcept;

    //! \brief Copy the complete data set from another object.
    //!
    //! \param [in] orig: Object from which to copy data.
    //! \param [in] parse: If true, transfer data from map to array after
    //!   copying, if not already done.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The data was successfully copied.
    //! * LwSciError_InsufficientMemory: Unable to grow storage for additional
    //!   elements.
    //! * Any error returned by TrackArray::copy().
    //! Note: Errors below are for internal bug detection and should never
    //!   reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_AccessDenied: This instance is not intended for copy
    //!   operations.
    //! * LwSciError_AlreadyDone: The data has already been filled.
    //! * LwSciError_Busy: Another thread is lwrrently accessing the data.
    LwSciError dataCopy(
        Elements const& orig,
        bool const parse) noexcept;

    //! \brief Packs element data into an IpcBuffer.
    //!
    //! \param [in,out] buf: The buffer into which to pack the data.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The data was successfully packed.
    //! * Any error returned by IpcBuffer::packVal().
    //! * Any error returned by TrackArray::pack().
    //! * Any error returned by ipcBufferPack(std::map<uint32_t, Entry>).
    //! Note: Errors below are for internal bug detection and should never
    //!   reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_NotYetAvailable: The element data is not yet completed.
    //! * LwSciError_NoLongerAvailable: The element data has been freed.
    //! * LwSciError_Busy: Another thread is lwrrently accessing the data.
    LwSciError dataPack(
        IpcBuffer& buf) noexcept;

    //! \brief Unpacks element data from an IpcBuffer.
    //!
    //! \param [in,out] buf: The buffer from which to unpack the data.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The data was successfully unpacked.
    //! * Any error returned by IpcBuffer::unpackVal().
    //! * Any error returned by TrackArray::unpack().
    //! * Any error returned by ipcBufferUnpack(std::map<uint32_t, Entry>).
    //! Note: Errors below are for internal bug detection and should never
    //!   reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_AccessDenied: This instance is not intended for unpack
    //!   operations.
    //! * LwSciError_AlreadyDone: Data has already been unpacked into the
    //!   object.
    LwSciError dataUnpack(
        IpcBuffer& buf) noexcept;

    //! \brief Temporarily lock the data while performing some action
    //!   related to sending it to other blocks.
    //!
    //! \param [in] sendAction: Action function to actually send the data.
    //! \param [in] clear: If set, after sending, clear the data.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The send operation was successfully ilwoked.
    //! Note: Errors below are for internal bug detection and should never
    //!   reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_NotYetAvailable: The data is not yet complete.
    //! * LwSciError_NoLongerAvailable: The data has been cleared.
    //! * LwSciError_Busy: Another thread is lwrrently accessing the data.
    LwSciError dataSend(
        Action const& sendAction,
        bool const clear) noexcept;

    //! \brief Clear data when it is no longer needed to recover space.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The send operation was successful ilwoked.
    //! * LwSciError_Busy: Another thread is lwrrently accessing the data.
    //! Note: Errors below are for internal bug detection and should never
    //!   reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_NotYetAvailable: The data is not yet complete.
    LwSciError dataClear(void) noexcept;

    //! \brief Queries whether data has arrived. Note that this continues
    //!   to return true after the data has been cleared. Its used as a
    //!   phase check by the caller. Lwrrently there is no need for a check
    //!   for whether the data is actually available, because any attempts
    //!   to query the data will handle checking that.
    //!
    //! \return bool, whether or not data has arrived.
    bool dataArrived(void) const noexcept
    {
        return parsed && (DataStatus::Complete <= status);
    };

    //! \brief Light-weight query of the number of elements. Used when caller
    //!   knows (through locking or other access restrictions) that the object
    //!   is stable and no error checking or thread-safety is needed.
    //!
    //! <b>Sequence of operations</b>
    //! - If parsed, calls TrackArray::getSize() to retrieve the size of
    //!   elemeArray, otherwise calls std::map::size() to retrieve the
    //!   size of elemMap.
    //!
    //! \return size_t, the number of elements in the object.
    size_t sizePeek(void) const noexcept
    {
        return parsed ? elemArray.sizeGet() : elemMap.size();
    };

    //! \brief Light-weight query of the type for an indexed element. Used
    //!   when caller knows (through locking or other access restrictions)
    //!   that the object is stable and no thread-safety is needed.
    //!   This operation is only allowed when the data has been parsed.
    //!
    //! \param [in] index: Index of the entry to query.
    //!
    //! \return a pair with the completion code of the operation in the first
    //!   element and the queried type in the second.
    //! The completion codes are:
    //! * LwSciError_Success: The query was successful.
    //! * Any error returned by TrackArray::peek.
    //! The type will be 0 if not successful.
    auto typePeek(size_t const index) const noexcept
    {
        using rvType = std::pair<LwSciError, uint32_t>;
        auto const val { elemArray.peek(index) };
        if (LwSciError_Success != val.first) {
            return rvType(val.first, 0U);
        }
        return rvType(LwSciError_Success, val.second->first);
    };

    //! \brief Light-weight query of the attribute list (without duplication)
    //!   for an indexed element. Used when caller knows (through locking or
    //!   other access restrictions) that the object is stable and no
    //!   thread-safety is needed.
    //!   This operation is only allowed when the data has been parsed.
    //!
    //! \param [in] index: Index of the entry to query.
    //!
    //! \return a pair with the completion code of the operation in the first
    //!   element and the queried attribute list in the second.
    //! The completion codes are:
    //! * LwSciError_Success: The query was successful.
    //! * Any error returned by TrackArray::peek.
    //! * Any error encountered by LwSciWrap::BufAttr during a previous
    //!   duplication of the attribute list.
    //! The attribute list will be nullptr if not successful.
    auto attrPeek(size_t const index) const noexcept
    {
        using rvType = std::pair<LwSciError, LwSciBufAttrList>;
        auto const val { elemArray.peek(index) };
        if (LwSciError_Success != val.first) {
            return rvType(val.first, nullptr);
        }
        return rvType(val.second->second.getErr(),
                      val.second->second.viewVal());
    };

    //! \brief Query the number of elements. This is heavier weight than
    //!   sizePeek() and provides thread-safety and state checking.
    //!   This operation is only allowed when the data has been parsed.
    //!
    //! \param [in,out] size: Location in which to write element count.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The size was successfully queried.
    //! * LwSciError_NotYetAvailable: The data is not yet complete.
    //! * LwSciError_NoLongerAvailable: The data has been cleared.
    //! * LwSciError_Busy: Another thread is lwrrently accessing the data.
    LwSciError sizeGet(
        uint32_t& size) noexcept;

    //! \brief Query the type for an indexed element. This is heavier weight
    //!   than typePeek() and provides thread-safety and state checking.
    //!   This operation is only allowed when the data has been parsed.
    //!
    //! \param [in] index: Index of the entry to query.
    //! \param [in,out] elemType: Location in which to write element type.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The type was successfully queried.
    //! * LwSciError_NotYetAvailable: The data is not yet complete.
    //! * LwSciError_NoLongerAvailable: The data has been cleared.
    //! * LwSciError_Busy: Another thread is lwrrently accessing the data.
    //! * Any error returned by TrackArray::peek().
    LwSciError typeGet(
        size_t const index,
        uint32_t& elemType) noexcept;

    //! \brief Query the attribute list (with duplication) for an indexed
    //!   element. This is heavier weight than attrPeek() and provides
    //!   thread-safety and state checking.
    //!   This operation is only allowed when the data has been parsed.
    //!
    //! \param [in] index: Index of the entry to query.
    //! \param [in,out] elemAttr: Location in which to write attribute list.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The attribute list was successfully queried.
    //! * LwSciError_NotYetAvailable: The data is not yet complete.
    //! * LwSciError_NoLongerAvailable: The data has been cleared.
    //! * LwSciError_Busy: Another thread is lwrrently accessing the data.
    //! * Any error returned by TrackArray::get().
    //! * Any error encountered by LwSciWrap::BufAttr during duplication
    //!   of the attribute list.
    LwSciError attrGet(
        size_t const index,
        LwSciBufAttrList& elemAttr) noexcept;

    //! \brief Query event flag, clearing it if it was set.
    //!
    //! \return bool, value of the event flag
    bool eventGet(void) noexcept;

    //! \brief Retrieve constant reference to element array.
    //!   Caller is expected to make sure it is complete before using it.
    //!
    //! \return TrackArray<Entry> reference, the element list
    TrackArray<Entry> const& elemArrayGet(void) const noexcept
    {
        assert(parsed);
        return elemArray;
    };

private:
    //! \brief Enum used to specify current state of data.
    //!   Used with atomics to guard against thread collisions.
    enum class DataStatus : uint8_t {
        //! \brief Ready for data to be added.
        Ready,
        //! \brief A thread is lwrrently populating data.
        Filling,
        //! \brief Data is complete and cannot be updated.
        Complete,
        //! \brief Data is lwrrently locked for processing.
        Locked,
        //! \brief Data is no longer needed and has been cleared.
        Cleared
    };

    //! \brief Fill mode for this object's data (except the array,
    //!   which has its own mode).  It is initialized at construction
    //!   to the value specified by the owner.
    FillMode const              elemFill;

    //! \brief Current status of map representation of data.
    //!   It is initialized to DataStatus::Ready at creation.
    std::atomic<DataStatus>     status;

    //! \brief Map in which list is compiled from user input as a set of
    //!   (elemType, (index, elemAttr)) entries. Once fully populated and
    //!   merged with any other maps, the contents are transfered to elemArray.
    //!   The map is chosen for the initial insertion because it makes
    //!   building and merging lists easier, but the array is better for
    //!   query once the list is complete.
    //!   It is initalized to empty at creation.
    std::map<uint32_t, Entry>   elemMap;

    //! \brief Array into which (elemType, elemAttr) pairs are parsed after
    //!   the list is fully populated.
    TrackArray<Entry>           elemArray;

    //! \brief Boolean indicating pending event for arrival of data.
    //!   Not all blocks which track elements make use of this flag.
    //!   It is initialized to false at creation and set to true when
    //!   the data is complete.
    std::atomic<bool>           eventReady;

    //! \brief Flag indicating data has been transfered from map to array.
    //!   It is initialized to false at creation.
    bool                        parsed;
};

//! \brief Simple wrapper to unpack a buffer object using an Element::Entry
//!   for the auxiliary value.
inline LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciWrap::BufObj& val,
    Elements::Entry const& aux) noexcept
{
    return ipcBufferUnpack(buf, val, aux.second);
}

} // namespace LwSciStream
#endif // ELEMENTS_H
