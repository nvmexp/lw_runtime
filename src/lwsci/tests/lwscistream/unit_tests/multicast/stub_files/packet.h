//! \file
//! \brief LwSciStream packet class declaration.
//!
//! \copyright
//! Copyright (c) 2019-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef PACKET_H
#define PACKET_H
#include <cstdint>
#include <cassert>
#include <atomic>
#include <array>
#include <vector>
#include <utility>
#include <memory>
#include <unordered_map>
#include "covanalysis.h"
#include "lwscistream_common.h"
#include "branch.h"
#include "endinfo.h"
#include "trackarray.h"
#include "elements.h"
#include "syncsignal.h"
#include "lwscibuf_c2c_internal.h"

namespace LwSciStream {

// Forward declaration
class Packet;

//! \brief Alias for smart pointer of Packet class.
//!
//! \if TIER4_SWAD
//! \implements{20036997}
//! \endif
using PacketPtr  = std::shared_ptr<Packet>;

//! \brief Alias for the unordered map of LwSciStreamPacket and PacketPtr as
//!   key-value pairs.
//!
//! \if TIER4_SWAD
//! \implements{20037000}
//! \endif
using PacketMap  = std::unordered_map<LwSciStreamPacket, PacketPtr>;

//! \brief Alias for PacketMap iterator.
//!
//! \if TIER4_SWAD
//! \implements{20037003}
//! \endif
using PacketIter = PacketMap::iterator;

//! \brief Stores a block's information for the packet.
//!   Each type of Block needs to track its own specific set of per-packet
//!   information. However, they all draw different combinations of elements
//!   from a common global set of information types. Therefore, rather than
//!   defining independent packet types for each Block, we define a common
//!   configurable packet type, for which each Block can specify its needs,
//!   using only the relevant functions.
//!
//!   The sets of information tracked are broken down into sub-objects. Any
//!   given block makes use of only some of them. To minimize wasted memory,
//!   the packet contains a vector of each of these. We dynamically allocate
//!   the contents based on a given block's requirements. If a block doesn't
//!   need a particular one, the vector is left empty.
//!
//! \if TIER4_SWAD
//! \implements{20036994}
//! \endif
//!
//! \if TIER4_SWUD
//! \implements{20262789}
//! \endif
//
//   The sub-objects are:
//   * StatusEvent  - Used to queue up packet acceptance-status events.
class Packet final
{
public:
    //! \brief Enum identifying location of a packet instance with regards to
    //!   owning block.
    //!
    //! \if TIER4_SWAD
    //! \implements{20037006}
    //! \endif
    enum class Location : std::uint8_t
    {
        //! \brief Not yet specified.
        Unknown,
        //! \brief Packet is queued in the owning block ready for use.
        Queued,
        //! \brief Packet is held by the application where the owning block
        //!   resides.
        Application,
        //! \brief Packet is somewhere upstream of the owning block.
        Upstream,
        //! \brief Packet is somewhere downstream of the owning block.
        Downstream
    };

    //! \brief Description used to set up packet(s) when they are created for
    //!   a block instance.
    //!
    //! \if TIER4_SWAD
    //! \implements{20037009}
    //! \endif
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M11_0_1), "Approved TID-577")
    class Desc final
    {
        // NOTE: In order to qualify for deviation from M11-0-1, no
        //       other member functions may be added to this class.
        //       Access must be in PoD fashion, with no relationship
        //       between members enforced (by this object).
        // TODO: We have now eliminated all non-POD members of this
        //       object. If it remains that way after all the API
        //       changes are done, we can simplify this to a struct
        //       and won't have to worry about M11-0-1 or defining
        //       default functions.

    public:
        Desc(void) noexcept                   = default;
        Desc(Desc const&) noexcept            = default;
        Desc(Desc&&) noexcept                 = default;
        ~Desc(void) noexcept                  = default;
        Desc& operator=(Desc const&) & noexcept = default;
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_5), "Bug 2782263")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A18_9_2), "Bug 2782263")
        Desc& operator=(Desc&&) & noexcept      = default;
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A18_9_2))
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_5))

        //! \brief Initial Location of a packet instance after creation.
        //!   Initialized to Location::Unknown.
        Location initialLocation    { Location::Unknown };

        //! \brief Fill mode used to populate definitions.
        //!   Initialized to FillMode::None.
        FillMode defineFillMode     { FillMode::None };

        //! \brief Fill mode used to populate producer status.
        //!   Initialized to FillMode::None.
        FillMode statusProdFillMode { FillMode::None };

        //! \brief Fill mode used to populate consumer status.
        //!   Initialized to FillMode::None.
        FillMode statusConsFillMode { FillMode::None };

        //! \brief Fill mode used to populate producer fences.
        //!   Initialized to FillMode::None.
        FillMode fenceProdFillMode { FillMode::None };

        //! \brief Fill mode used to populate consumer fences.
        //!   Initialized to FillMode::None.
        FillMode fenceConsFillMode { FillMode::None };

        //! \brief Number of consumers in the parent block's subtree.
        //!   Valid values: [0,MAX_INT_SIZE]. Initialized to 0 and
        //!   overwritten during connection process.
        size_t   numConsumer        { 0U };

        //! \brief Number of branches in owning block if it is a
        //!   multicast block. Initialized to 0.
        size_t   branchCount        { 0U };

        //! \brief Number of elements in the packet layout.
        //!   Valid values: [0,MAX_INT_SIZE]. Initialized to 0 and
        //!   overwritten when the allocated element info arrives.
        size_t   elementCount       { 0U };

        //! \brief Flag indicating whether packets for the parent block
        //!  require buffer information. Initialized to false.
        bool     needBuffers        { false };

        //! \brief Flag to indicate if the parent block needs to store the
        //!   C2C buffer handles in the packet layout. Will only be set
        //!   by C2CSrc block.
        //!   Initialized to false.
        bool     useC2CBuffer       { false };
    };
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M11_0_1))

    //! \brief Type for packet member function which checks, and potentially
    //!   clears, some pending event criteria.
    using Pending = bool (Packet::*)(void);

    //! \brief Ilwokes a provided member function of type Pending.
    // TODO: Would not be needed with std::ilwoke (C++17 feature)
    bool ilwokePending(Pending func) noexcept
    {
        return (this->*func)();
    };

public:
    //
    // Core packet operations
    //

    //! \brief Constructs an instance of the Packet class, sets
    //!   LwSciStreamPacket and LwSciStreamCookie to the input value and
    //!   initializes all data fields for the created instance with the input
    //!   Packet::Desc instance.
    //!
    //! \param [in] pktDesc: Packet::Desc used by the block that owns the
    //!   packet instance.
    //! \param [in] paramHandle: LwSciStreamPacket.
    //! \param [in] paramCookie: LwSciStreamCookie. It is specified when
    //!   creating the packet instance at the pool. It may be provided later
    //!   in other blocks.
    //!
    //! \if TIER4_SWAD
    //! \implements{20208450}
    //! \endif
    Packet(
        Desc const&             paramPktDesc,
        LwSciStreamPacket const paramHandle,
        LwSciStreamCookie const paramCookie) noexcept;

    //! \brief Packet destructor
    ~Packet(void) noexcept;

    // Other default constructors and copy operations not supported
    Packet(void) noexcept                      = delete;
    Packet(const Packet&) noexcept             = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Packet(Packet&&) noexcept                  = delete;
    Packet& operator=(const Packet&) & noexcept  = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Packet& operator=(Packet&&) & noexcept       = delete;

    //! \brief Queries any error encountered during initialization.
    //!
    //! \return LwSciError
    LwSciError initErrorGet(void) const noexcept
    {
        return initError;
    };

    //! \brief Retrieves LwSciStreamPacket of the packet instance.
    //!
    //! \return LwSciStreamPacket
    //!
    //! \if TIER4_SWAD
    //! \implements{20208453}
    //! \endif
    LwSciStreamPacket handleGet(void) const noexcept
    {
        return handle;
    }

    //! \brief Retrieves LwSciStreamCookie associated with the packet instance.
    //!
    //! \return LwSciStreamCookie
    //!
    //! \if TIER4_SWAD
    //! \implements{20208456}
    //! \endif
    LwSciStreamCookie cookieGet(void) const noexcept
    {
        return cookie;
    }

    //! \brief Checks, without modification, whether any setup
    //!   (definition or status) actions are still pending.
    //!
    //! return bool, whether or not more setup is needed.
    bool setupPending(void) noexcept
    {
        return (((!defineCompleted) ||
                 (defineEvent.load() || defineHandleEvent.load())) ||
                ((!statusArrived() || (0U != statusEvent.load()))));
    }

    //! \brief Validates current packet Location.
    //!
    //! \param [in] expectLocation: Expected Location of the packet.
    //!
    //! \return boolean
    //! * true: If current Location of the packet instance matches
    //!   expected one.
    //! * false: Otherwise.
    //!
    //! \if TIER4_SWAD
    //! \implements{20208459}
    //! \endif
    bool locationCheck(Location const expectLocation) const noexcept;

    //! \brief Atomically validates current packet Location and replaces it
    //!   with the new one.
    //!
    //! \param [in] oldLocation: Previous Location of the packet instance.
    //! \param [in] newLocation: New Location of the packet instance.
    //!
    //! \return boolean
    //! * true: If the Location is successfully updated.
    //! * false: Otherwise.
    //!
    //! \if TIER4_SWAD
    //! \implements{20208462}
    //! \endif
    bool locationUpdate(Location const oldLocation,
                        Location const newLocation) noexcept;

private:

    //
    // Core packet state
    //

    //! \brief Error encountered during construction.
    //!   Initalized to LwSciError_Success and then overwritten by the
    //!   first failure, if any, during the constructor.
    LwSciError                      initError;

    //! \brief Reference to the parent Block's packet descriptor.
    //!   Initialized when the packet is created.
    Desc const&                     pktDesc;

    //! \brief Handle for the packet.
    //!   Initialized when a packet instance is created.
    // Note that our implementation lwrrently uses the same handle for each
    //   packet across all blocks within a process, but that is not required
    //   by the LwSciStream spec.
    LwSciStreamPacket               handle;

    //! \brief Application's LwSciStreamCookie for the packet for the owning
    //!   block. It's either initialized to a valid LwSciStreamCookie value
    //!   when the packet is created, or initialized to
    //!   LwSciStreamCookie_Ilwalid and may be changed to a valid
    //!   LwSciStreamCookie value if the packet instance is accepted.
    LwSciStreamCookie               cookie;

    //! \brief Current Location of the packet instance relative to the owning
    //!   block. It's initialized with the Packet::Desc::initialLocation,
    //!   which is provided at the creation of a packet instance.
    std::atomic<Location>           lwrrLocation;

public:

    //
    // Packet definition operations
    //

    //! \brief Retrieves a copy of the indexed buffer.
    //!
    //! \param [in] elemIndex: The index of the buffer to query.
    //! \param [in,out] elemBufObj: Location in which to store the buffer.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray<LwSciWrap::BufObj>::get().
    LwSciError bufferGet(
        size_t const elemIndex,
        LwSciWrap::BufObj& elemBufObj) const noexcept;

    //! \brief Sets the indexed entry in the buffer list.
    //!
    //! \param [in] elemIndex: The index of the buffer to set.
    //! \param [in,out] elemBufObj: Buffer to store in the list.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray<LwSciWrap::BufObj>::set().
    //! * Any error encountered by LwSciWrap::BufObj during duplication.
    //! Note: Errors below are for internal bug detection and should never
    //! reach public APIs if LwSciStream code uses this object correctly.
    //! * Any error returned by TrackArray<LwSciWrap::BufObj>::peek().
    LwSciError bufferSet(
        size_t const elemIndex,
        LwSciWrap::BufObj const& elemBufObj) noexcept;

    //! \brief Make sure the data is fully defined and then lock it down.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_AlreadyDone: The definition has already been completed.
    //! * LwSciError_InsufficientData: Not all buffers are set.
    //! Note: Errors below are for internal bug detection and should never
    //! reach public APIs if LwSciStream code uses this object correctly.
    //! * Any error returned by TrackArray<LwSciWrap::BufObj>::peek().
    LwSciError defineDone(void) noexcept;

    //! \brief Copies the packet definitions from another Packet instance.
    //!
    //! \param [in] origPacket: The source Packet to copy.
    //! \param [in] setEvent: If set, this packet will be flagged to trigger
    //!   a LwSciStreamEventType_PacketCreate event.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_AlreadyDone: The definition has already been filled in.
    //! * Any error returned by TrackArray<LwSciWrap::BufObj>::copy().
    LwSciError defineCopy(
        Packet const& origPacket,
        bool const    setEvent) noexcept;

    //! \brief Packs the packet's definition to an IpcBuffer.
    //!
    //! \param [in,out] buf: The buffer to which to pack the data.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_InsufficientData: The definition is not complete.
    //! * Any error returned by TrackArray<LwSciWrap::BufObj>::pack().
    LwSciError definePack(
        IpcBuffer& buf) const noexcept;

    //! \brief Unpacks the packet's definition from an IpcBuffer.
    //!
    //! \param [in,out] buf: The buffer from which to unpack the data.
    //! \param [in] aux: Elements object containing the buffer attributes.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_AlreadyDone: The definition has already been completed.
    //! * Any error returned by TrackArray<LwSciWrap::BufObj>::unpack().
    // TODO: May want to have the elements referenced by the Packet::Desc,
    //       where it can also be used for validation of other operations.
    LwSciError defineUnpack(
        IpcBuffer& buf,
        Elements const& aux) noexcept;

    //! \brief Releases packet definition resources that are no longer needed.
    //!
    //! \return void
    void defineClear(void) noexcept;

    //! \brief Queries and clears the flag indicating packet define event
    //!   is pending, and if so sets the flag for pending packet handle.
    bool definePending(void) noexcept
    {
        bool expected { true };
        if (defineEvent.compare_exchange_strong(expected, false)) {
            defineHandleEvent.store(true);
            return true;
        }
        return false;
    };

    //! \brief Queries and clears the flag indicating this packet's handle
    //!   should be returned after a packet creation event.
    bool defineHandlePending(void) noexcept
    {
        bool expected { true };
        return defineHandleEvent.compare_exchange_strong(expected, false);
    };

private:

    //
    // Packet definition state
    //

    //! \brief Array of buffer objects defining the packet.
    //!   Allocated to an empty array of the appropriate size during
    //!   construction, and then populated based on its fill mode.
    //!   Only blocks which need the buffer objects will use this
    //!   array, and they will free the contents once no longer needed.
    TrackArray<LwSciWrap::BufObj>   buffers;

    //! \brief Flag indicating definition has completed.
    //!   Initialized to false at creation.
    std::atomic<bool>               defineCompleted;

    //! \brief Flag indicating packet creation event is pending.
    //!   Initialized to false at creation.
    std::atomic<bool>               defineEvent;

    //! \brief Flag indicating packet creation event has been retrieved and
    //!   now retrieval of the handle is pending.
    //!   Initialized to false at creation.
    std::atomic<bool>               defineHandleEvent;


public:

    //
    // C2C Buffer operations
    //

    //! \brief Registers LwSciBufObj with C2C service and saves the returned
    //!   source handle.
    //!
    //! \param [in] channelHandle: Handle to the C2C channel.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NotYetAvailable: The LwSciBufObj(s) not arrived yet.
    //! * Any error returned by TrackArray::peek() and Wrapper::viewVal().
    //! * Any error returned by LwSciBufRegisterSourceObjIndirectChannelC2c().
    LwSciError registerC2CBufSourceHandles(
        LwSciC2cHandle const channelHandle) noexcept;

    //! \brief Retrieves the source handle of the indexed C2C buffer.
    //!
    //! \param [in] bufIndex: The index of the buffer to query.
    //!
    //! \return LwSciC2cBufSourceHandle: source handle of C2C buffer.
    LwSciC2cBufSourceHandle c2cBufSourceHandleGet(
        size_t const bufIndex) const noexcept
    {
        assert(c2cBufHandles.size() > bufIndex);
        return c2cBufHandles[bufIndex];
    };

private:
    // List of C2C buffer source handles
    std::vector<LwSciC2cBufSourceHandle> c2cBufHandles;


public:

    //
    // Packet status operations
    //

    //! \brief Retrieves the producer's status value.
    //!
    //! \param [in,out] outStatus: Location to store the value.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NotYetAvailable: The status has not yet been set.
    LwSciError statusProdGet(
        LwSciError& outStatus) const noexcept;

    //! \brief Retrieves the status value from a specific consumer.
    //!
    //! \param [in] consIndex: The index of the consumer to query.
    //! \param [in,out] outStatus: Location to store the value.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NotYetAvailable: The status has not yet been set.
    //! * Any error returned TrackArray<LwSciError>::get().
    LwSciError statusConsGet(
        size_t const consIndex,
        LwSciError& outStatus) const noexcept;

    //! \brief Sets the producer status value and cookie.
    //!   Only used by the producer block.
    //!
    //! \param [in] paramStatus: The status value.
    //! \param [in] paramCookie: The cookie to use for the packet.
    //!   Ignored if @a paramStatus is not Success.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_BadParameter: Illegal value for @a paramStatus.
    //! * LwSciError_StreamBadCookie: Value for @a paramStatus is Success
    //!   and @a paramCookie is is invalid.
    //! * LwSciError_AlreadyDone: Status was already set.
    //! Note: Errors below are for internal bug detection and should never
    //! reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_IlwalidOperation: Packet instance does not support
    //!   this function.
    LwSciError statusProdSet(
        LwSciError const paramStatus,
        LwSciStreamCookie const paramCookie) noexcept;

    //! \brief Sets the consumer status value and cookie.
    //!   Only used by the consumer block.
    //!
    //! \param [in] paramStatus: The status value.
    //! \param [in] paramCookie: The cookie to use for the packet.
    //!   Ignored if @a paramStatus is not Success.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_BadParameter: Illegal value for @a paramStatus.
    //! * LwSciError_StreamBadCookie: Value for @a paramStatus is Success
    //!   and @a paramCookie is is invalid.
    //! * LwSciError_AlreadyDone: Status was already set.
    //! * Any error returned by TrackArray<LwSciError>::set().
    LwSciError statusConsSet(
        LwSciError const paramStatus,
        LwSciStreamCookie const paramCookie) noexcept;

    //! \brief Copy producer status from another Packet instance.
    //!
    //! \param [in] origPacket: The packet from which to copy the status.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_AlreadyDone: The Packet's producer status has already
    //!   been filled in.
    //! Note: Errors below are for internal bug detection and should never
    //! reach public APIs if LwSciStream code uses this object correctly.
    //! * LwSciError_IlwalidOperation: Packet instance does not support
    //!   this function.
    LwSciError statusProdCopy(
        Packet const& origPacket) noexcept;

    //! \brief Copy consumer status from another Packet instance.
    //!
    //! \param [in] origPacket: The packet from which to copy the status.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_AlreadyDone: The Packet's consumer status has already
    //!   been filled in.
    //! * Any error returned by TrackArray<LwSciError>::copy().
    LwSciError statusConsCopy(
        Packet const& origPacket) noexcept;

    // Note: There are no collate, pack, or unpack functions for the producer
    //       status because we don't lwrrently need them. Only the consumer
    //       status list gets combined with others and flows over IPC. IPC
    //       functions might become necessary with C2C.

    //! \brief Copy a range of consumer statuses from another Packet instance.
    //!
    //! \param [in] origPacket: The packet from which to copy the status.
    //! \param [in] branchIndex: Index of branch being collated in.
    //! \param [in] rangeStart: Index of first entry in range.
    //! \param [in] rangeCount: Number of entries in range.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_AlreadyDone: The Packet's consumer status has already
    //!   been completed filled in.
    //! * Any error returned by BranchTrack::set().
    //! * Any error returned by TrackArray::collate().
   LwSciError statusConsCollate(
        Packet const& origPacket,
        size_t const branchIndex,
        size_t const rangeStart,
        size_t const rangeCount) noexcept;

    //! \brief Check if collated consumer status data is complete and take
    //!   responsibility for sending it upstream.
    //!
    //! \return bool, Whether or not all status is ready and caller should
    //!   pass it on.
    bool statusConsCollateDone(void) noexcept;

    //! \brief Packs the packet's consumer status to an IpcBuffer.
    //!
    //! \param [in,out] buf: The buffer to which to pack the data.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_InsufficientData: The status is not complete.
    //! * Any error returned by TrackArray<LwSciError>::pack().
    LwSciError statusConsPack(
        IpcBuffer& buf) const noexcept;

    //! \brief Unpacks the packet's consumer status from an IpcBuffer.
    //!
    //! \param [in,out] buf: The buffer from which to unpack the data.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_AlreadyDone: The status has already been completed.
    //! * Any error returned by TrackArray<LwSciError>::unpack().
    LwSciError statusConsUnpack(
        IpcBuffer& buf) noexcept;

    //! \brief Queries whether there is a pending status event.
    //!
    //! return bool, whether or not an event is pending.
    bool statusPending(void) noexcept;

    //! \brief Releases packet status resources that are no longer needed.
    //!
    //! \return void
    void statusClear(void) noexcept;

    //! \brief Checks whether all status tracked by the parent block
    //!   has arrived.
    //!
    //! \return bool, whether or not status is available for query.
    bool statusArrived(void) const noexcept
    {
        return ((FillMode::None == pktDesc.statusProdFillMode) ||
                statusProdCompleted) &&
               ((FillMode::None == pktDesc.statusConsFillMode) ||
                statusConsCompleted);
    };

    //! \brief Checks whether packet has been accepted by the endpoints
    //!   tracked by the parent block. Note that this is only reliable
    //!   after statusArrived() reports true.
    //!
    //! \return bool, whether or not packet was accepted.
    bool statusAccepted(void) const noexcept
    {
        return statusArrived() && !rejected;
    };

private:

    //
    // Packet status state
    //

    //! \brief Vector of consumer status values.
    //!   Allocated during construction to an array sized for the number
    //!   of consumers in the parent block's subtree with initial values
    //!   of LwSciError_StreamInternalError, and then populated based on
    //!   fill mode.
    TrackArray<LwSciError>          statusCons;

    //! \brief Producer status value.
    //!   Initialized at construction to LwSciError_StreamInternalError,
    //!   and then populated based on fill mode.
    // Note: We could handle producer symmetrically with consumer, but since
    //       there is only 0 or 1 values, a vector would be overkill.
    LwSciError                      statusProd;

    //! \brief Flag indicating consumer status has completed.
    //!   Initialized to false at creation.
    std::atomic<bool>               statusConsCompleted;

    //! \brief Flag indicating producer status has completed.
    //!   Initialized to false at creation.
    std::atomic<bool>               statusProdCompleted;

    //! \brief Tracker used by multicast to collate status from all branches.
    //!   Initialized with size at construction.
    BranchTrack                     statusConsBranch;

    //! \brief Integer indicating packet status event is pending.
    //!   For most blocks, a value of 1 indicates the event is ready.
    //!   For the pool, where status must arrive from both directions,
    //!   a value of 2 is required.
    //!   Initialized to 0 at creation.
    std::atomic<uint32_t>           statusEvent;

    //! \brief Flag set if any producer or consumer rejected the packet.
    //!   Initialized to false at creation.
    bool                            rejected;

public:

    //
    // Packet fence operations
    //

    //! \brief Gets the indexed producer fence.
    //!
    //! \param [in] elemIndex: Index of fence to query.
    //! \param [in,out] fence: Wrapper in which to return copy of fence.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::get().
    LwSciError fenceProdGet(
        size_t const elemIndex,
        LwSciWrap::SyncFence& fence) const noexcept;

    //! \brief Gets the indexed consumer fence.
    //!
    //! \param [in] consIndex: Index of consumer to query.
    //! \param [in] elemIndex: Index of fence to query.
    //! \param [in,out] fence: Wrapper in which to return copy of fence.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::get().
    LwSciError fenceConsGet(
        size_t const consIndex,
        size_t const elemIndex,
        LwSciWrap::SyncFence& fence) const noexcept;

    //! \brief Sets the indexed producer fence.
    //!
    //! \param [in] elemIndex: Index of fence to set.
    //! \param [in] fence: Wrapper containing fence to store.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::set().
    LwSciError fenceProdSet(
        size_t const elemIndex,
        LwSciWrap::SyncFence const& fence) noexcept;

    //! \brief Sets the indexed consumer fence.
    //!
    //! \param [in] elemIndex: Index of fence to set.
    //! \param [in] fence: Wrapper containing fence to store.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::set().
    LwSciError fenceConsSet(
        size_t const elemIndex,
        LwSciWrap::SyncFence const& fence) noexcept;

    //! \brief Marks producer fence setup as done.
    //!
    //! \return void
    void fenceProdDone(void) noexcept;

    //! \brief Marks consumer fence setup as done.
    //!
    //! \return void
    void fenceConsDone(void) noexcept;

    //! \brief Fill all producer fences with copies of one fence.
    //!   Used for C2C, where a single C2C fence is waited for by
    //!   all endpoint engines.
    //!
    //! \param [in] fence: Wrapper containing fence to store.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::setAll().
    LwSciError fenceProdFill(
        LwSciWrap::SyncFence const& fence) noexcept;

    //! \brief Fill all consumer fences with copies of one fence.
    //!   Used for C2C, where a single C2C fence is waited for by
    //!   all endpoint engines.
    //!
    //! \param [in] fence: Wrapper containing fence to store.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::setAll().
    LwSciError fenceConsFill(
        LwSciWrap::SyncFence const& fence) noexcept;

    //! \brief Copy incoming producer fence data.
    //!
    //! \param [in] origPacket: Packet with fence data to copy.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::copy().
    LwSciError fenceProdCopy(Packet const& origPacket) noexcept;

    //! \brief Copy incoming consumer fence data.
    //!
    //! \param [in] origPacket: Packet with fence data to copy.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::copy().
    LwSciError fenceConsCopy(Packet const& origPacket) noexcept;

    //! \brief Collate incoming consumer fence data.
    //!
    //! \param [in] origPacket: Packet with fence data to copy.
    //! \param [in] branchIndex: Index of branch being collated in.
    //! \param [in] endRangeStart: Beginning of endpoint range into which
    //!   to copy the fences.
    //! \param [in] endRangeCount: Size of endpoint range into which
    //!   to copy the fences.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_IndexOutOfRange: Branch index is too large.
    //! * Any error returned by BranchTrack::set().
    //! * Any error returned by TrackArray::collate().
    LwSciError fenceConsCollate(
        Packet const& origPacket,
        size_t const branchIndex,
        size_t const endRangeStart,
        size_t const endRangeCount) noexcept;

    //! \brief Check if collated consumer fence data is complete and take
    //!   responsibility for sending it upstream.
    //!
    //! \param [in] mask: bitset value indicating disconnected consumer
    //!   branches.
    //!
    //! \return bool, Whether or not all fences are ready and caller should
    //!   pass them on.
    bool fenceConsCollateDone(std::bitset<MAX_DST_CONNECTIONS> const mask) noexcept;

    //! \brief Pack producer fence data.
    //!
    //! \param [in,out] buf: Buffer in which to pack the data.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::pack().
    LwSciError fenceProdPack(IpcBuffer& buf) const noexcept;

    //! \brief Pack consumer fence data.
    //!
    //! \param [in,out] buf: Buffer in which to pack the data.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::pack().
    LwSciError fenceConsPack(IpcBuffer& buf) const noexcept;

    //! \brief Unpack producer fence data.
    //!
    //! \param [in,out] buf: Buffer from which to unpack the data.
    //! \param [in] signal: Object containing sync objects for fences.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::unpack().
    LwSciError fenceProdUnpack(IpcBuffer& buf, Signals const& signal) noexcept;

    //! \brief Unpack consumer fence data.
    //!
    //! \param [in,out] buf: Buffer from which to unpack the data.
    //! \param [in] signal: Object containing sync objects for fences.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::unpack().
    LwSciError fenceConsUnpack(IpcBuffer& buf, Signals const& signal) noexcept;

    //! \brief CPU waits for all producer fences.
    //!
    //! \param [in] ctx: Context for CPU waiting.
    //! \param [in] timeout: Timeout to abort waiting.
    //!
    //! Note: Potentially, if every fence expires just before the timeout
    //!   value, this call could take timeout * numFences total time.
    //!   But we actually only ever use this with a timeout of 0 or infinity,
    //!   so this doesn't matter.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error encountered during fence duplication.
    //! * Any error returned by LwSciSyncFenceWait().
    LwSciError fenceProdWait(
        LwSciSyncCpuWaitContext const ctx,
        uint64_t const timeout=INFINITE_TIMEOUT) noexcept;

    //! \brief CPU waits for all consumer fences.
    //!
    //! \param [in] ctx: Context for CPU waiting.
    //! \param [in] timeout: Timeout to abort waiting.
    //!
    //! Note: Potentially, if every fence expires just before the timeout
    //!   value, this call could take timeout * numFences total time.
    //!   But we actually only ever use this with a timeout of 0 or infinity,
    //!   so this doesn't matter.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error encountered during fence duplication.
    //! * Any error returned by LwSciSyncFenceWait().
    LwSciError fenceConsWait(
        LwSciSyncCpuWaitContext const ctx,
        uint64_t const timeout=INFINITE_TIMEOUT) noexcept;

    //! \brief Resets producer fences.
    //!
    //! \return void
    void fenceProdReset(void) noexcept;

    //! \brief Resets consumer fences.
    //!
    //! \return void
    void fenceConsReset(void) noexcept;

private:

    //
    // Packet fence state
    //

    //! \brief Array of fences from the producer.
    //!   Allocated to an empty array of the appropriate size during
    //!   construction, and then populated based on its fill mode.
    //!   Only blocks which need to store producer fences will use this.
    TrackArray<LwSciWrap::SyncFence>    fenceProd;

    //! \brief Array of fences from the consumer.
    //!   Allocated to an empty array of the appropriate size during
    //!   construction, and then populated based on its fill mode.
    //!   Only blocks which need to store consumer fences will use this.
    TrackArray<LwSciWrap::SyncFence>    fenceCons;

    //! \brief Tracker used by multicast to collate fences from all branches.
    //!   Initialized with size at construction.
    BranchTrack                         fenceConsBranch;

public:

    //
    // Packet teardown operations
    //

    //! \brief Checks whether the packet instance is marked for deletion.
    //!
    //! \return boolean
    //! * true: If the packet instance is marked for deletion.
    //! * false: Otherwise.
    //!
    //! \if TIER4_SWAD
    //! \implements{20208468}
    //! \endif
    bool deleteGet(void) const noexcept;

    //! \brief Atomically marks the packet instance as deleted by the
    //!   application and enqueues a LwSciStreamEventType_PacketDelete event.
    //!
    //! \return boolean
    //! * true: If the packet instance was not previously marked as deleted.
    //! * false: Otherwise.
    //!
    //! \if TIER4_SWAD
    //! \implements{20208465}
    //! \endif
    bool deleteSet(void) noexcept;

    //! \brief Queries and clears the flag indicating packet delete event
    //!   is pending, and if so sets the flag for pending packet cookie.
    bool deletePending(void) noexcept
    {
        if (LwSciStreamCookie_Ilwalid != cookie) {
            bool expected { true };
            if (deleteEvent.compare_exchange_strong(expected, false)) {
                deleteCookieEvent.store(true);
                return true;
            }
        }
        return false;
    };

    //! \brief Queries and clears the flag indicating this packet's cookie
    //!   should be returned after a packet deletion event.
    bool deleteCookiePending(void) noexcept
    {
        bool expected { true };
        return deleteCookieEvent.compare_exchange_strong(expected, false);
    };

private:

    //
    // Packet teardown state
    //

    //! \brief Flag indicating packet deletion event is pending.
    //!   Initialized to false at creation.
    std::atomic<bool>               deleteEvent;

    //! \brief Flag indicating packet deletion event has been retrieved and
    //!   now retrieval of the cookie is pending.
    //!   Initialized to false at creation.
    std::atomic<bool>               deleteCookieEvent;

    // Not exported to doxygen. See Readme-doxygen-criteria.txt for details.
    // Flag indicating deletion has been triggered, but not yet performed.
    std::atomic<bool>               zombie;

public:
    //! \brief A simple queue, which is used by the Block to manage Payload(s)
    //!   on their way downstream or upstream in the correct order.
    //!
    //! \if TIER4_SWAD
    //! \implements{20037012}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{20383074}
    //! \endif
    //
    // Ideally, we would rely on some standardized queue template for this but:
    //   * The std:: containers are all non-intrusive, and therefore involve
    //     memory allocation/free whenever things are added to or removed from
    //     the queue, which we can't have in safety builds.
    //   * boost:: provides intrusive containers which would suit our needs,
    //     but this is not available to us. There are proposals to add such
    //     containers to std:: but this has not yet oclwrred.
    //   * Our needs are too simple and specialized to invest in implementing
    //     (and safety certifying) a full-fledged intrusive queue template.
    // Therefore, we implement a minimal queue called PayloadQ with a double
    // linked list specifically for Packets, as part of the Packet unit.
    class PayloadQ final
    {
    public:
        PayloadQ(void) noexcept                         = default;
        ~PayloadQ(void) noexcept                        = default;
        PayloadQ(const PayloadQ&) noexcept              = delete;
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
        PayloadQ(PayloadQ&&) noexcept                   = delete;
        PayloadQ& operator=(const PayloadQ&) & noexcept = delete;
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
        PayloadQ& operator=(PayloadQ&&) & noexcept      = delete;

        //! \brief Adds a packet instance at the tail of PayloadQ.
        //!
        //! \param [in] newPacket: Smart pointer of a packet instance.
        //!
        //! \return void.
        //!
        //! \if TIER4_SWAD
        //! \implements{20208531}
        //! \endif
        void enqueue(PacketPtr const& newPacket) noexcept;

        //! \brief Removes a packet instance from the head of PayloadQ.
        //!
        //! \return Smart pointer of the packet instance removed.
        //!   If PayloadQ is empty, returns nullptr.
        //!
        //! \if TIER4_SWAD
        //! \implements{20208534}
        //! \endif
        PacketPtr dequeue(void) noexcept;

        //! \brief Extracts a packet instance from anywhere in PayloadQ,
        //!   which is used only by pool when deleting.
        //!
        //! \param [in] oldPacket: Smart pointer of the packet instance to be
        //!   extracted.
        //!
        //! \return boolean
        //! * true: If the packet was queued and is extracted successfully.
        //! * false: Otherwise.
        //!
        //! \if TIER4_SWAD
        //! \implements{20208537}
        //! \endif
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
        bool extract(Packet& oldPacket) noexcept;

        //! \brief Checks if the PayloadQ is empty.
        //!
        //! \return boolean
        //! * true: If the PayloadQ is empty.
        //! * false: Otherwise.
        //!
        //! \if TIER4_SWAD
        //! \implements{20208540}
        //! \endif
        bool empty(void) const noexcept
        {
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            return (nullptr == head);
        };

    private:
        //! \cond TIER4_SWUD
        //! \brief Head of the PayloadQ.
        PacketPtr   head { };
        //! \brief Tail of the PayloadQ.
        PacketPtr   tail { };
        //! \endcond
    };

private:
    // Set prev/next pointers and queued flag, returning old value
    // TODO: Use move semantics?
    // Swaps payloadPrev with the newPrev.
    PacketPtr  swapPrev(PacketPtr const& newPrev) noexcept;

    // Swaps payloadNext with the newNext.
    PacketPtr  swapNext(PacketPtr const& newNext) noexcept;

    // Sets payloadQueued with the newQueued to indicate whether
    // the packet instance is in the PayloadQ or not.
    bool       swapQueued(bool const newQueued) noexcept;

private:
    //! \cond TIER4_SWUD

    // Pointers to adjacent packets in payload queue
    // Note: Prev is closer to tail. Next is closer to head.

    //! \brief Pointer to its previous packet instance in PayloadQ.
    PacketPtr           payloadPrev;
    //! \brief Pointer to its next packet instance in PayloadQ.
    PacketPtr           payloadNext;
    //! \brief Indicates whether the packet instance is in PayloadQ.
    bool                payloadQueued;

    //! \endcond
};

} // namespace LwSciStream

#endif // PACKET_H
