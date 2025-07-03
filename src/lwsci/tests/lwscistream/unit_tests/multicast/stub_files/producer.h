//! \file
//! \brief LwSciStream producer class declaration.
//!
//! \copyright
//! Copyright (c) 2018-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef PRODUCER_H
#define PRODUCER_H
#include <cstdint>
#include <atomic>
#include <cstdlib>
#include <utility>
#include "covanalysis.h"
#include "block.h"
#include "trackarray.h"
#include "elements.h"
#include "syncwait.h"
#include "syncsignal.h"

namespace LwSciStream {

//! \brief Producer class implements the functionality of the producer block.
//!
//! * It inherits from the Block class which provides common functionalities
//!   for all blocks.
//! * It has to override the supported API functions from APIBlockInterface,
//!   which are called by LwSciStream public APIs.
//! * Producer block accepts downstream connections. So it has to override
//!   SrcBlockInterface functions, which are called by downstream (destination)
//!   block.
//!
//! \if TIER4_SWAD
//! \implements{19465737}
//! \endif
//!
//! \if TIER4_SWUD
//! \implements{19388940}
//! \endif
class Producer :
    public Block
{
public:
    //! \brief Constructs an instance of the producer class and
    //!  initializes the producer specific data members.
    //!
    //! \if TIER4_SWAD
    //! \implements{19464768}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    Producer(void) noexcept;

    Producer(const Producer&) noexcept = delete;

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Producer(Producer&&) noexcept = delete;

    Producer& operator=(const Producer&) & noexcept = delete;

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Producer& operator=(Producer&&) & noexcept = delete;

    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    ~Producer(void) noexcept final = default;

    //
    // Connection definition functions
    //

    //! \brief Connects the producer block to the pool block referenced by
    //!  the @a paramPool.
    //!
    //!  <b>Preconditions</b>
    //!   - Producer block instance should have been already registered
    //!     by a successful call to Block::registerBlock() interface.
    //!
    //! \param [in] paramPool: reference to the pool block instance.
    //!  Valid value: paramPool is valid if it is not NULL.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: If producer and pool blocks are connected
    //!   successfully.
    //! * LwSciError_BadParameter: If pool block reference is NULL or the
    //!   @a paramPool is not referring to a pool block.
    //! * LwSciError_InsufficientResource: Producer block has no available output
    //!   connection or the Pool block has no available input connection.
    //!
    //! \if TIER4_SWAD
    //! \implements{19464333}
    //! \endif
    LwSciError BindPool(BlockPtr const& paramPool) noexcept;

    //! \brief Producer block retrieves the associated pool block object.
    //!
    //! \param [in,out] paramBlock: pointer to pool block object.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: If producer block initialization is successful.
    //! * LwSciError_NotInitialized: If producer block initialization is failed.
    //!
    //! \if TIER4_SWAD
    //! \implements{19464354}
    //! \endif
    LwSciError getOutputConnectPoint(
        BlockPtr& paramBlock) const noexcept final;

    //! \brief Saves producer block's endpoint info into the generic vector
    //!    shared by all blocks and ilwokes base block option finalization.
    //!
    //! \return void
    void finalizeConfigOptions(void) noexcept final;

    //! \copydoc LwSciStream::APIBlockInterface::apiSetupStatusSet
    LwSciError apiSetupStatusSet(
        LwSciStreamSetup const setupType) noexcept final;

    //! \brief Receives consumer info from downstream, saves, and initiates
    //!   flow of producer info downstream.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvConsInfo
    //!
    //! \return void
    void dstRecvConsInfo(
        uint32_t const dstIndex,
        EndInfoVector const& info) noexcept final;

    //
    // Element definition functions
    //

    //! \copydoc LwSciStream::APIBlockInterface::apiElementAttrSet
    LwSciError apiElementAttrSet(
         uint32_t const elemType,
         LwSciWrap::BufAttr const& elemBufAttr) noexcept final;

    //! \copydoc LwSciStream::APIBlockInterface::apiElementCountGet
    LwSciError apiElementCountGet(
        LwSciStreamBlockType const queryBlockType,
        uint32_t& numElements) noexcept final;

    //! \copydoc LwSciStream::APIBlockInterface::apiElementTypeGet
    LwSciError apiElementTypeGet(
        LwSciStreamBlockType const queryBlockType,
        uint32_t const elemIndex,
        uint32_t& userType) noexcept final;

    //! \copydoc LwSciStream::APIBlockInterface::apiElementAttrGet
    LwSciError apiElementAttrGet(
        LwSciStreamBlockType const queryBlockType,
        uint32_t const elemIndex,
        LwSciBufAttrList& bufAttrList) noexcept final;

    //! \brief Receives allocated element information from pool and prepares
    //!   LwSciStreamEventType_Elements event.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvAllocatedElements
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Elements::dataCopy().
    void dstRecvAllocatedElements(
        uint32_t const dstIndex,
        Elements const& inElements) noexcept final;

    //
    // Packet definition functions
    //

    //! \brief Retrieves handle for newly created packet after dequeuing a
    //!   LwSciStreamEventType_PacketCreate event.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPacketNewHandleGet
    LwSciError apiPacketNewHandleGet(
        LwSciStreamPacket& handle) noexcept final;

    //! \brief Retrieves a buffer object handle for the indexed element of
    //!   the specified packet. The handle is owned by the caller and must
    //!   be freed when it is no longer needed.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPacketBufferGet
    LwSciError apiPacketBufferGet(
        LwSciStreamPacket const handle,
        uint32_t const elemIndex,
        LwSciWrap::BufObj& bufObjWrap) noexcept final;

    //! \brief Retrieves cookie for deleted a packet after dequeuing
    //!   a LwSciStreamEventType_PacketDelete event. After this call,
    //!   the Packet instance is removed from the map and may no
    //!   longer be used in API calls.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPacketOldCookieGet
    LwSciError apiPacketOldCookieGet(
        LwSciStreamCookie& cookie) noexcept final;

    //! \brief Accepts or rejects a packet, providing a cookie on acceptance.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPacketStatusSet
    LwSciError apiPacketStatusSet(
        LwSciStreamPacket const handle,
        LwSciStreamCookie const cookie,
        LwSciError const status) noexcept final;

    //! \brief Creates a new packet instance for producer block and prepares
    //!  LwSciStreamEventType_PacketCreate event.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPacketCreate
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Block::pktCreate().
    void dstRecvPacketCreate(
        uint32_t const dstIndex,
        Packet const& origPacket) noexcept final;

    //! \brief Marks the producer block's packet instance referenced by
    //!   the given @a handle for deletion and prepares the
    //!   LwSciStreamEventType_PacketDelete event.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPacketDelete
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: Packet handle not found in map.
    //! - LwSciError_StreamPacketInaccessible: Packet is not lwrrently
    //!   downstream of the producer.
    //! - LwSciError_StreamPacketDeleted: Packet was already deleted.
    void dstRecvPacketDelete(
        uint32_t const dstIndex,
        LwSciStreamPacket const handle) noexcept final;

    //! \brief Triggers LwSciStreamEventType_PacketsComplete event.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPacketsComplete
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_AlreadyDone: Packet set was already marked completed.
    void dstRecvPacketsComplete(
        uint32_t const dstIndex) noexcept final;

    //
    // Sync waiter functions
    //

    //! \brief Specifies per-element waiter attribute requirements
    //!   for the producer.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiElementWaiterAttrSet
    LwSciError apiElementWaiterAttrSet(
        uint32_t const elemIndex,
        LwSciWrap::SyncAttr const& syncAttr) noexcept final;

    //! \brief Retrieves per-element waiter attribute requirements
    //!   from the consumer(s).
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiElementWaiterAttrGet
    LwSciError apiElementWaiterAttrGet(
        uint32_t const elemIndex,
        LwSciWrap::SyncAttr& syncAttr) noexcept final;

    //! \brief Receives the LwSciSync waiter information from the consumer(s)
    //!   and prepares LwSciStreamEventType_WaiterAttr event.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSyncWaiter
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Waiters::copy().
    void dstRecvSyncWaiter(
        uint32_t const dstIndex,
        Waiters const& syncWaiter) noexcept final;

    //
    // Sync signaller functions
    //

    //! \brief Specifies the @a syncObj that the producer uses to signal
    //!   when it is done writing to element @a elemIndex.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiElementSignalObjSet
    LwSciError apiElementSignalObjSet(
        uint32_t const elemIndex,
        LwSciWrap::SyncObj const& syncObj) noexcept final;

    //! \brief Retrieves the @a syncObj that consumer @a queryBlockIndex
    //!   signals when it is done reading from element @a elemIndex.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiElementSignalObjGet
    LwSciError apiElementSignalObjGet(
        uint32_t const queryBlockIndex,
        uint32_t const elemIndex,
        LwSciWrap::SyncObj& syncObj) noexcept final;

    //! \brief Receives the LwSciSync signal information from the consumer(s)
    //!   and prepares LwSciStreamEventType_SignalObj event.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSyncSignal
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Signals::copy().
    void dstRecvSyncSignal(
        uint32_t const dstIndex,
        Signals const& syncSignal) noexcept final;

    //
    // Payload functions
    //

    //! \brief Obtains an empty packet from the pool to use for new data.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPayloadObtain
    LwSciError apiPayloadObtain(
        LwSciStreamCookie& cookie) noexcept final;

    //! \brief Inserts a completed packet into the stream for processing
    //!   by the consumer(s).
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPayloadReturn
    LwSciError apiPayloadReturn(
        LwSciStreamPacket const handle) noexcept final;

    //! \brief Specifics the fence indicating when the producer will be
    //!   done writing to the indexed element of the referenced packet.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPayloadFenceSet
    LwSciError apiPayloadFenceSet(
        LwSciStreamPacket const handle,
        uint32_t const elemIndex,
        LwSciWrap::SyncFence const& postfence) noexcept final;

    //! \brief Specifics the fence indicating when the indexed consumer will
    //!   done reading from the indexed element of the referenced packet.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPayloadFenceGet
    LwSciError apiPayloadFenceGet(
        LwSciStreamPacket const handle,
        uint32_t const queryBlockIndex,
        uint32_t const elemIndex,
        LwSciWrap::SyncFence& prefence) noexcept final;

    //! \brief Informs producer that a packet is available in the pool,
    //!   triggering an LwSciStreamEventType_PacketReady event, but
    //!   ignoring the Packet information.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPayload
    void dstRecvPayload(
        uint32_t const dstIndex,
        Packet const& consPayload) noexcept final;

    //! \brief Producer block disconnects its destination block.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::disconnect
    //!
    //! \if TIER4_SWAD
    //! \implements{19464561}
    //! \endif
    LwSciError disconnect(void) noexcept final;

    //! \brief Disconnects the destination block and prepares
    //!  LwSciStreamEventType_Disconnected event.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstDisconnect
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19464753}
    //! \endif
    void dstDisconnect(
        uint32_t const dstIndex) noexcept final;

    //! \copydoc LwSciStream::APIBlockInterface::apiUserInfoSet
    LwSciError apiUserInfoSet(
        uint32_t const userType,
        InfoPtr const& info) noexcept final;

protected:
    //! \brief Queries any pending event from producer block.
    //!
    //! \param [out] event: Queried event.
    //!
    //! \return bool
    //!  - True: If event is queried successfully.
    //!  - False: If no pending events on Producer block.
    //!
    //! \if TIER4_SWAD
    //! \implements{19464780}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    bool pendingEvent(LwSciStreamEventType& event) noexcept final;

    //! \brief Overrides the default and triggers the phase change.
    //!
    //! \return void
    void phaseSendReady(void) noexcept final;

private:
    //! \cond TIER4_SWAD
    //! \brief Pointer to pool block instance. It is initialized when a pool
    //!  block is connected with the producer block in Producer::BindPool()
    //!  interface.
    BlockPtr pool;
    //! \endcond

    //! \brief Information for this endpoint. It is initialized to default
    //!   values at construction.
    EndInfo                     endpointInfo { };

    //! \brief Tracks list of supported elements provided by user.
    //!   It is initialized to default values at creation and filled
    //!   by application calls to apiElementAttrSet().
    Elements                    supportedElements { FillMode::User,
                                                    FillMode::None };

    //! \brief Tracks list of allocated elements determined by pool.
    //!   It is initialized to default values at creation and filled
    //!   when the information is received from the pool.
    Elements                    allocatedElements { FillMode::Copy,
                                                    FillMode::Copy};

    //! \brief Tracks the waiter sync attributes provided by this producer
    //!   for each element, and any related information.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Waiters                     prodSyncWaiter { FillMode::User, false };

    //! \brief Tracks the waiter sync attributes provided by the consumer(s)
    //!   for each element, and any related information.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Waiters                     consSyncWaiter { FillMode::Copy, true };

    //! \brief Tracks the signaller sync objects provided by the producer
    //!   for each element, and any related information.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Signals                     prodSyncSignal { FillMode::User,
                                                 &consSyncWaiter };

    //! \brief Tracks the signaller sync objects provided by the consumer(s)
    //!   for each element, and any related information.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Signals                     consSyncSignal { FillMode::Copy,
                                                 &prodSyncWaiter };

    //! \brief Tracks completition of element import.
    //!   It is initialized to false at creation.
    // TODO: Maybe should be in Elements
    std::atomic<bool>           elementImportDone { false };

    //! \brief Tracks completion of packet export by pool.
    //!   It is initialized to false at creation.
    std::atomic<bool>           packetExportDone { false };

    //! \brief Indicates PacketsComplete event is pending.
    //!   It is initialized to false at creation.
    std::atomic<bool>           packetExportEvent { false };

    //! \brief Tracks completion of packet import.
    //!   It is initialized to false at creation.
    std::atomic<bool>           packetImportDone { false };

    //! \brief Tracks completion of waiter export.
    //!   It is initialized to false at creation.
    std::atomic<bool>           waiterExportDone { false };

    //! \brief Tracks completion of waiter import.
    //!   It is initialized to false at creation.
    std::atomic<bool>           waiterImportDone { false };

    //! \brief Tracks completion of signal export.
    //!   It is initialized to false at creation.
    std::atomic<bool>           signalExportDone { false };

    //! \brief Tracks completion of signal import.
    //!   It is initialized to false at creation.
    std::atomic<bool>           signalImportDone { false };

    //! \cond TIER4_SWUD
    //! \brief Count of pending LwSciStreamEventType_PacketReady events. It is
    //!  initialized to zero when a producer instance is created. It is
    //!  incremented when pool block informs the packet availability for reuse
    //!  through Producer::dstRecvPayload() interface and decremented when a
    //!  LwSciStreamEventType_PacketReady event is queried using
    //!  Producer::pendingEvent() interface.
    std::atomic<uint32_t>        pktReadyEvents { 0U };
    //! \endcond
};

} //namespace LwSciStream

#endif // PRODUCER_H
