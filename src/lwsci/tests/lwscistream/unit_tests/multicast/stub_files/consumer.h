//! \file
//! \brief LwSciStream consumer class declaration.
//!
//! \copyright
//! Copyright (c) 2018-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef CONSUMER_H
#define CONSUMER_H
#include <cstdint>
#include <utility>
#include <atomic>
#include "covanalysis.h"
#include "block.h"
#include "trackarray.h"
#include "elements.h"
#include "syncwait.h"
#include "syncsignal.h"

namespace LwSciStream {

//! \brief Consumer class implements the functionality of the consumer block.
//!
//! * It inherits from the Block class which provides common functionalities
//!   for all blocks.
//! * It has to override the supported API functions from APIBlockInterface,
//!   which are called by LwSciStream public APIs.
//! * Consumer block accepts upstream connections. So it has to override
//!   DstBlockInterface functions, which are called by upstream (source)
//!   block.
//!
//! \if TIER4_SWAD
//! \implements{19500540}
//! \endif
//!
//! \if TIER4_SWUD
//! \implements{19500561}
//! \endif
class Consumer :
    public Block
{
public:
    //! \brief Constructs an instance of the consumer class and
    //!  initializes the consumer specific data members.
    //!
    //! \if TIER4_SWAD
    //! \implements{19507656}
    //! \endif
    Consumer(void) noexcept;

    Consumer(const Consumer&) noexcept = delete;

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Consumer(Consumer&&) noexcept = delete;

    Consumer& operator=(const Consumer&) & noexcept = delete;

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Consumer& operator=(Consumer&&) & noexcept = delete;

    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    ~Consumer(void) noexcept final = default;

    //
    // Connection definition functions
    //

    //! \brief Connects the consumer block to the queue block referenced by
    //!  the @a paramQueue.
    //!
    //!  <b>Preconditions</b>
    //!   - Consumer block instance should have been already registered
    //!     by a successful call to Block::registerBlock() interface.
    //!
    //! \param [in] paramQueue: reference to the queue block instance.
    //!  Valid value: paramQueue is valid if it is not NULL.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: If consumer and queue blocks are connected
    //!   successfully.
    //! * LwSciError_BadParameter: If queue block reference is NULL or the
    //!   @a paramQueue is not referring to a queue block.
    //! * LwSciError_InsufficientResource: Consumer block has no available input
    //!   connection or the Queue block has no available output connection.
    //!
    //! \if TIER4_SWAD
    //! \implements{19507662}
    //! \endif
    LwSciError BindQueue(BlockPtr const& paramQueue) noexcept;

    // Override functions inherited from APIBlockInterface
    //! \brief Consumer block retrieves the associated queue block object.
    //!
    //! \param [in,out] paramBlock: pointer to queue block object.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: If consumer block initialization is successful.
    //! * LwSciError_NotInitialized: If consumer block initialization is failed.
    //!
    //! \if TIER4_SWAD
    //! \implements{19507668}
    //! \endif
    LwSciError getInputConnectPoint(
        BlockPtr& paramBlock) const noexcept final;

    //! \copydoc LwSciStream::Block::finalizeConfigOptions
    void finalizeConfigOptions(void) noexcept final;

    //! \copydoc LwSciStream::APIBlockInterface::apiSetupStatusSet
    LwSciError apiSetupStatusSet(
        LwSciStreamSetup const setupType) noexcept final;

    //! \brief Receives producer info from downstream and saves
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvProdInfo
    //!
    //! \return void
    void srcRecvProdInfo(
        uint32_t const srcIndex,
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

    //! \copydoc LwSciStream::APIBlockInterface::apiElementUsageSet
    LwSciError apiElementUsageSet(
        uint32_t const elemIndex,
        bool const used) noexcept final;

    //! \brief Receives allocated element information from pool and prepares
    //!   LwSciStreamEventType_Elements event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvAllocatedElements
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_InsufficientMemory: Unable to allocate memory for the
    //!   vector to track element usage.
    //! - Any error returned by Elements::dataCopy().
    void srcRecvAllocatedElements(
        uint32_t const srcIndex,
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

    //! \brief Creates a new packet instance for consumer block and prepares
    //!  LwSciStreamEventType_PacketCreate event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPacketCreate
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Block::pktCreate().
    void srcRecvPacketCreate(
        uint32_t const srcIndex,
        Packet const& origPacket) noexcept final;

    //! \brief Marks the consumer block's packet instance referenced by
    //!   the given @a handle for deletion and prepares the
    //!   LwSciStreamEventType_PacketDelete event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPacketDelete
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: Packet handle not found in map.
    //! - LwSciError_StreamPacketInaccessible: Packet is not lwrrently
    //!   upstream of the consumer.
    //! - LwSciError_StreamPacketDeleted: Packet was already deleted.
    void srcRecvPacketDelete(
        uint32_t const srcIndex,
        LwSciStreamPacket const handle) noexcept final;

    //! \brief Triggers LwSciStreamEventType_PacketsComplete event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPacketsComplete
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_AlreadyDone: Packet set was already marked completed.
    void srcRecvPacketsComplete(
        uint32_t const srcIndex) noexcept final;

    //
    // Packet definition functions
    //

    //! \brief Specifies per-element waiter attribute requirements
    //!   for the consumer.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiElementWaiterAttrSet
    LwSciError apiElementWaiterAttrSet(
        uint32_t const elemIndex,
        LwSciWrap::SyncAttr const& syncAttr) noexcept final;

    //! \brief Retrieves per-element waiter attribute requirements
    //!   from the producer.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiElementWaiterAttrGet
    LwSciError apiElementWaiterAttrGet(
        uint32_t const elemIndex,
        LwSciWrap::SyncAttr& syncAttr) noexcept final;

    //! \brief Receives the LwSciSync waiter information from the producer
    //!   and prepares LwSciStreamEventType_WaiterAttr event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvSyncWaiter
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Waiters::copy().
    void srcRecvSyncWaiter(
        uint32_t const srcIndex,
        Waiters const& syncWaiter) noexcept final;

    //
    // Sync signaller functions
    //

    //! \brief Specifies the @a syncObj that the consumer uses to signal
    //!   when it is done reading from element @a elemIndex.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiElementSignalObjSet
    LwSciError apiElementSignalObjSet(
        uint32_t const elemIndex,
        LwSciWrap::SyncObj const& syncObj) noexcept final;

    //! \brief Retrieves the @a syncObj that the producer
    //!   signals when it is done writing to element @a elemIndex.
    //!   There is only one producer, so @a queryBlockIndex must be 0.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiElementSignalObjGet
    LwSciError apiElementSignalObjGet(
        uint32_t const queryBlockIndex,
        uint32_t const elemIndex,
        LwSciWrap::SyncObj& syncObj) noexcept final;

    //! \brief Receives the LwSciSync signal information from the producer
    //!   and prepares LwSciStreamEventType_SignalObj event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvSyncSignal
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Signals::copy().
    void srcRecvSyncSignal(
        uint32_t const srcIndex,
        Signals const& syncSignal) noexcept final;

    //
    // Payload functions
    //

    //! \brief Obtains a full packet from the queue with available data.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPayloadObtain
    LwSciError apiPayloadObtain(
        LwSciStreamCookie& cookie) noexcept final;

    //! \brief Inserts a no longer needed packet into the stream for reuse
    //!   by the producer.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPayloadReturn
    LwSciError apiPayloadReturn(
        LwSciStreamPacket const handle) noexcept final;

    //! \brief Specifics the fence indicating when the consumer will be
    //!   done reading from the indexed element of the referenced packet.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPayloadFenceSet
    LwSciError apiPayloadFenceSet(
        LwSciStreamPacket const handle,
        uint32_t const elemIndex,
        LwSciWrap::SyncFence const& postfence) noexcept final;

    //! \brief Specifics the fence indicating when the producer will
    //!   done writing to the indexed element of the referenced packet.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPayloadFenceGet
    LwSciError apiPayloadFenceGet(
        LwSciStreamPacket const handle,
        uint32_t const queryBlockIndex,
        uint32_t const elemIndex,
        LwSciWrap::SyncFence& prefence) noexcept final;

    //! \brief Informs consumer that a packet is available in the queue,
    //!   triggering an LwSciStreamEventType_PacketReady event, but
    //!   ignoring the Packet information.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPayload
    void srcRecvPayload(
        uint32_t const srcIndex,
        Packet const& prodPayload) noexcept final;

    //! \brief Consumer block disconnects its source block.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::disconnect
    //!
    //! \if TIER4_SWAD
    //! \implements{19508229}
    //! \endif
    LwSciError disconnect(void) noexcept final;

    //! \brief Disconnects the source block and prepares
    //!  LwSciStreamEventType_Disconnected event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcDisconnect
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19508265}
    //! \endif
    void srcDisconnect(
        uint32_t const srcIndex) noexcept final;

    //! \copydoc LwSciStream::APIBlockInterface::apiUserInfoSet
    LwSciError apiUserInfoSet(
        uint32_t const userType,
        InfoPtr const& info) noexcept final;

protected:
    // Consumer-specific query for next pending event
    //! \brief Queries any pending event from consumer block.
    //!
    //! \param [out] event: Queried event.
    //!
    //! \return bool
    //!  - True: If event is queried successfully.
    //!  - False: If no pending events on Consumer block.
    //!
    //! \if TIER4_SWAD
    //! \implements{19508271}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    bool pendingEvent(LwSciStreamEventType& event) noexcept final;

private:
    //! \cond TIER4_SWAD
    //! \brief Pointer to queue block instance. It is initialized when a queue
    //!  block is connected with the consumer block in Consumer::BindQueue().
    BlockPtr                     queue { nullptr };
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
                                                    FillMode::Copy };

    //! \brief Tracks which allocated elements the consumer will use.
    //!   It is initialized when the allocatedElements list is received
    //!   to an array with a value of 1 for each element.
    //!   Note: We only need booleans values, but std::vector<bool> is not
    //!         allowed by AUTOSAR.
    //!   TODO: This will become part of a Usage object that will be passed
    //!         between blocks in a future optimization.
    std::vector<uint8_t>        usedElements { };

    //! \brief Tracks the waiter sync attributes provided by this consumer
    //!   for each element, and any related information.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Waiters                     consSyncWaiter { FillMode::User, true };

    //! \brief Tracks the waiter sync attributes provided by the producer
    //!   for each element, and any related information.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Waiters                     prodSyncWaiter { FillMode::Copy, false };

    //! \brief Tracks the signaller sync objects provided by the consumer(s)
    //!   for each element, and any related information.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Signals                     consSyncSignal { FillMode::User,
                                                 &prodSyncWaiter };

    //! \brief Tracks the signaller sync objects provided by the producer
    //!   for each element, and any related information.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Signals                     prodSyncSignal { FillMode::Copy,
                                                 &consSyncWaiter };

    //! \brief Tracks completition of element import.
    //!   It is initialized to false at creation
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
    //!  initialized to zero when a consumer instance is created. It is
    //!  incremented when queue block sends the packet availability
    //!  information through Consumer::srcRecvPayload() and decremented
    //!  when a LwSciStreamEventType_PacketReady event is queried using
    //!  Consumer::pendingEvent() interface.
    std::atomic<uint32_t>        pktReadyEvents { 0U };
    //! \endcond
};

} //namespace LwSciStream

#endif // CONSUMER_H
