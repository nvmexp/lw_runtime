//! \file
//! \brief LwSciStream pool class declaration.
//!
//! \copyright
//! Copyright (c) 2018-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef POOL_H
#define POOL_H
#include <cstdint>
#include <cstdlib>
#include <utility>
#include "covanalysis.h"
#include "block.h"
#include "trackarray.h"
#include "elements.h"

namespace LwSciStream {

//! \brief Pool class provides a pool of Packets to upstream blocks to produce
//! the data. Packets are returned back to Pool class by downstream blocks once
//! the consumption of packets is finished.
//!
//! The pool attached to the producer will be considered the "primary" pool,
//! which decides the packet layout. Any additional pools are "secondary".
//! When the application attaches packets to them, they will be expected to
//! adhere to the layout defined by the primary pool.

//! - Pool class is inherited from Block class which provides common
//! functionality for all the blocks.
//! - Pool class overrides functions from APIBlockInterface which are called by
//! LwSciStream public APIs.
//! - Pool class also overrides functions from SrcBlockInterface and
//! DstBlockInterface since it serves Producer class which is upstream block
//! and Consumer class which is downstream block.
//!
//! \if TIER4_SWAD
//! \implements{19426977}
//! \endif
//!
//! \if TIER4_SWUD
//! \implements{20716128}
//! \endif
class Pool :
    public Block
{
public:
    //! \brief Constructs Pool for number of packets specified by
    //! @a paramNumPackets.
    //!
    //! \param[in] paramNumPackets: number of desired packets in the Pool.
    //!
    //! \if TIER4_SWAD
    //! \implements{19737150}
    //! \endif
    explicit Pool(uint32_t const paramNumPackets) noexcept;

    Pool(const Pool&) noexcept = delete;

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Pool(Pool&&) noexcept = delete;

    Pool& operator=(const Pool&) & noexcept = delete;

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Pool& operator=(Pool&&) & noexcept = delete;

    //! \brief Destroys the pool block instance.
    //!
    //! \if TIER4_SWAD
    //! \implements{19737243}
    //! \endif
    ~Pool(void) noexcept override;

    //! \brief Sets the flag to indicate it a secondary pool, which is not
    //!   attached to Producer block.
    //!
    //! \if TIER4_SWAD
    //! \implements{}
    //! \endif
    void makeSecondary(void) noexcept;

    //! \brief A stub implementation which always returns
    //! LwSciError_AccessDenied, as pool does not allow output connections
    //! through public APIs.
    //!
    //! \param[out] paramBlock: unused.
    //!
    //! \return LwSciError, Always LwSciError_AccessDenied.
    //!
    //! \implements{19737153}
    LwSciError getOutputConnectPoint(
        BlockPtr& paramBlock) const noexcept final;

    //! \brief A stub implementation which always returns
    //! LwSciError_AccessDenied, as pool does not allow input connections
    //! through public APIs.
    //!
    //! \param[out] paramBlock: unused.
    //!
    //! \return LwSciError, Always LwSciError_AccessDenied.
    //!
    //! \implements{19737156}
    LwSciError getInputConnectPoint(
        BlockPtr& paramBlock) const noexcept final;

    //! \copydoc LwSciStream::APIBlockInterface::apiSetupStatusSet
    LwSciError apiSetupStatusSet(
        LwSciStreamSetup const setupType) noexcept final;

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

    //! \brief Receives supported element information from producer and
    //!   saves a copy. If supported elements from both producer and
    //!   consumer(s) have arrived, signals an LwSciStreamEventType_Elements
    //!   is available.
    //!   This interface is not supported on the secondary pool.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvSupportedElements
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Elements::dataCopy().
    void srcRecvSupportedElements(
        uint32_t const srcIndex,
        Elements const& inElements) noexcept final;

    //! \brief Receives supported element information from consumer(s) and
    //!   saves a copy. If supported elements from both producer and
    //!   consumer(s) have arrived, signals an LwSciStreamEventType_Elements
    //!   is available.
    //!   If not the primary pool, forwards the elements upstream.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSupportedElements
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Elements::dataCopy().
    void dstRecvSupportedElements(
        uint32_t const dstIndex,
        Elements const& inElements) noexcept final;

    //! \brief Receives allocated element information from primary pool
    //!   and prepares LwSciStreamEventType_Elements event.
    //!   This interface is only supported on the secondary pool.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvAllocatedElements
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_NotYetAvailable: Consumer elements not arrived yet.
    //! - Any error returned by Elements::dataCopy() and Elements::dataSend().
    void srcRecvAllocatedElements(
        uint32_t const srcIndex,
        Elements const& inElements) noexcept final;

    //! \brief Creates a new Packet instance for the pool, stores the given
    //!  @a cookie for it, and sends the new LwSciStreamPacket upstream and
    //!  downstream to producer and consumer blocks respectively.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPacketCreate
    //!
    //! \if TIER4_SWAD
    //! \implements{19737204}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
    LwSciError apiPacketCreate(
        LwSciStreamCookie const cookie,
        LwSciStreamPacket& handle) noexcept final;

    //! \brief Registers LwSciBufObj contained by the @a elemBufObj for
    //!  the packet element referenced by the given @a elemIndex of the
    //!  LwSciStreamPacket instance referenced by the given @a packetHandle.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPacketBuffer
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    LwSciError apiPacketBuffer(
        LwSciStreamPacket const packetHandle,
        uint32_t const elemIndex,
        LwSciWrap::BufObj const& elemBufObj) noexcept final;

    //! \brief Verifies that the LwSciStreamPacket represented by @a handle
    //!   is fully specified and sends the packet information to the rest
    //!   of the stream.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPacketComplete
    LwSciError apiPacketComplete(
        LwSciStreamPacket const handle) noexcept final;

    //! \brief Schedules the packet instance referenced by the given
    //!  @a handle for deletion.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPacketDelete
    LwSciError apiPacketDelete(
        LwSciStreamPacket const handle) noexcept final;

    //! \brief Queries whether the packet instance referenced by @ handle
    //!   was accepted by the producer and all consumers.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPacketStatusAcceptGet
    LwSciError apiPacketStatusAcceptGet(
        LwSciStreamPacket const handle,
        bool& accepted) noexcept override;

    //! \brief Queries the packet instance referenced by @ handle for the
    //!   status value sent by the endpoint identified by @a queryBlockType
    //!   and @a queryBlockIndex.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::apiPacketStatusValueGet
    LwSciError apiPacketStatusValueGet(
        LwSciStreamPacket const handle,
        LwSciStreamBlockType const queryBlockType,
        uint32_t const queryBlockIndex,
        LwSciError& status) noexcept override;

    //! \brief Receives acceptance or rejection of a packet from the
    //!   producer and saves the information. If status has arrived
    //!   for the packet for the consumer(s) as well, prepares
    //!   LwSciStreamEventType_PacketStatus event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPacketStatus
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: Packet handle not found in map.
    //! - Any error returned by Packet::statusProdCopy().
    void srcRecvPacketStatus(
        uint32_t const srcIndex,
        Packet const& origPacket) noexcept final;

    //! \brief Receives acceptance or rejection of a packet from the
    //!   consumer(s) and saves the information. If status has arrived
    //!   for the packet for the producer as well, prepares
    //!   LwSciStreamEventType_PacketStatus event.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPacketStatus
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: Packet handle not found in map.
    //! - Any error returned by Packet::statusConsCopy().
    void dstRecvPacketStatus(
        uint32_t const dstIndex,
        Packet const& origPacket) noexcept final;

    //! \brief Gets the next available packet from the pool.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcDequeuePayload
    //!
    //! \if TIER4_SWAD
    //! \implements{19737210}
    //! \endif
    PacketPtr srcDequeuePayload(
        uint32_t const srcIndex) noexcept final;

    //! \brief Receives returned payload from consumer(s) and stores it.
    //!   If the packet was previously deleted by the user, frees it and
    //!   informs the rest of the stream. Otherwise, informs the producer
    //!   that a packet is available.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPayload
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: Packet provided is not valid.
    //! - LwSciError_StreamPacketInaccessible: Packet provided is not
    //!   downstream.
    //! - Any error returned by Packet::fenceConsCopy().
    //!
    //! \if TIER4_SWAD
    //! \implements{19737216}
    //! \endif
    void dstRecvPayload(
        uint32_t const dstIndex,
        Packet const& consPayload) noexcept final;

    //! \brief Disconnects the source and destination blocks of the pool.
    //!
    //! \return LwSciError, Always LwSciError_Success.
    //!
    //! \if TIER4_SWAD
    //! \implements{19737234}
    //! \endif
    LwSciError disconnect(
        void) noexcept final;

    //! \brief Disconnects the source and destination blocks and prepares
    //!  LwSciStreamEventType_Disconnected event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcDisconnect
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19737237}
    //! \endif
    void srcDisconnect(
        uint32_t const srcIndex) noexcept final;

    //! \brief Disconnects the source and destination blocks and prepares
    //!  LwSciStreamEventType_Disconnected event.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstDisconnect
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19737240}
    //! \endif
    void dstDisconnect(
        uint32_t const dstIndex) noexcept final;

protected:
    //! \brief Queries any pending event from pool block.
    //!
    //! \param [out] event: Location in which to return LwSciStreamEventType.
    //!
    //! \return boolean
    //! - True if event is queried successfully.
    //! - False otherwise.
    //!
    //! \if TIER4_SWAD
    //! \implements{19737246}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    bool pendingEvent(
        LwSciStreamEventType& event) noexcept final;

private:
    // Not exported to doxygen. See Readme-doxygen-criteria.txt for details.
    //! \cond
    //! \brief Constant index for producer endpoint in received data lists.
    static constexpr uint32_t   prodEndIndex  { 0U };

    //! \brief Constant index for consumer endpoint in received data lists.
    static constexpr uint32_t   consEndIndex  { 1U };

    //! \brief Constant index for number of endpoints.
    static constexpr uint32_t   endpointCount { 2U };

    //! \brief Number of packet this pool expects
    uint32_t const        numPacketsDesired;

    //! \brief Number of packets created
    uint32_t              numPacketsCreated;

    //! \brief Number of packets fully defined
    uint32_t              numPacketsDefined;
    //! \endcond

    //! \brief Tracks list of elements supported by producer.
    //!   It is initialized to default values at creation and filled
    //!   when the information is received from the producer.
    Elements                    producerElements;

    //! \brief Tracks list of elements supported by consumer(s).
    //!   It is initialized to default values at creation and filled
    //!   when the information is received from the consumer(s).
    Elements                    consumerElements;

    //! \brief Tracks list of allocated elements provided by user.
    //!   It is initialized to default values at creation and filled
    //!   by application calls to apiElementAttrSet().
    Elements                    allocatedElements;

    //! \brief Tracks completition of element import.
    //!   It is initialized to false at creation.
    // TODO: Maybe should be in Elements
    std::atomic<bool>           elementImportDone;

    //! \brief Tracks completition of element export.
    //!   It is initialized to false at creation.
    // TODO: Maybe should be in Elements
    std::atomic<bool>           elementExportDone;

    //! \brief Tracks completition of packet export.
    //!   It is initialized to false at creation.
    std::atomic<bool>           packetExportDone;

    //! \brief Tracks completition of packet status import.
    //!   It is initialized to false at creation.
    std::atomic<bool>           packetImportDone;

    //! \cond TIER4_SWUD
    //! \brief Queue of packets available for use by producer. Initialized to
    //! empty queue when Pool object is constructed.
    Packet::PayloadQ             payloadQueue;

    //! \brief Flag indicates that the pool instance is a secondary pool,
    //!   which is not attached to the producer. Initialized to false.
    bool                                      secondary;
    //! \endcond
};

} // namespace LwSciStream

#endif // POOL_H
