//! \file
//! \brief LwSciStream Limiter class declaration.
//!
//! \copyright
//! Copyright (c) 2020 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef LIMITER_H
#define LIMITER_H

#include "lwscistream_common.h"
#include "block.h"
#include <atomic>

namespace LwSciStream {

//! \brief Limiter Block places a cap on the number of packets which it will
//!   allow to be sent downstream of it. If a new packet arrives, and the
//!   current number of downstream packets is at the cap, the packet will
//!   immediately be returned upstream without the consumer ever seeing it.
//!
//! - It inherits from the Block class which provides common functionalities
//!   for all Blocks.
//! - It does not need to track any of the stream setup details like the
//!   LwSciSync / LwSciBuf information.
//! - Most of the APIBlockInterface interfaces can fall back on the default
//!   implementation from the base Block except the disconnect() interface.
//! - Most of the SrcBlockInterface interfaces are implemented as pass-throughs
//!   except the dstReusePacket() interface.
//! - Most of the DstBlockInterface interfaces are implemented as pass-throughs
//!   except the srcSendPacket() interface.
//!
//! \if TIER4_SWAD
//! \implements{21138840}
//! \endif
//!
//! \if TIER4_SWUD
//! \implements{21138780}
//! \endif
class Limiter :
    public Block
{
public:
    //! \brief Constructs an instance of the Limiter class and initializes
    //!   the Limiter specific data members.
    //!
    //! \param [in] maxPackets: Max number of packets allowed to be sent
    //!   downstream.
    //!
    //! \if TIER4_SWAD
    //! \implements{21175311}
    //! \endif
    Limiter(uint32_t const maxPackets) noexcept;

    ~Limiter(void) noexcept final = default;

    // Disable copy/move constructors and assignment operators.
    Limiter(const Limiter&) noexcept               = delete;
    LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Limiter(Limiter&&) noexcept                    = delete;
    Limiter& operator=(const Limiter &) & noexcept = delete;
    LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Limiter& operator=(Limiter &&) & noexcept      = delete;

    // API Block Interface

    //! \brief Disconnects the source and destination blocks.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::disconnect
    //!
    //! \if TIER4_SWAD
    //! \implements{21175314}
    //! \endif
    LwSciError disconnect(void) noexcept final;

    // Dst Block Interface

    //! \brief Forwards the producer's LwSciSynObj waiter requirements
    //!   downstream to consumer block.
    //!
    //! \note This interface assumes that the @a synchronousOnly and
    //!  @a syncAttr arguments are already validated by the public-facing
    //!  interfaces, so these are not validated again by this interface.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcSendSyncAttr
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175317}
    //! \endif
    LWCOV_WHITELIST_BEGIN(LWCOV_AUTOSAR(M2_10_1), "TID-547")
    LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    void srcSendSyncAttr(
        uint32_t const srcIndex,
        bool const synchronousOnly,
        LwSciWrap::SyncAttr& syncAttr) noexcept final;
    LWCOV_WHITELIST_END(LWCOV_AUTOSAR(M2_10_1))

    //! \brief Forwards the producer's count of LwSciSyncObj(s) used for
    //!   signaling downstream to consumer block.
    //!
    //! \note This interface assumes that the @a count argument is already
    //!  validated by the public facing interfaces, so this is not validated
    //!  again by this interface.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcSendSyncCount
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175320}
    //! \endif
    void srcSendSyncCount(
        uint32_t const srcIndex,
        uint32_t const count) noexcept final;

    //! \brief Forwards one of the Producer's LwSciSyncObj(s) contained by the
    //!   @a syncObj for the given @a syncIndex downstream to consumer block.
    //!
    //! \note This interface assumes that the @a syncIndex and @a syncObj
    //!  arguments are already validated by the public facing interfaces, so
    //!  these are not validated again by this interface.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcSendSyncDesc
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175323}
    //! \endif
    LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    void srcSendSyncDesc(
        uint32_t const srcIndex,
        uint32_t const syncIndex,
        LwSciWrap::SyncObj& syncObj) noexcept final;

    //! \brief Forwards the consolidated packet element count downstream to
    //!   consumer block.
    //!
    //! \note This interface assumes that the @a count argument is already
    //!  validated by the public facing interfaces, so this is not validated
    //!  again by this interface.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcSendPacketElementCount
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175326}
    //! \endif
    void srcSendPacketElementCount(
        uint32_t const srcIndex,
        uint32_t const count) noexcept final;

    //! \brief Forwards the consolidated packet element information for the
    //!   given @a elemIndex downstream to consumer block.
    //!
    //! \note This interface assumes that the @a elemIndex, @a elemType,
    //!  @a elemSyncMode and @a elemBufAttr arguments are already validated
    //!  by the public facing interfaces, so these are not validated again
    //!  by this interface.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcSendPacketAttr
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175329}
    //! \endif
    LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    void srcSendPacketAttr(
        uint32_t const srcIndex,
        uint32_t const elemIndex,
        uint32_t const elemType,
        LwSciStreamElementMode const elemSyncMode,
        LwSciWrap::BufAttr& elemBufAttr) noexcept final;

    //! \brief Forwards the @a handle to the new packet downstream to consumer
    //!   block.
    //!
    //! \note This interface assumes that the @a handle argument is already
    //!  validated by the public-facing interfaces, so this is not validated
    //!  again by this interface.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcCreatePacket
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175332}
    //! \endif
    void srcCreatePacket(
        uint32_t const srcIndex,
        LwSciStreamPacket const handle) noexcept final;

    //! \brief Forwards a LwSciBufObj contained by the @a elemBufObj for a
    //!   packet element referenced by the given @a elemIndex of the packet
    //!   referenced by the given @a handle downstream to consumer block.
    //!
    //! \note This interface assumes that the input arguments are already
    //!  validated by the public facing interfaces, so these are not validated
    //!  again by this interface.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcInsertBuffer
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175335}
    //! \endif
    LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    void srcInsertBuffer(
        uint32_t const srcIndex,
        LwSciStreamPacket const handle,
        uint32_t const elemIndex,
        LwSciWrap::BufObj& elemBufObj) noexcept final;

    //! \brief Forwards the @a handle to the deleted packet downstream to
    //!   consumer block.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcDeletePacket
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175338}
    //! \endif
    void srcDeletePacket(
        uint32_t const srcIndex,
        LwSciStreamPacket const handle) noexcept final;

    //! \brief Forwards the packet referenced by the given @a handle and the
    //!   associated FenceArray (@a postfences) downstream if the number of
    //!   packets held by downstream blocks is less than the limit and updates
    //!   the tracked count of downstream packets, otherwise returns the packet
    //!   upstream for reuse.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcSendPacket
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175341}
    //! \endif
    void srcSendPacket(
        uint32_t const srcIndex,
        LwSciStreamPacket const handle,
        FenceArray& postfences) noexcept final;

    //! \brief Disconnects the source and destination blocks and prepares
    //!   LwSciStreamEventType_Disconnected event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcDisconnect
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175344}
    //! \endif
    void srcDisconnect(
        uint32_t const srcIndex) noexcept final;

    // Src Block Interface

    //! \brief Forwards the consumer's LwSciSyncObj waiter requirements
    //!   upstream to producer block.
    //!
    //! \note This interface assumes that the @a synchronousOnly and
    //!  @a syncAttr arguments are already validated by the public facing
    //!  interfaces, so these are not validated again by this interface.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstSendSyncAttr
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175347}
    //! \endif
    LWCOV_WHITELIST_BEGIN(LWCOV_AUTOSAR(M2_10_1), "TID-547")
    LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    void dstSendSyncAttr(
        uint32_t const dstIndex,
        bool const synchronousOnly,
        LwSciWrap::SyncAttr& syncAttr) noexcept final;
    LWCOV_WHITELIST_END(LWCOV_AUTOSAR(M2_10_1))

    //! \brief Forwards the consumer's count of LwSciSyncObj(s) used for
    //!   signaling upstream to producer block.
    //!
    //! \note This interface assumes that the @a count argument is already
    //!  validated by the public facing interfaces, so this is not validated
    //!  again by this interface.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstSendSyncCount
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175350}
    //! \endif
    void dstSendSyncCount(
        uint32_t const dstIndex,
        uint32_t const count) noexcept final;

    //! \brief Forwards one of the Consumer's LwSciSyncObj(s) contained by the
    //!   @a syncObj for the given @a syncIndex upstream to producer block.
    //!
    //! \note This interface assumes that the @a syncIndex and @a syncObj
    //!  arguments are already validated by the public facing interfaces, so
    //!  these are not validated again by this interface.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstSendSyncDesc
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175353}
    //! \endif
    LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    void dstSendSyncDesc(
        uint32_t const dstIndex,
        uint32_t const syncIndex,
        LwSciWrap::SyncObj& syncObj) noexcept final;

    //! \brief Forwards the consumer's supported packet element count upstream
    //!   to pool block.
    //!
    //! \note This interface assumes that the @a count argument is already
    //!  validated by the public facing interfaces, so this is not validated
    //!  again by this interface.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstSendPacketElementCount
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175356}
    //! \endif
    void dstSendPacketElementCount(
        uint32_t const dstIndex,
        uint32_t const count) noexcept final;

    //! \brief Forwards the consumer's supported packet element information
    //!   for the given @a elemIndex upstream to pool block.
    //!
    //! \note This interface assumes that the @a elemIndex, @a elemType,
    //!  @a elemSyncMode and @a elemBufAttr arguments are already validated by
    //!  the public facing interfaces, so these are not validated again by
    //!  this interface.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstSendPacketAttr
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175359}
    //! \endif
    LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    void dstSendPacketAttr(
        uint32_t const dstIndex,
        uint32_t const elemIndex,
        uint32_t const elemType,
        LwSciStreamElementMode const elemSyncMode,
        LwSciWrap::BufAttr& elemBufAttr) noexcept final;

    //! \brief Forwards the consumer's acceptance status (@a packetStatus) for
    //!   the packet referenced by the given @a handle upstream to pool block.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstSendPacketStatus
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175362}
    //! \endif
    void dstSendPacketStatus(
        uint32_t const dstIndex,
        LwSciStreamPacket const handle,
        LwSciError const packetStatus) noexcept final;

    //! \brief Forwards consumer's acceptance status (@a elemStatus) for the
    //!   element referenced by the given @a elemIndex of the packet referenced
    //!   by the given @a handle upstream to pool block.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstSendElementStatus
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175365}
    //! \endif
    void dstSendElementStatus(
        uint32_t const dstIndex,
        LwSciStreamPacket const handle,
        uint32_t const elemIndex,
        LwSciError const elemStatus) noexcept final;

    //! \brief Forwards the released packet referenced by the given @a handle
    //!   and the associated FenceArray (@a postfences) upstream to pool block.
    //!   Also updates the tracked count of downstream packets.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstReusePacket
    //!
    //! \return void
    //!  - Panics if the tracked count of downstream packets is zero.
    //!
    //! \if TIER4_SWAD
    //! \implements{21175368}
    //! \endif
    LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    void dstReusePacket(
        uint32_t const dstIndex,
        LwSciStreamPacket const handle,
        FenceArray& postfences) noexcept final;

    //! \brief Disconnects the source and destination blocks and prepares
    //!   LwSciStreamEventType_Disconnected event.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstDisconnect
    //!
    //! \if TIER4_SWAD
    //! \implements{21175371}
    //! \endif
    void dstDisconnect(
        uint32_t const dstIndex) noexcept final;


private:
    //! \cond TIER4_SWAD
    //! \brief Max number of packets allowed to be sent downstream.
    //!   Initialized when a Limiter instance is created.
    uint32_t const              numPacketsLimit;
    //! \endcond

    //! \cond TIER4_SWUD
    //! \brief Current number of packets held by downstream blocks.
    //!   Initialized to zero when a Limiter instance is created.
    std::atomic<uint32_t>       numPackets;
    //! \endcond
};

} // namespace LwSciStream
#endif // LIMITER_H
