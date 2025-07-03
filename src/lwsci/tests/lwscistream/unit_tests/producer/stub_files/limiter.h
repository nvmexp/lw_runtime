//! \file
//! \brief LwSciStream Limiter class declaration.
//!
//! \copyright
//! Copyright (c) 2020-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef LIMITER_H
#define LIMITER_H
#include <cstdint>
#include <utility>
#include <atomic>
#include "covanalysis.h"
#include "lwscistream_common.h"
#include "block.h"

namespace LwSciStream {

//! \brief Limiter Block places a cap on the number of packets which it will
//!   allow to be sent downstream of it. If a new packet arrives, and the
//!   current number of downstream packets is at the cap, the packet will
//!   immediately be returned upstream without the consumer ever seeing it.
//!
//! - It inherits from the Block class which provides common functionalities
//!   for all Blocks.
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
    explicit Limiter(uint32_t const maxPackets) noexcept;

    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    ~Limiter(void) noexcept final = default;

    // Disable copy/move constructors and assignment operators.
    Limiter(const Limiter&) noexcept               = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Limiter(Limiter&&) noexcept                    = delete;
    Limiter& operator=(const Limiter &) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
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

    //! \brief Forwards the @a prodPayload downstream if the number of
    //!   payloads held by downstream blocks is less than the limit and
    //!   updates the tracked count of downstream payloads, otherwise
    //!   returns the payload upstream for reuse.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPayload
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175341}
    //! \endif
    void srcRecvPayload(
        uint32_t const srcIndex,
        Packet const& prodPayload) noexcept final;

    //! \brief Decrements the number of downstream payloads and forwards
    //!   the @a consPayload upstream.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPayload
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{21175368}
    //! \endif
    void dstRecvPayload(
        uint32_t const dstIndex,
        Packet const& consPayload) noexcept final;

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
