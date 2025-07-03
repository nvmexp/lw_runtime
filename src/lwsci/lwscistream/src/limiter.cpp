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
#include <cstdint>
#include <iostream>
#include <array>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <vector>
#include <functional>
#include "covanalysis.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "lwscistream_common.h"
#include "sciwrap.h"
#include "lwscistream_types.h"
#include "lwscicommon_os.h"
#include "safeconnection.h"
#include "enumbitset.h"
#include "block.h"
#include "limiter.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//!   - Calls the constructor of the Block base class with BlockType::LIMITER.
//!
//! \implements{21175200}
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A3_1_1), "Bug 2758535")
Limiter::Limiter(uint32_t const maxPackets) noexcept :
LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    Block(BlockType::LIMITER),
    numPacketsLimit(maxPackets),
    numPackets(0U)
{
    // Set up packet description
    // Limiter does not need to store any information about the packets,
    //   but must send a packet without fences back upstream when one
    //   is rejected.
    Packet::Desc desc { };
    pktDescSet(std::move(desc));
};

//! <b>Sequence of operations</b>
//!   - Disconnects the source block by calling the Block::disconnectSrc()
//!     interface.
//!   - Disconnects the destination block by calling the Block::disconnectDst()
//!     interface.
//!
//! \implements{21175203}
LwSciError Limiter::disconnect(void) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectSrc();
    disconnectDst();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    // TODO: Does this function need a return value?
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Increases the number of packets held by downstream.
//! - If the current count is within the limit, call srcRecvPayload()
//!   interface of destination block to send the payload downstream.
//! - Otherwise:
//! -- Call Packet::handleGet() to retrieve the packet handle, then call
//!    Block::pktFindByHandle() to look up the local packet.
//! -- Call dstRecvPayload() interface of source block to send the packet
//!    back upstream.
//! -- Decrement the count.
//!
//! \implements{21175230}
void Limiter::srcRecvPayload(
    uint32_t const srcIndex,
    Packet const& prodPayload) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation{};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Atomically increments the count of downstream packets.
    // Checks whether the number of packets downstream is at the cap.
    // If not, pass the packet downwards. Otherwise, return it upstream.
    if (numPackets.fetch_add(ONE) < numPacketsLimit) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getDst().srcRecvPayload(prodPayload);
    } else {
        // Note: We need to send the local packet upstream to ensure what
        //       gets sent won't have any fences. That's the only reason
        //       this block maintains a packet map.
        PacketPtr const pkt { pktFindByHandle(prodPayload.handleGet()) };
        if (nullptr == pkt) {
            setErrorEvent(LwSciError_StreamBadPacket);
        } else {
            LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
            getSrc().dstRecvPayload(*pkt);
            if (0U == numPackets.fetch_sub(ONE)) {
                setErrorEvent(LwSciError_StreamInternalError);
            }
        }
    }
}

//! <b>Sequence of operations</b>
//! - Decrease the number of packets held by downstream.
//! - Call dstRecvPayload() interface of source block to send the packet
//!   back upstream.
//!
//! \implements{21175257}
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
void Limiter::dstRecvPayload(
    uint32_t const dstIndex,
    Packet const& consPayload) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // If disconnect has oclwrred, return but don't report an error
    // TODO: Need to distinguish various disconnect cases and handle
    //       in the validate function above.
    if (!connComplete()) {
        return;
    }

    // Decrease the current count before passing the packet upstream.
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (0U == numPackets.fetch_sub(ONE)) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        setErrorEvent(LwSciError_StreamInternalError);
        return;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // Send payload upstream
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    getSrc().dstRecvPayload(consPayload);
}

//! <b>Sequence of operations</b>
//! - Disconnects the source block by calling the Block::disconnectSrc()
//!   interface.
//! - Triggers the LwSciStreamEventType_Disconnected event by calling the
//!   Block::disconnectEvent() interface.
//! - Disconnects the destination block by calling the Block::disconnectDst()
//!   interface.
//!
//! \implements{21175233}
void Limiter::srcDisconnect(
    uint32_t const srcIndex) noexcept
{
    // Validate block/input state
    //   Index must be valid
    ValidateBits validation{};

    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectSrc();
    disconnectEvent();
    disconnectDst();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
}

//! <b>Sequence of operations</b>
//! - Disconnects the destination block by calling the Block::disconnectDst()
//!   interface.
//! - Triggers the LwSciStreamEventType_Disconnected event by calling the
//!   Block::disconnectEvent() interface.
//! - Disconnects the source block by calling the Block::disconnectSrc()
//!   interface.
//!
//! \implements{21175260}
void Limiter::dstDisconnect(
    uint32_t const dstIndex) noexcept
{
    // Validate block/input state
    //   Index must be valid
    ValidateBits validation{};

    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectDst();
    disconnectEvent();
    disconnectSrc();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
}

} // namespace LwSciStream
