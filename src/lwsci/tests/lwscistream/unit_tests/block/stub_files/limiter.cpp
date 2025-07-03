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

#include "limiter.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

namespace LwSciStream {

//! <b>Sequence of operations</b>
//!   - Calls the constructor of the Block base class with BlockType::LIMITER.
//!
//! \implements{21175200}
LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A3_1_1), "Bug 2758535")
Limiter::Limiter(uint32_t const maxPackets) noexcept :
    Block(BlockType::LIMITER),
    numPacketsLimit(maxPackets),
    numPackets(0U)
{
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
    disconnectSrc();
    disconnectDst();
    // TODO: Does this function need a return value?
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//!  - Calls the srcSendSyncAttr() interface of the destination block through
//!    the destination SafeConnection to forward the LwSciSyncObj waiter
//!    requirements downstream.
//!
//! \implements{21175206}
LWCOV_WHITELIST_BEGIN(LWCOV_AUTOSAR(M2_10_1), "TID-547")
LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
void Limiter::srcSendSyncAttr(
    uint32_t const srcIndex,
    bool const synchronousOnly,
    LwSciWrap::SyncAttr& syncAttr) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    getDst().srcSendSyncAttr(synchronousOnly, syncAttr);
}
LWCOV_WHITELIST_END(LWCOV_AUTOSAR(M2_10_1))

//! <b>Sequence of operations</b>
//!  - Calls the srcSendSyncCount() interface of the destination block through
//!    the destination SafeConnection to forward the count downstream.
//!
//! \implements{21175209}
void Limiter::srcSendSyncCount(
    uint32_t const srcIndex,
    uint32_t const count) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    getDst().srcSendSyncCount(count);
}

//! <b>Sequence of operations</b>
//!  - Calls the srcSendSyncDesc() interface of the destination block through
//!    the destination SafeConnection to forward the Producer's LwSciSyncObj
//!    downstream.
//!
//! \implements{21175212}
LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
void Limiter::srcSendSyncDesc(
    uint32_t const srcIndex,
    uint32_t const syncIndex,
    LwSciWrap::SyncObj& syncObj) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    getDst().srcSendSyncDesc(syncIndex, syncObj);
}

//! <b>Sequence of operations</b>
//!  - Calls the srcSendPacketElementCount() interface of the destination
//!    block through the destination SafeConnection to forward the count
//!    downstream.
//!
//! \implements{21175215}
void Limiter::srcSendPacketElementCount(
    uint32_t const srcIndex,
    uint32_t const count) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    getDst().srcSendPacketElementCount(count);
}

//! <b>Sequence of operations</b>
//!  - Calls the srcSendPacketAttr() interface of the destination block through
//!    the destination SafeConnection to forward the packet element information
//!    downstream.
//!
//! \implements{21175218}
LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
void Limiter::srcSendPacketAttr(
    uint32_t const srcIndex,
    uint32_t const elemIndex,
    uint32_t const elemType,
    LwSciStreamElementMode const elemSyncMode,
    LwSciWrap::BufAttr& elemBufAttr) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    getDst().srcSendPacketAttr(elemIndex, elemType,
                               elemSyncMode, elemBufAttr);
}

//! <b>Sequence of operations</b>
//!  - Calls the srcCreatePacket() interface of the destination block through
//!    the destination SafeConnection to forward the LwSciStreamPacket
//!    downstream.
//!
//! \implements{21175221}
void Limiter::srcCreatePacket(
    uint32_t const srcIndex,
    LwSciStreamPacket const handle) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    getDst().srcCreatePacket(handle);
}

//! <b>Sequence of operations</b>
//!  - Calls the srcInsertBuffer() interface of the destination block through
//!    the destination SafeConnection to forward the LwSciBufObj downstream.
//!
//! \implements{21175224}
LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
void Limiter::srcInsertBuffer(
    uint32_t const srcIndex,
    LwSciStreamPacket const handle,
    uint32_t const elemIndex,
    LwSciWrap::BufObj& elemBufObj) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    getDst().srcInsertBuffer(handle, elemIndex, elemBufObj);
}

//! <b>Sequence of operations</b>
//!  - Calls the srcDeletePacket() interface of the destination block through
//!    the destination SafeConnection to forward the LwSciStreamPacket
//!    downstream.
//!
//! \implements{21175227}
void Limiter::srcDeletePacket(
    uint32_t const srcIndex,
    LwSciStreamPacket const handle) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    getDst().srcDeletePacket(handle);
}

//! <b>Sequence of operations</b>
//! - Increases the number of packets held by downstream.
//! - If the current count is more than the limit, calls the
//!   Limiter::dstReusePacket() interface to return the packet upstream
//!   with empty FenceArray.
//! - If not, calls the srcSendPacket() interface of the destination block
//!   through the destination SafeConnection to forward the packet and the
//!   associated FenceArray downstream.
//!
//! \implements{21175230}
LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
void Limiter::srcSendPacket(
    uint32_t const srcIndex,
    LwSciStreamPacket const handle,
    FenceArray& postfences) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation{};

    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Atomically increments the count of downstream packets.
    // Checks whether the number of packets downstream is at the cap.
    // If not, pass the packet downwards. Otherwise, return it upstream.
    if (numPackets.fetch_add(ONE) >= numPacketsLimit) {
        FenceArray emptyFences{};
        // dstReusePacket() will undo the increment of
        // the downstream-packet count.
        LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        dstReusePacket(0U, handle, emptyFences);
    } else {
        getDst().srcSendPacket(handle, postfences);
    }
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
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    disconnectSrc();
    disconnectEvent();
    disconnectDst();
}

//! <b>Sequence of operations</b>
//!  - Calls the dstSendSyncAttr() interface of the source block through
//!    the source SafeConnection to forward the consumer's LwSciSyncObj
//!    waiter requirements upstream.
//!
//! \implements{21175236}
LWCOV_WHITELIST_BEGIN(LWCOV_AUTOSAR(M2_10_1), "TID-547")
LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
void Limiter::dstSendSyncAttr(
    uint32_t const dstIndex,
    bool const synchronousOnly,
    LwSciWrap::SyncAttr& syncAttr) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation{};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    getSrc().dstSendSyncAttr(synchronousOnly, syncAttr);
}
LWCOV_WHITELIST_END(LWCOV_AUTOSAR(M2_10_1))

//! <b>Sequence of operations</b>
//!  - Calls the dstSendSyncCount() interface of the source block through
//!    the source SafeConnection to forward the consumer's LwSciSyncObj(s)
//!    count upstream.
//!
//! \implements{21175239}
void Limiter::dstSendSyncCount(
    uint32_t const dstIndex,
    uint32_t const count) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation{};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    getSrc().dstSendSyncCount(count);
}

//! <b>Sequence of operations</b>
//!  - Calls the dstSendSyncDesc() interface of the source block through
//!    the source SafeConnection to forward the consumer's LwSciSyncObj
//!    upstream.
//!
//! \implements{21175242}
LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
void Limiter::dstSendSyncDesc(
    uint32_t const dstIndex,
    uint32_t const syncIndex,
    LwSciWrap::SyncObj& syncObj) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation{};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    getSrc().dstSendSyncDesc(syncIndex, syncObj);
}

//! <b>Sequence of operations</b>
//!  - Calls the dstSendPacketElementCount() interface of the source block
//!    through the source SafeConnection to forward the consumer's supported
//!    packet element count upstream.
//!
//! \implements{21175245}
void Limiter::dstSendPacketElementCount(
    uint32_t const dstIndex,
    uint32_t const count) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation{};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    getSrc().dstSendPacketElementCount(count);
}

//! <b>Sequence of operations</b>
//!  - Calls the dstSendPacketAttr() interface of the source block through
//!    the source SafeConnection to forward the consumer's supported packet
//!    element information upstream.
//!
//! \implements{21175248}
LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
void Limiter::dstSendPacketAttr(
    uint32_t const dstIndex,
    uint32_t const elemIndex,
    uint32_t const elemType,
    LwSciStreamElementMode const elemSyncMode,
    LwSciWrap::BufAttr& elemBufAttr) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation{};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    getSrc().dstSendPacketAttr(elemIndex, elemType,
                               elemSyncMode, elemBufAttr);
}

//! <b>Sequence of operations</b>
//!  - Calls the dstSendPacketStatus() interface of the source block through
//!    the source SafeConnection to forward the consumer's packet element
//!    acceptance status upstream.
//!
//! \implements{21175251}
void Limiter::dstSendPacketStatus(
    uint32_t const dstIndex,
    LwSciStreamPacket const handle,
    LwSciError const packetStatus) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    getSrc().dstSendPacketStatus(handle, packetStatus);
}

//! <b>Sequence of operations</b>
//!  - Calls the dstSendElementStatus() interface of the source block through
//!    the source SafeConnection to forward the consumer's element acceptance
//!    status upstream.
//!
//! \implements{21175254}
void Limiter::dstSendElementStatus(
    uint32_t const dstIndex,
    LwSciStreamPacket const handle,
    uint32_t const elemIndex,
    LwSciError const elemStatus) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    getSrc().dstSendElementStatus(handle, elemIndex, elemStatus);
}

//! <b>Sequence of operations</b>
//!   - Makes sure that stream disconnect is not yet done by calling
//!     Block::connComplete() interface.
//!   - Decreases the current number of packets held by downstream blocks.
//!   - Calls the dstReusePacket() interface of the source block through the
//!     source SafeConnection to forward the LwSciStreamPacket and FenceArray
//!     upstream.
//!
//! \implements{21175257}
LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
void Limiter::dstReusePacket(
    uint32_t const dstIndex,
    LwSciStreamPacket const handle,
    FenceArray& postfences) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::DstIndex);
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
    if (numPackets.fetch_sub(ONE) == 0U) {
        LWCOV_WHITELIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        setErrorEvent(LwSciError_StreamInternalError);
    }

    // Send reuse message upstream.
    getSrc().dstReusePacket(handle, postfences);
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
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    disconnectDst();
    disconnectEvent();
    disconnectSrc();
}

} // namespace LwSciStream
