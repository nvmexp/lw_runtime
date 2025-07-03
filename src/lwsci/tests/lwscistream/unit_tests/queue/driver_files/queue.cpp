//! \file
//! \brief LwSciStream queue class declaration.
//!
//! \copyright
//! Copyright (c) 2018-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.
#include <cstdint>
#include <array>
#include <unordered_map>
#include <memory>
#include <utility>
#include <atomic>
#include <cmath>
#include <vector>
#include <functional>
#include "covanalysis.h"
#include "sciwrap.h"
#include "lwscistream_types.h"
#include "lwscicommon_os.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "lwscistream_common.h"
#include "safeconnection.h"
#include "packet.h"
#include "enumbitset.h"
#include "block.h"
#include "queue.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//!   - Calls the constructor of the Block base class with BlockType::QUEUE.
//!
//! \implements{19471272}
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A3_1_1), "Bug 2758535")
Queue::Queue(void) noexcept :
LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    Block(BlockType::QUEUE)
{
    // Set up packet description
    Packet::Desc desc { };
    desc.initialLocation = Packet::Location::Upstream;
    desc.fenceProdFillMode = FillMode::Copy;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    pktDescSet(std::move(desc));
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
};

//! <b>Sequence of operations</b>
//!  - Empties the payloadQueue by dequeueing the Packet instances from it,
//!    discarding any unused payloads.
//!
//! \implements{19503318}
LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
Queue::~Queue(void) noexcept
{
    // Remove all packets from the payload queue so their pointers to
    //   each other don't keep them alive.
    PacketPtr oldPacket {};
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A6_5_3), "LwSciStream-ADV-AUTOSARC++14-004")
    do {
        oldPacket = payloadQueue.dequeue();
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    } while (nullptr != oldPacket);
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A6_5_3))
}
LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))

// A stub implementation which always returns LwSciError_AccessDenied,
// as Queue doesn't allow any connections through the public API.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_11), "Bug 2817427")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_13), "Bug 2817427")
LwSciError Queue::getOutputConnectPoint(
    BlockPtr& paramBlock) const noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_13))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_11))
{
    static_cast<void>(paramBlock);
    // Queues don't allow connection through public API
    return LwSciError_AccessDenied;
}

// A stub implementation which always returns LwSciError_AccessDenied,
// as Queue doesn't allow any connections through the public API.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_11), "Bug 2817427")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_13), "Bug 2817427")
LwSciError Queue::getInputConnectPoint(
    BlockPtr& paramBlock) const noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_13))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_11))
{
    static_cast<void>(paramBlock);
    // Queues don't allow connection through public API
    return LwSciError_AccessDenied;
}

//! <b>Sequence of operations</b>
//!   - Disconnects the source block by calling Block::disconnectSrc() interface.
//!   - Disconnects the destination block by calling Block::disconnectDst()
//!     interface.
//!
//! \implements{19471311}
LwSciError Queue::disconnect(void) noexcept {
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectSrc();
    disconnectDst();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    // TODO: Does this function need a return value?
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Disconnects the source block by calling the Block::disconnectSrc()
//!   interface.
//! - If the payloadQueue is empty, triggers the LwSciStreamEventType_Disconnected
//!   event by calling the Block::disconnectEvent() interface and disconnects
//!   the destination block by calling the Block::disconnectDst() interface.
//!   Otherwise, disconnection of the destination block is deferred until after
//!   the contents of the payloadQueue have been drained.
//!
//! \implements{19471314}
void Queue::srcDisconnect(
    uint32_t const srcIndex) noexcept
{
    // Validate block/input state
    //   Index must be valid
    ValidateBits validation {};

    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectSrc();
    // Delay disconnect if there are still frames in the queue.
    if (payloadQueue.empty()) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        disconnectEvent();
        disconnectDst();
        LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    }
}

//! <b>Sequence of operations</b>
//! - Disconnects the destination block by calling the Block::disconnectDst()
//!   interface and triggers the LwSciStreamEventType_Disconnected event by
//!   calling the Block::disconnectEvent() interface.
//! - Disconnects the source block by calling the Block::disconnectSrc()
//!   interface.
//!
//! \implements{19471317}
void Queue::dstDisconnect(
    uint32_t const dstIndex) noexcept
{
    // Validate block/input state
    //   Index must be valid
    ValidateBits validation {};

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

//! <b>Sequence of operations</b>
//! - Under thread protection provided by Block::blkMutexLock():
//! - Dequeues the next available packet instance if any by calling the
//!   Queue::dequeue() interface.
//! - Updates the Location of the packet instance from Location::Queued to
//!   Location::Downstream by calling the Packet::locationUpdate() interface to
//!   disassociate it from the Queue block.
//! - Scope of the thread protection ends.
//! - If the stream has been already disconnected and it is the last packet
//!   instance to be acquired by the consumer, it triggers the
//!   LwSciStreamEventType_Disconnected event by calling the
//!   Block::disconnectEvent() interface and disconnects the destination block
//!   by calling the Block::disconnectDst() interface.
//!
//! \implements{19471368}
PacketPtr Queue::dstDequeuePayload(
    uint32_t const dstIndex) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // TODO: Maybe should have a validation check that the stream _was_ fully
    //       connected, as opposed to _is_ fully connected, since a packet
    //       can be acquired after the upstream had disconnected.

    // Initialize return value to empty pointer
    PacketPtr pkt { };

    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return pkt;
    }

    // Scoped lock while manipulating queue and packet
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
        Lock const blockLock { blkMutexLock() };

        // Remove next payload, if any, from queue
        pkt = dequeue();

        // If found, mark packet as downstream
        if (nullptr != pkt) {
            if (!pkt->locationUpdate(Packet::Location::Queued,
                                     Packet::Location::Downstream)) {
                setErrorEvent(LwSciError_StreamInternalError, true);
                pkt.reset();
            }
        }
    }

    // If upstream path is disconnected, inform downstream block
    // of the disconnect once all frames are acquired.
    if (payloadQueue.empty() && (!connComplete())) {
        disconnectEvent();
        disconnectDst();
    }

    return pkt;

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Packet::handleGet() to look up the packet handle and call
//!   Block::pktFindByHandle() to find the corresponding local packet.
//! - Call Packet::locationUpdate() to move the packet from upstream
//!   to queued.
//! - Call Packet::fenceProdCopy() to copy the incoming fences.
//! - Call Packet::PayloadQ::enqueue() to add to the queue.
//!
//! \implements{19471380}
LwSciError
Queue::enqueue(
    Packet const& newPayload) noexcept
{
    // Look up the packet
    PacketPtr const pkt { pktFindByHandle(newPayload.handleGet(), true) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }

    // Validate and update location
    if (!pkt->locationUpdate(Packet::Location::Upstream,
                             Packet::Location::Queued)) {
        return LwSciError_StreamPacketInaccessible;
    }

    // Copy in the incoming fences
    LwSciError const err { pkt->fenceProdCopy(newPayload) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Enqueue the packet
    payloadQueue.enqueue(pkt);

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Packet::PayloadQ::dequeue() to dequeue a Packet instance from
//!   the queue, if there are any, and returns it.
//!
//! \implements{19471383}
PacketPtr
Queue::dequeue(void) noexcept
{
    return payloadQueue.dequeue();
}

//! <b>Sequence of operations</b>
//! - Call Packet::PayloadQ::requeue() to push a previously dequeued Packet
//!   instance back onto the queue.
void
Queue::requeue(PacketPtr const& pkt) noexcept
{
    payloadQueue.enqueue(pkt);
}

//
// Mailbox specific implementation
//

//! <b>Sequence of operations</b>
//!  - Calls the constructor of the Queue base class.
//!
//! \implements{19471401}
LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
Mailbox::Mailbox(void) noexcept :
    Queue()
{
};
LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))

//! <b>Sequence of operations</b>
//! - Under thread protection provided by Block::blkMutexLock():
//! -- Dequeue any old packet in the queue.
//! -- Call Queue::enqueue() to queue the incoming payload.
//! -- If enqueue fails, call Queue::requeue() to put any dequeued packet
//!    back in the queue and return.
//! - If the old packet instance is present:
//! -- Call Packet::locationUpdate to move it upstream
//! -- Call dstRecvPayload() interface of source block to return packet
//!    upstream for reuse.
//! - Otherwise, call srcRecvPayload() interface of destination block to
//!   inform consumer a payload is available.
void Mailbox::srcRecvPayload(
    uint32_t const srcIndex,
    Packet const& prodPayload) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Scoped lock while manipulating the map and queue
    PacketPtr oldPkt {nullptr};
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
        Lock const blockLock { blkMutexLock() };

        // Retrieve any old packet
        oldPkt = dequeue();

        // Enqueue incoming payload
        LwSciError const err { enqueue(prodPayload) };

        // On failure, need to put back the old packet
        if (LwSciError_Success != err) {
            if (nullptr != oldPkt) {
                requeue(oldPkt);
            }
            setErrorEvent(err, true);
            return;
        }
    }

    // If there is an old packet, it is returned upstream. Otherwise, we
    //   inform downstream that a frame is available.
    if (nullptr == oldPkt) {
        // Send payload downstream to trigger event
        getDst().srcRecvPayload(prodPayload);
    } else {
        // Update location (failure should not be possible)
        if (!oldPkt->locationUpdate(Packet::Location::Queued,
                                    Packet::Location::Upstream)) {
            setErrorEvent(LwSciError_StreamPacketInaccessible);
            return;
        }
        // Clear producer fences
        oldPkt->fenceProdReset();
        // Send upstream
        getSrc().dstRecvPayload(*oldPkt);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

/*
 * Fifo specific implementation
 */

//! <b>Sequence of operations</b>
//!  - Calls the constructor of the Queue base class.
//!
//! \implements{19471392}
LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
Fifo::Fifo(void) noexcept :
    Queue()
{
};
LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))

//! <b>Sequence of operations</b>
//!  - Under thread protection provided by Block::blkMutexLock():
//!  - Enqueues the local packet instance along with the associated fences
//!    by calling the Queue::enqueue() interface.
//!  - Scope of the thread protection ends.
//!  - Informs the consumer block about the availability of a packet for
//!    acquisition by calling the srcRecvPayload() interface of the consumer
//!    block through the destination SafeConnection.
//!
//! \implements{19471395}
void Fifo::srcRecvPayload(
    uint32_t const srcIndex,
    Packet const& prodPayload) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Scoped lock while manipulating the map and queue
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
        Lock const blockLock { blkMutexLock() };

        // Enqueue incoming payload
        LwSciError const err { enqueue(prodPayload) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }
    }

    // Send payload downstream to trigger event
    getDst().srcRecvPayload(prodPayload);

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

} // namespace LwSciStream
