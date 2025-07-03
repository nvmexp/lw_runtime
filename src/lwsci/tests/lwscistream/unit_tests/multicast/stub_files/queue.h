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

#ifndef QUEUE_H
#define QUEUE_H
#include <cstdint>
#include <utility>
#include <memory>
#include "covanalysis.h"
#include "lwscistream_common.h"
#include "block.h"
#include "elements.h"
#include "packet.h"

namespace LwSciStream {

//! \brief Constant to represent Null packet handle.
constexpr LwSciStreamPacket nullPacketHandle
                    {static_cast<LwSciStreamPacket>(0)};

//! \brief Queue class implements storage for buffering access to stream
//!   packets waiting to be processed, and some other common basic
//!   functionalities shared by Fifo and Mailbox queues.
//!
//! - It inherits from the Block class which provides common functionalities
//!   for all Blocks.
//! - It overrides the supported API functions from APIBlockInterface, which
//!   are called by LwSciStream public APIs.
//! - It overrides the SrcBlockInterface functions, which are called by the
//!   destination block.
//! - It overrides the DstBlockInterface functions, which are called
//!   by the source block.
//! - It is inherited by the Fifo and MailBox Queue classes.
//!
//! \if TIER4_SWAD
//! \implements{18790101}
//! \endif
class Queue :
    public Block
{
public:
    //! \brief Constructs an instance of the Queue class and initializes
    //!  the Queue specific data members.
    //!
    //! \if TIER4_SWAD
    //! \implements{19631841}
    //! \endif
    Queue(void) noexcept;

    //! \brief Destroys the Queue class instance.
    //!
    //! \if TIER4_SWAD
    //! \implements{19631847}
    //! \endif
    ~Queue(void) noexcept override;

    // Disable copy/move constructors and assignment operators.
    Queue(const Queue&) noexcept               = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Queue(Queue&&) noexcept                    = delete;
    Queue& operator=(const Queue &) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Queue& operator=(Queue &&) & noexcept      = delete;

    // api Block Interface
    //! \brief A stub implementation which always returns LwSciError_AccessDenied,
    //!  as Queue doesn't allow any connections through the public API.
    //!
    //! \param [out] paramBlock: Unused.
    //!
    //! \return LwSciError, Always LwSciError_AccessDenied
    //!
    //! \implements{19631844}
    LwSciError getOutputConnectPoint(
        BlockPtr& paramBlock) const noexcept final;

    //! \brief A stub implementation which always returns LwSciError_AccessDenied,
    //!  as Queue doesn't allow any connections through the public API.
    //!
    //! \param [out] paramBlock: Unused.
    //!
    //! \return LwSciError, Always LwSciError_AccessDenied
    //!
    //! \implements{19631853}
    LwSciError getInputConnectPoint(
        BlockPtr& paramBlock) const noexcept final;

    //! \brief Disconnects the source and destination blocks of the queue.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::disconnect
    //!
    //! \if TIER4_SWAD
    //! \implements{19631856}
    //! \endif
    LwSciError disconnect(void) noexcept final;

    //! \brief Disconnects the source and destination blocks and prepares
    //!  LwSciStreamEventType_Disconnected event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcDisconnect
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19631859}
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
    //! \implements{19631862}
    //! \endif
    void dstDisconnect(
        uint32_t const dstIndex) noexcept final;

    //! \brief Dequeues and returns a packet from the queue, if any.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstDequeuePacket
    //!
    //! \if TIER4_SWAD
    //! \implements{19632099}
    //! \endif
    PacketPtr dstDequeuePayload(
        uint32_t const dstIndex) noexcept final;

protected:
    //! \brief Enqueues an incoming payload in the payloadQueue.
    //!
    //! \param [in] newPayload: Pointer to incoming payload.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - LwSciError_StreamBadPacket: The packet handle is unknown.
    //! - LwSciError_StreamPacketInaccessible: The packet was not upstream.
    //! - Any error returned by Packet::fenceProdCopy().
    //!
    //! \if TIER4_SWAD
    //! \implements{19632105}
    //! \endif
    LwSciError enqueue(
        Packet const& newPayload) noexcept;

    //! \brief Dequeues a packet instance from payloadQueue.
    //!
    //! \return PacketPtr, Pointer to packet instance
    //!
    //! \if TIER4_SWAD
    //! \implements{19632108}
    //! \endif
    PacketPtr dequeue(void) noexcept;

    //! \brief Requeues a dequeied packet instance back into the payloadQueue.
    //!
    //! \param [in] pkt: Packet instance
    //!
    //! \return void
    void requeue(PacketPtr const& pkt) noexcept;

private:
    //! \cond TIER4_SWAD
    //! \brief Queue of packets available for use.
    //!  Initially this queue is empty. During streaming, packets will be
    //!  enqueued and dequeued in this Packet::PayloadQ instance by calling the
    //!  Queue::enqueue() and Queue::dequeue() interfaces.
    Packet::PayloadQ payloadQueue;
    //! \endcond
};

//! \brief MailBox Queue is a Queue Block used when only the most recently
//!   received data packet is necessary for processing. As such, only the most
//!   recent packet is stored in the queue. If a new packet is received when
//!   there is one waiting in the Mailbox, the previously buffered item will
//!   be skipped and its payload will be returned for reuse.
//!
//! - It inherits from the Queue class which provides basic functionalities
//!   for a Queue Block.
//! - It overrides LwSciStream::DstBockInterface::srcRecvPayload interface
//!   as required by the Mailbox functionality.
//!
//! \implements{18790104}
class Mailbox :
    public Queue
{
public:
    //! \brief Constructs an instance of the Mailbox class and initializes all
    //!  data fields.
    //!
    //! \if TIER4_SWAD
    //! \implements{19631832}
    //! \endif
    Mailbox(void) noexcept;
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    ~Mailbox(void) noexcept final = default;

    // Disable copy/move constructors and assignment operators.
    Mailbox(const Mailbox&) noexcept               = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Mailbox(Mailbox&&) noexcept                    = delete;
    Mailbox& operator=(const Mailbox &) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Mailbox& operator=(Mailbox &&) & noexcept      = delete;

    //! \brief Receives payload from producer, stores it, and returns any
    //!   payload already in the Mailbox upstream for reuse. Informs
    //!   consumer about available payload if there was not one pending
    //!   already.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPayload
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: Packet provided is not valid.
    //! - LwSciError_StreamPacketInaccessible: Packet provided is not
    //!   upstream, or old packet is not queued.
    //! - Any error returned by Packet::fenceProdCopy().
    void srcRecvPayload(
        uint32_t const srcIndex,
        Packet const& prodPayload) noexcept final;
};

//! \brief Fifo Queue is a Queue Block used when all Packets must be processed
//!   in the received order. Packets will be stored in the Fifo Queue until
//!   processed and dequeued.
//!
//! - It inherits from the Queue class which provides basic functionalities
//!   for a Queue Block.
//! - It overrides LwSciStream::DstBockInterface::srcRecvPayload interface
//!   as required by the Fifo functionality.
//!
//! \implements{18790110}
class Fifo :
    public Queue
{
public:
    //! \brief Constructs an instance of Fifo class and initializes all data
    //!  fields.
    //!
    //! \if TIER4_SWAD
    //! \implements{19631835}
    //! \endif
    Fifo(void) noexcept;
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    ~Fifo(void) noexcept final = default;

    // Disable copy/movve constructors and assignment operators.
    Fifo(const Fifo&) noexcept               = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Fifo(Fifo&&) noexcept                    = delete;
    Fifo& operator=(const Fifo &) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Fifo& operator=(Fifo &&) & noexcept      = delete;

    //! \brief Receives payload from producer, appends it to the queue,
    //!   and informs consumer about available payload.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPayload
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: Packet provided is not valid.
    //! - LwSciError_StreamPacketInaccessible: Packet provided is not
    //!   upstream.
    //! - Any error returned by Packet::fenceProdCopy().
    void srcRecvPayload(
        uint32_t const srcIndex,
        Packet const& prodPayload) noexcept final;
};

} // namespace LwSciStream
#endif // QUEUE_H
