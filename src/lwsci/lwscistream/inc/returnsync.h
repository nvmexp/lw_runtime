//! \file
//! \brief LwSciStream ReturnSync class declaration.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef RETURN_SYNC_H
#define RETURN_SYNC_H
#include <cstdint>
#include <utility>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "covanalysis.h"
#include "lwscistream_common.h"
#include "block.h"
#include "packet.h"
#include "trackarray.h"

namespace LwSciStream {

//! \brief ReturnSync Block does the waiting for fences for each received
//!   packet from downstream before sending it upstream. If fences for the
//!   receieved packet are already expired/empty then it returns the packet
//!   upstream immediately. If not, it places the packet in the queue then
//!   trigger a thread spawned by this block to manage the fence waiting
//!   before sending the packet upstream.
//!
//! - It inherits from the Block class which provides common functionalities
//!   for all Blocks.
//!
//! \if TIER4_SWAD
//! \implements{}
//! \endif
//!
//! \if TIER4_SWUD
//! \implements{}
//! \endif
class ReturnSync :
    public Block
{
public:
    //! \brief Constructs an instance of the ReturnSync class and initializes
    //!   the ReturnSync specific data members.
    //!
    //! \param [in] syncModule: Instance of LwSciSyncModule.
    //!
    //! \if TIER4_SWAD
    //! \implements{}
    //! \endif
    explicit ReturnSync(LwSciSyncModule const syncModule) noexcept;

    //! \brief Destroys the ReturnSync class instance.
    //!
    //! \if TIER4_SWAD
    //! \implements{}
    //! \endif
    ~ReturnSync(void) noexcept override;

    // Disable copy/move constructors and assignment operators.
    ReturnSync(const ReturnSync&) noexcept               = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    ReturnSync(ReturnSync&&) noexcept                    = delete;
    ReturnSync& operator=(const ReturnSync &) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    ReturnSync& operator=(ReturnSync &&) & noexcept      = delete;

    //! \brief Discards producer's LwSciSynObj waiter requirements and sends
    //!  attributes for CPU waiting downstream to consumer block.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvSyncWaiter
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_AlreadyDone: Sync waiter info was already processed.
    //! - Any error from LwSciSyncAttrListCreate(),
    //!   LwSciSyncAttrListSetAttrs() and LwSciSyncCpuWaitContextAlloc().
    //! - Any error returned by Waiters::sizeInit(), Waiters::attrSet(),
    //!   and Waiters::doneSet().
    //!
    //! \if TIER4_SWAD
    //! \implements{}
    //! \endif
    void srcRecvSyncWaiter(
        uint32_t const srcIndex,
        Waiters const& syncWaiter) noexcept final;

    //! \brief Discards the LwSciSync signal information from the consumer(s)
    //!   and sends an empty list of sync objects upstream.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSyncSignal
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Signals::sizeInit() or Signals::doneSet().
    void dstRecvSyncSignal(
        uint32_t const dstIndex,
        Signals const& syncSignal) noexcept final;

    //! \brief Receives payload released by consumer block(s) and holds it
    //!   until all fences have expired, then forwards it upstream.
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
    //! \implements{}
    //! \endif
    void dstRecvPayload(
        uint32_t const dstIndex,
        Packet const& consPayload) noexcept final;

    //! \brief Disconnects the source and destination blocks.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::disconnect
    //!
    //! \if TIER4_SWAD
    //! \implements{}
    //! \endif
    LwSciError disconnect(void) noexcept final;

    //! \brief Disconnects the source and destination blocks and prepares
    //!   LwSciStreamEventType_Disconnected event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcDisconnect
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{}
    //! \endif
    void srcDisconnect(
        uint32_t const srcIndex) noexcept final;

    //! \brief Disconnects the source and destination blocks and prepares
    //!   LwSciStreamEventType_Disconnected event.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstDisconnect
    //!
    //! \if TIER4_SWAD
    //! \implements{}
    //! \endif
    void dstDisconnect(
        uint32_t const dstIndex) noexcept final;

private:

    // I/O Thread loop for handling fence wait operations.
    void waitForFenceThreadFunc(void) noexcept;

private:
    //! \brief Parent LwSciSyncModule that is used during fence wait
    //!   operations.
    //!   Initialized from the provided LwSciSyncModule when
    //!   a new ReturnSync instance is created.
    LwSciSyncModule         syncModule;

    //! \brief Flag indicating whether sync waiter info has been set up.
    //!   Initialized to false at creation.
    std::atomic<bool>       waiterDone;

    //! \brief Tracks packets for which fences are unexpired.
    //!   Initialized to an empty queue when a new ReturnSync instance
    //!   is created.
    Packet::PayloadQ        fenceWaitQueue;

    //! \brief CPU context used for doing CPU waits.
    //!   Initialized in ReturnSync::srcSendSyncAttr().
    LwSciSyncCpuWaitContext waitContext;

    //! \brief Conditional variable to wait for packet with unexpired
    //!   fences in queue.
    std::condition_variable packetCond;

    //! \brief Representation of I/O thread used for managing the fence waiting.
    //!   This thread launches when a new ReturnSync instance is created
    //!   and exelwtes waitForFenceThreadFunc().
    std::thread  dispatchThread;

    //! \brief Flag to control the dispatchThread() when block is destroyed.
    //!   Initialized to false when a new ReturnSync instance
    //!   is created.
    bool  teardown;
};

} // namespace LwSciStream
#endif // RETURN_SYNC_H
