//! \file
//! \brief LwSciStream PresentSync class declaration.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef PRESENT_SYNC_H
#define PRESENT_SYNC_H
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

//! \brief PresentSync Block does the waiting for fences for each received
//!   packet from upstream before sending it downstream. Block spawns a
//!   thread which waits for fences in the order of the packets received
//!   and send them downstream once fences are reached.
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
class PresentSync :
    public Block
{
public:
    //! \brief Constructs an instance of the PresentSync class and initializes
    //!   the PresentSync specific data members.
    //!
    //! \param [in] syncModule: Instance of LwSciSyncModule.
    //!
    //! \if TIER4_SWAD
    //! \implements{}
    //! \endif
    explicit PresentSync(LwSciSyncModule const syncModule) noexcept;

    //! \brief Destroys the PresentSync class instance.
    //!
    //! \if TIER4_SWAD
    //! \implements{}
    //! \endif
    ~PresentSync(void) noexcept override;

    // Disable copy/move constructors and assignment operators.
    PresentSync(const PresentSync&) noexcept               = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    PresentSync(PresentSync&&) noexcept                    = delete;
    PresentSync& operator=(const PresentSync &) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    PresentSync& operator=(PresentSync &&) & noexcept      = delete;

    //! \brief Disconnects the source and destination blocks.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::disconnect
    //!
    //! \if TIER4_SWAD
    //! \implements{}
    //! \endif
    LwSciError disconnect(void) noexcept final;

    //! \brief Discards consumers' LwSciSynObj waiter requirements and sends
    //!  attributes for CPU waiting upstream to producer block.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSyncWaiter
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
    void dstRecvSyncWaiter(
        uint32_t const dstIndex,
        Waiters const& syncWaiter) noexcept final;

    //! \brief Discards the LwSciSync signal information from the producer
    //!   and sends an empty list of sync objects downstream.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvSyncSignal
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Signals::sizeInit() or Signals::doneSet().
    void srcRecvSyncSignal(
        uint32_t const srcIndex,
        Signals const& syncSignal) noexcept final;

    //! \brief Receives payload presented by producer block and holds it
    //!   until all fences have expired, then forwards it downstream.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPayload
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: Packet provided is not valid.
    //! - LwSciError_StreamPacketInaccessible: Packet provided is not
    //!   upstream.
    //! - Any error returned by Packet::fenceProdCopy().
    //!
    //! \if TIER4_SWAD
    //! \implements{}
    //! \endif
    void srcRecvPayload(
        uint32_t const srcIndex,
        Packet const& prodPayload) noexcept final;

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
    //!   a new PresentSync instance is created.
    LwSciSyncModule         syncModule;

    //! \brief Flag indicating whether sync waiter info has been set up.
    std::atomic<bool>       waiterDone;

    //! \brief Tracks packets for which fences are unexpired.
    //!   Initialized to an empty queue when a new PresentSync instance
    //!   is created.
    Packet::PayloadQ        fenceWaitQueue;

    //! \brief CPU context used for doing CPU waits.
    //!   Initialized in PresentSync::dstSendSyncAttr().
    LwSciSyncCpuWaitContext waitContext;

    //! \brief Conditional variable to wait for packet with unexpired
    //!   fences in queue.
    std::condition_variable packetCond;

    //! \brief Representation of I/O thread used for managing the fence waiting.
    //!   This thread launches when a new PresentSync instance is created
    //!   and exelwtes waitForFenceThreadFunc().
    std::thread  dispatchThread;

    //! \brief Flag to control the dispatchThread() when block is destroyed.
    //!   Initialized to false when a new PresentSync instance
    //!   is created.
    bool  teardown;
};

} // namespace LwSciStream
#endif // PRESENT_SYNC_H
