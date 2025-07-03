//! \file
//! \brief LwSciStream C2C destination block declaration.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef C2CDST_H
#define C2CDST_H
#include <cstdint>
#include <array>
#include <utility>
#include "covanalysis.h"
#include "block.h"
#include "ipcbuffer.h"
#include "ipcdst.h"

namespace LwSciStream {

//! \brief C2C destination block is the downstream half of an C2C block pair
//!  which allows packets to be transmitted between SOCs.
//!
//! C2CDst blocks have one normal output, and for their input must be
//! coupled with a corresponding C2C source block.
//!
class C2CDst :
    public IpcDst
{
public:
    C2CDst(void) noexcept                        = delete;
    C2CDst(const C2CDst &) noexcept              = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    C2CDst(C2CDst &&) noexcept                   = delete;
    C2CDst& operator=(const C2CDst &) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    C2CDst& operator=(C2CDst &&) & noexcept      = delete;
    ~C2CDst(void) noexcept                 = default;

    //! \brief Constructs an instance of the class
    //!
    //! \param [in] ipc: LwSciIpcEndpoint to be used for communication.
    //! \param [in] syncModule: Instance of LwSciSyncModule.
    //! \param [in] bufModule: Instance of LwSciBufModule.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
    explicit C2CDst(
        LwSciIpcEndpoint const ipc,
        LwSciSyncModule const syncModule,
        LwSciBufModule const bufModule) noexcept;

    //! \brief Connects the C2CDst block to the pool block referenced by
    //!  the @a paramPool.
    //!
    //!  <b>Preconditions</b>
    //!   - C2CDst block instance should have been already registered
    //!     by a successful call to Block::registerBlock() interface.
    //!
    //! \param [in] paramPool: reference to the pool block instance.
    //!  Valid value: paramPool is valid if it is not NULL.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: If C2CDst and pool blocks are connected
    //!   successfully.
    //! * LwSciError_BadParameter: If pool block reference is NULL or the
    //!   @a paramPool is not referring to a pool block.
    //! * LwSciError_InsufficientResource: C2CDst block has no available output
    //!   connection or the pool block has no available input connection.
    LwSciError BindPool(BlockPtr const& paramPool) noexcept;

    //! \brief C2CDst block retrieves the associated pool block object.
    //!
    //! \param [in,out] paramBlock: pointer to pool block object.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: If C2CDst block initialization is successful.
    //! * LwSciError_NotInitialized: If C2CDst block initialization is failed.
    LwSciError getOutputConnectPoint(
        BlockPtr& paramBlock) const noexcept;

    //! \brief Queues message to send new packet to C2CSrc.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPacketCreate
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_InsufficientMemory: If unable to create a new packet
    //!   instance.
    //! - LwSciError_StreamInternalError: If there's already a packet instance
    //!   in PacketMap with the same LwSciStreamPacket.
    void dstRecvPacketCreate(
        uint32_t const dstIndex,
        Packet const& origPacket) noexcept final;

    //! \brief Queues message to send C2C packet list completion to C2CSrc.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPacketsComplete
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_AlreadyDone: If message was already queued.
    //! - Any error returned by IpcDst::enqueueIpcWrite().
    void dstRecvPacketsComplete(
        uint32_t const dstIndex) noexcept final;

    //! \brief Queues message to send packet deletion to C2CSrc.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPacketDelete
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamInternalError: If the packet's current location
    //!   is not Location::Downstream or packet is already marked for deletion.
    void dstRecvPacketDelete(
        uint32_t const dstIndex,
        LwSciStreamPacket const handle) noexcept final;

    //! \brief Uses sync attributes from consumer to create reconciled
    //!   attributes that will signal to this side of the stream when
    //!   copy is done.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSyncWaiter
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_AlreadyDone: Waiter attributes already provided.
    //! - LwSciError_InsufficientMemory: Not able to allocate vector to
    //!   hold unreconciled attribute lists.
    //! - Any error returned by LwSciSyncAttrListReconcile().
    void dstRecvSyncWaiter(
        uint32_t const dstIndex,
        Waiters const& syncWaiter) noexcept final;

    //! \brief Saves LwSciSync signal info from consumer(s) and sends the
    //!   C2C signal info back downstream.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSyncSignal
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by LwSciSyncObjAlloc().
    //! - Any error returned by Signals::sizeInit(), Signals::syncSet(),
    //!   Signals::doneSet(), or Signals::copy().
    void dstRecvSyncSignal(
        uint32_t const dstIndex,
        Signals const& syncSignal) noexcept final;

    //! \brief Queues message to send packet to C2CSrc for reuse.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPayload
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by IpcDst::enqueueIpcWrite()
    void dstRecvPayload(
        uint32_t const dstIndex,
        Packet const& consPayload) noexcept final;

protected:

    //
    // Virtual block-specific message handling functions ilwoked by base
    //

    //! \brief Packs next pending message, if any, into sendBuffer.
    //!
    //! \copydetails LwSciStream::IpcDst::sendMessage
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by c2cSendPacketCreate().
    //! - Any error returned by c2cSendPacketsComplete().
    //! - Any error returned by c2cSendSyncSignalAttr().
    //! - Any error returned by c2cSendSyncWaiterAttr().
    //! - Any error returned by c2cSendSyncWaiterObj().
    //! - Any error returned by c2cSendPacketDelete().
    //! - Any error returned by c2cSendPayload().
    //! - Any error returned by IpcSrc::sendMessage().
    LwSciError sendMessage(IpcBuffer& sendBuf,
                           Lock& blockLock) noexcept final;

    //! \brief Unpack pending message from recvBuffer.
    //!
    //! \copydetails LwSciStream::IpcDst::recvMessage
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by c2cRecvPacketStatus().
    //! - Any error returned by c2cRecvSyncSignalAttr().
    //! - Any error returned by c2cRecvSyncWaiterAttr().
    //! - Any error returned by c2cRecvPayload().
    //! - Any error returned by IpcDst::recvMessage().
    LwSciError recvMessage(IpcBuffer& recvBuf,
                           IpcMsg const msgType) noexcept final;

private:

    //
    // Block-specific functions for handling individual messages
    //

    //! \brief Packs packet creation message from secondary pool.
    //!
    //! \param [in,out] sendBuf: Buffer into which to pack the message.
    //! \param [in] pkt: Packet being created.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcDst::sendHeader().
    //! - Any error returned by IpcBuffer::packVal().
    //! - Any error returned by Packet::definePack().
    LwSciError c2cSendPacketCreate(IpcBuffer& sendBuf,
                                   PacketPtr const& pkt) noexcept;

    //! \brief Packs packet list completion message from secondary pool.
    //!
    //! \param [in,out] sendBuf: Buffer into which to pack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcDst::sendHeader().
    LwSciError c2cSendPacketsComplete(IpcBuffer& sendBuf) noexcept;

    //! \brief Packs packet deletion message from secondary pool.
    //!
    //! \param [in,out] sendBuf: Buffer into which to pack the message.
    //! \param [in] pkt: Packet being deleted.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcDst::sendHeader().
    //! - Any error returned by IpcBuffer::packVal().
    LwSciError c2cSendPacketDelete(IpcBuffer& sendBuf,
                                   PacketPtr const& pkt) noexcept;

    //! \brief Packs signal sync attributes.
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcSrc::sendHeader().
    //! - Any error returned by ipcBufferPacket(LwSciWrap::SyncAttr).
    LwSciError c2cSendSyncSignalAttr(IpcBuffer& sendBuf) noexcept;

    //! \brief Packs message to initiate flow of waiter sync attributes.
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcSrc::sendHeader().
    LwSciError c2cSendSyncWaiterAttr(IpcBuffer& sendBuf) noexcept;

    //! \brief Packs sync object that C2CDst will use to wait for C2C
    //!   copies to finish.
    //!
    //! \param [in,out] sendBuf: Buffer into which to pack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcDst::sendHeader().
    //! - Any error returned by ipcBufferPack(LwSciWrap::SyncObj).
    //! - Any error returned by IpcBuffer::packVal().
    LwSciError c2cSendSyncWaiterObj(IpcBuffer& sendBuf) noexcept;

    //! \brief Packs message for return of payload for reuse.
    //!
    //! \param [in,out] sendBuf: Buffer into which to pack the message.
    //! \param [in,out] blockLock: Reference to Lock
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - LwSciError_NoStreamPacket: No payload waiting in queue.
    //! - LwSciError_StreamBadPacket: Packet obtained from queue is invalid.
    //! - LwSciError_StreamPacketInaccessible: Packet is not downstream.
    //! - Any error returned by Packet::fenceConsCopy() or
    //!   Packet::fenceConsWait().
    //! - Any error returned by IpcDst::dstRecvPayload().
    //! - Any error returned by IpcDst::sendHeader().
    //! - Any error returned by IpcBuffer::packVal().
    LwSciError c2cSendPayload(IpcBuffer& sendBuf,
                              Lock& blockLock) noexcept;

    //! \brief Unpacks status for C2C packet and sends downstream.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - LwSciError_StreamBadPacket: Packet handle not recognized.
    //! - Any error returned by IpcBuffer::unpackVal().
    //! - Any error returned by Packet::statusProdSet().
    LwSciError c2cRecvPacketStatus(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpacks sync signal attributes from the other side.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - LwSciError_AlreadyDone: Signal attributes were already received.
    //! - Any error returned by ipcBufferUnpack(LwSciWrap::SyncAttr).
    //! - Any error encountered during duplication or import of an
    //!   LwSciSyncAttrList.
    LwSciError c2cRecvSyncSignalAttr(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpacks message triggering flow of C2C block's waiter
    //!   attributes to the endpoint.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by Waiters::sizeInit() or Waiters::entryFill().
    LwSciError c2cRecvSyncWaiterAttr(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpacks C2C copy payload and sends downstream.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - LwSciError_StreamBadPacket: Packet handle unrecognized.
    //! - LwSciError_StreamPacketInaccessible: Packet is not upstream.
    //! - Any error returned by IpcBufferUnpack(LwSciWrap::SyncFence).
    //! - Any error returned by IpcBuffer::unpackval().
    //! - Any error returned by Packet::fenceProdFill().
    //! - Any error returned by LwSciSyncFenceWait().
    LwSciError c2cRecvPayload(IpcBuffer& recvBuf) noexcept;

private:

    BlockPtr                    pool;

    //! \brief Flag indicating pool has indicated packet list completion.
    //!   It is intialized to false at creation.
    std::atomic<bool>           secondaryPacketExportDone;

    //! \brief Flag indicating packet completion event is pending.
    //!   It is intialized to false at creation.
    // TODO: Probably doesn't need to be atomic
    std::atomic<bool>           secondaryPacketExportEvent;

    //! \brief Wrapped LwSciSyncAttrList for CPU waiting.
    //!   Initialized at construction.
    LwSciWrap::SyncAttr         cpuWaiterAttr;

    //! \brief Wrapped LwSciSyncAttrList for CPU signalling.
    //!   Initialized at construction.
    LwSciWrap::SyncAttr         cpuSignalAttr;

    //! \brief Wrapped LwSciSyncAttrList for C2C waiting.
    //!   Initialized at construction.
    LwSciWrap::SyncAttr         c2cWaiterAttr;

    //! \brief Wrapped LwSciSyncAttrList for C2C signalling.
    //!   Initialized at construction.
    LwSciWrap::SyncAttr         c2cSignalAttr;

    //! \brief Waiter attributes used by C2CDst before returning packet.
    //!   Used to create what Buf/Sync API docs call "engineWritesDoneConsObj".
    //!   Initialized at construction.
    LwSciWrap::SyncAttr         engineDoneWaiterAttr;

    //! \brief Signal attributes used by C2CSrc to tell C2CDst copy is done.
    //!   Used to create what Buf/Sync API docs call "copyDoneConsObj".
    //!   Initialized to empty at construction and then filled when
    //!   received from C2CSrc.
    LwSciWrap::SyncAttr         copyDoneSignalAttr;

    //! \brief Signal attributes used by C2CDst to tell C2CSrc read is done.
    //!   Used to create what Buf/Sync API docs call "consReadDoneProdObj".
    //!   Initialized at construction.
    LwSciWrap::SyncAttr         readDoneSignalAttr;

    //! \brief Reconciled attributes for sync object that will be signalled
    //!   when copy is done to alert this side of the stream.
    //!   Used to create what Buf/Sync API docs call "copyDoneConsObj".
    //!   Initialized to empty at construction and filled when consumer's
    //!   waiter requirements are received.
    LwSciWrap::SyncAttr         copyDoneAttr;

    //! \brief Flag indicating pending message to send readDoneSignalAttr.
    //!   Initialized to true at construction.
    bool                        readDoneSignalAttrMsg;

    //! \brief Flag indicating endpoint does not support fences for at
    //!   least one element, so C2C block must wait after copy.
    //!   Initialized to false at construction.
    bool                        cpuWaitAfterCopy;

    //! \brief Flag indicating pending message for waiter attribute flow.
    //!   Initialized to false at construction.
    bool                        waiterAttrMsg;

    //! \brief LwSciSyncObj used by C2CSrc to signal a copy has finished.
    //!   Initialized to empty at construction.
    LwSciWrap::SyncObj          c2cCopySyncObj;

    //! \brief Flag indicating sending of the c2c sync object is pending.
    //!   Intialized to false at construction.
    bool                        c2cCopySyncObjSend;

    //! \brief CPU context used for doing CPU waits.
    //!   Initialized at construction.
    LwSciSyncCpuWaitContext     waitContext;

    //! \brief Queue of packets waiting to be returned upstream.
    //!   Initialized at construction to empty.
    Packet::PayloadQ            c2cReusePayloadQueue;

    //! \brief Tracks number of payloads available for reuse.
    //!   Initialized to zero when a new C2CDst instance is created.
    std::atomic<uint32_t>       numPayloadsAvailable;
};

} //namespace LwSciStream

#endif // C2CDST_H
