//! \file
//! \brief LwSciStream C2C source block declaration.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef C2CSRC_H
#define C2CSRC_H
#include <cstdint>
#include <array>
#include <vector>
#include <utility>
#include <unordered_map>
#include "covanalysis.h"
#include "block.h"
#include "ipcsrc.h"

namespace LwSciStream {

// Forward declaration
class C2CPacket;

//! \brief Alias for smart pointer of C2CPacket class.
using C2CPacketPtr = std::shared_ptr<C2CPacket>;

//! \brief Alias for the unordered map of LwSciStreamPacket and C2CPacketPtr
//!   as key-value pairs.
using C2CPacketMap = std::unordered_map<LwSciStreamPacket, C2CPacketPtr>;

//! \brief Stores the packet information from C2CDst.
// TODO: Hopefully when both C2C and API changes are done, this can
//       be consolidated with the regular Packet class, but using
//       different packet descriptors. But with both in transition
//       as we determine the data each one requires, its best that
//       they be separate.
class C2CPacket
{
public:
    C2CPacket(
        LwSciStreamPacket const paramHandle,
        Elements const& paramElements,
        IpcBuffer& recvBuf) noexcept;

    ~C2CPacket(void) noexcept;

    C2CPacket(void) noexcept = delete;
    C2CPacket(const C2CPacket&) noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    C2CPacket(C2CPacket&&) noexcept = delete;
    C2CPacket& operator=(const C2CPacket&) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    C2CPacket& operator=(C2CPacket&&) & noexcept = delete;

    //! \brief Retrieves LwSciStreamPacket of the packet instance.
    //!
    //! \return LwSciStreamPacket
    LwSciStreamPacket handleGet(void) const noexcept
    {
        return handle;
    }

    bool isInitSuccess(void) const noexcept {
        return initSuccess;
    }
    bool pendingStatusEvent(void) noexcept;

private:
    // Indicate if initialized successfully.
    bool initSuccess;

    //! \brief Handle for the packet.
    //!   Initialized when a packet instance is created.
    LwSciStreamPacket handle;

    uint32_t elemCount;
    TrackArray<LwSciWrap::BufObj> elemBuffers;

    // Wheteher status event is pending.
    bool statusEvent;

public:

    //
    // C2C Buffer operations
    //

    //! \brief Registers LwSciBufObj with C2C service and saves the returned
    //!   target handle.
    //!
    //! \param [in] channelHandle: Handle to the C2C channel.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by TrackArray::peek() and Wrapper::viewVal().
    //! * Any error returned by LwSciBufRegisterTargetObjIndirectChannelC2c().
    LwSciError registerC2CBufTargetHandles(
        LwSciC2cHandle const channelHandle) noexcept;

    //! \brief Retrieves the target handle of the indexed C2C buffer.
    //!
    //! \param [in] bufIndex: The index of the buffer to query.
    //!
    //! \return LwSciC2cBufTargetHandle: target handle of C2C buffer.
    LwSciC2cBufTargetHandle c2cBufTargetHandleGet(
        size_t const bufIndex) const noexcept
    {
        assert(c2cBufHandles.size() > bufIndex);
        return c2cBufHandles[bufIndex];
    };
private:
    // List of C2C buffer target handles
    std::vector<LwSciC2cBufTargetHandle> c2cBufHandles;


public:
    //! \brief A simple queue, which is used by the C2CSrc to manage
    //!   available C2CPacket(s) from C2CDst.
    class PayloadQ final
    {
    public:
        PayloadQ(void) noexcept = default;
        ~PayloadQ(void) noexcept = default;
        PayloadQ(const PayloadQ&) noexcept = delete;
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
        PayloadQ(PayloadQ&&) noexcept = delete;
        PayloadQ& operator=(const PayloadQ&) & noexcept = delete;
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
        PayloadQ& operator=(PayloadQ&&) & noexcept = delete;

        //! \brief Adds a packet instance at the tail of PayloadQ.
        //!
        //! \param [in] newPacket: Smart pointer of a packet instance.
        //!
        //! \return void.
        void enqueue(C2CPacketPtr const& newPacket) noexcept;

        //! \brief Removes a packet instance from the head of PayloadQ.
        //!
        //! \return Smart pointer of the packet instance removed.
        //!   If PayloadQ is empty, returns nullptr.
        C2CPacketPtr dequeue(void) noexcept;

        //! \brief Checks if the PayloadQ is empty.
        //!
        //! \return boolean
        //! * true: If the PayloadQ is empty.
        //! * false: Otherwise.
        bool empty(void) const noexcept
        {
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            return (nullptr == head);
        };

    private:
        //! \cond TIER4_SWUD
        //! \brief Head of the PayloadQ.
        C2CPacketPtr   head{};
        //! \brief Tail of the PayloadQ.
        C2CPacketPtr   tail{};
        //! \endcond
    };

private:
    // Set prev/next pointers and queued flag, returning old value
    // TODO: Use move semantics?
    // Swaps payloadPrev with the newPrev.
    C2CPacketPtr  swapPrev(C2CPacketPtr const& newPrev) noexcept;

    // Swaps payloadNext with the newNext.
    C2CPacketPtr  swapNext(C2CPacketPtr const& newNext) noexcept;

    // Sets payloadQueued with the newQueued to indicate whether
    // the packet instance is in the PayloadQ or not.
    bool          swapQueued(bool const newQueued) noexcept;

    //! \cond TIER4_SWUD

    // Pointers to adjacent packets in payload queue
    // Note: Prev is closer to tail. Next is closer to head.

    //! \brief Pointer to its previous packet instance in PayloadQ.
    C2CPacketPtr        payloadPrev{};
    //! \brief Pointer to its next packet instance in PayloadQ.
    C2CPacketPtr        payloadNext{};
    //! \brief Indicates whether the packet instance is in PayloadQ.
    bool                payloadQueued{ false };

    //! \endcond
    // end TIER4_SWUD
};

// TODO: C2C block inherits from IPC block now to reuse implementation in
//       IPC block. But many of them will be overridden by C2C block. Once
//       C2C block is fully implemented, we may create a base class with
//       common stuff shared by both IPC and C2C blocks. C2C and IPC blocks
//       will inherit from the base class.

//! \brief C2C source block is the upstream half of an C2C block pair which
//!  allows packets to be transmitted between SOCs.
//!
//! C2CSrc blocks have one normal input, and for their output must be
//! coupled with a corresponding C2C destination block.
//!
class C2CSrc :
    public IpcSrc
{
public:
    C2CSrc(void) noexcept                        = delete;
    C2CSrc(const C2CSrc &) noexcept              = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    C2CSrc(C2CSrc &&) noexcept                   = delete;
    C2CSrc& operator=(const C2CSrc &) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    C2CSrc& operator=(C2CSrc &&) & noexcept      = delete;

    //! \brief Constructs an instance of the class
    //!
    //! \param [in] ipc: LwSciIpcEndpoint to be used for communication.
    //! \param [in] syncModule: Instance of LwSciSyncModule.
    //! \param [in] bufModule: Instance of LwSciBufModule.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
    explicit C2CSrc(LwSciIpcEndpoint const ipc,
                    LwSciSyncModule const syncModule,
                    LwSciBufModule const bufModule) noexcept;

    //! \brief Destroys the C2CSrc block instance.
    ~C2CSrc(void) noexcept final;

    //! \brief Connects the C2CSrc block to the queue block referenced by
    //!  the @a paramQueue.
    //!
    //!  <b>Preconditions</b>
    //!   - C2CSrc block instance should have been already registered
    //!     by a successful call to Block::registerBlock() interface.
    //!
    //! \param [in] paramQueue: reference to the queue block instance.
    //!  Valid value: paramQueue is valid if it is not NULL.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: If C2CSrc and queue blocks are connected
    //!   successfully.
    //! * LwSciError_BadParameter: If queue block reference is NULL or the
    //!   @a paramQueue is not referring to a queue block.
    //! * LwSciError_InsufficientResource: C2CSrc block has no available input
    //!   connection or the Queue block has no available output connection.
    LwSciError BindQueue(BlockPtr const& paramQueue) noexcept;

    //! \brief C2CSrc block retrieves the associated queue block object.
    //!
    //! \param [in,out] paramBlock: pointer to queue block object.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: If C2CSrc block initialization is successful.
    //! * LwSciError_NotInitialized: If C2CSrc block initialization is failed.
    LwSciError getInputConnectPoint(
        BlockPtr& paramBlock) const noexcept final;

    //! \brief Configs C2CSrc block to use event service and finalize other
    //!   block configuration at its first connection with other block.
    //!
    //! \return void
    void finalizeConfigOptions(void) noexcept final;

    //! \brief Receives alllocated element information from pool and
    //!   saves for transmission over IPC. Opens channel for C2C copy.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvAllocatedElements
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_InsufficientMemory: Unable to allocate memory for buffer
    //!   size array.
    //! - LwSciError_NotSupported: element buffer type not supported.
    //! - LwSciError_Overflow: element buffer size too big.
    //! - Any error returned by Elements::attrPeek().
    //! - Any error returned by LwSciBufAttrListGetAttrs().
    //! - Any error returned by LwSciBufOpenIndirectChannelC2c().
    //! - Any error returned by IpcSrc::srcRecvAllocatedElements().
    void srcRecvAllocatedElements(
        uint32_t const srcIndex,
        Elements const& inElements) noexcept final;

    //! \brief Queues message to send new packet to C2CDst.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPacketCreate
    //!
    //! \return void
    // TODO: srcRecvPacketCreate() doesn't queue up events and implementation
    // kept empty. Doxygen comments may need an update depending on the
    // enhancements.
    void srcRecvPacketCreate(
        uint32_t const srcIndex,
        Packet const& origPacket) noexcept final;

    //! \brief Uses sync attributes from producer to create reconciled
    //!   attributes that will signal to this side of the stream when
    //!   copy is done.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvSyncWaiter
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_AlreadyDone: Waiter attributes already provided.
    //! - LwSciError_InsufficientMemory: Not able to allocate vector to
    //!   hold unreconciled attribute lists.
    //! - Any error returned by LwSciSyncAttrListReconcile().
    void srcRecvSyncWaiter(
        uint32_t const srcIndex,
        Waiters const& syncWaiter) noexcept final;

    //! \brief Saves LwSciSync signal info from producer and sends the
    //!   C2C signal info back upstream.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvSyncSignal
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by LwSciSyncObjAlloc().
    //! - Any error returned by Signals::sizeInit(), Signals::syncSet(),
    //!   Signals::doneSet(), or Signals::copy().
    void srcRecvSyncSignal(
        uint32_t const srcIndex,
        Signals const& syncSignal) noexcept final;

    //! \brief Increments number of available paylods for sending downstream.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPayload
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by enqueueIpcWrite().
    void srcRecvPayload(
        uint32_t const srcIndex,
        Packet const& prodPayload) noexcept final;

protected:

    //
    // Virtual block-specific message handling functions ilwoked by base
    //

    //! \brief Packs next pending message, if any, into sendBuffer.
    //!
    //! \copydetails LwSciStream::IpcSrc::sendMessage
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by c2cSendPacketStatus().
    //! - Any error returned by c2cSendSyncSignalAttr().
    //! - Any error returned by c2cSendSyncWaiterAttr().
    //! - Any error returned by IpcSrc::sendMessage().
    LwSciError sendMessage(IpcBuffer& sendBuf,
                           Lock& blockLock) noexcept final;

    //! \brief Unpack pending message from recvBuffer.
    //!
    //! \copydetails LwSciStream::IpcDst::recvMessage
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by c2cRecvPacketCreate().
    //! - Any error returned by c2cRecvPacketsComplete().
    //! - Any error returned by c2cRecvPacketDelete().
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

    //! \brief Packs status for import of C2C packet.
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //! \param [in] pkt: Packet whose status should be sent.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcSrc::sendHeader().
    //! - Any error returned by IpcBuffer::packVal().
    LwSciError c2cSendPacketStatus(IpcBuffer& sendBuf,
                                   C2CPacketPtr const& pkt) noexcept;

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

    //! \brief Initiates C2C copy of payload and packs payload information.
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //! \param [in,out] blockLock: Reference to Lock
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - LwSciError_StreamInternalError: CERT ate our face.
    //! - LwSciError_NoStreamPacket: No C2C packet or no payload available.
    //! - LwSciError_StreamBadPacket: Handle for payload not recognized.
    //! - LwSciError_StreamPacketInaccessible: Payload packet not upstream.
    //! - Any error returned by Packet::fenceProdCopy.
    //! - Any error returned by Packet::fenceProdGet.
    //! - Any error returned by Packet::fenceConsFill.
    //! - Any error returned by LwSciBufPushWaitIndirectChannelC2c(),
    //!   LwSciBufPushCopyIndirectChannelC2c(),
    //!   LwSciBufPushSignalIndirectChannelC2c(), or
    //!   LwSciBufPushSubmitIndirectChannelC2c().
    //! - Any error returned by IpcSrc::sendHeader().
    //! - Any error returned by IpcBuffer::packVal().
    //! - Any error returned by ipcBufferPack(LwSciWrap::SyncFence).
    //! - Any error returned by LwSciSyncFenceWait().
    LwSciError c2cSendPayload(IpcBuffer& sendBuf,
                              Lock& blockLock) noexcept;

    //! \brief Creates new C2C packet, unpacking buffer information from
    //!   and sends status back downstream.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    // TODO: doxygen
    LwSciError c2cRecvPacketCreate(IpcBuffer &recvBuf) noexcept;

    //! \brief Handles deletion of C2C packet resources.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    // TODO: doxygen
    LwSciError c2cRecvPacketDelete(IpcBuffer &recvBuf) noexcept;

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

    //! \brief Unpacks sync object used by C2C copies to signal to
    //!   C2CDst when the copy is finished.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    // TODO: doxygen
    LwSciError c2cRecvSyncWaiterObj(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpacks message that C2C packet is available for reuse.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    // TODO: doxygen
    LwSciError c2cRecvPayload(IpcBuffer& recvBuf) noexcept;

protected:
    //! \brief Searches C2CPacketMap for a C2CPacket instance with the given
    //!   LwSciStreamPacket under thread protection provided by
    //!   Block::blkMutexLock() and returns the smart pointer to it if found.
    //!
    //!
    //! \param [in] paramHandle: LwSciStreamPacket
    //! \param [in] locked: Indicates whether blkMutex is held by the caller.
    //!
    //! \return C2CPacketPtr: Pointer to the matched C2CPacket instance if
    //!                       found, or null pointer if not found.
    C2CPacketPtr c2cPktFind(
                    LwSciStreamPacket const paramHandle,
                    bool const locked=false) noexcept;

    LwSciError   c2cPktCreate(LwSciStreamPacket const paramHandle,
                              IpcBuffer& recvBuf) noexcept;

    void         c2cPktRemove(LwSciStreamPacket const paramHandle) noexcept;

private:
    //! \brief Handle to the C2C channel used to copy buffer data
    //!   from C2CSrc block to C2CDst block. Initialized in
    //!   srcRecvAllocatedElements() when the number of allocated
    //!   buffers is decided. Deinited in destructor.
    LwSciC2cHandle          c2cChannel;

    //! \brief An event service object used to handle C2C callbacks.
    // TODO: Not in use now. It will be actively used when switching over our
    // internal thread to using LwSciEventService. Deinited in destructor.
    LwSciEventLoopService* service;

    //! \brief Pointer to queue block instance. It is initialized when a queue
    //!  block is connected with the C2CSrc block in C2CSrc::BindQueue()
    //!  interface.
    BlockPtr                queue;

    //! \brief Array of buffer size for each element. Initialized in
    //!   srcRecvAllocatedElements() when receiving the allocated elements.
    std::vector<uint64_t>   bufSize;

    //! \brief Wrapped LwSciSyncAttrList for CPU waiting.
    //!   Initialized at construction.
    LwSciWrap::SyncAttr     cpuWaiterAttr;

    //! \brief Wrapped LwSciSyncAttrList for CPU signalling.
    //!   Initialized at construction.
    LwSciWrap::SyncAttr     cpuSignalAttr;

    //! \brief Wrapped LwSciSyncAttrList for C2C waiting.
    //!   Initialized at construction.
    LwSciWrap::SyncAttr     c2cWaiterAttr;

    //! \brief Wrapped LwSciSyncAttrList for C2C signalling.
    //!   Initialized at construction.
    LwSciWrap::SyncAttr     c2cSignalAttr;

    //! \brief Waiter attributes used by C2CSrc before initiating copies.
    //!   Used to create what Buf/Sync API docs call "engineWritesDoneObj".
    //!   Initialized at construction.
    LwSciWrap::SyncAttr     engineDoneWaiterAttr;

    //! \brief Signal attributes used by C2CSrc to tell C2CDst copy is done.
    //!   Used to create what Buf/Sync API docs call "copyDoneConsObj".
    //!   Initialized at construction.
    LwSciWrap::SyncAttr     copyDoneSignalAttr;

    //! \brief Signal attributes used by C2CDst to tell C2CSrc read is done.
    //!   Used to create what Buf/Sync API docs call "consReadDoneProdObj".
    //!   Initialized to empty at construction and then filled when
    //!   received from C2CDst.
    LwSciWrap::SyncAttr     readDoneSignalAttr;

    //! \brief Reconciled attributes for sync object that will be signalled
    //!   when copy is done to alert this side of the stream.
    //!   Used to create what Buf/Sync API docs call "copyDoneProdObj".
    //!   Initialized to empty at construction and filled when producer's
    //!   waiter requirements are received.
    LwSciWrap::SyncAttr     copyDoneAttr;

    //! \brief Flag indicating pending message to send copyDoneSignalAttr.
    //!   Initialized to true at construction.
    bool                    copyDoneSignalAttrMsg;

    //! \brief Flag indicating endpoint does not support fences for at
    //!   least one element, so C2C block must wait after copy.
    //!   Initialized to false at construction.
    bool                    cpuWaitAfterCopy;

    //! \brief Flag indicating pending message for waiter attribute flow.
    //!   Initialized to false at construction.
    bool                    waiterAttrMsg;

    //! \brief LwSciC2cSyncHandle used for signalling to the C2CDst
    //!   that the copy has finished.
    //!   Initialized to NULL at construction.
    LwSciC2cSyncHandle      c2cSignalConsHandle;

    //! \brief LwSciC2cSyncHandle used for signalling to the local
    //!   endpoint that the copy has finished.
    //!   Initialized to NULL at construction.
    LwSciC2cSyncHandle      c2cSignalProdHandle;

    //! \brief LwSciC2cSyncHandle used for signalling to the C2C
    //    library that producer engine has completed writing to
    //    element buffer.
    std::vector<LwSciC2cSyncHandle>      c2cWaitProdEngineHandle;

    // TODO: Will be removed once the C2C sync is implemented
    //! \brief CPU context used for doing CPU waits.
    //!   Initialized at construction.
    LwSciSyncCpuWaitContext waitContext;

    //! \brief Map tracking all C2CPackets from downstream.
    //!   A new C2CPacket instance will be created and added to map in
    //!   C2CSrc::dstCreatePacket() interface. C2CPacket instances from this
    //!   map can be looked up by calling C2CSrc::c2cPktFind() and interface.
    C2CPacketMap            c2cPktMap;

    //! \brief Queue of C2CPackets available for reuse. Initialized to
    //!   empty queue when a new C2CSrc instance is created.
    C2CPacket::PayloadQ     c2cPayloadQueue;

    //! \brief Tracks number of payloads available to send.
    //!   Initialized to zero when a new C2CSrc instance is created.
    std::atomic<uint32_t>   numPayloadsAvailable;

    //! \brief Tracks number of C2CPackets available for reuse.
    //!   Initialized to zero when a new C2CSrc instance is created.
    uint32_t                numC2CPktsAvailable;

    //! \brief Tracks number of payloads pending adding to the IPC write queue.
    //!   Initialized to zero when a new C2CSrc instance is created.
    uint32_t                numPayloadsPendingWriteSignal;

    //! \brief Tracks number of C2CPackets avaliable for incoming payloads
    //!  to be added the IPC write queue.
    //!   Initialized to zero when a new C2CSrc instance is created.
    uint32_t                numC2CPktsForWriteSignal;
};

} //namespace LwSciStream

#endif // C2CSRC_H
