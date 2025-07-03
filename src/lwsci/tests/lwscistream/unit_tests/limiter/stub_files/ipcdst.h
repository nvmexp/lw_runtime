//! \file
//! \brief LwSciStream IPC destination block declaration.
//!
//! \copyright
//! Copyright (c) 2019-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef IPCDST_H
#define IPCDST_H
#include <cstdint>
#include <thread>
#include <atomic>
#include <array>
#include <utility>
#include "covanalysis.h"
#include "block.h"
#include "ipccomm.h"
#include "ipcbuffer.h"
#include "endinfo.h"
#include "elements.h"

namespace LwSciStream {

//! \brief IPC destination block is the downstream half of an IPC block pair
//!  which allows packets to be transmitted between processes.
//!
//! IpcDst blocks have one normal output, and for their input must be
//! coupled with a corresponding IPC source block. IpcDst blocks act like
//! upstream blocks, overriding SrcBlockInterface functions.
//!
//! \if TIER4_SWAD
//! \implements{19791651}
//! \endif
//! \if TIER4_SWUD
//! \implements{20283711}
//! \endif
class IpcDst :
    public Block
{
public:
    IpcDst(void) noexcept                        = delete;
    IpcDst(const IpcDst &) noexcept              = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    IpcDst(IpcDst &&) noexcept                   = delete;
    IpcDst& operator=(const IpcDst &) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    IpcDst& operator=(IpcDst &&) & noexcept      = delete;

    //! \brief Constructs an instance of the class and
    //!        initializes all data fields for a new IpcDst object.
    //!
    //! \param [in] ipc: LwSciIpcEndpoint to be used for communication.
    //! \param [in] syncModule: Instance of LwSciSyncModule.
    //! \param [in] bufModule: Instance of LwSciBufModule.
    //! \param [in] isC2C: Flag indicating C2CDst block.
    //!
    //! \if TIER4_SWAD
    //! \implements{19791777}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
    explicit IpcDst(
        LwSciIpcEndpoint const ipc,
        LwSciSyncModule const syncModule,
        LwSciBufModule const bufModule,
        bool const isC2C = false) noexcept;

    //! \brief Frees any LwSciBuf and or LwSciSync handles
    //!        still referenced by the IpcDst block and deletes the instance.
    //!        Destructor is called when refcount
    //!        to this object has reached zero.
    //!
    //! \if TIER4_SWAD
    //! \implements{19791792}
    //! \endif
    ~IpcDst(void) noexcept override;

    //! \brief Disconnects downstream blocks.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: Always, as this operation cannot fail.
    //! - Triggers the following error events:
    //!     - For any error code that IpcComm::signalDisconnect() can generate
    //!       in case of failure.
    //!
    //! \if TIER4_SWAD
    //! \implements{19791801}
    //! \endif
    LwSciError disconnect(void) noexcept final;

    //! \brief Receives consumer info from downstream, saves, and signals
    //!   it is ready to send upstream.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvConsInfo
    //!
    //! \return void
    void dstRecvConsInfo(
        uint32_t const dstIndex,
        EndInfoVector const& info) noexcept final;

    //! \brief Receives supported element information from consumer(s) and
    //!   saves for transmission over IPC.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSupportedElements
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Elements::dataCopy().
    //! - Any error returned by IpcComm::signalWrite().
    void dstRecvSupportedElements(
        uint32_t const dstIndex,
        Elements const& inElements) noexcept final;

    //! \brief Receives acceptance or rejection of a packet from the
    //!   consumer(s) and saves for transmission over IPC.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPacketStatus
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: Packet handle not found in map.
    //! - Any error returned by IpcComm::signalWrite().
    void dstRecvPacketStatus(
        uint32_t const dstIndex,
        Packet const& origPacket) noexcept final;

    //! \brief Queues message to send LwSciSync waiter information from
    //!   the consumer(s) to IpcSrc.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSyncWaiter
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Waiters::copy().
    //! - Any error returned by IpcComm::signalWrite().
    void dstRecvSyncWaiter(
        uint32_t const dstIndex,
        Waiters const& syncWaiter) noexcept override;

    //! \brief Queues message to send LwSciSync signal information from
    //!   the consumer(s) to IpcSrc.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSyncSignal
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Signals::copy().
    //! - Any error returned by IpcComm::signalWrite().
    void dstRecvSyncSignal(
        uint32_t const dstIndex,
        Signals const& syncSignal) noexcept override;

    //! \brief Queues message to send packet to IpcDst.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::dstRecvPayload
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: The packet is invalid.
    //! - LwSciError_StreamPacketInaccessible: The packet is not upstream.
    //! - Any error returned by Packet::fenceConsCopy().
    //! - Any error returned by IpcComm::signalWrite().
    void dstRecvPayload(
        uint32_t const dstIndex,
        Packet const& consPayload) noexcept override;

    //! \brief Disconnects from the downstream blocks and informs IpcSrc of the
    //!        downstream disconnect.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstDisconnect
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19791855}
    //! \endif
    void dstDisconnect(
        uint32_t const dstIndex) noexcept final;

    //! \brief Launches the dispatch thread for sending and receiving IPC messages.
    //!
    //! \return bool, true if thread launched successfully, false if failed.
    //!
    bool startDispatchThread(void) noexcept;

    //! \brief Queue message to send phase readiness to IpcSrc.
    //!
    //! \return void
    void phaseSendReady(void) noexcept final;

protected:
    //! \brief Gets LwSciSyncModule that the LwSciSync data associated with.
    //!
    //! \return LwSciSyncModule
    LwSciSyncModule getSyncModule(void) const noexcept
    {
        return ipcSyncModule;
    };

    //! \brief Gets LwSciIpcEndpoint that used for data export/import.
    //!
    //! \return LwSciIpcEndpoint
    LwSciIpcEndpoint getIpcEndpoint(void) const noexcept
    {
        return ipcEndpoint;
    };

    //! \brief Calls IpcComm::signalWrite interface to enqueue an IPC write
    //!   request.
    //!
    //! \return LwSciError, error returned by IpcComm::signalWrite interface
    LwSciError enqueueIpcWrite(void) noexcept
    {
        return comm.signalWrite();
    }

private:

    //
    // Top level message handling operations
    //

    // I/O Thread loop for processing Ipc read/write messages.
    void dispatchThreadFunc(void) noexcept;

    // Exits the I/O thread loop.
    void destroyIOLoop(bool const wait=false) noexcept;

    // Functions to process messages ilwoked by dispatch thread

    //! \brief Prepares next pending message, if any, for sending over
    //!    LwSciIpc channel. This handles the initial connection message
    //!    itself, then ilwokes the block-specific pack function.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by ipcSendConnect().
    //! - Any error returned by sendMessage().
    // Note: When there is a common IPC base block, this will be part of it.
    LwSciError processWriteMsg() noexcept;

    //! \brief Processes message received over IPC and held in recvBuffer.
    //!   This the initial connection message itself, then ilwokes the
    //!   block-specific unpack function.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - LwSciError_NotInitialized: Message arrived before connection
    //!   established.
    //! - Any error returned by recvHeader().
    //! - Any error returned by ipcRecvConnect().
    //! - Any error returned by recvMessage().
    LwSciError processReadMsg() noexcept;

protected:
    //
    // Virtual block-specific message handling functions ilwoked by base
    //   and utilities available to them.
    //

    //! \brief Queries whether runtime readiness message has been sent.
    //!   Prior to this, init phase messages may still be pending,
    //!   and should be sent before telling the upstream block to
    //!   transition to runtime.
    //!
    //! \return bool, the runtimeReadyDone flag
    bool runtimeEnabled(void) const noexcept
    {
        return runtimeReadyDone;
    };

    //! \brief Indicates whether sending messages is allowed yet
    //!   for the block, based on whether prerequisite messages
    //!   have been sent/received.
    //!
    //! \param [in] connect: Indicates whether outgoing message is
    //!   a connection message, which is allowed to proceed before others.
    //!
    //! \return bool, whether message sending can proceed.
    // Note: When there is a common IPC base block, this will be pure
    //       virtual in it, and specialized in the derived blocks.
    bool sendEnabled(bool const connect) const noexcept;

    //! \brief Packs next pending message, if any, into sendBuffer.
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //! \param [in,out] blockLock: Reference to Lock
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by sendSupportedElements().
    //! - Any error returned by ipcSendPacketStatus().
    //! - Any error returned by ipcSendSyncWaiterAttr().
    //! - Any error returned by ipcSendSyncSignalObj().
    //! - Any error returned by sendRuntimeRead().
    //! - Any error returned by ipcSendPayload().
    //! - Any error returned by sendDisconnect().
    // Note: When there is a common IPC base block, this will be pure
    //       virtual in it, and specialized in the derived blocks.
    virtual LwSciError sendMessage(IpcBuffer& sendBuf,
                                   Lock& blockLock) noexcept;

    //! \brief Packs message header into sendBuffer.
    //!
    //! \param [in] msgType: The message type.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcBuffer::changeMode().
    //! - Any error returned by IpcBuffer::packVal().
    // Note: When there is a common IPC base block, this will be part of it.
    LwSciError sendHeader(IpcMsg const msgType) noexcept;

    //! \brief Packs EndInfoVector into the IpcBuffer.
    //!
    //! \param [in,out] recvBuf: Buffer into which to pack data.
    //!
    //! \return LwSciError, the completion code of this operation
    //! - LwSciError_Success: If message is processed successfully.
    //! - Any error returned by sendHeader().
    //! - Any error returned by ipcBufferPack().
    // Note: When there is a common IPC base block, need to virtualize this
    //       so it does the right thing when called from src or dst.
    LwSciError ipcSendConnect(IpcBuffer& recvBuf) noexcept;

    //! \brief Indicates whether receiving messages is allowed yet
    //!   for the block, based on whether prerequisite messages have
    //!   been sent/received.
    //!
    //! \param [in] connect: Indicates whether incoming message is
    //!   a connection message, which is allowed to proceed before others.
    //!
    //! \return bool, whether message receiving can proceed.
    // Note: When there is a common IPC base block, this will be pure
    //       virtual in it, and specialized in the derived blocks.
    bool recvEnabled(bool const connect) const noexcept;

    //! \brief Unpacks message with specified type in recvBuffer.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - LwSciError_IlwalidOperation: The message type is not supported.
    //! - Any error returned by recvAllocatedElements().
    //! - Any error returned by ipcRecvPacketCreate().
    //! - Any error returned by ipcRecvPacketsComplete().
    //! - Any error returned by ipcRecvPacketDelete().
    //! - Any error returned by ipcRecvSyncWaiterAttr().
    //! - Any error returned by ipcRecvSyncSignalObj().
    //! - Any error returned by recvRuntimeBegin().
    //! - Any error returned by ipcRecvPayload().
    //! - Any error returned by recvDisconnect().
    // Note: When there is a common IPC base block, this will be pure
    //       virtual in it, and specialized in the derived blocks.
    virtual LwSciError recvMessage(IpcBuffer& recvBuf,
                                   IpcMsg const msgType) noexcept;

    //! \brief Unpacks message header from recvBuffer.
    //!
    //! \param [out] msgType: The received message type.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcBuffer::changeMode().
    //! - Any error returned by IpcBuffer::unpackVal().
    // Note: When there is a common IPC base block, this will be part of it.
    LwSciError recvHeader(IpcMsg& msgType) noexcept;

    //! \brief Imports EndInfoVector from the IpcBuffer and handles as if
    //!   it came directly from neighbor.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack data.
    //!
    //! \return LwSciError, the completion code of this operation
    //! * LwSciError_Success: If message is processed successfully.
    //! * Any error returned by ipcBufferUnpack().
    // Note: When there is a common IPC base block, need to virtualize this
    //       so it does the right thing when called from src or dst.
    LwSciError ipcRecvConnect(IpcBuffer& recvBuf) noexcept;

private:

    //
    // Common *Dst-specific functions for handling individual messages
    //

    //! \brief Pack supported elements into sendBuffer.
    //!   This is common to IPC and C2C.
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by sendHeader().
    //! - Any error returned by Elements::dataPack().
    LwSciError sendSupportedElements(IpcBuffer& sendBuf) noexcept;

    //! \brief Pack the runtime readiness message.
    //!   This is used for both IPC and C2C.
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //!   (Lwrrently unused)
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by sendHeader().
    LwSciError sendRuntimeReady(IpcBuffer& sendBuf) noexcept;

    //! \brief Pack the disconnect message.
    //!   This is used for both IPC and C2C.
    //!   (Lwrrently unused)
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by sendHeader().
    LwSciError sendDisconnect(IpcBuffer& sendBuf) noexcept;

    //! \brief Unpack allocated elements from recvBuffer and send downstream.
    //!   This is common to IPC and C2C.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by Elements::dataUnpack().
    //! - Any error returned by Waiters::sizeInit().
    //! - Any error returned by Signals::sizeInit().
    //! - Any error returned by Elements::dataSend().
    LwSciError recvAllocatedElements(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpack runtime beginning message from recvBuffer and send
    //!   downstream.
    //!   This is used for both IPC and C2C.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!   (Lwrrently unused)
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    LwSciError recvRuntimeBegin(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpack disconnect message from recvBuffer and send downstream.
    //!   This is used for both IPC and C2C.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!   (Lwrrently unused)
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    LwSciError recvDisconnect(IpcBuffer& recvBuf) noexcept;


    //
    // IpcDst-specific functions for handling individual messages
    //

    //! \brief Packs packet status message.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] sendBuf: Buffer into which to pack the message.
    //! \param [in] pkt: Packet being created.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcSrc::sendHeader().
    //! - Any error returned by IpcBuffer::packVal().
    //! - Any error returned by Packet::statusPack().
    LwSciError ipcSendPacketStatus(IpcBuffer& sendBuf,
                                   PacketPtr const& pkt) noexcept;

    //! \brief Pack this side's sync waiter attributes into sendBuffer.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by sendHeader().
    //! - Any error returned by Waiters::pack().
    LwSciError ipcSendSyncWaiterAttr(IpcBuffer& sendBuf) noexcept;

    //! \brief Pack this side's sync signal objects into sendBuffer.
    //!   This is only used by IPC, but this may be refined.
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by sendHeader().
    //! - Any error returned by Signals::pack().
    LwSciError ipcSendSyncSignalObj(IpcBuffer& sendBuf) noexcept;

    //! \brief Packs payload available for reuse.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] sendBuf: Buffer into which to pack the message.
    //! \param [in] pkt: Packet containing payload.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcSrc::sendHeader().
    //! - Any error returned by IpcBuffer::packVal().
    //! - Any error returned by Packet::fenceConsPack().
    LwSciError ipcSendPayload(IpcBuffer& sendBuf,
                              PacketPtr const& pkt) noexcept;


    //! \brief Unpacks packet creation message from recvBuffer and sends
    //!   downstream.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - LwSciError_StreamBadPacket: An internal error oclwred and the
    //!   newly created packet could not be found.
    //! - Any error returned by IpcBuffer::unpackVal().
    //! - Any error returned by Block::pktCreate().
    //! - Any error returned by Packet::defineUnpack().
    //!
    //! \implements{20050632}
    LwSciError ipcRecvPacketCreate(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpacks packet list completion message from recvBuffer and
    //!   sends downstream.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!   (Lwrrently unused)
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    LwSciError ipcRecvPacketsComplete(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpacks packet deletion message from recvBuffer and sends
    //!   downstream
    //!   This is only used for IPC.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcBuffer::unpackVal().
    //! - LwSciError_StreamPacketInaccessible: The packet is not upstream.
    //!
    //! \implements{20050638}
    LwSciError ipcRecvPacketDelete(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpack the other side's sync waiter attributes from recvBuffer
    //!   and send downstream.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by Waiters::unpack()
    LwSciError ipcRecvSyncWaiterAttr(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpack the other side's sync signal objects (which are this
    //!   side's waiter objects) from recvBuffer and sends downstream.
    //!   This is only used by IPC, but this may be refined.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by Signals::unpack()
    LwSciError ipcRecvSyncSignalObj(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpacks producer payload from recvBuffer and sends downstream.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - LwSciError_StreamBadPacket: Packet handle is not recognized.
    //! - LwSciError_StreamPacketInaccessible: Packet is not lwrrently
    //!   upstream.
    //! - Any error returned by Packet::fenceProdUnpack().
    //!
    //! \implements{20050641}
    LwSciError ipcRecvPayload(IpcBuffer& recvBuf) noexcept;

private:
    //! \cond
    //! \brief Flag indicating whether it is a C2C block or normal IPC block.
    //!   Initialized to false when a new IpcDst block instance is created or
    //!   true when a new C2CDst block instance is created.
    bool const                   isC2CBlock;
    //! \cond

    //! \cond TIER4_SWAD

    //! \brief LwSciIpcEndpoint leading to the upstream part of the block.
    //!        Initialized from the provided LwSciIpcEndpoint
    //!        when a new IpcDst instance is created.
    LwSciIpcEndpoint             ipcEndpoint;

    //! \brief Parent LwSciSyncModule that is used to import
    //!        LwSciSync data across an IPC boundary.
    //!        Initialized from the provided LwSciSyncModule
    //!        when a new IpcDst instance is created.
    LwSciSyncModule              ipcSyncModule;

    //! \brief Parent LwSciBufModule that is used to import LwSciBuf data across
    //!        an IPC boundary. Initialized from the provided LwSciBufModule
    //!        when a new IpcDst instance is created.
    LwSciBufModule               ipcBufModule;

    //! \brief IpcComm object to send and receive message via LwSciIpc channel.
    //!        Initialized based on the provided LwSciIpcEndpoint when
    //!        a new IpcDst instance is created.
    IpcComm                      comm;

    //! \endcond //TIER4_SWAD
    //! \cond TIER4_SWUD

    //! \brief Representation of dispatched I/O thread for processing IPC
    //!        read/write messages. This thread launches when a new IpcDst
    //!        instance is created and exelwtes dispatchThreadFunc().
    std::thread                  dispatchThread;

    //! \brief Disconnect I/O thread requested. Initialized to false
    //!        when a new IpcDst instance is created.
    std::atomic<bool>            disconnectRequested;

    //! \brief Flag indicating connection message pending for send.
    //!   Initialized at construction to false.
    bool                         connectMsg;

    //! \brief Flag indicating connection readiness message has been handled.
    //!   This can lag behind the block's connection state and is used for
    //!   restricting messages.
    //!   Initialized at construction to false.
    bool                         connectReadyDone;

    //! \brief Flag indicating connection completion message has been handled.
    //!   This can lag behind the block's connection state and is used for
    //!   restricting messages.
    //!   Initialized at construction to false.
    bool                         connectStartDone;

    //! \brief Flag indicating runtime message pending for send.
    //!   Initialized at construction to false.
    bool                         runtimeReadyMsg;

    //! \brief Flag indicating runtime ready message has been handled.
    //!   This can lag behind the block's runtime state and is used for
    //!   restricting messages.
    //!   Initialized at construction to false.
    bool                         runtimeReadyDone;

    //! \brief Flag indicating disconnect message pending. Initially set to
    //! false when a new IpcDst instance is created.
    bool                         disconnectMsg;

    //! \brief Flag indicating a message has been packed into the send buffer.
    //!        Initially set to false when a new IpcSrc instance is created.
    bool                         sendBufferPacked;

    //! \endcond //TIER4_SWUD

    //! \brief Tracks list of elements supported by consumer(s).
    //!   It is initialized to default values at creation and filled
    //!   when the information is received from downstream.
    Elements                    supportedElements;

    //! \brief Tracks list of allocated elements provided by the pool.
    //!   It is initialized to default values at creation and filled
    //!   when the information arrives over the channel.
    Elements                    allocatedElements;

    //! \brief Tracks the waiter sync attributes and related info provided
    //!   by the endpoint(s) on this side of the IPC channel.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Waiters                     endSyncWaiter;

    //! \brief Tracks the waiter sync attributes and related info provided
    //!   by the other side of the IPC channel.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Waiters                     ipcSyncWaiter;

    //! \brief Tracks the signal sync objects and related info provided
    //!   by the endpoint(s) on this side of the IPC channel.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Signals                     endSyncSignal;

    //! \brief Tracks the signal sync objects and related info provided
    //!   by the other side of the IPC channel.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Signals                     ipcSyncSignal;

    //! \cond TIER4_SWAD

    //! \brief Tracks Payload messages to be sent.
    //!        Initialized to an empty queue when a new IpcDst instance
    //!        is created.
    Packet::PayloadQ             reusePayloadQueue;

    //
    // As part of our conditions for deviating from Autosar rule M5-2-8,
    //   which forbids casting between regular and void pointers, all
    //   of our data transmission must use an intermediate copy to go
    //   between our specific data structures and a generic data structure.
    //   This ensures we avoid alignment errors and overflow.
    //
    // Because we make use of these buffers for all outgoing and incoming
    //   messages, only one thread can be sending or receiving at a time.
    //! \brief Buffer used for data write. Initialized with the comm's
    //!        frame size obtained with IpcComm::getFrameSize() when
    //!        a new IpcDst instance is created.
    IpcBuffer                    sendBuffer;
    //! \brief Buffer used for data read. Initialized with the comm's
    //!        frame size obtained with IpcComm::getFrameSize() when
    //!        a new IpcDst instance is created.
    IpcBuffer                    recvBuffer;

    //! \endcond //TIER4_SWAD
};

} //namespace LwSciStream

#endif // IPCDST_H
