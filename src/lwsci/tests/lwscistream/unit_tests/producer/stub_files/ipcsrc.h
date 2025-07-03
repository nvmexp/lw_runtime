//! \file
//! \brief LwSciStream IPC source block declaration.
//!
//! \copyright
//! Copyright (c) 2019-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef IPCSRC_H
#define IPCSRC_H
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
/**
 * @defgroup lwscistream_blanket_statements LwSciStream blanket statements.
 * Generic statements applicable for LwSciStream interfaces.
 * @{
 */

/**
 * \page lwscistream_page_blanket_statements LwSciStream blanket statements
 * \section lwscistream_transport_interaction Interaction for Ipc transport
 *   LwSciStream calls the following LwSciBuf and LwSciSync interfaces:
 *    - LwSciBufAttrListIpcExportReconciled(), LwSciBufObjIpcExport(),
 *      LwSciSyncAttrListIpcExportUnreconciled(), LwSciSyncIpcExportAttrListAndObj(),
 *      LwSciSyncIpcExportFence(), LwSciBufAttrListIpcExportUnreconciled()
 *      for exporting appropriate LwSciBuf and LwSciSync structures,
 *      LwSciBufAttrListIpcImportUnreconciled(), LwSciSyncAttrListIpcImportUnreconciled(),
 *      LwSciSyncIpcImportAttrListAndObj(), LwSciSyncIpcImportFence(),
 *      LwSciBufAttrListIpcImportReconciled(), LwSciBufObjIpcImport()
 *      for importing appropriate LwSciSync and LwSciBuf structures,
 *      LwSciBufAttrListFreeDesc(), LwSciSyncAttrListFreeDesc(),
 *      LwSciSyncAttrListAndObjFreeDesc() for freeing LwSciSync and LwSciBuf
 *      export descriptors.
 * \section lwscistream_transport_interaction Interaction for Ipc transport
 *   LwSciStream triggers error events on the following conditions
 *   during Ipc transport:
 *    - Any SrcBlockInterface interface overridden by IpcDst triggers
 *      error events if it fails to signal that a message is available to
 *      be transmitted upstream.
 *    - Any DstBlockInterface interface overridden by IpcSrc trigger error
 *      events if it fails to signal that a message is available to be
 *      transmitted downstream.
 * \section lwscistream_conlwrrency Conlwrrency
 * - In IpcSrc and IpcDst, incoming data from the IPC channel is only handled
 *   by the dispatch thread. No conlwrrent access on the incoming data or the
 *   class members used in processing the data.
 */

/**
 * @}
 */

//! \brief IPC source block is the upstream half of an IPC block pair which
//!  allows packets to be transmitted between processes.
//!
//! IpcSrc blocks have one normal input, and for their output must be
//! coupled with a corresponding IPC destination block. IpcSrc blocks act
//! like downstream blocks, overriding DstBlockInterface functions.
//!
//! \if TIER4_SWAD
//! \implements{19676004}
//! \endif
//! \if TIER4_SWUD
//! \implements{20283714}
//! \endif
class IpcSrc :
    public Block
{
public:
    IpcSrc(void) noexcept                        = delete;
    IpcSrc(const IpcSrc &) noexcept              = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    IpcSrc(IpcSrc &&) noexcept                   = delete;
    IpcSrc& operator=(const IpcSrc &) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    IpcSrc& operator=(IpcSrc &&) & noexcept      = delete;

    //! \brief Constructs an instance of the class and
    //!        initializes all data fields for the new IpcSrc object.
    //!
    //! \param [in] ipc: LwSciIpcEndpoint to be used for communication.
    //! \param [in] syncModule: Instance of LwSciSyncModule.
    //! \param [in] bufModule: Instance of LwSciBufModule.
    //! \param [in] isC2C: Flag indicating C2CSrc block.
    //!
    //! \if TIER4_SWAD
    //! \implements{19731420}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
    explicit IpcSrc(LwSciIpcEndpoint const ipc,
                    LwSciSyncModule const syncModule,
                    LwSciBufModule const bufModule,
                    bool const isC2C = false) noexcept;

    //! \brief Frees any LwSciBuf and or LwSciSync handles
    //!        still referenced by the IpcSrc block and deletes the instance.
    //!        Destructor is called when refcount
    //!        to this object has reached zero.
    //!
    //! \if TIER4_SWAD
    //! \implements{19731423}
    //! \endif
    ~IpcSrc(void) noexcept override;

    // Override functions inherited from APIBlockInterface

    //! \brief Disconnects upstream blocks.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: Always, as this operation cannot fail.
    //! - Triggers the following error events:
    //!     - For any error code that IpcComm::signalDisconnect() can generate
    //!       in case of failure.
    //!
    //! \if TIER4_SWAD
    //! \implements{19731426}
    //! \endif
    LwSciError disconnect(void) noexcept final;

    //! \brief Receives producer info from upstream, saves, and signals
    //!   it is ready to send downstream.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvProdInfo
    //!
    //! \return void
    void srcRecvProdInfo(
        uint32_t const srcIndex,
        EndInfoVector const& info) noexcept final;

    //! \brief Receives alllocated element information from pool and
    //!   saves for transmission over IPC.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvAllocatedElements
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Elements::dataCopy().
    //! - Any error returned by Waiters::initSize().
    //! - Any error returned by IpcComm::signalWrite().
    void srcRecvAllocatedElements(
        uint32_t const srcIndex,
        Elements const& inElements) noexcept override;

    //! \brief Saves new packet for transmission to IpcDst.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPacketCreate
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Block::pktCreate().
    //! - Any error returned by IpcComm::signalWrite().
    void srcRecvPacketCreate(
        uint32_t const srcIndex,
        Packet const& origPacket) noexcept override;

    //! \brief Queues message to send packet deletion to IpcDst.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPacketDelete
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: Packet handle not found in map.
    //! - LwSciError_StreamPacketInaccessible: Packet is not lwrrently
    //!   upstream of the IPC block.
    //! - LwSciError_StreamPacketDeleted: Packet was already marked deleted.
    //! - Any error returned by IpcComm::signalWrite().
    void srcRecvPacketDelete(
        uint32_t const srcIndex,
        LwSciStreamPacket const handle) noexcept final;

    //! \brief Queues message to send packet completion to IpcDst.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPacketsComplete
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_AlreadyDone: Packet set was already marked completed.
    //! - Any error returned by IpcComm::signalWrite().
    void srcRecvPacketsComplete(
        uint32_t const srcIndex) noexcept final;

    //! \brief Queues message to send LwSciSync waiter information from
    //!   the producer to IpcDst.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvSyncWaiter
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Waiters::copy().
    //! - Any error returned by IpcComm::signalWrite().
    void srcRecvSyncWaiter(
        uint32_t const srcIndex,
        Waiters const& syncWaiter) noexcept override;

    //! \brief Queues message to send LwSciSync signal information from
    //!   the producer to IpcDst.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvSyncSignal
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Signals::copy().
    //! - Any error returned by IpcComm::signalWrite().
    void srcRecvSyncSignal(
        uint32_t const srcIndex,
        Signals const& syncSignal) noexcept override;

    //! \brief Queues message to send Payload to IpcDst.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPayload
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: The packet is invalid.
    //! - LwSciError_StreamPacketInaccessible: The packet is not upstream.
    //! - Any error returned by Packet::fenceProdCopy().
    //! - Any error returned by IpcComm::signalWrite().
    void srcRecvPayload(
        uint32_t const srcIndex,
        Packet const& prodPayload) noexcept override;

    //! \brief Disconnects from the upstream blocks and informs IpcDst of the
    //!        upstream disconnect.
    //! \copydetails LwSciStream::DstBlockInterface::srcDisconnect
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19731462}
    //! \endif
    void srcDisconnect(
        uint32_t const srcIndex) noexcept final;

    //! \brief Launches the dispatch thread for sending and receiving IPC messages.
    //!
    //! \return bool, true if thread launched successfully, false if failed.
    //!
    bool startDispatchThread(void) noexcept;


    //! \brief Queue message to send phase change to IpcDst.
    //!
    //! \return void
    void phaseSendChange(void) noexcept final;

protected:
    //! \brief Gets LwSciSyncModule that the LwSciSync data associated with.
    //!
    //! \return LwSciSyncModule
    LwSciSyncModule getSyncModule(void) const noexcept
    {
        return ipcSyncModule;
    };

    //! \brief Gets LwSciBufModule that the LwSciBuf data associated with.
    //!
    //! \return LwSciBufModule
    LwSciBufModule getBufModule(void) const noexcept
    {
        return ipcBufModule;
    };

    //! \brief Gets LwSciIpcEndpoint that used for data export/import.
    //!
    //! \return LwSciIpcEndpoint
    LwSciIpcEndpoint getIpcEndpoint(void) const noexcept
    {
        return ipcEndpoint;
    };

    //! \brief Signals that a message is available to be written.
    //!
    //! \return LwSciError, any error from IpcComm::signalWrite().
    LwSciError enqueueIpcWrite(void) noexcept
    {
        return comm.signalWrite();
    }

    //! \brief Retrieve constant reference to the allocated elements.
    //!   Caller is expected to make sure it is complete before using it.
    //!
    //! \return Elements reference, the allocated elements.
    Elements const& allocatedElemGet(void) const noexcept
    {
        return allocatedElements;
    };

#if (!C2C_EVENT_SERVICE)
    void enqueue(PacketPtr const&  newPacket) noexcept
    {
        readyPayloadQueue.enqueue(newPacket);
    };

    PacketPtr dequeue(void) noexcept
    {
        return readyPayloadQueue.dequeue();
    };
#endif //(!C2C_EVENT_SERVICE)

    //! \brief Dequeue all packets from the readyPayloadQueue
    void dequeueAll(void) noexcept;
private:

    //
    // Top level message handling operations
    //

    // I/O Thread loop for processing Ipc read/write messages.
    void dispatchThreadFunc(void) noexcept;

    // Exits the I/O thread loop.
    void destroyIOLoop(bool const wait=false) noexcept;

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

    //! \brief Queries whether runtime change message has been sent.
    //!   Prior to this, init phase messages may still be pending,
    //!   and should be sent before telling the downstream block to
    //!   transition to runtime.
    //!
    //! \return bool, the runtimeBeginDone flag
    bool runtimeEnabled(void) const noexcept
    {
        return runtimeBeginDone;
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
    //! - Any error returned by sendAllocatedElements().
    //! - Any error returned by ipcSendPacketCreate().
    //! - Any error returned by ipcSendPacketsComplete().
    //! - Any error returned by ipcSendSyncWaiterAttr().
    //! - Any error returned by ipcSendSyncSignalObj().
    //! - Any error returned by sendRuntimeBegin().
    //! - Any error returned by ipcSendPayload().
    //! - Any error returned by ipcSendPacketDelete().
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
    //! - Any error returned by recvSupportedElements().
    //! - Any error returned by ipcRecvPacketStatus().
    //! - Any error returned by ipcRecvSyncWaiterAttr().
    //! - Any error returned by ipcRecvSyncSignalObj().
    //! - Any error returned by recvRuntimeReady().
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
    //! - LwSciError_Success: If message is processed successfully.
    //! - Any error returned by ipcBufferUnpack().
    // Note: When there is a common IPC base block, need to virtualize this
    //       so it does the right thing when called from src or dst.
    LwSciError ipcRecvConnect(IpcBuffer& recvBuf) noexcept;

private:

    //
    // Common *Src-specific functions for handling individual messages
    //

    //! \brief Pack allocated elements into sendBuffer.
    //!   This is common to IPC and C2C.
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by sendHeader().
    //! - Any error returned by Elements::dataPack().
    LwSciError sendAllocatedElements(IpcBuffer& sendBuf) noexcept;

    //! \brief Pack the runtime beginning message.
    //!   This is used for both IPC and C2C.
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //!   (Lwrrently unused)
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by sendHeader().
    LwSciError sendRuntimeBegin(IpcBuffer& sendBuf) noexcept;

    //! \brief Pack the disconnect message.
    //!   This is used for both IPC and C2C.
    //!
    //! \param [in,out] sendBuf: Buffer in which to pack the message.
    //!   (Lwrrently unused)
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by sendHeader().
    LwSciError sendDisconnect(IpcBuffer& sendBuf) noexcept;

    //! \brief Unpack supported elements from recvBuffer and send upstream.
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
    LwSciError recvSupportedElements(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpack runtime readiness message from recvBuffer and send
    //!   upstream.
    //!   This is used for both IPC and C2C.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!   (Lwrrently unused)
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    LwSciError recvRuntimeReady(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpack disconnect message from recvBuffer and send upstream.
    //!   This is used for both IPC and C2C.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!   (Lwrrently unused)
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    LwSciError recvDisconnect(IpcBuffer& recvBuf) noexcept;


    //
    // IpcSrc-specific functions for handling individual messages
    //

    //! \brief Packs packet creation message.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] sendBuf: Buffer into which to pack the message.
    //! \param [in] pkt: Packet being created.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcSrc::sendHeader().
    //! - Any error returned by IpcBuffer::packVal().
    //! - Any error returned by Packet::definePack().
    LwSciError ipcSendPacketCreate(IpcBuffer& sendBuf,
                                   PacketPtr const& pkt) noexcept;

    //! \brief Packs packet list completion message.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] sendBuf: Buffer into which to pack the message.
    //!   (Lwrrently unused)
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcSrc::sendHeader().
    LwSciError ipcSendPacketsComplete(IpcBuffer& sendBuf) noexcept;

    //! \brief Packs packet deletion message.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] sendBuf: Buffer into which to pack the message.
    //! \param [in] pkt: Packet being deleted.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcSrc::sendHeader().
    //! - Any error returned by IpcBuffer::packVal().
    LwSciError ipcSendPacketDelete(IpcBuffer& sendBuf,
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

    //! \brief Packs payload from the producer.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] sendBuf: Buffer into which to pack the message.
    //! \param [in] pkt: Packet containing payload.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by IpcSrc::sendHeader().
    //! - Any error returned by IpcBuffer::packVal().
    //! - Any error returned by Packet::fenceProdPack().
    LwSciError ipcSendPayload(IpcBuffer& sendBuf,
                              PacketPtr const& pkt) noexcept;


    //! \brief Unpacks packet status message from recvBuffer and sends
    //!   upstream.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - LwSciError_StreamBadPacket: The packet handle was not recognized.
    //! - Any error returned by IpcBuffer::unpackVal().
    //! - Any error returned by Packet::statusConsUnpack().
    LwSciError ipcRecvPacketStatus(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpack the other side's sync waiter attributes from recvBuffer
    //!   and send upstream.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by Waiters::unpack()
    LwSciError ipcRecvSyncWaiterAttr(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpack the other side's sync signal objects (which are this
    //!   side's waiter objects) from recvBuffer and sends upstream.
    //!   This is only used by IPC, but this may be refined.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - Any error returned by Signals::unpack()
    LwSciError ipcRecvSyncSignalObj(IpcBuffer& recvBuf) noexcept;

    //! \brief Unpacks reusable payload from recvBuffer and sends upstream.
    //!   This is only used for IPC.
    //!
    //! \param [in,out] recvBuf: Buffer from which to unpack the message.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: The operation completed successfully.
    //! - LwSciError_StreamBadPacket: Packet handle is not recognized.
    //! - LwSciError_StreamPacketInaccessible: Packet is not lwrrently
    //!   downstream.
    //! - Any error returned by Packet::fenceConsUnpack().
    //!
    //! \implements{20050578}
    LwSciError ipcRecvPayload(IpcBuffer& recvBuf) noexcept;

private:
    //! \cond
    //! \brief Flag indicating whether it is a C2C block or normal IPC block.
    //!   Initialized to false when a new IpcSrc block instance is created or
    //!   true when a new C2CSrc block instance is created.
    bool const                   isC2CBlock;
    //! \cond

    //! \cond TIER4_SWAD

    //! \brief LwSciIpcEndpoint leading to the downstream part of the block.
    //!        Initialized from the provided LwSciIpcEndpoint when
    //!        a new IpcSrc instance is created.
    LwSciIpcEndpoint             ipcEndpoint;

    //! \brief Parent LwSciSyncModule that is used to import
    //!        LwSciSync data across an IPC boundary.
    //!        Initialized from the provided LwSciSyncModule when
    //!        a new IpcSrc instance is created.
    LwSciSyncModule              ipcSyncModule;

    //! \brief Parent LwSciBufModule that is used to import LwSciBuf data across
    //!        an IPC boundary. Initialized from the provided LwSciBufModule
    //!        when a new IpcSrc instance is created.
    LwSciBufModule               ipcBufModule;

    //! \brief IpcComm object to send and receive message via
    //!        LwSciIpc channel. Initialized based on the provided
    //!        LwSciIpcEndpoint when a new IpcSrc instance is created.
    IpcComm                      comm;

    //! \endcond //TIER4_SWAD
    //! \cond TIER4_SWUD

    //! \brief Representation of dispatched I/O thread for processing IPC
    //!        read/write messages. This thread launches when a new IpcSrc
    //!        instance is created and exelwtes dispatchThreadFunc().
    std::thread                  dispatchThread;

    //! \brief Disconnect I/O thread requested. Initialized to false
    //!        when a new IpcSrc instance is created.
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
    bool                         runtimeBeginMsg;

    //! \brief Flag indicating runtime start message has been handled.
    //!   This can lag behind the block's runtime state and is used for
    //!   restricting messages.
    //!   Initialized at construction to false.
    bool                         runtimeBeginDone;

    //! \brief Flag indicating disconnect message pending. Initially set to
    //!        false when a new IpcSrc instance is created.
    bool                         disconnectMsg;

    //! \brief Flag indicating a message has been packed into the send buffer.
    //!        Initially set to false when a new IpcSrc instance is created.
    bool                         sendBufferPacked;

    //! \endcond //TIER4_SWUD

    //! \brief Tracks list of elements allocated by pool.
    //!   It is initialized to default values at creation and filled
    //!   when the information is received from upstream.
    Elements                    allocatedElements;

    //! \brief Flag indicating pool has indicated packet list completion.
    //!   It is intialized to false at creation.
    std::atomic<bool>           allocatedPacketExportDone;

    //! \brief Flag indicating packet completion event is pending.
    //!   It is intialized to false at creation.
    // TODO: Probably doesn't need to be atomic
    std::atomic<bool>           allocatedPacketExportEvent;

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

    //! \brief Tracks payload messages to be sent.
    //!        Initialized to an empty queue when a new IpcSrc instance
    //!        is created.
    Packet::PayloadQ             readyPayloadQueue;

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
    //!        a new IpcSrc instance is created.
    IpcBuffer                    sendBuffer;

    //! \brief Buffer used for data read. Initialized with the comm's
    //!        frame size obtained with IpcComm::getFrameSize() when
    //!        a new IpcSrc instance is created.
    IpcBuffer                    recvBuffer;

    //! \endcond //TIER4_SWAD
};

} //namespace LwSciStream

#endif // IPCSRC_H
