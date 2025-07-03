//! \file
//! \brief LwSciStream ipc endpoint class declaration.
//!
//! \copyright
//! Copyright (c) 2019-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef IPC_COMM_H
#define IPC_COMM_H
#include <cstdint>
#include <atomic>
#include <utility>
#include "covanalysis.h"
#include "ipccomm_common.h"
#include "ipcbuffer.h"

/**
 * @defgroup lwscistream_blanket_statements LwSciStream blanket statements.
 * Generic statements applicable for LwSciStream interfaces.
 * @{
 */

/**
 * \page lwscistream_page_blanket_statements LwSciStream blanket statements
 *
 * \section lwscistream_transport_interaction Interaction for Ipc transport
 *   LwSciStream calls the following QNX_System interfaces:
 *    - ChannelCreate_r() for creating a QNX channel
 *    - ConnectAttach_r() to establish connection with the QNX channel to send and receive message pulses.
 *    - MsgSendPulse_r() to send a message pulse to the QNX channel when a new message ready to be written to the QNX channel.
 *    - MsgReceivePulse_r() to receive a message pulse from the QNX channel when a new message is available to read.
 *
 * \section lwscistream_transport_interaction Interaction for Ipc transport
 *   LwSciStream calls the following LwSciIpc interfaces:
 *    - LwSciIpcSetQnxPulseParam() to set PulseParam for the connection.
 *    - LwSciIpcGetEndpointInfo() to get the LwSciIpcEndpointInfo.
 *    - LwSciIpcGetEvent() to get the events from LwSciIpc channel.
 *    - LwSciIpcRead() to read message from LwSciIpc channel.
 *    - LwSciIpcWrite() to write message to LwSciIpc channel.
 */

/**
 * @}
 */

namespace LwSciStream {

//! \brief IpcComm class provides API for sending and receiving data over the
//! IPC channel it associates with. The handle to the IPC channel is passed as
//! a parameter to IpcComm constructor.  It also provides API for a thread
//! to be blocked and wait on the signals of incoming messages from the
//! IPC channel, local write requests to send messages to the other side of the
//! IPC channel, or disconnection requests.
//!
//! It interacts with the IPC channel using LwSciIpc API, which is
//! platform-dependent. IpcComm abstracts the operations with LwSciIpc API
//! and provides platform-agnostic API to its clients.
//!
//! \if TIER4_SWAD
//! \implements{19700304}
//! \endif
//!
//! \if TIER4_SWUD
//! \implements{21079761}
//! \endif
class IpcComm final
{
public:
    //! \brief Constructs an instance of IpcComm and initializes
    //! all data fields.
    //!
    //! \param [in] ipcHandle: Handle to LwSciIpcEndpoint set by Application.
    //!
    //! \if TIER4_SWAD
    //! \implements{19866621}
    //! \endif
    explicit IpcComm(LwSciIpcEndpoint const ipcHandle) noexcept;

    //! \brief Destructor detaches connection for receiving LwSciIpc signals
    //! (pulses) and for receiving internal signals (pulses).
    //!
    //! \implements{19867794}
    ~IpcComm(void) noexcept;
    // Special functions not in use
    IpcComm() noexcept = delete;
    IpcComm(const IpcComm& another) noexcept              = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    IpcComm(IpcComm&& another) noexcept                   = delete;
    IpcComm& operator=(const IpcComm& another) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    IpcComm& operator=(IpcComm&& another) & noexcept      = delete;

    //! \brief Accessor to check if constructor encountered any errors.
    //!
    //! \return bool
    //! - true: Creation successful.
    //! - false: Creation failed.
    //!
    //! \implements{19867587}
    inline bool isInitSuccess(void) const noexcept
    {
        return initSuccess;
    }

    //! \brief Accessor to get IPC channel frame size from LwSciIpcEndpointInfo.
    //!
    //! \return uint32_t, frame size of the IPC channel.
    //!
    //! \implements{19867656}
    uint32_t getFrameSize(void) const noexcept;

    //! \brief Blocks the calling thread and waits for signals (pulses) from the
    //! IPC channel (LwSciIpc events), write requests, or disconnection
    //! request.
    //!
    //! \return IpcQueryFlags, containing the completion code of this operation,
    //! and the actions notified.
    //! Possible error code that can be set to err member in case of failure:
    //! - LwSciError_Success: Indicates a successful operation.
    //! - Any error code/panic behavior that LwSciIpcGetEvent() or
    //!   MsgReceivePulse_r() can generate.
    //! - LwSciError_InsufficientResource: If number of pending write requests
    //!   becomes UINT32_MAX.
    //!
    //! \if TIER4_SWAD
    //! \implements{19867689}
    //! \endif
    IpcQueryFlags waitForEvent(void) noexcept;

    //! \brief Blocks the calling thread and waits for connection-establishment
    //!  signal from the other end of IPC channel.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: Indicates a successful operation.
    //! - Any error code/panic behavior that LwSciIpcGetEvent() or
    //!   MsgReceivePulse_r() can generate.
    //!
    //! \if TIER4_SWAD
    //! \implements{19867692}
    //! \endif
    LwSciError waitForConnection(void) noexcept;

    //! \brief Signals that a disconnect has been requested by the IPC block owning
    //! this IpcComm object.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: Indicates a successful operation.
    //! - Any error code/panic behavior that MsgSendPulse_r() can generate.
    //!
    //! \if TIER4_SWAD
    //! \implements{19867767}
    //! \endif
    LwSciError signalDisconnect(void) const noexcept;

    //! \brief Signals that a message is available to be written over the IPC
    //! channel if IPC connection has been established, otherwise enqueues the
    //! write request.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: Indicates a successful operation.
    //! - Any error code/panic behavior that MsgSendPulse_r() can generate.
    //!
    //! \if TIER4_SWAD
    //! \implements{19867779}
    //! \endif
    LwSciError signalWrite(void) noexcept;

    //! \brief Sends a message to the other side of the IPC channel.
    //!
    //! \param [in] buffer: Buffer containing packed data to send.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: message was sent successfully.
    //! - LwSciError_StreamInternalError: Message to send is too big to fit
    //!   into one IPC frame or no message is pending to write.
    //! - Any error returned by IpcBuffer::changeMode,
    //!   IpcBuffer::sendInfoGet(), or IpcBuffer::sendSizeAdd().
    //! - Any error code/panic behavior that LwSciIpcWrite() can generate.
    //!
    //! \if TIER4_SWAD
    //! \implements{19867782}
    //! \endif
    LwSciError sendFrame(IpcBuffer& buffer) noexcept;

    //! \brief Reads single frame from the IPC channel.
    //!
    //! \param [in] buffer: Destination buffer to read IPC frame into.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: Message was read successfully.
    //! - Any error returned by IpcBuffer::changeMode,
    //!   IpcBuffer::recvInfoGet(), or IpcBuffer::recvSizeAdd().
    //! - Any error code/panic behavior that LwSciIpcRead() can generate.
    //!
    //! \if TIER4_SWAD
    //! \implements{19867785}
    //! \endif
    LwSciError readFrame(IpcBuffer& buffer) const noexcept;

    //! \brief Processes the enqueued write requests after IPC connection
    //!  has been established. This function should be called just once, after
    //!  IPC connection has been established.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: Indicates a successful operation.
    //! - Any error code/panic behavior that MsgSendPulse_r() can generate.
    //!
    //! \if TIER4_SWAD
    //! \implements{19867788}
    //! \endif
    LwSciError flushWriteSignals(void) noexcept;
private:
    //! \cond TIER4_SWAD
    //! \brief Handle to the endpoint of the IPC channel this object sends
    //! data to and receives data from. It is set in the constructor and the
    //! value is set to the handle value passed in as a parameter.
    LwSciIpcEndpoint handle;
    //! \brief Info of the IPC channel endpoint referred by handle.
    //! It is set in the constructor. The LwSciIpcEndpointInfo corresponding to
    //! handle is retrieved by LwSciIpcGetEndpointInfo(), and stored in this
    //! variable.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A2_7_3), "Bug 3261115")
    struct LwSciIpcEndpointInfo channelInfo;
    //! \brief Indicates if object was successfully constructed.
    //! Its value will be set to true in the constructor finishes without
    //! error; otherwise the value will be false.
    bool initSuccess;
    //! \endcond

    //! \cond TIER4_SWUD
    //! \brief Id of the QNX Channel used by this IpcComm object.
    //! It will have two QNX Connections, identified by qnxChannelId
    //! and ipcEventConnId. It allows the dispatch thread of IpcSrc and IpcDst
    //! to be blocked on waiting for the pulses coming from the QNX
    //! Connections. The QNX Channel is opened by ChannelCreate_r() in the
    //! constructor and its id stored in qnxChannelId, and it is closed by
    //! ChannelDestroy() in the destructor.
    int32_t qnxChannelId;
    //! \brief Id of the QNX Connection to be attached to the QNX Channel
    //! referred by qnxChannelId. The IPC channel endpoint will use it to
    //! notify the waiter that a LwSciIpc event is available. When a LwSciIpc
    //! event is available the QNX Channel will get a pulse.
    //! The QNX Connection is opened and attached to the QNX Channel by
    //! ConnectAttach_r() in the constructor, and closed by ConnectDetach() in
    //! the destructor.
    //!
    //! Note: LwSciIpc notifies the waiter in an edge-triggered manner. For
    //!       each type of LwSciIpc event, a pulse is generated only when an
    //!       event becomes available from being unavailable.
    int32_t ipcEventConnId;
    //! \brief Id of the QNX Connection to be attached to the QNX Channel
    //! referred by qnxChannelId. IpcSrc or IpcDst block owning this
    //! IpcComm object will send a pulse to the QNX Channel when there is
    //! a new message ready to be written to the IPC channel, or block
    //! disconnection happens.
    //! The QNX Connection is opened and attached to the QNX Channel by
    //! ConnectAttach_r() in the constructor, and closed by ConnectDetach() in
    //! the destructor.
    int32_t internalInterruptId;
    //! \brief Flag to indicate that the IPC connection is established, and it
    //! is safe to send write pulses to the QNX Channel, identified by
    //! qnxChannelId. It is initialized to false in the constructor. Once the
    //! IPC connection with other end is established, it is set to true.
    std::atomic<bool> connEstablished;
    //! \brief Number of write pulses aclwmulated before the IPC connection is
    //! established. It is initialized to zero in the constructor. Once
    //! connEstablished becomes true the unsent write pulses
    //! will be sent to the QNX Channel, afterward this variable won't be used
    //! again.
    uint32_t numUnsentSignals;
    //! \brief Number of internal write requests received in the dispatch
    //! thread (spawned by the IpcSrc or IpcDst block owning this object).
    //! It is initialized to zero in the constructor, and increments every
    //! time the dispatch thread gets a write request.
    uint32_t numWritePending;
    //! \endcond
};

} // namespace LwSciStream
#endif // IPC_COMM_H
