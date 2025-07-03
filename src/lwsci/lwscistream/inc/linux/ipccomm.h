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

namespace LwSciStream {

//! \brief IpcComm class provides API for sending and receiving data over the
//! IPC channel it associates with. The handle to the IPC channel is passed as
//! a parameter to IpcComm constructor.  It also provides API for a thread
//! to be blocked and wait on the signals of incoming messages from the
//! IPC channel, local write requests to send messages to the other side of the
//! IPC channel, or disconnection requests.
class IpcComm
{
public:
    //! \brief Constructs an instance of IpcComm and initializes
    //! all data fields.
    explicit IpcComm(LwSciIpcEndpoint const ipcHandle) noexcept;

    //! \brief Destructor detaches connection for receiving LwSciIpc signals
    //! (pulses) and for receiving internal signals (pulses).
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
    bool isInitSuccess(void) const noexcept;

    //! \brief Accessor to get IPC channel frame size from LwSciIpcEndpointInfo.
    uint32_t getFrameSize(void) const noexcept;

    //! \brief Blocks the calling thread and waits for signals (pulses) from the
    //! IPC channel (LwSciIpc events), write requests, or disconnection
    //! request.
    IpcQueryFlags waitForEvent(void) noexcept;

    //! \brief Blocks the calling thread and waits for connection-establishment
    //!  signal from the other end of IPC channel.
    LwSciError waitForConnection(void) noexcept;

    //! \brief Signals that a disconnect has been requested by the IPC block owning
    //! this IpcComm object.
    LwSciError signalDisconnect(void) const noexcept;

    //! \brief Signals that a message is available to be written over the IPC
    //! channel if IPC connection has been established, otherwise enqueues the
    //! write request.
    LwSciError signalWrite(void) noexcept;

    //! \brief Sends single frame to the other side of the IPC channel.
    LwSciError sendFrame(IpcBuffer& buffer) noexcept;

    //! \brief Reads single frame from the IPC channel.
    LwSciError readFrame(IpcBuffer& buffer) const noexcept;

    //! \brief Processes the enqueued write requests after IPC connection
    //!  has been established. This function should be called just once, after
    //!  IPC connection has been established.
    LwSciError flushWriteSignals(void) noexcept;
private:
    //! \brief ipc channel endpoint
    LwSciIpcEndpoint handle;
    //! \brief ipc channel info
    struct LwSciIpcEndpointInfo channelInfo;
    //! \brief Indicate if object was successfully constructed
    bool initSuccess;
    //
    // These are specific to Linux
    //
    //! \brief ipc primitive to wait on lwsciipc events
    int32_t ipcEventNotify;
    //! \brief internal primitive to wait on write requests
    int32_t internalWriteNotify;
    //! \brief internal primitive to wait on disconnect requests
    int32_t internalDisconnNotify;
    //! \brief Flag indicating dispatch thread has detected message to write
    bool internalWritePending;
};

} // namespace LwSciStream
#endif // IPC_COMM_H
