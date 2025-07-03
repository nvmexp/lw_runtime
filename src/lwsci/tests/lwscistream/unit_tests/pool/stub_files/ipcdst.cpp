//! \file
//! \brief LwSciStream IPC destination block definition.
//!
//! \copyright
//! Copyright (c) 2019-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.
#include <cstddef>
#include <iostream>
#include <array>
#include <array>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <cmath>
#include <vector>
#include <functional>
#include <utility>
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "covanalysis.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "lwscistream_common.h"
#include "sciwrap.h"
#include "lwscistream_types.h"
#include "lwscicommon_os.h"
#include "trackarray.h"
#include "safeconnection.h"
#include "ipccomm_common.h"
#include "ipcbuffer.h"
#include "packet.h"
#include "syncwait.h"
#include "ipccomm.h"
#include "ipcdst.h"

namespace LwSciStream {

//! \brief I/O Thread loop for processing Ipc read/write messages.
//!
//! <b>Sequence of operations</b>
//! - Waits for connection by calling IpcComm::waitForConnection() on comm.
//! - Under thread protection provided by Block::blkMutexLock() flushes all
//!   write signals from it with IpcComm::flushWriteSignals().
//! - In a loop: exits if disconnectRequested is true and disconnectMsg
//!   is false. Otherwise waits for event by calling IpcComm::waitForEvent()
//!   on comm.
//! - If the event indicates write ready, calls IpcDst::processWriteMsg()
//!   if there is not already a packed message waiting to be sent. If there
//!   is a message waiting to send, calls IpcComm::sendFrame to send as
//!   much as possible. If the entire message was sent, call
//!   IpcBuffer::changeMode() to Idle the sendBuffer, and pack a new
//!   message if there is one available.
//! - If the event indicates read ready, calls IpcComm::readFrame() to read
//!   a frame's worth of data. If the entire message has been received,
//!   calls IpcDst::processReadMsg(), and then calls IpcBuffer::changeMode()
//!   to Idle the recvBuffer.
//! - Proceeds to the next loop iteration.
//!
//! \return void
//! - Triggers the following error events:
//!     - For any error codes that IpcComm::waitForConnection(),
//!       IpcComm::flushWriteSignals(), IpcComm::waitForEvent(),
//!       IpcComm::sendFrame(), or IpcComm::readFrame can generate.
//!     - For any error codes that IpcDst::processWriteMsg() or
//!       IpcDst::processReadMsg() can generate.
//!     - For any error codes that IpcBuffer::changeMode() can generate.
//!
//! \implements{19865832}
void
IpcDst::dispatchThreadFunc(void) noexcept
{
    LwSciError err { comm.waitForConnection() };

    if (LwSciError_Success == err) {
        // Flush the write signals that were queued up before the connection
        // is established.
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        Lock const blockLock { blkMutexLock() };
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
        err = comm.flushWriteSignals();
    }

    // process ipc IO
    while (LwSciError_Success == err) {

        // If disconnect is requested, then take down loop once any pending
        //   disconnect message (which should be the last pending message)
        //   has been sent.
        // I.e. drain write message queue before servicing disconnect
        if (disconnectRequested.load() && !disconnectMsg) {
            break;
        }

        // query available events on ipc channel
        IpcQueryFlags const queryResult { comm.waitForEvent() };
        err = queryResult.err;

        // Note: Because the buffers used for reading/writing data are
        //   part of the IPC object, the read/write functions must only
        //   be called by one thread at a time. If they are ever made
        //   directly available to applications through the proposed
        //   LwSci common event handler, we will need a mutex to protect
        //   these calls, but not the block mutex, because we ilwoke
        //   inter-block calls during these operations.

        // If there isn't already a message packed and ready to send,
        //   check for one and pack it
        if ((LwSciError_Success == err) && !sendBufferPacked) {
            err = processWriteMsg();
        }

        // Send next frame of any pending message if space availble
        if ((LwSciError_Success == err) &&
            (sendBufferPacked && queryResult.writeReady)) {

            // Send as much as we can
            err = comm.sendFrame(sendBuffer);

            // If full message sent, get ready for the next one
            if ((LwSciError_Success == err) && sendBuffer.sendDone()) {
                err = sendBuffer.changeMode(IpcBuffer::UserMode::Idle);
                sendBufferPacked = false;

                // If there's another message to send, pack it right away
                //   so its ready when the next IPC frame is available
                if (LwSciError_Success == err) {
                    err = processWriteMsg();
                }
            }
        }

        // Process incoming frames
        if ((queryResult.readReady) && (LwSciError_Success == err)) {

            // Read the next frame
            err = comm.readFrame(recvBuffer);

            // If full message received, process it
            if ((LwSciError_Success == err) && recvBuffer.recvDone()) {
                err = processReadMsg();
                if (LwSciError_Success == err) {
                    err = recvBuffer.changeMode(IpcBuffer::UserMode::Idle);
                }
            }
        }
    }

    // report error if any and exit
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        setErrorEvent(err);
        LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    }
}

//! <b>Sequence of operations</b>
//!  - It initializes the base Block class with BlockType::IPCDST,
//!    one supported source connection, one supported destination connection and
//!    flag indicating source block presents in other process.
//!
//! \implements{19791573}
//!
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A12_1_5), "Bug 2989775")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A3_1_1), "Bug 2758535")
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
IpcDst::IpcDst(LwSciIpcEndpoint const ipc,
               LwSciSyncModule const syncModule,
               LwSciBufModule const bufModule,
               bool const isC2C) noexcept :
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A3_1_1))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A12_1_5))
LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    Block(BlockType::IPCDST, 1U, 1U, true, false),
    isC2CBlock(isC2C),
    ipcEndpoint(ipc),
    ipcSyncModule(syncModule),
    ipcBufModule(bufModule),
    comm(ipc),
    dispatchThread(),
    disconnectRequested(false),
    connectMsg(false),
    connectReadyDone(false),
    connectStartDone(false),
    runtimeReadyMsg(false),
    runtimeReadyDone(false),
    disconnectMsg(false),
    sendBufferPacked(false),
    supportedElements(FillMode::Copy, FillMode::None),
    allocatedElements(FillMode::IPC, FillMode::IPC),
    endSyncWaiter(FillMode::Copy, true),
    ipcSyncWaiter(FillMode::IPC, false),
    endSyncSignal(FillMode::Copy, nullptr),
    ipcSyncSignal(FillMode::IPC, &endSyncWaiter),
    reusePayloadQueue(),
    sendBuffer(static_cast<size_t>(comm.getFrameSize()),
               ipc, syncModule, bufModule, isC2C),
    recvBuffer(static_cast<size_t>(comm.getFrameSize()),
               ipc, syncModule, bufModule, isC2C)
{
    if (!comm.isInitSuccess()) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        setInitFail();
        return;
    }
    if (!sendBuffer.isInitSuccess() ||
        !recvBuffer.isInitSuccess()) {
        setInitFail();
        return;
    }

    // Set up packet description
    Packet::Desc desc { };
    desc.initialLocation = isC2CBlock ? Packet::Location::Downstream :
                                        Packet::Location::Upstream;
    desc.defineFillMode = isC2CBlock ? FillMode::Copy : FillMode::IPC;
    desc.statusProdFillMode = isC2CBlock ? FillMode::User : FillMode::None;
    desc.statusConsFillMode = isC2CBlock ? FillMode::None : FillMode::Copy;
    desc.fenceProdFillMode = isC2CBlock ? FillMode::User : FillMode::IPC;
    desc.fenceConsFillMode = FillMode::Copy;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    desc.needBuffers = true;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    pktDescSet(std::move(desc));
}

//! <b>Sequence of operations</b>
//! - Checks if the disconnect has already been requested by instrumenting
//!   disconnectRequested with std::atomic<bool>::load().
//! - If yes, exits the dispatched I/O thread, if it is still
//!   running, by setting disconnectRequested to true with
//!   std::atomic<bool>::store() and calling
//!   IpcComm::signalDisconnect() on comm.
//! - Waits for the dispatchThread to finish with std::thread::join().
//!
//! \implements{19791576}
LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_5_2), "Approved TID-482")
IpcDst::~IpcDst(void) noexcept
{
    // Terminate thread and wait for it to complete.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    destroyIOLoop(true);
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_5_2))
LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))

//! <b>Sequence of operations</b>
//! - Sets disconnectMsg to true.
//! - Calls IpcComm::signalWrite() on comm.
//! - Calls Block::disconnectDst() on itself.
//! - Checks if the disconnect has already been
//!   requested by instrumenting disconnectRequested with
//!   std::atomic<bool>::load().
//! - If yes, exits the dispatched I/O thread,
//!   if it is still running, by setting disconnectRequested to true with
//!   std::atomic<bool>::store()and calling IpcComm::signalDisconnect() on comm.
//! - Waits for the dispatchThread to finish with std::thread::join().
//!
//! \implements{19791579}
LwSciError IpcDst::disconnect(void) noexcept
{
    // signal disconnect upstream
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    disconnectMsg = true;
    static_cast<void>(comm.signalWrite());
    // signal disconnect downstream
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectDst();
    // takedown io loop
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    destroyIOLoop(true);
    // TODO: Does this function need a return value?
    return LwSciError_Success;
}

//! \brief Exits the I/O thread loop.
//!
//! \param [in] wait: Flag to indicate whether
//!             this function has to wait till the
//!             thread exits.
//!
//! \return void
void IpcDst::destroyIOLoop(bool const wait) noexcept
{
    if (!disconnectRequested.load())
    {
        // Ipc IO thread takedown
        // signal disconnect
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        disconnectRequested.store(true);
        // wake io thread in case it's waiting on a read / write
        // request, but won't receive any
        LwSciError const err { comm.signalDisconnect() };

        if (LwSciError_Success != err) {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            setErrorEvent(err);
            LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
        }
    }
    if (wait && dispatchThread.joinable()) {
        // block on ipc io thread to complete
        // must finish any outstanding writes
        dispatchThread.join();
    }
}

//! <b>Sequence of operations</b>
//!    Launches the dispatch thread by constructing a std::thread object.
bool IpcDst::startDispatchThread(void) noexcept
{
    if (!dispatchThread.joinable()) {
        try {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR50_CPP), "LwSciStream-ADV-CERTCPP-002")
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_5_2), "Approved TID-482")
            dispatchThread = std::thread(&IpcDst::dispatchThreadFunc, this);
            LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR50_CPP))

            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            return true;
        } catch (...) {}
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    return false;
}

} // namespace LwSciStream
