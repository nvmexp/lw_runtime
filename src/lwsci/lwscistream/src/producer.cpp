//! \file
//! \brief LwSciStream producer class definition.
//!
//! \copyright
//! Copyright (c) 2018-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include <cstdint>
#include <limits>
#include <array>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <utility>
#include <cmath>
#include <vector>
#include <functional>
#include "covanalysis.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "sciwrap.h"
#include "lwscistream_types.h"
#include "lwscicommon_os.h"
#include "lwscistream_common.h"
#include "trackarray.h"
#include "safeconnection.h"
#include "packet.h"
#include "syncwait.h"
#include "enumbitset.h"
#include "block.h"
#include "producer.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//!   - It initializes the base Block class with BlockType::PRODUCER,
//!     zero supported source connections, and one supported
//!     destination connection.
//!
//! \implements{19388988}
//!
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A3_1_1), "Bug 2758535")
Producer::Producer(void) noexcept :
LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    Block(BlockType::PRODUCER, 0U, 1U)
{
    // Set up packet description
    Packet::Desc desc { };
    desc.initialLocation = Packet::Location::Downstream;
    desc.defineFillMode = FillMode::Copy;
    desc.statusProdFillMode = FillMode::User;
    desc.fenceProdFillMode = FillMode::User;
    desc.fenceConsFillMode = FillMode::Copy;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    desc.needBuffers = true;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    pktDescSet(std::move(desc));
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
}

//! <b>Sequence of operations</b>
//!   - Retrieves the BlockType of the given block
//!     instance by calling Block::getBlockType() interface
//!     and validates whether the returned BlockType is
//!     BlockType::POOL.
//!   - Retrieves the handle to producer block by calling
//!     Block::getHandle() interface and retrieves the producer
//!     block instance by calling Block::getRegisteredBlock()
//!     interface.
//!   - Initializes the source connection of the pool block instance
//!     by calling Block::connSrcInitiate().
//!   - If successful, initializes the destination connection of the
//!     producer block instance by calling Block::connDstInitiate().
//!   - If initialization of destintion connection is not successful
//!     then cancels the source connection of the pool block instance
//!     by calling Block::connSrcCancel(). Otherwise, completes the
//!     destination connection of producer block instance and source
//!     connection of the pool block instance by calling the
//!     Block::connDstComplete() and Block::connSrcComplete() interfaces
//!     respectively.
//!
//! \implements{19388994}
//!
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_11), "Bug 2738197")
LwSciError
Producer::BindPool(BlockPtr const& paramPool) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_11))
{
    // Validate pool
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if ((nullptr == paramPool) ||
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_5_1), "Bug 2807673")
        (BlockType::POOL != paramPool->getBlockType())) {
        return LwSciError_BadParameter;
    }

    // Note: We get the shared_ptr from the registry rather than
    //       creating a new one from the this pointer
    BlockPtr const thisPtr {Block::getRegisteredBlock(getHandle())};

    // Reserve connections
    IndexRet const srcReserved { paramPool->connSrcInitiate(thisPtr) };
    if (LwSciError_Success != srcReserved.error) {
        return srcReserved.error;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    IndexRet const dstReserved { connDstInitiate(paramPool) };
    if (LwSciError_Success != dstReserved.error) {
        paramPool->connSrcCancel(srcReserved.index);
        return dstReserved.error;
    }

    // Finalize connections
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    paramPool->connSrcComplete(srcReserved.index, dstReserved.index);
    connDstComplete(dstReserved.index, srcReserved.index);
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    pool = paramPool;

    return LwSciError_Success;
}

// Override functions inherited from APIBlockInterface
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError Producer::getOutputConnectPoint(
    BlockPtr& paramBlock) const noexcept
{
    // Only way there can be no pool is if initialization error was ignored
    paramBlock = pool;
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    return isInitSuccess() ? LwSciError_Success : LwSciError_NotInitialized;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! <b>Sequence of operations</b>
//! - Call Block::finalizeConfigOptions() to handle generic setup.
//! - Create a temporary EndInfo vector with just the producer's info.
//! - Call Block::prodInfoSet() to save the vector in the base block.
void Producer::finalizeConfigOptions(void) noexcept
{
    // Lock configuration options
    Block::finalizeConfigOptions();

    // Lock while accessing vector
    Lock const blockLock { blkMutexLock() };

    // Create a temporary single-entry vector with the producer's info
    EndInfoVector tmpVec{};
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        tmpVec.push_back(endpointInfo);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
    } catch (...) {
        setErrorEvent(LwSciError_InsufficientMemory, true);
        return;
    }

    // Store vector in base block
    LwSciError const err { prodInfoSet(tmpVec) };
    if (LwSciError_Success != err) {
        setErrorEvent(err, true);
    }
}

//! <b>Sequence of operations</b>
//! - Call Block::blkMutexLock() to lock the mutex and call consInfoSet()
//!   to save the info.
//! - Call prodInfoFlow() to trigger event and send info downstream.
void Producer::dstRecvConsInfo(
    uint32_t const dstIndex,
    EndInfoVector const& info) noexcept
{
    // When the consumer info arrives at the producer, it doesn't need to be
    //   sent any further. Instead, this function is used to initiate sending
    //   the producer information downstream.

    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Lock and save the info
    {
        Lock const blockLock { blkMutexLock() };
        LwSciError err { consInfoSet(info) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }
    }

    // Pass producer info downstream and trigger event
    prodInfoFlow();
}

//! <b>Sequence of operations</b>
//! - Select operation to perform based on @a setupType.
//! - For ElementExport:
//! -- Call Elements::mapDone() to finallize the list of supported elements.
//! -- Call Elements::dataSend() to ilwoke srcRecvSupportedElements() on
//!    the downstream connection to pass on the supported elements,
//!    and then free the element list memory.
//! - For ElementImport:
//! -- Call Elements::dataArrived() to verify that the allocated element list
//!    was received.
//! -- Call Elements::dataClear() to free the element list memory.
//! -- Mark element import as completed.
//! - For PacketImport:
//! -- Call Block::pktPendingEvent() to check for any packets for which
//!    setup is not complete.
//! -- Mark packet import as completed.
//! -- Call phasePacketsDoneSet() to advance setup phase.
//! - For WaiterAttrExport:
//! -- Call Block::blkMutexLock to lock the mutex and then call
//!    Waiters::doneSet() to mark the setup as completed.
//! -- Call srcRecvSyncWaiter() on the destination connection to send
//!    the waiter information downstream.
//! - For WaiterAttrImport:
//! -- Mark waiter import as completed.
//! - For SignalObjExport:
//! -- Call Block::blkMutexLock to lock the mutex and then call
//!    Signals::doneSet() to mark the setup as completed.
//! -- Call srcRecvSyncSignal() on the destination connection to send
//!    the waiter information downstream.
//! -- Call phaseProdSyncDoneSet() to advance setup phase.
//! - For SignalObjImport:
//! -- Mark signal import as completed.
//! -- Call phaseConsSyncDoneSet() to advance setup phase.
LwSciError Producer::apiSetupStatusSet(
    LwSciStreamSetup const setupType) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    switch (setupType) {

    // All element information is specified. Export it.
    case LwSciStreamSetup_ElementExport:
    {
        // Mark element map complete
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        err = supportedElements.mapDone(false);
        if (LwSciError_Success != err) {
            return err;
        }

        // Define send operation
        Elements::Action const sendAction {
            [this](Elements const& elements) noexcept -> void {
                LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
                getDst().srcRecvSupportedElements(elements);
            }
        };

        // Send downstream and then clear the data
        err = supportedElements.dataSend(sendAction, true);
        if (LwSciError_Success != err) {
            return err;
        }

        break;
    }

    // Import of all element information from the pool is finished. Clean up.
    case LwSciStreamSetup_ElementImport:
    {
        // Info must have arrived before its use can be marked done
        if (!allocatedElements.dataArrived()) {
            return LwSciError_NotYetAvailable;
        }

        // Clean up no longer needed allocated element info
        err = allocatedElements.dataClear();
        if (LwSciError_Success != err) {
            return err;
        }

        // Mark as done
        bool expected { false };
        if (!elementImportDone.compare_exchange_strong(expected, true)) {
            return LwSciError_AlreadyDone;
        }

        break;
    }

    // Import of all packets from the pool is finished.
    case LwSciStreamSetup_PacketImport:
    {
        // Must have received packet completion signal
        if (!packetExportDone) {
            return LwSciError_NotYetAvailable;
        }

        // Must have received and provided status for all incoming packets
        if (nullptr != pktPendingEvent(&Packet::setupPending, false)) {
            return LwSciError_NotYetAvailable;
        }

        // TODO: Clean up packet resources that are no longer needed

        // Mark as done
        bool expected { false };
        if (!packetImportDone.compare_exchange_strong(expected, true)) {
            return LwSciError_AlreadyDone;
        }

        // Advance setup phase
        phasePacketsDoneSet();

        break;
    }

    // Specification of all producer LwSciSync waiter information is finished
    case LwSciStreamSetup_WaiterAttrExport:
    {
        // Must have completed element import
        if (!elementImportDone.load()) {
            return LwSciError_NotYetAvailable;
        }

        // Lock mutex and mark setup as completed
        {
            Lock const blockLock { blkMutexLock() };
            err = prodSyncWaiter.doneSet();
            if (LwSciError_Success != err) {
                return err;
            }
        }

        // Send info downstream
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getDst().srcRecvSyncWaiter(prodSyncWaiter);

        // Mark as done
        waiterExportDone.store(true);

        // TODO: Clean up resources that are no longer needed

        break;
    }

    // Import of all consumer LwSciSync waiter information is finished
    case LwSciStreamSetup_WaiterAttrImport:
    {
        // Must have completed element import and waiter info must have arrived
        if (!elementImportDone.load() || !consSyncWaiter.isComplete()) {
            return LwSciError_NotYetAvailable;
        }

        // Mark as done
        bool expected { false };
        if (!waiterImportDone.compare_exchange_strong(expected, true)) {
            return LwSciError_AlreadyDone;
        }

        // TODO: Clean up resources that are no longer needed

        break;
    }

    // Specification of all producer LwSciSync signal information is finished
    case LwSciStreamSetup_SignalObjExport:
    {
        // Must have completed waiter import
        if (!waiterImportDone.load()) {
            return LwSciError_NotYetAvailable;
        }

        // Lock mutex and mark setup as completed
        {
            Lock const blockLock { blkMutexLock() };
            err = prodSyncSignal.doneSet();
            if (LwSciError_Success != err) {
                return err;
            }
        }

        // Send info downstream
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getDst().srcRecvSyncSignal(prodSyncSignal);

        // Mark as done
        signalExportDone.store(true);

        // Advance setup phase
        phaseProdSyncDoneSet();

        // TODO: Clean up resources that are no longer needed

        break;
    }

    // Import of all consumer LwSciSync signal information is finished
    case LwSciStreamSetup_SignalObjImport:
    {
        // Signal info must have arrived.
        if (!consSyncSignal.isComplete()) {
            return LwSciError_NotYetAvailable;
        }

        // Mark as done
        bool expected { false };
        if (!signalImportDone.compare_exchange_strong(expected, true)) {
            return LwSciError_AlreadyDone;
        }

        // Advance setup phase
        phaseConsSyncDoneSet();

        // TODO: Clean up resources that are no longer needed

        break;
    }

    default:
        return LwSciError_NotSupported;

    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Wrapper::viewVal() to verify that the LwSciBufAttrList contained
//!   in @a elemBufAttr is valid.
//! - Call Elements::mapAdd to add an entry for (@a elemType, @a elemBufAttr)
//!   to the list of supported elements.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
LwSciError Producer::apiElementAttrSet(
    uint32_t const elemType,
    LwSciWrap::BufAttr const& elemBufAttr) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Validate attribute list
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == elemBufAttr.viewVal()) {
        return LwSciError_BadParameter;
    }

    // Add the element to the list
    return supportedElements.mapAdd(elemType, elemBufAttr);
}

//! <b>Sequence of operations</b>
//! - Validate @a queryBlockType.
//! - Call Elements::sizeGet() to query the number of allocated elements.
LwSciError Producer::apiElementCountGet(
    LwSciStreamBlockType const queryBlockType,
    uint32_t& numElements) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Block type being queried must be Pool
    if (LwSciStreamBlockType_Pool != queryBlockType) {
        return LwSciError_BadParameter;
    }

    // Query from allocated element tracker
    return allocatedElements.sizeGet(numElements);
}

//! <b>Sequence of operations</b>
//! - Validate @a queryBlockType.
//! - Call Elements::typeGet() to query the type for the indexed element.
LwSciError Producer::apiElementTypeGet(
    LwSciStreamBlockType const queryBlockType,
    uint32_t const elemIndex,
    uint32_t& userType) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Block type being queried must be Pool
    if (LwSciStreamBlockType_Pool != queryBlockType) {
        return LwSciError_BadParameter;
    }

    // Query from allocated element tracker
    return allocatedElements.typeGet(elemIndex, userType);
}

//! <b>Sequence of operations</b>
//! - Validate @a queryBlockType.
//! - Call Elements::attrGet() to obtain a copy of the attribute list for the
//!   indexed element.
LwSciError Producer::apiElementAttrGet(
    LwSciStreamBlockType const queryBlockType,
    uint32_t const elemIndex,
    LwSciBufAttrList& bufAttrList) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Block type being queried must be Pool
    if (LwSciStreamBlockType_Pool != queryBlockType) {
        return LwSciError_BadParameter;
    }

    // Query from allocated element tracker
    return allocatedElements.attrGet(elemIndex, bufAttrList);
}

//! <b>Sequence of operations</b>
//! - Call Elements::dataCopy() to save a copy of the incoming element list
//!   and transfer data from map to array.
//! - Call Elements::sizePeek() to retrieve the number of elements.
//! - Call Block::elementCountSet() to save the number of elements.
//! - Call Waiters::sizeInit() to initialize the sync attribute vectors.
//! - Call Block::eventPost() to signal element event is ready.
void Producer::dstRecvAllocatedElements(
    uint32_t const dstIndex,
    Elements const& inElements) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Copy incoming data and transfer from map to array
    LwSciError err { allocatedElements.dataCopy(inElements, true) };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Retrieve number of consumers and elements
    size_t const consCount { consumerCountGet() };
    size_t const elemCount { inElements.sizePeek() };

    // Save number of elements
    elementCountSet(elemCount);

    // Initialize waiter and signal info trackers
    err = prodSyncWaiter.sizeInit(elemCount);
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }
    err = consSyncWaiter.sizeInit(elemCount);
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }
    err = prodSyncSignal.sizeInit(ONE, elemCount);
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }
    err = consSyncSignal.sizeInit(consCount, elemCount);
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Trigger element event
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    eventPost(false);
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Block::pktPendingEvent() to retrieve a new packet, if any.
//! - Call Packet::handleGet() to retrieve the packet's handle.
LwSciError Producer::apiPacketNewHandleGet(
    LwSciStreamPacket& handle) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Packet info isn't available until element import is done
    if (!elementImportDone.load()) {
        return LwSciError_NotYetAvailable;
    }

    // Packet info isn't available after packet import is done
    if (packetImportDone) {
        return LwSciError_NoLongerAvailable;
    }

    // Check for new packet
    PacketPtr const pkt
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        { pktPendingEvent(&Packet::defineHandlePending, false) };
    if (nullptr == pkt) {
        return LwSciError_NoStreamPacket;
    }

    // Retrieve handle
    handle = pkt->handleGet();
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Block::blkMutexLock() to create a new Lock instance.
//! - Call Block::pktFindByHandle() to retrieve the Packet instance referenced
//!   by @a handle.
//! - Call Packet::bufferGet() to retrieve the indexed buffer.
LwSciError Producer::apiPacketBufferGet(
    LwSciStreamPacket const handle,
    uint32_t const elemIndex,
    LwSciWrap::BufObj& bufObjWrap) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Packet info isn't available until element import is done
    if (!elementImportDone.load()) {
        return LwSciError_NotYetAvailable;
    }

    // Packet info isn't available after packet import is done
    if (packetImportDone) {
        return LwSciError_NoLongerAvailable;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Take lock to prevent data from being cleared mid-query
    Lock const blockLock { blkMutexLock() };

    // Find the packet for this handle
    PacketPtr const pkt { pktFindByHandle(handle, true) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // Retrieve a copy of the buffer object
    return pkt->bufferGet(elemIndex, bufObjWrap);
}

//! <b>Sequence of operations</b>
//! - Call Block::pktPendingEvent() to retrieve a deleted packet, if any.
//! - Call Packet::handleGet() to retrieve the packet's handle.
//! - Call Block::pktRemove() to remove the packet from the map.
//! - Call Packet::cookieGet() to retrieve the packet's cookie.
LwSciError Producer::apiPacketOldCookieGet(
    LwSciStreamCookie& cookie) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Packet info isn't available until element import is done
    if (!elementImportDone.load()) {
        return LwSciError_NotYetAvailable;
    }

    // Check for deleted packet
    PacketPtr const pkt
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        { pktPendingEvent(&Packet::deleteCookiePending, false) };
    if (nullptr == pkt) {
        return LwSciError_NoStreamPacket;
    }

    // Delete packet from map
    pktRemove(pkt->handleGet(), false);

    // Retrieve cookie
    cookie = pkt->cookieGet();
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindByHandle() to retrieve the Packet instance referenced
//!   by @a handle.
//! - Call Block::blkMutexLock() to protect packet map operations:
//! -- If @ status is Success, call Block::pktFindByCookie() to check if
//!    any Packets have already been assigned the same @a cookie.
//! -- Call Packet::statusProdSet() to set the @a status and @a cookie.
//! - Call srcRecvPacketStatus() of the destination connection to send the
//!   status downstream.
LwSciError Producer::apiPacketStatusSet(
    LwSciStreamPacket const handle,
    LwSciStreamCookie const cookie,
    LwSciError const status) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Find the packet for this handle
    PacketPtr const pkt { pktFindByHandle(handle) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }

    // Take the mutex lock, make sure cookie is unique, and set status/cookie
    {
        // Lock the mutex
        Lock const blockLock { blkMutexLock() };

        // If status is success, make sure cookie is unique
        if (LwSciError_Success == status) {
            if (nullptr != pktFindByCookie(cookie, true)) {
                return LwSciError_AlreadyInUse;
            }
        }

        // Set status and cookie
        err = pkt->statusProdSet(status, cookie);
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // Send status downstream
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    getDst().srcRecvPacketStatus(*pkt);

    return LwSciError_Success;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Block::pktCreate() to create a new Packet instance, copy
//!   definition from the original packet, and insert it in the map.
//! - Call Block::eventPost() to wake threads waiting for events.
void Producer::dstRecvPacketCreate(
    uint32_t const dstIndex,
    Packet const& origPacket) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Create new packet and insert in map
    LwSciError const err { pktCreate(origPacket.handleGet(), &origPacket) };

    // Post any error, and wake waiting threads
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (LwSciError_Success == err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        eventPost(false);
    } else {
        setErrorEvent(err);
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindByHandle() to retrieve the Packet instance referenced
//!   by @a handle.
//! - Call Packet::locationCheck() to verify that the Packet is lwrrently
//!   downstream.
//! - Call Packet::deleteSet() to mark the packet for deletion.
//! - Call Block::eventPost() to wake any waiting threads.
void Producer::dstRecvPacketDelete(
    uint32_t const dstIndex,
    LwSciStreamPacket const handle) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Find the packet for this handle
    PacketPtr const pkt { pktFindByHandle(handle) };
    if (nullptr == pkt) {
        setErrorEvent(LwSciError_StreamBadPacket);
        return;
    }

    // Should only receive deletion message when the packet is downstream
    if (!pkt->locationCheck(Packet::Location::Downstream)) {
        setErrorEvent(LwSciError_StreamPacketInaccessible);
        return;
    }

    // Mark the packet for deletion
    // TODO: If pool previously indicated this packet was available, we
    //       need to decrease the number of available packets. This will
    //       involve tracking we don't lwrrently have.
    if (!pkt->deleteSet()) {
        setErrorEvent(LwSciError_StreamPacketDeleted);
        return;
    }

    // Wake any waiting threads
    eventPost(false);

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Set flag marking packet set as complete.
//! - Call Block::eventPost to wake any waiting threads.
void Producer::dstRecvPacketsComplete(
    uint32_t const dstIndex) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Mark packet export as done
    bool expected { false };
    if (!packetExportDone.compare_exchange_strong(expected, true)) {
        setErrorEvent(LwSciError_AlreadyDone);
    }

    // Prepare event and wake any waiting threads
    packetExportEvent.store(true);
    eventPost(false);

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Elements::dataArrived() to make sure prerequisite information
//!   is available.
//! - Call Block::blkMutexLock() to lock the block.
//! - Call Waiters::attrSet() to store the attribute list.
LwSciError Producer::apiElementWaiterAttrSet(
    uint32_t const elemIndex,
    LwSciWrap::SyncAttr const& syncAttr) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Can't set attributes until element information available
    if (!allocatedElements.dataArrived()) {
        return LwSciError_NotYetAvailable;
    }

    // Can't set attributes after export is done
    if (waiterExportDone.load()) {
        return LwSciError_NoLongerAvailable;
    }

    // Lock block while setting value
    Lock const blockLock { blkMutexLock() };

    // Set the attribute
    return prodSyncWaiter.attrSet(elemIndex, syncAttr);
}

//! <b>Sequence of operations</b>
//! - Call Waiters::isComplete() to make sure information is available.
//! - Call Waiters::attrGet() to retrieve the attribute list.
LwSciError Producer::apiElementWaiterAttrGet(
    uint32_t const elemIndex,
    LwSciWrap::SyncAttr& syncAttr) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Can't get attributes until information is available
    if (!consSyncWaiter.isComplete()) {
        return LwSciError_NotYetAvailable;
    }

    // Can't get attributes after import is done
    if (waiterImportDone.load()) {
        return LwSciError_NoLongerAvailable;
    }

    // Get the attribute
    return consSyncWaiter.attrGet(elemIndex, syncAttr);
}

//! <b>Sequence of operations</b>
//! - Call Block::blkMutexLock() to lock the block, and then call
//!   Waiters::copy() to copy the incoming sync information and
//!   prepare the LwSciStreamEventType_WaiterAttr event.
//! - Call Block::eventPost() to wake any waiting threads.
void Producer::dstRecvSyncWaiter(
    uint32_t const dstIndex,
    Waiters const& syncWaiter) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Take the mutex lock
    Lock const blockLock { blkMutexLock() };

    // Save a copy of the incoming information
    LwSciError const err { consSyncWaiter.copy(syncWaiter) };
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        setErrorEvent(err, true);
    } else {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        eventPost(true);
    }
}

//! <b>Sequence of operations</b>
//! - Call Signals::isComplete() to make sure prerequisite information
//!   is available.
//! - Call Block::blkMutexLock() to lock the block.
//! - Call Signals::syncSet() to store the sync object.
LwSciError Producer::apiElementSignalObjSet(
    uint32_t const elemIndex,
    LwSciWrap::SyncObj const& syncObj) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Can't set sync objects until waiter information available
    if (!consSyncWaiter.isComplete()) {
        return LwSciError_NotYetAvailable;
    }

    // Can't set sync objects after export is done
    if (signalExportDone.load()) {
        return LwSciError_NoLongerAvailable;
    }

    // Lock block while setting value
    Lock const blockLock { blkMutexLock() };

    // Set the sync object
    return prodSyncSignal.syncSet(0U, elemIndex, syncObj);
}

//! <b>Sequence of operations</b>
//! - Call Signals::isComplete() to make sure information is available.
//! - Call Signals::syncGet() to retrieve the sync object.
LwSciError Producer::apiElementSignalObjGet(
    uint32_t const queryBlockIndex,
    uint32_t const elemIndex,
    LwSciWrap::SyncObj& syncObj) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Can't get sync objects until information is available
    if (!consSyncSignal.isComplete()) {
        return LwSciError_NotYetAvailable;
    }

    // Can't get sync objects after import is done
    if (signalImportDone.load()) {
        return LwSciError_NoLongerAvailable;
    }

    // Get the sync object
    return consSyncSignal.syncGet(queryBlockIndex, elemIndex, syncObj);
}

//! <b>Sequence of operations</b>
//! - Call Block::blkMutexLock() to lock the block, and then call
//!   Signals::copy() to copy the incoming sync information and
//!   prepare the LwSciStreamEventType_WaiterObj event.
//! - Call Block::eventPost() to wake any waiting threads.
void Producer::dstRecvSyncSignal(
    uint32_t const dstIndex,
    Signals const& syncSignal) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Take the mutex lock
    Lock const blockLock { blkMutexLock() };

    // Save a copy of the incoming information
    LwSciError const err { consSyncSignal.copy(syncSignal) };
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        setErrorEvent(err, true);
    } else {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        eventPost(true);
    }
}

//! <b>Sequence of operations</b>
//! - Call srcDequeuePayload() interface of destination block (pool) to
//!   obtain a packet instance to reuse. Return if none available.
//! - Call Packet::handleGet() to retrieve the packet's handle and then call
//!   Block::pktFindByHandle() to retrieve the corresponding local packet.
//! - Call Packet::locationUpdate() to move the packet from downstream to
//!   application.
//! - Call Packet::cookieGet() to retrieve the cookie.
//! - Call Packet::fenceProdReset() to reset the packet's producer fences.
//! - Call Packet::fenceConsCopy() to copy the consumer fences.
LwSciError Producer::apiPayloadObtain(
    LwSciStreamCookie& cookie) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Validate block state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Request a packet from the pool
    PacketPtr const consPayload { getDst().srcDequeuePayload() };
    if (nullptr == consPayload) {
        return LwSciError_NoStreamPacket;
    }

    // TODO: If either of these fail, something has gone wrong with
    //       internal bookkeeping.

    // Find the local packet for this payload
    PacketPtr const pkt { pktFindByHandle(consPayload->handleGet()) };
    if (nullptr == pkt) {
        return LwSciError_StreamInternalError;
    }

    // Update location to show application ownership
    if (!pkt->locationUpdate(Packet::Location::Downstream,
                             Packet::Location::Application)) {
        return LwSciError_StreamInternalError;
    }

    // Retrieve cookie, clear producer fences, and copy consumer fences
    //   into the packet
    cookie = pkt->cookieGet();
    pkt->fenceProdReset();
    return pkt->fenceConsCopy(*consPayload);

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! - Call Block::pktFindByHandle() to retrieve the Packet instance referenced
//!   by @a handle.
//! - Call blkLockMutex() to lock the mutex.
//! - Call Packet::locationUpdate() to check and update packet location.
//! - Call Packet::fenceProdDone() to finalize fence list.
//! - Call srcRecvPayload() interface of destination block to send the
//!   packet downstream.
LwSciError Producer::apiPayloadReturn(
    LwSciStreamPacket const handle) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Validate block state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Find the packet for this handle
    PacketPtr const pkt { pktFindByHandle(handle) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }

    {
        // Take the mutex lock while updating the packet state
        Lock const blockLock { blkMutexLock() };

        // Make sure packet is lwrrently held by application and
        //   update location
        if (!pkt->locationUpdate(Packet::Location::Application,
                                 Packet::Location::Downstream)) {
            return LwSciError_StreamPacketInaccessible;
        }

        // Mark the producer fences as complete and clear consumer fences
        pkt->fenceProdDone();
        pkt->fenceConsReset();
    }

    // Send the packet downstream
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    getDst().srcRecvPayload(*pkt);

    return LwSciError_Success;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindByHandle() to retrieve the Packet instance referenced
//!   by @a handle.
//! - Call blkLockMutex() to lock the mutex.
//! - Call Packet::locationCheck() to verify that the Packet is lwrrently
//!   held by the application.
//! - Call Packet::fenceProdSet() to store the fence.
LwSciError Producer::apiPayloadFenceSet(
    LwSciStreamPacket const handle,
    uint32_t const elemIndex,
    LwSciWrap::SyncFence const& postfence) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Find the packet for this handle
    PacketPtr const pkt { pktFindByHandle(handle) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }

    // Take the mutex lock
    Lock const blockLock { blkMutexLock() };

    // Only allowed when packet is held by application
    if (!pkt->locationCheck(Packet::Location::Application)) {
        return LwSciError_StreamPacketInaccessible;
    }

    // Set the fence
    return pkt->fenceProdSet(elemIndex, postfence);
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindByHandle() to retrieve the Packet instance referenced
//!   by @a handle.
//! - Call blkLockMutex() to lock the mutex.
//! - Call Packet::locationCheck() to verify that the Packet is lwrrently
//!   held by the application.
//! - Call Packet::fenceConsGet() to retrieve the fence.
LwSciError Producer::apiPayloadFenceGet(
    LwSciStreamPacket const handle,
    uint32_t const queryBlockIndex,
    uint32_t const elemIndex,
    LwSciWrap::SyncFence& prefence) noexcept
{
    // Validate block state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Find the packet for this handle
    PacketPtr const pkt { pktFindByHandle(handle) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }

    // Take the mutex lock
    Lock const blockLock { blkMutexLock() };

    // Only allowed when packet is held by application
    if (!pkt->locationCheck(Packet::Location::Application)) {
        return LwSciError_StreamPacketInaccessible;
    }

    // Get the fence
    return pkt->fenceConsGet(queryBlockIndex, elemIndex, prefence);
}

//! <b>Sequence of operations</b>
//!  - Prepares a pending LwSciStreamEventType_PacketReady event
//!    then wakes up the threads waiting for the events if any by calling
//!    the Block::eventPost() interface.
void Producer::dstRecvPayload(
    uint32_t const dstIndex,
    Packet const& consPayload) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Payload info is ignored
    static_cast<void>(consPayload);

    // Validate block/input state
    //   In safety builds, only allowed in streaming
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // If disconnect has oclwrred, return but don't report an error
    // TODO: Need to distinguish various disconnect cases and handle
    //       in the validate function above.
    if (!connComplete()) {
        return;
    }

    // Increment ready packet counter and wake any waiting thread
    //   The bounds check is just to make CERT/Autosar happy. It will
    //   never be violated unless something fundamental is broken.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A13_5_3), "LwSciStream-ADV-AUTOSARC++14-001")
    if (std::numeric_limits<uint32_t>::max() > pktReadyEvents) {
        ++pktReadyEvents;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    eventPost(false);

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
}

//! <b>Sequence of operations</b>
//! - Call Elements::eventGet to check whether there is a pending event
//!   for the allocated elements.
//! - Call Block::pktPendingEvent() to check for pending packet creation.
//! - Check for completed packet set event.
//! - Call Waiters::pendingEvent() to check for pending waiter info event.
//! - Call Signals::pendingEvent() to check for pending signal info event.
//! - If all setup is done, check for ready packet events.
//! - Call Block::pktPendingEvent() to check for pending packet deletion.
//!
//! \implements{19389063}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
bool
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
Producer::pendingEvent(
    LwSciStreamEventType& event) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Check for Elements event
    if (allocatedElements.eventGet()) {
        event = LwSciStreamEventType_Elements;
        return true;
    }

    // No other producer-specific events until element import is done
    if (!elementImportDone.load()) {
        return false;
    }

    // Search for packet creation events
    if (nullptr != pktPendingEvent(&Packet::definePending, true)) {
        event = LwSciStreamEventType_PacketCreate;
        return true;
    }

    // Check for packet completion event
    if (packetExportEvent.load()) {
        event = LwSciStreamEventType_PacketsComplete;
        packetExportEvent.store(false);
        return true;
    }

    // Check for waiter info event
    if (!waiterImportDone) {
        if (consSyncWaiter.pendingEvent()) {
            event = LwSciStreamEventType_WaiterAttr;
            return true;
        }
    }

    // Handle sync objects from consumer
    if (waiterExportDone && !signalImportDone) {
        if (consSyncSignal.pendingEvent()) {
            event = LwSciStreamEventType_SignalObj;
            return true;
        }
    }

    // Search for packet deletion events
    if (nullptr != pktPendingEvent(&Packet::deletePending, true)) {
        event = LwSciStreamEventType_PacketDelete;
        return true;
    }

    // If there's a packet ready, and all the other setup is done, report it
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A13_5_3), "LwSciStream-ADV-AUTOSARC++14-001")
    if (phaseRuntimeGet(true) && (0U < pktReadyEvents)) {
        event = LwSciStreamEventType_PacketReady;
        --pktReadyEvents;
        return true;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A13_5_3))

    return false;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! <b>Sequence of operations</b>
//!   - Calls Block::disconnectDst() interface as it is the upstream
//!     endpoint of the stream.
//!
//! \implements{19389027}
//!
LwSciError Producer::disconnect(void) noexcept {
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectDst();
    // TODO: Does this function need a return value?
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//!   - The function disconnects the destination by calling Block::disconnectDst()
//!     interface and sets the disconnect event by calling
//!     Block::disconnectEvent() interface.
//!
//! \implements{19389057}
//!
void Producer::dstDisconnect(
    uint32_t const dstIndex) noexcept
{
    // Validate block/input state
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectDst();
    disconnectEvent();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
}

//! <b>Sequence of operations</b>
//! - Call Block::phaseChange().
void Producer::phaseSendReady(void) noexcept
{
    // Since this block and all the ones downstream are ready, the phase
    //   change can begin
    phaseChange();
}

//! <b>Sequence of operations</b>
//! - Call configComplete() to check whether the configuration of the block
//!   instance not finalized yet.
//! - Call blkMutexLock() to lock the mutex.
//! - Call EndInfo::infoSet() to add the producer information.
LwSciError
Producer::apiUserInfoSet(
    uint32_t const userType,
    InfoPtr const& info) noexcept
{
    if (configComplete()) {
        return LwSciError_NoLongerAvailable;
    }

    // Validate block state
    //   In safety builds, only allowed in setup
    ValidateBits validation{};
    validation.set(ValidateCtrl::SetupPhase);
    LwSciError const err{ validateWithError(validation) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Take the mutex lock to save the info
    Lock const blockLock{ blkMutexLock() };
    return endpointInfo.infoSet(userType, info);
}

} // namespace LwSciStream
