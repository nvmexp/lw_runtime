//! \file
//! \brief LwSciStream pool class declaration.
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
#include <cstddef>
#include <iostream>
#include <array>
#include <utility>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <cmath>
#include <vector>
#include <functional>
#include "covanalysis.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "sciwrap.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "lwscistream_types.h"
#include "lwscicommon_os.h"
#include "lwscistream_common.h"
#include "safeconnection.h"
#include "packet.h"
#include "enumbitset.h"
#include "block.h"
#include "trackarray.h"
#include "pool.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//!  - This constructor ilwokes Block constructor with BlockType::POOL. It also
//!    initializes Packet::Desc.
//!
//! \implements{19449423}
// Note: For unknown reasons, the allowlisting of A12-1-3 below is not working
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A3_1_1), "Bug 2758535")
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A12_1_3), "Bug 2806643")
Pool::Pool(uint32_t const paramNumPackets) noexcept :
LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    Block(BlockType::POOL),
    numPacketsDesired(paramNumPackets),
    numPacketsCreated(0U),
    numPacketsDefined(0U),
    producerElements(FillMode::Copy, FillMode::User),
    consumerElements(FillMode::Copy, FillMode::User),
    allocatedElements(FillMode::User, FillMode::User),
    elementImportDone(false),
    elementExportDone(false),
    packetExportDone(false),
    packetImportDone(false),
    payloadQueue(),
    secondary(false)
{
    // Set up packet description
    Packet::Desc desc { };
    desc.initialLocation = Packet::Location::Queued;
    desc.defineFillMode = FillMode::User;
    desc.statusProdFillMode = FillMode::Copy;
    desc.statusConsFillMode = FillMode::Copy;
    desc.fenceConsFillMode = FillMode::Copy;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    desc.needBuffers = true;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    pktDescSet(std::move(desc));
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A3_1_1))

LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
//! <b>Sequence of operations</b>
//!  - Removes all packets from payloadQueue by calling
//!    Packet::PayloadQ::dequeue() interface of the payloadQueue object.
//!
//! \implements{19723374}
Pool::~Pool(void) noexcept
{
    // Remove all packets from the payload queue so their pointers to
    //   each other don't keep them alive.
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A6_5_3), "LwSciStream-ADV-AUTOSARC++14-004")
    PacketPtr oldPacket {};
    do {
        oldPacket = payloadQueue.dequeue();
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    } while (nullptr != oldPacket);
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A6_5_3))

}
LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))

// Sets the flag to indicate it a secondary pool, which is not
// attached to Producer block.
void Pool::makeSecondary(void) noexcept
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    secondary = true;
}

// A stub implementation which always returns
// LwSciError_AccessDenied, as pool does not allow output connections
// through public APIs.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_11), "Bug 2817427")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_13), "Bug 2817427")
LwSciError Pool::getOutputConnectPoint(
    BlockPtr& paramBlock) const noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_13))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_11))
{
    static_cast<void>(paramBlock);
    // Pools don't allow connection through public API
    return LwSciError_AccessDenied;
}

// A stub implementation which always returns
// LwSciError_AccessDenied, as pool does not allow input connections
// through public APIs.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_11), "Bug 2817427")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_13), "Bug 2817427")
LwSciError Pool::getInputConnectPoint(
    BlockPtr& paramBlock) const noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_13))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_11))
{
    static_cast<void>(paramBlock);
    // Pools don't allow connection through public API
    return LwSciError_AccessDenied;
}

//! <b>Sequence of operations</b>
//!  - Calls Block::disconnectSrc() and Block::disconnectDst() interfaces to
//!    disconnect the source and destination blocks respectively.
//!
//! \implements{19514187}
LwSciError Pool::disconnect(void) noexcept {
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectSrc();
    disconnectDst();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    // TODO: Does this function need a return value?
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Disconnects the source block by calling the Block::disconnectSrc()
//! interface and triggers the LwSciStreamEventType_Disconnected event by
//! calling the Block::disconnectEvent() interface.
//! - Disconnects the destination block by calling the Block::disconnectDst()
//! interface.
//!
//! \implements{19514193}
void Pool::srcDisconnect(
    uint32_t const srcIndex) noexcept
{
    // Validate block/input state
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    disconnectSrc();
    disconnectEvent();
    disconnectDst();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
}

//! <b>Sequence of operations</b>
//! - Disconnects the destination block by calling the Block::disconnectDst()
//! interface and triggers the LwSciStreamEventType_Disconnected event by
//! calling the Block::disconnectEvent() interface.
//! - Disconnects the source block by calling the Block::disconnectSrc()
//! interface.
//!
//! \implements{19514199}
void Pool::dstDisconnect(
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
    disconnectSrc();
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
}

//! <b>Sequence of operations</b>
//! - Select operation to perform based on @a setupType.
//! - For ElementExport:
//! -- Verify import has completed.
//! -- Call Elements::mapDone() to finallize the list of allocated elements.
//! -- Call Block::elementCountSet() to save the number of elements.
//! -- Call Elements::dataSend() to ilwoke {src|dst}RecvAllocatedElements() on
//!    the {down|up}stream connections to pass on the allocated elements,
//!    mark the export as done, and then free the element list memory.
//! - For ElementImport:
//! -- Call Elements::dataArrived() to verify that the queried element lists
//!    were received from upstream and downstream.
//! -- Call Elements::dataClear() to free any element list memory that is no
//!    longer needed.
//! -- Mark element import as completed.
//! -- For seondary pool, mark element export as completed as well.
//! - For PacketExport:
//! -- Mark export as completed if not already done.
//! -- Call {src|dst}RecvPacketsComplete() on the {down|up}stream connections
//!    to inform the rest of the stream that the packet list is complete.
//! - For PacketImport:
//! -- Call Block::pktPendingEvent() to check for any packets for which
//!    setup is not complete.
//! -- Mark import as completed if not already done.
//! -- For each packet in the list, call dstRecvPayload() on the upstream
//!    connection to inform the producer the packet is available.
LwSciError Pool::apiSetupStatusSet(
    LwSciStreamSetup const setupType) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

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
        // Secondary pool will reuse packet layout from
        // the primary pool, no need to export elements.
        if (secondary) {
            return LwSciError_NotSupported;
        }

        // Can't export until import is done
        if (!elementImportDone) {
            return LwSciError_NotYetAvailable;
        }

        // Mark element map complete and parse
        err = allocatedElements.mapDone(true);
        if (LwSciError_Success != err) {
            return err;
        }

        // Save number of buffers
        elementCountSet(allocatedElements.sizePeek());

        // Define send operation
        Elements::Action const sendAction {
            [this](Elements const& elements) noexcept -> void {
                getDst().srcRecvAllocatedElements(elements);
                getSrc().dstRecvAllocatedElements(elements);
                elementExportDone.store(true);
            }
        };

        // Send up/downstream and then clear the data
        err = allocatedElements.dataSend(sendAction, true);
        if (LwSciError_Success != err) {
            return err;
        }

        break;
    }

    // Import of all element information from the pool is finished. Clean up.
    case LwSciStreamSetup_ElementImport:
    {
        Elements& srcElements{
            secondary ? allocatedElements : producerElements
        };
        Elements& dstElements{ consumerElements };
        // Info must have arrived before its use can be marked done
        if (!srcElements.dataArrived() ||
            !dstElements.dataArrived()) {
            return LwSciError_NotYetAvailable;
        }

        // Clean up no longer needed allocated element info.
        err = srcElements.dataClear();
        if (LwSciError_Success != err) {
            return err;
        }
        err = dstElements.dataClear();
        if (LwSciError_Success != err) {
            return err;
        }

        // Mark as done
        bool expected { false };
        if (!elementImportDone.compare_exchange_strong(expected, true)) {
            return LwSciError_AlreadyDone;
        }

        // Mark element export done for the secondary pool
        // to keep the bookkeeping the same as the primary pool
        if (secondary) {
            elementExportDone.store(true);
        }

        break;
    }

    // All packets are defined. Inform endpoints.
    case LwSciStreamSetup_PacketExport:
    {
        // Make sure target number of packets was created and accepted
        if (numPacketsDesired != numPacketsCreated) {
            return LwSciError_InsufficientData;
        }

        // Make sure only done once
        bool expected { false };
        if (!packetExportDone.compare_exchange_strong(expected, true)) {
            return LwSciError_AlreadyDone;
        }

        // Inform stream that packet definitiion is complete
        getSrc().dstRecvPacketsComplete();
        getDst().srcRecvPacketsComplete();

        // TODO: Clean up packet resources that are no longer needed

        break;
    }

    // All packets are accepted. Allow streaming to proceed.
    case LwSciStreamSetup_PacketImport:
    {
        // Make sure target number of packets was created and accepted
        if (numPacketsDesired != numPacketsDefined) {
            return LwSciError_InsufficientData;
        }

        // Make sure packet export was completed
        if (!packetExportDone.load()) {
            return LwSciError_NotYetAvailable;
        }

        // Must have received status for all packets
        if (nullptr != pktPendingEvent(&Packet::setupPending, false)) {
            return LwSciError_NotYetAvailable;
        }

        // Make sure only done once
        bool expected { false };
        if (!packetImportDone.compare_exchange_strong(expected, true)) {
            return LwSciError_AlreadyDone;
        }

        // Inform producer all packets are available
        Packet dummyPkt {
            Packet::Desc(),
            LwSciStreamPacket_Ilwalid,
            LwSciStreamCookie_Ilwalid
        };
        for (uint32_t i { 0U }; numPacketsDefined > i; ++i) {
            getSrc().dstRecvPayload(dummyPkt);
        }

        // Mark packet setup done and check if all setup is done
        phasePacketsDoneSet();

        // TODO: Clean up packet resources that are no longer needed

        break;
    }

    default:
        return LwSciError_NotSupported;

    }

    return LwSciError_Success;

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Elements::dataArrived() to verify supported element info arrived.
//! - Call Wrapper::viewVal() to verify that the LwSciBufAttrList contained
//!   in @a elemBufAttr is valid.
//! - Call Elements::mapAdd to add an entry for (@a elemType, @a elemBufAttr)
//!   to the list of allocated elements.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
LwSciError Pool::apiElementAttrSet(
    uint32_t const elemType,
    LwSciWrap::BufAttr const& elemBufAttr) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    // Secondary pool will reuse packet layout from
    // the primary pool, no need to set attributes.
    if (secondary) {
        return LwSciError_NotSupported;
    }

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

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    // Can't specify the allocated element information until the supported
    //   information has arrived from the endpoints. We don't enforce that
    //   it all have been queried. It is left to applications to determine
    //   how much of it they need.
    if (!producerElements.dataArrived() ||
        !consumerElements.dataArrived()) {
        return LwSciError_NotYetAvailable;
    }

    // Validate attribute list
    if (nullptr == elemBufAttr.viewVal()) {
        return LwSciError_BadParameter;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // Add the element to the list
    return allocatedElements.mapAdd(elemType, elemBufAttr);
}

//! <b>Sequence of operations</b>
//! - Call Elements::dataArrived() to verify queried element info arrived.
//! - Validate @a queryBlockType.
//! - Call Elements::sizeGet() to query the number of supported elements
//!   from the desired endpoint(s).
LwSciError Pool::apiElementCountGet(
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

    Elements& srcElements{
        secondary ? allocatedElements : producerElements
    };
    Elements& dstElements{ consumerElements };

    // Both upstream and downstream elements must have arrived,
    //   regardless of which one we're querying
    if (!srcElements.dataArrived() ||
        !dstElements.dataArrived()) {
        return LwSciError_NotYetAvailable;
    }

    // Query type from appropriate element list
    if (LwSciStreamBlockType_Producer == queryBlockType) {
        return srcElements.sizeGet(numElements);
    } else if (LwSciStreamBlockType_Consumer == queryBlockType) {
        return dstElements.sizeGet(numElements);
    } else {
        return LwSciError_BadParameter;
    }
}

//! <b>Sequence of operations</b>
//! - Call Elements::dataArrived() to verify queried element info arrived.
//! - Validate @a queryBlockType.
//! - Call Elements::typeGet() to query the type for the indexed element
//!   from the desired endpoint(s).
LwSciError Pool::apiElementTypeGet(
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

    Elements& srcElements{
        secondary ? allocatedElements : producerElements
    };
    Elements& dstElements{ consumerElements };

    // Both upstream and downstream elements must have arrived,
    //   regardless of which one we're querying
    if (!srcElements.dataArrived() ||
        !dstElements.dataArrived()) {
        return LwSciError_NotYetAvailable;
    }

    // Query type from appropriate element list
    if (LwSciStreamBlockType_Producer == queryBlockType) {
        return srcElements.typeGet(elemIndex, userType);
    } else if (LwSciStreamBlockType_Consumer == queryBlockType) {
        return dstElements.typeGet(elemIndex, userType);
    } else {
        return LwSciError_BadParameter;
    }
}

//! <b>Sequence of operations</b>
//! - Call Elements::dataArrived() to verify queried element info arrived.
//! - Validate @a queryBlockType.
//! - Call Elements::attrGet() to obtain a copy of the attribute list for the
//!   indexed element from the desired endpoint(s).
LwSciError Pool::apiElementAttrGet(
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

    Elements& srcElements{
        secondary ? allocatedElements : producerElements
    };
    Elements& dstElements{ consumerElements };

    // Both upstream and downstream elements must have arrived,
    //   regardless of which one we're querying
    if (!srcElements.dataArrived() ||
        !dstElements.dataArrived()) {
        return LwSciError_NotYetAvailable;
    }

    // Query type from appropriate element list
    if (LwSciStreamBlockType_Producer == queryBlockType) {
        return srcElements.attrGet(elemIndex, bufAttrList);
    } else if (LwSciStreamBlockType_Consumer == queryBlockType) {
        return dstElements.attrGet(elemIndex, bufAttrList);
    } else {
        return LwSciError_BadParameter;
    }
}

//! <b>Sequence of operations</b>
//! - Call Elements::dataCopy() to save a copy of the incoming element list
//!   and transfer data from map to array.
//! - Call Elements::dataArrived() to check if elements from producer have
//!   also arrived, and if so signal element event is ready.
//! - If a secondary pool, call dstRecvSupportedElements() on the upstream
//!   connection to pass @a inElements on.
void Pool::dstRecvSupportedElements(
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
    LwSciError const err { consumerElements.dataCopy(inElements, true) };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Trigger element event if producer data is also available
    // Note: If producer and consumer data arrive at the same time in
    //       separate threads, a race could potentially cause an extra
    //       spurious event wake signal. Applications are supposed to
    //       be able to handle this.
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (producerElements.dataArrived()) {
        eventPost(false);
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // For secondary pool, send the elements upstream to the primary
    if (secondary) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getSrc().dstRecvSupportedElements(inElements);
    }
}

//! <b>Sequence of operations</b>
//! - Call Elements::dataCopy() to save a copy of the incoming element list
//!   and transfer data from map to array.
//! - Call Elements::dataArrived() to check if elements from consumer have
//!   also arrived, and if so signal element event is ready.
void Pool::srcRecvSupportedElements(
    uint32_t const srcIndex,
    Elements const& inElements) noexcept
{
    if (secondary) {
        setErrorEvent(LwSciError_NotSupported);
        return;
    }

    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Copy incoming data and transfer from map to array
    LwSciError const err { producerElements.dataCopy(inElements, true) };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Trigger element event if consumer data is also available
    // Note: If producer and consumer data arrive at the same time in
    //       separate threads, a race could potentially cause an extra
    //       spurious event wake signal. Applications are supposed to
    //       be able to handle this.
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (consumerElements.dataArrived()) {
        eventPost(false);
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Elements::dataArrived() to verify consumer element info arrived.
//! - Call Elements::dataCopy() to save a copy of the incoming element list
//!   and transfer data from map to array.
//! - Call Block::elementCountSet() to save the number of elements.
//! - Call Elements::dataSend() to ilwoke srcRecvAllocatedElements() on the
//!   downstream connections to pass on the allocated elements.
//! - Call Block::eventPost() to Signal element event is ready.
void Pool::srcRecvAllocatedElements(
    uint32_t const srcIndex,
    Elements const& inElements) noexcept
{
    if (!secondary) {
        setErrorEvent(LwSciError_NotSupported);
        return;
    }

    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation{};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Consumer elements must have arrived
    if (!consumerElements.dataArrived()) {
        setErrorEvent(LwSciError_NotYetAvailable);
        return;
    }

    // Copy incoming data and transfer from map to array
    LwSciError err{ allocatedElements.dataCopy(inElements, true) };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // Save number of buffers
    elementCountSet(inElements.sizePeek());

    // Define send operation
    Elements::Action const sendAction{
        [this](Elements const& elements) noexcept -> void {
            LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
            getDst().srcRecvAllocatedElements(elements);
        }
    };

    // Send the allocated elements downstream
    err = allocatedElements.dataSend(sendAction, false);
    if (LwSciError_Success != err) {
        setErrorEvent(err);
    }

    // Trigger element event
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    eventPost(false);
}

//! <b>Sequence of operations</b>
//! - Create a unique LwSciStreamPacket.
//! - Call Block::pktCreate to Create a new Packet instance with the
//!   LwSciStreamPacket and the @a cookie value.
//! - If packet limit is already reached, call Block::pktRemove() to remove
//!   the packet instance that was created.
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError Pool::apiPacketCreate(
    LwSciStreamCookie const cookie,
    LwSciStreamPacket& handle) noexcept
{
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Require element export be completed
    if (!elementExportDone.load()) {
        return LwSciError_NotYetAvailable;
    }

    // Don't allow invalid cookies
    if (LwSciStreamCookie_Ilwalid == cookie) {
        return LwSciError_StreamBadCookie;
    }

    // TODO: this a temporary method for derivation
    // of the handle value. Once the final method has
    // has been established, the assignment below will
    // be updated accordingly.
    handle = ~cookie;

    // Create the packet and insert in map
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    err = pktCreate(handle, nullptr, cookie);
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(M0_1_2), "Bug 3003226")
    if (LwSciError_Success != err) {
        return err;
    }

    // Increment the packet count, making sure we don't exceed the limit
    {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        Lock const blockLock { blkMutexLock() };
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
        if (numPacketsCreated < numPacketsDesired) {
            ++numPacketsCreated;
        } else {
            err = LwSciError_Overflow;
        }
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(M0_1_2), "Bug 3003226")
    if (LwSciError_Success != err) {
        pktRemove(handle);
        return err;
    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Wrapper::viewVal() to check whether the incoming LwSciBufObj
//!   is valid.
//! - Call Block::pktFindByHandle() to retrieve the Packet instance referenced
//!   by @a handle.
//! - Call Block::blkMutexLock() to protect the packet.
//! - Call Packet::bufferSet() to set the buffer for @a elemIndex.
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
LwSciError Pool::apiPacketBuffer(
    LwSciStreamPacket const packetHandle,
    uint32_t const elemIndex,
    LwSciWrap::BufObj const& elemBufObj) noexcept
{
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Validate buffer
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == elemBufObj.viewVal()) {
        return LwSciError_BadParameter;
    }

    // Find the packet for this handle
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    PacketPtr const pkt { pktFindByHandle(packetHandle) };
    if ((nullptr == pkt) || pkt->deleteGet()) {
        return LwSciError_StreamBadPacket;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // Lock mutex while inserting the buffer
    Lock const blockLock { blkMutexLock() };

    // Insert the buffer in the packet
    return pkt->bufferSet(elemIndex, elemBufObj);
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindByHandle() to retrieve the Packet instance referenced
//!   by @a handle.
//! - Call Block::blkMutexLock() to protect the packet and call
//!   Packet::defineDone() to indicate packet definition is complete.
//! - Call {dst|src}RecvPacketCreate() on {source|destination} connections
//!   to send the packet definition {up|down}stream.
//! - Call Packet::defineClear() to free the local resources for the packet
//!   definition.
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
LwSciError Pool::apiPacketComplete(
    LwSciStreamPacket const packetHandle) noexcept
{
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Find the packet for this handle
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    PacketPtr const pkt { pktFindByHandle(packetHandle) };
    if ((nullptr == pkt) || pkt->deleteGet()) {
        return LwSciError_StreamBadPacket;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // Lock mutex and and try to finalize the packet
    //  The consumer fence list will be empty when first sent to the producer
    {
        Lock const blockLock { blkMutexLock() };
        LwSciError const err2 { pkt->defineDone() };
        if (LwSciError_Success != err2) {
            return err2;
        }
        pkt->fenceConsDone();
    }

    // Send packet to the rest of the stream
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    getSrc().dstRecvPacketCreate(*pkt);
    getDst().srcRecvPacketCreate(*pkt);
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))

    // Release the buffer objects
    pkt->defineClear();

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindByHandle() to retrieve the Packet instance referenced
//!   by @a handle.
//! - Call Packet::deleteSet() to mark the packet for deletion.
//! - Call Packet::PayloadQ::extract() to remove the packet from the
//!   payload queue, if it is present.
//! - If successful, call Block::pktRemove() to remove the Packet instance
//!   from the packet map and then call {dst|src}RecvPacketDelete() on the
//!   {source|destination} connections to send the deletion message
//!   {up|down}stream.
LwSciError
Pool::apiPacketDelete(
    LwSciStreamPacket const handle) noexcept
{
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Find the packet for this handle
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    PacketPtr const pkt { pktFindByHandle(handle) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // Mark packet as ready for deletion
    if (!pkt->deleteSet()) {
        // Already marked for deletion
        return LwSciError_StreamBadPacket;
    }

    // Try to get it from the queue.
    // TODO: We need to figure out handling of secondary pools. For the
    //       primary pool, the deletion waits until the packet arrives
    //       back here at the pool. But for secondary pools, we send
    //       all the packets upstream so they're ready at the C2CSrc
    //       block, and they may not come back down if the stream is
    //       done. But we don't want to just delete them right away
    //       if there's a chance a new frame will arrive with them.
    //       May need a new interface for unused packets to be sent
    //       back down.
    bool doDelete;
    {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        Lock const blockLock { blkMutexLock() };
        doDelete = payloadQueue.extract(*pkt);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
    }

    // If successful, handle deletion now.
    //   Otherwise it will be done when it is returned.
    if (doDelete) {
        pktRemove(handle);
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getSrc().dstRecvPacketDelete(handle);
        getDst().srcRecvPacketDelete(handle);
        LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindByHandle() to retrieve the Packet instance referenced
//!   by @a handle.
//! - Call Packet::statusArrived() to check whether status has arrived.
//! - Call Packet::statusAccepted() to check whether the packet was accepted.
LwSciError
Pool::apiPacketStatusAcceptGet(
    LwSciStreamPacket const handle,
    bool& accepted) noexcept
{
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Queries not available after import done
    if (packetImportDone) {
        return LwSciError_NoLongerAvailable;
    }

    // Find the packet for this handle
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    PacketPtr const pkt { pktFindByHandle(handle) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // Status must have arrived before querying acceptance
    if (!pkt->statusArrived()) {
        return LwSciError_NotYetAvailable;
    }

    // Query acceptance
    accepted = pkt->statusAccepted();

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindByHandle() to retrieve the Packet instance referenced
//!   by @a handle.
//! - Call Packet::statusArrived() to check whether status has arrived.
//! - Call Packet::status{Prod|Cons}Get() to retrieve the status value
//!   from the desired endpoint.
LwSciError
Pool::apiPacketStatusValueGet(
    LwSciStreamPacket const handle,
    LwSciStreamBlockType const queryBlockType,
    uint32_t const queryBlockIndex,
    LwSciError& status) noexcept
{
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Queries not available after import done
    if (packetImportDone) {
        return LwSciError_NoLongerAvailable;
    }

    // Find the packet for this handle
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    PacketPtr const pkt { pktFindByHandle(handle) };
    if (nullptr == pkt) {
        return LwSciError_StreamBadPacket;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    // Status must have arrived before querying acceptance
    if (!pkt->statusArrived()) {
        return LwSciError_NotYetAvailable;
    }

    // Query appropriate kind of status
    if (LwSciStreamBlockType_Producer == queryBlockType) {
        return pkt->statusProdGet(status);
    } else if (LwSciStreamBlockType_Consumer == queryBlockType) {
        return pkt->statusConsGet(queryBlockIndex, status);
    } else {
        return LwSciError_BadParameter;
    }
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindHandle() to find the local Packet instance
//!   corresponding to the incoming packet.
//! - Call Block::blkMutexLock() to take the lock and then call
//!   Packet::statusProdCopy() to copy the producer status information
//!   into the local Packet.
//! - Call Packet::statusArrived() to check whether consumer status is also
//!   available, and if so call Block::eventPost() to wake any waiting threads.
void Pool::srcRecvPacketStatus(
    uint32_t const srcIndex,
    Packet const& origPacket) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Find the packet for this handle
    PacketPtr const pkt { pktFindByHandle(origPacket.handleGet()) };
    if (nullptr == pkt) {
        setErrorEvent(LwSciError_StreamBadPacket);
        return;
    }

    // Take the lock and copy the status
    {
        Lock const blockLock { blkMutexLock() };
        LwSciError const err { pkt->statusProdCopy(origPacket) };
        if (LwSciError_Success != err) {
            setErrorEvent(err);
            return;
        }
    }

    // Trigger status event if consumer data is also available
    // Note: If producer and consumer data arrive at the same time in
    //       separate threads, a race could potentially cause an extra
    //       spurious event wake signal. Applications are supposed to
    //       be able to handle this.
    if (pkt->statusArrived()) {
        eventPost(false);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Block::pktFindHandle() to find the local Packet instance
//!   corresponding to the incoming packet.
//! - Call Block::blkMutexLock() to take the lock and then call
//!   Packet::statusConsCopy() to copy the consumer status information
//!   into the local Packet.
//! - Call Packet::statusArrived() to check whether producer status is also
//!   available, and if so call Block::eventPost() to wake any waiting threads.
void Pool::dstRecvPacketStatus(
    uint32_t const dstIndex,
    Packet const& origPacket) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::DstIndex);
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // Find the packet for this handle
    PacketPtr const pkt { pktFindByHandle(origPacket.handleGet()) };
    if (nullptr == pkt) {
        setErrorEvent(LwSciError_StreamBadPacket);
        return;
    }

    // Take the lock and copy the status
    {
        Lock const blockLock { blkMutexLock() };
        LwSciError const err { pkt->statusConsCopy(origPacket) };
        if (LwSciError_Success != err) {
            setErrorEvent(err);
            return;
        }
    }

    // Trigger status event if producer data is also available
    // Note: If producer and consumer data arrive at the same time in
    //       separate threads, a race could potentially cause an extra
    //       spurious event wake signal. Applications are supposed to
    //       be able to handle this.
    if (pkt->statusArrived()) {
        eventPost(false);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Gets the first available Packet instance from the Packet::PayloadQ by
//! calling Packet::PayloadQ::dequeue() interface of the payloadQueue object
//!  under thread protection provided by Block::blkMutexLock().
//! - Makes sure the Packet instance is not yet marked for deletion by calling
//! Packet::deleteGet() interface.
//! - If the Packet instance is not marked for deletion, updates its location
//! from Packet::Location::Queued to Packet::Location::Upstream by calling
//! Packet::locationUpdate() interface, then retrieves its Payload by calling
//! Packet::payloadGet() interface and returns the Payload.
//! - If the Packet instance is marked for deletion remove the Packet instance
//! by calling Block::pktRemove() interface and notifies upstream and downstream
//! to producer and consumer blocks of the packet deletion information by calling
//! dstRecvPacketDelete() interface of the source block through source
//! SafeConnection and srcRecvPacketDelete() interface of the destination block
//! through destination SafeConnection respectively and retries getting new
//! packet from Packet::PayloadQ.
//!
//! \implements{19506996}
PacketPtr Pool::srcDequeuePayload(
    uint32_t const srcIndex) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Initialize return value to empty pointer
    PacketPtr pkt { };

    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return pkt;
    }

    // Loop in case another thread is deleting things while we dequeue them
    // TODO: With the new APIs, should be able to make the deletion cleaner
    while (true) {

        // Lock mutex and dequeue next available packet, if any
        {
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
            Lock const blockLock { blkMutexLock() };

            pkt = payloadQueue.dequeue();
            if (nullptr == pkt) {
                return pkt;
            }
        }

        // Make sure packet hasn't been deleted
        if (!pkt->deleteGet()) {
            // Update location and return packet to producer
            static_cast<void>(pkt->locationUpdate(Packet::Location::Queued,
                                                  Packet::Location::Upstream));
            return pkt;
        } else {
            // We interrupted another thread during deletion.
            //   Just do the deletion for them, and try again.
            LwSciStreamPacket const handle { pkt->handleGet() };
            pktRemove(handle);
            LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
            getSrc().dstRecvPacketDelete(handle);
            getDst().srcRecvPacketDelete(handle);
            LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
            pkt.reset();
        }
    }

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>





//! - Validates whether the stream is connected by calling Block::connComplete()
//! interface. If not, returns immediately.
//! - Finds the corresponding packet instance of the Pool block by calling the
//! Block::pktFindByHandle() interface.
//! - If found, updates its Location from Packet::Location::Downstream to
//! Packet::Location::Queued by calling Packet::locationUpdate() interface of
//! the Packet instance.
//! - Queues the packet instance into payloadQueue by calling
//! Packet::PayloadQ::enqueue() interface of the payloadQueue object under
//! thread protection provided by Block::blkMutexLock().
//! - Checks whether the packet instance was already marked for deletion by
//! calling the Packet::deleteGet() interface. If so, extracts the packet instance
//! from payloadQueue using Packet::extract() and removes the packet
//! instance by calling Block::pktRemove() interface and notifies upstream and
//! downstream to producer and consumer blocks of the packet deletion information
//! by calling dstRecvPacketDelete() interface of the source block through source
//! SafeConnection and srcRecvPacketDelete() interface of the destination block
//! through destination SafeConnection respectively. Otherwise, it notifies
//! upstream to producer block of the packet availability for reuse by calling the
//! srcReusePacket() interface of the source block through source
//! SafeConnection.
//!
//! \implements{19513842}
void Pool::dstRecvPayload(
    uint32_t const dstIndex,
    Packet const& consPayload) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")

    // Validate block/input state
    //   In safety builds, only allowed in streaming
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

    // Find the local packet for this payload
    PacketPtr const pkt { pktFindByHandle(consPayload.handleGet()) };
    if (nullptr == pkt) {
        setErrorEvent(LwSciError_StreamBadPacket);
        return;
    }

    // Update packet location
    if (!pkt->locationUpdate(Packet::Location::Downstream,
                             Packet::Location::Queued)) {
        setErrorEvent(LwSciError_StreamPacketInaccessible);
        return;
    }

    // Save the return fences in the packet
    LwSciError const err { pkt->fenceConsCopy(consPayload) };
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return;
    }

    // To avoid race conditions with other threads retrieving and/or
    //   deleting packets, we always unconditionally return the packet
    //   to the queue for reuse, even if it is marked for deletion.
    //   Then we check for deletion and try to get it back again.
    //   Whichever thread dequeues it is the one responsible for
    //   doing whatever needs to be done.
    // TODO: Should be able to do this more cleanly now.
    bool isZombie;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A0_1_1), "Bug 2738197")
    bool doDelete { false };
    {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
        Lock const blockLock { blkMutexLock() };
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
        payloadQueue.enqueue(pkt);
        isZombie = pkt->deleteGet();
        if (isZombie) {
            doDelete = payloadQueue.extract(*pkt);
        }
    }

    // If it wasn't flagged for deletion, signal the producer.
    if (!isZombie) {
        getSrc().dstRecvPayload(*pkt);
    }

    // If we're responsible for deleting it, do that now.
    if (doDelete) {
        LwSciStreamPacket const handle { pkt->handleGet() };
        pktRemove(handle);
        getSrc().dstRecvPacketDelete(handle);
        getDst().srcRecvPacketDelete(handle);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! - Queries any status events on packets by calling Block::pktPendingEvent()
//!   interface, and return if found.
//! - Calls Elements::dataArrived() and Elements::eventGet() to check if
//!   producer element data is available and consumer element event
//!   is ready, respectively, returning  LwSciEventType_Elements event if so.
//!
//! \implements{19514298}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
bool
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
Pool::pendingEvent(
    LwSciStreamEventType& event) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    // Note: Lwrrently it is safe to handle packet events before
    //       attribute events because there can't be any packet
    //       events before attribute setup is done. Since packet
    //       events are the common case during streaming, checking
    //       them first makes this funcion more optimal.
    //
    //       This may need to be reorganized if/when we add dynamic
    //       changing of element attributes.

    // Search for packet status events
    PacketPtr pkt { pktPendingEvent(&Packet::statusPending, true) };
    if (nullptr != pkt) {
        // If accepted, add to queue and increment number of completed packets
        //   (with pointless check that can never fail to make CERT happy)
        if (pkt->statusAccepted()) {
            payloadQueue.enqueue(pkt);
            if (MAX_INT_SIZE > numPacketsDefined) {
                numPacketsDefined++;
            }
        }
        event = LwSciStreamEventType_PacketStatus;
        return true;
    }

    // Producer and consumer events could arrive in any order. We first
    //   just check whether producer data is available, then check the
    //   consumer event flag to actually signal an event. The producer
    //   event flag is expected to be set at this point, and is cleared.
    Elements& srcElements{
        secondary ? allocatedElements : producerElements
    };
    Elements& dstElements{ consumerElements };
    if (srcElements.dataArrived()) {
        if (dstElements.eventGet()) {
            if (!srcElements.eventGet()) {
                setErrorEvent(LwSciError_StreamInternalError);
            }
            event = LwSciStreamEventType_Elements;
            return true;
        }
    }

    // No events found
    return false;
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

} // namespace LwSciStream
