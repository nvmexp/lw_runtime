//! \file
//! \brief LwSciStream Block definition.
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
#include <utility>
#include <limits>
#include <new>
#include <iostream>
#include <array>
#include <cassert>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <system_error>
#include <unordered_map>
#include <memory>
#include <bitset>
#include <chrono>
#include <atomic>
#include <cmath>
#include <vector>
#include <functional>
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "lwscievent.h"
#include "lwscievent_internal.h"
#include "covanalysis.h"
#include "lwscicommon_os.h"
#include "lwscistream_common.h"
#include "lwscistream_types.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "sciwrap.h"
#include "safeconnection.h"
#include "enumbitset.h"
#include "packet.h"
#include "lwscistream_panic.h"
#include "block.h"
#include "elements.h"

namespace LwSciStream {

// A global registry which is used to store the mapping of block
// handles with their corresponding instances.
std::unique_ptr<HandleBlockPtrMap> Block::blockRegistry {};

// A global counter variable which keeps track of handle to be
// assigned to the next registered block.
LwSciStreamBlock Block::nextHandle {1U};

// A global mutex to prevent conlwrrent access to blockRegistry
// and nextHandle members.
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
std::mutex Block::registryMutex {};

//! <b>Sequence of operations</b>
//!  - Initializes BlockType, sets the maximum number of source/destination
//!    connections, the flags indicating whether the source/destination block
//!    presents in other process and assigns a new handle to this block instance
//!    by calling Block::assignHandle() interface.
//!  - As a protected constructor, only the derived concrete block can be
//!    instantiated.
//!
//! \implements{18794670}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A12_1_5), "Bug 2989775")
Block::Block(
    BlockType const type,
    uint32_t const paramConnSrcTotal,
    uint32_t const paramConnDstTotal,
    bool const paramConnSrcRemote,
    bool const paramConnDstRemote) noexcept :
        APIBlockInterface(),
        SrcBlockInterface(),
        DstBlockInterface(),
        blkType(type),
        blkHandle(ILWALID_BLOCK_HANDLE),
        blkMutex(),
        consInfo(),
        prodInfo(),
        consInfoMsg(false),
        prodInfoMsg(false),
        connSrc(),
        connDst(),
        connSrcTotal(paramConnSrcTotal),
        connDstTotal(paramConnDstTotal),
        connSrcRemote(paramConnSrcRemote),
        connDstRemote(paramConnDstRemote),
        configOptLocked(false),
        streamDone(false),
        disconnectDone(false),
        internalErr(LwSciError_Success),
        errEvent(false),
        connEvent(false),
        discEvent(false),
        initSuccess(false),
        signalMode(EventSignalMode::None),
        eventServiceEvent(nullptr),
        pktUsed(false),
        pktDesc(),
        pktMap(),
        phasePacketsDone(false),
        phaseProdSyncDone(false),
        phaseConsSyncDone(false),
        phaseDstRecv(0U),
        phaseSrcSend(false),
        phaseEvent(false),
        phaseRuntime(false)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
    assert(connSrcTotal <= MAX_SRC_CONNECTIONS);
    assert(connDstTotal <= MAX_DST_CONNECTIONS);
    assert(!paramConnSrcRemote || (1U == connSrcTotal));
    assert(!paramConnDstRemote || (1U == connDstTotal));
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_2_2))
    initSuccess = assignHandle();
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))

//! \brief Assigns a new handle to the block instance
//!   Called by the derived class constructor to map the BlockPtr to a
//!   unique handle.
//!
//! <b>Sequence of operations</b>
//!  - Creates an instance of Lock (std::unique_lock<std::mutex>) by
//!    passing the registryMutex as an argument to its constructor.
//!  - Upon successful locking, assigns a unique handle to the block instance.
//!
//! \return bool
//! * true: if a new handle is assigned to the block.
//! * false: if failed to assign a new handle to the block or Mutex locking
//!    fails.
//!
//! \implements{20994615}
bool Block::assignHandle(void) noexcept
{
    // Lock registry
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    Lock registryLock { registryMutex, std::defer_lock };
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        registryLock.lock();
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::system_error& e) {
        static_cast<void>(e);
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        return false;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))

    // Assign handle and increment counter
    blkHandle = nextHandle;
    if ((std::numeric_limits<uintptr_t>::max() - ONE) < nextHandle) {
        // CERT_INT30_C: 'nextHandle++' may wrap.
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        return false;
    }
    nextHandle++;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    return true;
}

// Sets the block initialization as failed,
// used by deriving blocks to mark construction failures.
void Block::setInitFail(void) noexcept
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    initSuccess = false;
}

// Returns the handle mapped to this block instance.
LwSciStreamBlock Block::getHandle(void) const noexcept
{
    return blkHandle;
}

// Returns the type of block.
BlockType Block::getBlockType() const noexcept
{
    return blkType;
}

// TODO: We could add a constant string or some other value to
//       provide more information about the source of the error.
// TODO2: May want a flag to disinguish fatal errors which may
//        leave the block in an unrecoverable state from errors
//        due to bad input from other blocks, which might mean
//        the stream as a whole is screwed but at least the local
//        block is okay.

//! <b>Sequence of operations</b>
//!  - Creates a new Block::Lock object with blkMutexLock() to protect the entire
//!    call against multithread access.
//!  - If error event is not already set, calls Block::eventPost() interfacee, with
//!    parameter 'locked' set to true, to trigger the error events.
//!
//! \implements{18794733}
void
Block::setErrorEvent(
    LwSciError const err,
    bool const       locked) noexcept
{
    // If lock is not already held, take it
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    Lock const blockLock { blkMutexLock(locked) };

    // If error is not already set, do so and trigger
    if (LwSciError_Success == internalErr) {
        internalErr = err;
        errEvent.store(true);
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        eventPost(true);
    }
}

// Validates all requested states.
LwSciError Block::validateWithError(
    ValidateBits const& bits,
    uint32_t const connIndex) const noexcept
{
    // If required, check for completeness of stream
    if (bits[ValidateCtrl::Complete] && !connComplete()) {
        return LwSciError_StreamNotConnected;
    }

    // If required, check whether stream connection event has been queried
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A13_5_3), "LwSciStream-ADV-AUTOSARC++14-001")
    if (bits[ValidateCtrl::CompleteQueried] &&
        (!connComplete() || connEvent)) {
        return LwSciError_StreamNotConnected;
    }

    // If required, validate connection index
    if (bits[ValidateCtrl::SrcIndex] && (connIndex >= connSrcTotal)) {
        return LwSciError_StreamBadSrcIndex;
    }
    if (bits[ValidateCtrl::DstIndex] && (connIndex >= connDstTotal)) {
        return LwSciError_StreamBadDstIndex;
    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//!  - Calls Block::validateWithError() interface to validate all the requested
//!   states.
//!  - Calls Block::setErrorEvent() interface if LwSciError returned from
//!  Block::validateWithError() interface is not LwSciError_Success.
//!
//! \implements{18794928}
bool Block::validateWithEvent(
    ValidateBits const& bits,
    uint32_t const connIndex) noexcept
{
    LwSciError const err {validateWithError(bits, connIndex)};
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (LwSciError_Success != err) {
        setErrorEvent(err);
        return false;
    }
    return true;
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//!  - Calls SafeConnection<T>::connInitiate() to initialize the source
//!    safeconnection of the block.
//!
//! \implements{18793548}
IndexRet Block::connSrcInitiate(
    BlockPtr const& srcPtr) noexcept
{
    IndexRet rv { LwSciError_NotSupported, 0U };

    // Reject outright if no input connections
    // TODO: This needs its own error code
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if ((0U == connSrcTotal) || connSrcRemote) {
        return rv;
    }

    // Find and initiate an available connection
    // (Lwrrently either 0 or 1 connection points but this will eventually
    //  loop over a vector)
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    for (rv.index=0U; rv.index < connSrcTotal; ++rv.index) {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_7_1), "Bug 2738489")
        if (connSrc[rv.index].connInitiate(srcPtr)) {
            rv.error = LwSciError_Success;
            return rv;
        }
    }

    rv.error = LwSciError_InsufficientResource;
    return rv;
}

//! <b>Sequence of operations</b>
//!  - Completes the source SafeConnection of the block using
//!    SafeConnection<T>::connComplete().
//!
//! \implements{18793614}
void Block::connSrcComplete(
    uint32_t const srcIndex,
    uint32_t const dstIndex) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(CTR50_CPP), "TID-1397")
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_7_1), "Bug 2738489")
    connSrc[srcIndex].connComplete(dstIndex);
    LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(CTR50_CPP))
}

//! <b>Sequence of operations</b>
//!  - Calls SafeConnection<T>::connCancel() to cancel the source
//!    safeconnection of the block.
//!
//! \implements{18793752}
void Block::connSrcCancel(
    uint32_t const srcIndex) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(CTR50_CPP), "TID-1397")
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_7_1), "Bug 2738489")
    connSrc[srcIndex].connCancel();
    LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(CTR50_CPP))
}

//! <b>Sequence of operations</b>
//!  - Calls SafeConnection<T>::connInitiate() to initialize the destination
//!    safeconnection of the block.
//!
//! \implements{18793761}
IndexRet Block::connDstInitiate(
    BlockPtr const& dstPtr) noexcept
{
    IndexRet rv { LwSciError_NotSupported, 0U };

    // Reject outright if no output connections
    // TODO: This needs its own error code
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if ((0U == connDstTotal) || connDstRemote) {
        return rv;
    }

    // Find and initiate an available connection
    // (Lwrrently either 0 or 1 connection points but this will eventually
    //  loop over a vector)
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    for (rv.index=0U; rv.index < connDstTotal; ++rv.index) {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_7_1), "Bug 2738489")
        if (connDst[rv.index].connInitiate(dstPtr)) {
            rv.error = LwSciError_Success;
            return rv;
        }
    }

    rv.error = LwSciError_InsufficientResource;
    return rv;
}

//! <b>Sequence of operations</b>
//!  - Completes the destination SafeConnection of the block using
//!    SafeConnection<T>::connComplete().
//!
//! \implements{18793785}
void Block::connDstComplete(
    uint32_t const dstIndex,
    uint32_t const srcIndex) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(CTR50_CPP), "TID-1397")
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_7_1), "Bug 2738489")
    connDst[dstIndex].connComplete(srcIndex);
    LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(CTR50_CPP))
}

//! <b>Sequence of operations</b>
//!  - Calls SafeConnection<T>::connCancel() to cancel the destination
//!    safeconnection of the block.
//!
//! \implements{18793794}
void Block::connDstCancel(
    uint32_t const dstIndex) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(CTR50_CPP), "TID-1397")
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_7_1), "Bug 2738489")
    connDst[dstIndex].connCancel();
    LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(CTR50_CPP))
}

//! <b>Sequence of operations</b>
//!  - Calls Block::getRegisteredBlock() interface to get the pointer to its
//!    own instance.
//!
//! \implements{18793800}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError Block::getOutputConnectPoint(BlockPtr& paramBlock) const noexcept
{
    // By default, returns this block
    paramBlock = getRegisteredBlock(blkHandle);
    return LwSciError_Success;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! <b>Sequence of operations</b>
//!  - Calls Block::getRegisteredBlock() interface to get the pointer to its
//!    own instance.
//!
//! \implements{18793815}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError Block::getInputConnectPoint(BlockPtr& paramBlock) const noexcept
{
    // By default, returns this block
    paramBlock = getRegisteredBlock(blkHandle);
    return LwSciError_Success;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! <b>Sequence of operations</b>
//! - Validates state.
//! - Queries consumer info tracker's map for the total number of
//!   consumers downstream of this block (including the block itself
//!   if this is a consumer) by calling the EndMap::getTotalEndCount()
//!   interface for the consumer info tracker.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError Block::apiConsumerCountGet(
    uint32_t& numConsumers) const noexcept
{
    // Validate block state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::Complete);
    LwSciError const err {validateWithError(validation)};
    if (LwSciError_Success != err) {
        return err;
    }

    // Query number of endpoints from consumer info
    //  With CERT-required check that will never fail
    numConsumers = (consInfo.size() < MAX_INT_SIZE)
                 ? static_cast<uint32_t>(consInfo.size()) : 0U;
    return LwSciError_Success;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//
// Element definition functions
//

// Default implementation of APIBlockInterface::apiSetupStatusSet(),
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiSetupStatusSet(
    LwSciStreamSetup const setupType) noexcept
{
    static_cast<void>(setupType);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiElementAttrSet(),
//   which always returns LwSciError_NotSupported error code.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
LwSciError Block::apiElementAttrSet(
    uint32_t const elemType,
    LwSciWrap::BufAttr const& elemBufAttr) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    static_cast<void>(elemType);
    static_cast<void>(elemBufAttr);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiElementCountGet(),
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiElementCountGet(
    LwSciStreamBlockType const queryBlockType,
    uint32_t& numElements) noexcept
{
    static_cast<void>(queryBlockType);
    static_cast<void>(numElements);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiElementTypeGet(),
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiElementTypeGet(
    LwSciStreamBlockType const queryBlockType,
    uint32_t const elemIndex,
    uint32_t& userType) noexcept
{
    static_cast<void>(queryBlockType);
    static_cast<void>(elemIndex);
    static_cast<void>(userType);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiElementAttrGet(),
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiElementAttrGet(
    LwSciStreamBlockType const queryBlockType,
    uint32_t const elemIndex,
    LwSciBufAttrList& bufAttrList) noexcept
{
    static_cast<void>(queryBlockType);
    static_cast<void>(elemIndex);
    static_cast<void>(bufAttrList);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiElementUsageSet(),
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiElementUsageSet(
    uint32_t const elemIndex,
    bool const used) noexcept
{
    static_cast<void>(elemIndex);
    static_cast<void>(used);
    return LwSciError_NotSupported;
}

// Default implementation of DstBlockInterface::srcRecvSupportedElements,
//   which always triggers LwSciError_NotImplemented error event.
void Block::srcRecvSupportedElements(
    uint32_t const srcIndex,
    Elements const& inElements) noexcept
{
    // Supported elements always flow upstream except at the pool, which
    //   must override this function anyways. So the base implementation
    //   just sets an error.
    static_cast<void>(srcIndex);
    static_cast<void>(inElements);
    setErrorEvent(LwSciError_NotImplemented);
}

// Default implementation of SrcBlockInterface::dstRecvSupportedElements,
//   which passes the information through for direct one-to-many links and
//   triggers LwSciError_NotImplemented error event otherwise.
//! <b>Sequence of operations</b>
//! - Checks whether this is a direct one-to-one link.
//! - Calls dstRecvSupportedElements() on all upstream connections to
//!   pass on the element list.
void Block::dstRecvSupportedElements(
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

    // Override required if not a one-to-many link.
    if (!isDstOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // Pass through to all upstream blocks
    for (uint32_t i {0U}; connSrcTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getSrc(i).dstRecvSupportedElements(inElements);
    }
}

// Default implementation of DstBlockInterface::srcRecvAllocatedElements,
//   which passes the information through for direct one-to-many links and
//   triggers LwSciError_NotImplemented error event otherwise.
//! <b>Sequence of operations</b>
//! - Checks whether this is a direct one-to-one link.
//! - Calls srcRecvAllocatedElements() on all downstream connections to
//!   pass on the element list.
void Block::srcRecvAllocatedElements(
    uint32_t const srcIndex,
    Elements const& inElements) noexcept
{
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

    // Override required if not a one-to-many link.
    if (!isSrcOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // Save the number of elements
    //   Most blocks need this even if they don't need the full info
    elementCountSet(inElements.sizePeek());

    // Pass through to all downstream blocks
    for (uint32_t i {0U}; connDstTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getDst(i).srcRecvAllocatedElements(inElements);
    }
}

// Default implementation of SrcBlockInterface::dstRecvAllocatedElements,
//   which always triggers LwSciError_NotImplemented error event.
void Block::dstRecvAllocatedElements(
    uint32_t const dstIndex,
    Elements const& inElements) noexcept
{
    // Allocated elements always flow downstream except at the producer, which
    //   must override this function anyways. So the base implementation
    //   just sets an error.
    static_cast<void>(dstIndex);
    static_cast<void>(inElements);
    setErrorEvent(LwSciError_NotImplemented);
}

//
// Packet definition functions
//

// Default implementation of APIBlockInterface::apiPacketCreate,
// which always returns LwSciError_NotSupported error code.
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError Block::apiPacketCreate(
    LwSciStreamCookie const cookie,
    LwSciStreamPacket& handle) noexcept
{
    static_cast<void>(cookie);
    static_cast<void>(handle);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiPacketBuffer,
// which always returns LwSciError_NotSupported error code.
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
LwSciError Block::apiPacketBuffer(
    LwSciStreamPacket const packetHandle,
    uint32_t const elemIndex,
    LwSciWrap::BufObj const& elemBufObj) noexcept
{
    static_cast<void>(packetHandle);
    static_cast<void>(elemIndex);
    static_cast<void>(elemBufObj);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiPacketComplete,
// which always returns LwSciError_NotSupported error code.
LwSciError Block::apiPacketComplete(
    LwSciStreamPacket const handle) noexcept
{
    static_cast<void>(handle);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiPacketDelete,
// which always returns LwSciError_NotSupported error code.
LwSciError Block::apiPacketDelete(
    LwSciStreamPacket const handle) noexcept
{
    static_cast<void>(handle);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiPacketNewHandleGet,
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiPacketNewHandleGet(
    LwSciStreamPacket& handle) noexcept
{
    static_cast<void>(handle);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiPacketBufferGet,
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiPacketBufferGet(
    LwSciStreamPacket const handle,
    uint32_t const elemIndex,
    LwSciWrap::BufObj& bufObjWrap) noexcept
{
    static_cast<void>(handle);
    static_cast<void>(elemIndex);
    static_cast<void>(bufObjWrap);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiPacketOldCookieGet,
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiPacketOldCookieGet(
    LwSciStreamCookie& cookie) noexcept
{
    static_cast<void>(cookie);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiPacketStatusSet,
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiPacketStatusSet(
    LwSciStreamPacket const handle,
    LwSciStreamCookie const cookie,
    LwSciError const status) noexcept
{
    static_cast<void>(handle);
    static_cast<void>(cookie);
    static_cast<void>(status);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiPacketStatusAcceptGet,
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiPacketStatusAcceptGet(
    LwSciStreamPacket const handle,
    bool& accepted) noexcept
{
    static_cast<void>(handle);
    static_cast<void>(accepted);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiPacketStatusValueGet,
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiPacketStatusValueGet(
    LwSciStreamPacket const handle,
    LwSciStreamBlockType const queryBlockType,
    uint32_t const queryBlockIndex,
    LwSciError& status) noexcept
{
    static_cast<void>(handle);
    static_cast<void>(queryBlockType);
    static_cast<void>(queryBlockIndex);
    static_cast<void>(status);
    return LwSciError_NotSupported;
}

//! <b>Sequence of operations</b>
//! - Checks whether this is a direct one-to-many link.
//! - If the block uses the packet map, call Block::pktCreate() to create
//!   a new packet and insert it in the map.
//! - Calls srcRecvPacketCreate() on all downstream connections to
//!   pass on the original Packet.
void Block::srcRecvPacketCreate(
    uint32_t const srcIndex,
    Packet const& origPacket) noexcept
{
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

    // Override required if not a one-to-many link.
    if (!isSrcOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // If packet map is used, create new packet and insert in map
    if (pktUsed) {
        LwSciError const err { pktCreate(origPacket.handleGet()) };
        if (LwSciError_Success != err) {
            setErrorEvent(err);
            return;
        }
    }

    // Pass through to all downstream blocks
    for (uint32_t i {0U}; connDstTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getDst(i).srcRecvPacketCreate(origPacket);
    }
}

// Default implementation of SrcBlockInterface::dstRecvPacketCreate,
//   which always triggers LwSciError_NotImplemented error event.
void Block::dstRecvPacketCreate(
    uint32_t const dstIndex,
    Packet const& origPacket) noexcept
{
    static_cast<void>(dstIndex);
    static_cast<void>(origPacket);
    setErrorEvent(LwSciError_NotImplemented);
}

//! <b>Sequence of operations</b>
//! - Check whether this is a direc one-to-many link.
//! - Check if this block uses the packet map
//! -- Call pktFindByHandle() to retrieve the Packet instance for @a handle.
//! -- Call Packet::locationCheck() to ensure the Packet is upstream.
//! -- Call pktRemove() to remove the Packet from the map.
//! - Call srcRecvPacketDelete() on all downstream connections to
//!   pass on the packet.
void Block::srcRecvPacketDelete(
    uint32_t const srcIndex,
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
    validation.set(ValidateCtrl::SrcIndex);
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Override required if not a one-to-many link.
    if (!isSrcOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // Handle local copy of the packet if map is used
    if (pktUsed) {
        // Look up the packet
        PacketPtr const pkt { pktFindByHandle(handle) };
        if (nullptr == pkt) {
            setErrorEvent(LwSciError_StreamBadPacket);
            return;
        }

        // Delete message should only be received when packet is upstream
        if (!pkt->locationCheck(Packet::Location::Upstream)) {
            setErrorEvent(LwSciError_StreamPacketInaccessible);
            return;
        }

        // Remove the packet from the map
        pktRemove(handle);
    }

    // Pass through to all downstream blocks
    for (uint32_t i {0U}; connDstTotal > i; ++i) {
        getDst(i).srcRecvPacketDelete(handle);
    }

    LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

// Default implementation of SrcBlockInterface::dstRecvPacketDelete,
// which always triggers LwSciError_NotImplemented error event.
void Block::dstRecvPacketDelete(
    uint32_t const dstIndex,
    LwSciStreamPacket const handle) noexcept
{
    static_cast<void>(dstIndex);
    static_cast<void>(handle);
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    setErrorEvent(LwSciError_NotImplemented);
}

// Default implementation of SrcBlockInterface::srcRecvPacketStatus,
// which always triggers LwSciError_NotImplemented error event.
void Block::srcRecvPacketStatus(
    uint32_t const srcIndex,
    Packet const& origPacket) noexcept
{
    static_cast<void>(srcIndex);
    static_cast<void>(origPacket);
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    setErrorEvent(LwSciError_NotImplemented);
}

//! <b>Sequence of operations</b>
//! - Checks whether this is a direct one-to-many link.
//! - Checks whether the block expects to process the status itself.
//! - Calls dstRecvPacketStatus() on all upstream connections to
//!   pass on the original Packet status.
void Block::dstRecvPacketStatus(
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

    // Override required if not a one-to-many link.
    if (!isDstOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // Override required if this block needs the status
    if (FillMode::None != pktDesc.statusConsFillMode) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // Pass through to all upstream blocks
    for (uint32_t i {0U}; connSrcTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getSrc(i).dstRecvPacketStatus(origPacket);
    }
}

//! <b>Sequence of operations</b>
//! - Checks whether this is a direct one-to-many link.
//! - Calls srcRecvPacketsComplete() on all downstream connections to
//!   pass on the message.
void Block::srcRecvPacketsComplete(
    uint32_t const srcIndex) noexcept
{
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

    // Override required if not a one-to-many link.
    if (!isSrcOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // Pass through to all downstream blocks
    for (uint32_t i {0U}; connDstTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getDst(i).srcRecvPacketsComplete();
    }

    // Advance setup phase
    phasePacketsDoneSet();
}

// Default implementation of SrcBlockInterface::dstRecvPacketsComplete,
// which always triggers LwSciError_NotImplemented error event.
void Block::dstRecvPacketsComplete(
    uint32_t const dstIndex) noexcept
{
    static_cast<void>(dstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    setErrorEvent(LwSciError_NotImplemented);
}

//
// Sync waiter functions
//

// Default implementation of APIBlockInterface::apiElementWaiterAttrSet
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiElementWaiterAttrSet(
    uint32_t const elemIndex,
    LwSciWrap::SyncAttr const& syncAttr) noexcept
{
    static_cast<void>(elemIndex);
    static_cast<void>(syncAttr);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiElementWaiterAttrGet
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiElementWaiterAttrGet(
    uint32_t const elemIndex,
    LwSciWrap::SyncAttr& syncAttr) noexcept
{
    static_cast<void>(elemIndex);
    static_cast<void>(syncAttr);
    return LwSciError_NotSupported;
}

//! <b>Sequence of operations</b>
//! - Checks whether this is a direct one-to-many link.
//! - Calls srcRecvSyncWaiter() on all downstream connections to
//!   pass on the message.
void Block::srcRecvSyncWaiter(
    uint32_t const srcIndex,
    Waiters const& syncWaiter) noexcept
{
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

    // Override required if not a one-to-many link.
    if (!isSrcOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // Pass through to all downstream blocks
    for (uint32_t i {0U}; connDstTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getDst(i).srcRecvSyncWaiter(syncWaiter);
    }
}

//! <b>Sequence of operations</b>
//! - Checks whether this is a direct one-to-many link.
//! - Calls dstRecvSyncWaiter() on all upstream connections to
//!   pass on the message.
void Block::dstRecvSyncWaiter(
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

    // Override required if not a one-to-many link.
    if (!isDstOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // Pass through to all upstream blocks
    for (uint32_t i {0U}; connSrcTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getSrc(i).dstRecvSyncWaiter(syncWaiter);
    }
}

//
// Sync signaller functions
//

// Default implementation of APIBlockInterface::apiElementSignalObjSet
// which always returns LwSciError_NotSupported error code.
LwSciError Block::apiElementSignalObjSet(
    uint32_t const elemIndex,
    LwSciWrap::SyncObj const& syncObj) noexcept
{
    static_cast<void>(elemIndex);
    static_cast<void>(syncObj);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiElementSignalObjGet
// which always returns LwSciError_NotSupported error code.
LwSciError Block::apiElementSignalObjGet(
    uint32_t const queryBlockIndex,
    uint32_t const elemIndex,
    LwSciWrap::SyncObj& syncObj) noexcept
{
    static_cast<void>(queryBlockIndex);
    static_cast<void>(elemIndex);
    static_cast<void>(syncObj);
    return LwSciError_NotSupported;
}

//! <b>Sequence of operations</b>
//! - Checks whether this is a direct one-to-many link.
//! - Calls srcRecvSyncSignal() on all downstream connections to
//!   pass on the message.
void Block::srcRecvSyncSignal(
    uint32_t const srcIndex,
    Signals const& syncSignal) noexcept
{
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

    // Override required if not a one-to-many link.
    if (!isSrcOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // Pass through to all downstream blocks
    for (uint32_t i {0U}; connDstTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getDst(i).srcRecvSyncSignal(syncSignal);
    }

    // Advance setup phase
    phaseProdSyncDoneSet();
}

//! <b>Sequence of operations</b>
//! - Checks whether this is a direct one-to-many link.
//! - Calls dstRecvSyncSignal() on all upstream connections to
//!   pass on the message.
void Block::dstRecvSyncSignal(
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

    // Override required if not a one-to-many link.
    if (!isDstOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // Pass through to all upstream blocks
    for (uint32_t i {0U}; connSrcTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getSrc(i).dstRecvSyncSignal(syncSignal);
    }

    // Advance setup phase
    phaseConsSyncDoneSet();
}

//
// Payload functions
//

// Default implementation of APIBlockInterface::apiPayloadObtain,
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiPayloadObtain(
    LwSciStreamCookie& cookie) noexcept
{
    static_cast<void>(cookie);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiPayloadReturn,
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiPayloadReturn(
    LwSciStreamPacket const handle) noexcept
{
    static_cast<void>(handle);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiPayloadFenceSet,
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiPayloadFenceSet(
    LwSciStreamPacket const handle,
    uint32_t const elemIndex,
    LwSciWrap::SyncFence const& postfence) noexcept
{
    static_cast<void>(handle);
    static_cast<void>(elemIndex);
    static_cast<void>(postfence);
    return LwSciError_NotSupported;
}

// Default implementation of APIBlockInterface::apiPayloadFenceGet,
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiPayloadFenceGet(
    LwSciStreamPacket const handle,
    uint32_t const queryBlockIndex,
    uint32_t const elemIndex,
    LwSciWrap::SyncFence& prefence) noexcept
{
    static_cast<void>(handle);
    static_cast<void>(queryBlockIndex);
    static_cast<void>(elemIndex);
    static_cast<void>(prefence);
    return LwSciError_NotSupported;
}

//! <b>Sequence of operations</b>
//! - Checks whether this is a direct one-to-many link.
//! - If the block uses packets:
//! -- Call blkMutexLock() to lock the block.
//! -- Call pktFindByHandle() to look up the local Packet instance.
//! -- Call Packet::LocationUpdate() to move the packet downstream.
//! -- Call Packet::fenceConsReset() to reset the consumer fences, if any.
//! - Calls srcRecvPayload() on all downstream connections to pass on
//!   the message.
void Block::srcRecvPayload(
    uint32_t const srcIndex,
    Packet const& prodPayload) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::Complete);
    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Override required if not a one-to-many link.
    if (!isSrcOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    {
        // Lock the mutex
        Lock const blockLock { blkMutexLock() };

        // Look up the packet
        PacketPtr const pkt { pktFindByHandle(prodPayload.handleGet(), true) };
        if (nullptr == pkt) {
            setErrorEvent(LwSciError_StreamBadPacket, true);
            return;
        }

        // Validate and update location
        if (!pkt->locationUpdate(Packet::Location::Upstream,
                                 Packet::Location::Downstream)) {
            setErrorEvent(LwSciError_StreamPacketInaccessible, true);
            return;
        }

        // Reset consumer fences
        pkt->fenceConsReset();
    }

    // Pass through to all downstream blocks
    //   The original payload is used because it has the fences
    for (uint32_t i {0U}; connDstTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getDst(i).srcRecvPayload(prodPayload);
    }
}

//! <b>Sequence of operations</b>
//! - Makes sure that stream disconnect is not yet done by calling
//!   Block::connComplete() interface. If connComplete() call is failed,
//!   the function returns immediately without performing any action.
//! - Checks whether this is a direct one-to-many link.
//! - If the block uses packets:
//! -- Call blkMutexLock() to lock the block.
//! -- Call pktFindByHandle() to look up the local Packet instance.
//! -- Call Packet::LocationUpdate() to move the packet upstream.
//! -- Call Packet::fenceProdReset() to reset the producer fences, if any.
//! - Calls dstRecvPayload() on all upstream connections to pass on
//!   the message.
void Block::dstRecvPayload(
    uint32_t const dstIndex,
    Packet const& consPayload) noexcept
{
    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Stream must be fully connected
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SafetyPhase);
    validation.set(ValidateCtrl::DstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, dstIndex)) {
        return;
    }

    // If disconnect has oclwrred, return but don't report an error
    // TODO: Need to distinguish various disconnect cases and handle
    //       in the validate function above.
    if (!connComplete()) {
        return;
    }

    // Override required if not a one-to-many link.
    if (!isDstOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    {
        // Lock the mutex
        Lock const blockLock { blkMutexLock() };

        // Look up the packet
        PacketPtr const pkt { pktFindByHandle(consPayload.handleGet(), true) };
        if (nullptr == pkt) {
            setErrorEvent(LwSciError_StreamBadPacket, true);
            return;
        }

        // Validate and update location
        if (!pkt->locationUpdate(Packet::Location::Downstream,
                                 Packet::Location::Upstream)) {
            setErrorEvent(LwSciError_StreamPacketInaccessible, true);
            return;
        }

        // Reset producer fences
        pkt->fenceProdReset();
    }

    // Pass through to all upstream blocks
    //   The original payload is used because it has the fences
    for (uint32_t i {0U}; connSrcTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getSrc(i).dstRecvPayload(consPayload);
    }
}

// Default implementation of DstBlockInterface::srcDequeuePayload, which
//  always sets a LwSciError_NotImplemented error and returns an empty pointer.
PacketPtr Block::srcDequeuePayload(
    uint32_t const srcIndex) noexcept
{
    static_cast<void>(srcIndex);
    setErrorEvent(LwSciError_NotImplemented);
    return PacketPtr();
}

// Default implementation of SrcBlockInterface::dstDequeuePayload, which
//  always sets a LwSciError_NotImplemented error and returns an empty pointer.
PacketPtr Block::dstDequeuePayload(
    uint32_t const dstIndex) noexcept
{
    static_cast<void>(dstIndex);
    setErrorEvent(LwSciError_NotImplemented);
    return PacketPtr();
}

//! <b>Sequence of operations</b>
//!  - Creates a new Block::Lock object with blkMutexLock().
//!  - Checks for any pending event and returns it. If no event
//!    is pending, and LwSciEventService is not used, it
//!    waits for the given timeout period by calling Block::eventWait()
//!    interface. If no new event is available even after the
//!    timeout period, it returns LwSciError_Timeout.
//!    The pending event is retrieved in the following order:
//!   - LwSciStreamEventType_Error event,
//!   - LwSciStreamEventType_Connected event,
//!   - any block-specific event calling pendingEvent() interface,
//!   - LwSciStreamEventType_Disconnected event.
//!   - If the error is not Success, return error to application instead
//!     of the event.
//!
//! \implements{18794355}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError
Block::getEvent(
    int64_t const         timeout_usec,
    LwSciStreamEventType& event) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Note: Can be used any time after block is created.
    // Timeout must be zero if the block is configured to
    // use LwSciEventService.
    if (EventSignalMode::EventService == signalMode.load()) {
        if (0L != timeout_usec) {
            return LwSciError_BadParameter;
        }
    }

    // Compute time to wait until
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A0_1_1), "Bug 2738197")
    ClockTime  timeout  { std::chrono::steady_clock::now() };
    ClockTime* pTimeout { nullptr };
    if (timeout_usec >= 0L) {
        timeout += std::chrono::microseconds(timeout_usec);
        pTimeout = &timeout;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A0_1_1))

    // Must hold lock during the entire event check to ensure
    //   signals to the condition variable aren't missed if an
    //   event arrives partway through checking. This also
    //   ensures the stability of packet maps while iterating
    //   over them.
    // TODO: Because there seems to be no good alternative to holding
    //       the lock, some of the atomic operations used by the
    //       tracking objects may be unnecessary. We should look
    //       into eliminating any that aren't needed.
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    Lock blockLock { blkMutexLock() };
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))

    // Loop until event is found or timeout
    // Note: Spurious wakes are possible. If no event is found we just
    //       go back to waiting.
    do {

        // See if there's a pending error event
        bool expected{ true };
        if (errEvent.compare_exchange_strong(expected, false)) {
            event = LwSciStreamEventType_Error;
            return LwSciError_Success;
        }

        // See if there's a pending connection event
        expected = true;
        if (connEvent.compare_exchange_strong(expected, false)) {
            event = LwSciStreamEventType_Connected;
            return LwSciError_Success;
        }

        // Check for block-specific events only if stream is complete
        if (streamDone) {
            if (pendingEvent(event)) {
                return LwSciError_Success;
            }
        }

        // Check for setup completion event
        if (phaseEvent) {
            phaseEvent = false;
            event = LwSciStreamEventType_SetupComplete;
            return LwSciError_Success;
        }

        // See if there's a pending disconnection event
        expected = true;
        if (discEvent.compare_exchange_strong(expected, false)) {
            event = LwSciStreamEventType_Disconnected;
            return LwSciError_Success;
        }
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A6_5_3), "LwSciStream-ADV-AUTOSARC++14-004")
    } while (eventWait(blockLock, pTimeout));

    // Report timeout
    return LwSciError_Timeout;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! <b>Sequence of operations</b>
//! - Retrieves the error code under thread protection provided by
//!   Block::blkMutexLock() and clears it.
LwSciError Block::apiErrorGet(void) noexcept
{
    // Take the lock
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    Lock const blockLock { blkMutexLock() };

    // Report error and clear it to allow new errors.
    LwSciError const status{ internalErr };
    internalErr = LwSciError_Success;
    return status;
}

// Default implementation of APIBlockInterface::disconnect,
// which always returns LwSciError_NotImplemented error code.
LwSciError Block::disconnect(void) noexcept
{
    return LwSciError_NotImplemented;
}

//! <b>Sequence of operations</b>
//!  - Atomically sets up the block to use internal event-signaling mode if
//!    the event-signaling mode has not been set up yet.
//!
//! \implements{21698749}
void Block::eventDefaultSetup(void) noexcept
{
    // If client didn't configure LwSciEventService on this block,
    // set the block to use event-query wait.
    EventSignalMode lwrr { EventSignalMode::None };
    static_cast<void>(
        signalMode.compare_exchange_strong(lwrr, EventSignalMode::Internal)
    );
}

//! <b>Sequence of operations</b>
//! - Creates a new Block::Lock object with blkMutexLock() to protect the entire
//!   call against multithread access.
//!  - If the event-signaling mode has not been set up yet, atomically
//!    sets up the block to use LwSciEventService for event signaling, and
//!    creates an LwSciEventNotifier instance associated with the block by
//!    calling LwSciEventService::CreateLocalEvent(), and returns the pointer
//!    to the LwSciEventNotifier instance.
//!
//! \implements{21698764}
EventSetupRet
Block::eventNotifierSetup(LwSciEventService& eventService) noexcept
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    Lock const blockLock { blkMutexLock() };
    EventSetupRet rv { nullptr, LwSciError_IlwalidState };

    EventSignalMode lwrr { EventSignalMode::None };
    if (signalMode.compare_exchange_strong(
        lwrr, EventSignalMode::EventService)) {
        // The event service has to be configured before any API call
        // into the block and can be only configured once.
        rv.err = eventService.CreateLocalEvent(
                &eventService, &eventServiceEvent);
        if (LwSciError_Success == rv.err) {
            rv.eventNotifier = eventServiceEvent->eventNotifier;
        }
    }
    return rv;
}

// Default implementation of APIBlockInterface::apiUserInfoSet
//   which always returns LwSciError_NotSupported error code.
LwSciError Block::apiUserInfoSet(
    uint32_t const userType,
    InfoPtr const& info) noexcept
{
    static_cast<void>(userType);
    static_cast<void>(info);
    return LwSciError_NotSupported;
}

//! <b>Sequence of operations</b>
//!  - Call EndInfoVector:getVector() to get the aclwmulated vector of
//!    EndInfo on prodInfo if @a queryBlockType is
//!    LwSciStreamBlockType_Producer or on consInfo if @a queryBlockType is
//!    LwSciStreamBlockType_Consumer.
//!  - Retrieve the list of queried information by @a queryBlockIndex.
//!  - Call EndInfo::infoGet() to get the queried information.
LwSciError Block::apiUserInfoGet(
    LwSciStreamBlockType const queryBlockType,
    uint32_t const queryBlockIndex,
    uint32_t const userType,
    InfoPtr& info) noexcept
{
    ValidateBits validation{};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::CompleteQueried);
    LwSciError const err{ validateWithError(validation) };
    if (LwSciError_Success != err) {
        return err;
    }

    if (phaseRuntime) {
        return LwSciError_NoLongerAvailable;
    }

    bool isProdType;
    if (LwSciStreamBlockType_Producer == queryBlockType) {
        isProdType = true;
    } else if (LwSciStreamBlockType_Consumer == queryBlockType) {
        isProdType = false;
    } else {
        return LwSciError_BadParameter;
    }

    EndInfoVector const& endInfoVec{ isProdType ? prodInfo : consInfo };
    if (endInfoVec.size() <= queryBlockIndex) {
        return LwSciError_IndexOutOfRange;
    }
    EndInfo const& endInfo{ endInfoVec[queryBlockIndex] };

    return endInfo.infoGet(userType, info);
}

//! <b>Sequence of operations</b>
//!  - If @a locked is false then tries to lock blkMutex before returning the
//!    lock instance. If locking fails, then panics by calling lwscistreamPanic().
//!  - If @a locked is true, returns the lock instance without performing
//!    any other action.
//!
//! \implements{18794673}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
Block::Lock Block::blkMutexLock(bool const locked) noexcept
{
    Lock blockLock { blkMutex, std::defer_lock };
    // Acquire the lock if not already locked.
    if (!locked) {
        try {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            blockLock.lock();
            LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
        } catch (std::system_error& e) {
            static_cast<void>(e);

            // Mutex locking may throw system_error.
            // Abort in safety build.
            lwscistreamPanic();
            // For nonsafety build, assert in debug build
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
            assert(false);
        }
    }
    return blockLock;
};
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))

//! <b>Sequence of operations</b>
//! - Check whether already set, and if not copy the incoming information.
LwSciError Block::prodInfoSet(
    EndInfoVector const& info) noexcept
{
    // Check if already done
    if (0U != prodInfo.size()) {
        return LwSciError_AlreadyDone;
    }

    // Copy into local vector
    try {
        prodInfo = info;
    } catch (...) {
        return LwSciError_InsufficientMemory;
    }

    // Set flag that data is ready to send downstream
    prodInfoMsg = true;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Check whether already set, and if not copy the incoming information.
LwSciError Block::consInfoSet(
    EndInfoVector const& info) noexcept
{
    // Check if already done
    if (0U != consInfo.size()) {
        return LwSciError_AlreadyDone;
    }

    // Copy into local vector
    try {
        consInfo = info;
    } catch (...) {
        return LwSciError_InsufficientMemory;
    }

    // Save count and set flag that data is ready to send upstream
    consumerCount = consInfo.size();
    pktDesc.numConsumer = consumerCount;
    consInfoMsg = true;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call srcRecvProdInfo() on all destination blocks to send the producer
//!   info downstream.
//! - Call eventPost() to signal a connection event is ready.
void Block::prodInfoFlow(void) noexcept
{
    // Skip if already sent
    if (!prodInfoMsg) {
        return;
    }

    // If downstream connection is not remote, send info
    if (!connDstRemote) {

        // Send producer info to all downstream blocks
        for (uint32_t i {0U}; connDstTotal > i; ++i) {
            LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
            getDst(i).srcRecvProdInfo(prodInfo);
        }

    }

    // Mark stream connected and post event
    assert(!connEvent);
    streamDone = true;
    connEvent = true;
    eventPost(false);

    // Clear flag
    prodInfoMsg = false;
}

//! <b>Sequence of operations</b>
//! - Call SafeConnection::isConnected() on all source blocks to check
//!   whether upstream connection completed.
//! - If completed, call dstRecvConsInfo() on all source blocks to send
//!   the consumer info upstream.
void Block::consInfoFlow(void) noexcept
{
    // Skip if already sent
    if (!consInfoMsg) {
        return;
    }

    // If upstream connection is not remote, send info
    if (!connSrcRemote) {

        // First check whether all source connections are ready
        for (uint32_t i {0U}; i < connSrcTotal; i++) {
            if (!connSrc[i].isConnected()) {
                return;
            }
        }

        // Send consumer info to all upstream blocks
        for (uint32_t i {0U}; connSrcTotal > i; ++i) {
            LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
            getSrc(i).dstRecvConsInfo(consInfo);
        }
    }

    // Clear flag
    consInfoMsg = false;
}

//! <b>Sequence of operations</b>
//! - Call isSrcOneToMany() to verify this is a direct one-to-many link.
//! - Call Block::blkMutexLock() to lock the mutex and call prodInfoSet()
//!   to copy the vector.
//! - Call prodInfoFlow() to send the info downstream and trigger event
//!   if appropriate.
void Block::srcRecvProdInfo(
    uint32_t const srcIndex,
    EndInfoVector const& info) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Validate block/input state
    //   In safety builds, only allowed in setup
    //   Index must be valid
    ValidateBits validation {};
    validation.set(ValidateCtrl::SetupPhase);
    validation.set(ValidateCtrl::SrcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    if (!validateWithEvent(validation, srcIndex)) {
        return;
    }

    // Override required if not a one-to-many link.
    if (!isSrcOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // Lock and save the info
    {
        Lock const blockLock { blkMutexLock() };
        LwSciError const err { prodInfoSet(info) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }
    }

    // Pass info on and trigger event if appropriate
    prodInfoFlow();

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call isDstOneToMany() to verify this is a direct one-to-many link.
//! - Call Block::blkMutexLock() to lock the mutex and call consInfoSet()
//!   to copy the vector.
//! - Call consInfoFlow() to send the info upstream if appropriate.
void Block::dstRecvConsInfo(
    uint32_t const dstIndex,
    EndInfoVector const& info) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

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

    // Override required if not a one-to-many link.
    if (!isDstOneToMany()) {
        setErrorEvent(LwSciError_NotImplemented);
        return;
    }

    // Lock and save the info
    {
        Lock const blockLock { blkMutexLock() };
        LwSciError const err { consInfoSet(info) };
        if (LwSciError_Success != err) {
            setErrorEvent(err, true);
            return;
        }
    }

    // Pass info on if appropriate
    consInfoFlow();

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Call Block::eventDefaultSetup() to set the default event-notification
//!   mode of the block if not configured to use LwSciEventService.
void Block::finalizeConfigOptions(void) noexcept
{
    // Finalize the block configuration at the first connection
    bool expected{ false };
    if (!configOptLocked.compare_exchange_strong(expected, true)) {
        return;
    }

    // Sets the default event-notification mode
    eventDefaultSetup();
}

// Default implementation of SrcBlockInterface::dstDisconnect,
// which always triggers LwSciError_NotImplemented error event.
void Block::dstDisconnect(
    uint32_t const dstIndex) noexcept
{
    static_cast<void>(dstIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    setErrorEvent(LwSciError_NotImplemented);
}

// Default implementation of DstBlockInterface::srcDisconnect,
// which always triggers LwSciError_NotImplemented error event.
void Block::srcDisconnect(
    uint32_t const srcIndex) noexcept
{
    static_cast<void>(srcIndex);
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    setErrorEvent(LwSciError_NotImplemented);
}

// Checks if the blockRegistry has been allocated.
// If not, allocates it.
bool Block::prepareRegistry(void) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blockRegistry) {
        try {
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            blockRegistry = std::make_unique<HandleBlockPtrMap>();
        } catch (std::bad_alloc& e) {
            static_cast<void>(e);
            return false;
        }
    }
    return true;
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

// Registers a new block instance by adding an entry in
// blockRegistry, with LwSciStreamBlock as the key and BlockPtr as the
// mapped value. If the blockRegistry has not yet been created, create it.
bool Block::registerBlock(BlockPtr const& blkPtr) noexcept
{
    // Lock registry
    try {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
        std::unique_lock<std::mutex> const lock{registryMutex};

        if (!prepareRegistry()) {
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            return false;
        }
        // Insert in registry
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
        std::pair<HandleBlockPtrMapIterator, bool> const ret {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            blockRegistry->emplace(blkPtr->getHandle(), blkPtr)
            LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
        };

        return ret.second;
    } catch (std::system_error& e) {
        // Mutex locking may throw system_error.
        static_cast<void>(e);
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        return false;
    } catch (std::bad_alloc& e) {
        // Insertion into blockRegistry may throw bad_alloc.
        static_cast<void>(e);
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        return false;
    }
}

//! <b>Sequence of operations</b>
//!  - Creates an instance of Lock (std::unique_lock<std::mutex>) by
//!    passing the registryMutex as an argument to its constructor.
//!  - Upon successful locking, searches the blockRegistry to retrieve the
//!    BlockPtr corresponding to the given @a handle.
//!  - If locking fails then panics by calling lwscistreamPanic().
//!
//! \implements{18794658}
BlockPtr Block::getRegisteredBlock(LwSciStreamBlock const handle) noexcept
{
    // Lock registry
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    Lock registryLock { registryMutex, std::defer_lock };
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        registryLock.lock();
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))

        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != blockRegistry) {
            // Look up handle
            HandleBlockPtrMap::iterator const mapEntry
                { blockRegistry->find(handle) };
            if (blockRegistry->cend() != mapEntry) {
                return mapEntry->second;
            }
        }
    } catch (std::system_error& e) {
        static_cast<void>(e);
        // Mutex locking may throw system_error
        // Proceed to error handling below.
        lwscistreamPanic();
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))

    // Return null if not found
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    return static_cast<BlockPtr>(nullptr);
}

//! <b>Sequence of operations</b>
//!  - Creates an instance of Lock (std::unique_lock<std::mutex>) by
//!    passing the registryMutex as an argument to its constructor.
//!  - Upon successful locking, Unregisters a block instance by removing
//!    the entry corresponding to the input @a handle from blockRegistry.
//!  - If locking fails then panics by calling lwscistreamPanic().
//!
//! \implements{18794664}
void Block::removeRegisteredBlock(LwSciStreamBlock const handle) noexcept
{
    // Lock registry
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    Lock registryLock { registryMutex, std::defer_lock };
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        registryLock.lock();
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::system_error& e) {
        static_cast<void>(e);
        // Mutex locking may throw system_error.
        // Abort for invalid handle in safety build or
        // NOP for non-safety build.
        lwscistreamPanic();
        return;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr != blockRegistry) {
        static_cast<void>(blockRegistry->erase(handle));
    }
}

//! <b>Sequence of operations</b>
//!  - Deletes the LwSciLocalEvent instance referenced by eventServiceEvent.
//!
//! \implements{21698808}
LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
Block::~Block(void) noexcept
{
    if (nullptr == eventServiceEvent) {
        return;
    }
    eventServiceEvent->Delete(eventServiceEvent);
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))

//! <b>Sequence of operations</b>
//!  - Creates a new Block::Lock object with blkMutexLock() to protect the entire
//!    call against multithread access.
//!  - If using LwSciEventService, calls LwSciLocalEvent::Signal()
//!    on eventServiceEvent variable to wake up any threads waiting on the
//!    associated LwSciEventNotifier instance. If fails to signal then panics
//!    by calling lwscistreamPanic().
//!  - Otherwise signals the eventCond variable to wake up any threads waiting
//!    on it.
//!
//! \implements{18794715}
void
Block::eventPost(
    bool const locked) noexcept
{
    // If lock is not already held, take it
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    Lock const blockLock { blkMutexLock(locked) };

    switch(signalMode.load()) {
    case EventSignalMode::Internal:
        // Wake any other threads waiting for an event
        eventCond.notify_all();
        break;
    case EventSignalMode::EventService:
        // Signal the LwSciEvent event object.
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
        assert (nullptr != eventServiceEvent);
        if (LwSciError_Success !=
            eventServiceEvent->Signal(eventServiceEvent)) {
            lwscistreamPanic();
        }
        break;
    case EventSignalMode::None: // Fallthrough
    default:
        break;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
}

//! <b>Sequence of operations</b>
//!  - If not using LwSciEventService, waits for the signal of eventCond
//!    variable.
//!
//! \implements{18794724}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
bool
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
Block::eventWait(
    Lock&                  blockLock,
    ClockTime const* const timeout) noexcept
{
    if (EventSignalMode::EventService == signalMode.load()) {
        // No waiting of events if the block uses LwSciEventService to
        // notify the client the availability of events.
        return false;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    if (nullptr == timeout) {
        // If no end time specified, wait for condition forever
        eventCond.wait(blockLock);
    } else {
        // If end time specified, wait for condition until time
        if (std::cv_status::timeout ==
            eventCond.wait_until(blockLock, *timeout)) {
            return false;
        }
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
    return true;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))

// Block-specific query for the next pending LwSciStreamEvent.
// Block class provides the default implementation for this
// interface which always returns false. The derived classes
// can override this interface as required.
bool
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
Block::pendingEvent(
    LwSciStreamEventType& event) noexcept
{
    static_cast<void>(event);

    // Base block class has no events beyond those handled in getEvent
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    return false;
}

//! <b>Sequence of operations</b>
//!  - Create a new Block::Lock object with blkMutexLock() to protect the
//!    entire call against multithread access.
//!  - If a LwSciStreamCookie is provided, call pktFindByCookie() with
//!    parameter 'locked' set to true, to ensure it is not in use by another
//!    Packet.
//!  - Create a new packet instance with the Block::pktDesc.
//!  - Call Packet::InitErrorGet() to ensure initialization succeeded.
//!  - If @a origPacket is provided, call Packet::defineCopy() to copy
//!    the portions of the packet's definition that are relevant to this block.
//!
//! \implements{18796275}
LwSciError
Block::pktCreate(
    LwSciStreamPacket const handle,
    Packet const* const     origPacket,
    LwSciStreamCookie const cookie) noexcept
{
    // Take lock
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    Lock const blockLock { blkMutexLock() };

    // If cookie is provided, make sure there isn't already a packet using it
    if (LwSciStreamCookie_Ilwalid != cookie) {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != pktFindByCookie(cookie, true)) {
            return LwSciError_AlreadyInUse;
        }
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
    try {
        // Create new packet
        PacketPtr const pkt
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
            { std::make_shared<Packet>(pktDesc, handle, cookie) };

        // Check for initialization failure
        if (LwSciError_Success != pkt->initErrorGet()) {
            return pkt->initErrorGet();
        }

        // Copy definition from original, if provided
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != origPacket) {
            LwSciError const err { pkt->defineCopy(*origPacket, true) };
            if (LwSciError_Success != err) {
                return err;
            }
        }
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

        // Insert in map and return on success.
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
        std::pair<PacketIter, bool> const result
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
            { pktMap.emplace(handle, pkt) };
        if (result.second) {
            return LwSciError_Success;
        }

        // See if failure is because handle already existed in map.
        //   This means we did something wrong.
        if (pktMap.end() != pktMap.find(handle)) {
            return LwSciError_StreamInternalError;
        }
    } catch (std::bad_alloc& e) {
        // Packet creation may throw bad_alloc.
        static_cast<void>(e);
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
    // Return allocation failure
    return LwSciError_InsufficientMemory;
}

// Removes a packet instance with the given LwSciStreamPacket
// from PacketMap.
void
Block::pktRemove(
    LwSciStreamPacket const handle,
    bool const              locked) noexcept
{
    // Take lock if not held by caller
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    Lock const blockLock { blkMutexLock(locked) };

    // Erase from map
    size_t const size { pktMap.erase(handle) };
    static_cast<void>(size);
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
    assert(1ULL == size);
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
}

// Searches PacketMap for a packet instance with the given
// LwSciStreamPacket and returns the smart pointer to it if found.
PacketPtr
Block::pktFindByHandle(
    LwSciStreamPacket const handle,
    bool const              locked) noexcept
{
    // If lock is not already held, take it
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    Lock const blockLock { blkMutexLock(locked) };

    // Look up packet
    PacketPtr pkt {};
    PacketMap::iterator const iter { pktMap.find(handle) };
    if (pktMap.cend() != iter) {
        pkt = iter->second;
    }
    return pkt;
};

//! <b>Sequence of operations</b>
//!  - Creates a new Block::Lock object with blkMutexLock() to protect the entire
//!    call against multithread access.
//!  - Retrieves the cookie associated with each packet using Packet::cookieGet()
//!    and returns smart pointer to packet instance if cookie matches with the
//!    given desired LwSciStreamCookie.
//!
//! \implements{18796284}
//!
PacketPtr
Block::pktFindByCookie(
    LwSciStreamCookie const cookie,
    bool const              locked) noexcept
{
    // Take lock if not held by caller
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    Lock const blockLock { blkMutexLock(locked) };

    // Iterate through the list, looking for a packet with desired cookie
    PacketPtr pkt {};
    for (PacketIter x { pktMap.begin() }; pktMap.end() != x; ++x) {
        if (x->second->cookieGet() == cookie) {
            pkt = x->second;
            break;
        }
    }
    return pkt;
};

//! <b>Sequence of operations</b>
//! - Creates a Block::Lock object with blkMutexLock() to protect the entire
//!   call against multithread access.
//! - Iterates all packet instances in PacketMap and sets the event type
//!   if any packet instance has a pending event by the call to the
//!   Packet::pendingEvent() interface.
//!
//! \implements{18796287}
PacketPtr
Block::pktPendingEvent(
    Packet::Pending const criteria,
    bool const            locked) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")

    // Take lock if not held by caller
    Lock const blockLock { blkMutexLock(locked) };

    // Search all packets for one that meets criteria
    for (PacketIter x { pktMap.begin() }; pktMap.end() != x; ++x) {
        PacketPtr const pkt { x->second };
        if (pkt->ilwokePending(criteria)) {
            return pkt;
        }
    }

    // Empty pointer on failure
    return PacketPtr();

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
}

//! <b>Sequence of operations</b>
//! - Check all requirments for readiness, returning if any not met.
//! - If not already done, call phaseSendReady() to pass the message along.
void Block::phaseCheck(void) noexcept
{
    // Quick return if already done
    if (phaseSrcSend.load()) {
        return;
    }

    // Packets and syncs must be done, and all downstream blocks must be ready
    if (!(phasePacketsDone && (phaseProdSyncDone && phaseConsSyncDone)) ||
        (connDstTotal != phaseDstRecv)) {
        return;
    }

    // First thread to reach here takes responsibility for sending the message
    bool expected { false };
    if (phaseSrcSend.compare_exchange_strong(expected, true)) {
        phaseSendReady();
    }
}

//! <b>Sequence of operations</b>
//! - Mark phase as changed and signal pending event.
//! - Call phaseSendChange() to pass the message along.
//! - Call EndInfoTrack::clear() to clear the producer and consumer
//!   endpoint info.
//! - Call eventPost() to wake any waitiing threads.
void Block::phaseChange(void) noexcept
{
    phaseEvent = true;
    phaseRuntime = true;
    phaseSendChange();

    // Clear the endpoint info
    prodInfo.clear();
    consInfo.clear();

    eventPost(false);
}

//! <b>Sequence of operations</b>
//! - Ilwoke dstRecvPhaseReady() interface of source block(s).
void Block::phaseSendReady(void) noexcept
{
    for (uint32_t i {0U}; connSrcTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getSrc(i).dstRecvPhaseReady();
    }
}

//! <b>Sequence of operations</b>
//! - Ilwoke srcRecvPhaseChange() interface of destination block(s).
void Block::phaseSendChange(void) noexcept
{
    for (uint32_t i {0U}; connDstTotal > i; ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        getDst(i).srcRecvPhaseChange();
    }
}

//! <b>Sequence of operations</b>
//! - Increment number of ready outputs.
//! - Call phaseDstDoneSet()) to mark dstIndex done and check whether entire
//!   block is ready.
void Block::dstRecvPhaseReady(
    uint32_t const dstIndex) noexcept
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

    // Mark done and check readiness
    phaseDstDoneSet(dstIndex);
}

//! <b>Sequence of operations</b>
//! - Call phaseSrcDoneSet() to mark srcIndex done and trigger phase change
//!   for block.
void Block::srcRecvPhaseChange(
    uint32_t const srcIndex) noexcept
{
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

    // Mark done and change phase
    phaseSrcDoneSet(srcIndex);
}

} // namespace LwSciStream
