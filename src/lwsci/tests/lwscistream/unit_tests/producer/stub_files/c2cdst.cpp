//! \file
//! \brief LwSciStream C2C destination block definition.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.
#include <cstddef>
#include <iostream>
#include <array>
#include <unordered_map>
#include <cmath>
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
#include "safeconnection.h"
#include "pool.h"
#include "c2cdst.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Initialize the base IpcDst class and all members.
//! - Call LwSciSyncAttrListCreate() to create attributes lists for cpu
//!   and c2s waiting and signalling, and use them to create instances
//!   of LwSciWrap::SyncAttr.
//! - Call Wrapper::getErr() to check for any failures in allocation.
//! - Call Wrapper::viewVal() to retrieve the attribute lists and
//!   LwSciSyncAttrListSetAttrs() to populate their contents.
//! - Call LwSciSyncCpuWaitContextAlloc() to allocate a conext for CPU waits.
//! - Call IpcSrc::enqueueIpcWrite() to signal waiting attribute message.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A12_1_5), "Bug 2989775")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A3_1_1), "Bug 2758535")
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
C2CDst::C2CDst(LwSciIpcEndpoint const ipc,
               LwSciSyncModule const syncModule,
               LwSciBufModule const bufModule) noexcept :
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A3_1_1))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A12_1_5))
LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    IpcDst(ipc, syncModule, bufModule, true),
    secondaryPacketExportDone(false),
    secondaryPacketExportEvent(false),
    cpuWaiterAttr(),
    cpuSignalAttr(),
    c2cWaiterAttr(),
    c2cSignalAttr(),
    engineDoneWaiterAttr(),
    copyDoneSignalAttr(),
    readDoneSignalAttr(),
    copyDoneAttr(),
    readDoneSignalAttrMsg(false),
    cpuWaitAfterCopy(false),
    waiterAttrMsg(false),
    c2cCopySyncObj(),
    c2cCopySyncObjSend(false),
    waitContext(nullptr),
    c2cReusePayloadQueue(),
    numPayloadsAvailable(0U)
{
    // TODO: When C2C is complete, we may not need all of these attribute
    //       lists on both Src and Dst sides. We can trim some out if
    //       needed. For now, it is easiest to provide all of them for
    //       symmetry, and doing so will make it easier to consolidate
    //       duplicate Src/Dst code.

    // Create and wrap a CPU waiting attribute list.
    LwSciSyncAttrList attrList;
    LwSciError err { LwSciSyncAttrListCreate(syncModule, &attrList) };
    cpuWaiterAttr = LwSciWrap::SyncAttr(attrList, true, false, err);

    // Create and wrap a CPU signalling attribute list.
    err = LwSciSyncAttrListCreate(syncModule, &attrList);
    cpuSignalAttr = LwSciWrap::SyncAttr(attrList, true, false, err);

    // Create and wrap a C2C waiting attribute list.
    err = LwSciSyncAttrListCreate(syncModule, &attrList);
    c2cWaiterAttr = LwSciWrap::SyncAttr(attrList, true, false, err);

    // Create and wrap a C2C signalling attribute list.
    err = LwSciSyncAttrListCreate(syncModule, &attrList);
    c2cSignalAttr = LwSciWrap::SyncAttr(attrList, true, false, err);

    // Abort if any allocations failed
    if (((LwSciError_Success != cpuWaiterAttr.getErr()) ||
         (LwSciError_Success != cpuSignalAttr.getErr())) ||
        ((LwSciError_Success != c2cWaiterAttr.getErr()) ||
         (LwSciError_Success != c2cSignalAttr.getErr()))) {
        setInitFail();
        return;
    }

    // Attributes values
    bool const cpuAccess { true };
    LwSciSyncAccessPerm const waiterPerm { LwSciSyncAccessPerm_WaitOnly };
    LwSciSyncAccessPerm const signalPerm { LwSciSyncAccessPerm_SignalOnly };
    LwSciSyncAttrKeyValuePair pairCpuAccess
        { LwSciSyncAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess) };
    LwSciSyncAttrKeyValuePair pairWaiterPerm
        { LwSciSyncAttrKey_RequiredPerm, &waiterPerm, sizeof(waiterPerm) };
    LwSciSyncAttrKeyValuePair pairSignalPerm
        { LwSciSyncAttrKey_RequiredPerm, &signalPerm, sizeof(signalPerm) };
    std::array<LwSciSyncAttrKeyValuePair, 2U> pairs;

    // Fill CPU waiter attribute list
    pairs[0] = pairCpuAccess;
    pairs[1] = pairWaiterPerm;
    if (LwSciError_Success !=
        LwSciSyncAttrListSetAttrs(cpuWaiterAttr.viewVal(),
                                  pairs.data(), pairs.size())) {
        setInitFail();
    }

    // Fill CPU signaller attribute list
    pairs[0] = pairCpuAccess;
    pairs[1] = pairSignalPerm;
    if (LwSciError_Success !=
        LwSciSyncAttrListSetAttrs(cpuSignalAttr.viewVal(),
                                  pairs.data(), pairs.size())) {
        setInitFail();
    }

    // Fill C2C waiter attribute list
    if (LwSciError_Success !=
        LwSciSyncFillAttrsIndirectChannelC2c(ipc,
                                             c2cWaiterAttr.viewVal(),
                                             LwSciSyncAccessPerm_WaitOnly)) {
        setInitFail();
    }

    // Fill C2C signaller attribute list
    if (LwSciError_Success !=
        LwSciSyncFillAttrsIndirectChannelC2c(ipc,
                                             c2cSignalAttr.viewVal(),
                                             LwSciSyncAccessPerm_SignalOnly)) {
        setInitFail();
    }

    // Allocate context for CPU waiting
    if (LwSciError_Success !=
        LwSciSyncCpuWaitContextAlloc(syncModule, &waitContext)) {
        setInitFail();
    }

    // C2CDst uses CPU waits for consumer to finish reading
    // TODO: Would prefer C2C handle the waiting, but that is lwrrently
    //       not supported for this direction.
    engineDoneWaiterAttr = cpuWaiterAttr;

    // C2CDst uses CPU to signal to C2CSrc that read is done
    // TODO: Would prefer C2C handle the signalling, but that is lwrrently
    //       not supported for this direction.
    readDoneSignalAttr = cpuSignalAttr;
    readDoneSignalAttrMsg = true;

    // Signal attributes will be sent ASAP after connection
    if (LwSciError_Success != enqueueIpcWrite()) {
        setInitFail();
    }
}

// C2CDst block retrieves the associated pool block object.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError C2CDst::getOutputConnectPoint(
    BlockPtr& paramBlock) const noexcept
{
    paramBlock = pool;
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    return isInitSuccess() ? LwSciError_Success : LwSciError_NotInitialized;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))


//! <b>Sequence of operations</b>
//!   - Retrieves the BlockType of the given block
//!     instance by calling Block::getBlockType() interface
//!     and validates whether the returned BlockType is
//!     BlockType::POOL.
//!   - Mark the Pool as secondary pool by calling
//!     Pool::makeSecondary() interface.
//!   - Retrieves the handle to C2CDst block by calling
//!     Block::getHandle() interface and retrieves the C2CDst
//!     block instance by calling Block::getRegisteredBlock()
//!     interface.
//!   - Initializes the source connection of the pool block instance
//!     by calling Block::connSrcInitiate().
//!   - If successful, initializes the destination connection of the
//!     C2CDst block instance by calling Block::connDstInitiate().
//!   - If initialization of destination connection is not successful
//!     then cancels the source connection of the pool block instance
//!     by calling Block::connSrcCancel(). Otherwise, completes the
//!     source connection of pool block instance and destination
//!     connection of the C2CDst block instance by calling the
//!     Block::connSrcComplete() and Block::connDstComplete() interfaces
//!     respectively.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_11), "Bug 2738197")
LwSciError
C2CDst::BindPool(BlockPtr const& paramPool) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_11))
{
    // Validate pool
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if ((nullptr == paramPool) ||
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_5_1), "Bug 2807673")
        (BlockType::POOL != paramPool->getBlockType())) {
        return LwSciError_BadParameter;
    }

    // TODO: This cast might be an AUTOSAR violation. This needs to be
    // fixed in the next update.
    // Mark the pool as secondary Pool
    std::static_pointer_cast<Pool>(paramPool)->makeSecondary();

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
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    paramPool->connSrcComplete(srcReserved.index, dstReserved.index);
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    connDstComplete(dstReserved.index, srcReserved.index);

    pool = paramPool;

    return LwSciError_Success;
}

} // namespace LwSciStream
