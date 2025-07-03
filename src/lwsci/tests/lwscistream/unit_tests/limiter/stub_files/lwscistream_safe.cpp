// \file
// \brief LwSciStream public APIs definition for non-safety build.
//
// \copyright
// Copyright (c) 2021 LWPU Corporation. All rights reserved.
//
// LWPU Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from LWPU Corporation is strictly prohibited.
#include <cstdint>
#include <new>
#include <memory>
#include "covanalysis.h"
#include "lwscierror.h"
#include "lwsciipc.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwscistream_types.h"
#include "lwscistream_api.h"
#include "apiblockinterface.h"
#include "lwscistream_common.h"
#include "ipcsrc.h"
#include "ipcdst.h"

using LwSciStream::Block;
using LwSciStream::BlockPtr;
using LwSciStream::IpcSrc;
using LwSciStream::IpcDst;

extern "C" {

LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_4_4), "Bug 3127842")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M0_1_10), "LwSciStream-ADV-AUTOSARC++14-006")

/**
 *  <b>Sequence of operations</b>
 *    - Creates an IpcSrc block instance and checks whether the initialization
 *      is successful by calling its Block::isInitSuccess() interface.
 *    - Registers the IpcSrc block instance by calling Block::registerBlock()
 *      interface.
 *    - Call Block::finalizeConfigOptions() to lock the configuration options
 *      of the queue block.
 *    - Retrieves the LwSciStreamBlock referencing the created IpcSrc block
 *      instance by calling its Block::getHandle() interface and returns it.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamIpcSrcCreate2(
    LwSciIpcEndpoint const ipcEndpoint,
    LwSciSyncModule const syncModule,
    LwSciBufModule const bufModule,
    LwSciStreamBlock const queue,
    LwSciStreamBlock *const ipc
)
{
    // Create IpcSrc block

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == ipc) {
        return LwSciError_BadParameter;
    }

    if (LwSciStream::ILWALID_BLOCK_HANDLE != queue) {
        return LwSciError_BadParameter;
    }

    std::shared_ptr<IpcSrc> obj {};

    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_5_2), "Approved TID-482")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR50_CPP), "LwSciStream-ADV-CERTCPP-002")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        obj = std::make_shared<IpcSrc>(ipcEndpoint, syncModule, bufModule);
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR50_CPP))
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_5_2))
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))

    } catch (std::bad_alloc &e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }

    if (!obj->isInitSuccess()) {
        return LwSciError_StreamInternalError;
    }

    if (!obj->startDispatchThread()) {
        return LwSciError_StreamInternalError;
    }

    if (!Block::registerBlock(obj)) {
        return  LwSciError_StreamInternalError;
    }

    *ipc = obj->getHandle();

    return LwSciError_Success;
}

/**
 *  <b>Sequence of operations</b>
 *    - If the ipcEndpoint is of type IPC creates an IpcDst block instance
 *      and checks whether the initialization is successful by calling its
 *      Block::isInitSuccess() interface.
 *    - If the ipcEndpoint is of type C2C creates an C2CDst block instance
 *      and checks whether the initialization is successful by calling its
 *      Block::isInitSuccess() interface.
 *    - If it's the C2CDst block and the pool block is not provided, creates
 *      a AutomaticPool by default. Note that this pool block is not visible
 *      to application. Connects pool block instance with the C2CDst block
 *      by calling its C2CDst::BindPool() interface.
 *    - Launches the dispatch thread.
 *    - Registers the IpcDst/C2CDst block instance by calling Block::registerBlock()
 *      interface.
 *    - Call Block::finalizeConfigOptions() to lock the configuration options of
 *      the pool block.
 *    - Retrieves the LwSciStreamBlock referencing the created IpcDst/C2CDst block
 *      instance by calling its Block::getHandle() interface and returns it.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamIpcDstCreate2(
    LwSciIpcEndpoint const ipcEndpoint,
    LwSciSyncModule const syncModule,
    LwSciBufModule const bufModule,
    LwSciStreamBlock const pool,
    LwSciStreamBlock *const ipc
)
{
    // Create IpcDst block

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == ipc) {
        return LwSciError_BadParameter;
    }

    if (LwSciStream::ILWALID_BLOCK_HANDLE != pool) {
        return LwSciError_BadParameter;
    }

    std::shared_ptr<IpcDst> obj{};
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_5_2), "Approved TID-482")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR50_CPP), "LwSciStream-ADV-CERTCPP-002")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        obj = std::make_shared<IpcDst>(ipcEndpoint, syncModule, bufModule);
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR50_CPP))
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_5_2))
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }

    if (!obj->isInitSuccess()) {
        return LwSciError_StreamInternalError;
    }

    if (!obj->startDispatchThread()) {
        return LwSciError_StreamInternalError;
    }

    if (!Block::registerBlock(obj)) {
        return LwSciError_StreamInternalError;
    }

    *ipc = obj->getHandle();

    return LwSciError_Success;
}

LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M0_1_10))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_4_4))

} // extern "C"
