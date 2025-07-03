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
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "lwsciipc_internal.h"
#include "lwscistream_types.h"
#include "lwscistream_api.h"
#include "apiblockinterface.h"
#include "lwscistream_common.h"
#include "pool.h"
#include "automaticpool.h"
#include "queue.h"
#include "ipcsrc.h"
#include "ipcdst.h"
#include "c2csrc.h"
#include "c2cdst.h"

using LwSciStream::BlockType;
using LwSciStream::Block;
using LwSciStream::BlockPtr;
using LwSciStream::Pool;
using LwSciStream::AutomaticPool;
using LwSciStream::Mailbox;
using LwSciStream::Fifo;
using LwSciStream::IpcSrc;
using LwSciStream::IpcDst;
using LwSciStream::C2CSrc;
using LwSciStream::C2CDst;

extern "C" {

LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_4_4), "Bug 3127842")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M0_1_10), "LwSciStream-ADV-AUTOSARC++14-006")

/**
 *  <b>Sequence of operations</b>
 *    - Call LwSciIpcEndpointGetTopoId() to check whether the @a ipcEndpoint
 *      is a IPC or C2C endpoint.
 *    - If the ipcEndpoint is of type IPC creates an IpcSrc block instance
 *      and checks whether the initialization is successful by calling its
 *      Block::isInitSuccess() interface.
 *    - If the ipcEndpoint is of type C2C and queue block is not provided, then
 *      the function create a FIFO queue by default and same is used in
 *      binding operation. Note that this block is not visible to application.
 *    - If the ipcEndpoint is of type C2C creates an C2CSrc block instance
 *      and checks whether the initialization is successful by calling its
 *      Block::isInitSuccess() interface and connects the queue block instance
 *      with the C2CSrc block by calling its C2CSrc::BindQueue() interface.
 *    - Launches the dispatch thread.
 *    - Registers the IpcSrc/C2CSrc block instance by calling
 *      Block::registerBlock() interface.
 *    - Call Block::finalizeConfigOptions() to lock the configuration options
 *      of the queue block.
 *    - Retrieves the LwSciStreamBlock referencing the created IpcSrc/C2CSrc
 *      block instance by calling its Block::getHandle() interface and returns
 *      it.
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
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == ipc) {
        return LwSciError_BadParameter;
    }

    // LwSciIpcEndpointGetTopoId() is not defined for safety builds
    LwSciIpcTopoId ipcTopoId;
    LwSciError const err{ LwSciIpcEndpointGetTopoId(ipcEndpoint, &ipcTopoId) };
    if (LwSciError_Success != err) {
        return err;
    }
    bool const isC2cBlock{ LWSCIIPC_SELF_SOCID != ipcTopoId.SocId };

    // Create IpcSrc block
    if (!isC2cBlock) {
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

    }

    // Create C2CSrc block
    else {
        BlockPtr queuePtr {};

        if (LwSciStream::ILWALID_BLOCK_HANDLE == queue) {
            // Create a fifo queue by default if queue block handle
            // is not provided.
            try {
                LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
                LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
                queuePtr = std::make_shared<Fifo>();
                LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
            } catch (std::bad_alloc& e) {
                static_cast<void>(e);
                return LwSciError_InsufficientMemory;
            }

            if (!queuePtr->isInitSuccess()) {
                return LwSciError_StreamInternalError;
            }
        } else {
            queuePtr = Block::getRegisteredBlock(queue);

            // Return if queue block handle is provided but not valid.
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            if (nullptr == queuePtr) {
                return LwSciError_BadParameter;
            }

            if (queuePtr->getBlockType() != LwSciStream::BlockType::QUEUE) {
                return LwSciError_BadParameter;
            }
        }

        std::shared_ptr<C2CSrc> obj {};

        try {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_5_2), "Approved TID-482")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR50_CPP), "LwSciStream-ADV-CERTCPP-002")
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            obj = std::make_shared<C2CSrc>(ipcEndpoint, syncModule, bufModule);
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

        // Bind queue block to C2CSrc block
        LwSciError const err { obj->BindQueue(queuePtr) };
        if (LwSciError_Success != err) {
            Block::removeRegisteredBlock(obj->getHandle());
            return err;
        }

        // Lock config options of queue block
        queuePtr->finalizeConfigOptions();

        *ipc = obj->getHandle();
    }
    return LwSciError_Success;
}

/**
 *  <b>Sequence of operations</b>
 *    - Call LwSciIpcEndpointGetTopoId() to check whether the @a ipcEndpoint
 *      is a IPC or C2C endpoint.
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
 *    - Registers the IpcDst/C2CDst block instance by calling
 *      Block::registerBlock() interface.
 *    - Call Block::finalizeConfigOptions() to lock the configuration options
 *      of the pool block.
 *    - Retrieves the LwSciStreamBlock referencing the created IpcDst/C2CDst
 *      block instance by calling its Block::getHandle() interface and returns
 *      it.
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
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == ipc) {
        return LwSciError_BadParameter;
    }

    // LwSciIpcEndpointGetTopoId() is not defined for safety builds
    LwSciIpcTopoId ipcTopoId;
    LwSciError const err{ LwSciIpcEndpointGetTopoId(ipcEndpoint, &ipcTopoId) };
    if (LwSciError_Success != err) {
        return err;
    }
    bool const isC2cBlock{ LWSCIIPC_SELF_SOCID != ipcTopoId.SocId };

    // Create IpcDst block
    if (!isC2cBlock) {
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

    }

    // Create C2CDst block
    else {
        BlockPtr poolPtr{};

        if (LwSciStream::ILWALID_BLOCK_HANDLE == pool) {
            // Create a AutomaticPooll if pool is not provided.
            try {
                LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
                LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
                poolPtr = std::make_shared<AutomaticPool>();
                LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
            } catch (std::bad_alloc& e) {
                static_cast<void>(e);
                return LwSciError_InsufficientMemory;
            }

            if (!poolPtr->isInitSuccess()) {
                return LwSciError_StreamInternalError;
            }
        } else {
            poolPtr = Block::getRegisteredBlock(pool);

            // Return if queue block handle is provided but not valid.
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            if (nullptr == poolPtr) {
                return LwSciError_BadParameter;
            }

            if (poolPtr->getBlockType() != LwSciStream::BlockType::POOL) {
                return LwSciError_BadParameter;
            }
        }
        if (poolPtr->getBlockType() != LwSciStream::BlockType::POOL) {
            return LwSciError_BadParameter;
        }

        std::shared_ptr<C2CDst> obj {};

        try {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_5_2), "Approved TID-482")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR50_CPP), "LwSciStream-ADV-CERTCPP-002")
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            obj = std::make_shared<C2CDst>(ipcEndpoint, syncModule, bufModule);
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
            return LwSciError_StreamInternalError;
        }

        // Bind pool block to C2CDst block
        LwSciError const err { obj->BindPool(poolPtr) };
        if (LwSciError_Success != err) {
            Block::removeRegisteredBlock(obj->getHandle());
            return err;
        }

        poolPtr->finalizeConfigOptions();

        *ipc = obj->getHandle();
    }

    return LwSciError_Success;
}

LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M0_1_10))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_4_4))

} // extern "C"
