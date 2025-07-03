/*
 * Copyright (c) 2020-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef LWSCISYNC_IPC_PEER_H
#define LWSCISYNC_IPC_PEER_H
#include "ipc_wrapper.h"
#include "lwscisync_peer.h"
// TODO: This file is intended to eventually replace lwscisync_ipc_peer_old.h.
// Once all the tests have been ported, delete the #include.
#include "lwscisync_ipc_peer_old.h"

#define CHECK_ERR(func, expected)                                              \
    do {                                                                       \
        LwSciError ret = (func);                                               \
        if (ret != expected) {                                                 \
            std::cerr << "[] " << #func << " failed: expected " << #expected   \
                      << " returned " << ret << std::endl;                     \
            return ret;                                                        \
        }                                                                      \
    } while (0)

class LwSciSyncIpcPeer : public LwSciSyncPeer
{
public:
    LwSciSyncIpcPeer(void) = default;

    virtual void SetUp(const char* endpointName)
    {
        LwSciSyncPeer::SetUp();
        ipcEndpoint = IpcWrapper::open(endpointName);
        ASSERT_TRUE(ipcEndpoint);
    }

    void SetUp(const char* endpointName, const LwSciSyncIpcPeer& other)
    {
        LwSciSyncPeer::SetUp(other);
        ipcEndpoint = IpcWrapper::open(endpointName);
        ASSERT_TRUE(ipcEndpoint);
    }

    void TearDown(void) override
    {
        if (ipcEndpoint) {
            ipcEndpoint.reset();
        }
        LwSciSyncPeer::TearDown();
    }

    LwSciError sendBuf(std::pair<std::shared_ptr<void>, uint64_t>& descBuf)
    {
        uint64_t size = descBuf.second;
        CHECK_ERR(ipcEndpoint->send(&size, sizeof(size), timeout_10s),
                  LwSciError_Success);
        return ipcEndpoint->send((descBuf.first.get()), size, timeout_10s);
    }

    std::vector<unsigned char> recvBuf(LwSciError* error)
    {
        uint64_t bufSize = 0U;
        *error = ipcEndpoint->recvFill(&bufSize, sizeof(uint64_t), timeout_10s);
        if (*error != LwSciError_Success) {
            EXPECT_TRUE(0) << "ipcRecvFill() failed: " << *error;
            return std::vector<unsigned char>();
        }
        auto buf = std::vector<unsigned char>(bufSize);
        *error = ipcEndpoint->recvFill(buf.data(), bufSize, timeout_10s);
        if (*error != LwSciError_Success) {
            EXPECT_TRUE(0) << "ipcRecvFill() failed: " << *error;
            return std::vector<unsigned char>();
        }

        return buf;
    }

    template <typename T>
    LwSciError sendExportDesc(std::shared_ptr<T> exportDesc)
    {
        uint64_t size = sizeof(T);
        return ipcEndpoint->send(const_cast<T*>(exportDesc.get()), size,
                                 timeout_10s);
    }

    template <typename T>
    std::shared_ptr<T> recvExportDesc(LwSciError* error)
    {
        auto exportDesc = std::make_shared<T>();
        *error = ipcEndpoint->recvFill(
            reinterpret_cast<void*>(exportDesc.get()), sizeof(T), timeout_10s);
        if (*error != LwSciError_Success) {
            EXPECT_TRUE(0) << "ipcRecvFill() failed: " << *error;
        }
        return exportDesc;
    }

    LwSciError signalComplete()
    {
        bool complete = true;
        return ipcEndpoint->send(&complete, sizeof(complete), timeout_10s);
    }

    LwSciError waitComplete()
    {
        bool complete = false;
        return ipcEndpoint->recvFill(&complete, sizeof(complete), timeout_10s);
    }

    std::pair<std::shared_ptr<void>, uint64_t> exportUnreconciledList(
        const std::vector<LwSciSyncAttrList>& unreconciledLists,
        LwSciError* error)
    {
        uint64_t attrListSize = 0U;
        void* attrListDesc = nullptr;
        *error = LwSciSyncAttrListIpcExportUnreconciled(
            const_cast<LwSciSyncAttrList*>(unreconciledLists.data()),
            unreconciledLists.size(), ipcEndpoint->getEndpoint(),
            (void**)&attrListDesc, &attrListSize);
        return std::pair<std::shared_ptr<void>, uint64_t>(
            std::shared_ptr<void>(attrListDesc, LwSciSyncAttrListFreeDesc),
            attrListSize);
    }

    std::shared_ptr<LwSciSyncAttrListRec>
    importUnreconciledList(const std::vector<unsigned char>& descBuf,
                           LwSciError* error)
    {
        LwSciSyncAttrList importedList = nullptr;
        *error = LwSciSyncAttrListIpcImportUnreconciled(
            module(), ipcEndpoint->getEndpoint(), descBuf.data(),
            descBuf.size(), &importedList);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciSyncAttrListRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciSyncAttrListRec>(importedList,
                                                         LwSciSyncAttrListFree);
        }
    }

    std::pair<std::shared_ptr<void>, uint64_t>
    exportReconciledList(LwSciSyncAttrList reconciledList, LwSciError* error)
    {
        uint64_t attrListSize = 0U;
        void* attrListDesc = nullptr;
        *error = LwSciSyncAttrListIpcExportReconciled(
            reconciledList, ipcEndpoint->getEndpoint(), &attrListDesc,
            &attrListSize);
        if (*error != LwSciError_Success) {
            return std::pair<std::shared_ptr<void>, uint64_t>(nullptr, 0);
        } else {
            return std::pair<std::shared_ptr<void>, uint64_t>(
                std::shared_ptr<void>(attrListDesc, LwSciSyncAttrListFreeDesc),
                attrListSize);
        }
    }

    std::shared_ptr<LwSciSyncAttrListRec> importReconciledList(
        const std::vector<unsigned char>& descBuf,
        const std::vector<LwSciSyncAttrList>& unreconciledLists,
        LwSciError* error)
    {
        LwSciSyncAttrList importedList = nullptr;
        *error = LwSciSyncAttrListIpcImportReconciled(
            module(), ipcEndpoint->getEndpoint(), descBuf.data(),
            descBuf.size(), unreconciledLists.data(), unreconciledLists.size(),
            &importedList);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciSyncAttrListRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciSyncAttrListRec>(importedList,
                                                         LwSciSyncAttrListFree);
        }
    }

    std::shared_ptr<LwSciSyncObjIpcExportDescriptor>
    exportSyncObj(LwSciSyncObj syncObj, LwSciSyncAccessPerm permissions,
                  LwSciError* error)
    {
        auto syncObjDesc = std::make_shared<LwSciSyncObjIpcExportDescriptor>();
        *error = LwSciSyncObjIpcExport(syncObj, permissions,
                                       ipcEndpoint->getEndpoint(),
                                       syncObjDesc.get());
        return syncObjDesc;
    }

    std::shared_ptr<LwSciSyncObjRec>
    importSyncObj(LwSciSyncObjIpcExportDescriptor* syncObjDesc,
                  LwSciSyncAttrList inputAttrList,
                  LwSciSyncAccessPerm permissions, LwSciError* error)
    {
        LwSciSyncObj syncObj = nullptr;
        *error = LwSciSyncObjIpcImport(ipcEndpoint->getEndpoint(), syncObjDesc,
                                       inputAttrList, permissions, timeout_10s,
                                       &syncObj);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciSyncObjRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);
        }
    }

    std::pair<std::shared_ptr<void>, uint64_t>
    exportAttrListAndObj(LwSciSyncObj syncObj, LwSciSyncAccessPerm permissions,
                         LwSciError* error)
    {
        void* attrListAndObjDesc = nullptr;
        size_t attrListAndObjDescSize = 0U;
        *error = LwSciSyncIpcExportAttrListAndObj(
            syncObj, permissions, ipcEndpoint->getEndpoint(),
            &attrListAndObjDesc, &attrListAndObjDescSize);
        if (*error != LwSciError_Success) {
            return std::pair<std::shared_ptr<void>, uint64_t>(
                std::shared_ptr<void>(nullptr), 0);
        } else {
            return std::pair<std::shared_ptr<void>, uint64_t>(
                std::shared_ptr<void>(attrListAndObjDesc,
                                      LwSciSyncAttrListAndObjFreeDesc),
                attrListAndObjDescSize);
        }
    }

    std::shared_ptr<LwSciSyncObjRec> importAttrListAndObj(
        std::vector<unsigned char> attrListAndObjDesc,
        const std::vector<LwSciSyncAttrList>& unreconciledLists,
        LwSciSyncAccessPerm permissions, LwSciError* error)
    {
        LwSciSyncObj syncObj = nullptr;
        *error = LwSciSyncIpcImportAttrListAndObj(
            module(), ipcEndpoint->getEndpoint(), attrListAndObjDesc.data(),
            attrListAndObjDesc.size(), unreconciledLists.data(),
            unreconciledLists.size(), permissions, timeout_10s, &syncObj);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciSyncObjRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);
        }
    }

    std::shared_ptr<LwSciSyncFenceIpcExportDescriptor> exportFence(const LwSciSyncFence* fence,
                                                  LwSciError* error)
    {
        LwSciSyncFenceIpcExportDescriptor fenceDesc = {};
        *error = LwSciSyncIpcExportFence(fence, ipcEndpoint->getEndpoint(),
                                         &fenceDesc);

        return std::make_shared<LwSciSyncFenceIpcExportDescriptor>(fenceDesc);
    }

    static void LwSciSyncFenceCleanup(LwSciSyncFence* syncFence)
    {
        LwSciSyncFenceClear(syncFence);
        delete syncFence;
    }

    std::shared_ptr<LwSciSyncFence>
    importFence(LwSciSyncFenceIpcExportDescriptor* desc,
                LwSciSyncObj syncObj,
                LwSciError* error)
    {
        std::shared_ptr<LwSciSyncFence> newFence =
            std::shared_ptr<LwSciSyncFence>(new LwSciSyncFence,
                                            LwSciSyncFenceCleanup);
        *(newFence.get()) = LwSciSyncFenceInitializer;
        *error = LwSciSyncIpcImportFence(syncObj, desc, newFence.get());
        return newFence;
    }


    std::unique_ptr<IpcWrapper> ipcEndpoint;
};
#endif
