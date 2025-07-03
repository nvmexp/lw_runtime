/*
 * Copyright (c) 2020-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef LWSCISYNC_IPC_PEER_OLD_H
#define LWSCISYNC_IPC_PEER_OLD_H

#include "ipc_wrapper_old.h"
#include "lwscisync_peer.h"
#include <memory>
#include <vector>

constexpr int64_t timeout_10s = 10000000000;

#define CHECK_ERR(func, expected)                                              \
    do {                                                                       \
        LwSciError ret = (func);                                               \
        if (ret != expected) {                                                 \
            std::cerr << "[] " << #func << " failed: expeceted " << #expected  \
                      << " returned " << ret << std::endl;                     \
            return ret;                                                        \
        }                                                                      \
    } while (0)

class LwSciSyncIpcPeerOld : public LwSciSyncPeer
{
public:
    LwSciSyncIpcPeerOld(void) = default;

    virtual void SetUp(const char* endpointName)
    {
        LwSciSyncPeer::SetUp();
        ASSERT_EQ(LwSciError_Success, ipcInit(endpointName, &ipcWrapper));
    }

    void SetUp(const char* endpointName, const LwSciSyncIpcPeerOld& other)
    {
        LwSciSyncPeer::SetUp(other);
        ASSERT_EQ(LwSciError_Success, ipcInit(endpointName, &ipcWrapper));
    }

    void TearDown(void) override
    {
        if (ipcWrapper != nullptr) {
            ipcDeinit(ipcWrapper);
        }
        LwSciSyncPeer::TearDown();
    }

    LwSciError sendBuf(std::pair<std::shared_ptr<void>, uint64_t>& descBuf)
    {
        uint64_t size = descBuf.second;
        CHECK_ERR(ipcSendTimeout(ipcWrapper, &size, sizeof(size), timeout_10s),
                  LwSciError_Success);
        return ipcSendTimeout(ipcWrapper, (descBuf.first.get()), size, timeout_10s);
    }

    std::vector<unsigned char> recvBuf(LwSciError* error)
    {
        uint64_t bufSize = 0U;
        *error = ipcRecvFillTimeout(ipcWrapper, &bufSize, sizeof(uint64_t),
                                    timeout_10s);
        if (*error != LwSciError_Success) {
            EXPECT_TRUE(0) << "ipcRecvFill() failed: " << *error;
            return std::vector<unsigned char>();
        }
        auto buf = std::vector<unsigned char>(bufSize);
        *error =
            ipcRecvFillTimeout(ipcWrapper, buf.data(), bufSize, timeout_10s);
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
        return ipcSendTimeout(ipcWrapper, const_cast<T*>(exportDesc.get()),
                              size, timeout_10s);
    }

    template <typename T>
    std::shared_ptr<T> recvExportDesc(LwSciError* error)
    {
        auto exportDesc = std::make_shared<T>();
        *error = ipcRecvFillTimeout(ipcWrapper,
                                    reinterpret_cast<void*>(exportDesc.get()),
                                    sizeof(T), timeout_10s);
        if (*error != LwSciError_Success) {
            EXPECT_TRUE(0) << "ipcRecvFill() failed: " << *error;
        }
        return exportDesc;
    }

    LwSciError signalComplete()
    {
        bool complete = true;
        return ipcSendTimeout(ipcWrapper, &complete, sizeof(complete),
                              timeout_10s);
    }

    LwSciError waitComplete()
    {
        bool complete = false;
        return ipcRecvFillTimeout(ipcWrapper, &complete, sizeof(complete),
                                  timeout_10s);
    }

    std::pair<std::shared_ptr<void>, uint64_t> exportUnreconciledList(
        const std::vector<LwSciSyncAttrList>& unreconciledLists,
        LwSciError* error)
    {
        uint64_t attrListSize = 0U;
        void* attrListDesc = nullptr;
        *error = LwSciSyncAttrListIpcExportUnreconciled(
            const_cast<LwSciSyncAttrList*>(unreconciledLists.data()),
            unreconciledLists.size(), ipcWrapperGetEndpoint(ipcWrapper),
            (void**)&attrListDesc, &attrListSize);
        return
            std::pair<std::shared_ptr<void>, uint64_t>(std::shared_ptr<void>(attrListDesc, LwSciSyncAttrListFreeDesc), attrListSize);
    }

    std::shared_ptr<LwSciSyncAttrListRec>
    importUnreconciledList(const std::vector<unsigned char>& descBuf,
                           LwSciError* error)
    {
        LwSciSyncAttrList importedList = nullptr;
        *error = LwSciSyncAttrListIpcImportUnreconciled(
            module(), ipcWrapperGetEndpoint(ipcWrapper), descBuf.data(),
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
            reconciledList, ipcWrapperGetEndpoint(ipcWrapper), &attrListDesc,
            &attrListSize);
        if (*error != LwSciError_Success) {
            return std::pair<std::shared_ptr<void>, uint64_t>(nullptr, 0);
        } else {
            return std::pair<std::shared_ptr<void>, uint64_t>(std::shared_ptr<void>(attrListDesc, LwSciSyncAttrListFreeDesc), attrListSize);
        }
    }

    std::shared_ptr<LwSciSyncAttrListRec> importReconciledList(
        const std::vector<unsigned char>& descBuf,
        const std::vector<LwSciSyncAttrList>& unreconciledLists,
        LwSciError* error)
    {
        LwSciSyncAttrList importedList = nullptr;
        *error = LwSciSyncAttrListIpcImportReconciled(
            module(), ipcWrapperGetEndpoint(ipcWrapper), descBuf.data(),
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
                                       ipcWrapperGetEndpoint(ipcWrapper),
                                       syncObjDesc.get());
        return syncObjDesc;
    }

    std::shared_ptr<LwSciSyncObjRec>
    importSyncObj(LwSciSyncObjIpcExportDescriptor* syncObjDesc,
                  LwSciSyncAttrList inputAttrList,
                  LwSciSyncAccessPerm permissions, LwSciError* error)
    {
        LwSciSyncObj syncObj = nullptr;
        *error = LwSciSyncObjIpcImport(ipcWrapperGetEndpoint(ipcWrapper),
                                       syncObjDesc, inputAttrList, permissions,
                                       timeout_10s, &syncObj);
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
            syncObj, permissions, ipcWrapperGetEndpoint(ipcWrapper),
            &attrListAndObjDesc, &attrListAndObjDescSize);
        if (*error != LwSciError_Success) {
            return std::pair<std::shared_ptr<void>, uint64_t>(std::shared_ptr<void>(nullptr), 0);
        } else {
            return std::pair<std::shared_ptr<void>, uint64_t>(std::shared_ptr<void>(attrListAndObjDesc, LwSciSyncAttrListAndObjFreeDesc), attrListAndObjDescSize);
        }
    }

    std::shared_ptr<LwSciSyncObjRec> importAttrListAndObj(
        std::vector<unsigned char> attrListAndObjDesc,
        const std::vector<LwSciSyncAttrList>& unreconciledLists,
        LwSciSyncAccessPerm permissions, LwSciError* error)
    {
        LwSciSyncObj syncObj = nullptr;
        *error = LwSciSyncIpcImportAttrListAndObj(
            module(), ipcWrapperGetEndpoint(ipcWrapper),
            attrListAndObjDesc.data(), attrListAndObjDesc.size(),
            unreconciledLists.data(), unreconciledLists.size(), permissions,
            timeout_10s, &syncObj);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciSyncObjRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);
        }
    }

    std::shared_ptr<LwSciSyncFenceIpcExportDescriptor>
    exportFence(LwSciSyncFence* syncFence, LwSciError* error)
    {
        auto fenceDesc = std::make_shared<LwSciSyncFenceIpcExportDescriptor>();
        *error = LwSciSyncIpcExportFence(
            syncFence, ipcWrapperGetEndpoint(ipcWrapper), fenceDesc.get());
        return fenceDesc;
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

    IpcWrapperOld ipcWrapper = nullptr;
};

template <int64_t JamaID>
class LwSciSyncTransportTest : public LwSciSyncBaseTest<JamaID>
{
protected:
    void TearDown() override
    {
        peer.TearDown();
        otherPeer.TearDown();
        deinitIpc();

        if (pid == 0) {
            // WAR: exit here so we do not have duplicate output
            // https://github.com/google/googletest/issues/1153
            exit(testing::Test::HasFailure());
        }
        LwSciSyncBaseTest<JamaID>::TearDown();
    };

    // LwSciIpcInit() is required to be called after fork() to get
    // the unique endpoint handle. This can be fixed in 5.2.
    // It's required to call initIpc in each process after fork() now.
    // We can move initIpc in SetUp after 5.2.
    void initIpc(void)
    {
        ASSERT_EQ(LwSciError_Success, LwSciIpcInit());
        ipcinit = true;
    }

    void deinitIpc(void)
    {
        if (ipcinit) {
            LwSciIpcDeinit();
            ipcinit = false;
        }
    }

    int wait_for_child_fork(int pid)
    {
        int status;
        if (0 > waitpid(pid, &status, 0)) {
            TEST_COUT << " Waitpid error!";
            return (-1);
        }
        if (WIFEXITED(status)) {
            const int exit_status = WEXITSTATUS(status);
            if (exit_status != 0) {
                TEST_COUT << "Non-zero exit status " << exit_status
                          << " from test!";
            }
            return exit_status;
        } else {
            TEST_COUT << " Non-normal exit from child!";
            return (-2);
        }
    };

    bool ipcinit = false;
    LwSciSyncIpcPeerOld peer;
    LwSciSyncIpcPeerOld otherPeer;
    pid_t pid = 0;
};

#endif
