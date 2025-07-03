/*
 * Copyright (c) 2020-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef LWSCIBUF_IPC_PEER_H
#define LWSCIBUF_IPC_PEER_H

#include "ipc_wrapper.h"
#include "lwscibuf_peer.h"
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

class LwSciBufIpcPeer : public LwSciBufPeer
{
public:
    LwSciBufIpcPeer(void) = default;

    virtual void SetUp(const char* endpointName)
    {
        LwSciBufPeer::SetUp();
        ipcEndpoint = IpcWrapper::open(endpointName);
        ASSERT_TRUE(ipcEndpoint);
    }

    void SetUp(const char* endpointName, const LwSciBufIpcPeer& other)
    {
        LwSciBufPeer::SetUp(other);
        ipcEndpoint = IpcWrapper::open(endpointName);
        ASSERT_TRUE(ipcEndpoint);
    }

    void TearDown(void) override
    {
        if (ipcEndpoint) {
            ipcEndpoint.reset();
        }
        LwSciBufPeer::TearDown();
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
        const std::vector<LwSciBufAttrList>& unreconciledLists,
        LwSciError* error)
    {
        uint64_t attrListSize = 0U;
        void* attrListDesc = nullptr;
        *error = LwSciBufAttrListIpcExportUnreconciled(
            const_cast<LwSciBufAttrList*>(unreconciledLists.data()),
            unreconciledLists.size(), ipcEndpoint->getEndpoint(),
            (void**)&attrListDesc, &attrListSize);
        return std::pair<std::shared_ptr<void>, uint64_t>(
            std::shared_ptr<void>(attrListDesc, LwSciBufAttrListFreeDesc),
            attrListSize);
    }

    std::shared_ptr<LwSciBufAttrListRec>
    importUnreconciledList(const std::vector<unsigned char>& descBuf,
                           LwSciError* error)
    {
        LwSciBufAttrList importedList = nullptr;
        *error = LwSciBufAttrListIpcImportUnreconciled(
            module(), ipcEndpoint->getEndpoint(), descBuf.data(),
            descBuf.size(), &importedList);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciBufAttrListRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciBufAttrListRec>(importedList,
                                                        LwSciBufAttrListFree);
        }
    }

    std::pair<std::shared_ptr<void>, uint64_t>
    exportReconciledList(LwSciBufAttrList reconciledList, LwSciError* error)
    {
        uint64_t attrListSize = 0U;
        void* attrListDesc = nullptr;
        *error = LwSciBufAttrListIpcExportReconciled(
            reconciledList, ipcEndpoint->getEndpoint(), &attrListDesc,
            &attrListSize);
        if (*error != LwSciError_Success) {
            return std::pair<std::shared_ptr<void>, uint64_t>(nullptr, 0);
        } else {
            return std::pair<std::shared_ptr<void>, uint64_t>(
                std::shared_ptr<void>(attrListDesc, LwSciBufAttrListFreeDesc),
                attrListSize);
        }
    }

    std::shared_ptr<LwSciBufAttrListRec>
    importReconciledList(const std::vector<unsigned char>& descBuf,
                         const std::vector<LwSciBufAttrList>& unreconciledLists,
                         LwSciError* error)
    {
        LwSciBufAttrList importedList = nullptr;
        *error = LwSciBufAttrListIpcImportReconciled(
            module(), ipcEndpoint->getEndpoint(), descBuf.data(),
            descBuf.size(), unreconciledLists.data(), unreconciledLists.size(),
            &importedList);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciBufAttrListRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciBufAttrListRec>(importedList,
                                                        LwSciBufAttrListFree);
        }
    }

    std::shared_ptr<LwSciBufObjIpcExportDescriptor>
    exportBufObj(LwSciBufObj bufObj, LwSciBufAttrValAccessPerm permissions,
                 LwSciError* error)
    {
        auto bufObjDesc = std::make_shared<LwSciBufObjIpcExportDescriptor>();
        *error = LwSciBufObjIpcExport(
            bufObj, permissions, ipcEndpoint->getEndpoint(), bufObjDesc.get());
        return bufObjDesc;
    }

    std::shared_ptr<LwSciBufObjRefRec>
    importBufObj(LwSciBufObjIpcExportDescriptor* bufObjDesc,
                 LwSciBufAttrList inputAttrList,
                 LwSciBufAttrValAccessPerm permissions, LwSciError* error)
    {
        LwSciBufObj bufObj = nullptr;
        *error = LwSciBufObjIpcImport(ipcEndpoint->getEndpoint(), bufObjDesc,
                                      inputAttrList, permissions, timeout_10s,
                                      &bufObj);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciBufObjRefRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciBufObjRefRec>(bufObj, LwSciBufObjFree);
        }
    }

    std::pair<std::shared_ptr<void>, uint64_t>
    exportAttrListAndObj(LwSciBufObj bufObj,
                         LwSciBufAttrValAccessPerm permissions,
                         LwSciError* error)
    {
        void* attrListAndObjDesc = nullptr;
        size_t attrListAndObjDescSize = 0U;
        *error = LwSciBufIpcExportAttrListAndObj(
            bufObj, permissions, ipcEndpoint->getEndpoint(),
            &attrListAndObjDesc, &attrListAndObjDescSize);
        if (*error != LwSciError_Success) {
            return std::pair<std::shared_ptr<void>, uint64_t>(
                std::shared_ptr<void>(nullptr), 0);
        } else {
            return std::pair<std::shared_ptr<void>, uint64_t>(
                std::shared_ptr<void>(attrListAndObjDesc,
                                      LwSciBufAttrListAndObjFreeDesc),
                attrListAndObjDescSize);
        }
    }

    std::shared_ptr<LwSciBufObjRefRec>
    importAttrListAndObj(std::vector<unsigned char> attrListAndObjDesc,
                         const std::vector<LwSciBufAttrList>& unreconciledLists,
                         LwSciBufAttrValAccessPerm permissions,
                         LwSciError* error)
    {
        LwSciBufObj bufObj = nullptr;
        *error = LwSciBufIpcImportAttrListAndObj(
            module(), ipcEndpoint->getEndpoint(), attrListAndObjDesc.data(),
            attrListAndObjDesc.size(), unreconciledLists.data(),
            unreconciledLists.size(), permissions, timeout_10s, &bufObj);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciBufObjRefRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciBufObjRefRec>(bufObj, LwSciBufObjFree);
        }
    }

    std::unique_ptr<IpcWrapper> ipcEndpoint;
};

#endif // LWSCIBUF_IPC_PEER_H
