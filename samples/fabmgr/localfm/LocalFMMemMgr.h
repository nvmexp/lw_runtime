/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once

#include <g_lwconfig.h>
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)

#include "fabricmanager.pb.h"
#include "memmgr.pb.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl000f.h"
#include "ctrl/ctrl000f_imex.h"
#include "lwos.h"


class LocalFabricManagerControl;
class LocalFMCoOpMgr;

class LocalFMMemMgr
{
public:
    LocalFMMemMgr(LocalFMCoOpMgr *pLocalCoopMgr, LocalFabricManagerControl *pLocalFmControl);
    ~LocalFMMemMgr();

    void handleMessage(lwswitch::fmMessage *pFmMessage);
    static void processEvents(void *arg);
private:
    bool sendImportRequest(LW000F_CTRL_FABRIC_EVENT &eventData);
    bool sendImportResponse(const lwswitch::memoryImportReq &reqMsg, uint32 peerNodeId, uint32 errCode);
    bool sendImportError(const lwswitch::memoryImportRsp &rspMsg, uint32 peerNodeId, uint32 errCode);
    bool sendUnimportRequest(LW000F_CTRL_FABRIC_EVENT &eventData);
    bool sendUnimportResponse(const lwswitch::memoryUnimportReq &reqMsg, uint32 peerNodeId, uint32 errCode);

    bool handleImportRequest(lwswitch::fmMessage *pFmMessage);
    bool handleImportResponse(lwswitch::fmMessage *pFmMessage);
    bool handleImportError(lwswitch::fmMessage *pFmMessage);
    bool readPageTableEntries(lwswitch::memoryImportRsp *rspMsg, LwHandle objectHandle);
    bool handleUnimportRequest(lwswitch::fmMessage *pFmMessage);
    bool handleUnimportResponse(lwswitch::fmMessage *pFmMessage);

    LocalFMCoOpMgr *mFMLocalCoOpMgr;
    LocalFabricManagerControl *mLocalFabricManagerControl;

    LwHandle mHandleFmClient;
    LwHandle mHandleFmSession;
    LWOSCriticalSection mLock;
};
#endif
