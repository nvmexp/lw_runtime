/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

#include "fabricmanager.pb.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl000f.h"
#include "lwos.h"

class LocalFabricManagerControl;

class LocalFmMulticastHndlr
{
public:
    LocalFmMulticastHndlr(LocalFabricManagerControl *pLocalFmControl);
    ~LocalFmMulticastHndlr();

    void handleMessage(lwswitch::fmMessage *pFmMessage);
    static void processEvents(void *arg);
private:
    LocalFabricManagerControl *mpLfm;

    FMIntReturn_t sendGroupCreateReqMsg(uint32_t numOfGpus, uint32_t memSize);
    FMIntReturn_t sendGroupBindReqMsg(uint64_t mcHandle, std::list<char*> gpuUuidList);
    FMIntReturn_t sendGroupSetupCompleteAckMsg(uint64_t mcHandle, lwswitch::configStatus rspCode);
    FMIntReturn_t sendGroupReleaseReqMsg(uint64_t mcHandle, std::list<char*> gpuUuidList);
    FMIntReturn_t sendGroupReleaseCompleteAckMsg(uint64_t mcHandle, lwswitch::configStatus rspCode);

    void handleGroupCreateRspMsg(lwswitch::fmMessage *pFmMessage);
    void handleGroupBindRspMsg(lwswitch::fmMessage *pFmMessage);
    void handleGroupSetupCompleteReqMsg(lwswitch::fmMessage *pFmMessage);
    void handleGroupReleaseRspMsg(lwswitch::fmMessage *pFmMessage);
    void handleGroupReleaseCompleteReqMsg(lwswitch::fmMessage *pFmMessage);

    LwHandle mHandleFmClient;
    LwHandle mHandleFmSession;
    LWOSCriticalSection mLock;
};

#endif
