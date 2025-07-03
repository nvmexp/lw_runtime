/*
 *  Copyright 2021-2022 LWPU Corporation.  All rights reserved.
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
#include "lwos.h"
#include "FMTimer.h"
#include "class/cl000f.h"
#include "ctrl/ctrl000f.h"
#include "ctrl/ctrl000f_imex.h"
#include "ctrl/ctrl00fb.h"
#include "class/cl00fb.h"
#include "memmgr.pb.h"
#include "FMHandleGenerator.h"
#include <queue>
#include <time.h>

using namespace std;

class LocalFabricManagerControl;
class LocalFMCoOpMgr;
class LocalFMMemMgrExporter;

#define REQ_TIMER_MAX_TIMEOUT_COUNT 10
#define SECONDS_TO_NANOSECONDS 1000000000L

class LocalFMMemMgrImporter
{
public:
	LocalFMMemMgrImporter(LocalFMCoOpMgr *pLocalCoopMgr, unsigned int imexReqTimeout, 
						  LocalFabricManagerControl *pLocalFmControl);
    ~LocalFMMemMgrImporter();

    void setExporter(LocalFMMemMgrExporter *exporter) {
    	mExporter = exporter;
    };
	
	static void fabricEventCallback(void *arg);
	void handleMessage(lwswitch::fmMessage *pFmMessage);

	bool handleImportResponse(lwswitch::fmMessage *pFmMessage);
    bool handleUnimportResponse(lwswitch::fmMessage *pFmMessage);
    void disableProcessing();


private:

	typedef struct ImportReqInfo
	{
		uint32 dupImportHandle;
        uint64 reqStartTime;
		uint32 nodeId;
	} ImportReqInfo;

	typedef struct UnimportReqInfo
	{
        uint64 reqStartTime;
		uint32 nodeId;

	} UnimportReqInfo;

    bool sendImportRequest(LW000F_CTRL_FABRIC_EVENT_V2 &eventData);
    bool sendUnimportRequest(LW000F_CTRL_FABRIC_EVENT_V2 &eventData);

    void processFabricEvents();
    void reportImportFailureToRM(uint32 importHandle);
    LwU32 reportUnimportCompleteToRM(uint64 unimportEventId);

    static void requestTimeoutTimerCallback(void *ctx);
    void processRequestTimeout();
    uint64 getMonotonicTime();
    pair<LwU64, uint64> addTimeoutToQueue(int eventId);

    LocalFMCoOpMgr *mFMLocalCoOpMgr;
    LocalFabricManagerControl *mLocalFmControl;

    LwHandle mHandleFmClient;
    LwHandle mHandleFmSession;
    LWOSCriticalSection mLock;
    FMTimer *mTimer;
    unsigned int mReqTimeout;

    std::map<LwU64, ImportReqInfo> mImportPendingMap;
    std::map<LwU64, UnimportReqInfo> mUnimportPendingMap;

	bool mEnableProcessing;
    struct compareTimestamp {
    	bool operator()(std::pair<LwU64, uint64> const &a, std::pair<LwU64, uint64> const &b) {
    		return (a.second > b.second);
    	}
    };

    priority_queue<pair<LwU64, uint64>, vector<pair<LwU64, uint64>>, compareTimestamp> mAllPendingReqs;

    LocalFMMemMgrExporter *mExporter;
};
#endif
