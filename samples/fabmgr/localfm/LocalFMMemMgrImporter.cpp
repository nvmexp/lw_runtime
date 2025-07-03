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

#include <g_lwconfig.h>
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)

#include <unistd.h>
#include <string.h>
#include <map>
#include <sstream>

#include "fm_log.h"
#include "lwRmApi.h"
#include "LocalFabricManager.h"
#include "LocalFMMemMgrImporter.h"
#include "LocalFMMemMgrExporter.h"
#include "FMAutoLock.h"
#include "FMHandleGenerator.h"
#include "FMUtils.h"

/*
 * Locks are taken at the entry point of exporter or importer this includes following functions just before the
 * first piece of code that accesses IMEX data structures:
 * Exporter:
 *  - handleUnimportRequest()
 *  - handleImportRequest()
 * Importer:
 *  - sendImportRequest()
 *  - sendUnimportRequest()
 *  - handleImportResponse()
 *  - handleUnimportResponse()
 *  - processRequestTimeout()
 *
 * Locks are released whenever exiting exporter or importer. This includes
 * - return from entry point functions listed above (Note: Autolock can only handle this case)
 * - before calls into sendFatalErrorToAllNodes(), SendMessageToPeerLFM()
 * - before importer calls exporter function or vice versa
 *
 * We have three threads of exelwtion:
 * - message processing
 * - GPU events
 * - timeouts
 *
 * In addition it is possible that simultaneously one application is doing a single node import+export and
 * another application is doing multi-node import+export.
 *
 * Also Note:
 * - message processing thread takes the libEvent lock before taking an IMEX lock
 * - GPU event thread takes an IMEX lock before taking and libEvent lock(in SendMessageToPeerLFM)
 *
 * To prevent deadlocks IMEX locks are released before sending/receiving any messages from the same context
 *
 * Future plan is to restructure the code such that locks are taken and released within the same function. This
 * can be accomplished by restructing the code and writing set/get functiions for each data structure.
 */


//This class uses RmClientHandle and FM session from LocalFMGpuMgr and assumes these will not be ilwalidated 
//during the lifetiime of this class
LocalFMMemMgrImporter::LocalFMMemMgrImporter(LocalFMCoOpMgr *pLocalCoopMgr, unsigned int imexReqTimeout, 
                                             LocalFabricManagerControl *pLocalFmControl)
    :mFMLocalCoOpMgr(pLocalCoopMgr),
     mLocalFmControl(pLocalFmControl),
     mHandleFmClient(mLocalFmControl->mFMGpuMgr->getRmClientHandle()), 
     mHandleFmSession(mLocalFmControl->mFMGpuMgr->getFmSessionHandle())
{
    FM_LOG_DEBUG( "LocalFMMemMgrImporter constructor called");
    lwosInitializeCriticalSection( &mLock );
    mExporter = NULL;
    mEnableProcessing = true;
    FMIntReturn_t retVal;
    mTimer = new FMTimer(LocalFMMemMgrImporter::requestTimeoutTimerCallback, this);
    retVal = mLocalFmControl->mFMGpuMgr->subscribeForGpuEvents(LW000F_NOTIFIERS_FABRIC_EVENT, &fabricEventCallback, (void *)this);
    if (retVal != FM_INT_ST_OK) {
        std::ostringstream ss;
        delete mTimer;
        ss << "failed to register watch requests for fabric events";
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }
    mTimer->start(1);
    mReqTimeout = imexReqTimeout;
}

LocalFMMemMgrImporter::~LocalFMMemMgrImporter()
{
    FM_LOG_DEBUG( "LocalFMMemMgrImporter destructor called");
 
    if( mEnableProcessing == true )
    {
        mLocalFmControl->mFMGpuMgr->unsubscribeGpuEvents(LW000F_NOTIFIERS_FABRIC_EVENT,
                                                         &fabricEventCallback, (void *)this);
        mTimer->stop();
        delete mTimer;
    }
    lwosDeleteCriticalSection( &mLock );
}

void
LocalFMMemMgrImporter::fabricEventCallback(void *args)
{
    FM_LOG_DEBUG("Got fabric event from RM, will read all available events");

    fmSubscriberCbArguments_t *eventArgs = (fmSubscriberCbArguments_t*) args;
    LocalFMMemMgrImporter *pMemImporter = (LocalFMMemMgrImporter *) eventArgs->subscriberCtx;

    pMemImporter->processFabricEvents();
}

void
LocalFMMemMgrImporter::processFabricEvents()
{
    // Read event data while more events remain queued in RM
    bool readMoreEvents = true;
    LwU32 retVal;
    while(readMoreEvents)
    {
        LW000F_CTRL_GET_FABRIC_EVENTS_V2_PARAMS events = {0};
        retVal = LwRmControl(mHandleFmClient, mHandleFmSession, LW000F_CTRL_CMD_GET_FABRIC_EVENTS_V2, &events, sizeof(events));
        if ( retVal != LWOS_STATUS_SUCCESS ) 
        {
            FM_LOG_ERROR("Memory import handler: Reading fabric events from GPU Driver failed with error %s", lwstatusToString(retVal));
            FM_SYSLOG_ERR("Memory import handler: Reading fabric events from GPU Driver failed with error %s", lwstatusToString(retVal));
            return ;
        }

        if( mEnableProcessing == false )
        {
            FM_LOG_DEBUG( "message processing disabled" );
            readMoreEvents = events.bMoreEvents;
            continue;
        }

        for (unsigned int i = 0; i < events.numEvents; i++)
        {
            // send import/unimport request based on event type
            switch (events.eventArray[i].type)
            {
                case LW000F_CTRL_FABRIC_EVENT_V2_TYPE_MEM_IMPORT:
                {
                    if(sendImportRequest(events.eventArray[i])) 
                        FM_LOG_DEBUG( "sendImportRequest Success");
                    else
                    {
                        FM_LOG_ERROR("sending import request to node id %d failed.",
                                     events.eventArray[i].data.import.exportNodeId);
                        FM_SYSLOG_ERR("sending import request to node id %d failed.",
                                      events.eventArray[i].data.import.exportNodeId);
                    }
                    break;
                }
                case LW000F_CTRL_FABRIC_EVENT_V2_TYPE_MEM_UNIMPORT:
                {
                    if(sendUnimportRequest(events.eventArray[i]))
                        FM_LOG_DEBUG( "sendUnimportRequest Success");
                    else
                    {
                        FM_LOG_ERROR("sending unimport request to node id %d failed.",
                                     events.eventArray[i].data.unimport.exportNodeId);
                        FM_SYSLOG_ERR("sending unimport request to node id %d failed.",
                                      events.eventArray[i].data.unimport.exportNodeId);
                    }
                    break;
                }
                default:
                    FM_LOG_ERROR("Memory import handler: Unknown event type %d received from GPU Driver.",
                                 events.eventArray[i].type);
                    FM_SYSLOG_ERR("Memory import handler: Unknown event type %d received from GPU Driver.",
                                  events.eventArray[i].type);
                    break;
            }
        }
        readMoreEvents = events.bMoreEvents;
    }
    FM_LOG_DEBUG( "Done reading all events");
    return ;
}

bool 
LocalFMMemMgrImporter::sendImportRequest(LW000F_CTRL_FABRIC_EVENT_V2 &eventData)
{
    uint32 peerNodeId = eventData.data.import.exportNodeId;
    std::string uuidStr = FMUtils::colwertUuidToHexStr( eventData.data.import.exportUuid, LW_FABRIC_UUID_LEN );

    FM_LOG_DEBUG( "peerNodeId %d event gpuId=%d UUID FAB-%s",
                  eventData.data.import.exportNodeId, eventData.data.import.gpuId, uuidStr.c_str() );


    LW00FB_ALLOCATION_PARAMETERS importAllocParams = {0};
    memcpy(importAllocParams.exportUuid, eventData.data.import.exportUuid, LW_FABRIC_UUID_LEN);
    importAllocParams.index = eventData.data.import.index;

    uint32 dupImportHandle = 0;
    lwosEnterCriticalSection(&mLock);
    if (!FMHandleGenerator::allocHandle(dupImportHandle)) {
        std::ostringstream ss;
        ss << "allocating a handle failed for imported object UUID FAB-" << uuidStr.c_str();
        ss << " error " << lwswitch::HANDLE_ALLOC_FAIL;
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        lwosLeaveCriticalSection(&mLock);
        mExporter->sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    LwU32 rmResult;
    rmResult = LwRmAlloc(mHandleFmClient, mHandleFmClient, dupImportHandle,
                         LW_MEMORY_FABRIC_IMPORTED_REF, &importAllocParams);

    if (rmResult == LW_WARN_NOTHING_TO_DO) {
        FMHandleGenerator::freeHandle(dupImportHandle);
        lwosLeaveCriticalSection(&mLock);
        return false;
    } else if (rmResult != LW_OK) {
        FMHandleGenerator::freeHandle(dupImportHandle);

        std::ostringstream ss;
        ss << " unable to dup handle for imported object UUID FAB-" << uuidStr.c_str();
        ss << " error " << lwstatusToString( rmResult );
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        lwosLeaveCriticalSection(&mLock);
        mExporter->sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    lwswitch::memoryFlaImportReq *reqMsg = new lwswitch::memoryFlaImportReq();

    //fill info into reqMsg
    reqMsg->set_exportuuid((char*)eventData.data.import.exportUuid, LW_FABRIC_UUID_LEN);
    reqMsg->set_exportgpuid(eventData.data.import.gpuId);
    reqMsg->set_index(eventData.data.import.index);
    reqMsg->set_importeventid(eventData.id);

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    pFmMessage->set_allocated_memoryflaimportreq(reqMsg);
    pFmMessage->set_type( lwswitch::FM_MEMORY_FLA_IMPORT_REQ);
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mLocalFmControl->getLocalNodeId() );

    ImportReqInfo reqInfo;
    reqInfo.dupImportHandle = dupImportHandle;
    reqInfo.reqStartTime = getMonotonicTime();
    reqInfo.nodeId = peerNodeId;

    // add entry for importPendingMap
    mImportPendingMap[eventData.id] = reqInfo;    

    // check if exportNodeId is same as localNodeId. If so
    // then it is same node import/export
    if (pFmMessage->nodeid() == peerNodeId)
    {
        lwosLeaveCriticalSection(&mLock);
        bool retVal = mExporter->handleImportRequest(pFmMessage);
        delete( pFmMessage );
        return retVal;
    }

    lwosLeaveCriticalSection(&mLock);
    FMIntReturn_t retVal;
    retVal =  mFMLocalCoOpMgr->SendMessageToPeerLFM(peerNodeId, pFmMessage, false);
    lwosEnterCriticalSection(&mLock);
    if (retVal != FM_INT_ST_OK)
    {
        // report failure to RM
        reportImportFailureToRM(dupImportHandle);
        LwRmFree(mHandleFmClient, mHandleFmSession, dupImportHandle);
        FMHandleGenerator::freeHandle(dupImportHandle);
        mImportPendingMap.erase(eventData.id);
        delete( pFmMessage );

        std::ostringstream ss;
        ss << "failed to send import request message to peer Memory Manager for UUID FAB-" << uuidStr.c_str();
        ss << " on node id" << peerNodeId << " with error " << retVal;
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        lwosLeaveCriticalSection(&mLock);
        mExporter->sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    // push current request to pending req priority queue
    mAllPendingReqs.push(addTimeoutToQueue(eventData.id));
    lwosLeaveCriticalSection(&mLock);

    delete( pFmMessage );
    return true;
}

bool 
LocalFMMemMgrImporter::sendUnimportRequest(LW000F_CTRL_FABRIC_EVENT_V2 &eventData)
{
    uint32 peerNodeId = eventData.data.unimport.exportNodeId;

    FM_LOG_DEBUG( "peerNodeId=%d unimport event id %llu import event id %llu", 
                  peerNodeId, eventData.id, eventData.data.unimport.importEventId );

    lwswitch::memoryFlaUnimportReq *reqMsg = new lwswitch::memoryFlaUnimportReq();

    //fill info into reqMsg
    reqMsg->set_importeventid(eventData.data.unimport.importEventId);
    reqMsg->set_unimporteventid(eventData.id);

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    pFmMessage->set_allocated_memoryflaunimportreq(reqMsg);
    pFmMessage->set_type( lwswitch::FM_MEMORY_FLA_UNIMPORT_REQ);
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mLocalFmControl->getLocalNodeId() );

    UnimportReqInfo reqInfo;
    reqInfo.reqStartTime = getMonotonicTime();
    reqInfo.nodeId = peerNodeId;

    lwosEnterCriticalSection(&mLock);
    // add entry for unimport pending map
    mUnimportPendingMap[eventData.id] = reqInfo;

    // check if exportNodeId is same as localNodeId. If so
    // then it is same node import/export
    if (pFmMessage->nodeid() == peerNodeId)
    {
        lwosLeaveCriticalSection(&mLock);
        bool returlwal = mExporter->handleUnimportRequest(pFmMessage);
        delete( pFmMessage );
        return returlwal;
    }

    FMIntReturn_t retVal;
    FM_LOG_DEBUG("SendMessageToPeerLFM unimport request to %d", peerNodeId);
    lwosLeaveCriticalSection(&mLock);
    retVal =  mFMLocalCoOpMgr->SendMessageToPeerLFM(peerNodeId, pFmMessage, false);
    lwosEnterCriticalSection(&mLock);
    if (retVal != FM_INT_ST_OK)
    {
        reportUnimportCompleteToRM(eventData.id);
        delete( pFmMessage );
        mUnimportPendingMap.erase(eventData.id);

        std::ostringstream ss;
        ss << "failed to send unimport request message to peer Memory Manager on node id " << peerNodeId;
        ss << " error " << retVal;
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        lwosLeaveCriticalSection(&mLock);
        mExporter->sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    // push current request to pending req priority queue
    mAllPendingReqs.push(addTimeoutToQueue(eventData.id));
    lwosLeaveCriticalSection(&mLock);
    
    delete( pFmMessage );
    return true;
}

bool
LocalFMMemMgrImporter::handleImportResponse(lwswitch::fmMessage *pFmMessage)
{
    bool ret = true;

    if (!pFmMessage->has_memoryflaimportrsp())
    {
        std::ostringstream ss;
        ss << "import response failed: memory import response not found in message from node id ";
        ss << pFmMessage->nodeid();
        // This is a code sanity check and is highly unlikely to happen
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        mExporter->sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    const lwswitch::memoryFlaImportRsp &impRsp = pFmMessage->memoryflaimportrsp();
    uint64 importEventId = impRsp.importeventid();

    lwosEnterCriticalSection(&mLock);
    if (mImportPendingMap.find(importEventId) == mImportPendingMap.end()) {
        // This is a code sanity check and is highly unlikely to happen
        std::ostringstream ss;
        ss << " import response from node id " << pFmMessage->nodeid() << " failed:";
        ss << " import event id " << importEventId << " not found";
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        lwosLeaveCriticalSection(&mLock);
        mExporter->sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    ImportReqInfo reqInfo = mImportPendingMap[importEventId];
    FM_LOG_DEBUG("received import response from node id %d for importEventId %llu ", 
                 pFmMessage->nodeid(), importEventId);

    if (impRsp.errcode() != lwswitch::MEMORY_REQ_SUCCESS) {
        FM_LOG_ERROR("received import response from node id %d with failure status %d", 
                     pFmMessage->nodeid(), impRsp.errcode());
        FM_SYSLOG_ERR("received import response from node id %d with failure status %d", 
                      pFmMessage->nodeid(), impRsp.errcode());
        reportImportFailureToRM(reqInfo.dupImportHandle);
        ret = false;
    } else {
        // success
        LwU32 pageNum = 0;
        LwU32 index = 0;
        bool done = false;
        for (pageNum = 0; pageNum < (LwU32) impRsp.pageframenumbers().size(); pageNum += LW00FB_CTRL_VALIDATE_PFN_ARRAY_SIZE) {
                LW00FB_CTRL_VALIDATE_PARAMS valParams = {};
                valParams.attrs.kind = impRsp.kind();
                valParams.attrs.pageSize = impRsp.pagesize();
                valParams.attrs.size = impRsp.size();

                valParams.memFlags = impRsp.memflags();
                // the flags parameter is set to 0 whenever there is no error
		        valParams.flags = 0;
                valParams.offset = pageNum;

                for (index = 0; (index < LW00FB_CTRL_VALIDATE_PFN_ARRAY_SIZE && 
                    (index + pageNum < (LwU32) impRsp.pageframenumbers().size())); index++) {
                    valParams.pfnArray[index] = impRsp.pageframenumbers(valParams.offset + index);
                    FM_LOG_DEBUG("Received PFN%d=%d", index, valParams.pfnArray[index]);
                }

                valParams.numPfns = index;
		        valParams.totalPfns = impRsp.pageframenumbers().size();

                uint32 retVal;
                retVal = LwRmControl(mHandleFmClient, reqInfo.dupImportHandle, 
                                     LW00FB_CTRL_CMD_VALIDATE, &valParams, sizeof(valParams));

                if (retVal != LW_OK)
                {
                    FM_LOG_ERROR("request to update page table information to GPU Driver failed"
                                 " for import event id %llu with error %s", importEventId, lwstatusToString(retVal));
                    FM_SYSLOG_ERR("request to update page table information to GPU Driver failed"
                                  " for import event id %llu with error %s", importEventId, lwstatusToString(retVal));
                    lwosLeaveCriticalSection(&mLock);
                    return false;
                }

                done = valParams.bDone;
        }

        if (((pageNum - LW00FB_CTRL_VALIDATE_PFN_ARRAY_SIZE + index ) == (LwU32) impRsp.pageframenumbers().size()) && (done == true))
        {
            FM_LOG_DEBUG("Import successful for event id %llu", importEventId);
        }
        else
        {
            FM_LOG_ERROR("import response received from node id %d for import event id %llu"
                         " has incorrect PFN information, expected %d PFNs and received %d PFNs", 
                         pFmMessage->nodeid(), importEventId, impRsp.pageframenumbers().size(),
                         (pageNum - LW00FB_CTRL_VALIDATE_PFN_ARRAY_SIZE + index));
            FM_SYSLOG_ERR("import response received from node id %d for import event id %llu"
                          " has incorrect PFN information, expected %d PFNs and received %d PFNs", 
                         pFmMessage->nodeid(), importEventId, impRsp.pageframenumbers().size(),
                         (pageNum - LW00FB_CTRL_VALIDATE_PFN_ARRAY_SIZE + index));
            reportImportFailureToRM(reqInfo.dupImportHandle);
            ret = false;
        }
    }

    // free the duplicated object
    FMHandleGenerator::freeHandle(reqInfo.dupImportHandle);
    LwRmFree(mHandleFmClient, mHandleFmSession, reqInfo.dupImportHandle);

    // delete entry from import pending map
    FM_LOG_DEBUG("deleting importEventId %llu from mImportPendingMap", importEventId);
    mImportPendingMap.erase(importEventId);
    lwosLeaveCriticalSection(&mLock);

    return ret;
}

bool
LocalFMMemMgrImporter::handleUnimportResponse(lwswitch::fmMessage *pFmMessage)
{
    LwU32 retVal;
    bool fmRet = true;

    if(!pFmMessage->has_memoryflaunimportrsp())
    {
        std::ostringstream ss;
        ss << "unimport response failed: memory unimport response not found in message from node id ";
        ss << pFmMessage->nodeid();
        // This is a code sanity check and is highly unlikely to happen
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        mExporter->sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    const lwswitch::memoryFlaUnimportRsp &unimpRsp = pFmMessage->memoryflaunimportrsp();
    uint64 unimportEventId = unimpRsp.unimporteventid();

    FM_LOG_DEBUG("received import response from node id %d for unimportEventId %llu ", 
                 pFmMessage->nodeid(), unimportEventId);

    lwosEnterCriticalSection(&mLock);
    if (mUnimportPendingMap.find(unimportEventId) == mUnimportPendingMap.end()) {
        std::ostringstream ss;
        ss << "unimport response failed: unimport event id not found in message from node id ";
        ss << pFmMessage->nodeid();
        // This is a code sanity check and is highly unlikely to happen
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        lwosLeaveCriticalSection(&mLock);
        mExporter->sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    if (unimpRsp.errcode() != lwswitch::MEMORY_REQ_SUCCESS) {
        FM_LOG_ERROR("received unimport response from requested node with failure status %d", unimpRsp.errcode());
        FM_SYSLOG_ERR("received unimport response from requested node with failure status %d", unimpRsp.errcode());
        fmRet = false;
    }

    retVal = reportUnimportCompleteToRM(unimportEventId);
    if (retVal != LW_OK) {
        FM_LOG_ERROR("unimport response failed: request to finish memory unimport failed with error %s",
                     lwstatusToString(retVal));
        FM_SYSLOG_ERR("unimport response failed: request to finish memory unimport failed with error %s",
                      lwstatusToString(retVal));
        fmRet = false;
    }

    lwosLeaveCriticalSection(&mLock);
    return fmRet;
}

LwU32
LocalFMMemMgrImporter::reportUnimportCompleteToRM(uint64 unimportEventId)
{
    UnimportReqInfo reqInfo = mUnimportPendingMap[unimportEventId];

    LW000F_CTRL_FINISH_MEM_UNIMPORT_PARAMS unimportParams = {0};
    LW000F_CTRL_FABRIC_UNIMPORT_TOKEN unImportToken = {0};
    unImportToken.unimportEventId = unimportEventId;
    unimportParams.tokenArray[0] = unImportToken;
    unimportParams.numTokens = 1;

    // delete entry from unimport pending map
    FM_LOG_DEBUG("deleting unimportEventId %llu from mUnimportPendingMap", unimportEventId);
    mUnimportPendingMap.erase(unimportEventId);

    return LwRmControl(mHandleFmClient, mHandleFmSession, LW000F_CTRL_CMD_FINISH_MEM_UNIMPORT, 
                       &unimportParams, sizeof(unimportParams));    
}

void
LocalFMMemMgrImporter::handleMessage(lwswitch::fmMessage *pFmMessage)
{
    if( mExporter == NULL )
    {
        FM_LOG_ERROR( "exporter not set yet, ignoring import/unimport response %d", pFmMessage->type() ); 
        FM_SYSLOG_ERR( "exporter not set yet, ignoring import/unimport response %d", pFmMessage->type() ); 
        return;
    }
    
    if( mEnableProcessing == false )
    {
        FM_LOG_DEBUG( "message processing disabled" );
        return;
    }

    // All messages are processed serially
    FM_LOG_DEBUG( "message type %d", pFmMessage->type());
    switch ( pFmMessage->type() )
    {
        case lwswitch::FM_MEMORY_FLA_IMPORT_RSP:
            // TODO: This might generate lot of debug logs. Figure a better way
            // for this after bring up
            FM_LOG_DEBUG("message type FM_MEMORY_FLA_IMPORT_RSP received");
            if (handleImportResponse(pFmMessage)) {
                FM_LOG_DEBUG("handleImportResponse success");
            }
            else {
                FM_LOG_DEBUG("handleImportResponse failed");
            }
            break;
            
        case lwswitch::FM_MEMORY_FLA_UNIMPORT_RSP:
            FM_LOG_DEBUG("message type FM_MEMORY_FLA_UNIMPORT_RSP received");
            if (handleUnimportResponse(pFmMessage)) {
                FM_LOG_DEBUG("handleUnimportResponse success");
            }
            else {
                FM_LOG_DEBUG("handleUnimportResponse failed");
            }
            break;
        default:
            FM_LOG_ERROR( "Unknown message type %d received by Memory Manager", pFmMessage->type());
            FM_SYSLOG_ERR( "Unknown message type %d received by Memory Manager", pFmMessage->type());
            break;
    }
}

void 
LocalFMMemMgrImporter::reportImportFailureToRM(uint32 importHandle)
{
    LW00FB_CTRL_VALIDATE_PARAMS valParams = {0};
    valParams.flags = LW00FB_CTRL_FLAGS_IMPORT_FAILED;
    uint32 retVal;
    retVal = LwRmControl(mHandleFmClient, importHandle, 
                         LW00FB_CTRL_CMD_VALIDATE, &valParams, sizeof(valParams));

    if(retVal != LW_OK)
    {
        FM_LOG_ERROR("request to update import status to GPU driver failed with error %s",
                     lwstatusToString(retVal));
        FM_SYSLOG_ERR("request to update import status to GPU driver failed with error %s",
                      lwstatusToString(retVal));
    }
}

void
LocalFMMemMgrImporter::requestTimeoutTimerCallback(void *ctx)
{
    LocalFMMemMgrImporter *pMemImporter = (LocalFMMemMgrImporter*) ctx;
    // walk through each req and age them out
    pMemImporter->processRequestTimeout();
}

void 
LocalFMMemMgrImporter::processRequestTimeout()
{
    bool timerRestartNeeded = true;
    lwosEnterCriticalSection(&mLock);
    while (!mAllPendingReqs.empty()) {
        pair<LwU64, uint64> lwrrentReq = mAllPendingReqs.top();

        // if the event id is not even in our pending import/unimport map, 
        // then this req might have been finished
        // then in that case we dont need to add it back to the priority queue
        if (mImportPendingMap.find(lwrrentReq.first) == mImportPendingMap.end() && 
            mUnimportPendingMap.find(lwrrentReq.first) == mUnimportPendingMap.end()) 
        {
            mAllPendingReqs.pop();
            continue;
        }
        
        // check whether the request has expired
        if (getMonotonicTime() >= lwrrentReq.second) 
        {
            mAllPendingReqs.pop();
            
            int nodeId = 0;
            uint64 reqStartTime = 0;
            string strRequest = "";
            if (mImportPendingMap.find(lwrrentReq.first) != mImportPendingMap.end()) {
                ImportReqInfo reqInfo = mImportPendingMap[lwrrentReq.first];
                nodeId = reqInfo.nodeId;
                reqStartTime = reqInfo.reqStartTime;
                strRequest = "import";
            } else {
                UnimportReqInfo reqInfo = mUnimportPendingMap[lwrrentReq.first];
                nodeId = reqInfo.nodeId;
                reqStartTime = reqInfo.reqStartTime;
                strRequest = "unimport";
            }

            FM_LOG_WARNING("response not received for %s event with id %llu sent to node id %d", 
                            strRequest.c_str(), lwrrentReq.first, nodeId);

            if (getMonotonicTime() > reqStartTime + 
                (mReqTimeout * REQ_TIMER_MAX_TIMEOUT_COUNT * SECONDS_TO_NANOSECONDS)) 
            {
                if (mImportPendingMap.find(lwrrentReq.first) != mImportPendingMap.end()) {
                    ImportReqInfo reqInfo = mImportPendingMap[lwrrentReq.first];
                    reportImportFailureToRM(reqInfo.dupImportHandle);
                    LwRmFree(mHandleFmClient, mHandleFmSession, reqInfo.dupImportHandle);
                    FMHandleGenerator::freeHandle(reqInfo.dupImportHandle);
                    mImportPendingMap.erase(lwrrentReq.first);

                    std::ostringstream ss;
                    ss << "failed to receive response for import event id " << lwrrentReq.first;
                    ss << " from peer Memory Manager on node id " <<  reqInfo.nodeId << " for import request";
                    FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
                    FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
                    lwosLeaveCriticalSection(&mLock);
                    mExporter->sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str());
                } else {
                    UnimportReqInfo reqInfo = mUnimportPendingMap[lwrrentReq.first];
                    reportUnimportCompleteToRM(lwrrentReq.first);
                    mUnimportPendingMap.erase(lwrrentReq.first);

                    std::ostringstream ss;
                    ss << "failed to receive response for unimport event id " << lwrrentReq.first;
                    ss << " from peer Memory Manager on node id " << reqInfo.nodeId << " for unimport request";
                    FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
                    FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
                    lwosLeaveCriticalSection(&mLock);
                    mExporter->sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str());
                }
                // Encountered fatal error, no need to restart timer
                timerRestartNeeded = false;
                goto done;
            }

            mAllPendingReqs.push(addTimeoutToQueue(lwrrentReq.first));
        } else {
            // first entry of priority queue has not expired
            // hence we do not need to walk further as subsequent
            // requests will definitely not be expired.
            lwosLeaveCriticalSection(&mLock);
            goto done;
        }
    }

    lwosLeaveCriticalSection(&mLock);
done:
    if (timerRestartNeeded)
        mTimer->restart();
}

uint64 
LocalFMMemMgrImporter::getMonotonicTime()
{
#ifdef _WINDOWS
    return GetTickCount64();
#else 
    struct timespec timerStart;
    clock_gettime(CLOCK_MONOTONIC, &timerStart);
    return (timerStart.tv_sec * SECONDS_TO_NANOSECONDS + timerStart.tv_nsec);
#endif
}

pair<LwU64, uint64> 
LocalFMMemMgrImporter::addTimeoutToQueue(int eventId)
{
    return make_pair(eventId, getMonotonicTime() + (mReqTimeout * SECONDS_TO_NANOSECONDS));
}

void
LocalFMMemMgrImporter::disableProcessing()
{
    if( mEnableProcessing == true )
    {
        mEnableProcessing = false;
        FM_LOG_ERROR( "memory importer disabling processing for memory import/unimport messages and events" );
        FM_SYSLOG_ERR( "memory importer disabling processing for memory import/unimport messages and events" );
        mLocalFmControl->mFMGpuMgr->unsubscribeGpuEvents(LW000F_NOTIFIERS_FABRIC_EVENT, &fabricEventCallback,
                                                         (void *)this);
        mTimer->stop();
        delete mTimer;
    }
}
#endif
