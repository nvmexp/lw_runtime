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

#include <g_lwconfig.h>
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)

#include <poll.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#include <map>
#include <list>

#include <g_lwconfig.h>

#include "fm_log.h"
#include "lwtypes.h"
#include "lwRmApi.h"
#include "lwos.h"
#include "lwos.h"
#include "FMCommonTypes.h"
#include "LocalFMMemMgr.h"
#include "memmgr.pb.h"
#include "LocalFabricManager.h"
#include "FmThread.h"
#include "FMDevInfoMsgHndlr.h"
#include "FMErrorCodesInternal.h"

#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl0000.h"
#include "class/cl0000.h"
#include "class/cl000f.h"
#include "class/cl0005.h"
#include "ctrl/ctrl000f.h"
#include "ctrl/ctrl00f4.h"
#include "ctrl/ctrl00f5.h"

#define MEM_DUP_FAILED                      1
#define MEM_DESCRIBE_FAILED                 2

#define MEM_VALIDATE_FAILED                 3
#define MEM_EXPORT_DUP_MISSING              4
#define MEM_OBJECT_FREE_FAILED              5

struct dupExportImportTuple {
    LwHandle handleToClient;
    LwHandle handleToExportObject;
};

//In order to use dupExportImportTuple as key in a map the "<" operator needs to be defined
//The tuple class is provided in C++11 when we move to it we can remove this
static inline bool operator < (const dupExportImportTuple& l, const dupExportImportTuple& r)
{
    if (l.handleToClient < r.handleToClient)
        return true;
    else if((l.handleToClient == r.handleToClient) && (l.handleToExportObject < r.handleToExportObject))
        return true;
    else
        return false;
}

typedef std::list<LwHandle> tExportDupList;
typedef std::map<dupExportImportTuple, tExportDupList> tExportMap;

static tExportMap exportDupMap;  //map an (exportClient, exportObject) tuple to a list of exported objects


//This class used RmClientHandle and FM session from LocalFMGpuMgr and assumes these will not be ilwalidated 
//during the lifetiime of this class
LocalFMMemMgr::LocalFMMemMgr(LocalFMCoOpMgr *pLocalCoopMgr, LocalFabricManagerControl *pLocalFmControl)
    :mFMLocalCoOpMgr(pLocalCoopMgr),
     mLocalFabricManagerControl(pLocalFmControl), 
     mHandleFmClient(mLocalFabricManagerControl->mFMGpuMgr->getRmClientHandle()), 
     mHandleFmSession(mLocalFabricManagerControl->mFMGpuMgr->getFmSessionHandle())
{
    lwosInitializeCriticalSection( &mLock );
    FM_LOG_DEBUG( "LocalFMMemMgr constructor called");
    mLocalFabricManagerControl->mFMGpuMgr->subscribeForGpuEvents(LW000F_NOTIFIERS_FABRIC_EVENT, &processEvents, (void *)this);
}

LocalFMMemMgr::~LocalFMMemMgr()
{
    mLocalFabricManagerControl->mFMGpuMgr->unsubscribeGpuEvents(LW000F_NOTIFIERS_FABRIC_EVENT, &processEvents, (void *)this);
    lwosDeleteCriticalSection( &mLock );
    FM_LOG_DEBUG( "LocalFMMemMgr destructor called");
}

void
LocalFMMemMgr::processEvents(void *args)
{
    FM_LOG_DEBUG("Got Mem event from RM, will read all available events");
    LwU32 retVal;
    LwU32 more_events = 1;

    fmSubscriberCbArguments_t *eventArgs = (fmSubscriberCbArguments_t*) args;
    LocalFMMemMgr *pMemMgr = (LocalFMMemMgr *) eventArgs->subscriberCtx;

    //Read event data while more  events remain queued in RM
    bool read_more_events = true;
    LW000F_CTRL_GET_FABRIC_EVENTS_PARAMS *pEvents = new LW000F_CTRL_GET_FABRIC_EVENTS_PARAMS;
    while(read_more_events)
    {
        memset(pEvents, 0, sizeof(LW000F_CTRL_GET_FABRIC_EVENTS_PARAMS));
        if(( retVal = LwRmControl(pMemMgr->mHandleFmClient, 
                                  pMemMgr->mHandleFmSession, 
                                  LW000F_CTRL_CMD_GET_FABRIC_EVENTS, 
                                  pEvents, 
                                  sizeof(LW000F_CTRL_GET_FABRIC_EVENTS_PARAMS)
                                 )
           ) == LWOS_STATUS_SUCCESS ) 
        {
            for (unsigned int i = 0; i < pEvents->numEvents; i++)
            {
                //send import/unimport request based on event type
                switch (pEvents->eventArray[i].eventType)
                {
                    case LW000F_CTRL_FABRIC_EVENT_TYPE_IMPORT:
                    {
                        if(pMemMgr->sendImportRequest(pEvents->eventArray[i])) 
                            FM_LOG_DEBUG( "sendImportRequest Success");
                        else
                            FM_LOG_ERROR( "Sending import request Failed");
                        break;
                    }
                    case LW000F_CTRL_FABRIC_EVENT_TYPE_RELEASE:
                    {
                        if(pMemMgr->sendUnimportRequest(pEvents->eventArray[i]))
                            FM_LOG_DEBUG( "sendUnimportRequest Success");
                        else
                            FM_LOG_ERROR( "Sending unimport request Failed");
                        break;
                    }
                    default:
                        FM_LOG_ERROR( "Unknown event type %d received from RM", pEvents->eventArray[i].eventType);
                        break;
                }
            }
            read_more_events = pEvents->bMoreEvents;
        }
        else
        {
            FM_LOG_ERROR( "Unable to read fabric events from RM %d", retVal);
            delete pEvents;
            return ;
        }
    }
    delete pEvents;
    FM_LOG_DEBUG( "Done reading all events");
    return ;
}

bool LocalFMMemMgr::sendImportRequest(LW000F_CTRL_FABRIC_EVENT &evData)
{
    uint32 peerNodeId = evData.nodeId;

    FM_LOG_DEBUG( "event peerNodeId=%d event gpuId=%d", evData.nodeId, evData.gpuId);

    FM_LOG_DEBUG( "impClient=%d impObj=%d expClient=%d, expObj=%d", 
                evData.hImportClient, evData.hImportObject, evData.hExportClient, evData.hExportObject);
    lwswitch::memoryImportReq *reqMsg = new lwswitch::memoryImportReq();

    //dup object
    LwHandle importObjectDup = 0;
    LwU32 rv =  LwRmDupObject2(mHandleFmClient, mHandleFmClient, &importObjectDup, evData.hImportClient, evData.hImportObject, 0);
    if(rv != LW_OK)
    {
        delete reqMsg;
        FM_LOG_ERROR( "Import object dup error %d", rv);
        return false;
    }
    else
    {
        FM_LOG_DEBUG( "Import object dup successful. lwdaImportClient=%d lwdaImportObject=%d", evData.hImportClient, evData.hImportObject);
    }

    reqMsg->set_kthandletoimportdup(importObjectDup);

    //fill info into reqMsg
    reqMsg->set_handletoexportclient(evData.hExportClient);
    reqMsg->set_handletoexportobject(evData.hExportObject);
    reqMsg->set_memorysize(evData.size);
    reqMsg->set_gpuid(evData.gpuId);

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    pFmMessage->set_allocated_memoryimportreq(reqMsg);
    pFmMessage->set_type( lwswitch::FM_MEMORY_IMPORT_REQ);
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mLocalFabricManagerControl->getLocalNodeId() );

    if (pFmMessage->nodeid() == peerNodeId)
    {
        bool rv = handleImportRequest(pFmMessage);
        delete( pFmMessage );
        return rv;
    }
    FMIntReturn_t retVal;
    retVal =  mFMLocalCoOpMgr->SendMessageToPeerLFM(peerNodeId, pFmMessage, false);
    if (retVal != FM_INT_ST_OK)
    {
        FM_LOG_ERROR( "Send import request message to peer LFM failed error=%d", retVal);
        LwRmFree(mHandleFmClient, mHandleFmClient, importObjectDup);
        delete( pFmMessage );
        return false;
    }
    
    delete( pFmMessage );
    return true;
}

bool
LocalFMMemMgr::readPageTableEntries(lwswitch::memoryImportRsp *rspMsg, LwHandle objectHandle)
{    
    LwU32 retVal;
    lwswitch::ktPageTableEntries *pPageTableEntries = new lwswitch::ktPageTableEntries();
    pPageTableEntries->set_offset(0);
    //loop read and fill in PTEs into rspMsg 
    LW00F4_CTRL_EXPORT_DESCRIBE_PARAMS params;
    params.offset = 0;
    while((retVal = LwRmControl(mHandleFmClient, objectHandle, LW00F4_CTRL_CMD_EXPORT_DESCRIBE, &params, sizeof(params))) == LW_OK)
    {
        FM_LOG_DEBUG("params.offset=%llu params.numPfns=%u params.totalPfns=%llu",params.offset, params.numPfns, params.totalPfns);
        for(unsigned int i = 0; i < params.numPfns; i++) 
        {
            pPageTableEntries->add_pageframenumberarray(params.pfnArray[i]);
            //FM_LOG_DEBUG("PFN%llu=%d",i  + params.offset, params.pfnArray[i]);
        }
        if((params.offset + params.numPfns) < params.totalPfns)
        {
            params.offset += params.numPfns;
            continue;
        }
        else
        {
            break;
        }
    }

    if(retVal != LW_OK)
    {
        FM_LOG_ERROR( "Memory export describe error %d", retVal);
        delete pPageTableEntries;
        return false;
    }

    rspMsg->set_allocated_ktptentries(pPageTableEntries);
    return true;
}

bool LocalFMMemMgr::sendImportResponse(const lwswitch::memoryImportReq &reqMsg, uint32 peerNodeId, uint32 errCode)
{
    FMIntReturn_t retVal;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::memoryImportRsp *rspMsg = new lwswitch::memoryImportRsp();

    rspMsg->set_handletoexportclient(reqMsg.handletoexportclient());
    rspMsg->set_handletoexportobject(reqMsg.handletoexportobject());
    rspMsg->set_kthandletoimportdup(reqMsg.kthandletoimportdup());
    rspMsg->set_gpuid(reqMsg.gpuid());

    dupExportImportTuple exportTuple = {reqMsg.handletoexportclient(), reqMsg.handletoexportobject()};
    lwosEnterCriticalSection(&mLock);
    if((errCode == 0) && (readPageTableEntries(rspMsg, exportDupMap[exportTuple].front()) == false))
    {
        FM_LOG_ERROR( "Unable to read page table entries");
        errCode = MEM_DESCRIBE_FAILED;
    }
    lwosLeaveCriticalSection(&mLock);
    rspMsg->set_errcode(errCode);
    
    pFmMessage->set_allocated_memoryimportrsp(rspMsg);
    pFmMessage->set_type( lwswitch::FM_MEMORY_IMPORT_RSP);
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mLocalFabricManagerControl->getLocalNodeId() );

    if (pFmMessage->nodeid() == peerNodeId)
    {
        bool rv = handleImportResponse(pFmMessage);
        delete( pFmMessage );
        return rv;
    }

    retVal =  mFMLocalCoOpMgr->SendMessageToPeerLFM(peerNodeId, pFmMessage, false);
    if (retVal != FM_INT_ST_OK)
    {
        FM_LOG_ERROR( "Send import response message to peer LFM failed error=%d", retVal);
        delete( pFmMessage );
        return false;
    }
    FM_LOG_DEBUG( "Sent Import Response to node %d", peerNodeId);
    
    delete( pFmMessage );
    return true;
}

bool LocalFMMemMgr::sendImportError(const lwswitch::memoryImportRsp &rspMsg, uint32 peerNodeId, uint32 errCode)
{
    FMIntReturn_t retVal;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::ktMemoryImportErr *errMsg = new lwswitch::ktMemoryImportErr();

    errMsg->set_handletoexportclient(rspMsg.handletoexportclient());
    errMsg->set_handletoexportobject(rspMsg.handletoexportobject());
    errMsg->set_gpuid(rspMsg.gpuid());
    errMsg->set_errcode(errCode);

    pFmMessage->set_allocated_ktmemoryimporterr(errMsg);
    pFmMessage->set_type( lwswitch::FM_KT_MEMORY_IMPORT_ERR);
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mLocalFabricManagerControl->getLocalNodeId() );

    if (pFmMessage->nodeid() == peerNodeId)
    {
        bool rv = handleImportError(pFmMessage);
        delete( pFmMessage );
        return rv;
    }

    retVal =  mFMLocalCoOpMgr->SendMessageToPeerLFM(peerNodeId, pFmMessage, false);
    if (retVal != FM_INT_ST_OK)
    {
        FM_LOG_ERROR( "Send import error message to peer LFM failed error=%d", retVal);
        delete( pFmMessage );
        return false;
    }
    FM_LOG_DEBUG( "Sent Import error %d to node %d ", errCode, peerNodeId);
    delete( pFmMessage );
    return true;
}

bool LocalFMMemMgr::sendUnimportRequest(LW000F_CTRL_FABRIC_EVENT &evData)
{
    FMIntReturn_t retVal;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::memoryUnimportReq *reqMsg = new lwswitch::memoryUnimportReq();

    uint32 peerNodeId = evData.nodeId;

    reqMsg->set_handletoexportclient(evData.hExportClient);
    reqMsg->set_handletoexportobject(evData.hExportObject);
    reqMsg->set_gpuid(evData.gpuId);

    pFmMessage->set_allocated_memoryunimportreq(reqMsg);
    pFmMessage->set_type( lwswitch::FM_MEMORY_UNIMPORT_REQ);
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mLocalFabricManagerControl->getLocalNodeId() );

    if (pFmMessage->nodeid() == peerNodeId)
    {
        bool rv = handleUnimportRequest(pFmMessage);
        delete( pFmMessage );
        return rv;
    }
    //TODO: enable tracking??
    retVal = mFMLocalCoOpMgr->SendMessageToPeerLFM(peerNodeId, pFmMessage, false);
    if (retVal != FM_INT_ST_OK)
    {
        FM_LOG_ERROR( "Send unimport request message to peer LFM failed error=%d", retVal);
        delete( pFmMessage );
        return false;
    }
    
    delete( pFmMessage );
    return true;
}

bool LocalFMMemMgr::sendUnimportResponse(const lwswitch::memoryUnimportReq &reqMsg, uint32 peerNodeId, uint32 errCode)
{
    FMIntReturn_t retVal;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::memoryUnimportRsp *rspMsg = new lwswitch::memoryUnimportRsp();

    rspMsg->set_handletoexportclient(reqMsg.handletoexportclient());
    rspMsg->set_handletoexportobject(reqMsg.handletoexportobject());
    rspMsg->set_errcode(errCode);

    pFmMessage->set_allocated_memoryunimportrsp(rspMsg);
    pFmMessage->set_type( lwswitch::FM_MEMORY_UNIMPORT_RSP);
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mLocalFabricManagerControl->getLocalNodeId() );

    if (pFmMessage->nodeid() == peerNodeId)
    {
        bool rv = handleUnimportResponse(pFmMessage);
        delete( pFmMessage );
        return rv;
    }

    retVal =  mFMLocalCoOpMgr->SendMessageToPeerLFM(peerNodeId, pFmMessage, false);
    if (retVal != FM_INT_ST_OK)
    {
        FM_LOG_ERROR( "Send unimport response message to peer LFM failed error=%d", retVal);
        delete( pFmMessage );
        return false;
    }
    FM_LOG_DEBUG( "Sent Unimport Response retVal=%d", retVal);
    
    delete( pFmMessage );
    return true;
}

bool LocalFMMemMgr::handleImportRequest(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "handleImportRequest called");
    if(!pFmMessage->has_memoryimportreq())
    {
        FM_LOG_ERROR( "No memory import request in message");
        return false;
    }
    const lwswitch::memoryImportReq &impReq = pFmMessage->memoryimportreq();
    dupExportImportTuple exportTuple = {impReq.handletoexportclient(), impReq.handletoexportobject()};
    //dup object
    LwHandle exportObjectDup = 0;
    LwU32 retVal = LwRmDupObject2(mHandleFmClient, mLocalFabricManagerControl->mFMGpuMgr->getMemSubDeviceHandleForGpuId(impReq.gpuid()), 
                                    &exportObjectDup, impReq.handletoexportclient(), impReq.handletoexportobject(), 0);
    if(retVal != LW_OK)
    {
        FM_LOG_ERROR( "Export object dup error %d importClient=%d importObject=%d", retVal, impReq.handletoexportclient(), impReq.handletoexportobject());
        return sendImportResponse(impReq, pFmMessage->nodeid(), MEM_DUP_FAILED);
    }

    lwosEnterCriticalSection(&mLock);
    exportDupMap[exportTuple].push_front(exportObjectDup);
    FM_LOG_DEBUG("Added tuple(%x, %x) map.size()=%lu",impReq.handletoexportclient(), impReq.handletoexportobject(), exportDupMap.size());
    lwosLeaveCriticalSection(&mLock);
    return sendImportResponse(impReq, pFmMessage->nodeid(), 0);
}

bool LocalFMMemMgr::handleImportResponse(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "handleImportResponse");
    if(!pFmMessage->has_memoryimportrsp())
    {
        FM_LOG_ERROR( "No memory import response in message");
        return false;
    }
    const lwswitch::memoryImportRsp &impRsp = pFmMessage->memoryimportrsp();

    //set all Page Table Entries to RM. If error sendImportError
    LwU32 pageFrameNumber;
    //loop  set PTEs to RM until RM says done or we runout of PTEs in impRsp
    LwU32 batchIndex = 0;
    for (pageFrameNumber = 0; 
         pageFrameNumber < (LwU32) impRsp.ktptentries().pageframenumberarray().size(); 
         pageFrameNumber += LW00F4_CTRL_EXPORT_DESCRIBE_PFN_ARRAY_SIZE)
    {
        LW00F5_CTRL_IMPORT_VALIDATE_PARAMS params;
        params.offset = pageFrameNumber;
        FM_LOG_DEBUG( "#PTE = %d ",  impRsp.ktptentries().pageframenumberarray().size());
        for( batchIndex = 0; 
             batchIndex < LW00F5_CTRL_IMPORT_VALIDATE_PFN_ARRAY_SIZE
             && (batchIndex + pageFrameNumber) < (LwU32) impRsp.ktptentries().pageframenumberarray().size(); 
             batchIndex++ )
        {
            params.pfnArray[batchIndex] = impRsp.ktptentries().pageframenumberarray(params.offset + batchIndex);
            //FM_LOG_DEBUG( "PFN%llu=%d ",  batchIndex + params.offset , params.pfnArray[batchIndex]);
        }
        params.numPfns = batchIndex;
        uint32 retVal;
        retVal = LwRmControl(mHandleFmClient, impRsp.kthandletoimportdup()
                      , LW00F5_CTRL_CMD_IMPORT_VALIDATE, &params, sizeof(params));

        if(retVal != LW_OK)
        {
            FM_LOG_ERROR( "Validate failed retVal=%d", retVal);
            sendImportError(impRsp, pFmMessage->nodeid(), MEM_VALIDATE_FAILED);
            break;
        }
    }

    if ((pageFrameNumber - LW00F4_CTRL_EXPORT_DESCRIBE_PFN_ARRAY_SIZE + batchIndex )== (LwU32) impRsp.ktptentries().pageframenumberarray().size())
    {
        FM_LOG_DEBUG( "Import successful");
    }
    else
    {
        FM_LOG_ERROR( "Import Failed PFN=%d expected=%d", 
                    (pageFrameNumber - LW00F4_CTRL_EXPORT_DESCRIBE_PFN_ARRAY_SIZE + batchIndex ),
                    impRsp.ktptentries().pageframenumberarray().size());
    }

    LwRmFree(mHandleFmClient, mHandleFmClient, impRsp.kthandletoimportdup());
    return true;
}

bool LocalFMMemMgr::handleImportError(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "handleImportError");

    if(!pFmMessage->has_ktmemoryimporterr())
    {
        FM_LOG_ERROR( "No memory import error in message");
        return false;
    }
    const lwswitch::ktMemoryImportErr &impErr = pFmMessage->ktmemoryimporterr();
    dupExportImportTuple exportTuple = {impErr.handletoexportclient(), impErr.handletoexportobject()};

    tExportMap::iterator it;
    lwosEnterCriticalSection(&mLock);
    if( (it = exportDupMap.find(exportTuple)) == exportDupMap.end())
    {
        FM_LOG_ERROR( "Import error request for map not previously imported");
        lwosLeaveCriticalSection(&mLock);
        return false;
    }

    LwRmFree(mHandleFmClient, impErr.gpuid(), it->second.front());
    if (it->second.size() == 1)
    {
        FM_LOG_DEBUG( "handleImportError: found last entry erasing list from map");
        exportDupMap.erase(exportTuple);
    } 
    else 
    {
        FM_LOG_DEBUG( "handleImportError: found entry removing from list");
        it->second.pop_front();
    }
    lwosLeaveCriticalSection(&mLock);
    return true;
}

bool LocalFMMemMgr::handleUnimportRequest(lwswitch::fmMessage *pFmMessage)
{
    LwU32 err = 0;
    FM_LOG_DEBUG( "handleUnimportRequest called");
    if(!pFmMessage->has_memoryunimportreq())
    {
        FM_LOG_ERROR( "No memory unimport request in message");
        return false;
    }

    const lwswitch::memoryUnimportReq &unimpReq = pFmMessage->memoryunimportreq();
    dupExportImportTuple exportTuple = {unimpReq.handletoexportclient(), unimpReq.handletoexportobject()};

    tExportMap::iterator it;
    lwosEnterCriticalSection(&mLock);
    if( (it = exportDupMap.find(exportTuple)) == exportDupMap.end())
    {
        FM_LOG_ERROR( "Unimport request for map not previously imported");
        lwosLeaveCriticalSection(&mLock);
        return sendUnimportResponse(unimpReq, pFmMessage->nodeid(), MEM_EXPORT_DUP_MISSING);;
    }

    err = LwRmFree(mHandleFmClient, unimpReq.gpuid(), it->second.front());
    if (it->second.size() == 1)
    {
        FM_LOG_DEBUG( "handleUnimportRequest: found last entry erasing map entry");
        exportDupMap.erase(exportTuple);
    } 
    else 
    {
        FM_LOG_DEBUG( "handleUnimportRequest: found entry removing from list");
        it->second.pop_front();
    }
    lwosLeaveCriticalSection(&mLock);
    if (err)
        return sendUnimportResponse(unimpReq, pFmMessage->nodeid(), MEM_OBJECT_FREE_FAILED);
    else
        return sendUnimportResponse(unimpReq, pFmMessage->nodeid(), 0);
}

bool LocalFMMemMgr::handleUnimportResponse(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "handleUnimportResponse");

    if(!pFmMessage->has_memoryunimportrsp())
    {
        FM_LOG_ERROR( "No memory unimport response in message");
        return false;
    }
    const lwswitch::memoryUnimportRsp &unimpReq = pFmMessage->memoryunimportrsp();
    FM_LOG_DEBUG( "Unimport response: handleToExportClient %u hExpoObj %u errcode %u", 
                unimpReq.handletoexportclient(), unimpReq.handletoexportobject(), unimpReq.errcode());

    return true;
}

void
LocalFMMemMgr::handleMessage(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "message type %d", pFmMessage->type());
    switch ( pFmMessage->type() )
    {
    case lwswitch::FM_MEMORY_IMPORT_REQ:
        handleImportRequest(pFmMessage);
        break;
        
    case lwswitch::FM_MEMORY_IMPORT_RSP:
        handleImportResponse(pFmMessage);
        break;

    case lwswitch::FM_KT_MEMORY_IMPORT_ERR:
        handleImportError(pFmMessage);
        break;

    case lwswitch::FM_MEMORY_UNIMPORT_REQ:
        handleUnimportRequest(pFmMessage);
        break;

    case lwswitch::FM_MEMORY_UNIMPORT_RSP:
        handleUnimportResponse(pFmMessage);
        break;

    default:
        FM_LOG_ERROR( "Unknown message type %d", pFmMessage->type());
        break;
    }
}
#endif
