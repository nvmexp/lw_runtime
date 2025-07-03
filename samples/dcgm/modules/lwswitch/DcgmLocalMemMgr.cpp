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

#include "logging.h"
#include "lwtypes.h"
#include "lwRmApi.h"
#include "lwos.h"
#include "lwos.h"
#include "DcgmLocalMemMgr.h"
#include "DcgmLocalFabricManager.h"

#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl0000.h"
#include "class/cl0000.h"
#include "class/cl000f.h"
#include "class/cl0005.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "ctrl/ctrl000f.h"
#include "ctrl/ctrl000f_imex.h"
#include "ctrl/ctrl00f4.h"
#include "ctrl/ctrl00f5.h"

#define FM_MEM_EVENTS_HANDLE                0x80000000
#define FM_GPU_DEVICE_HANDLE_BASE           FM_MEM_EVENTS_HANDLE + 1
#define FM_GPU_MEM_SUB_DEVICE_HANDLE_BASE   (FM_GPU_DEVICE_HANDLE_BASE + LW0000_CTRL_GPU_MAX_PROBED_GPUS)
#define FM_SESSION_HANDLE                   (FM_GPU_MEM_SUB_DEVICE_HANDLE_BASE + LW0000_CTRL_GPU_MAX_PROBED_GPUS)

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

DcgmLocalMemMgr::DcgmLocalMemMgr(DcgmFMLocalCoOpMgr *pLocalCoopMgr, DcgmLocalFabricManagerControl *pLocalFmControl)
    :mFMLocalCoOpMgr(pLocalCoopMgr),
     mLocalFabricManagerControl(pLocalFmControl), 
     mHandleFmClient(0), 
     mHandleFmSession(getHandleForFmSession()),
     mRMFd(-1)
{
    lwosInitializeCriticalSection( &mLock );
    PRINT_DEBUG("", "DcgmLocalMemMgr constructor called");
}

DcgmLocalMemMgr::~DcgmLocalMemMgr()
{
    StopAndWait(FM_EVENTS_POLL_INTERVAL_MS * 2);

    if(mHandleFmClient)
    {
        LwRmFree(mHandleFmClient, LW01_NULL_OBJECT, mHandleFmClient);
    }
    lwosDeleteCriticalSection( &mLock );
    PRINT_DEBUG("", "DcgmLocalMemMgr destructor called");
}

LwHandle
DcgmLocalMemMgr::getDeviceHandleForGpu(LwU32 gpuIndex)
{
    return (FM_GPU_DEVICE_HANDLE_BASE + gpuIndex);
}

LwHandle
DcgmLocalMemMgr::getHandleForFmSession()
{
    return FM_SESSION_HANDLE;
}

LwHandle
DcgmLocalMemMgr::getMemEventsHandle()
{
    return FM_MEM_EVENTS_HANDLE;
}

LwHandle
DcgmLocalMemMgr::getMemSubDeviceHandleForGpu(LwU32 gpuIndex)
{
    return (FM_GPU_MEM_SUB_DEVICE_HANDLE_BASE + gpuIndex);
}

bool DcgmLocalMemMgr::allocRmEvent()
{
    LwHandle hEvent = getMemEventsHandle();
    LwU32 retVal;

    retVal = LwRmAllocOsEvent(mHandleFmClient, 0, &mHandleOsEvent, &mRMFd);
    if (retVal != LWOS_STATUS_SUCCESS)
    {
        PRINT_ERROR("%x", "LwRmAllocOsEvent() returned 0x%x", retVal);
        return false;
    }

    LW0005_ALLOC_PARAMETERS params;
    memset(&params, 0, sizeof(params));
    params.hParentClient = mHandleFmClient;
    params.hSrcResource = mHandleFmSession;
    params.hClass = LW01_EVENT_OS_EVENT;
    params.notifyIndex = LW000F_NOTIFIERS_FABRIC_EVENT;
    params.data = (LwP64)(&mRMFd);

    retVal = LwRmAlloc(mHandleFmClient, mHandleFmSession, hEvent, LW01_EVENT, &params);
    if (retVal != LWOS_STATUS_SUCCESS)
    {
        PRINT_ERROR("%x", "LwRmAlloc() returned 0x%x when trying to alloc LW01_EVENT", retVal);
        return false;
    }

    return true;
}

bool DcgmLocalMemMgr::waitForEvents()
{
    while (!ShouldStop())
    {
        struct pollfd read_fds[FM_EVENTS_POLL_NUM_FDS];
        read_fds[0].fd = mRMFd;
        read_fds[0].events = POLLIN | POLLPRI;
        read_fds[0].revents = 0;

        int n_avail=0;

        if((n_avail = poll(read_fds, FM_EVENTS_POLL_NUM_FDS, FM_EVENTS_POLL_INTERVAL_MS)) == 0)
        {
            PRINT_DEBUG("", "DcgmLocalMemMgr poll timed out, will poll again");
            continue;
        }
        else if(n_avail == -1)
        {
            if(errno == EINTR)
            {
                PRINT_DEBUG("", "DcgmLocalMemMgr poll interrupted by signal");
                continue;
            }
            else 
            {
                PRINT_ERROR("%d", "DcgmLocalMemMgr poll error=%d", errno);
                return false;
            }
        }
        else
        {
            PRINT_DEBUG("%d", "read_fds[0].revents = %d", read_fds[0].revents);
            return true;
        }
    }
    return false; 
}

bool
DcgmLocalMemMgr::probeDevices(LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS &probeParams)
{
    LwU32 retVal;
    //Probe device IDs
    if((retVal = LwRmControl(mHandleFmClient, mHandleFmClient, LW0000_CTRL_CMD_GPU_GET_PROBED_IDS, &probeParams, sizeof(probeParams))) != LWOS_STATUS_SUCCESS)
    {
        if (LWOS_STATUS_ERROR_GPU_IS_LOST == retVal) 
        {
            PRINT_ERROR("", "GPU is lost");
            return false;
        }
        PRINT_ERROR("", "Failed to probe device IDs");
        return false;
    }
    PRINT_DEBUG("","Probe device IDs done!!!");

    //check if we found any GPUs
    if ((probeParams.gpuIds[0] == LW0000_CTRL_GPU_ILWALID_ID) &&
        (probeParams.excludedGpuIds[0] == LW0000_CTRL_GPU_ILWALID_ID))
    {
        // no GPUs found
        PRINT_ERROR("", "No GPUs found in probe");
        return false;
    }
    PRINT_DEBUG("","GPUs found during probe!!!");
    return true;
}

bool
DcgmLocalMemMgr::getPciInfo(LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS &probeParams, LW0000_CTRL_GPU_GET_PCI_INFO_PARAMS *pciInfoParams)
{
    LwU32 retVal;
    LwU32 i;
    // Get the list of GPUs
    for (i = 0;
         i < LW0000_CTRL_GPU_MAX_PROBED_GPUS && probeParams.gpuIds[i] != LW0000_CTRL_GPU_ILWALID_ID;
         i++)
    {
        pciInfoParams[i].gpuId = probeParams.gpuIds[i];

        PRINT_DEBUG("%d %d", "Getting PCI info for index = %d GPU ID = %d", i, probeParams.gpuIds[i]);
        if (LWOS_STATUS_SUCCESS !=
            (retVal = LwRmControl(mHandleFmClient, mHandleFmClient, LW0000_CTRL_CMD_GPU_GET_PCI_INFO, &(pciInfoParams[i]),
                       sizeof(pciInfoParams[i]))))
        {
            PRINT_ERROR("%x", "Failed to get PCI Info, error:0x%x", retVal);
            if (LWOS_STATUS_ERROR_GPU_IS_LOST == retVal)
            {
                PRINT_ERROR("", "GPU is lost");
                return false;
            }
            PRINT_ERROR("", "Failed to get the list of GPUs");
            return false;
        }
    }
    PRINT_DEBUG("", "Got the list of GPUs");

    mDeviceCount = i;
    return true;
}

bool
DcgmLocalMemMgr::attachGpus(const LW0000_CTRL_GPU_GET_PCI_INFO_PARAMS *pciInfoParams)
{
    LwU32 retVal;
    LwU32 i, j;
    LW0000_CTRL_GPU_MODIFY_DRAIN_STATE_PARAMS drainParms = {0};
    LW0000_CTRL_GPU_ATTACH_IDS_PARAMS attachParams = {{0}};
    for (i = 0; i < mDeviceCount; ++i)
    {
        LwBool bSkip = false;

        if (!bSkip)
        {
            drainParms.gpuId = pciInfoParams[i].gpuId;
            retVal = LwRmControl(mHandleFmClient, mHandleFmClient, LW0000_CTRL_CMD_GPU_QUERY_DRAIN_STATE,
                                    &drainParms, sizeof(drainParms));
            if (LWOS_STATUS_SUCCESS != retVal)
            {
                PRINT_ERROR("%d", "Unable to query drain state retVal=%d", retVal);
                return false;
            }

            bSkip = drainParms.newState == LW0000_CTRL_GPU_DRAIN_STATE_ENABLED;
            if (bSkip)
            {
                PRINT_INFO("%x","GPUid 0x%x is in a drain state",
                            pciInfoParams[i].gpuId);
                return false;
            }
            else
            {
                attachParams.gpuIds[0] = pciInfoParams[i].gpuId;
                attachParams.gpuIds[1] = LW0000_CTRL_GPU_ILWALID_ID;
                retVal = LwRmControl(mHandleFmClient, mHandleFmClient, LW0000_CTRL_CMD_GPU_ATTACH_IDS,
                                        &attachParams, sizeof(attachParams));
                // everything else just ignore
                // but these can be caused by cgroup
                bSkip = (retVal == LWOS_STATUS_ERROR_OPERATING_SYSTEM) ||
                        (retVal == LWOS_STATUS_ERROR_INSUFFICIENT_PERMISSIONS);
                if (bSkip)
                {
                    PRINT_INFO("%x","GPUid 0x%x has been blocked by the OS",
                                pciInfoParams[i].gpuId);
                    return false;
                }
            }
        }
    }
    PRINT_DEBUG("", "Attach GPU IDs done!!!");
 
    // Print list of GPUs found
    i = 0; j = 0;
    for (j = 0; j < mDeviceCount; ++j)
    {
        PRINT_DEBUG("%u 0x%X %u %u %u",
                    "Found GPU with index %u, GPU ID 0x%X, domain:%u bus:%u device:%u",
                    i,
                    pciInfoParams[j].gpuId,
                    pciInfoParams[j].domain,
                    pciInfoParams[j].bus,
                    pciInfoParams[j].slot);
        i++;
    }
    return true;
}

//TODO: build map of busId -> (deviceHandle, subDeviceHandle). Pass in PciInfoParams as arguement
bool 
DcgmLocalMemMgr::allocDevices()
{
    LwU32 retVal;
    LwU32 j;
    for (j = 0; j < mDeviceCount; j++)
    {
        //Alloc handle for each GPU
        LW0080_ALLOC_PARAMETERS devParams = { };
        devParams.deviceId = j;
        PRINT_DEBUG("%d %x", "Device ID=%d handle = %x", devParams.deviceId, getDeviceHandleForGpu(j));
        retVal = LwRmAlloc(mHandleFmClient, mHandleFmClient, getDeviceHandleForGpu(j), LW01_DEVICE_0, &devParams);
        if (retVal != LWOS_STATUS_SUCCESS)
        {
            PRINT_ERROR("%x", "LwRmAlloc() returned 0x%x when trying to alloc LW01_DEVICE_0", retVal);
            return false;
        }
        PRINT_DEBUG("%u", "Alloc GPU device %u done!!!", j);

        //Alloc sub-device for memory
        LW2080_ALLOC_PARAMETERS subDevParams = {0};
        subDevParams.subDeviceId = 0;
        PRINT_DEBUG("%x %x", "Sub-Device ID=%d handle = %x", subDevParams.subDeviceId, getMemSubDeviceHandleForGpu(j));
        retVal = LwRmAlloc(mHandleFmClient, getDeviceHandleForGpu(j), getMemSubDeviceHandleForGpu(j), LW20_SUBDEVICE_0, &subDevParams);
        if (retVal != LWOS_STATUS_SUCCESS)
        {
            PRINT_ERROR("%x", "LwRmAlloc() returned 0x%x when trying to alloc ", retVal);
            return false;
        }
        PRINT_DEBUG("%u", "Alloc memory sub-device for GPU %u done!!!", j);
    }
    return true;
}

void
DcgmLocalMemMgr::createProbedGpuIdToIndexMap(LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS &probeParams)
{
    for (LwU32 i = 0; i < mDeviceCount; i++)
    {
        mGpuProbedGpuIdToIndexMap[ probeParams.gpuIds[i] ] = i;
    }
    return;
}

LwU32
DcgmLocalMemMgr::getProbedGpuIdToIndex(LwU32 probedGpuId)
{
    return mGpuProbedGpuIdToIndexMap[probedGpuId];
}

bool
DcgmLocalMemMgr::initMemMgr()
{
    LwU32 retVal;
    LwU32 i, j;
    LW0000_CTRL_GPU_GET_PCI_INFO_PARAMS pciInfoParams[LW0000_CTRL_GPU_MAX_PROBED_GPUS] = {{0}};
    LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS probeParams = {{0}};

    //allocate handle for FM client
    if ((retVal = LwRmAllocRoot(&mHandleFmClient)) != LWOS_STATUS_SUCCESS)
    {
        PRINT_ERROR("%x", "Failed to allocate RM client. returned 0x%x", retVal);
        return false;
    }
    PRINT_DEBUG("","LwRmAllocRoot done!!!");
   
    if( probeDevices(probeParams) == false)
        return false;

    if( getPciInfo(probeParams, pciInfoParams) == false)
        return false;

    if( attachGpus( pciInfoParams ) == false)
        return false;

    if( allocDevices() == false)
        return false;

    createProbedGpuIdToIndexMap(probeParams);

    LW000F_ALLOCATION_PARAMETERS fmSessionAllocParams = {0};
    //Alloc FM session, pass in handle to the FM session.
    retVal = LwRmAlloc(mHandleFmClient, mHandleFmClient, mHandleFmSession,
                       FABRIC_MANAGER_SESSION, &fmSessionAllocParams);
    if (retVal != LWOS_STATUS_SUCCESS)
    {
        PRINT_ERROR("%x", "LwRmAlloc() returned 0x%x when trying to alloc FABRIC_MANAGER_SESSION", retVal);
        return false;
    }
    PRINT_DEBUG("", "alloc FM session done!!!");

    //set FM session  Node ID
    LW000F_CTRL_SET_FABRIC_NODE_ID_PARAMS fmSessionNodeIdParams;
    fmSessionNodeIdParams.nodeId = mLocalFabricManagerControl->getLocalNodeId();
    retVal = LwRmControl(mHandleFmClient, mHandleFmSession, LW000F_CTRL_CMD_SET_FABRIC_NODE_ID, 
                         &fmSessionNodeIdParams, sizeof(fmSessionNodeIdParams));
    if (retVal != LWOS_STATUS_SUCCESS)
    {
        PRINT_ERROR("%x", "Error 0x%x when setting Node Id for FABRIC_MANAGER_SESSION", retVal);
        return false;
    }
    PRINT_DEBUG("", "Set Node ID done!!!");

    //Alloc Events
    if(allocRmEvent() == false)
    {
        return false;
    }
    
    //set FM session state so lwca can work now
    retVal = LwRmControl(mHandleFmClient, mHandleFmSession, LW000F_CTRL_CMD_SET_FM_STATE, NULL, 0);
    if (retVal != LWOS_STATUS_SUCCESS)
    {
        PRINT_ERROR("%x", "Error 0x%x when setting state for FABRIC_MANAGER_SESSION", retVal);
        return false;
    }
    PRINT_DEBUG("", "Set FM session state done!!!");

    return true;
}

void
DcgmLocalMemMgr::run(void)
{
    PRINT_DEBUG("", "DcgmLocalMemMgr run called");

    if(false == initMemMgr()) {
        PRINT_ERROR("", "Unable to initialize DcgmLocalMemMgr");
        if(mHandleFmClient != 0)
            LwRmFree(mHandleFmClient, LW01_NULL_OBJECT, mHandleFmClient);
        return;
    }

    while(!ShouldStop())
    {
        if (waitForEvents())
        {
            if ( ShouldStop() )
            {
                break;
            }
            if(processEvents() == false)
            {
                PRINT_ERROR("", "Error processing events");
                return;
            }
        }
        else
        {
            PRINT_ERROR("", "Error waiting for events");
            return;
        }
    }
}

bool
DcgmLocalMemMgr::processEvents()
{
    PRINT_DEBUG("","Got Mem event from RM, will read all available events");
    LwU32 retVal;
    LwU32 more_events = 1;
    LwUnixEvent unixEvent;

    //clear event notification
    if((retVal = LwRmGetEventData(mHandleFmClient, mRMFd, &unixEvent, &more_events)) !=  LWOS_STATUS_SUCCESS ) 
    {
        PRINT_ERROR("", "processEvents called but no event found");
        return true;
    }

    //Read event data while more  events remain queued in RM
    bool read_more_events = true;
    while(read_more_events)
    {
        LW000F_CTRL_GET_FABRIC_EVENTS_PARAMS events;
        memset(&events, 0, sizeof(events));
        if(( retVal = LwRmControl(mHandleFmClient, mHandleFmSession, LW000F_CTRL_CMD_GET_FABRIC_EVENTS, &events, sizeof(events))) == LWOS_STATUS_SUCCESS ) 
        {
            for (unsigned int i = 0; i < events.numEvents; i++)
            {
                //send import/unimport request based on event type
                switch (events.eventArray[i].eventType)
                {
                    case LW000F_CTRL_FABRIC_EVENT_TYPE_IMPORT:
                    {
                        if(sendImportRequest(events.eventArray[i])) 
                            PRINT_DEBUG("", "sendImportRequest Success");
                        else
                            PRINT_ERROR("", "sendImportRequest Failed");
                        break;
                    }
                    case LW000F_CTRL_FABRIC_EVENT_TYPE_RELEASE:
                    {
                        if(sendUnimportRequest(events.eventArray[i]))
                            PRINT_DEBUG("", "sendUnimportRequest Success");
                        else
                            PRINT_ERROR("", "sendUnimportRequest Failed");
                        break;
                    }
                    default:
                        PRINT_ERROR("%d", "Unknown event type %d", events.eventArray[i].eventType);
                        break;
                }
            }
            read_more_events = events.bMoreEvents;
        }
        else
        {
            PRINT_ERROR("%d", "Error reading fabric events from RM %d", retVal);
            return false;
        }
    }
    PRINT_DEBUG("", "Done reading all events");
    return true;
}

bool DcgmLocalMemMgr::sendImportRequest(LW000F_CTRL_FABRIC_EVENT &evData)
{
    uint32 peerNodeId = evData.nodeId;

    PRINT_DEBUG("%d %d", "event peerNodeId=%d event gpuId=%d", evData.nodeId, evData.gpuId);

    PRINT_DEBUG("%d %d %d %d", "impClient=%d impObj=%d expClient=%d, expObj=%d", 
                evData.hImportClient, evData.hImportObject, evData.hExportClient, evData.hExportObject);
    lwswitch::memoryImportReq *reqMsg = new lwswitch::memoryImportReq();

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    //dup object
    LwHandle importObjectDup = 0;
    LwU32 rv =  LwRmDupObject2(mHandleFmClient, mHandleFmClient, &importObjectDup, evData.hImportClient, evData.hImportObject, 0);
    if(rv != LW_OK)
    {
        delete reqMsg;
        PRINT_ERROR("%d", "Import object dup error %d", rv);
        return false;
    }
    else
    {
        PRINT_DEBUG("%d %d", "Import object dup successful. lwdaImportClient=%d lwdaImportObject=%d", evData.hImportClient, evData.hImportObject);
    }

    reqMsg->set_kthandletoimportdup(importObjectDup);
#endif

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
    dcgmReturn_t retVal;
    retVal =  mFMLocalCoOpMgr->SendMessageToPeerLFM(peerNodeId, pFmMessage, false);
    if (retVal != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "SendMessageToPeerLFM failed error=%d", retVal);
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        LwRmFree(mHandleFmClient, mHandleFmClient, importObjectDup);
#endif
        delete( pFmMessage );
        return false;
    }
    
    delete( pFmMessage );
    return true;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
bool
DcgmLocalMemMgr::readPageTableEntries(lwswitch::memoryImportRsp *rspMsg, LwHandle objectHandle)
{    
    LwU32 retVal;
    lwswitch::ktPageTableEntries *pPageTableEntries = new lwswitch::ktPageTableEntries();
    pPageTableEntries->set_offset(0);
    //loop read and fill in PTEs into rspMsg 
    LW00F4_CTRL_EXPORT_DESCRIBE_PARAMS params;
    params.offset = 0;
    while((retVal = LwRmControl(mHandleFmClient, objectHandle, LW00F4_CTRL_CMD_EXPORT_DESCRIBE, &params, sizeof(params))) == LW_OK)
    {
        PRINT_DEBUG("%llu %u %llu","params.offset=%llu params.numPfns=%u params.totalPfns=%llu",params.offset, params.numPfns, params.totalPfns);
        for(unsigned int i = 0; i < params.numPfns; i++) 
        {
            pPageTableEntries->add_pageframenumberarray(params.pfnArray[i]);
            //PRINT_DEBUG("%llu %d","PFN%llu=%d",i  + params.offset, params.pfnArray[i]);
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
        PRINT_ERROR("%d", "Memory export describe error %d", retVal);
        delete pPageTableEntries;
        return false;
    }

    rspMsg->set_allocated_ktptentries(pPageTableEntries);
    return true;
}
#endif

bool DcgmLocalMemMgr::sendImportResponse(const lwswitch::memoryImportReq &reqMsg, uint32 peerNodeId, uint32 errCode)
{
    dcgmReturn_t retVal;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::memoryImportRsp *rspMsg = new lwswitch::memoryImportRsp();

    rspMsg->set_handletoexportclient(reqMsg.handletoexportclient());
    rspMsg->set_handletoexportobject(reqMsg.handletoexportobject());
    rspMsg->set_kthandletoimportdup(reqMsg.kthandletoimportdup());
    rspMsg->set_gpuid(reqMsg.gpuid());

    dupExportImportTuple exportTuple = {reqMsg.handletoexportclient(), reqMsg.handletoexportobject()};
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    lwosEnterCriticalSection(&mLock);
    if((errCode == 0) && (readPageTableEntries(rspMsg, exportDupMap[exportTuple].front()) == false))
    {
        PRINT_ERROR("", "Unable to read page table entries");
        errCode = MEM_DESCRIBE_FAILED;
    }
    lwosLeaveCriticalSection(&mLock);
#endif
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
    if (retVal != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "SendMessageToPeerLFM failed error=%d", retVal);
        delete( pFmMessage );
        return false;
    }
    PRINT_DEBUG("%d", "Sent Import Response to node %d", peerNodeId);
    
    delete( pFmMessage );
    return true;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
bool DcgmLocalMemMgr::sendImportError(const lwswitch::memoryImportRsp &rspMsg, uint32 peerNodeId, uint32 errCode)
{
    dcgmReturn_t retVal;
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
    if (retVal != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "sendImportError: SendMessageToPeerLFM failed error=%d", retVal);
        delete( pFmMessage );
        return false;
    }
    PRINT_DEBUG("%d %d", "Sent Import error %d to node %d ", errCode, peerNodeId);
    delete( pFmMessage );
    return true;
}
#endif

bool DcgmLocalMemMgr::sendUnimportRequest(LW000F_CTRL_FABRIC_EVENT &evData)
{
    dcgmReturn_t retVal;
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
    if (retVal != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "SendMessageToPeerLFM failed error=%d", retVal);
        delete( pFmMessage );
        return false;
    }
    
    delete( pFmMessage );
    return true;
}

bool DcgmLocalMemMgr::sendUnimportResponse(const lwswitch::memoryUnimportReq &reqMsg, uint32 peerNodeId, uint32 errCode)
{
    dcgmReturn_t retVal;
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
    if (retVal != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "SendMessageToPeerLFM failed error=%d", retVal);
        delete( pFmMessage );
        return false;
    }
    PRINT_DEBUG("%d", "Sent Unimport Response retVal=%d", retVal);
    
    delete( pFmMessage );
    return true;
}

bool DcgmLocalMemMgr::handleImportRequest(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG("", "handleImportRequest called");
    if(!pFmMessage->has_memoryimportreq())
    {
        PRINT_ERROR("", "No memory import request in message");
        return false;
    }
    const lwswitch::memoryImportReq &impReq = pFmMessage->memoryimportreq();
    dupExportImportTuple exportTuple = {impReq.handletoexportclient(), impReq.handletoexportobject()};
    LwU32 gpuIndex = getProbedGpuIdToIndex(impReq.gpuid());
    //dup object
    LwHandle exportObjectDup = 0;
    LwU32 retVal = LwRmDupObject2(mHandleFmClient, getMemSubDeviceHandleForGpu(gpuIndex), &exportObjectDup, 
                                                impReq.handletoexportclient(), impReq.handletoexportobject(), 0);
    if(retVal != LW_OK)
    {
        PRINT_ERROR("%d %d %d", "Export object dup error %d importClient=%d importObject=%d", retVal, impReq.handletoexportclient(), impReq.handletoexportobject());
        return sendImportResponse(impReq, pFmMessage->nodeid(), MEM_DUP_FAILED);
    }

    lwosEnterCriticalSection(&mLock);
    exportDupMap[exportTuple].push_front(exportObjectDup);
    PRINT_DEBUG("%x %x %lu","Added tuple(%x, %x) map.size()=%lu",impReq.handletoexportclient(), impReq.handletoexportobject(), exportDupMap.size());
    lwosLeaveCriticalSection(&mLock);
    return sendImportResponse(impReq, pFmMessage->nodeid(), 0);
}

bool DcgmLocalMemMgr::handleImportResponse(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG("", "handleImportResponse");
    if(!pFmMessage->has_memoryimportrsp())
    {
        PRINT_ERROR("", "No memory import response in message");
        return false;
    }
    const lwswitch::memoryImportRsp &impRsp = pFmMessage->memoryimportrsp();

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    //set all Page Table Entries to RM. If error sendImportError
    int pageFrameNumber;
    //loop  set PTEs to RM until RM says done or we runout of PTEs in impRsp
    int batchIndex = 0;
    for (pageFrameNumber = 0; 
         pageFrameNumber < impRsp.ktptentries().pageframenumberarray().size(); 
         pageFrameNumber += LW00F4_CTRL_EXPORT_DESCRIBE_PFN_ARRAY_SIZE)
    {
        LW00F5_CTRL_IMPORT_VALIDATE_PARAMS params;
        params.offset = pageFrameNumber;
        PRINT_DEBUG("%d", "#PTE = %d ",  impRsp.ktptentries().pageframenumberarray().size());
        for( batchIndex = 0; 
             batchIndex < LW00F5_CTRL_IMPORT_VALIDATE_PFN_ARRAY_SIZE
             && (batchIndex + pageFrameNumber) < impRsp.ktptentries().pageframenumberarray().size(); 
             batchIndex++ )
        {
            params.pfnArray[batchIndex] = impRsp.ktptentries().pageframenumberarray(params.offset + batchIndex);
            //PRINT_DEBUG("%llu %d", "PFN%llu=%d ",  batchIndex + params.offset , params.pfnArray[batchIndex]);
        }
        params.numPfns = batchIndex;
        uint32 retVal;
        retVal = LwRmControl(mHandleFmClient, impRsp.kthandletoimportdup()
                      , LW00F5_CTRL_CMD_IMPORT_VALIDATE, &params, sizeof(params));

        if(retVal != LW_OK)
        {
            PRINT_ERROR("%d", "Validate failed retVal=%d", retVal);
            sendImportError(impRsp, pFmMessage->nodeid(), MEM_VALIDATE_FAILED);
            break;
        }
    }

    if ((pageFrameNumber - LW00F4_CTRL_EXPORT_DESCRIBE_PFN_ARRAY_SIZE + batchIndex )== impRsp.ktptentries().pageframenumberarray().size())
    {
        PRINT_DEBUG("", "Import successful");
    }
    else
    {
        PRINT_ERROR("%d %d", "Import Failed PFN=%d expected=%d", 
                    (pageFrameNumber - LW00F4_CTRL_EXPORT_DESCRIBE_PFN_ARRAY_SIZE + batchIndex ),
                    impRsp.ktptentries().pageframenumberarray().size());
    }

    LwRmFree(mHandleFmClient, mHandleFmClient, impRsp.kthandletoimportdup());
#endif
    return true;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
bool DcgmLocalMemMgr::handleImportError(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG("", "handleImportError");

    if(!pFmMessage->has_ktmemoryimporterr())
    {
        PRINT_ERROR("", "No memory import error in message");
        return false;
    }
    const lwswitch::ktMemoryImportErr &impErr = pFmMessage->ktmemoryimporterr();
    dupExportImportTuple exportTuple = {impErr.handletoexportclient(), impErr.handletoexportobject()};

    tExportMap::iterator it;
    lwosEnterCriticalSection(&mLock);
    if( (it = exportDupMap.find(exportTuple)) == exportDupMap.end())
    {
        PRINT_ERROR("", "Import error request for map not previously imported");
        lwosLeaveCriticalSection(&mLock);
        return false;
    }

    LwRmFree(mHandleFmClient, impErr.gpuid(), it->second.front());
    if (it->second.size() == 1)
    {
        PRINT_DEBUG("", "handleImportError: found last entry erasing list from map");
        exportDupMap.erase(exportTuple);
    } 
    else 
    {
        PRINT_DEBUG("", "handleImportError: found entry removing from list");
        it->second.pop_front();
    }
    lwosLeaveCriticalSection(&mLock);
    return true;
}
#endif

bool DcgmLocalMemMgr::handleUnimportRequest(lwswitch::fmMessage *pFmMessage)
{
    LwU32 err = 0;
    PRINT_DEBUG("", "handleUnimportRequest called");
    if(!pFmMessage->has_memoryunimportreq())
    {
        PRINT_ERROR("", "No memory unimport request in message");
        return false;
    }

    const lwswitch::memoryUnimportReq &unimpReq = pFmMessage->memoryunimportreq();
    dupExportImportTuple exportTuple = {unimpReq.handletoexportclient(), unimpReq.handletoexportobject()};

    tExportMap::iterator it;
    lwosEnterCriticalSection(&mLock);
    if( (it = exportDupMap.find(exportTuple)) == exportDupMap.end())
    {
        PRINT_ERROR("", "Unimport request for map not previously imported");
        lwosLeaveCriticalSection(&mLock);
        return sendUnimportResponse(unimpReq, pFmMessage->nodeid(), MEM_EXPORT_DUP_MISSING);;
    }

    err = LwRmFree(mHandleFmClient, unimpReq.gpuid(), it->second.front());
    if (it->second.size() == 1)
    {
        PRINT_DEBUG("", "handleUnimportRequest: found last entry erasing map entry");
        exportDupMap.erase(exportTuple);
    } 
    else 
    {
        PRINT_DEBUG("", "handleUnimportRequest: found entry removing from list");
        it->second.pop_front();
    }
    lwosLeaveCriticalSection(&mLock);
    if (err)
        return sendUnimportResponse(unimpReq, pFmMessage->nodeid(), MEM_OBJECT_FREE_FAILED);
    else
        return sendUnimportResponse(unimpReq, pFmMessage->nodeid(), 0);
}

bool DcgmLocalMemMgr::handleUnimportResponse(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG("", "handleUnimportResponse");

    if(!pFmMessage->has_memoryunimportrsp())
    {
        PRINT_ERROR("", "No memory unimport response in message");
        return false;
    }
    const lwswitch::memoryUnimportRsp &unimpReq = pFmMessage->memoryunimportrsp();
    PRINT_DEBUG("%u %u %u", "Unimport response: handleToExportClient %u hExpoObj %u errcode %u", 
                unimpReq.handletoexportclient(), unimpReq.handletoexportobject(), unimpReq.errcode());

    return true;
}

void
DcgmLocalMemMgr::handleMessage(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG("%d", "message type %d", pFmMessage->type());
    switch ( pFmMessage->type() )
    {
    case lwswitch::FM_MEMORY_IMPORT_REQ:
        handleImportRequest(pFmMessage);
        break;
        
    case lwswitch::FM_MEMORY_IMPORT_RSP:
        handleImportResponse(pFmMessage);
        break;

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    case lwswitch::FM_KT_MEMORY_IMPORT_ERR:
        handleImportError(pFmMessage);
        break;
#endif

    case lwswitch::FM_MEMORY_UNIMPORT_REQ:
        handleUnimportRequest(pFmMessage);
        break;

    case lwswitch::FM_MEMORY_UNIMPORT_RSP:
        handleUnimportResponse(pFmMessage);
        break;

    default:
        PRINT_ERROR("%d", "unknown message type %d", pFmMessage->type());
        break;
    }
}
