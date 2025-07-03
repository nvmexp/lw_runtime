extern "C" {
#include <stdio.h>
#include <stdint.h>
#include "dcgm_client_internal.h"
#include "dcgm_agent_internal.h"    
#include "lwcm_util.h"
#include "lwcmvalue.h"
#include "spinlock.h"
}

#include "LwcmSettings.h"
#include "LwcmStatus.h"
#include "DcgmLogging.h"
#include "DcgmFvBuffer.h"
#include "DcgmPolicyRequest.h"
#include "DcgmModuleApi.h"
#include "dcgm_introspect_structs.h"
#include "dcgm_health_structs.h"
#include "dcgm_policy_structs.h"
#include "dcgm_config_structs.h"
#include "dcgm_diag_structs.h"
#include "dcgm_profiling_structs.h"
#include "DcgmVersion.hpp"

// Wrap each dcgmFunction with apiEnter and apiExit
#define DCGM_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)             \
        static dcgmReturn_t tsapiFuncname argtypes ;                                          \
        dcgmReturn_t DECLDIR dcgmFuncname argtypes                                            \
        {                                                                                     \
            dcgmReturn_t result;                                                              \
            PRINT_DEBUG("Entering %s%s " fmt,                                                 \
                        "Entering %s%s " fmt, #dcgmFuncname, #argtypes, ##__VA_ARGS__);       \
            result = apiEnter();                                                              \
            if (result != DCGM_ST_OK)                                                         \
            {                                                                                 \
                return result;                                                                \
            }                                                                                 \
            result = tsapiFuncname(__VA_ARGS__);                                              \
            apiExit();                                                                        \
            PRINT_DEBUG("Returning %d", "Returning %d", result);                              \
            return result;                                                                    \
        }

extern "C" {
#include "entry_point.h"
}

#include "LwcmClientHandler.h"
#include "LwcmHostEngineHandler.h"

/* Define these outside of C linkage since they use C++ features */

/**
 * Structure for representing the global variables of DCGM within a process
 *
 * Variables of this structure are controlled by dcgmGlobalsMutex. Call dcgmGlobalsLock() and dcgmGlobalsUnlock()
 * to access these variables consistently
 */
typedef struct dcgm_globals_t
{
    int isInitialized; /* Has DcgmInit() been successfully called? dcgmShutdown() sets this back to 0 */
    int loggingIsInitialized; /* Has loggingInit() been successfully called? */
    int fieldsAreInitialized; /* Has DcgmFieldsInit() been successfully called? */
    int embeddedEngineStarted; /* Have we started an embedded host engine? */
    int clientHandlerRefCount; /* How many threads are lwrrently using the client handler? This should be
                                  0 unless threads are in requests */
    LwcmClientHandler *clientHandler; /* Pointer to our client handler. This cannot be freed unless 
                                         clientHandlerRefCount reaches 0. Eventually, this should be replaced
                                         with a shared_ptr */
} dcgm_globals_t, *dcgm_globals_p;

// Globals
static volatile unsigned int g_dcgmGlobalsMutex = 0; /* Spin lock to control access to g_dcgmGlobals. This
                                                        is declared outside of g_dcgmGlobals so g_dcgmGlobals
                                                        can be memset to 0 in dcgmShutdown() */
static dcgm_globals_t g_dcgmGlobals = {0}; /* Declared static so we don't export it */


// Instructions:
//
// - Try to make Export Tables backward binary compatible
// - Number all internal functions. Otherwise it's hard to make integrations properly
// - Don't remove rows. Deprecate old functions by putting NULL instead
// - When you do integrations make sure to pad missing functions with NULLs
// - Never renumber functions when integrating. Numbers of functions should always match the
//   module_* numbering
DCGM_INIT_EXTERN_CONST etblDCGMClientInternal g_etblDCGMClientInternal =
{  
        sizeof (g_etblDCGMClientInternal),
        dcgmClientSaveCacheManagerStats,            // 1
        dcgmClientLoadCacheManagerStats,            // 2
};

// Instructions:
//
// - Try to make Export Tables backward binary compatible
// - Number all internal functions. Otherwise it's hard to make integrations properly
// - Don't remove rows. Deprecate old functions by putting NULL instead
// - When you do integrations make sure to pad missing functions with NULLs
// - Never renumber functions when integrating. Numbers of functions should always match the
//   module_* numbering
DCGM_INIT_EXTERN_CONST etblDCGMEngineInternal g_etblDCGMEngineInternal =
{  
        sizeof(g_etblDCGMEngineInternal),                 // Export Table Start
        dcgmEngineRun,                              // 0
        dcgmGetLatestValuesForFields,               // 1
        dcgmGetMultipleValuesForField,              // 2
        dcgmGetFieldValuesSince,                    // 3
        dcgmWatchFieldValue,                        // 4
        dcgmUnwatchFieldValue,                      // 5
        NULL,                                       // 6
        dcgmIntrospectGetFieldMemoryUsage,          // 7
        dcgmMetadataStateSetRunInterval,            // 8
        dcgmIntrospectGetFieldExecTime,             // 9
        NULL,                                       // 10
        dcgmVgpuConfigSet,                          // 11
        dcgmVgpuConfigGet,                          // 12
        dcgmVgpuConfigEnforce,                      // 13
        dcgmGetVgpuDeviceAttributes,                // 14
        dcgmGetVgpuInstanceAttributes,              // 15
        dcgmStopDiagnostic,                         // 16
};

// Instructions:
//
// - Try to make Export Tables backward binary compatible
// - Number all internal functions. Otherwise it's hard to make integrations properly
// - Don't remove rows. Deprecate old functions by putting NULL instead
// - When you do integrations make sure to pad missing functions with NULLs
// - Never renumber functions when integrating. Numbers of functions should always match the
//   module_* numbering
DCGM_INIT_EXTERN_CONST etblDCGMEngineTestInternal g_etblDCGMEngineTestInternal =
{  
        sizeof(g_etblDCGMEngineTestInternal),              // Export Table Start
        dcgmInjectFieldValue,                        // 0
        0,                                           // 1
        dcgmGetCacheManagerFieldInfo,                // 2
        dcgmCreateFakeEntities,                      // 3
        dcgmInjectEntityFieldValue,                  // 4
        dcgmSetEntityLwLinkLinkState                 // 5
};


/*****************************************************************************
* Functions used for locking/unlocking the globals of DCGM within a process
******************************************************************************/

static void dcgmGlobalsLock(void)
{
    lwmlSpinLock(&g_dcgmGlobalsMutex);
}

static void dcgmGlobalsUnlock(void)
{
    lwmlUnlock(&g_dcgmGlobalsMutex);
}

/*****************************************************************************
 *****************************************************************************/
/*****************************************************************************
 * Helper methods to unify code for remote cases and ISV agent case
 *****************************************************************************/
int helperUpdateErrorCodes(dcgmStatus_t statusHandle, int numStatuses, dcgm_config_status_t *statuses)
{
    LwcmStatus *pStatusObj;
    int i;

    if (!statusHandle || numStatuses < 1 || !statuses)
        return -1;

    pStatusObj = (LwcmStatus *)statusHandle;

    for (i = 0; i < numStatuses; i++) 
    {
        pStatusObj->Enqueue(statuses[i].gpuId, statuses[i].fieldId,
                            statuses[i].errorCode);
    }

    return 0;
}

/*****************************************************************************/
int helperUpdateErrorCodes(dcgmStatus_t statusHandle, lwcm::Command *pGroupCmd)
{
    LwcmStatus *pStatusObj;
    int index;

    if ((NULL == statusHandle) || (NULL == pGroupCmd))
        return -1;

    pStatusObj = (LwcmStatus *)statusHandle;


    for (index = 0; index < pGroupCmd->errlist_size(); index++) {
        pStatusObj->Enqueue(pGroupCmd->errlist(index).gpuid(), pGroupCmd->errlist(index).fieldid(),
                pGroupCmd->errlist(index).errorcode());
    }

    return 0;
}

/*****************************************************************************/
/* Get a pointer to the client handler. If this returns non-NULL, you need to call
   lwcmapiReleaseClientHandler() to decrease the reference count to it */
static LwcmClientHandler *lwcmapiAcquireClientHandler(bool shouldAllocate)
{
    LwcmClientHandler *retVal = NULL;

    dcgmGlobalsLock();

    if(g_dcgmGlobals.clientHandler)
    {
        g_dcgmGlobals.clientHandlerRefCount++;
        retVal = g_dcgmGlobals.clientHandler;
        PRINT_DEBUG("%d", "Incremented the client handler to %d",
                    g_dcgmGlobals.clientHandlerRefCount);
    }
    else if(shouldAllocate)
    {
        PRINT_INFO("", "Allocated the client handler");
        g_dcgmGlobals.clientHandler = new LwcmClientHandler();
        retVal = g_dcgmGlobals.clientHandler;
        g_dcgmGlobals.clientHandlerRefCount = 1;
    }
    /* Else: retVal is left as NULL. We want this in case another thread gets in and
             changes g_dcgmGlobals.clientHandler between the unlock and return */
    
    dcgmGlobalsUnlock();

    return retVal;
}

/*****************************************************************************/
/* Release a client handler that was acquired with lwcmapiAcquireClientHandler */
static void lwcmapiReleaseClientHandler()
{
    dcgmGlobalsLock();

    if(g_dcgmGlobals.clientHandler)
    {
        if(g_dcgmGlobals.clientHandlerRefCount < 1)
        {
            PRINT_ERROR("%d", "Client handler ref count underflowed. Tried to decrement from %d",
                        g_dcgmGlobals.clientHandlerRefCount);
        }
        else
        {
            g_dcgmGlobals.clientHandlerRefCount--;
            PRINT_DEBUG("%d", "Decremented the client handler to %d",
                        g_dcgmGlobals.clientHandlerRefCount);
        }
    }

    dcgmGlobalsUnlock();
}

/*****************************************************************************/
/* free the client handler that was allocated with lwcmapiAcquireClientHandler */
static void lwcmapiFreeClientHandler()
{
    while(1)
    {
        /* We must not have the globals lock here or nobody else will be able to decrement the ref count */
        while(g_dcgmGlobals.clientHandlerRefCount > 0)
        {
            PRINT_INFO("%d", "Waiting to destroy the client handler. Current refCount: %d", 
                       g_dcgmGlobals.clientHandlerRefCount);
            sleep(1);
        }

        dcgmGlobalsLock();

        /* Now that we have the lock, we have to re-check state */

        if(!g_dcgmGlobals.clientHandler)
        {
            /* Another thread did our work for us. Unlock and get out */
            PRINT_INFO("", "Another thread freed the client handler for us.");
            dcgmGlobalsUnlock();
            return;
        }
        if(g_dcgmGlobals.clientHandlerRefCount > 0)
        {
            /* Someone else got the lock and incremented the ref count while we were sleeping. Start over */
            dcgmGlobalsUnlock();
            PRINT_INFO("", "Another thread acquired the client handler while we were sleeping.");
            continue;
        }

        delete g_dcgmGlobals.clientHandler;
        g_dcgmGlobals.clientHandler = NULL;
        dcgmGlobalsUnlock();
        PRINT_INFO("","Freed the client handler");
        break;
    }
}

/*****************************************************************************/
dcgmReturn_t processAtRemoteHostEngine(dcgmHandle_t pDcgmHandle, LwcmProtobuf *pEncodePrb,
        LwcmProtobuf *pDecodePrb, vector<lwcm::Command *> *pVecCmds, LwcmRequest *request,
        unsigned int timeout = 60000)
{
    dcgmReturn_t ret;

    /* Check for Host Engine handle. This check is only needed in standalone
       case */
    if (NULL == pDcgmHandle)
    {
        if (request)
        {
            delete request;
        }

        PRINT_ERROR("Invalid handle", "Invalid DCGM handle passed to processAtRemoteHostEngine. Handle = 0");
        return DCGM_ST_BADPARAM;
    }

    LwcmClientHandler *clientHandler = lwcmapiAcquireClientHandler(true);
    if(!clientHandler)
    {
        PRINT_ERROR("", "Unable to acqire the client handler");
        if(request)
            delete request;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Ilwoke method on the client side */
    if(!request)
    {
        ret = clientHandler->ExchangeMsgBlocking(pDcgmHandle, pEncodePrb, pDecodePrb, pVecCmds, timeout);
    }
    else
    {
        dcgm_request_id_t requestId = 0;
        ret = clientHandler->ExchangeMsgAsync(pDcgmHandle, pEncodePrb, pDecodePrb, pVecCmds, request, &requestId);
    }

    lwcmapiReleaseClientHandler();

    return ret;
}

/*****************************************************************************/
dcgmReturn_t processAtEmbeddedHostEngine(LwcmProtobuf *pEncodePrb, vector<lwcm::Command *> *pVecCmds, LwcmRequest *request) 
{
    LwcmHostEngineHandler *pHEHandlerInstance = NULL;
    bool isComplete;
    dcgmReturn_t ret;
    dcgm_request_id_t requestId = 0;

    /* Get Instance to Host Engine Handler */
    pHEHandlerInstance = LwcmHostEngineHandler::Instance();
    if (NULL == pHEHandlerInstance) 
    {
        PRINT_ERROR("", "LwcmHostEngineHandler::Instance() returned NULL");
        if(request)
            delete request;
        return DCGM_ST_UNINITIALIZED;
    }

    /* Get Vector of commands from the protobuf messages */
    if (0 != pEncodePrb->GetAllCommands(pVecCmds)) 
    {
        /* This should never happen */
        PRINT_ERROR("", "GetAllCommands failed");
        if(request)
            delete request;
        return DCGM_ST_GENERIC_ERROR;
    }

    if(request)
    {
        /* Subscribe to be updated when this request updates. This will also assign request->requestId */
        ret = pHEHandlerInstance->AddRequestWatcher(request, &requestId);
        if(ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "AddRequestWatcher returned %d", ret);
            return ret;
        }
    }

    /* Ilwoke Request handler method on the host engine */
    ret = (dcgmReturn_t) pHEHandlerInstance->HandleCommands(pVecCmds, &isComplete, NULL, requestId);
    return ret;
}

/*****************************************************************************/
dcgmReturn_t processAtHostEngine(dcgmHandle_t pDcgmHandle, LwcmProtobuf *encodePrb,
                                 LwcmProtobuf *decodePrb, vector<lwcm::Command *> *vecCmdsRef,
                                 LwcmRequest *request=0, unsigned int timeout=60000)
{
    if (pDcgmHandle != (dcgmHandle_t)DCGM_EMBEDDED_HANDLE) /* Remote DCGM */
    {
        return processAtRemoteHostEngine(pDcgmHandle, encodePrb, decodePrb, 
                                         vecCmdsRef, request, timeout);
    }
    else    /* Implies Embedded HE mode. ISV Case */
    {
        return processAtEmbeddedHostEngine(encodePrb, vecCmdsRef, request);
    }
}

/*****************************************************************************/
dcgmReturn_t helperGroupCreate(dcgmHandle_t pLwcmHandle, dcgmGroupType_t type, 
        const char *groupName, dcgmGpuGrp_t *pLwcmGrpId)
{
    lwcm::GroupInfo *pGroupInfo;         /* Protobuf equivalent structure of the output parameter. */
    lwcm::GroupInfo *pGroupInfoOut;      /* Protobuf equivalent structure of the output parameter. */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    unsigned int length = strlen(groupName);

    if ((groupName == NULL) || (length <= 0) || (pLwcmGrpId == NULL))
    {
        return DCGM_ST_BADPARAM;
    }

    /* Update the desired group type and group name*/
    pGroupInfo = new lwcm::GroupInfo;
    pGroupInfo->set_grouptype(type);
    pGroupInfo->set_groupname(groupName, length);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::GROUP_CREATE, lwcm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp) {
        delete pGroupInfo;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is 
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_grpinfo(pGroupInfo);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    /* Make sure that the returned message has the required structures */
    if (!(vecCmdsRef[0]->arg_size() && vecCmdsRef[0]->arg(0).has_grpinfo()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        return DCGM_ST_GENERIC_ERROR;
    }    

    /* Update the Protobuf reference with the results */
    pGroupInfoOut = vecCmdsRef[0]->mutable_arg(0)->mutable_grpinfo();

    if (pGroupInfoOut->has_groupid()) {
        *pLwcmGrpId = (dcgmGpuGrp_t)(long long)pGroupInfoOut->groupid();
    } else {
        PRINT_ERROR("", "Failed to create group");
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t helperGroupDestroy(dcgmHandle_t pLwcmHandle, dcgmGpuGrp_t grpId)
{
    lwcm::GroupInfo *pGroupInfo;        /* Protobuf equivalent structure of the output parameter. */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    /* Set group ID to be removed from the hostengine */
    pGroupInfo = new lwcm::GroupInfo;
    pGroupInfo->set_groupid((intptr_t)grpId);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::GROUP_DESTROY, lwcm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp) {
        delete pGroupInfo;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is 
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_grpinfo(pGroupInfo);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t helperGroupAddEntity(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, 
                                dcgm_field_entity_group_t entityGroupId, 
                                dcgm_field_eid_t entityId)
{
    lwcm::GroupInfo *pGroupInfo;        /* Protobuf equivalent structure of the output parameter. */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    
    pGroupInfo = new lwcm::GroupInfo;
    pGroupInfo->set_groupid((intptr_t)groupId);

    lwcm::EntityIdPair *eidPair = pGroupInfo->add_entity();
    eidPair->set_entitygroupid((unsigned int)entityGroupId);
    eidPair->set_entityid((unsigned int)entityId);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::GROUP_ADD_DEVICE, lwcm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp) {
        delete pGroupInfo;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is 
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_grpinfo(pGroupInfo);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;    
}

/*****************************************************************************/
dcgmReturn_t helperGroupAddDevice(dcgmHandle_t pLwcmHandle, dcgmGpuGrp_t grpId, unsigned int gpuId)
{
    return helperGroupAddEntity(pLwcmHandle, grpId, DCGM_FE_GPU, gpuId);
}

/*****************************************************************************/
dcgmReturn_t tsapiGroupAddEntity(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, 
                                 dcgm_field_entity_group_t entityGroupId,
                                 dcgm_field_eid_t entityId)
{
    return helperGroupAddEntity(pDcgmHandle, groupId, entityGroupId, entityId);

}

/*****************************************************************************/
dcgmReturn_t helperGroupRemoveEntity(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, 
                                     dcgm_field_entity_group_t entityGroupId, 
                                     dcgm_field_eid_t entityId)
{
    lwcm::GroupInfo *pGroupInfo;        /* Protobuf equivalent structure of the output parameter. */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    /* Set group ID to be removed from the hostengine */
    pGroupInfo = new lwcm::GroupInfo;
    pGroupInfo->set_groupid((intptr_t)groupId);

    lwcm::EntityIdPair *eidPair = pGroupInfo->add_entity();
    eidPair->set_entitygroupid((unsigned int)entityGroupId);
    eidPair->set_entityid((unsigned int)entityId);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::GROUP_REMOVE_DEVICE, lwcm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp) {
        delete pGroupInfo;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is 
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_grpinfo(pGroupInfo);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;    
}

/*****************************************************************************/
dcgmReturn_t helperGroupRemoveDevice(dcgmHandle_t pLwcmHandle, dcgmGpuGrp_t grpId, unsigned int gpuId)
{
    return helperGroupRemoveEntity(pLwcmHandle, grpId, DCGM_FE_GPU, gpuId);
}

/*****************************************************************************/
dcgmReturn_t tsapiGroupRemoveEntity(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, 
                                    dcgm_field_entity_group_t entityGroupId,
                                    dcgm_field_eid_t entityId)
{
    return helperGroupRemoveEntity(pDcgmHandle, groupId, entityGroupId, entityId);
}

/*****************************************************************************/
dcgmReturn_t helperGroupGetAllIds(dcgmHandle_t pLwcmHandle, dcgmGpuGrp_t *pGroupIdList, unsigned int *pCount)
{
    lwcm::FieldMultiValues *pListGrpIdsOutput;
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;    

    if ((NULL == pGroupIdList) || (NULL == pCount)) {
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::GROUP_GETALL_IDS, lwcm::OPERATION_SYSTEM, -1, 0);
    if (NULL == pCmdTemp) {
        return DCGM_ST_GENERIC_ERROR;
    }

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    /* Make sure that the returned message has the required structures */
    if (!(vecCmdsRef[0]->arg_size() && vecCmdsRef[0]->arg(0).has_fieldmultivalues()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        return DCGM_ST_GENERIC_ERROR;
    }    

    /* Update the Protobuf reference with the results */
    pListGrpIdsOutput = vecCmdsRef[0]->mutable_arg(0)->mutable_fieldmultivalues();

    *pCount = pListGrpIdsOutput->vals_size();
    for (int index = 0; index < pListGrpIdsOutput->vals_size(); index++) {
        pGroupIdList[index] = (dcgmGpuGrp_t)pListGrpIdsOutput->mutable_vals(index)->i64();
    }

    return DCGM_ST_OK;    
}



/*****************************************************************************/
dcgmReturn_t helperGroupGetInfo(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId,
        dcgmGroupInfo_t *pDcgmGroupInfo, long long *hostEngineTimestamp)
{
    lwcm::GroupInfo *pGroupInfo;         /* Protobuf equivalent structure of the output parameter. */
    lwcm::GroupInfo *pGroupInfoOut;      /* Protobuf equivalent structure of the output parameter. */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    /* Input parameter validation */
    if (NULL == pDcgmGroupInfo) 
    {
        PRINT_ERROR("", "NULL pDcgmGroupInfo");
        return DCGM_ST_BADPARAM;
    }

    /* Check for version */
    if ((pDcgmGroupInfo->version < dcgmGroupInfo_version2) || (pDcgmGroupInfo->version > dcgmGroupInfo_version))
    {
        PRINT_ERROR("%X", "helperGroupGetInfo version mismatch on x%X", pDcgmGroupInfo->version);
        return DCGM_ST_VER_MISMATCH;
    }

    pGroupInfo = new lwcm::GroupInfo;
    pGroupInfo->set_groupid((intptr_t)groupId);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::GROUP_INFO, lwcm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp) {
        delete pGroupInfo;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is 
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_grpinfo(pGroupInfo);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }


    /* Make sure that the returned message has the required structures */
    if (!(vecCmdsRef[0]->arg_size() && vecCmdsRef[0]->arg(0).has_grpinfo()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        return DCGM_ST_GENERIC_ERROR;
    }    

    /* Update the Protobuf reference with the results */
    pGroupInfoOut = vecCmdsRef[0]->mutable_arg(0)->mutable_grpinfo();


    if (pGroupInfoOut->has_groupname()) 
    {
        size_t length;
        length = strlen(pGroupInfoOut->groupname().c_str());
        if (length + 1 > DCGM_MAX_STR_LENGTH) {
            PRINT_ERROR("", "String overflow error for the requested field");
            return DCGM_ST_MEMORY;
        }

        strncpy(pDcgmGroupInfo->groupName, pGroupInfoOut->groupname().c_str(), length + 1);
    } 
    else 
    {
        PRINT_ERROR("", "Can't find group name in the returned info from the hostengine");
        return DCGM_ST_GENERIC_ERROR;
    }

    if (pGroupInfoOut->entity_size() > DCGM_GROUP_MAX_ENTITIES) {
        PRINT_ERROR("", "Invalid number of GPU Ids returned from the hostengine");
        return DCGM_ST_GENERIC_ERROR;
    }

    if(hostEngineTimestamp)
    {
        if(vecCmdsRef[0]->has_timestamp())
        {
            *hostEngineTimestamp = vecCmdsRef[0]->timestamp();
        }
        else
        {
            PRINT_ERROR("", "No timestamp in command. Caller requires one.");
            return DCGM_ST_GENERIC_ERROR;
        }
    }

    pDcgmGroupInfo->count = pGroupInfoOut->entity_size();

    for (int index = 0; index < pGroupInfoOut->entity_size(); index++) 
    {
        const lwcm::EntityIdPair eidPair = pGroupInfoOut->entity(index);

        pDcgmGroupInfo->entityList[index].entityGroupId = (dcgm_field_entity_group_t)eidPair.entitygroupid();
        pDcgmGroupInfo->entityList[index].entityId = (dcgm_field_eid_t)eidPair.entityid();
    }

    return DCGM_ST_OK;
}

/*****************************************************************************
 * This method is a common helper to get value for multiple fields 
 * 
 * dcgmHandle       IN: Handle to the host engine
 * groupId          IN: Optional groupId that will be resolved by the host engine. 
 *                      This is ignored if entityList is provided.
 * entityList       IN: List of entities to retrieve values for. This value takes
 *                      precedence over groupId
 * entityListCount  IN: How many entries are contained in entityList[]
 * fieldGroupId     IN: Optional fieldGroupId that will be resolved by the host engine.
 *                      This is ignored if fieldIdList[] is provided
 * fieldIdList      IN: List of field IDs to retrieve values for. This value takes
 *                      precedence over fieldGroupId
 * fieldIdListCount IN: How many entries are contained in fieldIdList
 * fvBuffer        OUT: Field value buffer to save values into
 * flags            IN: Mask of DCGM_GMLV_FLAG_? flags that modify this request
 * 
 * 
 * @return DCGM_ST_OK on success
 *         Other DCGM_ST_? status code on error
 *
 *****************************************************************************/
dcgmReturn_t helperGetLatestValuesForFields(dcgmHandle_t dcgmHandle,
                                            dcgmGpuGrp_t groupId,
                                            dcgmGroupEntityPair_t *entityList,
                                            unsigned int entityListCount,
                                            dcgmFieldGrp_t fieldGroupId,
                                            unsigned short fieldIdList[],
                                            unsigned int fieldIdListCount,
                                            DcgmFvBuffer *fvBuffer,
                                            unsigned int flags)
{

    LwcmProtobuf encodePrb;                 /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;                 /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;                /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;     /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;    
    unsigned int i;
    size_t returnedByteSize = 0, returnedValuesCount = 0;
    dcgm_field_meta_p fieldMeta = 0;
    dcgmGetMultipleLatestValues_t msg;

    if ((entityList && !entityListCount) || (fieldIdList && !fieldIdListCount) || !fvBuffer ||
        entityListCount > DCGM_GROUP_MAX_ENTITIES || 
        fieldIdListCount > DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP)
    {
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to be sent over the network */
    pCmdTemp = encodePrb.AddCommand(lwcm::GET_MULTIPLE_LATEST_VALUES, lwcm::OPERATION_SYSTEM, 0, 0);
    if (NULL == pCmdTemp) 
    {
        PRINT_ERROR("", "encodePrb.AddCommand failed.");
        return DCGM_ST_GENERIC_ERROR;
    }

    lwcm::CmdArg *pCmdArg = pCmdTemp->add_arg();

    memset(&msg, 0, sizeof(msg));
    msg.version = dcgmGetMultipleLatestValues_version;
    msg.flags = flags;

    if(entityList)
    {
        memmove(&msg.entities[0], entityList, entityListCount * sizeof(entityList[0]));
        msg.entitiesCount = entityListCount;
    }
    else
    {
        msg.groupId = groupId;
    }

    if(fieldIdList)
    {
        memmove(&msg.fieldIds[0], fieldIdList, fieldIdListCount * sizeof(fieldIdList[0]));
        msg.fieldIdCount = fieldIdListCount;
    }
    else
    {
        msg.fieldGroupId = fieldGroupId;
    }

    pCmdArg->set_blob(&msg, sizeof(msg));

    ret = processAtHostEngine(dcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) 
    {
        PRINT_ERROR("%d", "processAtHostEngine returned %d", (int)ret);
        return ret;
    }

    if(vecCmdsRef.size() < 1 || vecCmdsRef[0]->arg_size() < 1)
    {
        PRINT_ERROR("", "Malformed GET_MULTIPLE_LATEST_VALUES response 1.");
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Did the request return a global request error (vs a field value status)? */
    if(vecCmdsRef[0]->has_status() && vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got message status %d", (int)vecCmdsRef[0]->status());
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    if(!vecCmdsRef[0]->arg(0).has_blob())
    {
        PRINT_ERROR("", "Malformed GET_MULTIPLE_LATEST_VALUES missing blob.");
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Make a FV buffer from our protobuf string */
    fvBuffer->SetFromBuffer(vecCmdsRef[0]->arg(0).blob().c_str(), vecCmdsRef[0]->arg(0).blob().size());
    return DCGM_ST_OK;
}

/****************************************************************************/
dcgmReturn_t tsapiEntitiesGetLatestValues(dcgmHandle_t dcgmHandle, dcgmGroupEntityPair_t entities[], 
                                          unsigned int entityCount, unsigned short fields[],
                                          unsigned int fieldCount, unsigned int flags, 
                                          dcgmFieldValue_v2 values[])
{
    dcgmReturn_t dcgmReturn;

    if(!entities || entityCount < 1 || !fields || fieldCount < 1 || !values)
    {
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    DcgmFvBuffer fvBuffer(0);

    dcgmReturn = helperGetLatestValuesForFields(dcgmHandle, 0, entities, entityCount,
                                                0, fields, fieldCount, &fvBuffer, flags);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;
    
    size_t bufferSize = 0, elementCount = 0;
    dcgmReturn = fvBuffer.GetSize(&bufferSize, &elementCount);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;
    
    /* Check that we got as many fields back as we requested */
    if(elementCount != fieldCount * entityCount)
    {
        PRINT_ERROR("%u %u", "Returned FV mismatch. Requested %u != returned %u", 
                    entityCount * fieldCount, (unsigned int)elementCount);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Colwert the buffered FVs to our output array */
    dcgmBufferedFvLwrsor_t cursor = 0;
    unsigned int valuesIndex = 0;
    for(dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
    {
        fvBuffer.ColwertBufferedFvToFv2(fv, &values[valuesIndex]);
        valuesIndex++;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************
 * Common helper method for standalone and embedded case to fetch LWCM GPU Ids from 
 * the system
 * @param mode          IN  :   Should be one of DCGM_MODE_?
 * @param pLwcmHandle   IN  :   HE Handle for Standalone case. NULL for Embedded case
 * @param pGpuIdList    OUT :   List of LWCM GPU Ids
 * @param pCount        OUT :   Number of GPUs in the list
 * @param onlySupported IN  :   Whether or not to only return devices that are supported
 *                              by DCGM. 1=only return DCGM-supported devices.
 *                                       0=return all devices in the system
 * @return
 *****************************************************************************/
dcgmReturn_t helperGetAllDevices(dcgmHandle_t pLwcmHandle, unsigned int *pGpuIdList, int *pCount,
                                 int onlySupported)
{
    lwcm::FieldMultiValues *pListGpuIdsOutput;
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    lwcm::CmdArg *cmdArg = 0;

    if ((NULL == pGpuIdList) || (NULL == pCount)) {
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::DISCOVER_DEVICES, lwcm::OPERATION_SYSTEM, -1, 0);
    if (NULL == pCmdTemp) {
        return DCGM_ST_GENERIC_ERROR;
    }


    cmdArg = pCmdTemp->add_arg();
    /* Use the int32 parameter to pass "onlySupported" */
    cmdArg->set_i32(onlySupported);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    /* Make sure that the returned message has the required structures */
    if (!(vecCmdsRef[0]->arg_size() && vecCmdsRef[0]->arg(0).has_fieldmultivalues()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        return DCGM_ST_GENERIC_ERROR;
    }    

    /* Update the Protobuf reference with the results */
    pListGpuIdsOutput = vecCmdsRef[0]->mutable_arg(0)->mutable_fieldmultivalues();

    *pCount = pListGpuIdsOutput->vals_size();
    for (int index = 0; index < pListGpuIdsOutput->vals_size(); index++) {
        pGpuIdList[index] = pListGpuIdsOutput->mutable_vals(index)->i64();
    }

    return DCGM_ST_OK;
}

/**
 * Common helper to get device attributes
 * @param mode
 * @param pLwcmHandle
 * @param gpuId
 * @param pLwcmDeviceAttr
 * @return 
 */
dcgmReturn_t helperDeviceGetAttributes(dcgmHandle_t pLwcmHandle, int gpuId, dcgmDeviceAttributes_t *pLwcmDeviceAttr)
{
    unsigned short fieldIds[32];
    dcgmBufferedFv_t *fv;
    unsigned int count = 0, i;
    dcgmReturn_t ret;

    if (NULL == pLwcmDeviceAttr) {
        return DCGM_ST_BADPARAM;
    }

    if ((pLwcmDeviceAttr->version < dcgmDeviceAttributes_version1) || (pLwcmDeviceAttr->version > dcgmDeviceAttributes_version))
    {
        return DCGM_ST_VER_MISMATCH;
    }


    fieldIds[count++] = DCGM_FI_DEV_SLOWDOWN_TEMP;
    fieldIds[count++] = DCGM_FI_DEV_SHUTDOWN_TEMP;
    fieldIds[count++] = DCGM_FI_DEV_ENFORCED_POWER_LIMIT;
    fieldIds[count++] = DCGM_FI_DEV_POWER_MGMT_LIMIT;
    fieldIds[count++] = DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF;
    fieldIds[count++] = DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX;
    fieldIds[count++] = DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN;
    fieldIds[count++] = DCGM_FI_DEV_SUPPORTED_CLOCKS;
    fieldIds[count++] = DCGM_FI_DEV_UUID;
    fieldIds[count++] = DCGM_FI_DEV_VBIOS_VERSION;
    fieldIds[count++] = DCGM_FI_DEV_INFOROM_IMAGE_VER;
    fieldIds[count++] = DCGM_FI_DEV_BRAND;
    fieldIds[count++] = DCGM_FI_DEV_NAME;
    fieldIds[count++] = DCGM_FI_DEV_SERIAL;
    fieldIds[count++] = DCGM_FI_DEV_PCI_BUSID;
    fieldIds[count++] = DCGM_FI_DEV_PCI_COMBINED_ID;
    fieldIds[count++] = DCGM_FI_DEV_PCI_SUBSYS_ID;
    fieldIds[count++] = DCGM_FI_DEV_BAR1_TOTAL;
    fieldIds[count++] = DCGM_FI_DEV_FB_TOTAL;
    fieldIds[count++] = DCGM_FI_DEV_FB_USED;
    fieldIds[count++] = DCGM_FI_DEV_FB_FREE;
    fieldIds[count++] = DCGM_FI_DRIVER_VERSION;
    fieldIds[count++] = DCGM_FI_DEV_VIRTUAL_MODE;

    if (count >= 32) {
        PRINT_ERROR("", "Update DeviceGetAttributes to accommodate more fields\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = DCGM_FE_GPU;
    entityPair.entityId = gpuId;
    DcgmFvBuffer fvBuffer(0);
    ret = helperGetLatestValuesForFields(pLwcmHandle, 0, &entityPair, 1, 0, fieldIds, count, &fvBuffer,
                                         DCGM_FV_FLAG_LIVE_DATA);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    size_t bufferSize = 0, elementCount = 0;
    ret = fvBuffer.GetSize(&bufferSize, &elementCount);
    if(elementCount != count)
    {
        PRINT_ERROR("%d %d %d", "Unexpected elementCount %d != count %d or ret %d", 
                    (int)elementCount, count, (int)ret);
        /* Keep going. We will only process what we have */
    }

    dcgmBufferedFvLwrsor_t cursor = 0;
    for (fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor)) {
        switch (fv->fieldId) {
            case DCGM_FI_DEV_SLOWDOWN_TEMP:
                pLwcmDeviceAttr->thermalSettings.slowdownTemp = (unsigned int)lwcmvalue_int64_to_int32(fv->value.i64);
                break;

            case DCGM_FI_DEV_SHUTDOWN_TEMP:
                pLwcmDeviceAttr->thermalSettings.shutdownTemp = (unsigned int)lwcmvalue_int64_to_int32(fv->value.i64);
                break;

            case DCGM_FI_DEV_POWER_MGMT_LIMIT:
                pLwcmDeviceAttr->powerLimits.lwrPowerLimit = (unsigned int)lwcmvalue_double_to_int32(fv->value.dbl);
                break;

            case DCGM_FI_DEV_ENFORCED_POWER_LIMIT:
                pLwcmDeviceAttr->powerLimits.enforcedPowerLimit = (unsigned int)lwcmvalue_double_to_int32(fv->value.dbl);
                break;

            case DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF:
                pLwcmDeviceAttr->powerLimits.defaultPowerLimit = (unsigned int)lwcmvalue_double_to_int32(fv->value.dbl);

                break;

            case DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX:
                pLwcmDeviceAttr->powerLimits.maxPowerLimit = (unsigned int)lwcmvalue_double_to_int32(fv->value.dbl);
                break;

            case DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN:
                pLwcmDeviceAttr->powerLimits.minPowerLimit = (unsigned int)lwcmvalue_double_to_int32(fv->value.dbl);
                break;

            case DCGM_FI_DEV_UUID:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pLwcmDeviceAttr->identifiers.uuid)) {
                    PRINT_ERROR("", "String overflow error for the requested UUID field");
                    strncpy(pLwcmDeviceAttr->identifiers.uuid, DCGM_STR_BLANK, strlen(DCGM_STR_BLANK) + 1);
                } else {
                    strncpy(pLwcmDeviceAttr->identifiers.uuid, fv->value.str, length + 1);
                }

                break;
            }

            case DCGM_FI_DEV_VBIOS_VERSION:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pLwcmDeviceAttr->identifiers.vbios)) {
                    PRINT_ERROR("", "String overflow error for the requested VBIOS field");
                    strncpy(pLwcmDeviceAttr->identifiers.vbios, DCGM_STR_BLANK, strlen(DCGM_STR_BLANK) + 1);
                } else {
                    strncpy(pLwcmDeviceAttr->identifiers.vbios, fv->value.str, length + 1);
                }

                break;
            }

            case DCGM_FI_DEV_INFOROM_IMAGE_VER:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pLwcmDeviceAttr->identifiers.inforomImageVersion)) {
                    PRINT_ERROR("", "String overflow error for the requested Inforom field");
                    strncpy(pLwcmDeviceAttr->identifiers.inforomImageVersion, DCGM_STR_BLANK, strlen(DCGM_STR_BLANK) + 1);
                } else {
                    strncpy(pLwcmDeviceAttr->identifiers.inforomImageVersion, fv->value.str, length + 1);
                }

                break;
            }                

            case DCGM_FI_DEV_BRAND:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pLwcmDeviceAttr->identifiers.brandName)) {
                    PRINT_ERROR("", "String overflow error for the requested brand name field");
                    strncpy(pLwcmDeviceAttr->identifiers.brandName, DCGM_STR_BLANK, strlen(DCGM_STR_BLANK) + 1);
                } else {
                    strncpy(pLwcmDeviceAttr->identifiers.brandName, fv->value.str, length + 1);
                }

                break;
            }                

            case DCGM_FI_DEV_NAME:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pLwcmDeviceAttr->identifiers.deviceName)) {
                    PRINT_ERROR("", "String overflow error for the requested device name field");
                    strncpy(pLwcmDeviceAttr->identifiers.deviceName, DCGM_STR_BLANK, strlen(DCGM_STR_BLANK) + 1);
                } else {
                    strncpy(pLwcmDeviceAttr->identifiers.deviceName, fv->value.str, length + 1);
                }

                break;
            }                

            case DCGM_FI_DEV_SERIAL:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pLwcmDeviceAttr->identifiers.serial)) {
                    PRINT_ERROR("", "String overflow error for the requested serial field");
                    strncpy(pLwcmDeviceAttr->identifiers.serial, DCGM_STR_BLANK, strlen(DCGM_STR_BLANK) + 1);
                } else {
                    strncpy(pLwcmDeviceAttr->identifiers.serial, fv->value.str, length + 1);
                }

                break;
            }                

            case DCGM_FI_DEV_PCI_BUSID:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pLwcmDeviceAttr->identifiers.pciBusId)) {
                    PRINT_ERROR("", "String overflow error for the requested serial field");
                    strncpy(pLwcmDeviceAttr->identifiers.pciBusId, DCGM_STR_BLANK, strlen(DCGM_STR_BLANK) + 1);
                } else {
                    strncpy(pLwcmDeviceAttr->identifiers.pciBusId, fv->value.str, length + 1);
                }

                break;
            }

            case DCGM_FI_DEV_SUPPORTED_CLOCKS:
            {
                dcgmDeviceSupportedClockSets_t *supClocks = (dcgmDeviceSupportedClockSets_t *)fv->value.blob;

                if(!supClocks)
                {
                    memset(&pLwcmDeviceAttr->clockSets, 0, sizeof(pLwcmDeviceAttr->clockSets));
                    PRINT_ERROR("", "Null field value for DCGM_FI_DEV_SUPPORTED_CLOCKS");
                }
                else if(supClocks->version != dcgmDeviceSupportedClockSets_version)
                {
                    memset(&pLwcmDeviceAttr->clockSets, 0, sizeof(pLwcmDeviceAttr->clockSets));
                    PRINT_ERROR("%d %d", "Expected dcgmDeviceSupportedClockSets_version %d. Got %d",
                            (int)dcgmDeviceSupportedClockSets_version, (int)supClocks->version);
                }
                else
                {
                    int payloadSize = (sizeof(*supClocks) - sizeof(supClocks->clockSet)) +
                                      (supClocks->count * sizeof(supClocks->clockSet[0]));
                    if(payloadSize > (int)(fv->length - (sizeof(*fv) - sizeof(fv->value))))
                    {
                        PRINT_ERROR("%d %d", "DCGM_FI_DEV_SUPPORTED_CLOCKS callwlated size %d > possible size %d",
                                    payloadSize, (int)(fv->length - (sizeof(*fv) - sizeof(fv->value))));
                        memset(&pLwcmDeviceAttr->clockSets, 0, sizeof(pLwcmDeviceAttr->clockSets));
                    }
                    else
                    {
                        /* Success */
                        memcpy(&pLwcmDeviceAttr->clockSets, supClocks, payloadSize);
                    }
                }
                break;
            }

            case DCGM_FI_DEV_PCI_COMBINED_ID:
                pLwcmDeviceAttr->identifiers.pciDeviceId = fv->value.i64;
                break;

            case DCGM_FI_DEV_PCI_SUBSYS_ID:
                pLwcmDeviceAttr->identifiers.pciSubSystemId = fv->value.i64;
                break;

            case DCGM_FI_DEV_BAR1_TOTAL:
                pLwcmDeviceAttr->memoryUsage.bar1Total = fv->value.i64;
                break;

            case DCGM_FI_DEV_FB_TOTAL:
                pLwcmDeviceAttr->memoryUsage.fbTotal = fv->value.i64;
                break;

            case DCGM_FI_DEV_FB_USED:
                pLwcmDeviceAttr->memoryUsage.fbUsed = fv->value.i64;
                break;

            case DCGM_FI_DEV_FB_FREE:
                pLwcmDeviceAttr->memoryUsage.fbFree = fv->value.i64;
                break;

            case DCGM_FI_DRIVER_VERSION:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pLwcmDeviceAttr->identifiers.driverVersion)) {
                    PRINT_ERROR("", "String overflow error for the requested driver version field");
                    strncpy(pLwcmDeviceAttr->identifiers.driverVersion, DCGM_STR_BLANK, strlen(DCGM_STR_BLANK) + 1);
                } else {
                    strncpy(pLwcmDeviceAttr->identifiers.driverVersion, fv->value.str, length + 1);
                }

                break;
            }

            case DCGM_FI_DEV_VIRTUAL_MODE:
                pLwcmDeviceAttr->identifiers.virtualizationMode = (unsigned int)lwcmvalue_int64_to_int32(fv->value.i64);
                break;

            default:
                /* This should never happen */
                return DCGM_ST_GENERIC_ERROR;
                break;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t helperWatchFieldValue(dcgmHandle_t pLwcmHandle,
        int gpuId, unsigned short fieldId,
        long long updateFreq, double maxKeepAge,
        int maxKeepSamples)
{
    lwcm::WatchFieldValue *pProtoWatchFieldValue;   /* Protobuf Arg */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if (!fieldId || updateFreq <= 0 || (maxKeepSamples <= 0 && maxKeepAge <= 0.0))
        return DCGM_ST_BADPARAM;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById((unsigned short)fieldId);
    if (NULL == fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        PRINT_ERROR("%u", "field ID %u is not a valid field ID", fieldId);
        return DCGM_ST_BADPARAM;
    }

    pProtoWatchFieldValue = new lwcm::WatchFieldValue;
    pProtoWatchFieldValue->set_version(dcgmWatchFieldValue_version);
    pProtoWatchFieldValue->set_fieldid(fieldId);
    pProtoWatchFieldValue->set_updatefreq(updateFreq);
    pProtoWatchFieldValue->set_maxkeepage(maxKeepAge);
    pProtoWatchFieldValue->set_maxkeepsamples(maxKeepSamples);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::WATCH_FIELD_VALUE, lwcm::OPERATION_SINGLE_ENTITY, gpuId, 0);
    if (NULL == pCmdTemp) {
        delete pProtoWatchFieldValue;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the entityGroupId */
    dcgm_field_entity_group_t entityGroupId = DCGM_FE_GPU;
    if(fieldMeta->scope == DCGM_FS_GLOBAL)
        entityGroupId = DCGM_FE_NONE;
    pCmdTemp->set_entitygroupid((int)entityGroupId);


    /* Set the arg to passed as a proto message. After this point the arg is 
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_watchfieldvalue(pProtoWatchFieldValue);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;        
}

/*****************************************************************************/
dcgmReturn_t helperUpdateAllFields(dcgmHandle_t pLwcmHandle, int waitForUpdate)
{
    lwcm::UpdateAllFields *pProtoUpdateAllFields; /* Protobuf Arg */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    pProtoUpdateAllFields = new lwcm::UpdateAllFields;
    pProtoUpdateAllFields->set_version(dcgmUpdateAllFields_version);
    pProtoUpdateAllFields->set_waitforupdate(waitForUpdate);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::UPDATE_ALL_FIELDS, lwcm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp) {
        delete pProtoUpdateAllFields;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is 
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_updateallfields(pProtoUpdateAllFields);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;    
}

/**
 * Common helper to get vGPU device attributes
 * @param mode
 * @param pLwcmHandle
 * @param gpuId
 * @param pLwcmVgpuDeviceAttr
 * @return
 */
dcgmReturn_t helperVgpuDeviceGetAttributes(dcgmHandle_t pLwcmHandle, int gpuId, dcgmVgpuDeviceAttributes_t *pLwcmVgpuDeviceAttr)
{
    unsigned short fieldIds[32];
    unsigned int count = 0, i;
    dcgmReturn_t ret;
    long long updateFreq = 30000000;
    double maxKeepAge = 14400.0;
    int maxKeepSamples = 480;

    if (NULL == pLwcmVgpuDeviceAttr) {
        return DCGM_ST_BADPARAM;
    }

    if ((pLwcmVgpuDeviceAttr->version < dcgmVgpuDeviceAttributes_version6) || (pLwcmVgpuDeviceAttr->version > dcgmVgpuDeviceAttributes_version))
    {
        return DCGM_ST_VER_MISMATCH;
    }

    fieldIds[count++] = DCGM_FI_DEV_SUPPORTED_TYPE_INFO;
    fieldIds[count++] = DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_INSTANCE_IDS;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_UTILIZATIONS;
    fieldIds[count++] = DCGM_FI_DEV_GPU_UTIL;
    fieldIds[count++] = DCGM_FI_DEV_MEM_COPY_UTIL;
    fieldIds[count++] = DCGM_FI_DEV_ENC_UTIL;
    fieldIds[count++] = DCGM_FI_DEV_DEC_UTIL;

    if (count >= 32) {
        PRINT_ERROR("", "Update DeviceGetAttributes to accommodate more fields\n");
        return DCGM_ST_GENERIC_ERROR;
    }
    
    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = DCGM_FE_GPU;
    entityPair.entityId = gpuId;

    DcgmFvBuffer fvBuffer(0);
    
    ret = helperGetLatestValuesForFields(pLwcmHandle, 0, &entityPair, 1, 0, fieldIds, count, &fvBuffer, 0);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    dcgmBufferedFv_t *fv;
    dcgmBufferedFvLwrsor_t cursor = 0;

    int anyWatched = 0; /* Did we add any watches? */
    for(fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
    {
        if (fv->status == DCGM_ST_NOT_WATCHED)
        {
            ret = helperWatchFieldValue(pLwcmHandle, gpuId, fv->fieldId, updateFreq, maxKeepAge, maxKeepSamples);
            if (DCGM_ST_OK != ret) {
                return ret;
            }
            anyWatched = 1;
        }
    }

    if(anyWatched)
    {
        /* Make sure the new watches have updated */
        helperUpdateAllFields(pLwcmHandle, 1);

        /* Get all of the field values again now that everything has been watched */
        entityPair.entityGroupId = DCGM_FE_GPU;
        entityPair.entityId = gpuId;
        ret = helperGetLatestValuesForFields(pLwcmHandle, 0, &entityPair, 1, 0, fieldIds, count, &fvBuffer, 0);
        if (DCGM_ST_OK != ret) {
            return ret;
        }
    }

    size_t bufferSize = 0, elementCount = 0;
    ret = fvBuffer.GetSize(&bufferSize, &elementCount);
    if(elementCount != count)
    {
        PRINT_ERROR("%d %d %d", "Unexpected elementCount %d != count %d or ret %d", 
                    (int)elementCount, count, (int)ret);
        /* Keep going. We will only process what we have */
    }

    cursor = 0;
    for(fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor)) {
        switch (fv->fieldId) {
            case DCGM_FI_DEV_SUPPORTED_TYPE_INFO:
            {
                dcgmDeviceVgpuTypeInfo_t *vgpuTypeInfo = (dcgmDeviceVgpuTypeInfo_t *)fv->value.blob;

                if(!vgpuTypeInfo)
                {
                    memset(&pLwcmVgpuDeviceAttr->supportedVgpuTypeInfo, 0, sizeof(pLwcmVgpuDeviceAttr->supportedVgpuTypeInfo));
                    PRINT_ERROR("", "Null field value for DCGM_FI_DEV_SUPPORTED_VGPU_TYPE_IDS");
                    pLwcmVgpuDeviceAttr->supportedVgpuTypeCount = 0;
                    break;
                }
                else {
                    pLwcmVgpuDeviceAttr->supportedVgpuTypeCount = vgpuTypeInfo[0].vgpuTypeInfo.supportedVgpuTypeCount;
                }

                if(sizeof(pLwcmVgpuDeviceAttr->supportedVgpuTypeInfo) < sizeof(*vgpuTypeInfo)*(pLwcmVgpuDeviceAttr->supportedVgpuTypeCount))
                {
                    memset(&pLwcmVgpuDeviceAttr->supportedVgpuTypeInfo, 0, sizeof(pLwcmVgpuDeviceAttr->supportedVgpuTypeInfo));
                    PRINT_ERROR("%d %d", "vGPU Type ID static info array size %d too small for %d vGPU static info",
                            (int)sizeof(pLwcmVgpuDeviceAttr->supportedVgpuTypeInfo), (int)sizeof(*vgpuTypeInfo)*(pLwcmVgpuDeviceAttr->supportedVgpuTypeCount));
                }
                else {
                    memcpy(&pLwcmVgpuDeviceAttr->supportedVgpuTypeInfo, vgpuTypeInfo + 1, sizeof(*vgpuTypeInfo)*(pLwcmVgpuDeviceAttr->supportedVgpuTypeCount));
                }
                break;
            }

            case DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS:
            {
                unsigned int *temp = (unsigned int *)fv->value.blob;

                if(!temp) {
                    memset(&pLwcmVgpuDeviceAttr->creatableVgpuTypeIds, 0, sizeof(pLwcmVgpuDeviceAttr->creatableVgpuTypeIds));
                    PRINT_ERROR("", "Null field value for DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS");
                    pLwcmVgpuDeviceAttr->creatableVgpuTypeCount = 0;
                    break;
                }
                else {
                    pLwcmVgpuDeviceAttr->creatableVgpuTypeCount = temp[0];
                }

                if(sizeof(pLwcmVgpuDeviceAttr->creatableVgpuTypeIds) < sizeof(*temp)*(pLwcmVgpuDeviceAttr->creatableVgpuTypeCount))
                {
                    memset(&pLwcmVgpuDeviceAttr->creatableVgpuTypeIds, 0, sizeof(pLwcmVgpuDeviceAttr->creatableVgpuTypeIds));
                    PRINT_ERROR("%d %d", "Creatable vGPU Type IDs array size %d too small for %d Id value",
                            (int)sizeof(pLwcmVgpuDeviceAttr->creatableVgpuTypeIds), (int)sizeof(*temp)*(pLwcmVgpuDeviceAttr->creatableVgpuTypeCount));
                }
                else {
                    memcpy(&pLwcmVgpuDeviceAttr->creatableVgpuTypeIds, temp + 1, sizeof(*temp)*(pLwcmVgpuDeviceAttr->creatableVgpuTypeCount));
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_INSTANCE_IDS:
            {
                unsigned int *temp = (unsigned int *)fv->value.blob;

                if(!temp) {
                    memset(&pLwcmVgpuDeviceAttr->activeVgpuInstanceIds, 0, sizeof(pLwcmVgpuDeviceAttr->activeVgpuInstanceIds));
                    PRINT_ERROR("", "Null field value for DCGM_FI_DEV_VGPU_INSTANCE_IDS");
                    pLwcmVgpuDeviceAttr->activeVgpuInstanceCount = 0;
                    break;
                }
                else {
                    pLwcmVgpuDeviceAttr->activeVgpuInstanceCount = temp[0];
                }

                if(sizeof(pLwcmVgpuDeviceAttr->activeVgpuInstanceIds) < sizeof(*temp)*(pLwcmVgpuDeviceAttr->activeVgpuInstanceCount))
                {
                    memset(&pLwcmVgpuDeviceAttr->activeVgpuInstanceIds, 0, sizeof(pLwcmVgpuDeviceAttr->activeVgpuInstanceIds));
                    PRINT_ERROR("%d %d", "Active vGPU Instance IDs array size %d too small for %d Id value",
                            (int)sizeof(pLwcmVgpuDeviceAttr->activeVgpuInstanceIds), (int)sizeof(*temp)*(pLwcmVgpuDeviceAttr->activeVgpuInstanceCount));
                }
                else {
                    memcpy(&pLwcmVgpuDeviceAttr->activeVgpuInstanceIds, temp + 1, sizeof(*temp)*(pLwcmVgpuDeviceAttr->activeVgpuInstanceCount));
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_UTILIZATIONS:
            {
                dcgmDeviceVgpuUtilInfo_t *vgpuUtilInfo = (dcgmDeviceVgpuUtilInfo_t *)fv->value.blob;

                if(!vgpuUtilInfo) {
                    memset(&pLwcmVgpuDeviceAttr->vgpuUtilInfo, 0, sizeof(pLwcmVgpuDeviceAttr->vgpuUtilInfo));
                    PRINT_ERROR("", "Null field value for DCGM_FI_DEV_VGPU_UTILIZATIONS");
                    break;
                }

                if(sizeof(pLwcmVgpuDeviceAttr->vgpuUtilInfo) < sizeof(*vgpuUtilInfo)*(pLwcmVgpuDeviceAttr->activeVgpuInstanceCount))
                {
                    memset(&pLwcmVgpuDeviceAttr->vgpuUtilInfo, 0, sizeof(pLwcmVgpuDeviceAttr->vgpuUtilInfo));
                    PRINT_ERROR("%d %d", "Active vGPU Instance IDs utilizations array size %d too small for %d Id value",
                                (int)sizeof(pLwcmVgpuDeviceAttr->vgpuUtilInfo), (int)sizeof(*vgpuUtilInfo)*(pLwcmVgpuDeviceAttr->activeVgpuInstanceCount));
                }
                else {
                    memcpy(&pLwcmVgpuDeviceAttr->vgpuUtilInfo, vgpuUtilInfo, sizeof(*vgpuUtilInfo)*(pLwcmVgpuDeviceAttr->activeVgpuInstanceCount));
                }
                break;
            }

            case DCGM_FI_DEV_GPU_UTIL:
            {
                pLwcmVgpuDeviceAttr->gpuUtil = fv->value.i64;
                break;
            }

            case DCGM_FI_DEV_MEM_COPY_UTIL:
            {
                pLwcmVgpuDeviceAttr->memCopyUtil = fv->value.i64;
                break;
            }

            case DCGM_FI_DEV_ENC_UTIL:
            {
                pLwcmVgpuDeviceAttr->enlwtil = fv->value.i64;
                break;
            }

            case DCGM_FI_DEV_DEC_UTIL:
            {
                pLwcmVgpuDeviceAttr->delwtil = fv->value.i64;
                break;
            }

            default:
                /* This should never happen */
                return DCGM_ST_GENERIC_ERROR;
                break;
        }
    }

    return DCGM_ST_OK;
}

/**
 * Common helper to get attributes specific to vGPU instance
 * @param mode
 * @param pLwcmHandle
 * @param vgpuId
 * @param pLwcmVgpuInstanceAttr
 * @return
 */
dcgmReturn_t helperVgpuInstanceGetAttributes(dcgmHandle_t pLwcmHandle, int vgpuId, dcgmVgpuInstanceAttributes_t *pLwcmVgpuInstanceAttr)
{
    unsigned short fieldIds[32];
    unsigned int count = 0, i;
    dcgmReturn_t ret;

    if (NULL == pLwcmVgpuInstanceAttr) {
        return DCGM_ST_BADPARAM;
    }

    if ((pLwcmVgpuInstanceAttr->version < dcgmVgpuInstanceAttributes_version1) || (pLwcmVgpuInstanceAttr->version > dcgmVgpuInstanceAttributes_version))
    {
        return DCGM_ST_VER_MISMATCH;
    }

    fieldIds[count++] = DCGM_FI_DEV_VGPU_VM_ID;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_VM_NAME;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_TYPE;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_UUID;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_DRIVER_VERSION;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_MEMORY_USAGE;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_LICENSE_STATUS;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT;

    if (count >= 32) {
        PRINT_ERROR("", "Update DeviceGetAttributes to accommodate more fields\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = DCGM_FE_VGPU;
    entityPair.entityId = vgpuId;

    DcgmFvBuffer fvBuffer(0);
    
    ret = helperGetLatestValuesForFields(pLwcmHandle, 0, &entityPair, 1, 0, fieldIds, count, &fvBuffer, 0);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    dcgmBufferedFv_t *fv;
    dcgmBufferedFvLwrsor_t cursor = 0;

    for (fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor)) {
        switch (fv->fieldId) {
            case DCGM_FI_DEV_VGPU_VM_ID:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pLwcmVgpuInstanceAttr->vmId)) {
                    PRINT_ERROR("", "String overflow error for the requested vGPU instance VM ID field");
                    strncpy(pLwcmVgpuInstanceAttr->vmId, DCGM_STR_BLANK, strlen(DCGM_STR_BLANK) + 1);
                } else {
                    strncpy(pLwcmVgpuInstanceAttr->vmId, fv->value.str, length + 1);
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_VM_NAME:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pLwcmVgpuInstanceAttr->vmName)) {
                    PRINT_ERROR("", "String overflow error for the requested vGPU instance VM name field");
                    strncpy(pLwcmVgpuInstanceAttr->vmName, DCGM_STR_BLANK, strlen(DCGM_STR_BLANK) + 1);
                } else {
                    strncpy(pLwcmVgpuInstanceAttr->vmName, fv->value.str, length + 1);
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_TYPE:
                pLwcmVgpuInstanceAttr->vgpuTypeId = fv->value.i64;
                break;

            case DCGM_FI_DEV_VGPU_UUID:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pLwcmVgpuInstanceAttr->vgpuUuid)) {
                    PRINT_ERROR("", "String overflow error for the requested vGPU instance UUID field");
                    strncpy(pLwcmVgpuInstanceAttr->vgpuUuid, DCGM_STR_BLANK, strlen(DCGM_STR_BLANK) + 1);
                } else {
                    strncpy(pLwcmVgpuInstanceAttr->vgpuUuid, fv->value.str, length + 1);
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_DRIVER_VERSION:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pLwcmVgpuInstanceAttr->vgpuDriverVersion)) {
                    PRINT_ERROR("", "String overflow error for the requested vGPU instance driver version field");
                    strncpy(pLwcmVgpuInstanceAttr->vgpuDriverVersion, DCGM_STR_BLANK, strlen(DCGM_STR_BLANK) + 1);
                } else {
                    strncpy(pLwcmVgpuInstanceAttr->vgpuDriverVersion, fv->value.str, length + 1);
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_MEMORY_USAGE:
                pLwcmVgpuInstanceAttr->fbUsage = fv->value.i64;
                break;

            case DCGM_FI_DEV_VGPU_LICENSE_STATUS:
                pLwcmVgpuInstanceAttr->licenseStatus = fv->value.i64;
                break;

            case DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT:
                pLwcmVgpuInstanceAttr->frameRateLimit = fv->value.i64;
                break;

            default:
                /* This should never happen */
                return DCGM_ST_GENERIC_ERROR;
                break;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************
 * Helper method to set the configuration for a group
 *****************************************************************************/
dcgmReturn_t helperConfigSet(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, 
        dcgmConfig_t *pDeviceConfig, dcgmStatus_t pLwcmStatusList)
{
    dcgm_config_msg_set_v1 msg;
    dcgmReturn_t dcgmReturn;
    int i;

    if(!pDeviceConfig)
    {
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    if (pDeviceConfig->version != dcgmConfig_version)
    {
        PRINT_ERROR("", "Version mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdConfig;
    msg.header.subCommand = DCGM_CONFIG_SR_SET;
    msg.header.version = dcgm_config_msg_set_version;
    msg.groupId = groupId;
    memcpy(&msg.config, pDeviceConfig, sizeof(msg.config));

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    
    /* Update error codes from the error list if there are any */
    if (msg.numStatuses && pLwcmStatusList)
    {
        helperUpdateErrorCodes(pLwcmStatusList, msg.numStatuses, msg.statuses);
    }

    return dcgmReturn;
}

/*****************************************************************************
 * Helper method to set the vGPU configuration for a group
 *****************************************************************************/
dcgmReturn_t helperVgpuConfigSet(dcgmHandle_t pLwcmHandle, dcgmGpuGrp_t groupId,
        dcgmVgpuConfig_t *pDeviceConfig, dcgmStatus_t pLwcmStatusList)
{
    if ((pDeviceConfig->version < dcgmVgpuConfig_version1)
            || (pDeviceConfig->version > dcgmVgpuConfig_version))
    {
        PRINT_ERROR("%x %x", "VgpuConfigSet version %x mismatches current version %x",
                    pDeviceConfig->version, dcgmVgpuConfig_version);
        return DCGM_ST_VER_MISMATCH;
    }

    /* This code never worked, this API is private, and the tests in test_vgpu.py are disabled.
       Returning NOT_SUPPORTED for now */
    return DCGM_ST_NOT_SUPPORTED;
}

/*****************************************************************************/
dcgmReturn_t helperConfigGet(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, 
                             dcgmConfigType_t reqType, int count,
                             dcgmConfig_t *pDeviceConfigList, 
                             dcgmStatus_t pLwcmStatusList)
{
    dcgm_config_msg_get_v1 msg;
    dcgmReturn_t dcgmReturn;
    int i;
    unsigned int versionAtBaseIndex;

    if ((!pDeviceConfigList) || (count <= 0) || 
        ((reqType != DCGM_CONFIG_TARGET_STATE) && (reqType != DCGM_CONFIG_LWRRENT_STATE)))

    {
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    /* Get version at the index 0 */
    versionAtBaseIndex = pDeviceConfigList[0].version;

    /* Verify requested version in the list of output parameters */
    for (i = 0; i < count; ++i) 
    {
        if (pDeviceConfigList[i].version != versionAtBaseIndex) 
        {
            PRINT_ERROR("", "Version mismatch");
            return DCGM_ST_VER_MISMATCH;
        }

        if (pDeviceConfigList[i].version != dcgmConfig_version)
        {
            PRINT_ERROR("", "Version mismatch");
            return DCGM_ST_VER_MISMATCH;
        }
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdConfig;
    msg.header.subCommand = DCGM_CONFIG_SR_GET;
    msg.header.version = dcgm_config_msg_get_version;
    msg.groupId = groupId;
    msg.reqType = reqType;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    
    /* Update error codes from the error list if there are any */
    if (msg.numStatuses && pLwcmStatusList)
    {
        helperUpdateErrorCodes(pLwcmStatusList, msg.numStatuses, msg.statuses);
    }

    /* Copy the configs back to the caller's array */
    if (msg.numConfigs > 0)
    {
        unsigned int numToCopy = DCGM_MIN(msg.numConfigs, (unsigned int)count);
        memcpy(pDeviceConfigList, msg.configs, numToCopy * sizeof(msg.configs[0]));
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t helperVgpuConfigGet(dcgmHandle_t pLwcmHandle, dcgmGpuGrp_t groupId,
        dcgmConfigType_t reqType, int count,
        dcgmVgpuConfig_t *pDeviceConfigList, dcgmStatus_t pLwcmStatusList)
{
    if(!pDeviceConfigList || count < 1)
        return DCGM_ST_BADPARAM;
    
    /* Get version at the index 0 */
    unsigned int versionAtBaseIndex = pDeviceConfigList[0].version;
    int index;

    /* Verify requested version in the list of output parameters */
    for (index = 0; index < count; ++index)
    {
        if (pDeviceConfigList[index].version != versionAtBaseIndex)
        {
            return DCGM_ST_VER_MISMATCH;
        }

        if ((pDeviceConfigList[index].version < dcgmVgpuConfig_version1)|| 
            (pDeviceConfigList[index].version > dcgmVgpuConfig_version))
        {
            return DCGM_ST_VER_MISMATCH;
        }
    }

    /* This code never worked, this API is private, and the tests in test_vgpu.py are disabled.
       Returning NOT_SUPPORTED for now */
    return DCGM_ST_NOT_SUPPORTED;
}

/*****************************************************************************/
dcgmReturn_t helperConfigEnforce(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, 
                                 dcgmStatus_t pLwcmStatusList)
{
    dcgm_config_msg_enforce_group_v1 msg;
    dcgmReturn_t dcgmReturn;
    int i;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdConfig;
    msg.header.subCommand = DCGM_CONFIG_SR_ENFORCE_GROUP;
    msg.header.version = dcgm_config_msg_enforce_group_version;
    msg.groupId = groupId;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    
    /* Update error codes from the error list if there are any */
    if (msg.numStatuses && pLwcmStatusList)
    {
        helperUpdateErrorCodes(pLwcmStatusList, msg.numStatuses, msg.statuses);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t helperVgpuConfigEnforce(dcgmHandle_t pLwcmHandle, dcgmGpuGrp_t groupId,
        dcgmStatus_t pLwcmStatusList)
{
   /* This code never worked, this API is private, and the tests in test_vgpu.py are disabled.
       Returning NOT_SUPPORTED for now */
    return DCGM_ST_NOT_SUPPORTED;
}

/*****************************************************************************/
dcgmReturn_t tsapiInjectEntityFieldValue(dcgmHandle_t pLwcmHandle, 
                                         dcgm_field_entity_group_t entityGroupId,
                                         dcgm_field_eid_t entityId, 
                                         dcgmInjectFieldValue_t *pDcgmInjectFieldValue)
{
    lwcm::InjectFieldValue *pProtoInjectFieldValue; /* Protobuf equivalent structure of the output parameter. */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if (NULL == pDcgmInjectFieldValue) 
    {
        return DCGM_ST_BADPARAM;
    }

    /* Allocate and set version in the protobuf struct */
    pProtoInjectFieldValue = new lwcm::InjectFieldValue;
    pProtoInjectFieldValue->set_entitygroupid(entityGroupId);
    pProtoInjectFieldValue->set_entityid(entityId);
    pProtoInjectFieldValue->set_version(dcgmInjectFieldValue_version);
    pProtoInjectFieldValue->mutable_fieldvalue()->set_fieldid(pDcgmInjectFieldValue->fieldId);
    pProtoInjectFieldValue->mutable_fieldvalue()->set_status(DCGM_ST_OK);
    pProtoInjectFieldValue->mutable_fieldvalue()->set_ts(pDcgmInjectFieldValue->ts);
    pProtoInjectFieldValue->mutable_fieldvalue()->set_version(dcgmFieldValue_version2);
    

    switch(pDcgmInjectFieldValue->fieldType)
    {
        case DCGM_FT_DOUBLE:
            pProtoInjectFieldValue->mutable_fieldvalue()->set_fieldtype(lwcm::DBL);
            pProtoInjectFieldValue->mutable_fieldvalue()->mutable_val()->set_dbl(pDcgmInjectFieldValue->value.dbl);
            break;
        case DCGM_FT_INT64:
            pProtoInjectFieldValue->mutable_fieldvalue()->set_fieldtype(lwcm::INT64);
            pProtoInjectFieldValue->mutable_fieldvalue()->mutable_val()->set_i64(pDcgmInjectFieldValue->value.i64);
            break;
        case DCGM_FT_STRING:
            pProtoInjectFieldValue->mutable_fieldvalue()->set_fieldtype(lwcm::STR);
            pProtoInjectFieldValue->mutable_fieldvalue()->mutable_val()->set_str(pDcgmInjectFieldValue->value.str);
            break;
        default:
            return DCGM_ST_BADPARAM;
    }    

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::INJECT_FIELD_VALUE, 
                                    lwcm::OPERATION_SINGLE_ENTITY, entityId, 0);
    if (NULL == pCmdTemp) {
        delete pProtoInjectFieldValue;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is 
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_injectfieldvalue(pProtoInjectFieldValue);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t helperInjectFieldValue(dcgmHandle_t pDcgmHandle, unsigned int gpuId, 
                                    dcgmInjectFieldValue_t *pDcgmInjectFieldValue)
{
    return tsapiInjectEntityFieldValue(pDcgmHandle, DCGM_FE_GPU, gpuId, 
                                       pDcgmInjectFieldValue);
}

/*****************************************************************************/
dcgmReturn_t helperGetCacheManagerFieldInfo(dcgmHandle_t pLwcmHandle,
        dcgmCacheManagerFieldInfo_t *fieldInfo)
{
    //lwcm::InjectFieldValue *pProtoInjectFieldValue; /* Protobuf equivalent structure of the output parameter. */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    std::string retFieldInfoStr;

    if (!fieldInfo)
        return DCGM_ST_BADPARAM;

    fieldInfo->version = dcgmCacheManagerFieldInfo_version;

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::CACHE_MANAGER_FIELD_INFO, lwcm::OPERATION_SINGLE_ENTITY, fieldInfo->gpuId, 0);
    if (!pCmdTemp)
    {
        PRINT_ERROR("", "encodePrb.AddCommand returned NULL");
        return DCGM_ST_GENERIC_ERROR;
    }

    pCmdTemp->add_arg()->set_cachemanagerfieldinfo(fieldInfo, sizeof(*fieldInfo));

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    retFieldInfoStr = *(vecCmdsRef[0]->mutable_arg(0)->mutable_cachemanagerfieldinfo());
    if(retFieldInfoStr.size() != sizeof(*fieldInfo))
    {
        PRINT_ERROR("%d %d", "Got CACHE_MANAGER_FIELD_INFO of %d bytes. Expected %d bytes",
                (int)retFieldInfoStr.size(), (int)sizeof(*fieldInfo));
        return DCGM_ST_VER_MISMATCH;
    }

    memcpy(fieldInfo, (dcgmCacheManagerFieldInfo_t *)retFieldInfoStr.c_str(), sizeof(*fieldInfo));

    if(fieldInfo->version != dcgmCacheManagerFieldInfo_version)
        return DCGM_ST_VER_MISMATCH; /* Same size struct with different version */

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t helperGetMultipleValuesForField(dcgmHandle_t pLwcmHandle,
        dcgm_field_entity_group_t entityGroup, dcgm_field_eid_t entityId,
        unsigned int fieldId, int *count, long long startTs, long long endTs,
        dcgmOrder_t order, dcgmFieldValue_v1 values[])
{
    lwcm::FieldMultiValues *pProtoGetMultiValuesForField = 0;
    lwcm::FieldMultiValues *pResponse = 0;
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    int maxCount, i;
    int fieldType;
    dcgm_field_meta_p fieldMeta;

    if(!count || (*count) < 1 || !fieldId || !values)
        return DCGM_ST_BADPARAM;

    maxCount = *count;
    *count = 0;

    PRINT_DEBUG("%u %u %d %d %lld %lld %d", "helperGetMultipleValuesForField eg %u eid %u, "
                "fieldId %d, maxCount %d, startTs %lld endTs %lld, order %d",
                entityGroup, entityId, (int)fieldId, maxCount, startTs, endTs, (int)order);

    /* Validate the fieldId */
    fieldMeta = DcgmFieldGetById(fieldId);
    if(!fieldMeta)
    {
        PRINT_ERROR("%u", "Invalid fieldId %u", fieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    memset(values, 0, sizeof(values[0]) * maxCount);

    pProtoGetMultiValuesForField = new lwcm::FieldMultiValues;
    pProtoGetMultiValuesForField->set_version(dcgmGetMultipleValuesForField_version);
    pProtoGetMultiValuesForField->set_fieldid(fieldId);
    //fieldType not required for request
    pProtoGetMultiValuesForField->set_startts(startTs);
    pProtoGetMultiValuesForField->set_endts(endTs);
    pProtoGetMultiValuesForField->set_maxcount(maxCount);
    pProtoGetMultiValuesForField->set_orderflag((lwcm::MultiValuesOrder)order);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::GET_FIELD_MULTIPLE_VALUES, lwcm::OPERATION_SINGLE_ENTITY, entityId, 0);
    if (NULL == pCmdTemp)
    {
        PRINT_ERROR("", "encodePrb.AddCommand failed");
        delete pProtoGetMultiValuesForField;
        return DCGM_ST_GENERIC_ERROR;
    }

    if(fieldMeta->scope == DCGM_FS_GLOBAL)
        pCmdTemp->set_entitygroupid(DCGM_FE_NONE);
    else
        pCmdTemp->set_entitygroupid(entityGroup);

    /* Set the arg to passed as a proto message. After this point the arg is
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_fieldmultivalues(pProtoGetMultiValuesForField);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        PRINT_DEBUG("%d", "ProcessAtEmbeddedHostEngine returned %d", (int)ret);
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        PRINT_DEBUG("%d", "vecCmdsRef[0]->status() %d", vecCmdsRef[0]->status());
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    if (!(vecCmdsRef[0]->arg_size() && vecCmdsRef[0]->arg(0).has_fieldmultivalues()))
    {
        PRINT_ERROR("", "arg or fieldmultivalue missing");
        return DCGM_ST_GENERIC_ERROR;
    }

    pResponse = vecCmdsRef[0]->mutable_arg(0)->mutable_fieldmultivalues();

    if(pResponse->version() != dcgmGetMultipleValuesForField_version1)
        return DCGM_ST_VER_MISMATCH;
    if(!pResponse->has_fieldtype())
    {
        PRINT_ERROR("", "Field type missing");
        return DCGM_ST_GENERIC_ERROR;
    }

    *count = pResponse->vals_size();

    fieldType = pResponse->fieldtype();

    for(i = 0; i < (*count); i++)
    {
        lwcm::Value *responseValue = pResponse->mutable_vals(i);

        if(responseValue->has_timestamp())
            values[i].ts = responseValue->timestamp();
        else
            PRINT_WARNING("%d %d", "timestamp missing at index %d/%d", i, (*count));


        values[i].version = dcgmFieldValue_version1;
        values[i].fieldId = fieldId;
        values[i].fieldType = fieldType;
        values[i].status = 0;

        switch (values[i].fieldType)
        {
            case DCGM_FT_DOUBLE:
                values[i].value.dbl = responseValue->dbl();
                break;

            case DCGM_FT_INT64:
            case DCGM_FT_TIMESTAMP:
                values[i].value.i64 = responseValue->i64();
                break;

            case DCGM_FT_STRING:
                size_t length;
                length = strlen(responseValue->str().c_str());
                if (length + 1 > DCGM_MAX_STR_LENGTH)
                {
                    PRINT_ERROR("", "String overflow error for the requested field");
                    return DCGM_ST_GENERIC_ERROR;
                }

                strncpy(values[i].value.str, responseValue->str().c_str(),
                        sizeof(values[i].value.str)-1);
                break;

            case DCGM_FT_BINARY:
            {
                if(responseValue->blob().size() > sizeof(values[i].value.blob))
                {
                    PRINT_ERROR("%d %d %d", "Buffer values[index].value.blob size %d too small for %d. fieldType %d",
                            (int)sizeof(values[i].value.blob), (int)responseValue->blob().size(),
                            (int)values[i].fieldType);
                    return DCGM_ST_MEMORY;
                }

                memcpy(values[i].value.blob, (void *)responseValue->blob().c_str(),
                        responseValue->blob().size());
                break;
            }

            default:
                PRINT_ERROR("%c", "Uknown type: %c", (char)fieldType);
                return DCGM_ST_GENERIC_ERROR;
        }

    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
/*
 * Helper to send a structure request to the host engine
 *
 * gpuId      IN: If >= 0, used as the GPU id for the request. Mutually exclusive with groupId
 * groupId    IN: If >= 0, used as the groupId for the request. Mutually exclusive with gpuId
 * cmdType    IN: lwcm::? command type to send. This is used by the switch statement in the host engine
 * structData IO: Pointer to the struct that will be transferred to the host engine and be populated
 *                with the results returned from the host engine
 * structSize IN: Size of the data structData points at
 *
 */

dcgmReturn_t helperSendStructRequest(dcgmHandle_t pLwcmHandle,
        unsigned int cmdType,
        int gpuId, int groupId,
        void *structData, int structSize)
{
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    unsigned int opMode;
    int opModeId;

    if(gpuId >= 0 && groupId >= 0)
    {
        PRINT_WARNING("%d %d", "Invalid combo of gpuId %d and groupId %d", gpuId, groupId);
        return DCGM_ST_BADPARAM;
    }

    if(!structData || structSize < 1)
        return DCGM_ST_BADPARAM;

    /* We already validated mutual exclusivity above. Now prepare to pass gpuId/groupId */
    if(gpuId >= 0)
    {
        opMode = lwcm::OPERATION_SINGLE_ENTITY;
        opModeId = gpuId;
    }
    else if(groupId >= 0) /* groupId */
    {
        opMode = lwcm::OPERATION_GROUP_ENTITIES;
        opModeId = groupId;
    }
    else
    {
        opMode = lwcm::OPERATION_SYSTEM;
        opModeId = -1;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(cmdType, opMode, opModeId, 0);
    if (NULL == pCmdTemp)
    {
        PRINT_ERROR("", "Error from AddCommand");
        return DCGM_ST_GENERIC_ERROR;
    }

    pCmdTemp->add_arg()->set_blob(structData, structSize);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    if(!vecCmdsRef[0]->arg_size())
    {
        PRINT_ERROR("", "Arg size of 0 unexpected");
        return DCGM_ST_GENERIC_ERROR;
    }

    if(!vecCmdsRef[0]->arg(0).has_blob())
    {
        PRINT_ERROR("", "Response missing blob");
        return DCGM_ST_GENERIC_ERROR;
    }

    if((int)vecCmdsRef[0]->arg(0).blob().size() > structSize)
    {
        PRINT_ERROR("%d %d", "Returned blob size %d > structSize %d",
                (int)vecCmdsRef[0]->arg(0).blob().size(), structSize);
        return DCGM_ST_GENERIC_ERROR;
    }

    memcpy(structData, (void *)vecCmdsRef[0]->arg(0).blob().c_str(),
            vecCmdsRef[0]->arg(0).blob().size());

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t helperUnwatchFieldValue(dcgmHandle_t pLwcmHandle, int gpuId, unsigned short fieldId, int clearCache)
{
    lwcm::UnwatchFieldValue *pProtoUnwatchFieldValue; /* Protobuf Arg */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if (!fieldId)
        return DCGM_ST_BADPARAM;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById((unsigned short)fieldId);
    if (NULL == fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        PRINT_ERROR("%u", "field ID %u is not a valid field ID", fieldId);
        return DCGM_ST_BADPARAM;
    }

    pProtoUnwatchFieldValue = new lwcm::UnwatchFieldValue;
    pProtoUnwatchFieldValue->set_version(dcgmUnwatchFieldValue_version);
    pProtoUnwatchFieldValue->set_fieldid(fieldId);
    pProtoUnwatchFieldValue->set_clearcache(clearCache);    

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::UNWATCH_FIELD_VALUE, lwcm::OPERATION_SINGLE_ENTITY, gpuId, 0);
    if (NULL == pCmdTemp) {
        delete pProtoUnwatchFieldValue;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the entityGroupId */
    dcgm_field_entity_group_t entityGroupId = DCGM_FE_GPU;
    if(fieldMeta->scope == DCGM_FS_GLOBAL)
        entityGroupId = DCGM_FE_NONE;
    pCmdTemp->set_entitygroupid((int)entityGroupId);

    /* Set the arg to passed as a proto message. After this point the arg is 
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_unwatchfieldvalue(pProtoUnwatchFieldValue);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t helperPolicyGet(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, 
                             int count, dcgmPolicy_t dcgmPolicy[], 
                             dcgmStatus_t dcgmStatusList)
{
    dcgm_policy_msg_get_policies_t msg;
    dcgmReturn_t dcgmReturn;
    int i;

    if ((NULL == dcgmPolicy) || (count <= 0))
    {
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    /* Note: dcgmStatusList has always been ignored by this request. Continuing this tradition on
             as I refactor this for modularity */

    /* Verify requested version in the list of output parameters */
    for (i = 0; i < count; i++)
    {
        if (dcgmPolicy[i].version != dcgmPolicy_version)
        {
            PRINT_ERROR("%d", "Version mismatch at index %d", i);
            return DCGM_ST_VER_MISMATCH;
        }
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdPolicy;
    msg.header.subCommand = DCGM_POLICY_SR_GET_POLICIES;
    msg.header.version = dcgm_policy_msg_get_policies_version;
    msg.groupId = groupId;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    
    if(dcgmReturn == DCGM_ST_OK)
    {
        if(msg.numPolicies > count)
            dcgmReturn = DCGM_ST_INSUFFICIENT_SIZE; /* Tell the user we only copied "count" entries */
        
        msg.numPolicies = DCGM_MIN(count, msg.numPolicies);

        memcpy(dcgmPolicy, msg.policies, msg.numPolicies * sizeof(msg.policies[0]));
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t helperPolicySet(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, 
                             dcgmPolicy_t *dcgmPolicy, dcgmStatus_t dcgmStatusList)
{
    dcgm_policy_msg_set_policy_t msg;
    dcgmReturn_t dcgmReturn;

    if (NULL == dcgmPolicy)
        return DCGM_ST_BADPARAM;

    /* Note: dcgmStatusList has always been ignored by this request. Continuing this tradition on
             as I refactor this for modularity */

    if (dcgmPolicy->version != dcgmPolicy_version)
    {
        PRINT_ERROR("", "Version mismatch.");
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdPolicy;
    msg.header.subCommand = DCGM_POLICY_SR_SET_POLICY;
    msg.header.version = dcgm_policy_msg_set_policy_version;
    msg.groupId = groupId;
    memcpy(&msg.policy, dcgmPolicy, sizeof(msg.policy));

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t helperPolicyRegister(dcgmHandle_t dcgmHandle,
        dcgmGpuGrp_t groupId, dcgmPolicyCondition_t condition, fpRecvUpdates beginCallback,
        fpRecvUpdates finishCallback)
{
    dcgmReturn_t dcgmReturn;
    dcgm_policy_msg_register_t msg;

    /* Make an ansync object. We're going to pass ownership off, so we won't have to free it */
    DcgmPolicyRequest *policyRequest = new DcgmPolicyRequest(beginCallback, finishCallback);

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdPolicy;
    msg.header.subCommand = DCGM_POLICY_SR_REGISTER;
    msg.header.version = dcgm_policy_msg_register_version;
    msg.groupId = groupId;
    msg.condition = condition;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, policyRequest);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t helperPolicyUnregister(dcgmHandle_t dcgmHandle,
        dcgmGpuGrp_t groupId, dcgmPolicyCondition_t condition)
{
    dcgmReturn_t dcgmReturn;
    dcgm_policy_msg_unregister_t msg;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdPolicy;
    msg.header.subCommand = DCGM_POLICY_SR_UNREGISTER;
    msg.header.version = dcgm_policy_msg_unregister_version;
    msg.groupId = groupId;
    msg.condition = condition;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t tsapiSetEntityLwLinkLinkState(dcgmHandle_t dcgmHandle, 
                                           dcgmSetLwLinkLinkState_v1 *linkState)
{
    dcgmReturn_t ret;

    if(!linkState)
        return DCGM_ST_BADPARAM;
    if(linkState->version != dcgmSetLwLinkLinkState_version1)
        return DCGM_ST_VER_MISMATCH;

    ret = helperSendStructRequest(dcgmHandle, lwcm::SET_LWLINK_LINK_STATUS, -1, -1, 
                                  linkState, sizeof(*linkState));
    return ret;
}

/*****************************************************************************/
static dcgmReturn_t helperGetFieldValuesSince(dcgmHandle_t pLwcmHandle,
        dcgmGpuGrp_t groupId, long long sinceTimestamp,
        unsigned short *fieldIds, int numFieldIds,
        long long *nextSinceTimestamp,
        dcgmFieldValueEnumeration_f enumCB, void *userData)
{
    dcgmReturn_t lwcmSt, retLwcmSt;
    dcgmGroupInfo_t groupInfo = {0};
    unsigned int i;
    int j;
    unsigned int gpuId, fieldId;
    int valuesAtATime = 100; /* How many values should we fetch at a time */
    int retNumFieldValues;
    dcgmFieldValue_v1 *fieldValues = 0;
    int callbackSt = 0;
    long long endQueryTimestamp = 0;

    retLwcmSt = DCGM_ST_OK;

    if(!fieldIds || !enumCB || !nextSinceTimestamp || numFieldIds < 1)
    {
        PRINT_ERROR("", "Bad param to helperGetFieldValuesSince");
        return DCGM_ST_BADPARAM;
    }

    PRINT_DEBUG("%p %lld %d %p", "helperGetFieldValuesSince groupId %p, sinceTs %lld, numFieldIds %d, userData %p",
                groupId, sinceTimestamp, numFieldIds, userData);

    *nextSinceTimestamp = sinceTimestamp;

    groupInfo.version = dcgmGroupInfo_version;

    /* Colwert groupId to list of GPUs. Note that this is an extra round trip to the server
     * in the remote case, but it keeps the code much simpler */
    lwcmSt = helperGroupGetInfo(pLwcmHandle, groupId, &groupInfo, &endQueryTimestamp);
    if(lwcmSt != DCGM_ST_OK)
    {
        PRINT_ERROR("%p %d", "helperGroupGetInfo groupId %p returned %d", groupId, (int)lwcmSt);
        return lwcmSt;
    }

    PRINT_DEBUG("%s %d", "Got group %s with %d entities", groupInfo.groupName, groupInfo.count);

    /* Pre-check the group for non-GPU/non-global entities */
    for(i=0; i<groupInfo.count; i++)
    {
        if(groupInfo.entityList[i].entityGroupId != DCGM_FE_GPU && 
           groupInfo.entityList[i].entityGroupId != DCGM_FE_NONE)
        {
            PRINT_ERROR("%p %u %u", "helperGetFieldValuesSince called on groupId %p with non-GPU eg %u, eid %u.",
                          groupId, groupInfo.entityList[i].entityGroupId, groupInfo.entityList[i].entityId);
            return DCGM_ST_NOT_SUPPORTED;
        }
    }

    fieldValues = (dcgmFieldValue_v1 *)malloc(sizeof(*fieldValues)*valuesAtATime);
    if(!fieldValues)
    {
        PRINT_ERROR("%d", "Unable to alloc %d bytes", (int)(sizeof(*fieldValues) * valuesAtATime));
        return DCGM_ST_MEMORY;
    }
    memset(fieldValues, 0, sizeof(*fieldValues)*valuesAtATime);

    /* Fetch valuesAtATime values for each GPU for each field since sinceTimestamp.
     * Make valuesAtATime large enough to offset the fact that this is a round trip
     * to the server for each combo of gpuId, fieldId, and valuesAtATime values
     */

    for(i=0; i<groupInfo.count; i++)
    {
        gpuId = groupInfo.entityList[i].entityId;

        for(j = 0; j < numFieldIds; j++)
        {
            fieldId = fieldIds[j];

            retNumFieldValues = valuesAtATime;
            lwcmSt = helperGetMultipleValuesForField(pLwcmHandle, DCGM_FE_GPU, gpuId, fieldId,
                                                     &retNumFieldValues, sinceTimestamp,
                                                     endQueryTimestamp,
                                                     DCGM_ORDER_ASCENDING, fieldValues);
            if(lwcmSt == DCGM_ST_NO_DATA)
            {
                PRINT_DEBUG("%u, %u %lld", "DCGM_ST_NO_DATA for gpuId %u, fieldId %u, sinceTs %lld",
                        gpuId, fieldId, sinceTimestamp);
                continue;
            }
            else if(lwcmSt != DCGM_ST_OK)
            {
                PRINT_ERROR("%d %u %u", "Got st %d from helperGetMultipleValuesForField gpuId %u, fieldId %u",
                        (int)lwcmSt, gpuId, fieldId);
                retLwcmSt = lwcmSt;
                goto CLEANUP;
            }

            PRINT_DEBUG("%d %u %u", "Got %d values for gpuId %u, fieldId %u", retNumFieldValues,
                    gpuId, fieldId);

            callbackSt = enumCB(gpuId, fieldValues, retNumFieldValues, userData);
            if(callbackSt != 0)
            {
                PRINT_DEBUG("", "User requested callback exit");
                /* Leaving status as OK. User requested the exit */
                goto CLEANUP;
            }
        }
    }

    /* Success. We can advance the caller's next query timestamp */
    *nextSinceTimestamp = endQueryTimestamp + 1;


    CLEANUP:
    if(fieldValues)
    {
        free(fieldValues);
        fieldValues = 0;
    }


    return retLwcmSt;
}

/*****************************************************************************/
static dcgmReturn_t helperGetValuesSince(dcgmHandle_t pDcgmHandle,
        dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId,
        long long sinceTimestamp, long long *nextSinceTimestamp,
        dcgmFieldValueEnumeration_f enumCB, 
        dcgmFieldValueEntityEnumeration_f enumCBv2,
        void *userData)
{
    dcgmReturn_t dcgmSt, retDcgmSt;
    dcgmGroupInfo_t groupInfo = {0};
    dcgmFieldGroupInfo_t fieldGroupInfo = {0};
    unsigned int i;
    int j;
    unsigned int fieldId;
    int valuesAtATime = 100; /* How many values should we fetch at a time */
    int retNumFieldValues;
    dcgmFieldValue_v1 *fieldValues = 0;
    int callbackSt = 0;
    long long endQueryTimestamp = 0;

    retDcgmSt = DCGM_ST_OK;

    if((!enumCB && !enumCBv2) || !nextSinceTimestamp)
    {
        PRINT_ERROR("", "Bad param to helperGetValuesSince");
        return DCGM_ST_BADPARAM;
    }

    *nextSinceTimestamp = sinceTimestamp;

    fieldGroupInfo.version = dcgmFieldGroupInfo_version;
    fieldGroupInfo.fieldGroupId = fieldGroupId;
    dcgmSt = dcgmFieldGroupGetInfo(pDcgmHandle, &fieldGroupInfo);
    if(dcgmSt != DCGM_ST_OK)
    {
        PRINT_ERROR("%d %p", "Got dcgmSt %d from dcgmFieldGroupGetInfo() fieldGroupId %p",
                    (dcgmReturn_t)dcgmSt, fieldGroupId);
        return dcgmSt;
    }

    PRINT_DEBUG("%p %s %u", "fieldGroup %p, name %s, numFieldIds %u", fieldGroupId,
                fieldGroupInfo.fieldGroupName, fieldGroupInfo.numFieldIds);

    /* Colwert groupId to list of GPUs. Note that this is an extra round trip to the server
     * in the remote case, but it keeps the code much simpler */
    groupInfo.version = dcgmGroupInfo_version;
    dcgmSt = helperGroupGetInfo(pDcgmHandle, groupId, &groupInfo, &endQueryTimestamp);
    if(dcgmSt != DCGM_ST_OK)
    {
        PRINT_ERROR("%p %d", "helperGroupGetInfo groupId %p returned %d", groupId, (int)dcgmSt);
        return dcgmSt;
    }

    PRINT_DEBUG("%s %d %lld", "Got group %s with %d GPUs, endQueryTimestamp %lld",
                groupInfo.groupName, groupInfo.count, endQueryTimestamp);
    
    /* Pre-check the group for non-GPU/non-global entities */
    if(!enumCBv2)
    {
        for(i=0; i<groupInfo.count; i++)
        {
            if(groupInfo.entityList[i].entityGroupId != DCGM_FE_GPU && 
            groupInfo.entityList[i].entityGroupId != DCGM_FE_NONE)
            {
                PRINT_ERROR("%p %u %u", "helperGetValuesSince called on groupId %p with non-GPU eg %u, eid %u.",
                            groupId, groupInfo.entityList[i].entityGroupId, groupInfo.entityList[i].entityId);
                return DCGM_ST_NOT_SUPPORTED;
            }
        }
    }

    fieldValues = (dcgmFieldValue_v1 *)malloc(sizeof(*fieldValues)*valuesAtATime);
    if(!fieldValues)
    {
        PRINT_ERROR("%d", "Unable to alloc %d bytes", (int)(sizeof(*fieldValues) * valuesAtATime));
        return DCGM_ST_MEMORY;
    }
    memset(fieldValues, 0, sizeof(*fieldValues)*valuesAtATime);

    /* Fetch valuesAtATime values for each GPU for each field since sinceTimestamp.
     * Make valuesAtATime large enough to offset the fact that this is a round trip
     * to the server for each combo of gpuId, fieldId, and valuesAtATime values
     */

    for(i=0; i<groupInfo.count; i++)
    {
        for(j = 0; j < (int)fieldGroupInfo.numFieldIds; j++)
        {
            fieldId = fieldGroupInfo.fieldIds[j];

            /* Using endQueryTimestamp as endTime here so we don't get values that update after the
               nextSinceTimestamp we're returning to the client */
            retNumFieldValues = valuesAtATime;
            dcgmSt = helperGetMultipleValuesForField(pDcgmHandle, groupInfo.entityList[i].entityGroupId, 
                                                     groupInfo.entityList[i].entityId, fieldId,
                                                     &retNumFieldValues, sinceTimestamp,
                                                     endQueryTimestamp,
                                                     DCGM_ORDER_ASCENDING, fieldValues);
            if(dcgmSt == DCGM_ST_NO_DATA)
            {
                PRINT_DEBUG("%u %u, %u %lld", "DCGM_ST_NO_DATA for eg %u, eid %u, fieldId %u, sinceTs %lld",
                            groupInfo.entityList[i].entityGroupId, groupInfo.entityList[i].entityId, 
                            fieldId, sinceTimestamp);
                continue;
            }
            else if(dcgmSt != DCGM_ST_OK)
            {
                PRINT_ERROR("%d %u %u %u", "Got st %d from helperGetMultipleValuesForField eg %u, eid %u, fieldId %u",
                            (int)dcgmSt, groupInfo.entityList[i].entityGroupId, 
                            groupInfo.entityList[i].entityId, fieldId);
                retDcgmSt = dcgmSt;
                goto CLEANUP;
            }

            PRINT_DEBUG("%d %u %u %u", "Got %d values for eg %u, eid %u, fieldId %u", 
                        retNumFieldValues, groupInfo.entityList[i].entityGroupId, 
                        groupInfo.entityList[i].entityId, fieldId);

            if(enumCB)
            {
                callbackSt = enumCB(groupInfo.entityList[i].entityId, fieldValues, 
                                    retNumFieldValues, userData);
                if(callbackSt != 0)
                {
                    PRINT_DEBUG("", "User requested callback exit");
                    /* Leaving status as OK. User requested the exit */
                    goto CLEANUP;
                }
            }
            if(enumCBv2)
            {
                callbackSt = enumCBv2(groupInfo.entityList[i].entityGroupId, 
                                      groupInfo.entityList[i].entityId, 
                                      fieldValues, retNumFieldValues, userData);
                if(callbackSt != 0)
                {
                    PRINT_DEBUG("", "User requested callback exit");
                    /* Leaving status as OK. User requested the exit */
                    goto CLEANUP;
                }
            }
        }
    }

    /* Success. We can advance the caller's next query timestamp */
    *nextSinceTimestamp = endQueryTimestamp + 1;
    PRINT_DEBUG("%lld", "nextSinceTimestamp advanced to %lld", *nextSinceTimestamp);

    CLEANUP:
    if(fieldValues)
    {
        free(fieldValues);
        fieldValues = 0;
    }


    return retDcgmSt;
}

/*****************************************************************************/
static dcgmReturn_t helperGetLatestValues(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId,
                                          dcgmFieldGrp_t fieldGroupId, 
                                          dcgmFieldValueEnumeration_f enumCB,
                                          dcgmFieldValueEntityEnumeration_f enumCBv2,
                                          void *userData)
{
    dcgmReturn_t dcgmSt, retDcgmSt;
    int callbackSt = 0;
    long long endQueryTimestamp = 0;
    retDcgmSt = DCGM_ST_OK;

    /* At least one enumCB must be provided */
    if(!enumCB && !enumCBv2)
    {
        PRINT_ERROR("", "Bad param to helperLatestValues");
        return DCGM_ST_BADPARAM;
    }

    DcgmFvBuffer fvBuffer(0);

    dcgmSt = helperGetLatestValuesForFields(pDcgmHandle, groupId, 0, 0,
                                            fieldGroupId, 0, 0, &fvBuffer, 0);
    if(dcgmSt != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from helperGetLatestValuesForFields()", (int)dcgmSt);
        return dcgmSt;
    }

    dcgmBufferedFv_t *fv;
    dcgmFieldValue_v1 fieldValueV1; /* Colwerted from fv */
    dcgmBufferedFvLwrsor_t cursor = 0;

    /* Loop over each returned value and call our callback for it */
    for(fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
    {
        PRINT_DEBUG("%u %u %u", "Got value for eg %u, eid %u, fieldId %u", 
                    fv->entityGroupId, fv->entityId, 
                    fv->fieldId);
        
        /* Get a v1 version to pass to our callbacks */
        fvBuffer.ColwertBufferedFvToFv1(fv, &fieldValueV1);

        if(enumCB)
        {
            if(fv->entityGroupId != DCGM_FE_GPU && 
               fv->entityGroupId != DCGM_FE_NONE)
            {
                PRINT_DEBUG("%p %u %u", "helperGetLatestValues called on groupId %p with non-GPU eg %u, eid %u.",
                            groupId, fv->entityGroupId, fv->entityId);
                continue;
            }
            callbackSt = enumCB(fv->entityId, &fieldValueV1, 1, userData);
            if(callbackSt != 0)
            {
                PRINT_DEBUG("", "User requested callback exit");
                /* Leaving status as OK. User requested the exit */
                break;
            }
        }
        if(enumCBv2)
        {
            callbackSt = enumCBv2((dcgm_field_entity_group_t)fv->entityGroupId, 
                                  fv->entityId, 
                                  &fieldValueV1, 1, userData);
            if(callbackSt != 0)
            {
                PRINT_DEBUG("", "User requested callback exit");
                /* Leaving status as OK. User requested the exit */
                break;
            }
        }
    }

    return retDcgmSt;
}

dcgmReturn_t tsapiWatchFields(dcgmHandle_t pLwcmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId,
                              long long updateFreq, double maxKeepAge, int maxKeepSamples)
{
    lwcm::WatchFields *pWatchFields;     /* Request message */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    int index;

    if (!groupId)
    {
        PRINT_ERROR("", "Bad param");
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::WATCH_FIELDS, lwcm::OPERATION_GROUP_ENTITIES, (intptr_t)groupId, 0);
    if (NULL == pCmdTemp) {
        PRINT_ERROR("", "encodePrb.AddCommand failed");
        return DCGM_ST_GENERIC_ERROR;
    }

    pWatchFields = pCmdTemp->add_arg()->mutable_watchfields();
    pWatchFields->set_version(dcgmWatchFields_version);
    pWatchFields->set_fieldgroupid((uintptr_t)fieldGroupId);
    pWatchFields->set_updatefreq(updateFreq);
    pWatchFields->set_maxkeepage(maxKeepAge);
    pWatchFields->set_maxkeepsamples(maxKeepSamples);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

dcgmReturn_t tsapiUnwatchFields(dcgmHandle_t pLwcmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId)
{
    lwcm::UnwatchFields *pUnwatchFields; /* Request message */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    int index;

    if (!groupId)
    {
        PRINT_ERROR("", "Bad param");
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::UNWATCH_FIELDS, lwcm::OPERATION_GROUP_ENTITIES, (intptr_t)groupId, 0);
    if (NULL == pCmdTemp) 
    {
        PRINT_ERROR("", "encodePrb.AddCommand failed");
        return DCGM_ST_GENERIC_ERROR;
    }

    pUnwatchFields = pCmdTemp->add_arg()->mutable_unwatchfields();
    pUnwatchFields->set_fieldgroupid((uintptr_t)fieldGroupId);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if(ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "processAtHostEngine returned %d", ret);
        return ret;
    }
        
    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        PRINT_DEBUG("%d", "vecCmdsRef[0]->status() returned %d", (int)vecCmdsRef[0]->status());
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

dcgmReturn_t tsapiFieldGroupCreate(dcgmHandle_t pDcgmHandle, int numFieldIds, unsigned short *fieldIds,
                                   char *fieldGroupName, dcgmFieldGrp_t *dcgmFieldGroupId)
{
    dcgmReturn_t dcgmReturn;
    dcgmFieldGroupInfo_t fieldGroupInfo;

    if(numFieldIds < 1 || numFieldIds >= DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP ||
       !fieldGroupName || strlen(fieldGroupName) >= DCGM_MAX_STR_LENGTH || !fieldGroupName[0] ||
       !fieldIds || !dcgmFieldGroupId)
    {
        return DCGM_ST_BADPARAM;
    }

    memset(&fieldGroupInfo, 0, sizeof(fieldGroupInfo));
    fieldGroupInfo.version = dcgmFieldGroupInfo_version;
    strncpy(fieldGroupInfo.fieldGroupName, fieldGroupName, sizeof(fieldGroupInfo.fieldGroupName)-1);
    fieldGroupInfo.numFieldIds = numFieldIds;
    memcpy(fieldGroupInfo.fieldIds, fieldIds, sizeof(fieldIds[0])*numFieldIds);

    dcgmReturn = helperSendStructRequest(pDcgmHandle, lwcm::FIELD_GROUP_CREATE,
                                         -1, -1, &fieldGroupInfo, sizeof(fieldGroupInfo));

    PRINT_DEBUG("%d", "tsapiFieldGroupCreate dcgmSt %d", (int)dcgmReturn);

    *dcgmFieldGroupId = fieldGroupInfo.fieldGroupId;
    return dcgmReturn;
}

dcgmReturn_t tsapiFieldGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId)
{
    dcgmReturn_t dcgmSt;
    dcgmFieldGroupInfo_t fieldGroupInfo;

    PRINT_DEBUG("%p", "dcgmFieldGroupDestroy fieldGroupId %p", dcgmFieldGroupId);

    memset(&fieldGroupInfo, 0, sizeof(fieldGroupInfo));
    fieldGroupInfo.version = dcgmFieldGroupInfo_version;
    fieldGroupInfo.fieldGroupId = dcgmFieldGroupId;

    dcgmSt = helperSendStructRequest(pDcgmHandle, lwcm::FIELD_GROUP_DESTROY,
                                     -1, -1, &fieldGroupInfo, sizeof(fieldGroupInfo));
    PRINT_DEBUG("%d %p", "tsapiFieldGroupDestroy dcgmSt %d, fieldGroupId %p",
                dcgmSt, fieldGroupInfo.fieldGroupId);
    return dcgmSt;
}

dcgmReturn_t tsapiFieldGroupGetInfo(dcgmHandle_t pDcgmHandle,
                                    dcgmFieldGroupInfo_t *fieldGroupInfo)
{
    dcgmReturn_t dcgmReturn;

    if(!fieldGroupInfo)
        return DCGM_ST_BADPARAM;

    /* Valid version can't be 0 */
    if(!fieldGroupInfo->version)
        return DCGM_ST_VER_MISMATCH;

    dcgmReturn = helperSendStructRequest(pDcgmHandle, lwcm::FIELD_GROUP_GET_ONE,
                                         -1, -1, fieldGroupInfo, sizeof(*fieldGroupInfo));

    PRINT_DEBUG("%d", "tsapiFieldGroupGetInfo got st %d", (int)dcgmReturn);
    return dcgmReturn;
}

dcgmReturn_t tsapiFieldGroupGetAll(dcgmHandle_t pDcgmHandle, dcgmAllFieldGroup_t *allGroupInfo)
{
    dcgmReturn_t dcgmReturn;

    if(!allGroupInfo)
        return DCGM_ST_BADPARAM;
    
    /* Valid version can't be 0 or just any random number  */
    if(allGroupInfo->version != dcgmAllFieldGroup_version)
    {
        PRINT_DEBUG("", "Version Mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    dcgmReturn = helperSendStructRequest(pDcgmHandle, lwcm::FIELD_GROUP_GET_ALL,
                                         -1, -1, allGroupInfo, sizeof(*allGroupInfo));

    PRINT_DEBUG("%d", "tsapiFieldGroupGetAll got st %d", (int)dcgmReturn);
    return DCGM_ST_OK;
}

dcgmReturn_t helperHealthSet(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t systems)
{
    dcgm_health_msg_set_systems_t msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdHealth;
    msg.header.subCommand = DCGM_HEALTH_SR_SET_SYSTEMS;
    msg.header.version = dcgm_health_msg_set_systems_version;
    msg.groupId = groupId;
    msg.systems = systems;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    return dcgmReturn;
}

dcgmReturn_t helperHealthGet(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t *systems)
{
    dcgm_health_msg_get_systems_t msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdHealth;
    msg.header.subCommand = DCGM_HEALTH_SR_GET_SYSTEMS;
    msg.header.version = dcgm_health_msg_get_systems_version;
    msg.groupId = groupId;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);

    *systems = msg.systems;
    return dcgmReturn;
}

dcgmReturn_t helperActionManager(dcgmHandle_t dcgmHandle, dcgmRunDiag_t *drd, 
                                 dcgmPolicyAction_t action, 
                                 dcgmDiagResponse_t *response)
{
    dcgm_diag_msg_run_v3 msg3;
    dcgm_diag_msg_run_v2 msg2;
    dcgm_diag_msg_run_v1 msg1;
    dcgmReturn_t dcgmReturn;

    if(!drd || !response)
    {
        PRINT_ERROR("%p %p", "drd %p or response %p was NULL.", drd, response);
        return DCGM_ST_BADPARAM;
    }

    switch (drd->version)
    {
        case dcgmRunDiag_version1: /* fallthrough */
        case dcgmRunDiag_version2: /* fallthrough */
        case dcgmRunDiag_version3: /* fallthrough */
        case dcgmRunDiag_version4: /* fallthrough */
        case dcgmRunDiag_version:  /* fallthrough */
            break;
        default:
            // unknown drd version
            PRINT_ERROR("%X %X", "dcgmRunDiag version mismatch %X != %X and isn't in accepted list", 
                    drd->version, dcgmRunDiag_version);
            return DCGM_ST_VER_MISMATCH;
    }
    
    dcgm_module_command_header_t *header;
    dcgmRunDiag_t *runDiag;

    switch (response->version)
    {
        case dcgmDiagResponse_version3:
            memset(&msg1, 0, sizeof(msg1));
            msg1.header.length = sizeof(msg1);
            msg1.action = action;
            msg1.header.version = dcgm_diag_msg_run_version1;
            msg1.diagResponse.version = dcgmDiagResponse_version3;
            runDiag = &(msg1.runDiag);
            header = &(msg1.header);
            break;

        case dcgmDiagResponse_version4:
            memset(&msg2, 0, sizeof(msg2));
            msg2.header.length = sizeof(msg2);
            msg2.header.version = dcgm_diag_msg_run_version2;
            msg2.action = action;
            msg2.diagResponse.version = dcgmDiagResponse_version4;
            runDiag = &(msg2.runDiag);
            header = &(msg2.header);
            break;

        case dcgmDiagResponse_version5:
            memset(&msg3, 0, sizeof(msg3));
            msg3.header.length = sizeof(msg3);
            msg3.header.version = dcgm_diag_msg_run_version3;
            msg3.diagResponse.version = dcgmDiagResponse_version5;
            msg3.action = action;
            runDiag = &(msg3.runDiag);
            header = &(msg3.header);
            break;
            
        default:
            return DCGM_ST_VER_MISMATCH;
    }
    
    header->moduleId = DcgmModuleIdDiag;
    header->subCommand = DCGM_DIAG_SR_RUN;

    switch (drd->version)
    {
        case dcgmRunDiag_version1:
            memcpy(runDiag, drd, sizeof(dcgmRunDiag_v1));
            break;
        case dcgmRunDiag_version2:
            memcpy(runDiag, drd, sizeof(dcgmRunDiag_v2));
            break;
        case dcgmRunDiag_version3:
            memcpy(runDiag, drd, sizeof(dcgmRunDiag_v3));
            break;
        case dcgmRunDiag_version4:
            memcpy(runDiag, drd, sizeof(dcgmRunDiag_v4));
            break;
        case dcgmRunDiag_version5:
            memcpy(runDiag, drd, sizeof(dcgmRunDiag_v5));
            break;
        default:
            // unknown dcgmRunDiag version
            PRINT_ERROR("%X %X", "dcgmRunDiag_version mismatch %X != %X and isn't in accepted list",
                        drd->version, dcgmRunDiag_version);
            return DCGM_ST_VER_MISMATCH;
    }
    
    static const int SIXTY_MINUTES_IN_MS = 3600000;
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, header, 0, SIXTY_MINUTES_IN_MS);

    switch (response->version)
    {
        case dcgmDiagResponse_version3:
            memcpy(response, &msg1.diagResponse, sizeof(msg1.diagResponse));
            break;

        case dcgmDiagResponse_version4:
            memcpy(response, &msg2.diagResponse, sizeof(msg2.diagResponse));
            break;

        case dcgmDiagResponse_version5:
            memcpy(response, &msg3.diagResponse, sizeof(msg3.diagResponse));
            break;
    }

    return dcgmReturn;
}

dcgmReturn_t helperStopDiag(dcgmHandle_t dcgmHandle)
{
    dcgm_diag_msg_stop_t msg;
    dcgmReturn_t dcgmReturn;
    
    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdDiag;
    msg.header.subCommand = DCGM_DIAG_SR_STOP;
    msg.header.version = dcgm_diag_msg_stop_version;
    
    static const int SIXTY_MINUTES_IN_MS = 3600000;
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, 0, SIXTY_MINUTES_IN_MS);

    return dcgmReturn;
}

/*****************************************************************************/
static dcgmReturn_t helperHealthCheckV2(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthResponse_v2 *response)
{
    dcgm_health_msg_check_v2 msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdHealth;
    msg.header.subCommand = DCGM_HEALTH_SR_CHECK_V2;
    msg.header.version = dcgm_health_msg_check_version2;

    msg.groupId = groupId;
    msg.startTime = 0;
    msg.endTime = 0;
    
    memcpy(&msg.response, response, sizeof(msg.response));
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    memcpy(response, &msg.response, sizeof(msg.response));
    return dcgmReturn;
}

/*****************************************************************************/
static dcgmReturn_t helperHealthCheckV3(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthResponse_v3 *response)
{
    dcgm_health_msg_check_v3 msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdHealth;
    msg.header.subCommand = DCGM_HEALTH_SR_CHECK_V3;
    msg.header.version = dcgm_health_msg_check_version3;

    msg.groupId = groupId;
    msg.startTime = 0;
    msg.endTime = 0;
    
    memcpy(&msg.response, response, sizeof(msg.response));
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    memcpy(response, &msg.response, sizeof(msg.response));
    return dcgmReturn;
}

/*****************************************************************************/
static dcgmReturn_t helperHealthCheckV1(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthResponse_v1 *response)
{
    dcgm_health_msg_check_v1 msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdHealth;
    msg.header.subCommand = DCGM_HEALTH_SR_CHECK_V1;
    msg.header.version = dcgm_health_msg_check_version1;

    msg.groupId = groupId;
    msg.startTime = 0;
    msg.endTime = 0;
    
    memcpy(&msg.response, response, sizeof(msg.response));
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    memcpy(response, &msg.response, sizeof(msg.response));
    return dcgmReturn;
}

/*****************************************************************************/
static dcgmReturn_t helperGetPidInfo(dcgmHandle_t pLwcmHandle, dcgmGpuGrp_t groupId, dcgmPidInfo_t *pidInfo)
{
    dcgmReturn_t dcgmSt;

    if(!pidInfo)
        return DCGM_ST_BADPARAM;

    /* Valid version can't be 0 or just any random number  */
    if(pidInfo->version != dcgmPidInfo_version)
    {
        PRINT_DEBUG("", "Version Mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    if(!pidInfo->pid)
    {
        PRINT_DEBUG("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    dcgmSt = helperSendStructRequest(pLwcmHandle, lwcm::GET_PID_INFORMATION,
            -1, (int)(intptr_t)groupId, pidInfo, sizeof(*pidInfo));
    return dcgmSt;
}

/*****************************************************************************/
static dcgmReturn_t helperGetTopologyAffinity(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmAffinity_t *groupAffinity)
{
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    lwcm::Command   *pGroupCmd;          /* Temp reference to the command */
    dcgmReturn_t ret;

    if (NULL == groupAffinity)
        return DCGM_ST_BADPARAM;

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::GET_TOPOLOGY_INFO_AFFINITY, lwcm::OPERATION_GROUP_ENTITIES, (intptr_t)groupId, 0);
    if (NULL == pCmdTemp) {
        return DCGM_ST_GENERIC_ERROR;
    }

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    // check to see if topology information was even applicable
    if (pGroupCmd->status() == DCGM_ST_NO_DATA)
        return (dcgmReturn_t)pGroupCmd->status();

    if (!(pGroupCmd->arg_size())) {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        return DCGM_ST_GENERIC_ERROR;
    }

    if (pGroupCmd->mutable_arg(0)->has_blob())
    {
        memcpy(groupAffinity, (void *)pGroupCmd->mutable_arg(0)->blob().c_str(), sizeof(dcgmAffinity_t));
    }

    return (dcgmReturn_t)pGroupCmd->status();
}

/*****************************************************************************/
static dcgmReturn_t helperSelectGpusByTopology(dcgmHandle_t pDcgmHandle, uint64_t inputGpuIds, uint32_t numGpus,
                                               uint64_t *outputGpuIds, uint64_t hintFlags)
{
    lwcm::Command *pCmdTemp;
    dcgmReturn_t ret;
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    lwcm::Command   *pGroupCmd;          /* Temp reference to the command */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */

    lwcm::SchedulerHintRequest *shr;
    
    if (NULL == pDcgmHandle)
        return DCGM_ST_BADPARAM;

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::SELECT_GPUS_BY_TOPOLOGY, lwcm::OPERATION_GROUP_ENTITIES, 0, 0);
    if (pCmdTemp == NULL)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    shr = new lwcm::SchedulerHintRequest;
    shr->set_version(dcgmTopoSchedHint_version1);
    shr->set_inputgpuids(inputGpuIds);
    shr->set_numgpus(numGpus);
    shr->set_hintflags(hintFlags);

    pCmdTemp->add_arg()->set_allocated_schedulerhintrequest(shr);

    // This should be fast. We'll do a 30 second timeout
    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret; 
    }    
    
    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];
    
    if (!(pGroupCmd->arg_size()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        PRINT_DEBUG("", "Return argument is missing");
        return DCGM_ST_GENERIC_ERROR;
    }
    
    if (pGroupCmd->mutable_arg(0)->has_i64())
    {
         *outputGpuIds = pGroupCmd->mutable_arg(0)->i64();
    }

    /* Return the global status returned from the operation at the hostengine */
    return (dcgmReturn_t)pGroupCmd->status();
}

static dcgmReturn_t helperGetFieldSummary(dcgmHandle_t pDcgmHandle, dcgmFieldSummaryRequest_t *request)
{
    dcgmReturn_t ret;

    if (!request)
        return DCGM_ST_BADPARAM;
    
    if (request->version != dcgmFieldSummaryRequest_version1)
        return DCGM_ST_VER_MISMATCH;

    ret = helperSendStructRequest(pDcgmHandle, lwcm::GET_FIELD_SUMMARY, request->entityId, -1, request,
                                  sizeof(*request));

    PRINT_DEBUG("%u %d", "Retrieved %u summary types. dcgmReturn %d", request->response.summaryCount, ret);

    return ret;
}

/*****************************************************************************/
static dcgmReturn_t helperGetTopologyPci(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmTopology_t *groupTopology)
{
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    lwcm::Command   *pGroupCmd;          /* Temp reference to the command */
    dcgmReturn_t ret;

    if (NULL == groupTopology)
        return DCGM_ST_BADPARAM;

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::GET_TOPOLOGY_INFO_IO, lwcm::OPERATION_GROUP_ENTITIES, (intptr_t)groupId, 0);
    if (NULL == pCmdTemp) {
        return DCGM_ST_GENERIC_ERROR;
    }

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    // no data is okay, topology struct returned will just numElements == 0
    if (DCGM_ST_OK != ret && DCGM_ST_NO_DATA != ret) {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    if (!(pGroupCmd->arg_size())) {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        return DCGM_ST_GENERIC_ERROR;
    }

    if (pGroupCmd->mutable_arg(0)->has_blob())
    {
        memcpy(groupTopology, (void *)pGroupCmd->mutable_arg(0)->blob().c_str(), sizeof(dcgmTopology_t));
    }

    return (dcgmReturn_t)pGroupCmd->status();
}

static dcgmReturn_t tsapiGroupGetAllIds(dcgmHandle_t pLwcmHandle, dcgmGpuGrp_t groupIdList[], unsigned int *count)
{
    return helperGroupGetAllIds(pLwcmHandle, groupIdList, count);
}

static dcgmReturn_t tsapiClientSaveCacheManagerStats(dcgmHandle_t pLwcmHandle, const char *filename, dcgmStatsFileType_t fileType) 
{
    lwcm::CacheManagerSave *pCacheManager; /* Protobuf equivalent structure of the output parameter. */
    LwcmProtobuf encodePrb; /* Protobuf message for encoding */
    LwcmProtobuf decodePrb; /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp; /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if (NULL == filename || DCGM_STATS_FILE_TYPE_JSON != fileType || NULL == pLwcmHandle) {
        return DCGM_ST_BADPARAM;
    }

    /* Allocate and set version in the protobuf struct */
    pCacheManager = new lwcm::CacheManagerSave;
    pCacheManager->set_version(dcgmConfig_version1);
    pCacheManager->set_filename(filename);
    pCacheManager->set_filetype((lwcm::CacheManagerFileType)fileType);


    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::SAVE_CACHED_STATS, lwcm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp) {
        delete pCacheManager;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is 
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_cachemanagersave(pCacheManager);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t) vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiClientLoadCacheManagerStats(dcgmHandle_t pLwcmHandle, const char *filename, dcgmStatsFileType_t fileType)
{
    lwcm::CacheManagerLoad *pProtoLoadCacheStats; /* Protobuf arg */
    LwcmProtobuf encodePrb; /* Protobuf message for encoding */
    LwcmProtobuf decodePrb; /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp; /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;    

    pProtoLoadCacheStats = new lwcm::CacheManagerLoad;
    pProtoLoadCacheStats->set_version(dcgmConfig_version1);
    pProtoLoadCacheStats->set_filename(filename);
    pProtoLoadCacheStats->set_filetype((lwcm::CacheManagerFileType)fileType);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::LOAD_CACHED_STATS, lwcm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp) {
        delete pProtoLoadCacheStats;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is 
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_cachemanagerload(pProtoLoadCacheStats);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t) vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;    
}


/*****************************************************************************
 The entry points for LWCM Host Engine APIs
 *****************************************************************************/


static dcgmReturn_t tsapiEngineRun(unsigned short portNumber, char *socketPath, unsigned int isConnectionTCP)
{
    if (NULL == LwcmHostEngineHandler::Instance()) {
        return DCGM_ST_UNINITIALIZED;
    }

    return (dcgmReturn_t) LwcmHostEngineHandler::Instance()->RunServer(portNumber, socketPath, isConnectionTCP);
}

static dcgmReturn_t tsapiEngineGroupCreate(dcgmHandle_t pDcgmHandle, dcgmGroupType_t type, char *groupName, dcgmGpuGrp_t *pLwcmGrpId)
{
    return helperGroupCreate(pDcgmHandle, type, groupName, pLwcmGrpId);
}

static dcgmReturn_t tsapiEngineGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId)
{
    return helperGroupDestroy(pDcgmHandle, groupId);
}

static dcgmReturn_t tsapiEngineGroupAddDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId)
{
    return helperGroupAddDevice(pDcgmHandle, groupId, gpuId);
}

static dcgmReturn_t tsapiEngineGroupRemoveDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId)
{
    return helperGroupRemoveDevice(pDcgmHandle, groupId, gpuId);
}

static dcgmReturn_t tsapiEngineGroupGetInfo(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmGroupInfo_t *pLwcmGroupInfo)
{
    return helperGroupGetInfo(pDcgmHandle, groupId, pLwcmGroupInfo, 0);
}

static dcgmReturn_t tsapiStatusCreate(dcgmStatus_t *pLwcmStatusList)
{
    if (NULL == pLwcmStatusList) {
        return DCGM_ST_BADPARAM;
    }

    *pLwcmStatusList = new LwcmStatus;
    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiStatusDestroy(dcgmStatus_t pLwcmStatusList)
{
    if (NULL == pLwcmStatusList) {
        return DCGM_ST_BADPARAM;
    }

    delete (LwcmStatus *)pLwcmStatusList;
    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiStatusGetCount(dcgmStatus_t pLwcmStatusList, unsigned int *count)
{
    if ((NULL == pLwcmStatusList) || (NULL == count)) {
        return DCGM_ST_BADPARAM;
    }

    *count = ((LwcmStatus *)pLwcmStatusList)->GetNumErrors();
    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiStatusPopError(dcgmStatus_t pLwcmStatusList, dcgmErrorInfo_t *pLwcmErrorInfo)
{
    if ((NULL == pLwcmStatusList) || (NULL == pLwcmErrorInfo)) {
        return DCGM_ST_BADPARAM;
    }

    if (((LwcmStatus *)pLwcmStatusList)->IsEmpty()) {
        return DCGM_ST_NO_DATA;
    }

    (void)((LwcmStatus *)pLwcmStatusList)->Dequeue(pLwcmErrorInfo);

    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiStatusClear(dcgmStatus_t pLwcmStatusList)
{
    if (NULL == pLwcmStatusList) {
        return DCGM_ST_BADPARAM;
    }

    (void)((LwcmStatus *)pLwcmStatusList)->RemoveAll();

    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiEngineGetAllDevices(dcgmHandle_t pDcgmHandle, unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES], int *count)
{
    return helperGetAllDevices(pDcgmHandle, gpuIdList, count, 0);
}

static dcgmReturn_t tsapiEngineGetAllSupportedDevices(dcgmHandle_t pDcgmHandle, unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES], int *count)
{
    return helperGetAllDevices(pDcgmHandle, gpuIdList, count, 1);
}

static dcgmReturn_t tsapiGetEntityGroupEntities(dcgmHandle_t dcgmHandle, 
                                                dcgm_field_entity_group_t entityGroup,
                                                dcgm_field_eid_t *entities, int *numEntities, 
                                                unsigned int flags)
{
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    lwcm::CmdArg *cmdArg = 0;
    lwcm::EntityList *pEntityList = NULL;
    int entitiesCapacity = *numEntities;

    int onlySupported = (flags & DCGM_GEGE_FLAG_ONLY_SUPPORTED) ? 1 : 0;

    if (!entities || !numEntities) 
    {
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::GET_ENTITY_LIST, lwcm::OPERATION_SYSTEM, -1, 0);
    if(!pCmdTemp) 
    {
        PRINT_ERROR("", "AddCommand failed");
        return DCGM_ST_GENERIC_ERROR;
    }

    cmdArg = pCmdTemp->add_arg();

    pEntityList = new lwcm::EntityList();
    pEntityList->set_entitygroupid(entityGroup);
    pEntityList->set_onlysupported(onlySupported);
    cmdArg->set_allocated_entitylist(pEntityList);

    ret = processAtHostEngine(dcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) 
    {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) 
    {
        PRINT_DEBUG("%d", "vecCmdsRef[0]->status() %d", vecCmdsRef[0]->status());
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    /* Make sure that the returned message has the required structures */
    if (!(vecCmdsRef[0]->arg_size() && vecCmdsRef[0]->arg(0).has_entitylist()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        PRINT_ERROR("", "Returned message was malformed");
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Update the Protobuf reference with the results */
    pEntityList = vecCmdsRef[0]->mutable_arg(0)->mutable_entitylist();
    
    *numEntities = pEntityList->entity_size();

    if(pEntityList->entity_size() > entitiesCapacity)
    {
        PRINT_DEBUG("%d %d", "Insufficient capacity: %d > %d", 
                    pEntityList->entity_size(), entitiesCapacity);
        *numEntities = pEntityList->entity_size();
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    for (int index = 0; index < pEntityList->entity_size(); index++) 
    {
        entities[index] = pEntityList->mutable_entity(index)->entityid();
    }

    return DCGM_ST_OK;
}

dcgmReturn_t tsapiGetLwLinkLinkStatus(dcgmHandle_t dcgmHandle, dcgmLwLinkStatus_v2 *linkStatus)
{
    dcgmReturn_t dcgmReturn;

    if(!linkStatus)
        return DCGM_ST_BADPARAM;
    
    if(linkStatus->version != dcgmLwLinkStatus_version2)
        return DCGM_ST_VER_MISMATCH;

    dcgmReturn = helperSendStructRequest(dcgmHandle, lwcm::GET_LWLINK_LINK_STATUS, -1, -1, 
                                         linkStatus, sizeof(*linkStatus));

    PRINT_DEBUG("%u %u %d", "Got %u GPUs and %u LwSwitches back. dcgmReturn %d", 
                linkStatus->numGpus, linkStatus->numLwSwitches, dcgmReturn);

    return dcgmReturn;
}

static dcgmReturn_t tsapiEngineGetDeviceAttributes(dcgmHandle_t pDcgmHandle, unsigned int gpuId, dcgmDeviceAttributes_t *pLwcmDeviceAttr)
{
    return helperDeviceGetAttributes(pDcgmHandle, gpuId, pLwcmDeviceAttr);
}

static dcgmReturn_t tsapiEngineGetVgpuDeviceAttributes(dcgmHandle_t pDcgmHandle, unsigned int gpuId, dcgmVgpuDeviceAttributes_t *pLwcmVgpuDeviceAttr)
{
    return helperVgpuDeviceGetAttributes(pDcgmHandle, gpuId, pLwcmVgpuDeviceAttr);
}

static dcgmReturn_t tsapiEngineGetVgpuInstanceAttributes(dcgmHandle_t pDcgmHandle, unsigned int vgpuId, dcgmVgpuInstanceAttributes_t *pLwcmVgpuInstanceAttr)
{
    return helperVgpuInstanceGetAttributes(pDcgmHandle, vgpuId, pLwcmVgpuInstanceAttr);
}

static dcgmReturn_t tsapiEngineUpdateDefaultConfig(int gpuId, dcgmConfig_t *pLwcmDefaultConfig)
{
    return DCGM_ST_OK;
    // Remove it later
    // return helperUpdateDefaultConfig(DCGM_MODE_EMBEDDED_HE, NULL, gpuId, pLwcmDefaultConfig);
}

static dcgmReturn_t tsapiEngineConfigSet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmConfig_t *pDeviceConfig, dcgmStatus_t statusHandle)
{
    return helperConfigSet(pDcgmHandle, groupId, pDeviceConfig, statusHandle);
}

static dcgmReturn_t tsapiEngineVgpuConfigSet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmVgpuConfig_t *pDeviceConfig, dcgmStatus_t statusHandle)
{
    return helperVgpuConfigSet(pDcgmHandle, groupId, pDeviceConfig, statusHandle);
}

static dcgmReturn_t tsapiEngineConfigEnforce(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t statusHandle)
{
    return helperConfigEnforce(pDcgmHandle, groupId, statusHandle);
}

static dcgmReturn_t tsapiEngineVgpuConfigEnforce(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t statusHandle)
{
    return helperVgpuConfigEnforce(pDcgmHandle, groupId, statusHandle);
}

static dcgmReturn_t tsapiEngineConfigGet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmConfigType_t type, int count,
        dcgmConfig_t deviceConfigList[], dcgmStatus_t statusHandle)
{
    return helperConfigGet(pDcgmHandle, groupId, type, count, deviceConfigList, statusHandle);
}

static dcgmReturn_t tsapiEngineVgpuConfigGet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmConfigType_t type, int count,
        dcgmVgpuConfig_t deviceConfigList[], dcgmStatus_t statusHandle)
{
    return helperVgpuConfigGet(pDcgmHandle, groupId, type, count, deviceConfigList, statusHandle);
}

static dcgmReturn_t tsapiEngineInjectFieldValue(dcgmHandle_t pDcgmHandle, unsigned int gpuId,
        dcgmInjectFieldValue_t *pLwcmInjectFieldValue)
{
    return helperInjectFieldValue(pDcgmHandle, gpuId, pLwcmInjectFieldValue);
}

static dcgmReturn_t tsapiEngineGetCacheManagerFieldInfo(dcgmHandle_t pDcgmHandle, dcgmCacheManagerFieldInfo_t *fieldInfo)
{
    return helperGetCacheManagerFieldInfo(pDcgmHandle, fieldInfo);
}

static dcgmReturn_t tsapiCreateFakeEntities(dcgmHandle_t pDcgmHandle, dcgmCreateFakeEntities_t *createFakeEntities)
{
    dcgmReturn_t dcgmReturn;

    if(!createFakeEntities)
        return DCGM_ST_BADPARAM;
    
    if(createFakeEntities->version != dcgmCreateFakeEntities_version)
        return DCGM_ST_VER_MISMATCH;

    dcgmReturn = helperSendStructRequest(pDcgmHandle, lwcm::CREATE_FAKE_ENTITIES, -1, -1, 
                                         createFakeEntities, sizeof(*createFakeEntities));

    PRINT_DEBUG("%u %d", "Created %u fake entities. dcgmReturn %d", 
                createFakeEntities->numToCreate, dcgmReturn);

    return dcgmReturn;
}

static dcgmReturn_t tsapiEngineGetLatestValuesForFields(dcgmHandle_t pDcgmHandle, int gpuId, 
                                                        unsigned short fieldIds[], unsigned int count, 
                                                        dcgmFieldValue_v1 values[])
{
    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = DCGM_FE_GPU;
    entityPair.entityId = gpuId;
    DcgmFvBuffer fvBuffer(0);
    dcgmReturn_t dcgmReturn =  helperGetLatestValuesForFields(pDcgmHandle, 0, &entityPair, 1, 0,
                                                              fieldIds, count, &fvBuffer, 0);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    dcgmReturn = fvBuffer.GetAllAsFv1(values, count, 0);
    return dcgmReturn;
}

static dcgmReturn_t tsapiEngineEntityGetLatestValues(dcgmHandle_t pDcgmHandle, 
                                                     dcgm_field_entity_group_t entityGroup, 
                                                     int entityId, unsigned short fieldIds[], 
                                                     unsigned int count, dcgmFieldValue_v1 values[])
{
    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = entityGroup;
    entityPair.entityId = entityId;
    DcgmFvBuffer fvBuffer(0);
    dcgmReturn_t dcgmReturn =  helperGetLatestValuesForFields(pDcgmHandle, 0, &entityPair, 
                                                              1, 0, fieldIds, count, &fvBuffer, 0);
    
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    dcgmReturn = fvBuffer.GetAllAsFv1(values, count, 0);
    return dcgmReturn;
}

static dcgmReturn_t tsapiEngineGetMultipleValuesForField(dcgmHandle_t pDcgmHandle, int gpuId,
        unsigned short fieldId, int *count,
        long long startTs, long long endTs,
        dcgmOrder_t order, dcgmFieldValue_v1 values[])
{
    return helperGetMultipleValuesForField(pDcgmHandle, DCGM_FE_GPU, gpuId,
                                           fieldId, count, startTs, endTs, order, values);
}

static dcgmReturn_t tsapiEngineWatchFieldValue(dcgmHandle_t pDcgmHandle, int gpuId, unsigned short fieldId, long long updateFreq,
        double maxKeepAge, int maxKeepSamples)
{
    return helperWatchFieldValue(pDcgmHandle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepSamples);
}

static dcgmReturn_t tsapiEngineUnwatchFieldValue(dcgmHandle_t pDcgmHandle, int gpuId, unsigned short fieldId, int clearCache)
{
    return helperUnwatchFieldValue(pDcgmHandle, gpuId, fieldId, clearCache);
}

static dcgmReturn_t tsapiEngineUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate)
{
    return helperUpdateAllFields(pDcgmHandle, waitForUpdate);
}

static dcgmReturn_t tsapiEnginePolicyGet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, int count, dcgmPolicy_t policy[], dcgmStatus_t statusHandle)
{
    return helperPolicyGet(pDcgmHandle, groupId, count, policy, statusHandle);
}

static dcgmReturn_t tsapiEnginePolicySet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPolicy_t *policy, dcgmStatus_t statusHandle)
{
    return helperPolicySet(pDcgmHandle, groupId, policy, statusHandle);
}

static dcgmReturn_t tsapiEnginePolicyTrigger(dcgmHandle_t pDcgmHandle)
{

    /* Policy management is now edge-triggered, so this function has no reason
       to exist anymore. Also, it only ever worked in the embedded case. 
       Just returning OK to not break old clients. */
    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiEnginePolicyRegister(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPolicyCondition_t condition,
        fpRecvUpdates beginCallback, fpRecvUpdates finishCallback)
{
    return helperPolicyRegister(pDcgmHandle, groupId, condition, beginCallback, finishCallback);
}

static dcgmReturn_t tsapiEnginePolicyUnregister(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPolicyCondition_t condition)
{
    return helperPolicyUnregister(pDcgmHandle, groupId, condition);
}

static dcgmReturn_t tsapiEngineGetFieldValuesSince(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, long long sinceTimestamp,
                                                   unsigned short *fieldIds, int numFieldIds, long long *nextSinceTimestamp,
                                                   dcgmFieldValueEnumeration_f enumCB, void *userData)
{
    return helperGetFieldValuesSince(pDcgmHandle, groupId, sinceTimestamp, fieldIds, numFieldIds,
            nextSinceTimestamp, enumCB, userData);
}

static dcgmReturn_t tsapiEngineGetValuesSince(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId,
                                              dcgmFieldGrp_t fieldGroupId,
                                              long long sinceTimestamp, long long *nextSinceTimestamp,
                                              dcgmFieldValueEnumeration_f enumCB, void *userData)
{
    return helperGetValuesSince(pDcgmHandle, groupId, fieldGroupId, sinceTimestamp,
                                nextSinceTimestamp, enumCB, 0, userData);
}

static dcgmReturn_t tsapiEngineGetValuesSince_v2(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId,
                                                 dcgmFieldGrp_t fieldGroupId,
                                                 long long sinceTimestamp, long long *nextSinceTimestamp,
                                                 dcgmFieldValueEntityEnumeration_f enumCB, void *userData)
{
    return helperGetValuesSince(pDcgmHandle, groupId, fieldGroupId, sinceTimestamp,
                                nextSinceTimestamp, 0, enumCB, userData);
}

static dcgmReturn_t tsapiEngineGetLatestValues(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId,
        dcgmFieldValueEnumeration_f enumCB, void *userData)
{
    return helperGetLatestValues(pDcgmHandle, groupId, fieldGroupId, enumCB, 0, userData);
}

static dcgmReturn_t tsapiEngineGetLatestValues_v2(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId,
        dcgmFieldValueEntityEnumeration_f enumCB, void *userData)
{
    return helperGetLatestValues(pDcgmHandle, groupId, fieldGroupId, 0, enumCB, userData);
}

static dcgmReturn_t tsapiEngineHealthSet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t systems)
{
    return helperHealthSet(pDcgmHandle, groupId, systems);
}

static dcgmReturn_t tsapiEngineHealthGet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t *systems)
{
    return helperHealthGet(pDcgmHandle, groupId, systems);
}

static dcgmReturn_t tsapiEngineHealthCheck(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthResponse_t *response)
{
    if(!response)
    {
        PRINT_ERROR("", "tsapiEngineHealthCheck: response was missing.");
        return DCGM_ST_BADPARAM;
    }
    
    if (response->version < dcgmHealthResponse_version1 ||
            response->version > dcgmHealthResponse_version)
    {
        PRINT_ERROR("%X", "tsapiEngineHealthCheck got bad version x%X", response->version);
        return DCGM_ST_VER_MISMATCH;
    }

    if(response->version == dcgmHealthResponse_version1)
        return helperHealthCheckV1(pDcgmHandle, groupId, (dcgmHealthResponse_v1 *)response);
    else if (response->version == dcgmHealthResponse_version2)
        return helperHealthCheckV2(pDcgmHandle, groupId, (dcgmHealthResponse_v2 *)response);
    else /* -> version checked above */
    {
        return helperHealthCheckV3(pDcgmHandle, groupId, reinterpret_cast<dcgmHealthResponse_v3 *>(response));
    }
}

static dcgmReturn_t tsapiEngineActiolwalidate_v2(dcgmHandle_t pDcgmHandle, dcgmRunDiag_t *drd, dcgmDiagResponse_t *response)
{
    return helperActionManager(pDcgmHandle, drd, DCGM_POLICY_ACTION_NONE, response);
}

static dcgmReturn_t tsapiEngineActiolwalidate(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPolicyValidation_t validate, dcgmDiagResponse_t *response)
{
    dcgmRunDiag_t drd = {0};
    drd.version = dcgmRunDiag_version1;
    drd.validate = validate;
    drd.groupId = groupId;
	return helperActionManager(pDcgmHandle, &drd, DCGM_POLICY_ACTION_NONE, response);
}

static dcgmReturn_t tsapiEngineRunDiagnostic(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmDiagnosticLevel_t diagLevel,
                                             dcgmDiagResponse_t *diagResponse)
{
    dcgmPolicyValidation_t validation = DCGM_POLICY_VALID_NONE;
    dcgmRunDiag_t          drd = {0};

    if(!diagResponse)
        return DCGM_ST_BADPARAM;

    if(!diagResponse->version)
    {
        PRINT_DEBUG("", "Version missing");
        return DCGM_ST_VER_MISMATCH;
    }

    /* diagLevel -> validation */
    switch(diagLevel)
    {
        case DCGM_DIAG_LVL_SHORT:
            validation = DCGM_POLICY_VALID_SV_SHORT;
            break;

        case DCGM_DIAG_LVL_MED:
            validation = DCGM_POLICY_VALID_SV_MED;
            break;

        case DCGM_DIAG_LVL_LONG:
            validation = DCGM_POLICY_VALID_SV_LONG;
            break;

        case DCGM_DIAG_LVL_ILWALID:
        default:
            PRINT_ERROR("%d", "Invalid diagLevel %d", (int)diagLevel);
            return DCGM_ST_BADPARAM;
    }
    
    drd.version = dcgmRunDiag_version;
    drd.groupId = groupId;
    drd.validate = validation;

    return helperActionManager(pDcgmHandle, &drd, DCGM_POLICY_ACTION_NONE, diagResponse);
}

static dcgmReturn_t tsapiEngineStopDiagnostic(dcgmHandle_t pDcgmHandle)
{
    return helperStopDiag(pDcgmHandle);
}

static dcgmReturn_t tsapiEngineGetPidInfo(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPidInfo_t *pidInfo)
{
    return helperGetPidInfo(pDcgmHandle, groupId, pidInfo);
}

static dcgmReturn_t tsapiEngineWatchPidFields(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, long long updateFreq,
                                              double maxKeepAge, int maxKeepSamples)
{
    dcgmWatchPredefined_t watchPredef;

    memset(&watchPredef, 0, sizeof(watchPredef));
    watchPredef.version = dcgmWatchPredefined_version;
    watchPredef.watchPredefType = DCGM_WATCH_PREDEF_PID;
    watchPredef.groupId = groupId;
    watchPredef.updateFreq = updateFreq;
    watchPredef.maxKeepAge = maxKeepAge;
    watchPredef.maxKeepSamples = maxKeepSamples;

    return helperSendStructRequest(pDcgmHandle, lwcm::WATCH_PREDEFINED,
                                   -1, (int)(intptr_t)groupId, &watchPredef, sizeof(watchPredef));
}

static dcgmReturn_t tsapiEngineWatchJobFields(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, long long updateFreq,
                                              double maxKeepAge, int maxKeepSamples)
{
    dcgmWatchPredefined_t watchPredef;

    memset(&watchPredef, 0, sizeof(watchPredef));
    watchPredef.version = dcgmWatchPredefined_version;
    watchPredef.watchPredefType = DCGM_WATCH_PREDEF_JOB;
    watchPredef.groupId = groupId;
    watchPredef.updateFreq = updateFreq;
    watchPredef.maxKeepAge = maxKeepAge;
    watchPredef.maxKeepSamples = maxKeepSamples;

    return helperSendStructRequest(pDcgmHandle, lwcm::WATCH_PREDEFINED,
                                   -1, (int)(intptr_t)groupId, &watchPredef, sizeof(watchPredef));
}


static dcgmReturn_t tsapiEngineJobStartStats(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, char jobId[64])
{
    LwcmProtobuf encodePrb;                 /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;                 /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;                /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;     /* Vector of proto commands. Used as output parameter */
    lwcm::Command   *pGroupCmd;             /* Temp reference to the command */
    dcgmReturn_t ret;

    if ((NULL == jobId) || (0 == jobId[0]))
    {
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::JOB_START_STATS, lwcm::OPERATION_GROUP_ENTITIES, (intptr_t)groupId, 0);
    if (NULL == pCmdTemp) {
        return DCGM_ST_GENERIC_ERROR;
    }

    pCmdTemp->add_arg()->set_str(jobId);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    /* Return the global status returned from the operation at the hostengine */
    return (dcgmReturn_t)pGroupCmd->status();
}

static dcgmReturn_t tsapiEngineJobStopStats(dcgmHandle_t pDcgmHandle, char jobId[64])
{
    LwcmProtobuf encodePrb;                 /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;                 /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;                /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;     /* Vector of proto commands. Used as output parameter */
    lwcm::Command   *pGroupCmd;             /* Temp reference to the command */
    dcgmReturn_t ret;

    if ((NULL == jobId) || (0 == jobId[0]))
    {
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::JOB_STOP_STATS, lwcm::OPERATION_SYSTEM, 0, 0);
    if (NULL == pCmdTemp) {
        return DCGM_ST_GENERIC_ERROR;
    }

    pCmdTemp->add_arg()->set_str(jobId);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    /* Return the global status returned from the operation at the hostengine */
    return (dcgmReturn_t)pGroupCmd->status();
}

static dcgmReturn_t tsapiEngineJobGetStats(dcgmHandle_t pDcgmHandle, char jobId[64], dcgmJobInfo_t *pJobInfo)
{
    LwcmProtobuf encodePrb;                 /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;                 /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;                /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;     /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if((NULL == jobId) || (NULL == pJobInfo))
        return DCGM_ST_BADPARAM;

    /* Valid version can't be 0 or just any random number  */
    if(pJobInfo->version != dcgmJobInfo_version)
    {
        PRINT_DEBUG("", "Version Mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    if ((0 == jobId[0]))
    {
        PRINT_DEBUG("", "Job ID was NULL");
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::JOB_GET_INFO, lwcm::OPERATION_SYSTEM, 0, 0);
    if (NULL == pCmdTemp) {
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Add the args to the command to be sent over the network */
    pCmdTemp->add_arg()->set_str(jobId);
    pCmdTemp->add_arg()->set_blob(pJobInfo, sizeof(*pJobInfo));

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Check the status of the LWCM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK) {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    if(!vecCmdsRef[0]->arg_size())
    {
        PRINT_ERROR("", "Arg size of 0 unexpected");
        return DCGM_ST_GENERIC_ERROR;
    }
    
    if(!vecCmdsRef[0]->arg(0).has_str())
    {
        PRINT_ERROR("", "Response missing job id");
        return DCGM_ST_GENERIC_ERROR;
    }    

    if(!vecCmdsRef[0]->arg(1).has_blob())
    {
        PRINT_ERROR("", "Response missing blob");
        return DCGM_ST_GENERIC_ERROR;
    }

    if(vecCmdsRef[0]->arg(1).blob().size() > sizeof(*pJobInfo))
    {
        PRINT_ERROR("%d %d", "Returned blob size %d > structSize %d",
                (int)vecCmdsRef[0]->arg(0).blob().size(), (int)sizeof(*pJobInfo));
        return DCGM_ST_GENERIC_ERROR;
    }

    memcpy(pJobInfo, (void *)vecCmdsRef[0]->arg(1).blob().c_str(),
            vecCmdsRef[0]->arg(1).blob().size());

    return DCGM_ST_OK;    
}

static dcgmReturn_t tsapiEngineJobRemove(dcgmHandle_t pDcgmHandle, char jobId[64])
{
    LwcmProtobuf encodePrb;                 /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;                 /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;                /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;     /* Vector of proto commands. Used as output parameter */
    lwcm::Command   *pGroupCmd;             /* Temp reference to the command */
    dcgmReturn_t ret;

    if ((NULL == jobId) || (0 == jobId[0]))
    {
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::JOB_REMOVE, lwcm::OPERATION_SYSTEM, 0, 0);
    if (NULL == pCmdTemp) {
        return DCGM_ST_GENERIC_ERROR;
    }

    pCmdTemp->add_arg()->set_str(jobId);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    /* Return the global status returned from the operation at the hostengine */
    return (dcgmReturn_t)pGroupCmd->status();
}

static dcgmReturn_t tsapiEngineJobRemoveAll(dcgmHandle_t pDcgmHandle)
{
    LwcmProtobuf encodePrb;                 /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;                 /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;                /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;     /* Vector of proto commands. Used as output parameter */
    lwcm::Command   *pGroupCmd;             /* Temp reference to the command */
    dcgmReturn_t ret;

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::JOB_REMOVE_ALL, lwcm::OPERATION_SYSTEM, 0, 0);
    if (NULL == pCmdTemp) {
        return DCGM_ST_GENERIC_ERROR;
    }

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    /* Return the global status returned from the operation at the hostengine */
    return (dcgmReturn_t)pGroupCmd->status();
}

static dcgmReturn_t tsapiEngineGetDeviceTopology(dcgmHandle_t pDcgmHandle, unsigned int gpuId, dcgmDeviceTopology_t* deviceTopology)
{
	dcgmTopology_t groupTopology;
	dcgmAffinity_t groupAffinity;
	dcgmReturn_t ret = DCGM_ST_OK;

	unsigned int numGpusInTopology = 0;

	memset(&groupTopology, 0, sizeof(groupTopology));
	memset(&groupAffinity, 0, sizeof(groupAffinity));
	deviceTopology->version = dcgmDeviceTopology_version;

	ret = helperGetTopologyPci(pDcgmHandle, (dcgmGpuGrp_t) DCGM_GROUP_ALL_GPUS, &groupTopology); // retrieve the topology for the entire system
	if (DCGM_ST_OK != ret)
	{
	    PRINT_DEBUG("%d", "helperGetTopologyPci returned %d", (int)ret);
		return ret;
	}

    // numElements from topology is going to be zero here if DCGM_ST_NO_DATA is returned

	// go through the entire topology looking for a match of gpuId in their gpuA or gpuB of the paths structs
	for (unsigned int index = 0; index < groupTopology.numElements; index++)
	{
		unsigned int gpuA = groupTopology.element[index].dcgmGpuA;
		unsigned int gpuB = groupTopology.element[index].dcgmGpuB;

		if (gpuA == gpuId || gpuB == gpuId)
		{
			deviceTopology->gpuPaths[numGpusInTopology].gpuId = (gpuA == gpuId) ? gpuB : gpuA;
			deviceTopology->gpuPaths[numGpusInTopology].path = groupTopology.element[index].path;
            // the GPU topo info is store always lowGpuId connected to highGpuId 
            // i.e. 0->1, 1->2, 1->4 ... never 3->1.
            // thus if gpuId == gpuA then we need to use the AtoBLwLinkIds entry as GPU A will always be a lower number
            // if gpuId == gpuB then use BtoALwLinkIds.
            if (gpuA == gpuId)
                deviceTopology->gpuPaths[numGpusInTopology].localLwLinkIds = groupTopology.element[index].AtoBLwLinkIds;
            else
                deviceTopology->gpuPaths[numGpusInTopology].localLwLinkIds = groupTopology.element[index].BtoALwLinkIds;
			numGpusInTopology++;
		}
	}
    deviceTopology->numGpus = numGpusInTopology;

	// it is ok at this point to have numGpusInTopology == 0 because there may only be one GPU on the system.

	ret = helperGetTopologyAffinity(pDcgmHandle, (dcgmGpuGrp_t) DCGM_GROUP_ALL_GPUS, &groupAffinity);
	if (DCGM_ST_OK != ret)
		return ret;

	bool found = false;
	for (unsigned int index = 0; index < groupAffinity.numGpus; index++)
	{
		if (groupAffinity.affinityMasks[index].dcgmGpuId == gpuId)
		{
			found = true;
			memcpy(deviceTopology->cpuAffinityMask, groupAffinity.affinityMasks[index].bitmask, sizeof(unsigned long) * DCGM_AFFINITY_BITMASK_ARRAY_SIZE);
			break;
		}
	}
	if (!found)  // the gpuId was illegal as ALL GPUs should have some affinity
		return DCGM_ST_BADPARAM;

	return ret;
}

/* 
 * Compare two topologies. Returns -1 if a is better than b. 0 if same. 1 if b is better than a. 
 * This is meant to be used in a qsort() callback, resulting in the elements being sorted in descending order of P2P speed
 */
static int dcgmGpuTopologyLevelCmpCB(dcgmGpuTopologyLevel_t a, dcgmGpuTopologyLevel_t b)
{
    //This code has to be complicated because a lower PCI value is better 
    //but a higher LwLink value is better. All LwLinks are better than all PCI
    unsigned int lwLinkPathA = DCGM_TOPOLOGY_PATH_LWLINK(a);
    unsigned int pciPathA = DCGM_TOPOLOGY_PATH_PCI(a);

    unsigned int lwLinkPathB = DCGM_TOPOLOGY_PATH_LWLINK(b);
    unsigned int pciPathB = DCGM_TOPOLOGY_PATH_PCI(b);

    /* If both have LwLinks, compare those. More is better */
    if(lwLinkPathA && lwLinkPathB)
    {
        if(lwLinkPathA > lwLinkPathB)
            return -1;
        else if(lwLinkPathA < lwLinkPathB)
            return 1;
        else
            return 0; /* Ignore PCI topology if we have LwLink */
    }
    
    /* If one or the other has LwLink, that one is faster */
    if(lwLinkPathA && !lwLinkPathB)
        return -1;
    if(lwLinkPathB && !lwLinkPathA)
        return 1;

    /* Neither has LwLink. Compare the PCI topologies. Less is better */
    if(pciPathA < pciPathB)
        return -1;
    else if(pciPathA > pciPathB)
        return 1;
    
    return 0;
}

static dcgmReturn_t tsapiEngineGroupTopology(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmGroupTopology_t* groupTopology)
{
	dcgmTopology_t topology;
	dcgmAffinity_t affinity;
	dcgmReturn_t ret = DCGM_ST_OK;

	groupTopology->version = dcgmGroupTopology_version;

	ret = helperGetTopologyPci(pDcgmHandle, groupId, &topology); // retrieve the topology for this group
	if (DCGM_ST_OK != ret && DCGM_ST_NO_DATA != ret)
		return ret;

    // numElements from topology is going to be zero here if DCGM_ST_NO_DATA is returned

	dcgmGpuTopologyLevel_t slowestPath = (dcgmGpuTopologyLevel_t) 0;

	// go through the entire topology looking for the slowest path
	for (unsigned int index = 0; index < topology.numElements; index++)
	{
        /* If slowest path hasn't been set yet or slowest path is better than what we're comparing to */
        if(!slowestPath || 0 > dcgmGpuTopologyLevelCmpCB(slowestPath, topology.element[index].path))
            slowestPath = topology.element[index].path;
	}

	groupTopology->slowestPath = slowestPath;

	ret = helperGetTopologyAffinity(pDcgmHandle, groupId, &affinity);
	if (DCGM_ST_OK != ret)
		return ret;

	bool foundDifference = false;

	// iterate through each element of the bitmask OR'ing them together and locating if there was a difference
	for (unsigned int i = 0; i < DCGM_AFFINITY_BITMASK_ARRAY_SIZE; i++)
	{
		unsigned long overallMask = 0;
		for (unsigned int index = 0; index < affinity.numGpus; index++)
		{
		
			overallMask |= affinity.affinityMasks[index].bitmask[i];
			if (overallMask != affinity.affinityMasks[index].bitmask[i])
				foundDifference = true;
		}
		groupTopology->groupCpuAffinityMask[i] = overallMask;
	}

	groupTopology->numaOptimalFlag = (foundDifference) ? 0 : 1;
	return ret;
}

static dcgmReturn_t tsapiMetadataToggleState(dcgmHandle_t dcgmHandle, dcgmIntrospectState_t enabledStatus)
{
    dcgm_introspect_msg_toggle_t msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_STATE_TOGGLE;
    msg.header.version = dcgm_introspect_msg_toggle_version;
    msg.enabledStatus = enabledStatus;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    return dcgmReturn;
}

static dcgmReturn_t tsapiMetadataStateSetRunInterval(dcgmHandle_t dcgmHandle,
                                                unsigned int runIntervalMs)
{
    dcgm_introspect_msg_set_interval_t msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_STATE_SET_RUN_INTERVAL;
    msg.header.version = dcgm_introspect_msg_set_interval_version;

    msg.runIntervalMs = runIntervalMs;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    return dcgmReturn;
}

static dcgmReturn_t tsapiMetadataUpdateAll(dcgmHandle_t dcgmHandle, int waitForUpdate)
{
    dcgm_introspect_msg_update_all_t msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_UPDATE_ALL;
    msg.header.version = dcgm_introspect_msg_update_all_version;

    msg.waitForUpdate = waitForUpdate;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    return dcgmReturn;
}

static dcgmReturn_t tsapiIntrospectGetFieldsMemoryUsage(dcgmHandle_t dcgmHandle,
                                                  dcgmIntrospectContext_t *context,
                                                  dcgmIntrospectFullMemory_t *memoryInfo,
                                                  int waitIfNoData)
{
    dcgm_introspect_msg_fields_mem_usage_t msg;
    dcgmReturn_t dcgmReturn;

    if ((!context) || (!memoryInfo))
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    /* Valid version can't be 0 or just any random number  */
    if(context->version != dcgmIntrospectContext_version)
    {
        PRINT_DEBUG("", "Version Mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    /* Valid version can't be 0 or just any random number  */
    if(memoryInfo->version != dcgmIntrospectFullMemory_version)
    {
        PRINT_DEBUG("", "Version Mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    if(context->introspectLvl <= DCGM_INTROSPECT_LVL_ILWALID || 
       context->introspectLvl > DCGM_INTROSPECT_LVL_ALL_FIELDS)
    {
        PRINT_ERROR("", "Bad introspection level");
        return DCGM_ST_BADPARAM;
    }
    
    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_FIELDS_MEM_USAGE;
    msg.header.version = dcgm_introspect_msg_fields_mem_usage_version;

    memcpy(&msg.context, context, sizeof(msg.context));
    memcpy(&msg.memoryInfo, memoryInfo, sizeof(msg.memoryInfo));
    msg.waitIfNoData = waitIfNoData;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);

    memcpy(memoryInfo, &msg.memoryInfo, sizeof(msg.memoryInfo));
    return dcgmReturn;
}

static dcgmReturn_t tsapiIntrospectGetFieldMemoryUsage(dcgmHandle_t dcgmHandle,
                                                       unsigned short fieldId,
                                                       dcgmIntrospectFullMemory_t *memoryInfo,
                                                       int waitIfNoData)
{
    dcgmIntrospectContext_t context;
    context.version = dcgmIntrospectContext_version;
    context.introspectLvl = DCGM_INTROSPECT_LVL_FIELD;
    context.fieldId = fieldId;


    return tsapiIntrospectGetFieldsMemoryUsage(dcgmHandle, &context,
                                               memoryInfo,
                                               waitIfNoData);
}

static dcgmReturn_t tsapiIntrospectGetFieldsExecTime(dcgmHandle_t dcgmHandle,
                                               dcgmIntrospectContext_t *context,
                                               dcgmIntrospectFullFieldsExecTime_t *execTime,
                                               int waitIfNoData)
{
    dcgm_introspect_msg_fields_exec_time_t msg;
    dcgmReturn_t dcgmReturn;

    if ((!context) || (!execTime))
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    /* Valid version can't be 0 or just any random number  */
    if(context->version != dcgmIntrospectContext_version)
    {
        PRINT_DEBUG("", "Version Mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    /* Valid version can't be 0 or just any random number  */
    if(execTime->version != dcgmIntrospectFullFieldsExecTime_version)
    {
        PRINT_DEBUG("", "Version Mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    if(context->introspectLvl <= DCGM_INTROSPECT_LVL_ILWALID || 
       context->introspectLvl > DCGM_INTROSPECT_LVL_ALL_FIELDS)
    {
        PRINT_ERROR("", "Bad introspection level");
        return DCGM_ST_BADPARAM;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_FIELDS_EXEC_TIME;
    msg.header.version = dcgm_introspect_msg_fields_exec_time_version;

    memcpy(&msg.context, context, sizeof(msg.context));
    memcpy(&msg.execTime, execTime, sizeof(msg.execTime));
    msg.waitIfNoData = waitIfNoData;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);

    memcpy(execTime, &msg.execTime, sizeof(msg.execTime));
    return dcgmReturn;
}

static dcgmReturn_t tsapiIntrospectGetFieldExecTime(dcgmHandle_t dcgmHandle,
                                                    unsigned short fieldId,
                                                    dcgmIntrospectFullFieldsExecTime_t *execTime,
                                                    int waitIfNoData)
{
    dcgmIntrospectContext_t context;
    context.version = dcgmIntrospectContext_version;
    context.introspectLvl = DCGM_INTROSPECT_LVL_FIELD;
    context.fieldId = fieldId;

    return tsapiIntrospectGetFieldsExecTime(dcgmHandle, &context,
                                            execTime, waitIfNoData);
}

static dcgmReturn_t tsapiIntrospectGetHostengineMemoryUsage(dcgmHandle_t dcgmHandle,
                                                            dcgmIntrospectMemory_t *memoryInfo,
                                                            int waitIfNoData)
{
    dcgm_introspect_msg_he_mem_usage_t msg;
    dcgmReturn_t dcgmReturn;

    if(!memoryInfo)
        return DCGM_ST_BADPARAM;
    if(memoryInfo->version != dcgmIntrospectMemory_version)
    {
        PRINT_ERROR("%X %X", "Version mismatch x%X != x%X", 
                    memoryInfo->version, dcgmIntrospectMemory_version);
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE;
    msg.header.version = dcgm_introspect_msg_he_mem_usage_version;
    msg.waitIfNoData = waitIfNoData;
    memcpy(&msg.memoryInfo, memoryInfo, sizeof(*memoryInfo));

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    
    /* Copy the response back over the request */
    memcpy(memoryInfo, &msg.memoryInfo, sizeof(*memoryInfo));
    return dcgmReturn;
}

static dcgmReturn_t tsapiIntrospectGetHostengineCpuUtilization(dcgmHandle_t dcgmHandle,
                                                               dcgmIntrospectCpuUtil_t *cpuUtil,
                                                               int waitIfNoData)
{
    dcgm_introspect_msg_he_cpu_util_t msg;
    dcgmReturn_t dcgmReturn;

    if(!cpuUtil)
        return DCGM_ST_BADPARAM;
    if(cpuUtil->version != dcgmIntrospectCpuUtil_version)
    {
        PRINT_ERROR("%X %X", "Version mismatch x%X != x%X", 
                    cpuUtil->version, dcgmIntrospectCpuUtil_version);
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL;
    msg.header.version = dcgm_introspect_msg_he_cpu_util_version;

    msg.waitIfNoData = waitIfNoData;

    memcpy(&msg.cpuUtil, cpuUtil, sizeof(*cpuUtil));
    
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    
    /* Copy the response back over the request */
    memcpy(cpuUtil, &msg.cpuUtil, sizeof(*cpuUtil));
    return dcgmReturn;
}

static dcgmReturn_t tsapiSelectGpusByTopology(dcgmHandle_t pDcgmHandle, uint64_t inputGpuIds, uint32_t numGpus,
                                              uint64_t *outputGpuIds, uint64_t hintFlags)
{
    return helperSelectGpusByTopology(pDcgmHandle, inputGpuIds, numGpus, outputGpuIds, hintFlags);
}

static dcgmReturn_t tsapiGetFieldSummary(dcgmHandle_t pDcgmHandle, dcgmFieldSummaryRequest_t *request)
{
    return helperGetFieldSummary(pDcgmHandle, request);
}

/*****************************************************************************/
static dcgmReturn_t tsapiModuleBlacklist(dcgmHandle_t pDcgmHandle, dcgmModuleId_t moduleId)
{

    dcgmModuleBlacklist_v1 msg;

    if(moduleId <= DcgmModuleIdCore || moduleId >= DcgmModuleIdCount)
    {
        PRINT_ERROR("%u", "Bad module ID %u", moduleId);
        return DCGM_ST_BADPARAM;
    }

    memset(&msg, 0, sizeof(msg));
    msg.version = dcgmModuleBlacklist_version1;
    msg.moduleId = moduleId;

    return helperSendStructRequest(pDcgmHandle, lwcm::MODULE_BLACKLIST, 
                                  -1, -1, &msg, sizeof(msg));
}

/*****************************************************************************/
static dcgmReturn_t tsapiModuleGetStatuses(dcgmHandle_t pDcgmHandle, dcgmModuleGetStatuses_t *moduleStatuses)
{

    if(!moduleStatuses)
    {
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }
    
    if(moduleStatuses->version != dcgmModuleGetStatuses_version)
    {
        PRINT_ERROR("%X %X", "Module status version x%X != x%X", 
                    moduleStatuses->version, dcgmModuleGetStatuses_version);
    }

    return helperSendStructRequest(pDcgmHandle, lwcm::MODULE_GET_STATUSES, 
                                  -1, -1, moduleStatuses, sizeof(*moduleStatuses));
}


/*****************************************************************************/
dcgmReturn_t tsapiProfGetSupportedMetricGroups(dcgmHandle_t dcgmHandle, 
                                               dcgmProfGetMetricGroups_t *metricGroups)
{
    dcgm_profiling_msg_get_mgs_t msg;
    dcgmReturn_t dcgmReturn;

    if(!metricGroups)
        return DCGM_ST_BADPARAM;
    if(metricGroups->version != dcgmProfGetMetricGroups_version)
    {
        PRINT_ERROR("%X %X", "Version mismatch x%X != x%X", 
                    metricGroups->version, dcgmProfGetMetricGroups_version);
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdProfiling;
    msg.header.subCommand = DCGM_PROFILING_SR_GET_MGS;
    msg.header.version = dcgm_profiling_msg_get_mgs_version;

    memcpy(&msg.metricGroups, metricGroups, sizeof(*metricGroups));
    
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    
    /* Copy the response back over the request */
    memcpy(metricGroups, &msg.metricGroups, sizeof(*metricGroups));
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t tsapiProfWatchFields(dcgmHandle_t dcgmHandle, 
                                       dcgmProfWatchFields_t *watchFields)
{
    dcgm_profiling_msg_watch_fields_t msg;
    dcgmReturn_t dcgmReturn;

    if(!watchFields)
        return DCGM_ST_BADPARAM;
    if(watchFields->version != dcgmProfWatchFields_version)
    {
        PRINT_ERROR("%X %X", "Version mismatch x%X != x%X", 
                    watchFields->version, dcgmProfWatchFields_version);
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdProfiling;
    msg.header.subCommand = DCGM_PROFILING_SR_WATCH_FIELDS;
    msg.header.version = dcgm_profiling_msg_watch_fields_version;

    memcpy(&msg.watchFields, watchFields, sizeof(*watchFields));
    
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    
    /* Copy the response back over the request */
    memcpy(watchFields, &msg.watchFields, sizeof(*watchFields));
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t tsapiProfUnwatchFields(dcgmHandle_t dcgmHandle, 
                                         dcgmProfUnwatchFields_t *unwatchFields)
{
    dcgm_profiling_msg_unwatch_fields_t msg;
    dcgmReturn_t dcgmReturn;

    if(!unwatchFields)
        return DCGM_ST_BADPARAM;
    if(unwatchFields->version != dcgmProfUnwatchFields_version)
    {
        PRINT_ERROR("%X %X", "Version mismatch x%X != x%X", 
                    unwatchFields->version, dcgmProfUnwatchFields_version);
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdProfiling;
    msg.header.subCommand = DCGM_PROFILING_SR_UNWATCH_FIELDS;
    msg.header.version = dcgm_profiling_msg_unwatch_fields_version;

    memcpy(&msg.unwatchFields, unwatchFields, sizeof(*unwatchFields));
    
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    
    /* Copy the response back over the request */
    memcpy(unwatchFields, &msg.unwatchFields, sizeof(*unwatchFields));
    return dcgmReturn;
}

/*****************************************************************************/
static
dcgmReturn_t helperProfPauseResume(dcgmHandle_t dcgmHandle, bool pause)
{
    dcgm_profiling_msg_pause_resume_t msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdProfiling;
    msg.header.subCommand = DCGM_PROFILING_SR_PAUSE_RESUME;
    msg.header.version = dcgm_profiling_msg_pause_resume_version;
    msg.pause = pause;
    
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t tsapiProfPause(dcgmHandle_t dcgmHandle)
{
    return helperProfPauseResume(dcgmHandle, true);
}

/*****************************************************************************/
dcgmReturn_t tsapiProfResume(dcgmHandle_t dcgmHandle)
{
    return helperProfPauseResume(dcgmHandle, false);
}

/*****************************************************************************/
dcgmReturn_t tsapiVersionInfo(dcgmVersionInfo_t* pVersionInfo)
{
    return GetVersionInfo(pVersionInfo);
}

/*****************************************************************************/
/*****************************************************************************/
dcgmReturn_t tsapiDcgmStartEmbedded(dcgmOperationMode_t opMode, dcgmHandle_t *pDcgmHandle)
{
    if(NULL == pDcgmHandle)
        return DCGM_ST_BADPARAM;
    if ((opMode != DCGM_OPERATION_MODE_AUTO) && (opMode != DCGM_OPERATION_MODE_MANUAL))
        return DCGM_ST_BADPARAM;
    if(!g_dcgmGlobals.isInitialized)
    {
        PRINT_ERROR("", "tsapiDcgmStartEmbedded before tsapiDcgmInit()");
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmGlobalsLock();

    /* Check again after lock */
    if(!g_dcgmGlobals.isInitialized)
    {
        dcgmGlobalsUnlock();
        PRINT_ERROR("", "tsapiDcgmStartEmbedded before tsapiDcgmInit() after lock");
        return DCGM_ST_UNINITIALIZED;
    }

    /* See if the host engine is running already */
    void *pHostEngineInstance = LwcmHostEngineHandler::Instance();
    if(pHostEngineInstance)
    {
        g_dcgmGlobals.embeddedEngineStarted = 1; /* No harm in making sure this is true */
        dcgmGlobalsUnlock();
        PRINT_DEBUG("", "tsapiDcgmStartEmbedded(): host engine was already running");
        return DCGM_ST_OK;
    }

    pHostEngineInstance = LwcmHostEngineHandler::Init(opMode);
    if (NULL == pHostEngineInstance)
    {
        dcgmGlobalsUnlock();
        PRINT_ERROR("", "LwcmHostEngineHandler::Init failed");
        return DCGM_ST_INIT_ERROR;
    }

    g_dcgmGlobals.embeddedEngineStarted = 1;

    dcgmGlobalsUnlock();

    *pDcgmHandle = (dcgmHandle_t)DCGM_EMBEDDED_HANDLE;
    PRINT_DEBUG("", "tsapiDcgmStartEmbedded(): Embedded host engine started");

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t tsapiDcgmStopEmbedded(dcgmHandle_t pDcgmHandle)
{
    if(!g_dcgmGlobals.isInitialized)
    {
        PRINT_ERROR("", "tsapiDcgmStopEmbedded before tsapiDcgmInit()");
        return DCGM_ST_UNINITIALIZED;
    }
    if(pDcgmHandle != (dcgmHandle_t)DCGM_EMBEDDED_HANDLE)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmGlobalsLock();

    /* Check again after lock */
    if(!g_dcgmGlobals.isInitialized)
    {
        dcgmGlobalsUnlock();
        PRINT_ERROR("", "tsapiDcgmStopEmbedded before tsapiDcgmInit() after lock");
        return DCGM_ST_UNINITIALIZED;
    }

    if(g_dcgmGlobals.embeddedEngineStarted)
    {
        LwcmHostEngineHandler *heHandler = LwcmHostEngineHandler::Instance();

        if(!heHandler)
            PRINT_ERROR("", "embeddedEngineStarted was set but heHandler is NULL");
        else
        {
            // Ilwoke the cleanup method
            (void)LwcmHostEngineHandler::Instance()->Cleanup();
            PRINT_DEBUG("", "embedded host engine cleaned up");
        }
        g_dcgmGlobals.embeddedEngineStarted = 0;
    }

    dcgmGlobalsUnlock();

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t tsapiDcgmConnect(char *ipAddress, dcgmHandle_t *pDcgmHandle)
{
    dcgmConnectV2Params_t connectParams;

    /* Set up default parameters for dcgmConnect_v2 */
    memset(&connectParams, 0, sizeof(connectParams));
    connectParams.version = dcgmConnectV2Params_version;
    connectParams.persistAfterDisconnect = 0;

    return tsapiDcgmConnect_v2(ipAddress, &connectParams, pDcgmHandle);
}

/*****************************************************************************/
static dcgmReturn_t sendClientLogin(dcgmHandle_t dcgmHandle, dcgmConnectV2Params_t *connectParams)
{
    lwcm::ClientLogin *pClientLogin;     /* Protobuf Arg */
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */    
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    vector<lwcm::Command *> vecCmdsRef;  /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if (!connectParams)
        return DCGM_ST_BADPARAM;

    pClientLogin = new lwcm::ClientLogin;
    pClientLogin->set_persistafterdisconnect(connectParams->persistAfterDisconnect);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::CLIENT_LOGIN, lwcm::OPERATION_SYSTEM, 0, 0);
    if (NULL == pCmdTemp) {
        delete pClientLogin;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is 
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_clientlogin(pClientLogin);

    ret = processAtHostEngine(dcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret) 
    {
        PRINT_ERROR("%d", "Got st %d from processAtHostEngine", (int)ret);
        return ret;
    }

    /* Check the status of the LWCM command */
    ret = (dcgmReturn_t)vecCmdsRef[0]->status();
    if (ret != DCGM_ST_OK) 
    {
        PRINT_ERROR("%d", "Got st %d from vecCmdsRef[0]->status()", 
                    (int)vecCmdsRef[0]->status());
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t tsapiDcgmConnect_v2(char *ipAddress, dcgmConnectV2Params_t *connectParams, dcgmHandle_t *pDcgmHandle)
{
    dcgmReturn_t dcgmReturn;
    dcgmConnectV2Params_v2 paramsCopy;

    if(!ipAddress || !ipAddress[0] || !pDcgmHandle || !connectParams)
        return DCGM_ST_BADPARAM;
    if(!g_dcgmGlobals.isInitialized)
    {
        PRINT_ERROR("", "tsapiDcgmConnect_v2 before tsapiDcgmInit()");
        return DCGM_ST_UNINITIALIZED;
    }

    /* Handle the old version by copying its parameters to the new version and changing the 
       pointer to our local struct */
    if(connectParams->version == dcgmConnectV2Params_version1)
    {
        memset(&paramsCopy, 0, sizeof(paramsCopy));
        paramsCopy.version = dcgmConnectV2Params_version;
        paramsCopy.persistAfterDisconnect = connectParams->persistAfterDisconnect;
        /* Other fields default to 0 from the memset above */
        connectParams = &paramsCopy;
    }
    else if(connectParams->version != dcgmConnectV2Params_version)
    {
        PRINT_ERROR("%X %X", "dcgmConnect_v2 Version mismatch %X != %X",
                    connectParams->version, dcgmConnectV2Params_version);
        return DCGM_ST_VER_MISMATCH;
    }

    LwcmClientHandler *clientHandler = lwcmapiAcquireClientHandler(true);
    if(!clientHandler)
    {
        PRINT_ERROR("", "Unable to allocate client handler");
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Add connection to the client handler */
    dcgmReturn_t status = clientHandler->GetConnHandleForHostEngine(
                                ipAddress, pDcgmHandle,
                                connectParams->timeoutMs,
                                connectParams->addressIsUnixSocket ? true : false);
    lwcmapiReleaseClientHandler();
    if (DCGM_ST_OK != status)
    {
        PRINT_ERROR("%s %d", "GetConnHandleForHostEngine ip %s returned %d",
                    ipAddress, (int)status);
        return status;
    }

    PRINT_DEBUG("%s %p", "Connected to ip %s as dcgmHandle %p",
                ipAddress, *pDcgmHandle);
    
    /* Send our connection options to the host engine */
    dcgmReturn = sendClientLogin(*pDcgmHandle, connectParams);
    if(dcgmReturn != DCGM_ST_OK)
    {
        /* Abandon the connection if we can't login */
        PRINT_ERROR("%d %p", "Got error %d from sendClientLogin on connection %p. Abandoning connection.",
                    (int)dcgmReturn, *pDcgmHandle);
        return tsapiDcgmDisconnect(*pDcgmHandle);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t tsapiDcgmDisconnect(dcgmHandle_t pDcgmHandle)
{
    if(!g_dcgmGlobals.isInitialized)
    {
        PRINT_WARNING("", "tsapiDcgmDisconnect before tsapiDcgmInit()");
        /* Returning OK here to prevent errors from being logged from the 
           python framework when DcgmHandle objects are garbage collected after 
           dcgmShutdown has already been called. */
        return DCGM_ST_OK;
    }

    LwcmClientHandler *clientHandler = lwcmapiAcquireClientHandler(false);
    if(!clientHandler)
    {
        PRINT_WARNING("", "tsapiDcgmDisconnect called while client handler was not allocated.");
        /* Returning OK here to prevent errors from being logged from the 
           python framework when DcgmHandle objects are garbage collected after 
           dcgmShutdown has already been called. */
        return DCGM_ST_OK;
    }

    /* Actually close the connection */
    clientHandler->CloseConnForHostEngine(pDcgmHandle);

    lwcmapiReleaseClientHandler();

    PRINT_DEBUG("%p", "dcgmDisconnect closed connection with handle %p", pDcgmHandle);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t tsapiDcgmInit(void)
{
    if(g_dcgmGlobals.isInitialized)
    {
        PRINT_DEBUG("", "dcgmInit was already initialized");
        return DCGM_ST_OK;
    }

    dcgmGlobalsLock();

    /* Check again now that we have the lock */
    if(g_dcgmGlobals.isInitialized)
    {
        dcgmGlobalsUnlock();
        PRINT_DEBUG("", "dcgmInit was already initialized after lock");
        return DCGM_ST_OK;
    }

    /* globals are uninitialized. Zero the structure */
    memset(&g_dcgmGlobals, 0, sizeof(g_dcgmGlobals));

    dcgmLoggingInit((char *)DCGM_ELW_DBG_LVL, (char *)DCGM_ELW_DBG_APPEND,
                    (char *)DCGM_ELW_DBG_FILE, (char *)DCGM_ELW_DBG_FILE_ROTATE);
    PRINT_DEBUG("", "Logging initialized");
    g_dcgmGlobals.loggingIsInitialized = 1;

    int ret = DcgmFieldsInit();
    if (ret != DCGM_ST_OK)
    {
        /* Undo any initialization done above */
        loggingShutdown();
        g_dcgmGlobals.loggingIsInitialized = 0;

        dcgmGlobalsUnlock();
        PRINT_ERROR("", "DcgmFieldsInit failed");
        return DCGM_ST_INIT_ERROR;
    }
    g_dcgmGlobals.fieldsAreInitialized = 1;

    /* Fully-initialized. Mark structure as such */
    g_dcgmGlobals.isInitialized = 1;

    dcgmGlobalsUnlock();

    PRINT_DEBUG("", "dcgmInit was successful");
    return DCGM_ST_OK;
}

/*****************************************************************************/

dcgmReturn_t tsapiDcgmShutdown()
{
    if(!g_dcgmGlobals.isInitialized)
    {
        PRINT_DEBUG("", "dcgmShutdown called when DCGM was uninitialized.");
        return DCGM_ST_OK;
    }

    /* Clean up remote connections - must NOT have dcgmGlobalsLock() here or we will
       deadlock */
    PRINT_DEBUG("", "Before lwcmapiFreeClientHandler");
    lwcmapiFreeClientHandler();
    PRINT_DEBUG("", "After lwcmapiFreeClientHandler");

    dcgmGlobalsLock();

    if(!g_dcgmGlobals.isInitialized)
    {
        dcgmGlobalsUnlock();
        PRINT_DEBUG("", "dcgmShutdown called when DCGM was uninitialized - after lock.");
        return DCGM_ST_UNINITIALIZED;
    }

    if(g_dcgmGlobals.embeddedEngineStarted)
    {
        LwcmHostEngineHandler *heHandler = LwcmHostEngineHandler::Instance();

        if(!heHandler)
            PRINT_ERROR("", "embeddedEngineStarted was set but heHandler is NULL");
        else
        {
            // Ilwoke the cleanup method
            (void)LwcmHostEngineHandler::Instance()->Cleanup();
            PRINT_DEBUG("", "host engine cleaned up");
        }
        g_dcgmGlobals.embeddedEngineStarted = 0;
    }

    DcgmFieldsTerm();
    g_dcgmGlobals.fieldsAreInitialized = 0;

    loggingShutdown();
    g_dcgmGlobals.loggingIsInitialized = 0;

    g_dcgmGlobals.isInitialized = 0;

    dcgmGlobalsUnlock();

    PRINT_DEBUG("", "dcgmShutdown completed successfully");

    return DCGM_ST_OK;
}


/*****************************************************************************/
