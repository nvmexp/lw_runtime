//
// Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
/// @file


#include <atomic>
#ifdef __linux__
#include <sys/eventfd.h>
#include <unistd.h>
#else
#include <sys/neutrino.h>
#endif
#include "lwscistream_common.h"
#include "common_includes.h"
#include "glob_test_vars.h"
#include "lwscibuf_c2c_internal.h"
#include "lwscisync.h"

using namespace LwSciStream;

// Ipc channel endpoints
Endpoint ipcSrc, ipcDst;

// ipcSrc & ipcDst frames read ready status flags
static std::atomic<bool> readReadySrcFr;
static std::atomic<bool> readReadyDstFr;

// Ipc Block disconnection status flag
bool ipcDisconnect;

// ipcSrc & ipcDst events
uint32_t ipcSrcEvent;
uint32_t ipcDstEvent;

// ipcSrc & ipcDst frame buffers
uint8_t ipcSrcFrame[1024];
uint8_t ipcDstFrame[1024];


// Checks whether the event LwSciStreamEventType_Disconnected=0x004005
// has oclwrred in either of the Ipc Blocks and sets the ipcDisconnect flag.
static inline void check_ipcBlock_conn(uint8_t ipcFrame[]) {
    if(ipcFrame[8] == 0x05 && ipcFrame[9] == 0x40)
        ipcDisconnect = true;
}

struct LwSciLocalEvent {
    LwSciEventNotifier* eventNotifier;
    LwSciError (*Signal)(LwSciLocalEvent* thisLocalEvent);
    void (*Delete)(LwSciLocalEvent* thisLocalEvent);
};

void stub_Delete(LwSciEventService* thisEventService)
{
    return;
}

LwSciError LwSciLocalEvent_Signal(LwSciLocalEvent *thisLocalEvent) {
    return LwSciError_Success;
}

void LwSciLocalEvent_Delete(LwSciLocalEvent *thisLocalEvent) {}

LwSciError LwSciEventService_CreateLocalEvent(
        LwSciEventService *thisEventService,
        LwSciLocalEvent **newLocalEvent)
{
    struct LwSciLinuxUserEventNotifier *linuxNotifier = NULL;
    struct LwSciLinuxUserLocalEvent *linuxLocalEvent = NULL;
    struct LwSciLocalEvent *localEvent = NULL;

    localEvent = (LwSciLocalEvent *)malloc(sizeof(struct LwSciLocalEvent));

    localEvent->Signal = &LwSciLocalEvent_Signal;
    localEvent->Delete = &LwSciLocalEvent_Delete;
    localEvent->eventNotifier = nullptr;
    *newLocalEvent = localEvent;
    if (test_block.LwSciEventService_CreateLocalEvent_fail == true)
    {
        test_block.LwSciEventService_CreateLocalEvent_fail = false;
        return LwSciError_BadParameter;
    }
    return  LwSciError_Success;
}

LwSciError LwSciEventService_CreateLocalEvent_fail(
        LwSciEventService *thisEventService,
        LwSciLocalEvent **newLocalEvent)
{
    struct LwSciLinuxUserEventNotifier *linuxNotifier = NULL;
    struct LwSciLinuxUserLocalEvent *linuxLocalEvent = NULL;
    struct LwSciLocalEvent *localEvent = NULL;

    localEvent = (LwSciLocalEvent *)malloc(sizeof(struct LwSciLocalEvent));
    localEvent->eventNotifier = nullptr;
    localEvent->Signal = &LwSciLocalEvent_Signal;
    localEvent->Delete = &LwSciLocalEvent_Delete;

    *newLocalEvent = localEvent;
    return  LwSciError_BadParameter;
}

LwSciError stub_WaitForEvent(LwSciEventNotifier* eventNotifier, int64_t microseconds)
{
    return LwSciError_Success;
}

struct LwSciEventLoopService stub_evt_loop_serv = {
    .EventService = {
        .CreateNativeEventNotifier = nullptr,
        .CreateLocalEvent = LwSciEventService_CreateLocalEvent,
	.CreateTimerEvent = nullptr,
        .Delete = stub_Delete,
    },
	.CreateEventLoop = nullptr,
	.WaitForEvent = stub_WaitForEvent,
	.WaitForMultipleEvents = nullptr,
};

struct LwSciEventLoopService stub_evt_loop_serv_fail = {
    .EventService = {
        .CreateNativeEventNotifier = nullptr,
        .CreateLocalEvent = LwSciEventService_CreateLocalEvent_fail,
        .CreateTimerEvent = nullptr,
        .Delete = stub_Delete,
    },
        .CreateEventLoop = nullptr,
        .WaitForEvent = stub_WaitForEvent,
        .WaitForMultipleEvents = nullptr,
};

LwSciError LwSciEventLoopServiceCreate(size_t maxEventLoops, LwSciEventLoopService **newEventLoopService)
{
    if (maxEventLoops == 12345)
    {
        printf("maxEventLoops");
       *newEventLoopService = &stub_evt_loop_serv_fail;
    } else
    {
       *newEventLoopService = &stub_evt_loop_serv;
    }
    return LwSciError_Success;
}

void LwSciSyncAttrListFree(
    LwSciSyncAttrList attrList)
{
}

void LwSciSyncObjFree(
    LwSciSyncObj syncObj)
{
}

void LwSciBufAttrListFree(
    LwSciBufAttrList attrList)
{
}

void LwSciCommonSleepNs(uint64_t timeNs)
{
}

void LwSciSyncFenceClear(
     LwSciSyncFence* syncFence)
{
}

void LwSciBufObjFree(
      LwSciBufObj bufObj) {
}

LwSciError LwSciSyncAttrListCreate(
           LwSciSyncModule module,
           LwSciSyncAttrList* attrList)
{
    *attrList = 0xABCDEF;
    return LwSciError_Success;
}

LwSciError LwSciSyncAttrListSetAttrs(
           LwSciSyncAttrList attrList,
           const LwSciSyncAttrKeyValuePair* pairArray,
           size_t pairCount)
{
    return LwSciError_Success;
}

LwSciError LwSciSyncAttrListReconcile(
           const LwSciSyncAttrList inputArray[],
           size_t inputCount,
           LwSciSyncAttrList* newReconciledList,
           LwSciSyncAttrList* newConflictList)
{
    *newReconciledList = 0xABCDEF;
    *newConflictList = nullptr;
    return LwSciError_Success;
}

LwSciError LwSciSyncObjAlloc(
           LwSciSyncAttrList reconciledList,
           LwSciSyncObj *syncObj)
{
    *syncObj = 0xABCDEF;
    return LwSciError_Success;
}

LwSciError LwSciBufAttrListCreate(
           LwSciBufModule module,
           LwSciBufAttrList* newAttrList)
{
    *newAttrList = 0xABCDEF;
    return LwSciError_Success;
}

LwSciError LwSciBufAttrListSetAttrs(
     LwSciBufAttrList attrList,
     LwSciBufAttrKeyValuePair* pairArray,
     size_t pairCount)
{
    return LwSciError_Success;
}

LwSciError LwSciBufAttrListReconcile(
     const LwSciBufAttrList inputArray[],
     size_t inputCount,
     LwSciBufAttrList* newReconciledAttrList,
     LwSciBufAttrList* newConflictList)
{
    *newReconciledAttrList = 0xABCDEF;
    *newConflictList = nullptr;
    return LwSciError_Success;
}

LwSciError LwSciBufObjAlloc(
     LwSciBufAttrList reconciledAttrList,
     LwSciBufObj* bufObj)
{
    *bufObj = 0xABCDEF;
    return LwSciError_Success;
}

void LwSciBufModuleClose(
    LwSciBufModule module)
{
}

void LwSciSyncModuleClose(
    LwSciSyncModule module)
{
}

LwSciError LwSciSyncAttrListAppendUnreconciled(
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciSyncAttrList* newUnreconciledAttrList)
{
    *newUnreconciledAttrList = 0xABCDEF;
    return LwSciError_Success;
}

LwSciError LwSciBufAttrListAppendUnreconciled(
    const LwSciBufAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciBufAttrList* newUnreconciledAttrList)
{
    *newUnreconciledAttrList = 0xABCDEF;
    return LwSciError_Success;
}

LwSciError LwSciSyncAttrListIpcExportUnreconciled(
    const LwSciSyncAttrList unreconciledAttrListArray[],
    size_t unreconciledAttrListCount,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen)
{
    *descBuf = 0xABCDEF;
    if(test_lwscisync.LwSciSyncAttrListIpcExportUnreconciled_fail) {
        test_lwscisync.LwSciSyncAttrListIpcExportUnreconciled_fail = false;
        return LwSciError_ResourceError;
    }

    return LwSciError_Success;
}

void LwSciSyncAttrListFreeDesc(void *descBuf)
{
}

LwSciError LwSciSyncIpcExportAttrListAndObj(
    LwSciSyncObj syncObj,
    LwSciSyncAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    void** attrListAndObjDesc,
    size_t* attrListAndObjDescSize)
{
    *attrListAndObjDesc = 0xABCDEF;
    if(test_lwscisync.LwSciSyncIpcExportAttrListAndObj_fail) {
        test_lwscisync.LwSciSyncIpcExportAttrListAndObj_fail = false;
        return LwSciError_ResourceError;
    }

    return LwSciError_Success;
}

void LwSciSyncAttrListAndObjFreeDesc(
    void* attrListAndObjDescBuf)
{
}

LwSciError LwSciBufAttrListIpcExportReconciled(
    LwSciBufAttrList reconciledAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen)
{
    if (test_lwscibuf.LwSciBufAttrListIpcExportReconciled_blobData_null == true)
    {
        *descBuf = nullptr;
        test_lwscibuf.LwSciBufAttrListIpcExportReconciled_blobData_null = false;
    }
    else
    {
        *descBuf = 0xABCDEF;
    }

    if(test_lwscibuf.LwSciBufAttrListIpcExportReconciled_fail) {
        test_lwscibuf.LwSciBufAttrListIpcExportReconciled_fail = false;
        return LwSciError_ResourceError;
    }

    return LwSciError_Success;
}

void LwSciBufAttrListFreeDesc(
    void* descBuf)
{
}

LwSciError LwSciBufObjIpcExport(
    LwSciBufObj bufObj,
    LwSciBufAttrValAccessPerm accPerm,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufObjIpcExportDescriptor* exportData)
{
    if(test_lwscibuf.LwSciBufObjIpcExport_fail) {
        test_lwscibuf.LwSciBufObjIpcExport_fail = false;
        return LwSciError_ResourceError;
    }

    return LwSciError_Success;
}

LwSciError LwSciSyncIpcExportFence(
    const LwSciSyncFence* syncFence,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncFenceIpcExportDescriptor* desc)
{
    if(test_lwscisync.LwSciSyncIpcExportFence_fail) {
        test_lwscisync.LwSciSyncIpcExportFence_fail = false;
        return LwSciError_ResourceError;
    }

    return LwSciError_Success;
}

LwSciError LwSciSyncAttrListIpcImportUnreconciled(
   LwSciSyncModule module,
   LwSciIpcEndpoint ipcEndpoint,
   const void* descBuf,
   size_t descLen,
   LwSciSyncAttrList* importedUnreconciledAttrList)
{
    if(test_lwscisync.LwSciSyncAttrListIpcImportUnreconciled_fail) {
        test_lwscisync.LwSciSyncAttrListIpcImportUnreconciled_fail = false;
        return LwSciError_ResourceError;
    }

    *importedUnreconciledAttrList = 0xABCEDF;

    return LwSciError_Success;
}

LwSciError LwSciSyncIpcImportFence(
    LwSciSyncObj syncObj,
    const LwSciSyncFenceIpcExportDescriptor* desc,
    LwSciSyncFence* syncFence)
{
    if(test_lwscisync.LwSciSyncIpcImportFence_fail) {
        test_lwscisync.LwSciSyncIpcImportFence_fail = false;
        return LwSciError_ResourceError;
    }

    syncFence = 0xABCEDF;

    return LwSciError_Success;
}

LwSciError LwSciSyncIpcImportAttrListAndObj(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* attrListAndObjDesc,
    size_t attrListAndObjDescSize,
    LwSciSyncAttrList const attrList[],
    size_t attrListCount,
    LwSciSyncAccessPerm minPermissions,
    int64_t timeoutUs,
    LwSciSyncObj* syncObj)
{
    if(test_lwscisync.LwSciSyncIpcImportAttrListAndObj_fail) {
        test_lwscisync.LwSciSyncIpcImportAttrListAndObj_fail = false;
        return LwSciError_ResourceError;
    }

    *syncObj = 0xABCEDF;

    return LwSciError_Success;
}

LwSciError LwSciBufAttrListIpcImportUnreconciled(
    LwSciBufModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    LwSciBufAttrList* importedUnreconciledAttrList)
{
    if(test_lwscibuf.LwSciBufAttrListIpcImportUnreconciled_fail) {
        test_lwscibuf.LwSciBufAttrListIpcImportUnreconciled_fail = false;
        return LwSciError_ResourceError;
    }

    *importedUnreconciledAttrList = 0xABCEDF;

    return LwSciError_Success;
}

LwSciError LwSciBufAttrListIpcExportUnreconciled(
    const LwSciBufAttrList unreconciledAttrListArray[],
    size_t unreconciledAttrListCount,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen)
{

    if (test_lwscibuf.LwSciBufAttrListIpcExportunreconciled_blobData_null == true)
    {
        *descBuf = nullptr;
        test_lwscibuf.LwSciBufAttrListIpcExportunreconciled_blobData_null = false;
    }
    else
    {
        *descBuf = 0xABCDEF;
    }

    if(test_lwscibuf.LwSciBufAttrListIpcExportUnreconciled_fail) {
        test_lwscibuf.LwSciBufAttrListIpcExportUnreconciled_fail = false;
        return LwSciError_ResourceError;
    }

    return LwSciError_Success;
}

LwSciError LwSciBufObjIpcImport(
    LwSciIpcEndpoint ipcEndpoint,
    const LwSciBufObjIpcExportDescriptor* desc,
    LwSciBufAttrList reconciledAttrList,
    LwSciBufAttrValAccessPerm minPermissions,
    int64_t timeoutUs,
    LwSciBufObj* bufObj)
{
    if(test_lwscibuf.LwSciBufObjIpcImport_fail) {
        test_lwscibuf.LwSciBufObjIpcImport_fail = false;
        return LwSciError_ResourceError;
    }

    *bufObj = 0xABCEDF;

    return LwSciError_Success;
}

LwSciError LwSciIpcInit(void)
{
#ifdef __linux__
    // Inititalise the event Fd for ipcSrc endpoint event handle
    ipcSrc.eventHandle = eventfd(0U, EFD_CLOEXEC | EFD_SEMAPHORE);
    if(ipcSrc.eventHandle == -1) {
        return LwSciError_NotInitialized;
    }

    // Inititalise the event Fd for ipcDst endpoint event handle
    ipcDst.eventHandle = eventfd(0U, EFD_CLOEXEC | EFD_SEMAPHORE);
    if(ipcDst.eventHandle == -1) {
        return LwSciError_NotInitialized;
    }
#endif

    // Initialise ipcSrc & ipcDst frames read ready status flags
    readReadySrcFr = false;
    readReadyDstFr = false;

    // Initialise Ipc Block disconnection status flag
    ipcDisconnect = false;

    // Initialise ipcSrc endpoint channel info
    // Set the channel num frames
    ipcSrc.info.nframes = 1U;
    // Set the channel frame size
    ipcSrc.info.frame_size = 512U;
    // Initialise frame buffer
    memset(ipcSrcFrame, 0, sizeof(ipcDstFrame));

    // Initialise ipcDst endpoint channel info
    // Set the channel num frames
    ipcDst.info.nframes = 1U;
    // Set the channel frame size
    ipcDst.info.frame_size = 512U;
    // Initialise frame buffer
    memset(ipcDstFrame, 0, sizeof(ipcDstFrame));

    return LwSciError_Success;
}

LwSciError LwSciIpcOpenEndpoint(const char *endpoint, LwSciIpcEndpoint *handle)
{
    if(!strncmp(endpoint, "InterProcessProducer", sizeof("InterProcessProducer"))) {
        // Inititalise handle for ipcSrc endpoint
        *handle = IPCSRC_ENDPOINT;
    }
    else if(!strncmp(endpoint, "InterProcessConsumer", sizeof("InterProcessConsumer"))) {
        // Inititalise handle for ipcDst endpoint
        *handle = IPCDST_ENDPOINT;
    }
    else {
        return LwSciError_NoSuchEntry;
    }

    return LwSciError_Success;
}

LwSciError LwSciIpcGetEndpointInfo(LwSciIpcEndpoint handle,
               struct LwSciIpcEndpointInfo *info)
{
    if (test_comm.LwSciIpcGetEndpointInfo_fail == true) {
        test_comm.LwSciIpcGetEndpointInfo_fail = false;
        return LwSciError_BadParameter;
    }
    if (test_comm.LwSciIpcGetEndpointInfo_Ilwalid_size == true)
    {
        test_comm.LwSciIpcGetEndpointInfo_Ilwalid_size = false;
        ipcSrc.info.frame_size = 0U;
    }
    if(handle == IPCSRC_ENDPOINT) {
        // Retrieve ipcSrc endpoint channel info
        memcpy(info, &ipcSrc.info, sizeof(struct LwSciIpcEndpointInfo));
    }
    else if(handle == IPCDST_ENDPOINT) {
        // Retrieve ipcDst endpoint channel info
        memcpy(info, &ipcDst.info, sizeof(struct LwSciIpcEndpointInfo));
    }
    else {
        return LwSciError_BadParameter;
    }

    ipcSrc.info.frame_size = 512U;
    return LwSciError_Success;
}

#ifdef __linux__
LwSciError LwSciIpcGetLinuxEventFd(LwSciIpcEndpoint handle, int32_t *fd)
{
    if(handle == IPCSRC_ENDPOINT) {
        // Retrieve ipcSrc endpoint event handle
        *fd = ipcSrc.eventHandle;
    }
    else if(handle == IPCDST_ENDPOINT) {
        // Retrieve ipcDst endpoint event handle
        *fd = ipcDst.eventHandle;
    }
    else {
        return LwSciError_BadParameter;
    }

    return LwSciError_Success;
}
#endif

void LwSciIpcResetEndpoint(LwSciIpcEndpoint handle)
{
    if(handle == IPCSRC_ENDPOINT) {
        // Initialise Ipc conn. reset for ipcSrc endpoint
        ipcSrcEvent = LW_SCI_IPC_EVENT_CONN_RESET;
    }
    else if(handle == IPCDST_ENDPOINT) {
        // Initialise Ipc conn. reset for ipcDst endpoint
        ipcDstEvent = LW_SCI_IPC_EVENT_CONN_RESET;
    }
}

LwSciError LwSciIpcGetEvent(LwSciIpcEndpoint handle, uint32_t *events)
{
    test_function_call.LwSciIpcGetEvent_counter++;
    if ((test_comm.waitForConnection_flag == true) &&
        (test_comm.counter == 1U))
    {
       *events = LW_SCI_IPC_EVENT_CONN_EST_ALL;
        return LwSciError_Success;

    }
    if ((test_comm.LwSciIpcGetEvent_Write_Pending == true) &&
             (test_comm.counter == 1U)) {
        *events = LW_SCI_IPC_EVENT_WRITE;
        return LwSciError_Success;
    }

    if ((test_comm.LwSciIpcGetEvent_Read_Pending == true) &&
       (test_comm.counter == 1U)) {
        *events = LW_SCI_IPC_EVENT_READ;
        return LwSciError_Success;
    }

    if (test_comm.LwSciIpcGetEvent_Disconnect_Request == true) {
        *events = 0U;
        return LwSciError_Success;
    }

    if (test_comm.LwSciIpcGetEvent_fail == true) {
        test_comm.LwSciIpcGetEvent_fail = false;
        *events = 0U;
        return LwSciError_StreamInternalError;
    }
    // Sleep is required here to prevent transient issues
    // in Ipc Block I/O threads synchronisation.
    usleep(10000);

    if(handle == IPCSRC_ENDPOINT) {
        // Retrieve the event for ipcSrc endpoint
        if(ipcSrcEvent == LW_SCI_IPC_EVENT_CONN_RESET ||
            !readReadyDstFr.load(std::memory_order_seq_cst))
            ipcSrcEvent = LW_SCI_IPC_EVENT_CONN_EST | LW_SCI_IPC_EVENT_WRITE;
        else
            ipcSrcEvent = LW_SCI_IPC_EVENT_READ;

        *events = ipcSrcEvent;
    }
    else if(handle == IPCDST_ENDPOINT) {
        // Retrieve the event for ipcDst endpoint
        if(ipcDstEvent == LW_SCI_IPC_EVENT_CONN_RESET ||
            !readReadySrcFr.load(std::memory_order_seq_cst))
            ipcDstEvent = LW_SCI_IPC_EVENT_CONN_EST | LW_SCI_IPC_EVENT_WRITE;
        else
            ipcDstEvent = LW_SCI_IPC_EVENT_READ;

        *events = ipcDstEvent;
    }
    else {
        return LwSciError_BadParameter;
    }

    if ((test_comm.LwSciIpcGetEvent_Write_Pending == true) ||
    (test_comm.LwSciIpcGetEvent_Read_Pending == true) ||
     (test_comm.waitForConnection_flag == true)) {
        *events = 0U;
        test_comm.counter++;
    }
    return LwSciError_Success;
}

LwSciError LwSciIpcWrite(LwSciIpcEndpoint handle, const void *buf, size_t size, int32_t *bytes)
{
    const uint64_t sig = 1ULL;

    test_function_call.LwSciIpcWrite_counter++;
    if (test_comm.LwSciIpcWrite_fail == true)
    {
        test_comm.LwSciIpcWrite_fail = false;
        return LwSciError_StreamInternalError;
    }
    if(handle == IPCSRC_ENDPOINT) {
        do {
            // Wait here until IpcSrc block is connected,
            // and the previous ipcSrc endpoint frame is read.
        }while(!ipcDisconnect && readReadySrcFr.load(std::memory_order_seq_cst));

        // Flush ipcSrc endpoint frame buffer
        memset(ipcSrcFrame, 0, sizeof(ipcSrcFrame));

        // Write to ipcSrc endpoint frame buffer
        memcpy(ipcSrcFrame, buf, size);

        // Check whether IpcSrc block is disconnected
        check_ipcBlock_conn(ipcSrcFrame);

        // Signal read ready for ipcSrc endpoint frame
        readReadySrcFr.store(true, std::memory_order_seq_cst);
#ifdef __linux__
        // Trigger an Ipc event for ipcDst endpoint
        write(ipcDst.eventHandle, &sig, sizeof(sig));
#else
        MsgSendPulse_r (ipcDst.coid, ipcDst.pulsePriority, ipcDst.pulseCode, 0);
#endif
    }
    else if(handle == IPCDST_ENDPOINT) {
        do {
            // Wait here until IpcDst block is connected,
            // and the previous ipcDst endpoint frame is read.
        }while(!ipcDisconnect && readReadyDstFr.load(std::memory_order_seq_cst));

        // Flush ipcDst endpoint frame buffer
        memset(ipcDstFrame, 0, sizeof(ipcDstFrame));

        // Write to ipcDst endpoint frame buffer
        memcpy(ipcDstFrame, buf, size);

        // Check whether IpcDst block is disconnected
        check_ipcBlock_conn(ipcDstFrame);

        // Signal read ready for ipcDst endpoint frame
        readReadyDstFr.store(true, std::memory_order_seq_cst);
#ifdef __linux__
        // Trigger an Ipc event for ipcSrc endpoint
        write(ipcSrc.eventHandle, &sig, sizeof(sig));
#else
        MsgSendPulse_r (ipcSrc.coid, ipcSrc.pulsePriority, ipcSrc.pulseCode, 0);
#endif
    }
    else {
        return LwSciError_BadParameter;
    }

    *bytes = (int32_t)size;

    return LwSciError_Success;
}

LwSciError LwSciIpcRead(LwSciIpcEndpoint handle, void *buf, size_t size, int32_t *bytes)
{
    test_function_call.LwSciIpcRead_counter++;
    if (test_comm.LwSciIpcRead_fail == true)
    {
        test_comm.LwSciIpcRead_fail = false;
        return LwSciError_BadParameter;
    }
    if (test_comm.LwSciIpcRead_flag == true)
    {
        *(size_t*)buf = 120;
        *bytes = 100U;
        test_comm.LwSciIpcRead_flag = false;
        return LwSciError_Success;
    }
    if(handle == IPCSRC_ENDPOINT) {
        // Read from ipcDst endpoint frame buffer
        memcpy(buf, ipcDstFrame, size);

        // Clear read ready for ipcDst endpoint frame.
        readReadyDstFr.store(false, std::memory_order_seq_cst);
    }
    else if(handle == IPCDST_ENDPOINT) {
        // Read from ipcSrc endpoint frame buffer
        memcpy(buf, ipcSrcFrame, size);

        // Clear read ready for ipcSrc endpoint frame.
        readReadySrcFr.store(false, std::memory_order_seq_cst);
    }
    else {
        return LwSciError_BadParameter;
    }

    if ((test_comm.unpackVal == 1U) ||
         (test_comm.unpackVal == 2U))
    {
       if (test_comm.unpackVal == 1U)
       {
           *(uint32_t*)buf = 12U;
       }
       else
       {
           *(uint32_t*)buf = 8U;
       }
       *((uint32_t*)buf+2U) = 0xABCD;
       *bytes = (int32_t)size;
       test_comm.unpackVal = 0U;
       return LwSciError_Success;
    }

    if ((test_comm.unpackValAndBlob == 1U) ||
     (test_comm.unpackValAndBlob == 2U) ||
     (test_comm.unpackValAndBlob == 3U) ||
     (test_comm.unpackValAndBlob == 4U) ||
     (test_comm.unpackValAndBlob == 5U))
    {
       if (test_comm.unpackValAndBlob == 1U)
       {
         *(uint32_t*)buf = 28U;
         *((uint32_t*)buf+3U) = 0x8U;
       }
       else if (test_comm.unpackValAndBlob == 2U)
       {
           *(uint32_t*)buf = 8U;
           *((uint32_t*)buf+3U) = 0x8U;
       }
       else if (test_comm.unpackValAndBlob == 3U)
       {
           *(uint32_t*)buf = 12U;
           *((uint32_t*)buf+3U) = 0x8U;
       }
       else if (test_comm.unpackValAndBlob == 4U)
       {
           *(uint32_t*)buf = 20U;
           *((uint32_t*)buf+3U) = 0x8U;
       }
       else
       {
           *(uint32_t*)buf = 28U;
           *((uint32_t*)buf+3U) = 0U;

       }
       *((uint32_t*)buf+2U) = 0xABCD;
       *((uint32_t*)buf+5U) = 0x9876U;
       *bytes = (int32_t)size;
       test_comm.unpackValAndBlob = 0U;
       return LwSciError_Success;
    }

if ((test_comm.unpackMsgSyncAttr == 1U) ||
    (test_comm.unpackMsgSyncAttr == 2U))
    {
       if (test_comm.unpackMsgSyncAttr == 1U)
       {
           *(uint32_t*)buf = 32U;
       }
       else {
           *(uint32_t*)buf = 10U;
       }
       *((uint8_t*)buf+8U) = 0x1U;
       *((uint8_t*)buf+9U) = 0x8U;
       *((uint8_t*)buf+17U) = 0x76;
       *((uint8_t*)buf+18U) = 0x98;
       *bytes = (int32_t)size;
       test_comm.unpackMsgSyncAttr = 0U;
       return LwSciError_Success;
    }

    if ((test_comm.unpackFenceExport == 1U) ||
    (test_comm.unpackFenceExport == 2U))
    {
       if (test_comm.unpackFenceExport == 1U)
       {
         *(uint32_t*)buf = 64U;
       }
       else {
           *(uint32_t*)buf = 20U;
       }
       *((uint32_t*)buf+2U) = 0xABCD;
       *((uint32_t*)buf+4U) = 0x1234;
       *((uint32_t*)buf+6U) = 0x5678;
       *((uint32_t*)buf+8U) = 0x9AFF;
       *((uint32_t*)buf+10U) = 0xABFF;
       *((uint32_t*)buf+12U) = 0xACFF;
       *((uint32_t*)buf+14U) = 0xADFF;
       *bytes = (int32_t)size;
       test_comm.unpackFenceExport = 0U;
       return LwSciError_Success;
    }

    if ((test_comm.unpackMsgElemAttr == 1U) ||
    (test_comm.unpackMsgElemAttr == 2U) ||
    (test_comm.unpackMsgElemAttr == 3U) ||
    (test_comm.unpackMsgElemAttr == 4U))
    {
       if (test_comm.unpackMsgElemAttr == 1U) {
           *(uint32_t*)buf = 40U;
       }
       else if (test_comm.unpackMsgElemAttr == 2U) {
           *(uint32_t*)buf = 8U;
       }
       else if (test_comm.unpackMsgElemAttr == 3U) {
           *(uint32_t*)buf = 12U;
       }
       else {
           *(uint32_t*)buf = 16U;
       }

       *((uint32_t*)buf+2U) = 0x1;
       *((uint32_t*)buf+3U) = 0x2;
       *((uint32_t*)buf+4U) = 0x1;
       *((uint32_t*)buf+5U) = 0x8;
       *((uint32_t*)buf+7U) = 0x9BFF;
       *bytes = (int32_t)size;
       test_comm.unpackMsgElemAttr = 0U;
       return LwSciError_Success;
    }

    if ((test_comm.unpackBufObjExport == 1U) ||
    (test_comm.unpackBufObjExport == 2U))
    {
       if (test_comm.unpackBufObjExport == 1U)
       {
            *(uint32_t*)buf = 264U;
       }
       else {
           *(uint32_t*)buf = 20U;
       }
       for (uint8_t i=1;i<=32;i++)
       {
        *((uint32_t*)buf+2*i) = 0x1234+i;
       }
       *bytes = (int32_t)size;
       test_comm.unpackBufObjExport = 0U;
       return LwSciError_Success;
    }

    if ((test_comm.unpackMsgPacketBuffer == 1U) ||
     (test_comm.unpackMsgPacketBuffer == 2U) ||
     (test_comm.unpackMsgPacketBuffer == 3U) ||
     (test_comm.unpackMsgPacketBuffer == 4U))
    {
       if(test_comm.unpackMsgPacketBuffer == 2U)
       {
           *(uint32_t*)buf = 8U;
       }
       else if (test_comm.unpackMsgPacketBuffer == 3U)
       {
           *(uint32_t*)buf = 16U;
       }
       else if (test_comm.unpackMsgPacketBuffer == 4U)
       {
           *(uint32_t*)buf = 20U;
       }
       else
       {
           *(uint32_t*)buf = 276U;
       }
       *((uint32_t*)buf+2) = 0x1234;
       *((uint32_t*)buf+4) = 0x1U;
       for (uint8_t i=0;i<32;i++)
       {
        *((uint32_t*)buf+5+2*i) = 0x1234+3*i;
       }
       *bytes = (int32_t)size;
       test_comm.unpackMsgPacketBuffer = 0U;
       return LwSciError_Success;
    }

    if ((test_comm.unpackMsgStatus == 1U) ||
     (test_comm.unpackMsgStatus == 2U) ||
     (test_comm.unpackMsgStatus == 3U) ||
     (test_comm.unpackMsgStatus == 4U))
    {
       if(test_comm.unpackMsgStatus == 2U)
       {
           *(uint32_t*)buf = 8U;
       }
       else if (test_comm.unpackMsgStatus == 3U)
       {
           *(uint32_t*)buf = 16U;
       }
       else if (test_comm.unpackMsgStatus == 4U)
       {
           *(uint32_t*)buf = 20U;
       }
       else
       {
           *(uint32_t*)buf = 24U;
       }
       *((uint32_t*)buf+2) = 0x1234;
       *((uint32_t*)buf+4) = 0x1;
       *((uint32_t*)buf+5) = LwSciError_BadParameter;
       *bytes = (int32_t)size;
       test_comm.unpackMsgStatus = 0U;
       return LwSciError_Success;
    }

    // *(size_t*)buf = 12U;
    *bytes = (int32_t)size;

    return LwSciError_Success;
}

void LwSciIpcCloseEndpoint(LwSciIpcEndpoint handle)
{
}

void LwSciIpcDeinit(void)
{
#ifdef __linux__
    // Close the the linux event Fd of ipcSrc endpoint event handle
    (void)close(ipcSrc.eventHandle);

    // Close the the linux event Fd of ipcDst endpoint event handle
    (void)close(ipcDst.eventHandle);
#else
    (void)ConnectDetach(ipcSrc.coid);
    (void)ConnectDetach(ipcDst.coid);
#endif
}

LwSciError LwSciBufAttrListIpcImportReconciled(
     LwSciBufModule module,
     LwSciIpcEndpoint ipcEndpoint,
     const void* descBuf,
     size_t descLen,
     const LwSciBufAttrList inputUnreconciledAttrListArray[],
     size_t inputUnreconciledAttrListCount,
     LwSciBufAttrList* importedReconciledAttrList)
{
    if(test_lwscibuf.LwSciBufAttrListIpcImportReconciled_fail) {
        test_lwscibuf.LwSciBufAttrListIpcImportReconciled_fail = false;
        return LwSciError_ResourceError;
    }

    *importedReconciledAttrList = 0xABCEDF;
     return LwSciError_Success;
}

LwSciError LwSciSyncModuleOpen(
    LwSciSyncModule* newModule)
{
    *newModule = 0xABCDEF;
    return LwSciError_Success;
}

LwSciError LwSciBufModuleOpen(
    LwSciBufModule* newModule)
{
    *newModule = 0xABCDEF;
    return LwSciError_Success;
}

#ifdef __cplusplus
extern "C" {
#endif

LwSciError LwSciIpcSetQnxPulseParam(LwSciIpcEndpoint handle,
        int32_t coid, int16_t pulsePriority, int16_t pulseCode,
        void *pulseValue)
{
    if (test_comm.LwSciIpcSetQnxPulseParam_fail == true)
    {
        test_comm.LwSciIpcSetQnxPulseParam_fail=false;
        return LwSciError_BadParameter;
    }
#ifndef __linux__
    if(handle == IPCSRC_ENDPOINT) {
        ipcSrc.coid = coid;
        ipcSrc.pulsePriority = pulsePriority;
        ipcSrc.pulseCode = pulseCode;
    } else if (handle == IPCDST_ENDPOINT){
        ipcDst.coid = coid;
        ipcDst.pulsePriority = pulsePriority;
        ipcDst.pulseCode = pulseCode;
    }
    else {
        return LwSciError_BadParameter;
    }
#endif
    return LwSciError_Success;
};

LwSciError LwSciIpcErrnoToLwSciErr(int32_t err)
{
     if (err == 0x104)
     {
         return LwSciError_Timeout;
     }
    return LwSciError_Success;
};

LwSciError LwSciBufFreeSourceObjIndirectChannelC2c(
    LwSciC2cBufSourceHandle sourceHandle)
{
    LwSciError err = LwSciError_Success;
    
    if (NULL == sourceHandle) {
           err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

 ret:
     return err;
}

LwSciError LwSciBufRegisterSourceObjIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciBufObj bufObj,
    LwSciC2cBufSourceHandle* sourceHandle)
{
    LwSciError err = LwSciError_Success;

ret:
    return err;
}

LwSciError LwSciSyncFenceWait(
    const LwSciSyncFence* syncFence,
    LwSciSyncCpuWaitContext context,
    int64_t timeoutUs)
{
    LwSciError error = LwSciError_Success;
    bool cleared = false;
    bool isDup = false;

    /** Check for invalid arguments */
    if (NULL == syncFence) {
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (NULL == context) {
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

 fn_exit:
    return error;
}

LwSciError LwSciSyncObjGetAttrList(
    LwSciSyncObj syncObj,
    LwSciSyncAttrList* syncAttrList)
{
    LwSciError error = LwSciError_Success;

    /** validate all input args */
    if (NULL == syncAttrList) {
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
fn_exit:
    return error;
}

LwSciError LwSciSyncAttrListValidateReconciled(
    LwSciSyncAttrList reconciledAttrList,
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    bool* isReconciledListValid)
{
    LwSciError error = LwSciError_Success;

    if ((NULL == inputUnreconciledAttrListArray) ||
            (0U == inputUnreconciledAttrListCount)) {
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

LwSciError LwSciSyncCpuWaitContextAlloc(
    LwSciSyncModule module,
    LwSciSyncCpuWaitContext* newContext)
{
    LwSciError error = LwSciError_Success;

    if (NULL == newContext) {
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
fn_exit:
   return error;
}

void LwSciCommonPanic(void)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 21_8), "Approved TID-1010")
}

LwSciError LwSciBufAttrListIsReconciled(
    LwSciBufAttrList attrList,
    bool* isReconciled)
{
    LwSciError err = LwSciError_Success;
	
ret:
    return err;
}

LwSciError LwSciSyncAttrListIsReconciled(
    LwSciSyncAttrList attrList,
    bool* isReconciled)
{
    LwSciError error = LwSciError_Success;

    return error;
}

LwSciError LwSciSyncAttrListIpcExportReconciled(
    const LwSciSyncAttrList reconciledAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen)
{
    LwSciError error = LwSciError_Success;

    return error;
}

LwSciError LwSciSyncAttrListIpcImportReconciled(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciSyncAttrList* importedReconciledAttrList)
{
    LwSciError error = LwSciError_Success;

    return error;
}

#ifdef __cplusplus
}
#endif
