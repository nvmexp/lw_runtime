/*
 * Copyright (c) 2021-2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

/**
 * @file
 *
 * @brief <b> LWPU Software Communications Interface (SCI) : LwSciC2cPcie </b>
 *
 * Allows applications to allocate and exchange buffers within two SoC's.
 */
#ifndef LWSCIC2C_PCIE_STREAM_H
#define LWSCIC2C_PCIE_STREAM_H

#include <stdbool.h>
#include <sys/types.h>
#include <stdint.h>
#include <lwscievent.h>
#include <lwsciipc.h>

#ifndef LWSCIC2C_X86
#if defined(__x86_64__)
#define LWSCIC2C_X86 1
#else
#define LWSCIC2C_X86 0
#endif
#endif

#if LWSCIC2C_X86
#include <lwtypes.h>
#include <lwRmShim/lwRmShim.h>
#else
#include <lwrm_memmgr_safe.h>
#include <lwrm_host1x_safe.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LWSCIC2C_PCIE_PERM_ILWALID = 0,
    LWSCIC2C_PCIE_PERM_READONLY = 1,
    LWSCIC2C_PCIE_PERM_READWRITE = 3,
} LwSciC2cPciePermissions;

typedef struct {
#if LWSCIC2C_X86
    LwRmShimSessionContext *session;
    LwRmShimDeviceContext  *device;
    LwRmShimMemoryContext  *memory;
#else
    LwRmMemHandle memHandle;
#endif
} LwSciC2cPcieBufRmHandle;

typedef struct {
#if LWSCIC2C_X86
    LwRmShimSessionContext *session;
    LwRmShimDeviceContext  *device;
    LwRmShimMemoryContext  *memory;
#else
    LwRmHost1xSyncpointHandle syncPoint;
#endif
} LwSciC2cPcieSyncRmHandle;

typedef struct {
    uint64_t offset;
    uint64_t size;
} LwSciC2cPcieFlushRange;

typedef struct LwSciC2cStream* LwSciC2cPcieStreamHandle;
typedef struct LwSciC2cBufSourceRef* LwSciC2cPcieBufSourceHandle;
typedef struct LwSciC2cBufTargetRef* LwSciC2cPcieBufTargetHandle;
typedef struct LwSciC2cSyncRef* LwSciC2cPcieSyncHandle;
typedef uint64_t LwSciC2cPcieAuthToken;

/*
 * To use LwSciC2c stream extension API's,
 * LwSciStream will fetch these function pointers via
 * LwSciIpc and then call stream extension API's directly.
 */
typedef struct {
    int (*openIndirectChannel)(LwSciIpcEndpoint, LwSciEventService*,
            size_t, size_t, size_t, size_t, LwSciC2cPcieStreamHandle*);
    int (*closeIndirectChannel)(LwSciC2cPcieStreamHandle);
    int (*bufMapTargetMemHandle)(LwSciC2cPcieBufRmHandle*, LwSciC2cPciePermissions,
            LwSciIpcEndpoint, LwSciC2cPcieBufTargetHandle*);
    int (*bufGetAuthTokenFromHandle)(LwSciC2cPcieBufTargetHandle, LwSciIpcEndpoint,
            LwSciC2cPcieAuthToken*);
    int (*bufGetHandleFromAuthToken)(LwSciC2cPcieAuthToken, LwSciIpcEndpoint,
            LwSciC2cPcieBufTargetHandle*);
    int (*bufRegisterTargetHandle)(LwSciC2cPcieStreamHandle, LwSciC2cPcieBufTargetHandle);
    int (*bufDupTargetHandle)(LwSciC2cPcieBufTargetHandle, LwSciC2cPcieBufTargetHandle*);
    int (*bufFreeTargetHandle)(LwSciC2cPcieBufTargetHandle);
    int (*bufMapSourceMemHandle)(LwSciC2cPcieBufRmHandle*, LwSciC2cPciePermissions,
            LwSciC2cPcieStreamHandle, LwSciC2cPcieBufSourceHandle*);
    int (*bufRegisterSourceHandle)(LwSciC2cPcieStreamHandle, LwSciC2cPcieBufSourceHandle);
    int (*bufFreeSourceHandle)(LwSciC2cPcieBufSourceHandle);
    int (*syncMapRemoteMemHandle)(LwSciC2cPcieSyncRmHandle*, LwSciIpcEndpoint,
            LwSciC2cPcieSyncHandle*);
    int (*syncGetAuthTokenFromHandle)(LwSciC2cPcieSyncHandle, LwSciIpcEndpoint,
            LwSciC2cPcieAuthToken*);
    int (*syncGetHandleFromAuthToken)(LwSciC2cPcieAuthToken, LwSciIpcEndpoint,
            LwSciC2cPcieSyncHandle*);
    int (*syncRegisterRemoteHandle)(LwSciC2cPcieStreamHandle, LwSciC2cPcieSyncHandle);
    int (*syncDupRemoteHandle)(LwSciC2cPcieSyncHandle, LwSciC2cPcieSyncHandle*);
    int (*syncMapLocalMemHandle)(LwSciC2cPcieSyncRmHandle*, LwSciC2cPcieStreamHandle,
            LwSciC2cPcieSyncHandle*);
    int (*syncCreateLocalHandle)(LwSciC2cPcieStreamHandle, LwSciC2cPcieSyncHandle*);
    int (*syncRegisterLocalHandle)(LwSciC2cPcieStreamHandle, LwSciC2cPcieSyncHandle);
    int (*syncFreeHandle)(LwSciC2cPcieSyncHandle);
    int (*syncCreateCpuMapping)(LwSciC2cPcieSyncHandle, void**);
    int (*syncDeleteCpuMapping)(LwSciC2cPcieSyncHandle, void*);
    int (*pushWaitIndirectChannel)(LwSciC2cPcieStreamHandle, LwSciC2cPcieSyncHandle,
            uint64_t, uint64_t);
    int (*pushSignalIndirectChannel)(LwSciC2cPcieStreamHandle, LwSciC2cPcieSyncHandle);
    int (*pushSignalWithValueIndirectChannel)(LwSciC2cPcieStreamHandle, LwSciC2cPcieSyncHandle, uint64_t);
    int (*pushCopyIndirectChannel)(LwSciC2cPcieStreamHandle, LwSciC2cPcieBufSourceHandle,
            LwSciC2cPcieBufTargetHandle, const LwSciC2cPcieFlushRange*, size_t);
    int (*pushSubmitIndirectChannel)(LwSciC2cPcieStreamHandle);
} LwSciC2cPcieCopyFuncs;

/*FIXME: We are checking with LwSciIpc to use internal info for context storage.
 * If not agreed then we may need to add context parameter too.
 * We will need to figure out how that can be allocated in LwSci layer and passed to C2C.
 */

/**
 * @brief Create LwSciC2cPcie channel handle interface to support LwSciBuf and LwSciSync.
 *
 * To support LwSciBuf and LwSciSync over LwSciC2cPcie
 *  - Allocate unique channel handle
 *  - Tie channel handle with LwSciIpc EP of control channel
 *  - Allocate queue for object pipelining
 *  - Allocate EventHandle and update callback to be called in Fence Expiry
 *
 * @pre LwSciIpc endpoint is established with remote.
 *
 * @param[in] ipcEndpoint   : Opened LwSciIpc handle for C2C endpoint
 * @param[in] eventService  : LwSciEventService handle created by application
 * @param[in] numRequest    : Number of total submit request can come
 * @param[in] numFlushranges: Number of total flush ranges for one submit
 * @param[in] numPreFences  : Number of pre fence with one submit request
 * @param[in] numPostFences : Number of post fence with one submit request
 * @param[out] channelHandle: LwSciC2cPcie allocated channel handle placeholder
 *
 * @return ::int, the completion code of the operation.
 * @note Producer Only API.
 */
int LwSciC2cPcieOpenIndirectChannel(LwSciIpcEndpoint ipcEndpoint, LwSciEventService *eventService,
        size_t numRequest, size_t numFlushRanges, size_t numPreFences, size_t numPostFences,
        LwSciC2cPcieStreamHandle *channelHandle);

/**
 * @brief Release channel handle
 *
 * Closes the indirect channel with LwSciBuf and LwSciSync
 *  - Free channel handle
 *
 * @param[in] channelHandle: LwSciC2cPcie allocated channel handle placeholder
 *
 * @return ::int, the completion code of the operation.
 */
int LwSciC2cPcieCloseIndirectChannel(LwSciC2cPcieStreamHandle channelHandle);

/**
 * @brief Map memory handle associated with Buf object to PCIe Inbound.
 *
 * @param[in] memHandle     : MemHandle associated with LwSciBuf object
 * @param[in] permissions   : Read/Write permission to map buffer
 * @param[in] ipcEndpoint   : Opened LwSciIpc handle for C2C endpoint
 * @param[out] targetHandle : LwSciC2c allocated handle for LwSciBuf object
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Consumer only API.
 * @note LwSciBuf memory size should be aligned to 4 to support eDMA alignment.
 */
int LwSciC2cPcieBufMapTargetMemHandle(LwSciC2cPcieBufRmHandle *memHandle,
        LwSciC2cPciePermissions permissions, LwSciIpcEndpoint ipcEndpoint,
        LwSciC2cPcieBufTargetHandle *targetHandle);

/**
 * @brief Generate authtoken from target handle.
 *
 * LwSciBufObjIpcExport() is used at LwSci layer to export the object to remote.
 * LwSciC2cPcie needs to generate authtoken for target handle,
 * which can be share with remote/producer.
 *
 * @param[in] targetHandle : LwSciC2c allocated handle for MemHandle
 * @param[in] ipcEndpoint  : Opened LwSciIpc handle for C2C endpoint
 * @param[out] authToken   : Authtoken to be shared with remote
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Consumer only API.
 */
int LwSciC2cPcieBufGetAuthTokenFromHandle(LwSciC2cPcieBufTargetHandle targetHandle,
        LwSciIpcEndpoint ipcEndpoint, LwSciC2cPcieAuthToken *authToken);

/**
 * @brief Import Authtoken exported by remote/consumer.
 *
 * LwSciBufObjIpcImport() is used at LwSci layer to import remote exported object.
 * LwSciC2cPcie will use Authtoken to import the object and generate handle.
 *
 * @param[in] authToken     : Authtoken shared by remote for exported object
 * @param[in] ipcEndpoint   : Opened LwSciIpc handle for C2C endpoint
 * @param[out] targetHandle : LwSciC2cPcie allocated handle for imported LwSciBuf object
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Producer only API.
 */
int LwSciC2cPcieBufGetHandleFromAuthToken(LwSciC2cPcieAuthToken authToken,
        LwSciIpcEndpoint ipcEndpoint, LwSciC2cPcieBufTargetHandle *targetHandle);

/**
 * @brief Associate C2C stream handle with Buf target handle.
 *
 * @param[in] channelHandle : LwSciC2cPcie allocated channel handle placeholder
 * @param[in] targetHandle  : target handle for memhandle associated with target obj
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Producer only API.
 */
int LwSciC2cPcieBufRegisterTargetHandle(LwSciC2cPcieStreamHandle channelHandle,
        LwSciC2cPcieBufTargetHandle targetHandle);

/**
 * @brief Duplicate Target handle.
 *
 * @param[in] origHandle  : Target handle for imported memhandle
 * @param[out] dupHandle  : Handle to be used as target handle for LwScibuf object
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Producer only API.
 */
int LwSciC2cPcieBufDupTargetHandle(LwSciC2cPcieBufTargetHandle origHandle,
        LwSciC2cPcieBufTargetHandle *dupHandle);

/**
 * @brief Release target handle created for memhandle.
 *
 * This function frees mapping created for target LwSciBuf object.
 *
 * @param[in] targetHandle : Target handle for memhandle associated with target obj
 *
 * @return ::int, the completion code of the operation.
 */
int LwSciC2cPcieBufFreeTargetHandle(LwSciC2cPcieBufTargetHandle targetHandle);

/**
 * @brief Map memory handle associated with source Buf object.
 *
 * Create PCIe mapping for memory handle to be used as source by EDMA.
 *
 * @param[in] memHandle     : MemHandle associated with LwSciBuf object
 * @param[in] permissions   : Read/Write permission to map buffer
 * @param[in] channelHandle : LwSciC2cPcie allocated channel handle placeholder
 * @param[out] sourceHandle : LwSciC2cPcie allocated placeholder for MemHandle
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Producer only API.
 * @note LwSciBuf memory size should be aligned to 4 to support eDMA alignment.
 */
int LwSciC2cPcieBufMapSourceMemHandle(LwSciC2cPcieBufRmHandle *memHandle,
        LwSciC2cPciePermissions permissions, LwSciC2cPcieStreamHandle channelHandle,
        LwSciC2cPcieBufSourceHandle *sourceHandle);

/**
 * @brief Associate C2C stream handle with source handle.
 *
 * @param[in] channelHandle : LwSciC2cPcie allocated channel handle placeholder
 * @param[in] sourceHandle : LwSciC2cPcie allocated placehodler for MemHandle
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Producer only API.
 */
int LwSciC2cPcieBufRegisterSourceHandle(LwSciC2cPcieStreamHandle channelHandle,
        LwSciC2cPcieBufSourceHandle sourceHandle);

/**
 * @brief Release source LwSciBuf object and LwSciC2cPcie handle mapping.
 *
 * This function frees the mapping created for source LwSciBuf object.
 *
 * @param[in] sourceHandle: LwSciC2cPcie allocated placehodler for LwSciBuf obj
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Producer only API.
 */
int LwSciC2cPcieBufFreeSourceHandle(LwSciC2cPcieBufSourceHandle sourceHandle);

/**
 * @brief Map syncpoint associated memhandle to PCIe Inbound
 *
 * @param[in] memHandle   : Memhandle associated with syncpoint
 * @param[in] ipcEndpoint : Opened LwSciIpc handle for C2C endpoint
 * @param[out] syncHandle : C2C allocated handle for syncpoint memhandle
 *
 * @return ::int, the completion code of the operation
 * @note Should be called for objects copyDoneConsObj and consReadsDoneProdObj
 * @note LwSciSync memory size should be aligned to 4 to support eDMA alignment.
 */
int LwSciC2cPcieSyncMapRemoteMemHandle(LwSciC2cPcieSyncRmHandle *memHandle,
        LwSciIpcEndpoint ipcEndpoint, LwSciC2cPcieSyncHandle *syncHandle);

/**
 * @brief Generate authtoken to be shared with remote for synpoint handle
 *
 * @param[in] syncHandle  : C2C allocated handle for syncpoint memhandle
 * @param[in] ipcEndpoint : Opened LwSciIpc handle for C2C endpoint
 * @param[out] authToken  : Authtoken generated by LwSciC2cPcie to be shared with remote
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Should be called for objects copyDoneConsObj and consReadsDoneProdObj
 */
int LwSciC2cPcieSyncGetAuthTokenFromHandle(LwSciC2cPcieSyncHandle syncHandle,
        LwSciIpcEndpoint ipcEndpoint, LwSciC2cPcieAuthToken *authToken);

/**
 * @brief Import exported LwSciSync object in LwSciC2c.
 *
 * @param[in] authToken   : Authoken shared by remote
 * @param[in] ipcEndpoint : Opened LwSciIpc handle for C2C endpoint
 * @param[out] syncHandle : Handle allocated at LwSciC2c for imported syncpoint
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Should be called for objects copyDoneConsObj and consReadsDoneProdObj
 */
int LwSciC2cPcieSyncGetHandleFromAuthToken(LwSciC2cPcieAuthToken authToken,
        LwSciIpcEndpoint ipcEndpoint, LwSciC2cPcieSyncHandle *syncHandle);

/**
 * @brief Associate C2C stream handle with sync handle
 *
 * @param[in] channelHandle : LwSciC2cPcie allocated channel handle placeholder
 * @param[in] syncHandle    : LwSciC2cPcie allocated handle for LwSciSync object
 *
 * @return ::int, the completion code of the operation
 *
 * @note Should be called for copyDoneConsObj and consReadsDoneProdObj
 * @note Producer only API
 */
int LwSciC2cPcieSyncRegisterRemoteHandle(LwSciC2cPcieStreamHandle channelHandle,
        LwSciC2cPcieSyncHandle syncHandle);

/**
 * @brief Duplicate Remote Sync object handle.
 *
 * @param[in] origHandle  : LwSciC2cPcieSync handle for imported LwSciSync
 * @param[out] dupHandle  : Duplicated handle for imported sync handle
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Producer only API
 */
int LwSciC2cPcieSyncDupRemoteHandle(LwSciC2cPcieSyncHandle origHandle,
        LwSciC2cPcieSyncHandle *dupHandle);

/**
 * @brief Map syncpoint associated memhandle to local IOVA
 *
 * @param[in] memHandle     : mem handle associated with syncpoint
 * @param[in] channelHandle : LwSciC2cPcie allocated channel handle placeholder
 * @param[out] syncHandle   : C2C allocated handle for syncpoint memhandle
 *
 * @return ::int, the completion code of the operation
 *
 * @note Should be called for copyDoneProdObj
 * @note Producer only API
 * @note LwSciBuf memory size should be aligned to 4 to support eDMA alignment.
 */
int LwSciC2cPcieSyncMapLocalMemHandle(LwSciC2cPcieSyncRmHandle *memHandle,
        LwSciC2cPcieStreamHandle channelHandle, LwSciC2cPcieSyncHandle *syncHandle);

/**
 * @brief Create a new syncHandle to be used for handling local preFences without memHandle.
 *
 * API creates dummy syncHandle in C2C which will be used in streaming along with
 * correct syncpoint_id.
 *
 * @param[in] channelHandle : LwSciC2cPcie allocated channel handle placeholder
 * @param[out] syncHandle   : C2C allocated handle
 *
 * @return ::int, the completion code of the operation
 *
 * @note Should be called for engineWritesDoneObj
 * @note Producer only API
 */
int LwSciC2cPcieSyncCreateLocalHandle(LwSciC2cPcieStreamHandle channelHandle,
        LwSciC2cPcieSyncHandle *syncHandle);

/**
 * @brief Register local LwSciSync wait object with LwSciC2cPcie
 *
 * This function registers LwSciSync waiting object with LwSciC2cPcie
 * Allocates LwSciC2cPcieSyncHandle and stores needed properties
 *
 * @param[in] channelHandle : LwSciC2cPcie allocated channel handle placeholder
 * @param[out] syncHandle   : LwSciC2c allocated handle for LwSciSync object
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Should be called for engineWritesDoneObj and copyDoneProdObj
 * @note Producer only API
 */
int LwSciC2cPcieSyncRegisterLocalHandle(LwSciC2cPcieStreamHandle channelHandle,
        LwSciC2cPcieSyncHandle syncHandle);

/**
 * @brief Release C2C handle for LwSciSync object exported/imported.
 *
 * @param[in] syncHandle  : LwSciC2c handle for LwSciSync object
 *
 * @return ::int, the completion code of the operation.
 */
int LwSciC2cPcieSyncFreeHandle(LwSciC2cPcieSyncHandle syncHandle);

/**
 * @brief Create user space CPU mapping for LwSciC2c sync handle.
 *
 *
 *
 * @param[in] syncHandle : Handle allocated by LwSciC2cPcie for LwSciSync object.
 * @param[out] pVirtAddr : Mapped CPU address for syncpoint handle
 *
 * @return ::int, the completion code of the operation.
 *
 * @note: Creation of CPU mapping will be allowed for imported handle only.
 * @note: Please use CPU mapping only to write first 4 bytes with typecasting to (uint32_t*).
 *        Any pointer operation shall lead to undefined behavior.
 * @note Consumer only API
 */
int LwSciC2cPcieSyncCreateCpuMapping(LwSciC2cPcieSyncHandle syncHandle, void **pVirtAddr);

/**
 * @brief Delete user space CPU mapping for LwSciC2c sync handle.
 *
 *
 *
 * @param[in] syncHandle : Handle allocated by LwSciC2cPcie for LwSciSync object.
 * @param[in] pVirtAddr  : Mapped CPU address for syncpoint handle.
 *
 * @return ::int, the completion code of the operation.
 *
 * @note Consumer only API
 */
int LwSciC2cPcieSyncDeleteCpuMapping(LwSciC2cPcieSyncHandle syncHandle, void *pVirtAddr);

/**
 * @brief Push entry in LwSciC2cPcie queue for LwSciSync wait.
 *
 * @param[in] channelHandle : LwSciC2cPcie allocated channel handle placeholder
 * @param[in] syncHandle    : Handle allocated by LwSciC2cPcie for LwSciSync object
 * @param[in] syncPointId   : Id associated with syncpoint for LwSciSync object
 * @param[in] threshold     : The value for the sync point register to obtain.
 *
 * @return ::int, the completion code of the operation.
 */
int LwSciC2cPciePushWaitIndirectChannel(LwSciC2cPcieStreamHandle channelHandle,
        LwSciC2cPcieSyncHandle syncHandle, uint64_t syncPointId, uint64_t threshold);

/**
 * @brief Push entry in LwSciC2cPcie queue for LwSciSync signal.
 *
 * @param[in] channelHandle : LwSciC2cPcie allocated channel handle placeholder
 * @param[in] syncHandle    : Handle allocated by LwSciC2cPcie for LwSciSync object
 * @param[in] fenceValue    : fenceValue to be updated for semaphore
 *
 * @return ::int, the completion code of the operation.
 * @note Fence management is done at LwSci layer.
 */
int LwSciC2cPciePushSignalWithValueIndirectChannel(LwSciC2cPcieStreamHandle channelHandle,
        LwSciC2cPcieSyncHandle syncHandle, uint64_t fenceValue);

/**
 * @brief Push entry in LwSciC2cPcie queue for LwSciSync signal.
 *
 * @param[in] channelHandle : LwSciC2cPcie allocated channel handle placeholder
 * @param[in] syncHandle    : Handle allocated by LwSciC2cPcie for LwSciSync object
 *
 * @return ::int, the completion code of the operation.
 * @note Fence management is done at LwSci layer.
 */
int LwSciC2cPciePushSignalIndirectChannel(LwSciC2cPcieStreamHandle channelHandle,
        LwSciC2cPcieSyncHandle syncHandle);
/**
 * @brief Push entry in LwSciC2cPcie queue for data transfer.
 *
 * @param[in] channelHandle  : LwSciC2cPcie allocated channel handle placeholder
 * @param[in] sourceHandle   : LwSciC2c allocated handle for source LwSciBuf object
 * @param[in] targetHandle   : LwSciC2c allocated handle for target LwSciBuf object
 * @param[in] flushRanges    : Range within the buffer to be transferred
 * @param[in] numFlushRanges : Number of ranges in buffer to be transferred
 *
 * @return ::int, the completion code of the operation.
 * @note For each flush range size and offset should be aligned to 4 to support eDMA alignment.
 */
int LwSciC2cPciePushCopyIndirectChannel(LwSciC2cPcieStreamHandle channelHandle,
        LwSciC2cPcieBufSourceHandle sourceHandle, LwSciC2cPcieBufTargetHandle targetHandle,
        const LwSciC2cPcieFlushRange *flushRanges, size_t numFlushRanges);

/**
 * @brief Initiate streaming over LwSciC2cPcie.
 *
 * LwSciC2cPcie will start dequeueing the object and execute transfer upto
 * current position.
 *
 * @param[in] channelHandle : LwSciC2cPcie allocated channel handle placeholder
 *
 * @return ::int, the completion code of the operation.
 * @note At least one valid flush_range should have been pushed before calling.
 */
int LwSciC2cPciePushSubmitIndirectChannel(LwSciC2cPcieStreamHandle channelHandle);

/**
 * @brief Fill LwSciC2c stream extension API function pointers.
 *
 * @param[out] fnSet : placeholder to fill function pointers
 *
 * @return ::int, the completion code of the operation.
 */
int LwSciC2cPcieGetCopyFuncSet(LwSciC2cPcieCopyFuncs *fnSet);

/**
 * @brief Validate LwSciC2c stream extension API function pointers.
 *
 * @param[in] fnSet : function pointer placeholder to validate
 *
 * @return ::int, the completion code of the operation.
 */
int LwSciC2cPcieValidateCopyFuncSet(LwSciC2cPcieCopyFuncs *fnSet);

#ifdef __cplusplus
}
#endif
#endif /* LWSCIC2C_PCIE_STREAM_H */
