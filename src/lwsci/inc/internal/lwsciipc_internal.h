/*
 * Copyright (c) 2019-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIIPC_INTERNAL_H
#define INCLUDED_LWSCIIPC_INTERNAL_H
// TODO: should be removed after updating lwrm_memmgr_safe.h
#define LWSCIIPC_INTERNAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <lwsciipc.h>
#include <lwscievent_internal.h>

/**
 * @defgroup lwsci_ipc_internal IPC Internal Declarations
 *
 * @ingroup lwsci_ipc_api
 * @{
 */

/*******************************************************************/
/************************ OS SPECIFIC ******************************/
/*******************************************************************/
#ifdef __QNX__
#include <sys/iofunc.h>
#include <sys/resmgr.h>
#endif /* __QNX__ */

#ifdef LINUX
/* to build LwSciIpcEndpointValidateAuthTokenQnx() API */
typedef int32_t resmgr_context_t;
#endif /* LINUX */

/*******************************************************************/
/************************ DATA TYPES *******************************/
/*******************************************************************/

/**
 * @brief abilityId can be obtained with
 * procmgr_ability_lookup(LWSCIIPC_ABILITY_ID)
 */
#define LWSCIIPC_ABILITY_ID "LwSciIpcEndpoint"

/**
 * @brief IPA region/notification type of Inter-VM backend
 */
#define LWSCIIPC_ILWALID_IPA    0U
#define LWSCIIPC_TRAP_IPA       1U
#define LWSCIIPC_MSI_IPA        2U

/**
 * @brief backend type definition
 */
#define LWSCIIPC_BACKEND_ITC    0U
#define LWSCIIPC_BACKEND_IPC    1U
#define LWSCIIPC_BACKEND_IVC    2U
#define LWSCIIPC_BACKEND_C2C    3U  /* TODO: for backward comp. remove later */
#define LWSCIIPC_BACKEND_C2C_PCIE 3U
#define LWSCIIPC_BACKEND_C2C_NPM  4U
#if (LW_IS_SAFETY == 0)
#define LWSCIIPC_BACKEND_MAX  (LWSCIIPC_BACKEND_C2C_NPM + 1U)
#else
#define LWSCIIPC_BACKEND_MAX  (LWSCIIPC_BACKEND_IVC + 1U)
#endif

/**
 * @brief VUID(VM unique ID) of the LwSciIpc endpoint.
 */
typedef uint64_t LwSciIpcEndpointVuid;

/**
 * @brief Authentication token of the LwSciIpc endpoint.
 */
typedef uint64_t LwSciIpcEndpointAuthToken;

typedef struct LwSciIpcEndpointAccessInfo LwSciIpcEndpointAccessInfo;

/**
 * @brief Defines access information about LwSciIpc endpoint
 */

struct LwSciIpcEndpointAccessInfo {
    /*! unique group id of the endpoint */
    gid_t gid;
    /*! backend type of the endpoint
     * LWSCIIPC_BACKEND_ITC, LWSCIIPC_BACKEND_IPC, LWSCIIPC_BACKEND_IVC, ...
     */
    int32_t backend;
    /*! VM-wide unique id(VUID) per endpoint */
    LwSciIpcEndpointVuid vuid;
    /*! physical memory address of channel data memory (0: N/A) */
    uint64_t phyAddr;
    /*! physical memory size of channel data memory (0: N/A) */
    uint64_t phySize;
    /*! IRQ number of IVC signalling (inter-VM endpoint only) (-1: N/A) */
    int32_t irq;
    /*! Queue id (Intra-VM: 0 or 1, Inter-VM: IVC queue ID of PCT */
    int32_t id;
    /*! IO(TRAP/MSI) IPA to used to notify peer in inter-VM */
    uint64_t notiIpa;
    /*! size of IO(TRAP/MSI) IPA to used to notify peer in inter-VM */
    uint64_t notiIpaSize;
    /*! type of IPA to used to notify peer in inter-VM
     * LWSCIIPC_TRAP_IPA, LWSCIIP_MSI_IPA
     */
    uint32_t notiIpaType;
};

typedef struct LwSciIpcEndpointInfoInternal LwSciIpcEndpointInfoInternal;

/**
 * @brief Defines internal information about C2C endpoint.
 */
typedef struct {
    int fd;
} LwSciIpcC2cEndpointInfoInternal;

/**
 * @brief Defines internal information about the LwSciIpc endpoint.
 */
struct LwSciIpcEndpointInfoInternal {
    /*! Holds IVC irq info */
    uint32_t irq;
    /*! Holds C2C endpoint info */
    LwSciIpcC2cEndpointInfoInternal c2cInfo;
};

#if (LW_IS_SAFETY == 0)
typedef uintptr_t LwSciIpcC2cCookie;
#endif /* (LW_IS_SAFETY == 0) */

/**
 * @brief Defines topology ID of the LwSciIpc endpoint.
 */
typedef struct {
    /*! Holds SOC ID */
    uint32_t SocId;
    /*! Holds VMID */
    uint32_t VmId;
} LwSciIpcTopoId;


/** Invalid VUID definition */
#define LWSCIIPC_ENDPOINT_VUID_ILWALID      0U
/** Invalid authentication token definition */
#define LWSCIIPC_ENDPOINT_AUTHTOKEN_ILWALID 0U
/** Current self SOC ID */
#define LWSCIIPC_SELF_SOCID 0xFFFFFFFFU
/** Current self VM ID */
#define LWSCIIPC_SELF_VMID  0xFFFFFFFFU

/*******************************************************************/
/********************* FUNCTION TYPES ******************************/
/*******************************************************************/

/**
 * @brief Bind EventService to an Endpoint.
 *
 * Binds an EventService to an already opened Endpoint
 * LwSciEventService can be created through LwSciEventLoopServiceCreate()
 * If binding of EventService to Endpoint is not done during Open Endpoint
 * then this API need to be used to bind to be able to use LwSciIpcGetEventNotifier()
 *
 * @param[in]  handle  LwSciIpc endpoint handle.
 * @param[in]  eventService  An abstract object to use LwSciEventService infrastructure.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates any argument was NULL or invalid.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * @pre Invocation of LwSciEventLoopServiceCreate() must be successful.
 *      Invocation of LwSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: Yes
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciIpcBindEventService(LwSciIpcEndpoint handle,
    LwSciEventService *eventService);

#if (LW_IS_SAFETY == 0)
/**
 * @brief Sets internal cookie for C2C context
 *
 * <b>This API is specific to x86 OS.</b>
 * The funtion sets cookie for C2C to maintain context
 *
 * @param[in]  handle  LwSciIpc endpoint handle.
 * @param[in]  cookie  A pointer to LwSciIpcC2cCookie object that
 *                     this function associates with the handle.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_BadParameter       Indicates an invalid or NULL argument.
 * - ::LwSciError_NotSupported       Indicates API is not supported on provided
 *
 * @pre Invocation of LwSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
LwSciError LwSciIpcSetCookie(LwSciIpcEndpoint handle,
                LwSciIpcC2cCookie cookie);

/**
 * @brief Gets internal cookie for C2C context
 *
 * <b>This API is specific to x86 OS.</b>
 * The funtion gets cookie used for maintaining C2C context
 *
 * @param[in]  handle  LwSciIpc endpoint handle.
 * @param[out] cookie  A pointer to LwSciIpcC2cCookie object that
 *                     this function associates with the handle.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_BadParameter       Indicates an invalid or NULL argument.
 * - ::LwSciError_NotSupported       Indicates API is not supported on provided
 *
 * @pre Invocation of LwSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
LwSciError LwSciIpcGetCookie(LwSciIpcEndpoint handle,
                LwSciIpcC2cCookie *cookie);

/**
 * @brief Get C2C stream/copy API function pointers
 *
 * This function provides C2C stream API function pointer set to upper S/W
 * layer. The upper S/W layer is able to call C2C stream API of C2C sublayer
 * directly.
 *
 * @param[in]  handle  LwSciIpc Endpoint handle
 * @param[out] fn      C2C stream/copy function pointer set structure

 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates invalid input parameter.
 * - ::LwSciError_NotSupported       Indicates API is not supported
 *                                   (C2C library is not ready).
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciIpcGetC2cCopyFuncSet(LwSciIpcEndpoint handle, void *fn);

/**
 * @brief Validate C2C stream/copy API function pointers
 *
 * This function validates C2C stream API function pointer set.
 *
 * @param[in]  handle  LwSciIpc Endpoint handle
 * @param[in]  fn      C2C stream/copy function pointer set structure
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates invalid input parameter.
 *                                   (including checksum error)
 * - ::LwSciError_NotSupported       Indicates API is not supported
 *                                   (C2C library is not ready).
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciIpcValidateC2cCopyFuncSet(LwSciIpcEndpoint handle,
    const void *fn);
#endif /* LW_IS_SAFETY == 0 */

#ifdef __QNX__
/**
 * @brief Opens LwSciIpc ConfigBlob.
 *
 * <b>This API is specific to QNX OS.</b>
 * The function opens and read LwSciIpc configuration blob and populates
 * internal DB to search endpoint entry.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotPermitted       Indicates opening blob has failed.
 * - ::LwSciError_NotSupported       Indicates API is not supported in the provided
 *                                   OS environment.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: PROCMGR_AID_MEM_PHYS
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciIpcOpenCfgBlob(void);

/**
 * @brief Gets endpoint access information from ConfigBlob.
 *
 * <b>This API is specific to QNX OS.</b>
 * The function search entry with endpoint name from configuration blob
 * shared memory, then return its group id and backend type information.
 * This information can be used to apply subgroup id, vuid and
 * QNX OS abilities to LwSciIpc client program.
 * <b>This API can be used in root user process only due to security issue.</b>
 *
 * @param[in]  endpoint  The name of the LwSciIpc endpoint to search.
 * @param[out] info      Endpoint access information on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_BadParameter       Indicates an invalid or NULL argument.
 * - ::LwSciError_NoSuchEntry        Indicates Endpoint name is not found.
 * - ::LwSciError_NotPermitted       Indicates process doesn't have the required
 *                                   privilege.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   OS environment.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state
 *
 * @pre Invocation of LwSciIpcOpenCfgBlob() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciIpcGetEndpointAccessInfo(const char *endpoint,
    LwSciIpcEndpointAccessInfo *info);

/**
 * @brief Closes LwSciIpc ConfigBlob.
 *
 * <b>This API is specific to QNX OS.</b>
 * Unmap LwSciIpc configuration blob shared memory and close it.
 *
 * @pre Invocation of LwSciIpcOpenCfgBlob() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 */
void LwSciIpcCloseCfgBlob(void);
#endif /* __QNX__ */

/**
 * @brief Gets internal endpoint information.
 *
 * The funtion returns additional endpoint information for DRIVE OS internal
 * purpose.
 *
 * @param[in]  handle  LwSciIpc endpoint handle.
 * @param[out] info    A pointer to LwSciIpcEndpointInfoInternal object that
 *                     this function copies the info to on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_BadParameter       Indicates an invalid or NULL argument.
 * - ::LwSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type or OS environment.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state
 *
 * @pre Invocation of LwSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciIpcGetEndpointInfoInternal(LwSciIpcEndpoint handle,
                LwSciIpcEndpointInfoInternal *info);

/**
 * @brief Non-Blocking Read Peek Endpoint Interface
 *
 * @warning This API will be deprecated in future release.
 * Use LwSciIpcReadGetNextFrame() and memcpy() instead of this.
 *
 * This is partial update version of LwSciIpcReadGetNextFrame(), and it includes
 * memcpy procedure.
 * If read channel of the endpoint is not empty, copy specific portion of
 * the next Frame into a provided buffer.
 * If the destination buffer is smaller than the requested byte read count
 * the trailing bytes are lost.
 * This API doesn't change reference count of channel and doesn't send
 * notification to peer Endpoint.
 * This operation cannot proceed if the Endpoint is in reset. However,
 * if the remote Endpoint has called LwSciIpcResetEndpoint(), calls to this
 * function can still succeed until the next event notification on the local
 * Endpoint.
 * Do not read the same memory location multiple times. If required, copy
 * specific memory location to a local buffer before using it.
 * The user must make sure if actual input buffer size is equal or bigger than
 * requested size before using this API.
 *
 * @param[in]  handle  LwSciIpc Endpoint handle
 * @param[out] buf     A pointer to a destination buffer for the contents of the next Frame
 * @param[in]  offset  The offset bytes to a source frame buffer pointer, which should
 *                     be not greater than frame size of endpoint.
 * @param[in]  count   The number of bytes to be copied from the Frame.
 *                     The sum of offset+count should be not greater than frame size of endpoint.
 * @param[out] bytes   The number of bytes actually read on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates an invalid @a handle or @a count.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_InsufficientMemory Indicates memory allocation failed,
 *                                   the frames cannot be read.
 * - ::LwSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::LwSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @pre LwSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciIpcReadPeek(LwSciIpcEndpoint handle, void *buf, int32_t offset,
    int32_t count, int32_t *bytes);

/**
 * @brief Non-Blocking Write Poke Endpoint Interface
 *
 * @warning This API will be deprecated in future release.
 * Use LwSciIpcWriteGetNextFrame() and memcpy() instead of this.
 *
 * This is partial update version of LwSciIpcWriteGetNextFrame(), and it includes
 * memcpy procedure.
 * If space is available in the Endpoint, this function updates specific portion of
 * destination Frame with the contents from the provided data buffer.
 * If count is less than the frame size, then the remaining bytes of the frame
 * are undefined.
 * This API does not change reference count of channel and
 * does not send notification to peer Endpoint.
 * This operation cannot proceed if the Endpoint is in reset.
 * The user must make sure if actual input buffer size is equal or bigger than
 * requested size before using this API.
 *
 * @param[in]  handle  LwSciIpc Endpoint handle
 * @param[in]  buf     A pointer to a source buffer for the contents of the next Frame
 * @param[in]  offset  The offset bytes to a source frame buffer pointer, which
 *                     should be not greater than frame size of endpoint.
 * @param[in]  count   The number of bytes to be copied to the Frame.
 *                     The sum of offset+count should be not greater than frame size of endpoint.
 * @param[out] bytes   The number of bytes actually written on success.
 *
 * @return ::LwSciError, the completion code of operations:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates an invalid @a handle or @a count.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_InsufficientMemory Indicates memory allocation failed,
 *                                   the frames cannot be written.
 * - ::LwSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::LwSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @pre LwSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciIpcWritePoke(LwSciIpcEndpoint handle, const void *buf, int32_t offset,
    int32_t count, int32_t *bytes);

/**
 * @brief Checks if the read channel/queue contains data.
 *
 * This is same operation with LwSciIpcGetEvent() to check
 * if LW_SCI_IPC_EVENT_READ event is available in the provided endpoint.
 * This function is thread-safe but the lock is not used internally.
 *
 * @param[in] handle  LwSciIpc Endpoint handle
 *
 * @return True if the read channel is not empty, false otherwise.
 *
 * @pre LwSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
bool LwSciIpcCanRead(LwSciIpcEndpoint handle);

/**
 * @brief Checks if the write channel or send queue is available to accept data.
 *
 * This is same operation with LwSciIpcGetEvent() to check
 * if LW_SCI_IPC_EVENT_WRITE event is available in provided endpoint.
 * This function is thread-safe but lock is not used internally.
 *
 * @param[in] handle  LwSciIpc Endpoint handle
 *
 * @return True if the write channel is not full, false otherwise.
 *
 * @pre LwSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
bool LwSciIpcCanWrite(LwSciIpcEndpoint handle);

/**
 * @brief Gets the endpoint authentication token.
 *
 * @param[in]  handle     LwSciIpc endpoint handle.
 * @param[out] authToken  A authentication token of endpoint on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_BadParameter       Indicates an invalid @a handle.
 * - ::LwSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState      Indicates an invalid operation state.
 *
 * @pre Invocation of LwSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciIpcEndpointGetAuthToken(LwSciIpcEndpoint handle,
    LwSciIpcEndpointAuthToken *authToken);

#ifdef __QNX__
/**
 * @brief Validates endpoint authentication token.
 *
 * <b>This API is specific to QNX OS.</b>
 * Validate authentication token of endpoint and translate it to VUID.
 *
 * @param[in]  ctp            A context information pointer of QNX resource manager.
 * @param[in]  authToken      An authentication token of the endpoint.
 * @param[out] localUserVuid  A VUID of the endpoint on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_AccessDenied       Indicates invalid authentication token.
 * - ::LwSciError_BadParameter       Indicates an invalid parameter.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type or OS environment.
 * - ::LwSciError_NotPermitted       Indicates process doesn't have the required
 *                                   privilege.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @pre This API should be called from QNX resource manager.
 *      Invocation of LwSciIpcInit() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciIpcEndpointValidateAuthTokenQnx(resmgr_context_t *ctp,
    LwSciIpcEndpointAuthToken authToken,
    LwSciIpcEndpointVuid *localUserVuid);
#endif /* __QNX__ */

/**
 * @brief Translates VUID to peer topology ID and VUID.
 *
 * Translate VUID(VM unique ID) to topology ID and VUID of peer endpoint.
 *
 * @param[in]  localUserVuid  A VUID of LwSciIpc endpoint.
 * @param[out] peerTopoId     The topology ID of peer endpoint on success.
 * @param[out] peerUserVuid   A VUID of peer endpoint on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_NoSuchEntry        Indicates provided VUID is not found.
 * - ::LwSciError_BadParameter       Indicates an invalid parameter.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type or OS environment.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @pre Invocation of LwSciIpcInit() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciIpcEndpointMapVuid(LwSciIpcEndpointVuid localUserVuid,
    LwSciIpcTopoId *peerTopoId, LwSciIpcEndpointVuid *peerUserVuid);

/**
 * @brief Gets VUID of endpoint.
 *
 * Get VUID(VM unique ID) of endpoint.
 *
 * @param[in]  handle  A handle of LwSciIpc endpoint.
 * @param[out] vuid    A VUID of the endpoint on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_BadParameter       Indicates an invalid @a handle.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type or OS environment.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @pre Invocation of LwSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciIpcEndpointGetVuid(LwSciIpcEndpoint handle,
    LwSciIpcEndpointVuid *vuid);

#if (LW_IS_SAFETY == 0)
/**
 * @brief Gets topology ID of local endpoint.
 *
 * @param[in]  handle       A handle of LwSciIpc endpoint.
 * @param[out] localTopoId  The topology ID of local endpoint on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_BadParameter       Indicates an invalid input parameter.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type or OS environment.
 *
 * @pre Invocation of LwSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciIpcEndpointGetTopoId(LwSciIpcEndpoint handle,
    LwSciIpcTopoId *localTopoId);
#endif /* LW_IS_SAFETY == 0 */

/**
 * @brief Colwerts an OS-specific error code to a value in LwSciError.
 *
 * @param[in] err  OS-specific error code.
 *
 * @return Error code from ::LwSciError.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: Yes
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
LwSciError LwSciIpcErrnoToLwSciErr(int32_t err);

/**
 * @brief Colwerts an error code from LwSciError to an OS-specific error code.
 *
 * @param[in] lwSciErr  An error from LwSciError.
 *
 * @return
 * OS-specific error code.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: Yes
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
int32_t LwSciIpcLwSciErrToErrno(LwSciError lwSciErr);

#if (LW_IS_SAFETY == 0) && defined(IVC_EVENTLIB)
/**
 * @brief Callback function for eventlib logging.
 *
 * @param[in] id  IVC queue ID.
 *
 */
extern void (*LwSciIpcEventlibNotify)(uint32_t id);
#endif
/** @} <!-- End lwsci_ipc_internal --> */

#ifdef __cplusplus
}
#endif
#endif /* INCLUDED_LWSCIIPC_INTERNAL_H */
