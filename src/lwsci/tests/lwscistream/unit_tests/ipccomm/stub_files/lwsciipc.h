/*
 * Copyright (c) 2018-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIIPC_H
#define INCLUDED_LWSCIIPC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <lwscierror.h>
#include <lwscievent.h>

/**
 * @file
 *
 * @brief <b> LWPU Software Communications Interface (SCI) : LwSci Inter-Process Communication </b>
 *
 */
/**
 * @defgroup lwsci_group_ipc Inter-Process Communication
 * IPC and Event Service APIs
 *
 * @ingroup lwsci_top
 * @{
 */
/**
 * @defgroup lwsci_ipc_api IPC APIs
 *
 *
 * @ingroup lwsci_group_ipc
 * @{
 *
 * The LwSciIpc library provides interfaces for any two entities in a system to
 * communicate with each other irrespective of where they are placed. Entities
 * can be in:
 * - Different threads in the same process
 * - The same process
 * - Different processes in the same VM
 * - Different VMs on the same SoC
 * @if (SWDOCS_LWSCIIPC_STANDARD)
 * - Different SoCs
 * @endif
 *
 * Each of these different boundaries will be abstracted by a library providing
 * unified communication (Read/Write) APIs to entities. The communication
 * consists of two bi-directional send/receive queues.
 *
 * When Init operation group APIs are used, the user should call them in the
 * following order, with or without LwSciEventService.
 *
 * <b> Typical call flow with LwSciIpc library </b>
 *
 * 1) Init mode
 *    - LwSciIpcInit()
 *    - LwSciIpcOpenEndpoint()
 *    - Set event reporting path
 *      LwSciIpcSetQnxPulseParam() (QNX OS-specific) or
 *      LwSciIpcGetLinuxEventFd() (Linux OS-specific)
 *    - LwSciIpcGetEndpointInfo()
 *    - LwSciIpcResetEndpoint()
 *    - Ensure a channel is established
 * ~~~~~~~~~~~~~~~~~~~~~
 *      loop {
 *          LwSciIpcGetEvent()
 *          if (event & LW_SCI_IPC_EVENT_CONN_EST_ALL) break
 *          else {
 *              MsgReceivePulse_r() (QNX OS-specific) or
 *              select(), epoll() (Linux OS-specific)
 *          }
 *      }
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * 2) Runtime mode (loop)
 *    - LwSciIpcGetEvent()
 *    - If an event is not desired,
 *      call OS-blocking API
 *      MsgReceivePulse_r() (QNX OS-specific) or
 *      select(), epoll() (Linux OS-specific)
 *    - LwSciIpcRead() or LwSciIpcWrite()
 *
 * 3) De-Init mode
 *    - LwSciIpcCloseEndpoint()
 *    - LwSciIpcDeinit()
 *
 * <b> Typical call flow with LwSciIpc and LwSciEventService library </b>
 *
 * LwSciEventService provides APIs that replace OS-specific event-blocking API.
 * They are only compatible with an endpoint which is opened with
 * LwSciOpenEndpointWithEventService().
 *
 * 1) Init mode
 *    - LwSciEventLoopServiceCreate() to get eventLoopService
 *    - LwSciIpcInit()
 *    - LwSciIpcOpenEndpointWithEventService()
 *    - LwSciIpcGetEventNotifier() to get eventNotifier
 *    - LwSciIpcGetEndpointInfo()
 *    - LwSciIpcResetEndpoint()
 *    - Ensure a channel is established
 * ~~~~~~~~~~~~~~~~~~~~~
 *      loop {
 *          LwSciIpcGetEvent()
 *          if (event & LW_SCI_IPC_EVENT_CONN_EST_ALL) break
 *          else {
 *              eventLoopService->WaitForEvent(eventNotifier)
 *          }
 *      }
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * 2) Runtime mode (loop)
 *    - LwSciIpcGetEvent()
 *    - If an event is not desired,
 *    - eventLoopService->WaitForEvent()
 *    - LwSciIpcRead() or LwSciIpcWrite()
 *
 * 3) De-Init mode
 *    - eventNotifier->Delete()
 *    - LwSciIpcCloseEndpoint()
 *    - LwSciIpcDeinit()
 *    - eventLoopService->EventService.Delete()
 *
 * <b>Using multi-threading in LwSciIpc - QNX OS</b>
 *
 * For Inter-VM and Inter-Process backend applications on QNX OS,
 * only a single event-blocking OS API (i.e. MsgReceivePulse_r(),
 * LwSciEventLoopService::WaitForEvent()) is allowed per endpoint
 * in the same process boundary.
 * If a client application tries to use receiving and sending thread separately for
 * the same endpoint handle, the event-blocking OS APIs must be used in a
 * single thread in order to receive remote notification.
 * Once a remote notification arrives in the thread, the notification should be forwarded
 * to the other thread using the same endpoint handle through any OS synchronization
 * method (e.g. sem_post, pthread_cond_signal or MsgSendPulse etc.)
 * Single thread usage is recommended to handle both TX and RX data.
 *
 * <b>Configuring thread pool of resource manager - QNX OS</b>
 *
 * LwSciIpc resource manager (io-lwsciipc) and IVC resource manager (devv-lwivc)
 * use thread pool to manage conlwrrent request from multiple LwSciIpc client
 * processes using LwSciIpc library.
 * io-lwsciipc is used during opening endpoint and devv-lwivc is used for
 * Inter-VM IVC signaling.
 * Drive OS users should evaluate thread pool capacity of io-lwsciipc and
 * devv-lwivc then configure them with -t option in startup script.
 * Thread pool capacity for LwSciIPC can be evaluated based on number of
 * parallel outstanding LwSciIPC requests, at any point of time, that are
 * expected in the system. Default value of thread pool capacity is 10.
 *
 * <b>Getting events before using Read/Write LwSciIpc API</b>
 *
 * Before using any Read/Write APIs, the user must check if @ref LW_SCI_IPC_EVENT_READ
 * or @ref LW_SCI_IPC_EVENT_WRITE event is available through LwSciIpcGetEvent().
 * LwSciIpcGetEvent() has additional support to establish connection between
 * two endpoint software entities.
 *
 * <b>When to use blocking API</b>
 *
 * Users of LwSciIpc must call OS event-blocking API to wait for an event when
 * LwSciIpcGetEvent() does not return desired events.
 * The following are OS event-blocking API examples:
 * - QNX  : MsgReceivePulse_r()
 * - LINUX: select(), epoll() etc.
 * - LwSciEventService: LwSciEventLoopService::WaitForEvent(),<br/>
 *                      LwSciEventLoopService::WaitForMultipleEvents()
 *
 * If user process needs to wait for events from multiple remote LwSciIpc
 * endpoint processes, use single blocking call from single thread instead of
 * using blocking call per endpoint thread. This is recommended to improve
 * performance by avoiding thread creation per endpoint.
 * LwSciEventLoopService::WaitForMultipleEvents() blocking call is suitable for
 * this use case.
 *
 * <b>How to check if peer endpoint entity receives a message</b>
 *
 * LwSciIpc library does not provide information about whether a peer endpoint
 * entity receives all sent messages from a local endpoint entity.
 * If such a mechanism is required, the client user should implement separate
 * message acknowledgment in the application layer.
 *
 * <b>Recommended Read/Write APIs</b>
 *
 * Using LwSciIpcRead() and LwSciIpcWrite() is recommended rather than following
 * Read/Write APIs. See detail constraints of API in each function description.
 * - LwSciIpcReadGetNextFrame()
 * - LwSciIpcWriteGetNextFrame()
 * - LwSciIpcReadAdvance()
 * - LwSciIpcWriteAdvance()
 * However, above functions are better to avoid extra memory copy.
 *
 * <b>Provide valid buffer pointers</b>
 *
 * The user of LwSciIpc must provide valid buffer pointers to LwSciIpcRead(),
 * LwSciIpcWrite() and other Read/Write LwSciIpc APIs as LwSciIpc library
 * validation to these parameters is limited to a NULL pointer check.
 *
 * <b>Maximum number of endpoints</b>
 *
 * One LwSciIpc client process is allowed to open up to 50 endpoints.
 * QNX OS safety manual imposes a restriction no process SHALL have more than
 * 100 open channels without disabling kernel preemption. User client needs
 * one channel/connection pair to receive an endpoint notification.
 *
 */

/*******************************************************************/
/************************ DATA TYPES *******************************/
/*******************************************************************/
/* implements_unitdesign QNXBSP_LWSCIIPC_LIBLWSCIIPC_40 */

/**
 * @brief Handle to the LwSciIpc endpoint.
 */
typedef uint64_t LwSciIpcEndpoint;

/**
 * @brief Defines information about the LwSciIpc endpoint.
 */
struct LwSciIpcEndpointInfo {
    /** Holds the number of frames. */
    uint32_t nframes;
    /** Holds the frame size in bytes. */
    uint32_t frame_size;
};

/**
 * Specifies maximum Endpoint name length
 * including null terminator
 */
#define LWSCIIPC_MAX_ENDPOINT_NAME   64U

/* LwSciIPC Event type */
/** Specifies the IPC read event. */
#define	LW_SCI_IPC_EVENT_READ       1U
/** Specifies the IPC write event. */
#define	LW_SCI_IPC_EVENT_WRITE      2U
/** Specifies the IPC connection established event. */
#define	LW_SCI_IPC_EVENT_CONN_EST   4U
/** Specifies the IPC connection reset event. */
#define	LW_SCI_IPC_EVENT_CONN_RESET 8U
/** Specifies single event mask to check IPC connection establishment */
#define	LW_SCI_IPC_EVENT_CONN_EST_ALL (LW_SCI_IPC_EVENT_CONN_EST | LW_SCI_IPC_EVENT_WRITE | LW_SCI_IPC_EVENT_READ)

/*******************************************************************/
/********************* FUNCTION TYPES ******************************/
/*******************************************************************/

/**
 * @brief Initializes the LwSciIpc library.
 *
 * This function parses the LwSciIpc configuration file and creates
 * an internal database of LwSciIpc endpoints that exist in a system.
 *
 * @return ::LwSciError, the completion code of the operation.
 * - ::LwSciError_Success      Indicates a successful operation.
 * - ::LwSciError_NotPermitted Indicates initialization has failed.
 * - ::LwSciError_IlwalidState Indicates an invalid operation state.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
LwSciError LwSciIpcInit(void);

/**
 * @brief De-initializes the LwSciIpc library.
 *
 * This function cleans up the LwSciIpc endpoint internal database
 * created by LwSciIpcInit().
 * Before calling this API, all existing opened endpoints must be closed
 * by LwSciIpcCloseEndpoint().
 *
 * @return @c void
 *
 * @pre Invocation of LwSciIpcInit() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: No
 *   - De-initialization: Yes
 */
void LwSciIpcDeinit(void);

/**
 * @brief Opens an endpoint with the given name.
 *
 * The function locates the LwSciIpc endpoint with the given name in the
 * LwSciIpc configuration table in the internal database, and returns a handle
 * to the endpoint if found. When the operation is successful, endpoint can
 * utilize the allocated shared data area and the corresponding signaling
 * mechanism setup. If the operation fails, the state of the LwSciIpc endpoint
 * is undefined.
 * In case of QNX OS, in order to authenticate user client process, LwSciIpc
 * uses custom ability "LwSciIpcEndpoint". Use procmgr_ability_lookup()
 * QNX OS API to get ability ID.
 *
 * @param[in]  endpoint The name of the LwSciIpc endpoint to open.
 * @param[out] handle   A handle to the endpoint on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates any argument was NULL or invalid.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_NoSuchEntry        Indicates the @a endpoint was not found.
 * - ::LwSciError_Busy               Indicates the @a endpoint is already in use.
 * - ::LwSciError_InsufficientMemory Indicates memory allocation failed for the operation.
 * - ::LwSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 * - ::LwSciError_NotPermitted       Indicates process doesn't have the required
 *                                   privilege.
 *
 * @pre Invocation of LwSciIpcInit() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): PROCMGR_AID_MEM_PHYS, "LwSciIpcEndpoint"
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
LwSciError LwSciIpcOpenEndpoint(const char *endpoint, LwSciIpcEndpoint *handle);

/**
 * @brief Opens an endpoint with the given name and event service.
 *
 * This API provides same functionality as LwSciIpcOpenEndpoint().
 * But, it requires additional event service abstract object as an input
 * parameter to utilize LwSciEventService infrastructure.
 * LwSciEventService can be created through LwSciEventLoopServiceCreate().
 * LwSciIpcGetEventNotifier() can be used only when this API is ilwoked
 * successfully.
 * In case of QNX OS, in order to authenticate user client process, LwSciIpc
 * uses custom ability "LwSciIpcEndpoint". Use procmgr_ability_lookup()
 * QNX OS API to get ability ID.
 *
 * @param[in]  endpoint      The name of the LwSciIpc endpoint to open.
 * @param[out] handle        A handle to the endpoint on success.
 * @param[in]  eventService  An abstract object to use LwSciEventService infrastructure.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates any argument was NULL or invalid.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_NoSuchEntry        Indicates the @a endpoint was not found.
 * - ::LwSciError_Busy               Indicates the @a endpoint is already in use.
 * - ::LwSciError_InsufficientMemory Indicates memory allocation failed for the operation.
 * - ::LwSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 * - ::LwSciError_NotPermitted       Indicates process doesn't have the required
 *                                   privilege.
 *
 * @pre Invocation of LwSciEventLoopServiceCreate() must be successful.
 *      Invocation of LwSciIpcInit() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): PROCMGR_AID_MEM_PHYS, "LwSciIpcEndpoint"
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
LwSciError LwSciIpcOpenEndpointWithEventService(const char *endpoint,
    LwSciIpcEndpoint *handle, LwSciEventService *eventService);

/**
 * @brief Get LwSciIpc event notifier.
 *
 * This API is used to connect LwSciIpc event handling with OS-provided
 * event interface.
 * It also utilizes LwSciEventService infrastructure.
 * Before calling LwSciIpcCloseEndpoint(), event notifier should be deleted
 * through Delete callback of LwSciEventNotifier.
 *
 * @note This API is only compatible with an endpoint that is opened with
 *       LwSciIpcOpenEndpointWithEventService()
 *
 * @param[in]  handle         LwSciIpc endpoint handle.
 * @param[out] eventNotifier  A pointer to LwSciEventNotifier object on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_BadParameter       Indicates an invalid or NULL argument.
 * - ::LwSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 * - ::LwSciError_InsufficientMemory Indicates memory allocation failed for the
 *                                   operation.
 * - ::LwSciError_ResourceError      Indicates not enough system resources.
 *
 * @pre Invocation of LwSciIpcOpenEndpointWithEventService() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
LwSciError LwSciIpcGetEventNotifier(LwSciIpcEndpoint handle,
               LwSciEventNotifier **eventNotifier);

/**
 * @brief Closes an endpoint with the given handle.
 *
 * The function frees the LwSciIpc endpoint associated with the given @a handle.
 *
 * @param[in] handle A handle to the endpoint to close.
 *
 * @return @c void
 *
 * @pre Invocation of LwSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: No
 *   - De-initialization: Yes
 */
void LwSciIpcCloseEndpoint(LwSciIpcEndpoint handle);

/**
 * @brief Resets an endpoint.
 *
 * Initiates a reset on the endpoint and notifies the remote endpoint.
 * Applications must call this function and complete the reset operation before
 * using the endpoint for communication.
 * Once this API is called, all existing data in channel will be discarded.
 * After ilwoking this function, client user shall call LwSciIpcGetEvent()
 * to get specific event type (READ, WRITE etc.). if desired event is not
 * returned from GetEvent API, OS-specific blocking call (select/poll/epoll
 * or MsgReceivePulse) should be called to wait remote notification.
 * This sequence must be done repeatedly to get event type that
 * endpoint wants.
 *
 * @param[in] handle A handle to the endpoint to reset.
 *
 * @return @c void
 *
 * @pre Invocation of LwSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: Yes
 *   - De-initialization: No
 */
void LwSciIpcResetEndpoint(LwSciIpcEndpoint handle);

/**
 * @brief Returns the contents of the next frame from an endpoint.
 *
 * This function removes the next frame and copies its contents
 * into a buffer. If the destination buffer is smaller than the configured
 * frame size of the endpoint, the trailing bytes are discarded.
 *
 * This is a non-blocking call. Read channel of the endpoint must not be empty.
 * If read channel of the endpoint was previously full, then the function
 * notifies the remote endpoint.
 *
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called LwSciIpcResetEndpoint(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 *
 * The user shall make sure if actual input buffer size is equal or bigger than
 * requested size before using this API.
 *
 * @param[in]  handle The handle to the endpoint to read from.
 * @param[out] buf    A pointer to a destination buffer to receive the contents
 *                    of the next frame.
 * @param[in]  size   The number of bytes to copy from the frame. If @a size
 *                    is greater than the size of the destination buffer, the
 *                    remaining bytes are discarded.
 * @param[out] bytes  The number of bytes read on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates an invalid @a handle or @a size.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_InsufficientMemory Indicates read channel is empty and the read
 *                                   operation aborted.
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
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
LwSciError LwSciIpcRead(LwSciIpcEndpoint handle, void *buf, size_t size,
	int32_t *bytes);

/**
 * @brief Returns a pointer to the location of the next frame from an endpoint.
 *
 * This is a non-blocking call.
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called LwSciIpcResetEndpoint(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 * Between LwSciIpcReadGetNextFrame() and LwSciIpcReadAdvance(), do not perform
 * any other LwSciIpc read operations with the same endpoint handle.
 * Once a read frame is released by LwSciIpcReadAdvance(), do not use previously
 * returned pointer of LwSciIpcReadGetNextFrame() since it is already invalid.
 * Do not write through a returned pointer of LwSciIpcReadGetNextFrame().
 * This is protected by a const volatile pointer return type.
 * Do not read the same memory location multiple times. If required, copy
 * specific memory location to a local buffer before using it.
 *
 * @param[in]  handle The handle to the endpoint to read from.
 * @param[out] buf    A pointer to a destination buffer to receive
 *                    the contents of the next frame on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates an invalid @a handle.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_InsufficientMemory Indicates read channel is empty and
 *                                   the read operation aborted.
 * - ::LwSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @pre LwSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
LwSciError LwSciIpcReadGetNextFrame(LwSciIpcEndpoint handle,
    const volatile void **buf);

/**
 * @brief Removes the next frame from an endpoint.
 *
 * This is a non-blocking call. Read channel of the endpoint must not be empty.
 * If a read channel of the endpoint was previously full, then this function
 * notifies the remote endpoint.
 *
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called LwSciIpcResetEndpoint(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 *
 * Between LwSciIpcReadGetNextFrame() and LwSciIpcReadAdvance(), do not perform
 * any other LwSciIpc read operations with the same endpoint handle.
 * Once a read frame is released by LwSciIpcReadAdvance(), do not use previously
 * returned pointer of LwSciIpcReadGetNextFrame() since it is already invalid.
 *
 * @param[in] handle The handle to the endpoint to read from.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates the frame was removed successfully.
 * - ::LwSciError_BadParameter       Indicates an invalid @a handle.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_InsufficientMemory Indicates read channel is empty and the read
 *                                   operation aborted.
 * - ::LwSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @pre LwSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
LwSciError LwSciIpcReadAdvance(LwSciIpcEndpoint handle);

/**
 * @brief Writes a new frame to the endpoint.
 *
 * If space is available in the endpoint, this function posts a new frame,
 * copying the contents from the provided data buffer.
 * If @a size is less than the frame size, then the remaining bytes of the frame
 * are undefined.
 *
 * This is a non-blocking call.
 * If write channel of the endpoint was previously empty, then the function
 * notifies the remote endpoint.
 *
 * This operation cannot proceed if the endpoint is being reset.
 *
 * The user shall make sure if actual input buffer size is equal or bigger than
 * requested size before using this API.
 *
 * @param[in]  handle The handle to the endpoint to write to.
 * @param[in]  buf    A pointer to a source buffer for the contents of
 *                    the next frame.
 * @param[in]  size   The number of bytes to be copied to the frame,
 *                    not to exceed the length of the destination buffer.
 * @param[out] bytes  The number of bytes written on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates an invalid @a handle or @a size.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_InsufficientMemory Indicates write channel is full and the write
 *                                   operation aborted.
 * - ::LwSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @pre LwSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
LwSciError LwSciIpcWrite(LwSciIpcEndpoint handle, const void *buf, size_t size,
	int32_t *bytes);

/**
 * @brief Returns a pointer to the location of the next frame for writing data.
 *
 * This is a non-blocking call. write channel of the endpoint must not be full.
 *
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called LwSciIpcResetEndpoint(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 * Between LwSciIpcWriteGetNextFrame() and LwSciIpcWriteAdvance(), do not
 * perform any other LwSciIpc write operations with the same endpoint handle.
 * Once a transmit message is committed by LwSciIpcWriteAdvance(), do not use
 * previously returned pointer of LwSciIpcWriteGetNextFrame() since it is already
 * invalid.
 * Do not read through a returned pointer of LwSciIpcWriteGetNextFrame().
 *
 * @param[in]  handle The handle to the endpoint to write to.
 * @param[out] buf    A pointer to a destination buffer to hold the contents of
 *                    the next frame on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates successful operation.
 * - ::LwSciError_BadParameter       Indicates an invalid @a handle.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_InsufficientMemory Indicates write channel is full and
 *                                   the write operation aborted.
 * - ::LwSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @pre LwSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
LwSciError LwSciIpcWriteGetNextFrame(LwSciIpcEndpoint handle,
    volatile void **buf);

/**
 * @brief Writes the next frame to the endpoint.
 *
 * This is a non-blocking call.
 * If write channel of the endpoint is not full, then post the next frame.
 * If write channel of the endpoint was previously empty, then this function
 * notifies the remote endpoint.
 *
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called LwSciIpcResetEndpoint(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 *
 * Between LwSciIpcWriteGetNextFrame() and LwSciIpcWriteAdvance(), do not
 * perform any other LwSciIpc write operations with the same endpoint handle.
 * Once transmit message is committed by LwSciIpcWriteAdvance(), do not use
 * previously returned pointer of LwSciIpcWriteGetNextFrame() since it is already
 * invalid.
 *
 * @param[in] handle The handle to the endpoint to write to.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates successful operation.
 * - ::LwSciError_BadParameter       Indicates an invalid @a handle.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_InsufficientMemory Indicates write channel is full and the write
 *                                   operation aborted.
 * - ::LwSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @pre LwSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
LwSciError LwSciIpcWriteAdvance(LwSciIpcEndpoint handle);

/**
 * @brief Returns endpoint information.
 *
 *
 * @param[in]  handle LwSciIpc endpoint handle.
 * @param[out] info   A pointer to LwSciIpcEndpointInfo object that
 *                    this function copies the info to on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_BadParameter       Indicates an invalid or NULL argument.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 *
 * @pre Invocation of LwSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: Yes
 *   - De-initialization: No
 */
LwSciError LwSciIpcGetEndpointInfo(LwSciIpcEndpoint handle,
                struct LwSciIpcEndpointInfo *info);

#ifndef __QNXNTO__
/**
 * Returns the LwSciIpc file descriptor for a given endpoint.
 *
 * <b> This API is specific to Linux OS. </b>
 * Event handle will be used to plug OS event notification
 * (can be read, can be written, established, reset etc.)
 *
 * @param handle LwSciIpc endpoint handle
 * @param fd     A pointer to the endpoint file descriptor.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_BadParameter       Indicates an invalid or NULL argument.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type or OS environment.
 */
LwSciError LwSciIpcGetLinuxEventFd(LwSciIpcEndpoint handle, int32_t *fd);
#endif /* !__QNXNTO__ */

/**
 * @brief Get Events
 *
 * Returns a bitwise OR operation on new events that oclwrred since the
 * last call to this function.
 *
 * This function sets @a events to the result of a bitwise OR operation of zero
 * or more @c LW_SCI_IPC_EVENT_* constants corresponding to all new events that
 * have oclwrred on the endpoint since:
 * - the preceding call to this function on the endpoint or
 * - opening the endpoint, if this is the first call to this function on the
 *   endpoint since it was opened.
 *
 * The parameter @a events is set to zero if no new events have
 * oclwrred.
 *
 * There are four types of events:
 * - @c LW_SCI_IPC_EVENT_CONN_EST   : IPC connection established
 * - @c LW_SCI_IPC_EVENT_WRITE      : IPC write
 * - @c LW_SCI_IPC_EVENT_READ       : IPC read
 * - @c LW_SCI_IPC_EVENT_CONN_RESET : IPC connection reset
 *
 * These may occur in arbitrary combinations, except for the following:
 * - @c LW_SCI_IPC_EVENT_CONN_EST is always combined with @c LW_SCI_IPC_EVENT_WRITE.
 * - @c LW_SCI_IPC_EVENT_CONN_RESET cannot be combined with any other events.

 * There are seven possible event combinations:
 * - 0
 * - @c LW_SCI_IPC_EVENT_CONN_EST and @c LW_SCI_IPC_EVENT_WRITE
 * - @c LW_SCI_IPC_EVENT_CONN_EST and @c LW_SCI_IPC_EVENT_WRITE and
 *   @c LW_SCI_IPC_EVENT_READ
 * - @c LW_SCI_IPC_EVENT_READ
 * - @c LW_SCI_IPC_EVENT_WRITE
 * - @c LW_SCI_IPC_EVENT_WRITE and @c LW_SCI_IPC_EVENT_READ
 * - @c LW_SCI_IPC_EVENT_CONN_RESET
 *
 * An @c LW_SCI_IPC_EVENT_CONN_EST event oclwrs on an endpoint each time a
 * connection is established through the endpoint (between the endpoint and
 * the other end of the corresponding channel).
 *
 * An @c LW_SCI_IPC_EVENT_WRITE event oclwrs on an endpoint:
 * -# In conjunction with the delivery of each @c LW_SCI_IPC_CONN_EST event.
 * -# Each time the endpoint ceases to be full after a prior @c LwSciIpcWrite*
 * call returned @c LwSciError_InsufficientMemory. Note however that an
 * implementation is permitted to delay the delivery of this type of
 * @c LW_SCI_IPC_EVENT_WRITE event, e.g., for purposes of improving throughput.
 *
 * An @c LW_SCI_IPC_EVENT_READ event oclwrs on an endpoint:
 * -# In conjunction with the delivery of each @c LW_SCI_IPC_EVENT_CONN_EST event,
 * if frames can already be read as of delivery.
 * -# Each time the endpoint ceases to be empty after a prior @c LwSciRead*
 * call returned @c LwSciError_InsufficientMemory. Note however that an
 * implementation is permitted to delay the delivery of this type of
 * @c LW_SCI_IPC_EVENT_READ event, e.g., for purposes of improving throughput.
 *
 * An @c LW_SCI_IPC_EVENT_CONN_RESET event oclwrs on an endpoint when the user
 * calls LwSciIpcResetEndpoint.
 *
 * If this function doesn't return desired events, user must call
 * OS-provided blocking API to wait for notification from remote endpoint.
 *
 * The following are blocking API examples:
 * - QNX  : MsgReceivePulse_r()
 * - LINUX: select(), epoll() etc.
 * - LwSciEventService: LwSciEventLoopService::WaitForEvent(), <br/>
 *                      LwSciEventLoopService::WaitForMultipleEvents()
 *
 * In case of QNX OS, in order to authenticate user client process, LwSciIpc
 * uses custom ability "LwSciIpcEndpoint". Use procmgr_ability_lookup()
 * QNX OS API to get ability ID.
 *
 * @param[in]  handle LwSciIpc endpoint handle.
 * @param[out] events  A pointer to the variable into which to store
 *                    the bitwise OR result of new events on success.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_BadParameter       Indicates an invalid or NULL argument.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 * - ::LwSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 *
 * @pre LwSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): PROCMGR_AID_MEM_PHYS, "LwSciIpcEndpoint"
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: Yes
 *   - De-initialization: No
 */
LwSciError LwSciIpcGetEvent(LwSciIpcEndpoint handle, uint32_t *events);

#ifdef __QNXNTO__
/**
 * @brief Sets the event pulse parameters for QNX.
 *
 * <b>This API is specific to QNX OS.</b>
 * When a notification from a peer endpoint is available, the LwSciIpc library
 * sends a pulse message to the application.
 * This API is to connect @a coid to the endpoint, plug OS event notification
 * and set pulse parameters (@a pulsePriority, @a pulseCode and @a pulseValue),
 * thereby enabling the application to receive peer notifications from the
 * LwSciIpc library.
 * An application can receive notifications from a peer endpoint using
 * @c MsgReceivePulse_r() which is blocking call.
 *
 * Prior to calling this function, both @c ChannelCreate_r() and @c ConnectAttach_r()
 * must be called in the application to obtain the value for @a coid to pass to
 * this function.
 *
 * To use the priority of the calling thread, set @a pulsePriority to
 * @c SIGEV_PULSE_PRIO_INHERIT(-1). The priority must fall within the valid
 * range, which can be determined by calling @c sched_get_priority_min() and
 * @c sched_get_priority_max().
 *
 * Applications can define any value per endpoint for @a pulseCode and @a pulseValue.
 * @a pulseCode will be used by LwSciIpc to signal IPC events and should be
 * reserved for this purpose by the application. @a pulseValue can be used
 * for the application cookie data.
 *
 * @note It is only compatible with an endpoint that is opened with
 *       LwSciIpcOpenEndpoint().
 *
 * @param[in] handle        LwSciIpc endpoint handle.
 * @param[in] coid          The connection ID created from calling @c ConnectAttach_r().
 * @param[in] pulsePriority The value for pulse priority.
 * @param[in] pulseCode     The 8-bit positive pulse code specified by the user. The
 *                          values must be between @c _PULSE_CODE_MINAVAIL and
 *                          @c _PULSE_CODE_MAXAVAIL
 * @param[in] pulseValue    A pointer to the user-defined pulse value.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_NotInitialized  Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_BadParameter    Indicates an invalid @a handle.
 * - ::LwSciError_NotSupported    Indicates API is not supported in provided
 *                                endpoint backend type or OS environment.
 * - ::LwSciError_ResourceError   Indicates not enough system resources.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 *
 * @pre Invocation of LwSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): PROCMGR_AID_INTERRUPTEVENT
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
LwSciError LwSciIpcSetQnxPulseParam(LwSciIpcEndpoint handle,
	int32_t coid, int16_t pulsePriority, int16_t pulseCode,
	void *pulseValue);
#endif /* __QNXNTO__ */
/** @} <!-- End lwsci_ipc_api --> */
/** @} <!-- End lwsci_group_ipc --> */

#ifdef __cplusplus
}
#endif
#endif /* INCLUDED_LWSCIIPC_H */
