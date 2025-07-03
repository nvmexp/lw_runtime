/*
 * Copyright (c) 2019-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIEVENT_H
#define INCLUDED_LWSCIEVENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <lwscierror.h>

/**
 * @file
 *
 * @brief <b> LWPU Software Communications Interface (SCI) : LwSci Event Service </b>
 *
 */
/**
 * @defgroup lwsci_ipc_event Event Service APIs
 *
 * @ingroup lwsci_group_ipc
 * @{
 *
 * The LwSciEventService library provides the ability to build portable
 * event-driven applications.
 * An event is any significant oclwrrence or change in the state for system hardware
 * or software. An event notification is a message or notification sent by one
 * software part to another to convey that an event has taken place.
 * An event-driven model consists of an event producer and event consumers.
 * Once an event producer detects an event, it represents the event as
 * a message (or notification). An event is transmitted from an event producer to
 * event consumers through an OS-specific event channel.
 * Event consumers must be informed when an event has oclwrred.
 * LwSciEventService will execute the correct response (or callback)
 * to an event.
 *
 * LwSciEventService provides a mandatory abstract interface between
 * other SCI technologies (especially LwSciIpc and LwSciStreams) and
 * the application-provided event loop that services them.
 *
 * The following common object type(s) are implemented:
 *
 * - User-visible object types (for application use)
 *    - LwSciEventService: An object that subsumes all state that commonly would
 *      have been maintained in global variables.
 *    - LwSciEventNotifier: An object that a library creates using an
 *      LwSciEventService and then provides to the user, and with which the user
 *      registers an event handler that is ilwoked whenever the library
 *      generates an event.
 *
 * @if (SWDOCS_LWSCIIPC_INTERNAL)
 * - Non-user-visible object types (for integrating libraries with an
 *    LwSciEventService)
 *    - LwSciNativeEvent: An object that a library fills in with
 *      environment-specific information that is necessary for an event service
 *      to wait for environment-specific events (from OS or other event
 *      producers).
 *    - LwSciLocalEvent: An object with which a library can signal events
 *      directly, without going through environment-specific mechanisms. Local
 *      events are limited to cases where the signaler and waiter are in the
 *      same process, but may be more efficient than environment-specific
 *      notifiers (which typically pass through an OS).
 * @endif
 *
 * <b>Typical call flow with LwSciIpc library</b>
 *
 * 1) Init mode
 *    - LwSciEventLoopServiceCreate()
 *    - LwSciIpcInit()
 *    - LwSciIpcOpenEndpointWithEventService()
 *    - LwSciIpcGetEventNotifier()
 *    - LwSciIpcGetEndpointInfo()
 *    - LwSciIpcResetEndpoint()
 *    - Ensure a channel is established
 * ~~~~~~~~~~~~~~~~~~~~~
 *      loop {
 *          LwSciIpcGetEvent()
 *          if (event & LW_SCI_IPC_EVENT_CONN_EST_ALL) break
 *          else {
 *              LwSciEventLoopService::WaitForEvent()
 *              or
 *              LwSciEventLoopService::WaitForMultipleEvents()
 *          }
 *      }
 * ~~~~~~~~~~~~~~~~~~~~~
 * 2) Runtime mode (loop)
 *    - LwSciIpcGetEvent()
 *    - If an event is not desired,
 *      LwSciEventLoopService::WaitForEvent()
 *      or
 *      LwSciEventLoopService::WaitForMultipleEvents()
 *    - LwSciIpcRead() or LwSciIpcWrite()
 *
 * 3) Deinit mode
 *    - If an eventNotifier is not required any more,
 *      LwSciEventNotifier::Delete()
 *    - LwSciIpcCloseEndpoint()
 *    - LwSciIpcDeinit()
 *    - LwSciEventService::Delete()
 */

/*****************************************************************************/
/*                               DATA TYPES                                  */
/*****************************************************************************/

/**
 * \brief Infinite timeout for LwSciEventLoopService::WaitForEvent() or
 * LwSciEventLoopService::WaitForMultipleEvents().
 */
#define LW_SCI_EVENT_INFINITE_WAIT -1
#define LW_SCI_EVENT_PRIORITIES 4

typedef struct LwSciEventService LwSciEventService;
typedef struct LwSciEventNotifier LwSciEventNotifier;
typedef struct LwSciEventLoopService LwSciEventLoopService;

/// @cond (SWDOCS_LWSCIIPC_INTERNAL)
typedef struct LwSciNativeEvent LwSciNativeEvent;
typedef struct LwSciLocalEvent LwSciLocalEvent;
typedef struct LwSciTimerEvent LwSciTimerEvent;
typedef struct LwSciEventLoop LwSciEventLoop;
/// @endcond

/**
 * \struct LwSciEventService
 * \brief An abstract interface for a program's event handling infrastructure.
 *
 * An LwSciEventService is an abstraction that a library can use to interact
 * with the event handling infrastructure of the containing program.
 *
 * If a library needs to handle asynchronous events or report asynchronous
 * events to its users, but the library does not wish to impose a threading
 * model on its users, the library can require each user to provide an
 * LwSciEventService when the user initializes the library (or a portion
 * thereof).
 *
 * An LwSciEventService provides two categories of services related to event
 * handling:
 *
 * (1) The ability to define "event notifiers", which are objects that can
 *     notify event handling infrastructure each time an event has oclwrred.
 *     Note that event notifications carry no payload; it is expected that any
 *     event payload information is colweyed separately.
 *
 * (2) The ability to bind an "event handler" to each event notifier. An event
 *     handler is essentially a callback that is ilwoked each time the bound
 *     event notifier reports the oclwrrence of an event.
 */
struct LwSciEventService {
    /**
     * @if (SWDOCS_LWSCIIPC_INTERNAL)
     * \brief Defines an event notifier for a native notifier.
     *
     * @note This API is for internal use only.
     *
     * The new LwSciEventNotifier will report the oclwrrence of an event to
     * the event service each time the provided native notifier reports an
     * event from the OS environment.
     *
     * This function creates event notifier which reports the oclwrrence of
     * an event from the OS environment to the event service.
     * To configure the event bound to OS environment, it calls the function
     * in @a nativeEvent with the notifier pointer, which is a supported function
     * in the LwSciIpc library.
     *
     * @param[in]   thisEventService LwSciEventService object pointer created by
     *                               LwSciEventLoopServiceCreate().
     * @param[in]   nativeEvent      LwSciNativeEvent object pointer.
     * @param[out]  newEventNotifier LwSciEventNotifier object pointer on
     *                               success.
     *
     * @return ::LwSciError, the completion code of operations:
     * - ::LwSciError_Success         Indicates a successful operation.
     * - ::LwSciError_InsufficientMemory  Indicates memory is not sufficient.
     * - ::LwSciError_BadParameter    Indicates an invalid input parameters.
     * - ::LwSciError_ResourceError   Indicates not enough system resources.
     * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
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
     *   - Runtime: No
     *   - De-Init: No
     * @endif
     */
    /// @cond (SWDOCS_LWSCIIPC_INTERNAL)
    LwSciError (*CreateNativeEventNotifier)(
            LwSciEventService* thisEventService,
            LwSciNativeEvent* nativeEvent,
            LwSciEventNotifier** newEventNotifier);
    /// @endcond

    /**
     * @if (SWDOCS_LWSCIIPC_INTERNAL)
     * \brief Creates an intra-process local event with an event notifier
     *        that reports each event signaled through it.
     *
     * @note This API is for internal use only.
     *
     * @param[in]   thisEventService LwSciEventService object pointer created by
     *                               LwSciEventLoopServiceCreate().
     * @param[out]  newLocalEvent    LwSciLocalEvent object pointer on
     *                               success.
     *
     * @return ::LwSciError, the completion code of operations:
     * - ::LwSciError_Success             Indicates a successful operation.
     * - ::LwSciError_InsufficientMemory  Indicates memory is not sufficient.
     * - ::LwSciError_BadParameter        Indicates an invalid input parameter.
     * - ::LwSciError_IlwalidState        Indicates an invalid operation state.
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
     *   - Runtime: No
     *   - De-Init: No
     * @endif
     */
    /// @cond (SWDOCS_LWSCIIPC_INTERNAL)
    LwSciError (*CreateLocalEvent)(
            LwSciEventService* thisEventService,
            LwSciLocalEvent** newLocalEvent);
    /// @endcond

    /**
     * @if (SWDOCS_LWSCIIPC_NOTSUPPORT)
     * \brief Creates a timer event with an event notifier that reports each
     *        event signaled through it.
     *
     * @note This API is not yet supported.
     *
     * @endif
     */
    /// @cond (SWDOCS_LWSCIIPC_NOTSUPPORT)
    LwSciError (*CreateTimerEvent)(
            LwSciEventService* thisEventService,
            LwSciTimerEvent** newTimerEvent);
    /// @endcond

    /**
     * \brief Releases any resources associated with this event service.
     *
     * Before this member function is called, the caller must ensure that all
     * other member function calls on @a thisEventService have completed and the
     * caller must never again ilwoke any member functions on
     * @a thisEventService.
     *
     * If there any LwSciEventNotifier objects created from this event service that
     * have not been deleted yet, the resources allocated for this event
     * service will not necessarily be released until all those
     * LwSciEventNotifier objects are first deleted.
     *
     * There may also be implementation-specific conditions that result in a
     * delay in the release of resources.
     *
     * Release resources associated with LwSciEventService and LwSciEventService
     * which is created by LwSciEventLoopServiceCreate().
     *
     * @note This API must be called after releasing notifier and LwSciEventService is
     * no longer required.
     *
     * @param[in]  thisEventService LwSciEventService object pointer created by
     *                              LwSciEventLoopServiceCreate().
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
     *   - Runtime: No
     *   - De-Init: Yes
     */
    void (*Delete)(LwSciEventService* thisEventService);
};

/**
 * \struct LwSciEventNotifier
 *
 * \brief An abstract interface to notify event to event consumer and
 * to register event handler of the event consumer client process.
 */
struct LwSciEventNotifier {
    /**
     * @if (SWDOCS_LWSCIIPC_NOTSUPPORT)
     * \brief Registers or unregisters a handler for a particular event notifier.
     *
     * @note This API is not yet supported.
     *
     * In general, handlers for distinct event notifiers may run
     * conlwrrently with one another. The LwSciEventService promises however
     * that no single event notifier will have its handler ilwoked conlwrrently.
     *
     * \param[in] eventNotifier The event notifier that reports each event. Must
     *                          not already be in use by another event loop.
     *
     * \param[in] callback The function to call to handle the event. If NULL,
     *                     handler will be unregistered.
     *
     * \param[in] cookie The parameter to pass to the callback.
     *
     * \param[in] priority The priority of the handler relative to other
     *                     handlers registered with eventLoop. Must be less
     *                     than @ref LW_SCI_EVENT_PRIORITIES.
     * @endif
     */
    /// @cond (SWDOCS_LWSCIIPC_NOTSUPPORT)
    LwSciError (*SetHandler)(LwSciEventNotifier* thisEventNotifier,
            void (*callback)(void* cookie),
            void* cookie,
            uint32_t priority);
    /// @endcond

    /**
     * \brief Unregisters any previously-registered event handler and delete
     * this event notifier.
     *
     * If the event handler's callback is conlwrrently exelwting in another
     * thread, then this function will still return immediately, but the event
     * handler will not be deleted until after the callback returns.
     *
     * This function releases the LwSciEventNotifier and unregisters the event handler.
     * It should be called when the LwSciEventNotifier is no longer required.
     *
     * @param[in]  thisEventNotifier The event handler to unregister and delete.
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
     *   - Runtime: No
     *   - De-Init: Yes
     */
    void (*Delete)(LwSciEventNotifier* thisEventNotifier);
};

/**
 * \brief Creates a new event loop service. The number of event loops that can
 * be created in the new event loop service will be limited to at most
 * @a maxEventLoops.
 *
 * This function creates a new event loop service @a newEventLoopService which is
 * a primary instance of event service. An application must call event service
 * functions along with @a newEventLoopService.
 * The number of event loops that can be created in the new event loop service
 * will be limited to at most @a maxEventLoops.
 *
 * @param[in]   maxEventLoops       The number of event loops, it must be 1.
 * @param[out]  newEventLoopService LwSciEventNotifier object double pointer.
 *
 * @return ::LwSciError, the completion code of operations:
 * - ::LwSciError_Success             Indicates a successful operation.
 * - ::LwSciError_InsufficientMemory  Indicates memory is not sufficient.
 * - ::LwSciError_NotSupported        Indicates a condition is unsupported.
 * - ::LwSciError_IlwalidState        Indicates an invalid operation state.
 * - ::LwSciError_BadParameter        Indicates an invalid or NULL argument.
 * - ::LwSciError_ResourceError       Indicates not enough system resources.
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
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciEventLoopServiceCreate(
        size_t maxEventLoops,
        LwSciEventLoopService** newEventLoopService);

/**
 * \struct LwSciEventLoopService

 * \brief An abstract interface that event consumer can wait for
 * events using event notifier in event loop.
 */
struct LwSciEventLoopService {
    LwSciEventService EventService;

    /**
     * @if (SWDOCS_LWSCIIPC_NOTSUPPORT)
     * \brief Creates an event loop that can handle events for LwSciEventLoopService.
     *
     * @note This API is not yet supported.
     *
     * The user is responsible for running the event loop from a thread by
     * calling the new event loop's Run() function.
     * @endif
     */
    /// @cond (SWDOCS_LWSCIIPC_NOTSUPPORT)
    LwSciError (*CreateEventLoop)(LwSciEventLoopService* eventLoopService,
            LwSciEventLoop** eventLoop);
    /// @endcond

    /**
     * \brief Waits up to a configurable timeout for a particular event
     * notification, servicing events with configured callbacks in the interim.
     *
     * Any asynchronous event notifiers that are pending before calling
     * this function will be claimed by some thread for handling before
     * this function returns.
     *
     * @a eventNotifier must have been created through EventService.
     *
     * @note This function must not be called from an event notifier
     *          callback.
     *
     * This function waits up to a configurable timeout to receive a pulse event
     * which is configured on LwSciQnxEventService_CreateNativeEventNotifier().
     * @a eventNotifier must have been created through EventService before calling.
     *
     * @param[in]  eventNotifier LwSciEventNotifier object pointer.
     * @param[in]  microseconds  A 64-bit integer timeout in microsecond.
     *                           Set to -1 for an infinite timeout, for example,
     *                           the value @ref LW_SCI_EVENT_INFINITE_WAIT.
     *
     * @return ::LwSciError, the completion code of operations:
     * - ::LwSciError_Success            Indicates a successful operation.
     * - ::LwSciError_BadParameter       Indicates an invalid input parameter.
     * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
     * - ::LwSciError_NotSupported       Indicates a condition is unsupported.
     * - ::LwSciError_Timeout            Indicates a timeout oclwrrence.
     * - ::LwSciError_ResourceError      Indicates not enough system resources.
     * - ::LwSciError_InterruptedCall    Indicates an interrupt oclwrred.
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
    LwSciError (*WaitForEvent)(
            LwSciEventNotifier* eventNotifier,
            int64_t microseconds);

    /**
     * \brief Waits up to a configurable timeout for any of a set of
     * particular event notifications, servicing events with configured
     * callbacks in the interim.
     *
     * Any asynchronous event notifiers that are pending before calling
     * this function will be claimed by some thread for handling before
     * this function returns.
     *
     * Each event notifier in @a eventNotifierArray must have been created
     * through EventService.
     *
     * On a successful return, for each integer `i` in the range
     * `[0, eventNotifierCount-1]`, `newEventArray[i]` will be true only if
     * `eventNotifierArray[i]` had a new event.
     *
     * @note This function must not be called from an event notifier
     *          callback.
     * @note This function will be deprecated in furture and user must use
     *          the newer version of the API which is
     *          LwSciEventWaitForMultipleEventsExt
     *
     * @param[in]  eventNotifierArray Array of LwSciEventNotifier object
     *                                pointers.
     * @param[in]  eventNotifierCount Event notifier count in eventNotifierArray.
     * @param[in]  microseconds  A 64-bit integer timeout in microsecond.
     *                           Set to -1 for infinite timeout, for example,
     *                           the value @ref LW_SCI_EVENT_INFINITE_WAIT.
     * @param[out] newEventArray Array of event oclwrrence.
     *
     * @return ::LwSciError, the completion code of operations:
     * - ::LwSciError_Success            Indicates a successful operation.
     * - ::LwSciError_BadParameter       Indicates an invalid input parameter.
     * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
     * - ::LwSciError_NotSupported       Indicates a condition is not supported.
     * - ::LwSciError_Timeout            Indicates a timeout oclwrrence.
     * - ::LwSciError_ResourceError      Indicates not enough system resources.
     * - ::LwSciError_InterruptedCall    Indicates an interrupt oclwrred.
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
    LwSciError (*WaitForMultipleEvents)(
            LwSciEventNotifier* const * eventNotifierArray,
            size_t eventNotifierCount,
            int64_t microseconds,
            bool* newEventArray);


#ifdef LINUX
    /**
     * \brief Waits up to a configurable timeout for any of a set of
     * particular event notifications, servicing events with configured
     * callbacks in the interim.
     *
     * Any asynchronous event notifiers that are pending before calling
     * this function will be claimed by some thread for handling before
     * this function returns.
     *
     * Each event notifier in @a eventNotifierArray must have been created
     * through EventService.
     *
     * On a successful return, for each integer `i` in the range
     * `[0, eventNotifierCount-1]`, `newEventArray[i]` will be true only if
     * `eventNotifierArray[i]` had a new event.
     *
     * @note This function must not be called from an event notifier
     *          callback.
     *
     * @param[in]  eventService Pointer to the event service object
     * @param[in]  eventNotifierArray Array of LwSciEventNotifier object
     *                                pointers.
     * @param[in]  eventNotifierCount Event notifier count in eventNotifierArray.
     * @param[in]  microseconds  A 64-bit integer timeout in microsecond.
     *                           Set to -1 for infinite timeout, for example,
     *                           the value @ref LW_SCI_EVENT_INFINITE_WAIT.
     * @param[out] newEventArray Array of event oclwrrence.
     *
     * @return ::LwSciError, the completion code of operations:
     * - ::LwSciError_Success            Indicates a successful operation.
     * - ::LwSciError_BadParameter       Indicates an invalid input parameter.
     * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
     * - ::LwSciError_NotSupported       Indicates a condition is not supported.
     * - ::LwSciError_Timeout            Indicates a timeout oclwrrence.
     * - ::LwSciError_ResourceError      Indicates not enough system resources.
     * - ::LwSciError_InterruptedCall    Indicates an interrupt oclwrred.
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
    LwSciError (*WaitForMultipleEventsExt)(
            LwSciEventService *eventService,
            LwSciEventNotifier* const * eventNotifierArray,
            size_t eventNotifierCount,
            int64_t microseconds,
            bool* newEventArray);
#endif /* LINUX */
};

/** @} <!-- End lwsci_ipc_event --> */

#ifdef __cplusplus
}
#endif
#endif /* INCLUDED_LWSCIEVENT_H */
