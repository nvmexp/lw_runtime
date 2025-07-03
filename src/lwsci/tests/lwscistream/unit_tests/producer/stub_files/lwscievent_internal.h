/*
 * Copyright (c) 2020-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIEVENT_INTERNAL_H
#define INCLUDED_LWSCIEVENT_INTERNAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <lwscievent.h>

/**
 * @defgroup lwsci_ipc_event_internal Event Service Internal Declarations
 *
 * @ingroup lwsci_ipc_event
 * @{
 */

/**
 * @if (SWDOCS_LWSCIIPC_INTERNAL)
 * \struct LwSciNativeEvent
 * \brief An OS-environment-specific object that describes how a thread can
 *        wait for events.
 *
 * @note This is for internal use only.
 * @endif
 */
/// @cond (SWDOCS_LWSCIIPC_INTERNAL)
struct LwSciNativeEvent {
#ifdef LINUX
    /** \brief A file descriptor to wait on using epoll. */
    int fd;
    /** \brief The epoll events to pass to epoll_ctl(). */
    uint32_t epollEvents;
#endif /* LINUX */

#ifdef __QNX__
    /**
     * \brief Configures the pulse to be generated when this native event
     *         oclwrs.
     *
     * Must only be called while this native event is in a disabled state. On a
     * successful return, this native event transitions to being enabled.
     */
    LwSciError (*ConfigurePulseParams)(
        LwSciNativeEvent* thisNativeEvent,
        int32_t coid, int16_t priority,
        int16_t code, void* value);

    /**
     * \brief Undo a previous ConfigurePulseParams() call.
     *
     * Must only be called while this native event is in an enabled state. This
     * native event transitions to being disabled.
     */
    void (*UnconfigurePulseParams)(
        LwSciNativeEvent* thisNativeEvent);

    /** \brief Unmask Inter-VM backend interrupt.
     *
     * Must only be called while this native event is in an enabled state.
     * The interrupt associated to this native event to be unmasked.
     */
    LwSciError (*UnmaskInterrupt)(
        LwSciNativeEvent* thisNativeEvent);
#endif /* __QNX__ */
};
/// @endcond
/** @} <!-- End lwsci_ipc_event_internal --> */

/**
 * @addtogroup lwsci_ipc_event_internal
 * @{
 */

/**
 * @if (SWDOCS_LWSCIIPC_INTERNAL)
 * \struct LwSciLocalEvent
 * \brief An OS-agnostic object that sends signal to another thread
 *        in the same process.
 * @note This is for internal use only.
 * @endif
 */
/// @cond (SWDOCS_LWSCIIPC_INTERNAL)
struct LwSciLocalEvent {
    /** \brief Event notifier associated with this local event. */
    LwSciEventNotifier* eventNotifier;

    /**
     * \brief Sends an intra-process local event signal.
     *
     * @note This is for internal use only.
     *
     * Any thread which is blocked by local event notifier associated with local event
     * will be unblocked by this signal.
     *
     * @param[in]  thisLocalEvent LwSciLocalEvent object pointer created by
     *                            LwSciEventService::CreateLocalEvent()
     *
     * @return ::LwSciError, the completion code of operations:
     * - ::LwSciError_Success         Indicates a successful operation.
     * - ::LwSciError_BadParameter    Indicates an invalid input parameter.
     * - ::LwSciError_TryItAgain       Indicates an kernel pulse queue shortage.
     * - ::LwSciError_IlwalidState      Indicates an invalid operation state.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes
     *   - Async/Sync: Sync
     * - Required Privileges: None
     * - API Group
     *   - Init: No
     *   - Runtime: Yes
     *   - De-Init: No
     */
    LwSciError (*Signal)(LwSciLocalEvent* thisLocalEvent);

    /**
     * \brief Releases any resources associated with this local event.
     *
     * @note This is for internal use only.
     *
     * This function must be called after releasing notifier and when LwSciLocalEvent is
     * no longer required.
     *
     * @param[in]  thisLocalEvent LwSciLocalEvent object pointer created by
     *                            LwSciEventService::CreateLocalEvent().
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes
     *   - Async/Sync: Sync
     * - Required Privileges: None
     * - API Group
     *   - Init: No
     *   - Runtime: No
     *   - De-Init: Yes
     */
    void (*Delete)(LwSciLocalEvent* thisLocalEvent);
};
/// @endcond
/** @} <!-- End lwsci_ipc_event_internal --> */

/**
 * @if (SWDOCS_LWSCIIPC_NOTSUPPORT)
 * \struct LwSciTimerEvent
 *
 * @note This is for internal use only.
 * @endif
 */
/// @cond (SWDOCS_LWSCIIPC_NOTSUPPORT)
struct LwSciTimerEvent {
    LwSciEventNotifier* eventNotifier;
    LwSciError (*SetTimer)(
            LwSciTimerEvent* thisTimerEvent,
            int64_t microSeconds);
    LwSciError (*ClearTimer)(
            LwSciTimerEvent* thisTimerEvent);
    void (*Delete)(
            LwSciTimerEvent* thisTimerEvent);
};
/// @endcond

/**
 * @if (SWDOCS_LWSCIIPC_NOTSUPPORT)
 * \struct LwSciEventLoop
 *
 * @note This is for internal use only.
 * @endif
 */
/// @cond (SWDOCS_LWSCIIPC_NOTSUPPORT)
struct LwSciEventLoop {
    /** \brief Use the calling thread to run the specified event loop. Will fail
     * any attempt to run the same event loop simultaneously from multiple
     * threads.
     *
     * @note This is for internal use only.
     */
    LwSciError (*Run)(LwSciEventLoop* eventLoop);
    /** \brief Stop the specified event loop. If the event loop is running, the
     * corresponding Run() call will return as soon as it is done handling any
     * outstanding events.
     *
     * @note This is for internal use only.
     */
    LwSciError (*Stop)(LwSciEventLoop* eventLoop);
    /** \brief Delete any resources associated with the specified event loop,
     *         stopping it first if it is running.
     *
     * @note This is for internal use only.
     *
     * Run() takes a reference to the event loop, and does not relinquish
     * that reference until it returns. This means for example that if Delete()
     * is called from a handler within Run(), the event loop will not actually
     * be deleted until Run() returns.
     */
    void (*Delete)(LwSciEventLoop* eventLoop);
};
/// @endcond

#ifdef __cplusplus
}
#endif
#endif /* INCLUDED_LWSCIEVENT_INTERNAL_H */
