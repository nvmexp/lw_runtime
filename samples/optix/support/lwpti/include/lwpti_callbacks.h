/*
 * Copyright 2010-2020 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__LWPTI_CALLBACKS_H__)
#define __LWPTI_CALLBACKS_H__

#include <lwca.h>
#include <builtin_types.h>
#include <string.h>
#include <lwda_stdint.h>
#include <lwpti_result.h>

#ifndef LWPTIAPI
#ifdef _WIN32
#define LWPTIAPI __stdcall
#else
#define LWPTIAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility push(default)
#endif    

/**
 * \defgroup LWPTI_CALLBACK_API LWPTI Callback API
 * Functions, types, and enums that implement the LWPTI Callback API.
 * @{
 */

/**
 * \brief Specifies the point in an API call that a callback is issued.
 *
 * Specifies the point in an API call that a callback is issued. This
 * value is communicated to the callback function via \ref
 * LWpti_CallbackData::callbackSite.
 */
typedef enum {
  /**
   * The callback is at the entry of the API call.
   */
  LWPTI_API_ENTER                 = 0,
  /**
   * The callback is at the exit of the API call.
   */
  LWPTI_API_EXIT                  = 1,
  LWPTI_API_CBSITE_FORCE_INT     = 0x7fffffff
} LWpti_ApiCallbackSite;

/**
 * \brief Callback domains.
 *
 * Callback domains. Each domain represents callback points for a
 * group of related API functions or LWCA driver activity.
 */
typedef enum {
  /**
   * Invalid domain.
   */
  LWPTI_CB_DOMAIN_ILWALID           = 0,
  /**
   * Domain containing callback points for all driver API functions.
   */
  LWPTI_CB_DOMAIN_DRIVER_API        = 1,
  /**
   * Domain containing callback points for all runtime API
   * functions.
   */
  LWPTI_CB_DOMAIN_RUNTIME_API       = 2,
  /**
   * Domain containing callback points for LWCA resource tracking.
   */
  LWPTI_CB_DOMAIN_RESOURCE          = 3,
  /**
   * Domain containing callback points for LWCA synchronization.
   */
  LWPTI_CB_DOMAIN_SYNCHRONIZE       = 4,
  /**
   * Domain containing callback points for LWTX API functions.
   */
  LWPTI_CB_DOMAIN_LWTX              = 5,
  LWPTI_CB_DOMAIN_SIZE              = 6,
  LWPTI_CB_DOMAIN_FORCE_INT         = 0x7fffffff
} LWpti_CallbackDomain;

/**
 * \brief Callback IDs for resource domain.
 *
 * Callback IDs for resource domain, LWPTI_CB_DOMAIN_RESOURCE.  This
 * value is communicated to the callback function via the \p cbid
 * parameter.
 */
typedef enum {
  /**
   * Invalid resource callback ID.
   */
  LWPTI_CBID_RESOURCE_ILWALID                               = 0,
  /**
   * A new context has been created.
   */
  LWPTI_CBID_RESOURCE_CONTEXT_CREATED                       = 1,
  /**
   * A context is about to be destroyed.
   */
  LWPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING              = 2,
  /**
   * A new stream has been created.
   */
  LWPTI_CBID_RESOURCE_STREAM_CREATED                        = 3,
  /**
   * A stream is about to be destroyed.
   */
  LWPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING               = 4,
  /**
   * The driver has finished initializing.
   */
  LWPTI_CBID_RESOURCE_LW_INIT_FINISHED                      = 5,
  /**
   * A module has been loaded.
   */
  LWPTI_CBID_RESOURCE_MODULE_LOADED                         = 6,
  /**
   * A module is about to be unloaded.
   */
  LWPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING                = 7,
  /**
   * The current module which is being profiled.
   */
  LWPTI_CBID_RESOURCE_MODULE_PROFILED                       = 8,
  /**
   * LWCA graph has been created.
   */
  LWPTI_CBID_RESOURCE_GRAPH_CREATED                         = 9,
  /**
   * LWCA graph is about to be destroyed.
   */
  LWPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING                = 10,
  /**
   * LWCA graph is cloned.
   */
  LWPTI_CBID_RESOURCE_GRAPH_CLONED                          = 11,
  /**
   * LWCA graph node is about to be created
   */
  LWPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING             = 12,
  /**
   * LWCA graph node is created.
   */
  LWPTI_CBID_RESOURCE_GRAPHNODE_CREATED                     = 13,
  /**
   * LWCA graph node is about to be destroyed.
   */
  LWPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING            = 14,
  /**
   * Dependency on a LWCA graph node is created.
   */
  LWPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED          = 15,
  /**
   * Dependency on a LWCA graph node is destroyed.
   */
  LWPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING = 16,
  /**
   * An exelwtable LWCA graph is about to be created.
   */
  LWPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING             = 17,
  /**
   * An exelwtable LWCA graph is created.
   */
  LWPTI_CBID_RESOURCE_GRAPHEXEC_CREATED                     = 18,
  /**
   * An exelwtable LWCA graph is about to be destroyed.
   */
  LWPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING            = 19,
  /**
   * LWCA graph node is cloned.
   */
  LWPTI_CBID_RESOURCE_GRAPHNODE_CLONED                      = 20,

  LWPTI_CBID_RESOURCE_SIZE,
  LWPTI_CBID_RESOURCE_FORCE_INT                   = 0x7fffffff
} LWpti_CallbackIdResource;

/**
 * \brief Callback IDs for synchronization domain.
 *
 * Callback IDs for synchronization domain,
 * LWPTI_CB_DOMAIN_SYNCHRONIZE.  This value is communicated to the
 * callback function via the \p cbid parameter.
 */
typedef enum {
  /**
   * Invalid synchronize callback ID.
   */
  LWPTI_CBID_SYNCHRONIZE_ILWALID                  = 0,
  /**
   * Stream synchronization has completed for the stream.
   */
  LWPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED      = 1,
  /**
   * Context synchronization has completed for the context.
   */
  LWPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED     = 2,
  LWPTI_CBID_SYNCHRONIZE_SIZE,
  LWPTI_CBID_SYNCHRONIZE_FORCE_INT                = 0x7fffffff
} LWpti_CallbackIdSync;

/**
 * \brief Data passed into a runtime or driver API callback function.
 *
 * Data passed into a runtime or driver API callback function as the
 * \p cbdata argument to \ref LWpti_CallbackFunc. The \p cbdata will
 * be this type for \p domain equal to LWPTI_CB_DOMAIN_DRIVER_API or
 * LWPTI_CB_DOMAIN_RUNTIME_API. The callback data is valid only within
 * the invocation of the callback function that is passed the data. If
 * you need to retain some data for use outside of the callback, you
 * must make a copy of that data. For example, if you make a shallow
 * copy of LWpti_CallbackData within a callback, you cannot
 * dereference \p functionParams outside of that callback to access
 * the function parameters. \p functionName is an exception: the
 * string pointed to by \p functionName is a global constant and so
 * may be accessed outside of the callback.
 */
typedef struct {
  /**
   * Point in the runtime or driver function from where the callback
   * was issued.
   */
  LWpti_ApiCallbackSite callbackSite;

  /**
   * Name of the runtime or driver API function which issued the
   * callback. This string is a global constant and so may be
   * accessed outside of the callback.
   */
  const char *functionName;

  /**
   * Pointer to the arguments passed to the runtime or driver API
   * call. See generated_lwda_runtime_api_meta.h and
   * generated_lwda_meta.h for structure definitions for the
   * parameters for each runtime and driver API function.
   */
  const void *functionParams;

  /**
   * Pointer to the return value of the runtime or driver API
   * call. This field is only valid within the exit::LWPTI_API_EXIT
   * callback. For a runtime API \p functionReturlwalue points to a
   * \p lwdaError_t. For a driver API \p functionReturlwalue points
   * to a \p LWresult.
   */
  void *functionReturlwalue;

  /**
   * Name of the symbol operated on by the runtime or driver API
   * function which issued the callback. This entry is valid only for
   * driver and runtime launch callbacks, where it returns the name of
   * the kernel.
   */
  const char *symbolName;

  /**
   * Driver context current to the thread, or null if no context is
   * current. This value can change from the entry to exit callback
   * of a runtime API function if the runtime initializes a context.
   */
  LWcontext context;

  /**
   * Unique ID for the LWCA context associated with the thread. The
   * UIDs are assigned sequentially as contexts are created and are
   * unique within a process.
   */
  uint32_t contextUid;

  /**
   * Pointer to data shared between the entry and exit callbacks of
   * a given runtime or drive API function invocation. This field
   * can be used to pass 64-bit values from the entry callback to
   * the corresponding exit callback.
   */
  uint64_t *correlationData;

  /**
   * The activity record correlation ID for this callback. For a
   * driver domain callback (i.e. \p domain
   * LWPTI_CB_DOMAIN_DRIVER_API) this ID will equal the correlation ID
   * in the LWpti_ActivityAPI record corresponding to the LWCA driver
   * function call. For a runtime domain callback (i.e. \p domain
   * LWPTI_CB_DOMAIN_RUNTIME_API) this ID will equal the correlation
   * ID in the LWpti_ActivityAPI record corresponding to the LWCA
   * runtime function call. Within the callback, this ID can be
   * recorded to correlate user data with the activity record. This
   * field is new in 4.1.
   */
  uint32_t correlationId;

} LWpti_CallbackData;

/**
 * \brief Data passed into a resource callback function.
 *
 * Data passed into a resource callback function as the \p cbdata
 * argument to \ref LWpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to LWPTI_CB_DOMAIN_RESOURCE. The callback
 * data is valid only within the invocation of the callback function
 * that is passed the data. If you need to retain some data for use
 * outside of the callback, you must make a copy of that data.
 */
typedef struct {
  /**
   * For LWPTI_CBID_RESOURCE_CONTEXT_CREATED and
   * LWPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING, the context being
   * created or destroyed. For LWPTI_CBID_RESOURCE_STREAM_CREATED and
   * LWPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING, the context
   * containing the stream being created or destroyed.
   */
  LWcontext context;

  union {
    /**
     * For LWPTI_CBID_RESOURCE_STREAM_CREATED and
     * LWPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING, the stream being
     * created or destroyed.
     */
    LWstream stream;
  } resourceHandle;

  /**
   * Reserved for future use.
   */
  void *resourceDescriptor;
} LWpti_ResourceData;


/**
 * \brief Module data passed into a resource callback function.
 *
 * LWCA module data passed into a resource callback function as the \p cbdata
 * argument to \ref LWpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to LWPTI_CB_DOMAIN_RESOURCE. The module
 * data is valid only within the invocation of the callback function
 * that is passed the data. If you need to retain some data for use
 * outside of the callback, you must make a copy of that data.
 */

typedef struct {
  /**
   * Identifier to associate with the LWCA module.
   */
    uint32_t moduleId;

  /**
   * The size of the lwbin.
   */
    size_t lwbinSize;

  /**
   * Pointer to the associated lwbin.
   */
    const char *pLwbin;
} LWpti_ModuleResourceData;

/**
 * \brief LWCA graphs data passed into a resource callback function.
 *
 * LWCA graphs data passed into a resource callback function as the \p cbdata
 * argument to \ref LWpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to LWPTI_CB_DOMAIN_RESOURCE. The graph
 * data is valid only within the invocation of the callback function
 * that is passed the data. If you need to retain some data for use
 * outside of the callback, you must make a copy of that data.
 */

typedef struct {
  /**
   * LWCA graph
   */
    LWgraph graph;
  /**
   * The original LWCA graph from which \param graph is cloned
   */
    LWgraph originalGraph;
  /**
   * LWCA graph node
   */
    LWgraphNode node;
  /**
   * The original LWCA graph node from which \param node is cloned
   */
    LWgraphNode originalNode;
  /**
   * Type of the \param node
   */
    LWgraphNodeType nodeType;
  /**
   * The dependent graph node
   * The size of the array is \param numDependencies.
   */
    LWgraphNode dependency;
  /**
   * LWCA exelwtable graph
   */
    LWgraphExec graphExec;
} LWpti_GraphData;

/**
 * \brief Data passed into a synchronize callback function.
 *
 * Data passed into a synchronize callback function as the \p cbdata
 * argument to \ref LWpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to LWPTI_CB_DOMAIN_SYNCHRONIZE. The
 * callback data is valid only within the invocation of the callback
 * function that is passed the data. If you need to retain some data
 * for use outside of the callback, you must make a copy of that data.
 */
typedef struct {
  /**
   * The context of the stream being synchronized.
   */
  LWcontext context;
  /**
   * The stream being synchronized.
   */
  LWstream  stream;
} LWpti_SynchronizeData;

/**
 * \brief Data passed into a LWTX callback function.
 *
 * Data passed into a LWTX callback function as the \p cbdata argument
 * to \ref LWpti_CallbackFunc. The \p cbdata will be this type for \p
 * domain equal to LWPTI_CB_DOMAIN_LWTX. Unless otherwise notes, the
 * callback data is valid only within the invocation of the callback
 * function that is passed the data. If you need to retain some data
 * for use outside of the callback, you must make a copy of that data.
 */
typedef struct {
  /**
   * Name of the LWTX API function which issued the callback. This
   * string is a global constant and so may be accessed outside of the
   * callback.
   */
  const char *functionName;

  /**
   * Pointer to the arguments passed to the LWTX API call. See
   * generated_lwtx_meta.h for structure definitions for the
   * parameters for each LWTX API function.
   */
  const void *functionParams;
} LWpti_LwtxData;

/**
 * \brief An ID for a driver API, runtime API, resource or
 * synchronization callback.
 *
 * An ID for a driver API, runtime API, resource or synchronization
 * callback. Within a driver API callback this should be interpreted
 * as a LWpti_driver_api_trace_cbid value (these values are defined in
 * lwpti_driver_cbid.h). Within a runtime API callback this should be
 * interpreted as a LWpti_runtime_api_trace_cbid value (these values
 * are defined in lwpti_runtime_cbid.h). Within a resource API
 * callback this should be interpreted as a \ref
 * LWpti_CallbackIdResource value. Within a synchronize API callback
 * this should be interpreted as a \ref LWpti_CallbackIdSync value.
 */
typedef uint32_t LWpti_CallbackId;

/**
 * \brief Function type for a callback.
 *
 * Function type for a callback. The type of the data passed to the
 * callback in \p cbdata depends on the \p domain. If \p domain is
 * LWPTI_CB_DOMAIN_DRIVER_API or LWPTI_CB_DOMAIN_RUNTIME_API the type
 * of \p cbdata will be LWpti_CallbackData. If \p domain is
 * LWPTI_CB_DOMAIN_RESOURCE the type of \p cbdata will be
 * LWpti_ResourceData. If \p domain is LWPTI_CB_DOMAIN_SYNCHRONIZE the
 * type of \p cbdata will be LWpti_SynchronizeData. If \p domain is
 * LWPTI_CB_DOMAIN_LWTX the type of \p cbdata will be LWpti_LwtxData.
 *
 * \param userdata User data supplied at subscription of the callback
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 * \param cbdata Data passed to the callback.
 */
typedef void (LWPTIAPI *LWpti_CallbackFunc)(
    void *userdata,
    LWpti_CallbackDomain domain,
    LWpti_CallbackId cbid,
    const void *cbdata);

/**
 * \brief A callback subscriber.
 */
typedef struct LWpti_Subscriber_st *LWpti_SubscriberHandle;

/**
 * \brief Pointer to an array of callback domains.
 */
typedef LWpti_CallbackDomain *LWpti_DomainTable;

/**
 * \brief Get the available callback domains.
 *
 * Returns in \p *domainTable an array of size \p *domainCount of all
 * the available callback domains.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param domainCount Returns number of callback domains
 * \param domainTable Returns pointer to array of available callback domains
 *
 * \retval LWPTI_SUCCESS on success
 * \retval LWPTI_ERROR_NOT_INITIALIZED if unable to initialize LWPTI
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p domainCount or \p domainTable are NULL
 */
LWptiResult LWPTIAPI lwptiSupportedDomains(size_t *domainCount,
                                           LWpti_DomainTable *domainTable);

/**
 * \brief Initialize a callback subscriber with a callback function
 * and user data.
 *
 * Initializes a callback subscriber with a callback function and
 * (optionally) a pointer to user data. The returned subscriber handle
 * can be used to enable and disable the callback for specific domains
 * and callback IDs.
 * \note Only a single subscriber can be registered at a time. To ensure
 * that no other LWPTI client interrupts the profiling session, it's the
 * responsibility of all the LWPTI clients to call this function before
 * starting the profling session. In case profiling session is already
 * started by another LWPTI client, this function returns the error code
 * LWPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED.
 * Note that this function returns the same error when application is
 * launched using LWPU tools like lwperf, Visual Profiler, Nsight Systems,
 * Nsight Compute, lwca-gdb and lwca-memcheck.
 * \note This function does not enable any callbacks.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param subscriber Returns handle to initialize subscriber
 * \param callback The callback function
 * \param userdata A pointer to user data. This data will be passed to
 * the callback function via the \p userdata paramater.
 *
 * \retval LWPTI_SUCCESS on success
 * \retval LWPTI_ERROR_NOT_INITIALIZED if unable to initialize LWPTI
 * \retval LWPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED if there is already a LWPTI subscriber
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p subscriber is NULL
 */
LWptiResult LWPTIAPI lwptiSubscribe(LWpti_SubscriberHandle *subscriber,
                                    LWpti_CallbackFunc callback,
                                    void *userdata);

/**
 * \brief Unregister a callback subscriber.
 *
 * Removes a callback subscriber so that no future callbacks will be
 * issued to that subscriber.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param subscriber Handle to the initialize subscriber
 *
 * \retval LWPTI_SUCCESS on success
 * \retval LWPTI_ERROR_NOT_INITIALIZED if unable to initialized LWPTI
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p subscriber is NULL or not initialized
 */
LWptiResult LWPTIAPI lwptiUnsubscribe(LWpti_SubscriberHandle subscriber);

/**
 * \brief Get the current enabled/disabled state of a callback for a specific
 * domain and function ID.
 *
 * Returns non-zero in \p *enable if the callback for a domain and
 * callback ID is enabled, and zero if not enabled.
 *
 * \note \b Thread-safety: a subscriber must serialize access to
 * lwptiGetCallbackState, lwptiEnableCallback, lwptiEnableDomain, and
 * lwptiEnableAllDomains. For example, if lwptiGetCallbackState(sub,
 * d, c) and lwptiEnableCallback(sub, d, c) are called conlwrrently,
 * the results are undefined.
 *
 * \param enable Returns non-zero if callback enabled, zero if not enabled
 * \param subscriber Handle to the initialize subscriber
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 *
 * \retval LWPTI_SUCCESS on success
 * \retval LWPTI_ERROR_NOT_INITIALIZED if unable to initialized LWPTI
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p enabled is NULL, or if \p
 * subscriber, \p domain or \p cbid is invalid.
 */
LWptiResult LWPTIAPI lwptiGetCallbackState(uint32_t *enable,
                                           LWpti_SubscriberHandle subscriber,
                                           LWpti_CallbackDomain domain,
                                           LWpti_CallbackId cbid);

/**
 * \brief Enable or disabled callbacks for a specific domain and
 * callback ID.
 *
 * Enable or disabled callbacks for a subscriber for a specific domain
 * and callback ID.
 *
 * \note \b Thread-safety: a subscriber must serialize access to
 * lwptiGetCallbackState, lwptiEnableCallback, lwptiEnableDomain, and
 * lwptiEnableAllDomains. For example, if lwptiGetCallbackState(sub,
 * d, c) and lwptiEnableCallback(sub, d, c) are called conlwrrently,
 * the results are undefined.
 *
 * \param enable New enable state for the callback. Zero disables the
 * callback, non-zero enables the callback.
 * \param subscriber - Handle to callback subscription
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 *
 * \retval LWPTI_SUCCESS on success
 * \retval LWPTI_ERROR_NOT_INITIALIZED if unable to initialized LWPTI
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p subscriber, \p domain or \p
 * cbid is invalid.
 */
LWptiResult LWPTIAPI lwptiEnableCallback(uint32_t enable,
                                         LWpti_SubscriberHandle subscriber,
                                         LWpti_CallbackDomain domain,
                                         LWpti_CallbackId cbid);

/**
 * \brief Enable or disabled all callbacks for a specific domain.
 *
 * Enable or disabled all callbacks for a specific domain.
 *
 * \note \b Thread-safety: a subscriber must serialize access to
 * lwptiGetCallbackState, lwptiEnableCallback, lwptiEnableDomain, and
 * lwptiEnableAllDomains. For example, if lwptiGetCallbackEnabled(sub,
 * d, *) and lwptiEnableDomain(sub, d) are called conlwrrently, the
 * results are undefined.
 *
 * \param enable New enable state for all callbacks in the
 * domain. Zero disables all callbacks, non-zero enables all
 * callbacks.
 * \param subscriber - Handle to callback subscription
 * \param domain The domain of the callback
 *
 * \retval LWPTI_SUCCESS on success
 * \retval LWPTI_ERROR_NOT_INITIALIZED if unable to initialized LWPTI
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p subscriber or \p domain is invalid
 */
LWptiResult LWPTIAPI lwptiEnableDomain(uint32_t enable,
                                       LWpti_SubscriberHandle subscriber,
                                       LWpti_CallbackDomain domain);

/**
 * \brief Enable or disable all callbacks in all domains.
 *
 * Enable or disable all callbacks in all domains.
 *
 * \note \b Thread-safety: a subscriber must serialize access to
 * lwptiGetCallbackState, lwptiEnableCallback, lwptiEnableDomain, and
 * lwptiEnableAllDomains. For example, if lwptiGetCallbackState(sub,
 * d, *) and lwptiEnableAllDomains(sub) are called conlwrrently, the
 * results are undefined.
 *
 * \param enable New enable state for all callbacks in all
 * domain. Zero disables all callbacks, non-zero enables all
 * callbacks.
 * \param subscriber - Handle to callback subscription
 *
 * \retval LWPTI_SUCCESS on success
 * \retval LWPTI_ERROR_NOT_INITIALIZED if unable to initialized LWPTI
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p subscriber is invalid
 */
LWptiResult LWPTIAPI lwptiEnableAllDomains(uint32_t enable,
                                           LWpti_SubscriberHandle subscriber);

/**
 * \brief Get the name of a callback for a specific domain and callback ID.
 *
 * Returns a pointer to the name c_string in \p **name.
 *
 * \note \b Names are available only for the DRIVER and RUNTIME domains.
 *
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 * \param name Returns pointer to the name string on success, NULL otherwise
 *
 * \retval LWPTI_SUCCESS on success
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p name is NULL, or if
 * \p domain or \p cbid is invalid.
 */
LWptiResult LWPTIAPI lwptiGetCallbackName(LWpti_CallbackDomain domain,
                                          uint32_t cbid,
                                          const char **name);

/** @} */ /* END LWPTI_CALLBACK_API */

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif  // file guard

