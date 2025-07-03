/*
 * Copyright 2011-2020 LWPU Corporation.  All rights reserved.
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

#if !defined(_LWPTI_ACTIVITY_H_)
#define _LWPTI_ACTIVITY_H_

#include <lwca.h>
#include <lwpti_callbacks.h>
#include <lwpti_events.h>
#include <lwpti_metrics.h>
#include <lwpti_result.h>
#if defined(LWPTI_DIRECTIVE_SUPPORT)
#include <Openacc/lwpti_openacc.h>
#include <Openmp/lwpti_openmp.h>
#endif

#ifndef LWPTIAPI
#ifdef _WIN32
#define LWPTIAPI __stdcall
#else
#define LWPTIAPI
#endif
#endif

#if defined(__LP64__)
#define LWPTILP64 1
#elif defined(_WIN64)
#define LWPTILP64 1
#else
#undef LWPTILP64
#endif

#define ACTIVITY_RECORD_ALIGNMENT 8
#if defined(_WIN32) // Windows 32- and 64-bit
#define START_PACKED_ALIGNMENT __pragma(pack(push,1)) // exact fit - no padding
#define PACKED_ALIGNMENT __declspec(align(ACTIVITY_RECORD_ALIGNMENT))
#define END_PACKED_ALIGNMENT __pragma(pack(pop))
#elif defined(__GNUC__) // GCC
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT __attribute__ ((__packed__)) __attribute__ ((aligned (ACTIVITY_RECORD_ALIGNMENT)))
#define END_PACKED_ALIGNMENT
#else // all other compilers
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT
#define END_PACKED_ALIGNMENT
#endif

#define LWPTI_UNIFIED_MEMORY_CPU_DEVICE_ID ((uint32_t) 0xFFFFFFFFU)
#define LWPTI_ILWALID_CONTEXT_ID ((uint32_t) 0xFFFFFFFFU)
#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup LWPTI_ACTIVITY_API LWPTI Activity API
 * Functions, types, and enums that implement the LWPTI Activity API.
 * @{
 */

/**
 * \brief The kinds of activity records.
 *
 * Each activity record kind represents information about a GPU or an
 * activity oclwrring on a CPU or GPU. Each kind is associated with a
 * activity record structure that holds the information associated
 * with the kind.
 * \see LWpti_Activity
 * \see LWpti_ActivityAPI
 * \see LWpti_ActivityContext
 * \see LWpti_ActivityDevice
 * \see LWpti_ActivityDevice2
 * \see LWpti_ActivityDeviceAttribute
 * \see LWpti_ActivityEvent
 * \see LWpti_ActivityEventInstance
 * \see LWpti_ActivityKernel
 * \see LWpti_ActivityKernel2
 * \see LWpti_ActivityKernel3
 * \see LWpti_ActivityKernel4
 * \see LWpti_ActivityKernel5
 * \see LWpti_ActivityCdpKernel
 * \see LWpti_ActivityPreemption
 * \see LWpti_ActivityMemcpy
 * \see LWpti_ActivityMemcpy3
 * \see LWpti_ActivityMemcpyPtoP
 * \see LWpti_ActivityMemcpyPtoP2
 * \see LWpti_ActivityMemset
 * \see LWpti_ActivityMemset2
 * \see LWpti_ActivityMetric
 * \see LWpti_ActivityMetricInstance
 * \see LWpti_ActivityName
 * \see LWpti_ActivityMarker
 * \see LWpti_ActivityMarker2
 * \see LWpti_ActivityMarkerData
 * \see LWpti_ActivitySourceLocator
 * \see LWpti_ActivityGlobalAccess
 * \see LWpti_ActivityGlobalAccess2
 * \see LWpti_ActivityGlobalAccess3
 * \see LWpti_ActivityBranch
 * \see LWpti_ActivityBranch2
 * \see LWpti_ActivityOverhead
 * \see LWpti_ActivityElwironment
 * \see LWpti_ActivityInstructionExelwtion
 * \see LWpti_ActivityUnifiedMemoryCounter
 * \see LWpti_ActivityFunction
 * \see LWpti_ActivityModule
 * \see LWpti_ActivitySharedAccess
 * \see LWpti_ActivityPCSampling
 * \see LWpti_ActivityPCSampling2
 * \see LWpti_ActivityPCSampling3
 * \see LWpti_ActivityPCSamplingRecordInfo
 * \see LWpti_ActivityLwdaEvent
 * \see LWpti_ActivityStream
 * \see LWpti_ActivitySynchronization
 * \see LWpti_ActivityInstructionCorrelation
 * \see LWpti_ActivityExternalCorrelation
 * \see LWpti_ActivityUnifiedMemoryCounter2
 * \see LWpti_ActivityOpenAccData
 * \see LWpti_ActivityOpenAccLaunch
 * \see LWpti_ActivityOpenAccOther
 * \see LWpti_ActivityOpenMp
 * \see LWpti_ActivityLwLink
 * \see LWpti_ActivityLwLink2
 * \see LWpti_ActivityLwLink3
 * \see LWpti_ActivityMemory
 * \see LWpti_ActivityPcie
 */
typedef enum {
  /**
   * The activity record is invalid.
   */
  LWPTI_ACTIVITY_KIND_ILWALID  = 0,
  /**
   * A host<->host, host<->device, or device<->device memory copy. The
   * corresponding activity record structure is \ref
   * LWpti_ActivityMemcpy3.
   */
  LWPTI_ACTIVITY_KIND_MEMCPY   = 1,
  /**
   * A memory set exelwting on the GPU. The corresponding activity
   * record structure is \ref LWpti_ActivityMemset2.
   */
  LWPTI_ACTIVITY_KIND_MEMSET   = 2,
  /**
   * A kernel exelwting on the GPU. This activity kind may significantly change
   * the overall performance characteristics of the application because all
   * kernel exelwtions are serialized on the GPU. Other activity kind for kernel
   * LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL doesn't break kernel conlwrrency.
   * The corresponding activity record structure is \ref LWpti_ActivityKernel5.
   */
  LWPTI_ACTIVITY_KIND_KERNEL   = 3,
  /**
   * A LWCA driver API function exelwtion. The corresponding activity
   * record structure is \ref LWpti_ActivityAPI.
   */
  LWPTI_ACTIVITY_KIND_DRIVER   = 4,
  /**
   * A LWCA runtime API function exelwtion. The corresponding activity
   * record structure is \ref LWpti_ActivityAPI.
   */
  LWPTI_ACTIVITY_KIND_RUNTIME  = 5,
  /**
   * An event value. The corresponding activity record structure is
   * \ref LWpti_ActivityEvent.
   */
  LWPTI_ACTIVITY_KIND_EVENT    = 6,
  /**
   * A metric value. The corresponding activity record structure is
   * \ref LWpti_ActivityMetric.
   */
  LWPTI_ACTIVITY_KIND_METRIC   = 7,
  /**
   * Information about a device. The corresponding activity record
   * structure is \ref LWpti_ActivityDevice2.
   */
  LWPTI_ACTIVITY_KIND_DEVICE   = 8,
  /**
   * Information about a context. The corresponding activity record
   * structure is \ref LWpti_ActivityContext.
   */
  LWPTI_ACTIVITY_KIND_CONTEXT  = 9,
  /**
   * A kernel exelwting on the GPU. This activity kind doesn't break
   * kernel conlwrrency. The corresponding activity record structure
   * is \ref LWpti_ActivityKernel5.
   */
  LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL = 10,
  /**
   * Thread, device, context, etc. name. The corresponding activity
   * record structure is \ref LWpti_ActivityName.
   */
  LWPTI_ACTIVITY_KIND_NAME     = 11,
  /**
   * Instantaneous, start, or end marker. The corresponding activity
   * record structure is \ref LWpti_ActivityMarker2.
   */
  LWPTI_ACTIVITY_KIND_MARKER = 12,
  /**
   * Extended, optional, data about a marker. The corresponding
   * activity record structure is \ref LWpti_ActivityMarkerData.
   */
  LWPTI_ACTIVITY_KIND_MARKER_DATA = 13,
  /**
   * Source information about source level result. The corresponding
   * activity record structure is \ref LWpti_ActivitySourceLocator.
   */
  LWPTI_ACTIVITY_KIND_SOURCE_LOCATOR = 14,
  /**
   * Results for source-level global acccess. The
   * corresponding activity record structure is \ref
   * LWpti_ActivityGlobalAccess3.
   */
  LWPTI_ACTIVITY_KIND_GLOBAL_ACCESS = 15,
  /**
   * Results for source-level branch. The corresponding
   * activity record structure is \ref LWpti_ActivityBranch2.
   */
  LWPTI_ACTIVITY_KIND_BRANCH = 16,
  /**
   * Overhead activity records. The
   * corresponding activity record structure is
   * \ref LWpti_ActivityOverhead.
   */
  LWPTI_ACTIVITY_KIND_OVERHEAD = 17,
  /**
   * A CDP (LWCA Dynamic Parallel) kernel exelwting on the GPU. The
   * corresponding activity record structure is \ref
   * LWpti_ActivityCdpKernel.  This activity can not be directly
   * enabled or disabled. It is enabled and disabled through
   * conlwrrent kernel activity i.e. _CONLWRRENT_KERNEL.
   */
  LWPTI_ACTIVITY_KIND_CDP_KERNEL = 18,
  /**
   * Preemption activity record indicating a preemption of a CDP (LWCA
   * Dynamic Parallel) kernel exelwting on the GPU. The corresponding
   * activity record structure is \ref LWpti_ActivityPreemption.
   */
  LWPTI_ACTIVITY_KIND_PREEMPTION = 19,
  /**
   * Environment activity records indicating power, clock, thermal,
   * etc. levels of the GPU. The corresponding activity record
   * structure is \ref LWpti_ActivityElwironment.
   */
  LWPTI_ACTIVITY_KIND_ELWIRONMENT = 20,
  /**
   * An event value associated with a specific event domain
   * instance. The corresponding activity record structure is \ref
   * LWpti_ActivityEventInstance.
   */
  LWPTI_ACTIVITY_KIND_EVENT_INSTANCE = 21,
  /**
   * A peer to peer memory copy. The corresponding activity record
   * structure is \ref LWpti_ActivityMemcpyPtoP2.
   */
  LWPTI_ACTIVITY_KIND_MEMCPY2 = 22,
  /**
   * A metric value associated with a specific metric domain
   * instance. The corresponding activity record structure is \ref
   * LWpti_ActivityMetricInstance.
   */
  LWPTI_ACTIVITY_KIND_METRIC_INSTANCE = 23,
  /**
   * Results for source-level instruction exelwtion.
   * The corresponding activity record structure is \ref
   * LWpti_ActivityInstructionExelwtion.
   */
  LWPTI_ACTIVITY_KIND_INSTRUCTION_EXELWTION = 24,
  /**
   * Unified Memory counter record. The corresponding activity
   * record structure is \ref LWpti_ActivityUnifiedMemoryCounter2.
   */
  LWPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER = 25,
  /**
   * Device global/function record. The corresponding activity
   * record structure is \ref LWpti_ActivityFunction.
   */
  LWPTI_ACTIVITY_KIND_FUNCTION = 26,
  /**
   * LWCA Module record. The corresponding activity
   * record structure is \ref LWpti_ActivityModule.
   */
  LWPTI_ACTIVITY_KIND_MODULE = 27,
  /**
   * A device attribute value. The corresponding activity record
   * structure is \ref LWpti_ActivityDeviceAttribute.
   */
  LWPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE   = 28,
  /**
   * Results for source-level shared acccess. The
   * corresponding activity record structure is \ref
   * LWpti_ActivitySharedAccess.
   */
  LWPTI_ACTIVITY_KIND_SHARED_ACCESS = 29,
  /**
   * Enable PC sampling for kernels. This will serialize
   * kernels. The corresponding activity record structure
   * is \ref LWpti_ActivityPCSampling3.
   */
  LWPTI_ACTIVITY_KIND_PC_SAMPLING = 30,
  /**
   * Summary information about PC sampling records. The
   * corresponding activity record structure is \ref
   * LWpti_ActivityPCSamplingRecordInfo.
   */
  LWPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO = 31,
  /**
   * SASS/Source line-by-line correlation record.
   * This will generate sass/source correlation for functions that have source
   * level analysis or pc sampling results. The records will be generated only
   * when either of source level analysis or pc sampling activity is enabled.
   * The corresponding activity record structure is \ref
   * LWpti_ActivityInstructionCorrelation.
   */
  LWPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION = 32,
  /**
   * OpenACC data events.
   * The corresponding activity record structure is \ref
   * LWpti_ActivityOpenAccData.
   */
  LWPTI_ACTIVITY_KIND_OPENACC_DATA = 33,
  /**
   * OpenACC launch events.
   * The corresponding activity record structure is \ref
   * LWpti_ActivityOpenAccLaunch.
   */
  LWPTI_ACTIVITY_KIND_OPENACC_LAUNCH = 34,
  /**
   * OpenACC other events.
   * The corresponding activity record structure is \ref
   * LWpti_ActivityOpenAccOther.
   */
  LWPTI_ACTIVITY_KIND_OPENACC_OTHER = 35,
  /**
   * Information about a LWCA event. The
   * corresponding activity record structure is \ref
   * LWpti_ActivityLwdaEvent.
   */
  LWPTI_ACTIVITY_KIND_LWDA_EVENT = 36,
  /**
   * Information about a LWCA stream. The
   * corresponding activity record structure is \ref
   * LWpti_ActivityStream.
   */
  LWPTI_ACTIVITY_KIND_STREAM = 37,
  /**
   * Records for synchronization management. The
   * corresponding activity record structure is \ref
   * LWpti_ActivitySynchronization.
   */
  LWPTI_ACTIVITY_KIND_SYNCHRONIZATION = 38,
  /**
   * Records for correlation of different programming APIs. The
   * corresponding activity record structure is \ref
   * LWpti_ActivityExternalCorrelation.
   */
  LWPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 39,
  /**
   * LWLink information.
   * The corresponding activity record structure is \ref
   * LWpti_ActivityLwLink3.
   */
  LWPTI_ACTIVITY_KIND_LWLINK = 40,
  /**
   * Instantaneous Event information.
   * The corresponding activity record structure is \ref
   * LWpti_ActivityInstantaneousEvent.
   */
  LWPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT = 41,
  /**
   * Instantaneous Event information for a specific event 
   * domain instance.
   * The corresponding activity record structure is \ref
   * LWpti_ActivityInstantaneousEventInstance
   */
  LWPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE = 42,
  /**
   * Instantaneous Metric information
   * The corresponding activity record structure is \ref
   * LWpti_ActivityInstantaneousMetric.
   */
  LWPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC = 43,
  /**
   * Instantaneous Metric information for a specific metric
   * domain instance.
   * The corresponding activity record structure is \ref
   * LWpti_ActivityInstantaneousMetricInstance.
   */
  LWPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE = 44,
  /**
   * Memory activity tracking allocation and freeing of the memory
   * The corresponding activity record structure is \ref
   * LWpti_ActivityMemory.
   */
  LWPTI_ACTIVITY_KIND_MEMORY = 45,
  /**
   * PCI devices information used for PCI topology.
   * The corresponding activity record structure is \ref
   * LWpti_ActivityPcie.
   */
  LWPTI_ACTIVITY_KIND_PCIE = 46,
  /**
   * OpenMP parallel events.
   * The corresponding activity record structure is \ref
   * LWpti_ActivityOpenMp.
   */
  LWPTI_ACTIVITY_KIND_OPENMP = 47,
  /**
   * A LWCA driver kernel launch oclwrring outside of any
   * public API function exelwtion.  Tools can handle these
   * like records for driver API launch functions, although
   * the cbid field is not used here.
   * The corresponding activity record structure is \ref
   * LWpti_ActivityAPI.
   */
  LWPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API = 48,

  LWPTI_ACTIVITY_KIND_COUNT = 49,
  
  LWPTI_ACTIVITY_KIND_FORCE_INT     = 0x7fffffff
} LWpti_ActivityKind;

/**
 * \brief The kinds of activity objects.
 * \see LWpti_ActivityObjectKindId
 */
typedef enum {
  /**
   * The object kind is not known.
   */
  LWPTI_ACTIVITY_OBJECT_UNKNOWN  = 0,
  /**
   * A process.
   */
  LWPTI_ACTIVITY_OBJECT_PROCESS  = 1,
  /**
   * A thread.
   */
  LWPTI_ACTIVITY_OBJECT_THREAD   = 2,
  /**
   * A device.
   */
  LWPTI_ACTIVITY_OBJECT_DEVICE   = 3,
  /**
   * A context.
   */
  LWPTI_ACTIVITY_OBJECT_CONTEXT  = 4,
  /**
   * A stream.
   */
  LWPTI_ACTIVITY_OBJECT_STREAM   = 5,

  LWPTI_ACTIVITY_OBJECT_FORCE_INT = 0x7fffffff
} LWpti_ActivityObjectKind;

/**
 * \brief Identifiers for object kinds as specified by
 * LWpti_ActivityObjectKind.
 * \see LWpti_ActivityObjectKind
 */
typedef union {
  /**
   * A process object requires that we identify the process ID. A
   * thread object requires that we identify both the process and
   * thread ID.
   */
  struct {
    uint32_t processId;
    uint32_t threadId;
  } pt;
  /**
   * A device object requires that we identify the device ID. A
   * context object requires that we identify both the device and
   * context ID. A stream object requires that we identify device,
   * context, and stream ID.
   */
  struct {
    uint32_t deviceId;
    uint32_t contextId;
    uint32_t streamId;
  } dcs;
} LWpti_ActivityObjectKindId;

/**
 * \brief The kinds of activity overhead.
 */
typedef enum {
  /**
   * The overhead kind is not known.
   */
  LWPTI_ACTIVITY_OVERHEAD_UNKNOWN               = 0,
  /**
   * Compiler(JIT) overhead.
   */
  LWPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER       = 1,
  /**
   * Activity buffer flush overhead.
   */
  LWPTI_ACTIVITY_OVERHEAD_LWPTI_BUFFER_FLUSH    = 1<<16,
  /**
   * LWPTI instrumentation overhead.
   */
  LWPTI_ACTIVITY_OVERHEAD_LWPTI_INSTRUMENTATION = 2<<16,
  /**
   * LWPTI resource creation and destruction overhead.
   */
  LWPTI_ACTIVITY_OVERHEAD_LWPTI_RESOURCE        = 3<<16,
  LWPTI_ACTIVITY_OVERHEAD_FORCE_INT             = 0x7fffffff
} LWpti_ActivityOverheadKind;

/**
 * \brief The kind of a compute API.
 */
typedef enum {
  /**
   * The compute API is not known.
   */
  LWPTI_ACTIVITY_COMPUTE_API_UNKNOWN    = 0,
  /**
   * The compute APIs are for LWCA.
   */
  LWPTI_ACTIVITY_COMPUTE_API_LWDA       = 1,
  /**
   * The compute APIs are for LWCA running
   * in MPS (Multi-Process Service) environment.
   */
  LWPTI_ACTIVITY_COMPUTE_API_LWDA_MPS   = 2,

  LWPTI_ACTIVITY_COMPUTE_API_FORCE_INT  = 0x7fffffff
} LWpti_ActivityComputeApiKind;

/**
 * \brief Flags associated with activity records.
 *
 * Activity record flags. Flags can be combined by bitwise OR to
 * associated multiple flags with an activity record. Each flag is
 * specific to a certain activity kind, as noted below.
 */
typedef enum {
  /**
   * Indicates the activity record has no flags.
   */
  LWPTI_ACTIVITY_FLAG_NONE          = 0,

  /**
   * Indicates the activity represents a device that supports
   * conlwrrent kernel exelwtion. Valid for
   * LWPTI_ACTIVITY_KIND_DEVICE.
   */
  LWPTI_ACTIVITY_FLAG_DEVICE_CONLWRRENT_KERNELS  = 1 << 0,

  /**
   * Indicates if the activity represents a LWdevice_attribute value
   * or a LWpti_DeviceAttribute value. Valid for
   * LWPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE.
   */
  LWPTI_ACTIVITY_FLAG_DEVICE_ATTRIBUTE_LWDEVICE  = 1 << 0,

  /**
   * Indicates the activity represents an asynchronous memcpy
   * operation. Valid for LWPTI_ACTIVITY_KIND_MEMCPY.
   */
  LWPTI_ACTIVITY_FLAG_MEMCPY_ASYNC  = 1 << 0,

  /**
   * Indicates the activity represents an instantaneous marker. Valid
   * for LWPTI_ACTIVITY_KIND_MARKER.
   */
  LWPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS  = 1 << 0,

  /**
   * Indicates the activity represents a region start marker. Valid
   * for LWPTI_ACTIVITY_KIND_MARKER.
   */
  LWPTI_ACTIVITY_FLAG_MARKER_START  = 1 << 1,

  /**
   * Indicates the activity represents a region end marker. Valid for
   * LWPTI_ACTIVITY_KIND_MARKER.
   */
  LWPTI_ACTIVITY_FLAG_MARKER_END  = 1 << 2,

  /**
   * Indicates the activity represents an attempt to acquire a user
   * defined synchronization object.
   * Valid for LWPTI_ACTIVITY_KIND_MARKER.
   */
  LWPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE = 1 << 3,

  /**
   * Indicates the activity represents success in acquiring the
   * user defined synchronization object.
   * Valid for LWPTI_ACTIVITY_KIND_MARKER.
   */
  LWPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_SUCCESS = 1 << 4,

  /**
   * Indicates the activity represents failure in acquiring the
   * user defined synchronization object.
   * Valid for LWPTI_ACTIVITY_KIND_MARKER.
   */
  LWPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_FAILED = 1 << 5,

  /**
   * Indicates the activity represents releasing a reservation on
   * user defined synchronization object.
   * Valid for LWPTI_ACTIVITY_KIND_MARKER.
   */
  LWPTI_ACTIVITY_FLAG_MARKER_SYNC_RELEASE = 1 << 6,

  /**
   * Indicates the activity represents a marker that does not specify
   * a color. Valid for LWPTI_ACTIVITY_KIND_MARKER_DATA.
   */
  LWPTI_ACTIVITY_FLAG_MARKER_COLOR_NONE  = 1 << 0,

  /**
   * Indicates the activity represents a marker that specifies a color
   * in alpha-red-green-blue format. Valid for
   * LWPTI_ACTIVITY_KIND_MARKER_DATA.
   */
  LWPTI_ACTIVITY_FLAG_MARKER_COLOR_ARGB  = 1 << 1,

  /**
   * The number of bytes requested by each thread
   * Valid for LWpti_ActivityGlobalAccess3.
   */
  LWPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_SIZE_MASK  = 0xFF << 0,
  /**
   * If bit in this flag is set, the access was load, else it is a
   * store access. Valid for LWpti_ActivityGlobalAccess3.
   */
  LWPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_LOAD       = 1 << 8,
  /**
   * If this bit in flag is set, the load access was cached else it is
   * uncached. Valid for LWpti_ActivityGlobalAccess3.
   */
  LWPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_CACHED     = 1 << 9,
  /**
   * If this bit in flag is set, the metric value overflowed. Valid
   * for LWpti_ActivityMetric and LWpti_ActivityMetricInstance.
   */
  LWPTI_ACTIVITY_FLAG_METRIC_OVERFLOWED     = 1 << 0,
  /**
   * If this bit in flag is set, the metric value couldn't be
   * callwlated. This oclwrs when a value(s) required to callwlate the
   * metric is missing.  Valid for LWpti_ActivityMetric and
   * LWpti_ActivityMetricInstance.
   */
  LWPTI_ACTIVITY_FLAG_METRIC_VALUE_ILWALID  = 1 << 1,
    /**
   * If this bit in flag is set, the source level metric value couldn't be
   * callwlated. This oclwrs when a value(s) required to callwlate the
   * source level metric cannot be evaluated.
   * Valid for LWpti_ActivityInstructionExelwtion.
   */
  LWPTI_ACTIVITY_FLAG_INSTRUCTION_VALUE_ILWALID  = 1 << 0,
  /**
   * The mask for the instruction class, \ref LWpti_ActivityInstructionClass
   * Valid for LWpti_ActivityInstructionExelwtion and
   * LWpti_ActivityInstructionCorrelation
   */
  LWPTI_ACTIVITY_FLAG_INSTRUCTION_CLASS_MASK    = 0xFF << 1,
  /**
   * When calling lwptiActivityFlushAll, this flag
   * can be set to force LWPTI to flush all records in the buffer, whether
   * finished or not
   */
  LWPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1 << 0,

  /**
   * The number of bytes requested by each thread
   * Valid for LWpti_ActivitySharedAccess.
   */
  LWPTI_ACTIVITY_FLAG_SHARED_ACCESS_KIND_SIZE_MASK  = 0xFF << 0,
  /**
   * If bit in this flag is set, the access was load, else it is a
   * store access.  Valid for LWpti_ActivitySharedAccess.
   */
  LWPTI_ACTIVITY_FLAG_SHARED_ACCESS_KIND_LOAD       = 1 << 8,

  /**
   * Indicates the activity represents an asynchronous memset
   * operation. Valid for LWPTI_ACTIVITY_KIND_MEMSET.
   */
  LWPTI_ACTIVITY_FLAG_MEMSET_ASYNC  = 1 << 0,

  /**
   * Indicates the activity represents thrashing in CPU.
   * Valid for counter of kind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING in
   * LWPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
   */
  LWPTI_ACTIVITY_FLAG_THRASHING_IN_CPU = 1 << 0,

 /**
   * Indicates the activity represents page throttling in CPU.
   * Valid for counter of kind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING in
   * LWPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
   */
  LWPTI_ACTIVITY_FLAG_THROTTLING_IN_CPU = 1 << 0,

  LWPTI_ACTIVITY_FLAG_FORCE_INT = 0x7fffffff
} LWpti_ActivityFlag;

/**
 * \brief The stall reason for PC sampling activity.
 */
typedef enum {
  /**
   * Invalid reason
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_ILWALID      = 0,
   /**
   * No stall, instruction is selected for issue
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_NONE         = 1,
  /**
   * Warp is blocked because next instruction is not yet available,
   * because of instruction cache miss, or because of branching effects
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH   = 2,
  /**
   * Instruction is waiting on an arithmatic dependency
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY   = 3,
  /**
   * Warp is blocked because it is waiting for a memory access to complete.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY   = 4,
  /**
   * Texture sub-system is fully utilized or has too many outstanding requests.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE   = 5,
  /**
   * Warp is blocked as it is waiting at __syncthreads() or at memory barrier.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC   = 6,
  /**
   * Warp is blocked waiting for __constant__ memory and immediate memory access to complete.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY   = 7,
  /**
   * Compute operation cannot be performed due to the required resources not
   * being available.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY   = 8,
  /**
   * Warp is blocked because there are too many pending memory operations.
   * In Kepler architecture it often indicates high number of memory replays.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE   = 9,
  /**
   * Warp was ready to issue, but some other warp issued instead.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED   = 10,
  /**
   * Miscellaneous reasons
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER   = 11,
  /**
   * Sleeping.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING   = 12,
  LWPTI_ACTIVITY_PC_SAMPLING_STALL_FORCE_INT  = 0x7fffffff
} LWpti_ActivityPCSamplingStallReason;

/**
 * \brief Sampling period for PC sampling method
 * Sampling period can be set using /ref lwptiActivityConfigurePCSampling
 */
typedef enum {
  /**
   * The PC sampling period is not set.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_PERIOD_ILWALID = 0,
  /**
   * Minimum sampling period available on the device.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN = 1,
  /**
   * Sampling period in lower range.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW = 2,
  /**
   * Medium sampling period.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID = 3,
  /**
   * Sampling period in higher range.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_PERIOD_HIGH = 4,
  /**
   * Maximum sampling period available on the device.
   */
  LWPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX = 5,
  LWPTI_ACTIVITY_PC_SAMPLING_PERIOD_FORCE_INT = 0x7fffffff
} LWpti_ActivityPCSamplingPeriod;

/**
 * \brief The kind of a memory copy, indicating the source and
 * destination targets of the copy.
 *
 * Each kind represents the source and destination targets of a memory
 * copy. Targets are host, device, and array.
 */
typedef enum {
  /**
   * The memory copy kind is not known.
   */
  LWPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN = 0,
  /**
   * A host to device memory copy.
   */
  LWPTI_ACTIVITY_MEMCPY_KIND_HTOD    = 1,
  /**
   * A device to host memory copy.
   */
  LWPTI_ACTIVITY_MEMCPY_KIND_DTOH    = 2,
  /**
   * A host to device array memory copy.
   */
  LWPTI_ACTIVITY_MEMCPY_KIND_HTOA    = 3,
  /**
   * A device array to host memory copy.
   */
  LWPTI_ACTIVITY_MEMCPY_KIND_ATOH    = 4,
  /**
   * A device array to device array memory copy.
   */
  LWPTI_ACTIVITY_MEMCPY_KIND_ATOA    = 5,
  /**
   * A device array to device memory copy.
   */
  LWPTI_ACTIVITY_MEMCPY_KIND_ATOD    = 6,
  /**
   * A device to device array memory copy.
   */
  LWPTI_ACTIVITY_MEMCPY_KIND_DTOA    = 7,
  /**
   * A device to device memory copy on the same device.
   */
  LWPTI_ACTIVITY_MEMCPY_KIND_DTOD    = 8,
  /**
   * A host to host memory copy.
   */
  LWPTI_ACTIVITY_MEMCPY_KIND_HTOH    = 9,
  /**
   * A peer to peer memory copy across different devices.
   */
  LWPTI_ACTIVITY_MEMCPY_KIND_PTOP    = 10,

  LWPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT = 0x7fffffff
} LWpti_ActivityMemcpyKind;

/**
 * \brief The kinds of memory accessed by a memory operation/copy.
 *
 * Each kind represents the type of the memory
 * accessed by a memory operation/copy.
 */
typedef enum {
  /**
   * The memory kind is unknown.
   */
  LWPTI_ACTIVITY_MEMORY_KIND_UNKNOWN            = 0,
  /**
   * The memory is pageable.
   */
  LWPTI_ACTIVITY_MEMORY_KIND_PAGEABLE           = 1,
  /**
   * The memory is pinned.
   */
  LWPTI_ACTIVITY_MEMORY_KIND_PINNED             = 2,
  /**
   * The memory is on the device.
   */
  LWPTI_ACTIVITY_MEMORY_KIND_DEVICE             = 3,
  /**
   * The memory is an array.
   */
  LWPTI_ACTIVITY_MEMORY_KIND_ARRAY              = 4,
  /**
   * The memory is managed
   */
  LWPTI_ACTIVITY_MEMORY_KIND_MANAGED            = 5,
  /**
   * The memory is device static
   */
  LWPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC      = 6,
  /**
   * The memory is managed static
   */
  LWPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC     = 7,
  LWPTI_ACTIVITY_MEMORY_KIND_FORCE_INT          = 0x7fffffff
} LWpti_ActivityMemoryKind;

/**
 * \brief The kind of a preemption activity.
 */
typedef enum {
  /**
   * The preemption kind is not known.
   */
  LWPTI_ACTIVITY_PREEMPTION_KIND_UNKNOWN    = 0,
  /**
   * Preemption to save CDP block.
   */
  LWPTI_ACTIVITY_PREEMPTION_KIND_SAVE       = 1,
  /**
   * Preemption to restore CDP block.
   */
  LWPTI_ACTIVITY_PREEMPTION_KIND_RESTORE    = 2,
  LWPTI_ACTIVITY_PREEMPTION_KIND_FORCE_INT  = 0x7fffffff
} LWpti_ActivityPreemptionKind;

/**
 * \brief The kind of environment data. Used to indicate what type of
 * data is being reported by an environment activity record.
 */
typedef enum {
  /**
   * Unknown data.
   */
  LWPTI_ACTIVITY_ELWIRONMENT_UNKNOWN = 0,
  /**
   * The environment data is related to speed.
   */
  LWPTI_ACTIVITY_ELWIRONMENT_SPEED = 1,
  /**
   * The environment data is related to temperature.
   */
  LWPTI_ACTIVITY_ELWIRONMENT_TEMPERATURE = 2,
  /**
   * The environment data is related to power.
   */
  LWPTI_ACTIVITY_ELWIRONMENT_POWER = 3,
  /**
   * The environment data is related to cooling.
   */
  LWPTI_ACTIVITY_ELWIRONMENT_COOLING = 4,

  LWPTI_ACTIVITY_ELWIRONMENT_COUNT,
  LWPTI_ACTIVITY_ELWIRONMENT_KIND_FORCE_INT    = 0x7fffffff
} LWpti_ActivityElwironmentKind;

/**
 * \brief Reasons for clock throttling.
 *
 * The possible reasons that a clock can be throttled. There can be
 * more than one reason that a clock is being throttled so these types
 * can be combined by bitwise OR.  These are used in the
 * clocksThrottleReason field in the Environment Activity Record.
 */
typedef enum {
  /**
   * Nothing is running on the GPU and the clocks are dropping to idle
   * state.
   */
  LWPTI_CLOCKS_THROTTLE_REASON_GPU_IDLE              = 0x00000001,
  /**
   * The GPU clocks are limited by a user specified limit.
   */
  LWPTI_CLOCKS_THROTTLE_REASON_USER_DEFINED_CLOCKS   = 0x00000002,
  /**
   * A software power scaling algorithm is reducing the clocks below
   * requested clocks.
   */
  LWPTI_CLOCKS_THROTTLE_REASON_SW_POWER_CAP          = 0x00000004,
  /**
   * Hardware slowdown to reduce the clock by a factor of two or more
   * is engaged.  This is an indicator of one of the following: 1)
   * Temperature is too high, 2) External power brake assertion is
   * being triggered (e.g. by the system power supply), 3) Change in
   * power state.
   */
  LWPTI_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN           = 0x00000008,
  /**
   * Some unspecified factor is reducing the clocks.
   */
  LWPTI_CLOCKS_THROTTLE_REASON_UNKNOWN               = 0x80000000,
  /**
   * Throttle reason is not supported for this GPU.
   */
  LWPTI_CLOCKS_THROTTLE_REASON_UNSUPPORTED           = 0x40000000,
  /**
   * No clock throttling.
   */
  LWPTI_CLOCKS_THROTTLE_REASON_NONE                  = 0x00000000,

  LWPTI_CLOCKS_THROTTLE_REASON_FORCE_INT             = 0x7fffffff
} LWpti_ElwironmentClocksThrottleReason;

/**
 * \brief Scope of the unified memory counter (deprecated in LWCA 7.0)
 */
typedef enum {
  /**
   * The unified memory counter scope is not known.
   */
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_UNKNOWN = 0,
  /**
   * Collect unified memory counter for single process on one device
   */
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE = 1,
  /**
   * Collect unified memory counter for single process across all devices
   */
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES = 2,

  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_COUNT,
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_FORCE_INT = 0x7fffffff
} LWpti_ActivityUnifiedMemoryCounterScope;

/**
 * \brief Kind of the Unified Memory counter
 *
 * Many activities are associated with Unified Memory mechanism; among them
 * are tranfer from host to device, device to host, page fault at
 * host side.
 */
typedef enum {
  /**
   * The unified memory counter kind is not known.
   */
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_UNKNOWN = 0,
  /**
   * Number of bytes transfered from host to device
   */
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD = 1,
  /**
   * Number of bytes transfered from device to host
   */
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH = 2,
  /**
   * Number of CPU page faults, this is only supported on 64 bit
   * Linux and Mac platforms
   */
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT = 3,
  /**
   * Number of GPU page faults, this is only supported on devices with 
   * compute capability 6.0 and higher and 64 bit Linux platforms
   */
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT = 4,
  /**
   * Thrashing oclwrs when data is frequently accessed by
   * multiple processors and has to be constantly migrated around
   * to achieve data locality. In this case the overhead of migration
   * may exceed the benefits of locality.
   * This is only supported on 64 bit Linux platforms.
   */ 
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING = 5,
  /**
   * Throttling is a prevention technique used by the driver to avoid
   * further thrashing. Here, the driver doesn't service the fault for
   * one of the contending processors for a specific period of time,
   * so that the other processor can run at full-speed.
   * This is only supported on 64 bit Linux platforms.
   */
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING = 6,
  /**
   * In case throttling does not help, the driver tries to pin the memory
   * to a processor for a specific period of time. One of the contending
   * processors will have slow  access to the memory, while the other will
   * have fast access.
   * This is only supported on 64 bit Linux platforms.
   */
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP = 7,

  /**
   * Number of bytes transferred from one device to another device.
   * This is only supported on 64 bit Linux platforms.
   */
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD = 8,

  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_COUNT,
  LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_FORCE_INT = 0x7fffffff
} LWpti_ActivityUnifiedMemoryCounterKind;

/**
 * \brief Memory access type for unified memory page faults
 *
 * This is valid for \ref LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT
 * and \ref LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT
 */
typedef enum {
    /**
     * The unified memory access type is not known
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_UNKNOWN = 0,
    /**
     * The page fault was triggered by read memory instruction
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_READ = 1,
    /**
     * The page fault was triggered by write memory instruction
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_WRITE = 2,
    /**
     * The page fault was triggered by atomic memory instruction
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_ATOMIC = 3,
    /**
     * The page fault was triggered by memory prefetch operation
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_PREFETCH = 4
} LWpti_ActivityUnifiedMemoryAccessType;

/**
 * \brief Migration cause of the Unified Memory counter
 *
 * This is valid for \ref LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD and
 * \ref LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH
 */
typedef enum {
    /**
     * The unified memory migration cause is not known
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_UNKNOWN = 0,
    /**
     * The unified memory migrated due to an explicit call from 
     * the user e.g. lwdaMemPrefetchAsync
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_USER = 1,
    /**
     * The unified memory migrated to guarantee data coherence
     * e.g. CPU/GPU faults on Pascal+ and kernel launch on pre-Pascal GPUs
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_COHERENCE = 2,
    /**
     * The unified memory was spelwlatively migrated by the UVM driver
     * before being accessed by the destination processor to improve 
     * performance
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_PREFETCH = 3,
    /**
     * The unified memory migrated to the CPU because it was evicted to make
     * room for another block of memory on the GPU
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_EVICTION = 4,
    /**
      * The unified memory migrated to another processor because of access counter
      * notifications
      */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_ACCESS_COUNTERS = 5,
} LWpti_ActivityUnifiedMemoryMigrationCause;

typedef enum {
    /**
     * The cause of mapping to remote memory was unknown
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_UNKNOWN = 0,
    /**
     * Mapping to remote memory was added to maintain data coherence.
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_COHERENCE = 1,
    /**
     * Mapping to remote memory was added to prevent further thrashing
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_THRASHING = 2,
    /**
     * Mapping to remote memory was added to enforce the hints 
     * specified by the programmer or by performance heuristics of the
     * UVM driver
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_POLICY = 3,
    /**
     * Mapping to remote memory was added because there is no more
     * memory available on the processor and eviction was not
     * possible
     */ 
    LWPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_OUT_OF_MEMORY = 4,
    /**
     * Mapping to remote memory was added after the memory was 
     * evicted to make room for another block of memory on the GPU
     */
    LWPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_EVICTION = 5,
} LWpti_ActivityUnifiedMemoryRemoteMapCause;

/**
 * \brief SASS instruction classification.
 *
 * The sass instruction are broadly divided into different class. Each enum represents a classification.
 */
typedef enum {
  /**
   * The instruction class is not known.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_UNKNOWN = 0,
  /**
   * Represents a 32 bit floating point operation.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_FP_32 = 1,
  /**
   * Represents a 64 bit floating point operation.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_FP_64 = 2,
  /**
   * Represents an integer operation.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_INTEGER = 3,
  /**
   * Represents a bit colwersion operation.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_BIT_COLWERSION = 4,
  /**
   * Represents a control flow instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_CONTROL_FLOW = 5,
  /**
   * Represents a global load-store instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL = 6,
  /**
   * Represents a shared load-store instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED = 7,
  /**
   * Represents a local load-store instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_LOCAL = 8,
  /**
   * Represents a generic load-store instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_GENERIC = 9,
  /**
   * Represents a surface load-store instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE = 10,
  /**
   * Represents a constant load instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_CONSTANT = 11,
  /**
   * Represents a texture load-store instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_TEXTURE = 12,
  /**
   * Represents a global atomic instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL_ATOMIC = 13,
  /**
   * Represents a shared atomic instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED_ATOMIC = 14,
  /**
   * Represents a surface atomic instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE_ATOMIC = 15,
  /**
   * Represents a inter-thread communication instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_INTER_THREAD_COMMUNICATION = 16,
  /**
   * Represents a barrier instruction.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_BARRIER = 17,
  /**
   * Represents some miscellaneous instructions which do not fit in the above classification.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_MISCELLANEOUS = 18,
  /**
   * Represents a 16 bit floating point operation.
   */
  LWPTI_ACTIVITY_INSTRUCTION_CLASS_FP_16 = 19,

  /**
   * Represents uniform instruction.
   */

  LWPTI_ACTIVITY_INSTRUCTION_CLASS_UNIFORM = 20,

  LWPTI_ACTIVITY_INSTRUCTION_CLASS_KIND_FORCE_INT     = 0x7fffffff
} LWpti_ActivityInstructionClass;

/**
 * \brief Partitioned global caching option
 */
typedef enum {
  /**
   * Partitioned global cache config unknown.
   */
  LWPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN = 0,
  /**
   * Partitioned global cache not supported.
   */
  LWPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED = 1,
  /**
   * Partitioned global cache config off.
   */
  LWPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF = 2,
  /**
   * Partitioned global cache config on.
   */
  LWPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON = 3,
  LWPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_FORCE_INT  = 0x7fffffff
} LWpti_ActivityPartitionedGlobalCacheConfig;

/**
 * \brief Synchronization type.
 *
 * The types of synchronization to be used with LWpti_ActivitySynchronization.
 */

typedef enum {
  /**
   * Unknown data.
   */
  LWPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN = 0,
  /**
   * Event synchronize API.
   */
  LWPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE = 1,
  /**
   * Stream wait event API.
   */
  LWPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT = 2,
  /**
   * Stream synchronize API.
   */
  LWPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE = 3,
  /**
   * Context synchronize API.
   */
  LWPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE = 4,

  LWPTI_ACTIVITY_SYNCHRONIZATION_TYPE_FORCE_INT     = 0x7fffffff
} LWpti_ActivitySynchronizationType;

/**
 * \brief stream type.
 *
 * The types of stream to be used with LWpti_ActivityStream.
 */

typedef enum {
  /**
   * Unknown data.
   */
  LWPTI_ACTIVITY_STREAM_CREATE_FLAG_UNKNOWN = 0,
  /**
   * Default stream.
   */
  LWPTI_ACTIVITY_STREAM_CREATE_FLAG_DEFAULT = 1,
  /**
   * Non-blocking stream.
   */
  LWPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING = 2,
  /**
   * Null stream.
   */
  LWPTI_ACTIVITY_STREAM_CREATE_FLAG_NULL = 3,
  /**
   * Stream create Mask
   */
  LWPTI_ACTIVITY_STREAM_CREATE_MASK = 0xFFFF,

  LWPTI_ACTIVITY_STREAM_CREATE_FLAG_FORCE_INT = 0x7fffffff
} LWpti_ActivityStreamFlag;

/**
* \brief Link flags.
*
* Describes link properties, to be used with LWpti_ActivityLwLink.
*/

typedef enum {
  LWPTI_LINK_FLAG_ILWALID = 0,
  /**
  * Is peer to peer access supported by this link.
  */
  LWPTI_LINK_FLAG_PEER_ACCESS = (1 << 1),
  /**
  * Is system memory access supported by this link.
  */
  LWPTI_LINK_FLAG_SYSMEM_ACCESS = (1 << 2),
  /**
  * Is peer atomic access supported by this link.
  */
  LWPTI_LINK_FLAG_PEER_ATOMICS = (1 << 3),
  /**
  * Is system memory atomic access supported by this link.
  */
  LWPTI_LINK_FLAG_SYSMEM_ATOMICS = (1 << 4),

  LWPTI_LINK_FLAG_FORCE_INT = 0x7fffffff
} LWpti_LinkFlag;

/**
 * The source-locator ID that indicates an unknown source
 * location. There is not an actual LWpti_ActivitySourceLocator object
 * corresponding to this value.
 */
#define LWPTI_SOURCE_LOCATOR_ID_UNKNOWN 0

/**
 * An invalid function index ID.
 */
#define LWPTI_FUNCTION_INDEX_ID_ILWALID 0

/**
 * An invalid/unknown correlation ID. A correlation ID of this value
 * indicates that there is no correlation for the activity record.
 */
#define LWPTI_CORRELATION_ID_UNKNOWN 0

/**
 * An invalid/unknown grid ID.
 */
#define LWPTI_GRID_ID_UNKNOWN 0LL

/**
 * An invalid/unknown timestamp for a start, end, queued, submitted,
 * or completed time.
 */
#define LWPTI_TIMESTAMP_UNKNOWN 0LL

/**
 * An invalid/unknown value.
 */
#define LWPTI_SYNCHRONIZATION_ILWALID_VALUE -1

/**
 * An invalid/unknown process id.
 */
#define LWPTI_AUTO_BOOST_ILWALID_CLIENT_PID 0

/**
 * Invalid/unknown LWLink port number.
*/
#define LWPTI_LWLINK_ILWALID_PORT -1

/**
 * Maximum LWLink port numbers.
*/
#define LWPTI_MAX_LWLINK_PORTS 16

START_PACKED_ALIGNMENT
/**
 * \brief Unified Memory counters configuration structure
 *
 * This structure controls the enable/disable of the various
 * Unified Memory counters consisting of scope, kind and other parameters.
 * See function /ref lwptiActivityConfigureUnifiedMemoryCounter
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * Unified Memory counter Counter scope. (deprecated in LWCA 7.0)
   */
  LWpti_ActivityUnifiedMemoryCounterScope scope;

  /**
   * Unified Memory counter Counter kind
   */
  LWpti_ActivityUnifiedMemoryCounterKind kind;

  /**
   * Device id of the traget device. This is relevant only
   * for single device scopes. (deprecated in LWCA 7.0)
   */
  uint32_t deviceId;

  /**
   * Control to enable/disable the counter. To enable the counter
   * set it to non-zero value while disable is indicated by zero.
   */
  uint32_t enable;
} LWpti_ActivityUnifiedMemoryCounterConfig;

/**
 * \brief Device auto boost state structure
 *
 * This structure defines auto boost state for a device.
 * See function /ref lwptiGetAutoBoostState
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * Returned auto boost state. 1 is returned in case auto boost is enabled, 0
   * otherwise
   */
  uint32_t enabled;

  /**
   * Id of process that has set the current boost state. The value will be
   * LWPTI_AUTO_BOOST_ILWALID_CLIENT_PID if the user does not have the
   * permission to query process ids or there is an error in querying the
   * process id.
   */
  uint32_t pid;

} LWpti_ActivityAutoBoostState;

/**
 * \brief PC sampling configuration structure
 *
 * This structure defines the pc sampling configuration.
 *
 * See function /ref lwptiActivityConfigurePCSampling
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * Size of configuration structure.
   * LWPTI client should set the size of the structure. It will be used in LWPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  uint32_t size;
  /**
   * There are 5 level provided for sampling period. The level
   * internally maps to a period in terms of cycles. Same level can
   * map to different number of cycles on different gpus. No of
   * cycles will be chosen to minimize information loss. The period
   * chosen will be given by samplingPeriodInCycles in
   * /ref LWpti_ActivityPCSamplingRecordInfo for each kernel instance.
   */
  LWpti_ActivityPCSamplingPeriod samplingPeriod;

  /**
   * This will override the period set by samplingPeriod. Value 0 in samplingPeriod2 will be
   * considered as samplingPeriod2 should not be used and samplingPeriod should be used.
   * Valid values for samplingPeriod2 are between 5 to 31 both inclusive.
   * This will set the sampling period to (2^samplingPeriod2) cycles.
   */
  uint32_t samplingPeriod2;
} LWpti_ActivityPCSamplingConfig;

/**
 * \brief The base activity record.
 *
 * The activity API uses a LWpti_Activity as a generic representation
 * for any activity. The 'kind' field is used to determine the
 * specific activity kind, and from that the LWpti_Activity object can
 * be cast to the specific activity record type appropriate for that kind.
 *
 * Note that all activity record types are padded and aligned to
 * ensure that each member of the record is naturally aligned.
 *
 * \see LWpti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The kind of this activity.
   */
  LWpti_ActivityKind kind;
} LWpti_Activity;

/**
 * \brief The activity record for memory copies.
 *
 * This activity record represents a memory copy
 * (LWPTI_ACTIVITY_KIND_MEMCPY).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_MEMCPY.
   */
  LWpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. \see LWpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. \see LWpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. \see LWpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see LWpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory copy is oclwrring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is oclwrring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is oclwrring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  uint32_t runtimeCorrelationId;

#ifdef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} LWpti_ActivityMemcpy;

/**
 * \brief The activity record for memory copies.
 *
 * This activity record represents a memory copy
 * (LWPTI_ACTIVITY_KIND_MEMCPY).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_MEMCPY.
   */
  LWpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. \see LWpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. \see LWpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. \see LWpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see LWpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory copy is oclwrring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is oclwrring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is oclwrring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  uint32_t runtimeCorrelationId;

#ifdef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that exelwted this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint64_t graphNodeId;
} LWpti_ActivityMemcpy3;

/**
 * \brief The activity record for peer-to-peer memory copies.
 *
 * This activity record represents a peer-to-peer memory copy
 * (LWPTI_ACTIVITY_KIND_MEMCPY2) but is no longer generated
 * by LWPTI. Peer-to-peer memory copy activities are now reported using the
 * LWpti_ActivityMemcpyPtoP2 activity record..
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_MEMCPY2.
   */
  LWpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size.  \see LWpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size.  \see LWpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size.  \see LWpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see
   * LWpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
  * The ID of the device where the memory copy is oclwrring.
  */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is oclwrring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is oclwrring.
   */
  uint32_t streamId;

  /**
   * The ID of the device where memory is being copied from.
   */
  uint32_t srcDeviceId;

  /**
   * The ID of the context owning the memory being copied from.
   */
  uint32_t srcContextId;

  /**
   * The ID of the device where memory is being copied to.
   */
  uint32_t dstDeviceId;

  /**
   * The ID of the context owning the memory being copied to.
   */
  uint32_t dstContextId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory copy.
   */
  uint32_t correlationId;

#ifndef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} LWpti_ActivityMemcpyPtoP;

typedef LWpti_ActivityMemcpyPtoP LWpti_ActivityMemcpy2;

/**
 * \brief The activity record for peer-to-peer memory copies.
 *
 * This activity record represents a peer-to-peer memory copy
 * (LWPTI_ACTIVITY_KIND_MEMCPY2).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_MEMCPY2.
   */
  LWpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size.  \see LWpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size.  \see LWpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size.  \see LWpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see
   * LWpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
  * The ID of the device where the memory copy is oclwrring.
  */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is oclwrring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is oclwrring.
   */
  uint32_t streamId;

  /**
   * The ID of the device where memory is being copied from.
   */
  uint32_t srcDeviceId;

  /**
   * The ID of the context owning the memory being copied from.
   */
  uint32_t srcContextId;

  /**
   * The ID of the device where memory is being copied to.
   */
  uint32_t dstDeviceId;

  /**
   * The ID of the context owning the memory being copied to.
   */
  uint32_t dstContextId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory copy.
   */
  uint32_t correlationId;

#ifndef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that exelwted the memcpy through graph launch.
   * This field will be 0 if memcpy is not done using graph launch.
   */
  uint64_t graphNodeId;
} LWpti_ActivityMemcpyPtoP2;

/**
 * \brief The activity record for memset.
 *
 * This activity record represents a memory set operation
 * (LWPTI_ACTIVITY_KIND_MEMSET).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_MEMSET.
   */
  LWpti_ActivityKind kind;

  /**
   * The value being assigned to memory by the memory set.
   */
  uint32_t value;

  /**
   * The number of bytes being set by the memory set.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory set is oclwrring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory set is oclwrring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory set is oclwrring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory set. Each memory set is assigned
   * a unique correlation ID that is identical to the correlation ID
   * in the driver API activity record that launched the memory set.
   */
  uint32_t correlationId;

  /**
   * The flags associated with the memset. \see LWpti_ActivityFlag
   */
  uint16_t flags;

  /**
   * The memory kind of the memory set \see LWpti_ActivityMemoryKind
   */
  uint16_t memoryKind;

#ifdef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} LWpti_ActivityMemset;

/**
 * \brief The activity record for memset.
 *
 * This activity record represents a memory set operation
 * (LWPTI_ACTIVITY_KIND_MEMSET).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_MEMSET.
   */
  LWpti_ActivityKind kind;

  /**
   * The value being assigned to memory by the memory set.
   */
  uint32_t value;

  /**
   * The number of bytes being set by the memory set.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory set is oclwrring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory set is oclwrring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory set is oclwrring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory set. Each memory set is assigned
   * a unique correlation ID that is identical to the correlation ID
   * in the driver API activity record that launched the memory set.
   */
  uint32_t correlationId;

  /**
   * The flags associated with the memset. \see LWpti_ActivityFlag
   */
  uint16_t flags;

  /**
   * The memory kind of the memory set \see LWpti_ActivityMemoryKind
   */
  uint16_t memoryKind;

#ifdef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that exelwted this memset through graph launch.
   * This field will be 0 if the memset is not exelwted through graph launch.
   */
  uint64_t graphNodeId;
} LWpti_ActivityMemset2;

/**
 * \brief The activity record for memory.
 *
 * This activity record represents a memory allocation and free operation
 * (LWPTI_ACTIVITY_KIND_MEMORY).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_MEMORY
   */
  LWpti_ActivityKind kind;

  /**
   * The memory kind requested by the user
   */
  LWpti_ActivityMemoryKind memoryKind;

  /**
   * The virtual address of the allocation
   */
  uint64_t address;

  /**
   * The number of bytes of memory allocated.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory operation, i.e.
   * the time when memory was allocated, in ns.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory operation, i.e.
   * the time when memory was freed, in ns.
   * This will be 0 if memory is not freed in the application
   */
  uint64_t end;

  /**
   * The program counter of the allocation of memory
   */
  uint64_t allocPC;

  /**
   * The program counter of the freeing of memory. This will
   * be 0 if memory is not freed in the application
   */
  uint64_t freePC;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The ID of the device where the memory allocation is taking place.
   */
  uint32_t deviceId;

  /**
   * The ID of the context. If context is NULL, \param contextId is set to LWPTI_ILWALID_CONTEXT_ID. 
   */
  uint32_t contextId;

#ifdef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Variable name. This name is shared across all activity
   * records representing the same symbol, and so should not be
   * modified.
   */
  const char* name;
} LWpti_ActivityMemory;

/**
 * \brief The activity record for kernel. (deprecated)
 *
 * This activity record represents a kernel exelwtion
 * (LWPTI_ACTIVITY_KIND_KERNEL and
 * LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL) but is no longer generated
 * by LWPTI. Kernel activities are now reported using the
 * LWpti_ActivityKernel5 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_KERNEL
   * or LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL.
   */
  LWpti_ActivityKind kind;

  /**
   * The cache configuration requested by the kernel. The value is one
   * of the LWfunc_cache enumeration values from lwca.h.
   */
  uint8_t cacheConfigRequested;

  /**
   * The cache configuration used for the kernel. The value is one of
   * the LWfunc_cache enumeration values from lwca.h.
   */
  uint8_t cacheConfigExelwted;

  /**
   * The number of registers required for each thread exelwting the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The start timestamp for the kernel exelwtion, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel exelwtion, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The ID of the device where the kernel is exelwting.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is exelwting.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is exelwting.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel exelwtion is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the kernel.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the kernel. Each kernel exelwtion
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the kernel.
   */
  uint32_t runtimeCorrelationId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} LWpti_ActivityKernel;

/**
 * \brief The activity record for kernel. (deprecated)
 *
 * This activity record represents a kernel exelwtion
 * (LWPTI_ACTIVITY_KIND_KERNEL and
 * LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL) but is no longer generated
 * by LWPTI. Kernel activities are now reported using the
 * LWpti_ActivityKernel5 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_KERNEL or
   * LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL.
   */
  LWpti_ActivityKind kind;

  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the LWfunc_cache enumeration values from lwca.h.
       */
      uint8_t requested:4;
      /**
       * The cache configuration used for the kernel. The value is one of
       * the LWfunc_cache enumeration values from lwca.h.
       */
      uint8_t exelwted:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the LWsharedconfig enumeration values from lwca.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread exelwting the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The start timestamp for the kernel exelwtion, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel exelwtion, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel exelwtion, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of LWPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is exelwting.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is exelwting.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is exelwting.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel exelwtion is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} LWpti_ActivityKernel2;

/**
 * \brief The activity record for a kernel (LWCA 6.5(with sm_52 support) onwards).
 * (deprecated in LWCA 9.0)
 *
 * This activity record represents a kernel exelwtion
 * (LWPTI_ACTIVITY_KIND_KERNEL and
 * LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL).
 * Kernel activities are now reported using the LWpti_ActivityKernel5 activity
 * record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_KERNEL or
   * LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL.
   */
  LWpti_ActivityKind kind;

  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the LWfunc_cache enumeration values from lwca.h.
       */
      uint8_t requested:4;
      /**
       * The cache configuration used for the kernel. The value is one of
       * the LWfunc_cache enumeration values from lwca.h.
       */
      uint8_t exelwted:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the LWsharedconfig enumeration values from lwca.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread exelwting the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  LWpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching exelwted for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the oclwpancy requirement of the launch cannot
   * support caching.
   */
  LWpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExelwted;

  /**
   * The start timestamp for the kernel exelwtion, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel exelwtion, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel exelwtion, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of LWPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is exelwting.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is exelwting.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is exelwting.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel exelwtion is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} LWpti_ActivityKernel3;

/**
 * \brief The type of the LWCA kernel launch.
 */
typedef enum {
  /**
  * The kernel was launched via a regular kernel call
  */
  LWPTI_ACTIVITY_LAUNCH_TYPE_REGULAR = 0,
  /**
  * The kernel was launched via API \ref lwdaLaunchCooperativeKernel() or
  * \ref lwLaunchCooperativeKernel()
  */
  LWPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_SINGLE_DEVICE = 1,
  /**
  * The kernel was launched via API \ref lwdaLaunchCooperativeKernelMultiDevice() or
  * \ref lwLaunchCooperativeKernelMultiDevice()
  */
  LWPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_MULTI_DEVICE = 2
} LWpti_ActivityLaunchType;

/**
 * \brief The activity record for a kernel (LWCA 9.0(with sm_70 support) onwards).
 * (deprecated in LWCA 11.0)
 *
 * This activity record represents a kernel exelwtion
 * (LWPTI_ACTIVITY_KIND_KERNEL and
 * LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL).
 * Kernel activities are now reported using the LWpti_ActivityKernel5 activity
 * record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_KERNEL or
   * LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL.
   */
  LWpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the LWfunc_cache enumeration values from lwca.h.
       */
      uint8_t requested:4;
      /**
       * The cache configuration used for the kernel. The value is one of
       * the LWfunc_cache enumeration values from lwca.h.
       */
      uint8_t exelwted:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the LWsharedconfig enumeration values from lwca.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread exelwting the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  LWpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching exelwted for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the oclwpancy requirement of the launch cannot
   * support caching.
   */
  LWpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExelwted;

  /**
   * The start timestamp for the kernel exelwtion, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel exelwtion, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel exelwtion, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of LWPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is exelwting.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is exelwting.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is exelwting.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel exelwtion is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of LWPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref lwptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by LWCA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of LWCA
   * kernels are asynchrnous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of LWPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * lwptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was exelwted via a regular launch or via a
   * single/multi device cooperative launch. \see LWpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if LW_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExelwted; 
} LWpti_ActivityKernel4;

/**
 * \brief The shared memory limit per block config for a kernel
 * This should be used to set 'lwdaOccFuncShmemConfig' field in oclwpancy calculator API
 */
typedef enum  {
    /* The shared memory limit config is default */
    LWPTI_FUNC_SHMEM_LIMIT_DEFAULT              = 0x00,
    /* User has opted for a higher dynamic shared memory limit using function attribute 
       'lwdaFuncAttributeMaxDynamicSharedMemorySize' for runtime API or 
       LW_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES for driver API */
    LWPTI_FUNC_SHMEM_LIMIT_OPTIN                = 0x01,
    LWPTI_FUNC_SHMEM_LIMIT_FORCE_INT            = 0x7fffffff
} LWpti_FuncShmemLimitConfig;

/**
 * This activity record represents a kernel exelwtion
 * (LWPTI_ACTIVITY_KIND_KERNEL and
 * LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_KERNEL or
   * LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL.
   */
  LWpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the LWfunc_cache enumeration values from lwca.h.
       */
      uint8_t requested:4;
      /**
       * The cache configuration used for the kernel. The value is one of
       * the LWfunc_cache enumeration values from lwca.h.
       */
      uint8_t exelwted:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the LWsharedconfig enumeration values from lwca.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread exelwting the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  LWpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching exelwted for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the oclwpancy requirement of the launch cannot
   * support caching.
   */
  LWpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExelwted;

  /**
   * The start timestamp for the kernel exelwtion, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel exelwtion, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel exelwtion, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of LWPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is exelwting.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is exelwting.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is exelwting.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel exelwtion is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of LWPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref lwptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by LWCA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of LWCA
   * kernels are asynchrnous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of LWPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * lwptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was exelwted via a regular launch or via a
   * single/multi device cooperative launch. \see LWpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if LW_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExelwted;

  /**
   * The unique ID of the graph node that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint64_t graphNodeId;

  /**
   * The shared memory limit config for the kernel. This field shows whether user has opted for a
   * higher per block limit of dynamic shared memory.
   */
  LWpti_FuncShmemLimitConfig shmemLimitConfig;

  uint32_t padding2;
} LWpti_ActivityKernel5;

/**
 * \brief The activity record for CDP (LWCA Dynamic Parallelism)
 * kernel.
 *
 * This activity record represents a CDP kernel exelwtion.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_CDP_KERNEL
   */
  LWpti_ActivityKind kind;

  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the LWfunc_cache enumeration values from lwca.h.
       */
      uint8_t requested:4;
      /**
       * The cache configuration used for the kernel. The value is one of
       * the LWfunc_cache enumeration values from lwca.h.
       */
      uint8_t exelwted:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the LWsharedconfig enumeration values from lwca.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread exelwting the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The start timestamp for the kernel exelwtion, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel exelwtion, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The ID of the device where the kernel is exelwting.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is exelwting.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is exelwting.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel exelwtion is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel exelwtion
   * is assigned a unique grid ID.
   */
  int64_t gridId;

  /**
   * The grid ID of the parent kernel.
   */
  int64_t parentGridId;

  /**
   * The timestamp when kernel is queued up, in ns. A value of
   * LWPTI_TIMESTAMP_UNKNOWN indicates that the queued time is
   * unknown.
   */
  uint64_t queued;

  /**
   * The timestamp when kernel is submitted to the gpu, in ns. A value
   * of LWPTI_TIMESTAMP_UNKNOWN indicates that the submission time is
   * unknown.
   */
  uint64_t submitted;

  /**
   * The timestamp when kernel is marked as completed, in ns. A value
   * of LWPTI_TIMESTAMP_UNKNOWN indicates that the completion time is
   * unknown.
   */
  uint64_t completed;

  /**
   * The X-dimension of the parent block.
   */
  uint32_t parentBlockX;

  /**
   * The Y-dimension of the parent block.
   */
  uint32_t parentBlockY;

  /**
   * The Z-dimension of the parent block.
   */
  uint32_t parentBlockZ;

#ifdef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;
} LWpti_ActivityCdpKernel;

/**
 * \brief The activity record for a preemption of a CDP kernel.
 *
 * This activity record represents a preemption of a CDP kernel.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_PREEMPTION
   */
  LWpti_ActivityKind kind;

  /**
  * kind of the preemption
  */
  LWpti_ActivityPreemptionKind preemptionKind;

  /**
   * The timestamp of the preemption, in ns. A value of 0 indicates
   * that timestamp information could not be collected for the
   * preemption.
   */
  uint64_t timestamp;

  /**
  * The grid-id of the block that is preempted
  */
  int64_t gridId;

  /**
   * The X-dimension of the block that is preempted
   */
  uint32_t blockX;

  /**
   * The Y-dimension of the block that is preempted
   */
  uint32_t blockY;

  /**
   * The Z-dimension of the block that is preempted
   */
  uint32_t blockZ;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} LWpti_ActivityPreemption;

/**
 * \brief The activity record for a driver or runtime API invocation.
 *
 * This activity record represents an invocation of a driver or
 * runtime API (LWPTI_ACTIVITY_KIND_DRIVER and
 * LWPTI_ACTIVITY_KIND_RUNTIME).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_DRIVER,
   * LWPTI_ACTIVITY_KIND_RUNTIME, or LWPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API.
   */
  LWpti_ActivityKind kind;

  /**
   * The ID of the driver or runtime function.
   */
  LWpti_CallbackId cbid;

  /**
   * The start timestamp for the function, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the function.
   */
  uint64_t start;

  /**
   * The end timestamp for the function, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the function.
   */
  uint64_t end;

  /**
   * The ID of the process where the driver or runtime LWCA function
   * is exelwting.
   */
  uint32_t processId;

  /**
   * The ID of the thread where the driver or runtime LWCA function is
   * exelwting.
   */
  uint32_t threadId;

  /**
   * The correlation ID of the driver or runtime LWCA function. Each
   * function invocation is assigned a unique correlation ID that is
   * identical to the correlation ID in the memcpy, memset, or kernel
   * activity record that is associated with this function.
   */
  uint32_t correlationId;

  /**
   * The return value for the function. For a LWCA driver function
   * with will be a LWresult value, and for a LWCA runtime function
   * this will be a lwdaError_t value.
   */
  uint32_t returlwalue;
} LWpti_ActivityAPI;

/**
 * \brief The activity record for a LWPTI event.
 *
 * This activity record represents a LWPTI event value
 * (LWPTI_ACTIVITY_KIND_EVENT). This activity record kind is not
 * produced by the activity API but is included for completeness and
 * ease-of-use. Profile frameworks built on top of LWPTI that collect
 * event data may choose to use this type to store the collected event
 * data.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_EVENT.
   */
  LWpti_ActivityKind kind;

  /**
   * The event ID.
   */
  LWpti_EventID id;

  /**
   * The event value.
   */
  uint64_t value;

  /**
   * The event domain ID.
   */
  LWpti_EventDomainID domain;

  /**
   * The correlation ID of the event. Use of this ID is user-defined,
   * but typically this ID value will equal the correlation ID of the
   * kernel for which the event was gathered.
   */
  uint32_t correlationId;
} LWpti_ActivityEvent;

/**
 * \brief The activity record for a LWPTI event with instance
 * information.
 *
 * This activity record represents the a LWPTI event value for a
 * specific event domain instance
 * (LWPTI_ACTIVITY_KIND_EVENT_INSTANCE). This activity record kind is
 * not produced by the activity API but is included for completeness
 * and ease-of-use. Profile frameworks built on top of LWPTI that
 * collect event data may choose to use this type to store the
 * collected event data. This activity record should be used when
 * event domain instance information needs to be associated with the
 * event.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be
   * LWPTI_ACTIVITY_KIND_EVENT_INSTANCE.
   */
  LWpti_ActivityKind kind;

  /**
   * The event ID.
   */
  LWpti_EventID id;

  /**
   * The event domain ID.
   */
  LWpti_EventDomainID domain;

  /**
   * The event domain instance.
   */
  uint32_t instance;

  /**
   * The event value.
   */
  uint64_t value;

  /**
   * The correlation ID of the event. Use of this ID is user-defined,
   * but typically this ID value will equal the correlation ID of the
   * kernel for which the event was gathered.
   */
  uint32_t correlationId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} LWpti_ActivityEventInstance;

/**
 * \brief The activity record for a LWPTI metric.
 *
 * This activity record represents the collection of a LWPTI metric
 * value (LWPTI_ACTIVITY_KIND_METRIC). This activity record kind is not
 * produced by the activity API but is included for completeness and
 * ease-of-use. Profile frameworks built on top of LWPTI that collect
 * metric data may choose to use this type to store the collected metric
 * data.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_METRIC.
   */
  LWpti_ActivityKind kind;

  /**
   * The metric ID.
   */
  LWpti_MetricID id;

  /**
   * The metric value.
   */
  LWpti_MetricValue value;

  /**
   * The correlation ID of the metric. Use of this ID is user-defined,
   * but typically this ID value will equal the correlation ID of the
   * kernel for which the metric was gathered.
   */
  uint32_t correlationId;

  /**
   * The properties of this metric. \see LWpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t pad[3];
} LWpti_ActivityMetric;

/**
 * \brief The activity record for a LWPTI metric with instance
 * information.
 *
 * This activity record represents a LWPTI metric value
 * for a specific metric domain instance
 * (LWPTI_ACTIVITY_KIND_METRIC_INSTANCE).  This activity record kind
 * is not produced by the activity API but is included for
 * completeness and ease-of-use. Profile frameworks built on top of
 * LWPTI that collect metric data may choose to use this type to store
 * the collected metric data. This activity record should be used when
 * metric domain instance information needs to be associated with the
 * metric.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be
   * LWPTI_ACTIVITY_KIND_METRIC_INSTANCE.
   */
  LWpti_ActivityKind kind;

  /**
   * The metric ID.
   */
  LWpti_MetricID id;

  /**
   * The metric value.
   */
  LWpti_MetricValue value;

  /**
   * The metric domain instance.
   */
  uint32_t instance;

  /**
   * The correlation ID of the metric. Use of this ID is user-defined,
   * but typically this ID value will equal the correlation ID of the
   * kernel for which the metric was gathered.
   */
  uint32_t correlationId;

  /**
   * The properties of this metric. \see LWpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t pad[7];
} LWpti_ActivityMetricInstance;

/**
 * \brief The activity record for source locator.
 *
 * This activity record represents a source locator
 * (LWPTI_ACTIVITY_KIND_SOURCE_LOCATOR).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_SOURCE_LOCATOR.
   */
  LWpti_ActivityKind kind;

  /**
   * The ID for the source path, will be used in all the source level
   * results.
   */
  uint32_t id;

  /**
   * The line number in the source .
   */
  uint32_t lineNumber;

#ifdef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The path for the file.
   */
  const char *fileName;
} LWpti_ActivitySourceLocator;

/**
 * \brief The activity record for source-level global
 * access. (deprecated)
 *
 * This activity records the locations of the global
 * accesses in the source (LWPTI_ACTIVITY_KIND_GLOBAL_ACCESS).
 * Global access activities are now reported using the
 * LWpti_ActivityGlobalAccess3 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_GLOBAL_ACCESS.
   */
  LWpti_ActivityKind kind;

  /**
   * The properties of this global access.
   */
  LWpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The pc offset for the access.
   */
  uint32_t pcOffset;

  /**
   * The number of times this instruction was exelwted per warp. It will be incremented 
   * when at least one of thread among warp is active with predicate and condition code 
   * evaluating to true.
   */
  uint32_t exelwted;

  /**
   * This increments each time when this instruction is exelwted by number
   * of threads that exelwted this instruction with predicate and condition code evaluating to true.
   */
  uint64_t threadsExelwted;

  /**
   * The total number of 32 bytes transactions to L2 cache generated by this access
   */
  uint64_t l2_transactions;
} LWpti_ActivityGlobalAccess;

/**
 * \brief The activity record for source-level global
 * access. (deprecated in LWCA 9.0)
 *
 * This activity records the locations of the global
 * accesses in the source (LWPTI_ACTIVITY_KIND_GLOBAL_ACCESS).
 * Global access activities are now reported using the
 * LWpti_ActivityGlobalAccess3 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_GLOBAL_ACCESS.
   */
  LWpti_ActivityKind kind;

  /**
   * The properties of this global access.
   */
  LWpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

 /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the access.
   */
  uint32_t pcOffset;

  /**
   * This increments each time when this instruction is exelwted by number
   * of threads that exelwted this instruction with predicate and condition code evaluating to true.
   */
  uint64_t threadsExelwted;

  /**
   * The total number of 32 bytes transactions to L2 cache generated by this access
   */
  uint64_t l2_transactions;

  /**
   * The minimum number of L2 transactions possible based on the access pattern.
   */
  uint64_t theoreticalL2Transactions;

  /**
   * The number of times this instruction was exelwted per warp. It will be incremented 
   * when at least one of thread among warp is active with predicate and condition code 
   * evaluating to true.
   */
  uint32_t exelwted;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} LWpti_ActivityGlobalAccess2;

/**
 * \brief The activity record for source-level global
 * access.
 *
 * This activity records the locations of the global
 * accesses in the source (LWPTI_ACTIVITY_KIND_GLOBAL_ACCESS).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_GLOBAL_ACCESS.
   */
  LWpti_ActivityKind kind;

  /**
   * The properties of this global access.
   */
  LWpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

 /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The number of times this instruction was exelwted per warp. It will be incremented 
   * when at least one of thread among warp is active with predicate and condition code 
   * evaluating to true.
   */
  uint32_t exelwted;

  /**
   * The pc offset for the access.
   */
  uint64_t pcOffset;

  /**
   * This increments each time when this instruction is exelwted by number of
   * threads that exelwted this instruction with predicate and condition code
   * evaluating to true.
   */
  uint64_t threadsExelwted;

  /**
   * The total number of 32 bytes transactions to L2 cache generated by this
     access
   */
  uint64_t l2_transactions;

  /**
   * The minimum number of L2 transactions possible based on the access pattern.
   */
  uint64_t theoreticalL2Transactions;
} LWpti_ActivityGlobalAccess3;

/**
 * \brief The activity record for source level result
 * branch. (deprecated)
 *
 * This activity record the locations of the branches in the
 * source (LWPTI_ACTIVITY_KIND_BRANCH).
 * Branch activities are now reported using the
 * LWpti_ActivityBranch2 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_BRANCH.
   */
  LWpti_ActivityKind kind;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The pc offset for the branch.
   */
  uint32_t pcOffset;

  /**
   * The number of times this instruction was exelwted per warp. It will be incremented 
   * regardless of predicate or condition code.
   */
  uint32_t exelwted;

  /**
   * Number of times this branch diverged
   */
  uint32_t diverged;

  /**
   * This increments each time when this instruction is exelwted by number
   * of threads that exelwted this instruction
   */
  uint64_t threadsExelwted;
} LWpti_ActivityBranch;

/**
 * \brief The activity record for source level result
 * branch.
 *
 * This activity record the locations of the branches in the
 * source (LWPTI_ACTIVITY_KIND_BRANCH).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_BRANCH.
   */
  LWpti_ActivityKind kind;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

 /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the branch.
   */
  uint32_t pcOffset;

  /**
   * Number of times this branch diverged
   */
  uint32_t diverged;

  /**
   * This increments each time when this instruction is exelwted by number
   * of threads that exelwted this instruction
   */
  uint64_t threadsExelwted;

  /**
   * The number of times this instruction was exelwted per warp. It will be incremented 
   * regardless of predicate or condition code.
   */
  uint32_t exelwted;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} LWpti_ActivityBranch2;


/**
 * \brief The activity record for a device. (deprecated)
 *
 * This activity record represents information about a GPU device
 * (LWPTI_ACTIVITY_KIND_DEVICE).
 * Device activity is now reported using the
 * LWpti_ActivityDevice2 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_DEVICE.
   */
  LWpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see LWpti_ActivityFlag
   */
  LWpti_ActivityFlag flags;

  /**
   * The global memory bandwidth available on the device, in
   * kBytes/sec.
   */
  uint64_t globalMemoryBandwidth;

  /**
   * The amount of global memory on the device, in bytes.
   */
  uint64_t globalMemorySize;

  /**
   * The amount of constant memory on the device, in bytes.
   */
  uint32_t constantMemorySize;

  /**
   * The size of the L2 cache on the device, in bytes.
   */
  uint32_t l2CacheSize;

  /**
   * The number of threads per warp on the device.
   */
  uint32_t numThreadsPerWarp;

  /**
   * The core clock rate of the device, in kHz.
   */
  uint32_t coreClockRate;

  /**
   * Number of memory copy engines on the device.
   */
  uint32_t numMemcpyEngines;

  /**
   * Number of multiprocessors on the device.
   */
  uint32_t numMultiprocessors;

  /**
   * The maximum "instructions per cycle" possible on each device
   * multiprocessor.
   */
  uint32_t maxIPC;

  /**
   * Maximum number of warps that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxWarpsPerMultiprocessor;

  /**
   * Maximum number of blocks that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxBlocksPerMultiprocessor;

  /**
   * Maximum number of registers that can be allocated to a block.
   */
  uint32_t maxRegistersPerBlock;

  /**
   * Maximum amount of shared memory that can be assigned to a block,
   * in bytes.
   */
  uint32_t maxSharedMemoryPerBlock;

  /**
   * Maximum number of threads allowed in a block.
   */
  uint32_t maxThreadsPerBlock;

  /**
   * Maximum allowed X dimension for a block.
   */
  uint32_t maxBlockDimX;

  /**
   * Maximum allowed Y dimension for a block.
   */
  uint32_t maxBlockDimY;

  /**
   * Maximum allowed Z dimension for a block.
   */
  uint32_t maxBlockDimZ;

  /**
   * Maximum allowed X dimension for a grid.
   */
  uint32_t maxGridDimX;

  /**
   * Maximum allowed Y dimension for a grid.
   */
  uint32_t maxGridDimY;

  /**
   * Maximum allowed Z dimension for a grid.
   */
  uint32_t maxGridDimZ;

  /**
   * Compute capability for the device, major number.
   */
  uint32_t computeCapabilityMajor;

  /**
   * Compute capability for the device, minor number.
   */
  uint32_t computeCapabilityMinor;

  /**
   * The device ID.
   */
  uint32_t id;

#ifdef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The device name. This name is shared across all activity records
   * representing instances of the device, and so should not be
   * modified.
   */
  const char *name;
} LWpti_ActivityDevice;

/**
 * \brief The activity record for a device. (LWCA 7.0 onwards)
 *
 * This activity record represents information about a GPU device
 * (LWPTI_ACTIVITY_KIND_DEVICE).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_DEVICE.
   */
  LWpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see LWpti_ActivityFlag
   */
  LWpti_ActivityFlag flags;

  /**
   * The global memory bandwidth available on the device, in
   * kBytes/sec.
   */
  uint64_t globalMemoryBandwidth;

  /**
   * The amount of global memory on the device, in bytes.
   */
  uint64_t globalMemorySize;

  /**
   * The amount of constant memory on the device, in bytes.
   */
  uint32_t constantMemorySize;

  /**
   * The size of the L2 cache on the device, in bytes.
   */
  uint32_t l2CacheSize;

  /**
   * The number of threads per warp on the device.
   */
  uint32_t numThreadsPerWarp;

  /**
   * The core clock rate of the device, in kHz.
   */
  uint32_t coreClockRate;

  /**
   * Number of memory copy engines on the device.
   */
  uint32_t numMemcpyEngines;

  /**
   * Number of multiprocessors on the device.
   */
  uint32_t numMultiprocessors;

  /**
   * The maximum "instructions per cycle" possible on each device
   * multiprocessor.
   */
  uint32_t maxIPC;

  /**
   * Maximum number of warps that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxWarpsPerMultiprocessor;

  /**
   * Maximum number of blocks that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxBlocksPerMultiprocessor;

  /**
   * Maximum amount of shared memory available per multiprocessor, in bytes.
   */
  uint32_t maxSharedMemoryPerMultiprocessor;

  /**
   * Maximum number of 32-bit registers available per multiprocessor.
   */
  uint32_t maxRegistersPerMultiprocessor;

  /**
   * Maximum number of registers that can be allocated to a block.
   */
  uint32_t maxRegistersPerBlock;

  /**
   * Maximum amount of shared memory that can be assigned to a block,
   * in bytes.
   */
  uint32_t maxSharedMemoryPerBlock;

  /**
   * Maximum number of threads allowed in a block.
   */
  uint32_t maxThreadsPerBlock;

  /**
   * Maximum allowed X dimension for a block.
   */
  uint32_t maxBlockDimX;

  /**
   * Maximum allowed Y dimension for a block.
   */
  uint32_t maxBlockDimY;

  /**
   * Maximum allowed Z dimension for a block.
   */
  uint32_t maxBlockDimZ;

  /**
   * Maximum allowed X dimension for a grid.
   */
  uint32_t maxGridDimX;

  /**
   * Maximum allowed Y dimension for a grid.
   */
  uint32_t maxGridDimY;

  /**
   * Maximum allowed Z dimension for a grid.
   */
  uint32_t maxGridDimZ;

  /**
   * Compute capability for the device, major number.
   */
  uint32_t computeCapabilityMajor;

  /**
   * Compute capability for the device, minor number.
   */
  uint32_t computeCapabilityMinor;

  /**
   * The device ID.
   */
  uint32_t id;

  /**
   * ECC enabled flag for device
   */
  uint32_t eccEnabled;

  /**
   * The device UUID. This value is the globally unique immutable
   * alphanumeric identifier of the device.
   */
  LWuuid uuid;

#ifndef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The device name. This name is shared across all activity records
   * representing instances of the device, and so should not be
   * modified.
   */
  const char *name;
} LWpti_ActivityDevice2;

/**
 * \brief The activity record for a device attribute.
 *
 * This activity record represents information about a GPU device:
 * either a LWpti_DeviceAttribute or LWdevice_attribute value
 * (LWPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be
   * LWPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE.
   */
  LWpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see LWpti_ActivityFlag
   */
  LWpti_ActivityFlag flags;

  /**
   * The ID of the device that this attribute applies to.
   */
  uint32_t deviceId;

  /**
   * The attribute, either a LWpti_DeviceAttribute or
   * LWdevice_attribute. Flag
   * LWPTI_ACTIVITY_FLAG_DEVICE_ATTRIBUTE_LWDEVICE is used to indicate
   * what kind of attribute this is. If
   * LWPTI_ACTIVITY_FLAG_DEVICE_ATTRIBUTE_LWDEVICE is 1 then
   * LWdevice_attribute field is value, otherwise
   * LWpti_DeviceAttribute field is valid.
   */
  union {
    LWdevice_attribute lw;
    LWpti_DeviceAttribute lwpti;
  } attribute;

  /**
   * The value for the attribute. See LWpti_DeviceAttribute and
   * LWdevice_attribute for the type of the value for a given
   * attribute.
   */
  union {
    double vDouble;
    uint32_t vUint32;
    uint64_t vUint64;
    int32_t vInt32;
    int64_t vInt64;
  } value;
} LWpti_ActivityDeviceAttribute;

/**
 * \brief The activity record for a context.
 *
 * This activity record represents information about a context
 * (LWPTI_ACTIVITY_KIND_CONTEXT).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_CONTEXT.
   */
  LWpti_ActivityKind kind;

  /**
   * The context ID.
   */
  uint32_t contextId;

  /**
   * The device ID.
   */
  uint32_t deviceId;

  /**
   * The compute API kind. \see LWpti_ActivityComputeApiKind
   */
  uint16_t computeApiKind;

  /**
   * The ID for the NULL stream in this context
   */
  uint16_t nullStreamId;
} LWpti_ActivityContext;

/**
 * \brief The activity record providing a name.
 *
 * This activity record provides a name for a device, context, thread,
 * etc.  (LWPTI_ACTIVITY_KIND_NAME).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_NAME.
   */
  LWpti_ActivityKind kind;

  /**
   * The kind of activity object being named.
   */
  LWpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object. 'objectKind' indicates
   * which ID is valid for this record.
   */
  LWpti_ActivityObjectKindId objectId;

#ifdef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The name.
   */
  const char *name;

} LWpti_ActivityName;

/**
 * \brief The activity record providing a marker which is an
 * instantaneous point in time. (deprecated in LWCA 8.0)
 *
 * The marker is specified with a descriptive name and unique id
 * (LWPTI_ACTIVITY_KIND_MARKER).
 * Marker activity is now reported using the
 * LWpti_ActivityMarker2 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_MARKER.
   */
  LWpti_ActivityKind kind;

  /**
   * The flags associated with the marker. \see LWpti_ActivityFlag
   */
  LWpti_ActivityFlag flags;

  /**
   * The timestamp for the marker, in ns. A value of 0 indicates that
   * timestamp information could not be collected for the marker.
   */
  uint64_t timestamp;

  /**
   * The marker ID.
   */
  uint32_t id;

  /**
   * The kind of activity object associated with this marker.
   */
  LWpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object associated with this
   * marker. 'objectKind' indicates which ID is valid for this record.
   */
  LWpti_ActivityObjectKindId objectId;

#ifdef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The marker name for an instantaneous or start marker. This will
   * be NULL for an end marker.
   */
  const char *name;

} LWpti_ActivityMarker;

/**
 * \brief The activity record providing a marker which is an
 * instantaneous point in time.
 *
 * The marker is specified with a descriptive name and unique id
 * (LWPTI_ACTIVITY_KIND_MARKER).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_MARKER.
   */
  LWpti_ActivityKind kind;

  /**
   * The flags associated with the marker. \see LWpti_ActivityFlag
   */
  LWpti_ActivityFlag flags;

  /**
   * The timestamp for the marker, in ns. A value of 0 indicates that
   * timestamp information could not be collected for the marker.
   */
  uint64_t timestamp;

  /**
   * The marker ID.
   */
  uint32_t id;

  /**
   * The kind of activity object associated with this marker.
   */
  LWpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object associated with this
   * marker. 'objectKind' indicates which ID is valid for this record.
   */
  LWpti_ActivityObjectKindId objectId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;


  /**
   * The marker name for an instantaneous or start marker. This will
   * be NULL for an end marker.
   */
  const char *name;

  /**
   * The name of the domain to which this marker belongs to.
   * This will be NULL for default domain.
   */
  const char *domain;

} LWpti_ActivityMarker2;

/**
 * \brief The activity record providing detailed information for a marker.
 *
 * The marker data contains color, payload, and category.
 * (LWPTI_ACTIVITY_KIND_MARKER_DATA).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be
   * LWPTI_ACTIVITY_KIND_MARKER_DATA.
   */
  LWpti_ActivityKind kind;

  /**
   * The flags associated with the marker. \see LWpti_ActivityFlag
   */
  LWpti_ActivityFlag flags;

  /**
   * The marker ID.
   */
  uint32_t id;

  /**
   * Defines the payload format for the value associated with the marker.
   */
  LWpti_MetricValueKind payloadKind;

  /**
   * The payload value.
   */
  LWpti_MetricValue payload;

  /**
   * The color for the marker.
   */
  uint32_t color;

  /**
   * The category for the marker.
   */
  uint32_t category;

} LWpti_ActivityMarkerData;

/**
 * \brief The activity record for LWPTI and driver overheads.
 *
 * This activity record provides LWPTI and driver overhead information
 * (LWPTI_ACTIVITY_OVERHEAD).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_OVERHEAD.
   */
  LWpti_ActivityKind kind;

  /**
   * The kind of overhead, LWPTI, DRIVER, COMPILER etc.
   */
  LWpti_ActivityOverheadKind overheadKind;

  /**
   * The kind of activity object that the overhead is associated with.
   */
  LWpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object. 'objectKind' indicates
   * which ID is valid for this record.
   */
  LWpti_ActivityObjectKindId objectId;

  /**
   * The start timestamp for the overhead, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the overhead.
   */
  uint64_t start;

  /**
   * The end timestamp for the overhead, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the overhead.
   */
  uint64_t end;
} LWpti_ActivityOverhead;

/**
 * \brief The activity record for LWPTI elwironmental data.
 *
 * This activity record provides LWPTI elwironmental data, include
 * power, clocks, and thermals.  This information is sampled at
 * various rates and returned in this activity record.  The consumer
 * of the record needs to check the elwironmentKind field to figure
 * out what kind of elwironmental record this is.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_ELWIRONMENT.
   */
  LWpti_ActivityKind kind;

  /**
   * The ID of the device
   */
  uint32_t deviceId;

  /**
   * The timestamp when this sample was retrieved, in ns. A value of 0
   * indicates that timestamp information could not be collected for
   * the marker.
   */
  uint64_t timestamp;

  /**
   * The kind of data reported in this record.
   */
  LWpti_ActivityElwironmentKind elwironmentKind;

  union {
    /**
     * Data returned for LWPTI_ACTIVITY_ELWIRONMENT_SPEED environment
     * kind.
     */
    struct {
      /**
       * The SM frequency in MHz
       */
      uint32_t smClock;

      /**
       * The memory frequency in MHz
       */
      uint32_t memoryClock;

      /**
       * The PCIe link generation.
       */
      uint32_t pcieLinkGen;

      /**
       * The PCIe link width.
       */
      uint32_t pcieLinkWidth;

      /**
       * The clocks throttle reasons.
       */
      LWpti_ElwironmentClocksThrottleReason clocksThrottleReasons;
    } speed;
    /**
     * Data returned for LWPTI_ACTIVITY_ELWIRONMENT_TEMPERATURE
     * environment kind.
     */
    struct {
      /**
       * The GPU temperature in degrees C.
       */
      uint32_t gpuTemperature;
    } temperature;
    /**
     * Data returned for LWPTI_ACTIVITY_ELWIRONMENT_POWER environment
     * kind.
     */
    struct {
      /**
       * The power in milliwatts consumed by GPU and associated
       * cirlwitry.
       */
      uint32_t power;

      /**
       * The power in milliwatts that will trigger power management
       * algorithm.
       */
      uint32_t powerLimit;
    } power;
    /**
     * Data returned for LWPTI_ACTIVITY_ELWIRONMENT_COOLING
     * environment kind.
     */
    struct {
      /**
       * The fan speed as percentage of maximum.
       */
      uint32_t fanSpeed;
    } cooling;
  } data;
} LWpti_ActivityElwironment;

/**
 * \brief The activity record for source-level instruction exelwtion.
 *
 * This activity records result for source level instruction exelwtion.
 * (LWPTI_ACTIVITY_KIND_INSTRUCTION_EXELWTION).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_INSTRUCTION_EXELWTION.
   */
  LWpti_ActivityKind kind;

  /**
   * The properties of this instruction exelwtion.
   */
  LWpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

 /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the instruction.
   */
  uint32_t pcOffset;

  /**
   * This increments each time when this instruction is exelwted by number
   * of threads that exelwted this instruction, regardless of predicate or condition code.
   */
  uint64_t threadsExelwted;

  /**
   * This increments each time when this instruction is exelwted by number
   * of threads that exelwted this instruction with predicate and condition code evaluating to true.
   */
  uint64_t notPredOffThreadsExelwted;

  /**
   * The number of times this instruction was exelwted per warp. It will be incremented 
   * regardless of predicate or condition code.
   */
  uint32_t exelwted;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} LWpti_ActivityInstructionExelwtion;

/**
 * \brief The activity record for PC sampling. (deprecated in LWCA 8.0)
 *
 * This activity records information obtained by sampling PC
 * (LWPTI_ACTIVITY_KIND_PC_SAMPLING).
 * PC sampling activities are now reported using the
 * LWpti_ActivityPCSampling2 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_PC_SAMPLING.
   */
  LWpti_ActivityKind kind;

  /**
   * The properties of this instruction.
   */
  LWpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

 /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the instruction.
   */
  uint32_t pcOffset;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * The same PC can be sampled with different stall reasons.
   */
  uint32_t samples;

  /**
   * Current stall reason. Includes one of the reasons from
   * \ref LWpti_ActivityPCSamplingStallReason
   */
  LWpti_ActivityPCSamplingStallReason stallReason;
} LWpti_ActivityPCSampling;

/**
 * \brief The activity record for PC sampling. (deprecated in LWCA 9.0)
 *
 * This activity records information obtained by sampling PC
 * (LWPTI_ACTIVITY_KIND_PC_SAMPLING).
 * PC sampling activities are now reported using the
 * LWpti_ActivityPCSampling3 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_PC_SAMPLING.
   */
  LWpti_ActivityKind kind;

  /**
   * The properties of this instruction.
   */
  LWpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

 /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the instruction.
   */
  uint32_t pcOffset;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * These samples indicate that no instruction was issued in that cycle from
   * the warp scheduler from where the warp was sampled.
   * Field is valid for devices with compute capability 6.0 and higher
   */
  uint32_t latencySamples;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * The same PC can be sampled with different stall reasons. The count includes
   * latencySamples.
   */
  uint32_t samples;

  /**
   * Current stall reason. Includes one of the reasons from
   * \ref LWpti_ActivityPCSamplingStallReason
   */
  LWpti_ActivityPCSamplingStallReason stallReason;

  uint32_t pad;
} LWpti_ActivityPCSampling2;

/**
 * \brief The activity record for PC sampling.
 *
 * This activity records information obtained by sampling PC
 * (LWPTI_ACTIVITY_KIND_PC_SAMPLING).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_PC_SAMPLING.
   */
  LWpti_ActivityKind kind;

  /**
   * The properties of this instruction.
   */
  LWpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

 /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * These samples indicate that no instruction was issued in that cycle from
   * the warp scheduler from where the warp was sampled.
   * Field is valid for devices with compute capability 6.0 and higher
   */
  uint32_t latencySamples;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * The same PC can be sampled with different stall reasons. The count includes
   * latencySamples.
   */
  uint32_t samples;

  /**
   * Current stall reason. Includes one of the reasons from
   * \ref LWpti_ActivityPCSamplingStallReason
   */
  LWpti_ActivityPCSamplingStallReason stallReason;

    /**
   * The pc offset for the instruction.
   */
  uint64_t pcOffset;
} LWpti_ActivityPCSampling3;

/**
 * \brief The activity record for record status for PC sampling.
 *
 * This activity records information obtained by sampling PC
 * (LWPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO.
   */
  LWpti_ActivityKind kind;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * Number of times the PC was sampled for this kernel instance including all
   * dropped samples.
   */
  uint64_t totalSamples;

  /**
   * Number of samples that were dropped by hardware due to backpressure/overflow.
   */
  uint64_t droppedSamples;
  /**
   * Sampling period in terms of number of cycles .
   */
  uint64_t samplingPeriodInCycles;
} LWpti_ActivityPCSamplingRecordInfo;

/**
 * \brief The activity record for Unified Memory counters (deprecated in LWCA 7.0)
 *
 * This activity record represents a Unified Memory counter
 * (LWPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
   */
  LWpti_ActivityKind kind;

  /**
   * The Unified Memory counter kind. See /ref LWpti_ActivityUnifiedMemoryCounterKind
   */
  LWpti_ActivityUnifiedMemoryCounterKind counterKind;

  /**
   * Scope of the Unified Memory counter. See /ref LWpti_ActivityUnifiedMemoryCounterScope
   */
  LWpti_ActivityUnifiedMemoryCounterScope scope;

  /**
   * The ID of the device ilwolved in the memory transfer operation.
   * It is not relevant if the scope of the counter is global (all devices).
   */
  uint32_t deviceId;

  /**
   * Value of the counter
   *
   */
  uint64_t value;

  /**
   * The timestamp when this sample was retrieved, in ns. A value of 0
   * indicates that timestamp information could not be collected
   */
  uint64_t timestamp;

  /**
   * The ID of the process to which this record belongs to. In case of
   * global scope, processId is undefined.
   */
  uint32_t processId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} LWpti_ActivityUnifiedMemoryCounter;

/**
 * \brief The activity record for Unified Memory counters (LWCA 7.0 and beyond)
 *
 * This activity record represents a Unified Memory counter
 * (LWPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
   */
  LWpti_ActivityKind kind;

  /**
   * The Unified Memory counter kind
   */
  LWpti_ActivityUnifiedMemoryCounterKind counterKind;

  /**
   * Value of the counter
   * For counterKind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD,
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH,
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THREASHING and
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP, it is the size of the
   * memory region in bytes.
   * For counterKind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT, it
   * is the number of page fault groups for the same page.
   * For counterKind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
   * it is the program counter for the instruction that caused fault.
   */
  uint64_t value;

  /**
   * The start timestamp of the counter, in ns.
   * For counterKind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD and
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH, timestamp is 
   * captured when activity starts on GPU. 
   * For counterKind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT and 
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT, timestamp is 
   * captured when LWCA driver started processing the fault.
   * For counterKind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING, timestamp
   * is captured when LWCA driver detected thrashing of memory region. 
   * For counterKind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING, 
   * timestamp is captured when throttling opeeration was started by LWCA driver.
   * For counterKind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP,
   * timestamp is captured when LWCA driver has pushed all required operations
   * to the processor specified by dstId.
   */
  uint64_t start;

  /**
   * The end timestamp of the counter, in ns.
   * Ignore this field if counterKind is 
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT or
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING or 
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP.
   * For counterKind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD and
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH, timestamp is 
   * captured when activity finishes on GPU. 
   * For counterKind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT, timestamp is 
   * captured when LWCA driver queues the replay of faulting memory accesses on the GPU
   * For counterKind LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING, timestamp
   * is captured when throttling operation was finished by LWCA driver
   */
  uint64_t end;

  /**
   * This is the virtual base address of the page/s being transferred. For cpu and
   * gpu faults, the virtual address for the page that faulted.
   */
  uint64_t address;

  /**
   * The ID of the source CPU/device ilwolved in the memory transfer, page fault, thrashing,
   * throttling or remote map operation. For counterKind 
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING, it is a bitwise ORing of the
   * device IDs fighting for the memory region. Ignore this field if counterKind is 
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT
   */
  uint32_t srcId;

  /**
   * The ID of the destination CPU/device ilwolved in the memory transfer or remote map
   * operation. Ignore this field if counterKind is 
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT or
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT or
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING or 
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING
   */
  uint32_t dstId;

  /**
   * The ID of the stream causing the transfer.
   * This value of this field is invalid.
   */
  uint32_t streamId;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The flags associated with this record. See enums \ref LWpti_ActivityUnifiedMemoryAccessType
   * if counterKind is LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT
   * and \ref LWpti_ActivityUnifiedMemoryMigrationCause if counterKind is 
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD or 
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD
   * and \ref LWpti_ActivityUnifiedMemoryRemoteMapCause if counterKind is 
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP and \ref LWpti_ActivityFlag 
   * if counterKind is LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING or
   * LWPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING
   */
  uint32_t flags;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} LWpti_ActivityUnifiedMemoryCounter2;

/**
 * \brief The activity record for global/device functions.
 *
 * This activity records function name and corresponding module
 * information.
 * (LWPTI_ACTIVITY_KIND_FUNCTION).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_FUNCTION.
   */
  LWpti_ActivityKind kind;

  /**
  * ID to uniquely identify the record
  */
  uint32_t id;

  /**
   * The ID of the context where the function is launched.
   */
  uint32_t contextId;

  /**
   * The module ID in which this global/device function is present.
   */
  uint32_t moduleId;

  /**
   * The function's unique symbol index in the module.
   */
  uint32_t functionIndex;

#ifdef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The name of the function. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;
} LWpti_ActivityFunction;

/**
 * \brief The activity record for a LWCA module.
 *
 * This activity record represents a LWCA module
 * (LWPTI_ACTIVITY_KIND_MODULE). This activity record kind is not
 * produced by the activity API but is included for completeness and
 * ease-of-use. Profile frameworks built on top of LWPTI that collect
 * module data from the module callback may choose to use this type to
 * store the collected module data.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_MODULE.
   */
  LWpti_ActivityKind kind;

  /**
   * The ID of the context where the module is loaded.
   */
  uint32_t contextId;

  /**
   * The module ID.
   */
  uint32_t id;

  /**
   * The lwbin size.
   */
  uint32_t lwbinSize;

#ifndef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The pointer to lwbin.
   */
  const void *lwbin;
} LWpti_ActivityModule;

/**
 * \brief The activity record for source-level shared
 * access.
 *
 * This activity records the locations of the shared
 * accesses in the source
 * (LWPTI_ACTIVITY_KIND_SHARED_ACCESS).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_SHARED_ACCESS.
   */
  LWpti_ActivityKind kind;

  /**
   * The properties of this shared access.
   */
  LWpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

 /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the access.
   */
  uint32_t pcOffset;

  /**
   * This increments each time when this instruction is exelwted by number
   * of threads that exelwted this instruction with predicate and condition code evaluating to true.
   */
  uint64_t threadsExelwted;

  /**
   * The total number of shared memory transactions generated by this access
   */
  uint64_t sharedTransactions;

  /**
   * The minimum number of shared memory transactions possible based on the access pattern.
   */
  uint64_t theoreticalSharedTransactions;

  /**
   * The number of times this instruction was exelwted per warp. It will be incremented 
   * when at least one of thread among warp is active with predicate and condition code 
   * evaluating to true.
   */
  uint32_t exelwted;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} LWpti_ActivitySharedAccess;

/**
 * \brief The activity record for LWCA event.
 *
 * This activity is used to track recorded events.
 * (LWPTI_ACTIVITY_KIND_LWDA_EVENT).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_LWDA_EVENT.
   */
  LWpti_ActivityKind kind;

  /**
   * The correlation ID of the API to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The ID of the context where the event was recorded.
   */
  uint32_t contextId;

  /**
   * The compute stream where the event was recorded.
   */
  uint32_t streamId;

  /**
   * A unique event ID to identify the event record.
   */
  uint32_t eventId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} LWpti_ActivityLwdaEvent;

/**
 * \brief The activity record for LWCA stream.
 *
 * This activity is used to track created streams.
 * (LWPTI_ACTIVITY_KIND_STREAM).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_STREAM.
   */
  LWpti_ActivityKind kind;
  /**
   * The ID of the context where the stream was created.
   */
  uint32_t contextId;

  /**
   * A unique stream ID to identify the stream.
   */
  uint32_t streamId;

  /**
   * The clamped priority for the stream.
   */
  uint32_t priority;

  /**
   * Flags associated with the stream.
   */
  LWpti_ActivityStreamFlag flag;

  /**
   * The correlation ID of the API to which this result is associated.
   */
  uint32_t correlationId;
} LWpti_ActivityStream;

/**
 * \brief The activity record for synchronization management.
 *
 * This activity is used to track various LWCA synchronization APIs.
 * (LWPTI_ACTIVITY_KIND_SYNCHRONIZATION).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_SYNCHRONIZATION.
   */
  LWpti_ActivityKind kind;

  /**
   * The type of record.
   */
  LWpti_ActivitySynchronizationType type;

  /**
   * The start timestamp for the function, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the function.
   */
  uint64_t start;

  /**
   * The end timestamp for the function, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the function.
   */
  uint64_t end;

  /**
   * The correlation ID of the API to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The ID of the context for which the synchronization API is called.
   * In case of context synchronization API it is the context id for which the API is called.
   * In case of stream/event synchronization it is the ID of the context where the stream/event was created.
   */
  uint32_t contextId;

  /**
   * The compute stream for which the synchronization API is called.
   * A LWPTI_SYNCHRONIZATION_ILWALID_VALUE value indicate the field is not applicable for this record.
   * Not valid for lwCtxSynchronize, lwEventSynchronize.
   */
  uint32_t streamId;

  /**
   * The event ID for which the synchronization API is called.
   * A LWPTI_SYNCHRONIZATION_ILWALID_VALUE value indicate the field is not applicable for this record.
   * Not valid for lwCtxSynchronize, lwStreamSynchronize.
   */
  uint32_t lwdaEventId;
} LWpti_ActivitySynchronization;


/**
 * \brief The activity record for source-level sass/source
 * line-by-line correlation.
 *
 * This activity records source level sass/source correlation
 * information.
 * (LWPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION.
   */
  LWpti_ActivityKind kind;

  /**
   * The properties of this instruction.
   */
  LWpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

 /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the instruction.
   */
  uint32_t pcOffset;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} LWpti_ActivityInstructionCorrelation;

/**
 * \brief The OpenAcc event kind for OpenAcc activity records.
 *
 * \see LWpti_ActivityKindOpenAcc
 */
typedef enum {
    LWPTI_OPENACC_EVENT_KIND_ILWALID              = 0,
    LWPTI_OPENACC_EVENT_KIND_DEVICE_INIT          = 1,
    LWPTI_OPENACC_EVENT_KIND_DEVICE_SHUTDOWN      = 2,
    LWPTI_OPENACC_EVENT_KIND_RUNTIME_SHUTDOWN     = 3,
    LWPTI_OPENACC_EVENT_KIND_ENQUEUE_LAUNCH       = 4,
    LWPTI_OPENACC_EVENT_KIND_ENQUEUE_UPLOAD       = 5,
    LWPTI_OPENACC_EVENT_KIND_ENQUEUE_DOWNLOAD     = 6,
    LWPTI_OPENACC_EVENT_KIND_WAIT                 = 7,
    LWPTI_OPENACC_EVENT_KIND_IMPLICIT_WAIT        = 8,
    LWPTI_OPENACC_EVENT_KIND_COMPUTE_CONSTRUCT    = 9,
    LWPTI_OPENACC_EVENT_KIND_UPDATE               = 10,
    LWPTI_OPENACC_EVENT_KIND_ENTER_DATA           = 11,
    LWPTI_OPENACC_EVENT_KIND_EXIT_DATA            = 12,
    LWPTI_OPENACC_EVENT_KIND_CREATE               = 13,
    LWPTI_OPENACC_EVENT_KIND_DELETE               = 14,
    LWPTI_OPENACC_EVENT_KIND_ALLOC                = 15,
    LWPTI_OPENACC_EVENT_KIND_FREE                 = 16,
    LWPTI_OPENACC_EVENT_KIND_FORCE_INT            = 0x7fffffff
} LWpti_OpenAccEventKind;

/**
 * \brief The OpenAcc parent construct kind for OpenAcc activity records.
 */
typedef enum {
    LWPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN          = 0,
    LWPTI_OPENACC_CONSTRUCT_KIND_PARALLEL         = 1,
    LWPTI_OPENACC_CONSTRUCT_KIND_KERNELS          = 2,
    LWPTI_OPENACC_CONSTRUCT_KIND_LOOP             = 3,
    LWPTI_OPENACC_CONSTRUCT_KIND_DATA             = 4,
    LWPTI_OPENACC_CONSTRUCT_KIND_ENTER_DATA       = 5,
    LWPTI_OPENACC_CONSTRUCT_KIND_EXIT_DATA        = 6,
    LWPTI_OPENACC_CONSTRUCT_KIND_HOST_DATA        = 7,
    LWPTI_OPENACC_CONSTRUCT_KIND_ATOMIC           = 8,
    LWPTI_OPENACC_CONSTRUCT_KIND_DECLARE          = 9,
    LWPTI_OPENACC_CONSTRUCT_KIND_INIT             = 10,
    LWPTI_OPENACC_CONSTRUCT_KIND_SHUTDOWN         = 11,
    LWPTI_OPENACC_CONSTRUCT_KIND_SET              = 12,
    LWPTI_OPENACC_CONSTRUCT_KIND_UPDATE           = 13,
    LWPTI_OPENACC_CONSTRUCT_KIND_ROUTINE          = 14,
    LWPTI_OPENACC_CONSTRUCT_KIND_WAIT             = 15,
    LWPTI_OPENACC_CONSTRUCT_KIND_RUNTIME_API      = 16,
    LWPTI_OPENACC_CONSTRUCT_KIND_FORCE_INT        = 0x7fffffff

} LWpti_OpenAccConstructKind;

typedef enum {
    LWPTI_OPENMP_EVENT_KIND_ILWALID               = 0,
    LWPTI_OPENMP_EVENT_KIND_PARALLEL              = 1,
    LWPTI_OPENMP_EVENT_KIND_TASK                  = 2,
    LWPTI_OPENMP_EVENT_KIND_THREAD                = 3,
    LWPTI_OPENMP_EVENT_KIND_IDLE                  = 4,
    LWPTI_OPENMP_EVENT_KIND_WAIT_BARRIER          = 5,
    LWPTI_OPENMP_EVENT_KIND_WAIT_TASKWAIT         = 6,
    LWPTI_OPENMP_EVENT_KIND_FORCE_INT             = 0x7fffffff
} LWpti_OpenMpEventKind;

/**
 * \brief The base activity record for OpenAcc records.
 *
 * The OpenACC activity API part uses a LWpti_ActivityOpenAcc as a generic
 * representation for any OpenACC activity. The 'kind' field is used to determine the
 * specific activity kind, and from that the LWpti_ActivityOpenAcc object can
 * be cast to the specific OpenACC activity record type appropriate for that kind.
 *
 * Note that all OpenACC activity record types are padded and aligned to
 * ensure that each member of the record is naturally aligned.
 *
 * \see LWpti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The kind of this activity.
   */
  LWpti_ActivityKind kind;

  /**
   * LWPTI OpenACC event kind (\see LWpti_OpenAccEventKind)
   */
  LWpti_OpenAccEventKind eventKind;

  /**
   * LWPTI OpenACC parent construct kind (\see LWpti_OpenAccConstructKind)
   *
   * Note that for applications using PGI OpenACC runtime < 16.1, this
   * will always be LWPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN.
   */
  LWpti_OpenAccConstructKind parentConstruct;

  /*
   * Version number
   */
  uint32_t version;

  /*
   * 1 for any implicit event, such as an implicit wait at a synchronous data construct
   * 0 otherwise
   */
  uint32_t implicit;

  /*
   * Device type
   */
  uint32_t deviceType;

  /*
   * Device number
   */
  uint32_t deviceNumber;

  /**
   * ThreadId
   */
  uint32_t threadId;

  /*
   * Value of async() clause of the corresponding directive
   */
  uint64_t async;

  /*
   * Internal asynchronous queue number used
   */
  uint64_t asyncMap;

  /*
   * The line number of the directive or program construct or the starting line
   * number of the OpenACC construct corresponding to the event.
   * A zero value means the line number is not known.
   */
  uint32_t lineNo;

  /*
   * For an OpenACC construct, this contains the line number of the end
   * of the construct. A zero value means the line number is not known.
   */
  uint32_t endLineNo;

  /*
   * The line number of the first line of the function named in funcName.
   * A zero value means the line number is not known.
   */
  uint32_t funcLineNo;

  /*
   * The last line number of the function named in funcName.
   * A zero value means the line number is not known.
   */
  uint32_t funcEndLineNo;

  /**
   * LWPTI start timestamp
   */
  uint64_t start;

  /**
   * LWPTI end timestamp
   */
  uint64_t end;

  /**
   * LWCA device id
   * Valid only if deviceType is acc_device_lwidia.
   */
  uint32_t lwDeviceId;

  /**
   * LWCA context id
   * Valid only if deviceType is acc_device_lwidia.
   */
  uint32_t lwContextId;

  /**
   * LWCA stream id
   * Valid only if deviceType is acc_device_lwidia.
   */
  uint32_t lwStreamId;

  /**
   * The ID of the process where the OpenACC activity is exelwting.
   */
  uint32_t lwProcessId;

  /**
   * The ID of the thread where the OpenACC activity is exelwting.
   */
  uint32_t lwThreadId;

  /**
   * The OpenACC correlation ID.
   * Valid only if deviceType is acc_device_lwidia.
   * If not 0, it uniquely identifies this record. It is identical to the
   * externalId in the preceeding external correlation record of type
   * LWPTI_EXTERNAL_CORRELATION_KIND_OPENACC.
   */
  uint32_t externalId;

  /*
   * A pointer to null-terminated string containing the name of or path to
   * the source file, if known, or a null pointer if not.
   */
  const char *srcFile;

  /*
   * A pointer to a null-terminated string containing the name of the
   * function in which the event oclwrred.
   */
  const char *funcName;
} LWpti_ActivityOpenAcc;

/**
 * \brief The activity record for OpenACC data.
 *
 * (LWPTI_ACTIVITY_KIND_OPENACC_DATA).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_OPENACC_DATA.
   */
  LWpti_ActivityKind kind;

  /**
   * LWPTI OpenACC event kind (\see LWpti_OpenAccEventKind)
   */
  LWpti_OpenAccEventKind eventKind;

  /*
   * LWPTI OpenACC parent construct kind (\see LWpti_OpenAccConstructKind)
   *
   * Note that for applications using PGI OpenACC runtime < 16.1, this
   * will always be LWPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN.
   */
  LWpti_OpenAccConstructKind parentConstruct;

  /*
   * Version number
   */
  uint32_t version;

  /*
   * 1 for any implicit event, such as an implicit wait at a synchronous data construct
   * 0 otherwise
   */
  uint32_t implicit;

  /*
   * Device type
   */
  uint32_t deviceType;

  /*
   * Device number
   */
  uint32_t deviceNumber;

  /**
   * ThreadId
   */
  uint32_t threadId;

  /*
   * Value of async() clause of the corresponding directive
   */
  uint64_t async;

  /*
   * Internal asynchronous queue number used
   */
  uint64_t asyncMap;

  /*
   * The line number of the directive or program construct or the starting line
   * number of the OpenACC construct corresponding to the event.
   * A negative or zero value means the line number is not known.
   */
  uint32_t lineNo;

  /*
   * For an OpenACC construct, this contains the line number of the end
   * of the construct. A negative or zero value means the line number is not known.
   */
  uint32_t endLineNo;

  /*
   * The line number of the first line of the function named in func_name.
   * A negative or zero value means the line number is not known.
   */
  uint32_t funcLineNo;

  /*
   * The last line number of the function named in func_name.
   * A negative or zero value means the line number is not known.
   */
  uint32_t funcEndLineNo;

  /**
   * LWPTI start timestamp
   */
  uint64_t start;

  /**
   * LWPTI end timestamp
   */
  uint64_t end;

  /**
   * LWCA device id
   * Valid only if deviceType is acc_device_lwidia.
   */
  uint32_t lwDeviceId;

  /**
   * LWCA context id
   * Valid only if deviceType is acc_device_lwidia.
   */
  uint32_t lwContextId;

  /**
   * LWCA stream id
   * Valid only if deviceType is acc_device_lwidia.
   */
  uint32_t lwStreamId;

  /**
   * The ID of the process where the OpenACC activity is exelwting.
   */
  uint32_t lwProcessId;

  /**
   * The ID of the thread where the OpenACC activity is exelwting.
   */
  uint32_t lwThreadId;

  /**
   * The OpenACC correlation ID.
   * Valid only if deviceType is acc_device_lwidia.
   * If not 0, it uniquely identifies this record. It is identical to the
   * externalId in the preceeding external correlation record of type
   * LWPTI_EXTERNAL_CORRELATION_KIND_OPENACC.
   */
  uint32_t externalId;

  /*
   * A pointer to null-terminated string containing the name of or path to
   * the source file, if known, or a null pointer if not.
   */
  const char *srcFile;

  /*
   * A pointer to a null-terminated string containing the name of the
   * function in which the event oclwrred.
   */
  const char *funcName;

  /* --- end of common LWpti_ActivityOpenAcc part --- */

  /**
   * Number of bytes
   */
  uint64_t bytes;

  /**
   * Host pointer if available
   */
  uint64_t hostPtr;

  /**
   * Device pointer if available
   */
  uint64_t devicePtr;

#ifndef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad1;
#endif

  /*
   * A pointer to null-terminated string containing the name of the variable
   * for which this event is triggered, if known, or a null pointer if not.
   */
  const char *varName;

} LWpti_ActivityOpenAccData;

/**
 * \brief The activity record for OpenACC launch.
 *
 * (LWPTI_ACTIVITY_KIND_OPENACC_LAUNCH).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_OPENACC_LAUNCH.
   */
  LWpti_ActivityKind kind;

  /**
   * LWPTI OpenACC event kind (\see LWpti_OpenAccEventKind)
   */
  LWpti_OpenAccEventKind eventKind;

  /*
   * LWPTI OpenACC parent construct kind (\see LWpti_OpenAccConstructKind)
   *
   * Note that for applications using PGI OpenACC runtime < 16.1, this
   * will always be LWPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN.
   */
  LWpti_OpenAccConstructKind parentConstruct;

  /*
   * Version number
   */
  uint32_t version;

  /*
   * 1 for any implicit event, such as an implicit wait at a synchronous data construct
   * 0 otherwise
   */
  uint32_t implicit;

  /*
   * Device type
   */
  uint32_t deviceType;

  /*
   * Device number
   */
  uint32_t deviceNumber;

  /**
   * ThreadId
   */
  uint32_t threadId;

  /*
   * Value of async() clause of the corresponding directive
   */
  uint64_t async;

  /*
   * Internal asynchronous queue number used
   */
  uint64_t asyncMap;

  /*
   * The line number of the directive or program construct or the starting line
   * number of the OpenACC construct corresponding to the event.
   * A negative or zero value means the line number is not known.
   */
  uint32_t lineNo;

  /*
   * For an OpenACC construct, this contains the line number of the end
   * of the construct. A negative or zero value means the line number is not known.
   */
  uint32_t endLineNo;

  /*
   * The line number of the first line of the function named in func_name.
   * A negative or zero value means the line number is not known.
   */
  uint32_t funcLineNo;

  /*
   * The last line number of the function named in func_name.
   * A negative or zero value means the line number is not known.
   */
  uint32_t funcEndLineNo;

  /**
   * LWPTI start timestamp
   */
  uint64_t start;

  /**
   * LWPTI end timestamp
   */
  uint64_t end;

  /**
   * LWCA device id
   * Valid only if deviceType is acc_device_lwidia.
   */
  uint32_t lwDeviceId;

  /**
   * LWCA context id
   * Valid only if deviceType is acc_device_lwidia.
   */
  uint32_t lwContextId;

  /**
   * LWCA stream id
   * Valid only if deviceType is acc_device_lwidia.
   */
  uint32_t lwStreamId;

  /**
   * The ID of the process where the OpenACC activity is exelwting.
   */
  uint32_t lwProcessId;

  /**
   * The ID of the thread where the OpenACC activity is exelwting.
   */
  uint32_t lwThreadId;

  /**
   * The OpenACC correlation ID.
   * Valid only if deviceType is acc_device_lwidia.
   * If not 0, it uniquely identifies this record. It is identical to the
   * externalId in the preceeding external correlation record of type
   * LWPTI_EXTERNAL_CORRELATION_KIND_OPENACC.
   */
  uint32_t externalId;

  /*
   * A pointer to null-terminated string containing the name of or path to
   * the source file, if known, or a null pointer if not.
   */
  const char *srcFile;

  /*
   * A pointer to a null-terminated string containing the name of the
   * function in which the event oclwrred.
   */
  const char *funcName;

  /* --- end of common LWpti_ActivityOpenAcc part --- */

  /**
   * The number of gangs created for this kernel launch
   */
  uint64_t numGangs;

  /**
   * The number of workers created for this kernel launch
   */
  uint64_t numWorkers;

  /**
   * The number of vector lanes created for this kernel launch
   */
  uint64_t vectorLength;

#ifndef LWPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad1;
#endif

  /*
   * A pointer to null-terminated string containing the name of the
   * kernel being launched, if known, or a null pointer if not.
   */
  const char *kernelName;

} LWpti_ActivityOpenAccLaunch;

/**
 * \brief The activity record for OpenACC other.
 *
 * (LWPTI_ACTIVITY_KIND_OPENACC_OTHER).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_OPENACC_OTHER.
   */
  LWpti_ActivityKind kind;

  /**
   * LWPTI OpenACC event kind (\see LWpti_OpenAccEventKind)
   */
  LWpti_OpenAccEventKind eventKind;

  /*
   * LWPTI OpenACC parent construct kind (\see LWpti_OpenAccConstructKind)
   *
   * Note that for applications using PGI OpenACC runtime < 16.1, this
   * will always be LWPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN.
   */
  LWpti_OpenAccConstructKind parentConstruct;

  /*
   * Version number
   */
  uint32_t version;

  /*
   * 1 for any implicit event, such as an implicit wait at a synchronous data construct
   * 0 otherwise
   */
  uint32_t implicit;

  /*
   * Device type
   */
  uint32_t deviceType;

  /*
   * Device number
   */
  uint32_t deviceNumber;

  /**
   * ThreadId
   */
  uint32_t threadId;

  /*
   * Value of async() clause of the corresponding directive
   */
  uint64_t async;

  /*
   * Internal asynchronous queue number used
   */
  uint64_t asyncMap;

  /*
   * The line number of the directive or program construct or the starting line
   * number of the OpenACC construct corresponding to the event.
   * A negative or zero value means the line number is not known.
   */
  uint32_t lineNo;

  /*
   * For an OpenACC construct, this contains the line number of the end
   * of the construct. A negative or zero value means the line number is not known.
   */
  uint32_t endLineNo;

  /*
   * The line number of the first line of the function named in func_name.
   * A negative or zero value means the line number is not known.
   */
  uint32_t funcLineNo;

  /*
   * The last line number of the function named in func_name.
   * A negative or zero value means the line number is not known.
   */
  uint32_t funcEndLineNo;

  /**
   * LWPTI start timestamp
   */
  uint64_t start;

  /**
   * LWPTI end timestamp
   */
  uint64_t end;

  /**
   * LWCA device id
   * Valid only if deviceType is acc_device_lwidia.
   */
  uint32_t lwDeviceId;

  /**
   * LWCA context id
   * Valid only if deviceType is acc_device_lwidia.
   */
  uint32_t lwContextId;

  /**
   * LWCA stream id
   * Valid only if deviceType is acc_device_lwidia.
   */
  uint32_t lwStreamId;

  /**
   * The ID of the process where the OpenACC activity is exelwting.
   */
  uint32_t lwProcessId;

  /**
   * The ID of the thread where the OpenACC activity is exelwting.
   */
  uint32_t lwThreadId;

  /**
   * The OpenACC correlation ID.
   * Valid only if deviceType is acc_device_lwidia.
   * If not 0, it uniquely identifies this record. It is identical to the
   * externalId in the preceeding external correlation record of type
   * LWPTI_EXTERNAL_CORRELATION_KIND_OPENACC.
   */
  uint32_t externalId;

  /*
   * A pointer to null-terminated string containing the name of or path to
   * the source file, if known, or a null pointer if not.
   */
  const char *srcFile;

  /*
   * A pointer to a null-terminated string containing the name of the
   * function in which the event oclwrred.
   */
  const char *funcName;

  /* --- end of common LWpti_ActivityOpenAcc part --- */
} LWpti_ActivityOpenAccOther;


/**
 * \brief The base activity record for OpenMp records.
 *
 * \see LWpti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT {
  
  /**
   * The kind of this activity.
   */
  LWpti_ActivityKind kind;

  /**
   * LWPTI OpenMP event kind (\see LWpti_OpenMpEventKind)
   */
  LWpti_OpenMpEventKind eventKind;

  /*
   * Version number
   */
  uint32_t version;

  /**
   * ThreadId
   */
  uint32_t threadId;

  /**
   * LWPTI start timestamp
   */
  uint64_t start;

  /**
   * LWPTI end timestamp
   */
  uint64_t end;

  /**
   * The ID of the process where the OpenMP activity is exelwting.
   */
  uint32_t lwProcessId;

  /**
   * The ID of the thread where the OpenMP activity is exelwting.
   */
  uint32_t lwThreadId;

} LWpti_ActivityOpenMp;

/**
 * \brief The kind of external APIs supported for correlation.
 *
 * Custom correlation kinds are reserved for usage in external tools.
 *
 * \see LWpti_ActivityExternalCorrelation
 */
typedef enum {
    LWPTI_EXTERNAL_CORRELATION_KIND_ILWALID              = 0,

    /**
     * The external API is unknown to LWPTI
     */
    LWPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN              = 1,

    /**
     * The external API is OpenACC
     */
    LWPTI_EXTERNAL_CORRELATION_KIND_OPENACC              = 2,

    /**
     * The external API is lwstom0
     */
    LWPTI_EXTERNAL_CORRELATION_KIND_LWSTOM0              = 3,

    /**
     * The external API is lwstom1
     */
    LWPTI_EXTERNAL_CORRELATION_KIND_LWSTOM1              = 4,

    /**
     * The external API is lwstom2
     */
    LWPTI_EXTERNAL_CORRELATION_KIND_LWSTOM2              = 5,

    /**
     * Add new kinds before this line
     */
    LWPTI_EXTERNAL_CORRELATION_KIND_SIZE,

    LWPTI_EXTERNAL_CORRELATION_KIND_FORCE_INT            = 0x7fffffff
} LWpti_ExternalCorrelationKind;

/**
 * \brief The activity record for correlation with external records
 *
 * This activity record correlates native LWCA records (e.g. LWCA Driver API,
 * kernels, memcpys, ...) with records from external APIs such as OpenACC.
 * (LWPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION).
 *
 * \see LWpti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The kind of this activity.
   */
  LWpti_ActivityKind kind;

  /**
   * The kind of external API this record correlated to.
   */
  LWpti_ExternalCorrelationKind externalKind;

  /**
   * The correlation ID of the associated non-LWCA API record.
   * The exact field in the associated external record depends
   * on that record's activity kind (\see externalKind).
   */
  uint64_t externalId;

  /**
   * The correlation ID of the associated LWCA driver or runtime API record.
   */
  uint32_t correlationId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t reserved;
} LWpti_ActivityExternalCorrelation;

/**
* \brief The device type for device connected to LWLink.
*/
typedef enum {
    LWPTI_DEV_TYPE_ILWALID = 0,
    /**
    * The device type is GPU.
    */
    LWPTI_DEV_TYPE_GPU = 1,
    /**
    * The device type is LWLink processing unit in CPU.
    */
    LWPTI_DEV_TYPE_NPU = 2,
    LWPTI_DEV_TYPE_FORCE_INT = 0x7fffffff
} LWpti_DevType;

/**
* \brief LWLink information. (deprecated in LWCA 9.0)
*
* This structure gives capabilities of each logical LWLink connection between two devices,
* gpu<->gpu or gpu<->CPU which can be used to understand the topology.
* LWLink information are now reported using the
* LWpti_ActivityLwLink2 activity record.
*/

typedef struct PACKED_ALIGNMENT {
    /**
    * The activity record kind, must be LWPTI_ACTIVITY_KIND_LWLINK.
    */
    LWpti_ActivityKind kind;
    /**
    * LWLink version.
    */
    uint32_t  lwlinkVersion;
    /**
    * Type of device 0 \ref LWpti_DevType
    */
    LWpti_DevType typeDev0;
    /**
    * Type of device 1 \ref LWpti_DevType
    */
    LWpti_DevType typeDev1;
    /**
    * If typeDev0 is LWPTI_DEV_TYPE_GPU, UUID for device 0. \ref LWpti_ActivityDevice2.
    * If typeDev0 is LWPTI_DEV_TYPE_NPU, struct npu for NPU.
    */
    union {
        LWuuid    uuidDev;
        struct {
            /**
            * Index of the NPU. First index will always be zero.
            */
            uint32_t  index;
            /**
            * Domain ID of NPU. On Linux, this can be queried using lspci.
            */
            uint32_t  domainId;
        } npu;
    } idDev0;
    /**
    * If typeDev1 is LWPTI_DEV_TYPE_GPU, UUID for device 1. \ref LWpti_ActivityDevice2.
    * If typeDev1 is LWPTI_DEV_TYPE_NPU, struct npu for NPU.
    */
    union {
        LWuuid    uuidDev;
        struct {
            /**
            * Index of the NPU. First index will always be zero.
            */
            uint32_t  index;
            /**
            * Domain ID of NPU. On Linux, this can be queried using lspci.
            */
            uint32_t  domainId;
        } npu;
    } idDev1;
    /**
    * Flag gives capabilities of the link \see LWpti_LinkFlag
    */
    uint32_t flag;
    /**
    * Number of physical LWLinks present between two devices.
    */
    uint32_t  physicalLwLinkCount;
    /**
    * Port numbers for maximum 4 LWLinks connected to device 0.
    * If typeDev0 is LWPTI_DEV_TYPE_NPU, ignore this field.
    * In case of invalid/unknown port number, this field will be set
    * to value LWPTI_LWLINK_ILWALID_PORT.
    * This will be used to correlate the metric values to individual
    * physical link and attribute traffic to the logical LWLink in
    * the topology.
    */
    int8_t  portDev0[4];
    /**
    * Port numbers for maximum 4 LWLinks connected to device 1.
    * If typeDev1 is LWPTI_DEV_TYPE_NPU, ignore this field.
    * In case of invalid/unknown port number, this field will be set
    * to value LWPTI_LWLINK_ILWALID_PORT.
    * This will be used to correlate the metric values to individual
    * physical link and attribute traffic to the logical LWLink in
    * the topology.
    */
    int8_t  portDev1[4];
    /**
    * Banwidth of LWLink in kbytes/sec
    */
    uint64_t  bandwidth;
} LWpti_ActivityLwLink;

/**
* \brief LWLink information. (deprecated in LWCA 10.0)
*
* This structure gives capabilities of each logical LWLink connection between two devices,
* gpu<->gpu or gpu<->CPU which can be used to understand the topology.
* LwLink information are now reported using the
* LWpti_ActivityLwLink3 activity record.
*/

typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_LWLINK.
   */
  LWpti_ActivityKind kind;
  /**
   * LwLink version.
   */
  uint32_t  lwlinkVersion;
  /**
   * Type of device 0 \ref LWpti_DevType
   */
  LWpti_DevType typeDev0;
  /**
   * Type of device 1 \ref LWpti_DevType
   */
  LWpti_DevType typeDev1;
  /**
   * If typeDev0 is LWPTI_DEV_TYPE_GPU, UUID for device 0. \ref LWpti_ActivityDevice2.
   * If typeDev0 is LWPTI_DEV_TYPE_NPU, struct npu for NPU.
   */
  union {
    LWuuid    uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t  index;
      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t  domainId;
    } npu;
  } idDev0;
  /**
   * If typeDev1 is LWPTI_DEV_TYPE_GPU, UUID for device 1. \ref LWpti_ActivityDevice2.
   * If typeDev1 is LWPTI_DEV_TYPE_NPU, struct npu for NPU.
   */
  union {
    LWuuid    uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t  index;
      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t  domainId;
    } npu;
  } idDev1;
  /**
   * Flag gives capabilities of the link \see LWpti_LinkFlag
   */
  uint32_t flag;
  /**
   * Number of physical LWLinks present between two devices.
   */
  uint32_t  physicalLwLinkCount;
  /**
   * Port numbers for maximum 16 LWLinks connected to device 0.
   * If typeDev0 is LWPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value LWPTI_LWLINK_ILWALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical LWLink in
   * the topology.
   */
  int8_t  portDev0[LWPTI_MAX_LWLINK_PORTS];
  /**
   * Port numbers for maximum 16 LWLinks connected to device 1.
   * If typeDev1 is LWPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value LWPTI_LWLINK_ILWALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical LWLink in
   * the topology.
   */
  int8_t  portDev1[LWPTI_MAX_LWLINK_PORTS];
  /**
   * Banwidth of LWLink in kbytes/sec
   */
  uint64_t  bandwidth;
} LWpti_ActivityLwLink2;

/**
* \brief LWLink information.
*
* This structure gives capabilities of each logical LWLink connection between two devices,
* gpu<->gpu or gpu<->CPU which can be used to understand the topology.
*/

typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_LWLINK.
   */
  LWpti_ActivityKind kind;
  /**
   * LwLink version.
   */
  uint32_t  lwlinkVersion;
  /**
   * Type of device 0 \ref LWpti_DevType
   */
  LWpti_DevType typeDev0;
  /**
   * Type of device 1 \ref LWpti_DevType
   */
  LWpti_DevType typeDev1;
  /**
   * If typeDev0 is LWPTI_DEV_TYPE_GPU, UUID for device 0. \ref LWpti_ActivityDevice2.
   * If typeDev0 is LWPTI_DEV_TYPE_NPU, struct npu for NPU.
   */
  union {
    LWuuid    uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t  index;
      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t  domainId;
    } npu;
  } idDev0;
  /**
   * If typeDev1 is LWPTI_DEV_TYPE_GPU, UUID for device 1. \ref LWpti_ActivityDevice2.
   * If typeDev1 is LWPTI_DEV_TYPE_NPU, struct npu for NPU.
   */
  union {
    LWuuid    uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t  index;
      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t  domainId;
    } npu;
  } idDev1;
  /**
   * Flag gives capabilities of the link \see LWpti_LinkFlag
   */
  uint32_t flag;
  /**
   * Number of physical LWLinks present between two devices.
   */
  uint32_t  physicalLwLinkCount;
  /**
   * Port numbers for maximum 16 LWLinks connected to device 0.
   * If typeDev0 is LWPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value LWPTI_LWLINK_ILWALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical LWLink in
   * the topology.
   */
  int8_t  portDev0[LWPTI_MAX_LWLINK_PORTS];
  /**
   * Port numbers for maximum 16 LWLinks connected to device 1.
   * If typeDev1 is LWPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value LWPTI_LWLINK_ILWALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical LWLink in
   * the topology.
   */
  int8_t  portDev1[LWPTI_MAX_LWLINK_PORTS];
   /**
   * Banwidth of LWLink in kbytes/sec
   */
  uint64_t  bandwidth;
   /**
   * LWSwitch is connected as an intermediate node.
   */
  uint8_t lwswitchConnected;
   /**
   * Undefined. reserved for internal use
   */
  uint8_t pad[7];
} LWpti_ActivityLwLink3;

#define LWPTI_MAX_GPUS 32
/**
 * Field to differentiate whether PCIE Activity record
 * is of a GPU or a PCI Bridge
 */
typedef enum {
    /**
     * PCIE GPU record
     */
    LWPTI_PCIE_DEVICE_TYPE_GPU       = 0,

    /**
     * PCIE Bridge record
     */
    LWPTI_PCIE_DEVICE_TYPE_BRIDGE    = 1,

    LWPTI_PCIE_DEVICE_TYPE_FORCE_INT = 0x7fffffff
} LWpti_PcieDeviceType;

/**
 * \brief PCI devices information required to construct topology
 *
 * This structure gives capabilities of GPU and PCI bridge connected to the PCIE bus
 * which can be used to understand the topology.
 */
typedef struct PACKED_ALIGNMENT {
    /**
     * The activity record kind, must be LWPTI_ACTIVITY_KIND_PCIE.
     */
    LWpti_ActivityKind kind;
    /**
     * Type of device in topology, \ref LWpti_PcieDeviceType. If type is
     * LWPTI_PCIE_DEVICE_TYPE_GPU use devId for id and gpuAttr and if type is
     * LWPTI_PCIE_DEVICE_TYPE_BRIDGE use bridgeId for id and bridgeAttr.
     */
    LWpti_PcieDeviceType type;
    /**
     * A unique identifier for GPU or Bridge in Topology
     */
    union {
      /**
       * GPU device ID
       */
      LWdevice devId;
      /**
       * A unique identifier for Bridge in the Topology
       */
      uint32_t bridgeId;
    } id;

    /**
     * Domain for the GPU or Bridge, required to identify which PCIE bus it belongs to in
     * multiple NUMA systems.
     */
    uint32_t domain;
    /**
     * PCIE Generation of GPU or Bridge.
     */
    uint16_t pcieGeneration;
    /**
     * Link rate of the GPU or bridge in gigatransfers per second (GT/s)
     */
    uint16_t linkRate;
    /**
     * Link width of the GPU or bridge
     */
    uint16_t linkWidth;

    /**
     * Upstream bus ID for the GPU or PCI bridge. Required to identify which bus it is 
     * connected to in the topology.
     */
    uint16_t upstreamBus;

    /**
     * Attributes for more information about GPU (gpuAttr) or PCI Bridge (bridgeAttr)
     */
    union {
      struct {
        /**
         * UUID for the device. \ref LWpti_ActivityDevice2.
         */
        LWuuid    uuidDev;
        /**
         * LWdevice with which this device has P2P capability.
         * This can also be obtained by querying lwDeviceCanAccessPeer or
         * lwdaDeviceCanAccessPeer APIs
         */
        LWdevice peerDev[LWPTI_MAX_GPUS];        
      } gpuAttr;

      struct {
        /**
         * The downstream bus number, used to search downstream devices/bridges connected 
         * to this bridge.
         */
        uint16_t secondaryBus;
        /**
         * Device ID of the bridge
         */
        uint16_t deviceId;
        /**
         * Vendor ID of the bridge
         */
        uint16_t vendorId;
        /**
         * Padding for alignment
         */
        uint16_t pad0;
      } bridgeAttr;
    } attr;
} LWpti_ActivityPcie;

/**
 * \PCIE Genearation. /ref pcieGeneration
 * Enumeration of PCIE Generation for 
 * pcie activity attribute pcieGeneration
 */
typedef enum {
  /**
  * PCIE Generation 1
  */
  LWPTI_PCIE_GEN_GEN1       = 1,
  /**
  * PCIE Generation 2
  */
  LWPTI_PCIE_GEN_GEN2       = 2,
  /**
  * PCIE Generation 3
  */
  LWPTI_PCIE_GEN_GEN3       = 3,
  /**
  * PCIE Generation 4
  */
  LWPTI_PCIE_GEN_GEN4       = 4,

  LWPTI_PCIE_GEN_FORCE_INT  = 0x7fffffff
} LWpti_PcieGen;

/**
 * \brief The activity record for an instantaneous LWPTI event.
 *
 * This activity record represents a LWPTI event value
 * (LWPTI_ACTIVITY_KIND_EVENT) sampled at a particular instant.
 * This activity record kind is not produced by the activity API but is
 * included for completeness and ease-of-use. Profiler frameworks built on
 * top of LWPTI that collect event data at a particular time may choose to
 * use this type to store the collected event data.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT.
   */
  LWpti_ActivityKind kind;

  /**
   * The event ID.
   */
  LWpti_EventID id;

  /**
   * The event value.
   */
  uint64_t value;

  /**
   * The timestamp at which event is sampled
   */
  uint64_t timestamp;

  /**
   * The device id
   */
  uint32_t deviceId;
  /**
   * Undefined. reserved for internal use
   */
  uint32_t reserved;
} LWpti_ActivityInstantaneousEvent;

/**
 * \brief The activity record for an instantaneous LWPTI event
 * with event domain instance information.
 *
 * This activity record represents the a LWPTI event value for a
 * specific event domain instance
 * (LWPTI_ACTIVITY_KIND_EVENT_INSTANCE) sampled at a particular instant.
 * This activity record kind is not produced by the activity API but is
 * included for completeness and ease-of-use. Profiler frameworks built on
 * top of LWPTI that collect event data may choose to use this type to store the
 * collected event data. This activity record should be used when
 * event domain instance information needs to be associated with the
 * event.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE.
   */
  LWpti_ActivityKind kind;

  /**
   * The event ID.
   */
  LWpti_EventID id;

  /**
   * The event value.
   */
  uint64_t value;

  /**
   * The timestamp at which event is sampled
   */
  uint64_t timestamp;

  /**
   * The device id
   */
  uint32_t deviceId;
  /**
   * The event domain instance
   */
  uint8_t instance;
  /**
   * Undefined. reserved for internal use
   */
  uint8_t pad[3];
} LWpti_ActivityInstantaneousEventInstance;

/**
 * \brief The activity record for an instantaneous LWPTI metric.
 *
 * This activity record represents the collection of a LWPTI metric
 * value (LWPTI_ACTIVITY_KIND_METRIC) at a particular instance.
 * This activity record kind is not produced by the activity API but
 * is included for completeness and ease-of-use. Profiler frameworks built
 * on top of LWPTI that collect metric data may choose to use this type to
 * store the collected metric data.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC.
   */
  LWpti_ActivityKind kind;

  /**
   * The metric ID.
   */
  LWpti_MetricID id;

  /**
   * The metric value.
   */
  LWpti_MetricValue value;

  /**
   * The timestamp at which metric is sampled
   */
  uint64_t timestamp;

  /**
   * The device id
   */
  uint32_t deviceId;

  /**
   * The properties of this metric. \see LWpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * Undefined. reserved for internal use
   */
  uint8_t pad[3];
} LWpti_ActivityInstantaneousMetric;

/**
 * \brief The instantaneous activity record for a LWPTI metric with instance
 * information.  

 * This activity record represents a LWPTI metric value
 * for a specific metric domain instance
 * (LWPTI_ACTIVITY_KIND_METRIC_INSTANCE) sampled at a particular time. This
 * activity record kind is not produced by the activity API but is included for
 * completeness and ease-of-use. Profiler frameworks built on top of
 * LWPTI that collect metric data may choose to use this type to store
 * the collected metric data. This activity record should be used when
 * metric domain instance information needs to be associated with the
 * metric.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be LWPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE.
   */
  LWpti_ActivityKind kind;

  /**
   * The metric ID.
   */
  LWpti_MetricID id;

  /**
   * The metric value.
   */
  LWpti_MetricValue value;

  /**
   * The timestamp at which metric is sampled
   */
  uint64_t timestamp;

  /**
   * The device id
   */
  uint32_t deviceId;

  /**
   * The properties of this metric. \see LWpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The metric domain instance
   */
  uint8_t instance;
  /**
   * Undefined. reserved for internal use
   */
  uint8_t pad[2];
} LWpti_ActivityInstantaneousMetricInstance;

END_PACKED_ALIGNMENT

/**
 * \brief Activity attributes.
 *
 * These attributes are used to control the behavior of the activity
 * API.
 */
typedef enum {
    /**
     * The device memory size (in bytes) reserved for storing profiling
     * data for non-CDP operations, especially for conlwrrent kernel
     * tracing, for each buffer on a context. The value is a size_t.
     *
     * Having larger buffer size means less flush operations but
     * consumes more device memory. Having smaller buffer size
     * increases the risk of dropping timestamps for kernel records
     * if too many kernels are launched/replayed at one time. This
     * value only applies to new buffer allocations.
     *
     * Set this value before initializing LWCA or before creating a
     * context to ensure it is considered for the following allocations.
     *
     * The default value is 8388608 (8MB).
     *
     * Note: The actual amount of device memory per buffer reserved by
     * LWPTI might be larger.
     */
    LWPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE              = 0,
    /**
     * The device memory size (in bytes) reserved for storing profiling
     * data for CDP operations for each buffer on a context. The
     * value is a size_t.
     *
     * Having larger buffer size means less flush operations but
     * consumes more device memory. This value only applies to new
     * allocations.
     *
     * Set this value before initializing LWCA or before creating a
     * context to ensure it is considered for the following allocations.
     *
     * The default value is 8388608 (8MB).
     *
     * Note: The actual amount of device memory per context reserved by
     * LWPTI might be larger.
     */
    LWPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP          = 1,
    /**
     * The maximum number of memory buffers per context. The value is
     * a size_t.
     *
     * Buffers can be reused by the context. Increasing this value
     * reduces the number of times LWPTI needs to flush the buffers.
     * Setting this value will not modify the number of memory buffers
     * lwrrently stored.
     *
     * Set this value before initializing LWCA to ensure the limit is
     * not exceeded.
     *
     * The default value is 100.
     */
    LWPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT        = 2,

    /**
     * The profiling semaphore pool size reserved for storing profiling
     * data for serialized kernels and memory operations for each context.
     * The value is a size_t.
     *
     * Having larger pool size means less semaphore query operations but
     * consumes more device resources. Having smaller pool size increases
     * the risk of dropping timestamps for kernel and memcpy records if
     * too many kernels or memcpy are launched/replayed at one time.
     * This value only applies to new pool allocations.
     *
     * Set this value before initializing LWCA or before creating a
     * context to ensure it is considered for the following allocations.
     *
     * The default value is 65536.
     *
     */
    LWPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE   = 3,
    /**
     * The maximum number of profiling semaphore pools per context. The
     * value is a size_t.
     *
     * Profiling semaphore pool can be reused by the context. Increasing
     * this value reduces the number of times LWPTI needs to query semaphores
     * in the pool. Setting this value will not modify the number of 
     * semaphore pools lwrrently stored.
     *
     * Set this value before initializing LWCA to ensure the limit is
     * not exceeded.
     *
     * The default value is 100.
     */
    LWPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT  = 4,

    /**
     * The flag to indicate whether user should provide activity buffer of zero value.
     * The value is a uint8_t.
     *
     * If the value of this attribute is non-zero, user should provide
     * a zero value buffer in the \ref LWpti_BuffersCallbackRequestFunc.
     * If the user does not provide a zero value buffer after setting this to non-zero,
     * the activity buffer may contain some uninitialized values when LWPTI returns it in 
     * \ref LWpti_BuffersCallbackCompleteFunc
     * 
     * If the value of this attribute is zero, LWPTI will initialize the user buffer
     * received in the \ref LWpti_BuffersCallbackRequestFunc to zero before filling it.
     * If the user sets this to zero, a few stalls may appear in critical path because LWPTI
     * will zero out the buffer in the main thread.
     * Set this value before returning from \ref LWpti_BuffersCallbackRequestFunc to
     * ensure it is considered for all the subsequent user buffers.
     *
     * The default value is 0.
     */
    LWPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER     = 5,

    LWPTI_ACTIVITY_ATTR_DEVICE_BUFFER_FORCE_INT         = 0x7fffffff
} LWpti_ActivityAttribute;

/**
 * \brief Thread-Id types.
 *
 * LWPTI uses different methods to obtain the thread-id depending on the
 * support and the underlying platform. This enum dolwments these methods
 * for each type. APIs \ref lwptiSetThreadIdType and \ref lwptiGetThreadIdType
 * can be used to set and get the thread-id type.
 */
typedef enum {
    /**
     * Default type
     * Windows uses API GetLwrrentThreadId()
     * Linux/Mac/Android/QNX use POSIX pthread API pthread_self()
     */
    LWPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT       = 0,

    /**
     * This type is based on the system API available on the underlying platform
     * and thread-id obtained is supposed to be unique for the process lifetime.
     * Windows uses API GetLwrrentThreadId()
     * Linux uses syscall SYS_gettid
     * Mac uses syscall SYS_thread_selfid
     * Android/QNX use gettid()
     */
    LWPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM        = 1,

    LWPTI_ACTIVITY_THREAD_ID_TYPE_FORCE_INT     = 0x7fffffff
} LWpti_ActivityThreadIdType;

/**
 * \brief Get the LWPTI timestamp.
 *
 * Returns a timestamp normalized to correspond with the start and end
 * timestamps reported in the LWPTI activity records. The timestamp is
 * reported in nanoseconds.
 *
 * \param timestamp Returns the LWPTI timestamp
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p timestamp is NULL
 */
LWptiResult LWPTIAPI lwptiGetTimestamp(uint64_t *timestamp);

/**
 * \brief Get the ID of a context.
 *
 * Get the ID of a context.
 *
 * \param context The context
 * \param contextId Returns a process-unique ID for the context
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_CONTEXT The context is NULL or not valid.
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p contextId is NULL
 */
LWptiResult LWPTIAPI lwptiGetContextId(LWcontext context, uint32_t *contextId);

/**
 * \brief Get the ID of a stream.
 *
 * Get the ID of a stream. The stream ID is unique within a context
 * (i.e. all streams within a context will have unique stream
 * IDs).
 *
 * \param context If non-NULL then the stream is checked to ensure
 * that it belongs to this context. Typically this parameter should be
 * null.
 * \param stream The stream
 * \param streamId Returns a context-unique ID for the stream
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_STREAM if unable to get stream ID, or
 * if \p context is non-NULL and \p stream does not belong to the
 * context
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p streamId is NULL
 *
 * **DEPRECATED** This method is deprecated as of LWCA 8.0.
 * Use method lwptiGetStreamIdEx instead.
 */
LWptiResult LWPTIAPI lwptiGetStreamId(LWcontext context, LWstream stream, uint32_t *streamId);

/**
* \brief Get the ID of a stream.
*
* Get the ID of a stream. The stream ID is unique within a context
* (i.e. all streams within a context will have unique stream
* IDs).
*
* \param context If non-NULL then the stream is checked to ensure
* that it belongs to this context. Typically this parameter should be
* null.
* \param stream The stream
* \param perThreadStream Flag to indicate if program is compiled for per-thread streams
* \param streamId Returns a context-unique ID for the stream
*
* \retval LWPTI_SUCCESS
* \retval LWPTI_ERROR_NOT_INITIALIZED
* \retval LWPTI_ERROR_ILWALID_STREAM if unable to get stream ID, or
* if \p context is non-NULL and \p stream does not belong to the
* context
* \retval LWPTI_ERROR_ILWALID_PARAMETER if \p streamId is NULL
*/
LWptiResult LWPTIAPI lwptiGetStreamIdEx(LWcontext context, LWstream stream, uint8_t perThreadStream, uint32_t *streamId);

/**
 * \brief Get the ID of a device
 *
 * If \p context is NULL, returns the ID of the device that contains
 * the lwrrently active context. If \p context is non-NULL, returns
 * the ID of the device which contains that context. Operates in a
 * similar manner to lwdaGetDevice() or lwCtxGetDevice() but may be
 * called from within callback functions.
 *
 * \param context The context, or NULL to indicate the current context.
 * \param deviceId Returns the ID of the device that is current for
 * the calling thread.
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_DEVICE if unable to get device ID
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p deviceId is NULL
 */
LWptiResult LWPTIAPI lwptiGetDeviceId(LWcontext context, uint32_t *deviceId);

/**
 * \brief Get the unique ID of a graph node
 *
 * Returns the unique ID of the LWCA graph node.
 *
 * \param node The graph node.
 * \param nodeId Returns the unique ID of the node
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p node is NULL
 */
LWptiResult LWPTIAPI lwptiGetGraphNodeId(LWgraphNode node, uint64_t *nodeId);

/**
 * \brief Enable collection of a specific kind of activity record.
 *
 * Enable collection of a specific kind of activity record. Multiple
 * kinds can be enabled by calling this function multiple times. By
 * default all activity kinds are disabled for collection.
 *
 * \param kind The kind of activity record to collect
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_NOT_COMPATIBLE if the activity kind cannot be enabled
 * \retval LWPTI_ERROR_ILWALID_KIND if the activity kind is not supported
 */
LWptiResult LWPTIAPI lwptiActivityEnable(LWpti_ActivityKind kind);

/**
 * \brief Disable collection of a specific kind of activity record.
 *
 * Disable collection of a specific kind of activity record. Multiple
 * kinds can be disabled by calling this function multiple times. By
 * default all activity kinds are disabled for collection.
 *
 * \param kind The kind of activity record to stop collecting
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_KIND if the activity kind is not supported
 */
LWptiResult LWPTIAPI lwptiActivityDisable(LWpti_ActivityKind kind);

/**
 * \brief Enable collection of a specific kind of activity record for
 * a context.
 *
 * Enable collection of a specific kind of activity record for a
 * context.  This setting done by this API will supersede the global
 * settings for activity records enabled by \ref lwptiActivityEnable.
 * Multiple kinds can be enabled by calling this function multiple
 * times.
 *
 * \param context The context for which activity is to be enabled
 * \param kind The kind of activity record to collect
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_NOT_COMPATIBLE if the activity kind cannot be enabled
 * \retval LWPTI_ERROR_ILWALID_KIND if the activity kind is not supported
 */
LWptiResult LWPTIAPI lwptiActivityEnableContext(LWcontext context, LWpti_ActivityKind kind);

/**
 * \brief Disable collection of a specific kind of activity record for
 * a context.
 *
 * Disable collection of a specific kind of activity record for a context.
 * This setting done by this API will supersede the global settings
 * for activity records.
 * Multiple kinds can be enabled by calling this function multiple times.
 *
 * \param context The context for which activity is to be disabled
 * \param kind The kind of activity record to stop collecting
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_KIND if the activity kind is not supported
 */
LWptiResult LWPTIAPI lwptiActivityDisableContext(LWcontext context, LWpti_ActivityKind kind);

/**
 * \brief Get the number of activity records that were dropped of
 * insufficient buffer space.
 *
 * Get the number of records that were dropped because of insufficient
 * buffer space.  The dropped count includes records that could not be
 * recorded because LWPTI did not have activity buffer space available
 * for the record (because the LWpti_BuffersCallbackRequestFunc
 * callback did not return an empty buffer of sufficient size) and
 * also CDP records that could not be record because the device-size
 * buffer was full (size is controlled by the
 * LWPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP attribute). The dropped
 * count maintained for the queue is reset to zero when this function
 * is called.
 *
 * \param context The context, or NULL to get dropped count from global queue
 * \param streamId The stream ID
 * \param dropped The number of records that were dropped since the last call
 * to this function.
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p dropped is NULL
 */
LWptiResult LWPTIAPI lwptiActivityGetNumDroppedRecords(LWcontext context, uint32_t streamId,
                                                       size_t *dropped);

/**
 * \brief Iterate over the activity records in a buffer.
 *
 * This is a helper function to iterate over the activity records in a
 * buffer. A buffer of activity records is typically obtained by
 * receiving a LWpti_BuffersCallbackCompleteFunc callback.
 *
 * An example of typical usage:
 * \code
 * LWpti_Activity *record = NULL;
 * LWptiResult status = LWPTI_SUCCESS;
 *   do {
 *      status = lwptiActivityGetNextRecord(buffer, validSize, &record);
 *      if(status == LWPTI_SUCCESS) {
 *           // Use record here...
 *      }
 *      else if (status == LWPTI_ERROR_MAX_LIMIT_REACHED)
 *          break;
 *      else {
 *          goto Error;
 *      }
 *    } while (1);
 * \endcode
 *
 * \param buffer The buffer containing activity records
 * \param record Inputs the previous record returned by
 * lwptiActivityGetNextRecord and returns the next activity record
 * from the buffer. If input value is NULL, returns the first activity
 * record in the buffer. Records of kind LWPTI_ACTIVITY_KIND_CONLWRRENT_KERNEL
 * may contain invalid (0) timestamps, indicating that no timing information could
 * be collected for lack of device memory.
 * \param validBufferSizeBytes The number of valid bytes in the buffer.
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_MAX_LIMIT_REACHED if no more records in the buffer
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p buffer is NULL.
 */
LWptiResult LWPTIAPI lwptiActivityGetNextRecord(uint8_t* buffer, size_t validBufferSizeBytes,
                                                LWpti_Activity **record);

/**
 * \brief Function type for callback used by LWPTI to request an empty
 * buffer for storing activity records.
 *
 * This callback function signals the LWPTI client that an activity
 * buffer is needed by LWPTI. The activity buffer is used by LWPTI to
 * store activity records. The callback function can decline the
 * request by setting \p *buffer to NULL. In this case LWPTI may drop
 * activity records.
 *
 * \param buffer Returns the new buffer. If set to NULL then no buffer
 * is returned.
 * \param size Returns the size of the returned buffer.
 * \param maxNumRecords Returns the maximum number of records that
 * should be placed in the buffer. If 0 then the buffer is filled with
 * as many records as possible. If > 0 the buffer is filled with at
 * most that many records before it is returned.
 */
typedef void (LWPTIAPI *LWpti_BuffersCallbackRequestFunc)(
    uint8_t **buffer,
    size_t *size,
    size_t *maxNumRecords);

/**
 * \brief Function type for callback used by LWPTI to return a buffer
 * of activity records.
 *
 * This callback function returns to the LWPTI client a buffer
 * containing activity records.  The buffer contains \p validSize
 * bytes of activity records which should be read using
 * lwptiActivityGetNextRecord. The number of dropped records can be
 * read using lwptiActivityGetNumDroppedRecords. After this call LWPTI
 * relinquished ownership of the buffer and will not use it
 * anymore. The client may return the buffer to LWPTI using the
 * LWpti_BuffersCallbackRequestFunc callback.
 * Note: LWCA 6.0 onwards, all buffers returned by this callback are
 * global buffers i.e. there is no context/stream specific buffer.
 * User needs to parse the global buffer to extract the context/stream
 * specific activity records.
 *
 * \param context The context this buffer is associated with. If NULL, the
 * buffer is associated with the global activities. This field is deprecated
 * as of LWCA 6.0 and will always be NULL.
 * \param streamId The stream id this buffer is associated with.
 * This field is deprecated as of LWCA 6.0 and will always be NULL.
 * \param buffer The activity record buffer.
 * \param size The total size of the buffer in bytes as set in
 * LWpti_BuffersCallbackRequestFunc.
 * \param validSize The number of valid bytes in the buffer.
 */
typedef void (LWPTIAPI *LWpti_BuffersCallbackCompleteFunc)(
    LWcontext context,
    uint32_t streamId,
    uint8_t *buffer,
    size_t size,
    size_t validSize);

/**
 * \brief Registers callback functions with LWPTI for activity buffer
 * handling.
 *
 * This function registers two callback functions to be used in asynchronous
 * buffer handling. If registered, activity record buffers are handled using
 * asynchronous requested/completed callbacks from LWPTI.
 *
 * Registering these callbacks prevents the client from using LWPTI's
 * blocking enqueue/dequeue functions.
 *
 * \param funcBufferRequested callback which is ilwoked when an empty
 * buffer is requested by LWPTI
 * \param funcBufferCompleted callback which is ilwoked when a buffer
 * containing activity records is available from LWPTI
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if either \p
 * funcBufferRequested or \p funcBufferCompleted is NULL
 */
LWptiResult LWPTIAPI lwptiActivityRegisterCallbacks(LWpti_BuffersCallbackRequestFunc funcBufferRequested,
        LWpti_BuffersCallbackCompleteFunc funcBufferCompleted);

/**
 * \brief Wait for all activity records to be delivered via the
 * completion callback.
 *
 * This function does not return until all activity records associated
 * with the specified context/stream are returned to the LWPTI client
 * using the callback registered in lwptiActivityRegisterCallbacks. To
 * ensure that all activity records are complete, the requested
 * stream(s), if any, are synchronized.
 *
 * If \p context is NULL, the global activity records (i.e. those not
 * associated with a particular stream) are flushed (in this case no
 * streams are synchonized).  If \p context is a valid LWcontext and
 * \p streamId is 0, the buffers of all streams of this context are
 * flushed.  Otherwise, the buffers of the specified stream in this
 * context is flushed.
 *
 * Before calling this function, the buffer handling callback api
 * must be activated by calling lwptiActivityRegisterCallbacks.
 *
 * \param context A valid LWcontext or NULL.
 * \param streamId The stream ID.
 * \param flag The flag can be set to indicate a forced flush. See LWpti_ActivityFlag
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_LWPTI_ERROR_ILWALID_OPERATION if not preceeded
 * by a successful call to lwptiActivityRegisterCallbacks
 * \retval LWPTI_ERROR_UNKNOWN an internal error oclwrred
 *
 * **DEPRECATED** This method is deprecated
 * CONTEXT and STREAMID will be ignored. Use lwptiActivityFlushAll
 * to flush all data.
 */
LWptiResult LWPTIAPI lwptiActivityFlush(LWcontext context, uint32_t streamId, uint32_t flag);

/**
 * \brief Request to deliver activity records via the buffer completion callback.
 *
 * This function returns the activity records associated with all contexts/streams
 * (and the global buffers not associated with any stream) to the LWPTI client
 * using the callback registered in lwptiActivityRegisterCallbacks.
 *
 * This is a blocking call but it doesn't issue any LWCA synchronization calls
 * implicitly thus it's not guaranteed that all activities are completed on the
 * underlying devices. Activity record is considered as completed if it has all
 * the information filled up including the timestamps if any. It is the client's
 * responsibility to issue necessary LWCA synchronization calls before calling
 * this function if all activity records with complete information are expected
 * to be delivered.
 *
 * Behavior of the function based on the input flag:
 * - ::For default flush i.e. without any flag to the function, it returns all the
 * activity buffers which have all the activity records completed. It doesn't return
 * buffers which have one or more incomplete records. Default flush can be done at
 * a regular interval in a separate thread.
 * - ::For forced flush i.e. when flag LWPTI_ACTIVITY_FLAG_FLUSH_FORCED is passed
 * to the function, it returns all the activity buffers including the ones which have
 * one or more incomplete activity records. It's suggested for clients to do the
 * force flush before the termination of the application to allow remaining buffers
 * to be flushed. This can be done in the at-exit handler.
 *
 * Before calling this function, the buffer handling callback api must be activated
 * by calling lwptiActivityRegisterCallbacks.
 *
 * \param flag The flag can be set to indicate a forced flush. See LWpti_ActivityFlag
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_OPERATION if not preceeded by a
 * successful call to lwptiActivityRegisterCallbacks
 * \retval LWPTI_ERROR_UNKNOWN an internal error oclwrred
 *
 * \see lwptiActivityFlushPeriod
 */
LWptiResult LWPTIAPI lwptiActivityFlushAll(uint32_t flag);

/**
 * \brief Read an activity API attribute.
 *
 * Read an activity API attribute and return it in \p *value.
 *
 * \param attr The attribute to read
 * \param valueSize Size of buffer pointed by the value, and
 * returns the number of bytes written to \p value
 * \param value Returns the value of the attribute
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p valueSize or \p value is NULL, or
 * if \p attr is not an activity attribute
 * \retval LWPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT Indicates that
 * the \p value buffer is too small to hold the attribute value.
 */
LWptiResult LWPTIAPI lwptiActivityGetAttribute(LWpti_ActivityAttribute attr,
        size_t *valueSize, void* value);

/**
 * \brief Write an activity API attribute.
 *
 * Write an activity API attribute.
 *
 * \param attr The attribute to write
 * \param valueSize The size, in bytes, of the value
 * \param value The attribute value to write
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p valueSize or \p value is NULL, or
 * if \p attr is not an activity attribute
 * \retval LWPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT Indicates that
 * the \p value buffer is too small to hold the attribute value.
 */
LWptiResult LWPTIAPI lwptiActivitySetAttribute(LWpti_ActivityAttribute attr,
        size_t *valueSize, void* value);


/**
 * \brief Set Unified Memory Counter configuration.
 *
 * \param config A pointer to \ref LWpti_ActivityUnifiedMemoryCounterConfig structures
 * containing Unified Memory counter configuration.
 * \param count Number of Unified Memory counter configuration structures
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p config is NULL or
 * any parameter in the \p config structures is not a valid value
 * \retval LWPTI_ERROR_UM_PROFILING_NOT_SUPPORTED One potential reason is that
 * platform (OS/arch) does not support the unified memory counters
 * \retval LWPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE Indicates that the device
 * does not support the unified memory counters
 * \retval LWPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES Indicates that
 * multi-GPU configuration without P2P support between any pair of devices
 * does not support the unified memory counters
 */
LWptiResult LWPTIAPI lwptiActivityConfigureUnifiedMemoryCounter(LWpti_ActivityUnifiedMemoryCounterConfig *config, uint32_t count);

/**
 * \brief Get auto boost state
 *
 * The profiling results can be inconsistent in case auto boost is enabled.
 * LWPTI tries to disable auto boost while profiling. It can fail to disable in
 * cases where user does not have the permissions or LWDA_AUTO_BOOST elw
 * variable is set. The function can be used to query whether auto boost is
 * enabled.
 *
 * \param context A valid LWcontext.
 * \param state A pointer to \ref LWpti_ActivityAutoBoostState structure which
 * contains the current state and the id of the process that has requested the
 * current state
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p LWcontext or \p state is NULL
 * \retval LWPTI_ERROR_NOT_SUPPORTED Indicates that the device does not support auto boost
 * \retval LWPTI_ERROR_UNKNOWN an internal error oclwrred
 */
LWptiResult LWPTIAPI lwptiGetAutoBoostState(LWcontext context, LWpti_ActivityAutoBoostState *state);

/**
 * \brief Set PC sampling configuration.
 *
 * \param ctx The context
 * \param config A pointer to \ref LWpti_ActivityPCSamplingConfig structure
 * containing PC sampling configuration.
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_ILWALID_OPERATION if this api is called while
 * some valid event collection method is set.
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p config is NULL or
 * any parameter in the \p config structures is not a valid value
 * \retval LWPTI_ERROR_NOT_SUPPORTED Indicates that the system/device
 * does not support the unified memory counters
 */
LWptiResult LWPTIAPI lwptiActivityConfigurePCSampling(LWcontext ctx, LWpti_ActivityPCSamplingConfig *config);

/**
 * \brief Returns the last error from a lwpti call or callback
 *
 * Returns the last error that has been produced by any of the lwpti api calls
 * or the callback in the same host thread and resets it to LWPTI_SUCCESS.
 */
LWptiResult LWPTIAPI lwptiGetLastError(void);

/**
 * \brief Set the thread-id type
 *
 * LWPTI uses the method corresponding to set type to generate the thread-id.
 * See enum /ref LWpti_ActivityThreadIdType for the list of methods.
 * Activity records having thread-id field contain the same value.
 * Thread id type must not be changed during the profiling session to
 * avoid thread-id value mismatch across activity records.
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_SUPPORTED if \p type is not supported on the platform
 */
LWptiResult LWPTIAPI lwptiSetThreadIdType(LWpti_ActivityThreadIdType type);

/**
 * \brief Get the thread-id type
 *
 * Returns the thread-id type used in LWPTI
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p type is NULL
  */
LWptiResult LWPTIAPI lwptiGetThreadIdType(LWpti_ActivityThreadIdType *type);

/**
* \brief Check support for a compute capability
*
* This function is used to check the support for a device based on
* it's compute capability. It sets the \p support when the compute
* capability is supported by the current version of LWPTI, and clears
* it otherwise. This version of LWPTI might not support all GPUs sharing
* the same compute capability. It is suggested to use API \ref
* lwptiDeviceSupported which provides correct information.
*
* \param major The major revision number of the compute capability
* \param minor The minor revision number of the compute capability
* \param support Pointer to an integer to return the support status
*
* \retval LWPTI_SUCCESS
* \retval LWPTI_ERROR_ILWALID_PARAMETER if \p support is NULL
*
* \sa ::lwptiDeviceSupported
*/
LWptiResult LWPTIAPI lwptiComputeCapabilitySupported(int major, int minor, int *support);

/**
* \brief Check support for a compute device
*
* This function is used to check the support for a compute device.
* It sets the \p support when the device is supported by the current
* version of LWPTI, and clears it otherwise.
*
* \param dev The device handle returned by LWCA Driver API lwDeviceGet
* \param support Pointer to an integer to return the support status
*
* \retval LWPTI_SUCCESS
* \retval LWPTI_ERROR_ILWALID_PARAMETER if \p support is NULL
* \retval LWPTI_ERROR_ILWALID_DEVICE if \p dev is not a valid device
*
* \sa ::lwptiComputeCapabilitySupported
*/
LWptiResult LWPTIAPI lwptiDeviceSupported(LWdevice dev, int *support);

/**
 * This indicates the virtualization mode in which LWCA device is running
 */
typedef enum {
  /**
   * No virtualization mode isassociated with the device
   * i.e. it's a baremetal GPU
   */
  LWPTI_DEVICE_VIRTUALIZATION_MODE_NONE = 0,
  /**
   * The device is associated with the pass-through GPU.
   * In this mode, an entire physical GPU is directly assigned
   * to one virtual machine (VM).
   */
  LWPTI_DEVICE_VIRTUALIZATION_MODE_PASS_THROUGH = 1,
  /**
   * The device is associated with the virtual GPU (vGPU).
   * In this mode multiple virtual machines (VMs) have simultaneous,
   * direct access to a single physical GPU.
   */
  LWPTI_DEVICE_VIRTUALIZATION_MODE_VIRTUAL_GPU = 2,

  LWPTI_DEVICE_VIRTUALIZATION_MODE_FORCE_INT = 0x7fffffff
} LWpti_DeviceVirtualizationMode;

/**
 * \brief Query the virtualization mode of the device
 *
 * This function is used to query the virtualization mode of the LWCA device.
 *
 * \param dev The device handle returned by LWCA Driver API lwDeviceGet
 * \param mode Pointer to an LWpti_DeviceVirtualizationMode to return the virtualization mode
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_ILWALID_DEVICE if \p dev is not a valid device
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p mode is NULL
 *
 */
LWptiResult LWPTIAPI lwptiDeviceVirtualizationMode(LWdevice dev, LWpti_DeviceVirtualizationMode *mode);

/**
 * \brief Detach LWPTI from the running process
 *
 * This API detaches the LWPTI from the running process. It destroys and cleans up all the
 * resources associated with LWPTI in the current process. After LWPTI detaches from the process,
 * the process will keep on running with no LWPTI attached to it.
 * For safe operation of the API, it is recommended this API is ilwoked from the exit callsite
 * of any of the LWCA Driver or Runtime API. Otherwise LWPTI client needs to make sure that
 * required LWCA synchronization and LWPTI activity buffer flush is done before calling the API.
 * Sample code showing the usage of the API in the lwpti callback handler code:
 * \code
    void LWPTIAPI
    lwptiCallbackHandler(void *userdata, LWpti_CallbackDomain domain,
        LWpti_CallbackId cbid, void *cbdata)
    {
        const LWpti_CallbackData *cbInfo = (LWpti_CallbackData *)cbdata;

        // Take this code path when LWPTI detach is requested
        if (detachLwpti) {
            switch(domain)
            {
            case LWPTI_CB_DOMAIN_RUNTIME_API:
            case LWPTI_CB_DOMAIN_DRIVER_API:
                if (cbInfo->callbackSite == LWPTI_API_EXIT) {
                    // call the LWPTI detach API
                    lwptiFinalize();
                }
                break;
            default:
                break;
            }
        }
    }
 \endcode
 */
LWptiResult LWPTIAPI lwptiFinalize(void);

/**
 * \brief Push an external correlation id for the calling thread
 *
 * This function notifies LWPTI that the calling thread is entering an external API region.
 * When a LWPTI activity API record is created while within an external API region and
 * LWPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION is enabled, the activity API record will
 * be preceeded by a LWpti_ActivityExternalCorrelation record for each \ref LWpti_ExternalCorrelationKind.
 *
 * \param kind The kind of external API activities should be correlated with.
 * \param id External correlation id.
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_ILWALID_PARAMETER The external API kind is invalid
 */
LWptiResult LWPTIAPI lwptiActivityPushExternalCorrelationId(LWpti_ExternalCorrelationKind kind, uint64_t id);

/**
 * \brief Pop an external correlation id for the calling thread
 *
 * This function notifies LWPTI that the calling thread is leaving an external API region.
 *
 * \param kind The kind of external API activities should be correlated with.
 * \param lastId If the function returns successful, contains the last external correlation id for this \p kind, can be NULL.
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_ILWALID_PARAMETER The external API kind is invalid.
 * \retval LWPTI_ERROR_QUEUE_EMPTY No external id is lwrrently associated with \p kind.
 */
LWptiResult LWPTIAPI lwptiActivityPopExternalCorrelationId(LWpti_ExternalCorrelationKind kind, uint64_t *lastId);

/**
 * \brief Controls the collection of queued and submitted timestamps for kernels.
 *
 * This API is used to control the collection of queued and submitted timestamps
 * for kernels whose records are provided through the struct \ref LWpti_ActivityKernel4.
 * Default value is 0, i.e. these timestamps are not collected. This API needs
 * to be called before initialization of LWCA and this setting should not be
 * changed during the profiling session.
 *
 * \param enable is a boolean, denoting whether these timestamps should be
 * collected
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 */
LWptiResult LWPTIAPI lwptiActivityEnableLatencyTimestamps(uint8_t enable);

/**
 * \brief Sets the flush period for the worker thread
 *
 * LWPTI creates a worker thread to minimize the perturbance for the application created
 * threads. LWPTI offloads certain operations from the application threads to the worker
 * thread, this includes synchronization of profiling resources between host and device,
 * delivery of the activity buffers to the client using the callback registered in
 * lwptiActivityRegisterCallbacks. For performance reasons, LWPTI wakes up the worker
 * thread based on certain heuristics.
 *
 * This API is used to control the flush period of the worker thread. This setting will
 * override the LWPTI heurtistics. Setting time to zero disables the periodic flush and
 * restores the default behavior.
 *
 * It's allowed to use the API \ref lwptiActivityFlushAll to flush the data on-demand, even
 * when client sets the periodic flush.
 *
 * \param time flush period in msec
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 *
 * \see lwptiActivityFlushAll
 */
LWptiResult LWPTIAPI lwptiActivityFlushPeriod(uint32_t time);

/** @} */ /* END LWPTI_ACTIVITY_API */

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility pop
#endif    

#if defined(__cplusplus)
}
#endif

#endif /*_LWPTI_ACTIVITY_H_*/
