/*
 * Copyright 2011-2017   LWPU Corporation.  All rights reserved.
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

#if !defined(_LWPTI_METRIC_H_)
#define _LWPTI_METRIC_H_

#include <lwca.h>
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
 * \defgroup LWPTI_METRIC_API LWPTI Metric API
 * Functions, types, and enums that implement the LWPTI Metric API.
 * @{
 */

/**
 * \brief ID for a metric.
 *
 * A metric provides a measure of some aspect of the device.
 */
typedef uint32_t LWpti_MetricID;

/**
 * \brief A metric category.
 *
 * Each metric is assigned to a category that represents the general
 * type of the metric. A metric's category is accessed using \ref
 * lwptiMetricGetAttribute and the LWPTI_METRIC_ATTR_CATEGORY
 * attribute.
 */
typedef enum {
  /**
   * A memory related metric.
   */
  LWPTI_METRIC_CATEGORY_MEMORY          = 0,
  /**
   * An instruction related metric.
   */
  LWPTI_METRIC_CATEGORY_INSTRUCTION     = 1,
  /**
   * A multiprocessor related metric.
   */
  LWPTI_METRIC_CATEGORY_MULTIPROCESSOR  = 2,
  /**
   * A cache related metric.
   */
  LWPTI_METRIC_CATEGORY_CACHE           = 3,
  /**
   * A texture related metric.
   */
  LWPTI_METRIC_CATEGORY_TEXTURE         = 4,
  /**
   *A Lwlink related metric.
   */
  LWPTI_METRIC_CATEGORY_LWLINK          = 5,
  /**
   *A PCIe related metric.
   */
  LWPTI_METRIC_CATEGORY_PCIE           = 6,
  LWPTI_METRIC_CATEGORY_FORCE_INT                         = 0x7fffffff,
} LWpti_MetricCategory;

/**
 * \brief A metric evaluation mode.
 *
 * A metric can be evaluated per hardware instance to know the load balancing
 * across instances of a domain or the metric can be evaluated in aggregate mode
 * when the events ilwolved in metric evaluation are from different event
 * domains. It might be possible to evaluate some metrics in both
 * modes for colwenience. A metric's evaluation mode is accessed using \ref
 * LWpti_MetricEvaluationMode and the LWPTI_METRIC_ATTR_EVALUATION_MODE
 * attribute.
 */
typedef enum {
  /**
   * If this bit is set, the metric can be profiled for each instance of the
   * domain. The event values passed to \ref lwptiMetricGetValue can contain
   * values for one instance of the domain. And \ref lwptiMetricGetValue can
   * be called for each instance.
   */
  LWPTI_METRIC_EVALUATION_MODE_PER_INSTANCE         = 1,
  /**
   * If this bit is set, the metric can be profiled over all instances. The
   * event values passed to \ref lwptiMetricGetValue can be aggregated values
   * of events for all instances of the domain.
   */
  LWPTI_METRIC_EVALUATION_MODE_AGGREGATE            = 1 << 1,
  LWPTI_METRIC_EVALUATION_MODE_FORCE_INT            = 0x7fffffff,
} LWpti_MetricEvaluationMode;

/**
 * \brief Kinds of metric values.
 *
 * Metric values can be one of several different kinds. Corresponding
 * to each kind is a member of the LWpti_MetricValue union. The metric
 * value returned by \ref lwptiMetricGetValue should be accessed using
 * the appropriate member of that union based on its value kind.
 */
typedef enum {
  /**
   * The metric value is a 64-bit double.
   */
  LWPTI_METRIC_VALUE_KIND_DOUBLE            = 0,
  /**
   * The metric value is a 64-bit unsigned integer.
   */
  LWPTI_METRIC_VALUE_KIND_UINT64            = 1,
  /**
   * The metric value is a percentage represented by a 64-bit
   * double. For example, 57.5% is represented by the value 57.5.
   */
  LWPTI_METRIC_VALUE_KIND_PERCENT           = 2,
  /**
   * The metric value is a throughput represented by a 64-bit
   * integer. The unit for throughput values is bytes/second.
   */
  LWPTI_METRIC_VALUE_KIND_THROUGHPUT        = 3,
  /**
   * The metric value is a 64-bit signed integer.
   */
  LWPTI_METRIC_VALUE_KIND_INT64             = 4,
  /**
   * The metric value is a utilization level, as represented by
   * LWpti_MetricValueUtilizationLevel.
   */
  LWPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL = 5,

  LWPTI_METRIC_VALUE_KIND_FORCE_INT  = 0x7fffffff
} LWpti_MetricValueKind;

/**
 * \brief Enumeration of utilization levels for metrics values of kind
 * LWPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL. Utilization values can
 * vary from IDLE (0) to MAX (10) but the enumeration only provides
 * specific names for a few values.
 */
typedef enum {
  LWPTI_METRIC_VALUE_UTILIZATION_IDLE      = 0,
  LWPTI_METRIC_VALUE_UTILIZATION_LOW       = 2,
  LWPTI_METRIC_VALUE_UTILIZATION_MID       = 5,
  LWPTI_METRIC_VALUE_UTILIZATION_HIGH      = 8,
  LWPTI_METRIC_VALUE_UTILIZATION_MAX       = 10,
  LWPTI_METRIC_VALUE_UTILIZATION_FORCE_INT = 0x7fffffff
} LWpti_MetricValueUtilizationLevel;

/**
 * \brief Metric attributes.
 *
 * Metric attributes describe properties of a metric. These attributes
 * can be read using \ref lwptiMetricGetAttribute.
 */
typedef enum {
  /**
   * Metric name. Value is a null terminated const c-string.
   */
  LWPTI_METRIC_ATTR_NAME              = 0,
  /**
   * Short description of metric. Value is a null terminated const c-string.
   */
  LWPTI_METRIC_ATTR_SHORT_DESCRIPTION = 1,
  /**
   * Long description of metric. Value is a null terminated const c-string.
   */
  LWPTI_METRIC_ATTR_LONG_DESCRIPTION  = 2,
  /**
   * Category of the metric. Value is of type LWpti_MetricCategory.
   */
  LWPTI_METRIC_ATTR_CATEGORY          = 3,
  /**
   * Value type of the metric. Value is of type LWpti_MetricValueKind.
   */
  LWPTI_METRIC_ATTR_VALUE_KIND          = 4,
  /**
   * Metric evaluation mode. Value is of type LWpti_MetricEvaluationMode.
   */
  LWPTI_METRIC_ATTR_EVALUATION_MODE     = 5,
  LWPTI_METRIC_ATTR_FORCE_INT         = 0x7fffffff,
} LWpti_MetricAttribute;

/**
 * \brief A metric value.
 *
 * Metric values can be one of several different kinds. Corresponding
 * to each kind is a member of the LWpti_MetricValue union. The metric
 * value returned by \ref lwptiMetricGetValue should be accessed using
 * the appropriate member of that union based on its value kind.
 */
typedef union {
  /*
   * Value for LWPTI_METRIC_VALUE_KIND_DOUBLE.
   */
  double metricValueDouble;
  /*
   * Value for LWPTI_METRIC_VALUE_KIND_UINT64.
   */
  uint64_t metricValueUint64;
  /*
   * Value for LWPTI_METRIC_VALUE_KIND_INT64.
   */
  int64_t metricValueInt64;
  /*
   * Value for LWPTI_METRIC_VALUE_KIND_PERCENT. For example, 57.5% is
   * represented by the value 57.5.
   */
  double metricValuePercent;
  /*
   * Value for LWPTI_METRIC_VALUE_KIND_THROUGHPUT.  The unit for
   * throughput values is bytes/second.
   */
  uint64_t metricValueThroughput;
  /*
   * Value for LWPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL.
   */
  LWpti_MetricValueUtilizationLevel metricValueUtilizationLevel;
} LWpti_MetricValue;

/**
 * \brief Device class.
 *
 * Enumeration of device classes for metric property
 * LWPTI_METRIC_PROPERTY_DEVICE_CLASS.
 */
typedef enum {
  LWPTI_METRIC_PROPERTY_DEVICE_CLASS_TESLA          = 0,
  LWPTI_METRIC_PROPERTY_DEVICE_CLASS_QUADRO         = 1,
  LWPTI_METRIC_PROPERTY_DEVICE_CLASS_GEFORCE        = 2,
  LWPTI_METRIC_PROPERTY_DEVICE_CLASS_TEGRA          = 3,
} LWpti_MetricPropertyDeviceClass;

/**
 * \brief Metric device properties.
 *
 * Metric device properties describe device properties which are needed for a metric.
 * Some of these properties can be collected using lwDeviceGetAttribute.
 */
typedef enum {
  /*
   * Number of multiprocessors on a device.  This can be collected
   * using value of \param LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT of
   * lwDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_MULTIPROCESSOR_COUNT,
  /*
   * Maximum number of warps on a multiprocessor. This can be
   * collected using ratio of value of \param
   * LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR and \param
   * LW_DEVICE_ATTRIBUTE_WARP_SIZE of lwDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_WARPS_PER_MULTIPROCESSOR,
  /*
   * GPU Time for kernel in ns. This should be profiled using LWPTI
   * Activity API.
   */
  LWPTI_METRIC_PROPERTY_KERNEL_GPU_TIME,
  /*
   * Clock rate for device in KHz.  This should be collected using
   * value of \param LW_DEVICE_ATTRIBUTE_CLOCK_RATE of
   * lwDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_CLOCK_RATE,
  /*
   * Number of Frame buffer units for device. This should be collected
   * using value of \param LWPTI_DEVICE_ATTRIBUTE_MAX_FRAME_BUFFERS of
   * lwptiDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_FRAME_BUFFER_COUNT,
  /*
   * Global memory bandwidth in KBytes/sec. This should be collected
   * using value of \param LWPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH
   * of lwptiDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_GLOBAL_MEMORY_BANDWIDTH,
  /*
   * PCIE link rate in Mega bits/sec. This should be collected using
   * value of \param LWPTI_DEVICE_ATTR_PCIE_LINK_RATE of
   * lwptiDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_PCIE_LINK_RATE,
  /*
   * PCIE link width for device. This should be collected using
   * value of \param LWPTI_DEVICE_ATTR_PCIE_LINK_WIDTH of
   * lwptiDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_PCIE_LINK_WIDTH,
  /*
   * PCIE generation for device. This should be collected using
   * value of \param LWPTI_DEVICE_ATTR_PCIE_GEN of
   * lwptiDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_PCIE_GEN,
  /*
   * The device class. This should be collected using
   * value of \param LWPTI_DEVICE_ATTR_DEVICE_CLASS of
   * lwptiDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_DEVICE_CLASS,
  /*
   * Peak single precision floating point operations that
   * can be performed in one cycle by the device.
   * This should be collected using value of
   * \param LWPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLE of
   * lwptiDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_FLOP_SP_PER_CYCLE,
  /*
   * Peak double precision floating point operations that
   * can be performed in one cycle by the device.
   * This should be collected using value of
   * \param LWPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLE of
   * lwptiDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_FLOP_DP_PER_CYCLE,
  /*
   * Number of L2 units on a device. This can be collected
   * using value of \param LWPTI_DEVICE_ATTR_MAX_L2_UNITS of
   * lwDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_L2_UNITS,
  /*
   * Whether ECC support is enabled on the device. This can be
   * collected using value of \param LW_DEVICE_ATTRIBUTE_ECC_ENABLED of
   * lwDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_ECC_ENABLED,
  /*
   * Peak half precision floating point operations that
   * can be performed in one cycle by the device.
   * This should be collected using value of
   * \param LWPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLE of
   * lwptiDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_FLOP_HP_PER_CYCLE,
  /*
   * LWLINK Bandwitdh for device. This should be collected
   * using value of \param LWPTI_DEVICE_ATTR_GPU_CPU_LWLINK_BW of
   * lwptiDeviceGetAttribute.
   */
  LWPTI_METRIC_PROPERTY_GPU_CPU_LWLINK_BANDWIDTH,
} LWpti_MetricPropertyID;

/**
 * \brief Get the total number of metrics available on any device.
 *
 * Returns the total number of metrics available on any LWCA-capable
 * devices.
 *
 * \param numMetrics Returns the number of metrics
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p numMetrics is NULL
*/
LWptiResult LWPTIAPI lwptiGetNumMetrics(uint32_t *numMetrics);

/**
 * \brief Get all the metrics available on any device.
 *
 * Returns the metric IDs in \p metricArray for all LWCA-capable
 * devices.  The size of the \p metricArray buffer is given by \p
 * *arraySizeBytes. The size of the \p metricArray buffer must be at
 * least \p numMetrics * sizeof(LWpti_MetricID) or all metric IDs will
 * not be returned. The value returned in \p *arraySizeBytes contains
 * the number of bytes returned in \p metricArray.
 *
 * \param arraySizeBytes The size of \p metricArray in bytes, and
 * returns the number of bytes written to \p metricArray
 * \param metricArray Returns the IDs of the metrics
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p arraySizeBytes or
 * \p metricArray are NULL
*/
LWptiResult LWPTIAPI lwptiEnumMetrics(size_t *arraySizeBytes,
                                      LWpti_MetricID *metricArray);

/**
 * \brief Get the number of metrics for a device.
 *
 * Returns the number of metrics available for a device.
 *
 * \param device The LWCA device
 * \param numMetrics Returns the number of metrics available for the
 * device
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_DEVICE
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p numMetrics is NULL
 */
LWptiResult LWPTIAPI lwptiDeviceGetNumMetrics(LWdevice device,
                                              uint32_t *numMetrics);

/**
 * \brief Get the metrics for a device.
 *
 * Returns the metric IDs in \p metricArray for a device.  The size of
 * the \p metricArray buffer is given by \p *arraySizeBytes. The size
 * of the \p metricArray buffer must be at least \p numMetrics *
 * sizeof(LWpti_MetricID) or else all metric IDs will not be
 * returned. The value returned in \p *arraySizeBytes contains the
 * number of bytes returned in \p metricArray.
 *
 * \param device The LWCA device
 * \param arraySizeBytes The size of \p metricArray in bytes, and
 * returns the number of bytes written to \p metricArray
 * \param metricArray Returns the IDs of the metrics for the device
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_DEVICE
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p arraySizeBytes or
 * \p metricArray are NULL
 */
LWptiResult LWPTIAPI lwptiDeviceEnumMetrics(LWdevice device,
                                            size_t *arraySizeBytes,
                                            LWpti_MetricID *metricArray);

/**
 * \brief Get a metric attribute.
 *
 * Returns a metric attribute in \p *value. The size of the \p
 * value buffer is given by \p *valueSize. The value returned in \p
 * *valueSize contains the number of bytes returned in \p value.
 *
 * If the attribute value is a c-string that is longer than \p
 * *valueSize, then only the first \p *valueSize characters will be
 * returned and there will be no terminating null byte.
 *
 * \param metric ID of the metric
 * \param attrib The metric attribute to read
 * \param valueSize The size of the \p value buffer in bytes, and
 * returns the number of bytes written to \p value
 * \param value Returns the attribute's value
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_METRIC_ID
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not a metric attribute
 * \retval LWPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT For non-c-string
 * attribute values, indicates that the \p value buffer is too small
 * to hold the attribute value.
 */
LWptiResult LWPTIAPI lwptiMetricGetAttribute(LWpti_MetricID metric,
                                             LWpti_MetricAttribute attrib,
                                             size_t *valueSize,
                                             void *value);

/**
 * \brief Find an metric by name.
 *
 * Find a metric by name and return the metric ID in \p *metric.
 *
 * \param device The LWCA device
 * \param metricName The name of metric to find
 * \param metric Returns the ID of the found metric or undefined if
 * unable to find the metric
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_DEVICE
 * \retval LWPTI_ERROR_ILWALID_METRIC_NAME if unable to find a metric
 * with name \p metricName. In this case \p *metric is undefined
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p metricName or \p
 * metric are NULL.
 */
LWptiResult LWPTIAPI lwptiMetricGetIdFromName(LWdevice device,
                                              const char *metricName,
                                              LWpti_MetricID *metric);

/**
 * \brief Get number of events required to callwlate a metric.
 *
 * Returns the number of events in \p numEvents that are required to
 * callwlate a metric.
 *
 * \param metric ID of the metric
 * \param numEvents Returns the number of events required for the metric
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_METRIC_ID
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p numEvents is NULL
 */
LWptiResult LWPTIAPI lwptiMetricGetNumEvents(LWpti_MetricID metric,
                                             uint32_t *numEvents);

/**
 * \brief Get the events required to callwlating a metric.
 *
 * Gets the event IDs in \p eventIdArray required to callwlate a \p
 * metric. The size of the \p eventIdArray buffer is given by \p
 * *eventIdArraySizeBytes and must be at least \p numEvents *
 * sizeof(LWpti_EventID) or all events will not be returned. The value
 * returned in \p *eventIdArraySizeBytes contains the number of bytes
 * returned in \p eventIdArray.
 *
 * \param metric ID of the metric
 * \param eventIdArraySizeBytes The size of \p eventIdArray in bytes,
 * and returns the number of bytes written to \p eventIdArray
 * \param eventIdArray Returns the IDs of the events required to
 * callwlate \p metric
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_METRIC_ID
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p eventIdArraySizeBytes or \p
 * eventIdArray are NULL.
 */
LWptiResult LWPTIAPI lwptiMetricEnumEvents(LWpti_MetricID metric,
                                           size_t *eventIdArraySizeBytes,
                                           LWpti_EventID *eventIdArray);

/**
 * \brief Get number of properties required to callwlate a metric.
 *
 * Returns the number of properties in \p numProp that are required to
 * callwlate a metric.
 *
 * \param metric ID of the metric
 * \param numProp Returns the number of properties required for the
 * metric
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_METRIC_ID
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p numProp is NULL
 */
LWptiResult LWPTIAPI lwptiMetricGetNumProperties(LWpti_MetricID metric,
                                                 uint32_t *numProp);

/**
 * \brief Get the properties required to callwlating a metric.
 *
 * Gets the property IDs in \p propIdArray required to callwlate a \p
 * metric. The size of the \p propIdArray buffer is given by \p
 * *propIdArraySizeBytes and must be at least \p numProp *
 * sizeof(LWpti_DeviceAttribute) or all properties will not be
 * returned. The value returned in \p *propIdArraySizeBytes contains
 * the number of bytes returned in \p propIdArray.
 *
 * \param metric ID of the metric
 * \param propIdArraySizeBytes The size of \p propIdArray in bytes,
 * and returns the number of bytes written to \p propIdArray
 * \param propIdArray Returns the IDs of the properties required to
 * callwlate \p metric
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_METRIC_ID
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p propIdArraySizeBytes or \p
 * propIdArray are NULL.
 */
LWptiResult LWPTIAPI lwptiMetricEnumProperties(LWpti_MetricID metric,
                                               size_t *propIdArraySizeBytes,
                                               LWpti_MetricPropertyID *propIdArray);


/**
 * \brief For a metric get the groups of events that must be collected
 * in the same pass.
 *
 * For a metric get the groups of events that must be collected in the
 * same pass to ensure that the metric is callwlated correctly. If the
 * events are not collected as specified then the metric value may be
 * inaclwrate.
 *
 * The function returns NULL if a metric does not have any required
 * event group. In this case the events needed for the metric can be
 * grouped in any manner for collection.
 *
 * \param context The context for event collection
 * \param metric The metric ID
 * \param eventGroupSets Returns a LWpti_EventGroupSets object that
 * indicates the events that must be collected in the same pass to
 * ensure the metric is callwlated correctly.  Returns NULL if no
 * grouping is required for metric
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_METRIC_ID
 */
LWptiResult LWPTIAPI lwptiMetricGetRequiredEventGroupSets(LWcontext context,
                                                          LWpti_MetricID metric,
                                                          LWpti_EventGroupSets **eventGroupSets);

/**
 * \brief For a set of metrics, get the grouping that indicates the
 * number of passes and the event groups necessary to collect the
 * events required for those metrics.
 *
 * For a set of metrics, get the grouping that indicates the number of
 * passes and the event groups necessary to collect the events
 * required for those metrics.
 *
 * \see lwptiEventGroupSetsCreate for details on event group set
 * creation.
 *
 * \param context The context for event collection
 * \param metricIdArraySizeBytes Size of the metricIdArray in bytes
 * \param metricIdArray Array of metric IDs
 * \param eventGroupPasses Returns a LWpti_EventGroupSets object that
 * indicates the number of passes required to collect the events and
 * the events to collect on each pass
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_CONTEXT
 * \retval LWPTI_ERROR_ILWALID_METRIC_ID
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p metricIdArray or
 * \p eventGroupPasses is NULL
 */
LWptiResult LWPTIAPI lwptiMetricCreateEventGroupSets(LWcontext context,
                                                     size_t metricIdArraySizeBytes,
                                                     LWpti_MetricID *metricIdArray,
                                                     LWpti_EventGroupSets **eventGroupPasses);

/**
 * \brief Callwlate the value for a metric.
 *
 * Use the events collected for a metric to callwlate the metric
 * value. Metric value evaluation depends on the evaluation mode
 * \ref LWpti_MetricEvaluationMode that the metric supports.
 * If a metric has evaluation mode as LWPTI_METRIC_EVALUATION_MODE_PER_INSTANCE,
 * then it assumes that the input event value is for one domain instance.
 * If a metric has evaluation mode as LWPTI_METRIC_EVALUATION_MODE_AGGREGATE,
 * it assumes that input event values are
 * normalized to represent all domain instances on a device. For the
 * most accurate metric collection, the events required for the metric
 * should be collected for all profiled domain instances. For example,
 * to collect all instances of an event, set the
 * LWPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES attribute on
 * the group containing the event to 1. The normalized value for the
 * event is then: (\p sum_event_values * \p totalInstanceCount) / \p
 * instanceCount, where \p sum_event_values is the summation of the
 * event values across all profiled domain instances, \p
 * totalInstanceCount is obtained from querying
 * LWPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT and \p instanceCount
 * is obtained from querying LWPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT (or
 * LWPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT).
 *
 * \param device The LWCA device that the metric is being callwlated for
 * \param metric The metric ID
 * \param eventIdArraySizeBytes The size of \p eventIdArray in bytes
 * \param eventIdArray The event IDs required to callwlate \p metric
 * \param eventValueArraySizeBytes The size of \p eventValueArray in bytes
 * \param eventValueArray The normalized event values required to
 * callwlate \p metric. The values must be order to match the order of
 * events in \p eventIdArray
 * \param timeDuration The duration over which the events were
 * collected, in ns
 * \param metricValue Returns the value for the metric
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_METRIC_ID
 * \retval LWPTI_ERROR_ILWALID_OPERATION
 * \retval LWPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT if the
 * eventIdArray does not contain all the events needed for metric
 * \retval LWPTI_ERROR_ILWALID_EVENT_VALUE if any of the
 * event values required for the metric is LWPTI_EVENT_OVERFLOW
 * \retval LWPTI_ERROR_ILWALID_METRIC_VALUE if the computed metric value
 * cannot be represented in the metric's value type. For example,
 * if the metric value type is unsigned and the computed metric value is negative
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p metricValue,
 * \p eventIdArray or \p eventValueArray is NULL
 */
LWptiResult LWPTIAPI lwptiMetricGetValue(LWdevice device,
                                         LWpti_MetricID metric,
                                         size_t eventIdArraySizeBytes,
                                         LWpti_EventID *eventIdArray,
                                         size_t eventValueArraySizeBytes,
                                         uint64_t *eventValueArray,
                                         uint64_t timeDuration,
                                         LWpti_MetricValue *metricValue);

/**
 * \brief Callwlate the value for a metric.
 *
 * Use the events and properties collected for a metric to callwlate
 * the metric value. Metric value evaluation depends on the evaluation
 * mode \ref LWpti_MetricEvaluationMode that the metric supports.  If
 * a metric has evaluation mode as
 * LWPTI_METRIC_EVALUATION_MODE_PER_INSTANCE, then it assumes that the
 * input event value is for one domain instance.  If a metric has
 * evaluation mode as LWPTI_METRIC_EVALUATION_MODE_AGGREGATE, it
 * assumes that input event values are normalized to represent all
 * domain instances on a device. For the most accurate metric
 * collection, the events required for the metric should be collected
 * for all profiled domain instances. For example, to collect all
 * instances of an event, set the
 * LWPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES attribute on
 * the group containing the event to 1. The normalized value for the
 * event is then: (\p sum_event_values * \p totalInstanceCount) / \p
 * instanceCount, where \p sum_event_values is the summation of the
 * event values across all profiled domain instances, \p
 * totalInstanceCount is obtained from querying
 * LWPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT and \p instanceCount
 * is obtained from querying LWPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT (or
 * LWPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT).
 *
 * \param metric The metric ID
 * \param eventIdArraySizeBytes The size of \p eventIdArray in bytes
 * \param eventIdArray The event IDs required to callwlate \p metric
 * \param eventValueArraySizeBytes The size of \p eventValueArray in bytes
 * \param eventValueArray The normalized event values required to
 * callwlate \p metric. The values must be order to match the order of
 * events in \p eventIdArray
 * \param propIdArraySizeBytes The size of \p propIdArray in bytes
 * \param propIdArray The metric property IDs required to callwlate \p metric
 * \param propValueArraySizeBytes The size of \p propValueArray in bytes
 * \param propValueArray The metric property values required to
 * callwlate \p metric. The values must be order to match the order of
 * metric properties in \p propIdArray
 * \param metricValue Returns the value for the metric
 *
 * \retval LWPTI_SUCCESS
 * \retval LWPTI_ERROR_NOT_INITIALIZED
 * \retval LWPTI_ERROR_ILWALID_METRIC_ID
 * \retval LWPTI_ERROR_ILWALID_OPERATION
 * \retval LWPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT if the
 * eventIdArray does not contain all the events needed for metric
 * \retval LWPTI_ERROR_ILWALID_EVENT_VALUE if any of the
 * event values required for the metric is LWPTI_EVENT_OVERFLOW
 * \retval LWPTI_ERROR_NOT_COMPATIBLE if the computed metric value
 * cannot be represented in the metric's value type. For example,
 * if the metric value type is unsigned and the computed metric value is negative
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p metricValue,
 * \p eventIdArray or \p eventValueArray is NULL
 */
LWptiResult LWPTIAPI lwptiMetricGetValue2(LWpti_MetricID metric,
                                          size_t eventIdArraySizeBytes,
                                          LWpti_EventID *eventIdArray,
                                          size_t eventValueArraySizeBytes,
                                          uint64_t *eventValueArray,
                                          size_t propIdArraySizeBytes,
                                          LWpti_MetricPropertyID *propIdArray,
                                          size_t propValueArraySizeBytes,
                                          uint64_t *propValueArray,
                                          LWpti_MetricValue *metricValue);

/** @} */ /* END LWPTI_METRIC_API */

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_LWPTI_METRIC_H_*/


