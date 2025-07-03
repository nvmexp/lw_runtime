/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2013-2020 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl90e7.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "lwfixedtypes.h"
#include "ctrl/ctrlxxxx.h"
/* GF100_SUBDEVICE_INFOROM control commands and parameters */

#define LW90E7_CTRL_CMD(cat,idx) LWXXXX_CTRL_CMD(0x90E7, LW90E7_CTRL_##cat, idx)

/* Command categories (6 bits) */
#define LW90E7_CTRL_RESERVED (0x00)
#define LW90E7_CTRL_BBX      (0x01)
#define LW90E7_CTRL_RPR      (0x02)

/*
 * LW90E7_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LW90E7_CTRL_CMD_NULL (0x90e70000) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_RESERVED_INTERFACE_ID << 8) | 0x0" */

/*
 * LW90E7_CTRL_BBX_VERSION_DRIVER
 *
 * This structure represents the LWPU driver version.
 *
 * Driver version is limited to 6 bytes
 * 2 ^ 48 => 281,474,976,710,656 = 14 full decimal digits
 * When version is truncated, partial 15th digit is set to 1.
 *
 *   version
 *     This parameter specifies the driver version.
 */
typedef struct LW90E7_CTRL_BBX_VERSION_DRIVER {
    LW_DECLARE_ALIGNED(LwU64 version, 8);
} LW90E7_CTRL_BBX_VERSION_DRIVER;

/*
 * LW90E7_CTRL_BBX_VERSION_VBIOS
 *
 * This structure represents the LWPU GPU's VBIOS version.
 *
 *   vbios
 *     This parameter specifies the VBIOS version.
 *
 *   oem
 *     This parameter specifies the OEM revision.
 */
typedef struct LW90E7_CTRL_BBX_VERSION_VBIOS {
    LwU32 vbios;
    LwU8  oem;
} LW90E7_CTRL_BBX_VERSION_VBIOS;

/*
 * LW90E7_CTRL_BBX_VERSION_OS
 *
 * This structure represents the OS version.
 *
 *   type
 *      This parameter specifies the OS. See LW90E7_CTRL_BBX_VERSION_OS_TYPE.*
 *
 *   major
 *      This parameter specifies the major version number of the OS.
 *
 *   minor
 *      This parameter specifies the minor version number of the OS.
 *
 *   build
 *      This parameter specifies the build number of the OS.
 */
typedef struct LW90E7_CTRL_BBX_VERSION_OS {
    LwU8  type;
    LwU8  major;
    LwU8  minor;
    LwU16 build;
} LW90E7_CTRL_BBX_VERSION_OS;

#define LW90E7_CTRL_BBX_VERSION_OS_TYPE_OTHER 0x00000000
#define LW90E7_CTRL_BBX_VERSION_OS_TYPE_UNIX  0x00000001
#define LW90E7_CTRL_BBX_VERSION_OS_TYPE_WIN   0x00000002

/*
 * LW90E7_CTRL_CMD_BBX_SET_FIELD_DIAG_RESULT
 *
 * This command can be used to write the LWPU field diagnostics result in the
 * InfoROM BBX object. It contains a 12 decimal digit (8 byte) long result.
 *
 *   fieldDiagResult
 *     This parameter specifies the LWPU field diagnostic result.
 *
 *   fieldDiagVersion
 *     This parameter specifies the LWPU field diagnostic version.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 */


#define LW90E7_CTRL_CMD_BBX_SET_FIELD_DIAG_RESULT (0x90e70100) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_BBX_INTERFACE_ID << 8) | LW90E7_CTRL_BBX_SET_FIELD_DIAG_RESULT_PARAMS_MESSAGE_ID" */

#define LW90E7_CTRL_BBX_SET_FIELD_DIAG_RESULT_PARAMS_MESSAGE_ID (0x0U)

typedef struct LW90E7_CTRL_BBX_SET_FIELD_DIAG_RESULT_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 fieldDiagResult, 8);
    LwU32 fieldDiagVersion;
} LW90E7_CTRL_BBX_SET_FIELD_DIAG_RESULT_PARAMS;

/*
 * LW90E7_CTRL_CMD_BBX_GET_FIELD_DIAG_DATA
 *
 * This command can be used to query LWPU field diagnostics data from the
 * InfoROM BBX object. It contains a 12 decimal digit (8 byte) long result.
 *
 *   fieldDiagResult
 *     This parameter specifies the LWPU field diagnostic result.
 *
 *   fieldDiagVersion
 *     This parameter specifies the LWPU field diagnostic version.
 *
 *   fieldDiagTimestamp
 *     This paramter specifies the timestamp for LWPU field diagnostic data.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW90E7_CTRL_CMD_BBX_GET_FIELD_DIAG_DATA (0x90e70101) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_BBX_INTERFACE_ID << 8) | LW90E7_CTRL_BBX_GET_FIELD_DIAG_DATA_PARAMS_MESSAGE_ID" */

#define LW90E7_CTRL_BBX_GET_FIELD_DIAG_DATA_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW90E7_CTRL_BBX_GET_FIELD_DIAG_DATA_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 fieldDiagResult, 8);
    LwU32 fieldDiagVersion;
    LwU32 fieldDiagTimestamp;
} LW90E7_CTRL_BBX_GET_FIELD_DIAG_DATA_PARAMS;

/*
 * LW90E7_CTRL_CMD_BBX_GET_TIME_DATA
 *
 * This command is used to query BBX recorded timing data.
 *
 *   timeStart
 *     First time (since EPOCH in sec) when RM was loaded and BBX was updated.
 *
 *   timeEnd
 *     Last time (since EPOCH in sec) when BBX was updated.
 *
 *   timeRun
 *     Total time (in sec) the GPU was running.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW90E7_CTRL_CMD_BBX_GET_TIME_DATA (0x90e70102) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_BBX_INTERFACE_ID << 8) | LW90E7_CTRL_BBX_GET_TIME_DATA_PARAMS_MESSAGE_ID" */

#define LW90E7_CTRL_BBX_GET_TIME_DATA_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW90E7_CTRL_BBX_GET_TIME_DATA_PARAMS {
    LwU32 timeStart;
    LwU32 timeEnd;
    LwU32 timeRun;
    LwU32 time24Hours;
    LwU32 time100Hours;
} LW90E7_CTRL_BBX_GET_TIME_DATA_PARAMS;

/*
 * LW90E7_CTRL_BBX_XID_ENTRY
 *
 * This structure represents the Xid data entry
 *
 *   timestamp
 *     Timestamp (EPOCH in sec) of the entry.
 *
 *   number
 *     Xid number.
 *
 *   bEccEnabled
 *     ECC enable/disable state.
 *
 *   osType
 *     OS type.
 *
 *   osMajorVersion
 *     OS major version.
 *
 *   osMinorVersion
 *     OS minor version.
 *
 *   osBuildNumber
 *     OS build number.
 *
 *   driverVersion
 *     LWPU RM driver version.
 */
typedef struct LW90E7_CTRL_BBX_XID_ENTRY {
    LwU32  timestamp;
    LwU8   number;
    LwBool bEccEnabled;
    LwU8   osType;
    LwU8   osMajorVersion;
    LwU8   osMinorVersion;
    LwU16  osBuildNumber;
    LW_DECLARE_ALIGNED(LwU64 driverVersion, 8);
} LW90E7_CTRL_BBX_XID_ENTRY;

/* Xid os type defines */
#define LW90E7_CTRL_BBX_XID_DATA_OS_TYPE_OTHER LW90E7_CTRL_BBX_VERSION_OS_TYPE_OTHER
#define LW90E7_CTRL_BBX_XID_DATA_OS_TYPE_UNIX  LW90E7_CTRL_BBX_VERSION_OS_TYPE_UNIX
#define LW90E7_CTRL_BBX_XID_DATA_OS_TYPE_WIN   LW90E7_CTRL_BBX_VERSION_OS_TYPE_WIN

/*
 * LW90E7_CTRL_CMD_BBX_GET_XID_DATA
 *
 * NOTE: This is deprecated in favour of LW90E7_CTRL_CMD_BBX_GET_XID2_DATA.
 *
 * This command is used to query BBX recorded Xid data.
 *
 *   xid13Count
 *     Number of times Xid 13 oclwred.
 *
 *   xidOtherCount
 *     Number of times other Xids oclwred.
 *
 *   xidEntryCount
 *     Number of Xid entries.
 *
 *   xidEntry
 *     Xid entires.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW90E7_CTRL_CMD_BBX_GET_XID_DATA       (0x90e70103) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_BBX_INTERFACE_ID << 8) | LW90E7_CTRL_BBX_GET_XID_DATA_PARAMS_MESSAGE_ID" */

#define LW90E7_CTRL_BBX_XID_ENTRIES            16

#define LW90E7_CTRL_BBX_GET_XID_DATA_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW90E7_CTRL_BBX_GET_XID_DATA_PARAMS {
    LwU32 xid13Count;
    LwU32 xidOtherCount;
    LwU32 xidEntryCount;
    LW_DECLARE_ALIGNED(LW90E7_CTRL_BBX_XID_ENTRY xidEntry[LW90E7_CTRL_BBX_XID_ENTRIES], 8);
} LW90E7_CTRL_BBX_GET_XID_DATA_PARAMS;

/*
 * LW90E7_CTRL_BBX_PWR_ENTRY
 *
 * This structure represents the power data entry
 *
 *   value
 *     This parameter specifies the power consumption (mW).
 *
 *   limit
 *     This parameter specifies the power limit (mW).
 *
 *   timestamp
 *     This parameter specifies the timestamp (EPOCH) of the entry.
 */
typedef struct LW90E7_CTRL_BBX_PWR_ENTRY {
    LwU32 value;
    LwU32 limit;
    LwU32 timestamp;
} LW90E7_CTRL_BBX_PWR_ENTRY;

/*
 * LW90E7_CTRL_BBX_PWR_HISTOGRAM
 *
 * This structure represents the power consumption histogram.
 *
 *   min
 *     This parameter specifies the minimum power (in mW) for this range.
 *
 *   max
 *     This parameter specifies the maximum power (in mW) for this range.
 *
 *   frequency
 *     This parameter specifies the amount of time (in seconds) power
 *     consumption remained in this range.
 */
typedef struct LW90E7_CTRL_BBX_PWR_HISTOGRAM {
    LwU32 min;
    LwU32 max;
    LwU32 frequency;
} LW90E7_CTRL_BBX_PWR_HISTOGRAM;

/*
 * LW90E7_CTRL_CMD_BBX_GET_PWR_DATA
 *
 * This command is used to query BBX (black box) recorded power data.
 *
 *   pwrExternalDisconnectCount
 *     This parameter specifies the number of times the PCIe external power
 *     connector has been disconnected.
 *
 *   pwrLimitCrossCount
 *     This parameter specifies the number of times power consumption crossed
 *     the power limit.
 *
 *   pwrMaxDayCount
 *     This parameter specifies the number of pwrMaxDay entries.
 *
 *   pwrMaxDay
 *     This parameter specifies the maximum power consumption per day, for last
 *     few days.
 *
 *   pwrMaxMonthCount
 *     This parameter specifies the number of pwrMaxMonth entries.
 *
 *   pwrMaxMonth
 *     This parameter specifies the maximum power consumption per month, for
 *     last few months.
 *
 *   pwrHistogram
 *     This parameter specifies the power consumption histogram for time spent
 *     in different ranges.
 *
 *   pwrAverageHour
 *     This parameter specifies the moving average of power consumption per
 *     hour, for last few hours.
 *
 *   pwrAverageDay
 *     This parameter specifies the moving average of power consumption per day,
 *     for last few days.
 *
 *   pwrAverageMonth
 *     This parameter specifies the moving average of power consumption per
 *     month, for last few months.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW90E7_CTRL_CMD_BBX_GET_PWR_DATA          (0x90e70104) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_BBX_INTERFACE_ID << 8) | LW90E7_CTRL_BBX_GET_PWR_DATA_PARAMS_MESSAGE_ID" */

#define LW90E7_CTRL_BBX_PWR_ENTRIES               8
#define LW90E7_CTRL_BBX_PWR_HISTOGRAM_ENTRIES     20
#define LW90E7_CTRL_BBX_PWR_AVERAGE_HOUR_ENTRIES  24
#define LW90E7_CTRL_BBX_PWR_AVERAGE_DAY_ENTRIES   5
#define LW90E7_CTRL_BBX_PWR_AVERAGE_MONTH_ENTRIES 3

#define LW90E7_CTRL_BBX_GET_PWR_DATA_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW90E7_CTRL_BBX_GET_PWR_DATA_PARAMS {
    LwU32                         pwrExternalDisconnectCount;
    LwU32                         pwrLimitCrossCount;
    LwU32                         pwrMaxDayCount;
    LW90E7_CTRL_BBX_PWR_ENTRY     pwrMaxDay[LW90E7_CTRL_BBX_PWR_ENTRIES];
    LwU32                         pwrMaxMonthCount;
    LW90E7_CTRL_BBX_PWR_ENTRY     pwrMaxMonth[LW90E7_CTRL_BBX_PWR_ENTRIES];
    LwU32                         pwrHistogramCount;
    LW90E7_CTRL_BBX_PWR_HISTOGRAM pwrHistogram[LW90E7_CTRL_BBX_PWR_HISTOGRAM_ENTRIES];
    LwU32                         pwrAverageHour[LW90E7_CTRL_BBX_PWR_AVERAGE_HOUR_ENTRIES];
    LwU32                         pwrAverageDay[LW90E7_CTRL_BBX_PWR_AVERAGE_DAY_ENTRIES];
    LwU32                         pwrAverageMonth[LW90E7_CTRL_BBX_PWR_AVERAGE_MONTH_ENTRIES];
} LW90E7_CTRL_BBX_GET_PWR_DATA_PARAMS;

/*
 * LW90E7_CTRL_BBX_TEMP_ENTRY
 *
 * This structure represents the GPU TEMP with its timestamp.
 *
 *   value
 *     This parameter specifies the GPU Temperature
 *     (LwTemp i.e. SFXP 24.8 format in Celsius).
 *
 *   timestamp
 *     This parameter specifies the timestamp (EPOCH) of the entry.
 */
typedef struct LW90E7_CTRL_BBX_TEMP_ENTRY {
    LwTemp value;
    LwU32  timestamp;
} LW90E7_CTRL_BBX_TEMP_ENTRY;

/*
 * LW90E7_CTRL_BBX_TEMP_HISTOGRAM
 *
 * This structure represents the GPU temperature histogram.
 *
 *   base
 *     This parameter specifies the temperature (in Celsius) of first entry.
 *
 *   step
 *     This parameter specifies the step change in temperature (in Celsius)
 *     between entries.
 *
 *   count
 *     This parameter specifies the number of entries in histogram.
 *
 *   frequency
 *     This parameter specifies the value for each entry.
 */
#define LW90E7_CTRL_BBX_TEMP_HISTOGRAM_ENTRIES 21

typedef struct LW90E7_CTRL_BBX_TEMP_HISTOGRAM {
    LwS8  base;
    LwS8  step;
    LwU8  count;
    LwU32 frequency[LW90E7_CTRL_BBX_TEMP_HISTOGRAM_ENTRIES];
} LW90E7_CTRL_BBX_TEMP_HISTOGRAM;

/*
 * LW90E7_CTRL_CMD_BBX_GET_TEMP_DATA
 *
 * This command is used to query BBX recorded temperature data.
 *
 *   tempSumDelta
 *     This parameter specifies the total sum of GPU temperature change in its
 *     lifetime in 0.1C granularity.
 *
 *   tempHistogramThreshold
 *     This parameter specifies the histogram of GPU temperature crossing
 *     various thresholds. See LW90E7_CTRL_BBX_TEMP_HISTOGRAM.
 *     .base/.step - temperature thresholds.
 *     .frequency  - number of times temperature crossed it.
 *
 *   tempHistogramTime
 *     This parameter specifies the histogram of time GPU was in various
 *     temperature ranges. See LW90E7_CTRL_BBX_TEMP_HISTOGRAM.
 *     .base/.step - upper bound of that range.
 *     .frequency  - amount of time temeprature remained in this range.
 *
 *   tempMaxDayCount
 *     This parameter specifies the number of tempMaxDay entries.
 *
 *   tempMaxDay
 *     This parameter specifies the maximum GPU temperature per day, for last
 *     few days.
 *
 *   tempMinDayCount
 *     This parameter specifies the number of tempMinDay entries.
 *
 *   tempMinDay
 *     This parameter specifies the minimum GPU temperature per day, for last
 *     few days.
 *
 *   tempMaxMonthCount
 *     This parameter specifies the number of tempMaxMonth entries.
 *
 *   tempMaxMonth
 *     This parameter specifies the maximum GPU temperature per month, for last
 *     few months.
 *
 *   tempMinMonthCount
 *     This parameter specifies the number of tempMinMonth entries.
 *
 *   tempMinMonth
 *     This parameter specifies the minimum GPU temperature per month, for last
 *     few months.
 *
 *   tempAverageHour
 *     This parameter specifies the moving average of GPU temperature per hour,
 *     for last few hours.
 *
 *   tempAverageDay
 *     This parameter specifies the moving average of GPU temperature per day,
 *     for last few days.
 *
 *   tempAverageMonth
 *     This parameter specifies the moving average of GPU temperature per month,
 *     for last few months.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW90E7_CTRL_CMD_BBX_GET_TEMP_DATA               (0x90e70105) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_BBX_INTERFACE_ID << 8) | LW90E7_CTRL_BBX_GET_TEMP_DATA_PARAMS_MESSAGE_ID" */

#define LW90E7_CTRL_BBX_TEMP_ENTRIES                    5
#define LW90E7_CTRL_BBX_TEMP_AVERAGE_HOUR_ENTRIES       24
#define LW90E7_CTRL_BBX_TEMP_AVERAGE_DAY_ENTRIES        5
#define LW90E7_CTRL_BBX_TEMP_AVERAGE_MONTH_ENTRIES      3
#define LW90E7_CTRL_BBX_TEMP_HOURLY_MAX_ENTRIES         168
#define LW90E7_CTRL_BBX_TEMP_COMPRESSION_BUFFER_ENTRIES 1096

#define LW90E7_CTRL_BBX_GET_TEMP_DATA_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW90E7_CTRL_BBX_GET_TEMP_DATA_PARAMS {
    LwU32                          tempSumDelta;
    LW90E7_CTRL_BBX_TEMP_HISTOGRAM tempHistogramThreshold;
    LW90E7_CTRL_BBX_TEMP_HISTOGRAM tempHistogramTime;
    LwU32                          tempMaxDayCount;
    LW90E7_CTRL_BBX_TEMP_ENTRY     tempMaxDay[LW90E7_CTRL_BBX_TEMP_ENTRIES];
    LwU32                          tempMinDayCount;
    LW90E7_CTRL_BBX_TEMP_ENTRY     tempMinDay[LW90E7_CTRL_BBX_TEMP_ENTRIES];
    LwU32                          tempMaxWeekCount;
    LW90E7_CTRL_BBX_TEMP_ENTRY     tempMaxWeek[LW90E7_CTRL_BBX_TEMP_ENTRIES];
    LwU32                          tempMinWeekCount;
    LW90E7_CTRL_BBX_TEMP_ENTRY     tempMinWeek[LW90E7_CTRL_BBX_TEMP_ENTRIES];
    LwU32                          tempMaxMonthCount;
    LW90E7_CTRL_BBX_TEMP_ENTRY     tempMaxMonth[LW90E7_CTRL_BBX_TEMP_ENTRIES];
    LwU32                          tempMinMonthCount;
    LW90E7_CTRL_BBX_TEMP_ENTRY     tempMinMonth[LW90E7_CTRL_BBX_TEMP_ENTRIES];
    LwU32                          tempMaxAllCount;
    LW90E7_CTRL_BBX_TEMP_ENTRY     tempMaxAll[LW90E7_CTRL_BBX_TEMP_ENTRIES];
    LwU32                          tempMinAllCount;
    LW90E7_CTRL_BBX_TEMP_ENTRY     tempMinAll[LW90E7_CTRL_BBX_TEMP_ENTRIES];
    LwTemp                         tempAverageHour[LW90E7_CTRL_BBX_TEMP_AVERAGE_HOUR_ENTRIES];
    LwTemp                         tempAverageDay[LW90E7_CTRL_BBX_TEMP_AVERAGE_DAY_ENTRIES];
    LwTemp                         tempAverageMonth[LW90E7_CTRL_BBX_TEMP_AVERAGE_MONTH_ENTRIES];
    LwU32                          tempHourlyMaxSampleCount;
    LW90E7_CTRL_BBX_TEMP_ENTRY     tempHourlyMaxSample[LW90E7_CTRL_BBX_TEMP_HOURLY_MAX_ENTRIES];
    LwU32                          tempCompressionBufferCount;
    LW90E7_CTRL_BBX_TEMP_ENTRY     tempCompressionBuffer[LW90E7_CTRL_BBX_TEMP_COMPRESSION_BUFFER_ENTRIES];
} LW90E7_CTRL_BBX_GET_TEMP_DATA_PARAMS;

/*
 * LW90E7_CTRL_CMD_BBX_GET_TEMP_SAMPLES
 *
 * This command is used to query BBX recorded temperature samples. If GPU was
 * not running or RM was not loaded at the time when a samples needs to be
 * taken, that reading would be 0.
 *
 *   tempSampleInterval
 *     This parameter specifies the periodic sampling interval in sec.
 *
 *   tmepSampleCount
 *     This parameter specifies the number of temperature samples.
 *
 *   tempSample
 *     This parameter specifies the temperature samples
 *     (LwTemp i.e. SFXP 24.8 format in Celsius).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW90E7_CTRL_CMD_BBX_GET_TEMP_SAMPLES (0x90e70106) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_BBX_INTERFACE_ID << 8) | LW90E7_CTRL_BBX_GET_TEMP_SAMPLES_PARAMS_MESSAGE_ID" */

#define LW90E7_CTRL_BBX_TEMP_SAMPLE_ENTRIES  800

#define LW90E7_CTRL_BBX_GET_TEMP_SAMPLES_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW90E7_CTRL_BBX_GET_TEMP_SAMPLES_PARAMS {
    LwU32  tempSampleInterval;
    LwU32  tempSampleCount;
    LwTemp tempSample[LW90E7_CTRL_BBX_TEMP_SAMPLE_ENTRIES];
} LW90E7_CTRL_BBX_GET_TEMP_SAMPLES_PARAMS;

/*
 * LW90E7_CTRL_CMD_BBX_GET_SYSTEM_DATA
 *
 * This command can be used to query system components' version history that
 * were used while running the GPU.
 *
 *   driverVersionCount
 *     This parameter specifies the number of valid driver entries.
 *
 *   driver
 *     This parameter specifies the most recent LWPU driver versions used.
 *
 *   vbiosVersionCount
 *     This parameter specifies the number of valid vbios entries.
 *
 *   vbios
 *     This parameter specifies the most recent LWPU vbios versions on the
 *     GPU.
 *
 *   osVersionCount
 *     This parameter specifies the number of valid os entries.
 *
 *   os
 *     This parameter specifies the most recent known OS versions on which the
 *     GPU was run.
 *
 * Each component entry specified above contains the following fields:
 *
 *   version
 *     This parameter specifies the component's version.
 *
 *   timestamp
 *     This parameter specifies the time (EPOCH) when this version was first
 *     detected.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW90E7_CTRL_CMD_BBX_GET_SYSTEM_DATA     (0x90e70109) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_BBX_INTERFACE_ID << 8) | LW90E7_CTRL_BBX_GET_SYSTEM_DATA_PARAMS_MESSAGE_ID" */

#define LW90E7_CTRL_BBX_GET_SYSTEM_DATA_ENTRIES 5

#define LW90E7_CTRL_BBX_GET_SYSTEM_DATA_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW90E7_CTRL_BBX_GET_SYSTEM_DATA_PARAMS {
    LwU32 driverVersionCount;
    struct {
        LW_DECLARE_ALIGNED(LW90E7_CTRL_BBX_VERSION_DRIVER version, 8);
        LwU32 timestamp;
    } driver[LW90E7_CTRL_BBX_GET_SYSTEM_DATA_ENTRIES];

    LwU32 vbiosVersionCount;
    struct {
        LW90E7_CTRL_BBX_VERSION_VBIOS version;
        LwU32                         timestamp;
    } vbios[LW90E7_CTRL_BBX_GET_SYSTEM_DATA_ENTRIES];

    LwU32 osVersionCount;
    struct {
        LW90E7_CTRL_BBX_VERSION_OS version;
        LwU32                      timestamp;
    } os[LW90E7_CTRL_BBX_GET_SYSTEM_DATA_ENTRIES];
} LW90E7_CTRL_BBX_GET_SYSTEM_DATA_PARAMS;

/*
 * LW90E7_CTRL_CMD_BBX_GET_PWR_SAMPLES
 *
 * This command is used to query BBX recorded power samples. If GPU was not
 * running or RM was not loaded at the time when a sample needs to be taken,
 * that reading would be 0.
 *
 *   pwrSampleInterval
 *     This parameter specifies the periodic sampling interval in sec.
 *
 *   pwrSampleCount
 *     This parameter specifies the number of power samples.
 *
 *   pwrSample
 *     This parameter specifies the power samples (in W).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW90E7_CTRL_CMD_BBX_GET_PWR_SAMPLES (0x90e70110) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_BBX_INTERFACE_ID << 8) | LW90E7_CTRL_BBX_GET_PWR_SAMPLES_PARAMS_MESSAGE_ID" */

#define LW90E7_CTRL_BBX_PWR_SAMPLE_ENTRIES  720

#define LW90E7_CTRL_BBX_GET_PWR_SAMPLES_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW90E7_CTRL_BBX_GET_PWR_SAMPLES_PARAMS {
    LwU32 pwrSampleInterval;
    LwU32 pwrSampleCount;
    LwU32 pwrSample[LW90E7_CTRL_BBX_PWR_SAMPLE_ENTRIES];
} LW90E7_CTRL_BBX_GET_PWR_SAMPLES_PARAMS;

/*
 * LW90E7_CTRL_BBX_PCIE_CORR_ERR_RATE_ENTRY
 *
 * This structure represents the PCIe correctable error rate entry.
 *
 *   value
 *     This parameter represents the error rate (errors per minute).
 *
 *   timestamp
 *     This parameter represents the timestamp (EPOCH) of the entry.
 */
typedef struct LW90E7_CTRL_BBX_PCIE_CORR_ERR_RATE_ENTRY {
    LwU32 value;
    LwU32 timestamp;
} LW90E7_CTRL_BBX_PCIE_CORR_ERR_RATE_ENTRY;

/*
 * LW90E7_CTRL_CMD_BBX_GET_PCIE_DATA
 *
 * This command is used to query BBX recorded PCIe error counters.
 *
 *   pcieNonFatalErrCount
 *     This parameter returns the number of PCIe uncorrectable non-fatal errors
 *     oclwred.
 *
 *   pcieFatalErrCount
 *     This parameter returns the number of PCIe uncorrectable fatal errors
 *     oclwred.
 *
 *   pcieAerErrCount
 *     This parameter returns the number of individual PCIe uncorrectable errors
 *     reported by Advanced Error Reporting.
 *
 *   pcieCorrErrRateMaxDayCount
 *     This parameter returns the number of pcieCorrErrRateMaxDay entries.
 *
 *   pcieCorrErrRateMaxDay
 *     This parameter returns the maximum PCIe correctable error rate per day,
 *     for last few days.
 *
 *   pcieCorrErrRateMaxMonthCount
 *     This parameter returns the number of pcieCorrErrRateMaxMonth entries.
 *
 *   pcieCorrErrRateMaxMonth
 *     This parameter returns the maximum PCIe correctable error rate per month,
 *     for last few months.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW90E7_CTRL_CMD_BBX_GET_PCIE_DATA                            (0x90e70111) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_BBX_INTERFACE_ID << 8) | LW90E7_CTRL_BBX_GET_PCIE_DATA_PARAMS_MESSAGE_ID" */

#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_RSVD                     0
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_DLINK_PROTO_ERR          1
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_SURPRISE_DOWN            2
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_POISONED_TLP             3
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_FC_PROTO_ERR             4
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_CPL_TIMEOUT              5
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_CPL_ABORT                6
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_UNEXP_CPL                7
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_RCVR_OVERFLOW            8
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_MALFORMED_TLP            9
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_ECRC_ERROR               10
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_UNSUPPORTED_REQ          11
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_ACS_VIOLATION            12
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_INTERNAL_ERROR           13
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_MC_BLOCKED_TLP           14
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_ATOMIC_OP_EGRESS_BLOCKED 15
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_TLP_PREFIX_BLOCKED       16
#define LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_ENTRIES                  18

#define LW90E7_CTRL_BBX_PCIE_CORR_ERR_RATE_ENTRIES                   5

#define LW90E7_CTRL_BBX_GET_PCIE_DATA_PARAMS_MESSAGE_ID (0x11U)

typedef struct LW90E7_CTRL_BBX_GET_PCIE_DATA_PARAMS {
    LwU32                                    pcieNonFatalErrCount;
    LwU32                                    pcieFatalErrCount;
    LwU32                                    pcieAerErrCount[LW90E7_CTRL_BBX_PCIE_AER_UNCORR_IDX_ENTRIES];
    LwU32                                    pcieCorrErrRateMaxDayCount;
    LW90E7_CTRL_BBX_PCIE_CORR_ERR_RATE_ENTRY pcieCorrErrRateMaxDay[LW90E7_CTRL_BBX_PCIE_CORR_ERR_RATE_ENTRIES];
    LwU32                                    pcieCorrErrRateMaxMonthCount;
    LW90E7_CTRL_BBX_PCIE_CORR_ERR_RATE_ENTRY pcieCorrErrRateMaxMonth[LW90E7_CTRL_BBX_PCIE_CORR_ERR_RATE_ENTRIES];
} LW90E7_CTRL_BBX_GET_PCIE_DATA_PARAMS;

/*
 * LW90E7_CTRL_BBX_XID2_ENTRY
 *
 * This structure represents the improved Xid data entry.
 *
 *   timestamp
 *     This parameter represents the timestamp (EPOCH) of this entry.
 *
 *   number
 *     This parameter represents the Xid number.
 *
 *   bEccEnabled
 *     This parameter represents the state of ECC i.e. enable/disable.
 *
 *   os
 *     This parameter represents the OS version. See LW90E7_CTRL_BBX_VERSION_OS.
 *
 *   driver
 *     This parameter represents the Driver version. See
 *     LW90E7_CTRL_BBX_VERSION_DRIVER.
 */
typedef struct LW90E7_CTRL_BBX_XID2_ENTRY {
    LwU32                      timestamp;
    LwU8                       number;
    LwBool                     bEccEnabled;
    LW90E7_CTRL_BBX_VERSION_OS os;
    LW_DECLARE_ALIGNED(LW90E7_CTRL_BBX_VERSION_DRIVER driver, 8);
} LW90E7_CTRL_BBX_XID2_ENTRY;

/*
 * LW90E7_CTRL_BBX_XID2_DETAILED_ENTRY
 *
 * This structure represents the Xid detailed data entry.
 *
 *   xid
 *     This parameter represents the Xid entry. See LW90E7_CTRL_BBX_XID_ENTRY.
 *
 *   data
 *     This parameter represents an array of data associated with the Xid.
 */
#define LW90E7_CTRL_BBX_XID2_DETAILED_DATA_ENTRIES 3
typedef struct LW90E7_CTRL_BBX_XID2_DETAILED_ENTRY {
    LW_DECLARE_ALIGNED(LW90E7_CTRL_BBX_XID2_ENTRY xid, 8);
    LwU32 data[LW90E7_CTRL_BBX_XID2_DETAILED_DATA_ENTRIES];
} LW90E7_CTRL_BBX_XID2_DETAILED_ENTRY;

/*
 * LW90E7_CTRL_CMD_BBX_GET_XID2_DATA
 *
 * This command is used to query detailed BBX recorded Xid data.
 *
 *   xid13Count
 *     This parameter returns the number of times Xid 13 oclwred.
 *
 *   xid31Count
 *     This parameter returns the number of times Xid 31 oclwred.
 *
 *   xidOtherCount
 *     This parameter returns the number of times other Xids oclwred.
 *
 *   xidFirstEntryCount
 *     This parameter returns the number of xidEntry entries.
 *
 *   xidFirstEntry
 *     This parameter returns the entires for first few Xids oclwred.
 *
 *   xidLastEntryCount
 *     This parameter returns the number of xidLastEntry entries.
 *
 *   xidLastEntry
 *     This parameter returns the entries for most recent Xids oclwred.
 *
 *   xidFirstDetailedEntryCount
 *     This parameter returns the number of xidFirstDetailedEntry entries.
 *
 *   xidFirstDetailedEntry
 *     This parameter returns the detailed entries for first few Xids oclwred.
 *
 *   xidLastDetailedEntryCount
 *     This parameter returns the number of xidLastDetailedEntry entries.
 *
 *   xidLastDetailedEntry
 *     This parameter returns the detailed entries for most recent Xids
 *     oclwred.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW90E7_CTRL_CMD_BBX_GET_XID2_DATA (0x90e70112) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_BBX_INTERFACE_ID << 8) | LW90E7_CTRL_BBX_GET_XID2_DATA_PARAMS_MESSAGE_ID" */

#define LW90E7_CTRL_BBX_XID2_ENTRIES      10

#define LW90E7_CTRL_BBX_GET_XID2_DATA_PARAMS_MESSAGE_ID (0x12U)

typedef struct LW90E7_CTRL_BBX_GET_XID2_DATA_PARAMS {
    LwU32 xid13Count;
    LwU32 xid31Count;
    LwU32 xidOtherCount;
    LwU32 xidFirstEntryCount;
    LW_DECLARE_ALIGNED(LW90E7_CTRL_BBX_XID2_ENTRY xidFirstEntry[LW90E7_CTRL_BBX_XID2_ENTRIES], 8);
    LwU32 xidLastEntryCount;
    LW_DECLARE_ALIGNED(LW90E7_CTRL_BBX_XID2_ENTRY xidLastEntry[LW90E7_CTRL_BBX_XID2_ENTRIES], 8);
    LwU32 xidFirstDetailedEntryCount;
    LW_DECLARE_ALIGNED(LW90E7_CTRL_BBX_XID2_DETAILED_ENTRY xidFirstDetailedEntry[LW90E7_CTRL_BBX_XID2_ENTRIES], 8);
    LwU32 xidLastDetailedEntryCount;
    LW_DECLARE_ALIGNED(LW90E7_CTRL_BBX_XID2_DETAILED_ENTRY xidLastDetailedEntry[LW90E7_CTRL_BBX_XID2_ENTRIES], 8);
} LW90E7_CTRL_BBX_GET_XID2_DATA_PARAMS;

/*!
 * LW90E7_CTRL_CMD_FB_GET_RPR_INFO
 *
 * Used to query the repair data stored in the InfoROM.
 *
 * @params[out] entryCount
 *   Number of valid entries in the RPR object.
 * @params[out] LW90E7_RPR_INFO
 *   Address-data pairs of the stored repair info.
 *
 * Possible status values returned are:
 *   LW_OK
 *      Success
 *   LW_ERR_NOT_SUPPORTED
 *      RPR object not present.
 *   LW_ERR_ILWALID_STATE
 *      RPR object corrupted.
 */


#define LW90E7_CTRL_CMD_RPR_GET_INFO   (0x90e70200) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_RPR_INTERFACE_ID << 8) | LW90E7_CTRL_RPR_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW90E7_CTRL_RPR_MAX_DATA_COUNT 54

typedef struct LW90E7_RPR_INFO {
    LwU32 address;
    LwU32 data;
} LW90E7_RPR_INFO;

#define LW90E7_CTRL_RPR_GET_INFO_PARAMS_MESSAGE_ID (0x0U)

typedef struct LW90E7_CTRL_RPR_GET_INFO_PARAMS {
    LwU32           entryCount;
    LW90E7_RPR_INFO repairData[LW90E7_CTRL_RPR_MAX_DATA_COUNT];
} LW90E7_CTRL_RPR_GET_INFO_PARAMS;

/*!
 * LW90E7_CTRL_CMD_RPR_WRITE_OBJECT
 *
 * Used to write a new RPR object to the InfoROM.
 *
 * @params[in] LW90E7_CTRL_RPR_WRITE_OBJECT_PARAMS
 *                  Refer LW90E7_CTRL_RPR_GET_INFO_PARAMS for details.
 *
 * Possible status values returned are:
 *   LW_OK
 *      Success
 *   LW_ERR_NOT_SUPPORTED
 *      RPR object not valid/present.
 *   LW_ERR_ILWALID_ADDRESS
 *      Input address is invalid.
 *   Other errors due to RM internal failures.
 */
#define LW90E7_CTRL_CMD_RPR_WRITE_OBJECT (0x90e70201) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_INFOROM_RPR_INTERFACE_ID << 8) | 0x1" */

typedef LW90E7_CTRL_RPR_GET_INFO_PARAMS LW90E7_CTRL_RPR_WRITE_OBJECT_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LW90E7_CTRL_CMD_RPR_WRITE_OBJECT_FINN_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW90E7_CTRL_CMD_RPR_WRITE_OBJECT_FINN_PARAMS {
    LW90E7_CTRL_RPR_WRITE_OBJECT_PARAMS params;
} LW90E7_CTRL_CMD_RPR_WRITE_OBJECT_FINN_PARAMS;



/* _ctrl90e7_h_ */

