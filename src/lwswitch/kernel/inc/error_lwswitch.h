/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


#ifndef _ERROR_LWSWITCH_H_
#define _ERROR_LWSWITCH_H_

#include "lwtypes.h"

#include "ctrl_dev_internal_lwswitch.h"
#include "ctrl_dev_lwswitch.h"

//
// Error logging
//

typedef struct
{
    LwU32 addr;
    LwU32 data;
    LwU32 info;
    LwU32 code;
} LWSWITCH_PRI_ERROR_LOG_TYPE;

typedef struct
{
    LwU32 addr;
    LwU32 data;
    LwU32 write;
    LwU32 dest;
    LwU32 subId;
    LwU32 errCode;
    LwU32 raw_data[4];
} LWSWITCH_PRI_TIMEOUT_ERROR_LOG_TYPE;

typedef struct
{
    LwU32 raw_pending;          // raw pending interrupt status
    LwU32 mask;                 // localized mask for current handler
    LwU32 raw_first;            // raw first register
    LwU32 raw_enable;           // raw mask/enable register
    LwU32 data[4];              // record of interrupt specific data
} LWSWITCH_INTERRUPT_LOG_TYPE;

typedef struct
{
    LwU32 data[16];
} LWSWITCH_RAW_ERROR_LOG_TYPE;

#define LWSWITCH_ERROR_NEXT_LOCAL_NUMBER(log) (log->error_total)

typedef struct
{
    LwU32   error_type;                     // LWSWITCH_ERR_*
    LwU64   local_error_num;                // Count of preceding errors (local error log)
    LwU64   global_error_num;               // Count of preceding errors (globally)
    LWSWITCH_ERROR_SRC_TYPE error_src;      // LWSWITCH_ERROR_SRC_*
    LWSWITCH_ERROR_SEVERITY_TYPE severity;  // LWSWITCH_ERROR_SEVERITY_*
    LwU32   instance;                       // Used for link# or subengine instance
    LwU32   subinstance;                    // Used for lane# or similar
    LwBool  error_resolved;
    LwU64   timer_count;                    // LwSwitch timer count
    LwU64   time;                           // Platform time, in ns
    LwU32   line;

    union
    {
        LwU64   address;
        LWSWITCH_PRI_ERROR_LOG_TYPE pri_error;
        LWSWITCH_PRI_TIMEOUT_ERROR_LOG_TYPE pri_timeout;
        LWSWITCH_INTERRUPT_LOG_TYPE intr;
        LWSWITCH_RAW_ERROR_LOG_TYPE raw;
    } data;
} LWSWITCH_ERROR_TYPE;

typedef struct
{
    LwU32               error_start;    // Start index within CB
    LwU32               error_count;    // Count of current errors in CB
    LwU64               error_total;    // Count of total errors logged
    LwU32               error_log_size; // CB size
    LWSWITCH_ERROR_TYPE *error_log;
    LwBool              overwritable;   // Old CB entries can be overwritten

} LWSWITCH_ERROR_LOG_TYPE;

//
// Helpful error logging wrappers
//

#define LWSWITCH_LOG_FATAL(_device, _errsrc, _errtype, _instance, _subinstance, _errresolved)\
    lwswitch_record_error(                                                              \
        _device,                                                                        \
        &(_device->log_FATAL_ERRORS),                                                   \
        LWSWITCH_ERR ## _errtype,                                                       \
        _instance, _subinstance,                                                        \
        LWSWITCH_ERROR_SRC ## _errsrc,                                                  \
        LWSWITCH_ERROR_SEVERITY_FATAL,                                                  \
        _errresolved,                                                                   \
        NULL, 0,                                                                        \
        __LINE__)

#define LWSWITCH_LOG_FATAL_DATA(_device, _errsrc, _errtype, _instance, _subinstance, _errresolved, _errdata)   \
    lwswitch_record_error(                                                              \
        _device,                                                                        \
        &(_device->log_FATAL_ERRORS),                                                   \
        LWSWITCH_ERR ## _errtype,                                                       \
        _instance, _subinstance,                                                        \
        LWSWITCH_ERROR_SRC ## _errsrc,                                                  \
        LWSWITCH_ERROR_SEVERITY_FATAL,                                                  \
        _errresolved,                                                                   \
        _errdata, sizeof(*_errdata),                                                    \
        __LINE__)


#define LWSWITCH_LOG_NONFATAL(_device, _errsrc, _errtype, _instance, _subinstance, _errresolved) \
    lwswitch_record_error(                                                              \
        _device,                                                                        \
        &(_device->log_NONFATAL_ERRORS),                                                \
        LWSWITCH_ERR ## _errtype,                                                       \
        _instance, _subinstance,                                                        \
        LWSWITCH_ERROR_SRC ## _errsrc,                                                  \
        LWSWITCH_ERROR_SEVERITY_NONFATAL,                                               \
        _errresolved,                                                                   \
        NULL, 0,                                                                        \
        __LINE__)

#define LWSWITCH_LOG_NONFATAL_DATA(_device, _errsrc, _errtype, _instance, _subinstance, _errresolved, _errdata)   \
    lwswitch_record_error(                                                              \
        _device,                                                                        \
        &(_device->log_NONFATAL_ERRORS),                                                \
        LWSWITCH_ERR ## _errtype,                                                       \
        _instance, _subinstance,                                                        \
        LWSWITCH_ERROR_SRC ## _errsrc,                                                  \
        LWSWITCH_ERROR_SEVERITY_NONFATAL,                                               \
        _errresolved,                                                                   \
        _errdata, sizeof(*_errdata),                                                    \
        __LINE__)

LWSWITCH_LWLINK_HW_ERROR lwswitch_translate_hw_error(LWSWITCH_ERR_TYPE type);
void lwswitch_translate_error(LWSWITCH_ERROR_TYPE *error_entry,
                              LWSWITCH_LWLINK_ARCH_ERROR *arch_error,
                              LWSWITCH_LWLINK_HW_ERROR *hw_error);
LwlStatus lwswitch_ctrl_get_errors(lwswitch_device *device,
                                   LWSWITCH_GET_ERRORS_PARAMS *p);

// Log correctable per-device error with data
#define LWSWITCH_REPORT_CORRECTABLE_DEVICE_DATA(_device, _logenum, _data, _fmt, ...)    \
    do                                                                                  \
    {                                                                                   \
        LWSWITCH_PRINT_SXID(_device, LWSWITCH_ERR ## _logenum,                          \
            "Correctable, " _fmt "\n", ## __VA_ARGS__ );                                \
        LWSWITCH_LOG_NONFATAL_DATA(_device, _HW, _logenum,                              \
            0, 0, LW_TRUE, _data);                                                      \
    } while(0)

// Log correctable per-link error with data
#define LWSWITCH_REPORT_CORRECTABLE_LINK_DATA(_device, _link, _logenum, _data, _fmt, ...) \
    do                                                                                    \
    {                                                                                     \
        LWSWITCH_PRINT_SXID(_device, LWSWITCH_ERR ## _logenum,                            \
            "Correctable, Link %02d " _fmt "\n", _link, ## __VA_ARGS__ );                 \
        LWSWITCH_LOG_NONFATAL_DATA(_device, _HW, _logenum,                                \
            _link, 0, LW_TRUE, _data);                                                    \
    } while(0)

// Log nonfatal per-link error
#define LWSWITCH_REPORT_NONFATAL_LINK(_device, _link, _logenum, _fmt, ...)              \
    do                                                                                  \
    {                                                                                   \
        LWSWITCH_PRINT_SXID(_device, LWSWITCH_ERR ## _logenum,                          \
            "Non-fatal, Link %02d " _fmt "\n", _link, ## __VA_ARGS__ );                 \
        LWSWITCH_LOG_NONFATAL(_device, _HW, _logenum,                                   \
            _link, 0, LW_FALSE);                                                        \
    } while(0)

#endif //_ERROR_LWSWITCH_H_
