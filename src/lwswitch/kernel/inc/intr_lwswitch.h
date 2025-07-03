/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _INTR_LWSWITCH_H_
#define _INTR_LWSWITCH_H_

#include "error_lwswitch.h"

//
// Wrapper to track interrupt servicing
//
#define LWSWITCH_UNHANDLED_INIT(val) (unhandled = (val))
#define LWSWITCH_HANDLED(mask)       (unhandled &= ~(mask))

#define LWSWITCH_UNHANDLED_CHECK(_device, _unhandled)                      \
    do                                                                     \
    {                                                                      \
        if (_unhandled)                                                    \
        {                                                                  \
            LWSWITCH_PRINT(_device, ERROR,                                 \
                        "%s:%d unhandled interrupt! %x\n",                 \
                        __FUNCTION__, __LINE__, _unhandled);               \
            LWSWITCH_PRINT_SXID(_device,                                   \
                  LWSWITCH_ERR_HW_HOST_UNHANDLED_INTERRUPT,                \
                  "Fatal, unhandled interrupt\n");                         \
            LWSWITCH_LOG_FATAL_DATA(_device, _HW,                          \
                _HW_HOST_UNHANDLED_INTERRUPT, 0, 0, LW_FALSE, &_unhandled);\
        }                                                                  \
    } while(0)

//
// Wrappers for basic leaf interrupt handling
//
#define LWSWITCH_PENDING(_bit) ((bit = (_bit)) && (pending & (_bit)))
#define LWSWITCH_FIRST()       (bit & report.raw_first) ? " (First)" : ""

//
// Report/log error interrupt helper.
//

//
// Print an intermediate point (non-leaf) in the interrupt tree.
//
#define LWSWITCH_REPORT_TREE(_logenum)                                        \
    do                                                                        \
    {                                                                         \
        LWSWITCH_PRINT(device, ERROR, "Intermediate, Link %02d \n", link);    \
    } while(0)

// Log correctable errors
#define LWSWITCH_REPORT_CORRECTABLE(_logenum, _str)                            \
    do                                                                         \
    {                                                                          \
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR ## _logenum,                  \
            "Correctable, Link %02d %s%s\n", link, _str, LWSWITCH_FIRST());    \
        LWSWITCH_LOG_NONFATAL_DATA(device, _HW, _logenum,                      \
             link, 0, LW_TRUE, &report);                                       \
        if (lwswitch_lib_notify_client_events(device,                          \
            LWSWITCH_DEVICE_EVENT_NONFATAL) != LWL_SUCCESS)                    \
        {                                                                      \
            LWSWITCH_PRINT(device, ERROR, "%s: Failed to notify event\n",      \
                           __FUNCTION__);                                      \
        }                                                                      \
    } while(0)

// Log uncorrectable error that is not fatal to the fabric
#define LWSWITCH_REPORT_NONFATAL(_logenum, _str)                             \
    do                                                                       \
    {                                                                        \
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR ## _logenum,                \
            "Non-fatal, Link %02d %s%s\n", link, _str, LWSWITCH_FIRST());    \
        LWSWITCH_LOG_NONFATAL_DATA(device, _HW, _logenum,                    \
            link, 0, LW_FALSE, &report);                                     \
        if (lwswitch_lib_notify_client_events(device,                        \
            LWSWITCH_DEVICE_EVENT_NONFATAL) != LWL_SUCCESS)                  \
        {                                                                    \
            LWSWITCH_PRINT(device, ERROR, "%s: Failed to notify event\n",    \
                           __FUNCTION__);                                    \
        }                                                                    \
    } while(0)

// Log uncorrectable error that is fatal to the fabric
#define LWSWITCH_REPORT_FATAL(_logenum, _str, device_fatal)              \
    do                                                                   \
    {                                                                    \
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR ## _logenum,            \
            "Fatal, Link %02d %s%s\n", link, _str, LWSWITCH_FIRST());    \
        LWSWITCH_LOG_FATAL_DATA(device, _HW, _logenum,                   \
            link, 0, LW_FALSE, &report);                                 \
        lwswitch_set_fatal_error(device, device_fatal, link);            \
        if (lwswitch_lib_notify_client_events(device,                    \
            LWSWITCH_DEVICE_EVENT_FATAL) != LWL_SUCCESS)                 \
        {                                                                \
            LWSWITCH_PRINT(device, ERROR, "%s: Failed to notify event\n",\
                           __FUNCTION__);                                \
        }                                                                \
    } while(0)

#define LWSWITCH_REPORT_PRI_ERROR_NONFATAL(_logenum, _str, instance, chiplet, err_data) \
    do                                                                                         \
    {                                                                                          \
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR ## _logenum,                                  \
            "Non-fatal, %s, instance=%d, chiplet=%d\n", _str, instance, chiplet);              \
        LWSWITCH_LOG_NONFATAL_DATA(device, _HW, _logenum,                                      \
            instance, chiplet, LW_FALSE, &err_data);                                            \
        if (lwswitch_lib_notify_client_events(device,                                          \
            LWSWITCH_DEVICE_EVENT_NONFATAL) != LWL_SUCCESS)                                    \
        {                                                                                      \
            LWSWITCH_PRINT(device, ERROR, "%s: Failed to notify event\n",                      \
                           __FUNCTION__);                                                      \
        }                                                                                      \
    } while(0)

#define LWSWITCH_REPORT_PRI_ERROR_FATAL(_logenum, _str, device_fatal, instance, chiplet, err_data) \
    do                                                                                      \
    {                                                                                       \
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR ## _logenum,                               \
            "Fatal, %s, instance=%d, chiplet=%d\n", _str, instance, chiplet);               \
        LWSWITCH_LOG_FATAL_DATA(device, _HW, _logenum,                                      \
            instance, chiplet, LW_FALSE, &err_data);                                         \
        lwswitch_set_fatal_error(device, device_fatal, 0);                                  \
        if (lwswitch_lib_notify_client_events(device,                                       \
            LWSWITCH_DEVICE_EVENT_FATAL) != LWL_SUCCESS)                                    \
        {                                                                                   \
            LWSWITCH_PRINT(device, ERROR, "%s: Failed to notify event\n",                   \
                           __FUNCTION__);                                                   \
        }                                                                                   \
    } while(0)

/*
 * Automatically determine if error is fatal to the fabric based on
 * if it is contained and will lock the port.
 */
#define LWSWITCH_REPORT_CONTAIN(_logenum, _str, device_fatal)       \
    do                                                              \
    {                                                               \
        if (bit & contain)                                          \
        {                                                           \
            LWSWITCH_REPORT_FATAL(_logenum, _str, device_fatal);    \
        }                                                           \
        else                                                        \
        {                                                           \
            LWSWITCH_REPORT_NONFATAL(_logenum, _str);               \
        }                                                           \
    } while (0)

/*
 * REPORT_*_DATA macros - optionally log data record for additional HW state. This
 * is typically a captured packet, but there are a few other cases.
 *
 * Most interrupt controllers only latch additional data for errors tagged as first.
 * For those cases use _FIRST to only log the data record when it is accurate.  If
 * two errors are detected in the same cycle, they will both be set in first.
 */
#define LWSWITCH_REPORT_DATA(_logenum, _data) \
    LWSWITCH_LOG_NONFATAL_DATA(device, _HW, _logenum, link, 0, LW_TRUE, &_data)

#define LWSWITCH_REPORT_DATA_FIRST(_logenum, _data) \
    do                                              \
    {                                               \
        if (report.raw_first & bit)                 \
        {                                           \
            LWSWITCH_REPORT_DATA(_logenum, _data);  \
        }                                           \
    } while(0)

#define LWSWITCH_REPORT_CONTAIN_DATA(_logenum, _data)               \
    do                                                              \
    {                                                               \
        if (bit & contain)                                          \
        {                                                           \
            LWSWITCH_LOG_FATAL_DATA(device, _HW, _logenum, link,    \
                                     0, LW_FALSE, &_data);          \
        }                                                           \
        else                                                        \
        {                                                           \
            LWSWITCH_LOG_NONFATAL_DATA(device, _HW, _logenum, link, \
                                       0, LW_FALSE, &data);         \
        }                                                           \
    } while(0)

#define LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_logenum, _data) \
    do                                                      \
    {                                                       \
        if (bit & report.raw_first)                         \
        {                                                   \
            LWSWITCH_REPORT_CONTAIN_DATA(_logenum, _data);  \
        }                                                   \
    } while(0)

#endif //_INTR_LWSWITCH_H_
