/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "error_lwswitch.h"

#define LWSWITCH_DATE_LEN    64

//
// Error logging
//
static void
_lwswitch_dump_error_entry
(
    lwswitch_device *device,
    LwU32   error_count,
    LWSWITCH_ERROR_TYPE *error_entry
)
{
    if ((error_entry != NULL) &&
        (error_entry->error_src == LWSWITCH_ERROR_SRC_HW))
    {
        LWSWITCH_PRINT_SXID(device, error_entry->error_type,
            "Severity %d Engine instance %02d Sub-engine instance %02d\n",
            error_entry->severity, error_entry->instance, error_entry->subinstance);

        LWSWITCH_PRINT_SXID(device, error_entry->error_type,
            "Data {0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x}\n",
            error_entry->data.raw.data[0], error_entry->data.raw.data[1],
            error_entry->data.raw.data[2], error_entry->data.raw.data[3],
            error_entry->data.raw.data[4], error_entry->data.raw.data[5],
            error_entry->data.raw.data[6], error_entry->data.raw.data[7]);

        if ((error_entry->data.raw.data[ 8] != 0) ||
            (error_entry->data.raw.data[ 9] != 0) ||
            (error_entry->data.raw.data[10] != 0) ||
            (error_entry->data.raw.data[11] != 0) ||
            (error_entry->data.raw.data[12] != 0) ||
            (error_entry->data.raw.data[13] != 0) ||
            (error_entry->data.raw.data[14] != 0) ||
            (error_entry->data.raw.data[15] != 0))

        {
            LWSWITCH_PRINT_SXID(device, error_entry->error_type,
                "Data {0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x}\n",
                error_entry->data.raw.data[ 8], error_entry->data.raw.data[ 9],
                error_entry->data.raw.data[10], error_entry->data.raw.data[11],
                error_entry->data.raw.data[12], error_entry->data.raw.data[13],
                error_entry->data.raw.data[14], error_entry->data.raw.data[15]);
        }
    }
}

//
// Construct an error log
//
// If error_log_size > 0 a cirlwlar buffer is created to record errors
//
LwlStatus
lwswitch_construct_error_log
(
    LWSWITCH_ERROR_LOG_TYPE *errors,
    LwU32 error_log_size,
    LwBool overwritable
)
{
    LwlStatus retval = LWL_SUCCESS;

    LWSWITCH_ASSERT(errors != NULL);

    errors->error_start = 0;
    errors->error_count = 0;
    errors->error_total = 0;
    errors->error_log_size = 0;
    errors->error_log = NULL;
    errors->overwritable = overwritable;

    if (error_log_size > 0)
    {
        errors->error_log = lwswitch_os_malloc(error_log_size * sizeof(LWSWITCH_ERROR_TYPE));
    }

    if (errors->error_log != NULL)
    {
        errors->error_log_size = error_log_size;
        lwswitch_os_memset(errors->error_log, 0, errors->error_log_size * sizeof(LWSWITCH_ERROR_TYPE));
    }

    if (error_log_size != errors->error_log_size)
    {
        retval = -LWL_NO_MEM;
    }

    return retval;
}

//
// Destroy an error log
//
void
lwswitch_destroy_error_log
(
    lwswitch_device *device,
    LWSWITCH_ERROR_LOG_TYPE *errors
)
{
    if (errors == NULL)
        return;

    errors->error_start = 0;
    errors->error_count = 0;
    //errors->error_total = 0;       // Don't reset total count of errors logged
    errors->error_log_size = 0;

    if (errors->error_log != NULL)
    {
        lwswitch_os_free(errors->error_log);
        errors->error_log = NULL;
    }
}

void
lwswitch_record_error
(
    lwswitch_device             *device,
    LWSWITCH_ERROR_LOG_TYPE     *errors,
    LwU32                        error_type, // LWSWITCH_ERR_*
    LwU32                        instance,
    LwU32                        subinstance,
    LWSWITCH_ERROR_SRC_TYPE      error_src,  // LWSWITCH_ERROR_SRC_*
    LWSWITCH_ERROR_SEVERITY_TYPE severity,   // LWSWITCH_ERROR_SEVERITY_*
    LwBool                       error_resolved,
    void                        *data,
    LwU32                        data_size,
    LwU32                        line
)
{
    LwU32 idx_error;

    LWSWITCH_ASSERT(errors != NULL);
    LWSWITCH_ASSERT(data_size <= sizeof(errors->error_log[idx_error].data));

    // If no error log has been created, don't log it.
    if ((errors->error_log_size != 0) && (errors->error_log != NULL))
    {
        idx_error = (errors->error_start + errors->error_count) % errors->error_log_size;

        if (errors->error_count == errors->error_log_size)
        {
            // Error ring buffer already full.
            if (errors->overwritable)
            {
                errors->error_start = (errors->error_start + 1) % errors->error_log_size;
            }
            else
            {
                // Return: ring buffer full
                return;
            }
        }
        else
        {
            errors->error_count++;
        }

        // Log error info
        errors->error_log[idx_error].error_type = error_type;
        errors->error_log[idx_error].instance   = instance;
        errors->error_log[idx_error].subinstance = subinstance;
        errors->error_log[idx_error].error_src  = error_src;
        errors->error_log[idx_error].severity   = severity;
        errors->error_log[idx_error].error_resolved = error_resolved;
        errors->error_log[idx_error].line       = line;

        // Log tracking info
        errors->error_log[idx_error].timer_count = lwswitch_hw_counter_read_counter(device);
        errors->error_log[idx_error].time = lwswitch_os_get_platform_time();
        errors->error_log[idx_error].local_error_num  = errors->error_total;
        errors->error_log[idx_error].global_error_num = device->error_total;

        // Copy ancillary data blob
        lwswitch_os_memset(&errors->error_log[idx_error].data, 0, sizeof(errors->error_log[idx_error].data));
        if ((data != NULL) && (data_size > 0))
        {
            lwswitch_os_memcpy(&errors->error_log[idx_error].data, data, data_size);
        }

        _lwswitch_dump_error_entry(device, idx_error, &errors->error_log[idx_error]);
    }
    errors->error_total++;
    device->error_total++;
}

//
// Discard N errors from the specified log
//

void
lwswitch_discard_errors
(
    LWSWITCH_ERROR_LOG_TYPE *errors,
    LwU32 error_discard_count
)
{
    error_discard_count = LW_MIN(error_discard_count, errors->error_count);
    errors->error_start = (errors->error_start+error_discard_count) % errors->error_log_size;
    errors->error_count -= error_discard_count;
}

//
// Retrieve an error entry by index.
// 0 = oldest error
// Out-of-range index does not return an error, but does return an error of type "NO_ERROR"
// error_count returns how many errors in the error log
//

void
lwswitch_get_error
(
    lwswitch_device *device,
    LWSWITCH_ERROR_LOG_TYPE *errors,
    LWSWITCH_ERROR_TYPE *error_entry,
    LwU32   error_idx,
    LwU32   *error_count
)
{
    LWSWITCH_ASSERT(errors != NULL);

    if (error_entry != NULL)
    {
        if (error_idx >= errors->error_count)
        {
            // Index out-of-range
            lwswitch_os_memset(error_entry, 0, sizeof(*error_entry));
            error_entry->error_type = 0;
            error_entry->instance   = 0;
            error_entry->subinstance = 0;
            error_entry->local_error_num  = errors->error_total;
            error_entry->global_error_num  = ((device == NULL) ? 0 : device->error_total);
            error_entry->error_src  = LWSWITCH_ERROR_SRC_NONE;
            error_entry->severity   = LWSWITCH_ERROR_SEVERITY_NONFATAL;
            error_entry->error_resolved = LW_TRUE;
            error_entry->line = 0;
            error_entry->timer_count = 
                ((device == NULL) ? 0 : lwswitch_hw_counter_read_counter(device));
            error_entry->time = lwswitch_os_get_platform_time();
        }
        else
        {
            *error_entry = errors->error_log[(errors->error_start + error_idx) % errors->error_log_size];
        }
    }

    if (error_count)
    {
        *error_count = errors->error_count;
    }
}


//
// Retrieve the oldest logged error entry.
// Optionally remove the error entry after reading
// error_count returns how many remaining errors in the error log
//

void
lwswitch_get_next_error
(
    lwswitch_device *device,
    LWSWITCH_ERROR_LOG_TYPE *errors,
    LWSWITCH_ERROR_TYPE *error_entry,
    LwU32   *error_count,
    LwBool  remove_from_list
)
{
    lwswitch_get_error(device, errors, error_entry, 0, error_count);

    // Optionally remove the error from the log
    if (remove_from_list)
    {
        lwswitch_discard_errors(errors, 1);
    }
}

LWSWITCH_LWLINK_HW_ERROR
lwswitch_translate_hw_error
(
    LWSWITCH_ERR_TYPE type
)
{
    if ((type >= LWSWITCH_ERR_HW_NPORT_INGRESS) &&
        (type <  LWSWITCH_ERR_HW_NPORT_INGRESS_LAST))
    {
        return LWSWITCH_LWLINK_HW_INGRESS;
    }
    else if ((type >= LWSWITCH_ERR_HW_NPORT_EGRESS) &&
             (type <  LWSWITCH_ERR_HW_NPORT_EGRESS_LAST))
    {
        return LWSWITCH_LWLINK_HW_EGRESS;
    }
    else if ((type >= LWSWITCH_ERR_HW_NPORT_FSTATE) &&
             (type <  LWSWITCH_ERR_HW_NPORT_FSTATE_LAST))
    {
        return LWSWITCH_LWLINK_HW_FSTATE;
    }
    else if ((type >= LWSWITCH_ERR_HW_NPORT_TSTATE) &&
             (type <  LWSWITCH_ERR_HW_NPORT_TSTATE_LAST))
    {
        return LWSWITCH_LWLINK_HW_TSTATE;
    }
    else if ((type >= LWSWITCH_ERR_HW_NPORT_ROUTE) &&
             (type <  LWSWITCH_ERR_HW_NPORT_ROUTE_LAST))
    {
        return LWSWITCH_LWLINK_HW_ROUTE;
    }
    else if ((type >= LWSWITCH_ERR_HW_NPORT) &&
             (type <  LWSWITCH_ERR_HW_NPORT_LAST))
    {
        return LWSWITCH_LWLINK_HW_NPORT;
    }
    else if ((type >= LWSWITCH_ERR_HW_LWLCTRL) &&
             (type <  LWSWITCH_ERR_HW_LWLCTRL_LAST))
    {
        return LWSWITCH_LWLINK_HW_LWLCTRL;
    }
    else if ((type >= LWSWITCH_ERR_HW_LWLIPT) &&
             (type <  LWSWITCH_ERR_HW_LWLIPT_LAST))
    {
        return LWSWITCH_LWLINK_HW_LWLIPT;
    }
    else if ((type >= LWSWITCH_ERR_HW_LWLTLC) &&
             (type <  LWSWITCH_ERR_HW_LWLTLC_LAST))
    {
        return LWSWITCH_LWLINK_HW_LWLTLC;
    }
    else if ((type >= LWSWITCH_ERR_HW_DLPL) &&
            (type <  LWSWITCH_ERR_HW_DLPL_LAST))
    {
        return LWSWITCH_LWLINK_HW_DLPL;
    }
    else if ((type >= LWSWITCH_ERR_HW_AFS) &&
             (type <  LWSWITCH_ERR_HW_AFS_LAST))
    {
        return LWSWITCH_LWLINK_HW_AFS;
    }
    else if ((type >= LWSWITCH_ERR_HW_HOST) &&
             (type <  LWSWITCH_ERR_HW_HOST_LAST))
    {
        return LWSWITCH_LWLINK_HW_HOST;
    }
    else if ((type >= LWSWITCH_ERR_HW_MINION) &&
             (type <  LWSWITCH_ERR_HW_MINION_LAST))
    {
        return LWSWITCH_LWLINK_HW_MINION;
    }
    else if ((type >= LWSWITCH_ERR_HW_NXBAR) &&
             (type <  LWSWITCH_ERR_HW_NXBAR_LAST))
    {
        return LWSWITCH_LWLINK_HW_NXBAR;
    }
    else if ((type >= LWSWITCH_ERR_HW_NPORT_SOURCETRACK) &&
             (type < LWSWITCH_ERR_HW_NPORT_SOURCETRACK_LAST))
    {
        return LWSWITCH_LWLINK_HW_SOURCETRACK;
    }
    else if ((type >= LWSWITCH_ERR_HW_LWLIPT_LNK) &&
             (type < LWSWITCH_ERR_HW_LWLIPT_LNK_LAST))
    {
        return LWSWITCH_ERR_HW_LWLIPT_LNK;
    }
    else if ((type >= LWSWITCH_ERR_HW_SOE) &&
             (type < LWSWITCH_ERR_HW_SOE_LAST))
    {
        return LWSWITCH_ERR_HW_SOE;
    }
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    else if ((type >= LWSWITCH_ERR_HW_CCI) &&
             (type < LWSWITCH_ERR_HW_CCI_LAST))
    {
        return LWSWITCH_ERR_HW_CCI;
    }
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    else
    {
        // Update this assert after adding a new translation entry above
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
        ct_assert(LWSWITCH_ERR_HW_CCI_LAST == (LWSWITCH_ERR_LAST - 1));
#else   
        ct_assert(LWSWITCH_ERR_HW_SOE_LAST == (LWSWITCH_ERR_LAST - 1));
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

        LWSWITCH_PRINT(NULL, ERROR,
            "%s: Undefined error type\n", __FUNCTION__);
        LWSWITCH_ASSERT(0);
        return LWSWITCH_LWLINK_HW_GENERIC;
    }
}

static LWSWITCH_LWLINK_ARCH_ERROR
_lwswitch_translate_arch_error
(
    LWSWITCH_ERROR_TYPE *error_entry
)
{
    if (error_entry->severity == LWSWITCH_ERROR_SEVERITY_FATAL)
    {
        return LWSWITCH_LWLINK_ARCH_ERROR_HW_FATAL;
    }
    else if (error_entry->severity == LWSWITCH_ERROR_SEVERITY_NONFATAL)
    {
        if (error_entry->error_resolved)
        {
            return LWSWITCH_LWLINK_ARCH_ERROR_HW_CORRECTABLE;
        }
        else
        {
            return LWSWITCH_LWLINK_ARCH_ERROR_HW_UNCORRECTABLE;
        }
    }

    return LWSWITCH_LWLINK_ARCH_ERROR_GENERIC;
}

void
lwswitch_translate_error
(
    LWSWITCH_ERROR_TYPE         *error_entry,
    LWSWITCH_LWLINK_ARCH_ERROR  *arch_error,
    LWSWITCH_LWLINK_HW_ERROR    *hw_error
)
{
    LWSWITCH_ASSERT(error_entry != NULL);

    if (arch_error)
    {
        *arch_error = LWSWITCH_LWLINK_ARCH_ERROR_NONE;
    }

    if (hw_error)
    {
        *hw_error = LWSWITCH_LWLINK_HW_ERROR_NONE;
    }

    if (error_entry->error_src == LWSWITCH_ERROR_SRC_HW)
    {
        if (arch_error)
        {
            *arch_error = _lwswitch_translate_arch_error(error_entry);
        }

        if (hw_error)
        {
            *hw_error = lwswitch_translate_hw_error(error_entry->error_type);
        }
    }
    else
    {
        LWSWITCH_PRINT(NULL, ERROR,
            "%s: Undefined error source\n", __FUNCTION__);
        LWSWITCH_ASSERT(0);
    }
}

LwlStatus
lwswitch_ctrl_get_errors
(
    lwswitch_device *device,
    LWSWITCH_GET_ERRORS_PARAMS *p
)
{
    LwU32 index = 0;
    LwU32 count = 0;
    LWSWITCH_ERROR_LOG_TYPE *error_log;
    LWSWITCH_ERROR_TYPE error;

    switch (p->errorType)
    {
        case LWSWITCH_ERROR_SEVERITY_FATAL:
            error_log = &device->log_FATAL_ERRORS;
            break;
        case LWSWITCH_ERROR_SEVERITY_NONFATAL:
            error_log = &device->log_NONFATAL_ERRORS;
            break;
        default:
            return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(p->error, 0, sizeof(LWSWITCH_ERROR) *
                       LWSWITCH_ERROR_COUNT_SIZE);
    p->nextErrorIndex = LWSWITCH_ERROR_NEXT_LOCAL_NUMBER(error_log);
    p->errorCount = 0;

    // If there is nothing to do, return.
    lwswitch_get_error(device, error_log, &error, index, &count);
    if (count == 0)
    {
        return LWL_SUCCESS;
    }

    //
    // If the error's local_error_num is smaller than the errorIndex
    // passed in by the client, fast-forward index by the difference.
    // This will skip over errors that were previously read by the client.
    //
    if (error.local_error_num < p->errorIndex)
    {
        index = (LwU32) (p->errorIndex - error.local_error_num);
    }

   // If there is nothing to do after fast-forwarding, return.
   if (index >= count)
   {
      return LWL_SUCCESS;
   }

    while ((p->errorCount < LWSWITCH_ERROR_COUNT_SIZE) && (index < count))
    {
        // Get the next error to consider from the log
        lwswitch_get_error(device, error_log, &error, index, NULL);

        p->error[p->errorCount].error_value = error.error_type;
        p->error[p->errorCount].error_src = error.error_src;
        p->error[p->errorCount].instance = error.instance;
        p->error[p->errorCount].subinstance = error.subinstance;
        p->error[p->errorCount].time = error.time;
        p->error[p->errorCount].error_resolved = error.error_resolved;
        p->errorCount++;
        index++;
    }

    p->errorIndex = error.local_error_num + 1;

    return LWL_SUCCESS;
}
