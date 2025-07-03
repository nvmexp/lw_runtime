/*
 * Copyright (c) 2001-2021, LWPU CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _cl0073_h_
#define _cl0073_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  LW04_DISPLAY_COMMON                                       (0x00000073)

/* event values */
#define LW0073_NOTIFIERS_SW                                        (0)
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW0073_NOTIFIERS_SRM_VALIDATION_COMPLETE                   (1)
#define LW0073_NOTIFIERS_STREAM_VALIDATION_COMPLETE                (2)
#define LW0073_NOTIFIERS_KSV_VALIDATION_COMPLETE                   (3)
#define LW0073_NOTIFIERS_HDCP_KSV_LIST_READY                       (4)

#endif
#define LW0073_NOTIFIERS_MAXCOUNT                                  (5)


#define LW0073_NOTIFICATION_STATUS_IN_PROGRESS              (0x8000)
#define LW0073_NOTIFICATION_STATUS_BAD_ARGUMENT             (0x4000)
#define LW0073_NOTIFICATION_STATUS_ERROR_ILWALID_STATE      (0x2000)
#define LW0073_NOTIFICATION_STATUS_ERROR_STATE_IN_USE       (0x1000)
#define LW0073_NOTIFICATION_STATUS_DONE_SUCCESS             (0x0000)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
/*
 * Stream validation status values
 *
 *   STATUS_DONE_SUCCESS
 *      validation passed
 *   STATUS_FALCON_FAIL
 *      indicates PMU/DPMU internal error
 *   STATUS_STREAM_ILWALID
 *      indicates stream invalid
 */
#define LW0073_NOTIFICATION_STREAM_VALIDATION_COMPLETE_STATUS_DONE_SUCCESS   (0)
#define LW0073_NOTIFICATION_STREAM_VALIDATION_COMPLETE_STATUS_FALCON_FAIL    (1)
#define LW0073_NOTIFICATION_STREAM_VALIDATION_COMPLETE_STATUS_STREAM_ILWALID (7)


/*
 * KSVList validation status values
 *
 *   STATUS_DONE_SUCCESS
 *      validation passed
 *   STATUS_FALCON_FAIL
 *      indicates PMU/DPMU internal error
 *   STATUS_KSV_ERROR
 *      indicates KSV list has invalid entries or zero entries.
 *   STATUS_SRM_ERROR
 *      indicates SRM is invalid or has incorrect Format
 *   STATUS_REVOKED
 *      indicates it has some revoked device
 */
#define LW0073_NOTIFICATION_KSVLIST_VALIDATION_COMPLETE_STATUS_DONE_SUCCESS   (0)
#define LW0073_NOTIFICATION_KSVLIST_VALIDATION_COMPLETE_STATUS_FALCON_FAIL    (1)
#define LW0073_NOTIFICATION_KSVLIST_VALIDATION_COMPLETE_STATUS_KSV_ERROR      (2)
#define LW0073_NOTIFICATION_KSVLIST_VALIDATION_COMPLETE_STATUS_SRM_ERROR      (3)
#define LW0073_NOTIFICATION_KSVLIST_VALIDATION_COMPLETE_STATUS_REVOKED        (4)
#define LW0073_NOTIFICATION_KSVLIST_VALIDATION_SUBMISSION_FAILED_REP_ERROR    (5)
#define LW0073_NOTIFICATION_KSVLIST_VALIDATION_SUBMISSION_FAILED_RM_ERROR     (6)

#endif

/* pio method data structure */
typedef volatile struct _cl0073_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw073Typedef, Lw04DisplayCommon;
#define  LW073_TYPEDEF                                             Lw04DisplayCommon

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl0073_h_ */
