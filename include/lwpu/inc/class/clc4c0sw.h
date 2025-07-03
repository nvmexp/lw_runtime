/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cl_volta_compute_b_sw_h_
#define _cl_volta_compute_b_sw_h_

/* This file is *not* auto-generated. */

/* LwNotification[] elements */
#define LWC4C0_NOTIFIERS_NOTIFY                                     (0)
#define LWC4C0_NOTIFIERS_DEBUG_INTR                                 (1)
#define LWC4C0_NOTIFIERS_BPT_INT                                    (2)
#define LWC4C0_NOTIFIERS_CILP_INT                                   (3)
#define LWC4C0_NOTIFIERS_PREEMPTION_STARTED                         (4)
#define LWC4C0_NOTIFIERS_MAXCOUNT                                   (5)

/* LwNotification[] fields and values */
#define LWC4C0_NOTIFICATION_STATUS_IN_PROGRESS                      (0x8000)
#define LWC4C0_NOTIFICATION_STATUS_DONE_SUCCESS                     (0x0000)

#ifndef LWC4C0_SET_CB_BASE
#ifdef LWC4C0_SET_RESERVED_SW_METHOD00
/* VPR=0 RM assigns global base, VPR=1 RM assigns VPR global base from _ADDR which is bits 40:12 */
#define LWC4C0_SET_CB_BASE                                          LWC4C0_SET_RESERVED_SW_METHOD00
#define LWC4C0_SET_CB_BASE_ADDR                                     27:0
#define LWC4C0_SET_CB_BASE_ADDR_SHIFT                               12
#define LWC4C0_SET_CB_BASE_VPR                                      31:31
#define LWC4C0_SET_CB_BASE_VPR_FALSE                                (0x00000000)
#define LWC4C0_SET_CB_BASE_VPR_TRUE                                 (0x00000001)
#endif
#endif

#endif /* _cl_volta_compute_b_sw_h_ */
