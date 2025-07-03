/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2015-2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cl_pascal_compute_b_sw_h_
#define _cl_pascal_compute_b_sw_h_

/* This file is *not* auto-generated. */

/* LwNotification[] elements */
#define LWC1C0_NOTIFIERS_NOTIFY                                     (0)
#define LWC1C0_NOTIFIERS_DEBUG_INTR                                 (1)
#define LWC1C0_NOTIFIERS_BPT_INT                                    (2)
#define LWC1C0_NOTIFIERS_CILP_INT                                   (3)
#define LWC1C0_NOTIFIERS_PREEMPTION_STARTED                         (4)
#define LWC1C0_NOTIFIERS_MAXCOUNT                                   (5)

/* LwNotification[] fields and values */
#define LWC1C0_NOTIFICATION_STATUS_IN_PROGRESS                      (0x8000)
#define LWC1C0_NOTIFICATION_STATUS_DONE_SUCCESS                     (0x0000)

#ifndef LWC1C0_SET_CB_BASE
#ifdef LWC1C0_SET_RESERVED_SW_METHOD00
/* VPR=0 RM assigns global base, VPR=1 RM assigns VPR global base from _ADDR which is bits 40:12 */
#define LWC1C0_SET_CB_BASE                                          LWC1C0_SET_RESERVED_SW_METHOD00
#define LWC1C0_SET_CB_BASE_ADDR                                     27:0
#define LWC1C0_SET_CB_BASE_ADDR_SHIFT                               12
#define LWC1C0_SET_CB_BASE_VPR                                      31:31
#define LWC1C0_SET_CB_BASE_VPR_FALSE                                (0x00000000)
#define LWC1C0_SET_CB_BASE_VPR_TRUE                                 (0x00000001)
#endif
#endif

/* Set RD COALESCE for current context */
#ifndef LWC1C0_SET_RD_COALESCE
#ifdef  LWC1C0_SET_RESERVED_SW_METHOD02
#define LWC1C0_SET_RD_COALESCE                                      LWC1C0_SET_RESERVED_SW_METHOD02
#define LWC1C0_SET_RD_COALESCE_LG_SU                                0:0
#define LWC1C0_SET_RD_COALESCE_LG_SU_DISABLE                        (0x00000000)
#define LWC1C0_SET_RD_COALESCE_LG_SU_ENABLE                         (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_BES_CROP_DEBUG4_CLAMP_FP_BLEND */
#ifndef LWC1C0_SET_BES_CROP_DEBUG4
#ifdef  LWC1C0_SET_RESERVED_SW_METHOD03
#define LWC1C0_SET_BES_CROP_DEBUG4                                  LWC1C0_SET_RESERVED_SW_METHOD03
#define LWC1C0_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND                   0:0
#define LWC1C0_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND_TO_INF            (0x00000000)
#define LWC1C0_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND_TO_MAXVAL         (0x00000001)
#endif
#endif

/* Toggle LW_PGRAPH_PRI_BES_CROP_DEBUG3 fields that control blend optimizations (bug 1942454). */
#ifndef LWC1C0_SET_BES_CROP_DEBUG3
#ifdef  LWC1C0_SET_RESERVED_SW_METHOD08
#define LWC1C0_SET_BES_CROP_DEBUG3                                  LWC1C0_SET_RESERVED_SW_METHOD08
#define LWC1C0_SET_BES_CROP_DEBUG3_BLENDOPT                         0:0
#define LWC1C0_SET_BES_CROP_DEBUG3_BLENDOPT_KEEP_ZBC                (0x00000000)
#define LWC1C0_SET_BES_CROP_DEBUG3_BLENDOPT_ENABLE                  (0x00000001)
#endif
#endif

#endif /* _cl_pascal_compute_b_sw_h_ */
