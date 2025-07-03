/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2014-2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cl_pascal_b_sw_h_
#define _cl_pascal_b_sw_h_

/* This file is *not* auto-generated. */

/* LwNotification[] elements */
#define LWC197_NOTIFIERS_NOTIFY                                     (0)
#define LWC197_NOTIFIERS_NONSTALL                                   (1)
#define LWC197_NOTIFIERS_SET_MEMORY_SURFACE_ATTR                    (0)
#define LWC197_NOTIFIERS_DEBUG_INTR                                 (2)
#define LWC197_NOTIFIERS_MAXCOUNT                                   (3)

/* LwNotification[] fields and values */
#define LWC197_NOTIFICATION_STATUS_IN_PROGRESS                      (0x8000)
#define LWC197_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
#define LWC197_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT               (0x2000)
#define LWC197_NOTIFICATION_STATUS_ERROR_ILWALID_STATE              (0x1000)
#define LWC197_NOTIFICATION_STATUS_ERROR_STATE_IN_USE               (0x0800)
#define LWC197_NOTIFICATION_STATUS_DONE_SUCCESS                     (0x0000)

#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_STATUS            0:0
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_STATUS_FAILED     (0x0000)
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_STATUS_SUCCESS    (0x0001)
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_COMPR             2:2
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_COMPR_DISABLED    (0x0000)
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_COMPR_ENABLED     (0x0001)
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_SHARED            4:4
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_SHARED_DISABLED   (0x0000)
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_SHARED_ENABLED    (0x0001)
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_LOWER             5:5
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_LOWER_DISABLED    (0x0000)
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_LOWER_ENABLED     (0x0001)
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_TILE_FORMAT       11:8
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_TILE_REGION       15:12
#define LWC197_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_COMPR_COVG        31:16

#ifndef LWC197_SET_CB_BASE
#ifdef LWC197_SET_RESERVED_SW_METHOD00
/* VPR=0 RM assigns global base, VPR=1 RM assigns VPR global base from _ADDR which is bits 40:12 */
#define LWC197_SET_CB_BASE                                          LWC197_SET_RESERVED_SW_METHOD00
#define LWC197_SET_CB_BASE_ADDR                                     27:0
#define LWC197_SET_CB_BASE_ADDR_SHIFT                               12
#define LWC197_SET_CB_BASE_VPR                                      31:31
#define LWC197_SET_CB_BASE_VPR_FALSE                                (0x00000000)
#define LWC197_SET_CB_BASE_VPR_TRUE                                 (0x00000001)
#endif
#endif

/* Set RD COALESCE for current context */
#ifndef LWC197_SET_RD_COALESCE
#ifdef  LWC197_SET_RESERVED_SW_METHOD02
#define LWC197_SET_RD_COALESCE                                      LWC197_SET_RESERVED_SW_METHOD02
#define LWC197_SET_RD_COALESCE_LG_SU                                0:0
#define LWC197_SET_RD_COALESCE_LG_SU_DISABLE                        (0x00000000)
#define LWC197_SET_RD_COALESCE_LG_SU_ENABLE                         (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_BES_CROP_DEBUG4_CLAMP_FP_BLEND */
#ifndef LWC197_SET_BES_CROP_DEBUG4
#ifdef  LWC197_SET_RESERVED_SW_METHOD03
#define LWC197_SET_BES_CROP_DEBUG4                                  LWC197_SET_RESERVED_SW_METHOD03
#define LWC197_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND                   0:0
#define LWC197_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND_TO_INF            (0x00000000)
#define LWC197_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND_TO_MAXVAL         (0x00000001)
#endif
#endif

/* Toggle LW_PGRAPH_PRI_BES_CROP_DEBUG3 fields that control blend optimizations (bug 1942454). */
#ifndef LWC197_SET_BES_CROP_DEBUG3
#ifdef  LWC197_SET_RESERVED_SW_METHOD08
#define LWC197_SET_BES_CROP_DEBUG3                                  LWC197_SET_RESERVED_SW_METHOD08
#define LWC197_SET_BES_CROP_DEBUG3_BLENDOPT                         0:0
#define LWC197_SET_BES_CROP_DEBUG3_BLENDOPT_KEEP_ZBC                (0x00000000)
#define LWC197_SET_BES_CROP_DEBUG3_BLENDOPT_ENABLE                  (0x00000001)
#endif
#endif

#endif /* _cl_pascal_b_sw_h_ */
