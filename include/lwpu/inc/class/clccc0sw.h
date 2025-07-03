/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef CLCCC0SW_H
#define CLCCC0SW_H

/* This file is *not* auto-generated. */

/* LwNotification[] elements */
#define LWCCC0_NOTIFIERS_NOTIFY                                     (0)
#define LWCCC0_NOTIFIERS_DEBUG_INTR                                 (1)
#define LWCCC0_NOTIFIERS_BPT_INT                                    (2)
#define LWCCC0_NOTIFIERS_CILP_INT                                   (3)
#define LWCCC0_NOTIFIERS_PREEMPTION_STARTED                         (4)
#define LWCCC0_NOTIFIERS_MAXCOUNT                                   (5)

/* LwNotification[] fields and values */
#define LWCCC0_NOTIFICATION_STATUS_IN_PROGRESS                      (0x8000)
#define LWCCC0_NOTIFICATION_STATUS_DONE_SUCCESS                     (0x0000)

#ifndef LWCCC0_SET_CB_BASE
#ifdef LWCCC0_SET_RESERVED_SW_METHOD00
/* VPR=0 RM assigns global base, VPR=1 RM assigns VPR global base from _ADDR which is bits 40:12 */
#define LWCCC0_SET_CB_BASE                                          LWCCC0_SET_RESERVED_SW_METHOD00
#define LWCCC0_SET_CB_BASE_ADDR                                     27:0
#define LWCCC0_SET_CB_BASE_ADDR_SHIFT                               12
#define LWCCC0_SET_CB_BASE_VPR                                      31:31
#define LWCCC0_SET_CB_BASE_VPR_FALSE                                (0x00000000)
#define LWCCC0_SET_CB_BASE_VPR_TRUE                                 (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_BES_CROP_DEBUG4_CLAMP_FP_BLEND */
#ifndef LWCCC0_SET_BES_CROP_DEBUG4
#ifdef  LWCCC0_SET_RESERVED_SW_METHOD03
#define LWCCC0_SET_BES_CROP_DEBUG4                                  LWCCC0_SET_RESERVED_SW_METHOD03
#define LWCCC0_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND                   0:0
#define LWCCC0_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND_TO_INF            (0x00000000)
#define LWCCC0_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND_TO_MAXVAL         (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD */
#ifndef LWCCC0_SET_SKED_DEBUG_1
#ifdef  LWCCC0_SET_RESERVED_SW_METHOD04
#define LWCCC0_SET_SKED_DEBUG_1                                     LWCCC0_SET_RESERVED_SW_METHOD04
#define LWCCC0_SET_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD                 0:0
#define LWCCC0_SET_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD_DISABLE         (0x00000000)
#define LWCCC0_SET_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD_ENABLE          (0x00000001)
#endif
#endif

/* Set LW_PTPC_PRI_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE
 * & LW_PTPC_PRI_SM_L1TAG_CTRL_CACHE_SURFACE_(LD/ST)
 */
#ifndef LWCCC0_SET_TEX_IN_DBG
#ifdef  LWCCC0_SET_RESERVED_SW_METHOD06
#define LWCCC0_SET_TEX_IN_DBG                                           LWCCC0_SET_RESERVED_SW_METHOD06
#define LWCCC0_SET_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE                      0:0
#define LWCCC0_SET_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE_DISABLE              (0x00000000)
#define LWCCC0_SET_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE_ENABLE               (0x00000001)
#define LWCCC0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_LD            1:1
#define LWCCC0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_LD_DISABLE    (0x00000000)
#define LWCCC0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_LD_ENABLE     (0x00000001)
#define LWCCC0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_ST            2:2
#define LWCCC0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_ST_DISABLE    (0x00000000)
#define LWCCC0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_ST_ENABLE     (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_SKED_HWW_ESR_EN_SKEDCHECK18_L1_CONFIG_TOO_SMALL
 * DEFAULT means keep the current config. i.e a noop
 * For bug 1864936
 */
#ifndef LWCCC0_SET_SKEDCHECK
#ifdef  LWCCC0_SET_RESERVED_SW_METHOD07
#define LWCCC0_SET_SKEDCHECK                                        LWCCC0_SET_RESERVED_SW_METHOD07
#define LWCCC0_SET_SKEDCHECK_18                                     1:0
#define LWCCC0_SET_SKEDCHECK_18_DEFAULT                             (0x00000000)
#define LWCCC0_SET_SKEDCHECK_18_DISABLE                             (0x00000001)
#define LWCCC0_SET_SKEDCHECK_18_ENABLE                              (0x00000002)
#endif
#endif

/* Toggle LW_PGRAPH_PRI_BES_CROP_DEBUG3 fields that control blend optimizations (bug 1942454). */
#ifndef LWCCC0_SET_BES_CROP_DEBUG3
#ifdef  LWCCC0_SET_RESERVED_SW_METHOD08
#define LWCCC0_SET_BES_CROP_DEBUG3                                  LWCCC0_SET_RESERVED_SW_METHOD08
#define LWCCC0_SET_BES_CROP_DEBUG3_BLENDOPT                         0:0
#define LWCCC0_SET_BES_CROP_DEBUG3_BLENDOPT_KEEP_ZBC                (0x00000000)
#define LWCCC0_SET_BES_CROP_DEBUG3_BLENDOPT_ENABLE                  (0x00000001)
#endif
#endif

/* Toggle LW_PGRAPH_PRI_GPCS_GCC_DBG_3 fields that controls GCC prefetch enablement(bug 3035996). */
#ifndef LWCCC0_SET_GPCS_GCC_DBG_3
#ifdef  LWCCC0_SET_RESERVED_SW_METHOD10
#define LWCCC0_SET_GPCS_GCC_DBG_3                                   LWCCC0_SET_RESERVED_SW_METHOD10
#define LWCCC0_SET_GPCS_GCC_DBG_3_PREFETCH                          0:0
#define LWCCC0_SET_GPCS_GCC_DBG_3_PREFETCH_DISABLE                  (0x00000000)
#define LWCCC0_SET_GPCS_GCC_DBG_3_PREFETCH_ENABLE                   (0x00000001)
#endif
#endif

/* Toggle 32b support. */
#ifndef LWCCC0_SET_32BIT_APP
#ifdef  LWCCC0_SET_RESERVED_SW_METHOD11
#define LWCCC0_SET_32BIT_APP                                        LWCCC0_SET_RESERVED_SW_METHOD11
#define LWCCC0_SET_32BIT_APP_SUPPORT                                0:0
#define LWCCC0_SET_32BIT_APP_SUPPORT_DISABLE                        (0x00000000)
#define LWCCC0_SET_32BIT_APP_SUPPORT_ENABLE                         (0x00000001)
#endif // LWCCC0_SET_RESERVED_SW_METHOD11
#endif // LWCCC0_SET_32BIT_APP

#endif /* CLCCC0SW_H */
