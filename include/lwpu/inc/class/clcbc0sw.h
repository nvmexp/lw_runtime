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

#ifndef CLCBC0SW_H
#define CLCBC0SW_H

/* This file is *not* auto-generated. */

/* LwNotification[] elements */
#define LWCBC0_NOTIFIERS_NOTIFY                                     (0)
#define LWCBC0_NOTIFIERS_DEBUG_INTR                                 (1)
#define LWCBC0_NOTIFIERS_BPT_INT                                    (2)
#define LWCBC0_NOTIFIERS_CILP_INT                                   (3)
#define LWCBC0_NOTIFIERS_PREEMPTION_STARTED                         (4)
#define LWCBC0_NOTIFIERS_MAXCOUNT                                   (5)

/* LwNotification[] fields and values */
#define LWCBC0_NOTIFICATION_STATUS_IN_PROGRESS                      (0x8000)
#define LWCBC0_NOTIFICATION_STATUS_DONE_SUCCESS                     (0x0000)

#ifndef LWCBC0_SET_CB_BASE
#ifdef LWCBC0_SET_RESERVED_SW_METHOD00
/* VPR=0 RM assigns global base, VPR=1 RM assigns VPR global base from _ADDR which is bits 40:12 */
#define LWCBC0_SET_CB_BASE                                          LWCBC0_SET_RESERVED_SW_METHOD00
#define LWCBC0_SET_CB_BASE_ADDR                                     27:0
#define LWCBC0_SET_CB_BASE_ADDR_SHIFT                               12
#define LWCBC0_SET_CB_BASE_VPR                                      31:31
#define LWCBC0_SET_CB_BASE_VPR_FALSE                                (0x00000000)
#define LWCBC0_SET_CB_BASE_VPR_TRUE                                 (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_BES_CROP_DEBUG4_CLAMP_FP_BLEND */
#ifndef LWCBC0_SET_BES_CROP_DEBUG4
#ifdef  LWCBC0_SET_RESERVED_SW_METHOD03
#define LWCBC0_SET_BES_CROP_DEBUG4                                  LWCBC0_SET_RESERVED_SW_METHOD03
#define LWCBC0_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND                   0:0
#define LWCBC0_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND_TO_INF            (0x00000000)
#define LWCBC0_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND_TO_MAXVAL         (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD */
#ifndef LWCBC0_SET_SKED_DEBUG_1
#ifdef  LWCBC0_SET_RESERVED_SW_METHOD04
#define LWCBC0_SET_SKED_DEBUG_1                                     LWCBC0_SET_RESERVED_SW_METHOD04
#define LWCBC0_SET_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD                 0:0
#define LWCBC0_SET_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD_DISABLE         (0x00000000)
#define LWCBC0_SET_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD_ENABLE          (0x00000001)
#endif
#endif

/* Set LW_PTPC_PRI_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE
 * & LW_PTPC_PRI_SM_L1TAG_CTRL_CACHE_SURFACE_(LD/ST)
 */
#ifndef LWCBC0_SET_TEX_IN_DBG
#ifdef  LWCBC0_SET_RESERVED_SW_METHOD06
#define LWCBC0_SET_TEX_IN_DBG                                           LWCBC0_SET_RESERVED_SW_METHOD06
#define LWCBC0_SET_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE                      0:0
#define LWCBC0_SET_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE_DISABLE              (0x00000000)
#define LWCBC0_SET_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE_ENABLE               (0x00000001)
#define LWCBC0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_LD            1:1
#define LWCBC0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_LD_DISABLE    (0x00000000)
#define LWCBC0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_LD_ENABLE     (0x00000001)
#define LWCBC0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_ST            2:2
#define LWCBC0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_ST_DISABLE    (0x00000000)
#define LWCBC0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_ST_ENABLE     (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_SKED_HWW_ESR_EN_SKEDCHECK18_L1_CONFIG_TOO_SMALL
 * DEFAULT means keep the current config. i.e a noop
 * For bug 1864936
 */
#ifndef LWCBC0_SET_SKEDCHECK
#ifdef  LWCBC0_SET_RESERVED_SW_METHOD07
#define LWCBC0_SET_SKEDCHECK                                        LWCBC0_SET_RESERVED_SW_METHOD07
#define LWCBC0_SET_SKEDCHECK_18                                     1:0
#define LWCBC0_SET_SKEDCHECK_18_DEFAULT                             (0x00000000)
#define LWCBC0_SET_SKEDCHECK_18_DISABLE                             (0x00000001)
#define LWCBC0_SET_SKEDCHECK_18_ENABLE                              (0x00000002)
#endif
#endif

/* Toggle LW_PGRAPH_PRI_BES_CROP_DEBUG3 fields that control blend optimizations (bug 1942454). */
#ifndef LWCBC0_SET_BES_CROP_DEBUG3
#ifdef  LWCBC0_SET_RESERVED_SW_METHOD08
#define LWCBC0_SET_BES_CROP_DEBUG3                                  LWCBC0_SET_RESERVED_SW_METHOD08
#define LWCBC0_SET_BES_CROP_DEBUG3_BLENDOPT                         0:0
#define LWCBC0_SET_BES_CROP_DEBUG3_BLENDOPT_KEEP_ZBC                (0x00000000)
#define LWCBC0_SET_BES_CROP_DEBUG3_BLENDOPT_ENABLE                  (0x00000001)
#endif
#endif

/* Toggle LW_PGRAPH_PRI_GPCS_GCC_DBG_3 fields that controls GCC prefetch enablement(bug 3035996). */
#ifndef LWCBC0_SET_GPCS_GCC_DBG_3
#ifdef  LWCBC0_SET_RESERVED_SW_METHOD10
#define LWCBC0_SET_GPCS_GCC_DBG_3                                   LWCBC0_SET_RESERVED_SW_METHOD10
#define LWCBC0_SET_GPCS_GCC_DBG_3_PREFETCH                          0:0
#define LWCBC0_SET_GPCS_GCC_DBG_3_PREFETCH_DISABLE                  (0x00000000)
#define LWCBC0_SET_GPCS_GCC_DBG_3_PREFETCH_ENABLE                   (0x00000001)
#endif
#endif


/* Toggle 32b support. */
#ifndef LWCBC0_SET_32BIT_APP
#ifdef  LWCBC0_SET_RESERVED_SW_METHOD11
#define LWCBC0_SET_32BIT_APP                                        LWCBC0_SET_RESERVED_SW_METHOD11
#define LWCBC0_SET_32BIT_APP_SUPPORT                                0:0
#define LWCBC0_SET_32BIT_APP_SUPPORT_DISABLE                        (0x00000000)
#define LWCBC0_SET_32BIT_APP_SUPPORT_ENABLE                         (0x00000001)
#endif // LWCBC0_SET_RESERVED_SW_METHOD11
#endif // LWCBC0_SET_32BIT_APP

#endif /* CLCBC0SW_H */
