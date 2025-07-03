/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cl_ada_compute_a_swh
#define _cl_ada_compute_a_swh

/* This file is *not* auto-generated. */

/* LwNotification[] elements */
#define LWC9C0_NOTIFIERS_NOTIFY                                     (0)
#define LWC9C0_NOTIFIERS_DEBUG_INTR                                 (1)
#define LWC9C0_NOTIFIERS_BPT_INT                                    (2)
#define LWC9C0_NOTIFIERS_CILP_INT                                   (3)
#define LWC9C0_NOTIFIERS_PREEMPTION_STARTED                         (4)
#define LWC9C0_NOTIFIERS_MAXCOUNT                                   (5)

/* LwNotification[] fields and values */
#define LWC9C0_NOTIFICATION_STATUS_IN_PROGRESS                      (0x8000)
#define LWC9C0_NOTIFICATION_STATUS_DONE_SUCCESS                     (0x0000)

#ifndef LWC9C0_SET_CB_BASE
#ifdef LWC9C0_SET_RESERVED_SW_METHOD00
/* VPR=0 RM assigns global base, VPR=1 RM assigns VPR global base from _ADDR which is bits 40:12 */
#define LWC9C0_SET_CB_BASE                                          LWC9C0_SET_RESERVED_SW_METHOD00
#define LWC9C0_SET_CB_BASE_ADDR                                     27:0
#define LWC9C0_SET_CB_BASE_ADDR_SHIFT                               12
#define LWC9C0_SET_CB_BASE_VPR                                      31:31
#define LWC9C0_SET_CB_BASE_VPR_FALSE                                (0x00000000)
#define LWC9C0_SET_CB_BASE_VPR_TRUE                                 (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_BES_CROP_DEBUG4_CLAMP_FP_BLEND */
#ifndef LWC9C0_SET_BES_CROP_DEBUG4
#ifdef  LWC9C0_SET_RESERVED_SW_METHOD03
#define LWC9C0_SET_BES_CROP_DEBUG4                                  LWC9C0_SET_RESERVED_SW_METHOD03
#define LWC9C0_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND                   0:0
#define LWC9C0_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND_TO_INF            (0x00000000)
#define LWC9C0_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND_TO_MAXVAL         (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD */
#ifndef LWC9C0_SET_SKED_DEBUG_1
#ifdef  LWC9C0_SET_RESERVED_SW_METHOD04
#define LWC9C0_SET_SKED_DEBUG_1                                     LWC9C0_SET_RESERVED_SW_METHOD04
#define LWC9C0_SET_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD                 0:0
#define LWC9C0_SET_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD_DISABLE         (0x00000000)
#define LWC9C0_SET_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD_ENABLE          (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_GPCS_TPCS_SM_CACHE_CONTROL_ILWALIDATE_BUNDLE_CBREF
 * & LW_PGRAPH_PRI_SCC_DEBUG_LC_BUFFER_OPEN
 * Per the bug 1893747, we need both these bits set in unison hence having
 * this as a single SW method instead of 2 diff methods
 */
#ifndef LWC9C0_BUG_1893747_WAR
#ifdef  LWC9C0_SET_RESERVED_SW_METHOD05
#define LWC9C0_BUG_1893747_WAR                                      LWC9C0_SET_RESERVED_SW_METHOD05
#define LWC9C0_BUG_1893747_WAR_CONTROL                              0:0
#define LWC9C0_BUG_1893747_WAR_CONTROL_DISABLE                      (0x00000000)
#define LWC9C0_BUG_1893747_WAR_CONTROL_ENABLE                       (0x00000001)
#endif
#endif

/* Set LW_PTPC_PRI_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE
 * & LW_PTPC_PRI_SM_L1TAG_CTRL_CACHE_SURFACE_(LD/ST)
 */
#ifndef LWC9C0_SET_TEX_IN_DBG
#ifdef  LWC9C0_SET_RESERVED_SW_METHOD06
#define LWC9C0_SET_TEX_IN_DBG                                           LWC9C0_SET_RESERVED_SW_METHOD06
#define LWC9C0_SET_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE                      0:0
#define LWC9C0_SET_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE_DISABLE              (0x00000000)
#define LWC9C0_SET_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE_ENABLE               (0x00000001)
#define LWC9C0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_LD            1:1
#define LWC9C0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_LD_DISABLE    (0x00000000)
#define LWC9C0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_LD_ENABLE     (0x00000001)
#define LWC9C0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_ST            2:2
#define LWC9C0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_ST_DISABLE    (0x00000000)
#define LWC9C0_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_ST_ENABLE     (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_SKED_HWW_ESR_EN_SKEDCHECK18_L1_CONFIG_TOO_SMALL
 * DEFAULT means keep the current config. i.e a noop
 * For bug 1864936
 */
#ifndef LWC9C0_SET_SKEDCHECK
#ifdef  LWC9C0_SET_RESERVED_SW_METHOD07
#define LWC9C0_SET_SKEDCHECK                                        LWC9C0_SET_RESERVED_SW_METHOD07
#define LWC9C0_SET_SKEDCHECK_18                                     1:0
#define LWC9C0_SET_SKEDCHECK_18_DEFAULT                             (0x00000000)
#define LWC9C0_SET_SKEDCHECK_18_DISABLE                             (0x00000001)
#define LWC9C0_SET_SKEDCHECK_18_ENABLE                              (0x00000002)
#endif
#endif

/* Toggle LW_PGRAPH_PRI_BES_CROP_DEBUG3 fields that control blend optimizations (bug 1942454). */
#ifndef LWC9C0_SET_BES_CROP_DEBUG3
#ifdef  LWC9C0_SET_RESERVED_SW_METHOD08
#define LWC9C0_SET_BES_CROP_DEBUG3                                  LWC9C0_SET_RESERVED_SW_METHOD08
#define LWC9C0_SET_BES_CROP_DEBUG3_BLENDOPT                         0:0
#define LWC9C0_SET_BES_CROP_DEBUG3_BLENDOPT_KEEP_ZBC                (0x00000000)
#define LWC9C0_SET_BES_CROP_DEBUG3_BLENDOPT_ENABLE                  (0x00000001)
#endif
#endif

/* Toggle LW_PGRAPH_PRI_GPCS_GCC_DBG_3 fields that controls GCC prefetch enablement(bug 3035996). */
#ifndef LWC9C0_SET_GPCS_GCC_DBG_3
#ifdef  LWC9C0_SET_RESERVED_SW_METHOD10
#define LWC9C0_SET_GPCS_GCC_DBG_3                                   LWC9C0_SET_RESERVED_SW_METHOD10
#define LWC9C0_SET_GPCS_GCC_DBG_3_PREFETCH                          0:0
#define LWC9C0_SET_GPCS_GCC_DBG_3_PREFETCH_DISABLE                  (0x00000000)
#define LWC9C0_SET_GPCS_GCC_DBG_3_PREFETCH_ENABLE                   (0x00000001)
#endif
#endif

/* Toggle LW_PTPC_PRI_SM_CBU_CNTRL_RELEASE_BARRIER fields (bug 3033648). */
#ifndef LWC9C0_SET_SM_CBU_CNTRL
#ifdef  LWC9C0_SET_RESERVED_SW_METHOD11
#define LWC9C0_SET_SM_CBU_CNTRL                                     LWC9C0_SET_RESERVED_SW_METHOD11
#define LWC9C0_SET_SM_CBU_CNTRL_ELECTION_RELEASE_BARRIER            0:0
#define LWC9C0_SET_SM_CBU_CNTRL_ELECTION_RELEASE_BARRIER_DISABLE    (0x00000000)
#define LWC9C0_SET_SM_CBU_CNTRL_ELECTION_RELEASE_BARRIER_ENABLE     (0x00000001)
#endif
#endif

#endif /* _cl_ada_compute_a_swh */
