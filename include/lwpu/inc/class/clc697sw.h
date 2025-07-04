/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cl_ampere_a_sw_h_
#define _cl_ampere_a_sw_h_

/* This file is *not* auto-generated. */

/* LwNotification[] elements */
#define LWC697_NOTIFIERS_NOTIFY                                     (0)
#define LWC697_NOTIFIERS_NONSTALL                                   (1)
#define LWC697_NOTIFIERS_SET_MEMORY_SURFACE_ATTR                    (0)
#define LWC697_NOTIFIERS_DEBUG_INTR                                 (2)
#define LWC697_NOTIFIERS_MAXCOUNT                                   (3)

/* LwNotification[] fields and values */
#define LWC697_NOTIFICATION_STATUS_IN_PROGRESS                      (0x8000)
#define LWC697_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
#define LWC697_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT               (0x2000)
#define LWC697_NOTIFICATION_STATUS_ERROR_ILWALID_STATE              (0x1000)
#define LWC697_NOTIFICATION_STATUS_ERROR_STATE_IN_USE               (0x0800)
#define LWC697_NOTIFICATION_STATUS_DONE_SUCCESS                     (0x0000)

#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_STATUS            0:0
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_STATUS_FAILED     (0x0000)
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_STATUS_SUCCESS    (0x0001)
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_COMPR             2:2
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_COMPR_DISABLED    (0x0000)
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_COMPR_ENABLED     (0x0001)
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_SHARED            4:4
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_SHARED_DISABLED   (0x0000)
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_SHARED_ENABLED    (0x0001)
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_LOWER             5:5
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_LOWER_DISABLED    (0x0000)
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_LOWER_ENABLED     (0x0001)
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_TILE_FORMAT       11:8
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_TILE_REGION       15:12
#define LWC697_NOTIFICATION_SET_MEMORY_SURFACE_ATTR_INFO32_COMPR_COVG        31:16

#ifndef LWC697_SET_CB_BASE
#ifdef LWC697_SET_RESERVED_SW_METHOD00
/* VPR=0 RM assigns global base, VPR=1 RM assigns VPR global base from _ADDR which is bits 40:12 */
#define LWC697_SET_CB_BASE                                          LWC697_SET_RESERVED_SW_METHOD00
#define LWC697_SET_CB_BASE_ADDR                                     27:0
#define LWC697_SET_CB_BASE_ADDR_SHIFT                               12
#define LWC697_SET_CB_BASE_VPR                                      31:31
#define LWC697_SET_CB_BASE_VPR_FALSE                                (0x00000000)
#define LWC697_SET_CB_BASE_VPR_TRUE                                 (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_BES_CROP_DEBUG4_CLAMP_FP_BLEND */
#ifndef LWC697_SET_BES_CROP_DEBUG4
#ifdef  LWC697_SET_RESERVED_SW_METHOD03
#define LWC697_SET_BES_CROP_DEBUG4                                  LWC697_SET_RESERVED_SW_METHOD03
#define LWC697_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND                   0:0
#define LWC697_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND_TO_INF            (0x00000000)
#define LWC697_SET_BES_CROP_DEBUG4_CLAMP_FP_BLEND_TO_MAXVAL         (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD */
#ifndef LWC697_SET_SKED_DEBUG_1
#ifdef  LWC697_SET_RESERVED_SW_METHOD04
#define LWC697_SET_SKED_DEBUG_1                                     LWC697_SET_RESERVED_SW_METHOD04
#define LWC697_SET_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD                 0:0
#define LWC697_SET_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD_DISABLE         (0x00000000)
#define LWC697_SET_SKED_DEBUG_1_AUTO_ILWALIDATE_QMD_ENABLE          (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_GPCS_TPCS_SM_CACHE_CONTROL_ILWALIDATE_BUNDLE_CBREF
 * & LW_PGRAPH_PRI_SCC_DEBUG_LC_BUFFER_OPEN
 * Per the bug 1893747, we need both these bits set in unison hence having
 * this as a single SW method instead of 2 diff methods
 */
#ifndef LWC697_BUG_1893747_WAR
#ifdef  LWC697_SET_RESERVED_SW_METHOD05
#define LWC697_BUG_1893747_WAR                                      LWC697_SET_RESERVED_SW_METHOD05
#define LWC697_BUG_1893747_WAR_CONTROL                              0:0
#define LWC697_BUG_1893747_WAR_CONTROL_DISABLE                      (0x00000000)
#define LWC697_BUG_1893747_WAR_CONTROL_ENABLE                       (0x00000001)
#endif
#endif

/* Set LW_PTPC_PRI_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE
 * & LW_PTPC_PRI_SM_L1TAG_CTRL_CACHE_SURFACE_(LD/ST)
 */
#ifndef LWC697_SET_TEX_IN_DBG
#ifdef  LWC697_SET_RESERVED_SW_METHOD06
#define LWC697_SET_TEX_IN_DBG                                           LWC697_SET_RESERVED_SW_METHOD06
#define LWC697_SET_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE                      0:0
#define LWC697_SET_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE_DISABLE              (0x00000000)
#define LWC697_SET_TEX_IN_DBG_TSL1_RVCH_ILWALIDATE_ENABLE               (0x00000001)
#define LWC697_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_LD            1:1
#define LWC697_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_LD_DISABLE    (0x00000000)
#define LWC697_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_LD_ENABLE     (0x00000001)
#define LWC697_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_ST            2:2
#define LWC697_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_ST_DISABLE    (0x00000000)
#define LWC697_SET_TEX_IN_DBG_SM_L1TAG_CTRL_CACHE_SURFACE_ST_ENABLE     (0x00000001)
#endif
#endif

/* Set LW_PGRAPH_PRI_SKED_HWW_ESR_EN_SKEDCHECK18_L1_CONFIG_TOO_SMALL
 * DEFAULT means keep the current config. i.e a noop
 * For bug 1864936
 */
#ifndef LWC697_SET_SKEDCHECK
#ifdef  LWC697_SET_RESERVED_SW_METHOD07
#define LWC697_SET_SKEDCHECK                                        LWC697_SET_RESERVED_SW_METHOD07
#define LWC697_SET_SKEDCHECK_18                                     1:0
#define LWC697_SET_SKEDCHECK_18_DEFAULT                             (0x00000000)
#define LWC697_SET_SKEDCHECK_18_DISABLE                             (0x00000001)
#define LWC697_SET_SKEDCHECK_18_ENABLE                              (0x00000002)
#endif
#endif

/* Toggle LW_PGRAPH_PRI_BES_CROP_DEBUG3 fields that control blend optimizations (bug 1942454). */
#ifndef LWC697_SET_BES_CROP_DEBUG3
#ifdef  LWC697_SET_RESERVED_SW_METHOD08
#define LWC697_SET_BES_CROP_DEBUG3                                  LWC697_SET_RESERVED_SW_METHOD08
#define LWC697_SET_BES_CROP_DEBUG3_BLENDOPT                         0:0
#define LWC697_SET_BES_CROP_DEBUG3_BLENDOPT_KEEP_ZBC                (0x00000000)
#define LWC697_SET_BES_CROP_DEBUG3_BLENDOPT_ENABLE                  (0x00000001)
#endif
#endif

/*
 * Set LW_PTPC_PRI_SM_L1TAG_CTRL_SURFACE_LWT_COLLECTOR.
 * For bug 2517016
 */
#ifndef LWC697_SET_SHADER_LWT_COLLECTOR
#ifdef  LWC697_SET_RESERVED_SW_METHOD12
#define LWC697_SET_SHADER_LWT_COLLECTOR                             LWC697_SET_RESERVED_SW_METHOD12
#define LWC697_SET_SHADER_LWT_COLLECTOR_STATE                       0:0
#define LWC697_SET_SHADER_LWT_COLLECTOR_STATE_DISABLE               (0x00000000)
#define LWC697_SET_SHADER_LWT_COLLECTOR_STATE_ENABLE                (0x00000001)
#endif
#endif

#endif /* _cl_ampere_a_sw_h_ */
