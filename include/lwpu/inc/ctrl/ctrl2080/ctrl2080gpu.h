/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080gpu.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl2080/ctrl2080base.h"
#include "ctrl/ctrl2080/ctrl2080gr.h"
#include "ctrl/ctrl0000/ctrl0000system.h"

/* LW20_SUBDEVICE_XX gpu control commands and parameters */

/* Valid feature values */
#define LW2080_CTRL_GPU_GET_FEATURES_CLK_ARCH_DOMAINS                     0:0
#define LW2080_CTRL_GPU_GET_FEATURES_CLK_ARCH_DOMAINS_FALSE (0x00000000)
#define LW2080_CTRL_GPU_GET_FEATURES_CLK_ARCH_DOMAINS_TRUE  (0x00000001)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW2080_CTRL_GPU_INFO
 *
 * This structure represents a single 32bit gpu value.  Clients request
 * a particular gpu value by specifying a unique gpu information index.
 *
 * Legal gpu information index values are:
 *   LW2080_CTRL_GPU_INFO_INDEX_FLOOR_SWEEP
 *     This index is used to request the floor sweeping value for the
 *     associated subdevice.  The return value is GPU implementation-dependent.
 *     A return value of 0 indicates the GPU does not support floor sweeping.
 *   LW2080_CTLR_GPU_INFO_INDEX_ECID_LO32
 *   LW2080_CTLR_GPU_INFO_INDEX_ECID_HI32
 *     These indices are used to request the low and high 32bits of the
 *     ECID value of the associated subdevice.
 *   LW2080_CTRL_GPU_INFO_INDEX_COMPUTE_ENABLE
 *     This index is used to request the fuse value for compute enable.
 *     The fuse value does not cause the compute class to be removed if zero.
 *   LW2080_CTRL_GPU_INFO_INDEX_MINOR_REVISION_EXT
 *     This index is used to request the extended minor revision value for
 *     the associated subdevice.  Return values for this index include, but are
 *     not limited to, the following defines:
 *       LW2080_CTRL_GPU_INFO_MINOR_REVISION_EXT_NONE
 *       LW2080_CTRL_GPU_INFO_MINOR_REVISION_EXT_P
 *       LW2080_CTRL_GPU_INFO_MINOR_REVISION_EXT_V
 *       LW2080_CTRL_GPU_INFO_MINOR_REVISION_EXT_PV
 *   LW2080_CTRL_GPU_INFO_INDEX_SAMPLE
 *     This index is used to request the sample value for the associated
 *     subdevice.  Legal return values for this index include:
 *       LW2080_CTRL_GPU_INFO_SAMPLE_NONE
 *         The GPU is a production GPU.
 *       LW2080_CTRL_GPU_INFO_SAMPLE_ES
 *         The GPU is an engineering sample.
 *       LW2080_CTRL_GPU_INFO_SAMPLE_QS
 *         The GPU is a qualification sample.
 *       LW2080_CTRL_GPU_INFO_SAMPLE_PS
 *         The GPU is a production sample.
 *       LW2080_CTRL_GPU_INFO_SAMPLE_QS_PS_PROD
 *         The GPU is a qualification sample, a production sample, or
 *         a production GPU (i.e. it is not an engineering sample).
 *   LW2080_CTRL_GPU_INFO_INDEX_HW_QUAL_TYPE
 *     This index is used to request the hardware qualification type
 *     for the associated subdevice.  A return value of
 *     LW2080_CTRL_GPU_INFO_INDEX_HW_QUAL_TYPE_NONE indicates the GPU
 *     is not a hardware qualification sample.  Legal return values
 *     for this index include:
 *       LW2080_CTRL_GPU_INFO_HW_QUAL_TYPE_NONE
 *       LW2080_CTRL_GPU_INFO_HW_QUAL_TYPE_NOMINAL
 *       LW2080_CTRL_GPU_INFO_HW_QUAL_TYPE_SLOW
 *       LW2080_CTRL_GPU_INFO_HW_QUAL_TYPE_FAST
 *       LW2080_CTRL_GPU_INFO_HW_QUAL_TYPE_HIGH_LEAKAGE
 *  LW2080_CTRL_GPU_INFO_INDEX_FOUNDRY
 *    This index is used to request the foundry information for the associated
 *    subdevice.  Legal return values for this index include.
 *      LW2080_CTRL_GPU_INFO_FOUNDRY_TSMC
 *      LW2080_CTRL_GPU_INFO_FOUNDRY_UMC
 *      LW2080_CTRL_GPU_INFO_FOUNDRY_IBM
 *      LW2080_CTRL_GPU_INFO_FOUNDRY_SMIC
 *      LW2080_CTRL_GPU_INFO_FOUNDRY_CHARTERED
 *      LW2080_CTRL_GPU_INFO_FOUNDRY_TOSHIBA
 *      LW2080_CTRL_GPU_INFO_FOUNDRY_SONY
 *   LW2080_CTRL_GPU_INFO_INDEX_FAB_CODE_ENG
 *     This index is used to request engineering flag for the associated
 *     subdevice.  Legal return values for this index include:
 *       LW2080_CTRL_GPU_INFO_FAB_CODE_ENG_ENG
 *       LW2080_CTRL_GPU_INFO_FAB_CODE_ENG_PROD
 *   LW2080_CTRL_GPU_INFO_INDEX_FAB_CODE_FAB
 *     This index is used to request the fab code for the associated subdevice.
 *   LW2080_CTRL_GPU_INFO_INDEX_LOT_CODE_HEAD
 *     This index is used to request the lot code heading for the associated
 *     subdevice.
 *   LW2080_CTRL_GPU_INFO_INDEX_LOT_CODE
 *     This index is used to request the lot code for the associated subdevice.
 *   LW2080_CTRL_GPU_INFO_INDEX_WAFER_ID
 *     This index is used to request the wafer id for the associated subdevice.
 *   LW2080_CTRL_GPU_INFO_INDEX_X_COORD
 *     This index is used to request the wafer x coordinate for the associated
 *     subdevice.
 *   LW2080_CTRL_GPU_INFO_INDEX_Y_COORD
 *     This index is used to request the wafer y coordinate for the associated
 *     subdevice.
 *   LW2080_CTRL_GPU_INFO_INDEX_CP_BIN
 *     This index is used to request the CP bin for the associated subdevice.
 *   LW2080_CTRL_GPU_INFO_INDEX_VENDOR_CODE
 *     This index is used to request the vendor code for the associated subdevice.
 *   LW2080_CTRL_GPU_INFO_INDEX_LW_SPECIAL
 *     This index is used to request the lw special flag for the associated
 *     subdevice.
 *   LW2080_CTRL_GPU_INFO_INDEX_TESLA_ENABLE
 *     This index is used to request the gpu's TESLA feature.
 *   LW2080_CTRL_GPU_INFO_INDEX_HDCP_FUSE_STATE
 *     This index is used to request the state of the HDCP fuse for the
 *      associated subdevice. Legal return values for this index include:
 *       LW2080_CTRL_GPU_INFO_HDCP_FUSE_STATE_DISABLED
 *         This value indicates that HDCP is disabled
 *         at the fuse (or strap) level.
 *       LW2080_CTRL_GPU_INFO_HDCP_FUSE_STATE_ENABLED
 *         This value indicates that HDCP is not disabled.
 *         at the fuse (or strap) level.
 *   LW2080_CTRL_GPU_INFO_INDEX_HDCP_KEY_SOURCE
 *     This index is used to request the source of HDCP keys in use by
 *     the subdevice. Legal return values for this index include:
 *       LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_NONE
 *         This value indicates there is no key source.
 *       LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_CRYPTO_ROM
 *         This value indicates that the key source is a crypto ROM.
 *       LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_SBIOS
 *         This value indicates that the key source is the SBIOS.
 *       LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_I2C_ROM
 *         This value indicates that the key source is an I2C ROM.
 *       LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_FUSES
 *         This value indicates that the key source are fuses,.
 *       LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_UNKNOWN
 *         This value indicates that the key source is unknown.
 *   LW2080_CTRL_GPU_INFO_INDEX_HDCP_KEY_SOURCE_STATE
 *     This index is used to request the status of the HDCP key
 *     source in use by the subdevice.
 *     Legal return values for this index include:
 *       LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_STATE_ABSENT
 *         This value indicates the key source is not available or is absent.
 *       LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_STATE_PRESENT
 *         This value indicates the key source state is present.
 *       LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_STATE_UNKNOWN
 *         This value indicates that the key source state is unknown.
 *   LW2080_CTRL_GPU_INFO_INDEX_FAB_ID
 *     This index is used  to request the fab ID for the associated subdevice.
 *       LW2080_CTRL_GPU_INFO_FAB_ID_28
 *       LW2080_CTRL_GPU_INFO_FAB_ID_40
 *       LW2080_CTRL_GPU_INFO_FAB_ID_55
 *       LW2080_CTRL_GPU_INFO_FAB_ID_65
 *   LW2080_CTRL_GPU_INFO_INDEX_INTERNAL_HDCP_CAPABLE
 *     This index is used to request whether or not the GPU has internal
 *     HDCP. Note that even if the GPU has internal HDCP support does not mean
 *     HDCP should be expected to work; there are other reasons why HDCP might
 *     not be enabled (such as VBIOS settings).
 *       LW2080_CTRL_GPU_INFO_INTERNAL_HDCP_CAPABLE_NO
 *       LW2080_CTRL_GPU_INFO_INTERNAL_HDCP_CAPABLE_YES
 *   LW2080_CTRL_GPU_INFO_INDEX_INTERNAL_HDMI_CAPABLE
 *     This index is used to request whether or not the GPU has internal
 *     HDMI. Note that even if the GPU has internal HDMI support does not mean
 *     HDMI should be expected to work; there are other reasons why HDMI might
 *     not be enabled (such as VBIOS settings).
 *       LW2080_CTRL_GPU_INFO_INTERNAL_HDMI_CAPABLE_NO
 *       LW2080_CTRL_GPU_INFO_INTERNAL_HDMI_CAPABLE_YES
 *   LW2080_CTRL_GPU_INFO_INDEX_P556_BINDING_SUPPORT
 *     This index is used to check if gpu can be bound to a p556.
 *       LW2080_CTRL_GPU_INFO_P556_BINDING_SUPPORT_NO
 *       LW2080_CTRL_GPU_INFO_P556_BINDING_SUPPORT_YES
 *   LW2080_CTRL_GPU_INFO_INDEX_1X_PCIE_OPTIMIZATIONS
 *     This index is used to check whether 1x pcie optimizations should be enabled
 *       LW2080_CTRL_GPU_INFO_INDEX_1X_PCIE_OPTIMIZATIONS_DISABLED
 *       LW2080_CTRL_GPU_INFO_INDEX_1X_PCIE_OPTIMIZATIONS_ENABLED
 *   LW2080_CTRL_GPU_INFO_INDEX_UNEXPECTED_ES_SAMPLE
 *       This index is used to check for the unexpected state in which the associated
 *       GPU has passed QS but is an ES sample.
 *       LW2080_CTRL_GPU_INFO_UNEXPECTED_ES_SAMPLE_YES
 *       LW2080_CTRL_GPU_INFO_UNEXPECTED_ES_SAMPLE_NO
 *   LW2080_CTRL_GPU_INFO_INDEX_SYSMEM_ACCESS
 *       This index is used to check if the GPU can access system memory.
 *       LW2080_CTRL_GPU_INFO_SYSMEM_ACCESS_YES
 *       LW2080_CTRL_GPU_INFO_SYSMEM_ACCESS_NO
 *   LW2080_CTRL_GPU_INFO_INDEX_GRID_CAPABILITY
 *       This index is used to check if the GPU is capable of running on GRID.
 *       LW2080_CTRL_GPU_INFO_INDEX_GRID_CAPABILITY_YES
 *       LW2080_CTRL_GPU_INFO_INDEX_GRID_CAPABILITY_NO
 *   LW2080_CTRL_GPU_INFO_INDEX_EDISON_ENABLE
 *     This index is used to request the gpu's EDISON feature.
 *   LW2080_CTLR_GPU_INFO_INDEX_PDI0
 *   LW2080_CTLR_GPU_INFO_INDEX_PDI1
 *     These indices are used to request the Per Device Identifier
 *     of the associated GPU.
 *   LW2080_CTRL_GPU_INFO_INDEX_SURPRISE_REMOVAL_POSSIBLE
 *     This index is used to check if surprise removal is possible for the
 *     gpu or not. Surprise removal is possible for External Gpu's connected via
 *     supported TB3 and hotplug capable bridge.
 *   LW2080_CTRL_GPU_INFO_INDEX_IBMNPU_RELAXED_ORDERING
 *     This index is used to query relaxed ordering capability for an IBM-NPU
 *     associated with the GPU.
 *     In brief, the POWER9 LWLink processing unit (NPU) uses relaxed ordering
 *     for writes from the PCIe to GPU memory. For more details, see section 8.1.
 *     file://sc-netapp15/gpu_dev_info/hw/doc/gpu/volta/volta/design/general/VOLTA_P9.docx
 *   LW2080_CTRL_GPU_INFO_INDEX_GLOBAL_POISON_FUSE_ENABLED
 *     Whether Global Poison fuse is enabled or disabled.
 *     LW2080_CTRL_GPU_INFO_INDEX_GLOBAL_POISON_FUSE_ENABLED_NO
 *     LW2080_CTRL_GPU_INFO_INDEX_GLOBAL_POISON_FUSE_ENABLED_YES
 *   LW2080_CTRL_GPU_INFO_INDEX_LWSWITCH_PROXY_DETECTED
 *     This index is used to check if the GPU has LWSwitch proxy connectivity,
 *     aka the shared LWSwitch virtualization configuration.
 *   LW2080_CTRL_GPU_INFO_INDEX_GPU_SR_SUPPORT
 *     Whether GPU can support panel self refresh
 *     LW2080_CTRL_GPU_INFO_INDEX_GPU_SR_SUPPORT_NO
 *     LW2080_CTRL_GPU_INFO_INDEX_GPU_SR_SUPPORT_YES
 *   LW2080_CTRL_GPU_INFO_INDEX_GPU_SMC_MODE
 *     This index is used to query the current SMC (a.k.a MIG) mode.
 *     The _PENDING values indicate that SMC mode is not yet effective due to
 *     pending GPU reset (PF-FLR).
 *     LW2080_CTRL_GPU_INFO_GPU_SMC_MODE_UNSUPPORTED
 *     LW2080_CTRL_GPU_INFO_GPU_SMC_MODE_ENABLED
 *     LW2080_CTRL_GPU_INFO_GPU_SMC_MODE_DISABLED
 *     LW2080_CTRL_GPU_INFO_GPU_SMC_MODE_ENABLE_PENDING
 *     LW2080_CTRL_GPU_INFO_GPU_SMC_MODE_DISABLE_PENDING
 *   LW2080_CTRL_GPU_INFO_INDEX_SPLIT_VAS_MGMT_SERVER_CLIENT_RM
 *     Whether Split Vaspace management between server/client RM is enabled or disabled.
 *     LW2080_CTRL_GPU_INFO_SPLIT_VAS_MGMT_SERVER_CLIENT_RM_YES
 *     LW2080_CTRL_GPU_INFO_SPLIT_VAS_MGMT_SERVER_CLIENT_RM_NO
 *   LW2080_CTRL_GPU_INFO_INDEX_GPU_FLA_CAPABILITY
 *       This index is used to check if the GPU is capable of supporting Fabric Linear Addressing (FLA)
 *       LW2080_CTRL_GPU_INFO_INDEX_FLA_CAPABILITY_YES
 *       LW2080_CTRL_GPU_INFO_INDEX_FLA_CAPABILITY_NO
  *   LW2080_CTRL_GPU_INFO_INDEX_PER_RUNLIST_CHANNEL_RAM
 *       This index is used to check if the GPU supports per runlist channel ram
 *       LW2080_CTRL_GPU_INFO_INDEX_PER_RUNLIST_CHANNEL_RAM_ENABLED
 *       LW2080_CTRL_GPU_INFO_INDEX_PER_RUNLIST_CHANNEL_RAM_DISABLED
 *   LW2080_CTRL_GPU_INFO_INDEX_GPU_ATS_CAPABILITY
 *     This index is used to check if the GPU is ATS( Address Translation Service ) capable
 *     LW2080_CTRL_GPU_INFO_INDEX_GPU_ATS_CAPABILITY_YES
 *     LW2080_CTRL_GPU_INFO_INDEX_GPU_ATS_CAPABILITY_NO
 *   LW2080_CTRL_GPU_INFO_INDEX_LWENC_STATS_REPORTING_STATE
 *     This index is used to get the LwEnc session stats reporting state.
 *     LW2080_CTRL_GPU_INFO_LWENC_STATS_REPORTING_STATE_DISABLED
 *     LW2080_CTRL_GPU_INFO_LWENC_STATS_REPORTING_STATE_ENABLED
 *     LW2080_CTRL_GPU_INFO_LWENC_STATS_REPORTING_STATE_NOT_SUPPORTED
 *   LW2080_CTRL_GPU_INFO_INDEX_4K_PAGE_ISOLATION_REQUIRED
 *     This index is used to check if the GPU needs 4K page isolation
 *     LW2080_CTRL_GPU_INFO_INDEX_4K_PAGE_ISOLATION_REQUIRED_NO
 *     LW2080_CTRL_GPU_INFO_INDEX_4K_PAGE_ISOLATION_REQUIRED_YES
 *   LW2080_CTRL_GPU_INFO_INDEX_DISPLAY_ENABLED
 *     This index queries whether display is enabled.
 *     LW2080_CTRL_GPU_INFO_DISPLAY_ENABLED_NO
 *     LW2080_CTRL_GPU_INFO_DISPLAY_ENABLED_YES
 *   LW2080_CTRL_GPU_INFO_INDEX_MOBILE_CONFIG_ENABLED
 *     This index queries whether mobile config is enabled.
 *     LW2080_CTRL_GPU_INFO_INDEX_MOBILE_CONFIG_ENABLED_NO
 *     LW2080_CTRL_GPU_INFO_INDEX_MOBILE_CONFIG_ENABLED_YES
 *   LW2080_CTRL_GPU_INFO_INDEX_GPU_PROFILING_CAPABILITY
 *     This index queries whether profiling/tracing is allowed
 *     LW2080_CTRL_GPU_INFO_INDEX_GPU_PROFILING_CAPABILITY_DISABLED
 *     LW2080_CTRL_GPU_INFO_INDEX_GPU_PROFILING_CAPABILITY_ENABLED
 *   LW2080_CTRL_GPU_INFO_INDEX_GPU_DEBUGGING_CAPABILITY
 *     This index queries whether debugging is allowed
 *     LW2080_CTRL_GPU_INFO_INDEX_GPU_DEBUGGING_CAPABILITY_DISABLED
 *     LW2080_CTRL_GPU_INFO_INDEX_GPU_DEBUGGING_CAPABILITY_ENABLED
 *   LW2080_CTRL_GPU_INFO_INDEX_GPU_VAB_CAPABILITY
 *     This index queries whether the GPU supports vidmem access bit tracking.
 *     LW2080_CTRL_GPU_INFO_INDEX_GPU_VAB_CAPABILITY_UNSUPPORTED
 *     LW2080_CTRL_GPU_INFO_INDEX_GPU_VAB_CAPABILITY_SUPPORTED
 *   LW2080_CTRL_GPU_INFO_INDEX_LWSWITCH_PROXY_DETECTED
 *     For more details on this, refer https://confluence.lwpu.com/pages/viewpage.action?pageId=119082379
 *   LW2080_CTRL_GPU_INFO_INDEX_GPU_LOCAL_EGM_CAPABILITY
 *     This index queries whether local EGM is supported
 *     EGM stands for Extended GPU Memory.Ideally, EGM will be a large carveout
 *     of TH500 CPU_MEM which will be accessed and managed as GPU_MEM by RM.
 *     The data returned by this control call gives us the following info:
 *     a. BIT-0 to represent whether local EGM is lwrrently enabled
 *        LW2080_CTRL_GPU_INFO_INDEX_GPU_LOCAL_EGM_CAPABILITY_NO
 *        LW2080_CTRL_GPU_INFO_INDEX_GPU_LOCAL_EGM_CAPABILITY_YES
 *     b. The remaining bits represent the peer ID used for local EGM
 *     when LW2080_CTRL_GPU_INFO_INDEX_GPU_LOCAL_EGM_CAPABILITY is true
 *   LW2080_CTRL_GPU_INFO_INDEX_GPU_SELF_HOSTED_CAPABILITY
 *     This index queries whether the GPU is operating in self hosted mode.
 *     LW2080_CTRL_GPU_INFO_INDEX_GPU_SELF_HOSTED_CAPABILITY_NO
 *     LW2080_CTRL_GPU_INFO_INDEX_GPU_SELF_HOSTED_CAPABILITY_YES
 *   LW2080_CTRL_GPU_INFO_INDEX_CMP_SKU
 *     This index detects if the underlying SKU is a CMP (Crypto Mining Processor) SKU.
 *     LW2080_CTRL_GPU_INFO_INDEX_CMP_SKU_YES
 *     LW2080_CTRL_GPU_INFO_INDEX_CMP_SKU_NO
 */
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


/*
 *   LW2080_CTRL_GPU_INFO_INDEX_DMABUF_CAPABILITY
 *     This index queries if dma-buf is supported on the driver and GPU.
 *     LW2080_CTRL_GPU_INFO_INDEX_DMABUF_CAPABILITY_YES
 *     LW2080_CTRL_GPU_INFO_INDEX_DMABUF_CAPABILITY_NO
 */

typedef struct LW2080_CTRL_GPU_INFO {
    LwU32 index;
    LwU32 data;
} LW2080_CTRL_GPU_INFO;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/* valid gpu info index values */
#define LW2080_CTRL_GPU_INFO_INDEX_FLOOR_SWEEP                         (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_ECID_LO32                           (0x00000001)
#define LW2080_CTRL_GPU_INFO_INDEX_ECID_HI32                           (0x00000002)
#define LW2080_CTRL_GPU_INFO_INDEX_COMPUTE_ENABLE                      (0x00000003)
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#define LW2080_CTRL_GPU_INFO_INDEX_MINOR_REVISION_EXT                  (0x00000004)
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#define LW2080_CTRL_GPU_INFO_INDEX_SAMPLE                              (0x00000005)
#define LW2080_CTRL_GPU_INFO_INDEX_HW_QUAL_TYPE                        (0x00000006)
#define LW2080_CTRL_GPU_INFO_INDEX_FOUNDRY                             (0x00000007)
#define LW2080_CTRL_GPU_INFO_INDEX_FAB_CODE_ENG                        (0x00000008)
#define LW2080_CTRL_GPU_INFO_INDEX_FAB_CODE_FAB                        (0x00000009)
#define LW2080_CTRL_GPU_INFO_INDEX_LOT_CODE_HEAD                       (0x0000000a)
#define LW2080_CTRL_GPU_INFO_INDEX_LOT_CODE                            (0x0000000b)
#define LW2080_CTRL_GPU_INFO_INDEX_WAFER_ID                            (0x0000000c)
#define LW2080_CTRL_GPU_INFO_INDEX_X_COORD                             (0x0000000d)
#define LW2080_CTRL_GPU_INFO_INDEX_Y_COORD                             (0x0000000e)
#define LW2080_CTRL_GPU_INFO_INDEX_CP_BIN                              (0x0000000f)
#define LW2080_CTRL_GPU_INFO_INDEX_LW_SPECIAL                          (0x00000010)
#define LW2080_CTRL_GPU_INFO_INDEX_TESLA_ENABLE                        (0x00000011)
#define LW2080_CTRL_GPU_INFO_INDEX_NETLIST_REV0                        (0x00000012)
#define LW2080_CTRL_GPU_INFO_INDEX_NETLIST_REV1                        (0x00000013)
#define LW2080_CTRL_GPU_INFO_INDEX_HDCP_FUSE_STATE                     (0x00000014)
#define LW2080_CTRL_GPU_INFO_INDEX_HDCP_KEY_SOURCE                     (0x00000015)
#define LW2080_CTRL_GPU_INFO_INDEX_HDCP_KEY_SOURCE_STATE               (0x00000016)
#define LW2080_CTRL_GPU_INFO_INDEX_INTERNAL_HDCP_CAPABLE               (0x00000017)
#define LW2080_CTRL_GPU_INFO_INDEX_INTERNAL_HDMI_CAPABLE               (0x00000018)
#define LW2080_CTRL_GPU_INFO_INDEX_FAB_ID                              (0x00000019)
#define LW2080_CTRL_GPU_INFO_INDEX_VENDOR_CODE                         (0x0000001a)
#define LW2080_CTRL_GPU_INFO_INDEX_ECID_EXTENDED                       (0x0000001b)
#define LW2080_CTRL_GPU_INFO_INDEX_P556_BINDING_SUPPORT                (0x0000001c)
#define LW2080_CTRL_GPU_INFO_INDEX_1X_PCIE_OPTIMIZATIONS               (0x0000001d)
#define LW2080_CTRL_GPU_INFO_INDEX_UNEXPECTED_ES_SAMPLE                (0x0000001e)
#define LW2080_CTRL_GPU_INFO_INDEX_SYSMEM_ACCESS                       (0x0000001f)
#define LW2080_CTRL_GPU_INFO_INDEX_GRID_CAPABILITY                     (0x00000020)
#define LW2080_CTRL_GPU_INFO_INDEX_EDISON_ENABLE                       (0x00000021)
#define LW2080_CTRL_GPU_INFO_INDEX_GEMINI_BOARD                        (0x00000022)
#define LW2080_CTRL_GPU_INFO_INDEX_PDI0                                (0x00000023)
#define LW2080_CTRL_GPU_INFO_INDEX_PDI1                                (0x00000024)
#define LW2080_CTRL_GPU_INFO_INDEX_SURPRISE_REMOVAL_POSSIBLE           (0x00000025)
#define LW2080_CTRL_GPU_INFO_INDEX_IBMNPU_RELAXED_ORDERING             (0x00000026)
#define LW2080_CTRL_GPU_INFO_INDEX_GLOBAL_POISON_FUSE_ENABLED          (0x00000027)
#define LW2080_CTRL_GPU_INFO_INDEX_LWSWITCH_PROXY_DETECTED             (0x00000028)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_SR_SUPPORT                      (0x00000029)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_SMC_MODE                        (0x0000002a)
#define LW2080_CTRL_GPU_INFO_INDEX_SPLIT_VAS_MGMT_SERVER_CLIENT_RM     (0x0000002b)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_SM_VERSION                      (0x0000002c)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_FLA_CAPABILITY                  (0x0000002d)
#define LW2080_CTRL_GPU_INFO_SKU_ID                                    (0x0000002e)
#define LW2080_CTRL_GPU_INFO_INDEX_PER_RUNLIST_CHANNEL_RAM             (0x0000002f)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_ATS_CAPABILITY                  (0x00000030)
#define LW2080_CTRL_GPU_INFO_INDEX_LWENC_STATS_REPORTING_STATE         (0x00000031)
#define LW2080_CTRL_GPU_INFO_FUSE_FILE_VERSION                         (0x00000032)
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#define LW2080_CTRL_GPU_INFO_INDEX_4K_PAGE_ISOLATION_REQUIRED          (0x00000033)
#define LW2080_CTRL_GPU_INFO_INDEX_DISPLAY_ENABLED                     (0x00000034)
#define LW2080_CTRL_GPU_INFO_INDEX_MOBILE_CONFIG_ENABLED               (0x00000035)
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#define LW2080_CTRL_GPU_INFO_INDEX_GPU_PROFILING_CAPABILITY            (0x00000036)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_DEBUGGING_CAPABILITY            (0x00000037)
#define LW2080_CTRL_GPU_INFO_INDEX_VR_READY                            (0x00000038)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_VAB_CAPABILITY                  (0x00000039)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_LOCAL_EGM_CAPABILITY            (0x0000003a)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_SELF_HOSTED_CAPABILITY          (0x0000003b)
#define LW2080_CTRL_GPU_INFO_INDEX_CMP_SKU                             (0x0000003c)
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#define LW2080_CTRL_GPU_INFO_INDEX_DMABUF_CAPABILITY                   (0x0000003d)
#define LW2080_CTRL_GPU_INFO_MAX_LIST_SIZE                             (0x0000003e)

/* valid minor revision extended values */
#define LW2080_CTRL_GPU_INFO_MINOR_REVISION_EXT_NONE                   (0x00000000)
#define LW2080_CTRL_GPU_INFO_MINOR_REVISION_EXT_P                      (0x00000001)
#define LW2080_CTRL_GPU_INFO_MINOR_REVISION_EXT_V                      (0x00000002)
#define LW2080_CTRL_GPU_INFO_MINOR_REVISION_EXT_PV                     (0x00000003)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/* valid sample values */
#define LW2080_CTRL_GPU_INFO_SAMPLE_NONE                               (0x00000000)
#define LW2080_CTRL_GPU_INFO_SAMPLE_ES                                 (0x00000001)
#define LW2080_CTRL_GPU_INFO_SAMPLE_QS                                 (0x00000002)
#define LW2080_CTRL_GPU_INFO_SAMPLE_PS                                 (0x00000003)
#define LW2080_CTRL_GPU_INFO_SAMPLE_QS_PS_PROD                         (0x00000004)

/* valid hardware qualification type values */
#define LW2080_CTRL_GPU_INFO_HW_QUAL_TYPE_NONE                         (0x00000000)
#define LW2080_CTRL_GPU_INFO_HW_QUAL_TYPE_NOMINAL                      (0x00000001)
#define LW2080_CTRL_GPU_INFO_HW_QUAL_TYPE_SLOW                         (0x00000002)
#define LW2080_CTRL_GPU_INFO_HW_QUAL_TYPE_FAST                         (0x00000003)
#define LW2080_CTRL_GPU_INFO_HW_QUAL_TYPE_HIGH_LEAKAGE                 (0x00000004)

/* valid foundry values */
#define LW2080_CTRL_GPU_INFO_FOUNDRY_TSMC                              (0x00000000)
#define LW2080_CTRL_GPU_INFO_FOUNDRY_UMC                               (0x00000001)
#define LW2080_CTRL_GPU_INFO_FOUNDRY_IBM                               (0x00000002)
#define LW2080_CTRL_GPU_INFO_FOUNDRY_SMIC                              (0x00000003)
#define LW2080_CTRL_GPU_INFO_FOUNDRY_CHARTERED                         (0x00000004)
#define LW2080_CTRL_GPU_INFO_FOUNDRY_TOSHIBA                           (0x00000005)
#define LW2080_CTRL_GPU_INFO_FOUNDRY_SONY                              (0x00000006)
#define LW2080_CTRL_GPU_INFO_FOUNDRY_SAMSUNG                           (0x00000007)

/* valid fab code engineering flag values */
#define LW2080_CTRL_GPU_INFO_FAB_CODE_ENG_PROD                         (0x00000000)
#define LW2080_CTRL_GPU_INFO_FAB_CODE_ENG_ENG                          (0x00000001)

/* valid Fab Id values */
#define LW2080_CTRL_GPU_INFO_FAB_ID_65                                 (0x00000000)
#define LW2080_CTRL_GPU_INFO_FAB_ID_55                                 (0x00000001)
#define LW2080_CTRL_GPU_INFO_FAB_ID_40                                 (0x00000002)
#define LW2080_CTRL_GPU_INFO_FAB_ID_28                                 (0x00000003)

/* valid hdcp fuse state values */
#define LW2080_CTRL_GPU_INFO_HDCP_FUSE_STATE_DISABLED                  (0x00000000)
#define LW2080_CTRL_GPU_INFO_HDCP_FUSE_STATE_ENABLED                   (0x00000001)

/* valid hdcp key source values */
#define LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_NONE                      (0x00000000)
#define LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_CRYPTO_ROM                (0x00000001)
#define LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_SBIOS                     (0x00000002)
#define LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_I2C_ROM                   (0x00000003)
#define LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_FUSES                     (0x00000004)
#define LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_UNKNOWN                   (0xFFFFFFFF)

/* valid hdcp key source state values */
#define LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_STATE_ABSENT              (0x00000000)
#define LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_STATE_PRESENT             (0x00000001)
#define LW2080_CTRL_GPU_INFO_HDCP_KEY_SOURCE_STATE_UNKNOWN             (0xFFFFFFFF)

/* valid hdcp capable values */
#define LW2080_CTRL_GPU_INFO_INTERNAL_HDCP_CAPABLE_NO                  (0x00000000)
#define LW2080_CTRL_GPU_INFO_INTERNAL_HDCP_CAPABLE_YES                 (0x00000001)

/* valid hdmi capable values */
#define LW2080_CTRL_GPU_INFO_INTERNAL_HDMI_CAPABLE_NO                  (0x00000000)
#define LW2080_CTRL_GPU_INFO_INTERNAL_HDMI_CAPABLE_YES                 (0x00000001)

/* valid p556 binding support values */
#define LW2080_CTRL_GPU_INFO_P556_BINDING_SUPPORT_NO                   (0x00000000)
#define LW2080_CTRL_GPU_INFO_P556_BINDING_SUPPORT_YES                  (0x00000001)

/* valid 1x pcie optimization values */
#define LW2080_CTRL_GPU_INFO_INDEX_1X_PCIE_OPTIMIZATIONS_DISABLED      (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_1X_PCIE_OPTIMIZATIONS_ENABLED       (0x00000001)

/* valid unexpected ES sample values */
#define LW2080_CTRL_GPU_INFO_UNEXPECTED_ES_SAMPLE_NO                   (0x00000000)
#define LW2080_CTRL_GPU_INFO_UNEXPECTED_ES_SAMPLE_YES                  (0x00000001)

/* valid system memory access capability values */
#define LW2080_CTRL_GPU_INFO_SYSMEM_ACCESS_NO                          (0x00000000)
#define LW2080_CTRL_GPU_INFO_SYSMEM_ACCESS_YES                         (0x00000001)

/* valid grid capability values */
#define LW2080_CTRL_GPU_INFO_INDEX_GRID_CAPABILITY_NO                  (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_GRID_CAPABILITY_YES                 (0x00000001)

/* valid gemini board values */
#define LW2080_CTRL_GPU_INFO_INDEX_GEMINI_BOARD_NO                     (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_GEMINI_BOARD_YES                    (0x00000001)

/* valid surprise removal values */
#define LW2080_CTRL_GPU_INFO_INDEX_SURPRISE_REMOVAL_POSSIBLE_NO        (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_SURPRISE_REMOVAL_POSSIBLE_YES       (0x00000001)

/* valid relaxed ordering values */
#define LW2080_CTRL_GPU_INFO_IBMNPU_RELAXED_ORDERING_DISABLED          (0x00000000)
#define LW2080_CTRL_GPU_INFO_IBMNPU_RELAXED_ORDERING_ENABLED           (0x00000001)
#define LW2080_CTRL_GPU_INFO_IBMNPU_RELAXED_ORDERING_UNSUPPORTED       (0xFFFFFFFF)

/* valid poison fuse capability values */
#define LW2080_CTRL_GPU_INFO_INDEX_GLOBAL_POISON_FUSE_ENABLED_NO       (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_GLOBAL_POISON_FUSE_ENABLED_YES      (0x00000001)

/* valid lwswitch proxy detected values */
#define LW2080_CTRL_GPU_INFO_LWSWITCH_PROXY_DETECTED_NO                (0x00000000)
#define LW2080_CTRL_GPU_INFO_LWSWITCH_PROXY_DETECTED_YES               (0x00000001)

/* valid LWSR GPU support info values */
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_SR_SUPPORT_NO                   (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_SR_SUPPORT_YES                  (0x00000001)

/* valid SMC mode values */
#define LW2080_CTRL_GPU_INFO_GPU_SMC_MODE_UNSUPPORTED                  (0x00000000)
#define LW2080_CTRL_GPU_INFO_GPU_SMC_MODE_ENABLED                      (0x00000001)
#define LW2080_CTRL_GPU_INFO_GPU_SMC_MODE_DISABLED                     (0x00000002)
#define LW2080_CTRL_GPU_INFO_GPU_SMC_MODE_ENABLE_PENDING               (0x00000003)
#define LW2080_CTRL_GPU_INFO_GPU_SMC_MODE_DISABLE_PENDING              (0x00000004)

/* valid split VAS mode values */
#define LW2080_CTRL_GPU_INFO_SPLIT_VAS_MGMT_SERVER_CLIENT_RM_NO        (0x00000000)
#define LW2080_CTRL_GPU_INFO_SPLIT_VAS_MGMT_SERVER_CLIENT_RM_YES       (0x00000001)

/* valid grid capability values */
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_FLA_CAPABILITY_NO               (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_FLA_CAPABILITY_YES              (0x00000001)

/* valid per runlist channel ram capability values */
#define LW2080_CTRL_GPU_INFO_INDEX_PER_RUNLIST_CHANNEL_RAM_DISABLED    (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_PER_RUNLIST_CHANNEL_RAM_ENABLED     (0x00000001)

/* valid ATS capability values */
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_ATS_CAPABILITY_NO               (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_ATS_CAPABILITY_YES              (0x00000001)

/* valid Lwenc Session Stats reporting state values */
#define LW2080_CTRL_GPU_INFO_LWENC_STATS_REPORTING_STATE_DISABLED      (0x00000000)
#define LW2080_CTRL_GPU_INFO_LWENC_STATS_REPORTING_STATE_ENABLED       (0x00000001)
#define LW2080_CTRL_GPU_INFO_LWENC_STATS_REPORTING_STATE_NOT_SUPPORTED (0x00000002)

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/* valid 4K PAGE isolation requirement values */
#define LW2080_CTRL_GPU_INFO_INDEX_4K_PAGE_ISOLATION_REQUIRED_NO       (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_4K_PAGE_ISOLATION_REQUIRED_YES      (0x00000001)

/* valid display enabled values */
#define LW2080_CTRL_GPU_INFO_DISPLAY_ENABLED_NO                        (0x00000000)
#define LW2080_CTRL_GPU_INFO_DISPLAY_ENABLED_YES                       (0x00000001)

/* valid mobile config enabled values */
#define LW2080_CTRL_GPU_INFO_INDEX_MOBILE_CONFIG_ENABLED_NO            (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_MOBILE_CONFIG_ENABLED_YES           (0x00000001)
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/* valid profiling capability values */
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_PROFILING_CAPABILITY_DISABLED   (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_PROFILING_CAPABILITY_ENABLED    (0x00000001)

/* valid debugging capability values */
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_DEBUGGING_CAPABILITY_DISABLED   (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_DEBUGGING_CAPABILITY_ENABLED    (0x00000001)

/* valid VR Ready values */
#define LW2080_CTRL_GPU_INFO_INDEX_VR_READY_DISABLED                   (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_VR_READY_ENABLED                    (0x00000001)

/* valid VAB capability values */
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_VAB_CAPABILITY_UNSUPPORTED      (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_VAB_CAPABILITY_SUPPORTED        (0x00000001)

/* valid local EGM supported values */
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_LOCAL_EGM_CAPABILITY_NO         (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_LOCAL_EGM_CAPABILITY_YES        (0x00000001)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_LOCAL_EGM_PEERID                          31:1 

/* valid self hosted values */
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_SELF_HOSTED_CAPABILITY_NO       (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_GPU_SELF_HOSTED_CAPABILITY_YES      (0x00000001)

/* valid CMP (Crypto Mining Processor) SKU values */
#define LW2080_CTRL_GPU_INFO_INDEX_CMP_SKU_NO                          (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_CMP_SKU_YES                         (0x00000001)
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/* valid dma-buf suport values */
#define LW2080_CTRL_GPU_INFO_INDEX_DMABUF_CAPABILITY_NO                (0x00000000)
#define LW2080_CTRL_GPU_INFO_INDEX_DMABUF_CAPABILITY_YES               (0x00000001)

/*
 * LW2080_CTRL_CMD_GPU_GET_INFO
 *
 * This command returns gpu information for the associated GPU.  Requests
 * to retrieve gpu information use a list of one or more LW2080_CTRL_GPU_INFO
 * structures.
 *
 *   gpuInfoListSize
 *     This field specifies the number of entries on the caller's
 *     gpuInfoList.
 *   gpuInfoList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer into which the gpu information is to be returned.
 *     This buffer must be at least as big as gpuInfoListSize multiplied
 *     by the size of the LW2080_CTRL_GPU_INFO structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_GPU_GET_INFO                                   (0x20800101) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_GPU_GET_INFO_PARAMS {
    LwU32 gpuInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 gpuInfoList, 8);
} LW2080_CTRL_GPU_GET_INFO_PARAMS;

#define LW2080_CTRL_CMD_GPU_GET_INFO_V2 (0x20800102) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x2" */

typedef struct LW2080_CTRL_GPU_GET_INFO_V2_PARAMS {
    LwU32                gpuInfoListSize;
    LW2080_CTRL_GPU_INFO gpuInfoList[LW2080_CTRL_GPU_INFO_MAX_LIST_SIZE];
} LW2080_CTRL_GPU_GET_INFO_V2_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_NAME_STRING
 *
 * This command returns the name of the GPU in string form in either ASCII
 * or UNICODE format.
 *
 *   gpuNameStringFlags
 *     This field specifies flags to use while creating the GPU name string.
 *     Valid flags values:
 *       LW2080_CTRL_GPU_GET_NAME_STRING_FLAGS_TYPE_ASCII
 *         The returned name string should be in standard ASCII format.
 *       LW2080_CTRL_GPU_GET_NAME_STRING_FLAGS_TYPE_UNICODE
 *         The returned name string should be in unicode format.
 *   gpuNameString
 *     This field contains the buffer into which the name string should be
 *     returned.  The length of the returned string will be no more than
 *     LW2080_CTRL_GPU_MAX_NAME_STRING_LENGTH bytes in size.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_GPU_GET_NAME_STRING                (0x20800110) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_NAME_STRING_PARAMS_MESSAGE_ID" */

#define LW2080_GPU_MAX_NAME_STRING_LENGTH                  (0x0000040)

// This field is deprecated - 'gpuNameStringFlags' is now a simple scalar.
// Field maintained (and extended from 0:0) for compile-time compatibility.
#define LW2080_CTRL_GPU_GET_NAME_STRING_FLAGS_TYPE                    31:0

/* valid gpu name string flags */
#define LW2080_CTRL_GPU_GET_NAME_STRING_FLAGS_TYPE_ASCII   (0x00000000)
#define LW2080_CTRL_GPU_GET_NAME_STRING_FLAGS_TYPE_UNICODE (0x00000001)

#define LW2080_CTRL_GPU_GET_NAME_STRING_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW2080_CTRL_GPU_GET_NAME_STRING_PARAMS {
    LwU32 gpuNameStringFlags;
    union {
        LwU8  ascii[LW2080_GPU_MAX_NAME_STRING_LENGTH];
        LwU16 unicode[LW2080_GPU_MAX_NAME_STRING_LENGTH];
    } gpuNameString;
} LW2080_CTRL_GPU_GET_NAME_STRING_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_SHORT_NAME_STRING
 *
 * This command returns the short name of the GPU in ASCII string form.
 *
 *   gpuShortNameString
 *     This field contains the buffer into which the short name string should
 *     be returned.  The length of the returned string will be no more than
 *     LW2080_MAX_NAME_STRING_LENGTH bytes in size.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_GPU_GET_SHORT_NAME_STRING (0x20800111) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_SHORT_NAME_STRING_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_SHORT_NAME_STRING_PARAMS_MESSAGE_ID (0x11U)

typedef struct LW2080_CTRL_GPU_GET_SHORT_NAME_STRING_PARAMS {
    LwU8 gpuShortNameString[LW2080_GPU_MAX_NAME_STRING_LENGTH];
} LW2080_CTRL_GPU_GET_SHORT_NAME_STRING_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW2080_CTRL_CMD_GPU_SET_POWER
 *
 * This command sets the power state for the GPU as a whole, various engines,
 * or clocks.
 *
 *   target
 *     One of LW2080_CTRL_GPU_SET_POWER_TARGET_*
 *
 *   newLevel
 *     One of LW2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_*
 *            LW2080_CTRL_GPU_SET_POWER_STATE_ENGINE_LEVEL_*
 *            LW2080_CTRL_GPU_SET_POWER_STATE_CLOCK_LEVEL_*
 *     depending on the target above.
 *
 *   oldLevel
 *     Previous level as appropriate.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_GPU_SET_POWER (0x20800112) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_SET_POWER_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_SET_POWER_PARAMS_MESSAGE_ID (0x12U)

typedef struct LW2080_CTRL_GPU_SET_POWER_PARAMS {
    LwU32 target;
    LwU32 newLevel;
    LwU32 oldLevel;
} LW2080_CTRL_GPU_SET_POWER_PARAMS;

#define LW2080_CTRL_GPU_SET_POWER_TARGET_GPU                   (0x00000001)

#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_GR             (0x00000002)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_ROP            (0x00000004)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_MPEG           (0x00000008)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_VIP            (0x00000010)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_HEADA          (0x00000020)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_HEADB          (0x00000040)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_VS             (0x00000080)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_IFPA           (0x00000100)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_IFPB           (0x00000200)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_FPA            (0x00000400)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_FPB            (0x00000800)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_MIO            (0x00001000)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_TVO            (0x00002000)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_DACA           (0x00004000)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_DACB           (0x00008000)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_DACC           (0x00010000)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_FRWR           (0x00020000)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_IDX            (0x00040000)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_DISP           (0x00080000)

//
// Alternate definitions to allow for indexed base queries
//

#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_ARY_HEAD(x)                  \
    (LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_HEADA<<(x))
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_ARY_IFP(x)                   \
    (LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_IFPA<<(x))
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_ARY_FP(x)                    \
    (LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_FPA<<(x))
#define LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_ARY_DAC(x)                   \
    (LW2080_CTRL_GPU_SET_POWER_TARGET_ENGINE_DACA<<(x))

#define LW2080_CTRL_GPU_SET_POWER_TARGET_CLOCK_LWPLL           (0x01000000)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_CLOCK_MPLL            (0x02000000)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_CLOCK_VPLL1           (0x04000000)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_CLOCK_VPLL2           (0x08000000)
#define LW2080_CTRL_GPU_SET_POWER_TARGET_CLOCK_TVCLK           (0x10000000)

#define LW2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_0            (0x00000000)
#define LW2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_1            (0x00000001)
#define LW2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_2            (0x00000002)
#define LW2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_3            (0x00000003)
#define LW2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_4            (0x00000004)
#define LW2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_7            (0x00000007)

#define LW2080_CTRL_GPU_SET_POWER_STATE_ENGINE_LEVEL_FULLPOWER (0x00000000)
#define LW2080_CTRL_GPU_SET_POWER_STATE_ENGINE_LEVEL_AUTOMATIC (0x00000001)
#define LW2080_CTRL_GPU_SET_POWER_STATE_ENGINE_LEVEL_SUSPENDED (0x00000002)
#define LW2080_CTRL_GPU_SET_POWER_STATE_ENGINE_LEVEL_DISABLED  (0x00000003)

#define LW2080_CTRL_GPU_SET_POWER_STATE_CLOCK_LEVEL_FULLPOWER  (0x00000010)
#define LW2080_CTRL_GPU_SET_POWER_STATE_CLOCK_LEVEL_BYPASS     (0x00000011)
#define LW2080_CTRL_GPU_SET_POWER_STATE_CLOCK_LEVEL_SUSPENDED  (0x00000012)

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_CMD_GPU_GET_SDM
 *
 * This command returns the subdevice mask value for the associated subdevice.
 * The subdevice mask value can be used with the SET_SUBDEVICE_MASK instruction
 * provided by the LW36_CHANNEL_DMA and newer channel dma classes.
 *
 *   subdeviceMask [out]
 *     This field return the subdevice mask value.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_GPU_GET_SDM                            (0x20800118) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_SDM_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_SDM_PARAMS_MESSAGE_ID (0x18U)

typedef struct LW2080_CTRL_GPU_GET_SDM_PARAMS {
    LwU32 subdeviceMask;
} LW2080_CTRL_GPU_GET_SDM_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_SET_SDM
 *
 * This command sets the subdevice instance and mask value for the associated subdevice.
 * The subdevice mask value can be used with the SET_SUBDEVICE_MASK instruction
 * provided by the LW36_CHANNEL_DMA and newer channel dma classes.
 * It must be called before the GPU HW is initialized otherwise 
 * LW_ERR_ILWALID_STATE is being returned.
 *
 *   subdeviceMask [in]
 *     This field configures the subdevice mask value for the GPU/Subdevice
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_DATA
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_GPU_SET_SDM (0x20800120) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_SET_SDM_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_SET_SDM_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW2080_CTRL_GPU_SET_SDM_PARAMS {
    LwU32 subdeviceMask;
} LW2080_CTRL_GPU_SET_SDM_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_SIMULATION_INFO
 *
 * This command returns the associated subdevices' simulation information.
 *
 *   type
 *     This field returns the simulation type.
 *     One of LW2080_CTRL_GPU_GET_SIMULATION_INFO_TYPE_*
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_SIMULATION_INFO (0x20800119) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_SIMULATION_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_SIMULATION_INFO_PARAMS_MESSAGE_ID (0x19U)

typedef struct LW2080_CTRL_GPU_GET_SIMULATION_INFO_PARAMS {
    LwU32 type;
} LW2080_CTRL_GPU_GET_SIMULATION_INFO_PARAMS;

#define LW2080_CTRL_GPU_GET_SIMULATION_INFO_TYPE_NONE          (0x00000000)
#define LW2080_CTRL_GPU_GET_SIMULATION_INFO_TYPE_MODS_AMODEL   (0x00000001)
#define LW2080_CTRL_GPU_GET_SIMULATION_INFO_TYPE_LIVE_AMODEL   (0x00000002)
#define LW2080_CTRL_GPU_GET_SIMULATION_INFO_TYPE_FMODEL        (0x00000003)
#define LW2080_CTRL_GPU_GET_SIMULATION_INFO_TYPE_RTL           (0x00000004)
#define LW2080_CTRL_GPU_GET_SIMULATION_INFO_TYPE_EMU           (0x00000005)
#define LW2080_CTRL_GPU_GET_SIMULATION_INFO_TYPE_EMU_LOW_POWER (0x00000006)
#define LW2080_CTRL_GPU_GET_SIMULATION_INFO_TYPE_DFPGA         (0x00000007)
#define LW2080_CTRL_GPU_GET_SIMULATION_INFO_TYPE_DFPGA_RTL     (0x00000008)
#define LW2080_CTRL_GPU_GET_SIMULATION_INFO_TYPE_DFPGA_FMODEL  (0x00000009)
#define LW2080_CTRL_GPU_GET_SIMULATION_INFO_TYPE_UNKNOWN       (0xFFFFFFFF)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW2080_CTRL_GPU_REG_OP
 *
 * This structure describes register operation information for use with
 * the LW2080_CTRL_CMD_GPU_EXEC_REG_OPS command.  The structure describes
 * a single register operation.  The operation can be a read or write and
 * can involve either 32bits or 64bits of data.
 *
 * For 32bit read operations, the operation takes the following form:
 *
 *   regValueLo = read(bar0 + regOffset)
 *   regValueHi = 0
 *
 * For 64bit read operations, the operation takes the following form:
 *
 *   regValueLo = read(bar0 + regOffset)
 *   regValueHi = read(bar0 + regOffset + 4)
 *
 * For 32bit write operations, the operation takes the following form:
 *
 *   new = ((read(bar0 + regOffset) & ~regAndNMaskLo) | regValueLo)
 *   write(bar0 + regOffset, new)
 *
 * For 64bit write operations, the operation takes the following form:
 *
 *   new_lo = ((read(bar0 + regOffset) & ~regAndNMaskLo) | regValueLo)
 *   new_hi = ((read(bar0 + regOffset + 4) &  ~regAndNMaskHi) | regValueHi)
 *   write(bar0 + regOffset, new_lo)
 *   write(bar0 + regOffset + 4, new_hi)
 *
 * Details on the parameters follow:
 *
 *   regOp
 *     This field specifies the operation to be applied to the register
 *     specified by the regOffset parameter.  Valid values for this
 *     parameter are:
  *      LW2080_CTRL_GPU_REG_OP_READ_08
 *         The register operation should be a 8bit global privileged register read.
 *       LW2080_CTRL_GPU_REG_OP_WRITE_08
 *         The register operation should be a 8bit global privileged register write.
 *       LW2080_CTRL_GPU_REG_OP_READ_32
 *         The register operation should be a 32bit register read.
 *       LW2080_CTRL_GPU_REG_OP_WRITE_32
 *         The register operation should be a 32bit register write.
 *       LW2080_CTRL_GPU_REG_OP_READ_64
 *         The register operation should be a 64bit register read.
 *       LW2080_CTRL_GPU_REG_OP_WRITE_64
 *         The register operation should be a 64bit register write.
 *   regType
 *     This field specifies the type of the register specified by the
 *     regOffset parameter.  Valid values for this parameter are:
 *       LW2080_CTRL_GPU_REG_OP_TYPE_GLOBAL
 *         The register is a global privileged register.  Read operations
 *         return the current value from the associated global register.
 *         Write operations for registers of this type take effect immediately.
 *       LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX
 *         The register is a graphics context register.  Read operations
 *         return the current value from the associated global register.
 *         Write operations are applied to all existing graphics engine
 *         contexts.  Any newly created graphics engine contexts will also
 *         be modified.
 *       LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_TPC
 *         This is a graphics context TPC register group. Write operations are
 *         applied to TPC group(s) specified by regGroupMask.
 *         This field is ignored for read operations.
 *       LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_SM
 *         This is a graphics context SM register group that is inside TPC
 *         group.  Write operations are applied to SM group(s) specified by
 *         regGroupMask (TPC) and regSubGroupMask (SM). This field is ignored
 *         for read operations.
 *       LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_CROP
 *         This is a graphics context CROP register group. Write operations
 *         are applied to registers specified by regGroupMask. This field is
 *         ignored for read operations.
 *       LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_ZROP
 *         This is a graphics context ZROP register group. Write operations
 *         are applied to registers specified by regGroupMask. This field is
 *         ignored for read operations.
 *       LW2080_CTRL_GPU_REG_OP_TYPE_FB
 *         This is a fb register group. Write operations are applied to
 *         registers specified by regGroupMask. This field is
 *         ignored for read operations.
 *       LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_QUAD
 *         This is a graphics context QUAD register group. Operations
 *         are applied to registers specified by regQuad value.
 *   regQuad
 *     This field specifies the quad to be accessed for register regOffsetwhen
 *     the regType specified is LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_QUAD.
 *   regGroupMask
 *     This field specifies which registers inside an array should be updated.
 *     This field is used when regType is one of below:
 *       LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_TPC
 *       LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_SM
 *       LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_CROP
 *       LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_ZROP
 *       LW2080_CTRL_GPU_REG_OP_TYPE_FB
 *     When regGroupMask is used, the regOffset MUST be the first register in
 *     an array.
 *   regSubGroupMask
 *     This field specifies which registers inside a group should be updated.
 *     This field is used for updating SM registers when regType is:
 *       LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_TPC
 *     When regSubGroupMask is used, regOffset MUST be the first register in an
 *     array AND also the first one in sub array.  regGroupMask specifies
 *     TPC(X) and regSubGroupMask specifies SM_CTX_N(Y)
 *   regStatus
 *     This field returns the completion status for the associated register
 *     operation in the form of a bitmask.  Possible status values for this
 *     field are:
 *       LW2080_CTRL_GPU_REG_OP_STATUS_SUCCESS
 *         This value indicates the operation completed successfully.
 *       LW2080_CTRL_GPU_REG_OP_STATUS_ILWALID_OP
 *         This bit value indicates that the regOp value is not valid.
 *       LW2080_CTRL_GPU_REG_OP_STATUS_ILWALID_TYPE
 *         This bit value indicates that the regType value is not valid.
 *       LW2080_CTRL_GPU_REG_OP_STATUS_ILWALID_OFFSET
 *         This bit value indicates that the regOffset value is invalid.
 *         The regOffset value must be within the legal BAR0 range for the
 *         associated GPU and must target a supported register with a
 *         supported operation.
 *       LW2080_CTRL_GPU_REG_OP_STATUS_UNSUPPORTED_OFFSET
 *         This bit value indicates that the operation to the register
 *         specified by the regOffset value is not supported for the
 *         associated GPU.
 *       LW2080_CTRL_GPU_REG_OP_STATUS_ILWALID_MASK
 *         This bit value indicates that the regTpcMask value is invalid.
 *         The regTpcMask must be a subset of TPCs that are enabled on the
 *         associated GPU.
 *       LW2080_CTRL_GPU_REG_OP_STATUS_NOACCESS
 *         The caller does not have access to the register at given offset
 *   regOffset
 *     This field specifies the register offset to access.  The specified
 *     offset must be a valid BAR0 offset for the associated GPU.
 *   regValueLo
 *     This field contains the low 32bits of the register value.
 *     For read operations, this value returns the current value of the
 *     register specified by regOffset.  For write operations, this field
 *     specifies the logical OR value applied to the current value
 *     contained in the register specified by regOffset.
 *   regValueHi
 *     This field contains the high 32bits of the register value.
 *     For read operations, this value returns the current value of the
 *     register specified by regOffset + 4.  For write operations, this field
 *     specifies the logical OR value applied to the current value
 *     contained in the register specified by regOffset + 4.
 *   regAndNMaskLo
 *     This field contains the mask used to clear a desired field from
 *     the current value contained in the register specified by regOffsetLo.
 *     This field is negated and ANDed to this current register value.
 *     This field is only used for write operations.  This field is ignored
 *     for read operations.
 *   regAndNMaskHi
 *     This field contains the mask used to clear a desired field from
 *     the current value contained in the register specified by regOffsetHi.
 *     This field is negated and ANDed to this current register value.
 *     This field is only used for write operations.  This field is ignored
 *     for read operations.
 */
typedef struct LW2080_CTRL_GPU_REG_OP {
    LwU8  regOp;
    LwU8  regType;
    LwU8  regStatus;
    LwU8  regQuad;
    LwU32 regGroupMask;
    LwU32 regSubGroupMask;
    LwU32 regOffset;
    LwU32 regValueHi;
    LwU32 regValueLo;
    LwU32 regAndNMaskHi;
    LwU32 regAndNMaskLo;
} LW2080_CTRL_GPU_REG_OP;

/* valid regOp values */
#define LW2080_CTRL_GPU_REG_OP_READ_32               (0x00000000)
#define LW2080_CTRL_GPU_REG_OP_WRITE_32              (0x00000001)
#define LW2080_CTRL_GPU_REG_OP_READ_64               (0x00000002)
#define LW2080_CTRL_GPU_REG_OP_WRITE_64              (0x00000003)
#define LW2080_CTRL_GPU_REG_OP_READ_08               (0x00000004)
#define LW2080_CTRL_GPU_REG_OP_WRITE_08              (0x00000005)

/* valid regType values */
#define LW2080_CTRL_GPU_REG_OP_TYPE_GLOBAL           (0x00000000)
#define LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX           (0x00000001)
#define LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_TPC       (0x00000002)
#define LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_SM        (0x00000004)
#define LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_CROP      (0x00000008)
#define LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_ZROP      (0x00000010)
#define LW2080_CTRL_GPU_REG_OP_TYPE_FB               (0x00000020)
#define LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX_QUAD      (0x00000040)
#define LW2080_CTRL_GPU_REG_OP_TYPE_DEVICE           (0x00000080)

/* valid regStatus values (note: LwU8 ie, 1 byte) */
#define LW2080_CTRL_GPU_REG_OP_STATUS_SUCCESS        (0x00)
#define LW2080_CTRL_GPU_REG_OP_STATUS_ILWALID_OP     (0x01)
#define LW2080_CTRL_GPU_REG_OP_STATUS_ILWALID_TYPE   (0x02)
#define LW2080_CTRL_GPU_REG_OP_STATUS_ILWALID_OFFSET (0x04)
#define LW2080_CTRL_GPU_REG_OP_STATUS_UNSUPPORTED_OP (0x08)
#define LW2080_CTRL_GPU_REG_OP_STATUS_ILWALID_MASK   (0x10)
#define LW2080_CTRL_GPU_REG_OP_STATUS_NOACCESS       (0x20)

/*
 * LW2080_CTRL_CMD_GPU_EXEC_REG_OPS
 *
 * This command is used to submit a buffer containing one or more
 * LW2080_CTRL_GPU_REG_OP structures for processing.  Each entry in the
 * buffer specifies a single read or write operation.  Each entry is checked
 * for validity in an initial pass over the buffer with the results for
 * each operation stored in the corresponding regStatus field. Unless
 * bNonTransactional flag is set to true, if any invalid entries are found
 * during this initial pass then none of the operations are exelwted. Entries
 * are processed in order within each regType with LW2080_CTRL_GPU_REG_OP_TYPE_GLOBAL
 * entries processed first followed by LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX entries.
 *
 *   hClientTarget
 *     This parameter specifies the handle of the client that owns the channel
 *     specified by hChannelTarget. If this parameter is set to 0 then the set
 *     of channel-specific register operations are applied to all current and
 *     future channels.
 *   hChannelTarget
 *     This parameter specifies the handle of the target channel (or channel
 *     group) object instance to which channel-specific register operations are
 *     to be directed. If hClientTarget is set to 0 then this parameter must
 *     also be set to 0.
 *   bNonTransactional
 *     This field specifies if command is non-transactional i.e. if set to
 *     true, all the valid operations will be exelwted.
 *   reserved00
 *     This parameter is reserved for future use.  It should be initialized to
 *     zero for correct operation.
 *   regOpCount
 *     This field specifies the number of entries on the caller's regOps
 *     list.
 *   regOps
 *     This field specifies a pointer in the caller's address space
 *     to the buffer from which the desired register information is to be
 *     retrieved.  This buffer must be at least as big as regInfoCount
 *     multiplied by the size of the LW2080_CTRL_GPU_REG_OP structure.
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine. When SMC is enabled, this
 *     is a mandatory parameter.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_GPU_EXEC_REG_OPS             (0x20800122) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x22" */

typedef struct LW2080_CTRL_GPU_EXEC_REG_OPS_PARAMS {
    LwHandle hClientTarget;
    LwHandle hChannelTarget;
    LwU32    bNonTransactional;
    LwU32    reserved00[2];
    LwU32    regOpCount;
    LW_DECLARE_ALIGNED(LwP64 regOps, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GPU_EXEC_REG_OPS_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_CMD_GPU_GET_ENGINES
 *
 * Returns a list of supported engine types along with the number of instances
 * of each type. Querying with engineList NULL returns engineCount.
 *
 *   engineCount
 *     This field specifies the number of entries on the caller's engineList
 *     field.
 *   engineList
 *     This field is a pointer to a buffer of LwU32 values representing the
 *     set of engines supported by the associated subdevice.  Refer to cl2080.h
 *     for the complete set of supported engine types.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_GPU_GET_ENGINES (0x20800123) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ENGINES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_ENGINES_PARAMS_MESSAGE_ID (0x23U)

typedef struct LW2080_CTRL_GPU_GET_ENGINES_PARAMS {
    LwU32 engineCount;
    LW_DECLARE_ALIGNED(LwP64 engineList, 8);
} LW2080_CTRL_GPU_GET_ENGINES_PARAMS;

#define LW2080_CTRL_CMD_GPU_GET_ENGINES_V2 (0x20800170) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ENGINES_V2_PARAMS_MESSAGE_ID" */

/* Must match LW2080_ENGINE_TYPE_LAST from cl2080.h */
#define LW2080_GPU_MAX_ENGINES_LIST_SIZE   0x34

#define LW2080_CTRL_GPU_GET_ENGINES_V2_PARAMS_MESSAGE_ID (0x70U)

typedef struct LW2080_CTRL_GPU_GET_ENGINES_V2_PARAMS {
    LwU32 engineCount;
    LwU32 engineList[LW2080_GPU_MAX_ENGINES_LIST_SIZE];
} LW2080_CTRL_GPU_GET_ENGINES_V2_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_ENGINE_CLASSLIST
 *
 * Returns a list of classes supported by a given engine type.
 *
 *   engineType
 *     This field specifies the engine type being queried.
 *     LW2080_CTRL_ENGINE_TYPE_ALLENGINES will return  classes
 *     supported by all engines.
 *
 *   numClasses
 *     This field specifies the number of classes supported by
 *     engineType.
 *
 *   classList
 *     This field is an array containing the list of supported
 *     classes. Is of type (LwU32*)
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_GPU_GET_ENGINE_CLASSLIST (0x20800124) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ENGINE_CLASSLIST_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_ENGINE_CLASSLIST_PARAMS_MESSAGE_ID (0x24U)

typedef struct LW2080_CTRL_GPU_GET_ENGINE_CLASSLIST_PARAMS {
    LwU32 engineType;
    LwU32 numClasses;
    LW_DECLARE_ALIGNED(LwP64 classList, 8);
} LW2080_CTRL_GPU_GET_ENGINE_CLASSLIST_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GPU_GET_ENGINE_FAULT_INFO
 *
 * This command returns the fault properties of the specified engine type.
 *
 *   engineType
 *     Input parameter.
 *     This field specifies the engine type being queried.
 *     Engine type is specified using the LW2080_ENGINE_TYPE_* defines in cl2080.h.
 *     The list of engines supported by a chip can be got using the
 *     LW2080_CTRL_CMD_GPU_GET_ENGINES ctrl call.
 *
 *   mmuFaultId
 *     Output parameter.
 *     This field returns the MMU fault ID for the specified engine.
 *     If the engine supports subcontext, this field provides the base fault id.
 *
 *   bSubcontextSupported
 *     Output parameter.
 *     Returns TRUE if subcontext faulting is supported by the engine.
 *     Engine that support subcontext use fault IDs in the range [mmuFaultId, mmuFaultId + maxSubCtx).
 *     "maxSubctx" can be found using the LW2080_CTRL_FIFO_INFO ctrl call with
 *     LW2080_CTRL_FIFO_INFO_INDEX_MAX_SUBCONTEXT_PER_GROUP as the index.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_GPU_GET_ENGINE_FAULT_INFO (0x20800125) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ENGINE_FAULT_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_ENGINE_FAULT_INFO_PARAMS_MESSAGE_ID (0x25U)

typedef struct LW2080_CTRL_GPU_GET_ENGINE_FAULT_INFO_PARAMS {
    LwU32  engineType;
    LwU32  mmuFaultId;
    LwBool bSubcontextSupported;
} LW2080_CTRL_GPU_GET_ENGINE_FAULT_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_HYBRID_CONTROLLER_INFO
 *
 * This command is used to get hybrid controller related info.
 *
 *   present
 *     This parameter is used to return whether the hybrid controller
 *     is present or not. If the controller is present it will take value
 *     LW_TRUE else LW_FALSE.
 *
 *   fwVersion
 *     This parameter is used to return the firmware version of the hybrid
 *     controller. Firmware version is zero if the hybrid controller is not
 *     present.
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW2080_CTRL_CMD_GPU_GET_HYBRID_CONTROLLER_INFO (0x20800126) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x26" */

typedef struct LW2080_CTRL_CMD_GPU_GET_HYBRID_CONTROLLER_INFO_PARAMS {
    LwBool present;
    LwU32  fwVersion;
} LW2080_CTRL_CMD_GPU_GET_HYBRID_CONTROLLER_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_ENGR_REV_INFO
 *
 * This command is used to get the engineering revision info.
 *
 *   majorRev
 *     This parameter contains the engineering major revision id of this
 *     gpu.
 *   minorRev
 *     This parameter contains the engineering minor revision id of this
 *     gpu.
 *   minorExtRev
 *     This parameter contains the engineering minor extended revision id
 *     of this gpu.  For example, on a GT200 B02P gpu, this value would be
 *     one.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *
 */
#define LW2080_CTRL_CMD_GPU_GET_ENGR_REV_INFO (0x20800127) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_CMD_GPU_GET_ENGR_REV_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_GPU_GET_ENGR_REV_INFO_PARAMS_MESSAGE_ID (0x27U)

typedef struct LW2080_CTRL_CMD_GPU_GET_ENGR_REV_INFO_PARAMS {
    LwU32 majorRev;
    LwU32 minorRev;
    LwU32 minorExtRev;
} LW2080_CTRL_CMD_GPU_GET_ENGR_REV_INFO_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_CMD_GPU_QUERY_MODE
 *
 * This command is used to detect the mode of the GPU associated with the
 * subdevice.
 *
 *   mode
 *     This parameter returns the current mode of GPU.  Legal values for
 *     this parameter include:
 *       LW2080_CTRL_GPU_QUERY_MODE_GRAPHICS_MODE
 *         The GPU is lwrrently operating in graphics mode.
 *       LW2080_CTRL_GPU_QUERY_MODE_COMPUTE_MODE
 *         The GPU is lwrrently operating in compute mode.
 *       LW2080_CTRL_GPU_QUERY_MODE_UNKNOWN_MODE
 *         The current mode of the GPU could not be determined.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_GPU_QUERY_MODE           (0x20800128) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_QUERY_MODE_PARAMS_MESSAGE_ID" */

/* valid mode parameter values */
#define LW2080_CTRL_GPU_QUERY_MODE_UNKNOWN_MODE  (0x00000000)
#define LW2080_CTRL_GPU_QUERY_MODE_GRAPHICS_MODE (0x00000001)
#define LW2080_CTRL_GPU_QUERY_MODE_COMPUTE_MODE  (0x00000002)

#define LW2080_CTRL_GPU_QUERY_MODE_PARAMS_MESSAGE_ID (0x28U)

typedef struct LW2080_CTRL_GPU_QUERY_MODE_PARAMS {
    LwU32 mode;
} LW2080_CTRL_GPU_QUERY_MODE_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


/*
 * LW2080_CTRL_CMD_GPU_SET_DEEP_IDLE_MODE
 *
 * This command sets the Deep Idle mode for the GPU.
 *
 *   mode
 *     Specifies the Deep Idle mode in which to place the GPU.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_SET_DEEP_IDLE_MODE (0x20800129) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x29" */

typedef struct LW2080_CTRL_GPU_SET_DEEP_IDLE_MODE_PARAMS {
    LwU32 mode;
} LW2080_CTRL_GPU_SET_DEEP_IDLE_MODE_PARAMS;

/* valid mode parameter values */
#define LW2080_CTRL_GPU_SET_DEEP_IDLE_MODE_OFF   (0x00000000)
#define LW2080_CTRL_GPU_SET_DEEP_IDLE_MODE_ON    (0x00000001)

/*
 * LW2080_CTRL_CMD_GPU_QUERY_DEEP_IDLE_MODE
 *
 * This command queries the mode of the Deep Idle state for the GPU.
 *
 *   mode
 *     Specifies the Deep Idle mode the GPU is lwrrently in; undefined in
 *     the case where Deep Idle is not supported.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_QUERY_DEEP_IDLE_MODE (0x2080012a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x2A" */

typedef struct LW2080_CTRL_GPU_QUERY_DEEP_IDLE_MODE_PARAMS {
    LwU32 mode;
} LW2080_CTRL_GPU_QUERY_DEEP_IDLE_MODE_PARAMS;

/* valid mode parameter values */
#define LW2080_CTRL_GPU_QUERY_DEEP_IDLE_MODE_OFF    (0x00000000)
#define LW2080_CTRL_GPU_QUERY_DEEP_IDLE_MODE_ON     (0x00000001)


/*
 * LW2080_CTRL_CMD_GPU_QUERY_DEEP_IDLE_SUPPORT
 *
 * This command queries for Deep Idle support and returns an array of pstates which support Deep Idle
 * If Deep Idle is not supported. It returns the flags which state the reasons why Deep Idle is not supported.
 *
 * numStates
 *    This parameter states the total number of pstates in which Deep Idle is supported(if Deep Idle is supported at all)
 * pstates
 *    This parameter is an array of pstates in which Deep Idle is supported. The array is filled up to NumStates number of pstates.
 * flags
 *    This parameter indicates whether Deep Idle is supported or not. It can have the following values
 *
 *      LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_SUPPORT_YES
 *         Indicates that Deep Idle is supported.
 *      LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_SUPPORT_NO
 *         Indicates that Deep Idle is not supported.
 *      LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_CHIPSET_NOT_CAPABLE
 *         This flag is no longer valid.
 *      LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_CHIPSET_CAPABLE
 *         This flag is no longer valid.
 *      LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_GPU_NOT_CAPABLE
 *         Indicates that GPU is not capable.
 *      LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_GPU_CAPABLE
 *         Indicates that GPU is capable.
 *      LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_RM_POWER_FEATURE_REGKEY_OFF
 *         Indicates that RMPowerFeature regkey is off.
 *      LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_RM_POWER_FEATURE_REGKEY_ON
 *         Indicates that RMPowerFeature regkey is on.
 *      LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_OPSB_REGKEY_OFF
 *         Indicates that OPSB regkey is off.
 *      LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_OPSB_REGKEY_ON
 *         Indicates that OPSB regkey is on
 *
 */

#define MAX_PSTATES                                 (0x00000010)

#define LW2080_CTRL_CMD_GPU_QUERY_DEEP_IDLE_SUPPORT (0x20800162) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_QUERY_DEEP_IDLE_SUPPORT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_QUERY_DEEP_IDLE_SUPPORT_PARAMS_MESSAGE_ID (0x62U)

typedef struct LW2080_CTRL_GPU_QUERY_DEEP_IDLE_SUPPORT_PARAMS {
    LwU8 numStates;
    LwU8 pstates[MAX_PSTATES];
    LwU8 flags;
} LW2080_CTRL_GPU_QUERY_DEEP_IDLE_SUPPORT_PARAMS;

#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_SUPPORT                                      0:0
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_SUPPORT_YES                 (0x00000000)
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_SUPPORT_NO                  (0x00000001)

// _FLAGS_CHIPSET is deprecated
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_CHIPSET                                      1:1
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_CHIPSET_CAPABLE             (0x00000000)
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_CHIPSET_NOT_CAPABLE         (0x00000001)

// _FLAGS_GPU is deprecated
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_GPU                                          2:2
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_GPU_CAPABLE                 (0x00000000)
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_GPU_NOT_CAPABLE             (0x00000001)
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_RM_POWER_FEATURE_REGKEY                      3:3
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_RM_POWER_FEATURE_REGKEY_ON  (0x00000000)
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_RM_POWER_FEATURE_REGKEY_OFF (0x00000001)
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_OPSB_REGKEY                                  4:4
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_OPSB_REGKEY_ON              (0x00000000)
#define LW2080_CTRL_GPU_DEEP_IDLE_FLAGS_OPSB_REGKEY_OFF             (0x00000001)

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*!
 * LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY
 * Data block describing a virtual context buffer to be promoted
 *
 *  gpuPhysAddr [IN]
 *    GPU Physical Address for the buffer
 *  gpuVirtAddr [IN]
 *    GPU Virtual Address for the buffer
 *   size[IN]
 *    Size of this virtual context buffer
 *  physAttr [IN]
 *    Physical memory attributes (aperture, cacheable)
 *  bufferId [IN]
 *    Virtual context buffer type, data type LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_*
 *  bInitialize [IN]
 *   Flag indicating that this virtual context buffer should be initialized prior to promotion.
 *   The client must clear (memset) the buffer to 0x0 prior to initialization.
 *   Following buffers need initialization:
 *    1. LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_MAIN
 *    2. LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PATCH
 *    3. LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PRIV_ACCESS_MAP
 *    4. LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_UNRESTRICTED_PRIV_ACCESS_MAP
 *  bNonmapped [IN]
 *   Flag indicating that the virtual address is not to be promoted with this
 *   call. It is illegal to set this flag and not set bInitialize.
 */
typedef struct LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY {
    LW_DECLARE_ALIGNED(LwU64 gpuPhysAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 gpuVirtAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 size, 8);
    LwU32 physAttr;
    LwU16 bufferId;
    LwU8  bInitialize;
    LwU8  bNonmapped;
} LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY;

#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_MAIN                         0
#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PM                           1
#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PATCH                        2
#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_BUFFER_BUNDLE_CB             3
#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PAGEPOOL                     4
#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_ATTRIBUTE_CB                 5
#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_RTV_CB_GLOBAL                6
#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_GFXP_POOL                    7
#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_GFXP_CTRL_BLK                8
#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_FECS_EVENT                   9
#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PRIV_ACCESS_MAP              10
#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_UNRESTRICTED_PRIV_ACCESS_MAP 11
#define LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_GLOBAL_PRIV_ACCESS_MAP       12

#define LW2080_CTRL_GPU_PROMOTE_CONTEXT_MAX_ENTRIES                        16

/*
 * LW2080_CTRL_CMD_GPU_PROMOTE_CTX
 *
 * This command is used to promote a Virtual Context
 *
 *   engineType
 *     Engine Virtual Context is for
 *   hClient
 *     Client Handle for hVirtMemory
 *   ChID
 *     Hw Channel -- Actually hw index for channel (deprecated)
 *   hChanClient
 *     The client handle for hObject
 *   hObject
 *     Passed in object handle for either a single channel or a channel group
 *   hVirtMemory
 *     Virtual Address handle to map Virtual Context to
 *   virtAddress
 *     Virtual Address to map Virtual Context to
 *   size
 *     size of the Virtual Context
 *   entryCount
 *     Number of valid entries in the promotion entry list
 *   promoteEntry
 *     List of context buffer entries to issue promotions for.
 *
 *   When not using promoteEntry, only hVirtMemory or (virtAddress, size) should be 
 *   specified, the code cases based on hVirtMemory(NULL vs non-NULL) so 
 *   if both are specified, hVirtMemory has precedence. 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED    - The Class does not support version info retrieval
 *   LW_ERR_ILWALID_DEVICE   - The Class/Device is not yet ready to provide this info.
 *   LW_ERR_ILWALID_ARGUMENT - Bad/Unknown Class ID specified.
 */
#define LW2080_CTRL_CMD_GPU_PROMOTE_CTX                                    (0x2080012b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_PROMOTE_CTX_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_PROMOTE_CTX_PARAMS_MESSAGE_ID (0x2BU)

typedef struct LW2080_CTRL_GPU_PROMOTE_CTX_PARAMS {
    LwU32    engineType;
    LwHandle hClient;
    LwU32    ChID;
    LwHandle hChanClient;
    LwHandle hObject;
    LwHandle hVirtMemory;
    LW_DECLARE_ALIGNED(LwU64 virtAddress, 8);
    LW_DECLARE_ALIGNED(LwU64 size, 8);
    LwU32    entryCount;
    // C form: LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY promoteEntry[LW2080_CTRL_GPU_PROMOTE_CONTEXT_MAX_ENTRIES];
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY promoteEntry[LW2080_CTRL_GPU_PROMOTE_CONTEXT_MAX_ENTRIES], 8);
} LW2080_CTRL_GPU_PROMOTE_CTX_PARAMS;
typedef struct LW2080_CTRL_GPU_PROMOTE_CTX_PARAMS *PLW2080_CTRL_GPU_PROMOTE_CTX_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_EVICT_CTX
 *
 * This command is used to evict a Virtual Context
 *
 *   engineType
 *     Engine Virtual Context is for
 *   hClient
 *     Client Handle
 *   ChID
 *     Hw Channel -- Actually hw index for channel (deprecated)
 *   hChanClient
 *     Client handle for hObject
 *   hObject
 *     Passed in object handle for either a single channel or a channel group
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED    - The Class does not support version info retrieval
 *   LW_ERR_ILWALID_DEVICE   - The Class/Device is not yet ready to provide this info.
 *   LW_ERR_ILWALID_ARGUMENT - Bad/Unknown Class ID specified.
 */
#define LW2080_CTRL_CMD_GPU_EVICT_CTX (0x2080012c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_EVICT_CTX_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_EVICT_CTX_PARAMS_MESSAGE_ID (0x2LW)

typedef struct LW2080_CTRL_GPU_EVICT_CTX_PARAMS {
    LwU32    engineType;
    LwHandle hClient;
    LwU32    ChID;
    LwHandle hChanClient;
    LwHandle hObject;
} LW2080_CTRL_GPU_EVICT_CTX_PARAMS;
typedef struct LW2080_CTRL_GPU_EVICT_CTX_PARAMS *PLW2080_CTRL_GPU_EVICT_CTX_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_INITIALIZE_CTX
 *
 * This command is used to initialize a Virtual Context. The ctx buffer must be
 * cleared (zerod) by the caller prior to ilwoking this method.
 *
 *   engineType
 *     Engine Virtual Context is for
 *   hClient
 *     Client Handle for the hVirtMemory
 *   ChID
 *      Hw channel -- Actually channel index (deprecated)
 *   hChanClient
 *     The client handle for hObject
 *   hObject
 *     Passed in object handle for either a single channel or a channel group
 *   hVirtMemory
 *     Virtual Address where to map Virtual Context to
 *   physAddress
 *     Physical offset in FB to use as Virtual Context
 *   physAttr
 *     Physical memory attributes
 *   hDmaHandle
 *     Dma Handle when using discontiguous context buffers
 *   index
 *     Start offset in Virtual DMA Context
 *   size
 *     Size of the Virtual Context
 *
 *   Only hVirtMemory or size should be specified, the code cases based on hVirtMemory
 *   (NULL vs non-NULL) so if both are specified, hVirtMemory has precedence.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED    - The Class does not support version info retrieval
 *   LW_ERR_ILWALID_DEVICE   - The Class/Device is not yet ready to provide this info.
 *   LW_ERR_ILWALID_ARGUMENT - Bad/Unknown Class ID specified.
 */
#define LW2080_CTRL_CMD_GPU_INITIALIZE_CTX (0x2080012d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_INITIALIZE_CTX_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_INITIALIZE_CTX_PARAMS_MESSAGE_ID (0x2DU)

typedef struct LW2080_CTRL_GPU_INITIALIZE_CTX_PARAMS {
    LwU32    engineType;
    LwHandle hClient;
    LwU32    ChID;
    LwHandle hChanClient;
    LwHandle hObject;
    LwHandle hVirtMemory;
    LW_DECLARE_ALIGNED(LwU64 physAddress, 8);
    LwU32    physAttr;
    LwHandle hDmaHandle;
    LwU32    index;
    LW_DECLARE_ALIGNED(LwU64 size, 8);
} LW2080_CTRL_GPU_INITIALIZE_CTX_PARAMS;
typedef struct LW2080_CTRL_GPU_INITIALIZE_CTX_PARAMS *PLW2080_CTRL_GPU_INITIALIZE_CTX_PARAMS;

#define LW2080_CTRL_GPU_INITIALIZE_CTX_APERTURE              1:0
#define LW2080_CTRL_GPU_INITIALIZE_CTX_APERTURE_VIDMEM   (0x00000000)
#define LW2080_CTRL_GPU_INITIALIZE_CTX_APERTURE_COH_SYS  (0x00000001)
#define LW2080_CTRL_GPU_INITIALIZE_CTX_APERTURE_NCOH_SYS (0x00000002)

#define LW2080_CTRL_GPU_INITIALIZE_CTX_GPU_CACHEABLE         2:2
#define LW2080_CTRL_GPU_INITIALIZE_CTX_GPU_CACHEABLE_YES (0x00000000)
#define LW2080_CTRL_GPU_INITIALIZE_CTX_GPU_CACHEABLE_NO  (0x00000001)

/*
 * LW2080_CTRL_GPU_INITIALIZE_CTX_PRESERVE_CTX - Tells RM Whether this Ctx buffer needs to
 * do a full initialization (Load the golden image). When a context is promoted on a different
 * channel than it was originally inited, the client can use this flag to tell RM
 * that this is an already inited Context. In such cases RM will update the internal state
 * to update the context address and state variables.
 */

#define LW2080_CTRL_GPU_INITIALIZE_CTX_PRESERVE_CTX              3:3
#define LW2080_CTRL_GPU_INITIALIZE_CTX_PRESERVE_CTX_NO   (0x00000000)
#define LW2080_CTRL_GPU_INITIALIZE_CTX_PRESERVE_CTX_YES  (0x00000001)

/*
 * LW2080_CTRL_CMD_CPU_QUERY_ECC_INTR
 * Queries the top level ECC PMC PRI register
 * TODO remove these parameters, tracked in bug #1975721
 */
#define LW2080_CTRL_CMD_GPU_QUERY_ECC_INTR               (0x2080012e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x2E" */

typedef struct LW2080_CTRL_GPU_QUERY_ECC_INTR_PARAMS {
    LwU32 eccIntrStatus;
} LW2080_CTRL_GPU_QUERY_ECC_INTR_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/**
 * LW2080_CTRL_CMD_GPU_QUERY_ECC_STATUS
 *
 * This command is used to query the ECC status of a GPU by a subdevice
 * handle.  Please see the LW2080_CTRL_GPU_QUERY_ECC_UNIT_STATUS
 * data structure description below for details on the data reported
 * per hardware unit.
 *
 *   units
 *     Array of structures used to describe per-unit state
 *
 *   flags
 *     See interface flag definitions below.
 *
 * Note that if ECC is statically disabled on a SKU, e.g. a on consumer
 * board, this command will report the feature as unsupported, even
 * if the GPU does support ECC.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_QUERY_ECC_STATUS                   (0x2080012f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_QUERY_ECC_STATUS_PARAMS_MESSAGE_ID" */

/* ECC unit list */
#define LW2080_CTRL_GPU_ECC_UNIT_FBPA                          (0x00000000)
#define LW2080_CTRL_GPU_ECC_UNIT_L2                            (0x00000001)
#define LW2080_CTRL_GPU_ECC_UNIT_L1                            (0x00000002)
#define LW2080_CTRL_GPU_ECC_UNIT_SM                            (0x00000003)
#define LW2080_CTRL_GPU_ECC_UNIT_SM_L1_DATA                    (0x00000004)
#define LW2080_CTRL_GPU_ECC_UNIT_SM_L1_TAG                     (0x00000005)
#define LW2080_CTRL_GPU_ECC_UNIT_SM_CBU                        (0x00000006)
#define LW2080_CTRL_GPU_ECC_UNIT_SHM                           (0x00000007)
#define LW2080_CTRL_GPU_ECC_UNIT_TEX                           (0x00000008)
#define LW2080_CTRL_GPU_ECC_UNIT_SM_ICACHE                     (0x00000009)
#define LW2080_CTRL_GPU_ECC_UNIT_GCC                           (0x0000000A)
#define LW2080_CTRL_GPU_ECC_UNIT_GPCMMU                        (0x0000000B)
#define LW2080_CTRL_GPU_ECC_UNIT_HUBMMU_L2TLB                  (0x0000000C)
#define LW2080_CTRL_GPU_ECC_UNIT_HUBMMU_HUBTLB                 (0x0000000D)
#define LW2080_CTRL_GPU_ECC_UNIT_HUBMMU_FILLUNIT               (0x0000000E)
#define LW2080_CTRL_GPU_ECC_UNIT_GPCCS                         (0x0000000F)
#define LW2080_CTRL_GPU_ECC_UNIT_FECS                          (0x00000010)
#define LW2080_CTRL_GPU_ECC_UNIT_PMU                           (0x00000011)
#define LW2080_CTRL_GPU_ECC_UNIT_SM_RAMS                       (0x00000012)
#define LW2080_CTRL_GPU_ECC_UNIT_HSHUB                         (0x00000013)
#define LW2080_CTRL_GPU_ECC_UNIT_PCIE_REORDER                  (0x00000014)
#define LW2080_CTRL_GPU_ECC_UNIT_PCIE_P2PREQ                   (0x00000015)
#define LW2080_CTRL_GPU_ECC_UNIT_COUNT                         (0x00000016)


/*
 * Due to limitations of early ECC hardware, the RM may need to limit
 * the number of errors reported; e.g. it may be forced to report
 * no more than a single double-bit error, or to omit reporting of
 * single-bit errors completely.
 *
 * For RM some clients, such as MODS, this may not be sufficient. In
 * those cases, the RM can be instructed to return the errors as
 * they are obtained from the hardware itself, unfiltered.
 */
#define LW2080_CTRL_GPU_QUERY_ECC_STATUS_FLAGS_TYPE             0:0
#define LW2080_CTRL_GPU_QUERY_ECC_STATUS_FLAGS_TYPE_FILTERED   (0x00000000)
#define LW2080_CTRL_GPU_QUERY_ECC_STATUS_FLAGS_TYPE_RAW        (0x00000001)

#define LW2080_CTRL_GPU_QUERY_ECC_STATUS_UNC_ERR_FALSE         0
#define LW2080_CTRL_GPU_QUERY_ECC_STATUS_UNC_ERR_TRUE          1
#define LW2080_CTRL_GPU_QUERY_ECC_STATUS_UNC_ERR_INDETERMINATE 2

/*
 * LW2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS
 *
 * This structure represents the exception status of a class of per-unit
 * exceptions
 *
 *   count
 *     number of exceptions that have oclwrred since boot
 */
typedef struct LW2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS {
    LW_DECLARE_ALIGNED(LwU64 count, 8);
} LW2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS;

/*
 * LW2080_CTRL_GPU_QUERY_ECC_UNIT_STATUS
 *
 * This structure represents the per-unit ECC exception status
 *
 *   enabled
 *     ECC enabled yes/no for this unit
 *   scrubComplete
 *     Scrub has completed yes/no. A scrub is performed for some units to ensure
 *     the checkbits are consistent with the protected data.
 *   supported
 *     Whether HW supports ECC in this unit for this GPU
 *   dbe
 *     Double bit error (DBE) status. The value returned reflects a counter
 *     that is monotonic, but can be reset by clients.
 *   dbeNonResettable
 *     Double bit error (DBE) status, not client resettable.
 *   sbe
 *     Single bit error (SBE) status. The value returned reflects a counter
 *     that is monotonic, but can be reset by clients.
 *   sbeNonResettable
 *     Single bit error (SBE) status, not client resettable.
 *
 */
typedef struct LW2080_CTRL_GPU_QUERY_ECC_UNIT_STATUS {
    LwBool enabled;
    LwBool scrubComplete;
    LwBool supported;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS dbe, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS dbeNonResettable, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS sbe, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS sbeNonResettable, 8);
} LW2080_CTRL_GPU_QUERY_ECC_UNIT_STATUS;

/*
 * LW2080_CTRL_GPU_QUERY_ECC_STATUS_PARAMS
 *
 * This structure returns ECC exception status and GPU Fatal Poison for all units
 *
 *   units
 *     This structure represents ECC exception status for all Units.
 *   bFatalPoisonError
 *     Whether GPU Fatal poison error oclwrred in this GPU. This will be set for Ampere_and_later
 *   uncorrectableError
 *     Indicates whether any uncorrectable GR ECC errors have oclwrred. When
 *     SMC is enabled, uncorrectableError is only valid when the client is
 *     subscribed to a partition. Check QUERY_ECC_STATUS_UNC_ERR_*
 *   flags
 *     Flags passed by caller. Refer  LW2080_CTRL_GPU_QUERY_ECC_STATUS_FLAGS_TYPE_* for details.
 *   grRouteInfo
 *     SMC partition information. This input is only valid when SMC is
 *     enabled on Ampere_and_later.
 *
 */
#define LW2080_CTRL_GPU_QUERY_ECC_STATUS_PARAMS_MESSAGE_ID (0x2FU)

typedef struct LW2080_CTRL_GPU_QUERY_ECC_STATUS_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_QUERY_ECC_UNIT_STATUS units[LW2080_CTRL_GPU_ECC_UNIT_COUNT], 8);
    LwBool bFatalPoisonError;
    LwU8   uncorrectableError;
    LwU32  flags;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GPU_QUERY_ECC_STATUS_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_SET_COMPUTE_MODE_RULES
 *
 * This command sets the compute mode rules for the associated subdevice.  The
 * default mode is equivalent to LW2080_CTRL_GPU_COMPUTE_MODE_RULES_NONE.  This
 * command is available to clients with administrator privileges only.  An
 * attempt to use this command by a client without administrator privileged
 * results in the return of an LW_ERR_INSUFFICIENT_PERMISSIONS status.
 *
 *   rules
 *     This parameter is used to specify the rules that govern the GPU with
 *     respect to LW50_COMPUTE objects. Legal values for this parameter include:
 *
 *       LW2080_CTRL_GPU_COMPUTE_MODE_RULES_NONE
 *         This mode indicate that no special restrictions apply to the
 *         allocation of LW50_COMPUTE objects.
 *
 *       LW2080_CTRL_GPU_COMPUTE_MODE_RULES_EXCLUSIVE_COMPUTE
 *         This mode means that only one instance of LW50_COMPUTE will be
 *         allowed at a time. This restriction is enforced at each subsequent
 *         LW50_COMPUTE allocation attempt. Setting this mode will not affect
 *         any existing compute programs that may be running. For example,
 *         if this mode is set while three compute programs are running, then
 *         all of those programs will be allowed to continue running. However,
 *         until they all finish running, no new LW50_COMPUTE objects may be
 *         allocated. User-mode clients should treat this as restricting access
 *         to a LW50_COMPUTE object to a single thread within a process.
 *
 *       LW2080_CTRL_GPU_COMPUTE_MODE_RULES_COMPUTE_PROHIBITED
 *         This mode means that that GPU is not ever allowed to instantiate an
 *         LW50_COMPUTE object, and thus cannot run any new compute programs.
 *         This restriction is enforced at each subsequent LW50_COMPUTE object
 *         allocation attempt. Setting this mode will not affect any existing
 *         compute programs that may be running. For example, if this mode is
 *         set while three compute programs are running, then all of those
 *         programs will be allowed to continue running. However, no new
 *         LW50_COMPUTE objects may be allocated.
 *
 *
 *       LW2080_CTRL_GPU_COMPUTE_MODE_EXCLUSIVE_COMPUTE_PROCESS
 *         This mode is identical to EXCLUSIVE_COMPUTE, where only one instance
 *         of LW50_COMPUTE will be allowed at a time. It is separate from
 *         EXCLUSIVE_COMPUTE to allow user-mode clients to differentiate
 *         exclusive access to a compute object from a single thread of a
 *         process from exclusive access to a compute object from all threads
 *         of a process. User-mode clients should not limit access to a
 *         LW50_COMPUTE object to a single thread when the GPU is set to
 *         EXCLUSIVE_COMPUTE_PROCESS.
 *
 *     An invalid rules parameter value results in the return of an
 *     LW_ERR_ILWALID_ARGUMENT status.
 *
 *   flags
 *     Reserved. Caller should set this field to zero.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT (if an invalid rule number is provided)
 *   LW_ERR_INSUFFICIENT_PERMISSIONS (if the user is not the Administrator or superuser)
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_SET_COMPUTE_MODE_RULES                   (0x20800130) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_SET_COMPUTE_MODE_RULES_PARAMS_MESSAGE_ID" */

/* valid rules parameter values */
#define LW2080_CTRL_GPU_COMPUTE_MODE_RULES_NONE                      (0x00000000)
#define LW2080_CTRL_GPU_COMPUTE_MODE_RULES_EXCLUSIVE_COMPUTE         (0x00000001)
#define LW2080_CTRL_GPU_COMPUTE_MODE_RULES_COMPUTE_PROHIBITED        (0x00000002)
#define LW2080_CTRL_GPU_COMPUTE_MODE_RULES_EXCLUSIVE_COMPUTE_PROCESS (0x00000003)

#define LW2080_CTRL_GPU_SET_COMPUTE_MODE_RULES_PARAMS_MESSAGE_ID (0x30U)

typedef struct LW2080_CTRL_GPU_SET_COMPUTE_MODE_RULES_PARAMS {
    LwU32 rules;
    LwU32 flags;
} LW2080_CTRL_GPU_SET_COMPUTE_MODE_RULES_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_QUERY_COMPUTE_MODE_RULES
 *
 * This command queries the compute mode rules for the associated subdevice.
 * Please see the LW2080_CTRL_CMD_GPU_SET_COMPUTE_MODE_RULES command, above, for
 * details as to what the rules mean.
 *
 *   rules
 *     Specifies the rules that govern the GPU, with respect to LW50_COMPUTE
 *     objects.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_QUERY_COMPUTE_MODE_RULES (0x20800131) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_QUERY_COMPUTE_MODE_RULES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_QUERY_COMPUTE_MODE_RULES_PARAMS_MESSAGE_ID (0x31U)

typedef struct LW2080_CTRL_GPU_QUERY_COMPUTE_MODE_RULES_PARAMS {
    LwU32 rules;
} LW2080_CTRL_GPU_QUERY_COMPUTE_MODE_RULES_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_SET_HYBRID_CONTROLLER_FW
 *
 * This command is used to get hybrid controller related info.
 *
 *   present
 *     This parameter is used to return whether the hybrid controller
 *     is present or not. If the controller is present it will take value
 *     LW_TRUE else LW_FALSE.
 *
 *   fwVersion
 *     This parameter is used to return the firmware version of the hybrid
 *     controller. Firmware version is zero if the hybrid controller is not
 *     present.
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW2080_CTRL_CMD_GPU_SET_HYBRID_CONTROLLER_FW (0x20800132) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x32" */

typedef struct LW2080_CTRL_CMD_GPU_SET_HYBRID_CONTROLLER_FW_PARAMS {
    LwU32 inSize;
    LW_DECLARE_ALIGNED(LwP64 inBytes, 8);
} LW2080_CTRL_CMD_GPU_SET_HYBRID_CONTROLLER_FW_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_QUERY_ECC_CONFIGURATION
 *
 * This command returns the current ECC configuration setting for
 * a GPU given its subdevice handle.  The value returned is
 * the current ECC setting for the GPU stored in non-volatile
 * memory on the board.
 *
 *   lwrrentConfiguration
 *      The current ECC configuration setting.
 *
 *   defaultConfiguration
 *      The factory default ECC configuration setting.
 *
 * Please see the LW2080_CTRL_CMD_GPU_QUERY_ECC_STATUS command if
 * you wish to determine if ECC is lwrrently enabled.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_GPU_QUERY_ECC_CONFIGURATION (0x20800133) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_QUERY_ECC_CONFIGURATION_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_ECC_CONFIGURATION_DISABLED  (0x00000000)
#define LW2080_CTRL_GPU_ECC_CONFIGURATION_ENABLED   (0x00000001)

#define LW2080_CTRL_GPU_QUERY_ECC_CONFIGURATION_PARAMS_MESSAGE_ID (0x33U)

typedef struct LW2080_CTRL_GPU_QUERY_ECC_CONFIGURATION_PARAMS {
    LwU32 lwrrentConfiguration;
    LwU32 defaultConfiguration;
} LW2080_CTRL_GPU_QUERY_ECC_CONFIGURATION_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_SET_ECC_CONFIGURATION
 *
 * This command changes the ECC configuration setting for a GPU
 * given its subdevice handle.  The value specified is
 * stored in non-volatile memory on the board and will take
 * effect with the next VBIOS POST.
 *
 *   newConfiguration
 *     The new configuration setting to take effect with
 *     the next VBIOS POST.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_SET_ECC_CONFIGURATION (0x20800134) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_SET_ECC_CONFIGURATION_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_ECC_CONFIGURATION_DISABLE (0x00000000)
#define LW2080_CTRL_GPU_ECC_CONFIGURATION_ENABLE  (0x00000001)

#define LW2080_CTRL_GPU_SET_ECC_CONFIGURATION_PARAMS_MESSAGE_ID (0x34U)

typedef struct LW2080_CTRL_GPU_SET_ECC_CONFIGURATION_PARAMS {
    LwU32 newConfiguration;
} LW2080_CTRL_GPU_SET_ECC_CONFIGURATION_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_QUERY_AGGREGATE_ECC_STATUS
 *
 * This command reports the aggregate ECC exception counts across
 * units for the lifetime of a given GPU, to the extent the
 * RM can determine these values.
 *
 *   sbe
 *     This parameter returns the number of single bit
 *     error (SBE) errors
 *
 *   dbe
 *     This parameter returns the number of double bit
 *     error (DBE) errors
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_QUERY_AGGREGATE_ECC_STATUS (0x20800135) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x35" */

typedef struct LW2080_CTRL_GPU_QUERY_AGGREGATE_ECC_STATUS_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS sbe, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS dbe, 8);
} LW2080_CTRL_GPU_QUERY_AGGREGATE_ECC_STATUS_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_RESET_ECC_ERROR_STATUS
 *
 * This command resets volatile and/or persistent ECC error
 * status information for a GPU given its subdevice
 * handle.
 *
 *   statuses
 *     The ECC error statuses (the current, volatile
 *     and/or the persistent error counter(s)) to
 *     be reset by the command.
 *   flags
 *     FORCE_PURGE
 *          Forcibly clean all the ECC InfoROM state if this flag is set
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_RESET_ECC_ERROR_STATUS                     (0x20800136) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_RESET_ECC_ERROR_STATUS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_ECC_ERROR_STATUS_NONE                          (0x00000000)
#define LW2080_CTRL_GPU_ECC_ERROR_STATUS_VOLATILE                      (0x00000001)
#define LW2080_CTRL_GPU_ECC_ERROR_STATUS_AGGREGATE                     (0x00000002)

#define LW2080_CTRL_GPU_RESET_ECC_ERROR_STATUS_FLAGS_FORCE_PURGE           0:0
#define LW2080_CTRL_GPU_RESET_ECC_ERROR_STATUS_FLAGS_FORCE_PURGE_FALSE 0
#define LW2080_CTRL_GPU_RESET_ECC_ERROR_STATUS_FLAGS_FORCE_PURGE_TRUE  1

#define LW2080_CTRL_GPU_RESET_ECC_ERROR_STATUS_PARAMS_MESSAGE_ID (0x36U)

typedef struct LW2080_CTRL_GPU_RESET_ECC_ERROR_STATUS_PARAMS {
    LwU32 statuses;
    LwU8  flags;
} LW2080_CTRL_GPU_RESET_ECC_ERROR_STATUS_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_FERMI_GPC_INFO
 *
 * This command returns a mask of enabled GPCs for the associated GPU.
 *
 *    gpcMask
 *      This parameter returns a mask of enabled GPCs. Each GPC has an ID
 *      that's equivalent to the corresponding bit position in the mask.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_GPU_GET_FERMI_GPC_INFO (0x20800137) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_FERMI_GPC_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_FERMI_GPC_INFO_PARAMS_MESSAGE_ID (0x37U)

typedef struct LW2080_CTRL_GPU_GET_FERMI_GPC_INFO_PARAMS {
    LwU32 gpcMask;
} LW2080_CTRL_GPU_GET_FERMI_GPC_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_FERMI_TPC_INFO
 *
 * This command returns a mask of enabled TPCs for a specified GPC.
 *
 *    gpcId
 *      This parameter specifies the GPC for which TPC information is
 *      to be retrieved. If the GPC with this ID is not enabled this command
 *      will return an tpcMask value of zero.
 *
 *    tpcMask
 *      This parameter returns a mask of enabled TPCs for the specified GPC.
 *      Each TPC has an ID that's equivalent to the corresponding bit
 *      position in the mask.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_GPU_GET_FERMI_TPC_INFO (0x20800138) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_FERMI_TPC_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_FERMI_TPC_INFO_PARAMS_MESSAGE_ID (0x38U)

typedef struct LW2080_CTRL_GPU_GET_FERMI_TPC_INFO_PARAMS {
    LwU32 gpcId;
    LwU32 tpcMask;
} LW2080_CTRL_GPU_GET_FERMI_TPC_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_FERMI_ZLWLL_INFO
 *
 * This command returns a mask of enabled ZLWLLs for a specified GPC.
 *
 *    gpcId
 *      This parameter specifies the GPC for which ZLWLL information is to be
 *      retrieved. If the GPC with this ID is not enabled this command will
 *      return an zlwllMask value of zero.
 *
 *    zlwllMask
 *      This parameter returns a mask of enabled ZLWLLs for the specified GPC.
 *      Each ZLWLL has an ID that's equivalent to the corresponding bit
 *      position in the mask.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *
 * Deprecated: Please use GR based control call
 * LW2080_CTRL_CMD_GR_GET_ZLWLL_MASK
 *
 */
#define LW2080_CTRL_CMD_GPU_GET_FERMI_ZLWLL_INFO (0x20800139) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_FERMI_ZLWLL_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_FERMI_ZLWLL_INFO_PARAMS_MESSAGE_ID (0x39U)

typedef struct LW2080_CTRL_GPU_GET_FERMI_ZLWLL_INFO_PARAMS {
    LwU32 gpcId;
    LwU32 zlwllMask;
} LW2080_CTRL_GPU_GET_FERMI_ZLWLL_INFO_PARAMS;

#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

/*
 * LW2080_CTRL_CMD_GPU_INJECT_ECC_EXCEPTIONS
 *
 * This command is used to inject fake ECC exceptions for a GPU given
 * its subdevice handle.  It is intended to facilitate testing of
 * ECC error status reporting throughout the driver stack.
 *
 *   units
 *     Array of structures used to inject single-bit and/or
 *     double-bit exceptions per unit.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_INJECT_ECC_EXCEPTIONS (0x2080013a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_INJECT_ECC_EXCEPTIONS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_INJECT_ECC_EXCEPTIONS_PARAMS_MESSAGE_ID (0x3AU)

typedef struct LW2080_CTRL_GPU_INJECT_ECC_EXCEPTIONS_PARAMS {
    struct {
        LW_DECLARE_ALIGNED(LwU64 sbe, 8);
        LW_DECLARE_ALIGNED(LwU64 dbe, 8);
    } units[LW2080_CTRL_GPU_ECC_UNIT_COUNT];
} LW2080_CTRL_GPU_INJECT_ECC_EXCEPTIONS_PARAMS;
/* LWRM_UNPUBLISHED_OPAQUE */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GPU_SET_DEEP_IDLE_STATISTICS_MODE
 *
 * This command switches the Deep Idle statistics mode with the option to reset
 * the data.
 *
 * deepIdleMode
 *   Specifies the Deep Idle mode to obtain statistics about: FO (framebuffer
 *   off), NH (no heads), VE (vblank extend) or SSC (sleep synchronize content).
 *
 * reset
 *   Specifies whether or not to reset the statistics data upon mode selection.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW2080_CTRL_CMD_GPU_SET_DEEP_IDLE_STATISTICS_MODE (0x2080013b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_SET_DEEP_IDLE_STATISTICS_MODE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_SET_DEEP_IDLE_STATISTICS_MODE_PARAMS_MESSAGE_ID (0x3BU)

typedef struct LW2080_CTRL_GPU_SET_DEEP_IDLE_STATISTICS_MODE_PARAMS {
    LwU32 deepIdleMode;
    LwU32 reset;
} LW2080_CTRL_GPU_SET_DEEP_IDLE_STATISTICS_MODE_PARAMS;

#define LW2080_CTRL_GPU_SET_DEEP_IDLE_STATISTICS_MODE_FO        (0x00000001)
#define LW2080_CTRL_GPU_SET_DEEP_IDLE_STATISTICS_MODE_NH        (0x00000002)
#define LW2080_CTRL_GPU_SET_DEEP_IDLE_STATISTICS_MODE_VE        (0x00000004) // TESLA-TODO: Remove
#define LW2080_CTRL_GPU_SET_DEEP_IDLE_STATISTICS_MODE_SSC       (0x00000008)

#define LW2080_CTRL_GPU_SET_DEEP_IDLE_STATISTICS_MODE_RESET_NO  (0x00000000)
#define LW2080_CTRL_GPU_SET_DEEP_IDLE_STATISTICS_MODE_RESET_YES (0x00000001)

/*
 * LW2080_CTRL_CMD_GPU_INITIATE_DEEP_IDLE_STATISTICS
 *
 * This is a dummy command now and returns LW_OK unconditionally.
 * We have kept it as many clients like LWAPI, MODS etc use this lwrrently.
 * It will be removed in near future
 *
 */
#define LW2080_CTRL_CMD_GPU_INITIATE_DEEP_IDLE_STATISTICS_READ  (0x2080013c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_INITIATE_DEEP_IDLE_STATISTICS_READ_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_INITIATE_DEEP_IDLE_STATISTICS_READ_PARAMS_MESSAGE_ID (0x3LW)

typedef struct LW2080_CTRL_GPU_INITIATE_DEEP_IDLE_STATISTICS_READ_PARAMS {
    LwU32 reserved;
} LW2080_CTRL_GPU_INITIATE_DEEP_IDLE_STATISTICS_READ_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_READ_DEEP_IDLE_STATISTICS
 *
 * This command reads the entry/exit statistics for the Deep Idle statistics
 * mode previously specified by LW2080_CTRL_CMD_SET_DEEP_IDLE_STATISTICS_MODE.
 *
 *   attempts
 *     Specifies the number of attempts made to enter Deep Idle; undefined in
 *     the case where Deep Idle is not supported.
 *   entries
 *     Specifies the number of successful entries into Deep Idle; undefined in
 *     the case where Deep Idle is not supported.
 *   exits
 *     Specifies the number of exits from Deep Idle; undefined in the case
 *     where Deep Idle is not supported.
 *   time
 *     Specifies the number of microseconds spent in Deep Idle; undefined in
 *     the case where Deep Idle is not supported.
 *   maxEntryLatency
 *     Specifies the maximum latency (in microseconds) for entering Deep Idle.
 *   maxExitLatency
 *     Specifies the maximum latency (in microseconds) for exiting Deep Idle.
 *   veFrames
 *     For Deep Idle VE only: specifies the number of frames spent in Deep
 *     Idle; undefined in the case where Deep Idle is not supported.
 *   veVblankExits
 *     For Deep Idle VE only: specifies the number of times Deep Idle VE exit
 *     due to the vblank time expiring.
 *   veDeepL1Exits
 *     For Deep Idle VE only: specifies the number of times Deep Idle VE exit
 *     due to Deep L1.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_GENERIC
 */
#define LW2080_CTRL_CMD_GPU_READ_DEEP_IDLE_STATISTICS (0x2080013d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_READ_DEEP_IDLE_STATISTICS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_READ_DEEP_IDLE_STATISTICS_PARAMS_MESSAGE_ID (0x3DU)

typedef struct LW2080_CTRL_GPU_READ_DEEP_IDLE_STATISTICS_PARAMS {
    LwU32 attempts;
    LwU32 entries;
    LwU32 exits;
    LwU32 time;
    LwU32 maxEntryLatency;
    LwU32 maxExitLatency;

    LwU32 veFrames;       // TESLA-TODO: Remove
    LwU32 veVblankExits;  // TESLA-TODO: Remove
    LwU32 veDeepL1Exits;  // TESLA-TODO: Remove
} LW2080_CTRL_GPU_READ_DEEP_IDLE_STATISTICS_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_SET_WORKLOAD_MODULATION
 *
 * This command is used to modulate the gpu workload. Among other
 * effects, this can be used to turn a steady, light workload into a bursty,
 * heavier workload.
 *
 * If frozenUs is non-zero and unfrozenUs is zero, the workload will be halted
 * once for the number of microseconds specified by frozenUs, and any further
 * workload modulation is canceled.
 *
 * If frozenUs and unfrozenUs are both zero, workload modulation is
 * canceled.
 *
 * frozenUs
 *   The period (in microseconds) that the workload will be frozen.
 * unfrozenUs
 *   The period (in microseconds) that the workload will be unfrozen.
 *
 * Possible return status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW2080_CTRL_CMD_GPU_SET_WORKLOAD_MODULATION (0x2080013e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x3E" */

typedef struct LW2080_CTRL_GPU_SET_WORKLOAD_MODULATION_PARAMS {
    LwU32 frozenUs;
    LwU32 unfrozenUs;
} LW2080_CTRL_GPU_SET_WORKLOAD_MODULATION_PARAMS;
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_CMD_GPU_GET_OEM_BOARD_INFO
 *
 * If an InfoROM with a valid OEM Board Object is present, this
 * command returns relevant information from the object to the
 * caller.
 *
 * The following data are lwrrently reported:
 *
 * buildDate
 *   The board's build date (8 digit BCD in format YYYYMMDD).
 *
 * marketingName
 *   The board's marketing name (24 ASCII letters e.g. "Lwdqro FX5800").
 *
 * boardSerialNumber
 *   The board's serial number.
 *
 * memoryManufacturer
 *   The board's memory manufacturer ('S'amsung/'H'ynix/'I'nfineon).
 *
 * memoryDateCode
 *   The board's memory datecode (LSB justified ASCII field with 0x00
 *   denoting empty space).
 *
 * productPartNumber
 *   The board's 900 product part number (LSB justified ASCII field with 0x00
 *   denoting empty space e.g. "900-21228-0208-200").
 *
 * boardRevision
 *   The board's revision (for e.g. A02, B01)
 *
 * boardType
 *   The board's type ('E'ngineering/'P'roduction)
 *
 * board699PartNumber
 *   The board's 699 product part number (LSB justified ASCII field with 0x00
 *   denoting empty space e.g. "699-21228-0208-200").
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_OEM_BOARD_INFO    (0x2080013f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_OEM_BOARD_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_GPU_MAX_MARKETING_NAME_LENGTH      (0x00000018)
#define LW2080_GPU_MAX_SERIAL_NUMBER_LENGTH       (0x00000010)
#define LW2080_GPU_MAX_MEMORY_PART_ID_LENGTH      (0x00000014)
#define LW2080_GPU_MAX_MEMORY_DATE_CODE_LENGTH    (0x00000006)
#define LW2080_GPU_MAX_PRODUCT_PART_NUMBER_LENGTH (0x00000014)

#define LW2080_CTRL_GPU_GET_OEM_BOARD_INFO_PARAMS_MESSAGE_ID (0x3FU)

typedef struct LW2080_CTRL_GPU_GET_OEM_BOARD_INFO_PARAMS {
    LwU32 buildDate;
    LwU8  marketingName[LW2080_GPU_MAX_MARKETING_NAME_LENGTH];
    LwU8  serialNumber[LW2080_GPU_MAX_SERIAL_NUMBER_LENGTH];
    LwU8  memoryManufacturer;
    LwU8  memoryPartID[LW2080_GPU_MAX_MEMORY_PART_ID_LENGTH];
    LwU8  memoryDateCode[LW2080_GPU_MAX_MEMORY_DATE_CODE_LENGTH];
    LwU8  productPartNumber[LW2080_GPU_MAX_PRODUCT_PART_NUMBER_LENGTH];
    LwU8  boardRevision[3];
    LwU8  boardType;
    LwU8  board699PartNumber[LW2080_GPU_MAX_PRODUCT_PART_NUMBER_LENGTH];
} LW2080_CTRL_GPU_GET_OEM_BOARD_INFO_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW2080_CTRL_CMD_GPU_GET_LICENSE_VAL
 *
 * Get the special license value written by the ucode after a license is
 * validated.
 *
 * licenseVal
 *   The special license value.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_GPU_GET_LICENSE_VAL (0x20800140) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_LICENSE_VAL_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_LICENSE_VAL_PARAMS_MESSAGE_ID (0x40U)

typedef struct LW2080_CTRL_GPU_GET_LICENSE_VAL_PARAMS {
    LwU32 licenseVal;
} LW2080_CTRL_GPU_GET_LICENSE_VAL_PARAMS;
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW2080_CTRL_CMD_GPU_GET_ID
 *
 * This command returns the gpuId of the associated object.
 *
 *   gpuId
 *     This field return the gpuId.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_ID (0x20800142) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ID_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_ID_PARAMS_MESSAGE_ID (0x42U)

typedef struct LW2080_CTRL_GPU_GET_ID_PARAMS {
    LwU32 gpuId;
} LW2080_CTRL_GPU_GET_ID_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_SET_GPU_DEBUG_MODE
 *
 * This command is used to enable or disable GPU debug mode. While this mode
 * is enabled,  some client RM calls that can potentially timeout return
 * LW_ERR_BUSY_RETRY, signalling the client to try again once GPU
 * debug mode is disabled.
 *
 * mode
 *   This parameter specifies whether GPU debug mode is to be enabled or
 *   disabled. Possible values are:
 *
 *     LW2080_CTRL_GPU_DEBUG_MODE_ENABLED
 *     LW2080_CTRL_GPU_DEBUG_MODE_DISABLED
 *
 * Possible return status values are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW2080_CTRL_CMD_GPU_SET_GPU_DEBUG_MODE (0x20800143) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_SET_GPU_DEBUG_MODE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_SET_GPU_DEBUG_MODE_PARAMS_MESSAGE_ID (0x43U)

typedef struct LW2080_CTRL_GPU_SET_GPU_DEBUG_MODE_PARAMS {
    LwU32 mode;
} LW2080_CTRL_GPU_SET_GPU_DEBUG_MODE_PARAMS;

#define LW2080_CTRL_GPU_DEBUG_MODE_ENABLED     (0x00000001)
#define LW2080_CTRL_GPU_DEBUG_MODE_DISABLED    (0x00000002)

/*
 * LW2080_CTRL_CMD_GPU_GET_GPU_DEBUG_MODE
 *
 * This command is used to query whether debug mode is enabled on the current
 * GPU. Please see the description of LW2080_CTRL_CMD_GPU_SET_GPU_DEBUG_MODE
 * for more details on GPU debug mode.
 *
 * lwrrentMode
 *   This parameter returns the state of GPU debug mode for the current GPU.
 *   Possible values are:
 *
 *     LW2080_CTRL_GPU_DEBUG_MODE_ENABLED
 *     LW2080_CTRL_GPU_DEBUG_MODE_DISABLED
 *
 * Possible return status values are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW2080_CTRL_CMD_GPU_GET_GPU_DEBUG_MODE (0x20800144) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_GPU_DEBUG_MODE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_GPU_DEBUG_MODE_PARAMS_MESSAGE_ID (0x44U)

typedef struct LW2080_CTRL_GPU_GET_GPU_DEBUG_MODE_PARAMS {
    LwU32 lwrrentMode;
} LW2080_CTRL_GPU_GET_GPU_DEBUG_MODE_PARAMS;
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


/*
 * LW2080_CTRL_CMD_GPU_ACQUIRE_COMPUTE_MODE_RESERVATION
 *
 * For multi-GPU elwironments that have long-running compute jobs (HPC, High Performance
 * Computing, for example), there is a need to make an intelligent choice as to which GPU
 * to run the next job on.  This method, along with the "release" method, provides an atomic
 * "query-and-lock" interface, which allows callers to implement a way to aclwrately select
 * the next available GPU for a compute job.
 *
 * This interacts intelligently with LWCA exclusive mode.  That means:
 *
 *   If a GPU is in "normal" mode, then the caller can always get the reservation.
 *
 *   If a GPU is in "lwca prohibited" mode, then the caller can not get the reservation at
 *   all, ever.
 *
 *   If a GPU is in "lwca exclusive" mode, then the caller can only get the reservation if no
 *   other client holds the reservation.
 *
 *   RM will use the client database to handle cleaning up any leaking reservations, if a
 *   process crashes before it can free its reservation.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_ACQUIRE_COMPUTE_MODE_RESERVATION (0x20800145) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x45" */

/*
 * LW2080_CTRL_CMD_GPU_RELEASE_COMPUTE_MODE_RESERVATION
 *
 * This command releases the compute mode reservation for the associated subdevice.
 *
 * This command usually succeeds, whether or not the reservation previously existed.
 * However, if for some reason the reservation is held by a client other than the one
 * making this call, then this method returns LW_ERR_STATE_IN_USE.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_RELEASE_COMPUTE_MODE_RESERVATION (0x20800146) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x46" */

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW2080_CTRL_CMD_GPU_GET_ENGINE_PARTNERLIST
 *
 * Returns a list of engines that can partner or coexist
 * when using the target channel or partnership class.
 * This list may include all engines (pre-Kepler), or as few
 * as 1 engine (Kepler and beyond).
 *
 *   engineType
 *     This field specifies the target engine type.
 *     See cl2080.h for a list of valid engines.
 *
 *   partnershipClassId
 *     This field specifies the target channel
 *     or partnership class ID.
 *     An example of such a class is GF100_CHANNEL_GPFIFO.
 *
 *   runqueue
 *     This field is an index which indicates the runqueue to
 *     return the list of supported engines for. This is the
 *     same field as what LWOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE
 *     specifies. This is only valid for TSG.
 *
 *   numPartners;
 *     This field returns the number of
 *     valid entries in the partnersList array
 *
 *   partnerList
 *     This field is an array containing the list of supported
 *     partner engines types, in no particular order, and
 *     may even be empty (numPartners = 0).
 *     See cl2080.h for a list of possible engines.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW2080_CTRL_CMD_GPU_GET_ENGINE_PARTNERLIST           (0x20800147) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ENGINE_PARTNERLIST_PARAMS_MESSAGE_ID" */

/* this macro specifies the maximum number of partner entries */
#define LW2080_CTRL_GPU_MAX_ENGINE_PARTNERS                  (0x00000020)

#define LW2080_CTRL_GPU_GET_ENGINE_PARTNERLIST_PARAMS_MESSAGE_ID (0x47U)

typedef struct LW2080_CTRL_GPU_GET_ENGINE_PARTNERLIST_PARAMS {
    LwU32 engineType;
    LwU32 partnershipClassId;
    LwU32 runqueue;
    LwU32 numPartners;
    // C form: LwU32 partnerList[LW2080_CTRL_GPU_MAX_ENGINE_PARTNERS];
    LwU32 partnerList[LW2080_CTRL_GPU_MAX_ENGINE_PARTNERS];
} LW2080_CTRL_GPU_GET_ENGINE_PARTNERLIST_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


/*
 * LW2080_CTRL_CMD_GPU_RESET_EXCEPTIONS
 *
 * Will reset all exception registers on the given engines that support
 * this capability.
 *
 *   targetEngine
 *     This field specifies the target engine type to reset exceptions on.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_GPU_RESET_EXCEPTIONS (0x20800148) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_RESET_EXCEPTIONS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_RESET_EXCEPTIONS_PARAMS_MESSAGE_ID (0x48U)

typedef struct LW2080_CTRL_GPU_RESET_EXCEPTIONS_PARAMS {
    LwU32 targetEngine;
} LW2080_CTRL_GPU_RESET_EXCEPTIONS_PARAMS;

#define LW2080_CTRL_GPU_RESET_EXCEPTIONS_ENGINE_GR (0x00000001)

/*
 * LW2080_CTRL_CMD_GPU_GET_LICENSE_STATUS
 *
 * Gets the current status of the license validation process.
 *
 * bComplete
 *   Set to LW_TRUE if the license validation process has finished.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_GPU_GET_LICENSE_STATUS     (0x20800149) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_LICENSE_STATUS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_LICENSE_STATUS_PARAMS_MESSAGE_ID (0x49U)

typedef struct LW2080_CTRL_GPU_GET_LICENSE_STATUS_PARAMS {
    LwBool bComplete;
} LW2080_CTRL_GPU_GET_LICENSE_STATUS_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW2080_CTRL_CMD_GPU_GET_GID_INFO
 *
 * This command returns the GPU ID (GID) string for the associated
 * GPU.  This value can be useful for GPU identification and security
 * system validation.
 *
 * The GPU ID is a SHA-1 based 16 byte ID, formatted as a 32 character
 *      hexadecimal string as "GPU-%08x-%04x-%04x-%04x-%012x" (the
 *      canonical format of a UUID)
 *
 * The GPU IDs are generated using the ECID, PMC_BOOT_0, and
 * PMC_BOOT_42 of the GPU as the hash message.
 *
 *   index
 *     (Input) "Select which GID set to get." Or so the original documentation
 *     said. In reality, there is only one GID per GPU, and the implementation
 *     completely ignores this parameter. You can too.
 *
 *   flags (Input) The _FORMAT* flags designate ascii or binary format. Binary
 *     format returns the raw bytes of either the 16-byte SHA-1 ID or the
 *     32-byte SHA-256 ID.
 *
 *     The _TYPE* flags needs to specify the _SHA1 type.
 *
 *   length
 *     (Output) Actual GID length, in bytes.
 *
 *   data[LW2080_BUS_MAX_GID_LENGTH]
 *     (Output) Result buffer: the GID itself, in a format that is determined by
 *     the "flags" field (described above).
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_GPU_GET_GID_INFO (0x2080014a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_GID_INFO_PARAMS_MESSAGE_ID" */

/* maximum possible number of bytes of GID information returned */
#define LW2080_GPU_MAX_GID_LENGTH        (0x000000100)

#define LW2080_CTRL_GPU_GET_GID_INFO_PARAMS_MESSAGE_ID (0x4AU)

typedef struct LW2080_CTRL_GPU_GET_GID_INFO_PARAMS {
    LwU32 index;
    LwU32 flags;
    LwU32 length;
    LwU8  data[LW2080_GPU_MAX_GID_LENGTH];
} LW2080_CTRL_GPU_GET_GID_INFO_PARAMS;

/* valid flags values */
#define LW2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT                  1:0
#define LW2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_ASCII  (0x00000000)
#define LW2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY (0x00000002)

#define LW2080_GPU_CMD_GPU_GET_GID_FLAGS_TYPE                    2:2
#define LW2080_GPU_CMD_GPU_GET_GID_FLAGS_TYPE_SHA1     (0x00000000)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GPU_GET_INFOROM_OBJECT_VERSION
 *
 * This command can be used by clients to retrieve the version of an
 * InfoROM object.
 *
 *   objectType
 *     This parameter specifies the name of the InfoROM object whose version
 *     should be queried. Possible values for this parameter include the
 *     LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_* definitions below. That is
 *     not an exhaustive list of all objects supported in the InfoROM, just
 *     the ones that clients are expected to care about.
 *
 *   version
 *     This parameter returns the version of the InfoROM object specified by
 *     the objectType parameter.
 *
 *   subversion
 *     This parameter returns the subversion of the InfoROM object specified
 *     by the objectType parameter.
 *
 * Possible return status values:
 *   LW_OK
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW2080_CTRL_CMD_GPU_GET_INFOROM_OBJECT_VERSION (0x2080014b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_INFOROM_OBJECT_VERSION_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_INFOROM_OBJ_TYPE_LEN           3

#define LW2080_CTRL_GPU_GET_INFOROM_OBJECT_VERSION_PARAMS_MESSAGE_ID (0x4BU)

typedef struct LW2080_CTRL_GPU_GET_INFOROM_OBJECT_VERSION_PARAMS {
    char objectType[LW2080_CTRL_GPU_INFOROM_OBJ_TYPE_LEN];
    LwU8 version;
    LwU8 subversion;
} LW2080_CTRL_GPU_GET_INFOROM_OBJECT_VERSION_PARAMS;

/* valid object names */
#define LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_BBX       "BBX"
#define LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_CFG       "CFG"
#define LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_ECC       "ECC"
#define LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_EEN       "EEN"
#define LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_IMG       "IMG"
#define LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_LWL       "LWL"
#define LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_OBD       "OBD"
#define LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_OEM       "OEM"
#define LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_PBL       "PBL"
#define LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_PWR       "PWR"
#define LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_RPR       "RPR"
#define LW2080_CTRL_GPU_INFOROM_OBJECT_NAME_RRL       "RRL"

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_CMD_SET_GPU_OPTIMUS_INFO
 *
 * This command will specify that system is Optimus enabled.
 *
 * isOptimusEnabled
 *     Set LW_TRUE if system is Optimus enabled.
 *
 * Possible status return values are:
 *      LW_OK
 */
#define LW2080_CTRL_CMD_SET_GPU_OPTIMUS_INFO (0x2080014c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_OPTIMUS_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_OPTIMUS_INFO_PARAMS_MESSAGE_ID (0x4LW)

typedef struct LW2080_CTRL_GPU_OPTIMUS_INFO_PARAMS {
    LwBool isOptimusEnabled;
} LW2080_CTRL_GPU_OPTIMUS_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_IP_VERSION
 *
 * Will return the IP VERSION on the given engine for engines that support
 * this capability.
 *
 *   targetEngine
 *     This parameter specifies the target engine type to query for IP_VERSION.
 *
 *   ipVersion
 *     This parameter returns the IP VERSION read from the unit's IP_VER
 *     register.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_GPU_GET_IP_VERSION (0x2080014d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_IP_VERSION_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_IP_VERSION_PARAMS_MESSAGE_ID (0x4DU)

typedef struct LW2080_CTRL_GPU_GET_IP_VERSION_PARAMS {
    LwU32 targetEngine;
    LwU32 ipVersion;
} LW2080_CTRL_GPU_GET_IP_VERSION_PARAMS;

#define LW2080_CTRL_GPU_GET_IP_VERSION_DISPLAY     (0x00000001)
#define LW2080_CTRL_GPU_GET_IP_VERSION_HDACODEC    (0x00000002)
#define LW2080_CTRL_GPU_GET_IP_VERSION_PMGR        (0x00000003)
#define LW2080_CTRL_GPU_GET_IP_VERSION_PPWR_PMU    (0x00000004)
#define LW2080_CTRL_GPU_GET_IP_VERSION_DISP_FALCON (0x00000005)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GPU_QUERY_OPERATION_MODE
 *
 * This command returns the current GPU Operation Mode for
 * a GPU given its subdevice handle.  The value returned is
 * the current GPU Operation mode for the GPU stored in non-volatile
 * memory on the board.
 *
 * bIsOperationModeConfigurable
*       LW_TRUE:  User is able to configure GPU Operation Mode.
 *      LW_FALSE: User Cannot configure GPU Operation Mode.
 * lwrrentOperationMode
 *      Current GPU Operation mode.
 * pendingOperationMode
 *      GPU Operation mode which will take effect after Booting.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_GPU_QUERY_OPERATION_MODE   (0x20800152) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_CMD_GPU_QUERY_OPERATION_MODE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_GPU_QUERY_OPERATION_MODE_PARAMS_MESSAGE_ID (0x52U)

typedef struct LW2080_CTRL_CMD_GPU_QUERY_OPERATION_MODE_PARAMS {
    LwBool bIsOperationModeConfigurable;
    LwU32  lwrrentOperationMode;
    LwU32  pendingOperationMode;
} LW2080_CTRL_CMD_GPU_QUERY_OPERATION_MODE_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_ID_ILLUM_SUPPORT
 *
 * This command returns an indicator which reports if the specified Illumination control
 * attribute is supported
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_GPU_ILLUM_ATTRIB_LOGO_BRIGHTNESS 0
#define LW2080_CTRL_GPU_ILLUM_ATTRIB_SLI_BRIGHTNESS  1
#define LW2080_CTRL_CMD_GPU_QUERY_ILLUM_SUPPORT      (0x20800153) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_CMD_GPU_QUERY_ILLUM_SUPPORT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_GPU_QUERY_ILLUM_SUPPORT_PARAMS_MESSAGE_ID (0x53U)

typedef struct LW2080_CTRL_CMD_GPU_QUERY_ILLUM_SUPPORT_PARAMS {
    LwU32  attribute;
    LwBool bSupported;
} LW2080_CTRL_CMD_GPU_QUERY_ILLUM_SUPPORT_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_ID_ILLUM
 *
 * This command returns the current value of the specified Illumination control attribute.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_ILLUM (0x20800154) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x54" */

typedef struct LW2080_CTRL_CMD_GPU_ILLUM_PARAMS {
    LwU32 attribute;
    LwU32 value;
} LW2080_CTRL_CMD_GPU_ILLUM_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_SET_ID_ILLUM
 *
 * This command sets a new valuefor the specified Illumination control attribute.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_SET_ILLUM                 (0x20800155) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x55" */

/*
 * LW2080_CTRL_CMD_GPU_GET_INFOROM_IMAGE_VERSION
 *
 * This command can be used by clients to retrieve the version of the entire
 * InfoROM image.
 *
 *   version
 *      This parameter returns the version of the InfoROM image as a NULL-
 *      terminated character string of the form "XXXX.XXXX.XX.XX" where each
 *      'X' is an integer character.
 *
 * Possible status return values are:
 *   LWOS_STATUS_SUCCES
 *   LW_ERR_INSUFFICIENT_RESOURCES
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_DATA
 */
#define LW2080_CTRL_CMD_GPU_GET_INFOROM_IMAGE_VERSION (0x20800156) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_INFOROM_IMAGE_VERSION_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_INFOROM_IMAGE_VERSION_LEN     16

#define LW2080_CTRL_GPU_GET_INFOROM_IMAGE_VERSION_PARAMS_MESSAGE_ID (0x56U)

typedef struct LW2080_CTRL_GPU_GET_INFOROM_IMAGE_VERSION_PARAMS {
    LwU8 version[LW2080_CTRL_GPU_INFOROM_IMAGE_VERSION_LEN];
} LW2080_CTRL_GPU_GET_INFOROM_IMAGE_VERSION_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_QUERY_INFOROM_ECC_SUPPORT
 *
 * This command returns whether or not ECC is supported via the InfoROM.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_QUERY_INFOROM_ECC_SUPPORT (0x20800157) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x57" */

/*
 * LW2080_CTRL_GPU_SW_FEATURE_INFO
 *
 * This structure describes a single feature value. Clients
 * request a particular feature value by specifying a unique feature
 * information index.
 *
 * Legal software feature information index values are:
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_WARPBLEND
 *     This index is used to get the feature information value for
 *     Warp & Blend.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_10BPC_OPENGL
 *     This index is used to get the feature information value for
 *     10 bit per color component OpenGL.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_MOSAIC
 *     This index is used to get the feature information value for
 *     Mosaic (basic or premium).
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_PREMIUM_MOSAIC
 *     This index is used to get the feature information value for
 *     Premium Mosaic (aka mosaic sync).
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_QUADRO_SYNC
 *     This index is used to get the feature information value for
 *     framelocking via a Gsync board connected to the GPU via an
 *     extension cable (P359/P2060 so far).
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_WKS_STEREO
 *     This index is used to get the feature information value for
 *     Workstation Quadbuffered Stereo (OpenGL and DX).
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_DIRECT_RDMA_WDDM
 *     This index is used to get the feature information value for
 *     DIRECT_RDMA WDDM mode.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_DIRECT_RDMA_TCC
 *     This index is used to get the feature information value for
 *     DIRECT_RDMA TCC (WDM) mode.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_GPU_DIRECT_FOR_VIDEO
 *     This index is used to get the feature information value for
 *     GPU Direct for Video.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_CLONE_TO_FIT
 *     This index is used to get the feature information value for
 *     clone to fit (scaling/clipping of secondary clone heads).
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_PAN_SCAN
 *     This index is used to get the feature information value for
 *     Pan Scan support in conjunction with clone to fit.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_LW_IFR_FBC
 *     This index is used to get the feature information value for
 *     IndirectFrameBufferRead / FrameBufferCapture/Compress.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_POSITIVE_OVERLAP
 *     This index is used to get the feature information value for
 *     Mosaic with positive overlap between displays.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_JVC_ESHIFT_4K
 *     This index is used to get the feature information value for
 *     pixel packing support for JVC e-shift 4k projectors.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_JVC_ESHIFT_8K
 *     This index is used to get the feature information value for
 *     pixel packing support for JVC e-shift 8k projectors.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_CLAW
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_CLAW_MID
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_CLAW_HIGH
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_CLAW_MT
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_SDDR
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_SDDR_MID
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_SDDR_HIGH
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_SDSR
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_SDSR_MID
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_SDSR_HIGH
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_WSVA
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_WSVA_MID
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_WSVA_HIGH
 *     These indices are used to get the feature information values for
 *     workstation performance features CLAW/SDDR/SDSR/WSVA
 *     where (XXXX=CLAW/SDDR/SDSR/WSVA):
 *     - LW2080_CTRL_GPU_SW_FEATURE_INDEX_XXXX is the master enable for feature XXXX
 *       If this feature is disabled, then the other XXXX_* are ignored.
 *     - LW2080_CTRL_GPU_SW_FEATURE_INDEX_XXXX_MID enables MID range features of XXXX
 *       If this feature is disabled, then XXXX_HIGH is ignored.
 *     - LW2080_CTRL_GPU_SW_FEATURE_INDEX_XXXX_HIGH enables HIGH end features of XXXX
 *     - possible combinations:
 *       |XXXX|XXXX_MID|XXXX_HIGH| meaning
 *       +----+--------+---------+-----------------------------------------------------------
 *       |  0 |    x   |    x    | disable XXXX
 *       |  1 |    0   |    x    | enable XXXX with basic features
 *       |  1 |    1   |    0    | enable XXXX with basic and MID range features
 *       |  1 |    1   |    1    | enable XXXX with basic and MID range and HIGH end features
 *       +----+--------+---------+-----------------------------------------------------------
 *     - LW2080_CTRL_GPU_SW_FEATURE_INDEX_CLAW_MT enables CLAW Multi-Threading
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_M_AND_E
 *     This index is used to get the feature information value for
 *     Media and Entertainmant related subfeatures.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_4_QUADRANT_PIXELSHIFT
 *     This index is used to get the feature information value for
 *     4 Quadrant pixelsift (added via RID 71672) related subfeatures.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_QUADRO_VR_READY
 *     This index is used to get the feature information value for
 *     'LWDQRO VR READY' (added via RID 74371) related subfeatures.
 *   LW2080_CTRL_GPU_SW_FEATURE_INDEX_VIDEO_DECRYPT_BLIT
 *     This index is used to get the feature information value for
 *     'HW Acceleration of Decryption Blit' (added via RID 74886) related subfeatures.
 *
 * Possible status return values in the data field of the
 * LW2080_CTRL_GPU_SW_FEATURE_INFO structure are:
 *
 *   LW2080_CTRL_GPU_SW_FEATURE_BIT_NOT_SUPPORTED
 *     The feature is not supported by the given gpu data.
 *   LW2080_CTRL_GPU_SW_FEATURE_BIT_SUPPORTED
 *     The feature is supported by the given gpu data.
 *   LW2080_CTRL_GPU_SW_FEATURE_BIT_NOT_HANDLED
 *     The feature is not yet handled by the feature matrix (matrix will
 *     need an update).
 */

typedef struct LW2080_CTRL_GPU_SW_FEATURE_INFO {
    LwU32 index;
    LwU32 data;
} LW2080_CTRL_GPU_SW_FEATURE_INFO;

/* The following bits are lwrrently returned in the data field */
#define LW2080_CTRL_GPU_SW_FEATURE_BIT_NOT_SUPPORTED           (0x00000000)
#define LW2080_CTRL_GPU_SW_FEATURE_BIT_SUPPORTED               (0x00000001)
#define LW2080_CTRL_GPU_SW_FEATURE_BIT_NOT_HANDLED             (0x00000002)

/* The following defines are supported in the index field */
// #define LW2080_CTRL_GPU_SW_FEATURE_INDEX_REUSABLE 0                (0x00000000)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_WARPBLEND             (0x00000001)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_10BPC_OPENGL          (0x00000002)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_MOSAIC                (0x00000003)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_PREMIUM_MOSAIC        (0x00000004)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_QUADRO_SYNC           (0x00000005)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_WKS_STEREO            (0x00000006)
// #define LW2080_CTRL_GPU_SW_FEATURE_INDEX_REUSABLE 1                (0x00000007)
// #define LW2080_CTRL_GPU_SW_FEATURE_INDEX_REUSABLE 2                (0x00000008)
// #define LW2080_CTRL_GPU_SW_FEATURE_INDEX_REUSABLE 3                (0x00000009)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_DIRECT_RDMA_WDDM      (0x0000000A)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_DIRECT_RDMA_TCC       (0x0000000B)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_GPU_DIRECT_FOR_VIDEO  (0x0000000C)
// #define LW2080_CTRL_GPU_SW_FEATURE_INDEX_REUSABLE 5                (0x0000000D)
// #define LW2080_CTRL_GPU_SW_FEATURE_INDEX_REUSABLE 6                (0x0000000E)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_CLONE_TO_FIT          (0x0000000F)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_PAN_SCAN              (0x00000010)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_LW_IFR_FBC            (0x00000011)
// #define LW2080_CTRL_GPU_SW_FEATURE_INDEX_REUSABLE 7                (0x00000012)
// #define LW2080_CTRL_GPU_SW_FEATURE_INDEX_REUSABLE 8                (0x00000013)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_POSITIVE_OVERLAP      (0x00000014)
// #define LW2080_CTRL_GPU_SW_FEATURE_INDEX_REUSABLE 9                (0x00000015)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_JVC_ESHIFT_4K         (0x00000016)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_JVC_ESHIFT_8K         (0x00000017)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_CLAW                  (0x00000018)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_CLAW_MID              (0x00000019)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_CLAW_HIGH             (0x0000001A)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_CLAW_MT               (0x0000001B)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_SDDR                  (0x0000001C)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_SDDR_MID              (0x0000001D)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_SDDR_HIGH             (0x0000001E)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_SDSR                  (0x0000001F)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_SDSR_MID              (0x00000020)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_SDSR_HIGH             (0x00000021)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_WSVA                  (0x00000022)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_WSVA_MID              (0x00000023)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_WSVA_HIGH             (0x00000024)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_M_AND_E               (0x00000025)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_4_QUADRANT_PIXELSHIFT (0x00000026)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_QUADRO_VR_READY       (0x00000027)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_VIDEO_DECRYPT_BLIT    (0x00000028)
#define LW2080_CTRL_GPU_SW_FEATURE_INDEX_TABLE_SIZE            (0x00000029)

/*
* LW2080_CTRL_CMD_GPU_GET_SW_FEATURES
*
*  This command returns software feature information for the associated GPU.
*  Requests to retrieve feature information use a list of one or more
 * LW2080_CTRL_GPU_SW_FEATURE_INFO structures.
 *
 *   featureInfoListSize
 *     This field specifies the number of entries on the caller's
 *     featureInfoList.
 *   featureInfoList
 *     This field specifies an array of size
 *     LW2080_CTRL_GPU_SW_FEATURE_INDEX_TABLE_SIZE.
 *     In this array the feature information is returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_SW_FEATURES                    (0x20800159) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_SW_FEATURE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_SW_FEATURE_PARAMS_MESSAGE_ID (0x59U)

typedef struct LW2080_CTRL_GPU_GET_SW_FEATURE_PARAMS {
    LwU32                           featureInfoListCount;
    LW2080_CTRL_GPU_SW_FEATURE_INFO featureInfoList[LW2080_CTRL_GPU_SW_FEATURE_INDEX_TABLE_SIZE];
} LW2080_CTRL_GPU_GET_SW_FEATURE_PARAMS;

/*
 * LW2080_CTRL_GPU_PHYSICAL_BRIDGE_VERSION
 *
 * This structure contains information about a single physical bridge.
 *
 *   fwVersion
 *     This field specifies Firmware Version of the bridge stored in
 *     bridge EEPROM.
 *   oemVersion
 *     This field specifies Oem Version of the firmware stored in
 *     bridge EEPROM.
 *   siliconRevision
 *     This field contains the silicon revision of the bridge hardware.
 *     It is set by the chip manufacturer.
 *   hwbcResourceType
 *     This field specifies the hardware broadcast resource type.
 *     Value denotes the kind of bridge - PLX or BR04
 *
 */

typedef struct LW2080_CTRL_GPU_PHYSICAL_BRIDGE_VERSION_PARAMS {
    LwU32 fwVersion;
    LwU8  oemVersion;
    LwU8  siliconRevision;
    LwU8  hwbcResourceType;
} LW2080_CTRL_GPU_PHYSICAL_BRIDGE_VERSION_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_PHYSICAL_BRIDGE_VERSION_INFO
 *
 * This command returns physical bridge information in the system.
 * Information consists of bridgeCount and a list of bridgeId's.
 * The bridge Id's are used by LW2080_CTRL_CMD_GPU_GET_PHYSICAL_BRIDGE_VERSION
 * to get firmware version, oem version and silicon revision info.
 *
 *   bridgeCount
 *     This field specifies the number of physical brides present
 *     in the system.
 *   hPhysicalBridges
 *     This field specifies an array of size LW2080_CTRL_MAX_PHYSICAL_BRIDGE.
 *     In this array, the bridge Id's are stored.
 *   bridgeList
 *     This field specifies an array of size LW2080_CTRL_MAX_PHYSICAL_BRIDGE.
 *     In this array, the bridge version details are stored.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_PHYSICAL_BRIDGE_VERSION_INFO (0x2080015a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_PHYSICAL_BRIDGE_VERSION_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_MAX_PHYSICAL_BRIDGE                      (100)
#define LW2080_CTRL_GPU_GET_PHYSICAL_BRIDGE_VERSION_INFO_PARAMS_MESSAGE_ID (0x5AU)

typedef struct LW2080_CTRL_GPU_GET_PHYSICAL_BRIDGE_VERSION_INFO_PARAMS {
    LwU8                                           bridgeCount;
    LwHandle                                       hPhysicalBridges[LW2080_CTRL_MAX_PHYSICAL_BRIDGE];
    LW2080_CTRL_GPU_PHYSICAL_BRIDGE_VERSION_PARAMS bridgeList[LW2080_CTRL_MAX_PHYSICAL_BRIDGE];
} LW2080_CTRL_GPU_GET_PHYSICAL_BRIDGE_VERSION_INFO_PARAMS;

/*
 * LW2080_CTRL_GPU_BRIDGE_VERSION
 *
 * This structure contains information about a single physical bridge.
 *
 *   bus
 *     This field specifies the bus id of the bridge.
 *   device
 *     This field specifies the device id of the bridge.
 *   func
 *     This field specifies the function id of the bridge.
 *   oemVersion
 *     This field specifies Oem Version of the firmware stored in
 *     bridge EEPROM.
 *   siliconRevision
 *     This field contains the silicon revision of the bridge hardware.
 *     It is set by the chip manufacturer.
 *   hwbcResourceType
 *     This field specifies the hardware broadcast resource type.
 *     Value denotes the kind of bridge - PLX or BR04
 *   domain
 *     This field specifies the respective domain of the PCI device.
 *   fwVersion
 *     This field specifies Firmware Version of the bridge stored in
 *     bridge EEPROM.
 *
 *   If (fwVersion, oemVersion, siliconRevision) == 0, it would mean that RM
 *   was unable to fetch the value from the bridge device.
 *
 */

typedef struct LW2080_CTRL_GPU_BRIDGE_VERSION_PARAMS {
    LwU8  bus;
    LwU8  device;
    LwU8  func;
    LwU8  oemVersion;
    LwU8  siliconRevision;
    LwU8  hwbcResourceType;
    LwU32 domain;
    LwU32 fwVersion;
} LW2080_CTRL_GPU_BRIDGE_VERSION_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_ALL_BRIDGES_UPSTREAM_OF_GPU
 *
 * This command returns information about all the upstream bridges of the GPU.
 * Information consists of bridge firmware version and its bus topology.
 *
 *   bridgeCount
 *     This field specifies the number of physical brides present
 *     in the system.
 *   physicalBridgeIds
 *     This field specifies an array of size LW2080_CTRL_MAX_PHYSICAL_BRIDGE.
 *     In this array, the bridge Ids are stored.
 *   bridgeList
 *     This field specifies an array of size LW2080_CTRL_MAX_PHYSICAL_BRIDGE.
 *     In this array, the bridge version details are stored.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_ALL_BRIDGES_UPSTREAM_OF_GPU (0x2080015b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ALL_BRIDGES_UPSTREAM_OF_GPU_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_ALL_BRIDGES_UPSTREAM_OF_GPU_PARAMS_MESSAGE_ID (0x5BU)

typedef struct LW2080_CTRL_GPU_GET_ALL_BRIDGES_UPSTREAM_OF_GPU_PARAMS {
    LwU8                                  bridgeCount;
    LwU32                                 physicalBridgeIds[LW2080_CTRL_MAX_PHYSICAL_BRIDGE];
    LW2080_CTRL_GPU_BRIDGE_VERSION_PARAMS bridgeList[LW2080_CTRL_MAX_PHYSICAL_BRIDGE];
} LW2080_CTRL_GPU_GET_ALL_BRIDGES_UPSTREAM_OF_GPU_PARAMS;

typedef enum LW_SESSION_LICENSE_TYPE {
    LW_SESSION_LICENSE_TYPE_LWENC = 0,
    LW_SESSION_LICENSE_TYPE_LWIFR = 1,
    LW_SESSION_LICENSE_TYPE_LWFBC = 2,
} LW_SESSION_LICENSE_TYPE;
#define LW_SESSION_LICENSE_TYPE_COUNT               (3)

/*
 * LW2080_CTRL_CMD_GPU_ACQUIRE_SESSION_LICENSE
 *
 * In case of a non-qualified GPU, this command determines whether GRID/LWENC
 * session can be created or not. Limited sessions are allowed on a system
 * containing non-qualified GPU. This command increments the counter used to
 * track number of respective sessions in the system and returns success if a
 * new session can be created, RM_ERR_MAX_SESSION_LIMIT_REACHED otherwise.
 *
 * In case of a qualified,Internal GUID this command should not be used
 *
 *   sessionType
 *     This field specifies UMD wants to create session of which type.
 *     It could be one of LwIFR, LwFBC or LWENC.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_OBJECT_PARENT
 */
#define LW2080_CTRL_CMD_GPU_ACQUIRE_SESSION_LICENSE (0x2080015c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_ACQUIRE_SESSION_LICENSE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_ACQUIRE_SESSION_LICENSE_PARAMS_MESSAGE_ID (0x5LW)

typedef struct LW2080_CTRL_GPU_ACQUIRE_SESSION_LICENSE_PARAMS {
    LW_SESSION_LICENSE_TYPE sessionType;
} LW2080_CTRL_GPU_ACQUIRE_SESSION_LICENSE_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_RELEASE_SESSION_LICENSE
 *
 * In case of a non-qualified this command decrements the number of respective
 * sessions running in the system.
 *
 *   sessionType
 *     This field specifies UMD wants to create session of which type.
 *     It could be one of LwIFR, LwFBC or LWENC.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_OBJECT_PARENT
 */
#define LW2080_CTRL_CMD_GPU_RELEASE_SESSION_LICENSE (0x2080015d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_RELEASE_SESSION_LICENSE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_RELEASE_SESSION_LICENSE_PARAMS_MESSAGE_ID (0x5DU)

typedef struct LW2080_CTRL_GPU_RELEASE_SESSION_LICENSE_PARAMS {
    LW_SESSION_LICENSE_TYPE sessionType;
} LW2080_CTRL_GPU_RELEASE_SESSION_LICENSE_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_IS_GRID_SDK_QUALIFIED_GPU
 *
 * Identifies whether the GPU is GRID_SDK qualified or not.
 *
 *   isGridSdkQualifiedGpu
 *     This field is written with the value TRUE if the GPU is a qualified
 *     GRID gpu, FALSE otherwise.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_GPU_IS_GRID_SDK_QUALIFIED_GPU (0x2080015e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_IS_GRID_SDK_QUALIFIED_GPU_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_IS_GRID_SDK_QUALIFIED_GPU_PARAMS_MESSAGE_ID (0x5EU)

typedef struct LW2080_CTRL_GPU_IS_GRID_SDK_QUALIFIED_GPU_PARAMS {
    LwBool isGridSdkQualifiedGpu;
} LW2080_CTRL_GPU_IS_GRID_SDK_QUALIFIED_GPU_PARAMS;


/*
 * LW2080_CTRL_CMD_GPU_QUERY_SCRUBBER_STATUS
 *
 * This command is used to query the status of the HW scrubber. If a scrub is
 * in progress then the range which is being scrubbed is also reported back.
 *
 *   scrubberStatus
 *     Reports the status of the scrubber unit - running/idle.
 *
 *   remainingtimeMs
 *     If scrubbing is going on, reports the remaining time in milliseconds
 *     required to finish the scrub.
 *
 *   scrubStartAddr
 *     This parameter reports the start address of the ongoing scrub if scrub
 *     is going on, otherwise reports the start addr of the last finished scrub
 *
 *   scrubEndAddr
 *     This parameter reports the end address of the ongoing scrub if scrub
 *     is going on, otherwise reports the end addr of the last finished scrub.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_GPU_QUERY_SCRUBBER_STATUS (0x2080015f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_QUERY_SCRUBBER_STATUS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_QUERY_SCRUBBER_STATUS_PARAMS_MESSAGE_ID (0x5FU)

typedef struct LW2080_CTRL_GPU_QUERY_SCRUBBER_STATUS_PARAMS {
    LwU32 scrubberStatus;
    LwU32 remainingTimeMs;
    LW_DECLARE_ALIGNED(LwU64 scrubStartAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 scrubEndAddr, 8);
} LW2080_CTRL_GPU_QUERY_SCRUBBER_STATUS_PARAMS;

/* valid values for scrubber status */
#define LW2080_CTRL_GPU_QUERY_SCRUBBER_STATUS_SCRUBBER_RUNNING (0x00000000)
#define LW2080_CTRL_GPU_QUERY_SCRUBBER_STATUS_SCRUBBER_IDLE    (0x00000001)

/*
 * LW2080_CTRL_CMD_GPU_GET_VPR_CAPS
 *
 * This command is used to query the VPR capability information for a
 * GPU. If VPR is supported, the parameters are filled accordingly.
 * The addresses returned are all physical addresses.
 *
 *   minStartAddr
 *     Returns the minimum start address that can be possible for VPR.
 *
 *   maxEndAddr
 *     Returns the maximum end   address that can be possible for VPR.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_GPU_GET_VPR_CAPS                       (0x20800160) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_VPR_CAPS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_VPR_CAPS_PARAMS_MESSAGE_ID (0x60U)

typedef struct LW2080_CTRL_GPU_GET_VPR_CAPS_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 minStartAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 maxEndAddr, 8);
} LW2080_CTRL_GPU_GET_VPR_CAPS_PARAMS;

/* Licensable features on GRID SW stack */
typedef enum LW_GRID_LICENSE_FEATURE_CODE {
    LW_GRID_LICENSE_FEATURE_CODE_VGPU = 1,
    LW_GRID_LICENSE_FEATURE_CODE_QUADRO = 2,
    LW_GRID_LICENSE_FEATURE_CODE_GAMING = 3,
    LW_GRID_LICENSE_FEATURE_CODE_COMPUTE = 4,
} LW_GRID_LICENSE_FEATURE_CODE;

#define LW_GRID_LICENSE_FEATURE_MAX_COUNT (3)
#define LW_GRID_LICENSE_INFO_MAX_LENGTH   (128)
/* Signature length for GRID License */
#define LW_GRID_LICENSE_SIGNATURE_SIZE    (128)

/* License info string for GPU Passthrough case. */
#define LW_GRID_LICENSE_FEATURE_VIRTUAL_WORKSTATION_EDITION "Lwdqro-Virtual-DWS,5.0;GRID-Virtual-WS,2.0;GRID-Virtual-WS-Ext,2.0"

/* license info string for vGaming. */
#define LW_GRID_LICENSE_FEATURE_GAMING_EDITION              "GRID-vGaming,8.0"

/* License info string for vCompute. */
#define LW_GRID_LICENSE_FEATURE_COMPUTE_EDITION "LWPU-vComputeServer,9.0;Lwdqro-Virtual-DWS,5.0"

#define LW_GRID_LICENSED_PRODUCT_VWS     "LWPU RTX Virtual Workstation"
#define LW_GRID_LICENSED_PRODUCT_GAMING  "LWPU Cloud Gaming"
#define LW_GRID_LICENSED_PRODUCT_VPC     "LWPU Virtual PC"
#define LW_GRID_LICENSED_PRODUCT_VAPPS   "LWPU Virtual Applications"
#define LW_GRID_LICENSED_PRODUCT_COMPUTE "LWPU Virtual Compute Server"

/* Status codes for GRID license expiry */
#define LW2080_CTRL_GPU_GRID_LICENSE_EXPIRY_NOT_AVAILABLE   0    // Expiry information not available
#define LW2080_CTRL_GPU_GRID_LICENSE_EXPIRY_ILWALID         1    // Invalid expiry or error fetching expiry
#define LW2080_CTRL_GPU_GRID_LICENSE_EXPIRY_VALID           2    // Valid expiry
#define LW2080_CTRL_GPU_GRID_LICENSE_EXPIRY_NOT_APPLICABLE  3    // Expiry not applicable
#define LW2080_CTRL_GPU_GRID_LICENSE_EXPIRY_PERMANENT       4    // Permanent expiry

typedef struct LW2080_CTRL_GPU_LICENSABLE_FEATURES {
    /* Feature code to identify the licensed feature */
    LW_GRID_LICENSE_FEATURE_CODE featureCode;

    /* Current state of feature : true=enabled/false=disabled */
    LwBool                       isFeatureEnabled;

    /* Current state of the enabled feature: true=licensed/false=unlicensed */
    LwBool                       isLicensed;

    /* The feature/license info we need to provide when requesting licenses
       from Flexera. This is a semicolon(;) separated list of license options
       (commas to delimit the parameters with a license option), in increasing
       order of preference. For e.g.:
       GRID-Virtual-PC,1.0;GRID-Virtual-WS,1.0;GRID-Virtual-WS-Ext,1.0 */
    char                         licenseInfo[LW_GRID_LICENSE_INFO_MAX_LENGTH];

    /* Licensed product name, one of LW_GRID_LICENSED_PRODUCT_xxx strings. */
    char                         productName[LW_GRID_LICENSE_INFO_MAX_LENGTH];

    /* License expiry in seconds since epoch time */
    LwU32                        licenseExpiryTimestamp;

    /* License expiry status */
    LwU8                         licenseExpiryStatus;
} LW2080_CTRL_GPU_LICENSABLE_FEATURES;

/*
 * LW2080_CTRL_CMD_GPU_GET_LICENSABLE_FEATURES
 *
 * Identifies whether the system supports vGPU Software Licensing. If yes,
 * returns the list of feature(s) that can be licensed on the system.
 *
 *   isLicenseSupported
 *     This field returns TRUE if vGPU Software Licensing is supported on
 *     the system, FALSE otherwise.
 *
 *   signature
 *     Dynamic signature passed out from RM that is used in the computation of
 *     response signature passed back in a subsequent license enable request.
 *
 *   licensableFeaturesCount
 *     This field specifies the number of entries returned in the
 *     licensableFeatures array.
 *
 *   licensableFeatures
 *     This field specifies an array of size LW_GRID_LICENSE_FEATURE_MAX_COUNT.
 *     If GRID Software Licensing is supported, in this array the list of
 *     licensable feature code and corresponding licenseInfo are returned.
 *
 * Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 */
#define LW2080_CTRL_CMD_GPU_GET_LICENSABLE_FEATURES (0x20800161) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_LICENSABLE_FEATURES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_LICENSABLE_FEATURES_PARAMS_MESSAGE_ID (0x61U)

typedef struct LW2080_CTRL_GPU_GET_LICENSABLE_FEATURES_PARAMS {
    LwBool                              isLicenseSupported;
    LwU8                                signature[LW_GRID_LICENSE_SIGNATURE_SIZE];
    LwU32                               licensableFeaturesCount;
    LW2080_CTRL_GPU_LICENSABLE_FEATURES licensableFeatures[LW_GRID_LICENSE_FEATURE_MAX_COUNT];
} LW2080_CTRL_GPU_GET_LICENSABLE_FEATURES_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_DISABLE_LICENSED_FEATURE
 *
 * Communicates to RM that a license has been disabled and switches RM to
 * disable the licensed feature provided as input parameter.
 *
 *   featureCode
 *     Licensed feature to be disabled.
 *
 * Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_DISABLE_LICENSED_FEATURE (0x20800164) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_DISABLE_LICENSED_FEATURE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_DISABLE_LICENSED_FEATURE_PARAMS_MESSAGE_ID (0x64U)

typedef struct LW2080_CTRL_GPU_DISABLE_LICENSED_FEATURE_PARAMS {
    LW_GRID_LICENSE_FEATURE_CODE featureCode;
} LW2080_CTRL_GPU_DISABLE_LICENSED_FEATURE_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_DISABLE_ALL_LICENSED_FEATURES
 *
 * Communicates to RM to disable all features licensable under
 * GRID software license.
 * This control call does a blanket disable of all licensable features
 * and switches to an unlicensed mode.
 *
 * Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 */
#define LW2080_CTRL_CMD_GPU_DISABLE_ALL_LICENSED_FEATURES      (0x20800165) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x65" */
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_CMD_GPU_HANDLE_GPU_SR
 *
 * Communicates to RM to handle GPU Surprise Removal
 * Called from client when it receives SR IRP from OS
 * Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 */
#define LW2080_CTRL_CMD_GPU_HANDLE_GPU_SR                      (0x20800167) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x67" */

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GPU_GET_PES_INFO
 *
 * This command provides the PES count and mask of enabled PES for a
 * specified GPC. It also returns the TPC to PES mapping information
 * for a given GPU.
 *
 *   gpcId[IN]
 *     This parameter specifies the GPC for which PES information is to be
 *     retrieved. If the GPC with this ID is not enabled this command will
 *     return an activePesMask of zero
 *
 *   numPesInGpc[OUT]
 *     This parameter returns the number of PES in this GPC.
 *
 *   activePesMask[OUT]
 *     This parameter returns a mask of enabled PESs for the specified GPC.
 *     Each PES has an ID that is equivalent to the corresponding bit position
 *     in the mask.
 *
 *   maxTpcPerGpcCount[OUT]
 *     This parameter returns the max number of TPCs in a GPC.
 *
 *   tpcToPesMap[OUT]
 *     This array stores the TPC to PES mappings. The value at tpcToPesMap[tpcIndex]
 *     is the index of the PES it belongs to.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_PES_INFO                       (0x20800168) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_PES_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_GPU_GET_PES_INFO_MAX_TPC_PER_GPC_COUNT 10

#define LW2080_CTRL_GPU_GET_PES_INFO_PARAMS_MESSAGE_ID (0x68U)

typedef struct LW2080_CTRL_GPU_GET_PES_INFO_PARAMS {
    LwU32 gpcId;
    LwU32 numPesInGpc;
    LwU32 activePesMask;
    LwU32 maxTpcPerGpcCount;
    LwU32 tpcToPesMap[LW2080_CTRL_CMD_GPU_GET_PES_INFO_MAX_TPC_PER_GPC_COUNT];
} LW2080_CTRL_GPU_GET_PES_INFO_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/* LW2080_CTRL_CMD_GPU_GET_OEM_INFO
 *
 * If an InfoROM with a valid OEM Object is present, this
 * command returns relevant information from the object to the
 * caller.
 *
 * oemInfo
 *  This array stores information specifically for OEM use
 *  (e.g. "their own serial number", "lot codes", etc)
 *  "The byte definition is up to the OEM"
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_GPU_GET_OEM_INFO (0x20800169) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_OEM_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_GPU_MAX_OEM_INFO_LENGTH   (0x000001F8)

#define LW2080_CTRL_GPU_GET_OEM_INFO_PARAMS_MESSAGE_ID (0x69U)

typedef struct LW2080_CTRL_GPU_GET_OEM_INFO_PARAMS {
    LwU8 oemInfo[LW2080_GPU_MAX_OEM_INFO_LENGTH];
} LW2080_CTRL_GPU_GET_OEM_INFO_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/* LW2080_CTRL_CMD_GPU_PROCESS_POST_GC6_EXIT_TASKS
 *
 * Complete any pending tasks the need to be run after GC6 exit is complete at OS/KMD level
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_READY
 */
#define LW2080_CTRL_CMD_GPU_PROCESS_POST_GC6_EXIT_TASKS (0x2080016a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x6A" */

/*
 * LW2080_CTRL_CMD_GPU_GET_VPR_INFO
 *
 * This command is used to query the VPR information for a GPU.
 * The following VPR related information can be queried by selecting the queryType:
 *   1. The current VPR range.
 *   2. The max VPR range ever possible on this GPU.
 *
 *   queryType [in]
 *     This input parameter is used to select the type of information to query.
 *     Possible values for this parameter are:
 *       1. LW2080_CTRL_GPU_GET_VPR_INFO_QUERY_VPR_CAPS: Use this to query the
 *              max VPR range ever possible on this GPU.
 *       2. LW2080_CTRL_GPU_GET_VPR_INFO_QUERY_LWR_VPR_RANGE: Use this to query
 *              the current VPR range on this GPU.
 *
 *   bVprEnabled [out]
 *     For query type "LW2080_CTRL_GPU_GET_VPR_INFO_LWR_RANGE", this parameter
 *     returns if VPR is lwrrently enabled or not.
 *
 *   vprStartAddress [out]
 *     For LW2080_CTRL_GPU_GET_VPR_INFO_CAPS, it returns minimum allowed VPR start address.
 *     For LW2080_CTRL_GPU_GET_VPR_INFO_RANGE, it returns current VPR start address.
 *
 *   vprEndAddress [out]
 *     For LW2080_CTRL_GPU_GET_VPR_INFO_CAPS, it returns maximum allowed VPR end address.
 *     For LW2080_CTRL_GPU_GET_VPR_INFO_RANGE, it returns current VPR end address.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_VPR_INFO                (0x2080016b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_VPR_INFO_PARAMS_MESSAGE_ID" */


typedef enum LW2080_CTRL_VPR_INFO_QUERY_TYPE {
    LW2080_CTRL_GPU_GET_VPR_INFO_QUERY_VPR_CAPS = 0,
    LW2080_CTRL_GPU_GET_VPR_INFO_QUERY_LWR_VPR_RANGE = 1,
} LW2080_CTRL_VPR_INFO_QUERY_TYPE;

#define LW2080_CTRL_GPU_GET_VPR_INFO_PARAMS_MESSAGE_ID (0x6BU)

typedef struct LW2080_CTRL_GPU_GET_VPR_INFO_PARAMS {
    LW2080_CTRL_VPR_INFO_QUERY_TYPE queryType;
    LwBool                          bIsVprEnabled;
    LW_DECLARE_ALIGNED(LwU64 vprStartAddressInBytes, 8);
    LW_DECLARE_ALIGNED(LwU64 vprEndAddressInBytes, 8);
} LW2080_CTRL_GPU_GET_VPR_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_ENCODER_CAPACITY
 *
 * This command is used to query the encoder capacity of the GPU.
 *
 *   queryType [in]
 *     This input parameter is used to select the type of information to query.
 *     Possible values for this parameter are:
 *       1. LW2080_CTRL_GPU_GET_ENCODER_CAPACITY_H264: Use this to query the
 *              H.264 encoding capacity on this GPU.
 *       2. LW2080_CTRL_GPU_GET_ENCODER_CAPACITY_HEVC: Use this to query the
 *              H.265/HEVC encoding capacity on this GPU.
 *
 *   encoderCapacity [out]
 *     Encoder capacity value from 0 to 100. Value of 0x00 indicates encoder performance
 *     may be minimal for this GPU and software should fall back to CPU-based encode.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_CMD_GPU_GET_ENCODER_CAPACITY (0x2080016c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ENCODER_CAPACITY_PARAMS_MESSAGE_ID" */

typedef enum LW2080_CTRL_ENCODER_CAPACITY_QUERY_TYPE {
    LW2080_CTRL_GPU_GET_ENCODER_CAPACITY_H264 = 0,
    LW2080_CTRL_GPU_GET_ENCODER_CAPACITY_HEVC = 1,
} LW2080_CTRL_ENCODER_CAPACITY_QUERY_TYPE;

#define LW2080_CTRL_GPU_GET_ENCODER_CAPACITY_PARAMS_MESSAGE_ID (0x6LW)

typedef struct LW2080_CTRL_GPU_GET_ENCODER_CAPACITY_PARAMS {
    LW2080_CTRL_ENCODER_CAPACITY_QUERY_TYPE queryType;
    LwU32                                   encoderCapacity;
} LW2080_CTRL_GPU_GET_ENCODER_CAPACITY_PARAMS;

/*
 * LW2080_CTRL_GPU_GET_LWENC_SW_SESSION_STATS
 *
 * This command is used to retrieve the GPU's count of encoder sessions,
 * trailing average FPS and encode latency over all active sessions.
 *
 *   encoderSessionCount
 *     This field specifies count of all active encoder sessions on this GPU.
 *
 *   averageEncodeFps
 *     This field specifies the average encode FPS for this GPU.
 *
 *   averageEncodeLatency
 *     This field specifies the average encode latency in microseconds for this GPU.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_GPU_GET_LWENC_SW_SESSION_STATS (0x2080016d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_LWENC_SW_SESSION_STATS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_LWENC_SW_SESSION_STATS_PARAMS_MESSAGE_ID (0x6DU)

typedef struct LW2080_CTRL_GPU_GET_LWENC_SW_SESSION_STATS_PARAMS {
    LwU32 encoderSessionCount;
    LwU32 averageEncodeFps;
    LwU32 averageEncodeLatency;
} LW2080_CTRL_GPU_GET_LWENC_SW_SESSION_STATS_PARAMS;

#define LW2080_CTRL_GPU_LWENC_SESSION_INFO_MAX_COPYOUT_ENTRIES 0x200  // 512 entries.

/*
 * LW2080_CTRL_GPU_GET_LWENC_SW_SESSION_INFO
 *
 * This command returns LWENC software sessions information for the associate GPU.
 * Request to retrieve session information use a list of one or more
 * LW2080_CTRL_LWENC_SW_SESSION_INFO structures.
 *
 *   sessionInfoTblEntry
 *     This field specifies the number of entries on the that are filled inside
 *     sessionInfoTbl. Max value of this field once returned from RM would be
 *     LW2080_CTRL_GPU_LWENC_SESSION_INFO_MAX_COPYOUT_ENTRIES,
 *
 *   sessionInfoTbl
 *     This field specifies a pointer in the caller's address space
 *     to the buffer into which the LWENC session information is to be returned.
 *     When buffer is NULL, RM assume that client is querying sessions count value
 *     and return the current encoder session counts in sessionInfoTblEntry field.
 *     To get actual buffer data, client should allocate sessionInfoTbl of size
 *     LW2080_CTRL_GPU_LWENC_SESSION_INFO_MAX_COPYOUT_ENTRIES  multiplied by the
 *     size of the LW2080_CTRL_LWENC_SW_SESSION_INFO structure. RM will fill the
 *     current session data in sessionInfoTbl buffer and then update the
 *     sessionInfoTblEntry to reflect current session count value.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NO_MEMORY
 *   LW_ERR_ILWALID_LOCK_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 */

typedef struct LW2080_CTRL_LWENC_SW_SESSION_INFO {
    LwU32 processId;
    LwU32 subProcessId;
    LwU32 sessionId;
    LwU32 codecType;
    LwU32 hResolution;
    LwU32 vResolution;
    LwU32 averageEncodeFps;
    LwU32 averageEncodeLatency;
} LW2080_CTRL_LWENC_SW_SESSION_INFO;

#define LW2080_CTRL_GPU_GET_LWENC_SW_SESSION_INFO_PARAMS_MESSAGE_ID (0x6EU)

typedef struct LW2080_CTRL_GPU_GET_LWENC_SW_SESSION_INFO_PARAMS {
    LwU32 sessionInfoTblEntry;
    LW_DECLARE_ALIGNED(LwP64 sessionInfoTbl, 8);
} LW2080_CTRL_GPU_GET_LWENC_SW_SESSION_INFO_PARAMS;

#define LW2080_CTRL_GPU_GET_LWENC_SW_SESSION_INFO (0x2080016e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_LWENC_SW_SESSION_INFO_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_GPU_SET_FABRIC_BASE_ADDR
 *
 * The command sets fabric base address which represents top N bits of a
 * peer memory address. These N bits will be used to index LwSwitch routing
 * tables to forward peer memory accesses to associated GPUs.
 *
 * The command is available to clients with administrator privileges only.
 * An attempt to use this command by a client without administrator privileged
 * results in the return of LW_ERR_INSUFFICIENT_PERMISSIONS status.
 *
 * The command allows to set fabricAddr once in a lifetime of a GPU. A GPU must
 * be destroyed in order to re-assign a different fabricAddr. An attempt to
 * re-assign address without destroying a GPU would result in the return of
 * LW_ERR_STATE_IN_USE status.
 *
 *   fabricBaseAddr[IN]
 *      - An address with at least 32GB alignment.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_STATE_IN_USE
 */

#define LW2080_CTRL_GPU_SET_FABRIC_BASE_ADDR_PARAMS_MESSAGE_ID (0x6FU)

typedef struct LW2080_CTRL_GPU_SET_FABRIC_BASE_ADDR_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 fabricBaseAddr, 8);
} LW2080_CTRL_GPU_SET_FABRIC_BASE_ADDR_PARAMS;

#define LW2080_CTRL_CMD_GPU_SET_FABRIC_BASE_ADDR (0x2080016f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_SET_FABRIC_BASE_ADDR_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_GPU_INTERRUPT_FUNCTION
 *
 * The command will trigger an interrupt to a specified PCIe Function.
 *
 *   gfid[IN]
 *      - The GPU function identifier
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_GPU_INTERRUPT_FUNCTION_PARAMS_MESSAGE_ID (0x71U)

typedef struct LW2080_CTRL_GPU_INTERRUPT_FUNCTION_PARAMS {
    LwU32 gfid;
} LW2080_CTRL_GPU_INTERRUPT_FUNCTION_PARAMS;

#define LW2080_CTRL_CMD_GPU_INTERRUPT_FUNCTION (0x20800171) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_INTERRUPT_FUNCTION_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_GPU_VIRTUAL_INTERRUPT
 *
 * The command will trigger the specified interrupt on the host from a guest.
 *
 *   handle[IN]
 *      - An opaque handle that will be passed in along with the interrupt
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_GPU_VIRTUAL_INTERRUPT_PARAMS_MESSAGE_ID (0x72U)

typedef struct LW2080_CTRL_GPU_VIRTUAL_INTERRUPT_PARAMS {
    LwU32 handle;
} LW2080_CTRL_GPU_VIRTUAL_INTERRUPT_PARAMS;

#define LW2080_CTRL_CMD_GPU_VIRTUAL_INTERRUPT                                      (0x20800172) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_VIRTUAL_INTERRUPT_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_GPU_QUERY_FUNCTION_STATUS
 *
 * This control call is to query the status of gpu function registers
 *
 *    statusMask[IN]
 *        - Input mask of required status registers
 *    xusbData[OUT]
 *        - data from querying XUSB status register
 *    ppcData[OUT]
 *        - data from querying PPC status register
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

// Bits to decide which functions to query
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_BIT                             (0x00000002)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_BIT                              (0x00000003)

// XUSB function defines
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_D_STATE                            1:0
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_D_STATE_D0                      (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_D_STATE_D3                      (0x00000003)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_PME_ENABLE                         2:2
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_PME_ENABLE_FALSE                (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_PME_ENABLE_TRUE                 (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_PME_STATUS                         3:3
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_PME_STATUS_FALSE                (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_PME_STATUS_TRUE                 (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_ISOCH_ACTIVE                       4:4
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_ISOCH_ACTIVE_FALSE              (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_ISOCH_ACTIVE_TRUE               (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_HS_LINK_SUSPEND                    16:16
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_HS_LINK_SUSPEND_FALSE           (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_HS_LINK_SUSPEND_TRUE            (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_HS_WCE                             17:17
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_HS_WCE_FALSE                    (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_HS_WCE_TRUE                     (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_HS_WDE                             18:18
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_HS_WDE_FALSE                    (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_HS_WDE_TRUE                     (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_HS_WOE                             19:19
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_HS_WOE_FALSE                    (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_HS_WOE_TRUE                     (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_SS_LINK_SUSPEND                    20:20
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_SS_LINK_SUSPEND_FALSE           (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_SS_LINK_SUSPEND_TRUE            (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_SS_WCE                             21:21
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_SS_WCE_FALSE                    (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_SS_WCE_TRUE                     (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_SS_WDE                             22:22
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_SS_WDE_FALSE                    (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_SS_WDE_TRUE                     (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_SS_WOE                             23:23
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_SS_WOE_FALSE                    (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_SS_WOE_TRUE                     (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_ENABLED                            31:31
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_ENABLED_FALSE                   (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_XUSB_STATUS_ENABLED_TRUE                    (0x00000001)

// PPC function defines
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_D_STATE                             1:0
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_D_STATE_D0                       (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_D_STATE_D3                       (0x00000003)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_PME_ENABLE                          2:2
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_PME_ENABLE_FALSE                 (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_PME_ENABLE_TRUE                  (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_PME_STATUS                          3:3
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_PME_STATUS_FALSE                 (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_PME_STATUS_TRUE                  (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ALT_MODE                            19:16
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ALT_MODE_DISABLED                (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ALT_MODE_USB_ONLY                (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ALT_MODE_DP_ASSIGNMENT_A         (0x00000004)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ALT_MODE_DP_ASSIGNMENT_B         (0x00000005)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ALT_MODE_DP_ASSIGNMENT_CE        (0x00000006)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ALT_MODE_DP_ASSIGNMENT_DF        (0x00000007)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ALT_MODE_DP_ONLY                 (0x0000000C)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ALT_MODE_SUPERLINK_ASSIGNMENT_A  (0x0000000D)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ALT_MODE_SUPERLINK_ASSIGNMENT_CE (0x0000000F)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ORIENTATION                         20:20
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ORIENTATION_NORMAL               (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ORIENTATION_FLIPPED              (0x00000001)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ENABLED                             31:31
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ENABLED_FALSE                    (0x00000000)
#define LW2080_CTRL_GPU_QUERY_FUNCTION_PPC_STATUS_ENABLED_TRUE                     (0x00000001)

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#define LW2080_CTRL_CMD_GPU_QUERY_FUNCTION_STATUS_PARAMS_MESSAGE_ID (0x73U)

typedef struct LW2080_CTRL_CMD_GPU_QUERY_FUNCTION_STATUS_PARAMS {
    LwU32 statusMask;
    LwU32 xusbData;
    LwU32 ppcData;
} LW2080_CTRL_CMD_GPU_QUERY_FUNCTION_STATUS_PARAMS;

#define LW2080_CTRL_CMD_GPU_QUERY_FUNCTION_STATUS (0x20800173) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_CMD_GPU_QUERY_FUNCTION_STATUS_PARAMS_MESSAGE_ID" */

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_GPU_PARTITION_SPAN
 *
 * This struct represents the span of a memory partition, which represents the
 * slices a given partition oclwpies (or may occupy) within a fixed range which
 * is defined per-chip. A partition containing more resources will cover more
 * GPU slices and therefore cover a larger span.
 *
 *   lo
 *      - The starting unit of this span, inclusive
 *
 *   hi
 *      - The ending unit of this span, inclusive
 *
 */
typedef struct LW2080_CTRL_GPU_PARTITION_SPAN {
    LW_DECLARE_ALIGNED(LwU64 lo, 8);
    LW_DECLARE_ALIGNED(LwU64 hi, 8);
} LW2080_CTRL_GPU_PARTITION_SPAN;

/*
 * LW2080_CTRL_GPU_SET_PARTITION_INFO
 *
 * This command partitions a GPU into different SMC-Memory partitions.
 * The command will configure HW partition table to create work and memory
 * isolation.
 *
 * The command is available to clients with administrator privileges only.
 * An attempt to use this command by a client without administrator privileged
 * results in the return of LW_ERR_INSUFFICIENT_PERMISSIONS status.
 *
 * The command allows partitioning an invalid partition only. An attempt to
 * re-partition a valid partition will resule in LW_ERR_STATE_IN_USE.
 * Repartitioning can be done only if a partition has been destroyed/ilwalidated
 * before re-partitioning.
 *
 *   swizzId[IN/OUT]
 *      - PartitionID associated with a newly created partition. Input in case
 *        of partition ilwalidation.
 *
 *   partitionFlag[IN]
 *      - Flags to determine if GPU is requested to be partitioned in FULL,
 *        HALF, QUARTER or ONE_EIGHTHED and whether the partition requires
 *        any additional resources.
 *        When flags include LW2080_CTRL_GPU_PARTITION_FLAG_REQ_DEC_JPG_OFA
 *        partition will be created with at least one video decode, jpeg and
 *        optical flow engines. This flag is valid only for partitions with
 *        a single GPC.
 *
 *   bValid[IN]
 *      - LW_TRUE if creating a partition. LW_FALSE if destroying a partition.
 *
 *   placement[IN]
 *      - Optional placement span to allocate the partition into. Valid
 *        placements are returned from LW2080_CTRL_CMD_GPU_GET_PARTITION_CAPACITY.
 *        The partition flag LW2080_CTRL_GPU_PARTITION_FLAG_PLACE_AT_SPAN must
 *        be set for this parameter to be used. If the flag is set and the given
 *        placement is not valid, an error will be returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_STATE_IN_USE
 */
typedef struct LW2080_CTRL_GPU_SET_PARTITION_INFO {
    LwU32  swizzId;
    LwU32  partitionFlag;
    LwBool bValid;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_PARTITION_SPAN placement, 8);
} LW2080_CTRL_GPU_SET_PARTITION_INFO;

#define PARTITIONID_ILWALID                                    LW2080_CTRL_GPU_PARTITION_ID_ILWALID
#define LW2080_CTRL_GPU_PARTITION_ID_ILWALID                   0xFFFFFFFF
#define LW2080_CTRL_GPU_MAX_PARTITIONS                         0x00000008
#define LW2080_CTRL_GPU_MAX_PARTITION_IDS                      0x00000009
#define LW2080_CTRL_GPU_MAX_SMC_IDS                            0x00000008
#define LW2080_CTRL_GPU_MAX_GPC_PER_SMC                        0x0000000c
#define LW2080_CTRL_GPU_MAX_CE_PER_SMC                         0x00000008

#define LW2080_CTRL_GPU_PARTITION_FLAG_MEMORY_SIZE              1:0
#define LW2080_CTRL_GPU_PARTITION_FLAG_MEMORY_SIZE_FULL        0x00000000
#define LW2080_CTRL_GPU_PARTITION_FLAG_MEMORY_SIZE_HALF        0x00000001
#define LW2080_CTRL_GPU_PARTITION_FLAG_MEMORY_SIZE_QUARTER     0x00000002
#define LW2080_CTRL_GPU_PARTITION_FLAG_MEMORY_SIZE_EIGHTH      0x00000003
#define LW2080_CTRL_GPU_PARTITION_FLAG_MEMORY_SIZE__SIZE       4
#define LW2080_CTRL_GPU_PARTITION_FLAG_COMPUTE_SIZE             4:2
#define LW2080_CTRL_GPU_PARTITION_FLAG_COMPUTE_SIZE_FULL       0x00000000
#define LW2080_CTRL_GPU_PARTITION_FLAG_COMPUTE_SIZE_HALF       0x00000001
#define LW2080_CTRL_GPU_PARTITION_FLAG_COMPUTE_SIZE_MINI_HALF  0x00000002
#define LW2080_CTRL_GPU_PARTITION_FLAG_COMPUTE_SIZE_QUARTER    0x00000003
#define LW2080_CTRL_GPU_PARTITION_FLAG_COMPUTE_SIZE_EIGHTH     0x00000004
#define LW2080_CTRL_GPU_PARTITION_FLAG_COMPUTE_SIZE__SIZE      5
#define LW2080_CTRL_GPU_PARTITION_MAX_TYPES                    8
#define LW2080_CTRL_GPU_PARTITION_FLAG_REQ_DEC_JPG_OFA          30:30
#define LW2080_CTRL_GPU_PARTITION_FLAG_REQ_DEC_JPG_OFA_DISABLE 0
#define LW2080_CTRL_GPU_PARTITION_FLAG_REQ_DEC_JPG_OFA_ENABLE  1
#define LW2080_CTRL_GPU_PARTITION_FLAG_PLACE_AT_SPAN            31:31
#define LW2080_CTRL_GPU_PARTITION_FLAG_PLACE_AT_SPAN_DISABLE   0
#define LW2080_CTRL_GPU_PARTITION_FLAG_PLACE_AT_SPAN_ENABLE    1

// TODO XXX Bug 2657907 Remove these once clients update
#define LW2080_CTRL_GPU_PARTITION_FLAG_FULL_GPU                 (DRF_DEF(2080, _CTRL_GPU_PARTITION_FLAG, _MEMORY_SIZE, _FULL)    | DRF_DEF(2080, _CTRL_GPU_PARTITION_FLAG, _COMPUTE_SIZE, _FULL))
#define LW2080_CTRL_GPU_PARTITION_FLAG_ONE_HALF_GPU             (DRF_DEF(2080, _CTRL_GPU_PARTITION_FLAG, _MEMORY_SIZE, _HALF)    | DRF_DEF(2080, _CTRL_GPU_PARTITION_FLAG, _COMPUTE_SIZE, _HALF))
#define LW2080_CTRL_GPU_PARTITION_FLAG_ONE_MINI_HALF_GPU        (DRF_DEF(2080, _CTRL_GPU_PARTITION_FLAG, _MEMORY_SIZE, _HALF)    | DRF_DEF(2080, _CTRL_GPU_PARTITION_FLAG, _COMPUTE_SIZE, _MINI_HALF))
#define LW2080_CTRL_GPU_PARTITION_FLAG_ONE_QUARTER_GPU          (DRF_DEF(2080, _CTRL_GPU_PARTITION_FLAG, _MEMORY_SIZE, _QUARTER) | DRF_DEF(2080, _CTRL_GPU_PARTITION_FLAG, _COMPUTE_SIZE, _QUARTER))
#define LW2080_CTRL_GPU_PARTITION_FLAG_ONE_EIGHTHED_GPU         (DRF_DEF(2080, _CTRL_GPU_PARTITION_FLAG, _MEMORY_SIZE, _EIGHTH)  | DRF_DEF(2080, _CTRL_GPU_PARTITION_FLAG, _COMPUTE_SIZE, _EIGHTH))

#define LW2080_CTRL_GPU_SET_PARTITIONS_PARAMS_MESSAGE_ID (0x74U)

typedef struct LW2080_CTRL_GPU_SET_PARTITIONS_PARAMS {
    LwU32 partitionCount;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_SET_PARTITION_INFO partitionInfo[LW2080_CTRL_GPU_MAX_PARTITIONS], 8);
} LW2080_CTRL_GPU_SET_PARTITIONS_PARAMS;

#define LW2080_CTRL_CMD_GPU_SET_PARTITIONS (0x20800174) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_SET_PARTITIONS_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_GPU_GET_PARTITION_INFO
 *
 * This command gets the partition information for requested partitions.
 * If GPU is not partitioned, the control call will return LW_ERR_NOT_SUPPORTED.
 *
 * The command will can return global partition information as well as single
 * partition information if global flag is not set.
 * In bare-metal user-mode can request all partition info while in virtualization
 * plugin should make an RPC with swizzId which is assigned to the requesting
 * VM.
 *
 *   swizzId[IN]
 *      - HW Partition ID associated with the requested partition.
 *
 *   partitionFlag[OUT]
 *      - partitionFlag that was provided during partition creation.
 *
 *   grEngCount[OUT]
 *      - Number of SMC engines/GR engines allocated in partition
 *        GrIDs in a partition will always start from 0 and end at grEngCount-1
 *
 *   veidCount[OUT]
 *      - VEID Count assigned to a partition. These will be divided across
 *        SMC engines once CONFIGURE_PARTITION call has been made. The current
 *        algorithm is to assign veidPerGpc * gpcCountPerSmc to a SMC engine.
 *
 *   smCount[OUT]
 *      - SMs assigned to a partition.
 *
 *   ceCount[OUT]
 *      - Copy Engines assigned to a partition.
 *
 *   lwEncCount[OUT]
 *      - LwEnc Engines assigned to a partition.
 *
 *   lwDecCount[OUT]
 *      - LwDec Engines assigned to a partition.
 *
 *   lwJpgCount[OUT]
 *      - LwJpg Engines assigned to a partition.
 *
 *   gpcCount[OUT]
 *      - Max GPCs assigned to a partition.
 *
 *   gpcsPerGr[LW2080_CTRL_GPU_MAX_SMC_IDS][OUT]
 *      - GPC count associated with every valid SMC/Gr.
 *
 *   veidsPerGr[LW2080_CTRL_GPU_MAX_SMC_IDS][OUT]
 *      - VEID count associated with every valid SMC. VEIDs within this SMC
 *        will start from 0 and go till veidCount[SMC_ID] - 1.
 *
 *   span[OUT]
 *      - The span covered by this partition
 *
 *   bValid[OUT]
 *      - LW_TRUE if partition is valid else LW_FALSE.
 *
 *   bPartitionError[OUT]
 *      - LW_TRUE if partition had poison error which requires drain and reset
 *        else LW_FALSE.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_NOT_SUPPORTED
 */
typedef struct LW2080_CTRL_GPU_GET_PARTITION_INFO {
    LwU32  swizzId;
    LwU32  partitionFlag;
    LwU32  grEngCount;
    LwU32  veidCount;
    LwU32  smCount;
    LwU32  ceCount;
    LwU32  lwEncCount;
    LwU32  lwDecCount;
    LwU32  lwJpgCount;
    LwU32  lwOfaCount;
    LwU32  gpcCount;
    LwU32  gpcsPerGr[LW2080_CTRL_GPU_MAX_SMC_IDS];
    LwU32  veidsPerGr[LW2080_CTRL_GPU_MAX_SMC_IDS];
    LW_DECLARE_ALIGNED(LwU64 memSize, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_PARTITION_SPAN span, 8);
    LwBool bValid;
    LwBool bPartitionError;
} LW2080_CTRL_GPU_GET_PARTITION_INFO;

/*
 * LW2080_CTRL_GPU_GET_PARTITIONS_PARAMS
 *
 *   queryPartitionInfo[IN]
 *      - Max sized array of LW2080_CTRL_GPU_GET_PARTITION_INFO to get partition
 *       Info
 *
 *   bGetAllPartitionInfo[In]
 *      - Flag to get all partitions info. Only root client will receive all
 *        partition's info. Non-Root clients should not use this flag
 *
 *   validPartitionCount[Out]
 *      - Valid partition count which has been filled by RM as part of the call
 *
 */
#define LW2080_CTRL_GPU_GET_PARTITIONS_PARAMS_MESSAGE_ID (0x75U)

typedef struct LW2080_CTRL_GPU_GET_PARTITIONS_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_GET_PARTITION_INFO queryPartitionInfo[LW2080_CTRL_GPU_MAX_PARTITIONS], 8);
    LwU32  validPartitionCount;
    LwBool bGetAllPartitionInfo;
} LW2080_CTRL_GPU_GET_PARTITIONS_PARAMS;

#define LW2080_CTRL_CMD_GPU_GET_PARTITIONS      (0x20800175) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_PARTITIONS_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_GPU_CONFIGURE_PARTITION
 *
 * This command configures a partition by associating GPCs with SMC Engines
 * available in that partition. Engines which are to have GPCs assigned to them
 * shall not already have any GPCs assigned to them. It is not valid to both
 * assign GPCs and remove GPCs as part of a single call to this function.
 *
 *   swizzId[IN]
 *      - PartitionID for configuring partition. If partition has a valid
 *        context created, then configuration is not allowed.
 *
 *   gpcCountPerSmcEng[IN]
 *      - Number of GPCs expected to be configured per SMC. Supported
 *        configurations are 0, 1, 2, 4 or 8. "0" means a particular SMC
 *        engine will be disabled with no GPC connected to it.
 *
 *   updateSmcEngMask[IN]
 *      - Mask tracking valid entries of gpcCountPerSmcEng. A value of
 *        0 in bit index i indicates that engine i will keep its current
 *        configuration.
 *
 *   bUseAllGPCs[IN]
 *      - Flag specifying alternate configuration mode, indicating that in
 *        swizzid 0 only, all non-floorswept GPCs should be connected to the
 *        engine indicated by a raised bit in updateSmcEngMask. Only a single
 *        engine may be targeted by this operation. The gpcCountPerSmcEng
 *        parameter should not be used with this flag.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_INSUFFICIENT_RESOURCES
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_STATE_IN_USE
 */
#define LW2080_CTRL_CMD_GPU_CONFIGURE_PARTITION (0x20800176) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_CONFIGURE_PARTITION_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_CONFIGURE_PARTITION_PARAMS_MESSAGE_ID (0x76U)

typedef struct LW2080_CTRL_GPU_CONFIGURE_PARTITION_PARAMS {
    LwU32  swizzId;
    LwU32  gpcCountPerSmcEng[LW2080_CTRL_GPU_MAX_SMC_IDS];
    LwU32  updateSmcEngMask;
    LwBool bUseAllGPCs;
} LW2080_CTRL_GPU_CONFIGURE_PARTITION_PARAMS;

/*
 * LW2080_CTRL_GPU_FAULT_PACKET
 *
 * This struct represents a GMMU fault packet.
 *
 */
#define LW2080_CTRL_GPU_FAULT_PACKET_SIZE 32
typedef struct LW2080_CTRL_GPU_FAULT_PACKET {
    LwU8 data[LW2080_CTRL_GPU_FAULT_PACKET_SIZE];
} LW2080_CTRL_GPU_FAULT_PACKET;

/*
 * LW2080_CTRL_GPU_REPORT_NON_REPLAYABLE_FAULT
 *
 * This command reports a nonreplayable fault packet to RM.
 * It is only used by UVM.
 *
 *   pFaultPacket[IN]
 *      - A fault packet that will be later cast to GMMU_FAULT_PACKET *.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_GPU_REPORT_NON_REPLAYABLE_FAULT (0x20800177) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_REPORT_NON_REPLAYABLE_FAULT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_REPORT_NON_REPLAYABLE_FAULT_PARAMS_MESSAGE_ID (0x77U)

typedef struct LW2080_CTRL_GPU_REPORT_NON_REPLAYABLE_FAULT_PARAMS {
    LW2080_CTRL_GPU_FAULT_PACKET faultPacket;
} LW2080_CTRL_GPU_REPORT_NON_REPLAYABLE_FAULT_PARAMS;

/*
 *  LW2080_CTRL_CMD_GPU_EXEC_REG_OPS_VGPU
 *
 *  This command is similar to LW2080_CTRL_CMD_GPU_EXEC_REG_OPS, except it is used
 *  by the VGPU plugin client only. This command provides access to the subset of
 *  privileged registers.
 *
 *  See confluence page "vGPU UMED Security" for details.
 *
 */
#define LW2080_CTRL_CMD_GPU_EXEC_REG_OPS_VGPU           (0x20800178) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x78" */

/*
 * LW2080_CTRL_CMD_GPU_GET_ENGINE_RUNLIST_PRI_BASE
 *
 * This command returns the runlist pri base of the specified engine(s).
 *
 *   engineList
 *     Input array.
 *     This array specifies the engines being queried for information.
 *     The list of engines supported by a chip can be fetched using the
 *     LW2080_CTRL_CMD_GPU_GET_ENGINES/GET_ENGINES_V2 ctrl call.
 *
 *   runlistPriBase
 *     Output array.
 *     Returns the runlist pri base for the specified engines
 *     Else, will return _NULL when the input is a LW2080_ENGINE_TYPE_NULL
 *     and will return _ERROR when the control call fails due to an invalid argument
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_ENGINE_RUNLIST_PRI_BASE (0x20800179) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ENGINE_RUNLIST_PRI_BASE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_ENGINE_RUNLIST_PRI_BASE_PARAMS_MESSAGE_ID (0x79U)

typedef struct LW2080_CTRL_GPU_GET_ENGINE_RUNLIST_PRI_BASE_PARAMS {
    LwU32 engineList[LW2080_GPU_MAX_ENGINES_LIST_SIZE];
    LwU32 runlistPriBase[LW2080_GPU_MAX_ENGINES_LIST_SIZE];
} LW2080_CTRL_GPU_GET_ENGINE_RUNLIST_PRI_BASE_PARAMS;

#define LW2080_CTRL_GPU_GET_ENGINE_RUNLIST_PRI_BASE_NULL  (0xFFFFFFFF)
#define LW2080_CTRL_GPU_GET_ENGINE_RUNLIST_PRI_BASE_ERROR (0xFFFFFFFB)

/*
 * LW2080_CTRL_CMD_GPU_GET_HW_ENGINE_ID
 *
 * This command returns the host hardware defined engine ID of the specified engine(s).
 *
 *   engineList
 *     Input array.
 *     This array specifies the engines being queried for information.
 *     The list of engines supported by a chip can be fetched using the
 *     LW2080_CTRL_CMD_GPU_GET_ENGINES/GET_ENGINES_V2 ctrl call.
 *
 *   hwEngineID
 *     Output array.
 *     Returns the host hardware engine ID(s) for the specified engines
 *     Else, will return _NULL when the input is a LW2080_ENGINE_TYPE_NULL
 *     and will return _ERROR when the control call fails due to an invalid argument
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_HW_ENGINE_ID              (0x2080017a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_HW_ENGINE_ID_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_HW_ENGINE_ID_PARAMS_MESSAGE_ID (0x7AU)

typedef struct LW2080_CTRL_GPU_GET_HW_ENGINE_ID_PARAMS {
    LwU32 engineList[LW2080_GPU_MAX_ENGINES_LIST_SIZE];
    LwU32 hwEngineID[LW2080_GPU_MAX_ENGINES_LIST_SIZE];
} LW2080_CTRL_GPU_GET_HW_ENGINE_ID_PARAMS;

#define LW2080_CTRL_GPU_GET_HW_ENGINE_ID_NULL      (0xFFFFFFFF)
#define LW2080_CTRL_GPU_GET_HW_ENGINE_ID_ERROR     (0xFFFFFFFB)

/*
 * LW2080_CTRL_GPU_GET_LWFBC_SW_SESSION_STATS
 *
 * This command is used to retrieve the GPU's count of FBC sessions,
 * average FBC calls and FBC latency over all active sessions.
 *
 *   sessionCount
 *     This field specifies count of all active fbc sessions on this GPU.
 *
 *   averageFPS
 *     This field specifies the average frames captured.
 *
 *   averageLatency
 *     This field specifies the average FBC latency in microseconds.
 *
 * Possible status values returned are :
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
*/
#define LW2080_CTRL_GPU_GET_LWFBC_SW_SESSION_STATS (0x2080017b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_LWFBC_SW_SESSION_STATS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_LWFBC_SW_SESSION_STATS_PARAMS_MESSAGE_ID (0x7BU)

typedef struct LW2080_CTRL_GPU_GET_LWFBC_SW_SESSION_STATS_PARAMS {
    LwU32 sessionCount;
    LwU32 averageFPS;
    LwU32 averageLatency;
} LW2080_CTRL_GPU_GET_LWFBC_SW_SESSION_STATS_PARAMS;

/*
* LW2080_CTRL_LWFBC_SW_SESSION_INFO
*
*   processId[OUT]
*           Process id of the process owning the LwFBC session.
*           On VGX host, this will specify the vGPU plugin process id.
*   subProcessId[OUT]
*           Process id of the process owning the LwFBC session if the
*           session is on VGX guest, else the value is zero.
*   vgpuInstanceId[OUT]
*           vGPU on which the process owning the LwFBC session
*           is running if session is on VGX guest, else
*           the value is zero.
*   sessionId[OUT]
*           Unique session id of the LwFBC session.
*   sessionType[OUT]
*           Type of LwFBC session.
*   displayOrdinal[OUT]
*           Display identifier associated with the LwFBC session.
*   sessionFlags[OUT]
*           One or more of LW2080_CTRL_LWFBC_SESSION_FLAG_xxx.
*   hMaxResolution[OUT]
*           Max horizontal resolution supported by the LwFBC session.
*   vMaxResolution[OUT]
*           Max vertical resolution supported by the LwFBC session.
*   hResolution[OUT]
*           Horizontal resolution requested by caller in grab call.
*   vResolution[OUT]
*           Vertical resolution requested by caller in grab call.
*   averageFPS[OUT]
*           Average no. of frames captured per second.
*   averageLatency[OUT]
*           Average frame capture latency in microseconds.
*/

#define LW2080_CTRL_LWFBC_SESSION_FLAG_DIFFMAP_ENABLED            0x00000001
#define LW2080_CTRL_LWFBC_SESSION_FLAG_CLASSIFICATIONMAP_ENABLED  0x00000002
#define LW2080_CTRL_LWFBC_SESSION_FLAG_CAPTURE_WITH_WAIT_NO_WAIT  0x00000004
#define LW2080_CTRL_LWFBC_SESSION_FLAG_CAPTURE_WITH_WAIT_INFINITE 0x00000008
#define LW2080_CTRL_LWFBC_SESSION_FLAG_CAPTURE_WITH_WAIT_TIMEOUT  0x00000010

typedef struct LW2080_CTRL_LWFBC_SW_SESSION_INFO {
    LwU32 processId;
    LwU32 subProcessId;
    LwU32 vgpuInstanceId;
    LwU32 sessionId;
    LwU32 sessionType;
    LwU32 displayOrdinal;
    LwU32 sessionFlags;
    LwU32 hMaxResolution;
    LwU32 vMaxResolution;
    LwU32 hResolution;
    LwU32 vResolution;
    LwU32 averageFPS;
    LwU32 averageLatency;
} LW2080_CTRL_LWFBC_SW_SESSION_INFO;

/*
* LW2080_CTRL_GPU_GET_LWFBC_SW_SESSION_INFO
*
* This command returns LWFBC software sessions information for the associate GPU.
*
*   sessionInfoCount
*     This field specifies the number of entries that are filled inside
*     sessionInfoTbl. Max value of this field once returned from RM would be
*     LW2080_GPU_LWFBC_MAX_COUNT.
*
*   sessionInfoTbl
*     This field specifies the array in which the LWFBC session information is to
*     be returned. RM will fill the current session data in sessionInfoTbl array
*     and then update the sessionInfoCount to reflect current session count value.
*
* Possible status values returned are:
*   LW_OK
*   LW_ERR_NO_MEMORY
*   LW_ERR_ILWALID_LOCK_STATE
*   LW_ERR_ILWALID_ARGUMENT
*/

#define LW2080_GPU_LWFBC_MAX_SESSION_COUNT 256

#define LW2080_CTRL_GPU_GET_LWFBC_SW_SESSION_INFO_PARAMS_MESSAGE_ID (0x7LW)

typedef struct LW2080_CTRL_GPU_GET_LWFBC_SW_SESSION_INFO_PARAMS {
    LwU32                             sessionInfoCount;
    LW2080_CTRL_LWFBC_SW_SESSION_INFO sessionInfoTbl[LW2080_GPU_LWFBC_MAX_SESSION_COUNT];
} LW2080_CTRL_GPU_GET_LWFBC_SW_SESSION_INFO_PARAMS;

#define LW2080_CTRL_GPU_GET_LWFBC_SW_SESSION_INFO (0x2080017c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_LWFBC_SW_SESSION_INFO_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_GPU_GET_PHYS_SYS_PIPE_IDS
 *
 * This command returns an array of non-floorswept sys pipes for the associated
 * subdevice. If the client is subscribed to a partition, this interface will
 * return the physical syspipes belonging to the partition.
 * There are multiple cases which are supported by this call
 *      SMC-Disabled - Only return GR0 in ID and "1" in count
 *      SMC Enabled with partition subscription - Returns partition localIds and count
 *      SMC Enabled with device_monitoring but no input swizzID - Returns device
 *          level data i.e. all available syspipes physical ID and count.
 *      SMC Enabled with device_monitoring and valid input swizzID - Attribution case.
            Returns physical syspipe IDs associated with a partition and count.
            This is allowed only for Mods and device monitoring case
 *
 *    swizzId[In]
 *     GPU Partition Instance ID in case device monitoring session/Mods is requesting
 *     partition to physical syspipe mapping
 *     This should be set to LW2080_CTRL_GPU_PARTITION_ID_ILWALID if device_level
 *     info is requested
 *
 *   physSysPipeIds[OUT]
 *     This parameter returns an array of physical indices corresponding to
 *     non-floorswept sys pipes. The array index is the partition localId for
 *     specific physical syspipe
 *
 *    physSysPipeCount[OUT]
 *     Number of valid physical syspipe count
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_GPU_GET_PHYS_SYS_PIPE_IDS (0x2080017d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_PHYS_SYS_PIPE_IDS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_PHYS_SYS_PIPE_IDS_PARAMS_MESSAGE_ID (0x7DU)

typedef struct LW2080_CTRL_GPU_GET_PHYS_SYS_PIPE_IDS_PARAMS {
    LwU32 swizzId;
    LwU32 physSysPipeId[LW2080_CTRL_GPU_MAX_SMC_IDS];
    LwU32 physSysPipeCount;
} LW2080_CTRL_GPU_GET_PHYS_SYS_PIPE_IDS_PARAMS;
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_CMD_GPU_GET_VMMU_SEGMENT_SIZE
 *
 * This command returns the VMMU page size
 *
 *   vmmuSegmentSize
 *     Output parameter.
 *     Returns the VMMU segment size (in bytes)
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_VMMU_SEGMENT_SIZE (0x2080017e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_VMMU_SEGMENT_SIZE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_VMMU_SEGMENT_SIZE_PARAMS_MESSAGE_ID (0x7EU)

typedef struct LW2080_CTRL_GPU_GET_VMMU_SEGMENT_SIZE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 vmmuSegmentSize, 8);
} LW2080_CTRL_GPU_GET_VMMU_SEGMENT_SIZE_PARAMS;

#define LW2080_CTRL_GPU_VMMU_SEGMENT_SIZE_32MB     0x02000000
#define LW2080_CTRL_GPU_VMMU_SEGMENT_SIZE_64MB     0x04000000
#define LW2080_CTRL_GPU_VMMU_SEGMENT_SIZE_128MB    0x08000000
#define LW2080_CTRL_GPU_VMMU_SEGMENT_SIZE_256MB    0x10000000
#define LW2080_CTRL_GPU_VMMU_SEGMENT_SIZE_512MB    0x20000000

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GPU_QUERY_ACTIVATION_STATE
 *
 * This control call has been disabled as the effort has been abandoned
 * in production. Any callers should remove their references to this
 * control call.
 *
 * See LWDARM-2383 for more justification.
 */
#define LW2080_CTRL_CMD_GPU_QUERY_ACTIVATION_STATE (0x2080017f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x7F" */

typedef struct LW2080_CTRL_GPU_QUERY_ACTIVATION_STATE_PARAMS {
    LwBool bActivationState;
} LW2080_CTRL_GPU_QUERY_ACTIVATION_STATE_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_SET_ACTIVATION_STATE
 *
 * This control call has been disabled as the effort has been abandoned
 * in production. Any callers should remove their references to this
 * control call.
 *
 * See LWDARM-2383 for more justification.
 */
#define LW2080_CTRL_CMD_GPU_SET_ACTIVATION_STATE (0x20800180) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x80" */

typedef struct LW2080_CTRL_GPU_SET_ACTIVATION_STATE_PARAMS {
    LwBool bActivationState;
} LW2080_CTRL_GPU_SET_ACTIVATION_STATE_PARAMS;

/*
 * LW2080_CTRL_GPU_GET_PARTITION_CAPACITY
 *
 * This command returns the count of partitions of given size (represented by
 * LW2080_CTRL_GPU_PARTITION_FLAG_*) which can be requested via
 * LW2080_CTRL_GPU_SET_PARTITIONS ctrl call.
 * Note that this API does not "reserve" any partitions, and there is no
 * guarantee that the reported count of available partitions of a given size
 * will remain consistent following creation of partitions of different size
 * through LW2080_CTRL_GPU_SET_PARTITIONS.
 * Note that this API is unsupported if SMC is feature-disabled.
 *
 *   partitionFlag[IN]
 *      - Partition flag indicating size of requested partitions
 *
 *   partitionCount[OUT]
 *      - Available number of partitions of the given size which can lwrrently be created.
 *
 *   availableSpans[OUT]
 *      - For each partition able to be created of the specified size, the span
 *        it could occupy.
 *
 *   availableSpansCount[OUT]
 *      - Number of valid entries in availableSpans.
 *
 *   totalPartitionCount[OUT]
 *      - Total number of partitions of the given size which can be created.
 *
 *   totalSpans[OUT]
 *      - List of spans which can possibly be oclwpied by partitions of the
 *        given type.
 *
 *   totalSpansCount[OUT]
 *      - Number of valid entries in totalSpans.
 *
 *   bStaticInfo[IN]
 *      - Flag indicating that client requests only the information from
 *        totalPartitionCount and totalSpans.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_PARTITION_CAPACITY (0x20800181) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_PARTITION_CAPACITY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_PARTITION_CAPACITY_PARAMS_MESSAGE_ID (0x81U)

typedef struct LW2080_CTRL_GPU_GET_PARTITION_CAPACITY_PARAMS {
    LwU32  partitionFlag;
    LwU32  partitionCount;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_PARTITION_SPAN availableSpans[LW2080_CTRL_GPU_MAX_PARTITIONS], 8);
    LwU32  availableSpansCount;
    LwU32  totalPartitionCount;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_PARTITION_SPAN totalSpans[LW2080_CTRL_GPU_MAX_PARTITIONS], 8);
    LwU32  totalSpansCount;
    LwBool bStaticInfo;
} LW2080_CTRL_GPU_GET_PARTITION_CAPACITY_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_CACHED_INFO
 *
 * This command returns cached(SW only) gpu information for the associated GPU.
 * Requests to retrieve gpu information use a list of one or more LW2080_CTRL_GPU_INFO
 * structures.
 * The gpuInfoList is aligned with LW2080_CTRL_GPU_GET_INFO_V2_PARAMS for security concern
 *
 *   gpuInfoListSize
 *     This field specifies the number of entries on the caller's
 *     gpuInfoList.
 *   gpuInfoList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer into which the gpu information is to be returned.
 *     This buffer must be at least as big as gpuInfoListSize multiplied
 *     by the size of the LW2080_CTRL_GPU_INFO structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_GPU_GET_CACHED_INFO (0x20800182) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | 0x82" */

typedef struct LW2080_CTRL_GPU_GET_CACHED_INFO_PARAMS {
    LwU32                gpuInfoListSize;
    LW2080_CTRL_GPU_INFO gpuInfoList[LW2080_CTRL_GPU_INFO_MAX_LIST_SIZE];
} LW2080_CTRL_GPU_GET_CACHED_INFO_PARAMS;

/*
 * LW2080_CTRL_GPU_SET_PARTITIONING_MODE
 *
 * This command configures this GPU to control global mode for partitioning.
 * This command may not be sent to a GPU with any active partitions.
 * This command may be used to set the following modes:
 *
 * LW2080_CTRL_GPU_SET_PARTITIONING_MODE_REPARTITIONING
 *  LW2080_CTRL_GPU_SET_PARTITIONING_MODE_REPARTITIONING_LEGACY
 *      This is the default mode. While this GPU is in this mode, no partitions
 *      will be allowed to be created via SET_PARTITIONS - a client must set one
 *      of the below modes prior to partitioning the GPU. When a client sets a
 *      GPU into this mode, any performance changes resulting from partitions
 *      made while in either of the below modes will be cleared. A
 *      physical-function-level reset is required after setting this mode.
 *
 *  LW2080_CTRL_GPU_SET_PARTITIONING_MODE_REPARTITIONING_MAX_PERF
 *      In this mode, when the GPU is partitioned, each partition will have the
 *      maximum possible performance which can be evenly distributed among all
 *      partitions. The total performance of the GPU, taking into account all
 *      partitions created in this mode, may be less than that of a GPU running
 *      in legacy non-SMC mode. Partitions created while in this mode require a
 *      physical-function-level reset before the partitioning may take full
 *      effect.  Destroying all partitions while in this mode may be
 *      insufficient to restore full performance to the GPU - only by setting
 *      the mode to _LEGACY can this be achieved. A physical-function-level
 *      reset is NOT required after setting this mode.
 *
 *  LW2080_CTRL_GPU_SET_PARTIITONING_MODE_REPARTITIONING_FAST_RECONFIG
 *      By setting this mode, the performance of the GPU will be restricted such
 *      that all partitions will have a consistent fraction of the total
 *      available performance, which may be less than the maximum possible
 *      performance available to each partition. Creating or destroying
 *      partitions on this GPU while in this mode will not require a
 *      physical-function-level reset, and will not affect other active
 *      partitions. Destroying all partitions while in this mode may be
 *      insufficient to restore full performance to the GPU - only by setting
 *      the mode to _LEGACY can this be achieved. A physical-function-level
 *      reset is required after setting this mode.
 *
 * Parameters:
 *   partitioningMode[IN]
 *      - Partitioning Mode to set for this GPU.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_SET_PARTITIONING_MODE                          (0x20800183) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_SET_PARTITIONING_MODE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_SET_PARTITIONING_MODE_REPARTITIONING                1:0
#define LW2080_CTRL_GPU_SET_PARTITIONING_MODE_REPARTITIONING_LEGACY        0
#define LW2080_CTRL_GPU_SET_PARTITIONING_MODE_REPARTITIONING_MAX_PERF      1
#define LW2080_CTRL_GPU_SET_PARTITIONING_MODE_REPARTITIONING_FAST_RECONFIG 2

#define LW2080_CTRL_GPU_SET_PARTITIONING_MODE_PARAMS_MESSAGE_ID (0x83U)

typedef struct LW2080_CTRL_GPU_SET_PARTITIONING_MODE_PARAMS {
    LwU32 partitioningMode;
} LW2080_CTRL_GPU_SET_PARTITIONING_MODE_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_ENGINE_TYPE_AND_INSTANCE
 *
 * This command returns the HW engine types and instance IDs of the specified
 * engine(s). These are direct HW enumerants and meant only for use by lwwddm
 * in order to communicate with lwwatch which engine it is talking about.
 * No other client code is expected to make use of it nor can it do anything
 * useful with it. The engine type and instance ID together uniquely identifies
 * a HW engine.
 *
 *   engineList
 *     Input array.
 *     This array specifies the engines being queried for information.
 *     The list of engines supported by a chip can be fetched using the
 *     LW2080_CTRL_CMD_GPU_GET_ENGINES/GET_ENGINES_V2 ctrl call.
 *
 *   engineType
 *     Output array.
 *     Returns the engine type for the specified engine(s)
 *     Else, will return LW2080_CTRL_GPU_GET_ENGINE_TYPE_NULL when the
 *     input engine is a LW2080_ENGINE_TYPE_NULL
 *     Else, will return LW2080_CTRL_GPU_GET_ENGINE_TYPE_ERROR when the
 *     control call fails due to an invalid argument
 *
 *   engineInstance
 *     Returns the instance id for the specified engine(s)
 *     Else, will return LW2080_CTRL_GPU_GET_ENGINE_INSTANCE_NULL when the
 *     input engine is a LW2080_ENGINE_TYPE_NULL
 *     Else, will return LW2080_CTRL_GPU_GET_ENGINE_INSTANCE_ERROR when the
 *     control call fails due to an invalid argument
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_ENGINE_TYPE_AND_INSTANCE (0x20800184) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ENGINE_TYPE_AND_INSTANCE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_ENGINE_TYPE_AND_INSTANCE_PARAMS_MESSAGE_ID (0x84U)

typedef struct LW2080_CTRL_GPU_GET_ENGINE_TYPE_AND_INSTANCE_PARAMS {
    LwU32 engineList[LW2080_GPU_MAX_ENGINES_LIST_SIZE];
    LwU32 engineType[LW2080_GPU_MAX_ENGINES_LIST_SIZE];
    LwU32 engineInstance[LW2080_GPU_MAX_ENGINES_LIST_SIZE];
} LW2080_CTRL_GPU_GET_ENGINE_TYPE_AND_INSTANCE_PARAMS;

#define LW2080_CTRL_GPU_GET_ENGINE_TYPE_NULL      (0xFFFFFFFF)
#define LW2080_CTRL_GPU_GET_ENGINE_INSTANCE_NULL  (0xFFFFFFFF)

#define LW2080_CTRL_GPU_GET_ENGINE_TYPE_ERROR     (0xFFFFFFFB)
#define LW2080_CTRL_GPU_GET_ENGINE_INSTANCE_ERROR (0xFFFFFFFB)

/* LW2080_CTRL_GPU_DESCRIBE_PARTITIONS_INFO
 *
 * This structure describes resources available in a partition requested of a
 * given type.
 *
 * [OUT] partitionFlag
 *      - Flags to specify in LW2080_CTRL_CMD_GPU_SET_PARTITIONS to request this
 *        partition
 *
 * [OUT] grCount
 *      - Number of SMC engines/GR engines
 *
 * [OUT] gpcCount
 *      - Number of GPCs in this partition
 *
 * [OUT] veidCount
 *      - Number of VEIDS in this partition
 *
 * [OUT] smCount
 *      - Number of SMs in this partition
 *
 * [OUT] ceCount
 *      - Copy Engines in this partition
 *
 * [OUT] lwEncCount
 *      - Encoder Engines in this partition
 *
 * [OUT] lwDecCount
 *      - Decoder Engines in this partition
 *
 * [OUT] lwJpgCount
 *      - Jpg Engines in this partition
 *
 * [OUT] lwOfaCount
 *      - Ofa engines in this partition
 * [OUT] memorySize
 *      - Total available memory within this partition
 */
typedef struct LW2080_CTRL_GPU_DESCRIBE_PARTITIONS_INFO {
    LwU32 partitionFlag;
    LwU32 grCount;
    LwU32 gpcCount;
    LwU32 veidCount;
    LwU32 smCount;
    LwU32 ceCount;
    LwU32 lwEncCount;
    LwU32 lwDecCount;
    LwU32 lwJpgCount;
    LwU32 lwOfaCount;
    LW_DECLARE_ALIGNED(LwU64 memorySize, 8);
} LW2080_CTRL_GPU_DESCRIBE_PARTITIONS_INFO;

/*
 * LW2080_CTRL_GPU_DESCRIBE_PARTITIONS_PARAMS
 *
 * This command returns information regarding GPU partitions which can be
 * requested via LW2080_CTRL_CMD_GPU_SET_PARTITIONS.
 *
 * [OUT] descCount
 *      - Number of valid partition types
 *
 * [OUT] partitionDescs
 *      - Information describing available partitions
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_GPU_DESCRIBE_PARTITIONS_PARAMS_MESSAGE_ID (0x85U)

typedef struct LW2080_CTRL_GPU_DESCRIBE_PARTITIONS_PARAMS {
    LwU32 descCount;
    // C form: LW2080_CTRL_GPU_DESCRIBE_PARTITION_INFO partitionDescs[LW2080_CTRL_GPU_PARTITION_MAX_TYPES];
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_DESCRIBE_PARTITIONS_INFO partitionDescs[LW2080_CTRL_GPU_PARTITION_MAX_TYPES], 8);
} LW2080_CTRL_GPU_DESCRIBE_PARTITIONS_PARAMS;

#define LW2080_CTRL_CMD_GPU_DESCRIBE_PARTITIONS (0x20800185) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_DESCRIBE_PARTITIONS_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_GPU_ENABLE_GRID_FEATURE
 *
 * Communicates to RM to enable the feature provided as input parameter.
 *
 *   featureCode
 *     feature to be enabled.
 *
 * Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 *   LWOS_STATUS_ERROR_ILWALID_DATA
 *   LWOS_STATUS_ERROR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_ENABLE_GRID_FEATURE (0x20800186) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_ENABLE_GRID_FEATURE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_ENABLE_GRID_FEATURE_PARAMS_MESSAGE_ID (0x86U)

typedef struct LW2080_CTRL_GPU_ENABLE_GRID_FEATURE_PARAMS {
    LwU8                         signature[LW_GRID_LICENSE_SIGNATURE_SIZE];
    LW_GRID_LICENSE_FEATURE_CODE featureCode;
} LW2080_CTRL_GPU_ENABLE_GRID_FEATURE_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_ENABLE_GRID_LICENSE
 *
 * Communicates to RM that license is successfully acquired
 *
 * licenseState
 *   set when license acquisition is successful
 *
 * licenseExpiryTimestamp
 *   license expiry in seconds since epoch time
 *
 * licenseExpiryStatus
 *   license expiry status
 *
 * Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 *   LWOS_STATUS_ERROR_ILWALID_DATA
 *   LWOS_STATUS_ERROR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_ENABLE_GRID_LICENSE (0x20800187) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_ENABLE_GRID_LICENSE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_ENABLE_GRID_LICENSE_PARAMS_MESSAGE_ID (0x87U)

typedef struct LW2080_CTRL_GPU_ENABLE_GRID_LICENSE_PARAMS {
    LwU8   signature[LW_GRID_LICENSE_SIGNATURE_SIZE];
    LwBool licenseState;
    LwU32  licenseExpiryTimestamp;
    LwU8   licenseExpiryStatus;
} LW2080_CTRL_GPU_ENABLE_GRID_LICENSE_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GPU_GET_MAX_SUPPORTED_PAGE_SIZE
 *
 * This command returns information regarding maximum page size supported
 * by GMMU on the platform on which RM is running.
 *
 * [OUT] maxSupportedPageSize
 *      - Maximum local vidmem page size supported by GMMU of a given GPU (HW)
 *        on a given platform (OS)
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_MAX_SUPPORTED_PAGE_SIZE (0x20800188) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_MAX_SUPPORTED_PAGE_SIZE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_MAX_SUPPORTED_PAGE_SIZE_PARAMS_MESSAGE_ID (0x88U)

typedef struct LW2080_CTRL_GPU_GET_MAX_SUPPORTED_PAGE_SIZE_PARAMS {
    LwU32 maxSupportedPageSize;
} LW2080_CTRL_GPU_GET_MAX_SUPPORTED_PAGE_SIZE_PARAMS;
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GPU_GET_GRID_SW_PACKAGE_IDENTIFIER
 *
 * Used to get information that identifies the gridsw package installed on the system
 *
 *   gridSwPkg
 *     A non-zero value which identifies the gridsw package installed
 *
 * Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 */
#define LW2080_CTRL_CMD_GPU_GET_GRID_SW_PACKAGE_IDENTIFIER (0x20800189) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_CMD_GPU_GET_GRID_SW_PACKAGE_IDENTIFIER_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_GPU_GET_GRID_SW_PACKAGE_IDENTIFIER_PARAMS_MESSAGE_ID (0x89U)

typedef struct LW2080_CTRL_CMD_GPU_GET_GRID_SW_PACKAGE_IDENTIFIER_PARAMS {
    LwU32 gridSwPkg;
} LW2080_CTRL_CMD_GPU_GET_GRID_SW_PACKAGE_IDENTIFIER_PARAMS;

/*
 * LW2080_CTRL_GPU_GET_NUM_MMUS_PER_GPC
 *
 * This command returns the max number of MMUs per GPC
 *
 *   gpcId [IN]
 *     Logical GPC id
 *   count [OUT]
 *     The number of MMUs per GPC
 *   grRouteInfo            
 *     This parameter specifies the routing information used to            
 *     disambiguate the target GR engine. When SMC is enabled, this            
 *     is a mandatory parameter.
 */
#define LW2080_CTRL_GPU_GET_NUM_MMUS_PER_GPC_PARAMS_MESSAGE_ID (0x8AU)

typedef struct LW2080_CTRL_GPU_GET_NUM_MMUS_PER_GPC_PARAMS {
    LwU32 gpcId;
    LwU32 count;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GPU_GET_NUM_MMUS_PER_GPC_PARAMS;

#define LW2080_CTRL_CMD_GPU_GET_NUM_MMUS_PER_GPC (0x2080018a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_NUM_MMUS_PER_GPC_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_GPU_GET_ACTIVE_PARTITION_IDS
 *
 * This command returns the GPU partition IDs for all active partitions
 * If GPU is not partitioned, the control call will return partition count as "0"
 *
 *   swizzId[OUT]
 *      - HW Partition ID associated with the active partitions
 *
 *   partitionCount[OUT]
 *      - Number of active partitions in system
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_GPU_GET_ACTIVE_PARTITION_IDS_PARAMS_MESSAGE_ID (0x8BU)

typedef struct LW2080_CTRL_GPU_GET_ACTIVE_PARTITION_IDS_PARAMS {
    LwU32 swizzId[LW2080_CTRL_GPU_MAX_PARTITION_IDS];
    LwU32 partitionCount;
} LW2080_CTRL_GPU_GET_ACTIVE_PARTITION_IDS_PARAMS;

#define LW2080_CTRL_CMD_GPU_GET_ACTIVE_PARTITION_IDS               (0x2080018b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ACTIVE_PARTITION_IDS_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_GPU_GET_GRID_UNLICENSED_STATE_MACHINE_INFO
 *
 * Query information about GRID unlicensed state machine
 *
 *   lwrrentState
 *     Current state of GRID unlicensed state machine
 *
 *   fpsValue
 *     FPS value corresponding to the current state
 *
 *   lwdaSleepInterval
 *     Amount of interval(in millisecond) for which LWCA APIs need to sleep
 *
 *   licenseExpiryTimestamp
 *     This parameter specifies the current value of license expiry in seconds since epoch time
 *
 *   licenseExpiryStatus
 *     This parameter specifies the license expiry status
 *
 * Possible status values returned are:
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GPU_GET_GRID_UNLICENSED_STATE_MACHINE_INFO (0x2080018c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GRID_UNLICENSED_STATE_MACHINE_INFO_PARAMS_MESSAGE_ID" */

// States of GRID unlicensed state machine
#define LW2080_CTRL_GPU_GRID_LICENSE_STATE_UNKNOWN                 0   // Unknown state
#define LW2080_CTRL_GPU_GRID_LICENSE_STATE_UNINITIALIZED           1   // Uninitialized state
#define LW2080_CTRL_GPU_GRID_LICENSE_STATE_UNLICENSED_UNRESTRICTED 2   // No capping
#define LW2080_CTRL_GPU_GRID_LICENSE_STATE_UNLICENSED_RESTRICTED_1 3   // Partial capping
#define LW2080_CTRL_GPU_GRID_LICENSE_STATE_UNLICENSED              4   // Full capping
#define LW2080_CTRL_GPU_GRID_LICENSE_STATE_LICENSED                5   // No capping when licensed

#define LW2080_CTRL_GPU_GRID_UNLICENSED_STATE_MACHINE_INFO_PARAMS_MESSAGE_ID (0x8LW)

typedef struct LW2080_CTRL_GPU_GRID_UNLICENSED_STATE_MACHINE_INFO_PARAMS {
    LwU32 lwrrentState;
    LwU32 fpsValue;
    LwU32 lwdaSleepInterval;
    LwU32 licenseExpiryTimestamp;
    LwU8  licenseExpiryStatus;
} LW2080_CTRL_GPU_GRID_UNLICENSED_STATE_MACHINE_INFO_PARAMS;
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_CMD_GPU_GET_PIDS
 *
 * Given a resource identifier and its type, this command returns a set of
 * process identifiers (PIDs) of processes that have instantiated this resource.
 * For example, given a class number, this command returns a list of all
 * processes with clients that have matching object allocations.
 * This is a SMC aware call and the scope of the information gets restricted
 * based on partition subscription.
 * The call enforces partition subscription if SMC is enabled, and client is not
 * a monitoring client.
 * Monitoring clients get global information without any scope based filtering.
 * Monitoring clients are also not expected to subscribe to a partition when
 * SMC is enabled.
 *
 *   idType
 *     Type of the resource identifier. See below for a list of valid types.
 *   id
 *     Resource identifier.
 *   pidTblCount
 *      Number of entries in the PID table.
 *   pidTbl
 *     Table which will contain the PIDs. Each table entry is of type LwU32.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_GPU_GET_PIDS       (0x2080018d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_PIDS_PARAMS_MESSAGE_ID" */

/* max size of pidTable */
#define LW2080_CTRL_GPU_GET_PIDS_MAX_COUNT 950

#define LW2080_CTRL_GPU_GET_PIDS_PARAMS_MESSAGE_ID (0x8DU)

typedef struct LW2080_CTRL_GPU_GET_PIDS_PARAMS {
    LwU32 idType;
    LwU32 id;
    LwU32 pidTblCount;
    LwU32 pidTbl[LW2080_CTRL_GPU_GET_PIDS_MAX_COUNT];
} LW2080_CTRL_GPU_GET_PIDS_PARAMS;

/*
 * Use class LW20_SUBDEVICE_0 with LW2080_CTRL_GPU_GET_PIDS_ID_TYPE_CLASS to query
 * PIDs with or without GPU contexts. For any other class id, PIDs only with GPU
 * contexts are returned.
 */
#define LW2080_CTRL_GPU_GET_PIDS_ID_TYPE_CLASS      (0x00000000)
#define LW2080_CTRL_GPU_GET_PIDS_ID_TYPE_VGPU_GUEST (0x00000001)

/*
 * LW2080_CTRL_SMC_SUBSCRIPTION_INFO
 *
 * This structure contains information about the SMC subscription type.
 * If MIG is enabled a valid ID is returned, it is set to PARTITIONID_ILWALID otherwise.
 *
 *  computeInstanceId
 *      This parameter returns a valid compute instance ID
 *  gpuInstanceId
 *      This parameter returns a valid GPU instance ID
 */
typedef struct LW2080_CTRL_SMC_SUBSCRIPTION_INFO {
    LwU32 computeInstanceId;
    LwU32 gpuInstanceId;
} LW2080_CTRL_SMC_SUBSCRIPTION_INFO;

/*
 * LW2080_CTRL_GPU_PID_INFO_VIDEO_MEMORY_USAGE_DATA
 *
 * This structure contains the video memory usage information.
 *
 *   memPrivate
 *     This parameter returns the amount of memory exclusively owned
 *     (i.e. private) to the client
 *   memSharedOwned
 *     This parameter returns the amount of shared memory owned by the client
 *   memSharedDuped
 *     This parameter returns the amount of shared memory duped by the client
 *   protectedMemPrivate
 *     This parameter returns the amount of protected memory exclusively owned
 *     (i.e. private) to the client whenever memory protection is enabled
 *   protectedMemSharedOwned
 *     This parameter returns the amount of shared protected memory owned by the
 *     client whenever memory protection is enabled
 *   protectedMemSharedDuped
 *     This parameter returns the amount of shared protected memory duped by the
 *     client whenever memory protection is enabled
 */
typedef struct LW2080_CTRL_GPU_PID_INFO_VIDEO_MEMORY_USAGE_DATA {
    LW_DECLARE_ALIGNED(LwU64 memPrivate, 8);
    LW_DECLARE_ALIGNED(LwU64 memSharedOwned, 8);
    LW_DECLARE_ALIGNED(LwU64 memSharedDuped, 8);
    LW_DECLARE_ALIGNED(LwU64 protectedMemPrivate, 8);
    LW_DECLARE_ALIGNED(LwU64 protectedMemSharedOwned, 8);
    LW_DECLARE_ALIGNED(LwU64 protectedMemSharedDuped, 8);
} LW2080_CTRL_GPU_PID_INFO_VIDEO_MEMORY_USAGE_DATA;

#define LW2080_CTRL_GPU_PID_INFO_INDEX_VIDEO_MEMORY_USAGE (0x00000000)

#define LW2080_CTRL_GPU_PID_INFO_INDEX_MAX                LW2080_CTRL_GPU_PID_INFO_INDEX_VIDEO_MEMORY_USAGE

typedef union LW2080_CTRL_GPU_PID_INFO_DATA {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_PID_INFO_VIDEO_MEMORY_USAGE_DATA vidMemUsage, 8);
} LW2080_CTRL_GPU_PID_INFO_DATA;


/*
 * LW2080_CTRL_GPU_PID_INFO
 *
 * This structure contains the per pid information. Each type of information
 * retrievable via LW2080_CTRL_CMD_GET_PID_INFO is assigned a unique index
 * below. In addition the process for which the lookup is for is also defined.
 * This is a SMC aware call and the scope of the information gets restricted
 * based on partition subscription.
 * The call enforces partition subscription if SMC is enabled, and client is not
 * a monitoring client.
 * Monitoring clients get global information without any scope based filtering.
 * Monitoring clients are also not expected to subscribe to a partition when
 * SMC is enabled.
 *
 *   pid
 *     This parameter specifies the PID of the process for which information is
 *     to be queried.
 *   index
 *     This parameter specifies the type of information being queried for the
 *     process of interest.
 *   result
 *     This parameter returns the result of the instruction's exelwtion.
 *   data
 *     This parameter returns the data corresponding to the information which is
 *     being queried.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *
 * Valid PID information indices are:
 *
 *   LW2080_CTRL_GPU_PID_INFO_INDEX_VIDEO_MEMORY_USAGE
 *     This index is used to request the amount of video memory on this GPU
 *     allocated to the process.
 */
typedef struct LW2080_CTRL_GPU_PID_INFO {
    LwU32                             pid;
    LwU32                             index;
    LwU32                             result;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_PID_INFO_DATA data, 8);
    LW2080_CTRL_SMC_SUBSCRIPTION_INFO smcSubscription;
} LW2080_CTRL_GPU_PID_INFO;

/*
 * LW2080_CTRL_CMD_GPU_GET_PID_INFO
 *
 * This command allows querying per-process information from the RM. Clients
 * request information by specifying a unique informational index and the
 * Process ID of the process in question. The result is set to indicate success
 * and the information queried (if available) is returned in the data parameter.
 *
 *   pidInfoListCount
 *     The number of valid entries in the pidInfoList array.
 *   pidInfoList
 *     An array of LW2080_CTRL_GPU_PID_INFO of maximum length
 *     LW2080_CTRL_GPU_GET_PID_INFO_MAX_COUNT.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_GPU_GET_PID_INFO       (0x2080018e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_PID_INFO_PARAMS_MESSAGE_ID" */

/* max size of pidInfoList */
#define LW2080_CTRL_GPU_GET_PID_INFO_MAX_COUNT 200

#define LW2080_CTRL_GPU_GET_PID_INFO_PARAMS_MESSAGE_ID (0x8EU)

typedef struct LW2080_CTRL_GPU_GET_PID_INFO_PARAMS {
    LwU32 pidInfoListCount;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_PID_INFO pidInfoList[LW2080_CTRL_GPU_GET_PID_INFO_MAX_COUNT], 8);
} LW2080_CTRL_GPU_GET_PID_INFO_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*!
 * LW2080_CTRL_CMD_GPU_INIT_VF_BARS
 *
 * @brief Initialize VF BAR1/BAR2 in SRIOV heavy mode
 *
 *   bIsInit
 *     Initialize VF BAR1/BAR2 if the value is true,
 *     otherwise teardown VF BAR1/BAR2
 *
 * Possible status values returned are:
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 *
 */

#define LW2080_CTRL_CMD_GPU_INIT_VF_BARS (0x20800190) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_CMD_GPU_INIT_VF_BARS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_GPU_INIT_VF_BARS_PARAMS_MESSAGE_ID (0x90U)

typedef struct LW2080_CTRL_CMD_GPU_INIT_VF_BARS_PARAMS {
    LwBool bIsInit;
} LW2080_CTRL_CMD_GPU_INIT_VF_BARS_PARAMS;

/*!
 * LW2080_CTRL_CMD_GPU_SETUP_VF_FAULT_BUFFER
 *
 * @brief Enable/Disable VF fault buffer in SRIOV heavy mode
 *
 *   index
 *     Index of the fault buffer
 *
 *   cpuCacheAttrib
 *     Cache Attrib of the fault buffer
 *
 *   size
 *     Size of the fault buffer
 *
 *   pageCount
 *     Number of pages in the fault buffer
 *
 *   flags
 *     LW2080_CTRL_GPU_SETUP_VF_FAULT_BUFFER_PARAMS_FLAGS_* flags
 *
 *   isEnable
 *     If True then enable fault buffer, otherwise disable fault buffer
 *
 *   address
 *     Guest physical address of the fault buffer.
 *
 * Possible status values returned are:
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 *
 */

#define LW2080_CTRL_CMD_GPU_SETUP_VF_FAULT_BUFFER                                 (0x20800191) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_CMD_GPU_SETUP_VF_FAULT_BUFFER_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_SETUP_VF_FAULT_BUFFER_PARAMS_FLAGS_APERTURE                0:0
#define LW2080_CTRL_GPU_SETUP_VF_FAULT_BUFFER_PARAMS_FLAGS_APERTURE_VIDEO_MEMORY  (0x00000000)
#define LW2080_CTRL_GPU_SETUP_VF_FAULT_BUFFER_PARAMS_FLAGS_APERTURE_SYSTEM_MEMORY (0x00000001)

#define LW2080_CTRL_CMD_GPU_SETUP_VF_FAULT_BUFFER_PARAMS_MESSAGE_ID (0x91U)

typedef struct LW2080_CTRL_CMD_GPU_SETUP_VF_FAULT_BUFFER_PARAMS {
    LwU32  index;
    LwU32  cpuCacheAttrib;
    LW_DECLARE_ALIGNED(LwU64 size, 8);
    LW_DECLARE_ALIGNED(LwU64 pageCount, 8);
    LwU32  flags;
    LwBool isEnable;
    LW_DECLARE_ALIGNED(LwU64 address, 8);
} LW2080_CTRL_CMD_GPU_SETUP_VF_FAULT_BUFFER_PARAMS;


/*!
 * LW2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT
 *
 * @brief Handle VF PRI faults
 *
 *   faultType
 *     BAR1, BAR2, PHYSICAL or UNBOUND_INSTANCE
 *
 * Possible status values returned are:
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 *
 */

#define LW2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT                       (0x20800192) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_TYPE_ILWALID          0
#define LW2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_TYPE_BAR1             1
#define LW2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_TYPE_BAR2             2
#define LW2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_TYPE_PHYSICAL         3
#define LW2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_TYPE_UNBOUND_INSTANCE 4

#define LW2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_PARAMS_MESSAGE_ID (0x92U)

typedef struct LW2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_PARAMS {
    LwU32 faultType;
} LW2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_PARAMS;

/*!
 * LW2080_CTRL_CMD_GPU_GET_VMMU_SPA_BITMASK
 *
 * @brief Get the VMMU SPA bitmask for a given gfid
 *
 *   gfid         Gfid
 *   numDwords    Number of dwords constituting the bitmask
 *   vmmuBitmask  VMMU bitmask for the given gfid
 *
 * Possible status values returned are:
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 *
 */

#define LW2080_CTRL_CMD_GPU_GET_VMMU_SPA_BITMASK             (0x20800193) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_VMMU_SPA_BITMASK_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_PER_GFID_VMMU_BITMASK_SIZE_IN_DWORDS 100

#define LW2080_CTRL_GPU_GET_VMMU_SPA_BITMASK_PARAMS_MESSAGE_ID (0x93U)

typedef struct LW2080_CTRL_GPU_GET_VMMU_SPA_BITMASK_PARAMS {
    LwU32 gfid;
    LwU32 numDwords;
    LwU32 vmmuBitmask[LW2080_CTRL_GPU_PER_GFID_VMMU_BITMASK_SIZE_IN_DWORDS];
} LW2080_CTRL_GPU_GET_VMMU_SPA_BITMASK_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_SET_COMPUTE_POLICY_CONFIG
 *
 * This is a privileged control command used to set the compute policy config for a GPU.
 *
 * For documentation of parameters, see @ref LW2080_CTRL_GPU_SET_COMPUTE_POLICY_CONFIG_PARAMS. 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW2080_CTRL_CMD_GPU_SET_COMPUTE_POLICY_CONFIG (0x20800194) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_SET_COMPUTE_POLICY_CONFIG_PARAMS_MESSAGE_ID" */

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


/*!
 * Compute policy types to be specified by callers to set a config.
 *
 * _TIMESLICE
 *     Set the timeslice config for the requested GPU.
 *     Check @ref LW2080_CTRL_GPU_COMPUTE_POLICY_CONFIG_DATA_TIMESLICE for
 *     permissible timeslice values.
 */
#define LW2080_CTRL_GPU_COMPUTE_POLICY_TIMESLICE      0
#define LW2080_CTRL_GPU_COMPUTE_POLICY_MAX            1

/*!
 * Enum consisting of permissible timeslice options that can configured
 * for a GPU. These can be queried by compute clients and the exact
 * timeslice values can be chosen appropriately as per GPU support
 */
typedef enum LW2080_CTRL_GPU_COMPUTE_POLICY_CONFIG_DATA_TIMESLICE {
    LW2080_CTRL_CMD_GPU_COMPUTE_TIMESLICE_DEFAULT = 0,
    LW2080_CTRL_CMD_GPU_COMPUTE_TIMESLICE_SHORT = 1,
    LW2080_CTRL_CMD_GPU_COMPUTE_TIMESLICE_MEDIUM = 2,
    LW2080_CTRL_CMD_GPU_COMPUTE_TIMESLICE_LONG = 3,
    LW2080_CTRL_CMD_GPU_COMPUTE_TIMESLICE_MAX = 4,
} LW2080_CTRL_GPU_COMPUTE_POLICY_CONFIG_DATA_TIMESLICE;
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)




typedef struct LW2080_CTRL_GPU_COMPUTE_POLICY_CONFIG {
    /*!
     * LW2080_CTRL_GPU_COMPUTE_POLICY_<xyz>
     */
    LwU32 type;

    /*!
     * Union of type-specific data
     */
    union {
        LW2080_CTRL_GPU_COMPUTE_POLICY_CONFIG_DATA_TIMESLICE timeslice;
    } data;
} LW2080_CTRL_GPU_COMPUTE_POLICY_CONFIG;

#define LW2080_CTRL_GPU_SET_COMPUTE_POLICY_CONFIG_PARAMS_MESSAGE_ID (0x94U)

typedef struct LW2080_CTRL_GPU_SET_COMPUTE_POLICY_CONFIG_PARAMS {
    LW2080_CTRL_GPU_COMPUTE_POLICY_CONFIG config;
} LW2080_CTRL_GPU_SET_COMPUTE_POLICY_CONFIG_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_COMPUTE_POLICY_CONFIG
 *
 * This command retrieves all compute policies configs for the associated gpu.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW2080_CTRL_CMD_GPU_GET_COMPUTE_POLICY_CONFIG  (0x20800195) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_COMPUTE_POLICY_CONFIG_PARAMS_MESSAGE_ID" */

/*!
 * This define limits the max number of policy configs that can be handled by
 * LW2080_CTRL_CMD_GPU_GET_COMPUTE_POLICY_CONFIG command.
 *
 * @note Needs to be in sync (greater or equal) to LW2080_CTRL_GPU_COMPUTE_POLICY_MAX.
 */

#define LW2080_CTRL_GPU_COMPUTE_POLICY_CONFIG_LIST_MAX 32

#define LW2080_CTRL_GPU_GET_COMPUTE_POLICY_CONFIG_PARAMS_MESSAGE_ID (0x95U)

typedef struct LW2080_CTRL_GPU_GET_COMPUTE_POLICY_CONFIG_PARAMS {
    LwU32                                 numConfigs;

    /*!
     * C form:
     * LW2080_CTRL_GPU_COMPUTE_POLICY_CONFIG configList[LW2080_CTRL_GPU_COMPUTE_POLICY_CONFIG_LIST_MAX];
     */
    LW2080_CTRL_GPU_COMPUTE_POLICY_CONFIG configList[LW2080_CTRL_GPU_COMPUTE_POLICY_CONFIG_LIST_MAX];
} LW2080_CTRL_GPU_GET_COMPUTE_POLICY_CONFIG_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_GFID
 *
 * This command returns the GFID (GPU Function ID) for a given SR-IOV 
 * Virtual Function (VF) of the physical GPU.
 *
 *   domain [IN]
 *     This field specifies the respective domain of the PCI device.
 *   bus [IN]
 *     This field specifies the bus id for a given VF.
 *   device [IN]
 *     This field specifies the device id for a given VF.
 *   func [IN]
 *     This field specifies the function id for a given VF.
 *   gfid[OUT]
 *      - This field returns GFID for a given VF BDF.
 *   gfidMask[OUT]
 *      - This field returns GFID mask value.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */

#define LW2080_CTRL_CMD_GPU_GET_GFID (0x20800196) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_GFID_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_GFID_PARAMS_MESSAGE_ID (0x96U)

typedef struct LW2080_CTRL_GPU_GET_GFID_PARAMS {
    LwU32 domain;
    LwU8  bus;
    LwU8  device;
    LwU8  func;
    LwU32 gfid;
    LwU32 gfidMask;
} LW2080_CTRL_GPU_GET_GFID_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_UPDATE_GFID_P2P_CAPABILITY
 *
 * This command informs the GPU driver that the GPU partition associated with
 * a given GFID has been activated or will be deactivated.
 *
 *   gfid[IN]
 *      - The GPU function identifier for a given VF BDF
 *   bEnable [IN]
 *      - Set to LW_TRUE if the GPU partition has been activated. 
 *      - Set to LW_FALSE if the GPU partition will be deactivated. 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */

#define LW2080_CTRL_CMD_GPU_UPDATE_GFID_P2P_CAPABILITY (0x20800197) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_CMD_GPU_UPDATE_GFID_P2P_CAPABILITY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_GPU_UPDATE_GFID_P2P_CAPABILITY_PARAMS_MESSAGE_ID (0x97U)

typedef struct LW2080_CTRL_CMD_GPU_UPDATE_GFID_P2P_CAPABILITY_PARAMS {
    LwU32  gfid;
    LwBool bEnable;
} LW2080_CTRL_CMD_GPU_UPDATE_GFID_P2P_CAPABILITY_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*!
 * LW2080_CTRL_CMD_GPU_VALIDATE_MEM_MAP_REQUEST
 *
 * @brief Validate the address range for memory map request by comparing the
 *        user supplied address range with GPU BAR0/BAR1 range.
 *
 * @param[in]   addressStart    Start address for memory map request
 * @param[in]   addressLength   Length for for memory map request
 * @param[out]  protection      LW_PROTECT_READ_WRITE, if both read/write is allowed
 *                              LW_PROTECT_READABLE, if only read is allowed
 *
 * Possible status values returned are:
 *     LW_OK
 *     LW_ERR_PROTECTION_FAULT
 *
 */
#define LW2080_CTRL_CMD_GPU_VALIDATE_MEM_MAP_REQUEST (0x20800198) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_VALIDATE_MEM_MAP_REQUEST_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_VALIDATE_MEM_MAP_REQUEST_PARAMS_MESSAGE_ID (0x98U)

typedef struct LW2080_CTRL_GPU_VALIDATE_MEM_MAP_REQUEST_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 addressStart, 8);
    LW_DECLARE_ALIGNED(LwU64 addressLength, 8);
    LwU32 protection;
} LW2080_CTRL_GPU_VALIDATE_MEM_MAP_REQUEST_PARAMS;
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GPU_SET_EGM_GPA_FABRIC_BASE_ADDR
 *
 * @brief This command is similar to LW2080_CTRL_CMD_GPU_SET_FABRIC_BASE_ADDR
 * but will be used to set the EGM fabric base addr associated with the gpu.
 * Note: For EGM FLA, we will be making use of the existing control call i.e
 * LW2080_CTRL_CMD_FLA_RANGE
 *
 */
#define LW2080_CTRL_CMD_GPU_SET_EGM_GPA_FABRIC_BASE_ADDR (0x20800199) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_SET_EGM_GPA_FABRIC_BASE_ADDR_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_SET_EGM_GPA_FABRIC_BASE_ADDR_PARAMS_MESSAGE_ID (0x99U)

typedef struct LW2080_CTRL_GPU_SET_EGM_GPA_FABRIC_BASE_ADDR_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 egmGpaFabricBaseAddr, 8);
} LW2080_CTRL_GPU_SET_EGM_GPA_FABRIC_BASE_ADDR_PARAMS;
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_CMD_GPU_GET_ENGINE_LOAD_TIMES
 *
 * This command is used to retrieve the load time (latency) of each engine.
 *  
 *   engineCount
 *     This field specifies the number of entries of the following
 *     three arrays.
 *
 *   engineList[LW2080_GPU_MAX_ENGINE_OBJECTS]
 *     An array of LwU32 which stores each engine's descriptor.
 *
 *   engineStateLoadTime[LW2080_GPU_MAX_ENGINE_OBJECTS]
 *     A array of LwU64 which stores each engine's load time.
 *
 *   engineIsInit[LW2080_GPU_MAX_ENGINE_OBJECTS]
 *     A array of LwBool which stores each engine's initialization status.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_GPU_GET_ENGINE_LOAD_TIMES (0x2080019b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ENGINE_LOAD_TIMES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_MAX_ENGINE_OBJECTS        0x90

#define LW2080_CTRL_GPU_GET_ENGINE_LOAD_TIMES_PARAMS_MESSAGE_ID (0x9BU)

typedef struct LW2080_CTRL_GPU_GET_ENGINE_LOAD_TIMES_PARAMS {
    LwU32  engineCount;
    LwU32  engineList[LW2080_CTRL_GPU_MAX_ENGINE_OBJECTS];
    LW_DECLARE_ALIGNED(LwU64 engineStateLoadTime[LW2080_CTRL_GPU_MAX_ENGINE_OBJECTS], 8);
    LwBool engineIsInit[LW2080_CTRL_GPU_MAX_ENGINE_OBJECTS];
} LW2080_CTRL_GPU_GET_ENGINE_LOAD_TIMES_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_ID_NAME_MAPPING
 *
 * This command is used to retrieve the mapping of engine ID and engine Name.
 * 
 *   engineCount
 *     This field specifies the size of the mapping.
 *   
 *   engineID
 *     An array of LwU32 which stores each engine's descriptor.
 *
 *   engineName
 *     An array of char[100] which stores each engine's name.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_GPU_GET_ID_NAME_MAPPING (0x2080019c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_ID_NAME_MAPPING_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_GET_ID_NAME_MAPPING_PARAMS_MESSAGE_ID (0x9LW)

typedef struct LW2080_CTRL_GPU_GET_ID_NAME_MAPPING_PARAMS {
    LwU32 engineCount;
    LwU32 engineID[LW2080_CTRL_GPU_MAX_ENGINE_OBJECTS];
    char  engineName[LW2080_CTRL_GPU_MAX_ENGINE_OBJECTS][100];
} LW2080_CTRL_GPU_GET_ID_NAME_MAPPING_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_EXEC_REG_OPS_NOPTRS
 *
 * Same as above LW2080_CTRL_CMD_GPU_EXEC_REG_OPS except that this CTRL CMD will
 * not allow any embedded pointers. The regOps array is inlined as part of the
 * struct. 
 * NOTE: This intended for gsp plugin only as it may override regOp access
 *       restrictions
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_GPU_EXEC_REG_OPS_NOPTRS (0x2080019d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_EXEC_REG_OPS_NOPTRS_PARAMS_MESSAGE_ID" */

/* setting this to 100 keeps it right below 4k in size */
#define LW2080_CTRL_REG_OPS_ARRAY_MAX           100
#define LW2080_CTRL_GPU_EXEC_REG_OPS_NOPTRS_PARAMS_MESSAGE_ID (0x9DU)

typedef struct LW2080_CTRL_GPU_EXEC_REG_OPS_NOPTRS_PARAMS {
    LwHandle               hClientTarget;
    LwHandle               hChannelTarget;
    LwU32                  bNonTransactional;
    LwU32                  reserved00[2];
    LwU32                  regOpCount;
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
    LW2080_CTRL_GPU_REG_OP regOps[LW2080_CTRL_REG_OPS_ARRAY_MAX];
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GPU_EXEC_REG_OPS_NOPTRS_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#define LW2080_CTRL_GPU_SKYLINE_INFO_MAX_SKYLINES            8
#define LW2080_CTRL_GPU_SKYLINE_INFO_MAX_NON_SINGLETON_VGPCS 8
/*!
 * LW2080_CTRL_GPU_SKYLINE_INFO
 * skylineVgpcSize[OUT]
 *      - TPC count of non-singleton VGPCs
 * singletolwgpcMask[OUT]
 *      - Mask of active Singletons
 * maxInstances[OUT]
 *      - Max allowed instances of this skyline conlwrrently on a GPU
 * computeSizeFlag
 *      - One of LW2080_CTRL_GPU_PARTITION_FLAG_COMPUTE_SIZE_* flags which is associated with this skyline
 */
typedef struct LW2080_CTRL_GPU_SKYLINE_INFO {
    LwU8  skylineVgpcSize[LW2080_CTRL_GPU_SKYLINE_INFO_MAX_NON_SINGLETON_VGPCS];
    LwU32 singletolwgpcMask;
    LwU32 maxInstances;
    LwU32 computeSizeFlag;
} LW2080_CTRL_GPU_SKYLINE_INFO;

/*!
 * LW2080_CTRL_GPU_GET_SKYLINE_INFO_PARAMS
 * skylineTable[OUT]
 *      - TPC count of non-singleton VGPCs
 *      - Mask of singleton vGPC IDs active
 *      - Max Instances of this skyline possible conlwrrently
 *      - Associated compute size with the indexed skyline
 * validEntries[OUT]
 *      - Number of entries which contain valid info in skylineInfo
 */
#define LW2080_CTRL_GPU_GET_SKYLINE_INFO_PARAMS_MESSAGE_ID (0x9FU)

typedef struct LW2080_CTRL_GPU_GET_SKYLINE_INFO_PARAMS {
    LW2080_CTRL_GPU_SKYLINE_INFO skylineTable[LW2080_CTRL_GPU_SKYLINE_INFO_MAX_SKYLINES];
    LwU32                        validEntries;
} LW2080_CTRL_GPU_GET_SKYLINE_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_GET_SKYLINE_INFO
 *
 * Retrieves skyline information about the GPU. Params are sized to lwrrently known max
 * values, but will need to be modified in the future should that change.
 */
#define LW2080_CTRL_CMD_GPU_GET_SKYLINE_INFO (0x2080019f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GPU_GET_SKYLINE_INFO_PARAMS_MESSAGE_ID" */

/*!
 * LW2080_CTRL_GPU_P2P_PEER_CAPS_PEER_INFO
 *
 * [in/out] gpuId
 *   GPU ID for which the capabilities are queried.
 *   For the LW2080_CTRL_CMD_GET_P2P_CAPS control:
 *     If bAllCaps == LW_TRUE, this parameter is an out parameter and equals to
 *     the GPU ID of an attached GPU.
 *     If bAllCaps == LW_FALSE, this parameter is an in parameter and the requester
 *     should set it to the ID of the GPU that needs to be queried from.
 * [out] p2pCaps
 *   Peer to peer capabilities discovered between the GPUs.
 *   See LW0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_V2 for the list of valid values.
 * [out] p2pOptimalReadCEs
 *   Mask of CEs to use for p2p reads over Lwlink.
 * [out] p2pOptimalWriteCEs
 *   Mask of CEs to use for p2p writes over Lwlink.
 * [out] p2pCapsStatus
 *   Status of all supported p2p capabilities.
 *   See LW0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_V2 for the list of valid values.
 * [out] busPeerId
 *   Bus peer ID. For an invalid or a non-existent peer this field
 *   has the value LW0000_CTRL_SYSTEM_GET_P2P_CAPS_ILWALID_PEER.
 */
typedef struct LW2080_CTRL_GPU_P2P_PEER_CAPS_PEER_INFO {
    LwU32 gpuId;
    LwU32 p2pCaps;
    LwU32 p2pOptimalReadCEs;
    LwU32 p2pOptimalWriteCEs;
    LwU8  p2pCapsStatus[LW0000_CTRL_P2P_CAPS_INDEX_TABLE_SIZE];
    LwU32 busPeerId;
} LW2080_CTRL_GPU_P2P_PEER_CAPS_PEER_INFO;

/*!
 * LW2080_CTRL_CMD_GET_P2P_CAPS
 *
 * Returns peer to peer capabilities present between GPUs.
 * The caller must either specify bAllCaps to query the capabilities for
 * all the attached GPUs or they must pass a valid list of GPU IDs.
 *
 *   [in] bAllCaps
 *     Set to LW_TRUE to query the capabilities for all the attached GPUs.
 *     Set to LW_FALSE and specify peerGpuCount and peerGpuCaps[].gpuId
 *     to retrieve the capabilities only for the specified GPUs.
 *   [in/out] peerGpuCount
 *     The number of the peerGpuCaps entries.
 *     If bAllCaps == LW_TRUE, this parameter is an out parameter and equals to
 *     the total number of the attached GPUs.
 *     If bAllCaps == LW_FALSE, this parameter is an in parameter and the requester
 *     should set it to the number of the peerGpuCaps entries.
 *   [in/out] peerGpuCaps
 *     The array of LW2080_CTRL_GPU_P2P_PEER_CAPS_PEER_INFO entries, describing
 *     the peer to peer capabilities of the GPUs.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT - Invalid peerGpuCount
 *   LW_ERR_OBJECT_NOT_FOUND - Invalid peerGpuCaps[].gpuId
 */
#define LW2080_CTRL_CMD_GET_P2P_CAPS (0x208001a0) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | LW2080_CTRL_GET_P2P_CAPS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GET_P2P_CAPS_PARAMS_MESSAGE_ID (0xA0U)

typedef struct LW2080_CTRL_GET_P2P_CAPS_PARAMS {
    LwBool                                  bAllCaps;
    LwU32                                   peerGpuCount;
    LW2080_CTRL_GPU_P2P_PEER_CAPS_PEER_INFO peerGpuCaps[LW0000_CTRL_SYSTEM_MAX_ATTACHED_GPUS];
} LW2080_CTRL_GET_P2P_CAPS_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/* _ctrl2080gpu_h_ */
