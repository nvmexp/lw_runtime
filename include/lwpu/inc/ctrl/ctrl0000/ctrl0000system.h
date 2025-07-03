/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0000/ctrl0000system.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrl0000/ctrl0000base.h"

/* LW01_ROOT (client) system control commands and parameters */

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_FEATURES
 *
 * This command returns a mask of supported features for the SYSTEM category
 * of the 0000 class.
 *
 *     Valid features include:
 *         
 *       LW0000_CTRL_GET_FEATURES_SLI
 *         When this bit is set, SLI is supported.
 *       LW0000_CTRL_GET_FEATURES_UEFI
 *         When this bit is set, it is a UEFI system.
 *       LW0000_CTRL_SYSTEM_GET_FEATURES_IS_EFI_INIT
 *         When this bit is set, EFI has initialized core channel 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_FEATURES (0x1f0) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_FEATURES_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_FEATURES_PARAMS_MESSAGE_ID (0xF0U)

typedef struct LW0000_CTRL_SYSTEM_GET_FEATURES_PARAMS {
    LwU32 featuresMask;
} LW0000_CTRL_SYSTEM_GET_FEATURES_PARAMS;



/* Valid feature values */
#define LW0000_CTRL_SYSTEM_GET_FEATURES_SLI                                 0:0
#define LW0000_CTRL_SYSTEM_GET_FEATURES_SLI_FALSE         (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_FEATURES_SLI_TRUE          (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_FEATURES_UEFI                                1:1
#define LW0000_CTRL_SYSTEM_GET_FEATURES_UEFI_FALSE        (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_FEATURES_UEFI_TRUE         (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_FEATURES_IS_EFI_INIT                         2:2
#define LW0000_CTRL_SYSTEM_GET_FEATURES_IS_EFI_INIT_FALSE (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_FEATURES_IS_EFI_INIT_TRUE  (0x00000001)
/*
 * LW0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION
 *
 * This command returns the current driver information.
 * The first time this is called the size of strings is
 * set with the greater of LW_BUILD_BRANCH_VERSION and
 * LW_DISPLAY_DRIVER_TITLE. The client then allocates memory
 * of size sizeOfStrings for pVersionBuffer and pTitleBuffer
 * and calls the command again to receive driver info.
 *
 *   sizeOfStrings
 *       This field returns the size in bytes of the pVersionBuffer and
 *       pTitleBuffer strings.
 *   pDriverVersionBuffer
 *       This field returns the version (LW_VERSION_STRING).
 *   pVersionBuffer
 *       This field returns the version (LW_BUILD_BRANCH_VERSION).
 *   pTitleBuffer
 *       This field returns the title (LW_DISPLAY_DRIVER_TITLE).
 *   changelistNumber
 *       This field returns the changelist value (LW_BUILD_CHANGELIST_NUM).
 *   officialChangelistNumber
 *       This field returns the last official changelist value
 *       (LW_LAST_OFFICIAL_CHANGELIST_NUM).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION          (0x101) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_PARAMS {
    LwU32 sizeOfStrings;
    LW_DECLARE_ALIGNED(LwP64 pDriverVersionBuffer, 8);
    LW_DECLARE_ALIGNED(LwP64 pVersionBuffer, 8);
    LW_DECLARE_ALIGNED(LwP64 pTitleBuffer, 8);
    LwU32 changelistNumber;
    LwU32 officialChangelistNumber;
} LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_PARAMS;

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_CPU_INFO
 *
 * This command returns system CPU information.
 *
 *   type
 *     This field returns the processor type.
 *     Legal processor types include:
 *       Intel processors:
 *         P55       : P55C - MMX
 *         P6        : PPro
 *         P2        : PentiumII
 *         P2XC      : Xeon & Celeron
 *         CELA      : Celeron-A
 *         P3        : Pentium-III
 *         P3_INTL2  : Pentium-III w/integrated L2 (fullspeed, on die, 256K)
 *         P4        : Pentium 4
 *         CORE2     : Core2 Duo Conroe
 *       AMD processors
 *         K62       : K6-2 w/ 3DNow
 *       IDT/Centaur processors
 *         C6        : WinChip C6
 *         C62       : WinChip 2 w/ 3DNow
 *       Cyrix processors
 *         GX        : MediaGX
 *         M1        : 6x86
 *         M2        : M2
 *         MGX       : MediaGX w/ MMX
 *       Transmeta processors
 *         TM_CRUSOE : Transmeta Crusoe(tm)
 *       PowerPC processors
 *         PPC603    : PowerPC 603
 *         PPC604    : PowerPC 604
 *         PPC750    : PowerPC 750
 *
 *   capabilities
 *     This field returns the capabilities of the processor.
 *     Legal processor capabilities include:
 *       MMX                 : supports MMX
 *       SSE                 : supports SSE
 *       3DNOW               : supports 3DNow
 *       SSE2                : supports SSE2
 *       SFENCE              : supports SFENCE
 *       WRITE_COMBINING     : supports write-combining
 *       ALTIVEC             : supports ALTIVEC
 *       PUT_NEEDS_IO        : requires OUT inst w/PUT updates
 *       NEEDS_WC_WORKAROUND : requires workaround for P4 write-combining bug
 *       3DNOW_EXT           : supports 3DNow Extensions
 *       MMX_EXT             : supports MMX Extensions
 *       CMOV                : supports CMOV
 *       CLFLUSH             : supports CLFLUSH
 *       SSE3                : supports SSE3
 *       NEEDS_WAR_124888    : requires write to GPU while spinning on
 *                           : GPU value
 *       HT                  : support hyper-threading
 *   clock
 *     This field returns the processor speed in MHz.
 *   L1DataCacheSize
 *     This field returns the level 1 data (or unified) cache size
 *     in kilobytes.
 *   L2DataCacheSize
 *     This field returns the level 2 data (or unified) cache size
 *     in kilobytes.
 *   dataCacheLineSize
 *     This field returns the bytes per line in the level 1 data cache.
 *   numLogicalCpus
 *     This field returns the number of logical processors.  On Intel x86
 *     systems that support it, this value will incorporate the current state
 *     of HyperThreading.
 *   numPhysicalCpus
 *     This field returns the number of physical processors.
 *   name
 *     This field returns the CPU name in ASCII string format.
 *   family
 *     Vendor defined Family and Extended Family combined
 *   model
 *     Vendor defined Model and Extended Model combined
 *   stepping
 *     Silicon stepping
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_CPU_INFO (0x102) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_CPU_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_CPU_INFO_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0000_CTRL_SYSTEM_GET_CPU_INFO_PARAMS {
    LwU32 type;                               /* processor type        */
    LwU32 capabilities;                       /* processor caps        */
    LwU32 clock;                              /* processor speed (MHz) */
    LwU32 L1DataCacheSize;                    /* L1 dcache size (KB)   */
    LwU32 L2DataCacheSize;                    /* L2 dcache size (KB)   */
    LwU32 dataCacheLineSize;                  /* L1 dcache bytes/line  */
    LwU32 numLogicalCpus;                     /* logial processor cnt  */
    LwU32 numPhysicalCpus;                    /* physical processor cnt*/
    LwU8  name[52];                           /* embedded cpu name     */
    LwU32 family;                             /* Vendor defined Family and Extended Family combined */
    LwU32 model;                              /* Vendor defined Model and Extended Model combined   */
    LwU8  stepping;                           /* Silicon stepping      */
    LwU32 coresOnDie;                         /* cpu cores per die     */
} LW0000_CTRL_SYSTEM_GET_CPU_INFO_PARAMS;

/*  processor type values */
#define LW0000_CTRL_SYSTEM_CPU_TYPE_UNKNOWN            (0x00000000)
/* Intel types */
#define LW0000_CTRL_SYSTEM_CPU_TYPE_P5                 (0x00000001)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_P55                (0x00000002)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_P6                 (0x00000003)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_P2                 (0x00000004)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_P2XC               (0x00000005)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_CELA               (0x00000006)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_P3                 (0x00000007)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_P3_INTL2           (0x00000008)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_P4                 (0x00000009)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_CORE2              (0x00000010)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_CELN_M16H          (0x00000011)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_CORE2_EXTRM        (0x00000012)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_ATOM               (0x00000013)
/* AMD types */
#define LW0000_CTRL_SYSTEM_CPU_TYPE_K5                 (0x00000030)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_K6                 (0x00000031)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_K62                (0x00000032)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_K63                (0x00000033)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_K7                 (0x00000034)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_K8                 (0x00000035)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_K10                (0x00000036)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_K11                (0x00000037)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_RYZEN              (0x00000038)
/* IDT/Centaur types */
#define LW0000_CTRL_SYSTEM_CPU_TYPE_C6                 (0x00000060)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_C62                (0x00000061)
/* Cyrix types */
#define LW0000_CTRL_SYSTEM_CPU_TYPE_GX                 (0x00000070)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_M1                 (0x00000071)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_M2                 (0x00000072)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_MGX                (0x00000073)
/* Transmeta types  */
#define LW0000_CTRL_SYSTEM_CPU_TYPE_TM_CRUSOE          (0x00000080)
/* IBM types */
#define LW0000_CTRL_SYSTEM_CPU_TYPE_PPC603             (0x00000090)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_PPC604             (0x00000091)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_PPC750             (0x00000092)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_POWERN             (0x00000093)
/* Unknown ARM architecture CPU type */
#define LW0000_CTRL_SYSTEM_CPU_TYPE_ARM_UNKNOWN        (0xA0000000)
/* ARM Ltd types */
#define LW0000_CTRL_SYSTEM_CPU_TYPE_ARM_A9             (0xA0000009)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_ARM_A15            (0xA000000F)
/* LWPU types */
#define LW0000_CTRL_SYSTEM_CPU_TYPE_LW_DELWER_1_0      (0xA0001000)
#define LW0000_CTRL_SYSTEM_CPU_TYPE_LW_DELWER_2_0      (0xA0002000)

/* Generic types */
#define LW0000_CTRL_SYSTEM_CPU_TYPE_ARMV8A_GENERIC     (0xA00FF000)

/* processor capabilities */
#define LW0000_CTRL_SYSTEM_CPU_CAP_MMX                 (0x00000001)
#define LW0000_CTRL_SYSTEM_CPU_CAP_SSE                 (0x00000002)
#define LW0000_CTRL_SYSTEM_CPU_CAP_3DNOW               (0x00000004)
#define LW0000_CTRL_SYSTEM_CPU_CAP_SSE2                (0x00000008)
#define LW0000_CTRL_SYSTEM_CPU_CAP_SFENCE              (0x00000010)
#define LW0000_CTRL_SYSTEM_CPU_CAP_WRITE_COMBINING     (0x00000020)
#define LW0000_CTRL_SYSTEM_CPU_CAP_ALTIVEC             (0x00000040)
#define LW0000_CTRL_SYSTEM_CPU_CAP_PUT_NEEDS_IO        (0x00000080)
#define LW0000_CTRL_SYSTEM_CPU_CAP_NEEDS_WC_WORKAROUND (0x00000100)
#define LW0000_CTRL_SYSTEM_CPU_CAP_3DNOW_EXT           (0x00000200)
#define LW0000_CTRL_SYSTEM_CPU_CAP_MMX_EXT             (0x00000400)
#define LW0000_CTRL_SYSTEM_CPU_CAP_CMOV                (0x00000800)
#define LW0000_CTRL_SYSTEM_CPU_CAP_CLFLUSH             (0x00001000)
#define LW0000_CTRL_SYSTEM_CPU_CAP_NEEDS_WAR_190854    (0x00002000) /* deprecated */
#define LW0000_CTRL_SYSTEM_CPU_CAP_SSE3                (0x00004000)
#define LW0000_CTRL_SYSTEM_CPU_CAP_NEEDS_WAR_124888    (0x00008000)
#define LW0000_CTRL_SYSTEM_CPU_CAP_HT_CAPABLE          (0x00010000)
#define LW0000_CTRL_SYSTEM_CPU_CAP_SSE41               (0x00020000)
#define LW0000_CTRL_SYSTEM_CPU_CAP_SSE42               (0x00040000)
#define LW0000_CTRL_SYSTEM_CPU_CAP_AVX                 (0x00080000)
#define LW0000_CTRL_SYSTEM_CPU_CAP_ERMS                (0x00100000)

/* feature mask (as opposed to bugs, requirements, etc.) */
#define LW0000_CTRL_SYSTEM_CPU_CAP_FEATURE_MASK        (0x1f5e7f) /* finn: Evaluated from "(LW0000_CTRL_SYSTEM_CPU_CAP_MMX | LW0000_CTRL_SYSTEM_CPU_CAP_SSE | LW0000_CTRL_SYSTEM_CPU_CAP_3DNOW | LW0000_CTRL_SYSTEM_CPU_CAP_SSE2 | LW0000_CTRL_SYSTEM_CPU_CAP_SFENCE | LW0000_CTRL_SYSTEM_CPU_CAP_WRITE_COMBINING | LW0000_CTRL_SYSTEM_CPU_CAP_ALTIVEC | LW0000_CTRL_SYSTEM_CPU_CAP_3DNOW_EXT | LW0000_CTRL_SYSTEM_CPU_CAP_MMX_EXT | LW0000_CTRL_SYSTEM_CPU_CAP_CMOV | LW0000_CTRL_SYSTEM_CPU_CAP_CLFLUSH | LW0000_CTRL_SYSTEM_CPU_CAP_SSE3 | LW0000_CTRL_SYSTEM_CPU_CAP_HT_CAPABLE | LW0000_CTRL_SYSTEM_CPU_CAP_SSE41 | LW0000_CTRL_SYSTEM_CPU_CAP_SSE42 | LW0000_CTRL_SYSTEM_CPU_CAP_AVX | LW0000_CTRL_SYSTEM_CPU_CAP_ERMS)" */

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_CAPS
 *
 * This command returns the set of system capabilities in the
 * form of an array of unsigned bytes.  System capabilities include
 * supported features and required workarounds for the system,
 * each represented by a byte offset into the table and a bit
 * position within that byte.
 *
 *   capsTblSize
 *     This parameter specifies the size in bytes of the caps table.
 *     This value should be set to LW0000_CTRL_SYSTEM_CAPS_TBL_SIZE.
 *   capsTbl
 *     This parameter specifies a pointer to the client's caps table buffer
 *     into which the system caps bits will be transferred by the RM.
 *     The caps table is an array of unsigned bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_CAPS                (0x103) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | 0x3" */

typedef struct LW0000_CTRL_SYSTEM_GET_CAPS_PARAMS {
    LwU32 capsTblSize;
    LW_DECLARE_ALIGNED(LwP64 capsTbl, 8);
} LW0000_CTRL_SYSTEM_GET_CAPS_PARAMS;

/* extract cap bit setting from tbl */
#define LW0000_CTRL_SYSTEM_GET_CAP(tbl,c)           (((LwU8)tbl[(1?c)]) & (0?c))

/* caps format is byte_index:bit_mask */
#define LW0000_CTRL_SYSTEM_CAPS_POWER_SLI_SUPPORTED                 0:0x01

/* size in bytes of system caps table */
#define LW0000_CTRL_SYSTEM_CAPS_TBL_SIZE        1

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_CHIPSET_INFO
 *
 * This command returns system chipset information.
 *
 *   vendorId
 *     This parameter returns the vendor identification for the chipset.
 *     A value of LW0000_SYSTEM_CHIPSET_ILWALID_ID indicates the chipset
 *     cannot be identified.
 *   deviceId
 *     This parameter returns the device identification for the chipset.
 *     A value of LW0000_SYSTEM_CHIPSET_ILWALID_ID indicates the chipset
 *     cannot be identified.
 *   subSysVendorId
 *     This parameter returns the subsystem vendor identification for the
 *     chipset.  A value of LW0000_SYSTEM_CHIPSET_ILWALID_ID indicates the
 *     chipset cannot be identified.
 *   subSysDeviceId
 *     This parameter returns the subsystem device identification for the 
 *     chipset. A value of LW0000_SYSTEM_CHIPSET_ILWALID_ID indicates the 
 *     chipset cannot be identified.
 *   HBvendorId
 *     This parameter returns the vendor identification for the chipset's
 *     host bridge. A value of LW0000_SYSTEM_CHIPSET_ILWALID_ID indicates
 *     the chipset's host bridge cannot be identified.
 *   HBdeviceId
 *     This parameter returns the device identification for the chipset's
 *     host bridge. A value of LW0000_SYSTEM_CHIPSET_ILWALID_ID indicates
 *     the chipset's host bridge cannot be identified.
 *   HBsubSysVendorId
 *     This parameter returns the subsystem vendor identification for the
 *     chipset's host bridge. A value of LW0000_SYSTEM_CHIPSET_ILWALID_ID
 *     indicates the chipset's host bridge cannot be identified.
 *   HBsubSysDeviceId
 *     This parameter returns the subsystem device identification for the
 *     chipset's host bridge. A value of LW0000_SYSTEM_CHIPSET_ILWALID_ID
 *     indicates the chipset's host bridge cannot be identified.
 *   sliBondId
 *     This parameter returns the SLI bond identification for the chipset.
 *   vendorNameString
 *     This parameter returns the vendor name string.
 *   chipsetNameString
 *     This parameter returns the vendor name string.
 *   sliBondNameString
 *     This parameter returns the SLI bond name string.
 *   flag
 *     This parameter specifies LW0000_CTRL_SYSTEM_CHIPSET_FLAG_XXX flags:
 *     _HAS_RESIZABLE_BAR_ISSUE_YES: Chipset where the use of resizable BAR1
 *     should be disabled - bug 3440153
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_CHIPSET_INFO (0x104) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_CHIPSET_INFO_PARAMS_MESSAGE_ID" */

/* maximum name string length */
#define LW0000_SYSTEM_MAX_CHIPSET_STRING_LENGTH (0x0000020)

/* invalid id */
#define LW0000_SYSTEM_CHIPSET_ILWALID_ID        (0xffff)

#define LW0000_CTRL_SYSTEM_GET_CHIPSET_INFO_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0000_CTRL_SYSTEM_GET_CHIPSET_INFO_PARAMS {
    LwU16 vendorId;
    LwU16 deviceId;
    LwU16 subSysVendorId;
    LwU16 subSysDeviceId;
    LwU16 HBvendorId;
    LwU16 HBdeviceId;
    LwU16 HBsubSysVendorId;
    LwU16 HBsubSysDeviceId;
    LwU32 sliBondId;
    LwU8  vendorNameString[LW0000_SYSTEM_MAX_CHIPSET_STRING_LENGTH];
    LwU8  subSysVendorNameString[LW0000_SYSTEM_MAX_CHIPSET_STRING_LENGTH];
    LwU8  chipsetNameString[LW0000_SYSTEM_MAX_CHIPSET_STRING_LENGTH];
    LwU8  sliBondNameString[LW0000_SYSTEM_MAX_CHIPSET_STRING_LENGTH];
    LwU32 flags;
} LW0000_CTRL_SYSTEM_GET_CHIPSET_INFO_PARAMS;

#define LW0000_CTRL_SYSTEM_CHIPSET_FLAG_HAS_RESIZABLE_BAR_ISSUE                  0:0
#define LW0000_CTRL_SYSTEM_CHIPSET_FLAG_HAS_RESIZABLE_BAR_ISSUE_NO  (0x00000000)
#define LW0000_CTRL_SYSTEM_CHIPSET_FLAG_HAS_RESIZABLE_BAR_ISSUE_YES (0x00000001)

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_CHIPSET_SLI_STATUS
 *
 * This command returns the SLI status of the chipset.
 *
 *   sliStatus
 *     This parameter returns the SLI status of the chipset.
 *
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_CHIPSET_SLI_STATUS               (0x105) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | 0x5" */

typedef struct LW0000_CTRL_SYSTEM_GET_CHIPSET_SLI_STATUS_PARAMS {
    LwU32 sliStatus;
} LW0000_CTRL_SYSTEM_GET_CHIPSET_SLI_STATUS_PARAMS;

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_APPROVAL_COOKIE
 *
 * This command returns the SLI approval cookie string retrieved
 * from the SBIOS.
 *
 * approvalCookieType (out)  
 *     This parameter returns the cookie type.
 *     Possible values are:
 *        LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_UNKNOWN:
 *          There is no cookie present, or it is not recognized as valid.
 *        LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_SLI:
 *          There is a valid SLI cookie.
 *        LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_COPROC:
 *          There is a valid Coproc cookie.
 *        LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_TEMPLATE:
 *          There is a cookie template present, but not valid for the system.
 *
 * approvalCookieString (out)
 *     This parameter returns the null terminated approval cookie string.
 * 
 * approvalCookieFeatures (out)
 *     Specific features associated with the cookie.
 *     See LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_FEATURE_* below.
 * 
 * vrrCookiePanelManufacturerID (out)
 *     This parameter contains the panel manufacturer ID stored in the VRR cookie.
 *     It is set to 0 if the cookie type is not VRR.
 * 
 *  vrrCookiePanelProductID (out)
 *     This parameter contains the panel product ID stored in the VRR cookie.
 *     It is set to 0 if the cookie type is not VRR.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_APPROVAL_COOKIE           (0x106) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_PARAMS_MESSAGE_ID" */

/* valid SBIOS approval cookie index values */
#define LW0000_SYSTEM_APPROVAL_COOKIE_INDEX_SLI              0
#define LW0000_SYSTEM_APPROVAL_COOKIE_INDEX_NBSI             1
#define LW0000_SYSTEM_APPROVAL_COOKIE_INDEX_VK               1
#define LW0000_SYSTEM_APPROVAL_COOKIE_INDEX_OP               2
/* maximum count of approval cookies in the SBIOS  */
#define LW0000_SYSTEM_MAX_APPROVAL_COOKIE_COUNT              3


/* maximum approval cookie string length */
#define LW0000_SYSTEM_MAX_APPROVAL_COOKIE_STRING_LENGTH      (0x0000101)
/* maximum approval cookie string length plus one*/
#define LW0000_SYSTEM_MAX_APPROVAL_COOKIE_STRING_BUFFER      (0x102) /* finn: Evaluated from "(LW0000_SYSTEM_MAX_APPROVAL_COOKIE_STRING_LENGTH + 1)" */
/* maximum approval cookie reserved data size, in bytes */
#define LW0000_SYSTEM_MAX_APPROVAL_COOKIE_RESERVED_DATA_SIZE (80)

/* approval cookie reserved data */
typedef struct LW0000_CTRL_SYSTEM_APPROVAL_COOKIE_RESERVED_DATA {
    LwU8 bytes[LW0000_SYSTEM_MAX_APPROVAL_COOKIE_RESERVED_DATA_SIZE];
} LW0000_CTRL_SYSTEM_APPROVAL_COOKIE_RESERVED_DATA;
typedef struct LW0000_CTRL_SYSTEM_APPROVAL_COOKIE_RESERVED_DATA *PLW0000_CTRL_SYSTEM_APPROVAL_COOKIE_RESERVED_DATA;

typedef struct LW0000_CTRL_SYSTEM_APPROVAL_COOKIE {
    LwU32 approvalCookieType;
    LwU8  approvalCookieString[LW0000_SYSTEM_MAX_APPROVAL_COOKIE_STRING_LENGTH];
    LwU32 approvalCookieFeatures;
    LwU16 vrrCookiePanelManufacturerID;
    LwU16 vrrCookiePanelProductID;
} LW0000_CTRL_SYSTEM_APPROVAL_COOKIE;

#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_PARAMS {
    LW0000_CTRL_SYSTEM_APPROVAL_COOKIE approvalCookie[LW0000_SYSTEM_MAX_APPROVAL_COOKIE_COUNT];
} LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_PARAMS;

/* valid approvalCookieType values */
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_NOT_RETRIEVED                      (0x00)
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_UNKNOWN                            (0x01)
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_SLI                                (0x02)
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_COPROC                             (0x03)
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_OPTIMUS_WITH_FEATURES              (0x04)
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_OPTIMUS_FAMILY_WITH_FEATURES       (0x05)
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_TEMPLATE                           (0x06)
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_OPTIMUS_GENERIC                    (0x07)
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_AIO                                (0x08)
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_VRR                                (0x09)
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_TYPE_OPTIMUS_PLATFORM                   (0x0A)

/* valid approvalCookieFeature values */
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_FEATURE_NONE                            0x00000000
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_FEATURE_TEMPLATE                        0x00000001
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_FEATURE_SLI                             0x00000002
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_FEATURE_4_WAY_SLI                       0x00000004
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_FEATURE_ALLOW_GEFORCE_ON_WS             0x00000008

#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_FEATURE_COPROC_1_DGPU_DISPLAY           0x00010000
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_FEATURE_COPROC_2_DGPU_DISPLAYS          0x00020000
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_FEATURE_COPROC_3_DGPU_DISPLAYS          0x00040000
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_FEATURE_COPROC_4_DGPU_DISPLAYS          0x00080000
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_FEATURE_COPROC_PROTECTED_CONTENT_OUTPUT 0x00100000
#define LW0000_CTRL_SYSTEM_GET_APPROVAL_COOKIE_FEATURE_COPROC_AIO_NB_GPU               0x00200000
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0000_CTRL_CMD_SYSTEM_SET_MEMORY_SIZE
 *
 * This command is used to set the system memory size in pages.
 *
 *   memorySize
 *     This parameter specifies the system memory size in pages.  All values
 *     are considered legal.
 *
 * 
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0000_CTRL_CMD_SYSTEM_SET_MEMORY_SIZE                                         (0x107) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_SET_MEMORY_SIZE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_SET_MEMORY_SIZE_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW0000_CTRL_SYSTEM_SET_MEMORY_SIZE_PARAMS {
    LwU32 memorySize;
} LW0000_CTRL_SYSTEM_SET_MEMORY_SIZE_PARAMS;

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_CLASSLIST
 *
 * This command is used to retrieve the set of system-level classes
 * supported by the platform.
 *
 *   numClasses
 *     This parameter returns the number of valid entries in the returned
 *     classes[] list.  This parameter will not exceed
 *     Lw0000_CTRL_SYSTEM_MAX_CLASSLIST_SIZE.
 *   classes
 *     This parameter returns the list of supported classes
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW0000_CTRL_CMD_SYSTEM_GET_CLASSLIST  (0x108) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_CLASSLIST_PARAMS_MESSAGE_ID" */

/* maximum number of classes returned in classes[] array */
#define LW0000_CTRL_SYSTEM_MAX_CLASSLIST_SIZE (32)

#define LW0000_CTRL_SYSTEM_GET_CLASSLIST_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW0000_CTRL_SYSTEM_GET_CLASSLIST_PARAMS {
    LwU32 numClasses;
    LwU32 classes[LW0000_CTRL_SYSTEM_MAX_CLASSLIST_SIZE];
} LW0000_CTRL_SYSTEM_GET_CLASSLIST_PARAMS;

/*
 * LW0000_CTRL_CMD_SYSTEM_NOTIFY_EVENT
 *
 * This command is used to send triggered mobile related system events
 * to the RM.
 *
 *   eventType
 *     This parameter indicates the triggered event type.  This parameter
 *     should specify a valid LW0000_CTRL_SYSTEM_EVENT_TYPE value.
 *   eventData
 *     This parameter specifies the type-dependent event data associated
 *     with EventType.  This parameter should specify a valid
 *     LW0000_CTRL_SYSTEM_EVENT_DATA value.
 *   bEventDataForced
 *     This parameter specifies what we have to do, Whether trust current
 *     Lid/Dock state or not. This parameter should specify a valid
 *     LW0000_CTRL_SYSTEM_EVENT_DATA_FORCED value.

 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 * Sync this up (#defines) with one in lwapi.spec!
 * (LW_ACPI_EVENT_TYPE & LW_ACPI_EVENT_DATA)
 */
#define LW0000_CTRL_CMD_SYSTEM_NOTIFY_EVENT (0x110) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_NOTIFY_EVENT_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_NOTIFY_EVENT_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW0000_CTRL_SYSTEM_NOTIFY_EVENT_PARAMS {
    LwU32  eventType;
    LwU32  eventData;
    LwBool bEventDataForced;
} LW0000_CTRL_SYSTEM_NOTIFY_EVENT_PARAMS;

/* valid eventType values */
#define LW0000_CTRL_SYSTEM_EVENT_TYPE_LID_STATE        (0x00000000)
#define LW0000_CTRL_SYSTEM_EVENT_TYPE_POWER_SOURCE     (0x00000001)
#define LW0000_CTRL_SYSTEM_EVENT_TYPE_DOCK_STATE       (0x00000002)
#define LW0000_CTRL_SYSTEM_EVENT_TYPE_TRUST_LID        (0x00000003)
#define LW0000_CTRL_SYSTEM_EVENT_TYPE_TRUST_DOCK       (0x00000004)

/* valid eventData values */
#define LW0000_CTRL_SYSTEM_EVENT_DATA_LID_OPEN         (0x00000000)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_LID_CLOSED       (0x00000001)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_POWER_BATTERY    (0x00000000)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_POWER_AC         (0x00000001)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_UNDOCKED         (0x00000000)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_DOCKED           (0x00000001)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_LID_DSM    (0x00000000)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_LID_DCS    (0x00000001)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_LID_LWIF   (0x00000002)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_LID_ACPI   (0x00000003)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_LID_POLL   (0x00000004)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_LID_COUNT  (0x5) /* finn: Evaluated from "(LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_LID_POLL + 1)" */
#define LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_DOCK_DSM   (0x00000000)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_DOCK_DCS   (0x00000001)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_DOCK_LWIF  (0x00000002)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_DOCK_ACPI  (0x00000003)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_DOCK_POLL  (0x00000004)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_DOCK_COUNT (0x5) /* finn: Evaluated from "(LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_DOCK_POLL + 1)" */

/* valid bEventDataForced values */
#define LW0000_CTRL_SYSTEM_EVENT_DATA_FORCED_FALSE     (0x00000000)
#define LW0000_CTRL_SYSTEM_EVENT_DATA_FORCED_TRUE      (0x00000001)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW000_CTRL_CMD_SYSTEM_GET_PLATFORM_TYPE
 *
 * This command is used to query the platform type.
 *
 *   systemType
 *     This parameter returns the type of the system.
 *     Legal values for this parameter include:
 *       LW0000_CTRL_SYSTEM_GET_PLATFORM_TYPE_DESKTOP
 *         The system is a desktop platform.
 *       LW0000_CTRL_SYSTEM_GET_PLATFORM_TYPE_MOBILE_GENERIC
 *         The system is a mobile (non-Toshiba) platform.
 *       LW0000_CTRL_SYSTEM_GET_PLATFORM_TYPE_DESKTOP
 *         The system is a mobile Toshiba platform.
 *       LW0000_CTRL_SYSTEM_GET_PLATFORM_TYPE_SOC
 *         The system is a system-on-a-chip (SOC) platform.
 *

 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_PLATFORM_TYPE       (0x111) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_CMD_SYSTEM_GET_PLATFORM_TYPE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_CMD_SYSTEM_GET_PLATFORM_TYPE_PARAMS_MESSAGE_ID (0x11U)

typedef struct LW0000_CTRL_CMD_SYSTEM_GET_PLATFORM_TYPE_PARAMS {
    LwU32 systemType;
} LW0000_CTRL_CMD_SYSTEM_GET_PLATFORM_TYPE_PARAMS;

/* valid systemType values */
#define LW0000_CTRL_SYSTEM_GET_PLATFORM_TYPE_DESKTOP        (0x000000)
#define LW0000_CTRL_SYSTEM_GET_PLATFORM_TYPE_MOBILE_GENERIC (0x000001)
#define LW0000_CTRL_SYSTEM_GET_PLATFORM_TYPE_MOBILE_TOSHIBA (0x000002)
#define LW0000_CTRL_SYSTEM_GET_PLATFORM_TYPE_SOC            (0x000003)

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_MXM_FROM_ROM
 *
 * This command is used to read the MXM data structure from ROM.
 *
 *   gpuId
 *     This parameter uniquely identifies the GPU whose associated
 *     MXM data is to be returned. The value of this field must
 *     match one of those in the table returned by
 *     LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS
 *
 *   pRomMXMBuffer
 *     This field specifies a pointer in the caller's address space to
 *     the buffer into which MXM data is to be returned.
 *
 *   romMXMBufSize
 *     This field specifies the size of the buffer referenced by romMXMBuffer
 *     in bytes.  Buffer size should be greater than or equal to
 *     LW0000_CTRL_SYSTEM_MXM_DATA_BUF_SIZE. Upon successful return
 *     it will have the MXM buffer size read from ROM. If the buffer is
 *     less, it will return buffer too small error and this will have
 *     the expected buffer size.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_INSUFFICIENT_RESOURCES
 *   LW_ERR_OPERATING_SYSTEM
 *   LW_ERR_BUFFER_TOO_SMALL
 *
 */

#define LW0000_CTRL_CMD_SYSTEM_GET_MXM_FROM_ROM             (0x11a) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_MXM_FROM_ROM_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_MXM_FROM_ROM_PARAMS_MESSAGE_ID (0x1AU)

typedef struct LW0000_CTRL_SYSTEM_GET_MXM_FROM_ROM_PARAMS {
    LwU32 gpuId;
    LW_DECLARE_ALIGNED(LwP64 pRomMXMBuffer, 8);
    LwU32 romMXMBufSize;
} LW0000_CTRL_SYSTEM_GET_MXM_FROM_ROM_PARAMS;

/* Size of the mxm header in bytes. */
#define LW0000_CTRL_SYSTEM_MXM_HEADER_SIZE       (0x00000008)
/* Byte offset of the Length field in the mxm header. */
#define LW0000_CTRL_SYSTEM_MXM_DATA_LEN_OFFSET   (0x00000006)
/* Size of the buffer to be allocated by the callers to read MXM data. 
 * Set to 1000 so everything will fit in 4k XAPI envelope.
 */
#define LW0000_CTRL_SYSTEM_MXM_DATA_BUF_SIZE     (0xfa0) /* finn: Evaluated from "(1000 * 4)" */


/*
 * LW0000_CTRL_CMD_SYSTEM_GET_MXM_FROM_ACPI
 *
 * This command is used to read the MXM data structure using ACPI
 * (used for both 2x and 3x).
 *
 *   gpuId
 *     This parameter uniquely identifies the GPU whose associated
 *     MXM data is to be returned. The value of this field must
 *     match one of those in the table returned by
 *     LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS
 *
 *   pAcpiMXMBuffer
 *     This field specifies a pointer in the caller's address space to
 *     the buffer into which MXM data is to be returned.
 *
 *   acpiMXMBufferSize
 *     This field specifies the size of the buffer referenced by pAcpiMXMBuffer
 *     in bytes. Callers should allocate LW0000_CTRL_SYSTEM_MXM_DATA_BUF_SIZE.
 *     If the buffer size is not sufficient, API will return buffer too small
 *     error and the expected buffer size will be returned in this.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_INSUFFICIENT_RESOURCES
 *   LW_ERR_OPERATING_SYSTEM
 *   LW_ERR_ILWALID_OBJECT_BUFFER
 *
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_MXM_FROM_ACPI (0x11b) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_MXM_FROM_ACPI_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_MXM_FROM_ACPI_PARAMS_MESSAGE_ID (0x1BU)

typedef struct LW0000_CTRL_SYSTEM_GET_MXM_FROM_ACPI_PARAMS {
    LwU32 gpuId;
    LW_DECLARE_ALIGNED(LwP64 pAcpiMXMBuffer, 8);
    LwU32 acpiMXMBufferSize;
} LW0000_CTRL_SYSTEM_GET_MXM_FROM_ACPI_PARAMS;



#define LW0000_CTRL_SYSTEM_GET_LVDS_SCALE_TYPE_CENTERED    (0x000000)
#define LW0000_CTRL_SYSTEM_GET_LVDS_SCALE_TYPE_SCALED      (0x000001)
#define LW0000_CTRL_SYSTEM_GET_LVDS_SCALE_TYPE_ASPECT      (0x000002)
#define LW0000_CTRL_SYSTEM_GET_LVDS_SCALE_TYPE_NATIVE      (0x000003)

/*
 * LW0000_CTRL_CMD_SYSTEM_INFORM_SBIOS_DISPLAY_SWITCH
 *
 * This command can be used to instruct the RM to inform the SBIOS of the
 * new display devices after a user-initiated display switch.  This command
 * is only supported on platforms that provide the associated LWIF interface.
 *
 *   newDisplayMask
 *     This parameter specifies the LW0073_DISPLAY_MASK value representing the
 *     mask of new display devices.  An enabled bit in newDisplayMask
 *     indicates a enabled display device with that displayId.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_GENERIC
 */
#define LW0000_CTRL_CMD_SYSTEM_INFORM_SBIOS_DISPLAY_SWITCH (0x11d) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_INFORM_SBIOS_DISPLAY_SWITCH_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_INFORM_SBIOS_DISPLAY_SWITCH_PARAMS_MESSAGE_ID (0x1DU)

typedef struct LW0000_CTRL_SYSTEM_INFORM_SBIOS_DISPLAY_SWITCH_PARAMS {
    LwU32 newDisplayDeviceMask;
} LW0000_CTRL_SYSTEM_INFORM_SBIOS_DISPLAY_SWITCH_PARAMS;

/* LW0000_CTRL_CMD_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES
 *
 * This command is used to enquire current lid and dock policies
 * that RM is following internally. This also returns if RM has any
 * hard overrides like registry keys.
 *
 *   lwrrentLidPolicy
 *     This parameter returns the current RM lid policy.
 *     This can be one of the LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_LID_* macros
 *   lwrrentDockPolicy
 *     This parameter returns the current RM dock policy.
 *     This can be one of the LW0000_CTRL_SYSTEM_EVENT_DATA_TRUST_DOCK_* macros
 *   lwrrentLidState
 *     This parameter returns the current RM lid state.
 *     This can be one of the  LW0000_CTRL_SYSTEM_EVENT_DATA_LID_* macros
 *   lwrrentDockState
 *     This parameter returns the current RM dock state.
 *     This can be one of the  following ...
 *     LW0000_CTRL_SYSTEM_EVENT_DATA_DOCKED
 *     LW0000_CTRL_SYSTEM_EVENT_DATA_UNDOCKED
 *   bForcedDockMechanismPresent
 *     This parameter returns whether RM has any forced mechanism applied.
 *     This can be one of the following
 *     LW0000_CTRL_CMD_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES_LID_OVERRIDEN_FALSE
 *     LW0000_CTRL_CMD_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES_LID_OVERRIDEN_TRUE
 *   bForcedLidMechanismPresent
 *     This parameter returns whether RM has any forced mechanism applied.
 *     This can be one of the following
 *     LW0000_CTRL_CMD_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES_DOCK_OVERRIDEN_FALSE
 *     LW0000_CTRL_CMD_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES_DOCK_OVERRIDEN_TRUE
 *   bEnforceStrictPolicy
 *     This parameter returns whether or not we are enforcing a strict policy
 *     where the mechanism is determined at the beginning and then locked.  The
 *     return value can either be TRUE or FALSE.
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES (0x11f) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES_PARAMS_MESSAGE_ID (0x1FU)

typedef struct LW0000_CTRL_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES_PARAMS {
    LwU32  lwrrentLidPolicy;
    LwU32  lwrrentDockPolicy;
    LwU32  lwrrentLidState;
    LwU32  lwrrentDockState;
    LwBool bForcedDockMechanismPresent;
    LwBool bForcedLidMechanismPresent;
    LwBool bEnforceStrictPolicy;
} LW0000_CTRL_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES_PARAMS;

#define LW0000_CTRL_CMD_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES_LID_OVERRIDEN_FALSE  (0x000000)
#define LW0000_CTRL_CMD_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES_LID_OVERRIDEN_TRUE   (0x000001)
#define LW0000_CTRL_CMD_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES_DOCK_OVERRIDEN_FALSE (0x000000)
#define LW0000_CTRL_CMD_SYSTEM_GET_LWRRENT_LID_DOCK_POLICIES_DOCK_OVERRIDEN_TRUE  (0x000001)

/*
 * LW0000_CTRL_CMD_SYSTEM_SET_POST_OUTPUT
 *
 * This command sets data to motherboard POST output.
 *      This command is supported only on Engineering VBIOS-es.
 *
 *   address
 *      This parameter specifies the POST port address to use for data output.
 *      Only addresses defined below are allowed,
 *      otherwise a LW_ERR_ILWALID_ARGUMENT is returned.
 *   data
 *      This is a byte value that will be sent to POST port.
 *      (only 8 lsb of 32 bits are used)
 * 
 * Possible status values returned are:
 *   LW_OK                        - success
 *   LW_ERR_ILWALID_ARGUMENT         - address out of range
 *   LW_ERR_NOT_SUPPORTED            - command not supported
 *   LW_ERR_ILWALID_OBJECT_HANDLE    - can not lookup GPU
 */
#define LW0000_CTRL_CMD_SYSTEM_SET_POST_OUTPUT                                    (0x120) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_SET_POST_OUTPUT_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_CMD_SYSTEM_SET_POST_OUTPUT_ADDRESS_X80                        (0x0080)
#define LW0000_CTRL_CMD_SYSTEM_SET_POST_OUTPUT_ADDRESS_X81                        (0x0081)
#define LW0000_CTRL_CMD_SYSTEM_SET_POST_OUTPUT_ADDRESS_X82                        (0x0082)
#define LW0000_CTRL_CMD_SYSTEM_SET_POST_OUTPUT_ADDRESS_X83                        (0x0083)
#define LW0000_CTRL_CMD_SYSTEM_SET_POST_OUTPUT_ADDRESS_X84                        (0x0084)

#define LW0000_CTRL_SYSTEM_SET_POST_OUTPUT_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW0000_CTRL_SYSTEM_SET_POST_OUTPUT_PARAMS {
    LwU32 address;
    LwU32 data;
} LW0000_CTRL_SYSTEM_SET_POST_OUTPUT_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0000_CTRL_CMD_SYSTEM_DEBUG_RMMSG_CTRL
 *
 * This command controls the current RmMsg filters. 
 *
 * It is only supported if RmMsg is enabled (e.g. debug builds).
 *
 *   cmd
 *     GET - Gets the current RmMsg filter string.
 *     SET - Sets the current RmMsg filter string.
 *
 *   count
 *     The length of the RmMsg filter string.
 *
 *   data
 *     The RmMsg filter string.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0000_CTRL_CMD_SYSTEM_DEBUG_RMMSG_CTRL     (0x121) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_DEBUG_RMMSG_CTRL_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_DEBUG_RMMSG_SIZE         512

#define LW0000_CTRL_SYSTEM_DEBUG_RMMSG_CTRL_CMD_GET (0x00000000)
#define LW0000_CTRL_SYSTEM_DEBUG_RMMSG_CTRL_CMD_SET (0x00000001)

#define LW0000_CTRL_SYSTEM_DEBUG_RMMSG_CTRL_PARAMS_MESSAGE_ID (0x21U)

typedef struct LW0000_CTRL_SYSTEM_DEBUG_RMMSG_CTRL_PARAMS {
    LwU32 cmd;
    LwU32 count;
    LwU8  data[LW0000_CTRL_SYSTEM_DEBUG_RMMSG_SIZE];
} LW0000_CTRL_SYSTEM_DEBUG_RMMSG_CTRL_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW0000_CTRL_SYSTEM_HWBC_INFO
 *
 * This structure contains information about the HWBC (BR04) specified by 
 * hwbcId.
 *   
 *   hwbcId
 *     This field specifies the HWBC ID.
 *   firmwareVersion
 *     This field returns the version of the firmware on the HWBC (BR04), if
 *     present. This is a packed binary number of the form 0x12345678, which
 *     corresponds to a firmware version of 12.34.56.78.
 *   subordinateBus
 *     This field returns the subordinate bus number of the HWBC (BR04).
 *   secondaryBus
 *     This field returns the secondary bus number of the HWBC (BR04).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

typedef struct LW0000_CTRL_SYSTEM_HWBC_INFO {
    LwU32 hwbcId;
    LwU32 firmwareVersion;
    LwU32 subordinateBus;
    LwU32 secondaryBus;
} LW0000_CTRL_SYSTEM_HWBC_INFO;

#define LW0000_CTRL_SYSTEM_HWBC_ILWALID_ID   (0xFFFFFFFF)

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_HWBC_INFO
 *
 * This command returns information about all Hardware Broadcast (HWBC) 
 * devices present in the system that are BR04s. To get the complete
 * list of HWBCs in the system, all GPUs present in the system must be 
 * initialized. See the description of LW0000_CTRL_CMD_GPU_ATTACH_IDS to 
 * accomplish this.
 *   
 *   hwbcInfo
 *     This field is an array of LW0000_CTRL_SYSTEM_HWBC_INFO structures into
 *     which HWBC information is placed. There is one entry for each HWBC
 *     present in the system. Valid entries are contiguous, invalid entries 
 *     have the hwbcId equal to LW0000_CTRL_SYSTEM_HWBC_ILWALID_ID. If no HWBC
 *     is present in the system, all the entries would be marked invalid, but
 *     the return value would still be SUCCESS.
 *     
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_HWBC_INFO (0x124) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_HWBC_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_MAX_HWBCS         (0x00000080)

#define LW0000_CTRL_SYSTEM_GET_HWBC_INFO_PARAMS_MESSAGE_ID (0x24U)

typedef struct LW0000_CTRL_SYSTEM_GET_HWBC_INFO_PARAMS {
    LW0000_CTRL_SYSTEM_HWBC_INFO hwbcInfo[LW0000_CTRL_SYSTEM_MAX_HWBCS];
} LW0000_CTRL_SYSTEM_GET_HWBC_INFO_PARAMS;

/*
 * LW0000_CTRL_CMD_SYSTEM_GPS_CONTROL
 *
 * This command is used to control GPS functionality.  It allows control of
 * GPU Performance Scaling (GPS), changing its operational parameters and read
 * most GPS dynamic parameters.
 *
 *   command
 *     This parameter specifies the command to execute.  Invalid commands
 *     result in the return of an LW_ERR_ILWALID_ARGUMENT status.
 *   locale
 *     This parameter indicates the specific locale to which the command
 *     'command' is to be applied.
 *     Supported range of CPU/GPU {i = 0, ..., 255}
 *   data
 *     This parameter contains a command-specific data payload.  It can
 *     be used to input data as well as output data.
 * 
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_COMMAND
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_DATA
 *   LW_ERR_ILWALID_REQUEST
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0000_CTRL_CMD_SYSTEM_GPS_CONTROL (0x122) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GPS_CONTROL_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GPS_CONTROL_PARAMS_MESSAGE_ID (0x22U)

typedef struct LW0000_CTRL_SYSTEM_GPS_CONTROL_PARAMS {
    LwU16 command;
    LwU16 locale;
    LwU32 data;
} LW0000_CTRL_SYSTEM_GPS_CONTROL_PARAMS;

/* 
 *  Valid command values :
 *  
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_INIT
 *    Is used to check if GPS was correctly initialized.
 *    Possible return (OUT) values are:
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_INIT_NO
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_INIT_YES
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_SET_EXEC
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_EXEC
 *    Are used to stop/start GPS functionality and to get current status.
 *    Possible IN/OUT values are:
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_EXEC_STOP
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_EXEC_START
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_SET_ACTIONS
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_ACTIONS
 *    Are used to control exelwtion of GPS actions and to get current status.
 *    Possible IN/OUT values are:
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_ACTIONS_OFF
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_ACTIONS_ON
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_SET_LOGIC
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_LOGIC
 *    Are used to switch current GPS logic and to retrieve current logic.
 *    Possible IN/OUT values are:
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_LOGIC_OFF
 *        Will cause that all GPS actions will be NULL.
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_LOGIC_FUZZY
 *        Fuzzy logic will determine GPS actions based on current ruleset.
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_LOGIC_DETERMINISTIC
 *        Deterministic logic will define GPS actions based on current ruleset.
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_SET_PREFERENCE
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_PREFERENCE
 *    Are used to set/retrieve system control preference.
 *    Possible IN/OUT values are:
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_PREFERENCE_CPU
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_PREFERENCE_GPU
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_PREFERENCE_BOTH
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_SET_GPU2CPU_LIMIT
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_GPU2CPU_LIMIT
 *    Are used to set/retrieve GPU2CPU pstate limits.
 *    IN/OUT values are four bytes packed into a 32-bit data field.
 *    The CPU cap index for GPU pstate 0 is in the lowest byte, the CPU cap
 *    index for the GPU pstate 3 is in the highest byte, etc.  One
 *    special value is to disable the override to the GPU2CPU map:
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_SET_PMU_GPS_STATE
 *    Is used to stop/start GPS PMU functionality.
 *  LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_PMU_GPS_STATE
 *    Is used to get the current status of PMU GPS.
 *      LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_NO_MAP_OVERRIDE
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_SET_MAX_POWER
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_MAX_POWER
 *    Are used to set/retrieve max power [mW] that system can provide.
 *    This is hardcoded GPS safety feature and logic/rules does not apply
 *    to this threshold.
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_SET_COOLING_BUDGET
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_COOLING_BUDGET
 *    Are used to set/retrieve current system cooling budget [mW].
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_SET_INTEGRAL_PERIOD
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_INTEGRAL_PERIOD
 *    Are used to set/retrieve integration interval [sec].
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_SET_RULESET
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_RULESET
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_RULE_COUNT
 *    Are used to set/retrieve used ruleset [#].  Value is checked
 *    against MAX number of rules for lwrrently used GPS logic. Also COUNT
 *    provides a way to find out how many rules exist for the current control 
 *    system.
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_SET_APP_BOOST
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_APP_BOOST
 *    Is used to set/get a delay relative to now during which to allow unbound 
 *    CPU performance.  Units are seconds.
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_SET_PWR_SUPPLY_MODE
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_PWR_SUPPLY_MODE
 *    Is used to override/get the actual power supply mode (AC/Battery).
 *    Possible IN/OUT values are:
 *      LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_PWR_SUPPLY_REAL
 *      LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_PWR_SUPPLY_FAKE_AC
 *      LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_PWR_SUPPLY_FAKE_BATT
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_VCT_SUPPORT_INFO
 *    Is used to get the Ventura system information for VCT tool
 *    Returned 32bit value should be treated as bitmask and decoded in
 *    following way:
 *    Encoding details are defined in objgps.h refer to
 *    LW_GPS_SYS_SUPPORT_INFO and corresponding bit defines.
 *  LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_SUPPORTED_FUNCTION
 *    Is used to get the supported sub-functions defined in SBIOS.  Returned
 *    value is a bitmask where each bit corresponds to different function:
 *      LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_SUPPORT
 *      LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_VENTURASTATUS
 *      LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_GETPSS
 *      LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_SETPPC
 *      LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_GETPPC
 *      LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_VENTURACB
 *      LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_SYSPARAMS
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_DELTA
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_FUTURE
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_LTMAVG
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_INTEGRAL
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_BURDEN
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_INTERMEDIATE
 *    Are used to retrieve appropriate power measurements and their derivatives
 *    in [mW] for required locale.  _BURDEN is defined only for _LOCALE_SYSTEM.
 *    _INTERMEDIATE is not defined for _LOCALE_SYSTEM, and takes an In value as
 *    index.
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_SENSOR_PARAMETERS
 *    Is used to retrieve parameters when adjusting raw sensor power reading.
 *    The values may come from SBIOS, VBIOS, registry or driver default.
 *    Possible IN value is the index of interested parameter.
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_TEMP
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_TEMP_DELTA
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_TEMP_FUTURE
 *    Are used to retrieve appropriate temperature measurements and their
 *    derivatives in [1/1000 Celsius].
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_PSTATE
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_PSTATE_CAP
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_PSTATE_MIN
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_PSTATE_MAX
 *    Are used to retrieve CPU(x)/GPU(x) p-state or it's limits.
 *    Not applicable to _LOCALE_SYSTEM.
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_PSTATE_ACTION
 *    Is used to retrieve last GPS action for given domain.
 *    Not applicable to _LOCALE_SYSTEM.
 *    Possible return (OUT) values are:
 *      LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_DEC_TO_P0
 *      LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_DEC_BY_1
 *      LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_DO_NOTHING
 *      LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_SET_LWRRENT
 *      LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_INC_BY_1
 *      LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_INC_BY_2
 *      LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_INC_TO_LFM
 *      LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_INC_TO_SLFM
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_SET_POWER_SIM_STATE
 *    Is used to set the power sensor simulator state.
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_SIM_STATE
 *    Is used to get the power simulator sensor simulator state.
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_SET_POWER_SIM_DATA
 *    Is used to set power sensor simulator data
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_SIM_DATA
 *    Is used to get power sensor simulator data
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_INIT_USING_SBIOS_AND_ACK
 *    Is used to respond to the ACPI event triggered by SBIOS.  RM will
 *    request value for budget and status, validate them, apply them
 *    and send ACK back to SBIOS.
 *  LW0000_CTRL_CMD_SYSTEM_GPS_DATA_PING_SBIOS_FOR_EVENT
 *    Is a test cmd that should notify SBIOS to send ACPI event requesting
 *    budget and status change.
 */
#define LW0000_CTRL_CMD_SYSTEM_GPS_ILWALID                       (0xFFFF)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_INIT                  (0x0000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_SET_EXEC                  (0x0001)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_EXEC                  (0x0002)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_SET_ACTIONS               (0x0003)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_ACTIONS               (0x0004)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_SET_LOGIC                 (0x0005)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_LOGIC                 (0x0006)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_SET_PREFERENCE            (0x0007)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_PREFERENCE            (0x0008)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_SET_GPU2CPU_LIMIT         (0x0009)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_GPU2CPU_LIMIT         (0x000A)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_SET_PMU_GPS_STATE         (0x000B)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_GET_PMU_GPS_STATE         (0x000C)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_SET_MAX_POWER             (0x0100)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_MAX_POWER             (0x0101)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_SET_COOLING_BUDGET        (0x0102)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_COOLING_BUDGET        (0x0103)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_SET_INTEGRAL_PERIOD       (0x0104)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_INTEGRAL_PERIOD       (0x0105)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_SET_RULESET               (0x0106)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_RULESET               (0x0107)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_RULE_COUNT            (0x0108)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_SET_APP_BOOST             (0x0109)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_APP_BOOST             (0x010A)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_SET_PWR_SUPPLY_MODE       (0x010B)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_PWR_SUPPLY_MODE       (0x010C)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_VCT_SUPPORT_INFO      (0x010D)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_GET_SUPPORTED_FUNCTIONS   (0x010E)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER                (0x0200)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_DELTA          (0x0201)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_FUTURE         (0x0202)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_LTMAVG         (0x0203)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_INTEGRAL       (0x0204)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_BURDEN         (0x0205)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_INTERMEDIATE   (0x0206)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_SENSOR_PARAMETERS    (0x0210)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_TEMP                 (0x0220)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_TEMP_DELTA           (0x0221)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_TEMP_FUTURE          (0x0222)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_PSTATE               (0x0240)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_PSTATE_CAP           (0x0241)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_PSTATE_MIN           (0x0242)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_PSTATE_MAX           (0x0243)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_PSTATE_ACTION        (0x0244)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_PSTATE_SLFM_PRESENT  (0x0245)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_SET_POWER_SIM_STATE      (0x0250)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_SIM_STATE      (0x0251)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_SET_POWER_SIM_DATA       (0x0252)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_GET_POWER_SIM_DATA       (0x0253)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_INIT_USING_SBIOS_AND_ACK (0x0320)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_PING_SBIOS_FOR_EVENT     (0x0321)

/* valid LOCALE values */
#define LW0000_CTRL_CMD_SYSTEM_GPS_LOCALE_ILWALID                (0xFFFF)
#define LW0000_CTRL_CMD_SYSTEM_GPS_LOCALE_SYSTEM                 (0x0000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_LOCALE_CPU(i)           (0x0100+((i)%0x100))
#define LW0000_CTRL_CMD_SYSTEM_GPS_LOCALE_GPU(i)           (0x0200+((i)%0x100))

/* valid data values for enums */
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_ILWALID               (0x80000000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_INIT_NO               (0x00000000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_INIT_YES              (0x00000001)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_EXEC_STOP             (0x00000000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_EXEC_START            (0x00000001)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_ACTIONS_OFF           (0x00000000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_ACTIONS_ON            (0x00000001)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_LOGIC_OFF             (0x00000000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_LOGIC_FUZZY           (0x00000001)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_LOGIC_DETERMINISTIC   (0x00000002)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_PREFERENCE_CPU        (0x00000000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_PREFERENCE_GPU        (0x00000001)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_PREFERENCE_BOTH       (0x00000002)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_NO_MAP_OVERRIDE       (0xFFFFFFFF)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_PMU_GPS_STATE_OFF     (0x00000000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_PMU_GPS_STATE_ON      (0x00000001)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_PWR_SUPPLY_REAL       (0x00000000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_PWR_SUPPLY_FAKE_AC    (0x00000001)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_PWR_SUPPLY_FAKE_BATT  (0x00000002)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_SUPPORT          (0x00000001)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_VENTURASTATUS    (0x00000002)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_GETPSS           (0x00000004)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_SETPPC           (0x00000008)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_GETPPC           (0x00000010)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_VENTURACB        (0x00000020)
#define LW0000_CTRL_CMD_SYSTEM_GPS_SYS_DEF_FUNC_SYSPARAMS        (0x00000040)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_DEC_TO_P0     (0x00000000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_DEC_BY_1      (0x00000001)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_DO_NOTHING    (0x00000002)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_SET_LWRRENT   (0x00000003)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_INC_BY_1      (0x00000004)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_INC_BY_2      (0x00000005)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_INC_TO_LFM    (0x00000006)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_ACTION_INC_TO_SLFM   (0x00000007)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_SLFM_PRESENT_NO      (0x00000000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_SLFM_PRESENT_YES     (0x00000001)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_POWER_SIM_STATE_OFF  (0x00000000)
#define LW0000_CTRL_CMD_SYSTEM_GPS_DATA_DEF_POWER_SIM_STATE_ON   (0x00000001)

/*
 *  LW0000_CTRL_CMD_SYSTEM_GPS_BATCH_CONTROL
 *
 *  This command allows exelwtion of multiple GpsControl commands within one
 *  RmControl call.  For practical reasons # of commands is limited to 16.
 *  This command shares defines with LW0000_CTRL_CMD_SYSTEM_GPS_CONTROL.
 *
 *    cmdCount
 *      Number of commands that should be exelwted.
 *      Less or equal to LW0000_CTRL_CMD_SYSTEM_GPS_BATCH_COMMAND_MAX.
 * 
 *    succeeded
 *      Number of commands that were succesully exelwted.
 *      Less or equal to LW0000_CTRL_CMD_SYSTEM_GPS_BATCH_COMMAND_MAX.
 *      Failing commands return LW0000_CTRL_CMD_SYSTEM_GPS_CMD_DEF_ILWALID
 *      in their data field.
 *
 *    cmdData
 *      Array of commands with following structure:
 *        command
 *          This parameter specifies the command to execute.
 *          Invalid commands result in the return of an
 *          LW_ERR_ILWALID_ARGUMENT status.
 *        locale
 *          This parameter indicates the specific locale to which
 *          the command 'command' is to be applied.
 *          Supported range of CPU/GPU {i = 0, ..., 255}
 *        data
 *          This parameter contains a command-specific data payload.
 *          It is used both to input data as well as to output data.
 *
 *  Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_REQUEST
 *    LW_ERR_NOT_SUPPORTED
 */
#define LW0000_CTRL_CMD_SYSTEM_GPS_BATCH_CONTROL                 (0x123) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GPS_BATCH_CONTROL_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_CMD_SYSTEM_GPS_BATCH_COMMAND_MAX             (16)
#define LW0000_CTRL_SYSTEM_GPS_BATCH_CONTROL_PARAMS_MESSAGE_ID (0x23U)

typedef struct LW0000_CTRL_SYSTEM_GPS_BATCH_CONTROL_PARAMS {
    LwU32 cmdCount;
    LwU32 succeeded;

    struct {
        LwU16 command;
        LwU16 locale;
        LwU32 data;
    } cmdData[LW0000_CTRL_CMD_SYSTEM_GPS_BATCH_COMMAND_MAX];
} LW0000_CTRL_SYSTEM_GPS_BATCH_CONTROL_PARAMS;

/*
 *  LW0000_CTRL_CMD_SYSTEM_GPS_GET_PSTATE_TABLE
 *
 *  This command is used to get the Pstate table of CPU.
 *
 *    CPU PState Table parameters :
 *      coreFreq
 *        CPU core frequency in MHz (650 = 650MHz)
 *
 *      power
 *        CPU Power in  mW - Milliwatt (8200mW = 8.2W)
 *
 *      transitionLatency
 *        us - microseconds (500 = 500us)
 *
 *      busMasterLatency
 *        us - microseconds (300 = 300us)
 *
 *      control
 *        Refer ACPI spec for details
 *
 *      status
 *        Refer ACPI spec for details
 *
 *  Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_REQUEST
 *    LW_ERR_NOT_SUPPORTED
 */

#define LW0000_CTRL_CMD_SYSTEM_GPS_GET_CPU_PSTATE_TABLE (0x126) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GPS_GET_CPU_PSTATE_TABLE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GPS_MAX_CPU_PSTATES          15

#define LW0000_CTRL_SYSTEM_GPS_GET_CPU_PSTATE_TABLE_PARAMS_MESSAGE_ID (0x26U)

typedef struct LW0000_CTRL_SYSTEM_GPS_GET_CPU_PSTATE_TABLE_PARAMS {
    LwU32 numOfCpuPstates;
    // This structure defines CPU PState table entries.
    struct {
        LwU32 coreFreq;
        LwU32 power;
        LwU32 transitionLatency;
        LwU32 busMasterLatency;
        LwU32 control;
        LwU32 status;
    } cpuPstateTable[LW0000_CTRL_SYSTEM_GPS_MAX_CPU_PSTATES];
} LW0000_CTRL_SYSTEM_GPS_GET_CPU_PSTATE_TABLE_PARAMS;

/*
 *  LW0000_CTRL_CMD_SYSTEM_GPS_GET_SENSOR_CONFIG (DEPRECATED)
 *
 *  This command is used to get the Sensor Data of CPU and GPU.
 *
 *    target
 *      0 = CPU sensor; 1 = GPU sensor
 *
 *    type
 *      0 = Relative_POWER; 1 = Absolute_POWER
 *
 *    i2cPort
 *      GPU I2C port for this sensor
 *
 *    i2cAddress
 *      I2C address for this sensor
 *
 *    configIndex   
 *      Sensor register location
 *
 *    configValue
 *      Value to write to sensor register
 *
 *    calibIndex
 *      Calibration register location
 *
 *    calibValue
 *      Value to write to calibration register
 *
 *    powerIndex
 *      Power register location
 *
 *    pollFreq
 *      Frequency to sample sensor (in Hz)
 *
 *    resistor
 *      Value of sense resistor (in milli-ohms)
 *
 *  Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_REQUEST
 *    LW_ERR_NOT_SUPPORTED
 */

#define LW0000_CTRL_CMD_SYSTEM_GPS_GET_SENSOR_CONFIG (0x125) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | 0x25" */ // DEPRECATED

#define LW0000_CTRL_SYSTEM_GPS_MAX_SENSORS           8

typedef struct LW0000_CTRL_SYSTEM_GPS_GET_SENSOR_CONFIG_PARAMS {
    LwU32 sensorCount;
    // This structure defines CPU GPU Sensor Data.
    struct {
        LwU32 target;
        LwU32 type;
        LwU32 i2cPort;
        LwU32 i2cAddress;
        LwU32 configIndex;
        LwU32 configValue;
        LwU32 calibIndex;
        LwU32 calibValue;
        LwU32 powerIndex;
        LwU32 pollFreq;
        LwU32 resistor;
    } sensorConfig[LW0000_CTRL_SYSTEM_GPS_MAX_SENSORS];
} LW0000_CTRL_SYSTEM_GPS_GET_SENSOR_CONFIG_PARAMS;

/*
 * Deprecated. Please use LW0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_V2 instead.
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS          (0x127) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PARAMS_MESSAGE_ID" */

/*
 * LW0000_CTRL_SYSTEM_MAX_ATTACHED_GPUS_SQUARED must remain equal to the square of
 * LW0000_CTRL_SYSTEM_MAX_ATTACHED_GPUS due to Check RM parsing issues.
 * LW0000_CTRL_SYSTEM_MAX_P2P_GROUP_GPUS is the maximum size of GPU groups
 * allowed for batched P2P caps queries provided by the RM control
 * LW0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_MATRIX.
 */
#define LW0000_CTRL_SYSTEM_MAX_ATTACHED_GPUS         32
#define LW0000_CTRL_SYSTEM_MAX_ATTACHED_GPUS_SQUARED 1024
#define LW0000_CTRL_SYSTEM_MAX_P2P_GROUP_GPUS        8
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_ILWALID_PEER 0xffffffff

/* P2P capabilities status index values */
#define LW0000_CTRL_P2P_CAPS_INDEX_READ              0
#define LW0000_CTRL_P2P_CAPS_INDEX_WRITE             1
#define LW0000_CTRL_P2P_CAPS_INDEX_LWLINK            2
#define LW0000_CTRL_P2P_CAPS_INDEX_ATOMICS           3
#define LW0000_CTRL_P2P_CAPS_INDEX_PROP              4
#define LW0000_CTRL_P2P_CAPS_INDEX_LOOPBACK          5
#define LW0000_CTRL_P2P_CAPS_INDEX_PCI               6
#define LW0000_CTRL_P2P_CAPS_INDEX_C2C               7
#define LW0000_CTRL_P2P_CAPS_INDEX_PCI_BAR1          8

#define LW0000_CTRL_P2P_CAPS_INDEX_TABLE_SIZE        9


#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PARAMS_MESSAGE_ID (0x27U)

typedef struct LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PARAMS {
    LwU32 gpuIds[LW0000_CTRL_SYSTEM_MAX_ATTACHED_GPUS];
    LwU32 gpuCount;
    LwU32 p2pCaps;
    LwU32 p2pOptimalReadCEs;
    LwU32 p2pOptimalWriteCEs;
    LwU8  p2pCapsStatus[LW0000_CTRL_P2P_CAPS_INDEX_TABLE_SIZE];
    LW_DECLARE_ALIGNED(LwP64 busPeerIds, 8);
} LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PARAMS;

/* valid p2pCaps values */
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_WRITES_SUPPORTED                    0:0
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_WRITES_SUPPORTED_FALSE           (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_WRITES_SUPPORTED_TRUE            (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_READS_SUPPORTED                     1:1
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_READS_SUPPORTED_FALSE            (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_READS_SUPPORTED_TRUE             (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PROP_SUPPORTED                      2:2
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PROP_SUPPORTED_FALSE             (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PROP_SUPPORTED_TRUE              (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_LWLINK_SUPPORTED                    3:3
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_LWLINK_SUPPORTED_FALSE           (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_LWLINK_SUPPORTED_TRUE            (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_ATOMICS_SUPPORTED                   4:4
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_ATOMICS_SUPPORTED_FALSE          (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_ATOMICS_SUPPORTED_TRUE           (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_LOOPBACK_SUPPORTED                  5:5
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_LOOPBACK_SUPPORTED_FALSE         (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_LOOPBACK_SUPPORTED_TRUE          (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PCI_SUPPORTED                       6:6
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PCI_SUPPORTED_FALSE              (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PCI_SUPPORTED_TRUE               (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_WRITES_SUPPORTED           7:7
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_WRITES_SUPPORTED_FALSE  (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_WRITES_SUPPORTED_TRUE   (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_READS_SUPPORTED            8:8
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_READS_SUPPORTED_FALSE   (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_READS_SUPPORTED_TRUE    (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_ATOMICS_SUPPORTED          9:9
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_ATOMICS_SUPPORTED_FALSE (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_ATOMICS_SUPPORTED_TRUE  (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_LWLINK_SUPPORTED           10:10
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_LWLINK_SUPPORTED_FALSE  (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_LWLINK_SUPPORTED_TRUE   (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_P2H2P_OPT_DISABLE                   11:11
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_P2H2P_OPT_DISABLE_FALSE          (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_P2H2P_OPT_DISABLE_TRUE           (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_C2C_SUPPORTED                       12:12
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_C2C_SUPPORTED_FALSE              (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_C2C_SUPPORTED_TRUE               (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PCI_BAR1_SUPPORTED                 13:13
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PCI_BAR1_SUPPORTED_FALSE         (0x00000000)
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PCI_BAR1_SUPPORTED_TRUE          (0x00000001)

/* P2P status codes */
#define LW0000_P2P_CAPS_STATUS_OK                                        (0x00)
#define LW0000_P2P_CAPS_STATUS_CHIPSET_NOT_SUPPORTED                     (0x01)
#define LW0000_P2P_CAPS_STATUS_GPU_NOT_SUPPORTED                         (0x02)
#define LW0000_P2P_CAPS_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED                (0x03)
#define LW0000_P2P_CAPS_STATUS_DISABLED_BY_REGKEY                        (0x04)
#define LW0000_P2P_CAPS_STATUS_NOT_SUPPORTED                             (0x05)

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_V2
 *
 * This command returns peer to peer capabilities present between GPUs.  
 * Valid requests must present a list of GPU Ids.
 *
 *   [in] gpuIds
 *     This member contains the array of GPU IDs for which we query the P2P
 *     capabilities. Valid entries are contiguous, beginning with the first 
 *     entry in the list.
 *   [in] gpuCount
 *     This member contains the number of GPU IDs stored in the gpuIds[] array.
 *   [out] p2pCaps
 *     This member returns the peer to peer capabilities discovered between the
 *     GPUs. Valid p2pCaps values include:
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_WRITES_SUPPORTED
 *         When this bit is set, peer to peer writes between subdevices owned
 *         by this device are supported.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_READS_SUPPORTED
 *         When this bit is set, peer to peer reads between subdevices owned
 *         by this device are supported.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PROP_SUPPORTED
 *         When this bit is set, peer to peer PROP between subdevices owned
 *         by this device are supported. This is enabled by default
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PCI_SUPPORTED
 *         When this bit is set, PCI is supported for all P2P between subdevices
 *         owned by this device.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_LWLINK_SUPPORTED
 *         When this bit is set, LWLINK is supported for all P2P between subdevices
 *         owned by this device.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_ATOMICS_SUPPORTED
 *         When this bit is set, peer to peer atomics between subdevices owned
 *         by this device are supported.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_LOOPBACK_SUPPORTED
 *         When this bit is set, peer to peer loopback is supported for subdevices
 *         owned by this device.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_WRITES_SUPPORTED
 *         When this bit is set, indirect peer to peer writes between subdevices
 *         owned by this device are supported.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_READS_SUPPORTED
 *         When this bit is set, indirect peer to peer reads between subdevices
 *         owned by this device are supported.
 *      LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_ATOMICS_SUPPORTED
 *         When this bit is set, indirect peer to peer atomics between
 *         subdevices owned by this device are supported.
 *      LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_LWLINK_SUPPORTED
 *         When this bit is set, indirect LWLINK is supported for subdevices
 *         owned by this device.
 *      LW0000_CTRL_SYSTEM_GET_P2P_CAPS_P2H2P_OPT_DISABLE
 *         When this bit is set, indicates the client that the POR is to 
 *         not use P2H2P double buffering.
 *      LW0000_CTRL_SYSTEM_GET_P2P_CAPS_C2C_SUPPORTED
 *         When this bit is set, C2C P2P is supported between the GPUs
 *      LW0000_CTRL_SYSTEM_GET_P2P_CAPS_BAR1_SUPPORTED
 *         When this bit is set, BAR1 P2P is supported between the GPUs
 *         mentioned in @ref gpuIds
 *   [out] p2pOptimalReadCEs
 *      For a pair of GPUs, return mask of CEs to use for p2p reads over Lwlink
 *   [out] p2pOptimalWriteCEs
 *      For a pair of GPUs, return mask of CEs to use for p2p writes over Lwlink
 *   [out] p2pCapsStatus
 *     This member returns status of all supported p2p capabilities. Valid
 *     status values include:
 *       LW0000_P2P_CAPS_STATUS_OK
 *         P2P capability is supported.
 *       LW0000_P2P_CAPS_STATUS_CHIPSET_NOT_SUPPORTED
 *         Chipset doesn't support p2p capability.
 *       LW0000_P2P_CAPS_STATUS_GPU_NOT_SUPPORTED
 *         GPU doesn't support p2p capability.
 *       LW0000_P2P_CAPS_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED
 *         IOH topology isn't supported. For e.g. root ports are on different
 *         IOH.
 *       LW0000_P2P_CAPS_STATUS_DISABLED_BY_REGKEY
 *         P2P Capability is disabled by a regkey.
 *       LW0000_P2P_CAPS_STATUS_NOT_SUPPORTED
 *         P2P Capability is not supported.
 *       LW0000_P2P_CAPS_STATUS_LWLINK_SETUP_FAILED
 *         Indicates that LwLink P2P link setup failed.
 *    [out] busPeerIds
 *        Peer ID matrix. It is a one-dimentional array.
 *        busPeerIds[X * gpuCount + Y] maps from index X to index Y in
 *        the gpuIds[] table. For invalid or non-existent peer busPeerIds[]
 *        has the value LW0000_CTRL_SYSTEM_GET_P2P_CAPS_ILWALID_PEER.
 * 
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_V2                           (0x12b) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_P2P_CAPS_V2_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_V2_PARAMS_MESSAGE_ID (0x2BU)

typedef struct LW0000_CTRL_SYSTEM_GET_P2P_CAPS_V2_PARAMS {
    LwU32 gpuIds[LW0000_CTRL_SYSTEM_MAX_ATTACHED_GPUS];
    LwU32 gpuCount;
    LwU32 p2pCaps;
    LwU32 p2pOptimalReadCEs;
    LwU32 p2pOptimalWriteCEs;
    LwU8  p2pCapsStatus[LW0000_CTRL_P2P_CAPS_INDEX_TABLE_SIZE];
    LwU32 busPeerIds[LW0000_CTRL_SYSTEM_MAX_ATTACHED_GPUS_SQUARED];
} LW0000_CTRL_SYSTEM_GET_P2P_CAPS_V2_PARAMS;

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_MATRIX
 *
 * This command returns peer to peer capabilities present between all pairs of
 * GPU IDs {(a, b) : a in gpuIdGrpA and b in gpuIdGrpB}. This can be used to
 * collect all P2P capabilities in the system - see the SRT:
 *     LW0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_MATRIX_TEST
 * for a demonstration.
 *
 * The call will query for all pairs between set A and set B, and returns
 * results in both link directions. The results are two-dimensional arrays where
 * the first dimension is the index within the set-A array of one GPU ID under
 * consideration, and the second dimension is the index within the set-B array
 * of the other GPU ID under consideration.
 *
 * That is, the result arrays are *ALWAYS* to be indexed first with the set-A
 * index, then with the set-B index. The B-to-A direction of results are put in
 * the b2aOptimal(Read|Write)CEs. This makes it unnecessary to call the query
 * twice, since the usual use case requires both directions.
 *
 * If a set is being compared against itself (by setting grpBCount to 0), then
 * the result matrices are symmetric - it doesn't matter which index is first.
 * However, the choice of indices is effectively a choice of which ID is "B" and
 * which is "A" for the "a2b" and "b2a" directional results.
 *
 *   [in] grpACount
 *     This member contains the number of GPU IDs stored in the gpuIdGrpA[]
 *     array. Must be >= 0.
 *   [in] grpBCount
 *     This member contains the number of GPU IDs stored in the gpuIdGrpB[]
 *     array. Can be == 0 to specify a check of group A against itself.
 *   [in] gpuIdGrpA
 *     This member contains the array of GPU IDs in "group A", each of which
 *     will have its P2P capabilities returned with respect to each GPU ID in
 *     "group B". Valid entries are contiguous, beginning with the first entry
 *     in the list.
 *   [in] gpuIdGrpB
 *     This member contains the array of GPU IDs in "group B", each of which
 *     will have its P2P capabilities returned with respect to each GPU ID in
 *     "group A". Valid entries are contiguous, beginning with the first entry
 *     in the list. May be equal to gpuIdGrpA, but best performance requires
 *     that the caller specifies grpBCount = 0 in this case, and ignores this.
 *   [out] p2pCaps
 *     This member returns the peer to peer capabilities discovered between the
 *     pairs of input GPUs between the groups, indexed by [A_index][B_index].
 *     Valid p2pCaps values include:
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_WRITES_SUPPORTED
 *         When this bit is set, peer to peer writes between subdevices owned
 *         by this device are supported.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_READS_SUPPORTED
 *         When this bit is set, peer to peer reads between subdevices owned
 *         by this device are supported.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PROP_SUPPORTED
 *         When this bit is set, peer to peer PROP between subdevices owned
 *         by this device are supported. This is enabled by default
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_PCI_SUPPORTED
 *         When this bit is set, PCI is supported for all P2P between subdevices
 *         owned by this device.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_LWLINK_SUPPORTED
 *         When this bit is set, LWLINK is supported for all P2P between subdevices
 *         owned by this device.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_ATOMICS_SUPPORTED
 *         When this bit is set, peer to peer atomics between subdevices owned
 *         by this device are supported.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_LOOPBACK_SUPPORTED
 *         When this bit is set, peer to peer loopback is supported for subdevices
 *         owned by this device.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_WRITES_SUPPORTED
 *         When this bit is set, indirect peer to peer writes between subdevices
 *         owned by this device are supported.
 *       LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_READS_SUPPORTED
 *         When this bit is set, indirect peer to peer reads between subdevices
 *         owned by this device are supported.
 *      LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_ATOMICS_SUPPORTED
 *         When this bit is set, indirect peer to peer atomics between
 *         subdevices owned by this device are supported.
 *      LW0000_CTRL_SYSTEM_GET_P2P_CAPS_INDIRECT_LWLINK_SUPPORTED
 *         When this bit is set, indirect LWLINK is supported for subdevices
 *         owned by this device.
 *      LW0000_CTRL_SYSTEM_GET_P2P_CAPS_P2H2P_OPT_DISABLE
 *         When this bit is set, indicates the client that the POR is to
 *         not use P2H2P double buffering.
 *   [out] a2bOptimalReadCes
 *      For a pair of GPUs, return mask of CEs to use for p2p reads over Lwlink
 *      in the A-to-B direction.
 *   [out] a2bOptimalWriteCes
 *      For a pair of GPUs, return mask of CEs to use for p2p writes over Lwlink
 *      in the A-to-B direction.
 *   [out] b2aOptimalReadCes
 *      For a pair of GPUs, return mask of CEs to use for p2p reads over Lwlink
 *      in the B-to-A direction.
 *   [out] b2aOptimalWriteCes
 *      For a pair of GPUs, return mask of CEs to use for p2p writes over Lwlink
 *      in the B-to-A direction.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_MATRIX (0x13a) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_P2P_CAPS_MATRIX_PARAMS_MESSAGE_ID" */

typedef LwU32 LW0000_CTRL_P2P_CAPS_MATRIX_ROW[LW0000_CTRL_SYSTEM_MAX_P2P_GROUP_GPUS];
#define LW0000_CTRL_SYSTEM_GET_P2P_CAPS_MATRIX_PARAMS_MESSAGE_ID (0x3AU)

typedef struct LW0000_CTRL_SYSTEM_GET_P2P_CAPS_MATRIX_PARAMS {
    LwU32                           grpACount;
    LwU32                           grpBCount;
    LwU32                           gpuIdGrpA[LW0000_CTRL_SYSTEM_MAX_P2P_GROUP_GPUS];
    LwU32                           gpuIdGrpB[LW0000_CTRL_SYSTEM_MAX_P2P_GROUP_GPUS];
    LW0000_CTRL_P2P_CAPS_MATRIX_ROW p2pCaps[LW0000_CTRL_SYSTEM_MAX_P2P_GROUP_GPUS];
    LW0000_CTRL_P2P_CAPS_MATRIX_ROW a2bOptimalReadCes[LW0000_CTRL_SYSTEM_MAX_P2P_GROUP_GPUS];
    LW0000_CTRL_P2P_CAPS_MATRIX_ROW a2bOptimalWriteCes[LW0000_CTRL_SYSTEM_MAX_P2P_GROUP_GPUS];
    LW0000_CTRL_P2P_CAPS_MATRIX_ROW b2aOptimalReadCes[LW0000_CTRL_SYSTEM_MAX_P2P_GROUP_GPUS];
    LW0000_CTRL_P2P_CAPS_MATRIX_ROW b2aOptimalWriteCes[LW0000_CTRL_SYSTEM_MAX_P2P_GROUP_GPUS];
} LW0000_CTRL_SYSTEM_GET_P2P_CAPS_MATRIX_PARAMS;

/*
 * LW0000_CTRL_CMD_SYSTEM_GPS_CTRL
 *
 * This command is used to execute general GPS Functions, most dealing with
 * calling SBIOS, or retrieving cached sensor and GPS state data.
 *
 *   version
 *     This parameter specifies the version of the interface.  Legal values
 *     for this parameter are 1.
 *   cmd
 *     This parameter specifies the GPS API to be ilwoked. 
 *     Valid values for this parameter are:
 *       LW0000_CTRL_GPS_CMD_GET_PS_STATUS
 *         This command gets the status of power steering enable.  When this
 *         command is specified the input parameter contains ???.
 *       LW0000_CTRL_GPS_CMD_SET_PS_STATUS
 *         This command sets the status of power steering enable.  When this
 *         command is specified the input parameter contains ???.
 *       LW0000_CTRL_GPS_CMD_GET_THERM_LIMIT
 *         This command gets the temperature limit for thermal controller. When
 *         this command is specified the input parameter contains ???.
 *      LW0000_CTRL_GPS_CMD_SET_THERM_LIMIT
 *         This command set the temperature limit for thermal controller.  When
 *         this command is specified the input parameter contains ???.
 *   input
 *     This parameter specifies the cmd-specific input value.
 *   result
 *     This parameter returns the cmd-specific output value.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0000_CTRL_CMD_SYSTEM_GPS_CTRL (0x12a) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GPS_CTRL_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GPS_CTRL_PARAMS_MESSAGE_ID (0x2AU)

typedef struct LW0000_CTRL_SYSTEM_GPS_CTRL_PARAMS {
    LwU32 cmd;
    LwS32 input[2];
    LwS32 result[4];
} LW0000_CTRL_SYSTEM_GPS_CTRL_PARAMS;

/* valid version values */
#define LW0000_CTRL_GPS_PSHARE_PARAMS_PSP_LWRRENT_VERSION           (0x00010000)

/* valid cmd values */
#define LW0000_CTRL_GPS_CMD_TYPE_GET_PS_STATUS                      (0x00000000)
#define LW0000_CTRL_GPS_RESULT_PS_STATUS                            (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_PS_STATUS                      (0x00000001)
#define LW0000_CTRL_GPS_INPUT_PS_STATUS                             (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_THERM_LIMIT                    (0x00000002)
#define LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_RESULT_THERMAL_LIMIT                        (0x00000000)
#define LW0000_CTRL_GPS_RESULT_MIN_LIMIT                            (0x00000001)
#define LW0000_CTRL_GPS_RESULT_MAX_LIMIT                            (0x00000002)
#define LW0000_CTRL_GPS_RESULT_LIMIT_SOURCE                         (0x00000003)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_THERM_LIMIT                    (0x00000003)
//      LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_INPUT_THERMAL_LIMIT                         (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_TEMP_CTRL_DOWN_N_DELTA         (0x00000004)
//      LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_RESULT_TEMP_CTRL_DOWN_N_DELTA               (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_TEMP_CTRL_DOWN_N_DELTA         (0x00000005)
//      LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_INPUT_TEMP_CTRL_DOWN_N_DELTA                (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_TEMP_CTRL_HOLD_DELTA           (0x00000006)
//      LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_RESULT_TEMP_CTRL_HOLD_DELTA                 (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_TEMP_CTRL_HOLD_DELTA           (0x00000007)
//      LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_INPUT_TEMP_CTRL_HOLD_DELTA                  (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_TEMP_CTRL_UP_DELTA             (0x00000008)
//      LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_RESULT_TEMP_CTRL_UP_DELTA                   (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_TEMP_CTRL_UP_DELTA             (0x00000009)
//      LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_INPUT_TEMP_CTRL_UP_DELTA                    (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_TEMP_CTRL_ENGAGE_DELTA         (0x0000000A)
//      LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_RESULT_TEMP_CTRL_ENGAGE_DELTA               (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_TEMP_CTRL_ENGAGE_DELTA         (0x0000000B)
//      LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_INPUT_TEMP_CTRL_ENGAGE_DELTA                (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_TEMP_CTRL_DISENGAGE_DELTA      (0x0000000C)
//      LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_RESULT_TEMP_CTRL_DISENGAGE_DELTA            (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_TEMP_CTRL_DISENGAGE_DELTA      (0x0000000D)
//      LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_INPUT_TEMP_CTRL_DISENGAGE_DELTA             (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_ENGAGE_THRESHOLD_UP            (0x0000000E)
#define LW0000_CTRL_GPS_RESULT_GET_ENGAGE_THRESHOLD_UP              (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_ENGAGE_THRESHOLD_UP            (0x0000000F)
#define LW0000_CTRL_GPS_INPUT_ENGAGE_THRESHOLD_UP                   (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_DISENGAGE_THRESHOLD_DOWN       (0x00000010)
#define LW0000_CTRL_GPS_RESULT_DISENGAGE_THRESHOLD_DOWN             (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_DISENGAGE_THRESHOLD_DOWN       (0x00000011)
#define LW0000_CTRL_GPS_INPUT_DISENGAGE_THRESHOLD_DOWN              (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_HYST_HIGH                      (0x00000012)
#define LW0000_CTRL_GPS_RESULT_HYST_HIGH                            (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_HYST_HIGH                      (0x00000013)
#define LW0000_CTRL_GPS_INPUT_HYST_HIGH                             (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_HYST_LOW                       (0x00000014)
#define LW0000_CTRL_GPS_RESULT_HYST_LOW                             (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_HYST_LOW                       (0x00000015)
#define LW0000_CTRL_GPS_INPUT_HYST_LOW                              (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_TEMP_CTRL_STATUS               (0x00000016)
#define LW0000_CTRL_GPS_RESULT_TEMP_CTRL_STATUS                     (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_TEMP_CTRL_STATUS               (0x00000017)
#define LW0000_CTRL_GPS_INPUT_TEMP_CTRL_STATUS                      (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_CPU_GET_UTIL_AVG_NUM           (0x00000018)
#define LW0000_CTRL_GPS_RESULT_CPU_SET_UTIL_AVG_NUM                 (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_CPU_SET_UTIL_AVG_NUM           (0x00000019)
#define LW0000_CTRL_GPS_INPUT_CPU_GET_UTIL_AVG_NUM                  (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_PERF_SENSOR                    (0x0000001A)
//      LW0000_CTRL_GPS_INPUT_SENSOR_INDEX                          (0x00000000)
#define LW0000_CTRL_GPS_INPUT_NEXT_EXPECTED_POLL                    (0x00000001)
#define LW0000_CTRL_GPS_RESULT_PERF_SENSOR_VALUE                    (0x00000000)
#define LW0000_CTRL_GPS_RESULT_PERF_SENSOR_AVAILABLE                (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_CALL_ACPI                          (0x0000001B)
#define LW0000_CTRL_GPS_INPUT_ACPI_CMD                              (0x00000000)
#define LW0000_CTRL_GPS_INPUT_ACPI_PARAM_IN                         (0x00000001)
#define LW0000_CTRL_GPS_OUTPUT_ACPI_RESULT_1                        (0x00000000)
#define LW0000_CTRL_GPS_OUTPUT_ACPI_RESULT_2                        (0x00000001)
#define LW0000_CTRL_GPS_OUTPUT_ACPI_PSHAREPARAM_STATUS              (0x00000000)
#define LW0000_CTRL_GPS_OUTPUT_ACPI_PSHAREPARAM_VERSION             (0x00000001)
#define LW0000_CTRL_GPS_OUTPUT_ACPI_PSHAREPARAM_SZ                  (0x00000002)
#define LW0000_CTRL_GPS_OUTPUT_ACPI_PSS_SZ                          (0x00000000)
#define LW0000_CTRL_GPS_OUTPUT_ACPI_PSS_COUNT                       (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_IGPU_TURBO                     (0x0000001C)
#define LW0000_CTRL_GPS_INPUT_SET_IGPU_TURBO                        (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_CPU_FREQ_MIN_MHZ               (0x0000001E)
#define LW0000_CTRL_GPS_OUTPUT_CPU_FREQ_MIN_MHZ                     (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_CPU_FREQ_MIN_MHZ               (0x0000001F)
#define LW0000_CTRL_GPS_INPUT_CPU_FREQ_MIN_MHZ                      (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_CPU_SAMPLE_PERIOD              (0x00000020)
#define LW0000_CTRL_GPS_OUTPUT_CPU_SAMPLE_PERIOD                    (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_CPU_SAMPLE_PERIOD              (0x00000021)
#define LW0000_CTRL_GPS_INPUT_CPU_SAMPLE_PERIOD                     (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_CPU_FREQ_MAX_MHZ               (0x00000024)
#define LW0000_CTRL_GPS_OUTPUT_CPU_FREQ_MAX_MHZ                     (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_CPU_FREQ_MAX_MHZ               (0x00000025)
#define LW0000_CTRL_GPS_INPUT_CPU_FREQ_MAX_MHZ                      (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_TEMP_PERIOD                    (0x00000026)
#define LW0000_CTRL_GPS_INPUT_TEMP_PERIOD                           (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_TEMP_PERIOD                    (0x00000027)
#define LW0000_CTRL_GPS_RESULT_TEMP_PERIOD                          (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_TEMP_NUDGE_FACTOR              (0x00000028)
#define LW0000_CTRL_GPS_INPUT_TEMP_NUDGE_UP                         (0x00000000)
#define LW0000_CTRL_GPS_INPUT_TEMP_NUDGE_DOWN                       (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_TEMP_NUDGE_FACTOR              (0x00000029)
#define LW0000_CTRL_GPS_RESULT_TEMP_NUDGE_UP                        (0x00000000)
#define LW0000_CTRL_GPS_RESULT_TEMP_NUDGE_DOWN                      (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_TEMP_THRESHOLD_SAMPLES         (0x0000002A)
#define LW0000_CTRL_GPS_INPUT_TEMP_THRESHOLD_SAMPLE_HOLD            (0x00000000)
#define LW0000_CTRL_GPS_INPUT_TEMP_THRESHOLD_SAMPLE_STEP            (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_TEMP_THRESHOLD_SAMPLES         (0x0000002B)
#define LW0000_CTRL_GPS_RESULT_TEMP_THRESHOLD_SAMPLE_HOLD           (0x00000000)
#define LW0000_CTRL_GPS_RESULT_TEMP_THRESHOLD_SAMPLE_STEP           (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_TEMP_PERF_LIMITS               (0x0000002C)
#define LW0000_CTRL_GPS_INPUT_TEMP_PERF_LIMIT_UPPER                 (0x00000000)
#define LW0000_CTRL_GPS_INPUT_TEMP_PERF_LIMIT_LOWER                 (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_TEMP_PERF_LIMITS               (0x0000002D)
#define LW0000_CTRL_GPS_RESULT_TEMP_PERF_LIMIT_UPPER                (0x00000000)
#define LW0000_CTRL_GPS_RESULT_TEMP_PERF_LIMIT_LOWER                (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_PM1_AVAILABLE                  (0x0000002E)
#define LW0000_CTRL_GPS_INPUT_PM1_AVAILABLE                         (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_PM1_AVAILABLE                  (0x0000002F)
#define LW0000_CTRL_GPS_OUTPUT_PM1_AVAILABLE                        (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_WEB_DATA                       (0x00000030)
#define LW0000_CTRL_GPS_INPUT_WEB_FRAME_TIME_FLOOR_US               (0x00000000)
#define LW0000_CTRL_GPS_INPUT_WEB_FRAME_TIME_US                     (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_WEB_DATA                       (0x00000031)
#define LW0000_CTRL_GPS_OUTPUT_WEB_ENABLED                          (0x00000000)
#define LW0000_CTRL_GPS_OUTPUT_WEB_FRAME_TIME_US                    (0x00000001)
#define LW0000_CTRL_GPS_OUTPUT_WEB_FRAME_TIME_FLOOR_US              (0x00000002)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_WEB_TIME_BOUNDARIES            (0x00000032)
#define LW0000_CTRL_GPS_INPUT_WEB_FRAME_TIME_US_MIN                 (0x00000000)
#define LW0000_CTRL_GPS_INPUT_WEB_FRAME_TIME_US_MAX                 (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_WEB_REF_DRAIN_RATE             (0x00000033)
#define LW0000_CTRL_GPS_OUTPUT_WEB_REF_DRAIN_RATE                   (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_WEB_REF_DRAIN_RATE             (0x00000034)
#define LW0000_CTRL_GPS_INPUT_WEB_REF_DRAIN_RATE                    (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_WEB_CTRL_STAT_N_TGT_DRAIN_RATE (0x00000035)
#define LW0000_CTRL_GPS_OUTPUT_WEB_CTRL_STAT_ALL                    (0x00000000)
#define LW0000_CTRL_GPS_OUTPUT_WEB_TGT_DRAIN_RATE_PCT               (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_WEB_CTRL_STAT_N_TGT_DRAIN_RATE (0x00000036)
#define LW0000_CTRL_GPS_INPUT_WEB_CTRL_STAT_API                     (0x00000000)
#define LW0000_CTRL_GPS_INPUT_WEB_TGT_DRAIN_RATE_PCT                (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_HYST_HIGH_AC                   (0x00000037)
#define LW0000_CTRL_GPS_RESULT_HYST_HIGH_AC                         (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_HYST_HIGH_AC                   (0x00000038)
#define LW0000_CTRL_GPS_INPUT_HYST_HIGH_AC                          (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_HYST_LOW_AC                    (0x00000039)
#define LW0000_CTRL_GPS_RESULT_HYST_LOW_AC                          (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_HYST_LOW_AC                    (0x0000003A)
#define LW0000_CTRL_GPS_INPUT_HYST_LOW_AC                           (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_HYST_HIGH_BATT                 (0x0000003B)
#define LW0000_CTRL_GPS_RESULT_HYST_HIGH_BATT                       (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_HYST_HIGH_BATT                 (0x0000003C)
#define LW0000_CTRL_GPS_INPUT_HYST_HIGH_BATT                        (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_HYST_LOW_BATT                  (0x0000003D)
#define LW0000_CTRL_GPS_RESULT_HYST_LOW_BATT                        (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_HYST_LOW_BATT                  (0x0000003E)
#define LW0000_CTRL_GPS_INPUT_HYST_LOW_BATT                         (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_HYST_HIGH_BBOOST               (0x0000003F)
#define LW0000_CTRL_GPS_RESULT_HYST_HIGH_BBOOST                     (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_HYST_HIGH_BBOOST               (0x00000040)
#define LW0000_CTRL_GPS_INPUT_HYST_HIGH_BBOOST                      (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_HYST_LOW_BBOOST                (0x00000041)
#define LW0000_CTRL_GPS_RESULT_HYST_LOW_BBOOST                      (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_HYST_LOW_BBOOST                (0x00000042)
#define LW0000_CTRL_GPS_INPUT_HYST_LOW_BBOOST                       (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_CPU_PACKAGE_LIMITS             (0x00000044)
#define LW0000_CTRL_GPS_CMD_TYPE_GET_CPU_PACKAGE_LIMITS_PL1         (0x00000000)
#define LW0000_CTRL_GPS_CMD_TYPE_GET_CPU_PACKAGE_LIMITS_PL2         (0x00000001)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_CPU_PACKAGE_LIMITS             (0x00000045)
#define LW0000_CTRL_GPS_CMD_TYPE_SET_CPU_PACKAGE_LIMITS_PL1         (0x00000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_CPU_FREQ_LIMIT                 (0x00000046)
#define LW0000_CTRL_GPS_CMD_TYPE_GET_CPU_FREQ_LIMIT_MHZ             (0000000000)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_CPU_FREQ_LIMIT                 (0x00000047)
#define LW0000_CTRL_GPS_CMD_TYPE_SET_CPU_FREQ_LIMIT_MHZ             (0000000000)

#define LW0000_CTRL_GPS_CMD_TYPE_GET_PPM                            (0x00000048)
#define LW0000_CTRL_GPS_CMD_TYPE_GET_PPM_INDEX                      (0000000000)
#define LW0000_CTRL_GPS_CMD_TYPE_GET_PPM_AVAILABLE_MASK             (0000000001)

#define LW0000_CTRL_GPS_CMD_TYPE_SET_PPM                            (0x00000049)
#define LW0000_CTRL_GPS_CMD_TYPE_SET_PPM_INDEX                      (0000000000)
#define LW0000_CTRL_GPS_CMD_TYPE_SET_PPM_INDEX_MAX                  (2)

#define LW0000_CTRL_GPS_PPM_INDEX                                   7:0
#define LW0000_CTRL_GPS_PPM_INDEX_MAXPERF                           (0)
#define LW0000_CTRL_GPS_PPM_INDEX_BALANCED                          (1)
#define LW0000_CTRL_GPS_PPM_INDEX_QUIET                             (2)
#define LW0000_CTRL_GPS_PPM_INDEX_ILWALID                           (0xFF)
#define LW0000_CTRL_GPS_PPM_MASK                                    15:8
#define LW0000_CTRL_GPS_PPM_MASK_ILWALID                            (0)

/* valid PS_STATUS result values */
#define LW0000_CTRL_GPS_CMD_PS_STATUS_OFF                           (0)
#define LW0000_CTRL_GPS_CMD_PS_STATUS_ON                            (1)


/*
 * LW0000_CTRL_CMD_SYSTEM_SET_SELWRITY_SETTINGS
 *
 * This command allows privileged users to update the values of
 * security settings governing RM behavior.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT,
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *
 * Please note: as implied above, administrator privileges are
 * required to modify security settings.
 */
#define LW0000_CTRL_CMD_SYSTEM_SET_SELWRITY_SETTINGS                (0x129) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | 0x29" */

/*
 * LW0000_CTRL_CMD_SYSTEM_GPS_CALL_ACPI
 *
 * This command allows users to call GPS ACPI commands for testing purposes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT,
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *
 */
#define GPS_MAX_COUNTERS_PER_BLOCK                                  32
typedef struct LW0000_CTRL_SYSTEM_GPS_GET_PERF_SENSOR_COUNTERS_PARAMS {
    LwU32 objHndl;
    LwU32 blockId;
    LwU32 nextExpectedSampleTimems;
    LwU32 countersReq;
    LwU32 countersReturned;
    LwU32 counterBlock[GPS_MAX_COUNTERS_PER_BLOCK];
} LW0000_CTRL_SYSTEM_GPS_GET_PERF_SENSOR_COUNTERS_PARAMS;

#define LW0000_CTRL_CMD_SYSTEM_GPS_GET_PERF_SENSORS          (0x12c) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | 0x2C" */

#define LW0000_CTRL_CMD_SYSTEM_GPS_GET_EXTENDED_PERF_SENSORS (0x12e) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | 0x2E" */


/*
 * LW0000_CTRL_CMD_SYSTEM_GPS_CALL_ACPI
 *
 * This command allows users to call GPS ACPI commands for testing purposes.
 *
 *   cmd
 *      This parameter specifies the GPS ACPI command to execute.
 * 
 *   input
 *      This parameter specified the cmd-dependent input value. 
 *
 *   resultSz
 *      This parameter returns the size (in bytes) of the valid data
 *      returned in  the result parameter.
 *
 *   result
 *      This parameter returns the results of the specified cmd.
 *      The maximum size (in bytes) of this returned data will
 *      not exceed GPS_MAX_ACPI_OUTPUT_BUFFER_SIZE
 *
 *   GPS_MAX_ACPI_OUTPUT_BUFFER_SIZE
 *      The size of buffer (result) in unit of LwU32.
 *      The smallest value is sizeof(PSS_ENTRY)*ACPI_PSS_ENTRY_MAX. 
 *      Since the prior one is 24 bytes, and the later one is 48, 
 *      this value cannot be smaller than 288.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT,
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *
 */
#define GPS_MAX_ACPI_OUTPUT_BUFFER_SIZE                      288
#define LW0000_CTRL_SYSTEM_GPS_CALL_ACPI_PARAMS_MESSAGE_ID (0x2DU)

typedef struct LW0000_CTRL_SYSTEM_GPS_CALL_ACPI_PARAMS {
    LwU32 cmd;
    LwU32 input;
    LwU32 resultSz;
    LwU32 result[GPS_MAX_ACPI_OUTPUT_BUFFER_SIZE];
} LW0000_CTRL_SYSTEM_GPS_CALL_ACPI_PARAMS;

#define LW0000_CTRL_CMD_SYSTEM_GPS_CALL_ACPI       (0x12d) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GPS_CALL_ACPI_PARAMS_MESSAGE_ID" */

/*
 * LW0000_CTRL_SYSTEM_PARAM_*
 *
 * The following is a list of system-level parameters (often sensors) that the
 * driver can be made aware of. They are primarily intended to be used by system
 * power-balancing algorithms that require system-wide visibility in order to
 * function. The names and values used here are established and specified in
 * several different LWPU dolwments that are made externally available. Thus,
 * updates to this list must be made with great caution. The only permissible
 * change is to append new parameters. Reordering is strictly prohibited.
 *
 * Brief Parameter Summary:
 *     TGPU - GPU temperature                   (LwTemp)
 *     PDTS - CPU package temperature           (LwTemp)
 *     SFAN - System fan speed                  (% of maximum fan speed)
 *     SKNT - Skin temperature                  (LwTemp)
 *     CPUE - CPU energy counter                (LwU32)
 *     TMP1 - Additional temperature sensor 1   (LwTemp)
 *     TMP2 - Additional temperature sensor 2   (LwTemp)
 *     CTGP - Mode 2 power limit offset         (LwU32)
 *     PPMD - Power mode data                   (LwU32)
 */
#define LW0000_CTRL_SYSTEM_PARAM_TGPU              (0x00000000)
#define LW0000_CTRL_SYSTEM_PARAM_PDTS              (0x00000001)
#define LW0000_CTRL_SYSTEM_PARAM_SFAN              (0x00000002)
#define LW0000_CTRL_SYSTEM_PARAM_SKNT              (0x00000003)
#define LW0000_CTRL_SYSTEM_PARAM_CPUE              (0x00000004)
#define LW0000_CTRL_SYSTEM_PARAM_TMP1              (0x00000005)
#define LW0000_CTRL_SYSTEM_PARAM_TMP2              (0x00000006)
#define LW0000_CTRL_SYSTEM_PARAM_CTGP              (0x00000007)
#define LW0000_CTRL_SYSTEM_PARAM_PPMD              (0x00000008)
#define LW0000_CTRL_SYSTEM_PARAM_COUNT             (0x00000009)

/*
 * LW0000_CTRL_CMD_SYSTEM_EXELWTE_ACPI_METHOD
 *
 * This command is used to execute general ACPI methods.
 *
 *  method
 *    This parameter identifies the MXM ACPI API to be ilwoked. 
 *    Valid values for this parameter are:
 *      LW0000_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_LWOP_OPTIMUSCAPS
 *        This value specifies that the DSM LWOP subfunction OPTIMUSCAPS
 *        API is to be ilwoked.
 *      LW0000_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_LWOP_OPTIMUSFLAG
 *        This value specifies that the DSM LWOP subfunction OPTIMUSFLAG
 *        API is to be ilwoked. This API will set a Flag in sbios to Indicate 
 *        that HD Audio Controller is disable/Enabled from GPU Config space.
 *        This flag will be used by sbios to restore Audio state after resuming
 *        from s3/s4.
 *      LW0000_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_JT_CAPS
 *        This value specifies that the DSM JT subfunction FUNC_CAPS is to
 *        to be ilwoked to get the SBIOS capabilities
 *      LW0000_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_JT_PLATPOLICY
 *        This value specifies that the DSM JT subfunction FUNC_PLATPOLICY is
 *        to be ilwoked to set and get the various platform policies for JT.
 *        Refer to the JT spec in more detail on various policies.
 *  inData
 *    This parameter specifies the method-specific input buffer.  Data is
 *    passed to the specified API using this buffer.
 *  inDataSize
 *    This parameter specifies the size of the inData buffer in bytes.
 *  outStatus
 *    This parameter returns the status code from the associated ACPI call.
 *  outData
 *    This parameter specifies the method-specific output buffer.  Data
 *    is returned by the specified API using this buffer.
 *  outDataSize
 *    This parameter specifies the size of the outData buffer in bytes. 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0000_CTRL_CMD_SYSTEM_EXELWTE_ACPI_METHOD (0x130) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_PARAMS_MESSAGE_ID (0x30U)

typedef struct LW0000_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_PARAMS {
    LwU32 method;
    LW_DECLARE_ALIGNED(LwP64 inData, 8);
    LwU16 inDataSize;
    LwU32 outStatus;
    LW_DECLARE_ALIGNED(LwP64 outData, 8);
    LwU16 outDataSize;
} LW0000_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_PARAMS;

/* valid method parameter values */
#define LW0000_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_LWOP_OPTIMUSCAPS (0x00000000)
#define LW0000_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_LWOP_OPTIMUSFLAG (0x00000001)
#define LW0000_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_JT_CAPS          (0x00000002)
#define LW0000_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_JT_PLATPOLICY    (0x00000003)
/*
 * LW0000_CTRL_CMD_SYSTEM_ENABLE_ETW_EVENTS
 *
 * This command can be used to instruct the RM to enable/disable specific module
 * of ETW events.
 *
 *   moduleMask
 *     This parameter specifies the module of events we would like to 
 *     enable/disable.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0000_CTRL_CMD_SYSTEM_ENABLE_ETW_EVENTS                    (0x131) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_ENABLE_ETW_EVENTS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_ENABLE_ETW_EVENTS_PARAMS_MESSAGE_ID (0x31U)

typedef struct LW0000_CTRL_SYSTEM_ENABLE_ETW_EVENTS_PARAMS {
    LwU32 moduleMask;
} LW0000_CTRL_SYSTEM_ENABLE_ETW_EVENTS_PARAMS;

#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_ALL         (0x00000001)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_NOFREQ      (0x00000002)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_FLUSH       (0x00000004)

#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_PERF        (0x00000010)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_ELPG        (0x00000020)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_LWDPS       (0x00000040)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_POWER       (0x00000080)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_DISP        (0x00000100)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_RMAPI       (0x00000200)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_INTR        (0x00000400)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_LOCK        (0x00000800)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_RCJOURNAL   (0x00001000)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_GENERIC     (0x00002000)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_THERM       (0x00004000)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_GPS         (0x00008000)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_PCIE        (0x00010000)
#define LW0000_CTRL_SYSTEM_RMTRACE_MODULE_LWTELEMETRY (0x00020000)

/*
 * LW0000_CTRL_CMD_SYSTEM_GPS_GET_FRM_DATA
 *
 * This command is used to read FRL data based on need.
 *
 *   nextSampleNumber
 *     This parameter returns the counter of next sample which is being filled.
 *   samples
 *     This parameter returns the frame time, render time, target time, client ID 
 *     with one reserve bit for future use.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0000_CTRL_CMD_SYSTEM_GPS_GET_FRM_DATA       (0x12f) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GPS_GET_FRM_DATA_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GPS_FRM_DATA_SAMPLE_SIZE   64

typedef struct LW0000_CTRL_SYSTEM_GPS_FRM_DATA_SAMPLE {
    LwU16 frameTime;
    LwU16 renderTime;
    LwU16 targetTime;
    LwU8  sleepTime;
    LwU8  sampleNumber;
} LW0000_CTRL_SYSTEM_GPS_FRM_DATA_SAMPLE;

#define LW0000_CTRL_SYSTEM_GPS_GET_FRM_DATA_PARAMS_MESSAGE_ID (0x2FU)

typedef struct LW0000_CTRL_SYSTEM_GPS_GET_FRM_DATA_PARAMS {
    LW0000_CTRL_SYSTEM_GPS_FRM_DATA_SAMPLE samples[LW0000_CTRL_SYSTEM_GPS_FRM_DATA_SAMPLE_SIZE];
    LwU8                                   nextSampleNumber;
} LW0000_CTRL_SYSTEM_GPS_GET_FRM_DATA_PARAMS;

/*
 * LW0000_CTRL_CMD_SYSTEM_GPS_SET_FRM_DATA
 *
 * This command is used to write FRM data based on need.
 *
 *   frameTime
 *     This parameter contains the frame time of current frame.
 *   renderTime
 *     This parameter contains the render time of current frame.
 *   targetTime
 *     This parameter contains the target time of current frame.
 *   sleepTime
 *     This parameter contains the sleep duration inserted by FRL for the latest frame.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0000_CTRL_CMD_SYSTEM_GPS_SET_FRM_DATA (0x132) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GPS_SET_FRM_DATA_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GPS_SET_FRM_DATA_PARAMS_MESSAGE_ID (0x32U)

typedef struct LW0000_CTRL_SYSTEM_GPS_SET_FRM_DATA_PARAMS {
    LW0000_CTRL_SYSTEM_GPS_FRM_DATA_SAMPLE sampleData;
} LW0000_CTRL_SYSTEM_GPS_SET_FRM_DATA_PARAMS;

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_VGX_SYSTEM_INFO
 *
 * This command returns the current host driver, host OS and 
 * plugin information. It is only valid when VGX is setup.
 *   szHostDriverVersionBuffer
 *       This field returns the host driver version (LW_VERSION_STRING).
 *   szHostVersionBuffer
 *       This field returns the host driver version (LW_BUILD_BRANCH_VERSION).
 *   szHostTitleBuffer
 *       This field returns the host driver title (LW_DISPLAY_DRIVER_TITLE).
 *   szPluginTitleBuffer
 *       This field returns the plugin build title (LW_DISPLAY_DRIVER_TITLE).
 *   szHostUnameBuffer
 *       This field returns the call of 'uname' on the host OS.
 *   iHostChangelistNumber
 *       This field returns the changelist value of the host driver (LW_BUILD_CHANGELIST_NUM).
 *   iPluginChangelistNumber
 *       This field returns the changelist value of the plugin (LW_BUILD_CHANGELIST_NUM).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW0000_CTRL_CMD_SYSTEM_GET_VGX_SYSTEM_INFO_BUFFER_SIZE 256
#define LW0000_CTRL_CMD_SYSTEM_GET_VGX_SYSTEM_INFO             (0x133) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_VGX_SYSTEM_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_VGX_SYSTEM_INFO_PARAMS_MESSAGE_ID (0x33U)

typedef struct LW0000_CTRL_SYSTEM_GET_VGX_SYSTEM_INFO_PARAMS {
    char  szHostDriverVersionBuffer[LW0000_CTRL_CMD_SYSTEM_GET_VGX_SYSTEM_INFO_BUFFER_SIZE];
    char  szHostVersionBuffer[LW0000_CTRL_CMD_SYSTEM_GET_VGX_SYSTEM_INFO_BUFFER_SIZE];
    char  szHostTitleBuffer[LW0000_CTRL_CMD_SYSTEM_GET_VGX_SYSTEM_INFO_BUFFER_SIZE];
    char  szPluginTitleBuffer[LW0000_CTRL_CMD_SYSTEM_GET_VGX_SYSTEM_INFO_BUFFER_SIZE];
    char  szHostUnameBuffer[LW0000_CTRL_CMD_SYSTEM_GET_VGX_SYSTEM_INFO_BUFFER_SIZE];
    LwU32 iHostChangelistNumber;
    LwU32 iPluginChangelistNumber;
} LW0000_CTRL_SYSTEM_GET_VGX_SYSTEM_INFO_PARAMS;

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_GPUS_POWER_STATUS
 *
 * This command returns the power status of the GPUs in the system, successfully attached or not because of
 * insufficient power. It is supported on Kepler and up only.
 *   gpuCount
 *       This field returns the count into the following arrays.
 *   busNumber
 *       This field returns the busNumber of a GPU.
 *   gpuExternalPowerStatus
 *       This field returns the corresponding external power status:
 *          LW0000_CTRL_SYSTEM_GPU_EXTERNAL_POWER_STATUS_CONNECTED
 *          LW0000_CTRL_SYSTEM_GPU_EXTERNAL_POWER_STATUS_NOT_CONNECTED
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0000_CTRL_CMD_SYSTEM_GET_GPUS_POWER_STATUS (0x134) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_GPUS_POWER_STATUS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_GPUS_POWER_STATUS_PARAMS_MESSAGE_ID (0x34U)

typedef struct LW0000_CTRL_SYSTEM_GET_GPUS_POWER_STATUS_PARAMS {
    LwU8 gpuCount;
    LwU8 gpuBus[LW0000_CTRL_SYSTEM_MAX_ATTACHED_GPUS];
    LwU8 gpuExternalPowerStatus[LW0000_CTRL_SYSTEM_MAX_ATTACHED_GPUS];
} LW0000_CTRL_SYSTEM_GET_GPUS_POWER_STATUS_PARAMS;

/* Valid gpuExternalPowerStatus values */
#define LW0000_CTRL_SYSTEM_GPU_EXTERNAL_POWER_STATUS_CONNECTED     0
#define LW0000_CTRL_SYSTEM_GPU_EXTERNAL_POWER_STATUS_NOT_CONNECTED 1

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_PRIVILEGED_STATUS
 * 
 * This command returns the caller's API access privileges using
 * this client handle.
 *
 *   privStatus
 *     This parameter returns a mask of possible access privileges:
 *       LW0000_CTRL_SYSTEM_PRIVILEGED_STATUS_PRIV_USER_FLAG
 *         The caller is running with elevated privileges
 *       LW0000_CTRL_SYSTEM_PRIVILEGED_STATUS_ROOT_HANDLE_FLAG
 *         Client is of LW01_ROOT class.
 *       LW0000_CTRL_SYSTEM_PRIVILEGED_STATUS_PRIV_HANDLE_FLAG
 *         Client has PRIV bit set.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#define LW0000_CTRL_CMD_SYSTEM_GET_PRIVILEGED_STATUS               (0x135) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_PRIVILEGED_STATUS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_PRIVILEGED_STATUS_PARAMS_MESSAGE_ID (0x35U)

typedef struct LW0000_CTRL_SYSTEM_GET_PRIVILEGED_STATUS_PARAMS {
    LwU8 privStatusFlags;
} LW0000_CTRL_SYSTEM_GET_PRIVILEGED_STATUS_PARAMS;


/* Valid privStatus values */
#define LW0000_CTRL_SYSTEM_GET_PRIVILEGED_STATUS_PRIV_USER_FLAG     (0x00000001)
#define LW0000_CTRL_SYSTEM_GET_PRIVILEGED_STATUS_KERNEL_HANDLE_FLAG (0x00000002)
#define LW0000_CTRL_SYSTEM_GET_PRIVILEGED_STATUS_PRIV_HANDLE_FLAG   (0x00000004)

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_FABRIC_STATUS
 *
 * The fabric manager (FM) notifies RM that fabric (system) is ready for peer to
 * peer (P2P) use or still initializing the fabric. This command allows clients
 * to query fabric status to allow P2P operations.
 *
 * Note, on systems where FM isn't used, RM just returns _SKIP.
 *
 * fabricStatus
 *     This parameter returns current fabric status:
 *          LW0000_CTRL_GET_SYSTEM_FABRIC_STATUS_SKIP
 *          LW0000_CTRL_GET_SYSTEM_FABRIC_STATUS_UNINITIALIZED
 *          LW0000_CTRL_GET_SYSTEM_FABRIC_STATUS_IN_PROGRESS
 *          LW0000_CTRL_GET_SYSTEM_FABRIC_STATUS_INITIALIZED
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

typedef enum LW0000_CTRL_GET_SYSTEM_FABRIC_STATUS {
    LW0000_CTRL_GET_SYSTEM_FABRIC_STATUS_SKIP = 1,
    LW0000_CTRL_GET_SYSTEM_FABRIC_STATUS_UNINITIALIZED = 2,
    LW0000_CTRL_GET_SYSTEM_FABRIC_STATUS_IN_PROGRESS = 3,
    LW0000_CTRL_GET_SYSTEM_FABRIC_STATUS_INITIALIZED = 4,
} LW0000_CTRL_GET_SYSTEM_FABRIC_STATUS;

#define LW0000_CTRL_CMD_SYSTEM_GET_FABRIC_STATUS (0x136) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_FABRIC_STATUS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_FABRIC_STATUS_PARAMS_MESSAGE_ID (0x36U)

typedef struct LW0000_CTRL_SYSTEM_GET_FABRIC_STATUS_PARAMS {
    LW0000_CTRL_GET_SYSTEM_FABRIC_STATUS fabricStatus;
} LW0000_CTRL_SYSTEM_GET_FABRIC_STATUS_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0000_CTRL_VGPU_GET_VGPU_VERSION_INFO
 *
 * This command is used to query the range of VGX version supported.
 *
 *  host_min_supported_version
 *     The minimum vGPU version supported by host driver
 *  host_max_supported_version
 *     The maximum vGPU version supported by host driver
 *  user_min_supported_version
 *     The minimum vGPU version set by user for vGPU support
 *  user_max_supported_version
 *     The maximum vGPU version set by user for vGPU support
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_REQUEST
 */
#define LW0000_CTRL_VGPU_GET_VGPU_VERSION (0x137) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_VGPU_GET_VGPU_VERSION_PARAMS_MESSAGE_ID" */

/*
 * LW0000_CTRL_VGPU_GET_VGPU_VERSION
 */
#define LW0000_CTRL_VGPU_GET_VGPU_VERSION_PARAMS_MESSAGE_ID (0x37U)

typedef struct LW0000_CTRL_VGPU_GET_VGPU_VERSION_PARAMS {
    LwU32 host_min_supported_version;
    LwU32 host_max_supported_version;
    LwU32 user_min_supported_version;
    LwU32 user_max_supported_version;
} LW0000_CTRL_VGPU_GET_VGPU_VERSION_PARAMS;

/*
 * LW0000_CTRL_VGPU_SET_VGPU_VERSION
 *
 * This command is used to query whether pGPU is live migration capable or not.
 *
 *  min_version
 *      The minimum vGPU version to be supported being set
 *  max_version
 *      The maximum vGPU version to be supported being set
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_REQUEST
 */
#define LW0000_CTRL_VGPU_SET_VGPU_VERSION (0x138) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_VGPU_SET_VGPU_VERSION_PARAMS_MESSAGE_ID" */

/*
 * LW0000_CTRL_VGPU_SET_VGPU_VERSION_PARAMS
 */
#define LW0000_CTRL_VGPU_SET_VGPU_VERSION_PARAMS_MESSAGE_ID (0x38U)

typedef struct LW0000_CTRL_VGPU_SET_VGPU_VERSION_PARAMS {
    LwU32 min_version;
    LwU32 max_version;
} LW0000_CTRL_VGPU_SET_VGPU_VERSION_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0000_CTRL_SYSTEM_GET_RM_INSTANCE_ID
 *
 * This command is used to get a unique identifier for the instance of RM.
 * The returned value will only change when the driver is reloaded. A previous
 * value will never be reused on a given machine.
 *
 *  rm_instance_id;
 *      The instance ID of the current RM instance
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_RM_INSTANCE_ID (0x139) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_RM_INSTANCE_ID_PARAMS_MESSAGE_ID" */

/*
 * LW0000_CTRL_SYSTEM_GET_RM_INSTANCE_ID_PARAMS
 */
#define LW0000_CTRL_SYSTEM_GET_RM_INSTANCE_ID_PARAMS_MESSAGE_ID (0x39U)

typedef struct LW0000_CTRL_SYSTEM_GET_RM_INSTANCE_ID_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 rm_instance_id, 8);
} LW0000_CTRL_SYSTEM_GET_RM_INSTANCE_ID_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0000_CTRL_CMD_SYSTEM_LWPCF_GET_POWER_MODE_INFO
 *
 * This API is used to get the TPP(total processing power) and 
 * the rated TGP(total GPU power) from SBIOS.
 *
 * LWPCF is an acronym for Lwpu Platform Controllers and Framework 
 * which implements platform level policies. LWPCF is implemented in
 * a kernel driver on windows. It is implemented in a user mode app 
 * called lwpu-powerd on Linux.
 * 
 *  gpuId
 *      GPU ID
 *  tpp
 *      Total processing power including CPU and GPU
 *  ratedTgp
 *      Rated total GPU Power
 *  subFunc
 *      LWPCF subfunction id
 *  ctgpOffsetmW
 *      Configurable TGP offset, in mW
 *  targetTppOffsetmW
 *      TPP, as offset in mW.
 *  maxOutputOffsetmW
 *      Maximum allowed output, as offset in mW.
 *  minOutputOffsetmW;
 *      Minimum allowed output, as offset in mW.
 *
 *   Valid subFunc ids for LWPCF 1x include :
 *   LWPCF0100_CTRL_CONFIG_DSM_1X_FUNC_GET_SUPPORTED
 *   LWPCF0100_CTRL_CONFIG_DSM_1X_FUNC_GET_DYNAMIC_PARAMS
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_REQUEST
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0000_CTRL_CMD_SYSTEM_LWPCF_GET_POWER_MODE_INFO (0x13b) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_CMD_SYSTEM_LWPCF_GET_POWER_MODE_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_CMD_SYSTEM_LWPCF_GET_POWER_MODE_INFO_PARAMS_MESSAGE_ID (0x3BU)

typedef struct LW0000_CTRL_CMD_SYSTEM_LWPCF_GET_POWER_MODE_INFO_PARAMS {
    LwU32 gpuId;
    LwU32 tpp;
    LwU32 ratedTgp;
    LwU32 subFunc;
    LwU32 ctgpOffsetmW;
    LwU32 targetTppOffsetmW;
    LwU32 maxOutputOffsetmW;
    LwU32 minOutputOffsetmW;
} LW0000_CTRL_CMD_SYSTEM_LWPCF_GET_POWER_MODE_INFO_PARAMS;
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/* Valid LWPCF subfunction case */
#define LWPCF0100_CTRL_CONFIG_DSM_1X_FUNC_GET_SUPPORTED_CASE 0
#define LWPCF0100_CTRL_CONFIG_DSM_1X_FUNC_GET_DYNAMIC_CASE   1
#define LWPCF0100_CTRL_CONFIG_DSM_2X_FUNC_GET_SUPPORTED_CASE 2
#define LWPCF0100_CTRL_CONFIG_DSM_2X_FUNC_GET_DYNAMIC_CASE   3

/* Valid LWPCF subfunction ids */
#define LWPCF0100_CTRL_CONFIG_DSM_1X_FUNC_GET_SUPPORTED      (0x00000000)
#define LWPCF0100_CTRL_CONFIG_DSM_1X_FUNC_GET_DYNAMIC_PARAMS (0x00000002)

/*
 *  Defines for get supported sub functions bit fields
 */
#define LWPCF0100_CTRL_CONFIG_DSM_FUNC_GET_SUPPORTED_IS_SUPPORTED        0:0
#define LWPCF0100_CTRL_CONFIG_DSM_FUNC_GET_SUPPORTED_IS_SUPPORTED_YES    1
#define LWPCF0100_CTRL_CONFIG_DSM_FUNC_GET_SUPPORTED_IS_SUPPORTED_NO     0

/*!
 * Config DSM 2x version specific defines
 */
#define LWPCF0100_CTRL_CONFIG_DSM_2X_VERSION                 (0x00000200)
#define LWPCF0100_CTRL_CONFIG_DSM_2X_FUNC_GET_SUPPORTED      (0x00000000)
#define LWPCF0100_CTRL_CONFIG_DSM_2X_FUNC_GET_DYNAMIC_PARAMS (0x00000002)


/*
 * LW0000_CTRL_CMD_SYSTEM_SYNC_EXTERNAL_FABRIC_MGMT
 *
 * This API is used to sync the external fabric management status with
 * GSP-RM
 *
 *  bExternalFabricMgmt
 *      Whether fabric is externally managed
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0000_CTRL_CMD_SYSTEM_SYNC_EXTERNAL_FABRIC_MGMT     (0x13c) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_CMD_SYSTEM_SYNC_EXTERNAL_FABRIC_MGMT_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_CMD_SYSTEM_SYNC_EXTERNAL_FABRIC_MGMT_PARAMS_MESSAGE_ID (0x3LW)

typedef struct LW0000_CTRL_CMD_SYSTEM_SYNC_EXTERNAL_FABRIC_MGMT_PARAMS {
    LwBool bExternalFabricMgmt;
} LW0000_CTRL_CMD_SYSTEM_SYNC_EXTERNAL_FABRIC_MGMT_PARAMS;

/*
 * LW0000_CTRL_SYSTEM_GET_CLIENT_DATABASE_INFO
 *
 * This API is used to get information about the RM client
 * database.
 *
 * clientCount [OUT]
 *  This field indicates the number of clients lwrrently allocated.
 *
 * resourceCount [OUT]
 *  This field indicates the number of resources lwrrently allocated
 *  across all clients.
 *
 */
#define LW0000_CTRL_CMD_SYSTEM_GET_CLIENT_DATABASE_INFO (0x13d) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_CLIENT_DATABASE_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_CLIENT_DATABASE_INFO_PARAMS_MESSAGE_ID (0x3DU)

typedef struct LW0000_CTRL_SYSTEM_GET_CLIENT_DATABASE_INFO_PARAMS {
    LwU32 clientCount;
    LW_DECLARE_ALIGNED(LwU64 resourceCount, 8);
} LW0000_CTRL_SYSTEM_GET_CLIENT_DATABASE_INFO_PARAMS;

/*
 * LW0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2
 *
 * This command returns the current driver information in
 * statically sized character arrays.
 *
 *   driverVersionBuffer
 *       This field returns the version (LW_VERSION_STRING).
 *   versionBuffer
 *       This field returns the version (LW_BUILD_BRANCH_VERSION).
 *   titleBuffer
 *       This field returns the title (LW_DISPLAY_DRIVER_TITLE).
 *   changelistNumber
 *       This field returns the changelist value (LW_BUILD_CHANGELIST_NUM).
 *   officialChangelistNumber
 *       This field returns the last official changelist value
 *       (LW_LAST_OFFICIAL_CHANGELIST_NUM).
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_MAX_STRING_SIZE 256
#define LW0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2             (0x13e) /* finn: Evaluated from "(FINN_LW01_ROOT_SYSTEM_INTERFACE_ID << 8) | LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_PARAMS_MESSAGE_ID (0x3EU)

typedef struct LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_PARAMS {
    char  driverVersionBuffer[LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_MAX_STRING_SIZE];
    char  versionBuffer[LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_MAX_STRING_SIZE];
    char  titleBuffer[LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_MAX_STRING_SIZE];
    LwU32 changelistNumber;
    LwU32 officialChangelistNumber;
} LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_PARAMS;

/* _ctrl0000system_h_ */
