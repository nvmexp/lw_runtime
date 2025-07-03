/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080fb.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl2080/ctrl2080base.h"

/* LW20_SUBDEVICE_XX fb control commands and parameters */

#include "ctrl2080common.h"
#include "lwlimits.h"

/*
 * LW2080_CTRL_FB_INFO
 *
 * This structure represents a single 32bit fb engine value.  Clients
 * request a particular fb engine value by specifying a unique fb
 * information index.
 *
 * Legal fb information index values are:
 *   LW2080_CTRL_FB_INFO_INDEX_TILE_REGION_COUNT
 *     This index is used to request the number of tiled regions supported
 *     by the associated subdevice.  The return value is GPU
 *     implementation-dependent.  A return value of 0 indicates the GPU
 *     does not support tiling.
 *   LW2080_CTRL_FB_INFO_INDEX_COMPRESSION_SIZE
 *     This index is used to request the amount of compression (in bytes)
 *     supported by the associated subdevice.  The return value is GPU
 *     implementation-dependent.  A return value of 0 indicates the GPU
 *     does not support compression.
 *   Lw2080_CTRL_FB_INFO_INDEX_DRAM_PAGE_STRIDE
 *     This index is used to request the DRAM page stride (in bytes)
 *     supported by the associated subdevice.  The return value is GPU
 *     implementation-dependent.
 *   LW2080_CTRL_FB_INFO_INDEX_TILE_REGION_FREE_COUNT
 *     This index is used to request the number of free tiled regions on
 *     the associated subdevice.  The return value represents the current
 *     number of free tiled regions at the time the command is processed and
 *     is not guaranteed to remain unchanged.  A return value of 0 indicates
 *     that there are no available tiled regions on the associated subdevice.
 *   LW2080_CTRL_FB_INFO_INDEX_PARTITION_COUNT
 *     This index is used to request the number of frame buffer partitions
 *     on the associated subdevice. Starting with Fermi there are now two units
 *     with the name framebuffer partitions. On those chips this index returns
 *     the number of FBPAs. For number of FBPs use
 *     LW2080_CTRL_FB_INFO_INDEX_FBP_COUNT.
 *     This an SMC aware attribute, thus necessary partition subscription is
 *     required if the device is partitioned.
 *   LW2080_CTRL_FB_INFO_INDEX_RAM_SIZE
 *     This index is used to request the amount of framebuffer memory in
 *     kilobytes physically present on the associated subdevice.  This
 *     value will never exceed the value reported by
 *     LW2080_CTRL_FB_INFO_INDEX_TOTAL_RAM_SIZE.
 *     This an SMC aware attribute, so the per-partition framebuffer memory
 *     size will be returned when the client has a partition subscription.
 *   LW2080_CTRL_FB_INFO_INDEX_TOTAL_RAM_SIZE
 *     This index is used to request the total amount of video memory in
 *     kilobytes for use with the associated subdevice.  This value will
 *     reflect both framebuffer memory as well as any system memory dedicated
 *     for use with the subdevice.
 *     This an SMC aware attribute, so the per-partition video memory size
 *     will be returned when the client has a partition subscription.
 *   LW2080_CTRL_FB_INFO_INDEX_HEAP_SIZE
 *     This index is used to request the amount of total RAM in kilobytes
 *     available for user allocations.  This value reflects the total ram
 *     size less the amount of memory reserved for internal use.
 *     This an SMC aware attribute, thus necessary partition subscription is
 *     required if the device is partitioned.
 *   LW2080_CTRL_FB_INFO_INDEX_HEAP_START
 *     This index is used to request the offset for start of heap in
 *     kilobytes.
 *     This an SMC aware attribute, thus necessary partition subscription is
 *     required if the device is partitioned.
 *   LW2080_CTRL_FB_INFO_INDEX_HEAP_FREE
 *     This index is used to request the available amount of video memory in
 *     kilobytes for use with the associated subdevice or the SMC partition.
 *     This an SMC aware attribute, thus necessary partition subscription is
 *     required to query per partition information, if the device is partitioned.
 *     Alternatively, the SMC/MIG monitor capability can be acquired to query
 *     aggregate available memory across all the valid partitions.
 *   LW2080_CTRL_FB_INFO_INDEX_MAPPABLE_HEAP_SIZE
 *     This index reflects the amount of heap memory in kilobytes that
 *     is accessible by the CPU.  On subdevices with video memory sizes that
 *     exceed the amount that can be bus mappable this value will be less
 *     than that reported by LW2080_CTRL_FB_INFO_INDEX_HEAP_SIZE.
 *     This an SMC aware attribute, thus necessary partition subscription is
 *     required if the device is partitioned.
 *   LW2080_CTRL_FB_INFO_INDEX_BUS_WIDTH
 *     This index is used to request the FB bus bandwidth on the associated
 *     subdevice.
 *   LW2080_CTRL_FB_INFO_INDEX_RAM_CFG
 *     This index is used to request the implementation-dependent RAM
 *     configuration value of the associated subdevice.
 *   LW2080_CTRL_FB_INFO_INDEX_RAM_TYPE
 *     This index is used to request the type of RAM used for the framebuffer
 *     on the associated subdevice.  Legal RAM types include:
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_UNKNOWN
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_SDRAM
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_DDR1
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_DDR2
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR2
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR3
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR4
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_DDR3
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR5
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR5X
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR6
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR6X
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_LPDDR2
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_LPDDR4
 *       LW2080_CTRL_FB_INFO_RAM_TYPE_LPDDR5
 *   LW2080_CTRL_FB_INFO_INDEX_BANK_COUNT
 *     This index is used to request the number of FB banks on the associated
 *     subdevice.
 *   LW2080_CTRL_FB_INFO_INDEX_OVERLAY_OFFSET_ADJUSTMENT
 *     This index is used to request the offset relative to the start of the
 *     overlay surface(s), in bytes, at which scanout should happen if the
 *     primary and the overlay surfaces are all aligned on large page
 *     boundaries.
 *   LW2080_CTRL_FB_INFO_INDEX_GPU_VADDR_SPACE_SIZE_KB
 *     This index is used to request the size of the GPU's virtual address
 *     space in kilobytes.
 *   LW2080_CTRL_FB_INFO_INDEX_GPU_VADDR_HEAP_SIZE_KB
 *     This index is used to request the size of the GPU's virtual address
 *     space heap (minus RM-reserved space) in kilobytes.
 *   LW2080_CTRL_FB_INFO_INDEX_GPU_VADDR_MAPPBLE_SIZE_KB
 *     This index is used to request the size of the GPU's BAR1 mappable
 *     virtual address space in kilobytes.
 *   LW2080_CTRL_FB_INFO_INDEX_EFFECTIVE_BW
 *     This index is used to request the effective bandwidth of the FB in
 *     MBytes/sec. This value reflects the link bandwidth for the current mode.
 *     Lwrrently only implemented for tesla based GPUs with carveout.
 *   LW2080_CTRL_FB_INFO_INDEX_PARTITION_MASK
 *     This index is used to request the mask of lwrrently active partitions.
 *     Each  active partition has an ID that's equivalent to the corresponding
 *     bit position in the mask.
 *     This an SMC aware attribute, thus necessary partition subscription is
 *     required if the device is partitioned.
 *   LW2080_CTRL_FB_INFO_INDEX_VISTA_RESERVED_HEAP_SIZE
 *     This index is used to request the amount of total RAM in kilobytes
 *     reserved for internal RM allocations on Vista.  This will need to
 *     be subtracted from the total heap size to get the amount available to
 *     KMD.
 *     This an SMC aware attribute, thus necessary partition subscription is
 *     required if the device is partitioned.
 *   LW2080_CTRL_FB_INFO_INDEX_RAM_LOCATION
 *     This index is used to distinguish between different memory
 *     configurations.
 *   LW2080_CTRL_FB_INFO_INDEX_FB_IS_BROKEN
 *     This index is used to check if the FB is functional
 *   LW2080_CTRL_FB_INFO_INDEX_FBP_COUNT
 *     This index is used to get the number of FBPs on the subdevice. This
 *     field is not to be confused with
 *     LW2080_CTRL_FB_INFO_INDEX_PARTITION_COUNT (returns number of FBPAs).
 *     Starting with Fermi the term partition is an ambiguous term, both FBP
 *     and FBPA mean FB partitions. The FBPA is the low level DRAM controller,
 *     while a FBP is the aggregation of one or more FBPAs, L2, ROP, and some
 *     other units.
 *     This an SMC aware attribute, thus necessary partition subscription is
 *     required if the device is partitioned.
 *   LW2080_CTRL_FB_INFO_INDEX_L2CACHE_SIZE
 *     This index is used to get the size of the L2 cache in Bytes.
 *     A value of zero indicates that the L2 cache isn't supported on the
 *     associated subdevice.
 *   LW2080_CTRL_FB_INFO_INDEX_MEMORYINFO_VENDOR_ID
 *     This index is used to get the memory vendor ID information from
 *     the Memory Information Table in the VBIOS.  Legal memory Vendor ID
 *     values include:
 *       LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_UNKNOWN
 *       LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_RESERVED
 *       LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_SAMSUNG
 *       LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_QIMONDA
 *       LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_ELPIDA
 *       LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_ETRON
 *       LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_NANYA
 *       LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_HYNIX
 *       LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_MOSEL
 *       LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_WINBOND
 *       LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_ESMT
 *       LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_MICRON
 *   LW2080_CTRL_FB_INFO_INDEX_BAR1_AVAIL_SIZE
 *     This index is used to request the amount of unused bar1 space. The
 *     data returned is a value in KB. It is not guaranteed to be entirely
 *     accurate since it is a snapshot at a particular time and can
 *     change quickly.
 *   LW2080_CTRL_FB_INFO_INDEX_BAR1_MAX_CONTIGUOUS_AVAIL_SIZE
 *     This index is used to request the amount of largest unused contiguous
 *     block in bar1 space.  The data returned is a value in KB. It is not
 *     guaranteed to be entirely accurate since it is a snapshot at a particular
 *     time and can change quickly.
 *   LW2080_CTRL_FB_INFO_INDEX_USABLE_RAM_SIZE
 *     This index is used to request the amount of usable framebuffer memory in
 *     kilobytes physically present on the associated subdevice.  This
 *     value will never exceed the value reported by
 *     LW2080_CTRL_FB_INFO_INDEX_TOTAL_RAM_SIZE.
 *     This an SMC aware attribute, thus necessary partition subscription is
 *     required if the device is partitioned.
 *   LW2080_CTRL_FB_INFO_INDEX_LTC_COUNT
 *     Returns the active LTC count across all active FBPs.
 *     This an SMC aware attribute, thus necessary partition subscription is
 *     required if the device is partitioned.
 *   LW2080_CTRL_FB_INFO_INDEX_LTS_COUNT
 *     Returns the active LTS count across all active LTCs.
 *     This an SMC aware attribute, thus necessary partition subscription is
 *     required if the device is partitioned.
 *   LW2080_CTRL_FB_INFO_INDEX_PSEUDO_CHANNEL_MODE
 *     This is used to identify if pseudo-channel mode is enabled for HBM
 *   LW2080_CTRL_FB_INFO_INDEX_SMOOTHDISP_RSVD_BAR1_SIZE
 *     This is used by WDDM-KMD to determine whether and how much RM reserved BAR1 for smooth transition
 *   LW2080_CTRL_FB_INFO_INDEX_HEAP_OFFLINE_SIZE
 *     Returns the total size of the all dynamically offlined pages in KiB
 *   LW2080_CTRL_FB_INFO_INDEX_1TO1_COMPTAG_ENABLED
 *     Returns true if 1to1 comptag is enabled
 *   LW2080_CTRL_FB_INFO_INDEX_SUSPEND_RESUME_RSVD_SIZE
 *     Returns the total size of the memory(FB) that will saved/restored during save/restore cycle
 *   LW2080_CTRL_FB_INFO_INDEX_ALLOW_PAGE_RETIREMENT
 *     Returns true if page retirement is allowed
 *   LW2080_CTRL_FB_INFO_POISON_FUSE_ENABLED
 *     Returns true if poison fuse is enabled
 *   LW2080_CTRL_FB_INFO_FBPA_ECC_ENABLED
 *     Returns true if ECC is enabled for FBPA
 *   LW2080_CTRL_FB_INFO_DYNAMIC_PAGE_OFFLINING_ENABLED
 *     Returns true if dynamic page blacklisting is enabled
 *   LW2080_CTRL_FB_INFO_INDEX_FORCED_BAR1_64KB_MAPPING_ENABLED
 *     Returns true if 64KB mapping on BAR1 is force-enabled
 *   LW2080_CTRL_FB_INFO_INDEX_P2P_MAILBOX_SIZE
 *     Returns the P2P mailbox size to be allocated by the client. 
 *     Returns 0 if the P2P mailbox is allocated by RM.
 *   LW2080_CTRL_FB_INFO_INDEX_P2P_MAILBOX_ALIGNMENT_SIZE
 *     Returns the P2P mailbox alignment requirement.
 *     Returns 0 if the P2P mailbox is allocated by RM.
 *   LW2080_CTRL_FB_INFO_INDEX_P2P_MAILBOX_BAR1_MAX_OFFSET_64KB
 *     Returns the P2P mailbox max offset requirement.
 *     Returns 0 if the P2P mailbox is allocated by RM.
 *   LW2080_CTRL_FB_INFO_INDEX_PROTECTED_MEM_SIZE_TOTAL_KB
 *     Returns total protected memory when memory protection is enabled
 *     Returns 0 when memory protection is not enabled.
 *   LW2080_CTRL_FB_INFO_INDEX_PROTECTED_MEM_SIZE_FREE_KB
 *     Returns protected memory available for allocation when memory
 *     protection is enabled.
 *     Returns 0 when memory protection is not enabled.
 */
typedef struct LW2080_CTRL_FB_INFO {
    LwU32 index;
    LwU32 data;
} LW2080_CTRL_FB_INFO;

/* valid fb info index values */
#define LW2080_CTRL_FB_INFO_INDEX_TILE_REGION_COUNT                (0x00000000) // Deprecated
#define LW2080_CTRL_FB_INFO_INDEX_COMPRESSION_SIZE                 (0x00000001)
#define LW2080_CTRL_FB_INFO_INDEX_DRAM_PAGE_STRIDE                 (0x00000002)
#define LW2080_CTRL_FB_INFO_INDEX_TILE_REGION_FREE_COUNT           (0x00000003)
#define LW2080_CTRL_FB_INFO_INDEX_PARTITION_COUNT                  (0x00000004)
#define LW2080_CTRL_FB_INFO_INDEX_BAR1_SIZE                        (0x00000005)
#define LW2080_CTRL_FB_INFO_INDEX_BANK_SWIZZLE_ALIGNMENT           (0x00000006)
#define LW2080_CTRL_FB_INFO_INDEX_RAM_SIZE                         (0x00000007)
#define LW2080_CTRL_FB_INFO_INDEX_TOTAL_RAM_SIZE                   (0x00000008)
#define LW2080_CTRL_FB_INFO_INDEX_HEAP_SIZE                        (0x00000009)
#define LW2080_CTRL_FB_INFO_INDEX_MAPPABLE_HEAP_SIZE               (0x0000000A)
#define LW2080_CTRL_FB_INFO_INDEX_BUS_WIDTH                        (0x0000000B)
#define LW2080_CTRL_FB_INFO_INDEX_RAM_CFG                          (0x0000000C)
#define LW2080_CTRL_FB_INFO_INDEX_RAM_TYPE                         (0x0000000D)
#define LW2080_CTRL_FB_INFO_INDEX_BANK_COUNT                       (0x0000000E)
#define LW2080_CTRL_FB_INFO_INDEX_OVERLAY_OFFSET_ADJUSTMENT        (0x0000000F) // Deprecated (index reused to return 0)
#define LW2080_CTRL_FB_INFO_INDEX_GPU_VADDR_SPACE_SIZE_KB          (0x0000000F) // Deprecated (index reused to return 0)
#define LW2080_CTRL_FB_INFO_INDEX_GPU_VADDR_HEAP_SIZE_KB           (0x0000000F) // Deprecated (index reused to return 0)
#define LW2080_CTRL_FB_INFO_INDEX_GPU_VADDR_MAPPBLE_SIZE_KB        (0x0000000F) // Deprecated (index reused to return 0)
#define LW2080_CTRL_FB_INFO_INDEX_EFFECTIVE_BW                     (0x0000000F) // Deprecated (index reused to return 0)
#define LW2080_CTRL_FB_INFO_INDEX_FB_TAX_SIZE_KB                   (0x00000010)
#define LW2080_CTRL_FB_INFO_INDEX_HEAP_BASE_KB                     (0x00000011)
#define LW2080_CTRL_FB_INFO_INDEX_LARGEST_FREE_REGION_SIZE_KB      (0x00000012)
#define LW2080_CTRL_FB_INFO_INDEX_LARGEST_FREE_REGION_BASE_KB      (0x00000013)
#define LW2080_CTRL_FB_INFO_INDEX_PARTITION_MASK                   (0x00000014)
#define LW2080_CTRL_FB_INFO_INDEX_VISTA_RESERVED_HEAP_SIZE         (0x00000015)
#define LW2080_CTRL_FB_INFO_INDEX_HEAP_FREE                        (0x00000016)
#define LW2080_CTRL_FB_INFO_INDEX_RAM_LOCATION                     (0x00000017)
#define LW2080_CTRL_FB_INFO_INDEX_FB_IS_BROKEN                     (0x00000018)
#define LW2080_CTRL_FB_INFO_INDEX_FBP_COUNT                        (0x00000019)
#define LW2080_CTRL_FB_INFO_INDEX_FBP_MASK                         (0x0000001A)
#define LW2080_CTRL_FB_INFO_INDEX_L2CACHE_SIZE                     (0x0000001B)
#define LW2080_CTRL_FB_INFO_INDEX_MEMORYINFO_VENDOR_ID             (0x0000001C)
#define LW2080_CTRL_FB_INFO_INDEX_BAR1_AVAIL_SIZE                  (0x0000001D)
#define LW2080_CTRL_FB_INFO_INDEX_HEAP_START                       (0x0000001E)
#define LW2080_CTRL_FB_INFO_INDEX_BAR1_MAX_CONTIGUOUS_AVAIL_SIZE   (0x0000001F)
#define LW2080_CTRL_FB_INFO_INDEX_USABLE_RAM_SIZE                  (0x00000020)
#define LW2080_CTRL_FB_INFO_INDEX_TRAINIG_2T                       (0x00000021)
#define LW2080_CTRL_FB_INFO_INDEX_LTC_COUNT                        (0x00000022)
#define LW2080_CTRL_FB_INFO_INDEX_LTS_COUNT                        (0x00000023)
#define LW2080_CTRL_FB_INFO_INDEX_L2CACHE_ONLY_MODE                (0x00000024)
#define LW2080_CTRL_FB_INFO_INDEX_PSEUDO_CHANNEL_MODE              (0x00000025)
#define LW2080_CTRL_FB_INFO_INDEX_SMOOTHDISP_RSVD_BAR1_SIZE        (0x00000026)
#define LW2080_CTRL_FB_INFO_INDEX_HEAP_OFFLINE_SIZE                (0x00000027)
#define LW2080_CTRL_FB_INFO_INDEX_1TO1_COMPTAG_ENABLED             (0x00000028)
#define LW2080_CTRL_FB_INFO_INDEX_SUSPEND_RESUME_RSVD_SIZE         (0x00000029)
#define LW2080_CTRL_FB_INFO_INDEX_ALLOW_PAGE_RETIREMENT            (0x0000002A)
#define LW2080_CTRL_FB_INFO_INDEX_LTC_MASK                         (0x0000002B)
#define LW2080_CTRL_FB_INFO_POISON_FUSE_ENABLED                    (0x0000002C)
#define LW2080_CTRL_FB_INFO_FBPA_ECC_ENABLED                       (0x0000002D)
#define LW2080_CTRL_FB_INFO_DYNAMIC_PAGE_OFFLINING_ENABLED         (0x0000002E)
#define LW2080_CTRL_FB_INFO_INDEX_FORCED_BAR1_64KB_MAPPING_ENABLED (0x0000002F)
#define LW2080_CTRL_FB_INFO_INDEX_P2P_MAILBOX_SIZE                 (0x00000030)
#define LW2080_CTRL_FB_INFO_INDEX_P2P_MAILBOX_ALIGNMENT            (0x00000031)
#define LW2080_CTRL_FB_INFO_INDEX_P2P_MAILBOX_BAR1_MAX_OFFSET_64KB (0x00000032)
#define LW2080_CTRL_FB_INFO_INDEX_PROTECTED_MEM_SIZE_TOTAL_KB      (0x00000033)
#define LW2080_CTRL_FB_INFO_INDEX_PROTECTED_MEM_SIZE_FREE_KB       (0x00000034)
#define LW2080_CTRL_FB_INFO_MAX_LIST_SIZE                          (0x00000035)

#define LW2080_CTRL_FB_INFO_INDEX_MAX                              (0x34) /* finn: Evaluated from "(LW2080_CTRL_FB_INFO_MAX_LIST_SIZE - 1)" */

/* valid fb RAM type values */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_UNKNOWN                       (0x00000000)
#define LW2080_CTRL_FB_INFO_RAM_TYPE_SDRAM                         (0x00000001)
#define LW2080_CTRL_FB_INFO_RAM_TYPE_DDR1                          (0x00000002) /* SDDR and GDDR (aka DDR1 and GDDR1) */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_SDDR2                         (0x00000003) /* SDDR2 Used on LW43 and later */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_DDR2                          LW2080_CTRL_FB_INFO_RAM_TYPE_SDDR2 /* Deprecated alias */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR2                         (0x00000004) /* GDDR2 Used on LW30 and some LW36 */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR3                         (0x00000005) /* GDDR3 Used on LW40 and later */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR4                         (0x00000006) /* GDDR4 Used on G80 and later (deprecated) */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_SDDR3                         (0x00000007) /* SDDR3 Used on G9x and later */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_DDR3                          LW2080_CTRL_FB_INFO_RAM_TYPE_SDDR3 /* Deprecated alias */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR5                         (0x00000008) /* GDDR5 Used on GT21x and later */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_LPDDR2                        (0x00000009) /* LPDDR (Low Power SDDR) used on T2x and later. */
/*
 * Note: _GDDR3_BGA144 and _GDDR3_BGA136 are not returned by the LW2080 API,
 * although they are used internally by the LWPU Resource Manager.
 */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR3_BGA144                  (0x0000000A) /* Not used in API */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR3_BGA136                  (0x0000000B) /* Not used in API */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_SDDR4                         (0x0000000C) /* SDDR4 Used on Maxwell and later */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_LPDDR4                        (0x0000000D) /* LPDDR (Low Power SDDR) used on T21x and later.*/
#define LW2080_CTRL_FB_INFO_RAM_TYPE_HBM1                          (0x0000000E) /* HBM1 (High Bandwidth Memory) used on GP100 */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_HBM2                          (0x0000000F) /* HBM2 (High Bandwidth Memory-pseudo channel) */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR5X                        (0x00000010) /* GDDR5X Used on GP10x */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR6                         (0x00000011) /* GDDR6 Used on TU10x */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_GDDR6X                        (0x00000012) /* GDDR6X Used on GA10x */
#define LW2080_CTRL_FB_INFO_RAM_TYPE_LPDDR5                        (0x00000013) /* LPDDR (Low Power SDDR) used on T23x and later.*/
#define LW2080_CTRL_FB_INFO_RAM_TYPE_HBM3                          (0x00000014) /* HBM3 (High Bandwidth Memory) v3 */

/* valid RAM LOCATION types */
#define LW2080_CTRL_FB_INFO_RAM_LOCATION_GPU_DEDICATED             (0x00000000)
#define LW2080_CTRL_FB_INFO_RAM_LOCATION_SYS_SHARED                (0x00000001)
#define LW2080_CTRL_FB_INFO_RAM_LOCATION_SYS_DEDICATED             (0x00000002)

/* valid Memory Vendor ID values */
#define LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_SAMSUNG           (0x00000001)
#define LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_QIMONDA           (0x00000002)
#define LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_ELPIDA            (0x00000003)
#define LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_ETRON             (0x00000004)
#define LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_NANYA             (0x00000005)
#define LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_HYNIX             (0x00000006)
#define LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_MOSEL             (0x00000007)
#define LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_WINBOND           (0x00000008)
#define LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_ESMT              (0x00000009)
#define LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_MICRON            (0x0000000F)
#define LW2080_CTRL_FB_INFO_MEMORYINFO_VENDOR_ID_UNKNOWN           (0xFFFFFFFF)

#define LW2080_CTRL_FB_INFO_PSEUDO_CHANNEL_MODE_UNSUPPORTED        (0x00000000)
#define LW2080_CTRL_FB_INFO_PSEUDO_CHANNEL_MODE_DISABLED           (0x00000001)
#define LW2080_CTRL_FB_INFO_PSEUDO_CHANNEL_MODE_ENABLED            (0x00000002)

/**
 * LW2080_CTRL_CMD_FB_GET_INFO
 *
 * This command returns fb engine information for the associated GPU.
 * Requests to retrieve fb information use a list of one or more
 * LW2080_CTRL_FB_INFO structures.
 *
 *   fbInfoListSize
 *     This field specifies the number of entries on the caller's
 *     fbInfoList.
 *   fbInfoList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer into which the fb information is to be returned.
 *     This buffer must be at least as big as fbInfoListSize multiplied
 *     by the size of the LW2080_CTRL_FB_INFO structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_FB_GET_INFO                                (0x20801301) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_GET_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_FB_GET_INFO_PARAMS {
    LwU32 fbInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 fbInfoList, 8);
} LW2080_CTRL_FB_GET_INFO_PARAMS;

#define LW2080_CTRL_CMD_FB_GET_INFO_V2 (0x20801303) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_GET_INFO_V2_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_FB_GET_INFO_V2_PARAMS {
    LwU32               fbInfoListSize;
    LW2080_CTRL_FB_INFO fbInfoList[LW2080_CTRL_FB_INFO_MAX_LIST_SIZE];
} LW2080_CTRL_FB_GET_INFO_V2_PARAMS;

/*
 * LW2080_CTRL_CMD_FB_GET_TILE_ADDRESS_INFO
 *
 * This command returns tile addressing information.
 *
 *   StartAddr
 *     This parameter returns BAR1 plus the size of the local FB.
 *   SpaceSize
 *     This parameter returns the BAR1 aperture size less the size of the
 *     local FB.
 *
 * Note that both parameters will contain zero if there is no system tile
 * address space.
 */
#define LW2080_CTRL_CMD_FB_GET_TILE_ADDRESS_INFO (0x20801302) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x2" */

typedef struct LW2080_CTRL_FB_GET_SYSTEM_TILE_ADDRESS_SPACE_INFO {
    LW_DECLARE_ALIGNED(LwU64 StartAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 SpaceSize, 8);
} LW2080_CTRL_FB_GET_SYSTEM_TILE_ADDRESS_SPACE_INFO;

/*
 * LW2080_CTRL_CMD_FB_GET_BAR1_OFFSET
 *
 * This command returns the GPU virtual address of a bar1
 * allocation, given the CPU virtual address.
 *
 *   cpuVirtAddress
 *     This field specifies the associated CPU virtual address of the
 *     memory allocation.
 *   gpuVirtAddress
 *     The GPU virtual address associated with the allocation
 *     is returned in this field.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FB_GET_BAR1_OFFSET (0x20801310) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_BAR1_OFFSET_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_GET_BAR1_OFFSET_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW2080_CTRL_FB_GET_BAR1_OFFSET_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 cpuVirtAddress, 8);
    LW_DECLARE_ALIGNED(LwU64 gpuVirtAddress, 8);
} LW2080_CTRL_FB_GET_BAR1_OFFSET_PARAMS;

/*
 * Note: Returns Zeros if no System carveout address info
 *
 * LW2080_CTRL_CMD_FB_GET_CARVEOUT_ADDRESS_INFO
 *
 * This command returns FB carveout address space information
 *
 *   StartAddr
 *     Returns the system memory address of the start of carveout space.
 *   SpaceSize
 *     Returns the size of carveout space.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_FB_GET_CARVEOUT_ADDRESS_INFO (0x2080130b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_SYSTEM_CARVEOUT_ADDRESS_SPACE_INFO_MESSAGE_ID" */

#define LW2080_CTRL_FB_GET_SYSTEM_CARVEOUT_ADDRESS_SPACE_INFO_MESSAGE_ID (0xBU)

typedef struct LW2080_CTRL_FB_GET_SYSTEM_CARVEOUT_ADDRESS_SPACE_INFO {
    LW_DECLARE_ALIGNED(LwU64 StartAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 SpaceSize, 8);
} LW2080_CTRL_FB_GET_SYSTEM_CARVEOUT_ADDRESS_SPACE_INFO;

/*
 * LW2080_CTRL_FB_CMD_GET_CALIBRATION_LOCK_FAILED
 *
 * This command returns the failure counts for calibration.
 *
 *   uFlags
 *     Just one for now -- ehether to reset the counts.
 *   driveStrengthRiseCount
 *     This parameter specifies the failure count for drive strength rising.
 *   driveStrengthFallCount
 *     This parameter specifies the failure count for drive strength falling.
 *   driveStrengthTermCount
 *     This parameter specifies the failure count for drive strength
 *     termination.
 *   slewStrengthRiseCount
 *     This parameter specifies the failure count for slew strength rising.
 *   slewStrengthFallCount
 *     This parameter specifies the failure count for slew strength falling.
 *   slewStrengthTermCount
 *     This parameter specifies the failure count for slew strength
 *     termination.
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LWOS_STATUS_ILWALID_PARAM_STRUCT
 *     LWOS_STATUS_NOT_SUPPORTED
 *     LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FB_GET_CALIBRATION_LOCK_FAILED (0x2080130c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_CALIBRATION_LOCK_FAILED_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_GET_CALIBRATION_LOCK_FAILED_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW2080_CTRL_FB_GET_CALIBRATION_LOCK_FAILED_PARAMS {
    LwU32 flags;
    LwU32 driveStrengthRiseCount;
    LwU32 driveStrengthFallCount;
    LwU32 driveStrengthTermCount;
    LwU32 slewStrengthRiseCount;
    LwU32 slewStrengthFallCount;
} LW2080_CTRL_FB_GET_CALIBRATION_LOCK_FAILED_PARAMS;

/* valid flags parameter values */
#define LW2080_CTRL_CMD_FB_GET_CAL_FLAG_NONE              (0x00000000)
#define LW2080_CTRL_CMD_FB_GET_CAL_FLAG_RESET             (0x00000001)

/*
 * LW2080_CTRL_CMD_FB_SET_SCANOUT_COMPACTION_ALLOWED
 *
 * This command specifies to RM if scanout compaction feature is allowed or
 * not in the current configuration. In hybrid mode when dGPU is rendering the
 * image, the dGPU blit to the scanout surface happens without mGPU's
 * knowledge (directly to system memory), which results in stale compacted
 * data resulting in corruption.
 *
 * This control call can be used to disable the compaction whenever the KMD
 * (client) is switching to the pref mode in Hybrid i.e., whenever there is a
 * possibility of dGPU doing a blit to mGpu scanout surface. Compaction can
 * be enabled when system is back in hybrid power mode as mGpu will be
 * rendering the image.
 *
 *   allowCompaction
 *     This parameter specifies if the display compaction feature is allowed
 *     or not allowed.
 *   immediate
 *     This parameter specifies whether compaction has to be enabled or
 *     disabled immediately (based on the value of allowCompaction field) or
 *     during the next modeset.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_ILWALID_PARAM_STRUCT
 *   LWOS_STATUS_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_CMD_FB_SET_SCANOUT_COMPACTION_ALLOWED (0x2080130d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0xD" */ // Deprecated, removed form RM

typedef struct LW2080_CTRL_FB_SET_SCANOUT_COMPACTION_ALLOWED_PARAMS {
    LwU32 allowCompaction;
    LwU32 immediate;
} LW2080_CTRL_FB_SET_SCANOUT_COMPACTION_ALLOWED_PARAMS;

/* valid allowCompaction values */
#define LW2080_CTRL_CMD_FB_SET_SCANOUT_COMPACTION_ALLOW         (0x00000001)
#define LW2080_CTRL_CMD_FB_SET_SCANOUT_COMPACTION_DISALLOW      (0x00000000)

/* valid immediate values */
#define LW2080_CTRL_CMD_FB_SET_SCANOUT_COMPACTION_IMMEDIATE     (000000001)
#define LW2080_CTRL_CMD_FB_SET_SCANOUT_COMPACTION_NOT_IMMEDIATE (000000000)

/*
 * LW2080_CTRL_CMD_FB_FLUSH_GPU_CACHE
 *
 * This command flushes a cache on the GPU which all memory accesses go
 * through.  The types of flushes supported by this API may not be supported by
 * all hardware.  Attempting an unsupported flush type will result in an error.
 *
 *   addressArray
 *     An array of physical addresses in the aperture defined by
 *     LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_APERTURE.  Each entry points to a
 *     contiguous block of memory of size memBlockSizeBytes.  The addresses are
 *     aligned down to addressAlign before coalescing adjacent addresses and
 *     sending flushes to hardware.
 *   addressAlign
 *     Used to align-down addresses held in addressArray.  A value of 0 will be
 *     forced to 1 to avoid a divide by zero.  Value is treated as minimum
 *     alignment and any hardware alignment requirements above this value will
 *     be honored.
 *   addressArraySize
 *     The number of entries in addressArray.
 *   memBlockSizeBytes
 *     The size in bytes of each memory block pointed to by addressArray.
 *   flags
 *     Contains flags to control various aspects of the flush.  Valid values
 *     are defined in LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS*.  Not all flags are
 *     valid for all defined FLUSH_MODEs or all GPUs.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 *
 * See Also:
 *   LW0080_CTRL_CMD_DMA_FLUSH
 *     Performs flush operations in broadcast for the GPU cache and other hardware
 *     engines.  Use this call if you want to flush all GPU caches in a
 *     broadcast device.
 *    LW0041_CTRL_CMD_SURFACE_FLUSH_GPU_CACHE
 *     Flushes memory associated with a single allocation if the hardware
 *     supports it.  Use this call if you want to flush a single allocation and
 *     you have a memory object describing the physical memory.
 */
#define LW2080_CTRL_CMD_FB_FLUSH_GPU_CACHE                      (0x2080130e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_FLUSH_GPU_CACHE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_MAX_ADDRESSES            500

#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_PARAMS_MESSAGE_ID (0xEU)

typedef struct LW2080_CTRL_FB_FLUSH_GPU_CACHE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 addressArray[LW2080_CTRL_FB_FLUSH_GPU_CACHE_MAX_ADDRESSES], 8);
    LwU32 addressArraySize;
    LwU32 addressAlign;
    LW_DECLARE_ALIGNED(LwU64 memBlockSizeBytes, 8);
    LwU32 flags;
} LW2080_CTRL_FB_FLUSH_GPU_CACHE_PARAMS;

/* valid fields and values for flags */
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_APERTURE              1:0
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_APERTURE_VIDEO_MEMORY    (0x00000000)
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_APERTURE_SYSTEM_MEMORY   (0x00000001)
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_APERTURE_PEER_MEMORY     (0x00000002)
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK            2:2
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK_NO            (0x00000000)
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK_YES           (0x00000001)
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_ILWALIDATE            3:3
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_ILWALIDATE_NO            (0x00000000)
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_ILWALIDATE_YES           (0x00000001)
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FLUSH_MODE            4:4
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FLUSH_MODE_ADDRESS_ARRAY (0x00000000)
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FLUSH_MODE_FULL_CACHE    (0x00000001)
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FB_FLUSH              5:5
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FB_FLUSH_NO              (0x00000000)
#define LW2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FB_FLUSH_YES             (0x00000001)

/*
 * LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY (deprecated; use LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2 instead)
 *
 * These commands access the cache allocation policy on a specific
 * engine, if supported.
 *
 *   engine
 *     Specifies the target engine.  Possible values are defined in
 *     LW2080_ENGINE_TYPE.
 *   allocPolicy
 *     Specifies the read/write allocation policy of the cache on the specified
 *     engine. Possible values are defined in
 *     LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_READS and
 *     LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_WRITES.
 *
 */
typedef struct LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_PARAMS {
    LwU32 engine;
    LwU32 allocPolicy;
} LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_PARAMS;

/* valid values for allocPolicy */
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_READS                0:0
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_READS_NO      (0x00000000)
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_READS_YES     (0x00000001)
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_WRITES               1:1
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_WRITES_NO     (0x00000000)
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_WRITES_YES    (0x00000001)


/*
 * LW2080_CTRL_CMD_FB_SET_GPU_CACHE_ALLOC_POLICY
 *
 * This command is deprecated.
 * Use LW2080_CTRL_CMD_FB_SET_GPU_CACHE_ALLOC_POLICY_V2 instead.
 *
 * This command sets the state of the cache allocation policy on a specific
 * engine, if supported.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FB_SET_GPU_CACHE_ALLOC_POLICY       (0x2080130f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0xF" */

/*
 * LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_PARAM
 *
 * These commands access the cache allocation policy on a specific
 * client, if supported.
 *
 *   count
 *     Specifies the number of entries in entry.
 *   entry
 *     Specifies an array of allocation policy entries.
 *
 * LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_ENTRY
 *
 *   clients
 *     Specifies the target client.  Possible values are defined in
 *     LW2080_CLIENT_TYPE_*.
 *   allocPolicy
 *     Specifies the read/write allocation policy of the cache on the specified
 *     engine. Possible values are defined in
 *     LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_READS and
 *     LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_WRITES.
 *
 * LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_ENTRY_SIZE
 *
 *     Specifies the maximum number of allocation policy entries allowed
 */
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_ENTRY_SIZE 11

typedef struct LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_ENTRY {
    LwU32 client;
    LwU32 allocPolicy;
} LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_ENTRY;

typedef struct LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_PARAMS {
    LwU32                                          count;
    LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_ENTRY entry[LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_ENTRY_SIZE];
} LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_PARAMS;

/* valid values for allocPolicy */
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_READS             0:0
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_READS_DISABLE    (0x00000000)
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_READS_ENABLE     (0x00000001)
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_READS_ALLOW       1:1
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_READS_ALLOW_NO   (0x00000000)
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_READS_ALLOW_YES  (0x00000001)
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_WRITES            2:2
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_WRITES_DISABLE   (0x00000000)
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_WRITES_ENABLE    (0x00000001)
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_WRITES_ALLOW      3:3
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_WRITES_ALLOW_NO  (0x00000000)
#define LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_WRITES_ALLOW_YES (0x00000001)


/*
 * LW2080_CTRL_CMD_FB_SET_GPU_CACHE_ALLOC_POLICY_V2
 *
 * This command sets the state of the cache allocation policy on a specific
 * engine, if supported.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FB_SET_GPU_CACHE_ALLOC_POLICY_V2          (0x20801318) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x18" */

/*
 * LW2080_CTRL_CMD_FB_GET_GPU_CACHE_ALLOC_POLICY (deprecated; use LW2080_CTRL_CMD_FB_GET_GPU_CACHE_ALLOC_POLICY_V2 instead)
 *
 * This command gets the state of the cache allocation policy on a specific
 * engine, if supported.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FB_GET_GPU_CACHE_ALLOC_POLICY             (0x20801312) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x12" */

/*
 * LW2080_CTRL_CMD_FB_GET_GPU_CACHE_ALLOC_POLICY_V2
 *
 * This command gets the state of the cache allocation policy on a specific
 * engine, if supported.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FB_GET_GPU_CACHE_ALLOC_POLICY_V2          (0x20801319) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x19" */


/*
 * LW2080_CTRL_CMD_FB_IS_KIND
 *
 * This command is used to perform various operations like 'IS_KIND_VALID',
 * 'IS_KIND_COMPRESSIBLE'on the kind passed by the caller. The operation to be
 * performed should be passed in the 'operation' parameter of
 * LW2080_CTRL_FB_IS_KIND_PARAMS, the kind on which the operation is to be
 * performed should be passed in the 'kind' parameter. The result of the
 * operation (true/false) will be returned in the 'result' parameter.
 *
 *   operation
 *     Specifies what operation is to be performed on the kind passed by the
 *     caller. The supported operations are
 *       LW2080_CTRL_FB_IS_KIND_OPERATION_SUPPORTED
 *         This operation checks whether the kind passed in the 'kind'
 *         parameter of the 'LW2080_CTRL_FB_IS_KIND_PARAMS' structure is
 *         supported for this GPU. Returns nonzero value in 'result' parameter
 *         if the input kind is supported, else returns zero in the result.
 *       LW2080_CTRL_FB_IS_KIND_OPERATION_COMPRESSIBLE
 *         This operation checks whether the kind passed in the 'kind'
 *         parameter of the 'LW2080_CTRL_FB_IS_KIND_PARAMS' structure is
 *         compressible. Returns nonzero value in 'result' parameter if the
 *         input kind is compressible, else returns zero in the result.
 *       LW2080_CTRL_FB_IS_KIND_OPERATION_COMPRESSIBLE_1
 *         This operation checks whether the kind passed in the 'kind'
 *         parameter of the 'LW2080_CTRL_FB_IS_KIND_PARAMS' structure supports
 *         1 bit compression. Returns nonzero value in 'result' parameter if
 *         kind supports 1 bit compression, else returns zero in the result.
 *       LW2080_CTRL_FB_IS_KIND_OPERATION_COMPRESSIBLE_2
 *         This operation checks whether the kind passed in the 'kind'
 *         parameter of the 'LW2080_CTRL_FB_IS_KIND_PARAMS' structure supports
 *         2 bit compression. Returns nonzero value in 'result' parameter if
 *         kind supports 1 bit compression, else returns zero in the result.
 *       LW2080_CTRL_FB_IS_KIND_OPERATION_COMPRESSIBLE_4
 *         This operation checks whether the kind passed in the 'kind'
 *         parameter of the 'LW2080_CTRL_FB_IS_KIND_PARAMS' structure supports
 *         4 bit compression. Returns nonzero value in 'result' parameter if
 *         kind supports 4 bit compression, else returns zero in the result.
 *       LW2080_CTRL_FB_IS_KIND_OPERATION_ZBC
 *         This operation checks whether the kind passed in the 'kind'
 *         parameter of the 'LW2080_CTRL_FB_IS_KIND_PARAMS' structure
 *         supports ZBC. Returns nonzero value in 'result' parameter if the
 *         input kind supports ZBC, else returns zero in the result.
 *       LW2080_CTRL_FB_IS_KIND_OPERATION_ZBC_ALLOWS_1
 *         This operation checks whether the kind passed in the 'kind'
 *         parameter of the 'LW2080_CTRL_FB_IS_KIND_PARAMS' structure
 *         supports 1 bit ZBC. Returns nonzero value in 'result' parameter if
 *         the input kind supports 1 bit ZBC, else returns zero in the result.
 *       LW2080_CTRL_FB_IS_KIND_OPERATION_ZBC_ALLOWS_2
 *         This operation checks whether the kind passed in the 'kind'
 *         parameter of the 'LW2080_CTRL_FB_IS_KIND_PARAMS' structure
 *         supports 2 bit ZBC. Returns nonzero value in 'result' parameter if
 *         the input kind supports 2 bit ZBC, else returns zero in the result.
 *       LW2080_CTRL_FB_IS_KIND_OPERATION_ZBC_ALLOWS_4
 *         This operation checks whether the kind passed in the 'kind'
 *         parameter of the 'LW2080_CTRL_FB_IS_KIND_PARAMS' structure
 *         supports 4 bit ZBC. Returns nonzero value in 'result' parameter if
 *         the input kind supports 4 bit ZBC, else returns zero in the result.
 *   kind
 *     Specifies the kind on which the operation is to be carried out. The
 *     legal range of values for the kind parameter is different on different
 *     GPUs. For e.g. on Fermi, valid range is 0x00 to 0xfe. Still, some values
 *     inside this legal range can be invalid i.e. not defined.
 *     So its always better to first check if a particular kind is supported on
 *     the current GPU with 'LW2080_CTRL_FB_IS_KIND_SUPPORTED' operation.
 *   result
 *     Upon return, this parameter will hold the result (true/false) of the
 *     operation performed on the kind.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FB_IS_KIND                                (0x20801313) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_IS_KIND_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_IS_KIND_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW2080_CTRL_FB_IS_KIND_PARAMS {
    LwU32  operation;
    LwU32  kind;
    LwBool result;
} LW2080_CTRL_FB_IS_KIND_PARAMS;

/* valid values for operation */
#define LW2080_CTRL_FB_IS_KIND_OPERATION_SUPPORTED      (0x00000000)
#define LW2080_CTRL_FB_IS_KIND_OPERATION_COMPRESSIBLE   (0x00000001)
#define LW2080_CTRL_FB_IS_KIND_OPERATION_COMPRESSIBLE_1 (0x00000002)
#define LW2080_CTRL_FB_IS_KIND_OPERATION_COMPRESSIBLE_2 (0x00000003)
#define LW2080_CTRL_FB_IS_KIND_OPERATION_COMPRESSIBLE_4 (0x00000004)
#define LW2080_CTRL_FB_IS_KIND_OPERATION_ZBC            (0x00000005)
#define LW2080_CTRL_FB_IS_KIND_OPERATION_ZBC_ALLOWS_1   (0x00000006)
#define LW2080_CTRL_FB_IS_KIND_OPERATION_ZBC_ALLOWS_2   (0x00000007)
#define LW2080_CTRL_FB_IS_KIND_OPERATION_ZBC_ALLOWS_4   (0x00000008)

/**
 * LW2080_CTRL_CMD_FB_GET_GPU_CACHE_INFO
 *
 * This command returns the state of a cache which all GPU memory accesess go
 * through.
 *
 *   powerState
 *     Returns the power state of the cache.  Possible values are defined in
 *     LW2080_CTRL_FB_GET_GPU_CACHE_INFO_POWER_STATE.
 *
 *   writeMode
 *     Returns the write mode of the cache.  Possible values are defined in
 *     LW2080_CTRL_FB_GET_GPU_CACHE_INFO_WRITE_MODE.
 *
 *   bypassMode
 *     Returns the bypass mode of the L2 cache.  Possible values are defined in
 *     LW2080_CTRL_FB_GET_GPU_CACHE_INFO_BYPASS_MODE.
 *
 *   rcmState
 *     Returns the RCM state of the cache.  Possible values are defined in
 *     LW2080_CTRL_FB_GET_GPU_CACHE_INFO_RCM_STATE.
 *
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_CMD_FB_GET_GPU_CACHE_INFO           (0x20801315) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_GPU_CACHE_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_GET_GPU_CACHE_INFO_PARAMS_MESSAGE_ID (0x15U)

typedef struct LW2080_CTRL_FB_GET_GPU_CACHE_INFO_PARAMS {
    LwU32 powerState;
    LwU32 writeMode;
    LwU32 bypassMode;
    LwU32 rcmState;
} LW2080_CTRL_FB_GET_GPU_CACHE_INFO_PARAMS;

/* valid values for powerState */
#define LW2080_CTRL_FB_GET_GPU_CACHE_INFO_POWER_STATE_ENABLED     (0x00000000)
#define LW2080_CTRL_FB_GET_GPU_CACHE_INFO_POWER_STATE_DISABLED    (0x00000001)
/* valid values for writeMode */
#define LW2080_CTRL_FB_GET_GPU_CACHE_INFO_WRITE_MODE_WRITETHROUGH (0x00000000)
#define LW2080_CTRL_FB_GET_GPU_CACHE_INFO_WRITE_MODE_WRITEBACK    (0x00000001)
/* valid values for bypassMode */
#define LW2080_CTRL_FB_GET_GPU_CACHE_INFO_BYPASS_MODE_DISABLED    (0x00000000)
#define LW2080_CTRL_FB_GET_GPU_CACHE_INFO_BYPASS_MODE_ENABLED     (0x00000001)
/* valid values for rcmState */
#define LW2080_CTRL_FB_GET_GPU_CACHE_INFO_RCM_STATE_FULL          (0x00000000)
#define LW2080_CTRL_FB_GET_GPU_CACHE_INFO_RCM_STATE_TRANSITIONING (0x00000001)
#define LW2080_CTRL_FB_GET_GPU_CACHE_INFO_RCM_STATE_REDUCED       (0x00000002)
#define LW2080_CTRL_FB_GET_GPU_CACHE_INFO_RCM_STATE_ZERO_CACHE    (0x00000003)

/*
 * LW2080_CTRL_FB_GPU_CACHE_PROMOTION_POLICY
 *
 * These commands access the cache promotion policy on a specific
 * engine, if supported by the hardware.
 *
 * Cache promotion refers to the GPU promoting a memory read to a larger
 * size to preemptively fill the cache so future reads to nearby memory
 * addresses will hit in the cache.
 *
 *   engine
 *     Specifies the target engine.  Possible values are defined in
 *     LW2080_ENGINE_TYPE.
 *   promotionPolicy
 *     Specifies the promotion policy of the cache on the specified
 *     engine. Possible values are defined by
 *     LW2080_CTRL_FB_GPU_CACHE_PROMOTION_POLICY_*.  These values are in terms
 *     of the hardware cache line size.
 *
 */
typedef struct LW2080_CTRL_FB_GPU_CACHE_PROMOTION_POLICY_PARAMS {
    LwU32 engine;
    LwU32 promotionPolicy;
} LW2080_CTRL_FB_GPU_CACHE_PROMOTION_POLICY_PARAMS;

/* valid values for promotionPolicy */
#define LW2080_CTRL_FB_GPU_CACHE_PROMOTION_POLICY_NONE    (0x00000000)
#define LW2080_CTRL_FB_GPU_CACHE_PROMOTION_POLICY_QUARTER (0x00000001)
#define LW2080_CTRL_FB_GPU_CACHE_PROMOTION_POLICY_HALF    (0x00000002)
#define LW2080_CTRL_FB_GPU_CACHE_PROMOTION_POLICY_FULL    (0x00000003)


/*
 * LW2080_CTRL_CMD_FB_SET_GPU_CACHE_PROMOTION_POLICY
 *
 * This command sets the cache promotion policy on a specific engine, if
 * supported by the hardware.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FB_SET_GPU_CACHE_PROMOTION_POLICY (0x20801316) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x16" */ // Deprecated, removed form RM


/*
 * LW2080_CTRL_CMD_FB_GET_GPU_CACHE_PROMOTION_POLICY
 *
 * This command gets the cache promotion policy on a specific engine, if
 * supported by the hardware.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FB_GET_GPU_CACHE_PROMOTION_POLICY (0x20801317) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x17" */ // Deprecated, removed form RM

/*
 * LW2080_CTRL_FB_CMD_GET_FB_REGION_INFO
 *
 * This command returns the FB memory region characteristics.
 *
 *   numFBRegions
 *     Number of valid regions returned in fbRegion[]
 *   fbRegion[].base
 *     Base address of region.  The first valid address in the range
 *     [base..limit].
 *   fbRegion[].limit
 *     Last/end address of region.  The last valid address in the range
 *     [base..limit].
 *     (limit - base + 1) = size of the region
 *   fbRegion[].reserved
 *     Amount of memory that RM spelwlatively needs within the region.  A
 *     client doing its own memory management should leave at least this much
 *     memory available for RM use.  This partilwlarly applies to a driver
 *     model like LDDM.
 *   fbRegion[].performance
 *     Relative performance of this region compared to other regions.
 *     The definition is vague, and only connotes relative bandwidth or
 *     performance.  The higher the value, the higher the performance.
 *   fbRegion[].supportCompressed
 *     TRUE if compressed surfaces/kinds are supported
 *     FALSE if compressed surfaces/kinds are not allowed to be allocated in
 *     this region
 *   fbRegion[].supportISO
 *     TRUE if ISO surfaces/kinds are supported (Display, cursor, video)
 *     FALSE if ISO surfaces/kinds are not allowed to be allocated in this
 *     region
 *   fbRegion[].bProtected
 *     TRUE if this region is a protected memory region.  If true only
 *     allocations marked as protected (LWOS32_ALLOC_FLAGS_PROTECTED) can be
 *     allocated in this region.
 *   fbRegion[].blackList[] - DEPRECATED: Use supportISO
 *     TRUE for each LWOS32_TYPE_IMAGE* that is NOT allowed in this region.
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_GET_FB_REGION_INFO             (0x20801320) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_FB_GET_FB_REGION_INFO_MEM_TYPES   17

typedef LwBool LW2080_CTRL_CMD_FB_GET_FB_REGION_SURFACE_MEM_TYPE_FLAG[LW2080_CTRL_CMD_FB_GET_FB_REGION_INFO_MEM_TYPES];

typedef struct LW2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO {
    LW_DECLARE_ALIGNED(LwU64 base, 8);
    LW_DECLARE_ALIGNED(LwU64 limit, 8);
    LW_DECLARE_ALIGNED(LwU64 reserved, 8);
    LwU32                                                  performance;
    LwBool                                                 supportCompressed;
    LwBool                                                 supportISO;
    LwBool                                                 bProtected;
    LW2080_CTRL_CMD_FB_GET_FB_REGION_SURFACE_MEM_TYPE_FLAG blackList;
} LW2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO;

#define LW2080_CTRL_CMD_FB_GET_FB_REGION_INFO_MAX_ENTRIES 16

#define LW2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS {
    LwU32 numFBRegions;
    LW_DECLARE_ALIGNED(LW2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO fbRegion[LW2080_CTRL_CMD_FB_GET_FB_REGION_INFO_MAX_ENTRIES], 8);
} LW2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS;

/* valid flags parameter values */
/*
 * LW2080_CTRL_CMD_FB_OFFLINE_PAGES
 *
 * This command adds video memory page addresses to the Inforom's list of 
 * offlined addresses so that they're not allocated to any client. The newly 
 * offlined addresses take effect after a reboot.
 *
 *   offlined
 *     This input parameter is an array of LW2080_CTRL_FB_OFFLINED_ADDRESS_INFO
 *     structures, containing the video memory physical page numbers that
 *     are to be blacklisted. This array can hold a maximum of LW2080_CTRL_FB_
 *     BLACKLIST_PAGES_MAX_PAGES address pairs. Valid entries are adjacent.
 *   pageSize
 *     This input parameter contains the size of the page that is to be
 *     blacklisted. If this does not match with the page size specified in the
 *     blacklist in the inforom, LW_ERR_ILWALID_ARGUMENT is
 *     returned.
 *   validEntries
 *     This input parameter specifies the number of valid entries in the
 *     offlined array.
 *   numPagesAdded
 *     This output parameter specifies how many of the validEntries were
 *     actually offlined, since some entries from the Inforom's list of offlined
 *     addresses may already be oclwpied. If numPagesAdded < validEntries, it
 *     means that only addresses from offlined[0] to offlined[numPagesAdded - 1] 
 *     were added to the Inforom's list of offlined addresses.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FB_OFFLINE_PAGES              (0x20801321) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_OFFLINE_PAGES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_OFFLINED_PAGES_MAX_PAGES       (0x00000040)
#define LW2080_CTRL_FB_OFFLINED_PAGES_ILWALID_ADDRESS (0xffffffffffffffffULL)
#define LW2080_CTRL_FB_OFFLINED_PAGES_PAGE_SIZE_4K    (0x00000000)
#define LW2080_CTRL_FB_OFFLINED_PAGES_PAGE_SIZE_64K   (0x00000001)
#define LW2080_CTRL_FB_OFFLINED_PAGES_PAGE_SIZE_128K  (0x00000002)

/*
 * LW2080_CTRL_FB_OFFLINED_ADDRESS_INFO
 *
 *   pageAddressWithEccOn
 *     Address of the memory page retired when ECC is enabled on the board.
 *   pageAddressWithEccOff
 *     Address of the memory page retired when ECC is disabled on the board.
 *   rbcAddress
 *     Row/Bank/Column Address of the faulty memory which caused the page to
 *     be retired, this will only be valid when source is either
 *     LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_DPR_MULTIPLE_SBE or
 *     LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_DPR_DBE
 *   source
 *     The reason for the page to be retired. Valid values for
 *     this parameter include:
 *        LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_STATIC_OR_USER
 *           Page retired either from static page retirement during
 *           manufacturing or by a user using the
 *           LW2080_CTRL_CMD_FB_OFFLINE_PAGES command.
 *        LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_MODS_SBE
 *           Page was retired due to a single bit error during
 *           MODS testing.
 *        LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_DPR_MULTIPLE_SBE
 *           Page retired by dynamic page retirement due to multiple
 *           single bit errors seen.
 *        LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_MODS_DBE
 *           Page was retired due to a double bit error during
 *           MODS testing.
 *        LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_DPR_DBE
 *           Page retired by dynamic page retirement due to a double bit
 *           error seen.
 *        LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_MODS_MEM_ERROR
 *           Page retired due to a memory error during
 *           MODS testing.
 *   status
 *      Non-exceptional reasons for a page retirement failure
 *         LW2080_CTRL_FB_OFFLINED_PAGES_STATUS_OK
 *            No error
 *         LW2080_CTRL_FB_OFFLINED_PAGES_STATUS_PENDING_RETIREMENT
 *            The given address is already pending retirement or has
 *            been retired during the current driver run. The page
 *            will be offlined during the next driver run.
 *         LW2080_CTRL_FB_OFFLINED_PAGES_STATUS_BLACKLISTING_FAILED
 *            The given page was retired on a previous driver run,
 *            so it should not be accessible unless offlining failed.
 *            Failing to offline a page is strongly indicative of a
 *            driver offlining bug.
 *         LW2080_CTRL_FB_OFFLINED_PAGES_STATUS_TABLE_FULL
 *            The PBL is full and no more pages can be retired
 *         LW2080_CTRL_FB_OFFLINED_PAGES_STATUS_INTERNAL_ERROR
 *            Internal driver error
 *
 */

typedef struct LW2080_CTRL_FB_OFFLINED_ADDRESS_INFO {
    LW_DECLARE_ALIGNED(LwU64 pageAddressWithEccOn, 8);
    LW_DECLARE_ALIGNED(LwU64 pageAddressWithEccOff, 8);
    LwU32 rbcAddress;
    LwU32 source;
    LwU32 status;
    LwU32 timestamp;
} LW2080_CTRL_FB_OFFLINED_ADDRESS_INFO;

/* valid values for source */
#define LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_STATIC_OR_USER      (0x00000000)
#define LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_MODS_SBE            (0x00000001)
#define LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_DPR_MULTIPLE_SBE    (0x00000002)
#define LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_MODS_DBE            (0x00000003)
#define LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_DPR_DBE             (0x00000004)
#define LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_MODS_MEM_ERROR      (0x00000005)

/* valid values for status */
#define LW2080_CTRL_FB_OFFLINED_PAGES_STATUS_OK                  (0x00000000)
#define LW2080_CTRL_FB_OFFLINED_PAGES_STATUS_PENDING_RETIREMENT  (0x00000001)
#define LW2080_CTRL_FB_OFFLINED_PAGES_STATUS_BLACKLISTING_FAILED (0x00000002)
#define LW2080_CTRL_FB_OFFLINED_PAGES_STATUS_TABLE_FULL          (0x00000003)
#define LW2080_CTRL_FB_OFFLINED_PAGES_STATUS_INTERNAL_ERROR      (0x00000004)

/* deprecated */
#define LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_MULTIPLE_SBE        LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_DPR_MULTIPLE_SBE
#define LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_DBE                 LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_DPR_DBE


#define LW2080_CTRL_FB_OFFLINE_PAGES_PARAMS_MESSAGE_ID (0x21U)

typedef struct LW2080_CTRL_FB_OFFLINE_PAGES_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_FB_OFFLINED_ADDRESS_INFO offlined[LW2080_CTRL_FB_OFFLINED_PAGES_MAX_PAGES], 8);
    LwU32 pageSize;
    LwU32 validEntries;
    LwU32 numPagesAdded;
} LW2080_CTRL_FB_OFFLINE_PAGES_PARAMS;

/*
 * LW2080_CTRL_CMD_FB_GET_OFFLINED_PAGES
 *
 * This command returns the list of video memory page addresses in the
 * Inforom's blacklist.
 *
 *   offlined
 *     This output parameter is an array of LW2080_CTRL_FB_BLACKLIST_ADDRESS_
 *     INFO structures, containing the video memory physical page numbers that
 *     are blacklisted. This array can hold a maximum of LW2080_CTRL_FB_
 *     BLACKLIST_PAGES_MAX_PAGES address pairs. Valid entries are adjacent.
 *     The array also contains the Row/Bank/Column Address and source.
 *   validEntries
 *     This output parameter specifies the number of valid entries in the
 *     offlined array.
 *   bRetirementPending (DEPRECATED, use retirementPending instead)
 *     This output parameter returns if any pages on the list are pending
 *     retirement.
 *   retirementPending
 *     Communicates to the caller whether retirement updates are pending and the
 *     reason for the updates. Possible fields are:
 *     LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_*:
 *       LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_SBE:
 *         Indicates whether pages are pending retirement due to SBE.
 *       LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_DBE:
 *         Indicates whether pages are pending retirement due to DBE. Driver
 *         reload needed to retire bad memory pages and allow compute app runs.
 *       LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_CLEAR:
 *         Indicates whether pages cleared from the InfoROM PBL object need to
 *         be added back to the available FB pages, happens after driver
 *         reload.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_GET_OFFLINED_PAGES                            (0x20801322) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_OFFLINED_PAGES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_SBE          0:0
#define LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_SBE_FALSE   0
#define LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_SBE_TRUE    1
#define LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_DBE          1:1
#define LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_DBE_FALSE   0
#define LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_DBE_TRUE    1
#define LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_CLEAR        2:2
#define LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_CLEAR_FALSE 0
#define LW2080_CTRL_FB_GET_OFFLINED_PAGES_RETIREMENT_PENDING_CLEAR_TRUE  1

#define LW2080_CTRL_FB_GET_OFFLINED_PAGES_PARAMS_MESSAGE_ID (0x22U)

typedef struct LW2080_CTRL_FB_GET_OFFLINED_PAGES_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_FB_OFFLINED_ADDRESS_INFO offlined[LW2080_CTRL_FB_OFFLINED_PAGES_MAX_PAGES], 8);
    LwU32  validEntries;
    LwBool bRetirementPending;
    LwU8   retirementPending;
} LW2080_CTRL_FB_GET_OFFLINED_PAGES_PARAMS;

/*
 * LW2080_CTRL_CMD_FB_QUERY_ACR_REGION
 *
 * This control command is used to query the selwred region allocated
 *
 * queryType
 *          LW2080_CTRL_CMD_FB_ACR_QUERY_GET_REGION_STATUS: Provides the alloc
 *          status and ACR region ID.
 *          LW2080_CTRL_CMD_FB_QUERY_MAP_REGION : Maps the region on BAR1
 *           it returns the "pCpuAddr" and pPriv to user.
 *          LW2080_CTRL_CMD_FB_QUERY_UNMAP_REGION: Unmaps the mapped region.
 *          it takes the pPriv as input
 *
 * clientReq : struct ACR_REQUEST_PARAMS
 *          It is used to find the allocated ACR region for that client
 *          clientId     : ACR Client ID
 *          reqReadMask  : read mask of ACR region
 *          reqWriteMask : Write mask of ACR region
 *          regionSize   : ACR region Size
 *
 * clientReqStatus : struct ACR_STATUS_PARAMS
 *          This struct is stores the output of requested ACR region.
 *          allocStatus     : Allocated Status of ACR region
 *          regionId        : ACR region ID
 *          physicalAddress : Physical address on FB
 *
 *
 * LW2080_CTRL_CMD_FB_ACR_QUERY_ERROR_CODE
 *          LW2080_CTRL_CMD_FB_ACR_QUERY_ERROR_NONE : Control command exelwted successfully
 *          LW2080_CTRL_CMD_FB_ACR_QUERY_ERROR_ILWALID_CLIENT_REQUEST : Please check the parameter
 *                      for ACR client request
 *          LW2080_CTRL_CMD_FB_ACR_QUERY_ERROR_FAILED_TO_MAP_ON_BAR1 : RM Fails to map ACR region
 *                      on BAR1
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
*/
#define LW2080_CTRL_CMD_FB_QUERY_ACR_REGION (0x20801325) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FB_QUERY_ACR_REGION_PARAMS_MESSAGE_ID" */

//
// We can create an ACR region by using RMCreateAcrRegion[1|2] regkey or mods -acr[1|2]_size
// Client ID for such region is 2 in RM.
//
#define LW2080_CTRL_CMD_FB_ACR_CLIENT_ID    2

typedef enum LW2080_CTRL_CMD_FB_ACR_QUERY_TYPE {
    LW2080_CTRL_CMD_FB_ACR_QUERY_GET_CLIENT_REGION_STATUS = 0,
    LW2080_CTRL_CMD_FB_ACR_QUERY_GET_REGION_PROPERTY = 1,
    LW2080_CTRL_CMD_FB_ACR_QUERY_GET_FALCON_STATUS = 2,
} LW2080_CTRL_CMD_FB_ACR_QUERY_TYPE;

typedef enum LW2080_CTRL_CMD_FB_ACR_QUERY_ERROR_CODE {
    LW2080_CTRL_CMD_FB_ACR_QUERY_ERROR_NONE = 0,
    LW2080_CTRL_CMD_FB_ACR_QUERY_ERROR_ILWALID_CLIENT_REQUEST = 1,
} LW2080_CTRL_CMD_FB_ACR_QUERY_ERROR_CODE;

typedef struct ACR_REQUEST_PARAMS {
    LwU32 clientId;
    LwU32 reqReadMask;
    LwU32 reqWriteMask;
    LwU32 regionSize;
} ACR_REQUEST_PARAMS;

typedef struct ACR_REGION_ID_PROP {
    LwU32 regionId;
    LwU32 readMask;
    LwU32 writeMask;
    LwU32 regionSize;
    LwU32 clientMask;
    LW_DECLARE_ALIGNED(LwU64 physicalAddress, 8);
} ACR_REGION_ID_PROP;

typedef struct ACR_STATUS_PARAMS {
    LwU32 allocStatus;
    LwU32 regionId;
    LW_DECLARE_ALIGNED(LwU64 physicalAddress, 8);
} ACR_STATUS_PARAMS;

typedef struct ACR_REGION_HANDLE {
    LwHandle hClient;
    LwHandle hParent;
    LwHandle hMemory;
    LwU32    hClass;
    LwHandle hDevice;
} ACR_REGION_HANDLE;

typedef struct ACR_FALCON_LS_STATUS {
    LwU16  falconId;
    LwBool bIsInLs;
} ACR_FALCON_LS_STATUS;

#define LW2080_CTRL_CMD_FB_QUERY_ACR_REGION_PARAMS_MESSAGE_ID (0x25U)

typedef struct LW2080_CTRL_CMD_FB_QUERY_ACR_REGION_PARAMS {
    LW2080_CTRL_CMD_FB_ACR_QUERY_TYPE       queryType;
    LW2080_CTRL_CMD_FB_ACR_QUERY_ERROR_CODE errorCode;
    LW_DECLARE_ALIGNED(ACR_REGION_ID_PROP acrRegionIdProp, 8);
    ACR_REQUEST_PARAMS                      clientReq;
    LW_DECLARE_ALIGNED(ACR_STATUS_PARAMS clientReqStatus, 8);
    ACR_REGION_HANDLE                       handle;
    ACR_FALCON_LS_STATUS                    falconStatus;
} LW2080_CTRL_CMD_FB_QUERY_ACR_REGION_PARAMS;

/*
 * LW2080_CTRL_CMD_FB_CLEAR_OFFLINED_PAGES
 *
 * This command clears offlined video memory page addresses from the Inforom.
 *
 *   sourceMask
 *     This is a bit mask of LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE. Pages
 *     offlined from the specified sources will be cleared/removed from the
 *     Inforom PBL object denylist.
 *     As of now only dynamically offlined pages i.e. from source
 *       LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_DPR_MULTIPLE_SBE
 *       LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_DPR_DBE
 *     can be cleared from the list.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_CLEAR_OFFLINED_PAGES (0x20801326) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_CLEAR_OFFLINED_PAGES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_CLEAR_OFFLINED_PAGES_PARAMS_MESSAGE_ID (0x26U)

typedef struct LW2080_CTRL_FB_CLEAR_OFFLINED_PAGES_PARAMS {
    LwU32 sourceMask;
} LW2080_CTRL_FB_CLEAR_OFFLINED_PAGES_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_GET_COMPBITCOPY_INFO
 *
 * Gets pointer to then object of class CompBitCopy, which is used for swizzling
 * compression bits in the compression backing store. The caller is expected to
 * have the appropriate headers for class CompBitCopy. Also retrieves values of some
 * parameters needed to call the compbit swizzling method.
 *
 * @params[out] void *pCompBitCopyObj
 *     Opaque pointer to object of class CompBitCopy
 * @params[out] void *pSwizzleParams
 *     Opaque pointer to values needed to call the compbit
 *     swizzle method.
 *
 * Possible status values returned are:
 *   LW_OK LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_GET_COMPBITCOPY_INFO (0x20801327) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FB_GET_COMPBITCOPY_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_FB_GET_COMPBITCOPY_INFO_PARAMS_MESSAGE_ID (0x27U)

typedef struct LW2080_CTRL_CMD_FB_GET_COMPBITCOPY_INFO_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pCompBitCopyObj, 8);
    LW_DECLARE_ALIGNED(LwP64 pSwizzleParams, 8);
} LW2080_CTRL_CMD_FB_GET_COMPBITCOPY_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_FB_GET_LTC_INFO_FOR_FBP
 *
 * Gets the count and mask of LTCs for a given FBP.
 *
 *   fbpIndex
 *     The physical index of the FBP to get LTC info for.
 *   ltcMask
 *     The mask of active LTCs for the given FBP.
 *   ltcCount
 *     The count of active LTCs for the given FBP.
 *   ltsMask
 *      The mask of active LTSs for the given FBP
 *   ltsCount
 *      The count of active LTSs for the given FBP
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_GET_LTC_INFO_FOR_FBP (0x20801328) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_LTC_INFO_FOR_FBP_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_GET_LTC_INFO_FOR_FBP_PARAMS_MESSAGE_ID (0x28U)

typedef struct LW2080_CTRL_FB_GET_LTC_INFO_FOR_FBP_PARAMS {
    LwU8  fbpIndex;
    LwU32 ltcMask;
    LwU32 ltcCount;
    LwU32 ltsMask;
    LwU32 ltsCount;
} LW2080_CTRL_FB_GET_LTC_INFO_FOR_FBP_PARAMS;


/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_SET_CONTEXT               < Deprecated >
 *
 * "set the context" for following CompBitCopy member functions.
 * These are the CompBitCopy member variables that remain connstant
 * over multiple CompBitCopy member function calls, yet stay the same
 * throughout a single surface eviction.
 *
 * @params[in] UINT64  backingStorePA;
 *     Physical Address of the Backing Store
 * @params[in] UINT08 *backingStoreVA;
 *     Virtual Address of the Backing Store
 * @params[in] UINT64  backingStoreChunkPA;
 *     Physical Address of the "Chunk Buffer"
 * @params[in] UINT08 *backingStoreChunkVA;
 *     Virtual Address of the "Chunk Buffer"
 * @params[in] UINT32  backingStoreChunkSize;
 *     Size of the "Chunk Buffer"
 * @params[in] UINT08 *cacheWriteBitMap;
 *     Pointer to the bitmap which parts of the
 *     "Chunk" was updated.
 * @params[in] bool    backingStoreChunkOverfetch;
 *     Overfetch factor.
 * @params[in] UINT32  PageSizeSrc;
 *     Page size of Source Surface.
 * @params[in] UINT32  PageSizeDest;
 *     page size of Destination Surface.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_SET_CONTEXT (0x20801329) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x29" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_SET_CONTEXT_PARAMS {
    LwU32  CBCBaseAddress;
    LW_DECLARE_ALIGNED(LwU64 backingStorePA, 8);
    LW_DECLARE_ALIGNED(LwU8 *backingStoreVA, 8);
    LW_DECLARE_ALIGNED(LwU64 backingStoreChunkPA, 8);
    LW_DECLARE_ALIGNED(LwU8 *backingStoreChunkVA, 8);
    LwU32  backingStoreChunkSize;
    LW_DECLARE_ALIGNED(LwU8 *cacheWriteBitMap, 8);
    LwBool backingStoreChunkOverfetch;
    LwU32  PageSizeSrc;
    LwU32  PageSizeDest;
} LW2080_CTRL_CMD_FB_COMPBITCOPY_SET_CONTEXT_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_COMPBITS              < Deprecated >
 *
 * Retrieves the Compression and Fast Clear bits for the surface+offset given.
 *
 * @params[out] LwU32  *fcbits;
 *     Fast Clear Bits returned
 * @params[out] LwU32  *compbits;
 *     Compression Bits returned
 * @params[in] LwU64  dataPhysicalStart;
 *     Start Address of Data
 * @params[in] LwU64  surfaceOffset;
 *     Offset in the surface
 * @params[in] LwU32  comptagLine;
 *     Compression Tag Number
 * @params[in] LwBool upper64KBCompbitSel;
 *     Selects Upper or Lower 64K
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTEDD
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_COMPBITS (0x2080132a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x2A" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_COMPBITS_PARAMS {
    LW_DECLARE_ALIGNED(LwU32 *fcbits, 8);
    LW_DECLARE_ALIGNED(LwU32 *compbits, 8);
    LW_DECLARE_ALIGNED(LwU64 dataPhysicalStart, 8);
    LW_DECLARE_ALIGNED(LwU64 surfaceOffset, 8);
    LwU32  comptagLine;
    LwBool upper64KBCompbitSel;
} LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_COMPBITS_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_PUT_COMPBITS              < Deprecated >
 *
 * Sets the Compression and Fast Clear bits for the surface+offset given.
 *
 * @params[in] LwU32  fcbits;
 *     Fast Clear Bits to write.
 * @params[in] LwU32  compbits;
 *     Compression Bits to write
 * @params[in] LwBool writeFc;
 *     Indicates if Fast Clear Bits should be written
 * @params[in] LwU64  dataPhysicalStart;
 *     Start Address of Data
 * @params[in] LwU64  surfaceOffset;
 *     Offset in the surface
 * @params[in] LwU32  comptagLine;
 *     Compression Tag Number
 * @params[in] LwBool upper64KBCompbitSel;
 *     Selects Upper or Lower 64K
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_PUT_COMPBITS (0x2080132b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x2B" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_PUT_COMPBITS_PARAMS {
    LwU32  fcbits;
    LwU32  compbits;
    LwBool writeFc;
    LW_DECLARE_ALIGNED(LwU64 dataPhysicalStart, 8);
    LW_DECLARE_ALIGNED(LwU64 surfaceOffset, 8);
    LwU32  comptagLine;
    LwBool upper64KBCompbitSel;
} LW2080_CTRL_CMD_FB_COMPBITCOPY_PUT_COMPBITS_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_READ_COMPBITS64KB         < Deprecated >
 *
 * Read 64KB chunk of CompBits
 *
 * @params[in] LwU64  SrcDataPhysicalStart;
 *     Start Address of Data
 * @params[in] LwU32  SrcComptagLine;
 *     Compression Tag Number
 * @params[in] LwU32  page64KB;
 *     Which 64K block to read from.
 * @params[out] LwU32  *compbitBuffer;
 *     Buffer for CompBits read,
 * @params[in] LwBool upper64KBCompbitSel;
 *     Selects Upper or Lower 64K
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_READ_COMPBITS64KB (0x2080132c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x2C" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_READ_COMPBITS64KB_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 SrcDataPhysicalStart, 8);
    LwU32  SrcComptagLine;
    LwU32  page64KB;
    LW_DECLARE_ALIGNED(LwU32 *compbitBuffer, 8);
    LwBool upper64KBCompbitSel;
} LW2080_CTRL_CMD_FB_COMPBITCOPY_READ_COMPBITS64KB_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_WRITE_COMPBITS64KB        < Deprecated >
 *
 * Write 64K chunk of COmpBits.
 *
 * @params[in] LwU64  SrcDataPhysicalStart;
 *     Start Address of Data
 * @params[in] LwU32  SrcComptagLine;
 *     Compression Tag Number
 * @params[in] LwU32  page64KB;
 *     Which 64K block to read from.
 * @params[in] LwU32  *compbitBuffer;
 *     Buffer for CompBits to write.
 * @params[in] LwBool upper64KBCompbitSel
 *     Selects Upper or Lower 64K
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_WRITE_COMPBITS64KB (0x2080132d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x2D" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_WRITE_COMPBITS64KB_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 DstDataPhysicalStart, 8);
    LwU32  DstComptagLine;
    LwU32  page64KB;
    LW_DECLARE_ALIGNED(LwU32 *compbitBuffer, 8);
    LwBool upper64KBCompbitSel;
} LW2080_CTRL_CMD_FB_COMPBITCOPY_WRITE_COMPBITS64KB_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_COMPBITSPS        < Deprecated >
 *
 * The PS (Performance Path, or Optimized path, or Per Slice version)
 * of PutCompBits.
 *
 * @params[out] LwU32  *fcbits;
 *     Buffer to receive Fast Clear Bits.
 * @params[out] LwU32  *compbits;
 *     Buffer to receive Compression Bits.
 * @params[out] LwU32  *compCacheLine;
 *     Buffer to receive Comp Cache Line data.
 * @params[in] LwU64  dataPhysicalStart;
 *     Start Address of Data
 * @params[in] LwU64  surfaceOffset;
 *     Offset in the surface
 * @params[in] LwU32  comptagLine;
 *     Compression Tag Line Number
 * @params[in] LwU32  ROPTile_offset;
 *     Offset in the surface of the ROP tile.
 * @params[in] LwBool upper64KBCompbitSel;
 *     Selects Upper or Lower 64K
 * @params[in] LwBool getFcBits;
 *   Indicates if fast clear bits should be returned.
 * @params[in] LwP64  derivedParams
 *   Actually a CompBitDerivedParams structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_COMPBITSPS (0x2080132e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x2E" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_COMPBITSPS_PARAMS {
    LW_DECLARE_ALIGNED(LwU32 *fcbits, 8);
    LW_DECLARE_ALIGNED(LwU32 *compbits, 8);
    LW_DECLARE_ALIGNED(LwU32 *compCacheLine, 8);
    LW_DECLARE_ALIGNED(LwU64 dataPhysicalStart, 8);
    LW_DECLARE_ALIGNED(LwU64 surfaceOffset, 8);
    LwU32  comptagLine;
    LwU32  ROPTile_offset;
    LwBool upper64KBCompbitSel;
    LwBool getFcBits;
    LW_DECLARE_ALIGNED(LwP64 derivedParams, 8);
} LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_COMPBITSPS_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_PUT_COMPBITSPS        < Deprecated >
 *
 * The PS (Performance Path, or Optimized path, or Per Slice version)
 * of GetCompBits.
 *
 * @params[in] LwU32  fcbits;
 *     Buffer with Fast Clear Bits to write.
 * @params[in] LwU32  compbits;
 *     Buffer to receive Compression Bits.
 * @params[in] LwBool writeFc
 *     Indicates of Fast Clear Bits should be written.
 * @params[in] LwU32  *compCacheLine;
 *     Buffer to receive Comp Cache Line data.
 * @params[in] LwU64  dataPhysicalStart;
 *     Start Address of Data
 * @params[in] LwU64  surfaceOffset;
 *     Offset in the surface
 * @params[in] LwU32  comptagLine;
 *     Compression Tag Line Number
 * @params[in] LwU32  ROPTile_offset;
 *     Offset in the surface of the ROP tile.
 * @params[in] LwBool upper64KBCompbitSel;
 *     Selects Upper or Lower 64K
 * @params[in] LwP64  derivedParams
 *   Actually a CompBitDerivedParams structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_PUT_COMPBITSPS (0x2080132f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x2F" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_PUT_COMPBITSPS_PARAMS {
    LwU32  fcbits;
    LwU32  compbits;
    LwBool writeFc;
    LW_DECLARE_ALIGNED(LwU32 *compCacheLine, 8);
    LW_DECLARE_ALIGNED(LwU64 dataPhysicalStart, 8);
    LW_DECLARE_ALIGNED(LwU64 surfaceOffset, 8);
    LwU32  comptagLine;
    LwU32  ROPTile_offset;
    LwBool upper64KBCompbitSel;
    LW_DECLARE_ALIGNED(LwP64 derivedParams, 8);
} LW2080_CTRL_CMD_FB_COMPBITCOPY_PUT_COMPBITSPS_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_READ_COMPCACHELINEPS              < Deprecated >
 *
 * The PS (Performance Path, or Optimized path, or Per Slice version)
 * of ReadCompCacheLine.
 *
 * @paramsLwU32  *compCacheLine;
 *    Buffer for Comp Cache Line Read
 * @paramsLwU32  comptagLine;
 *    Comp Tag Line Number to read
 * @paramsLwU32  partition;
 *    FB Partition of the desired Comp Cache Line
 * @paramsLwU32  slice;
 *    Slice of the desired Comp Cache Line
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_READ_COMPCACHELINEPS (0x20801330) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x30" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_READ_COMPCACHELINEPS_PARAMS {
    LW_DECLARE_ALIGNED(LwU32 *compCacheLine, 8);
    LwU32 comptagLine;
    LwU32 partition;
    LwU32 slice;
} LW2080_CTRL_CMD_FB_COMPBITCOPY_READ_COMPCACHELINEPS_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_WRITE_COMPCACHELINEPS             < Deprecated >
 *
 * The PS (Performance Path, or Optimized path, or Per Slice version)
 * of WriteCompCacheLine.
 *
 * @params[in] LwU32  *compCacheLine;
 *    Buffer for Comp Cache Line to Write
 * @params[in] LwU32  comptagLine;
 *    Comp Tag Line Number to Write
 * @params[in] LwU32  partition;
 *    FB Partition of the desired Comp Cache Line
 * @params[in] LwU32  slice;
 *    Slice of the desired Comp Cache Line
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_WRITE_COMPCACHELINEPS (0x20801331) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x31" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_WRITE_COMPCACHELINEPS_PARAMS {
    LW_DECLARE_ALIGNED(LwU32 *compCacheLine, 8);
    LwU32 comptagLine;
    LwU32 partition;
    LwU32 slice;
} LW2080_CTRL_CMD_FB_COMPBITCOPY_WRITE_COMPCACHELINEPS_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_COMPCACHELINE_BOUNDS          < Deprecated >
 *
 * Used by PS (Performance Path, or Optimized path, or Per Slice version)
 * to retrieve upper and lower Address of the CompCacheLine.
 *
 * @params[out] LwU64  *minCPUAddress;
 *    Minimum (lower bound) of the ComCacheLine.
 * @params[out] LwU64  *minCPUAddress;
 *    Minimum (lower bound) of the ComCacheLine.
 * @params[in] LwU32  comptagLine;
 *    CompTagLine to fetch the bounds of.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_COMPCACHELINE_BOUNDS (0x20801332) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x32" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_COMPCACHELINE_BOUNDS_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 *minCPUAddress, 8);
    LW_DECLARE_ALIGNED(LwU64 *maxCPUAddress, 8);
    LwU32 comptagLine;
} LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_COMPCACHELINE_BOUNDS_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_PART_SLICE_OFFSET             < Deprecated >
 *
 * Used by PS (Performance Path, or Optimized path, or Per Slice version)
 * to retrieve partition, slice and ROP Tile Offset  of the passed in
 * surface location.
 *
 * @params[out] LwU64  *part;
 *    Partition in which the target part of the surface resides.
 * @params[out] LwU64  *slice;
 *    Slice in which the target part of the surface resides.
 * @params[out] LwU64  *ropTileoffset;
 *    Offset to the start of the ROP Tile in which the target part of
 * the surface resides.
 * @params[in] LwU64  *dataPhysicalStart;
 *    Start address of data for which part/slice/offset is desired.
 * @params[in] LwU64  surfaceOffset;
 *    Byte offset of data for which part/slice/offset is desired.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_PART_SLICE_OFFSET (0x20801333) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x33" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_PART_SLICE_OFFSET_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 *part, 8);
    LW_DECLARE_ALIGNED(LwU64 *slice, 8);
    LW_DECLARE_ALIGNED(LwU64 *ropTileoffset, 8);
    LW_DECLARE_ALIGNED(LwU64 dataPhysicalStart, 8);
    LW_DECLARE_ALIGNED(LwU64 surfaceOffset, 8);
} LW2080_CTRL_CMD_FB_COMPBITCOPY_GET_PART_SLICE_OFFSET_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_ALLOC_AND_INIT_DERIVEDPARAMS      < Deprecated >
 *
 * Used by PS (Performance Path, or Optimized path, or Per Slice version)
 * to create a CompBitCopy::CompBitDerivedParams object
 *
 * @params[out] LwP64  derivedParams
 *   Actually a CompBitDerivedParams structure.
  * @params[in] LwU32  comptagLine;
 *     Compression Tag Line Number
 * @params[in] LwU32  ROPTile_offset;
 *     Offset in the surface of the ROP tile.
 * @params[in] LwBool upper64KBCompbitSel;
 *     Selects Upper or Lower 64K
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_ALLOC_AND_INIT_DERIVEDPARAMS (0x20801334) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x34" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_ALLOC_AND_INIT_DERIVEDPARAMS_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 derivedParams, 8);
    LwU32  comptagLine;
    LwBool upper64KBCompbitSel;
} LW2080_CTRL_CMD_FB_COMPBITCOPY_ALLOC_AND_INIT_DERIVEDPARAMS_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_COMPBITCOPY_SET_FORCE_BAR1                    < Deprecated >
 *
 * Used by MODS (and possibly other clients) to have compbit code write
 * write directly to BAR1, rather than a intermediate buffer.
 *
 * @params[in] LwBool bForceBar1;
 *     Enables or disables direct writes to BAR1.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_COMPBITCOPY_SET_FORCE_BAR1 (0x20801335) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | 0x35" */

typedef struct LW2080_CTRL_CMD_FB_COMPBITCOPY_SET_FORCE_BAR1_PARAMS {
    LwBool bForceBar1;
} LW2080_CTRL_CMD_FB_COMPBITCOPY_SET_FORCE_BAR1_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_GET_AMAP_CONF
 *
 * Fills in fields of a structure of class ConfParamsV1, which is used for
 * swizzling compression bits in the compression backing store.
 * The caller is expected to have the appropriate headers for class ConfParamsV1.
 *
 * @params[in|out] void *pAmapConfParms
 *     Opaque pointer to structure of values for ConfParamsV1
 * @params[in|out] void *pCbcSwizzleParms
 *     Opaque pointer to structure of values for CbcSwizzleParamsV1
 *
 * Possible status values returned are:
 *   LW_OK LW_ERR_NOT_SUPPORTED
 *
 * pCbcSwizzleParams will be filled in with certain parameters from
 * @CbcSwizzleParamsV1.  However, the caller is responsible for making sure
 * all parameters are filled in before using it.
 */
#define LW2080_CTRL_CMD_FB_GET_AMAP_CONF (0x20801336) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FB_GET_AMAP_CONF_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_FB_GET_AMAP_CONF_PARAMS_MESSAGE_ID (0x36U)

typedef struct LW2080_CTRL_CMD_FB_GET_AMAP_CONF_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pAmapConfParams, 8);
    LW_DECLARE_ALIGNED(LwP64 pCbcSwizzleParams, 8);
} LW2080_CTRL_CMD_FB_GET_AMAP_CONF_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_CBC_OP
 *
 * Provides a way for clients to request a CBC Operation
 *
 * @params[in] CTRL_CMD_FB_CBC_OP fbCBCOp
 *      CBC Operation requested.
 *      Valid Values:
 *          CTRL_CMD_FB_CBC_OP_CLEAN
 *          CTRL_CMD_FB_CBC_OP_ILWALIDATE
 *
 * Possible status values returned are:
 *   LW_OK LW_ERR_NOT_SUPPORTED LW_ERR_ILWALID_ARGUMENT LW_ERR_TIMEOUT
 */
#define LW2080_CTRL_CMD_FB_CBC_OP (0x20801337) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FB_CBC_OP_PARAMS_MESSAGE_ID" */

/*!
 * Permitted CBC Operations
 */
typedef enum CTRL_CMD_FB_CBC_OP {
    CTRL_CMD_FB_CBC_OP_CLEAN = 0,
    CTRL_CMD_FB_CBC_OP_ILWALIDATE = 1,
} CTRL_CMD_FB_CBC_OP;

#define LW2080_CTRL_CMD_FB_CBC_OP_PARAMS_MESSAGE_ID (0x37U)

typedef struct LW2080_CTRL_CMD_FB_CBC_OP_PARAMS {
    CTRL_CMD_FB_CBC_OP fbCBCOp;
} LW2080_CTRL_CMD_FB_CBC_OP_PARAMS;

/*!
 *  LW2080_CTRL_CMD_FB_GET_CTAGS_FOR_CBC_EVICTION
 *
 *  The call will fetch the compression tags reserved for CBC eviction.
 *
 *  Each comptag will correspond to a unique compression cacheline. The usage of
 *  these comptags is to evict the CBC by making accesses to a dummy compressed page,
 *  thereby evicting each CBC line.
 *
 *  @param [in][out] LwU32 pCompTags
 *     Array of reserved compression tags of size @ref LW2080_MAX_CTAGS_FOR_CBC_EVICTION
 *  @param [out] numCompTags
 *     Number of entries returned in @ref pCompTags
 *
 *  @returns
 *      LW_OK
 *      LW_ERR_ILWALID_STATE
 *      LW_ERR_OUT_OF_RANGE
 *      LW_ERR_ILWALID_PARAMETER
 */
#define LW2080_CTRL_CMD_FB_GET_CTAGS_FOR_CBC_EVICTION (0x20801338) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_CTAGS_FOR_CBC_EVICTION_PARAMS_MESSAGE_ID" */

/*!
 * Max size of @ref LW2080_CTRL_FB_GET_CTAGS_FOR_CBC_EVICTION_PARAMS::pCompTags
 * Arbitrary, but sufficiently large number. Should be checked against CBC size.
 */
#define LW2080_MAX_CTAGS_FOR_CBC_EVICTION             0x7F


#define LW2080_CTRL_FB_GET_CTAGS_FOR_CBC_EVICTION_PARAMS_MESSAGE_ID (0x38U)

typedef struct LW2080_CTRL_FB_GET_CTAGS_FOR_CBC_EVICTION_PARAMS {
    LwU32 pCompTags[LW2080_MAX_CTAGS_FOR_CBC_EVICTION];
    LwU32 numCompTags;
} LW2080_CTRL_FB_GET_CTAGS_FOR_CBC_EVICTION_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_ALLOC_COMP_RESOURCE
 *
 * This Call will allocate compression tag
 *
 * @params[in] LwU32 attr
 *      Stores the information:
 *          1. LWOS32_ATTR_COMPR_REQUIRED or not
 *          2. LWOS32_ATTR_PAGE_SIZE
 * @params[in] LwU32 attr2
 *      Determine whether to allocate
 *      an entire cache line or allocate by size
 * @params[in] LwU32 size
 *      Specify the size of allocation, in pages not bytes
 * @params[in] LwU32 ctagOffset
 *      Determine the offset usage of the allocation
 * @params[out] LwU32 hwResId
 *      Stores the result of the allocation
 * @params[out] LwU32 RetcompTagLineMin
 *      The resulting min Ctag Number from the allocation
 * @params[out] LwU32 RetcompTagLineMax
 *      The resulting max Ctag Number from the allocation
 * @returns
 *      LW_OK
 *      LW_ERR_INSUFFICIENT_RESOURCES
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_ILWALID_STATE
 */

#define LW2080_CTRL_CMD_FB_ALLOC_COMP_RESOURCE (0x20801339) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FB_ALLOC_COMP_RESOURCE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_FB_ALLOC_COMP_RESOURCE_PARAMS_MESSAGE_ID (0x39U)

typedef struct LW2080_CTRL_CMD_FB_ALLOC_COMP_RESOURCE_PARAMS {
    LwU32 attr;
    LwU32 attr2;
    LwU32 size;
    LwU32 ctagOffset;
    LwU32 hwResId;
    LwU32 retCompTagLineMin;
    LwU32 retCompTagLineMax;
} LW2080_CTRL_CMD_FB_ALLOC_COMP_RESOURCE_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_FREE_TILE
 *
 * This control call is used to release tile back to the free pool
 *
 * @params[in] LwU32 hwResId
 *      Stores the information of a previous allocation
 * @returns
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_ILWALID_STATE
 */

#define LW2080_CTRL_CMD_FB_FREE_TILE (0x2080133a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FB_FREE_TILE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_FB_FREE_TILE_PARAMS_MESSAGE_ID (0x3AU)

typedef struct LW2080_CTRL_CMD_FB_FREE_TILE_PARAMS {
    LwU32 hwResId;
} LW2080_CTRL_CMD_FB_FREE_TILE_PARAMS;


/*
 * LW2080_CTRL_CMD_FB_SETUP_VPR_REGION
 *
 * This control command is used to request vpr region setup
 *
 * requestType
 *          LW2080_CTRL_CMD_FB_SET_VPR: Request to setup VPR
 *
 * requestParams : struct VPR_REQUEST_PARAMS
 *          It contains the VPR region request details like,
 *          startAddr : FB offset from which we need to setup VPR
 *          size      : required size of the region
 *
 * statusParams  : struct VPR_STATUS_PARAMS
 *          This struct stores the output of requested VPR region
 *          status    : Whether the request was successful
 *
 * LW2080_CTRL_CMD_FB_VPR_ERROR_CODE :
 *          LW2080_CTRL_CMD_FB_VPR_ERROR_GENERIC : Some unknown error oclwrred
 *          LW2080_CTRL_CMD_FB_VPR_ERROR_ILWALID_CLIENT_REQUEST : Request was invalid
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_SETUP_VPR_REGION (0x2080133b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FB_SETUP_VPR_REGION_PARAMS_MESSAGE_ID" */

typedef enum LW2080_CTRL_CMD_FB_VPR_REQUEST_TYPE {
    LW2080_CTRL_CMD_FB_SET_VPR = 0,
} LW2080_CTRL_CMD_FB_VPR_REQUEST_TYPE;

typedef enum LW2080_CTRL_CMD_FB_VPR_ERROR_CODE {
    LW2080_CTRL_CMD_FB_VPR_ERROR_GENERIC = 0,
    LW2080_CTRL_CMD_FB_VPR_ERROR_ILWALID_CLIENT_REQUEST = 1,
} LW2080_CTRL_CMD_FB_VPR_ERROR_CODE;

typedef struct VPR_REQUEST_PARAMS {
    LwU32 startAddr;
    LwU32 size;
} VPR_REQUEST_PARAMS;

typedef struct VPR_STATUS_PARAMS {
    LwU32 status;
} VPR_STATUS_PARAMS;

#define LW2080_CTRL_CMD_FB_SETUP_VPR_REGION_PARAMS_MESSAGE_ID (0x3BU)

typedef struct LW2080_CTRL_CMD_FB_SETUP_VPR_REGION_PARAMS {
    LW2080_CTRL_CMD_FB_VPR_REQUEST_TYPE requestType;
    VPR_REQUEST_PARAMS                  requestParams;
    VPR_STATUS_PARAMS                   statusParams;
} LW2080_CTRL_CMD_FB_SETUP_VPR_REGION_PARAMS;
typedef struct LW2080_CTRL_CMD_FB_SETUP_VPR_REGION_PARAMS *PLW2080_CTRL_CMD_FB_SETUP_VPR_REGION_PARAMS;

/*
 * LW2080_CTRL_CMD_FB_GET_CLI_MANAGED_OFFLINED_PAGES
 *
 * This command returns the list of offlined video memory page addresses in the
 * region managed by Client
 *
 *   offlinedPages
 *     This output parameter is an array of video memory physical page numbers that
 *     are offlined. This array can hold a maximum of LW2080_CTRL_FB_
 *     OFFLINED_PAGES_MAX_PAGES addresses.
 *   pageSize
 *     This output parameter contains the size of the page that is offlined.
 *   validEntries
 *     This output parameter specifies the number of valid entries in the
 *     offlined array.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_FB_GET_CLI_MANAGED_OFFLINED_PAGES (0x2080133c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_CLI_MANAGED_OFFLINED_PAGES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_GET_CLI_MANAGED_OFFLINED_PAGES_PARAMS_MESSAGE_ID (0x3LW)

typedef struct LW2080_CTRL_FB_GET_CLI_MANAGED_OFFLINED_PAGES_PARAMS {
    LwU32 offlinedPages[LW2080_CTRL_FB_OFFLINED_PAGES_MAX_PAGES];    // A 32B can hold enough.
    LwU32 pageSize;
    LwU32 validEntries;
} LW2080_CTRL_FB_GET_CLI_MANAGED_OFFLINED_PAGES_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_GET_COMPBITCOPY_CONSTRUCT_INFO
 *
 * This command returns parameters required to initialize compbit copy object
 * used by address mapping library
 *
 *   defaultPageSize
 *     Page size used by @ref CompBitCopy methods
 *   comptagsPerCacheLine
 *     Number of compression tags in a single compression cache line.
 *   unpackedComptagLinesPerCacheLine;
 *     From hw (not adjusted for CompBits code)  Number of compression tags
 *     in a single compression cache line.
 *   compCacheLineSizePerLTC;
 *     Size of compression cache line per L2 slice. Size in Bytes.
 *   unpackedCompCacheLineSizePerLTC;
 *     From hw (not adjusted for CompBits code) size of compression
 *     cache line per L2 slice. Size in Bytes
 *   slicesPerLTC;
 *     Number of L2 slices per L2 cache.
 *   numActiveLTCs;
 *     Number of active L2 caches. (Not floorswept)
 *   familyName;
 *     Family name for the GPU.
 *   chipName;
 *     Chip name for the GPU.
 *   bitsPerRAMEntry;
 *     Bits per RAM entry. (Need better doc)
 *   ramBankWidth;
 *     Width of RAM bank. (Need better doc)
 *   bitsPerComptagLine;
 *     Number of bits per compression tag line.
 *   ramEntriesPerCompCacheLine;
 *     Number of RAM entries spanned by 1 compression cache line.
 *   comptagLineSize;
 *     Size of compression tag line, in Bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LW2080_CTRL_CMD_FB_GET_COMPBITCOPY_CONSTRUCT_INFO (0x2080133d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FB_GET_COMPBITCOPY_CONSTRUCT_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_FB_GET_COMPBITCOPY_CONSTRUCT_INFO_PARAMS_MESSAGE_ID (0x3DU)

typedef struct LW2080_CTRL_CMD_FB_GET_COMPBITCOPY_CONSTRUCT_INFO_PARAMS {
    LwU32 defaultPageSize;
    LwU32 comptagsPerCacheLine;
    LwU32 unpackedComptagLinesPerCacheLine;
    LwU32 compCacheLineSizePerLTC;
    LwU32 unpackedCompCacheLineSizePerLTC;
    LwU32 slicesPerLTC;
    LwU32 numActiveLTCs;
    LwU32 familyName;
    LwU32 chipName;
    LwU32 bitsPerRAMEntry;
    LwU32 ramBankWidth;
    LwU32 bitsPerComptagLine;
    LwU32 ramEntriesPerCompCacheLine;
    LwU32 comptagLineSize;
} LW2080_CTRL_CMD_FB_GET_COMPBITCOPY_CONSTRUCT_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_FB_SET_RRD
 *
 * Sets the row-to-row delay on the GPU's FB
 *
 * Possible status values returned are:
 *  LW_OK
 *  Any error code
 */
#define LW2080_CTRL_CMD_FB_SET_RRD (0x2080133e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_SET_RRD_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_SET_RRD_RESET_VALUE (~((LwU32)0))
#define LW2080_CTRL_FB_SET_RRD_PARAMS_MESSAGE_ID (0x3EU)

typedef struct LW2080_CTRL_FB_SET_RRD_PARAMS {
    LwU32 rrd;
} LW2080_CTRL_FB_SET_RRD_PARAMS;

/*
 * LW2080_CTRL_FB_SET_READ_WRITE_LIMIT_PARAMS
 *
 * This is not a control call of it's own, but there are common definitions for
 * the two LW2080_CTRL_CMD_FB_SET_READ/WRITE_LIMIT control calls.
 */
typedef struct LW2080_CTRL_FB_SET_READ_WRITE_LIMIT_PARAMS {
    LwU8 limit;
} LW2080_CTRL_FB_SET_READ_WRITE_LIMIT_PARAMS;
#define LW2080_CTRL_FB_SET_READ_WRITE_LIMIT_RESET_VALUE (0xff)

/*
 * LW2080_CTRL_CMD_FB_SET_READ_LIMIT
 *
 * Sets the READ_LIMIT to be used in the LW_PFB_FBPA_DIR_ARB_CFG0 register
 *
 *  limit
 *      The limit value to use
 *
 * Possible status values returned are:
 *  LW_OK
 *  Any error code
 */
#define LW2080_CTRL_CMD_FB_SET_READ_LIMIT               (0x2080133f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_SET_READ_LIMIT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_SET_READ_LIMIT_RESET_VALUE       LW2080_CTRL_FB_SET_READ_WRITE_LIMIT_RESET_VALUE
#define LW2080_CTRL_FB_SET_READ_LIMIT_PARAMS_MESSAGE_ID (0x3FU)

typedef LW2080_CTRL_FB_SET_READ_WRITE_LIMIT_PARAMS LW2080_CTRL_FB_SET_READ_LIMIT_PARAMS;

/*
 * LW2080_CTRL_CMD_FB_SET_WRITE_LIMIT
 *
 * Sets the WRITE_LIMIT to be used in the LW_PFB_FBPA_DIR_ARB_CFG0 register
 *
 *  limit
 *      The limit value to us
 *
 * Possible status values returned are:
 *  LW_OK
 *  Any error code
 */
#define LW2080_CTRL_CMD_FB_SET_WRITE_LIMIT         (0x20801340) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_SET_WRITE_LIMIT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_SET_WRITE_LIMIT_RESET_VALUE LW2080_CTRL_FB_SET_READ_WRITE_LIMIT_RESET_VALUE
#define LW2080_CTRL_FB_SET_WRITE_LIMIT_PARAMS_MESSAGE_ID (0x40U)

typedef LW2080_CTRL_FB_SET_READ_WRITE_LIMIT_PARAMS LW2080_CTRL_FB_SET_WRITE_LIMIT_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_PATCH_PBR_FOR_MINING
 *
 * Patches some VBIOS values related to PBR to better suit mining applications
 *
 *  bEnable
 *      Set the mining-specific values or reset to the original values
 *
 * Possible status values returned are:
 *  LW_OK
 *  Any error code
 */
#define LW2080_CTRL_CMD_FB_PATCH_PBR_FOR_MINING (0x20801341) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_PATCH_PBR_FOR_MINING_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_PATCH_PBR_FOR_MINING_PARAMS_MESSAGE_ID (0x41U)

typedef struct LW2080_CTRL_FB_PATCH_PBR_FOR_MINING_PARAMS {
    LwBool bEnable;
} LW2080_CTRL_FB_PATCH_PBR_FOR_MINING_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_GET_MEM_ALIGNMENT
 *
 * Get memory alignment. Replacement for LWOS32_FUNCTION_GET_MEM_ALIGNMENT
 */
#define LW2080_CTRL_CMD_FB_GET_MEM_ALIGNMENT       (0x20801342) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_MEM_ALIGNMENT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_GET_MEM_ALIGNMENT_MAX_BANKS (4)
#define LW2080_CTRL_FB_GET_MEM_ALIGNMENT_PARAMS_MESSAGE_ID (0x42U)

typedef struct LW2080_CTRL_FB_GET_MEM_ALIGNMENT_PARAMS {
    LwU32 alignType;                                 // Input
    LwU32 alignAttr;
    LwU32 alignInputFlags;
    LwU32 alignHead;
    LW_DECLARE_ALIGNED(LwU64 alignSize, 8);
    LwU32 alignHeight;
    LwU32 alignWidth;
    LwU32 alignPitch;
    LwU32 alignPad;
    LwU32 alignMask;
    LwU32 alignOutputFlags[LW2080_CTRL_FB_GET_MEM_ALIGNMENT_MAX_BANKS];
    LwU32 alignBank[LW2080_CTRL_FB_GET_MEM_ALIGNMENT_MAX_BANKS];
    LwU32 alignKind;
    LwU32 alignAdjust;                                // Output -- If non-zero the amount we need to adjust the offset
    LwU32 alignAttr2;
} LW2080_CTRL_FB_GET_MEM_ALIGNMENT_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_GET_CBC_BASEADDR
 *
 * Get the CBC Base physical address
 * This control call is required by error containment tests
 * LW2080_CTRL_CMD_FB_GET_AMAP_CONF can also return CBC base address
 * but it requires kernel privilege, and it not callalble from SRT test
 *
 * @params[out] LwU64 cbcBaseAddr
 *     Base physical address for CBC data.
 *
 * Possible status values returned are:
 *   LW_OK LW_ERR_NOT_SUPPORTED
 *
 */
#define LW2080_CTRL_CMD_FB_GET_CBC_BASE_ADDR (0x20801343) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FB_GET_CBC_BASE_ADDR_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_FB_GET_CBC_BASE_ADDR_PARAMS_MESSAGE_ID (0x43U)

typedef struct LW2080_CTRL_CMD_FB_GET_CBC_BASE_ADDR_PARAMS {
    LwU32 cbcBaseAddress;
    LwU32 compCacheLineSize;
    LW_DECLARE_ALIGNED(LwU64 backingStoreStartPA, 8);
    LW_DECLARE_ALIGNED(LwU64 backingStoreAllocPA, 8);
    LwU32 backingStoreChunkOverfetch;
} LW2080_CTRL_CMD_FB_GET_CBC_BASE_ADDR_PARAMS;

#define LW2080_CTRL_FB_REMAP_ENTRY_FLAGS_PENDING                             0:0
#define LW2080_CTRL_FB_REMAP_ENTRY_FLAGS_PENDING_FALSE 0
#define LW2080_CTRL_FB_REMAP_ENTRY_FLAGS_PENDING_TRUE  1

typedef struct LW2080_CTRL_FB_REMAP_ENTRY {
    LwU32 remapRegVal; // Exact value programmed by GFW ucode
    LwU32 timestamp;
    LwU8  fbpa;
    LwU8  sublocation;
    LwU8  source;
    LwU8  flags;
} LW2080_CTRL_FB_REMAP_ENTRY;

/* valid values for source */
#define LW2080_CTRL_FB_REMAPPED_ROW_SOURCE_SBE_FACTORY       (0x00000000)
#define LW2080_CTRL_FB_REMAPPED_ROW_SOURCE_DBE_FACTORY       (0x00000001)
#define LW2080_CTRL_FB_REMAPPED_ROW_SOURCE_SBE_FIELD         (0x00000002)
#define LW2080_CTRL_FB_REMAPPED_ROW_SOURCE_DBE_FIELD         (0x00000003)
#define LW2080_CTRL_FB_REMAPPED_ROW_SOURCE_MODS_MEM_ERROR    (0x00000004)

#define LW2080_CTRL_FB_REMAPPED_ROWS_MAX_ROWS                (0x00000200)

/*
 * LW2080_CTRL_CMD_FB_GET_REMAPPED_ROWS
 *
 * This command returns the list of remapped rows stored in the Inforom.
 *
 *   entryCount
 *     This output parameter specifies the number of remapped rows
 *   flags
 *     This output parameter right now contains info on whether or not
 *     there are pending remappings and whether or not a remapping failed
 *   entries
 *     This output parameter is an array of LW2080_CTRL_FB_REMAP_ENTRY
 *     containing inforomation on the remapping that oclwrred. This array can
 *     hold a maximum of LW2080_CTRL_FB_REMAPPED_ROWS_MAX_ROWS
 *
 *  Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_ILWALID_POINTER
 *    LW_ERR_OBJECT_NOT_FOUND
 *    LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_GET_REMAPPED_ROWS                 (0x20801344) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_REMAPPED_ROWS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_GET_REMAPPED_ROWS_FLAGS_PENDING                       \
    LW2080_CTRL_FB_REMAP_ENTRY_FLAGS_PENDING
#define LW2080_CTRL_FB_GET_REMAPPED_ROWS_FLAGS_PENDING_FALSE LW2080_CTRL_FB_REMAP_ENTRY_FLAGS_PENDING_FALSE
#define LW2080_CTRL_FB_GET_REMAPPED_ROWS_FLAGS_PENDING_TRUE  LW2080_CTRL_FB_REMAP_ENTRY_FLAGS_PENDING_TRUE
#define LW2080_CTRL_FB_GET_REMAPPED_ROWS_FLAGS_FAILURE                       1:1
#define LW2080_CTRL_FB_GET_REMAPPED_ROWS_FLAGS_FAILURE_FALSE 0
#define LW2080_CTRL_FB_GET_REMAPPED_ROWS_FLAGS_FAILURE_TRUE  1

#define LW2080_CTRL_FB_GET_REMAPPED_ROWS_PARAMS_MESSAGE_ID (0x44U)

typedef struct LW2080_CTRL_FB_GET_REMAPPED_ROWS_PARAMS {
    LwU32                      entryCount;
    LwU8                       flags;
    LW2080_CTRL_FB_REMAP_ENTRY entries[LW2080_CTRL_FB_REMAPPED_ROWS_MAX_ROWS];
} LW2080_CTRL_FB_GET_REMAPPED_ROWS_PARAMS;

// Max size of the queryParams in Bytes, so that the LW2080_CTRL_FB_FS_INFO_QUERY struct is still 32B
#define LW2080_CTRL_FB_FS_INFO_MAX_QUERY_SIZE 24

/*!
 * Structure holding the out params for LW2080_CTRL_FB_FS_INFO_ILWALID_QUERY.
 */
typedef struct LW2080_CTRL_FB_FS_INFO_ILWALID_QUERY_PARAMS {
    // Unused param, will ensure the size of LW2080_CTRL_FB_FS_INFO_QUERY struct to be 32B
    LwU8 data[LW2080_CTRL_FB_FS_INFO_MAX_QUERY_SIZE];
} LW2080_CTRL_FB_FS_INFO_ILWALID_QUERY_PARAMS;

/*!
 * Structure holding the in/out params for LW2080_CTRL_FB_FS_INFO_FBP_MASK.
 */
typedef struct LW2080_CTRL_FB_FS_INFO_FBP_MASK_PARAMS {
    /*!
     * [IN]: swizzId
     * PartitionID associated with a created smc partition. Lwrrently used only for a
     * device monitoring client to get the physical values of the FB. The client needs to pass
     * 'LW2080_CTRL_GPU_PARTITION_ID_ILWALID' explicitly if it wants RM to ignore the swizzId.
     * RM will consider this request similar to a legacy case.
     * The client's subscription is used only as a capability check and not as an input swizzId.
     */
    LwU32 swizzId;
    /*!
     * [OUT]: physical/local fbp mask.
     */
    LW_DECLARE_ALIGNED(LwU64 fbpEnMask, 8);
} LW2080_CTRL_FB_FS_INFO_FBP_MASK_PARAMS;

/*!
 * Structure holding the in/out params for LW2080_CTRL_FB_FS_INFO_LTC_MASK.
 */
typedef struct LW2080_CTRL_FB_FS_INFO_LTC_MASK_PARAMS {
    /*!
     * [IN]: physical/local FB partition index.
     */
    LwU32 fbpIndex;
    /*!
     * [OUT]: physical/local ltc mask.
     */
    LwU32 ltcEnMask;
} LW2080_CTRL_FB_FS_INFO_LTC_MASK_PARAMS;

/*!
 * Structure holding the in/out params for LW2080_CTRL_FB_FS_INFO_LTS_MASK.
 */
typedef struct LW2080_CTRL_FB_FS_INFO_LTS_MASK_PARAMS {
    /*!
     * [IN]: physical/local FB partition index.
     */
    LwU32 fbpIndex;
    /*!
     * [OUT]: physical/local lts mask.
     * Note that lts bits are flattened out for all ltc with in a fbp.
     */
    LwU32 ltsEnMask;
} LW2080_CTRL_FB_FS_INFO_LTS_MASK_PARAMS;

/*!
 * Structure holding the in/out params for LW2080_CTRL_FB_FS_INFO_FBPA_MASK.
 */
typedef struct LW2080_CTRL_FB_FS_INFO_FBPA_MASK_PARAMS {
    /*!
     * [IN]: physical/local FB partition index.
     */
    LwU32 fbpIndex;
    /*!
     * [OUT]: physical/local FBPA mask.
     */
    LwU32 fbpaEnMask;
} LW2080_CTRL_FB_FS_INFO_FBPA_MASK_PARAMS;

/*!
 * Structure holding the in/out params for LW2080_CTRL_FB_FS_INFO_FBPA_SUBP_MASK.
 */
typedef struct LW2080_CTRL_FB_FS_INFO_FBPA_SUBP_MASK_PARAMS {
    /*!
     * [IN]: physical/local FB partition index.
     */
    LwU32 fbpIndex;
    /*!
     * [OUT]: physical/local FBPA-SubPartition mask.
     */
    LwU32 fbpaSubpEnMask;
} LW2080_CTRL_FB_FS_INFO_FBPA_SUBP_MASK_PARAMS;

/*!
 * Structure holding the in/out params for LW2080_CTRL_FB_FS_INFO_FBP_LOGICAL_MAP
 */
typedef struct LW2080_CTRL_FB_FS_INFO_FBP_LOGICAL_MAP_PARAMS {
    /*!
     * [IN]: physical/local FB partition index.
     */
    LwU32 fbpIndex;
    /*!
     * [OUT]: Logical/local FBP index
     */
    LwU32 fbpLogicalIndex;
} LW2080_CTRL_FB_FS_INFO_FBP_LOGICAL_MAP_PARAMS;

/*!
 * Structure holding the in/out params for LW2080_CTRL_FB_FS_INFO_ROP_MASK.
 */
typedef struct LW2080_CTRL_FB_FS_INFO_ROP_MASK_PARAMS {
    /*!
     * [IN]: physical/local FB partition index.
     */
    LwU32 fbpIndex;
    /*!
     * [OUT]: physical/local ROP mask.
     */
    LwU32 ropEnMask;
} LW2080_CTRL_FB_FS_INFO_ROP_MASK_PARAMS;

/*!
 * Structure holding the in/out params for LW2080_CTRL_FB_FS_INFO_PROFILER_MON_LTC_MASK.
 */
typedef struct LW2080_CTRL_FB_FS_INFO_PROFILER_MON_LTC_MASK_PARAMS {
    /*!
     * [IN]: Physical FB partition index.
     */
    LwU32 fbpIndex;
    /*!
     * [IN]: swizzId
     * PartitionID associated with a created smc partition.
     */
    LwU32 swizzId;
    /*!
     * [OUT]: physical ltc mask.
     */
    LwU32 ltcEnMask;
} LW2080_CTRL_FB_FS_INFO_PROFILER_MON_LTC_MASK_PARAMS;

/*!
 * Structure holding the in/out params for LW2080_CTRL_FB_FS_INFO_PROFILER_MON_LTS_MASK.
 */
typedef struct LW2080_CTRL_FB_FS_INFO_PROFILER_MON_LTS_MASK_PARAMS {
    /*!
     * [IN]: Physical FB partition index.
     */
    LwU32 fbpIndex;
    /*!
     * [IN]: swizzId
     * PartitionID associated with a created smc partition.
     */
    LwU32 swizzId;
    /*!
     * [OUT]: physical lts mask.
     */
    LwU32 ltsEnMask;
} LW2080_CTRL_FB_FS_INFO_PROFILER_MON_LTS_MASK_PARAMS;

/*!
 * Structure holding the in/out params for LW2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_MASK.
 */
typedef struct LW2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_MASK_PARAMS {
    /*!
     * [IN]: Physical FB partition index.
     */
    LwU32 fbpIndex;
    /*!
     * [IN]: swizzId
     * PartitionID associated with a created smc partition.
     */
    LwU32 swizzId;
    /*!
     * [OUT]: physical fbpa mask.
     */
    LwU32 fbpaEnMask;
} LW2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_MASK_PARAMS;

/*!
 * Structure holding the in/out params for LW2080_CTRL_FB_FS_INFO_PROFILER_MON_ROP_MASK.
 */
typedef struct LW2080_CTRL_FB_FS_INFO_PROFILER_MON_ROP_MASK_PARAMS {
    /*!
     * [IN]: Physical FB partition index.
     */
    LwU32 fbpIndex;
    /*!
     * [IN]: swizzId
     * PartitionID associated with a created smc partition.
     */
    LwU32 swizzId;
    /*!
     * [OUT]: physical rop mask.
     */
    LwU32 ropEnMask;
} LW2080_CTRL_FB_FS_INFO_PROFILER_MON_ROP_MASK_PARAMS;

/*!
 * Structure holding the in/out params for LW2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_SUBP_MASK.
 */
typedef struct LW2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_SUBP_MASK_PARAMS {
    /*!
     * [IN]: Physical FB partition index.
     */
    LwU32 fbpIndex;
    /*!
     * [IN]: swizzId
     * PartitionID associated with a created smc partition. Lwrrently used only for a
     * device monitoring client to get the physical values of the FB. The client needs to pass
     * 'LW2080_CTRL_GPU_PARTITION_ID_ILWALID' explicitly if it wants RM to ignore the swizzId.
     * RM will consider this request similar to a legacy case.
     * The client's subscription is used only as a capability check and not as an input swizzId.
     */
    LwU32 swizzId;
    /*!
     * [OUT]: physical FBPA_SubPartition mask associated with requested partition.
     */
    LW_DECLARE_ALIGNED(LwU64 fbpaSubpEnMask, 8);
} LW2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_SUBP_MASK_PARAMS;

// Possible values for queryType
#define LW2080_CTRL_FB_FS_INFO_ILWALID_QUERY               0x0
#define LW2080_CTRL_FB_FS_INFO_FBP_MASK                    0x1
#define LW2080_CTRL_FB_FS_INFO_LTC_MASK                    0x2
#define LW2080_CTRL_FB_FS_INFO_LTS_MASK                    0x3
#define LW2080_CTRL_FB_FS_INFO_FBPA_MASK                   0x4
#define LW2080_CTRL_FB_FS_INFO_ROP_MASK                    0x5
#define LW2080_CTRL_FB_FS_INFO_PROFILER_MON_LTC_MASK       0x6
#define LW2080_CTRL_FB_FS_INFO_PROFILER_MON_LTS_MASK       0x7
#define LW2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_MASK      0x8
#define LW2080_CTRL_FB_FS_INFO_PROFILER_MON_ROP_MASK       0x9
#define LW2080_CTRL_FB_FS_INFO_FBPA_SUBP_MASK              0xA
#define LW2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_SUBP_MASK 0xB
#define LW2080_CTRL_FB_FS_INFO_FBP_LOGICAL_MAP             0xC

typedef struct LW2080_CTRL_FB_FS_INFO_QUERY {
    LwU16 queryType;
    LwU8  reserved[2];
    LwU32 status;
    union {
        LW2080_CTRL_FB_FS_INFO_ILWALID_QUERY_PARAMS          ilw;
        LW_DECLARE_ALIGNED(LW2080_CTRL_FB_FS_INFO_FBP_MASK_PARAMS fbp, 8);
        LW2080_CTRL_FB_FS_INFO_LTC_MASK_PARAMS               ltc;
        LW2080_CTRL_FB_FS_INFO_LTS_MASK_PARAMS               lts;
        LW2080_CTRL_FB_FS_INFO_FBPA_MASK_PARAMS              fbpa;
        LW2080_CTRL_FB_FS_INFO_ROP_MASK_PARAMS               rop;
        LW2080_CTRL_FB_FS_INFO_PROFILER_MON_LTC_MASK_PARAMS  dmLtc;
        LW2080_CTRL_FB_FS_INFO_PROFILER_MON_LTS_MASK_PARAMS  dmLts;
        LW2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_MASK_PARAMS dmFbpa;
        LW2080_CTRL_FB_FS_INFO_PROFILER_MON_ROP_MASK_PARAMS  dmRop;
        LW_DECLARE_ALIGNED(LW2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_SUBP_MASK_PARAMS dmFbpaSubp, 8);
        LW2080_CTRL_FB_FS_INFO_FBPA_SUBP_MASK_PARAMS         fbpaSubp;
        LW2080_CTRL_FB_FS_INFO_FBP_LOGICAL_MAP_PARAMS        fbpLogicalMap;
    } queryParams;
} LW2080_CTRL_FB_FS_INFO_QUERY;

// Max number of queries that can be batched in a single call to LW2080_CTRL_CMD_FB_GET_FS_INFO
#define LW2080_CTRL_FB_FS_INFO_MAX_QUERIES 96

#define LW2080_CTRL_FB_GET_FS_INFO_PARAMS_MESSAGE_ID (0x46U)

typedef struct LW2080_CTRL_FB_GET_FS_INFO_PARAMS {
    LwU16 numQueries;
    LwU8  reserved[6];
    LW_DECLARE_ALIGNED(LW2080_CTRL_FB_FS_INFO_QUERY queries[LW2080_CTRL_FB_FS_INFO_MAX_QUERIES], 8);
} LW2080_CTRL_FB_GET_FS_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_GET_FS_INFO
 *
 * This control call returns the fb engine information for a partition/GPU.
 * Supports an interface so that the caller can issue multiple queries by batching them
 * in a single call. Returns the first error it encounters.
 *
 * numQueries[IN]
 *     - Specifies the number of valid queries.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_GET_FS_INFO                             (0x20801346) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_FS_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_HISTOGRAM_IDX_NO_REMAPPED_ROWS              (0x0)
#define LW2080_CTRL_FB_HISTOGRAM_IDX_SINGLE_REMAPPED_ROW           (0x1)
#define LW2080_CTRL_FB_HISTOGRAM_IDX_MIXED_REMAPPED_REMAINING_ROWS (0x2)
#define LW2080_CTRL_FB_HISTOGRAM_IDX_SINGLE_REMAINING_ROW          (0x3)
#define LW2080_CTRL_FB_HISTOGRAM_IDX_MAX_REMAPPED_ROWS             (0x4)

#define LW2080_CTRL_FB_GET_ROW_REMAPPER_HISTOGRAM_PARAMS_MESSAGE_ID (0x47U)

typedef struct LW2080_CTRL_FB_GET_ROW_REMAPPER_HISTOGRAM_PARAMS {
    LwU32 histogram[5];
} LW2080_CTRL_FB_GET_ROW_REMAPPER_HISTOGRAM_PARAMS;

/*!
 * LW2080_CTRL_CMD_FB_GET_ROW_REMAPPER_HISTOGRAM
 *
 * This control call returns stats on the number of banks that have a certain
 * number of rows remapped in the bank. Specifically the number of banks that
 * have 0, 1, 2 through (max-2), max-1 and max number of rows remapped in the
 * bank. Values will be returned in an array.
 *
 * Index values are:
 *
 *   LW2080_CTRL_FB_HISTOGRAM_IDX_NO_REMAPPED_ROWS
 *     Number of banks with zero rows remapped
     LW2080_CTRL_FB_HISTOGRAM_IDX_SINGLE_REMAPPED_ROW
 *     Number of banks with one row remapped
     LW2080_CTRL_FB_HISTOGRAM_IDX_MIXED_REMAPPED_REMAINING_ROWS
 *     Number of banks with 2 through (max-2) rows remapped
     LW2080_CTRL_FB_HISTOGRAM_IDX_SINGLE_REMAINING_ROW
 *     Number of banks with (max-1) rows remapped
     LW2080_CTRL_FB_HISTOGRAM_IDX_MAX_REMAPPED_ROWS
 *     Number of banks with max rows remapped
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FB_GET_ROW_REMAPPER_HISTOGRAM (0x20801347) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_ROW_REMAPPER_HISTOGRAM_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_FB_GET_DYNAMICALLY_BLACKLISTED_PAGES
 *
 * This command returns the list of dynamically blacklisted video memory page addresses
 * after last driver load.
 *
 *   blackList
 *     This output parameter is an array of LW2080_CTRL_FB_DYNAMIC_BLACKLIST_ADDRESS_INFO
 *     This array can hold a maximum of LW2080_CTRL_FB_DYNAMIC_BLACKLIST_MAX_ENTRIES.
 *   validEntries
 *     This output parameter specifies the number of valid entries in the
 *     blackList array.
 *   baseIndex
 *     With the limit of up to 512 blacklisted pages, the size of this array
 *     exceeds the rpc buffer limit. This control call will collect the data
 *     in multiple passes. This parameter indicates the start index of the
 *     data to be passed back to the caller
 *     This cannot be greater than LW2080_CTRL_FB_DYNAMIC_BLACKLIST_MAX_PAGES
 *   bMore
 *     This parameter indicates whether there are more valid elements to be
 *     fetched.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_FB_GET_DYNAMIC_OFFLINED_PAGES (0x20801348) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_DYNAMIC_OFFLINED_PAGES_PARAMS_MESSAGE_ID" */

/* Maximum pages that can be dynamically blacklisted */
#define LW2080_CTRL_FB_DYNAMIC_BLACKLIST_MAX_PAGES    512

/*
 * Maximum entries that can be sent in a single pass of 
 * LW2080_CTRL_CMD_FB_GET_DYNAMIC_OFFLINED_PAGES
 */
#define LW2080_CTRL_FB_DYNAMIC_BLACKLIST_MAX_ENTRIES  64

/**
 * LW2080_CTRL_FB_DYNAMIC_OFFLINED_ADDRESS_INFO
 *   pageNumber
 *     This output parameter specifies the dynamically blacklisted page number.
 *   source
 *     The reason for the page to be retired. Valid values for
 *     this parameter include:
 *        LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_ILWALID
 *           Invalid source.
 *        LW2080_CTRL_FB_OFFLINED_PAGES_SOURCE_DPR_DBE
 *           Page retired by dynamic page retirement due to a double bit
 *           error seen.
 */
typedef struct LW2080_CTRL_FB_DYNAMIC_OFFLINED_ADDRESS_INFO {
    LW_DECLARE_ALIGNED(LwU64 pageNumber, 8);
    LwU8 source;
} LW2080_CTRL_FB_DYNAMIC_OFFLINED_ADDRESS_INFO;

#define LW2080_CTRL_FB_GET_DYNAMIC_OFFLINED_PAGES_PARAMS_MESSAGE_ID (0x48U)

typedef struct LW2080_CTRL_FB_GET_DYNAMIC_OFFLINED_PAGES_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_FB_DYNAMIC_OFFLINED_ADDRESS_INFO offlined[LW2080_CTRL_FB_DYNAMIC_BLACKLIST_MAX_ENTRIES], 8);
    LwU32  validEntries;
    LwU32  baseIndex;
    LwBool bMore;
} LW2080_CTRL_FB_GET_DYNAMIC_OFFLINED_PAGES_PARAMS;

/* valid values for source */

#define LW2080_CTRL_FB_DYNAMIC_BLACKLISTED_PAGES_SOURCE_ILWALID (0x00000000)
#define LW2080_CTRL_FB_DYNAMIC_BLACKLISTED_PAGES_SOURCE_DPR_DBE (0x00000001)

/*
 * LW2080_CTRL_CMD_FB_GET_CLIENT_ALLOCATION_INFO
 *
 * This control command is used by clients to query information pertaining to client allocations.
 *
 *
 * @params [IN/OUT] LwU64 allocCount:
 *        Client specifies the allocation count that it received using the
 *        previous LW2080_CTRL_CMD_FB_GET_CLIENT_ALLOCATION_INFO control call.
 *        RM will get the total number of allocations known by RM and fill
 *        allocCount with it.
 *
 * @params [IN] LwP64 pAllocInfo:
 *        Pointer to the buffer allocated by client of size LW2080_CTRL_CMD_FB_ALLOCATION_INFO *
 *        allocCount. RM returns the info pertaining to each  of the contiguous client
 *        allocation chunks in pAllocInfo. The format of the allocation information is given by
 *        LW2080_CTRL_CMD_FB_ALLOCATION_INFO. The client has to sort the returned information if
 *        it wants to retain the legacy behavior of SORTED BY OFFSET. Information is only returned
 *        if and only if allocCount[IN]>=allocCount[OUT] and clientCount[IN]>=clientCount[OUT].
 *
 * @params [IN/OUT] LwP64 clientCount:
 *        Client specifies the client count that it received using the
 *        previous LW2080_CTRL_CMD_FB_GET_CLIENT_ALLOCATION_INFO control call.
 *        RM will get the total number of clients that have allocations with RM
 *        and fill clientCount with it.
 *
 * @params [IN] LwP64 pClientInfo:
 *        Pointer to the buffer allocated by client of size LW2080_CTRL_CMD_FB_CLIENT_INFO *
 *        clientCount. RM returns the info pertaining to each of the clients that have allocations
 *        known about by RM in pClientInfo. The format of the allocation information is given by
 *        LW2080_CTRL_CMD_FB_CLIENT_INFO. Information is only returned if and only if
 *        allocCount[IN]>=allocCount[OUT] and clientCount[IN]>=clientCount[OUT].
 *
 * @returns Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_POINTER
 *   LW_ERR_NO_MEMORY
 *
 * @Usage: All privileged RM clients for debugging only. Initially, call this with allocCount =
 *         clientCount = 0 to get client count, and then call again with allocated memory and sizes.
 *         Client can repeat with the new count-sized allocations until a maximum try count is
 *         reached or client is out of memory.
 */

#define LW2080_CTRL_CMD_FB_GET_CLIENT_ALLOCATION_INFO           (0x20801349) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FB_GET_CLIENT_ALLOCATION_INFO_PARAMS_MESSAGE_ID" */

/*
 * These work with the FLD_SET_REF_NUM and FLD_TEST_REF macros and describe the 'flags' member
 * of the LW2080_CTRL_CMD_FB_ALLOCATION_INFO struct.
 */

// Address space of the allocation
#define LW2080_CTRL_CMD_FB_ALLOCATION_FLAGS_TYPE 4:0
#define LW2080_CTRL_CMD_FB_ALLOCATION_FLAGS_TYPE_SYSMEM         0
#define LW2080_CTRL_CMD_FB_ALLOCATION_FLAGS_TYPE_VIDMEM         1

// Whether the allocation is shared
#define LW2080_CTRL_CMD_FB_ALLOCATION_FLAGS_SHARED 5:5
#define LW2080_CTRL_CMD_FB_ALLOCATION_FLAGS_SHARED_FALSE        0
#define LW2080_CTRL_CMD_FB_ALLOCATION_FLAGS_SHARED_TRUE         1

// Whether this client owns this allocation
#define LW2080_CTRL_CMD_FB_ALLOCATION_FLAGS_OWNER 6:6
#define LW2080_CTRL_CMD_FB_ALLOCATION_FLAGS_OWNER_FALSE         0
#define LW2080_CTRL_CMD_FB_ALLOCATION_FLAGS_OWNER_TRUE          1

typedef struct LW2080_CTRL_CMD_FB_ALLOCATION_INFO {
    LwU32 client;                        /* [OUT] Identifies the client that made or shares the allocation (index into pClientInfo)*/
    LwU32 flags;                         /* [OUT] Flags associated with the allocation (see previous defines) */
    LW_DECLARE_ALIGNED(LwU64 beginAddr, 8);   /* [OUT] Starting physical address of the chunk */
    LW_DECLARE_ALIGNED(LwU64 size, 8);   /* [OUT] Size of the allocated contiguous chunk in bytes */
} LW2080_CTRL_CMD_FB_ALLOCATION_INFO;

typedef struct LW2080_CTRL_CMD_FB_CLIENT_INFO {
    LwHandle handle;                                    /* [OUT] Handle of the client that made or shares the allocation */
    LwU32    pid;                                       /* [OUT] PID of the client that made or shares the allocation */

    /* For the definition of the subprocessID and subprocessName params, see LW0000_CTRL_CMD_SET_SUB_PROCESS_ID */
    LwU32    subProcessID;                              /* [OUT] Subprocess ID of the client that made or shares the allocation */
    char     subProcessName[LW_PROC_NAME_MAX_LENGTH];   /* [OUT] Subprocess Name of the client that made or shares the allocation */
} LW2080_CTRL_CMD_FB_CLIENT_INFO;

#define LW2080_CTRL_CMD_FB_GET_CLIENT_ALLOCATION_INFO_PARAMS_MESSAGE_ID (0x49U)

typedef struct LW2080_CTRL_CMD_FB_GET_CLIENT_ALLOCATION_INFO_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 allocCount, 8);
    LW_DECLARE_ALIGNED(LwP64 pAllocInfo, 8);
    LW_DECLARE_ALIGNED(LwU64 clientCount, 8);
    LW_DECLARE_ALIGNED(LwP64 pClientInfo, 8);
} LW2080_CTRL_CMD_FB_GET_CLIENT_ALLOCATION_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_FB_UPDATE_NUMA_STATUS
 *
 * This control command is used by clients to update the NUMA status.
 *
 * @params [IN] LwBool bOnline:
 *
 * @returns Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *
 */
#define LW2080_CTRL_CMD_FB_UPDATE_NUMA_STATUS (0x20801350) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_UPDATE_NUMA_STATUS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_UPDATE_NUMA_STATUS_PARAMS_MESSAGE_ID (0x50U)

typedef struct LW2080_CTRL_FB_UPDATE_NUMA_STATUS_PARAMS {
    LwBool bOnline;
} LW2080_CTRL_FB_UPDATE_NUMA_STATUS_PARAMS;

/*
 * LW2080_CTRL_CMD_FB_GET_NUMA_INFO
 *
 * This control command is used by clients to get per-subdevice NUMA memory
 * information as assigned by the system.
 *
 * numaNodeId[OUT]
 *     - Specifies the NUMA node ID.
 *
 * numaMemAddr[OUT]
 *     - Specifies the NUMA memory address.
 *
 * numaMemSize[OUT]
 *     - Specifies the NUMA memory size.
 *
 * numaOfflineAddressesCount[IN/OUT]
 *     - If non-zero, then it specifies the maximum number of entries in
 *       numaOfflineAddresses[] for which the information is required.
 *       It will be updated with the actual number of entries present in
 *       the numaOfflineAddresses[].
 *
 * numaOfflineAddresses[OUT]
 *      - If numaOfflineAddressesCount is non-zero, it contains the addresses
 *        of offline pages in the NUMA region.
 *
 * @returns Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FB_GET_NUMA_INFO               (0x20801351) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FB_INTERFACE_ID << 8) | LW2080_CTRL_FB_GET_NUMA_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FB_NUMA_INFO_MAX_OFFLINE_ADDRESSES 64

#define LW2080_CTRL_FB_GET_NUMA_INFO_PARAMS_MESSAGE_ID (0x51U)

typedef struct LW2080_CTRL_FB_GET_NUMA_INFO_PARAMS {
    LwS32 numaNodeId;
    LW_DECLARE_ALIGNED(LwU64 numaMemAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 numaMemSize, 8);
    LwU32 numaOfflineAddressesCount;
    LW_DECLARE_ALIGNED(LwU64 numaOfflineAddresses[LW2080_CTRL_FB_NUMA_INFO_MAX_OFFLINE_ADDRESSES], 8);
} LW2080_CTRL_FB_GET_NUMA_INFO_PARAMS;

/* _ctrl2080fb_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

