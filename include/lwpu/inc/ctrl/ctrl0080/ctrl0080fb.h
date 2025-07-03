/*
 * SPDX-FileCopyrightText: Copyright (c) 2004-2017 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0080/ctrl0080fb.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl0080/ctrl0080base.h"

/* LW01_DEVICE_XX/LW03_DEVICE fb control commands and parameters */

/**
 * LW0080_CTRL_CMD_FB_GET_CAPS
 *
 * This command returns the set of framebuffer capabilities for the device
 * in the form of an array of unsigned bytes.  Framebuffer capabilities
 * include supported features and required workarounds for the framebuffer
 * engine(s) within the device, each represented by a byte offset into the
 * table and a bit position within that byte.
 *
 *   capsTblSize
 *     This parameter specifies the size in bytes of the caps table.
 *     This value should be set to LW0080_CTRL_FB_CAPS_TBL_SIZE.
 *   capsTbl
 *     This parameter specifies a pointer to the client's caps table buffer
 *     into which the framebuffer caps bits will be transferred by the RM.
 *     The caps table is an array of unsigned bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_POINTER
 */
#define LW0080_CTRL_CMD_FB_GET_CAPS (0x801301) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FB_INTERFACE_ID << 8) | LW0080_CTRL_FB_GET_CAPS_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FB_GET_CAPS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0080_CTRL_FB_GET_CAPS_PARAMS {
    LwU32 capsTblSize;
    LW_DECLARE_ALIGNED(LwP64 capsTbl, 8);
} LW0080_CTRL_FB_GET_CAPS_PARAMS;

/* extract cap bit setting from tbl */
#define LW0080_CTRL_FB_GET_CAP(tbl,c)              (((LwU8)tbl[(1?c)]) & (0?c))

/* caps format is byte_index:bit_mask */
#define LW0080_CTRL_FB_CAPS_SUPPORT_RENDER_TO_SYSMEM                                0:0x01
#define LW0080_CTRL_FB_CAPS_BLOCKLINEAR                                             0:0x02
#define LW0080_CTRL_FB_CAPS_SUPPORT_SCANOUT_FROM_SYSMEM                             0:0x04
#define LW0080_CTRL_FB_CAPS_SUPPORT_CACHED_SYSMEM                                   0:0x08
#define LW0080_CTRL_FB_CAPS_SUPPORT_C24_COMPRESSION                                 0:0x10 // Deprecated
#define LW0080_CTRL_FB_CAPS_SUPPORT_SYSMEM_COMPRESSION                              0:0x20
#define LW0080_CTRL_FB_CAPS_NISO_CFG0_BUG_534680                                    0:0x40 // Deprecated
#define LW0080_CTRL_FB_CAPS_ISO_FETCH_ALIGN_BUG_561630                              0:0x80 // Deprecated

#define LW0080_CTRL_FB_CAPS_BLOCKLINEAR_GOBS_512                                    1:0x01
#define LW0080_CTRL_FB_CAPS_L2_TAG_BUG_632241                                       1:0x02
#define LW0080_CTRL_FB_CAPS_SINGLE_FB_UNIT                                          1:0x04 // Deprecated
#define LW0080_CTRL_FB_CAPS_CE_RMW_DISABLE_BUG_897745                               1:0x08 // Deprecated
#define LW0080_CTRL_FB_CAPS_OS_OWNS_HEAP_NEED_ECC_SCRUB                             1:0x10
#define LW0080_CTRL_FB_CAPS_ASYNC_CE_L2_BYPASS_SET                                  1:0x20 // Deprecated
#define LW0080_CTRL_FB_CAPS_DISABLE_TILED_CACHING_ILWALIDATES_WITH_ECC_BUG_1521641  1:0x40

#define LW0080_CTRL_FB_CAPS_DISABLE_MSCG_WITH_VR_BUG_1681803                        2:0x01
#define LW0080_CTRL_FB_CAPS_VIDMEM_ALLOCS_ARE_CLEARED                               2:0x02
#define LW0080_CTRL_FB_CAPS_DISABLE_PLC_GLOBALLY                                    2:0x04
#define LW0080_CTRL_FB_CAPS_PLC_BUG_3046774                                         2:0x08


/* size in bytes of fb caps table */
#define LW0080_CTRL_FB_CAPS_TBL_SIZE         3


/*
 * LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT
 *
 * This command will create cpu visible segment of fb memory given 
 * a physical address, length, and optionally a virtual address
 *
 * [in] hCtxDma
 *    This field and (PTE_FORMAT == CTXDMA) are no longer supported.
 * [in] subDeviceIDMask
 *    Mask of subdevice Ids to indicate which indexes are usable in 
 *    pOffset[], pVidOffset[], pGpuAddress[] and ppCpuAddress[]
 * [in] dmaOffset
 *    This is the offset into the ctxdma in which the set of PTE's
 *    to map is stored.
 *    It is only used when (CONTIGUOUS == FALSE) && (PTE_FORMAT == CTXDMA).
 * [in] pPageArray
 *    Opaque pointer to an OS-specific page array, to be used for
 *    non-contiguous physical mappings without a CTXDMA.
 *    It is only used when (CONTIGUOUS == FALSE) && (PTE_FORMAT == OS_ARRAY).
 * [in] startPageIndex
 *    Starting page index into the pPageArray (index of the 0th page).
 *    It is only used when (CONTIGUOUS == FALSE) && (PTE_FORMAT == OS_ARRAY).
 * [in] VidOffset
 *    This parameter is an input specifying the physical video memory offset
 *    for the segment.
 *    It is only used when (CONTIGUOUS == TRUE).
 * [in] Offset - To be temporarily deprecated
 *    This parameter represents the offset into the BAR1 address space.  This
 *    is also the gpu virtual address.   This parameter is only valid if the
 *    FIXED_OFFSET_YES flag is passed.
 * [in] pOffset
 *    This parameter represents the offset into the BAR1 address spaces of each
 *    GPU  specified by subDeviceIDMask. These are also the gpu virtual 
 *    addresses. This parameter is only valid if the FIXED_OFFSET_YES flag is 
 *    passed.
 * [in] Length
 *    This parameter is the length of the allocation in bytes
 * [in] ValidLength
 *    This is the actual length of the block-linear surface.  The difference
 *    between Length and ValidLength is the padding used for the BL to
 *    pitch mapping.   This pad memory will be mapped back to VidOffset
 *    so that application errors will not cause random data corruption
 *    in sensitive system structures
 * [in] AllocHintHandle
 *     This parameter is the handle passed on a previous call to HeapAllocHint
 *     used to specify compression properties of the allocation.   If this 
 *     parameter is 0, then pitch memory with pitch equal to Length is
 *     mapped.
 * [in] Flags
 *     Flags which control the properties of the allocation.
 * [in] hMemory
 *     Memory handle which will be used to create a memory object for the allocation
 * [in] hClient
 *     Client handle for the allocation
 * [in] hDevice
 *     Caller's device handle
 * [out] pCpuAddress - To be temporarily deprecated
 *     The returned CPU virtual address of the allocation
 * [out] ppCpuAddress
 *     The returned CPU virtual address of the allocation for each GPU specified 
 *     by subDeviceIDMask.
 * [out] GpuAddress - To be temporarily deprecated
 *     The returned GPU virtual address. If the FIXED_OFFSET_YES flag is
 *     provided, then this will be the same value of Offset.
 * [out] pGpuAddress
 *     The returned GPU virtual addresses for each GPU specified by subDeviceIDMask.
 *     If the FIXED_OFFSET_YES flag is provided, then this will be the same value 
 *     of pOffset.
 * [in] hAllocHintClient
 *     This parameter is the client handle to be used with AllocHintHandle.
 *     If the value of this parameter is LW01_NULL_OBJECT then the default
 *     client handle (client handle of the last HeapHwAlloc) will be used. 
 *
 * Can be called without API & GPU locks if LWOS54_FLAGS_IRQL_RAISED and
 * LWOS54_FLAGS_LOCK_BYPASS are set in LWOS54_PARAMETERS.flags (this can only be done while
 * in a bugcheck). If the flags are not set then this control call is simply a wrapper
 * around an allocation of LW_FB_SEGMENT.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_GENERIC
 *   LWOS_STATUS_ERROR_UNSUPPORTED
 */
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT (0x801303) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FB_INTERFACE_ID << 8) | LW0080_CTRL_FB_CREATE_FB_SEGMENT_PARAMS_MESSAGE_ID" */

#include "class/cl00c1.h"

//
// Legacy control control path to create FB segment. Client code should be
// migrated to LwRmAlloc(LW_FB_SEGMENT)
//
#define LW0080_CTRL_FB_CREATE_FB_SEGMENT_PARAMS_MESSAGE_ID (0x3U)

typedef LW_FB_SEGMENT_ALLOCATION_PARAMS LW0080_CTRL_FB_CREATE_FB_SEGMENT_PARAMS;

#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_FIXED_OFFSET          0:0
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_FIXED_OFFSET_NO     (0x00000000)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_FIXED_OFFSET_YES    (0x00000001)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_MAP_CPUVA             1:1
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_MAP_CPUVA_NO        (0x00000000)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_MAP_CPUVA_YES       (0x00000001)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_APERTURE              3:2
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_APERTURE_VIDMEM     (0x00000000)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_APERTURE_COH_SYS    (0x00000001)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_APERTURE_NCOH_SYS   (0x00000002)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_CONTIGUOUS            4:4
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_CONTIGUOUS_FALSE    (0x00000000)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_CONTIGUOUS_TRUE     (0x00000001)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_GPU_CACHED            5:5
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_GPU_CACHED_FALSE    (0x00000000)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_GPU_CACHED_TRUE     (0x00000001)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_PAGE_SIZE             6:6
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_PAGE_SIZE_4K        (0x00000000)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_PAGE_SIZE_BIG       (0x00000001)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_IN_BUGCHECK           7:7
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_IN_BUGCHECK_NO      (0x00000000)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_IN_BUGCHECK_YES     (0x00000001)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_PTE_FORMAT            8:8
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_PTE_FORMAT_OS_ARRAY (0x00000001)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_KIND_PROVIDED         9:9
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_KIND_PROVIDED_N0    (0x00000000)
#define LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT_KIND_PROVIDED_YES   (0x00000001)

/*
 * LW0080_CTRL_CMD_FB_DESTROY_FB_SEGMENT
 *
 * This command will destroy the fb segment created with the hMemory parameter.
 *
 * hMemory
 *    Memory handle passed in to LW0080_CTRL_CMD_FB_CREATE_FB_SEGMENT.
 *
 * Possible status values returned are
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_GENERIC
 *    LWOS_STATUS_ERROR_UNSUPPORTED
 */
#define LW0080_CTRL_CMD_FB_DESTROY_FB_SEGMENT                    (0x801304) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FB_INTERFACE_ID << 8) | LW0080_CTRL_FB_DESTROY_FB_SEGMENT_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FB_DESTROY_FB_SEGMENT_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0080_CTRL_FB_DESTROY_FB_SEGMENT_PARAMS {
    LwHandle hMemory;
} LW0080_CTRL_FB_DESTROY_FB_SEGMENT_PARAMS;

/*!
 * LW0080_CTRL_CMD_FB_COMPBIT_STORE_GET_INFO
 *
 * This command returns compbit backing store-related information.
 *
 *   size
 *     [out] Size of compbit store, in bytes
 *   address
 *     [out] Address of compbit store
 *   addressSpace
 *     [out] Address space of compbit store (corresponds to type LW_ADDRESS_SPACE in lwrm.h)
 *   maxCompbitLine
 *     [out] Maximum compbitline possible, determined based on size
 *   comptagsPerCacheLine
 *     [out] Number of compression tags per compression cache line, across all
 *           L2 slices.
 *   cacheLineSize
 *     [out] Size of compression cache line, across all L2 slices. (bytes)
 *   cacheLineSizePerSlice
 *     [out] Size of the compression cache line per slice (bytes)
 *   cacheLineFetchAlignment
 *     [out] Alignment used while fetching the compression cacheline range in FB.
 *           If start offset of compcacheline in FB is S and end offset is E, then
 *           the range to fetch to ensure entire compcacheline data is extracted is:
 *           (align_down(S) , align_up(E))
 *           This is needed in GM20X+ because of interleaving of data in Linear FB space.
 *           Example - In GM204 every other 1K FB chunk of data is offset by 16K.
 *   backingStoreBase
 *     [out] Address of start of Backing Store in linear FB Physical Addr space.
 *           This is the actual offset in FB which HW starts using as the Backing Store and
 *           in general will be different from the start of the region that driver allocates
 *           as the backing store. This address is expected to be 2K aligned.
 *   gobsPerComptagPerSlice
 *     [out] (Only on Pascal) Number of GOBS(512 bytes of surface PA) that correspond to one 64KB comptgaline, per slice.
 *           One GOB stores 1 byte of compression bits.
 *           0 value means this field is not applicable for the current architecture.
 *   backingStoreCbcBase
 *     [out] 2KB aligned base address of CBC (post divide address)
 *   comptaglineAllocationPolicy
 *     [out] Policy used to allocate comptagline from CBC for the device
 *   privRegionStartOffset
 *     [out] Starting offset for any priv region allocated by clients. only used by MODS
 *   Possible status values returned are:
 *   LW_OK
 */
#define LW0080_CTRL_CMD_FB_GET_COMPBIT_STORE_INFO (0x801306) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FB_INTERFACE_ID << 8) | LW0080_CTRL_FB_GET_COMPBIT_STORE_INFO_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FB_GET_COMPBIT_STORE_INFO_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW0080_CTRL_FB_GET_COMPBIT_STORE_INFO_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 Size, 8);
    LW_DECLARE_ALIGNED(LwU64 Address, 8);
    LwU32 AddressSpace;
    LwU32 MaxCompbitLine;
    LwU32 comptagsPerCacheLine;
    LwU32 cacheLineSize;
    LwU32 cacheLineSizePerSlice;
    LwU32 cacheLineFetchAlignment;
    LW_DECLARE_ALIGNED(LwU64 backingStoreBase, 8);
    LwU32 gobsPerComptagPerSlice;
    LwU32 backingStoreCbcBase;
    LwU32 comptaglineAllocationPolicy;
    LW_DECLARE_ALIGNED(LwU64 privRegionStartOffset, 8);
} LW0080_CTRL_FB_GET_COMPBIT_STORE_INFO_PARAMS;

#define LW0080_CTRL_CMD_FB_GET_COMPBIT_STORE_INFO_ADDRESS_SPACE_UNKNOWN 0 // ADDR_UNKNOWN
#define LW0080_CTRL_CMD_FB_GET_COMPBIT_STORE_INFO_ADDRESS_SPACE_SYSMEM  1 // ADDR_SYSMEM
#define LW0080_CTRL_CMD_FB_GET_COMPBIT_STORE_INFO_ADDRESS_SPACE_FBMEM   2 // ADDR_FBMEM

// Policy used to allocate comptaglines
/**
 * Legacy mode allocates a comptagline for 64kb page. This mode will always allocate
 * contiguous comptaglines from a ctag heap.
 */
#define LW0080_CTRL_CMD_FB_GET_COMPBIT_STORE_INFO_POLICY_LEGACY         0
/**
 * 1TO1 mode allocates a comptagline for 64kb page. This mode will callwlate
 * comptagline offset based on physical address. This mode will allocate
 * contiguous comptaglines if the surface is contiguous and non-contiguous
 * comptaglines for non-contiguous surfaces.
 */
#define LW0080_CTRL_CMD_FB_GET_COMPBIT_STORE_INFO_POLICY_1TO1           1
/**
 * 1TO4_Heap mode allocates a comptagline for 256kb page granularity. This mode
 * will allocate comptagline from a heap. This mode will align the surface allocations
 * to 256kb before allocating comptaglines. The comptaglines allocated will always be
 * contiguous here.
 * TODO: For GA10x, this mode will support < 256kb surface allocations, by sharing
 * a comptagline with at most 3 different 64Kb allocations. This will result in
 * miixed-contiguity config where comptaglines will be allocated contiguously as well
 * as non-contiguous when shared with other allocations.
 */
#define LW0080_CTRL_CMD_FB_GET_COMPBIT_STORE_INFO_POLICY_1TO4           2
/**
 * Rawmode will transfer allocation of comptaglines to HW, where HW manages
 * comptaglines based on physical offset. The comptaglines are cleared when SW
 * issues physical/virtual scrub to the surface before reuse.
 */
#define LW0080_CTRL_CMD_FB_GET_COMPBIT_STORE_INFO_POLICY_RAWMODE        3

/**
 * LW0080_CTRL_CMD_FB_GET_CAPS_V2
 *
 * This command returns the same set of framebuffer capabilities for the
 * device as @ref LW0080_CTRL_CMD_FB_GET_CAPS. The difference is in the structure
 * LW0080_CTRL_FB_GET_CAPS_V2_PARAMS, which contains a statically sized array,
 * rather than a caps table pointer and a caps table size in
 * LW0080_CTRL_FB_GET_CAPS_PARAMS.
 *
 *   capsTbl
 *     This parameter specifies a pointer to the client's caps table buffer
 *     into which the framebuffer caps bits will be written by the RM.
 *     The caps table is an array of unsigned bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_POINTER
 */
#define LW0080_CTRL_CMD_FB_GET_CAPS_V2                                  (0x801307) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FB_INTERFACE_ID << 8) | LW0080_CTRL_FB_GET_CAPS_V2_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FB_GET_CAPS_V2_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW0080_CTRL_FB_GET_CAPS_V2_PARAMS {
    LwU8 capsTbl[LW0080_CTRL_FB_CAPS_TBL_SIZE];
} LW0080_CTRL_FB_GET_CAPS_V2_PARAMS;




/* _ctrl0080fb_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

