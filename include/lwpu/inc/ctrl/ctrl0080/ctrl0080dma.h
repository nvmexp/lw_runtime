/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2018 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0080/ctrl0080dma.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl0080/ctrl0080base.h"

/* LW01_DEVICE_XX/LW03_DEVICE dma control commands and parameters */

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW0080_CTRL_DMA_PTE_INFO_PTE_BLOCK
 *
 *    This parameter returns the parameters specific to a PTE as follows:
 *       pageSize
 *           GET: This parameter returns the page size of the PTE information
 *                being returned.  If 0, then this pteBlock[] array entry is
 *                invalid or not used.  (pteBlock[0] is always used.)
 *           SET: This parameter specifies the page size of the PTE information
 *                to be set.  If 0, then this pteBlock[] array entry is invalid
 *                or not used.  (pteBlock[0] is always used.)
 *       pteEntrySize
 *           GET: This parameter returns the size of the PTE in bytes for this GPU.
 *           SET: N/A
 *       comptagLine
 *           GET: This parameter returns the comptagline field of the corresponding PTE.
 *           SET: This parameter sets the comptagline field of the corresponding PTE.
 *                Incorrect values may lead to dire consequences.
 *       kind
 *           GET: This parameter returns the kind field of the corresponding PTE.
 *           SET: This parameter sets the kind field of the corresponding PTE.
 *                Incorrect values may lead to undesirable consequences.
 *       pteFlags
 *           This parameter returns various fields from the PTE, these are:
 *           FLAGS_VALID:
 *               GET: This flag returns the valid bit of the PTE.
 *               SET: This flag sets the valid bit of the PTE.
 *           FLAGS_ENCRYPTED:
 *               GET: This flag returns the encrypted bit of the PTE. Not all GPUs
 *                  support encryption. If not supported, this flag will be set to
 *                  NOT_SUPPORTED.
 *               SET: This flag sets the encrypted bit of the PTE.
 *           FLAGS_APERTURE:
 *               GET: This flag returns the aperture field of the PTE. See
 *                    LW0080_CTRL_DMA_GET_PTE_INFO_PARAMS_FLAGS_APERTURE_* for values.
 *               SET: This flag sets the aperture field of the PTE.  See
 *                    LW0080_CTRL_DMA_GET_PTE_INFO_PARAMS_FLAGS_APERTURE_* for values.
 *           FLAGS_COMPTAGS:
 *               GET: This flag returns the comptags field of the PTE.  (Not used on Fermi)
 *               SET: N/A
 *           FLAGS_GPU_CACHED:
 *               GET: This flag returns the GPU cacheable bit of the PTE. GPU caching of
 *                    sysmem was added in iGT21a and Fermi. If not supported, this flag
 *                    will be set to NOT_SUPPORTED.
 *               SET: N/A for specific chips, e.g., GF100
 *           FLAGS_SHADER_ACCESS:
 *               GET: This flag returns the shader access control of the PTE. This feature
 *                    was introduced in Kepler.  If not supported, this flag will be set to
 *                    NOT_SUPPORTED.
 *               SET: N/A
 */

typedef struct LW0080_CTRL_DMA_PTE_INFO_PTE_BLOCK {
    LwU32 pageSize;
    LW_DECLARE_ALIGNED(LwU64 pteEntrySize, 8);
    LwU32 comptagLine;
    LwU32 kind;
    LwU32 pteFlags;
} LW0080_CTRL_DMA_PTE_INFO_PTE_BLOCK;

#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_VALID                                     0:0
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_VALID_FALSE                         (0x00000000U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_VALID_TRUE                          (0x00000001U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_ENCRYPTED                                 2:1
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_ENCRYPTED_FALSE                     (0x00000000U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_ENCRYPTED_TRUE                      (0x00000001U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_ENCRYPTED_NOT_SUPPORTED             (0x00000002U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_APERTURE                                  6:3
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_APERTURE_VIDEO_MEMORY               (0x00000000U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_APERTURE_PEER_MEMORY                (0x00000001U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_APERTURE_SYSTEM_COHERENT_MEMORY     (0x00000002U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_APERTURE_SYSTEM_NON_COHERENT_MEMORY (0x00000003U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_COMPTAGS                                  10:7
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_COMPTAGS_NONE                       (0x00000000U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_COMPTAGS_1                          (0x00000001U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_COMPTAGS_2                          (0x00000002U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_COMPTAGS_4                          (0x00000004U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_GPU_CACHED                                12:11
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_GPU_CACHED_FALSE                    (0x00000000U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_GPU_CACHED_TRUE                     (0x00000001U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_GPU_CACHED_NOT_SUPPORTED            (0x00000002U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_SHADER_ACCESS                             14:13
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_SHADER_ACCESS_READ_WRITE            (0x00000000U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_SHADER_ACCESS_READ_ONLY             (0x00000001U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_SHADER_ACCESS_WRITE_ONLY            (0x00000002U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_SHADER_ACCESS_NOT_SUPPORTED         (0x00000003U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_READ_ONLY                                 15:15
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_READ_ONLY_FALSE                     (0x00000000U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_READ_ONLY_TRUE                      (0x00000001U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_ATOMIC                                    16:16
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_ATOMIC_DISABLE                      (0x00000000U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_ATOMIC_ENABLE                       (0x00000001U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_ACCESS_COUNTING                           17:17
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_ACCESS_COUNTING_DISABLE             (0x00000000U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_ACCESS_COUNTING_ENABLE              (0x00000001U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_PRIVILEGED                                18:18
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_PRIVILEGED_FALSE                    (0x00000000U)
#define LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_PRIVILEGED_TRUE                     (0x00000001U)

/*
 * LW0080_CTRL_DMA_GET_PTE_INFO
 *
 * This command queries PTE information for the specified GPU virtual address.
 *
 *   gpuAddr
 *      This parameter specifies the GPU virtual address for which PTE
 *      information is to be returned.
 *   skipVASpaceInit
 *      This parameter specifies(true/false) whether the VA Space
 *      initialization should be skipped in this ctrl call.
 *   pteBlocks
 *      This parameter returns the page size-specific attributes of a PTE.
 *      Please see LW0080_CTRL_DMA_PTE_INFO_PTE_BLOCK.
 *   hVASpace
 *      handle for the allocated VA space that this control call should operate
 *      on. If it's 0, it assumes to use the implicit allocated VA space.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_GENERIC
 */

#define LW0080_CTRL_CMD_DMA_GET_PTE_INFO                                          (0x801801U) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_GET_PTE_INFO_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_DMA_GET_PTE_INFO_PTE_BLOCKS                                   4U

#define LW0080_CTRL_DMA_GET_PTE_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0080_CTRL_DMA_GET_PTE_INFO_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 gpuAddr, 8);
    LwU32    subDeviceId;
    LwU8     skipVASpaceInit;
    LW_DECLARE_ALIGNED(LW0080_CTRL_DMA_PTE_INFO_PTE_BLOCK pteBlocks[LW0080_CTRL_DMA_GET_PTE_INFO_PTE_BLOCKS], 8);
    LwHandle hVASpace;
} LW0080_CTRL_DMA_GET_PTE_INFO_PARAMS;

/*
 * LW0080_CTRL_DMA_SET_PTE_INFO
 *
 * This command sets PTE information for the specified GPU virtual address.
 * Usage of parameter and field definitions is identical to that of
 * LW0080_CTRL_DMA_GET_PTE_INFO, with the following exception:
 *
 * - pteFlags field LW0080_CTRL_DMA_PTE_INFO_PARAMS_FLAGS_COMPTAGS is ignored,
 *   as this setting is specified via the kind specification.
 * - pteEntrySize is ignored, as this setting is read-only in the GET case.
 * - hVASpace
 *    handle for the allocated VA space that this control call should operate
 *    on. If it's 0, it assumes to use the implicit allocated VA space.
 *
 */

#define LW0080_CTRL_CMD_DMA_SET_PTE_INFO        (0x80180aU) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_SET_PTE_INFO_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_DMA_SET_PTE_INFO_PTE_BLOCKS 4U

#define LW0080_CTRL_DMA_SET_PTE_INFO_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW0080_CTRL_DMA_SET_PTE_INFO_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 gpuAddr, 8);
    LwU32    subDeviceId;
    LW_DECLARE_ALIGNED(LW0080_CTRL_DMA_PTE_INFO_PTE_BLOCK pteBlocks[LW0080_CTRL_DMA_SET_PTE_INFO_PTE_BLOCKS], 8);
    LwHandle hVASpace;
} LW0080_CTRL_DMA_SET_PTE_INFO_PARAMS;


/*
 * LW0080_CTRL_DMA_FILL_PTE_MEM
 *
 * This command fills a given region of memory with an array of PTE entries
 * entries suitable for page table update by blitting or other means.
 * This command is only available on Windows and MODS platforms.
 * This command can be called by kernel clients only.
 *
 *   pageCount
 *      This parameter specifies the number of PTE entries to be filled.
 *   hwResource{hClient, hDevice, hMemory}
 *      This parameter specifies the handle of the physical memory allocation
 *      from which to extract the kind and partition stride attributes.
 *   offset
 *      This parameter specifies the offset into the physical memory
 *      region given by the handle above.
 *   gpuAddr
 *      This parameter specifies the GPU virtual address of an existing
 *      mapping from which other attributes are extracted.
 *   pageArray
 *       This parameter specifies the new page frames in a format specified
 *       by the FORMAT flag.
 *       The number of entries must be at least pageCount unless the the
 *       DUPLICATE_FIRST_PTE flag is set, in which case only 1 entry is needed.
 *   pteMem
 *       This parameter specifies a pointer to memory which is filled with
 *       the GPU specific PTE data.
 *   pteMemPfn
 *       This parameter specifies the page frame number of pteMem and it is only
 *       needed in VGX mode when hypervisor wants to update push buffer directly.
 *   pageSize
 *     This parameter specifies the page size pointed to by each PTE entry
 *     if and only if OVERRIDE_PAGE_SIZE is set to _TRUE.
 *   startPageIndex
 *       Indicates the index into the pageArray to start the operation on.
 *       Normally this should be 0, but certain OS_ARRAY formats require an offset.
 *   flags
 *     This parameter specifies flags values to use for the fill operation:
 *       FLAGS_CONTIGUOUS
 *         If true, specifies that the page frame numbers are contiguous.
 *         If so, pageArray[0] is used as the initial page address and all
 *         successive addresses are computed by adding the page size.
 *       FLAGS_VALID
 *         Sets the _VALID bit in every PTE
 *       FLAGS_READ_ONLY
 *         Sets the _READ_ONLY bit in every PTE (writes cause page faults)
 *       FLAGS_PRIV
 *         Sets the _PRIV bit in every PTE (protected content)
 *       FLAGS_TLBlOCK
 *         Sets the _TLB_LOCK in every PTE (if supported, g84+)
 *       FLAGS_ENCRYPTED
 *         Sets the _ENCRYPTED in every PTE (if supported)
 *       FLAGS_DEFER_ILWALIDATE
 *         If nonzero, skips TLB ilwalidate, even when pteMem is NULL,
 *         which indicates the CPU is updating the page tables directly.
 *       FLAGS_SPARSE
 *         Only for MODS verification. Returns error on other platforms (not yet).
 *         If true, sets the sparse bit in invalid PTEs.
 *       FLAGS_APERTURE
 *         If nonzero, specifies to override the aperture setting. See
 *         LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_APERTURE_* for possible enums.
 *       FLAGS_PEER
 *         Uses to specify the peer if flags_aperture = _PEER
 *       FLAGS_PAGE_SIZE
 *         If nonzero overrides the page size with the value in pageSize.
 *       FLAGS_GPU_CACHED
 *         Sets the _CACHEABLE bit in every PTE (if supported).
 *       FLAGS_PTE_COALESCE_LEVEL_CAP
 *         If nonzero overrides the default PTE coalescing.
 *       FLAGS_SHADER_ACCESS
 *         Sets shader access fields if supported. A setting of _DEFAULT will mirror the
 *         READ_ONLY field.
 *       FLAGS_FORMAT
 *         Specifies the format of the provided pageArray.
 *         The default PTE_ARRAY format indicates an array of 64-bit unsigned integers.
 *         Each integer corresponds to one page, specifying the page's physical address
 *         right-shifted to a PTE address field value.
 *         The OS_ARRAY format indicates an OS-specific array format is used
 *         to avoid intermediate copies to the 64-bit array format.
 *       FLAGS_DUPLICATE_FIRST_PTE
 *         If true, the first page of the provided pageArray is used (duplicated) for all
 *         pages up to pageCount.
 *         This may be used to initialize all the pages in a range to a dummy or invalid
 *         page.
 *       FLAGS_BUS_ADDRESS
 *         In VGX, used to specify whether the actual physical bus address is available
 *         with KMD.
 *         If TRUE, then the vGPU plugin does not need to do further translation for it.
 *       FLAGS_USE_PEER_ID
 *         In SLI-Next, KMD will manage the lwlink activation and enablement. RM has no clue
 *         about peer GPUs. Hence relying on KMD's input to program them in the PTE. This field
 *         is used to activate the parameter peerId 
 *
 *   hSrcVASpace
 *     This parameter specifies what memory is being paged into the pagetables
 *     corresponding to hTgtVASpace. A zero value indicate physical memory,
 *     a non-zero value indicate SMMU virtual memory handle.
 *     https://wiki.lwpu.com/engwiki/index.php/KMD/T124#RMApi_Change
 *
 *   hTgtVASpace
 *     This parameter specifies the handle to the VASpace object whose 
 *     pagetables need to be updated. See the below wiki for more details.
 *     https://wiki.lwpu.com/engwiki/index.php/KMD/T124#RMApi_Change
 *
 *     Zero handle refers to the default per (client, device) VASpace object.
 *     This will be used for GMMU PTE updates (dGPU & CheetAh), and SMMU PTE 
 *     updates in Aurora class CheetAh chips (being deprecated, use a valid non-zero
 *     SMMU VASpace object handle for SMMU PTE updates).
 *
 *     On CheetAh chips with Big GPU, SMMU PTE updates need a valid non-zero
 *     SMMU VASpace object handle.
 *     Non-zero GMMU VASpace object handle is also supported (GMMU MultiVA support).
 *
 *   peerId
 *     In SLI-Next, KMD will manage the lwlink activation and enablement. RM has no clue
 *     about peer GPUs. Hence relying on KMD's input to program them in the PTE. This field
 *     is activated if the FLAGS_USE_PEER_ID is _TRUE
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_GENERIC
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 */
#define LW0080_CTRL_CMD_DMA_FILL_PTE_MEM (0x801802U) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS {
    LwU32 pageCount;
    struct {
        LwHandle hClient;
        LwHandle hDevice;
        LwHandle hMemory;
        LwU32    subDeviceId;
    } hwResource;
    struct {
        LwU32 fbKind;
        LwU32 sysKind;
        LwU32 compTagStartOffset;
    } comprInfo;
    LW_DECLARE_ALIGNED(LwU64 offset, 8);
    LW_DECLARE_ALIGNED(LwU64 gpuAddr, 8);
    LW_DECLARE_ALIGNED(LwP64 pageArray, 8);
    LW_DECLARE_ALIGNED(LwP64 pteMem, 8);
    LwU32    pteMemPfn;
    LwU32    pageSize;
    LwU32    startPageIndex;
    LwU32    flags;
    LwHandle hSrcVASpace;
    LwHandle hTgtVASpace;
    LwU32    peerId;
} LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS;

#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_CONTIGUOUS                            0:0
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_CONTIGUOUS_FALSE                    (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_CONTIGUOUS_TRUE                     (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_VALID                                 1:1
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_VALID_FALSE                         (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_VALID_TRUE                          (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_READ_ONLY                             2:2
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_READ_ONLY_FALSE                     (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_READ_ONLY_TRUE                      (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_PRIV                                  3:3
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_PRIV_FALSE                          (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_PRIV_TRUE                           (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_TLB_LOCK                              4:4
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_TLB_LOCK_FALSE                      (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_TLB_LOCK_TRUE                       (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_ENCRYPTED                             5:5
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_ENCRYPTED_FALSE                     (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_ENCRYPTED_TRUE                      (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_DEFER_ILWALIDATE                      6:6
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_DEFER_ILWALIDATE_FALSE              (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_DEFER_ILWALIDATE_TRUE               (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_SPARSE                                7:7
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_SPARSE_FALSE                        (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_SPARSE_TRUE                         (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_APERTURE                              11:8
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_APERTURE_VIDEO_MEMORY               (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_APERTURE_PEER_MEMORY                (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_APERTURE_SYSTEM_COHERENT_MEMORY     (0x00000002U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_APERTURE_SYSTEM_NON_COHERENT_MEMORY (0x00000003U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_PEER                                  15:12
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_OVERRIDE_PAGE_SIZE                    16:16
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_OVERRIDE_PAGE_SIZE_FALSE            (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_OVERRIDE_PAGE_SIZE_TRUE             (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_GPU_CACHED                            17:17
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_GPU_CACHED_FALSE                    (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_GPU_CACHED_TRUE                     (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_PTE_COALESCE_LEVEL_CAP                      21:18
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_PTE_COALESCE_LEVEL_CAP_DEFAULT            (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_PTE_COALESCE_LEVEL_CAP_1                  (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_PTE_COALESCE_LEVEL_CAP_2                  (0x00000002U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_PTE_COALESCE_LEVEL_CAP_4                  (0x00000003U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_PTE_COALESCE_LEVEL_CAP_8                  (0x00000004U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_PTE_COALESCE_LEVEL_CAP_16                 (0x00000005U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_PTE_COALESCE_LEVEL_CAP_32                 (0x00000006U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_PTE_COALESCE_LEVEL_CAP_64                 (0x00000007U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_PTE_COALESCE_LEVEL_CAP_128                (0x00000008U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_SHADER_ACCESS                         23:22
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_SHADER_ACCESS_DEFAULT               (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_SHADER_ACCESS_READ_ONLY             (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_SHADER_ACCESS_WRITE_ONLY            (0x00000002U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_SHADER_ACCESS_READ_WRITE            (0x00000003U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_FORMAT                                24:24
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_FORMAT_PTE_ARRAY                    (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_FORMAT_OS_ARRAY                     (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_DUPLICATE_FIRST_PTE                   25:25
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_DUPLICATE_FIRST_PTE_FALSE           (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_DUPLICATE_FIRST_PTE_TRUE            (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_VGPU_CACHE_FIRST_PTE                  26:26
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_VGPU_CACHE_FIRST_PTE_FALSE          (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_VGPU_CACHE_FIRST_PTE_TRUE           (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_ATOMIC_DISABLE                        27:27
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_ATOMIC_DISABLE_FALSE                (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_ATOMIC_DISABLE_TRUE                 (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_UNALIGNED_COMP                        28:28
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_UNALIGNED_COMP_FALSE                (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_UNALIGNED_COMP_TRUE                 (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_BUS_ADDRESS                           29:29
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_BUS_ADDRESS_FALSE                   (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_BUS_ADDRESS_TRUE                    (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_USE_COMPR_INFO                        30:30
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_USE_COMPR_INFO_FALSE                (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_USE_COMPR_INFO_TRUE                 (0x00000001U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_USE_PEER_ID                           31:31
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_USE_PEER_ID_FALSE                   (0x00000000U)
#define LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_USE_PEER_ID_TRUE                    (0x00000001U)

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0080_CTRL_DMA_FLUSH
 *
 * This command flushes the specified target unit
 *
 *   targetUnit
 *      The unit to flush, either L2 cache or compression tag cache.
 *      This field is a logical OR of the individual fields such as
 *      L2 cache or compression tag cache. Also L2 ilwalidation for
 *      either SYSMEM/PEERMEM is triggered. But this ilwalidation is 
 *      for FERMI.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_GENERIC
 *
 * See Also:
 *   LW2080_CTRL_CMD_FB_FLUSH_GPU_CACHE
 *     Flushes the entire GPU cache or a set of physical addresses (if the
 *     hardware supports it).  Use this call if you want to flush a set of
 *     addresses or the entire GPU cache in unicast mode.
 *   LW0041_CTRL_CMD_SURFACE_FLUSH_GPU_CACHE
 *     Flushes memory associated with a single allocation if the hardware
 *     supports it.  Use this call if you want to flush a single allocation and
 *     you have a memory object describing the physical memory.
 */
#define LW0080_CTRL_CMD_DMA_FLUSH                                                     (0x801805U) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_FLUSH_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_DMA_FLUSH_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW0080_CTRL_DMA_FLUSH_PARAMS {
    LwU32 targetUnit;
} LW0080_CTRL_DMA_FLUSH_PARAMS;

#define LW0080_CTRL_DMA_FLUSH_TARGET_UNIT_L2                         0:0
#define LW0080_CTRL_DMA_FLUSH_TARGET_UNIT_L2_DISABLE            (0x00000000U)
#define LW0080_CTRL_DMA_FLUSH_TARGET_UNIT_L2_ENABLE             (0x00000001U)
#define LW0080_CTRL_DMA_FLUSH_TARGET_UNIT_COMPTAG                    1:1
#define LW0080_CTRL_DMA_FLUSH_TARGET_UNIT_COMPTAG_DISABLE       (0x00000000U)
#define LW0080_CTRL_DMA_FLUSH_TARGET_UNIT_COMPTAG_ENABLE        (0x00000001U)
#define LW0080_CTRL_DMA_FLUSH_TARGET_UNIT_FB                         2:2
#define LW0080_CTRL_DMA_FLUSH_TARGET_UNIT_FB_DISABLE            (0x00000000U)
#define LW0080_CTRL_DMA_FLUSH_TARGET_UNIT_FB_ENABLE             (0x00000001U)

// This is exclusively for Fermi
// The selection of non-zero valued bit-fields avoids the routing 
// into the above cases and vice-versa
#define LW0080_CTRL_DMA_FLUSH_TARGET_UNIT_L2_ILWALIDATE              4:3
#define LW0080_CTRL_DMA_FLUSH_TARGET_UNIT_L2_ILWALIDATE_SYSMEM  (0x00000001U)
#define LW0080_CTRL_DMA_FLUSH_TARGET_UNIT_L2_ILWALIDATE_PEERMEM (0x00000002U)


#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/**
 * LW0080_CTRL_DMA_ADV_SCHED_GET_VA_CAPS
 *
 * This command returns information about the VA caps on the GPU
 *
 *   vaBitCount
 *     Returns number of bits in a virtual address
 *   pdeCoverageBitCount
 *     Returns number of VA bits covered in each PDE.  One PDE covers
 *     2^pdeCoverageBitCount bytes.
 *
 *   bigPageSize
 *     Size of the big page
 *   compressionPageSize
 *     Size of region each compression tag covers
 *   dualPageTableSupported
 *     TRUE if one page table can map with both 4KB and big pages
 *
 *   numPageTableFormats
 *     Returns the number of different page table sizes supported by the RM
 *   pageTableBigFormat
 *   pageTable4KFormat[]
 *     Returns size in bytes and number of VA bits covered by each page table
 *     format.  Up to MAX_NUM_PAGE_TABLE_FORMATS can be returned.  The most
 *     compact format will be pageTableSize[0] and the least compact format
 *     will be last.
 *   hVASpace
 *     handle for the allocated VA space that this control call should operate
 *     on. If it's 0, it assumes to use the implicit allocated VA space.
 *   vaRangeLo 
 *     Indicates the start of usable VA range.
 *
 *   hugePageSize
 *     Size of the huge page if supported, 0 otherwise.
 *
 *   vaSpaceId
 *     Virtual Address Space id assigned by RM.
 *     Only relevant on AMODEL.
 *
 *   pageSize512MB
 *     Size of the 512MB page if supported, 0 otherwise.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_GENERIC
 */
#define LW0080_CTRL_CMD_DMA_ADV_SCHED_GET_VA_CAPS               (0x801806U) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_ADV_SCHED_GET_VA_CAPS_PARAMS_MESSAGE_ID" */

typedef struct LW0080_CTRL_DMA_ADV_SCHED_GET_VA_CAPS_PAGE_TABLE_FORMAT {
    LwU32 pageTableSize;
    LwU32 pageTableCoverage;
} LW0080_CTRL_DMA_ADV_SCHED_GET_VA_CAPS_PAGE_TABLE_FORMAT;

#define LW0080_CTRL_DMA_ADV_SCHED_GET_VA_CAPS_MAX_NUM_PAGE_TABLE_FORMATS (16U)
#define LW0080_CTRL_DMA_ADV_SCHED_GET_VA_CAPS_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW0080_CTRL_DMA_ADV_SCHED_GET_VA_CAPS_PARAMS {
    LwU32                                                   vaBitCount;
    LwU32                                                   pdeCoverageBitCount;
    LwU32                                                   num4KPageTableFormats;
    LwU32                                                   bigPageSize;
    LwU32                                                   compressionPageSize;
    LwU32                                                   dualPageTableSupported;
    LwU32                                                   idealVRAMPageSize;
    LW0080_CTRL_DMA_ADV_SCHED_GET_VA_CAPS_PAGE_TABLE_FORMAT pageTableBigFormat;
    LW0080_CTRL_DMA_ADV_SCHED_GET_VA_CAPS_PAGE_TABLE_FORMAT pageTable4KFormat[LW0080_CTRL_DMA_ADV_SCHED_GET_VA_CAPS_MAX_NUM_PAGE_TABLE_FORMATS];
    LwHandle                                                hVASpace;
    LW_DECLARE_ALIGNED(LwU64 vaRangeLo, 8);
    LwU32                                                   hugePageSize;
    LwU32                                                   vaSpaceId;
    LwU32                                                   pageSize512MB;
} LW0080_CTRL_DMA_ADV_SCHED_GET_VA_CAPS_PARAMS;

/*
 * Adding a version define to allow clients to access valid
 * parameters based on version.
 */
#define LW0080_CTRL_CMD_DMA_ADV_SCHED_GET_VA_CAPS_WITH_VA_RANGE_LO 0x1U

/*
 * LW0080_CTRL_DMA_GET_PDE_INFO
 *
 * This command queries PDE information for the specified GPU virtual address.
 *
 *   gpuAddr
 *       This parameter specifies the GPU virtual address for which PDE
 *       information is to be returned.
 *   pdeVirtAddr
 *       This parameter returns the GPU virtual address of the PDE.
 *   pdeEntrySize
 *       This parameter returns the size of the PDE in bytes for this GPU.
 *   pdeAddrSpace
 *       This parameter returns the GPU address space of the PDE.
 *   pdeSize
 *       This parameter returns the fractional size of the page table(s) as
 *       actually set in the PDE, FULL, 1/2, 1/4 or 1/8.  (This amount may
 *       differ from that derived from pdeVASpaceSize.)  Intended for VERIF only.
 *   pteBlocks
 *       This parameter returns the page size-specific parameters as follows:
 *       ptePhysAddr
 *           This parameter returns the GPU physical address of the page table.
 *       pteCacheAttrib
 *           This parameter returns the caching attribute of the
 *           GPU physical address of the page table.
 *       pteEntrySize
 *           This parameter returns the size of the PTE in bytes for this GPU.
 *       pageSize
 *           This parameter returns the page size of the page table.
 *           If pageSize == 0, then this PTE block is not valid.
 *       pteAddrSpace
 *           This parameter returns the GPU address space of the page table.
 *       pdeVASpaceSize
 *           This parameter returns the size of the VA space addressable by
 *           the page table if fully used (i.e., if all PTEs marked VALID).
 *   pdbAddr
 *       This parameter returns the PDB address for the PDE.
 *   hVASpace
 *       handle for the allocated VA space that this control call should operate
 *       on. If it's 0, it assumes to use the implicit allocated VA space.
 *   
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_GENERIC
 */
#define LW0080_CTRL_CMD_DMA_GET_PDE_INFO                           (0x801809U) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS_MESSAGE_ID" */

typedef struct LW0080_CTRL_DMA_PDE_INFO_PTE_BLOCK {
    LW_DECLARE_ALIGNED(LwU64 ptePhysAddr, 8);
    LwU32 pteCacheAttrib;
    LwU32 pteEntrySize;
    LwU32 pageSize;
    LwU32 pteAddrSpace;
    LwU32 pdeVASpaceSize;
    LwU32 pdeFlags;
} LW0080_CTRL_DMA_PDE_INFO_PTE_BLOCK;

#define LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS_PTE_ADDR_SPACE_VIDEO_MEMORY               (0x00000000U)
#define LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS_PTE_ADDR_SPACE_SYSTEM_COHERENT_MEMORY     (0x00000001U)
#define LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS_PTE_ADDR_SPACE_SYSTEM_NON_COHERENT_MEMORY (0x00000002U)

#define LW0080_CTRL_DMA_PDE_INFO_PTE_BLOCKS                                           4U

#define LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 gpuAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 pdeVirtAddr, 8);
    LwU32    pdeEntrySize;
    LwU32    pdeAddrSpace;
    LwU32    pdeSize;
    LwU32    subDeviceId;
    LW_DECLARE_ALIGNED(LW0080_CTRL_DMA_PDE_INFO_PTE_BLOCK pteBlocks[LW0080_CTRL_DMA_PDE_INFO_PTE_BLOCKS], 8);
    LW_DECLARE_ALIGNED(LwU64 pdbAddr, 8);
    LwHandle hVASpace;
} LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS;

#define LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS_PDE_ADDR_SPACE_VIDEO_MEMORY               (0x00000000U)
#define LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS_PDE_ADDR_SPACE_SYSTEM_COHERENT_MEMORY     (0x00000001U)
#define LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS_PDE_ADDR_SPACE_SYSTEM_NON_COHERENT_MEMORY (0x00000002U)
#define LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS_PDE_SIZE_FULL                             1U
#define LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS_PDE_SIZE_HALF                             2U
#define LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS_PDE_SIZE_QUARTER                          3U
#define LW0080_CTRL_DMA_GET_PDE_INFO_PARAMS_PDE_SIZE_EIGHTH                           4U

/*
 * LW0080_CTRL_CMD_DMA_ILWALIDATE_PDB_TARGET
 *
 * This command ilwalidates PDB target setting in hardware.
 * After execeution of this command PDB target would be in undefined state.
 *
 * Returns error if the PDB target can not be ilwalidate.
 *
 * This call is only supported on chips fermi and later chips.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0080_CTRL_CMD_DMA_ILWALIDATE_PDB_TARGET                                     (0x80180bU) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | 0xB" */

/*
 * LW0080_CTRL_CMD_DMA_ILWALIDATE_TLB
 *
 * This command ilwalidates the GPU TLB. This is intended to be used
 * for RM clients that manage their own TLB consistency when updating
 * page tables on their own, or with DEFER_TLB_ILWALIDATION options
 * to other RM APIs.
 *
 *   hVASpace
 *     This parameter specifies the VASpace object whose MMU TLB entries
 *     needs to be ilwalidated, if the flag is set to LW0080_CTRL_DMA_ILWALIDATE_TLB_ALL_FALSE.
 *     Specifying a GMMU VASpace object handle will ilwalidate the GMMU TLB for the particular VASpace.
 *     Specifying a SMMU VASpace object handle will flush the entire SMMU TLB & PTC.
 *
 *   flags
 *     This parameter can be used to specify any flags needed
 *     for the ilwlalidation request.
 *       LW0080_CTRL_DMA_ILWALIDATE_TLB_ALL
 *         When set to TRUE this flag requests a global ilwalidate.
 *         When set to FALSE this flag requests a chip-specfic
 *         optimization to ilwalidate only the address space bound
 *         to the associated hDevice.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_TIMEOUT_RETRY
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0080_CTRL_CMD_DMA_ILWALIDATE_TLB                                            (0x80180lw) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_ILWALIDATE_TLB_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_DMA_ILWALIDATE_TLB_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW0080_CTRL_DMA_ILWALIDATE_TLB_PARAMS {
    LwHandle hVASpace;
    LwU32    flags;
} LW0080_CTRL_DMA_ILWALIDATE_TLB_PARAMS;

#define LW0080_CTRL_DMA_ILWALIDATE_TLB_ALL                     0:0
#define LW0080_CTRL_DMA_ILWALIDATE_TLB_ALL_FALSE (0x00000000U)
#define LW0080_CTRL_DMA_ILWALIDATE_TLB_ALL_TRUE  (0x00000001U)

/**
 * LW0080_CTRL_CMD_DMA_GET_CAPS
 *
 * This command returns the set of DMA capabilities for the device
 * in the form of an array of unsigned bytes.  DMA capabilities
 * include supported features and required workarounds for address
 * translation system within the device, each represented by a byte
 * offset into the table and a bit position within that byte.
 *
 *   capsTblSize
 *     This parameter specifies the size in bytes of the caps table.
 *     This value should be set to LW0080_CTRL_DMA_CAPS_TBL_SIZE.
 *
 *   capsTbl
 *     This parameter specifies a pointer to the client's caps table buffer
 *     into which the framebuffer caps bits will be transferred by the RM.
 *     The caps table is an array of unsigned bytes.
 *
 * 32BIT_POINTER_ENFORCED
 *     If this property is TRUE LWOS32 and LWOS46 calls with
 *     32BIT_POINTER_DISABLED will return addresses above 4GB.
 *
 * SHADER_ACCESS_SUPPORTED
 *     If this property is set, the MMU in the system supports the independent
 *     access bits for the shader.  This is accessed with the following fields:
 *         LWOS46_FLAGS_SHADER_ACCESS
 *         LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS_FLAGS_SHADER_ACCESS
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_DMA_GET_CAPS             (0x80180dU) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_GET_CAPS_PARAMS_MESSAGE_ID" */
/* size in bytes of fb caps table */
#define LW0080_CTRL_DMA_CAPS_TBL_SIZE            8U
#define LW0080_CTRL_DMA_GET_CAPS_PARAMS_MESSAGE_ID (0xDU)

typedef struct LW0080_CTRL_DMA_GET_CAPS_PARAMS {
    LwU32 capsTblSize;
    LwU8  capsTbl[LW0080_CTRL_DMA_CAPS_TBL_SIZE];
} LW0080_CTRL_DMA_GET_CAPS_PARAMS;

/* extract cap bit setting from tbl */
#define LW0080_CTRL_DMA_GET_CAP(tbl,c)              (((LwU8)tbl[(1?c)]) & (0?c))

/* caps format is byte_index:bit_mask */
#define LW0080_CTRL_DMA_CAPS_32BIT_POINTER_ENFORCED                0:0x01
#define LW0080_CTRL_DMA_CAPS_SHADER_ACCESS_SUPPORTED               0:0x04
#define LW0080_CTRL_DMA_CAPS_SPARSE_VIRTUAL_SUPPORTED              0:0x08
#define LW0080_CTRL_DMA_CAPS_MULTIPLE_VA_SPACES_SUPPORTED          0:0x10

/*
 * LW0080_CTRL_DMA_SET_VA_SPACE_SIZE
 *
 *   Change the size of an existing VA space.
 *   NOTE: Lwrrently this only supports growing the size, not shrinking.
 *
 *   1. Allocate new page directory able to map extended range.
 *   2. Copy existing PDEs from old directory to new directory.
 *   3. Initialize new PDEs to invalid.
 *   4. Update instmem to point to new page directory.
 *   5. Free old page directory.
 *
 *   vaSpaceSize
 *      On input, the requested size of the VA space in bytes.
 *      On output, the actual resulting VA space size.
 *
 *      The actual size will be greater than or equal to the requested size,
 *      unless LW0080_CTRL_DMA_GROW_VA_SPACE_SIZE_MAX is requested, which
 *      requests the maximum available.
 *      
 *      NOTE: Specific size requests (e.g. other than SIZE_MAX) must account
 *            for the VA hole at the beginning of the range which is used to
 *            distinguish NULL pointers. This region is not counted as part
 *            of the vaSpaceSize since it is not allocatable.
 *
 *   hVASpace
 *      handle for the allocated VA space that this control call should operate
 *      on. If it's 0, it assumes to use the implicit allocated VA space
 *      associated with the client/device pair.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_INSUFFICIENT_RESOURCES
 */
#define LW0080_CTRL_CMD_DMA_SET_VA_SPACE_SIZE (0x80180eU) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_SET_VA_SPACE_SIZE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_DMA_SET_VA_SPACE_SIZE_PARAMS_MESSAGE_ID (0xEU)

typedef struct LW0080_CTRL_DMA_SET_VA_SPACE_SIZE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 vaSpaceSize, 8);
    LwHandle hVASpace;
} LW0080_CTRL_DMA_SET_VA_SPACE_SIZE_PARAMS;

#define LW0080_CTRL_DMA_SET_VA_SPACE_SIZE_MAX (0xFFFFFFFFFFFFFFFFULL)

/*
 * LW0080_CTRL_DMA_UPDATE_PDE_2
 *
 * This command updates a single PDE for the given (hClient, hDevice)
 * with specific attributes.
 * This command is only available on Windows and MODS platforms.
 * This command can be called by kernel clients only.
 *
 * The VA range the PDE maps must be contained by a VA allocation marked with
 * LWOS32_ALLOC_FLAGS_EXTERNALLY_MANAGED.
 * However if the MODS-only FORCE_OVERRIDE flag is set this restriction is relaxed.
 *
 * RM does not track the PDE's attributes in SW - this control simply stuffs
 * the PDE in memory after translating and checking the parameters.
 *
 * Parameters are checked for relative consistency (e.g. valid domains),
 * but it is the client's responsibility to provide correct page table
 * addresses, e.g. global consistency is not checked.
 *
 * It is also the client's responsibility to flush/ilwalidate the MMU
 * when appropriate, either by setting the _FLUSH_PDE_CACHE flag for this
 * call or by flushing through other APIs.
 * This control does not flush automatically to allow batches of calls
 * to be made before a single flush.
 *
 *   ptParams
 *      Page-size-specific parameters, as follows:
 *
 *      physAddr
 *         Base address of physically contiguous memory of page table.
 *         Must be aligned sufficiently for the PDE address field.
 *      numEntries
 *         Deprecated and ignored.
 *         Use FLAGS_PDE_SIZE that applies to the tables for all page sizes.
 *      aperture
 *         Address space the base address applies to.
 *         Can be left as INVALID to ignore this page table size.
 *
 *   pdeIndex
 *      The PDE index this update applies to.
 *   flags
 *      See LW0080_CTRL_DMA_UPDATE_PDE_FLAGS_*.
 *   hVASpace
 *      handle for the allocated VA space that this control call should operate
 *      on. If it's 0, it assumes to use the implicit allocated VA space
 *      associated with the client/device pair.
 *   pPdeBuffer [out]
 *      Kernel pointer to 64 bit unsigned integer representing a Page Dir Entry
 *      that needs to be updated. It should point to memory as wide as the Page Dir
 *      Entry.
 *
 *      If NULL, Page Dir Entry updates will go to the internally managed Page Dir.
 *      If not NULL, the updates will be written to this buffer.
 *
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_GENERIC
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0080_CTRL_CMD_DMA_UPDATE_PDE_2      (0x80180fU) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_UPDATE_PDE_2_PARAMS_MESSAGE_ID" */

typedef struct LW0080_CTRL_DMA_UPDATE_PDE_2_PAGE_TABLE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 physAddr, 8);
    LwU32 numEntries;   // deprecated
    LwU32 aperture;
} LW0080_CTRL_DMA_UPDATE_PDE_2_PAGE_TABLE_PARAMS;

#define LW0080_CTRL_DMA_UPDATE_PDE_2_PT_APERTURE_ILWALID                    (0x00000000U)
#define LW0080_CTRL_DMA_UPDATE_PDE_2_PT_APERTURE_VIDEO_MEMORY               (0x00000001U)
#define LW0080_CTRL_DMA_UPDATE_PDE_2_PT_APERTURE_SYSTEM_COHERENT_MEMORY     (0x00000002U)
#define LW0080_CTRL_DMA_UPDATE_PDE_2_PT_APERTURE_SYSTEM_NON_COHERENT_MEMORY (0x00000003U)

#define LW0080_CTRL_DMA_UPDATE_PDE_2_PT_IDX_SMALL                           0U
#define LW0080_CTRL_DMA_UPDATE_PDE_2_PT_IDX_BIG                             1U
#define LW0080_CTRL_DMA_UPDATE_PDE_2_PT_IDX__SIZE                           2U

#define LW0080_CTRL_DMA_UPDATE_PDE_2_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW0080_CTRL_DMA_UPDATE_PDE_2_PARAMS {
    LwU32    pdeIndex;
    LwU32    flags;
    LW_DECLARE_ALIGNED(LW0080_CTRL_DMA_UPDATE_PDE_2_PAGE_TABLE_PARAMS ptParams[LW0080_CTRL_DMA_UPDATE_PDE_2_PT_IDX__SIZE], 8);
    LwHandle hVASpace;
    LW_DECLARE_ALIGNED(LwP64 pPdeBuffer, 8); // LW_MMU_VER2_PDE__SIZE
    LwU32    subDeviceId; // ID+1, 0 for BC
} LW0080_CTRL_DMA_UPDATE_PDE_2_PARAMS;

/*!
 * If set a PDE cache flush (MMU ilwalidate) will be performed.
 */
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_FLUSH_PDE_CACHE          0:0
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_FLUSH_PDE_CACHE_FALSE (0x00000000U)
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_FLUSH_PDE_CACHE_TRUE  (0x00000001U)

/*!
 * For verification purposes (MODS-only) this flag may be set to modify any PDE
 * in the VA space (RM managed or externally managed).
 * It is up to caller to restore any changes properly (or to expect faults).
 */
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_FORCE_OVERRIDE           1:1
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_FORCE_OVERRIDE_FALSE  (0x00000000U)
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_FORCE_OVERRIDE_TRUE   (0x00000001U)

/*!
 * Directly controls the PDE_SIZE field (size of the page tables pointed to by this PDE).
 */
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_PDE_SIZE                 3:2
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_PDE_SIZE_FULL         (0x00000000U)
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_PDE_SIZE_HALF         (0x00000001U)
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_PDE_SIZE_QUARTER      (0x00000002U)
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_PDE_SIZE_EIGHTH       (0x00000003U)

/*! 
 * Used to specify if the allocation is sparse. Applicable only in case of 
 * VA Space managed by OS, as in WDDM2.0
 */
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_SPARSE                   4:4
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_SPARSE_FALSE          (0x00000000U)
#define LW0080_CTRL_DMA_UPDATE_PDE_2_FLAGS_SPARSE_TRUE           (0x00000001U)

/*
 * LW0080_CTRL_DMA_ENABLE_PRIVILEGED_RANGE
 * This interface will create a corresponding privileged
 * kernel address space that will mirror user space allocations in this
 * VASPACE.
 * The user can either pass a FERMI_VASPACE_A handle or RM will use the 
 * vaspace associated with the client/device if hVaspace is passed as 
 * NULL.
 * Once this property is set, the user will not be able to make allocations
 * from  the top most PDE of this address space.
 *
 * The user is expected to call this function as soon as he has created 
 * the device/Vaspace object. If the user has already made VA allocations 
 * in this vaspace then this call will return a failure 
 * (LW_ERR_ILWALID_STATE). 
 * The Vaspace should have no VA allocations when this call is made.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
*/
#define LW0080_CTRL_DMA_ENABLE_PRIVILEGED_RANGE                  (0x801810U) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_ENABLE_PRIVILEGED_RANGE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_DMA_ENABLE_PRIVILEGED_RANGE_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW0080_CTRL_DMA_ENABLE_PRIVILEGED_RANGE_PARAMS {
    LwHandle hVASpace;
} LW0080_CTRL_DMA_ENABLE_PRIVILEGED_RANGE_PARAMS;

/*
 * LW0080_CTRL_DMA_SET_DEFAULT_VASPACE
 * This is a special control call provided for KMD to use. 
 * It will associate an allocated Address Space Object as the 
 * default address space of the device.
 * 
 * This is added so that the USER can move to using address space objects when they 
 * want to specify the size of the big page size they want to use but still want
 * to use the rest of the relevant RM apis without specifying the hVASpace.
 * 
 * This call will succeed only if there is already no VASPACE associated with the 
 * device. This means the user will have to call this before he has made any allocations
 * on this device/address space.
 *
 * The hVASpace that is passed in to be associated shoould belong to the parent device that
 * this call is made for. This call will fail if we try to associate a VASpace belonging to 
 * some other client/device.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *  
 */
#define LW0080_CTRL_DMA_SET_DEFAULT_VASPACE (0x801812U) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_SET_DEFAULT_VASPACE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_DMA_SET_DEFAULT_VASPACE_PARAMS_MESSAGE_ID (0x12U)

typedef struct LW0080_CTRL_DMA_SET_DEFAULT_VASPACE_PARAMS {
    LwHandle hVASpace;
} LW0080_CTRL_DMA_SET_DEFAULT_VASPACE_PARAMS;

/*!
 * LW0080_CTRL_DMA_SET_PAGE_DIRECTORY
 *
 * Move an existing VA space to an externally-managed top-level page directory.
 * The VA space must have been created in SHARED_MANAGEMENT mode.
 * For lifecycle details, see LW_VASPACE_ALLOCATION_PARAMETERS documentation in lwos.h.
 *
 * RM will propagate the update to all channels using the VA space.
 *
 * NOTE: All channels using this VA space are expected to be idle and unscheduled prior
 *       to and during this control call - it is responsibility of caller to ensure this.
 *
 *   physAddress
 *      Physical address of the new page directory within the aperture specified by flags.
 *   numEntries
 *      Number of entries in the new page directory.
 *      The backing phyical memory must be at least this size (multiplied by entry size).
 *   flags
 *      APERTURE
 *          Specifies which physical aperture the page directory resides.
 *      PRESERVE_PDES
 *          Deprecated - RM will always copy the RM-managed PDEs from the old page directory
 *          to the new page directory.
 *      ALL_CHANNELS
 *          If true, RM will update the instance blocks for all channels using
 *          the VAS and ignore the chId parameter.
 *      EXTEND_VASPACE
 *          If true, RM will use the client VA for client VA requests in VASPACE_SHARED_MANAGEMENT mode
 *          If false, RM will use the internal VA for client VA requests.
 *      IGNORE_CHANNEL_BUSY
 *          If true, RM will ignore the channel busy status during set page
 *          directory operation.
 *   hVASpace
 *      handle for the allocated VA space that this control call should operate
 *      on. If it's 0, it assumes to use the implicit allocated VA space
 *      associated with the client/device pair.
 *   chId
 *      ID of the Channel to be updated.
 *   pasid
 *      PASID (Process Address Space IDentifier) of the process corresponding to
 *      the VA space. Ignored unless the VA space has ATS enabled.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_LIMIT
 *   LW_ERR_GENERIC
 */
#define LW0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY (0x801813U) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 physAddress, 8);
    LwU32    numEntries;
    LwU32    flags;
    LwHandle hVASpace;
    LwU32    chId;
    LwU32    subDeviceId; // ID+1, 0 for BC
    LwU32    pasid;
} LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS;

#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_APERTURE                  1:0
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_APERTURE_VIDMEM           (0x00000000U)
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_APERTURE_SYSMEM_COH       (0x00000001U)
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_APERTURE_SYSMEM_NONCOH    (0x00000002U)
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_PRESERVE_PDES             2:2
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_PRESERVE_PDES_FALSE       (0x00000000U)
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_PRESERVE_PDES_TRUE        (0x00000001U)
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_ALL_CHANNELS              3:3
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_ALL_CHANNELS_FALSE        (0x00000000U)
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_ALL_CHANNELS_TRUE         (0x00000001U)
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_IGNORE_CHANNEL_BUSY       4:4
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_IGNORE_CHANNEL_BUSY_FALSE (0x00000000U)
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_IGNORE_CHANNEL_BUSY_TRUE  (0x00000001U)
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_EXTEND_VASPACE            5:5
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_EXTEND_VASPACE_FALSE      (0x00000000U)
#define LW0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_EXTEND_VASPACE_TRUE       (0x00000001U)

/*!
 * LW0080_CTRL_DMA_UNSET_PAGE_DIRECTORY
 *
 * Restore an existing VA space to an RM-managed top-level page directory.
 * The VA space must have been created in SHARED_MANAGEMENT mode and
 * previously relocated to an externally-managed page directory with
 * LW0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY (these two API are symmetric operations).
 * For lifecycle details, see LW_VASPACE_ALLOCATION_PARAMETERS documentation in lwos.h.
 *
 * RM will propagate the update to all channels using the VA space.
 *
 * NOTE: All channels using this VA space are expected to be idle and unscheduled prior
 *       to and during this control call - it is responsibility of caller to ensure this.
 *
 *   hVASpace
 *      handle for the allocated VA space that this control call should operate
 *      on. If it's 0, it assumes to use the implicit allocated VA space
 *      associated with the client/device pair.
 */
#define LW0080_CTRL_CMD_DMA_UNSET_PAGE_DIRECTORY                           (0x801814U) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_UNSET_PAGE_DIRECTORY_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_DMA_UNSET_PAGE_DIRECTORY_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW0080_CTRL_DMA_UNSET_PAGE_DIRECTORY_PARAMS {
    LwHandle hVASpace;
    LwU32    subDeviceId; // ID+1, 0 for BC
} LW0080_CTRL_DMA_UNSET_PAGE_DIRECTORY_PARAMS;

/*!
 * LW0080_CTRL_DMA_TRANSLATE_GPU_PTE
 *
 * This API is used for vGPU PTE translations, which will translate the
 * address inside GPU PTEs from guest address space into machine address space
 *
 * For system memory
 *
 *  With IOMMU support
 *
 *      Guest PFN --> Host PFN --> GPU --< IOMMU >--> MFN
 *
 *  Without IOMMU support
 *
 *      Guest PFN --> MFN --> GPU --> MFN
 *
 *  For FB
 *
 *      Guest PFN --< vGPU manager >--> Host MFN
 *
 * pageCount
 *      This parameter specifies the number of PTE entries to be translated
 *
 * pteMem
 *      This parameter specifies a pointer to memory filled with GPU specific
 *      PTE data, which will be translated
 *
 * flags
 *      This parameter specifies flag values to use for this translation operation
 *      FLAGS_APERTURE
 *          If nonzero, specifies to override aperture setting
 *      FLAGS_VGPU_CACHE_FIRST_PTE
 *          cache first PTE in guest RM
 *      FLAGS_DST_PHYS_ADDR
 *          flag to indidate destPhysAddr variable has valid value
 *      FLAGS_NEW_TRANSLATE_REQUEST
 *          flag to indicate new translate request. It is used by guest RM to
 *          indicate new translate request.
 *      FLAGS_LAST_TRANSLATE_REQUEST
 *          flag to indicate last translation request inside translate
 *          guest pte request. This is used by guest RM
 *      FLAGS_TLB_FLUSH_REQUEST
 *          flag to indicate tlb flush request
 *      FLAGS_VALID
 *          flag indicates PTE present in pteMem are valid and require translation
 *      FLAGS_IMMEDIATE_BLIT
 *          flag indicates PTEs are blit immediately using private channel
 *      FLAGS_RESERVED
 *          flag is free to reuse
 *      FLAGS_TRANSLATE_DST_PHYS_ADDR
 *          flag indicates if destPhysAddr also needs to be translated by vGPU host
 *      FLAGS_PAGE_TABLE_LEVEL
 *          flag contains page table level for which translation requested
 *      FLAGS_DST_PHYS_ADDR_BAR1_OFFSET
 *          flag set to TRUE if pteMem is CPU VA pointing to BAR1 and
 *          destPhysAddr contains BAR1 offset.
 *      FLAGS_ZLWLL_BACKING_STORE
 *          flag set to TRUE if the memory pointed to by ptes is ZLWLL backing
 *          store buffer
 *
 * gpuAddr
 *      This parameter specifies the GPU virtual address of an existing
 *      mapping from which other attributes are extracted.
 *
 * destPhysAddr
 *      This parameter specifies destination physical address for page tables
 *
 * vgpuBlitSemaValue
 *      This parameter specifies synchronization semaphore value used by plugin
 *      during blit operation.
 *
 * ptesDstPteAperture
 *      This parameter specifies destination page table aperture
 *
 * pageSize
 *      This parameter specifies the page size pointed to by each PTE entry.
 *
 * hVASpace
 *      The caller's vaspace handle.
 *
 * pdeSize
 *      This parameter specifies the pde size when translation is for page directory
 *
 * hwResource{hClient, hMemory}
 *      This parameter specifies the handle of the physical memory allocation
 *      from which to extract the comptag information
 */
#define LW0080_CTRL_CMD_TRANSLATE_GUEST_GPU_PTES                                              (0x801815U) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_APERTURE                        0:0
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_APERTURE_VIDEO_MEMORY           (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_APERTURE_SYSTEM_MEMORY          (0x00000001U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_VGPU_CACHE_FIRST_PTE            1:1
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_VGPU_CACHE_FIRST_PTE_FALSE      (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_VGPU_CACHE_FIRST_PTE_TRUE       (0x00000001U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_DST_PHYS_ADDR                   2:2
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_DST_PHYS_ADDR_FALSE             (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_DST_PHYS_ADDR_TRUE              (0x00000001U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_NEW_TRANSLATE_REQUEST           3:3
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_NEW_TRANSLATE_REQUEST_FALSE     (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_NEW_TRANSLATE_REQUEST_TRUE      (0x00000001U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_LAST_TRANSLATE_REQUEST          4:4
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_LAST_TRANSLATE_REQUEST_FALSE    (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_LAST_TRANSLATE_REQUEST_TRUE     (0x00000001U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_TLB_FLUSH_REQUEST               5:5
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_TLB_FLUSH_REQUEST_FALSE         (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_TLB_FLUSH_REQUEST_TRUE          (0x00000001U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_VALID                           6:6
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_VALID_FALSE                     (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_VALID_TRUE                      (0x00000001U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_IMMEDIATE_BLIT                  7:7
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_IMMEDIATE_BLIT_FALSE            (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_IMMEDIATE_BLIT_TRUE             (0x00000001U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_RESERVED                        8:8
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_RESERVED_FALSE                  (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_RESERVED_TRUE                   (0x00000001U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_TRANSLATE_DST_PHYS_ADDR         9:9
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_TRANSLATE_DST_PHYS_ADDR_FALSE   (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_TRANSLATE_DST_PHYS_ADDR_TRUE    (0x00000001U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_PAGE_TABLE_LEVEL                11:10
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_PAGE_TABLE_LEVEL_ZERO           (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_PAGE_TABLE_LEVEL_ONE            (0x00000001U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_PAGE_TABLE_LEVEL_TWO            (0x00000002U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_DST_PHYS_ADDR_BAR1_OFFSET       12:12
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_DST_PHYS_ADDR_BAR1_OFFSET_FALSE (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_DST_PHYS_ADDR_BAR1_OFFSET_TRUE  (0x00000001U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_ZLWLL_BACKING_STORE             13:13
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_ZLWLL_BACKING_STORE_FALSE       (0x00000000U)
#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_FLAGS_ZLWLL_BACKING_STORE_TRUE        (0x00000001U)

#define LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS_MESSAGE_ID (0x15U)

typedef struct LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS {
    LwU32 flags;
    LwU32 pageCount;
    LW_DECLARE_ALIGNED(LwP64 pteMem, 8);
    LW_DECLARE_ALIGNED(LwU64 gpuAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 destPhysAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 vgpuBlitSemaValue, 8);
    LwU32 ptesDstPteAperture;
    LwU32 pageSize;
    LwU32 hVASpace;
    LwU32 pdeSize;
    struct {
        LwHandle hClient;
        LwHandle hMemory;
    } hwResource;
} LW0080_CTRL_DMA_TRANSLATE_GUEST_GPU_PTES_PARAMS;

/*!
 * LW0080_CTRL_DMA_UPDATE_GPU_PDES
 *
 * This API does the GPU page table manipulations, which is necessary to
 * handle the pascal 5-level MMU support in WDDDM-v2 through KMD.
 * This is used only on vGPU.
 *
 *  physAddrss
 *      This contains the physical address of OS specified PDE, if 
 *      the flag is SET. This field is not valid with flag UNSET.
 *
 *  flags
 *    FLAGS_OPERATION
 *      This controls whether the OS specified PDE is being set,
 *      or unset.
 *    FLAGS_COPY_NEXT_PDE_ENTRIES
 *      This controls whether we want to copy the (pdeLevel+1) entries,
 *      if the operation flag is set.
 *    FLAGS_ILWALIDATE_TLB
 *      This controls whether we need the MMU TLB to be ilwalidated.
 *
 *  pdeLevel
 *      This contains the page-table level whose entries are to be
 *      modified so as to point to OS specified PDE.
 *
 *  driverReservePdeLo
 *      Start of reserved VA range for PDE of (pdeLevel+1) in KMD.
 *      This is valid in OPERATION_SET and COPY_NEXT_PDE_ENTRIES_YES.
 *      
 *  driverReservePdeHi
 *      End of reserved VA range for PDE of (pdeLevel+1) in KMD.
 *      This is valid in OPERATION_SET and COPY_NEXT_PDE_ENTRIES_YES.
 *
 *  hVASpace
 *      The caller's vaspace handle.
 *
 *  hChannelClient
 *      The channel client handle to ilwalidate the TLB, during SET 
 *      operation, if the ILWALIDATE_TLB flag is set.
 *
 *  hChannelDevice
 *      The channel device handle to ilwalidate the TLB, during SET 
 *      operation, if the ILWALIDATE_TLB flag is set.
 *
 *  hChannelSubDevice
 *      The channel sub-device handle to ilwalidate the TLB, during SET 
 *      operation, if the ILWALIDATE_TLB flag is set.
 *
 */
#define LW0080_CTRL_CMD_DMA_UPDATE_GPU_PDES                                    (0x801816U) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_DMA_INTERFACE_ID << 8) | LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS_FLAGS_OPERATION                               0:0
#define LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS_FLAGS_OPERATION_UNSET           (0x00000000U)
#define LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS_FLAGS_OPERATION_SET             (0x00000001U)
#define LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS_FLAGS_COPY_NEXT_PDE_ENTRIES                   1:1
#define LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS_FLAGS_COPY_NEXT_PDE_ENTRIES_NO  (0x00000000U)
#define LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS_FLAGS_COPY_NEXT_PDE_ENTRIES_YES (0x00000001U)
#define LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS_FLAGS_ILWALIDATE_TLB                          2:2
#define LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS_FLAGS_ILWALIDATE_TLB_NO         (0x00000000U)
#define LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS_FLAGS_ILWALIDATE_TLB_YES        (0x00000001U)

#define LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 physAddress, 8);
    LwU32 flags;
    LwU32 pdeLevel;
    LwU32 driverReservePdeLo;
    LwU32 driverReservePdeHi;
    LwU32 hVASpace;
    LwU32 hChannelClient;
    LwU32 hChannelDevice;
    LwU32 hChannelSubDevice;
} LW0080_CTRL_DMA_UPDATE_GPU_PDES_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/* _ctrl0080dma_h_ */

