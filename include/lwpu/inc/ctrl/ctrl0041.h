/*
 * Copyright (c) 2004-2021, LWPU CORPORATION. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
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
// Source file: ctrl/ctrl0041.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#include "lwos.h"



#include "ctrl/ctrlxxxx.h"
/* LW04_MEMORY control commands and parameters */

#define LW0041_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0x0041, LW0041_CTRL_##cat, idx)

/* LW04_MEMORY command categories (6bits) */
#define LW0041_CTRL_RESERVED (0x00)
#define LW0041_CTRL_MEMORY   (0x01)

/*
 * LW0041_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0041_CTRL_CMD_NULL (0x410000) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW0041_CTRL_CMD_UPDATE_PTE_ARRAY
 *
 * This command can be used to update the PteArray of a memory object, based
 * on the current state of the corresponding OS memory descriptor.  As such,
 * it is only valid on objects of class LW01_MEMORY_SYSTEM_OS_DESCRIPTOR.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_ERROR
 *   LW_ERR_ILWALID_CLASS
 *   LW_ERR_ILWALID_ARGUMENT
 */
//#define LW0041_CTRL_CMD_UPDATE_PTE_ARRAY          LW0041_CTRL_CMD(MEMORY, 0x02)

/*
 * LW0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR
 *
 * This command returns attributes associated with the memory object
 * at the given offset. The architecture dependent return parameter
 * comprFormat determines the meaningfulness (or not) of comprOffset.
 *
 * This call is only lwrrently supported in the MODS environment.
 *
 *   memOffset
 *     This parameter is both an input and an output. As input, this
 *     parameter holds an offset into the memory surface. The return
 *     value is the physical address of the surface at the given offset.
 *   memFormat
 *     This parameter returns the memory kind of the surface.
 *   comprOffset
 *     This parameter returns the compression offset of the surface.
 *   comprFormat
 *     This parameter returns the type of compression of the surface.
 *   memAperture
 *     The aperture of the surface is returned in this field.
 *     Legal return values for this parameter are
 *       LW0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR_APERTURE_VIDMEM
 *       LW0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR_APERTURE_SYSMEM
 *   gpuCacheAttr
 *     gpuCacheAttr returns the gpu cache attribute of the surface.
 *     Legal return values for this field are
 *       LW0041_CTRL_GET_SURFACE_PHYS_ATTR_GPU_CACHED_UNKNOWN
 *       LW0041_CTRL_GET_SURFACE_PHYS_ATTR_GPU_CACHED
 *       LW0041_CTRL_GET_SURFACE_PHYS_ATTR_GPU_UNCACHED
 *   gpuP2PCacheAttr
 *     gpuP2PCacheAttr returns the gpu peer-to-peer cache attribute of the surface.
 *     Legal return values for this field are
 *       LW0041_CTRL_GET_SURFACE_PHYS_ATTR_GPU_CACHED_UNKNOWN
 *       LW0041_CTRL_GET_SURFACE_PHYS_ATTR_GPU_CACHED
 *       LW0041_CTRL_GET_SURFACE_PHYS_ATTR_GPU_UNCACHED
 *   mmuContext
 *     mmuContext indicates the type of physical address to be returned (input parameter).
 *     Legal return values for this field are
 *       TEGRA_VASPACE_A  --  return the device physical address for CheetAh (non-GPU) engines. This is the system physical address itself.
 *                            returns the system physical address. This may change to use a class value in future.
 *       FERMI_VASPACE_A  --  return the device physical address for GPU engines. This can be a system physical address or a GPU SMMU virtual address.
 *                     0  --  return the device physical address for GPU engines. This can be a system physical address or a GPU SMMU virtual address.
 *                            use of zero may be deprecated in future.
 *   contigSegmentSize
 *     If the underlying surface is physically contiguous, this parameter
 *     returns the size in bytes of the piece of memory starting from
 *     the offset specified in the memOffset parameter extending to the last
 *     byte of the surface.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_BAD_OBJECT_HANDLE
 *   LWOS_STATUS_BAD_OBJECT_PARENT
 *   LWOS_STATUS_NOT_SUPPORTED
 *
 */
#define LW0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR (0x410103) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS_MESSAGE_ID" */

#define LW0041_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0041_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 memOffset, 8);
    LwU32 memFormat;
    LwU32 comprOffset;
    LwU32 comprFormat;
    LwU32 memAperture;
    LwU32 gpuCacheAttr;
    LwU32 gpuP2PCacheAttr;
    LwU32 mmuContext;
    LW_DECLARE_ALIGNED(LwU64 contigSegmentSize, 8);
} LW0041_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS;

/* valid memAperture return values */
#define LW0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR_APERTURE_VIDMEM (0x00000000)
#define LW0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR_APERTURE_SYSMEM (0x00000001)

/* valid gpuCacheAttr return values */
#define LW0041_CTRL_GET_SURFACE_PHYS_ATTR_GPU_CACHED_UNKNOWN  (0x00000000)
#define LW0041_CTRL_GET_SURFACE_PHYS_ATTR_GPU_CACHED          (0x00000001)
#define LW0041_CTRL_GET_SURFACE_PHYS_ATTR_GPU_UNCACHED        (0x00000002)

/*
 * LW0041_CTRL_CMD_GET_SURFACE_ZLWLL_ID
 *
 * This command returns the Z-lwll identifier for a surface.
 * The value of ~0 is returned if there is none associated.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_BAD_OBJECT_HANDLE
 *   LWOS_STATUS_BAD_OBJECT_PARENT
 *   LWOS_STATUS_NOT_SUPPORTED
 *
 */
#define LW0041_CTRL_CMD_GET_SURFACE_ZLWLL_ID                  (0x410104) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_GET_SURFACE_ZLWLL_ID_PARAMS_MESSAGE_ID" */

#define LW0041_CTRL_GET_SURFACE_ZLWLL_ID_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0041_CTRL_GET_SURFACE_ZLWLL_ID_PARAMS {
    LwU32 zlwllId;
} LW0041_CTRL_GET_SURFACE_ZLWLL_ID_PARAMS;

/*
 * LW0041_CTRL_CMD_GET_SURFACE_PARTITION_STRIDE
 *
 * This command returns the partition stride (in bytes) for real memory 
 * associated with the memory object.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_BAD_OBJECT_HANDLE
 *   LWOS_STATUS_BAD_OBJECT_PARENT
 *   LWOS_STATUS_NOT_SUPPORTED
 *
 */
#define LW0041_CTRL_CMD_GET_SURFACE_PARTITION_STRIDE (0x410105) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_GET_SURFACE_PARTITION_STRIDE_PARAMS_MESSAGE_ID" */

#define LW0041_CTRL_GET_SURFACE_PARTITION_STRIDE_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW0041_CTRL_GET_SURFACE_PARTITION_STRIDE_PARAMS {
    LwU32 partitionStride;
} LW0041_CTRL_GET_SURFACE_PARTITION_STRIDE_PARAMS;



// return values for 'tilingFormat'
// XXX - the names for these are misleading
#define LW0041_CTRL_CMD_GET_SURFACE_TILING_FORMAT_ILWALID   (0x00000000)
#define LW0041_CTRL_CMD_GET_SURFACE_TILING_FORMAT_FB        (0x00000001)
#define LW0041_CTRL_CMD_GET_SURFACE_TILING_FORMAT_FB_1HIGH  (0x00000002)
#define LW0041_CTRL_CMD_GET_SURFACE_TILING_FORMAT_FB_4HIGH  (0x00000003)
#define LW0041_CTRL_CMD_GET_SURFACE_TILING_FORMAT_UMA_1HIGH (0x00000004)
#define LW0041_CTRL_CMD_GET_SURFACE_TILING_FORMAT_UMA_4HIGH (0x00000005)

/*
 * LW0041_CTRL_SURFACE_INFO
 *
 * This structure represents a single 32bit surface value.  Clients
 * request a particular surface value by specifying a unique surface
 * information index.
 *
 * Legal surface information index values are:
 *   LW0041_CTRL_SURFACE_INFO_INDEX_ATTRS
 *     This index is used to request the set of hw attributes associated
 *     with the surface.  Each distinct attribute is represented by a
 *     single bit flag in the returned value.
 *     Legal flags values for this index are:
 *       LW0041_CTRL_SURFACE_INFO_ATTRS_COMPR
 *         This surface has compression resources bound to it.
 *       LW0041_CTRL_SURFACE_INFO_ATTRS_ZLWLL
 *         This surface has zlwll resources bound to it.
 *   LW0041_CTRL_SURFACE_INFO_INDEX_COMPR_COVERAGE
 *     This index is used to request the compression coverage (if any)
 *     in units of 64K for the associated surface.  A value of zero indicates
 *     there are no compression resources associated with the surface.
 *     Legal return values range from zero to a maximum number of 64K units
 *     that is GPU implementation dependent.
 *   LW0041_CTRL_SURFACE_INFO_INDEX_PHYS_SIZE
 *     This index is used to request the physically allocated size in units
 *     of 4K(LW0041_CTRL_SURFACE_INFO_PHYS_SIZE_SCALE_FACTOR) for the associated
 *     surface.
 *   LW0041_CTRL_SURFACE_INFO_INDEX_PHYS_ATTR
 *     This index is used to request the surface attribute field. The returned
 *     field value can be decoded using the LW0041_CTRL_SURFACE_INFO_PHYS_ATTR_*
 *     DRF-style macros provided below.
 *   LW0041_CTRL_SURFACE_INFO_INDEX_ADDR_SPACE_TYPE
 *     This index is used to request the surface address space type.
 *     Returned values are described by LW0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE.
 */
typedef struct LW0041_CTRL_SURFACE_INFO {
    LwU32 index;
    LwU32 data;
} LW0041_CTRL_SURFACE_INFO;

/* valid surface info index values */
#define LW0041_CTRL_SURFACE_INFO_INDEX_ATTRS                 (0x00000001)
#define LW0041_CTRL_SURFACE_INFO_INDEX_COMPR_COVERAGE        (0x00000005)
#define LW0041_CTRL_SURFACE_INFO_INDEX_PHYS_SIZE             (0x00000007)
#define LW0041_CTRL_SURFACE_INFO_INDEX_PHYS_ATTR             (0x00000008)
#define LW0041_CTRL_SURFACE_INFO_INDEX_ADDR_SPACE_TYPE       (0x00000009)

/*
 * This define indicates the scale factor of the reported physical size to the
 * actual size in bytes. We use the scale factor to save space from the
 * interface and account for large surfaces. To get the actual size,
 * use `(LwU64)reported_size * LW0041_CTRL_SURFACE_INFO_PHYS_SIZE_SCALE_FACTOR`.
 */
#define LW0041_CTRL_SURFACE_INFO_PHYS_SIZE_SCALE_FACTOR      (0x1000)

/* valid surface info attr flags */
#define LW0041_CTRL_SURFACE_INFO_ATTRS_COMPR                 (0x00000002)
#define LW0041_CTRL_SURFACE_INFO_ATTRS_ZLWLL                 (0x00000004)

/* Valid surface info page size */
#define LW0041_CTRL_SURFACE_INFO_PHYS_ATTR_PAGE_SIZE                 LWOS32_ATTR_PAGE_SIZE
#define LW0041_CTRL_SURFACE_INFO_PHYS_ATTR_PAGE_SIZE_DEFAULT LWOS32_ATTR_PAGE_SIZE_DEFAULT
#define LW0041_CTRL_SURFACE_INFO_PHYS_ATTR_PAGE_SIZE_4KB     LWOS32_ATTR_PAGE_SIZE_4KB
#define LW0041_CTRL_SURFACE_INFO_PHYS_ATTR_PAGE_SIZE_BIG     LWOS32_ATTR_PAGE_SIZE_BIG
#define LW0041_CTRL_SURFACE_INFO_PHYS_ATTR_PAGE_SIZE_HUGE    LWOS32_ATTR_PAGE_SIZE_HUGE

/*
 * LW0041_CTRL_CMD_GET_SURFACE_INFO
 *
 * This command returns surface information for the associated memory object.
 * Requests to retrieve surface information use a list of one or more
 * LW0041_CTRL_SURFACE_INFO structures.
 *
 *   surfaceInfoListSize
 *     This field specifies the number of entries on the caller's
 *     surfaceInfoList.
 *   surfaceInfoList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer into which the surface information is to be returned.
 *     This buffer must be at least as big as surfaceInfoListSize multiplied
 *     by the size of the LW0041_CTRL_SURFACE_INFO structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW0041_CTRL_CMD_GET_SURFACE_INFO                     (0x410110) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_GET_SURFACE_INFO_PARAMS_MESSAGE_ID" */

#define LW0041_CTRL_GET_SURFACE_INFO_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW0041_CTRL_GET_SURFACE_INFO_PARAMS {
    LwU32 surfaceInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 surfaceInfoList, 8);
} LW0041_CTRL_GET_SURFACE_INFO_PARAMS;

/*
 * LW0041_CTRL_CMD_GET_SURFACE_COMPRESSION_COVERAGE
 *
 * This command returns the percentage of surface compression tag coverage.
 * The value of 0 is returned if there are no tags associated.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_BAD_OBJECT_HANDLE
 *   LWOS_STATUS_BAD_OBJECT_PARENT
 *   LWOS_STATUS_NOT_SUPPORTED
 *
 */
#define LW0041_CTRL_CMD_GET_SURFACE_COMPRESSION_COVERAGE (0x410112) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_GET_SURFACE_COMPRESSION_COVERAGE_PARAMS_MESSAGE_ID" */

#define LW0041_CTRL_GET_SURFACE_COMPRESSION_COVERAGE_PARAMS_MESSAGE_ID (0x12U)

typedef struct LW0041_CTRL_GET_SURFACE_COMPRESSION_COVERAGE_PARAMS {
    LwHandle hSubDevice; /* if non zero subDevice handle of local GPU */
    LwU32    lineMin;
    LwU32    lineMax;
    LwU32    format;
} LW0041_CTRL_GET_SURFACE_COMPRESSION_COVERAGE_PARAMS;

/*
 * LW0041_CTRL_CMD_GET_FBMEM_BUS_ADDR
 *
 * This command returns the BAR1 physical address of a
 * Memory mapping made using LwRmMapMemory()
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_ILWALID_DATA
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *
 */
#define LW0041_CTRL_CMD_GET_FBMEM_BUS_ADDR (0x410114) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_GET_FBMEM_BUS_ADDR_PARAMS_MESSAGE_ID" */

#define LW0041_CTRL_GET_FBMEM_BUS_ADDR_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW0041_CTRL_GET_FBMEM_BUS_ADDR_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pLinearAddress, 8);  /* [in] Linear address of CPU mapping */
    LW_DECLARE_ALIGNED(LwU64 busAddress, 8);  /* [out] BAR1 address */
} LW0041_CTRL_GET_FBMEM_BUS_ADDR_PARAMS;

/*
 * LW0041_CTRL_CMD_SURFACE_FLUSH_GPU_CACHE
 * 
 * This command flushes a cache on the GPU which all memory accesses go
 * through.  The types of flushes supported by this API may not be supported by
 * all hardware.  Attempting an unsupported flush type will result in an error.
 *
 *   flags
 *     Contains flags to control various aspects of the flush.  Valid values
 *     are defined in LW0041_CTRL_SURFACE_FLUSH_GPU_CACHE_FLAGS*.  Not all
 *     flags are valid for all GPUs.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LWOS_STATUS_ILWALID_ARGUMENT
 *   LWOS_STATUS_ILWALID_STATE
 *
 * See Also:
 *   LW0080_CTRL_CMD_DMA_FLUSH 
 *     Performs flush operations in broadcast for the GPU cache and other hardware
 *     engines.  Use this call if you want to flush all GPU caches in a
 *     broadcast device.
 *   LW2080_CTRL_CMD_FB_FLUSH_GPU_CACHE
 *     Flushes the entire GPU cache or a set of physical addresses (if the
 *     hardware supports it).  Use this call if you want to flush a set of
 *     addresses or the entire GPU cache in unicast mode.
 *
 */
#define LW0041_CTRL_CMD_SURFACE_FLUSH_GPU_CACHE (0x410116) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_SURFACE_FLUSH_GPU_CACHE_PARAMS_MESSAGE_ID" */

#define LW0041_CTRL_SURFACE_FLUSH_GPU_CACHE_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW0041_CTRL_SURFACE_FLUSH_GPU_CACHE_PARAMS {
    LwU32 flags;
} LW0041_CTRL_SURFACE_FLUSH_GPU_CACHE_PARAMS;

#define LW0041_CTRL_SURFACE_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK                0:0
#define LW0041_CTRL_SURFACE_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK_NO  (0x00000000)
#define LW0041_CTRL_SURFACE_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK_YES (0x00000001)
#define LW0041_CTRL_SURFACE_FLUSH_GPU_CACHE_FLAGS_ILWALIDATE                1:1
#define LW0041_CTRL_SURFACE_FLUSH_GPU_CACHE_FLAGS_ILWALIDATE_NO  (0x00000000)
#define LW0041_CTRL_SURFACE_FLUSH_GPU_CACHE_FLAGS_ILWALIDATE_YES (0x00000001)

/*
 * LW0041_CTRL_CMD_GET_EME_PAGE_SIZE
 *
 * This command may be used to get the memory page size 
 *
 * Parameters:
 *   pageSize [OUT]
 *     pageSize with associated memory descriptor
 *
 * Possible status values are:
 *   LW_OK
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0041_CTRL_CMD_GET_MEM_PAGE_SIZE                        (0x410118) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_GET_MEM_PAGE_SIZE_PARAMS_MESSAGE_ID" */

#define LW0041_CTRL_GET_MEM_PAGE_SIZE_PARAMS_MESSAGE_ID (0x18U)

typedef struct LW0041_CTRL_GET_MEM_PAGE_SIZE_PARAMS {
    LwU32 pageSize;             /* [out] - page size */
} LW0041_CTRL_GET_MEM_PAGE_SIZE_PARAMS;

/*
 * LW0041_CTRL_CMD_UPDATE_SURFACE_COMPRESSION
 *
 * Acquire/release compression for surface
 *
 * Parameters:
 *   bRelease [IN]
 *     true = release compression; false = acquire compression
 */
#define LW0041_CTRL_CMD_UPDATE_SURFACE_COMPRESSION (0x410119) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_UPDATE_SURFACE_COMPRESSION_PARAMS_MESSAGE_ID" */

#define LW0041_CTRL_UPDATE_SURFACE_COMPRESSION_PARAMS_MESSAGE_ID (0x19U)

typedef struct LW0041_CTRL_UPDATE_SURFACE_COMPRESSION_PARAMS {
    LwBool bRelease;             /* [in] - acquire/release setting */
} LW0041_CTRL_UPDATE_SURFACE_COMPRESSION_PARAMS;

#define LW0041_CTRL_CMD_PRINT_LABELS_PARAMS_MESSAGE_ID (0x50U)

typedef struct LW0041_CTRL_CMD_PRINT_LABELS_PARAMS {
    LwU32 tag; /* [in] */
} LW0041_CTRL_CMD_PRINT_LABELS_PARAMS;
#define LW0041_CTRL_CMD_SET_LABEL_PARAMS_MESSAGE_ID (0x51U)

typedef struct LW0041_CTRL_CMD_SET_LABEL_PARAMS {
    LwU32 tag; /* [in] */
} LW0041_CTRL_CMD_SET_LABEL_PARAMS;
#define LW0041_CTRL_CMD_SET_LABEL (0x410151) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_CMD_SET_LABEL_PARAMS_MESSAGE_ID" */
#define LW0041_CTRL_CMD_GET_LABEL (0x410152) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_CMD_GET_LABEL_PARAMS_MESSAGE_ID" */
#define LW0041_CTRL_CMD_GET_LABEL_PARAMS_MESSAGE_ID (0x52U)

typedef struct LW0041_CTRL_CMD_GET_LABEL_PARAMS {
    LwU32 tag; /* [in] */
} LW0041_CTRL_CMD_GET_LABEL_PARAMS;

/*
 * LW0041_CTRL_CMD_SET_TAG
 *
 * This command sets memory allocation tag used for debugging.
* Every client has it's own memory allocation tag and tag is copying when object is duping.
 * This control can be used for shared allocations to change it's tag.
 */
#define LW0041_CTRL_CMD_SET_TAG (0x410120) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_CMD_SET_TAG_PARAMS_MESSAGE_ID" */

#define LW0041_CTRL_CMD_SET_TAG_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW0041_CTRL_CMD_SET_TAG_PARAMS {
    LwU32 tag; /* [in] */
} LW0041_CTRL_CMD_SET_TAG_PARAMS;

/*
 * LW0041_CTRL_CMD_GET_TAG
 *
 * This command returns memory allocation tag used for debugging.
 */
#define LW0041_CTRL_CMD_GET_TAG (0x410121) /* finn: Evaluated from "(FINN_LW01_ROOT_USER_MEMORY_INTERFACE_ID << 8) | LW0041_CTRL_CMD_GET_TAG_PARAMS_MESSAGE_ID" */

#define LW0041_CTRL_CMD_GET_TAG_PARAMS_MESSAGE_ID (0x21U)

typedef struct LW0041_CTRL_CMD_GET_TAG_PARAMS {
    LwU32 tag; /* [out] */
} LW0041_CTRL_CMD_GET_TAG_PARAMS;

/* _ctrl0041_h_ */
