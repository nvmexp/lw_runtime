/*
 * Copyright (c) 2004-2015, LWPU CORPORATION. All rights reserved.
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
// Source file: ctrl/ctrl003e.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW01_MEMORY_SYSTEM control commands and parameters */

#define LW003E_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0x003E, LW003E_CTRL_##cat, idx)

/* LW01_MEMORY_SYSTEM command categories (6bits) */
#define LW003E_CTRL_RESERVED (0x00)
#define LW003E_CTRL_MEMORY   (0x01)

/*
 * LW003E_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW003E_CTRL_CMD_NULL (0x3e0000) /* finn: Evaluated from "(FINN_LW01_MEMORY_SYSTEM_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW003E_CTRL_CMD_GET_SURFACE_PHYS_ATTR
 *
 * This command returns attributes associated with the memory object
 * at the given offset. The architecture dependent return parameter
 * comprFormat determines the meaningfulness (or not) of comprOffset.
 *
 * This call is lwrrently only supported in the MODS environment.
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
 *   gpuCacheAttr
 *     gpuCacheAttr returns the gpu cache attribute of the surface.
 *     Legal return values for this field are
 *       LW003E_CTRL_GET_SURFACE_PHYS_ATTR_GPU_CACHED_UNKNOWN
 *       LW003E_CTRL_GET_SURFACE_PHYS_ATTR_GPU_CACHED
 *       LW003E_CTRL_GET_SURFACE_PHYS_ATTR_GPU_UNCACHED
 *   gpuP2PCacheAttr
 *     gpuP2PCacheAttr returns the gpu peer-to-peer cache attribute of the surface.
 *     Legal return values for this field are
 *       LW003E_CTRL_GET_SURFACE_PHYS_ATTR_GPU_CACHED_UNKNOWN
 *       LW003E_CTRL_GET_SURFACE_PHYS_ATTR_GPU_CACHED
 *       LW003E_CTRL_GET_SURFACE_PHYS_ATTR_GPU_UNCACHED
 *   mmuContext
 *     mmuContext returns the requested type of physical address
 *     Legal return values for this field are
 *       TEGRA_VASPACE_A  --  return the non-GPU device physical address ( the system physical address itself) for CheetAh engines.
 *                            returns the system physical address, may change to use a class value in future.
 *       FERMI_VASPACE_A  --  return the GPU device physical address( the system physical address, or the SMMU VA) for Big GPU engines.
 *                     0  --  return the GPU device physical address( the system physical address, or the SMMU VA) for Big GPU engines.
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
#define LW003E_CTRL_CMD_GET_SURFACE_PHYS_ATTR (0x3e0101) /* finn: Evaluated from "(FINN_LW01_MEMORY_SYSTEM_MEMORY_INTERFACE_ID << 8) | LW003E_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS_MESSAGE_ID" */

#define LW003E_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW003E_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 memOffset, 8);
    LwU32 memFormat;
    LwU32 comprOffset;
    LwU32 comprFormat;
    LwU32 gpuCacheAttr;
    LwU32 gpuP2PCacheAttr;
    LwU32 mmuContext;
    LW_DECLARE_ALIGNED(LwU64 contigSegmentSize, 8);
} LW003E_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS;

/* valid gpuCacheAttr return values */
#define LW003E_CTRL_GET_SURFACE_PHYS_ATTR_GPU_CACHED_UNKNOWN (0x00000000)
#define LW003E_CTRL_GET_SURFACE_PHYS_ATTR_GPU_CACHED         (0x00000001)
#define LW003E_CTRL_GET_SURFACE_PHYS_ATTR_GPU_UNCACHED       (0x00000002)

/* LW003E_CTRL_CMD_GET_SURFACE_NUM_PHYS_PAGES
 *
 * This command returns the number of physical pages associated with the
 * memory object.
 *
 * This call is lwrrently only implemented on Linux and assumes that linux
 * kernel in which RM module will be loaded has same page size as defined
 * in linux kernel source with which RM module was built.
 *
 *   numPages
 *     This parameter returns total number of physical pages associated with
 *     the memory object.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW003E_CTRL_CMD_GET_SURFACE_NUM_PHYS_PAGES           (0x3e0102) /* finn: Evaluated from "(FINN_LW01_MEMORY_SYSTEM_MEMORY_INTERFACE_ID << 8) | LW003E_CTRL_GET_SURFACE_NUM_PHYS_PAGES_PARAMS_MESSAGE_ID" */

#define LW003E_CTRL_GET_SURFACE_NUM_PHYS_PAGES_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW003E_CTRL_GET_SURFACE_NUM_PHYS_PAGES_PARAMS {
    LwU32 numPages;
} LW003E_CTRL_GET_SURFACE_NUM_PHYS_PAGES_PARAMS;


/* LW003E_CTRL_CMD_GET_SURFACE_PHYS_PAGES
 *
 * This command returns physical pages associated with the memory object.
 *
 * This call is lwrrently only implemented on Linux and assumes that linux
 * kernel in which RM module will be loaded has same page size as defined
 * in linux kernel source with which RM module was built.
 *
 *   pPages
 *     This parameter returns physical pages associated with the memory object.
 *
 *   numPages
 *     This parameter is both an input and an output. As an input parameter,
 *     it's value indicates maximum number of physical pages to be copied to
 *     pPages. As an output parameter, it's value indicates number of physical
 *     pages copied to pPages.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW003E_CTRL_CMD_GET_SURFACE_PHYS_PAGES (0x3e0103) /* finn: Evaluated from "(FINN_LW01_MEMORY_SYSTEM_MEMORY_INTERFACE_ID << 8) | LW003E_CTRL_GET_SURFACE_PHYS_PAGES_PARAMS_MESSAGE_ID" */

#define LW003E_CTRL_GET_SURFACE_PHYS_PAGES_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW003E_CTRL_GET_SURFACE_PHYS_PAGES_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pPages, 8);
    LwU32 numPages;
} LW003E_CTRL_GET_SURFACE_PHYS_PAGES_PARAMS;

/* _ctrl003e_h_ */
