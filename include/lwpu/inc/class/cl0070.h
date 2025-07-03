/*
 * Copyright (c) 2001-2019, LWPU CORPORATION. All rights reserved.
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

#ifndef _cl0070_h_
#define _cl0070_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define   LW01_MEMORY_VIRTUAL                                      (0x00000070)
#define   LW01_MEMORY_SYSTEM_DYNAMIC                               (0x00000070)

/*
 * LW_MEMORY_VIRTUAL_ALLOCATION_PARAMS
 *
 * Allocation params for LW01_MEMORY_VIRTUAL.
 *
 * LW01_MEMORY_SYSTEM_DYNAMIC is an alias for LW01_MEMORY_VIRTUAL.  This
 * was traditionally allocated with RmAllocMemory64(). The default GPU
 * virtual address space is used, and the limit of this address space is
 * returned in limit.  The LW01_MEMORY_SYSTEM_DYNAMIC handle can be
 * passed to RmAllocContextDma2() with an offset/limit.  The context dma
 * handle can then be used as the hDma handle for RmMapMemoryDma.
 *
 * This behavior is maintained in the RM compatibility shim.
 *
 * LW01_MEMORY_VIRTUAL replaces this behavior with a single object.
 *
 * hVASpace - if hVASpace is LW01_NULL_OBJECT the default GPU VA space is
 *      selected.  Alternatively a FERMI_VASPACE_A handle may be specified.
 *
 *      The LW_MEMORY_VIRTUAL_SYSMEM_DYNAMIC_HVASPACE is used for by the
 *      compatibility layer to emulate LW01_MEMORY_SYSTEM_DYNAMIC semantics.
 *
 * offset - An offset into the virtual address space may be specified.  This
 *      will limit range of the GPU VA returned by RmMapMemoryDma to be
 *      above offset.
 *
 * limit - When limit is zero the maximum limit used.  If a non-zero limit
 *      is specified then it will be used.  The final limit is returned.
 */
typedef struct
{
    LwU64     offset     LW_ALIGN_BYTES(8); // [IN] - offset into address space
    LwU64     limit      LW_ALIGN_BYTES(8); // [IN/OUT] - limit of address space
    LwHandle  hVASpace;                     // [IN] - Address space handle
} LW_MEMORY_VIRTUAL_ALLOCATION_PARAMS;

#define LW_MEMORY_VIRTUAL_SYSMEM_DYNAMIC_HVASPACE       (0xffffffffu)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl0070_h_ */
