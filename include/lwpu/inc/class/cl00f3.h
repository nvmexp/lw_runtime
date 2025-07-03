/*
 * Copyright (c) 2018-2020, LWPU CORPORATION. All rights reserved.
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

#include "lwtypes.h"

/*
 * Class definition for creating a memory descriptor from a FLA range in RmAllocMemory.
 * No memory is allocated, only a memory descriptor and memory object is created
 * for later use in other calls. These classes are used by clients who tries to
 * import the memory exported by other GPU(s)/FAM/process. The range, size and
 * other parameters are passed as Lw01MemoryFla structure.
 */

#ifndef _cl00f3_h_
#define _cl00f3_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW01_MEMORY_FLA                                          (0x000000f3)

/*
 * Structure of LW_FLA_MEMORY_ALLOCATION_PARAMS
 *
 *
 */
typedef struct {
        LwU32    type;           /* FBMEM: LWOS32_TYPE_* */
        LwU32    flags;          /* FBMEM: LWOS32_ALLOC_FLAGS_* */
        LwU32    attr;           /* FBMEM: LWOS32_ATTR_* */
        LwU32    attr2;          /* FBMEM: LWOS32_ATTR2_* */
        LwU64    base;           /* base of FLA range */
        LwU64    align;          /* alignment for FLA range*/
        LwU64    limit LW_ALIGN_BYTES(8);
        //
        // For Direct connected systems, clients need to program this hSubDevice with
        // the exporting GPU, for RM to route the traffic to the destination GPU
        // Clients need not program this for LwSwitch connected systems
        //
        LwHandle hExportSubdevice; /* hSubdevice of the exporting GPU */
        //
        // Instead of base and limit, clients can also pass the FLA handle (or hExportHandle)
        // being exported from destination side to import on the access side
        //
        LwHandle hExportHandle;  /* FLA handle being exported or Export handle */
        // The RM client used to export memory
        LwHandle hExportClient;
        LwU32    flagsOs02;
} LW_FLA_MEMORY_ALLOCATION_PARAMS;

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl00f3_h

