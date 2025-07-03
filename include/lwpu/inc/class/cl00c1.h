/*
 * Copyright (c) 2018-2018, LWPU CORPORATION. All rights reserved.
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

#ifndef _cl00c1_h_
#define _cl00c1_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"
#include "lwlimits.h"

#define   LW_FB_SEGMENT                         (0x000000C1)

/*
 * LW_FB_SEGMENT_ALLOCATION_PARAMS - Allocation params to create FB segment through
 * LwRmAlloc.
 */
typedef struct
{
    LwHandle    hCtxDma;
    LwU32       subDeviceIDMask LW_ALIGN_BYTES(8);
    LwU64       dmaOffset       LW_ALIGN_BYTES(8);
    LwU64       VidOffset       LW_ALIGN_BYTES(8);
    LwU64       Offset          LW_ALIGN_BYTES(8); // To be deprecated
    LwU64       pOffset[LW_MAX_SUBDEVICES] LW_ALIGN_BYTES(8);
    LwU64       Length          LW_ALIGN_BYTES(8);
    LwU64       ValidLength     LW_ALIGN_BYTES(8);
    LwP64       pPageArray      LW_ALIGN_BYTES(8);
    LwU32       startPageIndex;
    LwHandle    AllocHintHandle;
    LwU32       Flags;
    LwHandle    hMemory; // Not used in LwRmAlloc path; only used in CTRL path
    LwHandle    hClient; // Not used in LwRmAlloc path; only used in CTRL path
    LwHandle    hDevice; // Not used in LwRmAlloc path; only used in CTRL path
    LwP64       pCpuAddress     LW_ALIGN_BYTES(8); // To be deprecated
    LwP64       ppCpuAddress[LW_MAX_SUBDEVICES] LW_ALIGN_BYTES(8);
    LwU64       GpuAddress      LW_ALIGN_BYTES(8); // To be deprecated
    LwU64       pGpuAddress[LW_MAX_SUBDEVICES] LW_ALIGN_BYTES(8);
    LwHandle    hAllocHintClient;
    LwU32       kind;
    LwU32       compTag;
} LW_FB_SEGMENT_ALLOCATION_PARAMS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl00c1_h_ */
