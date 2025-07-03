/*
 * Copyright (c) 2001-2014, LWPU CORPORATION. All rights reserved.
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

#ifndef _cl84A0_h_
#define _cl84A0_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/*
 * Class definitions for creating a memory descriptor from a list of page numbers
 * in RmAllocMemory.   No memory is allocated: only a memory descriptor and
 * memory object are created for later use in other calls.  These classes
 * are used by vGPU to create references to memory assigned to a guest VM.
 * In all cases, the list is passed as reference, in the pAddress argument
 * of RmAllocMemory, to a Lw01MemoryList structure (cast to a void **).
 */

/* List of system memory physical page numbers */
#define  LW01_MEMORY_LIST_SYSTEM                                   (0x00000081)
/* List of frame buffer physical page numbers */
#define  LW01_MEMORY_LIST_FBMEM                                    (0x00000082)
/* List of page numbers relative to the start of the specified object */
#define  LW01_MEMORY_LIST_OBJECT                                   (0x00000083)

/*
 * List structure of LW01_MEMORY_LIST_* classes 
 *
 * The pageNumber array is variable in length, with pageCount elements,
 * so the allocated size of the structure must reflect that.
 *
 * FBMEM items apply only to LW01_MEMORY_LIST_FBMEM and to
 * LW01_MEMORY_LIST_OBJECT when the underlying object is
 * FBMEM (must be zero for other cases)
 *
 * Lw01MemoryList is deprecated. LW_MEMORY_LIST_ALLOCATION_PARAMS should be used
 * instead.
 */
typedef struct Lw01MemoryListRec {
    LwHandle    hClient;    /* client to which object belongs 
                             * (may differ from client creating the mapping).
                             * May be LW01_NULL_OBJECT, in which case client
                             * handle is used */
    LwHandle    hParent;    /* device with which object is associated.
                             * Must be LW01_NULL_OBJECT if hClient is LW01_NULL_OBJECT.
                             * Must not be LW01_NULL_OBJECT if hClient is
                             * not LW01_NULL_OBJECT. */
    LwHandle    hObject;    /* object to which pages are relative
                             * (LW01_NULL_OBJECT for LW01_MEMORY_LIST_SYSTEM
                             *  and LW01_MEMORY_LIST_FBMEM) */
    LwHandle   hHwResClient;/* client associated with the backdoor vnc surface*/
    LwHandle   hHwResDevice;/* device associated to the bacdoor vnc surface*/
    LwHandle   hHwResHandle;/* handle to hardware resources allocated to 
                             * backdoor vnc surface*/
    LwU32   pteAdjust;      /* offset of data in first page */
    LwU32   type;           /* FBMEM: LWOS32_TYPE_* */
    LwU32   flags;          /* FBMEM: LWOS32_ALLOC_FLAGS_* */
    LwU32   attr;           /* FBMEM: LWOS32_ATTR_* */
    LwU32   attr2;          /* FBMEM: LWOS32_ATTR2_* */
    LwU32   height;         /* FBMEM: height in pixels */
    LwU32   width;          /* FBMEM: width in pixels */
    LwU32   format;         /* FBMEM: memory kind */
    LwU32   comprcovg;      /* FBMEM: compression coverage */
    LwU32   zlwllcovg;      /* FBMEM: Z-lwll coverage */
    LwU32   pageCount;      /* count of elements in pageNumber array */
    LwU32   heapOwner;      /* heap owner information from client */
    LwU32   reserved_1;     /* reserved: must be 0 */
    LwU64   LW_DECLARE_ALIGNED(guestId,8); 
                            /* ID of the guest VM. e.g., domain ID in case of Xen */
    LwU64   LW_DECLARE_ALIGNED(rangeBegin,8);
                            /* preferred VA range start address */
    LwU64   LW_DECLARE_ALIGNED(rangeEnd,8);
                            /* preferred VA range end address */
    LwU32   pitch;
    LwU32   ctagOffset;
    LwU64   size;
    LwU64   align;
    LwU64   pageNumber[1];  /* variable length array of page numbers */
} Lw01MemoryList;

/*
 * LW_MEMORY_LIST_ALLOCATION_PARAMS - Allocation params to create memory list
 * through LwRmAlloc.
 */
typedef struct
{
    LwHandle    hClient;    /* client to which object belongs 
                             * (may differ from client creating the mapping).
                             * May be LW01_NULL_OBJECT, in which case client
                             * handle is used */
    LwHandle    hParent;    /* device with which object is associated.
                             * Must be LW01_NULL_OBJECT if hClient is LW01_NULL_OBJECT.
                             * Must not be LW01_NULL_OBJECT if hClient is
                             * not LW01_NULL_OBJECT. */
    LwHandle    hObject;    /* object to which pages are relative
                             * (LW01_NULL_OBJECT for LW01_MEMORY_LIST_SYSTEM
                             *  and LW01_MEMORY_LIST_FBMEM) */
    LwHandle   hHwResClient;/* client associated with the backdoor vnc surface*/
    LwHandle   hHwResDevice;/* device associated to the bacdoor vnc surface*/
    LwHandle   hHwResHandle;/* handle to hardware resources allocated to 
                             * backdoor vnc surface*/
    LwU32   pteAdjust;      /* offset of data in first page */
    LwU32   reserved_0;     /* reserved: must be 0 */
    LwU32   type;           /* FBMEM: LWOS32_TYPE_* */
    LwU32   flags;          /* FBMEM: LWOS32_ALLOC_FLAGS_* */
    LwU32   attr;           /* FBMEM: LWOS32_ATTR_* */
    LwU32   attr2;          /* FBMEM: LWOS32_ATTR2_* */
    LwU32   height;         /* FBMEM: height in pixels */
    LwU32   width;          /* FBMEM: width in pixels */
    LwU32   format;         /* FBMEM: memory kind */
    LwU32   comprcovg;      /* FBMEM: compression coverage */
    LwU32   zlwllcovg;      /* FBMEM: Z-lwll coverage */
    LwU32   pageCount;      /* count of elements in pageNumber array */
    LwU32   heapOwner;      /* heap owner information from client */

    LwU64   LW_DECLARE_ALIGNED(guestId,8); 
                            /* ID of the guest VM. e.g., domain ID in case of Xen */
    LwU64   LW_DECLARE_ALIGNED(rangeBegin,8);
                            /* preferred VA range start address */
    LwU64   LW_DECLARE_ALIGNED(rangeEnd,8);
                            /* preferred VA range end address */
    LwU32   pitch;
    LwU32   ctagOffset;
    LwU64   size;
    LwU64   align;
    LwP64   pageNumberList LW_ALIGN_BYTES(8);
    LwU64   limit LW_ALIGN_BYTES(8);
    LwU32   flagsOs02;
} LW_MEMORY_LIST_ALLOCATION_PARAMS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl84A0_h_ */
