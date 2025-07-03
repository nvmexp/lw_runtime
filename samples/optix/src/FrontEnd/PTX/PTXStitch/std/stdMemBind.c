/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2015-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : stdMemBind.h
 *
 *  Description              :
 *
 *               This module defines the lowest level memory manage,
 *               on top of which stdMemSpace is built.
 */

/*--------------------------------- Includes ---------------------------------*/

#include "stdMemBind.h"
#include "stdLocal.h"

/*-------------------------------- Debugging ---------------------------------*/

#if defined(MEMORY_DEBUG)

    #define CRAZYPOOL_SIZE   0x100000

    static Address  crazyPool[CRAZYPOOL_SIZE];
    static Address *crazyPtr = crazyPool;

    static Pointer mb_malloc( SizeT size )
    {
        Pointer result  = Nil;
        SizeT   nrofWords = (stdROUNDUP64(size,stdALIGNMENT)) / sizeof(Address);

        if (crazyPtr == crazyPool) {
            String skip= getelw("MEMORY_DEBUG_SKIP");

            if (skip) {
                crazyPtr += atoi(skip)*stdALIGNMENT;
            }
        }

        if ( (crazyPtr+nrofWords+1) <= &crazyPool[CRAZYPOOL_SIZE]) {
           *crazyPtr  = size;
            result    = crazyPtr  + 1;
            crazyPtr += nrofWords + 1;
        }

        return result;
    }


    static Pointer mb_realloc( Pointer b, SizeT newSize )
    {
        Address oldSize = ((Address*)b)[-1];

        if (oldSize >= newSize) {
            return b;
        } else {
            Byte* result = mb_malloc(newSize);

            stdMEMCOPY_N(result,b,oldSize);

            return result;
        }
    }

    #define mb_free(b)

/*--------------------------- Windows Heap Manager ---------------------------*/

#elif defined(STD_OS_win32) && defined(USE_NATIVE_HEAP)

    #include <Windows.h>
    #include <Winbase.h>

    static HANDLE stdGetHeapHandle()
    {
        return stdGetThreadContext()->winHeapHandle;
    }

    static void stdSetHeapHandle(HANDLE handle)
    {
        stdGetThreadContext()->winHeapHandle = handle;
    }

    void STD_CDECL allocateHeap()
    {
        HANDLE lwrHandle = HeapCreate(0, 1024 * 1024, 0);
        stdSetHeapHandle(lwrHandle);
    }

    void STD_CDECL deallocateHeap()
    {
        if(stdGetHeapHandle()) {
            HeapDestroy(stdGetHeapHandle());
            stdSetHeapHandle(0);
        }
    }

    HANDLE STD_CDECL resetHeap(HANDLE newHeapHandle)
    {
        HANDLE lwrHandle = stdGetHeapHandle();
        stdSetHeapHandle(newHeapHandle);
        return lwrHandle;
    }

    static Pointer mb_malloc(SizeT size)
    {
        if (stdGetHeapHandle()) {
            return HeapAlloc(stdGetHeapHandle(), 0, size);
        } else {
            return malloc(size);
        }
    }

    static Pointer mb_realloc(Pointer block, SizeT size)
    {
        if (stdGetHeapHandle()) {
            return HeapReAlloc(stdGetHeapHandle(), 0, block, size);
        } else {
            return realloc(block,size);
        }
    }

    static void mb_fFree(Pointer block)
    {
        if (stdGetHeapHandle()) {
            HeapFree(stdGetHeapHandle(), 0, block);
        } else {
            free(block);
        }
    }

/*--------------------------------- Default ----------------------------------*/

#else
    #define mb_malloc(s)    malloc(s)
    #define mb_realloc(b,s) realloc(b,s)
    #define mb_free(b)      free(b)
#endif

/*--------------------------------- Functions --------------------------------*/

#ifdef ALIGNED_ALLOCATIONS
    #define BLOCK_HEADER_SIZE stdROUNDUP(sizeof(SizeT), stdALIGNMENT)
#else
    #define BLOCK_HEADER_SIZE sizeof(SizeT)
#endif

/*
 * Function        : Allocate memory block from memory space
 * Parameters      : size    (I) Size to allocate.
 *                   reserve (I) The allocated buffer is a block that is
 *                               kept as reserve by the client of this module,
 *                               to be given back in case of memory shortage
 *                               so that a limited amount of further calls
 *                               to mbMalloc will succeed.
 * Function Result : Requested memory block, or Nil
 */
Pointer STD_CDECL mbMalloc(SizeT size, Bool reserve )
{
#ifdef MEMORY_COUNT
    SizeT   totalIncrease = size + BLOCK_HEADER_SIZE;
    Pointer block         = malloc(totalIncrease);
    if (block) {
        if (!reserve) { increaseMemory(totalIncrease); }

        *(SizeT*)block = totalIncrease;
        block = (Byte*)block + BLOCK_HEADER_SIZE;
    }
    return block;
#else
    return malloc(size);
#endif
}

/*
 * Function        : Reallocate memory block previously allocated
 *                   via mbMalloc to specified size
 * Parameters      : block  (I) Memory block to reallocate.
 *                   size   (I) New size of memory block
 * Function Result : Requested memory block, or Nil
 */
Pointer STD_CDECL mbRealloc(Pointer block, SizeT size)
{
#ifdef MEMORY_COUNT
    SizeT oldSize;
    SizeT totalIncrease = size + BLOCK_HEADER_SIZE;
    block               = (Byte*)block - BLOCK_HEADER_SIZE;
    oldSize             = *(SizeT*)block;

    block = realloc(block,totalIncrease);

    if (block) {
        if (oldSize > size) {
            decreaseMemory(oldSize - totalIncrease);
        } else {
            increaseMemory(totalIncrease - oldSize);
        }

        *((SizeT*)block) = totalIncrease;
        block = (Byte*)block + BLOCK_HEADER_SIZE;
    }
    return block;
#else
    return realloc(block,size);
#endif
}

/*
 * Function        : Free memory block previously allocated via mbMalloc
 *                   NB: Nil is a legal memory block argument to this function
 * Parameters      : block   (I) Memory block to free, or Nil
 *                   reserve (I) The freed block was kept as reserve
 *                               by the client of this module, and is now
 *                               thrown back in case of memory shortage.
 */
void STD_CDECL mbFree(Pointer block, Bool reserve)
{
#ifdef MEMORY_COUNT
    if (block) {
        block = (Byte*)block - BLOCK_HEADER_SIZE;
        if (!reserve) { decreaseMemory(*(SizeT*)block); }
        free(block);
    }
#else
    free(block);
#endif
}

#ifdef MEMORY_COUNT

static Bool measurePeakMemory;
static uInt64 peakMemory, totMemory;

/*
 * Function        : Increases the total memory by size and updates peak memory
 *                   if the trakced memory exceeds the previously tracked peak memory
 * Parameters      : size   (I) amount of memory by which total memory should be increased by
 */
void STD_CDECL increaseMemory(SizeT size)
{
    if(isMemoryStatsEnabled()) {
        totMemory += size;
        if (totMemory >= peakMemory) {
            peakMemory = totMemory;
        }
    }
}

/*
 * Function        : Decreases the total memory by size
 * Parameters      : size   (I) amount of memory by which total memory should be decreased by
 */
void STD_CDECL decreaseMemory(SizeT size)
{
    if (isMemoryStatsEnabled()) {
        totMemory -= size;
    }
}

/*
 * Function        : Enables and initializes the memory tracking counters
 *
 */
void STD_CDECL enablePerfMeasuring(void)
{
    measurePeakMemory = True;
    peakMemory = 0;
    totMemory = 0;
}

/*
 * Function        : Disables the memory tracking counters
 *
 */
void STD_CDECL disablePerfMeasuring(void)
{
    measurePeakMemory = False;
}

/*
 * Function        : Queries whether memory tracking is enabled or not
 * Function Result : True if memory tracking is enabled,
 *                   Else False
 */
Bool STD_CDECL isMemoryStatsEnabled(void)
{
    return measurePeakMemory;
}

/*
 * Function        : Query the peak memory used till now
 * Function Result : Peak amount of memory used till now in KB
 */
double STD_CDECL stdGetPeakMemUsage(void)
{
    return (double)peakMemory / (1024.0);
}

/*
 * Function        : Query the total memory used till now
 * Function Result : Total amount of memory used till now in KB
 */
double STD_CDECL stdGetMemUsage(void)
{
    return (double)totMemory / (1024.0);
}

#endif
