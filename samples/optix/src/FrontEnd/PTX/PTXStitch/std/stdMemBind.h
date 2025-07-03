/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2015-2019, LWPU CORPORATION.  All rights reserved.
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

#ifndef stdMemBind_INCLUDED
#define stdMemBind_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Functions --------------------------------*/

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
Pointer STD_CDECL mbMalloc(SizeT size, Bool reserve);

/*
 * Function        : Reallocate memory block previously allocated
 *                   via mbMalloc to specified size
 * Parameters      : block  (I) Memory block to reallocate.
 *                   size   (I) New size of memory block
 * Function Result : Requested memory block, or Nil
 */
Pointer STD_CDECL mbRealloc(Pointer block, SizeT size);

/*
 * Function        : Free memory block previously allocated via mbMalloc
 *                   NB: Nil is a legal memory block argument to this function
 * Parameters      : block   (I) Memory block to free, or Nil
 *                   reserve (I) The freed block was kept as reserve
 *                               by the client of this module, and is now
 *                               thrown back in case of memory shortage.
 */
void STD_CDECL mbFree(Pointer block, Bool reserve);

#ifdef MEMORY_COUNT
/*
 * Function        : Increases the total memory by size and updates peak memory
 *                   if the trakced memory exceeds the previously tracked peak memory
 * Parameters      : size   (I) amount of memory by which total memory should be increased by
 */
void STD_CDECL increaseMemory(SizeT size);

/*
 * Function        : Decreases the total memory by size
 * Parameters      : size   (I) amount of memory by which total memory should be decreased by
 */
void STD_CDECL decreaseMemory(SizeT size);

/*
 * Function        : Enables and initializes the memory tracking counters
 *
 */
void STD_CDECL enablePerfMeasuring(void);

/*
 * Function        : Disables the memory tracking counters
 *
 */
void STD_CDECL disablePerfMeasuring(void);

/*
 * Function        : Queries whether memory tracking is enabled or not
 * Function Result : True if memory tracking is enabled,
 *                   Else False
 */
Bool STD_CDECL isMemoryStatsEnabled(void);

/*
 * Function        : Query the peak memory used till now
 * Function Result : Peak amount of memory used till now in KB
 */
double STD_CDECL stdGetPeakMemUsage(void);


/*
 * Function        : Query the total memory used till now
 * Function Result : Total amount of memory used till now in KB
 */
double STD_CDECL stdGetMemUsage(void);

#endif

#ifdef __cplusplus
}
#endif

#endif
