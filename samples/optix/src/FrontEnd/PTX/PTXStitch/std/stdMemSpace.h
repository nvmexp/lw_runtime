/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2007-2021, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdMemSpace.h
 *
 *  Description              :
 *     
 */

#ifndef stdMemSpace_INCLUDED
#define stdMemSpace_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdTypes.h"
//#include "stdWriter.h" avoid include cycle via stdLocal

#ifdef __cplusplus
extern "C" {
#endif

/*---------------------------------- Types -----------------------------------*/

/*
 * Memory space type, plus a definition of a memory space
 * that allocates from the native memory manager (malloc,free).
 * this global memory space can never be deleted:
 */

typedef struct stdMemSpaceRec *stdMemSpace_t;

#define memspNativeMemSpace     Nil
#define memspDEFAULT_PAGESIZE   0x10000

/*--------------------------------- Functions --------------------------------*/


/*
 * Function        : Allocate memory block from memory space
 * Parameters      : space  (I) Space to allocate from.
 *                   size   (I) Size to allocate.
 * Function Result : Requested memory block, or raise an out of memory
 *                   exception when memory allocation failed.
 */
Pointer STD_CDECL memspMalloc(stdMemSpace_t space, SizeT size);



/*
 * Function        : Reallocate memory block previously allocated 
 *                   via memspMalloc to specified size
 *                   NB: Nil is *not* a legal memory block argument 
 *                       to this function.
 * Parameters      : block  (I) Memory block to reallocate.
 *                   size   (I) New size of memory block
 * Function Result : Requested memory block, or raise an out of memory
 *                   exception when memory allocation failed.
 */
Pointer STD_CDECL memspRealloc(Pointer block, SizeT size);



/*
 * Function        : Free memory block previously allocated via memspMalloc
 *                   NB: Nil is a legal memory block argument to this function,
 *                       in which case it behaves as a nop
 * Parameters      : block  (I) Memory block to free, or Nil
 * Function Result : 
 */
void STD_CDECL memspFree(Pointer block);



/*
 * Function        : Obtain memory space that holds the specified address
 *                   NB: the specified argument may be any address within
 *                       a block returned by memspMalloc or memspRealloc
 *                       (and which has not yet been freed).
 * Parameters      : address  (I) address value to colwert
 * Function Result : the memory space from which the address has been allocated
 */
stdMemSpace_t STD_CDECL memspGetSpace(Pointer address);



/*
 * Function        : Allocate new memory space
 * Parameters      : name     (I) Name of memory space for use in memspPrint, or Nil.
 *                                This name will be copied to be part of the
 *                                created memory space.
 *                   parent   (I) memory space that will provide all memory
 *                                for the implementation of the allocated 
 *                                memory space. A memspDelete on this parent
 *                                will implicitly perform a memspDelete on
 *                                the resulting memory space.
 *                   pageSize (I) Increment size of memory space.
 *                                When this value is equal to 0, then the pageSize
 *                                of the parent mempool will be taken, 
 *                                or memspDEFAULT_PAGESIZE when the parent is Nil.
 * Function Result : New memory space
 */
stdMemSpace_t STD_CDECL memspCreate( cString name, stdMemSpace_t parent, SizeT pageSize );



/*
 * Function        : Free memory space, plus optionally all live memory  
 *                   blocks that were allocated from it, plus
 *                   relwrsively delete all of its child spaces.
 * Parameters      : space       (I) Memory space to free
 *                   addToParent (I) If this parameter is True,
 *                                   then then the contents of the child
 *                                   spaces will be moved to the parent of
 *                                   the memory space to be deleted (and hence remain
 *                                   live). Otherwise these contents will be
 *                                   freed as well. 
 *                                   NB : This parameter cannot have value True if
 *                                        the parent of the specified memspace is
 *                                        memspNativeMemSpace.
 * Function Result : 
 */
void STD_CDECL memspDelete( stdMemSpace_t space, Bool addToParent );



/*
 * Function        : Swap larga and small block tune counts 
 *                   for specified memory space with specified values.
 * Parameters      : space       (I) Memory space to modify.
 *                   lbTuneCount (I) Pointer to value to swap large block
 *                                   tune count with, or Nil..
 *                   sbTuneCount (I) Pointer to value to swap small block
 *                                   tune count with, or Nil..
 * Function Result : New memory space
 */
void STD_CDECL memspSwapTuneCount( stdMemSpace_t space, uInt *lbTuneCount, uInt *sbTuneCount );



/*
 * Function        : Print memory space statistics via writer object.
 * Parameters      : wr            (I) Writer to print to
 *                   space         (I) Memory space to print
 *                   cleanup       (I) Cleanup the internal administration
 *                                     before printing, so that e.g. no merges
 *                                     of adjacent blocks are pending
 *                   printLevel    (I) Required detail of printed information
 *                   printChildren (I) Relwrsively print the children of this memory space
 *                   indent        (I) Indent child statistics
 * Function Result : 
 */
 
   typedef enum { memspPrintSummary, memspPrintBrief, memspPrintFull } memspPrintLevel;
 
void STD_CDECL memspPrint( /*stdWriter_t*/ Pointer wr, stdMemSpace_t space, Bool cleanup, memspPrintLevel level, Bool printChildren, Bool indent );


/*
 * Function        : Hash a pointer value. If the specified pointer is an address within a memory
 *                   block that has been allocated on a memory space controlled by this module, 
 *                   then the result is a deterministic hash. That is, contrary to the results 
 *                   of e.g. malloc on Linux, the hash value will be the same over different runs 
 *                   of the same application with the same input. If the specified pointer is outside
 *                   any memory space, then the returned hash value of 'the same' pointer may differ
 *                   over different runs.
 * Parameters      : address       (I) Pointer to hash
 * Function Result : 
 */
uInt STD_CDECL memspPointerHash( Pointer address );


/*
 * Function        : Compare two pointer values. If the specified pointers are addresses within
 *                   memory blocks that have been allocated on memory spaces controlled by this module, 
 *                   then the result is a comparison based on a deterministic total memory address order.
 *                   That is, contrary to the results of e.g. malloc on Linux, the result will be the same 
 *                   over different runs of the same application with the same input. If either 
 *                   or both of the compared addresses are outside any memory space, then the result of
 *                  'the same' comparison may differ over different runs.
 * Parameters      : l,r          (I) Pointers to compare
 * Function Result : True iff. l less than or equal r
 */
uInt STD_CDECL memspPointerLE( Pointer l, Pointer r );


/*------------------------------ Debug Functions -----------------------------*/

/*
 * Function        : Enable global debugging mode. 
 *                   In this mode, memory spaces will not actually be deleted at memspDelete,
 *                   but left alive with a marker set. Function memspCheckBlock can then be used
 *                   to check whether the corresponding memory space is still alive.
 * Parameters      : set           (I) When true, global debugging will be enabled. 
 *                                     Otherwise it will be disabled.
 * Function Result : 
 */
void STD_CDECL memspSetGlobalDebug( Bool set );



/*
 * Function        : Check whether the block's memory pool is still alive,
 *                   and complain via stdSYSLOG if this is not the case.
 * Parameters      : block         (I) Block to check
 * Function Result : 
 */
void STD_CDECL memspCheckBlock( Pointer block );


/*
 * Function        : Print an identification of the specified memory block via writer object.
 * Parameters      : wr            (I) Writer to print to
 *                   block         (I) Block to describe
 * Function Result : 
 */
void STD_CDECL memspDescribeBlock( /*stdWriter_t*/ Pointer wr, Pointer block );

/*------------------------------ Fuzzy Functions -----------------------------*/

// Don't use this
/*static*/ void STD_CDECL stdMemSpaceManagerCleanup(void);


#ifdef __cplusplus
}
#endif

#endif
