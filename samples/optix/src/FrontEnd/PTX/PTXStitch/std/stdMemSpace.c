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
 *  Module name              : stdMemSpace.c
 *
 *  Description              :
 *
 */

/*--------------------------------- Includes ---------------------------------*/

#include <limits.h>
#include "stdSet.h"
#include "stdMap.h"
#include "stdList.h"
#include "stdMemSpace.h"
#include "stdStdFun.h"
#include "stdRangeMap.h"
#include "stdBitSet.h"
#include "stdMemBind.h"
#include "stdLocal.h"
#include "stdThreads.h"

#ifndef LWALGRIND
 #include "valgrind/valgrind.h"
#endif

#if defined(STD_OS_Hos)
#define IS_0BYTE_ALLOC_NULL
#endif

/*------------------------------- Module State -------------------------------*/

/*
 * The following is intended for guaranteeing that memspMalloc completes
 * after it is allowed to start; that is, without the danger of running halfway
 * out of memory and then leaving an inconsistent state such as a partially updated
 * memSpace administration, or with the global mutex aquired.
 *
 * Note that the memory needed for all MemSpace and module administration is
 * taken from the Nil memSpace, which directly eats from mbMalloc; memspMalloc
 * allocates one page, or one large block, and then needs a few small memory blocks
 * for updating the administration; the next #define is intended for covering those
 * 'few small blocks'.
 */

#define SAFETY_OVERHEAD_SIZE 4096

static stdMutex_t safetyValLock = Nil;
static Pointer   safetyValve = Nil;
static uInt      safetyValveSize;
static uInt      safetyValveIsSet;
static uInt      safetyValveTimeout;

static void stdMemspInitLock(stdMutex_t *lock)
{
    if (!*lock) {
        stdGlobalEnter();
        if (!*lock) {
            stdMemSpace_t savedSpace = stdSwapMemSpace( memspNativeMemSpace );
            *lock = stdMutexCreate(); 
            stdSwapMemSpace(savedSpace);
        }
        stdGlobalExit();
    } 
}

static void stdMemspInitAndAquireLock(stdMutex_t *lock)
{
    stdMemspInitLock(lock);
    stdMutexEnter(*lock);
}

static void stdMemspReleaseLock(stdMutex_t *lock)
{
    stdASSERT(*lock, ("uninitialized  mutex is released"));
    stdMutexExit(*lock);
}

static void stdMemspDeleteLock(stdMutex_t *lock)
{
    if (*lock) {
        stdGlobalEnter();
        if (*lock) {
            stdMemSpace_t savedSpace = stdSwapMemSpace( memspNativeMemSpace );
            stdMutexDelete(*lock);
            *lock = Nil;
            stdSwapMemSpace(savedSpace);
        }
        stdGlobalExit();
    } 
}

static Bool setSafetyValve( SizeT size )
{
    Bool result;
    
    stdMemspInitAndAquireLock(&safetyValLock);
    if (safetyValveIsSet++) {
       /*
        * Don't adapt the safety valve in relwrsive mallocs:
        */
        result = True;
    } else {
        size += SAFETY_OVERHEAD_SIZE;

        if ( safetyValveSize >= size
         &&  safetyValveTimeout--
           ) {
           /*
            * We are fine with the previously used valve:
            */
            result = True;
        } else {
           /*
            * Choose a larger valve:
            */
            mbFree( safetyValve, True );
            safetyValve = mbMalloc(size,True);

            if (safetyValve) {
                safetyValveTimeout = 100;
                safetyValveSize    = size;
                result = True;
            } else {
                safetyValveSize = 0;
                safetyValveIsSet--;
                result = False;
            }
        }
    }
    stdMemspReleaseLock(&safetyValLock);
    return result;
}

static void releaseSafetyValve()
{
    stdMemspInitAndAquireLock(&safetyValLock);
    safetyValveIsSet--;
    stdMemspReleaseLock(&safetyValLock);
}

static void blowSafetyValve( Bool weNeedIt )
{
    stdASSERT( stdIMPLIES(weNeedIt,safetyValve), ("SAFETY_OVERHEAD_SIZE is too small") );

    if (safetyValve) {
        stdMemspInitAndAquireLock(&safetyValLock);
        if (safetyValve) {
            mbFree( safetyValve, True );
            safetyValve     = Nil;
            safetyValveSize = 0;
        }
        stdMemspReleaseLock(&safetyValLock);
    }
}

/*--------------------------------- Constants --------------------------------*/

/*
 * Amount of changes in the large block administration
 * before the next large or small block malloc causes a retune.
 */
#define TUNE_PARAMETER        1000000

/*
 * The smallest large block size, in bytes:
 */
#define LARGE_BLOCK_THRESHOLD  stdROUNDUP(5000,stdALIGNMENT)


/*
 * Enabling the following will cause the writing of patterns
 * in malloc'd and in freed memory. This in order to better catch
 * the use of uninitialized memory contents, and the use of memory
 * blocks after they are freed.
 * This test also enables spurious-free checks on small memory blocks
 * (these checks are enabled for large blocks via macro NDEBUG):
 */
//#define MEMORY_CHECK

#define BLOCK_ALLOCATED_PATTERN   0x3
#define BLOCK_FREED_PATTERN       0xf

#define PAGE_ALLOCATED_PATTERN    0x1
#define PAGE_FREED_PATTERN        0xb


/*
 * Enabling the following will cause the lowest level of memory allocation
 * (which normally is malloc) to be performed from a global data array at a
 * fixed memory address. This will give totally deterministic memory allocation
 * over different runs of the program (as long as it is not recompiled).
 */
//#define MEMORY_DEBUG


/*
 * Enabling the following will enable code in memspMalloc and memspFree
 * to call debug function STOP when a specified address is about to be
 * returned, or freed respectively. The selected trap address must be
 * placed into global variable 'mallocTrap'/'freeTrap'.
 * By placing a breakpoint in STOP from within the debugger, it can
 * be found out who allocates/frees the selected memory block.
 *
 * Note that this functionality could also be achieved by means of
 * conditional breakpoints, but in large applications that will
 * result in an unacceptable drop of performance.
 */
//#define MALLOC_TRAP

#ifdef MALLOC_TRAP
    Pointer mallocTrap;
    Pointer freeTrap = (Pointer)(Address)-1;
#endif


/*---------------------------------- Types -----------------------------------*/

typedef struct PageHeaderRec   *PageHeader;
typedef struct LBHeaderRec     *LBHeader;
typedef struct LBPageRec       *LBPage;
typedef struct SBPageRec       *SBPage;

/*
 * Mapping from small block size (which is always a multiple of
 * stdALIGNMENT) to the index in the small block list table in
 * the memory space struct:
 */
#define SB_INDEX(size)  (((Address)(size))/stdALIGNMENT)



/*
 * Note: the first field of the following
 *       structs (LBHeaderRec and PageHeader) must
 *       remain in place, so that these
 *       can be considered 'subtypes' of stdList.
 */

struct LBHeaderRec {
    LBHeader           next;
    LBHeader           prev;
    SizeT              size;
    SizeT              prevSize;
};

typedef
struct PageHeaderRec {
    PageHeader         next;
    SizeT              freeSize;

    SizeT              size;
    stdMemSpace_t      space;
    Byte              *contents;
    Bool               sbp;
    uInt               count;
} PageHeaderRec;


/*
 * Large Block Page
 */
struct LBPageRec {
    PageHeaderRec       header;
    LBHeader            postSentinel;
    struct LBHeaderRec  freeList;
};


/*
 * Small Block Page
 */
struct SBPageRec {
    PageHeaderRec       header;
    uInt                blockSize;
  #ifdef MEMORY_CHECK
    stdBitSet_t         freeBlocks;
  #endif
};

    /* -------- . -------- */

struct stdMemSpaceRec {
    String              name;
    Bool                deleted;

    stdMemSpace_t       parent;
    stdSet_t            children;

    uInt                pageSize;

    uInt                sbTuneCount;
    uInt                lbTuneCount;

    uInt                nrofSmallPages;

   /* Large Block State */
    LBPage              lbPages;
    uInt                lbTuningSlack;
    uInt                largestBlocksIndex;
    struct LBHeaderRec  largeBlocks[ stdBITSIZEOF(SizeT) ];

   /* Small Block State */
    stdMap_t            sbPages;
    uInt64              sbTuningSlack;
    stdList_t           smallBlocks[ SB_INDEX(LARGE_BLOCK_THRESHOLD) ];
    stdMutex_t          memspLock;

};


/*------------------------------- Module State -------------------------------*/


/*
 * Query stdTreads cleanup opon main thread termination:
 */
extern Bool threadsDoCleanupOnTermination;

static Bool isCleanupHandlerSet = False;

static void stdMemspSetCleanupHandler() 
{
    if (!isCleanupHandlerSet) {
        stdSetCleanupHandler((stdDataFun) stdMemSpaceManagerCleanup, Nil);
        isCleanupHandlerSet = True;
    }
     
}
/*
 * The following variable keeps track of small/large block allocation
 * order, used in deterministic pointer hashing.
 */
static uInt blockCount;

/*
 * The following is a global lookup structure that maps
 * every memory address to the PageHeader that contains it,
 * or Nil. This is a relatively efficient way (in terms of space
 * and exelwtion time) to find the memory space to which
 * a particular memory block belongs.
 */
static stdRangeMap_t blockToSpacePage = Nil;   /* Pointer --> Page */

/*
 * Mutex to control access of blockToSpacePage map
 */
static stdMutex_t    blockToSpacePageLock = Nil;

/*
 * Global set to store all individual thread's blockToPageMap.
 * We traverse this set in stdMemSpaceManagerCleanup to delete thread local
 * map if it is not freed by corresponding thread in memspDelete
 */
static stdSet_t      threadLocalblockToPageMaps = Nil;

static stdRangeMap_t getThreadlocalBlockSpacePageRecord(void)
{
    return (stdRangeMap_t) stdGetThreadContext()->memInfo.blockToSpacePage;
}

static void STD_CDECL stdMemspDeleteThreadLocalMap(Pointer map, Pointer dummy)
{
    stdRangeMap_t threadLocalMap = (stdRangeMap_t) map;
    rangemapDelete(threadLocalMap);
}

/*static*/ void STD_CDECL stdMemSpaceManagerCleanup(void)
{
    if (threadsDoCleanupOnTermination) {
        stdRangeMap_t savedMap= blockToSpacePage;
        blockToSpacePage = Nil;

        if (savedMap) {
            rangemapDelete(savedMap);
        }

        blowSafetyValve(False);
    
        savedMap = stdGetThreadContext()->memInfo.blockToSpacePage;
        if (savedMap) {
            stdGetThreadContext()->memInfo.blockToSpacePage = Nil;
            setRemove(threadLocalblockToPageMaps, savedMap);
            rangemapDelete(savedMap);
        }

        if (threadLocalblockToPageMaps) {
            if (!setIsEmpty(threadLocalblockToPageMaps)) { 
                setTraverse(threadLocalblockToPageMaps, stdMemspDeleteThreadLocalMap,  Nil);
            }
            setDelete(threadLocalblockToPageMaps);
            threadLocalblockToPageMaps = Nil;
        }

        stdMemspDeleteLock(&safetyValLock);
        stdMemspDeleteLock(&blockToSpacePageLock);
    }
}

static Bool isGlobalBlockToSpacePageDefined()
{
    return blockToSpacePage != Nil;
}

static Bool isThreadLocalBlockToSpacePageDefined()
{
    // Do not query thread's context at the end of the process when threadsDoCleanupOnTermination is false
    return threadsDoCleanupOnTermination &&  stdGetThreadContext()->memInfo.blockToSpacePage != Nil;
}

static uInt getIncrementedBlockCount()
{
    stdAtomicFetchAndAdd(&blockCount, 1);
    return blockCount;
}

static void addBlockToSpacePageEntry(rangemapDomain_t dom, Pointer page)
{
    stdMemSpace_t    savedSpace;

    savedSpace = stdSwapMemSpace( memspNativeMemSpace );

    if (!isThreadLocalBlockToSpacePageDefined()) {
        stdGetThreadContext()->memInfo.blockToSpacePage = rangemapCreate();
        stdMemspInitAndAquireLock(&blockToSpacePageLock);
        if (!threadLocalblockToPageMaps) {
            threadLocalblockToPageMaps = setNEW(Pointer, 8);
        }
        setInsert(threadLocalblockToPageMaps, getThreadlocalBlockSpacePageRecord());
        stdMemspSetCleanupHandler();
        stdMemspReleaseLock(&blockToSpacePageLock);
    }

    rangemapDefine(getThreadlocalBlockSpacePageRecord(), dom, page );
    stdSwapMemSpace(savedSpace);

    savedSpace = stdSwapMemSpace( memspNativeMemSpace );

    stdMemspInitAndAquireLock(&blockToSpacePageLock);
    // Insert block->page record in global map as well
    if (!isGlobalBlockToSpacePageDefined()) {
        blockToSpacePage = rangemapCreate();
        stdMemspSetCleanupHandler();
    }
    rangemapDefine(blockToSpacePage, dom, page );
    stdMemspReleaseLock(&blockToSpacePageLock);
    stdSwapMemSpace(savedSpace);
}

static PageHeader getBlockToSpacePageEntry(Pointer block)
{
    PageHeader page = Nil;

    if (!block) {
        return Nil;
    }

    if (isThreadLocalBlockToSpacePageDefined()) {
        page = rangemapApply(getThreadlocalBlockSpacePageRecord(),  SB_INDEX(block));
    }
    if (page) {
        return page;
    }
    if (!isGlobalBlockToSpacePageDefined()) {
        return page;
    }
    // Check in the global bock->page map if record is not present in thread local map
    stdMemspInitAndAquireLock(&blockToSpacePageLock);
    page = rangemapApply(blockToSpacePage, SB_INDEX(block)); 
    stdMemspReleaseLock(&blockToSpacePageLock);
    return page;
}

static void removeBlockToSpacePageEntry(PageHeader page)
{
    rangemapDomain_t dom;

    dom.start  = SB_INDEX(page->contents);
    dom.length = SB_INDEX(page->size);

    if (isThreadLocalBlockToSpacePageDefined() && rangemapApply(getThreadlocalBlockSpacePageRecord(), dom.start)) {
        stdMemSpace_t    savedSpace = stdSwapMemSpace( memspNativeMemSpace );
        rangemapUndefine(getThreadlocalBlockSpacePageRecord(), dom);
        stdSwapMemSpace(savedSpace);
    }

    stdMemSpace_t    savedSpace = stdSwapMemSpace( memspNativeMemSpace );
    stdMemspInitAndAquireLock(&blockToSpacePageLock);
    rangemapUndefine(blockToSpacePage, dom);
    stdMemspReleaseLock(&blockToSpacePageLock);
    stdSwapMemSpace(savedSpace);
}

static inline PageHeader getPage( Pointer block )
{
    return getBlockToSpacePageEntry(block);
}

static Bool globalDebug;

/*----------------------------- Utility Functions ----------------------------*/

#define LB_HEADER_SIZE   stdROUNDUP( sizeof( struct LBHeaderRec ), stdALIGNMENT )

#define MINUS_ONE        ((Pointer)-1)


static inline Bool isAllocated(LBHeader header)
{
    return header->next == MINUS_ONE;
}

static inline LBHeader nextHeader(LBHeader header)
{
    return (LBHeader)((Address)(header) + header->size);
}

static inline LBHeader prevHeader(LBHeader header)
{
    return (LBHeader)((Address)(header) - header->prevSize);
}

static inline Pointer HTOB(LBHeader header)
{
    return (Pointer)((Address)(header) + LB_HEADER_SIZE);
}

static inline LBHeader BTOH(Pointer block)
{
    return (LBHeader)((Address)(block) - LB_HEADER_SIZE);
}

static inline SizeT blockUserSize(PageHeader page, Pointer block)
{
    if (page->sbp) {
        return ((SBPage)page)->blockSize;
    } else {
        return BTOH(block)->size - LB_HEADER_SIZE;
    }
}


/*----------------------------------------------------------------------------*/
/*--------------------------- Large Block Allocator --------------------------*/
/*----------------------------------------------------------------------------*/


static void lbLinkHeader( stdMemSpace_t space, LBHeader header )
{
    LBHeader freeList = &space->largeBlocks[ stdLOG2_64(header->size) ];

    header->prev = freeList;
    header->next = freeList->next;

    freeList->next = header;

    if (header->next) {
        header->next->prev = header;
    }
}


static void lbUnlinkHeader( LBHeader header )
{
    if (header->next) {
        header->next->prev= header->prev;
    }

    if (header->prev) {
        header->prev->next= header->next;
    }

    header->next = MINUS_ONE;
}


static void lbAllocPage( stdMemSpace_t space, SizeT size )
{
    LBPage        page;
    Byte         *contents;
    LBHeader      preSentinel;
    LBHeader      header;
    LBHeader      postSentinel;
    stdMemSpace_t savedSpace = stdSwapMemSpace( memspNativeMemSpace );

    stdNEW(page);

    size           = stdMAX(size,space->pageSize);
    contents       = stdMALLOC(size + 2*LB_HEADER_SIZE);

    preSentinel    = (LBHeader)(contents);
    header         = (LBHeader)(contents + LB_HEADER_SIZE);
    postSentinel   = (LBHeader)(contents + LB_HEADER_SIZE + size);


    preSentinel->next      = MINUS_ONE;
    preSentinel->prev      = Nil;
    preSentinel->size      = LB_HEADER_SIZE;
    preSentinel->prevSize  = 0;

    header->size           = size;
    header->prevSize       = LB_HEADER_SIZE;

    lbLinkHeader( space, header );

    postSentinel->next     = MINUS_ONE;
    postSentinel->prev     = Nil;
    postSentinel->size     = LB_HEADER_SIZE;
    postSentinel->prevSize = size;


    page->header.freeSize = size;
    page->header.size     = size;
    page->header.space    = space;
    page->header.contents = contents;
    page->header.sbp      = False;
    page->header.count    = getIncrementedBlockCount();
    page->postSentinel    = postSentinel;

    page->header.next     = (PageHeader)space->lbPages;
    space->lbPages        = page;

    {
        uInt blockIndex = stdLOG2_64(size);
        space->largestBlocksIndex = stdMAX( space->largestBlocksIndex, blockIndex );
    }

    {
        rangemapDomain_t dom;

        dom.start  = SB_INDEX(page->header.contents);
        dom.length = SB_INDEX(page->header.size);

        addBlockToSpacePageEntry(dom, page);
    }

    stdSwapMemSpace( savedSpace );
}


static void freePage( PageHeader page )
{
    removeBlockToSpacePageEntry(page);

  #ifdef MEMORY_CHECK
    stdMEMSET_N(page->contents,PAGE_FREED_PATTERN,page->size);
  #endif

    stdFREE(page->contents);
    stdFREE(page);
}


/*----------------------------- Utility Functions ----------------------------*/

static void lbSpaceTune(stdMemSpace_t space)
{
    LBPage *pp = &(space->lbPages);

    while (*pp) {
        LBPage page= *pp;

        if (page->header.size == page->header.freeSize) {
            *pp = (Pointer) page->header.next;
             freePage(&page->header);
        } else {
             pp = (Pointer)&page->header.next;
        }
    }

    space->lbTuningSlack = space->lbTuneCount;
}


/*----------------------------- Utility Functions ----------------------------*/

static inline Pointer lbMalloc(stdMemSpace_t space, SizeT size)
{
    size += LB_HEADER_SIZE;

    //if (space->lbTuningSlack == 0) { lbSpaceTune(space); }

    while (True) {
        uInt i;

        for (i = stdLOG2_64(size); i <= space->largestBlocksIndex; i++) {
            LBHeader header= space->largeBlocks[i].next;

            while (header) {

                if (header->size < size) {
                    header = header->next;
                } else {
                    Pointer result     = HTOB(header);
                    SizeT   excessSize = header->size - size;

                    lbUnlinkHeader(header);

                    if ( excessSize >= (LB_HEADER_SIZE + stdALIGNMENT) ) {
                        LBHeader nHeader,nnHeader;

                        header->size       = size;

                        nHeader            = nextHeader(header);
                        nHeader->size      = excessSize;
                        nHeader->prevSize  = size;

                        nnHeader           = nextHeader(nHeader);
                        nnHeader->prevSize = excessSize;

                        lbLinkHeader( space, nHeader);

                        if (space->lbTuningSlack) { space->lbTuningSlack--; }
                    }

                    {
                        PageHeader page = getPage(header);
                        if(page) { page->freeSize -= header->size; }
                    }

                    return result;
                }
            }
        }

        if (setSafetyValve(size)) {
            lbAllocPage(space, size);
            releaseSafetyValve();
        } else {
            return Nil;
        }
    }
}


static inline void lbFree(LBPage page, Pointer block)
{
    stdMemSpace_t space   = page->header.space;
    LBHeader      header;
    LBHeader      nHeader;
    LBHeader      pHeader;

    stdASSERT( block != Nil, ("Invalid block"));
    header = BTOH(block);
    nHeader = nextHeader(header);
    pHeader = prevHeader(header);
    stdASSERT( isAllocated(header), ("Freeing already freed block") );

    page->header.freeSize += header->size;

    if (space->lbTuningSlack) { space->lbTuningSlack--; }

    if (!isAllocated(nHeader)) {
       /*
        * Merge with next block:
        */
        LBHeader nnHeader= nextHeader(nHeader);

        lbUnlinkHeader(nHeader);

        header->size      += nHeader->size;

        nnHeader->prevSize = header->size;

        nHeader = nnHeader;
    }

    if (!isAllocated(pHeader)) {
       /*
        * Merge with previous block:
        */
        pHeader->size     += header->size;

        nHeader->prevSize  = pHeader->size;
    } else {
        lbLinkHeader( space, header );
    }
}


/*----------------------------------------------------------------------------*/
/*--------------------------- Small Block Allocator --------------------------*/
/*----------------------------------------------------------------------------*/

static void sbAllocPage( stdMemSpace_t space, SizeT blockSize )
{
    SizeT         size;
    SBPage        page;
    Byte         *contents;
    stdMemSpace_t savedSpace = stdSwapMemSpace( memspNativeMemSpace );

    stdNEW(page);

    size     = stdROUNDUP(space->pageSize,blockSize);
    contents = stdMALLOC(size);

    page->header.freeSize = size;
    page->header.size     = size;
    page->header.space    = space;
    page->header.contents = contents;
    page->header.sbp      = True;
    page->header.count    = getIncrementedBlockCount();

    page->blockSize       = blockSize;

  #ifdef MEMORY_CHECK
    page->freeBlocks = bitSetCreate();
    bitSetInsertRange( page->freeBlocks, 0, (size / blockSize) );
  #endif

    {
        SBPage op= mapApply( space->sbPages, (Pointer)(Address)blockSize );
        page->header.next= (PageHeader)op;
        mapDefine(space->sbPages, (Pointer)(Address)blockSize, page);
    }

    {
        stdList_t blockList= Nil;
        Byte*     block    = contents;
        Byte*     limit    = contents + size;

        while (block<limit) {
            stdList_t b= (stdList_t)block;

            b->head   = page;
            b->tail   = blockList;
            blockList = b;

            block += blockSize;
        }

        space->smallBlocks[ SB_INDEX(blockSize) ]= blockList;
    }

    {
        rangemapDomain_t dom;

        dom.start  = SB_INDEX(page->header.contents);
        dom.length = SB_INDEX(page->header.size);

        addBlockToSpacePageEntry(dom, page);
    }

    space->nrofSmallPages++;

    stdSwapMemSpace( savedSpace );
}


static void sbFreePage( SBPage page )
{
  #ifdef MEMORY_CHECK
    bitSetDelete(page->freeBlocks);
  #endif

    page->header.space->nrofSmallPages--;

    freePage(&page->header);
}


/*----------------------------- Utility Functions ----------------------------*/

     static SBPage tuneSbP( SBPage pages, uInt blockSize, stdMemSpace_t space )
     {
         {
             stdList_t *blocks= &space->smallBlocks[ SB_INDEX(blockSize) ];

             while (*blocks) {
                 SBPage p= (*blocks)->head;

                 if (p->header.size == p->header.freeSize) {
                    *blocks= (Pointer) (*blocks)->tail;
                 } else {
                     blocks= (Pointer)&(*blocks)->tail;
                 }
             }
         }

         {
             SBPage *pp= &pages;

             while (*pp) {
                 SBPage p= *pp;

                 if (p->header.size == p->header.freeSize) {
                    *pp= (Pointer) p->header.next;

                     sbFreePage(p);
                 } else {
                     pp= (Pointer)&p->header.next;
                 }
             }
         }

         return pages;
     }

     static void STD_CDECL tuneSbPages( uInt blockSize, SBPage pages, stdMemSpace_t space)
     {
         mapDefine(space->sbPages, (Pointer)(Address)blockSize, tuneSbP(pages,blockSize,space) );
     }

static void sbSpaceTune(stdMemSpace_t space)
{
    mapTraverse(space->sbPages,(stdPairFun)tuneSbPages, space);
    space->sbTuningSlack = space->sbTuneCount * (space->nrofSmallPages / 128);
}


/*----------------------------- Utility Functions ----------------------------*/

static inline Pointer sbMalloc(stdMemSpace_t space, SizeT size)
{
    uInt      sbIndex;
    stdList_t sb;
    SBPage    page;

    size    = stdMAX( size, sizeof(stdListRec) );

    sbIndex = SB_INDEX(size);
    sb      = space->smallBlocks[sbIndex];

    if (!sb) {
        if (setSafetyValve(space->pageSize)) {
            sbAllocPage(space,size);
            releaseSafetyValve();
        } else {
            return Nil;
        }
        sb = space->smallBlocks[sbIndex];
    }

    space->smallBlocks[sbIndex]= sb->tail;
    page= sb->head;

    page->header.freeSize -= size;

  #ifdef MEMORY_CHECK
    {
        uInt index= ((Byte*)sb - page->header.contents) / size;
        bitSetRemove(page->freeBlocks,index);
    }
  #endif

    return sb;
}

static inline void sbFree(SBPage page, Pointer block, stdMemSpace_t space, SizeT size)
{
    uInt      sbIndex = SB_INDEX(size);
    stdList_t sb      = block;

  #ifdef MEMORY_CHECK
    {
        uInt index= ((Byte*)sb - page->header.contents) / size;
        stdASSERT( !bitSetElement(page->freeBlocks,index), ("Freeing already freed block") );
        bitSetInsert(page->freeBlocks,index);
    }
  #endif

    sb->tail                    = space->smallBlocks[sbIndex];
    sb->head                    = page;
    space->smallBlocks[sbIndex] = sb;

    page->header.freeSize += size;

    //if (space->sbTuningSlack-- == 0) { sbSpaceTune(space); }
}


/*------------------------------- API Functions ------------------------------*/

/*
 * Function        : Allocate memory block from memory space
 * Parameters      : space  (I) Space to allocate from.
 *                   size   (I) Size to allocate.
 * Function Result : Requested memory block, or raise an out of memory
 *                   exception when memory allocation failed.
 */
Pointer STD_CDECL memspMalloc(stdMemSpace_t space, SizeT size)
{
    Byte *result= Nil;
#ifdef IS_0BYTE_ALLOC_NULL
    /* Normally malloc(0) returns a non-null value, 
     * but some platforms return NULL,
     * in that case alloc 1 byte so don't get out-of-memory error. */
    if (size == 0) {
       size = 1;
    }
#endif


    if (!space) {
        result= mbMalloc(size,False);

        if (!result && safetyValveIsSet) {
            blowSafetyValve(True);
            result= mbMalloc(size,False);
        }


        if (!result) { stdOutOfMemory(); }

    } else {
        stdMemspInitAndAquireLock(&space->memspLock);
        size= stdROUNDUP64(size,stdALIGNMENT);

        if (size < LARGE_BLOCK_THRESHOLD) {
            result= sbMalloc(space,size);
        } else {
            result= lbMalloc(space,size);
        }

        stdMemspReleaseLock(&space->memspLock);
        #ifdef MEMORY_CHECK
            stdMEMSET_N(result,BLOCK_ALLOCATED_PATTERN,size);
        #endif

        #ifndef LWALGRIND
            VALGRIND_MEMPOOL_ALLOC(space,result,size);
        #endif

        #ifdef MALLOC_TRAP
            if (result == mallocTrap) { STOP(0); }
        #endif

        stdASSERT( result, ("If we end up here, we should have the requested memory") );
    }

#ifdef ALIGNED_ALLOCATIONS
        stdASSERT( stdISALIGNED(result, stdALIGNMENT), ("un-aligned memory"));
#endif

    return result;
}



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
Pointer STD_CDECL memspRealloc(Pointer block, SizeT size)
{
    Byte      *result;
    PageHeader page;


    page = getPage(block);

    stdASSERT( block, ("Don't realloc empty block" ));

    if (!page) {
        result= mbRealloc(block,size);

    } else {
        stdMemspInitAndAquireLock(&page->space->memspLock);
        SizeT   oldSize = blockUserSize (page,block);

        #ifdef MALLOC_TRAP
            if (block == freeTrap) { STOP(0); }
        #endif

        stdMemspReleaseLock(&page->space->memspLock);

        result = memspMalloc(page->space,size);

        stdASSERT( !page->space->deleted, ("Block allocation from deleted memory space '%s'", page->space->name) );

        stdMEMCOPY_N( result, block, stdMIN(size,oldSize) );

        memspFree(block);

        #ifdef MALLOC_TRAP
            if (result == mallocTrap) { STOP(0); }
        #endif
    }

#ifdef ALIGNED_ALLOCATIONS
        stdASSERT( stdISALIGNED(result, stdALIGNMENT), ("un-aligned memory"));
#endif

    return result;
}



/*
 * Function        : Free memory block previously allocated via memspMalloc
 *                   NB: Nil is a legal memory block argument to this function,
 *                       in which case it behaves as a nop
 * Parameters      : block  (I) Memory block to free, or Nil
 * Function Result :
 */
void STD_CDECL memspFree(Pointer block)
{
    PageHeader page;

    page = getPage(block);

    if (!page) {

        mbFree(block,False);

    } else {
        SizeT blockSize;

        stdMemspInitAndAquireLock(&page->space->memspLock);
        #ifdef MALLOC_TRAP
            if (block == freeTrap) { STOP(0); }
        #endif

        blockSize = blockUserSize(page,block);

        stdASSERT( !page->space->deleted, ("Block freeing from deleted memory space '%s'", page->space->name) );

        #ifdef MEMORY_CHECK
        {
            Byte *b = block;
            stdMEMSET_N(b,BLOCK_FREED_PATTERN,blockSize);
        }
        #endif

        if (blockSize < LARGE_BLOCK_THRESHOLD) {
            sbFree((SBPage)page,block,page->space,blockSize);
        } else {
            lbFree((LBPage)page,block);
        }

        stdMemspReleaseLock(&page->space->memspLock);

        #ifndef LWALGRIND
            VALGRIND_MEMPOOL_FREE(page->space,block);
        #endif
    }
}



/*
 * Function        : Obtain memory space that holds the specified address
 *                   NB: the specified argument may be any address within
 *                       a block returned by memspMalloc or memspRealloc
 *                       (and which has not yet been freed).
 * Parameters      : address  (I) address value to colwert
 * Function Result : the memory space from which the address has been allocated
 */
stdMemSpace_t STD_CDECL memspGetSpace(Pointer address)
{
    PageHeader page;

    page = getPage(address);

    if (page) {
        return page->space;
    } else {
        return Nil;
    }
}



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
stdMemSpace_t STD_CDECL memspCreate( cString name, stdMemSpace_t parent, SizeT pageSize )
{
    stdMemSpace_t result;
    stdMemSpace_t savedSpace = stdSwapMemSpace( memspNativeMemSpace );

    if (pageSize == 0) {
        if (parent) { pageSize= parent->pageSize;      }
               else { pageSize= memspDEFAULT_PAGESIZE; }
    }

    pageSize= stdROUNDUP64(pageSize,stdALIGNMENT);

    stdNEW(result);
    result->parent      = parent;
    result->pageSize    = pageSize;
    result->children    = setNEW(Pointer,8);
    result->sbPages     = mapNEW(uInt,   8);

    result->sbTuneCount = TUNE_PARAMETER;
    result->lbTuneCount = TUNE_PARAMETER;
    result->memspLock   = stdMutexCreate();

    if (parent) {
        stdGlobalEnter();
        setInsert(parent->children,result);
        stdGlobalExit();
    }

    if (!name) { name= "<anonymous>"; }

    result->name = stdCOPYSTRING(name);

    stdSwapMemSpace(savedSpace);

    #ifndef LWALGRIND
        VALGRIND_CREATE_MEMPOOL(result,0,0);
    #endif

    return result;
}



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
     static void setSpace( PageHeader page, stdMemSpace_t space )
     { while (page) { PageHeader p= page; page=page->next; p->space = space; } }

     static void STD_CDECL freePages( PageHeader page, stdMemSpace_t space )
     { while (page) { PageHeader p= page; page=page->next; sbFreePage((SBPage)p); } }

     static void STD_CDECL addSbPages( uInt blockSize, PageHeader pages, stdMemSpace_t space)
     {
         SBPage pages1= mapApply( space->sbPages, (Pointer)(Address)blockSize );
         setSpace(pages,space);
         mapDefine(space->sbPages, (Pointer)(Address)blockSize, listConcat((stdList_t)pages,(stdList_t)pages1) );
     }

     static void llistConcat( LBHeader l, LBHeader r )
     {
         r= r->next;  // skip sentinel

         if (r) {
             while (l->next) { l= l->next; }

             l->next = r;
             r->prev = l;
         }
     }

void STD_CDECL memspDelete( stdMemSpace_t space, Bool addToParent )
{
    stdMemSpace_t parent = space->parent;

    stdASSERT( space != memspNativeMemSpace,                        ("Don't delete the native memory space"           ));
    stdASSERT( space != stdLwrrentMemspace,                         ("Don't delete the current memory space"          ));
    stdASSERT( stdIMPLIES(addToParent, space!=memspNativeMemSpace), ("Don't add contents to the native memory space"  ));
    stdASSERT( !space->deleted,                                     ("Double delete of memory space '%s'", space->name));

    space->deleted= True;
    if (globalDebug) {
        return;
    }

    //sbSpaceTune(space);
    //lbSpaceTune(space);

    setTraverse( space->children, (stdEltFun)memspDelete, (Pointer)(Address)addToParent );

    setDelete(space->children);

    if (parent) {
        setRemove(parent->children, space);
    }

    {
        LBPage page= space->lbPages;

        while (page) {
            LBPage np= (LBPage)page->header.next;

            if (addToParent) {

                np->header.space      = parent;
                np->header.next       = (PageHeader)parent->lbPages;

                parent->lbPages       = np;
                parent->lbTuningSlack = 0;
            } else {
                freePage(&page->header);
            }

            page= np;
        }
    }

    if (addToParent) {
        uInt i;

        stdMemSpace_t savedSpace = stdSwapMemSpace( memspNativeMemSpace );
        mapTraverse(space->sbPages,(stdPairFun)addSbPages,parent);
        stdSwapMemSpace(savedSpace);

        for (i=0; i<SB_INDEX(LARGE_BLOCK_THRESHOLD); i++) {
            parent->smallBlocks[i]= listConcat( parent->smallBlocks[i], space->smallBlocks[i] );
        }

        for (i=0; i<stdBITSIZEOF(SizeT); i++) {
            llistConcat( &parent->largeBlocks[i], &space->largeBlocks[i] );
        }

        parent->largestBlocksIndex = stdMAX( parent->largestBlocksIndex, space->largestBlocksIndex );

    } else {
        mapRangeTraverse( space->sbPages, (stdEltFun)freePages, Nil );
    }

    if (isThreadLocalBlockToSpacePageDefined() && rangemapIsEmpty(getThreadlocalBlockSpacePageRecord())) {
        stdRangeMap_t savedMap =  getThreadlocalBlockSpacePageRecord();
        setRemove(threadLocalblockToPageMaps, savedMap);
        stdGetThreadContext()->memInfo.blockToSpacePage = Nil;
        rangemapDelete(savedMap);
    }

    mapDelete(space->sbPages);
    stdMutexDelete(space->memspLock);
    stdFREE(space->name);
    stdFREE(space);

    #ifndef LWALGRIND
        VALGRIND_DESTROY_MEMPOOL(space);
    #endif
}




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
void STD_CDECL memspSwapTuneCount( stdMemSpace_t space, uInt *lbTuneCount, uInt *sbTuneCount )
{
    if (lbTuneCount) { stdSWAP( space->lbTuneCount, *lbTuneCount, uInt ); }
    if (sbTuneCount) { stdSWAP( space->sbTuneCount, *sbTuneCount, uInt ); }
}



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
    typedef struct {
        stdWriter_t        wr;
        Bool               cleanup;
        memspPrintLevel    level;
        Bool               printChildren;
        Int                indent;
        Int                indentSize;
    } PrintRec;

    static void PRIND( PrintRec *rec )
    {
        uInt i;
        for (i=0; i<rec->indent; i++) { wtrPrintf(rec->wr,"\t"); }
    }

    static void PRHDR( String name, PrintRec *rec )
    {
        Char buffer[10000];

        sprintf(buffer,"'%s'", name );

        PRIND(rec); wtrPrintf(rec->wr, "Memory space statistics for %-40s", buffer );

        if (rec->level==memspPrintSummary) {
            wtrPrintf(rec->wr, ": " );
        } else {
            wtrPrintf(rec->wr, "\n" );
            PRIND(rec); wtrPrintf(rec->wr, "============================" );
                    {
                        uInt i= strlen(buffer);
                        while (i--) { wtrPrintf(rec->wr,"="); }
                    }
            wtrPrintf(rec->wr, "\n" );
        }
    }

static void STD_CDECL msprint( stdMemSpace_t space, PrintRec *rec )
{
    Address allocated  = 0;
    Address available  = 0;
    uInt    nrLBPages  = 0;
    uInt    nrSBPages  = 0;
    uInt    maxList    = 0;
    uInt    totalList  = 0;
    uInt    blockSize;

    LBPage page;

    PRHDR(space->name,rec);

    if (rec->cleanup) {
        sbSpaceTune(space);
        lbSpaceTune(space);
    }

    page= space->lbPages;

    while (page) {
        LBHeader  header    = nextHeader((LBHeader)page->header.contents);
        Address   minSize   = (Address)-1;
        Address   maxSize   = 0;
        uInt      amount    = 0;

        while (header != page->postSentinel) {
            minSize    = stdMIN(minSize,header->size);
            maxSize    = stdMAX(minSize,header->size);

            amount++;

            header= nextHeader(header);
        }

        if (rec->level==memspPrintFull) {
            PRIND(rec); wtrPrintf(rec->wr, "@@ large block page %4d : 0x%" stdFMT_ADDR "/0x%" stdFMT_ADDR ", #=%d \tmax=0x%" stdFMT_ADDR "\n",
                                 nrLBPages, page->header.freeSize,page->header.size,amount,maxSize );
        }

        nrLBPages++;
        available += page->header.freeSize;
        allocated += page->header.size;

        maxList  = stdMAX(maxList, amount  );

        page= (LBPage)page->header.next;
    }


    for (blockSize=0; blockSize<LARGE_BLOCK_THRESHOLD; blockSize += stdALIGNMENT) {
        Address all      = 0;
        Address avble    = 0;
        uInt    nrpages  = 0;
        SBPage  pages    = mapApply( space->sbPages, (Pointer)(Address)blockSize );

        while (pages) {
            nrpages++;
            avble += pages->header.freeSize;
            all   += pages->header.size;
            pages  = (SBPage)pages->header.next;
        }

        if (rec->level==memspPrintFull && nrpages) {
            PRIND(rec); wtrPrintf(rec->wr, "@@ small block size %3d: 0x%" stdFMT_ADDR "/0x%" stdFMT_ADDR " (%d/%d blocks) %d page%s\n",
                    blockSize, avble,all, (uInt32)(avble/blockSize), (uInt32)(all/blockSize), nrpages, stdPLURAL(nrpages) );
        }

        nrSBPages += nrpages;
        available += avble;
        allocated += all;
    }

    {
        Char  availableB[100];
        Char  allocatedB[100];
        Char  usedB[100];

        sprintf( availableB, "0x%" stdFMT_ADDR, available );
        sprintf( allocatedB, "0x%" stdFMT_ADDR, allocated );
        sprintf( usedB,      "0x%" stdFMT_ADDR, allocated-available );

        if (rec->level == memspPrintSummary) {
            wtrPrintf(rec->wr, "\t available= \t%15s, allocated= \t%15s, used= \t%15s\n", availableB, allocatedB, usedB );

        } else {
            PRIND(rec); wtrPrintf(rec->wr, "Page size                 : 0x%x bytes\n", space->pageSize  );
            PRIND(rec); wtrPrintf(rec->wr, "Total allocated           : %15s bytes\n",       allocatedB );
            PRIND(rec); wtrPrintf(rec->wr, "Total available           : %15s bytes\n",       availableB );
            PRIND(rec); wtrPrintf(rec->wr, "Total in use              : %15s bytes\n",       usedB      );
            PRIND(rec); wtrPrintf(rec->wr, "Nrof small block pages    : %d\n",               nrSBPages  );
            PRIND(rec); wtrPrintf(rec->wr, "Nrof large block pages    : %d\n",               nrLBPages  );
          if (nrLBPages) {
            PRIND(rec); wtrPrintf(rec->wr, "Longest free list size    : %d\n",                           maxList    );
            PRIND(rec); wtrPrintf(rec->wr, "Average free list size    : %d\n",               totalList / nrLBPages  );
          }
            wtrPrintf(rec->wr, "\n\n" );
        }
    }
    if (rec->printChildren && space->children) {
        rec->indent += rec->indentSize;
        setTraverse( space->children, (stdEltFun)msprint, rec );
        rec->indent -= rec->indentSize;
    }
}

void STD_CDECL memspPrint( /*stdWriter_t*/ Pointer wr, stdMemSpace_t space, Bool cleanup, memspPrintLevel level, Bool printChildren, Bool indent )
{
    PrintRec rec;

    stdGlobalEnter();

    rec.wr            = wr;
    rec.cleanup       = cleanup;
    rec.level         = level;
    rec.printChildren = printChildren;
    rec.indent        = 0;
    rec.indentSize    = !!indent;

    msprint(space,&rec);

    stdGlobalExit();
}


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
uInt STD_CDECL memspPointerHash( Pointer address )
{
    uInt       result;
    PageHeader page;


    page = getPage(address);

    if (page) {
        result = STD_POINTER_HASH( ((Address)address) - ((Address)page) ) ^ page->count;
    } else {
        result = STD_POINTER_HASH( address );
    }


    return result;
}


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
uInt STD_CDECL memspPointerLE( Pointer l, Pointer r )
{
    if (l==r) {
        return True;
    } else {
        uInt       result;
        PageHeader lpage,rpage;


        lpage = getPage(l);
        rpage = getPage(r);

        if (lpage && rpage && (lpage != rpage)) { result= lpage->count <= rpage->count; }
                                           else { result= l            <= r;            }

        return result;
    }
}

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
void STD_CDECL memspSetGlobalDebug( Bool set )
{
    globalDebug = set;
}



/*
 * Function        : Check whether the block's memory pool is still alive,
 *                   and complain via stdSYSLOG if this is not the case.
 * Parameters      : block         (I) Block to check
 * Function Result :
 */
void STD_CDECL memspCheckBlock( Pointer block )
{
    PageHeader page;


    page = getPage(block);

    if (page) {
        stdASSERT( !page->space->deleted, ("Block from deleted memory space '%s' founc in memspCheckBlock", page->space->name) );
    }

}


/*
 * Function        : Print an identification of the specified memory block via writer object.
 * Parameters      : wr            (I) Writer to print to
 *                   block         (I) Block to describe
 * Function Result :
 */
void STD_CDECL memspDescribeBlock( /*stdWriter_t*/ Pointer wr, Pointer block )
{
    PageHeader page;

    stdGlobalEnter();

    page = getPage(block);

    if (page) {
        stdMemSpace_t space = page->space;
        wtrPrintf(wr,"Block %p: size= 0x%" stdFMT_ADDR ", memory space = '%s'\n", block, blockUserSize(page,block), space->name );
    } else {
        wtrPrintf(wr,"Block %p: not from known memory space, assumed obtained via malloc\n", block);
    }

    stdGlobalExit();
}



#ifdef MEMORY_CHECK

/*
 * Function        : Print all of the blocks in specified memspace that are lwrrently in use
 * Parameters      : space  (I) Memory space to print
 * Function Result :
 */
     static void printLbPages( LBPage page )
     {
         printf("*** LARGE BLOCKS : ");

         for (; page; page= (LBPage)page->header.next) {

             LBHeader  header = nextHeader((LBHeader)page->header.contents);

             while (header != page->postSentinel) {
                 if (isAllocated(header)) {
                     printf( "  0x%" stdFMT_ADDR "/%d", (Address)HTOB(header), (uInt)header->size );
                 }

                 header= nextHeader(header);
             }
         }

         printf("\n\n");
     }

     static void STD_CDECL printSbPages( uInt blockSize, SBPage page )
     {
         printf("*** SMALL BLOCK SIZE %d : ", blockSize);

         for (; page; page= (SBPage)page->header.next) {
             uInt      index    = 0;
             Byte*     block    = page->header.contents;
             Byte*     limit    = page->header.contents + page->header.size;

             while (block<limit) {
                 if (!bitSetElement(page->freeBlocks,index)) {
                     printf( "  0x%" stdFMT_ADDR, (Address)block );
                 }

                 index++;
                 block += blockSize;
             }
         }

         printf("\n\n");
     }

void STD_CDECL memspPrintAllocatedBlocks( stdMemSpace_t space )
{
    mapTraverse(space->sbPages,(stdPairFun)printSbPages, Nil);

    printLbPages(space->lbPages);
}

#endif


