/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2001-2015 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// heap.c
//
//*****************************************************

//
// includes
//
#include "heap.h"

#include "lwport/lwport.h"
#include "../../drivers/resman/kernel/inc/pma/pma.h"

#ifdef USERMODE
#include "lwos.h"
#endif // USERMODE

#ifndef USERMODE

typedef struct
{
    LwU32 owner;
    LwU32 type;
    LwU32 Hwres;
    LwU32 begin, end, size;
} HEAP_DUMP_BLOCK;

int __cdecl compHDB_OwnerUp(const void *elem1, const void *elem2 )
{
    const HEAP_DUMP_BLOCK *pHDB1 = (const HEAP_DUMP_BLOCK*)elem1,
                          *pHDB2 = (const HEAP_DUMP_BLOCK*)elem2;
    int diff = pHDB1->owner-pHDB2->owner;
    if (diff == 0)
        diff = pHDB1->begin-pHDB2->begin;
    return diff;
}
int __cdecl compHDB_OwnerDown(const void *elem1, const void *elem2 )
{
    const HEAP_DUMP_BLOCK *pHDB1 = (const HEAP_DUMP_BLOCK*)elem1,
                          *pHDB2 = (const HEAP_DUMP_BLOCK*)elem2;
    int diff = pHDB2->owner-pHDB1->owner;
    if (diff == 0)
        diff = pHDB1->begin-pHDB2->begin;
    return diff;
}
int __cdecl compHDB_TypeUp(const void *elem1, const void *elem2 )
{
    const HEAP_DUMP_BLOCK *pHDB1 = (const HEAP_DUMP_BLOCK*)elem1,
                          *pHDB2 = (const HEAP_DUMP_BLOCK*)elem2;
    int diff = pHDB1->type-pHDB2->type;
    if (diff == 0)
        diff = pHDB1->begin-pHDB2->begin;
    return diff;
}
int __cdecl compHDB_TypeDown(const void *elem1, const void *elem2 )
{
    const HEAP_DUMP_BLOCK *pHDB1 = (const HEAP_DUMP_BLOCK*)elem1,
                          *pHDB2 = (const HEAP_DUMP_BLOCK*)elem2;
    int diff = pHDB2->type-pHDB1->type;
    if (diff == 0)
        diff = pHDB1->begin-pHDB2->begin;
    return diff;
}
int __cdecl compHDB_AddrUp(const void *elem1, const void *elem2 )
{
    const HEAP_DUMP_BLOCK *pHDB1 = (const HEAP_DUMP_BLOCK*)elem1,
                          *pHDB2 = (const HEAP_DUMP_BLOCK*)elem2;
    return (pHDB1->begin-pHDB2->begin);
}
int __cdecl compHDB_AddrDown(const void *elem1, const void *elem2 )
{
    const HEAP_DUMP_BLOCK *pHDB1 = (const HEAP_DUMP_BLOCK*)elem1,
                          *pHDB2 = (const HEAP_DUMP_BLOCK*)elem2;
    return (pHDB2->begin-pHDB1->begin);
}
int __cdecl compHDB_SizeUp(const void *elem1, const void *elem2 )
{
    const HEAP_DUMP_BLOCK *pHDB1 = (const HEAP_DUMP_BLOCK*)elem1,
                          *pHDB2 = (const HEAP_DUMP_BLOCK*)elem2;
    int diff = pHDB1->size-pHDB2->size;
    if (diff == 0)
        diff = pHDB1->begin-pHDB2->begin;
    return diff;
}
int __cdecl compHDB_SizeDown(const void *elem1, const void *elem2 )
{
    const HEAP_DUMP_BLOCK *pHDB1 = (const HEAP_DUMP_BLOCK*)elem1,
                          *pHDB2 = (const HEAP_DUMP_BLOCK*)elem2;
    int diff = pHDB2->size-pHDB1->size;
    if (diff == 0)
        diff = pHDB1->begin-pHDB2->begin;
    return diff;
}


LW_STATUS dumpHeap_ReadBlocks_R60 (HEAP_DUMP_BLOCK *pHeapDumpBuffer, LwU32 firstBlockBufferAddr, LwU32 freeBlockLink, LwU32 *pBlockCount, LwU32 *pVidMemorySum)
{
    LW_STATUS            status;
    MEMBLOCK_R60    blockBuffer;
    LwU32           blockBufferAddr = firstBlockBufferAddr;
    LwU32           blockCount = 0;
    LwU64           bytesRead;

    *pVidMemorySum = 0;
    do
    {
        status = readVirtMem((LwUPtr)blockBufferAddr, &blockBuffer, sizeof(MEMBLOCK_R60), &bytesRead);
        if (status != LW_OK)
            return status;

        pHeapDumpBuffer[blockCount].owner = blockBuffer.owner;
        pHeapDumpBuffer[blockCount].type  = blockBuffer.u0.type;    // Don't care in a free block
        pHeapDumpBuffer[blockCount].Hwres = blockBuffer.u1.hwres;   // Don't care in a free block
        pHeapDumpBuffer[blockCount].begin = blockBuffer.begin;
        pHeapDumpBuffer[blockCount].end   = blockBuffer.end;
        pHeapDumpBuffer[blockCount].size  = blockBuffer.end-blockBuffer.begin+1;

        *pVidMemorySum += pHeapDumpBuffer[blockCount].size;
        blockCount++;
        if (freeBlockLink)
            blockBufferAddr = (LwU32)(LwUPtr)blockBuffer.u1.nextFree;
        else
        {
            blockBufferAddr = (LwU32)(LwUPtr)blockBuffer.next;

            // Verify if this is using correct version of structure.
            // Always call dump all memory block first.
            if (blockCount == 2 && (LwU32)(LwUPtr)blockBuffer.prev != firstBlockBufferAddr)
                return !LW_OK;
        }

    } while (blockBufferAddr != firstBlockBufferAddr);

    *pBlockCount = blockCount;
    return LW_OK;
}

LW_STATUS dumpHeap_ReadBlocks_R65 (HEAP_DUMP_BLOCK *pHeapDumpBuffer, LwU32 firstBlockBufferAddr, LwU32 freeBlockLink, LwU32 *pBlockCount, LwU32 *pVidMemorySum)
{
    LW_STATUS            status;
    MEMBLOCK_R65    blockBuffer;
    LwU32           blockBufferAddr = firstBlockBufferAddr;
    LwU32           blockCount = 0;
    LwU64           bytesRead;

    *pVidMemorySum = 0;
    do
    {
        status = readVirtMem((LwUPtr)blockBufferAddr, &blockBuffer, sizeof(MEMBLOCK_R65), &bytesRead);
        if (status != LW_OK)
            return status;

        pHeapDumpBuffer[blockCount].owner = blockBuffer.owner;
        pHeapDumpBuffer[blockCount].type  = blockBuffer.u0.type;    // Don't care in a free block
        pHeapDumpBuffer[blockCount].Hwres = blockBuffer.u1.hwres;   // Don't care in a free block
        pHeapDumpBuffer[blockCount].begin = blockBuffer.begin;
        pHeapDumpBuffer[blockCount].end   = blockBuffer.end;
        pHeapDumpBuffer[blockCount].size  = blockBuffer.end-blockBuffer.begin+1;

        *pVidMemorySum += pHeapDumpBuffer[blockCount].size;
        blockCount++;
        if (freeBlockLink)
            blockBufferAddr = (LwU32)(LwUPtr)blockBuffer.u1.nextFree;
        else
        {
            blockBufferAddr = (LwU32)(LwUPtr)blockBuffer.next;

            // Verify if this is using correct version of structure.
            // Always call dump all memory block first.
            if (blockCount == 2 && (LwU32)(LwUPtr)blockBuffer.prev != firstBlockBufferAddr)
                return !LW_OK;
        }
    } while (blockBufferAddr != firstBlockBufferAddr);

    *pBlockCount = blockCount;
    return LW_OK;
}

LW_STATUS dumpHeap_ReadBlocks_R70 (HEAP_DUMP_BLOCK *pHeapDumpBuffer, LwU32 firstBlockBufferAddr, LwU32 freeBlockLink, LwU32 *pBlockCount, LwU32 *pVidMemorySum)
{
    LW_STATUS            status;
    MEMBLOCK_R70    blockBuffer;
    LwU32           blockBufferAddr = firstBlockBufferAddr;
    LwU32           blockCount = 0;
    LwU64           bytesRead;

    *pVidMemorySum = 0;
    do
    {
        status = readVirtMem((LwUPtr)blockBufferAddr, &blockBuffer, sizeof(MEMBLOCK_R70), &bytesRead);
        if (status != LW_OK)
            return status;

        pHeapDumpBuffer[blockCount].owner = blockBuffer.owner;
        pHeapDumpBuffer[blockCount].type  = blockBuffer.u0.type;    // Don't care in a free block
        pHeapDumpBuffer[blockCount].Hwres = blockBuffer.u1.hwres;   // Don't care in a free block
        pHeapDumpBuffer[blockCount].begin = blockBuffer.begin;
        pHeapDumpBuffer[blockCount].end   = blockBuffer.end;
        pHeapDumpBuffer[blockCount].size  = blockBuffer.end-blockBuffer.begin+1;

        *pVidMemorySum += pHeapDumpBuffer[blockCount].size;
        blockCount++;
        if (freeBlockLink)
            blockBufferAddr = (LwU32)(LwUPtr)blockBuffer.u1.nextFree;
        else
            blockBufferAddr = (LwU32)(LwUPtr)blockBuffer.next;
    } while (blockBufferAddr != firstBlockBufferAddr);

    *pBlockCount = blockCount;
    return LW_OK;
}

void dumpHeap_AllBlocks (HEAP_DUMP_BLOCK *pHeapDumpBuffer, LwU32 blockCount, LwU32 SortOption,  LwU32 OwnerFilter)
{
    LwU32   i;

    if (SortOption == OWNER_UP)
        qsort(pHeapDumpBuffer, blockCount, sizeof(HEAP_DUMP_BLOCK), compHDB_OwnerUp);
    else if (SortOption == OWNER_DOWN)
        qsort(pHeapDumpBuffer, blockCount, sizeof(HEAP_DUMP_BLOCK), compHDB_OwnerDown);
    else if (SortOption == TYPE_UP)
        qsort(pHeapDumpBuffer, blockCount, sizeof(HEAP_DUMP_BLOCK), compHDB_TypeUp);
    else if (SortOption == TYPE_DOWN)
        qsort(pHeapDumpBuffer, blockCount, sizeof(HEAP_DUMP_BLOCK), compHDB_TypeDown);
    else if (SortOption == ADDR_UP)
        qsort(pHeapDumpBuffer, blockCount, sizeof(HEAP_DUMP_BLOCK), compHDB_AddrUp);
    else if (SortOption == ADDR_DOWN)
        qsort(pHeapDumpBuffer, blockCount, sizeof(HEAP_DUMP_BLOCK), compHDB_AddrDown);
    else if (SortOption == SIZE_UP)
        qsort(pHeapDumpBuffer, blockCount, sizeof(HEAP_DUMP_BLOCK), compHDB_SizeUp);
    else if (SortOption == SIZE_DOWN)
        qsort(pHeapDumpBuffer, blockCount, sizeof(HEAP_DUMP_BLOCK), compHDB_SizeDown);

    dprintf("\r\n");
    dprintf("Owner   Type        Hwres       Begin       End         Size\r\n");
    dprintf("-----------------------------------------------------------------\r\n");
    for (i = 0; i < blockCount; i++)
    {
        if (OwnerFilter)
        {
            if (pHeapDumpBuffer[i].owner != OwnerFilter)
                continue;
        }

        if (pHeapDumpBuffer[i].owner == FREE_BLOCK)
        {
            dprintf("FREE                            0x%08x  0x%08x  0x%08x\r\n",
                    pHeapDumpBuffer[i].begin, pHeapDumpBuffer[i].end, pHeapDumpBuffer[i].size);
            continue;
        }
        dprintf("%.4s    0x%-8x  0x%-8x  0x%08x  0x%08x  0x%08x\r\n",
                (char *)&pHeapDumpBuffer[i].owner, pHeapDumpBuffer[i].type, pHeapDumpBuffer[i].Hwres,
                pHeapDumpBuffer[i].begin, pHeapDumpBuffer[i].end, pHeapDumpBuffer[i].size);
    }

    dprintf("\r\n");
}

void dumpHeap_FreeBlocks (HEAP_DUMP_BLOCK *pHeapDumpBuffer, LwU32 blockCount, LwU32 SortOption,  LwU32 OwnerFilter)
{
    LwU32   i;

    if (SortOption == ADDR_UP)
        qsort(pHeapDumpBuffer, blockCount, sizeof(HEAP_DUMP_BLOCK), compHDB_AddrUp);
    else if (SortOption == ADDR_DOWN)
        qsort(pHeapDumpBuffer, blockCount, sizeof(HEAP_DUMP_BLOCK), compHDB_AddrDown);
    else if (SortOption == SIZE_UP)
        qsort(pHeapDumpBuffer, blockCount, sizeof(HEAP_DUMP_BLOCK), compHDB_SizeUp);
    else if (SortOption == SIZE_DOWN)
        qsort(pHeapDumpBuffer, blockCount, sizeof(HEAP_DUMP_BLOCK), compHDB_SizeDown);

    dprintf("\r\n");
    dprintf("Owner   Begin       End         Size\r\n");
    dprintf("------------------------------------------\r\n");
    for (i = 0; i < blockCount; i++)
    {
        if (OwnerFilter)
        {
            if (pHeapDumpBuffer[i].owner != OwnerFilter)
                continue;
        }
        dprintf("FREE    0x%08x  0x%08x  0x%08x\r\n",
                pHeapDumpBuffer[i].begin, pHeapDumpBuffer[i].end, pHeapDumpBuffer[i].size);
    }
    dprintf("\r\n");
}

//-----------------------------------------------------
// dumpHeap_R60
//
//-----------------------------------------------------
void dumpHeap_R60(void *heapAddr, LwU32 SortOption, LwU32 OwnerFilter)
{
    LW_STATUS       status;
    OBJHEAP_R60     heapBuffer;
    LwU64           bytesRead;
    LwU32           i;
    LwU32           freeVid;
    LwU32           blockCount;
    HEAP_DUMP_BLOCK *pHeapDumpBuffer;
    LwU32           heapVer = 60;

    if (!heapAddr)
    {
        dprintf("lw: Not a valid heap handle...\n");
        return;
    }

    status = readVirtMem((LwUPtr)heapAddr, &heapBuffer, sizeof(OBJHEAP_R60), &bytesRead);
    if (status != LW_OK)
        return;

    blockCount      = (LwU32) heapBuffer.numBlocks;

    if (blockCount > 0 && blockCount < 0x100000)   // A valid number.
    {
        dprintf("lw: Note: Using R60 heap structures...\n");
    }
    else    // try R70
    {
        if (dumpHeap_R70(heapAddr, SortOption, OwnerFilter))
            return;
        else
        {
            dprintf("dumpHeap_R70 failed.");
            return;
        }
    }


    dprintf("lw: Heap dump.  Size = 0x%x\n", heapBuffer.total);
    dprintf("lw:             Free = 0x%x\n", heapBuffer.free);
    dprintf("lw: ==============================================\n");
    for (i = 0; i < heapBuffer.numBanks; i++)
    {
        dprintf("lw: Bank: 0x%x\n", i);
        dprintf("lw: \tOffset     = 0x%x\n", heapBuffer.Bank[i].offset);
        dprintf("lw: \tSize       = 0x%x\n", heapBuffer.Bank[i].size);
    }
    dprintf("\n");

    blockCount      = (LwU32) heapBuffer.numBlocks;
    dprintf("lw: Total number of Blocks = %d.\r\n", blockCount);
    if (!blockCount)
        return;

    pHeapDumpBuffer = (HEAP_DUMP_BLOCK*)malloc( sizeof(HEAP_DUMP_BLOCK)*blockCount);
    if (pHeapDumpBuffer == NULL)
        return;

    //
    // Buffer List
    //
    if (OwnerFilter == 'EERF' || OwnerFilter == 'FREE')
        OwnerFilter = FREE_BLOCK;

    if (heapBuffer.pBlockList)
    {
        if (dumpHeap_ReadBlocks_R60 (pHeapDumpBuffer, (LwU32)(LwUPtr)heapBuffer.pBlockList, 0, &blockCount, &freeVid) != LW_OK)
        {
            if (dumpHeap_ReadBlocks_R65 (pHeapDumpBuffer, (LwU32)(LwUPtr)heapBuffer.pBlockList, 0, &blockCount, &freeVid) != LW_OK)
            {
                free(pHeapDumpBuffer);
                return;
            }
            else
                heapVer = 65;
        }

        dprintf("lw: %sBlock List Forward:\r\n", ((heapVer == 6)?"R60 ":""));

        dumpHeap_AllBlocks (pHeapDumpBuffer, blockCount, SortOption,  OwnerFilter);
    }

    //
    // Block List
    //
    dprintf("lw: FREE Block List Forward:\r\n");
    freeVid  = 0;

    if (heapBuffer.pFreeBlockList)
    {
        if (heapVer == 60)
        {
            if (dumpHeap_ReadBlocks_R60 (pHeapDumpBuffer, (LwU32)(LwUPtr)heapBuffer.pFreeBlockList, 1, &blockCount, &freeVid) != LW_OK)
            {
                free(pHeapDumpBuffer);
                return;
            }
        }
        else
        {
            if (dumpHeap_ReadBlocks_R65 (pHeapDumpBuffer, (LwU32)(LwUPtr)heapBuffer.pFreeBlockList, 1, &blockCount, &freeVid) != LW_OK)
            {
                free(pHeapDumpBuffer);
                return;
            }
        }

        dumpHeap_FreeBlocks (pHeapDumpBuffer, blockCount, SortOption,  OwnerFilter);

        dprintf("\n");
        dprintf("lw: \tThe number of free blocks = %d, Callwlated free video memory = 0x%x\n", blockCount, freeVid);
    }

    free(pHeapDumpBuffer);
}

//-----------------------------------------------------
// dumpHeap_R70
//
//-----------------------------------------------------
LwU32 dumpHeap_R70(void *heapAddr, LwU32 SortOption, LwU32 OwnerFilter)
{
    LW_STATUS            status;
    OBJHEAP_R70     heapBuffer;
    LwU64           bytesRead;
    LwU32           i;
    LwU32           freeVid;
    LwU32           blockCount;
    HEAP_DUMP_BLOCK *pHeapDumpBuffer;

    if (!heapAddr)
    {
        dprintf("lw: Not a valid heap handle...\n");
        return FALSE;
    }

    status = readVirtMem((LwUPtr)heapAddr, &heapBuffer, sizeof(OBJHEAP_R70), &bytesRead);
    if (status != LW_OK)
        return FALSE;

    blockCount      = (LwU32) heapBuffer.numBlocks;

    if (blockCount > 0 && blockCount < 0x100000)   // A valid number.
    {
        dprintf("lw: Note: Using R70 heap structures...\n");
    }
    else
        return  FALSE;


    dprintf("lw: Heap dump.  Size = 0x%x\n", heapBuffer.total);
    dprintf("lw:             Free = 0x%x\n", heapBuffer.free);
    dprintf("lw: ==============================================\n");
    for (i = 0; i < heapBuffer.numBanks; i++)
    {
        dprintf("lw: Bank: 0x%x\n", i);
        dprintf("lw: \tOffset     = 0x%x\n", heapBuffer.Bank[i].offset);
        dprintf("lw: \tSize       = 0x%x\n", heapBuffer.Bank[i].size);
    }
    dprintf("\n");

    blockCount      = (LwU32) heapBuffer.numBlocks;
    dprintf("lw: Total number of Blocks = %d.\r\n", blockCount);
    if (!blockCount)
        return  FALSE;

    pHeapDumpBuffer = (HEAP_DUMP_BLOCK*)malloc( sizeof(HEAP_DUMP_BLOCK)*blockCount);
    if (pHeapDumpBuffer == NULL)
        return FALSE;

    //
    // Buffer List
    //
    dprintf("lw: Block List Forward:\r\n");

    if (OwnerFilter == 'EERF' || OwnerFilter == 'FREE')
        OwnerFilter = FREE_BLOCK;

    if (heapBuffer.pBlockList)
    {
        if (dumpHeap_ReadBlocks_R70 (pHeapDumpBuffer, (LwU32)(LwUPtr)heapBuffer.pBlockList, 0, &blockCount, &freeVid) != LW_OK)
        {
            free(pHeapDumpBuffer);
            return FALSE;
        }

        dumpHeap_AllBlocks (pHeapDumpBuffer, blockCount, SortOption,  OwnerFilter);
    }

    //
    // Block List
    //
    dprintf("lw: FREE Block List Forward:\r\n");
    freeVid  = 0;

    if (heapBuffer.pFreeBlockList)
    {
        if (dumpHeap_ReadBlocks_R70 (pHeapDumpBuffer, (LwU32)(LwUPtr)heapBuffer.pFreeBlockList, 1, &blockCount, &freeVid) != LW_OK)
        {
            free(pHeapDumpBuffer);
            return FALSE;
        }

        dumpHeap_FreeBlocks (pHeapDumpBuffer, blockCount, SortOption,  OwnerFilter);

        dprintf("\n");
        dprintf("lw: \tThe number of free blocks = %d, Callwlated free video memory = 0x%x\n", blockCount, freeVid);
    }

    free(pHeapDumpBuffer);

    return TRUE;
}

//-----------------------------------------------------
// dumpPMA
//
//-----------------------------------------------------
void dumpPMA(void *pmaAddr)
{
    LW_STATUS       status;
    PMA             pmaBuffer;
    PMA_REGION_DESCRIPTOR pmaRegionBuffer;
    LwU64           bytesRead;
    LwU32           i;
    LwU32           regionCount;

    if (!pmaAddr)
    {
        dprintf("lw: Not a valid PMA object...\n");
        return;
    }

    status = readVirtMem((LwUPtr)pmaAddr, &pmaBuffer, sizeof(PMA), &bytesRead);
    if ((status != LW_OK) || (bytesRead != sizeof(PMA)))
    {
        dprintf("lw: Error parsing PMA object...\n");
        return;
    }

    regionCount = pmaBuffer.regSize;
    dprintf("lw: PMA region count = 0x%x\n", regionCount);
    if (regionCount == 0)
    {
        dprintf("lw: PMA not initialized...\n");
        return;
    }
    if (regionCount > PMA_REGION_SIZE)
    {
        dprintf("lw: Invalid PMA region count...\n");
        return;
    }

    dprintf("lw: PMA Scrubbing = %s\n", pmaBuffer.initScrubbing == PMA_SCRUB_INITIALIZE ? "INIT" :
                                       (pmaBuffer.initScrubbing == PMA_SCRUB_DONE ? "DONE" : "IN_PROGRESS"));

    for (i=0; i<regionCount; i++)
    {
        status = readVirtMem((LwUPtr)pmaBuffer.pRegDescriptors[i], &pmaRegionBuffer, sizeof(PMA_REGION_DESCRIPTOR), &bytesRead);
        if ((status != LW_OK) || (bytesRead != sizeof(PMA_REGION_DESCRIPTOR)))
        {
            dprintf("lw: Error parsing PMA region descriptor #%d...\n", i);
            return;
        }

        dprintf("lw: PMA region[%d]: 0x" LwU64_FMT "..0x" LwU64_FMT " perf=%d %s %s %s\n", i,
            pmaRegionBuffer.base,
            pmaRegionBuffer.limit,
            pmaRegionBuffer.performance,
            ( pmaRegionBuffer.bSupportCompressed ? "COMPRESSED  "  : "UNCOMPRESSED" ),
            ( pmaRegionBuffer.bSupportISO        ? "ISO    "       : "Non-ISO" ),
            ( pmaRegionBuffer.bProtected         ? "PROTECTED    " : "Non-PROTECTED" )
            );
    }
}

#else  // USERMODE follows

//-----------------------------------------------------
// dumpHeap -- uses the DUMP_HEAP call to get data.
//              used on platforms that don't have access
//              to resman memory space.
//
//-----------------------------------------------------
extern LwU32 hClient;
void dumpHeap(void)
{
    dprintf("dump heap not supported yet\n");
    return;
}

//-----------------------------------------------------
// dumpPMA --
//
//-----------------------------------------------------

void dumpPMA(void *pmaAddr)
{
    dprintf("dump PMA not supported yet\n");
    return;
}

#endif
