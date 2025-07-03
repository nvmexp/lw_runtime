/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2015 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _HEAP_H_
#define _HEAP_H_

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// heap.h
//
//*****************************************************

#include "os.h"
#include "lwstatus.h"

//
// defines
//
#define MEM_TYPE_MAX            15
#define NUM_MEM_TYPES           MEM_TYPE_MAX+1
#define FREE_BLOCK              0xFFFFFFFF

#define BANK_PLACEMENT_IMAGE              0
#define BANK_PLACEMENT_DEPTH              1
#define BANK_PLACEMENT_TEX_OVERLAY_FONT   2
#define BANK_PLACEMENT_OTHER              3
#define BANK_PLACEMENT_NUM_GROUPS          0x00000004

typedef struct OBJGPU *POBJGPU;
typedef struct OBJGPU  OBJGPU;

typedef LW_STATUS (*HeapDummyFunc)(POBJGPU, void *, LwU32);


// R60 Structures ------------------------------------
//
typedef struct OBJHEAP_R60  OBJHEAP_R60, *POBJHEAP_R60;

typedef LwU8            ODBCOMMON[0x80];    // Just for simple. Need to change it late if need full implementation.

typedef struct
{
    LwU32 offset;
    LwU32 size;
} MEMBANK_R60, *PMEMBANK_R60;

// size of the texture buffer array, when more than 4 clients detected,
// kill one of the clients listed in the client texture buffer 
#define MAX_TEXTURE_CLIENT_IDS  4

typedef struct
{
    LwU32 clientId;              // texture client id
    LwU32 refCount;              // how many textures have been allocated wrt this client
    LwU8 placementFlags;        // how texture is grown,
    BOOL mostRecentAllocatedFlag;   // most recently allocated client
} TEXINFO_R60;


typedef struct def_block_R60
{
    LwU8 growFlags;
    LwU8 reserved1; // pad to 64 bit
    LwU8 reserved2;
    LwU8 reserved3;
    LwU32 owner;
    LwU32 mhandle;
    LwU32 begin;
    LwU32 align;
    LwU32 end;
    LwU32 textureId;
    union
    {
        LwU32             type;
        struct def_block_R60 *prevFree;
    } u0;
    union
    {
        LwU32             hwres;
        struct def_block_R60 *nextFree;
    } u1;
    struct def_block_R60 *prev;
    struct def_block_R60 *next;
} MEMBLOCK_R60, *PMEMBLOCK_R60;

struct OBJHEAP_R60
{
    ODBCOMMON                  odbCommon;  // Must be first!

    // Special destructor for objheap (handled by fb)
    HeapDummyFunc              destruct;

    // Public heap interface methods
    HeapDummyFunc              heapInit;
    HeapDummyFunc              heapAlloc;
    HeapDummyFunc              heapReAlloc;
    HeapDummyFunc              heapFbSetAllocParameters;
    HeapDummyFunc              heapFree;
    HeapDummyFunc              heapPurge;
    HeapDummyFunc              heapRetain;
    HeapDummyFunc              heapInfo;
    HeapDummyFunc              heapInfoFreeBlocks;
    HeapDummyFunc              heapInfoTypeAllocBlocks;
    HeapDummyFunc              heapCompact;
    HeapDummyFunc              heapGetSize;
    HeapDummyFunc              heapGetFree;
    HeapDummyFunc              heapGetBase;
    HeapDummyFunc              heapGetBlockHandle;
    HeapDummyFunc              heapGetNumBlocks;
    HeapDummyFunc              heapGetBlockInfo;
    HeapDummyFunc              heapNotifyBeginLog;
    HeapDummyFunc              heapReleaseReacquireCompr;
    HeapDummyFunc              heapModifyDeferredTiles;

    // private data
    LwU8      *base;
    LwU32      total;
    LwU32      free;
    PMEMBLOCK_R60  pBlockList;
    PMEMBLOCK_R60  pFreeBlockList;
    PMEMBLOCK_R60  pPrimaryBlocks[2];
    LwU32      numBanks;
    LwU32      memHandle;
    LwU32      numBlocks;
    TEXINFO_R60    textureData[MAX_TEXTURE_CLIENT_IDS];

    //
    // During heapCreate, set up 2 arrays of bank placement then grow direction 
    // for images, depth, everything else under 2 cases: primary surface allocated
    // in low mem, or in high mem.
    // Each of these arrays is allocated to 4 LwU32s, one for each allocation group
    // Each byte of the LwU32s is assigned like this:
    // Bit  7 = grow direction if this bank placement fails (only matters for last
    //          bank tried and failed that has this bit set/unset)
    // Bit  6 = grow direction inside a bank
    // Bits 5..0 = bank to try allocation in. If this is the last bank to try, and it
    //             fails, allocate anywhere using grow direction
    // The LwU32s are assigned like this:
    // LwU32 0 = image placement
    // LwU32 1 = depth placement
    // LwU32 2 = texture/overlay/font placement
    // LwU32 3 = other / everything else placement
    // If multiple banks should be tried before falling back to any direction
    // allocation, fill in the other bytes of the LwU32 for each allocation type
    // during heapCreate for that allocation type. The LwU32 is searched
    // from byte 0 to MEM_NUM_BANKS_TO_TRY - 1 for banks to try before giving up
    // and using grow direction to allocate.
    //
    LwU32 primaryLowBankPlacement[BANK_PLACEMENT_NUM_GROUPS];
    LwU32 primaryHighBankPlacement[BANK_PLACEMENT_NUM_GROUPS];
    MEMBANK_R60    Bank[1];  // must be last in this struct
};

//
// End R60 Structures --------------------------------

// R65 Structures ------------------------------------
//
typedef MEMBANK_R60 MEMBANK_R65, *PMEMBANK_R65;
typedef TEXINFO_R60 TEXINFO_R65;

typedef struct  def_block_R65
{
    LwU8 growFlags;
    LwU8 reserved1; // pad to 64 bit
    LwU8 reserved2;
    LwU8 reserved3;
    LwU32 owner;
    LwU32 mhandle;
    LwU32 begin;
    LwU32 align;
    LwU32 end;
    LwU32 textureId;
    LwU32 retAttr;   // actual attributes of surface allocated
    LwU32 pitch;     // allocated surface pitch, needed for realloc
    LwU32 height;    // allocated surface height, needed for realloc
    union
    {
        LwU32     type;
        struct  def_block_R65 *prevFree;
    } u0;
    union
    {
        LwU32     hwres;
        struct  def_block_R65 *nextFree;
    } u1;
    struct  def_block_R65 *prev;
    struct  def_block_R65 *next;
} MEMBLOCK_R65, *PMEMBLOCK_R65;

//
// End R65 Structures --------------------------------

// R70 Structures ------------------------------------
//

typedef struct OBJHEAP_R70  OBJHEAP_R70, *POBJHEAP_R70;
typedef MEMBANK_R60 MEMBANK_R70, *PMEMBANK_R70;
typedef TEXINFO_R60 TEXINFO_R70;

typedef struct def_block_R70

{
    LwU8 growFlags;
    LwU8 reserved1; // pad to 64 bit
    LwU8 reserved2;
    LwU8 reserved3;
    LwU32 owner;
    LwU32 mhandle;
    LwU32 begin;
    LwU32 align;
    LwU32 end;
    LwU32 textureId;
    LwU32 format;
    LwU32 retAttr;   // actual attributes of surface allocated
    LwU32 pitch;     // allocated surface pitch, needed for realloc
    LwU32 height;    // allocated surface height, needed for realloc
    union
    {
        LwU32     type;
        struct def_block_R70 *prevFree;
    } u0;
    union
    {
        LwU32     hwres;
        struct def_block_R70 *nextFree;
    } u1;
    struct def_block_R70 *prev;
    struct def_block_R70 *next;
} MEMBLOCK_R70, *PMEMBLOCK_R70;

struct OBJHEAP_R70
{
    ODBCOMMON                  odbCommon;  // Must be first!

    // Special destructor for objheap (handled by fb)
    HeapDummyFunc              destruct;

    // Public heap interface methods
    HeapDummyFunc              heapInit;
    HeapDummyFunc              heapAlloc;
    HeapDummyFunc              heapReAlloc;
    HeapDummyFunc              heapFbSetAllocParameters;
    HeapDummyFunc              heapFree;
    HeapDummyFunc              heapPurge;
    HeapDummyFunc              heapRetain;
    HeapDummyFunc              heapInfo;
    HeapDummyFunc              heapInfoFreeBlocks;
    HeapDummyFunc              heapInfoTypeAllocBlocks;
    HeapDummyFunc              heapCompact;
    HeapDummyFunc              heapGetSize;
    HeapDummyFunc              heapGetFree;
    HeapDummyFunc              heapGetBase;
    HeapDummyFunc              heapGetBlock;
    HeapDummyFunc              heapGetBlockHandle;
    HeapDummyFunc              heapGetNumBlocks;
    HeapDummyFunc              heapGetBlockInfo;
    HeapDummyFunc              heapNotifyBeginLog;
    HeapDummyFunc              heapReleaseReacquireCompr;
    HeapDummyFunc              heapModifyDeferredTiles;

    // private data
    LwU8      *base;
    LwU32      total;
    LwU32      free;
    PMEMBLOCK_R70  pBlockList;
    PMEMBLOCK_R70  pFreeBlockList;
    PMEMBLOCK_R70  pPrimaryBlocks[2];
    LwU32      numBanks;
    LwU32      memHandle;
    LwU32      numBlocks;
    TEXINFO_R70    textureData[MAX_TEXTURE_CLIENT_IDS];

    //
    // During heapCreate, set up 2 arrays of bank placement then grow direction 
    // for images, depth, everything else under 2 cases: primary surface allocated
    // in low mem, or in high mem.
    // Each of these arrays is allocated to 4 LwU32s, one for each allocation group
    // Each byte of the LwU32s is assigned like this:
    // Bit  7 = grow direction if this bank placement fails (only matters for last
    //          bank tried and failed that has this bit set/unset)
    // Bit  6 = grow direction inside a bank
    // Bits 5..0 = bank to try allocation in. If this is the last bank to try, and it
    //             fails, allocate anywhere using grow direction
    // The LwU32s are assigned like this:
    // LwU32 0 = image placement
    // LwU32 1 = depth placement
    // LwU32 2 = texture/overlay/font placement
    // LwU32 3 = other / everything else placement
    // If multiple banks should be tried before falling back to any direction
    // allocation, fill in the other bytes of the LwU32 for each allocation type
    // during heapCreate for that allocation type. The LwU32 is searched
    // from byte 0 to MEM_NUM_BANKS_TO_TRY - 1 for banks to try before giving up
    // and using grow direction to allocate.
    //
    LwU32 primaryLowBankPlacement[BANK_PLACEMENT_NUM_GROUPS];
    LwU32 primaryHighBankPlacement[BANK_PLACEMENT_NUM_GROUPS];
    MEMBANK_R70    Bank[1];  // must be last in this struct
};

//
// End R70 Structures --------------------------------
//

//
// heap routines - heap.c
//
#define OWNER_UP    1
#define OWNER_DOWN  2
#define TYPE_UP     3
#define TYPE_DOWN   4
#define ADDR_UP     5
#define ADDR_DOWN   6
#define SIZE_UP     7
#define SIZE_DOWN   8

void    dumpHeap_R60(void *heapAddr, LwU32 SortOption, LwU32 OwnerFilter);
LW_STATUS    dumpHeap_R70(void *heapAddr, LwU32 SortOption, LwU32 OwnerFilter);
void    dumpHeap(void);

void    dumpPMA(void *pmaAddr);

#endif // _HEAP_H_
