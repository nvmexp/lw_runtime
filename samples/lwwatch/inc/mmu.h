/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _MMU_H_
#define _MMU_H_

#include "mmu/gmmu_fmt.h"

//*****************************************************
//
// vgupta@lwpu.com - August 13 2004
// mmu.h - include file for mmu specific routines
//
//*****************************************************

#define PDE_MULTI_BIG_INDEX   0
#define PDE_MULTI_SMALL_INDEX 1

#define VMEM_MAX_MMU_FMTS 7

#define ILWALID_ADDRESS (0xF)

//
// VMEM TYPE data structures
//
typedef struct 
{
    LwU32 chId;                   // ChannelId of channel
    LwU32 rlId;                   // RunlistId of channel
} VMEM_INPUT_TYPE_CHANNEL;

typedef struct 
{
    LwU64 instPtr;                // Instance Pointer of a channel
    MEM_TYPE targetMemType;       // Memory Type of memory pointed by Instance pointer
} VMEM_INPUT_TYPE_INST;

typedef struct 
{
    LwU32 asId;                   // ASID
} VMEM_INPUT_TYPE_IOMMU;

typedef struct 
{
    LwU64 flaImbAddr;             // Fla Instance Memory Block Address
    MEM_TYPE targetMemType;       // Memory Type of memory pointed by Instance pointer
} VMEM_INPUT_TYPE_FLA;

typedef union 
{
    VMEM_INPUT_TYPE_CHANNEL ch;
    VMEM_INPUT_TYPE_INST    inst;
    VMEM_INPUT_TYPE_IOMMU   iommu;
    VMEM_INPUT_TYPE_FLA     fla;
} VMEM_INPUT_TYPE;

typedef LwU32 (*readFn_t)(LwU64 pa, void* buffer, LwU32 length);
typedef LwU32 (*writeFn_t)(LwU64 pa, void* buffer, LwU32 length);

//
// PDE data structure
//
typedef struct
{
    LwU32  w0;
    LwU32  w1;
} pde_entry_t;
typedef pde_entry_t PdeEntry;

//
// PTE data structure
//
typedef struct
{
    LwU32  w0;
    LwU32  w1;
} pte_entry_t;
typedef pte_entry_t PteEntry;

typedef struct VMemTableWalkEntries VMemTableWalkEntries;

/*!
 *  @typedef typedef struct VMemTableWalkInfo VMemTableWalkInfo
 *  @see struct VMemTableWalkInfo
 */
typedef struct VMemTableWalkInfo VMemTableWalkInfo;

/*!
 *  @typedef typedef struct VMemFmtPTE VMemFmtPTE
 *  @see struct VMemFmtPTE
 */
typedef struct VMemFmtPte VMemFmtPte;

/*!
 *  @typedef typedef struct VMemFmtPDE VMemFmtPDE
 *  @see struct VMemFmtPDE
 */
typedef struct VMemFmtPde VMemFmtPde;

/*!
 *  @typedef typedef struct VMemLayout VMemLayout
 *  @see struct VMemLayout
 */
typedef struct VMemLayout VMemLayout;

/*!
 *  @typedef typedef struct VMemSpace VMemSpace
 *  @see struct VMemSpace
 */
typedef struct VMemSpace VMemSpace;

/*!
 *  @typedef typedef struct InstBlock InstBlock
 *  @see struct InstBlock
 */
typedef struct InstBlock InstBlock;

typedef union MMU_FMT           MMU_FMT;
typedef union MMU_FMT_PDE_MULTI MMU_FMT_PDE_MULTI;
typedef union MMU_FMT_PDE       MMU_FMT_PDE;
typedef union MMU_FMT_PTE       MMU_FMT_PTE;

/*!
 *  Describes an instance block. The type of memory this is located can vary
 *  and is abstraced via the readFn.
 */
struct InstBlock
{
    LwU64       instBlockAddr;  /*!< physical address of base of instance block */
    readFn_t    readFn;         /*!< Specific to where memory block resides */
    writeFn_t   writeFn;        /*!< Specific to where memory block resides */
    MEM_TYPE    memType;        /*!< Memory type where the instance block resides*/
};

union MMU_FMT
{
    GMMU_FMT gmmu;
};

union MMU_FMT_PDE_MULTI
{
    GMMU_FMT_PDE_MULTI gmmu;
};

union MMU_FMT_PDE
{
    GMMU_FMT_PDE gmmu;
};

union MMU_FMT_PTE
{
    GMMU_FMT_PTE gmmu;
};

/*!
 *  Describes the paging structure of the virtual address space.
 */
struct VMemLayout
{
    MMU_FMT_LEVEL       fmtLevels[VMEM_MAX_MMU_FMTS];
    MMU_FMT             fmt;
    MMU_FMT_PDE_MULTI   pdeMulti;
    MMU_FMT_PDE         pde;
    MMU_FMT_PTE         pte;
    LW_FIELD_ENUM_ENTRY pdeApertures[GMMU_APERTURE__COUNT];
    LW_FIELD_ENUM_ENTRY pteApertures[GMMU_APERTURE__COUNT];

};

/*!
 *  Describes a Virtual Memory Space for a given entity (i.e. channel, PMU,
 *  etc.) via a pointer to the PDE table. The type of memory this is located
 *  can vary and is abstraced via readFn/writeFn. Any additional information could
 *  be added after instBlock. The instance block is included for the memory
 *  spaces that use instance blocks (lwrrently all of them) to get arbitrary
 *  data needed.
 */
struct VMemSpace
{
    LwU64       PdeBase;        /*!< Pointer to 0th PDE entry */
    readFn_t    readFn;         /*!< Specific to where memory PDE table resides */
    writeFn_t   writeFn;        /*!< Specific to where memory PDE table resides */

    InstBlock   instBlock;      /*!< Instance block */

    LwU32       pdeCount;       /*!< Number of PDEs in virtual space */
    LwU32       bigPageSize;    /*!< How big a large page is */
    VMemLayout  layout;         /*!< VA space paging structure */

    LwU32       class;          /*!< Used to distingush different memory spaces */
};

/*!
 *  Describes the format for a PDE.
 */
struct VMemFmtPde
{
    LwBool bMulti;
    union {
        MMU_FMT_PDE_MULTI multi;
        MMU_FMT_PDE       single;
    } fmts;
};

typedef LW_STATUS (*pteTableFunc_t)(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 va, LwU64 entryAddr, LwU32 level,
                                    LwU32 sublevel, LwU32 index, const MMU_FMT_LEVEL *pFmtLevel, const MMU_FMT_PTE *pFmtPte,
                                    GMMU_ENTRY_VALUE *pPte, LwBool valid, LwBool *pDone, void *arg);
typedef LW_STATUS (*pdeTableFunc_t)(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 va, LwU64 entryAddr, LwU32 level,
                                    LwU32 sublevel, LwU32 index, const MMU_FMT_LEVEL *pFmtLevel, const MMU_FMT_PDE *pFmtPde,
                                    GMMU_ENTRY_VALUE *pPde, LwBool valid, LwBool *pDone, void *arg);

/*!
 *  Describes what to do while walking the paging structure.
 */
struct VMemTableWalkInfo
{
    pteTableFunc_t pteFunc;
    pdeTableFunc_t pdeFunc;
    void          *pArg;
};

struct VMemTableWalkEntries
{
    LwU32            levels;
    LwU32            sublevels;
    LwBool           pdeValid[GMMU_FMT_MAX_LEVELS-1];
    LwBool           pteValid[MMU_FMT_MAX_SUB_LEVELS];
    LwU64            pdeAddr[GMMU_FMT_MAX_LEVELS-1];
    LwU64            pteAddr[MMU_FMT_MAX_SUB_LEVELS];
    GMMU_ENTRY_VALUE pdeEntry[GMMU_FMT_MAX_LEVELS-1];
    GMMU_ENTRY_VALUE pteEntry[MMU_FMT_MAX_SUB_LEVELS];
    GMMU_APERTURE    pdeAperture[GMMU_FMT_MAX_LEVELS-1];
    GMMU_APERTURE    pteAperture[MMU_FMT_MAX_SUB_LEVELS];
    MMU_FMT_LEVEL    fmtLevel[GMMU_FMT_MAX_LEVELS];
    MMU_FMT_PDE      fmtPde[GMMU_FMT_MAX_LEVELS-1];
    MMU_FMT_PTE      fmtPte[MMU_FMT_MAX_SUB_LEVELS];
};

/*!
 *  Describes the "types" of virtual memory that exist. Each type should have a
 *  matching function that will get the VMemSpace structure for that type. Also
 *  each type should have an entry in vmemGet().
 */
typedef enum VMemTypes VMemTypes;
enum VMemTypes
{
    VMEM_TYPE_CHANNEL,
    VMEM_TYPE_BAR1,
    VMEM_TYPE_BAR2,
    VMEM_TYPE_IFB,
    VMEM_TYPE_PMU,
    VMEM_TYPE_IOMMU,
    VMEM_TYPE_FLA,
    VMEM_TYPE_INST_BLK,
    VMEM_TYPE_ALL
};

typedef enum ApertureSize ApertureSize;
enum ApertureSize
{
    APERTURE_SIZE_NONE,
    APERTURE_SIZE_BIG,
    APERTURE_SIZE_SMALL
};

LW_STATUS vmemGet(VMemSpace *pVMemSpace, VMemTypes vMemType, VMEM_INPUT_TYPE *pArg);
LW_STATUS vmemTableWalk(VMemSpace *pVMemSpace, LwU64 va, VMemTableWalkInfo *pInfo, LwBool verbose);
LW_STATUS vmemVToP(VMemSpace *pVMemSpace, LwU64 va, LwU64 *pa, GMMU_APERTURE *aperture, LwBool dump);
void      vmemDoPdeDump(VMemSpace *pVMemSpace, LwU32 begin, LwU32 end);
LW_STATUS vmemDumpPde(VMemSpace *pVMemSpace, const GMMU_ENTRY_VALUE *pPde, LwU32 level);
LW_STATUS vmemDumpPte(VMemSpace *pVMemSpace, const GMMU_ENTRY_VALUE *pPte);
void      vmemDumpPa(LwU64 pa, GMMU_APERTURE aperture, LwU32 peerIndex);
LW_STATUS vmemPdeGetByIndex(VMemSpace *pVMemSpace, LwU32 index, GMMU_ENTRY_VALUE *pPde);
LW_STATUS vmemPdeGetByVa(VMemSpace *pVMemSpace, LwU64 va, LwU32 level, GMMU_ENTRY_VALUE *pPde);
LW_STATUS vmemPteGetByVa(VMemSpace *pVMemSpace, LwU64 va, VMemTableWalkEntries *pTableWalkEntries);
LW_STATUS vmemPteSetByVa(VMemSpace *pVMemSpace, LwU64 va, VMemTableWalkEntries *pTableWalkEntries);
LW_STATUS vmemPdeSetByIndex(VMemSpace *pVMemSpace, LwU32 index, GMMU_ENTRY_VALUE *pPde);
LW_STATUS vmemRead(VMemSpace  *pVMemSpace, LwU64 va, LwU32 length, void *pData);
LW_STATUS vmemWrite(VMemSpace  *pVMemSpace, LwU64 va, LwU32 length, void *pData);
LW_STATUS vmemFill(VMemSpace  *pVMemSpace, LwU64 va, LwU32 length, LwU32 data);
LW_STATUS vmemBeginBar1Mapping(VMemSpace *pVMemSpace, LwU64 va);
LW_STATUS vmemEndBar1Mapping(VMemSpace *pVMemSpace, LwU64 va);
LwU64     vmemGetPhysAddrFromPDE(VMemSpace *pVMemSpace, const MMU_FMT_PDE *pFmtPde, const GMMU_ENTRY_VALUE *pPde);
LwU64     vmemGetPhysAddrFromPTE(VMemSpace *pVMemSpace, const GMMU_ENTRY_VALUE *pPte);
LW_STATUS vmemGetPdeAperture(VMemSpace *pVMemSpace, LwU32 level, const GMMU_ENTRY_VALUE *pPde,
                             GMMU_APERTURE *pAperture, ApertureSize *pSize);
LW_STATUS vmemReadPhysical(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 pa, void* buffer, LwU32 length);
LW_STATUS vmemWritePhysical(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 pa, void* buffer, LwU32 length);

//
// GF100
//
LW_STATUS    gpuPhysMemCpy_GK104(LwU64 pa, void *buffer, GMMU_APERTURE aperture, LwU32 length);
LW_STATUS    readGpuVirtualAddr_GK104(LwU32 chId, LwU64 virtAddr, void* buffer, LwU32 length);

#define KB 1024
#define MB (1024*KB)

//
//  Page size definitions
//
#define VMEM_SMALL_PAGESIZE     (  4 * KB)
#define VMEM_BIG_PAGESIZE_64K   ( 64 * KB)
#define VMEM_BIG_PAGESIZE_128K  (128 * KB)
#define VMEM_HUGE_PAGESIZE_2M   (  2 * MB)
#define VMEM_HUGE_PAGESIZE_512M (512 * MB)

#include "g_mmu_hal.h"        // (rmconfig) public interface


#endif // _MMU_H_
