/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// vgupta@lwpu.com - August 13 2004
// vmem.c - page table routines
//
//*****************************************************

//
// Includes
//
#include "chip.h"
#include "fb.h"
#include "mmu.h"
#include "virtOp.h"
#include "vmem.h"

#include "class/cle3f1.h"      // TEGRA_VASPACE_A

#define INDENT(level)                                  \
    do {                                               \
        LwU32 _level;                                  \
        for (_level = 0; _level < level + 1; _level++) \
            dprintf("  ");                             \
    } while (0)

#define PRINT_FIELD_32(fmt, fmtPte, field, pte)                           \
    do {                                                                  \
        if (lwFieldIsValid32(&(fmtPte)->fld##field))                      \
            dprintf(fmt, lwFieldGet32(&(fmtPte)->fld##field, (pte)->v8)); \
    } while (0)

#define PRINT_FIELD_BOOL(fmt, fmtPte, field, pte)                           \
    do {                                                                    \
        if (lwFieldIsValid32(&(fmtPte)->fld##field.desc))                   \
            dprintf(fmt, lwFieldGetBool(&(fmtPte)->fld##field, (pte)->v8)); \
    } while (0)

LW_STATUS fbRead_GK104(LwU64 offset, void* buffer, LwU32 length);

static LW_STATUS _vmemTableWalk(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU32 level, LwU32 sublevel,
                                const MMU_FMT_LEVEL *pFmtLevel, LwU64 base, LwU64 va, LwBool *pDone,
                                VMemTableWalkInfo *pInfo, LwBool verbose);

static LW_STATUS _pteGetByVaPteFunc(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 va, LwU64 entryAddr, LwU32 level,
                                    LwU32 sublevel, LwU32 index, const MMU_FMT_LEVEL *pFmtLevel, const MMU_FMT_PTE *pFmtPte,
                                    GMMU_ENTRY_VALUE *pPte, LwBool valid, LwBool *pDone, void *pArg);
static LW_STATUS _pteGetByVaPdeFunc(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 va, LwU64 entryAddr, LwU32 level,
                                    LwU32 sublevel, LwU32 index, const MMU_FMT_LEVEL *pFmtLevel, const MMU_FMT_PDE *pFmtPte,
                                    GMMU_ENTRY_VALUE *pPde, LwBool valid, LwBool *pDone, void *pArg);

static LW_STATUS _pdeGetByVaPteFunc(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 va, LwU64 entryAddr, LwU32 level,
                                    LwU32 sublevel, LwU32 index, const MMU_FMT_LEVEL *pFmtLevel, const MMU_FMT_PTE *pFmtPte,
                                    GMMU_ENTRY_VALUE *pPte, LwBool valid, LwBool *pDone, void *pArg);
static LW_STATUS _pdeGetByVaPdeFunc(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 va, LwU64 entryAddr, LwU32 level,
                                    LwU32 sublevel, LwU32 index, const MMU_FMT_LEVEL *pFmtLevel, const MMU_FMT_PDE *pFmtPte,
                                    GMMU_ENTRY_VALUE *pPde, LwBool valid, LwBool *pDone, void *pArg);

static LW_STATUS _getMappingPteFunc(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 va, LwU64 entryAddr, LwU32 level,
                                    LwU32 sublevel, LwU32 index, const MMU_FMT_LEVEL *pFmtLevel, const MMU_FMT_PTE *pFmtPte,
                                    GMMU_ENTRY_VALUE *pPte, LwBool valid, LwBool *pDone, void *pArg);
static LW_STATUS _getMappingPdeFunc(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 va, LwU64 entryAddr, LwU32 level,
                                    LwU32 sublevel, LwU32 index, const MMU_FMT_LEVEL *pFmtLevel, const MMU_FMT_PDE *pFmtPte,
                                    GMMU_ENTRY_VALUE *pPde, LwBool valid, LwBool *pDone, void *pArg);

static LW_STATUS _VToPPteFunc(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 va, LwU64 entryAddr, LwU32 level,
                              LwU32 sublevel, LwU32 index, const MMU_FMT_LEVEL *pFmtLevel, const MMU_FMT_PTE *pFmtPte,
                              GMMU_ENTRY_VALUE *pPte, LwBool valid, LwBool *pDone, void *pArg);
static LW_STATUS _VToPPdeFunc(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 va, LwU64 entryAddr, LwU32 level,
                              LwU32 sublevel, LwU32 index, const MMU_FMT_LEVEL *pFmtLevel, const MMU_FMT_PDE *pFmtPte,
                              GMMU_ENTRY_VALUE *pPde, LwBool valid, LwBool *pDone, void *pArg);

static LW_STATUS _bar1MappingPteFunc(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 va, LwU64 entryAddr, LwU32 level,
                                     LwU32 sublevel, LwU32 index, const MMU_FMT_LEVEL *pFmtLevel, const MMU_FMT_PTE *pFmtPte,
                                     GMMU_ENTRY_VALUE *pPte, LwBool valid, LwBool *pDone, void *pArg);
static LW_STATUS _bar1MappingPdeFunc(VMemSpace *pVMemSpace, GMMU_APERTURE aperture, LwU64 va, LwU64 entryAddr, LwU32 level,
                                     LwU32 sublevel, LwU32 index, const MMU_FMT_LEVEL *pFmtLevel, const MMU_FMT_PDE *pFmtPte,
                                     GMMU_ENTRY_VALUE *pPde, LwBool valid, LwBool *pDone, void *pArg);


static LW_STATUS _vmemGetMapping(VMemSpace *pVMemSpace, LwU64 va, VMemTableWalkEntries *pTableWalkEntries);
static LW_STATUS _vmemDupMapping(VMemSpace *pVMemTargetSpace, VMemTableWalkEntries *pTargetWalkEntries,
                                 VMemSpace *pVMemSourceSpace, VMemTableWalkEntries *pSourceWalkEntries);
static LW_STATUS _vmemSetMapping(VMemSpace *pVMemSpace, VMemTableWalkEntries *pTableWalkEntries);

static LwU32     _vmemPageSize(VMemTableWalkEntries *pTableWalkEntries);

static const char *decodeSize[]     = {"full", "1/2", "1/4", "1/8"};

static const char *decodeAperture(GMMU_APERTURE aperture)
{
    switch (aperture)
    {
        case GMMU_APERTURE_VIDEO:       return "video";
        case GMMU_APERTURE_PEER:        return "peer";
        case GMMU_APERTURE_SYS_NONCOH:  return "sysnoncoh";
        case GMMU_APERTURE_SYS_COH:     return "syscoh";
        default:                        return "invalid";
    }
}

/*!
 *  Provides the ability to ask for a specific type of virtual memory in a
 *  generic way.
 *
 *  @param[out] pVmemSpace  Virtual memory space structure to fill in.
 *  @param[in]  vMemType    Type of virtual memory.
 *  @param[in]  pArg        Arbitrary argument used by various types.
 *
 *  @return LW_OK on success.
 */
LW_STATUS
vmemGet
(
    VMemSpace  *pVMemSpace,
    VMemTypes   vMemType,
    VMEM_INPUT_TYPE *pArg
)
{
    LW_STATUS status;

    switch (vMemType)
    {
        case VMEM_TYPE_BAR1:
            CHECK(pVmem[indexGpu].vmemGetBar1(pVMemSpace));
            break;
        case VMEM_TYPE_BAR2:
            CHECK(pVmem[indexGpu].vmemGetBar2(pVMemSpace));
            break;
        case VMEM_TYPE_IFB:
            CHECK(pVmem[indexGpu].vmemGetIfb(pVMemSpace));
            break;
        case VMEM_TYPE_CHANNEL:
            CHECK(pVmem[indexGpu].vmemGetByChId(pVMemSpace, (VMEM_INPUT_TYPE_CHANNEL*)pArg));
            break;
        case VMEM_TYPE_PMU:
            CHECK(pVmem[indexGpu].vmemGetPmu(pVMemSpace));
            break;
        case VMEM_TYPE_IOMMU:
            CHECK(pVmem[indexGpu].vmemGetByAsId(pVMemSpace, (VMEM_INPUT_TYPE_IOMMU*)pArg));
            break;
        case VMEM_TYPE_FLA:
            CHECK(pVmem[indexGpu].vmemGetFla(pVMemSpace, (VMEM_INPUT_TYPE_FLA*)pArg));
            break;
        case VMEM_TYPE_INST_BLK:
            CHECK(pVmem[indexGpu].vmemGetByInstPtr(pVMemSpace, (VMEM_INPUT_TYPE_INST*)pArg));
            break;
        default:
            return LW_ERR_GENERIC;
    }

    return pVmem[indexGpu].vmemInitLayout(pVMemSpace);
}


//----------------------------------------------------------
// pteValidate
//
//----------------------------------------------------------
LwU32 pteValidate_STUB( void )
{
    dprintf("lw: %s - Unsupported Chip...\n", __FUNCTION__);
    return 0;
}

/*!
 *  Traverse the paging structure for the given va in the vmemspace.
 *
 *  Algorithm:
 *
 * 1. Callwlate index of PDE by given va and PDB.
 * 2a. Apply the caller specified pde function.
 * 2b. Return if specified by the pde function.
 * 3. Callwlate the next base address using PDB (possibly
 *    checking multiple formats and index.
 * 4. If it's a page directory, go back to step 1.
 * 5. Extract Page Table Entry from given PDE and va
 * 6. Extract base address for physical page from given PTE
 *    and apply the caller specified pte function.
 *
 *  @param[in]     pVMemSpace  Virtual memory space.
 *  @param[in]     va          Virtual address.
 *  @param[in]     pteFunc     Function to handle page table entries
 *  @param[in]     pdeFunc     Function to handle page directory entry
 *  @param[in/out] arg         Pointer to argument to pass to the pte and pde functions
 *
 *  @return LW_ERR_GENERIC on error, LW_OK on success.
 */
LW_STATUS
vmemTableWalk
(
    VMemSpace         *pVMemSpace,
    LwU64              va,
    VMemTableWalkInfo *pInfo,
    LwBool             verbose
)
{
    GMMU_APERTURE        aperture;
    LwBool               done = FALSE;

    if (!pVMemSpace)
    {
        return LW_ERR_GENERIC;
    }
    // Check if any bits above the highest (inclusive) are set
    if (va > mmuFmtLevelVirtAddrMask(pVMemSpace->layout.fmtLevels))
    {
        if (verbose)
        {
            dprintf("*******************************************************************\n");
            dprintf("*\n");
            dprintf("* WARNING: Some versions of WinDbg/Windows will sign extend the VA!\n");
            dprintf("*\n");
            dprintf("*******************************************************************\n");
            dprintf("VA larger than address space\n");
        }
        return LW_ERR_GENERIC;
    }
    aperture = pVmem[indexGpu].vmemGetPDBAperture(pVMemSpace);

    if (verbose)
    {
        dprintf("PDB: " LwU64_FMT " (%s)\n", pVMemSpace->PdeBase, decodeAperture(aperture));
    }
    return _vmemTableWalk(pVMemSpace, aperture, 0, 0, pVMemSpace->layout.fmtLevels,
                          pVMemSpace->PdeBase, va, &done, pInfo, verbose);
}

typedef struct
{
    LwBool        valid;
    LwU64         pa;
    GMMU_APERTURE aperture;
    LwU32         peerIndex;
    LwBool        dump;
} VToPArg;

/*!
 *  Walks the page table to colwert a virtual address to a physical one.
 *
 *  @param[in]  pVmemSpace  Virtual memory space structure to fill in.
 *  @param[in]  va          Virtual address.
 *  @param[out] pPa         Physical address.
 *  @param[out] pAperture   Aperture.
 *
 *  @return LW_OK on success.
 */
LW_STATUS
vmemVToP
(
    VMemSpace     *pVMemSpace,
    LwU64          va,
    LwU64         *pPa,
    GMMU_APERTURE *pAperture,
    LwBool         dump
)
{
    VToPArg           arg  = {FALSE, 0, GMMU_APERTURE_ILWALID, dump};
    VMemTableWalkInfo info = {0};
    LwU64             smmuVa;
    LW_STATUS         status;

    if (!pVMemSpace)
    {
        return LW_ERR_GENERIC;
    }
    info.pteFunc = _VToPPteFunc;
    info.pdeFunc = NULL;
    info.pArg    = &arg;

    CHECK(vmemTableWalk(pVMemSpace, va, &info, dump));

    /* In case of "CheetAh+Android", the existing code crashes
     * in its current shape and needs debugging to get complete
     * functionality on android.print only the GPU PA. As of
     * now, we are not returning the actual physical address.
     */
    if (IsTegra() && IsAndroid())
    {
        dprintf("GPU physical address :"LwU64_FMT " \n", arg.pa);
        return LW_OK;
    }
    if (pMmu[indexGpu].mmuIsGpuIommuMapped(arg.pa, &smmuVa))
    {
        return pVmem[indexGpu].vmemVToP(pVMemSpace, smmuVa, pPa, pAperture);
    }
    // Return physical address if requested
    if (pPa)
    {
        *pPa = arg.pa;
    }
    // Return aperture if requested
    if (pAperture)
    {
        *pAperture = arg.aperture;
    }
    // Dump physical address if requested
    if (dump)
    {
        vmemDumpPa(arg.pa, arg.aperture, arg.peerIndex);
        dprintf("\n");
    }
    return LW_OK;
}

/*!
 *  Retrieves a pde at a specified level.
 *
 *  @param[in]  pVmemSpace  Virtual memory space structure to fill in.
 *  @param[in]  va          Virtual address.
 *  @param[in]  level       The page directory level.
 *  @param[out] pPde        Page directory entry.
 *
 *  @return LW_OK on success.
 */

typedef struct
{
    LwU32             level;
    GMMU_ENTRY_VALUE *pPde;
} PdeGetByVaArg;

LW_STATUS
vmemPdeGetByVa
(
    VMemSpace        *pVMemSpace,
    LwU64             va,
    LwU32             level,
    GMMU_ENTRY_VALUE *pPde
)
{
    PdeGetByVaArg     arg  = {0};
    VMemTableWalkInfo info = {0};

    if (!pVMemSpace || !pPde)
    {
        return LW_ERR_GENERIC;
    }
    arg.level = level;
    arg.pPde  = pPde;

    info.pteFunc = _pdeGetByVaPteFunc;
    info.pdeFunc = _pdeGetByVaPdeFunc;
    info.pArg    = &arg;

    return vmemTableWalk(pVMemSpace, va, &info, LW_FALSE);
}

void
vmemDumpPa
(
    LwU64         pa,
    GMMU_APERTURE aperture,
    LwU32         peerIndex
)
{
    dprintf(LwU64_FMT " ", pa);

    switch (aperture)
    {
        case GMMU_APERTURE_ILWALID:
        case GMMU_APERTURE_VIDEO:
        case GMMU_APERTURE_SYS_COH:
        case GMMU_APERTURE_SYS_NONCOH:
            dprintf("(%s)", decodeAperture(aperture));
            break;
        case GMMU_APERTURE_PEER:
            dprintf("(%s #%d)", decodeAperture(aperture), peerIndex);
            break;
        default:
            dprintf("(Bad aperture encoding)");
            break;
    }
}

LW_STATUS
vmemDumpPte
(
    VMemSpace              *pVMemSpace,
    const GMMU_ENTRY_VALUE *pPte
)
{
    const MMU_FMT_PTE *pFmtPte;
    const GMMU_FMT_PTE *pFmt;

    if (!pVMemSpace || !pPte)
    {
        return LW_ERR_GENERIC;
    }

    pFmtPte = pVmem[indexGpu].vmemGetPTEFmt(pVMemSpace);
    if (!pFmtPte)
    {
        return LW_ERR_GENERIC;
    }

    pFmt = &pFmtPte->gmmu;

    PRINT_FIELD_BOOL("Vld=%d, ",              pFmt, Valid,           pPte);
    PRINT_FIELD_BOOL("Priv=%d, ",             pFmt, Privilege,       pPte);
    PRINT_FIELD_BOOL("RO=%d, ",               pFmt, ReadOnly,        pPte);
    PRINT_FIELD_BOOL("RD=%d, ",               pFmt, ReadDisable,     pPte);
    PRINT_FIELD_BOOL("WD=%d, ",               pFmt, WriteDisable,    pPte);
    PRINT_FIELD_BOOL("Enc=%d, ",              pFmt, Encrypted,       pPte);
    PRINT_FIELD_BOOL("Vol=%d, ",              pFmt, Volatile,        pPte);
    PRINT_FIELD_BOOL("Lock=%d, ",             pFmt, Locked,          pPte);
    PRINT_FIELD_BOOL("AtomDis=%d, ",          pFmt, AtomicDisable,   pPte);
    PRINT_FIELD_32("Kind=%d, ",               pFmt, Kind,            pPte);
    PRINT_FIELD_32("CompTagLine=0x%x, ",      pFmt, CompTagLine,     pPte);
    PRINT_FIELD_32("CompTagSubIndex=0x%x, ",  pFmt, CompTagSubIndex, pPte);
    pVmem[indexGpu].vmemDumpPtePcf(pFmt, pPte);

    dprintf("\n");

    return LW_OK;
}

LW_STATUS
vmemDumpPde
(
    VMemSpace              *pVMemSpace,
    const GMMU_ENTRY_VALUE *pPde,
    LwU32                   level
)
{
    VMemFmtPde          fmtPde;
    GMMU_FMT_PDE_MULTI *pPdeMulti;
    GMMU_FMT_PDE       *pFmt;
    GMMU_APERTURE       aperture;
    LwU64               pa;

    if (!pPde)
    {
        return LW_ERR_GENERIC;
    }
    if (pVmem[indexGpu].vmemGetPDEFmt(pVMemSpace, &fmtPde, level) != LW_OK)
    {
        return LW_ERR_GENERIC;
    }

    pPdeMulti = &fmtPde.fmts.multi.gmmu;
    pFmt      = &fmtPde.fmts.single.gmmu;

    if (fmtPde.bMulti)
    {
        LW_STATUS status;

        CHECK(vmemGetPdeAperture(pVMemSpace, level, pPde, &aperture, NULL));
        if (aperture == GMMU_APERTURE_ILWALID)
        {
            dprintf("Invalid\n");
            return LW_OK;
        }
        if (lwFieldIsValid32(&pPdeMulti->fldSizeRecipExp))
        {
            dprintf(" %s\n", decodeSize[lwFieldGet32(&pPdeMulti->fldSizeRecipExp, pPde->v8)]);
        }
        return LW_OK;
    }
    aperture = gmmuFieldGetAperture(&pFmt->fldAperture, pPde->v8);
    pa       = vmemGetPhysAddrFromPDE(pVMemSpace, (MMU_FMT_PDE*)pFmt, pPde);

    vmemDumpPa(pa, aperture, 0);
    dprintf(" ");
    pVmem[indexGpu].vmemDumpPdeFlags(pFmt, pPde);
    dprintf("\n");

    return LW_OK;
}

/*!
 *  Dump out PDEs for a given channel. This function is the interface to exts.c.
 *
 *  @param[in]  pVMemSpace  vmemspace for the target channel/BAR etc.
 *  @param[in]  pdb         Page directory base
 *  @param[in]  level       Page directory level.
 *  @param[in]  begin       Starting PDE index.
 *  @param[in]  end         Ending PDE index.
 */
void
vmemDoPdeDump
(
    VMemSpace *pVMemSpace,
    LwU32      begin,
    LwU32      end
)
{
    const MMU_FMT_LEVEL *pFmt;
    LwU32                maxEntries;
    LwU32                i;
    GMMU_APERTURE        aperture;

    if (!pVMemSpace)
    {
        return;
    }

    pFmt       = pVMemSpace->layout.fmtLevels;
    maxEntries = mmuFmtLevelEntryCount(pFmt);
    aperture   = pVmem[indexGpu].vmemGetPDBAperture(pVMemSpace);

    if (begin >= maxEntries)
    {
        dprintf("lw: begin (%d) greater than max pde index(%d)\n",
                begin, maxEntries - 1);
        return;
    }

    if (end >= maxEntries)
    {
        dprintf("lw: end (%d) greater than max pde index(%d)\n",
                end, maxEntries);
        return;
    }

    for (i = begin; i <= end; i++)
    {
        LwU64            entryAddr = pVMemSpace->PdeBase + (i * pFmt->entrySize);
        GMMU_ENTRY_VALUE pde;

        if (osCheckControlC())
        {
            break;
        }

        dprintf("PDE[0x%x]: ", i);
        if (vmemReadPhysical(pVMemSpace, aperture, entryAddr, pde.v8, pFmt->entrySize) != LW_OK)
        {
            dprintf("Error reading\n");
            continue;
        }

        vmemDumpPde(pVMemSpace, &pde, 0);
    }
}

typedef struct
{
    VMemTableWalkEntries *pTableWalkEntries;
} PteGetByVaArg;

LW_STATUS
vmemPteGetByVa
(
    VMemSpace            *pVMemSpace,
    LwU64                 va,
    VMemTableWalkEntries *pTableWalkEntries
)
{
    VMemTableWalkInfo info = {0};
    PteGetByVaArg     arg  = {0};

    if (!pVMemSpace || !pTableWalkEntries)
    {
        return LW_ERR_GENERIC;
    }
    memset(pTableWalkEntries, 0, sizeof(VMemTableWalkEntries));

    arg.pTableWalkEntries = pTableWalkEntries;

    info.pteFunc = _pteGetByVaPteFunc;
    info.pdeFunc = _pteGetByVaPdeFunc;
    info.pArg    = &arg;

    return vmemTableWalk(pVMemSpace, va, &info, LW_FALSE);
}

LW_STATUS
vmemPteSetByVa
(
    VMemSpace            *pVMemSpace,
    LwU64                 va,
    VMemTableWalkEntries *pTableWalkEntries
)
{
    LwU64           base;
    LwU32           level;
    GMMU_APERTURE   aperture;
    LW_STATUS       status;

    if (!pVMemSpace || !pTableWalkEntries)
    {
        return LW_ERR_GENERIC;
    }
    // Get initial base (PDB) and aperture
    base     = pVMemSpace->PdeBase;
    aperture = pVmem[indexGpu].vmemGetPDBAperture(pVMemSpace);

    // Walk thru all page table levels (PDE/PTE)
    for (level = 0; level < pTableWalkEntries->levels; level++)
    {
        MMU_FMT_LEVEL    *pFmtLevel = &pTableWalkEntries->fmtLevel[level];
        GMMU_ENTRY_VALUE *pEntry    = &pTableWalkEntries->pteEntry[level];
        MMU_FMT_PDE      *pFmtPde   = &pTableWalkEntries->fmtPde[level];
        LwU32             index     = mmuFmtVirtAddrToEntryIndex(pFmtLevel, va);
        LwU64             entryAddr = base + (index * pFmtLevel->entrySize);

        CHECK(vmemWritePhysical(pVMemSpace, aperture, entryAddr, pEntry, pFmtLevel->entrySize));

        // Update base and aperture if not PTE level
        if (level < (pTableWalkEntries->levels - 1))
        {
            base     = vmemGetPhysAddrFromPDE(pVMemSpace, pFmtPde, pEntry);
            aperture = gmmuFieldGetAperture(&pFmtPde->gmmu.fldAperture, pEntry->v8);
        }
    }
    return pVmem[indexGpu].vmemIlwalidatePDB(pVMemSpace);
}

LW_STATUS
vmemPdeGetByIndex
(
    VMemSpace        *pVMemSpace,
    LwU32             index,
    GMMU_ENTRY_VALUE *pPde
)
{
    LwU64                entryAddr;
    const MMU_FMT_LEVEL *pFmt;
    GMMU_APERTURE        aperture;

    if (!pVMemSpace || !pPde)
    {
        return LW_ERR_GENERIC;
    }
    if (pVMemSpace->class == TEGRA_VASPACE_A)
    {
        return pVmem[indexGpu].vmemPdeGetByIndex(pVMemSpace, index, pPde);
    }
    pFmt = pVMemSpace->layout.fmtLevels;
    if (index >= mmuFmtLevelEntryCount(pFmt))
    {
        return LW_ERR_GENERIC;
    }
    aperture  = pVmem[indexGpu].vmemGetPDBAperture(pVMemSpace);
    entryAddr = pVMemSpace->PdeBase + (index * pFmt->entrySize);

    return vmemReadPhysical(pVMemSpace, aperture, entryAddr, pPde->v8, pFmt->entrySize);
}

LW_STATUS
vmemPdeSetByIndex
(
    VMemSpace        *pVMemSpace,
    LwU32             index,
    GMMU_ENTRY_VALUE *pPde
)
{
    LwU64                entryAddr;
    const MMU_FMT_LEVEL *pFmt;
    LW_STATUS            status;
    GMMU_APERTURE        aperture;

    if (!pVMemSpace || !pPde)
    {
        return LW_ERR_GENERIC;
    }
    pFmt = pVMemSpace->layout.fmtLevels;
    if (index >= mmuFmtLevelEntryCount(pFmt))
    {
        return LW_ERR_GENERIC;
    }
    aperture  = pVmem[indexGpu].vmemGetPDBAperture(pVMemSpace);
    entryAddr = pVMemSpace->PdeBase + (index * pFmt->entrySize);

    CHECK(vmemWritePhysical(pVMemSpace, aperture, entryAddr, pPde->v8, pFmt->entrySize));

    return pVmem[indexGpu].vmemIlwalidatePDB(pVMemSpace);
}

/*!
 *  Read @a length of data from a virtual address into a buffer.
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  va          Virtual address.
 *  @param[in]  length      Number of bytes to read.
 *  @param[out] pData       Buffer to store the bytes.
 */
LW_STATUS
vmemRead
(
    VMemSpace *pVMemSpace,
    LwU64      va,
    LwU32      length,
    void      *pData
)
{
    LW_STATUS          status;
    LwU64              pa       = 0;
    LwU64              lwrVA    = 0;
    LwU64              prevVA   = 0;
    GMMU_APERTURE      aperture = {0};
    LwU32              i;

    /* CheetAh specific implementation */
    if (IsTegra() && IsAndroid())
    {
        VToPArg           arg  = {0};
        VMemTableWalkInfo info = {0};

        info.pteFunc = _VToPPteFunc;
        info.pdeFunc = NULL;
        info.pArg    = &arg;

        status = vmemTableWalk(pVMemSpace, va, &info, LW_TRUE);
        if (status != LW_OK)
        {
            return status;
        }

        if (arg.aperture == GMMU_APERTURE_SYS_NONCOH ||
            arg.aperture == GMMU_APERTURE_SYS_COH)
        {
            return readSystem(arg.pa, pData, length);
        }

        return fbRead_GK104(arg.pa, pData, length);
    }

    if (pVMemSpace->class == TEGRA_VASPACE_A)
    {
        return pVmem[indexGpu].vmemRead(pVMemSpace, va, length, pData);
    }

    for (i = 0; i < length; i += 4)
    {
        lwrVA = va + i;

        // since 4K is the minimum page size avoid doing VToP unless in the first iteration of the loop or if crossing to next page.
        if ((prevVA >> 12 != lwrVA >> 12) || i == 0)
        {
            if ((status = vmemVToP(pVMemSpace, lwrVA, &pa, &aperture, LW_FALSE)) != LW_OK)
            {
                dprintf("lw: %s - vmemVToP failed at va=0x%llx\n", __FUNCTION__, lwrVA);
                return LW_ERR_GENERIC;
            }
        }
        else
        {
            pa += 4;
        }

        if ((status = vmemReadPhysical(pVMemSpace, aperture, pa, (void*)((char*)pData + i) , 4)) != LW_OK)
        {
            dprintf("lw: %s - vmemReadPhysical failed at pa=0x%llx va=0x%llx\n", __FUNCTION__, pa, lwrVA);
            return status;
        }

        prevVA = lwrVA;
    }

    return LW_OK;
}

/*!
 *  Write @a length of data to a virtual address from a buffer.
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  va          Virtual address.
 *  @param[in]  length      Number of bytes to write.
 *  @param[out] pData       Buffer of data to write.
 */
LW_STATUS
vmemWrite
(
    VMemSpace *pVMemSpace,
    LwU64      va,
    LwU32      length,
    void      *pData
)
{
    LW_STATUS          status;
    LwU64              pa       = 0;
    LwU64              lwrVA    = 0;
    LwU64              prevVA   = 0;
    GMMU_APERTURE      aperture = {0};
    LwU32              i;

    if (pVMemSpace->class == TEGRA_VASPACE_A)
    {
        return pVmem[indexGpu].vmemWrite(pVMemSpace, va, length, pData);
    }

    for (i = 0; i < length; i += 4)
    {
        lwrVA = va + i;

        // since 4K is the minimum page size avoid doing VToP unless in the first iteration of the loop or if crossing to next page.
        if ((prevVA >> 12 != lwrVA >> 12) || i == 0)
        {
            if ((status = vmemVToP(pVMemSpace, lwrVA, &pa, &aperture, LW_FALSE)) != LW_OK)
            {
                dprintf("lw: %s - vmemVToP failed. Last write at pa=0x%llx va=0x%llx\n", __FUNCTION__, pa, lwrVA);
                return LW_ERR_GENERIC;
            }
        }
        else
        {
            pa += 4;
        }

        if ((status = vmemWritePhysical(pVMemSpace, aperture, pa, ((char*)pData) + i, 4)) != LW_OK)
        {
            dprintf("lw: %s - vmemWritePhysical failed while attempting to write pa=0x%llx va=0x%llx\n", __FUNCTION__, pa, lwrVA);
            return status;
        }

        prevVA = lwrVA;
    }

    return LW_OK;
}

/*!
 *  Fill @a data to a virtual address.
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  va          Virtual address.
 *  @param[in]  length      Length to fill.
 *  @param[in]  data        Data to write.
 *
 *  @return LW_ERR_GENERIC on error, LW_OK on success.
 */
LW_STATUS
vmemFill
(
    VMemSpace *pVMemSpace,
    LwU64      va,
    LwU32      length,
    LwU32      data
)
{
    LW_STATUS          status;
    LwU64              pa       = 0;
    LwU64              lwrVA    = 0;
    LwU64              prevVA   = 0;
    GMMU_APERTURE      aperture = {0};
    LwU32              i;

    if (pVMemSpace->class == TEGRA_VASPACE_A)
    {
        return pVmem[indexGpu].vmemFill(pVMemSpace, va, length, data);
    }

    for (i = 0; i < length; i += 4)
    {
        lwrVA = va + i;

        // since 4K is the minimum page size avoid doing VToP unless in the first iteration of the loop or if crossing to next page.
        if ((prevVA >> 12 != lwrVA >> 12) || i == 0)
        {
            if ((status = vmemVToP(pVMemSpace, lwrVA, &pa, &aperture, LW_FALSE)) != LW_OK)
            {
                dprintf("lw: %s - vmemVToP failed. Last write at pa=0x%llx va=0x%llx\n", __FUNCTION__, pa, lwrVA);
                return LW_ERR_GENERIC;
            }
        }
        else
        {
            pa += 4;
        }

        if ((status = vmemWritePhysical(pVMemSpace, aperture, pa, &data, 4)) != LW_OK)
        {
            dprintf("lw: %s - vmemFill failed while attempting to write pa=0x%llx va=0x%llx\n", __FUNCTION__, pa, lwrVA);
            return status;
        }

        prevVA = lwrVA;
    }

    return LW_OK;
}

LwBool
vmemIsGvpteDeprecated()
{
    return pVmem[indexGpu].vmemIsGvpteDeprecated();
}

typedef struct
{
    VMemTableWalkEntries *pTableWalkEntries;
} GetMappingArg;

static LW_STATUS
_getMappingPteFunc
(
    VMemSpace              *pVMemSpace,
    GMMU_APERTURE           aperture,
    LwU64                   va,
    LwU64                   entryAddr,
    LwU32                   level,
    LwU32                   sublevel,
    LwU32                   index,
    const MMU_FMT_LEVEL    *pFmtLevel,
    const MMU_FMT_PTE      *pFmtPte,
    GMMU_ENTRY_VALUE       *pPte,
    LwBool                  valid,
    LwBool                 *pDone,
    void                   *pArg
)
{
    // Setup argument pointers
    GetMappingArg        *pGetMappingArg    = (GetMappingArg*)pArg;
    VMemTableWalkEntries *pTableWalkEntries = pGetMappingArg->pTableWalkEntries;
    LwBool               *pArgValid         = &pTableWalkEntries->pteValid[sublevel];
    LwU64                *pArgAddress       = &pTableWalkEntries->pteAddr[sublevel];
    GMMU_ENTRY_VALUE     *pArgEntry         = &pTableWalkEntries->pteEntry[sublevel];
    GMMU_APERTURE        *pArgAperture      = &pTableWalkEntries->pteAperture[sublevel];
    MMU_FMT_LEVEL        *pArgFmtLevel      = &pTableWalkEntries->fmtLevel[level];
    MMU_FMT_PTE          *pArgFmtPte        = &pTableWalkEntries->fmtPte[sublevel];

    // Update walk sublevel value
    pTableWalkEntries->sublevels = sublevel + 1;

    // Save PTE information
    *pArgValid    = valid;
    *pArgAddress  = entryAddr;
    *pArgAperture = aperture;

    memcpy(pArgEntry,    pPte,      pFmtLevel->entrySize);
    memcpy(pArgFmtLevel, pFmtLevel, sizeof(MMU_FMT_LEVEL));
    memcpy(pArgFmtPte,   pFmtPte,   sizeof(GMMU_FMT_PTE));

    if (valid)
        return LW_OK;
    else
        return LW_ERR_GENERIC;
}

static LW_STATUS
_getMappingPdeFunc
(
    VMemSpace              *pVMemSpace,
    GMMU_APERTURE           aperture,
    LwU64                   va,
    LwU64                   entryAddr,
    LwU32                   level,
    LwU32                   sublevel,
    LwU32                   index,
    const MMU_FMT_LEVEL    *pFmtLevel,
    const MMU_FMT_PDE      *pFmtPde,
    GMMU_ENTRY_VALUE       *pPde,
    LwBool                  valid,
    LwBool                 *pDone,
    void                   *pArg
)
{
    // Setup argument pointers
    GetMappingArg        *pGetMappingArg    = (GetMappingArg*)pArg;
    VMemTableWalkEntries *pTableWalkEntries = pGetMappingArg->pTableWalkEntries;
    LwBool               *pArgValid         = &pTableWalkEntries->pdeValid[level];
    LwU64                *pArgAddress       = &pTableWalkEntries->pdeAddr[level];
    GMMU_ENTRY_VALUE     *pArgEntry         = &pTableWalkEntries->pdeEntry[level];
    GMMU_APERTURE        *pArgAperture      = &pTableWalkEntries->pdeAperture[level];
    MMU_FMT_LEVEL        *pArgFmtLevel      = &pTableWalkEntries->fmtLevel[level];
    MMU_FMT_PDE          *pArgFmtPde        = &pTableWalkEntries->fmtPde[level];

    // Update walk level value
    pTableWalkEntries->levels = level + 1;

    // Save PDE information
    *pArgValid    = valid;
    *pArgAddress  = entryAddr;
    *pArgAperture = aperture;

    memcpy(pArgEntry,    pPde,      pFmtLevel->entrySize);
    memcpy(pArgFmtLevel, pFmtLevel, sizeof(MMU_FMT_LEVEL));
    memcpy(pArgFmtPde,   pFmtPde,   sizeof(GMMU_FMT_PDE));

    if (valid)
        return LW_OK;
    else
        return LW_ERR_GENERIC;
}

static LW_STATUS
_vmemGetMapping
(
    VMemSpace              *pVMemSpace,
    LwU64                   va,
    VMemTableWalkEntries   *pTableWalkEntries
)
{
    LwU32             sublevel;
    GetMappingArg     arg  = {0};
    VMemTableWalkInfo info = {0};

    if (!pVMemSpace || !pTableWalkEntries)
    {
        return LW_ERR_GENERIC;
    }
    memset(pTableWalkEntries, 0, sizeof(VMemTableWalkEntries));
    arg.pTableWalkEntries = pTableWalkEntries;

    info.pteFunc = _getMappingPteFunc;
    info.pdeFunc = _getMappingPdeFunc;
    info.pArg    = &arg;

    vmemTableWalk(pVMemSpace, va, &info, LW_FALSE);

    // Walk all sublevels until a valid sublevel is found (or not and return an error)
    for (sublevel = 0; sublevel < pTableWalkEntries->sublevels; sublevel++)
    {
        if (pTableWalkEntries->pteValid[sublevel])
            return LW_OK;
    }
    return LW_ERR_GENERIC;
}

static LW_STATUS
_vmemDupMapping
(
    VMemSpace              *pVMemTargetSpace,
    VMemTableWalkEntries   *pTargetWalkEntries,
    VMemSpace              *pVMemSourceSpace,
    VMemTableWalkEntries   *pSourceWalkEntries
)
{
    LwU32           sublevel;
    LwU32           pteSize;

    // Use target PTE size in case PDE/PTE level mismatch, in which case source size is PDE entry size
    pteSize = pTargetWalkEntries->fmtLevel[pTargetWalkEntries->levels].entrySize;

    // Check for incompatible address mappings (LwWatch can't create PDE/PTE's)
    if (pSourceWalkEntries->levels    > pTargetWalkEntries->levels)
    {
        return LW_ERR_GENERIC;
    }
    // Check to see if there is a PDE/PTE level mismatch (Changing a target PDE entry to a PTE entry)
    if (pSourceWalkEntries->levels != pTargetWalkEntries->levels)
    {
        // Reprogram last target PDE entry to the source PTE entry
        vmemWritePhysical(pVMemTargetSpace,
                          pTargetWalkEntries->pdeAperture[pSourceWalkEntries->levels],
                          pTargetWalkEntries->pdeAddr[pSourceWalkEntries->levels],
                          pSourceWalkEntries->pteEntry[0].v8,
                          pteSize);
    }
    else    // PDE/PTE's at the same level, just reprogram PTE entries
    {
        // Reprogram each target PTE entry to the source PTE entry
        for (sublevel = 0; sublevel < pSourceWalkEntries->sublevels; sublevel++)
        {
            // Reprogram the next PTE sublevel entry
            vmemWritePhysical(pVMemTargetSpace,
                              pTargetWalkEntries->pteAperture[sublevel],
                              pTargetWalkEntries->pteAddr[sublevel],
                              pSourceWalkEntries->pteEntry[sublevel].v8,
                              pteSize);
        }
    }
    return LW_OK;
}

static LW_STATUS
_vmemSetMapping
(
    VMemSpace              *pVMemSpace,
    VMemTableWalkEntries   *pTableWalkEntries
)
{
    LwU32           sublevel;
    LwU32           pteSize;

    // Use target PTE size as that's the size overwritten during mapping duplication (Even for mismatch)
    pteSize = pTableWalkEntries->fmtLevel[pTableWalkEntries->levels].entrySize;

    // Reprogram the last target PDE level in case there was a PDE/PTE level mismatch (Only need pteSize bytes restored)
    vmemWritePhysical(pVMemSpace,
                      pTableWalkEntries->pdeAperture[pTableWalkEntries->levels - 1],
                      pTableWalkEntries->pdeAddr[pTableWalkEntries->levels - 1],
                      pTableWalkEntries->pdeEntry[pTableWalkEntries->levels - 1].v8,
                      pteSize);

    // Reprogram each target PTE entry back to the original value
    for (sublevel = 0; sublevel < pTableWalkEntries->sublevels; sublevel++)
    {
        // Reprogram the next PTE sublevel entry
        vmemWritePhysical(pVMemSpace,
                          pTableWalkEntries->pteAperture[sublevel],
                          pTableWalkEntries->pteAddr[sublevel],
                          pTableWalkEntries->pteEntry[sublevel].v8,
                          pteSize);
    }
    return LW_OK;
}

static LwU32
_vmemPageSize
(
    VMemTableWalkEntries   *pTableWalkEntries
)
{
    LwU32           sublevel;
    LwU32           pageSize = 0;

    // Check all sublevels for a valid level
    for (sublevel = 0; sublevel < pTableWalkEntries->sublevels; sublevel++)
    {
        // Check to see if this sublevel PTE is valid
        if (pTableWalkEntries->pteValid[sublevel])
        {
            // Get and return the page size for this PTE sublevel
            pageSize = (LwU32)mmuFmtLevelPageSize(&pTableWalkEntries->fmtLevel[pTableWalkEntries->levels]);

            break;
        }
    }
    return pageSize;
}

/*!
 *  Sets up bar1 so that va's can be read from/written to. To handle the
 *  size restriction in bar1, the VA needs page directories and tables
 *  need to be modified such that when placed into bar1, the page boundary
 *  maps to virtual address 0. In the source virtual memory space, each
 *  level's entry is swapped with the 0th entry of that level. This gets
 *  restored in vmemEndBar1Mapping.
 *
 *  @param[in] pVMemSpace   Virtual memory space.
 *  @param[in] va           Virtual address.
 *  @param[in] pOrigBar1Pde Original pde to restore.
 *  @param[in] param        Staging remapping parameter.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */

#define STAGING_BAR1PDE_INDEX 0
#define STAGING_BAR1VA        0

typedef struct
{
    LwU64               targetVa;
    GMMU_ENTRY_VALUE    pde;
    LwU64               pdeAddr;
    LwU32               pdeIndex;
    LwU32               pdeSize;
    GMMU_APERTURE       pdeAperture;
    GMMU_ENTRY_VALUE    pte;
    LwU64               pteAddr;
    LwU32               pteIndex;
    LwU32               pteSize;
    GMMU_APERTURE       pteAperture;
    MMU_FMT_LEVEL       pdeLevel;
    MMU_FMT_LEVEL       pteLevel;
    MMU_FMT_PDE         fmtPde;
    MMU_FMT_PTE         fmtPte;
} Bar1MappingArg;

static VMemTableWalkEntries vmemTableWalkEntries;
static VMemTableWalkEntries sourceTableWalkEntries;
static VMemTableWalkEntries targetTableWalkEntries;
static VMemSpace vmemBar1;
static Bar1MappingArg arg;
static LwU32 bigPageSize;
static LwU64 stagingParam;
static LwU64 offset;

LW_STATUS
vmemBeginBar1Mapping
(
    VMemSpace        *pVMemSpace,
    LwU64             va
)
{
    LW_STATUS         status;
    VMemTableWalkInfo info = {0};

    if (!pVMemSpace)
    {
        return LW_ERR_GENERIC;
    }
    // Setup the BAR1 mapping arguments
    memset(&arg, 0, sizeof(arg));
    arg.targetVa = STAGING_BAR1VA;

    info.pteFunc = _bar1MappingPteFunc;
    info.pdeFunc = _bar1MappingPdeFunc;
    info.pArg    = &arg;

    CHECK(vmemGet(&vmemBar1, VMEM_TYPE_BAR1, NULL));

    // Save the original BAR1 big page size
    bigPageSize = vmemBar1.bigPageSize;

    // Check for BAR1 big page size update needed
    if (pVMemSpace->bigPageSize != bigPageSize)
    {
        // Temporarily set BAR1 big page size to match memory space
        CHECK(pVmem[indexGpu].vmemSetBigPageSize(&vmemBar1, pVMemSpace->bigPageSize));

        // Update the BAR1 VMem space
        CHECK(vmemGet(&vmemBar1, VMEM_TYPE_BAR1, NULL));
    }
    // Get the source and target mappings
    CHECK(_vmemGetMapping(pVMemSpace, va, &sourceTableWalkEntries));
    _vmemGetMapping(&vmemBar1, STAGING_BAR1VA, &targetTableWalkEntries);

    // Compute staging area offset based on PTE page size
    offset = va % _vmemPageSize(&sourceTableWalkEntries);

    // Duplicate the source mapping to the target address
    CHECK(_vmemDupMapping(&vmemBar1, &targetTableWalkEntries, pVMemSpace, &sourceTableWalkEntries));

    // Disable the BAR1 remappers (to prevent remapping during access)
    pMmu[indexGpu].mmuDisableBar1ActiveRemappers(&stagingParam);

    // Ilwalidate PDB (and all VA's) now that the PDE/PTE's have been updated
    pVmem[indexGpu].vmemIlwalidatePDB(&vmemBar1);

    return LW_OK;
}

/*!
 *  Restore any state modified at BeginBar1Mapping time
 *
 *  @param[in] pVMemSpace   Virtual memory space.
 *  @param[in] va           Virtual address.
 *  @param[in] pOrigBar1Pde Original pde to restore.
 *  @param[in] param        Staging remapping parameter.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemEndBar1Mapping
(
    VMemSpace        *pVMemSpace,
    LwU64             va
)
{
    LW_STATUS         status;
    VMemTableWalkInfo info = {0};

    // Check for BAR1 big page size update needed
    if (pVMemSpace->bigPageSize != bigPageSize)
    {
        // Restore original BAR1 big page size
        CHECK(pVmem[indexGpu].vmemSetBigPageSize(&vmemBar1, bigPageSize));

        // Update the BAR1 VMem space
        CHECK(vmemGet(&vmemBar1, VMEM_TYPE_BAR1, NULL));
    }
    // Restore the original BAR1 PDE/PTE entries
    _vmemSetMapping(&vmemBar1, &targetTableWalkEntries);

    // Re-enable the BAR1 remappers
    pMmu[indexGpu].mmuReenableBar1ActiveRemappers(stagingParam);

    // Ilwalidate PDB (and all VA's) now that the PDE/PTE's have been restored
    pVmem[indexGpu].vmemIlwalidatePDB(&vmemBar1);

    return LW_OK;
}

LW_STATUS
vmemReadPhysical
(
    VMemSpace    *pVMemSpace,
    GMMU_APERTURE aperture,
    LwU64         pa,
    void         *buffer,
    LwU32         length
)
{
    if (!pVMemSpace)
    {
        return LW_ERR_GENERIC;
    }
    switch (aperture)
    {
    case GMMU_APERTURE_PEER:
    case GMMU_APERTURE_VIDEO:
        return pFb[indexGpu].fbRead(pa, buffer, length);
    case GMMU_APERTURE_SYS_NONCOH:
    case GMMU_APERTURE_SYS_COH:
        return readSystem(pa, buffer, length);
    default:
        return LW_ERR_GENERIC;
    }
}

LW_STATUS
vmemWritePhysical
(
    VMemSpace    *pVMemSpace,
    GMMU_APERTURE aperture,
    LwU64         pa,
    void         *buffer,
    LwU32         length
)
{
    if (!pVMemSpace)
    {
        return LW_ERR_GENERIC;
    }
    switch (aperture)
    {
    case GMMU_APERTURE_PEER:
    case GMMU_APERTURE_VIDEO:
        return pFb[indexGpu].fbWrite(pa, buffer, length);
    case GMMU_APERTURE_SYS_NONCOH:
    case GMMU_APERTURE_SYS_COH:
        return writeSystem(pa, buffer, length);
    default:
        return LW_ERR_GENERIC;
    }
}

LwU64
vmemGetPhysAddrFromPDE
(
    VMemSpace              *pVMemSpace,
    const MMU_FMT_PDE      *pFmtPde,
    const GMMU_ENTRY_VALUE *pPde
)
{
    GMMU_APERTURE             aperture;
    const GMMU_FIELD_ADDRESS *pAddrFld;

    if (!pVMemSpace)
    {
        return ILWALID_ADDRESS;
    }

    aperture = gmmuFieldGetAperture(&pFmtPde->gmmu.fldAperture, pPde->v8);

    if (aperture == GMMU_APERTURE_ILWALID)
    {
        return ILWALID_ADDRESS;
    }
    pAddrFld = gmmuFmtPdePhysAddrFld(&pFmtPde->gmmu, aperture);
    return pAddrFld ? gmmuFieldGetAddress(pAddrFld, pPde->v8) : ILWALID_ADDRESS;
}

LwU64
vmemGetPhysAddrFromPTE
(
    VMemSpace              *pVMemSpace,
    const GMMU_ENTRY_VALUE *pPte
)
{
    const MMU_FMT_PTE        *pFmtPte;
    const GMMU_FMT_PTE       *pFmt;
    GMMU_APERTURE             aperture;
    const GMMU_FIELD_ADDRESS *pAddrFld;

    if (!pVMemSpace || !pPte)
    {
        return ILWALID_ADDRESS;
    }
    pFmtPte = pVmem[indexGpu].vmemGetPTEFmt(pVMemSpace);
    if (!pFmtPte)
    {
        return ILWALID_ADDRESS;
    }

    pFmt     = &pFmtPte->gmmu;
    aperture = gmmuFieldGetAperture(&pFmt->fldAperture, pPte->v8);

    if (aperture == GMMU_APERTURE_ILWALID)
    {
        return ILWALID_ADDRESS;
    }
    pAddrFld = gmmuFmtPtePhysAddrFld(pFmt, aperture);
    return pAddrFld ? gmmuFieldGetAddress(pAddrFld, pPte->v8) : ILWALID_ADDRESS;
}

LW_STATUS
vmemGetPdeAperture
(
    VMemSpace              *pVMemSpace,
    LwU32                   level,
    const GMMU_ENTRY_VALUE *pPde,
    GMMU_APERTURE          *pAperture,
    ApertureSize           *pSize
)
{
    GMMU_APERTURE aperture    = GMMU_APERTURE_ILWALID;
    ApertureSize  size        = APERTURE_SIZE_NONE;
    VMemFmtPde    fmtPdeMulti;
    GMMU_FMT_PDE *pFmtPde;
    LW_STATUS     status;

    CHECK(pVmem[indexGpu].vmemGetPDEFmt(pVMemSpace, &fmtPdeMulti, level));

    if (!fmtPdeMulti.bMulti)
    {
        pFmtPde  = &fmtPdeMulti.fmts.single.gmmu;
        aperture = gmmuFieldGetAperture(&pFmtPde->fldAperture, pPde->v8);
    }
    else
    {
        pFmtPde  = &fmtPdeMulti.fmts.multi.gmmu.subLevels[PDE_MULTI_BIG_INDEX];
        aperture = gmmuFieldGetAperture(&pFmtPde->fldAperture, pPde->v8);
        if (aperture == GMMU_APERTURE_ILWALID)
        {
            pFmtPde  = &fmtPdeMulti.fmts.multi.gmmu.subLevels[PDE_MULTI_SMALL_INDEX];
            aperture = gmmuFieldGetAperture(&pFmtPde->fldAperture, pPde->v8);
            if (aperture != GMMU_APERTURE_ILWALID)
            {
                size = APERTURE_SIZE_SMALL;
            }
        }
        else
        {
            size = APERTURE_SIZE_BIG;
        }
    }
    // Return aperture if requested
    if (pAperture)
    {
        *pAperture = aperture;
    }
    // Return size if requested
    if (pSize)
    {
        *pSize = size;
    }
    return LW_OK;
}

/**********************************************************************************
 *
 *
 * Internal helper functions
 *
 *
 *********************************************************************************/

static LwBool _mmuFmtEntryIsPte
(
    VMemLayout          *pLayout,
    const MMU_FMT_LEVEL *pFmtLevel,
    GMMU_ENTRY_VALUE    *pEntry
)
{
    return gmmuFmtEntryIsPte(&pLayout->fmt.gmmu, pFmtLevel, pEntry->v8);
}

static LwBool _mmuPteIsValid
(
    VMemLayout       *pLayout,
    GMMU_ENTRY_VALUE *pEntry
)
{
    const GMMU_FMT_PTE *pFmtPte = pLayout->fmt.gmmu.pPte;

    return lwFieldGetBool(&pFmtPte->fldValid, pEntry->v8);
}

static const MMU_FMT_PDE *_mmuFmtGetPde
(
    VMemLayout          *pLayout,
    const MMU_FMT_LEVEL *pLevel,
    LwU32                sublevel
)
{
    return (const MMU_FMT_PDE*)gmmuFmtGetPde(&pLayout->fmt.gmmu, pLevel, sublevel);
}

static void _dumpPteInfo
(
    VMemSpace           *pVMemSpace,
    LwU64                va,
    LwU64                entryAddr,
    LwU32                level,
    LwU32                index,
    const MMU_FMT_LEVEL *pFmtLevel,
    const MMU_FMT_PTE   *pFmtPte,
    GMMU_ENTRY_VALUE    *pPte,
    LwBool               valid
)
{
    LwU64         baseAddr = entryAddr - (index * pFmtLevel->entrySize);
    LwU64         pageSize = mmuFmtLevelPageSize(pFmtLevel);
    LwU64         pa;
    LwU64         offset;
    LwU32         peerIndex;
    GMMU_APERTURE aperture;

    // Non-huge pages had their page tables printed for an extra level
    if (pageSize != VMEM_HUGE_PAGESIZE_2M && pageSize != VMEM_HUGE_PAGESIZE_512M)
    {
        level++;
    }
    INDENT(level);

    switch (pageSize)
    {
    case VMEM_HUGE_PAGESIZE_512M:
        if (!valid)
        {
            dprintf("PTE_512M[0x%x]: Invalid\n", index);
            return;
        }
        dprintf("PTE_512M[0x%03x]: ", index);
        break;
    case VMEM_HUGE_PAGESIZE_2M:
        if (!valid)
        {
            dprintf("PTE_2M[0x%x]: Invalid\n", index);
            return;
        }
        dprintf("PTE_2M[0x%03x]: ", index);
        break;
    case VMEM_BIG_PAGESIZE_128K:
        if (!valid)
        {
            dprintf("PTE_128K[0x%x]: Invalid\n", index);
            return;
        }
        dprintf("PTE_128K[0x%03x]: ", index);
        break;
    case VMEM_BIG_PAGESIZE_64K:
        if (!valid)
        {
            dprintf("PTE_64K[0x%x]: Invalid\n", index);
            return;
        }
        dprintf("PTE_64K[0x%03x]: ", index);
        break;
    case VMEM_SMALL_PAGESIZE:
        if (!valid)
        {
            dprintf("PTE_4K[0x%x]: Invalid\n", index);
            return;
        }
        dprintf("PTE_4K[0x%03x]: ", index);
        break;
    default:
        dprintf("PTE[0x%03x]: Invalid pagesize: 0x%llx\n", index, pageSize);
        return;
    }
    pa     = vmemGetPhysAddrFromPTE(pVMemSpace, pPte);
    offset = mmuFmtVirtAddrPageOffset(pFmtLevel, va);

    peerIndex = lwFieldGet32(&pFmtPte->gmmu.fldPeerIndex, pPte->v8);
    aperture  = gmmuFieldGetAperture(&pFmtPte->gmmu.fldAperture, pPte->v8);

    vmemDumpPa(pa, aperture, peerIndex);
    dprintf(", Offset: 0x%llx ", offset);

    vmemDumpPte(pVMemSpace, pPte);
}

static void _dumpPtInfo
(
    VMemSpace              *pVMemSpace,
    const MMU_FMT_LEVEL    *pFmtLevel,
    LwU32                   level,
    LwU32                   subLevel,
    const GMMU_ENTRY_VALUE *pPde
)
{
    const MMU_FMT_LEVEL *pFmtSub  = &pFmtLevel->subLevels[subLevel];
    LwU64                pageSize = mmuFmtLevelPageSize(pFmtSub);
    const MMU_FMT_PDE   *pFmtPde  = _mmuFmtGetPde(&pVMemSpace->layout, pFmtLevel, subLevel);
    GMMU_APERTURE        aperture = gmmuFieldGetAperture(&pFmtPde->gmmu.fldAperture, pPde->v8);
    LwU64                pt       = vmemGetPhysAddrFromPDE(pVMemSpace, pFmtPde, pPde);

    if (pt == ILWALID_ADDRESS)
    {
        pt = 0;
    }
    switch (pageSize)
    {
    case VMEM_BIG_PAGESIZE_128K:
    case VMEM_BIG_PAGESIZE_64K:
    case VMEM_SMALL_PAGESIZE:
        INDENT(level+1);
        dprintf("PT_%luK:\t" LwU64_FMT "(%s)\n",
                (unsigned long) (pageSize / KB), pt, decodeAperture(aperture));
        break;
    }
}

static void _dumpPdeInfo
(
    VMemSpace           *pVMemSpace,
    LwU32                level,
    LwU32                subLevel,
    LwU32                index,
    const MMU_FMT_LEVEL *pFmtLevel,
    GMMU_ENTRY_VALUE    *pPde,
    LwBool               valid
)
{
    LwU32 hwLevel = pVmem[indexGpu].vmemSWToHWLevel(pVMemSpace, level);

    INDENT(level);
    dprintf("PDE%u[0x%03x]: ", hwLevel, index);
    vmemDumpPde(pVMemSpace, pPde, level);

    if (valid && pFmtLevel->numSubLevels > 1)
    {
        LwU32 i;

        //
        // Print the other tables first
        //
        for (i = 0; i < pFmtLevel->numSubLevels; i++)
        {
            _dumpPtInfo(pVMemSpace, pFmtLevel, level, i, pPde);
        }
    }
}

static LW_STATUS _vmemTableWalk
(
    VMemSpace           *pVMemSpace,
    GMMU_APERTURE        aperture,
    LwU32                level,
    LwU32                sublevel,
    const MMU_FMT_LEVEL *pFmtLevel,
    LwU64                base,
    LwU64                va,
    LwBool              *pDone,
    VMemTableWalkInfo   *pInfo,
    LwBool               verbose
)
{
    LwU32            index     = mmuFmtVirtAddrToEntryIndex(pFmtLevel, va);
    LwU64            entryAddr = base + (index * pFmtLevel->entrySize);
    LW_STATUS        status    = LW_ERR_GENERIC;
    GMMU_ENTRY_VALUE entry;

    // Return error if invalid base address or index
    if (base == ILWALID_ADDRESS || index >= mmuFmtLevelEntryCount(pFmtLevel))
    {
        return LW_ERR_GENERIC;
    }
    CHECK(vmemReadPhysical(pVMemSpace, aperture, entryAddr, &entry.v8, pFmtLevel->entrySize));

    // PTE is the base case
    if (_mmuFmtEntryIsPte(&pVMemSpace->layout, pFmtLevel, &entry))
    {
        MMU_FMT_PTE *pFmtPte = &pVMemSpace->layout.pte;
        LwBool       valid   = _mmuPteIsValid(&pVMemSpace->layout, &entry);

        // Dump PTE information if requested
        if (verbose)
        {
            _dumpPteInfo(pVMemSpace, va, entryAddr, level, index, pFmtLevel, pFmtPte, &entry, valid);
        }
        // Check for PTE function return function result if present
        if (pInfo->pteFunc)
        {
            return pInfo->pteFunc(pVMemSpace, aperture, va, entryAddr, level, sublevel, index, pFmtLevel, pFmtPte,
                                  &entry, valid, pDone, pInfo->pArg);
        }
        // Return based on valid PTE translation
        return (valid ? LW_OK : LW_ERR_GENERIC);
    }
    // This must be a PDE, attempt translation of each sub-level
    for (sublevel = 0; sublevel < pFmtLevel->numSubLevels; sublevel++)
    {
        const MMU_FMT_PDE *pFmtPde      = _mmuFmtGetPde(&pVMemSpace->layout, pFmtLevel, sublevel);
        LwU64              nextBase     = vmemGetPhysAddrFromPDE(pVMemSpace, pFmtPde, &entry);
        GMMU_APERTURE      nextAperture = gmmuFieldGetAperture(&pFmtPde->gmmu.fldAperture, entry.v8);
        LwBool             valid        = (nextBase != ILWALID_ADDRESS);

        // Check for early exit
        if (!valid)
        {
            GMMU_FMT *pFmt = &pVMemSpace->layout.fmt.gmmu;

            if (pFmt->bSparseHwSupport && (sublevel == 0) &&
                pVmem[indexGpu].vmemIsPdeVolatile(&pFmtPde->gmmu, &entry))
            {
                return LW_ERR_GENERIC;
            }
        }
        // Dump PDE information if requested (and first sublevel)
        if (verbose && (sublevel == 0))
        {
            _dumpPdeInfo(pVMemSpace, level, sublevel, index, pFmtLevel, &entry, LW_TRUE);
        }
        // Check for PDE function
        if (pInfo->pdeFunc)
        {
            status = pInfo->pdeFunc(pVMemSpace, aperture, va, entryAddr, level, sublevel, index, pFmtLevel,
                                    pFmtPde, &entry, valid, pDone, pInfo->pArg);

            // Check for early exit requested
            if (*pDone)
            {
                return status;
            }
        }
        // Relwrse into sub-level translation if valid
        if (valid)
        {
            status = _vmemTableWalk(pVMemSpace, nextAperture, level + 1, sublevel, &pFmtLevel->subLevels[sublevel],
                                    nextBase, va, pDone, pInfo, verbose);
        }
        // Check for early exit requested (from sub-level)
        if (*pDone)
        {
            return status;
        }
    }
    return status;
}

static LW_STATUS _pteGetByVaPteFunc
(
    VMemSpace           *pVMemSpace,
    GMMU_APERTURE        aperture,
    LwU64                va,
    LwU64                entryAddr,
    LwU32                level,
    LwU32                sublevel,
    LwU32                index,
    const MMU_FMT_LEVEL *pFmtLevel,
    const MMU_FMT_PTE   *pFmtPte,
    GMMU_ENTRY_VALUE    *pPte,
    LwBool               valid,
    LwBool              *pDone,
    void                *pArg
)
{
    // Get argument pointers
    PteGetByVaArg        *pPteGetByVaArg    = (PteGetByVaArg*)pArg;
    VMemTableWalkEntries *pTableWalkEntries = pPteGetByVaArg->pTableWalkEntries;
    GMMU_ENTRY_VALUE     *pArgEntry         = &pTableWalkEntries->pteEntry[sublevel];
    MMU_FMT_LEVEL        *pArgFmtLevel      = &pTableWalkEntries->fmtLevel[level];
    MMU_FMT_PTE          *pArgFmtPte        = &pTableWalkEntries->fmtPte[sublevel];

    // Save PTE information
    pTableWalkEntries->pteValid[sublevel]   = valid;
    pTableWalkEntries->sublevels            = sublevel + 1;

    // Only copy PTE format information if valid
    if (valid)
    {
        memcpy(pArgEntry,    pPte,      pFmtLevel->entrySize);
        memcpy(pArgFmtLevel, pFmtLevel, sizeof(MMU_FMT_LEVEL));
        memcpy(pArgFmtPte,   pFmtPte,   sizeof(GMMU_FMT_PTE));

        *pDone = TRUE;
    }
    else
    {
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

static LW_STATUS _pteGetByVaPdeFunc
(
    VMemSpace           *pVMemSpace,
    GMMU_APERTURE        aperture,
    LwU64                va,
    LwU64                entryAddr,
    LwU32                level,
    LwU32                sublevel,
    LwU32                index,
    const MMU_FMT_LEVEL *pFmtLevel,
    const MMU_FMT_PDE   *pFmtPde,
    GMMU_ENTRY_VALUE    *pPde,
    LwBool               valid,
    LwBool              *pDone,
    void                *pArg
)
{
    // Setup argument pointers
    PteGetByVaArg        *pPteGetByVaArg    = (PteGetByVaArg*)pArg;
    VMemTableWalkEntries *pTableWalkEntries = pPteGetByVaArg->pTableWalkEntries;
    GMMU_ENTRY_VALUE     *pArgEntry         = &pTableWalkEntries->pdeEntry[level];
    MMU_FMT_LEVEL        *pArgFmtLevel      = &pTableWalkEntries->fmtLevel[level];
    MMU_FMT_PDE          *pArgFmtPde        = &pTableWalkEntries->fmtPde[level];

    // Update walk level value
    pTableWalkEntries->levels = level + 1;

    // Only update PDE format information if valid
    if (valid)
    {
        memcpy(pArgEntry,    pPde,      pFmtLevel->entrySize);
        memcpy(pArgFmtLevel, pFmtLevel, sizeof(MMU_FMT_LEVEL));
        memcpy(pArgFmtPde,   pFmtPde,   sizeof(GMMU_FMT_PDE));
    }
    else
    {
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

static LW_STATUS _pdeGetByVaPteFunc
(
    VMemSpace           *pVMemSpace,
    GMMU_APERTURE        aperture,
    LwU64                va,
    LwU64                entryAddr,
    LwU32                level,
    LwU32                sublevel,
    LwU32                index,
    const MMU_FMT_LEVEL *pFmtLevel,
    const MMU_FMT_PTE   *pFmtPte,
    GMMU_ENTRY_VALUE    *pPte,
    LwBool               valid,
    LwBool              *pDone,
    void                *pArg
)
{
    return LW_ERR_GENERIC;
}

static LW_STATUS _pdeGetByVaPdeFunc
(
    VMemSpace           *pVMemSpace,
    GMMU_APERTURE        aperture,
    LwU64                va,
    LwU64                entryAddr,
    LwU32                level,
    LwU32                sublevel,
    LwU32                index,
    const MMU_FMT_LEVEL *pFmtLevel,
    const MMU_FMT_PDE   *pFmtPde,
    GMMU_ENTRY_VALUE    *pPde,
    LwBool               valid,
    LwBool              *pDone,
    void                *pArg
)
{
    PdeGetByVaArg *pPdeGetByVaArg = (PdeGetByVaArg*)pArg;

    // Copy PDE if desired level and valid
    if (valid)
    {
        if (pPdeGetByVaArg->level == level)
        {
            memcpy(pPdeGetByVaArg->pPde, pPde, pFmtLevel->entrySize);
            *pDone = TRUE;
        }
    }
    else
    {
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

static LW_STATUS _VToPPteFunc
(
    VMemSpace           *pVMemSpace,
    GMMU_APERTURE        aperture,
    LwU64                va,
    LwU64                entryAddr,
    LwU32                level,
    LwU32                sublevel,
    LwU32                index,
    const MMU_FMT_LEVEL *pFmtLevel,
    const MMU_FMT_PTE   *pFmtPte,
    GMMU_ENTRY_VALUE    *pPte,
    LwBool               valid,
    LwBool              *pDone,
    void                *pArg
)
{
    VToPArg *pVToPArg = (VToPArg*)pArg;

    // Only update translation if not already valid from previous PTE
    if (!pVToPArg->valid)
    {
        pVToPArg->valid     = valid;
        pVToPArg->pa        = vmemGetPhysAddrFromPTE(pVMemSpace, pPte);
        pVToPArg->pa       += mmuFmtVirtAddrPageOffset(pFmtLevel, va);
        pVToPArg->aperture  = gmmuFieldGetAperture(&pFmtPte->gmmu.fldAperture, pPte->v8);
        pVToPArg->peerIndex = lwFieldGet32(&pFmtPte->gmmu.fldPeerIndex, pPte->v8);
    }
    // Return result based on *any* valid translation
    if (pVToPArg->valid)
    {
        return LW_OK;
    }
    else
    {
        return LW_ERR_GENERIC;
    }
}

/*!
 * For Bar1 mapping, at each level, an entry is swapped with that of the index of another
 * provided va. The target va is allowed to be specified so that when we fix the mapping
 * later, we can use exactly the same code with just the two addresses swapped. This is
 * necessary in order to avoid the bar1 aperture limit background.
 */
static LW_STATUS _bar1MappingPteFunc
(
    VMemSpace           *pVMemSpace,
    GMMU_APERTURE        aperture,
    LwU64                va,
    LwU64                entryAddr,
    LwU32                level,
    LwU32                sublevel,
    LwU32                index,
    const MMU_FMT_LEVEL *pFmtLevel,
    const MMU_FMT_PTE   *pFmtPte,
    GMMU_ENTRY_VALUE    *pPte,
    LwBool               valid,
    LwBool              *pDone,
    void                *pArg
)
{
    Bar1MappingArg  *pBar1MappingArg = (Bar1MappingArg*)pArg;

    // Only update PTE mapping information if valid
    if (valid)
    {
        memcpy(&pBar1MappingArg->pte,      pPte,      pFmtLevel->entrySize);
        memcpy(&pBar1MappingArg->pteLevel, pFmtLevel, sizeof(MMU_FMT_LEVEL));
        memcpy(&pBar1MappingArg->fmtPte,   pFmtPte,   sizeof(GMMU_FMT_PTE));

        pBar1MappingArg->pteAddr     = entryAddr;
        pBar1MappingArg->pteIndex    = index;
        pBar1MappingArg->pteSize     = pFmtLevel->entrySize;
        pBar1MappingArg->pteAperture = aperture;

        *pDone = TRUE;
    }
    else
    {
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

static LW_STATUS _bar1MappingPdeFunc
(
    VMemSpace           *pVMemSpace,
    GMMU_APERTURE        aperture,
    LwU64                va,
    LwU64                entryAddr,
    LwU32                level,
    LwU32                sublevel,
    LwU32                index,
    const MMU_FMT_LEVEL *pFmtLevel,
    const MMU_FMT_PDE   *pFmtPde,
    GMMU_ENTRY_VALUE    *pPde,
    LwBool               valid,
    LwBool              *pDone,
    void                *pArg
)
{
    Bar1MappingArg  *pBar1MappingArg = (Bar1MappingArg*)pArg;

    // Only update PDE mapping information if valid
    if (valid)
    {
        memcpy(&pBar1MappingArg->pde,      pPde,      pFmtLevel->entrySize);
        memcpy(&pBar1MappingArg->pdeLevel, pFmtLevel, sizeof(MMU_FMT_LEVEL));
        memcpy(&pBar1MappingArg->fmtPde,   pFmtPde,   sizeof(GMMU_FMT_PDE));

        pBar1MappingArg->pdeAddr     = entryAddr;
        pBar1MappingArg->pdeIndex    = index;
        pBar1MappingArg->pdeSize     = pFmtLevel->entrySize;
        pBar1MappingArg->pdeAperture = aperture;
    }
    else
    {
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}
