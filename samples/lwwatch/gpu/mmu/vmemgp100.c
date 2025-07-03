/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "pascal/gp100/dev_bus.h"
#include "pascal/gp100/dev_mmu.h"
#include "pascal/gp100/dev_ram.h"
#include "pascal/gp100/dev_fb.h"
#include "turing/tu102/dev_vm.h"
#include "fifo.h"
#include "vgpu.h"

#include "g_vmem_private.h"

#define VALID_SHIFT(s) ((s) == 16 || (s) == 17)

static LwU32 _mmuFmtVersion_GP100(VMemSpace *pVMemSpace);
static void  _gmmuFmtInitPdeMulti_GP10X(GMMU_FMT_PDE_MULTI *pPdeMulti, LW_FIELD_ENUM_ENTRY *pdeApertures);
static void  _gmmuFmtInitPde_GP10X(GMMU_FMT_PDE *pPde, LW_FIELD_ENUM_ENTRY *pdeApertures);
static void  _gmmuInitMmuFmt_GP10X(MMU_FMT_LEVEL *pLevels, LwU32 bigPageShift);


/*!
 *  Returns the largest virtual address for given vmem space.
 *
 *  @param[in]  pVMemSpace  The vmem space for which VAL is to be found out
 *
 *  @return   largest virtual address (VAL) from inst block.
 */
LwU64
vmemGetLargestVirtAddr_GP100(VMemSpace *pVMemSpace)
{
    LwU32       buf = 0;
    LwU64       vaLimit = 0;
    LwU64       temp = 0;
    InstBlock*  pInstBlock;

    if (pVMemSpace == NULL)
    {
        return LW_ERR_GENERIC;
    }

    pInstBlock = &(pVMemSpace->instBlock);

    if (pInstBlock->readFn == NULL)
    {
        dprintf("**ERROR: NULL value of readFn.\n");
        return LW_ERR_NOT_SUPPORTED;
    }
    // read the part of the instance block where part of the address limit is kept
    pInstBlock->readFn(pInstBlock->instBlockAddr + SF_OFFSET(LW_RAMIN_ADR_LIMIT_LO), &buf, 4);

    // read the lower 32 bits, address is 4k aligned
    vaLimit = SF_VAL(_RAMIN, _ADR_LIMIT_LO, buf) << 12;

    // read the part of the instance block where antoher part of the PDB is kept
    pInstBlock->readFn(pInstBlock->instBlockAddr + SF_OFFSET(LW_RAMIN_ADR_LIMIT_HI), &buf, 4);

    // read the higher 32 bits of the PDB
    temp = SF_VAL(_RAMIN, _ADR_LIMIT_HI, buf);

    // these are the higher bits of the address, shift that value
    temp <<= 32;
    vaLimit = vaLimit | temp;

    return vaLimit;
}

LW_STATUS
vmemInitLayout_GP100(VMemSpace *pVMemSpace)
{
    VMemLayout *pVMemLayout;
    LwU32       bigPageShift;

    if (!pVMemSpace)
    {
        return LW_ERR_GENERIC;
    }

    if (_mmuFmtVersion_GP100(pVMemSpace) == GMMU_FMT_VERSION_1) {
        return vmemInitLayout_GK104(pVMemSpace);
    }

    bigPageShift = BIT_IDX_32(pVMemSpace->bigPageSize);
    if (!VALID_SHIFT(bigPageShift))
    {
        return LW_ERR_GENERIC;
    }

    pVMemLayout = &pVMemSpace->layout;

    memset(pVMemLayout, 0, sizeof(VMemLayout));

    pVMemLayout->fmt.gmmu.version          = GMMU_FMT_VERSION_2;
    pVMemLayout->fmt.gmmu.pPdeMulti        = &pVMemLayout->pdeMulti.gmmu;
    pVMemLayout->fmt.gmmu.pPde             = &pVMemLayout->pde.gmmu;
    pVMemLayout->fmt.gmmu.pPte             = &pVMemLayout->pte.gmmu;
    pVMemLayout->fmt.gmmu.bSparseHwSupport = LW_TRUE;

    pMmu[indexGpu].mmuFmtInitPdeApertures(pVMemLayout->pdeApertures);
    pMmu[indexGpu].mmuFmtInitPteApertures(pVMemLayout->pteApertures);

    _gmmuFmtInitPdeMulti_GP10X(&pVMemLayout->pdeMulti.gmmu, pVMemLayout->pdeApertures);
    _gmmuFmtInitPde_GP10X(&pVMemLayout->pde.gmmu, pVMemLayout->pdeApertures);
    pVmem[indexGpu].vmemGmmuFmtInitPte(&pVMemLayout->pte.gmmu, pVMemLayout->pteApertures);
    _gmmuInitMmuFmt_GP10X(pVMemLayout->fmtLevels, bigPageShift);

    return LW_OK;
}

const MMU_FMT_PTE*
vmemGetPTEFmt_GP100(VMemSpace *pVMemSpace)
{
    if (!pVMemSpace)
    {
        return NULL;
    }

    if (_mmuFmtVersion_GP100(pVMemSpace) == GMMU_FMT_VERSION_1)
    {
        return vmemGetPTEFmt_GK104(pVMemSpace);
    }

    return &pVMemSpace->layout.pte;
}

LW_STATUS
vmemGetPDEFmt_GP100(VMemSpace *pVMemSpace, VMemFmtPde *pFmtPde, LwU32 level)
{
    if (!pVMemSpace || !pFmtPde)
    {
        return LW_ERR_GENERIC;
    }

    if (_mmuFmtVersion_GP100(pVMemSpace) == GMMU_FMT_VERSION_1)
    {
        return vmemGetPDEFmt_GK104(pVMemSpace, pFmtPde, level);
    }

    if (level > 3)
    {
        return LW_ERR_GENERIC;
    }

    if (level == 3)
    {
        pFmtPde->bMulti = TRUE;
        _gmmuFmtInitPdeMulti_GP10X(&pFmtPde->fmts.multi.gmmu, pVMemSpace->layout.pdeApertures);
    }
    else
    { 
        pFmtPde->bMulti = FALSE;
        _gmmuFmtInitPde_GP10X(&pFmtPde->fmts.single.gmmu, pVMemSpace->layout.pdeApertures);
    }

    return LW_OK;
}

LW_STATUS
vmemIlwalidatePDB_GP100(VMemSpace *pVMemSpace)
{
    LwU32 pdb;
    LwU32 aperture;
    LwU32 regval;
    LwU32 pdbregval;

    if (!pVMemSpace)
    {
        return LW_ERR_GENERIC;
    }

    if (pVMemSpace->instBlock.readFn == NULL)
    {
        dprintf("**ERROR: NULL value of readFn.\n");
        return LW_ERR_NOT_SUPPORTED;
    }
    pVMemSpace->instBlock.readFn(pVMemSpace->instBlock.instBlockAddr + SF_OFFSET(LW_RAMIN_PAGE_DIR_BASE_TARGET), &pdb, 4);
    switch (SF_VAL(_RAMIN, _PAGE_DIR_BASE_TARGET, pdb))
    {
    case LW_RAMIN_PAGE_DIR_BASE_TARGET_VID_MEM:
        aperture = LW_PFB_PRI_MMU_ILWALIDATE_PDB_APERTURE_VID_MEM;
        break;
    case LW_RAMIN_PAGE_DIR_BASE_TARGET_SYS_MEM_COHERENT:
    case LW_RAMIN_PAGE_DIR_BASE_TARGET_SYS_MEM_NONCOHERENT:
        aperture = LW_PFB_PRI_MMU_ILWALIDATE_PDB_APERTURE_SYS_MEM;
        break;
    default :
        dprintf("lw: %s: unknown PAGE_DIR_BASE_TARGET (0x08%x)\n",
                __FUNCTION__, pdb);
        return LW_ERR_GENERIC;
    }

    pdb >>= LW_PFB_PRI_MMU_ILWALIDATE_PDB_ADDR_ALIGNMENT;

    pdbregval = DRF_NUM(_PFB_PRI, _MMU_ILWALIDATE_PDB, _ADDR, pdb) | 
                DRF_NUM(_PFB_PRI, _MMU_ILWALIDATE_PDB, _APERTURE, aperture);

    regval = DRF_DEF(_PFB_PRI, _MMU_ILWALIDATE, _ALL_VA,  _TRUE) |
             DRF_DEF(_PFB_PRI, _MMU_ILWALIDATE, _ALL_PDB, _TRUE) |
             DRF_DEF(_PFB_PRI, _MMU_ILWALIDATE, _TRIGGER, _TRUE);

    if (isVirtualWithSriov())
    {
        GPU_REG_WR32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_MMU_ILWALIDATE_PDB, pdbregval);
        GPU_REG_WR32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_MMU_ILWALIDATE, regval);
    }
    else
    {
        GPU_REG_WR32(LW_PFB_PRI_MMU_ILWALIDATE_PDB, pdbregval);
        GPU_REG_WR32(LW_PFB_PRI_MMU_ILWALIDATE, regval);
    }

    return LW_OK;
}

LwU32
vmemSWToHWLevel_GP100(VMemSpace *pVMemSpace, LwU32 level)
{
    if (!pVMemSpace)
    {
        return 0;
    }

    if (_mmuFmtVersion_GP100(pVMemSpace) == GMMU_FMT_VERSION_1)
    {
        return vmemSWToHWLevel_GK104(pVMemSpace, level);
    }

    switch (level)
    {
    case 0:
    case 1:
    case 2:
    case 3:
        return 3 - level;
    default:
        return 0;
    }
}

/**********************************************************************************
 *
 *
 * Internal helper functions
 *
 *
 *********************************************************************************/

static LwU32
_mmuFmtVersion_GP100(VMemSpace *pVMemSpace)
{
    InstBlock *pInstBlock = &pVMemSpace->instBlock;
    LwU32      pdb;

    if (pInstBlock->readFn == NULL)
    {
        dprintf("**ERROR: NULL value of readFn.\n");
        return LW_ERR_NOT_SUPPORTED;
    }
    pInstBlock->readFn(pInstBlock->instBlockAddr + SF_OFFSET(LW_RAMIN_PAGE_DIR_BASE_LO), &pdb, 4);

    if (SF_VAL(_RAMIN, _USE_NEW_PT_FORMAT, pdb) == LW_RAMIN_USE_NEW_PT_FORMAT_TRUE)
    {
        return GMMU_FMT_VERSION_2;
    }
    else
    {
        return GMMU_FMT_VERSION_1;
    }
}

static void 
_gmmuFmtInitPdeMulti_GP10X(GMMU_FMT_PDE_MULTI *pPdeMulti, LW_FIELD_ENUM_ENTRY *pdeApertures)
{
    GMMU_FMT_PDE *pPdeBig   = &pPdeMulti->subLevels[PDE_MULTI_BIG_INDEX];
    GMMU_FMT_PDE *pPdeSmall = &pPdeMulti->subLevels[PDE_MULTI_SMALL_INDEX];

    pPdeBig->version   = GMMU_FMT_VERSION_2;
    pPdeSmall->version = GMMU_FMT_VERSION_2;

    // Common PDE fields.
    INIT_FIELD_DESC32(&pPdeMulti->fldSizeRecipExp, LW_MMU_PDE_SIZE);

    // Dual PDE - big part.
    INIT_FIELD_APERTURE(&pPdeBig->fldAperture, LW_MMU_VER2_DUAL_PDE_APERTURE_BIG,
                        pdeApertures);
    INIT_FIELD_ADDRESS(&pPdeBig->fldAddrVidmem, LW_MMU_VER2_DUAL_PDE_ADDRESS_BIG_SYS,
                       LW_MMU_VER2_DUAL_PDE_ADDRESS_BIG_SHIFT);
    INIT_FIELD_ADDRESS(&pPdeBig->fldAddrSysmem, LW_MMU_VER2_DUAL_PDE_ADDRESS_BIG_VID,
                       LW_MMU_VER2_DUAL_PDE_ADDRESS_BIG_SHIFT);
    INIT_FIELD_BOOL(&pPdeBig->fldVolatile, LW_MMU_VER2_DUAL_PDE_VOL_BIG);

    // Dual PDE - small part.
    INIT_FIELD_APERTURE(&pPdeSmall->fldAperture, LW_MMU_VER2_DUAL_PDE_APERTURE_SMALL,
                        pdeApertures);
    INIT_FIELD_ADDRESS(&pPdeSmall->fldAddrVidmem, LW_MMU_VER2_DUAL_PDE_ADDRESS_SMALL_SYS,
                       LW_MMU_VER2_DUAL_PDE_ADDRESS_SHIFT);
    INIT_FIELD_ADDRESS(&pPdeSmall->fldAddrSysmem, LW_MMU_VER2_DUAL_PDE_ADDRESS_SMALL_VID,
                       LW_MMU_VER2_DUAL_PDE_ADDRESS_SHIFT);
    INIT_FIELD_BOOL(&pPdeSmall->fldVolatile, LW_MMU_VER2_DUAL_PDE_VOL_SMALL);
}

static void 
_gmmuFmtInitPde_GP10X(GMMU_FMT_PDE *pPde, LW_FIELD_ENUM_ENTRY *pdeApertures)
{
    pPde->version = GMMU_FMT_VERSION_2;

    INIT_FIELD_APERTURE(&pPde->fldAperture, LW_MMU_VER2_PDE_APERTURE,
                        pdeApertures);
    INIT_FIELD_ADDRESS(&pPde->fldAddrVidmem, LW_MMU_VER2_PDE_ADDRESS_VID,
                       LW_MMU_VER2_PDE_ADDRESS_SHIFT);
    INIT_FIELD_ADDRESS(&pPde->fldAddrSysmem, LW_MMU_VER2_PDE_ADDRESS_SYS,
                       LW_MMU_VER2_PDE_ADDRESS_SHIFT);
    INIT_FIELD_BOOL(&pPde->fldVolatile, LW_MMU_VER2_PDE_VOL);
}

void 
vmemGmmuFmtInitPte_GP100(GMMU_FMT_PTE *pPte, LW_FIELD_ENUM_ENTRY *pteApertures)
{
    pPte->version = GMMU_FMT_VERSION_2;

    INIT_FIELD_BOOL(&pPte->fldValid, LW_MMU_VER2_PTE_VALID);
    INIT_FIELD_APERTURE(&pPte->fldAperture, LW_MMU_VER2_PTE_APERTURE, pteApertures);
    INIT_FIELD_ADDRESS(&pPte->fldAddrSysmem, LW_MMU_VER2_PTE_ADDRESS_SYS,
                       LW_MMU_VER2_PTE_ADDRESS_SHIFT);
    INIT_FIELD_ADDRESS(&pPte->fldAddrVidmem, LW_MMU_VER2_PTE_ADDRESS_VID,
                       LW_MMU_VER2_PTE_ADDRESS_SHIFT);
    INIT_FIELD_ADDRESS(&pPte->fldAddrPeer, LW_MMU_VER2_PTE_ADDRESS_VID,
                       LW_MMU_VER2_PTE_ADDRESS_SHIFT);
    INIT_FIELD_DESC32(&pPte->fldPeerIndex, LW_MMU_VER2_PTE_ADDRESS_VID_PEER);
    INIT_FIELD_BOOL(&pPte->fldVolatile, LW_MMU_VER2_PTE_VOL);
    INIT_FIELD_BOOL(&pPte->fldReadOnly, LW_MMU_VER2_PTE_READ_ONLY);
    INIT_FIELD_BOOL(&pPte->fldPrivilege, LW_MMU_VER2_PTE_PRIVILEGE);
    INIT_FIELD_BOOL(&pPte->fldEncrypted, LW_MMU_VER2_PTE_ENCRYPTED);
    INIT_FIELD_BOOL(&pPte->fldAtomicDisable, LW_MMU_VER2_PTE_ATOMIC_DISABLE);
    INIT_FIELD_DESC32(&pPte->fldKind, LW_MMU_VER2_PTE_KIND);
    INIT_FIELD_DESC32(&pPte->fldCompTagLine, LW_MMU_VER2_PTE_COMPTAGLINE);
}

/*!
 * The format for the paging directory. pteMmuFmt does not have the const qualifier
 * since the format for the big page table varies depending on the big page size.
 *
 *  PD3 [48:47]
 *  |
 *  v
 *  PD2 [46:38]
 *  |
 *  v
 *  PD1 [37:29]
 *  |
 *  v
 *  PD0 [28:21] / PT_LARGE [28:21] (2MB page)
 *  |        \
 *  |         \
 *  v          v
 *  PT_SMALL  PT_BIG (64KB page)
 *  [20:12]   [20:16]
 */
static void 
_gmmuInitMmuFmt_GP10X(MMU_FMT_LEVEL *pLevels, LwU32 bigPageShift)
{
    pLevels[0].virtAddrBitLo = 47;
    pLevels[0].virtAddrBitHi = 48;
    pLevels[0].entrySize     = LW_MMU_VER2_PDE__SIZE;
    pLevels[0].bPageTable    = LW_FALSE;
    pLevels[0].numSubLevels  = 1;
    pLevels[0].subLevels     = &pLevels[1];

    pLevels[1].virtAddrBitLo = 38;
    pLevels[1].virtAddrBitHi = 46;
    pLevels[1].entrySize     = LW_MMU_VER2_PDE__SIZE;
    pLevels[1].bPageTable    = LW_FALSE;
    pLevels[1].numSubLevels  = 1;
    pLevels[1].subLevels     = &pLevels[2];

    pLevels[2].virtAddrBitLo = 29;
    pLevels[2].virtAddrBitHi = 37;
    pLevels[2].entrySize     = LW_MMU_VER2_PDE__SIZE;
    pLevels[2].bPageTable    = LW_FALSE;
    pLevels[2].numSubLevels  = 1;
    pLevels[2].subLevels     = &pLevels[3];

    pLevels[3].virtAddrBitLo = 21;
    pLevels[3].virtAddrBitHi = 28;
    pLevels[3].entrySize     = LW_MMU_VER2_DUAL_PDE__SIZE;
    pLevels[3].bPageTable    = LW_TRUE;
    pLevels[3].numSubLevels  = 2;
    pLevels[3].subLevels     = &pLevels[4];

    pLevels[4].virtAddrBitLo = (LwU8)bigPageShift;
    pLevels[4].virtAddrBitHi = 20;
    pLevels[4].entrySize     = LW_MMU_VER2_PTE__SIZE;
    pLevels[4].bPageTable    = LW_TRUE;
    pLevels[4].numSubLevels  = 0;
    pLevels[4].subLevels     = NULL;

    pLevels[5].virtAddrBitLo = 12;
    pLevels[5].virtAddrBitHi = 20;
    pLevels[5].entrySize     = LW_MMU_VER2_PTE__SIZE;
    pLevels[5].bPageTable    = LW_TRUE;
    pLevels[5].numSubLevels  = 0;
    pLevels[5].subLevels     = NULL;
}
