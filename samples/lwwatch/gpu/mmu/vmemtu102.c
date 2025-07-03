/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
//
// Includes
//
#include "fifo.h"
#include "fb.h"
#include "vgpu.h"

#include "turing/tu102/dev_bus.h"
#include "turing/tu102/dev_vm.h"
#include "turing/tu102/dev_mmu.h"
#include "turing/tu102/dev_ram.h"
#include "turing/tu102/dev_fb.h"

#include "g_vmem_private.h"

//-----------------------------------------------------
// vmemGetInstanceMemoryAddrForBAR1_TU102
//
// Returns the instance memory base address for BAR1.
//-----------------------------------------------------
LwU64
vmemGetInstanceMemoryAddrForBAR1_TU102(readFn_t* readFn, writeFn_t* writeFn, MEM_TYPE* pMemType)
{
    LwU32       reg;
    LwU64       instMemAddr;

    // read the instance memory description for BAR1
    reg = isVirtualWithSriov() ? 
            GPU_REG_RD32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_BAR1_BLOCK) : 
            GPU_REG_RD32(LW_PBUS_BAR1_BLOCK);

    instMemAddr = (LwU64)SF_VAL(_PBUS, _BAR1_BLOCK_PTR, reg) << 12;

    if (SF_VAL(_PBUS, _BAR1_BLOCK_TARGET, reg) == LW_PBUS_BAR1_BLOCK_TARGET_VID_MEM)
    {
        if (readFn)
            *readFn = pFb[indexGpu].fbRead;
        if (writeFn)
            *writeFn = pFb[indexGpu].fbWrite;
        if (pMemType)
            *pMemType = FRAMEBUFFER;
    }
    else
    {
        if (readFn)
            *readFn = readSystem;
        if (writeFn)
            *writeFn = writeSystem;
        if (pMemType)
            *pMemType = SYSTEM_PHYS;
    }

    return instMemAddr;
}

//-----------------------------------------------------
// vmemGetInstanceMemoryAddrForBAR2_TU102
//
// Returns the instance memory base address for BAR2.
//-----------------------------------------------------
LwU64
vmemGetInstanceMemoryAddrForBAR2_TU102(readFn_t* readFn, writeFn_t* writeFn, MEM_TYPE* pMemType)
{
    LwU32       reg;
    LwU64       instMemAddr;

    // read the instance memory description for BAR1
    reg = isVirtualWithSriov() ? 
            GPU_REG_RD32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_BAR2_BLOCK) : 
            GPU_REG_RD32(LW_PBUS_BAR2_BLOCK);

    instMemAddr = (LwU64)SF_VAL(_PBUS, _BAR2_BLOCK_PTR, reg) << 12;

    if (SF_VAL(_PBUS, _BAR2_BLOCK_TARGET, reg) == LW_PBUS_BAR2_BLOCK_TARGET_VID_MEM)
    {
        if (readFn)
            *readFn = pFb[indexGpu].fbRead;
        if (writeFn)
            *writeFn = pFb[indexGpu].fbWrite;
        if (pMemType)
            *pMemType = FRAMEBUFFER;
    }
    else
    {
        if (readFn)
            *readFn = readSystem;
        if (writeFn)
            *writeFn = writeSystem;
        if (pMemType)
            *pMemType = SYSTEM_PHYS;
    }

    return instMemAddr;
}

void
vmemRebindBAR1Block_TU102(void)
{
    LwU32 reg;

    if (isVirtualWithSriov())
    {
        reg = GPU_REG_RD32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_BAR1_BLOCK);
        GPU_REG_WR32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_BAR1_BLOCK, reg);
    }
    else
    {
        reg = GPU_REG_RD32(LW_PBUS_BAR1_BLOCK);
        GPU_REG_WR32(LW_PBUS_BAR1_BLOCK, reg);
    }
}

void 
vmemGmmuFmtInitPte_TU102(GMMU_FMT_PTE *pPte, LW_FIELD_ENUM_ENTRY *pteApertures)
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

LW_STATUS
vmemIlwalidatePDB_TU102(VMemSpace *pVMemSpace)
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

