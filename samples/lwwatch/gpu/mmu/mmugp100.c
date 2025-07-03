/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "pascal/gp100/dev_mmu.h"
#include "pascal/gp100/dev_ram.h"
#include "fifo.h"

#include "g_mmu_private.h"


//-----------------------------------------------------
// mmuGetPDETableStartAddress_GP100
//
// Returns the page directory base for a instance memory block's
// virtual memory space.
//-----------------------------------------------------
LwU64
mmuGetPDETableStartAddress_GP100(LwU64 instMemAddr, readFn_t instMemReadFn)
{
    LwU32       buf = 0;
    LwU64       pPDE = 0;
    LwU64       temp = 0;

    // read the part of the instance block where part of the PDB is kept
    instMemReadFn(instMemAddr + SF_OFFSET(LW_RAMIN_PAGE_DIR_BASE_LO), &buf, 4);
    // read the PDB address lower 32 bits, PDB is 4k aligned
    pPDE = SF_VAL(_RAMIN, _PAGE_DIR_BASE_LO, buf) << 12;
    // read the part of the instance block where another part of the PDB is kept
    instMemReadFn(instMemAddr + SF_OFFSET(LW_RAMIN_PAGE_DIR_BASE_HI), &buf, 4);
    // read the higher 32 bits of the PDB
    temp = SF_VAL(_RAMIN, _PAGE_DIR_BASE_HI, buf);
    // these are the higher bits of the address, shift that value
    temp <<= 32;
    pPDE += temp;

    // this is the page directory base address for this channel
    return pPDE;
}

void
mmuFmtInitPdeApertures_GP100(LW_FIELD_ENUM_ENTRY *pEntries)
{
    lwFieldEnumEntryInit(pEntries + GMMU_APERTURE_ILWALID,
                         LW_MMU_PDE_APERTURE_BIG_ILWALID);

    lwFieldEnumEntryInit(pEntries + GMMU_APERTURE_VIDEO,
                         LW_MMU_PDE_APERTURE_BIG_VIDEO_MEMORY);

    lwFieldEnumEntryInit(pEntries + GMMU_APERTURE_SYS_COH,
                         LW_MMU_PDE_APERTURE_BIG_SYSTEM_COHERENT_MEMORY);

    lwFieldEnumEntryInit(pEntries + GMMU_APERTURE_SYS_NONCOH,
                         LW_MMU_PDE_APERTURE_BIG_SYSTEM_NON_COHERENT_MEMORY);
}

void
mmuFmtInitPteApertures_GP100(LW_FIELD_ENUM_ENTRY *pEntries)
{
    lwFieldEnumEntryInit(pEntries + GMMU_APERTURE_VIDEO,
                         LW_MMU_PTE_APERTURE_VIDEO_MEMORY);

    lwFieldEnumEntryInit(pEntries + GMMU_APERTURE_PEER,
                         LW_MMU_PTE_APERTURE_PEER_MEMORY);

    lwFieldEnumEntryInit(pEntries + GMMU_APERTURE_SYS_COH,
                         LW_MMU_PTE_APERTURE_SYSTEM_COHERENT_MEMORY);

    lwFieldEnumEntryInit(pEntries + GMMU_APERTURE_SYS_NONCOH,
                         LW_MMU_PTE_APERTURE_SYSTEM_NON_COHERENT_MEMORY);
}
