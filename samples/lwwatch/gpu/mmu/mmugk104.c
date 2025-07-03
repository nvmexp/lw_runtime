/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// mmugk104.c - page table routines for Kepler
//
//*****************************************************

//
// Includes
//
#include "fifo.h"
#include "kepler/gk104/dev_bus.h"
#include "kepler/gk104/dev_mmu.h"
#include "kepler/gk104/dev_ram.h"

#include "g_mmu_private.h"     // (rmconfig)  implementation prototypes
#include "g_vmem_private.h"

#include "bus.h"


/*
 * From GK104 dev_mmu.ref:
 *    Each PDE points to two page tables.  One page table maps memory using big pages.
 *    The other maps memory using small pages.  Small pages are always 4KB.  Big pages
 *    are 64KB or 128KB depending on a PRI bit in the MMU.  Each big page page tables
 *    contains 1K entries, regardless of the size of the big pages.  Each small page
 *    page table contains 16K entries when using 64KB big pages, or 32K entries when
 *    using 128KB big pages.
 *    ...
 *    Each PDE maps a 64MB region of virtual memory when using 64KB big pages, or 128MB
 *    when using 128KB big pages.  To map all 40b of virtual address space, the page
 *    directory consists of 16K entries when using 64KB big pages (64MB * 16K = 2^26 * 2^14 = 2^40),
 *    or 8K entries when using 128KB big pages (128MB * 8K = 2^27 * 2^13 = 2^40).
 *
 *      In other words....
 * *****MMU PAGE TABLE STRUCTURE*****
 *
 *    PDB  ->|PDE TABLE|        /-----> |PTE small(4KB) TABLE   |
 *           |_________|       /        |_______________________|
 *           |_________|------<         |_______________________| ----> |4KB Physical memory page|
 *           |_________|       \        |_______________________|       |________________________|
 *           |_________|        \       (16K entries when 64KB big pages used)
 *           |_________|         \      (32K entries when 128KB big  pages used)
 *           |_________|          \
 *  PDE TABLE HAS                  \---->|PTE BIG(64||128KB) TABLE|
 *  8KB ENTRIES WHEN 128KB pages used    |________________________|
 *  16KB ENTRIES WHEN 64KB pages used    |________________________| ---> |64||128KB Physical memory page|
 *                                       |________________________|      |______________________________|
 *  PTE SMALL TABLE HAS                  |________________________|      |______________________________|
 *  16K ENTRIES WHEN 64KB pages used     |________________________|
 *  32K ENTRIES WHEN 128KB pages used    |________________________|
 *                                       |________________________|
 *                                       (--1024 entries always--)
 *  PTE BIG TABLE have always 1K of entries
 *  SIZE OF BIG PAGE TABLE
 *
 * Size of big pages is determined by LW_PFB_PRI_MMU_CTRL_VM_PG_SIZE
 *
 * *****VIRTUAL ADDRESS MAPPING*****
 *  For 128kB pages, i.e 128kB big pages
 *  |39         27|26       17|16                   0|
 *  | PDE INDEX   | PTE INDEX | Offset within page   |
 *
 *  For 64kB pages, i.e 64kB big pages
 *  |39           26|25       16|15                 0|
 *  | PDE INDEX     | PTE INDEX | Offset within page |
 *
 *  For 4kB pages, with 128kB big pages
 *  |39         27|26           12|11               0|
 *  | PDE INDEX   | PTE INDEX     |Offset within page|
 *
 *  For 4kB pages, with 64kB big pages
 *  |39           26|25         12|11               0|
 *  | PDE INDEX     | PTE INDEX   |Offset within page|
 */

LW_STATUS vmemPdeCheck_GK104(VMemSpace *pVMemSpace, LwU32 pdeId);

typedef union _BAR1PARAM
{
    struct
    {
        LwU32 bar1SnoopEnable : 1;
        LwU32 remapperEnableMask : 8;
        LwU32 p2pWMboxAddr : 8;
    } bits;
    LwU64 value;
} BAR1PARAM;

void
mmuDisableBar1ActiveRemappers_GK104(void *pParam)
{
    BAR1PARAM* pBar1Param = (BAR1PARAM*)pParam;
    LwU32 i, reg, wmBoxEnabledMask;

    pBar1Param->value = 0;

    //Disable any active remappers
    for (i = 0; i < LW_PBUS_BL_REMAP_1__SIZE_1; i++)
    {
        reg = GPU_REG_RD32(LW_PBUS_BL_REMAP_1(i));

        if (FLD_TEST_DRF(_PBUS, _BL_REMAP_1, _BL_ENABLE, _ON, reg))
        {
            reg = FLD_SET_DRF(_PBUS, _BL_REMAP_1, _BL_ENABLE, _OFF, reg);
            GPU_REG_WR32(LW_PBUS_BL_REMAP_1(i), reg);
            pBar1Param->bits.remapperEnableMask |= BIT(i);
        }
    }

    //Disable the P2P mailboxes
    if (LW_OK == pBus[indexGpu].busDisableWmBoxes(&wmBoxEnabledMask))
    {
        pBar1Param->bits.p2pWMboxAddr = wmBoxEnabledMask;
    }
}

void
mmuReenableBar1ActiveRemappers_GK104(LwU64 param)
{
    BAR1PARAM bar1Param;
    LwU32 i, reg;

    bar1Param.value = param;

    //Re-enable any active remappers
    for (i = 0; i < LW_PBUS_BL_REMAP_1__SIZE_1; i++)
    {
        if (bar1Param.bits.remapperEnableMask & BIT(i))
        {
            reg = GPU_REG_RD32(LW_PBUS_BL_REMAP_1(i));
            reg = FLD_SET_DRF(_PBUS, _BL_REMAP_1, _BL_ENABLE, _ON, reg);
            GPU_REG_WR32(LW_PBUS_BL_REMAP_1(i), reg);
        }
    }
    //Re-enable the P2P mailboxes
    pBus[indexGpu].busEnableWmBoxes(bar1Param.bits.p2pWMboxAddr);
}

/*!
 *  Walks through all the page tables in all contexts on the gpu and verifies
 *  that the page tables are self-consistent.
 *
 *  @return   LW_OK , LW_ERR_GENERIC.
 */
LW_STATUS mmuPteValidate_GK104( void )
{
    PRINT_LWWATCH_NOT_IMPLEMENTED_MESSAGE_AND_RETURN0();
}

//-----------------------------------------------------
// mmuGetPDETableStartAddress_GK104
//
// Returns the page directory base for a instance memory block's
// virtual memory space.
//-----------------------------------------------------
LwU64
mmuGetPDETableStartAddress_GK104(LwU64 instMemAddr, readFn_t instMemReadFn)
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

    // this is the page direcotry base address for this channel
    return pPDE;
}

/*!
 *  Parse through the page table entries and check for anomalies
 *  Used pdeCheck to parse and check for inconsistencies

 *  @param[in]  chid        The channel to be checked
 *  @param[in]  pdeId       Pde index to be checked
 *
 *  @return   LW_OK , LW_ERR_GENERIC.
 */
LW_STATUS
mmuPdeCheck_GK104(LwU32 chid, LwU32 pdeId)
{
    LW_STATUS   status = LW_OK;
    VMemSpace   vMemSpace;
    VMEM_INPUT_TYPE Id;
    memset(&Id, 0, sizeof(Id));
    Id.ch.chId = chid;

    dprintf("\n\t lw: PDE check for Channel #%d\n", chid);
    status = vmemGet(&vMemSpace, VMEM_TYPE_CHANNEL, &Id);
    if (status == LW_ERR_GENERIC)
    {
        dprintf("lw: Could not fetch vmemspace for chId 0x%x \n", chid);
        return LW_ERR_GENERIC;
    }

    return vmemPdeCheck_GK104(&vMemSpace, pdeId);
}

void
mmuFmtInitPdeApertures_GK104(LW_FIELD_ENUM_ENTRY *pEntries)
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
mmuFmtInitPteApertures_GK104(LW_FIELD_ENUM_ENTRY *pEntries)
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
