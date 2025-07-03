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
// vmemgk104.c - page table routines for Kepler
//
//*****************************************************

//
// Includes
//
#include "fifo.h"
#include "kepler/gk104/dev_bus.h"
#include "kepler/gk104/dev_mmu.h"
#include "kepler/gk104/dev_fb.h"
#include "kepler/gk104/dev_ram.h"
#include "kepler/gk104/dev_fifo.h"
#include "kepler/gk104/dev_pwr_pri.h"
#include "turing/tu102/dev_vm.h"

#include "pmu.h"
#include "fb.h"
#include "vgpu.h"

#include "g_mmu_private.h"     // (rmconfig)  implementation prototypes
#include "g_vmem_private.h"

//
// Definitions
//
#define VALID_SHIFT(s) ((s) == 16 || (s) == 17)

#define MMU_PDE_PAGE_SIZE_IS_4KB_BIG_128KB          0
#define MMU_PDE_PAGE_SIZE_IS_4KB_BIG_64KB           1
#define MMU_PDE_PAGE_SIZE_IS_64KB                   2
#define MMU_PDE_PAGE_SIZE_IS_128KB                  3

#define SIZEOF_VA_SPACE_PER_PDE_MAP(bigpagesz)      (bigpagesz * 1024)
#define PDE_INDEX(bigpagesz, v)                     ((v) / SIZEOF_VA_SPACE_PER_PDE_MAP(bigpagesz))
#define PTE_INDEX(bigpagesz, v, pagesz)             ((v) % SIZEOF_VA_SPACE_PER_PDE_MAP(bigpagesz) / (pagesz))
#define PAGE_OFFSET(v, pagesz)                      ((v) % (pagesz))
#define PAGETABLE_INDEX_SMALL                       0
#define PAGETABLE_INDEX_BIG                         1
#define PAGETABLE_COUNT                             2

#define PRINT_FIELD_BOOL(fmt, fmtPte, field, pte)                           \
    do {                                                                    \
        if (lwFieldIsValid32(&(fmtPte)->fld##field.desc))                   \
            dprintf(fmt, lwFieldGetBool(&(fmtPte)->fld##field, (pte)->v8)); \
    } while (0)

#define FERMI_VASPACE_A                             (0x000090f1)

//
// Helper functions
//
static LwU64 _getPTEByIndex_GK104(VMemSpace *pVMemSpace, LwU32 pteIndex, LwU32 bigPageSize, LwU32 pageSize, PdeEntry *pPdeEntry,
                                  PteEntry *pPteEntry, BOOL *pValid, BOOL dump);
static LwU64 _getPTE_GK104(VMemSpace *pVMemSpace, LwU64 va, LwU32 bigPageSize, LwU32 pageSize, PdeEntry *pPdeEntry, PteEntry *pPteEntry,
                           BOOL *pValid, BOOL dump);
static LwU64 _getInstanceMemoryAddr_GK104(VMemTypes vMemType, ChannelId *pChannelId, readFn_t* instMemReadFn, writeFn_t* writeMemReadFN, MEM_TYPE* pMemType);

LW_STATUS printPageTable_GK104(VMemSpace *pVMemSpace, LwU64 pPTE, char* printPteType, LwU32 begin, LwU32 end, readFn_t pteReadFunc);
LwU64 getInstanceMemoryAddrForPMU_GK104 (readFn_t* readFn, writeFn_t* writeFn, MEM_TYPE* pMemType);

//Added
LW_STATUS vmemreadGpuVirtualAddr_GK104(VMemSpace *pVMemSpace, LwU64 virtAddr, void* buffer, LwU32 length);
LwU64 getPTBaseAddr_GK104(PdeEntry* pPde, BOOL isBig, readFn_t *pteReadFn);
LW_STATUS fillVMemSpace_GK104(VMemSpace *pVMemSpace, LwU64 instMemAddr, readFn_t instMemReadFn, writeFn_t writeMemReadFn, MEM_TYPE memType);
LW_STATUS vmemPdeCheck_GK104(VMemSpace *pVMemSpace, LwU32 pdeId);



/*!
 *  Get the VMemSpace for a given channel.
 *
 *  @param[out] pVMemSpace  Virtual memory space structure to populate.
 *  @param[in]  pCh         Pointer Channel Struct
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemGetByChId_GK104
(
    VMemSpace *pVMemSpace,
    VMEM_INPUT_TYPE_CHANNEL *pCh
)
{
    LwU64       instMemAddr=0;
    readFn_t    instMemReadFn;
    writeFn_t   instMemWriteFn;
    VMemTypes   vMemType = VMEM_TYPE_CHANNEL;
    MEM_TYPE    memType;
    ChannelId   channelId;
    LW_STATUS   status;

    if (pVMemSpace == NULL)
    {
        return LW_ERR_GENERIC;
    }

    status = pFifo[indexGpu].fifoGetDeviceInfo();
    if (LW_OK != status)
    {
        return status;
    }

    channelId.bRunListValid = LW_TRUE;
    channelId.bChramPriBaseValid  = LW_FALSE;
    channelId.id = pCh->chId;
    channelId.runlistId = pCh->rlId;

    instMemAddr = _getInstanceMemoryAddr_GK104(vMemType, &channelId, &instMemReadFn, &instMemWriteFn, &memType);
    if (instMemAddr == 0)
    {
        dprintf("lwwatch:    Channel does not exist\n");
        return LW_ERR_GENERIC;
    }

    if (fillVMemSpace_GK104(pVMemSpace, instMemAddr, instMemReadFn, instMemWriteFn, memType) != LW_OK)
    {
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}

/*!
 *  Get the VMemSpace for a given instance block pointer.
 *
 *  @param[out] pVMemSpace      Virtual memory space structure to populate.
 *  @param[in]  pInst           Pointer Instance Block Struct
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemGetByInstPtr_GK104
(
    VMemSpace   *pVMemSpace,
    VMEM_INPUT_TYPE_INST *pInst
)
{
    readFn_t    readFn = 0;
    writeFn_t   writeFn = 0;
    MEM_TYPE    memType;

    if (pVMemSpace == NULL)
    {
        return LW_ERR_GENERIC;
    }

    memType = (MEM_TYPE)pInst->targetMemType;

    switch (memType)
    {
        case FRAMEBUFFER:
        {
            readFn = pFb[indexGpu].fbRead;
            writeFn = pFb[indexGpu].fbWrite;
            break;
        }
        case SYSTEM_PHYS:
        {
            readFn = readSystem;
            writeFn = writeSystem;
            break;
        }
        default:
        {
            dprintf("lw: Invalid aperture\n");
            return LW_ERR_GENERIC;
        }
    }

    if (fillVMemSpace_GK104(pVMemSpace, pInst->instPtr, readFn, writeFn, memType) != LW_OK)
    {
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}

/*!
 *  Get the VMemSpace for a BAR1.
 *
 *  @param[out] pVMemSpace  Virtual memory space structure to populate.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemGetBar1_GK104(VMemSpace *pVMemSpace)
{
    LwU64       instMemAddr=0;
    readFn_t    instMemReadFn;
    writeFn_t   instMemWriteFn;
    VMemTypes   vMemType = VMEM_TYPE_BAR1;
    MEM_TYPE    memType;
    ChannelId   channelId;

    if (pVMemSpace == NULL)
    {
        return LW_ERR_GENERIC;
    }

    channelId.bRunListValid = LW_FALSE;
    channelId.bChramPriBaseValid  = LW_FALSE;
    channelId.id = 0;
    instMemAddr = _getInstanceMemoryAddr_GK104(vMemType, &channelId, &instMemReadFn, &instMemWriteFn, &memType);
    if (instMemAddr == 0)
    {
        dprintf("lwwatch:    Channel does not exist\n");
        return LW_ERR_GENERIC;
    }
    if (fillVMemSpace_GK104(pVMemSpace, instMemAddr, instMemReadFn, instMemWriteFn, memType) != LW_OK)
    {
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}


/*!
 *  Get the VMemSpace for a BAR2.
 *
 *  @param[out] pVMemSpace  Virtual memory space structure to populate.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemGetBar2_GK104(VMemSpace *pVMemSpace)
{
    LwU64       instMemAddr=0;
    readFn_t    instMemReadFn;
    writeFn_t   instMemWriteFn;
    VMemTypes   vMemType = VMEM_TYPE_BAR2;
    MEM_TYPE    memType;
    ChannelId   channelId;

    if (pVMemSpace == NULL)
    {
        return LW_ERR_GENERIC;
    }

    channelId.bRunListValid = LW_FALSE;
    channelId.bChramPriBaseValid  = LW_FALSE;
    channelId.id = 0;
    instMemAddr = _getInstanceMemoryAddr_GK104(vMemType, &channelId, &instMemReadFn, &instMemWriteFn, &memType);
    if (instMemAddr == 0)
    {
        dprintf("lwwatch:    Channel does not exist\n");
        return LW_ERR_GENERIC;
    }
    if (fillVMemSpace_GK104(pVMemSpace, instMemAddr, instMemReadFn, instMemWriteFn, memType) != LW_OK)
    {
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}


/*!
 *  Get the VMemSpace for a IFB.
 *
 *  @param[out] pVMemSpace  Virtual memory space structure to populate.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemGetIfb_GK104(VMemSpace *pVMemSpace)
{
    LwU64       instMemAddr=0;
    readFn_t    instMemReadFn;
    writeFn_t   instMemWriteFn;
    VMemTypes   vMemType = VMEM_TYPE_IFB;
    MEM_TYPE    memType;
    ChannelId   channelId;

    if (pVMemSpace == NULL)
    {
        return LW_ERR_GENERIC;
    }

    channelId.bRunListValid = LW_FALSE;
    channelId.bChramPriBaseValid  = LW_FALSE;
    channelId.id = 0;
    instMemAddr = _getInstanceMemoryAddr_GK104(vMemType, &channelId, &instMemReadFn, &instMemWriteFn, &memType);
    if (instMemAddr == 0)
    {
        dprintf("lwwatch:    Channel does not exist\n");
        return LW_ERR_GENERIC;
    }

    if (fillVMemSpace_GK104(pVMemSpace, instMemAddr, instMemReadFn, instMemWriteFn, memType) != LW_OK)
    {
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}

/*!
 *  Get the @a VMemSpace for a PMU.
 *
 *  @param[out] pVMemSpace  Virtual memory space structure to populate.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemGetPmu_GK104(VMemSpace *pVMemSpace)
{
    LwU64       instMemAddr=0;
    readFn_t    instMemReadFn;
    writeFn_t   instMemWriteFn;
    VMemTypes   vMemType = VMEM_TYPE_PMU;
    MEM_TYPE    memType;
    ChannelId   channelId;

    if (pVMemSpace == NULL)
    {
        return LW_ERR_GENERIC;
    }

    channelId.bRunListValid = LW_FALSE;
    channelId.bChramPriBaseValid  = LW_FALSE;
    channelId.id = 0;
    instMemAddr = _getInstanceMemoryAddr_GK104(vMemType, &channelId, &instMemReadFn, &instMemWriteFn, &memType);
    if (instMemAddr == 0)
    {
        dprintf("lwwatch:    Channel does not exist\n");
        return LW_ERR_GENERIC;
    }
    if (fillVMemSpace_GK104(pVMemSpace, instMemAddr, instMemReadFn, instMemWriteFn, memType) != LW_OK)
    {
        return LW_ERR_GENERIC;
    }

    // Following is taken from pmugt212.c comment.
    // TODO: This is not correct; however, the exact number of PDEs in the PMU
    //       table is not quite known at the moment. Working to find out what
    //       the value should be. In the mean time, just let anything work.
    //
    pVMemSpace->pdeCount = ~0;

    return LW_OK;
}


static LwU64
_getPTEByIndex_GK104
(
    VMemSpace *pVMemSpace,
    LwU32 pteIndex,
    LwU32 bigPageSize,
    LwU32 pageSize,
    PdeEntry *pPdeEntry,
    PteEntry *pPteEntry,
    BOOL *pValid,
    BOOL dump
)
{
    readFn_t    pteReadFunc;
    LwU64       pteOffset = 0;

    pteOffset = getPTBaseAddr_GK104(pPdeEntry, (pageSize != VMEM_SMALL_PAGESIZE), &pteReadFunc);   //TRUE for big

    if (pteOffset == 0)
        return 0;

    pteOffset += pteIndex * LW_MMU_PTE__SIZE;
    pteReadFunc(pteOffset + SF_OFFSET(LW_MMU_PTE_VALID), &pPteEntry->w0, 4);
    pteReadFunc(pteOffset + SF_OFFSET(LW_MMU_PTE_VOL), &pPteEntry->w1, 4);
    *pValid = SF_VAL(_MMU, _PTE_VALID, pPteEntry->w0);

    if (dump)
    {
        if (pageSize != VMEM_SMALL_PAGESIZE)
        {
            dprintf("lwwatch: => Big (%dk) PTE[0x%x] at " LwU40_FMT "\n",
                bigPageSize / 1024, pteIndex, pteOffset);
        }
        else
        {
            dprintf("lwwatch: => Small (4k) PTE[0x%x] at " LwU40_FMT "\n",
                pteIndex, pteOffset);
        }

        vmemDumpPte(pVMemSpace, (GMMU_ENTRY_VALUE*)pPteEntry);
    }

    return pteOffset;
}

static LwU64
_getPTE_GK104(VMemSpace *pVMemSpace, LwU64 va, LwU32 bigPageSize, LwU32 pageSize, PdeEntry *pPdeEntry, PteEntry *pPteEntry, BOOL *pValid, BOOL dump)
{
    LwU32       pteIndex = 0;

    pteIndex = (LwU32)PTE_INDEX(bigPageSize, va, pageSize);

    return _getPTEByIndex_GK104(pVMemSpace, pteIndex, bigPageSize, pageSize, pPdeEntry, pPteEntry, pValid, dump);
}

LwU32
vmemGetBigPageSize_GK104(VMemSpace *pVMemSpace)
{
    return (GPU_REG_RD_DRF(_PFB_PRI,_MMU_CTRL, _VM_PG_SIZE) == LW_PFB_PRI_MMU_CTRL_VM_PG_SIZE_128KB)?
            VMEM_BIG_PAGESIZE_128K : VMEM_BIG_PAGESIZE_64K;
}

LW_STATUS
vmemSetBigPageSize_GK104(VMemSpace *pVMemSpace, LwU32 pageSize)
{
    LwU32 reg;

    // Read the base value (in case we need to change it)
    reg = GPU_REG_RD32(LW_PFB_PRI_MMU_CTRL);

    // Switch on the page size value
    switch(pageSize)
    {
        case VMEM_BIG_PAGESIZE_64K:

            // Set the requested page size
            reg = FLD_SET_DRF(_PFB_PRI, _MMU_CTRL, _VM_PG_SIZE, _64KB, reg);
            GPU_REG_WR32(LW_PFB_PRI_MMU_CTRL, reg);

            break;

        case VMEM_BIG_PAGESIZE_128K:

            // Set the requested page size
            reg = FLD_SET_DRF(_PFB_PRI,_MMU_CTRL, _VM_PG_SIZE, _128KB, reg);
            GPU_REG_WR32(LW_PFB_PRI_MMU_CTRL, reg);

            break;

        default:

            return LW_ERR_GENERIC;
    }
    return LW_OK;
}

/*!
 *  Read PDE for @a va from @a pVMemSpace. If @a va is not in the valid
 *  allowed range for PDE indices, an error message is printed and the call
 *  returns failure. Any fatal error will print an error message.
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  va          Virtual Address to read.
 *  @param[out] pPde        PDE structure to populate.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemPdeGetByVa_GK104(VMemSpace *pVMemSpace, LwU64 va, PdeEntry *pPde)
{
    LwU32 pdeIndex;

    pdeIndex = (LwU32)PDE_INDEX(pVMemSpace->bigPageSize, va);

    return pVmem[indexGpu].vmemPdeGetByIndex(pVMemSpace, pdeIndex, (GMMU_ENTRY_VALUE*)pPde);
}

/*!
 *  Write PDE at @a index from @a pVMemSpace. If @a index is not in the valid
 *  allowed range for PDE indices, an error message is printed and the call
 *  returns failure. Any fatal error will print an error message.
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  pdeIndex    PDE index to write.
 *  @param[in]  pde         PDE structure to write.
 *  @param[in]  ilwalidate  Should the page tables be ilwalidated.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemWritePde_GK104
(
    VMemSpace  *pVMemSpace,
    LwU32       pdeIndex,
    PdeEntry    pde,
    BOOL        ilwalidate
)
{
    LwU64   pdeBase;
    LW_STATUS   status;
    LwU32   pageDirBase;
    LwU32   ilwalidateType;
    writeFn_t   writePdeFn;

    if (pdeIndex >= pVMemSpace->pdeCount)
    {
        dprintf("lw: %s: PDE #%d is greater than max PDE table size (%d)\n",
                __FUNCTION__, pdeIndex, pVMemSpace->pdeCount);
        return LW_ERR_GENERIC;
    }

    pdeBase = pVMemSpace->PdeBase + (pdeIndex * LW_MMU_PDE__SIZE);

    pVMemSpace->instBlock.readFn(pVMemSpace->instBlock.instBlockAddr + SF_OFFSET(LW_RAMIN_PAGE_DIR_BASE_TARGET), &pageDirBase, 4);
    switch (SF_VAL(_RAMIN, _PAGE_DIR_BASE_TARGET, pageDirBase))
    {
    case LW_RAMIN_PAGE_DIR_BASE_TARGET_VID_MEM:
        writePdeFn = pFb[indexGpu].fbWrite;
        ilwalidateType = LW_PFB_PRI_MMU_ILWALIDATE_PDB_APERTURE_VID_MEM;
        break;
    break;
    case LW_RAMIN_PAGE_DIR_BASE_TARGET_SYS_MEM_COHERENT:
    case LW_RAMIN_PAGE_DIR_BASE_TARGET_SYS_MEM_NONCOHERENT:
        writePdeFn = writeSystem;
        ilwalidateType = LW_PFB_PRI_MMU_ILWALIDATE_PDB_APERTURE_SYS_MEM;
        break;
    default :
        dprintf("lw: %s: unknown PAGE_DIR_BASE_TARGET (0x08%x)\n",
                __FUNCTION__, pageDirBase);
        return LW_ERR_GENERIC;
        break;
    }

    CHECK(writePdeFn(pdeBase, &pde.w0, 4));
    CHECK(writePdeFn(pdeBase+sizeof(LwU32), &pde.w1, 4));

    //ilwalidate the pde
    if (ilwalidate)
    {
        CHECK(pVmem[indexGpu].vmemIlwalidatePDB(pVMemSpace));
    }

    return LW_OK;
}

/*!
 *  Perform any operations necessary before hijacking IFB for virtual accesses.
 *
 *  Overrides an IFB PDE to reference the VA/PDE passed in
 *
 *  @param[in]      pVMemSpace  Virtual memory space.
 *  @param[in]      va          VA that we want to move up
 *  @param[in]      pde         PDE that be written to IFB address space
 *  @param[out]     pOrigIfbPde The original PDE that will need to be restored at EndIfbMapping time
 *  @param[out]     param       This exact value must be preserved passed back into EndBar1Mapping
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
#define STAGING_IFBPDE_INDEX      (8)

LW_STATUS
vmemBeginIfbMapping_GK104(
    VMemSpace    *pVMemSpace,
    LwU64         va,
    PdeEntry     *pPde,
    PdeEntry     *pOrigIfbPde,
    LwU64        *pParam
)
{
    LW_STATUS status;
    VMemSpace vmemIfb;
    LwU32 ifbBlock;
    LwU64 ifbOffset;

    ifbBlock = GPU_REG_RD32(LW_PBUS_IFB_BLOCK);
    if (DRF_VAL(_PBUS, _IFB_BLOCK, _MAP, ifbBlock) == 0 ||
        FLD_TEST_DRF(_PBUS, _IFB_BLOCK, _MODE, _PHYSICAL, ifbBlock))
    {
        dprintf("lw: %s: IFB block not configured correctly 0x%x.\n",
                __FUNCTION__, ifbBlock);
        return LW_ERR_GENERIC;
    }

    ifbOffset = (SIZEOF_VA_SPACE_PER_PDE_MAP(pVMemSpace->bigPageSize)-1) & va;

    //backup the old IFB RD/WR address so we can restore it a EndIfbMapping
    *pParam = GPU_REG_RD32(LW_PBUS_IFB_RDWR_ADDR_HI);
    *pParam <<= 32;
    *pParam |= GPU_REG_RD32(LW_PBUS_IFB_RDWR_ADDR);

    CHECK(vmemGetIfb_GK104(&vmemIfb));

    {
        LwU64 pdeAddr = vmemIfb.PdeBase + (STAGING_IFBPDE_INDEX * LW_MMU_PDE__SIZE);
        CHECK(vmemIfb.readFn(pdeAddr, pOrigIfbPde, LW_MMU_PDE__SIZE));
    }

    CHECK(vmemWritePde_GK104(&vmemIfb, STAGING_IFBPDE_INDEX, *pPde, TRUE));

    ifbOffset += STAGING_IFBPDE_INDEX * SIZEOF_VA_SPACE_PER_PDE_MAP(pVMemSpace->bigPageSize);

    GPU_REG_WR32(LW_PBUS_IFB_RDWR_ADDR_HI, LwU64_HI32(ifbOffset));
    GPU_REG_WR32(LW_PBUS_IFB_RDWR_ADDR,    LwU64_LO32(ifbOffset));

    return LW_OK;
}

/*!
 *  Restore any state modified at BeginIfbMapping time
 *
 *  @param[in]      origIfbPde  The original PDE (returned by BeginIfbMapping) that is restored by this function
 *  @param[in]      param       Value returned by BeginIfbMapping which must be passed back in
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemEndIfbMapping_GK104(
    PdeEntry     origIfbPde,
    LwU64        param
)
{
    VMemSpace vmemIfb;
    LW_STATUS status;

    GPU_REG_WR32(LW_PBUS_IFB_RDWR_ADDR_HI, LwU64_HI32(param));
    GPU_REG_WR32(LW_PBUS_IFB_RDWR_ADDR,    LwU64_LO32(param));

    CHECK(vmemGetIfb_GK104(&vmemIfb));

    CHECK(vmemWritePde_GK104(&vmemIfb, STAGING_IFBPDE_INDEX, origIfbPde, TRUE));

    return LW_OK;
}

/*!
 *  Read 4B from the current IFB RDWR_ADDR - relies on autoincrementing so
 *  next read/write will access next 4B
 *
 *  @returns data.
 */
LwU32
vmemReadIfb_GK104()
{
    LwU32 data;

    data = GPU_REG_RD32(LW_PBUS_IFB_RDWR_DATA);

    return data;
}

/*!
 *  Write 4B to the current IFB RDWR_ADDR - relies on autoincrementing so\
 *  next read will access next 4B
 *
 *  @returns LW_OK.
 */
LwU32
vmemWriteIfb_GK104(LwU32 data)
{
    GPU_REG_WR32(LW_PBUS_IFB_RDWR_DATA, data);

    return LW_OK;
}

/*!
 *  Dump out PTEs for a given PDE. This function is the interface to exts.c and
 *  HAL specific. Dumps out both page tables (big/small) if both are valid.
 *
 *  @param[in]  pVMemSpace  vmemspace for target channel/BAR etc
 *  @param[in]  pdeIndex    PDE index where PTE indexes reside.
 *  @param[in]  begin       Starting PTE index.
 *  @param[in]  end         Ending PTE index.
 */
void
vmemDoPteDump_GK104(VMemSpace *pVMemSpace, LwU32 pdeIndex, LwU32 begin, LwU32 end)
{
    PdeEntry    pde;

    LwU64         pPTE = 0;
    readFn_t      pteReadFunc = NULL;
    LwU32         bigIndexLimit = 0;
    LwU32         smallIndexLimit = 0;
    LwU32         vmPageSize;
    LwU32         pageSizeKind;
    LwU32         bigValid = 0;
    LwU32         smallValid = 0;
    char          pteTypeLen32[32];
    LwU32         sizeRatio = 1;
    LwU32         bigEnd;

    if (pVMemSpace == NULL)
    {
        return;
    }

    if (pVmem[indexGpu].vmemPdeGetByIndex(pVMemSpace, pdeIndex, (GMMU_ENTRY_VALUE*)&pde) != LW_OK)
    {
        return;
    }

    if (begin > end)
    {
        dprintf("lw: begin (%d) > end  (%d)\n", begin, end);
        return;
    }

    vmPageSize = GPU_REG_RD_DRF(_PFB_PRI,_MMU_CTRL, _VM_PG_SIZE);
    pageSizeKind = (vmPageSize == LW_PFB_PRI_MMU_CTRL_VM_PG_SIZE_128KB) ?
                         MMU_PDE_PAGE_SIZE_IS_128KB : MMU_PDE_PAGE_SIZE_IS_64KB;


    //use LW_MMU_PDE_SIZE to determine big/small Page Table sizes (FULL,HALF,QUARTER,EIGHTH)
    sizeRatio = 1 << (DRF_VAL(_MMU, _PDE, _SIZE, pde.w0));

    bigIndexLimit = 1024 / sizeRatio;
    smallIndexLimit = ((pageSizeKind == MMU_PDE_PAGE_SIZE_IS_128KB) ? 16*1024 : 32*1024)/sizeRatio;

    //check begin and end with index limit for small pages which is always greater than index limit for big pages
    if (begin > smallIndexLimit)
    {
        dprintf("lw: begin (%d) should atleast be less than %d \n", begin, smallIndexLimit);
        return;
    }

    //check with higher index val
    if (end > smallIndexLimit)
    {
        dprintf("lw: end (%d) should atleast be less than %d \n", end, smallIndexLimit);
        return;
    }


    //check the appropriate apperture for the given PDE
    //display the big PT if its not invalid
    if (SF_VAL(_MMU, _PDE_APERTURE_BIG, pde.w0) != LW_MMU_PDE_APERTURE_BIG_ILWALID)
    {
        //check whether begin and end lie within index limit for big pages
        if (begin < bigIndexLimit)
        {
            //if end falls beyond index limit; show till index limit
            if (end > bigIndexLimit)
            {
                bigEnd = bigIndexLimit;
                dprintf("\nlw: Truncating output to largest legal big page table index : %d \n", bigIndexLimit);
            }
            else
            {
                bigEnd = end;
            }

            bigValid = 1;
            strcpy(pteTypeLen32, "big");
            pPTE = getPTBaseAddr_GK104(&pde, TRUE, &pteReadFunc);
            dprintf("\n\n              Big PTEs for given indices     \n\n");
            printPageTable_GK104(pVMemSpace, pPTE, pteTypeLen32, begin, bigEnd, pteReadFunc);
        }
        else
        {
            dprintf("lw: begin (%d) greater than legal big page table index %d \n", begin, bigIndexLimit);
        }
    }

    //display the small PT if its not invalid
    if (SF_VAL(_MMU, _PDE_APERTURE_SMALL, pde.w1) != LW_MMU_PDE_APERTURE_SMALL_ILWALID)
    {
        smallValid = 1;
        strcpy(pteTypeLen32, "small");
        pPTE = getPTBaseAddr_GK104(&pde, FALSE, &pteReadFunc);
        dprintf("\n\n                  Small PTEs for given indices   \n\n");
        printPageTable_GK104(pVMemSpace, pPTE, pteTypeLen32, begin, end, pteReadFunc);
    }

    //both appertures are invalid ; hence PDE Fault for the given PDE
    if (bigValid == 0 && smallValid == 0)
    {
        dprintf("lw: PDE FAULT for given pde index: %d\n", pdeIndex);
        return;
    }

}

/*!
 *  Get a PTE for the given PTE index
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  va          Virtual address.
 *  @param[in]  pPde        PDE which PTE is in.
 *  @param[out] pPte        PTE for @a va.
 *
 *  @return LW_OK on success, LW_ERR_GENERIC otherwise.
 */
LW_STATUS
vmemPteGetByIndex_GK104(VMemSpace *pVMemSpace, LwU32 index, PdeEntry *pPde, PteEntry *pPte)
{
    PteEntry    pteEntryBig = {0}, pteEntrySmall = {0};
    BOOL        validBig = FALSE, validSmall = FALSE;
    LwU32       bigPageSize = 0;
     LwU64      bigOffset = 0, smallOffset = 0;

    bigPageSize = pVMemSpace->bigPageSize;
    bigOffset   = _getPTEByIndex_GK104(pVMemSpace, index, bigPageSize, bigPageSize,         pPde, &pteEntryBig,   &validBig,   TRUE);
    smallOffset = _getPTEByIndex_GK104(pVMemSpace, index, bigPageSize, VMEM_SMALL_PAGESIZE, pPde, &pteEntrySmall, &validSmall, TRUE);

    if ((!bigOffset) && (!smallOffset))
    {
        return LW_ERR_GENERIC;
    }

    if (validBig)
    {
        *pPte = pteEntryBig;
        return LW_OK;
    }
    else if (validSmall)
    {
        *pPte = pteEntrySmall;
        return LW_OK;
    }
    else return LW_ERR_GENERIC;
}

/*!
 *  Get a PTE for the given va
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  va          Virtual address.
 *  @param[in]  pPde        PDE which PTE is in.
 *  @param[out] pPte        PTE for @a va.
 *
 *  @return LW_OK on success, LW_ERR_GENERIC otherwise.
 */
LW_STATUS
vmemPteGetByVa_GK104(VMemSpace *pVMemSpace, LwU64 va, PdeEntry *pPdeEntry, PteEntry *pPteEntry)
{
    PteEntry    pteEntryBig = {0}, pteEntrySmall = {0};
    LwU32       bigPageSize = 0;
    BOOL        validBig = FALSE, validSmall = FALSE;
    LwU64       bigOffset = 0, smallOffset = 0;

    bigPageSize = pVMemSpace->bigPageSize;
    bigOffset   = _getPTE_GK104(pVMemSpace, va, bigPageSize, bigPageSize,         pPdeEntry, &pteEntryBig,   &validBig,   TRUE);
    smallOffset = _getPTE_GK104(pVMemSpace, va, bigPageSize, VMEM_SMALL_PAGESIZE, pPdeEntry, &pteEntrySmall, &validSmall, TRUE);

     if ((!bigOffset) && (!smallOffset))
    {
        return LW_ERR_GENERIC;
    }

    if (validBig)
    {
        *pPteEntry = pteEntryBig;
        return LW_OK;
    }
    else if (validSmall)
    {
        *pPteEntry = pteEntrySmall;
        return LW_OK;
    }
    else return LW_ERR_GENERIC;
}

/*!
 *  Populate the @a VMemSpace.(helper function)
 *
 *  @param[in]  instMemAddr      Base Address for the instance mem block.
 *  @param[in]  instMemReadFn    Function to be used to read instance block data.
 *  @param[out] pVMemSpace       Virtual memory space structure to populate.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
fillVMemSpace_GK104(VMemSpace *pVMemSpace, LwU64 instMemAddr, readFn_t instMemReadFn,
                    writeFn_t instMemWriteFn, MEM_TYPE memType)
{
    LwU32       buf = 0;
    LwU64       vaLimit = 0;
    InstBlock*  pInstBlock;

    if (pVMemSpace == NULL)
    {
        return LW_ERR_GENERIC;
    }

    if (!instMemReadFn)
    {
        return LW_ERR_GENERIC;
    }

    //for brevity
    pInstBlock = &(pVMemSpace->instBlock);

    pInstBlock->instBlockAddr = instMemAddr;
    pInstBlock->readFn = instMemReadFn;
    pInstBlock->writeFn = instMemWriteFn;
    pInstBlock->memType = memType;

    //1. Now fetch PDB from instBlock
    pVMemSpace->PdeBase = pMmu[indexGpu].mmuGetPDETableStartAddress(instMemAddr, instMemReadFn);

    // check where the PD lives
    if (LW_OK != instMemReadFn(instMemAddr + SF_OFFSET(LW_RAMIN_PAGE_DIR_BASE_TARGET), &buf, 4))
    {
        return LW_ERR_GENERIC;
    }
    if (SF_VAL(_RAMIN, _PAGE_DIR_BASE_TARGET, buf) == LW_RAMIN_PAGE_DIR_BASE_TARGET_VID_MEM)
    {
        pVMemSpace->readFn  = pFb[indexGpu].fbRead;
        pVMemSpace->writeFn = pFb[indexGpu].fbWrite;
    }
    else
    {
        pVMemSpace->readFn  = readSystem;
        pVMemSpace->writeFn = writeSystem;
    }

    //2. VAL fetch
    vaLimit = pVmem[indexGpu].vmemGetLargestVirtAddr(pVMemSpace);

    pVMemSpace->bigPageSize = pVmem[indexGpu].vmemGetBigPageSize(pVMemSpace);
    pVMemSpace->pdeCount = (LwU32)(PDE_INDEX(pVMemSpace->bigPageSize, vaLimit)) + 1;

    // Indicate this is a FERMI VA Space and use BAR1 mapping
    pVMemSpace->class = FERMI_VASPACE_A;

    return LW_OK;

}



/*!
 *  Returns the largest virtual address for given vmem space.
 *
 *  @param[in]  pVMemSpace  The vmem space for which VAL is to be found out
 *
 *  @return   largest virtual address (VAL) from inst block.
 */
LwU64
vmemGetLargestVirtAddr_GK104(VMemSpace *pVMemSpace)
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



/*!
 *  Fills in the buffer with length of physical data for the va in the
 *  given VMemSpace.
 *
 *  @param[in]  pVMemSpace  The vmem space corresponding to the va
 *  @param[in]  virtAddr    Virtual Address
 *  @param[in]  length      The length of the data that has to be copied to the buffer.
 *  @param[out] The populated buffer
 *
 *  @return   LW_OK , LW_ERR_GENERIC.
 */
LW_STATUS
vmemreadGpuVirtualAddr_GK104
(
    VMemSpace *pVMemSpace,
    LwU64 virtAddr,
    void* buffer,
    LwU32 length
)
{
    GMMU_APERTURE aperture;
    LwU32          status;
    LwU64         pa;

    if (pVMemSpace == NULL)
    {
        return LW_ERR_GENERIC;
    }

    // Align by 4 bytes in order to avoid SIGSEGV
    virtAddr = virtAddr & ((LwU64)~3);

    // colwert the virtual address to a physical one
    status = vmemVToP(pVMemSpace, virtAddr, &pa, &aperture, LW_FALSE);
    if (status == LW_ERR_GENERIC)
    {
        dprintf("lwwatch: %s: Failed to get GPU virtual to physical address mapping\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    return gpuPhysMemCpy_GK104(pa, buffer, aperture, length);
}
//-----------------------------------------------------
// readGpuVirtualAddr_GK104
//
// Given a virtual address and a channel ID finds the
// corresponding physical address and the aperture and
// reads the amount of data needed to the buffer.
// This function is limited and doesn't support length != 4.
//
// Now exists because of historic dependency from other modules.
//-----------------------------------------------------
LW_STATUS
readGpuVirtualAddr_GK104
(
    LwU32 chId,
    LwU64 virtAddr,
    void* buffer,
    LwU32 length
)
{
    VMemSpace   vMemSpace;
    VMEM_INPUT_TYPE Id;
    memset(&Id, 0, sizeof(Id));
    Id.ch.chId = chId;

    if (vmemGet(&vMemSpace, VMEM_TYPE_CHANNEL, &Id) != LW_OK)
    {
        dprintf("lw: %s: Could not fetch vmemspace for chId 0x%x\n",
                __FUNCTION__, chId);
        return LW_ERR_GENERIC;
    }

    return vmemreadGpuVirtualAddr_GK104(&vMemSpace, virtAddr, buffer, length);
}

/*!
 *  Finds out the PT base address and readfunction to be used.
 *
 *
 *  @param[in]  pPde        The PDE that points to the Page Table
 *  @param[in]  isBig       BOOL specifying whether the PT to be fetched is big or small.(BIG == TRUE)
 *
 *  @param[out] pteReadFn   Function to be used to read the PTE
 *
 *  @return   Page Table Base Address (0 if invalid)
 */
LwU64
getPTBaseAddr_GK104(PdeEntry* pPde, BOOL isBig, readFn_t *pteReadFn)
{
    LwU64 PTBase=0;

    if ((pteReadFn == NULL) || (pPde == NULL))
    {
        return 0LL;
    }

    if (isBig)
    {
        if (SF_VAL(_MMU, _PDE_APERTURE_BIG, pPde->w0) == LW_MMU_PDE_APERTURE_BIG_ILWALID)
        {
            return 0LL;
        }
        // page is located in video memory
        if (SF_VAL(_MMU, _PDE_APERTURE_BIG, pPde->w0) == LW_MMU_PDE_APERTURE_BIG_VIDEO_MEMORY)
        {
            *pteReadFn = pFb[indexGpu].fbRead;
            // Read the value showing where the page table lives in video memory
            PTBase = (LwU64)SF_VAL(_MMU, _PDE_ADDRESS_BIG_VID, pPde->w0) << LW_MMU_PDE_ADDRESS_SHIFT;
        }
        else   // page is located in system memory
        {
            *pteReadFn = readSystem;
            // Read the value showing where the page table lives in system memory
            PTBase = (LwU64)SF_VAL(_MMU, _PDE_ADDRESS_BIG_SYS, pPde->w0) << LW_MMU_PDE_ADDRESS_SHIFT;
        }

        return PTBase;
    }
    else
    {

        if (SF_VAL(_MMU, _PDE_APERTURE_SMALL, pPde->w1) == LW_MMU_PDE_APERTURE_SMALL_ILWALID)
        {
            return 0LL;
        }

        if (SF_VAL(_MMU, _PDE_APERTURE_SMALL, pPde->w1) == LW_MMU_PDE_APERTURE_SMALL_VIDEO_MEMORY)
        {
            *pteReadFn = pFb[indexGpu].fbRead;
            // Read the value showing where the page table lives in video memory
            PTBase = (LwU64)SF_VAL(_MMU, _PDE_ADDRESS_SMALL_VID, pPde->w1) << LW_MMU_PDE_ADDRESS_SHIFT;
        }
        else
        {
            *pteReadFn = readSystem;
            // Read the value showing where the page table lives in system memory
            PTBase = (LwU64)SF_VAL(_MMU, _PDE_ADDRESS_SMALL_SYS, pPde->w1) << LW_MMU_PDE_ADDRESS_SHIFT;
        }

        return PTBase;
    }
}

//-----------------------------------------------------
// gpuPhysMemCpy_GK104
//
// Copies data of size 'length' from Physical address
// 'pa' of aperture type 'aperture' to destination 'buffer'.
//-----------------------------------------------------
LW_STATUS
gpuPhysMemCpy_GK104
(
    LwU64         pa,
    void         *buffer,
    GMMU_APERTURE aperture,
    LwU32         length
)
{
    if (buffer == NULL)
    {
        return LW_ERR_GENERIC;
    }

    switch (aperture)
    {
    case GMMU_APERTURE_VIDEO:
        return pFb[indexGpu].fbRead(pa, buffer, length);
    case GMMU_APERTURE_PEER:
        // not supported
        dprintf("lwwatch: %s: LW_MMU_PTE_APERTURE_PEER_MEMORY is not supported\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    case GMMU_APERTURE_SYS_COH: /* Fallthrough */
    case GMMU_APERTURE_SYS_NONCOH:
        return readSystem(pa, buffer, length);
    default:
        dprintf("lwwatch: %s: Unknown aperture type\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }
}

//-----------------------------------------------------
// vmemGetInstanceMemoryAddrForBAR1_GK104
//
// Returns the instance memory base address for BAR1.
//-----------------------------------------------------
LwU64
vmemGetInstanceMemoryAddrForBAR1_GK104(readFn_t* readFn, writeFn_t* writeFn, MEM_TYPE* pMemType)
{
    LwU32       reg;
    LwU64       instMemAddr;

    // read the instance memory description for BAR1
    reg = GPU_REG_RD32(LW_PBUS_BAR1_BLOCK);

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
// vmemGetInstanceMemoryAddrForBAR2_GK104
//
// Returns the instance memory base address for BAR2.
//-----------------------------------------------------
LwU64
vmemGetInstanceMemoryAddrForBAR2_GK104(readFn_t* readFn, writeFn_t* writeFn, MEM_TYPE* pMemType)
{
    LwU32       reg;
    LwU64       instMemAddr;

    // read the instance memory description for BAR1
    reg = GPU_REG_RD32(LW_PBUS_BAR2_BLOCK);

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

//-----------------------------------------------------
// vmemGetInstanceMemoryAddrForIfb_GK104
//
// Returns the instance memory base address for IFB.
//-----------------------------------------------------
LwU64
vmemGetInstanceMemoryAddrForIfb_GK104(readFn_t* readFn, writeFn_t* writeFn, MEM_TYPE* pMemType)
{
    LwU32       reg;
    LwU64       instMemAddr;

    // read the instance memory description for BAR1
    reg = GPU_REG_RD32(LW_PBUS_IFB_BLOCK);
    instMemAddr = (LwU64)SF_VAL(_PBUS, _IFB_BLOCK_PTR, reg) << 12;

    if (SF_VAL(_PBUS, _IFB_BLOCK_TARGET, reg) == LW_PBUS_IFB_BLOCK_TARGET_VID_MEM)
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
// getInstanceMemoryAddrForPMU_GK104
//
// Returns the instance memory base address for PMU.
//-----------------------------------------------------
LwU64
getInstanceMemoryAddrForPMU_GK104(readFn_t* readFn, writeFn_t* writeFn, MEM_TYPE* pMemType)
{

    LwU32       memType;
    LwU32       instBlockReg;
    LwU64       instBlockPtr;
    LwU32       bValid;

    *readFn = (readFn_t)0;

    instBlockReg    = pPmu[indexGpu].pmuReadPmuNewInstblk();
    memType         = DRF_VAL(_PPWR_PMU, _NEW_INSTBLK, _TARGET, instBlockReg);
    instBlockPtr    = ((LwU64)DRF_VAL(_PPWR_PMU, _NEW_INSTBLK, _PTR, instBlockReg) << 12);
    bValid          = DRF_VAL(_PPWR_PMU, _NEW_INSTBLK, _VALID, instBlockReg);

    if (!bValid)
    {
        dprintf("lw: %s: PMU instance block is not valid\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    switch (memType)
    {
    case LW_PPWR_PMU_NEW_INSTBLK_TARGET_FB:
        if (readFn)
            *readFn = pFb[indexGpu].fbRead;
        if (writeFn)
            *writeFn = pFb[indexGpu].fbWrite;
        if (pMemType)
            *pMemType = FRAMEBUFFER;
        break;
    case LW_PPWR_PMU_NEW_INSTBLK_TARGET_SYS_COH:
    case LW_PPWR_PMU_NEW_INSTBLK_TARGET_SYS_NONCOH:
        if (readFn)
            *readFn = readSystem;
        if (writeFn)
            *writeFn = writeSystem;
        if (pMemType)
            *pMemType = SYSTEM_PHYS;
        break;
    default:
        dprintf("lw: %s: Unknown memory type for PMU instance block\n",
                __FUNCTION__);
       break;
    }

    return instBlockPtr;
}

//-----------------------------------------------------
// vmemGetInstanceMemoryAddrForChId_GK104
//
// Returns the instance memory base address for a
// channel ID. It works for both PIO and DMA mode.
//-----------------------------------------------------
LwU64
vmemGetInstanceMemoryAddrForChId_GK104(ChannelId *pChannelId, readFn_t* readFn, writeFn_t* writeFn, MEM_TYPE* pMemType)
{
    LW_STATUS   status = LW_OK;
    LwU32       pioTarget = 0;
    LwU32       reg;
    LwU64       instMemAddr = 0;
    LwU64       instMemTarget = 0;
    LwU64       targetMemType;
    ChannelInst channelInst;

    // Read the pio target register
    pioTarget = GPU_REG_RD32(LW_PFIFO_PIO_TARGET);

    // If we are in PIO mode
    if (SF_VAL(_PFIFO, _PIO_TARGET_ENABLE, pioTarget) == LW_PFIFO_PIO_TARGET_ENABLE_TRUE)
    {
        reg = GPU_REG_RD32(LW_PFIFO_PIO_CTXSW);
        instMemAddr = (LwU64)SF_VAL(_PFIFO, _PIO_CTXSW_POINTER, reg) << 12;
        instMemTarget = SF_VAL(_PFIFO, _PIO_CTXSW_TARGET, reg);
    }
    // Otherwise read channel instance memory for channel ID
    else
    {
        pFifo[indexGpu].fifoGetChannelInstForChid(pChannelId, &channelInst);
        instMemAddr = channelInst.instPtr;
        instMemTarget = channelInst.target;
    }

    status = pVmem[indexGpu].vmemGetMemTypeFromTarget(instMemTarget, &targetMemType);

    if ( status != LW_OK )
    {
        dprintf("lwwatch: %s: Failed to determine memory type from instance memory target\n", __FUNCTION__);
    }

    if (pMemType)
        *pMemType = (MEM_TYPE)targetMemType;

    switch (targetMemType)
    {
    case FRAMEBUFFER:
        if (readFn)
            *readFn = pFb[indexGpu].fbRead;
        if (writeFn)
            *writeFn = pFb[indexGpu].fbWrite;
        break;

    case SYSTEM_PHYS:
        if (readFn)
            *readFn = readSystem;
        if (writeFn)
            *writeFn = writeSystem;
        break;
    }

    return instMemAddr;
}



//-----------------------------------------------------
// _getInstanceMemoryAddr_GK104
//
// Returns the instance memory base address for a
// channel ID. chId can be BARs or IFB too.
//-----------------------------------------------------
static LwU64
_getInstanceMemoryAddr_GK104(VMemTypes vMemType, ChannelId *pChannelId, readFn_t* instMemReadFn,
                            writeFn_t* instMemWriteFn, MEM_TYPE* pMemType)
{
    LwU64       instMemAddr = 0;

    switch (vMemType)
    {
        case VMEM_TYPE_CHANNEL:
             instMemAddr = pVmem[indexGpu].vmemGetInstanceMemoryAddrForChId(pChannelId, instMemReadFn,
                                                                            instMemWriteFn, pMemType);
             break;

        case VMEM_TYPE_BAR1:
            instMemAddr = pVmem[indexGpu].vmemGetInstanceMemoryAddrForBAR1(instMemReadFn, instMemWriteFn, pMemType);
            break;

        case VMEM_TYPE_BAR2:
            instMemAddr = pVmem[indexGpu].vmemGetInstanceMemoryAddrForBAR2(instMemReadFn, instMemWriteFn, pMemType);
            break;

        case VMEM_TYPE_IFB:
            instMemAddr = pVmem[indexGpu].vmemGetInstanceMemoryAddrForIfb(instMemReadFn, instMemWriteFn, pMemType);
            break;

        case VMEM_TYPE_PMU:
            instMemAddr = getInstanceMemoryAddrForPMU_GK104(instMemReadFn, instMemWriteFn, pMemType);
            break;

        default:
            dprintf("lw: invalid vmem type\n");
            break;
    }

    return instMemAddr;
}


//-----------------------------------------------------
// printPageTable_GK104
//
// Prints out Page Table entries from begin to end using
// the pteReadFunc to fetch PTEs.
// -----------------------------------------------------
LW_STATUS
printPageTable_GK104(VMemSpace *pVMemSpace, LwU64 pPTE, char* printPteType, LwU32 begin, LwU32 end, readFn_t pteReadFunc)
{
    LwU32       i;
    PteEntry    PTE;

    if (printPteType == NULL)
    {
        printPteType = " ";
    }

    for (i = begin, pPTE += begin * LW_MMU_PTE__SIZE; i <= end; i++, pPTE += LW_MMU_PTE__SIZE)
    {
        if (osCheckControlC())
            break;

        //read the lower 32 bits of the PTE starting from (0*32 + 0)
        pteReadFunc(pPTE + SF_OFFSET(LW_MMU_PTE_VALID), &PTE.w0, 4);

        //read the upper 32 bits of the PTE starting from (1*32 + 0)
        pteReadFunc(pPTE + SF_OFFSET(LW_MMU_PTE_VOL), &PTE.w1, 4);


        dprintf("\n----------------- PTE %s ------------------\n", printPteType);
        vmemDumpPte(pVMemSpace, (GMMU_ENTRY_VALUE*)&PTE);
    }

    return LW_OK;
}

enum
{
    PTE_ERROR_COMPTAG,
    PTE_ERROR_OFFSET,
    PTE_ERROR_NUM_TYPES
} PTE_ERROR;

static LW_STATUS
_checkPteBase_GK104(VMemSpace* pVmemSpace, GMMU_ENTRY_VALUE *pPde, LwU64 fbLimit, LwU32* pPteCount)
{
    LW_STATUS     status = LW_OK;
    LwU64         ptBase = 0;
    LwU64         vaLimit = pVmem[indexGpu].vmemGetLargestVirtAddr(pVmemSpace);
    LwU32         bigPageSize = pVmemSpace->bigPageSize;
    LwU32         countIndex = PAGETABLE_INDEX_BIG;
    char         *sizeStr = "Big";
    VMemFmtPde    fmtPdeMulti;
    GMMU_FMT_PDE *pFmtPde;
    GMMU_APERTURE aperture;

    CHECK(pVmem[indexGpu].vmemGetPDEFmt(pVmemSpace, &fmtPdeMulti, 0));

    //check for BIG first
    pFmtPde = &fmtPdeMulti.fmts.multi.gmmu.subLevels[PDE_MULTI_BIG_INDEX];
    aperture = gmmuFieldGetAperture(&pFmtPde->fldAperture, ((GMMU_ENTRY_VALUE*)pPde)->v8);
    if (aperture == GMMU_APERTURE_ILWALID)
    {
        //check for SMALL
        countIndex = PAGETABLE_INDEX_SMALL;
        sizeStr    = "Small";
        pFmtPde    = &fmtPdeMulti.fmts.multi.gmmu.subLevels[PDE_MULTI_SMALL_INDEX];
        aperture   = gmmuFieldGetAperture(&pFmtPde->fldAperture, ((GMMU_ENTRY_VALUE*)pPde)->v8);
    }

    if (aperture == GMMU_APERTURE_ILWALID)
    {
        return status;
    }

    ptBase = vmemGetPhysAddrFromPDE(pVmemSpace, (MMU_FMT_PDE*)pFmtPde, pPde);
    pPteCount[countIndex] = (LwU32)PTE_INDEX(bigPageSize, vaLimit, bigPageSize);

    switch (aperture)
    {
        case GMMU_APERTURE_VIDEO:
            if (ptBase >= fbLimit)
            {
                dprintf("lw:  %s PT base exceeds fb limit!\n", sizeStr);
                status = LW_ERR_GENERIC;
            }
            break;
        case GMMU_APERTURE_SYS_COH:
        case GMMU_APERTURE_SYS_NONCOH:
            if (ptBase >= ((LwU64)1) << 32)
            {
                dprintf("lw:  %s PT base exceeds sysmem limit!\n", sizeStr);
                status = LW_ERR_GENERIC;
            }
            break;
        default:
            dprintf("lw:  Unknown aperture\n");
            status = LW_ERR_GENERIC;
    }

    return status;
}


/*!
 *  Checks PTEs for the following:
 *  - there are no duplicate references to comptaglines,
 *  - any page aperture of video memory, should have a PA that is a valid fb offset.
 *
 *  @param[in]  pPte            pte entry
 *  @param[in]  fbLimit         framebuffer limit
 *  @param[in]  pteId           pte index
 *  @param[out] pTagramAddrMap  array address to account for comptagline references
 *  @param[out] pteErrorCount   Number of pte anomalies encountered
 *
 *  @return   LW_OK , LW_ERR_GENERIC.
 */
static LW_STATUS
_checkPTE_GK104(VMemSpace *pVMemSpace, pte_entry_t* pPte, LwU64 fbLimit, LwU32 pteId, LwU32* pTagramAddrMap, LwU32* pteErrCount)
{
    LwU32           compTags;
    LwU32           memType;
    LwU64           pa;
    LwU32           errCount = 0;
    LwU32            status;

    //1. test if within FB range
    memType = DRF_VAL(_MMU, _PTE, _APERTURE, pPte->w1);
    if (memType == LW_MMU_PTE_APERTURE_VIDEO_MEMORY)
    {
        pa = vmemGetPhysAddrFromPTE(pVMemSpace, (GMMU_ENTRY_VALUE*)pPte);
        if (pa >= fbLimit)
        {
            //report the first error
            if (pteErrCount[PTE_ERROR_OFFSET] == 0)
            {
                dprintf("lw: ERROR - pa outside limit for PTE 0x%x \n", pteId);
            }
            pteErrCount[PTE_ERROR_OFFSET]++;
            errCount++;
        }
    }

    //2. test for duplicate COMPTAG
    compTags = DRF_VAL(_MMU, _PTE, _COMPTAGLINE, pPte->w1);
    if (compTags != LW_MMU_PTE_COMPTAGS_NONE)
    {
        if (pTagramAddrMap[compTags] == 0)
        {
            pTagramAddrMap[compTags] = 1;
        }
        else
        {
            //report the first error
            if (pteErrCount[PTE_ERROR_COMPTAG] == 0)
            {
                dprintf("lw: ERROR - found duplicate COMPTAGLINE for PTE 0x%x \n", pteId);
            }
            pteErrCount[PTE_ERROR_COMPTAG]++;
            errCount++;
        }
    }

    if (errCount > 0)
    {
        status = LW_ERR_GENERIC;
    }

    return status;
}


/*!
 *  Analyse a PDE(associated PTEs) for anomalies.
 *  Uses _checkPTE_GK104 for the purpose.
 *
 *  @param[in]  pVmemSpace      vmem space to be analyzed for the given pde entry
 *  @param[in]  pPde            pde entry
 *  @param[in]  pdeIndex        pde index
 *
 *  @return   LW_OK , LW_ERR_GENERIC.
 */
static LW_STATUS
_checkPDE_GK104(VMemSpace* pVMemSpace, GMMU_ENTRY_VALUE *pPde, LwU32 pdeIndex)
{
    LwU32        retValue = LW_OK;
    LwU64       fbLimit = 0;
    LwU32       numPtes[PAGETABLE_COUNT] = {0};
    LwU32       pteErrorCount[PTE_ERROR_NUM_TYPES] = {0};
    LwU32       numCompTags;
    LwU32       *pTagramAddrMap = NULL;
    LwU32       i;
    pte_entry_t pte;
    BOOL        bValid = FALSE;
    LwU32       bigPageSize = pVMemSpace->bigPageSize;

    fbLimit = pFb[indexGpu].fbGetMemSizeMb();
    fbLimit <<= 20;

    if (LW_ERR_GENERIC == _checkPteBase_GK104(pVMemSpace, pPde, fbLimit, numPtes))
    {
        dprintf ("lw: bad PDE (index %d) encountered\n", pdeIndex);
        retValue = LW_ERR_GENERIC;
        goto END;
    }

    numCompTags = 1 << (DEVICE_EXTENT(LW_MMU_PTE_COMPTAGLINE) -
                        DEVICE_BASE(LW_MMU_PTE_COMPTAGLINE)+1 );
    pTagramAddrMap =(LwU32*) malloc(numCompTags * sizeof(LwU32));
    memset((void*)pTagramAddrMap, 0, numCompTags * sizeof(LwU32));

    vmemDumpPde(pVMemSpace, pPde, 0);

    //check big page ptes
    for (i=0; i<numPtes[PAGETABLE_INDEX_BIG]; i++)
    {
        if (osCheckControlC())
            break;

        _getPTEByIndex_GK104(pVMemSpace, i, bigPageSize, bigPageSize, (PdeEntry*)pPde, &pte, &bValid, FALSE);
        //leave invalid ptes
        if (!bValid)
        {
            continue;
        }
        retValue = _checkPTE_GK104(pVMemSpace, &pte, fbLimit, i, pTagramAddrMap, pteErrorCount);
    }

    //check small page ptes
    for (i=0; i<numPtes[PAGETABLE_INDEX_SMALL]; i++)
    {
        if (osCheckControlC())
            break;

        _getPTEByIndex_GK104(pVMemSpace, i, bigPageSize, VMEM_SMALL_PAGESIZE, (PdeEntry*)pPde, &pte, &bValid, FALSE);
        //leave invalid ptes
        if (!bValid)
        {
            continue;
        }
        retValue = _checkPTE_GK104(pVMemSpace, &pte, fbLimit, i, pTagramAddrMap, pteErrorCount);
    }


    if ( pteErrorCount[PTE_ERROR_OFFSET] > 0 )
    {
        dprintf("lw: PDE #%d: Out of FB limit errors: %d\n", pdeIndex, pteErrorCount[PTE_ERROR_OFFSET]);
        retValue = LW_ERR_GENERIC;
    }

    if ( pteErrorCount[PTE_ERROR_COMPTAG] > 0 )
    {
        dprintf("lw: PDE #%d: COMPTAG errors: %d\n", pdeIndex, pteErrorCount[PTE_ERROR_COMPTAG]);
        retValue = LW_ERR_GENERIC;
    }

    END:
    if (pTagramAddrMap)
    {
        free(pTagramAddrMap);
    }
    return retValue;
}


/*!
 *  Parse through the page table entries and check for anomalies
 *  Used pdeCheck to parse and check for inconsistencies

 *  @param[in]  pVMemSpace  The vmem space corresponding to the va
 *  @param[in]  pdeId       Pde index to be checked
 *
 *  @return   LW_OK , LW_ERR_GENERIC.
 */
LW_STATUS
vmemPdeCheck_GK104(VMemSpace *pVMemSpace, LwU32 pdeId)
{
    LW_STATUS        status = LW_OK;
    GMMU_ENTRY_VALUE pde;
    GMMU_APERTURE    aperture;

    if (pdeId >= pVMemSpace->pdeCount)
    {
        dprintf("\n\t lw: PDE #%d exceeds the VAL .. aborting\n", pdeId);
        return LW_ERR_GENERIC;
    }

    dprintf("\n\t lw: Testing channel PDE #%d\n", pdeId);
    status = vmemPdeGetByIndex(pVMemSpace, pdeId, &pde);
    if (status != LW_OK)
    {
        dprintf("lw: Read of PDE #%d failed.\n", pdeId);
        return LW_ERR_GENERIC;
    }

    CHECK(vmemGetPdeAperture(pVMemSpace, 0, &pde, &aperture, NULL));

    if (aperture == GMMU_APERTURE_ILWALID)
    {
        dprintf("lw: Error: LW_MMU_PDE_APERTURE _ILWALID\n");
        return LW_ERR_GENERIC;
    }

    status = _checkPDE_GK104(pVMemSpace, &pde, pdeId);
    if (status != LW_OK)
    {
        dprintf("lw: PDE #%d failed.\n", pdeId);
        return LW_ERR_GENERIC;
    }

    return status;
}

static void
_gmmuInitMmuFmt_GK104(MMU_FMT_LEVEL *pLevels, LwU32 bigPageShift)
{
    pLevels[0].virtAddrBitLo = (LwU8)bigPageShift + 10;
    pLevels[0].virtAddrBitHi = 39;
    pLevels[0].entrySize     = LW_MMU_PDE__SIZE;
    pLevels[0].bPageTable    = LW_FALSE;
    pLevels[0].numSubLevels  = 2;
    pLevels[0].subLevels     = &pLevels[1];

    pLevels[1].virtAddrBitLo = (LwU8)bigPageShift;
    pLevels[1].virtAddrBitHi = pLevels[0].virtAddrBitLo - 1;
    pLevels[1].entrySize     = LW_MMU_PTE__SIZE;
    pLevels[1].bPageTable    = LW_TRUE;
    pLevels[1].numSubLevels  = 0;
    pLevels[1].subLevels     = NULL;

    pLevels[2].virtAddrBitLo = 12;
    pLevels[2].virtAddrBitHi = pLevels[0].virtAddrBitLo - 1;
    pLevels[2].entrySize     = LW_MMU_PTE__SIZE;
    pLevels[2].bPageTable    = LW_TRUE;
    pLevels[2].numSubLevels  = 0;
    pLevels[2].subLevels     = NULL;
}

static void
_gmmuFmtInitPdeMulti_GK104(GMMU_FMT_PDE_MULTI *pPdeMulti, LW_FIELD_ENUM_ENTRY *pdeApertures)
{
    GMMU_FMT_PDE *pPdeBig   = &pPdeMulti->subLevels[PDE_MULTI_BIG_INDEX];
    GMMU_FMT_PDE *pPdeSmall = &pPdeMulti->subLevels[PDE_MULTI_SMALL_INDEX];

    pPdeBig->version   = GMMU_FMT_VERSION_1;
    pPdeSmall->version = GMMU_FMT_VERSION_1;

    // Common PDE fields.
    INIT_FIELD_DESC32(&pPdeMulti->fldSizeRecipExp, LW_MMU_PDE_SIZE);

    // Dual PDE - big part.
    INIT_FIELD_APERTURE(&pPdeBig->fldAperture, LW_MMU_PDE_APERTURE_BIG, pdeApertures);
    INIT_FIELD_ADDRESS(&pPdeBig->fldAddrVidmem, LW_MMU_PDE_ADDRESS_BIG_VID,
                       LW_MMU_PDE_ADDRESS_SHIFT);
    INIT_FIELD_ADDRESS(&pPdeBig->fldAddrSysmem, LW_MMU_PDE_ADDRESS_BIG_SYS,
                       LW_MMU_PDE_ADDRESS_SHIFT);
    INIT_FIELD_BOOL(&pPdeBig->fldVolatile, LW_MMU_PDE_VOL_BIG);

    // Dual PDE - small part.
    INIT_FIELD_APERTURE(&pPdeSmall->fldAperture, LW_MMU_PDE_APERTURE_SMALL, pdeApertures);
    INIT_FIELD_ADDRESS(&pPdeSmall->fldAddrVidmem, LW_MMU_PDE_ADDRESS_SMALL_SYS,
                       LW_MMU_PDE_ADDRESS_SHIFT);
    INIT_FIELD_ADDRESS(&pPdeSmall->fldAddrSysmem, LW_MMU_PDE_ADDRESS_SMALL_VID,
                       LW_MMU_PDE_ADDRESS_SHIFT);
    INIT_FIELD_BOOL(&pPdeSmall->fldVolatile, LW_MMU_PDE_VOL_SMALL);
}

void
vmemGmmuFmtInitPte_GK104(GMMU_FMT_PTE *pPte, LW_FIELD_ENUM_ENTRY *pteApertures)
{
    pPte->version = GMMU_FMT_VERSION_1;

    INIT_FIELD_BOOL(&pPte->fldValid, LW_MMU_PTE_VALID);
    INIT_FIELD_APERTURE(&pPte->fldAperture, LW_MMU_PTE_APERTURE, pteApertures);
    INIT_FIELD_ADDRESS(&pPte->fldAddrSysmem, LW_MMU_PTE_ADDRESS_SYS, LW_MMU_PTE_ADDRESS_SHIFT);
    INIT_FIELD_ADDRESS(&pPte->fldAddrVidmem, LW_MMU_PTE_ADDRESS_VID, LW_MMU_PTE_ADDRESS_SHIFT);
    INIT_FIELD_ADDRESS(&pPte->fldAddrPeer, LW_MMU_PTE_ADDRESS_VID, LW_MMU_PTE_ADDRESS_SHIFT);
    INIT_FIELD_DESC32(&pPte->fldPeerIndex, LW_MMU_PTE_ADDRESS_VID_PEER);
    INIT_FIELD_BOOL(&pPte->fldVolatile, LW_MMU_PTE_VOL);
    INIT_FIELD_BOOL(&pPte->fldReadOnly, LW_MMU_PTE_READ_ONLY);
    INIT_FIELD_BOOL(&pPte->fldPrivilege, LW_MMU_PTE_PRIVILEGE);
    INIT_FIELD_BOOL(&pPte->fldEncrypted, LW_MMU_PTE_ENCRYPTED);
    INIT_FIELD_BOOL(&pPte->fldLocked, LW_MMU_PTE_LOCK);
    INIT_FIELD_DESC32(&pPte->fldKind, LW_MMU_PTE_KIND);
    INIT_FIELD_DESC32(&pPte->fldCompTagLine, LW_MMU_PTE_COMPTAGLINE);
}

LW_STATUS
vmemInitLayout_GK104(VMemSpace *pVMemSpace)
{
    VMemLayout *pVMemLayout;
    LwU32       bigPageShift;

    if (!pVMemSpace)
    {
        return LW_ERR_GENERIC;
    }

    bigPageShift = BIT_IDX_32(pVMemSpace->bigPageSize);
    if (!VALID_SHIFT(bigPageShift))
    {
        return LW_ERR_GENERIC;
    }

    pVMemLayout = &pVMemSpace->layout;

    memset(pVMemLayout, 0, sizeof(VMemLayout));

    pVMemLayout->fmt.gmmu.version          = GMMU_FMT_VERSION_1;
    pVMemLayout->fmt.gmmu.pPdeMulti        = &pVMemLayout->pdeMulti.gmmu;
    pVMemLayout->fmt.gmmu.pPde             = &pVMemLayout->pde.gmmu;
    pVMemLayout->fmt.gmmu.pPte             = &pVMemLayout->pte.gmmu;
    pVMemLayout->fmt.gmmu.bSparseHwSupport = LW_TRUE;

    pMmu[indexGpu].mmuFmtInitPdeApertures(pVMemLayout->pdeApertures);
    pMmu[indexGpu].mmuFmtInitPteApertures(pVMemLayout->pteApertures);

    _gmmuFmtInitPdeMulti_GK104(&pVMemLayout->pdeMulti.gmmu, pVMemLayout->pdeApertures);
    pVmem[indexGpu].vmemGmmuFmtInitPte(&pVMemLayout->pte.gmmu, pVMemLayout->pteApertures);
    _gmmuInitMmuFmt_GK104(pVMemSpace->layout.fmtLevels, bigPageShift);

    return LW_OK;
}

const MMU_FMT_PTE*
vmemGetPTEFmt_GK104(VMemSpace *pVMemSpace)
{
    return pVMemSpace ? &pVMemSpace->layout.pte : NULL;
}

LW_STATUS
vmemGetPDEFmt_GK104
(
    VMemSpace  *pVMemSpace,
    VMemFmtPde *pFmtPde,
    LwU32       level
)
{
    if (!pFmtPde || level != 0)
    {
        return LW_ERR_GENERIC;
    }

    pFmtPde->bMulti = TRUE;
    _gmmuFmtInitPdeMulti_GK104(&pFmtPde->fmts.multi.gmmu, pVMemSpace->layout.pdeApertures);

    return LW_OK;
}

LW_STATUS
vmemIlwalidatePDB_GK104(VMemSpace *pVMemSpace)
{
    InstBlock *pInstBlock;
    LwU32      pdb;
    LwU32      aperture;

    if (!pVMemSpace)
    {
        return LW_ERR_GENERIC;
    }

    pInstBlock = &pVMemSpace->instBlock;
    pInstBlock->readFn(pInstBlock->instBlockAddr + SF_OFFSET(LW_RAMIN_PAGE_DIR_BASE_TARGET), &pdb, 4);

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

    GPU_REG_WR32(LW_PFB_PRI_MMU_ILWALIDATE_PDB,
             DRF_NUM(_PFB_PRI, _MMU_ILWALIDATE_PDB, _ADDR, pdb) | DRF_NUM(_PFB_PRI, _MMU_ILWALIDATE_PDB, _APERTURE, aperture));

    GPU_REG_WR32(LW_PFB_PRI_MMU_ILWALIDATE, DRF_DEF(_PFB_PRI, _MMU_ILWALIDATE, _ALL_VA,  _TRUE) |
                                            DRF_DEF(_PFB_PRI, _MMU_ILWALIDATE, _ALL_PDB, _TRUE) |
                                            DRF_DEF(_PFB_PRI, _MMU_ILWALIDATE, _TRIGGER, _TRUE));

    return LW_OK;
}

GMMU_APERTURE
vmemGetPDBAperture_GK104(VMemSpace *pVMemSpace)
{
    InstBlock *pInstBlock;
    LwU32      pdb;

    if (!pVMemSpace)
    {
        return GMMU_APERTURE_ILWALID;
    }

    pInstBlock = &pVMemSpace->instBlock;
    pInstBlock->readFn(pInstBlock->instBlockAddr + SF_OFFSET(LW_RAMIN_PAGE_DIR_BASE_TARGET), &pdb, 4);

    switch (SF_VAL(_RAMIN, _PAGE_DIR_BASE_TARGET, pdb))
    {
    case LW_RAMIN_PAGE_DIR_BASE_TARGET_VID_MEM:
        return GMMU_APERTURE_VIDEO;
    case LW_RAMIN_PAGE_DIR_BASE_TARGET_SYS_MEM_COHERENT:
        return GMMU_APERTURE_SYS_COH;
    case LW_RAMIN_PAGE_DIR_BASE_TARGET_SYS_MEM_NONCOHERENT:
        return GMMU_APERTURE_SYS_NONCOH;
    default:
        return GMMU_APERTURE_ILWALID;
    }

    return GMMU_APERTURE_ILWALID;
}

LwU32
vmemSWToHWLevel_GK104(VMemSpace *pVMemSpace, LwU32 level)
{
    return vmemSWToHWLevel_STUB(pVMemSpace, level);
}

//-----------------------------------------------------
// Colwerts an instance memory target to a MEM_TYPE.
// @param[in] instMemTarget
// @param[out] pMemType
//-----------------------------------------------------
LW_STATUS
vmemGetMemTypeFromTarget_GK104(LwU64 instMemTarget, LwU64* pMemType)
{
    LW_STATUS status = LW_OK;
    MEM_TYPE memType = SYSTEM_PHYS;

    switch (instMemTarget)
    {
        case LW_PCCSR_CHANNEL_INST_TARGET_VID_MEM:
            memType = FRAMEBUFFER;
            break;

        case LW_PCCSR_CHANNEL_INST_TARGET_SYS_MEM_COHERENT:
        case LW_PCCSR_CHANNEL_INST_TARGET_SYS_MEM_NONCOHERENT:
            memType = SYSTEM_PHYS;
            break;

        default:
            status = LW_ERR_GENERIC;
    }

    if (status == LW_OK && pMemType)
    {
        *pMemType = (LwU64)memType;
    }

    return status;
}

void
vmemDumpPdeFlags_GK104(const GMMU_FMT_PDE *pFmt, const GMMU_ENTRY_VALUE *pPde)
{
    PRINT_FIELD_BOOL("Vol=%d", pFmt, Volatile, pPde);
}

LwBool
vmemIsPdeVolatile_GK104(const GMMU_FMT_PDE *pFmt, const GMMU_ENTRY_VALUE *pPde)
{
    return lwFieldGetBool(&pFmt->fldVolatile, pPde->v8);
}
