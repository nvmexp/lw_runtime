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
// rypanwar@lwpu.com - June 28 2018
// mmuga100.c - FLA page table reading for Ampere
//
//*****************************************************

//
// Includes
//

#include "ga10x/mmuga100.h"
#include "ampere/ga100/dev_bus.h"
#include "ampere/ga100/dev_mmu.h"
#include "ampere/ga100/dev_fb.h"
#include "ampere/ga100/dev_ram.h"
#include "ampere/ga100/dev_fifo.h"
#include "ampere/ga100/dev_pwr_pri.h"
#include "os.h"
#include "chip.h"
#include "hal.h"
#include "mmu.h"
#include "inst.h"
#include "fb.h"
#include "fifo.h"
#include "bus.h"
#include "vmem.h"
#include "mmu/mmu_fmt.h"
#include "class/cl90f1.h"      // FERMI_VASPACE_A
#include "g_mmu_private.h"
#include "g_bus_private.h"
#include "g_vmem_private.h"
#include "ctrl/ctrl2080.h"


LW_STATUS fillVMemSpace_GA100(VMemSpace *pVMemSpace, LwU64 instMemAddr, readFn_t instMemReadFn, writeFn_t writeFn, MEM_TYPE pMemType);

/*!
 *  Get the VMemSpace for FLA.
 *
 *  @param[out] pVMemSpace  Virtual memory space structure to populate.
 *  @param[in]  pFla        Pointer FLA Struct
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemGetFla_GA100
(
    VMemSpace *pVMemSpace,
    VMEM_INPUT_TYPE_FLA *pFla
)
{
    readFn_t    instMemReadFn;
    writeFn_t   instMemWriteFn;
    MEM_TYPE    memType;

    if (pVMemSpace == NULL)
    {
        return LW_ERR_GENERIC;
    }

    memType = pFla->targetMemType;
    switch (memType)
    {
        case FRAMEBUFFER:
        {
            instMemReadFn = pFb[indexGpu].fbRead;
            instMemWriteFn = pFb[indexGpu].fbWrite;
            break;
        }
        case SYSTEM_PHYS:
        {
            instMemReadFn = readSystem;
            instMemWriteFn = writeSystem;
            break;
        }
        default:
        {
            dprintf("lwwatch:    Invalid IMB address specified\n");
            return LW_ERR_GENERIC;
        }
    }

    if (pFla->flaImbAddr == 0)
    {
        dprintf("lwwatch:    Invalid IMB address specified\n");
        return LW_ERR_GENERIC;
    }

    if (fillVMemSpace_GA100(pVMemSpace, pFla->flaImbAddr, instMemReadFn, instMemWriteFn, memType) != LW_OK)
    {
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}

/*!
 *  Populate the VMemSpace.(helper function)
 *
 *  @param[in]  instMemAddr      Base Address for the instance mem block.
 *  @param[in]  instMemReadFn    Function to be used to read instance block data.
 *  @param[out] pVMemSpace       Virtual memory space structure to populate.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
fillVMemSpace_GA100(VMemSpace *pVMemSpace, LwU64 instMemAddr, readFn_t instMemReadFn,
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
 * @brief Returns the VMMU segment size
 *
 */
LwU64
mmuGetSegmentSize_GA100
(void)
{
    LwU32 segmentSize;

    segmentSize = GPU_REG_RD32(LW_PFB_PRI_MMU_VMMU_CFG0);
    segmentSize = DRF_VAL( _PFB, _PRI_MMU_VMMU_CFG0, _SEGMENT_SIZE, segmentSize);

    switch(segmentSize)
    {
        case 0: 
            return LW2080_CTRL_GPU_VMMU_SEGMENT_SIZE_32MB;

        case 1: 
            return LW2080_CTRL_GPU_VMMU_SEGMENT_SIZE_64MB;

        case 2: 
            return LW2080_CTRL_GPU_VMMU_SEGMENT_SIZE_128MB;

        case 3: 
            return LW2080_CTRL_GPU_VMMU_SEGMENT_SIZE_256MB;

        case 4: 
            return LW2080_CTRL_GPU_VMMU_SEGMENT_SIZE_512MB;

        default:
        {
            dprintf("lw: Error: Invalid Segment Size\n");
            return 0;
        }
    }
}
