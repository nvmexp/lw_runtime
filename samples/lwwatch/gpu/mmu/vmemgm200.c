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
// adimitrov@lwpu.com - July 20 2007
// vmemgm200.c - page table routines for Fermi
//
//*****************************************************

//
// Includes
//
#include "fifo.h"
#include "maxwell/gm200/dev_bus.h"
#include "maxwell/gm200/dev_fb.h"
#include "maxwell/gm200/dev_ram.h"
#include "turing/tu102/dev_vm.h"
#include "vgpu.h"

#include "g_mmu_private.h"
#include "g_vmem_private.h"

LwU32
vmemGetBigPageSize_GM200(VMemSpace *pVMemSpace)
{
    InstBlock*  pInstBlock;
    LwU32 buf = 0;
    LwU64 instMemAddr = 0;
    //for brevity
    pInstBlock = &(pVMemSpace->instBlock);
    instMemAddr = pInstBlock->instBlockAddr;
    
    // Check for big page size stored in instance block    
    if ((GPU_REG_RD_DRF(_PFB_PRI,_MMU_CTRL, _USE_PDB_BIG_PAGE_SIZE) == LW_PFB_PRI_MMU_CTRL_USE_PDB_BIG_PAGE_SIZE_TRUE))
    {
        if (pInstBlock->readFn == NULL)
        {
            dprintf("**ERROR: NULL value of readFn.\n");
            return LW_ERR_NOT_SUPPORTED;
        }
        // Get big page size from instance block
        if (LW_OK != pInstBlock->readFn(instMemAddr + SF_OFFSET(LW_RAMIN_BIG_PAGE_SIZE), &buf, 4))
        {
            return 0;
        }
        if (SF_VAL(_RAMIN, _BIG_PAGE_SIZE, buf) == LW_RAMIN_BIG_PAGE_SIZE_128KB)
        {
            return (LwU32) VMEM_BIG_PAGESIZE_128K;
        }
        else if (SF_VAL(_RAMIN, _BIG_PAGE_SIZE, buf) == LW_RAMIN_BIG_PAGE_SIZE_64KB)
        {
            return (LwU32) VMEM_BIG_PAGESIZE_64K;
        }
        else 
        {
            dprintf("lwwatch: Big page size is set up wrong\n");         
            return 0;
        }
    }
    else    // Big page size from MMU PRI control
    {
        // Return big page size from MMU PRI control register
        return (LwU32) (GPU_REG_RD_DRF(_PFB_PRI,_MMU_CTRL, _VM_PG_SIZE) == LW_PFB_PRI_MMU_CTRL_VM_PG_SIZE_128KB)?
            VMEM_BIG_PAGESIZE_128K : VMEM_BIG_PAGESIZE_64K;
    }    
}

void
vmemRebindBAR1Block_GM200(void)
{
    LwU32 reg;
    reg = GPU_REG_RD32(LW_PBUS_BAR1_BLOCK);
    GPU_REG_WR32(LW_PBUS_BAR1_BLOCK, reg);
}

LW_STATUS
vmemSetBigPageSize_GM200(VMemSpace *pVMemSpace, LwU32 pageSize)
{
    InstBlock*  pInstBlock;
    LwU32 reg;
    LwU32 buf = 0;
    LwU64 instMemAddr = 0;
    //for brevity
    pInstBlock = &(pVMemSpace->instBlock);
    instMemAddr = pInstBlock->instBlockAddr;

    // Check for page size coming from instance block    
    if ((GPU_REG_RD_DRF(_PFB_PRI,_MMU_CTRL, _USE_PDB_BIG_PAGE_SIZE) == LW_PFB_PRI_MMU_CTRL_USE_PDB_BIG_PAGE_SIZE_TRUE))
    {
        if (pInstBlock->readFn == NULL)
        {
            dprintf("**ERROR: NULL value of readFn.\n");
            return LW_ERR_NOT_SUPPORTED;
        }
        // Get size value from instance block
        if (LW_OK != pInstBlock->readFn(instMemAddr + SF_OFFSET(LW_RAMIN_BIG_PAGE_SIZE), &buf, 4))
        {
            return LW_ERR_GENERIC;
        }
        // Switch on the page size value
        switch(pageSize)
        {
            case VMEM_BIG_PAGESIZE_64K:

                // Set the requested page size
                buf = (buf & ~(SF_MASK(LW_RAMIN_BIG_PAGE_SIZE) << SF_SHIFT(LW_RAMIN_BIG_PAGE_SIZE))) | (LW_RAMIN_BIG_PAGE_SIZE_64KB << SF_SHIFT(LW_RAMIN_BIG_PAGE_SIZE));

                break;

            case VMEM_BIG_PAGESIZE_128K:

                // Set the requested page size
                buf = (buf & ~(SF_MASK(LW_RAMIN_BIG_PAGE_SIZE) << SF_SHIFT(LW_RAMIN_BIG_PAGE_SIZE))) | (LW_RAMIN_BIG_PAGE_SIZE_128KB << SF_SHIFT(LW_RAMIN_BIG_PAGE_SIZE));

                break;

            default:

                return LW_ERR_GENERIC;
        }
        // Set size value into instance block
        if (LW_OK != pInstBlock->writeFn(instMemAddr + SF_OFFSET(LW_RAMIN_BIG_PAGE_SIZE), &buf, 4))
        {
            return LW_ERR_GENERIC;
        }

        pVmem[indexGpu].vmemRebindBAR1Block();
    }
    else        // Big page size from MMU PRI control
    {
        // Read the base value to use in page size update
        reg = GPU_REG_RD32(LW_PFB_PRI_MMU_CTRL);

        // Switch on the page size value
        switch(pageSize)
        {
            case VMEM_BIG_PAGESIZE_64K:

                // Set the requested page size
                reg = FLD_SET_DRF(_PFB_PRI, _MMU_CTRL, _VM_PG_SIZE, _64KB, reg);

                break;

            case VMEM_BIG_PAGESIZE_128K:

                // Set the requested page size
                reg = FLD_SET_DRF(_PFB_PRI,_MMU_CTRL, _VM_PG_SIZE, _128KB, reg);

                break;

            default:

                return LW_ERR_GENERIC;
        }
        // Update big page size in MMU PRI control register
        GPU_REG_WR32(LW_PFB_PRI_MMU_CTRL, reg);
    }    
    return LW_OK;
}
