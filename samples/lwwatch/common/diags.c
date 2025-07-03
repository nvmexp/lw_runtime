/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// diags.c
//
//*****************************************************

//
// includes
//
#include "kepler/gk104/dev_fifo.h"
#include "kepler/gk104/dev_master.h"
#include "diags.h"
#include "hal.h"
#include "chip.h"
#include "disp.h"
#include "dcb.h"
#include "fifo.h"
#include "gr.h"

//-----------------------------------------------------
// diagFillStruct
//
//-----------------------------------------------------
void diagFillStruct(PLWWATCHDIAGSTRUCT pdiagstruct)
{
    pdiagstruct->lw_pfifo_intr_0       =  GPU_REG_RD32(LW_PFIFO_INTR_0);
    pdiagstruct->lw_pfifo_intr_en_0    =  GPU_REG_RD32(LW_PFIFO_INTR_EN_0);
    
    // Ampere removed the LW_PMC_INTR registers
    if (!IsGA100orLater())
    {
        pdiagstruct->lw_pmc_intr_0         =  GPU_REG_RD32(LW_PMC_INTR_0);
        pdiagstruct->lw_pmc_intr_en_0      =  GPU_REG_RD32(LW_PMC_INTR_EN_0);
        pdiagstruct->lw_pmc_enable         =  GPU_REG_RD32(LW_PMC_ENABLE);
    }
}

//-----------------------------------------------------
// diagMaster
//
//-----------------------------------------------------
void diagMaster(PLWWATCHDIAGSTRUCT pdiagstruct)
{
    // Ampere removed the LW_PMC_INTR registers
    if (!IsGA100orLater())
    {
        dprintf("lw: **********************************************\n");
        dprintf("lw: Running Master Control diagnostic...\n");
        dprintf("lw: **********************************************\n");
        
        if (IsTU102orLater())
        {
            dprintf("lw: Warning: Most interrupts do not report to LW_PMC_INTR on Turing.\n");
            dprintf("lw: Warning: See -intr command for interrupts that do not report to LW_PMC_INTR.\n");
        }

        //
        // what interrupts are pending?
        //
        if (pdiagstruct->lw_pmc_intr_0)
        {
            dprintf("lw: LW_PMC_INTR_0:     0x%08x\n", pdiagstruct->lw_pmc_intr_0);
            dprintf("lw: The Chip has Interrupts pending.\n");
            dprintf("\n");

            dprintf("lw: LW_PMC_INTR_EN_0:  0x%08x\n", pdiagstruct->lw_pmc_intr_en_0);
            if (!pdiagstruct->lw_pmc_intr_en_0)
            {
                dprintf("lw: The Chip has Interrupts disabled.\n");
                dprintf("lw:   - WARNING - The Chip will be stalled under these conditions.\n");
            }
            else
            {
                dprintf("lw: The Chip has Interrupts enabled.\n");
                dprintf("lw:   - All appears to be normal here.\n");
            }
            dprintf("\n");
        }
        else
        {
            dprintf("lw: LW_PMC_INTR_0:     0x%08x\n", pdiagstruct->lw_pmc_intr_0);
            dprintf("lw: The Chip has NO Interrupts pending.\n");

            dprintf("lw: LW_PMC_INTR_EN_0:      0x%08x\n\n", pdiagstruct->lw_pmc_intr_en_0);
            if (!pdiagstruct->lw_pmc_intr_en_0)
            {
                dprintf("lw: The Chip has Interrupts disabled.\n");
                dprintf("lw:   - The Chip is in an idle (disabled) state.\n");
            }
            else
            {
                dprintf("lw: The Chip has Interrupts enabled.\n");
                dprintf("lw:   - All appears to be normal here.\n");
            }
        }

        //
        // what engines are enabled?
        //
        dprintf("lw: LW_PMC_ENABLE:     0x%08x\n", pdiagstruct->lw_pmc_enable);
    }
    else
    {
        dprintf("lw: Warning: The LW_PMC_INTR registers do not exist on Ampere+.\n");
        dprintf("lw: Warning: See the -intr command for all interrupts.\n");
    }

}

//-----------------------------------------------------
// diagFifo
//
//-----------------------------------------------------
void diagFifo(PLWWATCHDIAGSTRUCT pdiagstruct)
{
    dprintf("lw: **********************************************\n");
    dprintf("lw: Running Fifo diagnostic...\n");
    dprintf("lw: **********************************************\n");

    //
    // any fifo interrupts pending?
    //
    if (pdiagstruct->lw_pfifo_intr_0)
    {
        dprintf("lw: LW_PFIFO_INTR_0:       0x%08x\n\n", pdiagstruct->lw_pfifo_intr_0);
        dprintf("lw: Fifo has Interrupts pending.\n");

        if (!pdiagstruct->lw_pfifo_intr_en_0){
            dprintf("lw: Fifo has Interrupts disabled.\n");
            dprintf("lw:   - WARNING - The fifo will be stalled under these conditions.\n");
        }else{
            dprintf("lw: Fifo has Interrupts enabled.\n");
            dprintf("lw:   - WARNING - The fifo appears to be waiting for the RM for service.\n");
            dprintf("lw:               If this condition persists there could be a problem.\n");
        }
    }
    else
    {
        if(!pdiagstruct->lw_pfifo_intr_en_0){
            dprintf("lw: Fifo has Interrupts disabled.\n");
            dprintf("lw:   - The fifo is in an idle (disabled) state.\n");
        }else{
            dprintf("lw: Fifo has Interrupts enabled.\n");
            dprintf("lw:   - All appears to be normal here.\n");
        }
    }
    dprintf("\n");

    pFifo[indexGpu].fifoGetInfo();
}

//-----------------------------------------------------
// diagGraphics
//
//-----------------------------------------------------
void diagGraphics(PLWWATCHDIAGSTRUCT pdiagstruct, LwU32 grIdx)
{
    dprintf("lw: **********************************************\n");
    dprintf("lw: Running Graphics diagnostic...\n");
    dprintf("lw: **********************************************\n");

    pGr[indexGpu].grGetInfo();
    pGr[indexGpu].grDumpFifo(TRUE, grIdx);
}
