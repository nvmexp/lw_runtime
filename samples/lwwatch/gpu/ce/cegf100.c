/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2008-2014 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


//
// includes
//

#include "fermi/gf100/dev_ce_pri.h"
#include "gpuanalyze.h"
#include "mmu.h"
#include "falcon.h"

#include "g_ce_private.h"     // (rmconfig)  implementation prototypes


/*!
 *  
 *  check Ce engine status by checking _FALCON and _COP intr states
 *  @param[in]      CE base offset
 *  
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
static LW_STATUS checkCeState_GF100( LwU32 base )
{
    LW_STATUS status = LW_OK;
    LwU32     data32;
    LwU32     regIntr;
    LwU32     regIntrEn;
    char      engName[8];

    //set the CE engine #
     sprintf( engName, "PCE%d", (base == LW_PCE_CE1_BASE) ? 1 : 0);

    //check falcon interrupts
    regIntr = GPU_REG_RD32( LW_PCE_FALCON_IRQSTAT + base );
    regIntrEn = GPU_REG_RD32( LW_PCE_FALCON_IRQMASK + base );
    regIntr &= regIntrEn;

    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t LW_%s_FALCON_IRQSTAT interrupts are pending\n",
            engName);
        status = LW_ERR_GENERIC;
    }


    if ( DRF_VAL( _PCE,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_%s_FALCON_IRQSTAT_GPTMR pending\n", engName);

        dprintf("lw: LW_%s_FALCON_GPTMRINT:    0x%08x\n", 
            engName, GPU_REG_RD32(LW_PCE_FALCON_GPTMRINT + base) );
        dprintf("lw: LW_%s_FALCON_GPTMRVAL:    0x%08x\n", 
            engName, GPU_REG_RD32(LW_PCE_FALCON_GPTMRVAL + base) );

    }

    if ( DRF_VAL( _PCE,_FALCON_IRQSTAT, _WPTMR, regIntr))
    {
        dprintf("lw: LW_%s_FALCON_IRQSTAT_WPTMR pending\n", engName);
    }

    if ( DRF_VAL( _PCE,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_%s_FALCON_IRQSTAT_MTHD pending\n", engName);

        dprintf("lw: LW_%s_FALCON_MTHDDATA_DATA:    0x%08x\n", 
            engName, GPU_REG_RD32(LW_PCE_FALCON_MTHDDATA + base) );

        data32 = GPU_REG_RD32(LW_PCE_FALCON_MTHDID + base);

        dprintf("lw: LW_%s_FALCON_MTHDID_ID:        0x%08x\n", 
            engName, DRF_VAL( _PCE,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_%s_FALCON_MTHDID_SUBCH:     0x%08x\n", 
            engName, DRF_VAL( _PCE,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_%s_FALCON_MTHDID_PRIV:      0x%08x\n", 
            engName, DRF_VAL( _PCE,_FALCON_MTHDID, _PRIV, data32)  );
    }

    if ( DRF_VAL( _PCE,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_%s_FALCON_IRQSTAT_CTXSW pending\n", engName);
    }

    if ( DRF_VAL( _PCE,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_%s_FALCON_IRQSTAT_HALT pending\n", engName);
    }

    if ( DRF_VAL( _PCE,_FALCON_IRQSTAT, _STALLREQ, regIntr))
    {
        dprintf("lw: LW_%s_FALCON_IRQSTAT_STALLREQ pending\n", engName);
    }

    if ( DRF_VAL( _PCE,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_%s_FALCON_IRQSTAT_SWGEN0 pending\n", engName);

        pFalcon[indexGpu].falconPrintMailbox( base );
    }

    if ( DRF_VAL( _PCE,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_%s_FALCON_IRQSTAT_SWGEN1 pending\n", engName);
    }

    //
    // Bit  |  Signal meaning
    // 8      FBIF Ctx error interrupt.  
    // 9      Limit violation interrupt. 
    // 10     Stalling interrupt requested via rom Launchdma method 
    // 11     Non-stalling interrupt requested via rom Launchdma method 
    //

    if ( regIntr & BIT(8))
    {
        dprintf("lw: LW_%s_FALCON_IRQSTAT: FBIF Ctx error interrupt\n", engName);
        data32 = GPU_REG_RD32(LW_PCE_FBIF_CTL + base);

        if (DRF_VAL(_PCE, _FBIF_CTL, _ENABLE, data32))
        {
            if (DRF_VAL(_PCE, _FBIF_CTL, _ILWAL_CONTEXT, data32))
            {
                dprintf("lw: + LW_%s_FBIF_CTL_ILWAL_CONTEXT\n", engName);
            }
        }
    }

    if (regIntr & BIT(9))
    {
        dprintf("lw: LW_%s_FALCON_IRQSTAT: Limit violation interrupt\n", engName);
    }


    if (regIntr & BIT(10))
    {
        dprintf("lw: LW_%s_FALCON_IRQSTAT: Stalling interrupt from "
            "Launchdma method pending\n", engName);
    }

    if (regIntr & BIT(11))
    {
        dprintf("lw: LW_%s_FALCON_IRQSTAT: Non-Stalling interrupt from "
            "Launchdma method pending\n", engName);
    }

    //print memctl status
    data32 = GPU_REG_RD32(LW_PCE_PRI_MEMCTL + base);
    dprintf("lw: LW_%s_ PRI_MEMCTL:      0x%08x\n", engName, data32);
    if ( DRF_VAL( _PCE, _PRI_MEMCTL, _SELWRE_WRITE_VIO, data32))
    {
        dprintf("lw: + LW_%s_ PRI_MEMCTL_SELWRE_WRITE_VIO\n", engName);
        addUnitErr("\t LW_%s_ PRI_MEMCTL_SELWRE_WRITE_VIO\n", engName);
        status = LW_ERR_GENERIC;
    }

    //
    //print falcon states
    //     Bit |  Signal meaning
    //     0      FALCON busy
    //     1      FBIF busy (includes COP as the EXT unit of FBIF)
    //

    data32 = GPU_REG_RD32(LW_PCE_FALCON_IDLESTATE + base);
    if ( DRF_VAL( _PCE, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_%s_ FALCON_IDLESTATE_FALCON_BUSY\n", engName);
        addUnitErr("\t LW_%s_ FALCON_IDLESTATE_FALCON_BUSY\n", engName);
        status = LW_ERR_GENERIC;
    }
    if ( data32 & DRF_SHIFTMASK(1:1))
    {
        dprintf("lw: + LW_%s_ FALCON_IDLESTATE_FBIF_BUSY\n", engName);
        addUnitErr("\t LW_%s_ FALCON_IDLESTATE_FBIF_BUSY\n", engName);
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PCE_FALCON_FHSTATE + base);
    if ( DRF_VAL( _PCE, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_%s_ FALCON_FHSTATE_FALCON_HALTED\n", engName);
        addUnitErr("\t LW_%s_ FALCON_FHSTATE_FALCON_HALTED\n", engName);
        status = LW_ERR_GENERIC;
    }
    if ( data32 & DRF_SHIFTMASK(1:1))
    {
        dprintf("lw: + LW_%s_ FALCON_FHSTATE_FBIF_HALTED\n", engName);
        addUnitErr("\t LW_%s_ FALCON_FHSTATE_FBIF_HALTED\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PCE, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_%s_ FALCON_FHSTATE_ENGINE_FAULTED\n", engName);
        addUnitErr("\t LW_%s_ FALCON_FHSTATE_ENGINE_FAULTED\n", engName);
        status = LW_ERR_GENERIC;
    }
    if ( DRF_VAL( _PCE, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_%s_ FALCON_FHSTATE_STALL_REQ\n", engName);
        addUnitErr("\t LW_%s_ FALCON_FHSTATE_STALL_REQ\n", engName);
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = GPU_REG_RD32(LW_PCE_FALCON_ENGCTL + base);
    if ( DRF_VAL( _PCE, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_%s_ FALCON_ENGCTL_ILW_CONTEXT\n", engName);
        addUnitErr("\t LW_%s_ FALCON_ENGCTL_ILW_CONTEXT\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PCE, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_%s_ FALCON_ENGCTL_STALLREQ\n", engName);
        addUnitErr("\t LW_%s_ FALCON_ENGCTL_STALLREQ\n", engName);
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PCE_FALCON_CPUCTL + base);
    if ( DRF_VAL( _PCE, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_%s_ FALCON_CPUCTL_IILWAL\n", engName);
        addUnitErr("\t LW_%s_ FALCON_CPUCTL_IILWAL\n", engName);
        status = LW_ERR_GENERIC;
    }
    if ( DRF_VAL( _PCE, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_%s_ FALCON_CPUCTL_HALTED\n", engName);
        addUnitErr("\t LW_%s_ FALCON_CPUCTL_HALTED\n", engName);
        status = LW_ERR_GENERIC;
    }
    if ( DRF_VAL( _PCE, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_%s_ FALCON_CPUCTL_STOPPED\n", engName);
        addUnitErr("\t Warning: LW_%s_ FALCON_CPUCTL_STOPPED\n", engName);
        status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface 
    data32 = GPU_REG_RD32(LW_PCE_FALCON_ITFEN + base);
    if (DRF_VAL( _PCE, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_%s_ FALCON_ITFEN_CTXEN enabled\n", engName);

        if (pFalcon[indexGpu].falconTestCtxState( base, engName) == LW_ERR_GENERIC)
        {
            dprintf("lw: Current ctx state invalid\n");
            addUnitErr("\t Current ctx state is invalid\n");
            status = LW_ERR_GENERIC;
        }
        else
        {
            dprintf("lw: Current ctx state valid\n");
        }
    }
    else
    {
        dprintf("lw: + LW_%s_FALCON_ITFEN_CTXEN disabled\n", engName);
    }

    if ( DRF_VAL( _PCE, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_%s_FALCON_ITFEN_MTHDEN enabled\n", engName);
    }
    else
    {
        dprintf("lw: + LW_%s_FALCON_ITFEN_MTHDEN disabled\n", engName);
    }

    //check if PC is stuck
    if ( pFalcon[indexGpu].falconTestPC( base, engName) == LW_ERR_GENERIC )
    {

        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");

        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;
    }


    //check COP interrupts
    regIntr = GPU_REG_RD32( LW_PCE_COP2_INTRPT_STATUS + base );
    regIntrEn = GPU_REG_RD32( LW_PCE_COP2_INTRPT_EN + base );
    regIntr &= regIntrEn;

    if ( DRF_VAL(_PCE, _COP2_INTRPT_STATUS, _BLOCKPIPE, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_INTRPT_STATUS_BLOCKPIPE_ACTIVE\n", engName);
        addUnitErr("\t LW_%s_COP2_INTRPT_STATUS_BLOCKPIPE_ACTIVE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_INTRPT_STATUS, _NONBLOCKPIPE, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_INTRPT_STATUS_NONBLOCKPIPE_ACTIVE\n", engName);
        addUnitErr("\t LW_%s_COP2_INTRPT_STATUS_NONBLOCKPIPE_ACTIVE\n", engName);
        status = LW_ERR_GENERIC;
    }

    //check COP status
    data32 = GPU_REG_RD32( LW_PCE_COP2_PIPESTATUS + base );
    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _CTL, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_CTL_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_CTL_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _GSTRIP, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_GSTRIP_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_GSTRIP_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _RALIGN, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_RALIGN_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_RALIGN_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _SWIZ, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_SWIZ_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_SWIZ_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }
    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _WALIGN, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_WALIGN_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_WALIGN_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _GPAD, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_GPAD_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_GPAD_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _RDAT, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_RDAT_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_RDAT_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _RDACK, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_RDACK_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_RDACK_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _WRACK, regIntrEn) )
    {
        dprintf("lw: LW_%s_COP2_PIPESTATUS_WRACK_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_WRACK_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _RDAT, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_RDAT_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_RDAT_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _BLOCKINTRPT, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_BLOCKINTRPT_ACTIVE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_BLOCKINTRPT_ACTIVE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _RCMD_STALL, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_RCMD_STALL_ACTIVE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_RCMD_STALL_ACTIVE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _WCMD_STALL, regIntrEn) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_WCMD_STALL_ACTIVE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_WCMD_STALL_ACTIVE\n", engName);
        status = LW_ERR_GENERIC;
    }

    return status;
}


#define NUM_CE    2

LwU32 ceBase[NUM_CE] = {
    LW_PCE_CE0_BASE,
    LW_PCE_CE1_BASE
};

 
/*!
 *  
 *  show Ce state
 *  @param[in]      which engine to test: 0-CE0 , 1-CE1 , 2-both
 *  
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS ceTestCeState_GF100( LwU32 indexGpu, LwU32 eng )
{
    LW_STATUS status = LW_OK;
    LwU32     i,limit;

    if (eng > NUM_CE)
    {
        dprintf("lw:  Improper engine index\n");
        return LW_ERR_GENERIC;
    }

    if (eng == NUM_CE)
    {
        limit = NUM_CE-1;
        i = 0;
    }
    else
    {
        i = limit = eng;
    }

    for (;i<=limit;i++)
    {
        dprintf("\n\tlw: ******** CE%d state test... ********\n", i);
        status = checkCeState_GF100(ceBase[i]);
        status == LW_ERR_GENERIC ? dprintf("lw: ******** CE%d state test FAILED ********\n", i):
                             dprintf("lw: ******** CE%d state test succeeded ********\n", i);
    }

   return status;
}
