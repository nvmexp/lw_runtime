/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//-----------------------------------------------------
//
// hda0200.c - HDA routines
// 
//-----------------------------------------------------


#include "kepler/gk104/dev_hdafalcon_pri.h"
#include "kepler/gk104/dev_falcon_v4.h"

#include "hda.h"

#include "g_hda_private.h"     // (rmconfig)  implementation prototypes

//-----------------------------------------------------
// hdaIsSupported_v02_00
//-----------------------------------------------------
BOOL hdaIsSupported_v02_00( LwU32 indexGpu )
{
    dprintf("HDA is supported on this GPU\n");
    return TRUE;
}

//-----------------------------------------------------
// hdaDumpImem_v02_00 - Dumps HDA instruction memory
//-----------------------------------------------------
LW_STATUS hdaDumpImem_v02_00( LwU32 indexGpu , LwU32 imemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 imemSizeMax;
    LwU32 addrssImem = LW_PHDAFALCON_FALCON_IMEMD(0);
    LwU32 address2Imem = LW_PHDAFALCON_FALCON_IMEMC(0);
    LwU32 address2Imemt = LW_PHDAFALCON_FALCON_IMEMT(0);
    LwU32 u;
    LwU32 blk=0;
    imemSizeMax = (GPU_REG_RD_DRF(_PHDAFALCON_FALCON, _HWCFG, _IMEM_SIZE)<<8) ;
    if (imemSize > 0)
        imemSize = min(imemSizeMax, imemSize);
    else
        imemSize = imemSizeMax;

    dprintf("\n");
    dprintf("lw: -- Gpu %u HDAFALCON IMEM -- \n", indexGpu);    
    dprintf("lw: -- Gpu %u HDAFALCON IMEM SIZE =  0x%08x-- \n", indexGpu,imemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    for(u=0;u<(imemSize+3)/4;u++)
    {
        LwU32 i;
        if((u%64)==0) {
            GPU_REG_WR32(address2Imemt, blk++);
        }
        i = (u<<(0?LW_PHDAFALCON_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2Imem,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  GPU_REG_RD32(addrssImem));
    }
    dprintf("\n");
    // IMEM dump done

    return status;
}

//-----------------------------------------------------
// hdaDumpDmem_v02_00 - Dumps HDA data memory
//-----------------------------------------------------
LW_STATUS hdaDumpDmem_v02_00( LwU32 indexGpu , LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    // these are the variables defined for use in parsing and printinf the methods and data
    LwU32 addrss, address2, u, i = 0, classNum;

    dmemSizeMax = (GPU_REG_RD_DRF(_PHDAFALCON_FALCON, _HWCFG, _DMEM_SIZE)<<8) ;

    if(dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
       dmemSize = dmemSizeMax;

    addrss      = LW_PHDAFALCON_FALCON_DMEMD(0);
    address2    = LW_PHDAFALCON_FALCON_DMEMC(0);
    classNum    = 0xC0B7;

    dprintf("\n");
    dprintf("lw: -- Gpu %u HDAFALCON DMEM -- \n", indexGpu);
    dprintf("lw: -- Gpu %u HDAFALCON DMEM SIZE =  0x%08x-- \n", indexGpu,dmemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");

    i = 0;
    for(u=0;u<(dmemSize+3)/4;u++)
    {
        i = (u<<(0?LW_PHDAFALCON_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  GPU_REG_RD32(addrss));
    }
    dprintf("\n");

    // DMEM dump end

    return status;
}

//-----------------------------------------------------
// hdaTestState_v02_00 - Test basic HDA state
//-----------------------------------------------------
LW_STATUS hdaTestState_v02_00( LwU32 indexGpu )
{
    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   data32;

    //check falcon interrupts
    regIntr = GPU_REG_RD32(LW_PHDAFALCON_FALCON_IRQSTAT);
    regIntrEn = GPU_REG_RD32(LW_PHDAFALCON_FALCON_IRQMASK);
    regIntr &= regIntrEn;

    if ( !DRF_VAL(_PHDAFALCON, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQMASK_GPTMR disabled\n");

    if ( !DRF_VAL(_PHDAFALCON, _FALCON_IRQMASK, _WDTMR, regIntrEn))
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQMASK_WDTMR disabled\n");

    if ( !DRF_VAL(_PHDAFALCON, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQMASK_MTHD disabled\n");

    if ( !DRF_VAL(_PHDAFALCON, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQMASK_CTXSW disabled\n");

    if ( !DRF_VAL(_PHDAFALCON, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQMASK_HALT disabled\n");

    if ( !DRF_VAL(_PHDAFALCON, _FALCON_IRQMASK, _EXTERR, regIntrEn))
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQMASK_EXTERR disabled\n");

    if ( !DRF_VAL(_PHDAFALCON, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQMASK_SWGEN0 disabled\n");

    if ( !DRF_VAL(_PHDAFALCON, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQMASK_SWGEN1 disabled\n");

   
    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PHDAFALCON,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQSTAT_GPTMR pending\n");

        dprintf("lw: LW_PHDAFALCON_FALCON_GPTMRINT:    0x%08x\n", 
            GPU_REG_RD32(LW_PHDAFALCON_FALCON_GPTMRINT) );
        dprintf("lw: LW_PHDAFALCON_FALCON_GPTMRVAL:    0x%08x\n", 
            GPU_REG_RD32(LW_PHDAFALCON_FALCON_GPTMRVAL) );
        
    }
    
    if ( DRF_VAL( _PHDAFALCON,_FALCON_IRQSTAT, _WDTMR, regIntr))
    {
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQSTAT_WDTMR pending\n");
    }

    if ( DRF_VAL( _PHDAFALCON,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQSTAT_MTHD pending\n");

        dprintf("lw: LW_PHDAFALCON_FALCON_MTHDDATA_DATA:    0x%08x\n", 
            GPU_REG_RD32(LW_PHDAFALCON_FALCON_MTHDDATA) );
        
        data32 = GPU_REG_RD32(LW_PHDAFALCON_FALCON_MTHDID);
        dprintf("lw: LW_PHDAFALCON_FALCON_MTHDID_ID:    0x%08x\n", 
           DRF_VAL( _PHDAFALCON,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_PHDAFALCON_FALCON_MTHDID_SUBCH:    0x%08x\n", 
           DRF_VAL( _PHDAFALCON,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_PHDAFALCON_FALCON_MTHDID_PRIV:    0x%08x\n", 
           DRF_VAL( _PHDAFALCON,_FALCON_MTHDID, _PRIV, data32)  );
    }
    
    if ( DRF_VAL( _PHDAFALCON,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQSTAT_CTXSW pending\n");
    }
    
    if ( DRF_VAL( _PHDAFALCON,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQSTAT_HALT pending\n");
    }
    
    if ( DRF_VAL( _PHDAFALCON,_FALCON_IRQSTAT, _EXTERR, regIntr))
    {
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQSTAT_EXTERR pending\n");
    }
    
    if ( DRF_VAL( _PHDAFALCON,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQSTAT_SWGEN0 pending\n");

        pFalcon[indexGpu].falconPrintMailbox(LW_FALCON_HDA_BASE);
    }

    if ( DRF_VAL( _PHDAFALCON,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_PHDAFALCON_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

     //
    //print falcon states
    //Bit |  Signal meaning
    //0      FALCON busy
    //

    data32 = GPU_REG_RD32(LW_PHDAFALCON_FALCON_IDLESTATE);

    if ( DRF_VAL( _PHDAFALCON, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

  
    data32 = GPU_REG_RD32(LW_PHDAFALCON_FALCON_FHSTATE);
 
    if ( DRF_VAL( _PHDAFALCON, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_FHSTATE_FALCON_HALTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PHDAFALCON, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PHDAFALCON, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = GPU_REG_RD32(LW_PHDAFALCON_FALCON_ENGCTL);
    
    if ( DRF_VAL( _PHDAFALCON, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_ENGCTL_ILW_CONTEXT\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PHDAFALCON, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PHDAFALCON_FALCON_CPUCTL);

    if ( DRF_VAL( _PHDAFALCON, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PHDAFALCON, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PHDAFALCON, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface 
    data32 = GPU_REG_RD32(LW_PHDAFALCON_FALCON_ITFEN);

    if (DRF_VAL( _PHDAFALCON, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_ITFEN_CTXEN enabled\n");
             
        if (pFalcon[indexGpu].falconTestCtxState(LW_FALCON_HDA_BASE, "PHDAFALCON") == LW_ERR_GENERIC)
        {
            dprintf("lw: Current ctx state invalid\n");
            status = LW_ERR_GENERIC;
        }
        else
        {
            dprintf("lw: Current ctx state valid\n");
        }
    }
    else
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_ITFEN_CTXEN disabled\n");
    }

    if ( DRF_VAL( _PHDAFALCON, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_PHDAFALCON_FALCON_ITFEN_MTHDEN disabled\n");
    }

    //check if falcon is hung (instr ptr)
    if ( pFalcon[indexGpu].falconTestPC(LW_FALCON_HDA_BASE, "PHDAFALCON") == LW_ERR_GENERIC )
    {
        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");
        
        //TODO: treat falcon PC errors as warnings now, need to report as error
        //status = LW_ERR_GENERIC;
    }

    return status;  
}
