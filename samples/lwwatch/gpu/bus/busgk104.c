/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


//*****************************************************
//
// lwwatch extension
// busgk104.c
//
//*****************************************************

//
// includes
//
#include "kepler/gk104/dev_bus.h"
#include "kepler/gk104/dev_timer.h"
#include "kepler/gk104/dev_lw_p2p.h"
#include "gpuanalyze.h"


#include "g_bus_private.h"     // (rmconfig)  implementation prototypes


/*!
 *  Test top level interrupt tree
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS busTestBusInterrupts_GK104( void )
{
    LwU32     data32;
    LW_STATUS status = LW_OK;
    LwU32     regIntr0 = 0;
    LwU32     regIntrEn0 = 0;
   

    regIntrEn0 = GPU_REG_RD32(LW_PBUS_INTR_EN_0);
    regIntr0 = GPU_REG_RD32(LW_PBUS_INTR_0) & regIntrEn0;


//INTR_EN_0
    dprintf("lw: + LW_PBUS_INTR_EN_0_\n");

    if (DRF_VAL(_PBUS, _INTR_EN_0, _PRI_SQUASH, regIntrEn0) == 0)
        dprintf("lw: +         _PRI_SQUASH_DISABLED\n");

    if (DRF_VAL(_PBUS, _INTR_EN_0, _PRI_FECSERR, regIntrEn0) == 0)
        dprintf("lw: +         _PRI_FECSERR_DISABLED\n");

    if (DRF_VAL(_PBUS, _INTR_EN_0, _PRI_TIMEOUT, regIntrEn0) == 0)
        dprintf("lw: +         _PRI_TIMEOUT_DISABLED\n");

    if (DRF_VAL(_PBUS, _INTR_EN_0, _FB_REQ_TIMEOUT, regIntrEn0) == 0)
        dprintf("lw: +         _FB_REQ_TIMEOUT_DISABLED\n");

    if (DRF_VAL(_PBUS, _INTR_EN_0, _FB_ACK_TIMEOUT, regIntrEn0) == 0)
        dprintf("lw: +         _FB_ACK_TIMEOUT_DISABLED\n");

    if (DRF_VAL(_PBUS, _INTR_EN_0, _FB_ACK_EXTRA, regIntrEn0) == 0)
        dprintf("lw: +         _FB_ACK_EXTRA_DISABLED\n");

    if (DRF_VAL(_PBUS, _INTR_EN_0, _FB_RDATA_TIMEOUT, regIntrEn0) == 0)
        dprintf("lw: +         _FB_RDATA_TIMEOUT_DISABLED\n");

    if (DRF_VAL(_PBUS, _INTR_EN_0, _FB_RDATA_EXTRA, regIntrEn0) == 0)
        dprintf("lw: +        _FB_RDATA_EXTRA_DISABLED\n");

    if (DRF_VAL(_PBUS, _INTR_EN_0, _SW, regIntrEn0) == 0)
        dprintf("lw: +        _SW_DISABLED\n");

    if (DRF_VAL(_PBUS, _INTR_EN_0, _POSTED_DEADLOCK_TIMEOUT, regIntrEn0) == 0)
        dprintf("lw: +       _POSTED_DEADLOCK_TIMEOUT_DISABLED\n");

    if (DRF_VAL(_PBUS, _INTR_EN_0, _MPMU, regIntrEn0) == 0)
        dprintf("lw: +       _MPMU_DISABLED\n");

    if (DRF_VAL(_PBUS, _INTR_EN_0, _ACCESS_TIMEOUT, regIntrEn0) == 0)
        dprintf("lw: +       _ACCESS_TIMEOUT_DISABLED\n");

    dprintf("lw: + LW_PBUS_INTR_0_\n");

    //interrupts
    if (DRF_VAL(_PBUS, _INTR_0, _PRI_SQUASH, regIntr0) == 1)
    {
        dprintf("lw: +      _PRI_SQUASH_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_PBUS, _INTR_0, _PRI_FECSERR, regIntr0) == 1)
    {
        dprintf("lw: +      _PRI_FECSERR_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_PBUS, _INTR_0, _FB_REQ_TIMEOUT, regIntr0) == 1)
    {
        dprintf("lw: +      _FB_REQ_TIMEOUT_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_PBUS, _INTR_0, _FB_ACK_TIMEOUT, regIntr0) == 1)
    {
        dprintf("lw: +     _FB_ACK_TIMEOUT_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_PBUS, _INTR_0, _FB_ACK_EXTRA, regIntr0) == 1)
    {
        dprintf("lw: +     _FB_ACK_EXTRA_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_PBUS, _INTR_0, _FB_RDATA_TIMEOUT, regIntr0) == 1)
    {
        dprintf("lw: +     _FB_RDATA_TIMEOUT_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_PBUS, _INTR_0, _FB_RDATA_EXTRA, regIntr0) == 1)
    {
        dprintf("lw: +     _FB_RDATA_EXTRA_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_PBUS, _INTR_0, _POSTED_DEADLOCK_TIMEOUT, regIntr0) == 1)
    {
        dprintf("lw: +    _POSTED_DEADLOCK_TIMEOUT_PENDING\n");
        status = LW_ERR_GENERIC;
    }
    
    if (DRF_VAL(_PBUS, _INTR_EN_0, _ACCESS_TIMEOUT, regIntr0) == 1)
    {
        dprintf("lw: +    _ACCESS_TIMEOUT_PENDING\n");
        status = LW_ERR_GENERIC;
    }
     
    //do not report PRI_TIMEOUT as error
    if (DRF_VAL(_PBUS, _INTR_0, _PRI_TIMEOUT, regIntr0))
    {
        dprintf("lw: +     _PRI_TIMEOUT_PENDING\n");
        addUnitErr("\t Warning: LW_PBUS_INTR_0_PRI_TIMEOUT_PENDING\n");

        //print timed out PRI access address
        data32 = GPU_REG_RD32(LW_PTIMER_PRI_TIMEOUT_SAVE_0);
        dprintf("lw: + LW_PTIMER_PRI_TIMEOUT_SAVE_0_ADDR:       0x%08x\n",
                  DRF_VAL(_PTIMER, _PRI_TIMEOUT_SAVE_0, _ADDR, data32));
        dprintf("lw: + LW_PTIMER_PRI_TIMEOUT_SAVE_0_WRITE:      0x%01x\n",
                  DRF_VAL(_PTIMER, _PRI_TIMEOUT_SAVE_0, _WRITE, data32));
        dprintf("lw: + LW_PTIMER_PRI_TIMEOUT_SAVE_0_TO:         0x%01x (1 if there "
            "was a timeout).\n", DRF_VAL(_PTIMER, _PRI_TIMEOUT_SAVE_0, _TO, data32));
        dprintf("lw: + LW_PTIMER_PRI_TIMEOUT_SAVE_1_DATA:       0x%08x\n",
                   GPU_REG_RD32(LW_PTIMER_PRI_TIMEOUT_SAVE_1));
    }


    if (DRF_VAL(_PBUS, _INTR_EN_0, _MPMU, regIntr0) == 1)
    {
        dprintf("lw: +    _MPMU_PENDING\n");
        addUnitErr("\t LW_PBUS_INTR_0_MPMU_PENDING\n");
        //print scratch pad registers

        dprintf("lw:        LW_PBUS_MPMU_INTR_1:     0x%08x\n", GPU_REG_RD32(LW_PBUS_MPMU_INTR_1));
        dprintf("lw:        LW_PBUS_MPMU_INTR_2:     0x%08x\n", GPU_REG_RD32(LW_PBUS_MPMU_INTR_2));
        dprintf("lw:        LW_PBUS_MPMU_INTR_3:     0x%08x\n", GPU_REG_RD32(LW_PBUS_MPMU_INTR_3));
        dprintf("lw:        LW_PBUS_MPMU_INTR_4:     0x%08x\n", GPU_REG_RD32(LW_PBUS_MPMU_INTR_4));
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_PBUS, _INTR_0, _SW  , regIntr0) == 1)
    {
        dprintf("lw: +    _SW_PENDING\n");
        addUnitErr("\t LW_PBUS_INTR_0_SW_PENDING\n");

        //print scratch pad registers
        dprintf("lw: LW_PBUS_SW_INTR_1:     0x%08x\n", GPU_REG_RD32(LW_PBUS_SW_INTR_1));
        dprintf("lw: LW_PBUS_SW_INTR_2:     0x%08x\n", GPU_REG_RD32(LW_PBUS_SW_INTR_2));
        dprintf("lw: LW_PBUS_SW_INTR_3:     0x%08x\n", GPU_REG_RD32(LW_PBUS_SW_INTR_3));
        dprintf("lw: LW_PBUS_SW_INTR_4:     0x%08x\n", GPU_REG_RD32(LW_PBUS_SW_INTR_4));
        status = LW_ERR_GENERIC;
    }

   return status;
}

LW_STATUS busDisableWmBoxes_GK104(LwU32 *pEnabledMask)
{
    LwU32 i, reg;
    LwU32 enabledMask = 0;

    // Disable any already enabled P2P mailbox
    for (i = 0; i < LW_P2P_WMBOX_ADDR__SIZE_1; ++i)
    {
        reg = GPU_REG_RD32(LW_P2P_WMBOX_ADDR(i));

        if (FLD_TEST_DRF(_P2P, _WMBOX_ADDR, _DIS, _ENABLED, reg))
        {
            reg = FLD_SET_DRF(_P2P, _WMBOX_ADDR, _DIS, _DISABLED, reg);
            GPU_REG_WR32(LW_P2P_WMBOX_ADDR(i), reg);
            enabledMask |= BIT(i);
        }
    }
    if (pEnabledMask)
    {
        *pEnabledMask = enabledMask;
    }
    return LW_OK;
}


LW_STATUS busEnableWmBoxes_GK104(LwU32 enableMask)
{
    LwU32 i, reg;

    // Enable the P2P mailboxes specified via 'enableMask' param
    for (i = 0; i < LW_P2P_WMBOX_ADDR__SIZE_1; i++)
    {
        if (enableMask & BIT(i))
        {
            reg = GPU_REG_RD32(LW_P2P_WMBOX_ADDR(i));
            reg = FLD_SET_DRF(_P2P, _WMBOX_ADDR, _DIS, _ENABLED, reg);
            GPU_REG_WR32(LW_P2P_WMBOX_ADDR(i), reg);
        }
    }
    return LW_OK;
}
