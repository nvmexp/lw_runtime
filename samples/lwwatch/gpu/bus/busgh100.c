/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


//*****************************************************
//
// lwwatch extension
// busgh100.c
//
//*****************************************************

//
// includes
//
#include "hopper/gh100/pri_lw_xal_ep.h"
#include "hopper/gh100/pri_lw_xal_ep_p2p.h"
#include "bus.h"
#include "gpuanalyze.h"
#include "g_bus_private.h"     // (rmconfig)  implementation prototypes

/*!
 *  Test top level interrupt tree
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS busTestBusInterrupts_GH100( void )
{
    LW_STATUS status = LW_OK;
    LwU32     regIntr0 = 0;
    LwU32     regIntrEn0 = 0;

    regIntrEn0 = GPU_REG_RD32(LW_XAL_EP_INTR_EN_0);
    regIntr0 = GPU_REG_RD32(LW_XAL_EP_INTR_0) & regIntrEn0;

    dprintf("lw: + LW_XAL_EP_INTR_EN_0_\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _JTAG_TIMEOUT, regIntrEn0) == 0)
        dprintf("lw: +         _JTAG_TIMEOUT_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _PRI_FECSERR, regIntrEn0) == 0)
        dprintf("lw: +         _PRI_FECSERR_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _PRI_REQ_TIMEOUT, regIntrEn0) == 0)
        dprintf("lw: +         _PRI_REQ_TIMEOUT_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _PRI_RSP_TIMEOUT, regIntrEn0) == 0)
        dprintf("lw: +         _PRI_RSP_TIMEOUT_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_REQ_TIMEOUT, regIntrEn0) == 0)
        dprintf("lw: +         _FB_REQ_TIMEOUT_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_ACK_TIMEOUT, regIntrEn0) == 0)
        dprintf("lw: +         _FB_ACK_TIMEOUT_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_RDATA_TIMEOUT, regIntrEn0) == 0)
        dprintf("lw: +         _FB_RDATA_TIMEOUT_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_REQ_P_TIMEOUT_NOTIFICATION, regIntrEn0) == 0)
        dprintf("lw: +        _FB_REQ_P_TIMEOUT_NOTIFICATION_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_REQ_NP_TIMEOUT_NOTIFICATION, regIntrEn0) == 0)
        dprintf("lw: +        _FB_REQ_NP_TIMEOUT_NOTIFICATION_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_REQ_CPL_TIMEOUT_NOTIFICATION, regIntrEn0) == 0)
        dprintf("lw: +       _FB_REQ_CPL_TIMEOUT_NOTIFICATION_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_ACK_TIMEOUT_NOTIFICATION, regIntrEn0) == 0)
        dprintf("lw: +       _FB_ACK_TIMEOUT_NOTIFICATION_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_RDATA_TIMEOUT_NOTIFICATION, regIntrEn0) == 0)
        dprintf("lw: +       _FB_RDATA_TIMEOUT_NOTIFICATION_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _PRI_REQ_TIMEOUT_NOTIFICATION, regIntrEn0) == 0)
        dprintf("lw: +       _PRI_REQ_TIMEOUT_NOTIFICATION_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _PRI_RSP_TIMEOUT_NOTIFICATION, regIntrEn0) == 0)
        dprintf("lw: +       _PRI_RSP_TIMEOUT_NOTIFICATION_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_ILLEGAL_OP, regIntrEn0) == 0)
        dprintf("lw: +       _FB_ILLEGAL_OP_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _P2P_FAULT, regIntrEn0) == 0)
        dprintf("lw: +       _P2P_FAULT_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _P2P_READ_TIMEOUT_NOTIFICATION, regIntrEn0) == 0)
        dprintf("lw: +       _P2P_READ_TIMEOUT_NOTIFICATION_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _MEMOP_RSP_TIMEOUT_NOTIFICATION, regIntrEn0) == 0)
        dprintf("lw: +       _MEMOP_RSP_TIMEOUT_NOTIFICATION_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _INGRESS_POISON, regIntrEn0) == 0)
        dprintf("lw: +       _INGRESS_POISON_DISABLED\n");    

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _EGRESS_POISON, regIntrEn0) == 0)
        dprintf("lw: +       _EGRESS_POISON_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _ECC_CORRECTABLE, regIntrEn0) == 0)
        dprintf("lw: +       _ECC_CORRECTABLE_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _ECC_UNCORRECTABLE, regIntrEn0) == 0)
        dprintf("lw: +       _ECC_UNCORRECTABLE_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _DECOUPLER_ERROR, regIntrEn0) == 0)
        dprintf("lw: +       _DECOUPLER_ERROR_DISABLED\n");

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _INTR_DEBUG, regIntrEn0) == 0)
        dprintf("lw: +       _INTR_DEBUG_DISABLED\n");

    dprintf("lw: + LW_XAL_EP_INTR_0_\n");

    if (DRF_VAL(_XAL_EP, _INTR_0, _JTAG_TIMEOUT, regIntr0) == 1)
    {
        dprintf("lw: +      _JTAG_TIMEOUT_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_0, _PRI_FECSERR, regIntr0) == 1)
    {
        dprintf("lw: +      _PRI_FECSERR_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_0, _PRI_REQ_TIMEOUT, regIntr0) == 1)
    {
        dprintf("lw: +      _PRI_REQ_TIMEOUT_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_0, _FB_ACK_TIMEOUT, regIntr0) == 1)
    {
        dprintf("lw: +     _FB_ACK_TIMEOUT_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_0, _FB_RDATA_TIMEOUT, regIntr0) == 1)
    {
        dprintf("lw: +     _FB_RDATA_TIMEOUT_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_0, _FB_REQ_P_TIMEOUT_NOTIFICATION, regIntr0) == 1)
    {
        dprintf("lw: +    _FB_REQ_P_TIMEOUT_NOTIFICATION_PENDING\n");
        status = LW_ERR_GENERIC;
    }
    
    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_REQ_NP_TIMEOUT_NOTIFICATION, regIntr0) == 1)
    {
        dprintf("lw: +    _FB_REQ_NP_TIMEOUT_NOTIFICATION_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_REQ_CPL_TIMEOUT_NOTIFICATION, regIntr0) == 1)
    {
        dprintf("lw: +    _FB_REQ_CPL_TIMEOUT_NOTIFICATION_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_ACK_TIMEOUT_NOTIFICATION, regIntr0) == 1)
    {
        dprintf("lw: +    _FB_ACK_TIMEOUT_NOTIFICATION_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_RDATA_TIMEOUT_NOTIFICATION, regIntr0) == 1)
    {
        dprintf("lw: +    _FB_RDATA_TIMEOUT_NOTIFICATION_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _PRI_REQ_TIMEOUT_NOTIFICATION, regIntr0) == 1)
    {
        dprintf("lw: +    _PRI_REQ_TIMEOUT_NOTIFICATION_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _PRI_RSP_TIMEOUT_NOTIFICATION, regIntr0) == 1)
    {
        dprintf("lw: +    _PRI_RSP_TIMEOUT_NOTIFICATION_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _FB_ILLEGAL_OP, regIntr0) == 1)
    {
        dprintf("lw: +    _FB_ILLEGAL_OP_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _P2P_FAULT, regIntr0) == 1)
    {
        dprintf("lw: +    _P2P_FAULT_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _P2P_READ_TIMEOUT_NOTIFICATION, regIntr0) == 1)
    {
        dprintf("lw: +    _P2P_READ_TIMEOUT_NOTIFICATION_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _MEMOP_RSP_TIMEOUT_NOTIFICATION, regIntr0) == 1)
    {
        dprintf("lw: +    _MEMOP_RSP_TIMEOUT_NOTIFICATION_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _INGRESS_POISON, regIntr0) == 1)
    {
        dprintf("lw: +    _INGRESS_POISON_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _EGRESS_POISON, regIntr0) == 1)
    {
        dprintf("lw: +    _EGRESS_POISON_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _ERR_DATA, regIntr0) == 1)
    {
        dprintf("lw: +    _ERR_DATA_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _ECC_CORRECTABLE, regIntr0) == 1)
    {
        dprintf("lw: +    _ECC_CORRECTABLE_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _ECC_UNCORRECTABLE, regIntr0) == 1)
    {
        dprintf("lw: +    _ECC_UNCORRECTABLE_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _DECOUPLER_ERROR, regIntr0) == 1)
    {
        dprintf("lw: +    _DECOUPLER_ERROR_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_XAL_EP, _INTR_EN_0, _INTR_DEBUG, regIntr0) == 1)
    {
        dprintf("lw: +    _INTR_DEBUG_PENDING\n");
        status = LW_ERR_GENERIC;
    }

    //do not report PRI_TIMEOUT as error
    if (DRF_VAL(_XAL_EP, _INTR_0, _PRI_REQ_TIMEOUT, regIntr0))
    {
        LwU32 regTimeout;
        dprintf("lw: +     _PRI_REQ_TIMEOUT\n");
        addUnitErr("\t Warning: LW_XAL_INTR_0_PRI_REQ_TIMEOUT_PENDING\n");

        regTimeout = GPU_REG_RD32(LW_XAL_EP_INTR_TRIGGERED_PRI_REQ_TIMEOUT);

        // print timed out PRI access address
        dprintf("lw: + LW_XAL_EP_INTR_TRIGGERED_PRI_REQ_TIMEOUT_WRITE:   0x%08x\n",
                DRF_VAL(_XAL_EP, _INTR_TRIGGERED_PRI_REQ_TIMEOUT, _WRITE, regTimeout));
        dprintf("lw: + LW_XAL_EP_INTR_TRIGGERED_PRI_REQ_TIMEOUT_ADR:     0x%08x\n",
                (DRF_VAL(_XAL_EP, _INTR_TRIGGERED_PRI_REQ_TIMEOUT, _ADR, regTimeout) << 2));
        dprintf("lw: + LW_XAL_EP_INTR_TRIGGERED_PRI_REQ_TIMEOUT_OVERFLOW:0x%08x\n",
                DRF_VAL(_XAL_EP, _INTR_TRIGGERED_PRI_REQ_TIMEOUT, _OVERFLOW, regTimeout));
    }

    if (DRF_VAL(_XAL_EP, _INTR_0, _PRI_RSP_TIMEOUT, regIntr0))
    {
        LwU32 regTimeout;
        dprintf("lw: +     _PRI_RSP_TIMEOUT\n");
        addUnitErr("\t Warning: LW_XAL_INTR_0_PRI_RSP_TIMEOUT_PENDING\n");

        regTimeout = GPU_REG_RD32(LW_XAL_EP_INTR_TRIGGERED_PRI_RSP_TIMEOUT);

        // print timed out PRI access address
        dprintf("lw: + LW_XAL_EP_INTR_TRIGGERED_PRI_RSP_TIMEOUT_WRITE:   0x%08x\n",
                DRF_VAL(_XAL_EP, _INTR_TRIGGERED_PRI_RSP_TIMEOUT, _WRITE, regTimeout));
        dprintf("lw: + LW_XAL_EP_INTR_TRIGGERED_PRI_RSP_TIMEOUT_ADR:     0x%08x\n",
                (DRF_VAL(_XAL_EP, _INTR_TRIGGERED_PRI_RSP_TIMEOUT, _ADR, regTimeout) << 2));
        dprintf("lw: + LW_XAL_EP_INTR_TRIGGERED_PRI_RSP_TIMEOUT_OVERFLOW:0x%08x\n",        
                DRF_VAL(_XAL_EP, _INTR_TRIGGERED_PRI_RSP_TIMEOUT, _OVERFLOW, regTimeout));
    }

    return status;
}

LW_STATUS busDisableWmBoxes_GH100(LwU32 *pEnabledMask)
{
    LwU32 i, reg;
    LwU32 enabledMask = 0;

    // Disable any already enabled P2P mailbox
    for (i = 0; i < LW_XAL_EP_P2P_WMBOX_ADDR__SIZE_1; ++i)
    {
        reg = GPU_REG_RD32(LW_XAL_EP_P2P_WMBOX_ADDR(i));

        if (FLD_TEST_DRF(_XAL_EP_P2P, _WMBOX_ADDR, _DIS, _ENABLED, reg))
        {
            reg = FLD_SET_DRF(_XAL_EP_P2P, _WMBOX_ADDR, _DIS, _DISABLED, reg);
            GPU_REG_WR32(LW_XAL_EP_P2P_WMBOX_ADDR(i), reg);
            enabledMask |= BIT(i);
        }
    }
    if (pEnabledMask)
    {
        *pEnabledMask = enabledMask;
    }

    return LW_OK;
}

LW_STATUS busEnableWmBoxes_GH100(LwU32 enableMask)
{
    LwU32 i, reg;

    // Enable the P2P mailboxes specified via 'enableMask' param
    for (i = 0; i < LW_XAL_EP_P2P_WMBOX_ADDR__SIZE_1; ++i)
    {
        if (enableMask & BIT(i))
        {
            reg = GPU_REG_RD32(LW_XAL_EP_P2P_WMBOX_ADDR(i));
            reg = FLD_SET_DRF(_XAL_EP_P2P, _WMBOX_ADDR, _DIS, _ENABLED, reg);
            GPU_REG_WR32(LW_XAL_EP_P2P_WMBOX_ADDR(i), reg);
        }
    }

    return LW_OK;
}
