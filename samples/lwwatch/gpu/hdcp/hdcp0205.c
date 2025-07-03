/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/****************************** LwWatch ***********************************\
*                                                                          *
*                      HDCP V02_05 routines                                *
*                                                                          *
\***************************************************************************/

#include "inc/disp.h"
#include "disp/v02_05/dev_disp.h"
#include "class/cl907d.h"

#include "g_disp_private.h"

// Dumps all the Info of hdcp22 Status Reg info
void dispHdcp22PrintStatusRegInfo_v02_05
(
    LwU8 sorIndex,
    LwU8 totalLinks
)
{
    LwU32  statusLink1 = 0;
    LwU32  statusLink2 = 0;
    LwBool dualLink = LW_FALSE;

    // Read the HDCP 2.2 status register
    statusLink1 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_STATUS(sorIndex, 0));
    if( totalLinks == 2 )
    {
        statusLink2 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_STATUS(sorIndex, 1));
        dualLink = LW_TRUE;
    }

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_STATUS", "_CRYPT_STATUS", 
                DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _CRYPT_STATUS, statusLink1) ? "ACTIVE" : "INACTIVE");
    if( dualLink )
            dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _CRYPT_STATUS, statusLink2) ? "ACTIVE" : "INACTIVE" );

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_STATUS", "_KRAM_EN",
                DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _KRAM_EN, statusLink1) ? "ENABLED" : "DISABLED");
    if( dualLink )
            dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _KRAM_EN, statusLink2) ? "ENABLED" : "DISABLED");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_STATUS", "_KRAM_ERR",
                DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _KRAM_ERR, statusLink1) ? "ACTIVE" : "INACTIVE");
    if( dualLink )
            dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _KRAM_ERR, statusLink2) ? "ACTIVE" : "INACTIVE");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_STATUS", "_BFM",
                DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _BFM, statusLink1) ? "INACTIVE" : "ACTIVE");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _BFM, statusLink2) ? "INACTIVE" : "ACTIVE");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_STATUS", "_FRAME_CNT_OVERFLOW",
                DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _FRAME_CNT_OVERFLOW, statusLink1) ? "YES" : "NO");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _FRAME_CNT_OVERFLOW, statusLink2) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_STATUS", "_DATA_CNT_OVERFLOW",
                DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _DATA_CNT_OVERFLOW, statusLink1) ? "YES" : "NO");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _DATA_CNT_OVERFLOW, statusLink2) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_STATUS", "_DETACHED_DISABLE",
                DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _DETACHED_DISABLE, statusLink1) ? "YES" : "NO");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _DETACHED_DISABLE, statusLink2) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_STATUS", "_LANE_CNT0_DISABLE",
                DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _LANE_CNT0_DISABLE, statusLink1) ? "YES" : "NO");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _LANE_CNT0_DISABLE, statusLink2) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_STATUS", "_LC128_ERROR",
                DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _LC128_ERROR, statusLink1) ? "YES" : "NO");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _LC128_ERROR, statusLink2) ? "YES" : "NO");

    dprintf("\n%-35s %-35s ", "LW_PDISP_SOR_HDCP22_STATUS", "_AUTODIS_STATE");

    switch (DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _AUTODIS_STATE, statusLink1))
    {
        case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_IDLE:
            dprintf("%-35s","Idle");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_ENCRYPTING:
            dprintf("%-35s","Encrypting");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_DISABLE_LC_0:
            dprintf("%-35s","Disable LC 0");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_DISABLE_DETACH:
            dprintf("%-35s","Disable detach");
            break;
        default:
            dprintf("%-35s","Bad State");
            break;
    }
    if(dualLink)
    {
        switch (DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _AUTODIS_STATE, statusLink2))
        {
            case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_IDLE:
                dprintf("%-35s","Idle");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_ENCRYPTING:
                dprintf("%-35s","Encrypting");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_DISABLE_LC_0:
                dprintf("%-35s","Disable LC 0");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_DISABLE_DETACH:
                dprintf("%-35s","Disable detach");
                break;
            default:
                dprintf("%-35s","Bad State");
                break;
        }
    }

    dprintf("\n%-35s %-35s ", "LW_PDISP_SOR_HDCP22_STATUS", "_HDCP_STATE");

    switch (DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _HDCP_STATE, statusLink1))
    {
        case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_IDLE:
            dprintf("%-35s","Idle");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_LC128:
            dprintf("%-35s","Waiting on LC128");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_AES_READY:
            dprintf("%-35s","Waiting on AES to be ready");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_HDCP22_ENABLE:
            dprintf("%-35s","Waiting on enabling of HDCP");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_HDMI_ENCRYPT_ON:
            dprintf("%-35s","Waiting on encryption for HDMI");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_DP_ENCRYPT_ON:
            dprintf("%-35s","Waiting on encryption for DP");
        default:
            dprintf("%-35s","Bad State");
            break;
    }
    if( dualLink )
    {
        switch (DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _HDCP_STATE, statusLink2))
        {
            case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_IDLE:
                dprintf("%-35s","Idle");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_LC128:
                dprintf("%-35s","Waiting on LC128");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_AES_READY:
                dprintf("%-35s","Waiting on AES to be ready");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_HDCP22_ENABLE:
                dprintf("%-35s","Waiting on enabling of HDCP");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_HDMI_ENCRYPT_ON:
                dprintf("%-35s","Waiting on encryption for HDMI");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_DP_ENCRYPT_ON:
                dprintf("%-35s","Waiting on encryption for DP");
            default:
                dprintf("%-35s","Bad State");
                break;
        }
    }
}

// Dumps all the Info of hdcp22 Ctrl Reg info
void dispHdcp22PrintCtrlRegInfo_v02_05
(
    LwU8 sorIndex,
    LwU8 totalLinks
)
{
    LwU32  statusLink1 = 0;
    LwU32  statusLink2 = 0;
    LwBool dualLink = LW_FALSE;

    // Read the HDCP2.2 Ctrl Regs
    statusLink1 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_CTRL(sorIndex, 0));
    if( totalLinks == 2 )
    {
        statusLink2 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_CTRL(sorIndex, 1));
        dualLink = LW_TRUE;
    }

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_CTRL", "_CRYPT",
                DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _CRYPT, statusLink1) ? "ENABLE" : "DISABLE");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _CRYPT, statusLink2) ? "ENABLE" : "DISABLE");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_CTRL", "_INIT",
                DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _INIT, statusLink1) ? "PENDING/TRIGGER" : "DONE/INIT");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _INIT, statusLink2) ? "PENDING/TRIGGER" : "DONE/INIT");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_CTRL", "_LOCK_ECF",
                DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _LOCK_ECF, statusLink1) ? "LOCKED" : "UNLOCKED");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _LOCK_ECF, statusLink2) ? "LOCKED" : "UNLOCKED");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_CTRL", "_REPEATER",
                DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _REPEATER, statusLink1) ? "YES" : "NO");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _REPEATER, statusLink2) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_CTRL", "_DISABLE_DETACH",
                DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _DISABLE_DETACH, statusLink1) ? "NO" : "YES");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _DISABLE_DETACH, statusLink2) ? "NO" : "YES");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_CTRL", "_DISABLE_LANE_CNT0",
                DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _DISABLE_LANE_CNT0, statusLink1) ? "NO" : "YES");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _DISABLE_LANE_CNT0, statusLink2) ? "NO" : "YES");
}

// Dumps all the Info of hdcp22 debug Reg info 
void dispHdcp22PrintDebugRegInfo_v02_05
(
    LwU8 sorIndex,
    LwU8 totalLinks
)
{
    LwU32  statusLink1 = 0;
    LwU32  statusLink2 = 0;
    LwBool dualLink = LW_FALSE;

    // HDCP2.2 Debug0
    statusLink1 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_DEBUG0(sorIndex, 0));
    if( totalLinks == 2 )
    {
        statusLink2 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_DEBUG0(sorIndex, 1));
        dualLink = LW_TRUE;
    }
    
    dprintf("\n%-35s %-35s %-35d", "LW_PDISP_SOR_HDCP22_DEBUG0", "_DATA_CNTR",
                DRF_VAL(_PDISP, _SOR_HDCP22_DEBUG0, _DATA_CNTR, statusLink1));
    if( dualLink )
        dprintf(" %-35d",DRF_VAL(_PDISP, _SOR_HDCP22_DEBUG0, _DATA_CNTR, statusLink2));

    dprintf("\n%-35s %-35s %-35d", "LW_PDISP_SOR_HDCP22_DEBUG0", "_FRAME_CNTR",
                DRF_VAL(_PDISP, _SOR_HDCP22_DEBUG0, _FRAME_CNTR, statusLink1));
    if( dualLink )
        dprintf(" %-35d",DRF_VAL(_PDISP, _SOR_HDCP22_DEBUG0, _FRAME_CNTR, statusLink2));     
}    


// Dumps all the Info of hdcp22 SST DP type Reg info 
void dispHdcp22PrintSstDpTypeRegInfo_v02_05
(
    LwU8 sorIndex,
    LwU8 totalLinks
)
{
    LwU32  statusLink1 = 0;
    LwU32  statusLink2 = 0;
    LwBool dualLink = LW_FALSE;

    // HDCP2.2 SST_DP_TYPE
    dualLink = LW_FALSE;
    statusLink1 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_SST_DP_TYPE(sorIndex, 0));
    if( totalLinks == 2 )
    {
        statusLink2 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_SST_DP_TYPE(sorIndex, 1));
        dualLink = LW_TRUE;
    }
    
    dprintf("\n%-35s %-35s %-35d", "LW_PDISP_SOR_HDCP22_SST_DP_TYPE", "_VALUE",
                DRF_VAL(_PDISP, _SOR_HDCP22_SST, _DP_TYPE_VALUE, statusLink1));
    if( dualLink )
        dprintf(" %-35d",DRF_VAL(_PDISP, _SOR_HDCP22_SST, _DP_TYPE_VALUE, statusLink2)); 
    
}

// Prints the value of HDCP2.2 DP TYPE Regs
void dispHdcp22PrintDpTypeRegInfo_v02_05
(
    LwU8 sorIndex
)
{
    LwU32 data32;
    // HDCP2.2 DP TYPE
    data32 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_DP_TYPE_MSB(sorIndex));
    dprintf("\n%-35s %-35s %-35d", "LW_PDISP_SOR_HDCP22_DP_TYPE_MSB", "_VALUE",
             DRF_VAL(_PDISP, _SOR_HDCP22_DP, _TYPE_MSB_VALUE, data32));
    
    
    data32 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_DP_TYPE_LSB(sorIndex));
    dprintf("\n%-35s %-35s %-35d", "LW_PDISP_SOR_HDCP22_DP_TYPE_LSB", "_VALUE",
             DRF_VAL(_PDISP, _SOR_HDCP22_DP, _TYPE_LSB_VALUE, data32)); 
}

void dispHdcpPrintSorStatus_v02_05
(
    LwU8 sorIndex
)
{

    LwU32  dpLaneCtrl = 0;
    LwU8   totalLinks = 0;
    LwBool bIsLaneCnt8;

    // Check whether we have a secondary link too.
    dpLaneCtrl = GPU_REG_RD32(LW_PDISP_SOR_DP_LINKCTL0(sorIndex));
    bIsLaneCnt8 = FLD_TEST_DRF(_PDISP, _SOR_DP_LINKCTL0, _LANECOUNT, _EIGHT, dpLaneCtrl);
    if (bIsLaneCnt8)
    {
        totalLinks = 2;
    }
    else
    {
        totalLinks = 1;
    }

    dprintf("HDCP 2.2 Register Info for SOR %d\n",sorIndex);
    dprintf("====================================================================================================\n");

    dprintf("%-35s %-35s %-35s","REGISTER","FIELD","Link1");
    if( totalLinks == 2 )
            dprintf("%-35s","Link2");

    dprintf("\n----------------------------------------------------------------------------------------------------");    

    pDisp[indexGpu].dispHdcp22PrintStatusRegInfo(sorIndex, totalLinks);
    pDisp[indexGpu].dispHdcp22PrintDebugRegInfo(sorIndex, totalLinks);
    pDisp[indexGpu].dispHdcp22PrintCtrlRegInfo(sorIndex, totalLinks);
    pDisp[indexGpu].dispHdcp22PrintSstDpTypeRegInfo(sorIndex, totalLinks);

    dprintf("\n----------------------------------------------------------------------------------------------------");  

    pDisp[indexGpu].dispHdcp22PrintHdmiTypeRegInfo(sorIndex);
    pDisp[indexGpu].dispHdcp22PrintDpTypeRegInfo(sorIndex);
    dprintf("\n====================================================================================================\n\n");
}


LW_STATUS dispPrintHdcp22Status_v02_05
(
    char *numSor
)
{
    LwU8    orNum;
    LwU32   data32;
    LwU32   ownerMask;
    LWOR    orType = LW_OR_SOR;

    LwU8    orNumIteratorStart = (LwU8)(strcmp(numSor, "*") ? 
                                        strtoul(numSor, NULL, 0) : 0);

    LwU8    orNumIteratorEnd = (LwU8)(strcmp(numSor, "*") ? 
                                      strtoul(numSor, NULL, 0) + 1 : pDisp[indexGpu].dispGetNumOrs(orType));


    for (orNum = orNumIteratorStart; orNum < orNumIteratorEnd; orNum++)
    {
        if (pDisp[indexGpu].dispResourceExists(orType, orNum) != TRUE)
        {
            continue;
        }

        // Get SOR SET Control and decide if any head is attached to it
        data32 = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(0) + 
                              LW907D_SOR_SET_CONTROL(orNum));
        ownerMask = DRF_VAL(907D, _SOR_SET_CONTROL, _OWNER_MASK, data32);
        if (!ownerMask)
        {
            dprintf("SOR %d is not attached to any head\n\n", orNum);
            continue;
        }
        pDisp[indexGpu].dispHdcpPrintSorStatus(orNum);
    }

    return LW_OK;
}

