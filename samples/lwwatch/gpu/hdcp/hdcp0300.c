/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/****************************** LwWatch ***********************************\
*                                                                          *
*                      HDCP V03_00 routines                                *
*                                                                          *
\***************************************************************************/

#include "inc/disp.h"
#include "disp/v03_00/dev_disp.h"
#include "class/clc37d.h"

#include "g_disp_private.h"


// Prints the value of HDCP2.2 HDMI TYPE Regs
void dispHdcp22PrintHdmiTypeRegInfo_v03_00
(
    LwU8 sorIndex
)
{
    LwU32 data32;
    // HDCP2.2 HDMI TYPE
    data32 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_HDMI_TYPE(sorIndex));
    dprintf("\n%-35s %-35s %-35d", "LW_PDISP_SOR_HDCP22_HDMI_TYPE", "_VALUE",
             DRF_VAL(_PDISP, _SOR_HDCP22_HDMI, _TYPE_VALUE, data32));
}

void dispHdcpGetOrOwner_v03_00
(
    LWOR    orType,
    LwU32   orNum,
    LwU32   *pOwnerMask,
    LwU32   *pData32
)
{
    LwU32   ownerMask;
    LwU32   data32;

    switch (orType)
    {
        case LW_OR_SOR:
            data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE +  
                              LWC37D_SOR_SET_CONTROL(orNum));
            ownerMask = DRF_VAL(C37D, _SOR_SET_CONTROL, _OWNER_MASK, data32);
            break;

        default:
            data32 = 0;
            ownerMask = 0;
            break;
    }
    
    *pOwnerMask = ownerMask;
    *pData32 = data32;
}

void dispHdcpGetOrProtocol_v03_00
(
    LWOR    orType,
    LwU32   data32,
    ORPROTOCOL  *pOrProtocol
)
{
    switch (orType)
    {
        case LW_OR_SOR:
            *pOrProtocol = pDisp[indexGpu].dispGetOrProtocol(LW_OR_SOR, 
                            DRF_VAL(C37D, _SOR_SET_CONTROL, _PROTOCOL, data32));
        break;

        default:
            *pOrProtocol = 0;
            break;
    }
  
}

LW_STATUS dispHdcpPrintStatus_v03_00
(
)
{
    dprintf("====================================================================================================\n");
    dprintf("OR#     OWNER       PROTOCOL         INT/EXT     Protected    Encryption   Repeater\n");
    dprintf("----------------------------------------------------------------------------------------------------\n");

    pDisp[indexGpu].dispHdcpPrintOrStatus(LW_OR_SOR, "SOR");

    dprintf("====================================================================================================\n\n");

    return LW_OK;
}

LW_STATUS dispPrintHdcp22Status_v03_00
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
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE +  LWC37D_SOR_SET_CONTROL(orNum));
        ownerMask = DRF_VAL(C37D, _SOR_SET_CONTROL, _OWNER_MASK, data32);
        if (!ownerMask)
        {
            dprintf("SOR %d is not attached to any head\n\n", orNum);
            continue;
        }

        pDisp[indexGpu].dispHdcpPrintSorStatus(orNum);
    }

    return LW_OK;
}