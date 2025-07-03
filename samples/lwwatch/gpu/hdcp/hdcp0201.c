/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/****************************** LwWatch ***********************************\
*                                                                          *
*                      HDCP V02_01 routines                                *
*                                                                          *
\***************************************************************************/

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "inc/disp.h"
#include "disp/v02_01/dev_disp.h"
#include "class/cl907d.h"

#include "g_disp_private.h"

#define HDCP_SPRIME_SIZE        9

#define HDCP_IS_PROTECTED(x)    (!DRF_VAL(_PDISP_UPSTREAM, _HDCP, \
                                    _SPRIME_LSB2_STATUS_UNPROTECTED, (x)))
#define HDCP_IS_EXT_DEVICE(x)   DRF_VAL(_PDISP_UPSTREAM, _HDCP, \
                                    _SPRIME_LSB2_STATUS_EXTPNL, (x))
#define HDCP_IS_ENCRYPTING(x)   DRF_VAL(_PDISP_UPSTREAM, _HDCP, \
                                    _SPRIME_LSB2_STATUS_ENCRYPTING, (x))
#define HDCP_IS_REPEATER(x)     DRF_VAL(_PDISP_UPSTREAM, _HDCP, \
                                    _SPRIME_LSB2_STATUS_RPTR, (x))



LW_STATUS dispHdcpReadUpstreamSPrimeValid_v02_01
(
    LwU32   Head,
    LwU32   orIndex,
    LwU32   apIndex

)
{
    LwU32   data        = 0;

    //forge session ID, and key vector
    LwU32   Cn0 = 1;
    LwU32   Cn1 = 0;
    LwU8    Cksv[8] = {0, 0, 0xf0, 0xff, 0xff, 0,  };

    // update the mode read status
    data = GPU_REG_RD32(LW_PDISP_UPSTREAM_HDCP_CMODE);
    data = FLD_SET_DRF(_PDISP_UPSTREAM, _HDCP_CMODE, _MODE, _READ_S, data);

    //
    // Zero passed down represents an invalid apIndex and therefore, we
    // shall use the last apIndex last specified.  THere are cases 
    // where we don't care what apIndex is so zero representative of this.
    //
    if (apIndex)
    {
        data = FLD_SET_DRF_NUM(_PDISP_UPSTREAM, _HDCP, _CMODE_INDEX, apIndex,
                               data);
    }

    data = FLD_SET_DRF_NUM(_PDISP_UPSTREAM, _HDCP, _CMODE_HEAD_INDEX, Head, 
                           data);
    GPU_REG_WR32(LW_PDISP_UPSTREAM_HDCP_CMODE, data);

    // Wait 1000us for CMODE_INDEX and CMODE_HEAD_INDEX to get latched.
    osPerfDelay(1000);

    data = GPU_REG_RD32(LW_PDISP_UPSTREAM_HDCP_CTRL);
  

    GPU_REG_WR32(LW_PDISP_UPSTREAM_HDCP_CN_LSB, Cn0);
    GPU_REG_WR32(LW_PDISP_UPSTREAM_HDCP_CN_MSB, Cn1);

    
    GPU_REG_WR32(LW_PDISP_UPSTREAM_HDCP_CKSV_LSB, *(LwU32*)Cksv);
    GPU_REG_WR32(LW_PDISP_UPSTREAM_HDCP_CKSV_MSB, Cksv[4]);

    // Wait 1000us before reading sprime valid - give hardware time.
    osPerfDelay(1000);

    data = GPU_REG_RD32(LW_PDISP_UPSTREAM_HDCP_CTRL);
    if (!FLD_TEST_DRF(_PDISP, _UPSTREAM_HDCP_CTRL, _SPRIME, _VALID, data))
    {
        dprintf("%s: time out\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    return LW_OK;

}


LW_STATUS dispHdcpReadUpstreamSPrimeRegs_v02_01
(
    LwU8    *pSPrime
)
{
    LwU32   data32 = 0;
    LwU32   *pData = NULL;

    if (pSPrime == NULL)
    {
        dprintf("%s: SPrime is null\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }    

    data32 = GPU_REG_RD32(LW_PDISP_UPSTREAM_HDCP_CTRL);
    if (FLD_TEST_DRF(_PDISP, _UPSTREAM_HDCP_CTRL, _SPRIME, _VALID, data32))
    {
        // Read SPrime
        pData = (LwU32 *)pSPrime;
        *pData = GPU_REG_RD32(LW_PDISP_UPSTREAM_HDCP_SPRIME_LSB1);
        pData = (LwU32 *)&pSPrime[4];
        *pData = GPU_REG_RD32(LW_PDISP_UPSTREAM_HDCP_SPRIME_LSB2);
        data32 = GPU_REG_RD32(LW_PDISP_UPSTREAM_HDCP_SPRIME_MSB);
        pSPrime[8] = (LwU8)(data32 & 0xFF);
    }    
    else
    {
        dprintf("%s: SPrime is invalid\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}


LwU32 dispHdcpGetApIndex
(
    LwU32   orNum,
    LWOR    orType
)
{
    LwU32   idx;

    switch (orType)
    {
        case LW_OR_SOR:
            idx = LW_PDISP_UPSTREAM_HDCP_CMODE_INDEX_SOR(orNum);
            break;
    
        case LW_OR_DAC:
            idx = LW_PDISP_UPSTREAM_HDCP_CMODE_INDEX_DAC(orNum);
            break;
    
        case LW_OR_PIOR:
            idx = LW_PDISP_UPSTREAM_HDCP_CMODE_INDEX_PIOR(orNum);
            break;
    
        default:
            idx = 0;
            break;
    }

    return idx;
    
}
 
LW_STATUS dispHdcpReadUpstreamSPrime_v02_01
(
    LwU32   orNum,
    LwU32   head,
    LWOR    orType,
    LwU8    *SPrime

)
{
    LwU32   apIndex;

    if (!SPrime)
    {
        dprintf("%s: SPrime was null\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    apIndex = dispHdcpGetApIndex(orNum, orType);

    if (LW_ERR_GENERIC == pDisp[indexGpu].dispHdcpReadUpstreamSPrimeValid(head, 
                                                                    orNum, 
                                                                    apIndex))
    {
        return LW_ERR_GENERIC;
    } 

    
    if (LW_ERR_GENERIC == pDisp[indexGpu].dispHdcpReadUpstreamSPrimeRegs(SPrime))
    {
        return LW_ERR_GENERIC;
    }
    
    return LW_OK;

}

//print HDCP information by interpreting SPrime register
void dispHdcpPrintSPrimeStatus
(
    LwU8    *pSPrime,
    LWOR    orType
)
{
    LwU32   *pSprime32;

    //get SPrime_LSB2, to interpret SPrime register
    pSprime32 = (LwU32 *)&pSPrime[4];

    if(orType != LW_OR_DAC)
    {
        if (HDCP_IS_EXT_DEVICE(*pSprime32))
        {
            dprintf("%-12s", "External");
        }
        else
        {
            dprintf("%-12s", "Internal");
        }
    }
    else
    {
        dprintf("%-12s", "N/A");
    }

    if (HDCP_IS_PROTECTED(*pSprime32))
    {
        dprintf("%-13s", "Yes");
    }
    else
    {
        dprintf("%-13s", "No");
    }

    if (HDCP_IS_ENCRYPTING(*pSprime32))
    {
        dprintf("%-13s", "Enabled");
    }
    else
    {
        dprintf("%-13s", "Disabled");
    }

    if (HDCP_IS_REPEATER(*pSprime32))
    {
        dprintf("%-11s", "Yes");
    }
    else
    {
        dprintf("%-11s", "No");
    }
    dprintf("\n");
}

void dispHdcpGetOrOwner_v02_01
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
        case LW_OR_DAC:
            data32 = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(0) + 
                              LW907D_DAC_SET_CONTROL(orNum));
            ownerMask = DRF_VAL(907D, _DAC_SET_CONTROL, _OWNER_MASK, data32);
            break;
    
        case LW_OR_SOR:
            data32 = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(0) + 
                              LW907D_SOR_SET_CONTROL(orNum));
            ownerMask = DRF_VAL(907D, _SOR_SET_CONTROL, _OWNER_MASK, data32);
            break;
    
        case LW_OR_PIOR:
            data32 = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(0) + 
                              LW907D_PIOR_SET_CONTROL(orNum));
            ownerMask = DRF_VAL(907D, _PIOR_SET_CONTROL, _OWNER_MASK, data32);
            break;
    
        default:
            data32 = 0;
            ownerMask = 0;
            break;
    }
    
    *pOwnerMask = ownerMask;
    *pData32 = data32;

    
}

void dispHdcpGetOrProtocol_v02_01
(
    LWOR    orType,
    LwU32   data32,
    ORPROTOCOL  *pOrProtocol
)
{
    switch (orType)
    {
        case LW_OR_DAC:
            *pOrProtocol = pDisp[indexGpu].dispGetOrProtocol(LW_OR_DAC, 
                            DRF_VAL(907D, _DAC_SET_CONTROL, _PROTOCOL, data32));
        break;
    
        case LW_OR_SOR:
            *pOrProtocol = pDisp[indexGpu].dispGetOrProtocol(LW_OR_SOR, 
                            DRF_VAL(907D, _SOR_SET_CONTROL, _PROTOCOL, data32));
        break;
    
        case LW_OR_PIOR:
            *pOrProtocol = pDisp[indexGpu].dispGetOrProtocol(LW_OR_PIOR, 
                           DRF_VAL(907D, _PIOR_SET_CONTROL, _PROTOCOL, data32));
           break;
    
        default:
            *pOrProtocol = 0;
            break;
    }
  
}

void dispHdcpPrintOrStatus_v02_01
(
    LWOR    orType,
    char    *pOrString
)
{
    LwU32   orNum;
    LwU32   head;
    LwU32   ownerMask;
    LwU32   data32;
    ORPROTOCOL  orProtocol;
    LwU8    SPrime[HDCP_SPRIME_SIZE];
    
    for (orNum = 0; orNum < pDisp[indexGpu].dispGetNumOrs(orType); orNum++)
    {

        if (pDisp[indexGpu].dispResourceExists(orType, orNum) != TRUE)
        {
            continue;
        }
        
        pDisp[indexGpu].dispHdcpGetOrOwner(orType, orNum, &ownerMask, &data32);
        if (!ownerMask)
        {
            dprintf("%s%d    NONE        N/A\n", pOrString, orNum);
            continue;
        }
     
        pDisp[indexGpu].dispHdcpGetOrProtocol(orType, data32, &orProtocol);

        for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
        {
            if (BIT(head) & ownerMask)
            {
                dprintf("%s%u    HEAD%d       %-17s", pOrString, orNum, head, 
                         dispGetStringForOrProtocol(orType, orProtocol));
        
                //get hdcp status
                if (LW_ERR_GENERIC == pDisp[indexGpu].dispHdcpReadUpstreamSPrime(orNum, 
                                 head, orType, SPrime))
                {
                    dprintf("\n");
                    continue;
                } 
                dispHdcpPrintSPrimeStatus(SPrime, orType);
            }
        }  
    }

}

LW_STATUS dispHdcpPrintStatus_v02_01
(
)
{
    dprintf("====================================================================================================\n");
    dprintf("OR#     OWNER       PROTOCOL         INT/EXT     Protected    Encryption   Repeater \n");
    dprintf("----------------------------------------------------------------------------------------------------\n");


    pDisp[indexGpu].dispHdcpPrintOrStatus(LW_OR_DAC, "DAC");
    pDisp[indexGpu].dispHdcpPrintOrStatus(LW_OR_SOR, "SOR");
    pDisp[indexGpu].dispHdcpPrintOrStatus(LW_OR_PIOR, "PIOR");

    dprintf("====================================================================================================\n\n");

    return LW_OK;
}
