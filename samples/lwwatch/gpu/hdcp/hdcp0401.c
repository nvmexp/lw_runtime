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
*                      HDCP V04_01 routines                                *
*                                                                          *
\***************************************************************************/

#include "inc/disp.h"
#include "class/clc37d.h"
#include "disp/v04_01/dev_disp.h"
#include "ampere/ga102/dev_fuse.h"

#include "g_disp_private.h"

#define MAX_TIMEOUT_MS 4

// checks the hdcp keys are loaded or not
static LwBool _dispHdcpKeysLoaded
(
)
{
    LwU32 data32 = GPU_REG_RD32(LW_PDISP_HDCPRIF_STATUS);
    return FLD_TEST_DRF(_PDISP, _HDCPRIF_STATUS, _KEY_READY, _YES, data32);
}

// returns the number of unset bits
static LwU32 _getUnsetBits
(
    LwU32 num,
    LwU32 numBits
)
{
    LwU32 iter;
    LwU32 zeroCount = 0;

    for(iter = 0; iter < numBits; iter++ )
    {
        zeroCount = ((num>>iter) & (0x1)) ? zeroCount : zeroCount + 1;
    }

    return zeroCount;
}

static LW_STATUS _dispHdcpUpstreamStatus()
{
    LwBool bStatus;
    LwU32  data32;
    LwU32  aksvMsbVal;
    LwU32  aksvLsbVal;
    LwU32  unsetBits;

    data32 = GPU_REG_RD32(LW_PDISP_UPSTREAM_HDCP_AKSV_MSB);  
    aksvMsbVal = DRF_VAL(_PDISP, _UPSTREAM_HDCP_AKSV_MSB, _VALUE, data32);

    data32 = GPU_REG_RD32(LW_PDISP_UPSTREAM_HDCP_AKSV_LSB);
    aksvLsbVal = DRF_VAL(_PDISP, _UPSTREAM_HDCP_AKSV_LSB, _VALUE, data32);
    
    unsetBits = _getUnsetBits(aksvMsbVal, 8) + _getUnsetBits(aksvLsbVal, 32);
    
    // checking for equal number of set and unset bits in a 40 bit val
    bStatus = (unsetBits == 20) ? LW_OK : LW_ERR_GENERIC; 
    return bStatus;
}

LW_STATUS dispFuseCrcStatus_v04_01
(
)
{
    LwU32 data32 = GPU_REG_RD32(LW_FUSE_CRC_STATUS);
    
    if (!((FLD_TEST_DRF(_FUSE_CRC, _STATUS, _H2, _CRC_PRESENT_NO_ERRORS, data32)) ||
        (FLD_TEST_DRF(_FUSE_CRC, _STATUS, _H2, _CRC_PRESENT_CORRECTED_ERRORS, data32))))
    {
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

// Dumps all the Info of hdcp22 Status Reg info
void dispHdcp22PrintStatusRegInfo_v04_01
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
            dprintf("%-35s", "Idle");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_ENCRYPTING:
            dprintf("%-35s", "Encrypting");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_DISABLE_LC_0:
            dprintf("%-35s", "Disable LC 0");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_DISABLE_DETACH:
            dprintf("%-35s", "Disable detach");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_DISABLE_HDMI_FRL_DIS:
            dprintf("%-35s", "Disable Hdmi Frl DIS");
            break;
        default:
            dprintf("%-35s", "Bad State");
            break;
    }
    if(dualLink)
    {
        switch (DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _AUTODIS_STATE, statusLink2))
        {
            case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_IDLE:
                dprintf("%-35s", "Idle");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_ENCRYPTING:
                dprintf("%-35s", "Encrypting");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_DISABLE_LC_0:
                dprintf("%-35s", "Disable LC 0");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_DISABLE_DETACH:
                dprintf("%-35s", "Disable detach");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_AUTODIS_STATE_DISABLE_HDMI_FRL_DIS:
                dprintf("%-35s", "Disable Hdmi Frl DIS");
                break;                
            default:
                dprintf("%-35s", "Bad State");
                break;
        }
    }

    dprintf("\n%-35s %-35s ", "LW_PDISP_SOR_HDCP22_STATUS", "_HDCP_STATE");

    switch (DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _HDCP_STATE, statusLink1))
    {
        case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_IDLE:
            dprintf("%-35s", "Idle");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_LC128:
            dprintf("%-35s", "Waiting on LC128");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_AES_READY:
            dprintf("%-35s", "Waiting on AES to be ready");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_HDCP22_ENABLE:
            dprintf("%-35s", "Waiting on enabling of HDCP");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_HDMI_ENCRYPT_ON:
            dprintf("%-35s", "Waiting on encryption for HDMI");
            break;
        case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_DP_ENCRYPT_ON:
            dprintf("%-35s", "Waiting on encryption for DP");
        default:
            dprintf("%-35s", "Bad State");
            break;
    }
    if( dualLink )
    {
        switch (DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _HDCP_STATE, statusLink2))
        {
            case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_IDLE:
                dprintf("%-35s", "Idle");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_LC128:
                dprintf("%-35s", "Waiting on LC128");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_AES_READY:
                dprintf("%-35s", "Waiting on AES to be ready");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_HDCP22_ENABLE:
                dprintf("%-35s", "Waiting on enabling of HDCP");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_HDMI_ENCRYPT_ON:
                dprintf("%-35s", "Waiting on encryption for HDMI");
                break;
            case LW_PDISP_SOR_HDCP22_STATUS_HDCP_STATE_WAIT_DP_ENCRYPT_ON:
                dprintf("%-35s", "Waiting on encryption for DP");
            default:
                dprintf("%-35s", "Bad State");
                break;
        }
    }

    // Read the HDCP 2.2 status register
    statusLink1 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_STATUS(sorIndex, 0));
    if( totalLinks == 2 )
    {
        statusLink2 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_STATUS(sorIndex, 1));
        dualLink = LW_TRUE;
    }

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_STATUS", "_HDMI_FRL_DIS_DISABLE", 
                DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _HDMI_FRL_DIS_DISABLE, statusLink1) ? "YES" : "NO");
    if( dualLink )
            dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_STATUS, _HDMI_FRL_DIS_DISABLE, statusLink2) ? "YES" : "NO" );
}

// Dumps all the Info of hdcp22 Ctrl Reg info
void dispHdcp22PrintCtrlRegInfo_v04_01
(
    LwU8 sorIndex,
    LwU8 totalLinks
)
{
    LwU32  statusLink1 = 0;
    LwU32  statusLink2 = 0;
    LwBool dualLink = LW_FALSE;

    dispHdcp22PrintCtrlRegInfo_v02_05(sorIndex, totalLinks);

    // Read the HDCP 2.2 Ctrl Regs
    statusLink1 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_CTRL(sorIndex, 0));
    if( totalLinks == 2 )
    {
        statusLink2 = GPU_REG_RD32(LW_PDISP_SOR_HDCP22_CTRL(sorIndex, 1));
        dualLink = LW_TRUE;
    }

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_SOR_HDCP22_CTRL", "_DISABLE_HDMI_FRL_DIS",
                DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _DISABLE_HDMI_FRL_DIS, statusLink1) ? "NO" : "YES");
    if( dualLink )
        dprintf(" %-35s",DRF_VAL(_PDISP, _SOR_HDCP22_CTRL, _DISABLE_HDMI_FRL_DIS, statusLink2) ? "NO" : "YES");  
}

void dispPrintFuseCrcStatus_v04_01
(
)
{
    LwU32 data32 = GPU_REG_RD32(LW_FUSE_CRC_STATUS);
    dprintf("\n\n%-35s %-35s", "LW_FUSE_CRC_STATUS", "_H2");
    switch (DRF_VAL(_FUSE, _CRC_STATUS,  _H2, data32))
    {
        case LW_FUSE_CRC_STATUS_H2_CRC_NOT_PARSED_YET:
            dprintf("%-35s", "CRC_NOT_PARSED_YET");
            break;
    
        case LW_FUSE_CRC_STATUS_H2_CRC_NOT_PRESENT:
            dprintf("%-35s", "CRC_NOT_PRESENT");
            break;

        case LW_FUSE_CRC_STATUS_H2_CRC_PRESENT_NO_ERRORS:
            dprintf("%-35s", "CRC_PRESENT_NO_ERRORS");
            break;

        case LW_FUSE_CRC_STATUS_H2_CRC_PRESENT_CORRECTED_ERRORS:
            dprintf("%-35s", "CRC_PRESENT_CORRECTED_ERRORS");
            break;

        case LW_FUSE_CRC_STATUS_H2_CRC_PRESENT_UNCORRECTED_ERRORS:
            dprintf("%-35s", "CRC_PRESENT_UNCORRECTED_ERRORS");
            break;
            
        default:
            dprintf("%-35s", "Unknown State");
            break;
    }
}

void dispHdcpKeydecryptionRegInfo_v04_01
(
)
{
    LwU32 data32;

    dprintf("\n====================================================================================================");
    dprintf("\n%-35s %-35s %-35s","REGISTER","FEILD","VALUE");
           
    dprintf("\n----------------------------------------------------------------------------------------------------");

    data32 = GPU_REG_RD32(LW_PDISP_HDCPRIF_STATUS);
    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_KEY_RAM_READY", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _KEY_RAM_READY, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_KP_READY", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _KP_READY, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_KEY_ID");
    switch (DRF_VAL(_PDISP, _HDCPRIF_STATUS, _KEY_ID, data32))
    {
        case LW_PDISP_HDCPRIF_STATUS_KEY_ID_DEBUG:
            dprintf("%-35s", "INIT/DEBUG");
            break;
    
        case LW_PDISP_HDCPRIF_STATUS_KEY_ID_ORIN:
            dprintf("%-35s", "ORIN");
            break;

        case LW_PDISP_HDCPRIF_STATUS_KEY_ID_AMPERE:
            dprintf("%-35s", "AMPERE");
            break;

        default:
            dprintf("%-35s", "Unknown State");
            break;
    }         

    dprintf("\n%-35s %-35s %-35d", "LW_PDISP_HDCPRIF_STATUS", "_ENTRY_ID", 
             DRF_VAL(_PDISP, _HDCPRIF_STATUS, _ENTRY_ID, data32));

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_DECRYPT_DONE", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _DECRYPT_DONE, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_KEY_WRITTEN", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _KEY_WRITTEN, data32) ? "YES" : "NO");

    
    dprintf("\n%-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_FUSE_FSM");
    switch (DRF_VAL(_PDISP, _HDCPRIF_STATUS, _FUSE_FSM , data32))
    {
        case LW_PDISP_HDCPRIF_STATUS_FUSE_FSM_SCRUB_RAM:
            dprintf("%-35s", "INIT/SCRUB_RAM");
            break;
    
        case LW_PDISP_HDCPRIF_STATUS_FUSE_FSM_WAIT_FUSE:
            dprintf("%-35s", "WAIT_FUSE");
            break;

        case LW_PDISP_HDCPRIF_STATUS_FUSE_FSM_READ_KP:
            dprintf("%-35s", "READ_KP");
            break;

        case LW_PDISP_HDCPRIF_STATUS_FUSE_FSM_READ_KFUSE_ENTRY:
            dprintf("%-35s", "READ_KFUSE_ENTRY");
            break;

        case LW_PDISP_HDCPRIF_STATUS_FUSE_FSM_DECRYPT_KEYS:
            dprintf("%-35s", "DECRYPT_KEYS");
            break;

        case LW_PDISP_HDCPRIF_STATUS_FUSE_FSM_WRITE_KEY_RAM:
            dprintf("%-35s", "WRITE_KEY_RAM");
            break;

        case LW_PDISP_HDCPRIF_STATUS_FUSE_FSM_KEYS_READY:
            dprintf("%-35s", "KEYS_READY");
            break;
            
        default:
            dprintf("%-35s", "Unknown State");
            break;
    }       

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_KEY_RAM_EMPTY", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _KEY_RAM_EMPTY, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_KEY_READY", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _KEY_READY, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_KP_ERROR", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _KP_ERROR, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_FUSE_ERROR", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _FUSE_ERROR, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_FUSE_TIMEOUT", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _FUSE_TIMEOUT, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_KP_ALL_0", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _KP_ALL_0, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_KP_ALL_1", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _KP_ALL_1, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_FUSE_ALL_0", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _FUSE_ALL_0, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_FUSE_ALL_1", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _FUSE_ALL_1, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_DEBUG_MODE", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _DEBUG_MODE, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_DIAG_VALID", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _DIAG_VALID, data32) ? "YES" : "NO");

    dprintf("\n%-35s %-35s %-35s", "LW_PDISP_HDCPRIF_STATUS", "_DIAG_ERROR", 
            DRF_VAL(_PDISP, _HDCPRIF_STATUS, _DIAG_ERROR, data32) ? "YES" : "NO");

    pDisp[indexGpu].dispPrintFuseCrcStatus();

    data32 = GPU_REG_RD32(LW_PDISP_UPSTREAM_HDCP_AKSV_MSB);
    dprintf("\n\n%-35s %-35s %-35d", "LW_PDISP_UPSTREAM_HDCP_AKSV_MSB", "_VALUE", 
            DRF_VAL(_PDISP, _UPSTREAM_HDCP_AKSV_MSB, _VALUE, data32));

    data32 = GPU_REG_RD32(LW_PDISP_UPSTREAM_HDCP_AKSV_LSB);
    dprintf("\n%-35s %-35s %-35d", "LW_PDISP_UPSTREAM_HDCP_AKSV_LSB", "_VALUE", 
            DRF_VAL(_PDISP, _UPSTREAM_HDCP_AKSV_LSB, _VALUE, data32));

    dprintf("\n====================================================================================================\n");              

}

LW_STATUS dispHdcpKeydecryptionStatus_v04_01
(
)
{
    LwBool     bIskeyReady;
    LW_STATUS  crcStatus;
    LW_STATUS  errorStatus = LW_OK;
    LW_STATUS  keyDecryptionStatus;
    LW_STATUS  upstreamStatus;
    LwU32      data32;
    LwU32      timer = 0;
   
    do
    {
        data32 = GPU_REG_RD32(LW_PDISP_HDCPRIF_STATUS);

        if (++timer == MAX_TIMEOUT_MS)
        {
            LW_PRINTF(LEVEL_ERROR,
                      "ERROR: Timeout while waiting for HDCP Keydecryption to complete,\
                       HDCPRIF_STATUS is 0x%x\n", data32);
            errorStatus = LW_ERR_TIMEOUT;
            break;
        }        
        else if (FLD_TEST_DRF(_PDISP, _HDCPRIF_STATUS, _DIAG_ERROR, _YES, data32) ||
                 FLD_TEST_DRF(_PDISP, _HDCPRIF_STATUS, _KP_ERROR, _YES, data32)   ||
                 FLD_TEST_DRF(_PDISP, _HDCPRIF_STATUS, _KP_ALL_0, _YES, data32)   ||
                 FLD_TEST_DRF(_PDISP, _HDCPRIF_STATUS, _KP_ALL_1, _YES, data32)   ||
                 FLD_TEST_DRF(_PDISP, _HDCPRIF_STATUS, _FUSE_TIMEOUT, _YES, data32) ||
                 FLD_TEST_DRF(_PDISP, _HDCPRIF_STATUS, _FUSE_ERROR, _YES, data32) ||
                 FLD_TEST_DRF(_PDISP, _HDCPRIF_STATUS, _FUSE_ALL_0, _YES, data32) ||
                 FLD_TEST_DRF(_PDISP, _HDCPRIF_STATUS, _FUSE_ALL_1, _YES, data32))
        {
            errorStatus = LW_ERR_GENERIC;
        }
        osPerfDelay(1000); // 1ms
    } while (!FLD_TEST_DRF(_PDISP, _HDCPRIF_STATUS, _KEY_READY, _YES, data32));

    bIskeyReady =  _dispHdcpKeysLoaded();
    upstreamStatus = _dispHdcpUpstreamStatus();
    crcStatus = pDisp[indexGpu].dispFuseCrcStatus();
    keyDecryptionStatus = ((bIskeyReady == LW_TRUE) && (crcStatus == LW_OK) && \
                           (upstreamStatus == LW_OK) && (errorStatus == LW_OK)) ? LW_OK : LW_ERR_GENERIC;

    dprintf("\nHDCP Key Decryption: %s", ((keyDecryptionStatus == LW_OK) ? "PASS" : "FAIL"));
    dprintf("\nKey Decryption Error: %s", ((errorStatus == LW_OK) ? "NO ERROR FOUND" : "ERROR FOUND"));
    dprintf("\nCRC: %s", ((crcStatus == LW_OK) ? "GOOD" : "NOT GOOD"));
    dprintf("\nUpstream Register: %s", ((upstreamStatus == LW_OK) ? "GOOD" : "NOT GOOD"));

    pDisp[indexGpu].dispHdcpKeydecryptionRegInfo();

    return LW_OK;
}

LW_STATUS dispHdcpIsKfuseReady_v04_01
(
)
{
    LwU32 kfuseStateVal;
    LwU32 kfuseErrorCountVal;
    LwU32 timer = 0;

    do
    {
        if (++timer == MAX_TIMEOUT_MS)
        {
            LW_PRINTF(LEVEL_ERROR,
                      "ERROR: Timeout while waiting for KFuse to complete\n");
            return LW_ERR_TIMEOUT;
        }
        kfuseStateVal = GPU_REG_RD32(LW_FUSE_KFUSE_STATE);
        osPerfDelay(1000);
    }
    while (!(kfuseStateVal & DRF_NUM(_FUSE_KFUSE, _STATE, _DONE, 1)));

    
    kfuseStateVal = GPU_REG_RD32(LW_FUSE_KFUSE_STATE);
    kfuseErrorCountVal = GPU_REG_RD32(LW_FUSE_KFUSE_ERRCOUNT);
    if ( (! ((kfuseStateVal & DRF_NUM(_FUSE_KFUSE, _STATE, _DONE, 1)) &&
             (kfuseStateVal & DRF_NUM(_FUSE_KFUSE, _STATE, _CRCPASS, 1))) ) ||
         (kfuseErrorCountVal & DRF_NUM(_FUSE_KFUSE, _ERRCOUNT, _ERR_FATAL,  0xffffffff)) )
    {
        LW_PRINTF(LEVEL_ERROR,
                  "KFUSE reported not complete or in error state: LW_FUSE_KFUSE_STATE: 0x%x; "
                  "LW_FUSE_KFUSE_ERRCOUNT: 0x%x\n", kfuseStateVal, kfuseErrorCountVal);
        return LW_ERR_GENERIC;
    }

    if (pDisp[indexGpu].dispFuseCrcStatus() != LW_OK)
    {
        dprintf("\nCRC errors exist\n");
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}

LW_STATUS dispFuseOptHdcpEnStatus_v04_01
(
)
{
    LwU32 fuseOptHdcpElwal = GPU_REG_RD32(LW_FUSE_OPT_HDCP_EN);
    LwU32 fuseEnSwOverrideVal = GPU_REG_RD32(LW_FUSE_EN_SW_OVERRIDE);

    if(!FLD_TEST_DRF(_FUSE, _OPT_HDCP_EN, _DATA, _YES, fuseOptHdcpElwal))
    {
        if (!FLD_TEST_DRF(_FUSE, _EN_SW_OVERRIDE, _VAL, _ENABLE, fuseEnSwOverrideVal))
        {
            GPU_REG_WR32(LW_FUSE_EN_SW_OVERRIDE, 
                    FLD_SET_DRF(_FUSE, _EN_SW_OVERRIDE, _VAL, _ENABLE, fuseEnSwOverrideVal));
        }

        GPU_REG_WR32(LW_FUSE_OPT_HDCP_EN, 
                FLD_SET_DRF(_FUSE, _OPT_HDCP_EN, _DATA, _YES, fuseOptHdcpElwal));

        if(!DRF_VAL(_FUSE, _OPT_HDCP_EN, _DATA, fuseOptHdcpElwal))
        {
            dprintf("\nHDCP not supported\n");
            return LW_ERR_GENERIC;
        }
    }

    return LW_OK;
}

LW_STATUS dispHdcpKeydecryptionTrigger_v04_01
(
)
{
    LwU32 data32;

    if(pDisp[indexGpu].dispFuseOptHdcpEnStatus() != LW_OK)
    {
        return LW_ERR_GENERIC;
    }

    if(_dispHdcpKeysLoaded())
    {
        dprintf("\nKey is already loaded\n");
        return LW_OK;
    }
    
    // Enabling fuse clocks only for CheetAh Orin
    if(pDisp[indexGpu].dispEnableFuseClocks() != LW_OK)
    {
        return LW_ERR_GENERIC;
    }

    // checking the fuse ready
    if (pDisp[indexGpu].dispHdcpIsKfuseReady() != LW_OK)
    {
        return LW_ERR_GENERIC;
    }
    
    // Set Fuse valid bit for starting HDCP Keydecryption process
    data32 = GPU_REG_RD32(LW_PDISP_HDCPRIF_CTRL);
    data32 = FLD_SET_DRF(_PDISP, _HDCPRIF_CTRL, _FUSE_VALID, _YES, data32);

    // Trigger HW to start HDCP Keydecryption
    GPU_REG_WR32(LW_PDISP_HDCPRIF_CTRL, data32);

    pDisp[indexGpu].dispHdcpKeydecryptionStatus();

    // Disabling fuse clocks only for CheetAh Orin
    if(pDisp[indexGpu].dispDisableFuseClocks() != LW_OK)
    {
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}    