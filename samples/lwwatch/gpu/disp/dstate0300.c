/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2017-2018 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "inc/disp.h"
#include "clk.h"
#include "disp/v03_00/dev_disp.h"

#include "class/clc37d.h"

#include "g_disp_private.h"
#include "gpuanalyze.h"

#define LW_MAX_HEADS 8

typedef struct DisplayInfo
{
    LwU32 displayIdMask;
    LwU32 head;
    ORPROTOCOL orProtoCol;
    BOOL connected;
    BOOL powerUp;
    BOOL blanked;
    LwU32 orNum;
}DisplayInfo;

DisplayInfo displayInfo[LW_MAX_HEADS];

void dispFillDisplayInfo_v03_00(void);
void dumpDispDmaCtxNotifier_v03_00(void);
LW_STATUS dispCheckChnCtlCore_v03_00(void);
LW_STATUS dispCheckOrStatus_v03_00(LwU32 sorNum);
LW_STATUS dispCheckSFStatus_v03_00(LwU32 sfIndex);
LW_STATUS dispCheckIsoHubStatus_v03_00(LwU32 winNum);
LW_STATUS dispCheckLoadVStatus_v03_00(LwU32 headNum, LwU32 sorNum);
LW_STATUS dispCheckClockStatus_v03_00(LwU32 headNum);
LW_STATUS dispCheckDPStatus_v03_00(LwU32 sfIndex);
LW_STATUS dispCheckHDMIStatus_v03_00(LwU32 headNum, LwU32 sorNum);


void dispFillDisplayInfo_v03_00(void)
{
    LwU32       orNum, data32, head, ownerMask, headDisplayId = 0;
    ORPROTOCOL  orProtocol;

    for (orNum = 0; orNum < pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR); orNum++)
    {
        if (pDisp[indexGpu].dispResourceExists(LW_OR_SOR, orNum) != TRUE)
        {
            continue;
        }

        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE + LWC37D_SOR_SET_CONTROL(orNum));
        ownerMask = DRF_VAL(C37D, _SOR_SET_CONTROL, _OWNER_MASK, data32);
        if (ownerMask)
        {
            orProtocol = pDisp[indexGpu].dispGetOrProtocol(LW_OR_SOR, DRF_VAL(C37D, _SOR_SET_CONTROL, _PROTOCOL, data32));

            for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
            {
                if (BIT(head) & ownerMask)
                {
                    headDisplayId = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE + LWC37D_HEAD_SET_DISPLAY_ID(head, 0));
                    displayInfo[head].orProtoCol = orProtocol;
                    displayInfo[head].displayIdMask = headDisplayId;
                    displayInfo[head].connected = TRUE;
                    displayInfo[head].orNum = orNum;

                    data32 = GPU_REG_RD32(LW_PDISP_SF_BLANK(head));
                    if (DRF_VAL(_PDISP, _SF_BLANK, _STATUS, data32) == LW_PDISP_SF_BLANK_STATUS_BLANKED)
                    {
                        displayInfo[head].blanked = TRUE;
                    }
                    else
                    {
                        displayInfo[head].blanked = FALSE;
                    }
                }
            }
        }
    }
}

//-----------------------------------------------------
// dispTestDisplayState_v03_00()
//
//-----------------------------------------------------
LW_STATUS dispTestDisplayState_v03_00(void)
{
    LW_STATUS    status = LW_OK;
    LwU32 head = 0;
    LwU32 win = 0;

    if ((status = dispCheckChnCtlCore_v03_00()) != LW_OK)
    {
        dprintf("\n*******************************************************************************");
        dprintf("\n*             LW_PDISP_FE_CHNCTL/CHNSTATUS_CORE status is invalid             *");
        dprintf("\n*******************************************************************************\n");
        addUnitErr("\t LW_PDISP_FE_CHNCTL_CORE status is invalid\n");
    }
    else
    {
        dprintf("\n*******************************************************************************");
        dprintf("\n*             LW_PDISP_FE_CHNCTL/CHNSTATUS_CORE status is OK                  *");
        dprintf("\n*******************************************************************************\n");
    }

    dispFillDisplayInfo_v03_00();

    for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
    {
        if (!displayInfo[head].connected)
        {
            continue;
        }
        if ((status = dispAnalyzeCoreUpdSm_v03_00(head)) != LW_OK)
        {
            dprintf("\n*******************************************************************************");
            dprintf("\n*       See the debug prints to check core update state mechine hang          *");
            dprintf("\n*******************************************************************************\n");
        }
        if ((status = dispCheckOrStatus_v03_00(displayInfo[head].orNum)) != LW_OK)
        {
            dprintf("\n*******************************************************************************");
            dprintf("\n*                   See the debug prints to know OR status                    *");
            dprintf("\n*******************************************************************************\n");
        }
        if ((status = dispCheckSFStatus_v03_00(head)) != LW_OK)
        {
            dprintf("\n*******************************************************************************");
            dprintf("\n*                   See the debug prints to know SF status                    *");
            dprintf("\n*******************************************************************************\n");
        }

        for (win = head << 1; win < (head + 1) << 1; win++)
        {
            if (head != GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE + LWC37D_WINDOW_SET_CONTROL(win)))
            {
                continue;
            }

            if ((status = dispCheckIsoHubStatus_v03_00(win)) != LW_OK)
            {
                dprintf("\n*******************************************************************************");
                dprintf("\n*                See the debug prints to know ISOHUB status                   *");
                dprintf("\n*******************************************************************************\n");
            }
        }

        if ((status = dispCheckLoadVStatus_v03_00(head, displayInfo[head].orNum)) != LW_OK)
        {
            dprintf("\n*******************************************************************************");
            dprintf("\n*                     See the debug prints to know LoadV status               *");
            dprintf("\n*******************************************************************************\n");
        }

        if ((status = dispCheckClockStatus_v03_00(head)) != LW_OK)
        {
            dprintf("\n*******************************************************************************");
            dprintf("\n*                    See the debug prints to know Clock status                *");
            dprintf("\n*******************************************************************************\n");
        }

        if ((status = dispCheckDPStatus_v03_00(head)) != LW_OK)
        {
            dprintf("\n*******************************************************************************");
            dprintf("\n*                   See the debug prints to know DP status                    *");
            dprintf("\n*******************************************************************************\n");
        }

        if ((status = dispCheckHDMIStatus_v03_00(head, displayInfo[head].orNum)) != LW_OK)
        {
            dprintf("\n*******************************************************************************");
            dprintf("\n*                  See the debug prints to know HDMI status                   *");
            dprintf("\n*******************************************************************************\n");
        }

        if ((status = pDisp[indexGpu].dispCheckHdmifrlStatus(head, displayInfo[head].orNum)) != LW_OK)
        {
            dprintf("\n*******************************************************************************");
            if(status == LW_ERR_GENERIC)
            {
                dprintf("\n*              See the debug prints to know HDMI FRL status                   *");
            }
            else
            {
                dprintf("\n*                    HDMI FRL not supported on chip                           *");
            }
            
            dprintf("\n*******************************************************************************\n");
        }

    }

    return status;
}

//-----------------------------------------------------
// dumpDispDmaCtxNotifier_v03_00()
// go through state cache notifiers and check if any was
// set, if so, look it up in HT & dump
//-----------------------------------------------------
void dumpDispDmaCtxNotifier_v03_00(void)
{
    LwU32   handle;
    dprintf("\n==============================================================================");
    dprintf("\n                          Dumping DMA Ctx Notifiers                           ");
    dprintf("\n==============================================================================");

    // Dump context DMA here
    handle = GPU_REG_RD32(LWC37D_SET_CONTEXT_DMA_NOTIFIER);
    dprintf("\n%-55s   0x%08x", "LW857D_SC_SET_CONTEXT_DMA_NOTIFIER_HANDLE : ", handle); //todo

    if (handle != 0 )
    {
        pDisp[indexGpu].dispCtxDmaDescription(handle, -1, FALSE);
    }
    dprintf("\n---------------------------------------------------------------------\n");
}

//-----------------------------------------------------
// dispCheckChnCtlCore_v03_00()
//
//-----------------------------------------------------
LW_STATUS dispCheckChnCtlCore_v03_00(void)
{
    LW_STATUS    status = LW_OK;
    LwU32   data32 = 0;
    BOOL    isIdle = FALSE;

    dprintf("\n==============================================================================");
    dprintf("\n                          CHANNEL CONTROL CORE                                ");
    dprintf("\n==============================================================================");

    data32 = GPU_REG_RD32(LW_PDISP_FE_CHNCTL_CORE);

    dprintf("\n%-55s 0x%08x", "LW_PDISP_CHNCTL_CORE : ", data32);

    if (FLD_TEST_DRF( _PDISP, _FE_CHNCTL_CORE, _ALLOCATION, _ALLOCATE, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_ALLOCATION", "ALLOCATE");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_ALLOCATION", "DEALLOCATE");
        status = LW_ERR_GENERIC;
    }

    if (FLD_TEST_DRF(_PDISP, _FE_CHNCTL_CORE, _CONNECTION, _CONNECT, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_CONNECTION", "CONNECT");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_CONNECTION", "DISCONNECT");
        status = LW_ERR_GENERIC;
        return status;
    }

    if (FLD_TEST_DRF(_PDISP, _FE_CHNCTL_CORE, _PUTPTR_WRITE, _ENABLE, data32))
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_PUTPTR_WRITE", "ENABLE");
    else
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_PUTPTR_WRITE", "DISABLE");

    if (FLD_TEST_DRF(_PDISP, _FE_CHNCTL_CORE, _SKIP_NOTIF, _ENABLE, data32))
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_SKIP_NOTIF", "ENABLE");
    else
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_SKIP_NOTIF", "DISABLE");

    if (FLD_TEST_DRF(_PDISP, _FE_CHNCTL_CORE, _IGNORE_INTERLOCK, _ENABLE, data32))
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_IGNORE_INTERLOCK", "ENABLE");
    else
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_IGNORE_INTERLOCK", "DISABLE");

    if (FLD_TEST_DRF(_PDISP, _FE_CHNCTL_CORE, _ERRCHECK_WHEN_DISCONNECTED, _YES, data32))
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_ERRCHECK_WHEN_DISCONNECTED", "YES");
    else
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_ERRCHECK_WHEN_DISCONNECTED", "NO");

    switch ( DRF_VAL(_PDISP, _FE_CHNCTL_CORE, _TRASH_MODE, data32))
    {
        case LW_PDISP_FE_CHNCTL_CORE_TRASH_MODE_DISABLE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_TRASH_MODE", "DISABLE");
            break;

        case LW_PDISP_FE_CHNCTL_CORE_TRASH_MODE_TRASH_ONLY:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_TRASH_MODE", "TRASH_ONLY");
            break;

        case LW_PDISP_FE_CHNCTL_CORE_TRASH_MODE_TRASH_AND_ABORT:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_TRASH_MODE", "TRASH_AND_ABORT");
            break;

        default:
            dprintf("\n%-55s  0x%02x", "Unknown LW_PDISP_FE_CHNCTL_CORE_TRASH_MODE",
                DRF_VAL( _PDISP, _FE_CHNCTL_CORE, _TRASH_MODE, data32));
            addUnitErr("\t Unknown LW_PDISP_FE_CHNCTL_CORE_TRASH_MODE:  0x%02x\n",
                DRF_VAL( _PDISP, _FE_CHNCTL_CORE, _TRASH_MODE, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    if ( DRF_VAL(_PDISP, _FE_CHNCTL_CORE, _INTR_DURING_SHTDWN, data32))
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_INTR_DURING_SHTDWN", "ENABLE");
    else
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_INTR_DURING_SHTDWN", "DISABLE");

    dprintf("\n---------------------------------------------------------------------\n");

    // Read core channel state, below is good case
    // LW_PDISP_FE_CHNSTATUS_CORE @(0x00610630) = 0x200b0000
    //        STG1_STATE (3:0)                  = <IDLE> [0x0000]
    //        STG2_STATE (7:4)                  = <IDLE> [0x0000]
    //        STATE (20:16)                     = <IDLE> [0x000b]
    //        FIRSTTIME (24:24)                 = <NO> [0x0000]
    //        STATUS_METHOD_FIFO (25:25)        = <EMPTY> [0x0000]
    //        STATUS_READ_PENDING (26:26)       = <NO> [0x0000]
    //        STATUS_NOTIF_WRITE_PENDING (27:27) = <NO> [0x0000]
    //        SUBDEVICE_STATUS (29:29)          = <ACTIVE> [0x0001]
    //        STATUS_QUIESCENT (30:30)          = <NO> [0x0000]
    //        STATUS_METHOD_EXEC (31:31)        = <IDLE> [0x0000]
    //

    dprintf("\n==============================================================================");
    dprintf("\n                              CHANNEL STATUS CORE                             ");
    dprintf("\n==============================================================================");
    
    data32 = GPU_REG_RD32(LW_PDISP_FE_CHNSTATUS_CORE);
    dprintf("\n%-55s 0x%08x", "LW_PDISP_FE_CHNSTATUS_CORE : ", data32);

    switch (DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _STATE, data32))
    {
        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_IDLE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_CHNSTATUS_CORE_STATE","IDLE");
            isIdle = TRUE;
            break;

        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_DEALLOC:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATE","DEALLOC");
            status = LW_ERR_GENERIC;
            break;

        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_DEALLOC_LIMBO:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATE", "DEALLOC_LIMBO");
            status = LW_ERR_GENERIC;
            break;

        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_VBIOS_INIT1:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATE", "VBIOS_INIT1");
            status = LW_ERR_GENERIC;
            break;

        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_VBIOS_INIT2:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATE", "VBIOSINIT2");
            status = LW_ERR_GENERIC;
            break;

        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_UNCONNECTED:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATE", "UNCONNECTED");
            status = LW_ERR_GENERIC;
            break;

        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_VBIOS_OPERATION:
        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_EFI_OPERATION:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATE", "VBIOS_OPERATION/EFI_OPERATION\n");
            status = LW_ERR_GENERIC;
            break;

        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_BUSY:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATE", "BUSY");
            addUnitErr("\t LW_PDISP_FE_CHNSTATUS_CORE_STATE_BUSY\n");
            status = LW_ERR_GENERIC;
            break;

        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_SHUTDOWN1:
        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_SHUTDOWN2:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATE", "SHUTDOWN1 or 2");
            status = LW_ERR_GENERIC;
            break;

        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_EFI_INIT1:
        case LW_PDISP_FE_CHNSTATUS_CORE_STATE_EFI_INIT2:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATE", "EFI_INIT1 or 2");
            status = LW_ERR_GENERIC;

        default:
            dprintf("\n%-55s 0x%02x", "UNKNOWN LW_PDISP_FE_CHNSTATUS_CORE_STATE",
                DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _STATE, data32));
            addUnitErr("\t Unknown LW_PDISP_FE_CHNSTATUS_CORE_STATE:  0x%02x\n",
                DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _STATE, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    //read only when in EMPTY/WRIDLE/IDLE state or when incoming methods
    //have been stopped or paused.
    //
    if (isIdle)
    {
        if (FLD_TEST_DRF(_PDISP, _FE_CHNSTATUS_CORE, _STATUS_READ_PENDING, _YES, data32))
        {
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATUS_READ_PENDING", "YES");
        }
        else
        {
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATUS_READ_PENDING", "NO");
        }
    }
    else
    {
        dprintf("\n\t%-55s", "STATE not idle, unable to check STATUS_READ_PENDING");
    }

    if (FLD_TEST_DRF(_PDISP, _FE_CHNSTATUS_CORE, _STATUS_NOTIF_WRITE_PENDING, _NO, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATUS_NOTIF_WRITE_PENDING", "NO");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATUS_NOTIF_WRITE_PENDING", "YES");
        addUnitErr("\t LW_PDISP_FE_CHNSTATUS_CORE_STATUS_NOTIF_WRITE_PENDING_YES\n");
        status = LW_ERR_GENERIC;
        dumpDispDmaCtxNotifier_v03_00();
    }

    if (FLD_TEST_DRF(_PDISP, _FE_CHNSTATUS_CORE, _STATUS_QUIESCENT, _YES, data32))
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATUS_QUIESCENT", "YES");
    else
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNCTL_CORE_STATUS_QUIESCENT", "NO");


    if (FLD_TEST_DRF(_PDISP, _FE_CHNSTATUS_CORE, _SUBDEVICE_STATUS, _ACTIVE, data32))
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_SUBDEVICE_STATUS", "ACTIVE");
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_SUBDEVICE_STATUS", "INACTIVE");
        addUnitErr("\t LW_PDISP_FE_CHNSTATUS_CORE_SUBDEVICE_STATUS_INACTIVE\n");
        status = LW_ERR_GENERIC;
    }

    if ( FLD_TEST_DRF(_PDISP, _FE_CHNSTATUS_CORE, _STATUS_METHOD_EXEC, _RUNNING, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATUS_METHOD_EXEC", "RUNNING");
        addUnitErr("\t LW_PDISP_FE_CHNSTATUS_CORE_STATUS_METHOD_EXEC_RUNNING\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CHNSTATUS_CORE_STATUS_METHOD_EXEC", "IDLE");
    }

    dprintf("\n------------------------------------------------------------------------------");
    
    return status;
}


//
// This function  analyzes HEAD, MODE SWITCH STATE MACHINE, CMGR 
//
LW_STATUS dispAnalyzeCoreUpdSm_v03_00(LwU32 head)
{
    LwU32 data32 = 0;
    LW_STATUS status = LW_OK;

    dprintf("\n==============================================================================");
    dprintf("\n                 Analyzing RG Underflow state for Head: %d              ", head);
    dprintf("\n==============================================================================");
    

    data32 = GPU_REG_IDX_RD_DRF(_PDISP, _RG_UNDERFLOW, head, _ENABLE);
    dprintf("\n%-55s 0x%08x", "LW_PDISP_RG_UNDERFLOW : ", data32);

    if(data32 == LW_PDISP_RG_UNDERFLOW_ENABLE_DISABLE)
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_RG_UNDERFLOW_ENABLE", "DISABLE");
    }

    data32 = GPU_REG_IDX_RD_DRF(_PDISP, _RG_UNDERFLOW, head, _UNDERFLOWED);
    if(data32 == LW_PDISP_RG_UNDERFLOW_UNDERFLOWED_YES)
        dprintf("\n\t%-55s %-55s", "LW_PDISP_RG_UNDERFLOW_UNDERFLOWED", "YES");
    else
        dprintf("\n\t%-55s %-55s", "LW_PDISP_RG_UNDERFLOW_UNDERFLOWED", "NO");

    data32 = GPU_REG_IDX_RD_DRF(_PDISP, _RG_UNDERFLOW, head, _MODE);

    if(data32 == LW_PDISP_RG_UNDERFLOW_MODE_RED)
        dprintf("\n\t%-55s %-55s", "LW_PDISP_RG_UNDERFLOW_MODE", "RED");
    else
        dprintf("\n\t%-55s %-55s", "LW_PDISP_RG_UNDERFLOW_MODE", "REPEAT");

    dprintf("\n------------------------------------------------------------------------------\n");

    dprintf("\n==============================================================================");
    dprintf("\n             Analyzing Main State machine controlling core updates            ");
    dprintf("\n==============================================================================");

    data32 = GPU_REG_RD_DRF(_PDISP,_FE_CORE_UPD_STATE,_MAIN);
    dprintf("\n%-55s 0x%08x", "LW_PDISP_FE_CORE_UPD_STATE : ", data32);

    switch (data32)
    {
        case LW_PDISP_FE_CORE_UPD_STATE_MAIN_WAIT_ILK_PHASE_1:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MAIN", "WAIT_ILK_PHASE_1");
            break;

        case LW_PDISP_FE_CORE_UPD_STATE_MAIN_WAIT_UPD_FLAGS:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MAIN", "WAIT_UPD_FLAGS");
            break;

        case LW_PDISP_FE_CORE_UPD_STATE_MAIN_WAIT_BLOCK_SAT:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MAIN", "WAIT_BLOCK_SAT");
            break;

        case LW_PDISP_FE_CORE_UPD_STATE_MAIN_WAIT_STATE_ERRCHK:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MAIN", "WAIT_STATE_ERRCHK");
            break;

        case LW_PDISP_FE_CORE_UPD_STATE_MAIN_EXCEPTION:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MAIN", "EXCEPTION");
            break;

        case LW_PDISP_FE_CORE_UPD_STATE_MAIN_WAIT_READY_FLIP:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MAIN", "WAIT_READY_FLIP");
            break;

        case LW_PDISP_FE_CORE_UPD_STATE_MAIN_WAIT_MODESM:
            switch (GPU_REG_RD_DRF(_PDISP,_FE_CORE_UPD_STATE,_MSW))
            {
                case LW_PDISP_FE_CORE_UPD_STATE_MSW_IDLE:
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW"," IDLE");
                    dprintf("\n\tIdle and waiting for new modeswitch request\n");     
                    break;

                case LW_PDISP_FE_CORE_UPD_STATE_MSW_WAIT_SV1:
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW", "WAIT_SV1");
                    dprintf("\n\tSV1 has been raised; waiting for response from RM/VBIOS/EFI\n");
                    break;

                case LW_PDISP_FE_CORE_UPD_STATE_MSW_WAIT_MSFLAGS:
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW", "WAIT_MSFLAGS");
                    dprintf("\n\tWaiting for msw flags to be sent, then start pre-update FSM\n");
                    break;

                case LW_PDISP_FE_CORE_UPD_STATE_MSW_SEND_MSFLAGS:
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW", "SEND_MSFLAG");
                    dprintf("\n\tsending mode-switch flags method\n");
                    break;

                case LW_PDISP_FE_CORE_UPD_STATE_MSW_WAIT_PRESM :
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW", "WAIT_PRESM");
                    dprintf("\n\tfor the PRE state machine to complete\n");
                    pDisp[indexGpu].dispAnalyzeCoreUpdPreSm(head);
                    break;

                case LW_PDISP_FE_CORE_UPD_STATE_MSW_WAIT_PRECLK:
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW", "WAIT_PRECLK");
                    dprintf("\n\tfor the CMGR to finish disabling clocks or switching them to safe mode\n");      
                    pDisp[indexGpu].dispAnalyzeCoreUpdCmgrSm();
                    break;

                case LW_PDISP_FE_CORE_UPD_STATE_MSW_WAIT_SV2:
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW", "WAIT_SV2");
                    dprintf("\n\tfor the RM to write LW_PDISP_SUPERVISOR_RESTART\n");
                    break;

                case LW_PDISP_FE_CORE_UPD_STATE_MSW_WAIT_USUB:
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW", "WAIT_USUB");
                    dprintf("\n\tfor the USUB state machines to finish\n");
                    pDisp[indexGpu].dispAnalyzeCoreUpdUsubSm(head);
                    break;

                case LW_PDISP_FE_CORE_UPD_STATE_MSW_WAIT_POSTCLK:
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW", "WAIT_POSTCLK");
                    dprintf("\n\tfor the CMGR state machine to finish\n");
                    pDisp[indexGpu].dispAnalyzeCoreUpdCmgrSm();
                    break;

                case LW_PDISP_FE_CORE_UPD_STATE_MSW_WAIT_POST1SM:
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW", "WAIT_POST1SM");
                    dprintf("\n\tfor the POST state machines to finish\n");
                    pDisp[indexGpu].dispAnalyzeCoreUpdPost1Sm(head);
                    break;

                case LW_PDISP_FE_CORE_UPD_STATE_MSW_WAIT_SV3:
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW", "WAIT_SV3");
                    dprintf("\n for the RM to write LW_PDISP_SUPERVISOR_RESTART\n");
                    break;

                case LW_PDISP_FE_CORE_UPD_STATE_MSW_WAIT_POST2SM:
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW", "WAIT_POST2SM");
                    dprintf("\n\tfor the POST state machines to finish\n");
                    pDisp[indexGpu].dispAnalyzeCoreUpdPost2Sm(head);
                    break;

                case  LW_PDISP_FE_CORE_UPD_STATE_MSW_WAIT_CMPN:
                    dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_UPD_STATE_MSW", "WAIT_CMPN");
                    dprintf("\n\twaiting for condition to send completion notifier\n");
                    dumpDispDmaCtxNotifier_v03_00();

                default:
                    dprintf("\t%-55s for NONE\n", __FUNCTION__);
                    status = LW_OK;
            }
    }

    if (status == LW_OK)
    {
        dprintf("\n FE CORE UPDATE MAIN statemachine is OK.");
    }
    dprintf("\n------------------------------------------------------------------------------\n");
    
    return status;
}

//
// Analyze PRE state machine to debug display hang.
//
LW_STATUS dispAnalyzeCoreUpdPreSm_v03_00(LwU32 head)
{
    LW_STATUS status = LW_ERR_GENERIC;
    LwU32 data32     = 0;
    
    dprintf("\n==============================================================================");
    dprintf("\n                 Analyzing PRE State machine in Head : %d               ", head);
    dprintf("\n==============================================================================");

    data32 = GPU_REG_IDX_RD_DRF(_PDISP, _FE_CORE_HEAD_UPD_STATE, head, _PRE);
    dprintf("\n%-55s  0x%08x", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE : ", data32);

    switch (data32)
    {
        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE_WAIT_SNOOZE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE", "WAIT_SNOOZE");
            dprintf("\n\tsending method to put head in SNOOZE mode\n");
            break;

        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE_WAIT_SAFE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE", "WAIT_SAFE");
            break;

        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE_WAIT_SUPERUPD1:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE", "WAIT_SUPERUPD1");
            dprintf("\n\tfor a response from the bundle arbiter saying that the bundle was sent\n");
            break;

        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE_POLL_SOR_SAFE_SNOOZE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE", "SEND_UPD1");
            dprintf("\n\tto send the update to the Sor\n");
            break;

        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE_POLL_PIOR_SAFE_SNOOZE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE", "POLL_PIOR_SAFE_SNOOZE");
            dprintf("\n\tfor a response from the bundle arbiter saying that the bundle was sent\n");
            break;

        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE_WAIT_SLEEP:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE", "WAIT_SLEEP"); 
            break;

        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE_WAIT_SUPERUPD2:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE", "WAIT_SUPERUPD2"); 
            break;

        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE_POLL_SOR_SAFE_SLEEP:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE", "POLL_SOR_SAFE_SLEEP"); 
            break;

        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE_POLL_PIOR_SAFE_SLEEP:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE", "POLL_PIOR_SAFE_SLEEP");
            break;

        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE_POLL_SOR_DETACH:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE", "POLL_SOR_DETACH"); 
            break;

        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE_POLL_PIOR_DETACH:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE", "POLL_PIOR_DETACH");
            break;

        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE_POLL_SF_DETACH:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_PRE","POLL_SF_DETACH");
            break;

        default:
            status = LW_OK;
            dprintf(" %s for NONE\n", __FUNCTION__);
    }
    dprintf("\n------------------------------------------------------------------------------\n"); 
    return status;
}

//
// Analyze CMGR state machine to debug display hang this is for all heads
//
LW_STATUS dispAnalyzeCoreUpdCmgrSm_v03_00(void)
{
    LW_STATUS status = LW_ERR_GENERIC;
    LwU32 data32 = 0;
    
    dprintf("\n==============================================================================");
    dprintf("\n                         Analyzing CMGR State machine                         ");
    dprintf("\n==============================================================================");

    data32 = GPU_REG_RD_DRF(_PDISP, _FE_CMGR_STATUS, _STATE);
    dprintf("\n%-55s  0x%08x", "LW_PDISP_FE_CMGR_STATUS_STATE : ", data32);

    switch (data32)
    {
        case  LW_PDISP_FE_CMGR_STATUS_STATE_BEGIN_PRECLK:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "BEGIN_PRECLK");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_WAIT_SAFE_SETTLE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "WAIT_SAFE_SETTLE");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_PLLRESET_ENABLE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "PLLRESET_ENABLE");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_BYPASS_VPLL:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "BYPASS_VPLL");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_WAIT_BYPASS_SETTLE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "WAIT_BYPASS_SETTLE");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_DISABLE_VPLL:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "DISABLE_VPLL");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_BEGIN_POSTCLK:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "BEGIN_POSTCLK");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_ENABLE_VPLL:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "ENABLE_VPLL");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_WAIT_VPLL_LOCK:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "WAIT_VPLL_LOCK");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_WAIT_UNTIL_VBYPASS_SETTLE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "WAIT_UNTIL_VBYPASS_SETTLE");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_SET_OWNER:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "SET_OWNER");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_WAIT_OWNER_SETTLE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "WAIT_OWNER_SETTLE");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_PLLRESET_DISABLE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "PLLRESET_DISABLE");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_UNSAFE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "UNSAFE");
            break;

        case LW_PDISP_FE_CMGR_STATUS_STATE_WAIT_UNSAFE_SETTLE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_STATUS_STATE", "WAIT_UNSAFE_SETTLE");
            break;

        default:
            dprintf("\n\t%-55s for NONE", __FUNCTION__ );
            status = LW_OK;
    }
    dprintf("\n------------------------------------------------------------------------------\n");
    return status;
}

// Analyze USUB state machine to debug display hang.
LW_STATUS dispAnalyzeCoreUpdUsubSm_v03_00(LwU32 head)
{
    LW_STATUS status = LW_ERR_GENERIC;
    LwU32 data32     = 0;

    dprintf("\n==============================================================================");
    dprintf("\n                Analyzing USUB State machine in Head : %d               ", head);
    dprintf("\n==============================================================================");


    data32 = GPU_REG_IDX_RD_DRF(_PDISP, _FE_CORE_HEAD_UPD_STATE, head, _USUB_MAIN);
    dprintf("\n%-55s  0x%08x", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN : ", data32);

    switch (data32)
    {
    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN_USUB_START:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN", "USUB_START");
        dprintf("\n\tto send a request to the OR Polling mechanism to make sure there \
                    is room in the OR's bundle fifo\n");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN_SEND_HEAD_STATE:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN", "SEND_HEAD_STATE");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN_SEND_SETSHOWVGA:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN", "SEND_SETSHOWVGA");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN_WAIT_ILK_PH2:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN", "WAIT_ILK_PH2");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN_CHECK_PEND_LOADV:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN", "CHECK_PEND_LOADV");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN_SEND_UPD:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN", "SEND_UPD");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN_WAIT_UPD_RCVD:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN", "WAIT_UPD_RCVD");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN_SEND_SUPER_UPD:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN", "SEND_SUPER_UPD");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN_WAIT_SUPER_UPD_ACCEPT:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_MAIN", "WAIT_SUPER_UPD_ACCEPT");
        break;

    default:
        dprintf("\n\t%-55s for NONE", __FUNCTION__ );
        status = LW_OK;
    }
    
    data32 = 0;
    data32 = GPU_REG_IDX_RD_DRF(_PDISP, _FE_CORE_HEAD_UPD_STATE, head, _USUB_HEAD);
    dprintf("\n%-55s  0x%08x", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD : ", data32); 

    switch (data32)
    {
    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD_CHECK_WIN:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD", "CHECK_WIN");
        break;
    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD_SEND_SETCONTROL:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD", "SEND_SETCONTROL");
        break;
    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD_SEND_SETPROCAMP:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD", "SEND_SETPROCAMP");
        break;
    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD_SEND_SETCRCCONTROL:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD", "SEND_SETPROCAMP");
        break;
    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD_SEND_SETDESKTOPCOLOR:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD", "SEND_SETDESKTOPCOLOR");
        break;
    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD_SEND_OPMODE:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD", "SEND_OPMODE");
        break;
    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD_SEND_SHOWVGA:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_USUB_HEAD", "SEND_SHOWVGA");
        break;
    default:
        dprintf("\n\t%-55s for NONE", __FUNCTION__);
        status = LW_OK;
    }
    dprintf("\n------------------------------------------------------------------\n");
    return status;
}


// Analyze POST state machine to debug display hang.
LW_STATUS dispAnalyzeCoreUpdPost1Sm_v03_00(LwU32 head)
{
    LwU32 data32     = 0;
    LW_STATUS status = LW_ERR_GENERIC;

    dprintf("\n==============================================================================");
    dprintf("\n                Analyzing POST1 State machine for Head : %d             ", head);
    dprintf("\n==============================================================================");

    data32 = GPU_REG_IDX_RD_DRF(_PDISP, _FE_CORE_HEAD_UPD_STATE, head, _POST1);
    dprintf("\n%-55s  0x%08x", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1 : ", data32);

    switch (data32)
    {
    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1_SEND_ATTACH:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1", "SEND_ATTACH");
        dprintf("\n\tfor an ack from the bundle bus arbiter that the attach command has been sent\n");        
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1_POLL_SOR_ATTACH:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1", "POLL_SOR_ATTACH");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1_POLL_PIOR_ATTACH:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1", "POLL_PIOR_ATTACH");
        dprintf("\n\tto send the unsleep command to the OR\n");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1_POLL_SF_ATTACH:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1", "POLL_SF_ATTACH");
        dprintf("\n\tfor a response from the bundle arbiter saying that the bundle was sent\n");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1_SEND_SNOOZE:
        dprintf("\n%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1", "SEND_SNOOZE");
        dprintf("\n\tto send the unsafe command to the OR\n");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1_CHECK_NORMAL:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1", "CHECK_NORMAL");
        dprintf("\n\tfor a response from the bundle arbiter saying that the bundle was sent\n");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1_CHECK_UPDATE:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1", "CHECK_UPDATE");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1_SEND_UPDATE:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1", "SEND_UPDATE");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1_POLL_SOR_NORMAL:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1", "POLL_SOR_NORMAL");
        break;

    case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1_POLL_PIOR_NORMAL:
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST1", "POLL_PIOR_NORMAL");
        break;

    default:
        dprintf("\n\t%-55s for NONE", __FUNCTION__);
        status = LW_OK;
    }
    dprintf("\n------------------------------------------------------------------------------\n");
    return status;
}

LW_STATUS dispAnalyzeCoreUpdPost2Sm_v03_00(LwU32 head)
{
    LW_STATUS status = LW_ERR_GENERIC;
    LwU32 data32     = 0;

    dprintf("\n==============================================================================");
    dprintf("\n              Analyzing POST2 State machine for Head : %d               ", head);
    dprintf("\n==============================================================================");

    data32 = GPU_REG_IDX_RD_DRF(_PDISP, _FE_CORE_HEAD_UPD_STATE, head, _POST2);
    dprintf("\n%-55s  0x%08x", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST2 : ", data32);

    switch (data32)
    {
        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST2_SEND_WAKE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST2", "SEND_WAKE");
            break;
        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST2_SEND_UPDATE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST2", "SEND_UPDATE");
            break;
        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST2_POLL_SOR_AWAKE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST2", "POLL_SOR_AWAKE");
            break;
        case LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST2_POLL_PIOR_AWAKE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CORE_HEAD_UPD_STATE_POST2", "POLL_PIOR_AWAKE");
            break;
        default:
            dprintf("\n\t%-55s for NONE", __FUNCTION__);
            status = LW_OK;
    }
    dprintf("\n------------------------------------------------------------------------------\n");
    return status;
}

LW_STATUS dispCheckSFStatus_v03_00(LwU32 sfIndex)
{
    LW_STATUS status = LW_OK;
    LwU32 data32 = 0;
    LwU32 head = sfIndex;

    dprintf("\n==============================================================================");
    dprintf("\n                   Checking SF status for Sfindex : %d               ", sfIndex);
    dprintf("\n==============================================================================");

    data32 = GPU_REG_RD32(LW_PDISP_SF_TEST(sfIndex));

    dprintf("\n%-55s 0x%08x", "LW_PDISP_SF_TEST : ", data32);

    if (DRF_VAL(_PDISP, _SF_TEST, _OWNER_MASK, data32) != (LwU32) BIT(head))
    {
        dprintf("\n ERROR : LW_PDISP_SF_TEST[%d]_OWNER_MASK", sfIndex);
        status = LW_ERR_GENERIC;
    }

    switch (DRF_VAL(_PDISP, _SF_TEST, _ACT_HEAD_OPMODE, data32))
    {
        case LW_PDISP_SF_TEST_ACT_HEAD_OPMODE_SLEEP:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_TEST_ACT_HEAD_OPMODE", "SLEEP");
            status = LW_ERR_GENERIC;
            break;

        case LW_PDISP_SF_TEST_ACT_HEAD_OPMODE_SNOOZE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_TEST_ACT_HEAD_OPMODE", "SNOOZE");
            status = LW_ERR_GENERIC;
            break;

        case LW_PDISP_SF_TEST_ACT_HEAD_OPMODE_AWAKE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_TEST_ACT_HEAD_OPMODE", "AWAKE");
            break;

        default:
            dprintf("\n%-55s  0x%02x", "Unknown LW_PDISP_SF_TEST_ACT_HEAD_OPMODE: ",
                 DRF_VAL(_PDISP, _SF_TEST, _ACT_HEAD_OPMODE, data32));
            addUnitErr("\t Unknown LW_PDISP_SF_TEST_ACT_HEAD_OPMODE: 0x%02x\n",
                 DRF_VAL(_PDISP, _SF_TEST, _ACT_HEAD_OPMODE, data32));
            status = LW_ERR_GENERIC;
            break;
        }

    data32 = GPU_REG_RD32(LW_PDISP_SF_BLANK(sfIndex));
    dprintf("\n%-55s 0x%08x", "LW_PDISP_SF_BLANK : ", data32); 

    if (DRF_VAL(_PDISP, _SF_BLANK, _OVERRIDE, data32))
    {
        dprintf("\n\tError: LW_PDISP_SF_BLANK[%d]_OVERRIDE_TRUE", sfIndex);
        status = LW_ERR_GENERIC;
    }
    else{
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_BLANK_OVERRIDE", "FALSE");
    }

    if (DRF_VAL(_PDISP, _SF_BLANK, _STATUS, data32))
    {
        dprintf("\n\tError: LW_PDISP_SF_BLANK[%d]_STATUS_BLANKED", sfIndex);
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_BLANK_STATUS", "NOT_BLANKED");
    }

    dprintf("\n------------------------------------------------------------------------------\n");
    return status;
}

LW_STATUS dispCheckOrStatus_v03_00(LwU32 sorNum)
{
    LW_STATUS status = LW_OK;
    LwU32 data32 = 0;

    dprintf("\n==============================================================================");
    dprintf("\n                      Checking status for Sor: %d                     ", sorNum);
    dprintf("\n==============================================================================");

    //check sequencer
    data32 = GPU_REG_RD32(LW_PDISP_SOR_SEQ_CTL(sorNum));

    dprintf("\n%-55s 0x%08x", "LW_PDISP_SOR_LANE_SEQ_CTL : ", data32);

    if (DRF_VAL(_PDISP, _SOR_SEQ_CTL, _STATUS, data32) == 
        LW_PDISP_SOR_SEQ_CTL_STATUS_STOPPED)
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_SEQ_CTL", "STATE_IDLE");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "ERROR: LW_PDISP_SOR_SEQ_CTL", "STATUS_RUNNING");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PDISP_SOR_PWR(sorNum));
    dprintf("\n%-55s 0x%08x", "LW_PDISP_SOR_PWR : ", data32);


    if (DRF_VAL(_PDISP, _SOR_PWR, _MODE, data32) == LW_PDISP_SOR_PWR_MODE_SAFE)
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_PWR_MODE", "SAFE");
    else
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_PWR_MODE", "NORMAL");
    
    data32 = GPU_REG_RD32(LW_PDISP_SOR_TEST(sorNum));
    dprintf("\n%-55s 0x%08x", "LW_PDISP_SOR_TEST : ", data32);

    if (FLD_TEST_DRF(_PDISP, _SOR_TEST, _ASYNC_FIFO_OVERFLOW, _YES, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_TEST_ASYNC_FIFO_OVERFLOW", "YES");
        dprintf("\n\tASYNC FIFO OVERFLOW ERROR\n");
        status = LW_ERR_GENERIC;
    }
    else
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_TEST_ASYNC_FIFO_OVERFLOW", "NO");

    if (FLD_TEST_DRF(_PDISP, _SOR_TEST, _ASYNC_FIFO_UNDERFLOW, _YES, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_TEST_ASYNC_FIFO_UNDERFLOW", "YES");
        dprintf("\n\tASYNC FIFO UNDERFLOW ERROR\n");
        status = LW_ERR_GENERIC;
    }
    else
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_TEST_ASYNC_FIFO_UNDERFLOW", "NO");

    dprintf("\n------------------------------------------------------------------------------\n");
    return status;
}

LW_STATUS dispCheckIsoHubStatus_v03_00(LwU32 winNum)
{
    LW_STATUS status = LW_OK;
    LwU32 data32 = 0;

    dprintf("\n=================================================================================================");
    dprintf("\n                                Checking ISOHUB status for Win: %d                      ", winNum );
    dprintf("\n=================================================================================================");

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_POOL_CONFIG(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_WINDOW_POOL_CONFIG : ", data32);

    if (data32 == 0)
    {
        dprintf("\n\tWINDOW POOL CONFIG ERROR");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_POOL_CONFIG(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_LWRS_POOL_CONFIG : ", data32);

    if (data32 == 0)
    {
        dprintf("\n\tLWRSOR POOL CONFIG ERROR");
        status = LW_ERR_GENERIC;
    }
    
    data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_REQ_LIMIT(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_WINDOW_REQ_LIMIT : ", data32);

    if (data32 == 0)
    {
        dprintf("\n\tWINDOW REQUEST LIMIT ERROR");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_REQ_LIMIT(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_LWRS_REQ_LIMIT : ", data32);

    if (data32 == 0)
    {
        dprintf("\n\tLWRSOR REQUEST LIMIT ERROR");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_DEBUG_REQ_LIMIT(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_WINDOW_DEBUG_REQ_LIMIT : ", data32);

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_DEBUG_REQ_LIMIT(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_LWRS_DEBUG_REQ_LIMIT : ", data32);

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_OCC(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_WINDOW_OCC : ", data32);

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_FETCH_METER(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_WINDOW_FETCH_METER : ", data32);

    if (data32 == 0)
    {
        dprintf("\n\tWINDOW FETCH MERTERING ERROR");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_FETCH_METER(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_LWRS_FETCH_METER : ", data32);

    if (data32 == 0)
    {
        dprintf("\n\tLWRSOR FETCH MERTERING ERROR");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_DEBUG_STATUS(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_WINDOW_DEBUG_STATUS : ", data32);

    switch (DRF_VAL(_PDISP, _IHUB_WINDOW_DEBUG_STATUS, _CMREQSM, data32))
    {
        case LW_PDISP_IHUB_WINDOW_DEBUG_STATUS_CMREQSM_REQ_IDLE:
            dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_DEBUG_STATUS_CMREQSM",  "REQ_IDLE");
            break;

        case LW_PDISP_IHUB_WINDOW_DEBUG_STATUS_CMREQSM_REQ_TF_FETCH:
            dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_DEBUG_STATUS_CMREQSM", "REQ_TF_FETCH");
            status = LW_ERR_GENERIC;
            break;

        case LW_PDISP_IHUB_WINDOW_DEBUG_STATUS_CMREQSM_REQ_LUMA_FETCH:
            dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_DEBUG_STATUS_CMREQSM", "REQ_LUMA_FETCH");
            break;

        case LW_PDISP_IHUB_WINDOW_DEBUG_STATUS_CMREQSM_REQ_CHROMA_U_FETCH:
            dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_DEBUG_STATUS_CMREQSM", "REQ_CHROMA_U_FETCH");
            break;

        case LW_PDISP_IHUB_WINDOW_DEBUG_STATUS_CMREQSM_REQ_CHROMA_V_FETCH:
            dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_DEBUG_STATUS_CMREQSM", "REQ_CHROMA_V_FETCH");
            break;

        default:
            dprintf("\nUnknown LW_PDISP_SOR_TEST_CMREQSM:      0x%02x",
                    DRF_VAL(_PDISP, _IHUB_WINDOW_DEBUG_STATUS, _CMREQSM, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_REQ_SENT(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_WINDOW_REQ_SENT : ", data32);

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_RSP_RCVD(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_WINDOW_RSP_RCVD : ", data32);

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_STATUS(winNum));

    dprintf("\n%-70s 0x%08x", "LW_PDISP_IHUB_WINDOW_STATUS : ", data32);

    if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_STATUS, _DP_FETCH_ORDER_ERR_DETECTED, _YES, data32))
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_DP_FETCH_ORDER_ERR_DETECTED", "YES");
        dprintf("\n\tDP FETCH ORDER ERROR DETECTED\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_DP_FETCH_ORDER_ERR_DETECTED", "NO");
    }

    if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_STATUS, _POOL_NOT_EMPTY_AT_FRAME_END, _YES, data32))
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_POOL_NOT_EMPTY_AT_FRAME_END", "YES");
        dprintf("\n\tPOOL NOT EMPTY AT FRAME END\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_POOL_NOT_EMPTY_AT_FRAME_END", "NO");
    }

    if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_STATUS, _FORMATTER_NOT_IN_SPOOLUP_AT_FRAME_START, _YES, data32))
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_FORMATTER_NOT_IN_SPOOLUP_AT_FRAME_START", "YES");
        dprintf("\n\tFORMATTER NOT IN SPOOLUP AT FRAME START\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_FORMATTER_NOT_IN_SPOOLUP_AT_FRAME_START", "NO");
    }

    if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_STATUS, _FORMATTER_FIFO_OVERFLOW_ERR, _YES, data32))
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_FORMATTER_FIFO_OVERFLOW_ERR", "YES");
        dprintf("\n\tFORMATTER FIFO OVERFLOW ERROR\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_FORMATTER_FIFO_OVERFLOW_ERR", "NO");
    }

    if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_STATUS, _PENDING_REQUESTS_AT_END_OF_FRAME, _YES, data32))
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_PENDING_REQUESTS_AT_END_OF_FRAME", "YES");
        dprintf("\n\tPENDING REQUESTS AT END OF FRAME\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_PENDING_REQUESTS_AT_END_OF_FRAME", "NO");
    }

    if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_STATUS, _PENDING_CREDITS_AT_END_OF_FRAME, _YES, data32))
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_PENDING_CREDITS_AT_END_OF_FRAME", "YES");
        dprintf("\n\tPENDING CREDITS AT END OF FRAME\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_PENDING_CREDITS_AT_END_OF_FRAME", "NO");
    }

    if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_STATUS, _FORMATTER_CNTR_UNDERFLOW_ERR, _YES, data32))
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_FORMATTER_CNTR_UNDERFLOW_ERR ", "YES");
        dprintf("\n\tFORMATTER COUNTER UNDERFLOW ERROR\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_FORMATTER_CNTR_UNDERFLOW_ERR", "NO");
    }

    if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_STATUS, _THREAD_GROUP_NOT_ENABLED, _YES, data32))
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_THREAD_GROUP_NOT_ENABLED", "YES");
        dprintf("\n\tTHREAD GROUP NOT ENABLED\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_THREAD_GROUP_NOT_ENABLED", "NO");
    }

    if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_STATUS, _ROTATION_NOT_ENABLED_IN_BLX4, _YES, data32))
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_ROTATION_NOT_ENABLED_IN_BLX4", "YES");
        dprintf("\n\tROTATION NOT ENABLED IN BLX4\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_ROTATION_NOT_ENABLED_IN_BLX4", "NO");
    }

    if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_STATUS, _UNSUPPORTED_COLOR_FORMAT, _YES, data32))
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_UNSUPPORTED_COLOR_FORMAT", "YES");
        dprintf("\n\tUNSUPPORTED COLOR FORMAT\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_UNSUPPORTED_COLOR_FORMAT", "NO");
    }

    //
    // Bug 1831218 - this debug register doesn't work as intended.
    //if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_STATUS, _FETCH_PARAM_CHANGED_BEFORE_LINE_END, _YES, data32))
    //{
    //    dprintf("FETCH PARAM CHANGED BEFORE LINE END\n");
    //    status = LW_ERR_GENERIC;
    //}
    //

    if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_STATUS, _EVEN_OFFSET_NOT_PROGRAMMED_ALONG_WITH_PLANAR_FORMAT, _YES, data32))
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_EVEN_OFFSET_NOT_PROGRAMMED_ALONG_WITH_PLANAR_FORMAT", "YES");
        dprintf("\n\tEVEN OFFSET NOT PROGRAMMED ALONG WITH PLANAR FORMAT\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t%-70s %-70s", "LW_PDISP_IHUB_WINDOW_STATUS_EVEN_OFFSET_NOT_PROGRAMMED_ALONG_WITH_PLANAR_FORMAT", "NO");
    }

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_REQ(winNum));

    dprintf("\n%-55s 0x%08x", "LW_PDISP_IHUB_WINDOW_REQ : ", data32);
    dprintf("\n-------------------------------------------------------------------------------------------------");
    return status;
}

LW_STATUS dispCheckLoadVStatus_v03_00(LwU32 headNum, LwU32 sorNum)
{
    LW_STATUS status = LW_OK;
    LwU32 data32 = 0;
    LwU32 lwrldvcnt1 = 0;
    LwU32 precompldvcnt1 = 0;
    LwU32 postcompldvcnt1 = 0;
    LwU32 rgldvcnt1 = 0;
    LwU32 sfldvcnt1 = 0;
    LwU32 sorldvcnt1 = 0;
#ifdef WIN32
    LwU32 lwrldvcnt2 = 0;
    LwU32 precompldvcnt2 = 0;
    LwU32 postcompldvcnt2 = 0;
    LwU32 rgldvcnt2 = 0;
    LwU32 sfldvcnt2 = 0;
    LwU32 sorldvcnt2 = 0;
#endif

    dprintf("\n==============================================================================");
    dprintf("\n               Checking loadv status Head : %d, Sor : %d     ", headNum, sorNum);
    dprintf("\n==============================================================================");

    //
    // Check LW_PDISP_FE_HEAD_DEBUGA
    // This register reports the states of the LP (LoadV Processor) state machines as
    // dolwmented in the FE Real-Time IAS.
    //
    data32 = GPU_REG_RD32(LW_PDISP_FE_HEAD_DEBUGA(headNum));

    dprintf("\n%-55s 0x%08x", "LW_PDISP_FE_HEAD_DEBUGA : ", data32);

    switch (DRF_VAL(_PDISP, _FE_HEAD_DEBUGA, _ELV_FSM_STATE, data32))
    {
        case LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE_WAIT_ELV:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE", "WAIT_ELV");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE_WAIT_BLOCK:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE", "WAIT_BLOCK");
            status = LW_ERR_GENERIC;
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE_SET_LVPEND:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE", "SET_LVPEND");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE_WAIT_MP:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE", "WAIT_MP");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE_PRECALC:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE", "PRECALC");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE_WAIT_CALC:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE", "WAIT_CALC");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE_WAIT_DATA:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE", "WAIT_DATA");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE_REQ_LOADV:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE", "REQ_LOADV");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE_WAIT_FLUSH:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE", "WAIT_FLUSH");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE_CLR_LVPEND:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE", "CLR_LVPEND");
            break;

        default:
            dprintf("\n%-55s 0x%02x", "Unknown LW_PDISP_FE_HEAD_DEBUGA_ELV_FSM_STATE : ",
                    DRF_VAL(_PDISP, _FE_HEAD_DEBUGA, _ELV_FSM_STATE, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    switch (DRF_VAL(_PDISP, _FE_HEAD_DEBUGA, _SFF_FSM_STATE, data32))
    {
        case LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE_IDLE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE", "IDLE");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE_SEND_LOADV:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE", "SEND_LOADV");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE_PROMOTE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE", "PROMOTE");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE_START_DLY:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE" ,"START_DLY");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE_WAIT_VBL:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE", "WAIT_VBL");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE_WAIT_EN_SF:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE", "WAIT_EN_SF");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE_SEND_STARTF:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE", "SEND_STARTF");
            break;

        default:
            dprintf("\n%-55s 0x%02x", "Unknown LW_PDISP_FE_HEAD_DEBUGA_SFF_FSM_STATE : ",
                    DRF_VAL(_PDISP, _FE_HEAD_DEBUGA, _SFF_FSM_STATE, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    switch (DRF_VAL(_PDISP, _FE_HEAD_DEBUGA, _FETCH_FSM_STATE, data32))
    {
        case LW_PDISP_FE_HEAD_DEBUGA_FETCH_FSM_STATE_WAIT_SF:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_FETCH_FSM_STATE", "WAIT_SF");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_FETCH_FSM_STATE_WAIT_DATA:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_FETCH_FSM_STATE", "WAIT_DATA");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_FETCH_FSM_STATE_START_DLY:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_FETCH_FSM_STATE", "START_DLY");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_FETCH_FSM_STATE_WAIT_VBL:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_FETCH_FSM_STATE", "WAIT_VBL");
            break;

        default:
            dprintf("\n%-55s 0x%02x", "Unknown LW_PDISP_FE_HEAD_DEBUGA_FETCH_FSM_STATE : ",
                    DRF_VAL(_PDISP, _FE_HEAD_DEBUGA, _FETCH_FSM_STATE, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    switch (DRF_VAL(_PDISP, _FE_HEAD_DEBUGA, _ELV_BLOCK_STATE, data32))
    {
        case LW_PDISP_FE_HEAD_DEBUGA_ELV_BLOCK_STATE_NOBLK:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_BLOCK_STATE", "NOBLK");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_ELV_BLOCK_STATE_BLK_CON:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_BLOCK_STATE", "BLK_CON");
            break;

        case LW_PDISP_FE_HEAD_DEBUGA_ELV_BLOCK_STATE_BLK_ONESHOT:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_ELV_BLOCK_STATE", "BLK_ONESHOT");
            break;

        default:
            dprintf("\n%-55s 0x%02x", "Unknown LW_PDISP_FE_HEAD_DEBUGA_ELV_BLOCK_STATE : ",
                    DRF_VAL(_PDISP, _FE_HEAD_DEBUGA, _ELV_BLOCK_STATE, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    if (FLD_TEST_DRF(_PDISP, _FE_HEAD_DEBUGA, _VBL_DLY_STATE, _DONE, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_VBL_DLY_STATE", "DONE");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_VBL_DLY_STATE", "RUNNING");
    }

    if (FLD_TEST_DRF(_PDISP, _FE_HEAD_DEBUGA, _VBL_DLY_MODE, _CONTINUOUS, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_VBL_DLY_MODE", "CONTINUOUS");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_HEAD_DEBUGA_VBL_DLY_MODE", "ONE_SHOT");
    }

    // Check loadv counter and check further counters through each pipeline stages.
    lwrldvcnt1 = GPU_REG_RD32(LW_PDISP_LWRSOR_LOADV_COUNTER(headNum));
    dprintf("\nChecking head %d CURSOR loadv counter: %d", headNum, lwrldvcnt1);

    precompldvcnt1 = GPU_REG_RD32(LW_PDISP_PRECOMP_WIN_LOADV_COUNTER(headNum));
    dprintf("\nChecking head %d PRECOMP loadv counter: %d", headNum, precompldvcnt1);

    postcompldvcnt1 = GPU_REG_RD32(LW_PDISP_POSTCOMP_HEAD_LOADV_COUNTER(headNum));
    dprintf("\nChecking head %d POSTCOMP loadv counter: %d", headNum, postcompldvcnt1);
    //if (precompldvcnt1 != postcompldvcnt1)
    //{
    //    addUnitErr("\t Loadv mismatch between PRECOMP and POSTCOMP\n");
    //    dprintf("\t ERROR: Loadv mismatch between PRECOMP and POSTCOMP\n");
    //    status = LW_ERR_GENERIC;
    //}

    rgldvcnt1 = GPU_REG_RD32(LW_PDISP_RG_IN_LOADV_COUNTER(headNum));
    dprintf("\nChecking head %d RG loadv counter: %d", headNum, rgldvcnt1);
    if (postcompldvcnt1 != rgldvcnt1)
    {
        addUnitErr("\t Loadv mismatch between POSTCOMP and RG\n");
        dprintf("\n ERROR: Loadv mismatch between POSTCOMP and RG\n");
        status = LW_ERR_GENERIC;
    }

    sfldvcnt1 = GPU_REG_RD32(LW_PDISP_SF_IN_LOADV_COUNTER(headNum));
    dprintf("\nChecking head %d Sf loadv counter: %d", headNum, sfldvcnt1);
    if (rgldvcnt1 != sfldvcnt1)
    {
        addUnitErr("\t Loadv mismatch between RG and Sf\n");
        dprintf("\n ERROR: Loadv mismatch between RG and Sf\n");
        status = LW_ERR_GENERIC;
    }

    sorldvcnt1 = GPU_REG_RD32(LW_PDISP_SOR_IN_LOADV_COUNTER(sorNum));
    dprintf("\nChecking Sor %d loadv counter: %d", sorNum, sorldvcnt1);
    if (sfldvcnt1 != sorldvcnt1)
    {
        addUnitErr("\t Loadv mismatch between Sf and Sor\n");
        dprintf("\n ERROR: Loadv mismatch between Sf and Sor\n");
        status = LW_ERR_GENERIC;
    }

#ifdef WIN32
    Sleep(1000);

    lwrldvcnt2 = GPU_REG_RD32(LW_PDISP_LWRSOR_LOADV_COUNTER(headNum));
    dprintf("\nRecheck head %d CURSOR loadv counter after 1 second: %d", headNum, lwrldvcnt2);
    if (lwrldvcnt2 == lwrldvcnt1)
    {
        addUnitErr("\t No new loadv in CURSOR\n");
        dprintf("\n ERROR: No new loadv in CURSOR\n");
        status = LW_ERR_GENERIC;
    }

    precompldvcnt2 = GPU_REG_RD32(LW_PDISP_PRECOMP_WIN_LOADV_COUNTER(headNum));
    dprintf("\nRecheck head %d PRECOMP loadv counter after 1 second: %d", headNum, precompldvcnt2);
    if (precompldvcnt2 == precompldvcnt1)
    {
        addUnitErr("\t No new loadv in PRECOMP\n");
        dprintf("\n ERROR: No new loadv in PRECOMP\n");
        status = LW_ERR_GENERIC;
    }

    postcompldvcnt2 = GPU_REG_RD32(LW_PDISP_POSTCOMP_HEAD_LOADV_COUNTER(headNum));
    dprintf("\nRecheck head %d POSTCOMP loadv counter after 1 second: %d", headNum, postcompldvcnt2);
    if (postcompldvcnt2 == postcompldvcnt1)
    {
        addUnitErr("\t No new loadv in POSTCOMP\n");
        dprintf("\n ERROR: No new loadv in POSTCOMP\n");
        status = LW_ERR_GENERIC;
    }

    rgldvcnt2 = GPU_REG_RD32(LW_PDISP_RG_IN_LOADV_COUNTER(headNum));
    dprintf("\nRecheck head %d RG loadv counter after 1 second: %d", headNum, rgldvcnt2);
    if (rgldvcnt2 == rgldvcnt1)
    {
        addUnitErr("\t No new loadv in RG\n");
        dprintf("\n ERROR: No new loadv in RG\n");
        status = LW_ERR_GENERIC;
    }

    sfldvcnt2 = GPU_REG_RD32(LW_PDISP_SF_IN_LOADV_COUNTER(headNum));
    dprintf("\nRecheck head %d Sf loadv counter after 1 second: %d", headNum, sfldvcnt2);
    if (sfldvcnt2 == sfldvcnt1)
    {
        addUnitErr("\t No new loadv in Sf\n");
        dprintf("\n ERROR: No new loadv in Sf\n");
        status = LW_ERR_GENERIC;
    }

    sorldvcnt2 = GPU_REG_RD32(LW_PDISP_SOR_IN_LOADV_COUNTER(sorNum));
    dprintf("\nRecheck sor %d loadv counter after 1 second: %d", sorNum, sorldvcnt2);
    if (sorldvcnt2 == sorldvcnt1)
    {
        addUnitErr("\t No new loadv in Sor\n");
        dprintf("\n ERROR: No new loadv in Sor\n");
        status = LW_ERR_GENERIC;
    }
#endif

    dprintf("\n------------------------------------------------------------------------------");
    return status;
}

LW_STATUS dispCheckClockStatus_v03_00(LwU32 headNum)
{
    LW_STATUS status = LW_OK;
    LwU32 data32 = 0;
    LwU32 rgPclkKHz = 0;
    LwU32 scPclkKHz = 0;
    LwU32 VPLL;
    LwU32 rgDiv;

    dprintf("\n==============================================================================");
    dprintf("\n                  Checking Clock status for Head : %d                ", headNum);
    dprintf("\n==============================================================================");

    data32 = GPU_REG_RD32(LW_PDISP_FE_CMGR_CLK_RG(headNum));

    dprintf("\n%-55s 0x%08x", "LW_PDISP_FE_CMGR_CLK_RG : ", data32);

    switch (DRF_VAL(_PDISP, _FE_CMGR_CLK_RG, _MODE, data32))
    {
        case LW_PDISP_FE_CMGR_CLK_RG_MODE_NORMAL:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_CLK_RG_MODE", "NORMAL");
            VPLL = pClk[indexGpu].clkGetVClkFreqKHz(headNum);
            rgDiv = DRF_VAL(_PDISP, _FE_CMGR_CLK_RG, _DIV, data32) + 1;
            rgPclkKHz = VPLL / rgDiv;
            break;

        case LW_PDISP_FE_CMGR_CLK_RG_MODE_SAFE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_FE_CMGR_CLK_RG_MODE", "SAFE");
            rgPclkKHz = pClk[indexGpu].clkReadCrystalFreqKHz();
            break;

        default:
            dprintf("\n%-55s  0x%02x", "Unknown LW_PDISP_FE_CMGR_CLK_RG_MODE : ",
                    DRF_VAL(_PDISP, _FE_CMGR_CLK_RG, _MODE, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE +
                          LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY(headNum));
    scPclkKHz = DRF_VAL(C37D, _HEAD_SET_PIXEL_CLOCK_FREQUENCY, _HERTZ, data32) / 1000;

    if (rgPclkKHz != scPclkKHz)
    {
        addUnitErr("\t Pixel clock mismatch between PMGR (%d KHz) and SC (%d KHz)\n",
            rgPclkKHz, scPclkKHz);
        dprintf("\n ERROR: Pixel clock mismatch between PMGR (%d KHz) and SC (%d KHz)",
            rgPclkKHz, scPclkKHz);
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n Pixel clock = %d KHz\n", rgPclkKHz);
    }
    dprintf("\n------------------------------------------------------------------------------");

    return status;
}

LW_STATUS dispCheckDPStatus_v03_00(LwU32 sfIndex)
{
    LW_STATUS status = LW_OK;
    LwU32 data32 = 0;
    LwU32 head = sfIndex;
    LwU32 dplink;
    LwU32 laneCount;

    dprintf("\n==============================================================================");
    dprintf("\n                Checking DP_DEBUG status for Sfindex : %d            ", sfIndex);
    dprintf("\n==============================================================================");

    for (dplink = 0; dplink < LW_PDISP_SF_DP_DEBUG__SIZE_2; dplink++)
    {
        data32 = GPU_REG_RD32(LW_PDISP_SF_DP_DEBUG(sfIndex, dplink));

        dprintf("\nLW_PDISP_SF_DP_DEBUG[%d,%d] :      0x%08x", sfIndex, dplink, data32);

        for (laneCount = 0; laneCount < LW_PDISP_SF_DP_DEBUG_LANE_FIFO_UNDERFLOW__SIZE_1; laneCount++)
        {
            if (DRF_VAL(_PDISP, _SF_DP_DEBUG, _LANE_FIFO_UNDERFLOW(laneCount), data32))
            {
                dprintf("\n Error : LW_PDISP_SF_DP_DEBUG[%d,%d]_LANE%d_FIFO_UNDERFLOW", sfIndex, dplink, laneCount);
                status = LW_ERR_GENERIC;
            }

            if (DRF_VAL(_PDISP, _SF_DP_DEBUG, _LANE_PIXPACK_OVERFLOW(laneCount), data32))
            {
                dprintf("\n Error : LW_PDISP_SF_DP_DEBUG[%d,%d]_LANE%d_PIXPACK_OVERFLOW", sfIndex, dplink, laneCount);
                status = LW_ERR_GENERIC;
            }

            if (DRF_VAL(_PDISP, _SF_DP_DEBUG, _LANE_STEER_ERROR(laneCount), data32))
            {
                dprintf("\n Error : LW_PDISP_SF_DP_DEBUG[%d,%d]_LANE%d_LANE_STEER_ERROR", sfIndex, dplink, laneCount);
                status = LW_ERR_GENERIC;
            }

            if (DRF_VAL(_PDISP, _SF_DP_DEBUG, _LANE_FIFO_OVERFLOW(laneCount), data32))
            {
                dprintf("\n Error : LW_PDISP_SF_DP_DEBUG[%d,%d]_LANE%d_LANE_FIFO_OVERFLOW", sfIndex, dplink, laneCount);
                status = LW_ERR_GENERIC;
            }
        }

        if (DRF_VAL(_PDISP, _SF_DP_DEBUG, _SPKT_OVERRUN, data32))
        {
            dprintf("\n Error : LW_PDISP_SF_DP_DEBUG[%d,%d]_SPKT_OVERRUN", sfIndex, dplink);
            status = LW_ERR_GENERIC;
        }

        if (DRF_VAL(_PDISP, _SF_DP_DEBUG, _8LANE_SKEW_GT64, data32))
        {
            dprintf("\n Error : LW_PDISP_SF_DP_DEBUG[%d,%d]_8LANE_SKEW_GT64", sfIndex, dplink);
            status = LW_ERR_GENERIC;
        }
    }

    dprintf("\n------------------------------------------------------------------------------");
    return status;
}

LW_STATUS dispCheckHDMIStatus_v03_00(LwU32 headNum, LwU32 sorNum)
{
    LwU32 data32;
    LwU32 val;
    LW_STATUS status = LW_OK;

    dprintf("\n=================================================================================");
    dprintf("\n          Checking Sf HDMI control status for Head: %d, Sor: %d ", headNum, sorNum);
    dprintf("\n=================================================================================");

    data32 = GPU_REG_RD32(LW_PDISP_SF_HDMI_CTRL(headNum));
    dprintf("\n%-55s 0x%08x", "LW_PDISP_SF_HDMI_CTRL : ", data32);

    if (FLD_TEST_DRF(_PDISP, _SF_HDMI_CTRL, _ENABLE, _YES, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_CTRL_ENABLE", "YES");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_CTRL_ENABLE", "NO");
    }

    val = DRF_VAL(_PDISP, _SF_HDMI_CTRL, _REKEY, data32);
    if (val == LW_PDISP_SF_HDMI_CTRL_REKEY_INIT)
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_CTRL_REKEY", "INIT");
    }
    else
    {
        dprintf("\n\t%-55s %-55d", "LW_PDISP_SF_HDMI_CTRL_REKEY", val);
    }

    switch (DRF_VAL(_PDISP, _SF_HDMI_CTRL, _AUDIO_LAYOUT, data32))
    {
        case LW_PDISP_SF_HDMI_CTRL_AUDIO_LAYOUT_2CH:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_CTRL_AUDIO_LAYOUT", "2CH");
            break;

        case LW_PDISP_SF_HDMI_CTRL_AUDIO_LAYOUT_8CH:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_CTRL_AUDIO_LAYOUT", "8CH");
            break;

        default:
            dprintf("\n%-55s  0x%02x", "Unknown LW_PDISP_SF_HDMI_CTRL_AUDIO_LAYOUT : ",
                DRF_VAL(_PDISP, _SF_HDMI_CTRL, _AUDIO_LAYOUT, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    switch (DRF_VAL(_PDISP, _SF_HDMI_CTRL, _AUDIO_LAYOUT_SELECT, data32))
    {
        case LW_PDISP_SF_HDMI_CTRL_AUDIO_LAYOUT_SELECT_HW_BASED:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_CTRL_AUDIO_LAYOUT_SELECT", "HW_BASED");
            break;

        case LW_PDISP_SF_HDMI_CTRL_AUDIO_LAYOUT_SELECT_SW_BASED:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_CTRL_AUDIO_LAYOUT_SELECT", "SW_BASED");
            break;

        default:
            dprintf("\n\t%-55s  0x%02x", "Unknown LW_PDISP_SF_HDMI_CTRL_AUDIO_LAYOUT_SELECT : ",
                DRF_VAL(_PDISP, _SF_HDMI_CTRL, _AUDIO_LAYOUT_SELECT, data32));
            status = LW_ERR_GENERIC;
            break;
    }
    
    switch (DRF_VAL(_PDISP, _SF_HDMI_CTRL, _SAMPLE_FLAT, data32))
    {
        case LW_PDISP_SF_HDMI_CTRL_SAMPLE_FLAT_CLR:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_CTRL_SAMPLE_FLAT", "CLR");
            break;

        case LW_PDISP_SF_HDMI_CTRL_SAMPLE_FLAT_SET:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_CTRL_SAMPLE_FLAT", "SET");
            break;

        default:
            dprintf("\n%-55s  0x%02x", "Unknown LW_PDISP_SF_HDMI_CTRL_SAMPLE_FLAT : ",
                DRF_VAL(_PDISP, _SF_HDMI_CTRL, _SAMPLE_FLAT, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    val = DRF_VAL(_PDISP, _SF_HDMI_CTRL, _MAX_AC_PACKET, data32);
    if (val == LW_PDISP_SF_HDMI_CTRL_MAX_AC_PACKET_INIT)
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_CTRL_MAX_AC_PACKET", "INIT");
    }
    else
    {
        dprintf("\n\t%-55s %-55d", "LW_PDISP_SF_HDMI_CTRL_MAX_AC_PACKET", val);
    }

    if (FLD_TEST_DRF(_PDISP, _SF_HDMI_CTRL, _AUDIO, _EN, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_CTRL_AUDIO", "EN");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_CTRL_AUDIO", "DIS");
    }

    data32 = GPU_REG_RD32(LW_PDISP_SF_HDMI_VSYNC_KEEPOUT(headNum));
    dprintf("\n%-55s 0x%08x", "LW_PDISP_SF_HDMI_VSYNC_KEEPOUT : ", data32);

    val = DRF_VAL(_PDISP, _SF_HDMI_VSYNC_KEEPOUT, _END, data32);
    if (val == LW_PDISP_SF_HDMI_VSYNC_KEEPOUT_END_INIT)
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_VSYNC_KEEPOUT_END", "INIT");
    }
    else
    {
        dprintf("\n\t%-55s %-55d", "LW_PDISP_SF_HDMI_VSYNC_KEEPOUT_END", val);
    }

    val = DRF_VAL(_PDISP, _SF_HDMI_VSYNC_KEEPOUT, _START, data32);
    if (val == LW_PDISP_SF_HDMI_VSYNC_KEEPOUT_START_INIT)
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_VSYNC_KEEPOUT_START", "INIT");
    }
    else
    {
        dprintf("\n\t%-55s %-55d", "LW_PDISP_SF_HDMI_VSYNC_KEEPOUT_START", val);
    }

    if (FLD_TEST_DRF(_PDISP, _SF_HDMI_VSYNC_KEEPOUT, _ENABLE, _YES, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_VSYNC_KEEPOUT_ENABLE", "YES");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SF_HDMI_VSYNC_KEEPOUT_ENABLE", "NO");
    }
    dprintf("\n------------------------------------------------------------------------------\n");

    dprintf("\n==============================================================================");
    dprintf("\n           Checking HDMI control status for Head: %d, Sor: %d", headNum, sorNum);
    dprintf("\n==============================================================================");

    data32 = GPU_REG_RD32(LW_PDISP_SOR_HDMI2_CTRL(sorNum));
    dprintf("\n%-55s 0x%08x", "LW_PDISP_SOR_HDMI2_CTRL : ", data32);

    if (FLD_TEST_DRF(_PDISP, _SOR_HDMI2_CTRL, _SCRAMBLE, _ENABLE, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_CTRL_SCRAMBLE", "ENABLE");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_CTRL_SCRAMBLE", "DISABLE");
    }

    if (FLD_TEST_DRF(_PDISP, _SOR_HDMI2_CTRL, _CLOCK_MODE, _NORMAL, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_CTRL_CLOCK_MODE", "NORMAL");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_CTRL_CLOCK_MODE", "DIV_BY_4");
    }

    if (FLD_TEST_DRF(_PDISP, _SOR_HDMI2_CTRL, _SCRAMBLE_AT_LOADV, _ENABLE, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_CTRL_SCRAMBLE_AT_LOADV", "ENABLE");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_CTRL_SCRAMBLE_AT_LOADV", "DISABLE");
    }

    if (FLD_TEST_DRF(_PDISP, _SOR_HDMI2_CTRL, _SSCP_LENGTH, _INIT, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_CTRL_SSCP_LENGTH", "INIT (set to default)");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_CTRL_SSCP_LENGTH", "NOT set to default");
    }

    if (FLD_TEST_DRF(_PDISP, _SOR_HDMI2_CTRL, _SSCP_START, _INIT, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_CTRL_SSCP_START", "INIT (set to default)");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_CTRL_SSCP_START", "NOT set to default");
    }

    data32 = GPU_REG_RD32(LW_PDISP_SOR_HDMI2_LFSR0(sorNum));
    dprintf("\n%-55s 0x%08x", "LW_PDISP_SOR_HDMI2_LFSR0 : ", data32);
    
    if (FLD_TEST_DRF(_PDISP, _SOR_HDMI2_LFSR0, _LANE0_SEED, _INIT, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_LFSR0_LANE0_SEED", "INIT (set to default)");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_LFSR0_LANE0_SEED", "NOT set to default");
    }

    if (FLD_TEST_DRF(_PDISP, _SOR_HDMI2_LFSR0, _LANE1_SEED, _INIT, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_LFSR0_LANE1_SEED", "INIT (set to default)");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_LFSR0_LANE1_SEED", "NOT set to default");
    }

    data32 = GPU_REG_RD32(LW_PDISP_SOR_HDMI2_LFSR1(sorNum));
    dprintf("\n%-55s 0x%08x", "LW_PDISP_SOR_HDMI2_LFSR1 : ", data32);

    if (FLD_TEST_DRF(_PDISP, _SOR_HDMI2_LFSR1, _LANE2_SEED, _INIT, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_LFSR1_LANE2_SEED", "INIT (set to default)");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI2_LFSR1_LANE2_SEED", "NOT set to default");
    }

    dprintf("\n------------------------------------------------------------------------------");
    return status;
}
