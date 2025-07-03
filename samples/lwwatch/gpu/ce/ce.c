
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2016 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// CE routines 
// 
//*****************************************************

#include "lwwatch.h"
#include "hal.h"
#include "g_ce_hal.h"
#include "ce.h"
#include "exts.h"


//-----------------------------------------------------
// ceIsValid - Determines if CE is valid
//-----------------------------------------------------
BOOL ceIsValid( LwU32 indexGpu, LwU32 indexCe )
{
    return pCe[indexGpu].ceIsValid(indexCe);
}

//-----------------------------------------------------
// ceIsSupported - Determines if CE is supported
//-----------------------------------------------------
BOOL ceIsSupported( LwU32 indexGpu, LwU32 indexCe )
{
    if (!pCe[indexGpu].ceIsSupported(indexGpu, indexCe))
    {
        // It claims to not be supported, see if its a valid CE and modifiy our messgage if needed
        if (ceIsValid( indexGpu, indexCe ))
        {
            if (!pCe[indexGpu].ceIsPresent(indexCe))
            {
                dprintf("lw: CE%d not supported on GPU %d (disabled in HW).\n", indexCe, indexGpu);
            }
            else
            {
                if (!pCe[indexGpu].ceIsEnabled(indexGpu, indexCe))
                {
                    dprintf("lw: CE%d on GPU %d has no PCEs assigned to it.\n", indexCe, indexGpu);
                }
            }
        }
        else
        {
            dprintf("lw: CE%d not supported on GPU %d.\n", indexCe, indexGpu);
        }
        return FALSE;
    }
    else
    {
        dprintf("lw: CE%d supported on GPU %d.\n", indexCe, indexGpu);
        return TRUE;
    }
}

//-----------------------------------------------------
// ceDumpPriv - Dumps CE priv reg space
//-----------------------------------------------------
LW_STATUS ceDumpPriv( LwU32 indexGpu, LwU32 indexCe )
{
    if (!pCe[indexGpu].ceIsSupported(indexGpu, indexCe))
    {
        // It claims to not be supported, see if its a valid CE and modifiy our messgage if needed
        if (ceIsValid( indexGpu, indexCe ))
        {
            if (!pCe[indexGpu].ceIsPresent(indexCe))
            {
                dprintf("lw: ceDumpPriv: CE%d not supported on GPU %d (disabled in HW).\n", indexCe, indexGpu);
            }
            else
            {
                if (!pCe[indexGpu].ceIsEnabled(indexGpu, indexCe))
                {
                    dprintf("lw: ceDumpPriv: CE%d on GPU %d has no PCEs assigned to it.\n", indexCe, indexGpu);
                }
            }
        }
        else
        {
            dprintf("lw: ceDumpPriv: CE%d not supported on GPU %d.\n", indexCe, indexGpu);
        }
        return LW_ERR_NOT_SUPPORTED;
    }

    return pCe[indexGpu].ceDumpPriv(indexGpu, indexCe);
}

//-----------------------------------------------------
// ceTestState - Test basic CE state
//-----------------------------------------------------
LW_STATUS ceTestState( LwU32 indexGpu, LwU32 indexCe )
{
    if (!pCe[indexGpu].ceIsSupported(indexGpu, indexCe))
    {
        // It claims to not be supported, see if its a valid CE and modifiy our messgage if needed
        if (ceIsValid( indexGpu, indexCe ))
        {
            if (!pCe[indexGpu].ceIsPresent(indexCe))
            {
                dprintf("lw: ceTestState: CE%d not supported on GPU %d (disabled in HW).\n", indexCe, indexGpu);
            }
            else
            {
                if (!pCe[indexGpu].ceIsEnabled(indexGpu, indexCe))
                {
                    dprintf("lw: ceTestState: CE%d on GPU %d has no PCEs assigned to it.\n", indexCe, indexGpu);
                }
            }
        }
        else
        {
            dprintf("lw: ceTestState: CE%d not supported on GPU %d.\n", indexCe, indexGpu);
        }
        return LW_ERR_NOT_SUPPORTED;
    }

    return pCe[indexGpu].ceTestState(indexGpu, indexCe);
}

//-----------------------------------------------
// cePrintPceLceMap - Prints the pce-lce mappings
//-----------------------------------------------
void cePrintPceLceMap( LwU32 indexGpu )
{
    pCe[indexGpu].cePrintPceLceMap();
}

//-----------------------------------------------------
// cePrintPriv: used by various ceDumpPriv_ HAL implementations
//-----------------------------------------------------
void cePrintPriv(LwU32 clmn, char *tag, LwU32 id)
{
    size_t len = strlen(tag);
    dprintf("lw: %s",tag);

    if((len>0)&&(len<(clmn+4)))
    {
        LwU32 i;
        for(i=0;i<clmn-len;i++)
        {
            dprintf(" ");
        }
    }
    dprintf("(0x%08X)  = 0x%08X\n", id, GPU_REG_RD32(id));
}

//-----------------------------------------------------
// ceDisplayHelp - Display related help info
//-----------------------------------------------------
void ceDisplayHelp(void)
{
    if (LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX))
    {
        dprintf("CE commands:\n");
        dprintf(" lws ce \"help\"               - Displays the CE related help menu\n");
        dprintf(" lws ce \"supported <index>\"  - Determines if CE is supported on available GPUs\n");
        dprintf(" lws ce \"priv <index>\"       - Dumps CE priv registers\n");
        dprintf(" lws ce \"state <index>\"      - Checks the current state of CE\n");
        dprintf(" lws ce \"pcelcemap\"          - Prints the current pce-lce mapping\n");
    }
    else
    {
        dprintf("CE commands:\n");
        dprintf(" ce \"help\"               - Displays the CE related help menu\n");
        dprintf(" ce \"supported <index>\"  - Determines if CE is supported on available GPUs\n");
        dprintf(" ce \"priv <index>\"       - Dumps CE priv registers\n");
        dprintf(" ce \"state <index>\"      - Checks the current state of CE\n");
        dprintf(" ce \"pcelcemap\"          - Prints the current pce-lce mapping\n");
    }
}

