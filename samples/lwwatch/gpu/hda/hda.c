
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// HDA routines 
// 
//*****************************************************

#include "hal.h"
#include "g_hda_hal.h"
#include "hda.h"
#include "exts.h"

//-----------------------------------------------------
// hdaIsSupported - Determines if HDA is supported
//-----------------------------------------------------
BOOL hdaIsSupported(LwU32 indexGpu)
{
    if (!pHda[indexGpu].hdaIsSupported(indexGpu))
    {
        return FALSE;
    }
    else
    {
        return TRUE;
    }
}

//-----------------------------------------------------
// hdaDumpImem - Dumps HDA instruction memory
//-----------------------------------------------------
LW_STATUS hdaDumpImem(LwU32 indexGpu, LwU32 imemSize)
{
    if (!pHda[indexGpu].hdaIsSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pHda[indexGpu].hdaDumpImem(indexGpu, imemSize);
}

//-----------------------------------------------------
// hdaDumpDmem - Dumps HDA data memory
//-----------------------------------------------------
LW_STATUS hdaDumpDmem(LwU32 indexGpu, LwU32 dmemSize)
{
    if (!pHda[indexGpu].hdaIsSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pHda[indexGpu].hdaDumpDmem(indexGpu, dmemSize);
}

//-----------------------------------------------------
// hdaTestState - Test basic HDA state
//-----------------------------------------------------
LW_STATUS hdaTestState(LwU32 indexGpu)
{
    if (!pHda[indexGpu].hdaIsSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pHda[indexGpu].hdaTestState(indexGpu);
}

//-----------------------------------------------------
// hdaDisplayHelp - Display related help info
//-----------------------------------------------------
void hdaDisplayHelp(void)
{
    dprintf("HDA commands:\n");
    dprintf(" hda \"-help\"             - Displays the HDA related help menu\n");
    dprintf(" hda \"-supported\"        - Determines if HDA is supported on available GPUs.\n");
    dprintf(" hda \"-imem <imemsize>\"  - Dumps HDA instruction memory.\n");
    dprintf(" hda \"-dmem <dmemsize>\"  - Dumps HDA data memory.\n");
    dprintf(" hda \"-state\"            - Checks the current state of HDA.\n");
}
