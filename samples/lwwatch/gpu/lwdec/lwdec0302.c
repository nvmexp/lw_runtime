/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2017 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//-----------------------------------------------------
//
// lwdec0302.c - LWDEC 3.2 routines
//
//-----------------------------------------------------

#include "lwdec.h"
#include "g_lwdec_private.h"     // (rmconfig)  implementation prototypes
#include "class/clc3b0.h"
#include "lwdec0302.h"

//-----------------------------------------------------
// lwdecIsSupported_v03_02
//-----------------------------------------------------
BOOL lwdecIsSupported_v03_02(LwU32 indexGpu, LwU32 engineId)
{
    if (engineId != LWWATCH_LWDEC_0)
        return FALSE;

    pLwdecPrivReg[engineId] = lwdecPrivReg_v03_02;
    pLwdecMethodTable = lwdecMethodTable_v03_02;
    return TRUE;
}

//-----------------------------------------------------
// lwdecGetClassId_v03_02
//-----------------------------------------------------
LwU32
lwdecGetClassId_v03_02 (void)
{
    return LWC3B0_VIDEO_DECODER;
}
