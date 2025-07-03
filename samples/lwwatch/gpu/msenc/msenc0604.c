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
// msenc0604.c - LWENC routines for GV100+
//
//-----------------------------------------------------

#include "volta/gv100/dev_lwenc_pri_sw.h"
#include "volta/gv100/dev_falcon_v4.h"
#include "class/clc3b7.h"

#include "msenc.h"
#include "hwref/lwutil.h"
#include "g_msenc_private.h"     // (rmconfig)  implementation prototypes
#include "msenc0604.h"

//-----------------------------------------------------
// msencIsSupported_v06_04
//-----------------------------------------------------
BOOL msencIsSupported_v06_04( LwU32 indexGpu )
{
    if(lwencId > LWWATCH_MSENC_2)
    {
        dprintf("Only MSENC0, MSENC1 and MSENC2 supported on this GPU\n");
        return FALSE;
    }

    pMsencPrivReg[0] = msencPrivReg_v06_04_eng0;
    pMsencPrivReg[1] =  msencPrivReg_v06_04_eng1;
    pMsencPrivReg[2] =  msencPrivReg_v06_04_eng2;
    pMsencMethodTable = msencMethodTable_v06_04;

    cmnMethodArraySize = CMNMETHODARRAYSIZEC1B7;
    appMethodArraySize = APPMETHODARRAYSIZEC1B7;

    engineId = lwencId;

    return TRUE;
}

//-----------------------------------------------------
// msencGetClassId_v06_04 - Returns Class ID supported
//                          for IP 06.4
//-----------------------------------------------------
LwU32
msencGetClassId_v06_04 (void)
{
    return LWC3B7_VIDEO_ENCODER;
}
