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
// msenc0702.c - LWENC routines
//
//-----------------------------------------------------

#include "turing/tu102/dev_lwenc_pri_sw.h"
#include "turing/tu102/dev_falcon_v4.h"
#include "class/clc4b7.h"

#include "msenc.h"
#include "hwref/lwutil.h"
#include "g_msenc_private.h"     // (rmconfig)  implementation prototypes

#define USE_LWENC_7_2

#if defined(USE_LWENC_7_2)
#include "msenc0702.h"
#endif

//-----------------------------------------------------
// msencIsSupported_v07_02
//-----------------------------------------------------
BOOL msencIsSupported_v07_02( LwU32 indexGpu )
{
    if(lwencId != LWWATCH_MSENC_0)
    {
        dprintf("Only MSENC0 is supported on this GPU\n");
        return FALSE;
    }

    pMsencPrivReg[0] = msencPrivReg_v07_02_eng0;
    pMsencMethodTable = msencMethodTable_v07_02;

    cmnMethodArraySize = CMNMETHODARRAYSIZEC4B7;
    appMethodArraySize = APPMETHODARRAYSIZEC4B7;

    engineId = lwencId;

    return TRUE;
}

//-----------------------------------------------------
// msencGetClassId_v07_02 - Returns Class ID supported
//                          for IP 07.2
//-----------------------------------------------------
LwU32
msencGetClassId_v07_02 (void)
{
    return LWC4B7_VIDEO_ENCODER;
}

