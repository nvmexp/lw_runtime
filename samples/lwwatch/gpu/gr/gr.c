/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch GR common calls.
// gr.c
//
//*****************************************************

//
// includes
//
#include "hal.h"

//-----------------------------------------------------
// _grGetAperture
// - Used for windows work-around, need to be removed after windows compiler version enhancement. Bug: 3110134
//-----------------------------------------------------
#ifdef _WIN32
LW_STATUS _grGetAperture(GR_IO_APERTURE *pApertureIn, GR_IO_APERTURE **ppApertureOut, LwU32 count, ...)
{
    va_list argptr;
    LW_STATUS status = LW_ERR_GENERIC;

    va_start(argptr, count);
    status = pGr[indexGpu].grGetAperture((pApertureIn), (ppApertureOut), ((LwU32*)argptr), count);
    va_end(argptr);
    return status;
}
#endif

