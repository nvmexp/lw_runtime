/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "psdl.h"
#include "exts.h"
#include "g_psdl_hal.h"

//-----------------------------------------------------
// psdlIsSupported - Determines if psdl is supported
//-----------------------------------------------------
BOOL psdlIsSupported(LwU32 indexGpu)
{
    if (!pPsdl[indexGpu].psdlIsSupported())
    {
        dprintf("lw: PSDL not supported on GPU %d.\n", indexGpu);
        return FALSE;
    }
    else
    {
        return TRUE;
    }
}

