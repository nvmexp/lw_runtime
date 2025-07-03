/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// psdl.h
//
//*****************************************************

#ifndef _LWWATCH_PSDL_H_
#define _LWWATCH_PSDL_H_

#include "os.h"
#include "hal.h"
#include "falcon.h"
#include "gpuanalyze.h"
#include "g_psdl_private.h"

typedef struct _psdl_engine_config
{
    const   FLCN_CORE_IFACES   *pFCIF;
    const   FLCN_ENGINE_IFACES *pFEIF;
    LwU32   *pUcodeData;
    LwU32   *pUcodeHeader;
    LwU32   ucodeSize;
    LwU32   *pSigDbg;
    LwU32   *pSigProd;
    LwU32   *pSigPatchLoc;
    LwU32   *pSigPatchSig;
    
}psdl_engine_config;

BOOL      psdlIsSupported(LwU32 indexGpu);

#endif // _LWWATCH_PSDL_H_
