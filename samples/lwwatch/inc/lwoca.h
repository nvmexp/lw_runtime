/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _LWOCA_H_
#define _LWOCA_H_

#include "lwld.h"

//******************************************************************************
//
// Module Name: LWOCA.H
//
// This file contains structures and constants that are defined in lwoca.cpp 
// and used else where in lwwatch.
//
//******************************************************************************
#ifdef __cplusplus
extern "C" {
#endif
LwBool IsWindowsKernelDump();
LwBool IsWindowsFullKernelDump();
LwBool IsWindowsMiniKernelDump();
LwBool IsWindowsDumpFile();
void  *findAndLoadOcaProtoBuffer(LwU32 *);
LwBool lwlogInitFromOca(LWLD_Decoder *lwld);
void dumpModeReleaseLwlog(void);
#ifdef __cplusplus
}
#endif

#endif  // _LWOCA_H_
