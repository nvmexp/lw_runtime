/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef _FBFLCN_H_
#define _FBFLCN_H_

#include "os.h"
#include "hal.h"
#include "falcon.h"
#include "g_fbflcn_hal.h"     // (rmconfig)  public interfaces

POBJFLCN    fbflcnGetFalconObject  (void);
const char* fbflcnGetEngineName    (void);
LwU32       fbflcnGetDmemAccessPort(void);
const char* fbflcnGetSymFilePath   (void);

#endif // _FBFLCN_H_
