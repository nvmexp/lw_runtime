/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef _GSP_H_
#define _GSP_H_

#include "os.h"
#include "hal.h"
#include "falcon.h"
#include "g_gsp_hal.h"     // (rmconfig)  public interfaces

#define GSP_MUTEX_TIMEOUT_US (0x2000)
#define GSP_RESET_TIMEOUT_US (0x1000)


POBJFLCN    gspGetFalconObject  (void);
LwU32       gspGetDmemAccessPort(void);

#endif // _GSP_H_
