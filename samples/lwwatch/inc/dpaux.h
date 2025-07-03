/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2011-2018 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _DPAUX_H_
#define _DPAUX_H_

#include "hal.h"

#define MAX_DP_AUX_ADDRESS 0x000FFFFF

typedef enum
{
    AUXPORT_0 = 0,
    AUXPORT_1,
    AUXPORT_2,
    AUXPORT_3,
    AUXPORT_4,
    AUXPORT_5,
    AUXPORT_6,

    AUXPORT_MAX,
    AUXPORT_NONE = -1
}AUXPORT;

#include "g_dpaux_hal.h"                    // (rmconfig) public interface


#endif // _DPAUX_H_
