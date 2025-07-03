/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <unistd.h>
#include <sys/time.h>

#include "os.h"

void osPerfDelay(LwU32 us)
{
    usleep(us);
}

LW_STATUS osGetLwrrentTick(LwU64 *pTimeInNs)
{
    struct timeval tv;

    if (pTimeInNs == NULL)
    {
        return LW_ERR_GENERIC;
    }

    if (gettimeofday(&tv, NULL) != 0)
    {
        *pTimeInNs = 0;
        return LW_ERR_GENERIC;
    }

    *pTimeInNs = (LwU64)tv.tv_sec * 1000000000 + (LwU64)tv.tv_usec * 1000;
    return LW_OK;
}
