/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwsync_porting.h"

#include <stdlib.h>

lwmutex_t lwmutex_create() {

#if LWWATCHCFG_IS_PLATFORM(WINDOWS)

    return CreateMutexA(NULL, FALSE, NULL);

#elif LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)

    return (int *) 1l;

#endif
}
