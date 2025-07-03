/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef LWSYNC_PORTING_H
#define LWSYNC_PORTING_H

#include "lwwatch.h"

#if LWWATCHCFG_IS_PLATFORM(WINDOWS)

#include <windows.h>

#define lwmutex_t HANDLE

#define lwmutex_lock(mutex) WaitForSingleObject((mutex), -1)

#define lwmutex_unlock(mutex) ReleaseMutex((mutex))

#elif LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)

#define lwmutex_t int *

#define lwmutex_lock(mutex) do {} while(0)

#define lwmutex_unlock(mutex) do {} while(0)

#else

#error This platform is not supported by LwSync

#endif

lwmutex_t lwmutex_create();

#endif // LWSYNC_PORTING_H
