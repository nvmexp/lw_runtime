/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef _RISCV_PORTING_H_
#define _RISCV_PORTING_H_

#include <os.h>

// Windows supports strings differently
#if LWWATCHCFG_IS_PLATFORM(WINDOWS)
#define strtok_r strtok_s
#define strtoull _strtoui64
#define strncasecmp _strnicmp
#endif

// Delay - different for each platform
#if LWWATCHCFG_IS_PLATFORM(WINDOWS)
#include <WinBase.h>
#define PLATFORM_DELAY_MS(X) Sleep((X))
#elif LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
// TODO: Add silicon support
extern void ModsDrvClockSimulator(void);
#define PLATFORM_DELAY_MS(X) do ModsDrvClockSimulator(); while ((X)--)
#elif LWWATCHCFG_IS_PLATFORM(UNIX)
#include <unistd.h>
#define PLATFORM_DELAY_MS(X) do usleep(1000); while ((X)--)
#else
#error Missing delay implementation for platform.
#endif

#endif
