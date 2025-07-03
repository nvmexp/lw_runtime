/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2016-2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : ctArch.c
 *
 *  Last update              :
 *
 *  Description              :
 *
 */

#include <g_lwconfig.h>
#include <ctArch.h>
#include <stdMessages.h>
#include "compilerToolsMessageDefs.h"

static unsigned int ctParseArchVersionOrZero(String str)
{
  if (!str) return 0;

  if (strncmp(str, "sm_", 3) == 0)
    return atoi(str+3);

  if (strncmp(str, "compute_", 8) == 0)
    return atoi(str+8);

  if (strncmp(str, "lto_", 4) == 0)
    return atoi(str+4);

#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
  if (strncmp(str, "sass_", 5) == 0)
      return atoi(str+5);
#endif
  return 0;
}

unsigned int ctParseArchVersion(String str)
{
  unsigned int ret = ctParseArchVersionOrZero(str);
  stdCHECK(ret, (ctBadArchName, str));
  return ret;
}

Bool ctIsVirtualArch(String str)
{
  if (strncmp(str, "compute_", 8) == 0)
    return True;
  if (strncmp(str, "lto_", 4) == 0)
    return True;
  if (strncmp(str, "sm_", 3) != 0 
#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    && strncmp(str, "sass_", 5) != 0
#endif
      )
  {
    stdASSERT(False, ("invalid arch version: %s", str));
  }
  return False;
}

Bool ctIsJITableArch(String str)
{
  // only LTO cannot be JIT (lwrrently)
  return (strncmp(str, "lto_", 4) != 0);
}
