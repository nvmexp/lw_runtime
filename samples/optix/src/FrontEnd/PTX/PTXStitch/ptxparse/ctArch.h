/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2015-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : ctArch.h
 *
 *  Description              :
 *
 */

#ifndef __CT_ARCH_H__
#define __CT_ARCH_H__

#include <stdTypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Function   : Extract the arch version as a number (weak version).
 *
 * Parameters : str  (I) String of the form "sm_xx" , "sass_xx", "lto_xx" or "compute_xx"
 *                   (O) integer value if valid, else fail with message
 */
unsigned int ctParseArchVersion(String str);

/*
 * Function   : Determine if the specified arch is virtual.
 *
 * Parameters : str  (I) String of the form "sm_xx" , "sass_xx", "lto_xx" or "compute_xx"
 *                   (O) True if the string is "compute_xx", else False
 */
Bool ctIsVirtualArch(String str);

/* Return True if arch can be JIT to SASS */
Bool ctIsJITableArch(String str);

#ifdef __cplusplus
}
#endif

#endif /* __CT_ARCH_H__ */
