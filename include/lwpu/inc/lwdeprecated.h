/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef LW_DEPRECATED_H
#define LW_DEPRECATED_H

/*!
 * @file lwdeprecated.h
 *
 * @brief Deprecation in the LWPU SDK
 *
 * Why deprecate:
 *     Code is deprecated when you want to remove a feature entirely, but cannot
 *     do so immediately, nor in a single step, due to a requirement to remain
 *     backward compatible (keep older clients working).
 *
 * Backwards compatibility:
 *     Deprecated symbols and features may be supported for an unknown amount of
 *     time. "Deprecated" means that we want that time interval to be small, but
 *     that may not be under our control.
 *
 *     This file provides the following ways to support deprecated features:
 *
 *     1) Defining LW_STRICT_SDK before including a SDK headers. This will
 *        remove *all* deprecated APIs from the LW SDK, for example:
 *
 *        #define LW_STRICT_SDK
 *        #include "sdk/foo.h"
 *
 *     2) Defining the per-feature compatibility setting before including the
 *        SDK, for example:
 *
 *        #define LW_DEPRECATED_LWOS_STATUS   0   // enable compatibility mode
 *        #include "sdk/foo.h"
 *
 * How to deprecate a feature in the SDK:
 *
 *    1) Define the deprecated feature in this file. Often, you'll want to
 *       start with SDK compatibility enabled by default, for example:
 *
 *       #ifndef LW_DEPRECATED_FEATURE_NAME
 *       #define LW_DEPRECATED_FEATURE_NAME 0
 *       #endif
 *
 *    2) Wrap SDK definitions with compatibility #ifdefs:
 *
 *       #if LW_DEPRECATED_COMPAT(FEATURE_NAME)
 *           ...legacy definitions...
 *       #endif
 *
 *    3) In the API implementation, consider stubbing or wrapping the new API.
 *
 *    4) Update older clients: file bugs to track this effort. Bug numbers
 *       should be placed in comments near the deprecated features that RM is
 *       supporting. That way, people reading the code can easily find the
 *       bug(s) that show the status of completely removing the deprecated
 *       feature.
 *
 *    5) Once all the client (calling) code has been upgraded, change the
 *       macro to "compatibility off". This is a little more cautious and
 *       conservative than jumping directly to step (6), because it allows you
 *       to recover from a test failure (remember, there are extended, offline
 *       tests that are not, unfortunately, run in DVS, nor per-CL checkin)
 *       with a tiny change in code.
 *
 *    6) Once the code base has migrated, remove all definitions from the SDK.
 */

/*
 *  \defgroup Deprecated SDK Features
 *
 *            0 = Compatibility on by default (i.e.: defines present in SDK)
 *            1 = Compatibility off by default (i.e.: defines NOT in SDK)
 *
 *  @{
 */

/*!
 * RM Config Get/Set API is deprecated and RmControl should be used instead.
 * Bugs: XXXXXX, XXXXXX, etc
 */
#ifndef LW_DEPRECATED_RM_CONFIG_GET_SET
#define LW_DEPRECATED_RM_CONFIG_GET_SET     0
#endif

#ifndef LW_DEPRECATED_LWOS_STATUS
/* LWOS_STATUS codes is deprecated. LW_STATUS to be used instead */
#define LW_DEPRECATED_LWOS_STATUS           0
#endif

#ifndef LW_DEPRECATED_RM_STATUS
/* RM_STATUS codes is deprecated. LW_STATUS to be used instead */
#define LW_DEPRECATED_RM_STATUS           0
#endif

#ifndef LW_DEPRECATED_UNSAFE_HANDLES
/* Using LwU32 for handles is deprecated. LwHandle to be used instead */
#define LW_DEPRECATED_UNSAFE_HANDLES        0
#endif

/**@}*/

/*!
 *  Utility Macros
 */

#ifdef LW_STRICT_SDK
// In strict mode, all obsolete features are unavailable in the SDK.
#define LW_DEPRECATED_COMPAT(feature)       0
#else
#define LW_DEPRECATED_COMPAT(feature)       (!LW_DEPRECATED_##feature)
#endif

#endif
