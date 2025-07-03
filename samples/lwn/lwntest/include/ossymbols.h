/*
 * Copyright (c) 2013 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __LWOGTEST_OSSYMBOLS_H
#define __LWOGTEST_OSSYMBOLS_H

#if defined(_WIN32)

#pragma warning(disable: 4244)  // possible loss of data
#pragma warning(disable: 4305)  // VC++ 5.0 version of above warning

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#define LW_WINDOWS

#if defined(_WIN64)
#define LW_WINDOWS_64
#endif

#endif // defined(_WIN32)

#if !defined(LW_WINDOWS)
#define CDECL
#endif

#endif // __LWOGTEST_OSSYMBOLS_H
