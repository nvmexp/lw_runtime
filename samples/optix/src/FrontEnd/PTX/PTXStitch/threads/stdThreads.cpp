/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2015-2016, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "stdPlatformDefs.h"

#if (defined(STD_OS_win32) || defined(STD_OS_MinGW) || defined(STD_OS_CygWin))
    #include "arch/stdThreadsWin32.cpp"
#else
    #include "arch/stdThreadsPOSIX.cpp"
#endif
