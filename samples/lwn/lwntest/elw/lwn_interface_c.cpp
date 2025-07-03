/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//
// lwn_interface_c.cpp
//
// This module includes lwntest utility code specific to the native C
// interface for LWN.
//
#include "lwntest_c.h"
#include "lwn_utils.h"
#include "lwn/lwn_FuncPtr.h"
#include "lwn/lwn_FuncPtrImpl.h"

// Utility code to reload the native C interface for a new device or when
// switching back to an old one.
void ReloadCInterface(LWNdevice *device, PFNLWNDEVICEGETPROCADDRESSPROC getProcAddress)
{
    lwnLoadCProcs(device, getProcAddress);
}
