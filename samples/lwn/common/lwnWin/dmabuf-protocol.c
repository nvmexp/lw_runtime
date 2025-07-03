/*
 * Copyright (c) 2020, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// In order to use the dmabuf protocol, the following .c file needs to
// be included in our builds. But direct inclusion generates an error.
// Create a wrapper file and disable the offending error

#pragma GCC diagnostic ignored "-Wattributes"
#include "linux-dmabuf-unstable-v1-protocol.c"
