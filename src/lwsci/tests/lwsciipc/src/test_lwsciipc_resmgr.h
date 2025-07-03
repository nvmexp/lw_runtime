/*
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_TEST_LWSCIIPC_LWMAP_H
#define INCLUDED_TEST_LWSCIIPC_LWMAP_H

#include <devctl.h>

#define LWSCIIPC_LWMAPDEV "/dev/lwsciipc_lwmap"

#define _DCMD_LWSCIIPC   _DCMD_MISC

#define DCMD_LWSCIIPC_TEST_LWMAP     __DIOTF (_DCMD_LWSCIIPC, 20, LwSciIpcTestLwMap)

typedef struct {
    LwSciIpcEndpointVuid vuid;
    LwSciIpcEndpointAuthToken authToken;
} LwSciIpcTestLwMap;

#endif /* INCLUDED_TEST_LWSCIIPC_LWMAP_H */
