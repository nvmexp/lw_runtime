/*
* Copyright (c) 2021 LWPU Corporation.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#include "ogtest.h"
#include "cmdline.h"
#include "lwca.h"

int InitializeLWDA()
{
    LWresult result;
    assert(lwdaEnabled);

    result = lwInit(0);
    if (result != LWDA_SUCCESS) {
        return 0;
    }

    int lwDeviceCount = 0;
    result = lwDeviceGetCount(&lwDeviceCount);
    if (result != LWDA_SUCCESS || lwDeviceCount == 0) {
        return 0;
    }

    return 1;
}
