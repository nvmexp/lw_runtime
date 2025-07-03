/*
 *  Copyright 2019-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once

#include <stdexcept>
#include <string.h>
#include <sstream>
#include <lwVer.h>
#include "lwRmApi.h"

class FMGpuDriverVersionCheck
{
public:
    FMGpuDriverVersionCheck();
    ~FMGpuDriverVersionCheck();
    void checkGpuDriverVersionCompatibility(std::string errorCtx);

private:

#define RM_VERSION_STRING_SIZE 80

    typedef struct {
        char version[RM_VERSION_STRING_SIZE];
    } RMLibVersionInfo_t;

    void checkRMLibVersionCompatibility(std::string errorCtx);
    void openGpuDriverHandle(void);
    void closeGpuDriverHandle(void);

    LwHandle mRmClientHandle;
    static RMLibVersionInfo_t mGpuDrvWhitelistedVersions[];
};
