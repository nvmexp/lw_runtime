/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include <iostream>

#include "supportContext.hpp"


SupportContext::SupportContext() {}

SupportContext::~SupportContext() {}

std::vector<PlatformClassEnum> SupportContext::PlatformClass() const {
    std::vector<PlatformClassEnum> retPlatforms;

    // LwSwitch is not supported on iGPU, so return dGPU platform.
    retPlatforms.push_back(PLATFORM_CLASS_DGPU);

    return retPlatforms;
}
