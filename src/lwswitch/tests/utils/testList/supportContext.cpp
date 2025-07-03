/* _LWRM_COPYRIGHTBEGIN
 * *
 * * Copyright 2019 by LWPU Corporation. All rights reserved. All
 * * information contained herein is proprietary and confidential to LWPU
 * * Corporation. Any use, reproduction, or disclosure without the written
 * * permission of LWPU Corporation is prohibited.
 * *
 * * _LWRM_COPYRIGHTEND
 * */

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
