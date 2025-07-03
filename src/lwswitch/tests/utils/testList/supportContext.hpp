/* _LWRM_COPYRIGHTBEGIN
 * *
 * * Copyright 2019 by LWPU Corporation. All rights reserved. All
 * * information contained herein is proprietary and confidential to LWPU
 * * Corporation. Any use, reproduction, or disclosure without the written
 * * permission of LWPU Corporation is prohibited.
 * *
 * * _LWRM_COPYRIGHTEND
 * */

#ifndef SUPPORT_CONTEXT_HPP
#define SUPPORT_CONTEXT_HPP

#include <memory>
#include <vector>

enum PlatformClassEnum {
    PLATFORM_CLASS_IGPU,
    PLATFORM_CLASS_DGPU
};

/// A bundle of information the support checkers are allowed to reference
class SupportContext {
  public:
    SupportContext();
    virtual ~SupportContext();

    std::vector<PlatformClassEnum> PlatformClass() const;
};

#endif // SUPPPORT_CONTEXT_HPP
