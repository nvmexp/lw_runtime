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
