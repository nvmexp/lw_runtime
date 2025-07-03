/*
 * Copyright (c) 2015-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#pragma once

#include <cstdint>

#if defined(LW_HOS)
  #include "lwos.h"
#endif

namespace LwnUtil
{
    class Timer
    {
    private:
        int m_frequency;

        Timer();
        Timer(Timer const&);
        ~Timer();
    public:
        static Timer* instance();
        double ticksToSecs(uint64_t t);
        uint64_t getTicks();
    };
};
