/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#pragma once

#include <lwn/lwn.h>
#include <lwn/lwn_Cpp.h>
#include <lwn/lwn_CppMethods.h>

#if defined(LW_HOS)
#include <nn/vi/vi_Types.h>
#endif

namespace llgd_lwn
{
class DisplayUtil {
public:
    void Initialize();
    void Finalize();
    LWNnativeWindow GetNativeWindow() { return _nativeWindow; }

private:
    LWNnativeWindow _nativeWindow;
#if defined(LW_HOS)
    nn::vi::Display* _display;
    nn::vi::Layer* _displayLayer;
#endif
};
} // llgd_lwn
