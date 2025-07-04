// This code contains LWPU Confidential Information and is disclosed to you
// under a form of LWPU software license agreement provided separately to you.
//
// Notice
// LWPU Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and
// any modifications thereto. Any use, reproduction, disclosure, or
// distribution of this software and related documentation without an express
// license agreement from LWPU Corporation is strictly prohibited.
//
// ALL LWPU DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". LWPU MAKES
// NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Information and code furnished is believed to be accurate and reliable.
// However, LWPU Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of LWPU Corporation. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// LWPU Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// LWPU Corporation.
//
// Copyright 2015 LWPU Corporation. All rights reserved.

#pragma once
#include <ansel/Defines.h>

namespace ansel
{
    struct SessionConfiguration
    {
        // User can move the camera during session
        bool isTranslationAllowed;
        // Camera can be rotated during session
        bool isRotationAllowed;
        // FoV can be modified during session
        bool isFovChangeAllowed;
        // Game is paused during session
        bool isPauseAllowed;
        // Game allows highres capture during session
        bool isHighresAllowed;
        // Game allows 360 capture during session
        bool is360MonoAllowed;
        // Game allows 360 stereo capture during session
        bool is360StereoAllowed;

        SessionConfiguration()
        {
            isTranslationAllowed = true;
            isRotationAllowed = true;
            isFovChangeAllowed = true;
            isPauseAllowed = true;
            isHighresAllowed = true;
            is360MonoAllowed = true;
            is360StereoAllowed = true;
        }
    };

    enum StartSessionStatus
    {
        kDisallowed = 0,
        kAllowed
    };

    typedef StartSessionStatus(*StartSessionCallback)(SessionConfiguration& settings, void* userPointer);
    typedef void(*StopSessionCallback)(void* userPointer);
    typedef void(*StartCaptureCallback)(void* userPointer);
    typedef void(*StopCaptureCallback)(void* userPointer);

    // Stops current session if there is one active
    ANSEL_SDK_API void stopSession();
}