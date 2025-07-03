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
// Copyright 2016 LWPU Corporation. All rights reserved.

#pragma once
#include <anselcontrol/Defines.h>
#include <anselcontrol/Version.h>
#include <anselcontrol/Common.h>
#include <stdint.h>

namespace anselcontrol
{
    enum SetConfigurationStatus
    {
        // successfully initialized the Ansel SDK
        kSetConfigurationSuccess,
        // the Ansel Control SDK wasn't yet initialized on the Ansel side
        kSetConfigurationNotInitialized,
        // the version provided in the Configuration structure is not the same as the one stored inside the SDK binary (header/binary mismatch)
        kSetConfigurationIncompatibleVersion,
        // the Configuration structure supplied for the setConfiguration call is not consistent
        kSetConfigurationIncorrectConfiguration,
        // the Ansel SDK is delay loaded and setConfiguration is called before the SDK is actually loaded
        kSetConfigurationSdkNotLoaded
    };

    typedef void(__cdecl *ReadyCallback)(void* userPointer);
    typedef void(__cdecl *CaptureProgressCallback)(CaptureProgressStatus, int, void* userPointer);

    struct Configuration
    {
        // User defined pointer which is then passed to all the callbacks (nullptr by default)
        void* userPointer;

        bool exclusiveMode;
        ReadyCallback readyCallback;
        CaptureProgressCallback captureProgressCallback;

        // Holds the sdk version, doesn't require modifications
        uint64_t sdkVersion;

        Configuration()
        {
            userPointer = nullptr;
            exclusiveMode = false;
            readyCallback = nullptr;
            captureProgressCallback = nullptr;
            sdkVersion = ANSEL_CONTROL_SDK_VERSION;
        }
    };

    // Called during startup by the game. See 'Configuration' for further documentation.
    ANSEL_CONTROL_SDK_API SetConfigurationStatus setConfiguration(const Configuration& cfg);
    
    // Functions to acquire last capture absolute path; lock retreives UTF8 string pointer, and unlock ilwalidates the pointer
    ANSEL_CONTROL_SDK_API Status lockLastCaptureAbsolutePath(const char ** utf8Path);
    ANSEL_CONTROL_SDK_API Status unlockLastCaptureAbsolutePath();
}
