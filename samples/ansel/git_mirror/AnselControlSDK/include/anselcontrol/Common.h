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

namespace anselcontrol
{
    enum ShotType
    {
        // regular screenshot
        kShotTypeRegular,
        // shot that is several times larger than the regular screenshot
        kShotTypeSuperResolution,
        // spherical panorama
        kShotType360Mono,
        // regular screenshot stereo pair
        kShotTypeStereo,
        // spherical panorama stereo pair
        kShotType360Stereo
    };

    struct ShotDescription
    {
        // type of the shot
        ShotType shotType;

        union
        {
            size_t superresMult;
            size_t sphericalResolution;
        };
    };

    enum Status
    {
        // Ansel Control SDK is operational
        kControlSuccess,
        // Operation failed
        kControlFailed,
        // Ansel Control SDK wasn't yet initialized from the Ansel side
        kControlNotInitialized,
        // Unsupported driver version
        kControlDriverVersionMismatch
    };

    enum CaptureProgressStatus
    {
        // Ansel capture started
        kCaptureStarted,
        // Ansel took one shot of the requested sequence
        kCaptureShotTaken,
        // Ansel capture has stopped, either due to error, or sequence finished
        kCaptureStopped,
        // Ansel capture processing finished
        kCaptureProcessed
    };
}
