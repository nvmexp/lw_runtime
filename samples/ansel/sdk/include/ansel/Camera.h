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
#include <lw/Vec3.h>
#include <lw/Quat.h>

namespace ansel
{
    struct Camera
    {
        // Position of camera, in the game's coordinate space
        lw::Vec3 position;
        // Rotation of the camera, in the game's coordinate space. I.e. if you apply this
        // rotation to the default orientation of the game's camera you will get the current
        // orientation of the camera (again, in game's coordinate space)
        lw::Quat rotation;
        // Field of view in degrees. This value is either vertical or horizontal field of
        // view based on the 'fovType' setting passed in with setConfiguration.
        float fov;
        // The amount that the projection matrix needs to be offset by. These values are
        // applied directly as translations to the projection matrix. These values are only
        // non-zero during Highres capture.
        float projectionOffsetX, projectionOffsetY;
    };

    // Must be called on every frame an Ansel session is active. The 'camera' must contain
    // the current display camera settings when called. After calling 'camera' will contain the
    // new requested camera from Ansel.
    ANSEL_SDK_API void updateCamera(Camera& camera);


}

