//
//  Copyright (c) 2021 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#pragma once

#include <optix_types.h>

enum OptixProgramTypeSphereHit
{
    OPTIX_PROGRAM_TYPE_ANY_HIT,
    OPTIX_PROGRAM_TYPE_CLOSEST_HIT
};

enum ExpectedSphereHitType
{
    EXPECTED_SPHERE_HIT_TYPE_FRONT_FACE_HIT,
    EXPECTED_SPHERE_HIT_TYPE_BACK_FACE_HIT
};

enum UseHitKindArgument
{
    USE_IMPLICIT_HIT_KIND,
    USE_HIT_KIND_ARGUMENT
};

// Unique ID for sphere hit result to be sent back to host for verification.
enum SphereHitId
{
    SPHERE_HIT_ID_FRONT = 0xfeedface,
    SPHERE_HIT_ID_BACK  = 0xdeadbeef,
    SPHERE_HIT_ID_NONE  = 0xffffffff
};

struct Params
{
    OptixProgramTypeSphereHit optixProgramType;
    ExpectedSphereHitType     expectedSphereHitType;
    UseHitKindArgument        useHitKindArgument;
    SphereHitId*              d_sphereHitResultOutPointer;
    OptixPrimitiveType*       d_optixPrimitiveTypeResultOutPointer;
    OptixTraversableHandle    handle;
    float*                    d_sphereSecondHit;
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    char* covered;
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
};
