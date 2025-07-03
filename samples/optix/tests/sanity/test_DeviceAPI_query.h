//
//  Copyright (c) 2020 LWPU Corporation.  All rights reserved.
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

enum OptixProgramTypeQuery
{
    OPTIX_PROGRAM_TYPE_RAYGEN,
    OPTIX_PROGRAM_TYPE_INTERSECTION,
    OPTIX_PROGRAM_TYPE_ANY_HIT,
    OPTIX_PROGRAM_TYPE_CLOSEST_HIT,
    OPTIX_PROGRAM_TYPE_MISS,
    OPTIX_PROGRAM_TYPE_EXCEPTION
};

// as OptixTraversableType is lacking the instance transform
enum OptixTraversableTypeExtended
{
    OPTIX_TRAVERSABLE_TYPE_EXTENDED_STATIC_TRANSFORM,
    OPTIX_TRAVERSABLE_TYPE_EXTENDED_MATRIX_MOTION_TRANSFORM,
    OPTIX_TRAVERSABLE_TYPE_EXTENDED_SRT_MOTION_TRANSFORM,
    OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM
};

struct Matrix
{
    explicit Matrix( const float* f )
    {
        float* p = m;
        for( unsigned int i = 0; i < 12; ++i )
            *p++ = *f++;
    }
    Matrix( float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, float f10, float f11 )
    {
        m[0]  = f0;
        m[1]  = f1;
        m[2]  = f2;
        m[3]  = f3;
        m[4]  = f4;
        m[5]  = f5;
        m[6]  = f6;
        m[7]  = f7;
        m[8]  = f8;
        m[9]  = f9;
        m[10] = f10;
        m[11] = f11;
    }
    explicit Matrix( float4* f )
    {
        for( unsigned int i = 0; i < 3; ++i )
        {
            float4 v     = f[i];
            m[i * 4]     = v.x;
            m[i * 4 + 1] = v.y;
            m[i * 4 + 2] = v.z;
            m[i * 4 + 3] = v.w;
        }
    }
    float m[12];
};

#define OPTIX_TEST_INSTANCES_COUNT 5

struct Results
{
    int                    instanceIdOut;
    int                    ilwalidInstanceIdOut;
    OptixTraversableHandle instanceChildHandleOut;
    OptixTraversableHandle instanceChild2HandleOut;
    int                    instanceTraversableIdsOut[OPTIX_TEST_INSTANCES_COUNT];
    uint3                  launchDimensionsOut;
    uint3                  launchIndexOut;
    int                    primIndexOut;
    int                    instanceIndexOut;
    bool                   setUndefinedValueSuccessfully;
    bool                   transformationsEqual;
    int                    sbtData;
    int                    sbtGASIndex;
};

struct Params
{
    OptixProgramTypeQuery  optixProgramType;
    OptixTraversableHandle handle;
    bool                   testQuery;  // test original set of tests
    bool                   testQueryInst;
    bool                   testGetInstanceIdFromHandle;
    bool                   testGetInstanceChildFromHandle;
    bool                   testTransformData;  // test transformFromHandle
    bool                   testSBTData;
    bool                   testGetInstanceTraversableFromIAS;
    Results*               testResultsOut;
    OptixTraversableHandle transformHandle;
    float*                 transformData;
    unsigned int           transformDataCount;
    int                    instanceTraversablesCount;
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    char* covered;
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
};
