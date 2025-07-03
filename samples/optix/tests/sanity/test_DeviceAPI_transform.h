
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

#include <vector>

const float EPSILON = 0.1f;

const float TEST_TRANSFORM[12] = { 2.f, 0.f, 0.f, 1.f, 0.f, 0.f, 3.f, 2.f, 0.f, -2.f, 1.f, 3.f };

const std::vector<float3> TEST_VALUES = {

    { 2.f, 5.f, 7.f },
    { 0.f, 0.f, 0.f },
    { 1.f, 1.f, 1.f } };

const std::vector<float3> EXPECTED_POINT_OBJECT_WORLD_VALUES = {

    { 5.f, 23.f, 0.f },
    { 1.f, 2.f, 3.f },
    { 3.f, 5.f, 2.f } };

const std::vector<float3> EXPECTED_POINT_WORLD_OBJECT_VALUES = {

    { 0.5f, -1.5f, 1.f },
    { -0.5f, 1.16f, -0.66f },
    { 0.0f, 0.83f, -0.33f } };

const std::vector<float3> EXPECTED_VECTOR_OBJECT_WORLD_VALUES = {

    { 4.f, 21.f, -3.f },
    { 0.f, 0.f, 0.f },
    { 2.f, 3.f, -1.f } };

const std::vector<float3> EXPECTED_VECTOR_WORLD_OBJECT_VALUES = {

    { 1.f, -2.6f, 1.6f },
    { 0.f, 0.f, 0.f },
    { 0.5f, -0.33f, 0.33f } };

const std::vector<float3> EXPECTED_NORMAL_OBJECT_WORLD_VALUES = {

    { 1.f, 3.16f, -2.5f },
    { 0.f, 0.f, 0.f },
    { 0.5f, 0.5f, -0.5f } };

const std::vector<float3> EXPECTED_NORMAL_WORLD_OBJECT_VALUES = {

    { 4.f, -14.f, 22.f },
    { 0.f, 0.f, 0.f },
    { 2.f, -2.f, 4.f } };

// The object2world matrix is equals to the given TEST_TRANSFORM
const std::vector<float4> EXPECTED_MATRIX_OBJECT_WORLD_VALUES = {

    { TEST_TRANSFORM[0], TEST_TRANSFORM[1], TEST_TRANSFORM[2], TEST_TRANSFORM[3] },
    { TEST_TRANSFORM[4], TEST_TRANSFORM[5], TEST_TRANSFORM[6], TEST_TRANSFORM[7] },
    { TEST_TRANSFORM[8], TEST_TRANSFORM[9], TEST_TRANSFORM[10], TEST_TRANSFORM[11] } };

// The world2object matrix is equals to the ilwerse TEST_TRANSFORM
const std::vector<float4> EXPECTED_MATRIX_WORLD_OBJECT_VALUES = {

    { .5f, 0.f, 0.f, -.5f },
    { 0.f, 1.f / 6.f, -.5f, 7.f / 6.f },
    { 0.f, 1.f / 3.f, 0.f, -2.f / 3.f } };


// Transform device functions are valid for IS, AH, and CH programs.
enum OptixProgramTypeTransform
{
    OPTIX_PROGRAM_TYPE_INTERSECTION,
    OPTIX_PROGRAM_TYPE_ANY_HIT,
    OPTIX_PROGRAM_TYPE_CLOSEST_HIT,
};

// Object to world are even.
// World to object are odd.
enum TransformType
{
    TRANSFORM_TYPE_POINT_FROM_OBJECT_TO_WORLD,
    TRANSFORM_TYPE_POINT_FROM_WORLD_TO_OBJECT,
    TRANSFORM_TYPE_VECTOR_FROM_OBJECT_TO_WORLD,
    TRANSFORM_TYPE_VECTOR_FROM_WORLD_TO_OBJECT,
    TRANSFORM_TYPE_NORMAL_FROM_OBJECT_TO_WORLD,
    TRANSFORM_TYPE_NORMAL_FROM_WORLD_TO_OBJECT,
    TRANSFORM_TYPE_MATRIX_FROM_OBJECT_TO_WORLD,
    TRANSFORM_TYPE_MATRIX_FROM_WORLD_TO_OBJECT
};

struct Params
{
    unsigned int              numTestValues;
    OptixProgramTypeTransform programType;
    TransformType             transformType;
    float3*                   testValues;
    float3*                   outputValues;
    float4*                   outputMatrixValues;
    OptixTraversableHandle    handle;
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    char* covered;
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
};

// Do we have to deal with a transformation or transform matrix retrieval call?
inline __host__ __device__ bool isTransformationCall( TransformType transformType )
{
    return transformType != TRANSFORM_TYPE_MATRIX_FROM_OBJECT_TO_WORLD && transformType != TRANSFORM_TYPE_MATRIX_FROM_WORLD_TO_OBJECT;
}
