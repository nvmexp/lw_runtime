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

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <exception>
#include <exptest/exptest.h>
#include <exptest/loggingHelpers.h>
#include <exptest/testTools.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

#include "CommonAsserts.h"

using namespace testing;

#include "test_DeviceAPI_query.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <tests/sanity/test_DeviceAPI_query_ptx_bin.h>

// Expected values.
const int   EXPECTED_ID              = 123;
const int   EXPECTED_PARENT_ID       = 321;
const int   EXPECTED_ILWALID_ID      = 0;
const uint3 EXPECTED_DIMS            = { 1, 1, 1 };
const uint3 EXPECTED_INDEX           = { 0, 0, 0 };
const int   EXPECTED_PRIMITIVE_INDEX = 2;
const int   EXPECTED_SBT_DATA        = 55;
const int   EXPECTED_SBT_GAS_INDEX   = 4;

using RaygenSbtRecord    = SbtRecord<unsigned int>;
using ExceptionSbtRecord = SbtRecord<unsigned int>;
using MissSbtRecord      = SbtRecord<unsigned int>;
using HitgroupSbtRecord  = SbtRecord<unsigned int>;
using CallableSbtRecord  = SbtRecord<void>;

//
// Setting up a custom primitives GAS (three aabbs, one around the origin which gets hit),
// a transform level according to the parameter OptixTraversableTypeExtended and one outer instance IAS.
// This forms the base setup for the following tests.
//
using QueryBaseTestParams = std::tuple<OptixProgramTypeQuery, OptixTraversableTypeExtended>;

class O7_API_Device_Query_Base : public testing::Test, public testing::WithParamInterface<QueryBaseTestParams>
{
  public:
    static void SetUpTestCase();
    static void TearDownTestCase();

    void SetUp() override
    {
        m_pipelineCompileOptions = OptixPipelineCompileOptions{};
        m_params = Params{};
    }
    void TearDown() override
    {
        for( LWdeviceptr ptr : m_toBeFreed )
            LWDA_CHECK( lwdaFree( (void*)ptr ) );
        m_toBeFreed.clear();
        m_instanceTraversableIdsOut.clear();
        m_instanceTraversableIdsOriginal.clear();
    }

    void                   runTest( OptixTraversableHandle newIasHandle = 0 );
    OptixTraversableHandle buildTestIAS();
    const static int       s_sbtRecordCount = 6;

  private:
    static OptixDeviceContext   s_context;
    static OptixRecordingLogger s_logger;
    static LWdeviceptr          s_d_gasOutputBuffer;

    // in attempt to reduce run time we generate all transform IAS once and keep them around, ie static
    static OptixTraversableHandle s_staticTransformTraversableHandle;
    static OptixTraversableHandle s_matrixTransformTraversableHandle;
    static OptixTraversableHandle s_srtTransformTraversableHandle;
    static OptixTraversableHandle s_instanceTransformTraversableHandle;
    static OptixTraversableHandle s_instanceTransformTraversableHandle2;

    static LWdeviceptr s_d_iasOutputBuffer;
    static LWdeviceptr s_d_iasOutputBuffer2;
    static LWdeviceptr s_d_staticTransform;
    static LWdeviceptr s_d_matrixMotionTransform;
    static LWdeviceptr s_d_srtMotionTransform;

    OptixTraversableHandle m_traversableHandle{};
    OptixTraversableHandle m_iasHandle{};

    OptixModule                 m_ptxModule;
    OptixPipelineCompileOptions m_pipelineCompileOptions;

    std::vector<LWdeviceptr> m_toBeFreed;

    unsigned int copyTransformData();

  protected:
    Params  m_params;
    Results m_testResultsOut;

    static const std::vector<float>        m_matrixTransforms;
    static const Matrix                    m_staticTransform;
    static const Matrix                    m_staticIlwTransform;
    static const std::vector<OptixSRTData> m_srtTransforms;
    static OptixTraversableHandle          s_gasHandle;

    std::vector<unsigned int> m_instanceTraversableIdsOut;
    std::vector<unsigned int> m_instanceTraversableIdsOriginal;
};

const int              O7_API_Device_Query_Base::s_sbtRecordCount;
OptixDeviceContext     O7_API_Device_Query_Base::s_context;
OptixRecordingLogger   O7_API_Device_Query_Base::s_logger;
LWdeviceptr            O7_API_Device_Query_Base::s_d_gasOutputBuffer{};
OptixTraversableHandle O7_API_Device_Query_Base::s_staticTransformTraversableHandle{};
OptixTraversableHandle O7_API_Device_Query_Base::s_matrixTransformTraversableHandle{};
OptixTraversableHandle O7_API_Device_Query_Base::s_srtTransformTraversableHandle{};
OptixTraversableHandle O7_API_Device_Query_Base::s_instanceTransformTraversableHandle{};
OptixTraversableHandle O7_API_Device_Query_Base::s_instanceTransformTraversableHandle2{};

// clang-format off
const std::vector<float> O7_API_Device_Query_Base::m_matrixTransforms = {
    // staticTransform
    1.0, 0.0, 0.0,-1.f,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    // staticIlwTransform
    1.0, 0.0, 0.0,+1.f,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0
};
// clang-format on
const Matrix O7_API_Device_Query_Base::m_staticTransform( &O7_API_Device_Query_Base::m_matrixTransforms[0] );
const Matrix O7_API_Device_Query_Base::m_staticIlwTransform( &O7_API_Device_Query_Base::m_matrixTransforms[12] );

const std::vector<OptixSRTData> O7_API_Device_Query_Base::m_srtTransforms = {
    { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -.1f, 0.0, 0.0 },
    { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, +.1f, 0.0, 0.0 },
};
LWdeviceptr            O7_API_Device_Query_Base::s_d_iasOutputBuffer{};
LWdeviceptr            O7_API_Device_Query_Base::s_d_iasOutputBuffer2{};
LWdeviceptr            O7_API_Device_Query_Base::s_d_staticTransform{};
LWdeviceptr            O7_API_Device_Query_Base::s_d_matrixMotionTransform{};
LWdeviceptr            O7_API_Device_Query_Base::s_d_srtMotionTransform{};
OptixTraversableHandle O7_API_Device_Query_Base::s_gasHandle{};

void O7_API_Device_Query_Base::SetUpTestCase()
{
    exptest::lwdaInitialize();
    OPTIX_CHECK( optixInit() );

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &OptixRecordingLogger::callback;
    options.logCallbackData           = &s_logger;
    options.logCallbackLevel          = 2;
    LWcontext lwCtx                   = 0;

    OPTIX_CHECK( optixDeviceContextCreate( lwCtx, &options, &s_context ) );

    // GAS
    //
    OptixBuildInput gasInput{};

    const unsigned int primCount = 3;
    // for the sake of testing we have here 3 aabbs even though only the third one will ever be hit
    // clang-format off
    OptixAabb   aabbs[primCount] = {
        { -15.0f,-15.0f,-15.0f,-14.0f,-14.0f,-14.0f },
        {  10.0f, 10.0f, 10.0f, 11.0f, 11.0f, 11.0f },
        {  -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f } };
    // clang-format on

    // The first two indices reference geometry that will not be hit due to their bounds as
    // specified in the first two aabbs above. The indices therefore do not need to reference
    // any particular SBT entry, but just need to be within the valid range of SBT indices,
    // ie <s_sbtRecordCount.
    unsigned int sbtIndices[primCount] = { 0, 5, EXPECTED_SBT_GAS_INDEX };
    LWdeviceptr  d_aabbs;
    LWDA_CHECK( lwdaMalloc( (void**)&d_aabbs, primCount * sizeof( OptixAabb ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_aabbs, &aabbs, primCount * sizeof( OptixAabb ), lwdaMemcpyHostToDevice ) );
    LWdeviceptr d_sbtIndices;
    LWDA_CHECK( lwdaMalloc( (void**)&d_sbtIndices, primCount * sizeof( unsigned int ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_sbtIndices, &sbtIndices, primCount * sizeof( unsigned int ), lwdaMemcpyHostToDevice ) );

    gasInput.type                                           = OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES;
    gasInput.lwstomPrimitiveArray.aabbBuffers               = &d_aabbs;
    gasInput.lwstomPrimitiveArray.numPrimitives             = primCount;
    unsigned int gasInputFlags[s_sbtRecordCount]            = { OPTIX_GEOMETRY_FLAG_NONE, OPTIX_GEOMETRY_FLAG_NONE,
                                                     OPTIX_GEOMETRY_FLAG_NONE, OPTIX_GEOMETRY_FLAG_NONE,
                                                     OPTIX_GEOMETRY_FLAG_NONE, OPTIX_GEOMETRY_FLAG_NONE };
    gasInput.lwstomPrimitiveArray.flags                     = gasInputFlags;
    gasInput.lwstomPrimitiveArray.numSbtRecords             = s_sbtRecordCount;
    gasInput.lwstomPrimitiveArray.sbtIndexOffsetBuffer      = d_sbtIndices;
    gasInput.lwstomPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof( unsigned int );

    OptixAccelBuildOptions gasAccelOptions = {};
    gasAccelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    gasAccelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &gasAccelOptions, &gasInput, 1, &gasBufferSizes ) );

    LWdeviceptr d_tempBuffer;
    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_tempBuffer ), gasBufferSizes.tempSizeInBytes ) );
    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &s_d_gasOutputBuffer ), gasBufferSizes.outputSizeInBytes ) );

    OptixTraversableHandle gasHandle;
    OPTIX_CHECK( optixAccelBuild( s_context, 0, &gasAccelOptions, &gasInput, 1, d_tempBuffer, gasBufferSizes.tempSizeInBytes,
                                  s_d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes, &gasHandle, nullptr, 0 ) );
    s_gasHandle = gasHandle;
    LWDA_CHECK( lwdaFree( (void*)d_aabbs ) );
    LWDA_CHECK( lwdaFree( (void*)d_sbtIndices ) );
    LWDA_CHECK( lwdaFree( (void*)d_tempBuffer ) );

    // inner transform
    //
    {
        // OPTIX_TRAVERSABLE_TYPE_EXTENDED_STATIC_TRANSFORM
        OptixStaticTransform staticTransform = {};
        staticTransform.child                = gasHandle;
        memcpy( (void*)staticTransform.transform, m_staticTransform.m, 12 * sizeof( float ) );
        memcpy( (void*)staticTransform.ilwTransform, m_staticIlwTransform.m, 12 * sizeof( float ) );

        LWDA_CHECK( lwdaMalloc( (void**)&s_d_staticTransform, sizeof( OptixStaticTransform ) ) );
        LWDA_CHECK( lwdaMemcpy( (void*)s_d_staticTransform, &staticTransform, sizeof( OptixStaticTransform ), lwdaMemcpyHostToDevice ) );

        OPTIX_CHECK( optixColwertPointerToTraversableHandle( s_context, s_d_staticTransform, OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM,
                                                             &s_staticTransformTraversableHandle ) );
    }
    {
        // OPTIX_TRAVERSABLE_TYPE_EXTENDED_MATRIX_MOTION_TRANSFORM
        size_t                     N                     = 2;
        OptixMatrixMotionTransform matrixMotionTransform = {};
        matrixMotionTransform.child                      = gasHandle;
        matrixMotionTransform.motionOptions.numKeys      = static_cast<unsigned short>( N );
        matrixMotionTransform.motionOptions.timeBegin    = 0.0f;
        matrixMotionTransform.motionOptions.timeEnd      = 1.0f;
        matrixMotionTransform.motionOptions.flags        = OPTIX_MOTION_FLAG_NONE;
        memcpy( (void*)&matrixMotionTransform.transform[0], &m_matrixTransforms[0], N * 12 * sizeof( float ) );

        LWDA_CHECK( lwdaMalloc( (void**)&s_d_matrixMotionTransform, sizeof( OptixMatrixMotionTransform ) ) );
        LWDA_CHECK( lwdaMemcpy( (void*)s_d_matrixMotionTransform, &matrixMotionTransform,
                                sizeof( OptixMatrixMotionTransform ), lwdaMemcpyHostToDevice ) );

        OPTIX_CHECK( optixColwertPointerToTraversableHandle( s_context, s_d_matrixMotionTransform, OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
                                                             &s_matrixTransformTraversableHandle ) );
    }
    {
        // OPTIX_TRAVERSABLE_TYPE_EXTENDED_SRT_MOTION_TRANSFORM
        size_t                  N                  = 2;
        OptixSRTMotionTransform srtMotionTransform = {};
        srtMotionTransform.child                   = gasHandle;
        srtMotionTransform.motionOptions.numKeys   = static_cast<unsigned short>( N );
        srtMotionTransform.motionOptions.timeBegin = 0.0f;
        srtMotionTransform.motionOptions.timeEnd   = 1.0f;
        srtMotionTransform.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
        memcpy( &srtMotionTransform.srtData[0], &m_srtTransforms[0], N * sizeof( OptixSRTData ) );

        LWDA_CHECK( lwdaMalloc( (void**)&s_d_srtMotionTransform, sizeof( OptixSRTMotionTransform ) ) );
        LWDA_CHECK( lwdaMemcpy( (void*)s_d_srtMotionTransform, &srtMotionTransform, sizeof( OptixSRTMotionTransform ),
                                lwdaMemcpyHostToDevice ) );

        OPTIX_CHECK( optixColwertPointerToTraversableHandle( s_context, s_d_srtMotionTransform, OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM,
                                                             &s_srtTransformTraversableHandle ) );
    }
    {
        // OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM
        OptixInstance instance = {};
        memcpy( instance.transform, m_staticTransform.m, sizeof( float ) * 12 );

        instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        instance.instanceId        = EXPECTED_ID;
        instance.visibilityMask    = 255;
        instance.sbtOffset         = 0;
        instance.traversableHandle = gasHandle;

        LWdeviceptr d_instance;
        LWDA_CHECK( lwdaMalloc( (void**)&d_instance, sizeof( OptixInstance ) ) );
        LWDA_CHECK( lwdaMemcpy( (void*)d_instance, &instance, sizeof( OptixInstance ), lwdaMemcpyHostToDevice ) );

        OptixBuildInput buildInput            = {};
        buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInput.instanceArray.instances    = d_instance;
        buildInput.instanceArray.numInstances = 1;
        OptixAccelBufferSizes  iasBufferSizes;
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &accelOptions, &buildInput, 1, &iasBufferSizes ) );

        LWdeviceptr d_tempIasBuffer;
        LWDA_CHECK( lwdaMalloc( (void**)&d_tempIasBuffer, iasBufferSizes.tempSizeInBytes ) );
        LWDA_CHECK( lwdaMalloc( (void**)&s_d_iasOutputBuffer, iasBufferSizes.outputSizeInBytes ) );

        OPTIX_CHECK( optixAccelBuild( s_context, nullptr, &accelOptions, &buildInput, 1, d_tempIasBuffer,
                                      iasBufferSizes.tempSizeInBytes, s_d_iasOutputBuffer, iasBufferSizes.outputSizeInBytes,
                                      &s_instanceTransformTraversableHandle, nullptr, 0 ) );

        LWDA_CHECK( lwdaFree( (void*)d_instance ) );
        LWDA_CHECK( lwdaFree( (void*)d_tempIasBuffer ) );
    }
    {
        // setup yet another IAS of instance transform - this time with two instances, the first one far off
        std::vector<OptixInstance> instances;
        {
            // the bad one
            OptixInstance instance = {};
            // clang-format off
            const Matrix staticTransform = {
                1.0, 0.0, 0.0,-100.f,
                0.0, 1.0, 0.0,-100.f,
                0.0, 0.0, 1.0,-100.f };
            // clang-format on
            memcpy( instance.transform, staticTransform.m, sizeof( float ) * 12 );

            instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
            instance.instanceId        = 777;
            instance.visibilityMask    = 255;
            instance.sbtOffset         = 0;
            instance.traversableHandle = gasHandle;
            instances.push_back( instance );
        }
        {
            // the good one
            OptixInstance instance = {};
            memcpy( instance.transform, m_staticTransform.m, sizeof( float ) * 12 );

            instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
            instance.instanceId        = EXPECTED_ID;
            instance.visibilityMask    = 255;
            instance.sbtOffset         = 0;
            instance.traversableHandle = gasHandle;
            instances.push_back( instance );
        }

        LWdeviceptr d_instances;
        LWDA_CHECK( lwdaMalloc( (void**)&d_instances, instances.size() * sizeof( OptixInstance ) ) );
        LWDA_CHECK( lwdaMemcpy( (void*)d_instances, instances.data(), instances.size() * sizeof( OptixInstance ), lwdaMemcpyHostToDevice ) );

        OptixBuildInput buildInput            = {};
        buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInput.instanceArray.instances    = d_instances;
        buildInput.instanceArray.numInstances = instances.size();
        OptixAccelBufferSizes  iasBufferSizes;
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &accelOptions, &buildInput, 1, &iasBufferSizes ) );

        LWdeviceptr d_tempIasBuffer;
        LWDA_CHECK( lwdaMalloc( (void**)&d_tempIasBuffer, iasBufferSizes.tempSizeInBytes ) );
        LWDA_CHECK( lwdaMalloc( (void**)&s_d_iasOutputBuffer2, iasBufferSizes.outputSizeInBytes ) );

        OPTIX_CHECK( optixAccelBuild( s_context, nullptr, &accelOptions, &buildInput, 1, d_tempIasBuffer,
                                      iasBufferSizes.tempSizeInBytes, s_d_iasOutputBuffer2, iasBufferSizes.outputSizeInBytes,
                                      &s_instanceTransformTraversableHandle2, nullptr, 0 ) );

        LWDA_CHECK( lwdaFree( (void*)d_instances ) );
        LWDA_CHECK( lwdaFree( (void*)d_tempIasBuffer ) );
    }
}

void O7_API_Device_Query_Base::TearDownTestCase()
{
    LWDA_CHECK( lwdaFree( (void*)s_d_srtMotionTransform ) );
    LWDA_CHECK( lwdaFree( (void*)s_d_matrixMotionTransform ) );
    LWDA_CHECK( lwdaFree( (void*)s_d_staticTransform ) );
    LWDA_CHECK( lwdaFree( (void*)s_d_iasOutputBuffer ) );
    LWDA_CHECK( lwdaFree( (void*)s_d_iasOutputBuffer2 ) );

    LWDA_CHECK( lwdaFree( (void*)s_d_gasOutputBuffer ) );
    OPTIX_CHECK( optixDeviceContextDestroy( s_context ) );
}

// Utility to return the number of floats of the transformation data of the given type.
unsigned int getTransformFloatCount( OptixTraversableTypeExtended type )
{
    switch( type )
    {
        case OPTIX_TRAVERSABLE_TYPE_EXTENDED_STATIC_TRANSFORM:
            return 2 * sizeof( Matrix ) / sizeof( float );  // both normal and ilwerse matrix
        case OPTIX_TRAVERSABLE_TYPE_EXTENDED_MATRIX_MOTION_TRANSFORM:
            return 2 * sizeof( Matrix ) / sizeof( float );
        case OPTIX_TRAVERSABLE_TYPE_EXTENDED_SRT_MOTION_TRANSFORM:
            return 2 * sizeof( OptixSRTData ) / sizeof( float );
        case OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM:
            return 2 * sizeof( Matrix ) / sizeof( float );  // both normal and ilwerse matrix
        default:
            return 0;
    }
}

unsigned int O7_API_Device_Query_Base::copyTransformData()
{
    unsigned int floatCount = getTransformFloatCount( std::get<1>( GetParam() ) );
    LWDA_CHECK( lwdaMalloc( (void**)&m_params.transformData, floatCount * sizeof( float ) ) );
    switch( std::get<1>( GetParam() ) )
    {
        case OPTIX_TRAVERSABLE_TYPE_EXTENDED_STATIC_TRANSFORM:
        case OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM: {
            // here we copy both the transform matrix and the ilwerse as contiguous memory
            LWDA_CHECK( lwdaMemcpy( (void*)m_params.transformData, &m_matrixTransforms[0], floatCount * sizeof( float ),
                                    lwdaMemcpyHostToDevice ) );
            break;
        }
        case OPTIX_TRAVERSABLE_TYPE_EXTENDED_MATRIX_MOTION_TRANSFORM:
            LWDA_CHECK( lwdaMemcpy( (void*)m_params.transformData, &m_matrixTransforms[0], floatCount * sizeof( float ),
                                    lwdaMemcpyHostToDevice ) );
            break;
        case OPTIX_TRAVERSABLE_TYPE_EXTENDED_SRT_MOTION_TRANSFORM:
            LWDA_CHECK( lwdaMemcpy( (void*)m_params.transformData, &m_srtTransforms[0], floatCount * sizeof( float ),
                                    lwdaMemcpyHostToDevice ) );
            break;
        default:
            floatCount = 0;
    }
    return floatCount;
}

void O7_API_Device_Query_Base::runTest( OptixTraversableHandle newIasHandle )
{
    // testrun-dependent input values
    OptixTraversableHandle traversableHandle = newIasHandle;
    // potentially already build by O7_API_Device_Query_Simple::buildTestIAS()
    if( !traversableHandle )
    {
        switch( std::get<1>( GetParam() ) )
        {
            case OPTIX_TRAVERSABLE_TYPE_EXTENDED_STATIC_TRANSFORM:
                traversableHandle = s_staticTransformTraversableHandle;
                break;
            case OPTIX_TRAVERSABLE_TYPE_EXTENDED_MATRIX_MOTION_TRANSFORM:
                traversableHandle = s_matrixTransformTraversableHandle;
                break;
            case OPTIX_TRAVERSABLE_TYPE_EXTENDED_SRT_MOTION_TRANSFORM:
                traversableHandle = s_srtTransformTraversableHandle;
                break;
            case OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM: {
                if( !m_params.testQueryInst )
                {
                    traversableHandle = s_instanceTransformTraversableHandle;
                }
                else
                {
                    traversableHandle = s_instanceTransformTraversableHandle2;
                }
                break;
            }
            default:
                FAIL() << "We shouldn't get here.";
        }

        OptixInstance instance = {};
        memcpy( (void*)instance.transform, m_staticIlwTransform.m, sizeof( float ) * 12 );

        instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        instance.instanceId        = EXPECTED_PARENT_ID;
        instance.visibilityMask    = 255;
        instance.sbtOffset         = 0;
        instance.traversableHandle = traversableHandle;

        LWdeviceptr d_instance;
        LWDA_CHECK( lwdaMalloc( (void**)&d_instance, sizeof( OptixInstance ) ) );
        LWDA_CHECK( lwdaMemcpy( (void*)d_instance, &instance, sizeof( OptixInstance ), lwdaMemcpyHostToDevice ) );

        OptixBuildInput buildInput            = {};
        buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInput.instanceArray.instances    = d_instance;
        buildInput.instanceArray.numInstances = 1;

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
        if( std::get<1>( GetParam() ) == OPTIX_TRAVERSABLE_TYPE_EXTENDED_MATRIX_MOTION_TRANSFORM
            || std::get<1>( GetParam() ) == OPTIX_TRAVERSABLE_TYPE_EXTENDED_SRT_MOTION_TRANSFORM )
        {
            accelOptions.motionOptions.numKeys   = 2;
            accelOptions.motionOptions.timeBegin = 0.0f;
            accelOptions.motionOptions.timeEnd   = 1.0f;
            accelOptions.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
        }
        OptixAccelBufferSizes iasBufferSizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &accelOptions, &buildInput, 1, &iasBufferSizes ) );

        LWdeviceptr d_iasOutputBuffer{};
        LWdeviceptr d_tempIasBuffer;
        LWDA_CHECK( lwdaMalloc( (void**)&d_tempIasBuffer, iasBufferSizes.tempSizeInBytes ) );
        LWDA_CHECK( lwdaMalloc( (void**)&d_iasOutputBuffer, iasBufferSizes.outputSizeInBytes ) );
        m_toBeFreed.push_back( d_iasOutputBuffer );

        OPTIX_CHECK( optixAccelBuild( s_context, nullptr, &accelOptions, &buildInput, 1, d_tempIasBuffer, iasBufferSizes.tempSizeInBytes,
                                      d_iasOutputBuffer, iasBufferSizes.outputSizeInBytes, &m_iasHandle, nullptr, 0 ) );
        LWDA_SYNC_CHECK();

        LWDA_CHECK( lwdaFree( (void*)d_instance ) );
        LWDA_CHECK( lwdaFree( (void*)d_tempIasBuffer ) );
    }

    // module
    OptixModuleCompileOptions moduleCompileOptions = {};

    if( std::get<1>( GetParam() ) != OPTIX_TRAVERSABLE_TYPE_EXTENDED_MATRIX_MOTION_TRANSFORM
        && std::get<1>( GetParam() ) != OPTIX_TRAVERSABLE_TYPE_EXTENDED_SRT_MOTION_TRANSFORM )
        m_pipelineCompileOptions.usesMotionBlur = false;
    else
        m_pipelineCompileOptions.usesMotionBlur = true;

    m_pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    m_pipelineCompileOptions.numPayloadValues      = 2;
    m_pipelineCompileOptions.numAttributeValues    = 2;
    m_pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_USER | OPTIX_EXCEPTION_FLAG_DEBUG
                                              | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    m_pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    OPTIX_CHECK_THROW( optixModuleCreateFromPTX( s_context, &moduleCompileOptions, &m_pipelineCompileOptions,
                                                 optix::data::gettest_DeviceAPI_querySources()[1],
                                                 optix::data::gettest_DeviceAPI_querySourceSizes()[0], 0, 0, &m_ptxModule ) );

    // program groups
    OptixProgramGroupOptions programGroupOptions = {};

    OptixProgramGroupDesc rgProgramGroupDesc    = {};
    rgProgramGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgProgramGroupDesc.raygen.module            = m_ptxModule;
    rgProgramGroupDesc.raygen.entryFunctionName = "__raygen__";
    OptixProgramGroup rgProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &rgProgramGroupDesc, 1, &programGroupOptions, 0, 0, &rgProgramGroup ) );

    OptixProgramGroupDesc exProgramGroupDesc       = {};
    exProgramGroupDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    exProgramGroupDesc.exception.module            = m_ptxModule;
    exProgramGroupDesc.exception.entryFunctionName = "__exception__";
    OptixProgramGroup exProgramGroup;
    OPTIX_CHECK_THROW( optixProgramGroupCreate( s_context, &exProgramGroupDesc, 1, &programGroupOptions, 0, 0, &exProgramGroup ) );

    OptixProgramGroupDesc msProgramGroupDesc  = {};
    msProgramGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msProgramGroupDesc.miss.module            = m_ptxModule;
    msProgramGroupDesc.miss.entryFunctionName = "__miss__";
    OptixProgramGroup msProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &msProgramGroupDesc, 1, &programGroupOptions, 0, 0, &msProgramGroup ) );

    OptixProgramGroupDesc hitgroupProgramGroupDesc        = {};
    hitgroupProgramGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupProgramGroupDesc.hitgroup.moduleCH            = m_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__";
    hitgroupProgramGroupDesc.hitgroup.moduleIS            = m_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameIS = "__intersection__";
    hitgroupProgramGroupDesc.hitgroup.moduleAH            = m_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameAH = "__anyhit__";
    OptixProgramGroup hitgroupProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &hitgroupProgramGroupDesc, 1, &programGroupOptions, 0, 0, &hitgroupProgramGroup ) );

    OptixPipeline                  pipeline;
    std::vector<OptixProgramGroup> programGroups = { rgProgramGroup, exProgramGroup, msProgramGroup, hitgroupProgramGroup };
    OptixPipelineLinkOptions       pipelineLinkOptions = {};
    const unsigned int             maxTraceDepth       = 5;
    pipelineLinkOptions.maxTraceDepth                  = maxTraceDepth;
    pipelineLinkOptions.debugLevel                     = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OPTIX_CHECK( optixPipelineCreate( s_context, &m_pipelineCompileOptions, &pipelineLinkOptions, &programGroups[0],
                                      programGroups.size(), 0, 0, &pipeline ) );
    // stack size
    OptixStackSizes stackSizes = {};
    for( auto& progGroup : programGroups )
    {
        OPTIX_CHECK( optixUtilAclwmulateStackSizes( progGroup, &stackSizes ) );
    }

    uint32_t           direct_callable_stack_size_from_traversal;
    uint32_t           direct_callable_stack_size_from_state;
    uint32_t           continuation_stack_size;
    const unsigned int maxTraversableDepth = 8;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stackSizes, maxTraceDepth,
                                             0,  // maxCCDepth
                                             0,  // maxDCDepth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state,
                                            continuation_stack_size, maxTraversableDepth ) );

    // SBT
    RaygenSbtRecord rgSBT;
    OPTIX_CHECK( optixSbtRecordPackHeader( rgProgramGroup, &rgSBT ) );
    rgSBT.data = EXPECTED_SBT_DATA;
    LWdeviceptr d_raygenRecord;
    size_t      raygenRecordSize = sizeof( RaygenSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_raygenRecord, raygenRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_raygenRecord, &rgSBT, raygenRecordSize, lwdaMemcpyHostToDevice ) );

    ExceptionSbtRecord exSBT;
    OPTIX_CHECK( optixSbtRecordPackHeader( exProgramGroup, &exSBT ) );
    exSBT.data = EXPECTED_SBT_DATA;
    LWdeviceptr d_exceptionRecord;
    size_t      exceptionRecordSize = sizeof( ExceptionSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_exceptionRecord, exceptionRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_exceptionRecord, &exSBT, exceptionRecordSize, lwdaMemcpyHostToDevice ) );

    // keeping MS sbt records aligned with hit group one's, ie only using the EXPECTED_SBT_GAS_INDEX' one
    MissSbtRecord msSBT[s_sbtRecordCount];
    for( size_t i = 0; i < s_sbtRecordCount; ++i )
    {
        OPTIX_CHECK( optixSbtRecordPackHeader( msProgramGroup, &msSBT[i] ) );
        if( i == EXPECTED_SBT_GAS_INDEX )
            msSBT[i].data = EXPECTED_SBT_DATA;
        else
            msSBT[i].data = 0;
    }
    LWdeviceptr d_missSbtRecord;
    size_t      missSbtRecordSize = s_sbtRecordCount * sizeof( MissSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_missSbtRecord, missSbtRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_missSbtRecord, &msSBT, missSbtRecordSize, lwdaMemcpyHostToDevice ) );

    HitgroupSbtRecord hgSBT[s_sbtRecordCount];
    for( size_t i = 0; i < s_sbtRecordCount; ++i )
    {
        OPTIX_CHECK( optixSbtRecordPackHeader( hitgroupProgramGroup, &hgSBT[i] ) );
        if( i == EXPECTED_SBT_GAS_INDEX )
            hgSBT[i].data = EXPECTED_SBT_DATA;
        else
            hgSBT[i].data = 0;
    }
    LWdeviceptr d_hitgroupSbtRecord;
    size_t      hitgroupSbtRecordSize = s_sbtRecordCount * sizeof( HitgroupSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_hitgroupSbtRecord, hitgroupSbtRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_hitgroupSbtRecord, &hgSBT, hitgroupSbtRecordSize, lwdaMemcpyHostToDevice ) );

    OptixShaderBindingTable sbt     = {};
    sbt.raygenRecord                = d_raygenRecord;
    sbt.exceptionRecord             = d_exceptionRecord;
    sbt.missRecordBase              = d_missSbtRecord;
    sbt.missRecordStrideInBytes     = (unsigned int)sizeof( MissSbtRecord );
    sbt.missRecordCount             = s_sbtRecordCount;
    sbt.hitgroupRecordBase          = d_hitgroupSbtRecord;
    sbt.hitgroupRecordStrideInBytes = (unsigned int)sizeof( HitgroupSbtRecord );
    sbt.hitgroupRecordCount         = s_sbtRecordCount;

    LWstream stream;
    LWDA_CHECK( lwdaStreamCreate( &stream ) );

    // params
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    SETUP_API_CALL_COVERAGE( m_params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    // passing the handle to the transformation to the programs
    // Unfortunately that doesn't work for instance arrays - hence we pass 0 in that case
    if( std::get<1>( GetParam() ) != OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM )
        m_params.transformHandle = traversableHandle;
    else
        m_params.transformHandle = 0;
    LWDA_CHECK( lwdaMalloc( (void**)&m_params.testResultsOut, sizeof( Results ) ) );
    LWDA_CHECK( lwdaMemset( (void*)m_params.testResultsOut, 0, sizeof( Results ) ) );
    if( newIasHandle )
    {
        Results results{};
        m_params.instanceTraversablesCount = OPTIX_TEST_INSTANCES_COUNT;
        LWDA_CHECK( lwdaMemcpy( (void*)m_params.testResultsOut, &results, sizeof( Results ), lwdaMemcpyHostToDevice ) );

        m_params.handle = newIasHandle;
    }
    else
        m_params.handle = m_iasHandle;

    // copy transformation data to the params
    m_params.transformDataCount = copyTransformData();

    LWdeviceptr d_params;
    LWDA_CHECK( lwdaMalloc( (void**)&d_params, sizeof( Params ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_params, &m_params, sizeof( Params ), lwdaMemcpyHostToDevice ) );

    OPTIX_CHECK( optixLaunch( pipeline, stream, d_params, sizeof( Params ), &sbt, 1, 1, 1 ) );
    LWDA_SYNC_CHECK();

    LWDA_CHECK( lwdaMemcpy( (void*)&m_testResultsOut, (void*)m_params.testResultsOut, sizeof( Results ), lwdaMemcpyDeviceToHost ) );
    m_instanceTraversableIdsOut.resize( m_params.instanceTraversablesCount );
    if( newIasHandle )
        for( int i = 0; i < m_params.instanceTraversablesCount; ++i )
            m_instanceTraversableIdsOut[i] = m_testResultsOut.instanceTraversableIdsOut[i];
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    ANALYZE_API_CALL_COVERAGE( m_params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

    LWDA_CHECK( lwdaFree( (void*)d_params ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.transformData ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.testResultsOut ) );
    LWDA_CHECK( lwdaFree( (void*)d_hitgroupSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_missSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_exceptionRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_raygenRecord ) );
}

// Putting all transforms into an instance array as one IAS. This will allow to check the traversable handles.
OptixTraversableHandle O7_API_Device_Query_Base::buildTestIAS()
{
    // clang-format off
    const std::vector<float> identityMatrix = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    };
    // clang-format on
    std::vector<OptixInstance> instances( OPTIX_TEST_INSTANCES_COUNT, OptixInstance{} );
    m_instanceTraversableIdsOriginal.clear();
    for( size_t i = 0; i < OPTIX_TEST_INSTANCES_COUNT; ++i )
    {
        instances[i].flags          = OPTIX_INSTANCE_FLAG_NONE;
        instances[i].instanceId     = 123 + i;
        instances[i].sbtOffset      = 0;
        instances[i].visibilityMask = 1;
        memcpy( instances[i].transform, identityMatrix.data(), sizeof( float ) * 12 );
        // for the sake of easing hit computation etc re-use/instantiate the same existing transformation
        instances[i].traversableHandle = s_instanceTransformTraversableHandle;
        m_instanceTraversableIdsOriginal.push_back( instances[i].instanceId );
    }
    LWdeviceptr d_instances;
    size_t      instancesSizeInBytes = instances.size() * sizeof( OptixInstance );
    LWDA_CHECK( lwdaMalloc( (void**)&d_instances, instancesSizeInBytes ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_instances, instances.data(), instancesSizeInBytes, lwdaMemcpyHostToDevice ) );

    OptixBuildInput instanceBuildInput{};
    instanceBuildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceBuildInput.instanceArray.instances    = d_instances;
    instanceBuildInput.instanceArray.numInstances = static_cast<unsigned int>( instances.size() );

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    if( std::get<1>( GetParam() ) == OPTIX_TRAVERSABLE_TYPE_EXTENDED_MATRIX_MOTION_TRANSFORM
        || std::get<1>( GetParam() ) == OPTIX_TRAVERSABLE_TYPE_EXTENDED_SRT_MOTION_TRANSFORM )
    {
        accelOptions.motionOptions.numKeys   = 2;
        accelOptions.motionOptions.timeBegin = 0.0f;
        accelOptions.motionOptions.timeEnd   = 1.0f;
        accelOptions.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
    }
    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &accelOptions, &instanceBuildInput, 1, &iasBufferSizes ) );

    LWdeviceptr d_iasOutputBuffer{};
    LWdeviceptr d_tempIasBuffer;
    LWDA_CHECK( lwdaMalloc( (void**)&d_tempIasBuffer, iasBufferSizes.tempSizeInBytes ) );
    LWDA_CHECK( lwdaMalloc( (void**)&d_iasOutputBuffer, iasBufferSizes.outputSizeInBytes ) );
    m_toBeFreed.push_back( d_iasOutputBuffer );

    OptixTraversableHandle newIasHandle;
    OPTIX_CHECK( optixAccelBuild( s_context, nullptr, &accelOptions, &instanceBuildInput, 1, d_tempIasBuffer,
                                  iasBufferSizes.tempSizeInBytes, d_iasOutputBuffer, iasBufferSizes.outputSizeInBytes,
                                  &newIasHandle, nullptr, 0 ) );
    LWDA_SYNC_CHECK();

    LWDA_CHECK( lwdaFree( (void*)d_instances ) );
    LWDA_CHECK( lwdaFree( (void*)d_tempIasBuffer ) );

    return newIasHandle;
}


class O7_API_Device_Query_Simple : public O7_API_Device_Query_Base
{
  public:
    void SetUp() override
    {
        O7_API_Device_Query_Base::SetUp();
        m_params.testQuery        = true;
        m_params.optixProgramType = std::get<0>( GetParam() );
    }
    void TearDown() override {}
};

TEST_P( O7_API_Device_Query_Simple, testGetInstanceIdFromHandle )
{
    m_params.testQuery                   = false;
    m_params.testGetInstanceIdFromHandle = true;
    runTest();

    EXPECT_EQ( EXPECTED_ID, m_testResultsOut.instanceIdOut );
    EXPECT_EQ( EXPECTED_ILWALID_ID, m_testResultsOut.ilwalidInstanceIdOut );
}

TEST_P( O7_API_Device_Query_Simple, testGetInstanceChildFromHandle )
{
    m_params.testQuery                      = false;
    m_params.testGetInstanceChildFromHandle = true;
    runTest();

    EXPECT_EQ( s_gasHandle, m_testResultsOut.instanceChildHandleOut );
    // with current AS that tests "Returns 0 if the traversable handle does not reference an OptixInstance"
    // as it returned a handle to the geometry
    EXPECT_EQ( 0, m_testResultsOut.instanceChild2HandleOut );
}

TEST_P( O7_API_Device_Query_Simple, testGetLaunchDimensions )
{
    runTest();

    EXPECT_EQ( EXPECTED_DIMS.x, m_testResultsOut.launchDimensionsOut.x );
    EXPECT_EQ( EXPECTED_DIMS.y, m_testResultsOut.launchDimensionsOut.y );
    EXPECT_EQ( EXPECTED_DIMS.z, m_testResultsOut.launchDimensionsOut.z );
}

TEST_P( O7_API_Device_Query_Simple, testGetLaunchIndex )
{
    runTest();

    EXPECT_EQ( EXPECTED_INDEX.x, m_testResultsOut.launchIndexOut.x );
    EXPECT_EQ( EXPECTED_INDEX.y, m_testResultsOut.launchIndexOut.y );
    EXPECT_EQ( EXPECTED_INDEX.z, m_testResultsOut.launchIndexOut.z );
}

// Does this program type allow usage of optixUndefinedValue()?
bool supportsGetUndefinedValue( OptixProgramTypeQuery programType )
{
    return programType == OPTIX_PROGRAM_TYPE_INTERSECTION || programType == OPTIX_PROGRAM_TYPE_ANY_HIT
           || programType == OPTIX_PROGRAM_TYPE_CLOSEST_HIT || programType == OPTIX_PROGRAM_TYPE_MISS;
}

TEST_P( O7_API_Device_Query_Simple, testUndefinedValue )
{
    runTest();

    if( supportsGetUndefinedValue( m_params.optixProgramType ) )
        EXPECT_TRUE( m_testResultsOut.setUndefinedValueSuccessfully );
    else
        EXPECT_FALSE( m_testResultsOut.setUndefinedValueSuccessfully );
}

// Does this program type allow usage of optixGetPrimitiveIndex()?
bool supportsGetPrimitiveIndex( OptixProgramTypeQuery programType )
{
    return programType == OPTIX_PROGRAM_TYPE_INTERSECTION || programType == OPTIX_PROGRAM_TYPE_ANY_HIT
           || programType == OPTIX_PROGRAM_TYPE_CLOSEST_HIT || programType == OPTIX_PROGRAM_TYPE_EXCEPTION;
}

TEST_P( O7_API_Device_Query_Simple, testGetPrimitiveIndex )
{
    if( !supportsGetPrimitiveIndex( m_params.optixProgramType ) )
        return;
    runTest();

    if( m_params.optixProgramType != OPTIX_PROGRAM_TYPE_EXCEPTION )
        EXPECT_EQ( EXPECTED_PRIMITIVE_INDEX, m_testResultsOut.primIndexOut );
    else
        // Testing for 0 is not 100% correct, but as we don't throw this specific OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_HIT_SBT...
        // According to the API doc:
        // "In EX with exception code OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_HIT_SBT corresponds
        //  to the active primitive index. Returns zero for all other exceptions."
        EXPECT_EQ( 0, m_testResultsOut.primIndexOut );
}

TEST_P( O7_API_Device_Query_Simple, testGetInstanceTraversableFromIAS )
{
    m_params.testQuery                         = false;
    m_params.testGetInstanceTraversableFromIAS = true;
    // build up a(n arbitrary) test IAS with OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS
    runTest( buildTestIAS() );

    EXPECT_EQ( m_instanceTraversableIdsOriginal, m_instanceTraversableIdsOut );
}

INSTANTIATE_TEST_SUITE_P( TestSimpleQueryCalls,
                          O7_API_Device_Query_Simple,
                          testing::Combine( testing::Values( OPTIX_PROGRAM_TYPE_RAYGEN,
                                                             OPTIX_PROGRAM_TYPE_INTERSECTION,
                                                             OPTIX_PROGRAM_TYPE_ANY_HIT,
                                                             OPTIX_PROGRAM_TYPE_CLOSEST_HIT,
                                                             OPTIX_PROGRAM_TYPE_MISS,
                                                             OPTIX_PROGRAM_TYPE_EXCEPTION ),
                                            testing::Values( OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM ) ),
                          []( const testing::TestParamInfo<O7_API_Device_Query_Base::ParamType>& info ) {
                              std::string name = "Q_INSTANCE";
                              switch( std::get<0>( info.param ) )
                              {
                                  case OPTIX_PROGRAM_TYPE_RAYGEN:
                                      name += "_RG";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_INTERSECTION:
                                      name += "_IS";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_ANY_HIT:
                                      name += "_AH";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_CLOSEST_HIT:
                                      name += "_CH";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_MISS:
                                      name += "_MS";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_EXCEPTION:
                                      name += "_EX";
                                      break;
                                  default:
                                      name += "_??";
                              }
                              return name;
                          } );


struct O7_API_Device_Query_TransformFromHandle : public O7_API_Device_Query_Base
{
  public:
    void SetUp() override
    {
        O7_API_Device_Query_Base::SetUp();
        m_params.testTransformData = true;
        m_params.optixProgramType  = std::get<0>( GetParam() );
    }
};


TEST_P( O7_API_Device_Query_TransformFromHandle, RunWithIASHierarchy )
{
    runTest();
    EXPECT_TRUE( m_testResultsOut.transformationsEqual );
}

INSTANTIATE_TEST_SUITE_P( TestTransformFromHandleCalls,
                          O7_API_Device_Query_TransformFromHandle,
                          testing::Combine( testing::Values( OPTIX_PROGRAM_TYPE_RAYGEN,
                                                             OPTIX_PROGRAM_TYPE_INTERSECTION,
                                                             OPTIX_PROGRAM_TYPE_ANY_HIT,
                                                             OPTIX_PROGRAM_TYPE_CLOSEST_HIT,
                                                             OPTIX_PROGRAM_TYPE_MISS,
                                                             OPTIX_PROGRAM_TYPE_EXCEPTION ),
                                            testing::Values( OPTIX_TRAVERSABLE_TYPE_EXTENDED_STATIC_TRANSFORM,
                                                             OPTIX_TRAVERSABLE_TYPE_EXTENDED_MATRIX_MOTION_TRANSFORM,
                                                             OPTIX_TRAVERSABLE_TYPE_EXTENDED_SRT_MOTION_TRANSFORM,
                                                             OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM ) ),
                          []( const testing::TestParamInfo<O7_API_Device_Query_TransformFromHandle::ParamType>& info ) {
                              std::string name;
                              switch( std::get<1>( info.param ) )
                              {
                                  case OPTIX_TRAVERSABLE_TYPE_EXTENDED_STATIC_TRANSFORM:
                                      name = "STATIC_TRANSFORM";
                                      break;
                                  case OPTIX_TRAVERSABLE_TYPE_EXTENDED_MATRIX_MOTION_TRANSFORM:
                                      name = "MOTION_TRANSFORM";
                                      break;
                                  case OPTIX_TRAVERSABLE_TYPE_EXTENDED_SRT_MOTION_TRANSFORM:
                                      name = "SRT_TRANSFORM";
                                      break;
                                  case OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM:
                                      name = "INSTANCE";
                                      break;
                                  default:
                                      name = "UNKNOWN";
                              }
                              switch( std::get<0>( info.param ) )
                              {
                                  case OPTIX_PROGRAM_TYPE_RAYGEN:
                                      name += "_RG";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_INTERSECTION:
                                      name += "_IS";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_ANY_HIT:
                                      name += "_AH";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_CLOSEST_HIT:
                                      name += "_CH";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_MISS:
                                      name += "_MS";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_EXCEPTION:
                                      name += "_EX";
                                      break;
                                  default:
                                      name += "_??";
                              }
                              return name;
                          } );


struct O7_API_Device_Query_Instance : public O7_API_Device_Query_Base
{
  public:
    void SetUp() override
    {
        O7_API_Device_Query_Base::SetUp();
        m_params.testQueryInst    = true;
        m_params.optixProgramType = std::get<0>( GetParam() );
    }
};


TEST_P( O7_API_Device_Query_Instance, testGetInstanceId )
{
    runTest();

    if( std::get<1>( GetParam() ) == OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM )
        EXPECT_EQ( EXPECTED_ID, m_testResultsOut.instanceIdOut );
    else
        EXPECT_EQ( EXPECTED_PARENT_ID, m_testResultsOut.instanceIdOut );
}

TEST_P( O7_API_Device_Query_Instance, testGetInstanceIndex )
{
    runTest();

    const unsigned int expectedIndex           = 1;
    const unsigned int expectedNoInstanceIndex = 0;
    if( std::get<1>( GetParam() ) == OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM )
        EXPECT_EQ( expectedIndex, m_testResultsOut.instanceIndexOut );
    else
        EXPECT_EQ( expectedNoInstanceIndex, m_testResultsOut.instanceIndexOut );
}

INSTANTIATE_TEST_SUITE_P( TestInstanceQueryCalls,
                          O7_API_Device_Query_Instance,
                          testing::Combine( testing::Values( OPTIX_PROGRAM_TYPE_INTERSECTION, OPTIX_PROGRAM_TYPE_ANY_HIT, OPTIX_PROGRAM_TYPE_CLOSEST_HIT ),
                                            testing::Values( OPTIX_TRAVERSABLE_TYPE_EXTENDED_STATIC_TRANSFORM,
                                                             OPTIX_TRAVERSABLE_TYPE_EXTENDED_MATRIX_MOTION_TRANSFORM,
                                                             OPTIX_TRAVERSABLE_TYPE_EXTENDED_SRT_MOTION_TRANSFORM,
                                                             OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM ) ),
                          []( const testing::TestParamInfo<O7_API_Device_Query_Base::ParamType>& info ) {
                              std::string name;
                              switch( std::get<1>( info.param ) )
                              {
                                  case OPTIX_TRAVERSABLE_TYPE_EXTENDED_STATIC_TRANSFORM:
                                      name = "STATIC_TRANSFORM";
                                      break;
                                  case OPTIX_TRAVERSABLE_TYPE_EXTENDED_MATRIX_MOTION_TRANSFORM:
                                      name = "MOTION_TRANSFORM";
                                      break;
                                  case OPTIX_TRAVERSABLE_TYPE_EXTENDED_SRT_MOTION_TRANSFORM:
                                      name = "SRT_TRANSFORM";
                                      break;
                                  case OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM:
                                      name = "INSTANCE";
                                      break;
                                  default:
                                      name = "UNKNOWN";
                              }
                              switch( std::get<0>( info.param ) )
                              {
                                  case OPTIX_PROGRAM_TYPE_INTERSECTION:
                                      name += "_IS";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_ANY_HIT:
                                      name += "_AH";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_CLOSEST_HIT:
                                      name += "_CH";
                                      break;
                                  default:
                                      name += "_??";
                              }
                              return name;
                          } );


struct O7_API_Device_Query_SBTData : public O7_API_Device_Query_Base
{
  public:
    void SetUp() override
    {
        O7_API_Device_Query_Base::SetUp();
        m_params.testSBTData      = true;
        m_params.optixProgramType = std::get<0>( GetParam() );
    }
    bool supported() const
    {
        return ( m_params.optixProgramType == OPTIX_PROGRAM_TYPE_INTERSECTION )
               || ( m_params.optixProgramType == OPTIX_PROGRAM_TYPE_ANY_HIT )
               || ( m_params.optixProgramType == OPTIX_PROGRAM_TYPE_CLOSEST_HIT );
    }
};

TEST_P( O7_API_Device_Query_SBTData, testGetSBTDataPointer )
{
    runTest();
    // the inserted value is 0 for all but the good prims' SBT index
    EXPECT_EQ( EXPECTED_SBT_DATA, m_testResultsOut.sbtData );
}

TEST_P( O7_API_Device_Query_SBTData, testGetSbtGASIndex )
{
    if( !supported() )
        return;
    runTest();
    EXPECT_EQ( EXPECTED_SBT_GAS_INDEX, m_testResultsOut.sbtGASIndex );
}

INSTANTIATE_TEST_SUITE_P( TestSBTQueryCalls,
                          O7_API_Device_Query_SBTData,
                          testing::Combine( testing::Values( OPTIX_PROGRAM_TYPE_RAYGEN,
                                                             OPTIX_PROGRAM_TYPE_INTERSECTION,
                                                             OPTIX_PROGRAM_TYPE_ANY_HIT,
                                                             OPTIX_PROGRAM_TYPE_CLOSEST_HIT,
                                                             OPTIX_PROGRAM_TYPE_MISS,
                                                             OPTIX_PROGRAM_TYPE_EXCEPTION ),
                                            testing::Values( OPTIX_TRAVERSABLE_TYPE_EXTENDED_INSTANCE_TRANSFORM ) ),
                          []( const testing::TestParamInfo<O7_API_Device_Query_Base::ParamType>& info ) {
                              std::string name = "Q_INSTANCE";
                              switch( std::get<0>( info.param ) )
                              {
                                  case OPTIX_PROGRAM_TYPE_RAYGEN:
                                      name += "_RG";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_INTERSECTION:
                                      name += "_IS";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_ANY_HIT:
                                      name += "_AH";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_CLOSEST_HIT:
                                      name += "_CH";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_MISS:
                                      name += "_MS";
                                      break;
                                  case OPTIX_PROGRAM_TYPE_EXCEPTION:
                                      name += "_EX";
                                      break;
                                  default:
                                      name += "_??";
                              }
                              return name;
                          } );
