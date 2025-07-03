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
//
//

#include <optix.h>
#include <optix_stubs.h>
#include <private/optix_7_enum_printers.h>

#include <exptest/exptest.h>
#include <exptest/loggingHelpers.h>
#include <exptest/testTools.h>

#include <gmock/gmock.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include "CommonAsserts.h"
#include "model.h"

using namespace testing;

// Common class, do basic Context initialization

namespace {

class DenoiserDeviceContext
{
  protected:
    void SetUpContext()
    {
        exptest::lwdaInitialize();
        ASSERT_OPTIX_SUCCESS( optixInit() );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( nullptr, nullptr, &m_context ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &OptixRecordingLogger::callback, &m_logger, 2 ) );
        ASSERT_LWDA_SUCCESS( lwdaStreamCreate( &m_stream ) );
    }

    void TearDownContext()
    {
        ASSERT_LWDA_SUCCESS( lwdaStreamDestroy( m_stream ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( m_context ) );
    }

    OptixDeviceContext m_context = nullptr;
    OptixRecordingLogger m_logger;
    LWstream           m_stream  = nullptr;
};


using Param = std::tuple<OptixDenoiserModelKind, bool, bool, OptixPixelFormat, OptixDenoiserAlphaMode, bool>;

class Denoiser : public DenoiserDeviceContext, public TestWithParam<Param>
{
  protected:
    void createDenoiser()
    {
        m_denoiserModelKind              = { std::get<0>( GetParam() ) };
        m_denoiserOptions.guideNormal    = { std::get<1>( GetParam() ) };
        m_denoiserOptions.guideAlbedo    = { std::get<2>( GetParam() ) };
        bool userModel                   = { std::get<5>( GetParam() ) };

        if( userModel )
        {
            // the user model provides rgb-albedo-normal, fixup the parameters
            m_denoiserOptions.guideNormal = true;
            m_denoiserOptions.guideAlbedo = true;
            ASSERT_EQ( OPTIX_SUCCESS, optixDenoiserCreateWithUserModel( m_context, (void*)userModelData, sizeof(userModelData), &m_denoiser ) );
        }
        else
            ASSERT_EQ( OPTIX_SUCCESS, optixDenoiserCreate( m_context, m_denoiserModelKind, &m_denoiserOptions, &m_denoiser ) );
    }

    void SetUp() override { SetUpContext(); }

    void TearDown() override
    {
        if( m_denoiser )
        {
            ASSERT_OPTIX_SUCCESS( optixDenoiserDestroy( m_denoiser ) );
        }

        TearDownContext();
    }

    OptixDenoiser          m_denoiser        = nullptr;
    OptixDenoiserModelKind m_denoiserModelKind;
    OptixDenoiserOptions   m_denoiserOptions = {};
};

}  // namespace

struct DenoiserIlwoke : public Denoiser
{
    void SetUp() override { Denoiser::SetUp(); Denoiser::createDenoiser(); }
    void TearDown() override { Denoiser::TearDown(); }

};

TEST_P( DenoiserIlwoke, validTest )
{
    unsigned int width  = 256;
    unsigned int height = 256;
    unsigned int inputOffsetX = 0;
    unsigned int inputOffsetY = 0;

    LWdeviceptr denoiserData, denoiserScratch;
    OptixDenoiserSizes sizes;
    ASSERT_EQ(OPTIX_SUCCESS, optixDenoiserComputeMemoryResources( m_denoiser, width, height, &sizes ) );
    lwdaMalloc( (void**)&denoiserScratch, sizes.withoutOverlapScratchSizeInBytes );
    lwdaMalloc( (void**)&denoiserData, sizes.stateSizeInBytes );

    ASSERT_EQ( OPTIX_SUCCESS, optixDenoiserSetup( m_denoiser, m_stream, width, height,
               denoiserData, sizes.stateSizeInBytes, denoiserScratch, sizes.withoutOverlapScratchSizeInBytes ) );

    OptixDenoiserModelKind mkind = std::get<0>( GetParam() );
    OptixPixelFormat format = std::get<3>( GetParam() );

    unsigned int pixSize;
    if( format == OPTIX_PIXEL_FORMAT_HALF3 )
        pixSize = 3 * sizeof( short );
    else if( format == OPTIX_PIXEL_FORMAT_HALF4 )
        pixSize = 4 * sizeof( short );
    else if( format == OPTIX_PIXEL_FORMAT_FLOAT3 )
        pixSize = 3 * sizeof( float );
    else if( format == OPTIX_PIXEL_FORMAT_FLOAT4 )
        pixSize = 4 * sizeof( float );
    else FAIL() << "pixel format not supported in this test";

    void * imagep;
    lwdaMalloc( &imagep, pixSize * width * height );
    OptixImage2D image = {(LWdeviceptr)imagep, width, height, width*pixSize, 0, format};
    OptixDenoiserGuideLayer guideLayer = {};
    OptixDenoiserParams params = {};

    void * internalp;
    lwdaMalloc( &internalp, sizes.internalGuideLayerSizeInBytes * width * height );
    OptixImage2D internalImage = {(LWdeviceptr)internalp, width, height, width*unsigned(sizes.internalGuideLayerSizeInBytes), unsigned(sizes.internalGuideLayerSizeInBytes), OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER};

    OptixDenoiserLayer layers[2] = {};
    layers[0].input  = image;
    layers[0].output = image;
    unsigned int numLayers = 1;

    // in AOV mode we add another layer. it would however also work with one layer.
    if( mkind == OPTIX_DENOISER_MODEL_KIND_AOV || mkind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV )
    {
        layers[1].input  = image;
        layers[1].output = image;
        numLayers++;
    }

    if( m_denoiserOptions.guideAlbedo )
        guideLayer.albedo = image;
    if( m_denoiserOptions.guideNormal )
        guideLayer.normal = image;
    if( mkind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL || mkind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV )
    {
        guideLayer.flow      = image;
        guideLayer.previousOutputInternalGuideLayer = internalImage;
        guideLayer.outputInternalGuideLayer = internalImage;
        layers[0].previousOutput = image;
        layers[1].previousOutput = image;
    }
    lwdaMalloc( (void**)&params.hdrIntensity, sizeof(float) );
    lwdaMalloc( (void**)&params.hdrAverageColor, 3 * sizeof(float) );
    ASSERT_EQ( OPTIX_SUCCESS,
               optixDenoiserComputeIntensity( m_denoiser, m_stream, &layers[0].input, params.hdrIntensity,
                                              denoiserScratch, sizes.withoutOverlapScratchSizeInBytes ) );
    ASSERT_EQ( OPTIX_SUCCESS,
               optixDenoiserComputeAverageColor( m_denoiser, m_stream, &layers[0].input, params.hdrAverageColor,
                                                 denoiserScratch, sizes.withoutOverlapScratchSizeInBytes ) );
    params.denoiseAlpha = std::get<4>( GetParam() );

    ASSERT_EQ( OPTIX_SUCCESS,
               optixDenoiserIlwoke( m_denoiser, m_stream, &params, denoiserData, sizes.stateSizeInBytes,
                                    &guideLayer, layers, numLayers, inputOffsetX, inputOffsetY,
                                    denoiserScratch, sizes.withoutOverlapScratchSizeInBytes ) );
    lwdaFree( (void*)params.hdrAverageColor );
    lwdaFree( (void*)params.hdrIntensity );
    lwdaFree( internalp );
    lwdaFree( imagep );
    lwdaFree( (void*)denoiserData );
    lwdaFree( (void*)denoiserScratch );
}

INSTANTIATE_TEST_SUITE_P( O7_API, DenoiserIlwoke,
    testing::Combine(
            testing::Values(OPTIX_DENOISER_MODEL_KIND_HDR,OPTIX_DENOISER_MODEL_KIND_AOV,
                            OPTIX_DENOISER_MODEL_KIND_TEMPORAL,OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV),
            testing::Values(false,true),        // albedo
            testing::Values(false,true),        // normal
            testing::Values(OPTIX_PIXEL_FORMAT_HALF3,OPTIX_PIXEL_FORMAT_HALF4,OPTIX_PIXEL_FORMAT_FLOAT3,OPTIX_PIXEL_FORMAT_FLOAT4),
            testing::Values(OPTIX_DENOISER_ALPHA_MODE_COPY,OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV,OPTIX_DENOISER_ALPHA_MODE_FULL_DENOISE_PASS),
            testing::Values(false,true) ) );    // builtin / user model
