/*
 * Copyright 2020 LWPU Corporation. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <vector>

#include <corelib/misc/String.h>
#include <corelib/system/LwdaDriver.h>
#include <exp/context/DeviceContext.h>
#include <exp/context/ErrorHandling.h>
#include <exp/context/OpaqueApiObject.h>
#include <exp/denoise/Denoise.h>
#include <exp/denoise/fp16_emu.h>
#include <exp/denoise/layerdata_load.h>
#include <exp/denoise/luminance.h>
#include <exp/denoise/rgbaverage.h>
#include <exp/denoise/training_load.h>
#include <prodlib/system/Logger.h>

#include <prodlib/misc/LWTXProfiler.h>

#include <algorithm>

namespace optix_exp {

namespace {

unsigned int getNaturalPixelStride( const OptixImage2D& image, unsigned int internalGuideNumChannels )
{
    switch( image.format )
    {
        case OPTIX_PIXEL_FORMAT_UCHAR3:
            return 3 * sizeof( unsigned char );
        case OPTIX_PIXEL_FORMAT_UCHAR4:
            return 4 * sizeof( unsigned char );
        case OPTIX_PIXEL_FORMAT_FLOAT2:
            return 2 * sizeof( float );
        case OPTIX_PIXEL_FORMAT_FLOAT3:
            return 3 * sizeof( float );
        case OPTIX_PIXEL_FORMAT_FLOAT4:
            return 4 * sizeof( float );
        case OPTIX_PIXEL_FORMAT_HALF2:
            return 2 * sizeof( short );
        case OPTIX_PIXEL_FORMAT_HALF3:
            return 3 * sizeof( short );
        case OPTIX_PIXEL_FORMAT_HALF4:
            return 4 * sizeof( short );
        case OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER:
            return internalGuideNumChannels * sizeof( short );
    }
    return 0;
}

}  // namespace

DeviceContextLogger& Denoiser::getLogger()
{
    return m_context->getLogger();
}

Denoiser::Denoiser( DeviceContext* context )
    : OpaqueApiObject( OpaqueApiObject::ApiType::Denoiser )
    , m_denoiser( nullptr )
    , m_context( context )
{
}

OptixResult Denoiser::destroy( bool doUnregisterDenoiser, ErrorDetails& errDetails )
{
    OptixResult result = doUnregisterDenoiser ? m_context->unregisterDenoiser( this, errDetails ) : OPTIX_SUCCESS;
    if( m_denoiser != nullptr )
    {
        OptixResult r = m_denoiser->destroy( errDetails );
        delete m_denoiser;
        m_denoiser = nullptr;
        if( !result )
            result = r;
    }
    return result;
}

OptixResult Denoiser::initModelDimension( const Layerdata& ld, ErrorDetails& errDetails )
{
    m_modelDimension   = ld.getNumInfChannels();
    m_kernelPrediction = ld.hKPN();
    m_temporalModel    = ld.getTemporalLayerIndex() != -1;
    m_hdrModel         = ld.isHDR();
    return OPTIX_SUCCESS;
}

OptixResult Denoiser::loadBuiltinModel( OptixDenoiserModelKind modelKind, ErrorDetails& errDetails )
{
    std::string libPath;
    if( OptixResult res = m_context->getLibraryPath( libPath, errDetails ) )
        return res;

    std::vector<char> data;

    if( OptixResult res = denoiseGetBuiltinTrainingSet( libPath, modelKind, data, errDetails ) )
        return res;

    bool  hasSet, isHdr;
    float hdrScale;
    if( const OptixResult res = Layerdata::hasTrainingSet( (void*)&data[0], getTrainingName( modelKind, m_inferenceMode ), hasSet, isHdr, hdrScale, errDetails ) )
        return res;

    if( !hasSet )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, corelib::stringf( "Could not load built-in weights for model kind: 0x%x, guide images: %s",
                                      modelKind,
                                      m_inferenceMode == OPTIX_DENOISER_INPUT_RGB ? "none" :
                                      m_inferenceMode == OPTIX_DENOISER_INPUT_RGB_ALBEDO ? "albedo" :
                                      m_inferenceMode == OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL ? "albedo, normal" : "invalid" ) );

    // This constructor can log errors (which are lwrrently disabled), but it does nothing else to communicate that the object is invalid.
    Layerdata ld;
    if( const OptixResult res = ld.load( &data[0], data.size(), getTrainingName( modelKind, m_inferenceMode ), errDetails ) )
        return res;

    m_denoiser->setWeights( ld );

    if( const OptixResult res = initModelDimension( ld, errDetails ) )
        return res;

    if( modelKind == OPTIX_DENOISER_MODEL_KIND_LDR )
        m_hdrModel = false;             // force autoexposure disable

    return OPTIX_SUCCESS;
}

OptixResult Denoiser::getInputKindForUserModel( const void* tdata, size_t sizeInBytes, OptixDenoiserInputKind_v1& inputKind, ErrorDetails& errDetails )
{
    std::string trName;
    if( const OptixResult res = getTrainingNameForUserModel( tdata, sizeInBytes, trName, errDetails ) )
        return res;
    if( trName == "rgb" )
        inputKind = OPTIX_DENOISER_INPUT_RGB;
    else if( trName == "rgb-albedo" )
        inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;
    else if( trName == "rgb-albedo-normal" || trName == "rgb-albedo-normal3" )
        inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
    else if( trName == "rgb-flow" )
        inputKind = OPTIX_DENOISER_INPUT_RGB;
    else if( trName == "rgb-albedo-flow" )
        inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;
    else if( trName == "rgb-albedo-normal-flow" || trName == "rgb-albedo-normal3-flow" )
        inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
    else
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      corelib::stringf( "User model defines unknown input kind %s", trName.c_str() ) );
    return OPTIX_SUCCESS;
}

OptixResult Denoiser::getTrainingNameForUserModel( const void* tdata, size_t sizeInBytes, std::string & trName, ErrorDetails& errDetails )
{
    std::vector<std::string> ids;
    if( Layerdata::getTrainingIds( tdata, ids, errDetails ) )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, corelib::stringf( "Could not get weight ids from user model " ) );

    if( ids.size() > 1 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      corelib::stringf( "User model defines more than one weight set" ) );
    else if( ids.size() == 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      corelib::stringf( "User model does not define any weights" ) );
    trName = ids[0];

    return OPTIX_SUCCESS;
}

OptixResult Denoiser::loadUserModel( const void* tdata, size_t sizeInBytes, ErrorDetails& errDetails )
{
    // This constructor can log errors (which are lwrrently disabled), but it does nothing else to communicate that the object is invalid.
    Layerdata ld;

    std::string trName;
    if( const OptixResult res = getTrainingNameForUserModel( tdata, sizeInBytes, trName, errDetails ) )
        return res;
    if( const OptixResult res = ld.load( tdata, sizeInBytes, trName.c_str(), errDetails ) )
        return res;

    m_denoiser->setWeights( ld );

    if( const OptixResult res = initModelDimension( ld, errDetails ) )
        return res;

    return OPTIX_SUCCESS;
}

OptixResult Denoiser::initBuiltinModel( OptixDenoiserModelKind modelKind, const OptixDenoiserOptions* options, ErrorDetails& errDetails )
{
    if( options->guideNormal )
        m_inferenceMode = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
    else if( options->guideAlbedo )
        m_inferenceMode = OPTIX_DENOISER_INPUT_RGB_ALBEDO;
    else
        m_inferenceMode = OPTIX_DENOISER_INPUT_RGB;

    m_denoiser = new Denoise( m_context->getLogger() );

    OptixResult result;
    if( ( result = m_denoiser->init( errDetails ) ) )
        destroy( false, errDetails );
    else
        result = loadBuiltinModel( modelKind, errDetails );
    return result;
}

OptixResult Denoiser::initUserModel( OptixDenoiserInputKind_v1 inputKind, const void* userData, size_t userDataSizeInBytes, ErrorDetails& errDetails )
{
    m_inferenceMode = inputKind;

    m_denoiser = new Denoise( m_context->getLogger() );

    OptixResult result;
    if( ( result = m_denoiser->init( errDetails ) ) )
        destroy( false, errDetails );
    else
        result = loadUserModel( userData, userDataSizeInBytes, errDetails );
    return result;
}

inline OptixResult implCast( OptixDenoiser denoiserAPI, Denoiser*& denoiser )
{
    denoiser = reinterpret_cast<Denoiser*>( denoiserAPI );
    if( denoiser && denoiser->m_apiType != OpaqueApiObject::ApiType::Denoiser )
    {
        return OPTIX_ERROR_ILWALID_VALUE;
    }
    return OPTIX_SUCCESS;
}

inline OptixDenoiser apiCast( Denoiser* denoiser )
{
    return reinterpret_cast<OptixDenoiser>( denoiser );
}

extern "C" OptixResult optixDenoiserCreateWithUserModel( OptixDeviceContext contextAPI,
                                                         const void*        userData,
                                                         size_t             userDataSizeInBytes,
                                                         OptixDenoiser*     denoiserAPI )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( DeviceContext, context, "OptixDeviceContext" );
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DENOISER_CREATE );

    DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( userData );
    OPTIX_CHECK_NULL_ARGUMENT( denoiserAPI );
    *denoiserAPI = nullptr;

    ErrorDetails errDetails;
    if( userDataSizeInBytes < Layerdata::MinUserDataSize )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, corelib::stringf("userDataSizeInBytes %llu, must be >= %llu",
                                      static_cast<unsigned long long>( userDataSizeInBytes ),
                                      static_cast<unsigned long long>( Layerdata::MinUserDataSize ) ) );

    try
    {
        std::unique_ptr<Denoiser> denoiser( new Denoiser( context ) );

        OptixDenoiserInputKind_v1 inputKind;
        OptixResult result = denoiser.get()->getInputKindForUserModel( userData, userDataSizeInBytes, inputKind, errDetails );
        if( result == OPTIX_SUCCESS )
            result = denoiser.get()->initUserModel( inputKind, userData, userDataSizeInBytes, errDetails );
        if( result == OPTIX_SUCCESS )
            result = context->registerDenoiser( denoiser.get(), errDetails );
        if( result != OPTIX_SUCCESS )
        {
            clog.sendError( errDetails );
            return result;
        }
        *denoiserAPI = apiCast( denoiser.release() );
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

// this function is called from the O6 denoiser, it properly fills out albedo and normal in the options,
// in order to get the matching weight set from the user model.
// temporal is not supported in the O6 denoiser.
OptixResult optixDenoiserCreateWithUserModel_internal( OptixDeviceContext          contextAPI,
                                                       const OptixDenoiserOptions* options,
                                                       const void*                 userData,
                                                       size_t                      userDataSizeInBytes,
                                                       OptixDenoiser*              denoiserAPI )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( DeviceContext, context, "OptixDeviceContext" );
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DENOISER_CREATE );

    DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( options );
    OPTIX_CHECK_NULL_ARGUMENT( userData );
    OPTIX_CHECK_NULL_ARGUMENT( denoiserAPI );
    *denoiserAPI = nullptr;

    try
    {
        OptixDenoiserInputKind_v1 inputKind;
        if( options->guideNormal )
            inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
        else if( options->guideAlbedo )
            inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;
        else
            inputKind = OPTIX_DENOISER_INPUT_RGB;

        std::unique_ptr<Denoiser> denoiser( new Denoiser( context ) );

        ErrorDetails errDetails;
        OptixResult  result = denoiser.get()->initUserModel( inputKind, userData, userDataSizeInBytes, errDetails );
        if( result == OPTIX_SUCCESS )
            result = context->registerDenoiser( denoiser.get(), errDetails );
        if( result != OPTIX_SUCCESS )
        {
            clog.sendError( errDetails );
            return result;
        }
        *denoiserAPI = apiCast( denoiser.release() );
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixDenoiserCreate( OptixDeviceContext          contextAPI,
                                            OptixDenoiserModelKind      modelKind,
                                            const OptixDenoiserOptions* options,
                                            OptixDenoiser*              denoiserAPI )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( DeviceContext, context, "OptixDeviceContext" );
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DENOISER_CREATE );

    DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( options );
    OPTIX_CHECK_NULL_ARGUMENT( denoiserAPI );
    *denoiserAPI = nullptr;

    try
    {
        std::unique_ptr<Denoiser> denoiser( new Denoiser( context ) );
        ErrorDetails              errDetails;

        if( modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV && denoiser->getDeviceContext()->getAbiVersion() < OptixABI::ABI_60 )
        {
            lwarn << "ABI < 60, modelKind " << modelKind << " set to " << OptixDenoiserModelKind( OPTIX_DENOISER_MODEL_KIND_TEMPORAL );
            modelKind = OPTIX_DENOISER_MODEL_KIND_TEMPORAL;
        }

        OptixResult result = denoiser.get()->initBuiltinModel( modelKind, options, errDetails );
        if( result == OPTIX_SUCCESS )
            result = context->registerDenoiser( denoiser.get(), errDetails );
        else
        {
            clog.sendError( errDetails );
            return result;
        }
        *denoiserAPI = apiCast( denoiser.release() );
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}
extern "C" OptixResult optixDenoiserDestroy( OptixDenoiser denoiserAPI )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Denoiser, denoiser, "OptixDenoiser" );
    SCOPED_LWTX_RANGE( denoiser->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::DENOISER_DESTROY );

    DeviceContextLogger& clog = denoiser->getLogger();

    try
    {
        ErrorDetails errDetails;
        OptixResult  result = denoiser->destroy( true, errDetails );
        if( result != OPTIX_SUCCESS )
            clog.sendError( errDetails );
        delete denoiser;
        return result;
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

typedef struct OptixDenoiserSizes_ABI27
{
    size_t       stateSizeInBytes;
    size_t       minimumScratchSizeInBytes;
    size_t       recommendedScratchSizeInBytes;
    unsigned int overlapWindowSizeInPixels;
} OptixDenoiserSizes_ABI27;

extern "C" OptixResult optixDenoiserComputeMemoryResources( const OptixDenoiser denoiserAPI,
                                                            unsigned int        outputWidth,
                                                            unsigned int        outputHeight,
                                                            OptixDenoiserSizes* returnSizes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Denoiser, denoiser, "OptixDenoiser" );
    SCOPED_LWTX_RANGE( denoiser->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::DENOISER_COMPUTE_MEMORY_RESOURCES );

    DeviceContextLogger& clog = denoiser->getLogger();
    OPTIX_CHECK_NULL_ARGUMENT( returnSizes );

    try
    {
        OptixResult  result = OPTIX_SUCCESS;
        ErrorDetails errDetails;
        {
            std::lock_guard<std::mutex> lock( denoiser->getMutex() );

            if( denoiser->getDeviceContext()->getAbiVersion() >= OptixABI::ABI_60 )
                returnSizes->internalGuideLayerSizeInBytes = sizeof(short) * denoiser->m_denoiser->getNumHiddenChannels();

            if( denoiser->getDeviceContext()->getAbiVersion() >= OptixABI::ABI_59 )
            {
                result = denoiseAutoexposureComputeMemoryResources( returnSizes->computeIntensitySizeInBytes, errDetails );
                OptixResult r = denoiseRGBAverageComputeMemoryResources( returnSizes->computeAverageColorSizeInBytes, errDetails );
                if( result == OPTIX_SUCCESS )
                    result = r;
            }

            if( denoiser->getDeviceContext()->getAbiVersion() >= OptixABI::ABI_28 )
            {
                returnSizes->overlapWindowSizeInPixels = denoiser->m_denoiser->getOverlapSize();

                OptixResult cmr = denoiser->m_denoiser->callwlateMemoryResources( outputWidth, outputHeight, denoiser->getInputChannels(),
                                                        3,  // denoiser always has RGB output
                                                        &returnSizes->withoutOverlapScratchSizeInBytes,
                                                        &returnSizes->stateSizeInBytes, errDetails );
                if( result == OPTIX_SUCCESS )
                    result = cmr;
                
                size_t st_size;
                cmr = denoiser->m_denoiser->callwlateMemoryResources(
                                outputWidth + 2 * denoiser->m_denoiser->getOverlapSize(),
                                outputHeight + 2 * denoiser->m_denoiser->getOverlapSize(), denoiser->getInputChannels(),
                                3,  // denoiser always has RGB output
                                &returnSizes->withOverlapScratchSizeInBytes, &st_size, errDetails );
                if( result == OPTIX_SUCCESS )
                    result = cmr;
                if( result == OPTIX_SUCCESS && st_size != returnSizes->stateSizeInBytes )
                    result =
                        errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, "wrong state size, expected same value" );
                returnSizes->withoutOverlapScratchSizeInBytes = returnSizes->withoutOverlapScratchSizeInBytes;
                returnSizes->withOverlapScratchSizeInBytes = returnSizes->withOverlapScratchSizeInBytes;
            }
            else
            {
                OptixDenoiserSizes_ABI27* rSize27  = (OptixDenoiserSizes_ABI27*)returnSizes;
                rSize27->overlapWindowSizeInPixels = denoiser->m_denoiser->getOverlapSize();

                result = denoiser->m_denoiser->callwlateMemoryResources( outputWidth, outputHeight,
                                                                         denoiser->getInputChannels(),
                                                                         3,  // denoiser always has RGB output
                                                                         &rSize27->minimumScratchSizeInBytes,
                                                                         &rSize27->stateSizeInBytes, errDetails );
                rSize27->minimumScratchSizeInBytes     = rSize27->minimumScratchSizeInBytes;
                rSize27->recommendedScratchSizeInBytes = rSize27->minimumScratchSizeInBytes;
            }
        }
        if( result )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

static OptixResult setupDenoiser( Denoiser*     denoiser,
                                  LWstream      stream,
                                  unsigned int  inputWidth,
                                  unsigned int  inputHeight,
                                  LWdeviceptr   denoiserState,
                                  size_t        denoiserStateSizeInBytes,
                                  LWdeviceptr   scratch,
                                  size_t        scratchSizeInBytes,
                                  ErrorDetails& errDetails )

{
    OPTIX_CHECK_VALIDATION_MODE_LWRRENT_LWDA_CONTEXT( denoiser->getDeviceContext() );
    OPTIX_CHECK_VALIDATION_MODE_STREAM_STATE( denoiser->getDeviceContext(), stream );
    OptixResult result = OPTIX_SUCCESS;
    if( ( result = denoiser->m_denoiser->deleteLayers( errDetails ) ) )
        return result;
    if( ( result = denoiser->m_denoiser->createLayers( inputWidth, inputHeight, denoiser->getInputChannels(), errDetails ) ) )
        return result;
    if( ( result = denoiser->m_denoiser->initializeWeights( (void*)scratch, scratchSizeInBytes, (void*)denoiserState,
                                                            denoiserStateSizeInBytes, stream, errDetails ) ) )
        return result;
    return result;
}

extern "C" OptixResult optixDenoiserSetup( OptixDenoiser denoiserAPI,
                                           LWstream      stream,
                                           unsigned int  inputWidth,
                                           unsigned int  inputHeight,
                                           LWdeviceptr   denoiserState,
                                           size_t        denoiserStateSizeInBytes,
                                           LWdeviceptr   scratch,
                                           size_t        scratchSizeInBytes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Denoiser, denoiser, "OptixDenoiser" );
    SCOPED_LWTX_RANGE( denoiser->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::DENOISER_SETUP );

    DeviceContextLogger& clog = denoiser->getLogger();

    try
    {
        ErrorDetails errDetails;
        OptixResult  result = OPTIX_SUCCESS;

        {
            std::lock_guard<std::mutex> lock( denoiser->getMutex() );
            result = setupDenoiser( denoiser, stream, inputWidth, inputHeight, denoiserState, denoiserStateSizeInBytes,
                                    scratch, scratchSizeInBytes, errDetails );
        }
        if( result != OPTIX_SUCCESS )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

static OptixResult checkOptixImage2D( const OptixImage2D& image,
                                      unsigned int        width,
                                      unsigned int        height,
                                      bool                allow2DImageFormat,
                                      bool                internalGuideImage,
                                      unsigned int        internalGuideNumChannels,
                                      const char*         infoString,
                                      ErrorDetails&       errDetails )

{
    if( image.data == 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, corelib::stringf( "%s: data pointer null", infoString ) );

    if( image.width != width || image.height != height )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      corelib::stringf( "%s: resolution %u x %u not the same as layer 0, %u x %u",
                                                        infoString, image.width, image.height, width, height ) );

    if( internalGuideImage )
    {
        if( image.format != OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          corelib::stringf( "%s: internal guide layer image must have format OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER, passed format 0x%x", infoString, (unsigned int)image.format ) );
        if( image.pixelStrideInBytes == 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                        corelib::stringf( "%s: internal guide layer image must set pixelStrideInBytes", infoString ) );
    }
    else
    {
        if( image.format == OPTIX_PIXEL_FORMAT_UCHAR3 || image.format == OPTIX_PIXEL_FORMAT_UCHAR4 )
            return errDetails.logDetails( OPTIX_ERROR_NOT_SUPPORTED,
                                          corelib::stringf( "%s: unsigned char pixel format not supported", infoString ) );

        if( image.format == OPTIX_PIXEL_FORMAT_FLOAT2 || image.format == OPTIX_PIXEL_FORMAT_HALF2 )
        {
            if( !allow2DImageFormat )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              corelib::stringf( "%s: 2d pixel format not allowed here", infoString ) );
        }
        else if( !( image.format == OPTIX_PIXEL_FORMAT_FLOAT3 || image.format == OPTIX_PIXEL_FORMAT_FLOAT4 ||
                    image.format == OPTIX_PIXEL_FORMAT_HALF3 || image.format == OPTIX_PIXEL_FORMAT_HALF4 ) )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, corelib::stringf( "%s: unsupported pixel format 0x%x",
                                                                                       infoString, (unsigned int)image.format ) );
    }

    if( image.pixelStrideInBytes != 0 )
    {
        unsigned int naturalPixelStride = getNaturalPixelStride( image, internalGuideNumChannels );
        if( image.pixelStrideInBytes < naturalPixelStride )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          corelib::stringf( "%s: pixel stride must be 0 or greater or equal %u for "
                                                            "pixel format "
                                                            "0x%x",
                                                            infoString, naturalPixelStride, (unsigned int)image.format ) );

        int strideElementAlignment = 0;
        switch( image.format )
        {
            case OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER:
            case OPTIX_PIXEL_FORMAT_HALF2:
            case OPTIX_PIXEL_FORMAT_HALF3:
            case OPTIX_PIXEL_FORMAT_HALF4:
                strideElementAlignment = 2;
                break;
            case OPTIX_PIXEL_FORMAT_FLOAT2:
            case OPTIX_PIXEL_FORMAT_FLOAT3:
            case OPTIX_PIXEL_FORMAT_FLOAT4:
                strideElementAlignment = 4;
                break;
            case OPTIX_PIXEL_FORMAT_UCHAR3:
            case OPTIX_PIXEL_FORMAT_UCHAR4:
                strideElementAlignment = 1;
                break;
        }
        if( image.pixelStrideInBytes % strideElementAlignment != 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          corelib::stringf( "%s: pixel stride must be a multiple of %d for pixel "
                                                            "format 0x%x",
                                                            infoString, strideElementAlignment,
                                                            static_cast<unsigned int>( image.format ) ) );
    }
    return OPTIX_SUCCESS;
}

static OptixResult runDenoiser( Denoiser*                      denoiser,
                                LWstream                       stream,
                                const OptixDenoiserParams*     params,
                                LWdeviceptr                    denoiserState,
                                size_t                         denoiserStateSizeInBytes,
                                const OptixDenoiserGuideLayer* guideLayer,
                                const OptixDenoiserLayer*      layers,
                                unsigned int                   numLayers,
                                unsigned int                   inputOffsetX,
                                unsigned int                   inputOffsetY,
                                LWdeviceptr                    scratch,
                                size_t                         scratchSizeInBytes,
                                ErrorDetails&                  errDetails )

{
    OPTIX_CHECK_VALIDATION_MODE_LWRRENT_LWDA_CONTEXT( denoiser->getDeviceContext() );
    OPTIX_CHECK_VALIDATION_MODE_STREAM_STATE( denoiser->getDeviceContext(), stream );

    // check arguments
    if( guideLayer == nullptr )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "guideLayer must be nonzero" );

    if( layers == nullptr )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "layers must be nonzero" );

    if( numLayers == 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "numLayers must be > 0" );

    unsigned int width  = layers[0].input.width;
    unsigned int height = layers[0].input.height;

    if( denoiser->getInputKind() == OPTIX_DENOISER_INPUT_RGB_ALBEDO )
    {
        if( const OptixResult result = checkOptixImage2D( guideLayer->albedo, width, height, false, false, 0, "guide albedo", errDetails ) )
            return result;
    }

    if( denoiser->getInputKind() == OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL )
    {
        if( const OptixResult result = checkOptixImage2D( guideLayer->normal, width, height, true, false, 0, "guide normal", errDetails ) )
            return result;
    }

    if( denoiser->isTemporalModel() )
    {
        if( const OptixResult result = checkOptixImage2D( guideLayer->flow, width, height, true, false, 0, "guide flow", errDetails ) )
            return result;

        if( denoiser->isKernelPrediction() )
        {
            if( const OptixResult result = checkOptixImage2D( guideLayer->previousOutputInternalGuideLayer, width, height, false, true, denoiser->m_denoiser->getNumHiddenChannels(), "previousOutputInternalGuideLayer", errDetails ) )
                return result;

            if( const OptixResult result = checkOptixImage2D( guideLayer->outputInternalGuideLayer, width, height, false, true, denoiser->m_denoiser->getNumHiddenChannels(), "outputInternalGuideLayer", errDetails ) )
                return result;
        }
    }

    if( layers[0].input.format != layers[0].output.format && params->blendFactor > 0.f )
        return errDetails.logDetails( OPTIX_ERROR_NOT_SUPPORTED,
                                      corelib::stringf( "input pixel format 0x%x must be identical to "
                                                        "output pixel format 0x%x in blend mode",
                                                        layers[0].input.format, layers[0].output.format ) );

    for( unsigned int layerIndex = 0; layerIndex < numLayers; layerIndex++ )
    {
        const OptixDenoiserLayer& layer = layers[layerIndex];

        char infoString[1024];

        sprintf( infoString, "input (layer %u)", layerIndex );
        if( const OptixResult result = checkOptixImage2D( layer.input, width, height, false, false, 0, infoString, errDetails ) )
            return result;

        // output layer could have different dimensions in tiling
        sprintf( infoString, "output (layer %u)", layerIndex );
        if( const OptixResult result =
                checkOptixImage2D( layer.output, layer.output.width, layer.output.height, false, false, 0, infoString, errDetails ) )
            return result;

        if( denoiser->isTemporalModel() )
        {
            // direct prediction: previous denoised required only for beauty image, not AOVs
            // kernel prediction: previous denoised also required for AOVs
            if( layerIndex == 0 || denoiser->isKernelPrediction() )
            {
                sprintf( infoString, "previousOutput (layer %u)", layerIndex );
                if( const OptixResult result = checkOptixImage2D( layer.previousOutput, width, height, false, false, 0, infoString, errDetails ) )
                    return result;
            }
        }
    }

    // run denoiser

    denoiser->m_denoiser->setDenoiseAlpha( params->denoiseAlpha );

    if( !denoiser->isHdrModel() )
    {
        denoiser->m_denoiser->setHdrAverageColor( nullptr );
        denoiser->m_denoiser->setHdrIntensityFactor( nullptr );
    }
    else
    {
        if( denoiser->isKernelPrediction() )
        {
            denoiser->m_denoiser->setHdrAverageColor( (float*)(uintptr_t)params->hdrAverageColor );
            denoiser->m_denoiser->setHdrIntensityFactor( nullptr );
        }
        else
        {
            denoiser->m_denoiser->setHdrAverageColor( nullptr );
            denoiser->m_denoiser->setHdrIntensityFactor( (float*)(uintptr_t)params->hdrIntensity );
        }
    }

    denoiser->m_denoiser->setBlendFactor( params->blendFactor );

    std::vector<OptixImage2D> inputLayers;
    inputLayers.push_back( layers[0].input );

    if( denoiser->getInputKind() == OPTIX_DENOISER_INPUT_RGB_ALBEDO )
        inputLayers.push_back( guideLayer->albedo );
    else if( denoiser->getInputKind() == OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL )
    {
        inputLayers.push_back( guideLayer->albedo );
        inputLayers.push_back( guideLayer->normal );
    }

    if( denoiser->isTemporalModel() )
        inputLayers.push_back( layers[0].previousOutput );

    std::vector<OptixImage2D> outputLayers;
    outputLayers.push_back( layers[0].output );

    // AOVs
    for( int i = 1; i < numLayers; i++ )
    {
        if( denoiser->isTemporalModel() && denoiser->isKernelPrediction() )
            inputLayers.push_back( layers[i].previousOutput );
        inputLayers.push_back( layers[i].input );
        outputLayers.push_back( layers[i].output );
    }

    return denoiser->m_denoiser->denoiseTile( &inputLayers[0], (unsigned int)inputLayers.size(),
                                              &guideLayer->flow,
                                              &outputLayers[0],
                                              &guideLayer->previousOutputInternalGuideLayer,
                                              &guideLayer->outputInternalGuideLayer,
                                              inputOffsetX, inputOffsetY,
                                              (void*)(uintptr_t)scratch, scratchSizeInBytes, stream, errDetails );
}

extern "C" OptixResult optixDenoiserIlwoke( OptixDenoiser                  denoiserAPI,              /* [in] */
                                            LWstream                       stream,                   /* [in] */
                                            const OptixDenoiserParams*     params,                   /* [in] */
                                            LWdeviceptr                    denoiserState,            /* [in] */
                                            size_t                         denoiserStateSizeInBytes, /* [in] */
                                            const OptixDenoiserGuideLayer* guideLayer,               /* [in] */
                                            const OptixDenoiserLayer*      layers,                   /* [in] */
                                            unsigned int                   numLayers,                /* [in] */
                                            unsigned int                   inputOffsetX,             /* [in] */
                                            unsigned int                   inputOffsetY,             /* [in] */
                                            LWdeviceptr                    scratch,                  /* [in] */
                                            size_t                         scratchSizeInBytes )                              /* [in] */
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Denoiser, denoiser, "OptixDenoiser" );
    SCOPED_LWTX_RANGE( denoiser->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::DENOISER_ILWOKE );

    DeviceContextLogger& clog = denoiser->getLogger();

    try
    {
        ErrorDetails errDetails;
        OptixResult  result = OPTIX_SUCCESS;

        {
            std::lock_guard<std::mutex> lock( denoiser->getMutex() );
            result = runDenoiser( denoiser, stream, params, denoiserState, denoiserStateSizeInBytes, guideLayer, layers,
                                  numLayers, inputOffsetX, inputOffsetY,
                                  scratch, scratchSizeInBytes, errDetails );
        }
        if( result != OPTIX_SUCCESS )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

static OptixResult computeIntensity( Denoiser*           denoiser,
                                     LWstream            stream,
                                     const OptixImage2D* inputImage,
                                     LWdeviceptr         outputIntensity,
                                     LWdeviceptr         scratch,
                                     size_t              scratchSizeInBytes,
                                     ErrorDetails&       errDetails )
{
    OPTIX_CHECK_VALIDATION_MODE_LWRRENT_LWDA_CONTEXT( denoiser->getDeviceContext() );
    OPTIX_CHECK_VALIDATION_MODE_STREAM_STATE( denoiser->getDeviceContext(), stream );

    if( !( inputImage->format == OPTIX_PIXEL_FORMAT_HALF3 || inputImage->format == OPTIX_PIXEL_FORMAT_HALF4 ||
           inputImage->format == OPTIX_PIXEL_FORMAT_FLOAT3 || inputImage->format == OPTIX_PIXEL_FORMAT_FLOAT4 ) )
        return errDetails.logDetails( OPTIX_ERROR_NOT_SUPPORTED, corelib::stringf( "unsupported pixel format 0x%x", inputImage->format ) );

    return denoiseAutoexposure( inputImage, (float*)(uintptr_t)outputIntensity,
                                (void*)(uintptr_t)scratch, scratchSizeInBytes, stream, errDetails );
}

extern "C" OptixResult optixDenoiserComputeIntensity( OptixDenoiser       denoiserAPI,
                                                      LWstream            stream,
                                                      const OptixImage2D* inputImage,
                                                      LWdeviceptr         outputIntensity,
                                                      LWdeviceptr         scratch,
                                                      size_t              scratchSizeInBytes )

{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Denoiser, denoiser, "OptixDenoiser" );
    SCOPED_LWTX_RANGE( denoiser->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::DENOISER_COMPUTE_INTENSITY );

    DeviceContextLogger& clog = denoiser->getLogger();
    OPTIX_CHECK_NULL_ARGUMENT( inputImage );
    OPTIX_CHECK_NULL_ARGUMENT( (void*)outputIntensity );
    OPTIX_CHECK_NULL_ARGUMENT( (void*)scratch );

    try
    {
        ErrorDetails errDetails;
        OptixResult  result =
            computeIntensity( denoiser, stream, inputImage, outputIntensity, scratch, scratchSizeInBytes, errDetails );
        if( result != OPTIX_SUCCESS )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

static OptixResult computeAverageColor( Denoiser*           denoiser,
                                        LWstream            stream,
                                        const OptixImage2D* inputImage,
                                        LWdeviceptr         outputAverageColor,
                                        LWdeviceptr         scratch,
                                        size_t              scratchSizeInBytes,
                                        ErrorDetails&       errDetails )
{
    OPTIX_CHECK_VALIDATION_MODE_LWRRENT_LWDA_CONTEXT( denoiser->getDeviceContext() );
    OPTIX_CHECK_VALIDATION_MODE_STREAM_STATE( denoiser->getDeviceContext(), stream );

    if( !( inputImage->format == OPTIX_PIXEL_FORMAT_HALF3 || inputImage->format == OPTIX_PIXEL_FORMAT_HALF4 ||
           inputImage->format == OPTIX_PIXEL_FORMAT_FLOAT3 || inputImage->format == OPTIX_PIXEL_FORMAT_FLOAT4 ) )
        return errDetails.logDetails( OPTIX_ERROR_NOT_SUPPORTED, corelib::stringf( "unsupported pixel format 0x%x", inputImage->format ) );

    return denoiseRGBAverage( inputImage, (float*)(uintptr_t)outputAverageColor,
                              (void*)(uintptr_t)scratch, scratchSizeInBytes, stream, errDetails );
}

extern "C" OptixResult optixDenoiserComputeAverageColor( OptixDenoiser       denoiserAPI,
                                                         LWstream            stream,
                                                         const OptixImage2D* inputImage,
                                                         LWdeviceptr         outputAverageColor,
                                                         LWdeviceptr         scratch,
                                                         size_t              scratchSizeInBytes )

{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Denoiser, denoiser, "OptixDenoiser" );
    SCOPED_LWTX_RANGE( denoiser->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::DENOISER_COMPUTE_INTENSITY );

    DeviceContextLogger& clog = denoiser->getLogger();
    OPTIX_CHECK_NULL_ARGUMENT( inputImage );
    OPTIX_CHECK_NULL_ARGUMENT( (void*)outputAverageColor );
    OPTIX_CHECK_NULL_ARGUMENT( (void*)scratch );

    try
    {
        ErrorDetails errDetails;
        OptixResult  result =
            computeAverageColor( denoiser, stream, inputImage, outputAverageColor, scratch, scratchSizeInBytes, errDetails );
        if( result != OPTIX_SUCCESS )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

OptixResult Denoiser::init_v1( OptixDenoiserInputKind_v1 inputKind, ErrorDetails& errDetails )
{
    OptixResult result;

    m_inferenceMode = inputKind;

    m_denoiser = new Denoise( m_context->getLogger() );

    if( ( result = m_denoiser->init( errDetails ) ) )
        destroy( false, errDetails );

    return result;
}

extern "C" OptixResult optixDenoiserCreate_v1( OptixDeviceContext contextAPI, const OptixDenoiserOptions_v1* options, OptixDenoiser* denoiserAPI )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( DeviceContext, context, "OptixDeviceContext" );
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DENOISER_CREATE );

    DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( options );
    OPTIX_CHECK_NULL_ARGUMENT( denoiserAPI );
    *denoiserAPI = nullptr;

    try
    {
        std::unique_ptr<Denoiser> denoiser( new Denoiser( context ) );
        ErrorDetails              errDetails;
        OptixResult               result = denoiser.get()->init_v1( options->inputKind, errDetails );
        if( result == OPTIX_SUCCESS )
            result = context->registerDenoiser( denoiser.get(), errDetails );
        else
        {
            clog.sendError( errDetails );
            return result;
        }
        *denoiserAPI = apiCast( denoiser.release() );
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

struct OptixDenoiserParams_ABI37
{
    unsigned int denoiseAlpha;
    LWdeviceptr  hdrIntensity;
    float        blendFactor;
};

static OptixResult runDenoiser_v1( Denoiser*                  denoiser,
                                   LWstream                   stream,
                                   const OptixDenoiserParams* params,
                                   LWdeviceptr                denoiserState,
                                   size_t                     denoiserStateSizeInBytes,
                                   const OptixImage2D*        inputLayers,
                                   unsigned int               numInputLayers,
                                   unsigned int               inputOffsetX,
                                   unsigned int               inputOffsetY,
                                   const OptixImage2D*        outputLayer,
                                   LWdeviceptr                scratch,
                                   size_t                     scratchSizeInBytes,
                                   ErrorDetails&              errDetails )
{
    OptixDenoiserParams params38;
    if( denoiser->getDeviceContext()->getAbiVersion() < OptixABI::ABI_38 )
    {
        OptixDenoiserParams_ABI37* p = (OptixDenoiserParams_ABI37*)params;
        params38.denoiseAlpha        = (OptixDenoiserAlphaMode)p->denoiseAlpha;
        params38.hdrIntensity        = p->hdrIntensity;
        params38.blendFactor         = p->blendFactor;
        params38.hdrAverageColor     = 0;
    }
    else
    {
        params38 = *params;
    }
    params = &params38;

    if( denoiser->isTemporalModel() )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "temporal model kinds not supported in the ABI used by the application" );

    if( inputLayers == nullptr )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "inputLayers must be nonzero" );

    if( outputLayer == nullptr )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "outputLayer must be nonzero" );

    unsigned int nLayer = 0, lwrOutput = 0;
    // add beauty
    std::vector<OptixDenoiserLayer> layers;
    if( nLayer >= numInputLayers )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "beauty layer missing in inputLayers" );

    OptixDenoiserLayer lwrLayer = {};
    lwrLayer.input  = inputLayers[nLayer++];
    lwrLayer.output = outputLayer[lwrOutput++];
    layers.push_back( lwrLayer );

    OptixDenoiserGuideLayer guideLayer = {};
    if( denoiser->getInputKind() == OPTIX_DENOISER_INPUT_RGB_ALBEDO ||
        denoiser->getInputKind() == OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL )
    {
        if( nLayer >= numInputLayers )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "albedo layer missing in inputLayers" );
        guideLayer.albedo = inputLayers[nLayer++];
    }
    if( denoiser->getInputKind() == OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL )
    {
        if( nLayer >= numInputLayers )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "normal layer missing in inputLayers" );
        guideLayer.normal = inputLayers[nLayer++];
    }

    // add AOVs
    while( nLayer < numInputLayers )
    {
        lwrLayer.input  = inputLayers[nLayer++];
        lwrLayer.output = outputLayer[lwrOutput++];
        layers.push_back( lwrLayer );
    }

    return runDenoiser( denoiser, stream, params, denoiserState, denoiserStateSizeInBytes,
                        &guideLayer, &layers[0], (unsigned int)layers.size(),
                        inputOffsetX, inputOffsetY, scratch, scratchSizeInBytes, errDetails );
}

extern "C" OptixResult optixDenoiserIlwoke_v1( OptixDenoiser              denoiserAPI,              /* [in] */
                                               LWstream                   stream,                   /* [in] */
                                               const OptixDenoiserParams* params,                   /* [in] */
                                               LWdeviceptr                denoiserState,            /* [in] */
                                               size_t                     denoiserStateSizeInBytes, /* [in] */
                                               const OptixImage2D*        inputLayers,              /* [in] */
                                               unsigned int               numInputLayers,           /* [in] */
                                               unsigned int               inputOffsetX,             /* [in] */
                                               unsigned int               inputOffsetY,             /* [in] */
                                               const OptixImage2D*        outputLayer,              /* [in] */
                                               LWdeviceptr                scratch,                  /* [in] */
                                               size_t                     scratchSizeInBytes )      /* [in] */
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Denoiser, denoiser, "OptixDenoiser" );
    SCOPED_LWTX_RANGE( denoiser->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::DENOISER_ILWOKE );

    DeviceContextLogger& clog = denoiser->getLogger();

    try
    {
        ErrorDetails errDetails;
        OptixResult  result = OPTIX_SUCCESS;

        {
            std::lock_guard<std::mutex> lock( denoiser->getMutex() );
            result = runDenoiser_v1( denoiser, stream, params, denoiserState, denoiserStateSizeInBytes, inputLayers, numInputLayers,
                                     inputOffsetX, inputOffsetY,
                                     outputLayer, scratch, scratchSizeInBytes, errDetails );
        }
        if( result != OPTIX_SUCCESS )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixDenoiserSetModel_v1( OptixDenoiser denoiserAPI, OptixDenoiserModelKind_v1 kind, void* data, size_t sizeInBytes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Denoiser, denoiser, "OptixDenoiser" );
    SCOPED_LWTX_RANGE( denoiser->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::DENOISER_SET_MODEL );

    DeviceContextLogger& clog = denoiser->getLogger();

    try
    {
        ErrorDetails errDetails;
        OptixResult  result;
        {
            std::lock_guard<std::mutex> lock( denoiser->getMutex() );
            if( kind == OPTIX_DENOISER_MODEL_KIND_USER_v1 )
                result = denoiser->loadUserModel( data, sizeInBytes, errDetails );
            else
                result = denoiser->loadBuiltinModel( (OptixDenoiserModelKind)kind,  errDetails );
        }
        if( result )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

}  // namespace optix_exp
