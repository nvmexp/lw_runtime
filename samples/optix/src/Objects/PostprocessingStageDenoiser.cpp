// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <LWCA/Memory.h>

#include <Objects/PostprocessingStageDenoiser.h>

#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Memory/MemoryManager.h>
#include <Objects/CommandList.h>
#include <Objects/Variable.h>
#include <Util/LinkedPtrHelpers.h>
#include <prodlib/exceptions/IlwalidValue.h>

#include <cstdio>
#include <lwda_runtime.h>

#define __OPTIX_INCLUDE_INTERNAL_HEADERS__
#include <optix_7_host.h>

#include <exp/context/DeviceContext.h>
#include <exp/denoise/internal.h>
#include <optix_denoiser_tiling.h>

namespace optix {

PostprocessingStageDenoiser::PostprocessingStageDenoiser( Context* context, void* denoiser )
    : PostprocessingStage( context, "DLDenoiser" )
{
    m_optixDevice = nullptr;
    m_denoiser    = nullptr;
    m_stateMem    = nullptr;
    m_scratchMem  = nullptr;

    m_stateSizeInBytes   = 0;
    m_scratchSizeInBytes = 0;

    m_width  = 0;
    m_height = 0;
}

PostprocessingStageDenoiser::~PostprocessingStageDenoiser()
{
    destroy();
}

void PostprocessingStageDenoiser::destroy()
{
    deleteVariables();

    if( m_stateMem )
        lwdaFree( m_stateMem );

    if( m_scratchMem )
        lwdaFree( m_scratchMem );

    m_stateSizeInBytes   = 0;
    m_scratchSizeInBytes = 0;

    if( m_denoiser )
        optixDenoiserDestroy( m_denoiser );

    if( m_optixDevice )
        optixDeviceContextDestroy( m_optixDevice );
}

void PostprocessingStageDenoiser::initialize( RTsize width, RTsize height )
{
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &DenoiserLogBuffer::callback;
    options.logCallbackData           = &m_logBuffer;
    options.logCallbackLevel          = 3;

    LWcontext lwCtx = nullptr;

    LWDADevice* lwdaDevice = getContext()->getDeviceManager()->primaryLWDADevice();
    if( !lwdaDevice )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                     "Denoiser: Failed to get primary LWCA "
                                     "device." );

    lwdaDevice->makeLwrrent();

    if( optix_exp::optixDeviceContextCreate_lwrrent( lwCtx, &options, &m_optixDevice ) )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                     m_logBuffer.getMessage( "Denoiser: device context creation failed" ) );

    m_trainingDataDirty = true;
}

void PostprocessingStageDenoiser::validate() const
{
    PostprocessingStage::validate();

    Variable* vInBuffer     = queryVariable( "input_buffer" );
    Variable* vAlbedoBuffer = queryVariable( "input_albedo_buffer" );
    Variable* vNormalBuffer = queryVariable( "input_normal_buffer" );
    Variable* vOutBuffer    = queryVariable( "output_buffer" );

    // Variable that contains the optional training data to use. If not specified the default
    // built-in training data will be used.
    Variable* vTrainingData = queryVariable( "training_data_buffer" );

    if( vAlbedoBuffer && !vAlbedoBuffer->isBuffer() && vAlbedoBuffer->getType() != VariableType() )
    {
        // Variable is set (type not unknown) but not to a buffer. This is not allowed.
        throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                     "Denoiser: input_albedo_buffer variable set to non-buffer type." );
    }

    if( vNormalBuffer && !vNormalBuffer->isBuffer() && vNormalBuffer->getType() != VariableType() )
    {
        // Variable is set (type not unknown) but not to a buffer. This is not allowed.
        throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                     "Denoiser: input_normal_buffer variable set to non-buffer type." );
    }

    if( vTrainingData && !vTrainingData->isBuffer() && vTrainingData->getType() != VariableType() )
    {
        // Variable is set (type not unknown) but not to a buffer. This is not allowed.
        throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                     "Denoiser: training_data_buffer variable set to non-buffer type." );
    }

    Buffer* inBuffer     = ( vInBuffer && vInBuffer->isBuffer() ) ? vInBuffer->getBuffer() : nullptr;
    Buffer* outBuffer    = ( vOutBuffer && vOutBuffer->isBuffer() ) ? vOutBuffer->getBuffer() : nullptr;
    Buffer* albedoBuffer = ( vAlbedoBuffer && vAlbedoBuffer->isBuffer() ) ? vAlbedoBuffer->getBuffer() : nullptr;
    Buffer* normalBuffer = ( vNormalBuffer && vNormalBuffer->isBuffer() ) ? vNormalBuffer->getBuffer() : nullptr;

    // Treat empty albedo and normal buffers as if they weren't set:
    if( albedoBuffer && ( albedoBuffer->getWidth() == 0 || albedoBuffer->getHeight() == 0 ) )
        albedoBuffer = nullptr;
    if( normalBuffer && ( normalBuffer->getWidth() == 0 || normalBuffer->getHeight() == 0 ) )
        normalBuffer = nullptr;

    if( !inBuffer || !outBuffer )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Denoiser: input_buffer and output_buffer must be declared and must be of type "
                                     "buffer." );

    if( ( inBuffer->getWidth() != outBuffer->getWidth() ) || ( inBuffer->getHeight() != outBuffer->getHeight() ) )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Denoiser: input_buffer and output_buffer do not have the same dimensions." );

    // TODO: support more buffer combinations
    if( !( ( inBuffer->getFormat() == RT_FORMAT_FLOAT4 )
           && ( ( outBuffer->getFormat() == RT_FORMAT_FLOAT4 ) || ( outBuffer->getFormat() == RT_FORMAT_UNSIGNED_BYTE4 ) ) ) )
        throw prodlib::IlwalidValue(
            RT_EXCEPTION_INFO, "Denoiser: input_buffer and output_buffer have an unsupported format combination." );

    if( albedoBuffer
        && ( ( inBuffer->getWidth() != albedoBuffer->getWidth() ) || ( inBuffer->getHeight() != albedoBuffer->getHeight() ) ) )
        throw prodlib::IlwalidValue(
            RT_EXCEPTION_INFO, "Denoiser: input_albedo_buffer and input_buffer do not have the same dimensions." );

    if( albedoBuffer && ( albedoBuffer->getFormat() != RT_FORMAT_FLOAT4 ) )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Denoiser: input_albedo_buffer has an unsupported format." );

    if( normalBuffer && ( normalBuffer->getFormat() != RT_FORMAT_FLOAT4 ) )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Denoiser: input_normal_buffer has an unsupported format." );

    if( !albedoBuffer && normalBuffer )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Denoiser: A normal buffer without an albedo buffer is lwrrently not supported." );
}

void PostprocessingStageDenoiser::doLaunch( RTsize width, RTsize height )
{
    Variable* vInBuffer     = queryVariable( "input_buffer" );
    Variable* vAlbedoBuffer = queryVariable( "input_albedo_buffer" );
    Variable* vNormalBuffer = queryVariable( "input_normal_buffer" );
    Variable* vOutBuffer    = queryVariable( "output_buffer" );

    // A float that defines a split of the image that won't be denoised. Defaults to 0.
    // 0..1 : left non denoised, right denoised
    // 1..2 : <->
    // 2..3 : bottom non denoised, top denoised
    // 3..4 : <->

    // A float that defines how much of the original image should be blended with the denoised result.
    // 0 means full denoised image and 1 means full original image. Defaults to 0.
    Variable* vBlend = queryVariable( "blend" );

    // A float (integral part taken) that defines the maximum GPU memory in bytes used by the denoiser.
    // If denoising takes more memory the image is split into tiles to stay below the specified maximum
    // amount of memory.
    Variable* vMaxmem = queryVariable( "maxmem" );

    // A Uint variable that defines how the alpha channel is denoised. By default the alpha channel
    // of the source image is copied to the result image. If this variable is set to nonzero, the alpha
    // channel is denoised as well instead of copying it from the noisy input image.
    Variable* vDenoiseAlpha = queryVariable( "denoise_alpha" );

    Buffer* inBuffer     = ( vInBuffer && vInBuffer->isBuffer() ) ? vInBuffer->getBuffer() : nullptr;
    Buffer* outBuffer    = ( vOutBuffer && vOutBuffer->isBuffer() ) ? vOutBuffer->getBuffer() : nullptr;
    Buffer* albedoBuffer = ( vAlbedoBuffer && vAlbedoBuffer->isBuffer() ) ? vAlbedoBuffer->getBuffer() : nullptr;
    Buffer* normalBuffer = ( vNormalBuffer && vNormalBuffer->isBuffer() ) ? vNormalBuffer->getBuffer() : nullptr;

    // Treat empty buffers as if they weren't set:
    if( albedoBuffer && ( albedoBuffer->getWidth() == 0 || albedoBuffer->getHeight() == 0 ) )
        albedoBuffer = nullptr;

    //temporary disabled normalBuffer support
    //if( normalBuffer && ( normalBuffer->getWidth() == 0 || normalBuffer->getHeight() == 0 ) )
    normalBuffer = nullptr;

    if( !inBuffer || !outBuffer )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Denoiser: input_buffer and output_buffer must be declared and must be of type "
                                     "buffer." );

    // Check the "hdr" variable to see if the user indicated that it will feed hdr images. This is
    // used to select the correct default training data set.
    Variable* vHdr = queryVariable( "hdr" );
    if( vHdr )
    {
        if( vHdr->getType().baseType() == VariableType::Uint )
        {
            bool useHDR = vHdr->get<uint>() != 0;
            if( useHDR != m_useHdrTrainingData )
            {
                m_useHdrTrainingData = useHDR;
                m_trainingDataDirty  = true;
            }
        }
        else
        {
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                         "Denoiser: Variable \"hdr\" has the "
                                         "wrong type. "
                                         "Should be unsigned int." );
        }
    }

    // Check the "autoexposure" variable. Default is true in HDR mode.
    bool      autoexposure = m_useHdrTrainingData;
    Variable* vAE          = queryVariable( "autoexposure" );
    if( vAE )
    {
        if( vAE->getType().baseType() == VariableType::Uint )
        {
            autoexposure = vAE->get<uint>() != 0;
        }
        else
        {
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                         "Denoiser: Variable \"autoexposure\" "
                                         "has the wrong type. "
                                         "Should be unsigned int." );
        }
    }

    LWDADevice* lwdaDevice = getContext()->getDeviceManager()->primaryLWDADevice();
    if( !lwdaDevice )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                     "Denoiser: Failed to get primary LWCA "
                                     "device." );

    lwdaDevice->makeLwrrent();

    OptixDenoiserOptions denoiserOptions = {};
    if( albedoBuffer && normalBuffer )
    {
        denoiserOptions.guideAlbedo = 1;
        denoiserOptions.guideNormal = 1;
    } else if( albedoBuffer )
    {
        denoiserOptions.guideAlbedo = 1;
    }

    // Set the training data, either to custom or built-in, if it has been marked as dirty.
    Buffer* trainingDataBuffer = nullptr;
    const unsigned char* trainingData       = nullptr;
    unsigned int         trainingDataLength = 0;

    if( m_trainingDataDirty )
    {
        // Treat an empty training data buffer as if no buffer was set.
        trainingDataBuffer = m_trainingDataBufferLink.get();
        if( trainingDataBuffer && ( trainingDataBuffer->getWidth() == 0 ) )
            trainingDataBuffer = nullptr;

        // Either use custom training data or the default built-in if not set.
        if( trainingDataBuffer )
        {
            if( trainingDataBuffer->isMappedHost() )
                trainingDataBuffer->unmap();

            const unsigned char* data = reinterpret_cast<const unsigned char*>( trainingDataBuffer->map( MAP_READ ) );
            trainingDataLength        = trainingDataBuffer->getWidth();
            trainingData              = data;
        }
    }

    bool needsSetup = false;

    if( !m_denoiser || !!(denoiserOptions.guideAlbedo) != !!(m_denoiserOptions.guideAlbedo) ||
                       !!(denoiserOptions.guideNormal) != !!(m_denoiserOptions.guideNormal) ||
                       m_trainingDataDirty )
    {
        if( m_denoiser )
            if( optixDenoiserDestroy( m_denoiser ) )
                throw prodlib::UnknownError( RT_EXCEPTION_INFO, m_logBuffer.getMessage(
                                                                    "Denoiser: failed to destroy denoiser instance" ) );

        if( trainingData )
        {
            if( optix_exp::optixDenoiserCreateWithUserModel_internal( m_optixDevice, &denoiserOptions, (void*)trainingData, trainingDataLength, &m_denoiser ) )
                throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                             m_logBuffer.getMessage( "Denoiser: failed to create denoiser instance (with user model)" ) );
        }
        else
        {
            if( optixDenoiserCreate( m_optixDevice, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiserOptions, &m_denoiser ) )
                throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                             m_logBuffer.getMessage( "Denoiser: failed to create denoiser instance" ) );
        }
        m_trainingDataDirty = false;
        needsSetup = true;
    }
    m_denoiserOptions = denoiserOptions;

    float blend = 0.0f;
    if( vBlend )
    {
        if( vBlend->getType().baseType() == VariableType::Float )
            blend = vBlend->get<float>();
        else
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                         "Denoiser: Variable \"blend\" has "
                                         "the wrong type. "
                                         "Should be float." );
    }


    size_t maxmem = 0;
    if( vMaxmem )
    {
        if( vMaxmem->getType().baseType() == VariableType::Float )
            maxmem = static_cast<size_t>( vMaxmem->get<float>() );
        else if( vMaxmem->getType().baseType() == VariableType::ULongLong )
            maxmem = static_cast<size_t>( vMaxmem->get<unsigned long long>() );
        else
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                         "Denoiser: Variable \"maxmem\" has "
                                         "the wrong type. "
                                         "Should be unsigned long long." );
    }

    int denoiseAlpha = 0;
    if( vDenoiseAlpha )
    {
        if( vDenoiseAlpha->getType().baseType() == VariableType::Uint )
            denoiseAlpha = vDenoiseAlpha->get<uint>();
        else
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                         "Denoiser: Variable "
                                         "\"denoise_alpha\" has the wrong type. "
                                         "Should be unsigned int." );
    }

    // Get device data pointers
    MBufferHandle inputMBuffer      = inBuffer->getMBuffer();
    MAccess       inputBufferAccess = inputMBuffer->getAccess( lwdaDevice->allDeviceListIndex() );
    if( inputBufferAccess.getKind() != MAccess::LINEAR )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Denoiser: Failed to get linear device "
                                     "buffer for input buffer." );
    const float* inData = reinterpret_cast<const float*>( inputBufferAccess.getLinearPtr() );

    MBufferHandle outputMBuffer      = outBuffer->getMBuffer();
    MAccess       outputBufferAccess = outputMBuffer->getAccess( lwdaDevice->allDeviceListIndex() );
    if( inputBufferAccess.getKind() != MAccess::LINEAR )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Denoiser: Failed to get linear device "
                                     "buffer for output buffer." );
    float* outData = reinterpret_cast<float*>( outputBufferAccess.getLinearPtr() );

    const float*  albedoData    = nullptr;
    MBufferHandle albedoMBuffer = albedoBuffer ? albedoBuffer->getMBuffer() : nullptr;
    if( albedoBuffer )
    {
        MAccess albedoBufferAccess = albedoMBuffer->getAccess( lwdaDevice->allDeviceListIndex() );
        if( albedoBufferAccess.getKind() != MAccess::LINEAR )
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                         "Denoiser: Failed to get linear "
                                         "device buffer for albedo buffer." );
        albedoData = reinterpret_cast<const float*>( albedoBufferAccess.getLinearPtr() );
    }

    const float*  normalData    = nullptr;
    MBufferHandle normalMBuffer = normalBuffer ? normalBuffer->getMBuffer() : nullptr;
    if( normalBuffer )
    {
        MAccess normalBufferAccess = normalMBuffer->getAccess( lwdaDevice->allDeviceListIndex() );
        if( normalBufferAccess.getKind() != MAccess::LINEAR )
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                         "Denoiser: Failed to get linear "
                                         "device buffer for normal buffer." );
        normalData = reinterpret_cast<const float*>( normalBufferAccess.getLinearPtr() );
    }

    // Check if the input buffer can be copied directly to the output buffer
    bool straightCopy = blend == 1.f && outBuffer->getFormat() == inBuffer->getFormat();

    // Check if the input/output buffer is allocated on slow memory in which case it will have to be
    // copied to the GPU before passed to the tonemapper and then back. This is more efficient
    // than operating on a zero copy buffer which is located in host memory.
    float* gpuOutDevPtr = static_cast<float*>( copyToGpuIfNeeded( outputMBuffer, outData, lwdaDevice ) );
    float* gpuInDevPtr =
        straightCopy ? const_cast<float*>( inData ) :
                       static_cast<float*>( copyToGpuIfNeeded( inputMBuffer, const_cast<float*>( inData ), lwdaDevice ) );
    float* gpuAlbedoDevPtr =
        static_cast<float*>( copyToGpuIfNeeded( albedoMBuffer, const_cast<float*>( albedoData ), lwdaDevice ) );
    float* gpuNormalDevPtr =
        static_cast<float*>( copyToGpuIfNeeded( normalMBuffer, const_cast<float*>( normalData ), lwdaDevice ) );

    OptixDenoiserSizes sizes;
    if( optixDenoiserComputeMemoryResources( m_denoiser, width, height, &sizes ) )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                     m_logBuffer.getMessage( "Denoiser: getting memory resources failed" ) );

    // determine tile size
    unsigned int tile_height = height;
    size_t       mem_free, mem_total;
    lwdaMemGetInfo( &mem_free, &mem_total );
    if( maxmem != 0 && maxmem < mem_free )
        mem_free = maxmem;
    float mem    = sizes.withoutOverlapScratchSizeInBytes + sizes.stateSizeInBytes;
    float ovm    = mem * ( 2.f * sizes.overlapWindowSizeInPixels ) / height;
    // callwlate the necessary tile size
    float r = ( float( mem_free ) - ovm ) / mem;
    if( r < 0.f )
        r = 0.f;
    if( r < 1.1f )
    {
        // start tiling slightly earlier than necessary
        if( r >= 1.f )
            r       = 1.f;
        tile_height = 1 + int( height * r * 0.9 );

        tile_height = ( ( tile_height + 32 - 1 ) / 32 ) * 32;  // pad to multiples of 32
        if( tile_height < 32 )
            tile_height = 32;  // minimum tile size
        if( height < tile_height )
            tile_height = height;
        if( tile_height > height / 2 && tile_height < height )
            tile_height = height / 2;
    }

    unsigned int overlap = ( tile_height < height ) ? sizes.overlapWindowSizeInPixels : 0;

    if( optixDenoiserComputeMemoryResources( m_denoiser, width, tile_height, &sizes ) )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                     m_logBuffer.getMessage( "Denoiser: getting memory resources failed" ) );
    if( sizes.stateSizeInBytes != m_stateSizeInBytes )
    {
        if( m_stateMem )
        {
            if( lwdaFree( m_stateMem ) )
                throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Denoiser: releasing state memory failed" );
        }
        if( lwdaMalloc( &m_stateMem, sizes.stateSizeInBytes ) )
            throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Denoiser: allocating state memory failed" );
        m_stateSizeInBytes = sizes.stateSizeInBytes;
        needsSetup = true;      // Probably redundant, see below. Dimensions have changed.
    }

    if( width != m_width || tile_height + 2 * overlap != m_height )
    {
        m_width    = width;
        m_height   = tile_height + 2 * overlap;
        needsSetup = true;
    }

    size_t ssize = overlap ? sizes.withOverlapScratchSizeInBytes : sizes.withoutOverlapScratchSizeInBytes;
    // we allocate scratch + float for the intensity result in a single buffer
    if( ssize + sizeof( float ) != m_scratchSizeInBytes )
    {
        if( m_scratchMem )
        {
            if( lwdaFree( m_scratchMem ) )
                throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Denoiser: releasing scratch memory failed" );
        }
        if( lwdaMalloc( &m_scratchMem, ssize + sizeof( float ) ) )
            throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Denoiser: allocation of scratch memory failed" );
        m_scratchSizeInBytes = ssize + sizeof( float );
    }
    void* intensityMem = (char*)m_scratchMem + ssize;

    LWstream stream = lwdaDevice->primaryStream().get();

    if( needsSetup &&
        optixDenoiserSetup( m_denoiser, stream, width, tile_height + 2 * overlap, (LWdeviceptr)m_stateMem,
                            sizes.stateSizeInBytes, (LWdeviceptr)m_scratchMem, ssize ) )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO, m_logBuffer.getMessage( "Denoiser: setup failed" ) );

    OptixDenoiserParams params;
    params.denoiseAlpha    = denoiseAlpha ? OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV : OPTIX_DENOISER_ALPHA_MODE_COPY;
    params.hdrIntensity    = autoexposure ? (LWdeviceptr)intensityMem : 0;
    params.blendFactor     = blend;
    params.hdrAverageColor = 0;

    unsigned int rsb = width * sizeof( float ) * 4;

    OptixDenoiserLayer inOutLayer = {};
    inOutLayer.input  = {(LWdeviceptr)gpuInDevPtr, (unsigned int)width, (unsigned int)height, rsb, 0, OPTIX_PIXEL_FORMAT_FLOAT4};
    inOutLayer.output = {(LWdeviceptr)gpuOutDevPtr, (unsigned int)width, (unsigned int)height, rsb, 0, OPTIX_PIXEL_FORMAT_FLOAT4};

    OptixDenoiserGuideLayer guideLayer = {};
    if( gpuAlbedoDevPtr )
        guideLayer.albedo = {(LWdeviceptr)gpuAlbedoDevPtr, (unsigned int)width, (unsigned int)height, rsb, 0, OPTIX_PIXEL_FORMAT_FLOAT4};
    if( gpuNormalDevPtr )
        guideLayer.normal = {(LWdeviceptr)gpuNormalDevPtr, (unsigned int)width, (unsigned int)height, rsb, 0, OPTIX_PIXEL_FORMAT_FLOAT4};

    if( params.hdrIntensity )
    {
        if( optixDenoiserComputeIntensity( m_denoiser, stream, &inOutLayer.input, params.hdrIntensity, (LWdeviceptr)m_scratchMem, ssize ) )
            throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                         m_logBuffer.getMessage( "Denoiser: intensity callwlation failed" ) );
    }

    if( straightCopy )
        lwca::memcpyDtoD( (LWdeviceptr)outData, (LWdeviceptr)inData, inBuffer->getDimensions().getTotalSizeInBytes() );
    else
    {
        if( optixUtilDenoiserIlwokeTiled( m_denoiser, stream, &params, (LWdeviceptr)m_stateMem, sizes.stateSizeInBytes,
                                          &guideLayer, &inOutLayer, 1, (LWdeviceptr)m_scratchMem, ssize, overlap,
                                          width, tile_height ) )
            throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                         m_logBuffer.getMessage( "Denoiser: failed to denoise image" ) );
    }

    lwdaStreamSynchronize( stream );

    if( m_context )
        m_context->incrDenoiserLaunchCount();

    if( trainingDataBuffer && trainingDataBuffer->isMappedHost() )
        trainingDataBuffer->unmap();

    // If a temp gpu pointer was used then copy back to the application provided buffer and free the temp one.
    if( outData != gpuOutDevPtr )
    {
        lwdaError_t error = lwdaMemcpy( outData, gpuOutDevPtr, outputMBuffer->getDimensions().getTotalSizeInBytes(),
                                        lwdaMemcpyDeviceToDevice );
        if( error != lwdaSuccess )
            throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                         "Denoiser: failed to copy device buffer back to zero copy memory." );
        lwdaFree( gpuOutDevPtr );
    }

    if( inData != gpuInDevPtr )
        lwdaFree( gpuInDevPtr );

    if( albedoData != gpuAlbedoDevPtr )
        lwdaFree( gpuAlbedoDevPtr );

    if( normalData != gpuNormalDevPtr )
        lwdaFree( gpuNormalDevPtr );
}

void PostprocessingStageDenoiser::detachLinkedChild( const LinkedPtr_Link* link )
{
    if( ( &m_trainingDataBufferLink ) == link )
        m_trainingDataBufferLink.reset();
}

void PostprocessingStageDenoiser::bufferWasMapped( LinkedPtr_Link* link )
{
    if( ( &m_trainingDataBufferLink ) == link )
    {
        m_trainingDataDirty = true;
    }
}

void PostprocessingStageDenoiser::bufferFormatDidChange( LinkedPtr_Link* link )
{
    if( ( &m_trainingDataBufferLink ) == link )
    {
        m_trainingDataDirty = true;
    }
}

void PostprocessingStageDenoiser::bufferVariableValueDidChange( Variable* var, Buffer* oldBuffer, Buffer* newBuffer )
{
    PostprocessingStage::bufferVariableValueDidChange( var, oldBuffer, newBuffer );

    if( var->getName() == "training_data_buffer" )
    {
        if( m_trainingDataBufferLink )
            m_trainingDataBufferLink.reset();
        m_trainingDataBufferLink.set( this, newBuffer );

        m_trainingDataDirty = true;
    }
}
}  // namespace optix
