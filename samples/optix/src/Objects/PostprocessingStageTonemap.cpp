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

#include <Objects/PostprocessingStageTonemap.h>
#include <Objects/PostprocessingStageTonemapLWDA.h>

#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/TableManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Memory/MemoryManager.h>
#include <Objects/CommandList.h>
#include <Objects/Variable.h>
#include <Util/LinkedPtrHelpers.h>

#include <prodlib/exceptions/IlwalidValue.h>

#include <corelib/system/LwdaDriver.h>
#include <corelib/system/System.h>
#include <corelib/system/Timer.h>

#include <LWCA/ErrorCheck.h>
#include <LWCA/Function.h>
#include <LWCA/Module.h>
#include <LWCA/Stream.h>

#include <lwda_runtime.h>

using namespace corelib;

namespace optix {

PostprocessingStageTonemap::PostprocessingStageTonemap( Context* context )
    : PostprocessingStage( context, "TonemapperSimple" )
{
}

PostprocessingStageTonemap::~PostprocessingStageTonemap()
{
    deleteVariables();
}

void PostprocessingStageTonemap::initialize( RTsize width, RTsize height )
{
}

void PostprocessingStageTonemap::validate() const
{
    PostprocessingStage::validate();

    Variable* vInBuffer  = queryVariable( "input_buffer" );
    Variable* vOutBuffer = queryVariable( "output_buffer" );


    if( !vInBuffer || !vOutBuffer )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Error in TonemapperSimple post-processing stage. input_buffer output_buffer must "
                                     "be "
                                     "declared." );

    Buffer* inBuffer  = vInBuffer->isBuffer() ? vInBuffer->getBuffer() : nullptr;
    Buffer* outBuffer = vOutBuffer->isBuffer() ? vOutBuffer->getBuffer() : nullptr;

    if( !inBuffer || !outBuffer )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Error in TonemapperSimple post-processing stage. input_buffer and/or "
                                     "output_buffer "
                                     "are not buffers." );

    if( ( inBuffer->getWidth() != outBuffer->getWidth() ) || ( inBuffer->getHeight() != outBuffer->getHeight() ) )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Error in TonemapperSimple post-processing stage. input_buffer and output_buffer "
                                     "do "
                                     "not have the same dimensions." );

    // TODO: support more buffer combinations
    if( !( ( inBuffer->getFormat() == RT_FORMAT_FLOAT4 ) && ( outBuffer->getFormat() == RT_FORMAT_FLOAT4 ) ) )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Error in TonemapperSimple post-processing stage. input_buffer and output_buffer "
                                     "have "
                                     "an unsupported format combination." );
}

void PostprocessingStageTonemap::doLaunch( RTsize width, RTsize height )
{
    Variable* vInBuffer  = queryVariable( "input_buffer" );
    Variable* vOutBuffer = queryVariable( "output_buffer" );

    Variable* vExposure = queryVariable( "exposure" );
    Variable* vGamma    = queryVariable( "gamma" );

    float exposure = 1.0f;
    if( vExposure )
    {
        if( vExposure->getType().baseType() == VariableType::Float )
            exposure = vExposure->get<float>();
        else
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                         "Failed to launch TonemapperSimple post-processing stage. Variable "
                                         "\"exposure\" has the wrong type. "
                                         "Should be float." );
    }

    float gamma = 1.0f;
    if( vGamma )
    {
        if( vGamma->getType().baseType() == VariableType::Float )
            gamma = vGamma->get<float>();
        else
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                         "Failed to launch TonemapperSimple post-processing stage. Variable \"gamma\" "
                                         "has the wrong type. "
                                         "Should be float." );
    }

    Buffer* inBuffer  = vInBuffer->isBuffer() ? vInBuffer->getBuffer() : nullptr;
    Buffer* outBuffer = vOutBuffer->isBuffer() ? vOutBuffer->getBuffer() : nullptr;

    LWDADevice* lwdaDevice = getContext()->getDeviceManager()->primaryLWDADevice();
    if( !lwdaDevice )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                     "Failed to launch TonemapperSimple post-processing stage. Failed to get primary "
                                     "LWCA device." );

    const lwca::ComputeCapability sver = lwdaDevice->computeCapability();
    RT_ASSERT( sver != lwca::SM_NONE() );

    MBufferHandle inputMBuffer      = inBuffer->getMBuffer();
    MAccess       inputBufferAccess = inputMBuffer->getAccess( lwdaDevice->allDeviceListIndex() );
    if( inputBufferAccess.getKind() != MAccess::LINEAR )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Failed to launch TonemapperSimple post-processing stage. Failed to get linear "
                                     "device buffer for input buffer." );
    const float* inDevPtr = reinterpret_cast<const float*>( inputBufferAccess.getLinearPtr() );

    MBufferHandle outputMBuffer      = outBuffer->getMBuffer();
    MAccess       outputBufferAccess = outputMBuffer->getAccess( lwdaDevice->allDeviceListIndex() );
    if( outputBufferAccess.getKind() != MAccess::LINEAR )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Failed to launch TonemapperSimple post-processing stage. Failed to get linear "
                                     "device buffer for output buffer." );
    float* outDevPtr = reinterpret_cast<float*>( outputBufferAccess.getLinearPtr() );

    lwdaDevice->makeLwrrent();

    // Check if the input/output buffer is allocated on slow memory in which case it will have to be
    // copied to the GPU before passed to the tonemapper and then back. This is more efficient
    // than operating on a zero copy buffer which is located in host memory.
    float* gpuOutDevPtr = static_cast<float*>( copyToGpuIfNeeded( outputMBuffer, outDevPtr, lwdaDevice ) );
    float* gpuInDevPtr = static_cast<float*>( copyToGpuIfNeeded( inputMBuffer, const_cast<float*>( inDevPtr ), lwdaDevice ) );

    // TODO: num_components is lwrrently hardcoded for both inputs and outputs since everything but float4
    // buffers is practically unusable anyway. will have to be more flexible in the future.
    // Additionally, this kind of setup should be generalized for other "shader-like" stages
    // to avoid code duplication.
    optix_lwda_postprocessing_tonemap( sver, gpuInDevPtr, gpuOutDevPtr, 4, exposure, gamma, width, height );

    // If a temp gpu pointer was used then copy back to the application provided buffer and free the temp one.
    if( outDevPtr != gpuOutDevPtr )
    {
        lwdaError_t error = lwdaMemcpyAsync( outDevPtr, gpuOutDevPtr, outputMBuffer->getDimensions().getTotalSizeInBytes(),
                                             lwdaMemcpyDeviceToDevice );
        if( error != lwdaSuccess )
            throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                         "Failed to launch TonemapperSimple post-processing stage. Failed to copy "
                                         "device buffer back to zero copy memory." );
        lwdaFree( gpuOutDevPtr );
    }

    if( inDevPtr != gpuInDevPtr )
        lwdaFree( gpuInDevPtr );

    // TODO temporary workaround: synchronize manually here before unpinning. There is probably a better way to do this.
    LWresult result = lwdaDriver().LwCtxSynchronize();
    if( result != LWDA_SUCCESS )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO, "tonemap postprocessing kernel failed!" );
}
}
