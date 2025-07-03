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

#include <Objects/PostprocessingStageSSIMPredictor.h>

#include <PostProcessor/SSIMPredictor/i_ssim.h>

#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Memory/MemoryManager.h>
#include <Objects/CommandList.h>
#include <Objects/Variable.h>
#include <Util/LinkedPtrHelpers.h>
#include <corelib/system/Timer.h>

#include <corelib/misc/String.h>
#include <prodlib/system/Knobs.h>

#include <prodlib/exceptions/IlwalidValue.h>

#include <cstdio>

using namespace corelib;

// clang-format off
namespace {
  Knob<int> k_dlSSIMPredictorLogLevel(RT_DSTRING("dlssimpredictor.logLevel"), -1, RT_DSTRING("Log level for the Deep Learning SSIM predictor. Defaults to -1 which will use the the optix log level"));
  Knob<bool> k_dlSSIMPredictorEnableSplit(RT_DSTRING("dlssimpredictor.enableSplit"), false, RT_DSTRING("Enables the split variable for the Deep Learning SSIM predictor."));
}
// clang-format on

namespace optix {


class SSIMPredictorLogger : public LW::SSIM::ILogger
{
  public:
    SSIMPredictorLogger( Context* context )
        : m_context( context )
    {
        // Get dlssimpredictor.logLevel knob value and translate it to a ssim_predictor internal log level.
        int level = k_dlSSIMPredictorLogLevel.get() < 0 ? prodlib::log::level() : k_dlSSIMPredictorLogLevel.get();

        switch( level )
        {
            case 0:
            case 1:
                m_loggingDisabled = true;
                break;
            case 2:
                m_logLevel = LW::SSIM::ILogger::S_ERROR;
                break;
            case 3:
                m_logLevel = LW::SSIM::ILogger::S_WARNING;
                break;
            case 4:
                m_logLevel = LW::SSIM::ILogger::S_INFO;
                break;
            default:
                if( level < 40 )
                    m_logLevel = LW::SSIM::ILogger::S_INFO;
                else if( level < 50 )
                    m_logLevel = LW::SSIM::ILogger::S_PROGRESS;
                else
                    m_logLevel = LW::SSIM::ILogger::S_DEBUG;
        }
    }

    void log( Severity severity, const char* msg ) override
    {
        if( m_loggingDisabled || ( severity > m_logLevel ) )
            return;

        UsageReport& ur = m_context->getUsageReport();

        switch( severity )
        {
            case LW::SSIM::ILogger::S_ERROR:
                if( ur.isActive( 0 ) )
                {
                    ureport0( ur, "DLSSIMPREDICTOR" ) << msg << std::endl;
                }
                break;
            case LW::SSIM::ILogger::S_WARNING:
                if( ur.isActive( 1 ) )
                {
                    ureport0( ur, "DLSSIMPREDICTOR" ) << msg << std::endl;
                }
                break;
            case LW::SSIM::ILogger::S_INFO:
            case LW::SSIM::ILogger::S_PROGRESS:
                if( ur.isActive( 2 ) )
                {
                    ureport0( ur, "DLSSIMPREDICTOR" ) << msg << std::endl;
                }
                break;
            default:
                llog( 50 ) << msg << "\n";
        }
    }

    // The log level
    bool                        m_loggingDisabled = false;
    LW::SSIM::ILogger::Severity m_logLevel        = LW::SSIM::ILogger::S_ERROR;
    Context*                    m_context;
};


std::unique_ptr<corelib::ExelwtableModule> PostprocessingStageSSIMPredictor::m_ssimPredictorLibrary;

PostprocessingStageSSIMPredictor::PostprocessingStageSSIMPredictor( Context* context, void* ssim_predictor )
    : PostprocessingStage( context, "DLSSIMPredictor" )
    , m_ssimPredictor( static_cast<LW::SSIM::ISsim*>( ssim_predictor ) )
{
}

PostprocessingStageSSIMPredictor::~PostprocessingStageSSIMPredictor()
{
    deleteVariables();

    if( m_ssimPredictor )
        m_ssimPredictor->release();

    m_ssimPredictorLogger.reset( nullptr );
}

void PostprocessingStageSSIMPredictor::initialize( RTsize width, RTsize height )
{
    if( !m_ssimPredictor )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                     "Failed to launch DLDenoiser post-processing stage. Failed to create Denoiser "
                                     "instance." );
    m_ssimPredictorLogger.reset( new SSIMPredictorLogger( m_context ) );
    m_ssimPredictor->set_logger( m_ssimPredictorLogger.get() );

    m_trainingDataDirty = true;
}

void PostprocessingStageSSIMPredictor::validate() const
{
    PostprocessingStage::validate();

    Variable* vInBuffer  = queryVariable( "input_buffer" );
    Variable* vOutBuffer = queryVariable( "output_buffer" );

    // Variable that contains the optional training data to use. If not specified the default
    // built-in training data will be used.
    Variable* vTrainingData = queryVariable( "training_data_buffer" );

    if( vTrainingData && !vTrainingData->isBuffer() && vTrainingData->getType() != VariableType() )
    {
        // Variable is set (type not unknown) but not to a buffer. This is not allowed.
        throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                     "DLSSIMPredictor training_data_buffer variable set to non-buffer type." );
    }

    Buffer* inBuffer  = ( vInBuffer && vInBuffer->isBuffer() ) ? vInBuffer->getBuffer() : nullptr;
    Buffer* outBuffer = ( vOutBuffer && vOutBuffer->isBuffer() ) ? vOutBuffer->getBuffer() : nullptr;

    if( !inBuffer || !outBuffer )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Error in DLSSIMPredictor post-processing stage. input_buffer and output_buffer "
                                     "must be declared and must be of type buffer." );

    // TODO: support more buffer combinations
    if( !( ( inBuffer->getFormat() == RT_FORMAT_FLOAT4 ) && ( outBuffer->getFormat() == RT_FORMAT_FLOAT ) ) )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Error in DLSSIMPredictor post-processing stage. input_buffer and output_buffer "
                                     "have an unsupported format combination." );

    // output needs to be large enough. (padded to ceil of input width / heatmap_shrink_factor)
    int requiredOutWidth = 0, requiredOutHeight = 0;
    m_ssimPredictor->get_heatmap_output_size( inBuffer->getWidth(), inBuffer->getHeight(), requiredOutWidth, requiredOutHeight );

    if( ( outBuffer->getWidth() < (uint)requiredOutWidth ) || ( outBuffer->getHeight() < (uint)requiredOutHeight ) )
    {
        char errMsg[256];
        snprintf( errMsg, sizeof( errMsg ),
                  "Error in DLSSIMPredictor post-processing stage."
                  " output_buffer must be at least %dx%d. (Got %dx%d).",
                  requiredOutWidth, requiredOutHeight, (int)outBuffer->getWidth(), (int)outBuffer->getHeight() );
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, errMsg );
    }
}

void PostprocessingStageSSIMPredictor::doLaunch( RTsize width, RTsize height )
{
    Variable* vInBuffer  = queryVariable( "input_buffer" );
    Variable* vOutBuffer = queryVariable( "output_buffer" );

    Buffer* inBuffer  = ( vInBuffer && vInBuffer->isBuffer() ) ? vInBuffer->getBuffer() : nullptr;
    Buffer* outBuffer = ( vOutBuffer && vOutBuffer->isBuffer() ) ? vOutBuffer->getBuffer() : nullptr;

    if( !inBuffer || !outBuffer )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Error in DLSSIMPredictor post-processing stage. input_buffer and output_buffer "
                                     "must be declared and must be of type buffer." );

    LWDADevice* lwdaDevice = getContext()->getDeviceManager()->primaryLWDADevice();
    if( !lwdaDevice )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO,
                                     "Failed to launch DLSSIMPredictor post-processing stage. Failed to get primary "
                                     "LWCA device." );

    // Get device data pointers
    MBufferHandle inputMBuffer      = inBuffer->getMBuffer();
    MAccess       inputBufferAccess = inputMBuffer->getAccess( lwdaDevice->allDeviceListIndex() );
    if( inputBufferAccess.getKind() != MAccess::LINEAR )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Failed to launch DLSSIMPredictor post-processing stage. Failed to get linear "
                                     "device buffer for input buffer." );
    const float* inData = reinterpret_cast<const float*>( inputBufferAccess.getLinearPtr() );

    MBufferHandle outputMBuffer      = outBuffer->getMBuffer();
    MAccess       outputBufferAccess = outputMBuffer->getAccess( lwdaDevice->allDeviceListIndex() );
    if( inputBufferAccess.getKind() != MAccess::LINEAR )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Failed to launch DLSSIMPredictor post-processing stage. Failed to get linear "
                                     "device buffer for output buffer." );
    float* outData = reinterpret_cast<float*>( outputBufferAccess.getLinearPtr() );

    int device_id = lwdaDevice->lwdaOrdinal();

    const int numInputChannels = 4;

    // A float (integral part taken) that defines the maximum GPU memory in bytes used by the ssim predictor.
    // If ssim prediction takes more memory, the image is split into tiles to stay below the specified maximum
    // amount of memory.

    if( Variable* v = queryVariable( "maxmem" ) )
    {
        float maxMemBytes = 0.f;

        // query maxmem (which should be in bytes)
        switch( v->getType().baseType() )
        {
            case VariableType::Float:
                maxMemBytes = float( v->get<float>() );
                break;
            case VariableType::LongLong:
                maxMemBytes = float( v->get<long long>() );
                break;
            case VariableType::ULongLong:
                maxMemBytes = float( v->get<unsigned long long>() );
                break;
            default:
                throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                             "Failed to launch DLDenoiser post-processing stage. Variable \"maxmem\" "
                                             "has the wrong type. Should be float or long long." );
        }

        // colwert to MiB, and pass to ssim->set_parameter
        const float mebiBytesPerByte = 1.f / float( 1 << 20 );
        m_ssimPredictor->set_parameter( "maxmem", maxMemBytes * mebiBytesPerByte );
    }

    LW::SSIM::Image_buffer inputBuffer = LW::SSIM::Image_buffer( (void*)inData, width, height, numInputChannels, device_id,
                                                                 LW::SSIM::DATA_FLOAT, true );  //, 4, device_id, LW::SSIM::INPUT_RGB)


    LW::SSIM::Data_type outDataFormat =
        ( outBuffer->getFormat() == RT_FORMAT_UNSIGNED_BYTE4 ) ? LW::SSIM::DATA_INT8 : LW::SSIM::DATA_FLOAT;

    const int numOutputChannels = 1;

    int heatmap_width = 0, heatmap_height = 0;
    m_ssimPredictor->get_heatmap_output_size( width, height, heatmap_width, heatmap_height );

    if( heatmap_width < 1 || heatmap_height < 1 )
    {
        char errMsg[256];
        snprintf( errMsg, sizeof( errMsg ),
                  "Failed to launch DLSSIMPredictor post-processing stage."
                  " Output buffer size (%dx%d) is invalid.",
                  heatmap_width, heatmap_height );
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, errMsg );
    }

    // Output buffer should be 1/heatmap_shrink_factor scale of the input buffer
    LW::SSIM::Image_buffer ssimPredictorOutputBuffer( (void*)outData, heatmap_width, heatmap_height, numOutputChannels,
                                                      device_id, outDataFormat, true );


    // Run ssim predictor on the primary LWCA device
    // int error = (int)((long)&inputBuffer & 0);
    int error = m_ssimPredictor->run( &device_id, 1, &inputBuffer, &ssimPredictorOutputBuffer, 0 );
    //error = 0;
    if( error < 0 )
    {
        char errMsg[256];
        snprintf( errMsg, sizeof( errMsg ),
                  "Failed to launch DLSSIMPredictor post-processing stage."
                  " DLSSIMPredictor run method failed with error %d.",
                  error );
        throw prodlib::UnknownError( RT_EXCEPTION_INFO, errMsg );
    }

    if( m_context )
        m_context->incrSSIMPredictorLaunchCount();
}

void PostprocessingStageSSIMPredictor::detachLinkedChild( const LinkedPtr_Link* link )
{
    if( ( &m_trainingDataBufferLink ) == link )
        m_trainingDataBufferLink.reset();
}

void PostprocessingStageSSIMPredictor::bufferWasMapped( LinkedPtr_Link* link )
{
    if( ( &m_trainingDataBufferLink ) == link )
        m_trainingDataDirty = true;
}

void PostprocessingStageSSIMPredictor::bufferFormatDidChange( LinkedPtr_Link* link )
{
    if( ( &m_trainingDataBufferLink ) == link )
        m_trainingDataDirty = true;
}

void PostprocessingStageSSIMPredictor::bufferVariableValueDidChange( Variable* var, Buffer* oldBuffer, Buffer* newBuffer )
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
