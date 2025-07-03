/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
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
#include <prodlib/misc/lwpti/LwptiProfiler.h>

#include <prodlib/misc/lwpti/Lwpti.h>
#include <prodlib/misc/lwpti/LwPerfHost.h>
#include <prodlib/misc/lwpti/LwPerfTarget.h>
#include <prodlib/system/Logger.h>

#include <corelib/misc/String.h>
#include <corelib/system/LwdaDriver.h>
#include <corelib/system/ExelwtableModule.h>
#include <corelib/system/SystemError.h>

#include <lwpti_profiler_target.h>
#include <lwpti_target.h>
#include <lwperf_lwda_host.h>
#include <lwperf_host.h>
#include <lwperf_target.h>

#include <iomanip>
#include <iostream>
#include <limits>

static const unsigned int MAX_LWPTI_BUFFER_SIZE = 512 * 1024 * 1024;  // Buffer up to 512MB of profiling data at a time.

namespace prodlib {

bool LwptiMetricsJson::open( const std::string& filePath )
{
    m_outputFile.open( filePath );

    if( m_outputFile.fail() )
        return false;

    // Write document start
    m_buffer << "{\n\t\"launches\": [";

    m_isOpen = true;
    return true;
}

void LwptiMetricsJson::addLaunch()
{
    if( m_firstLaunch )
    {
        m_buffer << "\n\t\t{";
        m_firstLaunch = false;
    }
    else
        m_buffer << ",\n\t\t{";
}

void LwptiMetricsJson::addMetricToLaunch( const std::string& metricName, double metricValue )
{
    // Make sure we don't print the metric value in scientific notation.
    const int prevPrecision = m_buffer.precision();
    const int newPrecision  = std::numeric_limits<double>::max_digits10;
    m_buffer << std::setprecision( newPrecision );

    if( m_firstMetric )
    {
        m_buffer << "\n\t\t\t\"" << metricName << "\": " << metricValue;
        m_firstMetric = false;
    }
    else
    {
        m_buffer << ",\n\t\t\t\"" << metricName << "\": " << metricValue;
    }

    m_buffer << std::setprecision( prevPrecision );

    flushIfFullBuffer();
}

void LwptiMetricsJson::finishLaunch()
{
    m_firstMetric = true;
    m_buffer << "\n\t\t}";
    flushIfFullBuffer();
}

void LwptiMetricsJson::endDolwment()
{
    m_firstLaunch = true;
    m_buffer << "\n\t]\n}";
    flush();
    m_outputFile.close();
}

bool LwptiMetricsJson::isOpen() const
{
    return m_isOpen;
}

void LwptiMetricsJson::flushIfFullBuffer()
{
    if( m_buffer.str().size() > MAX_LWPTI_BUFFER_SIZE )
        flush();
}

void LwptiMetricsJson::flush()
{
    m_outputFile << m_buffer.str();
    m_outputFile.flush();
    m_buffer.str( "" );
}

inline std::string trimSpaces( const std::string& s )
{
    unsigned int start = s.find_first_not_of( ' ' );
    unsigned int end   = s.find_last_not_of( ' ' );
    return s.substr( start, ( end + 1 ) - start );
}

std::vector<std::string> splitByComma( const std::string& listString )
{
    std::vector<std::string> results;

    std::string lwrrMetric;
    for( size_t i = 0; i < listString.size(); ++i )
    {
        if( listString[i] == ',' )
        {
            if( !lwrrMetric.empty() )
            {
                results.push_back( trimSpaces( lwrrMetric ) );
                lwrrMetric = "";
            }
            continue;
        }

        lwrrMetric += listString[i];
    }
    if( !lwrrMetric.empty() )
        results.push_back( trimSpaces( lwrrMetric ) );

    return results;
}

#define CHECK_LWPTI_CALL( call, msg )                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        const LWptiResult res = call;                                                                                  \
        if( res != LWPTI_SUCCESS )                                                                                     \
        {                                                                                                              \
            return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,                                                  \
                                          std::string( msg ) + std::string( " . Error: " ) + std::to_string( res ) );  \
        }                                                                                                              \
    } while( 0 )

#define CHECK_LWPERF_CALL( call, msg )                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        const LWPA_Status res = call;                                                                                  \
        if( res != LWPA_STATUS_SUCCESS )                                                                               \
        {                                                                                                              \
            return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,                                                  \
                                          std::string( msg ) + std::string( " . Error: " ) + std::to_string( res ) );  \
        }                                                                                                              \
    } while( 0 )

OptixResult LwptiProfiler::initialize( LWcontext lwdaContext, const std::string& metricsToCollect, optix_exp::ErrorDetails& errDetails )
{
    if( const OptixResult res = m_lwpti.initialize( errDetails ) )
        return res;

    if( const OptixResult res = m_lwPerfHost.initialize( errDetails ) )
        return res;

    if( const OptixResult res = m_lwPerfTarget.initialize( errDetails ) )
        return res;

    // Parse list of metrics
    m_metricNames = splitByComma( metricsToCollect );

    LWpti_Profiler_Initialize_Params profilerInitializeParams{LWpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CHECK_LWPTI_CALL( m_lwpti.profilerInitialize( &profilerInitializeParams ), "Failed to initialize LWPTI." );

    // Get the device ordinal associated with the given LWCA context.
    LWdevice deviceOrdinal;
    corelib::lwdaDriver().LwCtxGetDevice( &deviceOrdinal );

    // Retrieve chip name from device associated with this context.
    LWpti_Device_GetChipName_Params getChipNameParams{LWpti_Device_GetChipName_Params_STRUCT_SIZE};
    getChipNameParams.deviceIndex = static_cast<int>( deviceOrdinal );
    CHECK_LWPTI_CALL( m_lwpti.deviceGetChipName( &getChipNameParams ), "Failed to retrieve LWPTI chip name." );
    m_chipName = std::string( getChipNameParams.pChipName );

    // Query available performance counters from LWPTI.
    LWpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams{LWpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    getCounterAvailabilityParams.ctx = lwdaContext;
    CHECK_LWPTI_CALL( m_lwpti.profilerGetCounterAvailability( &getCounterAvailabilityParams ),
                      "Failed to query LWPTI counter availability." );

    // Set up buffer for performance counters.
    m_counterAvailabilityImage.clear();
    m_counterAvailabilityImage.resize( getCounterAvailabilityParams.counterAvailabilityImageSize );
    getCounterAvailabilityParams.pCounterAvailabilityImage = &m_counterAvailabilityImage[0];

    // Populate available perf counters.
    CHECK_LWPTI_CALL( m_lwpti.profilerGetCounterAvailability( &getCounterAvailabilityParams ),
                      "Failed to query LWPTI counter availability." );

    // Generate a configuration image.
    LWPW_InitializeHost_Params initializeHostParams{LWPW_InitializeHost_Params_STRUCT_SIZE};
    CHECK_LWPERF_CALL( m_lwPerfHost.initializeHost( &initializeHostParams ),
                       "Failed to initialize LWPTI host params." );

    if( const OptixResult result = createConfigImage( errDetails ) )
        return result;

    if( const OptixResult result = getCounterDataPrefixImage( errDetails ) )
        return result;

    if( const OptixResult result = createCounterDataImage( errDetails ) )
        return result;

    m_isInitialized = true;
    return OPTIX_SUCCESS;
}

OptixResult LwptiProfiler::deinitialize( optix_exp::ErrorDetails& errDetails )
{
    LWpti_Profiler_DeInitialize_Params profilerDeInitializeParams{LWpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CHECK_LWPTI_CALL( m_lwpti.profilerDeInitialize( &profilerDeInitializeParams ), "Failed to de-initialize LWPTI." );

    m_isInitialized = false;

    // Close the output file, since we won't be writing to it anymore.
    if( m_outputFile.isOpen() )
        closeOutputFile();

    return OPTIX_SUCCESS;
}

/*
 * Handles the allocation and cleanup of an LwPerfWorks metrics context.
 */
class MetricsContext
{
  public:
    MetricsContext( LwPerfHost* lwPerfHost )
        : m_lwPerfHost( lwPerfHost )
    {
    }

    ~MetricsContext()
    {
        if( m_metricsContext )
        {
            LWPW_MetricsContext_Destroy_Params metricsContextDestroyParams{LWPW_MetricsContext_Destroy_Params_STRUCT_SIZE};
            metricsContextDestroyParams.pMetricsContext = m_metricsContext;
            m_lwPerfHost->metricsContext_Destroy( &metricsContextDestroyParams );
        }
    }

    OptixResult initialize( const char* chipName, optix_exp::ErrorDetails& errDetails )
    {
        LWPW_LWDA_MetricsContext_Create_Params metricsContextCreateParams{LWPW_LWDA_MetricsContext_Create_Params_STRUCT_SIZE};
        metricsContextCreateParams.pChipName = chipName;
        CHECK_LWPERF_CALL( m_lwPerfHost->LWDA_MetricsContext_Create( &metricsContextCreateParams ),
                           "Failed to create LWPTI metrics context." );

        m_metricsContext = metricsContextCreateParams.pMetricsContext;

        return OPTIX_SUCCESS;
    }

    struct LWPA_MetricsContext* get() { return m_metricsContext; }

  private:
    LwPerfHost*                 m_lwPerfHost;
    struct LWPA_MetricsContext* m_metricsContext = nullptr;
};

/*
 * Handles the allocation and cleanup of an LwPerfWorks raw metrics config.
 */
class RawMetricsConfig
{
  public:
    RawMetricsConfig( LwPerfHost* lwPerfHost )
        : m_lwPerfHost( lwPerfHost )
    {
    }

    ~RawMetricsConfig()
    {
        if( m_rawMetricsConfig )
        {
            LWPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams{LWPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE};
            rawMetricsConfigDestroyParams.pRawMetricsConfig = m_rawMetricsConfig;
            m_lwPerfHost->rawMetricsConfig_Destroy( &rawMetricsConfigDestroyParams );
        }
    }

    OptixResult initialize( const char* chipName, optix_exp::ErrorDetails& errDetails )
    {
        LWPA_RawMetricsConfigOptions metricsConfigOptions{LWPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE};
        metricsConfigOptions.activityKind = LWPA_ACTIVITY_KIND_PROFILER;
        metricsConfigOptions.pChipName    = chipName;
        CHECK_LWPERF_CALL( m_lwPerfHost->rawMetricsConfig_Create( &metricsConfigOptions, &m_rawMetricsConfig ),
                           "Failed to create LWPTI raw metrics config." );

        return OPTIX_SUCCESS;
    }

    struct LWPA_RawMetricsConfig* get() { return m_rawMetricsConfig; }

  private:
    LwPerfHost*                   m_lwPerfHost;
    struct LWPA_RawMetricsConfig* m_rawMetricsConfig = nullptr;
};

OptixResult LwptiProfiler::createConfigImage( optix_exp::ErrorDetails& errDetails )
{
    MetricsContext metricsContext( &m_lwPerfHost );
    if( const OptixResult result = metricsContext.initialize( m_chipName.c_str(), errDetails ) )
        return result;

    // Get raw metric requests from metric names.
    if( const OptixResult result = getRawMetricRequests( metricsContext.get(), errDetails ) )
        return result;

    RawMetricsConfig rawMetricsConfig( &m_lwPerfHost );
    if( const OptixResult result = rawMetricsConfig.initialize( m_chipName.c_str(), errDetails ) )
        return result;

    LWPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams{
        LWPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
    setCounterAvailabilityParams.pRawMetricsConfig         = rawMetricsConfig.get();
    setCounterAvailabilityParams.pCounterAvailabilityImage = &m_counterAvailabilityImage[0];
    CHECK_LWPERF_CALL( m_lwPerfHost.rawMetricsConfig_SetCounterAvailability( &setCounterAvailabilityParams ),
                       "Failed to set LWPTI raw metrics config counter availability." );

    // Add raw metric requests to a pass group.
    LWPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams{LWPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE};
    beginPassGroupParams.pRawMetricsConfig = rawMetricsConfig.get();
    CHECK_LWPERF_CALL( m_lwPerfHost.rawMetricsConfig_BeginPassGroup( &beginPassGroupParams ),
                       "Failed to begin LWPTI pass group." );

    LWPW_RawMetricsConfig_AddMetrics_Params addMetricsParams{LWPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE};
    addMetricsParams.pRawMetricsConfig  = rawMetricsConfig.get();
    addMetricsParams.pRawMetricRequests = &m_rawMetricRequests[0];
    addMetricsParams.numMetricRequests  = m_rawMetricRequests.size();
    CHECK_LWPERF_CALL( m_lwPerfHost.rawMetricsConfig_AddMetrics( &addMetricsParams ),
                       "Failed to add metrics to LWPTI pass group." );

    LWPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams{LWPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE};
    endPassGroupParams.pRawMetricsConfig = rawMetricsConfig.get();
    CHECK_LWPERF_CALL( m_lwPerfHost.rawMetricsConfig_EndPassGroup( &endPassGroupParams ),
                       "Failed to end LWPTI pass group." );

    LWPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams{LWPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE};
    generateConfigImageParams.pRawMetricsConfig = rawMetricsConfig.get();
    CHECK_LWPERF_CALL( m_lwPerfHost.rawMetricsConfig_GenerateConfigImage( &generateConfigImageParams ),
                       "Failed to generate LWPTI raw metrics config image." );

    // Copy config image out.

    LWPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams{LWPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE};
    getConfigImageParams.pRawMetricsConfig = rawMetricsConfig.get();
    getConfigImageParams.bytesAllocated    = 0;
    getConfigImageParams.pBuffer           = nullptr;
    CHECK_LWPERF_CALL( m_lwPerfHost.rawMetricsConfig_GetConfigImage( &getConfigImageParams ),
                       "Failed to get LWPTI raw metrics config image info." );

    m_configImage.resize( getConfigImageParams.bytesCopied );

    getConfigImageParams.bytesAllocated = m_configImage.size();
    getConfigImageParams.pBuffer        = &m_configImage[0];
    CHECK_LWPERF_CALL( m_lwPerfHost.rawMetricsConfig_GetConfigImage( &getConfigImageParams ),
                       "Failed to get LWPTI raw metrics config image." );

    return OPTIX_SUCCESS;
}

/*
 * Handles the allocation and cleanup of an LwPerfWorks counter data builder.
 */
class CounterDataBuilder
{
  public:
    CounterDataBuilder( LwPerfHost* lwPerfHost )
        : m_lwPerfHost( lwPerfHost )
    {
    }

    ~CounterDataBuilder()
    {
        if( m_counterDataBuilder )
        {
            LWPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams{LWPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE};
            counterDataBuilderDestroyParams.pCounterDataBuilder = m_counterDataBuilder;
            m_lwPerfHost->counterDataBuilder_Destroy( &counterDataBuilderDestroyParams );
        }
    }

    OptixResult initialize( const char* chipName, optix_exp::ErrorDetails& errDetails )
    {
        LWPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams{LWPW_CounterDataBuilder_Create_Params_STRUCT_SIZE};
        counterDataBuilderCreateParams.pChipName = chipName;
        CHECK_LWPERF_CALL( m_lwPerfHost->counterDataBuilder_Create( &counterDataBuilderCreateParams ),
                           "Failed to create LWPTI counter builder." );

        m_counterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;

        return OPTIX_SUCCESS;
    }

    struct LWPA_CounterDataBuilder* get() { return m_counterDataBuilder; }

  private:
    LwPerfHost*              m_lwPerfHost;
    LWPA_CounterDataBuilder* m_counterDataBuilder = nullptr;
};

OptixResult LwptiProfiler::getCounterDataPrefixImage( optix_exp::ErrorDetails& errDetails )
{
    MetricsContext metricsContext( &m_lwPerfHost );
    if( const OptixResult result = metricsContext.initialize( m_chipName.c_str(), errDetails ) )
        return result;

    if( const OptixResult result = getRawMetricRequests( metricsContext.get(), errDetails ) )
        return result;

    CounterDataBuilder counterDataBuilder( &m_lwPerfHost );
    if( const OptixResult result = counterDataBuilder.initialize( m_chipName.c_str(), errDetails ) )
        return result;

    LWPW_CounterDataBuilder_AddMetrics_Params addMetricsParams{LWPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE};
    addMetricsParams.pCounterDataBuilder = counterDataBuilder.get();
    addMetricsParams.pRawMetricRequests  = &m_rawMetricRequests[0];
    addMetricsParams.numMetricRequests   = m_rawMetricRequests.size();
    CHECK_LWPERF_CALL( m_lwPerfHost.counterDataBuilder_AddMetrics( &addMetricsParams ),
                       "Failed to add metrics to LWPTI counter builder." );

    LWPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams{
        LWPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE};
    getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilder.get();
    getCounterDataPrefixParams.bytesAllocated      = 0;
    getCounterDataPrefixParams.pBuffer             = nullptr;
    CHECK_LWPERF_CALL( m_lwPerfHost.counterDataBuilder_GetCounterDataPrefix( &getCounterDataPrefixParams ),
                       "Failed to get LWPTI counter data prefix." );

    m_counterDataImagePrefix.resize( getCounterDataPrefixParams.bytesCopied );

    getCounterDataPrefixParams.bytesAllocated = m_counterDataImagePrefix.size();
    getCounterDataPrefixParams.pBuffer        = &m_counterDataImagePrefix[0];
    CHECK_LWPERF_CALL( m_lwPerfHost.counterDataBuilder_GetCounterDataPrefix( &getCounterDataPrefixParams ),
                       "Failed to get LWPTI counter data prefix." );

    return OPTIX_SUCCESS;
}

OptixResult LwptiProfiler::createCounterDataImage( optix_exp::ErrorDetails& errDetails )
{
    LWpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix    = &m_counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = m_counterDataImagePrefix.size();
    counterDataImageOptions.maxNumRanges          = m_metricNames.size();
    counterDataImageOptions.maxNumRangeTreeNodes  = m_metricNames.size();
    counterDataImageOptions.maxRangeNameLength    = 64;

    LWpti_Profiler_CounterDataImage_CallwlateSize_Params callwlateSizeParams{LWpti_Profiler_CounterDataImage_CallwlateSize_Params_STRUCT_SIZE};

    callwlateSizeParams.pOptions                      = &counterDataImageOptions;
    callwlateSizeParams.sizeofCounterDataImageOptions = LWpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

    CHECK_LWPTI_CALL( m_lwpti.profilerCounterDataImageCallwlateSize( &callwlateSizeParams ),
                      "Failed to callwlate LWPTI counter data image size." );

    LWpti_Profiler_CounterDataImage_Initialize_Params initializeParams{LWpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.sizeofCounterDataImageOptions = LWpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions                      = &counterDataImageOptions;
    initializeParams.counterDataImageSize          = callwlateSizeParams.counterDataImageSize;

    m_counterDataImage.resize( callwlateSizeParams.counterDataImageSize );
    initializeParams.pCounterDataImage = &m_counterDataImage[0];
    CHECK_LWPTI_CALL( m_lwpti.profilerCounterDataImageInitialize( &initializeParams ),
                      "Failed to initialize LWPTI counter data image." );

    LWpti_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params scratchBufferSizeParams{
        LWpti_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params_STRUCT_SIZE};
    scratchBufferSizeParams.counterDataImageSize = callwlateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage    = initializeParams.pCounterDataImage;
    CHECK_LWPTI_CALL( m_lwpti.profilerCounterDataImageCallwlateScratchBufferSize( &scratchBufferSizeParams ),
                      "Failed to callwlate LWPTI counter data image scratch size." );

    m_counterDataScratchBuffer.resize( scratchBufferSizeParams.counterDataScratchBufferSize );

    LWpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams{
        LWpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    initScratchBufferParams.counterDataImageSize = callwlateSizeParams.counterDataImageSize;

    initScratchBufferParams.pCounterDataImage            = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer    = &m_counterDataScratchBuffer[0];

    CHECK_LWPTI_CALL( m_lwpti.profilerCounterDataImageInitializeScratchBuffer( &initScratchBufferParams ),
                      "Failed to initialize LWPTI counter data image scratch buffer." );

    return OPTIX_SUCCESS;
}

bool LwptiProfiler::openOutputFile( const std::string& filePath )
{
    return m_outputFile.open( filePath );
}

void LwptiProfiler::closeOutputFile()
{
    m_outputFile.endDolwment();
}

OptixResult LwptiProfiler::beginProfile( optix_exp::ErrorDetails& errDetails )
{
    LWpti_Profiler_BeginSession_Params    beginSessionParams{LWpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    LWpti_Profiler_SetConfig_Params       setConfigParams{LWpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    LWpti_Profiler_EnableProfiling_Params enableProfilingParams{LWpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    LWpti_Profiler_PushRange_Params       pushRangeParams{LWpti_Profiler_PushRange_Params_STRUCT_SIZE};

    beginSessionParams.ctx                          = nullptr;
    beginSessionParams.counterDataImageSize         = m_counterDataImage.size();
    beginSessionParams.pCounterDataImage            = &m_counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = m_counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer    = &m_counterDataScratchBuffer[0];
    beginSessionParams.range                        = LWPTI_UserRange;
    LWpti_ProfilerReplayMode profilerReplayMode     = LWPTI_UserReplay;
    beginSessionParams.replayMode                   = profilerReplayMode;
    beginSessionParams.maxRangesPerPass             = m_metricNames.size();
    beginSessionParams.maxLaunchesPerPass           = m_metricNames.size();

    CHECK_LWPTI_CALL( m_lwpti.profilerBeginSession( &beginSessionParams ), "Failed to begin LWPTI profiling session." );

    setConfigParams.pConfig    = &m_configImage[0];
    setConfigParams.configSize = m_configImage.size();

    setConfigParams.passIndex        = 0;
    setConfigParams.minNestingLevel  = 1;
    setConfigParams.numNestingLevels = 1;
    CHECK_LWPTI_CALL( m_lwpti.profilerSetConfig( &setConfigParams ), "Failed to set LWPTI config." );

    /* User takes the resposiblity of replaying the kernel launches */
    LWpti_Profiler_BeginPass_Params beginPassParams{LWpti_Profiler_BeginPass_Params_STRUCT_SIZE};

    CHECK_LWPTI_CALL( m_lwpti.profilerBeginPass( &beginPassParams ), "Failed to begin LWPTI profiling pass." );
    CHECK_LWPTI_CALL( m_lwpti.profilerEnableProfiling( &enableProfilingParams ), "Failed to enable LWPTI profiling." );

    // TODO: Put an actual name here.
    std::string rangeName      = "userrangeA";
    pushRangeParams.pRangeName = rangeName.c_str();
    CHECK_LWPTI_CALL( m_lwpti.profilerPushRange( &pushRangeParams ), "Failed push LWPTI range." );

    return OPTIX_SUCCESS;
}

OptixResult LwptiProfiler::endProfile( optix_exp::ErrorDetails& errDetails )
{
    LWpti_Profiler_PopRange_Params popRangeParams{LWpti_Profiler_PopRange_Params_STRUCT_SIZE};
    CHECK_LWPTI_CALL( m_lwpti.profilerPopRange( &popRangeParams ), "Failed to pop LWPTI range." );

    LWpti_Profiler_DisableProfiling_Params disableProfilingParams{LWpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    CHECK_LWPTI_CALL( m_lwpti.profilerDisableProfiling( &disableProfilingParams ),
                      "Failed to disable LWPTI profiling." );

    LWpti_Profiler_EndPass_Params endPassParams{LWpti_Profiler_EndPass_Params_STRUCT_SIZE};
    CHECK_LWPTI_CALL( m_lwpti.profilerEndPass( &endPassParams ), "Failed to end LWPTI profiling pass." );

    if( !endPassParams.allPassesSubmitted )
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, "Supplied LWPTI counters require multiple passes." );

    LWpti_Profiler_FlushCounterData_Params flushCounterDataParams{LWpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
    CHECK_LWPTI_CALL( m_lwpti.profilerFlushCounterData( &flushCounterDataParams ), "Failed to flush LWPTI counters." );

    LWpti_Profiler_UnsetConfig_Params unsetConfigParams{LWpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    CHECK_LWPTI_CALL( m_lwpti.profilerUnsetConfig( &unsetConfigParams ),
                      "Failed to unset LWPTI profiling config params." );

    LWpti_Profiler_EndSession_Params endSessionParams{LWpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CHECK_LWPTI_CALL( m_lwpti.profilerEndSession( &endSessionParams ), "Failed to end LWPTI profiling session." );

    if( const OptixResult res = addMetricValuesToJson( errDetails ) )
        return res;

    return OPTIX_SUCCESS;
}

OptixResult LwptiProfiler::addMetricValuesToJson( optix_exp::ErrorDetails& errDetails )
{
    if( !m_counterDataImage.size() )
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, "LWPTI counter data image is empty." );

    MetricsContext metricsContext( &m_lwPerfHost );
    if( const OptixResult result = metricsContext.initialize( m_chipName.c_str(), errDetails ) )
        return result;

    LWPW_LWDA_MetricsContext_Create_Params metricsContextCreateParams{LWPW_LWDA_MetricsContext_Create_Params_STRUCT_SIZE};
    metricsContextCreateParams.pChipName = m_chipName.c_str();
    CHECK_LWPERF_CALL( m_lwPerfHost.LWDA_MetricsContext_Create( &metricsContextCreateParams ),
                       "Failed to create LWPTI metrics context." );

    LWPW_MetricsContext_Destroy_Params metricsContextDestroyParams{LWPW_MetricsContext_Destroy_Params_STRUCT_SIZE};
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;

    LWPW_CounterData_GetNumRanges_Params getNumRangesParams{LWPW_CounterData_GetNumRanges_Params_STRUCT_SIZE};
    getNumRangesParams.pCounterDataImage = &m_counterDataImage[0];
    CHECK_LWPERF_CALL( m_lwPerfTarget.counterData_GetNumRanges( &getNumRangesParams ),
                       "Failed to get number of LWPTI ranges." );

    std::vector<const char*> metricNamePtrs;
    for( std::string& metricName : m_metricNames )
        metricNamePtrs.push_back( metricName.c_str() );

    for( size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex )
    {
        LWPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams{
            LWPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE};
        getRangeDescParams.pCounterDataImage = &m_counterDataImage[0];
        getRangeDescParams.rangeIndex        = rangeIndex;
        CHECK_LWPERF_CALL( m_lwPerfTarget.profiler_CounterData_GetRangeDescriptions( &getRangeDescParams ),
                           "Failed to get LWPTI range description." );

        std::vector<const char*> descriptionPtrs( getRangeDescParams.numDescriptions );

        getRangeDescParams.ppDescriptions = &descriptionPtrs[0];
        CHECK_LWPERF_CALL( m_lwPerfTarget.profiler_CounterData_GetRangeDescriptions( &getRangeDescParams ),
                           "Failed to LWPTI range description." );

        std::string rangeName;
        for( size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex )
        {
            if( descriptionIndex != 0 )
            {
                rangeName += "/";
            }
            rangeName += descriptionPtrs[descriptionIndex];
        }

        std::vector<double> gpuValues;
        gpuValues.resize( m_metricNames.size() );

        LWPW_MetricsContext_SetCounterData_Params setCounterDataParams{LWPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE};
        setCounterDataParams.pMetricsContext   = metricsContextCreateParams.pMetricsContext;
        setCounterDataParams.pCounterDataImage = &m_counterDataImage[0];
        setCounterDataParams.isolated          = true;
        setCounterDataParams.rangeIndex        = rangeIndex;
        m_lwPerfHost.metricsContext_SetCounterData( &setCounterDataParams );

        LWPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams{LWPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE};
        evalToGpuParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
        evalToGpuParams.numMetrics      = metricNamePtrs.size();
        evalToGpuParams.ppMetricNames   = &metricNamePtrs[0];
        evalToGpuParams.pMetricValues   = &gpuValues[0];
        m_lwPerfHost.metricsContext_EvaluateToGpuValues( &evalToGpuParams );

        m_outputFile.addLaunch();
        for( size_t metricIndex = 0; metricIndex < m_metricNames.size(); ++metricIndex )
            m_outputFile.addMetricToLaunch( m_metricNames[metricIndex], gpuValues[metricIndex] );
        m_outputFile.finishLaunch();
    }

    return OPTIX_SUCCESS;
}

// Get raw metric requests from given list of metric names.
OptixResult LwptiProfiler::getRawMetricRequests( LWPA_MetricsContext* pMetricsContext, optix_exp::ErrorDetails& errDetails )
{
    m_rawMetricRequests.resize( 0 );
    m_rawMetricNames.resize( 0 );

    // TODO: What do these parameters do? The documentation doesn't specify.
    const bool isolated      = true;
    const bool keepInstances = true;

    // TODO: Sample does some sanitizing and sometimes sets isolated and keepInstances to true in Parser.h. Why?
    for( const std::string& metricName : m_metricNames )
    {
        LWPW_MetricsContext_GetMetricProperties_Begin_Params beginParams{LWPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE};
        beginParams.pMetricsContext = pMetricsContext;
        beginParams.pMetricName     = metricName.c_str();

        CHECK_LWPERF_CALL( m_lwPerfHost.metricsContext_GetMetricProperties_Begin( &beginParams ),
                           "Failed to get LWPTI metric properties." );

        for( const char** ppMetricDependencies = beginParams.ppRawMetricDependencies; *ppMetricDependencies; ++ppMetricDependencies )
            m_rawMetricNames.push_back( *ppMetricDependencies );

        LWPW_MetricsContext_GetMetricProperties_End_Params endParams{LWPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE};
        endParams.pMetricsContext = pMetricsContext;
        CHECK_LWPERF_CALL( m_lwPerfHost.metricsContext_GetMetricProperties_End( &endParams ),
                           "Failed to get LWPTI metric properties." );
    }

    for( const std::string& rawMetricName : m_rawMetricNames )
    {
        LWPA_RawMetricRequest metricRequest{LWPA_RAW_METRIC_REQUEST_STRUCT_SIZE};
        metricRequest.pMetricName   = rawMetricName.c_str();
        metricRequest.isolated      = isolated;
        metricRequest.keepInstances = keepInstances;
        m_rawMetricRequests.push_back( metricRequest );
    }

    return OPTIX_SUCCESS;
}

}  // namespace prodlib
