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

#include <Util/Metrics.h>

#include <Util/JsonEscape.h>
#include <Util/SystemInfo.h>

#include <corelib/misc/String.h>
#include <corelib/system/Timer.h>

#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/system/Knobs.h>

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

#include <stdint.h>
#include <stdio.h>

using namespace corelib;
using namespace prodlib;

namespace {
// clang-format off
PublicKnob<bool> k_metricsEnable(RT_PUBLIC_DSTRING("metrics.enable"), false, RT_PUBLIC_DSTRING("Enable metrics output to metrics.json") );
PublicKnob<std::string> k_metricsRunJson( RT_PUBLIC_DSTRING( "metrics.runJson" ), "",    RT_PUBLIC_DSTRING( "Raw JSON string to inject into the run_info section" ) );
PublicKnob<std::string> k_metricsFile( RT_PUBLIC_DSTRING( "metrics.file" ), "metrics.json",    RT_PUBLIC_DSTRING( "Path to metrics file" ) );
// clang-format on
}

namespace optix {

const int DEFAULT_BUFFER_SIZE = 32 * 1024;

namespace {

struct MetricDescription
{
    const char* name;  // Name of metric. Use "<tag>" for embedded tags
    const char* description;
};

}  // namespace

// clang-format off
static MetricDescription g_metricDescriptions[] = {
  { "api_msec_to_first_trace",              "Time spent in API calls in milliseconds from start until the first trace kernel launch." },
  { "build_accels_count",                   "Number of accels built in a launch call."},
  { "build_accels_msec",                    "Milliseconds spent building accels in a launch call."},
  { "build_aabbs_msec",                     "Milliseconds spent building aabbs in a launch call."},
  { "build_Mprims_per_sec",                 "Millions of primitives per second to build all dirty accels."},
  { "compile_msec",                         "Inclusive time in milliseconds for compile."},
  { "compile_msec_to_first_trace",          "Inclusive time in milliseconds for compile until the first trace kernel launch."},
  { "compile_sm_arch",                      "SM architecture used for compile."},
  { "entry_index",                          "Launch entry point."},
  { "kernel_launch_index",                  "Kernel launch index."},
  { "kernel_msec",                          "Milliseconds spent in the kernel."},
  { "kernel_elems",                         "Size of the kernel launch."},
  { "kernel_type",                          "Kernel type."},
  { "launch_msec",                          "Inclusive time in milliseconds of an entire launch."},
  { "launch_index",                         "Index of the launch range to which an event belongs."},
  { "llvm_instructions",                    "Number of instructions in kernel LLVM-IR."},
  { "llvm_basic_blocks",                    "Number of basic blocks in kernel LLVM-IR."},
  { "llvm_functions",                       "Number of functions in kernel LLVM-IR."},
  { "mem_used_dev<N>",                      "GPU memory used on device N."},
  { "mk_compile_msec",                      "Inclusive time in milliseconds for MegakernelCompile::compileToPtx()."},
  { "msec_to_launch_range<N>",              "Time in milliseconds from context creation until luanch index range N."},
  { "msec_to_first_launch",                 "Time in milliseconds from context creation until the first call to rtContextLaunch()."},
  { "msec_to_first_kernel",                 "Time in milliseconds from context creation until the first kernel launch (could be AABB)."},
  { "msec_to_first_trace",                  "Time in milliseconds from context creation until the first trace kernel launch."},
  { "msec_to_first_frame",                  "Time in milliseconds from context creation until the first context launch is over."},
  { "num_compiles",                         "Total number of compiles for the context."},
  { "ocg_msec",                             "Time in milliseconds to compile PTX to SASS."},
  { "ptx_msec",                             "Time in milliseconds to compile LLVM-IR to PTX."},
  { "scope<N>",                             "Unique scope name."},
  { "ptx_lines",                            "Number of lines in kernel PTX."},
  { "range_index",                          "Index of the launch index range to which an event belongs."},
  { "total_sec",                            "Total time in seconds spent in run."},

  { "lwda_device",                          "LWCA device ordinal."},
  { "gpu_name",                             "Name of GPU."},
  { "sm_arch",                              "SM architecture version."},
  { "sm_count",                             "Number of SMs."},
  { "sm_KHz",                               "SM clock rate in kilohertz."},
  { "gpu_total_MB",                         "GPU total memory in megabytes."},
  { "tcc",                                  "TCC driver mode."},
  { "compatible_devices",                   "Ordinals of compatible devices."},
  { "rtcore_version",                       "RT core version supported by the device."},

  { "host_name",                            "Name of the machine on which the metrics were collected."},
  { "platform",                             "Linux|Windows|Mac."},
  { "cpu_name",                             "Description of CPU."},
  { "num_cpu_cores",                        "Number of CPU cores."},
  { "driver_version",                       "GPU driver version."},
  { "all_gpus",                             "Names of all GPUs in the system."},
  { "start_time",                           "Time at which metrics collection began."},
  { "available_memory",                     "Available host memory in bytes."},
  { "build_description",                    "OptiX build description string."},
  { "nondefault_knobs",                     "List of nondefault knobs."},

  { "demand_texture_tiles_requested",       "Number of demand-loaded texture tiles requested." },
  { "demand_texture_tiles_requested_non_mipmapped", "Number of tiles requested from non-mipmapped textures (included in demand_texture_tiles_requested)." },
  { "demand_texture_megabytes_filled",      "Size of demand-loaded data filled (in MB)." },
  { "demand_texture_tiles_filled",          "Number of demand-loaded texture tiles filled." },
  { "demand_texture_tiles_filled_level<N>", "Number of tiles filled at level N (where coarsest level is zero)." },
  { "demand_texture_callback_msec",         "Time spent in demand loading callbacks (in milliseconds)." },
  { "demand_texture_copy_msec",             "Time spent copying demand loaded data to device (in milliseconds)." },
};
// clang-format on

// Removes '<.*>' from the string
static std::string stripEmbeddedVal( const std::string& str )
{
    size_t angleStartPos = str.find( '<' );
    if( angleStartPos == str.npos )
        return str;

    size_t angleEndPos = str.find( '>' );
    if( angleEndPos == str.npos )
        return str;

    return str.substr( 0, angleStartPos ) + str.substr( angleEndPos + 1 );
}

static std::string doubleToSingleQuotes( const std::string& str )
{
    std::string ret = str;
    for( char& c : ret )
        if( c == '"' )
            c = '\'';
    return ret;
}

namespace {

class MetricsLogger
{
  public:
    MetricsLogger( const std::string& filename );
    ~MetricsLogger();

    bool isEnabled();
    void setEnabled( bool val );
    void logInt( const char* name, uint64_t value );
    void logFloat( const char* name, double value );
    void logString( const char* name, const std::string& value );
    void pushScope( const char* name, Metrics::ScopeType type = Metrics::OBJECT );
    void popScope();
    void flush();

  private:
    timerTick             m_start     = 0;
    std::atomic<uint64_t> m_timestamp = {0};
    FILE*                 m_file      = nullptr;

    // Lock only taken on public functions, but not constructor or destructor as
    // there should be no contention there.
    std::mutex m_mutex;

    std::vector<char>     m_buffer;
    std::set<std::string> m_registeredMetrics;
    bool                  m_enabled = true;

    // A stack used to track the number of entries in each scope. This is used for
    // determining indentation level, when to insert commas, and unique names for
    // scopes.
    struct Scope
    {
        std::string        name;
        Metrics::ScopeType type;
        int                numEntries;
    };
    std::vector<Scope> m_scopeStack;

    void log( const char* name, const std::string& valueStr );

    // Adds separator char (if necessary), newline, and indentation to the level of
    // m_scopeStack.size(). Pass 0 for no separator char.
    std::string sep( char c = ',' );

    // Write string to m_buffer.
    void append( const std::string& str );

    // Track usage of metric
    void registerMetric( const char* name );

    // Throws if a registered metric doesn't have a description
    void verifyMetricDescriptions();

    void writeMetricDescriptions();

    double getTimestamp();
};

}  // namespace

MetricsLogger::MetricsLogger( const std::string& filename )
{
    m_start = getTimerTick();
    if( ( m_file = fopen( filename.c_str(), "wt" ) ) == nullptr )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Could not open file", filename );


    // Initialize the log
    m_buffer.reserve( DEFAULT_BUFFER_SIZE );
    append( "{" );
    m_scopeStack.push_back( {"", Metrics::OBJECT, 0} );

    // Add system info
    const SystemInfo info{getSystemInfo()};  // BL: I don't like this name. Environment info?

    std::string gpuList;
    for( size_t i = 0; i < info.gpuDescriptions.size(); ++i )
        gpuList += std::string( ( i > 0 ) ? "," : "" ) + "\"" + info.gpuDescriptions[i] + "\"";

    std::string nondefaultKnobList;
    for( const std::string& nondefaultKnob : info.nondefaultKnobs )
    {
        if( stringBeginsWith( nondefaultKnob, "metrics.runJson" ) )
            continue;  // filter this out because the values will already be included as part of run_info
        nondefaultKnobList += std::string( ( !nondefaultKnobList.empty() ) ? "," : "" ) + "\""
                              + escapeJsonString( doubleToSingleQuotes( nondefaultKnob ) ) + "\"";
    }

    pushScope( "system_info" );
    logString( "host_name", info.hostName );
    logString( "platform", info.platform );
    logString( "cpu_name", info.cpuName );
    logInt( "num_cpu_cores", info.numCpuCores );
    logString( "driver_version", info.driverVersion );
    log( "all_gpus", "[" + gpuList + "]" );
    popScope();

    pushScope( "run_info" );
    logString( "start_time", info.timestamp );
    logInt( "available_memory", info.availableMemory );
    logString( "build_description", info.buildDescription );
    log( "nondefault_knobs", "[" + nondefaultKnobList + "]" );
    if( !k_metricsRunJson.get().empty() )
        append( sep() + k_metricsRunJson.get() );
    popScope();

    // Start metrics
    pushScope( "metrics", Metrics::ARRAY );
}

MetricsLogger::~MetricsLogger()
{
    try
    {
        logFloat( "total_sec", getDeltaSeconds( m_start ) );
        popScope();

        // Note that verifyMetricDescriptions can throw exceptions, and nothing will catch it.
        verifyMetricDescriptions();
        writeMetricDescriptions();

        popScope();

        flush();
        fclose( m_file );
    }
    catch( const std::exception& bang )
    {
        lerr << "Caught exception while tearing down metrics: " << bang.what() << '\n';
    }
    catch( ... )
    {
        lerr << "Caught unknown exception while tearing down metrics\n";
    }
}

bool MetricsLogger::isEnabled()
{
    return m_enabled;
}

void MetricsLogger::setEnabled( bool val )
{
    m_enabled = val;
}

void MetricsLogger::logInt( const char* name, uint64_t value )
{
    std::lock_guard<std::mutex> guard( m_mutex );

    log( name, to_string( value ) );
}

void MetricsLogger::logFloat( const char* name, double value )
{
    std::lock_guard<std::mutex> guard( m_mutex );

    log( name, to_string( value ) );
}

void MetricsLogger::logString( const char* name, const std::string& value )
{
    std::lock_guard<std::mutex> guard( m_mutex );

    log( name, "\"" + escapeJsonString( value ) + "\"" );
}

void MetricsLogger::pushScope( const char* name, Metrics::ScopeType type )
{
    std::lock_guard<std::mutex> guard( m_mutex );

    Metrics::ScopeType lwrType       = m_scopeStack.back().type;
    std::string        scopeOpenChar = ( type == Metrics::ARRAY ) ? "[" : "{";
    std::string        field         = ( name ) ? name : "";
    if( name || lwrType == Metrics::OBJECT )
    {
        if( name )
            field = name;
        else
            field = stringf( "scope<%d>", m_scopeStack.back().numEntries );

        if( lwrType == Metrics::ARRAY )
        {
            // must be in an object to create a named scope
            append( sep() + "{" );
            m_scopeStack.back().numEntries++;
            m_scopeStack.push_back( {"<NAMED-WRAPPER>", Metrics::OBJECT, 0} );
        }

        append( sep() + "\"" + field + "\":" + scopeOpenChar );
    }
    else
    {
        append( sep() + scopeOpenChar );
    }

    m_scopeStack.back().numEntries++;
    m_scopeStack.push_back( {field, type, 0} );
}

void MetricsLogger::popScope()
{
    std::lock_guard<std::mutex> guard( m_mutex );

    std::string scopeCloseChar = ( m_scopeStack.back().type == Metrics::ARRAY ) ? "]" : "}";
    m_scopeStack.pop_back();
    append( sep( 0 ) + scopeCloseChar );

    if( !m_scopeStack.empty() && m_scopeStack.back().name == "<NAMED-WRAPPER>" )
    {
        // Pop again
        m_scopeStack.pop_back();
        append( sep( 0 ) + "}" );
    }
}

void MetricsLogger::flush()
{
    std::lock_guard<std::mutex> guard( m_mutex );

    fwrite( m_buffer.data(), 1, m_buffer.size(), m_file );
    fflush( m_file );
    m_buffer.clear();
}

void MetricsLogger::log( const char* name, const std::string& valueStr )
{
    std::string nameValue = stringf( "\"%s\":%s", name, valueStr.c_str() );
    std::string entry;
    if( m_scopeStack.back().type == Metrics::ARRAY )  // entry needs to be it's own object
        entry = sep() + "{" + nameValue + "}";
    else
        entry = sep() + nameValue;

    registerMetric( name );
    append( entry );
    m_scopeStack.back().numEntries++;
}

std::string MetricsLogger::sep( char c )
{
    std::string cStr;
    if( c && m_scopeStack.back().numEntries > 0 )
        cStr      = c;
    size_t indent = m_scopeStack.size();
    return cStr + "\n" + std::string( indent, '\t' );
}

void MetricsLogger::registerMetric( const char* name )
{
    m_registeredMetrics.insert( name );
}

void MetricsLogger::append( const std::string& str )
{
    m_buffer.insert( m_buffer.end(), str.begin(), str.end() );
}

void MetricsLogger::verifyMetricDescriptions()
{
    // Verify registered metrics
    std::set<std::string> nameWithDescSet;
    for( const MetricDescription& md : g_metricDescriptions )
        nameWithDescSet.insert( stripEmbeddedVal( md.name ) );

    for( const std::string& m : m_registeredMetrics )
    {
        if( nameWithDescSet.count( stripEmbeddedVal( m ) ) == 0 )
        {
            std::cerr << "CRASHING ERROR: Missing metric description " << m << std::endl;  // BL: There is no handler for this exception at this point
            throw IlwalidValue( RT_EXCEPTION_INFO, "Missing metric description", m );
        }
    }
}

void MetricsLogger::writeMetricDescriptions()
{
    pushScope( "descriptions" );
    for( const MetricDescription& md : g_metricDescriptions )
        logString( md.name, md.description );
    popScope();
}

double MetricsLogger::getTimestamp()
{
    return getDeltaMilliseconds( m_start );
}


// Singleton
static std::unique_ptr<MetricsLogger> g_metricsLogger;

void Metrics::init()
{
    if( k_metricsEnable.get() && !g_metricsLogger )
        g_metricsLogger.reset( new MetricsLogger( k_metricsFile.get() ) );
}

bool Metrics::isEnabled()
{
    return g_metricsLogger && g_metricsLogger->isEnabled();
}

void Metrics::setEnabled( bool val )
{
    if( g_metricsLogger )
        g_metricsLogger->setEnabled( val );
}

void Metrics::logInt( const char* name, uint64_t value )
{
    if( isEnabled() )
        g_metricsLogger->logInt( name, value );
}

void Metrics::logFloat( const char* name, double value )
{
    if( isEnabled() )
        g_metricsLogger->logFloat( name, value );
}


void Metrics::logString( const char* name, const char* value )
{
    if( isEnabled() )
        g_metricsLogger->logString( name, value );
}

void Metrics::pushScope( const char* name, ScopeType type )
{
    if( isEnabled() )
        g_metricsLogger->pushScope( name, type );
}

void Metrics::popScope()
{
    if( isEnabled() )
        g_metricsLogger->popScope();
}

void Metrics::flush()
{
    if( isEnabled() )
        g_metricsLogger->flush();
}

}  // namespace optix
