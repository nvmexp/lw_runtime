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

#include <Context/ProfileManager.h>

#include <Context/Context.h>
#include <Context/ProgramManager.h>
#include <Device/CPUDevice.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <Memory/MemoryManager.h>
#include <Objects/Program.h>

#include <Util/CodeRange.h>
#include <corelib/misc/String.h>
#include <corelib/system/Timer.h>
#include <prodlib/misc/IomanipHelpers.h>
#include <prodlib/system/Knobs.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>

#include <stack>

using namespace optix;
using namespace prodlib;
using namespace corelib;


namespace {
// clang-format off
Knob<bool>       k_hostTimings( RT_DSTRING("stats.hostTimings"),  false, RT_DSTRING("Dump host timings") );
PublicKnob<bool> k_timeVpcs( RT_PUBLIC_DSTRING("stats.timeVpcs"), false, RT_PUBLIC_DSTRING("Measure average warp clocks spent in each VPC for some number of blocks") );
// clang-format on

static const unsigned int MAX_PROFILE_SIZE = 2048;
}  // namespace

namespace optix {
class CodeRangeTimer : public CodeRange::IHandler
{
  public:
    CodeRangeTimer() {}

    void push( const char* name ) override
    {
        ActiveCodeRangeInfo ARI;
        ARI.idx   = m_log.size();
        ARI.start = getTimerTick();
        m_stack.push( ARI );

        CodeRangeInfo CRI;
        CRI.name  = name;
        CRI.time  = -1;
        CRI.level = (unsigned)m_stack.size() - 1;
        m_log.push_back( CRI );
    }

    void pop() override
    {
        const ActiveCodeRangeInfo& ARI = m_stack.top();
        m_log[ARI.idx].time            = getDeltaSeconds( ARI.start, getTimerTick() );
        m_stack.pop();
    }

    void printLog( std::ostream& out )
    {
        for( CodeRangeInfo& i : m_log )
            out << std::setw( i.level * 2 ) << "" << i.name << " : " << stringf( "%.4f ms\n", i.time * 1000 );
    }

    void resetLog() { m_log.clear(); }

  private:
    struct CodeRangeInfo
    {
        std::string name;
        double      time;
        unsigned    level;
    };

    struct ActiveCodeRangeInfo
    {
        size_t             idx;
        corelib::timerTick start;
    };

    std::vector<CodeRangeInfo>      m_log;
    std::stack<ActiveCodeRangeInfo> m_stack;
};

}  // namespace optix


static std::string shorten( const std::string& str, int maxWidth, int ellipsesPos )
{
    if( static_cast<int>( str.size() ) > maxWidth )
        return str.substr( 0, ellipsesPos ) + ".." + str.substr( str.size() - ( maxWidth - ellipsesPos - 2 ) );
    else
        return str;
}


ProfileManager::ProfileManager( Context* context )
    : m_context( context )
    , m_counterOffset( 0 )
    , m_eventOffset( 0 )
    , m_timerOffset( 0 )
    , m_totalSize( 0 )
{
    m_codeRangeTimer.reset( new CodeRangeTimer );
    CodeRange::setHandler( m_codeRangeTimer.get() );
}

ProfileManager::~ProfileManager()
{
    CodeRange::setHandler( nullptr );
}

void ProfileManager::preSetActiveDevices( const DeviceArray& removedDevices )
{
    if( !k_timeVpcs.get() )
        return;

    // Need to explicitly free the allocations on the devices to be removed, since
    // we're using device-specific buffers
    for( const auto& device : removedDevices )
    {
        unsigned int deviceIndex = device->allDeviceListIndex();
        m_profileData[deviceIndex].reset();
    }
}

void ProfileManager::postSetActiveDevices()
{
    if( !k_timeVpcs.get() )
        return;

    MemoryManager* mm = m_context->getMemoryManager();
    DeviceManager* dm = m_context->getDeviceManager();

    m_profileData.resize( dm->allDevices().size() );

    const DeviceArray& devices = dm->activeDevices();
    for( Device* device : devices )
    {
        unsigned int     deviceIndex = device->allDeviceListIndex();
        BufferDimensions dataSize( RT_FORMAT_USER, sizeof( cort::uint64 ), 1, MAX_PROFILE_SIZE, 1, 1 );
        m_profileData[deviceIndex] = mm->allocateMBuffer( dataSize, MBufferPolicy::internal_readwrite_manualSync, device );
        // Map and unmap to ensure that the host allocation is created before we begin any launches
        mm->mapToHost( m_profileData[deviceIndex], MAP_WRITE_DISCARD );
        mm->unmapFromHost( m_profileData[deviceIndex] );
    }
}

void ProfileManager::beginKernelLaunch()
{
    if( !k_timeVpcs.get() )
        return;

    MemoryManager* mm = m_context->getMemoryManager();
    DeviceManager* dm = m_context->getDeviceManager();

    RT_ASSERT( m_profileData.size() == dm->allDevices().size() );

    // Fill in params for each launched device.  TODO: use an async memset for LWCA devices?  Then we avoid the
    // manualsync.
    const DeviceArray& devices = dm->activeDevices();
    for( Device* device : devices )
    {
        unsigned int  deviceIndex = device->allDeviceListIndex();
        MBufferHandle mbuffer     = m_profileData[deviceIndex];
        char*         hptr        = mm->mapToHost( mbuffer, MAP_WRITE_DISCARD );
        memset( hptr, 0, mbuffer->getDimensions().getTotalSizeInBytes() );
        mm->unmapFromHost( mbuffer );
        mm->manualSynchronize( mbuffer );
    }
}

void ProfileManager::finalizeKernelLaunch( const ProfileMapping* profile )
{
    if( !k_timeVpcs.get() )
        return;

    MemoryManager* mm = m_context->getMemoryManager();
    DeviceManager* dm = m_context->getDeviceManager();

    RT_ASSERT( m_profileData.size() == dm->allDevices().size() );

    const DeviceArray& devices = dm->activeDevices();
    for( Device* device : devices )
    {
        unsigned int deviceIndex = device->allDeviceListIndex();

        MBufferHandle mbuffer = m_profileData[deviceIndex];
        mm->manualSynchronize( mbuffer );
        char*               hptr = mm->mapToHost( mbuffer, MAP_READ );
        unsigned long long* data = reinterpret_cast<unsigned long long*>( hptr );
        dumpProfilerOutputForDevice( deviceIndex, data, profile );
        mm->unmapFromHost( mbuffer );
    }
}

void ProfileManager::finalizeApiLaunch()
{
    if( k_hostTimings.get() )
        m_codeRangeTimer->printLog( lprint_stream );
    m_codeRangeTimer->resetLog();
}

void ProfileManager::printCounter( unsigned long long* data, int counterId, const std::string& name, bool printHeader ) const
{
    if( printHeader )
    {
        // Counter header
        lprint
            << "====================================================================================================\n";
        lprint
            << " item                                                                                       count   \n";
        lprint
            << "----------------------------------------------------------------------------------------------------\n";
    }

    cort::uint64 rawCount = data[counterId];
    lprint << std::right << std::setfill( ' ' ) << std::setw( 3 ) << counterId << ": " << std::left << std::setw( 80 )
           << name << std::right << std::setfill( ' ' ) << std::setw( 12 ) << std::showpoint
           << static_cast<long long>( rawCount ) << std::endl;
}

void ProfileManager::printEvent( unsigned long long* data, int eventId, const std::string& name, double scale, bool printHeader ) const
{
    if( printHeader )
    {
        // Event header
        lprint
            << "====================================================================================================\n";
        lprint
            << " item,                                                                ilwocs,            simd-util  \n";
        lprint
            << "----------------------------------------------------------------------------------------------------\n";
    }

    // TODO: use a struct here
    cort::uint64 rawUtilization = data[2 * eventId + 0];
    cort::uint64 rawIlwocations = data[2 * eventId + 1];
    double       ilwocations    = (double)rawIlwocations * scale;
    double       simdutil       = ilwocations > 0. ? (double)rawUtilization * scale / ilwocations : 0;

    lprint << std::right << std::setfill( ' ' ) << std::setw( 3 ) << eventId << "," << std::setw( 35 )
           << shorten( name, 35, 20 ) << "," << std::right << std::setfill( ' ' ) << std::setw( 34 ) << std::fixed
           << std::setprecision( 1 ) << ilwocations << ", " << std::setw( 19 ) << std::fixed << std::setprecision( 1 )
           << simdutil << "," << std::endl;
}

void ProfileManager::printTimer( unsigned long long* data,
                                 int                 timerId,
                                 const std::string&  name,
                                 unsigned long long  totalClocks,
                                 double              scale,
                                 bool                printHeader ) const
{
    if( printHeader )
    {
        // Timer header
        lprint
            << "====================================================================================================\n";
        lprint
            << " timer,           state_function_name,           clocks,     clk%,     ilwocs,   clk/ilwoc, simd-utl\n";
        lprint
            << "----------------------------------------------------------------------------------------------------\n";
    }

    // TODO: use a struct here
    cort::uint64 rawClocks      = data[3 * timerId + 0];
    cort::uint64 rawUtilization = data[3 * timerId + 1];
    cort::uint64 rawIlwocations = data[3 * timerId + 2];
    double       clkpercent     = totalClocks > 0 ? (double)rawClocks / totalClocks * 100 : 0;
    double       clocks         = (double)rawClocks * scale;
    double       ilwocations    = (double)rawIlwocations * scale;
    double       simdutil       = ilwocations > 0. ? (double)rawUtilization * scale / ilwocations : 0;
    double       clocksperilw   = ilwocations > 0. ? ( clocks / ilwocations ) : 0;

    lprint << std::right << std::setfill( ' ' ) << std::setw( 3 ) << timerId << ", " << std::setw( 32 )
           << shorten( name, 32, 20 ) << "," << std::right << std::setfill( ' ' ) << std::setw( 18 ) << std::showpoint
           << static_cast<long long>( clocks ) << "," << std::setw( 8 ) << std::fixed << std::setprecision( 1 )
           << clkpercent << "%, " << std::setw( 10 ) << std::fixed << std::setprecision( 1 ) << ilwocations << ", "
           << std::setw( 10 ) << std::showpoint << static_cast<long long>( clocksperilw ) << ", " << std::setw( 5 )
           << std::fixed << std::setprecision( 1 ) << simdutil << "," << std::endl;
}

void ProfileManager::dumpProfilerOutputForDevice( unsigned int deviceIndex, unsigned long long* data, const ProfileMapping* profile ) const
{
    double       scale       = 1;
    unsigned int numCounters = 0;
    unsigned int numEvents   = 0;
    unsigned int numTimers   = 0;
    if( profile )
    {
        numCounters = profile->m_counters.size();
        numEvents   = profile->m_events.size();
        numTimers   = profile->m_timers.size();
    }

    // Useful for not corrupting the logger stream with different formatting options.
    IOSSaver saveFormat( lprint_stream );

    lprint << "\n";
    lprint << "Profile data for device " << deviceIndex << ":\n";

    // Print built-in counters
    unsigned long long* cstart = data + m_counterOffset;
    printCounter( cstart, cort::PROFILE_TRACE_COUNTER, "Total trace ilwocations", true );

    // Print user counters
    for( unsigned int i = 0; i < numCounters; ++i )
    {
        const ProfileMapping::Counter& counterInfo = profile->m_counters[i];
        printCounter( cstart, i, counterInfo.name, false );
    }


    // Print user events
    unsigned long long* estart = data + m_eventOffset;
    for( unsigned int i = 0; i < numEvents; ++i )
    {
        const ProfileMapping::Event& eventInfo = profile->m_events[i];
        printEvent( estart, i, eventInfo.name, scale, i == 0 );
    }


    // Print built-in timers
    unsigned long long* tstart      = data + m_timerOffset;
    unsigned long long  totalClocks = tstart[3 * cort::PROFILE_FULL_KERNEL_TIMER + 0];
    printTimer( tstart, cort::PROFILE_FULL_KERNEL_TIMER, "Full kernel", totalClocks, scale, true );

    // Print user timers
    lprint << "----------------------------------------------------------------------------------------------------\n";
    for( unsigned int i = 0; i < numTimers; ++i )
    {
        const ProfileMapping::Timer& timerInfo = profile->m_timers[i];
        if( timerInfo.kind != ProfileMapping::Invalid )
            printTimer( tstart, i, timerInfo.name, totalClocks, scale, false );
    }

    // Print user timers again for each canonical program id
    lprint << "----------------------------------------------------------------------------------------------------\n";
    std::vector<unsigned long long> percpid;
    for( unsigned int i = 0; i < numTimers; ++i )
    {
        const ProfileMapping::Timer& timerInfo = profile->m_timers[i];
        if( timerInfo.kind == ProfileMapping::Program )
        {
            if( 3 * timerInfo.cpid >= (int)percpid.size() )
                percpid.resize( 3 * ( timerInfo.cpid + 1 ), 0 );
            percpid[3 * timerInfo.cpid + 0] += tstart[3 * i + 0];
            percpid[3 * timerInfo.cpid + 1] += tstart[3 * i + 1];
            percpid[3 * timerInfo.cpid + 2] += tstart[3 * i + 2];
        }
    }
    ProgramManager* pm = m_context->getProgramManager();
    for( unsigned int i = 0; i < percpid.size(); i += 3 )
    {
        int         idx  = i / 3;
        std::string name = pm->getCanonicalProgramById( idx )->getUniversallyUniqueName();
        printTimer( &percpid[0], idx, name, totalClocks, scale, false );
    }

    lprint << "====================================================================================================\n";
}

void* ProfileManager::getProfileDataDevicePointer( const Device* device )
{
    if( !k_timeVpcs.get() )
        return nullptr;

    unsigned int deviceIndex = device->allDeviceListIndex();
    if( !m_profileData[deviceIndex] )
        return nullptr;
    return m_profileData[deviceIndex]->getAccess( deviceIndex ).getLinear().ptr;
}

void ProfileManager::specializeModule( llvm::Module* module, int numCounters, int numEvents, int numTimers )
{
    using namespace llvm;
    LLVMContext& llvmContext = module->getContext();

    // Layout the profile data buffer. Reserved entities have negative indices so
    // offset accordingly
    m_counterOffset = 0 + cort::PROFILE_NUM_RESERVED_COUNTERS;
    m_eventOffset   = m_counterOffset + numCounters + cort::PROFILE_NUM_RESERVED_EVENTS * 2;
    m_timerOffset   = m_eventOffset + numEvents * 2 + cort::PROFILE_NUM_RESERVED_TIMERS * 3;
    m_totalSize     = m_timerOffset + numTimers;

    // Ensure that global buffer is big enough
    RT_ASSERT_MSG( m_totalSize <= MAX_PROFILE_SIZE, "Number of profile data slots exceeded" );

    // setup compile-time constant enable flag
    // disabled profiler should be removed by LLVM optimization pass
    GlobalVariable* enableFlag = module->getGlobalVariable( "Profile_enabled" );
    if( enableFlag )
    {
        enableFlag->setExternallyInitialized( false );
        enableFlag->setLinkage( GlobalValue::InternalLinkage );
        enableFlag->setInitializer( ConstantInt::get( Type::getInt32Ty( llvmContext ), k_timeVpcs.get() ) );
    }

    struct VarInfo
    {
        const char*  name;
        unsigned int offset;
    } varInfo[] = {
        {"Profile_counterOffset", m_counterOffset}, {"Profile_eventOffset", m_eventOffset}, {"Profile_timerOffset", m_timerOffset},
    };
    for( const VarInfo& i : varInfo )
    {
        GlobalVariable* offsetVar = module->getGlobalVariable( i.name );
        if( offsetVar )
        {
            offsetVar->setExternallyInitialized( false );
            offsetVar->setLinkage( GlobalValue::InternalLinkage );
            offsetVar->setInitializer( ConstantInt::get( Type::getInt32Ty( llvmContext ), i.offset ) );
        }
    }
}

std::unique_ptr<ProfileMapping> ProfileManager::makeProfileMappingAndUpdateModule( int           numCounters,
                                                                                   int           numEvents,
                                                                                   int           numTimers,
                                                                                   llvm::Module* module )
{
    specializeModule( module, numCounters, numEvents, numTimers );
    return std::unique_ptr<ProfileMapping>( new ProfileMapping( numCounters, numEvents, numTimers ) );
}

void ProfileManager::disableProfilingInModule( llvm::Module* module )
{
    specializeModule( module, 0, 0, 0 );
}

bool ProfileManager::launchTimingEnabled() const
{
    return k_timeVpcs.get();
}
