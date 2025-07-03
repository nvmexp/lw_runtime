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

#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/WaitHandle.h>
#include <Memory/MemoryManager.h>
#include <Objects/CommandList.h>
#include <Objects/PostprocessingStage.h>
#include <Util/LinkedPtrHelpers.h>

#include <lwda_runtime.h>

#include <prodlib/exceptions/IlwalidOperation.h>
#include <prodlib/misc/TimeViz.h>

#include <corelib/system/LwdaDriver.h>
#include <corelib/system/Timer.h>

using namespace optix;

CommandList::CommandList( Context* context )
    : ManagedObject( context, RT_OBJECT_COMMANDLIST )
{
    m_id = m_context->getObjectManager()->registerObject( this );
}

CommandList::~CommandList()
{
}

void CommandList::appendPostprocessingStage( PostprocessingStage* stage, RTsize launch_width, RTsize launch_height )
{
    if( m_isFinalized )
    {
        throw prodlib::IlwalidOperation(
            RT_EXCEPTION_INFO, "Cannot append postprocessing stage to command list: Command list has been finalized." );
    }
    unsigned int old_num_stages = m_stages.size();
    // this is really lwmbersome, but I'm not sure about using push_back with LinkedPtr's move semantics:
    m_stages.resize( old_num_stages + 1 );
    m_stages[old_num_stages].set( this, stage );
    CommandDescriptor desc( old_num_stages, 2, launch_width, launch_height, 1, false );
    m_commands.push_back( desc );
}

void CommandList::appendLaunch( unsigned int entryIndex, RTsize launch_width )
{
    if( m_isFinalized )
    {
        throw prodlib::IlwalidOperation( RT_EXCEPTION_INFO,
                                         "Cannot append launch to command list: Command list has been finalized." );
    }
    CommandDescriptor desc( entryIndex, 1, launch_width, 1, 1, true );
    m_commands.push_back( desc );
}

void CommandList::appendLaunch( unsigned int entryIndex, RTsize launch_width, RTsize launch_height )
{
    if( m_isFinalized )
    {
        throw prodlib::IlwalidOperation( RT_EXCEPTION_INFO,
                                         "Cannot append launch to command list: Command list has been finalized." );
    }
    CommandDescriptor desc( entryIndex, 2, launch_width, launch_height, 1, true );
    m_commands.push_back( desc );
}

void CommandList::appendLaunch( unsigned int entryIndex, RTsize launch_width, RTsize launch_height, RTsize launch_depth )
{
    if( m_isFinalized )
    {
        throw prodlib::IlwalidOperation( RT_EXCEPTION_INFO,
                                         "Cannot append launch to command list: Command list has been finalized." );
    }
    CommandDescriptor desc( entryIndex, 3, launch_width, launch_height, launch_depth, true );
    m_commands.push_back( desc );
}

void optix::CommandList::setLwdaStream( void* stream )
{
    if( m_isFinalized )
    {
        throw prodlib::IlwalidOperation( RT_EXCEPTION_INFO, "Cannot set lwca stream on a finalized command list." );
    }
    m_syncStream = static_cast<LWstream>( stream );
}

void optix::CommandList::getLwdaStream( void** stream )
{
    *stream = m_syncStream;
}

void CommandList::finalize()
{
    if( m_isFinalized )
    {
        // do nothing
        return;
    }
    m_isFinalized = true;

    // Check is the command list should execute async (sync stream set and no postprocessing stages)
    m_exelwteAsync = m_syncStream;

    for( CommandDescriptor& desc : m_commands )
    {
        if( desc.isLaunch )
        {
            // Do nothing for launches
        }
        else
        {
            // Call initialize on post-processing stages
            m_stages[desc.stageOrEntryIndex]->initialize( desc.width, desc.height );
            m_exelwteAsync = false;
        }
    }
}

void optix::CommandList::setDevices( std::vector<unsigned int>& deviceList )
{
    m_devices = deviceList;
}

std::vector<unsigned int> optix::CommandList::getDevices()
{
    return m_devices;
}

namespace {
struct CommandRunTimeInfo
{
    CommandRunTimeInfo( const std::string& desc, double time )
        : m_desc( desc )
        , m_time( time )
    {
    }
    std::string m_desc;
    double      m_time = 0.;
};
}

void CommandList::execute() const
{
    if( !m_isFinalized )
    {
        throw prodlib::IlwalidOperation( RT_EXCEPTION_INFO,
                                         "Cannot execute command list: Command list has not been finalized." );
    }

    // Execute asynchronously if activated, otherwise execute synchronously.
    if( m_exelwteAsync )
        return exelwteAsync();

    // Make sure there is no async launches running.
    m_context->finishAsyncLaunches();

    TIMEVIZ_FUNC;
    // data collection for usage report
    corelib::timerTick              startTime = corelib::getTimerTick();
    std::vector<CommandRunTimeInfo> commandRunTimes;
    commandRunTimes.reserve( m_commands.size() );

    // validate PostprocessingStages here on execute because:
    // a) they are LexicalScopes, but not attached, so won't be validated by the ValidationManager
    // b) only stages in command lists that are exelwted should have to be validated (having non-functional/unfinished command lists is okay.
    // This means that validation is lwrrently not done lazily for stages, which will cause some overhead. Could be optimized in the future.
    for( const CommandDescriptor& desc : m_commands )
    {
        if( !desc.isLaunch )
        {
            m_stages[desc.stageOrEntryIndex]->validate();
        }
    }

    m_context->getMemoryManager()->enterFromAPI();

    bool active = false;

    for( const CommandDescriptor& desc : m_commands )
    {
        // take time of either launch or stage for usage report
        corelib::timerTick cmdStartTime = corelib::getTimerTick();
        if( desc.isLaunch )
        {
            deactivateMemoryManager( active );
            const DeviceSet launchDevices =
                ( m_devices.size() > 0 ) ? DeviceSet( m_devices ) : m_context->getDeviceManager()->activeDevices();
            m_context->launchFromCommandList( desc.stageOrEntryIndex, desc.dim, desc.width, desc.height, desc.depth,
                                              launchDevices, lwca::Stream() );
        }
        else
        {
            activateMemoryManager( active );
            m_stages[desc.stageOrEntryIndex]->launch( desc.width, desc.height );
        }
        double cmdTime = corelib::getDeltaMilliseconds( cmdStartTime );
        // add description for usage report
        std::string description;
        if( desc.isLaunch )
            description = "Launch with id " + std::to_string( desc.stageOrEntryIndex );
        else
            description = "Stage " + m_stages[desc.stageOrEntryIndex]->getName();
        // collect result of each command for final usage report
        commandRunTimes.push_back( CommandRunTimeInfo( description, cmdTime ) );

        if( !desc.isLaunch )
        {
            std::string command = m_stages[desc.stageOrEntryIndex]->getName();
            if( command == "DLDenoiser" )
                m_context->addDenoiserTimeSpent( cmdTime );
        }
    }

    double runTimeMs = corelib::getDeltaMilliseconds( startTime );
    // generate usage report
    UsageReport& ur = m_context->getUsageReport();
    if( ur.isActive( 2 ) )
    {
        ureport2( ur, "POSTPROCESSING" ) << "Command list took " << runTimeMs << " ms" << std::endl;
        UsageReport::IndentFrame urif( ur );
        for( auto report : commandRunTimes )
            ureport2( ur, "POSTPROCESSING" ) << "Command \'" << report.m_desc.c_str() << "\' took " << report.m_time
                                             << " ms" << std::endl;
    }
    deactivateMemoryManager( active );

    m_context->getMemoryManager()->exitToAPI();
}

void CommandList::exelwteAsync() const
{
    RT_ASSERT_MSG( m_exelwteAsync,
                   "Trying to execute command list asynchronously when it is set to launch synchronously." );

    if( !m_isFinalized )
    {
        throw prodlib::IlwalidOperation(
            RT_EXCEPTION_INFO, "Can't execute command list asynchronously: Command list has not been finalized." );
    }

    if( !m_syncStream )
    {
        throw prodlib::IlwalidOperation( RT_EXCEPTION_INFO,
                                         "Can't execute command list asynchronously: No sync stream has been set." );
    }

    // Launch all stages in sequence
    for( const CommandDescriptor& desc : m_commands )
    {
        // Post-processing stages are not allows for now
        RT_ASSERT( desc.isLaunch );

        const DeviceSet launchDevices =
            !m_devices.empty() ? DeviceSet( m_devices ) : m_context->getDeviceManager()->activeDevices();
        m_context->launchAsync( desc.stageOrEntryIndex, desc.dim, desc.width, desc.height, desc.depth, launchDevices,
                                lwca::Stream( m_syncStream ) );
    }
}

void CommandList::detachLinkedChild( const LinkedPtr_Link* link )
{
    unsigned int index;
    if( getElementIndex( m_stages, link, index ) )
        m_stages[index].set( this, nullptr );
}

void CommandList::activateMemoryManager( bool& active ) const
{
    if( active )
        return;

    m_context->getMemoryManager()->syncAllMemoryBeforeLaunch();
    m_context->getMemoryManager()->finalizeTransfersBeforeLaunch();
    active = true;
}

void CommandList::deactivateMemoryManager( bool& active ) const
{
    if( !active )
        return;

    m_context->getMemoryManager()->releaseMemoryAfterLaunch();
    active = false;
}
