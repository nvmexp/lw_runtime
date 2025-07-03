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

#include <Context/ObjectManager.h>

#include <Context/Context.h>
#include <Context/SharedProgramManager.h>
#include <Context/TableManager.h>
#include <Context/UpdateManager.h>
#include <Device/CPUDevice.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <Memory/MemoryManager.h>
#include <Objects/Buffer.h>
#include <Objects/CommandList.h>
#include <Objects/GeometryTriangles.h>
#include <Objects/Group.h>
#include <Objects/LexicalScope.h>
#include <Objects/ManagedObject.h>
#include <Objects/Program.h>
#include <Objects/StreamBuffer.h>
#include <Objects/TextureSampler.h>
#include <Objects/VariableType.h>
#include <prodlib/misc/TimeViz.h>

#include <Util/BitSet.h>
#include <corelib/misc/Cast.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/UnknownError.h>
#include <prodlib/system/Knobs.h>

#include <algorithm>
#include <cstring>

using namespace optix;
using namespace corelib;
using namespace prodlib;

namespace {
// clang-format off
    Knob<bool> k_forceTrivialExceptionProgram( RT_DSTRING( "context.forceTrivialExceptionProgram" ), false, RT_DSTRING( "Force the use of a trivial exception program calling rtPrintExceptionDetails(). Needs printing enabled, e.g., print.enableOverride=1." ) );
// clang-format on
}


std::vector<ReusableIDValue> ObjectManager::reservedProgramIds;
std::vector<ReusableIDValue> ObjectManager::reservedBufferIds;
std::vector<ReusableIDValue> ObjectManager::reservedSamplerIds;

ObjectManager::ObjectManager( Context* context )
    : m_context( context )
    , m_scopes( 1 )                 // use 1-based IDs
    , m_programs( 1 )               // use 1-based IDs
    , m_buffers( 1 )                // use 1-based IDs
    , m_samplers( 1 )               // use 1-based IDs
    , m_stream_buffers( 1 )         // use 1-based IDs
    , m_command_lists( 1 )          // use 1-based IDs
    , m_postprocessing_stages( 1 )  // use 1-based IDs
    , m_traversables( 0 )           // 0-based
{
    // Apply any ID reservations
    for( auto id : reservedProgramIds )
        m_programs.reserveIdForHint( id );
    // required to preprocess reserved IDs before real usage starts
    m_programs.finalizeReservedIds();
    for( auto id : reservedBufferIds )
        m_buffers.reserveIdForHint( id );
    // required to preprocess reserved IDs before real usage starts
    m_buffers.finalizeReservedIds();
    for( auto id : reservedSamplerIds )
        m_samplers.reserveIdForHint( id );
    // required to preprocess reserved IDs before real usage starts
    m_samplers.finalizeReservedIds();
}

ObjectManager::~ObjectManager() NOEXCEPT_FALSE
{
    // If one of these fires, we're leaking objects somewhere
    RT_ASSERT_MSG( m_scopes.empty(), "ObjectManager deleted while holding scopes" );
    RT_ASSERT_MSG( m_programs.empty(), "ObjectManager deleted while holding programs" );
    RT_ASSERT_MSG( m_buffers.empty(), "ObjectManager deleted while holding buffers" );
    RT_ASSERT_MSG( m_samplers.empty(), "ObjectManager deleted while holding samplers" );
    RT_ASSERT_MSG( m_stream_buffers.empty(), "ObjectManager deleted while holding stream buffers" );
    RT_ASSERT_MSG( m_command_lists.empty(), "ObjectManager deleted while holding command lists" );
    RT_ASSERT_MSG( m_postprocessing_stages.empty(), "ObjectManager deleted while holding post-processing stages" );
}

void ObjectManager::preSetActiveDevices( const DeviceArray& removedDevices )
{
    const DeviceSet removedDeviceSet( removedDevices );

    // StreamBuffers may need to unload objects from devices
    for( const auto& stream : m_stream_buffers )
    {
        stream->preSetActiveDevices( removedDeviceSet );
    }

    for( const auto& scope : m_scopes )
    {
        if( GeometryGroup* gg = managedObjectCast<GeometryGroup>( scope ) )
            gg->preSetActiveDevices( removedDeviceSet );
    }
}

void ObjectManager::postSetActiveDevices( const DeviceArray& removedDevices )
{
    const DeviceSet removedDeviceSet( removedDevices );

    for( const auto& scope : m_scopes )
        if( GeometryGroup* gg = managedObjectCast<GeometryGroup>( scope ) )
            gg->postSetActiveDevices();

    // Buffers may decide to switch policy based on active devices
    for( const auto& buf : m_buffers )
        buf->postSetActiveDevices( removedDeviceSet );

    for( const auto& stream : m_stream_buffers )
        stream->postSetActiveDevices( removedDeviceSet );

    // If the active device list changes, all Program objects need to validate their set of
    // CanonicalPrograms against this.
    for( const auto& program : m_programs )
        program->postSetActiveDevices();
}

void ObjectManager::destroyAllObjects()
{
    // For debugging, save the nodegraph a this point (if the knob is
    // set).
    m_context->saveNodeGraph( "detaching all objects" );

    // Unmap all the buffers that the user left mapped.
    for( const auto& buffer : m_buffers )
    {
        if( buffer->isMappedHost() )
        {
            for( unsigned int level = 0; level < buffer->getMipLevelCount(); ++level )
            {
                if( buffer->isMappedHost( level ) )
                {
                    buffer->unmap( level );
                }
            }
        }
    }

    // Detach buffers and texture samplers
    for( const auto& buffer : m_buffers )
        buffer->detachFromParents();
    for( const auto& sampler : m_samplers )
        sampler->detachFromParents();
    for( const auto& sb : m_stream_buffers )
        sb->detachFromParents();

    // Detach all scopes - skipping the null program and the trivial exception program
    Program* nullProgram = m_context->getSharedProgramManager()->getNullProgram();
    Program* sharedExceptionProgram =
        k_forceTrivialExceptionProgram.get() ? m_context->getSharedProgramManager()->getTrivialExceptionProgram() : nullProgram;
    for( const auto& scope : m_scopes )
    {
        if( scope != nullProgram && scope != sharedExceptionProgram )
            // WARNING, for attribute programs, we attach the default attribute program if there was a custom attribute program!
            scope->detachFromParents();
    }
    GeometryTriangles::detachDefaultAttributeProgramFromParents( m_context );

    // And again, save the node graph
    m_context->saveNodeGraph( "destroying all objects" );

    // Destroy all scopes. Because the iterator becomes invalid on
    // delete, do not use a range based for loop.
    for( auto iter = m_scopes.begin(), end = m_scopes.end(); iter != end; )
    {
        LexicalScope* scope = *iter++;
        if( scope != nullProgram )
            delete scope;
    }

    // Now destroy the null program
    delete nullProgram;

    // Destroy buffers and texture samplers - being careful to not
    // ilwalidate the iterators.
    for( auto iter = m_buffers.begin(), end = m_buffers.end(); iter != end; )
        delete *iter++;
    for( auto iter = m_samplers.begin(), end = m_samplers.end(); iter != end; )
        delete *iter++;
    for( auto iter = m_stream_buffers.begin(), end = m_stream_buffers.end(); iter != end; )
        delete *iter++;
    for( auto iter = m_command_lists.begin(), end = m_command_lists.end(); iter != end; )
        delete *iter++;

    // Sanity check

    RT_ASSERT_MSG( m_scopes.empty(), "ObjectManager incomplete destruction of scopes" );
    RT_ASSERT_MSG( m_programs.empty(), "ObjectManager incomplete destruction of programs" );
    RT_ASSERT_MSG( m_buffers.empty(), "ObjectManager incomplete destruction of buffers" );
    RT_ASSERT_MSG( m_samplers.empty(), "ObjectManager incomplete destruction of samplers" );
    RT_ASSERT_MSG( m_stream_buffers.empty(), "ObjectManager incomplete destruction of stream buffers" );
    RT_ASSERT_MSG( m_command_lists.empty(), "ObjectManager incomplete destruction of command lists" );
    RT_ASSERT_MSG( m_postprocessing_stages.empty(), "ObjectManager incomplete destruction of post-processing stages" );
    RT_ASSERT_MSG( m_traversables.empty(), "ObjectManager incomplete destruction of traversables" );
}

ReusableID ObjectManager::registerObject( LexicalScope* scope )
{
    ReusableID scope_id = m_scopes.insert( scope );
    RT_ASSERT( *scope_id != 0 );
    TIMEVIZ_COUNT( "LexicalScopes", m_scopes.size() );
    return scope_id;
}

ReusableID ObjectManager::registerObject( Program* program )
{
    ReusableID program_id = m_programs.insert( program, m_nextProgramIdHint );
    llog( 30 ) << "Registered program with ID " << *program_id << " (hint: " << m_nextProgramIdHint << ")\n";
    m_nextProgramIdHint = ReusableIDMap<Program*>::NoHint;
    m_context->getTableManager()->notifyCreateProgram( *program_id, program );
    RT_ASSERT( *program_id != RT_PROGRAM_ID_NULL );
    TIMEVIZ_COUNT( "Programs", m_programs.size() );
    return program_id;
}

ReusableID ObjectManager::registerObject( Buffer* buffer )
{
    ReusableID buffer_id = m_buffers.insert( buffer, m_nextBufferIdHint );
    llog( 30 ) << "Registered buffer with ID " << *buffer_id << " (hint: " << m_nextBufferIdHint << ")\n";
    m_nextBufferIdHint = ReusableIDMap<Buffer*>::NoHint;
    m_context->getTableManager()->notifyCreateBuffer( *buffer_id, buffer );
    RT_ASSERT( *buffer_id != RT_BUFFER_ID_NULL );
    TIMEVIZ_COUNT( "Buffers", m_buffers.size() );
    return buffer_id;
}

ReusableID ObjectManager::registerObject( TextureSampler* sampler )
{
    ReusableID sampler_id = m_samplers.insert( sampler, m_nextTextureSamplerIdHint );
    llog( 30 ) << "Registered texture sampler with ID " << *sampler_id << " (hint: " << m_nextTextureSamplerIdHint << ")\n";
    m_nextTextureSamplerIdHint = ReusableIDMap<TextureSampler*>::NoHint;
    RT_ASSERT( *sampler_id != RT_TEXTURE_ID_NULL );
    m_context->getTableManager()->notifyCreateTextureSampler( *sampler_id, sampler );
    TIMEVIZ_COUNT( "TextureSamplers", m_samplers.size() );
    return sampler_id;
}

ReusableID ObjectManager::registerObject( StreamBuffer* stream_buffer )
{
    ReusableID stream_buffer_id = m_stream_buffers.insert( stream_buffer, ReusableIDMap<StreamBuffer*>::NoHint );
    llog( 30 ) << "Registered stream buffer with ID " << *stream_buffer_id << "\n";
    RT_ASSERT( *stream_buffer_id != RT_BUFFER_ID_NULL );
    TIMEVIZ_COUNT( "Stream Buffers", m_stream_buffers.size() );
    return stream_buffer_id;
}

ReusableID ObjectManager::registerObject( CommandList* command_list )
{
    ReusableID command_list_id = m_command_lists.insert( command_list, ReusableIDMap<CommandList*>::NoHint );
    llog( 30 ) << "Registered command list with ID " << *command_list_id << "\n";
    RT_ASSERT( *command_list_id != RT_COMMAND_LIST_ID_NULL );
    TIMEVIZ_COUNT( "Command Lists", m_command_lists.size() );
    return command_list_id;
}

ReusableID ObjectManager::registerObject( PostprocessingStage* stage )
{
    ReusableID postprocessing_stage_id = m_postprocessing_stages.insert( stage, ReusableIDMap<PostprocessingStage*>::NoHint );
    llog( 30 ) << "Registered post-processing stage with ID " << *postprocessing_stage_id << "\n";
    RT_ASSERT( *postprocessing_stage_id != RT_POSTPROCESSING_STAGE_ID_NULL );
    TIMEVIZ_COUNT( "PostprocessingStages", m_postprocessing_stages.size() );
    return postprocessing_stage_id;
}

ReusableID ObjectManager::registerTraversable( GraphNode* node )
{
    ReusableID traversable_id = m_traversables.insert( node, ReusableIDMap<GraphNode*>::NoHint );
    llog( 30 ) << "Registered traversable with ID " << *traversable_id << "\n";
    m_context->getTableManager()->notifyCreateTraversableHandle( *traversable_id, node );
    return traversable_id;
}

void ObjectManager::reserveProgramId( ReusableIDValue id )
{
    reservedProgramIds.push_back( id );
}

void ObjectManager::reserveBufferId( ReusableIDValue id )
{
    reservedBufferIds.push_back( id );
}

void ObjectManager::reserveTextureSamplerId( ReusableIDValue id )
{
    reservedSamplerIds.push_back( id );
}

void ObjectManager::setNextProgramIdHint( ReusableIDValue id )
{
    m_nextProgramIdHint = id;
}

void ObjectManager::setNextBufferIdHint( ReusableIDValue id )
{
    m_nextBufferIdHint = id;
}

void ObjectManager::setNextTextureSamplerIdHint( ReusableIDValue id )
{
    m_nextTextureSamplerIdHint = id;
}

LexicalScope* ObjectManager::getLexicalScopeById( ReusableIDValue id ) const
{
    return m_scopes.get( id );
}

Program* ObjectManager::getProgramById( ReusableIDValue id ) const
{
    return m_programs.get( id );
}

Program* ObjectManager::getProgramByIdNoThrow( ReusableIDValue id ) const
{
    Program* program = nullptr;
    m_programs.get( id, program );
    return program;
}

Buffer* ObjectManager::getBufferById( ReusableIDValue id ) const
{
    return m_buffers.get( id );
}

TextureSampler* ObjectManager::getTextureSamplerById( ReusableIDValue id ) const
{
    return m_samplers.get( id );
}

CommandList* ObjectManager::getCommandListById( ReusableIDValue id ) const
{
    return m_command_lists.get( id );
}

PostprocessingStage* ObjectManager::getPostprocessingStageById( ReusableIDValue id ) const
{
    return m_postprocessing_stages.get( id );
}

unsigned short ObjectManager::registerVariableName( const std::string& name )
{
    return m_variableNames.insert( name );
}

const std::string& ObjectManager::getVariableNameForToken( unsigned short token ) const
{
    return m_variableNames.get( token );
}

unsigned short ObjectManager::getTokenForVariableName( const std::string& name ) const
{
    unsigned short token = m_variableNames.getID( name );
    return token;
}

size_t ObjectManager::getVariableTokenCount() const
{
    return m_variableNames.size();
}

StreamBuffer* ObjectManager::getTargetStream( Buffer* source ) const
{
    // TODO: we can probably do this in a better way.
    for( const auto& buf : m_stream_buffers )
    {
        if( source == buf->getSource() )
            return buf;
    }

    return nullptr;
}
