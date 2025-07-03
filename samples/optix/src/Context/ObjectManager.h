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

#pragma once

#include <Device/Device.h>

#include <Util/IDMap.h>
#include <Util/ReusableIDMap.h>
#include <corelib/misc/Concepts.h>

#include <memory>
#include <set>


namespace optix {

class Buffer;
class CommandList;
class Context;
class GraphNode;
class LexicalScope;
class ManagedObject;
class PostprocessingStage;
class Program;
class StreamBuffer;
class TextureSampler;

class ObjectManager : private corelib::NonCopyable
{
  public:
    ObjectManager( Context* context );
    ~ObjectManager() NOEXCEPT_FALSE;

    // React to device changes
    void preSetActiveDevices( const DeviceArray& removedDevices );
    void postSetActiveDevices( const DeviceArray& removedDevices );

    // Destroy all objects in the system (on context destroy)
    void destroyAllObjects();

    // Register all objects of a specific type, internal or user-created, so that we have access
    // to the complete set for that type for internal processing. The registered API
    // object set can contain more than one type and only a subset of the objects of this
    // list for a specific type. Programs are also LexicalScopes and thus end up with 2 IDs
    // and appear in two lists.
    ReusableID registerObject( LexicalScope* scope );
    ReusableID registerObject( Program* program );
    ReusableID registerObject( Buffer* buffer );
    ReusableID registerObject( TextureSampler* sampler );
    ReusableID registerObject( StreamBuffer* stream_buffer );
    ReusableID registerObject( CommandList* command_list );
    ReusableID registerObject( PostprocessingStage* stage );

    // Traversables are used for RTX when a graphnode object is reachable directly by
    // a reference to a variable. The table manager will store an entry for each of these
    // that contains a traversable handle.
    ReusableID registerTraversable( GraphNode* graphNode );

    // Reserve/set the ID hints to be used the next time a corresponding object
    // is registered.  This is for OAC replay of legacy traces that used a
    // different ID allocation scheme. The trace might have ID values stored in
    // buffers and variables and thus wouldn't work if the IDs changed.  The
    // reservations are static so that we can take reservations before any
    // context is even created. This avoids clashes when the context creation
    // itself allocates IDs (such as for the null program).
    static void reserveProgramId( ReusableIDValue id );
    static void reserveBufferId( ReusableIDValue id );
    static void reserveTextureSamplerId( ReusableIDValue id );
    void setNextProgramIdHint( ReusableIDValue id );
    void setNextBufferIdHint( ReusableIDValue id );
    void setNextTextureSamplerIdHint( ReusableIDValue id );

    // Get the object through its ID
    LexicalScope* getLexicalScopeById( ReusableIDValue id ) const;
    Program* getProgramById( ReusableIDValue id ) const;
    Program* getProgramByIdNoThrow( ReusableIDValue id ) const;
    Buffer* getBufferById( ReusableIDValue id ) const;
    TextureSampler* getTextureSamplerById( ReusableIDValue id ) const;
    CommandList* getCommandListById( ReusableIDValue id ) const;
    PostprocessingStage* getPostprocessingStageById( ReusableIDValue id ) const;

    // Access the reserved IDs. Needed for VCA support
    static std::vector<ReusableIDValue>& getReservedProgramIds() { return ObjectManager::reservedProgramIds; }
    static std::vector<ReusableIDValue>& getReservedBufferIds() { return ObjectManager::reservedBufferIds; }
    static std::vector<ReusableIDValue>& getReservedTextureSamplerIds() { return ObjectManager::reservedSamplerIds; }

    // Raw object lists
    const ReusableIDMap<LexicalScope*>&        getLexicalScopes() const { return m_scopes; }
    const ReusableIDMap<Program*>&             getPrograms() const { return m_programs; }
    const ReusableIDMap<Buffer*>&              getBuffers() const { return m_buffers; }
    const ReusableIDMap<TextureSampler*>&      getTextureSamplers() const { return m_samplers; }
    const ReusableIDMap<StreamBuffer*>&        getStreamBuffers() const { return m_stream_buffers; }
    const ReusableIDMap<CommandList*>&         getCommandLists() const { return m_command_lists; }
    const ReusableIDMap<PostprocessingStage*>& getPostprocessingStages() const { return m_postprocessing_stages; }
    const ReusableIDMap<GraphNode*>&           getTraversables() const { return m_traversables; }

    // Register a variable name and return a permanent unique identifier token
    unsigned short registerVariableName( const std::string& name );
    const std::string& getVariableNameForToken( unsigned short token ) const;
    unsigned short getTokenForVariableName( const std::string& name ) const;  // returns -1 when not found
    size_t getVariableTokenCount() const;

    StreamBuffer* getTargetStream( Buffer* source ) const;

  private:
    Context* m_context;

    std::unique_ptr<Program> m_nullProgram;

    ReusableIDMap<LexicalScope*>        m_scopes;
    ReusableIDMap<Program*>             m_programs;
    ReusableIDMap<Buffer*>              m_buffers;
    ReusableIDMap<TextureSampler*>      m_samplers;
    ReusableIDMap<StreamBuffer*>        m_stream_buffers;
    ReusableIDMap<CommandList*>         m_command_lists;
    ReusableIDMap<PostprocessingStage*> m_postprocessing_stages;
    ReusableIDMap<GraphNode*>           m_traversables;

    // Reserved IDs and ID hints
    ReusableIDValue m_nextProgramIdHint        = ReusableIDMap<Program*>::NoHint;
    ReusableIDValue m_nextBufferIdHint         = ReusableIDMap<Buffer*>::NoHint;
    ReusableIDValue m_nextTextureSamplerIdHint = ReusableIDMap<TextureSampler*>::NoHint;

    static std::vector<ReusableIDValue> reservedProgramIds;
    static std::vector<ReusableIDValue> reservedBufferIds;
    static std::vector<ReusableIDValue> reservedSamplerIds;

    IDMap<std::string, unsigned short> m_variableNames;
};
}  // end namespace optix
