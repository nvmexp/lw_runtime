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

#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <Objects/GraphProperty.h>
#include <Objects/VariableReferenceBinding.h>

#include <corelib/misc/Concepts.h>
#include <prodlib/exceptions/Assert.h>

#include <map>
#include <memory>
#include <queue>
#include <vector>


namespace optix {

class Context;
class LexicalScope;
class Program;

class BindingManager : private corelib::NonCopyable
{
  public:
    // SGP Note: As the BindingManager evolves to track other global
    // properties, we should consider renaming it to ConnectionManager or
    // similar.  add/remove connection
    BindingManager( Context* context );
    ~BindingManager();


    // Bindings (scope+offset paairs) for variables. It is not practical
    // or necessary to store ilwerse bindings or references to specific
    // variables.
    void addOrRemove_VariableBinding( VariableReferenceID refid, const VariableReferenceBinding& binding, bool added );
    typedef GraphProperty<VariableReferenceBinding> VariableBindingSet;
    const VariableBindingSet& getVariableBindingsForReference( VariableReferenceID id ) const;


    // Bindings and ilwerse bindings for objects types (graphnode,
    // program, buffer, texture)
    void addOrRemove_GraphNodeBinding( VariableReferenceID refid, int scopeID, bool added );
    void addOrRemove_ProgramBinding( LexicalScope* atScope, Program* program, VariableReferenceID refid, bool added );
    void addOrRemove_ProgramIdBinding( VariableReferenceID refid, int programID, bool added );
    void addOrRemove_BufferBinding( VariableReferenceID refid, int bufferID, bool added );
    void addOrRemove_TextureBinding( VariableReferenceID refid, int textureID, bool added );

    typedef GraphProperty<int> GraphNodeBindingSet;
    typedef GraphProperty<int> ProgramBindingSet;
    typedef GraphProperty<int> BufferBindingSet;
    typedef GraphProperty<int> TextureBindingSet;
    const GraphNodeBindingSet& getGraphNodeBindingsForReference( VariableReferenceID id ) const;
    const ProgramBindingSet& getProgramBindingsForReference( VariableReferenceID id ) const;
    const BufferBindingSet& getBufferBindingsForReference( VariableReferenceID id ) const;
    const TextureBindingSet& getTextureBindingsForReference( VariableReferenceID id ) const;

    typedef GraphProperty<VariableReferenceID> IlwerseGraphNodeBindingSet;
    typedef GraphProperty<VariableReferenceID> IlwerseProgramBindingSet;
    typedef GraphProperty<VariableReferenceID> IlwerseBufferBindingSet;
    typedef GraphProperty<VariableReferenceID> IlwerseTextureBindingSet;
    const IlwerseGraphNodeBindingSet& getIlwerseBindingsForGraphNodeId( int scopeID );
    const IlwerseProgramBindingSet& getIlwerseBindingsForProgramId( int programID );
    const IlwerseBufferBindingSet& getIlwerseBindingsForBufferId( int bufferID );
    const IlwerseTextureBindingSet& getIlwerseBindingsForTextureId( int textureID );

    // Transform height (global across the graph)
    int  getMaxTransformHeight() const;
    void receivePropertyDidChange_TransformHeight( int height, bool added );

    // Acceleration height (global across the graph)
    int  getMaxAccelerationHeight() const;
    void receivePropertyDidChange_AccelerationHeight( int height, bool added );

    // Does any transform in the graph have motion
    bool hasMotionTransforms() const;
    void receivePropertyDidChange_HasMotionTransforms( bool added );

    // A global queue for managing unresolved reference and other
    // property connections / disconnections for bound callable
    // programs.
    void enqueueVirtualParentConnectOrDisconnect( Program* program, const ProgramRoot& root, bool connect );
    void processVirtualParentQueue();
    void forceDetach( const ProgramRoot& root );

  private:
    Context* m_context;

    // Bindings
    GraphPropertyMulti<VariableReferenceID, VariableReferenceBinding>  m_variableBindings;
    GraphPropertyMulti<VariableReferenceID, int>                       m_graphNodeBindings;
    GraphPropertyMulti<VariableReferenceID, int>                       m_programBindings;
    GraphPropertyMulti<VariableReferenceID, int>                       m_bufferBindings;
    GraphPropertyMulti<VariableReferenceID, int>                       m_textureBindings;
    GraphProperty<std::pair<VariableReferenceID, VariableReferenceID>> m_attributeBindings;

    // Ilwerse bindings
    GraphPropertyMulti<int, VariableReferenceID> m_ilwerseGraphNodeBindings;
    GraphPropertyMulti<int, VariableReferenceID> m_ilwerseProgramBindings;
    GraphPropertyMulti<int, VariableReferenceID> m_ilwerseBufferBindings;
    GraphPropertyMulti<int, VariableReferenceID> m_ilwerseTextureBindings;

    // Global transform height
    GraphProperty<int, true> m_transformHeight;
    GraphProperty<int, true> m_accelerationHeight;

    // Global: does graph have motion transforms anywhere
    GraphPropertySingle<int> m_hasMotionTransforms;

    // Queue of pending virtual parent connections/disconnections
    struct Modification
    {
        Program*    program;
        ProgramRoot root;
        bool        connect;
    };
    std::queue<Modification> modq;
    bool                     m_processingQueue = false;

    // Count of virtual programs attached to a particular root. Used to
    // clean up reference loops for bound callable programs in
    // Program::attachOrDetach_UnresolvedReference.
    GraphProperty<ProgramRoot, true> m_virtualChildren;
    ProgramRoot m_ignoreRoot = {-1, ST_ILWALID, ~0U};

    // Let NodegraphPrinter see the GraphProperties
    friend class NodegraphPrinter;
};
}
