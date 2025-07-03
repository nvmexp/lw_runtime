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

#include <Context/BindingManager.h>

#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <Context/UpdateManager.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <Objects/TextureSampler.h>

#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;


BindingManager::BindingManager( Context* context )
    : m_context( context )
{
}

BindingManager::~BindingManager()
{
}

const BindingManager::VariableBindingSet& BindingManager::getVariableBindingsForReference( VariableReferenceID refid ) const
{
    return m_variableBindings.getSubproperty( refid );
}

const BindingManager::BufferBindingSet& BindingManager::getBufferBindingsForReference( VariableReferenceID refid ) const
{
    return m_bufferBindings.getSubproperty( refid );
}

const BindingManager::TextureBindingSet& BindingManager::getTextureBindingsForReference( VariableReferenceID refid ) const
{
    return m_textureBindings.getSubproperty( refid );
}

const BindingManager::GraphNodeBindingSet& BindingManager::getGraphNodeBindingsForReference( VariableReferenceID refid ) const
{
    return m_graphNodeBindings.getSubproperty( refid );
}

const BindingManager::GraphNodeBindingSet& BindingManager::getProgramBindingsForReference( VariableReferenceID refid ) const
{
    return m_programBindings.getSubproperty( refid );
}

const BindingManager::IlwerseBufferBindingSet& BindingManager::getIlwerseBindingsForBufferId( int bufid )
{
    return m_ilwerseBufferBindings.getSubproperty( bufid );
}

const BindingManager::IlwerseBufferBindingSet& BindingManager::getIlwerseBindingsForTextureId( int texid )
{
    return m_ilwerseTextureBindings.getSubproperty( texid );
}

void BindingManager::addOrRemove_VariableBinding( VariableReferenceID refid, const VariableReferenceBinding& binding, bool added )
{
    const VariableReference* varref = m_context->getProgramManager()->getVariableReferenceById( refid );
    // If this is a linked reference, apply binding to the original reference id
    if( varref->getLinkedReference() )
        refid    = varref->getLinkedReference()->getReferenceID();
    bool changed = m_variableBindings.addOrRemoveProperty( refid, binding, added );
    if( changed )
        m_context->getUpdateManager()->eventVariableBindingsDidChange( refid, binding, true );
}

void BindingManager::addOrRemove_GraphNodeBinding( VariableReferenceID refid, int scopeid, bool added )
{
    bool changed  = m_graphNodeBindings.addOrRemoveProperty( refid, scopeid, added );
    bool ichanged = m_ilwerseGraphNodeBindings.addOrRemoveProperty( scopeid, refid, added );
    RT_ASSERT( changed == ichanged );

    // Direct Caller: propagates from reference
    // Trace Caller:  propagates from reference
    const VariableReference* varref = m_context->getProgramManager()->getVariableReferenceById( refid );
    GraphNode* node = managedObjectCast<GraphNode>( m_context->getObjectManager()->getLexicalScopeById( scopeid ) );
    CanonicalProgramID caller = varref->getParent()->getID();
    node->receivePropertyDidChange_DirectCaller( caller, added );
    node->receivePropertyDidChange_TraceCaller( caller, added );

    // No callback yet. Add if required.
}

void BindingManager::addOrRemove_ProgramBinding( LexicalScope* atScope, Program* program, VariableReferenceID refid, bool added )
{
    program->programBindingDidChange( atScope, refid, added );
    int programid = program->getId();

    const VariableReference* varref = m_context->getProgramManager()->getVariableReferenceById( refid );

    // If this is a linked reference, apply binding to the original reference id
    if( varref->getLinkedReference() )
    {
        varref = varref->getLinkedReference();
        refid  = varref->getReferenceID();
    }
    CanonicalProgramID caller = varref->getParent()->getID();
    // Direct Caller: propagates from reference
    program->receivePropertyDidChange_DirectCaller( caller, added );

    bool changed  = m_programBindings.addOrRemoveProperty( refid, programid, added );
    bool ichanged = m_ilwerseProgramBindings.addOrRemoveProperty( programid, refid, added );
    RT_ASSERT( changed == ichanged );
}

void BindingManager::addOrRemove_ProgramIdBinding( VariableReferenceID refid, int programid, bool added )
{
    const ProgramManager*    pm     = m_context->getProgramManager();
    const VariableReference* varref = pm->getVariableReferenceById( refid );

    // If this is a linked reference, apply binding to the original reference id
    if( varref->getLinkedReference() )
    {
        varref = varref->getLinkedReference();
        refid  = varref->getReferenceID();
    }

    // Update call site if it exists.
    std::string callSiteUniqueName = CallSiteIdentifier::generateCallSiteUniqueName( varref );

    CallSiteIdentifier* csId = pm->getCallSiteByUniqueName( callSiteUniqueName );
    if( csId )
    {
        Program* callee = m_context->getObjectManager()->getProgramByIdNoThrow( programid );
        if( callee )
        {
            RT_ASSERT( callee->isBindless() );
            std::vector<CanonicalProgramID> callees = callee->getCanonicalProgramIDs();
            csId->addOrRemovePotentialCallees( callees, added );
        }
    }
}

void BindingManager::addOrRemove_BufferBinding( VariableReferenceID refid, int bufid, bool added )
{
    bool changed  = m_bufferBindings.addOrRemoveProperty( refid, bufid, added );
    bool ichanged = m_ilwerseBufferBindings.addOrRemoveProperty( bufid, refid, added );
    RT_ASSERT( changed == ichanged );

    if( changed )
    {
        m_context->getUpdateManager()->eventBufferBindingsDidChange( refid, bufid, true );
        Buffer* buffer = m_context->getObjectManager()->getBufferById( bufid );
        buffer->bufferBindingsDidChange( refid, added );
    }
}

void BindingManager::addOrRemove_TextureBinding( VariableReferenceID refid, int texid, bool added )
{
    bool changed  = m_textureBindings.addOrRemoveProperty( refid, texid, added );
    bool ichanged = m_ilwerseTextureBindings.addOrRemoveProperty( texid, refid, added );
    RT_ASSERT( changed == ichanged );

    if( changed )
    {
        m_context->getUpdateManager()->eventTextureBindingsDidChange( refid, texid, true );
        TextureSampler* texture = m_context->getObjectManager()->getTextureSamplerById( texid );
        texture->textureBindingsDidChange();
    }
}

int BindingManager::getMaxTransformHeight() const
{
    if( m_transformHeight.empty() )
        return 0;

    return m_transformHeight.back();
}

void BindingManager::receivePropertyDidChange_TransformHeight( int height, bool added )
{
    // Note: transform height is lwrrently a global property, but it
    // would be straightforward to extend the system to compute the max
    // height at each GraphNode reference.
    int oldMax = getMaxTransformHeight();
    m_transformHeight.addOrRemoveProperty( height, added );
    int newMax = getMaxTransformHeight();

    if( oldMax != newMax )
        m_context->getUpdateManager()->eventContextMaxTransformDepthChanged( oldMax, newMax );
}

int BindingManager::getMaxAccelerationHeight() const
{
    if( m_accelerationHeight.empty() )
        return 0;

    return m_accelerationHeight.back();
}

void BindingManager::receivePropertyDidChange_AccelerationHeight( int height, bool added )
{
    int oldMax = getMaxAccelerationHeight();
    m_accelerationHeight.addOrRemoveProperty( height, added );
    int newMax = getMaxAccelerationHeight();

    if( oldMax != newMax )
        m_context->getUpdateManager()->eventContextMaxAccelerationHeightChanged( oldMax, newMax );
}

bool BindingManager::hasMotionTransforms() const
{
    const bool hasMotion = !m_hasMotionTransforms.empty();
    return hasMotion;
}

void BindingManager::receivePropertyDidChange_HasMotionTransforms( bool added )
{
    const bool changed = m_hasMotionTransforms.addOrRemoveProperty( added );
    if( changed )
        m_context->getUpdateManager()->eventContextHasMotionTransformsChanged( added );
}

void BindingManager::enqueueVirtualParentConnectOrDisconnect( Program* program, const ProgramRoot& root, bool connect )
{
    m_virtualChildren.addOrRemoveProperty( root, connect );
    if( root != m_ignoreRoot )
        modq.push( {program, root, connect} );
}

void BindingManager::processVirtualParentQueue()
{
    RT_ASSERT_MSG( !m_processingQueue, "Detected reentrant processing of virtual parent queue" );
    m_processingQueue = true;
    while( !modq.empty() )
    {
        Modification mod = modq.front();
        modq.pop();

        mod.program->connectOrDisconnectProperties_VirtualParent( mod.root, mod.connect );

        // Save the nodegraph for debugging
        m_context->saveNodeGraph();
    }

    m_processingQueue = false;
}

void BindingManager::forceDetach( const ProgramRoot& root )
{
    // Flush the queue
    processVirtualParentQueue();

    if( !m_virtualChildren.contains( root ) )
        return;

    // When detaching the outermost parent and removing the scope root
    // (virtualParent=false in
    // Program::attachOrDetach_UnresolvedReference), determine if all
    // virtual parent relationships have been removed for this
    // root. When there are self references from bound callable
    // programs, it is possible to have dangling references from the
    // reference counts. Clean those up now by forcing removal of all of
    // those children. It is brute force - looping through all programs
    // in the context. However, it should be rare. To enavle this, we
    // keep track of only the total number of virtual children to spare
    // us from maintaining that at all scopes. It may have also been
    // possible to keep a reference count or list of children in the
    // root, but since the programroot is passed by value this is not
    // possible.

    // Find all programs with the specified virtual parent and enqueue
    // the ilwoluntary disconnect for each phantom virtual parent. Note
    // that this is a REALLY big hammer...
    bool found = false;
    for( auto program : m_context->getObjectManager()->getPrograms() )
    {
        if( program->hasVirtualParent( root ) )
        {
            modq.push( {program, root, false} );
            found = true;
        }
    }
    RT_ASSERT_MSG( found, "Failed to find dangling virtual children" );

    // Tell enqueue to ignore subsequent removals from that root.
    m_ignoreRoot = root;
    processVirtualParentQueue();
    m_ignoreRoot = {-1, ST_ILWALID, ~0U};

    // Drop the virtual parents from the program and drop the virtual
    // children count.
    for( auto program : m_context->getObjectManager()->getPrograms() )
    {
        program->dropVirtualParents( root );
        int pcount = m_virtualChildren.count( root );
        for( int i = 0; i < pcount; ++i )
            m_virtualChildren.addOrRemoveProperty( root, false );
    }
}
