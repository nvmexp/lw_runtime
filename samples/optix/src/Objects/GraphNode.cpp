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
#include <Context/TableManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Memory/MemoryManager.h>
#include <Objects/Acceleration.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GlobalScope.h>
#include <Objects/GraphNode.h>
#include <Objects/Group.h>
#include <Objects/Program.h>
#include <Objects/ProgramRoot.h>
#include <Objects/Selector.h>
#include <Objects/Transform.h>
#include <Objects/Variable.h>
#include <Util/LinkedPtrHelpers.h>
#include <prodlib/exceptions/ValidationError.h>

using namespace optix;
using namespace prodlib;

//------------------------------------------------------------------------
// CTOR / DTOR
//------------------------------------------------------------------------

GraphNode::GraphNode( Context* context, ObjectClass objClass )
    : LexicalScope( context, objClass )
{
    Program* null_program = getSharedNullProgram();
    setVisitProgram( null_program );
    setBoundingBoxProgram( null_program );
    m_traversables.resize( context->getDeviceManager()->allDevices().size() );
}

GraphNode::~GraphNode()
{
    RT_ASSERT_MSG( m_transformHeight.empty(), "GraphNode destroyed while transform height properties remain" );
    RT_ASSERT_MSG( m_accelerationHeight.empty(), "GraphNode destroyed while acceleration height properties remain" );

    setVisitProgram( nullptr );
    setBoundingBoxProgram( nullptr );
}


//------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------

void GraphNode::setVisitProgram( Program* program )
{
    // Validate the semantic type before setting the program to avoid
    // exceptions during graph property propagation
    if( program )
        program->validateSemanticType( ST_NODE_VISIT );

    // Visit program properties
    //
    // Semantic type:            originates here
    // Attachment:               propagates from parent
    // Direct caller:            propagates from parent
    // Direct caller:            propagates from program to children
    // Unresolved references:    propagates from program
    // Note: Only selectors will use bound callable programs

    ProgramRoot root( getScopeID(), ST_NODE_VISIT, 0 );

    if( Program* oldProgram = m_visitProgram.get() )
    {
        // Remove properties from old program
        oldProgram->attachOrDetachProperty_UnresolvedReference( this, root, false );
        attachOrDetachProperty_DirectCaller_toChildren( oldProgram, false );
        if( oldProgram && oldProgram != getSharedNullProgram() )
        {
            this->attachOrDetachProperty_DirectCaller( oldProgram, false );
            oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_NODE_VISIT, false );
        }
        this->attachOrDetachProperty_Attachment( oldProgram, false );
    }

    m_visitProgram.set( this, program );

    if( Program* newProgram = m_visitProgram.get() )
    {
        // Add properties to new program
        this->attachOrDetachProperty_Attachment( newProgram, true );
        if( newProgram && newProgram != getSharedNullProgram() )
        {
            newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_NODE_VISIT, true );
            this->attachOrDetachProperty_DirectCaller( newProgram, true );
        }
        attachOrDetachProperty_DirectCaller_toChildren( newProgram, true );
        newProgram->attachOrDetachProperty_UnresolvedReference( this, root, true );
    }

    subscribeForValidation();
    writeRecord();
}

Program* GraphNode::getVisitProgram() const
{
    return m_visitProgram.get();
}

void GraphNode::setBoundingBoxProgram( Program* program )
{
    // Validate the semantic type before setting the program to avoid
    // exceptions during graph property propagation
    if( program )
        program->validateSemanticType( ST_BOUNDING_BOX );

    // Bounding box program properties
    //
    // Semantic type:            originates here
    // Attachment:               propagates from parent
    // Unresolved references:    propagates from program

    ProgramRoot root( getScopeID(), ST_BOUNDING_BOX, 0 );

    if( Program* oldProgram = m_boundingBoxProgram.get() )
    {
        // Remove properties from old program
        oldProgram->attachOrDetachProperty_UnresolvedReference( this, root, false );
        if( oldProgram && oldProgram != getSharedNullProgram() )
            oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_BOUNDING_BOX, false );
        this->attachOrDetachProperty_Attachment( oldProgram, false );
    }

    m_boundingBoxProgram.set( this, program );

    if( Program* newProgram = m_boundingBoxProgram.get() )
    {
        // Add properties to new program
        this->attachOrDetachProperty_Attachment( newProgram, true );
        if( newProgram && newProgram != getSharedNullProgram() )
            newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_BOUNDING_BOX, true );
        newProgram->attachOrDetachProperty_UnresolvedReference( this, root, true );
    }

    subscribeForValidation();
    writeRecord();
}

Program* GraphNode::getBoundingBoxProgram() const
{
    return m_boundingBoxProgram.get();
}

void GraphNode::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    LexicalScope::validate();

    // Verify that we have a visit program
    if( getBoundingBoxProgram() == getSharedNullProgram() )
    {
        throw ValidationError( RT_EXCEPTION_INFO, "GraphNode does not have an internal bounding box program" );
    }
}


//------------------------------------------------------------------------
// LinkedPtr relationship management
//------------------------------------------------------------------------

void GraphNode::detachFromParents()
{
    auto iter = m_linkedPointers.begin();
    while( iter != m_linkedPointers.end() )
    {
        LinkedPtr_Link* parentLink = *iter;

        if( GraphNode* parentNode = getLinkToGraphNodeFrom<Selector, Transform>( parentLink ) )
        {
            parentNode->detachLinkedChild( parentLink );
        }
        else if( LexicalScope* parentScope = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
        {
            RT_ASSERT_MSG( parentScope->getClass() == RT_OBJECT_GROUP, "Invalid object discovered in graphnode" );
            static_cast<GraphNode*>( parentScope )->detachLinkedChild( parentLink );
        }
        else if( Variable* variable = getLinkToGraphNodeFrom<Variable>( parentLink ) )
        {
            variable->detachLinkedChild( parentLink );
        }
        else
        {
            RT_ASSERT_FAIL_MSG( "Invalid parent link to GraphNode" );
        }

        iter = m_linkedPointers.begin();
    }
}

void GraphNode::detachLinkedProgram( const LinkedPtr_Link* link )
{
    if( link == &m_visitProgram )
        setVisitProgram( getSharedNullProgram() );

    else if( link == &m_boundingBoxProgram )
        setBoundingBoxProgram( getSharedNullProgram() );

    else
        RT_ASSERT_FAIL_MSG( "Invalid child link in detachLinkedChild" );
}


//------------------------------------------------------------------------
// Object record management
//------------------------------------------------------------------------

void GraphNode::writeRecord() const
{
    if( !recordIsAllocated() )
        return;
    cort::GraphNodeRecord* gn = getObjectRecord<cort::GraphNodeRecord>();
    RT_ASSERT( gn != nullptr );
    gn->traverse = getSafeOffset( m_visitProgram.get() );
    gn->bounds   = getSafeOffset( m_boundingBoxProgram.get() );

    gn->traversableId = m_traversableId ? *m_traversableId : -1;

    LexicalScope::writeRecord();
}

void GraphNode::notifyParents_offsetDidChange() const
{
    for( auto parentLink : m_linkedPointers )
    {
        if( GraphNode* parentNode = getLinkToGraphNodeFrom<Selector, Transform>( parentLink ) )
        {
            parentNode->childOffsetDidChange( parentLink );
        }
        else if( LexicalScope* parentScope = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
        {
            RT_ASSERT_MSG( parentScope->getClass() == RT_OBJECT_GROUP, "Invalid object discovered in graphnode" );
            static_cast<GraphNode*>( parentScope )->childOffsetDidChange( parentLink );
        }
        else if( Variable* variable = getLinkToGraphNodeFrom<Variable>( parentLink ) )
        {
            variable->graphNodeOffsetDidChange();
        }
        else
        {
            RT_ASSERT_FAIL_MSG( std::string( "Invalid parent link to GraphNode: " ) + typeid( *parentLink ).name() );
        }
    }
}


//------------------------------------------------------------------------
// Attachment
//------------------------------------------------------------------------
void GraphNode::sendPropertyDidChange_Attachment( bool added ) const
{
    if( getVisitProgram() )
        getVisitProgram()->receivePropertyDidChange_Attachment( added );

    if( getBoundingBoxProgram() )
        getBoundingBoxProgram()->receivePropertyDidChange_Attachment( added );
}

//------------------------------------------------------------------------
// RtxUniversalTraversal
//------------------------------------------------------------------------

bool GraphNode::rtxTraversableNeedUniversalTraversal() const
{
    // motion blur and multi-level hierarchies are only supported with universal traversal
    return hasMotionAabbs() || ( getMaxAccelerationHeight() > 2 );
}

//------------------------------------------------------------------------
// Direct Caller
//------------------------------------------------------------------------

void GraphNode::attachOrDetachProperty_DirectCaller_toChildren( Program* program, bool added ) const
{
    // See note in header
}

void GraphNode::sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const
{
    if( m_visitProgram && m_visitProgram != getSharedNullProgram() )
        m_visitProgram->receivePropertyDidChange_DirectCaller( cpid, added );
}

void GraphNode::attachOrDetachProperty_DirectCaller( Program* program, bool added ) const
{
    // Propagate current scope's direct caller to the program
    for( auto cpid : m_directCaller )
        program->receivePropertyDidChange_DirectCaller( cpid, added );
}


//------------------------------------------------------------------------
// Trace Caller
//------------------------------------------------------------------------

void GraphNode::attachOrDetachProperty_TraceCaller( LexicalScope* atScope, bool added ) const
{
    if( GeometryInstance* gi = managedObjectCast<GeometryInstance>( atScope ) )
    {
        // GI children
        for( auto cpid : m_traceCaller )
            gi->receivePropertyDidChange_TraceCaller( cpid, added );
    }
    else if( GraphNode* gn = managedObjectCast<GraphNode>( atScope ) )
    {
        for( auto cpid : m_traceCaller )
            gn->receivePropertyDidChange_TraceCaller( cpid, added );
    }
    else
    {
        RT_ASSERT_FAIL_MSG(
            std::string( "GraphNode::attachOrDetachProperty_TraceCaller called for non GI or GraphNode: " )
            + typeid( *atScope ).name() );
    }
}

void GraphNode::receivePropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added )
{
    bool changed = m_traceCaller.addOrRemoveProperty( cpid, added );
    if( changed )
        sendPropertyDidChange_TraceCaller( cpid, added );
}


//------------------------------------------------------------------------
// Acceleration height property
//------------------------------------------------------------------------

void GraphNode::receivePropertyDidChange_AccelerationHeight( int height, bool added )
{
    // Offset the height before storing in the set. +1 for
    // Groups/GeometryGroups, +0 for others.
    height += accelerationHeightOffset();

    // Add/remove the property
    int oldMax = getMaxAccelerationHeight();
    m_accelerationHeight.addOrRemoveProperty( height, added );
    int newMax = getMaxAccelerationHeight();

    // Remove old max and add new one. For efficiency, add the new
    // property before removing the old one. In case of perf problems,
    // consider propagating both min and max at once.
    if( oldMax != newMax )
    {
        if( newMax != -1 )
            sendPropertyDidChange_AccelerationHeight( newMax, true );
        if( oldMax != -1 )
            sendPropertyDidChange_AccelerationHeight( oldMax, false );
    }
}

void GraphNode::sendPropertyDidChange_AccelerationHeight( int height, bool added ) const
{
    // Parents can be other Variables and other GraphNodes but
    // AbstractGroup creates the link as LexicalScope.
    for( auto parentLink : m_linkedPointers )
    {
        if( GraphNode* parentNode = getLinkToGraphNodeFrom<Selector, Transform>( parentLink ) )
        {
            parentNode->receivePropertyDidChange_AccelerationHeight( height, added );
        }
        else if( LexicalScope* parentScope = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
        {
            RT_ASSERT_MSG( parentScope->getClass() == RT_OBJECT_GROUP, "Invalid object discovered in graphnode" );
            static_cast<GraphNode*>( parentScope )->receivePropertyDidChange_AccelerationHeight( height, added );
        }
        else
        {
            RT_ASSERT_MSG( getLinkToGraphNodeFrom<Variable>( parentLink ) != nullptr,
                           std::string( "Unexpected linked pointer type in GraphNode: " ) + typeid( *parentLink ).name() );
        }
    }

    // In addition there is a global property stored in the binding
    // manager.
    m_context->getBindingManager()->receivePropertyDidChange_AccelerationHeight( height, added );
}

int GraphNode::getMaxAccelerationHeight() const
{
    if( m_accelerationHeight.empty() )
        return -1;  // No GeometryGroups have attached to this node

    return m_accelerationHeight.back();
}

void GraphNode::attachOrDetachProperty_AccelerationHeight( GraphNode* child, bool attached ) const
{
    if( m_accelerationHeight.empty() )
        return;  // Undefined height

    unsigned int height = getMaxAccelerationHeight();
    child->receivePropertyDidChange_AccelerationHeight( height, attached );
}

void GraphNode::attachOrDetachProperty_AccelerationHeight( Acceleration* child, bool attached ) const
{
    RT_ASSERT_MSG( managedObjectCast<const AbstractGroup>( this ), "Acceleration set on a non-AbstractGroup" );
    RT_ASSERT_MSG( !m_accelerationHeight.empty(),
                   "AbstrctGroup objects must have at least one acceleration height (1)" );

    unsigned int height = getMaxAccelerationHeight();
    child->receivePropertyDidChange_AccelerationHeight( height, attached );
}


//------------------------------------------------------------------------
// Transform height property
//------------------------------------------------------------------------

int GraphNode::getMaxTransformHeight() const
{
    int height;
    if( m_transformHeight.empty() )
        height = 0;
    else
        height = m_transformHeight.back();

    height += transformHeightOffset();
    return height;
}

void GraphNode::attachOrDetachProperty_TransformHeight( GraphNode* child, bool attached ) const
{
    unsigned int height = getMaxTransformHeight();
    if( height != 0 )
        child->receivePropertyDidChange_TransformHeight( height, attached );
}

void GraphNode::receivePropertyDidChange_TransformHeight( int height, bool added )
{
    // Add/remove the property
    int oldMax = getMaxTransformHeight();
    m_transformHeight.addOrRemoveProperty( height, added );
    int newMax = getMaxTransformHeight();

    // Remove old max and add new one. For efficiency, add the new
    // property before removing the old one. In case of perf problems,
    // consider propagating both min and max at once.
    if( oldMax != newMax )
    {
        if( newMax != 0 )
            sendPropertyDidChange_TransformHeight( newMax, true );
        if( oldMax != 0 )
            sendPropertyDidChange_TransformHeight( oldMax, false );
    }
}

void GraphNode::sendPropertyDidChange_TransformHeight( int height, bool added ) const
{
    // Parents can be other Variables and other GraphNodes but
    // AbstractGroup creates the link as LexicalScope.
    for( auto parentLink : m_linkedPointers )
    {
        if( GraphNode* parentNode = getLinkToGraphNodeFrom<Selector, Transform>( parentLink ) )
        {
            parentNode->receivePropertyDidChange_TransformHeight( height, added );
        }
        else if( LexicalScope* parentScope = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
        {
            RT_ASSERT_MSG( parentScope->getClass() == RT_OBJECT_GROUP, "Invalid object discovered in graphnode" );
            static_cast<GraphNode*>( parentScope )->receivePropertyDidChange_TransformHeight( height, added );
        }
        else
        {
            RT_ASSERT_MSG( getLinkToGraphNodeFrom<Variable>( parentLink ) != nullptr,
                           std::string( "Unexpected linked pointer type in GraphNode: " ) + typeid( *parentLink ).name() );
        }
    }

    // In addition there is a global property stored in the binding
    // manager.
    m_context->getBindingManager()->receivePropertyDidChange_TransformHeight( height, added );
}

//------------------------------------------------------------------------
// HasMotionAabbs property
//------------------------------------------------------------------------

bool GraphNode::hasMotionAabbs() const
{
    const bool hasMotion = ( !m_hasMotionAabbs.empty() ) || hasMotionKeys();
    return hasMotion;
}

void GraphNode::attachOrDetachProperty_HasMotionAabbs( GraphNode* listener, bool attached ) const
{
    const bool hasMotion = hasMotionAabbs();
    // The property is a boolean (GraphPropertySingle), so we only send 'true'.  Empty property means false.
    if( hasMotion )
        listener->receivePropertyDidChange_HasMotionAabbs( attached );
}

void GraphNode::attachOrDetachProperty_HasMotionAabbs( Acceleration* listener, bool attached ) const
{
    RT_ASSERT_MSG( managedObjectCast<const AbstractGroup>( this ), "Acceleration set on a non-AbstractGroup" );

    const bool hasMotion = hasMotionAabbs();
    if( hasMotion )
        listener->receivePropertyDidChange_HasMotionAabbs( attached );
}

void GraphNode::receivePropertyDidChange_HasMotionAabbs( bool added )
{
    const bool oldValue = hasMotionAabbs();
    m_hasMotionAabbs.addOrRemoveProperty( added );
    const bool newValue = hasMotionAabbs();

    if( oldValue != newValue )
    {
        // Add new value or remove old one.  Since this is a boolean, we only have to do one of these.
        sendPropertyDidChange_HasMotionAabbs( newValue );
    }
}

void GraphNode::sendPropertyDidChange_HasMotionAabbs( bool added ) const
{
    // Parents can be other Variables and other GraphNodes but
    // AbstractGroup creates the link as LexicalScope.
    for( auto parentLink : m_linkedPointers )
    {
        if( GraphNode* parentNode = getLinkToGraphNodeFrom<Selector, Transform>( parentLink ) )
        {
            parentNode->receivePropertyDidChange_HasMotionAabbs( added );
        }
        else if( LexicalScope* parentScope = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
        {
            RT_ASSERT_MSG( parentScope->getClass() == RT_OBJECT_GROUP, "Invalid object discovered in graphnode" );
            static_cast<GraphNode*>( parentScope )->receivePropertyDidChange_HasMotionAabbs( added );
        }
        else
        {
            RT_ASSERT_MSG( getLinkToGraphNodeFrom<Variable>( parentLink ) != nullptr,
                           std::string( "Unexpected linked pointer type in GraphNode: " ) + typeid( *parentLink ).name() );
        }
    }
}

//------------------------------------------------------------------------
// Geometry flags and visibility mask properties
//------------------------------------------------------------------------

bool GraphNode::getVisibilityMaskInstanceFlags( RTvisibilitymask&, RTinstanceflags& ) const
{
    return false;
}

void GraphNode::sendPropertyDidChange_VisibilityMaskInstanceFlags( GraphNode*       parentNode,
                                                                   LinkedPtr_Link*  parentLink,
                                                                   RTvisibilitymask mask,
                                                                   RTinstanceflags  flags ) const
{
    parentNode->receivePropertyDidChange_VisibilityMaskInstanceFlags( parentLink, mask, flags );
}

void GraphNode::sendPropertyDidChange_VisibilityMaskInstanceFlags( RTvisibilitymask mask, RTinstanceflags flags ) const
{
    for( auto parentLink : m_linkedPointers )
    {
        if( AbstractGroup* parentGroup = getLinkFrom<AbstractGroup, LexicalScope>( parentLink ) )
        {
            parentGroup->receivePropertyDidChange_VisibilityMaskInstanceFlags( parentLink, mask, flags );
        }
        else if( GraphNode* parentNode = getLinkFrom<Transform, GraphNode>( parentLink ) )
        {
            parentNode->receivePropertyDidChange_VisibilityMaskInstanceFlags( parentLink, mask, flags );
        }
    }
}

void GraphNode::receivePropertyDidChange_VisibilityMaskInstanceFlags( LinkedPtr_Link*, RTvisibilitymask mask, RTinstanceflags flags )
{
    sendPropertyDidChange_VisibilityMaskInstanceFlags( mask, flags );
}

void GraphNode::attachOrDetachProperty_VisibilityMaskInstanceFlags( GraphNode*, LinkedPtr_Link* )
{
}

//------------------------------------------------------------------------
// Attached to variable property
//------------------------------------------------------------------------
void GraphNode::receivePropertyDidChange_AttachedToVariable( bool attached )
{
    bool changed = m_attachedToVariable.addOrRemoveProperty( attached );
    if( changed )
    {
        // Create or destroy the traversable ID
        if( attached )
        {
            m_traversableId = m_context->getObjectManager()->registerTraversable( this );
            // Init traversable handle
            for( Device* dev : m_context->getDeviceManager()->activeDevices() )
            {
                const unsigned int   allDeviceIndex = dev->allDeviceListIndex();
                RtcTraversableHandle travHandle     = getTraversableHandle( allDeviceIndex );
                m_context->getTableManager()->writeTraversableHeader( *m_traversableId, travHandle, allDeviceIndex );
            }
        }
        else
        {
            m_traversableId.reset();
        }
        writeRecord();
    }
}

//------------------------------------------------------------------------
// Traversable support
//------------------------------------------------------------------------

void GraphNode::receivePropertyDidChange_RequiresTraversable( GraphNode* fromParent, bool added )
{
    bool changed = m_requiresTraversable.addOrRemoveProperty( added );
    if( changed )
        updateTraversable();
}

// InstanceTransform property
Matrix4x4 GraphNode::getInstanceTransform() const
{
    RT_ASSERT_MSG( m_context->useRtxDataModel(), "Instance Transform only makes sense in RTX exelwtion model" );
    return Matrix4x4::identity();
}

void GraphNode::attachOrDetachProperty_InstanceTransform( GraphNode* parent, LinkedPtr_Link* child )
{
    if( m_context->useRtxDataModel() )
    {
        // NOTE: child, when properly downcast, is the same as this.
        parent->receiveProperty_InstanceTransform( child, getInstanceTransform() );
    }
}


void GraphNode::receiveProperty_InstanceTransform( LinkedPtr_Link* child, Matrix4x4 transform )
{
    // Do nothing.
}

void GraphNode::instanceTransformDidChange( LinkedPtr_Link* parent, Matrix4x4 transform )
{
    // Do nothing.
}

void GraphNode::sendPropertyDidChange_InstanceTransform( Matrix4x4 instanceTransform )
{
    if( !m_context->useRtxDataModel() )
    {
        return;
    }

    for( auto parentLink : m_linkedPointers )
    {
        if( AbstractGroup* parentGroup = getLinkFrom<AbstractGroup, LexicalScope>( parentLink ) )
        {
            parentGroup->instanceTransformDidChange( parentLink, instanceTransform );
        }
        else if( GraphNode* parentNode = getLinkFrom<Transform, GraphNode>( parentLink ) )
        {
            parentNode->instanceTransformDidChange( parentLink, instanceTransform );
        }
    }
}


//------------------------------------------------------------------------
// SBTIndex property
//------------------------------------------------------------------------
unsigned int GraphNode::getSBTIndex() const
{
    return 0;
}

void GraphNode::notifyParents_SBTIndexDidChange( unsigned int oldSBTIndex, unsigned int newSBTIndex )
{
    for( auto parentLink : m_linkedPointers )
    {
        if( AbstractGroup* parentGroup = getLinkFrom<AbstractGroup, LexicalScope>( parentLink ) )
        {
            parentGroup->childSBTIndexDidChange( parentLink, oldSBTIndex, newSBTIndex );
        }
        else if( GraphNode* parentNode = getLinkFrom<Transform, GraphNode>( parentLink ) )
        {
            parentNode->childSBTIndexDidChange( parentLink, oldSBTIndex, newSBTIndex );
        }
    }
}

void GraphNode::childSBTIndexDidChange( LinkedPtr_Link* /*parentLink*/, unsigned int /*oldSBTIndex*/, unsigned int /*newSBTIndex*/ )
{
    // Do nothing.
}

void GraphNode::attachOrDetachProperty_SBTIndex( GraphNode* parent, LinkedPtr_Link* child )
{
    if( m_context->useRtxDataModel() )
    {
        // NOTE: child, when properly downcast, is the same as this.
        parent->childSBTIndexDidChange( child, 0U, getSBTIndex() );
    }
}


//------------------------------------------------------------------------
// Traversable properties
//------------------------------------------------------------------------
GraphNode::TravSource GraphNode::getTraversableSource() const
{
    return TravSource::OTHER_DIRECT;
}

void GraphNode::childTraversableHandleDidChange( LinkedPtr_Link* /*parentLink*/,
                                                 TravSource /*source*/,
                                                 unsigned int /*allDevicesListIndex*/,
                                                 RtcTraversableHandle /*newHandle*/ )
{
    throw prodlib::AssertionFailure( RT_EXCEPTION_INFO, "Attempt to call childTraversableHandleDidChange" );
}

void GraphNode::attachTraversableHandle( GraphNode* parent, LinkedPtr_Link* child )
{
    if( !m_context->useRtxDataModel() )
        return;

    // NOTE: child, when properly downcast, is the same as this.
    if( !m_requiresTraversable.empty() )
    {
        for( Device* dev : m_context->getDeviceManager()->activeDevices() )
        {
            const unsigned int allDeviceIndex = dev->allDeviceListIndex();
            parent->childTraversableHandleDidChange( child, getTraversableSource(), allDeviceIndex,
                                                     getTraversableHandle( allDeviceIndex ) );
        }
    }
}

GraphNode::TraversableDataForTest GraphNode::getTraversableDataForTest( unsigned int /*allDeviceIndex*/ ) const
{
    throw prodlib::AssertionFailure( RT_EXCEPTION_INFO,
                                     "Attempt to call getTraversableDataForTest for a node without a traversable" );
}

void GraphNode::releaseTraversableDataForTest( unsigned int allDeviceIndex )
{
    m_context->getMemoryManager()->unmapFromHost( m_traversables[allDeviceIndex] );
}

void GraphNode::receivePropertyDidChange_TransformRequiresTraversable( GraphNode* fromParent, bool attached )
{
    //// Nothing interesting to do on a change, just update the property.
    //m_transformRequiresTraversable.addOrRemoveProperty( attached );
    const bool changed = m_transformRequiresTraversable.addOrRemoveProperty( attached );
    if( changed )
    {
        receivePropertyDidChange_RequiresTraversable( fromParent, attached );
    }
}

// Notify selector, group, transform and variable parents of the traversable change.
void GraphNode::sendDidChange_TraversableHandle( TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle travHandle )
{
    for( auto parentLink : m_linkedPointers )
    {
        if( Selector* selector = getLinkToGraphNodeFrom<Selector>( parentLink ) )
        {
            selector->childTraversableHandleDidChange( parentLink, source, allDeviceIndex, travHandle );
        }
        else if( Transform* transform = getLinkToGraphNodeFrom<Transform>( parentLink ) )
        {
            transform->childTraversableHandleDidChange( parentLink, source, allDeviceIndex, travHandle );
        }
        else if( AbstractGroup* abstractGroup = getLinkFrom<AbstractGroup, LexicalScope>( parentLink ) )
        {
            RT_ASSERT( abstractGroup->getClass() == RT_OBJECT_GROUP );
            managedObjectCast<Group>( abstractGroup )->childTraversableHandleDidChange( parentLink, source, allDeviceIndex, travHandle );
        }
        else if( getLinkToGraphNodeFrom<Variable>( parentLink ) )
        {
            variableChildTraversableHandleDidChange( source, allDeviceIndex, travHandle );
        }
    }
}

void GraphNode::variableChildTraversableHandleDidChange( TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle travHandle )
{
    if( source == TravSource::GEOMGROUP_DIRECT || source == TravSource::TRANSFORM_DIRECT || source == TravSource::OTHER_DIRECT )
    {
        // the traversable handle may change while the graph node is being linked to a variable, before the traversableId has been registered.
        // this happens when the graph node is part of the scene graph previously connected to the variable.
        if( m_traversableId )
        {
            m_context->getTableManager()->writeTraversableHeader( *m_traversableId, travHandle, allDeviceIndex );
        }
    }
}

//------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------

void GraphNode::checkForRelwrsiveGraph( GraphNode* newChild ) const
{
    if( newChild == nullptr )
        // This can happen during tear down
        return;

    if( newChild->getClass() == RT_OBJECT_GEOMETRY_GROUP )
        // Geometry groups cannot be a parent to any node, so skip the search
        return;

    if( newChild == this )
        throw ValidationError( RT_EXCEPTION_INFO, "Self-reference detected in node graph" );

    if( this->hasAncestor( newChild ) )
        throw ValidationError( RT_EXCEPTION_INFO, "Cycle detected in node graph" );
}

bool GraphNode::hasAncestor( const GraphNode* node ) const
{
    if( this == node )
        return true;

    // Parents can be other Variables and other GraphNodes but
    // AbstractGroup creates the link as LexicalScope.
    for( auto parentLink : m_linkedPointers )
    {
        if( GraphNode* parentNode = getLinkToGraphNodeFrom<Selector, Transform>( parentLink ) )
        {
            if( parentNode->hasAncestor( node ) )
                return true;
        }
        else if( LexicalScope* parentScope = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
        {
            RT_ASSERT_MSG( parentScope->getClass() == RT_OBJECT_GROUP, "Invalid object discovered in graphnode" );
            if( static_cast<GraphNode*>( parentScope )->hasAncestor( node ) )
                return true;
        }
        else
        {
            RT_ASSERT_MSG( getLinkToGraphNodeFrom<Variable>( parentLink ) != nullptr,
                           std::string( "Unexpected linked pointer type in GraphNode: " ) + typeid( *parentLink ).name() );
        }
    }

    return false;
}
