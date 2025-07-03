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

#include <Objects/Group.h>

#include <AS/ASManager.h>
#include <AS/Builder.h>
#include <Context/BindingManager.h>
#include <Context/ObjectManager.h>
#include <Context/RTCore.h>
#include <Context/SBTManager.h>
#include <Context/TableManager.h>
#include <Context/UpdateManager.h>
#include <Device/LWDADevice.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <Memory/MemoryManager.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GeometryTriangles.h>
#include <Objects/Selector.h>
#include <Objects/Transform.h>
#include <Objects/Variable.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/LinkedPtrHelpers.h>
#include <Util/MotionAabb.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/ValidationError.h>

using namespace optix;
using namespace prodlib;
using namespace corelib;

//------------------------------------------------------------------------
// CTOR/DTOR
//------------------------------------------------------------------------

AbstractGroup::AbstractGroup( Context* context, ObjectClass objClass )
    : GraphNode( context, objClass )
    , m_childrenBuffer( new Buffer( m_context, RT_BUFFER_INPUT ) )
{
    // Geometry groups need to record an AS depth of 1. To avoid calling
    // a virtual function in a constructor, we do not use
    // receivePropertyDidChange.
    m_accelerationHeight.addOrRemoveProperty( 1, true );
    m_context->getBindingManager()->receivePropertyDidChange_AccelerationHeight( 1, true );
    LexicalScope::reallocateRecord();

    // Declare the child object pointer buffer and mark it as bindless
    m_childrenBuffer->setFormat( RT_FORMAT_UNSIGNED_INT );
    m_childrenBuffer->markAsBindlessForInternalUse();
}

AbstractGroup::~AbstractGroup()
{
    // Children need to be removed in ~Group/~GeometryGroup, because the methods used
    // (specifically isGeometryGroup()) need to know whether it is a GeometryGroup or Group.
    // By the time it gets to ~AbstractGroup we've already destroyed whether it was one or
    // the other.
    RT_ASSERT_MSG( m_children.empty(), "Children must be removed in child class before ~AbstractGroup" );
    AbstractGroup::setAcceleration( nullptr );
    setVisitProgram( nullptr );
    setBoundingBoxProgram( nullptr );
    deleteVariables();

    // Remove the depth property
    m_accelerationHeight.addOrRemoveProperty( 1, false );
    m_context->getBindingManager()->receivePropertyDidChange_AccelerationHeight( 1, false );

    getContext()->getASManager()->addOrRemoveDirtyGroup( this, false );
}

//------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------

void AbstractGroup::setAcceleration( Acceleration* acceleration )
{
    // Acceleration properties
    //
    // Attachment:               propagates from parent
    // Acceleration height:      propagates from parent
    // Unresolved references:    propagates from parent

    Acceleration* oldAcceleration = m_acceleration.get();
    if( oldAcceleration )
    {
        // Remove properties
        this->attachOrDetachProperty_HasMotionAabbs( oldAcceleration, false );
        this->attachOrDetachProperty_UnresolvedReference( oldAcceleration, false );
        this->attachOrDetachProperty_AccelerationHeight( oldAcceleration, false );
        this->attachOrDetachProperty_Attachment( oldAcceleration, false );
    }

    m_acceleration.set( this, acceleration );

    Acceleration* newAcceleration = m_acceleration.get();
    if( newAcceleration )
    {
        // Add properties
        this->attachOrDetachProperty_Attachment( newAcceleration, true );
        this->attachOrDetachProperty_AccelerationHeight( newAcceleration, true );
        this->attachOrDetachProperty_UnresolvedReference( newAcceleration, true );
        this->attachOrDetachProperty_HasMotionAabbs( newAcceleration, true );
    }

    // Update the baking state after the linked pointers are updated
    if( oldAcceleration )
        oldAcceleration->updateBakingState();
    if( newAcceleration )
        newAcceleration->updateBakingState();

    // Update the AS with the kind of leaf (group or geometry group)
    if( m_acceleration )
        m_acceleration->setLeafKind( getClass() );

    // Update the visit program and bounding box program
    if( m_acceleration )
    {
        if( Program* visit = m_acceleration->getLwrrentVisitProgram() )
            setVisitProgram( visit );
        else
            setVisitProgram( getSharedNullProgram() );
        if( Program* bounds = m_acceleration->getLwrrentBoundingBoxProgram() )
            setBoundingBoxProgram( bounds );
        else
            setBoundingBoxProgram( getSharedNullProgram() );
    }

    subscribeForValidation();
    writeRecord();

    // This could be called from the destructor with null Acceleration, and can cause
    // problems with the virtual method updateTraversable.
    if( m_acceleration )
        updateTraversable();
}

Acceleration* AbstractGroup::getAcceleration() const
{
    if( !m_acceleration )
        return nullptr;  // It can't be an error to ask for the acceleration just because it's not set yet. http://lwbugs/948423

    return m_acceleration.get();
}

LexicalScope* AbstractGroup::getChild( unsigned int index ) const
{
    if( index >= getChildCount() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Child index out of range" );

    if( !m_children[index] )
        return nullptr;  // It can't be an error to ask for the child pointer just because it's not set yet.

    return m_children[index].get();
}

void Group::updateRtCoreData( unsigned int index )
{
    // Update the traversable for the child
    updateChildTraversable( index );
}

void AbstractGroup::setChild( unsigned int index, LexicalScope* child )
{
    // Note: the API entry function ensures that the child is
    // appropriate for this group.
    if( index >= getChildCount() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Child index out of range" );

    // Detect relwrsion in the graph
    if( child )
        checkForRelwrsiveGroup( managedObjectCast<GraphNode>( child ) );

    // Group child properties
    //
    // Direct caller:             originates from visit program to child
    // Trace Caller:              propagates from parent (must be attached before / removed after attachment)
    // Attachment:                propagates from parent
    // Transform height:          propagates from child (only for GraphNode children)
    // Acceleration height:       propagates from child (only for GraphNode children)
    // Instance transform:        propagates from child (only for Transform children in RTX data model)

    bool previouslySet{false};
    if( LexicalScope* oldChild = m_children[index].get() )
    {
        this->attachOrDetachChild( oldChild, &m_children[index], false );

        // Avoid cycles while propagating attachment (see lwbugswb #2422313)
        m_children[index].set( this, nullptr );
        this->attachOrDetachProperty_Attachment( oldChild, false );
        m_children[index].set( this, oldChild );  // probably unnecessary

        this->attachOrDetachProperty_TraceCaller( oldChild, false );
        if( getVisitProgram() && getVisitProgram() != getSharedNullProgram() )
        {
            getVisitProgram()->attachOrDetachProperty_DirectCaller( oldChild, false );
        }
        previouslySet = true;
    }

    m_children[index].set( this, child );

    if( LexicalScope* newChild = m_children[index].get() )
    {
        if( getVisitProgram() && getVisitProgram() != getSharedNullProgram() )
        {
            getVisitProgram()->attachOrDetachProperty_DirectCaller( newChild, true );
        }
        this->attachOrDetachProperty_TraceCaller( newChild, true );
        this->attachOrDetachProperty_Attachment( newChild, true );
        this->attachOrDetachChild( newChild, &m_children[index], true );
        if( GraphNode* gn = managedObjectCast<GraphNode>( newChild ) )
        {
            gn->attachTraversableHandle( this, &m_children[index] );
        }
        if( isGeometryGroup() )
        {
            ++numSetChildren;
        }
    }

    if( child != nullptr )
    {
        if( isGeometryGroup() )
        {
            if( previouslySet )
            {
                updateRtCoreData( index );
            }
            else if( numSetChildren >= getChildCount() )
            {
                reallocateInstances();
                for( unsigned int i = 0; i < getChildCount(); ++i )
                {
                    updateRtCoreData( i );
                }
            }
        }
        else
        {
            updateRtCoreData( index );
            writeRecord();
        }
    }

    subscribeForValidation();
    getContext()->getASManager()->addOrRemoveDirtyGroup( this, true );
}

void AbstractGroup::setChildCount( unsigned int newCount )
{
    m_childrenBuffer->setSize1D( newCount );

    // Resize the vector after the instance descriptors are ready
    resizeVector( m_children, newCount, [this]( int index ) { setChild( index, nullptr ); },
                  [this]( int index ) { setChild( index, nullptr ); } );

    // Validate and update
    reallocateInstances();
    subscribeForValidation();
    writeRecord();
}

void AbstractGroup::setVisibilityMask( RTvisibilitymask mask )
{
    if( mask > 0xffu )
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "Visibility mask is limited to 8 bits" );
    }

    if( mask != m_visibilityMask )
    {
        m_visibilityMask = mask;
        sendPropertyDidChange_VisibilityMaskInstanceFlags( m_visibilityMask, m_flags );
    }
}

void AbstractGroup::setFlags( RTinstanceflags flags )
{
    if( flags != m_flags )
    {
        m_flags = flags;
        sendPropertyDidChange_VisibilityMaskInstanceFlags( m_visibilityMask, m_flags );
    }
}

//------------------------------------------------------------------------
// Internal API
//------------------------------------------------------------------------


ObjectClass AbstractGroup::getChildType( unsigned int index ) const
{
    if( index >= getChildCount() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Child index out of range" );

    if( !m_children[index] )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Null child at index ", index );

    return m_children[index]->getClass();
}

void AbstractGroup::fillChildren()
{
    // Fill the child object pointer buffer

    const unsigned int nchildren = getChildCount();
    m_childrenBuffer->setSize1D( nchildren );

    unsigned int* offsets = static_cast<unsigned int*>( m_childrenBuffer->map( MAP_WRITE_DISCARD ) );
    for( unsigned int i = 0; i < nchildren; i++ )
    {
        const LexicalScope* child = getChild<LexicalScope>( i );
        offsets[i]                = static_cast<unsigned int>( getSafeOffset( child ) );
    }
    m_childrenBuffer->unmap();
}


//------------------------------------------------------------------------
// LinkedPtr relationship management
//------------------------------------------------------------------------

void AbstractGroup::detachLinkedChild( const LinkedPtr_Link* link )
{
    unsigned int index;
    if( getElementIndex( m_children, link, index ) )
        setChild( index, nullptr );

    else if( link == &m_acceleration )
        setAcceleration( nullptr );

    else
        detachLinkedProgram( link );
}

void AbstractGroup::childOffsetDidChange( const LinkedPtr_Link* link )
{
    LexicalScope::childOffsetDidChange( link );

    unsigned int index;
    if( getElementIndex( m_children, link, index ) )
    {
        if( isGeometryGroup() && m_SBTIndex )
            m_context->getSBTManager()->geometryInstanceOffsetDidChange( this, index );

        getContext()->getASManager()->addOrRemoveDirtyGroup( this, true );

        // Also notify the AS that the child offset changed. If baking
        // is enabled, then the AS may need to be rebuilt.
        if( m_acceleration )
            m_acceleration->groupsChildOffsetDidChange();
    }
}


//------------------------------------------------------------------------
// Object record management
//------------------------------------------------------------------------

size_t AbstractGroup::getRecordBaseSize() const
{
    return sizeof( cort::AbstractGroupRecord );
}

void AbstractGroup::writeRecord() const
{
    if( !recordIsAllocated() )
        return;
    cort::AbstractGroupRecord* g = getObjectRecord<cort::AbstractGroupRecord>();
    RT_ASSERT( g != nullptr );
    g->accel    = getSafeOffset( m_acceleration.get() );
    g->children = m_childrenBuffer->getId();

    GraphNode::writeRecord();
}

void AbstractGroup::offsetDidChange() const
{
    notifyParents_offsetDidChange();
}


//------------------------------------------------------------------------
// SBTRecord management
//------------------------------------------------------------------------
void AbstractGroup::rayTypeCountDidChange()
{
    reallocateInstances();
}

void AbstractGroup::materialCountDidChange( const LinkedPtr_Link* giLink )
{
    reallocateInstances();
}

void AbstractGroup::geometryDidChange( const LinkedPtr_Link* giLink ) const
{
    if( !m_context->useRtxDataModel() || !m_SBTIndex )
        return;
    const GeometryGroup* gg;
    unsigned int         giIndex;
    getGGandIndexFromLink( gg, giIndex, giLink );
    m_context->getSBTManager()->geometryDidChange( gg, giIndex );
}

void AbstractGroup::geometryIntersectionDidChange( const LinkedPtr_Link* giLink ) const
{
    if( !m_context->useRtxDataModel() || !m_SBTIndex )
        return;
    const GeometryGroup* gg;
    unsigned int         giIndex;
    getGGandIndexFromLink( gg, giIndex, giLink );
    m_context->getSBTManager()->intersectionProgramDidChange( gg, giIndex );
}

void AbstractGroup::materialDidChange( const LinkedPtr_Link* giLink, unsigned int materialIndex ) const
{
    if( !m_context->useRtxDataModel() || !m_SBTIndex )
        return;
    const GeometryGroup* gg;
    unsigned int         giIndex;
    getGGandIndexFromLink( gg, giIndex, giLink );
    m_context->getSBTManager()->materialDidChange( gg, giIndex, materialIndex );
}

void AbstractGroup::materialOffsetDidChange( const LinkedPtr_Link* giLink, unsigned int materialIndex ) const
{
    if( !m_context->useRtxDataModel() || !m_SBTIndex )
        return;
    const GeometryGroup* gg;
    unsigned int         giIndex;
    getGGandIndexFromLink( gg, giIndex, giLink );
    m_context->getSBTManager()->materialOffsetDidChange( gg, giIndex, materialIndex );
}

void AbstractGroup::materialClosestHitProgramDidChange( const LinkedPtr_Link* giLink, unsigned int materialIndex, unsigned int rayTypeIndex ) const
{
    if( !m_context->useRtxDataModel() || !m_SBTIndex )
        return;
    const GeometryGroup* gg;
    unsigned int         giIndex;
    getGGandIndexFromLink( gg, giIndex, giLink );
    m_context->getSBTManager()->closestHitProgramDidChange( gg, giIndex, materialIndex, rayTypeIndex );
}

void AbstractGroup::materialAnyHitProgramDidChange( const LinkedPtr_Link* giLink, unsigned int materialIndex, unsigned int rayTypeIndex ) const
{
    if( !m_context->useRtxDataModel() || !m_SBTIndex )
        return;
    const GeometryGroup* gg;
    unsigned int         giIndex;
    getGGandIndexFromLink( gg, giIndex, giLink );
    m_context->getSBTManager()->anyHitProgramDidChange( gg, giIndex, materialIndex, rayTypeIndex );
}

unsigned int AbstractGroup::getSBTRecordIndex() const
{
    RT_ASSERT_MSG( m_SBTIndex, "No SBT index allocated for Group" );
    return *m_SBTIndex;
}

unsigned int AbstractGroup::getChildIndexFromLink( const LinkedPtr_Link* link ) const
{
    unsigned int index;
    bool         found = getElementIndex( m_children, link, index );
    RT_ASSERT_MSG( found, "Didn't find child index from link" );
    return index;
}

void AbstractGroup::getGGandIndexFromLink( const GeometryGroup*& gg, unsigned int& index, const LinkedPtr_Link* link ) const
{
    gg = managedObjectCast<const GeometryGroup>( this );
    RT_ASSERT_MSG( gg, "anyHitProgramDidChange called on a Group instead of GeometryGroup" );

    index = getChildIndexFromLink( link );
}

void AbstractGroup::reallocateInstances()
{
    // Do nothing.
}


//------------------------------------------------------------------------
// Unresolved reference property
//------------------------------------------------------------------------

static inline void computeReferenceResolutionLogic( bool bIN, bool bV, bool& bR, bool& bOUT )
{
    /*
   * IN = union(visit.out, BPV.OUT)          // Counting set.
   * R = intersect(IN, V)                    // Resolution change
   * OUT = IN - V                            // Notify Context
   */

    bR   = bIN && bV;
    bOUT = bIN && !bV;
}

void AbstractGroup::attachOrDetachProperty_UnresolvedReference( Acceleration* acceleration, bool attached ) const
{
    scopeTrace( "begin attachOrDetachProperty_UnresolvedReference", ~0, attached, acceleration );

    // Callwlate output set
    for( auto refid : m_unresolvedSet )
    {
        bool bIN = true;
        bool bV  = haveVariableForReference( refid );
        // Callwlate derived set
        bool bR, bOUT;
        computeReferenceResolutionLogic( bIN, bV, bR, bOUT );
        if( bOUT )
            acceleration->receivePropertyDidChange_UnresolvedReference( this, refid, attached );
    }

    scopeTrace( "end attachOrDetachProperty_UnresolvedReference", ~0, attached, acceleration );
}

void AbstractGroup::sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool added ) const
{
    // Acceleration is the only scope parent of group
    if( m_acceleration )
        m_acceleration->receivePropertyDidChange_UnresolvedReference( this, refid, added );
}


//------------------------------------------------------------------------
// Attachment
//------------------------------------------------------------------------

void AbstractGroup::sendPropertyDidChange_Attachment( bool added ) const
{
    GraphNode::sendPropertyDidChange_Attachment( added );

    if( m_acceleration )
        m_acceleration->receivePropertyDidChange_Attachment( added );

    for( const auto& child : m_children )
        if( child )
            child->receivePropertyDidChange_Attachment( added );
}

//------------------------------------------------------------------------
// RtxUniversalTraversal
//------------------------------------------------------------------------

void AbstractGroup::sendPropertyDidChange_RtxUniversalTraversal() const
{
    if( m_acceleration )
        m_acceleration->receivePropertyDidChange_RtxUniversalTraversal();

    for( const auto& child : m_children )
        if( child )
            child->receivePropertyDidChange_RtxUniversalTraversal();
}


//------------------------------------------------------------------------
// Direct Caller
//------------------------------------------------------------------------
void AbstractGroup::attachOrDetachProperty_DirectCaller_toChildren( Program* program, bool added ) const
{
    if( !program || program == getSharedNullProgram() )
        return;
    for( const auto& child : m_children )
        if( child )
            program->attachOrDetachProperty_DirectCaller( child.get(), added );
}


//------------------------------------------------------------------------
// Trace Caller
//------------------------------------------------------------------------
void AbstractGroup::sendPropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added ) const
{
    for( const auto& child : m_children )
    {
        if( !child )
            continue;

        if( GraphNode* gn = managedObjectCast<GraphNode>( child.get() ) )
            gn->receivePropertyDidChange_TraceCaller( cpid, added );
        else if( GeometryInstance* gi = managedObjectCast<GeometryInstance>( child.get() ) )
            gi->receivePropertyDidChange_TraceCaller( cpid, added );
        else
            RT_ASSERT_FAIL_MSG( "Illegal child type" );
    }
}

//------------------------------------------------------------------------
// InstanceTransform property
//------------------------------------------------------------------------

void Group::instanceTransformDidChange( LinkedPtr_Link* parentLink, Matrix4x4 transform )
{
    unsigned int index;
    if( getElementIndex( m_children, parentLink, index ) )
    {
        const float*                               data = transform.getData();
        InstanceDescriptorHost::DeviceIndependent* desc = m_instanceDescriptors.mapDeviceIndependentTable( index );
        std::copy( &data[0], &data[12], desc->transform );
    }
    else
    {
        throw AssertionFailure( RT_EXCEPTION_INFO, "Child for link not found" );
    }
}


//------------------------------------------------------------------------
// Acceleration height property
//------------------------------------------------------------------------

void AbstractGroup::sendPropertyDidChange_AccelerationHeight( int height, bool added ) const
{
    GraphNode::sendPropertyDidChange_AccelerationHeight( height, added );
    if( m_acceleration )
        m_acceleration->receivePropertyDidChange_AccelerationHeight( height, added );
}

//------------------------------------------------------------------------
// HasMotionAabbs property
//------------------------------------------------------------------------

void AbstractGroup::sendPropertyDidChange_HasMotionAabbs( bool added ) const
{
    GraphNode::sendPropertyDidChange_HasMotionAabbs( added );
    if( m_acceleration )
        m_acceleration->receivePropertyDidChange_HasMotionAabbs( added );
}

//------------------------------------------------------------------------
// Geometry flags and visibility mask properties
//------------------------------------------------------------------------

bool AbstractGroup::getVisibilityMaskInstanceFlags( RTvisibilitymask& mask, RTinstanceflags& flags ) const
{
    mask  = m_visibilityMask;
    flags = m_flags;
    return true;
}

void AbstractGroup::attachOrDetachProperty_VisibilityMaskInstanceFlags( GraphNode* parent, LinkedPtr_Link* child )
{
    parent->receivePropertyDidChange_VisibilityMaskInstanceFlags( child, m_visibilityMask, m_flags );
}

//------------------------------------------------------------------------
// Group subclass methods
//------------------------------------------------------------------------

Group::Group( Context* context )
    : AbstractGroup( context, RT_OBJECT_GROUP )
    , m_instanceDescriptors( context )
{
    receivePropertyDidChange_RequiresTraversable( nullptr, true );
}

Group::~Group()
{
    receivePropertyDidChange_RequiresTraversable( nullptr, false );

    Group::setChildCount( 0 );
}

void Group::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    // parent class
    GraphNode::validate();

    const unsigned int numChildren = getChildCount();
    for( unsigned int i = 0; i < numChildren; ++i )
    {
        try
        {
            getChild( i );
        }
        catch( const prodlib::Exception& )
        {
            throw ValidationError( RT_EXCEPTION_INFO, "Group has null child" );
        }
    }

    // check that acceleration structure is present
    Acceleration* accel = getAcceleration();
    if( !accel )
        throw ValidationError( RT_EXCEPTION_INFO, "Group does not have an Acceleration Structure" );
}

void Group::setChildCount( unsigned int newCount )
{
    if( m_context->useRtxDataModel() )
    {
        const unsigned int oldCapacity = m_instanceDescriptors.capacity();
        if( m_instanceDescriptors.resize( newCount ) )
        {
            const unsigned int                         newCapacity = m_instanceDescriptors.capacity();
            InstanceDescriptorHost::DeviceIndependent* di =
                m_instanceDescriptors.mapDeviceIndependentTable( oldCapacity, newCapacity - 1 );
            std::fill( &di[0], &di[newCapacity - oldCapacity], InstanceDescriptorHost::DeviceIndependent{} );
        }
    }
    AbstractGroup::setChildCount( newCount );
}


//------------------------------------------------------------------------
// InstanceDescriptor support
//------------------------------------------------------------------------
InstanceDescriptorHost::DeviceIndependent Group::getInstanceDescriptor( unsigned int child ) const
{
    InstanceDescriptorHost::DeviceIndependent di = *m_instanceDescriptors.mapDeviceIndependentTable( child );
    m_instanceDescriptors.unmapDeviceIndependentTable();
    return di;
}

void Group::setInstanceDescriptor( unsigned int child, const InstanceDescriptorHost::DeviceIndependent& descriptor )
{
    *m_instanceDescriptors.mapDeviceIndependentTable( child ) = descriptor;
    m_instanceDescriptors.unmapDeviceIndependentTable();
}

void Group::setInstanceDescriptorDeviceDependent( unsigned int                                   child,
                                                  unsigned int                                   allDevicesIndex,
                                                  const InstanceDescriptorHost::DeviceDependent& descriptor )
{
    *m_instanceDescriptors.mapDeviceDependentTable( allDevicesIndex, child ) = descriptor;
    m_instanceDescriptors.unmapDeviceDependentTable( allDevicesIndex );
}

InstanceDescriptorHost::DeviceDependent Group::getInstanceDescriptorDeviceDependent( unsigned int child, unsigned int allDevicesIndex ) const
{
    InstanceDescriptorHost::DeviceDependent dd = *m_instanceDescriptors.mapDeviceDependentTable( allDevicesIndex, child );
    m_instanceDescriptors.unmapDeviceDependentTable( allDevicesIndex );
    return dd;
}

RtcInstance* Group::getInstanceDescriptorDevicePtr( Device* device ) const
{
    return reinterpret_cast<RtcInstance*>( m_instanceDescriptors.getInterleavedTableDevicePtr( device->allDeviceListIndex() ) );
}

const char* Group::getInstanceDescriptorTableDevicePtr( unsigned int allDevicesIndex ) const
{
    return reinterpret_cast<char*>( m_instanceDescriptors.getInterleavedTableDevicePtr( allDevicesIndex ) );
}

void Group::syncInstanceDescriptors()
{
    m_instanceDescriptors.sync();
}


//------------------------------------------------------------------------
// InstanceTransform property
//------------------------------------------------------------------------

void Group::receiveProperty_InstanceTransform( LinkedPtr_Link* child, Matrix4x4 transform )
{
    instanceTransformDidChange( child, transform );
}

GraphNode::TraversableDataForTest Group::getTraversableDataForTest( unsigned int allDeviceIndex ) const
{
    TraversableDataForTest result{};
    result.m_type          = RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL;
    result.m_traversableId = m_traversableId ? *m_traversableId : -1;
    return result;
}

//------------------------------------------------------------------------
// Visibility Mask and Instance Flags
//------------------------------------------------------------------------

void Group::receivePropertyDidChange_VisibilityMaskInstanceFlags( LinkedPtr_Link* parentLink, RTvisibilitymask mask, RTinstanceflags flags )
{
    if( m_context->useRtxDataModel() )
    {
        unsigned int index;
        if( getElementIndex( m_children, parentLink, index ) )
        {
            RT_ASSERT( index < m_instanceDescriptors.size() );
            InstanceDescriptorHost::DeviceIndependent& desc = *m_instanceDescriptors.mapDeviceIndependentTable( index );
            desc.flags                                      = flags;
            desc.mask                                       = mask;
        }
        else
        {
            throw AssertionFailure( RT_EXCEPTION_INFO, "Child for link not found" );
        }
    }
}

//------------------------------------------------------------------------
// Traversable support
//------------------------------------------------------------------------
RtcTraversableHandle Group::getTraversableHandle( unsigned int allDeviceIndex ) const
{
    return m_acceleration ? m_acceleration->getTraversableHandle( allDeviceIndex ) : 0U;
}

void Group::updateTraversable()
{
    // Nothing to update since the traversable comes from the Acceleration
}

void Group::checkForRelwrsiveGroup( GraphNode* child )
{
    checkForRelwrsiveGraph( child );
}

void Group::attachOrDetachChild( LexicalScope* child, LinkedPtr_Link* childLink, bool attached )
{
    GraphNode* gn = managedObjectCast<GraphNode>( child );
    gn->attachOrDetachProperty_HasMotionAabbs( this, attached );
    gn->attachOrDetachProperty_TransformHeight( this, attached );
    gn->attachOrDetachProperty_AccelerationHeight( this, attached );
    if( attached )
    {
        gn->attachOrDetachProperty_InstanceTransform( this, childLink );
        gn->attachOrDetachProperty_SBTIndex( this, childLink );
        gn->attachOrDetachProperty_VisibilityMaskInstanceFlags( this, childLink );
    }
}

void Group::childTraversableHandleDidChange( LinkedPtr_Link* child, TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle travHandle )
{
    if( !m_context->useRtxDataModel() )
        return;

    if( source == TravSource::GEOMGROUP_ACCEL || source == TravSource::OTHER_DIRECT )
    {
        // Write the instance descriptor
        const unsigned int childIndex = getChildIndexFromLink( child );
        setInstanceDescriptorDeviceDependent( childIndex, allDeviceIndex, InstanceDescriptorHost::DeviceDependent{travHandle} );

        // Unfortunately, there is no way to update the group. We must issue a rebuild.
        if( m_acceleration )
            m_acceleration->markDirty( true );
    }
}

void Group::childSBTIndexDidChange( LinkedPtr_Link* parentLink, unsigned int oldSBTIndex, unsigned int newSBTIndex )
{
    if( !m_context->useRtxDataModel() )
        return;

    unsigned int index;
    if( getElementIndex( m_children, parentLink, index ) )
    {
        m_instanceDescriptors.mapDeviceIndependentTable( index )->instanceOffset = newSBTIndex;
    }

    if( m_acceleration )
        m_acceleration->markDirty( true );
}

void Group::asTraversableHandleDidChange( RtcTraversableHandle travHandle, unsigned int allDeviceIndex )
{
    sendDidChange_TraversableHandle( TravSource::OTHER_DIRECT, allDeviceIndex, travHandle );
}

void Group::updateChildTraversable( unsigned int childIndex )
{
    if( !m_context->useRtxDataModel() )
        return;

    // Find the appropriate traversable by skipping simple transforms
    // that were folded into the leaf.
    GraphNode* gn = nullptr;
    if( LexicalScope* child = getChild( childIndex ) )
    {
        gn = managedObjectCast<GraphNode>( child );
        while( gn && gn->getClass() == RT_OBJECT_TRANSFORM )
        {
            Transform* xform = managedObjectCast<Transform>( gn );
            if( xform->hasMotionAabbs() )
            {
                // Use motion transform traversable directly
                break;
            }
            gn = xform->getChild();
        }
    }

    for( LWDADevice* dev : LWDADeviceArrayView( m_context->getDeviceManager()->activeDevices() ) )
    {
        const unsigned int   allDeviceListIndex = dev->allDeviceListIndex();
        RtcTraversableHandle travHandle         = 0;

        // If the node is a geometry group, use the traversable from
        // the AS. Otherwise, use it directly.
        if( gn != nullptr )
        {
            if( gn->getClass() == RT_OBJECT_GEOMETRY_GROUP )
            {
                if( Acceleration* accel = managedObjectCast<GeometryGroup>( gn )->getAcceleration() )
                {
                    travHandle = accel->getTraversableHandle( allDeviceListIndex );
                }
            }
            else
            {
                travHandle = gn->getTraversableHandle( allDeviceListIndex );
            }
        }

        setInstanceDescriptorDeviceDependent( childIndex, allDeviceListIndex, InstanceDescriptorHost::DeviceDependent{travHandle} );
    }
}


//------------------------------------------------------------------------
// GeometryGroup subclass methods
//------------------------------------------------------------------------

GeometryGroup::GeometryGroup( Context* context )
    : AbstractGroup( context, RT_OBJECT_GEOMETRY_GROUP )
{
    if( m_context->useRtxDataModel() )
    {
        const size_t numDevices = context->getDeviceManager()->allDevices().size();
        m_asTraversables.resize( numDevices );

        if( !m_context->RtxUniversalTraversalEnabled() )
        {
            m_topLevelInstances.resize( numDevices );
            m_topLevelTempBuffers.resize( numDevices );

            for( Device* dev : m_context->getDeviceManager()->activeDevices() )
            {
                LWDADevice* lwdaDev = deviceCast<LWDADevice>( dev );
                if( !lwdaDev )
                    continue;

                const unsigned int allDeviceListIndex   = dev->allDeviceListIndex();
                m_topLevelInstances[allDeviceListIndex] = createTopLevelInstanceBuffer( dev );
            }
        }
    }
}

GeometryGroup::~GeometryGroup()
{
    AbstractGroup::setChildCount( 0 );
    for( auto& traversable : m_traversables )
        if( traversable )
            traversable->removeListener( this );
}

MBufferHandle GeometryGroup::createTopLevelInstanceBuffer( Device* device )
{
    static const BufferDimensions g_instanceDims( RT_FORMAT_BYTE, 1, 1, sizeof( RtcInstance ), 1, 1 );
    return m_context->getMemoryManager()->allocateMBuffer( g_instanceDims, MBufferPolicy::gpuLocal, DeviceSet( device ) );
}

static RtcAccelType accelTypeForBuilder( const std::string& builderName )
{
    if( builderName == "NoAccel" )
        return RTC_ACCEL_TYPE_NOACCEL;
    if( builderName == "Bvh8" )
        return RTC_ACCEL_TYPE_BVH8;
    if( builderName == "TTU" )
        return RTC_ACCEL_TYPE_TTU;
    return RTC_ACCEL_TYPE_BVH2;
}

void GeometryGroup::updateTraversable()
{
    if( !m_context->useRtxDataModel() )
        return;

    const bool     requiresTraversable = !m_requiresTraversable.empty();
    MemoryManager* mm                  = m_context->getMemoryManager();
    for( LWDADevice* lwdaDev : LWDADeviceArrayView( m_context->getDeviceManager()->activeDevices() ) )
    {
        const unsigned int allDeviceListIndex = lwdaDev->allDeviceListIndex();
        if( requiresTraversable )
        {
            if( !m_traversables[allDeviceListIndex] )
            {
                DeviceSet set( lwdaDev );
                if( m_context->RtxUniversalTraversalEnabled() )
                {
                    BufferDimensions size( RT_FORMAT_BYTE, 1, 1, sizeof( RtcTravBottomLevelInstance ), 1, 1 );
                    m_traversables[allDeviceListIndex] = mm->allocateMBuffer( size, MBufferPolicy::gpuLocal, set, this );
                }
                else if( m_acceleration )
                {
                    RtcAccelOptions options{};
                    options.accelType                  = accelTypeForBuilder( m_acceleration->getBuilderType() );
                    options.buildFlags                 = 0;
                    options.refit                      = false;
                    options.usePrimBits                = true;
                    options.useRemapForPrimBits        = false;
                    options.bakeTriangles              = false;
                    options.highPrecisionMath          = true;
                    options.useProvizBuilderStrategies = true;
                    options.motionSteps                = 1;
                    options.motionTimeBegin            = 0;
                    options.motionTimeEnd              = 0;
                    options.motionFlags                = 0;
                    options.useUniversalFormat         = false;

                    RtcBuildInput buildInputs{};
                    buildInputs.type                       = RTC_BUILD_INPUT_TYPE_INSTANCE_ARRAY;
                    buildInputs.instanceArray.numInstances = 1;

                    lwdaDev->enable();
                    RtcDeviceContext devContext = lwdaDev->rtcContext();

                    RtcAccelBufferSizes bufferSizes;
                    m_acceleration->getContext()->getRTCore()->accelComputeMemoryUsage( devContext, &options, 1, &buildInputs,
                                                                                        nullptr, &bufferSizes );
                    BufferDimensions size( RT_FORMAT_BYTE, 1, 1, bufferSizes.outputSizeInBytes, 1, 1 );
                    m_traversables[allDeviceListIndex] = mm->allocateMBuffer( size, MBufferPolicy::gpuLocal, set, this );
                    BufferDimensions tempSize( RT_FORMAT_BYTE, 1, 1, bufferSizes.tempSizeInBytes, 1, 1 );
                    m_topLevelTempBuffers[allDeviceListIndex] = mm->allocateMBuffer( tempSize, MBufferPolicy::gpuLocal, set );

                    // The dummy top-level AS build is only triggered when the bottom level AS is rebuild.
                    // If the AS is shared with another GeometryGroup it may have already been build so we need to
                    // mark the acceleration as dirty to force the dummy top-level AS build.
                    m_acceleration->markDirty( true );
                }
            }

            if( m_SBTIndex )
            {
                const RtcTraversableHandle asTraversable =
                    m_acceleration ? m_acceleration->getTraversableHandle( allDeviceListIndex ) : 0;
                writeTraversable( asTraversable, allDeviceListIndex );
            }
        }
        else
        {
            if( m_traversables[allDeviceListIndex] )
            {
                m_traversables[allDeviceListIndex].reset();
            }
        }
    }
}

GraphNode::TravSource GeometryGroup::getTraversableSource() const
{
    return TravSource::GEOMGROUP_DIRECT;
}

RtcTraversableHandle GeometryGroup::getTraversableHandle( unsigned int allDeviceIndex ) const
{
    const LWDADevice* lwdaDevice = deviceCast<const LWDADevice>( m_context->getDeviceManager()->allDevices()[allDeviceIndex] );
    if( !lwdaDevice )
        return 0;

    if( !m_traversables[allDeviceIndex] )
        return 0;

    // Get the pointer from the memory manager
    MAccess access = m_traversables[allDeviceIndex]->getAccess( lwdaDevice );
    if( access.getKind() != MAccess::LINEAR )
        return 0;

    // Colwert the pointer to a traversable handle using rtcore
    RtcDeviceContext     devContext = lwdaDevice->rtcContext();
    RtcTraversableType   travtype   = RTC_TRAVERSABLE_TYPE_BOTTOM_LEVEL_INSTANCE;
    RtcAccelType         acceltype  = RTC_ACCEL_TYPE_NOACCEL;  // Unused here
    RtcTraversableHandle travHandle = 0;
    RtcGpuVA             devPtr     = (RtcGpuVA)access.getLinearPtr();
    if( m_context->RtxUniversalTraversalEnabled() )
        m_context->getRTCore()->colwertPointerToTraversableHandle( devContext, devPtr, travtype, acceltype, &travHandle );
    else
        travHandle = devPtr;
    return travHandle;
}

#define STATIC_ASSERT( a, b ) static_assert( (unsigned)a == (unsigned)b, "" );
STATIC_ASSERT( RTC_INSTANCE_FLAG_NONE, RT_INSTANCE_FLAG_NONE );
STATIC_ASSERT( RTC_INSTANCE_FLAG_TRIANGLE_LWLL_DISABLE, RT_INSTANCE_FLAG_DISABLE_TRIANGLE_LWLLING );
STATIC_ASSERT( RTC_INSTANCE_FLAG_TRIANGLE_LWLL_FLIP_WINDING, RT_INSTANCE_FLAG_FLIP_TRIANGLE_FACING );
STATIC_ASSERT( RTC_INSTANCE_FLAG_FORCE_OPAQUE, RT_INSTANCE_FLAG_DISABLE_ANYHIT );
STATIC_ASSERT( RTC_INSTANCE_FLAG_FORCE_NO_OPAQUE, RT_INSTANCE_FLAG_FORCE_ANYHIT );
#undef STATIC_ASSERT

void GeometryGroup::writeTopLevelTraversable( const cort::Aabb* aabbDevicePtr, LWDADevice* lwdaDevice )
{
    if( m_context->RtxUniversalTraversalEnabled() )
        return;
    const unsigned int allDeviceIndex = lwdaDevice->allDeviceListIndex();
    if( !m_traversables[allDeviceIndex] )
        return;
    if( !m_SBTIndex )
        return;

    // cribbed from RtcBvh::build( const BuildParameters& params, const BuildSetup& setup, const GroupData& groupData )
    MemoryManager* mm          = m_context->getMemoryManager();
    RtcCommandList commandList = lwdaDevice->primaryRtcCommandList();

    RtcAccelBuffers buffers;
    buffers.input = static_cast<RtcGpuVA>( 0 );
    buffers.output =
        reinterpret_cast<RtcGpuVA>( mm->getWritablePointerForBuild( m_traversables[allDeviceIndex], lwdaDevice, true ) );
    buffers.outputSizeInBytes = m_traversables[allDeviceIndex]->getDimensions().getTotalSizeInBytes();
    buffers.temp =
        reinterpret_cast<RtcGpuVA>( mm->getWritablePointerForBuild( m_topLevelTempBuffers[allDeviceIndex], lwdaDevice, true ) );
    buffers.tempSizeInBytes = m_topLevelTempBuffers[allDeviceIndex]->getDimensions().getTotalSizeInBytes();

    RtcAccelOptions options            = {};
    options.accelType                  = accelTypeForBuilder( m_acceleration->getBuilderType() );
    options.buildFlags                 = 0;
    options.refit                      = false;
    options.usePrimBits                = true;
    options.useRemapForPrimBits        = false;
    options.enableBuildReordering      = false;
    options.clampAabbsToValidRange     = true;
    options.bakeTriangles              = false;
    options.highPrecisionMath          = true;
    options.useProvizBuilderStrategies = true;
    options.motionSteps                = 1;
    options.motionTimeBegin            = 0;
    options.motionTimeEnd              = 0;
    options.motionFlags                = 0;
    options.useUniversalFormat         = false;

    RtcBuildInput buildInput = {};
    buildInput.type = RTC_BUILD_INPUT_TYPE_INSTANCE_ARRAY;

    RtcInstance* instance =
        reinterpret_cast<RtcInstance*>( mm->mapToHost( m_topLevelInstances[allDeviceIndex], MAP_WRITE_DISCARD ) );
    static const float transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    algorithm::copy( transform, &instance->transform[0] );
    instance->instanceId               = 0U;
    instance->mask                     = 0xff;
    instance->instanceOffset           = *m_SBTIndex;
    instance->flags                    = 0u;
    instance->accelOrTraversableHandle = m_asTraversables[allDeviceIndex];
    mm->unmapFromHost( m_topLevelInstances[allDeviceIndex] );
    mm->manualSynchronize( m_topLevelInstances[allDeviceIndex] );

    RtcBuildInputInstanceArray& instances = buildInput.instanceArray;
    instances.instanceDescs = (RtcGpuVA)mm->getWritablePointerForBuild( m_topLevelInstances[allDeviceIndex], lwdaDevice, true );
    instances.numInstances = 1;

    m_context->getRTCore()->accelBuild( commandList, &options, 1, &buildInput, nullptr, &buffers, 0, nullptr );
}

void GeometryGroup::writeTraversable( RtcTraversableHandle asTraversable, unsigned int allDeviceIndex )
{
    if( asTraversable == 0U )
        return;

    m_asTraversables[allDeviceIndex] = asTraversable;

    if( m_context->RtxUniversalTraversalEnabled() )
    {
        MemoryManager* mm     = m_context->getMemoryManager();
        char*          bufPtr = mm->mapToHost( m_traversables[allDeviceIndex], MAP_WRITE_DISCARD );

        RtcTravBottomLevelInstance* bottomLevelInstance = reinterpret_cast<RtcTravBottomLevelInstance*>( bufPtr );
        bottomLevelInstance->sbtOffset                  = m_SBTIndex ? *m_SBTIndex : 0U;
        bottomLevelInstance->accel                      = m_asTraversables[allDeviceIndex];
        mm->unmapFromHost( m_traversables[allDeviceIndex] );
    }
}

void GeometryGroup::asTraversableHandleDidChange( RtcTraversableHandle travHandle, unsigned int allDeviceIndex )
{
    if( m_traversables[allDeviceIndex] )
    {
        writeTraversable( travHandle, allDeviceIndex );
    }

    sendDidChange_TraversableHandle( TravSource::GEOMGROUP_ACCEL, allDeviceIndex, travHandle );
}

void GeometryGroup::attachTraversableHandle( GraphNode* parent, LinkedPtr_Link* child )
{
    if( !m_context->useRtxDataModel() )
    {
        return;
    }

    if( m_acceleration )
    {
        for( Device* dev : m_context->getDeviceManager()->activeDevices() )
        {
            int allDeviceIndex = dev->allDeviceListIndex();
            parent->childTraversableHandleDidChange( child, TravSource::GEOMGROUP_ACCEL, allDeviceIndex,
                                                     m_acceleration->getTraversableHandle( allDeviceIndex ) );
        }
    }

    GraphNode::attachTraversableHandle( parent, child );
}

GraphNode::TraversableDataForTest GeometryGroup::getTraversableDataForTest( unsigned int allDeviceIndex ) const
{
    TraversableDataForTest result{};
    result.m_type                = RTC_TRAVERSABLE_TYPE_BOTTOM_LEVEL_INSTANCE;
    result.m_size                = sizeof( RtcTravBottomLevelInstance );
    result.m_bottomLevelInstance = reinterpret_cast<RtcTravBottomLevelInstance*>(
        m_context->getMemoryManager()->mapToHost( m_traversables[allDeviceIndex], MAP_READ ) );
    result.m_traversableId = m_traversableId ? *m_traversableId : -1;
    return result;
}

void GeometryGroup::eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA )
{
    const LWDADevice* lwdaDevice = deviceCast<const LWDADevice>( device );
    if( !lwdaDevice )
        return;

    // The GPU pointer to our dummy traversable changed. Form new
    // traversable handle.
    RtcTraversableHandle travHandle = 0;
    if( newMBA.getKind() == MAccess::LINEAR )
    {
        RtcGpuVA devPtr = (RtcGpuVA)newMBA.getLinearPtr();
        if( m_context->RtxUniversalTraversalEnabled() )
        {
            RtcDeviceContext   devContext = lwdaDevice->rtcContext();
            RtcTraversableType travtype   = RTC_TRAVERSABLE_TYPE_BOTTOM_LEVEL_INSTANCE;
            RtcAccelType       acceltype  = RTC_ACCEL_TYPE_NOACCEL;  // Unused here
            m_context->getRTCore()->colwertPointerToTraversableHandle( devContext, devPtr, travtype, acceltype, &travHandle );
        }
        else
        {
            travHandle = devPtr;
        }
    }

    // Notify selector, variable and transform parents of the
    // traversable change. Group parents are not notified.
    unsigned int allDeviceIndex = device->allDeviceListIndex();
    sendDidChange_TraversableHandle( TravSource::GEOMGROUP_DIRECT, allDeviceIndex, travHandle );
}

void GeometryGroup::rtxUniversalTraversalDidChange()
{
    // universal requires a BLAS traversable while non-universal requires a dummy top-level traversable.
    // release and rebuild traversable.

    for( auto& traversable : m_traversables )
        traversable.reset();

    updateTraversable();
}

void GeometryGroup::updateChildTraversable( unsigned int childIndex )
{
    // Not relevant for GeometryGroup
}

void GeometryGroup::checkForRelwrsiveGroup( GraphNode* child )
{
    // Do nothing for GeometryGroup
}

void GeometryGroup::attachOrDetachChild( LexicalScope* child, LinkedPtr_Link* childLink, bool attached )
{
    GeometryInstance* gi = managedObjectCast<GeometryInstance>( child );
    gi->attachOrDetachProperty_HasMotionAabbs( managedObjectCast<GeometryGroup>( this ), attached );
}

void GeometryGroup::updateRtCoreData( unsigned int index )
{
    // Update GeometryGroup information
    if( m_SBTIndex )
    {
        // This notification that the GeometryInstance changed might be superfluous if we just reallocated the GG
        m_context->getSBTManager()->geometryInstanceDidChange( managedObjectCast<GeometryGroup>( this ), index );
    }
}

void GeometryGroup::reallocateInstances()
{
    if( !m_context->useRtxDataModel() )
        return;

    const bool         haveOldSBTIndex = static_cast<bool>( m_SBTIndex );
    const unsigned int oldSBTIndex     = haveOldSBTIndex ? *m_SBTIndex : 0U;

    GeometryGroup* gg  = managedObjectCast<GeometryGroup>( this );
    SBTManager*    sbt = m_context->getSBTManager();

    m_SBTIndex.reset();
    m_SBTIndex = sbt->allocateInstances( gg, m_context->getRayTypeCount() );  // allocates and registers
    if( m_SBTIndex && ( !haveOldSBTIndex || oldSBTIndex != *m_SBTIndex ) )
    {
        updateTraversable();
        notifyParents_SBTIndexDidChange( oldSBTIndex, *m_SBTIndex );
    }
}

void GeometryGroup::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    // parent class
    GraphNode::validate();

    const unsigned int numChildren               = getChildCount();
    bool               geometryTrianglesChildren = false;
    for( unsigned int i = 0; i < numChildren; ++i )
    {
        GeometryInstance* gi;
        try
        {
            gi = managedObjectCast<GeometryInstance>( getChild( i ) );
        }
        catch( const prodlib::Exception& )
        {
            throw ValidationError( RT_EXCEPTION_INFO, "GeometryGroup has null child" );
        }
        GeometryTriangles* geomTris = managedObjectCast<GeometryTriangles>( gi->getGeometry() );
        if( i == 0 )
        {
            geometryTrianglesChildren = geomTris != nullptr;
        }
        else if( ( geometryTrianglesChildren && geomTris == nullptr ) || ( !geometryTrianglesChildren && geomTris != nullptr ) )
        {
            throw ValidationError( RT_EXCEPTION_INFO,
                                   "GeometryGroup does not support mixing Geometry and GeometryTriangles children" );
        }
    }

    // check that acceleration structure is present
    Acceleration* accel = getAcceleration();
    if( !accel )
        throw ValidationError( RT_EXCEPTION_INFO, "GeometryGroup does not have an Acceleration Structure" );
}

void GeometryGroup::preSetActiveDevices( const DeviceSet& removedDeviceSet )
{
    for( int allDeviceListIndex : removedDeviceSet )
    {
        if( m_traversables.size() > allDeviceListIndex )
            m_traversables[allDeviceListIndex].reset();
        if( m_topLevelInstances.size() > allDeviceListIndex )
            m_topLevelInstances[allDeviceListIndex].reset();
        if( m_topLevelTempBuffers.size() > allDeviceListIndex )
            m_topLevelTempBuffers[allDeviceListIndex].reset();
    }
}

void GeometryGroup::postSetActiveDevices()
{
    if( m_context->useRtxDataModel() && !m_context->RtxUniversalTraversalEnabled() )
    {
        for( LWDADevice* dev : LWDADeviceArrayView( m_context->getDeviceManager()->activeDevices() ) )
        {
            const unsigned int allDeviceListIndex = dev->allDeviceListIndex();
            if( !m_topLevelInstances[allDeviceListIndex] )
                m_topLevelInstances[allDeviceListIndex] = createTopLevelInstanceBuffer( dev );
        }
    }

    updateTraversable();
}
