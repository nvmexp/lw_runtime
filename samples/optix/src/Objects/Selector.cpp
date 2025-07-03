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

#include <Objects/Selector.h>

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/RTCore.h>
#include <Context/SharedProgramManager.h>
#include <Context/UpdateManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <Memory/MemoryManager.h>
#include <Objects/GlobalScope.h>
#include <Util/BufferUtil.h>
#include <Util/LinkedPtrHelpers.h>
#include <Util/MotionAabb.h>
#include <corelib/math/MathUtil.h>
#include <corelib/misc/Cast.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/ValidationError.h>

#include <llvm/IR/DerivedTypes.h>

#include <algorithm>

using namespace optix;
using namespace prodlib;
using namespace corelib;

//------------------------------------------------------------------------
// CTOR/DTOR
//------------------------------------------------------------------------

Selector::Selector( Context* context )
    : GraphNode( context, RT_OBJECT_SELECTOR )
{
    // Declare child index buffer
    m_childrenBuffer.reset( createBuffer<cort::ObjectRecordOffset>( context, 1, RT_BUFFER_INPUT ) );

    // We need to mark the buffer as bindless since we don't access the buffer by name, but only through its ID.
    m_childrenBuffer->markAsBindlessForInternalUse();

    // Compute the union of all children on the GPU
    Program* bounds = m_context->getSharedProgramManager()->getBoundsRuntimeProgram( "bounds_selector", false, false );
    setBoundingBoxProgram( bounds );

    reallocateRecord();

    receivePropertyDidChange_RequiresTraversable( nullptr, true );
}

Selector::~Selector()
{
    receivePropertyDidChange_RequiresTraversable( nullptr, false );

    setChildCount( 0 );
    setVisitProgram( nullptr );
    setBoundingBoxProgram( nullptr );
    deleteVariables();
    for( auto& traversable : m_traversables )
        if( traversable )
            traversable->removeListener( this );
}

//------------------------------------------------------------------------
// LinkedPtr relationship mangement
//------------------------------------------------------------------------

void Selector::setChildCount( unsigned int newCount )
{
    unsigned int oldCount = getChildCount();
    if( oldCount == newCount )
        return;

    resizeVector( m_children, newCount, [this]( int index ) { setChild( index, nullptr ); },
                  [this]( int index ) { setChild( index, nullptr ); } );

    // Write object record and validation the object
    writeRecord();
    subscribeForValidation();
    updateTraversable();
}

unsigned int Selector::getChildCount() const
{
    return static_cast<unsigned int>( m_children.size() );
}

void Selector::setChild( unsigned int index, GraphNode* child )
{
    if( index >= getChildCount() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Selector child index out of range" );
    checkForRelwrsiveGraph( child );

    // Selector child properties
    //
    // Direct caller:             originates from visit program to child
    // Trace Caller:              propagates from parent (must be attached before / removed after attachment)
    // Attachment:                propagates from parent
    // Transform height:          propagates from child
    // Acceleration height:       propagates from child
    // Has motion aabbs:          propagates from child
    // Requires traversable:      originates from parent

    if( GraphNode* oldChild = m_children[index].get() )
    {
        oldChild->receivePropertyDidChange_RequiresTraversable( this, false );
        oldChild->attachOrDetachProperty_HasMotionAabbs( this, false );
        oldChild->attachOrDetachProperty_AccelerationHeight( this, false );
        oldChild->attachOrDetachProperty_TransformHeight( this, false );

        // Avoid cycles while propagating attachment (see lwbugswb #2422313)
        m_children[index].set( this, nullptr );
        this->attachOrDetachProperty_Attachment( oldChild, false );
        m_children[index].set( this, oldChild );  // probably unnecessary

        this->attachOrDetachProperty_TraceCaller( oldChild, false );
        if( getVisitProgram() )
            getVisitProgram()->attachOrDetachProperty_DirectCaller( oldChild, false );
    }

    m_children[index].set( this, child );

    if( GraphNode* newChild = m_children[index].get() )
    {
        if( getVisitProgram() )
            getVisitProgram()->attachOrDetachProperty_DirectCaller( newChild, true );
        this->attachOrDetachProperty_TraceCaller( newChild, true );
        this->attachOrDetachProperty_Attachment( newChild, true );
        newChild->attachOrDetachProperty_TransformHeight( this, true );
        newChild->attachOrDetachProperty_AccelerationHeight( this, true );
        newChild->attachOrDetachProperty_HasMotionAabbs( this, true );
        newChild->receivePropertyDidChange_RequiresTraversable( this, true );
        newChild->attachTraversableHandle( this, &m_children[index] );
    }

    // TODO: Writing the record every time a child changes is wasteful.
    //       We need a way of delaying the record writing until it is needed.
    //       This could be implemented with a set somewhere (ObjectManager?), that tracks
    //       GraphNodes that need their object records written.
    subscribeForValidation();
    writeRecord();
}

GraphNode* Selector::getChild( unsigned int index ) const
{
    if( index >= getChildCount() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Selector child index out of range" );

    return m_children[index].get();
}


void Selector::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    // check parent class
    GraphNode::validate();

    // Verify that we have a visit program
    if( getVisitProgram() == getSharedNullProgram() )
    {
        throw ValidationError( RT_EXCEPTION_INFO, "Selector does not have a visit program" );
    }

    // Verify that each child is valid
    const unsigned int numChildren = getChildCount();
    for( unsigned int i = 0; i < numChildren; ++i )
    {
        try
        {
            getChild( i );
        }
        catch( const prodlib::Exception& )
        {
            throw ValidationError( RT_EXCEPTION_INFO, "Selector has null child" );
        }
    }
}


//------------------------------------------------------------------------
// Internal API
//------------------------------------------------------------------------

ObjectClass Selector::getChildType( unsigned int index ) const
{
    if( index >= getChildCount() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Selector child index out of range" );

    if( !m_children[index] )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Null selector child at index ", index );

    return m_children[index]->getClass();
}


//------------------------------------------------------------------------
// LinkedPtr relationship management
//------------------------------------------------------------------------

void Selector::detachLinkedChild( const LinkedPtr_Link* link )
{
    unsigned int index;
    if( getElementIndex( m_children, link, index ) )
        setChild( index, nullptr );

    else
        detachLinkedProgram( link );
}


//------------------------------------------------------------------------
// Object record access and management
//------------------------------------------------------------------------

void Selector::fillChildren() const
{
    // Resize buffer and get handle
    const unsigned int                     nchildren = getChildCount();
    MappedBuffer<cort::ObjectRecordOffset> childbuf( m_childrenBuffer.get(), MAP_WRITE_DISCARD, nchildren );
    // Copy child offsets
    for( unsigned int i = 0; i < nchildren; ++i )
    {
        if( m_children[i] == nullptr )
            continue;
        const LexicalScope* child = getChild<LexicalScope>( i );
        if( child->recordIsAllocated() )
            childbuf.ptr()[i] = static_cast<cort::ObjectRecordOffset>( child->getRecordOffset() );
    }
}

size_t Selector::getRecordBaseSize() const
{
    return sizeof( cort::SelectorRecord );
}

void Selector::writeRecord() const
{
    if( !recordIsAllocated() )
        return;
    cort::SelectorRecord* sel = getObjectRecord<cort::SelectorRecord>();
    RT_ASSERT( sel != nullptr );
    sel->children = m_childrenBuffer->getId();
    fillChildren();
    GraphNode::writeRecord();
}


//------------------------------------------------------------------------
// Unresolved reference property
//------------------------------------------------------------------------

static inline void computeReferenceResolutionLogic( bool bIN, bool bV, bool bA, bool& bR, bool& bOUT )
{
    /*
   * IN = union(visit.out, BPV.OUT)                    // Counting set.
   * R = intersect(IN, V)                              // Resolution change
   * OUT = intersect(IN - V, A)                        // Notify Context
   */

    bR   = bIN && bV;
    bOUT = bIN && !bV && bA;
}

void Selector::receivePropertyDidChange_UnresolvedReference( const LexicalScope* scope, VariableReferenceID refid, bool added )
{
    scopeTrace( "begin receivePropertyDidChange_UnresolvedReference", refid, added, scope );

    // Callwlate new/old input bits
    bool setChanged = m_unresolvedSet.addOrRemoveProperty( refid, added );
    bool old_IN     = !added || !setChanged;
    bool new_IN     = added || !setChanged;
    bool old_V      = haveVariableForReference( refid );
    bool new_V      = old_V;
    bool A          = isAttached();

    // Callwlate derived sets
    bool old_R, old_OUT;
    computeReferenceResolutionLogic( old_IN, old_V, A, old_R, old_OUT );
    bool new_R, new_OUT;
    computeReferenceResolutionLogic( new_IN, new_V, A, new_R, new_OUT );

    // Propagate changes if necessary
    if( old_OUT != new_OUT )
        sendPropertyDidChange_UnresolvedReference( refid, new_OUT );
    if( old_R != new_R )
        updateResolvedSet( refid, new_R );

    scopeTrace( "end receivePropertyDidChange_UnresolvedReference_preResolve", refid, added, scope );
}

void Selector::variableDeclarationDidChange( VariableReferenceID refid, bool variableWasAdded )
{
    scopeTrace( "begin variableDeclarationDidChange", refid, variableWasAdded );

    // Callwlate new/old input bits
    bool old_IN = m_unresolvedSet.contains( refid );
    bool new_IN = old_IN;
    bool old_V  = !variableWasAdded;
    bool new_V  = variableWasAdded;
    bool old_A  = isAttached();
    bool new_A  = old_A;

    // Callwlate derived sets
    bool old_R, old_OUT;
    computeReferenceResolutionLogic( old_IN, old_V, old_A, old_R, old_OUT );
    bool new_R, new_OUT;
    computeReferenceResolutionLogic( new_IN, new_V, new_A, new_R, new_OUT );

    // Propagate changes if necessary
    if( old_OUT != new_OUT )
        sendPropertyDidChange_UnresolvedReference( refid, new_OUT );
    if( old_R != new_R )
        updateResolvedSet( refid, new_R );

    scopeTrace( "end variableDeclarationDidChange", refid, variableWasAdded );
}

void Selector::sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool added ) const
{
    // Global scope is the only scope parent (implicitly)
    m_context->getGlobalScope()->receivePropertyDidChange_UnresolvedReference( this, refid, added );
}

void Selector::computeUnresolvedOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const
{
    for( auto refid : m_unresolvedSet )
    {
        bool bIN = true;
        bool bV  = haveVariableForReference( refid );
        bool bA  = isAttached();
        bool bR, bOUT;

        computeReferenceResolutionLogic( bIN, bV, bA, bR, bOUT );
        if( bOUT )
            out.addOrRemoveProperty( refid, true );
    }
}


//------------------------------------------------------------------------
// Attachment
//------------------------------------------------------------------------

void Selector::sendPropertyDidChange_Attachment( bool added ) const
{
    GraphNode::sendPropertyDidChange_Attachment( added );

    for( const auto& child : m_children )
        if( child )
            child->receivePropertyDidChange_Attachment( added );
}

void Selector::attachmentDidChange( bool new_A )
{
    bool old_A = !new_A;

    // Unresolved references are sensitive to attachment for Selector
    // and GI only.
    for( auto refid : m_unresolvedSet )
    {
        bool old_IN = true;
        bool new_IN = old_IN;
        bool old_V  = haveVariableForReference( refid );
        bool new_V  = old_V;

        // Callwlate output bit
        bool old_R, old_OUT;
        computeReferenceResolutionLogic( old_IN, old_V, old_A, old_R, old_OUT );
        bool new_R, new_OUT;
        computeReferenceResolutionLogic( new_IN, new_V, new_A, new_R, new_OUT );

        // Propagate changes if necessary
        if( old_OUT != new_OUT )
            sendPropertyDidChange_UnresolvedReference( refid, new_OUT );
    }
}

//------------------------------------------------------------------------
// RtxUniversalTraversal
//------------------------------------------------------------------------

void Selector::sendPropertyDidChange_RtxUniversalTraversal() const
{
    for( const auto& child : m_children )
        if( child )
            child->receivePropertyDidChange_RtxUniversalTraversal();
}

//------------------------------------------------------------------------
// Direct Caller
//------------------------------------------------------------------------

void Selector::attachOrDetachProperty_DirectCaller_toChildren( Program* program, bool added ) const
{
    for( const auto& child : m_children )
        if( child )
            program->attachOrDetachProperty_DirectCaller( child.get(), added );
}


//------------------------------------------------------------------------
// Trace Caller
//------------------------------------------------------------------------

void Selector::sendPropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added ) const
{
    for( const auto& child : m_children )
        if( child )
            child->receivePropertyDidChange_TraceCaller( cpid, added );
}

//------------------------------------------------------------------------
// Traversable support
//------------------------------------------------------------------------
RtcTraversableHandle Selector::getTraversableHandle( unsigned int allDeviceIndex ) const
{
    if( !m_context->useRtxDataModel() )
        return 0;

    const LWDADevice* lwdaDevice = deviceCast<const LWDADevice>( m_context->getDeviceManager()->allDevices()[allDeviceIndex] );
    if( !lwdaDevice )
        return 0;

    if( !m_traversables[allDeviceIndex] )
        return 0;

    return getTraversableHandleFromMAccess( lwdaDevice, m_traversables[allDeviceIndex]->getAccess( lwdaDevice ) );
}

void Selector::updateTraversable()
{
    if( !m_context->useRtxDataModel() )
        return;

    // if the primary device has a traverable, all active devices have one
    const unsigned int primaryAllDeviceListIndex = m_context->getDeviceManager()->primaryDevice()->allDeviceListIndex();
    if( m_traversables[primaryAllDeviceListIndex] )
        resizeTraversables();
    else
        allocateTraversables();

    for( Device* dev : m_context->getDeviceManager()->activeDevices() )
    {
        writeTraversable( dev->allDeviceListIndex() );
    }
}

static size_t getTraversableSize( unsigned int numChildren )
{
    return sizeof( RtcTravSelector ) + ( numChildren >= 1 ? numChildren - 1 : 0 ) * sizeof( RtcTraversableHandle );
}

BufferDimensions Selector::getTraversableDimensions() const
{
    return BufferDimensions( RT_FORMAT_USER, getTraversableSize( m_children.size() ), 1, 1, 1, 1 );
}

void Selector::allocateTraversables()
{
    for( Device* device : m_context->getDeviceManager()->allDevices() )
    {
        if( !device->isActive() )
            continue;

        const unsigned int allDeviceIndex = device->allDeviceListIndex();
        if( !m_traversables[allDeviceIndex] )
        {
            DeviceSet set( device );
            m_traversables[allDeviceIndex] =
                m_context->getMemoryManager()->allocateMBuffer( getTraversableDimensions(), MBufferPolicy::gpuLocal, set, this );
        }
    }
}

void Selector::resizeTraversables()
{
    const BufferDimensions desiredSize = getTraversableDimensions();
    for( Device* device : m_context->getDeviceManager()->allDevices() )
    {
        if( !device->isActive() )
            continue;

        const unsigned int allDeviceIndex = device->allDeviceListIndex();
        if( m_traversables[allDeviceIndex]->getDimensions() != desiredSize )
            m_context->getMemoryManager()->changeSize( m_traversables[allDeviceIndex], desiredSize );
    }
}

void Selector::writeTraversable( unsigned int allDeviceIndex )
{
    MemoryManager*   mm = m_context->getMemoryManager();
    RtcTravSelector* travData =
        reinterpret_cast<RtcTravSelector*>( mm->mapToHost( m_traversables[allDeviceIndex], MAP_WRITE_DISCARD ) );
    travData->numChildren = m_children.size();
    travData->sbtOffset   = getSBTIndex();
    if( travData->numChildren > 0 )
        std::fill( &travData->children[0], &travData->children[travData->numChildren], 0U );
    else
        travData->children[0] = 0U;
    mm->unmapFromHost( m_traversables[allDeviceIndex] );
}

void Selector::writeChildTraversable( unsigned int allDeviceIndex, unsigned int childIndex, RtcTraversableHandle childHandle )
{
    RT_ASSERT( childIndex < m_children.size() );
    MemoryManager*   mm = m_context->getMemoryManager();
    RtcTravSelector* travData =
        reinterpret_cast<RtcTravSelector*>( mm->mapToHost( m_traversables[allDeviceIndex], MAP_WRITE_DISCARD ) );
    travData->children[childIndex] = childHandle;
    mm->unmapFromHost( m_traversables[allDeviceIndex] );
}

void Selector::childTraversableHandleDidChange( LinkedPtr_Link* child, TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle travHandle )
{
    if( !m_context->useRtxDataModel() )
        return;

    if( source == TravSource::GEOMGROUP_DIRECT || source == TravSource::OTHER_DIRECT || source == TravSource::TRANSFORM_DIRECT )
    {
        unsigned int childIndex;
        bool         found = getElementIndex( m_children, child, childIndex );
        RT_ASSERT_MSG( found, "Didn't find child index from link" );
        writeChildTraversable( allDeviceIndex, childIndex, travHandle );
    }
}

void Selector::eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA )
{
    const LWDADevice* lwdaDevice = deviceCast<const LWDADevice>( device );
    if( !lwdaDevice )
        return;

    const unsigned int         allDeviceIndex = device->allDeviceListIndex();
    const RtcTraversableHandle travHandle     = getTraversableHandleFromMAccess( lwdaDevice, newMBA );
    sendDidChange_TraversableHandle( getTraversableSource(), allDeviceIndex, travHandle );
}

RtcTraversableHandle Selector::getTraversableHandleFromMAccess( const LWDADevice* lwdaDevice, const MAccess& access ) const
{
    // TODO: rtcore doesn't support selectors. return a null pointer so rtcore filters out TLAS instances referencing selectors.
    RtcTraversableHandle travHandle = 0;
    return travHandle;
}

GraphNode::TraversableDataForTest Selector::getTraversableDataForTest( unsigned int allDeviceIndex ) const
{
    TraversableDataForTest data{};
    data.m_type     = RTC_TRAVERSABLE_TYPE_SELECTOR;
    data.m_size     = getTraversableSize( m_children.size() );
    data.m_selector = reinterpret_cast<const RtcTravSelector*>(
        m_context->getMemoryManager()->mapToHost( m_traversables[allDeviceIndex], MAP_READ ) );
    data.m_traversableId = m_traversableId ? *m_traversableId : -1;
    return data;
}
