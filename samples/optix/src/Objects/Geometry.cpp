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

#include <Objects/Geometry.h>

#include <limits>

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/ProgramManager.h>
#include <Context/UpdateManager.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GlobalScope.h>
#include <Objects/Program.h>
#include <Objects/ProgramRoot.h>
#include <Objects/Variable.h>
#include <Util/LinkedPtrHelpers.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/ValidationError.h>


using namespace optix;
using namespace prodlib;


static inline void computeReferenceResolutionLogic( bool bIN, bool bPC, bool bV, bool& bINP, bool& bR, bool& bOUT );

//------------------------------------------------------------------------
// CTOR / DTOR
//------------------------------------------------------------------------

Geometry::Geometry( Context* context, bool isDerivedClass )
    : LexicalScope( context, RT_OBJECT_GEOMETRY )
{
    Program* null_program = getSharedNullProgram();
    setIntersectionProgram( null_program );
    setBoundingBoxProgram( null_program );
    if( !isDerivedClass )
        reallocateRecord();
}

Geometry::~Geometry()
{
    setIntersectionProgram( nullptr );
    setBoundingBoxProgram( nullptr );
    deleteVariables();
}

//------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------

void Geometry::setIntersectionProgram( Program* program )
{
    // Validate the semantic type before setting the program to avoid
    // exceptions during graph property propagation.
    if( program )
        program->validateSemanticType( ST_INTERSECTION );

    // Geometry program properties
    //
    // Semantic type:                    originates here
    // Attachment:                       propagates from parent
    // Direct caller:                    propagates from parent
    // Unresolved attribute references:  propagates from child
    // Unresolved references:            propagates from child

    ProgramRoot root( getScopeID(), ST_INTERSECTION, 0 );

    if( Program* oldProgram = m_intersectionProgram.get() )
    {
        // Remove properties from old program before updating the pointer
        intersectionProgramDidChange( oldProgram, false );
        oldProgram->attachOrDetachProperty_UnresolvedReference( this, root, false );
        oldProgram->attachOrDetachProperty_UnresolvedAttributeReference( this, false );
        this->attachOrDetachProperty_DirectCaller( oldProgram, false );
        this->attachOrDetachProperty_Attachment( oldProgram, false );
        oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_INTERSECTION, false );
    }

    m_intersectionProgram.set( this, program );

    if( Program* newProgram = m_intersectionProgram.get() )
    {
        // Add new properties
        newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_INTERSECTION, true );
        this->attachOrDetachProperty_Attachment( newProgram, true );
        this->attachOrDetachProperty_DirectCaller( newProgram, true );
        newProgram->attachOrDetachProperty_UnresolvedAttributeReference( this, true );
        newProgram->attachOrDetachProperty_UnresolvedReference( this, root, true );
        intersectionProgramDidChange( newProgram, true );
    }


    notifyParents_intersectionProgramDidChange();
    subscribeForValidation();
    writeRecord();
}

void Geometry::setBoundingBoxProgram( Program* program )
{
    // Validate the semantic type before setting the program to avoid
    // exceptions during graph property propagation
    if( program )
        program->validateSemanticType( ST_BOUNDING_BOX );

    // Bounding box program properties
    //
    // Semantic type:                    originates here
    // Attachment:                       propagates from parent
    // Unresolved references:            propagates from child
    // Bound callable programs:          attached here (using deferred attachment)

    ProgramRoot root( getScopeID(), ST_BOUNDING_BOX, 0 );

    if( Program* oldProgram = m_boundingBoxProgram.get() )
    {
        // Remove properties from old program before updating the pointer
        oldProgram->attachOrDetachProperty_UnresolvedReference( this, root, false );
        this->attachOrDetachProperty_Attachment( oldProgram, false );
        oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_BOUNDING_BOX, false );
    }

    m_boundingBoxProgram.set( this, program );

    if( Program* newProgram = m_boundingBoxProgram.get() )
    {
        // Add new properties
        newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_BOUNDING_BOX, true );
        this->attachOrDetachProperty_Attachment( newProgram, true );
        newProgram->attachOrDetachProperty_UnresolvedReference( this, root, true );
    }

    subscribeForValidation();
    writeRecord();
}

Program* Geometry::getIntersectionProgram() const
{
    return m_intersectionProgram.get();
}

Program* Geometry::getBoundingBoxProgram() const
{
    return m_boundingBoxProgram.get();
}

void Geometry::setPrimitiveCount( int primitiveCount )
{
    m_primitiveCount = primitiveCount;
    m_initialized    = true;
    subscribeForValidation();
}

int Geometry::getPrimitiveCount() const
{
    return m_primitiveCount;
}

void Geometry::setPrimitiveIndexOffset( int primitiveIndexOffset )
{
    m_primitiveIndexOffset = primitiveIndexOffset;
    writeRecord();
    subscribeForValidation();
}

int Geometry::getPrimitiveIndexOffset() const
{
    return m_primitiveIndexOffset;
}

void Geometry::setMotionSteps( int n )
{
    // Track whether this changes the HasMotionAabbs graph property.
    // Motion steps 1 (static): HasMotionAabbs == false
    // Motion steps > 1:        HasMotionAabbs == true
    const bool oldHasMotionAabbs = m_motionSteps > 1;

    m_motionSteps = n;
    subscribeForValidation();

    const bool newHasMotionAabbs = m_motionSteps > 1;

    // Remove old property and add new one
    if( newHasMotionAabbs != oldHasMotionAabbs )
        sendPropertyDidChange_HasMotionAabbs( newHasMotionAabbs );
}

int Geometry::getMotionSteps() const
{
    return m_motionSteps;
}

void Geometry::setMotionRange( float timeBegin, float timeEnd )
{
    m_timeBegin = timeBegin;
    m_timeEnd   = timeEnd;
}

void Geometry::getMotionRange( float& timeBegin, float& timeEnd ) const
{
    timeBegin = m_timeBegin;
    timeEnd   = m_timeEnd;
}

void Geometry::setMotionBorderMode( RTmotionbordermode borderBegin, RTmotionbordermode borderEnd )
{
    m_beginBorderMode = borderBegin;
    m_endBorderMode   = borderEnd;
}

void Geometry::getMotionBorderMode( RTmotionbordermode& borderBegin, RTmotionbordermode& borderEnd ) const
{
    borderBegin = m_beginBorderMode;
    borderEnd   = m_endBorderMode;
}

void Geometry::setFlags( RTgeometryflags flags )
{
    m_flags = flags;
}

RTgeometryflags Geometry::getFlags() const
{
    return m_flags;
}

void Geometry::markDirty()
{
    m_dirty = true;
}

bool Geometry::isDirty() const
{
    return m_dirty;
}

void Geometry::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    LexicalScope::validate();
    if( !m_initialized )
        throw ValidationError( RT_EXCEPTION_INFO, "Geometry does not have a primitive count" );

    // Ensure presence of intersection and aabb programs
    Program* nullProgram = getSharedNullProgram();
    if( getIntersectionProgram() == nullProgram )
        throw ValidationError( RT_EXCEPTION_INFO, "Geometry does not have an intersection program" );

    if( getBoundingBoxProgram() == nullProgram )
        throw ValidationError( RT_EXCEPTION_INFO, "Geometry does not have a bounding box program" );

    if( m_motionSteps > 1 && !getBoundingBoxProgram()->hasMotionIndexArg() )
        throw ValidationError( RT_EXCEPTION_INFO,
                               "Geometry has motion steps but does not have a motion bounding box program" );

    if( m_motionSteps == 0 )
        throw ValidationError( RT_EXCEPTION_INFO, "Geometry has motion steps == 0" );
}


//------------------------------------------------------------------------
// LinkedPtr relationship mangement
//------------------------------------------------------------------------

void Geometry::detachFromParents()
{
    auto iter = m_linkedPointers.begin();
    while( iter != m_linkedPointers.end() )
    {
        LinkedPtr_Link* parentLink = *iter;

        if( GeometryInstance* parent = getLinkToGeometryFrom<GeometryInstance>( parentLink ) )
            parent->detachLinkedChild( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to Geometry" );

        iter = m_linkedPointers.begin();
    }
}

void Geometry::detachLinkedChild( const LinkedPtr_Link* link )
{
    if( link == &m_intersectionProgram )
        setIntersectionProgram( getSharedNullProgram() );

    else if( link == &m_boundingBoxProgram )
        setBoundingBoxProgram( getSharedNullProgram() );

    else
        RT_ASSERT_FAIL_MSG( "Invalid child link in detachLinkedChild" );
}


//------------------------------------------------------------------------
// Object record management
//------------------------------------------------------------------------

size_t Geometry::getRecordBaseSize() const
{
    return sizeof( cort::GeometryRecord );
}

void Geometry::writeRecord() const
{
    if( !recordIsAllocated() )
        return;
    cort::GeometryRecord* g = getObjectRecord<cort::GeometryRecord>();
    RT_ASSERT( g != nullptr );
    g->indexOffset          = m_primitiveIndexOffset;
    g->intersectOrAttribute = getSafeOffset( m_intersectionProgram.get() );
    g->aabb                 = getSafeOffset( m_boundingBoxProgram.get() );
    if( m_intersectionProgram )
        g->attributeKind = m_intersectionProgram->get32bitAttributeKind();
    else
        g->attributeKind = 0;

    LexicalScope::writeRecord();
}

void Geometry::notifyParents_offsetDidChange() const
{
    for( auto parentLink : m_linkedPointers )
    {
        if( GeometryInstance* parent = getLinkToGeometryFrom<GeometryInstance>( parentLink ) )
            parent->childOffsetDidChange( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to Geometry" );
    }
}

//------------------------------------------------------------------------
// SBTRecord management
//------------------------------------------------------------------------

void Geometry::notifyParents_intersectionProgramDidChange() const
{
    for( auto parentLink : m_linkedPointers )
    {
        if( GeometryInstance* parent = getLinkToGeometryFrom<GeometryInstance>( parentLink ) )
            parent->geometryIntersectionProgramDidChange( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to Geometry" );
    }
}


//------------------------------------------------------------------------
// Unresolved reference property
//------------------------------------------------------------------------

static inline void computeReferenceResolutionLogic( bool bIN, bool bPC, bool bV, bool& bINP, bool& bR, bool& bC )
{
    /*
   * IN = union(AABB.OUT, Intersect.OUT )          // Counting set (m_unresolvedSet). Notify GI.IN
   * PC = union(all parent GI.C)                   // Counting set (m_unresolvedSet_midResolve)
   * IN' = intersect( IN, PC )                     // Not stored.  Computed only for NodegraphPrinter
   * R = intersect(IN', V)                         // Resolution change
   * C = IN - V                                    // Notify GI.IN'
   */

    bINP = bIN && bPC;
    bR   = bINP && bV;
    bC   = bIN && !bV;
}

void Geometry::receivePropertyDidChange_UnresolvedReference( const LexicalScope* child, VariableReferenceID refid, bool added )
{
    scopeTrace( "begin receivePropertyDidChange_UnresolvedReference", refid, added, child );

    // Callwlate new/old input bits
    bool setChanged = m_unresolvedSet.addOrRemoveProperty( refid, added );
    bool old_IN     = !added || !setChanged;
    bool new_IN     = added || !setChanged;
    bool old_PC     = m_unresolvedSet_giCantResolve.contains( refid );
    bool new_PC     = old_PC;
    bool old_V      = haveVariableForReference( refid );
    bool new_V      = old_V;

    // Callwlate derived sets
    bool old_INP, old_R, old_C;
    computeReferenceResolutionLogic( old_IN, old_PC, old_V, old_INP, old_R, old_C );
    bool new_INP, new_R, new_C;
    computeReferenceResolutionLogic( new_IN, new_PC, new_V, new_INP, new_R, new_C );

    // Propagate changes if necessary
    if( old_IN != new_IN )
        sendPropertyDidChange_UnresolvedReference_preResolve( refid, new_IN );
    if( old_C != new_C )
        sendPropertyDidChange_UnresolvedReference_childCantResolve( refid, new_IN );
    if( old_R != new_R )
        updateResolvedSet( refid, new_R );

    scopeTrace( "end receivePropertyDidChange_UnresolvedReference", refid, added, child );
}

void Geometry::sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool added ) const
{
    // pre/post-resolve is used instead
    RT_ASSERT_FAIL_MSG( "property change improperly triggered from Geometry" );
}

void Geometry::sendPropertyDidChange_UnresolvedReference_preResolve( VariableReferenceID refid, bool added ) const
{
    for( auto parentLink : m_linkedPointers )
    {
        GeometryInstance* parent = getLinkToGeometryFrom<GeometryInstance>( parentLink );
        if( parent )
            parent->receivePropertyDidChange_UnresolvedReference_preResolve( this, refid, added );
        else
            RT_ASSERT_FAIL_MSG( "Unexpected linked pointer type to Geometry" );
    }
}

void Geometry::sendPropertyDidChange_UnresolvedReference_childCantResolve( VariableReferenceID refid, bool added ) const
{
    for( auto parentLink : m_linkedPointers )
    {
        GeometryInstance* parent = getLinkToGeometryFrom<GeometryInstance>( parentLink );
        if( parent )
            parent->receivePropertyDidChange_UnresolvedReference_childCantResolve( this, refid, added );
        else
            RT_ASSERT_FAIL_MSG( "Unexpected linked pointer type to Geometry" );
    }
}

void Geometry::variableDeclarationDidChange( VariableReferenceID refid, bool variableWasAdded )
{
    bool old_IN = m_unresolvedSet.contains( refid );
    bool new_IN = old_IN;
    bool old_PC = m_unresolvedSet_giCantResolve.contains( refid );
    bool new_PC = old_PC;
    bool old_V  = !variableWasAdded;
    bool new_V  = variableWasAdded;

    // Callwlate derived sets
    bool old_INP, old_R, old_C;
    computeReferenceResolutionLogic( old_IN, old_PC, old_V, old_INP, old_R, old_C );
    bool new_INP, new_R, new_C;
    computeReferenceResolutionLogic( new_IN, new_PC, new_V, new_INP, new_R, new_C );

    // Propagate changes if necessary. Note: resolution cannot change the PC set.
    RT_ASSERT( old_IN == new_IN );
    if( old_C != new_C )
        sendPropertyDidChange_UnresolvedReference_childCantResolve( refid, new_C );
    if( old_R != new_R )
        updateResolvedSet( refid, new_R );
}

void Geometry::attachOrDetachProperty_UnresolvedReference_preResolve( GeometryInstance* gi, bool attached ) const
{
    // Add/remove IN references from parent GI. No need to compute the
    // reference logic - all references are passed through.
    for( auto refid : m_unresolvedSet )
        gi->receivePropertyDidChange_UnresolvedReference_preResolve( this, refid, attached );
}

void Geometry::attachOrDetachProperty_UnresolvedReference_childCantResolve( GeometryInstance* gi, bool attached ) const
{
    // Add/remove IN' references from parent GI
    for( auto refid : m_unresolvedSet )
    {
        bool bIN = true;
        bool bPC = m_unresolvedSet_giCantResolve.contains( refid );
        bool bV  = haveVariableForReference( refid );
        bool bINP, bR, bC;
        computeReferenceResolutionLogic( bIN, bPC, bV, bINP, bR, bC );
        if( bC )
            gi->receivePropertyDidChange_UnresolvedReference_childCantResolve( this, refid, attached );
    }
}

void Geometry::receivePropertyDidChange_UnresolvedReference_giCantResolve( const GeometryInstance* parent,
                                                                           VariableReferenceID     refid,
                                                                           bool                    addToUnresolvedSet )
{
    scopeTrace( "begin receivePropertyDidChange_UnresolvedReference_giCantResolve", refid, addToUnresolvedSet, parent );

    // Callwlate new/old input bits
    bool old_IN     = m_unresolvedSet.contains( refid );
    bool new_IN     = old_IN;
    bool setChanged = m_unresolvedSet_giCantResolve.addOrRemoveProperty( refid, addToUnresolvedSet );
    bool old_PC     = !addToUnresolvedSet || !setChanged;
    bool new_PC     = addToUnresolvedSet || !setChanged;
    bool old_V      = haveVariableForReference( refid );
    bool new_V      = old_V;

    // Callwlate derived sets
    bool old_INP, old_R, old_C;
    computeReferenceResolutionLogic( old_IN, old_PC, old_V, old_INP, old_R, old_C );
    bool new_INP, new_R, new_C;
    computeReferenceResolutionLogic( new_IN, new_PC, new_V, new_INP, new_R, new_C );

    // Propagate changes if necessary
    RT_ASSERT( old_IN == new_IN );
    RT_ASSERT( old_C == new_C );
    if( old_R != new_R )
        updateResolvedSet( refid, new_R );

    scopeTrace( "end receivePropertyDidChange_UnresolvedReference_giCantResolve", refid, addToUnresolvedSet, parent );
}

void Geometry::computeUnresolvedOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const
{
    // Note: this is the childCantResolve set...
    for( auto refid : m_unresolvedSet )
    {
        bool bIN = true;
        bool bPC = m_unresolvedSet_giCantResolve.contains( refid );
        bool bV  = haveVariableForReference( refid );
        bool bINP, bR, bC;
        computeReferenceResolutionLogic( bIN, bPC, bV, bINP, bR, bC );
        if( bC )
            out.addOrRemoveProperty( refid, true );
    }
}

void Geometry::computeUnresolvedGIOutputForDebugging( GraphProperty<VariableReferenceID, false>& inp ) const
{
    // Note: this is IN' - the mid-resolve set intersected with our input set
    for( auto refid : m_unresolvedSet )
    {
        bool bIN = true;
        bool bPC = m_unresolvedSet_giCantResolve.contains( refid );
        bool bV  = haveVariableForReference( refid );
        bool bINP, bR, bC;
        computeReferenceResolutionLogic( bIN, bPC, bV, bINP, bR, bC );
        if( bINP )
            inp.addOrRemoveProperty( refid, true );
    }
}


//------------------------------------------------------------------------
// Unresolved attribute property
//------------------------------------------------------------------------

void Geometry::sendPropertyDidChange_UnresolvedAttributeReference( VariableReferenceID refid, bool added ) const
{
    for( auto parentLink : m_linkedPointers )
    {
        if( GeometryInstance* parent = getLinkToGeometryFrom<GeometryInstance>( parentLink ) )
            parent->receivePropertyDidChange_UnresolvedAttributeReference( this, refid, added );
        else
            RT_ASSERT_FAIL_MSG( "Unexpected linked pointer type to Geometry" );
    }
}

const GraphProperty<VariableReferenceID>& Geometry::getUnresolvedAttributeSet() const
{
    return m_unresolvedAttributeSet;
}

void Geometry::attachOrDetachProperty_UnresolvedAttributeReference( GeometryInstance* gi, bool attached ) const
{
    for( auto refid : m_unresolvedAttributeSet )
        gi->receivePropertyDidChange_UnresolvedAttributeReference( this, refid, attached );
}

void Geometry::receivePropertyDidChange_UnresolvedAttributeReference( const Program* program, VariableReferenceID refid, bool added )
{
    // Although attributes cannot change after the program is attached,
    // this method is used when the Geometry->Program connection is
    // established.
    bool setChanged = m_unresolvedAttributeSet.addOrRemoveProperty( refid, added );
    bool old_A      = !added || !setChanged;
    bool new_A      = added || !setChanged;

    // Propagate changes if necessary
    if( old_A != new_A )
        sendPropertyDidChange_UnresolvedAttributeReference( refid, added );
}


//------------------------------------------------------------------------
// Attachment
//------------------------------------------------------------------------

void Geometry::sendPropertyDidChange_Attachment( bool added ) const
{
    if( m_intersectionProgram )
        m_intersectionProgram->receivePropertyDidChange_Attachment( added );
    if( m_boundingBoxProgram )
        m_boundingBoxProgram->receivePropertyDidChange_Attachment( added );
}

//------------------------------------------------------------------------
// Direct Caller
//------------------------------------------------------------------------

void Geometry::sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const
{
    if( m_intersectionProgram )
        m_intersectionProgram->receivePropertyDidChange_DirectCaller( cpid, added );
}

void Geometry::attachOrDetachProperty_DirectCaller( Program* program, bool added ) const
{
    // Send our direct caller to program, and send program's direct caller to parents
    for( auto cpid : m_directCaller )
        program->receivePropertyDidChange_DirectCaller( cpid, added );
    for( auto parentLink : m_linkedPointers )
        if( GeometryInstance* gi = getLinkToGeometryFrom<GeometryInstance>( parentLink ) )
            gi->attachOrDetachProperty_DirectCaller_toChildren( program, added );
}


//------------------------------------------------------------------------
// HasMotionAabbs
//------------------------------------------------------------------------

// Copied from GraphNode, with custom hasMotionAabbs()

bool Geometry::hasMotionAabbs() const
{
    return m_motionSteps > 1;
}

void Geometry::attachOrDetachProperty_HasMotionAabbs( GeometryInstance* gi, bool attached ) const
{
    const bool hasMotion = hasMotionAabbs();
    if( hasMotion )
        gi->receivePropertyDidChange_HasMotionAabbs( attached );
}

void Geometry::sendPropertyDidChange_HasMotionAabbs( bool added ) const
{
    for( auto parentLink : m_linkedPointers )
    {
        if( GeometryInstance* gi = getLinkToGeometryFrom<GeometryInstance>( parentLink ) )
        {
            gi->receivePropertyDidChange_HasMotionAabbs( added );
        }
    }
}

// No "receive_HasMotionAabbs" method since nothing attached to Geometry can change this property
