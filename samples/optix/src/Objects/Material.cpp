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

#include <Objects/Material.h>

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/ProgramManager.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GlobalScope.h>
#include <Objects/Program.h>
#include <Objects/ProgramRoot.h>
#include <Objects/Variable.h>
#include <Util/LinkedPtrHelpers.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;

static inline void computeReferenceResolutionLogic( bool bIN, bool bPC, bool bV, bool& bINP, bool& bR, bool& bOUT );

//------------------------------------------------------------------------
// CTOR / DTOR
//------------------------------------------------------------------------

Material::Material( Context* context )
    : LexicalScope( context, RT_OBJECT_MATERIAL )
    , m_hasAtLeastOneAnyHitProgram( false )
{
    setRayTypeCount( context->getRayTypeCount() );

    reallocateRecord();
}


Material::~Material()
{
    // Reset programs
    setRayTypeCount( 0 );
    deleteVariables();
}


//------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------

Program* Material::getClosestHitProgram( unsigned int rayTypeIndex ) const
{
    RT_ASSERT( rayTypeIndex < m_closestHitPrograms.size() );
    return m_closestHitPrograms[rayTypeIndex].get();
}

Program* Material::getAnyHitProgram( unsigned int rayTypeIndex ) const
{
    RT_ASSERT( rayTypeIndex < m_anyHitPrograms.size() );
    return m_anyHitPrograms[rayTypeIndex].get();
}

void Material::setClosestHitProgram( unsigned int rayTypeIndex, Program* program )
{
    RT_ASSERT( rayTypeIndex < m_closestHitPrograms.size() );

    // Call this before doing the attachment to help avoid exceptions during attachment.
    if( program )
        program->validateSemanticType( ST_CLOSEST_HIT );

    // Closest hit program properties
    //
    // Semantic type:                    originates here
    // Used by ray type:                 originates here
    // Attachment:                       propagates from parent
    // Trace caller:                     propagates from parent to closestHit
    // Unresolved attribute references:  propagates from child
    // Unresolved references:            propagates from child

    ProgramRoot root( getScopeID(), ST_CLOSEST_HIT, rayTypeIndex );

    if( Program* oldProgram = m_closestHitPrograms[rayTypeIndex].get() )
    {
        // Remove properties from old program before updating the pointer
        oldProgram->attachOrDetachProperty_UnresolvedReference( this, root, false );
        oldProgram->attachOrDetachProperty_UnresolvedAttributeReference( this, false );
        this->attachOrDetachProperty_TraceCaller( oldProgram, false );
        this->attachOrDetachProperty_Attachment( oldProgram, false );
        oldProgram->receivePropertyDidChange_UsedByRayType( rayTypeIndex, false );
        oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_CLOSEST_HIT, false );
    }

    m_closestHitPrograms[rayTypeIndex].set( this, program );

    if( Program* newProgram = m_closestHitPrograms[rayTypeIndex].get() )
    {
        newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_CLOSEST_HIT, true );
        newProgram->receivePropertyDidChange_UsedByRayType( rayTypeIndex, true );
        this->attachOrDetachProperty_Attachment( newProgram, true );
        this->attachOrDetachProperty_TraceCaller( newProgram, true );
        newProgram->attachOrDetachProperty_UnresolvedAttributeReference( this, true );
        newProgram->attachOrDetachProperty_UnresolvedReference( this, root, true );
    }

    notifyParents_closestHitProgramChanged( rayTypeIndex );
    subscribeForValidation();
    writeRecord();
}

void Material::setAnyHitProgram( unsigned int rayTypeIndex, Program* program )
{
    RT_ASSERT( rayTypeIndex < m_anyHitPrograms.size() );

    // Validate the semantic type before setting the program to avoid
    // exceptions during graph property propagation
    if( program )
        program->validateSemanticType( ST_ANY_HIT );

    // Any hit program properties (same as closest hit)
    //
    // Semantic type:                    originates here
    // Used by ray type:                 originates here
    // Attachment:                       propagates from parent
    // Direct caller:                    propagates from parent to anyHit
    // Unresolved attribute references:  propagates from child
    // Unresolved references:            propagates from child

    ProgramRoot root( getScopeID(), ST_ANY_HIT, rayTypeIndex );

    if( Program* oldProgram = m_anyHitPrograms[rayTypeIndex].get() )
    {
        // Remove properties from old program before updating the pointer
        oldProgram->attachOrDetachProperty_UnresolvedReference( this, root, false );
        oldProgram->attachOrDetachProperty_UnresolvedAttributeReference( this, false );
        this->attachOrDetachProperty_DirectCaller( oldProgram, false );
        this->attachOrDetachProperty_Attachment( oldProgram, false );
        oldProgram->receivePropertyDidChange_UsedByRayType( rayTypeIndex, false );
        oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_ANY_HIT, false );
    }

    m_anyHitPrograms[rayTypeIndex].set( this, program );

    if( Program* newProgram = m_anyHitPrograms[rayTypeIndex].get() )
    {
        newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_ANY_HIT, true );
        newProgram->receivePropertyDidChange_UsedByRayType( rayTypeIndex, true );
        this->attachOrDetachProperty_Attachment( newProgram, true );
        this->attachOrDetachProperty_DirectCaller( newProgram, true );
        newProgram->attachOrDetachProperty_UnresolvedAttributeReference( this, true );
        newProgram->attachOrDetachProperty_UnresolvedReference( this, root, true );
    }

    updateHasAtLeastOneAnyHitProgram();

    notifyParents_anyHitProgramChanged( rayTypeIndex );
    subscribeForValidation();
    writeRecord();
}

void Material::updateHasAtLeastOneAnyHitProgram()
{
    m_hasAtLeastOneAnyHitProgram = false;
    const Program* nullProgram   = getSharedNullProgram();
    for( size_t i = 0; i < m_anyHitPrograms.size(); ++i )
    {
        if( m_anyHitPrograms[i].get() != nullProgram )
        {
            m_hasAtLeastOneAnyHitProgram = true;
            break;
        }
    }
}

bool Material::hasAtLeastOneAnyHitProgram() const
{
    return m_hasAtLeastOneAnyHitProgram;
}

void Material::setRayTypeCount( unsigned int numRayTypes )
{
    unsigned int oldSize = (unsigned int)m_anyHitPrograms.size();
    // Resize the closest hit and anyhit programs, filling new entries
    // with builtin null program
    resizeVector( m_closestHitPrograms, numRayTypes, [this]( int index ) { setClosestHitProgram( index, nullptr ); },
                  [this]( int index ) { setClosestHitProgram( index, getSharedNullProgram() ); } );
    resizeVector( m_anyHitPrograms, numRayTypes, [this]( int index ) { setAnyHitProgram( index, nullptr ); },
                  [this]( int index ) { setAnyHitProgram( index, getSharedNullProgram() ); } );

    if( oldSize > numRayTypes )
    {
        updateHasAtLeastOneAnyHitProgram();
    }

    subscribeForValidation();
    reallocateRecord();
}

void Material::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    LexicalScope::validate();
    // Nothing else to do
}

//------------------------------------------------------------------------
// LinkedPtr relationship mangement
//------------------------------------------------------------------------

void Material::detachFromParents()
{
    auto iter = m_linkedPointers.begin();
    while( iter != m_linkedPointers.end() )
    {
        LinkedPtr_Link* parentLink = *iter;

        if( GeometryInstance* parent = getLinkToMaterialFrom<GeometryInstance>( parentLink ) )
            parent->detachLinkedChild( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to Material" );

        iter = m_linkedPointers.begin();
    }
}

void Material::detachLinkedChild( const LinkedPtr_Link* link )
{
    unsigned int index;
    if( getElementIndex( m_closestHitPrograms, link, index ) )
        setClosestHitProgram( index, getSharedNullProgram() );

    else if( getElementIndex( m_anyHitPrograms, link, index ) )
        setAnyHitProgram( index, getSharedNullProgram() );

    else
        RT_ASSERT_FAIL_MSG( "Invalid child link in detachLinkedChild" );
}


//------------------------------------------------------------------------
// Object record management
//------------------------------------------------------------------------

size_t Material::getRecordBaseSize() const
{
    // Material has a variable-sized array for the any-hit and
    // closest-hit programs that exist per ray type.
    size_t varsize = m_closestHitPrograms.size();
    RT_ASSERT( varsize == m_anyHitPrograms.size() );
    cort::MaterialRecord* matl = nullptr;
    return (char*)( &matl->programs[varsize] ) - (char*)( matl );
}

void Material::writeRecord() const
{
    if( !recordIsAllocated() )
        return;

    cort::MaterialRecord* m = getObjectRecord<cort::MaterialRecord>();
    RT_ASSERT( m != nullptr );

    // This relies on the size telling the truth about the past-the-end allocations
    for( size_t i                 = 0; i < m_closestHitPrograms.size(); ++i )
        m->programs[i].closestHit = getSafeOffset( m_closestHitPrograms[i].get() );
    for( size_t i             = 0; i < m_anyHitPrograms.size(); ++i )
        m->programs[i].anyHit = getSafeOffset( m_anyHitPrograms[i].get() );
    LexicalScope::writeRecord();
}

void Material::notifyParents_offsetDidChange() const
{
    for( auto parentLink : m_linkedPointers )
    {
        if( GeometryInstance* parent = getLinkToMaterialFrom<GeometryInstance>( parentLink ) )
            parent->childOffsetDidChange( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to Material" );
    }
}

//------------------------------------------------------------------------
// SBTRecord management
//------------------------------------------------------------------------

void Material::notifyParents_closestHitProgramChanged( unsigned int rayTypeIndex ) const
{
    for( auto parentLink : m_linkedPointers )
    {
        if( GeometryInstance* parent = getLinkToMaterialFrom<GeometryInstance>( parentLink ) )
            parent->materialClosestHitProgramDidChange( parentLink, rayTypeIndex );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to Material" );
    }
}

void Material::notifyParents_anyHitProgramChanged( unsigned int rayTypeIndex ) const
{
    for( auto parentLink : m_linkedPointers )
    {
        if( GeometryInstance* parent = getLinkToMaterialFrom<GeometryInstance>( parentLink ) )
            parent->materialAnyHitProgramDidChange( parentLink, rayTypeIndex );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to Material" );
    }
}


//------------------------------------------------------------------------
// Unresolved reference property
//------------------------------------------------------------------------

static inline void computeReferenceResolutionLogic( bool bIN, bool bPC, bool bV, bool& bINP, bool& bR, bool& bC )
{
    /*
   * IN = union(CH.OUT, AH.OUT )                   // Counting set (m_unresolvedSet). Notify GI.IN
   * PC = union(all parent GI.C)                   // Counting set (m_unresolvedSet_giCantResolve)
   * IN' = intersect( IN, PC )                     // Not stored.  Computed only for NodegraphPrinter
   * R = intersect(IN', V)                         // Resolution change
   * C = IN - V                                    // Notify GI.IN'
   */

    bINP = bIN && bPC;
    bR   = bINP && bV;
    bC   = bIN && !bV;
}

void Material::sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool added ) const
{
    // pre/post-resolve are used instead
    RT_ASSERT_FAIL_MSG( "property change improperly triggered from Material" );
}

void Material::receivePropertyDidChange_UnresolvedReference( const LexicalScope* child, VariableReferenceID refid, bool added )
{
    scopeTrace( "begin receivePropertyDidChange_UnresolvedReference", refid, added, child );

    // Callwlate new/old input bits
    bool setChanged = m_unresolvedSet.addOrRemoveProperty( refid, added );

    bool old_IN = !added || !setChanged;
    bool new_IN = added || !setChanged;
    bool old_PC = m_unresolvedSet_giCantResolve.contains( refid );
    bool new_PC = old_PC;
    bool old_V  = haveVariableForReference( refid );
    bool new_V  = old_V;

    // Callwlate derived sets
    bool old_INP, old_R, old_C;
    computeReferenceResolutionLogic( old_IN, old_PC, old_V, old_INP, old_R, old_C );
    bool new_INP, new_R, new_C;
    computeReferenceResolutionLogic( new_IN, new_PC, new_V, new_INP, new_R, new_C );

    // Propagate changes if necessary
    if( old_IN != new_IN )
        sendPropertyDidChange_UnresolvedReference_preResolve( refid, new_IN );
    if( old_C != new_C )
        sendPropertyDidChange_UnresolvedReference_childCantResolve( refid, new_C );
    if( old_R != new_R )
        updateResolvedSet( refid, new_R );

    scopeTrace( "end receivePropertyDidChange_UnresolvedReference", refid, added, child );
}

void Material::variableDeclarationDidChange( VariableReferenceID refid, bool variableWasAdded )
{
    scopeTrace( "begin variableDeclarationDidChange", refid, variableWasAdded );

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

    // Propagate changes if necessary
    RT_ASSERT( old_IN == new_IN );
    if( old_C != new_C )
        sendPropertyDidChange_UnresolvedReference_childCantResolve( refid, new_C );
    if( old_R != new_R )
        updateResolvedSet( refid, new_R );

    scopeTrace( "end variableDeclarationDidChange", refid, variableWasAdded );
}

void Material::receivePropertyDidChange_UnresolvedReference_giCantResolve( const GeometryInstance* parent,
                                                                           VariableReferenceID     refid,
                                                                           bool                    added )
{
    scopeTrace( "begin receivePropertyDidChange_UnresolvedReference_giCantResolve", refid, added, parent );

    // Callwlate new/old input bits
    bool old_IN     = m_unresolvedSet.contains( refid );
    bool new_IN     = old_IN;
    bool setChanged = m_unresolvedSet_giCantResolve.addOrRemoveProperty( refid, added );
    bool old_PC     = !added || !setChanged;
    bool new_PC     = added || !setChanged;
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

    scopeTrace( "end receivePropertyDidChange_UnresolvedReference_giCantResolve", refid, added, parent );
}

void Material::attachOrDetachProperty_UnresolvedReference_preResolve( GeometryInstance* gi, bool attached ) const
{
    scopeTrace( "begin attachOrDetachProperty_UnresolvedReference_preResolve", ~0, attached, gi );

    // Add/remove IN references from parent GI. No need to compute the
    // reference logic - all references are passed through.
    for( auto refid : m_unresolvedSet )
        gi->receivePropertyDidChange_UnresolvedReference_preResolve( this, refid, attached );

    scopeTrace( "end attachOrDetachProperty_UnresolvedReference_preResolve", ~0, attached, gi );
}

void Material::attachOrDetachProperty_UnresolvedReference_childCantResolve( GeometryInstance* gi, bool attached ) const
{
    scopeTrace( "begin attachOrDetachProperty_UnresolvedReference_childCantResolve", ~0, attached, gi );

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

    scopeTrace( "begin attachOrDetachProperty_UnresolvedReference_childCantResolve", ~0, attached, gi );
}

void Material::sendPropertyDidChange_UnresolvedReference_preResolve( VariableReferenceID refid, bool added ) const
{
    for( auto parentLink : m_linkedPointers )
    {
        GeometryInstance* parent = getLinkToMaterialFrom<GeometryInstance>( parentLink );
        if( parent )
            parent->receivePropertyDidChange_UnresolvedReference_preResolve( this, refid, added );
        else
            RT_ASSERT_FAIL_MSG( "Unexpected linked pointer type to Geometry" );
    }
}

void Material::sendPropertyDidChange_UnresolvedReference_childCantResolve( VariableReferenceID refid, bool added ) const
{
    for( auto parentLink : m_linkedPointers )
    {
        GeometryInstance* parent = getLinkToMaterialFrom<GeometryInstance>( parentLink );
        if( parent )
            parent->receivePropertyDidChange_UnresolvedReference_childCantResolve( this, refid, added );
        else
            RT_ASSERT_FAIL_MSG( "Unexpected linked pointer type to Geometry" );
    }
}

void Material::computeUnresolvedOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const
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

void Material::computeUnresolvedGIOutputForDebugging( GraphProperty<VariableReferenceID, false>& inp ) const
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

void Material::sendPropertyDidChange_UnresolvedAttributeReference( VariableReferenceID refid, bool added ) const
{
    for( auto parentLink : m_linkedPointers )
    {
        GeometryInstance* parent = getLinkToMaterialFrom<GeometryInstance>( parentLink );
        if( parent )
            parent->receivePropertyDidChange_UnresolvedAttributeReference( this, refid, added );
        else
            RT_ASSERT_FAIL_MSG( "Unexpected linked pointer type to Material" );
    }
}

void Material::attachOrDetachProperty_UnresolvedAttributeReference( GeometryInstance* gi, bool attached ) const
{
    for( auto refid : m_unresolvedAttributeSet )
        gi->receivePropertyDidChange_UnresolvedAttributeReference( this, refid, attached );
}

void Material::receivePropertyDidChange_UnresolvedAttributeReference( const Program* program, VariableReferenceID refid, bool added )
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

void Material::sendPropertyDidChange_Attachment( bool added ) const
{
    for( const auto& closestHit : m_closestHitPrograms )
        if( closestHit )
            closestHit->receivePropertyDidChange_Attachment( added );

    for( const auto& anyHit : m_anyHitPrograms )
        if( anyHit )
            anyHit->receivePropertyDidChange_Attachment( added );
}

//------------------------------------------------------------------------
// Direct Caller
//------------------------------------------------------------------------

void Material::sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const
{
    for( const auto& anyHit : m_anyHitPrograms )
        if( anyHit )
            anyHit->receivePropertyDidChange_DirectCaller( cpid, added );
}

void Material::attachOrDetachProperty_DirectCaller( Program* program, bool added ) const
{
    for( auto cpid : m_directCaller )
        program->receivePropertyDidChange_DirectCaller( cpid, added );
}


//------------------------------------------------------------------------
// Trace Caller
//------------------------------------------------------------------------

void Material::receivePropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added )
{
    bool changed = m_traceCaller.addOrRemoveProperty( cpid, added );
    if( changed )
        for( const auto& closestHit : m_closestHitPrograms )
            if( closestHit )
                closestHit->receivePropertyDidChange_DirectCaller( cpid, added );
}

void Material::attachOrDetachProperty_TraceCaller( Program* program, bool added ) const
{
    for( auto cpid : m_traceCaller )
        program->receivePropertyDidChange_DirectCaller( cpid, added );
}
