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
#include <Context/ProgramManager.h>
#include <Context/UpdateManager.h>
#include <Exceptions/TypeMismatch.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Objects/Buffer.h>
#include <Objects/Geometry.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GeometryTriangles.h>
#include <Objects/GlobalScope.h>
#include <Objects/Group.h>
#include <Objects/Material.h>
#include <Objects/Variable.h>
#include <Objects/VariableReferenceBinding.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/LinkedPtrHelpers.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/ValidationError.h>
#include <prodlib/system/Knobs.h>

#include <algorithm>

using namespace optix;
using namespace prodlib;
using namespace corelib;

namespace {
Knob<bool> k_scopeTrace( RT_DSTRING( "scope.trace" ), false, RT_DSTRING( "Trace unresolved reference set events" ) );
}
static inline void computeReferenceResolutionLogic( bool bIN, bool bV, bool bINP, bool bA, bool& bR, bool& bC, bool& bOUT );

//------------------------------------------------------------------------
// CTOR/DTOR
//------------------------------------------------------------------------

GeometryInstance::GeometryInstance( Context* context )
    : LexicalScope( context, RT_OBJECT_GEOMETRY_INSTANCE )
{
    reallocateRecord();
}

GeometryInstance::~GeometryInstance()
{
    setMaterialCount( 0 );
    setGeometry( nullptr );
    deleteVariables();
}


//------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------

void GeometryInstance::setGeometry( Geometry* geometry )
{

    // Check for setting the same value
    if( geometry == m_geometry.get() )
        return;

    // In addition to normal parent updates, GI also needs to notify
    // children because the of the reversed scope lookup.  Note: the
    // order of propagation for the intertwined unresolved refernce
    // sets. This order must be preserved to prevent double-counting of
    // references.

    // Geometry properties:
    //
    // Attachment:                             propagates from parent
    // Direct caller:                          propagates from parent
    // Direct caller (IS):                     propagates from child
    // Unresolved attribute references:        propagates from child
    // Unresolved GI reference (postResolve):  propagates from child
    // Unresolved GI reference (midResolve):   propagates from parent
    // Unresolved GI reference (preResolve):   propagates from child
    // HasMotionAabbs                      :   propagates from child

    if( Geometry* oldGeometry = m_geometry.get() )
    {
        // Remove properties from old program before updating the pointer
        oldGeometry->attachOrDetachProperty_HasMotionAabbs( this, false );
        oldGeometry->attachOrDetachProperty_UnresolvedReference_preResolve( this, false );
        this->attachOrDetachProperty_UnresolvedReference_giCantResolve( oldGeometry, false );
        oldGeometry->attachOrDetachProperty_UnresolvedReference_childCantResolve( this, false );
        oldGeometry->attachOrDetachProperty_UnresolvedAttributeReference( this, false );
        if( oldGeometry->getIntersectionProgram() )
            attachOrDetachProperty_DirectCaller_toChildren( oldGeometry->getIntersectionProgram(), false );
        this->attachOrDetachProperty_DirectCaller( oldGeometry, false );
        this->attachOrDetachProperty_Attachment( oldGeometry, false );
    }

    m_geometry.set( this, geometry );

    if( Geometry* newGeometry = m_geometry.get() )
    {
        // Add new properties
        this->attachOrDetachProperty_Attachment( newGeometry, true );
        this->attachOrDetachProperty_DirectCaller( newGeometry, true );
        if( newGeometry->getIntersectionProgram() )
            attachOrDetachProperty_DirectCaller_toChildren( newGeometry->getIntersectionProgram(), true );
        newGeometry->attachOrDetachProperty_UnresolvedAttributeReference( this, true );
        newGeometry->attachOrDetachProperty_UnresolvedReference_childCantResolve( this, true );
        this->attachOrDetachProperty_UnresolvedReference_giCantResolve( newGeometry, true );
        newGeometry->attachOrDetachProperty_UnresolvedReference_preResolve( this, true );
        newGeometry->attachOrDetachProperty_HasMotionAabbs( this, true );
    }

    // Flush the virtual parent queue
    m_context->getBindingManager()->processVirtualParentQueue();

    // Validate and update
    notifyParents_geometryDidChange();
    subscribeForValidation();
    writeRecord();
}

Geometry* GeometryInstance::getGeometry() const
{
    return m_geometry.get();
}

void GeometryInstance::setMaterial( unsigned int index, Material* material )
{
    if( index >= m_materials.size() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Material index out of bounds", index );

    // Check for setting the same value
    if( material == m_materials[index].get() )
        return;

    // See note in setGeometry regarding ordering.

    // Material properties:
    //
    // Direct Caller:                          propagates from Geometry's IS to child
    // Trace Caller:                           propagates from parent (must be before attachment)
    // Unresolved attribute references:        propagates from child
    // Unresolved GI reference (postResolve):  propagates from child
    // Unresolved GI reference (midResolve):   propagates from parent
    // Unresolved GI reference (preResolve):   propagates from child
    // Attachment:                             propagates from parent

    if( Material* oldMaterial = m_materials[index].get() )
    {
        // Remove properties from old program before updating the pointer
        this->attachOrDetachProperty_Attachment( oldMaterial, false );
        oldMaterial->attachOrDetachProperty_UnresolvedReference_preResolve( this, false );
        this->attachOrDetachProperty_UnresolvedReference_giCantResolve( oldMaterial, false );
        oldMaterial->attachOrDetachProperty_UnresolvedReference_childCantResolve( this, false );
        oldMaterial->attachOrDetachProperty_UnresolvedAttributeReference( this, false );
        this->attachOrDetachProperty_TraceCaller( oldMaterial, false );
        if( getGeometry() && getGeometry()->getIntersectionProgram() )
            getGeometry()->getIntersectionProgram()->attachOrDetachProperty_DirectCaller( oldMaterial, false );
    }

    m_materials[index].set( this, material );

    if( Material* newMaterial = m_materials[index].get() )
    {
        // Add new properties
        if( getGeometry() && getGeometry()->getIntersectionProgram() )
            getGeometry()->getIntersectionProgram()->attachOrDetachProperty_DirectCaller( newMaterial, true );
        this->attachOrDetachProperty_TraceCaller( newMaterial, true );
        newMaterial->attachOrDetachProperty_UnresolvedAttributeReference( this, true );
        newMaterial->attachOrDetachProperty_UnresolvedReference_childCantResolve( this, true );
        this->attachOrDetachProperty_UnresolvedReference_giCantResolve( newMaterial, true );
        newMaterial->attachOrDetachProperty_UnresolvedReference_preResolve( this, true );
        this->attachOrDetachProperty_Attachment( newMaterial, true );
    }

    // Flush the virtual parent queue
    m_context->getBindingManager()->processVirtualParentQueue();

    // Validate and update
    notifyParents_materialDidChange( index );
    subscribeForValidation();
    writeRecord();
}

Material* GeometryInstance::getMaterial( unsigned int index )
{
    if( index >= m_materials.size() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Material index out of bounds", index );

    return m_materials[index].get();
}

const Material* GeometryInstance::getMaterial( unsigned int index ) const
{
    if( index >= m_materials.size() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Material index out of bounds", index );

    return m_materials[index].get();
}

void GeometryInstance::setMaterialCount( int numMaterials )
{
    // Check for setting the same value
    if( numMaterials == static_cast<int>( m_materials.size() ) )
        return;

    resizeVector( m_materials, numMaterials, [this]( int index ) { setMaterial( index, nullptr ); },
                  [this]( int index ) { setMaterial( index, nullptr ); } );

    notifyParents_materialCountDidChange();
    subscribeForValidation();
    reallocateRecord();
}

int GeometryInstance::getMaterialCount() const
{
    return static_cast<int>( m_materials.size() );
}

static void checkAttrTypeMatching( const VariableReference* attr0, const VariableReference* attr1 )
{
    const CanonicalProgram* p0 = attr0->getParent();
    const CanonicalProgram* p1 = attr1->getParent();
    const VariableType&     t0 = attr0->getType();
    const VariableType&     t1 = attr1->getType();
    if( t0 != t1 )
    {
        std::ostringstream oss;
        oss << "Attribute \"" << p0->getInputFunctionName() << "::" << attr0->getInputName() << "\" assigned type "
            << t0.toString() << " mismatches \"" << p1->getInputFunctionName() << "::" << attr1->getInputName()
            << "\" assigned type " << t1.toString();
        throw TypeMismatch( RT_EXCEPTION_INFO, oss.str() );
    }
}

void GeometryInstance::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    LexicalScope::validate();
    if( !getGeometry() )
        throw ValidationError( RT_EXCEPTION_INFO, "Missing geometry object in instance" );

    if( getMaterialCount() == 0 )
        throw ValidationError( RT_EXCEPTION_INFO, "No materials in instance" );

    int n = getMaterialCount();

    GeometryTriangles* gt = managedObjectCast<GeometryTriangles>( getGeometry() );
    if( gt )
        if( gt->getMaterialCount() != m_materials.size() )
            throw ValidationError( RT_EXCEPTION_INFO,
                                   "GeometryInstance: number of materials must match the number of materials set on "
                                   "the GeometryTriangles node" );

    for( int i = 0; i < n; ++i )
    {
        const Material* matl = getMaterial( i );
        if( !matl )
        {
            std::ostringstream out;
            out << "Material index " << i << " is null";
            throw ValidationError( RT_EXCEPTION_INFO, out.str() );
        }
    }

    // Attributes must be resolved at this scope
    if( !m_unresolvedAttributesRemaining.empty() )
    {
        // Include the names of all unresolved attributes when reporting the
        // error.
        std::vector<std::string> attributeNames;
        for( VariableReferenceID matRefId : m_unresolvedAttributesRemaining )
        {
            const VariableReference* matVarRef = m_context->getProgramManager()->getVariableReferenceById( matRefId );
            attributeNames.push_back( matVarRef->getInputName() );
        }

        // Sort the attribute names and remove duplicates
        algorithm::sort( attributeNames );
        attributeNames.erase( std::unique( attributeNames.begin(), attributeNames.end() ), attributeNames.end() );

        // Assemble the names into a comma-separated list
        std::string namesList;
        for( const std::string& name : attributeNames )
        {
            namesList += " \"" + name + "\",";
        }
        namesList.pop_back();

        if( gt )
            throw ValidationError( RT_EXCEPTION_INFO,
                                   "The following attributes are referenced in a hit program "
                                   "but are not produced in the attached attribute program:"
                                       + namesList );
        else
            throw ValidationError( RT_EXCEPTION_INFO,
                                   "The following attributes are referenced in a hit program "
                                   "but are not produced in the attached intersection program:"
                                       + namesList );
    }

    // And attributes must typematch. If this is a perf bottleneck,
    // consider moving the match to the updateResolvedAttributeSet
    // method below - only inserting mismatched elements into a set.
    for( const auto& mat_geo : m_resolvedAttributes )
    {
        const VariableReference* matVarRef  = m_context->getProgramManager()->getVariableReferenceById( mat_geo.first );
        const VariableReference* geomVarRef = m_context->getProgramManager()->getVariableReferenceById( mat_geo.second );
        checkAttrTypeMatching( matVarRef, geomVarRef );
    }
}


//------------------------------------------------------------------------
// LinkedPtr relationship mangement
//------------------------------------------------------------------------

void GeometryInstance::detachFromParents()
{
    auto iter = m_linkedPointers.begin();
    while( iter != m_linkedPointers.end() )
    {
        LinkedPtr_Link* parentLink = *iter;

        // Abstract group is the only parent, but it attaches to us as a
        // LexicalScope.
        if( AbstractGroup* parent = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
            parent->detachLinkedChild( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to GeometryInstance" );

        iter = m_linkedPointers.begin();
    }
}

void GeometryInstance::detachLinkedChild( const LinkedPtr_Link* link )
{
    unsigned int index;
    if( getElementIndex( m_materials, link, index ) )
        setMaterial( index, nullptr );

    else if( link == &m_geometry )
        setGeometry( nullptr );

    else
        RT_ASSERT_FAIL_MSG( "Invalid child link in detachLinkedChild" );
}

void GeometryInstance::childOffsetDidChange( const LinkedPtr_Link* link )
{
    LexicalScope::childOffsetDidChange( link );

    unsigned int index;
    if( getElementIndex( m_materials, link, index ) )
        notifyParents_materialOffsetDidChange( index );
}

//------------------------------------------------------------------------
// Object record management
//------------------------------------------------------------------------

size_t GeometryInstance::getRecordBaseSize() const
{
    size_t varsize = 0;

    // In RTX mode, the material pointers are collected directly from the SBT
    // payload. Therefore, they do not need to exist in the object record.
    // Eliminating also has the useful side-effect of starting the variables at a
    // consistent address, avoiding dynamic lookups (see bug 2284827). Should this
    // data be needed in the object record again in the future, we should put the
    // material pointers to the left of (before) the GeometryInstance object
    // record.
    if( !m_context->useRtxDataModel() )
    {
        // Material has a variable-sized array for the child materials
        varsize = m_materials.size();
    }

    cort::GeometryInstanceRecord* gi = nullptr;
    return (char*)( &gi->materials[varsize] ) - (char*)( gi );
}

void GeometryInstance::writeRecord() const
{
    if( !recordIsAllocated() )
        return;
    cort::GeometryInstanceRecord* g = getObjectRecord<cort::GeometryInstanceRecord>();
    RT_ASSERT( g != nullptr );
    g->geometry = getSafeOffset( m_geometry.get() );
    // This relies on the size telling the truth about the past-the-end allocations
    g->numMaterials = m_materials.size();

    // See comment in getRecordBaseSize() for why material pointers are not needed in RTX mode.
    if( !m_context->useRtxDataModel() )
    {
        for( size_t i = 0; i < m_materials.size(); ++i )
        {
            g->materials[i] = getSafeOffset( m_materials[i].get() );
        }
    }
    LexicalScope::writeRecord();
}

void GeometryInstance::notifyParents_offsetDidChange() const
{
    for( auto parentLink : m_linkedPointers )
    {
        // Abstract group is the only parent, but it attaches to us as a LexicalScope.
        if( AbstractGroup* parent = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
            parent->childOffsetDidChange( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to GeometryInstance" );
    }
}

//------------------------------------------------------------------------
// SBTRecord management
//------------------------------------------------------------------------
void GeometryInstance::notifyParents_materialCountDidChange() const
{
    for( auto parentLink : m_linkedPointers )
    {
        // Abstract group is the only parent, but it attaches to us as a LexicalScope.
        if( AbstractGroup* parent = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
            parent->materialCountDidChange( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to GeometryInstance" );
    }
}

void GeometryInstance::notifyParents_geometryDidChange() const
{
    for( auto parentLink : m_linkedPointers )
    {
        // Abstract group is the only parent, but it attaches to us as a LexicalScope.
        if( AbstractGroup* parent = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
            parent->geometryDidChange( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to GeometryInstance" );
    }
}

void GeometryInstance::notifyParents_materialDidChange( unsigned int materialIndex ) const
{
    for( auto parentLink : m_linkedPointers )
    {
        // Abstract group is the only parent, but it attaches to us as a LexicalScope.
        if( AbstractGroup* parent = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
            parent->materialDidChange( parentLink, materialIndex );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to GeometryInstance" );
    }
}

void GeometryInstance::notifyParents_materialOffsetDidChange( unsigned int materialIndex ) const
{
    for( auto parentLink : m_linkedPointers )
    {
        // Abstract group is the only parent, but it attaches to us as a LexicalScope.
        if( AbstractGroup* parent = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
            parent->materialOffsetDidChange( parentLink, materialIndex );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to GeometryInstance" );
    }
}

void GeometryInstance::geometryIntersectionProgramDidChange( const LinkedPtr_Link* link )
{
    for( auto parentLink : m_linkedPointers )
    {
        // Abstract group is the only parent, but it attaches to us as a LexicalScope.
        if( AbstractGroup* parent = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
            parent->geometryIntersectionDidChange( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to GeometryInstance" );
    }
}

void GeometryInstance::materialClosestHitProgramDidChange( const LinkedPtr_Link* link, unsigned int rayTypeIndex )
{
    unsigned int materialIndex;
    bool         foundMaterial = getElementIndex( m_materials, link, materialIndex );
    RT_ASSERT_MSG( foundMaterial, "Didn't find material in materialClosestHitProgramDidChange" );

    for( auto parentLink : m_linkedPointers )
    {
        // Abstract group is the only parent, but it attaches to us as a LexicalScope.
        if( AbstractGroup* parent = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
            parent->materialClosestHitProgramDidChange( parentLink, materialIndex, rayTypeIndex );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to GeometryInstance" );
    }
}

void GeometryInstance::materialAnyHitProgramDidChange( const LinkedPtr_Link* link, unsigned int rayTypeIndex )
{
    unsigned int materialIndex;
    bool         foundMaterial = getElementIndex( m_materials, link, materialIndex );
    RT_ASSERT_MSG( foundMaterial, "Didn't find material in materialAnyHitProgramDidChange" );

    for( auto parentLink : m_linkedPointers )
    {
        // Abstract group is the only parent, but it attaches to us as a LexicalScope.
        if( AbstractGroup* parent = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
            parent->materialAnyHitProgramDidChange( parentLink, materialIndex, rayTypeIndex );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to GeometryInstance" );
    }
}


//------------------------------------------------------------------------
// Unresolved reference property. Note: two sets are maintained to
// implement the reversed GI / material lookup.
//------------------------------------------------------------------------

static inline void computeReferenceResolutionLogic( bool bIN, bool bV, bool bINP, bool bA, bool& bR, bool& bC, bool& bOUT )
{
    /*
   * IN = union(Geometry.IN, all Material.IN)      // Counting set (m_unresolvedSet)
   * R = intersect(IN, V)                          // Resolution change
   * C = IN - V                                    // Notify Geometry and Material children
   * IN' = union(Geometry.C, all Material.C)       // Counting set (m_unresolvedSet_childCantResolve)
   * OUT = intersect( IN' - V, A )                 // Notify Context
   */

    bR   = bIN && bV;
    bC   = bIN && !bV;
    bOUT = bINP && !bV && bA;
}

void GeometryInstance::sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool added ) const
{
    // Global scope is the only parent (implicitly)
    m_context->getGlobalScope()->receivePropertyDidChange_UnresolvedReference( this, refid, added );
}

void GeometryInstance::receivePropertyDidChange_UnresolvedReference( const LexicalScope* child, VariableReferenceID refid, bool added )
{
    // Child changes should go through the IN/OUT reference change
    // methods below.
    RT_ASSERT_FAIL_MSG( "Unexpected reference change received at GeometryInstance" );
}

void GeometryInstance::attachOrDetachProperty_UnresolvedReference_giCantResolve( Geometry* child, bool attached ) const
{
    for( auto refid : m_unresolvedSet )
    {
        bool bIN  = true;
        bool bV   = haveVariableForReference( refid );
        bool bINP = false;  // Not relevant for this change
        bool bA   = true;   // Not relevant for this change
        bool bR, bC, bOUT;

        computeReferenceResolutionLogic( bIN, bV, bINP, bA, bR, bC, bOUT );
        if( bC )
            child->receivePropertyDidChange_UnresolvedReference_giCantResolve( this, refid, attached );
    }
}

// Duplicate of the function above for Material instead of Geometry. If you change one, change both.
void GeometryInstance::attachOrDetachProperty_UnresolvedReference_giCantResolve( Material* child, bool attached ) const
{
    for( auto refid : m_unresolvedSet )
    {
        bool bIN  = true;
        bool bV   = haveVariableForReference( refid );
        bool bINP = false;  // Not relevant for this change
        bool bA   = true;   // Not relevant for this change
        bool bR, bC, bOUT;

        computeReferenceResolutionLogic( bIN, bV, bINP, bA, bR, bC, bOUT );
        if( bC )
            child->receivePropertyDidChange_UnresolvedReference_giCantResolve( this, refid, attached );
    }
}

void GeometryInstance::sendPropertyDidChange_UnresolvedReference_giCantResolve( VariableReferenceID refid, bool added ) const
{
    // Propagate change to geometry and all materials
    if( m_geometry )
        m_geometry->receivePropertyDidChange_UnresolvedReference_giCantResolve( this, refid, added );
    for( const auto& material : m_materials )
        if( material )
            material->receivePropertyDidChange_UnresolvedReference_giCantResolve( this, refid, added );
}

void GeometryInstance::variableDeclarationDidChange( VariableReferenceID refid, bool variableWasAdded )
{
    scopeTrace( "begin variableDeclarationDidChange", refid, variableWasAdded );

    // Callwlate new/old input bits
    bool old_IN  = m_unresolvedSet.contains( refid );
    bool new_IN  = old_IN;
    bool old_V   = !variableWasAdded;
    bool new_V   = variableWasAdded;
    bool old_INP = m_unresolvedSet_childCantResolve.contains( refid );
    bool new_INP = old_IN;
    bool A       = isAttached();

    // Callwlate derived sets
    bool old_R, old_C, old_OUT;
    computeReferenceResolutionLogic( old_IN, old_V, old_INP, A, old_R, old_C, old_OUT );
    bool new_R, new_C, new_OUT;
    computeReferenceResolutionLogic( new_IN, new_V, new_INP, A, new_R, new_C, new_OUT );

    // Propagate changes if necessary
    if( old_C != new_C )
        sendPropertyDidChange_UnresolvedReference_giCantResolve( refid, new_C );
    if( old_OUT != new_OUT )
        sendPropertyDidChange_UnresolvedReference( refid, new_OUT );
    if( old_R != new_R )
        updateResolvedSet( refid, new_R );

    scopeTrace( "end variableDeclarationDidChange", refid, variableWasAdded );
}

void GeometryInstance::receivePropertyDidChange_UnresolvedReference_preResolve( const LexicalScope* scope,
                                                                                VariableReferenceID refid,
                                                                                bool                added )
{
    scopeTrace( "begin receivePropertyDidChange_UnresolvedReference_preResolve", refid, added, scope );

    // Callwlate new/old input bits
    bool setChanged = m_unresolvedSet.addOrRemoveProperty( refid, added );
    bool old_IN     = !added || !setChanged;
    bool new_IN     = added || !setChanged;
    bool old_V      = haveVariableForReference( refid );
    bool new_V      = old_V;
    bool A          = isAttached();
    bool old_INP    = false;  // Not used for this change, lookup unnecessary
    bool new_INP    = false;  // Not used for this change

    // Callwlate derived sets
    bool old_R, old_C, old_OUT;
    computeReferenceResolutionLogic( old_IN, old_V, old_INP, A, old_R, old_C, old_OUT );
    bool new_R, new_C, new_OUT;
    computeReferenceResolutionLogic( new_IN, new_V, new_INP, A, new_R, new_C, new_OUT );

    // Propagate changes if necessary
    if( old_C != new_C )
        sendPropertyDidChange_UnresolvedReference_giCantResolve( refid, new_C );
    if( old_R != new_R )
        updateResolvedSet( refid, new_R );

    scopeTrace( "end receivePropertyDidChange_UnresolvedReference_preResolve", refid, added, scope );
}

void GeometryInstance::receivePropertyDidChange_UnresolvedReference_childCantResolve( const LexicalScope* scope,
                                                                                      VariableReferenceID refid,
                                                                                      bool                added )
{
    scopeTrace( "begin receivePropertyDidChange_UnresolvedReference_postResolve", refid, added, scope );

    // Callwlate new/old input bits
    bool old_IN     = false;  // Not used for this change, lookup unnecessary
    bool new_IN     = false;  // Not used for this change, lookup unnecessary
    bool old_V      = haveVariableForReference( refid );
    bool new_V      = old_V;
    bool A          = isAttached();
    bool setChanged = m_unresolvedSet_childCantResolve.addOrRemoveProperty( refid, added );
    bool old_INP    = !added || !setChanged;
    bool new_INP    = added || !setChanged;

    // Callwlate derived sets
    bool old_R, old_C, old_OUT;
    computeReferenceResolutionLogic( old_IN, old_V, old_INP, A, old_R, old_C, old_OUT );
    bool new_R, new_C, new_OUT;
    computeReferenceResolutionLogic( new_IN, new_V, new_INP, A, new_R, new_C, new_OUT );

    // Propagate changes if necessary
    RT_ASSERT( old_C == new_C );
    RT_ASSERT( old_R == new_R );
    if( old_OUT != new_OUT )
        sendPropertyDidChange_UnresolvedReference( refid, new_OUT );

    scopeTrace( "end receivePropertyDidChange_UnresolvedReference_postResolve", refid, added, scope );
}

void GeometryInstance::computeUnresolvedOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const
{
    for( auto refid : m_unresolvedSet )
    {
        bool bIN  = true;
        bool bV   = haveVariableForReference( refid );
        bool bINP = true;
        bool bA   = isAttached();
        bool bR, bC, bOUT;

        computeReferenceResolutionLogic( bIN, bV, bINP, bA, bR, bC, bOUT );
        if( bOUT )
            out.addOrRemoveProperty( refid, true );
    }
}

void GeometryInstance::computeUnresolvedChildOutputForDebugging( GraphProperty<VariableReferenceID, false>& c ) const
{
    for( auto refid : m_unresolvedSet )
    {
        bool bIN  = true;
        bool bV   = haveVariableForReference( refid );
        bool bINP = true;
        bool bA   = isAttached();
        bool bR, bC, bOUT;

        computeReferenceResolutionLogic( bIN, bV, bINP, bA, bR, bC, bOUT );
        if( bC )
            c.addOrRemoveProperty( refid, true );
    }
}


//------------------------------------------------------------------------
// Unresolved attribute property
//------------------------------------------------------------------------

void GeometryInstance::updateResolvedAttributeSet( VariableReferenceID matRefId, VariableReferenceID geomRefId, bool added )
{
    if( k_scopeTrace.get() )
    {
        lprint << "scope: " << ( added ? "creating" : "dissolving" ) << " binding for material-geometry attributes "
               << matRefId << " : " << geomRefId << '\n';
    }
    if( added )
    {
        RT_ASSERT_MSG( m_resolvedAttributes.count( matRefId ) == 0, "Error adding attribute resolution" );
        m_resolvedAttributes.insert( std::make_pair( matRefId, geomRefId ) );
    }
    else
    {
        RT_ASSERT_MSG( m_resolvedAttributes.count( matRefId ) == 1, "Error removing attribute resolution" );
        RT_ASSERT_MSG( m_resolvedAttributes.at( matRefId ) == geomRefId, "Error removing attribute resolution" );
        m_resolvedAttributes.erase( matRefId );
    }
    subscribeForValidation();
}

void GeometryInstance::updateRemainingAttributeSet( VariableReferenceID matRefId, bool added )
{
    m_unresolvedAttributesRemaining.addOrRemoveProperty( matRefId, added );
    subscribeForValidation();
}

bool GeometryInstance::haveGeometryAttributeForReference( unsigned short token, VariableReferenceID& geomRefId ) const
{
    geomRefId = ~0;

    if( !m_geometry )
        return false;

    const GraphProperty<VariableReferenceID>& attrs = m_geometry->getUnresolvedAttributeSet();
    const ProgramManager*                     pm    = m_context->getProgramManager();
    for( auto refid : pm->getReferencesForVariable( token ) )
    {
        if( attrs.contains( refid ) )
        {
            geomRefId = refid;
            return true;
        }
    }
    return false;
}

void GeometryInstance::receivePropertyDidChange_UnresolvedAttributeReference( const Geometry*     geometry,
                                                                              VariableReferenceID geomRefId,
                                                                              bool geometryAttributeWasAdded )
{
    if( k_scopeTrace.get() )
    {
        lprint << "scope: GeometryInstance ID " << getScopeID()
               << " start receivePropertyDidChange_UnresolvedAttributeReference attribute id " << geomRefId << " "
               << ( geometryAttributeWasAdded ? "added" : "removed" ) << "\n";
    }

    // Callwlate new/old input bits for all references of the same name
    const ProgramManager*    pm         = m_context->getProgramManager();
    const VariableReference* geomVarRef = pm->getVariableReferenceById( geomRefId );
    for( auto matRefId : pm->getReferencesForVariable( geomVarRef->getVariableToken() ) )
    {
        bool old_M = m_unresolvedAttributeSet.contains( matRefId );
        bool new_M = old_M;
        bool old_G = !geometryAttributeWasAdded;
        bool new_G = geometryAttributeWasAdded;

        // Callwlate derived sets (U = M-G)
        bool old_R = old_M && old_G;
        bool new_R = new_M && new_G;
        bool old_U = old_M && !old_G;
        bool new_U = new_M && !new_G;

        // Propagate changes if necessary
        if( old_R != new_R )
            updateResolvedAttributeSet( matRefId, geomRefId, new_R );
        if( old_U != new_U )
            updateRemainingAttributeSet( matRefId, new_U );

        if( k_scopeTrace.get() )
        {
            lprint << "matRefId = " << matRefId << "\n";
            lprint << "U   (" << old_U << ", " << new_U << ")\n";
        }
    }
    if( k_scopeTrace.get() )
    {
        lprint << "scope: GeometryInstance ID " << getScopeID()
               << " end   childGeometryAttributeDidChange attribute id " << geomRefId << " "
               << ( geometryAttributeWasAdded ? "added" : "removed" ) << "\n";
    }
}

void GeometryInstance::receivePropertyDidChange_UnresolvedAttributeReference( const Material*     material,
                                                                              VariableReferenceID matRefId,
                                                                              bool materialAttributeWasAdded )
{
    if( k_scopeTrace.get() )
    {
        lprint << "scope: GeometryInstance ID " << getScopeID()
               << " start childMaterialAttributeDidChange material: " << material->getScopeID() << " attribute id "
               << matRefId << " " << ( materialAttributeWasAdded ? "added" : "removed" ) << "\n";
    }
    const ProgramManager*    pm     = m_context->getProgramManager();
    const VariableReference* varref = pm->getVariableReferenceById( matRefId );
    VariableReferenceID      geomRefId;  // Returned by haveGeometryAttributeForReference

    // Callwlate new/old input bits
    bool setChanged = m_unresolvedAttributeSet.addOrRemoveProperty( matRefId, materialAttributeWasAdded );
    bool old_M      = !materialAttributeWasAdded || !setChanged;
    bool new_M      = materialAttributeWasAdded || !setChanged;
    bool old_G      = haveGeometryAttributeForReference( varref->getVariableToken(), geomRefId );
    bool new_G      = old_G;

    // Callwlate derived sets (U = M-G)
    bool old_R = old_M && old_G;
    bool new_R = new_M && new_G;
    bool old_U = old_M && !old_G;
    bool new_U = new_M && !new_G;

    // Propagate changes if necessary
    if( old_R != new_R )
        updateResolvedAttributeSet( matRefId, geomRefId, new_R );
    if( old_U != new_U )
        updateRemainingAttributeSet( matRefId, new_U );

    if( k_scopeTrace.get() )
    {
        lprint << "U   (" << old_U << ", " << new_U << ")\n";
        lprint << "scope: GeometryInstance ID " << getScopeID()
               << " end   childMaterialAttributeDidChange material: " << material->getScopeID() << " attribute id "
               << matRefId << " " << ( materialAttributeWasAdded ? "added" : "removed" ) << "\n";
    }
}


//------------------------------------------------------------------------
// Attachment
//------------------------------------------------------------------------

void GeometryInstance::sendPropertyDidChange_Attachment( bool added ) const
{
    if( m_geometry )
        m_geometry->receivePropertyDidChange_Attachment( added );

    for( const auto& material : m_materials )
    {
        if( material )
            material->receivePropertyDidChange_Attachment( added );
    }
}

void GeometryInstance::attachmentDidChange( bool new_A )
{
    bool old_A = !new_A;

    // Unresolved references are sensitive to attachment for GI - only
    // for the postResolve set.
    for( auto refid : m_unresolvedSet_childCantResolve )
    {
        bool old_IN  = true;
        bool new_IN  = old_IN;
        bool old_V   = haveVariableForReference( refid );
        bool new_V   = old_V;
        bool old_INP = true;
        bool new_INP = old_IN;

        // Callwlate output bit
        bool old_R, old_C, old_OUT;
        computeReferenceResolutionLogic( old_IN, old_V, old_INP, old_A, old_R, old_C, old_OUT );
        bool new_R, new_C, new_OUT;
        computeReferenceResolutionLogic( new_IN, new_V, new_INP, new_A, new_R, new_C, new_OUT );

        // Propagate changes if necessary
        if( old_OUT != new_OUT )
            sendPropertyDidChange_UnresolvedReference( refid, new_OUT );
    }

    // Process the virtual parent queue, since attachment can cause new virtual parents
    m_context->getBindingManager()->processVirtualParentQueue();
}

//------------------------------------------------------------------------
// Direct Caller
//------------------------------------------------------------------------

void GeometryInstance::attachOrDetachProperty_DirectCaller( Geometry* geometry, bool added ) const
{
    // Propagate current scope's direct caller to the program
    for( auto cpid : m_directCaller )
        geometry->receivePropertyDidChange_DirectCaller( cpid, added );
}

void GeometryInstance::sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const
{
    if( m_geometry )
        m_geometry->receivePropertyDidChange_DirectCaller( cpid, added );
}

void GeometryInstance::attachOrDetachProperty_DirectCaller_toChildren( Program* program, bool added ) const
{
    for( const auto& material : m_materials )
        if( material )
            program->attachOrDetachProperty_DirectCaller( material.get(), added );
}


//------------------------------------------------------------------------
// Trace Caller
//------------------------------------------------------------------------

void GeometryInstance::receivePropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added )
{
    bool changed = m_traceCaller.addOrRemoveProperty( cpid, added );
    if( changed )
        for( const auto& material : m_materials )
            if( material )
                material->receivePropertyDidChange_TraceCaller( cpid, added );
}

void GeometryInstance::attachOrDetachProperty_TraceCaller( Material* material, bool added ) const
{
    for( auto cpid : m_traceCaller )
        material->receivePropertyDidChange_TraceCaller( cpid, added );
}


//------------------------------------------------------------------------
// HasMotionAabbs
//------------------------------------------------------------------------

// Copied from GraphNode, with custom hasMotionAabbs()

bool GeometryInstance::hasMotionAabbs() const
{
    return !m_hasMotionAabbs.empty();
}

// Called by a parent GeometryGroup as GeometryInstance is attached, to get initial info about motion aabbs in Geometry
void GeometryInstance::attachOrDetachProperty_HasMotionAabbs( AbstractGroup* group, bool attached )
{
    const bool hasMotion = hasMotionAabbs();
    // The property is a boolean (GraphPropertySingle), so we only send 'true'.  Empty property means false.
    if( hasMotion )
        group->receivePropertyDidChange_HasMotionAabbs( attached );
}

// Called by child Geometry when motion aabbs prop changes.
void GeometryInstance::receivePropertyDidChange_HasMotionAabbs( bool added )
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

void GeometryInstance::sendPropertyDidChange_HasMotionAabbs( bool added ) const
{
    // GeometryInstance cannot change this property, so just send it up
    for( auto parentLink : m_linkedPointers )
    {
        // Abstract group is the only parent, but it attaches to us as a
        // LexicalScope.
        if( AbstractGroup* parent = getLinkToLexicalScopeFrom<AbstractGroup>( parentLink ) )
            parent->receivePropertyDidChange_HasMotionAabbs( added );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to GeometryInstance" );
    }
}
