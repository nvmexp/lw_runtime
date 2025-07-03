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

#include <Objects/Geometry.h>  // LinkedPtr<,Geometry> needs to know about Geometry
#include <Objects/LexicalScope.h>
#include <Objects/Material.h>  // LinkedPtr<,Material> needs to know about Material
#include <vector>

namespace optix {

class AbstractGroup;

class GeometryInstance : public LexicalScope
{
    //////////////////////////////////////////////////////////////////////////
  public:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR/DTOR
    //------------------------------------------------------------------------
  public:
    GeometryInstance( Context* context );
    ~GeometryInstance() override;


    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    // Geometry
    void      setGeometry( Geometry* );
    Geometry* getGeometry() const;


    // Materials
    void setMaterial( unsigned int index, Material* material );
    Material* getMaterial( unsigned int index );
    const Material* getMaterial( unsigned int index ) const;

    void setMaterialCount( int n );
    int getMaterialCount() const;

    void validate() const override;


    //------------------------------------------------------------------------
    // LinkedPtr relationship mangement
    //------------------------------------------------------------------------
    void detachFromParents() override;
    void detachLinkedChild( const LinkedPtr_Link* link ) override;
    void childOffsetDidChange( const LinkedPtr_Link* link ) override;

    //////////////////////////////////////////////////////////////////////////
  private:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // Object record management
    //------------------------------------------------------------------------
    size_t getRecordBaseSize() const override;
    void   writeRecord() const override;
    void   notifyParents_offsetDidChange() const override;

    //------------------------------------------------------------------------
    // SBTRecord management
    //------------------------------------------------------------------------
  private:
    void notifyParents_materialCountDidChange() const;
    void notifyParents_geometryDidChange() const;
    void notifyParents_materialDidChange( unsigned int materialIndex ) const;
    void notifyParents_materialOffsetDidChange( unsigned int materialIndex ) const;

  public:
    void geometryIntersectionProgramDidChange( const LinkedPtr_Link* link );
    void materialClosestHitProgramDidChange( const LinkedPtr_Link* link, unsigned int rayTypeIndex );
    void materialAnyHitProgramDidChange( const LinkedPtr_Link* link, unsigned int rayTypeIndex );

    //------------------------------------------------------------------------
    // Unresolved reference property.
    //------------------------------------------------------------------------
  private:
    void sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool addedToUnresolvedSet ) const override;
    void receivePropertyDidChange_UnresolvedReference( const LexicalScope* child, VariableReferenceID refid, bool addedToUnresolvedSet ) override;

    //
    // Note: two sets are maintained to implement the reversed GI /
    // material lookup. To ensure that this protocol is robust, the
    // normal propagation methods are not used. Instead, the
    // {preResolve, midResolve, postResolve} versions are used.
    void attachOrDetachProperty_UnresolvedReference_giCantResolve( Geometry* geometry, bool attached ) const;
    void attachOrDetachProperty_UnresolvedReference_giCantResolve( Material* material, bool attached ) const;
    void sendPropertyDidChange_UnresolvedReference_giCantResolve( VariableReferenceID refid, bool added ) const;

    // Update the resolution sets based on the addition or removal of
    // the specified variable. Overridden from LexicalScope to implement
    // the reversed scope lookup.
    void variableDeclarationDidChange( VariableReferenceID refid, bool added ) override;

  public:
    void receivePropertyDidChange_UnresolvedReference_preResolve( const LexicalScope* scope, VariableReferenceID refid, bool added );
    void receivePropertyDidChange_UnresolvedReference_childCantResolve( const LexicalScope* scope, VariableReferenceID refid, bool added );

    // Debugging only - compute the output set.
    void computeUnresolvedOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const override;
    void computeUnresolvedChildOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const;

  private:
    //------------------------------------------------------------------------
    // Unresolved attribute property
    // ------------------------------------------------------------------------
    // Keep track of the attributes with no resolution. Must be empty
    // before launch.
    void updateRemainingAttributeSet( VariableReferenceID matRefId, bool addToResolvedSet );

    // Track the attributes resolved in the scope
    void updateResolvedAttributeSet( VariableReferenceID matRefId, VariableReferenceID geomVarRef, bool addToResolvedSet );

    // Returns true if we have at least one geometry attribute that matches the
    // specifed reference ID.
    bool haveGeometryAttributeForReference( unsigned short token, VariableReferenceID& geomRefId ) const;

  public:
    void receivePropertyDidChange_UnresolvedAttributeReference( const Geometry* geometry, VariableReferenceID refid, bool added );
    void receivePropertyDidChange_UnresolvedAttributeReference( const Material* material, VariableReferenceID refid, bool added );

  private:
    //------------------------------------------------------------------------
    // Attachment
    //------------------------------------------------------------------------
    // Notify children of a change in attachment
    void sendPropertyDidChange_Attachment( bool added ) const override;

    // Output set resolution is affected by attachment at the
    // GeometryInstance so that unresolved references do not propagate
    // to the global scope for nodes not attached. See also Selector.
    void attachmentDidChange( bool newAttached ) override;


    //------------------------------------------------------------------------
    // Direct Caller
    //------------------------------------------------------------------------
    void attachOrDetachProperty_DirectCaller( Geometry* geometry, bool added ) const;

  public:
    void sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const override;
    void attachOrDetachProperty_DirectCaller_toChildren( Program* program, bool added ) const;

  private:
    //------------------------------------------------------------------------
    // Trace Caller
    //------------------------------------------------------------------------
    void attachOrDetachProperty_TraceCaller( Material* material, bool added ) const;

  public:
    void receivePropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added );

  private:
    //------------------------------------------------------------------------
    // HasMotionAabbs
    //------------------------------------------------------------------------
    bool hasMotionAabbs() const;
    void sendPropertyDidChange_HasMotionAabbs( bool added ) const;

  public:
    void attachOrDetachProperty_HasMotionAabbs( AbstractGroup* group, bool attached );
    void receivePropertyDidChange_HasMotionAabbs( bool added );

  private:
    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------
  private:
    LinkedPtr<GeometryInstance, Geometry>              m_geometry;
    std::vector<LinkedPtr<GeometryInstance, Material>> m_materials;

    // The set of unresolved references that children {Geometry,Material} can't resolve.
    GraphProperty<VariableReferenceID> m_unresolvedSet_childCantResolve;

    // Unresolved attributes from material children
    GraphProperty<VariableReferenceID> m_unresolvedAttributeSet;

    // The map of material attribute -> geometry attributes for this scope.
    std::map<VariableReferenceID, VariableReferenceID> m_resolvedAttributes;

    // The set of attribute references that remain unresolved resolved
    // at this scope after binding geometry and material
    // attributes. Must be empty at launch.
    GraphProperty<VariableReferenceID, false> m_unresolvedAttributesRemaining;

    // The set of canonical programs that indirectly call through this
    // GI via trace.
    GraphProperty<CanonicalProgramID> m_traceCaller;

    GraphPropertySingle<int> m_hasMotionAabbs;

    // Let NodegraphPrinter see the reference sets
    friend class NodegraphPrinter;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_GEOMETRY_INSTANCE};
};

inline bool GeometryInstance::isA( ManagedObjectType type ) const
{
    return type == m_objectType || LexicalScope::isA( type );
}

}  // namespace optix
