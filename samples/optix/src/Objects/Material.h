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

#include <Objects/LexicalScope.h>
#include <Objects/Program.h>  // LinkedPtr<,Program> needs to know about Program
#include <vector>

namespace optix {
class GeometryInstance;

class Material : public LexicalScope
{
    //////////////////////////////////////////////////////////////////////////
  public:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR / DTOR
    //------------------------------------------------------------------------
    Material( Context* context );
    ~Material() override;


    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    Program* getClosestHitProgram( unsigned int index ) const;
    Program* getAnyHitProgram( unsigned int index ) const;

    void setClosestHitProgram( unsigned int index, Program* program );
    void setAnyHitProgram( unsigned int index, Program* program );

    // true if an anyhit program is set for any ray type
    bool hasAtLeastOneAnyHitProgram() const;

    // Passed from context
    void setRayTypeCount( unsigned int numRayTypes );

    void validate() const override;

    //------------------------------------------------------------------------
    // LinkedPtr relationship mangement
    //------------------------------------------------------------------------
    void detachFromParents() override;
    void detachLinkedChild( const LinkedPtr_Link* link ) override;


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
    void notifyParents_closestHitProgramChanged( unsigned int rayTypeIndex ) const;
    void notifyParents_anyHitProgramChanged( unsigned int rayTypeIndex ) const;

    //------------------------------------------------------------------------
    // Unresolved reference property
    //------------------------------------------------------------------------
    // Notify all the scope parent (GeometryInstance) of changes in
    // unresolved references. Note: this is not used because the
    // preResolve and postResolve versions are used instead.
    void sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool addedToUnresolvedSet ) const override;

    void receivePropertyDidChange_UnresolvedReference( const LexicalScope* child, VariableReferenceID refid, bool addedToUnresolvedSet ) override;

    // Update the resolution sets based on the addition or removal of
    // the specified variable. Overridden from LexicalScope to implement
    // the reversed scope lookup.
    void variableDeclarationDidChange( VariableReferenceID refid, bool added ) override;

  public:
    void receivePropertyDidChange_UnresolvedReference_giCantResolve( const GeometryInstance* parent, VariableReferenceID, bool addToUnresolvedSet );

    void attachOrDetachProperty_UnresolvedReference_preResolve( GeometryInstance* gi, bool attached ) const;
    void attachOrDetachProperty_UnresolvedReference_childCantResolve( GeometryInstance* gi, bool attached ) const;

    void sendPropertyDidChange_UnresolvedReference_preResolve( VariableReferenceID refid, bool addedToUnresolvedSet ) const;
    void sendPropertyDidChange_UnresolvedReference_childCantResolve( VariableReferenceID refid, bool addedToUnresolvedSet ) const;

    // Debugging only - compute the output set.
    void computeUnresolvedOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const override;
    void computeUnresolvedGIOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const;

  private:
    //------------------------------------------------------------------------
    // Unresolved attribute property
    //------------------------------------------------------------------------
    void sendPropertyDidChange_UnresolvedAttributeReference( VariableReferenceID refid, bool added ) const;

  public:
    void attachOrDetachProperty_UnresolvedAttributeReference( GeometryInstance* gi, bool attached ) const;
    void receivePropertyDidChange_UnresolvedAttributeReference( const Program* program, VariableReferenceID, bool added );

  private:
    //------------------------------------------------------------------------
    // Attachment
    //------------------------------------------------------------------------
    // Notify children of a change in attachment
    void sendPropertyDidChange_Attachment( bool added ) const override;


    //------------------------------------------------------------------------
    // Direct Caller
    //------------------------------------------------------------------------
    void attachOrDetachProperty_DirectCaller( Program* program, bool added ) const;

  public:
    void sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const override;

  private:
    //------------------------------------------------------------------------
    // Trace Caller
    //------------------------------------------------------------------------
    void attachOrDetachProperty_TraceCaller( Program* program, bool added ) const;

  public:
    void receivePropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added );

  private:
    void updateHasAtLeastOneAnyHitProgram();

    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------
  private:
    std::vector<LinkedPtr<Material, Program>> m_closestHitPrograms;
    std::vector<LinkedPtr<Material, Program>> m_anyHitPrograms;
    bool m_hasAtLeastOneAnyHitProgram;

    // The set of unresolved references after the GI has resolved it's
    // references.
    GraphProperty<VariableReferenceID> m_unresolvedSet_giCantResolve;

    // The set of attributes that are unresolved at this scope.
    GraphProperty<VariableReferenceID> m_unresolvedAttributeSet;

    // The set of canonical programs that indirectly call through this
    // material via trace.
    GraphProperty<CanonicalProgramID> m_traceCaller;

    // Allow NodeGraph printer to access unresolved sets
    friend class NodegraphPrinter;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_MATERIAL};
};

inline bool Material::isA( ManagedObjectType type ) const
{
    return type == m_objectType || LexicalScope::isA( type );
}

}  // namespace optix
