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

namespace llvm {
class Type;
}

namespace optix {
class Context;
class GeometryInstance;

class Geometry : public LexicalScope
{
    //////////////////////////////////////////////////////////////////////////
  public:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR / DTOR
    //------------------------------------------------------------------------
    Geometry( Context* context, bool isDerivedClass );
    ~Geometry() override;


    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void setIntersectionProgram( Program* program );
    void setBoundingBoxProgram( Program* program );
    Program* getIntersectionProgram() const;
    Program* getBoundingBoxProgram() const;

    void setPrimitiveCount( int primitiveCount );
    int getPrimitiveCount() const;

    void setPrimitiveIndexOffset( int primitiveIndexOffset );
    int getPrimitiveIndexOffset() const;

    // Motion blur data
    void setMotionSteps( int n );
    int getMotionSteps() const;

    void setMotionRange( float timeBegin, float timeEnd );
    void getMotionRange( float& timeBegin, float& timeEnd ) const;

    void setMotionBorderMode( RTmotionbordermode beginMode, RTmotionbordermode endMode );
    void getMotionBorderMode( RTmotionbordermode& beginMode, RTmotionbordermode& endMode ) const;

    void                    setFlags( RTgeometryflags );
    virtual RTgeometryflags getFlags() const;

    // Dirty bits on geometry do not have any effect as of OptiX
    // 4.0. Track the dirty bit, which will never get cleared.
    void markDirty();
    bool isDirty() const;

    // Throw exceptions if the object is not valid
    void validate() const override;


    //------------------------------------------------------------------------
    // LinkedPtr relationship management
    //------------------------------------------------------------------------
    void detachFromParents() override;
    void detachLinkedChild( const LinkedPtr_Link* link ) override;


    //////////////////////////////////////////////////////////////////////////
  protected:
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
    void notifyParents_intersectionProgramDidChange() const;

    //------------------------------------------------------------------------
    // Unresolved reference property
    //------------------------------------------------------------------------
    // Notify all the scope parent (GeometryInstance) of changes in
    // unresolved references. Note: this is not used because the
    // preResolve and postResolve versions are used instead.
    void receivePropertyDidChange_UnresolvedReference( const LexicalScope* child, VariableReferenceID refid, bool added ) override;
    void sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool addedToUnresolvedSet ) const override;

    // Send and receive the sets that implement the reversed lookup.
    void sendPropertyDidChange_UnresolvedReference_preResolve( VariableReferenceID refid, bool addedToUnresolvedSet ) const;
    void sendPropertyDidChange_UnresolvedReference_childCantResolve( VariableReferenceID refid, bool addedToUnresolvedSet ) const;

    // Update the resolution sets based on the addition or removal of
    // the specified variable. Overridden from LexicalScope to implement
    // the reversed scope lookup.
    void variableDeclarationDidChange( VariableReferenceID refid, bool added ) override;

  public:
    // Used by GI to manage reverse scope lookup
    void attachOrDetachProperty_UnresolvedReference_preResolve( GeometryInstance* gi, bool attached ) const;
    void attachOrDetachProperty_UnresolvedReference_childCantResolve( GeometryInstance* gi, bool attached ) const;
    void receivePropertyDidChange_UnresolvedReference_giCantResolve( const GeometryInstance* parent, VariableReferenceID, bool added );

    // Debugging only - compute the output set.
    void computeUnresolvedOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const override;
    void computeUnresolvedGIOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const;

  protected:
    //------------------------------------------------------------------------
    // Unresolved attribute property
    //------------------------------------------------------------------------
    // Returns the set of attributes that are unresolved at this scope.
    // Used by GeometryInstance to avoid making a copy of the INA
    // (input) set on the Geometry child. This works because the Program
    // does not resolve any attributes.
    void sendPropertyDidChange_UnresolvedAttributeReference( VariableReferenceID refid, bool added ) const;

  public:
    const GraphProperty<VariableReferenceID>& getUnresolvedAttributeSet() const;
    void attachOrDetachProperty_UnresolvedAttributeReference( GeometryInstance* gi, bool attached ) const;
    void receivePropertyDidChange_UnresolvedAttributeReference( const Program* program, VariableReferenceID, bool added );

  protected:
    //------------------------------------------------------------------------
    // Attachment
    //------------------------------------------------------------------------
    // Notify children of a change in attachment
    void sendPropertyDidChange_Attachment( bool added ) const override;


    //------------------------------------------------------------------------
    // Direct Caller
    //------------------------------------------------------------------------
    void sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const override;
    void attachOrDetachProperty_DirectCaller( Program* program, bool added ) const;

    //------------------------------------------------------------------------
    // HasMotionAabbs
    //------------------------------------------------------------------------
    void sendPropertyDidChange_HasMotionAabbs( bool added ) const;
    bool hasMotionAabbs() const;

  public:
    void attachOrDetachProperty_HasMotionAabbs( GeometryInstance* gi, bool attached ) const;


    //------------------------------------------------------------------------
    // Allow children classes a chance to do something when the intersection program changes
    //------------------------------------------------------------------------
  protected:
    virtual void intersectionProgramDidChange( Program* program, bool added ){};

    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------
  protected:
    LinkedPtr<Geometry, Program> m_intersectionProgram;
    LinkedPtr<Geometry, Program> m_boundingBoxProgram;

    int                m_motionSteps     = 1;
    RTmotionbordermode m_beginBorderMode = RT_MOTIONBORDERMODE_CLAMP;
    RTmotionbordermode m_endBorderMode   = RT_MOTIONBORDERMODE_CLAMP;
    float              m_timeBegin       = 0.0f;
    float              m_timeEnd         = 1.0f;

    int             m_primitiveCount       = 0;
    int             m_primitiveIndexOffset = 0;
    bool            m_dirty                = true;
    bool            m_initialized          = false;
    RTgeometryflags m_flags                = RT_GEOMETRY_FLAG_NONE;

    // The set of unresolved references after the GI has resolved it's
    // references.
    GraphProperty<VariableReferenceID> m_unresolvedSet_giCantResolve;

    // The set of attributes that are unresolved at this scope.
    GraphProperty<VariableReferenceID> m_unresolvedAttributeSet;

    // Allow NodeGraph printer to access unresolved sets
    friend class NodegraphPrinter;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_GEOMETRY};
};

inline bool Geometry::isA( ManagedObjectType type ) const
{
    return type == m_objectType || LexicalScope::isA( type );
}

}  // namespace optix
