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
class Context;
class GeometryInstance;

class GlobalScope : public LexicalScope
{
    //////////////////////////////////////////////////////////////////////////
  public:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR / DTOR
    //------------------------------------------------------------------------
    GlobalScope( Context* context );
    ~GlobalScope() override;


    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    // Control # of ray types and entry points active
    void setRayTypeCount( unsigned int numRaytypes );
    void setEntryPointCount( unsigned int numEntryPoints );

    /* set children */
    void setRayGenerationProgram( unsigned int index, Program* child );
    void setExceptionProgram( unsigned int index, Program* child );
    void setMissProgram( unsigned int index, Program* child );
    void setAabbComputeProgram( Program* child );
    void setAabbExceptionProgram( Program* child );

    /* get children */
    Program* getRayGenerationProgram( unsigned int index ) const;
    Program* getExceptionProgram( unsigned int index ) const;
    Program* getMissProgram( unsigned int index ) const;
    Program* getAabbComputeProgram() const;
    Program* getAabbExceptionProgram() const;

    // Includes AABB iterator. The number of user specified entry points
    // can be had from Context
    unsigned int getAllEntryPointCount() const;

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

  public:
    // GlobalScope overrides reallocate so that it can ensure a start address of zero.
    void reallocateRecord() override;

  private:
    //------------------------------------------------------------------------
    // Unresolved reference property
    //------------------------------------------------------------------------
    // Notify all the scope parent (GlobalScope) of changes* in
    // unresolved references.
    void sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool addedToUnresolvedSet ) const override;

  public:
    // The set of references remaining after variable resolution. Launching with any
    // unresolved references is not legal.
    const GraphProperty<VariableReferenceID>& getRemainingUnresolvedReferences() const;

  private:
    //------------------------------------------------------------------------
    // Attachment
    //------------------------------------------------------------------------
    // Notify children of a change in attachment
    void sendPropertyDidChange_Attachment( bool added ) const override;


    //------------------------------------------------------------------------
    // Direct Caller
    //------------------------------------------------------------------------
  public:
    void sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const override;

  private:
    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------
  private:
    // Ray generation programs and exception programs have one
    // additional entry for the built-in compute_aabb program.
    static const int numInternalEntryPoints = 1;
    std::vector<LinkedPtr<GlobalScope, Program>> m_rayGenerationPrograms;
    std::vector<LinkedPtr<GlobalScope, Program>> m_exceptionPrograms;
    std::vector<LinkedPtr<GlobalScope, Program>> m_missPrograms;
    LinkedPtr<GlobalScope, Program>              m_aabbComputeProgram;
    LinkedPtr<GlobalScope, Program>              m_aabbExceptionProgram;

    // The set of variables that are unresolved at the output from this
    // scope. Marked mutable so that the sendPropertyDidChange can
    // modify.
    mutable GraphProperty<VariableReferenceID> m_unresolvedRemaining;

    // The set of attributes that are unresolved at any GI
    GraphProperty<VariableReferenceID> m_unresolvedAttributesRemaining;
};

}  // namespace optix
