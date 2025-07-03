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

#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <Objects/GraphProperty.h>
#include <Objects/ManagedObject.h>
#include <Objects/ObjectClass.h>
#include <Objects/SemanticType.h>
#include <Util/LinkedPtr.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace optix {
class GeometryInstance;
class GraphNode;
class Program;
class TextureSampler;
class Variable;
class VariableType;

class LexicalScope : public ManagedObject
{
    //////////////////////////////////////////////////////////////////////////
  public:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR/DTOR
    //------------------------------------------------------------------------
  protected:
    LexicalScope( Context* context, ObjectClass objClass );

  public:
    ~LexicalScope() override;


    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    Variable* declareVariable( const std::string& name );
    void removeVariable( Variable* variable );
    Variable* queryVariable( const std::string& name ) const;
    Variable* getVariableByIndex( unsigned int index ) const;
    unsigned int getVariableCount() const;

    // Throws an exception if object is invalid
    virtual void validate() const;


    //------------------------------------------------------------------------
    // Internal API
    //------------------------------------------------------------------------
    // Each LexicalScope has an ID that uniquely identifies the object
    // over it's lifetime.
    int getScopeID() const;

    Variable* getOrDeclareVariable( const std::string& name );
    Variable* getOrDeclareVariable( const std::string& name, const VariableType& type );
    Variable* getVariableByToken( unsigned short token ) const;


    //------------------------------------------------------------------------
    // LinkedPtr relationship mangement
    //------------------------------------------------------------------------
    virtual void detachFromParents()                             = 0;
    virtual void detachLinkedChild( const LinkedPtr_Link* link ) = 0;
    virtual void childOffsetDidChange( const LinkedPtr_Link* link );


    //------------------------------------------------------------------------
    // Notifications from variable (used ONLY by variable)
    //------------------------------------------------------------------------
    void variableSizeDidChange( Variable* var );
    void variableTypeDidChange( Variable* var );
    void variableValueDidChange( Variable* var );
    void variableValueDidChange( Variable* var, int oldValue, int newValue );
    void variableValueDidChange( Variable* var, GraphNode* oldNode, GraphNode* newNode );
    void variableValueDidChange( Variable* var, Buffer* oldBuffer, Buffer* newBuffer );
    void variableValueDidChange( Variable* var, TextureSampler* oldTexture, TextureSampler* newTexture );
    void variableValueDidChange( Variable* var, Program* oldProgram, Program* newProgram );
    static size_t getSafeOffset( const LexicalScope* scope );

    //------------------------------------------------------------------------
    // Index for validation list
    //------------------------------------------------------------------------
    struct validationIndex_fn
    {
        int& operator()( const LexicalScope* scope ) { return scope->m_validationIndex; }
    };


    //////////////////////////////////////////////////////////////////////////
  protected:
    //////////////////////////////////////////////////////////////////////////

    // Used by PostprocessingStage.
    virtual void bufferVariableValueDidChange( Variable* var, Buffer* oldBuffer, Buffer* newBuffer );

    //------------------------------------------------------------------------
    // Object record management
    //------------------------------------------------------------------------
    virtual void   writeRecord() const;
    void           writeObjectRecord() const;
    char*          getObjectRecord() const;
    virtual size_t getRecordBaseSize() const = 0;
    unsigned short getDynamicVariableTableCount() const;
    virtual void   notifyParents_offsetDidChange() const = 0;
    virtual void   offsetDidChange() const;

  public:
    size_t       getRecordOffset() const;
    size_t       getRecordSize() const;
    bool         recordIsAllocated() const;
    void         releaseRecord();
    virtual void reallocateRecord();  // GlobalScope overrides
    template <class T>
    T* getObjectRecord() const
    {
        return reinterpret_cast<T*>( getObjectRecord() );
    }


  private:
    //------------------------------------------------------------------------
    // DynamicVariableTable management
    //------------------------------------------------------------------------
    // So far, all member functions are private and non-virtual
    void  reallocateDynamicVariableTable();
    void  writeDynamicVariableTable() const;
    char* getDynamicVariableTable() const;
    bool  dynamicVariableTableIsAllocated() const;
    void  releaseDynamicVariableTable();

  protected:
    //------------------------------------------------------------------------
    // Unresolved reference property
    //------------------------------------------------------------------------
    // Notify parent scopes about addition/deletion in the unresolved
    // variable set by calling childUnresolvedReferenceDidChange for
    // each parent. Note: this is not const because some scope types
    // need to track ancillary information (global scope, program)
    virtual void sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool added ) const = 0;

    // Add the specified reference to the resolution set indicating
    // that the reference is resolved by this scope.  It is illegal to
    // add to this set more than once or remove a non-existing entry.
    void updateResolvedSet( VariableReferenceID refid, bool addToUnresolvedSet );

    // Update the resolution sets based on the addition or removal of
    // the specified variable by iterating over all known references of
    // the variable and calling variableDeclarationDidChange for those
    // references.
    void variableDeclarationDidChange( Variable* var, bool variableWasAdded );

    // Update the resolution sets based on the addition or removal of
    // the specified reference.  LexicalScope provides a default
    // implementation but Geometry, Material and GeometryInstance
    // provide an override.
    virtual void variableDeclarationDidChange( VariableReferenceID refid, bool added );

    // Debugging only - compute the output set.
    virtual void computeUnresolvedOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const;

  public:
    // Returns true if there is lwrrently a resolution for the specified
    // reference ID.
    bool haveResolutionForReference( VariableReferenceID refid ) const;

    // The set of references resolved at this scope. Used by validation
    // to perform type-checking of resolved variables.
    const GraphProperty<VariableReferenceID, false>& getResolvedReferenceSet() const;

    // Receive notification that a child scope has changed their
    // unresolved reference set.  LexicalScope provides a default
    // implementation for most object types, but Geometry, Material and
    // GeometryInstance provide an override to implement the reverse
    // lookup.
    virtual void receivePropertyDidChange_UnresolvedReference( const LexicalScope* child, VariableReferenceID refid, bool added );

  protected:
    //------------------------------------------------------------------------
    // Attachment property
    //------------------------------------------------------------------------
    // Notify children of a change in entrypoint reachability.
    virtual void sendPropertyDidChange_Attachment( bool added ) const = 0;

    // Local notification that the overall attachment of this node
    // changed. This is used to control whether unresolved references
    // proceed to GlobalScope from Selector and GeometryInstance (only).
    virtual void attachmentDidChange( bool newAttached ) {}

    // Notify variables that care about attachment
    void sendPropertyDidChangeToVariables_Attachment( bool added ) const;

  public:
    // Returns true if this node is attached on the graph (through children or variables)
    //  - Skip validation of unattached nodes in ValidationManager
    //  - Block Unresolved references propagating to GlobalScope from
    //    Selector and GeometryInstance nodes
    bool isAttached() const;

    // Propagate attachment to the specified scope, buffer or texture sampler
    void attachOrDetachProperty_Attachment( LexicalScope* linkHead, bool attached ) const;
    void attachOrDetachProperty_Attachment( Buffer* buffer, bool attached ) const;
    void attachOrDetachProperty_Attachment( TextureSampler* texture, bool attached ) const;

    // Receive notification that a parent's attachment property changed
    void receivePropertyDidChange_Attachment( bool added );

  protected:
    //------------------------------------------------------------------------
    // RtxUniversalTraversal property
    //------------------------------------------------------------------------

    // Notify children of a change in traversal mode
    virtual void sendPropertyDidChange_RtxUniversalTraversal() const {};

    // Local notification that the the traversal mode of this node
    // changed. This is used to update traversal data to the required
    // format.
    virtual void rtxUniversalTraversalDidChange() {}

  public:
    // Receive notification that a parent's RtxUniversalTraversal property changed
    void receivePropertyDidChange_RtxUniversalTraversal();

    // Propagate attachment to the specified scope
    void attachOrDetachProperty_RtxUniversalTraversal( LexicalScope* linkHead, bool attached ) const;

  protected:
    //------------------------------------------------------------------------
    // Direct Caller
    //------------------------------------------------------------------------
    virtual void sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const = 0;

  public:
    void receivePropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added );

  protected:
    //------------------------------------------------------------------------
    // Helper functions
    //------------------------------------------------------------------------
    // Helper function to avoid having to include Context.h and ProgramManager.h to get
    // the NullProgram.
    Program* getSharedNullProgram() const;

    // Notify the validation manager that this object needs to be validated on next launch
    void subscribeForValidation();

    // Resize a vector, calling the shrink closure for entries that will
    // disappear and calling the fill closure for entries that will be
    // added.
    template <typename T, typename ShrinkClosure, typename FillClosure>
    static void resizeVector( std::vector<T>& vec, unsigned int newSize, ShrinkClosure shrink, FillClosure fill );

    // Returns true if we have a variable that matches the specified
    // reference ID (regardless of whether it is resolved).
    bool haveVariableForReference( VariableReferenceID refid ) const;

    // During destruction in derived types, it is useful to delete all the Variables before
    // any other objects in the class are deleted.  One example is Acceleration which
    // contains variables pointing to Buffers which are held by the Builder.  When the
    // Builder is destroyed in ~Acceleration, if the variables haven't been removed you get
    // errors about the Buffer having existing links to it.
    void deleteVariables();

    // Emit a log message if the scopeTrace property is set
    void scopeTrace( const char* msg, VariableReferenceID refid, bool attached, const LexicalScope* fromScope = nullptr ) const;


    //////////////////////////////////////////////////////////////////////////
  private:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // Helper functions
    //------------------------------------------------------------------------
    // Makes sure the name follows variable naming colwentions
    void checkName( const std::string& name ) const;


    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------
  private:
    // The ID (reused when object is destroyed)
    ReusableID m_id;

    // Object record and dynamic variable table allocation and sizes
    std::shared_ptr<size_t> m_recordOffset;
    std::shared_ptr<size_t> m_dynamicVariableTableOffset;
    size_t                  m_recordSize               = 0;
    size_t                  m_dynamicVariableTableSize = 0;

    // Variables
    typedef std::vector<Variable*> VariableArrayType;
    VariableArrayType              m_variables;
    typedef std::map<unsigned short, Variable*> VariableMapType;  // maps token to Variable*
    VariableMapType  m_tokenToVarMap;
    static const int ILWALID_OFFSET = -1;

    // The set of variables that are resolved by this scope.  This is
    // computed by calls to updateResolvedSet by derived classes who can
    // have custom resolution logic (more than IN - V). This set does
    // not need to be counted.
    GraphProperty<VariableReferenceID, false> m_resolvedSet;

  protected:
    // The set of variables that are unresolved at the input to this
    // scope.
    GraphProperty<VariableReferenceID> m_unresolvedSet;

    // Count the number of node graph attachments from attached objects
    GraphPropertySingle<> m_attachment;

    // The set of CanonicalProgram IDs that call the program attached to this object
    GraphProperty<CanonicalProgramID> m_directCaller;

    // Node needs to be in universal format.
    bool m_rtxUniversalTraversal = false;

    // Used by IndexedVector in Validationmanager. Mutable so that scope
    // can remain const in validation manager.
    mutable int m_validationIndex = -1;

    // Let NodegraphPrinter see the reference sets
    friend class NodegraphPrinter;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_LEXICAL_SCOPE};
};


template <typename T, typename ShrinkClosure, typename FillClosure>
void LexicalScope::resizeVector( std::vector<T>& vec, unsigned int new_size, ShrinkClosure shrink, FillClosure fill )
{
    size_t old_size = vec.size();

    // Reset old elements with the shrink functor
    for( size_t i = new_size; i < old_size; ++i )
        shrink( (int)i );

    vec.resize( new_size );

    // Set new elements with the shrink functor
    for( size_t i = old_size; i < new_size; ++i )
        fill( (int)i );
}

inline bool LexicalScope::isA( ManagedObjectType type ) const
{
    return type == m_objectType || ManagedObject::isA( type );
}

}  // namespace optix
