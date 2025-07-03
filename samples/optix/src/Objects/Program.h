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

#include <LWCA/ComputeCapability.h>
#include <Device/DeviceSet.h>
#include <FrontEnd/Canonical/CallSiteIdentifier.h>
#include <FrontEnd/PTX/Canonical/CanonicalType.h>
#include <Objects/LexicalScope.h>
#include <Objects/ProgramRoot.h>
#include <Objects/SemanticType.h>
#include <prodlib/misc/String.h>

#include <string>
#include <vector>

namespace llvm {
class Function;
}

namespace optix {

struct ProgramRoot;
class CanonicalProgram;
class Device;

class Program : public LexicalScope
{
    //////////////////////////////////////////////////////////////////////////
  public:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR/DTOR
    //------------------------------------------------------------------------
    Program( Context* context );
    ~Program() override;


    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void createFromFile( const std::string& fileName, const std::string& functionName, lwca::ComputeCapability targetMax );
    void createFromFiles( const std::vector<std::string>& fileNames, const std::string& functionName, lwca::ComputeCapability targetMax );
    void createFromString( const char* ptx, const std::string& fileName, const std::string& functionName, lwca::ComputeCapability targetMax );
    void createFromStrings( const std::vector<prodlib::StringView>& ptx,
                            const std::vector<std::string>&         fileNames,
                            const std::string&                      functionName,
                            lwca::ComputeCapability                 targetMax );
    void createFromProgram( Program* program );

    // In addition to the LexicalScope ID, Programs have their own
    // ID. If the ID is queried by the API outside of OptiX, the program
    // is marked as bindless.
    int getAPIId();

    // Throw exceptions if the object is not valid
    void validate() const override;

    void setCallsitePotentialCallees( const std::string& callSiteName, const std::vector<int>& calleeIds );

    //------------------------------------------------------------------------
    // Canonical program management
    //------------------------------------------------------------------------
    // Return the canonical program for the specified device. If no
    // program is applicable, throw an exception.
    const CanonicalProgram* getCanonicalProgram( const Device* forDevice ) const;

    // Adds the given PTX code to the list of canonical programs.
    void addPTX( const prodlib::StringView& ptx,
                 const std::string&         file,
                 const std::string&         name,
                 lwca::ComputeCapability    targetMax,
                 bool                       useDiskCache = true );

    // Combines PTX code from multiple strings/files into a single canonical program.
    void addPTX( const std::vector<prodlib::StringView>& ptx,
                 const std::vector<std::string>&         files,
                 const std::string&                      name,
                 lwca::ComputeCapability                 targetMax,
                 bool                                    useDiskCache = true );

    void addLLVM( llvm::Function* func, CanonicalizationType type, lwca::ComputeCapability targetMin, lwca::ComputeCapability targetMax );

    // Called after finished adding all code variants (e.g. multiple calls to
    // addPTX/addLLVM).  No more additions can be made.
    void finishAddingCanonicalPrograms();

    // When devices are changed, ensure that there is a valid variant
    // for each program.
    void postSetActiveDevices();

  private:
    // Checks conditions before adding of a new variant to Program
    void preAddCodeCheck( const char* functionName ) const;

    // Note: CanonicalPrograms should not be added after the Program is
    // used in the wild.
    void addCanonicalProgram( const CanonicalProgram* cp );

    // This is internal and doesn't throw an exception if it doesn't find the
    // CanonicalProgram for the device.
    const CanonicalProgram* getCanonicalProgramNoThrow( const Device* device ) const;

  public:
    //------------------------------------------------------------------------
    // Internal API
    //------------------------------------------------------------------------
    // Returns a unique identifier identifying the arguments to a
    // function
    unsigned getFunctionSignature() const;

    // Returns the name of the function as originally compiled.
    std::string getInputFunctionName() const;

    // Returns the ProgramID not the lexical scope id
    int getId() const;

    // Returns true if the program has been marked as bindless anytime
    // previously.
    bool isBindless() const;

    bool isUsedAsBoundingBoxProgram() const;

    // Returns true if the program has a motion index argument.  This can only
    // possibly be true for AABB programs.
    bool hasMotionIndexArg() const;

    // The null program has special signficance in the system
    void markAsNullProgram();

    std::vector<CanonicalProgramID> getCanonicalProgramIDs();

    unsigned int get32bitAttributeKind() const;

    //------------------------------------------------------------------------
    // LinkedPtr relationship mangement
    //------------------------------------------------------------------------
    void detachFromParents() override;
    void detachLinkedChild( const LinkedPtr_Link* link ) override;


    //////////////////////////////////////////////////////////////////////////
  private:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // Object record access and management
    //------------------------------------------------------------------------
    size_t getRecordBaseSize() const override;
    void   writeRecord() const override;
    void   notifyParents_offsetDidChange() const override;
    void   offsetDidChange() const override;

  public:
    void writeHeader() const;

  private:
    //------------------------------------------------------------------------
    // Unresolved reference property
    //------------------------------------------------------------------------
    // Notify all scope parents (Geometry, Material, GraphNode,
    // GlobalScope) and virtual parents of changed in unresolved
    // references.
    void sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool addedToUnresolvedSet ) const override;

    // Unresolved references cannot change on program after
    // creation. Throw an error if this is attempted.
    void receivePropertyDidChange_UnresolvedReference( const LexicalScope* child, VariableReferenceID refid, bool addedToUnresolvedSet ) override;

  public:
    // Attach the specified parent at a particular root. This method is
    // not const because it does have side-effects in the program (for
    // tracking bound callable program references).
    virtual void attachOrDetachProperty_UnresolvedReference( LexicalScope*      scopeParent,
                                                             const ProgramRoot& root,
                                                             bool               attached,
                                                             bool               isVirtual = false );

  private:
    //------------------------------------------------------------------------
    // Unresolved attribute property
    //------------------------------------------------------------------------
  public:
    void attachOrDetachProperty_UnresolvedAttributeReference( Geometry* geometry, bool attached ) const;
    void attachOrDetachProperty_UnresolvedAttributeReference( Material* material, bool attached ) const;

  private:
    //------------------------------------------------------------------------
    // Attachment
    //------------------------------------------------------------------------
    // Notify children of a change in attachment. The children of Program are
    // CanonicalPrograms, but they don't care about attachment.
    void sendPropertyDidChange_Attachment( bool added ) const override;


    //------------------------------------------------------------------------
    // Direct Caller
    //------------------------------------------------------------------------
  public:
    void sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const override;
    void attachOrDetachProperty_DirectCaller( LexicalScope* atScope, bool added ) const;

  private:
    //------------------------------------------------------------------------
    // Semantic type property and used by raytype property
    //------------------------------------------------------------------------
  public:
    void receivePropertyDidChange_UsedAsSemanticType( SemanticType type, bool added );
    void receivePropertyDidChange_UsedByRayType( unsigned int rayType, bool added );
    void receivePropertyDidChange_InheritedSemanticType( SemanticType type, bool added );

    // Throws an exception if the SemanticType is not legal for this
    // Program
    void validateSemanticType( SemanticType stype ) const;

  private:
    //------------------------------------------------------------------------
    // Handling of bound callable programs
    //------------------------------------------------------------------------
    // Add the specified root as a virtual parent. If this is a new
    // parent, schedule the necessary property attachments.
    void addOrRemove_VirtualParent( const ProgramRoot& root, bool added );

  public:
    // Connect a virtual parent to the specified root. Note that this is
    // not called directly - but via
    // BindingManager::enqueueVirtualParentConnectOrDisconnect.
    void connectOrDisconnectProperties_VirtualParent( const ProgramRoot& root, bool connecting );

    // Receive notification (from binding manager) of a change in
    // program bindings. Used for bound callable program tracking.
    void programBindingDidChange( LexicalScope* atScope, VariableReferenceID refid, bool added );

    // Interface for BindingManager to handle forced detach of virtual
    // parents. See binding manager for hacky details.
    bool hasVirtualParent( const ProgramRoot& root ) const;
    void dropVirtualParents( const ProgramRoot& root );

    //------------------------------------------------------------------------
    // SBTRecord management for callable programs
    //------------------------------------------------------------------------
    size_t                    getSBTRecordIndex() const;
    std::vector<SemanticType> getInheritedSemanticTypes() const;

  private:
    void reallocateSbtIndex( bool added );

  private:
    //------------------------------------------------------------------------
    // Helper functions
    //------------------------------------------------------------------------
    // Marks whether the program is bindless - i.e., program's ID has
    // been released through the API
    void markAsBindless();

    // Returns true if any of the variants may call a callable program.
    bool callsBoundCallableProgram() const;


    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------
  private:
    // Not the same as the LexicalScope::m_id. Used on the device.
    ReusableID m_progId;

    typedef std::vector<const CanonicalProgram*> VariantListType;
    VariantListType                              m_canonicalPrograms;

    bool m_isBindless                      = false;
    bool m_finishedAddingCanonicalPrograms = false;

    // Attributes propagated from canonical programs
    GraphProperty<VariableReferenceID> m_unresolvedAttributeSet;

    // Annotated set of program references for programs with bound
    // callable programs.
    struct Annotation
    {
        int count = 0;
        std::map<VariableReferenceID, VariableReferenceID> programReferences;
        VariableReferenceID mapReference( VariableReferenceID ) const;
    };
    std::map<ProgramRoot, Annotation> m_rootAnnotations;

    // The set of virtual parents for programs that are used as bound
    // callable programs.
    GraphProperty<ProgramRoot> m_virtualParents;

    // When a program is added as a virtual child, we keep track of the
    // semantic type inherited by the calling semantic type.  If this
    // set changes, we need to subscribe to validation.  We can't
    // immediately validate since there may be intermediate illegal
    // states of the graph before launch.
    GraphProperty<SemanticType> m_inheritedSemanticType;

    // Tracks whether the program is used as a bounding box program.
    GraphPropertySingle<> m_usedAsBoundingBoxProgram;

    GraphPropertySingle<> m_usedAsBoundCallableProgram;

    mutable std::shared_ptr<size_t> m_SBTIndex;

    std::map<std::string, std::vector<CanonicalProgramID>> m_callSitePotentialCallees;

    // Nodegraph can see variant list
    friend class NodegraphPrinter;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_PROGRAM};
};

inline bool Program::isA( ManagedObjectType type ) const
{
    return type == m_objectType || LexicalScope::isA( type );
}

}  // namespace optix
