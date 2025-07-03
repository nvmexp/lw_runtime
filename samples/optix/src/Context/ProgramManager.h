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

#include <AS/Traversers.h>
#include <LWCA/ComputeCapability.h>
#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <FrontEnd/PTX/Canonical/CanonicalType.h>
#include <FrontEnd/PTX/DataLayout.h>
#include <Objects/SemanticType.h>
#include <corelib/misc/Concepts.h>
#include <prodlib/misc/String.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace llvm {
class Function;
class FunctionType;
class Module;
}

namespace optix {

struct ProgramRoot;
class CallSiteIdentifier;
class CanonicalProgram;
class Context;
class VariableType;
class Program;
class VariableReference;

/*
 * Manage canonical programs, including:
 *   Unique copies of PTX
 *   Raw LLVM translations of PTX
 *   Canonicalized LLVM representations of each function
 *   Future: Native LLVM functions and their canonicalized versions
 *
 * Owns all canonical programs. Tracks information which applies to the
 * collection of canonical programs, such as variable references.
 */

class ProgramManager : private corelib::NonCopyable
{
  public:
    ProgramManager( Context* context );
    ~ProgramManager();

    CanonicalProgram* canonicalizeFunction( llvm::Function*         fn,
                                            CanonicalizationType    type,
                                            lwca::ComputeCapability targetMin,
                                            lwca::ComputeCapability targetMax,
                                            size_t                  ptxHash );
    const CanonicalProgram* canonicalizePTX( const std::vector<prodlib::StringView>& ptxStrings,
                                             const std::vector<std::string>&         filenames,
                                             const std::string&                      functionName,
                                             lwca::ComputeCapability                 targetMax,
                                             bool                                    useDiskCache = true );

    const CanonicalProgram* getCanonicalProgramById( CanonicalProgramID id ) const;

    // Return the map from IDs to known canonical programs.
    typedef ReusableIDMap<CanonicalProgram*> CanonicalProgramMap;
    const CanonicalProgramMap&               getCanonicalProgramMap() const { return m_idMap; }

    // Manage variable references
    const VariableReference* getVariableReferenceById( VariableReferenceID id ) const;
    const VariableReference* getVariableReferenceByUniversallyUniqueName( const std::string& uuname ) const;
    typedef std::vector<VariableReferenceID> VariableReferenceIDListType;
    const VariableReferenceIDListType& getReferencesForVariable( unsigned short vartoken ) const;
    int numberOfAssignedReferences();

    // Virtual variable references for bound callable programs
    const VariableReference* registerVirtualVariableReference( const ProgramRoot& root, const VariableReference* varref );
    void removeVirtualVariableReference( const VariableReference* varref );

    // Used by canonicalization only
    ReusableID createCanonicalProgramId( CanonicalProgram* program );
    void registerVariableReference( VariableReference* varref );

    void registerCallSite( CallSiteIdentifier* csId );
    CallSiteIdentifier* getCallSiteByUniqueName( const std::string& csName ) const;

    // Register a FunctionType and return a permanent unique identifier token, not necessarily sequential.
    unsigned registerFunctionType( llvm::FunctionType* ftype );
    std::string getFunctionSignatureForToken( unsigned token ) const;

    // Whenever a Program adds/removes ptx that has an ill-formed/escaped bindless buffer pointer
    // We need to change the policy of all bindless buffers when this count switches between zero and non-zero
    bool hasRawBindlessBufferAccesses() const;
    void addOrRemoveRawBindlessBufferAccesses( bool added );

  private:
    Context*                             m_context;
    std::vector<const CanonicalProgram*> m_canonicalPrograms;  // Expected not to grow too large.
    CanonicalProgramMap                  m_idMap;

    struct PTXModule
    {
        typedef std::map<std::string, const CanonicalProgram*> CanonicalProgramByFunctionName;

        size_t                         m_hash = 0;
        lwca::ComputeCapability        m_targetMin;
        lwca::ComputeCapability        m_targetMax;
        llvm::Module*                  m_rawModule = nullptr;
        CanonicalProgramByFunctionName m_canonicalPrograms;
        PTXModule( size_t hash, lwca::ComputeCapability targetMin, lwca::ComputeCapability targetMax );
    };
    typedef std::map<size_t, std::shared_ptr<PTXModule>> PTXModuleMap;  // hash -> PTXModule
    PTXModuleMap m_ptxModules;

    typedef std::vector<VariableReference*> VariableReferenceListType;
    VariableReferenceListType               m_variableReferences;
    std::map<std::string, VariableReference*> m_variableReferencesByUniqueName;
    typedef std::vector<VariableReferenceIDListType*> VariableReferenceIlwerseType;
    VariableReferenceIlwerseType                      m_ilwerseReferences;
    std::vector<VariableReferenceID>                  m_emptyList;

    std::map<std::string, CallSiteIdentifier*> m_callSiteByUniqueName;

    std::map<unsigned, std::string> m_functionSignatures;

    // Total count of canonical programs that access a bindless buffer in a raw way
    int m_rawBindlessBufferAccesses = 0;

    // Caching methods
    static const uint32_t m_teaKey[4];
    llvm::Function* findFunction( llvm::Module* module, const std::string& functionName ) const;
    std::string constructDiskCacheKey( size_t hash, const std::string& functionName ) const;
    CanonicalProgram* loadCanonicalProgramFromDiskCache( const std::string& cacheKey, size_t hash );
    void saveCanonicalProgramToDiskCache( const std::string& cacheKey, CanonicalProgram* cp );
    void resetVariableReferences( unsigned int beginReferenceID );
};
}
