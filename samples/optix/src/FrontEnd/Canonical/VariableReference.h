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

#include <Compile/TextureLookup.h>
#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <Objects/ProgramRoot.h>
#include <Objects/VariableType.h>

#include <string>
#include <vector>

namespace optix {
class PersistentStream;
class CanonicalProgram;

class VariableReference
{
  public:
    ~VariableReference();
    const CanonicalProgram* getParent() const;
    VariableReferenceID     getReferenceID() const;
    const std::string&      getInputName() const;
    unsigned short          getVariableToken() const;
    const VariableType&     getType() const;

    // Query access properties of the variable
    bool isInitialized() const;
    bool usesTextureLookupKind( TextureLookup::LookupKind kind ) const;
    bool canPointerEscape() const { return m_pointerMayEscape; }
    bool hasBufferStores() const { return m_hasBufferStores; }
    bool hasIllFormedAccess() const { return m_hasIllFormedAccess; }

    // Print the reference in the form id:token:name, useful for output
    std::string getInfoString() const;
    // Print the reference in the form {cp.uuname}.{variablename},
    // where cp.uuname is the univerally unique name of the associated
    // canonical program. This is used for symbol names in
    // canonicalization.
    std::string getUniversallyUniqueName() const;

    // Hooks for bound callable programs
    const ProgramRoot&       getBoundProgramRoot() const;
    const VariableReference* getLinkedReference() const;

    // Persistence support - including constructor to create empty variable reference.
    VariableReference( const CanonicalProgram* parent );
    friend void readOrWrite( PersistentStream* stream, VariableReference*, const char* label );

  private:
    //
    // WARNING: This is a persistent class. If you change anything you
    // should also update the readOrWrite function and bump the the
    // version number.
    //
    const CanonicalProgram* m_parent = nullptr;
    VariableReferenceID     m_refid;
    std::string             m_variableName;
    unsigned short          m_variableToken;
    VariableType            m_vtype;
    bool                    m_isInitialized;
    bool                    m_pointerMayEscape   = false;
    bool                    m_hasBufferStores    = false;
    bool                    m_hasIllFormedAccess = false;
    std::string             m_annotation;
    std::vector<bool>       m_textureLookupKinds;

    // Members for bound program references (virtual references created for each root)
    const VariableReference* m_linkedReference = nullptr;
    ProgramRoot              m_root;

    // Only canonicalization can create these.  ProgramManager sets the refid.
    friend class C14n;
    friend class ProgramManager;
    VariableReference( const CanonicalProgram* cp,
                       const std::string&      variableName,
                       unsigned short          variableToken,
                       const VariableType&     vtype,
                       bool                    isInitialized,
                       const std::string&      annotation );

    // Copy is legal but private because only ProgramManager should do
    // it.
    VariableReference( const VariableReference& copy ) = default;
    VariableReference& operator=( const VariableReference& rhs ) = delete;

    // Canonicalization sets the properties at creation time. These
    // are immutable once canonicalization is complete.
    void addTextureLookupKind( TextureLookup::LookupKind kind );
    void markPointerMayEscape() { m_pointerMayEscape = true; }
    void markHasBufferStores() { m_hasBufferStores = true; }
    void markHasIllFormedAccess() { m_hasIllFormedAccess = true; }

    // Empty ctor only for deserialization
    VariableReference();
};
void readOrWrite( PersistentStream* stream, VariableReference*, const char* label );

}  // namespace optix
