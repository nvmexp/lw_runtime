// Copyright (c) 2018, LWPU CORPORATION.
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
#include <string>

#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <Objects/GraphProperty.h>
#include <Util/ReusableIDMap.h>
#include <corelib/misc/Concepts.h>

namespace optix {
class PersistentStream;

class ProgramManager;
class Context;
class CanonicalProgram;
class VariableReference;


class CallSiteIdentifier : private corelib::NonCopyable
{
  public:
    // Persistence support to create unnamed CallSiteIdentifiers (to be filled after loading)
    CallSiteIdentifier( const CanonicalProgram* parent );
    ~CallSiteIdentifier() NOEXCEPT_FALSE;

    void addOrRemovePotentialCallees( const std::vector<CanonicalProgramID>& calleeIds, bool added );
    std::vector<CanonicalProgramID> getPotentialCallees() const;

    const std::string& getInputName() const;

    std::string getUniversallyUniqueName() const;

    const CanonicalProgram* getParent() const { return m_parent; }

    friend void readOrWrite( PersistentStream* stream, CallSiteIdentifier*, const char* label );

    // Generates a unique name for auto generated call sites based on the given
    // variable reference's unique name.
    static std::string generateCallSiteUniqueName( const VariableReference* varref );

  private:
    CallSiteIdentifier( const std::string& name, const CanonicalProgram* parent );

    //
    // WARNING: This is a persistent class. If you change anything you
    // should also update the readOrWrite function and bump the the
    // version number.
    //
    std::string             m_InputName;
    const CanonicalProgram* m_parent = nullptr;

    GraphProperty<CanonicalProgramID> m_potentialCallees;

    friend class C14n;
    friend class NodegraphPrinter;
};
void readOrWrite( PersistentStream* stream, CallSiteIdentifier*, const char* label );
}
