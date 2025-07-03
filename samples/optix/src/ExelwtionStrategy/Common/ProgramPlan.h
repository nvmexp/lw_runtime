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

#include <Context/UpdateManager.h>

#include <ExelwtionStrategy/Compile.h>  // CPIDSet

#include <array>
#include <string>

namespace optix {

class Context;
class Plan;

// Will ilwalidate the associated Plan when needed.
class ProgramPlan : public UpdateEventListenerNop
{
  public:
    ProgramPlan( Plan* parent, Context* context, unsigned int entryIndex, const Device* device );
    ~ProgramPlan() override;

    // Determine if the specializations are compatible with another
    bool isCompatibleWith( const ProgramPlan& otherPlan ) const;

    // Return the specialization sumamry
    std::string summaryString() const;

    // Return the set of reachable programs for a given semantic type
    const CPIDSet& getReachablePrograms( SemanticType stype ) const;

    // Return all the reachable programs organized by semantic type.
    const std::array<CPIDSet, NUM_SEMANTIC_TYPES>& getReachablePrograms() const;

    // Return all reachable programs.
    // This function is useful to iterate over all the unique canonical programs in plan.
    // Note that iterating over the semantic types and then over the set in each
    // semantic type might touch the same program multiple times.
    // This is an expensive operation because it ilwolves constructing the output set.
    CPIDSet getAllReachablePrograms() const;

    // Computes the heavyweight information for all reachable bindless callable
    // programs in the plan.
    void                                       computeHeavyweightBindlessPrograms();
    const std::set<const CallSiteIdentifier*>& getHeavyweightCallsites() const;
    bool isHeavyweight( const CanonicalProgram* cp ) const;
    bool needsLightweightCompilation( const CanonicalProgram* cp ) const;
    bool needsHeavyweightCompilation( const CanonicalProgram* cp ) const;
    // Returns the program ids of canonical programs that call heavyweight
    // bindless programs for continuation stack callwlation.
    const std::set<int>& getHeavyBindlessCallers() const;

  private:
    //------------------------------------------------------------------------
    // Potentially ilwalidating events
    //------------------------------------------------------------------------

    void eventCanonicalProgramSemanticTypeDidChange( const CanonicalProgram* cp, SemanticType stype, bool added ) override;
    void eventCanonicalProgramInheritedSemanticTypeDidChange( const CanonicalProgram* cp,
                                                              SemanticType            stype,
                                                              SemanticType            inheritedStype,
                                                              bool                    added ) override;
    void eventCanonicalProgramUsedByRayTypeDidChange( const CanonicalProgram* cp, unsigned int rayType, bool added ) override;
    void eventCanonicalProgramDirectCallerDidChange( const CanonicalProgram* cp, CanonicalProgramID cpid, bool added ) override;
    void eventCanonicalProgramPotentialCalleesDidChange( const CanonicalProgram* cp, const CallSiteIdentifier* cs, bool added ) override;

    void eventGlobalScopeRayGenerationProgramDidChange( unsigned int index, Program* oldProgram, Program* newProgram ) override;
    void eventGlobalScopeExceptionProgramDidChange( unsigned int index, Program* oldProgram, Program* newProgram ) override;

    //------------------------------------------------------------------------
    // Helper functions
    //------------------------------------------------------------------------
    void createPlan( const Device* device );

    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------

    // The reachable programs, sorted by semantic type
    std::array<CPIDSet, NUM_SEMANTIC_TYPES> m_reachablePrograms;

    // Heavyweight bindless callables handling (RTX only). Filled during compilation.
    std::set<const CanonicalProgram*>   m_heavyBindlessCallables;
    std::set<const CanonicalProgram*>   m_hybridBindlessCallables;
    std::set<int>                       m_heavyBindlessCallerIds;
    std::set<const CallSiteIdentifier*> m_heavyCallsites;

    // The plan to which we are attached
    Plan*    m_plan    = nullptr;
    Context* m_context = nullptr;

    // The parent Plan might not have an entry number (entry is defined only for MKPlan and RTXPlan).
    // So keep a copy of the current entry here. This is useful to decide whether to ilwalidate the plan
    // for events that involve changes of the raygen and exception programs.
    unsigned int m_entry = ~0U;
};
}
