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

#include <ExelwtionStrategy/Common/ProgramPlan.h>

#include <Context/Context.h>
#include <Context/ProgramManager.h>
#include <ExelwtionStrategy/Plan.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <Objects/GlobalScope.h>

#include <prodlib/exceptions/IlwalidValue.h>

#include <deque>
#include <sstream>

using namespace optix;

// -----------------------------------------------------------------------------
ProgramPlan::ProgramPlan( Plan* parent, Context* context, unsigned int entryIndex, const Device* device )
    : m_plan( parent )
    , m_context( context )
    , m_entry( entryIndex )
{
    m_context->getUpdateManager()->registerUpdateListener( this );
    createPlan( device );
}

// -----------------------------------------------------------------------------
ProgramPlan::~ProgramPlan()
{
    m_context->getUpdateManager()->unregisterUpdateListener( this );
}

// -----------------------------------------------------------------------------
bool ProgramPlan::isCompatibleWith( const ProgramPlan& other ) const
{
    if( m_reachablePrograms != other.m_reachablePrograms )
        return false;

    if( m_heavyBindlessCallables != other.m_heavyBindlessCallables )
        return false;

    if( m_hybridBindlessCallables != other.m_hybridBindlessCallables )
        return false;

    if( m_heavyCallsites != m_heavyCallsites )
        return false;

    return true;
}

// -----------------------------------------------------------------------------
std::string ProgramPlan::summaryString() const
{
    std::ostringstream out;
    out << "{";
    for( int si = 0; si < NUM_SEMANTIC_TYPES; ++si )
    {
        const CPIDSet& cpids = m_reachablePrograms[si];
        if( cpids.empty() )
            continue;

        out << semanticTypeToAbbreviationString( SemanticType( si ) ) << "(";
        for( CPIDSet::const_iterator iter = cpids.begin(); iter != cpids.end(); ++iter )
        {
            if( iter != cpids.begin() )
                out << ",";
            out << *iter;
        }
        out << ") ";
    }

    out << "}";
    if( !m_heavyBindlessCallables.empty() )
    {
        out << " {Heavy BCP: ";
        for( const CanonicalProgram* cp : m_heavyBindlessCallables )
        {
            out << cp->getInputFunctionName() << " ";
        }
        out << "} {Heavy CS: ";
        for( const CallSiteIdentifier* cs : m_heavyCallsites )
        {
            out << cs->getParent()->getInputFunctionName() << "/" << cs->getInputName() << " ";
        }
        out << "}";

        if( !m_hybridBindlessCallables.empty() )
        {
            out << " {Hybrid BCP : ";
            for( const CanonicalProgram* cp : m_hybridBindlessCallables )
            {
                out << cp->getInputFunctionName() << " ";
            }
            out << "}";
        }
    }
    return out.str();
}

// -----------------------------------------------------------------------------
const CPIDSet& ProgramPlan::getReachablePrograms( SemanticType stype ) const
{
    RT_ASSERT_MSG( static_cast<unsigned int>( stype ) < m_reachablePrograms.size(), "Invalid semantic type" );
    return m_reachablePrograms[stype];
}

// -----------------------------------------------------------------------------
const std::array<CPIDSet, NUM_SEMANTIC_TYPES>& ProgramPlan::getReachablePrograms() const
{
    return m_reachablePrograms;
}

// -----------------------------------------------------------------------------
CPIDSet ProgramPlan::getAllReachablePrograms() const
{
    CPIDSet result;
    for( const CPIDSet& cpIDSet : m_reachablePrograms )
        result.insert( cpIDSet.begin(), cpIDSet.end() );
    return result;
}
// -----------------------------------------------------------------------------
void ProgramPlan::computeHeavyweightBindlessPrograms()
{
    // Collect bindless callables that need to be compiled as continuation callables
    m_heavyBindlessCallables.clear();
    const ProgramManager* pm = m_context->getProgramManager();
    for( const CanonicalProgramID cpid : getReachablePrograms( ST_BINDLESS_CALLABLE_PROGRAM ) )
    {
        const CanonicalProgram* cp = pm->getCanonicalProgramById( cpid );
        if( cp->callsTrace() )
        {
            m_heavyBindlessCallables.emplace( cp );
        }
    }

    m_heavyCallsites.clear();
    std::deque<const CanonicalProgram*> queue( m_heavyBindlessCallables.begin(), m_heavyBindlessCallables.end() );
    while( !queue.empty() )
    {
        const CanonicalProgram* cp = queue.front();
        queue.pop_front();
        for( const CallSiteIdentifier* cs : cp->getCallingCallSites() )
        {
            const CanonicalProgram* parent = cs->getParent();

            if( m_heavyCallsites.emplace( cs ).second )
            {
                if( parent->isUsedAsSemanticType( ST_BINDLESS_CALLABLE_PROGRAM ) )
                {
                    if( m_heavyBindlessCallables.emplace( parent ).second )
                        queue.push_back( parent );
                }
                else
                {
                    m_heavyBindlessCallerIds.emplace( parent->getID() );
                }
            }
        }
    }

    for( const CallSiteIdentifier* cs : m_heavyCallsites )
    {
        if( cs->getParent()->isUsedAsSemanticTypes( {ST_EXCEPTION, ST_NODE_VISIT, ST_BOUNDING_BOX} ) )
        {
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Program " + cs->getParent()->getInputFunctionName() + " is not allowed to "
                "call a program that calls trace at call site " + cs->getInputName() );
        }
    }


    // Collect bindless callables that need to be compiled as continuation
    // callable AND as direct callable. Also collect names of the heavyweight
    // call sites as they are needed for plan compatibility check.

    // And also heavy programs that are callable from AH and IS.


    std::vector<const CallSiteIdentifier*> toRemove;
    queue.clear();
    for( const CallSiteIdentifier* cs : m_heavyCallsites )
    {
        if( cs->getParent()->isUsedAsSemanticTypes( {ST_INTERSECTION, ST_ANY_HIT} ) )
        {
            toRemove.push_back( cs );
            for( CanonicalProgramID cpid : cs->getPotentialCallees() )
            {
                const CanonicalProgram* cp = pm->getCanonicalProgramById( cpid );
                queue.push_back( cp );
            }
        }
    }
    for( const CallSiteIdentifier* cs : toRemove )
        m_heavyCallsites.erase( cs );

    m_hybridBindlessCallables.clear();
    while( !queue.empty() )
    {
        const CanonicalProgram* cp = queue.front();
        queue.pop_front();
        if( m_hybridBindlessCallables.emplace( cp ).second )
        {
            for( const CallSiteIdentifier* cs : cp->getCallSites() )
            {
                if( m_heavyCallsites.count( cs ) )
                {
                    for( CanonicalProgramID cpid : cs->getPotentialCallees() )
                    {
                        queue.push_back( pm->getCanonicalProgramById( cpid ) );
                    }
                }
            }
        }
    }

    for( const CallSiteIdentifier* cs : m_heavyCallsites )
    {
        for( CanonicalProgramID cpid : cs->getPotentialCallees() )
        {
            const CanonicalProgram* cp = pm->getCanonicalProgramById( cpid );
            if( !m_heavyBindlessCallables.count( cp ) )
            {
                m_hybridBindlessCallables.emplace( cp );
            }
        }
    }
}
// -----------------------------------------------------------------------------
const std::set<const CallSiteIdentifier*>& ProgramPlan::getHeavyweightCallsites() const
{
    return m_heavyCallsites;
}
// -----------------------------------------------------------------------------
const std::set<int>& ProgramPlan::getHeavyBindlessCallers() const
{
    return m_heavyBindlessCallerIds;
}
// -----------------------------------------------------------------------------
bool ProgramPlan::isHeavyweight( const CanonicalProgram* cp ) const
{
    return m_heavyBindlessCallables.count( cp );
}
// -----------------------------------------------------------------------------
bool ProgramPlan::needsLightweightCompilation( const CanonicalProgram* cp ) const
{
    return !m_heavyBindlessCallables.count( cp ) || m_hybridBindlessCallables.count( cp );
}
// -----------------------------------------------------------------------------
bool ProgramPlan::needsHeavyweightCompilation( const CanonicalProgram* cp ) const
{
    return m_heavyBindlessCallables.count( cp ) || m_hybridBindlessCallables.count( cp );
}
// -----------------------------------------------------------------------------
void ProgramPlan::createPlan( const Device* device )
{
    ProgramManager*                            pm                = m_context->getProgramManager();
    GlobalScope*                               gs                = m_context->getGlobalScope();
    const ProgramManager::CanonicalProgramMap& canonicalPrograms = pm->getCanonicalProgramMap();

    // Seed the set with the canonical program from entry points (raygen and entry)
    const CanonicalProgram* rgp = gs->getRayGenerationProgram( m_entry )->getCanonicalProgram( device );
    const CanonicalProgram* exp = gs->getExceptionProgram( m_entry )->getCanonicalProgram( device );
    std::set<const CanonicalProgram*, CanonicalProgram::IDCompare> usedCPs;
    usedCPs.insert( rgp );
    usedCPs.insert( exp );

    // AABB iterator also includes bounding box programs
    if( rgp->isUsedAsSemanticType( ST_INTERNAL_AABB_ITERATOR ) )
        for( auto candidate : canonicalPrograms )
            if( candidate->isUsedOnDevice( device ) && candidate->isUsedAsSemanticType( ST_BOUNDING_BOX ) )
                usedCPs.insert( candidate );

    // Iteratively refine the programs and raytypes associated with
    // this entry point.
    GraphProperty<unsigned int, false> allProducedRayTypes;
    bool changed = true;
    while( changed )
    {
        changed = false;

        // Find ray types required so far
        for( auto cp : usedCPs )
            allProducedRayTypes.addPropertiesFrom( cp->producesRayTypes() );

        // Determine other canonical programs required
        for( auto candidate : canonicalPrograms )
        {
            // Skip programs not applicable to this device
            if( !candidate->isUsedOnDevice( device ) )
                continue;

            // Miss programs don't need to check the direct caller set, so check to see if we need to add it now
            if( candidate->isUsedAsSemanticType( ST_MISS ) && candidate->isUsedByRayTypes( allProducedRayTypes ) )
            {
                changed |= usedCPs.insert( candidate ).second;
                continue;
            }

            // Skip programs with no direct caller in the current set
            if( !candidate->hasDirectCaller( usedCPs ) )
                continue;

            // Add if it is a shading program for one of the current ray types or
            // any other traversal/bcp program.
            if( ( candidate->isUsedAsSemanticTypes( {ST_CLOSEST_HIT, ST_ANY_HIT} ) && candidate->isUsedByRayTypes( allProducedRayTypes ) )
                || candidate->isUsedAsSemanticTypes( {ST_NODE_VISIT, ST_INTERSECTION, ST_ATTRIBUTE, ST_BOUND_CALLABLE_PROGRAM} ) )
                changed |= usedCPs.insert( candidate ).second;
        }

        // Add bindless callable programs if necessary
        bool needsBindless = false;
        for( auto cp : usedCPs )
            needsBindless |= cp->callsBindlessCallableProgram();
        if( needsBindless )
            for( auto candidate : canonicalPrograms )
                if( candidate->isUsedAsSemanticType( ST_BINDLESS_CALLABLE_PROGRAM ) )
                    changed |= usedCPs.insert( candidate ).second;
    }

    // Now that we have the full set, split them out into the
    // different sets by semantic type.
    for( const auto& cp : usedCPs )
    {
        CanonicalProgramID cpID = cp->getID();
        for( int si = 0; si < NUM_SEMANTIC_TYPES; ++si )
        {
            SemanticType semType = static_cast<SemanticType>( si );
            if( cp->isUsedAsSemanticType( semType ) )
            {
                m_reachablePrograms[si].insert( cpID );

                llog( 20 ) << "Planning to include program: " << cp->getInputFunctionName() << " [id: " << cpID << ", "
                           << semanticTypeToString( semType ) << "]\n";
            }
        }
    }
}


// -----------------------------------------------------------------------------
// Event handlers.
// -----------------------------------------------------------------------------

// Note: adding a canonical program does not ilwalidate the plan for megakernel.
// It will get ilwalidated the first time the canonical program is attached to a
// partilwlate semantic type location. This is to allow specializations for
// different semantic types. -- Bigler -- Which appears to not be happening
// right now.

// -----------------------------------------------------------------------------
void ProgramPlan::eventCanonicalProgramSemanticTypeDidChange( const CanonicalProgram* cp, SemanticType stype, bool added )
{
    if( !m_plan->isValid() )
        return;

    if( added )
    {
        // Conservatively ilwalidate the plan for additions
        m_plan->ilwalidatePlan();
        return;
    }

    // Ilwalidate the plan only if the program is contained in our plan.
    const CPIDSet& cpids = m_reachablePrograms[stype];
    if( cpids.count( cp->getID() ) )
    {
        m_plan->ilwalidatePlan();
        return;
    }
}
// -----------------------------------------------------------------------------
void ProgramPlan::eventCanonicalProgramInheritedSemanticTypeDidChange( const CanonicalProgram* cp,
                                                                       SemanticType            stype,
                                                                       SemanticType            inheritedStype,
                                                                       bool                    added )
{
    if( !m_plan->isValid() )
        return;

    // Ilwalidate the plan only if the program is contained in our plan.
    const CPIDSet& cpids = m_reachablePrograms[stype];
    if( cpids.count( cp->getID() ) )
    {
        m_plan->ilwalidatePlan();
        return;
    }
}
// -----------------------------------------------------------------------------
void ProgramPlan::eventCanonicalProgramUsedByRayTypeDidChange( const CanonicalProgram* cp, unsigned int rayType, bool added )
{
    if( !m_plan->isValid() )
        return;

    // TODO SGP Bigler - We could keep a list of the ray types we found previously and if it
    // is in this set then ilwalidate the plan.
    m_plan->ilwalidatePlan();
}

// -----------------------------------------------------------------------------
void ProgramPlan::eventCanonicalProgramDirectCallerDidChange( const CanonicalProgram* cp, CanonicalProgramID cpid, bool added )
{
    if( !m_plan->isValid() )
        return;

    // TODO (Bigler): determine if there is a more precise way of knowing whether to ilwalidate the plan
    m_plan->ilwalidatePlan();
}

// -----------------------------------------------------------------------------
void ProgramPlan::eventGlobalScopeRayGenerationProgramDidChange( unsigned int index, Program* oldProgram, Program* newProgram )
{
    if( index == m_entry )
        m_plan->ilwalidatePlan();  // Technically not necessary if the canonical program did not change, but assume that it did.
}

// -----------------------------------------------------------------------------
void ProgramPlan::eventGlobalScopeExceptionProgramDidChange( unsigned int index, Program* oldProgram, Program* newProgram )
{
    if( index == m_entry )
        m_plan->ilwalidatePlan();  // Technically not necessary if the canonical program did not change, but assume that it did.
}
//------------------------------------------------------------------------------
void ProgramPlan::eventCanonicalProgramPotentialCalleesDidChange( const CanonicalProgram* cp, const CallSiteIdentifier* cs, bool added )
{
    if( !m_plan->isValid() )
        return;
    std::vector<SemanticType> usedSemanticTypes;
    cp->getAllUsedSemanticTypes( usedSemanticTypes );
    for( SemanticType stype : usedSemanticTypes )
    {
        const CPIDSet& cpids = m_reachablePrograms[stype];
        if( cpids.count( cp->getID() ) )
        {
            m_plan->ilwalidatePlan();
            return;
        }
    }
}
