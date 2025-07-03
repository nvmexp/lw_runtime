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

#include <ExelwtionStrategy/Common/SpecializationPlan.h>

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <Device/LWDADevice.h>
#include <ExelwtionStrategy/Common/ProgramPlan.h>
#include <ExelwtionStrategy/Plan.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <Memory/MemoryManager.h>
#include <Objects/TextureSampler.h>
#include <Util/UsageReport.h>

#include <corelib/misc/String.h>
#include <prodlib/misc/RTFormatUtil.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

using namespace optix;
using namespace corelib;
using namespace prodlib;

namespace {
// clang-format off
  Knob<bool> k_useLDG( RT_DSTRING("compile.useLDG"), true,  RT_DSTRING( "Use LDG for read-only buffer access" ) );
  Knob<bool> k_useLDGVectorsOnly( RT_DSTRING("compile.useLDGVectorsOnly"), false, RT_DSTRING( "Limit use of LDG to vector loads" ) );
  Knob<bool> k_specializeSingleId( RT_DSTRING("compile.specializeSingleId"), true,  RT_DSTRING( "Specialize buffer and texture access from a single variable.  This can increase recompiles" ) );
// clang-format on
}  // namespace

//------------------------------------------------------------------------
// Main interface
//------------------------------------------------------------------------

SpecializationPlan::SpecializationPlan( Plan* parent, Context* context, const PerDeviceProgramPlan& perDeviceProgramPlan )
    : m_plan( parent )
    , m_context( context )
{
    m_context->getUpdateManager()->registerUpdateListener( this );
    createPlan( perDeviceProgramPlan );
}

SpecializationPlan::~SpecializationPlan()
{
    m_context->getUpdateManager()->unregisterUpdateListener( this );
}

bool SpecializationPlan::isCompatibleWith( const SpecializationPlan& otherPlan ) const
{
    if( m_hasMotionTransforms != otherPlan.m_hasMotionTransforms )
        return false;

    return m_specializations.isCompatibleWith( otherPlan.m_specializations );
}

std::string SpecializationPlan::summaryString() const
{
    ProgramManager* pm = m_context->getProgramManager();
    return m_specializations.summaryString( pm );
}

const Specializations& SpecializationPlan::getSpecializations() const
{
    return m_specializations;
}

Specializations SpecializationPlan::narrowFor( const CanonicalProgram* cp ) const
{
    Specializations narrowed( m_specializations );

    // At this point, the excpetion flags, print enabled and transform
    // depths are not narrowed.

    // Narrow the specializations to the ones applicable to this
    // program.
    VariableSpecializations narrowed_varspec;
    for( const auto& varref : cp->getVariableReferences() )
    {
        VariableReferenceID refid = varref->getReferenceID();

        auto it = narrowed.m_varspec.find( refid );
        RT_ASSERT( it != narrowed.m_varspec.end() );
        narrowed_varspec.insert( std::make_pair( refid, std::move( it->second ) ) );
    }
    narrowed.m_varspec = std::move( narrowed_varspec );


    return narrowed;
}

void SpecializationPlan::createPlan( const PerDeviceProgramPlan& perDeviceProgramPlan )
{
    // Compute global plan information
    m_specializations.m_exceptionFlags    = m_context->getExceptionFlags();
    m_specializations.m_printEnabled      = m_context->getPrintEnabled();
    m_specializations.m_minTransformDepth = 0;  // Min transform depth is not computed yet
    m_specializations.m_maxTransformDepth = m_context->getBindingManager()->getMaxTransformHeight();

    m_hasMotionTransforms = m_context->getBindingManager()->hasMotionTransforms();


    // Compute set of reachable programs in the plan on all devices
    CPIDSet uniqueReachableProgs;
    for( const auto& pp : perDeviceProgramPlan )
    {
        const ProgramPlan* programPlan    = pp.second;
        CPIDSet            reachableProgs = programPlan->getAllReachablePrograms();
        uniqueReachableProgs.insert( reachableProgs.begin(), reachableProgs.end() );
    }

    // Compute set of variables in the plan
    std::set<const VariableReference*> varReferences;
    ProgramManager*                    pm = m_context->getProgramManager();
    for( const auto& cpID : uniqueReachableProgs )
    {
        const CanonicalProgram* cp   = pm->getCanonicalProgramById( cpID );
        const auto&             vars = cp->getVariableReferences();
        varReferences.insert( vars.begin(), vars.end() );
    }

    // Determine specializations for variables in reachable programs
    for( const VariableReference* varref : varReferences )
    {
        VariableSpecialization& vs = m_specializations.m_varspec[varref->getReferenceID()];

        determineSpecializationForVariable( varref, vs, m_context->getPreferFastRecompiles() );
    }
}

static bool isVectorBuffer( const ObjectManager* om, const BindingManager::BufferBindingSet& bbindings )
{
    RT_ASSERT( !bbindings.empty() );
    const unsigned int bufid  = *bbindings.begin();
    const Buffer*      buffer = om->getBufferById( bufid );
    return isVector( buffer->getFormat() );
}

static bool isReadOnly( const ObjectManager* om, const BindingManager::BufferBindingSet& bbindings )
{
    for( auto bufid : bbindings )
    {
        const Buffer* buffer = om->getBufferById( bufid );
        if( buffer->getType() != RT_BUFFER_INPUT )
            return false;
    }

    return true;
}

static bool shouldPreferLDG( const ObjectManager* om, const BindingManager::BufferBindingSet& bbindings )
{
    if( !k_useLDG.get() )
        return false;

    // If there is any buffer binding that is not read-only, we can't use LDG.
    if( !isReadOnly( om, bbindings ) )
        return false;

    // If we *can* use LDG, determine if we actually *want* to, e.g. because
    // we might only want to do so for buffers of vector type.
    if( k_useLDGVectorsOnly.get() )
        return isVectorBuffer( om, bbindings );

    return true;
}

void SpecializationPlan::determineSpecializationForVariable( const VariableReference* varref, VariableSpecialization& vs, bool preferFastRecompiles )
{
    BindingManager* bm = m_context->getBindingManager();
    MemoryManager*  mm = m_context->getMemoryManager();
    ObjectManager*  om = m_context->getObjectManager();

    // BindingManager ties references to variables: For a given reference in the
    // program, what are all the variables that can be seen by the program?
    const BindingManager::BufferBindingSet&  bbindings = bm->getBufferBindingsForReference( varref->getReferenceID() );
    const BindingManager::TextureBindingSet& tbindings = bm->getTextureBindingsForReference( varref->getReferenceID() );
    const BindingManager::VariableBindingSet& vbindings = bm->getVariableBindingsForReference( varref->getReferenceID() );

    RT_ASSERT_MSG( bbindings.empty() || tbindings.empty(),
                   "must not have buffer and texture binding at the same time" );

    // If this reference is a buffer or texture, try to specialize it entirely
    // for best performance. If that is not possible due to more than one
    // binding, or if the reference is no buffer or texture, try to specialize
    // the corresponding variable. If that is not possible due to more than one
    // variable binding, don't specialize.

    const bool specializeSingleId = k_specializeSingleId.get() && !preferFastRecompiles;
    if( specializeSingleId && bbindings.size() == 1 )
    {
        // Only one buffer binding
        vs.setSingleId( bbindings.front() );
    }
    else if( specializeSingleId && tbindings.size() == 1 )
    {
        // Only one texture binding
        vs.setSingleId( tbindings.front() );
    }
    else if( vbindings.size() == 1 )
    {
        // There is only one binding in the set.  Specialize for a
        // single scope lookup at the specified offset.
        vs.setSingleBinding( vbindings.front() );
    }
    else if( vbindings.empty() )
    {
        vs.setUnused();
    }
    else
    {
        vs.setGeneric( varref->getVariableToken() );

        issueUnspecializedWarning( varref->getReferenceID() );
    }


    // Determine if we can use hardware texture (only).  This is not
    // possible if:
    // 1. texture added to binding that requires sw
    // 2. texturesampler was reallocated in sw, or
    // 3. any texture demoted to software.
    if( varref->getType().isTextureSampler() )
    {
        MAccess access    = MAccess::makeNone();
        bool    mixedKind = false, mixedPointer = false;
        for( auto texid : tbindings )
        {
            TextureSampler* tex = om->getTextureSamplerById( texid );
            mm->determineMixedAccess( tex->getMTextureSampler(), mixedKind, mixedPointer, access );
        }
        if( !mixedKind )
        {
            if( access.getKind() == MAccess::TEX_OBJECT || access.getKind() == MAccess::TEX_REFERENCE
                || access.getKind() == MAccess::DEMAND_TEX_OBJECT )
            {
                vs.accessKind = VariableSpecialization::HWTextureOnly;
            }
            else if( access.getKind() == MAccess::LINEAR || access.getKind() == MAccess::MULTI_PITCHED_LINEAR )
            {
                vs.accessKind = VariableSpecialization::SWTextureOnly;
            }
        }
    }

    // Determine if all buffers attached to the reference can be
    // specialized as either global-loadstore only, LDG, or texture
    // only.
    if( varref->getType().isBuffer() )
    {
        MAccess access    = MAccess::makeNone();
        bool    mixedKind = false, mixedPointer = false;
        for( auto bufid : bbindings )
        {
            Buffer* buf = om->getBufferById( bufid );
            mm->determineMixedAccess( buf->getMBuffer(), mixedKind, mixedPointer, access );
        }

        // Specialize based on the types of buffer access
        if( !mixedKind )
        {
            if( access.getKind() == MAccess::TEX_REFERENCE )
            {
                // All texture. At the moment, the only way this can happen
                // is with texheap.
                vs.accessKind  = VariableSpecialization::TexHeap;
                vs.texheapUnit = access.getTexReference().texUnit;

                if( !mixedPointer )
                {
                    // All texture, single offset
                    vs.accessKind   = VariableSpecialization::TexHeapSingleOffset;
                    vs.singleOffset = access.getTexReference().indexOffset;
                }
            }
            else if( access.getKind() == MAccess::LINEAR || access.getKind() == MAccess::NONE )
            {
                // All normal loads/stores.  Note: we could specialize on
                // the pointer value here, but it is likely to cause
                // excessive recompiles.
                vs.accessKind = VariableSpecialization::PitchedLinear;

                vs.isReadOnly = isReadOnly( om, bbindings );

                // Determine if this is a buffer that should be loaded
                // through LDG.  LWVM has a generic phase for transforming
                // loads to LDG, but it doesn't know as much as we do and
                // has to assume aliasing everywhere.
                if( shouldPreferLDG( om, bbindings ) )
                    vs.accessKind = VariableSpecialization::PitchedLinearPreferLDG;
            }
        }
    }
    // TODO: add other specializations
}

void SpecializationPlan::issueUnspecializedWarning( const VariableReferenceID refid ) const
{
    UsageReport& ur                 = m_context->getUsageReport();
    const int    log_level          = 20;
    const int    usage_report_level = 2;
    if( !log::active( log_level ) && !ur.isActive( usage_report_level ) )
        return;

    BindingManager* bm = m_context->getBindingManager();
    ProgramManager* pm = m_context->getProgramManager();

    const BindingManager::VariableBindingSet& vbindings = bm->getVariableBindingsForReference( refid );
    const VariableReference*                  varref    = pm->getVariableReferenceById( refid );

    if( log::active( log_level ) )
    {
        lwarn << "No specialization for refid: " << refid << " (" << varref->getInputName() << ") in "
              << varref->getParent()->getInputFunctionName() << "\n";
    }
    std::set<ObjectClass> classes;
    std::set<size_t>      offsets;
    std::set<bool>        isDefaultValue;
    llog( log_level ) << "\tvbindings.size() = " << vbindings.size() << "\n";
    for( auto binding : vbindings )
    {
        if( classes.insert( binding.scopeClass() ).second )
        {
            llog( log_level ) << "\tclass: " << getNameForClass( binding.scopeClass() ) << "\n";
        }
        if( offsets.insert( binding.offset() ).second )
        {
            llog( log_level ) << "\toffset: " << binding.offset() << "\n";
        }
        if( isDefaultValue.insert( binding.isDefaultValue() ).second )
        {
            llog( log_level ) << "\tdefault: " << binding.isDefaultValue() << "\n";
        }
    }

    // Usage report warning too

    if( !ur.isActive( usage_report_level ) )
        return;

    std::ostream& os = ur.getStream( usage_report_level, "PERF WARNING" );
    {
        // list of semantic types for parent canonical program, e.g., ANY_HIT
        std::vector<SemanticType> stypes;
        varref->getParent()->getAllUsedSemanticTypes( stypes );

        os << "Variable \"" << varref->getInputName() << "\""
           << " used in program \"" << varref->getParent()->getInputFunctionName() << "\"";
        if( !stypes.empty() )
        {
            os << " (";
            for( auto it = stypes.begin(); it != stypes.end(); ++it )
            {
                if( it != stypes.begin() )
                    os << ", ";
                os << semanticTypeToString( *it );
            }
            os << ")";
        }

        // string list of scopes, e.g., "GeometryInstance, Geometry"
        std::string scopesstr;
        {
            std::ostringstream ss;
            for( auto it = classes.begin(); it != classes.end(); ++it )
            {
                if( it != classes.begin() )
                    ss << ", ";
                ss << getNameForClass( *it );
            }
            scopesstr = ss.str();
        }

        // Describe one problem only, for clarity, even though multiple problems may exist.

        if( classes.size() > 1 )
        {
            os << " is declared at different scopes (" << scopesstr << ").";
        }
        else if( offsets.size() > 1 )
        {
            os << " appears in a different order in the list of variables declared at scope (" << scopesstr << ").";
        }

        os << std::endl;
        os << "For best runtime performance, a variable should be declared at one scope only, and in the same order "
              "with other variables on that scope."
           << std::endl;
    }
}


//------------------------------------------------------------------------
// Potentially ilwalidating events
//------------------------------------------------------------------------

void SpecializationPlan::eventContextSetExceptionFlags( const uint64_t oldFlags, const uint64_t newFlags )
{
    // Ilwalidate the plan if its enabled/disabled exception information does not
    // match the 'newFlags'. 'oldFlags' is ignored since the plan knows with which
    // set of enabled flags it was created.
    if( m_specializations.m_exceptionFlags != newFlags )
        m_plan->ilwalidatePlan();
}

void SpecializationPlan::eventContextSetPrinting( bool        oldEnabled,
                                                  size_t      oldBufferSize,
                                                  const int3& oldLaunchIndex,
                                                  bool        newEnabled,
                                                  size_t      newBufferSize,
                                                  const int3& newLaunchIndex )
{
    // Ilwalidate the plan if its enabled/disabled printing information does not
    // match the new one. We don't care about the buffer size and which launch
    // indices should print since we do not use that information to generate
    // different code.
    if( m_specializations.m_printEnabled != newEnabled )
        m_plan->ilwalidatePlan();
}

void SpecializationPlan::eventContextSetPreferFastRecompiles( bool /*oldValue*/, bool /*newValue*/ )
{
    // This is expected to happen rarely, so don't bother with a detailed comparison.
    m_plan->ilwalidatePlan();
}

void SpecializationPlan::eventContextHasMotionTransformsChanged( bool newValue )
{
    if( m_hasMotionTransforms != newValue )
        m_plan->ilwalidatePlan();
}

void SpecializationPlan::ilwalidateSpecialization( VariableReferenceID refid )
{
    // early exit
    if( !m_plan->isValid() )
        return;

    auto specIt = m_specializations.m_varspec.find( refid );
    if( specIt == m_specializations.m_varspec.end() )
    {
        // This is a new reference ID that did not exist when the plan was
        // created.
        m_plan->ilwalidatePlan();
        return;
    }
    if( specIt->second.lookupKind != VariableSpecialization::GenericLookup )
    {
        // There is a new binding or an old one was dropped.  Technically
        // we do not need to ilwalidate the plan for a dropped binding but
        // until we have a mechanism for conservative plans we will
        // ilwalidate.
        m_plan->ilwalidatePlan();
        return;
    }
    if( specIt->second.accessKind != VariableSpecialization::GenericAccess )
    {
        // There is a new binding or an old one was dropped.  Technically
        // we do not need to ilwalidate the plan for a dropped binding but
        // until we have a mechanism for conservative plans we will
        // ilwalidate.
        m_plan->ilwalidatePlan();
        return;
    }
}

void SpecializationPlan::eventVariableBindingsDidChange( VariableReferenceID refid, const VariableReferenceBinding& binding, bool added )
{
    ilwalidateSpecialization( refid );
}

void SpecializationPlan::eventBufferBindingsDidChange( VariableReferenceID refid, int bufid, bool added )
{
    // If the plan is already invalid, don't ilwalidate again
    if( !m_plan->isValid() )
        return;

    auto oldSpecIt = m_specializations.m_varspec.find( refid );

    // If it's a new specialization, it isn't a change, so don't ilwalidate
    if( oldSpecIt == m_specializations.m_varspec.end() )
        return;

    BindingManager* bm = m_context->getBindingManager();
    MemoryManager*  mm = m_context->getMemoryManager();
    ObjectManager*  om = m_context->getObjectManager();

    const VariableSpecialization& oldSpecialization = oldSpecIt->second;

    bool invalid = false;

    if( oldSpecialization.lookupKind == VariableSpecialization::SingleScope )
    {
        const BindingManager::VariableBindingSet& vbindings = bm->getVariableBindingsForReference( refid );
        if( vbindings.size() != 1 )
            invalid = true;
    }
    else
    {
        invalid = true;
    }

    Buffer* buf = om->getBufferById( bufid );

    // If the specialization was read-only before, but the new buffer
    // is writable (!RT_BUFFER_INPUT), then ilwalidate.
    if( oldSpecialization.isReadOnly && buf->getType() != RT_BUFFER_INPUT )
        invalid = true;

    if( invalid == false && ( oldSpecialization.accessKind == VariableSpecialization::AccessKind::PitchedLinear
                              || oldSpecialization.accessKind == VariableSpecialization::AccessKind::PitchedLinearPreferLDG ) )
    {

        MAccess access       = MAccess::makeNone();
        bool    mixedKind    = false;
        bool    mixedPointer = false;
        mm->determineMixedAccess( buf->getMBuffer(), mixedKind, mixedPointer, access );

        if( mixedKind )
            invalid = true;

        if( access.getKind() != MAccess::LINEAR && access.getKind() != MAccess::NONE )
            invalid = true;
    }
    else
    {
        invalid = true;
    }

    if( invalid )
        ilwalidateSpecialization( refid );
}

void SpecializationPlan::eventTextureBindingsDidChange( VariableReferenceID refid, int texid, bool added )
{
    ilwalidateSpecialization( refid );

    // Check for validity after ilwalidateSpecialization which could ilwalidate the plan
    if( !m_plan->isValid() )
        return;

    // Handle ilwalidation of the texture specialization
    MemoryManager* mm = m_context->getMemoryManager();
    ObjectManager* om = m_context->getObjectManager();

    // Removing a binding is never a problem
    if( !added )
        return;

    TextureSampler* tex = om->getTextureSamplerById( texid );
    if( !tex->getMTextureSampler() )
        return;

    // General lookup is not a problem
    const VariableSpecialization& vs = m_specializations.m_varspec.at( refid );
    if( vs.lookupKind == VariableSpecialization::GenericLookup && vs.accessKind == VariableSpecialization::GenericAccess )
        return;

    // Otherwise, determine if the added bindings ilwalidates one of our specializations
    MAccess access    = MAccess::makeNone();
    bool    mixedKind = false, mixedPointer = false;
    mm->determineMixedAccess( tex->getMTextureSampler(), mixedKind, mixedPointer, access );
    if( !mixedKind )
    {
        if( access.getKind() == MAccess::NONE )
            return;  // None is still valid
        if( ( access.getKind() == MAccess::LINEAR || access.getKind() == MAccess::MULTI_PITCHED_LINEAR )
            && vs.accessKind == VariableSpecialization::SWTextureOnly )
            return;  // Still SW tex
        if( ( access.getKind() == MAccess::TEX_REFERENCE || access.getKind() == MAccess::TEX_OBJECT
              || access.getKind() == MAccess::DEMAND_TEX_OBJECT )
            && vs.accessKind == VariableSpecialization::HWTextureOnly )
            return;  // Still HW tex
    }

    // Otherwise, ilwalidate the plan
    m_plan->ilwalidatePlan();
}

void SpecializationPlan::eventTextureSamplerMAccessDidChange( const TextureSampler* tex,
                                                              const Device*         device,
                                                              const MAccess&        oldMA,
                                                              const MAccess&        newMA )
{
    // early exit, also prevent access to m_specializations with out of bound indices
    if( !m_plan->isValid() )
        return;

    BindingManager* bm = m_context->getBindingManager();
    MemoryManager*  mm = m_context->getMemoryManager();

    // Determine if texture sampler specialization was changed
    MAccess access    = MAccess::makeNone();
    bool    mixedKind = false, mixedPointer = false;
    mm->determineMixedAccess( tex->getMTextureSampler(), mixedKind, mixedPointer, access );

    for( const auto& refid : bm->getIlwerseBindingsForTextureId( tex->getId() ) )
    {
        auto it = m_specializations.m_varspec.find( refid );
        if( it == m_specializations.m_varspec.end() )
        {
            // Skip variables that are irrelevant for this plan.
            continue;
        }
        const VariableSpecialization& vs = it->second;

        if( vs.lookupKind == VariableSpecialization::GenericLookup && vs.accessKind == VariableSpecialization::GenericAccess )
            continue;
        if( !mixedKind )
        {
            if( access.getKind() == MAccess::NONE )
                return;  // None is still valid
            if( ( access.getKind() == MAccess::LINEAR || access.getKind() == MAccess::MULTI_PITCHED_LINEAR )
                && vs.accessKind == VariableSpecialization::SWTextureOnly )
                continue;  // Still SW tex
            if( ( access.getKind() == MAccess::TEX_REFERENCE || access.getKind() == MAccess::TEX_OBJECT
                  || access.getKind() == MAccess::DEMAND_TEX_OBJECT )
                && vs.accessKind == VariableSpecialization::HWTextureOnly )
                continue;  // Still HW tex
        }

        // Otherwise, ilwalidate the plan
        m_plan->ilwalidatePlan();
        break;
    }
}

void SpecializationPlan::eventBufferMAccessDidChange( const Buffer* buf, const Device* device, const MAccess& oldMA, const MAccess& newMA )
{
    // early exit, also prevent access to m_specializations with out of bound indices
    if( !m_plan->isValid() )
        return;

    BindingManager* bm = m_context->getBindingManager();
    MemoryManager*  mm = m_context->getMemoryManager();

    // Early exit if there are no variable bindings.
    auto& bindings = bm->getIlwerseBindingsForBufferId( buf->getId() );
    if( bindings.empty() )
        return;

    MAccess access       = MAccess::makeNone();
    bool    mixedKind    = false;
    bool    mixedPointer = false;
    mm->determineMixedAccess( buf->getMBuffer(), mixedKind, mixedPointer, access );

    for( const auto& refid : bindings )
    {
        auto varspec = m_specializations.m_varspec.find( refid );
        if( varspec == m_specializations.m_varspec.end() )
        {
            // Skip variables that are irrelevant for this plan.
            continue;
        }
        const VariableSpecialization& vs = varspec->second;

        if( vs.lookupKind == VariableSpecialization::GenericLookup && vs.accessKind == VariableSpecialization::GenericAccess )
            continue;
        if( !mixedKind )
        {
            if( access.getKind() == MAccess::NONE )
                return;  // None is still valid
            if( access.getKind() == MAccess::LINEAR && ( vs.accessKind == VariableSpecialization::PitchedLinearPreferLDG
                                                         || vs.accessKind == VariableSpecialization::PitchedLinear ) )
                continue;  // Still simple global
            if( access.getKind() == MAccess::TEX_REFERENCE && newMA.getKind() == MAccess::TEX_REFERENCE )
            {
                if( newMA.getTexReference().texUnit != vs.texheapUnit )
                {
                    if( vs.accessKind == VariableSpecialization::TexHeap )
                        continue;
                    if( vs.accessKind == VariableSpecialization::TexHeapSingleOffset
                        && vs.singleOffset == access.getTexReference().indexOffset )
                        continue;
                }
            }
        }

        // Otherwise, ilwalidate the plan
        m_plan->ilwalidatePlan();
        break;
    }
}
