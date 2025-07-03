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

#include <Objects/GlobalScope.h>

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <Context/SBTManager.h>
#include <Context/SharedProgramManager.h>
#include <Context/UpdateManager.h>
#include <Exceptions/VariableNotFound.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <Objects/Buffer.h>
#include <Objects/Program.h>
#include <Objects/ProgramRoot.h>
#include <Objects/TextureSampler.h>
#include <Util/LinkedPtrHelpers.h>

#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/ValidationError.h>
#include <prodlib/system/Knobs.h>

using namespace optix;
using namespace prodlib;

namespace {
// clang-format off
PublicKnob<std::string> k_forceExelwtionStrategy( RT_PUBLIC_DSTRING( "context.forceExelwtionStrategy" ),   "",    RT_PUBLIC_DSTRING( "Exelwtion strategy to use (default is empty, meaning rtx)" ) );
Knob<bool> k_forceTrivialExceptionProgram( RT_DSTRING("context.forceTrivialExceptionProgram"), false, RT_DSTRING("Force the use of a trivial exception program calling rtPrintExceptionDetails(). Needs printing enabled, e.g., print.enableOverride=1."));
// clang-format on
}

//------------------------------------------------------------------------
// CTOR / DTOR
//------------------------------------------------------------------------

GlobalScope::GlobalScope( Context* context )
    : LexicalScope( context, RT_OBJECT_GLOBAL_SCOPE )
{
    m_attachment.addOrRemoveProperty( true );  // GlobalScope is always attached
    RT_ASSERT_MSG( context->getEntryPointCount() == 0, "GlobalScope created after entry points added" );
    RT_ASSERT_MSG( context->getRayTypeCount() == 0, "GlobalScope created after ray types changed" );

    Program* nullProgram = getSharedNullProgram();
    setAabbComputeProgram( nullProgram );
    setAabbExceptionProgram( nullProgram );

    reallocateRecord();
}

GlobalScope::~GlobalScope()
{
    // Reset programs
    setEntryPointCount( 0 );
    setRayTypeCount( 0 );
    setAabbComputeProgram( nullptr );
    setAabbExceptionProgram( nullptr );
    deleteVariables();

    // Remove attachment after resetting programs
    m_attachment.addOrRemoveProperty( false );
}


//------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------

void GlobalScope::setRayTypeCount( unsigned int numRayTypes )
{
    bool growing = numRayTypes > m_missPrograms.size();
    if( growing )
    {
        // Growing, need to tell the SBT before doing the actual resizing,
        // otherwise setMissProgram will try notify of the SBT of a program
        // change outside the allocated range.
        m_context->getSBTManager()->rayTypeCountDidChange();
    }
    // Resize the miss program array, filling new entries with builtin
    // null program
    resizeVector( m_missPrograms, numRayTypes, [this]( int index ) { setMissProgram( index, nullptr ); },
                  [this]( int index ) { setMissProgram( index, getSharedNullProgram() ); } );

    if( !growing )
    {
        // When shrinking, we tell the SBTManager after resizing, otherwise
        // the notification for setMissProgram(index, nullptr) will
        // be outside the range.
        m_context->getSBTManager()->rayTypeCountDidChange();
    }
    // Validate and allocate
    subscribeForValidation();
    reallocateRecord();
}

void GlobalScope::setEntryPointCount( unsigned int numEntryPoints )
{
    bool growing = numEntryPoints > m_rayGenerationPrograms.size();
    if( growing )
    {
        // Growing, need to tell the SBT before doing the actual resizing,
        // otherwise the notifications for the program change will be
        // outside the allocated range.
        m_context->getSBTManager()->entryPointCountDidChange();
    }

    // Resize the arrays, filling new entries with the builtin null
    // program
    resizeVector( m_rayGenerationPrograms, numEntryPoints,
                  [this]( int index ) { setRayGenerationProgram( index, nullptr ); },
                  [this]( int index ) { setRayGenerationProgram( index, getSharedNullProgram() ); } );
    resizeVector( m_exceptionPrograms, numEntryPoints, [this]( int index ) { setExceptionProgram( index, nullptr ); },
                  [this]( int index ) { setExceptionProgram( index, getSharedNullProgram() ); } );

    if( !growing )
    {
        // When shrinking, we tell the SBTManager after resizing, otherwise
        // the notifications for the program changes will
        // be outside the range.
        m_context->getSBTManager()->entryPointCountDidChange();
    }

    if( m_aabbComputeProgram.get() )
    {
        // The AABB compute program is not part of m_rayGenerationPrograms and it should
        // always be put after the "actual" RG programs. Growing the SBT allocation
        // for RG programs will overwrite the record for the AABB program with the Null program
        // while shrinking it will discard the record which might never be written again (unless the
        // AABB program's object record offset changes.
        m_context->getSBTManager()->rayGenerationProgramDidChange( m_aabbComputeProgram.get(), numEntryPoints );
    }

    if( m_aabbExceptionProgram.get() )
    {
        // The same as for the AABB compute program is true for the SBT EX allocation
        // and the AABB exception program.
        m_context->getSBTManager()->exceptionProgramDidChange( m_aabbExceptionProgram.get(), numEntryPoints );
    }

    // Validate and allocate
    subscribeForValidation();
    reallocateRecord();
}

void GlobalScope::setRayGenerationProgram( unsigned int index, Program* newProgram )
{
    RT_ASSERT_MSG( index < m_rayGenerationPrograms.size(), "Ray generation program index out of range" );

    // Raygen program properties:
    //
    // Semantic type:             originates here
    // Attachment:                propagates from parent
    // Unresolved references:     propagates from child

    ProgramRoot root( getScopeID(), ST_RAYGEN, index );
    Program*    oldProgram = m_rayGenerationPrograms[index].get();

    if( oldProgram == newProgram )
        return;

    // Validate the semantic type before setting the program to avoid
    // exceptions during graph property propagation
    if( newProgram )
        newProgram->validateSemanticType( ST_RAYGEN );

    if( oldProgram )
    {
        // Remove properties from old program before updating the pointer
        oldProgram->attachOrDetachProperty_UnresolvedReference( this, root, false );
        this->attachOrDetachProperty_Attachment( oldProgram, false );
        oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_RAYGEN, false );
    }

    m_rayGenerationPrograms[index].set( this, newProgram );

    if( newProgram )
    {
        // Add properties to new program after updating the pointer
        newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_RAYGEN, true );
        this->attachOrDetachProperty_Attachment( newProgram, true );
        newProgram->attachOrDetachProperty_UnresolvedReference( this, root, true );
    }

    // Since plans may specialize on a particular entry point, we need to notify them of this change.
    m_context->getUpdateManager()->eventGlobalScopeRayGenerationProgramDidChange( index, oldProgram, newProgram );

    m_context->getSBTManager()->rayGenerationProgramDidChange( newProgram, index );

    // Validate and update
    subscribeForValidation();
    writeRecord();
}

void GlobalScope::setExceptionProgram( unsigned int index, Program* newProgram )
{
    RT_ASSERT_MSG( index < m_exceptionPrograms.size(), "Exception program index out of range" );

    // Exception program properties:
    //
    // Semantic type:             originated here
    // Attachment:                propagates from parent
    // Unresolved references:     propagates from child

    ProgramRoot root( getScopeID(), ST_EXCEPTION, index );
    Program*    oldProgram = m_exceptionPrograms[index].get();

    if( oldProgram == newProgram )
        return;

    if( newProgram && k_forceTrivialExceptionProgram.get() )
    {
        if( !m_context->getPrintEnabled() )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "context.forceTrivialExceptionProgram without printing enabled does not make sense" );

        newProgram = m_context->getSharedProgramManager()->getTrivialExceptionProgram();
    }

    // Validate the semantic type before setting the program to avoid
    // exceptions during graph property propagation
    if( newProgram )
        newProgram->validateSemanticType( ST_EXCEPTION );

    if( oldProgram )
    {
        // Remove properties from old program
        oldProgram->attachOrDetachProperty_UnresolvedReference( this, root, false );
        this->attachOrDetachProperty_Attachment( oldProgram, false );
        oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_EXCEPTION, false );
    }

    m_exceptionPrograms[index].set( this, newProgram );

    if( newProgram )
    {
        // Add properties to new program
        newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_EXCEPTION, true );
        this->attachOrDetachProperty_Attachment( newProgram, true );
        newProgram->attachOrDetachProperty_UnresolvedReference( this, root, true );
    }

    // Since plans may specialize on a particular entry point, we need to notify them of this change.
    m_context->getUpdateManager()->eventGlobalScopeExceptionProgramDidChange( index, oldProgram, newProgram );

    if( m_context->getSBTManager() )
        m_context->getSBTManager()->exceptionProgramDidChange( newProgram, index );

    // Validate and update
    subscribeForValidation();
    writeRecord();
}

void GlobalScope::setMissProgram( unsigned int index, Program* program )
{
    RT_ASSERT( index < m_missPrograms.size() );

    // Miss program properties:
    //
    // Semantic type:             originates here
    // Used by ray type:          originates here
    // Attachment:                propagates from parent
    // Unresolved references:     propagates from child

    ProgramRoot root( getScopeID(), ST_MISS, index );
    Program*    oldProgram = m_missPrograms[index].get();

    if( oldProgram == program )
        return;

    // Validate the semantic type before setting the program to avoid
    // exceptions during graph property propagation
    if( program )
        program->validateSemanticType( ST_MISS );

    if( oldProgram )
    {
        // Remove properties from old program
        oldProgram->attachOrDetachProperty_UnresolvedReference( this, root, false );
        this->attachOrDetachProperty_Attachment( oldProgram, false );
        oldProgram->receivePropertyDidChange_UsedByRayType( index, false );
        oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_MISS, false );
    }

    m_missPrograms[index].set( this, program );

    if( Program* newProgram = m_missPrograms[index].get() )
    {
        // Add properties to new program
        newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_MISS, true );
        newProgram->receivePropertyDidChange_UsedByRayType( index, true );
        this->attachOrDetachProperty_Attachment( newProgram, true );
        newProgram->attachOrDetachProperty_UnresolvedReference( this, root, true );
    }

    if( m_context->getSBTManager() )
        m_context->getSBTManager()->missProgramDidChange( program, index );

    // Validate and update
    subscribeForValidation();
    writeRecord();
}

void GlobalScope::setAabbComputeProgram( Program* program )
{
    if( m_aabbComputeProgram == program )
        return;

    // Validate the semantic type before setting the program to avoid
    // exceptions during graph property propagation
    if( program )
        program->validateSemanticType( ST_INTERNAL_AABB_ITERATOR );

    // AABB compute properties:
    //
    // Semantic type:             originates here
    // Attached:                  propagates from parent
    // Unresolved references:     propagates from child
    // Note: AABB cannot use bound callable programs

    ProgramRoot root( getScopeID(), ST_INTERNAL_AABB_ITERATOR, 0 );

    if( Program* oldProgram = m_aabbComputeProgram.get() )
    {
        // Remove properties from old program
        oldProgram->attachOrDetachProperty_UnresolvedReference( this, root, false );
        this->attachOrDetachProperty_Attachment( oldProgram, false );
        oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_INTERNAL_AABB_ITERATOR, false );
    }

    // Update link
    m_aabbComputeProgram.set( this, program );

    if( Program* newProgram = m_aabbComputeProgram.get() )
    {
        // Add properties to new program
        newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_INTERNAL_AABB_ITERATOR, true );
        this->attachOrDetachProperty_Attachment( newProgram, true );
        newProgram->attachOrDetachProperty_UnresolvedReference( this, root, true );
    }

    if( m_context->getSBTManager() )
        m_context->getSBTManager()->rayGenerationProgramDidChange( m_aabbComputeProgram.get(), m_context->getEntryPointCount() );

    // Validate and update
    subscribeForValidation();
    writeRecord();
}

void GlobalScope::setAabbExceptionProgram( Program* program )
{
    if( m_aabbExceptionProgram == program )
        return;

    // Validate the semantic type before setting the program to avoid
    // exceptions during graph property propagation
    if( program )
        program->validateSemanticType( ST_INTERNAL_AABB_EXCEPTION );

    // AABB compute properties:
    //
    // Semantic type:             originates here
    // Attached:                  propagates from parent
    // Unresolved references:     propagates from child
    // Note: AABB cannot use bound callable programs

    ProgramRoot root( getScopeID(), ST_INTERNAL_AABB_EXCEPTION, 0 );

    if( Program* oldProgram = m_aabbExceptionProgram.get() )
    {
        // Remove properties from old program
        oldProgram->attachOrDetachProperty_UnresolvedReference( this, root, false );
        this->attachOrDetachProperty_Attachment( oldProgram, false );
        oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_INTERNAL_AABB_EXCEPTION, false );
    }

    // Update link
    m_aabbExceptionProgram.set( this, program );

    if( Program* newProgram = m_aabbExceptionProgram.get() )
    {
        // Add properties to new program
        newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_INTERNAL_AABB_EXCEPTION, true );
        this->attachOrDetachProperty_Attachment( newProgram, true );
        newProgram->attachOrDetachProperty_UnresolvedReference( this, root, true );
    }

    if( m_context->getSBTManager() )
        m_context->getSBTManager()->exceptionProgramDidChange( m_aabbExceptionProgram.get(), m_context->getEntryPointCount() );

    // Validate and update
    subscribeForValidation();
    writeRecord();
}

Program* GlobalScope::getRayGenerationProgram( unsigned int index ) const
{
    RT_ASSERT( index < ( m_rayGenerationPrograms.size() + numInternalEntryPoints ) );
    if( index < m_rayGenerationPrograms.size() )
        return m_rayGenerationPrograms[index].get();
    else
        return m_aabbComputeProgram.get();
}

Program* GlobalScope::getExceptionProgram( unsigned int index ) const
{
    RT_ASSERT( index < ( m_exceptionPrograms.size() + numInternalEntryPoints ) );
    if( index < m_exceptionPrograms.size() )
        return m_exceptionPrograms[index].get();
    else
        return m_aabbExceptionProgram.get();
}

Program* GlobalScope::getAabbComputeProgram() const
{
    return m_aabbComputeProgram.get();
}

Program* GlobalScope::getAabbExceptionProgram() const
{
    return m_aabbExceptionProgram.get();
}

Program* GlobalScope::getMissProgram( unsigned int index ) const
{
    RT_ASSERT( index < m_missPrograms.size() );
    return m_missPrograms[index].get();
}

unsigned int GlobalScope::getAllEntryPointCount() const
{
    return m_rayGenerationPrograms.size() + numInternalEntryPoints;
}

void GlobalScope::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    // Check if there are post-processing command lists. If so, it is a valid use case that there
    // are no entry points, for instance tonemapping and denoising an image that comes from another
    // rendering pipeline.
    bool hasPostprocessing = getContext()->getObjectManager()->getCommandLists().size() > 0;

    LexicalScope::validate();
    int ne = getContext()->getEntryPointCount();
    if( ne == 0 && !hasPostprocessing )
        throw ValidationError( RT_EXCEPTION_INFO, "Context does not contain any entry points" );

    Program* nullProgram = getSharedNullProgram();
    for( int i = 0; i < ne; i++ )
    {
        if( getRayGenerationProgram( i ) == nullProgram )
        {
            std::ostringstream out;
            out << "Ray generation program " << i << " is null";
            throw ValidationError( RT_EXCEPTION_INFO, out.str() );
        }
    }

    // Validate remaining variables
    if( !m_unresolvedRemaining.empty() )
    {
        const VariableReference* varRef =
            m_context->getProgramManager()->getVariableReferenceById( m_unresolvedRemaining.front() );
        throw VariableNotFound( RT_EXCEPTION_INFO, this, "Unresolved reference to variable " + varRef->getInputName()
                                                             + " from " + varRef->getParent()->getInputFunctionName() );
    }
}


//------------------------------------------------------------------------
// LinkedPtr relationship mangement
//------------------------------------------------------------------------

void GlobalScope::detachFromParents()
{
    // No parents
}

void GlobalScope::detachLinkedChild( const LinkedPtr_Link* link )
{
    // GlobalScope only has program children. Figure out which list this
    // link is in. Note: it may be possible to use metadata in the
    // linked pointer to simplify this, but would probably make the
    // overall handling of linked pointers less efficient.
    unsigned int index;
    if( getElementIndex( m_rayGenerationPrograms, link, index ) )
        setRayGenerationProgram( index, getSharedNullProgram() );

    else if( getElementIndex( m_exceptionPrograms, link, index ) )
        setExceptionProgram( index, getSharedNullProgram() );

    else if( getElementIndex( m_missPrograms, link, index ) )
        setMissProgram( index, getSharedNullProgram() );

    else if( link == &m_aabbComputeProgram )
        setAabbComputeProgram( getSharedNullProgram() );

    else if( link == &m_aabbExceptionProgram )
        setAabbExceptionProgram( getSharedNullProgram() );

    else
        RT_ASSERT_FAIL_MSG( "Invalid child link in detachLinkedChild" );
}

void GlobalScope::childOffsetDidChange( const LinkedPtr_Link* link )
{
    LexicalScope::childOffsetDidChange( link );

    // Determine if it is a raygen/exception/miss program and notify SBTManager if so
    if( !m_context->useRtxDataModel() )
        return;

    unsigned int index;
    if( getElementIndex( m_rayGenerationPrograms, link, index ) )
        m_context->getSBTManager()->rayGenerationProgramOffsetDidChange( m_rayGenerationPrograms[index].get(), index );

    else if( getElementIndex( m_exceptionPrograms, link, index ) )
        m_context->getSBTManager()->exceptionProgramOffsetDidChange( m_exceptionPrograms[index].get(), index );

    else if( getElementIndex( m_missPrograms, link, index ) )
        m_context->getSBTManager()->missProgramOffsetDidChange( m_missPrograms[index].get(), index );

    else if( link == &m_aabbComputeProgram )
        m_context->getSBTManager()->rayGenerationProgramOffsetDidChange( m_aabbComputeProgram.get(),
                                                                         m_context->getEntryPointCount() );
}


//------------------------------------------------------------------------
// Object record management
//------------------------------------------------------------------------

size_t GlobalScope::getRecordBaseSize() const
{
    // Global scope has a variable-sized array for the child programs.
    // The array will be sized to the larger of the # of ray types and
    // the # of entry points.
    size_t varsize = std::max( m_rayGenerationPrograms.size() + numInternalEntryPoints, m_missPrograms.size() );
    RT_ASSERT( m_exceptionPrograms.size() == m_rayGenerationPrograms.size() );
    cort::GlobalScopeRecord* gs = nullptr;
    return (char*)( &gs->programs[varsize] ) - (char*)( gs );
}

void GlobalScope::writeRecord() const
{
    if( !recordIsAllocated() )
        return;
    cort::GlobalScopeRecord* gs = getObjectRecord<cort::GlobalScopeRecord>();
    for( unsigned int i                                 = 0; i < m_rayGenerationPrograms.size(); i++ )
        gs->programs[i].raygen                          = getSafeOffset( m_rayGenerationPrograms[i].get() );
    gs->programs[m_rayGenerationPrograms.size()].raygen = getSafeOffset( m_aabbComputeProgram.get() );
    for( unsigned int i                                = 0; i < m_exceptionPrograms.size(); i++ )
        gs->programs[i].exception                      = getSafeOffset( m_exceptionPrograms[i].get() );
    gs->programs[m_exceptionPrograms.size()].exception = getSafeOffset( m_aabbExceptionProgram.get() );
    for( unsigned int i      = 0; i < m_missPrograms.size(); i++ )
        gs->programs[i].miss = getSafeOffset( m_missPrograms[i].get() );
    LexicalScope::writeRecord();
}

void GlobalScope::notifyParents_offsetDidChange() const
{
    // Global scope has no parents (and the offset is always 0)
}

void GlobalScope::reallocateRecord()
{
    // GlobalScope always has an address of zero. So we initially try to allocate
    // this object at address zero. When it fails, we temporarily free all scopes' records
    // in the range 0..GlobalScope::getRecordSize(), reallocating the GlobalScope's record
    // and then again the ones temporarily released.
    // attempt to allocate GlobalScope's record at offset zero
    LexicalScope::reallocateRecord();
    if( getRecordOffset() == 0 )
        return;
    // since it was allocated nevertheless, we have to release the record
    // Note why we release it here and not in the following loop with the other scopes:
    // its offset might well be outside of the range 0..GlobalScope::getRecordSize() and
    // by releasing it already here we guarantee that the allocator will use the memory
    // inside the given range when reallocating the record (it is using the last(!) freed
    // block first - if it fits).
    releaseRecord();

    // now "release" as many scopes' records as required to make room
    size_t                     sizeNeeded = getRecordSize();
    std::vector<LexicalScope*> needsReallocation;
    for( auto scope : m_context->getObjectManager()->getLexicalScopes() )
    {
        if( scope != this && scope->recordIsAllocated() && scope->getRecordOffset() < sizeNeeded )
        {
            needsReallocation.push_back( scope );
            scope->releaseRecord();
        }
    }

    // then, reallocate GlobalScope's record
    LexicalScope::reallocateRecord();
    RT_ASSERT( getRecordOffset() == 0 );

    // finally, reallocate "released" scopes' records again
    for( auto scope : needsReallocation )
        if( scope != this )
            scope->reallocateRecord();
}


//------------------------------------------------------------------------
// Unresolved reference property
//------------------------------------------------------------------------

const GraphProperty<VariableReferenceID>& GlobalScope::getRemainingUnresolvedReferences() const
{
    return m_unresolvedRemaining;
}

void GlobalScope::sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool added ) const
{
    const VariableReference* varref = m_context->getProgramManager()->getVariableReferenceById( refid );
    if( varref->isInitialized() )
    {
        // Variable has a default value, so create an implicit binding to the default value
        m_context->getBindingManager()->addOrRemove_VariableBinding( refid, VariableReferenceBinding::makeDefaultValueBinding(), added );
    }
    else
    {
        // We do not propagate references further, but take this opportunity
        // to note that the reference is still unresolved. Validation will
        // later use this to determine if it is safe to launch.
        bool changed = m_unresolvedRemaining.addOrRemoveProperty( refid, added );
        if( changed && added )
        {
            // const cast required because subscribe is non-const
            GlobalScope* nonconst_this = const_cast<GlobalScope*>( this );
            nonconst_this->subscribeForValidation();
        }
    }
}


//------------------------------------------------------------------------
// Attachment
//------------------------------------------------------------------------

void GlobalScope::sendPropertyDidChange_Attachment( bool added ) const
{
    // Global scope should not have any attached children by the time its attachment changes
    // (in its destructor).
    RT_ASSERT_FAIL_MSG( "GlobalScope attchment changed" );
}


//------------------------------------------------------------------------
// Direct Caller
//------------------------------------------------------------------------

void GlobalScope::sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const
{
    // GlobalScope should not have any direct caller changes
    RT_ASSERT_FAIL_MSG( "GlobalScope direct caller changed" );
}
