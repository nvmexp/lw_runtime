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
//
#include <FrontEnd/Canonical/CallSiteIdentifier.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/VariableReference.h>

#include <Context/Context.h>
#include <Context/ProgramManager.h>
#include <Context/UpdateManager.h>

#include <corelib/misc/String.h>

using namespace optix;

CallSiteIdentifier::CallSiteIdentifier( const std::string& name, const CanonicalProgram* parent )
    : m_InputName( name )
    , m_parent( parent )
{
}

CallSiteIdentifier::CallSiteIdentifier( const CanonicalProgram* parent )
    : m_parent( parent )
{
}

CallSiteIdentifier::~CallSiteIdentifier() NOEXCEPT_FALSE
{
    // Problem: How to update potential callees that were set through variable bindings?
    //          Those lwrrently remain in the potential callees.
    // RT_ASSERT_MSG( m_potentialCallees.empty(), "Potential callees not empty at destroy" );
}

void CallSiteIdentifier::addOrRemovePotentialCallees( const std::vector<CanonicalProgramID>& calleeIds, bool added )
{
    bool changed = false;
    for( CanonicalProgramID calleeId : calleeIds )
    {
        bool change = m_potentialCallees.addOrRemoveProperty( calleeId, added );

        if( change )
        {
            changed                    = true;
            const CanonicalProgram* cp = m_parent->getContext()->getProgramManager()->getCanonicalProgramById( calleeId );
            cp->receivePropertyDidChange_DirectCaller( m_parent->getID(), added );
            cp->receivePropertyDidChange_calledFromCallsite( this, added );
        }
    }
    if( changed )
        m_parent->getContext()->getUpdateManager()->eventCanonicalProgramPotentialCalleesDidChange( getParent(), this, added );
}

std::vector<CanonicalProgramID> CallSiteIdentifier::getPotentialCallees() const
{
    std::vector<CanonicalProgramID> callees;
    callees.reserve( m_potentialCallees.size() );
    for( CanonicalProgramID csId : m_potentialCallees )
        callees.push_back( csId );
    return callees;
}

const std::string& CallSiteIdentifier::getInputName() const
{
    return m_InputName;
}

std::string CallSiteIdentifier::getUniversallyUniqueName() const
{
    return m_parent->getUniversallyUniqueName() + "." + m_InputName;
}

std::string CallSiteIdentifier::generateCallSiteUniqueName( const VariableReference* varref )
{
    // Adding the CP's PTX hash to the variable reference's name to avoid collisions
    // with user declared call sites (in case a variable has the same name as a call site)
    // Result: cp->getUniversallyUniqueName() + "." + varref->getName() + cp->getPTXHash()
    return varref->getUniversallyUniqueName() + corelib::ptr_to_string( varref->getParent()->getPTXHash(), 32 );
}

void optix::readOrWrite( PersistentStream* stream, CallSiteIdentifier* csId, const char* label )
{
    auto                       tmp     = stream->pushObject( label, "CallSiteIdentifier" );
    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );

    readOrWrite( stream, &csId->m_InputName, "inputName" );
}
