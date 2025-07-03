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

#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Util/PersistentStream.h>
#include <Util/optixUuid.h>

#include <sstream>

using namespace optix;

VariableReference::VariableReference()
{
}

VariableReference::VariableReference( const CanonicalProgram* parent,
                                      const std::string&      variableName,
                                      unsigned short          variableToken,
                                      const VariableType&     vtype,
                                      bool                    isInitialized,
                                      const std::string&      annotation )
    : m_parent( parent )
    , m_refid( 0 - 1 )
    , m_variableName( variableName )
    , m_variableToken( variableToken )
    , m_vtype( vtype )
    , m_isInitialized( isInitialized )
    , m_annotation( annotation )
{
}

VariableReference::~VariableReference()
{
}

const CanonicalProgram* VariableReference::getParent() const
{
    return m_parent;
}

VariableReferenceID VariableReference::getReferenceID() const
{
    return m_refid;
}

const std::string& VariableReference::getInputName() const
{
    return m_variableName;
}

unsigned short VariableReference::getVariableToken() const
{
    return m_variableToken;
}

const VariableType& VariableReference::getType() const
{
    return m_vtype;
}

bool VariableReference::isInitialized() const
{
    return m_isInitialized;
}

void VariableReference::addTextureLookupKind( TextureLookup::LookupKind kind )
{
    if( (size_t)kind >= m_textureLookupKinds.size() )
        m_textureLookupKinds.resize( kind + 1, false );
    m_textureLookupKinds[kind] = true;
}

bool VariableReference::usesTextureLookupKind( TextureLookup::LookupKind kind ) const
{
    if( kind >= m_textureLookupKinds.size() )
        return false;
    return m_textureLookupKinds[kind];
}

std::string VariableReference::getInfoString() const
{
    std::ostringstream out;
    out << m_variableName << " (" << m_variableToken << "): " << m_refid;
    return out.str();
}

std::string VariableReference::getUniversallyUniqueName() const
{
    // MTA OP-1082 This function is called quite often, most of the times just for a string comparison.
    // It'd be more useful to compare the components instead of the full string in those cases.
    return m_parent->getUniversallyUniqueName() + "." + m_variableName;
}

const ProgramRoot& VariableReference::getBoundProgramRoot() const
{
    RT_ASSERT( m_linkedReference != nullptr );
    return m_root;
}

const VariableReference* VariableReference::getLinkedReference() const
{
    return m_linkedReference;
}

VariableReference::VariableReference( const CanonicalProgram* parent )
    : m_parent( parent )
{
}


void optix::readOrWrite( PersistentStream* stream, VariableReference* varref, const char* label )
{
    RT_ASSERT_MSG( varref->m_linkedReference == nullptr, "Persistence not implemented for bound callable programs" );
    auto                       tmp     = stream->pushObject( label, "VariableReference" );
    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );

    // TODO: Do refid and variableToken still need to be serialized?
    // They are no longer used for validation and get assigned in ProgramManager::loadCanonicalProgramFromDiskCache
    // after successfully loading the CP from the diskcache

    readOrWrite( stream, &varref->m_refid, "refid" );
    readOrWrite( stream, &varref->m_variableName, "variableName" );
    readOrWrite( stream, &varref->m_variableToken, "variableToken" );
    readOrWrite( stream, &varref->m_vtype, "vtype" );
    readOrWrite( stream, &varref->m_isInitialized, "isInitialized" );
    readOrWrite( stream, &varref->m_pointerMayEscape, "pointerMayEscape" );
    readOrWrite( stream, &varref->m_hasBufferStores, "hasBufferStores" );
    readOrWrite( stream, &varref->m_hasIllFormedAccess, "hasIllFormedAccess" );
    readOrWrite( stream, &varref->m_annotation, "annotation" );
    readOrWrite( stream, &varref->m_textureLookupKinds, "textureLookupKinds" );
}
