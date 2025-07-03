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

#include <Objects/VariableReferenceBinding.h>

#include <Objects/Variable.h>
#include <Util/PersistentStream.h>
#include <Util/optixUuid.h>

#include <iomanip>
#include <sstream>

using namespace optix;


VariableReferenceBinding::VariableReferenceBinding( ObjectClass scopeClass, size_t offset )
    : m_scopeClass( scopeClass )
    , m_offset( offset )
    , m_isDefaultValue( false )
{
}

VariableReferenceBinding VariableReferenceBinding::makeDefaultValueBinding()
{
    VariableReferenceBinding b;
    b.m_isDefaultValue = true;
    return b;
}

bool VariableReferenceBinding::operator<( const VariableReferenceBinding& rhs ) const
{
    if( scopeClass() != rhs.scopeClass() )
        return scopeClass() < rhs.scopeClass();
    if( offset() != rhs.offset() )
        return offset() < rhs.offset();
    return isDefaultValue() < rhs.isDefaultValue();
}

bool VariableReferenceBinding::operator==( const VariableReferenceBinding& rhs ) const
{
    return scopeClass() == rhs.scopeClass() && offset() == rhs.offset() && isDefaultValue() == rhs.isDefaultValue();
}

bool VariableReferenceBinding::operator!=( const VariableReferenceBinding& rhs ) const
{
    return !( *this == rhs );
}

std::string VariableReferenceBinding::toString() const
{
    std::ostringstream out;
    out << getNameForClass( scopeClass() ) << "+";
    if( offset() == Variable::ILWALID_OFFSET )
        out << "??";
    else
        out << "0x" << std::hex << offset();
    return out.str();
}

void optix::readOrWrite( PersistentStream* stream, VariableReferenceBinding* vrb, const char* label )
{
    auto                       tmp     = stream->pushObject( label, "VariableReferenceBinding" );
    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );
    readOrWrite( stream, &vrb->m_scopeClass, "scopeClass" );
    readOrWrite( stream, &vrb->m_offset, "offset" );
    readOrWrite( stream, &vrb->m_isDefaultValue, "isDefaultValue" );
}
