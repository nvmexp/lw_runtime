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


#include <ExelwtionStrategy/Common/VariableSpecialization.h>
#include <Util/PersistentStream.h>
#include <Util/optixUuid.h>

using namespace optix;

void VariableSpecialization::setGeneric( unsigned short token )
{
    lookupKind           = GenericLookup;
    dynamicVariableToken = token;
}

void VariableSpecialization::setSingleBinding( const VariableReferenceBinding& b )
{
    lookupKind    = SingleScope;
    singleBinding = b;
}

void VariableSpecialization::setSingleId( int id )
{
    lookupKind = SingleId;
    singleId   = id;
}

void VariableSpecialization::setUnused()
{
    lookupKind = Unused;
}

bool VariableSpecialization::operator==( const VariableSpecialization& b ) const
{
    return lookupKind == b.lookupKind && singleBinding == b.singleBinding && singleId == b.singleId
           && accessKind == b.accessKind && singleOffset == b.singleOffset && dynamicVariableToken == b.dynamicVariableToken
           && texheapUnit == b.texheapUnit && isReadOnly == b.isReadOnly;
}

bool VariableSpecialization::operator!=( const VariableSpecialization& b ) const
{
    return !( *this == b );
}

bool VariableSpecialization::operator<( const VariableSpecialization& b ) const
{
    if( lookupKind != b.lookupKind )
        return lookupKind < b.lookupKind;
    if( accessKind != b.accessKind )
        return accessKind < b.accessKind;
    if( singleBinding != b.singleBinding )
        return singleBinding < b.singleBinding;
    if( singleId != b.singleId )
        return singleId < b.singleId;
    if( singleOffset != b.singleOffset )
        return singleOffset < b.singleOffset;
    if( dynamicVariableToken != b.dynamicVariableToken )
        return dynamicVariableToken < b.dynamicVariableToken;
    if( texheapUnit != b.texheapUnit )
        return texheapUnit < b.texheapUnit;
    if( isReadOnly != b.isReadOnly )
        return isReadOnly < b.isReadOnly;
    return false;
}

bool VariableSpecialization::preferLDG() const
{
    return accessKind == PitchedLinearPreferLDG;
}

void optix::readOrWrite( PersistentStream* stream, VariableSpecialization* vs, const char* label )
{
    auto                       tmp     = stream->pushObject( label, "VariableSpecialization" );
    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );
    readOrWrite( stream, &vs->lookupKind, "lookupKind" );
    readOrWrite( stream, &vs->accessKind, "accessKind" );
    readOrWrite( stream, &vs->singleBinding, "singleBinding" );
    readOrWrite( stream, &vs->singleId, "singleId" );
    readOrWrite( stream, &vs->singleOffset, "singleOffset" );
    readOrWrite( stream, &vs->dynamicVariableToken, "dynamicVariableToken" );
    readOrWrite( stream, &vs->texheapUnit, "texheapUnit" );
    readOrWrite( stream, &vs->isReadOnly, "isReadOnly" );
}
