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

#include <Objects/Program.h>
#include <Objects/ProgramRoot.h>

using namespace optix;

ProgramRoot::ProgramRoot( int scopeid, SemanticType stype, unsigned int index )
    : scopeid( scopeid )
    , stype( stype )
    , index( index )
{
}

bool ProgramRoot::operator<( const ProgramRoot& rhs ) const
{
    if( stype != rhs.stype )
        return stype < rhs.stype;
    if( index != rhs.index )
        return index < rhs.index;
    return scopeid < rhs.scopeid;
}


bool ProgramRoot::operator==( const ProgramRoot& rhs ) const
{
    return scopeid == rhs.scopeid && stype == rhs.stype && index == rhs.index;
}

bool ProgramRoot::operator!=( const ProgramRoot& rhs ) const
{
    return scopeid != rhs.scopeid || stype != rhs.stype || index != rhs.index;
}

static inline bool needsIndex( SemanticType stype )
{
    return stype == ST_CLOSEST_HIT || stype == ST_ANY_HIT || stype == ST_MISS || stype == ST_RAYGEN || stype == ST_EXCEPTION;
}

std::string ProgramRoot::toString() const
{
    return "s" + std::to_string( scopeid ) + "." + semanticTypeToAbbreviationString( stype )
           + ( needsIndex( stype ) ? std::to_string( index ) : "" );
}
