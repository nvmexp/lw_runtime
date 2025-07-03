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

#include <Objects/SemanticType.h>

//
// Class to identify the "root" of a program - specific scope,
// semantic type and optional index. This uniquely defines all of the
// attachment points.  Used primarily for bound callable program
// references.
//

namespace optix {
struct ProgramRoot
{
    ProgramRoot() = default;
    ProgramRoot( int scopeid, SemanticType stype, unsigned int index );

    int          scopeid;
    SemanticType stype;
    unsigned int index;  // RayType for MISS, CLOSEST_HIT, ANY_HIT, Entry point for RAYGEN, EXCEPTION, zero for all others

    bool operator<( const ProgramRoot& rhs ) const;
    bool operator==( const ProgramRoot& rhs ) const;
    bool operator!=( const ProgramRoot& rhs ) const;

    std::string toString() const;
};
}  // end namespace optix
