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

#include <FrontEnd/Canonical/Mangle.h>

#include <sstream>

namespace optix {

//------------------------------------------------------------------------------
// Canonical Mangling
// The canonical mangling is the PTX mangling, at least for now.  We
// can revisit this when we support direct LLVM inputs.
//------------------------------------------------------------------------------

std::string canonicalMangleVariableName( const std::string& name )
{
    return ptxMangleVariableName( name );
}

std::string canonicalDemangleVariableName( const std::string& name )
{
    return ptxDemangleVariableName( name );
}

std::string canonicalPrependNamespace( const std::string& name, const std::string& prepend )
{
    // Assuming name is already demangled, find the last oclwrance of "::".  That is the
    // insertion point.
    size_t pos = name.find_last_of( "::" );
    if( pos == std::string::npos )
        return prepend + name;
    std::string result( name );
    result.insert( pos + 1, prepend );
    return result;
}

//------------------------------------------------------------------------------
// PTX Mangling
//------------------------------------------------------------------------------

std::string ptxMangleVariableName( const std::string& name )
{
    if( name.find( "::" ) == std::string::npos )
        return name;

    std::ostringstream out;
    out << "_ZN";
    size_t namelen = name.size();
    size_t lwr     = 0;
    while( lwr < namelen )
    {
        size_t colon_pos = name.find( "::", lwr );
        if( colon_pos == std::string::npos )
            colon_pos = namelen;
        size_t length = colon_pos - lwr;
        out << length << name.substr( lwr, colon_pos - lwr );
        lwr = colon_pos + 2;
    }
    out << "E";
    return out.str();
}

std::string ptxDemangleVariableName( const std::string& name )
{
    if( name.substr( 0, 3 ) != "_ZN" )
        return name;

    std::ostringstream out;
    size_t             nbeg = 3;
    size_t             nend;

    for( ;; )
    {
        for( nend = nbeg; nend != name.size() && isdigit( name[nend] ); nend++ )
            ;
        if( nend == nbeg )
            break;
        if( nbeg != 3 )
            out << "::";

        int len = atoi( name.substr( nbeg, nend - nbeg ).c_str() );
        out << name.substr( nend, len );
        nend += len;
        nbeg = nend;
    }

    return out.str();
}

}  // namespace optix
