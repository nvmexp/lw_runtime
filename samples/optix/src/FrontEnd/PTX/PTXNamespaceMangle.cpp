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

#include <FrontEnd/PTX/PTXNamespaceMangle.h>

#include <sstream>

std::string optix::PTXNamespaceMangle( const std::string& name, bool mangle_non_namespace_vars, bool add_z_prefix, const std::string& suffix )
{
    std::ostringstream out;
    size_t             namelen = name.size();
    size_t             colon_pos;
    size_t             last_pos = 0;
    if( add_z_prefix )
        out << "_Z";
    for( ;; )
    {
        colon_pos = name.find( "::", last_pos );
        if( colon_pos == std::string::npos )
            break;
        if( last_pos == 0 )
        {
            out << "N";
        }
        out << ( colon_pos - last_pos ) << name.substr( last_pos, colon_pos - last_pos );
        last_pos = colon_pos + 2;
    }
    if( mangle_non_namespace_vars || last_pos != 0 )
        out << ( namelen - last_pos );
    out << name.substr( last_pos );
    out << suffix;
    return out.str();
}
