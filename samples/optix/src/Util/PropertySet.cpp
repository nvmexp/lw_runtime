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


#include <Util/PropertySet.h>
#include <Util/Serializer.h>

#include <iomanip>

using namespace optix;

void PropertySet::print( std::ostream& out, const std::string& indent ) const
{
    for( const auto& prop : m_properties )
        out << indent << std::left << std::setw( 25 ) << prop.first << ": " << std::setw( 25 ) << prop.second << std::endl;
}

void PropertySet::serialize( Serializer& serializer ) const
{
    optix::serialize( serializer, m_properties );
}

void PropertySet::deserialize( Deserializer& deserializer )
{
    optix::deserialize( deserializer, m_properties );
}
