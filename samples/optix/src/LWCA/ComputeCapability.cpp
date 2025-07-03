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

#include <LWCA/ComputeCapability.h>

#include <prodlib/exceptions/Assert.h>
#include <sstream>
#include <string>

using namespace optix;
using namespace optix::lwca;

ComputeCapability::ComputeCapability()
{
}

ComputeCapability::ComputeCapability( unsigned int version )
    : m_version( version )
{
}

ComputeCapability::ComputeCapability( unsigned int major, unsigned int minor )
    : m_version( major * 10 + minor )
{
    RT_ASSERT( minor < 10 );
}

unsigned int ComputeCapability::major() const
{
    return m_version / 10;
}

unsigned int ComputeCapability::minor() const
{
    return m_version % 10;
}

unsigned int ComputeCapability::version() const
{
    return m_version;
}

std::string ComputeCapability::toString( bool useDottedNotation ) const
{
    std::stringstream ss;
    if( useDottedNotation )
        ss << major() << "." << minor();
    else
        ss << m_version;
    return ss.str();
}

namespace optix {
namespace lwca {
std::ostream& operator<<( std::ostream& os, const ComputeCapability& c )
{
    os << c.major() << c.minor();
    return os;
}
}
}

bool ComputeCapability::operator<( const ComputeCapability& c ) const
{
    return m_version < c.m_version;
}

bool ComputeCapability::operator<=( const ComputeCapability& c ) const
{
    return m_version <= c.m_version;
}

bool ComputeCapability::operator>( const ComputeCapability& c ) const
{
    return m_version > c.m_version;
}

bool ComputeCapability::operator>=( const ComputeCapability& c ) const
{
    return m_version >= c.m_version;
}

bool ComputeCapability::operator==( const ComputeCapability& c ) const
{
    return m_version == c.m_version;
}

bool ComputeCapability::operator!=( const ComputeCapability& c ) const
{
    return m_version != c.m_version;
}

ComputeCapability optix::lwca::SM_NONE()
{
    return ComputeCapability( 0 );
}

ComputeCapability optix::lwca::SM( unsigned int version )
{
    return ComputeCapability( version );
}
