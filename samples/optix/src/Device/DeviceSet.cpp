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

#include <Device/DeviceSet.h>

#include <Device/Device.h>

#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/math/Bits.h>

#include <algorithm>
#include <sstream>

using namespace optix;
using namespace prodlib;


DeviceSet::DeviceSet()
    : m_devices( 0 )
{
}

DeviceSet::DeviceSet( const DeviceSet& other )
    : m_devices( other.m_devices )
{
}

DeviceSet::DeviceSet( const const_iterator& iter )
    : m_devices( 1 << *iter )
{
}

DeviceSet::DeviceSet( const Device* device )
    : m_devices( 1 << device->allDeviceListIndex() )
{
}

DeviceSet::DeviceSet( const std::vector<Device*>& devices )
    : m_devices( 0 )
{
    for( const Device* device : devices )
    {
        insert( device );
    }
}

DeviceSet::DeviceSet( const std::vector<unsigned int>& allDeviceListIndices )
    : m_devices( 0 )
{
    for( const unsigned int allDeviceListIndex : allDeviceListIndices )
    {
        m_devices |= ( 1 << allDeviceListIndex );
    }
}

DeviceSet::DeviceSet( const int allDeviceListIndex )
    : m_devices( 1 << allDeviceListIndex )
{
}

DeviceSet::~DeviceSet()
{
}

// Manipulate individual devices
void DeviceSet::remove( const Device* device )
{
    m_devices &= ~( 1 << device->allDeviceListIndex() );
}

void DeviceSet::insert( const Device* device )
{
    m_devices |= ( 1 << device->allDeviceListIndex() );
}


// Union
DeviceSet DeviceSet::operator|( const DeviceSet& b ) const
{
    DeviceSet d = *this;
    d |= b;
    return d;
}

DeviceSet& DeviceSet::operator|=( const DeviceSet& b )
{
    m_devices |= b.m_devices;
    return *this;
}

// Intersection
DeviceSet DeviceSet::operator&( const DeviceSet& b ) const
{
    DeviceSet d = *this;
    d &= b;
    return d;
}

DeviceSet& DeviceSet::operator&=( const DeviceSet& b )
{
    m_devices &= b.m_devices;
    return *this;
}

// Difference
DeviceSet DeviceSet::operator-( const DeviceSet& b ) const
{
    DeviceSet d = *this;
    d -= b;
    return d;
}

DeviceSet& DeviceSet::operator-=( const DeviceSet& b )
{
    m_devices &= ~b.m_devices;
    return *this;
}

bool DeviceSet::operator==( const DeviceSet& b ) const
{
    return m_devices == b.m_devices;
}

bool DeviceSet::operator!=( const DeviceSet& b ) const
{
    return m_devices != b.m_devices;
}

DeviceSet DeviceSet::operator~() const
{
    DeviceSet r;
    r.m_devices = ~m_devices;
    return r;
}

bool DeviceSet::overlaps( const DeviceSet& b ) const
{
    return !( *this & b ).empty();
}

bool DeviceSet::empty() const
{
    return m_devices == 0;
}

unsigned int DeviceSet::count() const
{
    // Table of counts for up to 5 devices
    // clang-format off
    static const int COUNTS[] = {
        0, // 0
        1, // 1
        1, 2, // 10, 11
        1, 2, 2, 3, // 100, 101, 110, 111
        1, 2, 2, 3, 2, 3, 3, 4, // 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5
    };
    // clang-format on

    return ( m_devices < 32 ) ? COUNTS[m_devices] : popCount( m_devices );
}

bool DeviceSet::isSet( int allDeviceListIndex ) const
{
    const unsigned int mask = 1 << allDeviceListIndex;
    return ( m_devices & mask ) != 0;
}

bool DeviceSet::isSet( const Device* device ) const
{
    return isSet( device->allDeviceListIndex() );
}

void DeviceSet::clear()
{
    m_devices = 0;
}

std::string DeviceSet::toString() const
{
    if( empty() )
        return "{empty}";
    if( m_devices == ~0U )
        return "{all}";

    std::ostringstream out;
    out << "{";
    for( const_iterator iter = begin(); iter != end(); ++iter )
    {
        if( iter != begin() )
            out << ",";
        out << *iter;
    }
    out << "}";
    return out.str();
}

DeviceSet::position DeviceSet::const_iterator::operator*() const
{
    return pos;
}

DeviceSet::const_iterator& DeviceSet::const_iterator::operator++()
{
    if( pos + 1 == sizeof( parent->m_devices ) * 8 )
    {
        pos = -1;
    }
    else
    {
        unsigned int mask = ( 1 << ( pos + 1 ) ) - 1;
        pos               = leastSignificantBitSet( parent->m_devices & ~mask ) - 1;
    }
    return *this;
}

DeviceSet::position DeviceSet::const_iterator::operator++( int )
{
    position temp = **this;
    ++*this;
    return temp;
}


bool DeviceSet::const_iterator::operator!=( const const_iterator& b ) const
{
    return !( ( *this ) == b );
}

bool DeviceSet::const_iterator::operator==( const const_iterator& b ) const
{
    return parent == b.parent && pos == b.pos;
}

int DeviceSet::operator[]( int n ) const
{
    const_iterator it = begin();
    while( n-- )
        ++it;
    return *it;
}

int DeviceSet::getArrayPosition( int allDeviceListIndex ) const
{
    if( !isSet( allDeviceListIndex ) )
        throw IlwalidValue( RT_EXCEPTION_INFO, "allDeviceListIndex is not set for current DeviceSet", allDeviceListIndex );

    const unsigned int mask = ( 1u << allDeviceListIndex ) - 1u;
    return popCount( m_devices & mask );
}

DeviceSet::const_iterator::const_iterator( const DeviceSet* parent, position pos )
    : parent( parent )
    , pos( pos )
{
}

DeviceSet::const_iterator DeviceSet::begin() const
{
    return const_iterator( this, leastSignificantBitSet( m_devices ) - 1 );
}

DeviceSet::const_iterator DeviceSet::end() const
{
    return const_iterator( this, -1 );
}
