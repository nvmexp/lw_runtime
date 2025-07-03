
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

#include <Control/VariablePacker.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/misc/IomanipHelpers.h>

#include <iostream>

using namespace optix;
using namespace prodlib;


VariablePacker::VariablePacker( ReferenceKind kind, int initial_size )
    : m_kind( kind )
    , m_lwrrentSize( initial_size )
{
}

VariablePacker::~VariablePacker()
{
}

void VariablePacker::reset()
{
    m_lwrrentSize = 0;
    m_offsets.clear();
    m_newSizeAndAlign.clear();
}

unsigned int VariablePacker::frameSize() const
{
    return ( m_lwrrentSize + FRAME_ALIGNMENT - 1 ) & ~( FRAME_ALIGNMENT - 1 );
}

void VariablePacker::insertVariable( const std::string& name, int size, int align )
{
    RT_ASSERT( size != 0 );
    RT_ASSERT( align != 0 );

    std::pair<std::string, std::pair<int, int>> x;

    x.first         = name;
    x.second.first  = size;
    x.second.second = align;

    m_newSizeAndAlign.insert( x );
}

void VariablePacker::build()
{
    for( std::map<std::string, std::pair<int, int>>::const_iterator iter = m_newSizeAndAlign.begin();
         iter != m_newSizeAndAlign.end(); iter++ )
    {
        const std::string& name  = iter->first;
        int                size  = iter->second.first;
        int                align = iter->second.second;

        offsetFor( name, size, align );
    }
    m_newSizeAndAlign.clear();
}

int VariablePacker::offsetFor( const std::string& name, int size, int align )
{
    OffsetMap::iterator iter = m_offsets.find( name );
    if( iter != m_offsets.end() )
        return iter->second;

    RT_ASSERT( size != 0 );
    RT_ASSERT( align != 0 );

    if( m_kind == ReferenceBottom )
    {
        // Stack pointer is at bottom of stack, so offsets are positive
        // and point to the bottom of the object
        m_lwrrentSize   = ( m_lwrrentSize + align - 1 ) & ~( align - 1 );
        m_offsets[name] = m_lwrrentSize;
        m_lwrrentSize += size;
    }
    else
    {
        // Stack pointer is at top of stack, so offsets are negative
        // and point to the bottom of the object
        m_lwrrentSize += size;
        m_lwrrentSize   = ( m_lwrrentSize + align - 1 ) & ~( align - 1 );
        m_offsets[name] = -m_lwrrentSize;
    }
    RT_ASSERT( m_offsets[name] % align == 0 );
    return m_offsets[name];
}

int VariablePacker::offsetFor( const std::string& name ) const
{
    OffsetMap::const_iterator iter = m_offsets.find( name );
    if( iter != m_offsets.end() )
        return iter->second;
    throw IlwalidValue( RT_EXCEPTION_INFO, "VariablePacker does not have offset for: ", name );
}

// Manually set the offset.  Inserts the variable if it isn't there already.
void VariablePacker::setOffset( const std::string& name, int offset, int size )
{
    if( m_kind == ReferenceBottom )
    {
        m_offsets[name] = offset;
        m_lwrrentSize   = offset + size;
    }
    else
    {
        m_lwrrentSize   = -offset;
        m_offsets[name] = offset;
    }
}

bool VariablePacker::foundOffset( OffsetMap::const_iterator& iter, const std::string& name ) const
{
    iter = m_offsets.find( name );
    return iter != m_offsets.end();
}

void VariablePacker::print( std::ostream& out ) const
{
    IOSSaver saver( out );
    out << "VariablePacker\n";
    out << "  m_lwrrentSize = " << m_lwrrentSize << "\n";
    std::map<int, std::string> sorted_by_offset;
    for( const auto& offset : m_offsets )
    {
        sorted_by_offset.insert( std::make_pair( offset.second, offset.first ) );
    }
    RT_ASSERT( sorted_by_offset.size() == m_offsets.size() );
    out << std::right;
    for( auto& iter : sorted_by_offset )
    {
        out << "\t[" << std::setw( 4 ) << iter.first << "] = " << iter.second << "\n";
    }
}
