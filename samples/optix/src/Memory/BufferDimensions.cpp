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

#include <Memory/BufferDimensions.h>

#include <prodlib/misc/BufferFormats.h>
#include <prodlib/misc/RTFormatUtil.h>

#include <sstream>

using namespace optix;
using namespace prodlib;


BufferDimensions::BufferDimensions( RTformat     format,
                                    size_t       elementSize,
                                    unsigned int dimensionality,
                                    size_t       width,
                                    size_t       height,
                                    size_t       depth,
                                    unsigned int levels,
                                    bool         lwbe,
                                    bool         layered )
    : m_format( format )
    , m_elementSize( elementSize )
    , m_dimensionality( dimensionality )
    , m_width( width )
    , m_height( height )
    , m_depth( depth )
    , m_levels( levels )
    , m_lwbe( lwbe )
    , m_layered( layered )
{
}

bool BufferDimensions::operator==( const BufferDimensions& b ) const
{
    return m_format == b.m_format && m_elementSize == b.m_elementSize && m_dimensionality == b.m_dimensionality
           && m_width == b.m_width && m_height == b.m_height && m_depth == b.m_depth && m_levels == b.m_levels
           && m_lwbe == b.m_lwbe && m_layered == b.m_layered;
}

bool BufferDimensions::operator!=( const BufferDimensions& b ) const
{
    return !( *this == b );
}

RTformat BufferDimensions::format() const
{
    return m_format;
}

size_t BufferDimensions::elementSize() const
{
    return m_elementSize;
}

unsigned int BufferDimensions::dimensionality() const
{
    return m_dimensionality;
}

size_t BufferDimensions::width() const
{
    return m_width;
}

size_t BufferDimensions::height() const
{
    return m_height;
}

size_t BufferDimensions::depth() const
{
    return m_depth;
}

unsigned int BufferDimensions::mipLevelCount() const
{
    return m_levels;
}

bool BufferDimensions::isLwbe() const
{
    return m_lwbe;
}

bool BufferDimensions::isLayered() const
{
    return m_layered;
}

void BufferDimensions::setMipLevelCount( unsigned int levels )
{
    m_levels = levels;
}

void BufferDimensions::setFormat( RTformat fmt, size_t elementSize )
{
    m_format      = fmt;
    m_elementSize = elementSize;
}

void BufferDimensions::setSize( size_t w )
{
    m_dimensionality = 1;
    m_width          = w;
    m_height = m_depth = 1;
}

void BufferDimensions::setSize( size_t w, size_t h )
{
    m_dimensionality = 2;
    m_width          = w;
    m_height         = h;
    m_depth          = 1;
}

void BufferDimensions::setSize( size_t w, size_t h, size_t d )
{
    m_dimensionality = 3;
    m_width          = w;
    m_height         = h;
    m_depth          = d;
}

bool BufferDimensions::zeroSized() const
{
    return m_width == 0 || m_height == 0 || m_depth == 0;
}

size_t BufferDimensions::getTotalSizeInBytes() const
{
    return getBufferTotalByteSize( m_width, m_height, m_depth, m_levels, m_elementSize, m_lwbe || m_layered );
}

size_t BufferDimensions::getLevelOffsetInBytes( unsigned int level ) const
{
    return getBufferTotalByteSize( m_width, m_height, m_depth, level, m_elementSize, m_lwbe || m_layered );
}

size_t BufferDimensions::getLevelSizeInBytes( unsigned int level ) const
{
    return getBufferLevelByteSize( m_width, m_height, m_depth, level, m_elementSize, m_lwbe || m_layered );
}

size_t BufferDimensions::levelWidth( unsigned int level ) const
{
    return getBufferLevelWidth( m_width, level );
}

size_t BufferDimensions::levelHeight( unsigned int level ) const
{
    return getBufferLevelHeight( m_height, level );
}

size_t BufferDimensions::levelDepth( unsigned int level ) const
{
    return getBufferLevelDepth( m_depth, level, m_lwbe || m_layered );
}

size_t BufferDimensions::getNaturalPitchInBytes() const
{
    return m_elementSize * m_width;
}

size_t BufferDimensions::getLevelNaturalPitchInBytes( unsigned int level ) const
{
    return m_elementSize * levelWidth( level );
}

std::string BufferDimensions::toString() const
{
    std::ostringstream out;
    switch( m_dimensionality )
    {
        case 1:
            out << m_width;
            break;
        case 2:
            out << m_width << "x" << m_height;
            break;
        case 3:
            out << m_width << "x" << m_height << "x" << m_depth;
            break;
    }
    out << "/" << ::toString( m_format, m_elementSize );
    return out.str();
}
