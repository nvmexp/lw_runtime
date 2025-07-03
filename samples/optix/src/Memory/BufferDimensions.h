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

#include <internal/optix_declarations.h>

#include <string>


namespace optix {

class BufferDimensions
{
  public:
    BufferDimensions() = default;
    BufferDimensions( RTformat     format,
                      size_t       elementSize,
                      unsigned int dimensionality,
                      size_t       width,
                      size_t       height,
                      size_t       depth,
                      unsigned int levels  = 1,
                      bool         lwbe    = false,
                      bool         layered = false );

    bool operator==( const BufferDimensions& b ) const;
    bool operator!=( const BufferDimensions& b ) const;

    RTformat     format() const;
    size_t       elementSize() const;
    unsigned int dimensionality() const;
    size_t       width() const;
    size_t       height() const;
    size_t       depth() const;
    unsigned int mipLevelCount() const;
    bool         isLwbe() const;
    bool         isLayered() const;

    void setFormat( RTformat fmt, size_t elementSize );
    void setMipLevelCount( unsigned int levels );
    void setSize( size_t w );
    void setSize( size_t w, size_t h );
    void setSize( size_t w, size_t h, size_t d );
    void setLayered( bool layered ) { m_layered = layered; }

    bool   zeroSized() const;
    size_t getTotalSizeInBytes() const;
    size_t getLevelOffsetInBytes( unsigned int level ) const;
    size_t getLevelSizeInBytes( unsigned int level ) const;
    size_t levelWidth( unsigned int level ) const;
    size_t levelHeight( unsigned int level ) const;
    size_t levelDepth( unsigned int level ) const;
    size_t getNaturalPitchInBytes() const;
    size_t getLevelNaturalPitchInBytes( unsigned int level ) const;

    std::string toString() const;

  private:
    RTformat     m_format         = RT_FORMAT_BYTE;
    size_t       m_elementSize    = 1;
    unsigned int m_dimensionality = 1;
    size_t       m_width          = 0;
    size_t       m_height         = 0;
    size_t       m_depth          = 0;
    unsigned int m_levels         = 1;
    bool         m_lwbe           = false;
    bool         m_layered        = false;
};
}
