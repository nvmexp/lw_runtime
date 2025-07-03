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

#include <iosfwd>
#include <string>

// GNU compiles introduce the macros major and minor (they expand to gnu_dev_major and
// gnu_dev_minor. I'm not sure that these #undefs will always be sufficient
#ifdef major
#undef major
#undef minor
#endif

namespace optix {
namespace lwca {
class ComputeCapability
{
  public:
    ComputeCapability();

    // version = major * 10 + minor
    explicit ComputeCapability( unsigned int version );
    ComputeCapability( unsigned int major, unsigned int minor );

    unsigned int major() const;
    unsigned int minor() const;
    unsigned int version() const;

    std::string toString( bool useDottedNotation = false ) const;
    friend std::ostream& operator<<( std::ostream& os, const ComputeCapability& bt );

    bool operator<( const ComputeCapability& c ) const;
    bool operator<=( const ComputeCapability& c ) const;
    bool operator>( const ComputeCapability& c ) const;
    bool operator>=( const ComputeCapability& c ) const;
    bool operator==( const ComputeCapability& c ) const;
    bool operator!=( const ComputeCapability& c ) const;

  private:
    unsigned int m_version = 0;
};

ComputeCapability SM_NONE();
ComputeCapability SM( unsigned int version );
}

}  // end namespace optix
