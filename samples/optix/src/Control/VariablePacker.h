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

#include <map>
#include <string>
#include <unordered_map>

namespace optix {

class VariablePacker
{
  public:
    typedef std::unordered_map<std::string, int> OffsetMap;

    enum ReferenceKind
    {
        ReferenceBottom,
        ReferenceTop
    };
    VariablePacker( ReferenceKind kind = ReferenceBottom, int initial_size = 0 );
    ~VariablePacker();

    // Layout all new varialbles at once in alphabetical order
    void build();

    // Removes all variables and sets the size back to zero.
    void reset();

    // Returns the current size of the packed variables.
    int getLwrrentSize() const { return m_lwrrentSize; }

    // Returns the current size aligned to FRAME_ALIGNMENT.
    static const int FRAME_ALIGNMENT = 16;
    unsigned int     frameSize() const;

    // Insert the variable with the given size and alignment. The offset can be requested after calling build().
    // Such non-lazy offset callwlation allows to have stable order of variables and avoid recompilation
    void insertVariable( const std::string& name, int size, int align );

    // Insert the variable with the given size and alignment and return the offset.  If
    // the variable has already been added then return the stored offset.
    int offsetFor( const std::string& name, int size, int align );
    // Return the offset for the given variable.  Throws an exception if the variable
    // isn't found.
    int offsetFor( const std::string& name ) const;
    // Manually set the offset.  Inserts the variable if it isn't there already.
    void setOffset( const std::string& name, int offset, int size );
    // Return true if the given offset if found for name.  iter contains the result of
    // attempting to find name.
    bool foundOffset( OffsetMap::const_iterator& iter, const std::string& name ) const;

    void print( std::ostream& out ) const;

  protected:
    ReferenceKind m_kind;
    int           m_lwrrentSize;
    std::map<std::string, std::pair<int, int>> m_newSizeAndAlign;

    OffsetMap m_offsets;
};
}
