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

#include <lwca.h>

namespace optix {
namespace lwca {

class Array;
class MipmappedArray;
class Module;

class TexRef
{
  public:
    TexRef();

    // Get the low-level tex
    LWtexref       get();
    const LWtexref get() const;


    // gets the address associated with a texture reference.
    LWdeviceptr getAddress( LWresult* returnResult = nullptr );

    // gets the addressing mode used by a texture reference.
    LWaddress_mode getAddressMode( int dim, LWresult* returnResult = nullptr );

    // gets the array bound to a texture reference.
    Array getArray( LWresult* returnResult = nullptr );

    // gets the filter-mode used by a texture reference.
    LWfilter_mode getFilterMode( LWresult* returnResult = nullptr );

    // gets the flags used by a texture reference.
    unsigned int getFlags( LWresult* returnResult = nullptr );

    // gets the format used by a texture reference.
    void getFormat( LWarray_format& format, int& numChannels, LWresult* returnResult = nullptr );

    // gets the maximum anisotropy for a texture reference.
    int getMaxAnisotropy( LWresult* returnResult = nullptr );

    // gets the mipmap filtering mode for a texture reference.
    LWfilter_mode getMipmapFilterMode( LWresult* returnResult = nullptr );

    // gets the mipmap level bias for a texture reference.
    float getMipmapLevelBias( LWresult* returnResult = nullptr );

    // gets the min/max mipmap level clamps for a texture reference.
    void getMipmapLevelClamp( float& pminMipmapLevelClamp, float& pmaxMipmapLevelClamp, LWresult* returnResult = nullptr );

    // gets the mipmapped array bound to a texture reference.
    MipmappedArray getMipmappedArray( LWresult* returnResult = nullptr );

    // Binds an address as a texture reference and returns the byte offset.
    size_t setAddress( LWdeviceptr dptr, size_t bytes, LWresult* returnResult = nullptr );

    // Binds an address as a 2D texture reference.
    void setAddress2D( const LWDA_ARRAY_DESCRIPTOR& desc, LWdeviceptr dptr, size_t Pitch, LWresult* returnResult = nullptr );

    // sets the addressing mode for a texture reference.
    void setAddressMode( int dim, LWaddress_mode am, LWresult* returnResult = nullptr );

    // Binds an array as a texture reference.
    void setArray( const Array& array, unsigned int Flags, LWresult* returnResult = nullptr );

    // sets the filtering mode for a texture reference.
    void setFilterMode( LWfilter_mode fm, LWresult* returnResult = nullptr );

    // sets the flags for a texture reference.
    void setFlags( unsigned int Flags, LWresult* returnResult = nullptr );

    // sets the format for a texture reference.
    void setFormat( LWarray_format fmt, int numPackedComponents, LWresult* returnResult = nullptr );

    // sets the maximum anisotropy for a texture reference.
    void setMaxAnisotropy( unsigned int maxAniso, LWresult* returnResult = nullptr );

    // sets the mipmap filtering mode for a texture reference.
    void setMipmapFilterMode( LWfilter_mode fm, LWresult* returnResult = nullptr );

    // sets the mipmap level bias for a texture reference.
    void setMipmapLevelBias( float bias, LWresult* returnResult = nullptr );

    // sets the mipmap min/max mipmap level clamps for a texture reference.
    void setMipmapLevelClamp( float minMipmapLevelClamp, float maxMipmapLevelClamp, LWresult* returnResult = nullptr );

    // Binds a mipmapped array to a texture reference.
    void setMipmappedArray( const MipmappedArray& hMipmappedArray, unsigned int Flags, LWresult* returnResult = nullptr );

  protected:
    friend class Module;
    explicit TexRef( LWtexref texref );

    LWtexref m_texRef;
};

}  // namespace lwca
}  // namespace optix
