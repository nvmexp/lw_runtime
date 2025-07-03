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

class Array
{
  public:
    Array();

    // Get the low-level array
    LWarray get() const;

    // Creates a 1D or 2D LWCA array.
    static Array create( const LWDA_ARRAY_DESCRIPTOR& pAllocateArray, LWresult* returnResult = nullptr );

    // Creates a 3D LWCA array.
    static Array create( const LWDA_ARRAY3D_DESCRIPTOR& pAllocateArray, LWresult* returnResult = nullptr );

    // Creates an array from a graphics resource
    static Array create( LWgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel, LWresult* returnResult = nullptr );  // BL: This probably should take lwca::GraphicsResource instead.

    // Destroys a LWCA array.
    void destroy( LWresult* returnResult = nullptr );

    // Get a 1D or 2D LWCA array descriptor.
    LWDA_ARRAY_DESCRIPTOR getDescriptor( LWresult* returnResult = nullptr ) const;

    // Get a 3D LWCA array descriptor.
    LWDA_ARRAY3D_DESCRIPTOR getDescriptor3D( LWresult* returnResult = nullptr ) const;

  private:
    friend class TexRef;
    friend class GraphicsResource;
    friend class MipmappedArray;
    explicit Array( LWarray array );

    LWarray m_array;
};

class MipmappedArray
{
  public:
    MipmappedArray();

    // Get the low-level mipmappedArray
    LWmipmappedArray get() const;

    // Creates a LWCA mipmapped array.
    static MipmappedArray create( const LWDA_ARRAY3D_DESCRIPTOR& pMipmappedArrayDesc,
                                  unsigned int                   numMipmapLevels,
                                  LWresult*                      returnResult = nullptr );

    // Create from a graphics resource
    static MipmappedArray create( LWgraphicsResource resource, LWresult* returnResult = nullptr );

    // Destroys a LWCA mipmapped array.
    void destroy( LWresult* returnResult = nullptr );

    // Gets a mipmap level of a LWCA mipmapped array.
    Array getLevel( unsigned int level, LWresult* returnResult = nullptr ) const;

    bool isSparse() const;

    // Unmaps the entire mip level from the sparse array.
    void unmapSparseLevel( int mipLevel, int deviceOrdinal, LWresult* returnReult = nullptr );

    // Unmaps the mip tail from the sparse array.
    void unmapSparseMipTail( int deviceOrdinal, LWresult* returnResult = nullptr );

    // Gets the array's sparse texture properties.
    LWDA_ARRAY_SPARSE_PROPERTIES getSparseProperties( LWresult* returnResult = nullptr ) const;

  private:
    friend class GraphicsResource;
    friend class TexRef;
    explicit MipmappedArray( LWmipmappedArray mmarray );

    LWmipmappedArray m_mmarray;
};

}  // namespace lwca
}  // namespace optix
