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

#include <Memory/MAccess.h>

#include <algorithm>
#include <cstring>

namespace optix {

MAccess::MAccess()
{
    m_kind = NONE;
}

MAccess MAccess::makeLinear( char* ptr )
{
    MAccess ret;
    ret.m_kind       = LINEAR;
    ret.m_linear.ptr = ptr;
    return ret;
}

MAccess MAccess::makeMultiPitchedLinear( const PitchedLinearAccess* pitchedLinear, int count )
{
    RT_ASSERT( count >= 0 && count <= OPTIX_MAX_MIP_LEVELS );
    MAccess ret;
    ret.m_kind = MULTI_PITCHED_LINEAR;
    ret.m_pitchedLinear.resize( count );
    std::copy( pitchedLinear, pitchedLinear + count, ret.m_pitchedLinear.begin() );
    return ret;
}

MAccess MAccess::makeTexObject( const lwca::TexObject& tex, unsigned int offset )
{
    MAccess ret;
    ret.m_kind                  = TEX_OBJECT;
    ret.m_texObject.texObject   = tex;
    ret.m_texObject.indexOffset = offset;
    return ret;
}

MAccess MAccess::makeDemandTexObject( const lwca::TexObject& tex,
                                      unsigned int           offset,
                                      unsigned int           pageBegin,
                                      unsigned int           numPages,
                                      unsigned int           minMipLevel,
                                      unsigned int           maxMipLevel )
{
    MAccess ret;
    ret.m_kind                        = DEMAND_TEX_OBJECT;
    ret.m_demandTexObject.texObject   = tex;
    ret.m_demandTexObject.indexOffset = offset;
    ret.m_demandTexObject.startPage   = pageBegin;
    ret.m_demandTexObject.numPages    = numPages;
    ret.m_demandTexObject.minMipLevel = minMipLevel;
    ret.m_demandTexObject.maxMipLevel = maxMipLevel;
    return ret;
}

MAccess MAccess::makeLwdaSparseBacking( LWmemGenericAllocationHandle handle )
{
    MAccess ret;
    ret.m_kind                     = LWDA_SPARSE_BACKING;
    ret.m_lwdaSparseBacking.handle = handle;
    return ret;
}

MAccess MAccess::makeTexReference( unsigned int texUnit, unsigned int indexOffset )
{
    MAccess ret;
    ret.m_kind                     = TEX_REFERENCE;
    ret.m_texReference.texUnit     = texUnit;
    ret.m_texReference.indexOffset = indexOffset;
    return ret;
}

MAccess MAccess::makeDemandLoad( unsigned int pageBegin )
{
    MAccess ret;
    ret.m_kind                        = DEMAND_LOAD;
    ret.m_demandLoad.virtualPageBegin = pageBegin;
    return ret;
}

MAccess MAccess::makeLwdaSparse( unsigned int pageBegin )
{
    MAccess ret;
    ret.m_kind                        = LWDA_SPARSE;
    ret.m_lwdaSparse.virtualPageBegin = pageBegin;
    return ret;
}

MAccess MAccess::makeDemandLoadArray( unsigned int pageBegin, unsigned int numPages, unsigned int minMipLevel )
{
    MAccess ret;
    ret.m_kind                             = DEMAND_LOAD_ARRAY;
    ret.m_demandLoadArray.virtualPageBegin = pageBegin;
    ret.m_demandLoadArray.numPages         = numPages;
    ret.m_demandLoadArray.minMipLevel      = minMipLevel;
    return ret;
}

MAccess MAccess::makeDemandLoadTileArray()
{
    MAccess ret;
    ret.m_kind = DEMAND_LOAD_TILE_ARRAY;
    return ret;
}

MAccess MAccess::makeNone()
{
    MAccess ret;
    ret.m_kind = NONE;
    return ret;
}


bool MAccess::operator==( const MAccess& other ) const
{
    if( m_kind != other.m_kind )
        return false;
    switch( m_kind )
    {
        case LINEAR:
            return m_linear.ptr == other.m_linear.ptr;
        case MULTI_PITCHED_LINEAR:
            if( m_pitchedLinear.size() != other.m_pitchedLinear.size() )
                return false;
            return 0 == std::memcmp( m_pitchedLinear.data(), other.m_pitchedLinear.data(),
                                     m_pitchedLinear.size() * sizeof( PitchedLinearAccess ) );
        case TEX_OBJECT:
            return m_texObject.texObject.get() == other.m_texObject.texObject.get()
                   && m_texObject.indexOffset == other.m_texObject.indexOffset;
        case TEX_REFERENCE:
            return m_texReference.texUnit == other.m_texReference.texUnit
                   && m_texReference.indexOffset == other.m_texReference.indexOffset;
        case DEMAND_LOAD:
        case LWDA_SPARSE:
            return m_demandLoad.virtualPageBegin == other.m_demandLoad.virtualPageBegin;
        case LWDA_SPARSE_BACKING:
            return m_lwdaSparseBacking.handle == other.m_lwdaSparseBacking.handle;
        case DEMAND_LOAD_ARRAY:
            return ( m_pitchedLinear.size() == other.m_pitchedLinear.size() )
                   && ( std::memcmp( m_pitchedLinear.data(), other.m_pitchedLinear.data(),
                                     m_pitchedLinear.size() * sizeof( PitchedLinearAccess ) )
                        == 0 )
                   && ( m_demandLoad.virtualPageBegin == other.m_demandLoad.virtualPageBegin );
        case DEMAND_LOAD_TILE_ARRAY:
            // TODO: make this correct for tile array instead of copy/paste from DEMAND_LOAD_ARRAY
            return ( m_pitchedLinear.size() == other.m_pitchedLinear.size() )
                   && ( std::memcmp( m_pitchedLinear.data(), other.m_pitchedLinear.data(),
                                     m_pitchedLinear.size() * sizeof( PitchedLinearAccess ) )
                        == 0 )
                   && ( m_demandLoad.virtualPageBegin == other.m_demandLoad.virtualPageBegin );
        case DEMAND_TEX_OBJECT:
            return m_demandTexObject.texObject.get() == other.m_demandTexObject.texObject.get()
                   && m_demandTexObject.indexOffset == other.m_demandTexObject.indexOffset
                   && m_demandTexObject.startPage == other.m_demandTexObject.startPage
                   && m_demandTexObject.numPages == other.m_demandTexObject.numPages
                   && m_demandTexObject.minMipLevel == other.m_demandTexObject.minMipLevel
                   && m_demandTexObject.maxMipLevel == other.m_demandTexObject.maxMipLevel;
        case NONE:
            return true;
            // Default case intentionally omitted
    }
    return true;
}

bool MAccess::operator!=( const MAccess& other ) const
{
    return !operator==( other );
}

LinearAccess MAccess::getLinear() const
{
    RT_ASSERT( m_kind == LINEAR );
    return m_linear;
}

PitchedLinearAccess MAccess::getPitchedLinear( int idx ) const
{
    RT_ASSERT( m_kind == MULTI_PITCHED_LINEAR );
    RT_ASSERT( idx >= 0 && idx < (int)m_pitchedLinear.size() );
    return m_pitchedLinear[idx];
}

TexObjectAccess MAccess::getTexObject() const
{
    RT_ASSERT( m_kind == TEX_OBJECT );
    return m_texObject;
}

DemandTexObjectAccess MAccess::getDemandTexObject() const
{
    RT_ASSERT( m_kind == DEMAND_TEX_OBJECT );
    return m_demandTexObject;
}

TexReferenceAccess MAccess::getTexReference() const
{
    RT_ASSERT( m_kind == TEX_REFERENCE );
    return m_texReference;
}

LwdaSparseBackingAccess MAccess::getLwdaSparseBacking() const
{
    RT_ASSERT( m_kind == LWDA_SPARSE_BACKING );
    return m_lwdaSparseBacking;
}

LwdaSparseAccess MAccess::getLwdaSparse() const
{
    RT_ASSERT( m_kind == LWDA_SPARSE );
    return m_lwdaSparse;
}

DemandLoadAccess MAccess::getDemandLoad() const
{
    RT_ASSERT( m_kind == DEMAND_LOAD || m_kind == LWDA_SPARSE );
    return m_demandLoad;
}

DemandLoadArrayAccess MAccess::getDemandLoadArray() const
{
    RT_ASSERT( m_kind == DEMAND_LOAD_ARRAY );
    return m_demandLoadArray;
}

DemandLoadTileArrayAccess MAccess::getDemandLoadTileArray() const
{
    RT_ASSERT( m_kind == DEMAND_LOAD_TILE_ARRAY );
    return m_demandLoadTileArray;
}

// TODO: SGP: review all code that asserts PITCHED_LINEAR kind and replace with this function.
char* MAccess::getLinearPtr() const
{
    RT_ASSERT_MSG( m_kind == LINEAR, "Not a linear pointer" );
    return m_linear.ptr;
}

}  // namespace optix
