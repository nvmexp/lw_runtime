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

#include <LWCA/TexRef.h>
#include <Util/TextureDescriptor.h>

#include <prodlib/exceptions/IlwalidValue.h>

#include <cstring>  // memset

using namespace optix;

static const char* RTtextureindexmodeToString[] = {"RT_TEXTURE_INDEX_NORMALIZED_COORDINATES",
                                                   "RT_TEXTURE_INDEX_ARRAY_INDEX"};
static const char* RTwrapmodeToString[] = {"RT_WRAP_REPEAT", "RT_WRAP_CLAMP_TO_EDGE", "RT_WRAP_MIRROR",
                                           "RT_WRAP_CLAMP_TO_BORDER"};
static const char* RTfiltermodeToString[] = {"RT_FILTER_NEAREST", "RT_FILTER_LINEAR", "RT_FILTER_NONE"};

static inline LWfilter_mode getLwdaFilterMode( RTfiltermode filter )
{
    switch( filter )
    {
        case RT_FILTER_NEAREST:
            return LW_TR_FILTER_MODE_POINT;

        case RT_FILTER_LINEAR:
            return LW_TR_FILTER_MODE_LINEAR;

        case RT_FILTER_NONE:
            return LW_TR_FILTER_MODE_POINT;

        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unknown filtering mode: ", filter );
    }
}

static inline LWaddress_mode getLwdaAddressMode( RTwrapmode wrap )
{
    switch( wrap )
    {
        case RT_WRAP_REPEAT:
            return LW_TR_ADDRESS_MODE_WRAP;

        case RT_WRAP_CLAMP_TO_EDGE:
            return LW_TR_ADDRESS_MODE_CLAMP;

        case RT_WRAP_MIRROR:
            return LW_TR_ADDRESS_MODE_MIRROR;

        case RT_WRAP_CLAMP_TO_BORDER:
            return LW_TR_ADDRESS_MODE_BORDER;

        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unknown wrap mode: ", wrap );
    }
}

static inline unsigned int getLwdaTextureFlags( RTtexturereadmode readmode, RTtextureindexmode indexmode )
{
    unsigned int flags = 0;

    if( readmode == RT_TEXTURE_READ_ELEMENT_TYPE || readmode == RT_TEXTURE_READ_ELEMENT_TYPE_SRGB )
        flags |= LW_TRSF_READ_AS_INTEGER;

    if( readmode & RT_TEXTURE_READ_ELEMENT_TYPE_SRGB )
        flags |= LW_TRSF_SRGB;

    if( indexmode == RT_TEXTURE_INDEX_NORMALIZED_COORDINATES )
        flags |= LW_TRSF_NORMALIZED_COORDINATES;

    return flags;
}

void TextureDescriptor::getLwdaTextureDescriptor( LWDA_TEXTURE_DESC* descTex ) const
{
    memset( descTex, 0, sizeof( *descTex ) );
    descTex->addressMode[0]      = getLwdaAddressMode( wrapMode[0] );
    descTex->addressMode[1]      = getLwdaAddressMode( wrapMode[1] );
    descTex->addressMode[2]      = getLwdaAddressMode( wrapMode[2] );
    descTex->filterMode          = getLwdaFilterMode( magFilterMode );
    descTex->flags               = getLwdaTextureFlags( readMode, indexMode );
    descTex->maxAnisotropy       = maxAnisotropy;
    descTex->mipmapFilterMode    = getLwdaFilterMode( mipFilterMode );
    descTex->mipmapLevelBias     = mipLevelBias;
    descTex->minMipmapLevelClamp = minMipLevelClamp;
    descTex->maxMipmapLevelClamp = maxMipLevelClamp;
}

void TextureDescriptor::fillLwdaTexRef( lwca::TexRef& texref ) const
{
    texref.setAddressMode( 0, getLwdaAddressMode( wrapMode[0] ) );
    texref.setAddressMode( 1, getLwdaAddressMode( wrapMode[1] ) );
    texref.setAddressMode( 2, getLwdaAddressMode( wrapMode[2] ) );
    texref.setFilterMode( getLwdaFilterMode( magFilterMode ) );
    texref.setFlags( getLwdaTextureFlags( readMode, indexMode ) );
    texref.setMaxAnisotropy( maxAnisotropy );
    texref.setMipmapFilterMode( getLwdaFilterMode( mipFilterMode ) );
    texref.setMipmapLevelBias( mipLevelBias );
    texref.setMipmapLevelClamp( minMipLevelClamp, maxMipLevelClamp );
}

static bool isAnyEq( const RTwrapmode wrapMode[3], RTwrapmode mode, int& wrapIndex )
{
    for( int index = 0; index < 3; index++ )
    {
        if( wrapMode[index] == mode )
        {
            wrapIndex = index;
            return true;
        }
    }
    return false;
}


void TextureDescriptor::validate() const
{
    // Unsupported texture modes by LWCA
    int wrapIndex = -1;
    if( ( indexMode == RT_TEXTURE_INDEX_NORMALIZED_COORDINATES
          && isAnyEq( wrapMode, RT_WRAP_CLAMP_TO_BORDER, wrapIndex ) && magFilterMode == RT_FILTER_LINEAR )
        || ( indexMode == RT_TEXTURE_INDEX_ARRAY_INDEX && isAnyEq( wrapMode, RT_WRAP_REPEAT, wrapIndex ) && magFilterMode == RT_FILTER_LINEAR )
        || ( indexMode == RT_TEXTURE_INDEX_ARRAY_INDEX && isAnyEq( wrapMode, RT_WRAP_MIRROR, wrapIndex ) && magFilterMode == RT_FILTER_NEAREST )
        || ( indexMode == RT_TEXTURE_INDEX_ARRAY_INDEX && isAnyEq( wrapMode, RT_WRAP_CLAMP_TO_BORDER, wrapIndex )
             && magFilterMode == RT_FILTER_NEAREST ) )
    {
        std::ostringstream s;
        s << RTtextureindexmodeToString[indexMode] << ", " << RTwrapmodeToString[wrapMode[wrapIndex]] << ", "
          << RTfiltermodeToString[magFilterMode];
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Unsupported combination of texture index, wrap and filter modes: ", s.str() );
    }
}
