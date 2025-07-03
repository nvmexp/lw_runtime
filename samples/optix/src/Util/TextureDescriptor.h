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
#include <o6/optix.h>


namespace optix {
namespace lwca {
class TexRef;
}

struct TextureDescriptor
{
    TextureDescriptor()
    {
        wrapMode[0] = RT_WRAP_REPEAT;
        wrapMode[1] = RT_WRAP_REPEAT;
        wrapMode[2] = RT_WRAP_REPEAT;
    }
    RTwrapmode         wrapMode[3];
    RTfiltermode       minFilterMode    = RT_FILTER_LINEAR;
    RTfiltermode       magFilterMode    = RT_FILTER_LINEAR;
    RTfiltermode       mipFilterMode    = RT_FILTER_LINEAR;
    float              maxAnisotropy    = 1.f;
    float              maxMipLevelClamp = 1000.f;  // default value in OpenGL
    float              minMipLevelClamp = 0.f;
    float              mipLevelBias     = 0.f;
    RTtextureindexmode indexMode        = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES;
    RTtexturereadmode  readMode         = RT_TEXTURE_READ_NORMALIZED_FLOAT;

    void getLwdaTextureDescriptor( LWDA_TEXTURE_DESC* descTex ) const;
    void fillLwdaTexRef( lwca::TexRef& texef ) const;
    void validate() const;  // may throw an exception
};
}
