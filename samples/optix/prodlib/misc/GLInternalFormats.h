/*
 * Copyright (c) 2010 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#pragma once

#include <prodlib/exceptions/Exception.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/misc/BufferFormats.h>
#include <prodlib/misc/GLFunctions.h>


namespace prodlib {

inline int getGLTextureLwbeFlag( RTgltarget rtTarget )
{
    return rtTarget == RT_TARGET_GL_TEXTURE_LWBE_MAP || rtTarget == RT_TARGET_GL_TEXTURE_LWBE_MAP_ARRAY;
}

inline int getGLTextureLayeredFlag( RTgltarget rtTarget )
{
    return rtTarget == RT_TARGET_GL_TEXTURE_1D_ARRAY || rtTarget == RT_TARGET_GL_TEXTURE_2D_ARRAY
           || rtTarget == RT_TARGET_GL_TEXTURE_LWBE_MAP_ARRAY;
}

inline int getGLDimension( RTgltarget rtTarget )
{
    switch( rtTarget )
    {
        case RT_TARGET_GL_TEXTURE_1D:
            return 1;
        case RT_TARGET_GL_TEXTURE_2D:
        case RT_TARGET_GL_TEXTURE_RECTANGLE:
        case RT_TARGET_GL_TEXTURE_1D_ARRAY:
        case RT_TARGET_GL_RENDER_BUFFER:
            return 2;

        case RT_TARGET_GL_TEXTURE_3D:
        case RT_TARGET_GL_TEXTURE_2D_ARRAY:
        case RT_TARGET_GL_TEXTURE_LWBE_MAP:
        case RT_TARGET_GL_TEXTURE_LWBE_MAP_ARRAY:
            return 3;

        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Specified GL render target is not supported" );
    }
    return -1;
}

inline GLint getGLTextureTarget( RTgltarget rtTarget )
{
    GLint glBindTarget = 0;

    switch( rtTarget )
    {
        case RT_TARGET_GL_TEXTURE_1D:
            glBindTarget = GL_TEXTURE_1D;
            break;
        case RT_TARGET_GL_TEXTURE_2D:
            glBindTarget = GL_TEXTURE_2D;
            break;
        case RT_TARGET_GL_TEXTURE_RECTANGLE:
            glBindTarget = GL_TEXTURE_RECTANGLE_ARB;
            break;
        case RT_TARGET_GL_TEXTURE_3D:
            glBindTarget = GL_TEXTURE_3D;
            break;
        case RT_TARGET_GL_RENDER_BUFFER:
            glBindTarget = GL_RENDERBUFFER;
            break;
        case RT_TARGET_GL_TEXTURE_1D_ARRAY:
            glBindTarget = GL_TEXTURE_1D_ARRAY;
            break;
        case RT_TARGET_GL_TEXTURE_2D_ARRAY:
            glBindTarget = GL_TEXTURE_2D_ARRAY;
            break;
        case RT_TARGET_GL_TEXTURE_LWBE_MAP:
            glBindTarget = GL_TEXTURE_LWBE_MAP;
            break;
        case RT_TARGET_GL_TEXTURE_LWBE_MAP_ARRAY:
            glBindTarget = GL_TEXTURE_LWBE_MAP_ARRAY;
            break;
        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Specified GL render target is not supported" );
    }

    return glBindTarget;
}

inline GLint getGLTextureBindingTarget( RTgltarget rtTarget )
{
    GLint glBindTarget = 0;

    switch( rtTarget )
    {
        case RT_TARGET_GL_TEXTURE_1D:
            glBindTarget = GL_TEXTURE_BINDING_1D;
            break;
        case RT_TARGET_GL_TEXTURE_2D:
            glBindTarget = GL_TEXTURE_BINDING_2D;
            break;
        case RT_TARGET_GL_TEXTURE_RECTANGLE:
            glBindTarget = GL_TEXTURE_BINDING_RECTANGLE_ARB;
            break;
        case RT_TARGET_GL_TEXTURE_3D:
            glBindTarget = GL_TEXTURE_BINDING_3D;
            break;
        case RT_TARGET_GL_RENDER_BUFFER:
            glBindTarget = GL_RENDERBUFFER_BINDING;
            break;
        case RT_TARGET_GL_TEXTURE_1D_ARRAY:
            glBindTarget = GL_TEXTURE_BINDING_1D_ARRAY;
            break;
        case RT_TARGET_GL_TEXTURE_2D_ARRAY:
            glBindTarget = GL_TEXTURE_BINDING_2D_ARRAY;
            break;
        case RT_TARGET_GL_TEXTURE_LWBE_MAP:
            glBindTarget = GL_TEXTURE_BINDING_LWBE_MAP;
            break;
        case RT_TARGET_GL_TEXTURE_LWBE_MAP_ARRAY:
            glBindTarget = GL_TEXTURE_BINDING_LWBE_MAP_ARRAY;
            break;
        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Specified GL render target is not supported" );
    }

    return glBindTarget;
}

inline GLint getGLType( GLint internal_format )
{
    int format = 0;

    switch( internal_format )
    {
        case GL_R8I:
        case GL_RG8I:
        case GL_RGBA8I:
            format = GL_BYTE;
            break;

        case GL_R8:
        case GL_R8UI:
        case GL_RG8:
        case GL_RG8UI:
        case GL_RGBA8:
        case GL_RGBA8UI:
        case GL_SRGB_ALPHA:
        case GL_SRGB8_ALPHA8:
            format = GL_UNSIGNED_BYTE;
            break;

        case GL_R16I:
        case GL_RG16I:
        case GL_RGBA16I:
            format = GL_SHORT;
            break;

        case GL_R16:
        case GL_R16UI:
        case GL_RG16:
        case GL_RG16UI:
        case GL_RGBA16:
        case GL_RGBA16UI:
            format = GL_UNSIGNED_SHORT;
            break;

        case GL_R32I:
        case GL_RG32I:
        case GL_RGBA32I:
            format = GL_INT;
            break;

        case GL_R32UI:
        case GL_RG32UI:
        case GL_RGBA32UI:
            format = GL_UNSIGNED_INT;
            break;

        case GL_R32F:
        case GL_RG32F:
        case GL_RGBA32F:
            format = GL_FLOAT;
            break;

        case GL_R16F:
        case GL_RG16F:
        case GL_RGBA16F:
            format = GL_HALF_FLOAT;
            break;

        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unsupported texture format.", format );
    }

    return format;
}

inline GLint getGLFormat( GLint internal_format )
{
    int format = 0;

    switch( internal_format )
    {
        case GL_R8:
        case GL_R16:
        case GL_R16F:
        case GL_R32F:
            format = GL_RED;
            break;

        case GL_RG8:
        case GL_RG16:
        case GL_RG16F:
        case GL_RG32F:
            format = GL_RG;
            break;

        case GL_RGBA8:
        case GL_RGBA16:
        case GL_RGBA16F:
        case GL_RGBA32F:
        case GL_SRGB_ALPHA:
        case GL_SRGB8_ALPHA8:
            format = GL_RGBA;
            break;

        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
            format = GL_RED_INTEGER;
            break;

        case GL_RG8I:
        case GL_RG8UI:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_RG32I:
        case GL_RG32UI:
            format = GL_RG_INTEGER;
            break;

        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGBA16I:
        case GL_RGBA16UI:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            format = GL_RGBA_INTEGER;
            break;

        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unsupported GL texture format.", format );
    }

    return format;
}

inline GLint getDepthType( GLint internal_format )
{
    GLint format = GL_UNSIGNED_INT;

    switch( internal_format )
    {
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
            format = GL_UNSIGNED_INT;
            break;
        case GL_DEPTH_COMPONENT32F:
            format = GL_FLOAT;
            break;
    }

    return format;
}

inline RTformat glDepthToOptixFormat( GLint gl_internal_format )
{
    RTformat format = RT_FORMAT_UNSIGNED_INT;

    switch( gl_internal_format )
    {
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
            format = RT_FORMAT_UNSIGNED_INT;
            break;
        case GL_DEPTH_COMPONENT32F:
            format = RT_FORMAT_FLOAT;
            break;
    }

    return format;
}

inline RTformat glToOptixFormat( GLint gl_internal_format )
{
    RTformat format = RT_FORMAT_BYTE;

    switch( gl_internal_format )
    {
        // unnormalized integer formats
        case GL_R8I:
            format = RT_FORMAT_BYTE;
            break;
        case GL_R8:
        case GL_R8UI:
            format = RT_FORMAT_UNSIGNED_BYTE;
            break;

        case GL_R16I:
            format = RT_FORMAT_SHORT;
            break;
        case GL_R16:
        case GL_R16UI:
            format = RT_FORMAT_UNSIGNED_SHORT;
            break;

        case GL_R32I:
            format = RT_FORMAT_INT;
            break;
        case GL_R32UI:
            format = RT_FORMAT_UNSIGNED_INT;
            break;

        case GL_RG8I:
            format = RT_FORMAT_BYTE2;
            break;
        case GL_RG8:
        case GL_RG8UI:
            format = RT_FORMAT_UNSIGNED_BYTE2;
            break;

        case GL_RG16I:
            format = RT_FORMAT_SHORT2;
            break;
        case GL_RG16:
        case GL_RG16UI:
            format = RT_FORMAT_UNSIGNED_SHORT2;
            break;

        case GL_RG32I:
            format = RT_FORMAT_INT2;
            break;
        case GL_RG32UI:
            format = RT_FORMAT_UNSIGNED_INT2;
            break;

        case GL_RGBA8I:
            format = RT_FORMAT_BYTE4;
            break;
        case GL_RGBA8:
        case GL_RGBA8UI:
        case GL_SRGB_ALPHA:
        case GL_SRGB8_ALPHA8:
            format = RT_FORMAT_UNSIGNED_BYTE4;
            break;

        case GL_RGBA16I:
            format = RT_FORMAT_SHORT4;
            break;
        case GL_RGBA16:
        case GL_RGBA16UI:
            format = RT_FORMAT_UNSIGNED_SHORT4;
            break;

        case GL_RGBA32I:
            format = RT_FORMAT_INT4;
            break;
        case GL_RGBA32UI:
            format = RT_FORMAT_UNSIGNED_INT4;
            break;

        // half formats
        case GL_R16F:
            format = RT_FORMAT_HALF;
            break;
        case GL_RG16F:
            format = RT_FORMAT_HALF2;
            break;
        case GL_RGBA16F:
            format = RT_FORMAT_HALF4;
            break;

        // float formats
        case GL_R32F:
            format = RT_FORMAT_FLOAT;
            break;
        case GL_RG32F:
            format = RT_FORMAT_FLOAT2;
            break;
        case GL_RGBA32F:
            format = RT_FORMAT_FLOAT4;
            break;

        // compressed formats
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
            format = RT_FORMAT_UNSIGNED_BC1;
            break;
        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
            format = RT_FORMAT_UNSIGNED_BC2;
            break;
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
            format = RT_FORMAT_UNSIGNED_BC3;
            break;
        case GL_COMPRESSED_SIGNED_RED_RGTC1:
            format = RT_FORMAT_BC4;
            break;
        case GL_COMPRESSED_RED_RGTC1:
            format = RT_FORMAT_UNSIGNED_BC4;
            break;
        case GL_COMPRESSED_SIGNED_RG_RGTC2:
            format = RT_FORMAT_BC5;
            break;
        case GL_COMPRESSED_RG_RGTC2:
            format = RT_FORMAT_UNSIGNED_BC5;
            break;
#if !defined( __APPLE__ )
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
            format = RT_FORMAT_BC6H;
            break;
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
            format = RT_FORMAT_UNSIGNED_BC6H;
            break;
        case GL_COMPRESSED_RGBA_BPTC_UNORM:
            format = RT_FORMAT_UNSIGNED_BC7;
            break;
#endif

        // not supported
        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unsupported texture format.", format );
    }

    return format;
}

}  // end namespace prodlib
