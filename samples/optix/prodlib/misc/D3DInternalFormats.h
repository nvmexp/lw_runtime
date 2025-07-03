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

#if defined( _WIN32 )

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <prodlib/exceptions/Exception.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/misc/BufferFormats.h>

#if _MSC_VER < 1700
#include <../support/DX/DXGIFormat.h>
#include <../support/DX/d3d9types.h>
#else
#include <d3d11.h>
#include <d3d9.h>
#endif

#include <memory>


namespace prodlib {

using std::unique_ptr;
struct IUnknownDeleter : public std::default_delete<IUnknown>
{
    void operator()( IUnknown* p ) const { p->Release(); }
};

template <typename IWhatever>
unique_ptr<IWhatever, IUnknownDeleter> make_unique_from_COM( IWhatever* p )
{
    unique_ptr<IWhatever, IUnknownDeleter> pw( p );
    return pw;
}

inline REFIID get_com_id( IDirect3DSurface9* )
{
    return IID_IDirect3DSurface9;
}
inline REFIID get_com_id( IDirect3DTexture9* )
{
    return IID_IDirect3DTexture9;
}
inline REFIID get_com_id( IDirect3DLwbeTexture9* )
{
    return IID_IDirect3DLwbeTexture9;
}
inline REFIID get_com_id( IDirect3DVolumeTexture9* )
{
    return IID_IDirect3DVolumeTexture9;
}
inline REFIID get_com_id( IDirect3DVertexBuffer9* )
{
    return IID_IDirect3DVertexBuffer9;
}
inline REFIID get_com_id( IDirect3DIndexBuffer9* )
{
    return IID_IDirect3DIndexBuffer9;
}

inline REFIID get_com_id( ID3D11Texture1D* )
{
    return IID_ID3D11Texture1D;
}
inline REFIID get_com_id( ID3D11Texture2D* )
{
    return IID_ID3D11Texture2D;
}
inline REFIID get_com_id( ID3D11Texture3D* )
{
    return IID_ID3D11Texture3D;
}
inline REFIID get_com_id( ID3D11Buffer* )
{
    return IID_ID3D11Buffer;
}

template <typename To, typename From>
inline unique_ptr<To, IUnknownDeleter> com_cast( From src )
{
    To* dst = 0;
    OacAssert( src->QueryInterface( get_com_id( dst ), (void**)&dst ) == S_OK, "QueryInterface failed" );
    return unique_ptr<To, IUnknownDeleter>( dst );
}

typedef unique_ptr<IDirect3DSurface9, IUnknownDeleter>       IDirect3DSurface9Ptr;
typedef unique_ptr<IDirect3DTexture9, IUnknownDeleter>       IDirect3DTexture9Ptr;
typedef unique_ptr<IDirect3DLwbeTexture9, IUnknownDeleter>   IDirect3DLwbeTexture9Ptr;
typedef unique_ptr<IDirect3DVolumeTexture9, IUnknownDeleter> IDirect3DVolumeTexture9Ptr;
typedef unique_ptr<IDirect3DVertexBuffer9, IUnknownDeleter>  IDirect3DVertexBuffer9Ptr;
typedef unique_ptr<IDirect3DIndexBuffer9, IUnknownDeleter>   IDirect3DIndexBuffer9Ptr;
typedef unique_ptr<ID3D11Texture1D, IUnknownDeleter>         ID3D11Texture1DPtr;
typedef unique_ptr<ID3D11Texture2D, IUnknownDeleter>         ID3D11Texture2DPtr;
typedef unique_ptr<ID3D11Texture3D, IUnknownDeleter>         ID3D11Texture3DPtr;
typedef unique_ptr<ID3D11Buffer, IUnknownDeleter>            ID3D11BufferPtr;
typedef unique_ptr<ID3D11Resource, IUnknownDeleter>          ID3D11ResourcePtr;
typedef unique_ptr<ID3D11DeviceContext, IUnknownDeleter>     ID3D11DeviceContextPtr;
typedef unique_ptr<ID3D11Device, IUnknownDeleter>            ID3D11DevicePtr;
typedef unique_ptr<IDXGISwapChain, IUnknownDeleter>          IDXGISwapChainPtr;
typedef unique_ptr<ID3D11RenderTargetView, IUnknownDeleter>  ID3D11RenderTargetViewPtr;


#define FORMATCASE( x, y )                                                                                             \
    case x:                                                                                                            \
        format = y;                                                                                                    \
        break;

inline RTformat d3dtoOptixFormat( D3DFORMAT d3d_format )
{
    RTformat format = RT_FORMAT_BYTE;

    switch( d3d_format )
    {
        FORMATCASE( D3DFMT_R16F, RT_FORMAT_HALF );
        FORMATCASE( D3DFMT_R32F, RT_FORMAT_FLOAT );
        FORMATCASE( D3DFMT_L16, RT_FORMAT_UNSIGNED_SHORT );
        FORMATCASE( D3DFMT_L8, RT_FORMAT_UNSIGNED_BYTE );
        FORMATCASE( D3DFMT_A8, RT_FORMAT_UNSIGNED_BYTE );

        FORMATCASE( D3DFMT_G16R16F, RT_FORMAT_HALF2 );
        FORMATCASE( D3DFMT_G32R32F, RT_FORMAT_FLOAT2 );
        FORMATCASE( D3DFMT_G16R16, RT_FORMAT_UNSIGNED_SHORT2 );
        FORMATCASE( D3DFMT_V16U16, RT_FORMAT_SHORT2 );
        FORMATCASE( D3DFMT_A8L8, RT_FORMAT_UNSIGNED_BYTE2 );
        FORMATCASE( D3DFMT_V8U8, RT_FORMAT_BYTE2 );

        FORMATCASE( D3DFMT_A16B16G16R16F, RT_FORMAT_HALF4 );
        FORMATCASE( D3DFMT_A32B32G32R32F, RT_FORMAT_FLOAT4 );
        FORMATCASE( D3DFMT_A16B16G16R16, RT_FORMAT_UNSIGNED_SHORT4 );
        FORMATCASE( D3DFMT_A8R8G8B8, RT_FORMAT_UNSIGNED_BYTE4 );
        FORMATCASE( D3DFMT_X8R8G8B8, RT_FORMAT_UNSIGNED_BYTE4 );
        FORMATCASE( D3DFMT_A8B8G8R8, RT_FORMAT_UNSIGNED_BYTE4 );
        FORMATCASE( D3DFMT_X8B8G8R8, RT_FORMAT_UNSIGNED_BYTE4 );
        FORMATCASE( D3DFMT_Q16W16V16U16, RT_FORMAT_SHORT4 );
        FORMATCASE( D3DFMT_Q8W8V8U8, RT_FORMAT_BYTE4 );

        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unsupported D3D texture format.", format );
    }

    return format;
}

inline RTformat d3dtoOptixFormat( DXGI_FORMAT d3d_format )
{
    RTformat format = RT_FORMAT_BYTE;

    switch( d3d_format )
    {
        FORMATCASE( DXGI_FORMAT_R32G32B32A32_FLOAT, RT_FORMAT_FLOAT4 );
        FORMATCASE( DXGI_FORMAT_R32G32B32A32_UINT, RT_FORMAT_UNSIGNED_INT4 );
        FORMATCASE( DXGI_FORMAT_R32G32B32A32_SINT, RT_FORMAT_INT4 );

        FORMATCASE( DXGI_FORMAT_R16G16B16A16_FLOAT, RT_FORMAT_HALF4 );
        FORMATCASE( DXGI_FORMAT_R16G16B16A16_UNORM, RT_FORMAT_UNSIGNED_SHORT4 );
        FORMATCASE( DXGI_FORMAT_R16G16B16A16_UINT, RT_FORMAT_UNSIGNED_SHORT4 );
        FORMATCASE( DXGI_FORMAT_R16G16B16A16_SNORM, RT_FORMAT_SHORT4 );
        FORMATCASE( DXGI_FORMAT_R16G16B16A16_SINT, RT_FORMAT_SHORT4 );

        FORMATCASE( DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, RT_FORMAT_UNSIGNED_BYTE4 );
        FORMATCASE( DXGI_FORMAT_R8G8B8A8_UNORM, RT_FORMAT_UNSIGNED_BYTE4 );
        FORMATCASE( DXGI_FORMAT_R8G8B8A8_UINT, RT_FORMAT_UNSIGNED_BYTE4 );
        FORMATCASE( DXGI_FORMAT_R8G8B8A8_SNORM, RT_FORMAT_BYTE4 );
        FORMATCASE( DXGI_FORMAT_R8G8B8A8_SINT, RT_FORMAT_BYTE4 );

        FORMATCASE( DXGI_FORMAT_R32G32_FLOAT, RT_FORMAT_FLOAT2 );
        FORMATCASE( DXGI_FORMAT_R32G32_UINT, RT_FORMAT_UNSIGNED_INT2 );
        FORMATCASE( DXGI_FORMAT_R32G32_SINT, RT_FORMAT_INT2 );

        FORMATCASE( DXGI_FORMAT_R16G16_FLOAT, RT_FORMAT_HALF2 );
        FORMATCASE( DXGI_FORMAT_R16G16_UNORM, RT_FORMAT_UNSIGNED_SHORT2 );
        FORMATCASE( DXGI_FORMAT_R16G16_UINT, RT_FORMAT_UNSIGNED_SHORT2 );
        FORMATCASE( DXGI_FORMAT_R16G16_SNORM, RT_FORMAT_SHORT2 );
        FORMATCASE( DXGI_FORMAT_R16G16_SINT, RT_FORMAT_SHORT2 );

        FORMATCASE( DXGI_FORMAT_R8G8_UNORM, RT_FORMAT_UNSIGNED_BYTE2 );
        FORMATCASE( DXGI_FORMAT_R8G8_UINT, RT_FORMAT_UNSIGNED_BYTE2 );
        FORMATCASE( DXGI_FORMAT_R8G8_SNORM, RT_FORMAT_BYTE2 );
        FORMATCASE( DXGI_FORMAT_R8G8_SINT, RT_FORMAT_BYTE2 );

        FORMATCASE( DXGI_FORMAT_R32_FLOAT, RT_FORMAT_FLOAT );
        FORMATCASE( DXGI_FORMAT_R32_UINT, RT_FORMAT_UNSIGNED_INT );
        FORMATCASE( DXGI_FORMAT_R32_SINT, RT_FORMAT_INT );

        FORMATCASE( DXGI_FORMAT_R16_FLOAT, RT_FORMAT_HALF );
        FORMATCASE( DXGI_FORMAT_R16_UNORM, RT_FORMAT_UNSIGNED_SHORT );
        FORMATCASE( DXGI_FORMAT_R16_UINT, RT_FORMAT_UNSIGNED_SHORT );
        FORMATCASE( DXGI_FORMAT_R16_SNORM, RT_FORMAT_SHORT );
        FORMATCASE( DXGI_FORMAT_R16_SINT, RT_FORMAT_SHORT );

        FORMATCASE( DXGI_FORMAT_R8_UNORM, RT_FORMAT_UNSIGNED_BYTE );
        FORMATCASE( DXGI_FORMAT_R8_UINT, RT_FORMAT_UNSIGNED_BYTE );
        FORMATCASE( DXGI_FORMAT_R8_SNORM, RT_FORMAT_BYTE );
        FORMATCASE( DXGI_FORMAT_R8_SINT, RT_FORMAT_BYTE );

        //FORMATCASE( DXGI_FORMAT_BC1_TYPELESS, RT_FORMAT_ );
        FORMATCASE( DXGI_FORMAT_BC1_UNORM, RT_FORMAT_UNSIGNED_BC1 );
        FORMATCASE( DXGI_FORMAT_BC1_UNORM_SRGB, RT_FORMAT_UNSIGNED_BC1 );
        //FORMATCASE( DXGI_FORMAT_BC4_TYPELESS, RT_FORMAT_ );
        FORMATCASE( DXGI_FORMAT_BC4_UNORM, RT_FORMAT_UNSIGNED_BC4 );
        FORMATCASE( DXGI_FORMAT_BC4_SNORM, RT_FORMAT_BC4 );
        //FORMATCASE( DXGI_FORMAT_BC2_TYPELESS, RT_FORMAT_ );
        FORMATCASE( DXGI_FORMAT_BC2_UNORM, RT_FORMAT_UNSIGNED_BC2 );
        FORMATCASE( DXGI_FORMAT_BC2_UNORM_SRGB, RT_FORMAT_UNSIGNED_BC2 );
        //FORMATCASE( DXGI_FORMAT_BC3_TYPELESS, RT_FORMAT_ );
        FORMATCASE( DXGI_FORMAT_BC3_UNORM, RT_FORMAT_UNSIGNED_BC3 );
        FORMATCASE( DXGI_FORMAT_BC3_UNORM_SRGB, RT_FORMAT_UNSIGNED_BC3 );
        //FORMATCASE( DXGI_FORMAT_BC5_TYPELESS, RT_FORMAT_ );
        FORMATCASE( DXGI_FORMAT_BC5_UNORM, RT_FORMAT_UNSIGNED_BC5 );
        FORMATCASE( DXGI_FORMAT_BC5_SNORM, RT_FORMAT_BC5 );
        //FORMATCASE( DXGI_FORMAT_BC6H_TYPELESS, RT_FORMAT_ );
        FORMATCASE( DXGI_FORMAT_BC6H_UF16, RT_FORMAT_UNSIGNED_BC6H );
        FORMATCASE( DXGI_FORMAT_BC6H_SF16, RT_FORMAT_BC6H );
        //FORMATCASE( DXGI_FORMAT_BC7_TYPELESS, RT_FORMAT_ );
        FORMATCASE( DXGI_FORMAT_BC7_UNORM, RT_FORMAT_UNSIGNED_BC7 );
        FORMATCASE( DXGI_FORMAT_BC7_UNORM_SRGB, RT_FORMAT_UNSIGNED_BC7 );

        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unsupported D3D texture format.", format );
    }

    return format;
}
#undef FORMATCASE
}  // end namespace prodlib

#endif
