

/* this ALWAYS GENERATED file contains the definitions for the interfaces */


 /* File created by MIDL compiler version 8.00.0608 */
/* @@MIDL_FILE_HEADING(  ) */



/* verify that the <rpcndr.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 500
#endif

/* verify that the <rpcsal.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCSAL_H_VERSION__
#define __REQUIRED_RPCSAL_H_VERSION__ 100
#endif

#include "rpc.h"
#include "rpcndr.h"

#ifndef __RPCNDR_H_VERSION__
#error this stub requires an updated version of <rpcndr.h>
#endif // __RPCNDR_H_VERSION__


#ifndef __d3d11on12_h__
#define __d3d11on12_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

/* Forward Declarations */ 

/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"
#include "d3d11.h"
#include "d3d12.h"

#ifdef __cplusplus
extern "C"{
#endif 


/* interface __MIDL_itf_d3d11on12_0000_0000 */
/* [local] */ 


///////////////////////////////////////////////////////////////////////////
// D3D11CreateDeviceForD3D12
// ------------------
//
// pDevice
//      Specifies a pre-existing D3D12 device to use for D3D11 interop.
//      May not be NULL.
// Flags
//      Any of those dolwmented for D3D11CreateDeviceAndSwapChain.
// pFeatureLevels
//      Array of any of the following:
//          D3D_FEATURE_LEVEL_11_1
//          D3D_FEATURE_LEVEL_11_0
//          D3D_FEATURE_LEVEL_10_1
//          D3D_FEATURE_LEVEL_10_0
//          D3D_FEATURE_LEVEL_9_3
//          D3D_FEATURE_LEVEL_9_2
//          D3D_FEATURE_LEVEL_9_1
//       The first feature level which is less than or equal to the
//       D3D12 device's feature level will be used to perform D3D11 validation.
//       Creation will fail if no acceptable feature levels are provided.
//       Providing NULL will default to the D3D12 device's feature level.
// FeatureLevels
//      Size of feature levels array.
// SDKVersion
//      SDK version. Use the D3D11_SDK_VERSION macro.
// ppDevice
//      Pointer to returned interface. May be NULL.
// ppImmediateContext
//      Pointer to returned interface. May be NULL.
// pChosenFeatureLevel
//      Pointer to returned feature level. May be NULL.
//
// Return Values
//  Any of those dolwmented for 
//          D3D11CreateDevice
//
///////////////////////////////////////////////////////////////////////////
typedef HRESULT (WINAPI* PFN_D3D11_CREATE_DEVICE_FOR_D3D12)( _In_ IUnknown*, UINT, 
    _In_reads_opt_( FeatureLevels ) CONST D3D_FEATURE_LEVEL*, 
    UINT FeatureLevels, UINT, _COM_Outptr_opt_ ID3D11Device**, _COM_Outptr_opt_ ID3D11DeviceContext**, 
    _Out_opt_ D3D_FEATURE_LEVEL* );

HRESULT WINAPI D3D11CreateDeviceForD3D12(
    _In_ IUnknown* pDevice,
    UINT Flags,
    _In_reads_opt_( FeatureLevels ) CONST D3D_FEATURE_LEVEL* pFeatureLevels,
    UINT FeatureLevels,
    UINT SDKVersion,
    _COM_Outptr_opt_ ID3D11Device** ppDevice,
    _COM_Outptr_opt_ ID3D11DeviceContext** ppImmediateContext,
    _Out_opt_ D3D_FEATURE_LEVEL* pChosenFeatureLevel );

#if !defined( D3D12_NO_HELPERS ) && defined( __cplusplus )
}
struct CD3D12_HEAP_PROPERTIES_FROM11 : public CD3D12_HEAP_PROPERTIES
{
    CD3D12_HEAP_PROPERTIES_FROM11() : CD3D12_HEAP_PROPERTIES() { }
    static CD3D12_HEAP_PROPERTIES ColwertD3D11Usage( D3D11_USAGE Usage, UINT CPUAccessFlags )
    {
        switch (Usage)
        {
        case D3D11_USAGE_DEFAULT:
            if (CPUAccessFlags)
            {
                return CD3D12_HEAP_PROPERTIES(
                    CPUAccessFlags == D3D11_CPU_ACCESS_WRITE ? D3D12_CPU_PAGE_WRITE_COMBINE : D3D12_CPU_PAGE_WRITE_BACK
                    , D3D12_MEMORY_POOL_L0
                    );
            }
            else
            {
                return CD3D12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT );
            }
        case D3D11_USAGE_IMMUTABLE:
            return CD3D12_HEAP_PROPERTIES( CPUAccessFlags ? D3D12_HEAP_TYPE( 0 ) : D3D12_HEAP_TYPE_DEFAULT );
        case D3D11_USAGE_DYNAMIC:
            return CD3D12_HEAP_PROPERTIES( CPUAccessFlags == D3D11_CPU_ACCESS_WRITE ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE( 0 ) );
        case D3D11_USAGE_STAGING:
            return CD3D12_HEAP_PROPERTIES( CPUAccessFlags == D3D11_CPU_ACCESS_WRITE
                ? D3D12_HEAP_TYPE_UPLOAD
                : (CPUAccessFlags ? D3D12_HEAP_TYPE_READBACK : D3D12_HEAP_TYPE( 0 ))
                );
        default: return CD3D12_HEAP_PROPERTIES( D3D12_HEAP_TYPE( 0 ) );
        }
    }
    explicit CD3D12_HEAP_PROPERTIES_FROM11( const D3D11_BUFFER_DESC& o )
    {
        *static_cast<CD3D12_HEAP_PROPERTIES*>(this) = ColwertD3D11Usage( o.Usage, o.CPUAccessFlags );
    }
    explicit CD3D12_HEAP_PROPERTIES_FROM11( const D3D11_TEXTURE1D_DESC& o )
    {
        *static_cast<CD3D12_HEAP_PROPERTIES*>(this) = ColwertD3D11Usage( o.Usage, o.CPUAccessFlags );
    }
    explicit CD3D12_HEAP_PROPERTIES_FROM11( const D3D11_TEXTURE2D_DESC& o )
    {
        *static_cast<CD3D12_HEAP_PROPERTIES*>(this) = ColwertD3D11Usage( o.Usage, o.CPUAccessFlags );
    }
    explicit CD3D12_HEAP_PROPERTIES_FROM11( const D3D11_TEXTURE3D_DESC& o )
    {
        *static_cast<CD3D12_HEAP_PROPERTIES*>(this) = ColwertD3D11Usage( o.Usage, o.CPUAccessFlags );
    }
};
struct CD3D12_HEAP_DESC_FROM11 : public CD3D12_HEAP_DESC
{
    CD3D12_HEAP_DESC_FROM11() : CD3D12_HEAP_DESC() { }
    static D3D12_HEAP_MISC_FLAG ColwertD3D11MiscFlags( UINT miscFlags )
    {
        D3D12_HEAP_MISC_FLAG MiscFlags = D3D12_HEAP_MISC_NONE;
        if ((miscFlags & (D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX)) == 
            D3D11_RESOURCE_MISC_SHARED_NTHANDLE)
        {
            MiscFlags |= D3D12_HEAP_MISC_SHARED;
        }
        return MiscFlags;
    }
    explicit CD3D12_HEAP_DESC_FROM11( const D3D11_BUFFER_DESC& o )
    {
        *static_cast<CD3D12_HEAP_DESC*>(this) = CD3D12_HEAP_DESC( 0, CD3D12_HEAP_PROPERTIES_FROM11( o ), 0, ColwertD3D11MiscFlags( o.MiscFlags ) );
    }
    explicit CD3D12_HEAP_DESC_FROM11( const D3D11_TEXTURE1D_DESC& o )
    {
        *static_cast<CD3D12_HEAP_DESC*>(this) = CD3D12_HEAP_DESC( 0, CD3D12_HEAP_PROPERTIES_FROM11( o ), 0, ColwertD3D11MiscFlags( o.MiscFlags ) );
    }
    explicit CD3D12_HEAP_DESC_FROM11( const D3D11_TEXTURE2D_DESC& o )
    {
        *static_cast<CD3D12_HEAP_DESC*>(this) = CD3D12_HEAP_DESC( 0, CD3D12_HEAP_PROPERTIES_FROM11( o ), 0, ColwertD3D11MiscFlags( o.MiscFlags ) );
    }
    explicit CD3D12_HEAP_DESC_FROM11( const D3D11_TEXTURE3D_DESC& o )
    {
        *static_cast<CD3D12_HEAP_DESC*>(this) = CD3D12_HEAP_DESC( 0, CD3D12_HEAP_PROPERTIES_FROM11( o ), 0, ColwertD3D11MiscFlags( o.MiscFlags ) );
    }
};
struct CD3D12_RESOURCE_DESC_FROM11 : public CD3D12_RESOURCE_DESC
{
    CD3D12_RESOURCE_DESC_FROM11() : CD3D12_RESOURCE_DESC() { }
    static D3D12_RESOURCE_MISC_FLAG ColwertD3D11Flags( UINT bindFlags )
    {
        D3D12_RESOURCE_MISC_FLAG MiscFlags = D3D12_RESOURCE_MISC_NONE;
        if (!(bindFlags & D3D11_BIND_SHADER_RESOURCE) && (bindFlags & D3D11_BIND_DEPTH_STENCIL)) MiscFlags |= D3D12_RESOURCE_MISC_DENY_SHADER_RESOURCE;
        if (bindFlags & D3D11_BIND_RENDER_TARGET) MiscFlags |= D3D12_RESOURCE_MISC_ALLOW_RENDER_TARGET;
        if (bindFlags & D3D11_BIND_DEPTH_STENCIL) MiscFlags |= D3D12_RESOURCE_MISC_ALLOW_DEPTH_STENCIL;
        if (bindFlags & D3D11_BIND_UNORDERED_ACCESS) MiscFlags |= D3D12_RESOURCE_MISC_ALLOW_UNORDERED_ACCESS;
        return MiscFlags;
    }
    explicit CD3D12_RESOURCE_DESC_FROM11( const D3D11_BUFFER_DESC& o )
    {
        *static_cast<CD3D12_RESOURCE_DESC*>(this) = Buffer( o.ByteWidth, D3D12_RESOURCE_MISC_NONE );
        MiscFlags |= ColwertD3D11Flags( o.BindFlags );
    }
    explicit CD3D12_RESOURCE_DESC_FROM11( const D3D11_TEXTURE1D_DESC& o )
    {
        *static_cast<CD3D12_RESOURCE_DESC*>(this) = Tex1D( o.Format, o.Width, UINT16( o.ArraySize ), UINT16( o.MipLevels ) );
        MiscFlags |= ColwertD3D11Flags( o.BindFlags );
        if (o.MiscFlags & D3D11_RESOURCE_MISC_TILED) Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    }
    explicit CD3D12_RESOURCE_DESC_FROM11( const D3D11_TEXTURE2D_DESC& o )
    {
        *static_cast<CD3D12_RESOURCE_DESC*>(this) = Tex2D( o.Format, o.Width, o.Height, UINT16( o.ArraySize ), UINT16( o.MipLevels ), 
            o.SampleDesc.Count, o.SampleDesc.Quality );
        MiscFlags |= ColwertD3D11Flags( o.BindFlags );
        if (o.MiscFlags & D3D11_RESOURCE_MISC_TILED) Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    }
    explicit CD3D12_RESOURCE_DESC_FROM11( const D3D11_TEXTURE3D_DESC& o )
    {
        *static_cast<CD3D12_RESOURCE_DESC*>(this) = Tex3D( o.Format, o.Width, o.Height, UINT16( o.Depth ), UINT16( o.MipLevels ) );
        MiscFlags |= ColwertD3D11Flags( o.BindFlags );
        if (o.MiscFlags & D3D11_RESOURCE_MISC_TILED) Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    }
};
extern "C"{
#endif


extern RPC_IF_HANDLE __MIDL_itf_d3d11on12_0000_0000_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d11on12_0000_0000_v0_0_s_ifspec;

/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


