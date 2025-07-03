

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

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif /*COM_NO_WINDOWS_H*/

#ifndef __d3d12InternalTest_h__
#define __d3d12InternalTest_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

/* Forward Declarations */ 

#ifndef __ID3D12MonitoredFenceTest_FWD_DEFINED__
#define __ID3D12MonitoredFenceTest_FWD_DEFINED__
typedef interface ID3D12MonitoredFenceTest ID3D12MonitoredFenceTest;

#endif 	/* __ID3D12MonitoredFenceTest_FWD_DEFINED__ */


#ifndef __ID3D12KeyedMutexSyncObjects_FWD_DEFINED__
#define __ID3D12KeyedMutexSyncObjects_FWD_DEFINED__
typedef interface ID3D12KeyedMutexSyncObjects ID3D12KeyedMutexSyncObjects;

#endif 	/* __ID3D12KeyedMutexSyncObjects_FWD_DEFINED__ */


#ifndef __ID3D12DeviceTest_FWD_DEFINED__
#define __ID3D12DeviceTest_FWD_DEFINED__
typedef interface ID3D12DeviceTest ID3D12DeviceTest;

#endif 	/* __ID3D12DeviceTest_FWD_DEFINED__ */


#ifndef __ID3D12CommandQueueTest_FWD_DEFINED__
#define __ID3D12CommandQueueTest_FWD_DEFINED__
typedef interface ID3D12CommandQueueTest ID3D12CommandQueueTest;

#endif 	/* __ID3D12CommandQueueTest_FWD_DEFINED__ */


#ifndef __ID3D12CommandListTest_FWD_DEFINED__
#define __ID3D12CommandListTest_FWD_DEFINED__
typedef interface ID3D12CommandListTest ID3D12CommandListTest;

#endif 	/* __ID3D12CommandListTest_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"
#include "d3d11on12.h"

#ifdef __cplusplus
extern "C"{
#endif 


#ifndef __ID3D12MonitoredFenceTest_INTERFACE_DEFINED__
#define __ID3D12MonitoredFenceTest_INTERFACE_DEFINED__

/* interface ID3D12MonitoredFenceTest */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12MonitoredFenceTest;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("9bef1f2f-dfd1-4a2f-aea8-0279462c758e")
    ID3D12MonitoredFenceTest : public IUnknown
    {
    public:
        virtual UINT64 STDMETHODCALLTYPE GetLwrrentValue( void) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12MonitoredFenceTestVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12MonitoredFenceTest * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12MonitoredFenceTest * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12MonitoredFenceTest * This);
        
        UINT64 ( STDMETHODCALLTYPE *GetLwrrentValue )( 
            ID3D12MonitoredFenceTest * This);
        
        END_INTERFACE
    } ID3D12MonitoredFenceTestVtbl;

    interface ID3D12MonitoredFenceTest
    {
        CONST_VTBL struct ID3D12MonitoredFenceTestVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12MonitoredFenceTest_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12MonitoredFenceTest_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12MonitoredFenceTest_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12MonitoredFenceTest_GetLwrrentValue(This)	\
    ( (This)->lpVtbl -> GetLwrrentValue(This) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12MonitoredFenceTest_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12InternalTest_0000_0001 */
/* [local] */ 

typedef 
enum D3D12_SYNC_OBJECT_FLAG
    {
        D3D12_SYNC_OBJECT_FLAG_SHARED	= ( 1 << 0 ) 
    } 	D3D12_SYNC_OBJECT_FLAG;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_SYNC_OBJECT_FLAG);
typedef struct D3D11_RESOURCE_DESC
    {
    D3D11_RESOURCE_DIMENSION Dimension;
    UINT Width;
    UINT Height;
    UINT Depth;
    UINT MipLevels;
    UINT ArraySize;
    DXGI_FORMAT Format;
    DXGI_SAMPLE_DESC SampleDesc;
    D3D11_USAGE Usage;
    UINT BindFlags;
    UINT CPUAccessFlags;
    UINT MiscFlags;
    UINT StructureByteStride;
    } 	D3D11_RESOURCE_DESC;

#if !defined( D3D12_NO_HELPERS ) && defined( __cplusplus )
}
struct CD3D11_RESOURCE_DESC : public D3D11_RESOURCE_DESC
{
    CD3D11_RESOURCE_DESC()
    {}
    explicit CD3D11_RESOURCE_DESC( const D3D11_RESOURCE_DESC& o ) :
        D3D11_RESOURCE_DESC( o )
    {}
    void ColwertD3D12Info( D3D12_TEXTURE_LAYOUT layout, D3D12_RESOURCE_MISC_FLAG miscFlags )
    {
        if (miscFlags & D3D12_RESOURCE_MISC_ALLOW_RENDER_TARGET) BindFlags |= D3D11_BIND_RENDER_TARGET;
        if (miscFlags & D3D12_RESOURCE_MISC_ALLOW_DEPTH_STENCIL) BindFlags |= D3D11_BIND_DEPTH_STENCIL;
        if (miscFlags & D3D12_RESOURCE_MISC_ALLOW_UNORDERED_ACCESS) BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
        if (!(miscFlags & D3D12_RESOURCE_MISC_DENY_SHADER_RESOURCE)) BindFlags |= D3D11_BIND_SHADER_RESOURCE;
        if (layout == D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE) MiscFlags |= D3D11_RESOURCE_MISC_TILED;
    }
    CD3D11_RESOURCE_DESC( const D3D12_RESOURCE_DESC& o )
    {
        switch (o.Dimension)
        {
        case D3D12_RESOURCE_DIMENSION_BUFFER: Dimension = D3D11_RESOURCE_DIMENSION_BUFFER; break;
        case D3D12_RESOURCE_DIMENSION_TEXTURE_1D: Dimension = D3D11_RESOURCE_DIMENSION_TEXTURE1D; break;
        case D3D12_RESOURCE_DIMENSION_TEXTURE_2D: Dimension = D3D11_RESOURCE_DIMENSION_TEXTURE2D; break;
        case D3D12_RESOURCE_DIMENSION_TEXTURE_3D: Dimension = D3D11_RESOURCE_DIMENSION_TEXTURE3D; break;
        default: Dimension = D3D11_RESOURCE_DIMENSION_UNKNOWN; break;
        }
        Width = UINT( o.Width );
        Height = o.Height;
        Depth = (Dimension == D3D11_RESOURCE_DIMENSION_TEXTURE3D ? o.DepthOrArraySize : 1);
        MipLevels = o.MipLevels;
        ArraySize = (Dimension == D3D11_RESOURCE_DIMENSION_TEXTURE3D ? 1 : o.DepthOrArraySize);
        Format = o.Format;
        SampleDesc = o.SampleDesc;
        Usage = D3D11_USAGE_DEFAULT;
        BindFlags = 0;
        CPUAccessFlags = 0;
        MiscFlags = 0;
        StructureByteStride = 0;
        ColwertD3D12Info( o.Layout, o.MiscFlags );
    }
    CD3D11_RESOURCE_DESC( const D3D11_BUFFER_DESC& o )
    {
        Dimension = D3D11_RESOURCE_DIMENSION_BUFFER;
        Width = o.ByteWidth;
        Height = 1;
        Depth = 1;
        MipLevels = 1;
        ArraySize = 1;
        Format = DXGI_FORMAT_UNKNOWN;
        SampleDesc.Count = 1;
        SampleDesc.Quality = 0;
        Usage = o.Usage;
        BindFlags = o.BindFlags;
        CPUAccessFlags = o.CPUAccessFlags;
        MiscFlags = o.MiscFlags;
        StructureByteStride = o.StructureByteStride;
    }
    CD3D11_RESOURCE_DESC( const D3D11_TEXTURE1D_DESC& o )
    {
        Dimension = D3D11_RESOURCE_DIMENSION_TEXTURE1D;
        Width = o.Width;
        Height = 1;
        Depth = 1;
        MipLevels = o.MipLevels;
        ArraySize = o.ArraySize;
        Format = o.Format;
        SampleDesc.Count = 1;
        SampleDesc.Quality = 0;
        Usage = o.Usage;
        BindFlags = o.BindFlags;
        CPUAccessFlags = o.CPUAccessFlags;
        MiscFlags = o.MiscFlags;
        StructureByteStride = 0;
    }
    CD3D11_RESOURCE_DESC( const D3D11_TEXTURE2D_DESC& o )
    {
        Dimension = D3D11_RESOURCE_DIMENSION_TEXTURE2D;
        Width = o.Width;
        Height = o.Height;
        Depth = 1;
        MipLevels = o.MipLevels;
        ArraySize = o.ArraySize;
        Format = o.Format;
        SampleDesc = o.SampleDesc;
        Usage = o.Usage;
        BindFlags = o.BindFlags;
        CPUAccessFlags = o.CPUAccessFlags;
        MiscFlags = o.MiscFlags;
        StructureByteStride = 0;
    }
    CD3D11_RESOURCE_DESC( const D3D11_TEXTURE3D_DESC& o )
    {
        Dimension = D3D11_RESOURCE_DIMENSION_TEXTURE3D;
        Width = o.Width;
        Height = o.Height;
        Depth = o.Depth;
        MipLevels = o.MipLevels;
        ArraySize = 1;
        Format = o.Format;
        SampleDesc.Count = 1;
        SampleDesc.Quality = 0;
        Usage = o.Usage;
        BindFlags = o.BindFlags;
        CPUAccessFlags = o.CPUAccessFlags;
        MiscFlags = o.MiscFlags;
        StructureByteStride = 0;
    }
    ~CD3D11_RESOURCE_DESC() {}
    operator const D3D11_RESOURCE_DESC&() const { return *this; }
    operator CD3D12_RESOURCE_DESC() const
    {
        CD3D12_RESOURCE_DESC RetDesc
            ( D3D12_RESOURCE_DIMENSION_TEXTURE_1D
            , 0
            , Width
            , Height
            , UINT16( Dimension == D3D11_RESOURCE_DIMENSION_TEXTURE3D ? Depth : ArraySize )
            , UINT16( MipLevels )
            , Format
            , SampleDesc.Count
            , SampleDesc.Quality
            , D3D12_TEXTURE_LAYOUT_UNKNOWN
            , D3D12_RESOURCE_MISC_NONE
            );
        switch (Dimension)
        {
        case D3D11_RESOURCE_DIMENSION_BUFFER:
            RetDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            RetDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            break;
        case D3D11_RESOURCE_DIMENSION_TEXTURE2D: RetDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE_2D; break;
        case D3D11_RESOURCE_DIMENSION_TEXTURE3D: RetDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE_3D; break;
        default: break;
        }
        RetDesc.MiscFlags |= CD3D12_RESOURCE_DESC_FROM11::ColwertD3D11Flags( BindFlags );
        return RetDesc;
    }
    operator D3D12_RESOURCE_DESC() const { return D3D12_RESOURCE_DESC( this->operator CD3D12_RESOURCE_DESC() ); }
};
extern "C"{
#endif
typedef 
enum D3D12_SET_EVENT_CONDITION
    {
        D3D12_SET_EVENT_ALL_COMPLETE	= 0,
        D3D12_SET_EVENT_ANY_COMPLETE	= ( D3D12_SET_EVENT_ALL_COMPLETE + 1 ) 
    } 	D3D12_SET_EVENT_CONDITION;

typedef struct D3D12_MONITORED_FENCE_SPECIFIER
    {
    ID3D12MonitoredFenceTest *pFence;
    UINT64 Value;
    } 	D3D12_MONITORED_FENCE_SPECIFIER;

typedef 
enum D3D12TEST_FEATURE
    {
        D3D12TEST_FEATURE_TEXTURE_LAYOUT	= 0,
        D3D12TEST_FEATURE_ROW_MAJOR	= 1,
        D3D12TEST_FEATURE_MEMORY_ARCHITECTURE	= 2,
        D3D12TEST_FEATURE_RESIDENCY	= 3
    } 	D3D12TEST_FEATURE;

typedef 
enum D3D12TEST_RESOURCE_LAYOUT
    {
        D3D12TEST_RL_UNDEFINED	= 0,
        D3D12TEST_RL_PLACED_PHYSICAL_SUBRESOURCE_PITCHED	= 1,
        D3D12TEST_RL_PLACED_VIRTUAL_SUBRESOURCE_PITCHED	= 2
    } 	D3D12TEST_RESOURCE_LAYOUT;

typedef struct D3D12TEST_FEATURE_DATA_TEXTURE_LAYOUT
    {
    UINT DeviceDependentLayoutCount;
    UINT DeviceDependentSwizzleCount;
    } 	D3D12TEST_FEATURE_DATA_TEXTURE_LAYOUT;

typedef struct D3D12TEST_ROW_MAJOR_ALIGNMENT_CAPS
    {
    UINT16 MaxElementSize;
    UINT16 BaseOffsetAlignment;
    UINT16 PitchAlignment;
    UINT16 DepthPitchAlignment;
    } 	D3D12TEST_ROW_MAJOR_ALIGNMENT_CAPS;

typedef 
enum D3D12TEST_ROW_MAJOR_LAYOUT_FLAG
    {
        D3D12TEST_ROW_MAJOR_LAYOUT_FLAG_NONE	= 0,
        D3D12TEST_ROW_MAJOR_LAYOUT_FLAG_FLEXIBLE_DEPTH_PITCH	= 0x1,
        D3D12TEST_ROW_MAJOR_LAYOUT_FLAG_DEPTH_PITCH_4_8_16_HEIGHT_MULTIPLE	= 0x2
    } 	D3D12TEST_ROW_MAJOR_LAYOUT_FLAG;

DEFINE_ENUM_FLAG_OPERATORS( D3D12TEST_ROW_MAJOR_LAYOUT_FLAG );
typedef struct D3D12TEST_ROW_MAJOR_FUNCTIONAL_UNIT_CAPS
    {
    D3D12TEST_ROW_MAJOR_ALIGNMENT_CAPS AlignmentCaps[ 2 ];
    D3D12TEST_ROW_MAJOR_LAYOUT_FLAG Flags;
    } 	D3D12TEST_ROW_MAJOR_FUNCTIONAL_UNIT_CAPS;

typedef 
enum D3D12TEST_FUNCTIONAL_UNIT
    {
        D3D12TEST_FUNLWNIT_COMBINED	= 0,
        D3D12TEST_FUNLWNIT_COPY_SRC	= ( D3D12TEST_FUNLWNIT_COMBINED + 1 ) ,
        D3D12TEST_FUNLWNIT_COPY_DST	= ( D3D12TEST_FUNLWNIT_COPY_SRC + 1 ) 
    } 	D3D12TEST_FUNCTIONAL_UNIT;

typedef struct D3D12TEST_FEATURE_DATA_ROW_MAJOR
    {
    D3D12TEST_ROW_MAJOR_FUNCTIONAL_UNIT_CAPS FunlwnitCaps[ 3 ];
    } 	D3D12TEST_FEATURE_DATA_ROW_MAJOR;

typedef struct D3D12TEST_SWIZZLE_BIT_ENTRY
    {
    UINT8 Valid	: 1;
    UINT8 ChannelIndex	: 2;
    UINT8 SourceBitIndex	: 5;
    } 	D3D12TEST_SWIZZLE_BIT_ENTRY;

typedef struct D3D12TEST_SWIZZLE_PATTERN_DESC
    {
    D3D12TEST_SWIZZLE_BIT_ENTRY InterleavePatternSourceBits[ 32 ];
    D3D12TEST_SWIZZLE_BIT_ENTRY InterleavePatternXORSourceBits[ 32 ];
    D3D12TEST_SWIZZLE_BIT_ENTRY InterleavePatternXOR2SourceBits[ 32 ];
    } 	D3D12TEST_SWIZZLE_PATTERN_DESC;

typedef struct D3D12TEST_SWIZZLE_PATTERN_ORDINAL
    {
    D3D12_TEXTURE_LAYOUT Layout;
    UINT UnknownOrdinal;
    } 	D3D12TEST_SWIZZLE_PATTERN_ORDINAL;

typedef struct D3D12TEST_TEXTURE_LAYOUT_ORDINAL
    {
    D3D12_TEXTURE_LAYOUT Layout;
    UINT UnknownOrdinal;
    } 	D3D12TEST_TEXTURE_LAYOUT_ORDINAL;

typedef struct D3D12TEST_SWIZZLE_PATTERN_LAYOUT
    {
    D3D12TEST_SWIZZLE_PATTERN_ORDINAL Patterns[ 2 ];
    } 	D3D12TEST_SWIZZLE_PATTERN_LAYOUT;

typedef struct D3D12TEST_SUBRESOURCE_INFO
    {
    UINT64 RowStride;
    UINT64 DepthStride;
    UINT16 RowBytePreSwizzleOffset;
    UINT16 ColumnPreSwizzleOffset;
    UINT16 DepthPreSwizzleOffset;
    } 	D3D12TEST_SUBRESOURCE_INFO;

typedef struct D3D12TEST_FEATURE_DATA_RESIDENCY
    {
    BOOL SupportsMakeResident;
    BOOL SupportsOfferReclaim;
    } 	D3D12TEST_FEATURE_DATA_RESIDENCY;

typedef 
enum D3D12TEST_HEAP_MISC_FLAG
    {
        D3D12TEST_HEAP_MISC_NONE	= 0,
        D3D12TEST_HEAP_MISC_SHARED_HANDLE_NT	= 0x2,
        D3D12TEST_HEAP_MISC_SHARED_HANDLE_GDI	= 0x4,
        D3D12TEST_HEAP_MISC_SHARED_HANDLE_MASK	= 0x6,
        D3D12TEST_HEAP_MISC_SHARED_KEYED_MUTEX	= 0x8,
        D3D12TEST_HEAP_MISC_SHARED_CROSS_ADAPTER	= 0x10,
        D3D12TEST_HEAP_MISC_SHARED_RESTRICTED_TO_DWM	= 0x20,
        D3D12TEST_HEAP_MISC_FLIP_MODEL_BACK_BUFFER	= 0x40,
        D3D12TEST_HEAP_MISC_PRESENTABLE	= 0x80,
        D3D12TEST_HEAP_MISC_INFER_DIMENSION_ATTRIBUTION	= 0x100,
        D3D12TEST_HEAP_MISC_PRIMARY	= 0x200,
        D3D12TEST_HEAP_MISC_CREATED_NOT_FROM_D3D12	= 0x400,
        D3D12TEST_HEAP_MISC_MSAA	= 0x800
    } 	D3D12TEST_HEAP_MISC_FLAG;

DEFINE_ENUM_FLAG_OPERATORS( D3D12TEST_HEAP_MISC_FLAG );
typedef struct D3D12TEST_FEATURE_DATA_MEMORY_ARCHITECTURE
    {
    UINT DriverSupportedVer;
    BOOL UMA;
    BOOL CacheCoherentUMA;
    BOOL IOCoherent;
    D3D12TEST_HEAP_MISC_FLAG DisabledHeapFlags;
    } 	D3D12TEST_FEATURE_DATA_MEMORY_ARCHITECTURE;

typedef 
enum D3D12TEST_RESOURCE_MISC_FLAG
    {
        D3D12TEST_RESOURCE_MISC_NONE	= 0,
        D3D12TEST_RESOURCE_MISC_VIDEO	= 0x1
    } 	D3D12TEST_RESOURCE_MISC_FLAG;

DEFINE_ENUM_FLAG_OPERATORS( D3D12TEST_RESOURCE_MISC_FLAG );


extern RPC_IF_HANDLE __MIDL_itf_d3d12InternalTest_0000_0001_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12InternalTest_0000_0001_v0_0_s_ifspec;

#ifndef __ID3D12KeyedMutexSyncObjects_INTERFACE_DEFINED__
#define __ID3D12KeyedMutexSyncObjects_INTERFACE_DEFINED__

/* interface ID3D12KeyedMutexSyncObjects */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12KeyedMutexSyncObjects;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("11d09140-26d5-4aa4-8a5d-bdf262c5f192")
    ID3D12KeyedMutexSyncObjects : public ID3D12DeviceChild
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12KeyedMutexSyncObjectsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12KeyedMutexSyncObjects * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12KeyedMutexSyncObjects * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12KeyedMutexSyncObjects * This);
        
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12KeyedMutexSyncObjects * This,
            /* [annotation] */ 
            _In_  REFGUID guid,
            /* [annotation] */ 
            _Inout_  UINT *pDataSize,
            /* [annotation] */ 
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12KeyedMutexSyncObjects * This,
            /* [annotation] */ 
            _In_  REFGUID guid,
            /* [annotation] */ 
            _In_  UINT DataSize,
            /* [annotation] */ 
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12KeyedMutexSyncObjects * This,
            /* [annotation] */ 
            _In_  REFGUID guid,
            /* [annotation] */ 
            _In_opt_  const IUnknown *pData);
        
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12KeyedMutexSyncObjects * This,
            /* [annotation] */ 
            _In_z_  LPCWSTR Name);
        
        void ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12KeyedMutexSyncObjects * This,
            /* [annotation] */ 
            _Out_  ID3D12Device **ppDevice);
        
        END_INTERFACE
    } ID3D12KeyedMutexSyncObjectsVtbl;

    interface ID3D12KeyedMutexSyncObjects
    {
        CONST_VTBL struct ID3D12KeyedMutexSyncObjectsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12KeyedMutexSyncObjects_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12KeyedMutexSyncObjects_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12KeyedMutexSyncObjects_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12KeyedMutexSyncObjects_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12KeyedMutexSyncObjects_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12KeyedMutexSyncObjects_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12KeyedMutexSyncObjects_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12KeyedMutexSyncObjects_GetDevice(This,ppDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,ppDevice) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12KeyedMutexSyncObjects_INTERFACE_DEFINED__ */


#ifndef __ID3D12DeviceTest_INTERFACE_DEFINED__
#define __ID3D12DeviceTest_INTERFACE_DEFINED__

/* interface ID3D12DeviceTest */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12DeviceTest;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("5bb2f236-b46a-46bb-99e9-f334724dbf12")
    ID3D12DeviceTest : public IUnknown
    {
    public:
        virtual void *STDMETHODCALLTYPE GetCallbacks( void) = 0;
        
        virtual void *STDMETHODCALLTYPE GetUmdThunkCallbacks( 
            /* [annotation] */ 
            _Outptr_  const void **ppTable) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateMonitoredFence( 
            D3D12_SYNC_OBJECT_FLAG Flags,
            UINT64 InitialValue,
            REFIID iid,
            /* [annotation] */ 
            _Out_  void **ppMonitoredFence) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE SetEventOnFenceValues( 
            SIZE_T FenceCount,
            /* [annotation] */ 
            _In_reads_(FenceCount)  const D3D12_MONITORED_FENCE_SPECIFIER *pFences,
            HANDLE hEvent,
            D3D12_SET_EVENT_CONDITION Condition) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE SignalImmediateValue( 
            SIZE_T FenceCount,
            /* [annotation] */ 
            _In_reads_(FenceCount)  const D3D12_MONITORED_FENCE_SPECIFIER *pFences) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CheckTestFeatureSupport( 
            D3D12TEST_FEATURE Feature,
            /* [annotation] */ 
            _Out_writes_bytes_( FeatureDataSize )  void *pFeatureData,
            UINT FeatureDataSize) = 0;
        
        virtual void *STDMETHODCALLTYPE GetAdapterFunctionTable( 
            /* [annotation] */ 
            _Outptr_  void **ppTable) = 0;
        
        virtual void *STDMETHODCALLTYPE GetFunctionTableAndInfo( 
            UINT TableType,
            UINT TableNum,
            BOOL bDeviceRemoved,
            /* [annotation] */ 
            _Out_opt_  SIZE_T *pOffset,
            /* [annotation] */ 
            _Out_opt_  SIZE_T *pSize,
            /* [annotation] */ 
            _Outptr_opt_  void **ppTable) = 0;
        
        virtual void STDMETHODCALLTYPE GetGlobalTableAndSize( 
            /* [annotation] */ 
            _Outptr_result_bytebuffer_(*pSize)  void **ppTable,
            /* [annotation] */ 
            _Out_  SIZE_T *pSize) = 0;
        
        virtual void STDMETHODCALLTYPE GetPrevGlobalTables( 
            /* [annotation] */ 
            _Inout_  UINT32 *pNum,
            /* [annotation] */ 
            _Out_writes_opt_(*pNum)  void **prevTables) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateTestResourceAndHeap( 
            /* [annotation] */ 
            _In_opt_  const D3D12_HEAP_DESC *pHeapDesc,
            D3D12TEST_HEAP_MISC_FLAG HeapTestMiscFlag,
            /* [annotation] */ 
            _In_opt_  ID3D12Heap *pHeap,
            UINT64 HeapOffset,
            /* [annotation] */ 
            _In_opt_  const D3D11_SUBRESOURCE_DATA *pInitialData,
            /* [annotation] */ 
            _Inout_opt_  void *pPrimaryDesc,
            /* [annotation] */ 
            _In_opt_  const D3D12_RESOURCE_DESC *pResourceDesc,
            /* [annotation] */ 
            _In_reads_opt_(2)  const D3D11_RESOURCE_DESC *pResource11Desc,
            D3D12TEST_RESOURCE_MISC_FLAG ResourceTestMiscFlag,
            D3D12_RESOURCE_USAGE InitialState,
            /* [annotation] */ 
            _In_opt_  const D3D12_CLEAR_VALUE *pOptimizedClearValue,
            /* [annotation] */ 
            _In_opt_  IUnknown *pOuterUnk,
            /* [annotation] */ 
            _In_opt_  ID3D12KeyedMutexSyncObjects *pSyncObjects,
            /* [in] */ REFIID riidResource,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvResource,
            /* [in] */ REFIID riidHeap,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvHeap) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE OpenTestResourceAndHeap( 
            HANDLE hBundleNTHandle,
            HANDLE hBundleGDIHandle,
            D3D12TEST_HEAP_MISC_FLAG HeapTestMiscFlag,
            D3D12TEST_RESOURCE_MISC_FLAG ResourceTestMiscFlag,
            D3D12_RESOURCE_USAGE InitialState,
            /* [in] */ REFIID riidResource,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvResource,
            /* [in] */ REFIID riidHeap,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvHeap,
            /* [annotation][out] */ 
            _COM_Outptr_opt_result_maybenull_  ID3D12KeyedMutexSyncObjects **ppSyncObjects) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateSharedHandleInternal( 
            /* [annotation] */ 
            _In_  ID3D12DeviceChild *pObject,
            /* [annotation] */ 
            _In_opt_  const SELWRITY_ATTRIBUTES *pAttributes,
            DWORD Access,
            /* [annotation] */ 
            _In_opt_  LPCWSTR pName,
            /* [annotation] */ 
            _In_opt_  ID3D12KeyedMutexSyncObjects *pSyncObjects,
            /* [annotation] */ 
            _Out_  HANDLE *pHandle) = 0;
        
        virtual D3D12_HEAP_DESC STDMETHODCALLTYPE GetResolvedHeapDesc( 
            /* [annotation] */ 
            _In_  ID3D12Heap *pHeap) = 0;
        
        virtual ID3D12Heap *STDMETHODCALLTYPE GetImplicitHeap( 
            /* [annotation] */ 
            _In_  ID3D12Resource *pResource) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetResourceDesc11( 
            /* [annotation] */ 
            _In_  ID3D12Resource *__MIDL__ID3D12DeviceTest0000,
            /* [annotation] */ 
            _Out_  D3D11_RESOURCE_DESC *__MIDL__ID3D12DeviceTest0001) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE SetDisplayMode( 
            /* [annotation] */ 
            _In_  ID3D12Resource *pResource,
            UINT PrimaryFlags) = 0;
        
        virtual D3D12TEST_TEXTURE_LAYOUT_ORDINAL STDMETHODCALLTYPE GetTestTextureLayout( 
            /* [annotation] */ 
            _In_  ID3D12Resource *pResource) = 0;
        
        virtual UINT STDMETHODCALLTYPE GetMipLevelSwizzleTransition( 
            /* [annotation] */ 
            _In_  ID3D12Resource *pResource) = 0;
        
        virtual D3D12TEST_SWIZZLE_PATTERN_LAYOUT STDMETHODCALLTYPE GetSwizzlePatternLayout( 
            UINT Ordinal) = 0;
        
        virtual D3D12TEST_SWIZZLE_PATTERN_DESC STDMETHODCALLTYPE GetSwizzlePattern( 
            UINT Ordinal) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE TestMap( 
            /* [annotation] */ 
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            /* [annotation] */ 
            _In_opt_  const D3D12_RANGE *pReadRange,
            /* [annotation] */ 
            _Outptr_  void **ppData,
            /* [annotation] */ 
            _Out_  D3D12TEST_SUBRESOURCE_INFO *pSubresourceInfo) = 0;
        
        virtual void STDMETHODCALLTYPE TestUnmap( 
            /* [annotation] */ 
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            /* [annotation] */ 
            _In_opt_  const D3D12_RANGE *pWrittenRange) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12DeviceTestVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12DeviceTest * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12DeviceTest * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12DeviceTest * This);
        
        void *( STDMETHODCALLTYPE *GetCallbacks )( 
            ID3D12DeviceTest * This);
        
        void *( STDMETHODCALLTYPE *GetUmdThunkCallbacks )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _Outptr_  const void **ppTable);
        
        HRESULT ( STDMETHODCALLTYPE *CreateMonitoredFence )( 
            ID3D12DeviceTest * This,
            D3D12_SYNC_OBJECT_FLAG Flags,
            UINT64 InitialValue,
            REFIID iid,
            /* [annotation] */ 
            _Out_  void **ppMonitoredFence);
        
        HRESULT ( STDMETHODCALLTYPE *SetEventOnFenceValues )( 
            ID3D12DeviceTest * This,
            SIZE_T FenceCount,
            /* [annotation] */ 
            _In_reads_(FenceCount)  const D3D12_MONITORED_FENCE_SPECIFIER *pFences,
            HANDLE hEvent,
            D3D12_SET_EVENT_CONDITION Condition);
        
        HRESULT ( STDMETHODCALLTYPE *SignalImmediateValue )( 
            ID3D12DeviceTest * This,
            SIZE_T FenceCount,
            /* [annotation] */ 
            _In_reads_(FenceCount)  const D3D12_MONITORED_FENCE_SPECIFIER *pFences);
        
        HRESULT ( STDMETHODCALLTYPE *CheckTestFeatureSupport )( 
            ID3D12DeviceTest * This,
            D3D12TEST_FEATURE Feature,
            /* [annotation] */ 
            _Out_writes_bytes_( FeatureDataSize )  void *pFeatureData,
            UINT FeatureDataSize);
        
        void *( STDMETHODCALLTYPE *GetAdapterFunctionTable )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _Outptr_  void **ppTable);
        
        void *( STDMETHODCALLTYPE *GetFunctionTableAndInfo )( 
            ID3D12DeviceTest * This,
            UINT TableType,
            UINT TableNum,
            BOOL bDeviceRemoved,
            /* [annotation] */ 
            _Out_opt_  SIZE_T *pOffset,
            /* [annotation] */ 
            _Out_opt_  SIZE_T *pSize,
            /* [annotation] */ 
            _Outptr_opt_  void **ppTable);
        
        void ( STDMETHODCALLTYPE *GetGlobalTableAndSize )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _Outptr_result_bytebuffer_(*pSize)  void **ppTable,
            /* [annotation] */ 
            _Out_  SIZE_T *pSize);
        
        void ( STDMETHODCALLTYPE *GetPrevGlobalTables )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _Inout_  UINT32 *pNum,
            /* [annotation] */ 
            _Out_writes_opt_(*pNum)  void **prevTables);
        
        HRESULT ( STDMETHODCALLTYPE *CreateTestResourceAndHeap )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _In_opt_  const D3D12_HEAP_DESC *pHeapDesc,
            D3D12TEST_HEAP_MISC_FLAG HeapTestMiscFlag,
            /* [annotation] */ 
            _In_opt_  ID3D12Heap *pHeap,
            UINT64 HeapOffset,
            /* [annotation] */ 
            _In_opt_  const D3D11_SUBRESOURCE_DATA *pInitialData,
            /* [annotation] */ 
            _Inout_opt_  void *pPrimaryDesc,
            /* [annotation] */ 
            _In_opt_  const D3D12_RESOURCE_DESC *pResourceDesc,
            /* [annotation] */ 
            _In_reads_opt_(2)  const D3D11_RESOURCE_DESC *pResource11Desc,
            D3D12TEST_RESOURCE_MISC_FLAG ResourceTestMiscFlag,
            D3D12_RESOURCE_USAGE InitialState,
            /* [annotation] */ 
            _In_opt_  const D3D12_CLEAR_VALUE *pOptimizedClearValue,
            /* [annotation] */ 
            _In_opt_  IUnknown *pOuterUnk,
            /* [annotation] */ 
            _In_opt_  ID3D12KeyedMutexSyncObjects *pSyncObjects,
            /* [in] */ REFIID riidResource,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvResource,
            /* [in] */ REFIID riidHeap,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvHeap);
        
        HRESULT ( STDMETHODCALLTYPE *OpenTestResourceAndHeap )( 
            ID3D12DeviceTest * This,
            HANDLE hBundleNTHandle,
            HANDLE hBundleGDIHandle,
            D3D12TEST_HEAP_MISC_FLAG HeapTestMiscFlag,
            D3D12TEST_RESOURCE_MISC_FLAG ResourceTestMiscFlag,
            D3D12_RESOURCE_USAGE InitialState,
            /* [in] */ REFIID riidResource,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvResource,
            /* [in] */ REFIID riidHeap,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvHeap,
            /* [annotation][out] */ 
            _COM_Outptr_opt_result_maybenull_  ID3D12KeyedMutexSyncObjects **ppSyncObjects);
        
        HRESULT ( STDMETHODCALLTYPE *CreateSharedHandleInternal )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _In_  ID3D12DeviceChild *pObject,
            /* [annotation] */ 
            _In_opt_  const SELWRITY_ATTRIBUTES *pAttributes,
            DWORD Access,
            /* [annotation] */ 
            _In_opt_  LPCWSTR pName,
            /* [annotation] */ 
            _In_opt_  ID3D12KeyedMutexSyncObjects *pSyncObjects,
            /* [annotation] */ 
            _Out_  HANDLE *pHandle);
        
        D3D12_HEAP_DESC ( STDMETHODCALLTYPE *GetResolvedHeapDesc )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _In_  ID3D12Heap *pHeap);
        
        ID3D12Heap *( STDMETHODCALLTYPE *GetImplicitHeap )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _In_  ID3D12Resource *pResource);
        
        HRESULT ( STDMETHODCALLTYPE *GetResourceDesc11 )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _In_  ID3D12Resource *__MIDL__ID3D12DeviceTest0000,
            /* [annotation] */ 
            _Out_  D3D11_RESOURCE_DESC *__MIDL__ID3D12DeviceTest0001);
        
        HRESULT ( STDMETHODCALLTYPE *SetDisplayMode )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _In_  ID3D12Resource *pResource,
            UINT PrimaryFlags);
        
        D3D12TEST_TEXTURE_LAYOUT_ORDINAL ( STDMETHODCALLTYPE *GetTestTextureLayout )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _In_  ID3D12Resource *pResource);
        
        UINT ( STDMETHODCALLTYPE *GetMipLevelSwizzleTransition )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _In_  ID3D12Resource *pResource);
        
        D3D12TEST_SWIZZLE_PATTERN_LAYOUT ( STDMETHODCALLTYPE *GetSwizzlePatternLayout )( 
            ID3D12DeviceTest * This,
            UINT Ordinal);
        
        D3D12TEST_SWIZZLE_PATTERN_DESC ( STDMETHODCALLTYPE *GetSwizzlePattern )( 
            ID3D12DeviceTest * This,
            UINT Ordinal);
        
        HRESULT ( STDMETHODCALLTYPE *TestMap )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            /* [annotation] */ 
            _In_opt_  const D3D12_RANGE *pReadRange,
            /* [annotation] */ 
            _Outptr_  void **ppData,
            /* [annotation] */ 
            _Out_  D3D12TEST_SUBRESOURCE_INFO *pSubresourceInfo);
        
        void ( STDMETHODCALLTYPE *TestUnmap )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            /* [annotation] */ 
            _In_opt_  const D3D12_RANGE *pWrittenRange);
        
        END_INTERFACE
    } ID3D12DeviceTestVtbl;

    interface ID3D12DeviceTest
    {
        CONST_VTBL struct ID3D12DeviceTestVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12DeviceTest_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12DeviceTest_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12DeviceTest_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12DeviceTest_GetCallbacks(This)	\
    ( (This)->lpVtbl -> GetCallbacks(This) ) 

#define ID3D12DeviceTest_GetUmdThunkCallbacks(This,ppTable)	\
    ( (This)->lpVtbl -> GetUmdThunkCallbacks(This,ppTable) ) 

#define ID3D12DeviceTest_CreateMonitoredFence(This,Flags,InitialValue,iid,ppMonitoredFence)	\
    ( (This)->lpVtbl -> CreateMonitoredFence(This,Flags,InitialValue,iid,ppMonitoredFence) ) 

#define ID3D12DeviceTest_SetEventOnFenceValues(This,FenceCount,pFences,hEvent,Condition)	\
    ( (This)->lpVtbl -> SetEventOnFenceValues(This,FenceCount,pFences,hEvent,Condition) ) 

#define ID3D12DeviceTest_SignalImmediateValue(This,FenceCount,pFences)	\
    ( (This)->lpVtbl -> SignalImmediateValue(This,FenceCount,pFences) ) 

#define ID3D12DeviceTest_CheckTestFeatureSupport(This,Feature,pFeatureData,FeatureDataSize)	\
    ( (This)->lpVtbl -> CheckTestFeatureSupport(This,Feature,pFeatureData,FeatureDataSize) ) 

#define ID3D12DeviceTest_GetAdapterFunctionTable(This,ppTable)	\
    ( (This)->lpVtbl -> GetAdapterFunctionTable(This,ppTable) ) 

#define ID3D12DeviceTest_GetFunctionTableAndInfo(This,TableType,TableNum,bDeviceRemoved,pOffset,pSize,ppTable)	\
    ( (This)->lpVtbl -> GetFunctionTableAndInfo(This,TableType,TableNum,bDeviceRemoved,pOffset,pSize,ppTable) ) 

#define ID3D12DeviceTest_GetGlobalTableAndSize(This,ppTable,pSize)	\
    ( (This)->lpVtbl -> GetGlobalTableAndSize(This,ppTable,pSize) ) 

#define ID3D12DeviceTest_GetPrevGlobalTables(This,pNum,prevTables)	\
    ( (This)->lpVtbl -> GetPrevGlobalTables(This,pNum,prevTables) ) 

#define ID3D12DeviceTest_CreateTestResourceAndHeap(This,pHeapDesc,HeapTestMiscFlag,pHeap,HeapOffset,pInitialData,pPrimaryDesc,pResourceDesc,pResource11Desc,ResourceTestMiscFlag,InitialState,pOptimizedClearValue,pOuterUnk,pSyncObjects,riidResource,ppvResource,riidHeap,ppvHeap)	\
    ( (This)->lpVtbl -> CreateTestResourceAndHeap(This,pHeapDesc,HeapTestMiscFlag,pHeap,HeapOffset,pInitialData,pPrimaryDesc,pResourceDesc,pResource11Desc,ResourceTestMiscFlag,InitialState,pOptimizedClearValue,pOuterUnk,pSyncObjects,riidResource,ppvResource,riidHeap,ppvHeap) ) 

#define ID3D12DeviceTest_OpenTestResourceAndHeap(This,hBundleNTHandle,hBundleGDIHandle,HeapTestMiscFlag,ResourceTestMiscFlag,InitialState,riidResource,ppvResource,riidHeap,ppvHeap,ppSyncObjects)	\
    ( (This)->lpVtbl -> OpenTestResourceAndHeap(This,hBundleNTHandle,hBundleGDIHandle,HeapTestMiscFlag,ResourceTestMiscFlag,InitialState,riidResource,ppvResource,riidHeap,ppvHeap,ppSyncObjects) ) 

#define ID3D12DeviceTest_CreateSharedHandleInternal(This,pObject,pAttributes,Access,pName,pSyncObjects,pHandle)	\
    ( (This)->lpVtbl -> CreateSharedHandleInternal(This,pObject,pAttributes,Access,pName,pSyncObjects,pHandle) ) 

#define ID3D12DeviceTest_GetResolvedHeapDesc(This,pHeap)	\
    ( (This)->lpVtbl -> GetResolvedHeapDesc(This,pHeap) ) 

#define ID3D12DeviceTest_GetImplicitHeap(This,pResource)	\
    ( (This)->lpVtbl -> GetImplicitHeap(This,pResource) ) 

#define ID3D12DeviceTest_GetResourceDesc11(This,__MIDL__ID3D12DeviceTest0000,__MIDL__ID3D12DeviceTest0001)	\
    ( (This)->lpVtbl -> GetResourceDesc11(This,__MIDL__ID3D12DeviceTest0000,__MIDL__ID3D12DeviceTest0001) ) 

#define ID3D12DeviceTest_SetDisplayMode(This,pResource,PrimaryFlags)	\
    ( (This)->lpVtbl -> SetDisplayMode(This,pResource,PrimaryFlags) ) 

#define ID3D12DeviceTest_GetTestTextureLayout(This,pResource)	\
    ( (This)->lpVtbl -> GetTestTextureLayout(This,pResource) ) 

#define ID3D12DeviceTest_GetMipLevelSwizzleTransition(This,pResource)	\
    ( (This)->lpVtbl -> GetMipLevelSwizzleTransition(This,pResource) ) 

#define ID3D12DeviceTest_GetSwizzlePatternLayout(This,Ordinal)	\
    ( (This)->lpVtbl -> GetSwizzlePatternLayout(This,Ordinal) ) 

#define ID3D12DeviceTest_GetSwizzlePattern(This,Ordinal)	\
    ( (This)->lpVtbl -> GetSwizzlePattern(This,Ordinal) ) 

#define ID3D12DeviceTest_TestMap(This,pResource,Subresource,pReadRange,ppData,pSubresourceInfo)	\
    ( (This)->lpVtbl -> TestMap(This,pResource,Subresource,pReadRange,ppData,pSubresourceInfo) ) 

#define ID3D12DeviceTest_TestUnmap(This,pResource,Subresource,pWrittenRange)	\
    ( (This)->lpVtbl -> TestUnmap(This,pResource,Subresource,pWrittenRange) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */



D3D12_HEAP_DESC STDMETHODCALLTYPE ID3D12DeviceTest_GetResolvedHeapDesc_Proxy( 
    ID3D12DeviceTest * This,
    /* [annotation] */ 
    _In_  ID3D12Heap *pHeap);


void __RPC_STUB ID3D12DeviceTest_GetResolvedHeapDesc_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


D3D12TEST_TEXTURE_LAYOUT_ORDINAL STDMETHODCALLTYPE ID3D12DeviceTest_GetTestTextureLayout_Proxy( 
    ID3D12DeviceTest * This,
    /* [annotation] */ 
    _In_  ID3D12Resource *pResource);


void __RPC_STUB ID3D12DeviceTest_GetTestTextureLayout_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


D3D12TEST_SWIZZLE_PATTERN_LAYOUT STDMETHODCALLTYPE ID3D12DeviceTest_GetSwizzlePatternLayout_Proxy( 
    ID3D12DeviceTest * This,
    UINT Ordinal);


void __RPC_STUB ID3D12DeviceTest_GetSwizzlePatternLayout_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


D3D12TEST_SWIZZLE_PATTERN_DESC STDMETHODCALLTYPE ID3D12DeviceTest_GetSwizzlePattern_Proxy( 
    ID3D12DeviceTest * This,
    UINT Ordinal);


void __RPC_STUB ID3D12DeviceTest_GetSwizzlePattern_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);



#endif 	/* __ID3D12DeviceTest_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12InternalTest_0000_0003 */
/* [local] */ 

typedef 
enum D3D12_COMMAND_QUEUE_CAP
    {
        D3D12_COMMAND_LIST_SIGNAL_SUPPORTED	= ( 1 << 0 ) 
    } 	D3D12_COMMAND_QUEUE_CAP;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_COMMAND_QUEUE_CAP);


extern RPC_IF_HANDLE __MIDL_itf_d3d12InternalTest_0000_0003_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12InternalTest_0000_0003_v0_0_s_ifspec;

#ifndef __ID3D12CommandQueueTest_INTERFACE_DEFINED__
#define __ID3D12CommandQueueTest_INTERFACE_DEFINED__

/* interface ID3D12CommandQueueTest */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12CommandQueueTest;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("228808a2-ede7-487e-85a0-0ec4e94ef96c")
    ID3D12CommandQueueTest : public IUnknown
    {
    public:
        virtual BOOL STDMETHODCALLTYPE IsLockAcquired( void) = 0;
        
        virtual D3D12_COMMAND_QUEUE_CAP STDMETHODCALLTYPE GetCaps( void) = 0;
        
        virtual void STDMETHODCALLTYPE SignalFence( 
            SIZE_T FenceCount,
            /* [annotation] */ 
            _In_reads_(FenceCount)  const D3D12_MONITORED_FENCE_SPECIFIER *pFences) = 0;
        
        virtual void STDMETHODCALLTYPE WaitFence( 
            SIZE_T FenceCount,
            /* [annotation] */ 
            _In_reads_(FenceCount)  const D3D12_MONITORED_FENCE_SPECIFIER *pFences) = 0;
        
        virtual void *STDMETHODCALLTYPE GetCriticalSection( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetQueueFence( 
            /* [in] */ REFIID riidFence,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvFence) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12CommandQueueTestVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12CommandQueueTest * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12CommandQueueTest * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12CommandQueueTest * This);
        
        BOOL ( STDMETHODCALLTYPE *IsLockAcquired )( 
            ID3D12CommandQueueTest * This);
        
        D3D12_COMMAND_QUEUE_CAP ( STDMETHODCALLTYPE *GetCaps )( 
            ID3D12CommandQueueTest * This);
        
        void ( STDMETHODCALLTYPE *SignalFence )( 
            ID3D12CommandQueueTest * This,
            SIZE_T FenceCount,
            /* [annotation] */ 
            _In_reads_(FenceCount)  const D3D12_MONITORED_FENCE_SPECIFIER *pFences);
        
        void ( STDMETHODCALLTYPE *WaitFence )( 
            ID3D12CommandQueueTest * This,
            SIZE_T FenceCount,
            /* [annotation] */ 
            _In_reads_(FenceCount)  const D3D12_MONITORED_FENCE_SPECIFIER *pFences);
        
        void *( STDMETHODCALLTYPE *GetCriticalSection )( 
            ID3D12CommandQueueTest * This);
        
        HRESULT ( STDMETHODCALLTYPE *GetQueueFence )( 
            ID3D12CommandQueueTest * This,
            /* [in] */ REFIID riidFence,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvFence);
        
        END_INTERFACE
    } ID3D12CommandQueueTestVtbl;

    interface ID3D12CommandQueueTest
    {
        CONST_VTBL struct ID3D12CommandQueueTestVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12CommandQueueTest_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12CommandQueueTest_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12CommandQueueTest_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12CommandQueueTest_IsLockAcquired(This)	\
    ( (This)->lpVtbl -> IsLockAcquired(This) ) 

#define ID3D12CommandQueueTest_GetCaps(This)	\
    ( (This)->lpVtbl -> GetCaps(This) ) 

#define ID3D12CommandQueueTest_SignalFence(This,FenceCount,pFences)	\
    ( (This)->lpVtbl -> SignalFence(This,FenceCount,pFences) ) 

#define ID3D12CommandQueueTest_WaitFence(This,FenceCount,pFences)	\
    ( (This)->lpVtbl -> WaitFence(This,FenceCount,pFences) ) 

#define ID3D12CommandQueueTest_GetCriticalSection(This)	\
    ( (This)->lpVtbl -> GetCriticalSection(This) ) 

#define ID3D12CommandQueueTest_GetQueueFence(This,riidFence,ppvFence)	\
    ( (This)->lpVtbl -> GetQueueFence(This,riidFence,ppvFence) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12CommandQueueTest_INTERFACE_DEFINED__ */


#ifndef __ID3D12CommandListTest_INTERFACE_DEFINED__
#define __ID3D12CommandListTest_INTERFACE_DEFINED__

/* interface ID3D12CommandListTest */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12CommandListTest;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("4dc1292b-e547-4ad4-ba8e-9c800549c5ef")
    ID3D12CommandListTest : public IUnknown
    {
    public:
        virtual void STDMETHODCALLTYPE GetFunctionTableOffsets( 
            /* [annotation] */ 
            _Out_  SIZE_T *pPrimaryOffset,
            /* [annotation] */ 
            _Out_  SIZE_T *pSecondaryOffset,
            /* [annotation] */ 
            _Out_  SIZE_T *pDROffset) = 0;
        
        virtual void *STDMETHODCALLTYPE GetCoreObject( void) = 0;
        
        virtual void STDMETHODCALLTYPE TestResourceBarrier( 
            UINT Count,
            /* [annotation] */ 
            _In_reads_(Count)  const D3D12_RESOURCE_BARRIER_DESC *pDesc) = 0;
        
        virtual UINT64 STDMETHODCALLTYPE GetAPISequenceNumber( void) = 0;
        
        virtual void STDMETHODCALLTYPE TestCopyTextureRegion( 
            /* [annotation] */ 
            _In_  const D3D12_TEXTURE_COPY_LOCATION *pDst,
            UINT DstDepthPitch,
            UINT DstX,
            UINT DstY,
            UINT DstZ,
            /* [annotation] */ 
            _In_  const D3D12_TEXTURE_COPY_LOCATION *pSrc,
            UINT SrcDepthPitch,
            /* [annotation] */ 
            _In_opt_  const D3D12_BOX *pSrcBox,
            D3D12_COPY_FLAGS CopyFlags) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12CommandListTestVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12CommandListTest * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12CommandListTest * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12CommandListTest * This);
        
        void ( STDMETHODCALLTYPE *GetFunctionTableOffsets )( 
            ID3D12CommandListTest * This,
            /* [annotation] */ 
            _Out_  SIZE_T *pPrimaryOffset,
            /* [annotation] */ 
            _Out_  SIZE_T *pSecondaryOffset,
            /* [annotation] */ 
            _Out_  SIZE_T *pDROffset);
        
        void *( STDMETHODCALLTYPE *GetCoreObject )( 
            ID3D12CommandListTest * This);
        
        void ( STDMETHODCALLTYPE *TestResourceBarrier )( 
            ID3D12CommandListTest * This,
            UINT Count,
            /* [annotation] */ 
            _In_reads_(Count)  const D3D12_RESOURCE_BARRIER_DESC *pDesc);
        
        UINT64 ( STDMETHODCALLTYPE *GetAPISequenceNumber )( 
            ID3D12CommandListTest * This);
        
        void ( STDMETHODCALLTYPE *TestCopyTextureRegion )( 
            ID3D12CommandListTest * This,
            /* [annotation] */ 
            _In_  const D3D12_TEXTURE_COPY_LOCATION *pDst,
            UINT DstDepthPitch,
            UINT DstX,
            UINT DstY,
            UINT DstZ,
            /* [annotation] */ 
            _In_  const D3D12_TEXTURE_COPY_LOCATION *pSrc,
            UINT SrcDepthPitch,
            /* [annotation] */ 
            _In_opt_  const D3D12_BOX *pSrcBox,
            D3D12_COPY_FLAGS CopyFlags);
        
        END_INTERFACE
    } ID3D12CommandListTestVtbl;

    interface ID3D12CommandListTest
    {
        CONST_VTBL struct ID3D12CommandListTestVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12CommandListTest_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12CommandListTest_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12CommandListTest_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12CommandListTest_GetFunctionTableOffsets(This,pPrimaryOffset,pSecondaryOffset,pDROffset)	\
    ( (This)->lpVtbl -> GetFunctionTableOffsets(This,pPrimaryOffset,pSecondaryOffset,pDROffset) ) 

#define ID3D12CommandListTest_GetCoreObject(This)	\
    ( (This)->lpVtbl -> GetCoreObject(This) ) 

#define ID3D12CommandListTest_TestResourceBarrier(This,Count,pDesc)	\
    ( (This)->lpVtbl -> TestResourceBarrier(This,Count,pDesc) ) 

#define ID3D12CommandListTest_GetAPISequenceNumber(This)	\
    ( (This)->lpVtbl -> GetAPISequenceNumber(This) ) 

#define ID3D12CommandListTest_TestCopyTextureRegion(This,pDst,DstDepthPitch,DstX,DstY,DstZ,pSrc,SrcDepthPitch,pSrcBox,CopyFlags)	\
    ( (This)->lpVtbl -> TestCopyTextureRegion(This,pDst,DstDepthPitch,DstX,DstY,DstZ,pSrc,SrcDepthPitch,pSrcBox,CopyFlags) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12CommandListTest_INTERFACE_DEFINED__ */


/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


