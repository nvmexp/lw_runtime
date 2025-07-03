

/* this ALWAYS GENERATED file contains the definitions for the interfaces */


 /* File created by MIDL compiler version 8.00.0611 */
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
#endif /* __RPCNDR_H_VERSION__ */

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


/* interface __MIDL_itf_d3d12InternalTest_0000_0000 */
/* [local] */ 

typedef 
enum D3D12TEST_FEATURE
    {
        D3D12TEST_FEATURE_TEXTURE_LAYOUT	= 0,
        D3D12TEST_FEATURE_ROW_MAJOR	= 1,
        D3D12TEST_FEATURE_MEMORY_ARCHITECTURE	= 2,
        D3D12TEST_FEATURE_RESIDENCY	= 3,
        D3D12TEST_FEATURE_DEVICE_VALIDATION_INFO	= 4
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
    BOOL StackDepthSlices;
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
enum D3D12TEST_HEAP_FLAGS
    {
        D3D12TEST_HEAP_NONE	= 0,
        D3D12TEST_HEAP_SHARED_HANDLE_NT	= 0x2,
        D3D12TEST_HEAP_SHARED_HANDLE_GDI	= 0x4,
        D3D12TEST_HEAP_SHARED_HANDLE_MASK	= 0x6,
        D3D12TEST_HEAP_SHARED_KEYED_MUTEX	= 0x8,
        D3D12TEST_HEAP_SHARED_CROSS_ADAPTER	= 0x10,
        D3D12TEST_HEAP_SHARED_RESTRICTED_TO_DWM	= 0x20,
        D3D12TEST_HEAP_FLIP_MODEL_BACK_BUFFER	= 0x40,
        D3D12TEST_HEAP_PRESENTABLE	= 0x80,
        D3D12TEST_HEAP_INFER_DIMENSION_ATTRIBUTION	= 0x100,
        D3D12TEST_HEAP_PRIMARY	= 0x200,
        D3D12TEST_HEAP_CREATED_NOT_FROM_D3D12	= 0x400,
        D3D12TEST_HEAP_ACQUIREABLE_ON_WRITE	= 0x800,
        D3D12TEST_HEAP_UNSYNCHRONIZED_FLIPS	= 0x1000,
        D3D12TEST_HEAP_LEGACY_SEMANTICS	= 0x2000
    } 	D3D12TEST_HEAP_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS( D3D12TEST_HEAP_FLAGS );
typedef struct D3D12TEST_FEATURE_DATA_MEMORY_ARCHITECTURE
    {
    UINT DriverSupportedVer;
    BOOL UMA;
    BOOL CacheCoherentUMA;
    BOOL IOCoherent;
    D3D12TEST_HEAP_FLAGS DisabledHeapFlags;
    } 	D3D12TEST_FEATURE_DATA_MEMORY_ARCHITECTURE;

typedef 
enum D3D12TEST_RESOURCE_FLAGS
    {
        D3D12TEST_RESOURCE_NONE	= 0,
        D3D12TEST_RESOURCE_VIDEO	= 0x1,
        D3D12TEST_RESOURCE_IMPLICIT_BUFFER	= 0x2
    } 	D3D12TEST_RESOURCE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS( D3D12TEST_RESOURCE_FLAGS );
typedef 
enum D3D12TEST_OFFER_RESOURCE_PRIORITY
    {
        D3D12TEST_OFFER_RESOURCE_PRIORITY_LOW	= 1,
        D3D12TEST_OFFER_RESOURCE_PRIORITY_NORMAL	= ( D3D12TEST_OFFER_RESOURCE_PRIORITY_LOW + 1 ) ,
        D3D12TEST_OFFER_RESOURCE_PRIORITY_HIGH	= ( D3D12TEST_OFFER_RESOURCE_PRIORITY_NORMAL + 1 ) 
    } 	D3D12TEST_OFFER_RESOURCE_PRIORITY;



extern RPC_IF_HANDLE __MIDL_itf_d3d12InternalTest_0000_0000_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12InternalTest_0000_0000_v0_0_s_ifspec;

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
        
        virtual HRESULT STDMETHODCALLTYPE CheckTestFeatureSupport( 
            D3D12TEST_FEATURE Feature,
            /* [annotation] */ 
            _Out_writes_bytes_( FeatureDataSize )  void *pFeatureData,
            UINT FeatureDataSize) = 0;
        
        virtual UINT64 STDMETHODCALLTYPE GetDriverVersion( void) = 0;
        
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
            D3D12TEST_HEAP_FLAGS HeapTestMiscFlag,
            /* [annotation] */ 
            _In_opt_  ID3D12Heap *pHeap,
            UINT64 HeapOffset,
            /* [annotation] */ 
            _Inout_opt_  void *pPrimaryDesc,
            /* [annotation] */ 
            _In_opt_  const D3D12_RESOURCE_DESC *pResourceDesc,
            /* [annotation] */ 
            _In_opt_  const D3D11_RESOURCE_FLAGS *pFlags11,
            D3D12TEST_RESOURCE_FLAGS ResourceTestMiscFlag,
            D3D12_RESOURCE_STATES InitialState,
            /* [annotation] */ 
            _In_opt_  const D3D12_CLEAR_VALUE *pOptimizedClearValue,
            /* [annotation] */ 
            _In_opt_  IUnknown *pOuterUnk,
            /* [in] */ REFIID riidResource,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvResource,
            /* [in] */ REFIID riidHeap,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvHeap) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE OpenTestResourceAndHeap( 
            HANDLE hHandle,
            D3D12_RESOURCE_STATES InitialState,
            /* [in] */ REFIID riidResource,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvResource,
            /* [in] */ REFIID riidHeap,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvHeap) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateSharedHandleInternal( 
            /* [annotation] */ 
            _In_  ID3D12DeviceChild *pObject,
            /* [annotation] */ 
            _In_opt_  const SELWRITY_ATTRIBUTES *pAttributes,
            DWORD Access,
            /* [annotation] */ 
            _In_opt_  LPCWSTR pName,
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
            _Out_  D3D11_RESOURCE_FLAGS *__MIDL__ID3D12DeviceTest0001) = 0;
        
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
        
        virtual UINT STDMETHODCALLTYPE GetDDINodeIndex( 
            UINT Mask) = 0;
        
        virtual void STDMETHODCALLTYPE ReleaseResources( 
            /* [annotation] */ 
            _In_reads_(Surfaces)  ID3D12Resource *const *pReleaseSurfaces,
            /* [annotation] */ 
            _In_range_(0,DXGI_MAX_SWAP_CHAIN_BUFFERS+1)  UINT Surfaces,
            /* [annotation] */ 
            _In_reads_opt_(1)  ID3D12Resource *pLwrSurface) = 0;
        
        virtual UINT STDMETHODCALLTYPE GetMaxFrameLatency( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE OfferResources( 
            UINT NumObjects,
            /* [annotation] */ 
            _In_reads_(NumObjects)  ID3D12Pageable *const *ppObjects,
            D3D12TEST_OFFER_RESOURCE_PRIORITY Priority) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE ReclaimResources( 
            UINT NumObjects,
            /* [annotation] */ 
            _In_reads_(NumObjects)  ID3D12Pageable *const *ppObjects,
            /* [annotation] */ 
            _Out_writes_(NumObjects)  BOOL *pDiscarded) = 0;
        
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
        
        HRESULT ( STDMETHODCALLTYPE *CheckTestFeatureSupport )( 
            ID3D12DeviceTest * This,
            D3D12TEST_FEATURE Feature,
            /* [annotation] */ 
            _Out_writes_bytes_( FeatureDataSize )  void *pFeatureData,
            UINT FeatureDataSize);
        
        UINT64 ( STDMETHODCALLTYPE *GetDriverVersion )( 
            ID3D12DeviceTest * This);
        
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
            D3D12TEST_HEAP_FLAGS HeapTestMiscFlag,
            /* [annotation] */ 
            _In_opt_  ID3D12Heap *pHeap,
            UINT64 HeapOffset,
            /* [annotation] */ 
            _Inout_opt_  void *pPrimaryDesc,
            /* [annotation] */ 
            _In_opt_  const D3D12_RESOURCE_DESC *pResourceDesc,
            /* [annotation] */ 
            _In_opt_  const D3D11_RESOURCE_FLAGS *pFlags11,
            D3D12TEST_RESOURCE_FLAGS ResourceTestMiscFlag,
            D3D12_RESOURCE_STATES InitialState,
            /* [annotation] */ 
            _In_opt_  const D3D12_CLEAR_VALUE *pOptimizedClearValue,
            /* [annotation] */ 
            _In_opt_  IUnknown *pOuterUnk,
            /* [in] */ REFIID riidResource,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvResource,
            /* [in] */ REFIID riidHeap,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvHeap);
        
        HRESULT ( STDMETHODCALLTYPE *OpenTestResourceAndHeap )( 
            ID3D12DeviceTest * This,
            HANDLE hHandle,
            D3D12_RESOURCE_STATES InitialState,
            /* [in] */ REFIID riidResource,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvResource,
            /* [in] */ REFIID riidHeap,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_opt_  void **ppvHeap);
        
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
            _Out_  D3D11_RESOURCE_FLAGS *__MIDL__ID3D12DeviceTest0001);
        
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
        
        UINT ( STDMETHODCALLTYPE *GetDDINodeIndex )( 
            ID3D12DeviceTest * This,
            UINT Mask);
        
        void ( STDMETHODCALLTYPE *ReleaseResources )( 
            ID3D12DeviceTest * This,
            /* [annotation] */ 
            _In_reads_(Surfaces)  ID3D12Resource *const *pReleaseSurfaces,
            /* [annotation] */ 
            _In_range_(0,DXGI_MAX_SWAP_CHAIN_BUFFERS+1)  UINT Surfaces,
            /* [annotation] */ 
            _In_reads_opt_(1)  ID3D12Resource *pLwrSurface);
        
        UINT ( STDMETHODCALLTYPE *GetMaxFrameLatency )( 
            ID3D12DeviceTest * This);
        
        HRESULT ( STDMETHODCALLTYPE *OfferResources )( 
            ID3D12DeviceTest * This,
            UINT NumObjects,
            /* [annotation] */ 
            _In_reads_(NumObjects)  ID3D12Pageable *const *ppObjects,
            D3D12TEST_OFFER_RESOURCE_PRIORITY Priority);
        
        HRESULT ( STDMETHODCALLTYPE *ReclaimResources )( 
            ID3D12DeviceTest * This,
            UINT NumObjects,
            /* [annotation] */ 
            _In_reads_(NumObjects)  ID3D12Pageable *const *ppObjects,
            /* [annotation] */ 
            _Out_writes_(NumObjects)  BOOL *pDiscarded);
        
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

#define ID3D12DeviceTest_CheckTestFeatureSupport(This,Feature,pFeatureData,FeatureDataSize)	\
    ( (This)->lpVtbl -> CheckTestFeatureSupport(This,Feature,pFeatureData,FeatureDataSize) ) 

#define ID3D12DeviceTest_GetDriverVersion(This)	\
    ( (This)->lpVtbl -> GetDriverVersion(This) ) 

#define ID3D12DeviceTest_GetAdapterFunctionTable(This,ppTable)	\
    ( (This)->lpVtbl -> GetAdapterFunctionTable(This,ppTable) ) 

#define ID3D12DeviceTest_GetFunctionTableAndInfo(This,TableType,TableNum,bDeviceRemoved,pOffset,pSize,ppTable)	\
    ( (This)->lpVtbl -> GetFunctionTableAndInfo(This,TableType,TableNum,bDeviceRemoved,pOffset,pSize,ppTable) ) 

#define ID3D12DeviceTest_GetGlobalTableAndSize(This,ppTable,pSize)	\
    ( (This)->lpVtbl -> GetGlobalTableAndSize(This,ppTable,pSize) ) 

#define ID3D12DeviceTest_GetPrevGlobalTables(This,pNum,prevTables)	\
    ( (This)->lpVtbl -> GetPrevGlobalTables(This,pNum,prevTables) ) 

#define ID3D12DeviceTest_CreateTestResourceAndHeap(This,pHeapDesc,HeapTestMiscFlag,pHeap,HeapOffset,pPrimaryDesc,pResourceDesc,pFlags11,ResourceTestMiscFlag,InitialState,pOptimizedClearValue,pOuterUnk,riidResource,ppvResource,riidHeap,ppvHeap)	\
    ( (This)->lpVtbl -> CreateTestResourceAndHeap(This,pHeapDesc,HeapTestMiscFlag,pHeap,HeapOffset,pPrimaryDesc,pResourceDesc,pFlags11,ResourceTestMiscFlag,InitialState,pOptimizedClearValue,pOuterUnk,riidResource,ppvResource,riidHeap,ppvHeap) ) 

#define ID3D12DeviceTest_OpenTestResourceAndHeap(This,hHandle,InitialState,riidResource,ppvResource,riidHeap,ppvHeap)	\
    ( (This)->lpVtbl -> OpenTestResourceAndHeap(This,hHandle,InitialState,riidResource,ppvResource,riidHeap,ppvHeap) ) 

#define ID3D12DeviceTest_CreateSharedHandleInternal(This,pObject,pAttributes,Access,pName,pHandle)	\
    ( (This)->lpVtbl -> CreateSharedHandleInternal(This,pObject,pAttributes,Access,pName,pHandle) ) 

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

#define ID3D12DeviceTest_GetDDINodeIndex(This,Mask)	\
    ( (This)->lpVtbl -> GetDDINodeIndex(This,Mask) ) 

#define ID3D12DeviceTest_ReleaseResources(This,pReleaseSurfaces,Surfaces,pLwrSurface)	\
    ( (This)->lpVtbl -> ReleaseResources(This,pReleaseSurfaces,Surfaces,pLwrSurface) ) 

#define ID3D12DeviceTest_GetMaxFrameLatency(This)	\
    ( (This)->lpVtbl -> GetMaxFrameLatency(This) ) 

#define ID3D12DeviceTest_OfferResources(This,NumObjects,ppObjects,Priority)	\
    ( (This)->lpVtbl -> OfferResources(This,NumObjects,ppObjects,Priority) ) 

#define ID3D12DeviceTest_ReclaimResources(This,NumObjects,ppObjects,pDiscarded)	\
    ( (This)->lpVtbl -> ReclaimResources(This,NumObjects,ppObjects,pDiscarded) ) 

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


/* interface __MIDL_itf_d3d12InternalTest_0000_0001 */
/* [local] */ 

typedef 
enum D3D12_COMMAND_QUEUE_CAP
    {
        D3D12_COMMAND_LIST_SIGNAL_SUPPORTED	= ( 1 << 0 ) 
    } 	D3D12_COMMAND_QUEUE_CAP;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_COMMAND_QUEUE_CAP);


extern RPC_IF_HANDLE __MIDL_itf_d3d12InternalTest_0000_0001_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12InternalTest_0000_0001_v0_0_s_ifspec;

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
        
        virtual void *STDMETHODCALLTYPE GetCriticalSection( void) = 0;
        
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
        
        void *( STDMETHODCALLTYPE *GetCriticalSection )( 
            ID3D12CommandQueueTest * This);
        
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

#define ID3D12CommandQueueTest_GetCriticalSection(This)	\
    ( (This)->lpVtbl -> GetCriticalSection(This) ) 

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
            _In_reads_(Count)  const D3D12_RESOURCE_BARRIER *pDesc) = 0;
        
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
            _In_opt_  const D3D12_BOX *pSrcBox) = 0;
        
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
            _In_reads_(Count)  const D3D12_RESOURCE_BARRIER *pDesc);
        
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
            _In_opt_  const D3D12_BOX *pSrcBox);
        
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

#define ID3D12CommandListTest_TestCopyTextureRegion(This,pDst,DstDepthPitch,DstX,DstY,DstZ,pSrc,SrcDepthPitch,pSrcBox)	\
    ( (This)->lpVtbl -> TestCopyTextureRegion(This,pDst,DstDepthPitch,DstX,DstY,DstZ,pSrc,SrcDepthPitch,pSrcBox) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12CommandListTest_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12InternalTest_0000_0003 */
/* [local] */ 

typedef 
enum D3D12_TEST_CREATE_DEVICE_FLAGS
    {
        D3D12_TEST_CREATE_DEVICE_FLAGS_NONE	= 0,
        D3D12_TEST_CREATE_DEVICE_FLAGS_DEBUG	= 0x1
    } 	D3D12_TEST_CREATE_DEVICE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_TEST_CREATE_DEVICE_FLAGS);


extern RPC_IF_HANDLE __MIDL_itf_d3d12InternalTest_0000_0003_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12InternalTest_0000_0003_v0_0_s_ifspec;

/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


