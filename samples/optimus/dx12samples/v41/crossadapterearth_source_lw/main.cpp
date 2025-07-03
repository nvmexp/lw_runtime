#include <windows.h>
#include <d3d11.h>
#include <exception>
#include <iostream>
#include <atlbase.h>
#include <vector>
#include <list>
#include <algorithm>
#include <fstream>
#include <map>
#include <assert.h>
#include <d3d12.h>
#include <d3dx12.h>
#include <directxmath.h>
#include <wincodec.h>
#include <dxgi1_4.h>
#include <mmsystem.h>

#include "shaders.h"

using namespace DirectX;
using namespace std;

static const D3D12_COMMAND_QUEUE_DESC s_CommandQueueDesc;
static const D3D12_COMMAND_QUEUE_DESC s_CopyCommandQueueDesc = {D3D12_COMMAND_LIST_TYPE_COPY, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};

class CDescriptorHeapWrapper
{
public:
    CDescriptorHeapWrapper() { memset(this, 0, sizeof(*this)); }

    HRESULT Create(
        ID3D12Device* pDevice, 
        D3D12_DESCRIPTOR_HEAP_TYPE Type, 
        UINT NumDescriptors, 
        bool bShaderVisible = false)
    {
        Desc.Type = Type;
        Desc.NumDescriptors = NumDescriptors;
        Desc.Flags = (bShaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE);

        HRESULT hr = pDevice->CreateDescriptorHeap(&Desc, 
                               __uuidof(ID3D12DescriptorHeap), 
                               (void**)&pDH);
        if (FAILED(hr)) return hr;

        hCPUHeapStart = pDH->GetCPUDescriptorHandleForHeapStart();
        if (bShaderVisible)
        {
            hGPUHeapStart = pDH->GetGPUDescriptorHandleForHeapStart();
        }
        else
        { 
            hGPUHeapStart.ptr = 0;
        }
        HandleIncrementSize = pDevice->GetDescriptorHandleIncrementSize(Desc.Type);
        return hr;
    }
    operator ID3D12DescriptorHeap*() { return pDH; }

    CD3DX12_CPU_DESCRIPTOR_HANDLE hCPU(UINT index)
    {
        return CD3DX12_CPU_DESCRIPTOR_HANDLE(hCPUHeapStart,index,HandleIncrementSize); 
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE hGPU(UINT index) 
    {
        assert(Desc.Flags&D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE); 
        return CD3DX12_GPU_DESCRIPTOR_HANDLE(hGPUHeapStart,index,HandleIncrementSize); 
    }

    D3D12_DESCRIPTOR_HEAP_DESC Desc;
    CComPtr<ID3D12DescriptorHeap> pDH;
    CD3DX12_CPU_DESCRIPTOR_HANDLE hCPUHeapStart;
    CD3DX12_GPU_DESCRIPTOR_HANDLE hGPUHeapStart;
    UINT HandleIncrementSize;
};

HRESULT D3D12CreateDeviceHelper(
    _In_opt_ IDXGIAdapter* pAdapter,
    D3D_FEATURE_LEVEL MinimumFeatureLevel,
    _In_ REFIID riidQueue,
    _COM_Outptr_opt_ void** ppQueue,    
    _In_ REFIID riidDevice,
    _COM_Outptr_opt_ void** ppDevice 
    )
{
    CComPtr<ID3D12Device> pDevice;
    CComPtr<ID3D12CommandQueue> pQueue;

    CComPtr<IUnknown> pDeviceOut;
    CComPtr<IUnknown> pQueueOut;

    HRESULT hr = D3D12CreateDevice(
        pAdapter,
        MinimumFeatureLevel,
        IID_PPV_ARGS(&pDevice)
        );
    if(FAILED(hr)) { return hr; }

    hr = pDevice->CreateCommandQueue(&s_CommandQueueDesc, IID_PPV_ARGS(&pQueue));
    if(FAILED(hr)) { return hr; }
    
    hr = pDevice->QueryInterface(riidDevice, reinterpret_cast<void**>(&pDeviceOut));
    if(FAILED(hr)) { return hr; }

    hr = pQueue->QueryInterface(riidQueue, reinterpret_cast<void**>(&pQueueOut));
    if(FAILED(hr)) { return hr; }

    *ppDevice = pDeviceOut.Detach();
    *ppQueue = pQueueOut.Detach();

    return S_OK;
}

#define NUM_SPHERE_RINGS    15
#define NUM_SPHERE_SEGMENTS 30
const int NUM_VERTICES        = 2 * NUM_SPHERE_RINGS * ( NUM_SPHERE_SEGMENTS + 1 );

const GUID WKPDID_D3DDebugObjectNameW = { 0x4cca5fd8, 0x921f, 0x42c8, 0x85, 0x66, 0x70, 0xca, 0xf2, 0xa9, 0xb7, 0x41 };

struct ConstantBufferContents
{
    XMMATRIX worldViewProjection;    // World * View * Projection matrix
    XMMATRIX ilwWorldViewProjection;
    XMMATRIX world;
    XMMATRIX worldView;
    XMVECTOR eyePt;
};

struct
{
    int          argc;
    const char** argv;
} g_commandLine;

bool g_bCreateOurOwnDevice = false;
bool g_bCreateWarpRenderDevice = false;
bool g_bUseCopyQueue = false;
bool g_bUseAA = false;

const char* 
GetCommandLine(
    const char* key, 
    const char* dflt
    )
{
    for (int i = 1; i < (g_commandLine.argc-1); i++)
    {
        if (0 == _stricmp(g_commandLine.argv[i], key))
        {
            return g_commandLine.argv[i+1];
        }
    }

    return dflt;
}

void CheckBool(BOOL b)
{
    if (!b)
    {
        __debugbreak();
    }
}

void CheckHr(HRESULT hr)
{
    if (FAILED(hr))
    {
        __debugbreak();
        throw exception("Unexpected HRESULT");
    }
}

struct D3DXVECTOR2
{
    float x;
    float y;

    D3DXVECTOR2()
    {
    }

    D3DXVECTOR2(float u, float v)
    {
        x = u;
        y = v;
    }
};

struct D3DXVECTOR3
{
    float x;
    float y;
    float z;

    D3DXVECTOR3()
    {
    }

    D3DXVECTOR3(float a, float b, float c)
    {
        x = a;
        y = b;
        z = c;
    }
};

struct D3DXVECTOR4
{
    float x;
    float y;
    float z;
    float w;

    D3DXVECTOR4()
    {
    }

    D3DXVECTOR4(float a, float b, float c, float d)
    {
        x = a;
        y = b;
        z = c;
        w = d;
    }
};

const float D3DX_PI = 3.14159f;

struct GEOMETRY_VERTEX
{
    D3DXVECTOR3 pos;
    D3DXVECTOR2 tex1;
};

struct ELW_VERTEX
{
    D3DXVECTOR4 pos;
    D3DXVECTOR2 tex;
};

void CreateElwVertices(ELW_VERTEX* vertices)
{
    float fHighW = -1.0f;
    float fHighH = -1.0f;
    float fLowW  = 1.0f;
    float fLowH  = 1.0f;

    vertices[0].pos = D3DXVECTOR4( fLowW, fLowH, 1.0f, 1.0f );
    vertices[1].pos = D3DXVECTOR4( fLowW, fHighH, 1.0f, 1.0f );
    vertices[2].pos = D3DXVECTOR4( fHighW, fLowH, 1.0f, 1.0f );
    vertices[3].pos = D3DXVECTOR4( fHighW, fHighH, 1.0f, 1.0f );

    vertices[0].tex = D3DXVECTOR2(0.8f, 0.2f);
    vertices[1].tex = D3DXVECTOR2(0.8f, 0.8f);
    vertices[2].tex = D3DXVECTOR2(0.3f, 0.2f);
    vertices[3].tex = D3DXVECTOR2(0.3f, 0.8f);
}

void CreateSphereVertices(GEOMETRY_VERTEX *pSphereVertices)
{
    // Establish constants used in sphere generation
    float fDeltaRingAngle = ( D3DX_PI / NUM_SPHERE_RINGS );
    float fDeltaSegAngle  = ( 2.0f * D3DX_PI / NUM_SPHERE_SEGMENTS );

    D3DXVECTOR4 vT;
    float fScale;

    // Generate the group of rings for the sphere
    for( UINT ring = 0; ring < NUM_SPHERE_RINGS; ring++ )
    {
        float r0 = sinf( (ring+0) * fDeltaRingAngle );
        float r1 = sinf( (ring+1) * fDeltaRingAngle );
        float y0 = cosf( (ring+0) * fDeltaRingAngle );
        float y1 = cosf( (ring+1) * fDeltaRingAngle );

        // Generate the group of segments for the current ring
        for( UINT seg = 0; seg < (NUM_SPHERE_SEGMENTS + 1); seg++ )
        {
            float x0 =  r0 * sinf( seg * fDeltaSegAngle );
            float z0 =  r0 * cosf( seg * fDeltaSegAngle );
            float x1 =  r1 * sinf( seg * fDeltaSegAngle );
            float z1 =  r1 * cosf( seg * fDeltaSegAngle );

            // Add two vertices to the strip which makes up the sphere

            pSphereVertices->pos = D3DXVECTOR3( x0, y0, z0 );
            pSphereVertices->tex1.x = 1.0f - ((FLOAT)seg) / NUM_SPHERE_SEGMENTS;
            pSphereVertices->tex1.y = (ring+0) / (FLOAT)NUM_SPHERE_RINGS;

            pSphereVertices++;

            pSphereVertices->pos = D3DXVECTOR3( x1, y1, z1 );
            pSphereVertices->tex1.x = 1.0f - ((FLOAT)seg) / NUM_SPHERE_SEGMENTS;
            pSphereVertices->tex1.y = (ring+1) / (FLOAT)NUM_SPHERE_RINGS;

            pSphereVertices++;
        }
    }
}

bool g_bClosing = false;
LRESULT CALLBACK MyWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_CLOSE:
        g_bClosing = true;
        return 0;

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
}

class App
{
public:
    App()
        : m_rotateAngle(0.0f)
        , m_fenceValue(0)
    {
    }

    ~App()
    {
       // WaitForSingleObject(m_previousFrameCompleteEvent, INFINITE);
        FlushAndFinish();

        if (m_PresentDevice.spSwapChain)
        {
            m_PresentDevice.spSwapChain->SetFullscreenState(FALSE, NULL);
        }
    }

    void Init()
    {
        const UINT windowSize = 768;

        m_previousFrameCompleteEvent = CreateEvent(
            NULL,
            FALSE,
            TRUE,
            NULL
            );

        WNDCLASSEX wcex;
        ZeroMemory(&wcex, sizeof(wcex));

        wcex.cbSize         = sizeof(WNDCLASSEX);
        wcex.style          = CS_HREDRAW | CS_VREDRAW;
        wcex.lpfnWndProc    = MyWndProc;
        wcex.lpszClassName  = "BlakeWindow";

        RegisterClassEx(&wcex);

        m_window = CreateWindow(
                "BlakeWindow",
                "D3D12Test",
                WS_VISIBLE | WS_OVERLAPPEDWINDOW,
                0,
                0,
                windowSize,
                windowSize,
                NULL,
                NULL,
                NULL,
                NULL
                );

        RECT clientRect;
        GetClientRect(m_window, &clientRect);

        UINT width = clientRect.right;
        UINT height = clientRect.bottom;

        m_width = width;
        m_height = height;

        if (g_bCreateOurOwnDevice)
        {
            CComPtr<IDXGIAdapter> pAdapter0;
            CComPtr<IDXGIAdapter> pAdapter1;
            DXGI_ADAPTER_DESC adapter0Desc;
            DXGI_ADAPTER_DESC adapter1Desc;
            {
                CComPtr<IDXGIFactory4> spFactory;
                CheckHr(CreateDXGIFactory2(0, IID_PPV_ARGS(&spFactory)));
                CheckHr(spFactory->EnumAdapters(0, &pAdapter0));
                CheckHr(spFactory->EnumAdapters(1, &pAdapter1));

                CheckHr(pAdapter0->GetDesc(&adapter0Desc));
                CheckHr(pAdapter1->GetDesc(&adapter1Desc));
            }

            // Lwpu device always a render device
            // Intel's a present device
            CComPtr<IDXGIAdapter> pAdapterR;
            CComPtr<IDXGIAdapter> pAdapterP;
            if (adapter0Desc.VendorId == 0x10de)
            {
                pAdapterR = pAdapter0;
                pAdapterP = pAdapter1;
            }
            else if (adapter1Desc.VendorId == 0x10de)
            {
                pAdapterR = pAdapter1;
                pAdapterP = pAdapter0;
            }
            
            if (g_bCreateWarpRenderDevice)
            {
                CComPtr<IDXGIAdapter> spWarpAdapter;
                {
                    CComPtr<IDXGIFactory4> spFactory;
                    CheckHr(CreateDXGIFactory2(0, IID_PPV_ARGS(&spFactory)));
                    CheckHr(spFactory->EnumWarpAdapter(IID_PPV_ARGS(&spWarpAdapter)));
                }

                pAdapterR = spWarpAdapter;
            }

            CheckHr(D3D12CreateDeviceHelper(
                pAdapterP,
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&m_PresentDevice.spCommandQueue),
                IID_PPV_ARGS(&m_PresentDevice.spDevice)
                ));

            CheckHr(D3D12CreateDeviceHelper(
                pAdapterR,
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&m_RenderDevice.spCommandQueue),
                IID_PPV_ARGS(&m_RenderDevice.spDevice)
                ));

            if (g_bUseCopyQueue)
            {
                CheckHr(m_RenderDevice.spDevice->CreateCommandQueue(&s_CopyCommandQueueDesc, IID_PPV_ARGS(&m_RenderDevice.spCopyCommandQueue)));
            }
        }
        else
        {
            CComPtr<IDXGIAdapter> spWarpAdapter;
            {
                CComPtr<IDXGIFactory4> spFactory;
                CheckHr(CreateDXGIFactory2(0, IID_PPV_ARGS(&spFactory)));
                CheckHr(spFactory->EnumWarpAdapter(IID_PPV_ARGS(&spWarpAdapter)));
            }

            CheckHr(D3D12CreateDeviceHelper(
                NULL,
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&m_RenderDevice.spCommandQueue),
                IID_PPV_ARGS(&m_RenderDevice.spDevice)
                ));

            CheckHr(D3D12CreateDeviceHelper(
                spWarpAdapter,
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&m_PresentDevice.spCommandQueue),
                IID_PPV_ARGS(&m_PresentDevice.spDevice)
                ));
        }

        {
            DXGI_SWAP_CHAIN_DESC scd = {};

            scd.BufferDesc.Width = width;
            scd.BufferDesc.Height = height;
            scd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            scd.SampleDesc.Count = 1;
            scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            scd.BufferCount = 2;
            scd.OutputWindow = m_window;
            scd.Windowed = TRUE;
            scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

            CComPtr<IDXGIFactory> spDxgiFactory;
            CheckHr(CreateDXGIFactory1(IID_PPV_ARGS(&spDxgiFactory)));

            CheckHr(spDxgiFactory->CreateSwapChain(
                m_PresentDevice.spCommandQueue,
                &scd,
                &m_PresentDevice.spSwapChain
                ));
        }

        CheckHr(m_PresentDevice.spDevice->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&m_PresentDevice.spCommandAllocator)
            ));

        D3D12_RESOURCE_DESC SharedResourceDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_B8G8R8A8_UNORM, width, height, 1, 1);
        SharedResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        SharedResourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;

        D3D12_RESOURCE_ALLOCATION_INFO AllocationInfo = m_RenderDevice.spDevice->GetResourceAllocationInfo(0x1, 1, &SharedResourceDesc);

        D3D12_HEAP_DESC HeapDesc = {};
        HeapDesc.SizeInBytes = AllocationInfo.SizeInBytes;
        HeapDesc.Properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        HeapDesc.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;

        CheckHr(m_RenderDevice.spDevice->CreateHeap(&HeapDesc, IID_PPV_ARGS(&m_RenderDevice.spSharedHeap)));
        HANDLE SharedHandle = nullptr;
        
        CheckHr(m_RenderDevice.spDevice->CreateSharedHandle(
            m_RenderDevice.spSharedHeap,
            nullptr,
            GENERIC_ALL,
            nullptr,
            &SharedHandle
            ));

        CheckHr(m_PresentDevice.spDevice->OpenSharedHandle(
            SharedHandle,
            IID_PPV_ARGS( &m_PresentDevice.spSharedHeap )
            ));

        CloseHandle(SharedHandle);

        // Place resources on both heaps
        CheckHr(m_RenderDevice.spDevice->CreatePlacedResource(
            m_RenderDevice.spSharedHeap,
            0,
            &SharedResourceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS( &m_RenderDevice.spSharedResource )
            ));

        CheckHr(m_PresentDevice.spDevice->CreatePlacedResource(
            m_PresentDevice.spSharedHeap,
            0,
            &SharedResourceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS( &m_PresentDevice.spSharedResource )
            ));

        UINT sampleCount   = 1;
        UINT sampleQuality = 0;

        if (g_bUseAA)
        {
            // use 2xAA
            sampleCount = 2;

            // create a resolved resource, spRenderTargetResolvedResource
            CheckHr(m_RenderDevice.spDevice->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_B8G8R8A8_UNORM, width, height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                nullptr,
                IID_PPV_ARGS(&m_RenderDevice.spRenderTargetResolvedResource)
                ));
        }

        CheckHr(m_RenderDevice.spDevice->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_B8G8R8A8_UNORM, width, height, 1, 1, sampleCount, sampleQuality, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            nullptr,
            IID_PPV_ARGS(&m_RenderDevice.spRenderTargetResource)
            ));

        CheckHr(m_RenderDevice.RTVHeap.Create(m_RenderDevice.spDevice,D3D12_DESCRIPTOR_HEAP_TYPE_RTV,1));
        CheckHr(m_RenderDevice.OnlineDH.Create(m_RenderDevice.spDevice,D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,sc_NumSRVs+sc_NumCBVs,true));

        CheckHr(m_RenderDevice.spDevice->CreateFence(
            0, 
            D3D12_FENCE_FLAG_SHARED | D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER, 
            IID_PPV_ARGS(&m_RenderDevice.spSharedFence)
            ));

        HANDLE FenceHandle;
        CheckHr(m_RenderDevice.spDevice->CreateSharedHandle(
            m_RenderDevice.spSharedFence,
            nullptr,
            GENERIC_ALL,
            nullptr,
            &FenceHandle
            ));

        CheckHr(m_PresentDevice.spDevice->OpenSharedHandle(
            FenceHandle,
            IID_PPV_ARGS(&m_PresentDevice.spSharedFence)
            ));

        CloseHandle(FenceHandle);

        m_hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        CheckHr(m_RenderDevice.spDevice->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&m_RenderDevice.spCommandAllocator)
            ));

        if (g_bUseCopyQueue)
        {
            CheckHr(m_RenderDevice.spDevice->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_COPY,
                IID_PPV_ARGS(&m_RenderDevice.spCopyAllocator)
                ));
        }
        {
            CheckHr(m_RenderDevice.spDevice->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(8*1024*1024),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&m_RenderDevice.spUploadBuffer)
                ));

            CheckHr(m_RenderDevice.spUploadBuffer->Map(0, NULL,reinterpret_cast<void**>(&m_RenderDevice.pDataBegin)));
            m_RenderDevice.pDataLwr = m_RenderDevice.pDataBegin;
            m_RenderDevice.pDataEnd = m_RenderDevice.pDataBegin + m_RenderDevice.spUploadBuffer->GetDesc().Width;
        }

        {
            CheckHr(m_RenderDevice.spDevice->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(NUM_VERTICES*sizeof(GEOMETRY_VERTEX)+4*sizeof(ELW_VERTEX)),
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(&m_RenderDevice.spDefaultBuffer)
                ));
        }

        m_RenderDevice.VBVs[sc_ElwVBVHeapOffset].BufferLocation = m_RenderDevice.spDefaultBuffer->GetGPUVirtualAddress() + NUM_VERTICES*sizeof(GEOMETRY_VERTEX);
        m_RenderDevice.VBVs[sc_ElwVBVHeapOffset].SizeInBytes = 4 * sizeof(ELW_VERTEX);
        m_RenderDevice.VBVs[sc_ElwVBVHeapOffset].StrideInBytes = sizeof(ELW_VERTEX);

        m_RenderDevice.VBVs[sc_GeomVBVHeapOffset].BufferLocation = m_RenderDevice.spDefaultBuffer->GetGPUVirtualAddress();
        m_RenderDevice.VBVs[sc_GeomVBVHeapOffset].SizeInBytes = NUM_VERTICES*sizeof(GEOMETRY_VERTEX);
        m_RenderDevice.VBVs[sc_GeomVBVHeapOffset].StrideInBytes = sizeof(GEOMETRY_VERTEX);

        // Root signature
        {
            CD3DX12_DESCRIPTOR_RANGE DescRange[2];
            DescRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV,3,0); // t0-t2
            DescRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV,1,0); // b0

            CD3DX12_ROOT_PARAMETER RTSlot[2];

            RTSlot[sc_SRVTableBindSlot].InitAsDescriptorTable(1,&DescRange[0],D3D12_SHADER_VISIBILITY_PIXEL); // t0-t2
            RTSlot[sc_CBTableBindSlot].InitAsDescriptorTable(1,&DescRange[1],D3D12_SHADER_VISIBILITY_ALL); // b0

            D3D12_STATIC_SAMPLER_DESC Sampler[sc_NumSamplers];
            for(UINT i = 0; i < sc_NumSamplers; i++)
            {
                Sampler[i].Filter = D3D12_FILTER_ANISOTROPIC;
                Sampler[i].AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
                Sampler[i].AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
                Sampler[i].AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
                Sampler[i].MipLODBias = 0;
                Sampler[i].MaxAnisotropy = 16;
                Sampler[i].ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
                Sampler[i].BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
                Sampler[i].MinLOD = 0.0f;
                Sampler[i].MaxLOD = 9999.0f;
                Sampler[i].ShaderRegister = i;
                Sampler[i].RegisterSpace = 0;
                Sampler[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            }

            CD3DX12_ROOT_SIGNATURE_DESC RootSig(2,RTSlot,sc_NumSamplers,Sampler,D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
            CComPtr<ID3DBlob> pSerializedLayout;
            D3D12SerializeRootSignature(&RootSig, D3D_ROOT_SIGNATURE_VERSION_1, &pSerializedLayout,NULL);

            CheckHr(m_RenderDevice.spDevice->CreateRootSignature(
                1,
                pSerializedLayout->GetBufferPointer(), 
                pSerializedLayout->GetBufferSize(),
                __uuidof(ID3D12RootSignature),
                (void**)&m_RenderDevice.spRootSignature));
        }

        CheckHr(m_RenderDevice.spDevice->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS((&m_RenderDevice.spNonRenderingCommandAllocator.p))));

        CheckHr(m_RenderDevice.spDevice->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT, 
            m_RenderDevice.spNonRenderingCommandAllocator,
            nullptr,
            IID_PPV_ARGS(&m_RenderDevice.spCommandList)));

        UINT VerticesOffset = 0;
        {
            UINT8* pVertices = SuballocateFromUploadHeap(m_RenderDevice.spDefaultBuffer->GetDesc().Width);
            VerticesOffset = pVertices-m_RenderDevice.pDataBegin;

            CreateSphereVertices(reinterpret_cast<GEOMETRY_VERTEX*>(pVertices));
            
            CreateElwVertices(reinterpret_cast<ELW_VERTEX*>(pVertices+NUM_VERTICES*sizeof(GEOMETRY_VERTEX)));

            D3D12_RESOURCE_BARRIER desc;
            ZeroMemory( &desc, sizeof( desc ) );

            m_RenderDevice.spCommandList->CopyBufferRegion(
                m_RenderDevice.spDefaultBuffer,
                0,
                m_RenderDevice.spUploadBuffer,
                VerticesOffset,
                m_RenderDevice.spDefaultBuffer->GetDesc().Width
                );

            LoadTextures();

            desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            desc.Transition.pResource = m_RenderDevice.spDefaultBuffer;
            desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            desc.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            desc.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
            
            m_RenderDevice.spCommandList->ResourceBarrier(1, &desc);
        }

        CheckHr(m_RenderDevice.spCommandList->Close());

        m_RenderDevice.spCommandQueue->ExelwteCommandLists(1, CommandListCast(&m_RenderDevice.spCommandList.p));

        CheckHr(m_RenderDevice.spCommandList->Reset(m_RenderDevice.spNonRenderingCommandAllocator, nullptr));

        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc;
            ZeroMemory(&psoDesc, sizeof(psoDesc));
            psoDesc.pRootSignature = m_RenderDevice.spRootSignature;
            psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
            psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);

            D3D12_INPUT_ELEMENT_DESC layout[] =
            {
                { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
                { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            };

            UINT numElements = sizeof(layout)/sizeof(layout[0]);

            psoDesc.VS = { g_ElwVS, sizeof(g_ElwVS) };
            psoDesc.PS = { g_ElwPS, sizeof(g_ElwPS) };
            psoDesc.SampleMask = UINT_MAX;
            psoDesc.InputLayout = { layout, numElements };
            psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            psoDesc.NumRenderTargets = 1;
            psoDesc.RTVFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
            psoDesc.SampleDesc.Count = 1;
   
            CheckHr(m_RenderDevice.spDevice->CreateGraphicsPipelineState(
                &psoDesc,
                IID_PPV_ARGS(&m_RenderDevice.spBackgroundPSO)
                ));
        }

        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc;
            ZeroMemory(&psoDesc, sizeof(psoDesc));
            psoDesc.pRootSignature = m_RenderDevice.spRootSignature;
            psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
            psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);

            D3D12_INPUT_ELEMENT_DESC layout[] =
            {
                { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
                { "TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            };

            UINT numElements = sizeof(layout)/sizeof(layout[0]);

            psoDesc.VS = { g_GeometryVS, sizeof(g_GeometryVS) };
            psoDesc.PS = { g_GeometryPS, sizeof(g_GeometryPS) };
            psoDesc.SampleMask = UINT_MAX;
            psoDesc.InputLayout = { layout, numElements };
            psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            psoDesc.NumRenderTargets = 1;
            psoDesc.RTVFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
            psoDesc.SampleDesc.Count = 1;
   
            CheckHr(m_RenderDevice.spDevice->CreateGraphicsPipelineState(
                &psoDesc,
                IID_PPV_ARGS(&m_RenderDevice.spSpherePSO)
                ));
        }
    }

    CComPtr<ID3D12GraphicsCommandList> RecordCommandList()
    {
        CComPtr<ID3D12GraphicsCommandList> spCommandList;
        CheckHr(m_RenderDevice.spDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_RenderDevice.spCommandAllocator, m_RenderDevice.spBackgroundPSO, IID_PPV_ARGS(&spCommandList)));

        m_RenderDevice.spDevice->CreateRenderTargetView(m_RenderDevice.spRenderTargetResource, NULL, m_RenderDevice.RTVHeap.hCPU(0));

        m_RenderDevice.spDiffuseTexture->SetName(L"spDiffuseResource");
        m_RenderDevice.spBumpTexture->SetName(L"spBumpResource");
        m_RenderDevice.spElwTexture->SetName(L"spElwResource");

        ID3D12DescriptorHeap* pDH[1] = {m_RenderDevice.OnlineDH};
        spCommandList->SetDescriptorHeaps(1,pDH);
        spCommandList->SetGraphicsRootSignature(m_RenderDevice.spRootSignature);

        //
        // Resource barriers to transition all resources from to approriate usage
        //
        struct
        {
            ID3D12Resource* pResource;
            D3D12_RESOURCE_STATES stateBefore;
            D3D12_RESOURCE_STATES stateAfter;
        } barriers[] = 
        {
            { m_RenderDevice.spRenderTargetResource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET },
        };

        if (g_bUseAA)
        {
            barriers[0].pResource = m_RenderDevice.spRenderTargetResolvedResource;
        }

        {
            vector< D3D12_RESOURCE_BARRIER > barrierArray;

            for( UINT i = 0; i < _countof( barriers ); i++ )
            {
                D3D12_RESOURCE_BARRIER desc;
                ZeroMemory( &desc, sizeof( desc ) );

                desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                desc.Transition.pResource = barriers[i].pResource;
                desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                desc.Transition.StateBefore = barriers[i].stateBefore;
                desc.Transition.StateAfter = barriers[i].stateAfter;

                barrierArray.push_back( desc );
            }

            spCommandList->ResourceBarrier(
                barrierArray.size(),
                &(barrierArray[0])
                );
        }

        float clearColor[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
        spCommandList->ClearRenderTargetView(
            m_RenderDevice.RTVHeap.hCPU(0),
            clearColor,
            NULL,
            0
            );

        spCommandList->OMSetRenderTargets(1,&m_RenderDevice.RTVHeap.hCPU(0),true,NULL);

        D3D12_VIEWPORT viewport = 
        {
            0.0f,
            0.0f, 
            static_cast<float>(m_width), 
            static_cast<float>(m_height), 
            0.0f, 
            1.0f 
        };

        spCommandList->RSSetViewports(1, &viewport);

        D3D12_RECT scissorRect = {0, 0, m_width, m_height};
        spCommandList->RSSetScissorRects(1, &scissorRect);

        spCommandList->SetGraphicsRootDescriptorTable( sc_CBTableBindSlot, m_RenderDevice.OnlineDH.hGPU(sc_CBVHeapOffset));

        //
        // Render the background
        //
        {
            UINT offsets[] = { 0 };
            UINT strides[] = { sizeof(ELW_VERTEX) };
            
            spCommandList->SetPipelineState(m_RenderDevice.spBackgroundPSO);

            spCommandList->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );

            spCommandList->IASetVertexBuffers(0,1,&m_RenderDevice.VBVs[sc_ElwVBVHeapOffset]);

            spCommandList->SetGraphicsRootDescriptorTable(sc_SRVTableBindSlot, m_RenderDevice.OnlineDH.hGPU(sc_ElwOnlineHeapOffset));

            spCommandList->DrawInstanced( 4, 1, 0, 0 );
        }

        //
        // Render the sphere
        //
        {
            spCommandList->SetPipelineState(m_RenderDevice.spSpherePSO);

            spCommandList->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );

            spCommandList->IASetVertexBuffers(0,1,&m_RenderDevice.VBVs[sc_GeomVBVHeapOffset]);

            spCommandList->SetGraphicsRootDescriptorTable(sc_SRVTableBindSlot, m_RenderDevice.OnlineDH.hGPU(sc_DiffuseOnlineHeapOffset)); // all srvs

            spCommandList->DrawInstanced( NUM_VERTICES - 1, 1, 0, 0 );
        }

        //
        // resolve the rendertarget
        if (g_bUseAA)
        {
            spCommandList->ResolveSubresource(m_RenderDevice.spRenderTargetResolvedResource, 0, m_RenderDevice.spRenderTargetResource, 0, DXGI_FORMAT_B8G8R8A8_UNORM);
        }

       {
            //
            // Resource barriers to transition all resources back to D3D12_RESOURCE_STATE_COMMON
            //
            vector< D3D12_RESOURCE_BARRIER > barrierArray;

            for( UINT i = 0; i < _countof( barriers ); i++ )
            {
                D3D12_RESOURCE_BARRIER desc;
                ZeroMemory( &desc, sizeof( desc ) );

                desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                desc.Transition.pResource = barriers[i].pResource;
                desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                desc.Transition.StateBefore = barriers[i].stateAfter;
                desc.Transition.StateAfter = barriers[i].stateBefore;

                barrierArray.push_back( desc );
            }

            spCommandList->ResourceBarrier(
                barrierArray.size(),
                &(barrierArray[0])
                );
        }

        return spCommandList;
    }

    static SIZE_T Align(SIZE_T uLocation, UINT32 uAlign=MEMORY_ALLOCATION_ALIGNMENT)
    {
        return (uLocation + ((SIZE_T)uAlign-1)) & ~((SIZE_T)uAlign-1);
    }

    void FlushAndFinish()
    {
        CheckHr(m_RenderDevice.spCommandList->Close());
        m_RenderDevice.spCommandQueue->ExelwteCommandLists(1, CommandListCast(&m_RenderDevice.spCommandList.p));
      
        CheckHr(m_RenderDevice.spCommandQueue->Signal(m_RenderDevice.spSharedFence, ++m_fenceValue));
        CheckHr(m_RenderDevice.spSharedFence->SetEventOnCompletion(m_fenceValue, m_hEvent));
        WaitForSingleObject(m_hEvent, INFINITE);
        CheckHr(m_RenderDevice.spNonRenderingCommandAllocator->Reset());
        CheckHr(m_RenderDevice.spCommandList->Reset(m_RenderDevice.spNonRenderingCommandAllocator, nullptr));
    }

    UINT8* SuballocateFromUploadHeap(SIZE_T uSize, UINT32 uAlign=MEMORY_ALLOCATION_ALIGNMENT)
    {
        CheckHr(uSize<m_RenderDevice.pDataEnd-m_RenderDevice.pDataBegin?S_OK:E_OUTOFMEMORY);

        m_RenderDevice.pDataLwr = reinterpret_cast<UINT8*>(Align(reinterpret_cast<SIZE_T>(m_RenderDevice.pDataLwr), uAlign));
        if (m_RenderDevice.pDataLwr >= m_RenderDevice.pDataEnd || m_RenderDevice.pDataLwr + uSize >= m_RenderDevice.pDataEnd)
        {
            FlushAndFinish();
            m_RenderDevice.pDataLwr = m_RenderDevice.pDataBegin;
        }

        UINT8* pRet = m_RenderDevice.pDataLwr;
        m_RenderDevice.pDataLwr += uSize;
        return pRet;
    }

    void Render()
    {
        //
        // Wait for the previous frame
        //
        WaitForSingleObject(m_previousFrameCompleteEvent, INFINITE);

        //
        // Reset backing store for command allocator
        //
        CheckHr(m_RenderDevice.spCommandAllocator->Reset());

        //
        // Record a new command list
        //
        CComPtr<ID3D12GraphicsCommandList> spCommandList = RecordCommandList();

        //
        // Update the constant buffer
        //
        {
            XMMATRIX world = XMMatrixRotationY(m_rotateAngle);

            XMVECTOR eye = XMVectorSet( 0.0f, 0.0f, -2.0f, 0.0f );
            XMVECTOR at = XMVectorSet( 0.0f, 0.0f, 0.0f, 0.0f );
            XMVECTOR up = XMVectorSet( 0.0f, 1.0f, 0.0f, 0.0f );
            XMMATRIX view = XMMatrixLookAtLH( eye, at, up );

            XMMATRIX projection = XMMatrixPerspectiveFovLH( XM_PIDIV2, m_width / (FLOAT)m_height, 0.01f, 100.0f );

            XMMATRIX wvp = world * view * projection;
            XMMATRIX ilwwvp = XMMatrixIlwerse(NULL, wvp);

            ConstantBufferContents cbContents;

            cbContents.worldViewProjection = XMMatrixTranspose(wvp);
            cbContents.ilwWorldViewProjection = XMMatrixTranspose(ilwwvp);
            cbContents.world = XMMatrixTranspose(world);
            cbContents.worldView = XMMatrixTranspose(world * view);
            cbContents.eyePt = eye;

            UINT8* pData = SuballocateFromUploadHeap(sizeof(cbContents), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            UINT64 Offset = pData - m_RenderDevice.pDataBegin;
            
            memcpy(pData, &cbContents, sizeof(cbContents));

            // CPU and GPU are serialized
            D3D12_CONSTANT_BUFFER_VIEW_DESC CBDesc = { m_RenderDevice.spUploadBuffer->GetGPUVirtualAddress() + Offset, Align(sizeof(cbContents), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT) };
            m_RenderDevice.spDevice->CreateConstantBufferView(
                &CBDesc,
                m_RenderDevice.OnlineDH.hCPU(sc_CBVHeapOffset)
                );
        }

        // before using copy queue, execute the current render command list
        UINT FenceValue;
        if (g_bUseCopyQueue)
        {
            CheckHr(spCommandList->Close());
            m_RenderDevice.spCommandQueue->ExelwteCommandLists(1, CommandListCast(&spCommandList.p));

            // sync before start copy queue
            FenceValue = ++m_fenceValue;
            CheckHr(m_RenderDevice.spCommandQueue->Signal(m_RenderDevice.spSharedFence, FenceValue));
            CheckHr(m_RenderDevice.spCopyCommandQueue->Wait(m_RenderDevice.spSharedFence, FenceValue));
        }

        // Copy to the shared resource
        CComPtr<ID3D12GraphicsCommandList> spCopyCommandList;
        {
            if (g_bUseCopyQueue)
            {
                // create copy command list
                CheckHr(m_RenderDevice.spDevice->CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_COPY,
                    m_RenderDevice.spCopyAllocator,
                    nullptr,
                    IID_PPV_ARGS( &spCopyCommandList )
                    ));
            }
            else
            {
                spCopyCommandList = spCommandList;
            }

            // share copy path between copy queue and render queue
            // note: copy command list only support D3D12_RESOURCE_STATE_COMMON, COPY_DEST, and COPY_SOURCE (not dolwmented)
            D3D12_RESOURCE_BARRIER BarrierDesc = {};
            BarrierDesc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            BarrierDesc.Transition.pResource = m_RenderDevice.spSharedResource;
            BarrierDesc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            BarrierDesc.Transition.StateBefore = g_bUseCopyQueue ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_GENERIC_READ;
            BarrierDesc.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;

            spCopyCommandList->ResourceBarrier(1, &BarrierDesc);

            // Copy
            if (g_bUseAA)
            {
                spCopyCommandList->CopyResource(m_RenderDevice.spSharedResource, m_RenderDevice.spRenderTargetResolvedResource);
            }
            else
            {
                spCopyCommandList->CopyResource(m_RenderDevice.spSharedResource, m_RenderDevice.spRenderTargetResource);
            }

            // Transition the back buffer to the Present state
            BarrierDesc.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            BarrierDesc.Transition.StateAfter = g_bUseCopyQueue ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_GENERIC_READ;

            spCopyCommandList->ResourceBarrier(1, &BarrierDesc);
        }

        CComPtr<ID3D12CommandQueue> spLocalCommandQueue;
        if (g_bUseCopyQueue)
        {
            spLocalCommandQueue = m_RenderDevice.spCopyCommandQueue;
        }
        else
        {
            spLocalCommandQueue = m_RenderDevice.spCommandQueue;
        }
        CheckHr(spCopyCommandList->Close());
        spLocalCommandQueue->ExelwteCommandLists(1, CommandListCast(&spCopyCommandList.p));

        // Signal a fence indicating that rendering is complete
        FenceValue = ++m_fenceValue;
        CheckHr(spLocalCommandQueue->Signal(m_RenderDevice.spSharedFence, FenceValue));

        // Wait for the fence on the presenting device
        CheckHr(m_PresentDevice.spCommandQueue->Wait(m_PresentDevice.spSharedFence, FenceValue));

        // Copy from the shared resource to the back buffer
        {
            CheckHr(m_PresentDevice.spCommandAllocator->Reset());

            CComPtr<ID3D12GraphicsCommandList> spCommandList;
            CheckHr(m_PresentDevice.spDevice->CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                m_PresentDevice.spCommandAllocator,
                nullptr,
                IID_PPV_ARGS( &spCommandList )
                ));

            // retrieve the backbuffer from swapchain
            CComPtr<IDXGISwapChain3> spSwapChain3;
            CheckHr(m_PresentDevice.spSwapChain->QueryInterface(&spSwapChain3));
            UINT backBufferIndex = spSwapChain3->GetLwrrentBackBufferIndex();

            CComPtr<ID3D12Resource> spBackBuffer;
            CheckHr(spSwapChain3->GetBuffer(backBufferIndex, IID_PPV_ARGS(&spBackBuffer)));

            D3D12_RESOURCE_BARRIER BarrierDesc = {};
            BarrierDesc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            BarrierDesc.Transition.pResource = spBackBuffer;
            BarrierDesc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            BarrierDesc.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            BarrierDesc.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;

            spCommandList->ResourceBarrier(1, &BarrierDesc);

            spCommandList->CopyResource(spBackBuffer, m_PresentDevice.spSharedResource);

            BarrierDesc.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            BarrierDesc.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;

            spCommandList->ResourceBarrier(1, &BarrierDesc);

            CheckHr(spCommandList->Close());
            m_PresentDevice.spCommandQueue->ExelwteCommandLists(1, CommandListCast(&spCommandList.p));
        }

        HRESULT hr = m_PresentDevice.spSwapChain->Present(0, 0);

        // present can fail after fs<->windowed transition without a call to ResizeBuffers
        if (FAILED(hr))
        {
            FlushAndFinish();
            m_PresentDevice.spSwapChain->ResizeBuffers(2 /*same as the one used in DXGI_SWAP_CHAIN_DESC*/, 0, 0, DXGI_FORMAT_UNKNOWN, 0);

            DXGI_SWAP_CHAIN_DESC newDesc;
            CheckHr(m_PresentDevice.spSwapChain->GetDesc(&newDesc));

            if (m_width  != newDesc.BufferDesc.Width  ||
                m_height != newDesc.BufferDesc.Height)
            {
                // recreate shared resources here, so that it matches the new window
                //__debugbreak();
                m_width  = newDesc.BufferDesc.Width;
                m_height = newDesc.BufferDesc.Height;

                RecreateSharedResources();
            }
        }

        CheckHr(m_PresentDevice.spCommandQueue->Signal(m_PresentDevice.spSharedFence, ++m_fenceValue));
        CheckHr(m_PresentDevice.spSharedFence->SetEventOnCompletion(m_fenceValue, m_previousFrameCompleteEvent));

        m_rotateAngle += 0.01f;
    }

    void RecreateSharedResources()
    {
        m_PresentDevice.spSharedHeap.Release();  // A
        m_RenderDevice.spSharedHeap.Release();   // B
                
        m_RenderDevice.spSharedResource.Release();  // C
        m_PresentDevice.spSharedResource.Release();  // D

        m_RenderDevice.spRenderTargetResolvedResource.Release();  // E
        m_RenderDevice.spRenderTargetResource.Release();  // F

        D3D12_RESOURCE_DESC SharedResourceDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_B8G8R8A8_UNORM, m_width, m_height, 1, 1);
        SharedResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        SharedResourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;

        D3D12_RESOURCE_ALLOCATION_INFO AllocationInfo = m_RenderDevice.spDevice->GetResourceAllocationInfo(0x1, 1, &SharedResourceDesc);

        D3D12_HEAP_DESC HeapDesc = {};
        HeapDesc.SizeInBytes = AllocationInfo.SizeInBytes;
        HeapDesc.Properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        HeapDesc.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;

        CheckHr(m_RenderDevice.spDevice->CreateHeap(&HeapDesc, IID_PPV_ARGS(&m_RenderDevice.spSharedHeap)));  // B
        HANDLE SharedHandle = nullptr;
        
        CheckHr(m_RenderDevice.spDevice->CreateSharedHandle(
            m_RenderDevice.spSharedHeap,
            nullptr,
            GENERIC_ALL,
            nullptr,
            &SharedHandle
            ));

        CheckHr(m_PresentDevice.spDevice->OpenSharedHandle(
            SharedHandle,
            IID_PPV_ARGS( &m_PresentDevice.spSharedHeap ) // A
            ));

        CloseHandle(SharedHandle);

        // Place resources on both heaps
        CheckHr(m_RenderDevice.spDevice->CreatePlacedResource(
            m_RenderDevice.spSharedHeap,
            0,
            &SharedResourceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS( &m_RenderDevice.spSharedResource )    // C
            ));

        CheckHr(m_PresentDevice.spDevice->CreatePlacedResource(
            m_PresentDevice.spSharedHeap,
            0,
            &SharedResourceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS( &m_PresentDevice.spSharedResource )  // D
            ));
        UINT sampleCount   = 1;
        UINT sampleQuality = 0;

        if (g_bUseAA)
        {
            // use 2xAA
            sampleCount = 2;

            // create a resolved resource, spRenderTargetResolvedResource
            CheckHr(m_RenderDevice.spDevice->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_B8G8R8A8_UNORM, m_width, m_height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                nullptr,
                IID_PPV_ARGS(&m_RenderDevice.spRenderTargetResolvedResource)  // E
                ));
        }

        CheckHr(m_RenderDevice.spDevice->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_B8G8R8A8_UNORM, m_width, m_height, 1, 1, sampleCount, sampleQuality, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            nullptr,
            IID_PPV_ARGS(&m_RenderDevice.spRenderTargetResource)  // F
            ));
    }

    void LoadTextures()
    {
        const int textures = 3;
        const char* names[textures] = {"EARTH","EARTHBUMP","LOBBY"};
        CComPtr<IWICBitmapFrameDecode> spFrameDecode[textures];
        CComPtr<IWICFormatColwerter> spFormatColwerter[textures];
        CD3DX12_RESOURCE_DESC RDescs[textures];
        const D3D12_RESOURCE_DESC* pRDescs = reinterpret_cast<const D3D12_RESOURCE_DESC*>(RDescs);
        D3D12_RESOURCE_ALLOCATION_INFO RAInfo[textures];
        CComPtr<ID3D12Resource> spTexture2D[textures];
        D3D12_RESOURCE_BARRIER RBDesc[textures];
        ZeroMemory(&RBDesc, sizeof(RBDesc));

        CComPtr<IWICImagingFactory> spWicFactory;
        CheckHr(CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, __uuidof(spWicFactory), (void**)&spWicFactory));

        for (int i = 0; i < textures; ++i)
        {
            HRSRC hResource = FindResource(
                NULL,
                names[i],
                RT_RCDATA
                );

            HGLOBAL hGlobal = LoadResource(
                NULL,
                hResource
                );

            void *pData = LockResource(hGlobal);
            UINT dataSize = SizeofResource(NULL, hResource);

            CComPtr<IWICStream> spWicStream;
            CheckHr(spWicFactory->CreateStream(&spWicStream));

            CheckHr(spWicStream->InitializeFromMemory(
                (WICInProcPointer)pData,
                dataSize
                ));

            CComPtr<IWICBitmapDecoder> spBitmapDecoder;
            CheckHr(spWicFactory->CreateDecoderFromStream(
                spWicStream,
                NULL,
                WICDecodeMetadataCacheOnDemand,
                &spBitmapDecoder
                ));

            CheckHr(spBitmapDecoder->GetFrame(0, &spFrameDecode[i]));

            CheckHr(spWicFactory->CreateFormatColwerter(&spFormatColwerter[i]));
            
            CheckHr(spFormatColwerter[i]->Initialize(
                spFrameDecode[i],
                GUID_WICPixelFormat32bppPBGRA,
                WICBitmapDitherTypeNone,
                NULL,
                0.0,
                WICBitmapPaletteTypeLwstom
                ));

            UINT width;
            UINT height;
            CheckHr(spFrameDecode[i]->GetSize(&width, &height));

            RDescs[i] = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_B8G8R8A8_UNORM, width, height, 1, 1);
            RAInfo[i] = m_RenderDevice.spDevice->GetResourceAllocationInfo(1, 1, &RDescs[i]);
        }

        CheckHr(m_RenderDevice.spDevice->CreateHeap(
            &CD3DX12_HEAP_DESC(
                m_RenderDevice.spDevice->GetResourceAllocationInfo(1, textures, pRDescs),
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES),
            IID_PPV_ARGS(&m_RenderDevice.spTextureHeap)
            ));

        UINT64 DefHeapOffset = 0;
        for (int i = 0; i < textures; ++i)
        {
            DefHeapOffset = Align(DefHeapOffset, RAInfo[i].Alignment);

            CheckHr(m_RenderDevice.spDevice->CreatePlacedResource(
                m_RenderDevice.spTextureHeap, DefHeapOffset,
                &RDescs[i],
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(&spTexture2D[i])
                ));

            UINT64 UplHeapSize;
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedTex2D = { 0 };
            m_RenderDevice.spDevice->GetCopyableFootprints(&RDescs[i], 0, 1, 0, &placedTex2D, NULL, NULL, &UplHeapSize);
            
            DefHeapOffset += RAInfo[i].SizeInBytes;

            UINT8* pixels = SuballocateFromUploadHeap(SIZE_T(UplHeapSize), D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
            placedTex2D.Offset += UINT64( pixels-m_RenderDevice.pDataBegin );
            
            CheckHr(spFormatColwerter[i]->CopyPixels(
                NULL,
                placedTex2D.Footprint.RowPitch,
                placedTex2D.Footprint.RowPitch * placedTex2D.Footprint.Height,
                pixels
            ));
            
            CD3DX12_TEXTURE_COPY_LOCATION Dst(spTexture2D[i], 0);
            CD3DX12_TEXTURE_COPY_LOCATION Src(m_RenderDevice.spUploadBuffer, placedTex2D);
            m_RenderDevice.spCommandList->CopyTextureRegion(
                &Dst,
                0, 0, 0,
                &Src,
                NULL
                );

            D3D12_RESOURCE_BARRIER& desc = RBDesc[i];
            desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            desc.Transition.pResource = spTexture2D[i];
            desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            desc.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            desc.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }

        m_RenderDevice.spCommandList->ResourceBarrier(ARRAYSIZE(RBDesc), RBDesc);
        
        m_RenderDevice.spDiffuseTexture = spTexture2D[0];
        m_RenderDevice.spBumpTexture = spTexture2D[1];
        m_RenderDevice.spElwTexture = spTexture2D[2];
      
        m_RenderDevice.spDevice->CreateShaderResourceView(
            m_RenderDevice.spDiffuseTexture,
            NULL,
            m_RenderDevice.OnlineDH.hCPU(sc_DiffuseOnlineHeapOffset));

        m_RenderDevice.spDevice->CreateShaderResourceView(
            m_RenderDevice.spBumpTexture, 
            NULL,
            m_RenderDevice.OnlineDH.hCPU(sc_BumpOnlineHeapOffset));

        m_RenderDevice.spDevice->CreateShaderResourceView(
            m_RenderDevice.spElwTexture,
            NULL,
            m_RenderDevice.OnlineDH.hCPU(sc_ElwOnlineHeapOffset));
    }

private:
    static const UINT           sc_NumVBVs = 2;
    static const UINT           sc_RTVHeapOffset = 0;
    static const UINT           sc_ElwVBVHeapOffset = 0;
    static const UINT           sc_GeomVBVHeapOffset = 1;
    static const UINT           sc_NumCBVs = 1;
    static const UINT           sc_DiffuseOnlineHeapOffset = 0;
    static const UINT           sc_BumpOnlineHeapOffset = 1;
    static const UINT           sc_ElwOnlineHeapOffset = 2;
    static const UINT           sc_CBVHeapOffset = 3;
    static const UINT           sc_NumSRVs = 3;
    static const UINT           sc_NumSamplers = 2;

    // Root signature parameter slots
    static const UINT           sc_SRVTableBindSlot = 0;
    static const UINT           sc_CBTableBindSlot = 1;

    struct
    {
        CComPtr<ID3D12Device>                   spDevice;
        CComPtr<ID3D12CommandQueue>             spCommandQueue;
        CComPtr<ID3D12CommandAllocator>         spCommandAllocator;
        CComPtr<ID3D12CommandAllocator>         spCopyAllocator;
        CComPtr<ID3D12CommandQueue>             spCopyCommandQueue;
        CComPtr<ID3D12CommandAllocator>         spNonRenderingCommandAllocator;

        CComPtr<ID3D12GraphicsCommandList>      spCommandList;
        CComPtr<ID3D12RootSignature>            spRootSignature;
        D3D12_VERTEX_BUFFER_VIEW                VBVs[sc_NumVBVs];
        CDescriptorHeapWrapper                  RTVHeap;
        CDescriptorHeapWrapper                  OnlineDH;

        CComPtr<ID3D12Heap>                     spUploadBufferHeap;
        CComPtr<ID3D12Resource>                 spUploadBuffer;
        UINT8*                                  pDataBegin;
        UINT8*                                  pDataLwr;
        UINT8*                                  pDataEnd;

        CComPtr<ID3D12Resource>                 spRenderTargetResource;
        CComPtr<ID3D12Resource>                 spRenderTargetResolvedResource;
    
        CComPtr<ID3D12Resource>                 spDefaultBuffer;

        CComPtr<ID3D12Heap>                     spTextureHeap;

        CComPtr<ID3D12Resource>                 spDiffuseTexture;
        CComPtr<ID3D12Resource>                 spBumpTexture;
        CComPtr<ID3D12Resource>                 spElwTexture;

        CComPtr<ID3D12PipelineState>            spBackgroundPSO;
        CComPtr<ID3D12PipelineState>            spSpherePSO;

        CComPtr<ID3D12Heap>                     spSharedHeap;
        CComPtr<ID3D12Resource>                 spSharedResource;

        CComPtr<ID3D12Fence>                    spSharedFence;
    } m_RenderDevice; 

    struct
    {
        CComPtr<ID3D12Device>                   spDevice;
        CComPtr<ID3D12CommandQueue>             spCommandQueue;
        CComPtr<ID3D12CommandAllocator>         spCommandAllocator;
        CComPtr<IDXGISwapChain>                 spSwapChain;
        CComPtr<ID3D12Heap>                     spSharedHeap;
        CComPtr<ID3D12Resource>                 spSharedResource;

        CComPtr<ID3D12Fence>                    spSharedFence;
    } m_PresentDevice;

    UINT64 m_fenceValue;

    UINT m_width;
    UINT m_height;
    
    HWND m_window;

    HANDLE m_hEvent;

    HANDLE m_previousFrameCompleteEvent;

    float m_rotateAngle;
};

void 
PumpMessages()
{
    MSG msg;
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

void __cdecl main(
    int argc,
    const char **argv
    )
{
    try
    {
        CoInitialize(NULL);

        g_commandLine.argc = argc;
        g_commandLine.argv = argv;

        // add command line options
        // -lw_intel   : create both lw and intel devices
        // -warp_intel : create warp and intel devices
        int count = argc;
        for (int i = 1; i < argc; ++i)
        {
            if (strcmp(argv[i], "-lw_intel") == 0)
            {
                g_bCreateOurOwnDevice = true;
                g_bCreateWarpRenderDevice = false;
            }

            if (strcmp(argv[i], "-warp_intel") == 0)
            {
                g_bCreateOurOwnDevice = true;
                g_bCreateWarpRenderDevice = true;
            }

            if (strcmp(argv[i], "-copyqueue") == 0)
            {
                g_bUseCopyQueue = true;
            }

            if (strcmp(argv[i], "-useaa" ) == 0)
            {
                g_bUseAA = true;
            }
        }

        // Can't use hw copy queue if warp render device is used.
        if (g_bCreateWarpRenderDevice)
            g_bUseCopyQueue = false;

        App app;

        app.Init();

        while(!g_bClosing)
        {
            PumpMessages();

            app.Render();
        }
    }
    catch (exception& e)
    {
        cout << "Error: " << e.what() << "\n";
    }
}
