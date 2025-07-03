#define NOMINMAX

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
#include <d3d12internaltest.h>
#include <directxmath.h>
#include <wincodec.h>
#include <dxgi1_4.h>
#include <mmsystem.h>

#include "shaders.h"

using namespace DirectX;
using namespace std;

static const D3D12_COMMAND_QUEUE_DESC s_CommandQueueDesc;

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
    _In_opt_ CONST DXGI_SWAP_CHAIN_DESC* pSwapChainDesc,
    _In_ REFIID riidSwapchain,
    _COM_Outptr_opt_ void** ppSwapChain,
    _In_ REFIID riidQueue,
    _COM_Outptr_opt_ void** ppQueue,    
    _In_ REFIID riidDevice,
    _COM_Outptr_opt_ void** ppDevice 
    )
{
    CComPtr<ID3D12Device> pDevice;
    CComPtr<ID3D12CommandQueue> pQueue;
    CComPtr<IDXGIFactory> pDxgiFactory;
    CComPtr<IDXGISwapChain> pDxgiSwapChain;

    CComPtr<IUnknown> pDeviceOut;
    CComPtr<IUnknown> pQueueOut;
    CComPtr<IUnknown> pDxgiSwapChainOut;

    HRESULT hr = D3D12CreateDevice(
        pAdapter,
        MinimumFeatureLevel,
        IID_PPV_ARGS(&pDevice)
        );
    if(FAILED(hr)) { return hr; }

    hr = CreateDXGIFactory1(IID_PPV_ARGS(&pDxgiFactory));
    if(FAILED(hr)) { return hr; }

    hr = pDevice->CreateCommandQueue(&s_CommandQueueDesc, IID_PPV_ARGS(&pQueue));
    if(FAILED(hr)) { return hr; }
    
    DXGI_SWAP_CHAIN_DESC LocalSCD = *pSwapChainDesc;
    hr = pDxgiFactory->CreateSwapChain(
        pQueue,
        &LocalSCD,
        &pDxgiSwapChain
        );
    if(FAILED(hr)) { return hr; }

    hr = pDevice->QueryInterface(riidDevice, reinterpret_cast<void**>(&pDeviceOut));
    if(FAILED(hr)) { return hr; }

    hr = pQueue->QueryInterface(riidQueue, reinterpret_cast<void**>(&pQueueOut));
    if(FAILED(hr)) { return hr; }

    hr = pDxgiSwapChain->QueryInterface(riidSwapchain, reinterpret_cast<void**>(&pDxgiSwapChainOut));
    if(FAILED(hr)) { return hr; }

    *ppDevice = pDeviceOut.Detach();
    *ppQueue = pQueueOut.Detach();
    *ppSwapChain = pDxgiSwapChain.Detach();

    return S_OK;
}


#define NUM_SPHERE_RINGS    15
#define NUM_SPHERE_SEGMENTS 30
const int NUM_VERTICES        = 2 * NUM_SPHERE_RINGS * ( NUM_SPHERE_SEGMENTS + 1 );

const GUID WKPDID_D3DDebugObjectNameW = { 0x4cca5fd8, 0x921f, 0x42c8, 0x85, 0x66, 0x70, 0xca, 0xf2, 0xa9, 0xb7, 0x41 };

struct GraphicsConstantBufferContents
{
    XMMATRIX worldViewProjection;    // World * View * Projection matrix
    XMMATRIX ilwWorldViewProjection;
    XMMATRIX world;
    XMMATRIX worldView;
    XMVECTOR eyePt;
};

struct ComputeConstantBufferContents
{
    XMMATRIX backgroundMatrix;
    float backgroundZoom;
};

struct WholeResourceTransitionBarrier : public D3D12_RESOURCE_BARRIER
{
    WholeResourceTransitionBarrier(ID3D12Resource *pR, D3D12_RESOURCE_STATES Before, D3D12_RESOURCE_STATES After)
    {
        Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        Transition.pResource = pR;
        Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        Transition.StateBefore = Before;
        Transition.StateAfter = After;      
        Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    }
};

struct IlwerseResourceTransitionBarrier : public D3D12_RESOURCE_BARRIER
{
    IlwerseResourceTransitionBarrier(const D3D12_RESOURCE_BARRIER &in)
    {
        assert(in.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION);

        Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        Transition.pResource = in.Transition.pResource;
        Transition.Subresource = in.Transition.Subresource;
        Transition.StateBefore = in.Transition.StateAfter;
        Transition.StateAfter = in.Transition.StateBefore;
        Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    }
};

struct
{
    int          argc;
    const char** argv;
} g_commandLine;

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

LRESULT CALLBACK MyWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_CLOSE:
        TerminateProcess(GetLwrrentProcess(), -1);
        return 0;

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
}

class App
{
public:
    App()
        : m_rotateAngle(0.0f),
        m_backgroundAngle(0.0f),
        m_backgroundZoom(1.0f),
        m_fenceValue(0)
    {
    }

    void Init()
    {
        m_Benchmark = (0 != atoi(GetCommandLine("benchmark", "0")));
        if (m_Benchmark)
        {
            timeBeginPeriod(1);
        }
        m_StartTime = timeGetTime();

        const UINT windowSize = m_Benchmark ? 320 : 768;

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
        wcex.lpszClassName  = _T("BlakeWindow");

        RegisterClassEx(&wcex);

        m_window = CreateWindow(
                _T("BlakeWindow"),
                m_Benchmark ? _T("D3D12Test - Benchmark Mode") : _T("D3D12Test"),
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

        bool fullScreen = (0 != atoi(GetCommandLine("Fullscreen", "0")));
        bool proxy = (0 != atoi(GetCommandLine("Proxy", "0")));

        DXGI_SWAP_CHAIN_DESC scd;
        ZeroMemory(&scd, sizeof(scd));

        scd.BufferDesc.Width = fullScreen ? 0 : width;
        scd.BufferDesc.Height = fullScreen ? 0 : height;
        scd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;

        scd.SampleDesc.Count = 1;
        scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scd.BufferCount = 2;
        scd.OutputWindow = m_window;
        scd.Windowed = !fullScreen;
#if 0
        scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
#else
        scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
#endif
        CComPtr<IDXGIAdapter> spAdapter;
        if (0 != atoi(GetCommandLine("Warp", "0")))
        {
            CComPtr<IDXGIFactory4> spFactory;
            CheckHr(CreateDXGIFactory2(0, IID_PPV_ARGS(&spFactory)));
            CheckHr(spFactory->EnumWarpAdapter(IID_PPV_ARGS(&spAdapter)));
        }
        else
        {
            int Adapter = 0;
            if (0 != (Adapter = atoi(GetCommandLine("Adapter", "0"))))
            {
                CComPtr<IDXGIFactory4> spFactory;
                CheckHr(CreateDXGIFactory2(0, IID_PPV_ARGS(&spFactory)));
                CheckHr(spFactory->EnumAdapters(Adapter, &spAdapter));
            }
        }

        CheckHr(D3D12CreateDeviceHelper(
            spAdapter,
            D3D_FEATURE_LEVEL_11_0,
            &scd,
            IID_PPV_ARGS(&m_spSwapChain),
            IID_PPV_ARGS(&m_spCommandQueue),
            IID_PPV_ARGS(&m_spDevice12)
            ));

        {
            ATL::CComPtr< ID3D12DeviceTest > spDeviceTest;
            CheckHr(m_spDevice12->QueryInterface(&spDeviceTest));
            
            (void)spDeviceTest->CheckTestFeatureSupport(
                D3D12TEST_FEATURE_MEMORY_ARCHITECTURE,
                &m_TestMemArch,
                sizeof( m_TestMemArch ) );
        }

        if (fullScreen && proxy)
        {
            // Force proxy mode
            m_spSwapChain->ResizeBuffers(2, 800, 800, DXGI_FORMAT_UNKNOWN, 0 );
        }

        CheckHr(m_RTVHeap.Create(m_spDevice12,D3D12_DESCRIPTOR_HEAP_TYPE_RTV,1));
        CheckHr(m_OnlineSamplerDH.Create(m_spDevice12,D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,sc_NumSamplers,true));
        CheckHr(m_OnlineDH.Create(m_spDevice12,D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,sc_NumSRVsTotal+sc_NumCBVsTotal+sc_NumUAVs,true));


        CheckHr(m_spDevice12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_spFence)));
        if (GetComputeCommandListType() == D3D12_COMMAND_LIST_TYPE_COMPUTE)
        {
            D3D12_COMMAND_QUEUE_DESC CQDesc = {D3D12_COMMAND_LIST_TYPE_COMPUTE};
            m_spDevice12->CreateCommandQueue(&CQDesc, IID_PPV_ARGS(&m_spComputeQueue));
        }
        else
        {
            D3D12_COMMAND_QUEUE_DESC CQDesc = {D3D12_COMMAND_LIST_TYPE_DIRECT};
            m_spDevice12->CreateCommandQueue(&CQDesc, IID_PPV_ARGS(&m_spComputeQueue));
        }

        m_spDevice12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_spComputeFence));
        m_hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        // In full-screen mode, we let the swap chain choose the size
        // Query it now
        DXGI_SWAP_CHAIN_DESC newDesc;
        CheckHr(m_spSwapChain->GetDesc(&newDesc));

        m_width = newDesc.BufferDesc.Width;
        m_height = newDesc.BufferDesc.Height;

        CheckHr(m_spDevice12->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&m_spCommandAllocator)
            ));

        for (unsigned i = 0; i < sc_NumComputeBuffers; ++i)
        {
            CheckHr(m_spDevice12->CreateCommandAllocator(GetComputeCommandListType(), IID_PPV_ARGS(&m_spComputeAllocators[i])));
        }

        {
            CheckHr(m_spDevice12->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(8*1024*1024),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&m_spUploadBuffer)
                ));

            CheckHr(m_spUploadBuffer->Map(0, NULL,reinterpret_cast<void**>(&m_pDataBegin)));
            m_pDataLwr = m_pDataBegin;
            m_pDataEnd = m_pDataBegin + m_spUploadBuffer->GetDesc().Width;
        }

        {
            CheckHr(m_spDevice12->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(GetAlignedConstantBufferSize()*sc_NumComputeBuffers),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&m_spConstantBuffer)
                ));

            CheckHr(m_spConstantBuffer->Map(0, NULL,reinterpret_cast<void**>(&m_pConstantBufferBegin)));
        }


        for (unsigned i = 0; i < sc_NumComputeBuffers; ++i)
        {
            D3D12_CONSTANT_BUFFER_VIEW_DESC CBDesc = { 
                m_spConstantBuffer->GetGPUVirtualAddress() + GetAlignedConstantBufferSize() * i,
                (UINT)GetAlignedConstantBufferSize() };

            m_spDevice12->CreateConstantBufferView(
                &CBDesc,
                m_OnlineDH.hCPU(sc_CBVHeapOffset+i)
                );
        }

        {
            CheckHr(m_spDevice12->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(NUM_VERTICES*sizeof(GEOMETRY_VERTEX)+4*sizeof(ELW_VERTEX)),
                m_TestMemArch.DriverSupportedVer > 0 ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_PPV_ARGS(&m_spDefaultBuffer)
                ));
        }

        m_VBVs[sc_ElwVBVHeapOffset].BufferLocation = m_spDefaultBuffer->GetGPUVirtualAddress() + NUM_VERTICES*sizeof(GEOMETRY_VERTEX);
        m_VBVs[sc_ElwVBVHeapOffset].SizeInBytes = 4 * sizeof(ELW_VERTEX);
        m_VBVs[sc_ElwVBVHeapOffset].StrideInBytes = sizeof(ELW_VERTEX);

        m_VBVs[sc_GeomVBVHeapOffset].BufferLocation = m_spDefaultBuffer->GetGPUVirtualAddress();
        m_VBVs[sc_GeomVBVHeapOffset].SizeInBytes = NUM_VERTICES*sizeof(GEOMETRY_VERTEX);
        m_VBVs[sc_GeomVBVHeapOffset].StrideInBytes = sizeof(GEOMETRY_VERTEX);

        // Root signature
        {
            CD3DX12_DESCRIPTOR_RANGE DescRange[4];
            DescRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV,3,0); // t0-t2
            DescRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,2,0); // s0-s1
            DescRange[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV,1,0); // b0
            DescRange[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV,1,0); // u0

            CD3DX12_ROOT_PARAMETER RP[4];

            RP[sc_SRVTableBindSlot].InitAsDescriptorTable(1,&DescRange[0],D3D12_SHADER_VISIBILITY_PIXEL); // t0-t2
            RP[sc_SamplerTableBindSlot].InitAsDescriptorTable(1,&DescRange[1],D3D12_SHADER_VISIBILITY_PIXEL); // s0
            RP[sc_CBTableBindSlot].InitAsDescriptorTable(1,&DescRange[2],D3D12_SHADER_VISIBILITY_ALL); // b0
            RP[sc_UAVTableBindSlot].InitAsDescriptorTable(1,&DescRange[3],D3D12_SHADER_VISIBILITY_ALL); // u0

            CD3DX12_ROOT_SIGNATURE_DESC RTLayout(4,RP,0,NULL,D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
            CComPtr<ID3DBlob> pSerializedLayout;
            D3D12SerializeRootSignature(&RTLayout, D3D_ROOT_SIGNATURE_VERSION_1, &pSerializedLayout,NULL);

            CheckHr(m_spDevice12->CreateRootSignature(
                1,
                pSerializedLayout->GetBufferPointer(), 
                pSerializedLayout->GetBufferSize(),
                __uuidof(ID3D12RootSignature),
                (void**)&m_spRootSignature));
        }

        CheckHr(m_spDevice12->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS((&m_spNonRenderingCommandAllocator.p))));

        CheckHr(m_spDevice12->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT, 
            m_spNonRenderingCommandAllocator,
            nullptr,
            IID_PPV_ARGS(&m_spCommandList)));

        UINT VerticesOffset = 0;
        {
            UINT8* pVertices = SuballocateFromUploadHeap(m_spDefaultBuffer->GetDesc().Width);
            VerticesOffset = (UINT)(pVertices-m_pDataBegin);

            CreateSphereVertices(reinterpret_cast<GEOMETRY_VERTEX*>(pVertices));
            
            CreateElwVertices(reinterpret_cast<ELW_VERTEX*>(pVertices+NUM_VERTICES*sizeof(GEOMETRY_VERTEX)));

            D3D12_RESOURCE_BARRIER desc;
            ZeroMemory( &desc, sizeof( desc ) );
            if (m_TestMemArch.DriverSupportedVer == 0)
            {
                desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                desc.Transition.pResource = m_spDefaultBuffer;
                desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                desc.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                desc.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;

                m_spCommandList->ResourceBarrier(1, &desc);
            }

            m_spCommandList->CopyBufferRegion(
                m_spDefaultBuffer,
                0,
                m_spUploadBuffer,
                VerticesOffset,
                m_spDefaultBuffer->GetDesc().Width
                );

            LoadTextures();

            desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            desc.Transition.pResource = m_spDefaultBuffer;
            desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            desc.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            desc.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
            
            m_spCommandList->ResourceBarrier(1, &desc);
        }

        for(UINT i = 0; i < sc_NumSamplers; i++)
        {
            D3D12_SAMPLER_DESC SamplerDesc;
            SamplerDesc.Filter = D3D12_FILTER_ANISOTROPIC;
            SamplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            SamplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            SamplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            SamplerDesc.MipLODBias = 0;
            SamplerDesc.MaxAnisotropy = 16;
            SamplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            SamplerDesc.BorderColor[0] = 0.0f;
            SamplerDesc.BorderColor[1] = 0.0f;
            SamplerDesc.BorderColor[2] = 0.0f;
            SamplerDesc.BorderColor[3] = 0.0f;
            SamplerDesc.MinLOD = 0.0f;
            SamplerDesc.MaxLOD = 9999.0f;
            m_spDevice12->CreateSampler(&SamplerDesc,m_OnlineSamplerDH.hCPU(sc_Stage0SamplerHeapOffset+i));

        }

        CheckHr(m_spCommandList->Close());

        m_spCommandQueue->SetMarker(0, L"Setting marker", sizeof(L"Setting marker"));
        m_spCommandQueue->ExelwteCommandLists(1, CommandListCast(&m_spCommandList));

        //CheckHr(m_spNonRenderingCommandAllocator->Reset());
        CheckHr(m_spCommandList->Reset(m_spNonRenderingCommandAllocator, nullptr));

        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc;
            ZeroMemory(&psoDesc, sizeof(psoDesc));
            psoDesc.pRootSignature = m_spRootSignature;
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
   
            CheckHr(m_spDevice12->CreateGraphicsPipelineState(
                &psoDesc,
                IID_PPV_ARGS(&m_spBackgroundPSO)
                ));
        }

        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc;
            ZeroMemory(&psoDesc, sizeof(psoDesc));
            psoDesc.pRootSignature = m_spRootSignature;
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
   
            CheckHr(m_spDevice12->CreateGraphicsPipelineState(
                &psoDesc,
                IID_PPV_ARGS(&m_spSpherePSO)
                ));
        }

        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc;
            ZeroMemory(&psoDesc, sizeof(psoDesc));
            psoDesc.pRootSignature = m_spRootSignature;            
            psoDesc.CS = { g_BackgroundCS, sizeof(g_BackgroundCS) };

            CheckHr(m_spDevice12->CreateComputePipelineState(
                &psoDesc,
                IID_PPV_ARGS(&m_spComputePSO)
                ));            
        }
    }

    CComPtr<ID3D12GraphicsCommandList> RecordComputeCommandList(UINT Index)
    {
        auto pElwTexture = m_spElwTexture[Index];
        CComPtr<ID3D12GraphicsCommandList> spCommandList;
        m_spComputeAllocators[Index]->Reset();
        CheckHr(m_spDevice12->CreateCommandList(0, GetComputeCommandListType(), m_spComputeAllocators[Index], m_spComputePSO,
            IID_PPV_ARGS(&spCommandList)));

        ID3D12DescriptorHeap* pDH[2] = {m_OnlineDH, m_OnlineSamplerDH};
        spCommandList->SetDescriptorHeaps(2,pDH);

        // When the SHARED state is added, this is intended to be used every frame
        // For now, it's only applied once, since the resource remains in the UAV state on this queue
        static bool bInUAVState = false;
        auto BarrierInitialUnordered = WholeResourceTransitionBarrier(pElwTexture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        spCommandList->SetComputeRootSignature(m_spRootSignature);

        spCommandList->SetComputeRootDescriptorTable(sc_UAVTableBindSlot, m_OnlineDH.hGPU(sc_UAVHeapOffset + Index));
        spCommandList->SetComputeRootDescriptorTable(sc_CBTableBindSlot, m_OnlineDH.hGPU(sc_CBVHeapOffset + Index));

        if (!bInUAVState)
        {
            spCommandList->ResourceBarrier(1, &BarrierInitialUnordered);
            bInUAVState = true;
        }

        for (unsigned i = 0; i < sc_ComputeRepeat; ++i)
        {
            spCommandList->Dispatch( sc_ElwTextureSize / sc_ComputeThreadsX, sc_ElwTextureSize / sc_ComputeThreadsY, 1 );
        }

#if EXTRA_UAV_BARRIER
        D3D12_RESOURCE_BARRIER barrier = { D3D12_RESOURCE_BARRIER_TYPE_UAV };
        barrier.UAV.pResource = pElwTexture;
        spCommandList->ResourceBarrier(1, &barrier);
#endif

        return spCommandList;
    }

    CComPtr<ID3D12GraphicsCommandList> RecordCommandList(UINT BackgroundIndex, UINT backBufferIndex)
    {
        auto pElwTexture = m_spElwTexture[BackgroundIndex];

        CComPtr<ID3D12GraphicsCommandList> spCommandList;
        CheckHr(m_spDevice12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_spCommandAllocator, m_spBackgroundPSO, IID_PPV_ARGS(&spCommandList)));

        CComPtr<ID3D12Resource>         spBackBuffer;
        CheckHr(m_spSwapChain->GetBuffer(backBufferIndex, IID_PPV_ARGS(&spBackBuffer)));
        m_spDevice12->CreateRenderTargetView(spBackBuffer, NULL, m_RTVHeap.hCPU(0));

        ID3D12DescriptorHeap* pDH[2] = {m_OnlineDH, m_OnlineSamplerDH};
        spCommandList->SetDescriptorHeaps(2,pDH);
        spCommandList->SetGraphicsRootSignature(m_spRootSignature);

        //
        // Resource barriers to transition all resources from initial state to approriate usage
        //
        const struct
        {
            ID3D12Resource* pResource;
            D3D12_RESOURCE_STATES stateBefore;
            D3D12_RESOURCE_STATES stateAfter;
        } barriers[] = 
        {
            { spBackBuffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET },
        };

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
                (UINT)barrierArray.size(),
                &(barrierArray[0])
                );
        }

        float clearColor[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
        spCommandList->BeginEvent(0, L"Clear", sizeof(L"Clear"));
        spCommandList->ClearRenderTargetView(
            m_RTVHeap.hCPU(0),
            clearColor,
            NULL,
            0
            );
        spCommandList->EndEvent();

        spCommandList->OMSetRenderTargets(1,&m_RTVHeap.hCPU(0),true,NULL);

        D3D12_VIEWPORT viewport = 
        {
            0.0f,
            0.0f, 
            static_cast<float>(m_width), 
            static_cast<float>(m_height), 
            0.0f, 
            1.0f 
        };

        D3D_SET_OBJECT_NAME_W(spCommandList, L"Commandlist");


        spCommandList->RSSetViewports(1, &viewport);

        D3D12_RECT scissorRect = {0, 0, m_width, m_height};
        spCommandList->RSSetScissorRects(1, &scissorRect);

        spCommandList->SetGraphicsRootDescriptorTable( sc_SamplerTableBindSlot, m_OnlineSamplerDH.hGPU(sc_Stage0SamplerHeapOffset) );

        spCommandList->SetGraphicsRootDescriptorTable( sc_CBTableBindSlot, m_OnlineDH.hGPU(sc_CBVHeapOffset+BackgroundIndex) );

        //
        // Render the background
        //
        {
            UINT offsets[] = { 0 };
            UINT strides[] = { sizeof(ELW_VERTEX) };
            
            spCommandList->SetPipelineState(m_spBackgroundPSO);

            spCommandList->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );

            spCommandList->IASetVertexBuffers(0,1,&m_VBVs[sc_ElwVBVHeapOffset]);

            spCommandList->SetGraphicsRootDescriptorTable(sc_SRVTableBindSlot, m_OnlineDH.hGPU(sc_ElwOnlineHeapOffset+sc_NumSRVsPerFrame*BackgroundIndex));

            spCommandList->DrawInstanced( 4, 1, 0, 0 );
        }

        //
        // Render the sphere
        //
        {
            spCommandList->SetPipelineState(m_spSpherePSO);

            spCommandList->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );

            spCommandList->IASetVertexBuffers(0,1,&m_VBVs[sc_GeomVBVHeapOffset]);

            spCommandList->SetGraphicsRootDescriptorTable(sc_SRVTableBindSlot, m_OnlineDH.hGPU(sc_DiffuseOnlineHeapOffset+sc_NumSRVsPerFrame*BackgroundIndex)); // all srvs

            spCommandList->DrawInstanced( NUM_VERTICES - 1, 1, 0, 0 );
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
                (UINT)barrierArray.size(),
                &(barrierArray[0])
                );
        }

        return spCommandList;
    }

    static SIZE_T Align(SIZE_T uLocation, UINT32 uAlign=MEMORY_ALLOCATION_ALIGNMENT)
    {
        return (uLocation + ((SIZE_T)uAlign-1)) & ~((SIZE_T)uAlign-1);
    }

    void WaitForGPU()
    {
        CheckHr(m_spCommandQueue->Signal(m_spFence, ++m_fenceValue));
        CheckHr(m_spFence->SetEventOnCompletion(m_fenceValue, m_hEvent));
        WaitForSingleObject(m_hEvent, INFINITE);
    }

    void FlushAndFinish()
    {
        CheckHr(m_spCommandList->Close());
        m_spCommandQueue->ExelwteCommandLists(1, CommandListCast(&m_spCommandList.p));
        
        WaitForGPU();

        CheckHr(m_spNonRenderingCommandAllocator->Reset());
        CheckHr(m_spCommandList->Reset(m_spNonRenderingCommandAllocator, nullptr));
    }

    UINT8* SuballocateFromUploadHeap(SIZE_T uSize, UINT32 uAlign=MEMORY_ALLOCATION_ALIGNMENT)
    {
        CheckHr(uSize<(SIZE_T)(m_pDataEnd-m_pDataBegin)?S_OK:E_OUTOFMEMORY);

        m_pDataLwr = reinterpret_cast<UINT8*>(Align(reinterpret_cast<SIZE_T>(m_pDataLwr), uAlign));
        if (m_pDataLwr >= m_pDataEnd || m_pDataLwr + uSize >= m_pDataEnd)
        {
            FlushAndFinish();
            m_pDataLwr = m_pDataBegin;
        }

        UINT8* pRet = m_pDataLwr;
        m_pDataLwr += uSize;
        return pRet;
    }

    size_t GetAlignedConstantBufferSize()
    {
        static_assert(sizeof(GraphicsConstantBufferContents) >= sizeof(ComputeConstantBufferContents), "Constant buffer size ilwariant");
        return Align(sizeof(GraphicsConstantBufferContents),D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    }

    void UpdateComputeConstantBuffer(UINT Index)
    {
        // Increment animation values
        m_backgroundAngle += 0.01f;
        m_backgroundZoom += 0.01f;

        // Compute constant buffer contents for compute        
        ComputeConstantBufferContents cbContents;
        cbContents.backgroundMatrix = XMMatrixRotationZ(m_backgroundAngle*2);
        cbContents.backgroundZoom = 1.0f + fmod(m_backgroundZoom,1.0);

        UINT8* pData = m_pConstantBufferBegin + Index * GetAlignedConstantBufferSize();
        
        memcpy(pData, &cbContents, sizeof(cbContents));
    }

    void UpdateGraphicsConstantBuffer(UINT Index)
    {
        // Increment animation values
        m_rotateAngle += 0.01f;

        // Compute constant buffer contents for graphics
        XMMATRIX world = XMMatrixRotationY(m_rotateAngle);

        XMVECTOR eye = XMVectorSet( 0.0f, 0.0f, -2.0f, 0.0f );
        XMVECTOR at = XMVectorSet( 0.0f, 0.0f, 0.0f, 0.0f );
        XMVECTOR up = XMVectorSet( 0.0f, 1.0f, 0.0f, 0.0f );
        XMMATRIX view = XMMatrixLookAtLH( eye, at, up );

        XMMATRIX projection = XMMatrixPerspectiveFovLH( XM_PIDIV2, m_width / (FLOAT)m_height, 0.01f, 100.0f );

        XMMATRIX wvp = world * view * projection;
        XMMATRIX ilwwvp = XMMatrixIlwerse(NULL, wvp);

        GraphicsConstantBufferContents cbContents;

        cbContents.worldViewProjection = XMMatrixTranspose(wvp);
        cbContents.ilwWorldViewProjection = XMMatrixTranspose(ilwwvp);
        cbContents.world = XMMatrixTranspose(world);
        cbContents.worldView = XMMatrixTranspose(world * view);
        cbContents.eyePt = eye;

        UINT8* pData = m_pConstantBufferBegin + Index * GetAlignedConstantBufferSize();
        
        memcpy(pData, &cbContents, sizeof(cbContents));
    }

    void SubmitComputeFrame(UINT64 FrameCount)
    {
        auto BufferIndex = FrameCount % sc_NumComputeBuffers;
        UpdateComputeConstantBuffer(BufferIndex);

        CComPtr<ID3D12GraphicsCommandList> spCommandListCompute = RecordComputeCommandList(BufferIndex);
        CheckHr(spCommandListCompute->Close());

        m_spComputeQueue->ExelwteCommandLists(1, CommandListCast(&spCommandListCompute));
        m_spComputeQueue->Signal(m_spComputeFence, FrameCount+1);
    }

    void Render()
    {
        static UINT64 ComputeFramesSubmitted = 0;

        //
        // Wait for the previous frame
        //
        WaitForSingleObject(m_previousFrameCompleteEvent, INFINITE);

        if (!ComputeFramesSubmitted)
        {
            for (unsigned i = 0; i < sc_NumComputeBuffers; ++i)
            {
                SubmitComputeFrame(i);
            }
            ComputeFramesSubmitted = sc_NumComputeBuffers;
        }

        auto ComputeFramesCompleted = m_spComputeFence->GetCompletedValue();

        if (!ComputeFramesCompleted)
        {
            m_spComputeFence->SetEventOnCompletion(1, m_previousFrameCompleteEvent);
            return;
        }

        while(ComputeFramesSubmitted + 1 < ComputeFramesCompleted + sc_NumComputeBuffers)
        {
            SubmitComputeFrame(ComputeFramesSubmitted);
            ++ComputeFramesSubmitted;
        }

        auto ComputeBufferCompleted = (ComputeFramesCompleted-1) % sc_NumComputeBuffers;

        //
        // Record a new command list
        //
        CComPtr<IDXGISwapChain3> spSwapChain3;
        CheckHr(m_spSwapChain->QueryInterface(&spSwapChain3));
        UINT backBufferIndex = spSwapChain3->GetLwrrentBackBufferIndex();

        //
        // Reset backing store for command allocator
        //
        CheckHr(m_spCommandAllocator->Reset());

        CComPtr<ID3D12GraphicsCommandList> spCommandList = RecordCommandList(ComputeBufferCompleted, backBufferIndex);


        CheckHr(spCommandList->Close());
        UpdateGraphicsConstantBuffer(ComputeBufferCompleted);

        m_spCommandQueue->ExelwteCommandLists(1, CommandListCast(&spCommandList));

        HRESULT hr = m_spSwapChain->Present(m_Benchmark ? 0 : 1, 0);

        // Present can fail after fs<->windowed transition without a call to ResizeBuffers
        if( FAILED(hr) )
        {
            // Can't resize before the queued command lists are finished exelwting on the GPU (PTE fault, TDR and Crash otherwise).
            WaitForGPU();

            m_spSwapChain->ResizeBuffers(2, 0, 0, DXGI_FORMAT_UNKNOWN, 0);

            DXGI_SWAP_CHAIN_DESC newDesc;
            CheckHr(m_spSwapChain->GetDesc(&newDesc));

            m_width = newDesc.BufferDesc.Width;
            m_height = newDesc.BufferDesc.Height;
        }

        CheckHr(m_spCommandQueue->Signal(m_spFence, ++m_fenceValue));
        CheckHr(m_spFence->SetEventOnCompletion(m_fenceValue, m_previousFrameCompleteEvent));


        // Front-buffer synchronization is not yet implemented
        // So wait for vblank to ensure that flips do not get queued
        // Note that this will fail on a render-only adapter (like WARP)
        CComPtr<IDXGIOutput> spOutput;
        if (!m_Benchmark && SUCCEEDED( m_spSwapChain->GetContainingOutput(&spOutput) ) )
        {
            CheckHr(spOutput->WaitForVBlank());
        }

        // In 'benchmark mode' exit the application after 5 seconds with an ExitCode equalt to the number of frames rendered
        m_Frames++;
        DWORD duration = timeGetTime() - m_StartTime;
        if (m_Benchmark && duration > 5000)
        {
            ExitProcess(m_Frames);
        }
    }

    void LoadTextures()
    {
        const unsigned c_ImageTextures = 2;
        const unsigned c_TotalTextures = c_ImageTextures + sc_NumComputeBuffers;

        const WCHAR* names[c_ImageTextures] = { _T("EARTH.BMP"), _T("EARTHBUMP.BMP") };

        CComPtr<IWICBitmapFrameDecode> spFrameDecode[c_ImageTextures];
        CComPtr<IWICFormatColwerter> spFormatColwerter[c_ImageTextures];
        CD3DX12_RESOURCE_DESC RDescs[c_TotalTextures];
        const D3D12_RESOURCE_DESC* pRDescs = reinterpret_cast<const D3D12_RESOURCE_DESC*>(RDescs);
        D3D12_RESOURCE_ALLOCATION_INFO RAInfo[c_TotalTextures];
        CComPtr<ID3D12Resource> spTexture2D[c_TotalTextures];
        D3D12_RESOURCE_BARRIER RBDesc[c_TotalTextures];
        ZeroMemory(&RBDesc, sizeof(RBDesc));

        CComPtr<IWICImagingFactory> spWicFactory;
        CheckHr(CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, __uuidof(spWicFactory), (void**)&spWicFactory));

        for (int i = 0; i < c_ImageTextures; ++i)
        {

            CComPtr<IWICBitmapDecoder> spBitmapDecoder;
            CheckHr(spWicFactory->CreateDecoderFromFilename(
                names[i],
                NULL,
                GENERIC_READ,
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
            RAInfo[i] = m_spDevice12->GetResourceAllocationInfo(1, 1, &RDescs[i]);
        }

        for (int i = c_ImageTextures; i < c_TotalTextures; ++i)
        {
            RDescs[i] = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, sc_ElwTextureSize, sc_ElwTextureSize, 1, 1);
            RDescs[i].Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }

        if (m_TestMemArch.DriverSupportedVer > 0)
        {
            CheckHr(m_spDevice12->CreateHeap(
                &CD3DX12_HEAP_DESC(
                m_spDevice12->GetResourceAllocationInfo(1, c_TotalTextures, RDescs),
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES),
                IID_PPV_ARGS(&m_spTextureHeap)
                ));
        }

        UINT64 DefHeapOffset = 0;
        for (int i = 0; i < c_TotalTextures; ++i)
        {
            auto RAInfo = m_spDevice12->GetResourceAllocationInfo(1, 1, &RDescs[i]);

            DefHeapOffset = Align(DefHeapOffset, RAInfo.Alignment);

            if (m_TestMemArch.DriverSupportedVer == 0)
            {
                CheckHr(m_spDevice12->CreateCommittedResource(
                    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                    D3D12_HEAP_FLAG_NONE,
                    &RDescs[i],
                    D3D12_RESOURCE_STATE_COMMON,
                    nullptr,
                    IID_PPV_ARGS(&spTexture2D[i])));

                D3D12_RESOURCE_BARRIER desc0;
                ZeroMemory(&desc0, sizeof(desc0));
                desc0.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                desc0.Transition.pResource = spTexture2D[i];
                desc0.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                desc0.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                desc0.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;

                m_spCommandList->ResourceBarrier(1, &desc0);
            }
            else
            {
                CheckHr(m_spDevice12->CreatePlacedResource(
                    m_spTextureHeap, DefHeapOffset,
                    &RDescs[i],
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    nullptr,
                    IID_PPV_ARGS(&spTexture2D[i])
                    ));
            }

            if (i < c_ImageTextures)
            {
                UINT64 UplHeapSize;
                D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedTex2D = { 0 };
                m_spDevice12->GetCopyableFootprints(&RDescs[i], 0, 1, 0, &placedTex2D, NULL, NULL, &UplHeapSize);

                DefHeapOffset += RAInfo.SizeInBytes;

                UINT8* pixels = SuballocateFromUploadHeap(SIZE_T(UplHeapSize), D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
                placedTex2D.Offset += UINT64(pixels - m_pDataBegin);

                CheckHr(spFormatColwerter[i]->CopyPixels(
                    NULL,
                    placedTex2D.Footprint.RowPitch,
                    placedTex2D.Footprint.RowPitch * placedTex2D.Footprint.Height,
                    pixels
                    ));

                CD3DX12_TEXTURE_COPY_LOCATION Dst(spTexture2D[i], 0);
                CD3DX12_TEXTURE_COPY_LOCATION Src(m_spUploadBuffer, placedTex2D);
                m_spCommandList->CopyTextureRegion(
                    &Dst,
                    0, 0, 0,
                    &Src,
                    NULL
                    );
            }

            D3D12_RESOURCE_BARRIER& desc = RBDesc[i];
            desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            desc.Transition.pResource = spTexture2D[i];
            desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            desc.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            desc.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        }

        m_spCommandList->ResourceBarrier(ARRAYSIZE(RBDesc), RBDesc);

        m_spDiffuseTexture = spTexture2D[0];
        m_spBumpTexture = spTexture2D[1];
        m_spDiffuseTexture->SetName(L"DiffuseTexture");
        m_spBumpTexture->SetName(L"BumpTexture");
        for (int i = 0; i < sc_NumComputeBuffers; ++i)
        {
            m_spElwTexture[i] = spTexture2D[2 + i];

            auto srvBankOffset = i*sc_NumSRVsPerFrame;

            m_spDevice12->CreateShaderResourceView(
                m_spDiffuseTexture,
                NULL,
                m_OnlineDH.hCPU(sc_DiffuseOnlineHeapOffset + srvBankOffset));

            m_spDevice12->CreateShaderResourceView(
                m_spBumpTexture,
                NULL,
                m_OnlineDH.hCPU(sc_BumpOnlineHeapOffset + srvBankOffset));

            m_spDevice12->CreateShaderResourceView(
                m_spElwTexture[i],
                NULL,
                m_OnlineDH.hCPU(sc_ElwOnlineHeapOffset + srvBankOffset));

            D3D12_UNORDERED_ACCESS_VIEW_DESC UAViewDesc = {};
            ZeroMemory(&UAViewDesc, sizeof(UAViewDesc));

            UAViewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            UAViewDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

            m_spDevice12->CreateUnorderedAccessView(
                m_spElwTexture[i],
                nullptr,
                &UAViewDesc,
                m_OnlineDH.hCPU(sc_UAVHeapOffset + i));
        }
    }

private:
    static const UINT sc_NumComputeBuffers = 3;
    static const UINT sc_ElwTextureSize = 1024;

    // Increase this to artificially slow the compute work down
    static const UINT sc_ComputeRepeat = 2;

    // These need to be kept in sync with the compute shader
    static const UINT sc_ComputeThreadsX = 4;
    static const UINT sc_ComputeThreadsY = 64;

    CComPtr<ID3D12CommandAllocator> m_spCommandAllocator;
    CComPtr<ID3D12CommandAllocator> m_spComputeAllocators[sc_NumComputeBuffers];
    CComPtr<ID3D12CommandAllocator> m_spNonRenderingCommandAllocator;

    CComPtr<IDXGISwapChain>      m_spSwapChain;

    CComPtr<ID3D12Device>        m_spDevice12;
    D3D12TEST_FEATURE_DATA_MEMORY_ARCHITECTURE m_TestMemArch;

    CComPtr<ID3D12CommandQueue>  m_spCommandQueue;
    CComPtr<ID3D12CommandQueue>  m_spComputeQueue;
    CComPtr<ID3D12Fence>         m_spComputeFence;
    CComPtr<ID3D12Fence>         m_spFence;
    UINT64                       m_fenceValue;
    HANDLE                       m_hEvent;

    CComPtr<ID3D12GraphicsCommandList>   m_spCommandList;

    CComPtr<ID3D12RootSignature>  m_spRootSignature;

    static const UINT          sc_NumVBVs = 2;
    D3D12_VERTEX_BUFFER_VIEW   m_VBVs[sc_NumVBVs];

    CDescriptorHeapWrapper          m_RTVHeap;
    CDescriptorHeapWrapper          m_OnlineSamplerDH;
    CDescriptorHeapWrapper          m_OnlineDH;

    CComPtr<ID3D12Resource>         m_spUploadBuffer;
    UINT8*                          m_pDataBegin;
    UINT8*                          m_pDataLwr;
    UINT8*                          m_pDataEnd;

    CComPtr<ID3D12Resource>         m_spConstantBuffer;
    UINT8*                          m_pConstantBufferBegin;
    
    CComPtr<ID3D12Resource>         m_spDefaultBuffer;

    CComPtr<ID3D12Heap>               m_spTextureHeap;

    static const UINT           sc_RTVHeapOffset = 0;
    static const UINT           sc_ElwVBVHeapOffset = 0;
    static const UINT           sc_GeomVBVHeapOffset = 1;
    CComPtr<ID3D12Resource>     m_spDiffuseTexture;
    CComPtr<ID3D12Resource>     m_spBumpTexture;
    CComPtr<ID3D12Resource>     m_spElwTexture[sc_NumComputeBuffers];
    
    static const UINT           sc_DiffuseOnlineHeapOffset = 0;
    static const UINT           sc_BumpOnlineHeapOffset = 1;
    static const UINT           sc_ElwOnlineHeapOffset = 2;
    static const UINT           sc_NumSRVsPerFrame = 3;    
    static const UINT           sc_NumSRVsTotal = 3 * sc_NumComputeBuffers;
    
    static const UINT           sc_CBVHeapOffset = sc_NumSRVsTotal;
    static const UINT           sc_NumCBVsTotal = sc_NumComputeBuffers;
    static const UINT           sc_NumCBVsPerFrame = 1;

    static const UINT           sc_UAVHeapOffset = sc_CBVHeapOffset + sc_NumCBVsTotal;
    static const UINT           sc_NumUAVs = sc_NumComputeBuffers;

    static const UINT           sc_Stage0SamplerHeapOffset = 0;
    static const UINT           sc_Stage1SamplerHeapOffset = 1;
    static const UINT           sc_NumSamplers = 2;

    // Root parameter slots
    static const UINT           sc_SRVTableBindSlot = 0;
    static const UINT           sc_SamplerTableBindSlot = 1;
    static const UINT           sc_CBTableBindSlot = 2;
    static const UINT           sc_UAVTableBindSlot = 3;

    D3D12_COMMAND_LIST_TYPE GetComputeCommandListType()
    {
        return D3D12_COMMAND_LIST_TYPE_COMPUTE;
    };


    CComPtr<ID3D12PipelineState>      m_spBackgroundPSO;
    CComPtr<ID3D12PipelineState>      m_spSpherePSO;
    CComPtr<ID3D12PipelineState>      m_spComputePSO;

    UINT m_width;
    UINT m_height;
    
    HWND m_window;

    HANDLE m_previousFrameCompleteEvent;

    float m_rotateAngle;
    float m_backgroundAngle;
    float m_backgroundZoom;

    bool m_Benchmark;
    UINT m_Frames;
    DWORD m_StartTime;
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

        App app;

        app.Init();

        while(1)
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
