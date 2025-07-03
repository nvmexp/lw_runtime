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
#include <direct.h>
#include <dxutils.h>

// Pick a test
#define TEST_DEFAULT (m_TestCase == TC_DEFAULT) // Original from MSFT
#define TEST_0 (m_TestCase == TC_0) // 2 slots Sampler + (SRV + CBV + UAV + SRV + CBV + UAV)
#define TEST_1 (m_TestCase == TC_1) // 5 slots Sampler + SRV + SRV + CBV + CBV + UAV + UAV
#define TEST_2 (m_TestCase == TC_2) // 3 slots Sampler + (SRV + SRV) + (CBV + CBV) + (UAV + UAV)
#define TEST_3 (m_TestCase == TC_3) // root cbvs root uavs
#define TEST_4 (m_TestCase == TC_4) // root cbvs & ranged cbvs root uavs and ranged uavs
#define TEST_5 (m_TestCase == TC_5) // base shader register
#define TEST_6 (m_TestCase == TC_6) // root cbv ranged cbv in same space and root CBV in alternate space

#include "shaders_default.h"
#include "shaders_0.h"
#include "shaders_1.h"
#include "shaders_2.h"

using namespace DirectX;
using namespace std;

static const D3D12_COMMAND_QUEUE_DESC s_CommandQueueDesc;
static UINT g_EventWaitTime = 5000;

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

struct ConstantBufferContents
{
    XMMATRIX worldViewProjection;    // World * View * Projection matrix
    XMMATRIX ilwWorldViewProjection;
    XMMATRIX world;
    XMMATRIX worldView;
    XMVECTOR eyePt;
};

struct Constants_TC6
{
    float scaleFactor;
    float half;
    float quarter;
    float zero;
};

struct Constants
{
    float scaleFactor[2];
    float half[2];
    float quarter[2];
    float zero[2];
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
        : m_rotateAngle(0.0f),
          m_fenceValue(0), 
          m_TestCase(TC_DEFAULT)

    {
        ZeroMemory(m_UAVGPUOffset, sizeof(m_UAVGPUOffset));
    }

    ~App()
    {
        FlushAndFinish();

        // Ensure swapchain is not fullscreen:
        if (m_spSwapChain)
        {
            m_spSwapChain->SetFullscreenState( FALSE, NULL );
        }
        
        CComPtr<ID3D12DebugDevice> spDebug;
        if (SUCCEEDED(m_spDevice12->QueryInterface(&spDebug)))
        {
            spDebug->ReportLiveDeviceObjects(D3D12_RLDO_IGNORE_INTERNAL | D3D12_RLDO_DETAIL);
        }
    }
    void InitHeapOffsets()
    {

        switch (m_TestCase)
        {
        case TC_DEFAULT:
        {
            m_DiffuseOnlineHeapOffset[m_TestCase] = 0;
            m_BumpOnlineHeapOffset[m_TestCase] = 1;
            m_ElwOnlineHeapOffset[m_TestCase] = 2;
            m_CBV0HeapOffset[m_TestCase] = 3;
            m_UAV0HeapOffset[m_TestCase] = 0;
            m_UAV1HeapOffset[m_TestCase] = 0;
            m_UAV2HeapOffset[m_TestCase] = 0;

            break;
        }
        case TC_0:
        {
            m_SRV0HeapOffsetStart[m_TestCase] = 0;
            m_DiffuseOnlineHeapOffset[m_TestCase] = m_SRV0HeapOffsetStart[m_TestCase] + 10; // This must match the register index in the shader

            m_CBV0HeapOffsetStart[m_TestCase] = sc_NumSRVs0;
            m_CBV0HeapOffset[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + 2;

            m_UAV0HeapOffsetStart[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + sc_NumCBVs0;
            m_UAV0HeapOffset[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + 1;

            m_SRV1HeapOffsetStart[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + sc_NumUAVs0;
            m_BumpOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 10;
            m_ElwOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 11;

            m_CBV1HeapOffsetStart[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + sc_NumSRVs1;
            m_CBV1HeapOffset[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + 2;

            m_UAV1HeapOffsetStart[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + sc_NumCBVs1;
            m_UAV1HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 1;
            m_UAV2HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 2;

            break;
        }
        case TC_1:
        {
            m_SRV0HeapOffsetStart[m_TestCase] = 0;
            m_DiffuseOnlineHeapOffset[m_TestCase] = m_SRV0HeapOffsetStart[m_TestCase] + 10; // This must match the register index in the shader

            m_SRV1HeapOffsetStart[m_TestCase] = sc_NumSRVs0;
            m_BumpOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 10;
            m_ElwOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 11;

            m_CBV0HeapOffsetStart[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + sc_NumSRVs1;
            m_CBV0HeapOffset[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + 2;

            m_CBV1HeapOffsetStart[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + sc_NumCBVs0;
            m_CBV1HeapOffset[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + 2;

            m_UAV0HeapOffsetStart[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + sc_NumCBVs1;
            m_UAV0HeapOffset[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + 1;

            m_UAV1HeapOffsetStart[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + sc_NumUAVs0;
            m_UAV1HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 1;
            m_UAV2HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 2;

            static const UINT           sc_SRVTableBindSlot0 = 0;
            static const UINT           sc_SRVTableBindSlot1 = 1;
            static const UINT           sc_CBTableBindSlot0 = 2;
            static const UINT           sc_CBTableBindSlot1 = 3;
            static const UINT           sc_UAVTableBindSlot0 = 4;
            static const UINT           sc_UAVTableBindSlot1 = 5;
            break;
        }
        case TC_2:
        {
            m_SRV0HeapOffsetStart[m_TestCase] = 0;
            m_DiffuseOnlineHeapOffset[m_TestCase] = m_SRV0HeapOffsetStart[m_TestCase] + 10; // This must match the register index in the shader

            m_SRV1HeapOffsetStart[m_TestCase] = sc_NumSRVs0;
            m_BumpOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 10;
            m_ElwOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 11;

            m_CBV0HeapOffsetStart[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + sc_NumSRVs1;
            m_CBV0HeapOffset[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + 2;

            m_CBV1HeapOffsetStart[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + sc_NumCBVs0;
            m_CBV1HeapOffset[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + 2;

            m_UAV0HeapOffsetStart[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + sc_NumCBVs1;
            m_UAV0HeapOffset[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + 1;

            m_UAV1HeapOffsetStart[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + sc_NumUAVs0;
            m_UAV1HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 1;
            m_UAV2HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 2;

            // Root signature parameter slots
            static const UINT           sc_SRVTableBindSlot = 0;
            static const UINT           sc_CBTableBindSlot = 1;
            static const UINT           sc_UAVTableBindSlot = 2;
            break;
        }
        case TC_3:
        {
            m_SRV0HeapOffsetStart[m_TestCase] = 0;
            m_DiffuseOnlineHeapOffset[m_TestCase] = m_SRV0HeapOffsetStart[m_TestCase] + 10; // This must match the register index in the shader

            m_SRV1HeapOffsetStart[m_TestCase] = sc_NumSRVs0;
            m_BumpOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 10;
            m_ElwOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 11;

            m_CBV0HeapOffsetStart[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + sc_NumSRVs1;
            m_CBV0HeapOffset[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + 2;

            m_CBV1HeapOffsetStart[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + sc_NumCBVs0;
            m_CBV1HeapOffset[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + 2;

            m_UAV0HeapOffsetStart[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + sc_NumCBVs1;
            m_UAV0HeapOffset[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + 1;

            m_UAV1HeapOffsetStart[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + sc_NumUAVs0;
            m_UAV1HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 1;
            m_UAV2HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 2;

            static const UINT           sc_SRVTableBindSlot = 0;
            static const UINT           sc_CBTableBindSlot0 = 1;
            static const UINT           sc_CBTableBindSlot1 = 2;
            static const UINT           sc_CBTableBindSlot2 = 3;
            static const UINT           sc_UAVTableBindSlot0 = 4;
            static const UINT           sc_UAVTableBindSlot1 = 5;
            static const UINT           sc_UAVTableBindSlot2 = 6;
            break;
        }
        case TC_4:
        {
            m_SRV0HeapOffsetStart[m_TestCase] = 0;
            m_DiffuseOnlineHeapOffset[m_TestCase] = m_SRV0HeapOffsetStart[m_TestCase] + 10; // This must match the register index in the shader

            m_CBV0HeapOffsetStart[m_TestCase] = sc_NumSRVs0;
            m_CBV0HeapOffset[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + 2;

            m_UAV0HeapOffsetStart[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + sc_NumCBVs0;
            m_UAV0HeapOffset[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + 1;

            m_SRV1HeapOffsetStart[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + sc_NumUAVs0;
            m_BumpOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 10;
            m_ElwOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 11;

            m_CBV1HeapOffsetStart[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + sc_NumSRVs1;
            m_CBV1HeapOffset[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + 2;

            m_UAV1HeapOffsetStart[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + sc_NumCBVs1;
            m_UAV1HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 1;
            m_UAV2HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 2;

            static const UINT           sc_SRV_CBV_UAV_TableBindSlot = 0;
            static const UINT           sc_CBTableBindSlot0 = 1;
            static const UINT           sc_CBTableBindSlot1 = 2;
            static const UINT           sc_UAVTableBindSlot0 = 3;
            static const UINT           sc_UAVTableBindSlot1 = 4;
            break;
        }
        case TC_5:
        {
            m_SRV0HeapOffsetStart[m_TestCase] = 0;
            m_DiffuseOnlineHeapOffset[m_TestCase] = m_SRV0HeapOffsetStart[m_TestCase] + 10; // This must match the register index in the shader

            m_CBV0HeapOffsetStart[m_TestCase] = sc_NumSRVs0;
            m_CBV0HeapOffset[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + 2;

            m_UAV0HeapOffsetStart[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + sc_NumCBVs0;
            m_UAV0HeapOffset[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + 1;

            m_SRV1HeapOffsetStart[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + sc_NumUAVs0;
            m_BumpOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 10;
            m_ElwOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 11;

            m_CBV1HeapOffsetStart[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + sc_NumSRVs1;
            m_CBV1HeapOffset[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + 2;

            m_UAV1HeapOffsetStart[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + sc_NumCBVs1;
            m_UAV1HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 1;
            m_UAV2HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 2;

            static const UINT           sc_SRV_CBV_UAV_TableBindSlot = 0;
            break;
        }
        case TC_6:
        {

            m_SRV0HeapOffsetStart[m_TestCase] = 0;
            m_DiffuseOnlineHeapOffset[m_TestCase] = m_SRV0HeapOffsetStart[m_TestCase] + 10; // This must match the register index in the shader

            m_CBV0HeapOffsetStart[m_TestCase] = sc_NumSRVs0;
            m_CBV0HeapOffset[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + 2;

            m_UAV0HeapOffsetStart[m_TestCase] = m_CBV0HeapOffsetStart[m_TestCase] + sc_NumCBVs0;
            m_UAV0HeapOffset[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + 1;

            m_SRV1HeapOffsetStart[m_TestCase] = m_UAV0HeapOffsetStart[m_TestCase] + sc_NumUAVs0;
            m_BumpOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 10;
            m_ElwOnlineHeapOffset[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + 11;

            m_CBV1HeapOffsetStart[m_TestCase] = m_SRV1HeapOffsetStart[m_TestCase] + sc_NumSRVs1;
            m_CBV1HeapOffset[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + 2;

            m_UAV1HeapOffsetStart[m_TestCase] = m_CBV1HeapOffsetStart[m_TestCase] + sc_NumCBVs1;
            m_UAV1HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 1;
            m_UAV2HeapOffset[m_TestCase] = m_UAV1HeapOffsetStart[m_TestCase] + 2;

            static const UINT           sc_SRV_CBV_UAV_TableBindSlot = 0;
            static const UINT           sc_CBTableBindSlot0 = 1;
            static const UINT           sc_CBTableBindSlot1 = 2;
            static const UINT           sc_UAVTableBindSlot0 = 3;
            static const UINT           sc_UAVTableBindSlot1 = 4;
            break;
        }
        }
    }
    void Init()
    {
        m_Benchmark = (0 != atoi(GetCommandLine("benchmark", "0")));
        m_DumpFrames = atoi(GetCommandLine("dumpframes", "0"));
        m_Frames = 0;
        const char *tc_str = GetCommandLine("Test", NULL);
        if (tc_str && _stricmp(tc_str, "default") != 0)
        {
            int index = atoi(tc_str);
            if (index >= 0 && index < NUM_TC - 1)
            {
                m_TestCase = (TEST_CASE)(index + 1);
            }
        }
        InitHeapOffsets();

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

        bool enableDebug12 = false;

        if (0 != atoi(GetCommandLine("Debug", "0")))
        {
            enableDebug12 = true;
        }

        if (enableDebug12)
        {
            CComPtr<ID3D12Debug> spDebug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&spDebug))))
            {
                spDebug->EnableDebugLayer();
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

        if (fullScreen && proxy)
        {
            // Force proxy mode
            m_spSwapChain->ResizeBuffers(2, 800, 800, DXGI_FORMAT_UNKNOWN, 0 );
        }

        CheckHr(m_RTVHeap.Create(m_spDevice12, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1));
        CheckHr(m_OnlineDH.Create(m_spDevice12, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, sc_NumSRVs0 + sc_NumSRVs1 + sc_NumCBVs0 + sc_NumCBVs1 + sc_NumUAVs0 + sc_NumUAVs1, true));

        CheckHr(m_spDevice12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_spFence)));

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
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(NUM_VERTICES*sizeof(GEOMETRY_VERTEX)+4*sizeof(ELW_VERTEX)),
                D3D12_RESOURCE_STATE_COPY_DEST,
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
            CD3DX12_STATIC_SAMPLER_DESC Sampler[sc_NumSamplers];
            for (UINT i = 0; i < sc_NumSamplers; i++)
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
                Sampler[i].RegisterSpace = TEST_DEFAULT ? 0 : 1;
                Sampler[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            }
            CD3DX12_DESCRIPTOR_RANGE DescRange[6];
            CD3DX12_ROOT_PARAMETER RTSlot[7];
            CD3DX12_ROOT_SIGNATURE_DESC RootSig;
            switch (m_TestCase)
            {
            case TC_DEFAULT:
                DescRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0); // t0-t2
                DescRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0); // b0

                RTSlot[sc_SRVTableBindSlot].InitAsDescriptorTable(1, &DescRange[0], D3D12_SHADER_VISIBILITY_PIXEL); // t0-t2
                RTSlot[sc_CBTableBindSlot].InitAsDescriptorTable(1, &DescRange[1], D3D12_SHADER_VISIBILITY_ALL); // b0
                RootSig = CD3DX12_ROOT_SIGNATURE_DESC(2, RTSlot, sc_NumSamplers, Sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
                break;
            case TC_0:
                DescRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs0, 0, 2);
                DescRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, sc_NumCBVs0, 0, 4);
                DescRange[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, sc_NumUAVs0, 0, 6);
                DescRange[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs1, 0, 3);
                DescRange[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, sc_NumCBVs1, 0, 5);
                DescRange[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, sc_NumUAVs1, 0, 7);

                RTSlot[sc_SRV_CBV_UAV_TableBindSlot].InitAsDescriptorTable(6, &DescRange[0], D3D12_SHADER_VISIBILITY_ALL);
                RootSig = CD3DX12_ROOT_SIGNATURE_DESC(1, RTSlot, sc_NumSamplers, Sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
                break;
            case TC_1:
                DescRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs0, 0, 2);
                DescRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs1, 0, 3);
                DescRange[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, sc_NumCBVs0, 0, 4);
                DescRange[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, sc_NumCBVs1, 0, 5);
                DescRange[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, sc_NumUAVs0, 0, 6);
                DescRange[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, sc_NumUAVs1, 0, 7);

                RTSlot[sc1_SRVTableBindSlot0].InitAsDescriptorTable(1, &DescRange[0], D3D12_SHADER_VISIBILITY_PIXEL);
                RTSlot[sc1_SRVTableBindSlot1].InitAsDescriptorTable(1, &DescRange[1], D3D12_SHADER_VISIBILITY_PIXEL);
                RTSlot[sc1_CBTableBindSlot0].InitAsDescriptorTable(1, &DescRange[2], D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc1_CBTableBindSlot1].InitAsDescriptorTable(1, &DescRange[3], D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc1_UAVTableBindSlot0].InitAsDescriptorTable(1, &DescRange[4], D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc1_UAVTableBindSlot1].InitAsDescriptorTable(1, &DescRange[5], D3D12_SHADER_VISIBILITY_ALL);
                RootSig = CD3DX12_ROOT_SIGNATURE_DESC(6, RTSlot, sc_NumSamplers, Sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
                break;
            case TC_2:
                DescRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs0, 0, 2);
                DescRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs1, 0, 3);
                DescRange[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, sc_NumCBVs0, 0, 4);
                DescRange[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, sc_NumCBVs1, 0, 5);
                DescRange[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, sc_NumUAVs0, 0, 6);
                DescRange[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, sc_NumUAVs1, 0, 7);

                RTSlot[sc_SRVTableBindSlot].InitAsDescriptorTable(2, &DescRange[0], D3D12_SHADER_VISIBILITY_PIXEL);
                RTSlot[sc_CBTableBindSlot].InitAsDescriptorTable(2, &DescRange[2], D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc_UAVTableBindSlot].InitAsDescriptorTable(2, &DescRange[4], D3D12_SHADER_VISIBILITY_ALL);
                RootSig = CD3DX12_ROOT_SIGNATURE_DESC(3, RTSlot, sc_NumSamplers, Sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
                break;
            case TC_3:
                DescRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs0, 0, 2);
                DescRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs1, 0, 3);

                RTSlot[sc_SRVTableBindSlot].InitAsDescriptorTable(2, &DescRange[0], D3D12_SHADER_VISIBILITY_PIXEL);
                RTSlot[sc3_CBTableBindSlot0].InitAsConstantBufferView(2, 4, D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc3_CBTableBindSlot1].InitAsConstantBufferView(2, 5, D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc3_CBTableBindSlot2].InitAsConstantBufferView(3, 5, D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc3_UAVTableBindSlot0].InitAsUnorderedAccessView(1, 6, D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc3_UAVTableBindSlot1].InitAsUnorderedAccessView(1, 7, D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc3_UAVTableBindSlot2].InitAsUnorderedAccessView(2, 7, D3D12_SHADER_VISIBILITY_ALL);
                RootSig = CD3DX12_ROOT_SIGNATURE_DESC(7, RTSlot, sc_NumSamplers, Sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
                break;
            case TC_4:
                DescRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs0, 0, 2);
                DescRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, sc_NumCBVs0, 0, 4);
                DescRange[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, sc_NumUAVs0, 0, 6);
                DescRange[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs1, 0, 3);

                RTSlot[sc_SRV_CBV_UAV_TableBindSlot].InitAsDescriptorTable(4, &DescRange[0], D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc4_CBTableBindSlot0].InitAsConstantBufferView(2, 5, D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc4_CBTableBindSlot1].InitAsConstantBufferView(3, 5, D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc4_UAVTableBindSlot0].InitAsUnorderedAccessView(1, 7, D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc4_UAVTableBindSlot1].InitAsUnorderedAccessView(2, 7, D3D12_SHADER_VISIBILITY_ALL);
                RootSig = CD3DX12_ROOT_SIGNATURE_DESC(5, RTSlot, sc_NumSamplers, Sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
                break;
            case TC_5:
                DescRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs0, 10, 2);
                DescRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, sc_NumCBVs0, 10, 4);
                DescRange[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, sc_NumUAVs0, 10, 6);
                DescRange[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs1, 10, 3);
                DescRange[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, sc_NumCBVs1, 10, 5);
                DescRange[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, sc_NumUAVs1, 10, 7);

                RTSlot[sc_SRV_CBV_UAV_TableBindSlot].InitAsDescriptorTable(6, &DescRange[0], D3D12_SHADER_VISIBILITY_ALL); // t0-t2
                RootSig = CD3DX12_ROOT_SIGNATURE_DESC(1, RTSlot, sc_NumSamplers, Sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
                break;
            case TC_6:
                DescRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs0, 0, 2);
                DescRange[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, sc_NumCBVs0, 0, 4);
                DescRange[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, sc_NumUAVs0, 0, 6);
                DescRange[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, sc_NumSRVs1, 0, 3);

                RTSlot[sc_SRV_CBV_UAV_TableBindSlot].InitAsDescriptorTable(4, &DescRange[0], D3D12_SHADER_VISIBILITY_ALL); // t0-t2
                RTSlot[sc6_CBTableBindSlot0].InitAsConstants(sizeof(Constants_TC6) / 4, 2, 5, D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc6_CBTableBindSlot1].InitAsConstants(sizeof(Constants_TC6) / 4, 3, 5, D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc6_UAVTableBindSlot0].InitAsUnorderedAccessView(1, 7, D3D12_SHADER_VISIBILITY_ALL);
                RTSlot[sc6_UAVTableBindSlot1].InitAsUnorderedAccessView(2, 7, D3D12_SHADER_VISIBILITY_ALL);
                RootSig = CD3DX12_ROOT_SIGNATURE_DESC(5, RTSlot, sc_NumSamplers, Sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
                break;
            };
            CComPtr<ID3DBlob> pSerializedLayout;
            D3D12SerializeRootSignature(&RootSig, D3D_ROOT_SIGNATURE_VERSION_1, &pSerializedLayout,NULL);

            CheckHr(m_spDevice12->CreateRootSignature(
                1,
                pSerializedLayout->GetBufferPointer(), 
                pSerializedLayout->GetBufferSize(),
                __uuidof(ID3D12RootSignature),
                (void**)&m_spRootSignature));
        }

        CheckHr(m_spDevice12->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&m_spNonRenderingCommandAllocator.p)));

        CheckHr(m_spDevice12->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT, 
            m_spNonRenderingCommandAllocator,
            nullptr,
            IID_PPV_ARGS(&m_spCommandList)));

        UINT64 VerticesOffset = 0;
        {
            UINT8* pVertices = SuballocateFromUploadHeap((SIZE_T)m_spDefaultBuffer->GetDesc().Width);
            VerticesOffset = pVertices-m_pDataBegin;

            CreateSphereVertices(reinterpret_cast<GEOMETRY_VERTEX*>(pVertices));
            
            CreateElwVertices(reinterpret_cast<ELW_VERTEX*>(pVertices+NUM_VERTICES*sizeof(GEOMETRY_VERTEX)));

            D3D12_RESOURCE_BARRIER desc;
            ZeroMemory( &desc, sizeof( desc ) );

            m_spCommandList->CopyBufferRegion(
                m_spDefaultBuffer,
                0,
                m_spUploadBuffer,
                VerticesOffset,
                m_spDefaultBuffer->GetDesc().Width
                );

            LoadTextures();

            if (!TEST_DEFAULT) LoadUAVs();

            desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            desc.Transition.pResource = m_spDefaultBuffer;
            desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            desc.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            desc.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
            
            m_spCommandList->ResourceBarrier(1, &desc);
        }

        CheckHr(m_spCommandList->Close());

        m_spCommandQueue->SetMarker(0, L"Setting marker", sizeof(L"Setting marker"));
        m_spCommandQueue->ExelwteCommandLists(1, CommandListCast(&m_spCommandList.p));

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
            const void *pShader = NULL;
            SIZE_T shaderSize = 0;
            if (TEST_DEFAULT) { pShader = g_ElwVS; shaderSize = sizeof(g_ElwVS); };
            if (TEST_0 || TEST_1 || TEST_2) { pShader = g_ElwVS0; shaderSize = sizeof(g_ElwVS0); };
            if (TEST_3 || TEST_4 || TEST_6) { pShader = g_ElwVS1; shaderSize = sizeof(g_ElwVS1); };
            if (TEST_5) { pShader = g_ElwVS2; shaderSize = sizeof(g_ElwVS2); };
            psoDesc.VS = { pShader, shaderSize };
            if (TEST_DEFAULT) { pShader = g_ElwPS; shaderSize = sizeof(g_ElwPS); };
            if (TEST_0 || TEST_1 || TEST_2) { pShader = g_ElwPS0; shaderSize = sizeof(g_ElwPS0); };
            if (TEST_3 || TEST_4 || TEST_6) { pShader = g_ElwPS1; shaderSize = sizeof(g_ElwPS1); };
            if (TEST_5) { pShader = g_ElwPS2; shaderSize = sizeof(g_ElwPS2); };
            psoDesc.PS = { pShader, shaderSize };
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

            if (0 != atoi(GetCommandLine("Library", "0")))
            {
                CComPtr<ID3DBlob> spBlob;
                CheckHr(m_spBackgroundPSO->GetCachedBlob(&spBlob));
                m_spBackgroundPSO.Release();
                psoDesc.CachedPSO.pCachedBlob = spBlob->GetBufferPointer();
                psoDesc.CachedPSO.CachedBlobSizeInBytes = spBlob->GetBufferSize();
                CheckHr(m_spDevice12->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_spBackgroundPSO)));
            }
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

            const void *pShader = NULL;
            SIZE_T shaderSize = 0;
            if (TEST_DEFAULT) { pShader = g_GeometryVS; shaderSize = sizeof(g_GeometryVS); };
            if (TEST_0 || TEST_1 || TEST_2) { pShader = g_GeometryVS0; shaderSize = sizeof(g_GeometryVS0); };
            if (TEST_3 || TEST_4 || TEST_6) { pShader = g_GeometryVS1; shaderSize = sizeof(g_GeometryVS1); };
            if (TEST_5) { pShader = g_GeometryVS2; shaderSize = sizeof(g_GeometryVS2); };
            psoDesc.VS = { pShader, shaderSize };
            if (TEST_DEFAULT) { pShader = g_GeometryPS; shaderSize = sizeof(g_GeometryPS); };
            if (TEST_0 || TEST_1 || TEST_2) { pShader = g_GeometryPS0; shaderSize = sizeof(g_GeometryPS0); };
            if (TEST_3 || TEST_4 || TEST_6) { pShader = g_GeometryPS1; shaderSize = sizeof(g_GeometryPS1); };
            if (TEST_5) { pShader = g_GeometryPS2; shaderSize = sizeof(g_GeometryPS2); };
            psoDesc.PS = { pShader, shaderSize };
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

            if (0 != atoi(GetCommandLine("Library", "0")))
            {
                CComPtr<ID3DBlob> spBlob;
                CheckHr(m_spSpherePSO->GetCachedBlob(&spBlob));
                m_spSpherePSO.Release();
                psoDesc.CachedPSO.pCachedBlob = spBlob->GetBufferPointer();
                psoDesc.CachedPSO.CachedBlobSizeInBytes = spBlob->GetBufferSize();
                CheckHr(m_spDevice12->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_spSpherePSO)));
            }
        }
    }

    void RecordCommandList(UINT backBufferIndex, CComPtr<ID3D12GraphicsCommandList> spCommandList)
    {
        CComPtr<ID3D12Resource>         spBackBuffer;
        CheckHr(m_spSwapChain->GetBuffer(backBufferIndex, IID_PPV_ARGS(&spBackBuffer)));
        m_spDevice12->CreateRenderTargetView(spBackBuffer, NULL, m_RTVHeap.hCPU(0));

        m_spDiffuseTexture->SetName(L"spDiffuseResource");
        m_spBumpTexture->SetName(L"spBumpResource");
        m_spElwTexture->SetName(L"spElwResource");

        ID3D12DescriptorHeap* pDH[1] = { m_OnlineDH };
        spCommandList->SetDescriptorHeaps(1, pDH);

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
            { m_spUAV0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS },
            { m_spUAV1, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS },
            { m_spUAV2, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS },
        };
        UINT barrier_count = TEST_DEFAULT ? 1 : _countof(barriers);

        {
            vector< D3D12_RESOURCE_BARRIER > barrierArray;

            for (UINT i = 0; i < barrier_count; i++)
            {
                D3D12_RESOURCE_BARRIER desc;
                ZeroMemory(&desc, sizeof(desc));

                desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                desc.Transition.pResource = barriers[i].pResource;
                desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                desc.Transition.StateBefore = barriers[i].stateBefore;
                desc.Transition.StateAfter = barriers[i].stateAfter;

                barrierArray.push_back(desc);
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
            0,
            NULL
            );
        spCommandList->EndEvent();

        spCommandList->OMSetRenderTargets(1, &m_RTVHeap.hCPU(0), true, NULL);

        D3D12_VIEWPORT viewport =
        {
            0.0f,
            0.0f,
            static_cast<float>(m_width),
            static_cast<float>(m_height),
            0.0f,
            1.0f
        };

        spCommandList->SetPrivateData(WKPDID_D3DDebugObjectNameW, (UINT)wcslen(_T("Commandlist")), _T("Commandlist"));


        spCommandList->RSSetViewports(1, &viewport);

        D3D12_RECT scissorRect = { 0, 0, m_width, m_height };
        spCommandList->RSSetScissorRects(1, &scissorRect);
        switch (m_TestCase)
        {
        case TC_DEFAULT:
            spCommandList->SetGraphicsRootDescriptorTable(sc_CBTableBindSlot, m_OnlineDH.hGPU(m_CBV0HeapOffset[m_TestCase]));
            break;
        case TC_0:
            spCommandList->SetGraphicsRootDescriptorTable(sc_SRV_CBV_UAV_TableBindSlot, m_OnlineDH.hGPU(m_SRV0HeapOffsetStart[m_TestCase]));
            break;
        case TC_1:
            spCommandList->SetGraphicsRootDescriptorTable(sc1_SRVTableBindSlot0, m_OnlineDH.hGPU(m_SRV0HeapOffsetStart[m_TestCase]));

            spCommandList->SetGraphicsRootDescriptorTable(sc1_SRVTableBindSlot1, m_OnlineDH.hGPU(m_SRV1HeapOffsetStart[m_TestCase]));

            spCommandList->SetGraphicsRootDescriptorTable(sc1_CBTableBindSlot0, m_OnlineDH.hGPU(m_CBV0HeapOffsetStart[m_TestCase]));

            spCommandList->SetGraphicsRootDescriptorTable(sc1_CBTableBindSlot1, m_OnlineDH.hGPU(m_CBV1HeapOffsetStart[m_TestCase]));

            spCommandList->SetGraphicsRootDescriptorTable(sc1_UAVTableBindSlot0, m_OnlineDH.hGPU(m_UAV0HeapOffsetStart[m_TestCase]));

            spCommandList->SetGraphicsRootDescriptorTable(sc1_UAVTableBindSlot1, m_OnlineDH.hGPU(m_UAV1HeapOffsetStart[m_TestCase]));
            break;
        case TC_2:
            spCommandList->SetGraphicsRootDescriptorTable(sc_SRVTableBindSlot, m_OnlineDH.hGPU(m_SRV0HeapOffsetStart[m_TestCase]));

            spCommandList->SetGraphicsRootDescriptorTable(sc_CBTableBindSlot, m_OnlineDH.hGPU(m_CBV0HeapOffsetStart[m_TestCase]));

            spCommandList->SetGraphicsRootDescriptorTable(sc_UAVTableBindSlot, m_OnlineDH.hGPU(m_UAV0HeapOffsetStart[m_TestCase]));
            break;
        case TC_3:
            spCommandList->SetGraphicsRootDescriptorTable(sc_SRVTableBindSlot, m_OnlineDH.hGPU(m_SRV0HeapOffsetStart[m_TestCase]));

            spCommandList->SetGraphicsRootUnorderedAccessView(sc3_UAVTableBindSlot0, m_spUploadBuffer->GetGPUVirtualAddress() + m_UAVGPUOffset[0]);
            spCommandList->SetGraphicsRootUnorderedAccessView(sc3_UAVTableBindSlot1, m_spUploadBuffer->GetGPUVirtualAddress() + m_UAVGPUOffset[1]);
            spCommandList->SetGraphicsRootUnorderedAccessView(sc3_UAVTableBindSlot2, m_spUploadBuffer->GetGPUVirtualAddress() + m_UAVGPUOffset[2]);
            break;
        case TC_4:
            spCommandList->SetGraphicsRootDescriptorTable(sc_SRV_CBV_UAV_TableBindSlot, m_OnlineDH.hGPU(m_SRV0HeapOffsetStart[m_TestCase]));

            spCommandList->SetGraphicsRootUnorderedAccessView(sc4_UAVTableBindSlot0, m_spUploadBuffer->GetGPUVirtualAddress() + m_UAVGPUOffset[1]);
            spCommandList->SetGraphicsRootUnorderedAccessView(sc4_UAVTableBindSlot1, m_spUploadBuffer->GetGPUVirtualAddress() + m_UAVGPUOffset[2]);
            break;
        case TC_5:
            spCommandList->SetGraphicsRootDescriptorTable(sc_SRV_CBV_UAV_TableBindSlot, m_OnlineDH.hGPU(m_SRV0HeapOffsetStart[m_TestCase]));
            break;
        case TC_6:
            spCommandList->SetGraphicsRootDescriptorTable(sc_SRV_CBV_UAV_TableBindSlot, m_OnlineDH.hGPU(m_SRV0HeapOffsetStart[m_TestCase]));

            spCommandList->SetGraphicsRootUnorderedAccessView(sc6_UAVTableBindSlot0, m_spUploadBuffer->GetGPUVirtualAddress() + m_UAVGPUOffset[1]);
            spCommandList->SetGraphicsRootUnorderedAccessView(sc6_UAVTableBindSlot1, m_spUploadBuffer->GetGPUVirtualAddress() + m_UAVGPUOffset[2]);
            break;
        }
        //
        // Render the background
        //
        {
            UINT offsets[] = { 0 };
            UINT strides[] = { sizeof(ELW_VERTEX) };

            spCommandList->SetPipelineState(m_spBackgroundPSO);

            spCommandList->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

            spCommandList->IASetVertexBuffers(0, 1, &m_VBVs[sc_ElwVBVHeapOffset]);
            if (TEST_DEFAULT) spCommandList->SetGraphicsRootDescriptorTable(sc_SRVTableBindSlot, m_OnlineDH.hGPU(m_ElwOnlineHeapOffset[m_TestCase]));
            spCommandList->DrawInstanced(4, 1, 0, 0);
        }

        //
        // Render the sphere
        //
        {
            spCommandList->SetPipelineState(m_spSpherePSO);

            spCommandList->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

            spCommandList->IASetVertexBuffers(0, 1, &m_VBVs[sc_GeomVBVHeapOffset]);
            if (TEST_DEFAULT) spCommandList->SetGraphicsRootDescriptorTable(sc_SRVTableBindSlot, m_OnlineDH.hGPU(m_DiffuseOnlineHeapOffset[m_TestCase])); // all srvs

            spCommandList->DrawInstanced(NUM_VERTICES - 1, 1, 0, 0);
        }

       {
            //
            // Resource barriers to transition all resources back to D3D12_RESOURCE_STATE_COMMON
            //
            vector< D3D12_RESOURCE_BARRIER > barrierArray;

            for( UINT i = 0; i < barrier_count; i++ )
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
    }

    static SIZE_T Align(SIZE_T uLocation, UINT32 uAlign=MEMORY_ALLOCATION_ALIGNMENT)
    {
        return (uLocation + ((SIZE_T)uAlign-1)) & ~((SIZE_T)uAlign-1);
    }

    void FlushAndFinish()
    {
        CheckHr(m_spCommandList->Close());
        m_spCommandQueue->ExelwteCommandLists(1, CommandListCast(&m_spCommandList.p));
        
        CheckHr(m_spCommandQueue->Signal(m_spFence, ++m_fenceValue));
        CheckHr(m_spFence->SetEventOnCompletion(m_fenceValue, m_hEvent));
        WaitForSingleObject(m_hEvent, INFINITE);
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

#define CHECKHR CheckHr
#define D3D12_NODEMASK    BIT(0),
#define BIT(x)      \
    (1 << (x))

    const UINT c_ProtectionMask = PAGE_NOACCESS | PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXELWTE |
        PAGE_EXELWTE_READ | PAGE_EXELWTE_READWRITE | PAGE_EXELWTE_WRITECOPY | PAGE_GUARD | PAGE_NOCACHE | PAGE_WRITECOMBINE;

    void FinishAndExelwteCommandList(ID3D12GraphicsCommandList *pCommandList)
    {
        CHECKHR(pCommandList->Close());

        m_spCommandQueue->ExelwteCommandLists(1, CommandListCast(&pCommandList));
    }


// Borrowed from d3d12test, dx11/UAV parts removed. 
    void GetBitmap(Bitmap *pBitmap)
    {
        // Get the front buffer
        UINT bbIndex = 0;
        if (m_spSwapChain)
        {
#ifndef NOD3D12
            CComPtr<IDXGISwapChain3> spSwapChain3;
            CHECKHR(m_spSwapChain->QueryInterface(&spSwapChain3));
            bbIndex = spSwapChain3->GetLwrrentBackBufferIndex();
#endif
            if (0 != atoi(GetCommandLine("Present", "1")))
            {
                bbIndex = 1 - bbIndex;
            }
        }

        D3D11_MAPPED_SUBRESOURCE mappedSr;
        //UINT bpp = sizeof(DWORD);
        {
            CComPtr< ID3D12Resource > pTex2D;
            if (m_spSwapChain)
            {
                CHECKHR(m_spSwapChain->GetBuffer(bbIndex, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&pTex2D)));
            }
            else
            {
                //pTex2D = m_BackBuffer.m_spResource12;
                //pTex2D.p->AddRef();
            }

            D3D12_RESOURCE_DESC texDesc = pTex2D->GetDesc();
            //assert(texDesc.Format == BACK_BUFFER_FORMAT);

            pBitmap->width = (UINT)texDesc.Width;
            pBitmap->height = texDesc.Height;
            pBitmap->pixels.resize((UINT)texDesc.Width * texDesc.Height);

            CComPtr< ID3D12Resource > pTex2D12;
            CHECKHR(pTex2D->QueryInterface(&pTex2D12));

            CComPtr<ID3D12Resource> spStaging12;

            D3D12_PLACED_SUBRESOURCE_FOOTPRINT PlacedTexture2D;
            PlacedTexture2D.Offset = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
            PlacedTexture2D.Footprint.Format = texDesc.Format;
            PlacedTexture2D.Footprint.Width = (UINT)texDesc.Width;
            PlacedTexture2D.Footprint.Height = texDesc.Height;
            PlacedTexture2D.Footprint.Depth = 1;
            PlacedTexture2D.Footprint.RowPitch = bpp * (UINT)texDesc.Width;

            // Align Pitch:
            PlacedTexture2D.Footprint.RowPitch =
                (PlacedTexture2D.Footprint.RowPitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) &
                ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);

            CHECKHR(m_spDevice12->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(max(PlacedTexture2D.Offset + PlacedTexture2D.Footprint.RowPitch * PlacedTexture2D.Footprint.Height, 64 * 1024)),
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(&spStaging12)));

            CComPtr<ID3D12GraphicsCommandList> spCommandList;
            CHECKHR(m_spDevice12->CreateCommandList(D3D12_NODEMASK D3D12_COMMAND_LIST_TYPE_DIRECT, m_spCommandAllocator, nullptr, IID_PPV_ARGS(&spCommandList)));

            D3D12_RESOURCE_BARRIER barrierDesc;
            ZeroMemory(&barrierDesc, sizeof(barrierDesc));
            if (!m_spSwapChain)
            {
                barrierDesc.Transition.pResource = pTex2D12;
                barrierDesc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                barrierDesc.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
                barrierDesc.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                spCommandList->ResourceBarrier(1, &barrierDesc);
            }

            CD3DX12_TEXTURE_COPY_LOCATION Dst(spStaging12, PlacedTexture2D);
            CD3DX12_TEXTURE_COPY_LOCATION Src(pTex2D12, 0);
            spCommandList->CopyTextureRegion(
                &Dst,
                0, 0, 0,
                &Src,
                NULL);

            if (!m_spSwapChain)
            {
                barrierDesc.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
                barrierDesc.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
                spCommandList->ResourceBarrier(1, &barrierDesc);
            }

            FinishAndExelwteCommandList(spCommandList);

            CHECKHR(m_spCommandQueue->Signal(m_spFence, ++m_fenceValue));
            CHECKHR(m_spFence->SetEventOnCompletion(m_fenceValue, m_hEvent));

            DWORD dwRes = WaitForSingleObject(m_hEvent, g_EventWaitTime);
            if (WAIT_TIMEOUT == dwRes)
            {
                throw exception("Hung GPU");
            }

            CHECKHR(spStaging12->Map(
                0,
                NULL,
                &mappedSr.pData
                ));

            MEMORY_BASIC_INFORMATION MBInfo;
            (void)VirtualQuery(mappedSr.pData, &MBInfo, sizeof(MBInfo));

            const UINT MaskedProtection = MBInfo.Protect & c_ProtectionMask;

            if (PAGE_READWRITE != MaskedProtection)
            {
                throw exception("Page protection settings on READBACK buffer doesn\'t conform to spec.");
            }

            for (UINT y = 0; y < texDesc.Height; y++)
            {
                DWORD *pScan =
                    reinterpret_cast<DWORD *>(
                    static_cast<BYTE *>(mappedSr.pData) + PlacedTexture2D.Offset + (y * PlacedTexture2D.Footprint.RowPitch)
                    );

                memcpy(&(pBitmap->pixels[y * (UINT)texDesc.Width]), pScan, sizeof(DWORD)* (UINT)texDesc.Width);
            }

            spStaging12->Unmap(0, NULL);
        }


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
        CheckHr(m_spCommandAllocator->Reset());

        //
        // Record a new command list
        //
        CComPtr<IDXGISwapChain3> spSwapChain3;
        CheckHr(m_spSwapChain->QueryInterface(&spSwapChain3));
        UINT backBufferIndex = spSwapChain3->GetLwrrentBackBufferIndex();

        CComPtr<ID3D12GraphicsCommandList> spCommandList;
        CheckHr(m_spDevice12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_spCommandAllocator, m_spBackgroundPSO, IID_PPV_ARGS(&spCommandList)));
        spCommandList->SetGraphicsRootSignature(m_spRootSignature);

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
            UINT64 Offset = pData - m_pDataBegin;
            
            memcpy(pData, &cbContents, sizeof(cbContents));

            // CPU and GPU are serialized
            D3D12_CONSTANT_BUFFER_VIEW_DESC CBDesc = { m_spUploadBuffer->GetGPUVirtualAddress() + Offset, (UINT)Align(sizeof(cbContents), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT) };
            m_spDevice12->CreateConstantBufferView(
                &CBDesc,
                m_OnlineDH.hCPU(m_CBV0HeapOffset[m_TestCase])
                );
            if (TEST_3) spCommandList->SetGraphicsRootConstantBufferView(sc3_CBTableBindSlot0, m_spUploadBuffer->GetGPUVirtualAddress() + Offset);
        }

        //
        // Update the other constant buffers
        //
        if (!TEST_DEFAULT)
        {
            for (int i = 0; i < 2; i++)
            {
                float scaleFactor = 1.37f;
                float half = 0.5f;
                float quarter = 0.25f;
                float zero = 0.0f;

                void *cb_data = NULL;
                size_t cb_size = 0;
                if (TEST_6)
                {
                    Constants_TC6 cbContents;
                    cbContents.scaleFactor = scaleFactor;
                    cbContents.half = half;
                    cbContents.quarter = quarter;
                    cbContents.zero = zero;
                    cb_data = &cbContents;
                    cb_size = sizeof(cbContents);
                }
                else
                {
                    Constants cbContents;
                    for (int j = 0; j < 2; j++)
                    {
                        cbContents.scaleFactor[j] = scaleFactor;
                        cbContents.half[j] = half;
                        cbContents.quarter[j] = quarter;
                        cbContents.zero[j] = zero;
                    }
                    cb_data = &cbContents;
                    cb_size = sizeof(cbContents);
                }

                UINT8* pData = SuballocateFromUploadHeap(cb_size, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
                UINT64 Offset = pData - m_pDataBegin;

                memcpy(pData, cb_data, cb_size);

                // CPU and GPU are serialized
                D3D12_CONSTANT_BUFFER_VIEW_DESC CBDesc = { m_spUploadBuffer->GetGPUVirtualAddress() + Offset, (UINT)Align(cb_size, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT) };
                m_spDevice12->CreateConstantBufferView(
                    &CBDesc,
                    m_OnlineDH.hCPU(m_CBV1HeapOffset[m_TestCase] + i)
                    );
                if (TEST_3) spCommandList->SetGraphicsRootConstantBufferView(sc3_CBTableBindSlot1 + i, m_spUploadBuffer->GetGPUVirtualAddress() + Offset);

                if (TEST_4) spCommandList->SetGraphicsRootConstantBufferView(sc4_CBTableBindSlot0 + i, m_spUploadBuffer->GetGPUVirtualAddress() + Offset);

                if (TEST_6) spCommandList->SetGraphicsRoot32BitConstants(sc6_CBTableBindSlot0 + i, sizeof(Constants_TC6) / 4, cb_data, 0);
            }
        }


        RecordCommandList(backBufferIndex, spCommandList);

        CheckHr(spCommandList->Close());
        m_spCommandQueue->ExelwteCommandLists(1, CommandListCast(&spCommandList.p));

        HRESULT hr = m_spSwapChain->Present(m_Benchmark ? 0 : 1, 0);

        // Present can fail after fs<->windowed transition without a call to ResizeBuffers
        if( FAILED(hr) )
        {
            FlushAndFinish();
            m_spSwapChain->ResizeBuffers(2, 0, 0, DXGI_FORMAT_UNKNOWN, 0);

            DXGI_SWAP_CHAIN_DESC newDesc;
            CheckHr(m_spSwapChain->GetDesc(&newDesc));

            m_width = newDesc.BufferDesc.Width;
            m_height = newDesc.BufferDesc.Height;
        }
        else if (m_DumpFrames) 
        {
            Bitmap tmp;
            GetBitmap(&tmp);
            const char *outputDir = GetCommandLine("-o", NULL);
            if (!atoi(GetCommandLine("Present", "1")) || outputDir != NULL) //If not presenting, save bmps
            {
                char outfileName[100];
                sprintf_s(outfileName, 100, "bumpearth12_out%02u.bmp", m_Frames);
                SaveBitmap(&tmp, m_Frames == 0 ? outputDir : NULL, outfileName);
            }

        }

        CheckHr(m_spCommandQueue->Signal(m_spFence, ++m_fenceValue));
        CheckHr(m_spFence->SetEventOnCompletion(m_fenceValue, m_previousFrameCompleteEvent));

        if (m_DumpFrames) m_rotateAngle += 0.2f;
        else m_rotateAngle += 0.01f;

        // Front-buffer synchronization is not yet implemented
        // So wait for vblank to ensure that flips do not get queued
        // Note that this will fail on a render-only adapter (like WARP)
        CComPtr<IDXGIOutput> spOutput;
        if (!m_Benchmark && SUCCEEDED( m_spSwapChain->GetContainingOutput(&spOutput) ) )
        {
            CheckHr(spOutput->WaitForVBlank());
        }

        // In 'benchmark mode' exit the application after 5 seconds with an ExitCode equal to the number of frames rendered
        m_Frames++;
        DWORD duration = timeGetTime() - m_StartTime;
        if (m_Benchmark && duration > 5000)
        {
            ExitProcess(m_Frames);
        }
        if (m_DumpFrames > 0 && m_Frames > m_DumpFrames)
        {
            ExitProcess(m_Frames);
        }

    }

    void LoadTextures()
    {
        const int textures = 3;
        const LPCWSTR names[textures] = { _T("EARTH.BMP"), _T("EARTHBUMP.BMP"), _T("LOBBY.BMP") };
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

        CheckHr(m_spDevice12->CreateHeap(
            &CD3DX12_HEAP_DESC(m_spDevice12->GetResourceAllocationInfo(1, textures, pRDescs), D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES),
            IID_PPV_ARGS(&m_spTextureHeap)
            ));

        UINT64 DefHeapOffset = 0;
        for (int i = 0; i < textures; ++i)
        {
            DefHeapOffset = Align((SIZE_T)DefHeapOffset, (UINT32)RAInfo[i].Alignment);

            CheckHr(m_spDevice12->CreatePlacedResource(
                m_spTextureHeap, DefHeapOffset,
                &RDescs[i],
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(&spTexture2D[i])
                ));

            UINT64 UplHeapSize;
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedTex2D = { 0 };
            m_spDevice12->GetCopyableFootprints(&RDescs[i], 0, 1, 0, &placedTex2D, NULL, NULL, &UplHeapSize);

            DefHeapOffset += RAInfo[i].SizeInBytes;

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

            D3D12_RESOURCE_BARRIER& desc = RBDesc[i];
            desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            desc.Transition.pResource = spTexture2D[i];
            desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            desc.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            desc.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }

        m_spCommandList->ResourceBarrier(ARRAYSIZE(RBDesc), RBDesc);

        m_spDiffuseTexture = spTexture2D[0];
        m_spBumpTexture = spTexture2D[1];
        m_spElwTexture = spTexture2D[2];

        m_spDevice12->CreateShaderResourceView(
            m_spDiffuseTexture,
            NULL,
            m_OnlineDH.hCPU(m_DiffuseOnlineHeapOffset[m_TestCase]));

        m_spDevice12->CreateShaderResourceView(
            m_spBumpTexture,
            NULL,
            m_OnlineDH.hCPU(m_BumpOnlineHeapOffset[m_TestCase]));

        m_spDevice12->CreateShaderResourceView(
            m_spElwTexture,
            NULL,
            m_OnlineDH.hCPU(m_ElwOnlineHeapOffset[m_TestCase]));
    }

    void LoadUAVs()
    {
        const int uavs = 3;
        CD3DX12_RESOURCE_DESC RDescs[uavs];
        const D3D12_RESOURCE_DESC* pRDescs = reinterpret_cast<const D3D12_RESOURCE_DESC*>(RDescs);
        CComPtr<ID3D12Resource> spUAVs[uavs];
        D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc[uavs];
        ZeroMemory(&UAVDesc, sizeof(UAVDesc));
        D3D12_RESOURCE_BARRIER RBDesc[uavs];
        ZeroMemory(&RBDesc, sizeof(RBDesc));

        DWORD initialData0[2] = {  0, 127 };
        DWORD initialData1[2] = { 31, 159 };
        DWORD initialData2[2] = { 63, 191 };
        
        CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(initialData0), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        D3D11_SUBRESOURCE_DATA srd[uavs] = { initialData0, 0, 0, initialData1, 0, 0, initialData2, 0, 0 };

        for (int i = 0; i < uavs; ++i)
        {
            CheckHr(m_spDevice12->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &bufferDesc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                NULL,
                IID_PPV_ARGS(&spUAVs[i])));

            UINT64 BufferSize = GetRequiredIntermediateSize(spUAVs[i], 0, 1);
            if (BufferSize >= (SIZE_T)-1)
            {
                throw exception("Update requires too large a buffer.");
            }

            UINT8* pData = SuballocateFromUploadHeap((SIZE_T)BufferSize, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
            m_UAVGPUOffset[i] = pData - m_pDataBegin;

            D3D12_SUBRESOURCE_DATA InitialData = { srd[i].pSysMem, srd[i].SysMemPitch, srd[i].SysMemSlicePitch };
            UpdateSubresources<1>(m_spCommandList, spUAVs[i], m_spUploadBuffer, m_UAVGPUOffset[i], 0u, 1, &InitialData);

            UAVDesc[i].ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            if (TEST_3 || TEST_4 || TEST_6)
            {
                UAVDesc[i].Format = DXGI_FORMAT_R32_TYPELESS;
                UAVDesc[i].Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            }
            else 
            {
                UAVDesc[i].Format = DXGI_FORMAT_R32_UINT;
            }
            UAVDesc[i].Buffer.NumElements = _countof(initialData0);

            D3D12_RESOURCE_BARRIER& desc = RBDesc[i];
            desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            desc.Transition.pResource = spUAVs[i];
            desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            desc.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            desc.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        }

        m_spCommandList->ResourceBarrier(ARRAYSIZE(RBDesc), RBDesc);
        
        m_spUAV0 = spUAVs[0];
        m_spUAV1 = spUAVs[1];
        m_spUAV2 = spUAVs[2];

        m_spDevice12->CreateUnorderedAccessView(
            m_spUAV0,
            NULL,
            &UAVDesc[0],
            m_OnlineDH.hCPU(m_UAV0HeapOffset[m_TestCase]));

        m_spDevice12->CreateUnorderedAccessView(
            m_spUAV1,
            NULL,
            &UAVDesc[1],
            m_OnlineDH.hCPU(m_UAV1HeapOffset[m_TestCase]));

        m_spDevice12->CreateUnorderedAccessView(
            m_spUAV2,
            NULL,
            &UAVDesc[2],
            m_OnlineDH.hCPU(m_UAV2HeapOffset[m_TestCase]));
    }

    enum TEST_CASE
    {
        TC_DEFAULT = 0,
        TC_0,
        TC_1,
        TC_2,
        TC_3,
        TC_4,
        TC_5,
        TC_6,
        NUM_TC
    };
    TEST_CASE m_TestCase;

private:
    CComPtr<ID3D12CommandAllocator> m_spCommandAllocator;
    CComPtr<ID3D12CommandAllocator> m_spNonRenderingCommandAllocator;

    CComPtr<IDXGISwapChain>      m_spSwapChain;

    CComPtr<ID3D12Device>        m_spDevice12;

    CComPtr<ID3D12CommandQueue>  m_spCommandQueue;
    CComPtr<ID3D12Fence>         m_spFence;
    UINT64                       m_fenceValue;
    HANDLE                       m_hEvent;

    CComPtr<ID3D12GraphicsCommandList>   m_spCommandList;
    CComPtr<ID3D12RootSignature>  m_spRootSignature;
    static const UINT           sc_NumVBVs = 2;
    D3D12_VERTEX_BUFFER_VIEW   m_VBVs[sc_NumVBVs];
    CDescriptorHeapWrapper          m_RTVHeap;
    CDescriptorHeapWrapper          m_OnlineSamplerDH;
    CDescriptorHeapWrapper          m_OnlineDH;

    CComPtr<ID3D12Heap>             m_spUploadBufferHeap;
    CComPtr<ID3D12Resource>         m_spUploadBuffer;
    UINT8*                          m_pDataBegin;
    UINT8*                          m_pDataLwr;
    UINT8*                          m_pDataEnd;
    
    CComPtr<ID3D12Resource>         m_spDefaultBuffer;

    CComPtr<ID3D12Heap>             m_spTextureHeap;

    static const UINT           sc_RTVHeapOffset = 0;
    static const UINT           sc_ElwVBVHeapOffset = 0;
    static const UINT           sc_GeomVBVHeapOffset = 1;
    CComPtr<ID3D12Resource>     m_spDiffuseTexture;
    CComPtr<ID3D12Resource>     m_spBumpTexture;
    CComPtr<ID3D12Resource>     m_spElwTexture;
    CComPtr<ID3D12Resource>     m_spUAV0;
    CComPtr<ID3D12Resource>     m_spUAV1;
    CComPtr<ID3D12Resource>     m_spUAV2;

    static const UINT           sc_NumSamplers = 2;
    static const UINT           sc_NumCBVs = 1;
    static const UINT           sc_NumSRVs = 3;

    UINT           m_DiffuseOnlineHeapOffset[NUM_TC];
    UINT           m_BumpOnlineHeapOffset[NUM_TC];
    UINT           m_ElwOnlineHeapOffset[NUM_TC];
    UINT           m_CBV0HeapOffset[NUM_TC];
    UINT           m_CBV1HeapOffset[NUM_TC];
    UINT           m_UAV0HeapOffset[NUM_TC];
    UINT           m_UAV1HeapOffset[NUM_TC];
    UINT           m_UAV2HeapOffset[NUM_TC];
    UINT           m_SRV0HeapOffsetStart[NUM_TC];
    UINT           m_CBV0HeapOffsetStart[NUM_TC];
    UINT           m_UAV0HeapOffsetStart[NUM_TC];
    UINT           m_SRV1HeapOffsetStart[NUM_TC];
    UINT           m_CBV1HeapOffsetStart[NUM_TC];
    UINT           m_UAV1HeapOffsetStart[NUM_TC];

    // Root signature parameter slots
    static const UINT           sc_SRVTableBindSlot = 0;
    static const UINT           sc_CBTableBindSlot = 1;
    static const UINT           sc_UAVTableBindSlot = 2;
    static const UINT           sc_SRV_CBV_UAV_TableBindSlot = 0;
#if 0
    static const UINT           sc_DiffuseOnlineHeapOffset = 0;
    static const UINT           sc_BumpOnlineHeapOffset = 1;
    static const UINT           sc_ElwOnlineHeapOffset = 2;
    static const UINT           sc_CBV0HeapOffset = 3;
    static const UINT           sc_UAV0HeapOffset = 0;
    static const UINT           sc_UAV1HeapOffset = 0;
    static const UINT           sc_UAV2HeapOffset = 0;

#endif
    static const UINT           sc_NumSRVs0 = 50;
    static const UINT           sc_NumSRVs1 = 50;
    static const UINT           sc_NumCBVs0 = 6;
    static const UINT           sc_NumCBVs1 = 6;

    static const UINT           sc_NumUAVs0 = 4;
    static const UINT           sc_NumUAVs1 = 4;
    
    UINT64                      m_UAVGPUOffset[3];


// TEST_1

    // Root signature parameter slots
    static const UINT           sc1_SRVTableBindSlot0 = 0;
    static const UINT           sc1_SRVTableBindSlot1 = 1;
    static const UINT           sc1_CBTableBindSlot0 = 2;
    static const UINT           sc1_CBTableBindSlot1 = 3;
    static const UINT           sc1_UAVTableBindSlot0 = 4;
    static const UINT           sc1_UAVTableBindSlot1 = 5;

// TEST_3

    //static const UINT           sc_SRVTableBindSlot = 0;
    static const UINT           sc3_CBTableBindSlot0 = 1;
    static const UINT           sc3_CBTableBindSlot1 = 2;
    static const UINT           sc3_CBTableBindSlot2 = 3;
    static const UINT           sc3_UAVTableBindSlot0 = 4;
    static const UINT           sc3_UAVTableBindSlot1 = 5;
    static const UINT           sc3_UAVTableBindSlot2 = 6;

// TEST_4
    //static const UINT           sc_SRV_CBV_UAV_TableBindSlot = 0;
    static const UINT           sc4_CBTableBindSlot0 = 1;
    static const UINT           sc4_CBTableBindSlot1 = 2;
    static const UINT           sc4_UAVTableBindSlot0 = 3;
    static const UINT           sc4_UAVTableBindSlot1 = 4;

// TEST_6
    static const UINT           sc6_CBTableBindSlot0 = 1;
    static const UINT           sc6_CBTableBindSlot1 = 2;
    static const UINT           sc6_UAVTableBindSlot0 = 3;
    static const UINT           sc6_UAVTableBindSlot1 = 4;

    CComPtr<ID3D12PipelineState>      m_spBackgroundPSO;
    CComPtr<ID3D12PipelineState>      m_spSpherePSO;

    UINT m_width;
    UINT m_height;
    
    HWND m_window;

    HANDLE m_previousFrameCompleteEvent;

    float m_rotateAngle;

    bool m_Benchmark;
    UINT m_DumpFrames;
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


