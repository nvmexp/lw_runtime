//--------------------------------------------------------------------------------------
// File: StutterLatencyTestsApp.cpp
//
// This application displays 500 frames with a triangle and logs for 
// Flip Times 
// Time taken by Present api 
// Present to Present time
// 
//Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include <windows.h>
#include <d3d10.h>
#include <d3dx10.h>
#include "resource.h"
#include <stdio.h>
#include "lwapi.h"

#define MAX_QUERY 32
#define DATAPOINTS 1020
#define PASS_CONDITION 4.0
#define IGNORE_DATA_POINTS 20
//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------
struct SimpleVertex
{
    D3DXVECTOR3 Pos;
};

struct QueryStructure
{
    bool dirty;
    ID3D10Query * pQueryTimestamp1, * pQueryTimestamp2,* pQueryTimestamp3, * pQueryDisjoint;
    double CPUCounter, Frame2FrameTime;
};

struct StdDeviation
{
    double GPUTime,CPUTime,Frame2Frame;
};

//--------------------------------------------------------------------------------------
// Global Variables
//--------------------------------------------------------------------------------------
HINSTANCE               g_hInst = NULL;
HWND                    g_hWnd = NULL;
D3D10_DRIVER_TYPE       g_driverType = D3D10_DRIVER_TYPE_NULL;
ID3D10Device*           g_pd3dDevice = NULL;
IDXGISwapChain*         g_pSwapChain = NULL;
ID3D10RenderTargetView* g_pRenderTargetView = NULL;
ID3D10Effect*           g_pEffect = NULL;
ID3D10EffectTechnique*  g_pTechnique = NULL;
ID3D10InputLayout*      g_pVertexLayout = NULL;
ID3D10Buffer*           g_pVertexBuffer = NULL;
struct QueryStructure   g_QueryPointer[MAX_QUERY];
int                     g_MakeQuery = -1, g_GetData = 0;
D3D10_QUERY_DESC        g_queryDescTimestamp,g_queryDescDisjoint;
HANDLE                  g_hThread = NULL;
DWORD                   g_dwThreadId;
double                  g_PCFreq = 0.0;
__int64                 Frame2Frame = 0;
bool                    g_ExitMain = false;
FILE                    *g_fp;
double                  g_MeanQueryTimestamp = 0, g_MeanCPUCounter = 0, g_MeanFrame2Frame = 0;
UINT                    g_Vsync;
int                     g_PASS = 0, g_FAIL = 0, g_Ignore = 0, g_Count=0;
struct StdDeviation     g_StdDev;
LARGE_INTEGER           F2FStart, F2FEnd;


//--------------------------------------------------------------------------------------
// Forward declarations
//--------------------------------------------------------------------------------------
HRESULT InitWindow( HINSTANCE hInstance, int nCmdShow );
HRESULT InitDevice();
void CleanupDevice();
LRESULT CALLBACK    WndProc( HWND, UINT, WPARAM, LPARAM );
void Render();
DWORD WINAPI GetTimestamps(LPVOID lpParam );
void QueryDescInit();
void StartCounter(__int64*);
double GetCounter(__int64*);
void InitializeAndCreateQuery();
void StandardDeviation();
void GenerateResult();

//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI WinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine, int nCmdShow )
{
    char *nextToken = NULL, *token = NULL, fname[30];
    token = strtok_s(lpCmdLine, " ", &nextToken); 
    strcpy_s(fname,token);
    strcat_s(fname, 30, ".txt");
    fopen_s(&g_fp, fname, "w+");
    
    token = strtok_s(NULL, " ", &nextToken);
    g_Vsync = strtol(token, 0, 0);
    
    LARGE_INTEGER li;
    QueryPerformanceFrequency(&li);
    g_PCFreq = double(li.QuadPart)/1000.0;
    F2FStart.QuadPart = 0;

    UNREFERENCED_PARAMETER( hPrevInstance );

    if( FAILED( InitWindow( hInstance, nCmdShow ) ) )
        return 0;

    if( FAILED( InitDevice() ) )
    {
        CleanupDevice();
        return 0;
    }
    
    for(int i=0; i<MAX_QUERY;i++)
        g_QueryPointer[i].dirty=false;

    //Initialize and Create queries   
    InitializeAndCreateQuery();

    g_hThread = CreateThread(NULL, 0, GetTimestamps, NULL, 0, &g_dwThreadId); 
    if(g_hThread == NULL)
    {
        CleanupDevice();
        return 0;
    }

    LwAPI_Status lwapi_status = LWAPI_OK;
    LwPhysicalGpuHandle phys;
    unsigned long cnt;
    lwapi_status = LwAPI_EnumPhysicalGPUs(&phys, &cnt);
    lwapi_status = LwAPI_GPU_SetForcePstateEx(phys,LWAPI_GPU_PERF_PSTATE_P0,LWAPI_GPU_PERF_PSTATE_FALLBACK_HIGHER_PERF,0);

    // Main message loop
    MSG msg = {0};
    while( WM_QUIT != msg.message )
    {
        if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
        {
            TranslateMessage( &msg );
            DispatchMessage( &msg );
        }
        else
        {
            g_Count++;
            if(g_Count >= DATAPOINTS)
                break;
            Render();
        }
    }
    g_ExitMain=true;
    WaitForSingleObject(g_hThread,INFINITE);
    CloseHandle(g_hThread);
    fclose(g_fp);

    fopen_s(&g_fp, fname, "r+");
    GenerateResult();
    
    CleanupDevice();

    return ( int )msg.wParam;
}


//--------------------------------------------------------------------------------------
// Register class and create window
//--------------------------------------------------------------------------------------
HRESULT InitWindow( HINSTANCE hInstance, int nCmdShow )
{
    //Full Screen mode
    DEVMODE dmScreenSettings;
    int screenWidth = 0, screenHeight = 0;
    screenWidth  = GetSystemMetrics(SM_CXSCREEN);
    screenHeight = GetSystemMetrics(SM_CYSCREEN);
    memset(&dmScreenSettings, 0, sizeof(dmScreenSettings));
    dmScreenSettings.dmSize       = sizeof(dmScreenSettings);
    dmScreenSettings.dmPelsWidth  = (unsigned long)screenWidth;
    dmScreenSettings.dmPelsHeight = (unsigned long)screenHeight;
    dmScreenSettings.dmBitsPerPel = 32;
    dmScreenSettings.dmFields     = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;

    // Change the display settings to full screen.
    ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN);

    // Register class
    WNDCLASSEX wcex;
    wcex.cbSize = sizeof( WNDCLASSEX );
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon( hInstance, ( LPCTSTR )IDI_TUTORIAL1 );
    wcex.hLwrsor = LoadLwrsor( NULL, IDC_ARROW );
    wcex.hbrBackground = ( HBRUSH )( COLOR_WINDOW + 1 );
    wcex.lpszMenuName = NULL;
    wcex.lpszClassName = L"StutterLatencyTestsApp";
    wcex.hIconSm = LoadIcon( wcex.hInstance, ( LPCTSTR )IDI_TUTORIAL1 );
    if( !RegisterClassEx( &wcex ) )
        return E_FAIL;

    // Create window
    g_hInst = hInstance;
    RECT rc = { 0, 0, 640, 480 };
    AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );
    g_hWnd = CreateWindow( L"StutterLatencyTestsApp", L"",
                           WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_POPUP,
                           0, 0, screenWidth, screenHeight, NULL, NULL, hInstance,
                           NULL );
    if( !g_hWnd )
        return E_FAIL;

    ShowWindow( g_hWnd, nCmdShow );

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create Direct3D device and swap chain
//--------------------------------------------------------------------------------------
HRESULT InitDevice()
{
    HRESULT hr = S_OK;

    RECT rc;
    GetClientRect( g_hWnd, &rc );
    UINT width = rc.right - rc.left;
    UINT height = rc.bottom - rc.top;

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
   // createDeviceFlags |= D3D10_CREATE_DEVICE_DEBUG;
#endif

    D3D10_DRIVER_TYPE driverTypes[] =
    {
        D3D10_DRIVER_TYPE_HARDWARE,
        D3D10_DRIVER_TYPE_REFERENCE,
    };
    UINT numDriverTypes = sizeof( driverTypes ) / sizeof( driverTypes[0] );

    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory( &sd, sizeof( sd ) );
    sd.BufferCount = 1;
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = g_hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = false;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;


    for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ )
    {
        g_driverType = driverTypes[driverTypeIndex];
        hr = D3D10CreateDeviceAndSwapChain( NULL, g_driverType, NULL, createDeviceFlags,
                                            D3D10_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice );
        if( SUCCEEDED( hr ) )
            break;
    }
    if( FAILED( hr ) )
        return hr;

    // Create a render target view
    ID3D10Texture2D* pBuffer;
    hr = g_pSwapChain->GetBuffer( 0, __uuidof( ID3D10Texture2D ), ( LPVOID* )&pBuffer );
    if( FAILED( hr ) )
        return hr;

    hr = g_pd3dDevice->CreateRenderTargetView( pBuffer, NULL, &g_pRenderTargetView );
    pBuffer->Release();
    if( FAILED( hr ) )
        return hr;

    g_pd3dDevice->OMSetRenderTargets( 1, &g_pRenderTargetView, NULL );

    // Setup the viewport
    D3D10_VIEWPORT vp;
    vp.Width = width;
    vp.Height = height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pd3dDevice->RSSetViewports( 1, &vp );

    // Create the effect
    DWORD dwShaderFlags = D3D10_SHADER_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
    // Set the D3D10_SHADER_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release configuration of this program.
    dwShaderFlags |= D3D10_SHADER_DEBUG;
    #endif
    hr = D3DX10CreateEffectFromFile( L"StutterLatencyTestsApp.fx", NULL, NULL, "fx_4_0", dwShaderFlags, 0,
                                         g_pd3dDevice, NULL, NULL, &g_pEffect, NULL, NULL );
    if( FAILED( hr ) )
    {
        MessageBox( NULL,
                    L"The FX file cannot be located.  Please run this exelwtable from the directory that contains the FX file.", L"Error", MB_OK );
        return hr;
    }

    // Obtain the technique
    g_pTechnique = g_pEffect->GetTechniqueByName( "Render" );

    // Define the input layout
    D3D10_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 },
    };
    UINT numElements = sizeof( layout ) / sizeof( layout[0] );

    // Create the input layout
    D3D10_PASS_DESC PassDesc;
    g_pTechnique->GetPassByIndex( 0 )->GetDesc( &PassDesc );
    hr = g_pd3dDevice->CreateInputLayout( layout, numElements, PassDesc.pIAInputSignature,
                                          PassDesc.IAInputSignatureSize, &g_pVertexLayout );
    if( FAILED( hr ) )
        return hr;

    // Set the input layout
    g_pd3dDevice->IASetInputLayout( g_pVertexLayout );

    // Create vertex buffer
    SimpleVertex vertices[] =
    {
        D3DXVECTOR3( 0.0f, 0.5f, 0.5f ),
        D3DXVECTOR3( 0.5f, -0.5f, 0.5f ),
        D3DXVECTOR3( -0.5f, -0.5f, 0.5f ),
    };
    D3D10_BUFFER_DESC bd;
    bd.Usage = D3D10_USAGE_DEFAULT;
    bd.ByteWidth = sizeof( SimpleVertex ) * 3;
    bd.BindFlags = D3D10_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = 0;
    bd.MiscFlags = 0;
    D3D10_SUBRESOURCE_DATA InitData;
    InitData.pSysMem = vertices;
    hr = g_pd3dDevice->CreateBuffer( &bd, &InitData, &g_pVertexBuffer );
    if( FAILED( hr ) )
        return hr;

    // Set vertex buffer
    UINT stride = sizeof( SimpleVertex );
    UINT offset = 0;
    g_pd3dDevice->IASetVertexBuffers( 0, 1, &g_pVertexBuffer, &stride, &offset );

    // Set primitive topology
    g_pd3dDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Clean up the objects we've created
//--------------------------------------------------------------------------------------
void CleanupDevice()
{
    if( g_pd3dDevice ) g_pd3dDevice->ClearState();

    if( g_pVertexBuffer ) g_pVertexBuffer->Release();
    if( g_pVertexLayout ) g_pVertexLayout->Release();
    if( g_pEffect ) g_pEffect->Release();
    if( g_pRenderTargetView ) g_pRenderTargetView->Release();
    if( g_pSwapChain ) g_pSwapChain->Release();
    if( g_pd3dDevice ) g_pd3dDevice->Release();
}


//--------------------------------------------------------------------------------------
// Called every time the application receives a message
//--------------------------------------------------------------------------------------
LRESULT CALLBACK WndProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam )
{
    PAINTSTRUCT ps;
    HDC hdc;

    switch( message )
    {
        case WM_PAINT:
            hdc = BeginPaint( hWnd, &ps );
            EndPaint( hWnd, &ps );
            break;

        case WM_DESTROY:
            PostQuitMessage( 0 );
            break;

        default:
            return DefWindowProc( hWnd, message, wParam, lParam );
    }

    return 0;
}


//--------------------------------------------------------------------------------------
// Render a frame
//--------------------------------------------------------------------------------------
void Render()
{
    g_MakeQuery++;
    g_MakeQuery %= MAX_QUERY;

    // Clear the back buffer 
    float ClearColor[4] = { 1.0f, 0.125f, 0.3f, 1.0f }; // red,green,blue,alpha
    g_pd3dDevice->ClearRenderTargetView( g_pRenderTargetView, ClearColor );
    LARGE_INTEGER CPUStart, CPUEnd;
    
    // Render a triangle
    D3D10_TECHNIQUE_DESC techDesc;
    g_pTechnique->GetDesc( &techDesc );

    if(!g_QueryPointer[g_MakeQuery].dirty)
    {
        QueryPerformanceCounter(&F2FEnd);
        g_QueryPointer[g_MakeQuery].Frame2FrameTime = (F2FEnd.QuadPart - F2FStart.QuadPart)/g_PCFreq;
        QueryPerformanceCounter(&F2FStart);

        if( g_QueryPointer[g_MakeQuery].pQueryDisjoint && 
            g_QueryPointer[g_MakeQuery].pQueryTimestamp1 && 
            g_QueryPointer[g_MakeQuery].pQueryTimestamp2 &&
            g_QueryPointer[g_MakeQuery].pQueryTimestamp3)
        {
            //Query Disjoint Begin
            g_QueryPointer[g_MakeQuery].pQueryDisjoint->Begin();

            //First GPU Timestamp
            g_QueryPointer[g_MakeQuery].pQueryTimestamp1->End();

            for( UINT p = 0; p < techDesc.Passes; ++p )
            {
                g_pTechnique->GetPassByIndex( p )->Apply( 0 );
                g_pd3dDevice->Draw( 3, 0 );
            }

            //Second GPU Timestamp
            g_QueryPointer[g_MakeQuery].pQueryTimestamp2->End();

            //CPU Counter Start
            QueryPerformanceCounter(&CPUStart);
            
            g_pSwapChain->Present( g_Vsync, 0 );
            
            QueryPerformanceCounter(&CPUEnd);
            
            //CPU Counter Stop
            g_QueryPointer[g_MakeQuery].CPUCounter = (CPUEnd.QuadPart - CPUStart.QuadPart)/g_PCFreq;
            
            //Third GPU Timestamp
            g_QueryPointer[g_MakeQuery].pQueryTimestamp3->End();
            
            //Query Disjoint End
            g_QueryPointer[g_MakeQuery].pQueryDisjoint->End();
            
            g_QueryPointer[g_MakeQuery].dirty = true;
            
            
        }
    }
    else
        g_Count--;
        

}

void InitializeAndCreateQuery()
{
    QueryDescInit();
    for(int i=0;i<MAX_QUERY;i++)
    {
        g_pd3dDevice->CreateQuery(&g_queryDescDisjoint,&(g_QueryPointer[i].pQueryDisjoint));
        g_pd3dDevice->CreateQuery(&g_queryDescTimestamp,&(g_QueryPointer[i].pQueryTimestamp1));
        g_pd3dDevice->CreateQuery(&g_queryDescTimestamp,&(g_QueryPointer[i].pQueryTimestamp2));
        g_pd3dDevice->CreateQuery(&g_queryDescTimestamp,&(g_QueryPointer[i].pQueryTimestamp3));
    }
}

DWORD WINAPI GetTimestamps(LPVOID lpParam)
{
    UNREFERENCED_PARAMETER(lpParam);	
    UINT64 queryTimestampData1, queryTimestampData2, queryTimestampData3;
    double compGPU = 0, compCPU = 0, compFrame2Frame = 0;
    D3D10_QUERY_DATA_TIMESTAMP_DISJOINT queryDisjointData;
    
    while(true)
    {
        if(g_QueryPointer[g_GetData].dirty)
        {
            while(S_OK != g_QueryPointer[g_GetData].pQueryDisjoint->GetData( &queryDisjointData, sizeof(D3D10_QUERY_DATA_TIMESTAMP_DISJOINT),0) )
            {}
        
            while(S_OK != g_QueryPointer[g_GetData].pQueryTimestamp1->GetData( &queryTimestampData1, sizeof(UINT64),0) )
            {}
            
            while(S_OK != g_QueryPointer[g_GetData].pQueryTimestamp2->GetData( &queryTimestampData2, sizeof(UINT64),0) )
            {}

            while(S_OK != g_QueryPointer[g_GetData].pQueryTimestamp3->GetData( &queryTimestampData3, sizeof(UINT64),0) )
            {}
                
                            
            if(!queryDisjointData.Disjoint)
            {			
                //GPU Timestamp(time taken by GPU to flip the buffer)
                compGPU = (queryTimestampData3 - queryTimestampData1) * 1000.0 / queryDisjointData.Frequency;
                //CPU Timestamp(Time taken for Present api)
                compCPU = g_QueryPointer[g_GetData].CPUCounter;
                //Frame to Frame Timestamp
                compFrame2Frame = g_QueryPointer[g_GetData].Frame2FrameTime;

                if(g_Ignore >= IGNORE_DATA_POINTS )//Ignoring first IGNORE_DATA_POINTS datapoints to allow the system to stabalize
                {
                    fprintf_s(g_fp,"%f\t%f\t%f\n" ,compGPU, compCPU,compFrame2Frame );
                    
                    g_MeanQueryTimestamp += compGPU;
                    g_MeanCPUCounter += compCPU;
                    g_MeanFrame2Frame += compFrame2Frame;
                }
                else
                    g_Ignore++;
            }
        
            g_QueryPointer[g_GetData].dirty = false;
            g_GetData++;
            g_GetData %= MAX_QUERY;
        }
        else if(g_ExitMain)//Signal from main thread to exit
            break;
    }
    
    return 0;
}

void QueryDescInit()
{
    g_queryDescTimestamp.Query = D3D10_QUERY_TIMESTAMP;
    g_queryDescTimestamp.MiscFlags = 0;
    
    g_queryDescDisjoint.Query = D3D10_QUERY_TIMESTAMP_DISJOINT;
    g_queryDescDisjoint.MiscFlags = 0;
}

void StandardDeviation()
{
    double GPU,CPU,F2F;
    char read[50];
    double sumGPU = 0,sumCPU = 0,sumF2F = 0;
    int r=0;
    for(int i=0;i < DATAPOINTS - IGNORE_DATA_POINTS - 1 ;i++)
    {
        r = fscanf_s(g_fp,"%s",read,_countof(read));
        GPU = strtod(read,0);
        if( GPU <= 2 * g_MeanQueryTimestamp)
        {
            sumGPU += (g_MeanQueryTimestamp - GPU) * (g_MeanQueryTimestamp - GPU);
        }
        else
        {
            sumGPU += 4 * g_MeanQueryTimestamp * g_MeanQueryTimestamp;
        }
        r = fscanf_s(g_fp,"%s",read,_countof(read));
        CPU = strtod(read,0);
        if( CPU <= 2 * g_MeanCPUCounter)
        {
            sumCPU += (g_MeanCPUCounter - CPU) * (g_MeanCPUCounter - CPU);
        }
        else
        {
            sumCPU += 4 * g_MeanCPUCounter * g_MeanCPUCounter;
        }
        r = fscanf_s(g_fp,"%s",read,_countof(read));
        F2F = strtod(read,0);
        if( F2F <= 2 * g_MeanFrame2Frame)
        {
            sumF2F += (g_MeanFrame2Frame - F2F) * (g_MeanFrame2Frame - F2F);
        }
        else
        {
            sumF2F += 4 * g_MeanFrame2Frame * g_MeanFrame2Frame;
        }
    }
    g_StdDev.CPUTime = sqrt(sumCPU/ (DATAPOINTS - IGNORE_DATA_POINTS)) ;
    g_StdDev.GPUTime = sqrt(sumGPU/ (DATAPOINTS - IGNORE_DATA_POINTS)) ;
    g_StdDev.Frame2Frame = sqrt(sumF2F/ (DATAPOINTS - IGNORE_DATA_POINTS));	
}

void GenerateResult()
{
    g_MeanQueryTimestamp = g_MeanQueryTimestamp/(DATAPOINTS - IGNORE_DATA_POINTS);
    g_MeanCPUCounter = g_MeanCPUCounter/(DATAPOINTS - IGNORE_DATA_POINTS);
    g_MeanFrame2Frame =	g_MeanFrame2Frame/(DATAPOINTS - IGNORE_DATA_POINTS);
    
    StandardDeviation();

    fclose(g_fp);
    
    FILE *fp;
    fopen_s(&fp,"result.txt", "a+");

    if(g_Vsync)
        fprintf_s(fp, "Vsync On\n");
    else
        fprintf_s(fp, "Vsync Off\n");

    fprintf_s(fp, "Number of Test Cases : %d\n", DATAPOINTS);
    fprintf_s(fp, "MeanGPU = %f\nMeanCPU = %f\nMeanFram2Frame = %f\n", g_MeanQueryTimestamp, g_MeanCPUCounter, g_MeanFrame2Frame);
    fprintf_s(fp, "StdDevGPU = %f\nStdDevCPU = %f\nStdDevFram2Frame = %f\n", g_StdDev.GPUTime, g_StdDev.CPUTime, g_StdDev.Frame2Frame);
    if(g_StdDev.GPUTime < PASS_CONDITION && g_StdDev.CPUTime < PASS_CONDITION && g_StdDev.Frame2Frame < PASS_CONDITION)
        fprintf_s(fp, "PASS\n\n");
    else
        fprintf_s(fp, "FAIL\n\n");

    fclose(fp);
}