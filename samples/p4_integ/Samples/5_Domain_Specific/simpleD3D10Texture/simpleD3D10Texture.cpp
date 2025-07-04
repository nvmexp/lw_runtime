/* Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This example demonstrates how to use the LWCA Direct3D bindings to
 * transfer data between LWCA and DX9 2D, LwbeMap, and Volume Textures.
 */

#pragma warning(disable : 4312)

#include <windows.h>
#include <mmsystem.h>

// This header inclues all the necessary D3D10 and LWCA includes
#include <dynlink_d3d10.h>
#include <lwda_runtime_api.h>
#include <lwda_d3d10_interop.h>

// includes, project
#include <rendercheck_d3d10.h>
#include <helper_lwda.h>  // helper functions for LWCA error checking and initialization

#define MAX_EPSILON 10

static char *SDK_name = "simpleD3D10Texture";

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
IDXGIAdapter *g_pLwdaCapableAdapter = NULL;  // Adapter to use
ID3D10Device *g_pd3dDevice = NULL;           // Our rendering device
IDXGISwapChain *g_pSwapChain = NULL;         // The swap chain of the window
ID3D10RenderTargetView *g_pSwapChainRTV =
    NULL;  // The Render target view on the swap chain ( used for clear)
ID3D10RasterizerState *g_pRasterState = NULL;

ID3D10InputLayout *g_pInputLayout = NULL;
ID3D10Effect *g_pSimpleEffect = NULL;
ID3D10EffectTechnique *g_pSimpleTechnique = NULL;
ID3D10EffectVectorVariable *g_pvQuadRect = NULL;
ID3D10EffectScalarVariable *g_pUseCase = NULL;
ID3D10EffectShaderResourceVariable *g_pTexture2D = NULL;
ID3D10EffectShaderResourceVariable *g_pTexture3D = NULL;
ID3D10EffectShaderResourceVariable *g_pTextureLwbe = NULL;

static const char g_simpleEffectSrc[] =
    "float4 g_vQuadRect; \n"
    "int g_UseCase; \n"
    "Texture2D g_Texture2D; \n"
    "Texture3D g_Texture3D; \n"
    "TextureLwbe g_TextureLwbe; \n"
    "\n"
    "SamplerState samLinear{ \n"
    "    Filter = MIN_MAG_LINEAR_MIP_POINT; \n"
    "};\n"
    "\n"
    "struct Fragment{ \n"
    "    float4 Pos : SV_POSITION;\n"
    "    float3 Tex : TEXCOORD0; };\n"
    "\n"
    "Fragment VS( uint vertexId : SV_VertexID )\n"
    "{\n"
    "    Fragment f;\n"
    "    f.Tex = float3( 0.f, 0.f, 0.f); \n"
    "    if (vertexId == 1) f.Tex.x = 1.f; \n"
    "    else if (vertexId == 2) f.Tex.y = 1.f; \n"
    "    else if (vertexId == 3) f.Tex.xy = float2(1.f, 1.f); \n"
    "    \n"
    "    f.Pos = float4( g_vQuadRect.xy + f.Tex * g_vQuadRect.zw, 0, 1);\n"
    "    \n"
    "    if (g_UseCase == 1) { \n"
    "        if (vertexId == 1) f.Tex.z = 0.5f; \n"
    "        else if (vertexId == 2) f.Tex.z = 0.5f; \n"
    "        else if (vertexId == 3) f.Tex.z = 1.f; \n"
    "    } \n"
    "    else if (g_UseCase >= 2) { \n"
    "        f.Tex.xy = f.Tex.xy * 2.f - 1.f; \n"
    "    } \n"
    "    return f;\n"
    "}\n"
    "\n"
    "float4 PS( Fragment f ) : SV_Target\n"
    "{\n"
    "    if (g_UseCase == 0) return g_Texture2D.Sample( samLinear, f.Tex.xy ); "
    "\n"
    "    else if (g_UseCase == 1) return g_Texture3D.Sample( samLinear, f.Tex "
    "); \n"
    "    else if (g_UseCase == 2) return g_TextureLwbe.Sample( samLinear, "
    "float3(f.Tex.xy, 1.0) ); \n"
    "    else if (g_UseCase == 3) return g_TextureLwbe.Sample( samLinear, "
    "float3(f.Tex.xy, -1.0) ); \n"
    "    else if (g_UseCase == 4) return g_TextureLwbe.Sample( samLinear, "
    "float3(1.0, f.Tex.xy) ); \n"
    "    else if (g_UseCase == 5) return g_TextureLwbe.Sample( samLinear, "
    "float3(-1.0, f.Tex.xy) ); \n"
    "    else if (g_UseCase == 6) return g_TextureLwbe.Sample( samLinear, "
    "float3(f.Tex.x, 1.0, f.Tex.y) ); \n"
    "    else if (g_UseCase == 7) return g_TextureLwbe.Sample( samLinear, "
    "float3(f.Tex.x, -1.0, f.Tex.y) ); \n"
    "    else return float4(f.Tex, 1);\n"
    "}\n"
    "\n"
    "technique10 Render\n"
    "{\n"
    "    pass P0\n"
    "    {\n"
    "        SetVertexShader( CompileShader( vs_4_0, VS() ) );\n"
    "        SetGeometryShader( NULL );\n"
    "        SetPixelShader( CompileShader( ps_4_0, PS() ) );\n"
    "    }\n"
    "}\n"
    "\n";

// testing/tracing function used pervasively in tests.  if the condition is
// unsatisfied
// then spew and fail the function immediately (doing no cleanup)
#define AssertOrQuit(x)                                                  \
  if (!(x)) {                                                            \
    fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, \
            __FILE__, __LINE__);                                         \
    return 1;                                                            \
  }

bool g_bDone = false;
bool g_bPassed = true;

const unsigned int g_WindowWidth = 720;
const unsigned int g_WindowHeight = 720;

int g_iFrameToCompare = 10;

int *pArgc = NULL;
char **pArgv = NULL;

// Data structure for 2D texture shared between DX10 and LWCA
struct {
  ID3D10Texture2D *pTexture;
  ID3D10ShaderResourceView *pSRView;
  lwdaGraphicsResource *lwdaResource;
  void *lwdaLinearMemory;
  size_t pitch;
  int width;
  int height;
} g_texture_2d;

// Data structure for volume textures shared between DX10 and LWCA
struct {
  ID3D10Texture3D *pTexture;
  ID3D10ShaderResourceView *pSRView;
  lwdaGraphicsResource *lwdaResource;
  void *lwdaLinearMemory;
  size_t pitch;
  int width;
  int height;
  int depth;
} g_texture_3d;

// Data structure for lwbe texture shared between DX10 and LWCA
struct {
  ID3D10Texture2D *pTexture;
  ID3D10ShaderResourceView *pSRView;
  lwdaGraphicsResource *lwdaResource;
  void *lwdaLinearMemory;
  size_t pitch;
  int size;
} g_texture_lwbe;

// The LWCA kernel launchers that get called
extern "C" {
bool lwda_texture_2d(void *surface, size_t width, size_t height, size_t pitch,
                     float t);
bool lwda_texture_3d(void *surface, int width, int height, int depth,
                     size_t pitch, size_t pitchslice, float t);
bool lwda_texture_lwbe(void *surface, int width, int height, size_t pitch,
                       int face, float t);
}

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd);
HRESULT InitTextures();

void RunKernels();
void DrawScene();
void Cleanup();
void Render();

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

#define NAME_LEN 512

bool findLWDADevice() {
  int nGraphicsGPU = 0;
  int deviceCount = 0;
  bool bFoundGraphics = false;
  char devname[NAME_LEN];

  // This function call returns 0 if there are no LWCA capable devices.
  lwdaError_t error_id = lwdaGetDeviceCount(&deviceCount);

  if (error_id != lwdaSuccess) {
    printf("lwdaGetDeviceCount returned %d\n-> %s\n", (int)error_id,
           lwdaGetErrorString(error_id));
    exit(EXIT_FAILURE);
  }

  if (deviceCount == 0) {
    printf("> There are no device(s) supporting LWCA\n");
    return false;
  } else {
    printf("> Found %d LWCA Capable Device(s)\n", deviceCount);
  }

  // Get LWCA device properties
  lwdaDeviceProp deviceProp;

  for (int dev = 0; dev < deviceCount; ++dev) {
    lwdaGetDeviceProperties(&deviceProp, dev);
    STRCPY(devname, NAME_LEN, deviceProp.name);
    printf("> GPU %d: %s\n", dev, devname);
  }

  return true;
}

bool findDXDevice(char *dev_name) {
  HRESULT hr = S_OK;
  lwdaError lwStatus;

  // Iterate through the candidate adapters
  IDXGIFactory *pFactory;
  hr = sFnPtr_CreateDXGIFactory(__uuidof(IDXGIFactory), (void **)(&pFactory));

  if (!SUCCEEDED(hr)) {
    printf("> No DXGI Factory created.\n");
    return false;
  }

  UINT adapter = 0;

  for (; !g_pLwdaCapableAdapter; ++adapter) {
    // Get a candidate DXGI adapter
    IDXGIAdapter *pAdapter = NULL;
    hr = pFactory->EnumAdapters(adapter, &pAdapter);

    if (FAILED(hr)) {
      break;  // no compatible adapters found
    }

    // Query to see if there exists a corresponding compute device
    int lwDevice;
    lwStatus = lwdaD3D10GetDevice(&lwDevice, pAdapter);
    printLastLwdaError("lwdaD3D10GetDevice failed");  // This prints and resets
                                                      // the lwdaError to
                                                      // lwdaSuccess

    if (lwdaSuccess == lwStatus) {
      // If so, mark it as the one against which to create our d3d10 device
      g_pLwdaCapableAdapter = pAdapter;
      g_pLwdaCapableAdapter->AddRef();
    }

    pAdapter->Release();
  }

  printf("> Found %d D3D10 Adapater(s).\n", (int)adapter);

  pFactory->Release();

  if (!g_pLwdaCapableAdapter) {
    printf("> Found 0 D3D10 Adapater(s) /w Compute capability.\n");
    return false;
  }

  DXGI_ADAPTER_DESC adapterDesc;
  g_pLwdaCapableAdapter->GetDesc(&adapterDesc);
  wcstombs_s(NULL, dev_name, 256, adapterDesc.Description, 128);

  printf("> Found 1 D3D10 Adapater(s) /w Compute capability.\n");
  printf("> %s\n", dev_name);

  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  char device_name[256];
  char *ref_file = NULL;

  pArgc = &argc;
  pArgv = argv;

  printf("[%s] - Starting...\n", SDK_name);

  if (!findLWDADevice())  // Search for LWCA GPU
  {
    printf("> LWCA Device NOT found on \"%s\".. Exiting.\n", device_name);
    exit(EXIT_SUCCESS);
  }

  // Search for D3D API (locate drivers, does not mean device is found)
  if (!dynlinkLoadD3D10API()) {
    printf("> D3D10 API libraries NOT found on.. Exiting.\n");
    dynlinkUnloadD3D10API();
    exit(EXIT_SUCCESS);
  }

  if (!findDXDevice(device_name)) {  // Search for D3D Hardware Device
    printf("> D3D10 Graphics Device NOT found.. Exiting.\n");
    dynlinkUnloadD3D10API();
    exit(EXIT_SUCCESS);
  }

  // command line options
  if (argc > 1) {
    // automatied build testing harness
    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
      getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
  }

//
// create window
//
// Register the window class
#if 1
  WNDCLASSEX wc = {sizeof(WNDCLASSEX),
                   CS_CLASSDC,
                   MsgProc,
                   0L,
                   0L,
                   GetModuleHandle(NULL),
                   NULL,
                   NULL,
                   NULL,
                   NULL,
                   "LWCA SDK",
                   NULL};
  RegisterClassEx(&wc);

  // Create the application's window
  int xBorder = ::GetSystemMetrics(SM_CXSIZEFRAME);
  int yMenu = ::GetSystemMetrics(SM_CYMENU);
  int yBorder = ::GetSystemMetrics(SM_CYSIZEFRAME);
  HWND hWnd = CreateWindow(
      wc.lpszClassName, "LWCA/D3D10 Texture InterOP", WS_OVERLAPPEDWINDOW, 0, 0,
      g_WindowWidth + 2 * xBorder, g_WindowHeight + 2 * yBorder + yMenu, NULL,
      NULL, wc.hInstance, NULL);
#else
  static WNDCLASSEX wc = {
      sizeof(WNDCLASSEX),    CS_CLASSDC, MsgProc, 0L,   0L,
      GetModuleHandle(NULL), NULL,       NULL,    NULL, NULL,
      "LwdaD3D9Tex",         NULL};
  RegisterClassEx(&wc);
  HWND hWnd = CreateWindow("LwdaD3D9Tex", "LWCA D3D9 Texture Interop",
                           WS_OVERLAPPEDWINDOW, 0, 0, 800, 320,
                           GetDesktopWindow(), NULL, wc.hInstance, NULL);
#endif

  ShowWindow(hWnd, SW_SHOWDEFAULT);
  UpdateWindow(hWnd);

  // Initialize Direct3D
  if (SUCCEEDED(InitD3D(hWnd)) && SUCCEEDED(InitTextures())) {
    // 2D
    // register the Direct3D resources that we'll use
    // we'll read to and write from g_texture_2d, so don't set any special map
    // flags for it
    lwdaGraphicsD3D10RegisterResource(&g_texture_2d.lwdaResource,
                                      g_texture_2d.pTexture,
                                      lwdaGraphicsRegisterFlagsNone);
    getLastLwdaError("lwdaGraphicsD3D10RegisterResource (g_texture_2d) failed");
    // lwca cannot write into the texture directly : the texture is seen as a
    // lwdaArray and can only be mapped as a texture
    // Create a buffer so that lwca can write into it
    // pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
    lwdaMallocPitch(&g_texture_2d.lwdaLinearMemory, &g_texture_2d.pitch,
                    g_texture_2d.width * sizeof(float) * 4,
                    g_texture_2d.height);
    getLastLwdaError("lwdaMallocPitch (g_texture_2d) failed");
    lwdaMemset(g_texture_2d.lwdaLinearMemory, 1,
               g_texture_2d.pitch * g_texture_2d.height);

    // LWBE
    lwdaGraphicsD3D10RegisterResource(&g_texture_lwbe.lwdaResource,
                                      g_texture_lwbe.pTexture,
                                      lwdaGraphicsRegisterFlagsNone);
    getLastLwdaError(
        "lwdaGraphicsD3D10RegisterResource (g_texture_lwbe) failed");
    // create the buffer. pixel fmt is DXGI_FORMAT_R8G8B8A8_SNORM
    lwdaMallocPitch(&g_texture_lwbe.lwdaLinearMemory, &g_texture_lwbe.pitch,
                    g_texture_lwbe.size * 4, g_texture_lwbe.size);
    getLastLwdaError("lwdaMallocPitch (g_texture_lwbe) failed");
    lwdaMemset(g_texture_lwbe.lwdaLinearMemory, 1,
               g_texture_lwbe.pitch * g_texture_lwbe.size);
    getLastLwdaError("lwdaMemset (g_texture_lwbe) failed");

    // 3D
    lwdaGraphicsD3D10RegisterResource(&g_texture_3d.lwdaResource,
                                      g_texture_3d.pTexture,
                                      lwdaGraphicsRegisterFlagsNone);
    getLastLwdaError("lwdaGraphicsD3D10RegisterResource (g_texture_3d) failed");
    // create the buffer. pixel fmt is DXGI_FORMAT_R8G8B8A8_SNORM
    // lwdaMallocPitch(&g_texture_3d.lwdaLinearMemory, &g_texture_3d.pitch,
    // g_texture_3d.width * 4, g_texture_3d.height * g_texture_3d.depth);
    lwdaMalloc(
        &g_texture_3d.lwdaLinearMemory,
        g_texture_3d.width * 4 * g_texture_3d.height * g_texture_3d.depth);
    g_texture_3d.pitch = g_texture_3d.width * 4;
    getLastLwdaError("lwdaMallocPitch (g_texture_3d) failed");
    lwdaMemset(g_texture_3d.lwdaLinearMemory, 1,
               g_texture_3d.pitch * g_texture_3d.height * g_texture_3d.depth);
    getLastLwdaError("lwdaMemset (g_texture_3d) failed");
  } else {
    printf("> WARNING: No D3D10 Device found.\n");
    g_bPassed = true;
    exit(g_bPassed ? EXIT_SUCCESS : EXIT_FAILURE);
  }

  //
  // the main loop
  //
  while (false == g_bDone) {
    Render();

    //
    // handle I/O
    //
    MSG msg;
    ZeroMemory(&msg, sizeof(msg));

    while (msg.message != WM_QUIT) {
      if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
      } else {
        Render();

        if (ref_file) {
          for (int count = 0; count < g_iFrameToCompare; count++) {
            Render();
          }

          const char *lwr_image_path = "simpleD3D10Texture.ppm";

          // Save a reference of our current test run image
          CheckRenderD3D10::ActiveRenderTargetToPPM(g_pd3dDevice,
                                                    lwr_image_path);

          // compare to offical reference image, printing PASS or FAIL.
          g_bPassed = CheckRenderD3D10::PPMvsPPM(lwr_image_path, ref_file,
                                                 argv[0], MAX_EPSILON, 0.15f);

          g_bDone = true;

          Cleanup();

          PostQuitMessage(0);
        } else {
          g_bPassed = true;
        }
      }
    }
  };

  // Release D3D Library (after message loop)
  dynlinkUnloadD3D10API();

  // Unregister windows class
  UnregisterClass(wc.lpszClassName, wc.hInstance);

  //
  // and exit
  //
  printf("> %s running on %s exiting...\n", SDK_name, device_name);

  exit(g_bPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}

//-----------------------------------------------------------------------------
// Name: InitD3D()
// Desc: Initializes Direct3D
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd) {
  // Set up the structure used to create the device and swapchain
  DXGI_SWAP_CHAIN_DESC sd;
  ZeroMemory(&sd, sizeof(sd));
  sd.BufferCount = 1;
  sd.BufferDesc.Width = g_WindowWidth;
  sd.BufferDesc.Height = g_WindowHeight;
  sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  sd.BufferDesc.RefreshRate.Numerator = 60;
  sd.BufferDesc.RefreshRate.Denominator = 1;
  sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  sd.OutputWindow = hWnd;
  sd.SampleDesc.Count = 1;
  sd.SampleDesc.Quality = 0;
  sd.Windowed = TRUE;

  // Create device and swapchain
  HRESULT hr = sFnPtr_D3D10CreateDeviceAndSwapChain(
      g_pLwdaCapableAdapter, D3D10_DRIVER_TYPE_HARDWARE, NULL, 0,
      D3D10_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice);
  AssertOrQuit(SUCCEEDED(hr));
  g_pLwdaCapableAdapter->Release();

  // Create a render target view of the swapchain
  ID3D10Texture2D *pBuffer;
  hr =
      g_pSwapChain->GetBuffer(0, __uuidof(ID3D10Texture2D), (LPVOID *)&pBuffer);
  AssertOrQuit(SUCCEEDED(hr));

  hr = g_pd3dDevice->CreateRenderTargetView(pBuffer, NULL, &g_pSwapChainRTV);
  AssertOrQuit(SUCCEEDED(hr));
  pBuffer->Release();

  g_pd3dDevice->OMSetRenderTargets(1, &g_pSwapChainRTV, NULL);

  // Setup the viewport
  D3D10_VIEWPORT vp;
  vp.Width = g_WindowWidth;
  vp.Height = g_WindowHeight;
  vp.MinDepth = 0.0f;
  vp.MaxDepth = 1.0f;
  vp.TopLeftX = 0;
  vp.TopLeftY = 0;
  g_pd3dDevice->RSSetViewports(1, &vp);

  // Setup the effect
  {
    ID3D10Blob *pCompiledEffect;
    ID3D10Blob *pErrors = NULL;
    hr = sFnPtr_D3D10CompileEffectFromMemory((void *)g_simpleEffectSrc,
                                             sizeof(g_simpleEffectSrc), NULL,
                                             NULL,  // pDefines
                                             NULL,  // pIncludes
                                             0,     // HLSL flags
                                             0,     // FXFlags
                                             &pCompiledEffect, &pErrors);

    if (pErrors) {
      LPVOID l_pError = NULL;
      l_pError = pErrors->GetBufferPointer();  // then cast to a char* to see it
                                               // in the locals window
      fprintf(stdout, "Compilation error: \n %s", (char *)l_pError);
    }

    AssertOrQuit(SUCCEEDED(hr));

    hr = sFnPtr_D3D10CreateEffectFromMemory(
        pCompiledEffect->GetBufferPointer(), pCompiledEffect->GetBufferSize(),
        0,  // FXFlags
        g_pd3dDevice, NULL, &g_pSimpleEffect);
    pCompiledEffect->Release();

    g_pSimpleTechnique = g_pSimpleEffect->GetTechniqueByName("Render");

    g_pvQuadRect =
        g_pSimpleEffect->GetVariableByName("g_vQuadRect")->AsVector();
    g_pUseCase = g_pSimpleEffect->GetVariableByName("g_UseCase")->AsScalar();

    g_pTexture2D =
        g_pSimpleEffect->GetVariableByName("g_Texture2D")->AsShaderResource();
    g_pTexture3D =
        g_pSimpleEffect->GetVariableByName("g_Texture3D")->AsShaderResource();
    g_pTextureLwbe =
        g_pSimpleEffect->GetVariableByName("g_TextureLwbe")->AsShaderResource();

    // Setup  no Input Layout
    g_pd3dDevice->IASetInputLayout(0);
    g_pd3dDevice->IASetPrimitiveTopology(
        D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
  }

  D3D10_RASTERIZER_DESC rasterizerState;
  rasterizerState.FillMode = D3D10_FILL_SOLID;
  rasterizerState.LwllMode = D3D10_LWLL_FRONT;
  rasterizerState.FrontCounterClockwise = false;
  rasterizerState.DepthBias = false;
  rasterizerState.DepthBiasClamp = 0;
  rasterizerState.SlopeScaledDepthBias = 0;
  rasterizerState.DepthClipEnable = false;
  rasterizerState.ScissorEnable = false;
  rasterizerState.MultisampleEnable = false;
  rasterizerState.AntialiasedLineEnable = false;
  g_pd3dDevice->CreateRasterizerState(&rasterizerState, &g_pRasterState);
  g_pd3dDevice->RSSetState(g_pRasterState);

  return S_OK;
}

//-----------------------------------------------------------------------------
// Name: InitTextures()
// Desc: Initializes Direct3D Textures (allocation and initialization)
//-----------------------------------------------------------------------------
HRESULT InitTextures() {
  //
  // create the D3D resources we'll be using
  //
  // 2D texture
  {
    g_texture_2d.width = 256;
    g_texture_2d.height = 256;

    D3D10_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(D3D10_TEXTURE2D_DESC));
    desc.Width = g_texture_2d.width;
    desc.Height = g_texture_2d.height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D10_USAGE_DEFAULT;
    desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;

    if (FAILED(
            g_pd3dDevice->CreateTexture2D(&desc, NULL, &g_texture_2d.pTexture)))
      return E_FAIL;

    if (FAILED(g_pd3dDevice->CreateShaderResourceView(
            g_texture_2d.pTexture, NULL, &g_texture_2d.pSRView)))
      return E_FAIL;

    g_pTexture2D->SetResource(g_texture_2d.pSRView);
  }

  // 3D texture
  {
    g_texture_3d.width = 64;
    g_texture_3d.height = 64;
    g_texture_3d.depth = 64;

    D3D10_TEXTURE3D_DESC desc;
    ZeroMemory(&desc, sizeof(D3D10_TEXTURE3D_DESC));
    desc.Width = g_texture_3d.width;
    desc.Height = g_texture_3d.height;
    desc.Depth = g_texture_3d.depth;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_SNORM;
    desc.Usage = D3D10_USAGE_DEFAULT;
    desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;

    if (FAILED(
            g_pd3dDevice->CreateTexture3D(&desc, NULL, &g_texture_3d.pTexture)))
      return E_FAIL;

    if (FAILED(g_pd3dDevice->CreateShaderResourceView(
            g_texture_3d.pTexture, NULL, &g_texture_3d.pSRView)))
      return E_FAIL;

    g_pTexture3D->SetResource(g_texture_3d.pSRView);
  }

  // lwbe texture
  {
    g_texture_lwbe.size = 64;

    D3D10_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(D3D10_TEXTURE2D_DESC));
    desc.Width = g_texture_lwbe.size;
    desc.Height = g_texture_lwbe.size;
    desc.MipLevels = 1;
    desc.ArraySize = 6;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D10_USAGE_DEFAULT;
    desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D10_RESOURCE_MISC_TEXTURELWBE;

    if (FAILED(g_pd3dDevice->CreateTexture2D(&desc, NULL,
                                             &g_texture_lwbe.pTexture)))
      return E_FAIL;

    D3D10_SHADER_RESOURCE_VIEW_DESC SRVDesc;
    ZeroMemory(&SRVDesc, sizeof(SRVDesc));
    SRVDesc.Format = desc.Format;
    SRVDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURELWBE;
    SRVDesc.TextureLwbe.MipLevels = desc.MipLevels;
    SRVDesc.TextureLwbe.MostDetailedMip = 0;

    if (FAILED(g_pd3dDevice->CreateShaderResourceView(
            g_texture_lwbe.pTexture, &SRVDesc, &g_texture_lwbe.pSRView)))
      return E_FAIL;

    g_pTextureLwbe->SetResource(g_texture_lwbe.pSRView);
  }

  return S_OK;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Lwca part of the computation
////////////////////////////////////////////////////////////////////////////////
void RunKernels() {
  static float t = 0.0f;

  // populate the 2d texture
  {
    lwdaArray *lwArray;
    lwdaGraphicsSubResourceGetMappedArray(&lwArray, g_texture_2d.lwdaResource,
                                          0, 0);
    getLastLwdaError(
        "lwdaGraphicsSubResourceGetMappedArray (lwda_texture_2d) failed");

    // kick off the kernel and send the staging buffer lwdaLinearMemory as an
    // argument to allow the kernel to write to it
    lwda_texture_2d(g_texture_2d.lwdaLinearMemory, g_texture_2d.width,
                    g_texture_2d.height, g_texture_2d.pitch, t);
    getLastLwdaError("lwda_texture_2d failed");

    // then we want to copy lwdaLinearMemory to the D3D texture, via its mapped
    // form : lwdaArray
    lwdaMemcpy2DToArray(
        lwArray,                                            // dst array
        0, 0,                                               // offset
        g_texture_2d.lwdaLinearMemory, g_texture_2d.pitch,  // src
        g_texture_2d.width * 4 * sizeof(float), g_texture_2d.height,  // extent
        lwdaMemcpyDeviceToDevice);                                    // kind
    getLastLwdaError("lwdaMemcpy2DToArray failed");
  }
  // populate the volume texture
  {
    size_t pitchSlice = g_texture_3d.pitch * g_texture_3d.height;
    lwdaArray *lwArray;
    lwdaGraphicsSubResourceGetMappedArray(&lwArray, g_texture_3d.lwdaResource,
                                          0, 0);
    getLastLwdaError(
        "lwdaGraphicsSubResourceGetMappedArray (lwda_texture_3d) failed");

    // kick off the kernel and send the staging buffer lwdaLinearMemory as an
    // argument to allow the kernel to write to it
    lwda_texture_3d(g_texture_3d.lwdaLinearMemory, g_texture_3d.width,
                    g_texture_3d.height, g_texture_3d.depth, g_texture_3d.pitch,
                    pitchSlice, t);
    getLastLwdaError("lwda_texture_3d failed");

    // then we want to copy lwdaLinearMemory to the D3D texture, via its mapped
    // form : lwdaArray
    struct lwdaMemcpy3DParms memcpyParams = {0};
    memcpyParams.dstArray = lwArray;
    memcpyParams.srcPtr.ptr = g_texture_3d.lwdaLinearMemory;
    memcpyParams.srcPtr.pitch = g_texture_3d.pitch;
    memcpyParams.srcPtr.xsize = g_texture_3d.width;
    memcpyParams.srcPtr.ysize = g_texture_3d.height;
    memcpyParams.extent.width = g_texture_3d.width;
    memcpyParams.extent.height = g_texture_3d.height;
    memcpyParams.extent.depth = g_texture_3d.depth;
    memcpyParams.kind = lwdaMemcpyDeviceToDevice;
    lwdaMemcpy3D(&memcpyParams);
    getLastLwdaError("lwdaMemcpy3D failed");
  }

  // populate the faces of the lwbe map
  for (int face = 0; face < 6; ++face) {
    lwdaArray *lwArray;
    lwdaGraphicsSubResourceGetMappedArray(&lwArray, g_texture_lwbe.lwdaResource,
                                          face, 0);
    getLastLwdaError(
        "lwdaGraphicsSubResourceGetMappedArray (lwda_texture_lwbe) failed");

    // kick off the kernel and send the staging buffer lwdaLinearMemory as an
    // argument to allow the kernel to write to it
    lwda_texture_lwbe(g_texture_lwbe.lwdaLinearMemory, g_texture_lwbe.size,
                      g_texture_lwbe.size, g_texture_lwbe.pitch, face, t);
    getLastLwdaError("lwda_texture_lwbe failed");

    // then we want to copy lwdaLinearMemory to the D3D texture, via its mapped
    // form : lwdaArray
    lwdaMemcpy2DToArray(lwArray,  // dst array
                        0, 0,     // offset
                        g_texture_lwbe.lwdaLinearMemory,
                        g_texture_lwbe.pitch,                          // src
                        g_texture_lwbe.size * 4, g_texture_lwbe.size,  // extent
                        lwdaMemcpyDeviceToDevice);                     // kind
    getLastLwdaError("lwdaMemcpy2DToArray failed");
  }

  t += 0.01f;
}

////////////////////////////////////////////////////////////////////////////////
//! Draw the final result on the screen
////////////////////////////////////////////////////////////////////////////////
void DrawScene() {
  // Clear the backbuffer to a black color
  float ClearColor[4] = {0.5f, 0.5f, 0.6f, 1.0f};
  g_pd3dDevice->ClearRenderTargetView(g_pSwapChainRTV, ClearColor);

  //
  // draw the 2d texture
  //
  g_pUseCase->SetInt(0);
  float quadRect[4] = {-0.9f, -0.9f, 0.7f, 0.7f};
  g_pvQuadRect->SetFloatVector((float *)&quadRect);
  g_pSimpleTechnique->GetPassByIndex(0)->Apply(0);
  g_pd3dDevice->Draw(4, 0);

  //
  // draw a slice the 3d texture
  //
  g_pUseCase->SetInt(1);
  quadRect[1] = 0.1f;
  g_pvQuadRect->SetFloatVector((float *)&quadRect);
  g_pSimpleTechnique->GetPassByIndex(0)->Apply(0);
  g_pd3dDevice->Draw(4, 0);

  //
  // draw the 6 faces of the lwbe texture
  //
  float faceRect[4] = {-0.1f, -0.9f, 0.5f, 0.5f};

  for (int f = 0; f < 6; f++) {
    if (f == 3) {
      faceRect[0] += 0.55f;
      faceRect[1] = -0.9f;
    }

    g_pUseCase->SetInt(2 + f);
    g_pvQuadRect->SetFloatVector((float *)&faceRect);
    g_pSimpleTechnique->GetPassByIndex(0)->Apply(0);
    g_pd3dDevice->Draw(4, 0);
    faceRect[1] += 0.6f;
  }

  // Present the backbuffer contents to the display
  g_pSwapChain->Present(0, 0);
}

//-----------------------------------------------------------------------------
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
void Cleanup() {
  // unregister the Lwca resources
  lwdaGraphicsUnregisterResource(g_texture_2d.lwdaResource);
  getLastLwdaError("lwdaGraphicsUnregisterResource (g_texture_2d) failed");
  lwdaFree(g_texture_2d.lwdaLinearMemory);
  getLastLwdaError("lwdaFree (g_texture_2d) failed");

  lwdaGraphicsUnregisterResource(g_texture_lwbe.lwdaResource);
  getLastLwdaError("lwdaGraphicsUnregisterResource (g_texture_lwbe) failed");
  lwdaFree(g_texture_lwbe.lwdaLinearMemory);
  getLastLwdaError("lwdaFree (g_texture_2d) failed");

  lwdaGraphicsUnregisterResource(g_texture_3d.lwdaResource);
  getLastLwdaError("lwdaGraphicsUnregisterResource (g_texture_3d) failed");
  lwdaFree(g_texture_3d.lwdaLinearMemory);
  getLastLwdaError("lwdaFree (g_texture_2d) failed");

  //
  // clean up Direct3D
  //
  {
    // release the resources we created
    g_texture_2d.pSRView->Release();
    g_texture_2d.pTexture->Release();
    g_texture_lwbe.pSRView->Release();
    g_texture_lwbe.pTexture->Release();
    g_texture_3d.pSRView->Release();
    g_texture_3d.pTexture->Release();

    if (g_pInputLayout != NULL) g_pInputLayout->Release();

    if (g_pSimpleEffect != NULL) g_pSimpleEffect->Release();

    if (g_pSwapChainRTV != NULL) g_pSwapChainRTV->Release();

    if (g_pSwapChain != NULL) g_pSwapChain->Release();

    if (g_pd3dDevice != NULL) g_pd3dDevice->Release();
  }
}

//-----------------------------------------------------------------------------
// Name: Render()
// Desc: Launches the LWCA kernels to fill in the texture data
//-----------------------------------------------------------------------------
void Render() {
  //
  // map the resources we've registered so we can access them in Lwca
  // - it is most efficient to map and unmap all resources in a single call,
  //   and to have the map/unmap calls be the boundary between using the GPU
  //   for Direct3D and Lwca
  //
  static bool doit = true;

  if (doit) {
    doit = true;
    lwdaStream_t stream = 0;
    const int nbResources = 3;
    lwdaGraphicsResource *ppResources[nbResources] = {
        g_texture_2d.lwdaResource, g_texture_3d.lwdaResource,
        g_texture_lwbe.lwdaResource,
    };
    lwdaGraphicsMapResources(nbResources, ppResources, stream);
    getLastLwdaError("lwdaGraphicsMapResources(3) failed");

    //
    // run kernels which will populate the contents of those textures
    //
    RunKernels();

    //
    // unmap the resources
    //
    lwdaGraphicsUnmapResources(nbResources, ppResources, stream);
    getLastLwdaError("lwdaGraphicsUnmapResources(3) failed");
  }

  //
  // draw the scene using them
  //
  DrawScene();
}

//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: The window's message handler
//-----------------------------------------------------------------------------
static LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam,
                              LPARAM lParam) {
  switch (msg) {
    case WM_KEYDOWN:
      if (wParam == VK_ESCAPE) {
        g_bDone = true;
        Cleanup();
        PostQuitMessage(0);
        return 0;
      }

      break;

    case WM_DESTROY:
      g_bDone = true;
      Cleanup();
      PostQuitMessage(0);
      return 0;

    case WM_PAINT:
      ValidateRect(hWnd, NULL);
      return 0;
  }

  return DefWindowProc(hWnd, msg, wParam, lParam);
}
