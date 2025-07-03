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

// This example demonstrates how to use the LWCA Direct3D bindings to fill
// a vertex buffer with LWCA and use Direct3D to render the data.
// Host code.

#pragma warning(disable : 4312)

#include <Windows.h>
#include <mmsystem.h>
#pragma warning(disable : 4996)  // disable deprecated warning
#include <strsafe.h>
#pragma warning(default : 4996)
#include <cassert>

// includes, lwca
#include <lwda_runtime_api.h>
#include <lwda_d3d9_interop.h>

// includes, project
#include <rendercheck_d3d9.h>
#include <helper_functions.h>  // Helper functions for other non-lwca utilities
#include <helper_lwda.h>       // LWCA Helper Functions for initialization
#include <DirectXMath.h>
using namespace DirectX;

#define MAX_EPSILON 10

static char *sSDKsample = "simpleD3D9";

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
IDirect3D9Ex *g_pD3D = NULL;              // Used to create the D3DDevice
unsigned int g_iAdapter = NULL;           // Our adapter
IDirect3DDevice9Ex *g_pD3DDevice = NULL;  // Our rendering device
IDirect3DVertexBuffer9 *g_pVB = NULL;     // Buffer to hold vertices

struct lwdaGraphicsResource *lwda_VB_resource;  // handles D3D9-LWCA exchange

D3DDISPLAYMODEEX g_d3ddm;
D3DPRESENT_PARAMETERS g_d3dpp;

bool g_bWindowed = true;
bool g_bDeviceLost = false;
bool g_bPassed = true;

// A structure for our custom vertex type
struct LWSTOMVERTEX {
  FLOAT x, y, z;  // The untransformed, 3D position for the vertex
  DWORD color;    // The vertex color
};

// Our custom FVF, which describes our custom vertex structure
#define D3DFVF_LWSTOMVERTEX (D3DFVF_XYZ | D3DFVF_DIFFUSE)

const unsigned int g_WindowWidth = 512;
const unsigned int g_WindowHeight = 512;

const unsigned int g_MeshWidth = 256;
const unsigned int g_MeshHeight = 256;

const unsigned int g_NumVertices = g_MeshWidth * g_MeshHeight;

bool g_bQAReadback = false;
int g_iFrameToCompare = 10;

int *pArgc = NULL;
char **pArgv = NULL;

float anim;

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
void runTest(int argc, char **argv, char *ref_file);
void runLwda();
bool SaveVBResult(int argc, char **argv);
HRESULT InitD3D9(HWND hWnd);
HRESULT InitD3D9RenderState();
HRESULT InitLWDA();
HRESULT RestoreContextResources();
HRESULT InitVertexBuffer();
HRESULT FreeVertexBuffer();
VOID Cleanup();
VOID SetupMatrices();
HRESULT Render();
LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// LWCA D3D9 kernel
extern "C" void simpleD3DKernel(float4 *pos, unsigned int width,
                                unsigned int height, float time);

#define NAME_LEN 512

char device_name[NAME_LEN];

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  char *ref_file = NULL;

  pArgc = &argc;
  pArgv = argv;

  printf("> %s starting...\n", sSDKsample);

  // command line options
  if (argc > 1) {
    if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
      getCmdLineArgumentString(argc, (const char **)argv, "file",
                               (char **)&ref_file);
    }
  }

  runTest(argc, argv, ref_file);

  //
  // and exit
  //
  printf("%s running on %s exiting...\n", sSDKsample, device_name);
  printf("%s sample finished returned: %s\n", sSDKsample,
         (g_bPassed ? "OK" : "ERROR!"));
  exit(g_bPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for LWCA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv, char *ref_file) {
  // Register the window class
  WNDCLASSEX wc = {sizeof(WNDCLASSEX),     CS_CLASSDC, MsgProc, 0L,   0L,
                   GetModuleHandle(NULL),  NULL,       NULL,    NULL, NULL,
                   "LWCA/D3D9 simpleD3D9", NULL};
  RegisterClassEx(&wc);

  // Create the application's window
  int xBorder = ::GetSystemMetrics(SM_CXSIZEFRAME);
  int yBorder = ::GetSystemMetrics(SM_CYSIZEFRAME);
  int yMenu = ::GetSystemMetrics(SM_CYMENU);
  HWND hWnd = CreateWindow(
      wc.lpszClassName, "LWCA/D3D9 simpleD3D9", WS_OVERLAPPEDWINDOW, 0, 0,
      g_WindowWidth + 2 * xBorder, g_WindowHeight + 2 * yBorder + yMenu, NULL,
      NULL, wc.hInstance, NULL);

  // Initialize Direct3D9
  if (SUCCEEDED(InitD3D9(hWnd)) && SUCCEEDED(InitLWDA())) {
    // Create the scene geometry
    if (SUCCEEDED(InitVertexBuffer())) {
      // This is the normal case (D3D9 device is present)
      if (!g_bDeviceLost) {
        // Initialize D3D9 vertex buffer contents using LWCA kernel
        runLwda();

        // Save result
        SaveVBResult(argc, argv);

        // Show the window
        ShowWindow(hWnd, SW_SHOWDEFAULT);
        UpdateWindow(hWnd);
      }

      // Enter the message loop
      MSG msg;
      ZeroMemory(&msg, sizeof(msg));

      while (msg.message != WM_QUIT) {
        if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
          TranslateMessage(&msg);
          DispatchMessage(&msg);
        } else {
          Render();

          if (ref_file != NULL) {
            for (int count = 0; count < g_iFrameToCompare; count++) {
              Render();
            }

            const char *lwr_image_path = "simpleD3D9.ppm";

            // Save a reference of our current test run image
            CheckRenderD3D9::BackbufferToPPM(g_pD3DDevice, lwr_image_path);

            // compare to offical reference image, printing PASS or FAIL.
            g_bPassed = CheckRenderD3D9::PPMvsPPM(lwr_image_path, ref_file,
                                                  argv[0], MAX_EPSILON, 0.15f);

            Cleanup();

            PostQuitMessage(0);
          }
        }
      }
    }
  }

  UnregisterClass(wc.lpszClassName, wc.hInstance);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Lwca part of the computation
////////////////////////////////////////////////////////////////////////////////
void runLwda() {
  HRESULT hr = S_OK;

  // Map vertex buffer to Lwca
  float4 *d_ptr;

  // LWCA Map call to the Vertex Buffer and return a pointer
  checkLwdaErrors(lwdaGraphicsMapResources(1, &lwda_VB_resource, 0));
  getLastLwdaError("lwdaGraphicsMapResources failed");
  // This gets a pointer from the Vertex Buffer
  size_t num_bytes;
  checkLwdaErrors(lwdaGraphicsResourceGetMappedPointer(
      (void **)&d_ptr, &num_bytes, lwda_VB_resource));
  getLastLwdaError("lwdaGraphicsResourceGetMappedPointer failed");

  // Execute kernel
  simpleD3DKernel(d_ptr, g_MeshWidth, g_MeshHeight, anim);

  // LWCA Map Unmap vertex buffer
  checkLwdaErrors(lwdaGraphicsUnmapResources(1, &lwda_VB_resource, 0));
  getLastLwdaError("lwdaGraphicsUnmapResource failed");
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
bool SaveVBResult(int argc, char **argv) {
  // Lock vertex buffer
  float *data;

  if (FAILED(g_pVB->Lock(0, 0, (void **)&data, 0))) {
    return false;
  }

  // Save result
  if (checkCmdLineFlag(argc, (const char **)argv, "regression")) {
    // write file for regression test
    sdkWriteFile<float>("./data/regression.dat", data, sizeof(LWSTOMVERTEX),
                        0.0f, false);
  }

  // unlock
  if (FAILED(g_pVB->Unlock())) {
    return false;
  }

  return true;
}

//-----------------------------------------------------------------------------
// Name: InitD3D9()
// Desc: Initializes Direct3D9
//-----------------------------------------------------------------------------
HRESULT InitD3D9(HWND hWnd) {
  // Create the D3D object.
  if (S_OK != Direct3DCreate9Ex(D3D_SDK_VERSION, &g_pD3D)) {
    return E_FAIL;
  }

  D3DADAPTER_IDENTIFIER9 adapterId;
  int device;
  bool bDeviceFound = false;
  printf("\n");

  lwdaError lwStatus;

  for (g_iAdapter = 0; g_iAdapter < g_pD3D->GetAdapterCount(); g_iAdapter++) {
    HRESULT hr = g_pD3D->GetAdapterIdentifier(g_iAdapter, 0, &adapterId);

    if (FAILED(hr)) {
      continue;
    }

    lwStatus = lwdaD3D9GetDevice(&device, adapterId.DeviceName);
    // This prints and resets the lwdaError to lwdaSuccess
    printLastLwdaError("lwdaD3D9GetDevice failed");

    printf("> Display Device #%d: \"%s\" %s Direct3D9\n", g_iAdapter,
           adapterId.Description,
           (lwStatus == lwdaSuccess) ? "supports" : "does not support");

    if (lwdaSuccess == lwStatus) {
      bDeviceFound = true;
      STRCPY(device_name, NAME_LEN, adapterId.Description);
      break;
    }
  }

  // we check to make sure we have found a lwca-compatible D3D device to work on
  if (!bDeviceFound) {
    printf("\n");
    printf("  No LWCA-compatible Direct3D9 device available\n");
    printf("PASSED\n");
    // destroy the D3D device
    g_pD3D->Release();
    exit(EXIT_SUCCESS);
  }

  RECT rc;
  GetClientRect(hWnd, &rc);
  g_pD3D->GetAdapterDisplayModeEx(g_iAdapter, &g_d3ddm, NULL);

  // Set up the structure used to create the D3DDevice
  ZeroMemory(&g_d3dpp, sizeof(g_d3dpp));
  g_d3dpp.Windowed = g_bWindowed;
  g_d3dpp.BackBufferCount = 1;
  g_d3dpp.hDeviceWindow = hWnd;
  g_d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
  g_d3dpp.BackBufferFormat = g_d3ddm.Format;
  g_d3dpp.FullScreen_RefreshRateInHz = 0;  // set to 60 for fullscreen, and also
                                           // don't forget to set Windowed to
                                           // FALSE
  g_d3dpp.PresentationInterval =
      D3DPRESENT_INTERVAL_ONE;  // D3DPRESENT_DONOTWAIT;

  g_d3dpp.BackBufferWidth = g_WindowWidth;
  g_d3dpp.BackBufferHeight = g_WindowHeight;

  // Create the D3DDevice
  if (FAILED(g_pD3D->CreateDeviceEx(g_iAdapter, D3DDEVTYPE_HAL, hWnd,
                                    D3DCREATE_HARDWARE_VERTEXPROCESSING,
                                    &g_d3dpp, NULL, &g_pD3DDevice))) {
    return E_FAIL;
  }

  if (FAILED(InitD3D9RenderState())) {
    return E_FAIL;
  }

  return S_OK;
}

// Initialize the D3D Rendering State
HRESULT InitD3D9RenderState() {
  // Turn off lwlling, so we see the front and back of the triangle
  if (FAILED(g_pD3DDevice->SetRenderState(D3DRS_LWLLMODE, D3DLWLL_NONE))) {
    return E_FAIL;
  }

  // Turn off D3D lighting, since we are providing our own vertex colors
  if (FAILED(g_pD3DDevice->SetRenderState(D3DRS_LIGHTING, FALSE))) {
    return E_FAIL;
  }

  return S_OK;
}

HRESULT InitLWDA() {
  printf("InitLWDA() g_pD3DDevice = %p\n", g_pD3DDevice);

  // Now we need to bind a LWCA context to the DX9 device
  // This is the LWCA 2.0 DX9 interface (required for Windows XP and Vista)
  lwdaD3D9SetDirect3DDevice(g_pD3DDevice);
  getLastLwdaError("lwdaD3D9SetDirect3DDevice failed");

  return S_OK;
}

////////////////////////////////////////////////////////////////////////////////
//! RestoreContextResourcess
//    - this function restores all of the LWCA/D3D resources and contexts
////////////////////////////////////////////////////////////////////////////////
HRESULT RestoreContextResources() {
  // Reinitialize D3D9 resources, LWCA resources/contexts
  InitLWDA();
  InitVertexBuffer();
  InitD3D9RenderState();

  return S_OK;
}

//-----------------------------------------------------------------------------
// Name: InitVertexBuffer()
// Desc: Creates the scene geometry (Vertex Buffer)
//-----------------------------------------------------------------------------
HRESULT InitVertexBuffer() {
  // Create vertex buffer
  if (FAILED(g_pD3DDevice->CreateVertexBuffer(
          g_NumVertices * sizeof(LWSTOMVERTEX), 0, D3DFVF_LWSTOMVERTEX,
          D3DPOOL_DEFAULT, &g_pVB, NULL))) {
    return E_FAIL;
  }

  // Initialize interoperability between LWCA and Direct3D9
  // Register vertex buffer with LWCA
  lwdaGraphicsD3D9RegisterResource(&lwda_VB_resource, g_pVB,
                                   lwdaD3D9RegisterFlagsNone);
  getLastLwdaError("lwdaGraphicsD3D9RegisterResource failed");

  return S_OK;
}

//-----------------------------------------------------------------------------
// Name: FreeVertexBuffer()
// Desc: Free's the Vertex Buffer resource
//-----------------------------------------------------------------------------
HRESULT FreeVertexBuffer() {
  if (g_pVB != NULL) {
    // Unregister vertex buffer
    lwdaGraphicsUnregisterResource(lwda_VB_resource);
    getLastLwdaError("lwdaGraphicsUnregisterResource failed");

    g_pVB->Release();
  }

  return S_OK;
}

//-----------------------------------------------------------------------------
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
VOID Cleanup() {
  FreeVertexBuffer();

  if (g_pD3DDevice != NULL) {
    g_pD3DDevice->Release();
  }

  if (g_pD3D != NULL) {
    g_pD3D->Release();
  }
}

//-----------------------------------------------------------------------------
// Name: SetupMatrices()
// Desc: Sets up the world, view, and projection transform matrices.
//-----------------------------------------------------------------------------
VOID SetupMatrices() {
  // For our world matrix, we will just rotate the object about the y-axis.
  XMFLOAT4X4 matWorldFloat;
  XMMATRIX matWorld;
  matWorld = XMMatrixIdentity();
  XMStoreFloat4x4(&matWorldFloat, matWorld);
  g_pD3DDevice->SetTransform(D3DTS_WORLD, (D3DMATRIX *)&matWorldFloat);

  // Set up our view matrix. A view matrix can be defined given an eye point,
  // a point to lookat, and a direction for which way is up. Here, we set the
  // eye five units back along the z-axis and up three units, look at the
  // origin, and define "up" to be in the y-direction.
  XMVECTOR vEyePt = {0.0f, 3.0f, -2.0f};
  XMVECTOR vLookatPt = {0.0f, 0.0f, 0.0f};
  XMVECTOR vUpVec = {0.0f, 1.0f, 0.0f};
  XMMATRIX matView;
  XMFLOAT4X4 matViewFloat;
  matView = XMMatrixLookAtLH(vEyePt, vLookatPt, vUpVec);
  XMStoreFloat4x4(&matViewFloat, matView);
  g_pD3DDevice->SetTransform(D3DTS_VIEW, (D3DMATRIX *)&matViewFloat);

  // For the projection matrix, we set up a perspective transform (which
  // transforms geometry from 3D view space to 2D viewport space, with
  // a perspective divide making objects smaller in the distance). To build
  // a perpsective transform, we need the field of view (1/4 pi is common),
  // the aspect ratio, and the near and far clipping planes (which define at
  // what distances geometry should be no longer be rendered).
  XMMATRIX matProj;
  XMFLOAT4X4 matProjFloat;
  matProj = XMMatrixPerspectiveFovLH((float)XM_PI / 4, 1.0f, 1.0f, 100.0f);
  XMStoreFloat4x4(&matProjFloat, matProj);
  g_pD3DDevice->SetTransform(D3DTS_PROJECTION, (D3DMATRIX *)&matProjFloat);
}

////////////////////////////////////////////////////////////////////////////////
//! DeviceLostHandler
//    - this function handles reseting and initialization of the D3D device
//      in the event this Device gets Lost
////////////////////////////////////////////////////////////////////////////////
HRESULT DeviceLostHandler() {
  HRESULT hr = S_OK;

  // test the cooperative level to see if it's okay
  // to render
  if (FAILED(hr = g_pD3DDevice->TestCooperativeLevel())) {
    // if the device was truly lost, (i.e., a fullscreen device just lost
    // focus), wait
    // until we g_et it back
    if (hr == D3DERR_DEVICELOST) {
      return S_OK;
    }

    // eventually, we will g_et this return value,
    // indicating that we can now reset the device
    if (hr == D3DERR_DEVICENOTRESET) {
      // if we are windowed, read the desktop mode and use the same format for
      // the back buffer; this effectively turns off color colwersion

      if (g_bWindowed) {
        g_pD3D->GetAdapterDisplayModeEx(g_iAdapter, &g_d3ddm, NULL);
        g_d3dpp.BackBufferFormat = g_d3ddm.Format;
      }

      // now try to reset the device
      if (FAILED(hr = g_pD3DDevice->Reset(&g_d3dpp))) {
        return hr;
      } else {
        // This is a common function we use to restore all hardware
        // resources/state
        RestoreContextResources();

        // we have acquired the device
        g_bDeviceLost = false;
      }
    }
  }

  return hr;
}

//-----------------------------------------------------------------------------
// Name: Render()
// Desc: Draws the scene
//-----------------------------------------------------------------------------
HRESULT Render() {
  HRESULT hr = S_OK;

  // Begin code to handle case where the D3D gets lost
  if (g_bDeviceLost) {
    if (FAILED(hr = DeviceLostHandler())) {
      fprintf(stderr, "DeviceLostHandler FAILED returned %08x\n", hr);
      return hr;
    }

    fprintf(stderr, "Render DeviceLost handler\n");

    // test the cooperative level to see if it's okay
    // to render
    if (FAILED(hr = g_pD3DDevice->TestCooperativeLevel())) {
      fprintf(stderr,
              "TestCooperativeLevel = %08x failed, will attempt to reset\n",
              hr);

      // if the device was truly lost, (i.e., a fullscreen device just lost
      // focus), wait
      // until we g_et it back

      if (hr == D3DERR_DEVICELOST) {
        fprintf(
            stderr,
            "TestCooperativeLevel = %08x DeviceLost, will retry next call\n",
            hr);
        return S_OK;
      }

      // eventually, we will g_et this return value,
      // indicating that we can now reset the device
      if (hr == D3DERR_DEVICENOTRESET) {
        fprintf(stderr,
                "TestCooperativeLevel = %08x will try to RESET the device\n",
                hr);
        // if we are windowed, read the desktop mode and use the same format for
        // the back buffer; this effectively turns off color colwersion

        if (g_bWindowed) {
          g_pD3D->GetAdapterDisplayModeEx(g_iAdapter, &g_d3ddm, NULL);
          g_d3dpp.BackBufferFormat = g_d3ddm.Format;
        }

        // now try to reset the device
        if (FAILED(hr = g_pD3DDevice->Reset(&g_d3dpp))) {
          fprintf(stderr, "TestCooperativeLevel = %08x RESET device FAILED\n",
                  hr);
          return hr;
        } else {
          fprintf(stderr, "TestCooperativeLevel = %08x RESET device SUCCESS!\n",
                  hr);

          // Reinitialize D3D9 resources, LWCA resources/contexts
          InitLWDA();
          InitVertexBuffer();
          InitD3D9RenderState();

          fprintf(stderr, "TestCooperativeLevel = %08x INIT device SUCCESS!\n",
                  hr);

          // we have acquired the device
          g_bDeviceLost = false;
        }
      }

      return hr;
    }
  }

  if (!g_bDeviceLost) {
    // Clear the backbuffer to a black color
    g_pD3DDevice->Clear(0, NULL, D3DCLEAR_TARGET, D3DCOLOR_XRGB(0, 0, 0), 1.0f,
                        0);

    // Run LWCA to update vertex positions
    runLwda();

    // Begin the scene
    if (SUCCEEDED(g_pD3DDevice->BeginScene())) {
      // Setup the world, view, and projection matrices
      SetupMatrices();

      // Render the vertex buffer contents
      g_pD3DDevice->SetStreamSource(0, g_pVB, 0, sizeof(LWSTOMVERTEX));
      g_pD3DDevice->SetFVF(D3DFVF_LWSTOMVERTEX);
      g_pD3DDevice->DrawPrimitive(D3DPT_POINTLIST, 0, g_NumVertices);

      // End the scene
      g_pD3DDevice->EndScene();
    }

    // Present the backbuffer contents to the display
    hr = g_pD3DDevice->Present(NULL, NULL, NULL, NULL);

    if (hr == D3DERR_DEVICELOST) {
      fprintf(stderr, "drawScene Present = %08x detected D3D DeviceLost\n", hr);
      g_bDeviceLost = true;

      FreeVertexBuffer();
    }
  }

  anim += 0.1f;

  return hr;
}

//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: The window's message handler
//-----------------------------------------------------------------------------
LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  switch (msg) {
    case WM_DESTROY:
    case WM_KEYDOWN:
      if (msg != WM_KEYDOWN || wParam == 27) {
        Cleanup();

        PostQuitMessage(0);
        return 0;
      }
  }

  return DefWindowProc(hWnd, msg, wParam, lParam);
}
