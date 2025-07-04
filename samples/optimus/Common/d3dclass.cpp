////////////////////////////////////////////////////////////////////////////////
// Filename: d3dclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "d3dclass.h"
//#include "ImageDumperDx11.h"
#include <windows.h>

D3DClass::D3DClass()
{
    m_device = 0;
    m_swapChain = 0;
    m_renderTargetView = 0;
    m_depthStencilBuffer = 0;
    m_depthStencilState = 0;
    m_depthStencilView = 0;
    m_rasterState = 0;
}


D3DClass::D3DClass(const D3DClass& other)
{
}


D3DClass::~D3DClass()
{
}

bool D3DClass::CreateSwapChain(int screenWidth, int screenHeight, bool vsync, HWND hwnd, bool fullscreen, 
                          float screenDepth, float screenNear, DWORD presentModel, DWORD dxgiFormat)
{
	HRESULT result;
	IDXGIFactory* factory;
	IDXGIAdapter* adapter;
	IDXGIOutput* adapterOutput;
	unsigned int numModes, i, numerator, denominator;
	DXGI_MODE_DESC* displayModeList;
	DXGI_SWAP_CHAIN_DESC swapChainDesc;
	m_StartTime = timeGetTime();
	m_FullScreen = fullscreen;
  // Create a DirectX graphics interface factory.
    result = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
    if(FAILED(result))
    {
        return false;
    }

    // Use the factory to create an adapter for the primary graphics interface (video card).
    result = factory->EnumAdapters(0, &adapter);
    if(FAILED(result))
    {
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: Failed to Enumerate Adapters");
		}
        return false;
    }

    // Enumerate the primary adapter output (monitor).
    result = adapter->EnumOutputs(0, &adapterOutput);
    if(FAILED(result))
    {
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: Failed to Enumerate Outputs");
		}
        return false;
    }

    // Get the number of modes that fit the DXGI_FORMAT_R8G8B8A8_UNORM display format for the adapter output (monitor).
    result = adapterOutput->GetDisplayModeList((DXGI_FORMAT)dxgiFormat, DXGI_ENUM_MODES_INTERLACED, &numModes, NULL);
    if(FAILED(result))
    {
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: Failed to GetDisplayModeList");
		}
        return false;
    }

    // Create a list to hold all the possible display modes for this monitor/video card combination.
    displayModeList = new DXGI_MODE_DESC[numModes];
    if(!displayModeList)
    {
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: OOM failure");
		}
        return false;
    }

    // Now fill the display mode list structures.
    result = adapterOutput->GetDisplayModeList((DXGI_FORMAT)dxgiFormat, DXGI_ENUM_MODES_INTERLACED, &numModes, displayModeList);
    if(FAILED(result))
    {
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: Failed to GetDisplayModeList");
		}
        return false;
    }

    // Now go through all the display modes and find the one that matches the screen width and height.
    // When a match is found store the numerator and denominator of the refresh rate for that monitor.
    for(i=0; i<numModes; i++)
    {
        if(displayModeList[i].Width == (unsigned int)screenWidth)
        {
            if(displayModeList[i].Height == (unsigned int)screenHeight)
            {
                numerator = displayModeList[i].RefreshRate.Numerator;
                denominator = displayModeList[i].RefreshRate.Denominator;
            }
        }
    }

    // Release the display mode list.
    delete [] displayModeList;
    displayModeList = 0;

    // Release the adapter output.
    adapterOutput->Release();
    adapterOutput = 0;

    // Release the adapter.
    adapter->Release();
    adapter = 0;

    // Release the factory.
    factory->Release();
    factory = 0;

    // Initialize the swap chain description.
    ZeroMemory(&swapChainDesc, sizeof(swapChainDesc));

    // Set to a single back buffer.
    swapChainDesc.BufferCount = 2;

    // Set the width and height of the back buffer.
    swapChainDesc.BufferDesc.Width = screenWidth;
    swapChainDesc.BufferDesc.Height = screenHeight;

    // Set regular 32-bit surface for the back buffer.
    swapChainDesc.BufferDesc.Format = (DXGI_FORMAT)dxgiFormat;

    // Set the refresh rate of the back buffer.
    if(m_vsync_enabled)
    {
        swapChainDesc.BufferDesc.RefreshRate.Numerator = numerator;
        swapChainDesc.BufferDesc.RefreshRate.Denominator = denominator;
    }
    else
    {
        swapChainDesc.BufferDesc.RefreshRate.Numerator = 0;
        swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
    }

    // Set the usage of the back buffer.
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;

    // Set the handle for the window to render to.
    swapChainDesc.OutputWindow = hwnd;

    // Turn multisampling off.
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;

    // Set to full screen or windowed mode.
    if(fullscreen)
    {
        swapChainDesc.Windowed = false;
    }
    else
    {
        swapChainDesc.Windowed = true;
    }

    // Set the scan line ordering and scaling to unspecified.
    swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;

    swapChainDesc.SwapEffect = (DXGI_SWAP_EFFECT) presentModel;
    // Don't set the advanced flags.
    swapChainDesc.Flags = 0;

    // Create the swap chain and the Direct3D device.
    result = D3D10CreateDeviceAndSwapChain1(NULL, D3D10_DRIVER_TYPE_HARDWARE, NULL, 0, D3D10_FEATURE_LEVEL_10_1, D3D10_1_SDK_VERSION, 
                                           &swapChainDesc, &m_swapChain, &m_device);
    if(FAILED(result))
    {
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: D3D10CreateDeviceAndSwapChain1 failed");
		}
        return false;
    }
	return true;
}

bool D3DClass::InitializeD3D(int screenWidth, int screenHeight, float screenDepth, float screenNear)
{
	HRESULT result;
    ID3D10Texture2D* backBufferPtr;
    D3D10_TEXTURE2D_DESC depthBufferDesc;
    D3D10_DEPTH_STENCIL_DESC depthStencilDesc;
    D3D10_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
    D3D10_VIEWPORT viewport;
    float fieldOfView, screenAspect;
    D3D10_RASTERIZER_DESC rasterDesc;

	// Get the pointer to the back buffer.
	result = m_swapChain->GetBuffer(0, __uuidof(ID3D10Texture2D), (LPVOID*)&backBufferPtr);
	if(FAILED(result))
	{
		return false;
	}

	// Create the render target view with the back buffer pointer.
	result = m_device->CreateRenderTargetView(backBufferPtr, NULL, &m_renderTargetView);
	if(FAILED(result))
	{
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: CreateRenderTargetView failed");
		}
		return false;
	}

	// Release pointer to the back buffer as we no longer need it.
	backBufferPtr->Release();
	backBufferPtr = 0;

	// Initialize the description of the depth buffer.
	ZeroMemory(&depthBufferDesc, sizeof(depthBufferDesc));

	// Set up the description of the depth buffer.
	depthBufferDesc.Width = screenWidth;
	depthBufferDesc.Height = screenHeight;
	depthBufferDesc.MipLevels = 1;
	depthBufferDesc.ArraySize = 1;
	depthBufferDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthBufferDesc.SampleDesc.Count = 1;
	depthBufferDesc.SampleDesc.Quality = 0;
	depthBufferDesc.Usage = D3D10_USAGE_DEFAULT;
	depthBufferDesc.BindFlags = D3D10_BIND_DEPTH_STENCIL;
	depthBufferDesc.CPUAccessFlags = 0;
	depthBufferDesc.MiscFlags = 0;

	// Create the texture for the depth buffer using the filled out description.
	result = m_device->CreateTexture2D(&depthBufferDesc, NULL, &m_depthStencilBuffer);
	if(FAILED(result))
	{
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: CreateTexture2D failed");
		}
		return false;
	}

	// Initialize the description of the stencil state.
	ZeroMemory(&depthStencilDesc, sizeof(depthStencilDesc));

	// Set up the description of the stencil state.
	depthStencilDesc.DepthEnable = true;
	depthStencilDesc.DepthWriteMask = D3D10_DEPTH_WRITE_MASK_ALL;
	depthStencilDesc.DepthFunc = D3D10_COMPARISON_LESS;

	depthStencilDesc.StencilEnable = true;
	depthStencilDesc.StencilReadMask = 0xFF;
	depthStencilDesc.StencilWriteMask = 0xFF;

	// Stencil operations if pixel is front-facing.
	depthStencilDesc.FrontFace.StencilFailOp = D3D10_STENCIL_OP_KEEP;
	depthStencilDesc.FrontFace.StencilDepthFailOp = D3D10_STENCIL_OP_INCR;
	depthStencilDesc.FrontFace.StencilPassOp = D3D10_STENCIL_OP_KEEP;
	depthStencilDesc.FrontFace.StencilFunc = D3D10_COMPARISON_ALWAYS;

	// Stencil operations if pixel is back-facing.
	depthStencilDesc.BackFace.StencilFailOp = D3D10_STENCIL_OP_KEEP;
	depthStencilDesc.BackFace.StencilDepthFailOp = D3D10_STENCIL_OP_DECR;
	depthStencilDesc.BackFace.StencilPassOp = D3D10_STENCIL_OP_KEEP;
	depthStencilDesc.BackFace.StencilFunc = D3D10_COMPARISON_ALWAYS;

	// Create the depth stencil state.
	result = m_device->CreateDepthStencilState(&depthStencilDesc, &m_depthStencilState);
	if(FAILED(result))
	{
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: CreateDepthStencilState failed");
		}
		return false;
	}

	// Set the depth stencil state on the D3D device.
	m_device->OMSetDepthStencilState(m_depthStencilState, 1);

	// Initailze the depth stencil view.
	ZeroMemory(&depthStencilViewDesc, sizeof(depthStencilViewDesc));

	// Set up the depth stencil view description.
	depthStencilViewDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthStencilViewDesc.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2D;
	depthStencilViewDesc.Texture2D.MipSlice = 0;

	// Create the depth stencil view.
	result = m_device->CreateDepthStencilView(m_depthStencilBuffer, &depthStencilViewDesc, &m_depthStencilView);
	if(FAILED(result))
	{
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: CreateDepthStencilView failed");
		}
		return false;
	}

	// Bind the render target view and depth stencil buffer to the output render pipeline.
	m_device->OMSetRenderTargets(1, &m_renderTargetView, m_depthStencilView);

	// Setup the raster description which will determine how and what polygons will be drawn.
	rasterDesc.AntialiasedLineEnable = false;
	rasterDesc.LwllMode = D3D10_LWLL_BACK;
	rasterDesc.DepthBias = 0;
	rasterDesc.DepthBiasClamp = 0.0f;
	rasterDesc.DepthClipEnable = true;
	rasterDesc.FillMode = D3D10_FILL_SOLID;
	rasterDesc.FrontCounterClockwise = false;
	rasterDesc.MultisampleEnable = false;
	rasterDesc.ScissorEnable = false;
	rasterDesc.SlopeScaledDepthBias = 0.0f;

	// Create the rasterizer state from the description we just filled out.
	result = m_device->CreateRasterizerState(&rasterDesc, &m_rasterState);
	if(FAILED(result))
	{
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: CreateRasterizerState failed");
		}
		return false;
	}

	// Now set the rasterizer state.
	m_device->RSSetState(m_rasterState);

	// Setup the viewport for rendering.
	viewport.Width = screenWidth;
	viewport.Height = screenHeight;
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;

	// Create the viewport.
	m_device->RSSetViewports(1, &viewport);

	// Setup the projection matrix.
	fieldOfView = (float)D3DX_PI / 4.0f;
	screenAspect = (float)screenWidth / (float)screenHeight;

	// Create the projection matrix for 3D rendering.
	D3DXMatrixPerspectiveFovLH(&m_projectionMatrix, fieldOfView, screenAspect, screenNear, screenDepth);

	// Initialize the world matrix to the identity matrix.
	D3DXMatrixIdentity(&m_worldMatrix);

	// Create an orthographic projection matrix for 2D rendering.
	D3DXMatrixOrthoLH(&m_orthoMatrix, (float)screenWidth, (float)screenHeight, screenNear, screenDepth);
	return true;
}

bool D3DClass::Initialize(int screenWidth, int screenHeight, bool vsync, HWND hwnd, bool fullscreen, 
                          float screenDepth, float screenNear, DWORD presentModel, DWORD dxgiFormat, bool bWindowedFullScreenTransition, bool bDXTLAutomationTesting)
{
    // Store the vsync setting.
    m_vsync_enabled = vsync;
	m_bWindowedFullScreenTransition = bWindowedFullScreenTransition;
	m_bDXTLAutomationTesting = bDXTLAutomationTesting;
	if (m_bDXTLAutomationTesting)
	{	
		errno_t err;
		m_fpError = NULL;
		err = fopen_s (&m_fpError, "Result.txt","a+");
	}

	if (m_fpError != NULL && bDXTLAutomationTesting)
	{
		fprintf_s(m_fpError, "Test Case: fullscreen = %d presentModel = %d dxgiFormat = %d bWindowedFullScreenTransition = %d \n", fullscreen, presentModel, dxgiFormat, bWindowedFullScreenTransition);
	}

	if(!(CreateSwapChain(screenWidth, screenHeight, vsync, hwnd, fullscreen, 
                          screenDepth, screenNear, presentModel, dxgiFormat)))
    {
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: Failed to create Swap Chain");
		}
        return false;
    }

	if(!(InitializeD3D(screenWidth, screenHeight, screenDepth, screenNear)))
    {
		if (m_fpError != NULL)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: Failed to Initialize D3D");
		}
        return false;
    }

    return true;
}
void D3DClass::ShutdownD3DDeviceResources()
{
    // Before shutting down set to windowed mode or when you release the swap chain it will throw an exception.
    if(m_swapChain)
    {
        m_swapChain->SetFullscreenState(false, NULL);
    }

    if(m_rasterState)
    {
        m_rasterState->Release();
        m_rasterState = 0;
    }

    if(m_depthStencilView)
    {
        m_depthStencilView->Release();
        m_depthStencilView = 0;
    }

    if(m_depthStencilState)
    {
        m_depthStencilState->Release();
        m_depthStencilState = 0;
    }

    if(m_depthStencilBuffer)
    {
        m_depthStencilBuffer->Release();
        m_depthStencilBuffer = 0;
    }

    if(m_renderTargetView)
    {
        m_renderTargetView->Release();
        m_renderTargetView = 0;
    }
}
void D3DClass::ShutdownSwapChain()
{	
    if(m_swapChain)
    {
        m_swapChain->Release();
        m_swapChain = 0;
    }

    if(m_device)
    {
        m_device->Release();
        m_device = 0;
    }

    return;
}

void D3DClass::Shutdown()
{
	ShutdownD3DDeviceResources();
	ShutdownSwapChain();
    return;
}

void D3DClass::BeginScene()
{
    float color[4];


    color[0] = 0.0f;  // Red
    color[1] = 0.0f;  // Green
    color[2] = 0.0f;  // Blue
    color[3] = 1.0f;  // Alpha

    // Clear the back buffer.
    m_device->ClearRenderTargetView(m_renderTargetView, color);
    
    // Clear the depth buffer.
    m_device->ClearDepthStencilView(m_depthStencilView, D3D10_CLEAR_DEPTH, 1.0f, 0);
	
	// Bind the render target view and depth stencil buffer to the output render pipeline.
	m_device->OMSetRenderTargets(1, &m_renderTargetView, m_depthStencilView);
    return;
}


void D3DClass::EndScene()
{
	HRESULT result = S_OK;
    // Present the back buffer to the screen since rendering is complete.

    if(m_vsync_enabled)
    {
        // Lock to screen refresh rate.
        result = m_swapChain->Present(1, 0);
    }
    else
    {
        // Present as fast as possible.
        result = m_swapChain->Present(0, 0);
    }
	// Present can fail after fs<->windowed transition without a call to ResizeBuffers
	if (result != S_OK)
	{
		DXGI_SWAP_CHAIN_DESC newDesc;
		m_swapChain->GetDesc(&newDesc);
		m_swapChain->ResizeBuffers(2, 0, 0, DXGI_FORMAT_UNKNOWN, 0);
		if(!(InitializeD3D(newDesc.BufferDesc.Width, newDesc.BufferDesc.Height, 1000.0f, 0.1f)))
		{
			return;
		}
	}

	DWORD duration = timeGetTime() - m_StartTime;
	if (m_bWindowedFullScreenTransition)
	{
		// Hardcoded transition duration to 5 sec.
		if (m_FullScreen && duration > 5000)
		{
			m_swapChain->SetFullscreenState(FALSE, NULL);
			m_StartTime = timeGetTime();
			m_FullScreen = false;
		}
		// Hardcoded transition duration to 5 sec.
		else if (duration > 5000)
		{
			m_swapChain->SetFullscreenState(TRUE, NULL);
			m_StartTime = timeGetTime();
			m_FullScreen = true;
		}
	}
	
	if (m_fpError != NULL && m_bDXTLAutomationTesting)
	{
		if (result != S_OK)
		{
			fprintf_s(m_fpError, "%s\n", "Fail:: SWAPChain Present failed \n");
		}
		else
		{
			fprintf_s(m_fpError, "%s\n", "Pass:: Test Passed");
		}

	}
    return;
}


ID3D10Device* D3DClass::GetDevice()
{
  return m_device;
}


void D3DClass::GetProjectionMatrix(D3DXMATRIX& projectionMatrix)
{
    projectionMatrix = m_projectionMatrix;
    return;
}


void D3DClass::GetWorldMatrix(D3DXMATRIX& worldMatrix)
{
    worldMatrix = m_worldMatrix;
    return;
}


void D3DClass::GetOrthoMatrix(D3DXMATRIX& orthoMatrix)
{
    orthoMatrix = m_orthoMatrix;
    return;
}