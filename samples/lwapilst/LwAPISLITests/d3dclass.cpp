////////////////////////////////////////////////////////////////////////////////
// Filename: d3dclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "d3dclass.h"
#include "stdafx.h"
#include <D3DX10Tex.h>

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


bool D3DClass::Initialize(int screenWidth, int screenHeight, bool vsync, HWND hwnd, bool fullscreen, 
                          float screenDepth, float screenNear, DWORD presentModel)
{
    HRESULT result;
    IDXGIFactory* factory;
    IDXGIAdapter* adapter;
    IDXGIOutput* adapterOutput;
    unsigned int numModes, i, numerator, denominator;
    DXGI_MODE_DESC* displayModeList;
    DXGI_SWAP_CHAIN_DESC swapChainDesc;
    ID3D10Texture2D* backBufferPtr;
    D3D10_TEXTURE2D_DESC depthBufferDesc;
    D3D10_DEPTH_STENCIL_DESC depthStencilDesc;
    D3D10_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
    D3D10_VIEWPORT viewport;
    float fieldOfView, screenAspect;
    D3D10_RASTERIZER_DESC rasterDesc;


    // Store the vsync setting.
    m_vsync_enabled = vsync;

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
        return false;
    }

    // Enumerate the primary adapter output (monitor).
    result = adapter->EnumOutputs(0, &adapterOutput);
    if(FAILED(result))
    {
        return false;
    }

    // Get the number of modes that fit the DXGI_FORMAT_R8G8B8A8_UNORM display format for the adapter output (monitor).
    result = adapterOutput->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_ENUM_MODES_INTERLACED, &numModes, NULL);
    if(FAILED(result))
    {
        return false;
    }

    // Create a list to hold all the possible display modes for this monitor/video card combination.
    displayModeList = new DXGI_MODE_DESC[numModes];
    if(!displayModeList)
    {
        return false;
    }

    // Now fill the display mode list structures.
    result = adapterOutput->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_ENUM_MODES_INTERLACED, &numModes, displayModeList);
    if(FAILED(result))
    {
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
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

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

    // Discard the back buffer contents after presenting.
    //swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    swapChainDesc.SwapEffect = /*DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;*/(DXGI_SWAP_EFFECT) presentModel;
    // Don't set the advanced flags.
    swapChainDesc.Flags = 0;

    // Create the swap chain and the Direct3D device.
    result = D3D10CreateDeviceAndSwapChain1(NULL, D3D10_DRIVER_TYPE_HARDWARE, NULL, 0, D3D10_FEATURE_LEVEL_10_1, D3D10_1_SDK_VERSION, 
                                           &swapChainDesc, &m_swapChain, &m_device);
    if(FAILED(result))
    {
        MessageBox(hwnd, L"Could not create swap chain", L"Error", MB_OK);
        return false;
    }

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

    // load the textures
    ID3D10Resource* pResource = NULL;
    D3DX10_IMAGE_LOAD_INFO info;
    ZeroMemory( &info, sizeof(D3DX10_IMAGE_LOAD_INFO) );
    //info.Width          = D3DX10_DEFAULT;
    //info.Height         = D3DX10_DEFAULT;
    info.Width          = 256;
    info.Height         = 256;
    info.Depth          = D3DX10_DEFAULT;
    info.FirstMipLevel  = 0;
    info.MipLevels      = 1;
    info.Usage          = D3D10_USAGE_DYNAMIC;
    info.BindFlags      = D3D10_BIND_SHADER_RESOURCE;
    info.CpuAccessFlags = D3D10_CPU_ACCESS_WRITE; //| D3D10_CPU_ACCESS_READ;
    info.MiscFlags      = D3DX10_DEFAULT;
    info.Format         = DXGI_FORMAT_R8G8B8A8_UNORM;
    info.Filter         = D3DX10_DEFAULT;
    info.MipFilter      = D3DX10_DEFAULT;
    info.pSrcInfo       = NULL;
    
    //D3D10_TEXTURE2D_DESC desc;
    //ZeroMemory(&desc,sizeof(desc));
    /*desc.Width = 256;
    desc.Height = 256;
    desc.MipLevels = desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D10_USAGE_DYNAMIC;
    desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = D3D10_CPU_ACCESS_WRITE;
    */
    /*m_desc.Width = 256;
    m_desc.Height = 256;
    m_desc.MipLevels = m_desc.ArraySize = 1;
    m_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    m_desc.SampleDesc.Count = 1;
    m_desc.Usage = D3D10_USAGE_DYNAMIC;
    m_desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
    m_desc.CPUAccessFlags = D3D10_CPU_ACCESS_WRITE;
    ID3D10Texture2D *pTexture = NULL;
    m_device->CreateTexture2D( &m_desc, NULL, &pTexture );
    */

    m_textureSRV[0] = 0;
    result = D3DX10CreateShaderResourceViewFromFile(m_device, L"stars.dds", &info, NULL, &m_textureSRV[0], NULL);
    if(FAILED(result))
    {
        return false;
    }

    //ID3D10View* pView;
    //pView = (ID3D10View*) m_textureSRV[0];
    //pView->GetResource(&pResource);
    m_textureSRV[0]->GetResource(&pResource);
    m_textures[0] = (ID3D10Texture2D*) pResource;
    //m_textures[0] = 0;
    //m_device->CopyResource(pTexture,pResource);
    //m_textures[0] = pResource;
    
    m_textureSRV[1] = 0;
    result = D3DX10CreateShaderResourceViewFromFile(m_device, L"seafloor.dds" , &info, NULL, &m_textureSRV[1], NULL);
    if(FAILED(result))
    {
        return false;
    }

    m_textureSRV[1]->GetResource(&pResource);
    m_textures[1] = 0;
    m_textures[1] = (ID3D10Texture2D*)pResource;

    m_textureSRV[2] = 0;
    result = D3DX10CreateShaderResourceViewFromFile(m_device, L"bark_diff.dds", &info, NULL, &m_textureSRV[2], NULL);
    if(FAILED(result))
    {
        return false;
    }

    m_textureSRV[2]->GetResource(&pResource);
    m_textures[2] = 0;
    m_textures[2] = (ID3D10Texture2D*)pResource;

    return true;
}

ID3D10ShaderResourceView** D3DClass::getTextureSRVArray()
{
    return m_textureSRV;
}

ID3D10Texture2D** D3DClass::getTextureArray()
{
    return m_textures;
}
void D3DClass::Shutdown()
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
    
    for (int i=0;i<3;i++)
    {
        if (m_textureSRV[i])
        {
            m_textureSRV[i]->Release();
            m_textureSRV[i] = 0;
        }
    }

    return;
}


void D3DClass::BeginScene()
{
    float color[4];


    color[0] = 1.0f;  // Red
    color[1] = 1.0f;  // Green
    color[2] = 1.0f;  // Blue
    color[3] = 0.5f;  // Alpha

    // Clear the back buffer.
    m_device->ClearRenderTargetView(m_renderTargetView, color);
    
    // Clear the depth buffer.
    m_device->ClearDepthStencilView(m_depthStencilView, D3D10_CLEAR_DEPTH, 1.0f, 0);

    return;
}


void D3DClass::EndScene()
{
    // Present the back buffer to the screen since rendering is complete.
    if(m_vsync_enabled)
    {
        // Lock to screen refresh rate.
        m_swapChain->Present(1, 0);
    }
    else
    {
        // Present as fast as possible.
        m_swapChain->Present(0, 0);
    }

    return;
}

void D3DClass::TexelCopy(unsigned int source,unsigned int x, unsigned int y)
{
    const D3D10_BOX Box = {x,y,0,(x+1),(y+1),1};
    m_device->CopySubresourceRegion(m_textures[source+1],0,x,y,0,m_textures[source],0,&Box);

}

bool D3DClass::ColwertTexturesToFiles()
{
    HRESULT HR[3];
    
    HR[0] = D3DX10SaveTextureToFile(m_textures[0],D3DX10_IFF_BMP,L"Texture_1.bmp");
    HR[1] = D3DX10SaveTextureToFile(m_textures[1],D3DX10_IFF_BMP,L"Texture_2.bmp");
    HR[2] = D3DX10SaveTextureToFile(m_textures[2],D3DX10_IFF_BMP,L"Texture_3.bmp");

    return ((SUCCEEDED(HR[0]))&&(SUCCEEDED(HR[1]))&&(SUCCEEDED(HR[2])));
}

bool D3DClass::VerifyTextureCopy()
{
    FILE* fp;
    errno_t err;
    err = fopen_s(&fp,"verify_texture_copy.txt","w");
    LwU32 size, correctCopies = 0;

    bool bIsAnyTexelCopyIncorrect = false;
    
    D3D10_TEXTURE2D_DESC desc;
    ZeroMemory(&desc,sizeof(desc));
    desc.Width = 256;
    desc.Height = 256;
    desc.MipLevels = desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D10_USAGE_STAGING;
    desc.BindFlags = NULL;
    desc.CPUAccessFlags = D3D10_CPU_ACCESS_READ | D3D10_CPU_ACCESS_WRITE;

    ID3D10Texture2D *pSource = NULL;
    m_device->CreateTexture2D( &desc, NULL, &pSource );
    m_device->CopyResource(pSource,m_textures[0]);

    ID3D10Texture2D *pTarget = NULL;
    m_device->CreateTexture2D( &desc, NULL, &pTarget );
    m_device->CopyResource(pTarget,m_textures[2]);

    size = 256*256*4; // 256 X 256 texels, r8g8b8a8 format
    
    UCHAR *pBuffSource = (UCHAR*)pSource;
    UCHAR *pBuffTarget = (UCHAR*)pTarget;

    fprintf(fp, "\n Total texture size : %u bytes \n",size);

    for (unsigned int i=0; i<size; i++)
    {
       fprintf(fp,"\n Verified %u texels \n",i);
       //if (pBuffSource[i] == pBuffTarget[i])
       //{
       //    correctCopies += 1;
       //}
    }
    
    bIsAnyTexelCopyIncorrect = (correctCopies == size);
    fprintf(fp, "\n Number of bytes correctly copied : %u",correctCopies);
    fprintf(fp, "\n percent success : %2f \n", (float)((correctCopies*100)/size));
    fclose(fp);

    //pSource->Release();
    //pSource = 0;
    //pTarget->Release();
    //pTarget = 0; 
    return !(bIsAnyTexelCopyIncorrect);
}
//void D3DClass::TextureCopy(unsigned int source)
//{
//    UINT rowStart = 0, colStart = 0;
//    HRESULT hr = S_OK;
//    FILE* fp;
//    errno_t err;
//    err = fopen_s(&fp,"texture_copy_log.txt","w");
//
//    if (!((source == 0)||(source == 1)))
//    {
//        fprintf(fp," Invalid source value. Exiting !!");
//        fclose(fp);
//        return;
//    }
//    else
//    {
//        D3D10_MAPPED_TEXTURE2D mappedTexSource;
//        //D3D10_MAPPED_TEXTURE2D mappedTexDest;
//        //D3D10_TEXTURE2D_DESC descSource, descDest;
//        
//        //m_textures[source]->GetDesc(&descSource);
//        //m_textures[source+1]->GetDesc(&descDest);
//        
//        /*if ((descSource.Width != descDest.Width) || (descSource.Height != descDest.Height))
//        {
//            fprintf(fp,"Source and Target files do not have matching dimensions. Exiting... ");
//            fprintf(fp,"\n Source dimensions : %u Width %u Height \n", descSource.Width,descSource.Height);
//            fprintf(fp,"\n Dest dimensions : %u Width %u Height \n", descDest.Width,descDest.Height);
//            fclose(fp);
//            return;
//        }*/
//
//        
//        hr = m_textures[source]->Map(D3D10CalcSubresource(0, 0, 1), D3D10_MAP_WRITE, 0, &mappedTexSource );
//        //hr = m_textures[source+1]->Map(D3D10CalcSubresource(0, 0, 1), D3D10_MAP_WRITE, 0, &mappedTexDest );
//        
//        if (FAILED(hr))
//        {
//            fprintf(fp,"\n Texture mapping failed!. Exiting \n");
//            if (m_textures[source] == 0)
//            {
//                fprintf(fp,"ptr uninit!!");
//            }
//            fclose(fp);
//            return;
//        }
//        UCHAR* pSourceTexels = (UCHAR*)mappedTexSource.pData;
//        //UCHAR* pDestTexels = (UCHAR*)mappedTexDest.pData; 
//
//        for( UINT row = 0; row < m_desc.Height; row++ )
//        {
//            //rowStart = row * mappedTexDest.RowPitch;
//            rowStart = row * mappedTexSource.RowPitch;
//            for( UINT col = 0; col < m_desc.Width; col++ )
//            {
//                colStart = col*4;
//                //pDestTexels[rowStart + colStart + 0] = 0;
//                pSourceTexels[rowStart + colStart + 0] = 0;
//                //pDestTexels[rowStart + colStart + 1] = 0;//pSourceTexels[rowStart + colStart + 1];
//                //pDestTexels[rowStart + colStart + 2] = 0;//pSourceTexels[rowStart + colStart + 2];
//                //pDestTexels[rowStart + colStart + 3] = 0;//pSourceTexels[rowStart + colStart + 3];
//            }
//        }
//    }
//
//    m_textures[source]->Unmap(D3D10CalcSubresource(0, 0, 1));
//    m_textures[source+1]->Unmap(D3D10CalcSubresource(0, 0, 1));
//
//    fclose(fp);
//    return;
//}

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