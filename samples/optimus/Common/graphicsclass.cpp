////////////////////////////////////////////////////////////////////////////////
// Filename: graphicsclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "graphicsclass.h"

GraphicsClass::GraphicsClass()
{
    m_D3D = 0;
    m_Camera = 0;
    m_Model = 0;
    m_UnusedTexture = 0;
}


GraphicsClass::GraphicsClass(const GraphicsClass& other)
{
}


GraphicsClass::~GraphicsClass()
{
}


bool GraphicsClass::Initialize(int screenWidth, int screenHeight, HWND hwnd, DWORD presentModel, bool FULL_SCREEN, DWORD dxgiFormat, bool bWindowedFullScreenTransition, bool bDXTLAutomationTesting)
{
    bool result;

        
    // Create the Direct3D object.
    m_D3D = new D3DClass;
    if(!m_D3D)
    {
        return false;
    }

    // Initialize the Direct3D object.
    result = m_D3D->Initialize(screenWidth, screenHeight, VSYNC_ENABLED, hwnd, FULL_SCREEN, SCREEN_DEPTH, SCREEN_NEAR, presentModel, dxgiFormat, bWindowedFullScreenTransition, bDXTLAutomationTesting);
    if(!result)
    {
        return false;
    }

    // Create the camera object.
    m_Camera = new CameraClass;
    if(!m_Camera)
    {
        return false;
    }

    // Create the model object.
    m_Model = new ModelClass;
    if(!m_Model)
    {
        return false;
    }

    // Initialize the model object.
    result = m_Model->Initialize(m_D3D->GetDevice(), hwnd);
    if(!result)
    {
        return false;
    }

    return true;
}


void GraphicsClass::Shutdown()
{
    // Release the model object.
    if(m_Model)
    {
        m_Model->Shutdown();
        delete m_Model;
        m_Model = 0;
    }

    // Release the camera object.
    if(m_Camera)
    {
        delete m_Camera;
        m_Camera = 0;
    }

    // Release the Direct3D object.
    if(m_D3D)
    {
        m_D3D->Shutdown();
        delete m_D3D;
        m_D3D = 0;
    }

    return;
}


void GraphicsClass::Frame()
{
    // Set the position of the camera.
    m_Camera->SetPosition(0.0f, 0.0f, -10.0f);

    return;
}

void GraphicsClass::WasteMemory(HWND hwnd, unsigned int numMB)
{
    ID3D10Device* device = m_D3D->GetDevice();
    HRESULT result;
    D3D10_TEXTURE3D_DESC desc;

    if( numMB > 512 ) numMB = 512;

    if( numMB == 0 )
    { // not zero, but close enough
        desc.Width = 1;
        desc.Height = 1;
        desc.Depth = 1;
    }
    else if( numMB > 512 ) // cap at 512MB
    {   // 512 * 512 * 512 * 4 Bpp = 512MB
        desc.Width = 512;
        desc.Height = 512;
        desc.Depth = 512;
    }
    else
    {   // 512 * 512 * 4 Bpp = 1MB, so we can just set the depth level to the number of MB we want to waste
        desc.Width = 512;
        desc.Height = 512;
        desc.Depth = numMB;
    }
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Usage = D3D10_USAGE_DEFAULT;
    desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    result = device->CreateTexture3D( &desc, NULL, &m_UnusedTexture );
    if(FAILED(result))
    {
        MessageBox(hwnd, L"Failed To Create Waste Texture", L"GraphicsClass Error", MB_OK);
    }

    result = m_Model->SetTexture(m_D3D->GetDevice(), m_UnusedTexture);
    if(FAILED(result))
    {
        MessageBox(hwnd, L"Failed To Bind Waste Texture", L"GraphicsClass Error", MB_OK);
    }
}


void GraphicsClass::Render(unsigned int numTris)
{
    D3DXMATRIX viewMatrix, projectionMatrix, worldMatrix;

    // Clear the buffers to begin the scene.
    m_D3D->BeginScene();

    // Generate the view matrix based on the camera's position.
    m_Camera->Render();

    // Get the view, projection, and world matrices from the camera and d3d objects.
    m_Camera->GetViewMatrix(viewMatrix);
    m_D3D->GetProjectionMatrix(projectionMatrix);
    m_D3D->GetWorldMatrix(worldMatrix);

    // Send the three matricies to the model so the shader can use them to render with.
    m_Model->SetMatrices(worldMatrix, viewMatrix, projectionMatrix);

    // Render the model.
    m_Model->Render(m_D3D->GetDevice(), numTris);

    // Present the rendered scene to the screen.
    m_D3D->EndScene();

    return;
}