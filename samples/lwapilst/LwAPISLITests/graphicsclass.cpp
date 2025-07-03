////////////////////////////////////////////////////////////////////////////////
// Filename: graphicsclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "graphicsclass.h"
//#include "lwapi.h"

GraphicsClass::GraphicsClass()
{
    m_D3D = 0;
    m_Camera = 0;
    m_Model = 0;
}


GraphicsClass::GraphicsClass(const GraphicsClass& other)
{
}


GraphicsClass::~GraphicsClass()
{
}


bool GraphicsClass::Initialize(int screenWidth, int screenHeight, HWND hwnd, DWORD presentModel, bool FULL_SCREEN)
{
    bool result;
    ID3D10ShaderResourceView** ppTextureSRV;
        
    // Create the Direct3D object.
    m_D3D = new D3DClass;
    if(!m_D3D)
    {
        return false;
    }

    // Initialize the Direct3D object.
    result = m_D3D->Initialize(screenWidth, screenHeight, VSYNC_ENABLED, hwnd, FULL_SCREEN, SCREEN_DEPTH, SCREEN_NEAR, presentModel);
    if(!result)
    {
        MessageBox(hwnd, L"Could not initialize Direct3D", L"Error", MB_OK);
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
    ppTextureSRV = m_D3D->getTextureSRVArray();
    result = m_Model->Initialize(m_D3D->GetDevice(), hwnd, ppTextureSRV);
    if(!result)
    {
        MessageBox(hwnd, L"Could not initialize the model.", L"Error", MB_OK);
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

void GraphicsClass::Render()
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
    m_Model->Render(m_D3D->GetDevice());

    // Present the rendered scene to the screen.
    m_D3D->EndScene();

    return;
}

void GraphicsClass::Render_TextureCopy(bool bIsOptionSetResourceHint,bool bIsOptionTrackResource, LwU32 dwFlags)
{
    LwAPI_Status LwAPIStatus = LWAPI_OK;
    LWDX_ObjectHandle Handle;
    LW_GET_LWRRENT_SLI_STATE *pSliState = NULL;
    
    //LW_GET_LWRRENT_SLI_STATE pSliState;

    FILE* fp;
    errno_t err;
    err = fopen_s(&fp,"texture_copy_log.txt","w");

    fprintf(fp,"\n LWAPI Usage \n");
    fprintf(fp,"\n Use LwAPI_SetResourceHint ?                                             : %s \n",(bIsOptionSetResourceHint)? ("Yes") : ("No"));
    fprintf(fp,"\n Use LwAPI_D3D_BeginResourceRendering & LwAPI_D3D_EndResourceRendering ? : %s \n",(bIsOptionTrackResource)? ("Yes") : ("No"));
    if (bIsOptionTrackResource)
    {
        fprintf(fp,"\n Flags for LwAPI_D3D_BeginResourceRendering : %u \n",dwFlags);
    }

    // query SLI State information
    LwAPIStatus = LwAPI_D3D_GetLwrrentSLIState(m_D3D->GetDevice(), pSliState);
    if (LwAPIStatus == LWAPI_OK)
    {
        fprintf(fp,"\n Current SLI state details \n");
        fprintf(fp,"\n Version              : %u \n",pSliState->version);
        fprintf(fp,"\n Maximum AFR Groups   : %u \n",pSliState->maxNumAFRGroups);
        fprintf(fp,"\n AFR Groups Enabled   : %u \n",pSliState->numAFRGroups);
        fprintf(fp,"\n Current AFR Index    : %u \n",pSliState->lwrrentAFRIndex);
        fprintf(fp,"\n Next Frame AFR Index : %u \n",pSliState->nextFrameAFRIndex);
        fprintf(fp,"\n Prev Frame AFR index : %u \n",pSliState->previousFrameAFRIndex);
        fprintf(fp,"\n LwrrentAFRGroupNew ? : %u \n",pSliState->bIsLwrAFRGroupNew);
    }
    else
    {
        fprintf(fp,"\n Call to LwAPI_D3D_GetLwrrentSLIState failed !!\n");
    }

    //bool bIsOptionTrackResource = false, bIsOptionSetResourceHint = false;
    D3DXMATRIX viewMatrix, projectionMatrix, worldMatrix;
    
    // Generate the view matrix based on the camera's position.
    m_Camera->Render();

    // Get the view, projection, and world matrices from the camera and d3d objects.
    m_Camera->GetViewMatrix(viewMatrix);
    m_D3D->GetProjectionMatrix(projectionMatrix);
    m_D3D->GetWorldMatrix(worldMatrix);

    // Send the three matricies to the model so the shader can use them to render with.
    m_Model->SetMatrices(worldMatrix, viewMatrix, projectionMatrix);

    // Get the object handle for the texture object 1
    ID3D10Texture2D** pTextures = m_D3D->getTextureArray();
    LwAPIStatus = LwAPI_D3D_GetObjectHandleForResource(m_D3D->GetDevice(),(ID3D10Resource*)pTextures[1], &Handle);
    if (LwAPIStatus != LWAPI_OK)
    {
        fprintf(fp,"\n Unable to object handle for the texture object!!\n");
        fclose(fp);
        return;
    }
    if (bIsOptionSetResourceHint)
    {
        LwU32 HintValue = 1;
        LwAPIStatus = LwAPI_D3D_SetResourceHint(m_D3D->GetDevice(),Handle,LWAPI_D3D_SRH_CATEGORY_SLI,LWAPI_D3D_SRH_SLI_APP_CONTROLLED_INTERFRAME_CONTENT_SYNC,&HintValue);
        if (LwAPIStatus != LWAPI_OK)
        {
            fprintf(fp,"\n Unable to set resource hint \n");
            fclose(fp);
            return;
        }
    }

    
    LARGE_INTEGER liFreq , liStart , liEnd ;
	long double Freq = 0.0, time = 0.0, time2 = 0.0;
	QueryPerformanceFrequency(&liFreq);
	Freq = long double(liFreq.QuadPart)/1000.0;

    // CopySubresourceRegion is an asynchronous call. Flush the command buufer before the copy starts, to ensure that the time measured for the copy is accurate as possible
    m_D3D->GetDevice()->Flush();
    Sleep(1000);

    QueryPerformanceCounter(&liStart);
            
	for (unsigned int x=0;x<256;x++)
    {
        for (unsigned int y=0;y<256;y++)
        {
            // Clear the buffers to begin the scene.
            m_D3D->BeginScene();

            if (bIsOptionTrackResource)
            {
                if (y % 2)
                {
                    LwAPIStatus = LwAPI_D3D_BeginResourceRendering(m_D3D->GetDevice(), Handle, dwFlags);
                    if (LwAPIStatus != LWAPI_OK)
                    {
                        fprintf(fp,"\n Error in LwAPI_D3D_BeginResourceRendering \n");
                        fclose(fp);
                    }
                }
            }
            
            // Make the texture copy from texture 0 to texture 1
            m_D3D->TexelCopy(0,x,y);
            
            if (bIsOptionTrackResource)
            {
                if (y % 2)
                {
                    // Call to LwAPI_D3D_EndResourceRendering only supports the LWAPI_D3D_RR_FLAG_DEFAULTS flag.
                    LwAPIStatus = LwAPI_D3D_EndResourceRendering(m_D3D->GetDevice(), Handle, LWAPI_D3D_RR_FLAG_DEFAULTS);
                    if (LwAPIStatus != LWAPI_OK)
                    {
                        fprintf(fp,"\n Error in LwAPI_D3D_EndResourceRendering \n");
                        fclose(fp);
                    }
                }
            }

            // Render the model.
            m_Model->Render(m_D3D->GetDevice());

            // Present the rendered scene to the screen.
            m_D3D->EndScene();

            // Repeat for immediate conselwtive frame
            m_D3D->BeginScene();

            // Make the texture copy from texture 1 to texture 2
            m_D3D->TexelCopy(1,x,y);

            // Render the model.
            m_Model->Render(m_D3D->GetDevice());

            // Present the rendered scene to the screen.
            m_D3D->EndScene();
        }
    }

    //m_D3D->GetDevice()->Flush();  // not needed as present submits current command buffer queue to runtime
    QueryPerformanceCounter(&liEnd);
            
    time += (long double(liEnd.QuadPart-liStart.QuadPart)/Freq);

    fprintf(fp, "\n Predicted time for texture copy in milliseconds : %lf \n", time);
    fclose(fp);
    
    
    bool result = m_D3D->ColwertTexturesToFiles();
    //bool result = m_D3D->VerifyTextureCopy();
    return;
}

bool GraphicsClass::Verify_Texture_Copy()
{
    return (m_D3D->VerifyTextureCopy());
}