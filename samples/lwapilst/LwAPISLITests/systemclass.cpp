////////////////////////////////////////////////////////////////////////////////
// Filename: systemclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "systemclass.h"
#include "stdafx.h"
#include "lwapi.h"

LwDRSSessionHandle hSession = 0;
    
SystemClass::SystemClass()
{
    m_Input = 0;
    m_Graphics = 0;
}


SystemClass::SystemClass(const SystemClass& other)
{
}


SystemClass::~SystemClass()
{
}

bool SystemClass::Initialize_LwAPIs()
{
    LwPhysicalGpuHandle phys;    
    LwAPI_Status lwapi_status = LWAPI_OK;

    FILE* fp;
    errno_t err;

    // Open log file
    err = fopen_s(&fp,"test_log.txt","w");
    fprintf(fp, "\n Initializing... \n"); 

    // Initialize LwAPI library
    LwAPI_Status status = LwAPI_Initialize();
    if (status != LWAPI_OK)
    {
        fprintf(fp, "\n LWAPI Initialization failed! \n");
        fclose(fp);
        return false;
    }

    // create session handle to access driver settings
    status = LwAPI_DRS_CreateSession(&hSession);
    if (status != LWAPI_OK)
    {
        fprintf(fp, "\n LWAPI CreateSession failed! \n");
        fclose(fp);
        return false;
    }
    
    // load all the system settings into the session
    status = LwAPI_DRS_LoadSettings(hSession);
    if (status != LWAPI_OK)
    {
        fprintf(fp, "\n LWAPI LoadSettings failed! \n");
        fclose(fp);
        return false;
    }

    LwU32 cnt;
    lwapi_status = LwAPI_EnumPhysicalGPUs(&phys, &cnt);
    if (lwapi_status != LWAPI_OK)
    {
        fprintf(fp, "\n Unable to get physical GPU handle! \n");
        fclose(fp);
        return false;
    }
    else
    {
        LwAPI_ShortString name;
        lwapi_status = LwAPI_GPU_GetFullName(phys,name);
        fprintf(fp, "\n Got physical GPU handle for GPU %s \n",name);        
    }

    // LwAPI Initialization completed.    
    fclose(fp);
    return true;
}

bool SystemClass::Initialize(bool FULL_SCREEN)
{
    int screenWidth, screenHeight;
    bool result;
    DWORD presentModel = 0;
    
    // Initialize the width and height of the screen to zero before sending the variables into the function.
    screenWidth = 0;
    screenHeight = 0;

    // Initialize the windows api.
    InitializeWindows(screenWidth, screenHeight, FULL_SCREEN);

    // Create the input object.  This object will be used to handle reading the keyboard input from the user.
    m_Input = new InputClass;
    if(!m_Input)
    {
        return false;
    }

    // Initialize the input object.
    m_Input->Initialize();

    // Create the graphics object.  This object will handle rendering all the graphics for this application.
    m_Graphics = new GraphicsClass;
    if(!m_Graphics)
    {
        return false;
    }

    // Initialize the graphics object.
    result = m_Graphics->Initialize(screenWidth, screenHeight, m_hwnd, presentModel, FULL_SCREEN);
    if(!result)
    {
        return false;
    }   
    
    // LwAPI Initialization
    result = Initialize_LwAPIs();
    if(!result)
    {
        return false;
    }
    
    return true;
}

void SystemClass::Shutdown(bool FULL_SCREEN)
{
    LwAPI_DRS_DestroySession(hSession);
    hSession = 0;
    LwAPI_Unload();                      // unload lwapi library

    // Release the graphics object.
    if(m_Graphics)
    {
        m_Graphics->Shutdown();
        delete m_Graphics;
        m_Graphics = 0;
    }

    // Release the input object.
    if(m_Input)
    {
        delete m_Input;
        m_Input = 0;
    }

    // Shutdown the window.
    ShutdownWindows(FULL_SCREEN);
    
    return;
}


void SystemClass::Run(bool bSetResHint,bool bBeginEndCalls, LwU32 dwBeginCallFlags)
{
    MSG msg;
    bool done;
    
    errno_t err;
    FILE* fp;
    err = fopen_s(&fp,"test_log.txt","a");
    
    // Loop until there is a quit message from the window or the user.
    done = false;
    
    while(!done)
    {
        // Handle the windows messages.
        if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        // If windows signals to end the application then exit out.
        if(msg.message == WM_QUIT)
        {
            done = true;
            break;
        }
        
        // do the frame processing.    
        if(Frame(bSetResHint, bBeginEndCalls, dwBeginCallFlags))
        {
            done = true;
            break;
        }
        
        // Check if the user pressed escape and wants to quit.
        if(m_Input->IsKeyDown(VK_ESCAPE))
        {
            done = true;
            break;
        }        
    }

    fclose(fp);    
    return;
}


unsigned int SystemClass::Frame(bool bSetResHint,bool bBeginEndCalls,LwU32 dwBeginCallFlags)
{
    unsigned int result = 1;
    // Do the frame processing for the graphics object.
    m_Graphics->Frame();

    // Finally render the graphics to the screen.
    //m_Graphics->Render();
    m_Graphics->Render_TextureCopy(bSetResHint,bBeginEndCalls,dwBeginCallFlags);
    //m_Graphics->Verify_Texture_Copy();
    
    return result;
}


LRESULT CALLBACK SystemClass::MessageHandler(HWND hwnd, UINT umsg, WPARAM wparam, LPARAM lparam)
{
    switch(umsg)
    {
        // Check if a key has been pressed on the keyboard.
        case WM_KEYDOWN:
        {
            // If a key is pressed send it to the input object so it can record that state.
            m_Input->KeyDown((unsigned int)wparam);
            return 0;
        }

        // Check if a key has been released on the keyboard.
        case WM_KEYUP:
        {
            // If a key is released then send it to the input object so it can unset the state for that key.
            m_Input->KeyUp((unsigned int)wparam);
            return 0;
        }

        // Any other messages send to the default message handler as our application won't make use of them.
        default:
        {
            return DefWindowProc(hwnd, umsg, wparam, lparam);
        }
    }
}


void SystemClass::InitializeWindows(int& screenWidth, int& screenHeight, bool FULL_SCREEN)
{
    WNDCLASSEX wc;
    DEVMODE dmScreenSettings;
    int posX, posY;

    // Get an external pointer to this object.
    ApplicationHandle = this;

    // Get the instance of this application.
    m_hinstance = GetModuleHandle(NULL);

    // Give the application a name.
    m_applicationName = L"LwAPI SLI Tests";

    // Setup the windows class with default settings.
    wc.style         = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wc.lpfnWndProc   = WndProc;
    wc.cbClsExtra    = 0;
    wc.cbWndExtra    = 0;
    wc.hInstance     = m_hinstance;
    wc.hIcon         = LoadIcon(NULL, IDI_WINLOGO);
    wc.hIconSm       = wc.hIcon;
    wc.hLwrsor       = LoadLwrsor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszMenuName  = NULL;
    wc.lpszClassName = m_applicationName;
    wc.cbSize        = sizeof(WNDCLASSEX);
    
    // Register the window class.
    RegisterClassEx(&wc);

    // Determine the resolution of the clients desktop screen.
    screenWidth  = GetSystemMetrics(SM_CXSCREEN);
    screenHeight = GetSystemMetrics(SM_CYSCREEN);

    // Setup the screen settings depending on whether it is running in full screen or in windowed mode.
    if(FULL_SCREEN)
    {
        // If full screen set the screen to maximum size of the users desktop and 32bit.
        memset(&dmScreenSettings, 0, sizeof(dmScreenSettings));
        dmScreenSettings.dmSize       = sizeof(dmScreenSettings);
        dmScreenSettings.dmPelsWidth  = (unsigned long)screenWidth;
        dmScreenSettings.dmPelsHeight = (unsigned long)screenHeight;
        dmScreenSettings.dmBitsPerPel = 32;
        dmScreenSettings.dmFields     = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;

        // Change the display settings to full screen.
        ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN);

        // Set the position of the window to the top left corner.
        posX = posY = 0;
    }
    else
    {
        // If windowed then set it to 800x600 resolution.
        screenWidth  = 800;
        screenHeight = 600;

        // Place the window in the middle of the screen.
        posX = (GetSystemMetrics(SM_CXSCREEN) - screenWidth)  / 2;
        posY = (GetSystemMetrics(SM_CYSCREEN) - screenHeight) / 2;
    }

    // Create the window with the screen settings and get the handle to it.
    m_hwnd = CreateWindowEx(WS_EX_APPWINDOW, m_applicationName, m_applicationName, 
                            WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_POPUP,
                            posX, posY, screenWidth, screenHeight, NULL, NULL, m_hinstance, NULL);

    // Bring the window up on the screen and set it as main focus.
    ShowWindow(m_hwnd, SW_SHOW);
    SetForegroundWindow(m_hwnd);
    SetFolws(m_hwnd);

    // Hide the mouse cursor.
    ShowLwrsor(false);

  //AllocConsole();
  //AttachConsole( GetLwrrentProcessId() );
    return;
}


void SystemClass::ShutdownWindows(bool FULL_SCREEN)
{
    // Show the mouse cursor.
    ShowLwrsor(true);

    // Fix the display settings if leaving full screen mode.
    if(FULL_SCREEN)
    {
        ChangeDisplaySettings(NULL, 0);
    }

    // Remove the window.
    DestroyWindow(m_hwnd);
    m_hwnd = NULL;

    // Remove the application instance.
    UnregisterClass(m_applicationName, m_hinstance);
    m_hinstance = NULL;

    // Release the pointer to this class.
    ApplicationHandle = NULL;

    return;
}


LRESULT CALLBACK WndProc(HWND hwnd, UINT umessage, WPARAM wparam, LPARAM lparam)
{
    switch(umessage)
    {
        // Check if the window is being destroyed.
        case WM_DESTROY:
        {
            PostQuitMessage(0);
            return 0;
        }

        // Check if the window is being closed.
        case WM_CLOSE:
        {
            PostQuitMessage(0);
            return 0;
        }

        // All other messages pass to the message handler in the system class.
        default:
        {
            return ApplicationHandle->MessageHandler(hwnd, umessage, wparam, lparam);
        }
    }
}



