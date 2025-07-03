////////////////////////////////////////////////////////////////////////////////
// Filename: systemclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "systemclass.h"

//////////////
// INCLUDES //
//////////////
#include <d3d10_1.h>
//#include <d3d10.h>
#include <d3dx10.h>

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


bool SystemClass::Initialize( DWORD initialTris, float incrementRatio, float fpsLwtoff, DWORD presentModel, bool FULL_SCREEN, DWORD dxgiFormat,  bool bWindowedFullScreenTransition, bool bDXTLAutomationTesting)
{
    int screenWidth, screenHeight;
    bool result;

    // Clear the frame data
    m_frameCount = 0;
    m_timeStamp = 0;
    m_triCount = initialTris;
    m_incrementRatio = incrementRatio;
    m_fpsLwtoff = fpsLwtoff;
	m_bWindowedFullScreenTransition = bWindowedFullScreenTransition;
	m_FullScreen  = FULL_SCREEN;
	m_bDXTLAutomationTesting = bDXTLAutomationTesting;
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
    result = m_Graphics->Initialize(screenWidth, screenHeight, m_hwnd, presentModel, FULL_SCREEN, dxgiFormat, bWindowedFullScreenTransition, bDXTLAutomationTesting);
    if(!result)
    {
        return false;
    }
    
    return true;
}


void SystemClass::Shutdown(bool FULL_SCREEN)
{
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


void SystemClass::Run()
{
    MSG msg;
    bool done;
    m_AverageScore = 0;
    m_TotalScore = 0;
    m_Count = 0;
    //memset (m_Score, 0, sizeof (m_Score));
    int count = 0;
    // Initialize the message structure.
    ZeroMemory(&msg, sizeof(MSG));
    int del= remove ("out.csv");
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
		}// Otherwise do the frame processing.
		else if(Frame() )
		{
			done = true;
		}
		if (m_Count == 100)
		{
			done = true;
		}
        // Check if the user pressed escape and wants to quit.
        if(m_Input->IsKeyDown(VK_ESCAPE))
        {
            done = true;
        }
    }
    if (m_Count != 0)
    {
        m_AverageScore = m_TotalScore/m_Count;
        RECT desktop;
        int horizontal = 0, vertical = 0;
        // Get a handle to the desktop window
        //const HWND hDesktop = GetDesktopWindow();
        // Get the size of screen to the variable desktop
        GetWindowRect(m_hwnd, &desktop);
        // The top left corner will have coordinates (0,0)
        // and the bottom right corner will have coordinates
        // (horizontal, vertical)
        horizontal = desktop.right - desktop.left;
        vertical = desktop.bottom - desktop.top;
        double score = m_AverageScore*horizontal*vertical*4/(1073741824);
        //FILE* fp1 =  NULL;
        errno_t err;

        FILE* fp =  NULL;
        err = fopen_s (&fp, "Score.txt","w+");
        if (fp != NULL)
        {
            fprintf (fp, "Resolution  = %dX%d Average FPS  = %f Final Score  = %f GBps\n", horizontal, vertical, m_AverageScore, score);
        }
    }
    return;
}


unsigned int SystemClass::Frame()
{
    // handle fps stats
    DWORD lwrrentTime = timeGetTime();
    DWORD elapsedTime = lwrrentTime - m_timeStamp;
    static int i = 0;
    m_frameCount++;
    errno_t err;
    FILE *fp = NULL;
    err = fopen_s(&fp, "out.csv", "a+");
    if (fp == NULL)
        return 1;

	//For DXTL automation, lwrrently render single frame only and see if passed for different test cases (combinations of different presentation models and DXGI formats)
	if (m_bDXTLAutomationTesting && m_frameCount > 1)
	{
		Sleep (2000);
		return 1;
	}

    // Go at least a second, and at least 100 frames
    if( (elapsedTime >= 1000 ) && (m_frameCount >= 101) )
    {
        if( m_timeStamp == 0 )
        { // first time through for some given number of triangles
            m_timeStamp = lwrrentTime;
            m_frameCount = 0;
        }
        else
        { // additional passes
            //char outputString[256];
            float fps = 1000.f * (float)(m_frameCount-1) / (float)elapsedTime;

			fprintf_s(fp, "%f,%d,%d,%d\n", fps, m_triCount, m_frameCount, elapsedTime);
			//OutputDebugStringA( outputString );
			//printf( outputString );
            m_Count++;
            m_TotalScore = m_TotalScore + fps;
            // If we are below fps target, bail
            if( fps < m_fpsLwtoff )
            {
                fclose (fp);
                return 1;
            }
           m_frameCount = 0;
            m_timeStamp = lwrrentTime;
            if(m_incrementRatio > 1.0f)
            {
                unsigned int newCount = (unsigned int)((float)m_triCount * m_incrementRatio);

                if( newCount == m_triCount )
                { m_triCount++; }
                else
                { m_triCount = newCount; }
            }
            else if( (m_incrementRatio < 1.0f) && (m_triCount != 0) )
            {
                unsigned int newCount = (unsigned int)((float)m_triCount * m_incrementRatio);

                if(newCount == m_triCount) 
                { m_triCount--; }
                else
                { m_triCount = newCount; }
            }
        }
    }
    
    // Do the frame processing for the graphics object.
    m_Graphics->Frame();

    // Finally render the graphics to the screen.
    m_Graphics->Render( m_triCount );
    fclose (fp);
    return 0;
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
    m_applicationName = L"Engine";

    // Setup the windows class with default settings.
    wc.style         = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wc.lpfnWndProc   = WndProc;
    wc.cbClsExtra    = 0;
    wc.cbWndExtra    = 0;
    wc.hInstance     = m_hinstance;
    wc.hIcon		 = LoadIcon(NULL, IDI_WINLOGO);
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