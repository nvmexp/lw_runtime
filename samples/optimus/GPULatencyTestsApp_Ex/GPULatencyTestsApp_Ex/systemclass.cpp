////////////////////////////////////////////////////////////////////////////////
// Filename: systemclass.cpp
////////////////////////////////////////////////////////////////////////////////
//#include "d3dclass.h"
#include "systemclass.h"
#include "stdafx.h"
#include "CoprocStatistics.h"
#include "lwapi.h"
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#define TARGET_RESOLUTION 1         // 1-millisecond target resolution

LwDRSSessionHandle hSession = 0;
LwPhysicalGpuHandle phys;    
LW_GPU_PERF_PSTATE_ID LwrrentPState = LWAPI_GPU_PERF_PSTATE_UNDEFINED;
LwAPI_Status lwapi_status = LWAPI_OK;
static const WORD MAX_CONSOLE_LINES = 500;
bool bCheckForGOLDTests = true;
bool G_FULL_SCREEN = false;
DWORD g_presentModel = 0;
int g_screenWidth = 0;
int g_screenHeight = 0;
double PCFreq = 0.0;
__int64 CounterStart = 0;
void StartCounter()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		return;

	PCFreq = double(li.QuadPart);

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}
double GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}

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

void SystemClass::errorParam(FILE *fp)
{
	fprintf(fp, "\n Incorrect input parameter format. Exiting... \n");
	fprintf(fp, "\n Please check wiki page 'GPULatencyTestsApp' for more details regarding correct usage \n");
	fclose(fp);
	return;
}
bool SystemClass::Initialize(DWORD initialTris, DWORD numCycles, DWORD delayInterval, bool FULL_SCREEN, bool bGOLDOptionCheck, DWORD numWasteMB )
{
    int screenWidth, screenHeight;
    bool result;
    DWORD presentModel = 0;
    bool gc6SupportStatus = false;
    LwAPI_Status coprocInfoStatus = LWAPI_ERROR;
    FILE* fp;
    errno_t err;
	err = fopen_s(&fp,"test_log.txt","w");
	printf( "\n\n GPULatencyTestsApp_Ex Version 1.0\n");
	fprintf(fp, "GPULatencyTestsApp_Ex Version 1.0\n");
	printf("\n Initializing... \n");
    fprintf(fp, "\n Initializing... \n"); 

    // Clear the frame data
    m_frameCount = 0;
    m_timeStamp = 0;
    m_triCount = initialTris;
    m_numCycles = numCycles;
    m_delayInterval = delayInterval;
	m_numWasteMB = numWasteMB;
	
    // set option for GOLD / GC6
    bCheckForGOLDTests = bGOLDOptionCheck;

    // Initialize the width and height of the screen to zero before sending the variables into the function.
    screenWidth = 0;
    screenHeight = 0;

    // Initialize the windows api.
    InitializeWindows(screenWidth, screenHeight, FULL_SCREEN);

    // populate the required global vars with input values
    G_FULL_SCREEN = FULL_SCREEN;
    g_presentModel = presentModel;
    g_screenWidth = screenWidth;
    g_screenHeight = screenHeight;
    
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

    m_Graphics->WasteMemory( m_hwnd, numWasteMB );
    
    // Initialize the LWAPI.
    LwAPI_Status status = LwAPI_Initialize();
    if (status != LWAPI_OK)
    {
        fprintf(fp, "\n LWAPI Initialization failed! \n");
		printf("\n LWAPI Initialization failed! \n");
    }

    // create session handle to access driver settings
    status = LwAPI_DRS_CreateSession(&hSession);
    if (status != LWAPI_OK)
    {
        fprintf(fp, "\n LWAPI CreateSession failed! \n");
		printf("\n LWAPI CreateSession failed! \n");
    }
    
    // load all the system settings into the session
    status = LwAPI_DRS_LoadSettings(hSession);
    if (status != LWAPI_OK)
    {
        fprintf(fp, "\n LWAPI LoadSettings failed! \n");
		printf("\n LWAPI LoadSettings failed! \n");
    }

    LwU32 cnt;
    lwapi_status = LwAPI_EnumPhysicalGPUs(&phys, &cnt);
    if (lwapi_status != LWAPI_OK)
    {
        fprintf(fp, "\n Unable to get physical GPU handle! \n");
		printf("\n Unable to get physical GPU handle! \n");
    }
    else
    {
        LwAPI_ShortString name;
        lwapi_status = LwAPI_GPU_GetFullName(phys,name);
        fprintf(fp, "\n Got physical GPU handle for GPU %s \n",name); 
		printf("\n Got physical GPU handle for GPU %s \n",name);
    }

    fprintf(fp, "**********************************************\n"); 
    printf("**********************************************\n");
    getAndShowCoprocInfo(phys, gc6SupportStatus, coprocInfoStatus, fp);
    fprintf(fp, "**********************************************\n"); 
    printf("**********************************************\n");

    fclose(fp);

    return true;
}
void SystemClass::RedirectIOToConsole()
{
	int hConHandle;
	long lStdHandle;
	CONSOLE_SCREEN_BUFFER_INFO coninfo;
	FILE *fp;
	
	// attach parent console for this app
	AttachConsole(ATTACH_PARENT_PROCESS);

	// set the screen buffer to be big enough to let us scroll text
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),&coninfo);
	coninfo.dwSize.Y = MAX_CONSOLE_LINES;
	SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE),coninfo.dwSize);

	// redirect unbuffered STDOUT to the console
	lStdHandle = (long)GetStdHandle(STD_OUTPUT_HANDLE);
	hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
	fp = _fdopen( hConHandle, "w" );
	*stdout = *fp;

	setvbuf( stdout, NULL, _IONBF, 0 );
	ios::sync_with_stdio();

}
bool SystemClass::Calibrate(DWORD presentModel, bool FULL_SCREEN,float CalibrateTime)
{
	MSG msg;
	m_CalibrateCount = 100000;
	m_CalibrateTime = CalibrateTime;
	ZeroMemory(&msg, sizeof(MSG));
	if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	if (msg.message == WM_QUIT)
	{
		return 0;
	}
	errno_t err;
	FILE *fp1 = NULL;
	err = fopen_s(&fp1, "calibrate.txt", "w+");
	if (fp1 == NULL)
		return 1;
	double t;
	m_Graphics->Frame();
	StartCounter();
	while ((t = GetCounter() ) < 1.0f)
	{
		StartCounter();
		m_Graphics->Render(m_CalibrateCount);
		m_CalibrateCount += 10000;
		if (m_Input->IsKeyDown(VK_ESCAPE))
		{
			return 0;
		}
	}
	m_CalibrateCount -= 10000;
	m_CalibrateCount = static_cast<DWORD>(m_CalibrateCount*m_CalibrateTime / 1.0f);
	fprintf_s(fp1, "Calibration Time : %f\nNumber of Triangles rendered : %d\n", m_CalibrateTime,m_CalibrateCount);
	
	fclose(fp1);
	return true;
}


void SystemClass::Shutdown(bool FULL_SCREEN)
{
    LwAPI_DRS_DestroySession(hSession);
    hSession = 0;

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

void SystemClass::ClearCoprocCycles(bool bResetCycles, LwU64 cycleCount)
{
    showCoprocCycleInformation(phys, bResetCycles, cycleCount, false, NULL);
}

void SystemClass::Run(bool bIsPStateForced)
{
    MSG msg;
    bool done;
    m_Count = 0;
    
    errno_t err;
    FILE* fp;
    err = fopen_s(&fp,"test_log.txt","a");
    
    // print test parameters to test log
    fprintf(fp,"\n Test parameters \n");
    fprintf(fp,"\n Number of triangles rendered / cycle : %u \n", m_triCount);
    fprintf(fp,"\n Number of render cycles              : %u \n", m_numCycles);
    fprintf(fp,"\n Delay interval                       : %u \n", m_delayInterval);
	fprintf(fp,"\n Memory footprint                     : %u \n", m_numWasteMB);

	printf("\n Test parameters \n");
    printf("\n Number of triangles rendered / cycle : %u \n", m_triCount);
    printf("\n Number of render cycles              : %u \n", m_numCycles);
    printf("\n Delay interval                       : %u \n", m_delayInterval);
	printf("\n Memory footprint                    : %u \n", m_numWasteMB);

    if (bCheckForGOLDTests)
    {
        fprintf(fp,"\n Checking GOLD cycles \n");
		printf("\n Checking GOLD cycles \n");
        // destroy existing context before run starts
        DestroyForGOLDTests();
        // Force P0
        lwapi_status = LwAPI_GPU_SetForcePstateEx(phys,LWAPI_GPU_PERF_PSTATE_P0,LWAPI_GPU_PERF_PSTATE_FALLBACK_HIGHER_PERF,0);
        if ((lwapi_status == LWAPI_OK) && (LwAPI_GPU_GetLwrrentPstate(phys,&LwrrentPState) == LWAPI_OK))
        {
            if(!(LwrrentPState == LWAPI_GPU_PERF_PSTATE_P0))
            {
                fprintf(fp,"\n LWAPI LwAPI_GPU_SetForcePstateEx failed to force P0 !!\n");
                fprintf(fp,"\n Current PState is P%u \n",LwrrentPState);
				
				printf("\n LWAPI LwAPI_GPU_SetForcePstateEx failed to force P0 !!\n");
                printf("\n Current PState is P%u \n",LwrrentPState);
            }
        }
    }
    else
    {
        fprintf(fp,"\n Checking for GC6 tests \n");
		 printf("\n Checking for GC6 tests \n");
        if (bIsPStateForced)
        {
            fprintf(fp,"\n PState will be forced to P0 and P8 at the start and end of the run respectively. \n");
			printf("\n PState will be forced to P0 and P8 at the start and end of the run respectively. \n");
        }
    }

    // Set the timer resolution
    TIMECAPS tc;
    UINT     wTimerRes;

    if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) 
    {
        fprintf(fp,"\n Could not obtain timer resolution!!! \n");
		printf("\n Could not obtain timer resolution!!! \n");
    }

    wTimerRes = min(max(tc.wPeriodMin, TARGET_RESOLUTION), tc.wPeriodMax);
    timeBeginPeriod(wTimerRes);
    fprintf(fp, "\n Timer resolution has been set to %u ms \n",wTimerRes);
    fprintf(fp, "\n Minimum timer resolution supported %u ms \n",tc.wPeriodMin);
	
	printf("\n Timer resolution has been set to %u ms \n",wTimerRes);
    printf("\n Minimum timer resolution supported %u ms \n",tc.wPeriodMin);

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
        else if (m_Count >= m_numCycles)
        {
            done = true;
            break;
        }
		
        if (bCheckForGOLDTests)       // do we need GOLD cycles
        {
            if(!(InitializeForGOLDTests()))
            {
                fprintf(fp,"\n Initialization for GOLD tests failed !! \n");
				printf("\n Initialization for GOLD tests failed !! \n");
            }            
        }
        else if (bIsPStateForced)   // GC6 cycles - force PState change to P0 only if the '/f' option is specified by the user.
        {
            // Check PState before frame rendering starts. Force to P0.
            lwapi_status = LwAPI_GPU_SetForcePstateEx(phys,LWAPI_GPU_PERF_PSTATE_P0,LWAPI_GPU_PERF_PSTATE_FALLBACK_HIGHER_PERF,0);
            if ((lwapi_status == LWAPI_OK) && (LwAPI_GPU_GetLwrrentPstate(phys,&LwrrentPState) == LWAPI_OK))
            {
                if(!(LwrrentPState == LWAPI_GPU_PERF_PSTATE_P0))
                {
                        fprintf(fp,"\n LWAPI LwAPI_GPU_SetForcePstateEx failed to force P0 !!\n");
                        fprintf(fp,"\n Current PState is P%u \n",LwrrentPState);

						printf("\n LWAPI LwAPI_GPU_SetForcePstateEx failed to force P0 !!\n");
                        printf("\n Current PState is P%u \n",LwrrentPState);
                }
            }            
        }

        // do the frame processing.    
        if(Frame())
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
        
        if (bCheckForGOLDTests)   // do we have to count GOLD cycles
        {
            DestroyForGOLDTests();
        }
        else if (bIsPStateForced)    // GC6 cycles - force PState change to P8 only if the '/f' option is specified by the user.
        {
            // Force GPU state to P8. 
            lwapi_status = LwAPI_GPU_SetForcePstateEx(phys,LWAPI_GPU_PERF_PSTATE_P8,LWAPI_GPU_PERF_PSTATE_FALLBACK_LOWER_PERF,0);
            if ((lwapi_status == LWAPI_OK) && (LwAPI_GPU_GetLwrrentPstate(phys,&LwrrentPState) == LWAPI_OK))
            {
                if (!(LwrrentPState == LWAPI_GPU_PERF_PSTATE_P8))
                {
                    fprintf(fp,"\n LWAPI LwAPI_GPU_SetForcePstateEx failed to force P8!!\n");
                    fprintf(fp,"\n Current PState is P%u \n",LwrrentPState);

					printf("\n LWAPI LwAPI_GPU_SetForcePstateEx failed to force P8!!\n");
                    printf("\n Current PState is P%u \n",LwrrentPState);
                }
            }
        }

        Sleep((m_delayInterval));
        //fprintf(fp,"\n Cycles completed : %u / %u \n",m_Count,m_numCycles);
    }

    // Dummy init to allow transition from last cycle to complete
    if (bCheckForGOLDTests)       // do we need GOLD cycles
    {
        if(!InitializeForGOLDTests())
        {
            fprintf(fp,"\n Initialization for GOLD tests failed !! \n");
			 printf("\n Initialization for GOLD tests failed !! \n");
        }
        
        DestroyForGOLDTests();
    }
    else if (bIsPStateForced)   // GC6 cycles - force PState change to P0 only if the '/f' option is specified by the user.
    {
        // Check PState before frame rendering starts. Force to P0.
        lwapi_status = LwAPI_GPU_SetForcePstateEx(phys,LWAPI_GPU_PERF_PSTATE_P0,LWAPI_GPU_PERF_PSTATE_FALLBACK_HIGHER_PERF,0);
        if ((lwapi_status == LWAPI_OK) && (LwAPI_GPU_GetLwrrentPstate(phys,&LwrrentPState) == LWAPI_OK))
        {
            if(!(LwrrentPState == LWAPI_GPU_PERF_PSTATE_P0))
            {
                    fprintf(fp,"\n LWAPI LwAPI_GPU_SetForcePstateEx failed to force P0 !!\n");
                    fprintf(fp,"\n Current PState is P%u \n",LwrrentPState);
					printf("\n LWAPI LwAPI_GPU_SetForcePstateEx failed to force P0 !!\n");
                    printf("\n Current PState is P%u \n",LwrrentPState);
            }
        }
    }

    fprintf(fp," \n Cycles completed %u out of %u \n",m_Count, m_numCycles);
	printf(" \n Cycles completed %u out of %u \n",m_Count, m_numCycles);
   

    if(!(timeEndPeriod(wTimerRes) == TIMERR_NOERROR))    // Clear the minimum timer resolution
    {
        fprintf(fp, "\n Error in clearing minimum timer resolution \n");
		printf("\n Error in clearing minimum timer resolution \n");
    }
	fclose(fp);
    return;
}


unsigned int SystemClass::Frame()
{
    m_frameCount++;
    m_Count++;

    // Do the frame processing for the graphics object.
    m_Graphics->Frame();

    // Finally render the graphics to the screen.
    m_Graphics->Render( m_triCount );
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
    m_applicationName = L"GPU Latency Tests App";

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

bool SystemClass::InitializeForGOLDTests()
{
    int screenWidth = 0, screenHeight = 0;
    bool result;
    
    // Create the graphics object.  This object will handle rendering all the graphics for this application.
    m_Graphics = new GraphicsClass;
    if(!m_Graphics)
    {
        return false;
    }

    // Initialize the graphics object.
    result = m_Graphics->Initialize(g_screenWidth, g_screenHeight, m_hwnd, g_presentModel, G_FULL_SCREEN);
    if(!result)
    {
        return false;
    }

    return true;
}

void SystemClass::DestroyForGOLDTests()
{
    // Release the graphics object.
    if(m_Graphics)
    {
        m_Graphics->Shutdown();
        delete m_Graphics;
        m_Graphics = 0;
    }
}