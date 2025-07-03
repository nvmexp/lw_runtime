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
	printf( "\n\n GPULatencyTestsApp Version 2.0\n");
	fprintf(fp, "GPULatencyTestsApp Version 2.0\n");
	printf("\n Initializing... \n");
    fprintf(fp, "\n Initializing... \n"); 

    // Clear the frame data
    m_frameCount = 0;
    m_timeStamp = 0;
    m_triCount = initialTris;
    m_numCycles = numCycles;
    m_delayInterval = delayInterval;
	
	
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

void SystemClass::Run(bool bIsPStateForced, CLEAR_TOOL_STATS doNotclearStat)
{
    printf("\n run started \n");
    MSG msg;
    bool done;
    m_Count = 0;
    
    errno_t err;
    FILE* fp;
    err = fopen_s(&fp,"test_log.txt","w+");
    if(err != 0) printf("\n file not created err code: %d \n",err);
    // print test parameters to test log
    fprintf(fp,"\n Test parameters \n");
    fprintf(fp,"\n Number of triangles rendered / cycle : %u \n", m_triCount);
    fprintf(fp,"\n Number of render cycles              : %u \n", m_numCycles);
    fprintf(fp,"\n Delay interval                       : %u \n", m_delayInterval);
    fprintf(fp,"\n Fullscreen mode                      : %s \n",((G_FULL_SCREEN)? ("yes") : ("no")));
	
	printf("\n Test parameters \n");
    printf("\n Number of triangles rendered / cycle : %u \n", m_triCount);
    printf("\n Number of render cycles              : %u \n", m_numCycles);
    printf("\n Delay interval                       : %u \n", m_delayInterval);
    printf("\n Fullscreen mode                      : %s \n",((G_FULL_SCREEN)? ("yes") : ("no")));

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

    // Clear existing coproc stats before starting run
    if(doNotclearStat & CLEAR_START)
    {
        if(!clearCoprocStats(phys))
        {
            fprintf(fp,"\n Failed to clear stats before run start!! \n"); 
            printf("\n Failed to clear stats before run start!! \n"); 
        }
    }

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
        printf("\n sleep called :   ");
        Sleep((m_delayInterval));
        printf("Cycles completed : %u / %u",m_Count,m_numCycles);
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
    getAndShowGC6Statistics(phys, false, fp);
    getAndShowGOLDStatistics(phys, false, (doNotclearStat & CLEAR_END), fp);

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

void SystemClass::Run_TargetCycles(bool bIsPStateForced,DWORD TargetCycles, DWORD DelayIntervalStep, int StepDirection, CLEAR_TOOL_STATS doNotclearStat)
{
    MSG msg;
    bool bIsRunCompleted = false, bIsTestCompleted = false,bExitAfterCollectingStats = false, bHasFirstRunStarted = false, bIsDirectionAutoAssigned = false;

    int direction = StepDirection;       // direction of delayinterval increment performed after each run. 
    int requiredDirection = 0;
    DWORD CompletedRuns = 0, TransitionCount = 0;
    unsigned int const MaximumRunsAllowed = 100;

    errno_t err;
    FILE* fp;
    err = fopen_s(&fp,"test_log.txt","a");
    
    // print test parameters to test log
    fprintf(fp,"\n Test parameters \n");
    fprintf(fp,"\n Number of triangles rendered / cycle : %u \n", m_triCount);
    fprintf(fp,"\n Number of render cycles              : %u \n", m_numCycles);
    fprintf(fp,"\n Initial Delay interval               : %u ms \n", m_delayInterval);
    fprintf(fp,"\n Target cycles                        : %u \n",TargetCycles);
    fprintf(fp,"\n Delay Interval Step                  : %u ms \n", DelayIntervalStep);
    fprintf(fp,"\n Fullscreen mode                      : %s \n",((G_FULL_SCREEN)? ("yes") : ("no")));

	printf("\n Test parameters \n");
    printf("\n Number of triangles rendered / cycle : %u \n", m_triCount);
    printf("\n Number of render cycles              : %u \n", m_numCycles);
    printf("\n Initial Delay interval               : %u ms \n", m_delayInterval);
    printf("\n Target cycles                        : %u \n",TargetCycles);
    printf("\n Delay Interval Step                  : %u ms \n", DelayIntervalStep);
    printf("\n Fullscreen mode                      : %s \n",((G_FULL_SCREEN)? ("yes") : ("no")));
    // initialize for GC6 / GOLD tests
    if (bCheckForGOLDTests)
    {
        fprintf(fp,"\n Checking for GOLD cycles \n");
		printf("\n Checking for GOLD cycles \n");
        // destroy existing context before test starts
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
        fprintf(fp,"\n Checking for GC6 cycles \n");
		printf("\n Checking for GC6 cycles \n");
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
    fprintf(fp, "\n Timer resolution has been set to %u ms \n", wTimerRes);
    fprintf(fp, "\n Minimum timer resolution supported %u ms \n", tc.wPeriodMin);
	printf("\n Timer resolution has been set to %u ms \n", wTimerRes);
    printf("\n Minimum timer resolution supported %u ms \n", tc.wPeriodMin);

    // Clear existing coproc stats before starting run
    if(doNotclearStat & CLEAR_START)
    {
        if(!clearCoprocStats(phys))
        {
            fprintf(fp,"\n Failed to clear stats before run start!! \n"); 
            printf("\n Failed to clear stats before run start!! \n"); 
        }
    }

    DWORD dwStartTime, dwEndTime;

    while(!bIsTestCompleted)
    {
        // Dummy init to allow transition from previous run to complete
        if (bHasFirstRunStarted)
        {

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

            // collect statistics for previous run. change delay interval if needed.

            if (bCheckForGOLDTests)
            {
                LW_COPROC_GOLD_STATISTICS goldStats  = {0};
                goldStats.version                    = LW_COPROC_GOLD_STATISTICS_VER2;
                LwAPI_Status status = LwAPI_Coproc_GetGoldStatisticsEx(phys, false, &goldStats);
    
                if(status != LWAPI_OK)
    	        {
					fprintf(fp, "\n Getting goldStats failed! \n");
            	    printf("\n Getting goldStats failed! \n");
                	fclose(fp);
	                return;
    	        }
                fprintf(fp," \n Delay Interval for run %u is %u ms ",CompletedRuns,m_delayInterval);
                TransitionCount = goldStats.dwGoldTransitionCount;
                fprintf(fp," \n GOLD Transition count for this run is %u\n",TransitionCount);
                
                printf(" \n Delay Interval for run %u is %u ms ",CompletedRuns,m_delayInterval);
                
                printf(" \n GOLD Transition count for this run is %u\n",TransitionCount);
            }
            else
            { 
                LW_GPU_GC6_STATISTICS gc6Stats         = {0};
                gc6Stats.version                       = LW_GPU_GC6_STATISTICS_VER9;
                gc6Stats.bEnableLPWRInfo               = LW_GPU_FEATURE_LPWR_INFO_DISABLE;
                LwAPI_Status status = LwAPI_GPU_GetGC6Statistics(phys, &gc6Stats);
                if(status != LWAPI_OK)
                {
                    fprintf(fp, "\n Getting gc6Stats failed! \n");
                    printf("\n Getting gc6Stats failed! \n");
                    fclose(fp);
                    return;
                }
                fprintf(fp," \n Delay Interval for run %u is %u ms ",CompletedRuns,m_delayInterval);
                TransitionCount = gc6Stats.GC6TransitionCount;
                fprintf(fp," \n GC6 Transition count is %u\n",TransitionCount,CompletedRuns);     
                printf(" \n Delay Interval for run %u is %u ms ",CompletedRuns,m_delayInterval);
                   
                printf(" \n GC6 Transition count is %u\n",TransitionCount,CompletedRuns);    
            }
                
            float fElapsedTime = (float)dwEndTime - dwStartTime;
            float fAveFrameTime = fElapsedTime / (float)m_numCycles;
            float fTrashTime = fAveFrameTime - (float)m_delayInterval;
            float fractionInSleep = (float)m_delayInterval / fAveFrameTime;
            fprintf(fp," \n ElapsedTime %7.2f (s), TrashTime %7.2f (ms), fraction in sleep %f\n , AveFrameTime %7.2f (ms)\n",
                fElapsedTime/1000.f, fTrashTime, fractionInSleep, fAveFrameTime );
            printf(" \n ElapsedTime %7.2f (s), TrashTime %7.2f (ms), fraction in sleep %f\n , AveFrameTime %7.2f (ms)\n",
                fElapsedTime/1000.f, fTrashTime, fractionInSleep, fAveFrameTime );

            if (TransitionCount == TargetCycles)
            {
                if (CompletedRuns == 1)
                {
                    fprintf(fp," \n Error : targetCycles equal the transition count in first run itself \n Unable to get cross over point. Exiting... \n",m_delayInterval);
                    printf(" \n Error : targetCycles equal the transition count in first run itself \n Unable to get cross over point. Exiting... \n",m_delayInterval);
                }
                else
                {
                    fprintf(fp, "\n targetCycles achieved in this run (delayInterval %u ms). Exiting ... \n",m_delayInterval);
                    printf("\n targetCycles achieved in this run (delayInterval %u ms). Exiting ... \n",m_delayInterval);
                    getAndShowGC6Statistics(phys, false, fp);
                    getAndShowGOLDStatistics(phys, false, (doNotclearStat & CLEAR_END), fp);
                }
                fclose(fp);
                return;
            }

            // assign direction if not obtained from the command line
            if (CompletedRuns == 1)
            {
                if (direction == 0)
                {
                    direction = ((TransitionCount < TargetCycles)? 1 : (-1));
                    bIsDirectionAutoAssigned = true;
                }

                if (direction == 1)
                {
                    fprintf(fp," \n Delay interval will now be increased by %u ms in each run. \n", DelayIntervalStep); 
                    printf(" \n Delay interval will now be increased by %u ms in each run. \n", DelayIntervalStep); 
                }
                else if (direction == -1)
                {
                    fprintf(fp," \n Delay interval will now be decreased by %u ms in each run. \n", DelayIntervalStep);
                    printf(" \n Delay interval will now be decreased by %u ms in each run. \n", DelayIntervalStep);
                }                                    
            }

            // internally track the direction needed
            if (requiredDirection == 0)
            {
                requiredDirection = ((TransitionCount < TargetCycles) ? 1 : (-1));
            }
            
            // Ensure that the delayInterval does not become non-positive
            if ((direction == -1) && (m_delayInterval <= DelayIntervalStep))
            {
                if ((TransitionCount > TargetCycles) || (direction != requiredDirection))  
                {
                    // target cycles not achieved, and the delayInterval cannot be reduced further by DelayIntervalStep
                    fprintf(fp, " \n Target Cycles have not been achieved till run %u. \n ",CompletedRuns);
                    fprintf(fp, " \n As delay interval needs to be positive, and as current delay interval < delay interval step, no further runs will be held. Exiting ... \n ");
                    printf(" \n Target Cycles have not been achieved till run %u. \n ",CompletedRuns);
                    printf(" \n As delay interval needs to be positive, and as current delay interval < delay interval step, no further runs will be held. Exiting ... \n ");
                }
                else   // target cycles achieved in last possible run
                {
                    fprintf(fp," \n targetCycles achieved in this run (delayInterval %u ms). exiting... \n",m_delayInterval);
                    printf(" \n targetCycles achieved in this run (delayInterval %u ms). exiting... \n",m_delayInterval);
                 }
                getAndShowGC6Statistics(phys, false, fp);
                getAndShowGOLDStatistics(phys, false, (doNotclearStat & CLEAR_END), fp);
                fclose(fp);
                return;
            }

            if (((int)(direction*TransitionCount)) < ((int)(direction*TargetCycles)))
            {
                m_delayInterval += (direction*DelayIntervalStep);
            }
            else if (requiredDirection == direction)    // direction was found automatically, or the user predicted it correctly
            {
                fprintf(fp," \n targetCycles achieved in this run (delayInterval %u ms). exiting... \n",m_delayInterval);
                printf(" \n targetCycles achieved in this run (delayInterval %u ms). exiting... \n",m_delayInterval);
               
                
                getAndShowGC6Statistics(phys, false, fp);
                getAndShowGOLDStatistics(phys, false, (doNotclearStat & CLEAR_END), fp);

                fclose(fp);
                return;
            }
            else      // step direction input by user was incorreclty predicted, this will mostly lead to exit bealwse of run count reaching its maximum allowed value.
            {
                m_delayInterval += (direction*DelayIntervalStep);
            }
        }

        if (bExitAfterCollectingStats)
        {
            // stats have been collected. exit now.
            return;
        }
        
        if (CompletedRuns >= MaximumRunsAllowed)
        {
            fprintf(fp, "\n Maximum runs allowed have been completed. Exiting... \n");
			printf("\n Maximum runs allowed have been completed. Exiting... \n");
            
            getAndShowGC6Statistics(phys, false, fp);
            getAndShowGOLDStatistics(phys, false, (doNotclearStat & CLEAR_END), fp);
            fclose(fp);
            return;
        }

        // Loop until there is a quit message from the window / user, or when the numCycles get completed
        m_Count = 0;
        bIsRunCompleted = false;
        bHasFirstRunStarted = true;

        dwStartTime = timeGetTime();
	
        while(!bIsRunCompleted)
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
                bIsRunCompleted = true;
                bIsTestCompleted = true;
            }
            else if (m_Count >= m_numCycles)
            {
                bIsRunCompleted = true;
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
            else if (bIsPStateForced)  // GC6 cycles - force PState change to P0 only if the '/f' option is specified by the user.
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
                bIsRunCompleted = true;
                break;
            }
            
            // Check if the user pressed escape and wants to quit.
            if(m_Input->IsKeyDown(VK_ESCAPE))
            {
                bIsRunCompleted = true;
                fprintf(fp,"\n Exiting application due to Escape key press from user \n");
				printf("\n Exiting application due to Escape key press from user \n");
                bExitAfterCollectingStats = true;
            }
            
            if (bCheckForGOLDTests)   // do we have to count GOLD cycles
            {
                DestroyForGOLDTests();
            }
            else if (bIsPStateForced)   // GC6 cycles - force PState change to P8 only if the '/f' option is specified by the user.
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

        dwEndTime = timeGetTime();

        CompletedRuns += 1;        
    }

    if(!(timeEndPeriod(wTimerRes) == TIMERR_NOERROR))    // Clear the minimum timer resolution
    {
        fprintf(fp, "\n Error in clearing minimum timer resolution \n");
		printf("\n Error in clearing minimum timer resolution \n");
    }
	fclose(fp);
    return;
}

void SystemClass::Run_MinMaxHold(bool bIsPStateForced, DWORD MaxTriangles, DWORD MaxDelayInterval, DWORD HoldInterval, CLEAR_TOOL_STATS doNotclearStat)
{
    MSG msg;
    bool done;

    DWORD MinTriangles = m_triCount, MinDelayInterval = m_delayInterval,Counter = 0;
    DWORD triRange = ((MaxTriangles - MinTriangles) + 1);
    DWORD delayRange = ((MaxDelayInterval - MinDelayInterval) + 1);
    ULONG LastKnownTransitionCount = 0;
    
    m_Count = 0;
    errno_t err;
    FILE* fp;
    err = fopen_s(&fp,"test_log.txt","a+");
    
    // any of the input params cannot be zero, min param should be less than max param
    if ((MaxTriangles == 0)||(MaxDelayInterval == 0)||(HoldInterval == 0)||(MinTriangles > MaxTriangles)||(MinDelayInterval > MaxDelayInterval))
    {
        fprintf(fp, "\n Incorrect input format. MaxTriangles, MaxDelayInterval and HoldInterval need to be non-zero");
        fprintf(fp, "\n Also, MinTriangles < MaxTriangles and MinDelayInterval < MaxDelayInterval need to be true \n");
		printf("\n Incorrect input format. MaxTriangles, MaxDelayInterval and HoldInterval need to be non-zero");
        printf("\n Also, MinTriangles < MaxTriangles and MinDelayInterval < MaxDelayInterval need to be true \n");
        
        fclose(fp);
        return;
    }
    
    // print test parameters to test log
    fprintf(fp,"\n Test parameters \n");
    fprintf(fp,"\n Minimum triangle count               : %u \n", MinTriangles);
    fprintf(fp,"\n Maximum triangle count               : %u \n", MaxTriangles);
    fprintf(fp,"\n Minimum Delay interval               : %u \n", MinDelayInterval);
    fprintf(fp,"\n Maximum Delay interval               : %u \n", MaxDelayInterval);    
    fprintf(fp,"\n Hold Interval                        : %u \n", HoldInterval);
    fprintf(fp,"\n Number of render cycles              : %u \n", m_numCycles);
    fprintf(fp,"\n Fullscreen mode                      : %s \n",((G_FULL_SCREEN)? ("yes") : ("no")));

	printf("\n Test parameters \n");
    printf("\n Minimum triangle count               : %u \n", MinTriangles);
	printf("\n Maximum triangle count               : %u \n", MaxTriangles);
    printf("\n Minimum Delay interval               : %u \n", MinDelayInterval);
    printf("\n Maximum Delay interval               : %u \n", MaxDelayInterval);    
    printf("\n Hold Interval                        : %u \n", HoldInterval);
    printf("\n Number of render cycles              : %u \n", m_numCycles);
    printf("\n Fullscreen mode                      : %s \n",((G_FULL_SCREEN)? ("yes") : ("no")));
    //print Gold or GC6 test
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

    // Clear existing coproc stats before starting run
    if(doNotclearStat & CLEAR_START)
    {
        if(!clearCoprocStats(phys))
        {
            fprintf(fp,"\n Failed to clear stats before run start!! \n"); 
            printf("\n Failed to clear stats before run start!! \n"); 
        }
    }

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
		
        else if (m_Count >= m_numCycles)  // test has completed
        {
            // Dummy init to allow transition from last hold interval to complete
            if (m_Count)
            {
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
            }

            fprintf_s(fp," \n Total cycles completed %u / %u ",m_Count, m_numCycles);
			fprintf_s(fp," \n Cycles completed in this hold interval : %u / %u \n",(HoldInterval - Counter),HoldInterval);
            
            printf_s(" \n Total cycles completed %u / %u ",m_Count, m_numCycles);
			printf_s(" \n Cycles completed in this hold interval : %u / %u \n",(HoldInterval - Counter),HoldInterval);
            
            if (bCheckForGOLDTests)   // GOLD
            {
                LW_COPROC_GOLD_STATISTICS goldStats  = {0};
                goldStats.version                    = LW_COPROC_GOLD_STATISTICS_VER2;
                LwAPI_Status status = LwAPI_Coproc_GetGoldStatisticsEx(phys, false, &goldStats);

                if(status != LWAPI_OK)
                {
                    fprintf(fp, "\n Getting goldStats failed! \n");
                    printf("\n Getting goldStats failed! \n");
                    fclose(fp);
                    return;
                }
                fprintf(fp, "\n GOLD transitions in this hold interval : %u \n", (goldStats.dwGoldTransitionCount - LastKnownTransitionCount));
                fprintf(fp, "\n Total GOLD transition count : %u ", goldStats.dwGoldTransitionCount);        
				printf("\n GOLD transitions in this hold interval : %u \n", (goldStats.dwGoldTransitionCount - LastKnownTransitionCount));
                printf("\n Total GOLD transition count : %u ", goldStats.dwGoldTransitionCount);
            }
            else   // GC6
            {
                LW_GPU_GC6_STATISTICS gc6Stats         = {0};
                gc6Stats.version                       = LW_GPU_GC6_STATISTICS_VER9;
                gc6Stats.bEnableLPWRInfo               = LW_GPU_FEATURE_LPWR_INFO_DISABLE;
                LwAPI_Status status = LwAPI_GPU_GetGC6Statistics(phys, &gc6Stats);
                if(status != LWAPI_OK)
                {
                    fprintf(fp, "\n Getting gc6Stats failed! \n");
                    printf("\n Getting gc6Stats failed! \n");
                    fclose(fp);
                    return;
                }

                fprintf(fp, "\n GC6 transitions in this hold interval : %u \n", (gc6Stats.GC6TransitionCount - LastKnownTransitionCount));
                fprintf(fp, "\n Total GC6 transition count : %u ", gc6Stats.GC6TransitionCount);
				printf("\n GC6 transitions in this hold interval : %u \n", (gc6Stats.GC6TransitionCount - LastKnownTransitionCount));
                printf("\n Total GC6 transition count : %u ", gc6Stats.GC6TransitionCount);
                
            }

            done = true;
            break;
        }
        
        if (Counter == 0)   // new hold interval to start
        {
            // Dummy init to allow transition from previous run to complete
            if (m_Count)
            {
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
            }

            if (m_Count) fprintf(fp," \n Hold interval completed. \n");
            
            // Get GC6 / Gold transition count obtained so far
            if (bCheckForGOLDTests)   // GOLD
            {
                LW_COPROC_GOLD_STATISTICS goldStats  = {0};
                goldStats.version                    = LW_COPROC_GOLD_STATISTICS_VER2;
                LwAPI_Status status = LwAPI_Coproc_GetGoldStatisticsEx(phys, false, &goldStats);

                if(status != LWAPI_OK)
                {
                    fprintf(fp, "\n Getting goldStats failed! \n");
                    printf("\n Getting goldStats failed! \n");
                    fclose(fp);
                    return;
                }
                if (m_Count) 
				{
					fprintf(fp, "\n GOLD transitions in this hold interval : %u \n", (goldStats.dwGoldTransitionCount - LastKnownTransitionCount));
					printf("\n GOLD transitions in this hold interval : %u \n", (goldStats.dwGoldTransitionCount - LastKnownTransitionCount));
				}
				fprintf(fp, "\n Total GOLD transition count : %u ", goldStats.dwGoldTransitionCount);
				printf("\n Total GOLD transition count : %u ", goldStats.dwGoldTransitionCount);
				LastKnownTransitionCount = goldStats.dwGoldTransitionCount;
            }
            else   // GC6
            {
                LW_GPU_GC6_STATISTICS gc6Stats         = {0};
                gc6Stats.version                       = LW_GPU_GC6_STATISTICS_VER9;
                gc6Stats.bEnableLPWRInfo               = LW_GPU_FEATURE_LPWR_INFO_DISABLE;
                LwAPI_Status status = LwAPI_GPU_GetGC6Statistics(phys, &gc6Stats);
                if(status != LWAPI_OK)
                {
                    fprintf(fp, "\n Getting gc6Stats failed! \n");
                    printf("\n Getting gc6Stats failed! \n");
                    fclose(fp);
                    return;
                }
                if (m_Count)
				{
					fprintf(fp, "\n GC6 transitions in this hold interval : %u \n", (gc6Stats.GC6TransitionCount - LastKnownTransitionCount));
					printf("\n GC6 transitions in this hold interval : %u \n", (gc6Stats.GC6TransitionCount - LastKnownTransitionCount));
				}
                fprintf(fp, "\n Total GC6 transition count : %u ", gc6Stats.GC6TransitionCount);
				printf("\n Total GC6 transition count : %u ", gc6Stats.GC6TransitionCount);
                LastKnownTransitionCount = gc6Stats.GC6TransitionCount;
            }
            fprintf(fp,"\n Total cycles completed : %u / %u \n",m_Count, m_numCycles);

            fprintf(fp, "\n New hold interval starting ...\n");
			printf("\n Total cycles completed : %u / %u \n",m_Count, m_numCycles);

            printf("\n New hold interval starting ...\n");
            // randomly generate new delayinterval and triangle count
            m_triCount = (rand()% triRange) + MinTriangles;
            m_delayInterval = (rand()% delayRange) + MinDelayInterval;
            
            fprintf(fp, "\n Triangle Count is now set to %u \n Delay interval is now set to %u ms \n", m_triCount, m_delayInterval);
			printf("\n Triangle Count is now set to %u \n Delay interval is now set to %u ms \n", m_triCount, m_delayInterval);
            Counter = HoldInterval;
        }

        if (bCheckForGOLDTests)       // do we need GOLD cycles
        {
            if(!(InitializeForGOLDTests()))
            {
                fprintf(fp,"\n Initialization for GOLD tests failed !! \n");
				printf("\n Initialization for GOLD tests failed !! \n");
            }            
        }
        else if(bIsPStateForced)   // GC6 cycles - force PState change to P0 only if the '/f' option is specified by the user.
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
        else if(bIsPStateForced)    // GC6 cycles - force PState change to P8 only if the '/f' option is specified by the user.
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

        Counter--;
		fprintf(fp,"\n sleeping for %d time  ", m_delayInterval);
		printf("\n sleeping for %d time  ", m_delayInterval);
        Sleep((m_delayInterval));
		
	}

    if(!(timeEndPeriod(wTimerRes) == TIMERR_NOERROR))    // Clear the minimum timer resolution
    {
        fprintf(fp, "\n Error in clearing minimum timer resolution \n");
		printf("\n Error in clearing minimum timer resolution \n");
    }
	getAndShowGC6Statistics(phys, false, fp);
    getAndShowGOLDStatistics(phys, false, (doNotclearStat & CLEAR_END), fp);

	fclose(fp);
	
    return;
}
