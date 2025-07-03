////////////////////////////////////////////////////////////////////////////////
// Filename: systemclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _SYSTEMCLASS_H_
#define _SYSTEMCLASS_H_


///////////////////////////////
// PRE-PROCESSING DIRECTIVES //
///////////////////////////////
#define WIN32_LEAN_AND_MEAN

/////////////
// LINKING //
/////////////
#pragma comment(lib, "winmm.lib")

//////////////
// INCLUDES //
//////////////
#include <windows.h>
#include <Mmsystem.h>


///////////////////////
// MY CLASS INCLUDES //
///////////////////////
#include "inputclass.h"
#include "graphicsclass.h"
#include <lwapi.h>

typedef enum _CLEAR_TOOL_STATS
{
    CLEAR_NONE              = 0,
    CLEAR_END               = 1,
    CLEAR_START             = 2,
    CLEAR_END_AND_START     = 3
} CLEAR_TOOL_STATS;

////////////////////////////////////////////////////////////////////////////////
// Class name: SystemClass
////////////////////////////////////////////////////////////////////////////////
class SystemClass
{
public:
    SystemClass();
    SystemClass(const SystemClass&);
    ~SystemClass();

	bool Initialize(DWORD initialTris, DWORD numCycles, DWORD delayInterval, bool FULL_SCREEN, bool bGOLDOptionCheck, DWORD numWasteMB);
    void Shutdown(bool FULL_SCREEN);
    void Run(bool bIsPStateForced, CLEAR_TOOL_STATS doNotclearStat);
    void Run_TargetCycles(bool bIsPStateForced,DWORD TargetCycles,DWORD DelayIntervalStep, int StepDirection, CLEAR_TOOL_STATS doNotclearStat);
    void Run_MinMaxHold(bool bIsPStateforced, DWORD MaxTriangles, DWORD MaxDelayInterval, DWORD HoldInterval, CLEAR_TOOL_STATS doNotclearStat);
	void errorParam(FILE *fp);
	bool Calibrate(DWORD presentModel, bool FULL_SCREEN,float CalibrateTime);
	void RedirectIOToConsole();
	DWORD Increment(DWORD NumTraingles);
    LRESULT CALLBACK MessageHandler(HWND, UINT, WPARAM, LPARAM);
    void ClearCoprocCycles(bool bResetCycles, LwU64 cycleCount);

private:
    unsigned int Frame();
    void InitializeWindows(int&, int&, bool);
    
    bool InitializeForGOLDTests();
    void DestroyForGOLDTests();
    
    void ShutdownWindows(bool);

private:
    LPCWSTR m_applicationName;
    HINSTANCE m_hinstance;
    HWND m_hwnd;

    InputClass* m_Input;
    GraphicsClass* m_Graphics;

    DWORD m_timeStamp;
    DWORD m_frameCount;
    DWORD m_triCount;
    DWORD m_numCycles;
    DWORD m_delayInterval;
    float m_TotalScore;
    float m_AverageScore;
	float m_CalibrateTime;
    bool m_IsMaxFrameRateMode;
    DWORD m_Count;
	DWORD m_CalibrateCount;
	DWORD m_CalibrateTriangles;
};


/////////////////////////
// FUNCTION PROTOTYPES //
/////////////////////////
static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);


/////////////
// GLOBALS //
/////////////
static SystemClass* ApplicationHandle = 0;


#endif