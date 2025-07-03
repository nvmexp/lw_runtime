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

////////////////////////////////////////////////////////////////////////////////
// Class name: SystemClass
////////////////////////////////////////////////////////////////////////////////
class SystemClass
{
public:
    SystemClass();
    SystemClass(const SystemClass&);
    ~SystemClass();

    bool Initialize( DWORD initialTris, float incrementRatio, float fpsLwtoff, DWORD presentModel, bool FULL_SCREEN, DWORD dxgiFormat = 28, bool bWindowedFullScreenTransition = false, bool bDXTLAutomationTesting = false);
    void Shutdown(bool FULL_SCREEN);
    void Run();
    void setSize(int width, int height)
    {
		SetWindowPos(m_hwnd, HWND_TOP, NULL, NULL, width, height, SWP_NOACTIVATE | SWP_NOMOVE);
	}
    LRESULT CALLBACK MessageHandler(HWND, UINT, WPARAM, LPARAM);

private:
    unsigned int Frame();
    void InitializeWindows(int&, int&, bool);
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
    float m_incrementRatio;
    float m_fpsLwtoff;
    float m_TotalScore;
    float m_AverageScore;
    bool m_IsMaxFrameRateMode;
	bool m_bWindowedFullScreenTransition;
	bool m_FullScreen;
	bool m_bDXTLAutomationTesting;
    DWORD m_Count;
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