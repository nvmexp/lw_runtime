#ifndef _SCINTERFACE_H
#define _SCINTERFACE_H

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <process.h>


class BoardVD
{
    HANDLE  m_StopEvent;
    HANDLE  m_ThreadHandle;
private:
public:
    BoardVD()
    {
        m_StopEvent = CreateEvent(0, 1, 0, 0); // This event will be closed by thread itself.
        m_ThreadHandle = (HANDLE)-1;
    };

    ~BoardVD()
    {
        CloseHandle(m_StopEvent);
    }

    void Stop()
    {
        SetEvent(m_StopEvent);
    }

    void WaitForStop()
    {
        if (m_ThreadHandle != (HANDLE)-1)
        {
            while (WaitForSingleObject(m_ThreadHandle,1000) == WAIT_TIMEOUT)
            {
                // NOP
            }
        }
    }

    // To stop validation thread set event returned by Validate function;
    void Validate (bool Sincronious=NULL)
    {
        m_ThreadHandle = (HANDLE)_beginthread(Run,0,m_StopEvent);
        if (m_ThreadHandle != (HANDLE)-1)
        {   // Started successfully
            // Wait for thread end.
            if (Sincronious)
            {
                while (WaitForSingleObject(m_ThreadHandle,1000) == WAIT_TIMEOUT)
                {
                    // NOP
                }
            }
        }
        return ;
    };

private:
#if defined (_WIN64) || defined (_DEBUG)
    static void __cdecl Run(void * Event)
    {
        typedef int (_cdecl pExe) (HANDLE);
        bool    rc = false;
        HMODULE hModule;
        pExe*   pDllRunExe;
        if (hModule = LoadLibrary("LW4_P0.dll"))
        {
            try
            {
                if (pDllRunExe=(pExe*)GetProcAddress(hModule,"Exe"))
                {
                    if (pDllRunExe((HANDLE)Event) == 0)
                    {
                        rc = true;
                    }
                }
            }
            catch(...)
            {
            }
            FreeLibrary(hModule);
            OutputDebugString(_T("Dll unloaded\n"));
        }
    }
#else   //!_WIN64 && !_DEBUG
    static void __cdecl Run(void * Event)
    {
        STARTUPINFOA            si;
        PROCESS_INFORMATION     pi;
        char    Buffer[_MAX_PATH+1];
        DWORD    Size = ExpandElwironmentStringsA("%SystemRoot%\\System32",Buffer,sizeof(Buffer));
        if (Size < sizeof(Buffer))
        {
        char    Args[2*_MAX_PATH+1];
            strcpy(Args,"Rundll32.exe ");
            strcat(Args,Buffer);
            strcat(Args,"\\lw4_p0.dll,DllEntry");
            strcat(Buffer,"\\Rundll32.exe");

            GetStartupInfoA(&si);
            CreateProcessA(Buffer,
                            Args,
                            NULL,
                            NULL,
                            TRUE,
                            NULL,
                            NULL,
                            NULL,
                            &si,
                            &pi);
        }
        if (Event)
        {
            SetEvent((HANDLE)Event);
        }
    }
#endif  //!_DEBUG
};
#endif //_SCINTERFACE_H
