#define ANSEL_SDK_EXPORTS
#include "ansel/Session.h"
#include <ansel/Defines.h>
#include <windows.h>
#include <stdio.h>

namespace ansel
{
    void* s_userData = nullptr;

    namespace
    {
        typedef void(*SessionFunc)(void* userData);
        SessionFunc s_startSessionFunc = nullptr;
        SessionFunc s_stopSessionFunc = nullptr;

        // This may seem like a funky construct so allow me to explain. This code serves only 
        // one purpose: to create a signaling object that Ansel (when loaded from driver shim)
        // can use to detect that a game has integrated the SDK.
        class SdkMutex
        {
        private:
            HANDLE m_handle = NULL;
        public:
            SdkMutex() 
            { 
                DWORD id = GetLwrrentProcessId();
                char name[MAX_PATH];
                sprintf_s(name, "LWPU/Ansel/%d", id);
                m_handle = CreateMutexA(NULL, false, name);
            }
            ~SdkMutex()
            {
                CloseHandle(m_handle);
            }
        };

        SdkMutex s_mySdkMutex;
    }

    void startSession()
    {
        if (s_startSessionFunc)
            s_startSessionFunc(s_userData);
    }

    void stopSession()
    {
        if (s_stopSessionFunc)
            s_stopSessionFunc(s_userData);
    }

    ANSEL_SDK_INTERNAL_API void setSessionFunctions(SessionFunc start, SessionFunc stop,
        void* userData)
    {
        s_startSessionFunc = start;
        s_stopSessionFunc = stop;
        s_userData = userData;
    }

    // This is an old relic that we are forced to keep around for backwards compatibility
    // Old versions of LwCamera DLL look for this entry point (do nothing with it) and
    // will fail to load the SDK if it's not there.
    // Never use this code for anything.
    typedef void(*StopSessionFunc)();
    ANSEL_SDK_INTERNAL_API void setStopSessionCallback(StopSessionFunc stopSessionFunc)
    {
        (void)stopSessionFunc;
        // intentionally do nothing!
    }
    // End of backwards compatibility support
}
