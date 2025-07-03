#define ANSEL_CONTROL_SDK_EXPORTS
#include <anselcontrol/Defines.h>
#include <anselcontrol/Common.h>
#include <anselcontrol/Configuration.h>
#include <anselcontrol/Interface.h>
#include <windows.h>
#include <stdio.h>
#include <string>

namespace anselcontrol
{
    namespace
    {
        typedef void(*SessionFunc)(void* userData);
        SessionFunc s_startSessionFunc = nullptr;
        SessionFunc s_stopSessionFunc = nullptr;
        void* s_userData = nullptr;

        // This may seem like a funky construct so allow me to explain. This code serves only 
        // one purpose: to create a signaling object that Ansel (when loaded from driver shim)
        // can use to detect that a game has integrated the SDK.
        class ControlSdkMutex
        {
        public:
            ControlSdkMutex() 
            { 
                DWORD id = GetLwrrentProcessId();
                char name[MAX_PATH];
                sprintf_s(name, "LWPU/AnselControl/%d", id);
                CreateMutexA(NULL, false, name);
            }
        };

        ControlSdkMutex s_myControlSdkMutex;
    }

    // State
    bool s_anselServerIsAvailable = false;

    bool s_anselControlIsAvailable = false;
    
    bool s_anselConfigurationChanged = true;
    Configuration s_config;

    bool lastCaptureUtf8PathLocked = false;
    std::string lastCaptureUtf8Path;
    std::string lastCaptureUtf8PathVolatile;

    // Public facing
    SetConfigurationStatus setConfiguration(const Configuration& config)
    {
        if (config.sdkVersion != ANSEL_CONTROL_SDK_VERSION)
        {
            s_anselControlIsAvailable = false;
            return kSetConfigurationIncompatibleVersion;
        }

        // in case we have a correct configuration
        s_config = config;
        s_anselConfigurationChanged = true;
        s_anselControlIsAvailable = true;
        return kSetConfigurationSuccess;
    }

    Status lockLastCaptureAbsolutePath(const char ** utf8Path)
    {
        if (!s_anselServerIsAvailable)
        {
            return kControlNotInitialized;
        }

        if (utf8Path)
            *utf8Path = lastCaptureUtf8Path.data();
        lastCaptureUtf8PathLocked = true;

        return kControlSuccess;
    }

    Status unlockLastCaptureAbsolutePath()
    {
        if (!s_anselServerIsAvailable)
        {
            return kControlNotInitialized;
        }

        lastCaptureUtf8PathLocked = false;
        lastCaptureUtf8Path = lastCaptureUtf8PathVolatile;

        return kControlSuccess;
    }

    // Internal
    ANSEL_CONTROL_SDK_INTERNAL_API void reportServerControlVersion(uint64_t serverControlVersion)
    {
        s_anselServerIsAvailable = true;
    }
    ANSEL_CONTROL_SDK_INTERNAL_API bool getControlConfigurationChanged()
    {
        return s_anselConfigurationChanged;
    }
    ANSEL_CONTROL_SDK_INTERNAL_API void getControlConfiguration(Configuration& cfg)
    {
        cfg = s_config;
        s_anselConfigurationChanged = false;
    }
    ANSEL_CONTROL_SDK_INTERNAL_API uint32_t getControlConfigurationSize()
    {
        return sizeof(Configuration);
    }
    ANSEL_CONTROL_SDK_INTERNAL_API void initializeControlConfiguration(Configuration& cfg)
    {
        cfg = Configuration();
    }

    ANSEL_CONTROL_SDK_INTERNAL_API bool ilwalidateAllState()
    {
        ilwalidateCaptureShotScenarioState();
        return true;
    }

    ANSEL_CONTROL_SDK_INTERNAL_API bool setLastCaptureUtf8Path(const char * utf8Path)
    {
        lastCaptureUtf8PathVolatile = utf8Path;
        if (!lastCaptureUtf8PathLocked)
        {
            lastCaptureUtf8Path = lastCaptureUtf8PathVolatile;
            return true;
        }
        return false;
    }
}
