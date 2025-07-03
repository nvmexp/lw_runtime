#include <Windows.h>

#define ANSEL_DLL_EXPORTS
#define ANSEL_ALLOWLISTING_ONLY
#include "Ansel.h"
#include <lwapi.h>
// Trick LWDRS lib into thinking that types are already defined
// They actually are, in LWAPI.h
#define _LWTYPES_H_

#include "Allowlisting.h"
#include "darkroom/StringColwersion.h"
#include "RegistrySettings.h"
#include "DenylistParser.h"

#include <stdio.h>

#   include <Shlwapi.h>
#   include <shlobj.h>

#   include "Log.h"
#   include "Log.cpp"

#define LWAPI_DRS_ADDITIONAL_LOGGING
#include "drs/LwDrsWrapper.cpp"

#include <LwApiDriverSettings.h>

extern "C"
{

class AllowlistingLog
{
public:
    AllowlistingLog()
    {
        RegistrySettings registry;
        int32_t severity = registry.getValue(registry.registryPathAnsel(), L"LogFiltering",
            (int32_t)LogSeverity::kFirst, (int32_t)LogSeverity::kLast,
            (int32_t)LogSeverity::kDisableLogging);
        if (severity == (int32_t)LogSeverity::kDisableLogging)
        {
            // Do not do anything if logging is not enabled
            return;
        }

        m_enabled = true;

        initLog();

        typedef BOOL (__stdcall * pfnPathIsRootW_t)(LPCWSTR pszPath);
        typedef BOOL (__stdcall * pfnPathIsDirectoryW_t)(LPCWSTR pszPath);
        typedef BOOL (__stdcall * pfnPathFileExistsW_t)(LPCWSTR pszPath);

        typedef LPWSTR (__stdcall * pfnPathFindFileNameW_t)(LPCWSTR pszPath);
        typedef LPWSTR (__stdcall * pfnPathFindExtensionW_t)(LPCWSTR pszPath);

        pfnPathIsRootW_t pfnPathIsRootW = nullptr;
        pfnPathIsDirectoryW_t pfnPathIsDirectoryW = nullptr;
        pfnPathFileExistsW_t pfnPathFileExistsW = nullptr;
        pfnPathFindFileNameW_t pfnPathFindFileNameW = nullptr;
        pfnPathFindExtensionW_t pfnPathFindExtensionW = nullptr;

        HMODULE shlwDLL = LoadLibrary(L"Shlwapi.dll");
        if (shlwDLL)
        {
            pfnPathIsRootW = (pfnPathIsRootW_t)GetProcAddress(shlwDLL, "PathIsRootW");
            pfnPathIsDirectoryW = (pfnPathIsDirectoryW_t)GetProcAddress(shlwDLL, "PathIsDirectoryW");
            pfnPathFileExistsW = (pfnPathFileExistsW_t)GetProcAddress(shlwDLL, "PathFileExistsW");
            pfnPathFindFileNameW = (pfnPathFindFileNameW_t)GetProcAddress(shlwDLL, "PathFindFileNameW");
            pfnPathFindExtensionW = (pfnPathFindExtensionW_t)GetProcAddress(shlwDLL, "PathFindExtensionW");
        }

        std::wstring channelsSeverityConfig = registry.getValue(registry.registryPathAnsel(),
            L"LogChannelsFiltering", L"");

        setLogSeverity((LogSeverity)severity, channelsSeverityConfig.c_str());

        auto getAppNameFromProcess = [&pfnPathFindFileNameW, &pfnPathFindExtensionW]() -> std::wstring
        {
            wchar_t appPath[MAX_PATH];
            GetModuleFileName(NULL, appPath, MAX_PATH);
            const wchar_t* fileNameInPath = pfnPathFindFileNameW(appPath);
            const wchar_t* fileNameExtension = pfnPathFindExtensionW(appPath);
            return std::wstring(fileNameInPath, fileNameExtension);
        };

        // Getting the app name from the module name (EXE)
        std::wstring appName = getAppNameFromProcess();

        // Create log filename
        std::wstring logDir = registry.getValue(registry.registryPathAnsel(), L"LogDir", L"");
        if (logDir.empty())
        {
            WCHAR path[MAX_PATH];
            if (SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_PROFILE, NULL, 0, path)))
            {
                logDir = path;
                logDir += L"\\ansel\\";
            }
        }

        auto directoryExists = [&pfnPathIsRootW, &pfnPathIsDirectoryW, &pfnPathFileExistsW](const wchar_t *path) -> bool
        {
            if (pfnPathIsRootW(path))
                return true;

            if (pfnPathIsDirectoryW(path))
                return true;

            if (pfnPathFileExistsW(path))
                return true;

            return false;
        };

        auto createDirectoryRelwrsively = [&directoryExists](const wchar_t *path) -> bool
        {
            if (directoryExists(path))
                return true;

            wchar_t folder[MAX_PATH];
            ZeroMemory(folder, MAX_PATH * sizeof(wchar_t));

            const wchar_t* endBackslash = wcschr(path, L'\\'), *endFrontslash = wcschr(path, L'/');
            const wchar_t* endFirst = endBackslash < endFrontslash ? endBackslash : endFrontslash;
            const wchar_t* endLast = endBackslash >= endFrontslash ? endBackslash : endFrontslash;
            const wchar_t* end = endFirst ? endFirst : endLast; //if the pointer is zero, try the other one

            bool success = true;
            while (end != NULL)
            {
                wcsncpy_s(folder, path, end - path + 1);

                if (!directoryExists(folder))
                {
                    if (!CreateDirectory(folder, NULL))
                    {
                        success = false;
                        break;
                    }
                }

                end += 1;
                endBackslash = wcschr(end, L'\\');
                endFrontslash = wcschr(end, L'/');
                endFirst = endBackslash < endFrontslash ? endBackslash : endFrontslash;
                endLast = endBackslash >= endFrontslash ? endBackslash : endFrontslash;
                end = endFirst ? endFirst : endLast; //if the pointer is zero, try the other one
            }
            return success;
        };

        auto appendTimeW = [](const wchar_t * inString_pre, const wchar_t * inString_post) -> std::wstring
        {
            std::chrono::time_point<std::chrono::system_clock> lwrrentTime;
            lwrrentTime = std::chrono::system_clock::now();

            long long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(lwrrentTime.time_since_epoch()).count();

            std::time_t tt = std::chrono::system_clock::to_time_t(lwrrentTime);

            char time[32];
            tm buf;
            // TODO: VS version of localtime_s seems to have weird argument sequence
            localtime_s(&buf, &tt);
            std::strftime(time, 32, "%Y%m%d_%H%M%S_", &buf);

            int ms_cnt = milliseconds % 1000;

            std::wstringstream in;

            in << inString_pre << darkroom::getWstrFromUtf8(time) << std::setw(3) << std::setfill(L'0') << ms_cnt;

            if (inString_post)
                in << inString_post;

            return in.str();
        };

        if (!logDir.empty())
        {
            // make sure directory exists:
            bool ok = createDirectoryRelwrsively(logDir.c_str());
            if (ok)
            {
                std::wstring filename = logDir;
                filename += L"ansel_Allowlisting_";
                filename += appName;
                filename += L"-";
                filename = appendTimeW(filename.c_str(), L".log");
                setLogFilename(filename.c_str());
                LOG_INFO("Log file opened");
                LOG_INFO("--LwCameraAllowlisting Log--");
                LOG_INFO("Log level: %s", getLogSeverityName((LogSeverity)severity));
            }
        }

        if (shlwDLL)
            FreeLibrary(shlwDLL);
    }
    ~AllowlistingLog()
    {
        if (m_enabled)
        {
            m_enabled = false;
            deinitLog();
        }
    }
private:
    bool m_enabled = false;
};

ANSEL_DLL_API void __cdecl AnselGetVersion(DWORD * pDwMajor, DWORD * pDwMinor)
{
    *pDwMajor = ANSEL_VERSION_MAJOR;
    *pDwMinor = ANSEL_VERSION_MINOR;
}

ANSEL_DLL_API bool __cdecl AnselEnableCheck()
{
    AllowlistingLog wl; // Handles both initialization and automatic deinitialization

    bool isAllowed = false;

    // The logic for Ansel/Freestyle Allowlisting needs to be spelled out here. These are the steps:
    // 0. Before this function is called, in Ansel's driver shim, ANSEL_ENABLE is checked.
    //      ANSEL_ENABLE is a global, system-wide setting that defaults to ON.
    //      ANSEL_ENABLE can be set to OFF to prevent the Ansel shim from being created on a system-wide basis.
    //      This function, AnselEnableCheck() is called after the check for ANSEL_ENABLE has passed in the driver shim.
    // 1. ANSEL_ALLOW is checked. This is a per-game setting and defaults to allowed.
    //      This can be used to prevent the Ansel shim from being created on a per-game basis.
    //      If this value is set to DISALLOWED, we return false, otherwise continue checking.
    // 2. Get ANSEL_ALLOWLISTED key for game or global profile if game doesn't exist
    // 3. If ANSEL_ALLOWLISTED key is set and set to non-zero (ALLOWED) we return true
    // 4. If ANSEL_ALLOWLISTED key is set and set to zero (DISALLOWED) we continue to check Freestyle allowlisting
    // 5. If ANSEL_ALLOWLISTED key is not set, we look for the SDK integration mutex, if we
    //      find it we return true - otherwise we continue on to check Freestyle allowlisting
    //      This allows users to control the enabling of Ansel per title and also globally
    //      while the default behavior is to determine enablement based on SDK integration (mutex).
    //
    // If we get here, then we check to see if Ansel is enabled for Freestyle.
    // 6. Check ANSEL_ALLOW_FREESTYLE_MODE. This is a global system-wide setting that can be used to denylist
    //      freestyle for the whole system. It defaults to allowed.
    //      If this value is set to DISABLED, we return false, otherwise continue checking.
    // 7. Check ANSEL_FREESTYLE_MODE. This value is per-game and defaults to DISABLED.
    //      If this value is anything other than ENABLED or APPROVED_ONLY, we return false.
    //      If this value is ENABLED or APPROVED_ONLY, we return true.
    //      A vaule of ENABLED will allow all filters to be available in Freestyle, regardless of hashing.
    //      A vaule of APPROVED_ONLY will only allow hashed filters to be available in Freestyle.

    const size_t appPathMaxLength = 1024;
    WCHAR appPath[appPathMaxLength];
    GetModuleFileName(NULL, appPath, appPathMaxLength);

    std::wstring appName = appPath;
    appName = appName.substr(appName.find_last_of(L"\\") + 1);
    LOG_INFO("Loaded App: \"%ls\"", appName.c_str());
    std::wstring profileName;
    drs::getProfileName(profileName);
    if (profileName.empty())
    {
        LOG_INFO("No DRS profile found!");
    }
    else
    {
        LOG_INFO("DRS Profile name: \"%ls\"", profileName.c_str());
    }

    LOG_DEBUG("Reading if LwCamera is denylisted for this title...");
    uint32_t value = 0;
    bool isKeySet = drs::getKeyValue(ANSEL_ALLOW_ID, value);
    if (isKeySet && !value)
    {
        // The title is denylisted, this overrides any other behavior
        LOG_INFO("Title is denylisted! Ansel is disabled. LwCamera will not be loaded.");
        return false;
    }
    else
    {
        LOG_DEBUG("Title is **NOT** denylisted for LwCamera. Continuing to verify allowlisting...");
    }

    DWORD pid = GetLwrrentProcessId();

    // First check if Ansel is allowed. If it is, no need to perform further checks
    LOG_DEBUG("Reading allowlisted ID...");
    value = 0;
    isKeySet = drs::getKeyValue(ANSEL_ALLOWLISTED_ID, value);
    if (isKeySet)
    {
        LOG_INFO("Allowlisted ID is set and has value %i", value);
        if (value)
            isAllowed = true;
    }
    else
    {
        LOG_INFO("Allowlisted ID is not set");

        char name[MAX_PATH];
        sprintf_s(name, "LWPU/Ansel/%d", pid);
        HANDLE mutex = OpenMutexA(SYNCHRONIZE, false, name);

        bool isSdkIntegrated = mutex ? true : false;
        if (isSdkIntegrated)
        {
            LOG_INFO("SDK integration detected");
            CloseHandle(mutex);
            isAllowed = true;
        }
        else
        {
            LOG_INFO("SDK integration *not* detected (possibly pre-1.2 SDK)");
        }
    }

    LOG_INFO("Application is %s for Ansel", (isAllowed ? "enabled" : "disabled"));

    bool shouldLoadAnsel = isAllowed;

    // If Ansel is not allowed in the title, we can start checking if LwCamera dll still needs to be loaded by the UMD
    if (!shouldLoadAnsel)
    {
        // Using same object as AnselServer uses to avoid reg path discrepancies
        RegistrySettings regSettings;
        // If modding main switch is set to "disable modding" - no need to parse the DRS mode
        bool isFreestyleEnabled = false;

        LOG_DEBUG("Reading if freestyle is allowed on this system...");
        uint32_t isFreestyleEnabledValDRS;
        isKeySet = drs::getKeyValue(ANSEL_ALLOW_FREESTYLE_MODE_ID, isFreestyleEnabledValDRS);
        if (isKeySet)
        {
            isFreestyleEnabled = (isFreestyleEnabledValDRS != 0);
        }
        else
        {
            if (regSettings.valueExists(regSettings.registryPathAnsel(), L"", darkroom::getWstrFromUtf8(Settings::FreestyleEnabled).c_str()))
            {
                isFreestyleEnabled = regSettings.getValue(regSettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::FreestyleEnabled).c_str(), false);
            }
            else
            {
                isFreestyleEnabled = regSettings.getValue(regSettings.registryPathAnselBackup(), darkroom::getWstrFromUtf8(Settings::FreestyleEnabled).c_str(), false);
            }

        }

        if (isFreestyleEnabled)
        {
            LOG_DEBUG("Freestyle is allowed on this system.");
        }
        else
        {
            LOG_WARN("Freestyle is **NOT** allowed on this system.");
        }
        if (isFreestyleEnabled)
        {
            uint32_t freeStyleModeValue = getFreeStyleMode(pid, NULL);
            switch (freeStyleModeValue)
            {
                // Archived dislwssion about freestyle modes https://docs.google.com/document/d/13uaIzUHb_GaY0mjiEqTOnXSoPWEQXVcEQYuaJIpG2iM/edit
                // Mitigation of potential cheating/exploitation is now addressed by a parsed denylisting DRS value that allows denylisting specific filters and buffers for specific Ansel build numbers.

                //case ANSEL_FREESTYLE_MODE_MULTIPLAYER_DISABLED:
                //case ANSEL_FREESTYLE_MODE_MULTIPLAYER_APPROVED_ONLY:
                //case ANSEL_FREESTYLE_MODE_MULTIPLAYER_DISABLE_EXTRA_BUFFERS:
                //case ANSEL_FREESTYLE_MODE_MULTIPLAYER_DISABLE_DEPTH:
                case ANSEL_FREESTYLE_MODE_ENABLED:
                case ANSEL_FREESTYLE_MODE_APPROVED_ONLY:
                {
                    shouldLoadAnsel = true;
                }
            };
        }
    }

    LOG_INFO("Application will %sload LwCamera...", (shouldLoadAnsel ? "" : "*NOT* "));
    return shouldLoadAnsel;
}

}

BOOL APIENTRY DllMain(HINSTANCE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        break;

    case DLL_THREAD_ATTACH:
        break;

    case DLL_THREAD_DETACH:
        break;

    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

