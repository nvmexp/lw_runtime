#include "LwCameraVersion.h"

#include <Windows.h>

#include <array>
#include <string>

namespace
{
    bool s_initialized = false;
    internal::Version s_version;

    HMODULE findLwCameraModule()
    {
        const std::array<std::wstring, 2> anselSDKnames = {
#if _M_AMD64
            L"LwCamera64.dll",
            L"LwCamera64.dll",
#else
            L"LwCamera32.dll",
            L"LwCamera32.dll",
#endif
        };

        HMODULE hLwCamera = NULL;

        for (auto& moduleName : anselSDKnames)
            if (hLwCamera = GetModuleHandle(moduleName.c_str()))
                break;

        return hLwCamera;
    }

    bool initialize()
    {
        DWORD  verHandle = 0;
        const HMODULE handle = findLwCameraModule();
        TCHAR lwCameraPath[MAX_PATH];
        if (!handle)
            return false;

        GetModuleFileName(handle, lwCameraPath, MAX_PATH);
        const DWORD verSize = GetFileVersionInfoSize(lwCameraPath, &verHandle);

        if (verSize != NULL)
        {
            LPSTR verData = new char[verSize];

            if (GetFileVersionInfo(lwCameraPath, verHandle, verSize, verData))
            {
                LPBYTE lpBuffer = NULL;
                UINT size = 0;
                if (VerQueryValue(verData, L"\\", (VOID FAR* FAR*)&lpBuffer, &size))
                {
                    if (size)
                    {
                        VS_FIXEDFILEINFO *verInfo = (VS_FIXEDFILEINFO *)lpBuffer;
                        if (verInfo->dwSignature == 0xfeef04bd)
                        {
                            // Doesn't matter if you are on 32 bit or 64 bit,
                            // DWORD is always 32 bits, so first two revision numbers
                            // come from dwFileVersionMS, last two come from dwFileVersionLS
                            s_version.major = (verInfo->dwFileVersionMS >> 16) & 0xffff;
                            s_version.minor = (verInfo->dwFileVersionMS) & 0xffff;
                            s_version.build = (verInfo->dwFileVersionLS >> 16) & 0xffff;
                            s_version.revision = (verInfo->dwFileVersionLS) & 0xffff;
                            s_initialized = true;
                        }
                    }
                }
            }
            delete[] verData;
        }

        return s_initialized;
    }
}

namespace internal
{
    bool getLwCameraVersion(Version& version)
    {
        if (s_initialized || initialize())
            version = s_version;

        return s_initialized;
    }
}
