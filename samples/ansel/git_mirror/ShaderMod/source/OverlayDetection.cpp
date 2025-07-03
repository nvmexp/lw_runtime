#include "OverlayDetection.h"
#include "ShadowPlayDDIShim.h" //for GFE Share overlay detection

// Overlay detection functions
static bool isInputActiveGOGSteamOrigin(unsigned int olIdx, HMODULE hModule)
{
    const int numOverlaysSupported = 3;
    assert(olIdx < numOverlaysSupported);

    char overlayFuncName[numOverlaysSupported][64] =
    {
        "_OverlayInputActive",          // GOG
#if defined(_M_IX86)
        "SteamOverlayIsUsingMouse",     // Steam (x86)
#else
        "SteamOverlayIsUsingMouse",     // Steam (x64)
#endif
        "IsIGOActive"                   //Origin
    };

    typedef bool(*GOGFunc)();

    GOGFunc pfnGOGFunc = (GOGFunc)GetProcAddress(hModule, overlayFuncName[olIdx]);

    if (pfnGOGFunc)
    {
        return pfnGOGFunc();
    }

    return false;
}

static bool isInputActiveGFEShare(unsigned int olIdx, HMODULE hModule)
{
    assert(olIdx == 3);

    QueryShadowPlayDdiShimInterface_FuncType QueryShadowPlayDdiShimInterface =
        (QueryShadowPlayDdiShimInterface_FuncType)GetProcAddress(hModule, "QueryShadowPlayDdiShimInterface");

    UINT ver;
    IShadowPlayDdiShimVer2 *iface = nullptr;

    if (QueryShadowPlayDdiShimInterface)
    {
        // Supported versions
        UINT version[] = { SHADOWPLAY_DDISHIM_INTERFACE_VER_6,
            SHADOWPLAY_DDISHIM_INTERFACE_VER_5,
            SHADOWPLAY_DDISHIM_INTERFACE_VER_4,
            SHADOWPLAY_DDISHIM_INTERFACE_VER_3,
            SHADOWPLAY_DDISHIM_INTERFACE_VER_2
        };

        static void *pProxyDDIShimInterface = NULL;

        for (UINT index = 0; index < (sizeof(version) / sizeof(version[0])); index++)
        {
            if ((S_OK == QueryShadowPlayDdiShimInterface(version[index], &pProxyDDIShimInterface)) &&
                pProxyDDIShimInterface && ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->CheckStatus)
            {
                if (((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->CheckStatus == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->CreateSession == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->DestroySession == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->GetSessionParam == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->SetSessionParam == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->DdiOpenAdapter == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->DdiCreateDevice == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->DdiDestroyDevice == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->DdiCreateResource == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->DdiDestroyResource == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->DdiPreSetDisplayMode == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->DdiPostSetDisplayMode == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->DdiPrePresent == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->DdiPostPresent == NULL ||
                    ((IShadowPlayDdiShimVer2 *)pProxyDDIShimInterface)->DdiBlt == NULL)
                {
                    pProxyDDIShimInterface = NULL;
                }

                iface = (IShadowPlayDdiShimVer2 *) pProxyDDIShimInterface;
                ver = version[index];
                break;
            }
        }
    }
    else
    {
        return false;
    }

    if (!iface)
    {
        return false;
    }

    unsigned int ret = 0;
    GetSessionParamArgs_V2 args;
    args.ver = DDISHIM_GETSESSIONPARAMARGS_VER_2;
    args.hDevice = 0;
    args.paramId = DdiGetSessionParamType_InputRedirectionStatus;
    args.pData = &ret;
    args.dataSize = sizeof(unsigned int);
    uint64_t dummyArg = SHADOWPLAY_DUMMY_ARG;
    HRESULT hres = iface->GetSessionParam(reinterpret_cast<HANDLE>(dummyArg), &args);

    if (SUCCEEDED(hres))
    {
        return ret != 0;
    }

    return false;
}

bool OverlayDetector::isOtherOverlayActive()
{
    // Processing overlays (we don't want both Ansel and e.g. Steam/GOG overlay to be active at the same time)
    const unsigned int overlaysNum = 4;
    wchar_t overlayDLLs[overlaysNum][32] =
    {
        L"d3d11.dll",                   // GOG
#if defined(_M_IX86)
        L"gameoverlayrenderer.dll",     // Steam (x86)
#else
        L"gameoverlayrenderer64.dll",   // Steam (x64)
#endif
#if defined(_M_IX86)
        L"igo32.dll",                   // Origin (x86)
#else
        L"igo64.dll",                   // Origin (x64)
#endif
#if defined(_M_IX86)
        L"lwspcap.dll",     // GFE Share (x86)
#else
        L"lwspcap64.dll",   // GFE Share (x64)
#endif
    };
    size_t overlayDLLNameLength[overlaysNum] =
    {
        9,                              // GOG
#if defined(_M_IX86)
        23,                             // Steam (x86)
#else
        25,                             // Steam (x64)
#endif
        9,                              // Origin (x86 / x64)
#if defined(_M_IX86)
        11,                             // GFE Share (x86)
#else
        13                              // GFE Share (x64)
#endif
    };

    typedef bool(*IsOverlayRunningFuncType)(unsigned int olIdx, HMODULE hModule);

    IsOverlayRunningFuncType overlayRunningFunc[overlaysNum] =
    {
        isInputActiveGOGSteamOrigin,                    // GOG
        isInputActiveGOGSteamOrigin,                    //Steam
        isInputActiveGOGSteamOrigin,                    //Origin
        isInputActiveGFEShare                           //GFE Share
    };

    DWORD cbNeeded;
    m_modulesEnumerated.resize(2048);
    size_t numModulesPreallocated = m_modulesEnumerated.size();

    HANDLE hLwrProcess = GetLwrrentProcess();

    if (EnumProcessModulesEx(hLwrProcess, &m_modulesEnumerated[0], static_cast<DWORD>(m_modulesEnumerated.size()) * sizeof(HINSTANCE), &cbNeeded, LIST_MODULES_ALL))
    {
        size_t numElementsReceived = cbNeeded / sizeof(HINSTANCE);
        m_modulesEnumerated.resize(numElementsReceived);

        // We need to re-run the function as we got more modukles than we initially allocated for
        if (numModulesPreallocated < numElementsReceived)
        {
            if (EnumProcessModulesEx(hLwrProcess, &m_modulesEnumerated[0], static_cast<DWORD>(m_modulesEnumerated.size()) * sizeof(HINSTANCE), &cbNeeded, LIST_MODULES_ALL))
            {
                numElementsReceived = cbNeeded / sizeof(HINSTANCE);
                if (numElementsReceived < m_modulesEnumerated.size())
                    m_modulesEnumerated.resize(numElementsReceived);
            }
            else
            {
                return false;
            }
        }
    }
    else
        return false;

    for (unsigned int olCnt = 0; olCnt < overlaysNum; ++olCnt)
    {
        bool isOverlayDetected = false;
        for (size_t dllCnt = 0, dllCntEnd = m_modulesEnumerated.size(); dllCnt < dllCntEnd; ++dllCnt)
        {
            wchar_t szModName[512]; // TODO avoroshilov: make something with the magic number and stack array

                                    // Get the full path to the module's file.
            DWORD moduleNameLen = GetModuleFileName(m_modulesEnumerated[dllCnt], szModName, sizeof(szModName) / sizeof(wchar_t));
            if (moduleNameLen >= overlayDLLNameLength[olCnt])
            {
                wchar_t * pathPart = szModName + moduleNameLen - overlayDLLNameLength[olCnt];
                _wcslwr_s(pathPart, overlayDLLNameLength[olCnt] + 1);   // +1 is required for it to not throw assertions about not-null-terminated string
                if (wcscmp(pathPart, overlayDLLs[olCnt]) == 0)
                {
                    isOverlayDetected = (overlayRunningFunc[olCnt])(olCnt, m_modulesEnumerated[dllCnt]);
                    break;
                }
            }
        }

        if (isOverlayDetected)
        {
            return true;
        }
    }

    return false;
}
