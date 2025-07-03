#define ANSEL_DLL_EXPORTS
#include <windows.h>
#include <shlobj.h>
#include <lwapi.h>
#include <LwApiDriverSettings.h>

#include "Config.h"
#include "drs/LwDrsWrapper.h"
#include "AnselServer.h"
#include "Ansel.h"
#include "AnselBuildNumber.h"
#include "darkroom/StringColwersion.h"

#include "RegistrySettings.h"
#include "Log.h"
#include "Utils.h"

#if IPC_ENABLED == 1
#include "google\protobuf\stubs\common.h"
#endif

#if DEBUG_LEAKS == 1
#include <vld.h>
#endif

// Make compiler give error if there are unused function parameters
#pragma warning(error : 4100)

static HINSTANCE s_hDLLInstance = 0;

__declspec(thread) bool tls_bIsCreatingAnselServer = false;

static HANSELCLIENT s_hMainClient = 0;

static std::vector<AnselServer*> s_servers;

//****************************************************************************
// C interface functions
//****************************************************************************

ANSEL_DLL_API DWORD __cdecl AnselGetFunctionTableSize(void)
{
    return sizeof(AnselFunctionTable);
}

HANSELSERVER AnselCreateServerOnAdapter(HANSELCLIENT hClient, ClientFunctionTable * pFunctionTable, LARGE_INTEGER AdapterLuid)
{
    if (tls_bIsCreatingAnselServer ||
        ((s_hMainClient != 0) && (s_hMainClient != hClient)))
    {
        return NULL;
    }
    tls_bIsCreatingAnselServer = true;

    AnselServer * pAnselServer = new AnselServer(s_hDLLInstance);
    HRESULT status = S_OK;
    if (!SUCCEEDED(status = pAnselServer->init(hClient, pFunctionTable, AdapterLuid)))
    {
        tls_bIsCreatingAnselServer = false;
        delete pAnselServer;
        return NULL;
    }
    tls_bIsCreatingAnselServer = false;
    s_servers.push_back(pAnselServer);
    return pAnselServer;
}

HANSELSERVER __cdecl AnselCreateServer(HANSELCLIENT hClient, ClientFunctionTable * pFunctionTable)
{
    return AnselCreateServerOnAdapter(hClient, pFunctionTable, { 0 });
}

void __cdecl AnselReleaseServer(HANSELSERVER hAnselServer)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return;
    }

    if (pAnselServer->isInitialized())
    {
        pAnselServer->destroy();
    }
    s_servers.erase(std::remove(s_servers.begin(), s_servers.end(), pAnselServer), s_servers.end());
    delete pAnselServer;
}

HRESULT AnselRestrictServerCreation(HANSELCLIENT hClient, bool restrictCreation)
{
    if (restrictCreation)
    {
#if _M_AMD64
        InterlockedCompareExchange64((LONGLONG*)&s_hMainClient, (LONGLONG)hClient, 0);
#else
        InterlockedCompareExchange((LONG*)&s_hMainClient, (LONG)hClient, 0);
#endif
        return (s_hMainClient == hClient) ? S_OK : E_FAIL;
    }
    else
    {
#if _M_AMD64
        InterlockedCompareExchange64((LONGLONG*)&s_hMainClient, 0, (LONGLONG)hClient);
#else
        InterlockedCompareExchange((LONG*)&s_hMainClient, 0, (LONG)hClient);
#endif
        return (s_hMainClient == 0) ? S_OK : E_FAIL;
    }
    return S_OK;
}

HRESULT AnselExelwtePostProcessing(HANSELSERVER hAnselServer, HCLIENTRESOURCE hPresentResource, DWORD subResIndex)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->exelwtePostProcessing(hPresentResource, subResIndex);
}

HRESULT __cdecl AnselCreateSharedResource(HANSELSERVER hAnselServer,
    DWORD width,
    DWORD height,
    DWORD sampleCount,
    DWORD sampleQuality,
    DWORD format,
    HANDLE * pHandle,
    void * pServerPrivateData)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->createSharedResource(width,
        height,
        sampleCount,
        sampleQuality,
        format,
        pHandle,
        pServerPrivateData);
}

HRESULT AnselNotifyDepthStencilCreate(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyDepthStencilCreate(hClientResource);
}

HRESULT AnselNotifyDepthStencilBind(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyDepthStencilBind(hClientResource);
}

HRESULT AnselNotifyDepthStencilBindWithFormat(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource, DWORD format, ANSEL_VIEW_DIMENSION viewDimension, ANSEL_DEPTH_STENCIL_VIEW_FLAGS viewFlags)
{
    // TODO : make use the format that the shim passed along, can be useful if the resource itself was created as a TYPELESS surface
    format; viewDimension; viewFlags;
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyDepthStencilBind(hClientResource);
}

HRESULT AnselNotifyRenderTargetBind(HANSELSERVER hAnselServer, HCLIENTRESOURCE* phClientResource, DWORD dwNumRTs)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyRenderTargetBind(phClientResource, dwNumRTs);
}

HRESULT AnselNotifyRenderTargetBindWithFormat(HANSELSERVER hAnselServer, HCLIENTRESOURCE* phClientResource, DWORD dwNumRTs, const DWORD * /*pFormats*/, const ANSEL_VIEW_DIMENSION * /*pViewDimensions*/)
{
    // TODO : make use the format that the shim passed along, can be useful if the resource itself was created as a TYPELESS surface
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyRenderTargetBind(phClientResource, dwNumRTs);
}

HRESULT AnselNotifyUnorderedAccessBind(HANSELSERVER hAnselServer, DWORD startOffset, DWORD count, HCLIENTRESOURCE* phClientResource)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyUnorderedAccessBind(startOffset, count, phClientResource);
}

HRESULT AnselNotifyDepthStencilDestroy(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyDepthStencilDestroy(hClientResource);
}

HRESULT AnselNotifyClientResourceDestroy(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyClientResourceDestroy(hClientResource);
}

HRESULT AnselNotifyDepthStencilClear(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyDepthStencilClear(hClientResource);
}

HRESULT AnselNotifyRenderTargetClear(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyRenderTargetClear(hClientResource);
}

HRESULT AnselNotifyDraw(HANSELSERVER hAnselServer)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyDraw();
}

HRESULT AnselNotifyHotkey(HANSELSERVER hAnselServer, DWORD vkey)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyHotkey(vkey);
}

HRESULT AnselSetConfig(HANSELSERVER hAnselServer, AnselConfig *pConfig)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->setConfig(pConfig);
}

HRESULT AnselUpdateGPUMask(HANSELSERVER hAnselServer, DWORD activeGPUMask)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->updateGPUMask(activeGPUMask);
}

ANSEL_DLL_API void __cdecl AnselGetVersion(DWORD * pDwMajor, DWORD * pDwMinor)
{
    *pDwMajor = ANSEL_VERSION_MAJOR;
    *pDwMinor = ANSEL_VERSION_MINOR;
}

HRESULT AnselNotifyCmdListCreate12(HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST *phAnselCmdList)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyCmdListCreate12(hCmdList, phAnselCmdList);
}

HRESULT AnselNotifyCmdListDestroy12(HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyCmdListDestroy12(hCmdList, hAnselCmdList);
}

HRESULT AnselNotifyCmdListReset12(HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyCmdListReset12(hCmdList, hAnselCmdList);
}

HRESULT AnselNotifyCmdListClose12(HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyCmdListClose12(hCmdList, hAnselCmdList);
}

HRESULT AnselNotifySetRenderTargetBake12(HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, void ** ppServerData)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifySetRenderTargetBake12(hCmdList, hAnselCmdList, ppServerData);
}

HRESULT AnselNotifySetRenderTargetBakeWithDeviceStates12(HANSELSERVER hAnselServer, const AnselDeviceStates deviceStates, HANSELCMDLIST hAnselCmdList, void ** ppServerData)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifySetRenderTargetBakeWithDeviceStates12(deviceStates, hAnselCmdList, ppServerData);
}

HRESULT AnselNotifySetRenderTargetExec12(HANSELSERVER hAnselServer, ANSEL_EXEC_DATA *pAnselExecData)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifySetRenderTargetExec12(pAnselExecData);
}

HRESULT AnselNotifyPresentBake12(HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, void ** ppServerData)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyPresentBake12(hCmdList, hAnselCmdList, ppServerData);
}

HRESULT AnselExelwtePostProcessing12(HANSELSERVER hAnselServer, ANSEL_EXEC_DATA *pAnselExecData, HCLIENTRESOURCE hPresentResource, DWORD subResIndex)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->exelwtePostProcessing12(pAnselExecData, hPresentResource, subResIndex);
}

HRESULT AnselNotifyDepthStencilClearBake12(HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, HCLIENTRESOURCE hDepthStencil, void ** pServerData)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyDepthStencilClearBake12(hCmdList, hAnselCmdList, hDepthStencil, pServerData);
}

HRESULT AnselNotifyRenderTargetClearBake12(HANSELSERVER /*hAnselServer*/, HCMDLIST /*hCmdList*/, HANSELCMDLIST /*hAnselCmdList*/, HCLIENTRESOURCE /*hDepthStencil*/, void ** /*pServerData*/)
{
    // TODO : do something with this notification
    return S_OK;
}

HRESULT AnselNotifyDepthStencilClearExec12(HANSELSERVER hAnselServer, ANSEL_EXEC_DATA *pExelwtionContext)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyDepthStencilClearExec12(pExelwtionContext);
}

HRESULT AnselNotifyRenderTargetClearExec12(HANSELSERVER /*hAnselServer*/, ANSEL_EXEC_DATA * /*pExelwtionContext*/)
{
    // TODO : do something with this notification
    return S_OK;
}

HRESULT AnselNotifyClientMode(HANSELSERVER hAnselServer, DWORD clientMode)
{
    AnselServer * pAnselServer = static_cast<AnselServer *>(hAnselServer);
    if (pAnselServer == nullptr)
    {
        return E_FAIL;
    }

    return pAnselServer->notifyClientMode(clientMode);
}

ANSEL_DLL_API HRESULT __cdecl AnselGetFunctionTable(void * pMem)
{
    AnselFunctionTable * pTable = static_cast<AnselFunctionTable *>(pMem);
    if (pTable == nullptr)
    {
        return E_FAIL;
    }

    pTable->CreateServer = AnselCreateServer;
    pTable->ReleaseServer = AnselReleaseServer;
    pTable->ExelwtePostProcessing = AnselExelwtePostProcessing;
    pTable->CreateSharedResource = AnselCreateSharedResource;
    pTable->NotifyDraw = AnselNotifyDraw;
    pTable->NotifyDepthStencilCreate = AnselNotifyDepthStencilCreate;
    pTable->NotifyDepthStencilBind = AnselNotifyDepthStencilBind;
    pTable->NotifyRenderTargetBind = AnselNotifyRenderTargetBind;
    pTable->NotifyUnorderedAccessBind = AnselNotifyUnorderedAccessBind;
    pTable->NotifyDepthStencilDestroy = AnselNotifyDepthStencilDestroy;
    pTable->NotifyClientResourceDestroy = AnselNotifyClientResourceDestroy;
    pTable->NotifyDepthStencilClear = AnselNotifyDepthStencilClear;
    pTable->NotifyHotkey = AnselNotifyHotkey;
    pTable->SetConfig = AnselSetConfig;
    pTable->UpdateGPUMask = AnselUpdateGPUMask;

    pTable->NotifyCmdListCreate12 = AnselNotifyCmdListCreate12;
    pTable->NotifyCmdListDestroy12 = AnselNotifyCmdListDestroy12;
    pTable->NotifyCmdListReset12 = AnselNotifyCmdListReset12;
    pTable->NotifyCmdListClose12 = AnselNotifyCmdListClose12;
    pTable->NotifySetRenderTargetBake12 = AnselNotifySetRenderTargetBake12;
    pTable->NotifySetRenderTargetBakeWithDeviceStates12 = AnselNotifySetRenderTargetBakeWithDeviceStates12;
    pTable->NotifySetRenderTargetExec12 = AnselNotifySetRenderTargetExec12;
    pTable->NotifyPresentBake12 = AnselNotifyPresentBake12;
    pTable->ExelwtePostProcessing12 = AnselExelwtePostProcessing12;
    pTable->NotifyDepthStencilClearBake12 = AnselNotifyDepthStencilClearBake12;
    pTable->NotifyDepthStencilClearExec12 = AnselNotifyDepthStencilClearExec12;

    pTable->NotifyClientMode = AnselNotifyClientMode;

    pTable->NotifyDepthStencilBindWithFormat = AnselNotifyDepthStencilBindWithFormat;
    pTable->NotifyRenderTargetBindWithFormat = AnselNotifyRenderTargetBindWithFormat;
    pTable->NotifyRenderTargetClear = AnselNotifyRenderTargetClear;
    pTable->NotifyRenderTargetClearBake12 = AnselNotifyRenderTargetClearBake12;
    pTable->NotifyRenderTargetClearExec12 = AnselNotifyRenderTargetClearExec12;

    pTable->CreateServerOnAdapter = AnselCreateServerOnAdapter;
    pTable->RestrictServerCreation = AnselRestrictServerCreation;

    return S_OK;
}

ANSEL_DLL_API bool __cdecl AnselEnableCheck()
{
    // After the introduction of specific allowlisting DLL, this shouldn't really be triggered
    // If it is, then something wrong happened
    return false;
}


BOOL APIENTRY DllMain(HINSTANCE hModule, DWORD ul_reason_for_call, LPVOID /*lpReserved*/)
{
    s_hDLLInstance = hModule;

    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        {
#if DEBUG_LEAKS == 1
            VLDSetOptions(VLD_OPT_AGGREGATE_DUPLICATES, 32, 64);
#endif
            initLog();
            RegistrySettings registry;
            int32_t severity = registry.getValue(registry.registryPathAnsel(), L"LogFiltering",
                (int32_t)LogSeverity::kFirst, (int32_t)LogSeverity::kLast,
                (int32_t)LogSeverity::kDisableLogging);
            std::wstring channelsSeverityConfig = registry.getValue(registry.registryPathAnsel(),
                L"LogChannelsFiltering", L"");

            setLogSeverity((LogSeverity)severity, channelsSeverityConfig.c_str());

            // Getting the app name from the module name (EXE)
            std::wstring appName = lwanselutils::getAppNameFromProcess();

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

            if (!logDir.empty())
            {
                // make sure directory exists:
                bool ok = lwanselutils::CreateDirectoryRelwrsively(logDir.c_str());
                if (ok)
                {
                    std::wstring filename = logDir;
                    filename += L"ansel_";
                    filename += appName;
                    filename += L"-";
                    filename = lwanselutils::appendTimeW(filename.c_str(), L".log");
                    setLogFilename(filename.c_str());
                    LOG_INFO("Log file opened");
                }
            }

            LOG_INFO("--LwCamera Log--", severity);
            LOG_INFO("Log level: %s", getLogSeverityName((LogSeverity)severity));
            LOG_INFO("Loaded App: \"%ls.exe\"", appName.c_str());
            LOG_INFO("Log channel severity configuration = %s",
                darkroom::getUtf8FromWstr(channelsSeverityConfig).c_str());
            LOG_INFO("Ansel version %d.%d.%d.%08x", ANSEL_VERSION_MAJOR, ANSEL_VERSION_MINOR,
                ANSEL_BUILD_NUMBER, ANSEL_COMMIT_HASH);
        }
        break;

    case DLL_THREAD_ATTACH:
        break;

    case DLL_THREAD_DETACH:
        break;

    case DLL_PROCESS_DETACH:
    {
        LOG_INFO("Log ending due to DLL process detach");
        // disconnect each server from MessageBus
        // before shutting down protobuf library

        if (false)
        {
            // it's not safe to do that actually, so disabling it for now
            for (auto& server : s_servers)
                if (server)
                {
                    //server->disconnectIpc();
                    if (server->isInitialized())
                    {
                        server->destroy();
                    }
                    delete server;
                }
        }

#if IPC_ENABLED == 1
        google::protobuf::ShutdownProtobufLibrary();
#endif

        LOG_DEBUG("Server release successful. Deinitializing log.");
        deinitLog();
    }
        break;
    }
    return TRUE;
}

#if DEBUG_LEAKS == 1
struct VLDCleanup
{
    ~VLDCleanup()
    {
        VLDReportLeaks();
    }
};

#pragma warning(disable : 4074) 
#pragma init_seg(compiler)
VLDCleanup gCleanup;
#endif
