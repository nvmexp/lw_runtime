#include "sideload\SideloadHandler.h"
#include "darkroom\JobProcessing.h"
#include <urlmon.h>
#include <WinInet.h>
#include <future>

#pragma comment(lib, "Urlmon.lib")
#pragma comment(lib, "Wininet.lib")

namespace
{
    class DownloadStatus : public IBindStatusCallback, public sideload::DownloadProgressHandler
    {
    public:
        STDMETHOD(OnStartBinding)(DWORD dwReserved, IBinding __RPC_FAR *pib) { return E_NOTIMPL; }
        STDMETHOD(GetPriority)(LONG __RPC_FAR *pnPriority) { return E_NOTIMPL; }
        STDMETHOD(OnLowResource)(DWORD reserved) { return E_NOTIMPL; }
        STDMETHOD(OnStopBinding)(HRESULT hresult, LPCWSTR szError) { return E_NOTIMPL; }
        STDMETHOD(OnDataAvailable)(DWORD grfBSCF, DWORD dwSize, FORMATETC __RPC_FAR *pformatetc, STGMEDIUM __RPC_FAR *pstgmed) { return E_NOTIMPL; }
        STDMETHOD(OnObjectAvailable)(REFIID riid, IUnknown __RPC_FAR *punk) { return E_NOTIMPL; }
        STDMETHOD_(ULONG, AddRef)() { return 0; }
        STDMETHOD_(ULONG, Release)() { return 0; }
        STDMETHOD(QueryInterface)(REFIID riid, void __RPC_FAR *__RPC_FAR *ppvObject) { return E_NOTIMPL; }
        STDMETHOD(GetBindInfo)(DWORD __RPC_FAR *grfBINDF, BINDINFO __RPC_FAR *pbindinfo) { return E_NOTIMPL; }

        STDMETHOD(OnProgress)(ULONG ulProgress, ULONG ulProgressMax, ULONG ulStatusCode, LPCWSTR wszStatusText)
        {
            if (!(ulStatusCode == BINDSTATUS_DOWNLOADINGDATA && ulProgressMax > 0u))
                return S_OK;

            m_progressInPercents = uint32_t(100.0f * float(ulProgress) / float(ulProgressMax));
            return S_OK;
        }
        uint32_t getProgress() { return m_progressInPercents; }
    private:
        uint32_t m_progressInPercents = 0u;
    };
}

namespace sideload
{
    uint32_t SideloadHandler::getDownloadProgress()
    { 
        if (m_downloadProgressHandler)
            return m_downloadProgressHandler->getProgress(); 
        return 0u;
    }
    SideloadStatus SideloadHandler::getSideloadStatus() { return m_status; }
    std::future<darkroom::Error>& SideloadHandler::getFuture() { return m_future; }

    darkroom::Error SideloadHandler::downloadAndInstall(const std::wstring& url, const std::wstring& path, const std::vector<std::wstring>& args, DWORD installTimeoutMs)
    {
        const auto work = [&](const std::wstring url, const std::wstring path, const std::vector<std::wstring> args, DWORD installTimeoutMs)
        {
            // we're using /EHa, so SEH exceptions will be catched inside a catch block
            try
            {
                m_downloadProgressHandler = std::make_unique<DownloadStatus>();
                DeleteUrlCacheEntry(url.c_str());
                // download the artifact
                m_status = SideloadStatus::kDownloading;
                HRESULT hr = URLDownloadToFile(NULL, url.c_str(), path.c_str(), 0, static_cast<IBindStatusCallback*>(static_cast<DownloadStatus*>(m_downloadProgressHandler.get())));
                if (hr != S_OK)
                {
                    // delete partially downloaded file if there is one
                    DeleteFile(path.c_str());
                    m_status = SideloadStatus::kNone;
                    return darkroom::Error::kDownloadFailed;
                }
                // then install it
                m_status = SideloadStatus::kInstalling;
                HANDLE handle = NULL;
                bool exelwted = darkroom::exelwteProcess(path.c_str(), { args }, L".", handle);

                if (!exelwted)
                {
                    m_status = SideloadStatus::kNone;
                    return darkroom::Error::kCouldntStartupTheProcess;
                }

                const DWORD wfsoResult = WaitForSingleObject(handle, installTimeoutMs);

                if (!(wfsoResult == WAIT_ABANDONED || wfsoResult == WAIT_OBJECT_0))
                {
                    CloseHandle(handle);
                    DeleteFile(path.c_str());
                    m_status = SideloadStatus::kNone;
                    return darkroom::Error::kInstallFailed;
                }

                DWORD exitCode = 0u;
                if (FALSE == GetExitCodeProcess(handle, &exitCode))
                {
                    CloseHandle(handle);
                    DeleteFile(path.c_str());
                    m_status = SideloadStatus::kNone;
                    return darkroom::Error::kInstallFailed;
                }

                if (exitCode == STILL_ACTIVE)
                {
                    TerminateProcess(handle, -1);
                    CloseHandle(handle);
                    DeleteFile(path.c_str());
                    m_status = SideloadStatus::kNone;
                    return darkroom::Error::kOperationTimeout;
                }

                if (exitCode != 0)
                {
                    CloseHandle(handle);
                    DeleteFile(path.c_str());
                    m_status = SideloadStatus::kNone;
                    return darkroom::Error::kInstallFailed;
                }

                CloseHandle(handle);
                DeleteFile(path.c_str());
                m_status = SideloadStatus::kNone;
                return darkroom::Error::kSuccess;
            }
            catch (...)
            {
                return darkroom::Error::kOperationFailed;
            }
        };

        m_future = std::async(work, url, path, args, installTimeoutMs);
        return darkroom::Error::kSuccess;
    }
}
