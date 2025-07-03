#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <future>
#include <urlmon.h>

#include "darkroom\Errors.h"

namespace sideload
{
    enum class SideloadStatus
    {
        kNone,
        kDownloading,
        kInstalling
    };

    class DownloadProgressHandler
    {
    public:
        virtual ~DownloadProgressHandler() {} // base classes that are deleted require a virtual destructor, see https://wiki.sei.cmu.edu/confluence/display/cplusplus/OOP52-CPP.+Do+not+delete+a+polymorphic+object+without+a+virtual+destructor
        virtual uint32_t getProgress() = 0;
    };

    class SideloadHandler
    {
    public:
        uint32_t getDownloadProgress();
        SideloadStatus getSideloadStatus();
        darkroom::Error downloadAndInstall(const std::wstring& url, const std::wstring& path, const std::vector<std::wstring>& args, DWORD installTimeoutMs);
        std::future<darkroom::Error>& getFuture();
    private:
        std::unique_ptr<DownloadProgressHandler> m_downloadProgressHandler;
        SideloadStatus m_status = SideloadStatus::kNone;
        std::future<darkroom::Error> m_future;
    };
}
