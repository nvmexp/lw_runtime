#pragma once

#include <Windows.h>
#include <string>

namespace i18n
{
    // These functions are not reentrant
    // In general case string references returned by this function
    // are only valid until the next call to getLocalizedString, 
    // so use it immediately or copy it
    const std::wstring& getLocalizedString(int resourceId, LANGID langId);
    void cleanCache();
}