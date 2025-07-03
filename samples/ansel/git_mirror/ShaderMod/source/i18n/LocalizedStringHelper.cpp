#include "LocalizedStringHelper.h"
#include <map>
#include <Windows.h>

namespace i18n
{
    namespace
    {
        std::map<std::pair<int, LANGID>, std::wstring> sResources;

        HMODULE determineHModule()
        {
            MEMORY_BASIC_INFORMATION mbi;
            if (!::VirtualQuery(&sResources, &mbi, sizeof(mbi)))
                return NULL;
            return static_cast<HMODULE>(mbi.AllocationBase);
        }

        HMODULE getHModule()
        {
            static HMODULE sModule = determineHModule();
            return sModule;
        }

        const wchar_t * LoadResourceString(HMODULE hModule, int stringID, WORD langID, size_t * stringLen)
        {
            const wchar_t * stringPtr = nullptr;
            if (stringLen)
                *stringLen = 0;

            const DWORD blockID = (stringID >> 4) + 1;
            const DWORD itemID = stringID % 0x10;

            HRSRC hRes = FindResourceEx(hModule, RT_STRING, MAKEINTRESOURCE(blockID), langID);
            if (hRes)
            {
                HGLOBAL hBlock = LoadResource(hModule, hRes);
                const WCHAR * tableDataBlock = reinterpret_cast<LPCWSTR>(LockResource(hBlock));
                const DWORD tableBlockSize = SizeofResource(hModule, hRes);
                DWORD searchOffset = 0;
                DWORD stringIndex = 0;

                while (searchOffset < tableBlockSize)
                {
                    if (stringIndex == itemID)
                    {
                        if (tableDataBlock[searchOffset] != 0x0000)
                        {
                            stringPtr = tableDataBlock + searchOffset + 1;
                            if (stringLen)
                                *stringLen = tableDataBlock[searchOffset];
                        }
                        else
                        {
                            stringPtr = NULL;
                            if (stringLen)
                                *stringLen = 0;
                        }

                        break;
                    }
                    searchOffset += tableDataBlock[searchOffset] + 1;
                    stringIndex++;
                }
            }

            return stringPtr;
        }
    }

    const std::wstring& getLocalizedString(int resourceId, LANGID langId)
    {
        const auto key = std::make_pair(resourceId, langId);
        // if resource is cached, return it
        if (sResources.find(key) != sResources.cend())
            return sResources[key];
        // otherwise load, cache and return
        size_t stringLen = 0;
        const wchar_t* lpsz = LoadResourceString(getHModule(), resourceId, langId, &stringLen);
        sResources[key] = std::wstring(lpsz, lpsz + stringLen);
        return sResources[key];
    }

    void cleanCache()
    {
        sResources = decltype(sResources)();
    }
}