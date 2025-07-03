// Copyright (c) 2016, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.

//#define PERF_MEASURE

#pragma once

#include "MessageBusSelwre.h"
#include <windows.h>
#include <string>
#include <list>
#include "lwSelwreLoadLibrary.h"
#ifdef PERF_MEASURE
#include <stdio.h>
#endif

#define MESSAGE_BUS_DEPENDENCY_MAP_FILE_NAME_W L"messagebus-dep.conf"

// This is the path, where message bus DLL is installed. It is loaded from config/registry.
static std::wstring s_basePath;
// This is the module dependency file, which lists all the absolute paths for the MB dependencies
static std::wstring s_moduleMapPath;

namespace lwcsdk
{
    static const int MAX_LINE_LENGTH = 512;

    /// @brief Json format Config Rules
    /// Format:
    /// <"key">:<"value">
    /// Keys are case sensitive
    /// Comment:
    /// Line shall starts with ';'
    /// Key: String
    /// Can't be empty and cant have char ';' and ilwerted commas '"' required for string (Json format)
    /// value: String
    /// Can have space and ilwerted commas '"' required for string (Json format)

    /// @brief Class to get and set key Value Pair, lifetime of returned pointer memory managed by  ConfigParser
    class Pair
    {
    public:
        Pair(const wchar_t* key, const wchar_t* value);
        ~Pair();

        /// @brief Get key name
        /// @return string
        const wchar_t* GetKey() const;

        /// @brief Set key name
        /// @param key [in] the key name
        /// @return bool, indicating success
        bool SetKey(const wchar_t* key);

        /// @brief Get key value
        /// @return string
        const wchar_t* GetValue() const;

        /// @brief Set key value
        /// @param value [in] the key value
        /// @return bool, indicating success
        bool SetValue(const wchar_t* value);

    private:
        const wchar_t* m_key;
        const wchar_t* m_value;
        Pair* m_nextPair;

        friend class ConfigParser;
    };

    /// @brief Class maintains lifetime of returned pointers' memory
    class ConfigParser
    {
    public:
        ConfigParser();
        ~ConfigParser();

        /// @brief Add key value pair
        /// @param key [in] key name
        /// @param value [in] key value
        /// @return bool indicating success
        bool AddPair(const wchar_t* key, const wchar_t* value);

        /// @brief Get key value
        /// @param key [in] the key name
        /// @param section [in] the section name, not supported yet
        /// @return string, the key value.
        const wchar_t* GetValue(const wchar_t* key) const;

        /// @brief Parse Config file
        /// @param fileName [in] The config file path.
        /// @return bool indicating success
        bool Load(const wchar_t* fileName);

        /// @brief Removes all the Pairs added during load
        void UnLoad();

    private:
        size_t m_count;

        Pair* m_head;
        Pair* m_tail;
    };

    // Strip space, comma or " from both end 
    static wchar_t* trim(wchar_t* str)
    {
        size_t l = wcslen(str);
        while (l && (isspace(str[l - 1]) || (str[l - 1] == '\"') || (str[l - 1] == ',')))
            str[--l] = L'\0';
        while (isspace(*str) || (*str == '\"') || (*str == ','))
            ++str;

        return str;
    }

    // Pair class implementation
    Pair::Pair(const wchar_t* key, const wchar_t* value)
        : m_key(nullptr), m_value(nullptr), m_nextPair(nullptr)
    {
        SetKey(key);
        SetValue(value);
    }

    Pair::~Pair()
    {
        delete[] m_value;
        delete[] m_key;
    }

    const wchar_t* Pair::GetKey() const { return m_key; }
    const wchar_t* Pair::GetValue() const { return m_value; }

    bool Pair::SetKey(const wchar_t* val)
    {
        if (val && *val != L'\0' && !wcschr(val, L';'))
        {
            size_t destsize = wcslen(val) + 1;
            wchar_t* dest = new (std::nothrow) wchar_t[destsize]();
            if (dest)
            {
                if (wcsncpy_s(dest, destsize, val, _TRUNCATE))
                {
                    delete[] dest;
                    return false;
                }
                delete[] m_key;
                m_key = dest;
                return true;
            }
        }
        return false;
    }

    bool Pair::SetValue(const wchar_t* val)
    {
        if (val)
        {
            size_t destsize = wcslen(val) + 1;
            wchar_t* dest = new (std::nothrow) wchar_t[destsize]();
            if (dest)
            {
                if (wcsncpy_s(dest, destsize, val, _TRUNCATE))
                {
                    delete[] dest;
                    return false;
                }
                delete[] m_value;
                m_value = dest;
                return true;
            }
        }
        return false;
    }

    // ConfigParser class implementation
    ConfigParser::ConfigParser()
        : m_count(0)
        , m_head(nullptr)
        , m_tail(nullptr)
    {
    }

    ConfigParser::~ConfigParser()
    {
        UnLoad();
    }

    void ConfigParser::UnLoad()
    {
        Pair* itr = m_head;
        Pair* prev = itr;
        while (itr)
        {
            prev = itr;
            itr = itr->m_nextPair;
            delete prev;
        }
        m_head = NULL;
        m_tail = NULL;
        m_count = 0;
    }

    bool ConfigParser::AddPair(const wchar_t* key, const wchar_t* value)
    {
        Pair* pairPtr = new (std::nothrow) Pair(key, value);
        if (pairPtr && pairPtr->GetKey() && pairPtr->GetValue())
        {
            if (!m_head)
            {
                m_head = m_tail = pairPtr;
            }
            else
            {
                m_tail->m_nextPair = pairPtr;
                m_tail = pairPtr;
            }
            ++m_count;
            return true;
        }
        else if (pairPtr)
        {
            delete pairPtr;
        }

        return false;
    }

    const wchar_t* ConfigParser::GetValue(const wchar_t* key) const
    {
        Pair* itr = m_head;
        while (itr && wcscmp(itr->GetKey(), key) != 0)
        {
            itr = itr->m_nextPair;
        }

        return itr ? itr->GetValue() : L"";
    }

    bool ConfigParser::Load(const wchar_t* fileName)
    {
        FILE* fptr = _wfsopen(fileName, L"rt", _SH_DENYWR);
        if (fptr == NULL)
        {
            return false;
        }

        wchar_t line[MAX_LINE_LENGTH];
        bool status = true;
        while (fgetws(line, sizeof(line), fptr) != NULL)
        {
            // if ;, {, } or blank line, ignore and continue to next line
            if (*line == L';' || *line == L'{' || *line == L'}' || *line == L'\r' || *line == L'\n')
            {
                continue;
            }
            // ensure each line is well formed. Check for the presence of atleast "key:"
            if (*line == L':' || !wcschr(line, L':'))
            {
                status = false;
                break;
            }

            // Get key. Cannot be NULL or empty
            wchar_t* buffer;
            wchar_t* token = wcstok_s(line, L":", &buffer);
            if (!token)
            {
                status = false;
                break;
            }
            wchar_t* tkey = trim(token);
            if (*tkey == L'\0' || wcschr(line, L';'))
            {
                status = false;
                break;
            }

            // Get Value (can be NULL/empty)
            token = wcstok_s(NULL, L"\r\n", &buffer);
            wchar_t* tvalue = NULL;
            if (token)
            {
                tvalue = trim(token);
                status = AddPair(tkey, tvalue);
            }
            else
            {
                // if no value present, add empty string
                status = AddPair(tkey, L"");
            }

            if (!status)
            {
                break;
            }
        }
        status = ferror(fptr) ? false : status;
        fclose(fptr);

        return status;
    }
}

static lwcsdk::ConfigParser s_moduleMap;

class DllContext
{
public:
    DllContext() : m_module(NULL) {}
    ~DllContext() { Unload(); }
    DllContext(const DllContext& from) : m_module(NULL) { Copy(from); }
    DllContext& operator=(const DllContext& from);

    bool Load(const std::wstring& name, bool useModuleMap);
    void Unload();
    bool IsLoaded() const { return m_module != NULL; }
    FARPROC GetProcAddress(const char* apiName) const;

private:
    void Copy(const DllContext& from);

    HMODULE m_module;
    std::wstring m_name;
};

DllContext& DllContext::operator=(const DllContext& from)
{
    if (&from != this)
    {
         Copy(from);
    }
    return *this;
}

void DllContext::Copy(const DllContext& from)
{
    if (from.m_name.empty())
    {
        return;
    }

    // Get a handle to the DLL in question.
    m_module = LoadLibraryW(from.m_name.c_str());
    if (m_module != NULL)
    {
        m_name = from.m_name;
    }
}

bool DllContext::Load(const std::wstring& name, bool useModuleMap)
{
    std::wstring fullPath;
    if (name != L"MessageBus.dll" && useModuleMap)
    {
        fullPath.assign(s_moduleMap.GetValue(name.c_str()));
    }
    else
    {
        // Load library selwrely combining base path with the file name.
        fullPath = s_basePath + L"\\" + name;
    }

#ifdef PERF_MEASURE
    // For testing purposes preload the library to make sure that it's been cached by the file system,
    // so that we profile only lwLoadLibrary itself.
    ULONGLONG time1 = GetTickCount64();
    HMODULE hModule = LoadLibraryW(fullPath.c_str());
    ULONGLONG time2 = GetTickCount64();
    FreeLibrary(hModule);

    ULONGLONG time3 = GetTickCount64();
    hModule = LoadLibraryW(fullPath.c_str());
    ULONGLONG time4 = GetTickCount64();
    FreeLibrary(hModule);
#endif

    ULONGLONG time5 = GetTickCount64();
    m_module = lwLoadSignedLibraryW(fullPath.c_str(), TRUE);
    ULONGLONG time6 = GetTickCount64();
    UNREFERENCED_PARAMETER(time5);
    UNREFERENCED_PARAMETER(time6);

#ifdef PERF_MEASURE
    printf(
        "%S %u - %u - %u\n",
        m_name.c_str(),
        (unsigned)(time2-time1),
        (unsigned)(time4-time3),
        (unsigned)(time6-time5));
#endif

    if (m_module != NULL)
    {
        m_name = fullPath;
    }
    return m_module != NULL;
}

void DllContext::Unload()
{
    if (m_module != NULL)
    {
        FreeLibrary(m_module);
        m_module = NULL;
    }
    m_name.clear();
}

FARPROC DllContext::GetProcAddress(const char* apiName) const
{
    if (!IsLoaded())
    {
        return NULL;
    }
    return ::GetProcAddress(m_module, apiName);
}


// Global variable for one-time initialization structure.
static INIT_ONCE s_InitOnce = INIT_ONCE_STATIC_INIT;

static std::list<DllContext> s_dependencies;

// MessageBus.dll is the last in the list of dependencies.
inline DllContext& MessageBusDll()
{
    return s_dependencies.back();
}

typedef MessageBusInterface* (*GET_BUS_INTERFACE_PROC)(uint32_t version);
typedef MessageBusInterface* (*GET_BUS_INTERFACE_WITH_CONFIG_PROC)(uint32_t version, const wchar_t* configFilePath);
typedef void (*RELEASE_BUS_INTERFACE_PROC)(MessageBusInterface* instance);
typedef bool(*GET_BUS_UNIQUEID_PROC)(MessageBusUniqueId* uniqueId);

static BOOL CALLBACK LoadMessageBusDllCallback(PINIT_ONCE, PVOID, PVOID*)
{
    // if the basepath is not already set, read from registry
    if (s_basePath.empty())
    {
        // First, load the base path from registry.
        HKEY busKey;
        LONG result = RegCreateKeyExW(
            HKEY_LOCAL_MACHINE,
            { L"SOFTWARE\\LWPU Corporation\\LwContainer\\MessageBus" },
            0,
            NULL,
            0,
            KEY_READ,
            NULL,
            &busKey,
            NULL);
        if (result != ERROR_SUCCESS)
        {
            return FALSE;
        }

        wchar_t basePath[MAX_PATH] = { L"" };
        DWORD size = (DWORD)sizeof(basePath);
        result = RegGetValueW(
            busKey,
            NULL,
            L"InstallPath",
            RRF_RT_REG_SZ,
            NULL,
            (LPBYTE)basePath,
            &size);
        if (result == ERROR_SUCCESS)
        {
            s_basePath = basePath;
        }

        RegCloseKey(busKey);
    }

    // We only allow full path names, so make sure that the base path is not empty.
    if (s_basePath.empty())
    {
        return FALSE;
    }

    // if the module map is present, load it
    if (!s_moduleMapPath.empty())
    {
        if (!s_moduleMap.Load(s_moduleMapPath.c_str()))
        {
            // If the module map cannot be parsed, fail
            return FALSE;
        }
    }

    // Extract dependency list from MessageBus.DLL's embedded resource data.
    // Suppress system popups for not being able to load dynamic library.
    UINT oldErrorMode = GetErrorMode();
    SetErrorMode(oldErrorMode | SEM_FAILCRITICALERRORS);

    int resourceDataSize = 0;
    wchar_t resourceData[1024] = { 0 };
    std::wstring messageBusBinaryFullPath = s_basePath + L"\\" + MESSAGE_BUS_BINARY_NAME_W;
    HMODULE hData = LoadLibraryExW(messageBusBinaryFullPath.c_str(), NULL, LOAD_LIBRARY_AS_DATAFILE);
    if (hData != NULL)
    {
        resourceDataSize = LoadStringW(hData, IDS_MESSAGE_BUS_DEPENDENCIES, resourceData, _countof(resourceData));
        FreeLibrary(hData);
    }

    // Reset to previous error mode.
    SetErrorMode(oldErrorMode);
    if (hData == NULL || resourceDataSize == 0)
    {
        return FALSE;
    }

    // Then, load all dependent DLLs.
    const std::wstring dependencies(resourceData);
    const std::wstring delimiter(L",");
    size_t start = 0, end = 0;
    while (end != std::string::npos)
    {
        end = dependencies.find(delimiter, start);
        DllContext dll;
        if (!dll.Load(dependencies.substr(start, (end == std::string::npos) ? std::string::npos : end - start), !s_moduleMapPath.empty()))
        {
            // Reset the list of dependencies in case of failure.
            s_dependencies.clear();
            s_moduleMap.UnLoad();
            return FALSE;
        }
        s_dependencies.push_back(dll);

        if (end != std::string::npos)
        {
            start = end+1;
        }
    }
    s_moduleMap.UnLoad();
    return TRUE;
}

static bool LoadMessageBusDll()
{
    // Execute the initialization callback function 
    return InitOnceExelwteOnce(&s_InitOnce, LoadMessageBusDllCallback, NULL, NULL) ? true : false;
}

// Read the InstallPath and ModuleMapPath from the input config file
static bool ProcessConfigFile(const wchar_t* configFilePath)
{
    lwcsdk::ConfigParser parser;
    if (parser.Load(configFilePath))
    {
        s_basePath.assign(parser.GetValue(L"InstallPath"));
        s_moduleMapPath = s_basePath + L"\\" + MESSAGE_BUS_DEPENDENCY_MAP_FILE_NAME_W;

        // InstallPath is expected to be present in the config file
        return (s_basePath.size() > 0);
    }
    return false;
}

MessageBusInterface* getMessageBusInterfaceSelwre(uint32_t version)
{
    // Load the message bus DLL if it has not been loaded yet.
    if (!LoadMessageBusDll())
    {
        return NULL;
    }

    GET_BUS_INTERFACE_PROC getBusInterfaceProc;
    getBusInterfaceProc = (GET_BUS_INTERFACE_PROC)MessageBusDll().GetProcAddress("getMessageBusInterface");
    if (getBusInterfaceProc == NULL)
    {
        return NULL;
    }

    return (*getBusInterfaceProc)(version);
}

MessageBusInterface* getSharedMessageBusInterfaceSelwre(uint32_t version)
{
    // Load the message bus DLL if it has not been loaded yet.
    if (!LoadMessageBusDll())
    {
        return NULL;
    }

    GET_BUS_INTERFACE_PROC getSharedBusInterfaceProc;
    getSharedBusInterfaceProc = (GET_BUS_INTERFACE_PROC)MessageBusDll().GetProcAddress("getSharedMessageBusInterface");
    if (getSharedBusInterfaceProc == NULL)
    {
        return NULL;
    }

    return (*getSharedBusInterfaceProc)(version);
}

MessageBusInterface* getMessageBusInterfaceWithConfigSelwre(uint32_t version, const wchar_t* configFilePath)
{
    // if config file is specified, process it
    if (configFilePath && !ProcessConfigFile(configFilePath))
    {
        return NULL;
    }

    // Load the message bus DLL if it has not been loaded yet.
    if (!LoadMessageBusDll())
    {
        return NULL;
    }

    GET_BUS_INTERFACE_WITH_CONFIG_PROC getBusInterfaceProc;
    getBusInterfaceProc = (GET_BUS_INTERFACE_WITH_CONFIG_PROC)MessageBusDll().GetProcAddress("getMessageBusInterfaceWithConfig");
    if (getBusInterfaceProc == NULL)
    {
        return NULL;
    }

    return (*getBusInterfaceProc)(version, configFilePath);
}

MessageBusInterface* getSharedMessageBusInterfaceWithConfigSelwre(uint32_t version, const wchar_t* configFilePath)
{
    // if config file is specified, process it
    if (configFilePath && !ProcessConfigFile(configFilePath))
    {
        return NULL;
    }

    // Load the message bus DLL if it has not been loaded yet.
    if (!LoadMessageBusDll())
    {
        return NULL;
    }

    GET_BUS_INTERFACE_WITH_CONFIG_PROC getSharedBusInterfaceProc;
    getSharedBusInterfaceProc = (GET_BUS_INTERFACE_WITH_CONFIG_PROC)MessageBusDll().GetProcAddress("getSharedMessageBusInterfaceWithConfig");
    if (getSharedBusInterfaceProc == NULL)
    {
        return NULL;
    }

    return (*getSharedBusInterfaceProc)(version, configFilePath);
}

void releaseMessageBusInterfaceSelwre(MessageBusInterface* instance)
{
    // By this time message bus DLL must have been loaded.
    RELEASE_BUS_INTERFACE_PROC releaseBusInterfaceProc;
    releaseBusInterfaceProc = (RELEASE_BUS_INTERFACE_PROC)MessageBusDll().GetProcAddress("releaseMessageBusInterface");
    if (releaseBusInterfaceProc != NULL)
    {
        (*releaseBusInterfaceProc)(instance);
    }
}

bool generateMessageBusUniqueIdSelwre(MessageBusUniqueId* uniqueId)
{
    // Load the message bus DLL if it has not been loaded yet.
    if (!LoadMessageBusDll())
    {
        return false;
    }

    GET_BUS_UNIQUEID_PROC getBusUniqueIdProc;
    getBusUniqueIdProc = (GET_BUS_UNIQUEID_PROC)MessageBusDll().GetProcAddress("generateMessageBusUniqueId");
    if (getBusUniqueIdProc == NULL)
    {
        return false;
    }

    return (*getBusUniqueIdProc)(uniqueId);
}