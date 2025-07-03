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


// This is the path, where message bus DLL and all of its dependencies are installed.
// It is loaded from registry.
static std::wstring s_basePath;


class DllContext
{
public:
    DllContext() : m_module(NULL) {}
    ~DllContext() { Unload(); }
    DllContext(const DllContext& from) : m_module(NULL) { Copy(from); }
    DllContext& operator=(const DllContext& from);

    bool Load(const std::wstring& name);
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

bool DllContext::Load(const std::wstring& name)
{
    // Load library selwrely combining base path with the file name.
    std::wstring fullPath = s_basePath + L"\\" + name;

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
    m_module = lwLoadLibraryW(fullPath.c_str(), TRUE);
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
typedef void (*RELEASE_BUS_INTERFACE_PROC)(MessageBusInterface* instance);

static BOOL CALLBACK LoadMessageBusDllCallback(PINIT_ONCE, PVOID, PVOID*)
{
    // First, load the base path from registry.
    HKEY busKey;
    LONG result = RegCreateKeyExW(
        HKEY_LOCAL_MACHINE,
        {L"SOFTWARE\\LWPU Corporation\\LwContainer\\MessageBus"},
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

    wchar_t basePath[MAX_PATH] = {L""};
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

    // We only allow full path names, so make sure that the base path is not empty.
    if (s_basePath.empty())
    {
        return FALSE;
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
        resourceDataSize = LoadString(hData, IDS_MESSAGE_BUS_DEPENDENCIES, resourceData, _countof(resourceData));
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
        if (!dll.Load(dependencies.substr(start, (end == std::string::npos) ? std::string::npos : end - start)))
        {
            // Reset the list of dependencies in case of failure.
            s_dependencies.clear();
            return FALSE;
        }
        s_dependencies.push_back(dll);

        if (end != std::string::npos)
        {
            start = end+1;
        }
    }

    return TRUE;
}

static bool LoadMessageBusDll()
{
    // Execute the initialization callback function 
    return InitOnceExelwteOnce(&s_InitOnce, LoadMessageBusDllCallback, NULL, NULL) ? true : false;
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
