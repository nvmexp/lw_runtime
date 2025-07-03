#pragma once

#include <string>
#include <windows.h>
#include <lwSelwreLoadLibrary.h>
#include "Log.h"

#include "D3DCompilerHandler.h"


void D3DCompilerHandler::setD3DCompilerPath(const wchar_t * path)
{
    if (m_D3DCompilerPath != path)
        freeD3DCompilerDLL();

    m_D3DCompilerPath = path;
}

PFND3DCOMPILEFUNC D3DCompilerHandler::getD3DCompileFunc()
{
    if (m_D3DCompileFunc)
        return m_D3DCompileFunc;

    m_D3DCompileFunc = (PFND3DCOMPILEFUNC)GetProcAddressFromD3DCompilerModule("D3DCompile");

    return m_D3DCompileFunc;
}

PFND3DCOMPILEFROMFILEFUNC D3DCompilerHandler::getD3DCompileFromFileFunc()
{
    if (m_D3DCompileFromFileFunc)
        return m_D3DCompileFromFileFunc;

    m_D3DCompileFromFileFunc = (PFND3DCOMPILEFROMFILEFUNC)GetProcAddressFromD3DCompilerModule("D3DCompileFromFile");

    return m_D3DCompileFromFileFunc;
}

PFND3DREFLECTFUNC D3DCompilerHandler::getD3DReflectFunc()
{
    if (m_D3DReflectFunc)
        return m_D3DReflectFunc;

    m_D3DReflectFunc = (PFND3DREFLECTFUNC)GetProcAddressFromD3DCompilerModule("D3DReflect");

    return m_D3DReflectFunc;
}

PFND3DCREATEBLOBFUNC D3DCompilerHandler::getD3DCreateBlobFunc()
{
    if (m_D3DCreateBlobFunc)
        return m_D3DCreateBlobFunc;

    m_D3DCreateBlobFunc = (PFND3DCREATEBLOBFUNC)GetProcAddressFromD3DCompilerModule("D3DCreateBlob");

    return m_D3DCreateBlobFunc;
}

PFND3DDISASSEMBLEFUNC D3DCompilerHandler::getD3DDisassembleFunc()
{
    if (m_D3DDisassembleFunc)
        return m_D3DDisassembleFunc;

    m_D3DDisassembleFunc = (PFND3DDISASSEMBLEFUNC)GetProcAddressFromD3DCompilerModule("D3DDisassemble");

    return m_D3DDisassembleFunc;
}

void D3DCompilerHandler::freeD3DCompilerDLL()
{
    if (m_hD3DCompilerModule)
        FreeLibrary(m_hD3DCompilerModule);

    m_hD3DCompilerModule = nullptr;
    m_D3DCompileFunc = nullptr;
    m_D3DReflectFunc = nullptr;
    m_D3DCompileFromFileFunc = nullptr;
    m_D3DCreateBlobFunc = nullptr;
    m_D3DDisassembleFunc = nullptr;
}

FARPROC D3DCompilerHandler::GetProcAddressFromD3DCompilerModule(const char* procName)
{
    if (!m_hD3DCompilerModule)
    {
        // We pass a false because the library is signed by Microsoft, not LWPU, and
        // lwLoadLibrary will verify the MS signature if we pass false.
        if (!m_D3DCompilerPath.empty())
            m_hD3DCompilerModule = lwLoadSignedLibraryW(m_D3DCompilerPath.c_str(), false);
    }

    if (m_hD3DCompilerModule)
        return GetProcAddress(m_hD3DCompilerModule, procName);

    DWORD errorCode = GetLastError();
    LOG_ERROR("D3DCompilerModule failed to load: ErrorCode 0x%08X", errorCode);

    return nullptr;
}
