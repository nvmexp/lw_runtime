#pragma once

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <string>

typedef HRESULT (__stdcall *PFND3DCOMPILEFUNC)(
    LPCVOID pSrcData, SIZE_T SrcDataSize,
    LPCSTR pSourceName, const D3D_SHADER_MACRO * pDefines, ID3DInclude * pInclude, LPCSTR pEntrypoint,
    LPCSTR pTarget, UINT Flags1, UINT Flags2,
    ID3DBlob **ppCode, ID3DBlob **ppErrorMsgs
    );

typedef HRESULT (__stdcall * PFND3DCOMPILEFROMFILEFUNC)(
    LPCWSTR pFileName,
    CONST D3D_SHADER_MACRO* pDefines,
    ID3DInclude* pInclude,
    LPCSTR pEntrypoint,
    LPCSTR pTarget,
    UINT Flags1,
    UINT Flags2,
    ID3DBlob** ppCode,
    ID3DBlob** ppErrorMsgs
    );

typedef HRESULT (__stdcall *PFND3DREFLECTFUNC)(
    LPCVOID pSrcData,
    SIZE_T SrcDataSize,
    REFIID pInterface,
    void** ppReflector
    );

typedef HRESULT(__stdcall *PFND3DCREATEBLOBFUNC) (
    SIZE_T Size,
    ID3DBlob** ppBlob
    );

typedef HRESULT(__stdcall *PFND3DDISASSEMBLEFUNC) (
    LPCVOID pSrcData,
    SIZE_T SrcDataSize,
    UINT Flags,
    LPCSTR szComments,
    ID3DBlob** ppDisassembly
    );

class D3DCompilerHandler
{
public:
    D3DCompilerHandler() {}
    D3DCompilerHandler(const wchar_t * path)
    {
        setD3DCompilerPath(path);
    }

    ~D3DCompilerHandler()
    {
        freeD3DCompilerDLL();
    }

    void setD3DCompilerPath(const wchar_t * path);
    PFND3DCOMPILEFUNC getD3DCompileFunc();
    PFND3DCOMPILEFROMFILEFUNC getD3DCompileFromFileFunc();
    PFND3DREFLECTFUNC getD3DReflectFunc();
    PFND3DCREATEBLOBFUNC getD3DCreateBlobFunc();
    PFND3DDISASSEMBLEFUNC getD3DDisassembleFunc();

    void freeD3DCompilerDLL();
    
    const std::wstring& getD3DCompilerPath() const 
    {
        return m_D3DCompilerPath;
    }

protected:
    FARPROC GetProcAddressFromD3DCompilerModule(const char* procName);

    std::wstring m_D3DCompilerPath = L"";
    PFND3DCOMPILEFUNC m_D3DCompileFunc = nullptr;
    PFND3DCOMPILEFROMFILEFUNC m_D3DCompileFromFileFunc = nullptr;
    PFND3DREFLECTFUNC m_D3DReflectFunc = nullptr;
    PFND3DCREATEBLOBFUNC m_D3DCreateBlobFunc = nullptr;
    PFND3DDISASSEMBLEFUNC m_D3DDisassembleFunc = nullptr;
    HMODULE m_hD3DCompilerModule = nullptr;
};
 