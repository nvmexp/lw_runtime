// CFW1Factory.cpp

#include "FW1Precompiled.h"

#include "CFW1Factory.h"

void IFW1Factory::initDWrite()
{
    FW1FontWrapper::CFW1Factory::initDWrite();
}
void IFW1Factory::deinitDWrite()
{
    FW1FontWrapper::CFW1Factory::deinitDWrite();
}

namespace FW1FontWrapper {

HMODULE CFW1Factory::hDWriteLib = NULL;

// Construct
CFW1Factory::CFW1Factory() :
    m_cRefCount(1)
{
    InitializeCriticalSection(&m_errorStringCriticalSection);
}


// Destruct
CFW1Factory::~CFW1Factory() {
    DeleteCriticalSection(&m_errorStringCriticalSection);
}


// Init
HRESULT CFW1Factory::initFactory() {
    return S_OK;
}

void CFW1Factory::initDWrite()
{
    hDWriteLib = LoadLibrary(TEXT("DWrite.dll"));
}

void CFW1Factory::deinitDWrite()
{
    if (hDWriteLib)
    {
        FreeLibrary(hDWriteLib);
        if (!GetModuleHandle(TEXT("DWrite.dll")))
        {
            hDWriteLib = NULL;
        }
    }
}

// Create a DWrite factory
HRESULT CFW1Factory::createDWriteFactory(IDWriteFactory **ppDWriteFactory) {
    HRESULT hResult = E_FAIL;
    
    typedef HRESULT (WINAPI * PFN_DWRITECREATEFACTORY)(__in DWRITE_FACTORY_TYPE factoryType, __in REFIID iid, __out IUnknown **factory);
    PFN_DWRITECREATEFACTORY pfnDWriteCreateFactory = NULL;
    
#ifdef FW1_DELAYLOAD_DWRITE_DLL
    if(hDWriteLib == NULL) {
        // No need to reset error, load library is now external
        //DWORD dwErr = GetLastError();
        //dwErr;
        setErrorString(L"Failed to load DWrite.dll");
    }
    else {
        pfnDWriteCreateFactory =
            reinterpret_cast<PFN_DWRITECREATEFACTORY>(GetProcAddress(hDWriteLib, "DWriteCreateFactory"));
        if(pfnDWriteCreateFactory == NULL) {
            DWORD dwErr = GetLastError();
            dwErr;
            setErrorString(L"Failed to load DWriteCreateFactory");
        }
    }
#else
    pfnDWriteCreateFactory = DWriteCreateFactory;
#endif
    
    if(pfnDWriteCreateFactory != NULL) {
        IDWriteFactory *pDWriteFactory;
        
        hResult = pfnDWriteCreateFactory(
            DWRITE_FACTORY_TYPE_SHARED,
            __uuidof(IDWriteFactory),
            reinterpret_cast<IUnknown**>(&pDWriteFactory)
        );
        if(FAILED(hResult)) {
            setErrorString(L"DWriteCreateFactory failed");
        }
        else {
            *ppDWriteFactory = pDWriteFactory;
                
            hResult = S_OK;
        }
    }
    
    return hResult;
}


// Set error string
void CFW1Factory::setErrorString(const wchar_t *str) {
    EnterCriticalSection(&m_errorStringCriticalSection);
    m_lastError = str;
    LeaveCriticalSection(&m_errorStringCriticalSection);
}


}// namespace FW1FontWrapper
