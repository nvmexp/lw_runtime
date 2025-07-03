// Starting with VS2015:
// The signature of the wcstok function has been changed to match what is required by the C Standard
// A new _wcstok function has been added with the old signature to ease porting.
// You may define_CRT_NON_CONFORMING_WCSTOK to cause _wcstok to be used in place of wcstok.
#if _MSC_VER >= 1900
#define _CRT_NON_CONFORMING_WCSTOK
#endif

// Include Win32 API headers first:
#include <windows.h>
#include <wincrypt.h>
#include <wintrust.h>
#include <strsafe.h>
#include <objbase.h>
#include <setupapi.h>
#include <shellapi.h>

// Some projects build lwSelwreLoadLibrary.c by including it into another
// source file which may drop winsvc.h from above windows.h include per
// #define NOSERVICE ( implied by _ATL_NO_SERVICE and/or VC_EXTRALEAN )
#include <winsvc.h>

// Include lwSelwreLoadLibrary declarations:
#include "lwSelwreLoadLibrary.h"

// Disable some warnings
#pragma warning(push)          // Store the warning states
#pragma warning(disable: 4127) // Don't warn about const conditional expressions
#pragma warning(disable: 4189) // Don't warn about variables that aren't referenced
#pragma warning(disable: 4306) // Don't warn about colwersions to types of larger size
#pragma warning(disable: 4706) // Don't warn about assignments within conditional expressions
#pragma warning(disable: 4996) // Don't warn about wcs* function


#ifndef WINTRUST_ACTION_GENERIC_VERIFY_V2
// Defined in softpub.h, but not all build systems have that in the include path
#define WINTRUST_ACTION_GENERIC_VERIFY_V2                       \
            { 0xaac56b,                                         \
              0xcd44,                                           \
              0x11d0,                                           \
              { 0x8c, 0xc2, 0x0, 0xc0, 0x4f, 0xc2, 0x95, 0xee } \
            }
#endif

// _countof helper macro
#if !defined(_countof)
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif // !defined(_countof)

// Use StringCchCopy/CatW instead of wcsncpy/cat_s:
#define _my_wcscpy_s(dst, size, src) StringCchCopyW(dst, size, src)
#define _my_wcscat_s(dst, size, src) StringCchCatW(dst, size, src)

// These LOAD_LIBRARY_FLAGS may be undefined:
#if !defined(LOAD_LIBRARY_SEARCH_APPLICATION_DIR)
#define LOAD_LIBRARY_SEARCH_APPLICATION_DIR 0x00000200
#endif // !defined(LOAD_LIBRARY_SEARCH_APPLICATION_DIR)

#if !defined(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
#define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS 0x00001000
#endif // !defined(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)

#if !defined(LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
#define LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR 0x00000100
#endif // !defined(LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)

#if !defined(LOAD_LIBRARY_SEARCH_SYSTEM32)
#define LOAD_LIBRARY_SEARCH_SYSTEM32 0x00000800
#endif // !defined(LOAD_LIBRARY_SEARCH_SYSTEM32)

#if !defined(LOAD_LIBRARY_SEARCH_USER_DIRS)
#define LOAD_LIBRARY_SEARCH_USER_DIRS 0x00000400
#endif // !defined(LOAD_LIBRARY_SEARCH_USER_DIRS)

#if !defined(LOAD_WITH_ALTERED_SEARCH_PATH)
#define LOAD_WITH_ALTERED_SEARCH_PATH 0x00000008
#endif // !defined(LOAD_WITH_ALTERED_SEARCH_PATH)

// VALID_LOADLIBRARYEX_FLAGS should exclude any DLL search-related flags:
#define VALID_LOADLIBRARYEX_FLAGS ~(LOAD_LIBRARY_SEARCH_APPLICATION_DIR |\
                                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |\
                                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |\
                                    LOAD_LIBRARY_SEARCH_USER_DIRS |\
                                    LOAD_WITH_ALTERED_SEARCH_PATH)

// Inorder to remove dependency on shell32.lib, we dynamically load the shell32.dll and use SHGetFolderPathW
#ifndef CSIDL_WINDOWS
#define CSIDL_WINDOWS           0x0024  // GetWindowsDirectory()
#endif
#ifndef CSIDL_SYSTEM
#define CSIDL_SYSTEM            0x0025  // GetSystemDirectory()
#endif
#ifndef CSIDL_PROGRAM_FILES
#define CSIDL_PROGRAM_FILES     0x0026  // C:\Program Files
#endif
#ifndef CSIDL_SYSTEMX86
#define CSIDL_SYSTEMX86         0x0029  // x86 system directory on RISC
#endif
#ifndef CSIDL_PROGRAM_FILESX86
#define CSIDL_PROGRAM_FILESX86  0x002a  // x86 C:\Program Files on RISC
#endif

// Handle builds that don't include LwCfg:
#ifndef LWCFG
#define LWCFG(foo) 0
#endif

#if !LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
// Just emphasising that LocalFree ignores NULL args:
#define SafeLocalFree(x) LocalFree(x)
// Close HANDLE without changing last OS error:
#define SafeCloseHandle(x)                                \
    if ((NULL != (x)) && (ILWALID_HANDLE_VALUE != (x))) { \
        DWORD lastError = GetLastError();                 \
        CloseHandle(x); (x) = ILWALID_HANDLE_VALUE;       \
        SetLastError(lastError);                          \
    }
#else
#define SafeLocalFree(x)
#define SafeCloseHandle(x)
#endif

#define GetModule(moduleName, hModule) (((NULL == hModule) && (NULL == (hModule = LoadSystemLibraryInternal(moduleName, 0)))) ? FALSE : TRUE)
#define GetProc(hModule, procName, proc) (((NULL == proc) && (NULL == (*((FARPROC*)&proc) = GetProcAddress(hModule, procName)))) ? FALSE : TRUE)

// Output for local debugging. This needs to be set manually when compiling.
// Have to do the weird (()) dance to appease some ancient compilers we're still
// using that don't understand variadic macros.
#if defined(LWSELWRELOADLIBRARY_DEBUG_PRINTS)
#define LWSLL_DBG_2(fmt, ...) fprintf(stderr, "lwSLL[% 4d] %s(): " fmt, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#define LWSLL_DBG(format_and_args) LWSLL_DBG_2 format_and_args
#else
#define LWSLL_DBG(format_and_args)
#endif

//
// Forward declarations for common static functionality:
//

static HMODULE hModSetupapi = NULL;

typedef HDEVINFO (WINAPI *PfnSetupDiGetClassDevsW)(
    IN CONST GUID   *ClassGuid,
    IN       PCWSTR  Enumerator,
    IN       HWND    hwndParent,
    IN       DWORD   Flags
);
static PfnSetupDiGetClassDevsW pfnSetupDiGetClassDevsW = NULL;

typedef BOOL (WINAPI *PfnSetupDiEnumDeviceInterfaces)(
    IN       HDEVINFO                  DeviceInfoSet,
    IN       PSP_DEVINFO_DATA          DeviceInfoData,
    IN CONST GUID                     *InterfaceClassGuid,
    IN       DWORD                     MemberIndex,
    IN       PSP_DEVICE_INTERFACE_DATA DeviceInterfaceData
);
static PfnSetupDiEnumDeviceInterfaces pfnSetupDiEnumDeviceInterfaces = NULL;

typedef BOOL (WINAPI *PfnSetupDiGetDeviceInterfaceDetailW)(
    IN  HDEVINFO                           DeviceInfoSet,
    IN  PSP_DEVICE_INTERFACE_DATA          DeviceInterfaceData,
    OUT PSP_DEVICE_INTERFACE_DETAIL_DATA_W DeviceInterfaceDetailData,
    IN  DWORD                              DeviceInterfaceDetailDataSize,
    OUT PDWORD                             RequiredSize,
    OUT PSP_DEVINFO_DATA                   DeviceInfoData
);
static PfnSetupDiGetDeviceInterfaceDetailW pfnSetupDiGetDeviceInterfaceDetailW = NULL;

typedef BOOL (WINAPI *PfnSetupDiGetDeviceRegistryPropertyW)(
    IN  HDEVINFO         DeviceInfoSet,
    IN  PSP_DEVINFO_DATA DeviceInfoData,
    IN  DWORD            Property,
    OUT PDWORD           PropertyRegDataType,
    OUT PBYTE            PropertyBuffer,
    IN  DWORD            PropertyBufferSize,
    OUT PDWORD           RequiredSize
);
static PfnSetupDiGetDeviceRegistryPropertyW pfnSetupDiGetDeviceRegistryPropertyW = NULL;

//
// devpkey.h/devpropdef.h is not available to all consumers so we need to
// inline required definitions. Note that we cannot inline DEVPROPTYPE as
// devpropdef.h does not protected its definition by preproc #defines
//

#ifndef DEVPROPKEY_DEFINED
#define DEVPROPKEY_DEFINED

typedef GUID  DEVPROPGUID, *PDEVPROPGUID;
typedef ULONG DEVPROPID,   *PDEVPROPID;

typedef struct _DEVPROPKEY {
    DEVPROPGUID fmtid;
    DEVPROPID   pid;
} DEVPROPKEY, *PDEVPROPKEY;

#endif // DEVPROPKEY_DEFINED

#ifndef DEVPROP_TYPE_STRING
#define DEVPROP_TYPE_STRING 0x00000012  // null-terminated string
#endif // DEVPROP_TYPE_STRING

static const DEVPROPKEY DEVPKEY_Device_DriverInfPath = {{0xa8b865dd, 0x2e3d, 0x4094, {0xad, 0x97, 0xe5, 0x93, 0xa7, 0xc, 0x75, 0xd6}}, 5};

typedef BOOL (WINAPI *PfnSetupDiGetDevicePropertyW)(
    IN       HDEVINFO         DeviceInfoSet,
    IN       PSP_DEVINFO_DATA DeviceInfoData,
    IN CONST DEVPROPKEY      *PropertyKey,
    OUT      ULONG           *PropertyType,
    OUT      PBYTE            PropertyBuffer,
    IN       DWORD            PropertyBufferSize,
    OUT      PDWORD           RequiredSize,
    IN       DWORD            Flags
);
static PfnSetupDiGetDevicePropertyW pfnSetupDiGetDevicePropertyW = NULL;

typedef BOOL (WINAPI *PfnSetupDiDestroyDeviceInfoList)(
    IN HDEVINFO DeviceInfoSet
);
static PfnSetupDiDestroyDeviceInfoList pfnSetupDiDestroyDeviceInfoList = NULL;

typedef BOOL (WINAPI *PfnSetupGetInfDriverStoreLocationW)(
    IN  PCWSTR               FileName,
    IN  PSP_ALTPLATFORM_INFO AlternatePlatformInfo,
    IN  PCWSTR               LocaleName,
    OUT PWSTR                ReturnBuffer,
    IN  DWORD                ReturnBufferSize,
    OUT PDWORD               RequiredSize
);
static PfnSetupGetInfDriverStoreLocationW pfnSetupGetInfDriverStoreLocationW = NULL;

static HMODULE hModShell32 = NULL;

typedef HRESULT (WINAPI *PfnSHGetFolderPath_W)(HWND, int, HANDLE, DWORD, LPWSTR);
static PfnSHGetFolderPath_W pfnSHGetFolderPath = NULL;

static BOOL ContainsPathSeparator(LPCWSTR szFileName);
static BOOL ContainsAbsolutePath(LPCWSTR szFileName);
static BOOL StartsWithChar(LPCWSTR szString, WCHAR ch);

static LPWSTR CreateUnicodeStringFromAscii(LPCSTR szAscii);
static LPWSTR CreateSystemFilePath(LPCWSTR szFileName);
static LPWSTR CreateSHFolderFilePath(int nFolderId, LPCWSTR szFilePath);
static BOOL IsTrustedLocation(LPCWSTR szFilePath);
static BOOL FileExists(LPCWSTR szFileName);
static HANDLE LockFileForGenericReadAccess(LPCWSTR szFilePath);
static BOOL ValidateCreateProcessParameters(LPCWSTR lpApplicationName, LPCWSTR lpCommandLine, BOOL checkForLwidiaSignature);
static BOOL VerifySignature(LPCWSTR filePath, BOOL checkForLwidiaSignature);

#define StartsWithQuote( szString ) StartsWithChar ( szString, L'\"' )

typedef enum
{
    eWindowsVistaBuild             = 6000,
    eWindows7Build                 = 7600,
    eWindows8Build                 = 9200,
    eWindows8Point1Build           = 9600,
    eWindows10TH1Build             = 10240,
    eWindows10TH2Build             = 10586,
    eWindows10RS1DriverStoreSwitch = 14308,
    eWindows10RS4Build             = 17130
}
KnownWindowsBuildNumbers;

static BOOL IsWindowsBuildOrGreater(DWORD dwBuildNumber);
static BOOL DetectWow64Process(BOOL * pIsWow64Process);
static HMODULE LoadSystemLibraryInternal(const LPCWSTR fileName, DWORD dwFlags);

//
// Forward declarations for crypt32-specific static functionality:
//

#define ENCODING (X509_ASN_ENCODING | PKCS_7_ASN_ENCODING)

static HMODULE hModCrypt32 = NULL;

typedef BOOL (WINAPI *PfnCryptMsgClose)(IN HCRYPTMSG hCryptMsg);
static PfnCryptMsgClose pfnCryptMsgClose = NULL;

typedef BOOL (WINAPI *PfnCertCloseStore)(IN HCERTSTORE hCertStore, DWORD dwFlags);
static PfnCertCloseStore pfnCertCloseStore = NULL;

typedef BOOL (WINAPI *PfnCertFreeCertificateContext)(IN PCCERT_CONTEXT pCertContext);
static PfnCertFreeCertificateContext pfnCertFreeCertificateContext = NULL;

typedef PCCERT_CONTEXT (WINAPI *PfnCertFindCertificateInStore)(
    IN HCERTSTORE hCertStore,
    IN DWORD dwCertEncodingType,
    IN DWORD dwFindFlags,
    IN DWORD dwFindType,
    IN const void *pvFindPara,
    IN PCCERT_CONTEXT pPrevCertContext
);
static PfnCertFindCertificateInStore pfnCertFindCertificateInStore = NULL;

typedef BOOL (WINAPI *PfnCryptMsgGetParam)(
    IN HCRYPTMSG hCryptMsg,
    IN DWORD dwParamType,
    IN DWORD dwIndex,
    OUT void *pvData,
    IN OUT DWORD *pcbData
);
static PfnCryptMsgGetParam pfnCryptMsgGetParam = NULL;

typedef BOOL (WINAPI *PfnCryptQueryObject)(
    DWORD            dwObjectType,
    const void       *pvObject,
    DWORD            dwExpectedContentTypeFlags,
    DWORD            dwExpectedFormatTypeFlags,
    DWORD            dwFlags,
    DWORD            *pdwMsgAndCertEncodingType,
    DWORD            *pdwContentType,
    DWORD            *pdwFormatType,
    HCERTSTORE       *phCertStore,
    HCRYPTMSG        *phMsg,
    const void       **ppvContext
);
static PfnCryptQueryObject pfnCryptQueryObject = NULL;

typedef DWORD (WINAPI *PfnCertGetNameStringA)(
    IN PCCERT_CONTEXT pCertContext,
    IN DWORD dwType,
    IN DWORD dwFlags,
    IN void *pvTypePara,
    OUT OPTIONAL LPSTR pszNameString,
    IN DWORD cchNameString
);
static PfnCertGetNameStringA pfnCertGetNameStringA = NULL;

typedef BOOL (WINAPI *PfnCryptDecodeObjectEx)(
    IN DWORD              dwCertEncodingType,
    IN LPCSTR             lpszStructType,
    IN const BYTE         *pbEncoded,
    IN DWORD              cbEncoded,
    IN DWORD              dwFlags,
    IN PCRYPT_DECODE_PARA pDecodePara,
    OUT void              *pvStructInfo,
    IN OUT DWORD          *pcbStructInfo
);
static PfnCryptDecodeObjectEx pfnCryptDecodeObjectEx = NULL;

typedef PCCERT_CONTEXT (WINAPI *PfnCertGetIssuerCertificateFromStore)(
  IN          HCERTSTORE     hCertStore,
  IN          PCCERT_CONTEXT pSubjectContext,
  IN OPTIONAL PCCERT_CONTEXT pPrevIssuerContext,
  IN OUT      DWORD          *pdwFlags
);
static PfnCertGetIssuerCertificateFromStore pfnCertGetIssuerCertificateFromStore = NULL;

static HMODULE hModAdvapi32 = NULL;

typedef LONG (APIENTRY *PfnRegOpenKeyExW)(
    IN HKEY hKey,
    IN LPCWSTR lpSubKey,
    IN DWORD ulOptions,
    IN REGSAM samDesired,
    OUT PHKEY phkResult
);
static PfnRegOpenKeyExW pfnRegOpenKeyExW = NULL;

typedef LONG (APIENTRY *PfnRegEnumValueW)(
    IN HKEY hKey,
    IN DWORD dwIndex,
    OUT LPWSTR lpValueName,
    IN OUT LPDWORD lpcbValueName,
    IN LPDWORD lpReserved,
    OUT LPDWORD lpType,
    OUT LPBYTE lpData,
    IN OUT LPDWORD lpcbData
);
static PfnRegEnumValueW pfnRegEnumValueW = NULL;

typedef LONG (APIENTRY *PfnRegQueryValueExW)(
    IN HKEY hKey,
    IN LPCWSTR lpValueName,
    IN  LPDWORD lpReserved,
    OUT LPDWORD lpType,
    OUT LPBYTE lpData,
    IN OUT LPDWORD lpcbData
);
static PfnRegQueryValueExW pfnRegQueryValueExW = NULL;

typedef LONG (APIENTRY *PfnRegCloseKey)(IN HKEY hKey);
static PfnRegCloseKey pfnRegCloseKey = NULL;

typedef SC_HANDLE (WINAPI *PfnOpenSCManagerW)(
    IN LPCWSTR lpMachineName,
    IN LPCWSTR lpDatabaseName,
    IN DWORD dwDesiredAccess
);
static PfnOpenSCManagerW pfnOpenSCManagerW = NULL;

typedef SC_HANDLE (WINAPI *PfnOpenServiceW)(
    IN SC_HANDLE hSCManager,
    IN LPCWSTR lpServiceName,
    IN DWORD dwDesiredAccess
);
static PfnOpenServiceW pfnOpenServiceW = NULL;

typedef BOOL (WINAPI *PfnQueryServiceConfigW)(
    IN SC_HANDLE hService,
    OUT LPQUERY_SERVICE_CONFIGW lpServiceConfig,
    IN DWORD cbBufSize,
    OUT LPDWORD pcbBytesNeeded
);
static PfnQueryServiceConfigW pfnQueryServiceConfigW = NULL;

typedef BOOL (WINAPI *PfnQueryServiceStatus)(
    IN SC_HANDLE hService,
    OUT LPSERVICE_STATUS lpServiceStatus
);
static PfnQueryServiceStatus pfnQueryServiceStatus = NULL;

typedef BOOL (WINAPI *PfnCloseServiceHandle)(
    IN SC_HANDLE hSCObject
);
static PfnCloseServiceHandle pfnCloseServiceHandle = NULL;

static HMODULE hModOle32 = NULL;

typedef int (WINAPI *PfnStringFromGUID2)(IN REFGUID rguid, OUT LPOLESTR lpsz, IN int cchMax);
static PfnStringFromGUID2 pfnStringFromGUID2 = NULL;

typedef BOOL (APIENTRY *PfnCreateProcessAsUserW)(
    IN HANDLE hToken,
    IN LPCWSTR lpApplicationName,
    IN LPWSTR lpCommandLine,
    IN LPSELWRITY_ATTRIBUTES lpProcessAttributes,
    IN LPSELWRITY_ATTRIBUTES lpThreadAttributes,
    IN BOOL bInheritHandles,
    IN DWORD dwCreationFlags,
    IN LPVOID lpElwironment,
    IN LPCWSTR lpLwrrentDirectory,
    IN LPSTARTUPINFOW lpStartupInfo,
    OUT LPPROCESS_INFORMATION lpProcessInformation
    );
static PfnCreateProcessAsUserW pfnCreateProcessAsUserW = NULL;

typedef BOOL (APIENTRY * PfnCreateProcessAsUserA)(
    IN HANDLE hToken,
    IN LPCSTR lpApplicationName,
    IN LPSTR lpCommandLine,
    IN LPSELWRITY_ATTRIBUTES lpProcessAttributes,
    IN LPSELWRITY_ATTRIBUTES lpThreadAttributes,
    IN BOOL bInheritHandles,
    IN DWORD dwCreationFlags,
    IN LPVOID lpElwironment,
    IN LPCSTR lpLwrrentDirectory,
    IN LPSTARTUPINFOA lpStartupInfo,
    OUT LPPROCESS_INFORMATION lpProcessInformation
    );
static PfnCreateProcessAsUserA pfnCreateProcessAsUserA = NULL;

typedef HINSTANCE(WINAPI *PfnShellExelwteA)(
    IN HWND hwnd,
    IN LPCSTR lpOperation,
    IN LPCSTR lpFile,
    IN LPCSTR lpParameters,
    IN LPCSTR lpDirectory,
    IN INT nShowCmd);
static PfnShellExelwteA pfnShellExelwteA = NULL;

typedef HINSTANCE(WINAPI *PfnShellExelwteW)(
    IN HWND hwnd, 
    IN LPCWSTR lpOperation, 
    IN LPCWSTR lpFile, 
    IN LPCWSTR lpParameters,
    IN LPCWSTR lpDirectory, 
    IN INT nShowCmd
    );
static PfnShellExelwteW pfnShellExelwteW = NULL;

typedef BOOL(WINAPI *PfnShellExelwteExA)(IN SHELLEXELWTEINFOA *pExecInfo);
static PfnShellExelwteExA pfnShellExelwteExA = NULL;

typedef BOOL(WINAPI *PfnShellExelwteExW)(IN SHELLEXELWTEINFOW *pExecInfo);
static PfnShellExelwteExW pfnShellExelwteExW = NULL;

static HMODULE hModWinTrust   = NULL;

typedef LONG (WINAPI *PfnWilwerifyTrust)(IN HWND hWnd,IN GUID *pgActionID,IN LPVOID pWVTData);
static PfnWilwerifyTrust pfnWilwerifyTrust = NULL;

static HMODULE hModGDI32 = NULL;
static HMODULE hModCryptBase = NULL;
static HMODULE hModCryptNet = NULL;

// Copied from d3dkmthk.h for wddm2
typedef struct _LW_D3DKMT_ADAPTERINFO
{
    UINT  hAdapter;
    LUID  AdapterLuid;
    ULONG NumOfSources;
    BOOL  bPresentMoveRegionsPreferred;
} LW_D3DKMT_ADAPTERINFO;
typedef struct _LW_D3DKMT_ENUMADAPTERS2
{
    ULONG                 NumAdapters;
    LW_D3DKMT_ADAPTERINFO *pAdapters;
} LW_D3DKMT_ENUMADAPTERS2;
typedef LONG (APIENTRY *PfnD3DKMTEnumAdapters2)(IN OUT CONST LW_D3DKMT_ENUMADAPTERS2*);
static PfnD3DKMTEnumAdapters2 pfnD3DKMTEnumAdapters2 = NULL;

typedef struct _LW_D3DKMT_QUERYADAPTERINFO
{
    UINT  hAdapter;
    INT   Type;
    VOID* pPrivateDriverData;
    UINT  PrivateDriverDataSize;
} LW_D3DKMT_QUERYADAPTERINFO;
typedef LONG (APIENTRY *PfnD3DKMTQueryAdapterInfo)(IN OUT CONST LW_D3DKMT_QUERYADAPTERINFO*);
static PfnD3DKMTQueryAdapterInfo pfnD3DKMTQueryAdapterInfo = NULL;


static LPCWSTR GetUMDPathFromQAI();
static LPCWSTR GetUMDPathFromLoadedDLLs();
static BOOL TestCertificateChain(PCCERT_CONTEXT pCertContext, const LPCSTR chain[]);
static BOOL IsPeLwidiaSigned(PCCERT_CONTEXT pCertContext);
static BOOL GetSignerInfoTimeStamp(PCMSG_SIGNER_INFO pSignerInfo, FILETIME *pFiletime);
static BOOL VerifyTimeStampSignerInfo(PCMSG_SIGNER_INFO pSignerInfo, HCERTSTORE hStore, FILETIME *pFiletime);
static BOOL VerifyTimeStampRFC3161(PCMSG_SIGNER_INFO pSignerInfo, FILETIME *pFiletime);
static BOOL OverrideSignatureVerificationFailure(LPCWSTR fileName, DWORD verificationError);

//
// lwSelwreLoadLibrary implementation:
//

HMODULE lwGetSystemModuleHandleW(LPCWSTR fileName)
{
    HMODULE hResult = NULL;

#if (DEVELOP || LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT))
    hResult = GetModuleHandleW(fileName);
#else

    // Clear last error.
    SetLastError(ERROR_SUCCESS);

    // Validate fileName.
    if (ContainsPathSeparator(fileName)) {

        // fileName isn't supposed to contain path fragments.
        SetLastError(ERROR_BAD_ARGUMENTS);

    } else {

        // Create absolute system file path and load library from there.
        LPWSTR systemFilePath = CreateSystemFilePath(fileName);
        if (!fileName || systemFilePath) {
            hResult = GetModuleHandleW(systemFilePath);
        }

        // Clean up.
        SafeLocalFree(systemFilePath);
    }
#endif

    // Done.
    return hResult;
}

HMODULE lwGetSystemModuleHandleA(LPCSTR fileName)
{
    // Simply use UNICODE version.
    HMODULE hResult = NULL;
#if !LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
    LPWSTR unicodeFileName = CreateUnicodeStringFromAscii(fileName);
    if (!fileName || unicodeFileName) {
        hResult = lwGetSystemModuleHandleW(unicodeFileName);
    }
    SafeLocalFree(unicodeFileName);
#endif
    return hResult;
}

static HMODULE LoadSystemLibraryInternal(const LPCWSTR fileName, DWORD dwFlags)
{
    LPWSTR systemFilePath = NULL;
    HMODULE hRet;
    static DWORD dwBuildNumber = 0;
    // The only valid place for system libraries to live is the system folder
    // We cannot build the full path to the system folder as on some OS builds
    // this folder is just a placeholder and the actual DLLs live in a different
    // location. Fortunately, SEARCH_SYSTEM32 is overloaded there so it will
    // always use the correct location.
    // Unfortunately, SEARCH_SYSTEM32 requires KB2533623 to be installed on Win7
    // or the call fails. Since win7 has no alternative locations aliased, we
    // can construct the path there.

    // First strip all flags affecting the load order..
    dwFlags &= VALID_LOADLIBRARYEX_FLAGS;

    // HACK! We cannot use IsWindowsBuildOrGreater() as that ends up calling back
    // here, triggering an infinite relwrsion. Since we only care about win7 vs 
    // later, we can use this API instead - it's not precise, but good enough.
    if (dwBuildNumber == 0)
    {
        OSVERSIONINFOEXW osvi;
        DWORDLONG dwlConditionMask = VerSetConditionMask(0, VER_BUILDNUMBER, VER_GREATER_EQUAL);
        memset(&osvi, 0, sizeof(osvi));
        osvi.dwBuildNumber = eWindows8Build;

        dwBuildNumber = VerifyVersionInfoW(&osvi, VER_BUILDNUMBER, dwlConditionMask) ? eWindows8Build : eWindows7Build;
    }

    if (dwBuildNumber >= eWindows8Build)
    {
        dwFlags |= LOAD_LIBRARY_SEARCH_SYSTEM32;
    }
    else // Win7
    {
        dwFlags &= ~LOAD_LIBRARY_SEARCH_SYSTEM32;
        systemFilePath = CreateSystemFilePath(fileName);
    }
    LWSLL_DBG(("fileName: '%S', dwFlags:0x%x\n", systemFilePath ? systemFilePath : fileName, dwFlags));
    hRet = LoadLibraryExW(systemFilePath ? systemFilePath : fileName, NULL, dwFlags);
    SafeLocalFree(systemFilePath);
    return hRet;
}

HMODULE lwLoadSystemLibraryExW(LPCWSTR fileName, DWORD dwFlags)
{
    HMODULE hResult = NULL;

#if ( DEVELOP || LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT))
    hResult = LoadLibraryExW(fileName, NULL, dwFlags & VALID_LOADLIBRARYEX_FLAGS);
#else
    // Clear last error.
    SetLastError(ERROR_SUCCESS);

    // fileName isn't supposed to be NULL and must not contain path fragments.
    if (ContainsPathSeparator(fileName)) {

        // fileName isn't supposed to contain path fragments.
        SetLastError(ERROR_BAD_ARGUMENTS);

    } else {

        //
        // lwLoadSystemLibrary/RS1 WAR
        //
        // Starting with Windows 10 RS1, driver modules are supposed to live
        // in the driver store instead of the system directory.
        //
        // Thus, we try to detect the location of LWPU display driver modules
        // to load them from there instead of the system directory.
        //
        // Note that this should not change the behavior w.r.t. already loaded
        // modules because both the original lwLoadSystemLibrary as well as
        // this WAR use fully qualified, ie absolute paths such that LoadLibrary
        // "searches only that path for the module", ie ignores modules loaded
        // from different locations (ie if the caller's intention was to re-use
        // already loaded modules from different locations  lwLoadSystemLibrary
        // was the wrong call in the 1st place).
        //
        // If the LOAD_LIBRARY_SEARCH_SYSTEM32 flag was passed explicitly, we
        // bypass the driver store and go for system32 anyway.
        //
        if (TRUE == IsWindowsBuildOrGreater(eWindows10RS1DriverStoreSwitch) &&
            !(dwFlags & LOAD_LIBRARY_SEARCH_SYSTEM32))
        {
            //
            // Optimization: skip module detection for !lw modules
            //
            // To minimize latency for non-LWPU modules, we leverage the
            // cirlwmstance that most/all LWPU modules have an "lw" prefix
            // and skip detection for all others.
            //
            // This should reduce an additional latency penalty to lw-prefixed
            // non-LWPU modules.
            //
            // However, this also excludes LWPU driver modules without an "lw"
            // prefix from this WAR. If there is/will be such a module then it
            // must get loaded by calling lwLoadDisplayDriverModule directly.
            //

            const LPCWSTR lwPrefix = L"lw";
            if (0 == _wcsnicmp(fileName, lwPrefix, wcslen(lwPrefix)))
            {
                // Try to detect LWPU display driver module location

                LPWSTR pModuleLocation = lwDetectDisplayDriverModuleLocationW(fileName);
                DWORD dwLastError = GetLastError();
                if (NULL != pModuleLocation)
                {
                    // Have module location: try to load module from
                    // there using lwLoadLibraryFromTrustedLocation

                    hResult = lwLoadLibraryFromTrustedLocationExW(pModuleLocation, dwFlags);
                    SafeLocalFree(pModuleLocation);

                    // Done: just return lwLoadLibraryFromTrustedLocation
                    // result, ie don't try LoadSystemLibraryInternal on
                    // lwLoadLibraryFromTrustedLocation failures

                    return hResult;
                }
                else if (ERROR_MOD_NOT_FOUND == dwLastError)
                {
                    // ERROR_MOD_NOT_FOUND just indicates that given module
                    // name doesn't refer to an LWPU driver module: just
                    // fall through and try to load it from system32 below
                }
                else
                {
                    // Anything else than ERROR_MOD_NOT_FOUND is a real/hard
                    // error: just bail without trying to load from system32

                    return hResult;
                }
            }
        }

        // Load provided module from the system directory

        hResult = LoadSystemLibraryInternal(fileName, dwFlags);
    }
#endif

    // Done.
    return hResult;
}

HMODULE lwLoadSystemLibraryExA(LPCSTR fileName, DWORD dwFlags)
{
    // Simply use UNICODE version.
    HMODULE hResult = NULL;
    LPWSTR unicodeFileName = CreateUnicodeStringFromAscii(fileName);
    if (!fileName || unicodeFileName) {
        hResult = lwLoadSystemLibraryExW(unicodeFileName, dwFlags);
    }
    SafeLocalFree(unicodeFileName);
    return hResult;
}

HMODULE lwLoadLibraryFromTrustedLocationExW(LPCWSTR filePath, DWORD dwFlags)
{
#if !( DEVELOP || LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT))
    // Clear last error.
    SetLastError(ERROR_SUCCESS);

    // filePath must specifiy an absolute path.
    if (!ContainsAbsolutePath(filePath)) {
        SetLastError(ERROR_BAD_ARGUMENTS);
        return NULL;
    }

    // Check if filePath is pointing to a trusted location
    if (!IsTrustedLocation(filePath)) {
        SetLastError(ERROR_BAD_PATHNAME);
        return NULL;
    }
#endif

    // Done.
    LWSLL_DBG(("filePath: '%S'; dwFlags: 0x%08x\n", filePath, dwFlags));
    return LoadLibraryExW(filePath, NULL, dwFlags & VALID_LOADLIBRARYEX_FLAGS);
}

HMODULE lwLoadLibraryFromTrustedLocationExA(LPCSTR filePath, DWORD dwFlags)
{
    // Simply use UNICODE version.
    HMODULE hResult = NULL;
    LPWSTR unicodeFilePath = CreateUnicodeStringFromAscii(filePath);
    if (!filePath || unicodeFilePath) {
        hResult = lwLoadLibraryFromTrustedLocationExW(unicodeFilePath, dwFlags);
    }
    SafeLocalFree(unicodeFilePath);
    return hResult;
}

HMODULE lwLoadSignedLibraryExW(LPCWSTR filePath, DWORD dwFlags, const BOOL checkForLwidiaSignature)
{
#if ( DEVELOP || LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT))
    return LoadLibraryExW(filePath, NULL, dwFlags & VALID_LOADLIBRARYEX_FLAGS);
#else
    HANDLE hFileLock = ILWALID_HANDLE_VALUE;
    HMODULE hResult = NULL;

    // Clear last error.
    SetLastError(ERROR_SUCCESS);

    // filePath must specifiy an absolute path.
    if (!ContainsAbsolutePath(filePath)) {
        SetLastError(ERROR_BAD_ARGUMENTS);
    }
    // Check if file exists.
    else if (!FileExists(filePath)) {
        SetLastError(ERROR_MOD_NOT_FOUND);
    }
    // Try to lock file for { validate, load } sequence
    else if (ILWALID_HANDLE_VALUE == (hFileLock = LockFileForGenericReadAccess(filePath)))
    {
        // Failed to lock file - report ERROR_SHARING_VIOLATION
        SetLastError(ERROR_SHARING_VIOLATION);
    }
    // { validate, load } sequence
    else
    {
        // Try to verify file signature:
        BOOL bSignedByLwidia = FALSE;
        BOOL bSignatureVerified = lwVerifyFileSignatureW(filePath, checkForLwidiaSignature ? &bSignedByLwidia : NULL);

        // Pretend verification failure for the case that given
        // file is signed, but not by LWPU (as requested):
        if (bSignatureVerified && checkForLwidiaSignature && !bSignedByLwidia) {
            SetLastError((DWORD)CRYPT_E_NO_MATCH);
            bSignatureVerified = FALSE;
        }

        // Load library if wecould verify its file signature
        // or override any signature verification failures:
        if (bSignatureVerified || OverrideSignatureVerificationFailure(filePath, GetLastError())) {
            LWSLL_DBG(("filePath: '%S'; dwFlags: 0x%08x\n", filePath, dwFlags));
            hResult = LoadLibraryExW(filePath, NULL, dwFlags & VALID_LOADLIBRARYEX_FLAGS);
        }

        // Release file lock.
        SafeCloseHandle(hFileLock);
    }

    // Done.
    return hResult;
#endif
}

HMODULE lwLoadSignedLibraryExA(LPCSTR filePath, DWORD dwFlags, const BOOL checkForLwidiaSignature)
{
    // Simply use UNICODE version.
    HMODULE hResult = NULL;
    LPWSTR unicodeFilePath = CreateUnicodeStringFromAscii(filePath);
    if (!filePath || unicodeFilePath) {
        hResult = lwLoadSignedLibraryExW(unicodeFilePath, dwFlags, checkForLwidiaSignature);
    }
    SafeLocalFree(unicodeFilePath);
    return hResult;
}

HMODULE lwLoadLibraryExW(LPCWSTR filePath, DWORD dwFlags, const BOOL checkForLwidiaSignature)
{
#if ( DEVELOP || LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT))
    return LoadLibraryExW(filePath, NULL, dwFlags & VALID_LOADLIBRARYEX_FLAGS);
#else
    HANDLE hFileLock = ILWALID_HANDLE_VALUE;
    HMODULE pResult = NULL;

    /* Clear last error */
    SetLastError(ERROR_SUCCESS);

    /* fileName must specifiy an absolute path. */
    if (!ContainsAbsolutePath(filePath)) {
        SetLastError(ERROR_BAD_ARGUMENTS);
    }
    /* Check if file exists. */
    else if (!FileExists(filePath)) {
        SetLastError(ERROR_MOD_NOT_FOUND);
    }
    /* Try to lock file for { validate, load } sequence */
    else if (ILWALID_HANDLE_VALUE == (hFileLock = LockFileForGenericReadAccess(filePath)))
    {
        /* Failed to lock file - report ERROR_SHARING_VIOLATION */
        SetLastError(ERROR_SHARING_VIOLATION);
    }
    /* { validate, load } sequence */
    else
    {
        /* Check if file lives in a trusted location */
        if (!IsTrustedLocation(filePath))
        {
            /* File doesn't live in a trusted location: try to verify file signature */
            BOOL bSignedByLwidia = FALSE;
            BOOL bSignatureVerified = lwVerifyFileSignatureW(filePath, checkForLwidiaSignature ? &bSignedByLwidia : NULL);

            /* Pretend verification failure for the case that given
               file is signed, but not by LWPU (as requested) */
            if (bSignatureVerified && checkForLwidiaSignature && !bSignedByLwidia) {
                SetLastError((DWORD)CRYPT_E_NO_MATCH);
                bSignatureVerified = FALSE;
            }

            /* Try to override signature verification failures */
            if (!bSignatureVerified && !OverrideSignatureVerificationFailure(filePath, GetLastError())) {
                SafeCloseHandle(hFileLock);
                return NULL;
            }
        }

        /* Load the library */
        LWSLL_DBG(("filePath: '%S'; dwFlags: 0x%08x\n", filePath, dwFlags));
        pResult = LoadLibraryExW(filePath, NULL, dwFlags);
        SafeCloseHandle(hFileLock);
    }

    /* Done. */
    return pResult;
#endif
}

HMODULE lwLoadLibraryExA(LPCSTR filePath, DWORD dwFlags, const BOOL checkForLwidiaSignature)
{
    /* Simply use UNICODE version */
    HMODULE hResult = NULL;
    LPWSTR unicodeFilePath = NULL;
    if (!filePath)
    {
        SetLastError(ERROR_ILWALID_PARAMETER);
        return NULL;
    }
    unicodeFilePath = CreateUnicodeStringFromAscii(filePath);
    if (unicodeFilePath) {
        hResult = lwLoadLibraryExW(unicodeFilePath, dwFlags, checkForLwidiaSignature);
    }
    SafeLocalFree(unicodeFilePath);
    return hResult;
}

static BOOL VerifySignature(LPCWSTR filePath, BOOL checkForLwidiaSignature)
{
    BOOL bSignedByLwidia = FALSE;
    BOOL bSignatureVerified = FALSE;

    bSignatureVerified = lwVerifyFileSignatureW(filePath, checkForLwidiaSignature ? &bSignedByLwidia : NULL);

    /* Pretend verification failure for the case that given
       file is signed, but not by LWPU (as requested) */
    if (bSignatureVerified && checkForLwidiaSignature && !bSignedByLwidia) 
    {
        SetLastError((DWORD)CRYPT_E_NO_MATCH);
        bSignatureVerified = FALSE;
    }

    /* Try to override signature verification failures */
    if (!bSignatureVerified && !OverrideSignatureVerificationFailure(filePath, GetLastError())) 
    {
        return FALSE;
    }

    return TRUE;
}

static BOOL ValidateCreateProcessParameters(LPCWSTR lpApplicationName, LPCWSTR lpCommandLine, BOOL checkForLwidiaSignature)
{
#if LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
    return TRUE;
#else
    LPWSTR pCommandLine = NULL;
    LPWSTR pApp = NULL;
    BOOL   bValid = FALSE;
    DWORD  dwLastError = 0;
    size_t size = 0;

    //
    // Clear last error
    //
    SetLastError(ERROR_SUCCESS);

    //
    // Make sure that either lpApplicationName or lpCommandLine is 
    // not NULL and we are referring to proper application name
    //
    pApp = (LPWSTR)lpApplicationName;
    if (!lpApplicationName)
    {
        if (!lpCommandLine)
        {
            SetLastError(ERROR_ILWALID_PARAMETER);
            return bValid;
        }
        size = (wcslen(lpCommandLine) + 1) * sizeof(WCHAR);
        pCommandLine = (LPWSTR)LocalAlloc(LPTR, size);
        if (!pCommandLine)
        {
            return bValid;
        }
        memcpy(pCommandLine, lpCommandLine, size);
        if (StartsWithQuote(pCommandLine))
        {
            //
            // If command line starts with quotes, then application name is 
            // something which is in quotes. Rest of it can be assumed to be
            // argument to that application
            // 
            pApp = wcstok(pCommandLine, L"\"");
        }
        else
        {
            //
            // If command line does not start with quotes then all of it 
            // or the string up to space character is application we are 
            // interested in. Remaining portion is command line to that 
            // application
            // 
            pApp = wcstok(pCommandLine, L" ");
        }
    }

    if (!pApp)
    {
        SafeLocalFree(pCommandLine);
        SetLastError(ERROR_ILWALID_PARAMETER);
        return bValid;
    }
    if (!ContainsAbsolutePath(pApp))
    {
        SafeLocalFree(pCommandLine);
        SetLastError(ERROR_BAD_ARGUMENTS);
        return bValid;
    }

    /* Check if file lives in a trusted location */
    if (IsTrustedLocation(pApp))
    {
        SafeLocalFree(pCommandLine);
        return TRUE;
    }

    bValid = VerifySignature(pApp, checkForLwidiaSignature);
    if (!bValid)
    {
        dwLastError = GetLastError();
    }
    SafeLocalFree(pCommandLine);
    if (!bValid)
    {
        SetLastError(dwLastError);
    }
    return bValid;
#endif
}

BOOL
lwCreateProcessW(
    LPCWSTR lpApplicationName,
    LPWSTR lpCommandLine,
    LPVOID lpProcessAttributes, // LPSELWRITY_ATTRIBUTES lpProcessAttributes,
    LPVOID lpThreadAttributes, // LPSELWRITY_ATTRIBUTES lpThreadAttributes,
    BOOL bInheritHandles,
    DWORD dwCreationFlags,
    LPVOID lpElwironment,
    LPCWSTR lpLwrrentDirectory,
    LPVOID lpStartupInfo, // LPSTARTUPINFOW lpStartupInfo,
    BOOL checkForLwidiaSignature,
    LPVOID lpProcessInformation // LPPROCESS_INFORMATION lpProcessInformation
    )
{
#ifndef DEVELOP
    SetLastError(ERROR_SUCCESS);
    if (!ValidateCreateProcessParameters(lpApplicationName, lpCommandLine, checkForLwidiaSignature))
    {
        return FALSE;
    }
#endif

    return CreateProcessW(lpApplicationName, lpCommandLine, (LPSELWRITY_ATTRIBUTES)lpProcessAttributes, (LPSELWRITY_ATTRIBUTES)lpThreadAttributes, bInheritHandles, dwCreationFlags, 
        lpElwironment, lpLwrrentDirectory, (LPSTARTUPINFOW)lpStartupInfo, (LPPROCESS_INFORMATION)lpProcessInformation);
}

BOOL
lwCreateProcessA(
    LPCSTR lpApplicationName,
    LPSTR lpCommandLine,
    LPVOID lpProcessAttributes, // LPSELWRITY_ATTRIBUTES lpProcessAttributes,
    LPVOID lpThreadAttributes, //LPSELWRITY_ATTRIBUTES lpThreadAttributes,
    BOOL bInheritHandles,
    DWORD dwCreationFlags,
    LPVOID lpElwironment,
    LPCSTR lpLwrrentDirectory,
    LPVOID lpStartupInfo, // LPSTARTUPINFOA lpStartupInfo,
    BOOL checkForLwidiaSignature,
    LPVOID lpProcessInformation // LPPROCESS_INFORMATION lpProcessInformation
    )
{
#ifndef DEVELOP
    /* Simply use UNICODE version */
    LPWSTR unicodeApplicationName = NULL;
    LPWSTR unicodeCommandLine = NULL;

    SetLastError(ERROR_SUCCESS);
    if (lpApplicationName)
    {
        unicodeApplicationName = CreateUnicodeStringFromAscii(lpApplicationName);
    }
    if (lpCommandLine)
    {
        unicodeCommandLine = CreateUnicodeStringFromAscii(lpCommandLine);
    }
    if (!ValidateCreateProcessParameters(unicodeApplicationName, unicodeCommandLine, checkForLwidiaSignature))
    {
        SafeLocalFree(unicodeApplicationName);
        SafeLocalFree(unicodeCommandLine);
        return FALSE;
    }

    SafeLocalFree(unicodeApplicationName);
    SafeLocalFree(unicodeCommandLine);
#endif

    return CreateProcessA(lpApplicationName, lpCommandLine, (LPSELWRITY_ATTRIBUTES)lpProcessAttributes, (LPSELWRITY_ATTRIBUTES)lpThreadAttributes, bInheritHandles, dwCreationFlags, 
        lpElwironment, lpLwrrentDirectory, (LPSTARTUPINFOA)lpStartupInfo, (LPPROCESS_INFORMATION)lpProcessInformation);

}

BOOL
lwCreateProcessAsUserW(
    HANDLE hToken,
    LPCWSTR lpApplicationName,
    LPWSTR lpCommandLine,
    LPVOID lpProcessAttributes, // LPSELWRITY_ATTRIBUTES lpProcessAttributes,
    LPVOID lpThreadAttributes, // LPSELWRITY_ATTRIBUTES lpThreadAttributes,
    BOOL bInheritHandles,
    DWORD dwCreationFlags,
    LPVOID lpElwironment,
    LPCWSTR lpLwrrentDirectory,
    LPVOID lpStartupInfo, // LPSTARTUPINFOW lpStartupInfo,
    BOOL checkForLwidiaSignature,
    LPVOID lpProcessInformation // LPPROCESS_INFORMATION lpProcessInformation
    )
{
#ifndef DEVELOP
    SetLastError(ERROR_SUCCESS);
    if (!ValidateCreateProcessParameters(lpApplicationName, lpCommandLine, checkForLwidiaSignature))
    {
        return FALSE;
    }
#endif

    if (NULL == hModAdvapi32)
    {
        // Dynamically load Advapi32.dll to remove dependency on Advapi32.lib
        hModAdvapi32 = LoadSystemLibraryInternal(L"Advapi32.dll", 0);
        if (NULL == hModAdvapi32)
        {
            return FALSE;
        }
    }
    if (NULL == pfnCreateProcessAsUserW)
    {
        pfnCreateProcessAsUserW = (PfnCreateProcessAsUserW)GetProcAddress(hModAdvapi32, "CreateProcessAsUserW");
        if (NULL == pfnCreateProcessAsUserW)
        {
            return FALSE;
        }
    }

    return pfnCreateProcessAsUserW(hToken, lpApplicationName, lpCommandLine, (LPSELWRITY_ATTRIBUTES)lpProcessAttributes, (LPSELWRITY_ATTRIBUTES)lpThreadAttributes, bInheritHandles, dwCreationFlags, 
        lpElwironment, lpLwrrentDirectory, (LPSTARTUPINFOW)lpStartupInfo, (LPPROCESS_INFORMATION)lpProcessInformation);
}

BOOL
lwCreateProcessAsUserA(
    HANDLE hToken,
    LPCSTR lpApplicationName,
    LPSTR lpCommandLine,
    LPVOID lpProcessAttributes, // LPSELWRITY_ATTRIBUTES lpProcessAttributes,
    LPVOID lpThreadAttributes, // LPSELWRITY_ATTRIBUTES lpThreadAttributes,
    BOOL bInheritHandles,
    DWORD dwCreationFlags,
    LPVOID lpElwironment,
    LPCSTR lpLwrrentDirectory,
    LPVOID lpStartupInfo, // LPSTARTUPINFOA lpStartupInfo,
    BOOL checkForLwidiaSignature,
    LPVOID lpProcessInformation // LPPROCESS_INFORMATION lpProcessInformation
    )
{
#ifndef DEVELOP
    /* Simply use UNICODE version */
    LPWSTR unicodeApplicationName = NULL;
    LPWSTR unicodeCommandLine = NULL;

    SetLastError(ERROR_SUCCESS);
    if (lpApplicationName)
    {
        unicodeApplicationName = CreateUnicodeStringFromAscii(lpApplicationName);
    }
    if (lpCommandLine)
    {
        unicodeCommandLine = CreateUnicodeStringFromAscii(lpCommandLine);
    }
    if (!ValidateCreateProcessParameters(unicodeApplicationName, unicodeCommandLine, checkForLwidiaSignature))
    {
        SafeLocalFree(unicodeApplicationName);
        SafeLocalFree(unicodeCommandLine);
        return FALSE;
    }

    SafeLocalFree(unicodeApplicationName);
    SafeLocalFree(unicodeCommandLine);
#endif

    if (NULL == hModAdvapi32) 
    {
        // Dynamically load Advapi32.dll to remove dependency on Advapi32.lib
        hModAdvapi32 = LoadSystemLibraryInternal(L"Advapi32.dll", 0);
        if (NULL == hModAdvapi32)
        {
            return FALSE;
        }
    }
    if (NULL == pfnCreateProcessAsUserA)
    {
        pfnCreateProcessAsUserA = (PfnCreateProcessAsUserA)GetProcAddress(hModAdvapi32, "CreateProcessAsUserA");
        if (NULL == pfnCreateProcessAsUserA)
        {
            return FALSE;
        }
    }
    return pfnCreateProcessAsUserA(hToken, lpApplicationName, lpCommandLine, (LPSELWRITY_ATTRIBUTES)lpProcessAttributes, (LPSELWRITY_ATTRIBUTES)lpThreadAttributes, bInheritHandles, dwCreationFlags, 
        lpElwironment, lpLwrrentDirectory, (LPSTARTUPINFOA)lpStartupInfo, (LPPROCESS_INFORMATION)lpProcessInformation);
}

HINSTANCE
lwShellExelwteA(
    HWND hwnd,
    LPCSTR lpOperation,
    LPCSTR lpFile,
    LPCSTR lpParameters,
    LPCSTR lpDirectory,
    INT nShowCmd,
    BOOL checkForLwidiaSignature
    )
{
#ifndef DEVELOP
    /* Simply use UNICODE version */
    LPWSTR unicodeApplicationName = NULL;
    LPWSTR unicodeCommandLine = NULL;

    SetLastError(ERROR_SUCCESS);
    if (lpFile)
    {
        unicodeApplicationName = CreateUnicodeStringFromAscii(lpFile);
    }
    if (lpParameters)
    {
        unicodeCommandLine = CreateUnicodeStringFromAscii(lpParameters);
    }
    if (!ValidateCreateProcessParameters(unicodeApplicationName, unicodeCommandLine, checkForLwidiaSignature))
    {
        SafeLocalFree(unicodeApplicationName);
        SafeLocalFree(unicodeCommandLine);
        return FALSE;
    }

    SafeLocalFree(unicodeApplicationName);
    SafeLocalFree(unicodeCommandLine);
#endif

    if (NULL == hModShell32)
    {
        // Dynamically load Shell32.dll to remove dependency on Shell32.lib
        hModShell32 = LoadSystemLibraryInternal(L"Shell32.dll", 0);
        if (NULL == hModShell32)
        {
            return FALSE;
        }
    }
    if (NULL == pfnShellExelwteA)
    {
        pfnShellExelwteA = (PfnShellExelwteA)GetProcAddress(hModShell32, "ShellExelwteA");
        if (NULL == pfnShellExelwteA)
        {
            return FALSE;
        }
    }

    return pfnShellExelwteA(hwnd, lpOperation, lpFile, lpParameters, lpDirectory, nShowCmd);
}

HINSTANCE
lwShellExelwteW(
    HWND hwnd,
    LPCWSTR lpOperation,
    LPCWSTR lpFile,
    LPCWSTR lpParameters,
    LPCWSTR lpDirectory,
    INT nShowCmd,
    BOOL checkForLwidiaSignature
)
{
#ifndef DEVELOP
    SetLastError(ERROR_SUCCESS);
    if (!ValidateCreateProcessParameters(lpFile, lpParameters, checkForLwidiaSignature))
    {
        return FALSE;
    }
#endif

    if (NULL == hModShell32)
    {
        // Dynamically load Shell32.dll to remove dependency on Shell32.lib
        hModShell32 = LoadSystemLibraryInternal(L"Shell32.dll", 0);
        if (NULL == hModShell32)
        {
            return FALSE;
        }
    }
    if (NULL == pfnShellExelwteW)
    {
        pfnShellExelwteW = (PfnShellExelwteW)GetProcAddress(hModShell32, "ShellExelwteW");
        if (NULL == pfnShellExelwteW)
        {
            return FALSE;
        }
    }

    return pfnShellExelwteW(hwnd, lpOperation, lpFile, lpParameters, lpDirectory, nShowCmd);
}

BOOL lwShellExelwteExA(
    SHELLEXELWTEINFOA *pExecInfo,
    BOOL checkForLwidiaSignature
)
{
#ifndef DEVELOP
    /* Simply use UNICODE version */
    LPWSTR unicodeApplicationName = NULL;
    LPWSTR unicodeCommandLine = NULL;

    SetLastError(ERROR_SUCCESS);

    if (pExecInfo == NULL)
    {
        SetLastError(ERROR_BAD_ARGUMENTS);
        return FALSE;
    }
    if (pExecInfo->lpFile)
    {
        unicodeApplicationName = CreateUnicodeStringFromAscii(pExecInfo->lpFile);
    }
    if (pExecInfo->lpParameters)
    {
        unicodeCommandLine = CreateUnicodeStringFromAscii(pExecInfo->lpParameters);
    }
    if (!ValidateCreateProcessParameters(unicodeApplicationName, unicodeCommandLine, checkForLwidiaSignature))
    {
        SafeLocalFree(unicodeApplicationName);
        SafeLocalFree(unicodeCommandLine);
        return FALSE;
    }

    SafeLocalFree(unicodeApplicationName);
    SafeLocalFree(unicodeCommandLine);
#endif

    if (NULL == hModShell32)
    {
        // Dynamically load Shell32.dll to remove dependency on Shell32.lib
        hModShell32 = LoadSystemLibraryInternal(L"Shell32.dll", 0);
        if (NULL == hModShell32)
        {
            return FALSE;
        }
    }
    if (NULL == pfnShellExelwteExA)
    {
        pfnShellExelwteExA = (PfnShellExelwteExA)GetProcAddress(hModShell32, "ShellExelwteExA");
        if (NULL == pfnShellExelwteExA)
        {
            return FALSE;
        }
    }

    return pfnShellExelwteExA(pExecInfo);
}

BOOL lwShellExelwteExW(
     SHELLEXELWTEINFOW *pExecInfo,
     BOOL checkForLwidiaSignature
     )
{
#ifndef DEVELOP

    SetLastError(ERROR_SUCCESS);

    if (pExecInfo == NULL)
    {
        SetLastError(ERROR_BAD_ARGUMENTS);
        return FALSE;
    }

    if (!ValidateCreateProcessParameters(pExecInfo->lpFile, pExecInfo->lpParameters, checkForLwidiaSignature))
    {
        return FALSE;
    }

#endif

    if (NULL == hModShell32)
    {
        // Dynamically load Shell32.dll to remove dependency on Shell32.lib
        hModShell32 = LoadSystemLibraryInternal(L"Shell32.dll", 0);
        if (NULL == hModShell32)
        {
            return FALSE;
        }
    }
    if (NULL == pfnShellExelwteExW)
    {
        pfnShellExelwteExW = (PfnShellExelwteExW)GetProcAddress(hModShell32, "ShellExelwteExW");
        if (NULL == pfnShellExelwteExW)
        {
            return FALSE;
        }
    }

    return pfnShellExelwteExW(pExecInfo);
}

LPWSTR CreateSystemFilePath(LPCWSTR szFileName)
{
    LPWSTR pResult = NULL;
#if !LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
    UINT i;

    // Just create system paths for !NULL fileNames
    // (, i.e. don't unroll NULL fileNames!)
    if (szFileName) {

        // Query required number of chars to store system directory.
        DWORD dwLength = GetSystemDirectoryW(pResult, 0);

        // Allocate system directory string
        pResult = (LPWSTR)LocalAlloc(LPTR, (dwLength + 1 + wcslen(szFileName)) * sizeof(WCHAR));
        if (!pResult) {
            return NULL;
        }

        // Query system directory.
        dwLength = GetSystemDirectoryW(pResult, dwLength);

        // Make sure that pResult ends with '\\'.
        if (pResult[dwLength-1] != '\\') {
            pResult[dwLength++] = '\\';
        }

        // Append fileName.
        for (i = 0; i < wcslen(szFileName); ++i) {
            pResult[dwLength+i] = szFileName[i];
        }
    }
#endif
    // Done.
    return pResult;
}

#if !LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
// nFolderId should be one of CSIDL_XXXXX values
static LPWSTR CreateSHFolderFilePath(int nFolderId, LPCWSTR szFilePath)
{
    LPWSTR pResult = NULL;

    // Just create program file paths for !NULL szFilePath
    // (, i.e. don't unroll NULL filePaths!)
    if (szFilePath) {

        HRESULT hGetFolderPathResult;
        WCHAR szKnownFolderPath[MAX_PATH+1];
        ZeroMemory(szKnownFolderPath, sizeof(szKnownFolderPath));

        if (NULL == hModShell32) {
            // Dynamically load shell32.dll to remove dependency on shell32.lib
            if (NULL != (hModShell32 = LoadSystemLibraryInternal(L"Shell32.dll", 0))) {
                // Note: SHGetFolderPathW is available only in Shell32.dll version 5 or
                // later (Windows Millennium Edition (Windows Me) and Windows 2000).
                // Since we support only XP and above, the following code should be okay.
                // If support needs to be extended to earlier versions, then
                // SHGetSpecialFolderPath (deprecated on the newer platforms) should 
                // be used.
                pfnSHGetFolderPath = (PfnSHGetFolderPath_W)GetProcAddress(hModShell32, "SHGetFolderPathW");
            }
        }

        // Sanity-check pfnSHGetFolderPath
        if (NULL == pfnSHGetFolderPath)
            return NULL;

        // Query nFolderId (for example CSIDL_PROGRAM_FILES)
        if (S_OK == (hGetFolderPathResult = (pfnSHGetFolderPath)(NULL, nFolderId, NULL, 0, szKnownFolderPath))) {

            // Allocate program files directory string
            const size_t pResultChars = wcslen(szKnownFolderPath) + 1 + wcslen(szFilePath) + 1;
            if (!(pResult = (LPWSTR)LocalAlloc(LPTR, pResultChars * sizeof(WCHAR)))) {
                return NULL;
            }

            // Concatenate FOLDERID_nFolderId + "\\" + szFileName
            _my_wcscpy_s(pResult, pResultChars, szKnownFolderPath);
            _my_wcscat_s(pResult, pResultChars, L"\\");
            _my_wcscat_s(pResult, pResultChars, szFilePath);
        }
    }

    // Done.
    return pResult;
}
#endif

static LPWSTR CreateUnicodeStringFromAscii(LPCSTR szAscii)
{
    LPWSTR pResult = NULL;
#if !LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
    UINT i;

    if (szAscii) {
        if (pResult = (LPWSTR)LocalAlloc(LPTR, (strlen(szAscii) + 1) * sizeof(WCHAR))) {
            for (i = 0; i < strlen(szAscii); ++i) {
                pResult[i] = szAscii[i];
            }
        }
    }
#endif
    return pResult;
}

static BOOL ContainsPathSeparator(LPCWSTR szFileName)
{
    UINT i;

    if (szFileName) {
        for (i = 0; i < wcslen(szFileName); ++i) {
            if ((szFileName[i] == '\\') || (szFileName[i] == '/')) {
                return TRUE;
            }
        }
    }

    return FALSE;
}

static BOOL ContainsAbsolutePath(LPCWSTR szFileName)
{
    if (!szFileName) {
        return FALSE;
    }

    if ((szFileName[0] == '\\') || (szFileName[0] == '/')) {
        return TRUE;
    }

    if ((isalpha(szFileName[0]) && (szFileName[1] == ':')) && ((szFileName[2] == '\\') || (szFileName[2] == '/'))) {
        return TRUE;
    }

    return FALSE;
}

static BOOL StartsWithChar(LPCWSTR szString, WCHAR ch)
{
    if (szString)
    {
        if (szString[0] == ch)
        {
            return TRUE;
        }
    }
    return FALSE;
}

static BOOL FileExists(LPCWSTR szFileName)
{
    DWORD fileAttributes = GetFileAttributesW(szFileName);
    const DWORD nonFileAttributes = FILE_ATTRIBUTE_DEVICE | FILE_ATTRIBUTE_DIRECTORY;
    return ((ILWALID_FILE_ATTRIBUTES == fileAttributes) || (fileAttributes & nonFileAttributes)) ? FALSE : TRUE;
}

static HANDLE LockFileForGenericReadAccess(LPCWSTR szFilePath)
{
    // CreateFile() should be sufficient because open files cannot be moved or deleted per
    // http://msdn.microsoft.com/en-us/library/windows/desktop/aa365244(v=vs.85).aspx and
    // http://msdn.microsoft.com/en-us/library/windows/desktop/bb540533(v=vs.85).aspx .
    // Note that we restrict shared access to FILE_SHARE_READ to make sure that there is
    // no write access to given file:
    return CreateFileW(
        szFilePath,      // _In_      LPCTSTR lpFileName,
        GENERIC_READ,    // _In_      DWORD dwDesiredAccess,
        FILE_SHARE_READ, // _In_      DWORD dwShareMode,
        NULL,            // _In_opt_  LPSELWRITY_ATTRIBUTES lpSelwrityAttributes,
        OPEN_EXISTING,   // _In_      DWORD dwCreationDisposition,
        0,               // _In_      DWORD dwFlagsAndAttributes,
        NULL             // _In_opt_  HANDLE hTemplateFile
    );
}

#ifndef _WIN32_WINNT_WIN7
// Copied from the winblue WDK's sdkddkver.h...
#define _WIN32_WINNT_WIN7 0x0601
#endif // #ifndef _WIN32_WINNT_WIN7

static OSVERSIONINFOEXW osVersionInfo = { 0 };
static BOOL bTrustAnyLocation = TRUE;

static BOOL IsTrustedLocation(LPCWSTR szFilePath)
{
    DWORD fullPathNameSize = 0;
    DWORD fullPathNameChars = 0;
    LPWSTR fullPathName = NULL;
    LPWSTR szAllowListedLocation = NULL;
    BOOL bResult = FALSE;
    const int nCSIDL_FoldersToCheck[] = {CSIDL_WINDOWS, CSIDL_PROGRAM_FILES, CSIDL_PROGRAM_FILESX86};
    const int range = _countof(nCSIDL_FoldersToCheck);
    int index = 0;

    // Check forQuery OS version info (if not done yet)
    if (!osVersionInfo.dwOSVersionInfoSize)
    {
        // CSIDL_SYSTEMX86 doesn't map to "SysWOW64" on WinXP 64bit (so there is
        // no way to allowlist that), and %ProgramW6432% is not available before
        // Win7. Long-story short: simply trust any location on OSes < Win7:

        DWORDLONG const dwlConditionMask = VerSetConditionMask(
            VerSetConditionMask(
            VerSetConditionMask(
                0, VER_MAJORVERSION, VER_GREATER_EQUAL),
                   VER_MINORVERSION, VER_GREATER_EQUAL),
                   VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL);

        osVersionInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEXW);
        osVersionInfo.dwMajorVersion = HIBYTE(_WIN32_WINNT_WIN7);
        osVersionInfo.dwMinorVersion = LOBYTE(_WIN32_WINNT_WIN7);
        osVersionInfo.wServicePackMajor = 0;

        // bTrustAnyLocation := (OS version >= Win7) ? FALSE : TRUE
#if LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
        bTrustAnyLocation = true;
#else 
        bTrustAnyLocation = VerifyVersionInfoW(&osVersionInfo, VER_MAJORVERSION | VER_MINORVERSION | VER_SERVICEPACKMAJOR, dwlConditionMask) ? FALSE : TRUE;
#endif
    }
    
    // Bail if we trust any location:
    if (bTrustAnyLocation) {
        return TRUE;
    }

#if !LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
    // Sanity-check szFilePath
    if (!szFilePath) {
        goto isTrustedLocationDone;
    }

    // Query size of full path name, where fullPathNameSize includes '\0' if nBufferLength is too small.
    if (0 == (fullPathNameSize = GetFullPathNameW(szFilePath, 0, NULL, NULL))) {
        goto isTrustedLocationDone;
    }

    // Allocate full path name buffer
    if (NULL == (fullPathName = (LPWSTR)LocalAlloc(LPTR, fullPathNameSize * sizeof(WCHAR)))) {
        goto isTrustedLocationDone;
    }

    // Query full path name, where fullPathNameChars does *NOT* include '\0' if nBufferLength was large
    // enough. However, fullPathNameChars might be eve less than reported fullPathNameSize because the
    // GetFullPathNameW(szFilePath, 0, NULL, NULL) above apparently doesn't consider that f.e. "\\\\"
    // (double backslash) will actually get condensed to "\\" (single backslash) in the final result!
    if (fullPathNameSize <= (fullPathNameChars = GetFullPathNameW(szFilePath, fullPathNameSize, fullPathName, NULL))) {
        goto isTrustedLocationDone;
    }

    //
    // We look at both the regular and WoW folders to check if the given path is from a trusted location, since there are some
    // components (eg, 3D Vision) that always get installed into WoW folders. See Bug: 1342161 for more details.
    // Note that calling into CSIDL_SYSTEMX86/CSIDL_PROGRAM_FILESX86 on a 32-bit machine would return the normal
    // system32 and Program Files location and hence the below code is generic. For more details on how the CSIDL_XXX 
    // macros are routed on various platforms, see http://msdn.microsoft.com/en-us/library/windows/desktop/dd378457(v=vs.85).aspx
    //
    for(index = 0; index < range; index++) {
        // Retrieve path of next allowlisted location szAllowListedLocation := nCSIDL_FoldersToCheck[index]
#ifndef _WIN64
        if (CSIDL_PROGRAM_FILES == nCSIDL_FoldersToCheck[index]) {
            // There are some 32bit LWPU applications that are getting installed
            // to the 64bit "Program Files" folder, such that they need this function
            // to consider not only "Program Files (x86)", but also "Program Files"
            // as a trusted location for a 32bit application. Although this should
            // get fixed in the according application, we're implementing a WAR as
            // a short-term solution here.
            // Unfortunatelly, the CSIDL_PROGRAM_FILES* mapping doesn't do this job
            // (as CSIDL_SYSTEM actually does for the system directory), and we
            // cannot just use Wow64DisableWow64FsRedirection to bless (because this
            // function just disables file system redirection, and not registry
            // redirection - which is what is being used by SHGetFolderPath). Thus,
            // we need to go the _unsafe_ path by evaluating %ProgramW6432% here:
            DWORD nSize = ExpandElwironmentStringsW(L"%ProgramW6432%", NULL, 0);
            if (!nSize || !(szAllowListedLocation = (LPWSTR)LocalAlloc(LPTR, (nSize+1) * sizeof(WCHAR)))) {
                goto isTrustedLocationDone;
            } else if (nSize != ExpandElwironmentStringsW(L"%ProgramW6432%", szAllowListedLocation, nSize)) {
                goto isTrustedLocationDone;
            } else if (!lstrcmpW(L"%ProgramW6432%", szAllowListedLocation)) {
                // Failed to expand %ProgramW6432% (likely because we're not running in
                // WOW mode), so use CreateSHFolderFilePath() instead (note that we need
                // check for existence by trying to expand via ExpandElwironmentStrings()
                // because GetElwironmentVariableW() doesn't know about this variable!):
                SafeLocalFree(szAllowListedLocation);
                if (!(szAllowListedLocation = CreateSHFolderFilePath(nCSIDL_FoldersToCheck[index], L""))) {
                    // Failed to retreive nCSIDL_FoldersToCheck[index]: bail w/ bResult == FALSE.
                    goto isTrustedLocationDone;
                }
            } else if (szAllowListedLocation[nSize-2] != L'\\') {
                // %ProgramW6432% doesn't include a trailing backslash. That's why
                // we had to allocate nSize+1 chars and add this backslash here:
                _my_wcscat_s(szAllowListedLocation, nSize+1, L"\\");
            }
        } else 
#endif // _WIN64
        if (!(szAllowListedLocation = CreateSHFolderFilePath(nCSIDL_FoldersToCheck[index], L""))) {
            // Failed to retreive nCSIDL_FoldersToCheck[index]: bail w/ bResult == FALSE.
            goto isTrustedLocationDone;
        }

        // Check if szFilePath points to (a subdirectory of) szAllowListedLocation.
        if (bResult = _wcsnicmp(fullPathName, szAllowListedLocation, wcslen(szAllowListedLocation)) ? FALSE : TRUE) {
            // fullPathName points to (a subdirectory of) szAllowListedLocation: bail w/ bResult == TRUE.
            goto isTrustedLocationDone;
        }

        // Release szAllowListedLocation and continue.
        SafeLocalFree(szAllowListedLocation);
        szAllowListedLocation = NULL;
    }

isTrustedLocationDone:

    // Clean up.
    SafeLocalFree(szAllowListedLocation);
    SafeLocalFree(fullPathName);

#endif
    // Done.
    return bResult;
}



static DWORD GetSignatureCount(LPCWSTR fileName)
{
    // TODO: Use WSS_GET_SECONDARY_SIG_COUNT on Win8+ to support dual-signing
    // TODO bug 2038341 lwSelwreLoadLibrary should support multiple signatures
    UNREFERENCED_PARAMETER(fileName);
    return 1;
}

#define LW_ANSEL_BUILD
static LONG VerifySingleSignature(LPCWSTR fileName, DWORD index)
{
#if defined(LW_DRIVER_BUILD_WGF2UM) || defined(LW_DRIVER_BUILD_LWLDUMD) || defined(LW_ANSEL_BUILD)
    // TODO: Disabled for these builds due to unacceptable performance regressions
    // This is not a regression from previous functinality, as this function
    // was never called before.
    //
    // Ansel is following driver compilation and also disabling this check due to
    // unacceptable performance regressions until a more performant alternative
    // is in place. Ansel lwrrently just uses this when loading the d3dcompiler dll,
    // which is used to compile the shaders for filter effects.
    return 0;
#else
    WINTRUST_FILE_INFO FileData;
    WINTRUST_DATA WinTrustData;
    GUID WVTPolicyGUID = WINTRUST_ACTION_GENERIC_VERIFY_V2;

    // TODO: Iterate through all signatures on win8+
    UNREFERENCED_PARAMETER(index);

    memset(&FileData, 0, sizeof(FileData));
    FileData.cbStruct = sizeof(WINTRUST_FILE_INFO);
    FileData.pcwszFilePath = fileName;

    memset(&WinTrustData, 0, sizeof(WinTrustData));
    WinTrustData.cbStruct = sizeof(WinTrustData);
    WinTrustData.dwUIChoice = WTD_UI_NONE;
    WinTrustData.fdwRevocationChecks = WTD_REVOKE_WHOLECHAIN;
    WinTrustData.dwUnionChoice = WTD_CHOICE_FILE;
    WinTrustData.dwStateAction = WTD_STATEACTION_IGNORE;
    WinTrustData.dwProvFlags |= WTD_CACHE_ONLY_URL_RETRIEVAL;
    WinTrustData.pFile = &FileData;

    return pfnWilwerifyTrust((HWND)ILWALID_HANDLE_VALUE, &WVTPolicyGUID, &WinTrustData);
#endif
}
static BOOL IsLwidiaSubordinatePublicKey(const CRYPT_BIT_BLOB *pPublicKey)
{
    // TODO: bug 2038342 lwSelwreLoadLibrary should not hardcode the LWPU Subordinate CA Public Key
    const BYTE LWIDIA_Subordinate_CA_2016_v2_PublicKey[] = 
    {
        0x30, 0x82, 0x01, 0x0a, 0x02, 0x82, 0x01, 0x01, 0x00, 0xd3, 0x23, 0xb1,
        0xcb, 0xcb, 0xd4, 0x03, 0xc8, 0x12, 0xf0, 0x9e, 0x0a, 0x9d, 0x0b, 0xac,
        0xf8, 0x28, 0x58, 0xb8, 0x16, 0xb3, 0x89, 0x86, 0x98, 0x52, 0x86, 0xcd,
        0x5b, 0xd3, 0x83, 0xa2, 0xf7, 0x53, 0x2e, 0x1c, 0x00, 0x3d, 0x7d, 0x67,
        0xcb, 0x99, 0x86, 0x03, 0xbd, 0xb0, 0xa0, 0x8a, 0x2d, 0x92, 0x4e, 0x7f,
        0x29, 0xd1, 0xfb, 0x34, 0xe9, 0x83, 0x96, 0x16, 0x93, 0x41, 0xdc, 0x54,
        0x03, 0x62, 0x3e, 0x51, 0x0a, 0x6f, 0xae, 0xe5, 0xe8, 0xf6, 0x8b, 0x60,
        0xf6, 0x24, 0xdb, 0x41, 0x8e, 0xef, 0x20, 0xc9, 0xec, 0x8d, 0x02, 0x26,
        0xd1, 0xc9, 0x2f, 0xa0, 0xbe, 0xd9, 0xd8, 0x7d, 0x38, 0xe1, 0x59, 0x1f,
        0x37, 0xc0, 0x17, 0x84, 0x76, 0x40, 0xb3, 0xcb, 0x5a, 0x76, 0xe1, 0x5a,
        0x8c, 0x38, 0xa5, 0xa9, 0x73, 0x4c, 0xbb, 0x00, 0x92, 0x85, 0xd1, 0x7f,
        0x4e, 0x74, 0x4d, 0x19, 0x63, 0x39, 0x62, 0x62, 0xb8, 0xc3, 0x69, 0x08,
        0x67, 0xf7, 0x23, 0xfe, 0x04, 0xab, 0x9e, 0x36, 0xc9, 0x75, 0x3a, 0xc8,
        0x43, 0xd7, 0xb8, 0x0c, 0x60, 0x25, 0x79, 0x12, 0x3e, 0x05, 0x93, 0x79,
        0x45, 0x56, 0x5d, 0x41, 0xf3, 0x62, 0x43, 0x19, 0xc1, 0x5a, 0xec, 0x44,
        0x22, 0xd2, 0xd2, 0x95, 0x5d, 0xf7, 0x5c, 0x41, 0xf5, 0xdb, 0x99, 0x03,
        0x8b, 0x89, 0xa4, 0xc3, 0x93, 0xad, 0x8d, 0xa8, 0x0f, 0xa9, 0x3a, 0xf0,
        0xa3, 0xd7, 0xee, 0x81, 0xe7, 0x80, 0x4b, 0x25, 0x53, 0x7d, 0x8e, 0xfa,
        0x3d, 0x49, 0x26, 0xfc, 0xdd, 0x31, 0x4c, 0x73, 0x2d, 0xcc, 0xb7, 0x89,
        0xb7, 0xb1, 0xe8, 0x14, 0xcd, 0xd6, 0x93, 0x07, 0xf9, 0x01, 0xb7, 0xf9,
        0x35, 0xb4, 0x2f, 0x92, 0xf5, 0x9a, 0xb3, 0xd0, 0x38, 0x8d, 0x08, 0xbf,
        0x22, 0xb0, 0xa5, 0x65, 0x82, 0x90, 0x2a, 0x12, 0x55, 0xa1, 0x48, 0x11,
        0xff, 0x02, 0x03, 0x01, 0x00, 0x01
    };

    const BYTE LWIDIA_Subordinate_CA_2018_Prod_SHA1[] = 
    {
        0x30, 0x82, 0x01, 0x0a, 0x02, 0x82, 0x01, 0x01, 0x00, 0xa6, 0xd4, 0xc0,
        0xcf, 0x17, 0x35, 0x65, 0x8e, 0x13, 0x84, 0xda, 0xfa, 0xe4, 0xa2, 0x6e,
        0x85, 0xe7, 0xe0, 0x14, 0x98, 0x62, 0x18, 0xec, 0x27, 0x9f, 0xad, 0x55,
        0xf7, 0x9b, 0x11, 0x1a, 0x2b, 0xbe, 0x43, 0xce, 0x53, 0x67, 0x42, 0x78,
        0xb7, 0xa3, 0xd7, 0x7b, 0x74, 0xc5, 0xd6, 0x79, 0x88, 0x7f, 0x5c, 0x7d,
        0x7b, 0xda, 0x74, 0xaa, 0x97, 0xa6, 0x01, 0x93, 0x60, 0xa0, 0x59, 0xf7,
        0x05, 0x4f, 0x4b, 0xf9, 0xec, 0xfd, 0x9b, 0xff, 0xe7, 0xc4, 0xc1, 0x99,
        0x70, 0xf1, 0x29, 0x60, 0x0e, 0xe2, 0x86, 0xd2, 0x60, 0x7f, 0x52, 0xb3,
        0x17, 0x02, 0xb1, 0xe9, 0x89, 0x71, 0x86, 0x1b, 0xb9, 0xf5, 0xef, 0xd3,
        0x95, 0xa3, 0x08, 0xe5, 0x62, 0x11, 0xd7, 0x1a, 0x74, 0xa2, 0x8e, 0x52,
        0x7b, 0xc9, 0x44, 0x32, 0x99, 0xfe, 0xb9, 0xb5, 0x6f, 0x7e, 0x25, 0x37,
        0x19, 0x41, 0x58, 0x54, 0x05, 0xf5, 0x3e, 0xa5, 0x1b, 0x8f, 0x2b, 0x8f,
        0xe6, 0xeb, 0x80, 0x1e, 0x86, 0xdc, 0x89, 0x00, 0x23, 0xad, 0x29, 0xd2,
        0x4a, 0x4c, 0x25, 0xbf, 0xf7, 0x18, 0x9e, 0x24, 0x78, 0xcf, 0xbd, 0x36,
        0x11, 0x58, 0x76, 0x24, 0xdf, 0x76, 0x35, 0x01, 0xc2, 0x34, 0x7b, 0x1a,
        0x60, 0x46, 0xc6, 0x57, 0x23, 0x28, 0x4f, 0x15, 0xe7, 0x36, 0xb8, 0x1e,
        0x7a, 0xe5, 0x8a, 0xbf, 0x11, 0xc8, 0x9a, 0x90, 0x38, 0xd1, 0x0a, 0xf7,
        0xd1, 0x85, 0xfa, 0x04, 0xaf, 0xb2, 0xd3, 0x81, 0x99, 0x91, 0x5b, 0xe3,
        0xe8, 0x62, 0x15, 0xa5, 0xd0, 0xc3, 0xfc, 0x6b, 0xc6, 0x40, 0x4f, 0x8a,
        0xda, 0xf4, 0x0b, 0x57, 0x44, 0x21, 0x90, 0x48, 0xa1, 0x55, 0xbb, 0x35,
        0x75, 0xe3, 0x8b, 0x73, 0x29, 0xfa, 0x7d, 0xbf, 0x89, 0xb9, 0xf0, 0x46,
        0xcd, 0xfb, 0xab, 0xf6, 0xc1, 0xf5, 0x6b, 0x07, 0xbb, 0x76, 0x47, 0x17,
        0xe7, 0x02, 0x03, 0x01, 0x00, 0x01
    };

    const BYTE LWIDIA_Subordinate_CA_2018_Prod_SHA2[] = 
    {
        0x30, 0x82, 0x01, 0x0a, 0x02, 0x82, 0x01, 0x01, 0x00, 0xaf, 0x3f, 0x6a,
        0xae, 0x3b, 0x5c, 0xb6, 0x2c, 0x29, 0xfe, 0x1a, 0x3b, 0x50, 0x66, 0xfe,
        0xf7, 0x1b, 0xd0, 0x78, 0xb9, 0x30, 0x59, 0xf8, 0xb8, 0x00, 0xa8, 0x7a,
        0x67, 0x3d, 0xc5, 0x86, 0x06, 0xd9, 0x10, 0xb0, 0xe6, 0x6b, 0x4f, 0xaa,
        0xa9, 0x8d, 0xe8, 0x69, 0x60, 0xcd, 0xb1, 0x88, 0xcc, 0x19, 0x2d, 0xc5,
        0x9c, 0x51, 0xd8, 0x37, 0xb6, 0x41, 0xd7, 0x6a, 0x79, 0xf0, 0x5c, 0x6a,
        0x4e, 0xfb, 0xa2, 0xef, 0xb2, 0xd0, 0x38, 0x32, 0xd3, 0xa9, 0xe5, 0x03,
        0xeb, 0x2b, 0xec, 0xd2, 0xf0, 0x17, 0x1f, 0x6f, 0xae, 0xe1, 0xe2, 0xda,
        0x5c, 0x10, 0xb5, 0x42, 0x63, 0xbe, 0x4d, 0x0b, 0x08, 0x72, 0xc1, 0xb8,
        0xbb, 0x1a, 0x1a, 0x46, 0x21, 0xd0, 0xe7, 0xdc, 0x85, 0x36, 0xf6, 0x89,
        0x2a, 0x9d, 0xcc, 0x83, 0x59, 0x67, 0x8a, 0xfd, 0x15, 0x53, 0x4e, 0xd2,
        0x8c, 0x3a, 0x23, 0x70, 0x5d, 0x1e, 0x63, 0x39, 0x6a, 0x77, 0x54, 0x54,
        0x66, 0x6c, 0x5a, 0x24, 0x7c, 0xdd, 0x18, 0xd0, 0xc0, 0xa6, 0x7c, 0x5a,
        0xde, 0xe7, 0xa3, 0xb9, 0x4d, 0x41, 0xad, 0x6d, 0x76, 0x5f, 0x2a, 0x82,
        0x1c, 0x6f, 0x0a, 0x3c, 0x1b, 0x4b, 0x0e, 0xd4, 0x93, 0x51, 0xaf, 0x8c,
        0x04, 0x1a, 0x9e, 0x6c, 0x81, 0x34, 0x94, 0xb0, 0x20, 0xf0, 0xa1, 0x64,
        0x5a, 0x3a, 0xa7, 0x66, 0xf6, 0x66, 0x4a, 0x2e, 0xf1, 0xde, 0x85, 0x26,
        0x54, 0xa9, 0x24, 0x91, 0x3f, 0xa5, 0x1e, 0xbf, 0x44, 0x9d, 0x9c, 0x1c,
        0x28, 0xf8, 0x16, 0x2c, 0x8c, 0x31, 0xf3, 0xc3, 0x79, 0x71, 0x9e, 0x7b,
        0xe9, 0xbe, 0x96, 0x0e, 0x6a, 0x55, 0x65, 0x23, 0x54, 0xd1, 0x71, 0xea,
        0x71, 0x51, 0x42, 0xb2, 0xcc, 0x88, 0xf0, 0x91, 0x28, 0x9e, 0xc8, 0xc6,
        0xab, 0xfd, 0x24, 0x85, 0x4b, 0xa4, 0x46, 0x58, 0xae, 0xd5, 0xc3, 0xd0,
        0xd5, 0x02, 0x03, 0x01, 0x00, 0x01
    };

    const BYTE LWIDIA_Subordinate_CA_2019_Prod_SHA1[] = 
    {
        0x30, 0x82, 0x01, 0x0a, 0x02, 0x82, 0x01, 0x01, 0x00, 0xb4, 0x85, 0x66,
        0x85, 0xcc, 0x49, 0x35, 0x56, 0xc0, 0x91, 0xf7, 0x0b, 0xb3, 0x04, 0x34,
        0xf2, 0xab, 0x6a, 0xb9, 0xb2, 0x25, 0x43, 0x64, 0xc5, 0xa4, 0x7f, 0x27,
        0x6b, 0x7c, 0x3c, 0x91, 0x6b, 0x85, 0x0a, 0xf6, 0x0c, 0xda, 0x76, 0xd6,
        0xa5, 0xaf, 0xbb, 0xc6, 0xf0, 0xef, 0x24, 0x3c, 0x09, 0xa7, 0x04, 0xdd,
        0xfa, 0x2f, 0x3d, 0x97, 0xa2, 0x4f, 0x91, 0xdc, 0x5f, 0x58, 0x84, 0x23,
        0x7f, 0xce, 0x61, 0xa5, 0x46, 0x98, 0xcf, 0x75, 0xc7, 0x4e, 0x3e, 0x3b,
        0x7e, 0x22, 0xc8, 0xab, 0xa8, 0xbf, 0x86, 0x3f, 0x9a, 0x02, 0x70, 0xc8,
        0x30, 0x8c, 0x86, 0x21, 0x97, 0x54, 0x8b, 0xbf, 0x2d, 0x4e, 0xb6, 0xf2,
        0xb0, 0xf3, 0xf2, 0xf6, 0x0d, 0xab, 0x56, 0xac, 0x5d, 0x99, 0x63, 0x13,
        0x4a, 0x8c, 0x1e, 0xf1, 0x33, 0x9a, 0xd5, 0x41, 0xc7, 0x3f, 0xb8, 0x70,
        0x95, 0xb5, 0x1e, 0xeb, 0xb8, 0xd7, 0x2d, 0x09, 0x12, 0x34, 0x84, 0xc4,
        0xc4, 0x4c, 0x9d, 0xfe, 0xce, 0x72, 0x30, 0x82, 0x5a, 0x01, 0x93, 0xd6,
        0x3e, 0x51, 0xef, 0xf6, 0x3c, 0x01, 0x5c, 0x2d, 0xa8, 0xbe, 0x67, 0x41,
        0x0f, 0x52, 0x6d, 0xfc, 0x8c, 0x45, 0xd4, 0x3f, 0xae, 0x27, 0x89, 0x37,
        0x3c, 0x08, 0xdc, 0xd9, 0xe2, 0x7c, 0xff, 0x0b, 0xb7, 0x53, 0xb5, 0xc4,
        0xe0, 0x1d, 0x67, 0x8a, 0x4c, 0x7e, 0x5c, 0xaf, 0x41, 0xe4, 0xf0, 0x96,
        0x3e, 0xc4, 0x12, 0xde, 0xad, 0x7a, 0xb9, 0x67, 0x2c, 0x76, 0xdc, 0xfd,
        0x65, 0x51, 0x61, 0xc5, 0x6f, 0xc1, 0xb6, 0x2d, 0x4e, 0x72, 0x51, 0xa1,
        0x9f, 0xd4, 0xba, 0x3e, 0x9d, 0xea, 0xb0, 0xdc, 0xdd, 0xf4, 0x40, 0x80,
        0x82, 0x83, 0x4a, 0x2f, 0x23, 0x5d, 0xa7, 0xa0, 0xa0, 0xf5, 0x96, 0x38,
        0x48, 0xd4, 0x12, 0x4d, 0xe1, 0x99, 0xdf, 0x79, 0xcb, 0xef, 0x5d, 0xf4,
        0xaf, 0x02, 0x03, 0x01, 0x00, 0x01
    };

    const BYTE LWIDIA_Subordinate_CA_2019_Prod_SHA2[] = 
    {
        0x30, 0x82, 0x01, 0x0a, 0x02, 0x82, 0x01, 0x01, 0x00, 0xcf, 0x8c, 0x8d,
        0xae, 0xa1, 0x57, 0x9f, 0x14, 0x60, 0x86, 0xae, 0xd1, 0x71, 0x59, 0x1b,
        0x05, 0xfa, 0x24, 0x10, 0x45, 0xdf, 0xc9, 0x0b, 0x15, 0x94, 0x62, 0xad,
        0x20, 0xe4, 0xc4, 0xbf, 0x0f, 0x52, 0xc8, 0x9c, 0xdf, 0xae, 0xd6, 0xa9,
        0x07, 0x21, 0x98, 0xa9, 0x2f, 0x84, 0x55, 0xdd, 0x89, 0x89, 0x84, 0x22,
        0x89, 0x87, 0xd0, 0x0a, 0xcf, 0xf3, 0x2f, 0xad, 0x40, 0x80, 0x10, 0x69,
        0x24, 0xbb, 0x28, 0x47, 0x73, 0xa7, 0xa6, 0x96, 0x67, 0x39, 0x22, 0xe9,
        0x0c, 0x2c, 0x59, 0x88, 0x30, 0x86, 0xa3, 0xae, 0x8c, 0xc9, 0xd8, 0xd8,
        0xa2, 0xf9, 0xfa, 0x72, 0x3a, 0x36, 0x55, 0x73, 0xda, 0xde, 0xed, 0xae,
        0x43, 0x0f, 0xf8, 0xe7, 0xc4, 0x62, 0x48, 0xe9, 0x0a, 0x24, 0x0e, 0xc0,
        0x79, 0xb8, 0xf9, 0xc4, 0x9f, 0xb1, 0xf0, 0x52, 0x9d, 0xe4, 0x1b, 0x9f,
        0xa1, 0xa8, 0xd4, 0x67, 0xd5, 0xd6, 0x11, 0x73, 0xd8, 0xeb, 0xff, 0xa2,
        0xc0, 0xbc, 0x04, 0x6b, 0xdd, 0x46, 0x79, 0x18, 0x9c, 0xf0, 0xfd, 0xb0,
        0x3d, 0x0d, 0x28, 0x51, 0x05, 0xa5, 0x48, 0x9c, 0x81, 0x70, 0x85, 0xba,
        0x3e, 0x7f, 0x43, 0x50, 0xc1, 0x3d, 0xf1, 0x4f, 0xfa, 0xf3, 0x65, 0xcd,
        0x17, 0xbc, 0x24, 0x04, 0x51, 0x23, 0x00, 0x31, 0xd1, 0x63, 0xa3, 0x96,
        0xd5, 0x54, 0x61, 0x20, 0x47, 0x31, 0x61, 0xbf, 0xcf, 0xdd, 0xf5, 0xf8,
        0x23, 0x5d, 0x87, 0xe4, 0xe3, 0xbb, 0xe1, 0x2b, 0x30, 0x85, 0x5f, 0xd8,
        0x99, 0x16, 0xe6, 0xb0, 0x9e, 0x7a, 0xa2, 0xd0, 0x9a, 0x24, 0xe7, 0x89,
        0xee, 0xda, 0xde, 0x05, 0xd9, 0xf8, 0xc2, 0x7a, 0x0f, 0x84, 0x5f, 0xe5,
        0xb2, 0x75, 0x6c, 0x73, 0xb5, 0xf9, 0x52, 0x03, 0x48, 0x32, 0x85, 0x9e,
        0x07, 0xeb, 0x40, 0x19, 0x8e, 0x0f, 0x3d, 0x41, 0xd7, 0x54, 0xc0, 0xd2,
        0xe1, 0x02, 0x03, 0x01, 0x00, 0x01
    };

    // All Public keys are same size.
    if (pPublicKey->cbData != sizeof(LWIDIA_Subordinate_CA_2016_v2_PublicKey)) {
        return FALSE;
    }
    else if (pPublicKey->lwnusedBits != 0) {
        return FALSE;
    }

    if (!IsWindowsBuildOrGreater(eWindows8Build))
    {
        //
        // BUG 200587860 - This non-LW subordinate's root is not installed on win7
        // machines in some cases. Temporary WAR until we resolve this with MSFT
        // is to hardcode the public key and treat it as trusted even if the OS
        // functions return untrusted root
        //
        const BYTE Symantec_Class_3_SHA256_Code_Signing[] =
        {
            0x30, 0x82, 0x01, 0x0a, 0x02, 0x82, 0x01, 0x01, 0x00, 0xd7, 0x95, 0x43,
            0xd4, 0xdc, 0xdf, 0x67, 0xae, 0x39, 0xfb, 0x52, 0xa4, 0xb6, 0x26, 0x31,
            0x04, 0x70, 0xe9, 0xb7, 0x8e, 0x5b, 0x2a, 0xba, 0x37, 0x69, 0x35, 0x95,
            0x8f, 0xbb, 0xc0, 0x30, 0xe7, 0x86, 0xd8, 0x73, 0xbb, 0xdf, 0xeb, 0xd1,
            0x76, 0x3f, 0x8a, 0x56, 0x8e, 0xeb, 0x2d, 0x4b, 0xf0, 0x57, 0x18, 0x4e,
            0xb1, 0x8d, 0xa5, 0x33, 0xd3, 0x0b, 0x75, 0x23, 0xd5, 0x6a, 0x79, 0x27,
            0xdd, 0xa3, 0xd3, 0xf7, 0x0e, 0x87, 0x65, 0xb5, 0xde, 0xad, 0x1c, 0xf1,
            0xf5, 0x35, 0xb4, 0x22, 0x51, 0xaf, 0x22, 0xa1, 0xc1, 0x5d, 0x4b, 0x90,
            0x7f, 0xc0, 0x59, 0x4e, 0xab, 0x9d, 0x79, 0xa9, 0x02, 0xd7, 0x1e, 0x49,
            0xb1, 0x3b, 0x4d, 0x87, 0xfe, 0xc2, 0x78, 0xab, 0xbf, 0xef, 0x52, 0xae,
            0x9c, 0xaf, 0x08, 0xd9, 0x39, 0xe5, 0x9a, 0x51, 0x3e, 0x69, 0x5f, 0x30,
            0x10, 0x4e, 0x71, 0x63, 0x6c, 0x58, 0xe1, 0xf0, 0x20, 0x33, 0x1b, 0x0f,
            0x74, 0xbe, 0x5b, 0xcb, 0x12, 0xe1, 0xdb, 0x8c, 0xc4, 0x80, 0x94, 0x72,
            0xbb, 0xf6, 0x45, 0x9a, 0x9e, 0xc1, 0x25, 0x0b, 0xfb, 0xa3, 0x1c, 0x9e,
            0xd8, 0xa6, 0x09, 0x70, 0x71, 0xec, 0xc0, 0x47, 0x4c, 0x8f, 0x7d, 0xc3,
            0xde, 0x19, 0xce, 0x3e, 0xee, 0x04, 0x72, 0x8e, 0x17, 0xd3, 0xff, 0xf0,
            0xb2, 0x05, 0x32, 0x19, 0x4c, 0xd3, 0xb0, 0x1c, 0x9f, 0xfe, 0xa5, 0x78,
            0x20, 0x42, 0x70, 0x41, 0xb0, 0x26, 0x8c, 0x6f, 0x00, 0xc8, 0x3a, 0x00,
            0x11, 0x41, 0x7a, 0x41, 0xb0, 0xa7, 0x8a, 0x91, 0x04, 0xa0, 0x99, 0x78,
            0xf4, 0x77, 0xb4, 0xf3, 0x60, 0x2a, 0xe6, 0x6c, 0x50, 0x04, 0xb7, 0x14,
            0x60, 0xff, 0x0d, 0x51, 0xf4, 0xa8, 0x8b, 0x84, 0xfe, 0x21, 0xda, 0x5c,
            0x5e, 0x7d, 0xf5, 0x29, 0x79, 0x4f, 0xb0, 0x44, 0x74, 0x6f, 0x6d, 0x72,
            0xff, 0x02, 0x03, 0x01, 0x00, 0x01
        };
        if (!memcmp(pPublicKey->pbData, Symantec_Class_3_SHA256_Code_Signing, pPublicKey->cbData))
            return TRUE;
    }

    return !memcmp(pPublicKey->pbData, LWIDIA_Subordinate_CA_2016_v2_PublicKey, pPublicKey->cbData) ||
           !memcmp(pPublicKey->pbData, LWIDIA_Subordinate_CA_2018_Prod_SHA1, pPublicKey->cbData) ||
           !memcmp(pPublicKey->pbData, LWIDIA_Subordinate_CA_2018_Prod_SHA2, pPublicKey->cbData) ||
           !memcmp(pPublicKey->pbData, LWIDIA_Subordinate_CA_2019_Prod_SHA1, pPublicKey->cbData) ||
           !memcmp(pPublicKey->pbData, LWIDIA_Subordinate_CA_2019_Prod_SHA2, pPublicKey->cbData);
}

static BOOL IsCryptSvcPausedOrRunning()
{
    SC_HANDLE hSCManager = NULL;
    SC_HANDLE hService = NULL;
    SERVICE_STATUS serviceStatus = { 0 };
    BOOL bIsCryptSvcPausedOrRunning = FALSE;

    if (!GetModule(L"Advapi32.dll", hModAdvapi32) ||
        !GetProc(hModAdvapi32, "OpenSCManagerW", pfnOpenSCManagerW) ||
        !GetProc(hModAdvapi32, "OpenServiceW", pfnOpenServiceW) ||
        !GetProc(hModAdvapi32, "QueryServiceStatus", pfnQueryServiceStatus) ||
        !GetProc(hModAdvapi32, "CloseServiceHandle", pfnCloseServiceHandle))
    {
        return FALSE;
    }

    do
    {
        // Open service manager for read access
        hSCManager = pfnOpenSCManagerW(NULL, NULL, GENERIC_READ);
        if (NULL == hSCManager)
        {
            break;
        }

        // Open CryptoGraphic Service
        hService = pfnOpenServiceW(hSCManager, L"CryptSvc", SERVICE_QUERY_STATUS);
        if (NULL == hService)
        {
            break;
        }

        // Query service status
        if (FALSE == pfnQueryServiceStatus(hService, &serviceStatus))
        {
            break;
        }

        // Check if the service has paused or running
        switch (serviceStatus.dwLwrrentState)
        {
        case SERVICE_CONTINUE_PENDING:
        case SERVICE_PAUSE_PENDING:
        case SERVICE_PAUSED:
        case SERVICE_RUNNING:
            bIsCryptSvcPausedOrRunning = TRUE;
            break;
        }

    } while (0);

    if (NULL != hService)
    {
        pfnCloseServiceHandle(hService);
    }

    if (NULL != hSCManager)
    {
        pfnCloseServiceHandle(hSCManager);
    }

    return bIsCryptSvcPausedOrRunning;
}

//
// Bug 2665357: The following libs are loaded by the internal functions and can
// be planted. So we preload them here before we ever call the function.
// However, there's special cases we need to handle:
// 1) cryptbase.dll is not available on Wine or some older Windows systems,
//    so failure to load it is not critical.
// 2) cryptnet.dll cannot be loaded by a program in the Protected Media Path.
//    In these cases the module is found, but rejected. There can be multiple
//    reasons (i.e. error codes) why this happens, but we can't easily check that
//    since running in this path requires special signing and and kind of
//    instrumentation is disabled.
//    It's safe to ignore planting attacks if we're exelwting in PMP as the OS
//    will reject the attacker's DLL as well.
//
static BOOL _preloadCryptDlls()
{
    BOOL ret = GetModule(L"cryptnet.dll", hModCryptNet);
    if (!ret && GetLastError() != ERROR_MOD_NOT_FOUND)
    {
        ret = TRUE;
    }
    GetModule(L"cryptbase.dll", hModCryptBase);
    return ret;
}
BOOL lwVerifyFileSignatureW(LPCWSTR fileName, BOOL * pSignedByLwidia)
{
#if LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
    return TRUE;
#else
    HANDLE hFile = NULL;
    DWORD dwEncoding, dwContentType, dwFormatType;
    HCERTSTORE hStore = NULL;
    HCRYPTMSG hMsg = NULL; 
    PCMSG_SIGNER_INFO pSignerInfo = NULL;
    DWORD dwSignerInfo;
    CERT_INFO CertInfo;     
    PCCERT_CONTEXT pCertContext = NULL;
    BOOL bResult = FALSE;
    DWORD dwError = ERROR_SUCCESS;
    FILETIME signingtime;

    // Clear last error.
    SetLastError(ERROR_SUCCESS);

    // Check if file exists.
    if (!FileExists(fileName)) {
        dwError = ERROR_FILE_NOT_FOUND;
        goto verifyFileSignatureDone;
    }

    if (!GetModule(L"crypt32.dll", hModCrypt32) ||
        !GetProc(hModCrypt32, "CryptMsgClose", pfnCryptMsgClose) ||
        !GetProc(hModCrypt32, "CertCloseStore", pfnCertCloseStore) ||
        !GetProc(hModCrypt32, "CertFreeCertificateContext", pfnCertFreeCertificateContext) ||
        !GetProc(hModCrypt32, "CertFindCertificateInStore", pfnCertFindCertificateInStore) ||
        !GetProc(hModCrypt32, "CryptMsgGetParam", pfnCryptMsgGetParam) ||
        !GetProc(hModCrypt32, "CryptQueryObject", pfnCryptQueryObject) ||
        !GetProc(hModCrypt32, "CertGetNameStringA", pfnCertGetNameStringA) ||
        !GetProc(hModCrypt32, "CryptDecodeObjectEx", pfnCryptDecodeObjectEx) ||
        !GetProc(hModCrypt32, "CertGetIssuerCertificateFromStore", pfnCertGetIssuerCertificateFromStore) ||
        !_preloadCryptDlls())
    {
        dwError = ERROR_MOD_NOT_FOUND;
        goto verifyFileSignatureDone;
    }

    // Get message handle and store handle from the signed file.
    bResult = pfnCryptQueryObject(CERT_QUERY_OBJECT_FILE,
                                  fileName,
                                  CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED,
                                  CERT_QUERY_FORMAT_FLAG_BINARY,
                                  0,
                                  &dwEncoding,
                                  &dwContentType,
                                  &dwFormatType,
                                  &hStore,
                                  &hMsg,
                                  NULL);
    if (!bResult) {
        dwError = GetLastError();
        goto verifyFileSignatureDone;
    }

    // Get signer information size.
    bResult = pfnCryptMsgGetParam(hMsg, 
                                  CMSG_SIGNER_INFO_PARAM, 
                                  0, 
                                  NULL, 
                                  &dwSignerInfo);
    if (!bResult) {
        dwError = GetLastError();
        goto verifyFileSignatureDone;
    }

    // Allocate memory for signer information.
    pSignerInfo = (PCMSG_SIGNER_INFO)LocalAlloc(LPTR, dwSignerInfo);
    if (!pSignerInfo) {
        dwError = GetLastError();
        bResult = FALSE;
        goto verifyFileSignatureDone;
    }

    // Get Signer Information.
    bResult = pfnCryptMsgGetParam(hMsg, 
                                 CMSG_SIGNER_INFO_PARAM, 
                                 0, 
                                 (PVOID)pSignerInfo, 
                                 &dwSignerInfo);
    if (!bResult) {
        dwError = GetLastError();
        goto verifyFileSignatureDone;
    }

    // Search for the signer certificate in the temporary certificate store.
    CertInfo.Issuer = pSignerInfo->Issuer;
    CertInfo.SerialNumber = pSignerInfo->SerialNumber;
    pCertContext = pfnCertFindCertificateInStore(hStore,
                                                ENCODING,
                                                0,
                                                CERT_FIND_SUBJECT_CERT,
                                                (PVOID)&CertInfo,
                                                NULL);
    if (!pCertContext) {
        dwError = GetLastError();
        bResult = FALSE;
        goto verifyFileSignatureDone;
    }

    // Optionally check for LWPU signature.
    if (NULL != pSignedByLwidia) {
        *pSignedByLwidia = IsPeLwidiaSigned(pCertContext);
        dwError = GetLastError();
        if (ERROR_SUCCESS != dwError) {
            goto verifyFileSignatureDone;
        }
    }

    if (!GetSignerInfoTimeStamp(pSignerInfo, &signingtime)) {
        // If we don't store the signing time in the main signature, assume it is always older
        memset(&signingtime, 0, sizeof(signingtime));
    }
    // Make sure the certificate is properly signed.
    // Check RSA counter-signatures first, and RFC3161 if no RSA found
    bResult = VerifyTimeStampSignerInfo(pSignerInfo, hStore, &signingtime) ||
              VerifyTimeStampRFC3161(pSignerInfo, &signingtime);

    if (!bResult) {
        dwError = GetLastError();
        if (ERROR_SUCCESS == dwError)
            dwError = (DWORD)TRUST_E_TIME_STAMP;
        goto verifyFileSignatureDone;
    }

    // Bug 200410089: On win7, signature check can take a long time during bootup if CryptSvc is not running.
    // So if the DLL lives in a trusted location (i.e. need admin priv to write to it),
    // skip loading wintrust.dll for the signature verification.
    if (!IsWindowsBuildOrGreater(eWindows8Build) &&
         IsTrustedLocation(fileName) &&
         !IsCryptSvcPausedOrRunning()) {
        bResult = TRUE;
        dwError = ERROR_SUCCESS;
        goto verifyFileSignatureDone;
    }

    if (!GetModule(L"wintrust.dll", hModWinTrust) ||
        !GetProc(hModWinTrust, "WilwerifyTrust", pfnWilwerifyTrust))
    {
        bResult = FALSE;
        dwError = ERROR_MOD_NOT_FOUND;
        goto verifyFileSignatureDone;
    }
    else {
        DWORD i;
        DWORD count = GetSignatureCount(fileName);
        bResult = FALSE;
        for (i = 0; i < count; i++) {
            LONG lResult = VerifySingleSignature(fileName, i);
            if (lResult == ERROR_SUCCESS) {
                bResult = TRUE;
                break;
            }
            dwError = (DWORD)lResult;
        }

        //
        // "LWPU Subordinate CA 2016 v2" is issued by "Microsoft 
        // Digital Media Authority 2005", which is not a regular root CA
        // on most systems, but rather embedded in the OS itself. So,
        // WilwerifyTrust will not find and we have to check manually.
        //
        // BUG 200587860: One of our chains is not trusted by default on win7
        //      piggy-back on this logic and allowlist it too, even though it's
        //      not LW owned. This is a temporary WAR.
        //
        if (count == 1 && dwError == CERT_E_CHAINING) {
            DWORD dwFlags = 0;
            PCCERT_CONTEXT pIssuer = NULL;

            pIssuer = pfnCertGetIssuerCertificateFromStore(hStore, pCertContext, NULL, &dwFlags);
            if (NULL == pIssuer)
                goto verifyFileSignatureDone;

            if (IsLwidiaSubordinatePublicKey(&pIssuer->pCertInfo->SubjectPublicKeyInfo.PublicKey))
            {
                LWSLL_DBG(("Overriding WilwerifyTrust result from hardcoded public keys\n"));
                bResult = TRUE;
                dwError = ERROR_SUCCESS;
            }
            pfnCertFreeCertificateContext(pIssuer);
        }

        if (!bResult)
            goto verifyFileSignatureDone;
    }

verifyFileSignatureDone:

    // Clean up.
    SafeLocalFree(pSignerInfo);
    if (pCertContext != NULL) {
        pfnCertFreeCertificateContext(pCertContext);
    }
    if (hStore != NULL) {
        pfnCertCloseStore(hStore, 0);
    }
    if (hMsg != NULL) {
        pfnCryptMsgClose(hMsg);
    }
    // Refresh last error so users can fetch it
    SetLastError(dwError);
    // Done.
    return bResult;
#endif
}

BOOL lwVerifyFileSignatureA(LPCSTR fileName, BOOL * pSignedByLwidia)
{
#if LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
    return TRUE;
#else
    // Simply use UNICODE version.
    BOOL bResult = FALSE;
    LPWSTR unicodeFileName = CreateUnicodeStringFromAscii(fileName);
    if (!fileName || unicodeFileName) {
        bResult = lwVerifyFileSignatureW(unicodeFileName, pSignedByLwidia);
    }
    SafeLocalFree(unicodeFileName);
    return bResult;
#endif
}

void lwReleaseSelwreLoadLibraryResources()
{
#if !LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
    // Release SetupApi.dll resources
    if (NULL != hModSetupapi) {
        pfnSetupDiGetClassDevsW = NULL;
        pfnSetupDiEnumDeviceInterfaces = NULL;
        pfnSetupDiGetDeviceInterfaceDetailW = NULL;
        pfnSetupDiGetDeviceRegistryPropertyW = NULL;
        pfnSetupDiGetDevicePropertyW = NULL;
        pfnSetupGetInfDriverStoreLocationW = NULL;
        pfnSetupDiDestroyDeviceInfoList = NULL;
        FreeLibrary(hModSetupapi);
        hModSetupapi = NULL;
    }

    // Release Shell32.dll resources
    if (NULL != hModShell32) {
        pfnSHGetFolderPath = NULL;
        pfnShellExelwteA = NULL;
        pfnShellExelwteW = NULL;
        pfnShellExelwteExA = NULL;
        pfnShellExelwteExW = NULL;
        FreeLibrary(hModShell32);
        hModShell32 = NULL;
    }

    // Release Advapi32.dll resources
    if (NULL != hModAdvapi32) {
        pfnRegOpenKeyExW = NULL;
        pfnRegEnumValueW = NULL;
        pfnRegQueryValueExW = NULL;
        pfnRegCloseKey = NULL;
        pfnCreateProcessAsUserW = NULL;
        pfnCreateProcessAsUserA = NULL;
        pfnOpenSCManagerW = NULL;
        pfnOpenServiceW = NULL;
        pfnQueryServiceConfigW = NULL;
        pfnCloseServiceHandle = NULL;
        FreeLibrary(hModAdvapi32);
        hModAdvapi32 = NULL;
    }

    // Release Ole32.dll resources
    if (NULL != hModOle32) {
        pfnStringFromGUID2 = NULL;
        FreeLibrary(hModOle32);
        hModOle32 = NULL;
    }

    // Release crypt32.dll resources
    if (NULL != hModCrypt32) {
        pfnCryptMsgClose = NULL;
        pfnCertCloseStore = NULL;
        pfnCertFreeCertificateContext = NULL;
        pfnCertFindCertificateInStore = NULL;
        pfnCryptMsgGetParam = NULL;
        pfnCryptQueryObject = NULL;
        pfnCertGetNameStringA = NULL;
        pfnCryptDecodeObjectEx = NULL;
        pfnCertGetIssuerCertificateFromStore = NULL;
        FreeLibrary(hModCrypt32);
        hModCrypt32 = NULL;
    }
    // Release wintrust.dll resources
    if (NULL != hModWinTrust) {
        pfnWilwerifyTrust = NULL;
        FreeLibrary(hModWinTrust);
        hModWinTrust = NULL;
    }

    // Release gdi32.dll resources
    if (NULL != hModGDI32) {
        pfnD3DKMTEnumAdapters2 = NULL;
        pfnD3DKMTQueryAdapterInfo = NULL;
        FreeLibrary(hModGDI32);
        hModGDI32 = NULL;
    }

    // Release libraries that were preloaded for security
    if (NULL != hModCryptBase) {
        FreeLibrary(hModCryptBase);
        hModCryptBase = NULL;
    }
    if (NULL != hModCryptNet) {
        FreeLibrary(hModCryptNet);
        hModCryptNet = NULL;
    }
#endif
}

static BOOL TestCertificateChain(PCCERT_CONTEXT pCertContext, const LPCSTR chain[])
{
    PCCERT_CONTEXT pLwrrentCert = pCertContext, pParentCert = NULL;
    BOOL bRootCheck = FALSE;
    BOOL bMatch = FALSE;
    UINT i;

    for (i = 0; chain[i] != NULL; i++)
    {
        DWORD dwFlags = 0;
        char szName[256];

        // The function will set szName to "" if the name is not found, but
        // static code analysis does not know that, so terminate manually.
        szName[0] = 0;
        pfnCertGetNameStringA(pLwrrentCert,
                              CERT_NAME_SIMPLE_DISPLAY_TYPE,
                              bRootCheck ? CERT_NAME_ISSUER_FLAG : 0,
                              NULL,
                              szName,
                              sizeof(szName));

        if (strcmp(szName, chain[i]) != 0)
        {
            bMatch = FALSE;
            break;
        }

        // Was that the last check?
        if (bRootCheck)
        {
            // Only report match if there's nothing more expected either
            bMatch = (chain[i+1] == NULL);
            break;
        }

        // Get next cert in chain. This will free pParentCert
        pParentCert = pfnCertGetIssuerCertificateFromStore(pCertContext->hCertStore,
                                                           pLwrrentCert,
                                                           pParentCert,
                                                           &dwFlags);
        if (pParentCert)
        {
            pLwrrentCert = pParentCert;
        }
        else
        {
            if (chain[i+1] == NULL)
            {
                bMatch = TRUE;
                break;
            }
            // No cert found. Use the issuer of current cert instead
            bRootCheck = TRUE;
        }
    }

    if ((pLwrrentCert != pCertContext) && !bRootCheck)
    {
        //
        // If we're still on the starting context, don't free it yet.
        // If we've hit the root check, we've already freed it above when we
        // passed the same cert context twice to CertGetIssuerCertificateFromStore
        // Otherwise, free the context.
        //
        pfnCertFreeCertificateContext(pLwrrentCert);
    }

    if (bMatch)
        SetLastError(ERROR_SUCCESS);

    return bMatch;
}

static BOOL IsPeLwidiaSigned(PCCERT_CONTEXT pCertContext)
{
#if LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
    return TRUE;
#else
    const LPCSTR validChains[][5] =
    {
        {
            "LWPU Corporation",
            "Symantec Class 3 SHA256 Code Signing CA",
            "VeriSign Class 3 Public Primary Certification Authority - G5",
            "Microsoft Code Verification Root",
            NULL
        },
        {
            "LWPU Corporation",
            "VeriSign Class 3 Code Signing 2010 CA",
            "VeriSign Class 3 Public Primary Certification Authority - G5",
            "Microsoft Code Verification Root",
            NULL
        },
        {
            "LWPU Corporation",
            "Symantec Class 3 SHA256 Code Signing CA - G2",
            "VeriSign Universal Root Certification Authority",
            "Microsoft Code Verification Root",
            NULL
        },
        // Sometimes the VeriSign is the actual system root, so repeat those.
        {
            "LWPU Corporation",
            "Symantec Class 3 SHA256 Code Signing CA",
            "VeriSign Class 3 Public Primary Certification Authority - G5",
            NULL
        },
        {
            "LWPU Corporation",
            "VeriSign Class 3 Code Signing 2010 CA",
            "VeriSign Class 3 Public Primary Certification Authority - G5",
            NULL
        },
        {
            "LWPU Corporation",
            "Symantec Class 3 SHA256 Code Signing CA - G2",
            "VeriSign Universal Root Certification Authority",
            NULL
        },
        {
            "LWPU Corporation PE Sign v2016",
            "LWPU Subordinate CA 2016 v2",
            "Microsoft Digital Media Authority 2005",
            NULL
        },
        {
            "LWPU Corporation-PE-Prod-Sha1",
            "LWPU Subordinate CA 2018-Prod-Sha1",
            "Microsoft Digital Media Authority 2005",
            NULL
        },
        {
            "LWPU Corporation-PE-Prod-Sha2",
            "LWPU Subordinate CA 2018-Prod-Sha2",
            "Microsoft Digital Media Authority 2005",
            NULL
        },
        {
            "LWPU Corporation-PE-Prod-Sha1",
            "LWPU Subordinate CA 2019-Prod-Sha1",
            "Microsoft Digital Media Authority 2005",
            NULL
        },
        {
            "LWPU Corporation-PE-Prod-Sha2",
            "LWPU Subordinate CA 2019-Prod-Sha2",
            "Microsoft Digital Media Authority 2005",
            NULL
        },
        {
            "LWPU CORPORATION",
            "LWPU Subordinate CA",
            "Microsoft Digital Media Authority 2005",
            NULL
        },
        {NULL}
    };

    UINT i;
    for (i = 0; validChains[i][0] != NULL; i++)
    {
        if (TestCertificateChain(pCertContext, validChains[i]))
            return TRUE;
    }

    return FALSE;
#endif
}

static BOOL GetSignerInfoTimeStamp(PCMSG_SIGNER_INFO pSignerInfo, FILETIME *pFiletime)
{
#ifndef szOID_RSA_signingTime
#define szOID_RSA_signingTime   "1.2.840.113549.1.9.5"
#endif
    DWORD dwSize = sizeof(FILETIME), n;
    // Loop through authenticated attributes and find szOID_RSA_signingTime OID.
    for (n = 0; n < pSignerInfo->AuthAttrs.cAttr; n++)
    {
        if (lstrcmpA(pSignerInfo->AuthAttrs.rgAttr[n].pszObjId, szOID_RSA_signingTime))
            continue;

        return pfnCryptDecodeObjectEx(ENCODING, szOID_RSA_signingTime,
                                      pSignerInfo->AuthAttrs.rgAttr[n].rgValue[0].pbData,
                                      pSignerInfo->AuthAttrs.rgAttr[n].rgValue[0].cbData,
                                      0, NULL, (PVOID)pFiletime, &dwSize);
    }
    return FALSE;
}

static BOOL VerifyTimeStampSignerInfo(PCMSG_SIGNER_INFO pSignerInfo, HCERTSTORE hStore, FILETIME *pFiletime)
{
    BOOL bReturn = FALSE;
#if !LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
    BOOL bResult;
    DWORD dwSize;
    DWORD n;
    PCMSG_SIGNER_INFO pCounterSignerInfo = NULL;
    PCCERT_CONTEXT pCertContext = NULL;
    CERT_INFO CertInfo;
    FILETIME ft;

    // Loop through unauthenticated attributes for szOID_RSA_counterSign OID.
    for (n = 0; n < pSignerInfo->UnauthAttrs.cAttr && !bReturn; ++n)
    {
        if (lstrcmpA(pSignerInfo->UnauthAttrs.rgAttr[n].pszObjId, szOID_RSA_counterSign)) {
            continue;
        }

        bResult = pfnCryptDecodeObjectEx(ENCODING,
                                        PKCS7_SIGNER_INFO,
                                        pSignerInfo->UnauthAttrs.rgAttr[n].rgValue[0].pbData,
                                        pSignerInfo->UnauthAttrs.rgAttr[n].rgValue[0].cbData,
                                        CRYPT_DECODE_ALLOC_FLAG, 
                                        NULL,
                                        (PVOID)&pCounterSignerInfo,
                                        &dwSize);

        if (!bResult) {
            goto VerifyTimeStampSignerInfoDone;
        }

        CertInfo.Issuer       = pCounterSignerInfo->Issuer;
        CertInfo.SerialNumber = pCounterSignerInfo->SerialNumber;

        // Search for Timestamp certificate in the temporary certificate store.
        pCertContext = pfnCertFindCertificateInStore(hStore,
                                                     ENCODING,
                                                     0,
                                                     CERT_FIND_SUBJECT_CERT,
                                                     (PVOID)&CertInfo,
                                                     NULL);
        if (!pCertContext) {
            goto VerifyTimeStampSignerInfoDone;
        }

        // Now check the actual timestamp
        if (GetSignerInfoTimeStamp(pCounterSignerInfo, &ft)) {
            // Check if signing time is <= countersign time
            bReturn = CompareFileTime(pFiletime, &ft) <= 0;
        }
        pfnCertFreeCertificateContext(pCertContext);

    }

VerifyTimeStampSignerInfoDone:
#endif
    // Done.
    SafeLocalFree(pCounterSignerInfo);
    return bReturn;
}

//
// ASN.1 Parsing for the timestamps
// http://luca.ntop.org/Teaching/Appunti/asn1.html
//
static DWORD PopASN1Sequence(const PBYTE pBuffer, PBYTE pTag, PBYTE *ppSeq, PDWORD pdwSeqSize)
{
    DWORD dwPosition = 0, dwSeqSize = 0;

    *pTag = pBuffer[dwPosition++];
    dwSeqSize = pBuffer[dwPosition++];
    if (dwSeqSize & 0x80)
    {
        // Length not encoded directly, instead we get "length of length"
        int i = dwSeqSize & 0x7F;
        dwSeqSize = 0;
        while (i-- > 0)
            dwSeqSize = (dwSeqSize << 8) | pBuffer[dwPosition++];
    }

    *pdwSeqSize = dwSeqSize;
    *ppSeq = &pBuffer[dwPosition];
    return dwPosition + dwSeqSize;
}

// We cannot use standard library as ABI doesn't match between MSVCRT
// See bug #200424522 for example.
static ULONGLONG ParseASN1Timestamp(const char *pTimestamp, size_t timestampSize)
{
    // Timestamp is in format YYYYMMDDhhmmss[.sss]Z
    const char FORMAT_PREFIX[] = "YYYYMMDDhhmmss";
    size_t i;
    ULONGLONG time = 0;

    for (i = 0; i < sizeof(FORMAT_PREFIX) - 1; i++) {
        if (i >= timestampSize || pTimestamp[i] < '0' || pTimestamp[i] > '9') {
            return 0ULL;
        }
        time = time * 10 + pTimestamp[i] - '0';
    }

    // If next letter is . then skip three digits of milliseconds
    if (i >= timestampSize && pTimestamp[i] == '.') {
        const char FORMAT_MS[] = "sss";
        size_t j;
        for (i++, j = 0; j < sizeof(FORMAT_MS) - 1; i++, j++) {
            if (i >= timestampSize || pTimestamp[i] < '0' || pTimestamp[i] > '9') {
                return 0ULL;
            }
        }
    }

    // Force the UTC timestamp
    if (i >= timestampSize || pTimestamp[i] != 'Z') {
        return 0ULL;
    }

    return time;
}

static ULONGLONG ParseASN1TimestampOID(const PBYTE pBuffer, DWORD dwBufferSize, BOOL bFoundOid)
{
    // From wincrypt.h, we need it binary encoded
    //#define szOID_TIMESTAMP_TOKEN           "1.2.840.113549.1.9.16.1.4"
    const BYTE OID_TIMESAMP_TOKEN[] = { 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x09, 0x10, 0x01, 0x04 };

    const BYTE TAG_OID = 0x06;
    const BYTE TAG_OCTET_STRING = 0x04;
    const BYTE TAG_GENERALIZED_TIME = 0x18;

    DWORD dwPosition = 0;

    while (dwPosition < dwBufferSize)
    {
        PBYTE pLwrrentSeq;
        DWORD dwSeqSize;
        BYTE Tag;
        dwPosition += PopASN1Sequence(&pBuffer[dwPosition], &Tag, &pLwrrentSeq, &dwSeqSize);

        if (Tag & 0x20)
        {
            // Search subsequence
            ULONGLONG t = ParseASN1TimestampOID(pLwrrentSeq, dwSeqSize, bFoundOid);
            if (t)
                return t;
            continue;
        }
        Tag &= 0x1F;

        if (!bFoundOid)
        {
            if (Tag == TAG_OID && dwSeqSize == sizeof(OID_TIMESAMP_TOKEN) && !memcmp(pLwrrentSeq, OID_TIMESAMP_TOKEN, dwSeqSize))
            {
                dwPosition += PopASN1Sequence(&pBuffer[dwPosition], &Tag, &pLwrrentSeq, &dwSeqSize);
                if (Tag & 0x20)
                    PopASN1Sequence(pLwrrentSeq, &Tag, &pLwrrentSeq, &dwSeqSize);
                if (Tag == TAG_OCTET_STRING)
                    return ParseASN1TimestampOID(pLwrrentSeq, dwSeqSize, TRUE);
            }
        }
        else if (Tag == TAG_GENERALIZED_TIME)
        {
            return ParseASN1Timestamp((const char*)pLwrrentSeq, (size_t)dwSeqSize);
        }
    }
    return 0ULL;
}
static BOOL VerifyTimeStampRFC3161(PCMSG_SIGNER_INFO pSignerInfo, FILETIME *pFiletime)
{
    BOOL bReturn = FALSE;
#if !LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)

    BOOL bResult;
    DWORD dwSize;
    DWORD n;
    PCRYPT_CONTENT_INFO pCounterSignerInfo = NULL;
    ULONGLONG tSign, tCounterSign;
    // Loop through unauthenticated attributes for szOID_RSA_counterSign OID.
    for (n = 0; n < pSignerInfo->UnauthAttrs.cAttr; ++n)
    {
#ifndef szOID_RFC3161_counterSign
#define szOID_RFC3161_counterSign "1.3.6.1.4.1.311.3.3.1"
#endif
        if (lstrcmpA(pSignerInfo->UnauthAttrs.rgAttr[n].pszObjId, szOID_RFC3161_counterSign)) {
            continue;
        }

        bResult = pfnCryptDecodeObjectEx(ENCODING,
                                         PKCS_CONTENT_INFO,
                                         pSignerInfo->UnauthAttrs.rgAttr[n].rgValue[0].pbData,
                                         pSignerInfo->UnauthAttrs.rgAttr[n].rgValue[0].cbData,
                                         CRYPT_DECODE_ALLOC_FLAG, 
                                         NULL,
                                         (PVOID)&pCounterSignerInfo,
                                         &dwSize);
        if (bResult)
        {
            tCounterSign = ParseASN1TimestampOID(pCounterSignerInfo->Content.pbData,
                                                 pCounterSignerInfo->Content.cbData, FALSE);
            if (tCounterSign > 0)
            {
                SYSTEMTIME st;
                FileTimeToSystemTime(pFiletime, &st);
                tSign = st.wSecond * 1ull         + st.wMinute * 100ull +
                        st.wHour   * 10000ull     + st.wDay    * 1000000ull +
                        st.wMonth  * 100000000ull + st.wYear   * 10000000000ull;
                bReturn = tSign <= tCounterSign;
            }
        }
        break;
    }
#endif
    // Done.
    SafeLocalFree(pCounterSignerInfo);
    return bReturn;
}

static BOOL bSignatureOverride = LW_PE_SIGNATURE_OVERRIDE ? TRUE : FALSE;
static BOOL bCheckRegistryForSignatureOverride = TRUE;
#if defined(_DEBUG)
static BOOL bDebugBreakOnSignatureVerificationFailures = TRUE;
#endif // #if defined(_DEBUG)

static BOOL OverrideSignatureVerificationFailure(LPCWSTR fileName, DWORD verificationError)
{
    BOOL bOverrideThisSignature = bSignatureOverride;

    // Dump anxous debug message
    int i; WCHAR errorStr[] = L"0x00000000\n";
    OutputDebugStringW(L"*** WARNING - PE SIGNATURE VERIFICATION FAILED !!!\n*** Filename: ");
    OutputDebugStringW(fileName);
    OutputDebugStringW(L"\n*** Error: ");
    for (i = 0; i < 8; ++i) {
        errorStr[i+2] = (WCHAR)((verificationError >> ((7-i) << 2)) & 0xF); // fetch next four bits
        errorStr[i+2] += (errorStr[i+2] < 10) ? (L'0') : (L'A' - 10); // translate to hex
    }
    OutputDebugStringW(errorStr);

    // Optionally check registry for signature failure override
    if (!bOverrideThisSignature && bCheckRegistryForSignatureOverride)
    {
        // Check whether we should ignore this per registry:
        const WORD dwRegOpenKeyFlags[] = {KEY_WOW64_64KEY, KEY_WOW64_32KEY, 0};
        const GUID overrideRegValueGuid = LW_PE_SIGNATURE_OVERRIDE_REGVALUE;
        WCHAR overrideRegValueName[39];

// This is defined in lwSelwreLoadLibrary.h, however apparently there are DVS builds that
// only sync this file, but not the header. So mirror here until that gets sorted out.
#ifndef LW_PE_SIGNATURE_OVERRIDE_REGKEY_RS4
#define LW_PE_SIGNATURE_OVERRIDE_REGKEY_RS4 L"SYSTEM\\ControlSet001\\Services\\lwlddmkm"
#endif
        // We use a different key or RS4+ due to possible CV restrictions
        LPCWSTR szRegKey = IsWindowsBuildOrGreater(eWindows10RS4Build) ?
                            LW_PE_SIGNATURE_OVERRIDE_REGKEY_RS4 :
                            LW_PE_SIGNATURE_OVERRIDE_REGKEY;

        // Try to keep this a one-time check:
        bCheckRegistryForSignatureOverride = FALSE;

        // Dynamically load Advapi32.dll to remove .lib dependency
        if (!GetModule(L"Advapi32.dll", hModAdvapi32) ||
            !GetProc(hModAdvapi32, "RegOpenKeyExW", pfnRegOpenKeyExW) ||
            !GetProc(hModAdvapi32, "RegEnumValueW", pfnRegEnumValueW) ||
            !GetProc(hModAdvapi32, "RegCloseKey", pfnRegCloseKey))
        {
            return FALSE;
        }

        // Dynamically load Ole32.dll to remove .lib dependency
        if (NULL == hModOle32) {
            if (NULL != (hModOle32 = LoadSystemLibraryInternal(L"Ole32.dll", 0))) {
                pfnStringFromGUID2 = (PfnStringFromGUID2)GetProcAddress(hModOle32, "StringFromGUID2");
            }
        }

        // Sanity-check Ole32 functions
        if (NULL == pfnStringFromGUID2) {
            return FALSE;
        }

        // Colwert GUID LW_SIGNATURE_OVERRIDE_REGVALUE to string
        ZeroMemory(overrideRegValueName, sizeof(overrideRegValueName));
        if (_countof(overrideRegValueName) != pfnStringFromGUID2(
// see guiddef.h:
#ifdef __cplusplus
            overrideRegValueGuid, // C++ wants const GUID &
#else
            &overrideRegValueGuid, // C wants const GUID *
#endif
            overrideRegValueName,
            _countof(overrideRegValueName)))
        {
            return FALSE; // StringFromGUID2 failed...
        }

        // Check KEY_WOW64_64KEY first, then KEY_WOW64_32KEY:
        for (i = 0; !bOverrideThisSignature && dwRegOpenKeyFlags[i]; ++i)
        {
            // Open LW_SIGNATURE_OVERRIDE_REGKEY
            HKEY hKey = NULL;
            LONG lResult = pfnRegOpenKeyExW(HKEY_LOCAL_MACHINE, szRegKey, 0, dwRegOpenKeyFlags[i]|KEY_READ, &hKey);
            if (ERROR_SUCCESS == lResult) 
            {
                // Enumerate LW_PE_SIGNATURE_OVERRIDE_REGKEY's values and check
                // them for LW_PE_SIGNATURE_OVERRIDE_REGVALUE != 0 in memory:
                DWORD dwIndex = 0, dwIgnore = 0;
                do {
                    WCHAR regValueName[128];
                    DWORD regValueNameChars = _countof(regValueName);
                    DWORD regValueData, regValueDataSize = sizeof(regValueData);
                    ZeroMemory(regValueName, sizeof(regValueName));

                    lResult = pfnRegEnumValueW(
                                hKey,                  // _In_         HKEY hKey,
                                dwIndex++,             // _In_         DWORD dwIndex,
                                regValueName,          // _Out_        LPTSTR lpValueName,
                                &regValueNameChars,    // _Inout_      LPDWORD lpcchValueName,
                                NULL,                  // _Reserved_   LPDWORD lpReserved,
                                NULL,                  // _Out_opt_    LPDWORD lpType,
                                (LPBYTE)&regValueData, // _Out_opt_    LPBYTE lpData,
                                &regValueDataSize);    // _Inout_opt_  LPDWORD lpcbData)

                    if ((ERROR_SUCCESS == lResult) &&
                        (wcslen(overrideRegValueName) == regValueNameChars) &&
                        !wcscmp(overrideRegValueName, regValueName))
                    {
                        dwIgnore |= (regValueData ? 1 : 0);
                    }
                } while (ERROR_NO_MORE_ITEMS != lResult);
                pfnRegCloseKey(hKey);
                // Patch bOverrideThisSignature and bSignatureOverride according to registry:
                bOverrideThisSignature = bSignatureOverride = dwIgnore ? TRUE : FALSE;
            }
        }
    }

#if defined(_DEBUG) && (_WIN32_WINNT >= 0x0400)
    if (!bOverrideThisSignature && bDebugBreakOnSignatureVerificationFailures && IsDebuggerPresent()) {
        //
        // PE signature verification failed and we cannot ignore
        // this (check verificationError for failure reason):
        //
            __debugbreak();
        //
        // These are your options to continue:
        //
        //   1. accept this failure and just continue debugging (F5?)
        //   2. tweak bOverrideThisSignature to ignore this one failure and continue
        //   3. tweak bSignatureOverride to ignore this and all future failures in this module
        //
        // In addition, you may want to consider adding
        //
        //   \\HKLM\<LW_PE_SIGNATURE_OVERRIDE_REGKEY>\<LW_PE_SIGNATURE_OVERRIDE_REGVALUE>:DWORD!=0
        //
        // to your registry to ignore signature verification failures
        // system-wide next time (see lwSelwreLoadLibrary.h for exact
        // regkey/value name strings).
        //
    }
#endif // defined(_DEBUG) && (_WIN32_WINNT >= 0x0400)


    // Finally check if we may override this signature verification
    // failure because either (our local) bOverrideThisSignature or
    // (the module-global) bSignatureOverride have been tweaked:
    if (bOverrideThisSignature || bSignatureOverride)
    {
        // Give some feedback on override (but don't print regkey/value, etc. names!)
        OutputDebugStringW(L"*** Signature override detected...\n");
        SetLastError(verificationError = ERROR_SUCCESS);
        return TRUE;
    }

    // Fail.
    SetLastError(verificationError);
    return FALSE;
}

//
// Forward declaration of static lwSelwreLoadLibrary/RS1 functions
//

// ctrdefs.h is not available to all consumers
// so we inline its ptrdiff_t definition here

#if !defined (_W64)
#if !defined (__midl) && (defined (_X86_) || defined (_M_IX86))
#define _W64 __w64
#else  /* !defined (__midl) && (defined (_X86_) || defined (_M_IX86)) */
#define _W64
#endif  /* !defined (__midl) && (defined (_X86_) || defined (_M_IX86)) */
#endif  /* !defined (_W64) */

#ifndef _PTRDIFF_T_DEFINED
#ifdef _WIN64
typedef __int64             ptrdiff_t;
#else  /* _WIN64 */
typedef _W64 int            ptrdiff_t;
#endif  /* _WIN64 */
#define _PTRDIFF_T_DEFINED
#endif  /* _PTRDIFF_T_DEFINED */

static LPCWSTR FindPathTail(const LPCWSTR path);
static LPCWSTR FindDotExt(const LPCWSTR path);
static BOOL IsPathToModule(const LPCWSTR path, const LPCWSTR moduleName);
static LPWSTR SafePathCombine(LPCWSTR pathHead, LPCWSTR pathTail);
static LPWSTR SafeStringDupe(const LPCWSTR str);
static LPWSTR SafeStringDupeN(const LPCWSTR str, const ptrdiff_t length);
static BOOL FindLwidiaDevice(const LPCGUID lpClassGuid, HDEVINFO * phDevInfo, SP_DEVINFO_DATA * pDevInfoData);
static LPWSTR ComplementModuleName(const LPCWSTR moduleName);

static LPWSTR DetectModulePathFromKernelModeServices(const LPCWSTR pszModuleName, const LPCGUID lpClassGuid);
static LPWSTR DetectModulePathFromRegistryValues(const LPCWSTR pszModuleName, const HKEY hKey, const LPCWSTR pszSubKey, const LPCWSTR * ppszValueNames);
static LPWSTR DetectModulePathFromRegistryValueNames(const LPCWSTR pszModuleName, const HKEY hKey, const LPCWSTR * ppszSubKeys);
static LPWSTR DetectModuleInLocation(const LPCWSTR pszModuleName, const LPCWSTR pszLocation);

//
// Implementation of non-static lwSelwreLoadLibrary/RS1 functions
//

HMODULE lwLoadDisplayDriverModuleExW(LPCWSTR fileName, DWORD dwFlags)
{
    LPWSTR pModuleLocation = NULL;
    HMODULE hResult = NULL;

    SetLastError(ERROR_SUCCESS);

    // Validate arguments

    if ((NULL == fileName) || ContainsPathSeparator(fileName))
    {
        SetLastError(ERROR_BAD_ARGUMENTS);
        return NULL;
    }

    // Detect display driver module location

    pModuleLocation = lwDetectDisplayDriverModuleLocationW(fileName);
    if (NULL != pModuleLocation)
    {
        // Load display driver module from detected location

        hResult = lwLoadLibraryFromTrustedLocationExW(pModuleLocation, dwFlags);
        SafeLocalFree(pModuleLocation);
    }

    return hResult;
}

HMODULE lwLoadDisplayDriverModuleExA(LPCSTR fileName, DWORD dwFlags)
{
    // Simply use UNICODE version

    HMODULE hResult = NULL;
    LPWSTR unicodeFileName = CreateUnicodeStringFromAscii(fileName);
    if (!fileName || unicodeFileName)
    {
        hResult = lwLoadDisplayDriverModuleExW(unicodeFileName, dwFlags);
    }
    SafeLocalFree(unicodeFileName);
    return hResult;
}

static const GUID LW_GUID_DISPLAY_DEVICE_ARRIVAL = {0x1CA05180, 0xA699, 0x450A, {0x9A, 0x0C, 0xDE, 0x4F, 0xBE, 0x3D, 0xDD, 0x89}};

LPWSTR lwDetectDisplayDriverModuleRegistrationW(LPCWSTR moduleName)
{
    //
    // Complement provided module name according to LoadLibrary
    // behavior and try to detect module registration by checking for
    //
    // 1. Kernel mode driver service binary paths
    //
    // 2. Display driver regkeys used to register UMD modules
    //    with the OS
    //
    // 3. Khronos regkeys used to register OpenCL ICD and
    //    Vulkan loader JSON with Khronos loaders
    //
    // 4. "DriverSupportModules[Wow]" display driver regkey
    //    for driver support modules with special locations
    //
    // to report module registration value on success, NULL otherwise.
    //

    BOOL bIsWow64 = FALSE;
    DWORD dwError = ERROR_SUCCESS;
    LPWSTR pszDriverRegKey = NULL;
    LPWSTR pszModuleName = NULL;
    LPWSTR pszResult = NULL;

    SetLastError(ERROR_SUCCESS);

    // Validate arguments

    if ((NULL == moduleName) || ContainsPathSeparator(moduleName))
    {
        SetLastError(ERROR_BAD_ARGUMENTS);
        return NULL;
    }

    do
    {
        LPCWSTR pszModuleNameDotExt = NULL;

        // Check Wow64

        if (FALSE == DetectWow64Process(&bIsWow64))
        {
            break;
        }

        // If we are on RS4+, use QAI to get the data first, as this does not
        // touch the registry and will work for CV.
        if (IsWindowsBuildOrGreater(eWindows10RS4Build))
        {
            LPCWSTR pszUmdPath;
            // IoTEdge OS SKU special case:
            // We don't have access to the gdi32.dll functions and can't call QAI or similar.
            // Fortunately, we only need to support a few UMDs on this SKU, and those are
            // already loaded when we get here, so just use their absolute path instead.
            //
            // Try the IoTEdge case first, fall back to QAI if not found.
            // UMD path is returned in a static buffer, should not be freed.
            pszUmdPath = GetUMDPathFromLoadedDLLs();
            if (pszUmdPath[0] == 0)
                pszUmdPath = GetUMDPathFromQAI();

            pszResult = SafePathCombine(pszUmdPath, moduleName);
            if (FileExists(pszResult))
            {
                break;
            }
            else
            {
                SafeLocalFree(pszResult);
                pszResult = NULL;
            }
        }

        // Get display driver regkey

        if (NULL == (pszDriverRegKey = lwGetDisplayDriverRegkeyW()))
        {
            break;
        }

        // Complement module name according to according to LoadLibrary behavior

        if (NULL == (pszModuleName = ComplementModuleName(moduleName)))
        {
            break;
        }

        pszModuleNameDotExt = FindDotExt(pszModuleName);

        // Optimization: skip step 1 for DLLs

        if (0 != _wcsicmp(pszModuleNameDotExt, L".dll"))
        {
            if ((NULL == pszResult) && (ERROR_SUCCESS == (dwError = GetLastError())))
            {
                //
                // 1. Try to detect module registration from
                //    kernel mode driver service binary paths
                //

                pszResult = DetectModulePathFromKernelModeServices(pszModuleName, &LW_GUID_DISPLAY_DEVICE_ARRIVAL);
            }
        }

        // Optimization: skip steps 2 and 3 for SYS files

        if (0 != _wcsicmp(pszModuleNameDotExt, L".sys"))
        {
            if ((NULL == pszResult) && (ERROR_SUCCESS == (dwError = GetLastError())))
            {
                //
                // 2. Try to detect module registration from display
                //    driver regkeys that are used to register UMD
                //    modules with the OS
                //

                const LPCWSTR pszDriverRegValuesWow[] =
                {
                    L"UserModeDriverNameWow",
                    L"OpenGLDriverNameWow",
                    L"UserModeDListDriverNameWow",
                    NULL
                };

                const LPCWSTR pszDriverRegValues[] =
                {
                    L"UserModeDriverName",
                    L"OpenGLDriverName",
                    L"UserModeDListDriverName",
                    NULL
                };

                const LPCWSTR * ppszDriverRegValues = bIsWow64
                    ? pszDriverRegValuesWow
                    : pszDriverRegValues;

                pszResult = DetectModulePathFromRegistryValues(pszModuleName, HKEY_LOCAL_MACHINE, pszDriverRegKey, ppszDriverRegValues);
            }

            if ((NULL == pszResult) && (ERROR_SUCCESS == (dwError = GetLastError())))
            {
                //
                // 3. Try to detect module registration from Khronos
                //    regkeys that are used to register OpenCL ICD and
                //    Vulkan loader JSON with Khronos loaders; note that
                //    for Vulkan, this would return just the location of
                //    the JSON file, not the location of the actual ICD
                //

                const LPCWSTR pszKhronosRegKeys[] =
                {
                    L"SOFTWARE\\Khronos\\Vulkan\\Drivers",
                    L"SOFTWARE\\Khronos\\OpenCL\\Vendors",
                    NULL
                };

                pszResult = DetectModulePathFromRegistryValueNames(pszModuleName, HKEY_LOCAL_MACHINE, pszKhronosRegKeys);
            }
        }

        if ((NULL == pszResult) && (ERROR_SUCCESS == (dwError = GetLastError())))
        {
            //
            // 4. Try to detect module registration from "DriverSupportModules[Wow]"
            //    display driver regkey for driver support modules with special
            //    locations
            //

            const LPCWSTR pszDriverRegValuesWow[] =
            {
                L"DriverSupportModulesWow",
                NULL
            };

            const LPCWSTR pszDriverRegValues[] =
            {
                L"DriverSupportModules",
                NULL
            };

            const LPCWSTR * ppszDriverRegValues = bIsWow64
                ? pszDriverRegValuesWow
                : pszDriverRegValues;

            pszResult = DetectModulePathFromRegistryValues(pszModuleName, HKEY_LOCAL_MACHINE, pszDriverRegKey, ppszDriverRegValues);
        }
    }
    while (0);

    SafeLocalFree(pszDriverRegKey);
    SafeLocalFree(pszModuleName);

    if (NULL != pszResult)
    {
        SetLastError(ERROR_SUCCESS);
    }
    else if (ERROR_SUCCESS == (dwError = GetLastError()))
    {
        SetLastError(ERROR_MOD_NOT_FOUND);
    }

    return pszResult;

}

// From d3dkmthk.h from wddm2, not available in earlier SDKs.
typedef struct _LW_D3DKMT_QUERY_DEVICE_IDS
{
    UINT PhysicalAdapterIndex; // IN
    struct _LW_D3DKMT_DEVICE_IDS
    {
        UINT VendorID;
        UINT DeviceID;
        UINT SubVendorID;
        UINT SubSystemID;
        UINT RevisionID;
        UINT BusType;
    } DeviceIds; // OUT
} LW_D3DKMT_QUERY_DEVICE_IDS;

typedef struct _LW_D3DKMT_UMDFILENAMEINFO
{
    INT        Version;                // In: UMD version
    WCHAR      UmdFileName[MAX_PATH];  // Out: UMD file name
} LW_D3DKMT_UMDFILENAMEINFO;
static LPCWSTR GetUMDPathFromQAI()
{
    static WCHAR szReturn[MAX_PATH + 1] = { 0 };

    UINT hLwAdapter = 0;
    ULONG i;
    LW_D3DKMT_QUERYADAPTERINFO qai;
    LW_D3DKMT_ENUMADAPTERS2 adapters;
    LW_D3DKMT_ADAPTERINFO *pAdapters = NULL;

    // Reset to "", so in case of error we don't return stale data.
    szReturn[0] = 0;

    if (!GetModule(L"gdi32.dll", hModGDI32) ||
        !GetProc(hModGDI32, "D3DKMTEnumAdapters2", pfnD3DKMTEnumAdapters2) ||
        !GetProc(hModGDI32, "D3DKMTQueryAdapterInfo", pfnD3DKMTQueryAdapterInfo))
    {
        goto done;
    }

    memset(&adapters, 0, sizeof(adapters));
    if (pfnD3DKMTEnumAdapters2(&adapters))
        goto done;

    pAdapters = (LW_D3DKMT_ADAPTERINFO*)LocalAlloc(LPTR, adapters.NumAdapters * sizeof(*pAdapters));
    if (!pAdapters)
        goto done;

    adapters.pAdapters = pAdapters;
    if (pfnD3DKMTEnumAdapters2(&adapters))
        goto done;

    // Look for an LWPU display adapter
    for (i = 0; i < adapters.NumAdapters; i++)
    {
        LW_D3DKMT_QUERY_DEVICE_IDS ids;
        memset(&ids, 0, sizeof(ids));
        memset(&qai, 0, sizeof(qai));

        qai.Type = 31/*KMTQAITYPE_PHYSICALADAPTERDEVICEIDS*/;
        qai.pPrivateDriverData = (void*) &ids;
        qai.PrivateDriverDataSize = sizeof(ids);
        qai.hAdapter = adapters.pAdapters[i].hAdapter;
        if (pfnD3DKMTQueryAdapterInfo(&qai))
            goto done;
        if (ids.DeviceIds.VendorID == 0x10de) // LWPU
        {
            hLwAdapter = adapters.pAdapters[i].hAdapter;
            break;
        }
    }

    if (hLwAdapter)
    {
        LW_D3DKMT_UMDFILENAMEINFO info;
        memset(&info, 0, sizeof(info));
        memset(&qai, 0, sizeof(qai));
        qai.Type = 1/*KMTQAITYPE_UMDRIVERNAME*/;
        qai.pPrivateDriverData = (void*) &info;
        qai.PrivateDriverDataSize = sizeof(info);
        qai.hAdapter = hLwAdapter;
        if (pfnD3DKMTQueryAdapterInfo(&qai))
            goto done;

        if (info.UmdFileName[0] != L'\0')
        {
            // Take the directory part of the full UMD file path
            size_t pathLength = FindPathTail(info.UmdFileName) - info.UmdFileName;
            memcpy(szReturn, info.UmdFileName, pathLength*sizeof(WCHAR));
            szReturn[pathLength] = 0;
        }
    }

done:
    SafeLocalFree(pAdapters);
    return szReturn;
}

static LPCWSTR GetUMDPathFromLoadedDLLs()
{
    // On IoTEdge OS SKU, we don't have access to the gdi32.dll functions.
    // Lwrrently, we only need to support a few UMDs on this SKU, and those
    // are already loaded when we get here, so just use their absolute path
    // instead.
    static WCHAR szReturn[MAX_PATH + 1] = { 0 };
    static LPCWSTR validUmds[] = 
    {
        L"lwldumdx.dll",
        L"lwldumd.dll",
        NULL
    };
    int i;

    // Reset to "", so in case of error we don't return stale data.
    szReturn[0] = 0;
    for (i = 0; validUmds[i] != NULL; i++)
    {
        HMODULE hUmdModule = GetModuleHandleW(validUmds[i]);
        if (hUmdModule != NULL)
        {
            DWORD size = GetModuleFileNameW(hUmdModule, szReturn, MAX_PATH);
            if (size != 0)
            {
                // Sanity check: GetModuleFileNameW should return absolute paths always, but verify nonetheless.
                if (!ContainsAbsolutePath(szReturn))
                {
                    LWSLL_DBG(("Path returned by GetModuleFileNameW not absolute: %S\n", szReturn));
                    continue;
                }
                // Cut the string at the last path separator.
                while (size > 0 && szReturn[size] != '\\')
                    --size;

                szReturn[size] = 0;
                LWSLL_DBG(("Returning DriverStore path as '%S' using loaded %S\n", szReturn, validUmds[i]));
                return szReturn;
            }
        }
        else LWSLL_DBG(("Unable to GetModuleHandleW('%S') - 0x%08x\n", validUmds[i], GetLastError()));
    }

    return szReturn;
}

LPWSTR lwDetectDisplayDriverModuleLocationW(LPCWSTR moduleName)
{
    //
    // 1. Try to detect display driver module location by module
    // registration and map it to an absolute path, if available
    //
    // 2. If module is not registered then try to detect existing
    // file in the default display driver location
    //
    // Note that either of above would translate to an absolute
    // path on success
    //

    LPWSTR pszResult = NULL;
    LPWSTR pszRegistration = NULL;
    DWORD dwError = ERROR_SUCCESS;

    SetLastError(ERROR_SUCCESS);

    // Validate arguments

    if ((NULL == moduleName) || ContainsPathSeparator(moduleName))
    {
        SetLastError(ERROR_BAD_ARGUMENTS);
        return NULL;
    }

    //
    // 1. Try to detect display driver module location by module
    // registration and map it to an absolute path, if available
    //

    pszRegistration = lwDetectDisplayDriverModuleRegistrationW(moduleName);
    if (NULL != pszRegistration)
    {
        // Found display driver module registration

        LPWSTR pszSystemDirectory = lwGetSystemDirectoryW();
        if (NULL != pszSystemDirectory)
        {
            if (ContainsPathSeparator(pszRegistration))
            {
                // Registration value reflects some path: try to resolve
                // path header by known system directory aliases - note
                // that we need to do this before checking for an absolute
                // path because we'd recognize Win8+ lwlddmkm ImagePath
                // as an absolute path otherwise:

                LPCWSTR ppszSysdirAliases[] =
                {
                    L"system32\\"              , // Win7  lwlddmkm ImagePath
                    L"\\SystemRoot\\system32\\", // Win8+ lwlddmkm ImagePath
                };

                const LPCWSTR * ppszSysdirAliasesEnd = ppszSysdirAliases + _countof(ppszSysdirAliases);
                LPCWSTR * ppszSysdirAliasIter = ppszSysdirAliases;

                for (; ppszSysdirAliasIter < ppszSysdirAliasesEnd; ++ppszSysdirAliasIter)
                {
                    const LPCWSTR pszSysdirAlias = *ppszSysdirAliasIter;
                    const size_t sysdirAliasLength = wcslen(pszSysdirAlias);

                    if (0 == _wcsnicmp(pszRegistration, pszSysdirAlias, sysdirAliasLength))
                    {
                        // 1a) alias found: resolve registration value to absolute path

                        pszResult = SafePathCombine(pszSystemDirectory, pszRegistration + sysdirAliasLength);
                        break;
                    }
                }

                if (ppszSysdirAliasIter == ppszSysdirAliasesEnd)
                {
                    // No alias found: check for absolute path

                    if (ContainsAbsolutePath(pszRegistration))
                    {
                        // 1b) absolute path: promote registration value to result

                        pszResult = pszRegistration;
                        pszRegistration = NULL;
                    }
                    else
                    {
                        // 1c) relative path: invalid case
                    }
                }
            }
            else
            {
                // 1d) plain filename: presume system directory

                pszResult = SafePathCombine(pszSystemDirectory, pszRegistration);
            }

            SafeLocalFree(pszSystemDirectory);
        }

        SafeLocalFree(pszRegistration);
    }
    else if (ERROR_MOD_NOT_FOUND == (dwError = GetLastError()))
    {
        //
        // 2. If module is not registered then try to detect existing
        // file in the default display driver location
        //

        LPWSTR pszModuleName = ComplementModuleName(moduleName);
        if (NULL != pszModuleName)
        {
            // Get default display driver location

            BOOL bUseDisplayDriverStore = IsWindowsBuildOrGreater(eWindows10RS1DriverStoreSwitch);
            LPWSTR pszDefaultDisplayDriverLocation = bUseDisplayDriverStore
                ? lwGetDisplayDriverStoreW()
                : lwGetSystemDirectoryW();

            if (NULL != pszDefaultDisplayDriverLocation)
            {
                // Try to detect module in default display driver location

                pszResult = DetectModuleInLocation(pszModuleName, pszDefaultDisplayDriverLocation);
                SafeLocalFree(pszDefaultDisplayDriverLocation);
            }

            SafeLocalFree(pszModuleName);
        }
    }

    if (NULL != pszResult)
    {
        SetLastError(ERROR_SUCCESS);
    }
    else if (ERROR_SUCCESS == (dwError = GetLastError()))
    {
        SetLastError(ERROR_MOD_NOT_FOUND);
    }

    return pszResult;
}

LPWSTR lwGetDriverRegkeyW(LPCGUID lpClassGuid) // not exported yet
{
    // Find LWPU device that implements given <lpClassGuid> interface,
    // query its driver name ie subkey, and generate according regkey

    HDEVINFO hDeviceInfo = NULL;
    SP_DEVINFO_DATA deviceInfoData = { sizeof(deviceInfoData), 0 };
    const LPCWSTR pszControlClassKey = L"SYSTEM\\LwrrentControlSet\\Control\\Class\\";
    LPWSTR pszDriverKey = NULL;
    LPWSTR pszResult = NULL;

    SetLastError(ERROR_SUCCESS);

    if (NULL == lpClassGuid)
    {
        SetLastError(ERROR_BAD_ARGUMENTS);
        return NULL;
    }

    if (!GetModule(L"Setupapi.dll", hModSetupapi) ||
        !GetProc(hModSetupapi, "SetupDiGetDeviceRegistryPropertyW", pfnSetupDiGetDeviceRegistryPropertyW) ||
        !GetProc(hModSetupapi, "SetupDiDestroyDeviceInfoList", pfnSetupDiDestroyDeviceInfoList))
    {
        return NULL;
    }

    // Find LWPU device that implements given <lpClassGuid> interface

    if (!FindLwidiaDevice(lpClassGuid, &hDeviceInfo, &deviceInfoData))
    {
        return NULL;
    }

    do
    {
        DWORD dwDriverKeySize = 0;
        size_t resultSize = 0;

        // Query size of driver name, ie registry subkey

        pfnSetupDiGetDeviceRegistryPropertyW(hDeviceInfo, &deviceInfoData, SPDRP_DRIVER, NULL, NULL, 0, &dwDriverKeySize);
        if (0 == dwDriverKeySize)
        {
            break;
        }

        // Allocate driver subkey buffer

        pszDriverKey = (LPWSTR)LocalAlloc(LPTR, dwDriverKeySize);
        if (NULL == pszDriverKey)
        {
            break;
        }

        // Query driver subkey

        if (!pfnSetupDiGetDeviceRegistryPropertyW(hDeviceInfo, &deviceInfoData, SPDRP_DRIVER, NULL, (PBYTE)pszDriverKey, dwDriverKeySize, NULL))
        {
            break;
        }

        // pszResult := pszControlClassKey\pszDriverKey

        pszResult = SafePathCombine(pszControlClassKey, pszDriverKey);
    }
    while (0);

    // Clean up and return result

    SafeLocalFree(pszDriverKey);

    if (NULL != hDeviceInfo)
    {
        pfnSetupDiDestroyDeviceInfoList(hDeviceInfo);
    }

    if (NULL != pszResult)
    {
        SetLastError(ERROR_SUCCESS);
    }

    return pszResult;
}

LPWSTR lwGetDisplayDriverRegkeyW()
{
    return lwGetDriverRegkeyW(&LW_GUID_DISPLAY_DEVICE_ARRIVAL);
}

LPWSTR lwGetDriverStoreW(LPCGUID lpClassGuid) // not exported yet
{
    // Find LWPU device that implements given <lpClassGuid> interface,
    // and query its INF path to return INF driver store location

    HDEVINFO hDeviceInfo = NULL;
    SP_DEVINFO_DATA deviceInfoData = { sizeof(deviceInfoData), 0 };
    LPWSTR pszInfPath = NULL;
    LPWSTR pszInfStoreLocation = NULL;
    LPWSTR pszResult = NULL;

    SetLastError(ERROR_SUCCESS);

    // SetupDiGetDevicePropertyW and SetupGetInfDriverStoreLocationW
    // are just available in Windows Vista and later versions of Windows

    if (!IsWindowsBuildOrGreater(eWindowsVistaBuild))
    {
        SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
        return NULL;
    }

    // Valite arguments

    if (NULL == lpClassGuid)
    {
        SetLastError(ERROR_BAD_ARGUMENTS);
        return NULL;
    }

    if (!GetModule(L"Setupapi.dll", hModSetupapi) ||
        !GetProc(hModSetupapi, "SetupDiGetDevicePropertyW", pfnSetupDiGetDevicePropertyW) ||
        !GetProc(hModSetupapi, "SetupGetInfDriverStoreLocationW", pfnSetupGetInfDriverStoreLocationW) ||
        !GetProc(hModSetupapi, "SetupDiDestroyDeviceInfoList", pfnSetupDiDestroyDeviceInfoList))
    {
        return NULL;
    }

    // Find LWPU device that implements given <lpClassGuid> interface

    if (!FindLwidiaDevice(lpClassGuid, &hDeviceInfo, &deviceInfoData))
    {
        return NULL;
    }

    do
    {
        DWORD dwRequiredSize = 0;
        ULONG ulPropertyType = 0;
        PDEVPROPKEY pPropertyKey = (PDEVPROPKEY)&DEVPKEY_Device_DriverInfPath;

        // Query INF path size

        pfnSetupDiGetDevicePropertyW(hDeviceInfo, &deviceInfoData, pPropertyKey, &ulPropertyType, NULL, 0, &dwRequiredSize, 0);
        if (0 == dwRequiredSize)
        {
            break;
        }

        // Allocate INF path buffer

        pszInfPath = (LPWSTR)LocalAlloc(LPTR, dwRequiredSize * sizeof(WCHAR));
        if (NULL == pszInfPath)
        {
            break;
        }

        // Query INF path

        if (!pfnSetupDiGetDevicePropertyW(hDeviceInfo, &deviceInfoData, pPropertyKey, &ulPropertyType, (PBYTE)pszInfPath, dwRequiredSize, NULL, 0))
        {
            break;
        }

        // Query INF driver store location size

        pfnSetupGetInfDriverStoreLocationW(pszInfPath, NULL, NULL, NULL, 0, &dwRequiredSize);
        if (0 == dwRequiredSize)
        {
            break;
        }

        // Allocate INF driver store location buffer

        pszInfStoreLocation = (LPWSTR)LocalAlloc(LPTR, dwRequiredSize * sizeof(WCHAR));
        if (NULL == pszInfStoreLocation)
        {
            break;
        }

        // Query INF driver store location and copy result

        if (pfnSetupGetInfDriverStoreLocationW(pszInfPath, NULL, NULL, pszInfStoreLocation, dwRequiredSize, 0))
        {
            pszResult = SafeStringDupeN(pszInfStoreLocation, FindPathTail(pszInfStoreLocation) - pszInfStoreLocation - 1);
        }
    }
    while (0);

    // Clean up and return result

    SafeLocalFree(pszInfStoreLocation);
    SafeLocalFree(pszInfPath);

    if (NULL != hDeviceInfo)
    {
        pfnSetupDiDestroyDeviceInfoList(hDeviceInfo);
    }

    if (NULL != pszResult)
    {
        SetLastError(ERROR_SUCCESS);
    }

    return pszResult;
}

LPWSTR lwGetDisplayDriverStoreW()
{
    return lwGetDriverStoreW(&LW_GUID_DISPLAY_DEVICE_ARRIVAL);
}

LPWSTR lwGetSystemDirectoryW()
{
    UINT uLength = 0;
    LPWSTR pResult = NULL;

    SetLastError(ERROR_SUCCESS);

    uLength = GetSystemDirectoryW(pResult, 0);
    pResult = (LPWSTR)LocalAlloc(LPTR, uLength * sizeof(WCHAR));
    if (NULL != pResult)
    {
        UINT uResult = GetSystemDirectoryW(pResult, uLength);
        if ((0 == uResult) || (uLength < uResult))
        {
            SafeLocalFree(pResult);
            pResult = NULL;
        }
    }

    return pResult;
}

void lwSafeLocalFree(HLOCAL mem)
{
    SafeLocalFree(mem);
}

//
// Implementation of static lwSelwreLoadLibrary/RS1 functions
//

static LPCWSTR FindPathTail(const LPCWSTR path)
{
    // Return pointer to last path backslash in
    // <path> if available; return <path> otherwise

    LPCWSTR tail = (LPCWSTR)wcsrchr(path, '\\');
    return (NULL != tail) ? ++tail : path;
}

static LPCWSTR FindDotExt(const LPCWSTR path)
{
    // Return pointer to last dot in <path> if
    // available; <path> + wcslen(<path>) otherwise

    LPCWSTR pathEnd = path + wcslen(path);
    LPCWSTR dotExt = pathEnd;
    while ((path < dotExt) && ('\\' != *dotExt) && ('.' != *dotExt))
    {
        --dotExt;
    }
    return ('.' == *dotExt) ? dotExt : pathEnd;
}

static BOOL IsPathToModule(const LPCWSTR path, const LPCWSTR moduleName)
{
    // Return TRUE if <path> points to <moduleName>, where
    // either arg may omit DLL ext; return FALSE otherwise

    const LPCWSTR pathFileName = FindPathTail(path);
    if (0 == _wcsicmp(pathFileName, moduleName))
    {
        return TRUE; // Perfect match
    }
    else
    {
        // Compare length of ext-less <path> file
        // name with length of ext-less <moduleName>

        const LPCWSTR pathFileNameExt = FindDotExt(pathFileName);
        const LPCWSTR moduleNameExt = FindDotExt(moduleName);

        if ((pathFileNameExt - pathFileName) != (moduleNameExt - moduleName))
        {
            return FALSE;
        }

        // Check if both args have either no ext or DLL ext

        if ((0 != wcscmp(moduleNameExt,   L"") && (0 != _wcsicmp(moduleNameExt,   L".dll"))) ||
            (0 != wcscmp(pathFileNameExt, L"") && (0 != _wcsicmp(pathFileNameExt, L".dll"))))
        {
            return FALSE;
        }

        // Compare ext-less <path> file name with ext-less <moduleName>

        if (0 != _wcsnicmp(pathFileName, moduleName, (size_t)(moduleNameExt - moduleName)))
        {
            return FALSE;
        }

        // <path> file name matches <moduleName> ignoring
        // DLL ext, ie <path> points to <moduleName>

        return TRUE;
    }
}

static LPWSTR SafePathCombine(LPCWSTR pathHead, LPCWSTR pathTail)
{
    // Return combined ie concatenated path, inserting path separator if required

    if (NULL == pathHead)
    {
        return SafeStringDupe(pathTail);
    }
    else if (NULL == pathTail)
    {
        return SafeStringDupe(pathHead);
    }
    else
    {
        size_t headLength = wcslen(pathHead);
        size_t tailLength = wcslen(pathTail);
        BOOL insertPathSeparator = ((0 < headLength) && (pathHead[headLength - 1] != L'\\') &&
                                    (0 < tailLength) && (pathTail[0] != L'\\')) ? TRUE : FALSE;
        size_t pathSize = (headLength + (insertPathSeparator ? 1 : 0) + tailLength + 1) * sizeof(WCHAR);

        LPWSTR path = (LPWSTR)LocalAlloc(LPTR, pathSize);
        if (NULL != path)
        {
            if (FAILED(StringCbCopyW(path, pathSize, pathHead)))
            {
                SafeLocalFree(path);
                path = NULL;
            }
            else if (insertPathSeparator && FAILED(StringCbCatW(path, pathSize, L"\\")))
            {
                SafeLocalFree(path);
                path = NULL;
            }
            else if (FAILED(StringCbCatW(path, pathSize, pathTail)))
            {
                SafeLocalFree(path);
                path = NULL;
            }
        }

        return path;
    }
}

static LPWSTR SafeStringDupe(const LPCWSTR str)
{
    // Returns dupe of non-NULL <str>, NULL otherwise

    size_t resultSize = 0;
    LPWSTR pResult = NULL;

    // Check argument

    if (NULL == str)
    {
        return NULL;
    }

    // Allocate result

    resultSize = wcslen(str) + 1;
    pResult = (LPWSTR)LocalAlloc(LPTR, resultSize * sizeof(WCHAR));
    if (NULL == pResult)
    {
        return NULL;
    }

    // Copy result; dispose result if copy failed

    if (FAILED(StringCchCopyW(pResult, resultSize, str)))
    {
        SafeLocalFree(pResult);
        pResult = NULL;
    }

    return pResult;
}

static LPWSTR SafeStringDupeN(const LPCWSTR str, const ptrdiff_t length)
{
    // Return dupe of the first 0 <= <length> chars of non-NULL <str>, NULL otherwise

    size_t stringLength = 0;
    size_t resultSize = 0;
    LPWSTR pResult = NULL;

    // Check arguments

    if ((NULL == str) || (length < 0))
    {
        return NULL;
    }

    // Allocate result

    stringLength = wcslen(str);
    resultSize = (((size_t)length < stringLength) ? (size_t)length : stringLength) + 1;
    pResult = (LPWSTR)LocalAlloc(LPTR, resultSize * sizeof(WCHAR));
    if (NULL == pResult)
    {
        return NULL;
    }

    // Copy result; dispose result if copy failed

    if (FAILED(StringCchCopyNW(pResult, resultSize, str, length)))
    {
        SafeLocalFree(pResult);
        pResult = NULL;
    }

    return pResult;
}

typedef BOOL (WINAPI *PfnIsWow64Process)(HANDLE, PBOOL);
static BOOL DetectWow64Process(BOOL * pIsWow64Process)
{
    // Report if this is a 32bit process on a 64bit system
    // and return TRUE on success, FALSE otherwise

#ifdef _WIN64

    *pIsWow64Process = FALSE;
    return TRUE;

#else

    static BOOL bIsWow64Process = FALSE;
    static PfnIsWow64Process pfnIsWow64Process = NULL;

    // Check IsWow64Process just once and recycle
    // result once GetProc address succeeded

    if (NULL == pfnIsWow64Process)
    {
        // Note that kernel32 is always mapped - below
        // check just silences static analysis warnings

        HMODULE hModKernel32 = GetModuleHandleA("kernel32");
        if (NULL == hModKernel32)
        {
            return FALSE; // Should never get here...
        }
        else if (!GetProc(hModKernel32, "IsWow64Process", pfnIsWow64Process))
        {
            return FALSE; // GetProcAddress failed
        }
        else if (!pfnIsWow64Process(GetLwrrentProcess(), &bIsWow64Process))
        {
            return FALSE; // IsWow64Process failed
        }
    }

    *pIsWow64Process = bIsWow64Process;
    return TRUE;

#endif
}

static LONG QueryRegistryValueFromKey(HKEY hKey, LPCWSTR pValueName, DWORD * pValueType, LPBYTE * ppValueData)
{
    // Query hKey pValueName

    DWORD dwValueType = 0;
    DWORD cbData = 0;
    DWORD cbExtra = 0;
    LONG lResult = ERROR_SUCCESS;

    if (!GetModule(L"Advapi32.dll", hModAdvapi32) ||
        !GetProc(hModAdvapi32, "RegQueryValueExW", pfnRegQueryValueExW))
    {
        return GetLastError();
    }

    // Query value size

    lResult = pfnRegQueryValueExW(hKey, pValueName, NULL, &dwValueType, NULL, &cbData);
    if ((ERROR_SUCCESS != lResult) || (0 == cbData))
    {
        return lResult;
    }

    // Allocate value buffer - allocate some extra bytes for REG_SZ/EPXAND_SZ/MULTI_SZ
    // and leverage LocalAlloc/LPTR's zero-initialization to make sure we end up with
    // properly terminated multi/strings

    switch (dwValueType)
    {
        case REG_SZ:
        case REG_EXPAND_SZ:
            cbExtra = sizeof(WCHAR);
            break;
        case REG_MULTI_SZ:
            cbExtra = sizeof(WCHAR) + sizeof(WCHAR);
            break;
        default:
            break;
    }

    *ppValueData = (LPBYTE)LocalAlloc(LPTR, cbData + cbExtra);
    if (NULL == *ppValueData)
    {
        return GetLastError();
    }

    // Query value data; dispose value buffer on any failure

    lResult = pfnRegQueryValueExW(hKey, pValueName, NULL, pValueType, *ppValueData, &cbData);
    if (ERROR_SUCCESS != lResult)
    {
        SafeLocalFree(*ppValueData);
        *ppValueData = NULL;
    }

    return lResult;
}

static LONG QueryRegistryValue(HKEY hKey, LPCWSTR lpSubKey, LPCWSTR pValueName, DWORD * pValueType, LPBYTE * ppValueData)
{
    // Query hKey/lpSubKey pValueName

    HKEY hSubKey = NULL;
    LONG lResult = ERROR_SUCCESS;

    if (!GetModule(L"Advapi32.dll", hModAdvapi32) ||
        !GetProc(hModAdvapi32, "RegOpenKeyExW", pfnRegOpenKeyExW) ||
        !GetProc(hModAdvapi32, "RegCloseKey", pfnRegCloseKey))
    {
        return GetLastError();
    }

    // Open hKey/lpSubKey

    lResult = pfnRegOpenKeyExW(hKey, lpSubKey, 0, KEY_QUERY_VALUE, &hSubKey);
    if (ERROR_SUCCESS == lResult)
    {
        // Query hSubKey pValueName

        lResult = QueryRegistryValueFromKey(hSubKey, pValueName, pValueType, ppValueData);

        // Close hKey/lpSubKey

        pfnRegCloseKey(hSubKey);
    }

    return lResult;
}

static BOOL IsWindowsBuildOrGreater(DWORD dwBuildNumber)
{
    // Starting with Win8, user mode version info APIs require compatibility manifests to tell the
    // truth (see https://msdn.microsoft.com/en-us/library/windows/desktop/dn905474(v=vs.85).aspx),
    // and starting with RS4/CV, we cannot read the LwrrentVersion registry. So the process is:
    // 1) Try the API - works for win7 and apps with manifests (CV)
    // 2) Parse HKLM\SOFTWARE\Microsoft\Windows NT\LwrrentVersion\LwrrentBuildNumber - Works for Non-CV.

    static DWORD dwLwrrentBuildNumber = 0;
    if (0 == dwLwrrentBuildNumber)
    {
#if !LWCFG(GLOBAL_FEATURE_WINMOBILE_SUPPORT)
        OSVERSIONINFOEXW osvi;
        DWORDLONG dwlConditionMask = VerSetConditionMask(0, VER_BUILDNUMBER, VER_GREATER_EQUAL);
        memset(&osvi, 0, sizeof(osvi));
        osvi.dwBuildNumber = dwBuildNumber;

        if (VerifyVersionInfoW(&osvi, VER_BUILDNUMBER, dwlConditionMask))
        {
            return TRUE;
        }
        else
#endif
        {
            // Translate REG_SZ HKLM\SOFTWARE\Microsoft\Windows NT\LwrrentVersion\LwrrentBuildNumber to DWORD

            DWORD dwType = 0;
            LPWSTR pszValue = NULL;
            LONG lResult = ERROR_SUCCESS;

            // Query REG_SZ HKLM\SOFTWARE\Microsoft\Windows NT\LwrrentVersion\LwrrentBuildNumber

            lResult = QueryRegistryValue(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows NT\\LwrrentVersion", L"LwrrentBuildNumber", &dwType, (LPBYTE*)&pszValue);
            if ((ERROR_SUCCESS == lResult) && (REG_SZ == dwType))
            {
                // Parse REG_SZ to DWORD, reset to zero and break on non-digit chars

                LPCWSTR pChar = pszValue;
                LPCWSTR pEnd = pszValue + wcslen(pszValue);
                for (; pChar < pEnd; ++pChar)
                {
                    if (isdigit(*pChar))
                    {
                        dwLwrrentBuildNumber *= 10;
                        dwLwrrentBuildNumber += *pChar - '0';
                    }
                    else
                    {
                        dwLwrrentBuildNumber = 0;
                        break;
                    }
                }
            }

            SafeLocalFree(pszValue); // Dispose registry value buffer
        }
    }

    // Compare current build number with required build number

    return (dwLwrrentBuildNumber >= dwBuildNumber) ? TRUE : FALSE;
}


static BOOL FindLwidiaDeviceInDeviceInfo(HDEVINFO hDeviceInfo, SP_DEVINFO_DATA *pDevInfoData, const LPCGUID lpClassGuid)
{
    DWORD dwDeviceIndex = 0;
    SP_DEVICE_INTERFACE_DATA deviceInterfaceData;
    BOOL bIsLwidiaDevice = FALSE;

    ZeroMemory(&deviceInterfaceData, sizeof(deviceInterfaceData));
    deviceInterfaceData.cbSize = sizeof(deviceInterfaceData);

    for(; !bIsLwidiaDevice && pfnSetupDiEnumDeviceInterfaces(hDeviceInfo, NULL, lpClassGuid, dwDeviceIndex, &deviceInterfaceData); ++dwDeviceIndex)
    {
        // Query device interface size in bytes

        DWORD dwRequiredSize = 0;
        PSP_DEVICE_INTERFACE_DETAIL_DATA_W pInterfaceDetail = NULL;

        pfnSetupDiGetDeviceInterfaceDetailW(hDeviceInfo, &deviceInterfaceData, NULL, 0, &dwRequiredSize, NULL);
        if (0 == dwRequiredSize)
        {
            continue;
        }

        // Allocate device interface struct

        pInterfaceDetail = (PSP_DEVICE_INTERFACE_DETAIL_DATA_W)LocalAlloc(LPTR, dwRequiredSize);
        if (NULL == pInterfaceDetail)
        {
            continue;
        }

        // Query device interface data and check for LWPU device by checking device path for LWPU vendor id

        ZeroMemory(pInterfaceDetail, sizeof(*pInterfaceDetail));
        pInterfaceDetail->cbSize = sizeof(*pInterfaceDetail);

        ZeroMemory(pDevInfoData, sizeof(*pDevInfoData));
        pDevInfoData->cbSize = sizeof(*pDevInfoData);

        if (pfnSetupDiGetDeviceInterfaceDetailW(hDeviceInfo, &deviceInterfaceData, pInterfaceDetail, dwRequiredSize, NULL, pDevInfoData))
        {
            _wcsupr(pInterfaceDetail->DevicePath);
            bIsLwidiaDevice = (NULL != wcsstr(pInterfaceDetail->DevicePath, L"VEN_10DE")) ? TRUE : FALSE;
        }

        SafeLocalFree(pInterfaceDetail);
    }
    return bIsLwidiaDevice;
}

static BOOL FindLwidiaDevice(const LPCGUID lpClassGuid, HDEVINFO * phDevInfo, SP_DEVINFO_DATA * pDevInfoData)
{
    //
    // Enumerate present devices that implement <lpClassGuid> interface until
    // first LWPU device.
    //
    // Return TRUE if such a device was found and copy according HDEVINFO and
    // SP_DEVINFO_DATA to provided pointers. Note that caller must destroy
    // reported HDEVINFO using SetupDiDestroyDeviceInfoList in that case.
    //
    // Return FALSE otherwise.
    //

    HDEVINFO hDeviceInfo = NULL;
    SP_DEVINFO_DATA deviceInfoData;

    if (!GetModule(L"Setupapi.dll", hModSetupapi) ||
        !GetProc(hModSetupapi, "SetupDiGetClassDevsW", pfnSetupDiGetClassDevsW) ||
        !GetProc(hModSetupapi, "SetupDiEnumDeviceInterfaces", pfnSetupDiEnumDeviceInterfaces) ||
        !GetProc(hModSetupapi, "SetupDiGetDeviceInterfaceDetailW", pfnSetupDiGetDeviceInterfaceDetailW) ||
        !GetProc(hModSetupapi, "SetupDiDestroyDeviceInfoList", pfnSetupDiDestroyDeviceInfoList))
    {
        return FALSE;
    }

    //
    // First iteration, we only look at present devices. If no LWPU device is found,
    // we repeat the process looking for non-present as well. The non-present check
    // is needed when the only LW decice is lwrrently in TCC/WDM mode.
    // We cannot first check the full list as in some (WDDM only) cases we
    // register a second, fake device in a different path that would be returned.
    //

    // Get handle to set of devices that implement <lpClassGuid> interface
    hDeviceInfo = pfnSetupDiGetClassDevsW(lpClassGuid, NULL, NULL, DIGCF_PRESENT|DIGCF_DEVICEINTERFACE);
    if (ILWALID_HANDLE_VALUE == hDeviceInfo)
    {
        return FALSE;
    }

    // If LWPU device was found then report HDEVINFO and SP_DEVINFO_DATA; cleanup otherwise
    if (FindLwidiaDeviceInDeviceInfo(hDeviceInfo, &deviceInfoData, lpClassGuid))
    {
        *phDevInfo = hDeviceInfo;
        *pDevInfoData = deviceInfoData;
        return TRUE;
    }

    // Now try again, including non-present devices
    pfnSetupDiDestroyDeviceInfoList(hDeviceInfo);
    hDeviceInfo = pfnSetupDiGetClassDevsW(lpClassGuid, NULL, NULL, DIGCF_DEVICEINTERFACE);
    if (ILWALID_HANDLE_VALUE == hDeviceInfo)
        return FALSE;
    if (FindLwidiaDeviceInDeviceInfo(hDeviceInfo, &deviceInfoData, lpClassGuid))
    {
        *phDevInfo = hDeviceInfo;
        *pDevInfoData = deviceInfoData;
        return TRUE;
    }

    pfnSetupDiDestroyDeviceInfoList(hDeviceInfo);
    return FALSE;
}

static LPWSTR ComplementModuleName(const LPCWSTR moduleName)
{
    // Complement omitted module extension according to LoadLibrary behavior

    const LPCWSTR dotDllExt = L".dll";
    const size_t dotDllExtLength = wcslen(dotDllExt);

    size_t moduleNameLength = 0;
    size_t moduleNameDotExtLength = 0;

    LPWSTR pszResult = NULL;

    if (NULL == moduleName)
    {
        return NULL;
    }

    moduleNameLength = wcslen(moduleName);
    moduleNameDotExtLength = wcslen(FindDotExt(moduleName));

    switch (moduleNameDotExtLength)
    {
        case 0:
        {
            //
            // No extension - add .dll extension for detection according to:
            //
            // "If the string specifies a module name without a path and the
            // file name extension is omitted, the function appends the default
            // library extension .dll to the module name."
            //

            const size_t resultSize = (moduleNameLength + dotDllExtLength + 1) * sizeof (WCHAR);
            pszResult = (LPWSTR)LocalAlloc(LPTR, resultSize);
            if (NULL != pszResult)
            {
                if (FAILED(StringCbCopyW(pszResult, resultSize, moduleName)))
                {
                    SafeLocalFree(pszResult);
                    pszResult = NULL;
                }
                else if (FAILED(StringCbCatW(pszResult, resultSize, dotDllExt)))
                {
                    SafeLocalFree(pszResult);
                    pszResult = NULL;
                }
            }
            break;
        }
        case 1:
        {
            //
            // Single digit/dot extension - strip trailing dot for detection according to:
            //
            // "To prevent the function from appending .dll to the module name,
            // include a trailing point character (.) in the module name string."
            //

            pszResult = SafeStringDupeN(moduleName, moduleNameLength - moduleNameDotExtLength);
            break;
        }
        default:
        {
            //
            // Caller provided extension - just this use input for detection:
            //

            pszResult = SafeStringDupe(moduleName);
            break;
        }
    }

    return pszResult;
}

static LPWSTR DetectModulePathFromKernelModeService(LPCWSTR pszModuleName, LPCWSTR pszServiceName)
{
    // Return copy of binary path of service given by <pszServiceName> if service
    // refers to a kernel mode driver and its binary path points to <pszModuleName>

    SC_HANDLE hSCManager = NULL;
    SC_HANDLE hService = NULL;
    LPQUERY_SERVICE_CONFIGW pQueryServiceConfig = NULL;
    LPWSTR pszResult = NULL;

    if (!GetModule(L"Advapi32.dll", hModAdvapi32) ||
        !GetProc(hModAdvapi32, "OpenSCManagerW", pfnOpenSCManagerW) ||
        !GetProc(hModAdvapi32, "OpenServiceW", pfnOpenServiceW) ||
        !GetProc(hModAdvapi32, "QueryServiceConfigW", pfnQueryServiceConfigW) ||
        !GetProc(hModAdvapi32, "CloseServiceHandle", pfnCloseServiceHandle))
    {
        return NULL;
    }

    do
    {
        DWORD dwBytesRequired = 0;

        // Open service manager for read access

        hSCManager = pfnOpenSCManagerW(NULL, NULL, GENERIC_READ);
        if (NULL == hSCManager)
        {
            break;
        }

        // Open service given by <pszServiceName>

        hService = pfnOpenServiceW(hSCManager, pszServiceName, GENERIC_READ);
        if (NULL == hService)
        {
            break;
        }

        // Query service config size

        pfnQueryServiceConfigW(hService, NULL, 0, &dwBytesRequired);
        if (0 == dwBytesRequired)
        {
            break;
        }

        // Allocate service config buffer

        pQueryServiceConfig = (LPQUERY_SERVICE_CONFIGW)LocalAlloc(LPTR, dwBytesRequired);
        if (NULL == pQueryServiceConfig)
        {
            break;
        }

        // Query service config

        if (!pfnQueryServiceConfigW(hService, pQueryServiceConfig, dwBytesRequired, &dwBytesRequired))
        {
            break;
        }

        // Check if service refers to a kernel mode driver

        if (SERVICE_KERNEL_DRIVER != pQueryServiceConfig->dwServiceType)
        {
            break;
        }

        // Check if binary path points to provided <pszModuleName>

        if (IsPathToModule(pQueryServiceConfig->lpBinaryPathName, pszModuleName))
        {
            pszResult = SafeStringDupe(pQueryServiceConfig->lpBinaryPathName); // Success!
        }
    }
    while (0);

    // Clean up and return result

    SafeLocalFree(pQueryServiceConfig);

    if (NULL != hService)
    {
        pfnCloseServiceHandle(hService);
    }

    if (NULL != hSCManager)
    {
        pfnCloseServiceHandle(hSCManager);
    }

    return pszResult;
}

static LPWSTR DetectModulePathFromKernelModeServices(const LPCWSTR pszModuleName, const LPCGUID lpClassGuid)
{
    // Find LWPU device that implements given <lpClassGuid> interface,
    // iterate over its kernel driver services, and return first service
    // binary path that points to provided <pszModuleName>

    HDEVINFO hDeviceInfo = NULL;
    SP_DEVINFO_DATA deviceInfoData = { sizeof(deviceInfoData), 0 };
    LPWSTR pszResult = NULL;

    if (!GetModule(L"Setupapi.dll", hModSetupapi) ||
        !GetProc(hModSetupapi, "SetupDiGetDeviceRegistryPropertyW", pfnSetupDiGetDeviceRegistryPropertyW) ||
        !GetProc(hModSetupapi, "SetupDiDestroyDeviceInfoList", pfnSetupDiDestroyDeviceInfoList))
    {
        return NULL;
    }

    // Find LWPU device that implements given <lpClassGuid> interface

    if (FindLwidiaDevice(lpClassGuid, &hDeviceInfo, &deviceInfoData))
    {
        const DWORD pdwProperties[] =
        {
            SPDRP_SERVICE,
            SPDRP_UPPERFILTERS,
            SPDRP_LOWERFILTERS
        };

        DWORD dwPropertyIndex = 0;

        // Iterate over device registry properties that refer to kernel mode services

        for (; (NULL == pszResult) && (dwPropertyIndex < _countof(pdwProperties)); ++dwPropertyIndex)
        {
            DWORD dwProperty = pdwProperties[dwPropertyIndex];
            DWORD dwPropertyType = REG_NONE;
            DWORD dwPropertyBufferSize = 0;
            DWORD dwPropertyBufferExtra = 0;
            LPBYTE pPropertyBuffer = NULL;

            // Query property size

            pfnSetupDiGetDeviceRegistryPropertyW(hDeviceInfo, &deviceInfoData, dwProperty, &dwPropertyType, NULL, 0, &dwPropertyBufferSize);
            if (0 == dwPropertyBufferSize)
            {
                continue;
            }

            // Allocate property buffer - allocate some extra bytes for REG_SZ/EPXAND_SZ/MULTI_SZ
            // and leverage LocalAlloc/LPTR's zero-initialization to make sure we end up with
            // properly terminated multi/strings

            switch (dwPropertyType)
            {
                case REG_SZ:
                case REG_EXPAND_SZ:
                    dwPropertyBufferExtra = sizeof(WCHAR);
                    break;
                case REG_MULTI_SZ:
                    dwPropertyBufferExtra = sizeof(WCHAR) + sizeof(WCHAR);
                    break;
                default:
                    break;
            }

            pPropertyBuffer = (PBYTE)LocalAlloc(LPTR, dwPropertyBufferSize + dwPropertyBufferExtra);
            if (NULL == pPropertyBuffer)
            {
                continue;
            }

            // Query property value

            if (pfnSetupDiGetDeviceRegistryPropertyW(hDeviceInfo, &deviceInfoData, dwProperty, &dwPropertyType, pPropertyBuffer, dwPropertyBufferSize, NULL))
            {
                // Detect module path from property value

                LPCWSTR pszEntry = (LPCWSTR)pPropertyBuffer;
                switch (dwPropertyType)
                {
                    case REG_MULTI_SZ:
                    {
                        // Detect module path from REG_MULTI_SZ value: iterate pszEntry over
                        // "entry0\0entry1\0...entryN\0\0" entries until we detected a module
                        // path  OR until pszEntry points to end of list, ie '\0'

                        while (('\0' != *pszEntry) && (NULL == (pszResult = DetectModulePathFromKernelModeService(pszModuleName, (LPCWSTR)pPropertyBuffer))))
                        {
                            pszEntry += wcslen(pszEntry) + 1;
                        }

                        break;
                    }
                    case REG_SZ:
                    {
                        // Detect module path from REG_SZ value

                        pszResult = DetectModulePathFromKernelModeService(pszModuleName, (LPCWSTR)pPropertyBuffer);

                        break;
                    }
                    default:
                    {
                        // Unsupported property value type

                        break;
                    }
                }
            }

            SafeLocalFree(pPropertyBuffer); // Dispose property buffer
        }

        pfnSetupDiDestroyDeviceInfoList(hDeviceInfo); // Dispose device info
    }

    return pszResult;
}

static LPWSTR DetectModulePathFromRegistryValues(const LPCWSTR pszModuleName, const HKEY hKey, const LPCWSTR pszSubKey, const LPCWSTR * ppszValueNames)
{
    // Check if given registry values point to provided module and return
    // value on success, NULL otherwise. Note that this function does not
    // report an error if a value was not found.

    HKEY hSubKey = NULL;
    LONG lResult = ERROR_SUCCESS;
    const LPCWSTR * ppszValueName = ppszValueNames;
    LPWSTR pszResult = NULL;

    if (!GetModule(L"Advapi32.dll", hModAdvapi32) ||
        !GetProc(hModAdvapi32, "RegOpenKeyExW", pfnRegOpenKeyExW) ||
        !GetProc(hModAdvapi32, "RegCloseKey", pfnRegCloseKey))
    {
        return NULL;
    }

    // Open hKey/pszSubKey

    lResult = pfnRegOpenKeyExW(hKey, pszSubKey, 0, KEY_QUERY_VALUE, &hSubKey);
    if (ERROR_SUCCESS != lResult)
    {
        return NULL;
    }

    // Iterate over given value names, checking if entries point to provided module

    for (; (NULL == pszResult) && (NULL != *ppszValueName); ++ppszValueName)
    {
        LPBYTE pValueData = NULL;
        DWORD dwValueType = REG_NONE;
        lResult = QueryRegistryValueFromKey(hSubKey, *ppszValueName, &dwValueType, &pValueData);
        if (ERROR_SUCCESS == lResult)
        {
            LPCWSTR pszPath = (LPCWSTR)pValueData;
            switch (dwValueType)
            {
                case REG_MULTI_SZ:
                {
                    // Detect module path from REG_MULTI_SZ value: iterate pszPath over
                    // "entry0\0entry1\0...entryN\0\0" entries until pszPath points to
                    // given module OR until pszPath points to end of list, ie '\0'

                    while (('\0' != *pszPath) && !IsPathToModule(pszPath, pszModuleName))
                    {
                        pszPath += wcslen(pszPath) + 1;
                    }

                    // Copy path as result if it does not point to end of list

                    if ('\0' != *pszPath)
                    {
                        pszResult = SafeStringDupe(pszPath);
                    }

                    break;
                }
                case REG_SZ:
                {
                    // Copy path as result if it points to given module

                    if (IsPathToModule(pszPath, pszModuleName))
                    {
                        pszResult = SafeStringDupe(pszPath);
                    }

                    break;
                }
                default:
                {
                    // Invalid registry value type!

                    SetLastError(ERROR_ILWALID_DATA);

                    break;
                }
            }
        }

        SafeLocalFree(pValueData);
    }

    pfnRegCloseKey(hSubKey); // Close registry key

    return pszResult;
}

static LPWSTR DetectModulePathFromRegistryValueNames(const LPCWSTR pszModuleName, const HKEY hKey, const LPCWSTR * ppszSubKeys)
{
    // Check given registry keys for value names that point to provided
    // module and return value name on success, NULL otherwise. Note that
    // this function does not report an error if a key was not found.

    const LPCWSTR * ppszSubKey = ppszSubKeys;
    HKEY hSubKey = NULL;
    LPWSTR pszResult = NULL;

    if (!GetModule(L"Advapi32.dll", hModAdvapi32) ||
        !GetProc(hModAdvapi32, "RegOpenKeyExW", pfnRegOpenKeyExW) ||
        !GetProc(hModAdvapi32, "RegEnumValueW", pfnRegEnumValueW) ||
        !GetProc(hModAdvapi32, "RegCloseKey", pfnRegCloseKey))
    {
        return NULL;
    }

    // Iterate over given subkeys, checking if value names point to provided module

    for (; (NULL == pszResult) && (NULL != *ppszSubKey); ++ppszSubKey)
    {
        // Open indicated registry key

        LONG lResult = pfnRegOpenKeyExW(hKey, *ppszSubKey, 0, KEY_READ, &hSubKey);
        if (ERROR_SUCCESS == lResult)
        {
            // Allocate buffer for value names to max UNICODE length

            const DWORD dwValueNameSize = 32768;
            LPWSTR pszValueName = (LPWSTR)LocalAlloc(LPTR, dwValueNameSize * sizeof(WCHAR));
            if (NULL != pszValueName)
            {
                // Enumerate regkey value names until a) we found a value name
                // that points to provided module, or b) there are no more items

                DWORD dwValueIndex = 0;

                do
                {
                    DWORD dwValueNameLength = dwValueNameSize - 1;
                    ZeroMemory(pszValueName, dwValueNameSize);
                    lResult = pfnRegEnumValueW(hSubKey, dwValueIndex++, pszValueName, &dwValueNameLength, NULL, NULL, NULL, NULL);
                    if ((ERROR_SUCCESS == lResult) && IsPathToModule(pszValueName, pszModuleName))
                    {
                        pszResult = SafeStringDupe(pszValueName); // Found value name that points to provided module
                    }
                }
                while ((NULL == pszResult) && (ERROR_NO_MORE_ITEMS != lResult));

                SafeLocalFree(pszValueName); // Dispose value name buffer
            }

            pfnRegCloseKey(hSubKey); // Close registry key
        }
    }

    return pszResult;
}

static LPWSTR DetectModuleInLocation(const LPCWSTR pszModuleName, const LPCWSTR pszLocation)
{
    // Check if pszResult = pszLocation\pszModuleName points to an existing file
    // and return pszResult on success, NULL otherwise. Note that this function
    // sets ERROR_SUCCESS in either case, unless there was another real/hard
    // error.

    // pszResult := pszLocation\pszModuleName
    LPWSTR pszResult = SafePathCombine(pszLocation, pszModuleName);
    if (NULL != pszResult)
    {
        // SOL FileExists should succeed for fully qualified module names
        if (!FileExists(pszResult))
        {
            DWORD dwLastError = GetLastError();
            if (ERROR_FILE_NOT_FOUND == dwLastError)
            {
                // Translate ERROR_FILE_NOT_FOUND to ERROR_SUCCESS
                // because we successfully found that given module
                // does not exist in the driver store

                SetLastError(ERROR_SUCCESS);
            }

            SafeLocalFree(pszResult);
            pszResult = NULL;
        }
    }

    return pszResult;
}

#pragma warning(pop) // Restore the warning states
