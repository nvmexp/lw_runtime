/*
 * lwSelwreLoadLibrary.h
 *
 * Provides (hopefully) secure loading of dynamic-link libraries.
 *
 * Copyright (c) 2013, Lwpu Corporation.  All rights reserved.
 *
 * THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
 * LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
 * IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
 */

#ifndef __LW_SELWRE_LOAD_LIBRARY_H__
#define __LW_SELWRE_LOAD_LIBRARY_H__

/*
 * PURPOSE
 *
 * See https://wiki.lwpu.com/engwiki/index.php/DriverSelwrity
 *
 * The functions below may replace LoadLibrary calls for the following
 * scenarios:
 *
 * 1. Loading a library from the system directory:
 *
 *      Use lwLoadSystemLibrary( fileName ), where fileName is really just the
 *      plain filename (i.e. does not conatin any path fragments). Since the
 *      system directory is considered a trusted location, lwLoadSystemLibrary
 *      just prepends the system directory path to given filename before
 *      passing it to LoadLibrary.
 *
 * 2. Loading a library from a trusted location given by an absolute path:
 *
 *      Use lwLoadLibraryFromTrustedLocation( filePath ), where filePath must
 *      specify an absolute file path. lwLoadLibraryFromTrustedLocation will
 *      then check if this path points to (a subdirectory of) a white-listed
 *      trusted location  before passing it to LoadLibrary.
 *      
 *      Note: When specifying a path, be sure to use backslashes (\), and not 
 *      the forward slashes (/)
 *
 * 3. Loading a signed library from any other location given by an absolute path:
 *
 *      Use lwLoadSignedLibrary( filePath, checkForLwidiaSignature ), where
 *      filePath must be an absolute path and checkForLwidiaSignature indicates
 *      whether this file should've been signed by LWPU. lwLoadSignedLibrary
 *      will verify the signature of given file accordingly before passing it
 *      to LoadLibrary.
 *
 *      Note: When specifying a path, be sure to use backslashes (\), and not 
 *      the forward slashes (/)
 * 
 * 4. Loading a library using absolute path:
 *
 *      Use lwLoadLibrary( filePath, checkForLwidiaSignature ), where filePath must
 *      be an absolute path of the library. Based on path, it will decide to check
 *      digital signature before calling LoadLibrary. For example, if filePath 
 *      points to a folder which is part of trusted location then this function 
 *      will call the Windows LoadLibrary API to load that library. If file is 
 *      not part of trusted location, then it will check certificate. If checkForLwidiaSignature 
 *      is set to TRUE, it will check if library is signed by LWPU. If not, it 
 *      will fail the function.
 *
 *      Note: When specifying a path, be sure to use backslashes (\), and not 
 *      the forward slashes (/)
 *
 * You may use lwReleaseSelwreLoadLibraryResources() to release any resources
 * that have been acquired be the functions above - see function documentation
 * below for details.
 *
 * Following are the functions which can be used to replace calls to CreateProcess and
 * different variants of it.  
 *
 * These wrapper methods first checks if the application which you have specified belongs 
 * to trusted location. If it isn't, then it look for the digital signature. If you have 
 * specified checkForLwidiaSignature, then it looks for LWPU digital signature. If the 
 * application you are passing is an LWPU application, then we suggest to keep checkForLwidiaSignature 
 * to true.
 *
 * Note: Make sure that you are using absolute path
 *
 * 1. Use lwCreateProcess() function to replace calls to 'CreateProcess' Windows API
 *
 * 2. Use lwCreateProcessAsUser() function to replace calls to 'CreateProcessAsUser' Windows API
 *    
 * FREQUENTLY ASKED QUESTIONS
 *
 * Q: Which OSes are supported?
 * A: Lwrrently just (Windows XP?), Windows Vista, Win7, Win8, Win8.1, and Win10.x
 *
 * Q: Is lwSelwreLoadLibrary thread-safe?
 * A: No, lwSelwreLoadLibrary does not use any sync objects in favor of
 *    minimized overhead - please sync at caller level if you need to
 *
 * Q: What about DllSearchPath semantics?
 * A: DllSearchPath is *NOT* supported (see function descriptions for details).
 *
 * Q: How to debug/profile unsigned libraries with signature-checking in place?
 * A: Either a) disable signature checks globally by setting
 *
 *    \HKLM\<LW_PE_SIGNATURE_OVERRIDE_REGKEY>\<LW_PE_SIGNATURE_OVERRIDE_REGVALUE>:DWORD!=0
 *    \HKLM\<LW_PE_SIGNATURE_OVERRIDE_REGKEY_RS4>\<LW_PE_SIGNATURE_OVERRIDE_REGVALUE>:DWORD!=0
 */
#define LW_PE_SIGNATURE_OVERRIDE_REGKEY     L"SOFTWARE\\LWPU Corporation\\Global"
#define LW_PE_SIGNATURE_OVERRIDE_REGKEY_RS4 L"SYSTEM\\ControlSet001\\Services\\lwlddmkm"
#define LW_PE_SIGNATURE_OVERRIDE_REGVALUE {0x41FCC608, 0x8496, 0x4DEF, {0xB4, 0x3E, 0x7D, 0x9B, 0xD6, 0x75, 0xA6, 0xFF}}
/*    (translates to a "{41FCC608-8496-4DEF-B43E-7D9BD675A6FF}" regvalue name)
 *
 *    , or b) when compiling your module by defining LW_PE_SIGNATURE_OVERRIDE
 *    at build-time, or by just temporarily toggling this #define right here:
 */
#ifndef LW_PE_SIGNATURE_OVERRIDE
#define LW_PE_SIGNATURE_OVERRIDE 0
#endif
/*
 *    . Note that in either case, you should a) remove this regkey again when
 *    you are done debugging a customer machine, respectivelly b) revert your
 *    #define before submitting a code change!
 *
 * Q: How much time does it take to check a file signature?
 * A: It was observed that lwVerifyFileSignature takes up to 25+ ms, which
 *    adds on top of LoadLibrary time when using lwLoadSignedLibrary.
 *
 * Q: How much additional memory is consumed when checking a file signature?
 * A: It was observed that lwVerifyFileSignature consumes around 1Mb of memory.
 *    However, observation was made while debugging a test app and most of
 *    that memory (90%) was re-used on subsequent lwVerifyFileSignature calls.
 *
 * Q: Do I need to link against any additional .libs when using this module?
 * A: No, any additional external libraries are dynamically linked during runtime.
 *
 * Q: I want to use this module in my DLL. How would I clean-up any resources
 *    that have been acquired by this module (by dynamically linking DLLs, etc.)?
 * A: Call lwReleaseSelwreLoadLibraryResources() from your DLL_PROCESS_DETACH
 *    routine - see the documentation of this function for more details.
 *
 * Q: Re driver store - the driver store is typically located in the system
 *    directory, so what about file system redirection on Wow64?
 * A: https://msdn.microsoft.com/en-us/library/windows/desktop/aa384187(v=vs.85).aspx
 *    states that the driver store is only redirected on XP, Server 2003, Vista
 *    , and Server 2008, ie it's not redirected on Windows 7 and above.
 *    While lwGetDisplayDriverStore reports whatever Setup API returns, this
 *    should not affect lwDetectDisplayDriverModule et al because those functions
 *    just report/use the driver store starting with Windows 10 RS1.
 *
 *
 * ADDITIONAL NOTES
 *
 * You may also want add the following lines directly after including Windows.h:
 *
 * #pragma deprecated (GetModuleHandleA, GetModuleHandleExA, LoadLibraryA, LoadLibraryExA, LoadModuleA)
 * #pragma deprecated (GetModuleHandleW, GetModuleHandleExW, LoadLibraryW, LoadLibraryExW, LoadModuleW)
 *
 * , which would declare those potentially unsafe functions deprecated and
 * cause the compiler to issue a warning during build-time. You may also add:
 *
 * #pragma warning (error: 4995)
 *
 * to implement an even more restrictive approach by translating any
 * deprecation warning C4995 to an error.
 *
 * Note that this header file doesn't impose any of those policies to leave
 * the choice up to the modules itselves (and having both would even cause the
 * compiler to warn/error on the second #pragma instance!).
 *
 * Note: For develop builds, we are by-passing LWPU security specific code flow and
 * calling associated Windows API directly
 */

#include <wchar.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * GetSystemModule wrappers for system modules.
 *
 * lwGetSystemModuleHandle prepends the system directory to
 * given fileName to pass a fully qualified module/library
 * name to GetSystemModule().
 *
 * Note: this functions does _not_ check certificates because
 * some OS-provided libraries like kernel32.dll don't have
 * any, but are covered by Windows File Protection.
 *
 * Note: GetLastError() may be used to retrieve detailed info
 * about the exelwtion status of this function. Common values
 * are:
 *   ERROR_SUCCESS       - no error oclwred (success!)
 *   ERROR_BAD_ARGUMENTS - fileName contains path fragments
 *   ERROR_MOD_NOT_FOUND - file/module not found
 *
 * Parameters:
 *   fileName: plain filename (without any path fragments!)
 *
 * Return values:
 *   Handle to specified module on success, NULL otherwise
 */
HMODULE lwGetSystemModuleHandleW( LPCWSTR fileName );
HMODULE lwGetSystemModuleHandleA( LPCSTR fileName );

#ifdef UNICODE
#define lwGetSystemModuleHandle lwGetSystemModuleHandleW
#else
#define lwGetSystemModuleHandle lwGetSystemModuleHandleA
#endif /* !UNICODE */

/*
 * LoadLibrary wrapper for system libraries.
 *
 * lwLoadSystemLibrary prepends the system directory to
 * given fileName to pass a fully qualified module/library
 * name to LoadLibrary().
 *
 * Note: this function does _not_ check certificates because
 * some OS-provided libraries like kernel32.dll don't have
 * any, but are covered by Windows File Protection.
 *
 * Note: Starting with Windows 10 RS1, driver modules are
 * supposed to live in the driver store instead of the OS
 * system directory. To avoid existing lwLoadSystemLibrary
 * calls starting to fail for display driver modules on RS1,
 * lwLoadSystemLibrary implements a WAR that loads display
 * driver modules from either its registered location or
 * from the default display driver location, but not direclty
 * from the OS system directory anymore.
 *
 * Also note: above WAR is limited to "lw"-prefixed modules
 * for performance reasons. Use lwLoadDisplayDriverModule if
 * you want to load a non-"lw" display driver module on RS1
 * and above, where lwLoadDisplayDriverModule should already
 * take care of OS version checks and pick the right driver
 * location accordingly.
 *
 * Note: GetLastError() may be used to retrieve detailed info
 * about the exelwtion status of this function. Common values
 * are:
 *   ERROR_SUCCESS       - no error oclwred (success!)
 *   ERROR_BAD_ARGUMENTS - fileName contains path fragments
 *   ERROR_MOD_NOT_FOUND - file/module not found
 *
 * Parameters:
 *   fileName: plain filename (without any path fragments!)
 *   dwFlags:  flags that are being forwarded to LoadLibraryEx
 *             after discarding any search path-related options
 *
 * Return values:
 *   Handle to specified module on success, NULL otherwise
 */
HMODULE lwLoadSystemLibraryExW( LPCWSTR fileName, DWORD dwFlags );
HMODULE lwLoadSystemLibraryExA( LPCSTR fileName, DWORD dwFlags );

#define lwLoadSystemLibraryW( fileName ) lwLoadSystemLibraryExW( fileName, 0 )
#define lwLoadSystemLibraryA( fileName ) lwLoadSystemLibraryExA( fileName, 0 )

#ifdef UNICODE
#define lwLoadSystemLibraryEx lwLoadSystemLibraryExW
#define lwLoadSystemLibrary   lwLoadSystemLibraryW
#else
#define lwLoadSystemLibraryEx lwLoadSystemLibraryExA
#define lwLoadSystemLibrary   lwLoadSystemLibraryA
#endif /* !UNICODE */

/*
 * LoadLibrary wrapper for libraries in trusted locations.
 *
 * lwLoadLibraryFromTrustedLocation checks if given filePath
 * points to (a subdirectory of) a white-listed/trusted location
 * before passing it to LoadLibrary. 
 *
 * Note: When specifying a path, be sure to use backslashes (\),
 * and not the forward slashes (/)
 *
 * Note: GetLastError() may be used to retrieve detailed info
 * about the exelwtion status of this function. Common values
 * are:
 *   ERROR_SUCCESS       - no error oclwred (success!)
 *   ERROR_BAD_ARGUMENTS - filePath is not absolute
 *   ERROR_BAD_PATHNAME  - filePath doesn't point to trusted location
 *   ERROR_MOD_NOT_FOUND - file/module not found
 *
 * Parameters:
 *   filePath: absolute file path
 *   dwFlags:  flags that are being forwarded to LoadLibraryEx
 *             after discarding any search path-related options
 *
 * Return values:
 *   Handle to specified module on success, NULL otherwise
 */

HMODULE lwLoadLibraryFromTrustedLocationExW( LPCWSTR filePath, DWORD dwFlags );
HMODULE lwLoadLibraryFromTrustedLocationExA( LPCSTR filePath, DWORD dwFlags );

#define lwLoadLibraryFromTrustedLocationW( filePath ) lwLoadLibraryFromTrustedLocationExW( filePath, 0 )
#define lwLoadLibraryFromTrustedLocationA( filePath ) lwLoadLibraryFromTrustedLocationExA( filePath, 0 )

#ifdef UNICODE
#define lwLoadLibraryFromTrustedLocationEx lwLoadLibraryFromTrustedLocationExW
#define lwLoadLibraryFromTrustedLocation   lwLoadLibraryFromTrustedLocationW
#else
#define lwLoadLibraryFromTrustedLocationEx lwLoadLibraryFromTrustedLocationExA
#define lwLoadLibraryFromTrustedLocation   lwLoadLibraryFromTrustedLocationA
#endif /* !UNICODE */

/*
 * LoadLibrary() wrapper for signed libraries.
 *
 * lwLoadSignedLibrary verifies the signature of given file
 * before passing it to LoadLibrary. If this file is owned by
 * LWPU then checkForLwidiaSignature should be set to TRUE,
 * such that lwLoadSignedLibrary checks its certificate for an
 * LWPU signature.
 *
 * Note: When specifying a path, be sure to use backslashes
 * (\), and not the forward slashes (/)
 *
 * Note: GetLastError() may be used to retrieve detailed info
 * about the exelwtion status of this function. Common values
 * are:
 * 
 *   ERROR_SUCCESS       - no error oclwred (success!)
 *   ERROR_BAD_ARGUMENTS - filePath is not absolute
 *   ERROR_MOD_NOT_FOUND - file/module not found
 *   CRYPT_E_NO_MATCH    - no certificate found, or certificate
 *                         wasn't issued by LWPU though it was
 *                         required per checkForLwidiaSignature
 *
 * Parameters:
 *   fileName:                library file name (may include path fragments)
 *   dwFlags:                 flags that are being forwarded
 *                            to LoadLibraryEx after discarding
 *                            any search path-related options
 *   checkForLwidiaSignature: whether to check for LWPU signature
 *
 * Return values:
 *   Handle to specified module on success, NULL otherwise
 */
HMODULE lwLoadSignedLibraryExW( LPCWSTR filePath, DWORD dwFlags, const BOOL checkForLwidiaSignature );
HMODULE lwLoadSignedLibraryExA( LPCSTR filePath, DWORD dwFlags, const BOOL checkForLwidiaSignature );

#define lwLoadSignedLibraryW( filePath, checkForLwidiaSignature ) lwLoadSignedLibraryExW( filePath, 0, checkForLwidiaSignature )
#define lwLoadSignedLibraryA( filePath, checkForLwidiaSignature ) lwLoadSignedLibraryExA( filePath, 0, checkForLwidiaSignature )

#ifdef UNICODE
#define lwLoadSignedLibraryEx lwLoadSignedLibraryExW
#define lwLoadSignedLibrary   lwLoadSignedLibraryW
#else
#define lwLoadSignedLibraryEx lwLoadSignedLibraryExA
#define lwLoadSignedLibrary   lwLoadSignedLibraryA
#endif /* !UNICODE */

/*
* lwLoadLibrary() wrapper for libraries
*
* lwLoadLibrary is a wrapper around LoadLibrary that performs additional
* actions:
* 
*   - Checks if the pathname is rooted in a trusted location.
* 
*     - If the file is not in a trusted location then fail with
*       CRYPT_E_NO_MATCH unless the file has a digital signature that is
*       trusted by the operating system. Furthermore, when
*       checkForLwidiaSignature is true, fail with CRYPT_E_NO_MATCH unless
*       the signature is LWPU's.
*
*     - If the file is in a trusted location then the signature check is
*       skipped and checkForLwidiaSignature is ignored.
*
* Note: When specifying a path, be sure to use backslashes (\), and not
* the forward slashes (/)
*
* Note: GetLastError() may be used to retrieve detailed info
* about the exelwtion status of this function. Common values
* are:
*
*   ERROR_SUCCESS           - no error, the library was loaded
*   ERROR_BAD_ARGUMENTS     - filePath is not absolute
*   ERROR_ILWALID_PARAMETER - parameter passed is NULL
*   ERROR_MOD_NOT_FOUND     - file/module not found
*   CRYPT_E_NO_MATCH        - no certificate found, or certificate
*                             wasn't issued by LWPU
*
* Parameters:
*   fileName:                library file name (may include path fragments)
*   checkForLwidiaSignature: whether to check for LWPU signature
*
* Return values:
*   Handle to specified module on success, NULL otherwise
*/
HMODULE lwLoadLibraryExW( LPCWSTR filePath, DWORD dwFlags, const BOOL checkForLwidiaSignature );
HMODULE lwLoadLibraryExA( LPCSTR filePath, DWORD dwFlags, const BOOL checkForLwidiaSignature );

#define lwLoadLibraryW( filePath, checkForLwidiaSignature ) lwLoadLibraryExW( filePath, 0, checkForLwidiaSignature )
#define lwLoadLibraryA( filePath, checkForLwidiaSignature ) lwLoadLibraryExA( filePath, 0, checkForLwidiaSignature )

#ifdef UNICODE
#define lwLoadLibraryEx lwLoadLibraryExW
#define lwLoadLibrary lwLoadLibraryW
#else
#define lwLoadLibraryEx lwLoadLibraryExA
#define lwLoadLibrary lwLoadLibraryA
#endif /* !UNICODE */

/*
 * Common file signature verification.
 *
 * lwVerifyFileSignature uses Microsoft's Crypto API to verify the
 * signature of a file (see http://support.microsoft.com/kb/323809).
 *
 * Note: GetLastError() may be used to retrieve detailed info
 * about the exelwtion status of this function. Common values
 * are:
 *
 *   ERROR_SUCCESS       - no error oclwred (success!)
 *   ERROR_MOD_NOT_FOUND - file/module not found
 *   CRYPT_E_NO_MATCH    - no certificate found, or certificate
 *                         wasn't issued by LWPU though it was
 *                         required per checkForLwidiaSignature
 *
 * Parameters:
 *   filePath:        relative or absolute file path
 *   pSignedByLwidia: optional, returns whether given file was signed
 *                    by LWPU
 *
 * Return values:
 *   TRUE on success, FALSE otherwise
 */
BOOL lwVerifyFileSignatureW( LPCWSTR filePath, BOOL * pSignedByLwidia );
BOOL lwVerifyFileSignatureA( LPCSTR filePath, BOOL * pSignedByLwidia );

#ifdef UNICODE
#define lwVerifyFileSignature lwVerifyFileSignatureW
#else
#define lwVerifyFileSignature lwVerifyFileSignatureA
#endif /* !UNICODE */

/*
 * lwSelwreLoadLibrary-specific resource clean-up.
 *
 * lwReleaseSelwreLoadLibraryResources releases any resources
 * that have been acquired by the lwSelwreLoadLibrary module.
 * This is especially usefull in the case that a DLL which used
 * lwSelwreLoadLibrary functionality gets unloaded: here, you
 * might want to call lwReleaseSelwreLoadLibraryResources() in
 * your DLL's PROCESS_DETACH routine.
 *
 * Note: the caller must sync lwReleaseSelwreLoadLibraryResources()
 * with any other lwSelwreLoadLibrary function call, i.e. it must
 * be ensured that there is no other lwSelwreLoadLibrary function
 * being exelwted when calling lwReleaseSelwreLoadLibraryResources()!
 */
void lwReleaseSelwreLoadLibraryResources();

/*
 * lwCreateProcess() is a wrapper for 'CreateProcess' Windows API
 *
 * It performs the following additional operations
 * 
 *   - Checks if application specified is rooted in a trusted location.
 * 
 *     - If it is not in a trusted location then fail with
 *       CRYPT_E_NO_MATCH unless the file has a digital signature that is
 *       trusted by the operating system. Furthermore, when
 *       checkForLwidiaSignature is true, fail with CRYPT_E_NO_MATCH unless
 *       the signature is LWPU's.
 *
 *     - If the file is in a trusted location then the signature check is
 *       skipped and checkForLwidiaSignature is ignored.
 *
 * Note: When specifying an absolute path, be sure to use backslashes (\), and not 
 * the forward slashes (/)
 *
 * Note: Make sure that application name is specified with extension. For example,
 * C:\Windows\System32\notepad.exe is correct whereas C:\Windows\System32\notepad 
 * is incorrect
 *
 *
 * Note: GetLastError() may be used to retrieve detailed info
 * about the exelwtion status of this function. Common values
 * are:
 *   ERROR_SUCCESS           - no error, the process is created 
 *   ERROR_ILWALID_PARAMETER - parameter(s) passed to function are incorrect
 *   ERROR_NOT_ENOUGH_MEMORY - not enough memory to process the request
 *   CRYPT_E_NO_MATCH        - no certificate found, or certificate
 *                             wasn't issued by LWPU
 *
 * Parameters:
 *   All parameters, checkForLwidiaSignature, are what you would pass to CreateProcess
 *   Windows API. Check MSDN document for CreateProcess to know more about parameters
 *  
 *   checkForLwidiaSignature: whether to check for LWPU signature
 *
 * Return values:
 *   returns TRUE if CreateProcess successfully called, else FALSE
 */
BOOL
lwCreateProcessA(
    LPCSTR lpApplicationName,
    LPSTR lpCommandLine,
    LPVOID lpProcessAttributes, // LPSELWRITY_ATTRIBUTES lpProcessAttributes,
    LPVOID lpThreadAttributes,// LPSELWRITY_ATTRIBUTES lpThreadAttributes,
    BOOL bInheritHandles,
    DWORD dwCreationFlags,
    LPVOID lpElwironment,
    LPCSTR lpLwrrentDirectory,
    LPVOID lpStartupInfo, // LPSTARTUPINFOA lpStartupInfo,
    BOOL checkForLwidiaSignature,
    LPVOID lpProcessInformation // LPPROCESS_INFORMATION lpProcessInformation
    );
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
    );

#ifdef UNICODE
#define lwCreateProcess lwCreateProcessW
#else
#define lwCreateProcess lwCreateProcessA
#endif /* !UNICODE */

/*
 * lwCreateProcessAsUser() is a wrapper for 'CreateProcessAsUser' Windows API
 *
 * It performs the following additional operations
 * 
 *   - Checks if application specified is rooted in a trusted location.
 * 
 *     - If it is not in a trusted location then fail with
 *       CRYPT_E_NO_MATCH unless the file has a digital signature that is
 *       trusted by the operating system. Furthermore, when
 *       checkForLwidiaSignature is true, fail with CRYPT_E_NO_MATCH unless
 *       the signature is LWPU's.
 *
 *     - If the file is in a trusted location then the signature check is
 *       skipped and checkForLwidiaSignature is ignored.
 *
 * Note: When specifying an absolute path, be sure to use backslashes (\), and not 
 * the forward slashes (/)
 *
 * Note: Make sure that application name is specified with extension. For example,
 * C:\Windows\System32\notepad.exe is correct whereas C:\Windows\System32\notepad 
 * is incorrect
 *
 *
 * Note: GetLastError() may be used to retrieve detailed info
 * about the exelwtion status of this function. Common values
 * are:
 *   ERROR_SUCCESS           - no error, the process is created 
 *   ERROR_ILWALID_PARAMETER - parameter(s) passed to function are incorrect
 *   ERROR_NOT_ENOUGH_MEMORY - not enough memory to process the request
 *   CRYPT_E_NO_MATCH        - no certificate found, or certificate
 *                             wasn't issued by LWPU
 *
 * Parameters:
 *   All parameters, checkForLwidiaSignature, are what you would pass to CreateProcessAsUser
 *   Windows API. Check MSDN document for CreateProcessAsUser to know more about parameters
 *  
 *   checkForLwidiaSignature: whether to check for LWPU signature
 *
 * Return values:
 *   returns TRUE if CreateProcessAsUser successfully called, else FALSE
 */

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
    );
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
    );

#ifdef UNICODE
#define lwCreateProcessAsUser lwCreateProcessAsUserW
#else
#define lwCreateProcessAsUser lwCreateProcessAsUserA
#endif /* !UNICODE */

/*
 * lwShellExelwte is a wrapper for 'ShellExelwte' windows API
 *
 * Note: Use this method only if the 'lpFile' argument is a file name. For
 *       use cases where the 'lpfile' argument point to a shell object on which
 *       an operation needs to be performed, directly call the windows API.
 *
 *  It performs the following additional operations
 *
 *   - Checks if application specified is rooted in a trusted location.
 *
 *     - If it is not in a trusted location then fail with
 *       CRYPT_E_NO_MATCH unless the file has a digital signature that is
 *       trusted by the operating system. Furthermore, when
 *       checkForLwidiaSignature is true, fail with CRYPT_E_NO_MATCH unless
 *       the signature is LWPU's.
 *
 *     - If the file is in a trusted location then the signature check is
 *       skipped and checkForLwidiaSignature is ignored.
 *
 * Note: When specifying an absolute path, be sure to use backslashes (\), and not
 * the forward slashes (/)
 *
 * Note: Make sure that application name is specified with extension. For example,
 * C:\Windows\System32\notepad.exe is correct whereas C:\Windows\System32\notepad
 * is incorrect
 *
 *
 * Note: GetLastError() may be used to retrieve detailed info
 * about the exelwtion status of this function. Common values
 * are:
 *   ERROR_SUCCESS           - no error, the process is created
 *   ERROR_ILWALID_PARAMETER - parameter(s) passed to function are incorrect.
 *                             Note that the 'lpfile' parameter is treated as a file 
 *                             name in this function and when the 'lpfile' does not 
 *                             resolve to a file name, this error code will be reported. 
*/
HINSTANCE lwShellExelwteA(
    HWND hwnd, 
    LPCSTR lpOperation, 
    LPCSTR lpFile, 
    LPCSTR lpParameters,
    LPCSTR lpDirectory, 
    INT nShowCmd,
    BOOL checkForLwidiaSignature    // Specify if LWPU signature check is needed.
    );

HINSTANCE lwShellExelwteW(
    HWND hwnd, 
    LPCWSTR lpOperation, 
    LPCWSTR lpFile, 
    LPCWSTR lpParameters,
    LPCWSTR lpDirectory, 
    INT nShowCmd,
    BOOL checkForLwidiaSignature    // Specify if LWPU signature check is needed.
    );

#ifdef UNICODE
#define lwShellExelwte  lwShellExelwteW
#else
#define lwShellExelwte  lwShellExelwteA
#endif // !UNICODE

// forward declaration
typedef struct _SHELLEXELWTEINFOA SHELLEXELWTEINFOA, *LPSHELLEXELWTEINFOA;

BOOL lwShellExelwteExA(
    LPSHELLEXELWTEINFOA pExecInfo,
    BOOL checkForLwidiaSignature    // Specify if LWPU signature check is needed.
    );

// forward declaration
typedef struct _SHELLEXELWTEINFOW SHELLEXELWTEINFOW, *LPSHELLEXELWTEINFOW;

BOOL lwShellExelwteExW(
    LPSHELLEXELWTEINFOW pExecInfo,
    BOOL checkForLwidiaSignature    // Specify if LWPU signature check is needed.
    );

#ifdef UNICODE
#define lwShellExelwteEx  lwShellExelwteExW
#else
#define lwShellExelwteEx  lwShellExelwteExA
#endif // !UNICODE

/*
 * LoadLibrary wrapper for display driver modules
 *
 * Presuming provided module name refers to a LWPU display driver
 * module, lwLoadDisplayDriverModule tries to detect its location
 * using lwDetectDisplayDriverModuleLocation and loads it from there.
 *
 * Note: GetLastError() may be used to retrieve detailed info
 * about the exelwtion status of this function. Common values
 * are:
 *   ERROR_SUCCESS       - no error oclwred (success!)
 *   ERROR_BAD_ARGUMENTS - moduleName contains path fragments
 *   ERROR_MOD_NOT_FOUND - file/module not found
 *
 * Parameters:
 *   fileName: display driver module name
 *   dwFlags:  flags that are being forwarded to LoadLibraryEx
 *             after discarding any search path-related options
 *
 * Return values:
 *   Handle to specified module on success, NULL otherwise.
 */

HMODULE lwLoadDisplayDriverModuleExW(LPCWSTR fileName, DWORD dwFlags);
HMODULE lwLoadDisplayDriverModuleExA(LPCSTR fileName, DWORD dwFlags);

#ifdef UNICODE
#define lwLoadDisplayDriverModuleEx  lwLoadDisplayDriverModuleExW
#else
#define lwLoadDisplayDriverModuleEx  lwLoadDisplayDriverModuleExA
#endif // !UNICODE

#define lwLoadDisplayDriverModuleW(fileName) lwLoadDisplayDriverModuleExW(fileName, 0)
#define lwLoadDisplayDriverModuleA(fileName) lwLoadDisplayDriverModuleExA(fileName, 0)

#ifdef UNICODE
#define lwLoadDisplayDriverModule  lwLoadDisplayDriverModuleW
#else
#define lwLoadDisplayDriverModule  lwLoadDisplayDriverModuleA
#endif // !UNICODE

/*
 * Display driver module location detection
 *
 * Presuming provided module name refers to a LWPU display driver
 * module, lwDetectDisplayDriverModuleLocationW tries to detect its
 * location by i) evaluating registry keys and setup info that is
 * used to register display driver modules with the OS or other
 * ecosystems, and ii) checking the LWPU display driver location
 * for such a module.
 *
 * On success, lwDetectDisplayDriverModuleLocationW returns an absolute
 * path to presumed location of given module. Use lwSafeLocalFree() to
 * free the returned string in this case.
 *
 * lwDetectDisplayDriverModuleLocationW returns NULL on any failure,
 * where GetLastError() may be used to retrieve detailed info about
 * the exelwtion status of this function. Common values are:
 *
 *   ERROR_SUCCESS              - no error oclwred (success!)
 *   ERROR_CALL_NOT_IMPLEMENTED - function not implemented on this
 *                                platform (need OS >= Vista)
 *   ERROR_BAD_ARGUMENTS        - moduleName contains path fragments
 *   ERROR_MOD_NOT_FOUND        - failed to detect module location
 *                                because moduleName does not refer
 *                                to a display driver module
 *   any other error            - failed to detect driver module
 *                                location due to indicated error
 *                                , where moduleName may or may not
 *                                refer to a display driver module
 *
 * Parameters:
 *   moduleName: display driver module name
 *
 * Return values:
 *   Module registration value or driver location-based path on
 *   success, NULL otherwise.
 *
 * Note: lwDetectDisplayDriverModuleLocationW really just detects the
 * presumed location for given module, ie it does not necessarily
 * check if the module really exists in this location.
 */

LPWSTR lwDetectDisplayDriverModuleLocationW(LPCWSTR moduleName);

/*
 * Display driver registry key retrieval
 *
 * On success, lwGetDisplayDriverRegkeyW() returns the path of the
 * LWPU display driver registry key, where lwSafeLocalFree() should
 * be used to free the returned path.
 *
 * lwGetDisplayDriverRegkeyW() returns NULL on any failure, where
 * GetLastError() may be used to retrieve detailed info about
 * the exelwtion status of this function. Common values are:
 * are:
 *   ERROR_SUCCESS              - no error oclwred (success!)
 *   ERROR_CALL_NOT_IMPLEMENTED - function not implemented on this
 *                                platform (need OS >= Vista)
 *
 * Return values:
 *   LWPU display driver registry key on success, NULL otherwise.
 *
 * Note: this function does NOT return a HKLM\SYSTEM\CCS\Control\Video
 * subkey, but a HKLM\SYSTEM\CCS\Control\Class subkey. However, this
 * CSS\Control\Video key is just a link to the CSS\Control\Class key.
 */

LPWSTR lwGetDisplayDriverRegkeyW();

/*
 * LWPU display driver store retrieval
 *
 * lwGetDisplayDriverStoreW() returns the path to the LWPU
 * display driver store on success, and NULL on any failure.
 *
 * The caller of this function is responsible to dispose
 * returned path using lwSafeLocalFree().
 *
 * GetLastError() may be used to retrieve detailed info about
 * the exelwtion status of this function. Common values are:
 *
 *   ERROR_SUCCESS              - no error oclwred (success!)
 *   ERROR_CALL_NOT_IMPLEMENTED - function not implemented on this
 *                                platform (need OS >= Vista)
 *
 * Return values:
 *   LWPU display driver store path on success, NULL otherwise.
 *
 * Note: lwGetDisplayDriverStoreW is just implemented on Windows
 * Vista and later versions of Windows.
 *
 * Also note: %windir%\system32\driverstore is redirected, ie
 * may point to %windir%\SysWow64\driverstore before Windows 7
 * - see FAQ for further details.
 */

LPWSTR lwGetDisplayDriverStoreW();

/*
 * OS system directory retrieval
 *
 * lwGetSystemDirectoryW() returns the path to the OS system
 * directory on success, and NULL on any failure.
 *
 * The caller of this function is responsible to dispose
 * returned path using lwSafeLocalFree().
 *
 * Return values:
 *   OS system directoy on success, NULL otherwise.
 */

LPWSTR lwGetSystemDirectoryW();

/*
 * lwSelwreLoadLibrary allocation disposal
 *
 * Any memory allocation that is returned by above functions
 * should be free'd using lwSafeLocalFree().
 */

void lwSafeLocalFree(HLOCAL mem);

#ifdef __cplusplus
}
#endif

#endif /* __LW_SELWRE_LOAD_LIBRARY_H__ */
