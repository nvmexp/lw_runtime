 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: exception.cpp                                                     *|
|*                                                                            *|
 \****************************************************************************/
#include "precomp.h"

//******************************************************************************
//
// Forwards
//
//******************************************************************************
CString         filename(const char* pPath);

//******************************************************************************
//
// Locals
//
//******************************************************************************
ULONG           CBreakException::m_ulReference = 0;

//******************************************************************************

CException::CException
(
    HRESULT             hResult,
    const char         *pFile,
    const char         *pFunction,
    int                 nLine
)
:   m_hResult(hResult),
    m_pFile(pFile),
    m_pFunction(pFunction),
    m_nLine(nLine)
{
    // This CException constructor is only for setting the error location

} // CException

//******************************************************************************

CException::CException
(
    HRESULT             hResult,
    const char         *pFile,
    const char         *pFunction,
    int                 nLine,
    const char         *pszFormat,
    ...
)
:   m_hResult(hResult),
    m_pFile(pFile),
    m_pFunction(pFunction),
    m_nLine(nLine)
{
    va_list va;

    assert(pszFormat != NULL);

    // Setup start of the variable argument list
    va_start(va, pszFormat);

    // Initialize and format the exception description
    memset(m_szDescription, 0, MAX_EXCEPTION_DESCRIPTION);
    vsprintf(m_szDescription, pszFormat, va);

    va_end(va);

} // CException

//******************************************************************************

CException::CException
(
    HRESULT             hResult,
    const char*         pszFormat, ...
)
:   m_hResult(hResult),
    m_pFile(NULL),
    m_pFunction(NULL),
    m_nLine(0)
{
    va_list va;

    assert(pszFormat != NULL);

    // Setup start of the variable argument list
    va_start(va, pszFormat);

    // Initialize and format the exception description
    memset(m_szDescription, 0, MAX_EXCEPTION_DESCRIPTION);
    vsprintf(m_szDescription, pszFormat, va);

    va_end(va);

} // CException

//******************************************************************************

CException::CException
(
    HRESULT             hResult
)
:   m_hResult(hResult),
    m_pFile(NULL),
    m_pFunction(NULL),
    m_nLine(0)
{
    // Initialize the exception description
    memset(m_szDescription, 0, MAX_EXCEPTION_DESCRIPTION);

} // CException

//******************************************************************************

CException::~CException()
{


} // ~CException

//******************************************************************************

CString
CException::location() const
{
    CString             sLocation;

    // Check for a valid location for this exception
    if ((file() != NULL) && (function() != NULL) && (line() != 0))
    {
        // Generate location string for this exception
        sLocation.sprintf("%s!%s@%d", filename(file()), function(), line());
    }
    return sLocation;

} // location

//******************************************************************************

void
CException::dPrintf() const
{
    CString             sLocation = location();

    // Print exception description (w/color if DML)
    if (dmlState())
    {
        // Check for a location for this exception
        if (sLocation.length() != 0)
        {
            ::dPrintf("\n%s %s [%s]\n", foreground("EXCEPTION", RED), description(), foreground(sLocation, BLUE));
        }
        else    // No location
        {
            ::dPrintf("\n%s %s\n", foreground("EXCEPTION", RED), description());
        }
        ::dPrintf("%s = 0x%08x (%s)\n", foreground("HRESULT", RED), hResult(), errorString(hResult()));
    }
    else    // No DML (plain text only)
    {
        // Check for a location for this exception
        if (sLocation.length() != 0)
        {
            ::dPrintf("\nEXCEPTION %s [%s]\n", description(), sLocation);
        }
        else    // No location
        {
            ::dPrintf("\nEXCEPTION %s\n", description());
        }
        ::dPrintf("HRESULT = 0x%08x (%s)\n", hResult(), errorString(hResult()));
    }

} // dPrintf

//******************************************************************************

void
CException::dbgPrintf() const
{
    CString             sLocation = location();

    // Print exception description (w/color if DML)
    if (dmlState())
    {
        // Check for a location for this exception
        if (sLocation.length() != 0)
        {
            ::dbgPrintf("\n%s %s [%s]\n", foreground("EXCEPTION", RED), description(), foreground(sLocation, BLUE));
        }
        else    // No location
        {
            ::dbgPrintf("\n%s %s\n", foreground("EXCEPTION", RED), description());
        }
        ::dbgPrintf("%s = 0x%08x (%s)\n", foreground("HRESULT", RED), hResult(), errorString(hResult()));
    }
    else    // No DML
    {
        // Check for a location for this exception
        if (sLocation.length() != 0)
        {
            ::dbgPrintf("\nEXCEPTION %s [%s]\n", description(), sLocation);
        }
        else    // No location
        {
            ::dbgPrintf("\nEXCEPTION %s\n", description());
        }
        ::dbgPrintf("HRESULT = 0x%08x (%s)\n", hResult(), errorString(hResult()));
    }

} // dbgPrintf

//******************************************************************************

CSymbolException::CSymbolException
(
    HRESULT             hResult,
    const char         *pFile,
    const char         *pFunction,
    int                 nLine,
    const char         *pszFormat,
    ...
)
:   CException(hResult, pFile, pFunction, nLine)
{
    va_list va;

    assert(pszFormat != NULL);

    // Setup start of the variable argument list
    va_start(va, pszFormat);

    // Initialize and format the exception description
    memset(m_szDescription, 0, MAX_EXCEPTION_DESCRIPTION);
    vsprintf(m_szDescription, pszFormat, va);

    va_end(va);

} // CSymbolException

//******************************************************************************

CSymbolException::CSymbolException
(
    HRESULT             hResult,
    const char         *pszFormat,
    ...
)
:   CException(hResult)
{
    va_list va;

    assert(pszFormat != NULL);

    // Setup start of the variable argument list
    va_start(va, pszFormat);

    // Initialize and format the exception description
    memset(m_szDescription, 0, MAX_EXCEPTION_DESCRIPTION);
    vsprintf(m_szDescription, pszFormat, va);

    va_end(va);

} // CSymbolException

//******************************************************************************

CSymbolException::~CSymbolException()
{


} // ~CSymbolException

//******************************************************************************

CTargetException::CTargetException
(
    HRESULT             hResult,
    const char         *pFile,
    const char         *pFunction,
    int                 nLine,
    const char         *pszFormat,
    ...
)
:   CException(hResult, pFile, pFunction, nLine)
{
    va_list va;

    assert(pszFormat != NULL);

    // Setup start of the variable argument list
    va_start(va, pszFormat);

    // Initialize and format the exception description
    memset(m_szDescription, 0, MAX_EXCEPTION_DESCRIPTION);
    vsprintf(m_szDescription, pszFormat, va);

    va_end(va);

} // CTargetException

//******************************************************************************

CTargetException::CTargetException
(
    HRESULT             hResult,
    const char         *pszFormat,
    ...
)
:   CException(hResult)
{
    va_list va;

    assert(pszFormat != NULL);

    // Setup start of the variable argument list
    va_start(va, pszFormat);

    // Initialize and format the exception description
    memset(m_szDescription, 0, MAX_EXCEPTION_DESCRIPTION);
    vsprintf(m_szDescription, pszFormat, va);

    va_end(va);

} // CTargetException

//******************************************************************************

CTargetException::~CTargetException()
{


} // ~CTargetException

//******************************************************************************

CStringException::CStringException
(
    HRESULT             hResult,
    const char         *pFile,
    const char         *pFunction,
    int                 nLine,
    const char         *pszFormat,
    ...
)
:   CException(hResult, pFile, pFunction, nLine)
{
    va_list va;

    assert(pszFormat != NULL);

    // Setup start of the variable argument list
    va_start(va, pszFormat);

    // Initialize and format the exception description
    memset(m_szDescription, 0, MAX_EXCEPTION_DESCRIPTION);
    vsprintf(m_szDescription, pszFormat, va);

    va_end(va);

} // CStringException

//******************************************************************************

CStringException::CStringException
(
    HRESULT             hResult,
    const char         *pszFormat,
    ...
)
:   CException(hResult)
{
    va_list va;

    assert(pszFormat != NULL);

    // Setup start of the variable argument list
    va_start(va, pszFormat);

    // Initialize and format the exception description
    memset(m_szDescription, 0, MAX_EXCEPTION_DESCRIPTION);
    vsprintf(m_szDescription, pszFormat, va);

    va_end(va);

} // CStringException

//******************************************************************************

CStringException::~CStringException()
{


} // ~CStringException

//******************************************************************************

CMemoryException::CMemoryException
(
    size_t              size,
    const char         *pFile,
    const char         *pFunction,
    int                 nLine
)
:   CException(E_OUTOFMEMORY, pFile, pFunction, nLine, "Memory allocation failure"),
    m_size(size)
{


} // CMemoryException

//******************************************************************************

CMemoryException::CMemoryException
(
    size_t              size
)
:   CException(E_OUTOFMEMORY, "Memory allocation failure"),
    m_size(size)
{


} // CMemoryException

//******************************************************************************

CMemoryException::~CMemoryException()
{


} // ~CMemoryException

//******************************************************************************

void
CMemoryException::dPrintf() const
{
    CString             sLocation = location();

    // Print exception description (w/color if DML)
    if (dmlState())
    {
        // Check for a location for this exception
        if (sLocation.length() != 0)
        {
            ::dPrintf("\n%s %s [%s]\n", foreground("EXCEPTION", RED), description(), foreground(sLocation, BLUE));
        }
        else    // No location
        {
            ::dPrintf("\n%s %s\n", foreground("EXCEPTION", RED), description());
        }
    }
    else    // No DML
    {
        // Check for a location for this exception
        if (sLocation.length() != 0)
        {
            ::dPrintf("\nEXCEPTION %s [%s]\n", description(), sLocation);
        }
        else    // No location
        {
            ::dPrintf("\nEXCEPTION %s\n", description());
        }
    }
    ::dPrintf("Failed to allocate 0x%I64x (%I64d) bytes\n", size(), size());

} // dPrintf

//******************************************************************************

void
CMemoryException::dbgPrintf() const
{
    CString             sLocation = location();

    // Print exception description (w/color if DML)
    if (dmlState())
    {
        // Check for a location for this exception
        if (sLocation.length() != 0)
        {
            ::dbgPrintf("\n%s %s [%s]\n", foreground("EXCEPTION", RED), description(), foreground(sLocation, BLUE));
        }
        else    // No location
        {
            ::dbgPrintf("\n%s %s\n", foreground("EXCEPTION", RED), description());
        }
    }
    else    // No DML
    {
        // Check for a location for this exception
        if (sLocation.length() != 0)
        {
            ::dbgPrintf("\nEXCEPTION %s [%s]\n", description(), sLocation);
        }
        else    // No location
        {
            ::dbgPrintf("\nEXCEPTION %s\n", description());
        }
    }
    ::dPrintf("Failed to allocate 0x%I64x (%I64d) bytes\n", size(), size());

} // dbgPrintf

//******************************************************************************

CBreakException::CBreakException
(
    HRESULT             hResult
)
:   CException(hResult)
{
    // Increment the break exception reference count
    m_ulReference++;

} // CBreakException

//******************************************************************************

CBreakException::~CBreakException()
{
    // Decrement the break exception reference count
    m_ulReference--;

} // ~CBreakException

//******************************************************************************

void
CBreakException::dPrintf() const
{
    // Print break (w/bold if DML)
    if (dmlState())
    {
        ::dPrintf("\n%s\n", bold("User break"));
    }
    else    // No DML
    {
        ::dPrintf("\nUser break\n");
    }

} // dPrintf

//******************************************************************************

void
CBreakException::dbgPrintf() const
{
    // Print break (w/bold if DML)
    if (dmlState())
    {
        ::dbgPrintf("\n%s\n", bold("User break"));
    }
    else    // No DML
    {
        ::dbgPrintf("\nUser break\n");
    }

} // dbgPrintf

//******************************************************************************

CString
errorString
(
    LONG                error
)
{
    CString             sCommand(MAX_COMMAND_STRING);
    regex_t             reError = {0};
    regmatch_t          reMatch[25];
    char               *pBuffer = NULL;
    char               *pStart;
    char               *pEnd;
    ULONG               ulSize;
    int                 reResult;
    HRESULT             hResult;
    CString             sError("No error");

    // Check for non-zero error code
    if (error != 0)
    {
        // Try to compile error regular expression
        reResult = regcomp(&reError, ERROREXPR, REG_EXTENDED + REG_ICASE);
        if (reResult == REG_NOERROR)
        {
            // Format the command to get error information
            sCommand.sprintf("!ext.error 0x%0x", error);        

            // Clear the capture buffer
            ClearCapture();

            // Setup to capture extension information
            FlushCallbacks();
            SetCaptureState(TRUE);

            // Try to get the error information
            Execute(DEBUG_OUTCTL_THIS_CLIENT | DEBUG_OUTCTL_NOT_LOGGED, sCommand, DEBUG_EXELWTE_NOT_LOGGED | DEBUG_EXELWTE_NO_REPEAT);

            // Turn output capture off (Make sure output is flushed)
            FlushCallbacks();
            SetCaptureState(FALSE);

            // Try to get the captured output
            hResult = getCaptureBuffer(&pBuffer, &ulSize);
            if (SUCCEEDED(hResult))
            {
                // Check for captured output present
                if (pBuffer != NULL)
                {
                    // Loop processing captured output lines
                    pStart  = pBuffer;
                    pEnd    = terminate(pBuffer, pStart, ulSize);
                    while (static_cast<ULONG>(pStart - pBuffer) < ulSize)
                    {
                        // Check to see if this matches an error information line
                        reResult = regexec(&reError, pStart, countof(reMatch), reMatch, 0);
                        if (reResult == REG_NOERROR)
                        {
                            // Get the error string
                            sError = subExpression(pStart, reMatch, ERROR_DESCRIPTION);

                            // Break out of the search loop
                            break;
                        }
                        // Skip this line (not the requested error information)
                        pStart = pEnd + 1;
                        pEnd   = terminate(pBuffer, pStart, ulSize);
                    }
                    // Free the capture output buffer
                    free(pBuffer);
                    pBuffer = NULL;
                }
            }
        }
        else    // Invalid regular expression
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &reError, ERROREXPR));
        }
    }
    return sError;

} // errorString

//******************************************************************************

CString
filename
(
    const char         *pPath
)
{
    regex_t             reFile = {0};
    regmatch_t          reMatch[25];
    int                 reResult;
    CString             sFilename(*pPath);

    // Try to compile filename regular expression
    reResult = regcomp(&reFile, FILEEXPR, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Try to parse the filename paath string
        reResult = regexec(&reFile, pPath, countof(reMatch), reMatch, 0);
        if (reResult == REG_NOERROR)
        {
            // Get the path filename
            sFilename = subExpression(pPath, reMatch, FILE_FILENAME);
        }
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reFile, FILEEXPR));
    }
    return sFilename;

} // filename

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
