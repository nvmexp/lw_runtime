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
|*  Module: helper.cpp                                                        *|
|*                                                                            *|
 \****************************************************************************/
#include "precomp.h"
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <errno.h>

//******************************************************************************
//
// Locals
//
//******************************************************************************
static const CString s_sBoolTable[]     = {
/* true  */                                "True ",
/* false */                                "False",
                                          };

static const CString s_sEnableTable[]   = {
/* true  */                                "Enabled",
/* false */                                "Disabled",
                                          };

static const CString s_sActiveTable[]   = {
/* true  */                                "Active",
/* false */                                "Inactive",
                                          };

static const CString s_sErrorTable[]    = {
/* true  */                                "Error",
/* false */                                "Ok",
                                          };

static const ULONG s_ulTimeFactor[]     = {
/* Minutes */                              60,
/* Hours   */                              60,
/* Days    */                              24,
                                          };

static const CString s_sFactorTable[]   = {
/* Minutes */                              "min",
/* Hours   */                              "hr ",
/* Days    */                              "day",
                                          };

static const CString s_sTimeTable[]     = {
/* Seconds      */                         "sec",
/* Milliseconds */                         "ms ",
/* Microseconds */                         "us ",
/* Nanoseconds  */                         "ns ",
                                          };

static const CString s_sTimeColor[]     = {
/* Seconds      */                         RED,
/* Milliseconds */                         RED,
/* Microseconds */                         GREEN,
/* Nanoseconds  */                         BLUE,
                                          };

static const CString s_sFreqTable[]     = {
/* Hertz        */                         "Hz ",
/* Kilohertz    */                         "KHz",
/* Megahertz    */                         "MHz",
/* Gigahertz    */                         "Ghz",
                                          };

static const CString s_sPwrTable[]      = {
/* Watts      */                           "wt",
/* Milliwatts */                           "mw",
/* Microwatts */                           "uw",
/* Nanowatts  */                           "nw",
                                          };

static const CString s_sSizeTable[]     = {
/* Bytes     */                            "B ",
/* Kilobytes */                            "Kb",
/* Megabytes */                            "Mb",
/* Gigabytes */                            "Gb",
                                          };

static const CString s_sSizeColor[]     = {
/* Bytes     */                            BLUE,
/* Kilobytes */                            GREEN,
/* Megabytes */                            RED,
/* Gigabytes */                            RED,
                                          };

static const CString s_sDataTypeTable[] = {"Unknown",
                                           "Char",
                                           "Uchar",
                                           "Short",
                                           "Ushort",
                                           "Long",
                                           "Ulong",
                                           "Long64",
                                           "Ulong64",
                                           "Float",
                                           "Double",
                                           "Pointer32",
                                           "Pointer64",
                                           "Pointer",
                                           "Boolean",
                                           "Enum",
                                           "Struct",
                                          };

static  BOOLEAN_ENTRY s_BooleanTable[] = {{"YES",   true},
                                          {"NO",    false},
                                          {"ON",    true},
                                          {"OFF",   false},
                                          {"TRUE",  true},
                                          {"FALSE", false},
                                         };

static  VERBOSE_ENTRY s_VerboseTable[] = {
                                          {"CPU_PHYSICAL_READ",         VERBOSE_CPU_PHYSICAL_READ},
                                          {"CPU_PHYSICAL_WRITE",        VERBOSE_CPU_PHYSICAL_WRITE},
                                          {"CPU_VIRTUAL_READ",          VERBOSE_CPU_VIRTUAL_READ},
                                          {"CPU_VIRTUAL_WRITE",         VERBOSE_CPU_VIRTUAL_WRITE},

                                          {"DBGENG_CLIENT",             VERBOSE_DBGENG_CLIENT},
                                          {"DBGENG_CONTROL",            VERBOSE_DBGENG_CONTROL},
                                          {"DBGENG_DATA_SPACES",        VERBOSE_DBGENG_DATA_SPACES},
                                          {"DBGENG_REGISTERS",          VERBOSE_DBGENG_REGISTERS},
                                          {"DBGENG_SYMBOLS",            VERBOSE_DBGENG_SYMBOLS},
                                          {"DBGENG_SYSTEM_OBJECTS",     VERBOSE_DBGENG_SYSTEM_OBJECTS},
                                          {"DBGENG_ADVANCED",           VERBOSE_DBGENG_ADVANCED},
                                          {"DBGENG_SYMBOL_GROUP",       VERBOSE_DBGENG_SYMBOL_GROUP},
                                          {"DBGENG_BREAKPOINT",         VERBOSE_DBGENG_BREAKPOINT},

                                          {"DBGENG_PHYSICAL_READ",      VERBOSE_DBGENG_PHYSICAL_READ},
                                          {"DBGENG_PHYSICAL_WRITE",     VERBOSE_DBGENG_PHYSICAL_WRITE},
                                          {"DBGENG_VIRTUAL_READ",       VERBOSE_DBGENG_VIRTUAL_READ},
                                          {"DBGENG_VIRTUAL_WRITE",      VERBOSE_DBGENG_VIRTUAL_WRITE},

                                          {"DBGENG_IO_READ",            VERBOSE_DBGENG_IO_READ},
                                          {"DBGENG_IO_WRITE",           VERBOSE_DBGENG_IO_WRITE},

                                          {"DBGENG_INPUT",              VERBOSE_DBGENG_INPUT},
                                          {"DBGENG_OUTPUT",             VERBOSE_DBGENG_OUTPUT},
                                         };

static CString s_sNullString = "";

//******************************************************************************

bool
isLowercase
(
    const char         *pString
)
{
    CString             sString(pString);
    bool                bLowercase = false;

    assert(pString != NULL);

    // Colwert the given string to lowercase
    sString.lower();
    
    // Check to see if lowercase string matches original string
    if (sString.compare(pString) == 0)    
    {
        // Indicate the original string was lowercase
        bLowercase = true;
    }
    return bLowercase;

} // isLowercase

//******************************************************************************

bool
isUppercase
(
    const char         *pString
)
{
    CString             sString(pString);
    bool                bUppercase = false;

    assert(pString != NULL);

    // Colwert the given string to uppercase
    sString.upper();
    
    // Check to see if upercase string matches original string
    if (sString.compare(pString) == 0)    
    {
        // Indicate the original string was uppercase
        bUppercase = true;
    }
    return bUppercase;

} // isUppercase

//******************************************************************************

bool
isRegularExpression
(
    const char         *pString
)
{
    size_t              length;
    bool                bRegularExpression = false;

    assert(pString);

    // Check for regular expression starting character (Slash/Backslash)
    if ((pString[0] == SLASH) || (pString[0] == BACKSLASH))
    {
        // Get the length of the input string (to check end character)
        length = strlen(pString);

        // Check for a regular expression (Closing slash/backslash)
        if (pString[0] == pString[length - 1])
        {
            // Indicate this is a regular expression
            bRegularExpression = true;
        }
    }
    return bRegularExpression;

} // isRegularExpression

//******************************************************************************

CString
getRegularExpression
(
    const char         *pString
)
{
    size_t              length;
    CString             sRegularExpression;

    assert(pString);    

    // Check for a regular expression
    if (isRegularExpression(pString))
    {
        // Get the length of the input string
        length = strlen(pString);
        if (length > 2)
        {
            // Extract the actual regular expression string (Between the slashes)
            sRegularExpression = CString(pString, 1, length - 2);
        }
    }
    else    // Not a "normal" regular expression (Missing slashes)
    {
        sRegularExpression = CString(pString);
    }
    return sRegularExpression;

} // getRegularExpression

//******************************************************************************

bool
isStringExpression
(
    const char         *pString
)
{
    size_t              length;
    bool                bStringExpression = false;

    assert(pString);

    // Check for string expression starting character (Single Quote/Double Quote)
    if ((pString[0] == SINGLE_QUOTE) || (pString[0] == DOUBLE_QUOTE))
    {
        // Get the length of the input string (to check end character)
        length = strlen(pString);

        // Check for a string expression (Closing single quote/double quote)
        if (pString[0] == pString[length - 1])
        {
            // Indicate this is a string expression
            bStringExpression = true;
        }
    }
    return bStringExpression;

} // isStringExpression

//******************************************************************************

CString
getStringExpression
(
    const char         *pString
)
{
    size_t              length;
    CString             sStringExpression;

    assert(pString);    

    // Check for a string expression
    if (isStringExpression(pString))
    {
        // Get the length of the input string
        length = strlen(pString);
        if (length > 2)
        {
            // Extract the actual string expression string (Between the quotes)
            sStringExpression = CString(pString, 1, length - 2);
        }
    }
    else    // Not a "normal" string expression (Missing quotes)
    {
        sStringExpression = CString(pString);
    }
    return sStringExpression;

} // getStringExpression

//******************************************************************************

bool
getEnableValue
(
    bool                bEnable
)
{
    // Check for enable/disable set
    if (isOption(DisableOption) || isSearch(DisableOption) || isOption(EnableOption) || isSearch(EnableOption))
    {
        // Check for disable set
        if (isOption(DisableOption) || isSearch(DisableOption))
        {
            // Set enable to correct state (Based on disable option)
            if (isOption(DisableOption))
            {
                // Simply disable specified, set enable to false
                bEnable = false;
            }
            else if (isSearch(DisableOption))
            {
                // Disable value provided, set enable to not disable value
                bEnable = (searchValue(DisableOption) == 0);
            }
        }
        // Check for enable set
        if (isOption(EnableOption) || isSearch(EnableOption))
        {
            // Set enable to correct state (Based on enable option)
            if (isOption(EnableOption))
            {
                // Simply enable specified, set enable to true
                bEnable = true;
            }
            else if (isSearch(EnableOption))
            {
                // Enable value provided, set enable to enable value
                bEnable = (searchValue(EnableOption) != 0);
            }
        }
    }
    return bEnable;

} // getEnableValue

//******************************************************************************

bool
getDisableValue
(
    bool                bDisable
)
{
    // Check for enable/disable set
    if (isOption(DisableOption) || isSearch(DisableOption) || isOption(EnableOption) || isSearch(EnableOption))
    {
        // Check for disable set
        if (isOption(DisableOption) || isSearch(DisableOption))
        {
            // Set disable to correct state (Based on disable option)
            if (isOption(DisableOption))
            {
                // Simply disable specified, set disable to true
                bDisable = true;
            }
            else if (isSearch(DisableOption))
            {
                // Disable value provided, set disable to disable value
                bDisable = (searchValue(DisableOption) != 0);
            }
        }
        // Check for enable set
        if (isOption(EnableOption) || isSearch(EnableOption))
        {
            // Set disable to correct state (Based on enable option)
            if (isOption(EnableOption))
            {
                // Simply enable specified, set disable to false
                bDisable = false;
            }
            else if (isSearch(EnableOption))
            {
                // Enable value provided, set disable to not enable value
                bDisable = (searchValue(EnableOption) == 0);
            }
        }
    }
    return bDisable;

} // getDisableValue

//******************************************************************************

const CString&
getBoolString
(
    bool                bBool
)
{
    // Check for a true or false value
    if (bBool)
    {
        // Return the correct boolean string (True)
        return s_sBoolTable[0];
    }
    else    // False
    {
        // Return the correct boolean string (False)
        return s_sBoolTable[1];
    }

} // getBoolString

//******************************************************************************

const CString&
getEnabledString
(
    bool                bEnable
)
{
    // Check for an enabled or disabled value
    if (bEnable)
    {
        // Return the correct enabled string (Enabled)
        return s_sEnableTable[0];
    }
    else    // Disabled
    {
        // Return the correct enabled string (Disabled)
        return s_sEnableTable[1];
    }

} // getEnableString

//******************************************************************************

const CString&
getActiveString
(
    bool                bActive
)
{
    // Check for an active or inactive value
    if (bActive)
    {
        // Return the correct active string (Active)
        return s_sActiveTable[0];
    }
    else    // Inactive
    {
        // Return the correct active string (Inactive)
        return s_sActiveTable[1];
    }

} // getActiveString

//******************************************************************************

const CString&
getErrorString
(
    bool                bError
)
{
    // Check for a true or false value
    if (bError)
    {
        // Return the correct error string (Error)
        return s_sErrorTable[0];
    }
    else    // Ok
    {
        // Return the correct error string (Ok)
        return s_sErrorTable[1];
    }

} // getErrorString

//******************************************************************************

float
getTimeValue
(
    float               fTime
)
{
    ULONG               ulTime;

    // Check for time reduction/expansion
    if (fTime >= 999.9995)
    {
        // Loop until time value is in the right range (Reduction)
        for (ulTime = 0; ulTime < static_cast<ULONG>(countof(s_sFactorTable)); ulTime++)
        {
            // Move to the next time range
            fTime /= s_ulTimeFactor[ulTime];

            // Check for time in the correct range
            if (fTime < 999.9995)
            {
                break;
            }
        }
    }
    else    // Time expansion
    {
        // Loop until time value is in the right range (Expansion)
        for (ulTime = 0; ulTime < static_cast<ULONG>(countof(s_sTimeTable)); ulTime++)
        {
            // Check for time already in the correct range
            if (fTime >= 1.0)
            {
                break;
            }
            // Move to the next time range (Factor of 1000)
            fTime *= 1000.0;
        }
    }
    // Return the modified time value
    return fTime;

} // getTimeValue

//******************************************************************************

const CString&
getTimeUnit
(
    float               fTime
)
{
    ULONG               ulTime;

    // Check for time reduction/expansion
    if (fTime >= 999.9995)
    {
        // Loop until time value is in the right range (Reduction)
        for (ulTime = 0; ulTime < static_cast<ULONG>(countof(s_sFactorTable)); ulTime++)
        {
            // Move to the next time range
            fTime /= s_ulTimeFactor[ulTime];

            // Check for time in the correct range
            if (fTime < 999.9995)
            {
                break;
            }
        }
        // Check for no valid time range found (Just use last range)
        if (ulTime == static_cast<ULONG>(countof(s_sFactorTable)))
        {
            ulTime = static_cast<ULONG>(countof(s_sFactorTable) - 1);
        }
        return s_sFactorTable[ulTime];
    }
    else    // Time expansion
    {
        // Loop until time value is in the right range (Expansion)
        for (ulTime = 0; ulTime < static_cast<ULONG>(countof(s_sTimeTable)); ulTime++)
        {
            // Check for time already in the correct range
            if (fTime >= 1.0)
            {
                break;
            }
            // Move to the next time range (Factor of 1000)
            fTime *= 1000.0;
        }
        // Check for no valid time range found (Just use last range)
        if (ulTime == static_cast<ULONG>(countof(s_sTimeTable)))
        {
            ulTime = static_cast<ULONG>(countof(s_sTimeTable) - 1);
        }
        return s_sTimeTable[ulTime];
    }

} // getTimeUnit

//******************************************************************************

const CString&
getTimeColor
(
    float               fTime
)
{
    ULONG               ulTime;

    // Loop until time value is in the right range
    for (ulTime = 0; ulTime < static_cast<ULONG>(countof(s_sTimeTable)); ulTime++)
    {
        // Check for time already in the correct range
        if (fTime >= 1.0)
        {
            break;
        }
        // Move to the next time range (Factor of 1000)
        fTime *= 1000.0;
    }
    // Check for no valid time range found (Just use last range)
    if (ulTime == static_cast<ULONG>(countof(s_sTimeTable)))
    {
        ulTime = static_cast<ULONG>(countof(s_sTimeTable) - 1);
    }
    // Return the time color string
    return s_sTimeColor[ulTime];

} // getTimeColor

//******************************************************************************

float
getFreqValue
(
    float               fFreq
)
{
    ULONG               ulFreq;

    // Loop until frequency value is in the right range
    for (ulFreq = 0; ulFreq < static_cast<ULONG>(countof(s_sFreqTable)); ulFreq++)
    {
        // Check for frequency already in the correct range
        if (fFreq < 1000.0)
        {
            break;
        }
        // Move to the next frequency range (Factor of 1000)
        fFreq /= 1000.0;
    }
    // Return the modified frequency value
    return fFreq;

} // getFreqValue

//******************************************************************************

const CString&
getFreqUnit
(
    float               fFreq
)
{
    ULONG               ulFreq;

    // Loop until frequency value is in the right range
    for (ulFreq = 0; ulFreq < static_cast<ULONG>(countof(s_sFreqTable)); ulFreq++)
    {
        // Check for frequency already in the correct range
        if (fFreq < 1000.0)
        {
            break;
        }
        // Move to the next frequency range (Factor of 1000)
        fFreq /= 1000.0;
    }
    // Check for no valid frequency range found (Just use last range)
    if (ulFreq == static_cast<ULONG>(countof(s_sFreqTable)))
    {
        ulFreq = static_cast<ULONG>(countof(s_sFreqTable) - 1);
    }
    // Return the frequency unit string
    return s_sFreqTable[ulFreq];

} // getFreqUnit

//******************************************************************************

float
getPwrValue
(
    float               fPwr
)
{
    ULONG               ulPwr;

    // Loop until power value is in the right range
    for (ulPwr = 0; ulPwr < static_cast<ULONG>(countof(s_sPwrTable)); ulPwr++)
    {
        // Check for power already in the correct range
        if (fPwr >= 1.0)
        {
            break;
        }
        // Move to the next power range (Factor of 1000)
        fPwr *= 1000.0;
    }
    // Return the modified power value
    return fPwr;

} // getPwrValue

//******************************************************************************

const CString&
getPwrUnit
(
    float               fPwr
)
{
    ULONG               ulPwr;

    // Loop until power value is in the right range
    for (ulPwr = 0; ulPwr < static_cast<ULONG>(countof(s_sPwrTable)); ulPwr++)
    {
        // Check for power already in the correct range
        if (fPwr >= 1.0)
        {
            break;
        }
        // Move to the next power range (Factor of 1000)
        fPwr *= 1000.0;
    }
    // Check for no valid power range found (Just use last range)
    if (ulPwr == static_cast<ULONG>(countof(s_sPwrTable)))
    {
        ulPwr = static_cast<ULONG>(countof(s_sPwrTable) - 1);
    }
    // Return the power unit string
    return s_sPwrTable[ulPwr];

} // getPwrUnit

//******************************************************************************

float
getSizeValue
(
    float               fSize
)
{
    ULONG               ulSize;

    // Loop until size value is in the right range
    for (ulSize = 0; ulSize < static_cast<ULONG>(countof(s_sSizeTable)); ulSize++)
    {
        // Check for size already in the correct range (xxx.yy)
        if (fSize < 1000.0)
        {
            break;
        }
        // Move to the next size range (Factor of 1024)
        fSize /= 1024.0;
    }
    // Return the modified size value
    return fSize;

} // getSizeValue

//******************************************************************************

const CString&
getSizeUnit
(
    float               fSize
)
{
    ULONG               ulSize;

    // Loop until size value is in the right range
    for (ulSize = 0; ulSize < static_cast<ULONG>(countof(s_sSizeTable)); ulSize++)
    {
        // Check for size already in the correct range (xxx.yy)
        if (fSize < 1000.0)
        {
            break;
        }
        // Move to the next size range (Factor of 1024)
        fSize /= 1024.0;
    }
    // Check for no valid size range found (Just use last range)
    if (ulSize == static_cast<ULONG>(countof(s_sSizeTable)))
    {
        ulSize = static_cast<ULONG>(countof(s_sSizeTable) - 1);
    }
    // Return the size unit string
    return s_sSizeTable[ulSize];

} // getSizeUnit

//******************************************************************************

const CString&
getSizeColor
(
    float               fSize
)
{
    ULONG               ulSize;

    // Loop until size value is in the right range
    for (ulSize = 0; ulSize < static_cast<ULONG>(countof(s_sSizeTable)); ulSize++)
    {
        // Check for size already in the correct range (xxx.yy)
        if (fSize < 1000.0)
        {
            break;
        }
        // Move to the next size range (Factor of 1024)
        fSize /= 1024.0;
    }
    // Check for no valid size range found (Just use last range)
    if (ulSize == static_cast<ULONG>(countof(s_sSizeTable)))
    {
        ulSize = static_cast<ULONG>(countof(s_sSizeTable) - 1);
    }
    // Return the size color string
    return s_sSizeColor[ulSize];

} // getSizeColor

//******************************************************************************

const CString&
getDataTypeString
(
    DataType            dataType
)
{
    ULONG               ulIndex;

    // Compute index to the correct data type string
    ulIndex = min(static_cast<ULONG>(dataType), static_cast<ULONG>(countof(s_sDataTypeTable) - 1));

    // Return the correct data type string
    return s_sDataTypeTable[ulIndex];

} // getDataTypeString

//******************************************************************************

ULONG64
getDebuggerVersion()
{
    HMODULE             hModule;
    char                sFilename[MAX_PATH];
    DWORD               dwNameSize;
    DWORD               dwVersionSize;
    DWORD               dwHandle;
    UINT                uLength;
    BYTE               *pVersionInfo;
    VS_FIXEDFILEINFO   *pFixedFileInfo;
    ULONG64             ulDebuggerVersion = 0;

    // Try to get the module handle for the debugger engine module
    hModule = LoadLibrary(DEBUG_MODULE_NAME);
    if (hModule != NULL)
    {
        // Try to get the fully qualified path of the debugger engine module
        dwNameSize = GetModuleFileName(hModule, sFilename, sizeof(sFilename));
        if (dwNameSize != 0)
        {
            // Try to get the size of the file version information
            dwVersionSize = GetFileVersionInfoSize(sFilename, &dwHandle);
            if (dwVersionSize != 0)
            {
                // Try to allocate enough space to hold version information
                pVersionInfo = new BYTE[dwVersionSize];
                if (pVersionInfo != NULL)
                {
                    // Type to get the debugger engine file version information
                    if (GetFileVersionInfo(sFilename, dwHandle, dwVersionSize, pVersionInfo))
                    {
                        // Try to get the fixed file information pointer
                        if (VerQueryValue(pVersionInfo, "\\", reinterpret_cast<LPVOID*>(&pFixedFileInfo), &uLength))
                        {
                            // Build the debugger version value
                            ulDebuggerVersion = (static_cast<ULONG64>(pFixedFileInfo->dwFileVersionMS) << 32) + pFixedFileInfo->dwFileVersionLS;
                        }
                    }
                    // Free the file version information
                    delete [] pVersionInfo;
                    pVersionInfo = NULL;
                }
            }
        }
        // Free reference to debugger engine module
        FreeLibrary(hModule);
        hModule = NULL;
    }
    return ulDebuggerVersion;

} // getDebuggerVersion

//******************************************************************************

CString
getTemporaryPath()
{
    const char         *pTemporaryElw;
    ULONG               ulTemporaryLen;
    CString             sTemporaryPath(MAX_COMMAND_STRING);

    // Try to get the "TEMP" environment variable
    pTemporaryElw = getelw("TEMP");
    if (pTemporaryElw == NULL)
    {
        // No "TEMP" environment variable, try to get "TMP" environment variable
        pTemporaryElw = getelw("TMP");
    }
    // Check for temporary environment variable set (No path otherwise)
    if (pTemporaryElw != NULL)
    {
        // Get the temporary path length
        ulTemporaryLen = static_cast<ULONG>(strlen(pTemporaryElw));

        // Check for path termination character (Backslash)
        if (pTemporaryElw[ulTemporaryLen - 1] == '\\')
        {
            // Remove path termination character from temporary path
            sTemporaryPath = CString(pTemporaryElw, 0, (ulTemporaryLen - 1));
        }
        else    // No termination character
        {
            // Simply set the temporary path
            sTemporaryPath = CString(pTemporaryElw);
        }
    }
    return sTemporaryPath;

} // getTemporaryPath

//******************************************************************************

CString
subExpression
(
    const char         *pString,
    const regmatch_t   *pRegMatch,
    ULONG               ulSubExpression
)
{
    ULONG               ulLength;
    CString             sSubExpression;

    // Check for this subexpression present
    if (pRegMatch[ulSubExpression].rm_so != -1)
    {
        // Compute the length of this subexpression
        ulLength = pRegMatch[ulSubExpression].rm_eo - pRegMatch[ulSubExpression].rm_so;

        // Resize and assign subexpression string
        sSubExpression.resize(ulLength);
        sSubExpression.assign(pString, pRegMatch[ulSubExpression].rm_so, ulLength);
    }
    return sSubExpression;

} // subExpression

//******************************************************************************

CString
centerString
(
    const CString&      sString,
    ULONG               ulWidth
)
{
    ULONG               ulSize;
    ULONG               ulLength;
    ULONG               ulMaximum;
    ULONG               ulFront;
    ULONG               ulBack;
    CString             sCenterString;

    // Compute the size of the new string (Given string may be a DML string)
    ulSize = ulWidth + static_cast<ULONG>((strlen(sString) - dmllen(sString)));


    // Get the given string length (May be a DML string)
    ulLength = static_cast<ULONG>(dmllen(sString));

    // Get maximum header width (Either given width or string length)
    ulMaximum = max(ulWidth, ulLength);

    // Compute front and back spacing from maximum and string length
    ulFront = (ulMaximum - ulLength) / 2;
    ulBack  = (ulMaximum - ulLength) - ulFront;

    // Create the centered string
    sCenterString.append(ulFront, BLANK);
    sCenterString.append(sString);
    sCenterString.append(ulBack, BLANK);

    return sCenterString;

} // centerString

//******************************************************************************

void
headerString
(
    const CString&      sString,
    ULONG               ulWidth,
    CString&            sHeader,
    CString&            sDash,
    ULONG               ulSpacing
)
{
    ULONG               ulLength;
    ULONG               ulMaximum;
    ULONG               ulFront;
    ULONG               ulBack;

    // Setup any required spacing
    sHeader.append(ulSpacing, BLANK);
    sDash.append(ulSpacing, BLANK);

    // Get the given string length (May be a DML string)
    ulLength = static_cast<ULONG>(dmllen(sString));

    // Get maximum header width (Either given width or string length)
    ulMaximum = max(ulWidth, ulLength);

    // Compute front and back spacing from maximum and string length
    ulFront = (ulMaximum - ulLength) / 2;
    ulBack  = (ulMaximum - ulLength) - ulFront;

    // Update the header and dash strings for the given string
    sHeader.append(ulFront, BLANK);
    sHeader.append(sString);
    sHeader.append(ulBack, BLANK);
    sDash.append(ulMaximum, DASH);

} // headerString

//******************************************************************************

void
titleString
(
    const CString&      sString,
    ULONG               ulWidth,
    CString&            sTitle,
    ULONG               ulSpacing
)
{
    ULONG               ulLength;
    ULONG               ulMaximum;
    ULONG               ulFront;
    ULONG               ulBack;

    // Setup any required spacing
    sTitle.append(ulSpacing, BLANK);

    // Get the given string length (May be a DML string)
    ulLength = static_cast<ULONG>(dmllen(sString));

    // Get maximum title width (Either given width or string length)
    ulMaximum = max(ulWidth, ulLength);

    // Compute front and back spacing from maximum and string length
    ulFront = (ulMaximum - ulLength) / 2;
    ulBack  = (ulMaximum - ulLength) - ulFront;

    // Update the title string for the given string
    sTitle.append(ulFront, BLANK);
    sTitle.append(sString);
    sTitle.append(ulBack, BLANK);

} // titleString

//******************************************************************************

ULONG
fieldWidth
(
    const CMemberField& memberField
)
{
    // Return the field width (Hexadecimal characters)
    return (memberField.size() * 2);

} // fieldWidth

//******************************************************************************

ULONG
memberWidth
(
    const CMember&      member
)
{
    // Return the member width (Hexadecimal characters)
    return (member.field()->size() * 2);

} // memberWidth

//******************************************************************************

bool
fileExists
(
    const char         *pFile
)
{
    struct _stat        fileInfo;
    int                 nResult;
    bool                bExists = false;

    // Try to get information about the given file (Check for existence)
    nResult = _stat(pFile, &fileInfo);
    if (nResult == 0)
    {
        // Indicate that the file exists
        bExists = true;
    }
    return bExists;

} // fileExists

//******************************************************************************

ULONG64
elapsedTime
(
    ULONG64             ulStartTime,
    ULONG64             ulEndTime
)
{
    LONG64              lElapsedTime;

    // Compute the elapsed time as a signed value (In case of wrapping)
    lElapsedTime = ulEndTime - ulStartTime;

    // Return the absolute value of the time value to handle a race condition
    // case in the driver logging code where the event ID got assigned but
    // the code was then interrupted *before* the timer count be read. This
    // means the start time is actually greater than the end time but they
    // should really be more reversed (which is really what the abs accomplishes)
    return _abs64(lElapsedTime);

} // elapsedTime

//******************************************************************************

ULONG64
extractBitfield
(
    ULONG64             ulValue,
    ULONG               ulPosition,
    ULONG               ulWidth
)
{
    ULONG               bitfieldMask;

    assert((ulPosition < 63) && ((ulPosition + ulWidth) <= 64));

    // Compute the bitfield mask
    bitfieldMask = (1 << ulWidth) - 1;

    // Mask off and shift bitfield into position
    ulValue  &= bitfieldMask << ulPosition;
    ulValue >>= ulPosition;

    // Return the extracted bitfield
    return ulValue;

} // extractBitfield

//******************************************************************************

ULONG64
boolealwalue
(
    const char         *pBooleanString
)
{
    regex_t             reBoolean = {0};
    regmatch_t          reMatch[10];
    ULONG               ulBooleanEntry;
    int                 reResult;
    ULONG64             ulValue = ILWALID_BOOLEAN_VALUE;

    assert(pBooleanString != NULL);

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reBoolean, pBooleanString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Loop checking all the boolean entries
        for (ulBooleanEntry = 0; ulBooleanEntry < countof(s_BooleanTable); ulBooleanEntry++)
        {
            // Compare the given string and boolean value string
            reResult = regexec(&reBoolean, s_BooleanTable[ulBooleanEntry].pString, countof(reMatch), reMatch, 0);
            if (reResult == REG_NOERROR)
            {
                // Set boolean value to matching value
                ulValue = s_BooleanTable[ulBooleanEntry].bValue;

                // Exit the search
                break;
            }
        }
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reBoolean, pBooleanString));
    }
    return ulValue;

} // boolealwalue

//******************************************************************************

ULONG64
verboseValue
(
    const char         *pVerboseString
)
{
    regex_t             reVerbose = {0};
    regmatch_t          reMatch[10];
    ULONG               ulVerboseEntry;
    int                 reResult;
    ULONG64             ulValue = ILWALID_VERBOSE_VALUE;

    assert(pVerboseString != NULL);

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reVerbose, pVerboseString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Loop checking all the verbose entries
        for (ulVerboseEntry = 0; ulVerboseEntry < countof(s_VerboseTable); ulVerboseEntry++)
        {
            // Compare the given string and verbose value string
            reResult = regexec(&reVerbose, s_VerboseTable[ulVerboseEntry].pString, countof(reMatch), reMatch, 0);
            if (reResult == REG_NOERROR)
            {
                // Check for first matching verbose value
                if (ulValue == ILWALID_VERBOSE_VALUE)
                {
                    // Set verbose value to initial matching value
                    ulValue = s_VerboseTable[ulVerboseEntry].ulValue;
                }
                else    // Not the first matching verbose value
                {
                    // Logically OR in the next matching verbose value
                    ulValue |= s_VerboseTable[ulVerboseEntry].ulValue;
                }
            }
        }
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reVerbose, pVerboseString));
    }
    return ulValue;

} // verboseValue

//******************************************************************************

ULONG64
factorial
(
    ULONG               ulValue
)
{
    ULONG64             ulFactorial = 1;

    // Loop computing the factorial value
    while (ulValue > 1)
    {
        ulFactorial *= ulValue;
        ulValue--;
    }
    return ulFactorial;

} // factorial

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
