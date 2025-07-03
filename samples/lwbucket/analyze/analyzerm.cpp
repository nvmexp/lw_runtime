 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2020 LWPU Corporation.  All rights reserved.         *|
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
|*  Module: analyzerm.cpp                                                     *|
|*                                                                            *|
 \****************************************************************************/
#include "alzprecomp.h"
#include "analyzehelper.h"

//******************************************************************************
//
//  Forwards
//
//******************************************************************************
static  HRESULT         decodeProtoBuffer();

static  CString         gpuFamily(ULONG ulArchitecture);

static  HRESULT         setGpuInfoTags(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags *pTagControl);
static  HRESULT         setUserOverclockedTags(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags *pTagControl);

static  HRESULT         setRmTagString(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags *pTagControl, RM_ANALYSIS_TAG tag, CString sString);
static  HRESULT         setRmTagUlong(PDEBUG_FAILURE_ANALYSIS2  pAnalysis, IDebugFAEntryTags *pTagControl, RM_ANALYSIS_TAG tag, ULONG ulValue);

static  bool            prbGetValue_LwU32(LwU32* pValue, const PRB_MSG* pMsg, const PRB_FIELD_DESC* pFieldDesc);
static  bool            prbGetValue_LwS32(LwS32* pValue, const PRB_MSG* pMsg, const PRB_FIELD_DESC* pFieldDesc);
static  bool            prbGetCount_LwS32(LwS32* pCount, const PRB_MSG* pMsg, const PRB_FIELD_DESC* pFieldDesc);

//******************************************************************************
//
//  Locals
//
//******************************************************************************
// RM custom tag names and descriptions (Should match what's in the ALZ file)
static const TAG_ENTRY s_rmTagTable[] = {
/* 0xa8100000 */    {LWIDIA_ANALYSIS_TAG_GPU_FAMILY,              "LWIDIA_ANALYSIS_TAG_GPU_FAMILY",              "GPU Family Name"},
/* 0xa8100001 */    {LWIDIA_ANALYSIS_TAG_GPU_TEMPERATURE,         "LWIDIA_ANALYSIS_TEMPERATURE",                 "GPU Temperature (Celsius)"},
/* 0xa8100002 */    {LWIDIA_ANALYSIS_TAG_USER_OVERCLOCKED,        "LWIDIA_ANALYSIS_TAG_USER_OVERCLOCKED",        "User Overclocked"},
                                        };

static const char      *s_szGpuFamily[] = {
/* 0x00 Lw0x                */             "Legacy",
/* 0x01 LW1x                */             "Celsius",
/* 0x02 LW2x                */             "Kelvin",
/* 0x03 LW3x                */             "Rankine",
/* 0x04 LW4x                */             "Lwrie",
/* 0x05 G80 (LW50)          */             "Tesla",
/* 0x06 G78, MCP67, MCP73   */             "Lwrie",
/* 0x07                     */             "Unknown",
/* 0x08 G8x                 */             "Tesla",
/* 0x09 G9x                 */             "Tesla",
/* 0x0a GT21x               */             "Tesla2",
/* 0x0b                     */             "Unknown",
/* 0x0c GF10x               */             "Fermi",
/* 0x0d GF11x               */             "Fermi",
/* 0x0e GK10x, GK20a        */             "Kepler",
/* 0x0f GK110, GK180, GK210 */             "Kepler",
/* 0x10 GK20x               */             "Kepler",
/* 0x11 GM10x               */             "Maxwell",
/* 0x12 GM20x               */             "Maxwell",
/* 0x13 GP10x               */             "Pascal",
/* 0x14 GV10x               */             "Volta",
/* 0x15 GV11x               */             "Volta",
/* 0x16 TU10x               */             "Turing",
/* 0x17 GA10x               */             "Ampere",
                                          };

// Static data
static PRB_MSG          s_dumpMsg = {NULL};

//******************************************************************************

HRESULT
setRmTags
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl
)
{
    HRESULT          hResult  = S_FALSE;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    // Try to decode the protobuffer
    hResult = decodeProtoBuffer();
    if (SUCCEEDED(hResult))
    {
        // Try to set the GPU information tags
        hResult = setGpuInfoTags(pAnalysis, pTagControl);
        if (SUCCEEDED(hResult))
            hResult = setUserOverclockedTags(pAnalysis, pTagControl);
        else
            setUserOverclockedTags(pAnalysis, pTagControl);
    }
    return hResult;

} // setRmTags

//******************************************************************************

LWD_PRINTF lwdPrintf;

// Redirect protobuf output to extension output
static int
lwdPrintfWrapper(const char *szFormat, ...)
{
    UNREFERENCED_PARAMETER(szFormat);

    int                 nReturn = 0;
    
// Drop all prbDecode output.  Set 1 to debugging issues.
#if 0
    char                szBuffer[1024];
    va_list             argList;

    // Setup argument list and format string
    va_start(argList, szFormat);

    // Perform the formatted print
    nReturn = vsnprintf(szBuffer, countof(szBuffer), szFormat, argList);

    // Terminate the variable argument list
    va_end(argList);

    // Check for characters to output
    if (nReturn > 0)
        dprintf("prbdec: %s", szBuffer);
#endif
    // Return the output count
    return nReturn;

} // lwdPrintfWrapper

//******************************************************************************

static HRESULT
decodeProtoBuffer()
{
    PRB_STATUS          prbStatus;
    ULONG               ulSize;
    const void         *pBuffer;
    HRESULT             hResult = S_FALSE;

    // Set the printf funtion pointer that lwDump uses.
    lwdPrintf = lwdPrintfWrapper;

    // Initialize the protobuf pointer and size
    pBuffer = rmProtoBufData();
    ulSize  = rmProtoBufSize();

    // Make sure protobuf is available
    if (pBuffer != NULL)
    {
        // Decode the protobuf message
        prbCreateMsg(&s_dumpMsg, LWDEBUG_LWDUMP);
        prbStatus = prbDecodeMsg(&s_dumpMsg, pBuffer, ulSize);
        if (prbStatus == PRB_OK)
        {
            // Indicate protobuf was decoded correctly
            hResult = S_OK;
        }
        else    // Error decoding protobuf
        {
            // Indicate invalid protobuf
            hResult = E_ILWALIDARG;
        }
    }
    return hResult;

} // decodeProtoBuffer

//******************************************************************************

static CString
gpuFamily
(
    ULONG               ulArchitecture
)
{
    CString             sGpuFamily("Unknown");

    // Check for a valid architecture
    if (ulArchitecture < countof(s_szGpuFamily))
    {
        // Get the GPU architecture family string
        sGpuFamily = s_szGpuFamily[ulArchitecture];
    }
    return sGpuFamily;

} // gpuFamily

//******************************************************************************

static HRESULT
setGpuInfoTags
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags       *pTagControl
)
{
    const PRB_MSG      *pMsgGpuInfo = prbGetMsg(&s_dumpMsg, LWDEBUG_SYSTEMINFO_GPUINFO);
    CString             sGpuFamily;
    LwU32               uJunctionTemp;
    LwU32               uPmcBoot0;
    LwU32               uArchitecture;
    HRESULT             hResult = S_FALSE;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    // Check for GPU information available
    if (pMsgGpuInfo != NULL)
    {
        // Try to get the PMC_BOOT0 value (message SystemInfo / message GpuInfo / uint32 pmc_boot0)
        if (prbGetValue_LwU32(&uPmcBoot0, pMsgGpuInfo, LWDEBUG_SYSTEMINFO_GPUINFO_PMCBOOT0))
        {
            // Extract the GPU architecture from PMC_BOOT0
            uArchitecture   = (uPmcBoot0 >> 24) & 0x1f;

            // Get the GPU family for this architecture
            sGpuFamily = gpuFamily(uArchitecture);
            if (!sGpuFamily.empty())
            {
                // Set RM TAG string for GPU family
                hResult = setRmTagString(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_GPU_FAMILY, sGpuFamily);
            }
        }
        // Try to get the GPU junction temperature (message SystemInfo / message GpuInfo / uint32 junction_temp)
        if (prbGetValue_LwU32(&uJunctionTemp, pMsgGpuInfo, LWDEBUG_SYSTEMINFO_GPUINFO_JUNCTION_TEMP))
        {
            // Get the GPU junction temperature (Round off to the nearest degree)
            uJunctionTemp = (uJunctionTemp + 128) / 256;

            // Set RM TAG value for GPU temperature
            if (SUCCEEDED(hResult))
                hResult = setRmTagUlong(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_GPU_TEMPERATURE, uJunctionTemp);
            else
                setRmTagUlong(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_GPU_TEMPERATURE, uJunctionTemp);
        }
    }

    return hResult;

} // setGpuInfoTags

//******************************************************************************

static HRESULT
setUserOverclockedTags
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags* pTagControl
)
{
    const PRB_MSG      *pMsgEngPerf = prbGetMsg(&s_dumpMsg, LWDEBUG_ENG_PERF);
    const PRB_MSG      *pMsgEngPerfClockProg = NULL;
    const PRB_MSG      *pMsgEngPerfOverclockState = NULL;
    CString             sUserOverclocked("");
    LwS32               nCount, i;
    LwS32               nOverclockStateUserOffset = 0;
    LwS32               nClkprogDelta = 0;
    HRESULT             hResult = S_FALSE;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    // Check for ENG PERF information available
    if (pMsgEngPerf != NULL)
    {
        if (prbGetCount_LwS32(&nCount, pMsgEngPerf, LWDEBUG_ENG_PERF_OVERCLOCK_STATE))
        {
            PRB_FIELD* prb_fields_lwdebug_eng_perf = pMsgEngPerf->fields;
            for (i = 0; i < nCount; i++)
            {
                pMsgEngPerfOverclockState = (PRB_MSG*)(LWDEBUG_ENG_PERF_OVERCLOCK_STATE->values[i].message.data);
                if (prbGetValue_LwS32(&nOverclockStateUserOffset, pMsgEngPerfOverclockState, LWDEBUG_ENG_PERF_OVERCLOCKSTATE_USER_OFFSET))
                {
                    if (nOverclockStateUserOffset > 0)
                    {
                        sUserOverclocked = "UserOC";
                        break;
                    }
                }

                pMsgEngPerfClockProg = prbGetMsg(pMsgEngPerfOverclockState, LWDEBUG_ENG_PERF_OVERCLOCKSTATE_CLKPROG);
                if (pMsgEngPerfClockProg)
                {
                    if (prbGetValue_LwS32(&nClkprogDelta, pMsgEngPerfClockProg, LWDEBUG_ENG_PERF_OVERCLOCKSTATE_CLKPROG_AVG_DELTA))
                    {
                        if (nClkprogDelta > 0)
                        {
                            sUserOverclocked = "UserOC";
                            break;
                        }
                    }
                    if (prbGetValue_LwS32(&nClkprogDelta, pMsgEngPerfClockProg, LWDEBUG_ENG_PERF_OVERCLOCKSTATE_CLKPROG_MIN_DELTA))
                    {
                        if (nClkprogDelta > 0)
                        {
                            sUserOverclocked = "UserOC";
                            break;
                        }
                    }
                    if (prbGetValue_LwS32(&nClkprogDelta, pMsgEngPerfClockProg, LWDEBUG_ENG_PERF_OVERCLOCKSTATE_CLKPROG_MAX_DELTA))
                    {
                        if (nClkprogDelta > 0)
                        {
                            sUserOverclocked = "UserOC";
                            break;
                        }
                    }
                }
            }

            // Set RM TAG string for OverClock
            if (!sUserOverclocked.empty())
                hResult = setRmTagString(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_USER_OVERCLOCKED, sUserOverclocked);
        }
    }
    return hResult;

} // setUserOverclockedTags

//******************************************************************************

static HRESULT
setRmTagString
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl,
    RM_ANALYSIS_TAG     tag,
    CString             sString
)
{
    ULONG               ulIndex;
    const TAG_ENTRY    *pTagEntry;
    HRESULT             hResult = S_FALSE;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    if (!sString.empty())
    {
        // Search RM tag table for matching tag
        for (ulIndex = 0; ulIndex < countof(s_rmTagTable); ulIndex++)
        {
            // Get the next tag table entry (Check for tag match)
            pTagEntry = &s_rmTagTable[ulIndex];
            if (pTagEntry->tag == tag)
            {
                // Try to set the RM tag string (w/name and description)
                hResult = setTagString(pAnalysis, pTagControl, static_cast<FA_TAG>(tag), sString, pTagEntry->szTagName, pTagEntry->szTagDescription);

                break;
            }
        }
    }
    return hResult;

} // setRmTagString

//******************************************************************************

static HRESULT
setRmTagUlong
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl,
    RM_ANALYSIS_TAG     tag,
    ULONG               ulValue
)
{
    ULONG               ulIndex;
    const TAG_ENTRY    *pTagEntry;
    HRESULT             hResult = S_FALSE;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    // Search RM tag table for matching tag
    for (ulIndex = 0; ulIndex < countof(s_rmTagTable); ulIndex++)
    {
        // Get the next tag table entry (Check for tag match)
        pTagEntry = &s_rmTagTable[ulIndex];
        if (pTagEntry->tag == tag)
        {
            // Try to set the RM tag value (w/name and description)
            hResult = setTagUlong(pAnalysis, pTagControl, static_cast<FA_TAG>(tag), ulValue, pTagEntry->szTagName, pTagEntry->szTagDescription);

            break;
        }
    }
    return hResult;

} // setRmTagUlong

//******************************************************************************

static bool
prbGetValue_LwU32
(
    LwU32                *pValue,
    const PRB_MSG        *pMsg,
    const PRB_FIELD_DESC *pFieldDesc
)
{
    PRB_FIELD const *pField;

    IF_NULL_RETURN(pValue);
    IF_NULL_RETURN(pMsg);
    IF_NULL_RETURN(pFieldDesc);

    pField = prbGetField(pMsg, pFieldDesc);
    if ((pField == NULL) || (pField->values == NULL))
    {
        *pValue = 0;
        return false;
    }
    else
    {
        *pValue = pField->values[0].uint32;
        return true;
    }
} // prbGetValue_LwU32

//******************************************************************************

static bool
prbGetValue_LwS32
(
    LwS32                *pValue,
    const PRB_MSG        *pMsg,
    const PRB_FIELD_DESC *pFieldDesc
)
{
    PRB_FIELD const *pField;

    IF_NULL_RETURN(pValue);
    IF_NULL_RETURN(pMsg);
    IF_NULL_RETURN(pFieldDesc);

    pField = prbGetField(pMsg, pFieldDesc);
    if ((pField == NULL) || (pField->values == NULL))
    {
        *pValue = 0;
        return false;
    }
    else
    {
        *pValue = pField->values[0].int32;
        return true;
    }
} // prbGetValue_LwU32

//******************************************************************************

static bool
prbGetCount_LwS32
(
    LwS32                *pCount,
    const PRB_MSG        *pMsg,
    const PRB_FIELD_DESC *pFieldDesc
)
{
    PRB_FIELD const *pField;

    IF_NULL_RETURN(pCount);
    IF_NULL_RETURN(pMsg);
    IF_NULL_RETURN(pFieldDesc);

    pField = prbGetField(pMsg, pFieldDesc);
    if ((pField == NULL) || (pField->count == NULL))
    {
        *pCount = 0;
        return false;
    }
    else
    {
        *pCount = pField->count;
        return true;
    }
} // prbGetCount_LwS32

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
