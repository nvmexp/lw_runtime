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
|*  Module: analyzekmd.cpp                                                    *|
|*                                                                            *|
 \****************************************************************************/
#include "alzprecomp.h"
#include "analyzehelper.h"

//******************************************************************************
//
//  Forwards
//
//******************************************************************************
static  HRESULT         setKmdString(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl, KMD_ANALYSIS_TAG tag, CString sString);
static  HRESULT         setKmdUlong(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl, KMD_ANALYSIS_TAG tag, ULONG ulValue);

static  HRESULT         setKmdAdapterTags(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl, const CAdapterOcaRecord* pAdapterOcaRecord);
static  HRESULT         setKmdEngineTags(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl, const CEngineIdOcaRecord* pEngineIdOcaRecord);
static  HRESULT         setKmdProcessTags(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl, const CKmdProcessOcaRecord* pKmdProcessOcaRecord);
static  HRESULT         setKmdWatchdogTags(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl, const CGpuWatchdogEvent* pGpuWatchdogEvent);
static  HRESULT         setKmdVsyncTags(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl, float fVsyncPeriod);

//******************************************************************************
//
//  Locals
//
//******************************************************************************
// KMD custom tag names and descriptions (Should match what's in the ALZ file)
static const TAG_ENTRY s_kmdTagTable[] = {
/* 0xa8000000 */                          {LWIDIA_ANALYSIS_TAG_TDR_PROCESS,         "LWIDIA_ANALYSIS_TAG_TDR_PROCESS",          "TDR Process Name"},
/* 0xa8000001 */                          {LWIDIA_ANALYSIS_TAG_TDR_ENGINE,          "LWIDIA_ANALYSIS_TAG_TDR_ENGINE",           "TDR Engine Name"},
/* 0xa8000002 */                          {LWIDIA_ANALYSIS_TAG_TDR_COUNT,           "LWIDIA_ANALYSIS_TAG_TDR_COUNT",            "TDR Count"},
/* 0xa8000003 */                          {LWIDIA_ANALYSIS_TAG_LATE_BUFFER_COUNT,   "LWIDIA_ANALYSIS_TAG_LATE_BUFFER_COUNT",    "Late Buffer Count"},
/* 0xa8000004 */                          {LWIDIA_ANALYSIS_TAG_BUFFER_ERROR_COUNT,  "LWIDIA_ANALYSIS_TAG_BUFFER_ERROR_COUNT",   "Buffer Error Count"},
                                         };

//******************************************************************************

HRESULT
setKmdTags
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl
)
{
    COcaDataPtr         pOcaData = ocaData();
    const COcaGuid     *pOcaGuid;
    const CLwcdHeader  *pLwcdHeader;
    const CAdapterOcaRecord *pAdapterOcaRecord;
    const CEngineIdOcaRecord *pEngineIdOcaRecord;
    const CDeviceOcaRecord *pDeviceOcaRecord;
    const CContextOcaRecord *pContextOcaRecord;
    const CKmdProcessOcaRecord *pKmdProcessOcaRecord;
    const CVblankInfoOcaRecord *pVblankInfoOcaRecord;
    const CVblankInfoData *pVblankInfoData;
    const CGpuWatchdogEvent *pGpuWatchdogEvent = NULL;
    const CBufferInfoRecord *pBufferInfoRecord = NULL;
    CString             sEngine;
    CString             sProcess;
    ULONG               ulBufferInfo;
    ULONG               ulVblankData;
    ULONG64             ulElapsedTime;
    ULONG64             ulLastTime = 0;
    float               fElapsedTime = 0.0;
    float               fVsyncPeriod = 0.0;
    HRESULT             hResult = S_OK;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    // Check for OCA data present
    if (pOcaData != NULL)
    {
        // Get the OCA data header
        pLwcdHeader = pOcaData->lwcdHeader();
        if (pLwcdHeader != NULL)
        {
            // Check for valid OCA data signature
            if (pLwcdHeader->dwSignature() == LWCD_SIGNATURE)
            {
                // Get the OCA header vesion GUID
                pOcaGuid = pLwcdHeader->gVersion();
                if (pOcaGuid != NULL)
                {
                    // Check for known KMD OCA data version
                    if ((memcmp(guidLwcd1(), pOcaGuid->ocaGuid(), sizeof(GUID)) == 0) ||
                        (memcmp(guidLwcd2(), pOcaGuid->ocaGuid(), sizeof(GUID)) == 0))
                    {
                        // Try to find the OCA adapter in TDR
                        pAdapterOcaRecord = findTdrAdapter();
                        if (pAdapterOcaRecord != NULL)
                        {
                            // Set the KMD tags for this adapter (Record error if no other errors)
                            if (SUCCEEDED(hResult))
                                hResult = setKmdAdapterTags(pAnalysis, pTagControl, pAdapterOcaRecord);
                            else
                                setKmdAdapterTags(pAnalysis, pTagControl, pAdapterOcaRecord);

                            // Try to find the TDR engine for this adapter (but *not* ignoring preemption)
                            pEngineIdOcaRecord = findTdrEngine(pAdapterOcaRecord, false);
                            if (pEngineIdOcaRecord == NULL)
                            {
                                // Try again but *ignore* preemption this time (Since we didn't find one in preemption)
                                pEngineIdOcaRecord = findTdrEngine(pAdapterOcaRecord, true);
                            }
                            // Check to see if we found the faulting engine
                            if (pEngineIdOcaRecord != NULL)
                            {
                                // Set the KMD tags for this engine (Record error if no other errors)
                                if (SUCCEEDED(hResult))
                                    hResult = setKmdEngineTags(pAnalysis, pTagControl, pEngineIdOcaRecord);
                                else
                                    setKmdEngineTags(pAnalysis, pTagControl, pEngineIdOcaRecord);
                            }
                            // Try to find the TDR context for this adapter
                            pContextOcaRecord = findTdrContext(pAdapterOcaRecord);
                            if (pContextOcaRecord != NULL)
                            {
                                // Try to find the device for this TDR context
                                pDeviceOcaRecord = findOcaDevice(pContextOcaRecord->Device().ptr());
                                if (pDeviceOcaRecord != NULL)
                                {
                                    // Try to find the OCA process for this TDR device
                                    pKmdProcessOcaRecord = findOcaProcess(pDeviceOcaRecord->Process().ptr());
                                    if (pKmdProcessOcaRecord == NULL)
                                    {
                                        // Try to find the KMD process for this TDR device
                                        pKmdProcessOcaRecord = findOcaKmdProcess(pDeviceOcaRecord->KmdProcess().ptr());
                                    }
                                    // Check to see if we found the faulting application process
                                    if (pKmdProcessOcaRecord != NULL)
                                    {
                                        // Set the KMD tags for this process (Record error if no other errors)
                                        if (SUCCEEDED(hResult))
                                            hResult = setKmdProcessTags(pAnalysis, pTagControl, pKmdProcessOcaRecord);
                                        else
                                            setKmdProcessTags(pAnalysis, pTagControl, pKmdProcessOcaRecord);
                                    }
                                }
                            }
                        }
                        else   // Unable to find OCA Adapter in TDR
                        {
                            // Check to see if there is a GPU watchdog event (Possible TDR source)
                            pGpuWatchdogEvent = findGpuWatchdogEvent();
                            if (pGpuWatchdogEvent != NULL)
                            {
                                // Set the KMD tags for this watchdog event (Record error if no other errors)
                                if (SUCCEEDED(hResult))
                                    hResult = setKmdWatchdogTags(pAnalysis, pTagControl, pGpuWatchdogEvent);
                                else
                                    setKmdWatchdogTags(pAnalysis, pTagControl, pGpuWatchdogEvent);

                                // Try to find the OCA adapter for this GPU watchdog event
                                pAdapterOcaRecord = findOcaAdapter(pGpuWatchdogEvent->AdapterOrdinal());
                                if (pAdapterOcaRecord != NULL)
                                {
                                    // Set the KMD tags for this adapter (Record error if no other errors)
                                    if (SUCCEEDED(hResult))
                                        hResult = setKmdAdapterTags(pAnalysis, pTagControl, pAdapterOcaRecord);
                                    else
                                        setKmdAdapterTags(pAnalysis, pTagControl, pAdapterOcaRecord);

                                    // Try to find the engine ID record for this GPU watchdog event
                                    pEngineIdOcaRecord = findEngineId(pGpuWatchdogEvent->AdapterOrdinal(), pGpuWatchdogEvent->EngineOrdinal());
                                    if (pEngineIdOcaRecord != NULL)
                                    {
                                        // Set the KMD tags for this engine (Record error if no other errors)
                                        if (SUCCEEDED(hResult))
                                            hResult = setKmdEngineTags(pAnalysis, pTagControl, pEngineIdOcaRecord);
                                        else
                                            setKmdEngineTags(pAnalysis, pTagControl, pEngineIdOcaRecord);

                                        // Loop searching the engine ID buffer info records for GPU watchdog event buffer
                                        for (ulBufferInfo = 0; ulBufferInfo < pEngineIdOcaRecord->bufferInfoCount(); ulBufferInfo++)
                                        {
                                            // Get the next buffer info record to check
                                            pBufferInfoRecord = pEngineIdOcaRecord->bufferInfoRecord(ulBufferInfo);
                                            if (pBufferInfoRecord != NULL)
                                            {
                                                // Check to see if this is the buffer for the GPU watchdog event
                                                if (pBufferInfoRecord->FenceId() == pGpuWatchdogEvent->FenceId())
                                                {
                                                    // Break out of the search loop
                                                    break;
                                                }
                                                else    // Not the matching buffer
                                                {
                                                    // Clear the buffer info record
                                                    pBufferInfoRecord = NULL;
                                                }
                                            }
                                        }
                                        // Check to see if buffer info record found for GPU watchdog event
                                        if (pBufferInfoRecord != NULL)
                                        {
                                            // Try to find the TDR context for this buffer info
                                            pContextOcaRecord = findOcaContext(pBufferInfoRecord->Context().ptr());
                                            if (pContextOcaRecord != NULL)
                                            {
                                                // Try to find the device for this TDR context
                                                pDeviceOcaRecord = findOcaDevice(pContextOcaRecord->Device().ptr());
                                                if (pDeviceOcaRecord != NULL)
                                                {
                                                    // Try to find the OCA process for this TDR device
                                                    pKmdProcessOcaRecord = findOcaProcess(pDeviceOcaRecord->Process().ptr());
                                                    if (pKmdProcessOcaRecord == NULL)
                                                    {
                                                        // Try to find the KMD process for this TDR device
                                                        pKmdProcessOcaRecord = findOcaKmdProcess(pDeviceOcaRecord->KmdProcess().ptr());
                                                    }
                                                    // Check to see if we found the faulting application process
                                                    if (pKmdProcessOcaRecord != NULL)
                                                    {
                                                        // Set the KMD tags for this process (Record error if no other errors)
                                                        if (SUCCEEDED(hResult))
                                                            hResult = setKmdProcessTags(pAnalysis, pTagControl, pKmdProcessOcaRecord);
                                                        else
                                                            setKmdProcessTags(pAnalysis, pTagControl, pKmdProcessOcaRecord);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Check for possible VSYNC TDR
                        pVblankInfoOcaRecord = findTdrVsync();
                        if (pVblankInfoOcaRecord != NULL)
                        {
                            // Loop searching for largest VSYNC period (Should be > TDR timeout)
                            for (ulVblankData = 0; ulVblankData < pVblankInfoOcaRecord->vblankDataCount(); ulVblankData++)
                            {
                                // Get the next vblank info data to check
                                pVblankInfoData = pVblankInfoOcaRecord->vblankInfoData(ulVblankData);
                                if (pVblankInfoData != NULL)
                                {
                                    // Make sure not the first vblank data record (No elapsed time value)
                                    if (ulVblankData != 0)
                                    {
                                        // Compute the elapsed time since last vblank
                                        ulElapsedTime = elapsedTime(pVblankInfoData->timestamp(), ulLastTime);

                                        // Compute the elapsed time value
                                        fElapsedTime = floatunits(ulElapsedTime, pVblankInfoOcaRecord->frequency());

                                        // Check for new largest VSYNC period
                                        if (fElapsedTime > fVsyncPeriod)
                                        {
                                            // Update the largest VSYNC period
                                            fVsyncPeriod = fElapsedTime;
                                        }
                                    }
                                    ulLastTime = pVblankInfoData->timestamp();
                                }
                            }
                            // Call routine to set the KMD Vsync tags (Record error if no other errors)
                            if (SUCCEEDED(hResult))
                                hResult = setKmdVsyncTags(pAnalysis, pTagControl, fVsyncPeriod);
                            else
                                setKmdVsyncTags(pAnalysis, pTagControl, fVsyncPeriod);
                        }
                    }
                }
            }
        }
    }
    return hResult;

} // setKmdTags

//******************************************************************************

static HRESULT
setKmdAdapterTags
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl,
    const CAdapterOcaRecord* pAdapterOcaRecord
)
{
    HRESULT             hResult = S_OK;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);
    IF_NULL_RETURN(pAdapterOcaRecord);

    // Try to set the TDR count tag value (Record error if no other errors)
    hResult = setKmdUlong(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_TDR_COUNT, pAdapterOcaRecord->TDRCount());

    // Try to set the late buffer count tag value (Record error if no other errors)
    if (SUCCEEDED(hResult))
        hResult = setKmdUlong(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_LATE_BUFFER_COUNT, pAdapterOcaRecord->LateBufferCompletionCount());
    else
        setKmdUlong(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_LATE_BUFFER_COUNT, pAdapterOcaRecord->LateBufferCompletionCount());

    // Try to set the buffer error count tag value (Record error if no other errors)
    if (SUCCEEDED(hResult))
        hResult = setKmdUlong(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_BUFFER_ERROR_COUNT, pAdapterOcaRecord->BufferSubmissionErrorCount());
    else
        setKmdUlong(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_BUFFER_ERROR_COUNT, pAdapterOcaRecord->BufferSubmissionErrorCount());

    return hResult;

} // setKmdAdapterTags

//******************************************************************************

static HRESULT
setKmdEngineTags
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl,
    const CEngineIdOcaRecord *pEngineIdOcaRecord
)
{
    HRESULT             hResult = S_OK;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);
    IF_NULL_RETURN(pEngineIdOcaRecord);

    // Check to see if we have engine type information
    if (pEngineIdOcaRecord->NodeTypeMember().isPresent())
        // Try to set the engine tag string
        hResult = setKmdString(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_TDR_ENGINE, pEngineIdOcaRecord->typeString());
    return hResult;

} // setKmdEngineTags

//******************************************************************************

static HRESULT
setKmdProcessTags
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl,
    const CKmdProcessOcaRecord  *pKmdProcessOcaRecord
)
{
    CString             sProcess;
    HRESULT             hResult = S_OK;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);
    IF_NULL_RETURN(pKmdProcessOcaRecord);

    // Get the suspect process name
    sProcess = pKmdProcessOcaRecord->processName();

    // Check for value process name (*not* a process address)
    if ((sProcess[0] != '0') || (sProcess[1] != 'x'))
    {
        // Try to set the TDR process string
        hResult = setKmdString(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_TDR_PROCESS, sProcess);
    }
    return hResult;

} // setKmdProcessTags

//******************************************************************************

static HRESULT
setKmdWatchdogTags
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl,
    const CGpuWatchdogEvent *pGpuWatchdogEvent
)
{
    HRESULT             hResult = S_OK;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);
    IF_NULL_RETURN(pGpuWatchdogEvent);

    // Try to set the watchdog buffer time tag value
    hResult = setKmdUlong(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_WATCHDOG_TIME, pGpuWatchdogEvent->BufferExelwtionTime());

    return hResult;

} // setKmdWatchdogTags

//******************************************************************************

static HRESULT
setKmdVsyncTags
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl,
    float               fVsyncPeriod
)
{
    HRESULT             hResult = S_OK;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    // Try to set the VSync period tag value
    hResult = setKmdUlong(pAnalysis, pTagControl, LWIDIA_ANALYSIS_TAG_VSYNC_PERIOD, static_cast<ULONG>(fVsyncPeriod * 1000.0));

    return hResult;

} // setKmdVsyncTags

//******************************************************************************

static HRESULT
setKmdString
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl,
    KMD_ANALYSIS_TAG    tag,
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
        // Search KMD tag table for matching tag
        for (ulIndex = 0; ulIndex < countof(s_kmdTagTable); ulIndex++)
        {
            // Get the next tag table entry (Check for tag match)
            pTagEntry = &s_kmdTagTable[ulIndex];
            if (pTagEntry->tag == tag)
            {
                // Try to set the KMD tag string (w/name and description)
                hResult = setTagString(pAnalysis, pTagControl, static_cast<FA_TAG>(tag), sString, pTagEntry->szTagName, pTagEntry->szTagDescription);

                break;
            }
        }
    }
    return hResult;

} // setKmdString

//******************************************************************************

static HRESULT
setKmdUlong
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl,
    KMD_ANALYSIS_TAG    tag,
    ULONG               ulValue
)
{
    ULONG               ulIndex;
    const TAG_ENTRY    *pTagEntry;
    HRESULT             hResult = E_FAIL;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    // Search KMD tag table for matching tag
    for (ulIndex = 0; ulIndex < countof(s_kmdTagTable); ulIndex++)
    {
        // Get the next tag table entry (Check for tag match)
        pTagEntry = &s_kmdTagTable[ulIndex];
        if (pTagEntry->tag == tag)
        {
            // Try to set the KMD tag value (w/name and description)
            hResult = setTagUlong(pAnalysis, pTagControl, static_cast<FA_TAG>(tag), ulValue, pTagEntry->szTagName, pTagEntry->szTagDescription);

            break;
        }
    }
    return hResult;

} // setKmdUlong

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
