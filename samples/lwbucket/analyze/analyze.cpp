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
|*  Module: analyze.cpp                                                       *|
|*                                                                            *|
 \****************************************************************************/
#include "alzprecomp.h"
#include "analyzehelper.h"

//******************************************************************************
//
// Forwards
//
//******************************************************************************
static  HRESULT         analyzeInitialization(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl);
static  HRESULT         analyzeStackAnalysis(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl);
static  HRESULT         analyzePreBucketing(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl);
static  HRESULT         analyzePostBucketing(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl);

static  HRESULT         setAnalysisTags(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl);
static  HRESULT         updateBucketTags(PDEBUG_FAILURE_ANALYSIS2 pAnalysis);
static  HRESULT         updateTagString(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, FA_TAG tag, CString sString);

//******************************************************************************
//
//  Locals
//
//******************************************************************************
// Current analyze phase (Used to prevent multiple exelwtions of any given phase)
static  FA_EXTENSION_PLUGIN_PHASE s_analyzePhase = ILWALID_PHASE;

//******************************************************************************
//
// _EFN_Analyze
//
// Extension Analyze entry point
//
//******************************************************************************

DEBUGGER_ANALYZE
_EFN_Analyze
(
    PDEBUG_CLIENT4      pClient,
    FA_EXTENSION_PLUGIN_PHASE CallPhase,
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis
)
{
    UNREFERENCED_PARAMETER(pClient);

    IDebugFAEntryTags  *pTagControl = NULL;
    HRESULT             hResult = E_UNEXPECTED;

    IF_NULL_RETURN(pClient);
    IF_NULL_RETURN(pAnalysis);

    // Try to get the FA Tag Control interface
    pAnalysis->GetDebugFATagControl(&pTagControl);
    if(pTagControl != NULL)
    {
        // Switch on the analyze call phase
        switch(CallPhase)
        {
            case FA_PLUGIN_INITILIZATION:       // Analyze initialization phase

                // Check for new initialization phase
                if (s_analyzePhase != FA_PLUGIN_INITILIZATION)
                {
                    // Update the analyze phase and call initialization routine
                    s_analyzePhase = FA_PLUGIN_INITILIZATION;

                    hResult = analyzeInitialization(pAnalysis, pTagControl);
                }
                break;

            case FA_PLUGIN_STACK_ANALYSIS:      // Analyze stack analysis phase

                // Check for new initialization phase
                if (s_analyzePhase != FA_PLUGIN_STACK_ANALYSIS)
                {
                    // Update the analyze phase and call stack analysis routine
                    s_analyzePhase = FA_PLUGIN_STACK_ANALYSIS;

                    hResult = analyzeStackAnalysis(pAnalysis, pTagControl);
                }
                break;

            case FA_PLUGIN_PRE_BUCKETING:       // Analyze pre-bucketing phase

                // Check for new initialization phase
                if (s_analyzePhase != FA_PLUGIN_PRE_BUCKETING)
                {
                    // Update the analyze phase and call pre-bucketing routine
                    s_analyzePhase = FA_PLUGIN_PRE_BUCKETING;

                    hResult = analyzePreBucketing(pAnalysis, pTagControl);
                }
                break;

            case FA_PLUGIN_POST_BUCKETING:      // Analyze post-bucketing phase

                // Check for new initialization phase
                if (s_analyzePhase != FA_PLUGIN_POST_BUCKETING)
                {
                    // Update the analyze phase and call initialization routine
                    s_analyzePhase = FA_PLUGIN_POST_BUCKETING;

                    hResult = analyzePostBucketing(pAnalysis, pTagControl);
                }
                break;

            default:                                // Unknown analyze call phase

                // Check for new initialization phase
                if (s_analyzePhase != CallPhase)
                {
                    // Update the analyze phase and report error
                    s_analyzePhase = CallPhase;

                    dPrintf("Lwbucket: Unknown analyze call phase (%d)!\n", CallPhase);
                    hResult = E_FAIL;
                }
                break;
        }
    }
    else    // Unable to get tag control interface
    {
        // Display error to the user
        dPrintf("Lwbucket: Unable to get Tag Control interface!\n");

        // Set result to no such interface
        hResult = E_NOINTERFACE;
    }
    return (FAILED(hResult) ? S_FALSE : hResult);

} // _EFN_Analyze

//******************************************************************************

static HRESULT
analyzeInitialization
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl
)
{
    UNREFERENCED_PARAMETER(pAnalysis);
    UNREFERENCED_PARAMETER(pTagControl);

    HRESULT             hResult = S_OK;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    return hResult;

} // analyzeInitializataion

//******************************************************************************

static HRESULT
analyzeStackAnalysis
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl
)
{
    ULONG               bugcheckCode;
    ULONG64             arg1;
    ULONG64             arg2;
    ULONG64             arg3;
    ULONG64             arg4;
    HRESULT             hResult = S_FALSE;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    // Check to see if this is a dump file (Lwrrently only supports bucketing on dump files using OCA data)
    if (isDumpFile())
    {
        // Read the bugcheck code for this dump file
        hResult = ReadBugCheckData(&bugcheckCode, &arg1, &arg2, &arg3, &arg4);
        if (SUCCEEDED(hResult))
        {
            // Check to see if this bugcheck is a TDR
            if ((bugcheckCode == VIDEO_TDR_ERROR)               ||
                (bugcheckCode == VIDEO_TDR_TIMEOUT_DETECTED)    ||
                (bugcheckCode == VIDEO_ENGINE_TIMEOUT_DETECTED) ||
                (bugcheckCode == VIDEO_TDR_APPLICATION_BLOCKED))
            {
                // Set lWpu analysis tag strings
                hResult = setAnalysisTags(pAnalysis, pTagControl);
            }
        }
        else
        {
            dPrintf("Lwbucket: analyzeStackAnalysis - Failed to ReadBugCheckData!\n");
            hResult = S_FALSE;
        }
    }
    return hResult;

} // analyzeStackAnalysis

//******************************************************************************

static HRESULT
analyzePreBucketing
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl
)
{
    UNREFERENCED_PARAMETER(pAnalysis);
    UNREFERENCED_PARAMETER(pTagControl);

    HRESULT             hResult = S_OK;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    return hResult;

} // analyzePreBucketing

//******************************************************************************

static HRESULT
analyzePostBucketing
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl
)
{
    UNREFERENCED_PARAMETER(pTagControl);

    HRESULT             hResult = E_FAIL;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    // Try to update the bucket tag strings (Set during stack analysis)
    hResult = updateBucketTags(pAnalysis);

    return hResult;

} // analyzePostBucketing

//******************************************************************************

static HRESULT
setAnalysisTags
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl
)
{
    HRESULT             hResult = S_OK;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    // Call routine to try and set KMD tags
    hResult = setKmdTags(pAnalysis, pTagControl);
    if (SUCCEEDED(hResult))
    {
        // Call routine to try and set RM tags
        hResult = setRmTags(pAnalysis, pTagControl);
    }
    else    // Error setting KMD tags
    {
        // Call routine to try and set RM tags (Ignore error since we already have one)
        setRmTags(pAnalysis, pTagControl);
    }
    return hResult;

} // setAnalysisTags

//******************************************************************************

HRESULT
setTagString
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl,
    FA_TAG              tag,
    CString             sString,
    CString             sName,
    CString             sDesc
)
{
    FA_ENTRY           *pEntry = NULL;
    HRESULT             hResult = S_FALSE;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    if(!sString.empty())
    {
        // Try to set the requested tag string
        pEntry = pAnalysis->SetString(tag, sString.data());
        IF_NULL_RETURN(pEntry);

        // Try to set the requested tag name and description
        hResult = pTagControl->SetProperties(tag, sName, sDesc, NULL);
        if (!SUCCEEDED(hResult))
        {
            dPrintf("Lwbucket: setTagString - Failed to set properties. Tag Name: %s, and Description: %s!\n", sName, sDesc);
            hResult = S_FALSE;
        }
    }
    return hResult;

} // setTagString

//******************************************************************************

HRESULT
setTagUlong
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl,
    FA_TAG              tag,
    ULONG               ulValue,
    CString             sName,
    CString             sDesc
)
{
    FA_ENTRY           *pEntry = NULL;
    HRESULT             hResult = S_OK;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    // Try to set the requested tag ULONG value
    pEntry = pAnalysis->SetUlong(tag, ulValue);
    IF_NULL_RETURN(pEntry);

    // Try to set the requested tag name and description
    hResult = pTagControl->SetProperties(tag, sName, sDesc, NULL);
    if (!SUCCEEDED(hResult))
    {
        dPrintf("Lwbucket: setTagUlong - Failed to set properties. Tag Name: %s, and Description: %s!\n", sName, sDesc);
        hResult = S_FALSE;
    }
    return hResult;

} // setTagUlong

//******************************************************************************

HRESULT
setTagUlong64
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    IDebugFAEntryTags  *pTagControl,
    FA_TAG              tag,
    ULONG64             ulValue,
    CString             sName,
    CString             sDesc
)
{
    FA_ENTRY           *pEntry = NULL;
    HRESULT             hResult = S_OK;

    IF_NULL_RETURN(pAnalysis);
    IF_NULL_RETURN(pTagControl);

    // Try to set the requested tag ULONG64 value
    pEntry = pAnalysis->SetUlong64(tag, ulValue);
    IF_NULL_RETURN(pEntry);

    // Try to set the requested tag name and description
    hResult = pTagControl->SetProperties(tag, sName, sDesc, NULL);
    if (!SUCCEEDED(hResult))
    {
        dPrintf("Lwbucket: setTagUlong64 - Failed to set properties. Tag Name: %s, and Description: %s!\n", sName, sDesc);
        hResult = S_FALSE;
    }
    return hResult;

} // setTagUlong64

//******************************************************************************

static HRESULT
updateBucketTags
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis
)
{
    FA_ENTRY           *pEntry = NULL;
    CString             sFamily(MAX_COMMAND_STRING);
    CString             sEngine(MAX_COMMAND_STRING);
    CString             sOverclocked(MAX_COMMAND_STRING);
    CString             sString;
    HRESULT             hResult = S_FALSE;

    IF_NULL_RETURN(pAnalysis);

    // Build the string to append to the bucket (Lwrrently GPU family and TDR engine)
    pEntry = pAnalysis->GetString(static_cast<FA_TAG>(LWIDIA_ANALYSIS_TAG_GPU_FAMILY), sFamily.data(), static_cast<ULONG>(sFamily.capacity()));
    if (pEntry != NULL)
    {
        // Add the GPU family string
        sString.append("_");
        sString.append(sFamily);
    }

    pEntry = pAnalysis->GetString(static_cast<FA_TAG>(LWIDIA_ANALYSIS_TAG_USER_OVERCLOCKED), sOverclocked.data(), static_cast<ULONG>(sOverclocked.capacity()));
    if (pEntry != NULL)
    {
        // Add the user overclocked string
        sString.append("_");
        sString.append(sOverclocked);
    }
    else
    {
        pEntry = pAnalysis->GetString(static_cast<FA_TAG>(LWIDIA_ANALYSIS_TAG_TDR_ENGINE), sEngine.data(), static_cast<ULONG>(sEngine.capacity()));
        if (pEntry != NULL)
        {
            // Add the engine string
            sString.append("_");
            sString.append(sEngine);
        }
    }

    // Check to see if there is an update to the bucket ID string
    if (!sString.empty())
    {
        // Try to update the bucket ID string
        hResult = updateTagString(pAnalysis, DEBUG_FLR_BUCKET_ID, sString);
        if (SUCCEEDED(hResult))
        {
            // Try to update the failure bucket ID string
            hResult = updateTagString(pAnalysis, DEBUG_FLR_FAILURE_BUCKET_ID, sString);
        }
        else    // Failed to update the bucket ID string
        {
            // Try to update failure bucket ID string (but ignore result since bucket ID string update failed)
            updateTagString(pAnalysis, DEBUG_FLR_FAILURE_BUCKET_ID, sString);
        }
    }
    return hResult;

} // updateBucketTags

//******************************************************************************

static HRESULT
updateTagString
(
    PDEBUG_FAILURE_ANALYSIS2 pAnalysis,
    FA_TAG              tag,
    CString             sString
)
{
    FA_ENTRY           *pEntry = NULL;
    CString             sTag(2048);
    HRESULT             hResult = S_OK;

    IF_NULL_RETURN(pAnalysis);

    // Try to get the requested tag string
    pEntry = pAnalysis->GetString(tag, sTag.data(), static_cast<ULONG>(sTag.capacity()));
    if (pEntry != NULL)
    {
        // Check to see if tag string has not already been added
        if (sTag.find(sString) == NOT_FOUND)
        {
            // Append the given string to the tag
            sTag.append(sString);

            // Try to update the "appended" tag string
            pEntry = pAnalysis->SetString(tag, sTag);
            if (pEntry == NULL)
            {
                // Indicate the update failed
                hResult = E_FAIL;
            }
        }
    }
    else    // Invalid tag?
    {
        // Set failure result
        hResult = E_FAIL;
    }
    return hResult;

} // updateString

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
