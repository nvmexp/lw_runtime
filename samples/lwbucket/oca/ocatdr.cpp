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
|*  Module: ocatdr.cpp                                                        *|
|*                                                                            *|
 \****************************************************************************/
#include "ocaprecomp.h"

//******************************************************************************
//
//  oca namespace
//
//******************************************************************************
namespace oca
{

//******************************************************************************
//
// Forwards
//
//******************************************************************************
static  bool            uniqueOcaProcess(ULONG ulResetEngine, CKmdProcessOcaRecord** pProcessArray);




//******************************************************************************
//
//  Locals
//
//******************************************************************************
// Define the forced tables (Error & Warning)
static const FORCE_DATA s_ForceError[] =   {
                                            {"reset with Device Detached",                  "due to reset with device detached!"},
                                            {"resetEngine with busResetTDR",                "due to reset with bus reset in progress!"},
                                            {"resetEngine with GC6 TDR",                    "due to reset when in GC6!"},
                                            {"resetEngine with forced TDR",                 "due to reset with forced TDR!"},
                                            {"called with unexpected irql",                 "due to reset at unexpected IRQL!"},
                                            {"Too many ResetEngine calls",                  "due to too many reset engine calls!"},
                                           };
static const FORCE_DATA s_ForceWarning[] = {
                                            {"called at unsupported hardware",              "due to reset called for unsupported hardware/configuration!"},
                                            {"could not determine adapter that needs reset","due to reset cannot determine adapter that needs reset!"},
                                            {"Failing buffer completions",                  "due to failing DMA buffer completions (Forced TDR or surprise removal)!"},
                                           };

//******************************************************************************

CString
tdrName
(
    ULONG               bugcheckCode
)
{
    CString             tdrName;

    // Switch on the bugcheck code
    switch(bugcheckCode)
    {
        case VIDEO_TDR_ERROR:               tdrName = "VIDEO_TDR_ERROR";                break;
        case VIDEO_TDR_TIMEOUT_DETECTED:    tdrName = "VIDEO_TDR_TIMEOUT_DETECTED";     break;
        case VIDEO_ENGINE_TIMEOUT_DETECTED: tdrName = "VIDEO_ENGINE_TIMEOUT_DETECTED";  break;
        case VIDEO_TDR_APPLICATION_BLOCKED: tdrName = "VIDEO_TDR_APPLICATION_BLOCKED";  break;

        default:                            tdrName = "UNKNOWN";                        break;
    }
    return tdrName;

} // tdrName

//******************************************************************************

CString
tdrReason
(
    ULONG64             reasonCode
)
{
    CString             tdrReason;

    // Switch on the reason code
    switch(reasonCode)
    {

        case TDR_REASON_UNKNOWN:                                tdrReason = "Unknown";                          break;
        case TDR_REASON_RECOVERY_DISABLED:                      tdrReason = "Recovery disabled";                break;
        case TDR_REASON_CONSELWTIVE_TIMEOUT:                    tdrReason = "Conselwtive timeout";              break;
        case TDR_REASON_DDI_RESET_FROM_TIMEOUT_FAILURE:         tdrReason = "ResetFromTimeout failure";         break;
        case TDR_REASON_DDI_RESTART_FROM_TIMEOUT_FAILURE:       tdrReason = "RestartFromTimeout failure";       break;
        case TDR_REASON_GDI_OFF_FAILURE:                        tdrReason = "GDI off failure";                  break;
        case TDR_REASON_GDI_ON_FAILURE:                         tdrReason = "GDI on failure";                   break;
        case TDR_REASON_GDI_RESET_THREAD_NEW_FAILURE:           tdrReason = "GDI reset thread new failure";     break;
        case TDR_REASON_GDI_RESET_THREAD_START_FAILURE:         tdrReason = "GDI reset thread start failure";   break;
        case TDR_REASON_VIDSCHI_PREPARE_FOR_RECOVERY_FAILURE:   tdrReason = "Prepare for recovery failure";     break;
        case TDR_REASON_ADAPTER_PREPARE_TO_RESET_FAILURE:       tdrReason = "Prepare to reset failure";         break;
        case TDR_REASON_ADAPTER_RESET_FAILURE:                  tdrReason = "Adapter reset failure";            break;
        case TDR_REASON_APCS_ARE_DISABLED:                      tdrReason = "APC's are disabled";               break;
        case TDR_REASON_RECOVERY_LIMIT_EXHAUSTED:               tdrReason = "Recovery limit exhausted";         break;

        default:                                                tdrReason = "Unknown";                          break;
    }
    return tdrReason;

} // tdrReason

//******************************************************************************

CString
tdrDescription
(
    ULONG               bugcheckCode,
    ULONG64             arg1,
    ULONG64             arg2,
    ULONG64             arg3,
    ULONG64             arg4
)
{
    UNREFERENCED_PARAMETER(bugcheckCode);
    UNREFERENCED_PARAMETER(arg1);
    UNREFERENCED_PARAMETER(arg2);

    NTSTATUS            status = static_cast<LONG>(arg3);
    CString             tdrDescription(MAX_DBGPRINTF_STRING);

    // Switch on the bugcheck code (TDR type)
    switch(bugcheckCode)
    {
        case VIDEO_TDR_ERROR:                       // Bugcheck 0x116

            // Switch on the reason code (arg4)
            switch(arg4)
            {
                case TDR_REASON_UNKNOWN:                                // Unknown TDR reason code

                    // Format the TDR description
                    tdrDescription.sprintf("Internal shouldn't happen except for OS bug/severe lack of resources %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                case TDR_REASON_RECOVERY_DISABLED:                      // TDR recovery disabled reason code

                    // Format the TDR description
                    tdrDescription.sprintf("TDR recovery is disabled in the registry %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                case TDR_REASON_CONSELWTIVE_TIMEOUT:                    // TDR conselwtive buffer timeout reason code

                    // Format the TDR description
                    tdrDescription.sprintf("Conselwtive TDR without a successful DMA buffer completion %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                case TDR_REASON_DDI_RESET_FROM_TIMEOUT_FAILURE:         // ResetFromTimeout DDI failure reason code

                    // Format the TDR description
                    tdrDescription.sprintf("ResetFromTimeout DDI failure %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                case TDR_REASON_DDI_RESTART_FROM_TIMEOUT_FAILURE:       // RestartFromTimeout DDI failure reason code

                    // Format the TDR description
                    tdrDescription.sprintf("RestartFromTimeout DDI failure %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                case TDR_REASON_GDI_OFF_FAILURE:                        // GDI off failure reason code

                    // Format the TDR description
                    tdrDescription.sprintf("Internal shouldn't happen except for OS bug/severe lack of resources %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                case TDR_REASON_GDI_ON_FAILURE:                         // GDI on failure reason code

                    // Format the TDR description
                    tdrDescription.sprintf("Internal shouldn't happen except for OS bug/severe lack of resources %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                case TDR_REASON_GDI_RESET_THREAD_NEW_FAILURE:           // GDI reset thread new failure reason code

                    // Format the TDR description
                    tdrDescription.sprintf("Internal shouldn't happen except for OS bug/severe lack of resources %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                case TDR_REASON_GDI_RESET_THREAD_START_FAILURE:         // GDI reset thread start failure reason code

                    // Format the TDR description
                    tdrDescription.sprintf("Internal shouldn't happen except for OS bug/severe lack of resources %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                case TDR_REASON_VIDSCHI_PREPARE_FOR_RECOVERY_FAILURE:   // Prepare for recovery failure reason code

                    // Format the TDR description
                    tdrDescription.sprintf("Internal shouldn't happen except for OS bug/severe lack of resources %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                case TDR_REASON_ADAPTER_PREPARE_TO_RESET_FAILURE:       // Prepare to reset failure reason code

                    // Check for I/O timeout (Driver failed to exit all threads in time)
                    if (arg3 == STATUS_IO_TIMEOUT)
                    {
                        // Format the TDR description
                        tdrDescription.sprintf("Prepare to reset failure, driver failed to exit all threads in time %s (%s)",
                                               DML(statusString(status)), DML(errorString(status)));
                    }
                    else    // Not failed to exit driver threads
                    {
                        // Format the TDR description
                        tdrDescription.sprintf("Prepare to reset failure %s (%s)",
                                               DML(statusString(status)), DML(errorString(status)));
                    }
                    break;

                case TDR_REASON_ADAPTER_RESET_FAILURE:                  // Adapter reset failure reason code

                    // Format the TDR description
                    tdrDescription.sprintf("Internal shouldn't happen except for OS bug/severe lack of resources %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                case TDR_REASON_APCS_ARE_DISABLED:                      // APC's disabled reason code

                    // Format the TDR description
                    tdrDescription.sprintf("Timeout called in an improper context (Kernel APC's disabled due to guarded mutex lock?) %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                case TDR_REASON_RECOVERY_LIMIT_EXHAUSTED:               // Recovery limit exhausted reason code

                    // Format the TDR description
                    tdrDescription.sprintf("Recovery limit exhausted, more than 5 TDR's in a 60 second period %s (%s)",
                                           DML(statusString(status)), DML(errorString(status)));

                    break;

                default:                                                // Unknown TDR reason code

                    // Format the TDR description
                    tdrDescription.sprintf("Unknown %s (%s)", DML(statusString(status)), DML(errorString(status)));
            
                    break;
            }
            break;

        case VIDEO_TDR_TIMEOUT_DETECTED:            // Bugcheck 0x117

            // Format the TDR description
            tdrDescription.sprintf("The display driver failed to respond in timely fashion");

            break;

        case VIDEO_ENGINE_TIMEOUT_DETECTED:         // Bugcheck 0x141

            // Format the TDR description
            tdrDescription.sprintf("One of the the display engines failed to respond in timely fashion");

            break;

        case VIDEO_TDR_APPLICATION_BLOCKED:         // Bugcheck 0x142

            // Format the TDR description
            tdrDescription.sprintf("Application has been blocked from accessing Graphics hardware");

            break;

        default:                                    // Unknown TDR bugcheck

            // Format the TDR description
            tdrDescription.sprintf("Unknown TDR bugcheck code 0x%x", bugcheckCode);

            break;
    }
    return tdrDescription;

} // tdrDescription

//******************************************************************************

bool
tdrForced
(
    ULONG               bugcheckCode,
    ULONG64             arg1,
    ULONG64             arg2,
    ULONG64             arg3,
    ULONG64             arg4
)
{
    UNREFERENCED_PARAMETER(arg1);
    UNREFERENCED_PARAMETER(arg2);
    UNREFERENCED_PARAMETER(arg3);

    CString             sForced;
    bool                bForced = false;

    // Switch on the bugcheck code (TDR type)
    switch(bugcheckCode)
    {
        case VIDEO_TDR_ERROR:                       // Bugcheck 0x116

            // Check for TDR recovery disabled
            if (arg4 == TDR_REASON_RECOVERY_DISABLED)
            {
                // Indicate a forced TDR
                dbgPrintf("Forced TDR due to TDR recovery disabled in the registry\n");
                bForced = true;
            }
            break;

        case VIDEO_TDR_TIMEOUT_DETECTED:            // Bugcheck 0x117

            // Try to get the forced string
            sForced = forcedString();

            // Check for forced string
            if (!sForced.empty())
            {
                // Display forced TDR error and set forced TDR
                dbgPrintf("Forced TDR %s\n", STR(sForced));

                sForced = true;
            }
            break;

        case VIDEO_ENGINE_TIMEOUT_DETECTED:         // Bugcheck 0x141

            // Try to get the forced string
            sForced = forcedString();

            // Check for forced string
            if (!sForced.empty())
            {
                // Display incoming TDR error and set forced TDR
                dbgPrintf("Incoming TDR %s\n", STR(sForced));

                sForced = true;
            }
            break;

        case VIDEO_TDR_APPLICATION_BLOCKED:         // Bugcheck 0x142

            // Try to get the forced string
            sForced = forcedString();

            // Check for forced string
            if (!sForced.empty())
            {
                // Display application blocked error and set forced TDR
                dbgPrintf("Application blocked %s\n", STR(sForced));

                sForced = true;
            }
            break;

        default:                                    // Unknown TDR bugcheck

            break;
    }
    return bForced;

} // tdrForced

//******************************************************************************

CString
forcedString()
{
    ULONG               ulForce;
    ULONG               ulErrorData;
    ULONG               ulWarningData;
    const CErrorInfoData *pErrorInfoData;
    const CErrorInfoOcaRecord *pErrorInfoOcaRecord;
    const CWarningInfoData *pWarningInfoData;
    const CWarningInfoOcaRecord *pWarningInfoOcaRecord;
    CString             sForced;

    // Check for OCA error records available
    pErrorInfoOcaRecord = findOcaErrorRecords();
    if (pErrorInfoOcaRecord != NULL)
    {
        // Search error records for forced TDR errors
        for (ulErrorData = 0; ulErrorData < pErrorInfoOcaRecord->errorDataCount(); ulErrorData++)
        {
            // Get the next error data record to check
            pErrorInfoData = pErrorInfoOcaRecord->errorInfoData(ulErrorData);
            if (pErrorInfoData != NULL)
            {
                // Search for forced TDR error record
                for (ulForce = 0; ulForce < countof(s_ForceError); ulForce++)
                {
                    // Check to see if this is a forced TDR error record
                    if (pErrorInfoData->annotation().find(s_ForceError[ulForce].pAnnotationString) != NOT_FOUND)
                    {
                        // Found the forced TDR error record, set forced string and stop search
                        sForced = s_ForceError[ulForce].pForceString;

                        break;
                    }
                }
                // Check for forced string found
                if (!sForced.empty())
                {
                    // Stop the search
                    break;
                }
            }
        }
    }
    // Check for forced string not already found
    if (sForced.empty())
    {
        // Check for OCA warning records available
        pWarningInfoOcaRecord = findOcaWarningRecords();
        if (pWarningInfoOcaRecord != NULL)
        {
            // Search warning records for forced TDR warnings
            for (ulWarningData = 0; ulWarningData < pWarningInfoOcaRecord->warningDataCount(); ulWarningData++)
            {
                // Get the next warning data record to check
                pWarningInfoData = pWarningInfoOcaRecord->warningInfoData(ulWarningData);
                if (pWarningInfoData != NULL)
                {
                    // Search for forced TDR waarning record
                    for (ulForce = 0; ulForce < countof(s_ForceWarning); ulForce++)
                    {
                        // Check to see if this is a forced TDR warning record
                        if (pWarningInfoData->annotation().find(s_ForceWarning[ulForce].pAnnotationString) != NOT_FOUND)
                        {
                            // Found the forced TDR warning record, set forced string and stop search
                            sForced = s_ForceWarning[ulForce].pForceString;

                            break;
                        }
                    }
                    // Check for forced string found
                    if (!sForced.empty())
                    {
                        // Stop the search
                        break;
                    }
                }
            }
        }
    }
    return sForced;

} // forcedString

//******************************************************************************

const CVblankInfoOcaRecord*
findTdrVsync()
{
    const COcaDataPtr   pOcaData = ocaData();
    ULONG               ulDrvOcaRecord;
    ULONG               ulVblankData;
    ULONG64             ulElapsedTime;
    ULONG64             ulLastTime = 0;
    float               fElapsedTime = 0.0;
    const CLwcdRecord  *pDrvOcaRecord;
    const CVblankInfoData *pVblankInfoData;
    const CVblankInfoOcaRecord *pVblankInfoOcaRecord = NULL;

    // Loop thru the driver (KMD) OCA records looking for vblank info records
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the OCA vblank info type
            if (pDrvOcaRecord->cRecordType() == KmdVblankInfo)
            {
                // Get the OCA vblank info record and check if it is a possibly VSYNC TDR
                pVblankInfoOcaRecord = static_cast<const CVblankInfoOcaRecord *>(pDrvOcaRecord);
                if (pVblankInfoOcaRecord != NULL)
                {
                    // Loop checking the vblank data records (For possible VSYNC timeout)
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

                                // Check for possible VSYNC timeout
                                if (fElapsedTime >= 2.0)
                                {
                                    // Possible VSYNC timeout, stop search
                                    break;
                                }
                                else    // No VSYNC timeout yet
                                {
                                    // Clear elapsed time in case last vblank info data
                                    fElapsedTime = 0.0;
                                }
                            }
                            ulLastTime = pVblankInfoData->timestamp();
                        }
                    }
                    // Check for possible VSYNC timeout detected
                    if (fElapsedTime >= 2.0)
                    {
                        // Stop the search of Vblank info OCA records
                        break;
                    }
                    else    // No VSYNC timeout detected
                    {
                        // Clear the vblank info pointer (No VSYNC timeout, in case last Vblank record)
                        pVblankInfoOcaRecord = NULL;
                    }
                }
            }
        }
    }
    return pVblankInfoOcaRecord;

} // findTdrVSync

//******************************************************************************

const CGpuWatchdogEvent*
findGpuWatchdogEvent()
{
    const COcaDataPtr   pOcaData = ocaData();
    ULONG               ulDrvOcaRecord;
    ULONG               ulGpuWatchdogEvent;
    const CLwcdRecord  *pDrvOcaRecord;
    const CGpuWatchdogEvent *pChkWatchdogEvent;
    const CGpuWatchdogEvent *pGpuWatchdogEvent = NULL;
    const CKmdRingBufferOcaRecord *pKmdRingBufferOcaRecord = NULL;

    // Loop thru the driver (KMD) OCA records looking for KMD ring buffer records
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the OCA KMD ring buffer type
            if ((pDrvOcaRecord->cRecordType() == KmdRingBufferInfo) || (pDrvOcaRecord->cRecordType() == KmdRingBufferInfo_V2))
            {
                // Get the OCA KMD ring buffer record and check for a GPU watchdog event
                pKmdRingBufferOcaRecord = static_cast<const CKmdRingBufferOcaRecord *>(pDrvOcaRecord);
                if (pKmdRingBufferOcaRecord != NULL)
                {
                    // Loop checking for a GPU watchdog event
                    for (ulGpuWatchdogEvent = 0; ulGpuWatchdogEvent < pKmdRingBufferOcaRecord->gpuWatchdogEventCount(); ulGpuWatchdogEvent++)
                    {
                        // Get the next GPU watchdog event to check
                        pGpuWatchdogEvent = pKmdRingBufferOcaRecord->gpuWatchdogEvent(ulGpuWatchdogEvent);
                        if (pGpuWatchdogEvent != NULL)
                        {
                            // Check for a GPU watchdog timeout event (Possible TDR source)
                            if (pGpuWatchdogEvent->BufferExelwtionTime() > 2000)
                            {
                                // Check for later GPU watchdog events
                                for (ulGpuWatchdogEvent = (ulGpuWatchdogEvent + 1); ulGpuWatchdogEvent < pKmdRingBufferOcaRecord->gpuWatchdogEventCount(); ulGpuWatchdogEvent++)
                                {
                                    // Get the next GPU watchdog event to check
                                    pChkWatchdogEvent = pKmdRingBufferOcaRecord->gpuWatchdogEvent(ulGpuWatchdogEvent);
                                    if (pChkWatchdogEvent != NULL)
                                    {
                                        // Check for matching GPU watchdog event (to previously found event)
                                        if ((pChkWatchdogEvent->AdapterOrdinal() == pGpuWatchdogEvent->AdapterOrdinal()) &&
                                            (pChkWatchdogEvent->NodeOrdinal()    == pGpuWatchdogEvent->NodeOrdinal())    &&
                                            (pChkWatchdogEvent->EngineOrdinal()  == pGpuWatchdogEvent->EngineOrdinal())  &&
                                            (pChkWatchdogEvent->FenceId()        == pGpuWatchdogEvent->FenceId()))
                                        {
                                            // Matching GPU watchdog event, check for later event
                                            if (pChkWatchdogEvent->BufferExelwtionTime() > pGpuWatchdogEvent->BufferExelwtionTime())
                                            {
                                                // Update the GPU watchdog event
                                                pGpuWatchdogEvent = pChkWatchdogEvent;
                                            }
                                        }
                                    }
                                }
                                // Found a GPU watchdog event (and got the latest matching one), stop search
                                break;
                            }
                            else    // Not long enough to cause a TDR yet
                            {
                                // Clear the GPU watchdog event (in case the last record)
                                pGpuWatchdogEvent = NULL;
                            }
                        }
                    }
                    // Check for a GPU watchdog event found
                    if (pGpuWatchdogEvent != NULL)
                    {
                        // Stop searching thru the driver OCA records
                        break;
                    }
                }
            }
        }
    }
    return pGpuWatchdogEvent;

} // findGpuWatchdogEvent

//******************************************************************************

const CAdapterOcaRecord*
findTdrAdapter()
{
    ULONG               ulErrorData;
    ULONG               ulAdapter;
    ULONG               ulEngineId;
    const CErrorInfoData *pErrorInfoData;
    const CErrorInfoOcaRecord *pErrorInfoOcaRecord;
    const CEngineIdOcaRecord *pEngineIdOcaRecord;
    const CAdapterOcaRecord *pAdapterOcaRecord = NULL;

    // Check for OCA error records available
    pErrorInfoOcaRecord = findOcaErrorRecords();
    if (pErrorInfoOcaRecord != NULL)
    {
        // Search error records for TDR error
        for (ulErrorData = 0; ulErrorData < pErrorInfoOcaRecord->errorDataCount(); ulErrorData++)
        {
            // Get the next error data record to check
            pErrorInfoData = pErrorInfoOcaRecord->errorInfoData(ulErrorData);
            if (pErrorInfoData != NULL)
            {
                // Check to see if this is a TDR error record
                if (pErrorInfoData->subTypeName().find("TDR Oclwrred") != NOT_FOUND)
                {
                    // Found the TDR error record, record adapter and stop search
                    pAdapterOcaRecord = findOcaAdapter(pErrorInfoData->adapter().ptr());

                    break;
                }
            }
        }
    }
    // Check for TDR adapter not found yet
    if (pAdapterOcaRecord == NULL)
    {
        // Search all the OCA adapter records
        for (ulAdapter = 0; ulAdapter < ocaAdapterCount(); ulAdapter++)
        {
            // Get the next OCA adapter to check
            pAdapterOcaRecord = findOcaAdapter(ulAdapter);
            if (pAdapterOcaRecord != NULL)
            {
                // Check for buffers exelwting on this adapter
                for (ulEngineId = 0; ulEngineId < ocaEngineIdCount(ulAdapter); ulEngineId++)
                {
                    // Get the next engine ID to check
                    pEngineIdOcaRecord = findEngineId(ulAdapter, ulEngineId);
                    if (pEngineIdOcaRecord != NULL)
                    {
                        // Check for buffers still exelwting on this adapter
                        if (pEngineIdOcaRecord->BufferId() != pEngineIdOcaRecord->ISRBufferId())
                        {
                            // Found OCA adapter with exelwting buffers, assume it's the TDR adapter (stop search)
                            break;
                        }
                    }
                }
                // Check for OCA adapter found (exelwting DMA buffers)
                if (ulEngineId != ocaEngineIdCount(ulAdapter))
                {
                    // OCA adapter found, break out of search loop
                    break;
                }
                else    // OCA adapter not found (no exelwting DMA buffers)
                {
                    // Check for adapter in TDR or non-zero TDR count (and assume this is the TDR adapter)
                    if (pAdapterOcaRecord->InTDR() || (pAdapterOcaRecord->TDRCount() != 0))
                    {
                        // Assume this is the adapter in TDR
                        break;
                    }
                    // Clear the OCA adapter (in case of last adapter)
                    pAdapterOcaRecord = NULL;
                }
            }
        }
    }
    return pAdapterOcaRecord;

} // findTdrAdapter

//******************************************************************************

const CContextOcaRecord*
findTdrContext
(
    const CAdapterOcaRecord* pAdapterOcaRecord
)
{
    const CBufferInfoRecord *pBufferInfoRecord;
    const CContextOcaRecord *pContextOcaRecord = NULL;

    // Try to find the exelwting DMA buffer (but *not* ignoring preemption)
    pBufferInfoRecord = findTdrBuffer(pAdapterOcaRecord, false);
    if (pBufferInfoRecord == NULL)
    {
        // Try again but *ignore* preemption this time (Since we didn't find one in preemption)
        pBufferInfoRecord = findTdrBuffer(pAdapterOcaRecord, true);
    }
    // Check for TDR buffer found
    if (pBufferInfoRecord != NULL)
    {
        // Try to get the TDR context for this TDR buffer
        pContextOcaRecord = findOcaContext(pBufferInfoRecord->Context().ptr());
    }
    return pContextOcaRecord;

} // findTdrContext

//******************************************************************************

const CBufferInfoRecord*
findTdrBuffer
(
    const CAdapterOcaRecord *pAdapterOcaRecord,
    bool                bIgnorePreemption
)
{
    ULONG               ulEngineId;
    ULONG               ulBufferInfo;
    const CEngineIdOcaRecord *pEngineIdOcaRecord;
    const CContextOcaRecord *pContextOcaRecord;
    const CBufferInfoRecord *pBufferInfoRecord;
    const CEngineIdOcaRecord *pTdrEngineIdOcaRecord = NULL;
    const CBufferInfoRecord *pTdrBufferInfoRecord = NULL;

    // Loop thru engine ID records looking for exelwting DMA buffer
    for (ulEngineId = 0; ulEngineId < ocaEngineIdCount(pAdapterOcaRecord->Adapter().ptr()); ulEngineId++)
    {
        // Try to get the next engine ID records for this adapter
        pEngineIdOcaRecord = findEngineId(pAdapterOcaRecord->Adapter().ptr(), ulEngineId);
        if (pEngineIdOcaRecord != NULL)
        {
            // Check to see if the engine is in preemption (or ignore preemption)
            if (bIgnorePreemption || (pEngineIdOcaRecord->PreemptionId() != pEngineIdOcaRecord->ISRPreemptionId()))
            {
                // Loop checking DMA buffer info records for exelwting DMA buffer
                for (ulBufferInfo = 0; ulBufferInfo < pEngineIdOcaRecord->bufferInfoCount(); ulBufferInfo++)
                {
                    // Get the next buffer info record to check
                    pBufferInfoRecord = pEngineIdOcaRecord->bufferInfoRecord(ulBufferInfo);
                    if (pBufferInfoRecord != NULL)
                    {
                        // Check to see if this DMA buffer is exelwting
                        if (pBufferInfoRecord->BufferId() > pEngineIdOcaRecord->ISRBufferId())
                        {
                            // Check for a valid context value for this DMA buffer
                            if (pBufferInfoRecord->Context() != NULL)
                            {
                                // Try to find context just in case this is the requested buffer
                                pContextOcaRecord = findOcaContext(pBufferInfoRecord->Context().ptr());
                                if (pContextOcaRecord != NULL)
                                {
                                    // Check to see if this is the exelwting DMA buffer
                                    if (pBufferInfoRecord->BufferId() == (pEngineIdOcaRecord->ISRBufferId() + 1))
                                    {
                                        // Check to see if we have a previous TDR buffer (Need to see who is exelwting the longest)
                                        if (pTdrBufferInfoRecord != NULL)
                                        {
                                            // Check to see if new DMA buffer has been exelwting longer (Check submit ID)
                                            if (pBufferInfoRecord->SubmitId() < pTdrBufferInfoRecord->SubmitId())
                                            {
                                                // This DMA buffer has been exelwting longer, update TDR information
                                                pTdrEngineIdOcaRecord = pEngineIdOcaRecord;
                                                pTdrBufferInfoRecord  = pBufferInfoRecord;
                                            }
                                        }
                                        else    // This is the first TDR DMA buffer
                                        {
                                            // Save the TDR information
                                            pTdrEngineIdOcaRecord = pEngineIdOcaRecord;
                                            pTdrBufferInfoRecord  = pBufferInfoRecord;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Check to see if we found a TDR buffer or not
    if (pTdrBufferInfoRecord != NULL)
    { 
        // Check to see if we found the desired DMA buffer (or had to settle for another exelwting one)
        if (pTdrBufferInfoRecord->BufferId() != (pTdrEngineIdOcaRecord->ISRBufferId() + 1))
        {
            // Warn user we are using a different DMA buffer
            dPrintf("Wanted DMA buffer ID 0x%0x but using DMA buffer ID 0x%0x context!\n", (pTdrEngineIdOcaRecord->ISRBufferId() + 1),
                                                                                           pTdrBufferInfoRecord->BufferId());
        }
    }
    return pTdrBufferInfoRecord;

} // findTdrBuffer

//******************************************************************************

const CEngineIdOcaRecord*
findTdrEngine
(
    const CAdapterOcaRecord *pAdapterOcaRecord,
    bool                bIgnorePreemption
)
{
    ULONG               ulEngineId;
    ULONG               ulBufferInfo;
    const CEngineIdOcaRecord *pEngineIdOcaRecord;
    const CContextOcaRecord *pContextOcaRecord;
    const CBufferInfoRecord *pBufferInfoRecord;
    const CEngineIdOcaRecord *pTdrEngineIdOcaRecord = NULL;
    const CBufferInfoRecord *pTdrBufferInfoRecord = NULL;

    // Loop thru engine ID records looking for exelwting DMA buffer
    for (ulEngineId = 0; ulEngineId < ocaEngineIdCount(pAdapterOcaRecord->Adapter().ptr()); ulEngineId++)
    {
        // Try to get the next engine ID records for this adapter
        pEngineIdOcaRecord = findEngineId(pAdapterOcaRecord->Adapter().ptr(), ulEngineId);
        if (pEngineIdOcaRecord != NULL)
        {
            // Check to see if the engine is in preemption (or ignore preemption)
            if (bIgnorePreemption || (pEngineIdOcaRecord->PreemptionId() != pEngineIdOcaRecord->ISRPreemptionId()))
            {
                // Loop checking DMA buffer info records for exelwting DMA buffer
                for (ulBufferInfo = 0; ulBufferInfo < pEngineIdOcaRecord->bufferInfoCount(); ulBufferInfo++)
                {
                    // Get the next buffer info record to check
                    pBufferInfoRecord = pEngineIdOcaRecord->bufferInfoRecord(ulBufferInfo);
                    if (pBufferInfoRecord != NULL)
                    {
                        // Check to see if this DMA buffer is exelwting
                        if (pBufferInfoRecord->BufferId() > pEngineIdOcaRecord->ISRBufferId())
                        {
                            // Check for a valid context value for this DMA buffer
                            if (pBufferInfoRecord->Context() != NULL)
                            {
                                // Try to find context just in case this is the requested buffer
                                pContextOcaRecord = findOcaContext(pBufferInfoRecord->Context().ptr());
                                if (pContextOcaRecord != NULL)
                                {
                                    // Check to see if this is the exelwting DMA buffer
                                    if (pBufferInfoRecord->BufferId() == (pEngineIdOcaRecord->ISRBufferId() + 1))
                                    {
                                        // Check to see if we have a previous TDR buffer (Need to see who is exelwting the longest)
                                        if (pTdrBufferInfoRecord != NULL)
                                        {
                                            // Check to see if new DMA buffer has been exelwting longer (Check submit ID)
                                            if (pBufferInfoRecord->SubmitId() < pTdrBufferInfoRecord->SubmitId())
                                            {
                                                // This DMA buffer has been exelwting longer, update TDR information
                                                pTdrEngineIdOcaRecord = pEngineIdOcaRecord;
                                                pTdrBufferInfoRecord  = pBufferInfoRecord;
                                            }
                                        }
                                        else    // This is the first TDR DMA buffer
                                        {
                                            // Save the TDR information
                                            pTdrEngineIdOcaRecord = pEngineIdOcaRecord;
                                            pTdrBufferInfoRecord  = pBufferInfoRecord;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Check to see if we found a TDR buffer or not
    if (pTdrBufferInfoRecord != NULL)
    { 
        // Check to see if we found the desired DMA buffer (or had to settle for another exelwting one)
        if (pTdrBufferInfoRecord->BufferId() != (pTdrEngineIdOcaRecord->ISRBufferId() + 1))
        {
            // Warn user we are using a different DMA buffer
            dPrintf("Wanted DMA buffer ID 0x%0x but using DMA buffer ID 0x%0x context!\n", (pTdrEngineIdOcaRecord->ISRBufferId() + 1),
                                                                                           pTdrBufferInfoRecord->BufferId());
        }
    }
    return pTdrEngineIdOcaRecord;

} // findTdrEngine

//******************************************************************************

const CRmProtoBufRecord*
findRmProtoBuf()
{
    COcaDataPtr         pOcaData = ocaData();
    union
    {
        const CLwcdRecord *pRmOcaRecord;
        const CRmProtoBufRecord *pRmProtoBufRecord;
    };
    ULONG               ulRmOcaRecord;
    const CRmProtoBufRecord *pRmProtoBuf = NULL;

    // Check for OCA data available
    if (pOcaData != NULL)
    {


        // Search RM OCA records for protobuf record
        for (ulRmOcaRecord = 0; ulRmOcaRecord < pOcaData->rmOcaRecordCount(); ulRmOcaRecord++)
        {
            // Get the next RM OCA record
            pRmOcaRecord = pOcaData->rmOcaRecord(ulRmOcaRecord);
            if (pRmOcaRecord != NULL)
            {
                // Switch on the RM OCA record type
                switch(pRmOcaRecord->cRecordType())
                {
                    case RmPrbErrorInfo:                // Protobuf error record
                    case RmProtoBuf:                    // Protobuf
                    case RmProtoBuf_V2:                 // Protobuf + LwDump
                    case RmPrbFullDump:                 // Full LwDump protobuf message

                        // Set the RM protobuf record
                        pRmProtoBuf = pRmProtoBufRecord;

                        break;
                }
                // Check for protobuf record found (Stop search if found)
                if (pRmProtoBuf != NULL)
                    break;
            }
        }
    }
    return pRmProtoBuf;

} // findRmProtoBuf

//******************************************************************************

const void*
rmProtoBufData()
{
    const CRmProtoBufRecord *pRmProtoBufRecord = findRmProtoBuf();
    const void         *pRmProtoBufData = NULL;

    // Check for RM protobuf record available
    if (pRmProtoBufRecord != NULL)
    {
        // Compute the address of the actual RM protobuf data
        pRmProtoBufData = constvoidptr(constbyteptr(pRmProtoBufRecord->rmProtoBufRecord()) + pRmProtoBufRecord->type().size());
    }
    return pRmProtoBufData;

} // rmProtoBufData

//******************************************************************************

ULONG
rmProtoBufSize()
{
    const CRmProtoBufRecord* pRmProtoBufRecord = findRmProtoBuf();
    ULONG               ulRmProtoBufSize = 0;

    // Check for RM protobuf record available
    if (pRmProtoBufRecord != NULL)
    {
        // Get the actual RM protobuf data size
        ulRmProtoBufSize = pRmProtoBufRecord->dwSize();
    }
    return ulRmProtoBufSize;

} // rmProtoBufSize

//******************************************************************************

ULONG
ocaAdapterCount()
{
    const COcaDataPtr   pOcaData = ocaData();
    const CLwcdRecord  *pDrvOcaRecord;
    ULONG               ulDrvOcaRecord;
    ULONG               ulAdapters = 0;

    // Loop thru the driver (KMD) OCA records looking for adapter records
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the OCA adapter type
            if (pDrvOcaRecord->cRecordType() == KmdAdapterInfo)
            {
                // Increment the OCA adapter count
                ulAdapters++;
            }
        }
    }
    return ulAdapters;

} // ocaAdapterCount

//******************************************************************************

ULONG
ocaEngineIdCount
(
    ULONG64             ulAdapter
)
{
    const COcaDataPtr   pOcaData = ocaData();
    const CLwcdRecord  *pDrvOcaRecord;
    const CLwcdRecord  *pNextOcaRecord;
    const CAdapterOcaRecord *pAdapterOcaRecord;
    ULONG               ulDrvOcaRecord;
    ULONG               ulNextOcaRecord;
    ULONG               ulOcaAdapter = 0;
    ULONG               ulEngineIds = 0;

    // Loop thru the driver (KMD) OCA records looking for adapter record
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the OCA adapter type
            if (pDrvOcaRecord->cRecordType() == KmdAdapterInfo)
            {
                // Get the OCA adapter record and check if it is the requested one
                pAdapterOcaRecord = static_cast<const CAdapterOcaRecord *>(pDrvOcaRecord);
                if (pAdapterOcaRecord != NULL)
                {
                    // Check to see if this is the requested adapter
                    if ((pAdapterOcaRecord->Adapter().ptr() == ulAdapter) || (ulOcaAdapter == ulAdapter))
                    {
                        // Search next records for engine ID records
                        for (ulNextOcaRecord = (ulDrvOcaRecord + 1); ulNextOcaRecord < pOcaData->drvOcaRecordCount(); ulNextOcaRecord++)
                        {
                            // Get the next driver OCA record
                            pNextOcaRecord = pOcaData->drvOcaRecord(ulNextOcaRecord);
                            if (pNextOcaRecord != NULL)
                            {
                                // Check the driver record for the OCA engine ID type
                                if (pNextOcaRecord->cRecordType() == KmdEngineIdInfo)
                                {
                                    // Increment the engine ID count
                                    ulEngineIds++;
                                }
                                else    // No more engine ID records (contiguous)
                                {
                                    // Stop the engine ID search
                                    break;
                                }
                            }
                        }
                        break;
                    }
                    // Increment the OCA adapter index
                    ulOcaAdapter++;
                }
            }
        }
    }
    return ulEngineIds;

} // ocaEngineIdCount

//******************************************************************************

ULONG
ocaResetEngineCount
(
    ULONG64             ulAdapter
)
{
    const COcaDataPtr   pOcaData = ocaData();
    const CErrorInfoData *pErrorInfoData;
    const CErrorInfoOcaRecord *pErrorInfoOcaRecord;
    const CAdapterOcaRecord *pAdapterOcaRecord;
    ULONG               ulErrorData;
    ULONG               ulResetEngineCount = 0;

    // Try to find the adapter OCA record for the requested adapter
    pAdapterOcaRecord = findOcaAdapter(ulAdapter);
    if (pAdapterOcaRecord != NULL)
    {
        // Check for OCA error records available
        pErrorInfoOcaRecord = findOcaErrorRecords();
        if (pErrorInfoOcaRecord != NULL)
        {
            // Search error records for reset engine error
            for (ulErrorData = 0; ulErrorData < pErrorInfoOcaRecord->errorDataCount(); ulErrorData++)
            {
                // Get the next error data record to check
                pErrorInfoData = pErrorInfoOcaRecord->errorInfoData(ulErrorData);
                if (pErrorInfoData != NULL)
                {
                    // Check to see if this is a reset engine error record
                    if (pErrorInfoData->description().find("ResetEngine oclwrred") != NOT_FOUND)
                    {
                        // Found a reset engine error record, check the adapter
                        if (pErrorInfoData->adapter().ptr() == pAdapterOcaRecord->Adapter().ptr())
                        {
                            // Increment the reset engine count for this adapter
                            ulResetEngineCount++;
                        }
                    }
                }
            }
        }
    }
    return ulResetEngineCount;

} // ocaResetEngineCount

//******************************************************************************

CKmdProcessOcaRecord**
findResetEngineProcesses
(
    ULONG64             ulAdapter
)
{
    ULONG               ulResetEngineCount = ocaResetEngineCount(ulAdapter);
    ULONG               ulResetEngine = 0;
    ULONG               ulErrorData;
    const COcaDataPtr   pOcaData = ocaData();
    const CErrorInfoData *pErrorInfoData;
    const CErrorInfoOcaRecord *pErrorInfoOcaRecord;
    const CAdapterOcaRecord *pAdapterOcaRecord;
    const CDeviceOcaRecord *pDeviceOcaRecord;
    const CContextOcaRecord *pContextOcaRecord;
    const CKmdProcessOcaRecord *pKmdProcessOcaRecord;
    CKmdProcessOcaRecord** pProcessArray = NULL;

    // Check for reset engine records present
    if (ulResetEngineCount != 0)
    {
        // Try to allocate the process array
        pProcessArray = new CKmdProcessOcaRecord*[ulResetEngineCount];
        if (pProcessArray != NULL)
        {
            // Initialize the process array
            memset(pProcessArray, 0, (ulResetEngineCount * sizeof(pKmdProcessOcaRecord)));

            // Try to find the adapter OCA record for the requested adapter
            pAdapterOcaRecord = findOcaAdapter(ulAdapter);
            if (pAdapterOcaRecord != NULL)
            {
                // Check for OCA error records available
                pErrorInfoOcaRecord = findOcaErrorRecords();
                if (pErrorInfoOcaRecord != NULL)
                {
                    // Search error records for reset engine error
                    for (ulErrorData = 0; ulErrorData < pErrorInfoOcaRecord->errorDataCount(); ulErrorData++)
                    {
                        // Get the next error data record to check
                        pErrorInfoData = pErrorInfoOcaRecord->errorInfoData(ulErrorData);
                        if (pErrorInfoData != NULL)
                        {
                            // Check to see if this is a reset engine error record
                            if (pErrorInfoData->description().find("ResetEngine oclwrred") != NOT_FOUND)
                            {
                                // Found a reset engine error record, check the adapter
                                if (pErrorInfoData->adapter().ptr() == pAdapterOcaRecord->Adapter().ptr())
                                {
                                    // Try to find the context for this error record (Use channel for search)
                                    pContextOcaRecord = findOcaContext(pErrorInfoData->channel().ptr());
                                    if (pContextOcaRecord != NULL)
                                    {
                                        // Try to find the device for this context record
                                        pDeviceOcaRecord = findOcaDevice(pContextOcaRecord->Device().ptr());
                                        if (pDeviceOcaRecord != NULL)
                                        {
                                            // Try to find the process for this device record
                                            pKmdProcessOcaRecord = findOcaProcess(pDeviceOcaRecord->Process().ptr());
                                            if (pKmdProcessOcaRecord != NULL)
                                            {
                                                // Save this KMD process record in the process array
                                                pProcessArray[ulResetEngine++] = const_cast<CKmdProcessOcaRecord*>(pKmdProcessOcaRecord);

                                                // Check for all reset engine records found
                                                if (ulResetEngine == ulResetEngineCount)
                                                {
                                                    // Stop the error record search
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return pProcessArray;

} // findResetEngineProcesses

//******************************************************************************

void
displayOcaResetEngineProcesses
(
    ULONG               ulResetEngineCount,
    CKmdProcessOcaRecord** pProcessArray
)
{
    ULONG               ulResetEngine;

    // Loop displaying the reset engine processes
    for (ulResetEngine = 0; ulResetEngine < ulResetEngineCount; ulResetEngine++)
    {
        // Check for process record found
        if (pProcessArray[ulResetEngine] != NULL)
        {
            // Check for unique reset engine process (No duplicates)
            if (uniqueOcaProcess(ulResetEngine, pProcessArray))
            {
                // Display this reset engine record
                dPrintf("ResetEngine for process '%s'\n", STR(pProcessArray[ulResetEngine]->processName()));
            }
        }
    }

} // displayOcaResetEngineProcesses

//******************************************************************************

bool
uniqueOcaProcess
(
    ULONG               ulResetEngine,
    CKmdProcessOcaRecord** pProcessArray
)
{
    ULONG               ulResetEngineTest;
    bool                bUnique = true;

    // Loop checking to see if requested reset engine process is unique
    for (ulResetEngineTest = 0; ulResetEngineTest < ulResetEngine; ulResetEngineTest++)
    {
        // Check for a duplicate reset engine process
        if (pProcessArray[ulResetEngineTest] == pProcessArray[ulResetEngine])
        {
            // Indicate process is not unique and stop search
            bUnique = false;

            break;
        }
    }
    return bUnique;

} // uniqueOcaProcess

} // oca namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
