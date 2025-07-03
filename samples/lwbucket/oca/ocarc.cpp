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
|*  Module: ocarc.cpp                                                         *|
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





//******************************************************************************
//
//  Locals
//
//******************************************************************************

// Define the engine table
static  ENGINE_DATA s_RcEngine[] =  {
                                     {RC_GRAPHICS_ENGINE,       "GR"},
                                     {RC_SEC_ENGINE,            "SEC"},
                                     {RC_COPY0_ENGINE,          "CE0"},
                                     {RC_COPY1_ENGINE,          "CE1"},
                                     {RC_COPY2_ENGINE,          "CE2"},
                                     {RC_MSPDEC_ENGINE,         "MSPDEC"},
                                     {RC_MSPPP_ENGINE,          "MSPPP"},
                                     {RC_MSVLD_ENGINE,          "MSLVD"},
                                     {RC_HOST_ENGINE,           "HOST"},
                                     {RC_DISPLAY_ENGINE,        "DISP"},
                                     {RC_CAPTURE_ENGINE,        "CAP"},
                                     {RC_PERF_MON_ENGINE,       "PERF"},
                                     {RC_PMU_ENGINE,            "PMU"},
                                     {RC_LWENC0_ENGINE,         "LWENC0"},
                                     {RC_SEC2_ENGINE,           "SEC2"},
                                     {RC_LWDEC_ENGINE,          "LWDEC"},
                                     {RC_LWENC1_ENGINE,         "LWENC1"},
                                     {RC_COPY3_ENGINE,          "CE3"},
                                     {RC_COPY4_ENGINE,          "CE4"},
                                     {RC_COPY5_ENGINE,          "CE5"},
                                     {RC_LWENC2_ENGINE,         "LWENC2"},
                                     {RC_COPY6_ENGINE,          "CE6"},
                                     {RC_COPY7_ENGINE,          "CE7"},
                                     {RC_COPY8_ENGINE,          "CE8"},
                                    };

//******************************************************************************

CString
rcType
(
    ULONG               ulRcType
)
{
    CString             sRcType;

    // Switch on the RC type value
    switch(ulRcType)
    {
        case RC_FIFO_ERROR_FIFO_METHOD:                 sRcType = "FIFO method error";                      break;
        case RC_FIFO_ERROR_SW_METHOD:                   sRcType = "SW method error";                        break;
        case RC_FIFO_ERROR_UNK_METHOD:                  sRcType = "Unknown method error";                   break;
        case RC_FIFO_ERROR_CHANNEL_BUSY:                sRcType = "Channel busy error";                     break;
        case RC_FIFO_ERROR_RUNOUT_OVERFLOW:             sRcType = "Runout overflow error";                  break;
        case RC_FIFO_ERROR_PARSE_ERR:                   sRcType = "Parse error";                            break;
        case RC_FIFO_ERROR_PTE_ERR:                     sRcType = "PTE error";                              break;
        case RC_FIFO_ERROR_IDLE_TIMEOUT:                sRcType = "Idle timeout error";                     break;
        case RC_GR_ERROR_INSTANCE:                      sRcType = "GR instance error";                      break;
        case RC_GR_ERROR_SINGLE_STEP:                   sRcType = "GR single step error";                   break;
        case RC_GR_ERROR_MISSING_HW:                    sRcType = "GR missing HW error";                    break;
        case RC_GR_ERROR_SW_METHOD:                     sRcType = "GR SW method error";                     break;
        case RC_GR_ERROR_SW_NOTIFY:                     sRcType = "GR SW notify error";                     break;
        case RC_FAKE_ERROR:                             sRcType = "Fake error";                             break;
        case RC_SCANLINE_TIMEOUT:                       sRcType = "Scanline timeout error";                 break;
        case RC_VBLANK_CALLBACK_TIMEOUT:                sRcType = "Vblank calback timeout error";           break;
        case RC_PARAMETER_ERROR:                        sRcType = "Parameter error";                        break;
        case RC_BUS_MASTER_TIMEOUT_ERROR:               sRcType = "Bus master timeout error";               break;
        case RC_DISP_MISSED_NOTIFIER:                   sRcType = "Display missed notifier error";          break;
        case RC_MPEG_ERROR_SW_METHOD:                   sRcType = "MPEG SW method error";                   break;
        case RC_ME_ERROR_SW_METHOD:                     sRcType = "ME SW method error";                     break;
        case RC_VP_ERROR_SW_METHOD:                     sRcType = "VP SW method error";                     break;
        case RC_RC_LOGGING_ENABLED:                     sRcType = "RC logging enabled error";               break;
        case RC_GR_SEMAPHORE_TIMEOUT:                   sRcType = "GR semaphore timeout error";             break;
        case RC_GR_ILLEGAL_NOTIFY:                      sRcType = "GR illegal notify error";                break;
        case RC_FIFO_ERROR_FBISTATE_TIMEOUT:            sRcType = "FBI state timeout error";                break;
        case RC_VP_ERROR:                               sRcType = "VP error";                               break;
        case RC_VP2_ERROR:                              sRcType = "VP2 error";                              break;
        case RC_BSP_ERROR:                              sRcType = "BSP error";                              break;
        case RC_BAD_ADDR_ACCESS:                        sRcType = "Bad address access error";               break;
        case RC_FIFO_ERROR_MMU_ERR_FLT:                 sRcType = "FIFO MMU error";                         break;
        case RC_PBDMA_ERROR:                            sRcType = "PBDMA error";                            break;
        case RC_SEC_ERROR:                              sRcType = "SEC error";                              break;
        case RC_MSVLD_ERROR:                            sRcType = "MSVLD error";                            break;
        case RC_MSPDEC_ERROR:                           sRcType = "MSPDEC error";                           break;
        case RC_MSPPP_ERROR:                            sRcType = "MSPPP error";                            break;
        case RC_FECS_ERR_UNIMP_FIRMWARE_METHOD:         sRcType = "FECS firmware method error";             break;
        case RC_FECS_ERR_WATCHDOG_TIMEOUT:              sRcType = "FECS watchdog timeout error";            break;
        case RC_CE0_ERROR:                              sRcType = "CE0 error";                              break;
        case RC_CE1_ERROR:                              sRcType = "CE1 error";                              break;
        case RC_CE2_ERROR:                              sRcType = "CE2 error";                              break;
        case RC_VIC_ERROR:                              sRcType = "VIC error";                              break;
        case RC_RESETCHANNEL_VERIF_ERROR:               sRcType = "Reset channel verif error";              break;
        case RC_GR_FAULT_DURING_CTXSW:                  sRcType = "GR fault during context switch error";   break;
        case RC_PREEMPTIVE_REMOVAL:                     sRcType = "Preemptive removal error";               break;
        case RC_GPU_TIMEOUT_ERROR:                      sRcType = "GPU timeout error";                      break;
        case RC_LWENC0_ERROR:                           sRcType = "LWENC0 error";                           break;
        case RC_GPU_ECC_DBE:                            sRcType = "GPU ECC DBE error";                      break;
        case RC_SR_CONSTANT_LEVEL_SET_BY_REGISTRY:      sRcType = "SR constant level error";                break;
        case RC_SR_LEVEL_TRANSITION_DUE_TO_RC_ERROR:    sRcType = "SR level transition due to RC";          break;
        case RC_SR_STRESS_TEST_FAILURE:                 sRcType = "SR stress test failure";                 break;
        case RC_SR_LEVEL_TRANS_DUE_TO_TEMP_RISE:        sRcType = "SR level transition due to temp";        break;
        case RC_SR_TEMP_REDUCED_CLOCKING:               sRcType = "SR temp reduced clocking";               break;
        case RC_SR_PWR_REDUCED_CLOCKING:                sRcType = "SR power reduced clocking";              break;
        case RC_SR_TEMPERATURE_READ_ERROR:              sRcType = "SR temperature read error";              break;
        case RC_DISPLAY_CHANNEL_EXCEPTION:              sRcType = "Display channel exception";              break;
        case RC_FB_LINK_TRAINING_FAILURE_ERROR:         sRcType = "FB link training failure";               break;
        case RC_FB_MEMORY_ERROR:                        sRcType = "FB memory error";                        break;
        case RC_PMU_ERROR:                              sRcType = "PMU error";                              break;
        case RC_SEC2_ERROR:                             sRcType = "SEC2 error";                             break;
        case RC_PMU_BREAKPOINT:                         sRcType = "PMU breakpoint";                         break;
        case RC_PMU_HALT_ERROR:                         sRcType = "PMU halt error";                         break;
        case RC_INFOROM_PAGE_RETIREMENT_EVENT:          sRcType = "Page retirement event";                  break;
        case RC_INFOROM_PAGE_RETIREMENT_FAILURE:        sRcType = "Page retirement failure";                break;
        case RC_LWENC1_ERROR:                           sRcType = "LWENC1 error";                           break;
        case RC_FECS_ERR_REG_ACCESS_VIOLATION:          sRcType = "FECS register access violation";         break;
        case RC_FECS_ERR_VERIF_VIOLATION:               sRcType = "FECS verif violation";                   break;
        case RC_LWDEC_ERROR:                            sRcType = "LWDEC error";                            break;
        case RC_GR_CLASS_ERROR:                         sRcType = "GR class error";                         break;
        case RC_CE3_ERROR:                              sRcType = "CE3 error";                              break;
        case RC_CE4_ERROR:                              sRcType = "CE4 error";                              break;
        case RC_CE5_ERROR:                              sRcType = "CE5 error";                              break;
        case RC_LWENC2_ERROR:                           sRcType = "LWENC2 error";                           break;
        case RC_LWLINK_LINK_DISABLED:                   sRcType = "LwLink disabled";                        break;
        case RC_CE6_ERROR:                              sRcType = "CE6 error";                              break;
        case RC_CE7_ERROR:                              sRcType = "CE7 error";                              break;
        case RC_CE8_ERROR:                              sRcType = "CE8 error";                              break;
        case RC_VGPU_START_ERROR:                       sRcType = "VGPU start error";                       break;
        case RC_GPU_HAS_FALLEN_OFF_THE_BUS:             sRcType = "GPU fallen off bus error";               break;
        case RC_PBDMA_PUSHBUFFER_CRC_MISMATCH:          sRcType = "Pushbuffer CRC mismatch";                break;

        default:                                        sRcType = foreground("Unknown error", RED);         break;
    }
    return sRcType;

} // rcType

//******************************************************************************

CString
rcLevel
(
    ULONG               ulRcLevel
)
{
    CString             sRcLevel;

    // Switch on the RC level value
    switch(ulRcLevel)
    {
        case RC_LEVEL_INFO:         sRcLevel = "Info RC error";                     break;
        case RC_LEVEL_NON_FATAL:    sRcLevel = "Non-fatal RC error";                break;
        case RC_LEVEL_FATAL:        sRcLevel = "Fatal RC error";                    break;

        default:                    sRcLevel = foreground("Unknown RC error", RED); break;
    }
    return sRcLevel;

} // rcLevel

//******************************************************************************

CString
rcEngine
(
    ULONG               ulRcEngine
)
{
    ULONG               ulEngine;
    CString             sEngine(MAX_DBGPRINTF_STRING);
    CString             sRcEngine;

    // Loop checking all the RC engines
    for (ulEngine = 0; ulEngine < countof(s_RcEngine); ulEngine++)
    {
        // Check to see if next engine should be included
        if ((ulRcEngine & s_RcEngine[ulEngine].ulEngineId) == s_RcEngine[ulEngine].ulEngineId)
        {
            // Include this engine in the engine string
            if (sRcEngine.empty())
            {
                // This is the first engine
                sRcEngine = s_RcEngine[ulEngine].pEngineString;
            }
            else    // Not the first engine
            {
                // Add this engine
                sEngine.sprintf(" | %s", s_RcEngine[ulEngine].pEngineString);
                sRcEngine.append(sEngine);
            }
            // Remove this engine from the engine value
            ulRcEngine &= ~s_RcEngine[ulEngine].ulEngineId;
        }
    }
    // Check for any unknown engines left (Non-zero engine value)
    if (ulRcEngine != 0)
    {
        // Add the unknown engine(s)
        sEngine.sprintf(" | Unknown (0x%x)", ulRcEngine);
        sRcEngine.append(sEngine);        
    }
    return sRcEngine;

} // rcEngine

//******************************************************************************

ULONG
ocaRcErrorCount
(
    ULONG64             ulAdapter
)
{
    const COcaDataPtr   pOcaData = ocaData();
    const CErrorInfoData *pErrorInfoData;
    const CErrorInfoOcaRecord *pErrorInfoOcaRecord;
    const CAdapterOcaRecord *pAdapterOcaRecord;
    ULONG               ulErrorData;
    ULONG               ulRcErrorCount = 0;

    // Try to find the adapter OCA record for the requested adapter
    pAdapterOcaRecord = findOcaAdapter(ulAdapter);
    if (pAdapterOcaRecord != NULL)
    {
        // Check for OCA error records available
        pErrorInfoOcaRecord = findOcaErrorRecords();
        if (pErrorInfoOcaRecord != NULL)
        {
            // Search error records for RC recovery error
            for (ulErrorData = 0; ulErrorData < pErrorInfoOcaRecord->errorDataCount(); ulErrorData++)
            {
                // Get the next error data record to check
                pErrorInfoData = pErrorInfoOcaRecord->errorInfoData(ulErrorData);
                if (pErrorInfoData != NULL)
                {
                    // Check to see if this is a RC recovery error record
                    if (pErrorInfoData->description().find("RC error recovery") != NOT_FOUND)
                    {
                        // Found a RC recovery error record, check the adapter
                        if (pErrorInfoData->adapter().ptr() == pAdapterOcaRecord->Adapter().ptr())
                        {
                            // Increment the RC error count for this adapter
                            ulRcErrorCount++;
                        }
                    }
                }
            }
        }
    }
    return ulRcErrorCount;

} // ocaRcErrorCount

//******************************************************************************

CErrorInfoData**
findRcErrorRecords
(
    ULONG64             ulAdapter
)
{
    ULONG               ulRcErrorCount = ocaRcErrorCount(ulAdapter);
    ULONG               ulRcError = 0;
    ULONG               ulErrorData;
    const COcaDataPtr   pOcaData = ocaData();
    const CErrorInfoData *pErrorInfoData;
    const CErrorInfoOcaRecord *pErrorInfoOcaRecord;
    const CAdapterOcaRecord *pAdapterOcaRecord;
    CErrorInfoData** pRcErrorArray = NULL;

    // Check for RC error records present
    if (ulRcErrorCount != 0)
    {
        // Try to allocate the RC error array
        pRcErrorArray = new CErrorInfoData*[ulRcErrorCount];
        if (pRcErrorArray != NULL)
        {
            // Initialize the RC error array
            memset(pRcErrorArray, 0, (ulRcErrorCount * sizeof(pErrorInfoData)));

            // Try to find the adapter OCA record for the requested adapter
            pAdapterOcaRecord = findOcaAdapter(ulAdapter);
            if (pAdapterOcaRecord != NULL)
            {
                // Check for OCA error records available
                pErrorInfoOcaRecord = findOcaErrorRecords();
                if (pErrorInfoOcaRecord != NULL)
                {
                    // Search error records for RC recovery errors
                    for (ulErrorData = 0; ulErrorData < pErrorInfoOcaRecord->errorDataCount(); ulErrorData++)
                    {
                        // Get the next error data record to check
                        pErrorInfoData = pErrorInfoOcaRecord->errorInfoData(ulErrorData);
                        if (pErrorInfoData != NULL)
                        {
                            // Check to see if this is a RC recovery error record
                            if (pErrorInfoData->description().find("RC error recovery") != NOT_FOUND)
                            {
                                // Found a RC recovery error record, check the adapter
                                if (pErrorInfoData->adapter().ptr() == pAdapterOcaRecord->Adapter().ptr())
                                {
                                    // Save this error info record in the RC error array
                                    pRcErrorArray[ulRcError++] = const_cast<CErrorInfoData*>(pErrorInfoData);

                                    // Check for all RC error records found
                                    if (ulRcError == ulRcErrorCount)
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
    return pRcErrorArray;

} // findRcErrorRecords

//******************************************************************************

void
displayOcaRcErrors
(
    ULONG               ulRcErrorCount,
    CErrorInfoData   **pRcErrorArray
)
{
    ULONG               ulRcError;
    const CChannelOcaRecord *pChannelOcaRecord;
    const CContextOcaRecord *pContextOcaRecord;
    const CDeviceOcaRecord *pDeviceOcaRecord;
    const CKmdProcessOcaRecord *pKmdProcessOcaRecord = NULL;

    // Loop displaying the RC recovery errors
    for (ulRcError = 0; ulRcError < ulRcErrorCount; ulRcError++)
    {
        // Check for error record found
        if (pRcErrorArray[ulRcError] != NULL)
        {
            // Try to find the channel for this error record (Use data0 which is RM client handle)
            pChannelOcaRecord = findOcaChannel(pRcErrorArray[ulRcError]->data0());
            if (pChannelOcaRecord != NULL)
            {
                // Try to find the context for this channel record
                pContextOcaRecord = findOcaContext(pChannelOcaRecord->Channel().ptr());
                if (pContextOcaRecord != NULL)
                {
                    // Try to find the device for this context record
                    pDeviceOcaRecord = findOcaDevice(pContextOcaRecord->Device().ptr());
                    if (pDeviceOcaRecord != NULL)
                    {
                        // Try to find the process for this device record
                        pKmdProcessOcaRecord = findOcaProcess(pDeviceOcaRecord->Process().ptr());
                    }
                }
            }
            // Check to see if corresponding process found
            if (pKmdProcessOcaRecord != NULL)
            {
                // Display this RC recovery error w/process
                dPrintf("%s - %s (%s)\n", DML(rcLevel(static_cast<ULONG>(pRcErrorArray[ulRcError]->data3()))),
                                          DML(rcType(static_cast<ULONG>(pRcErrorArray[ulRcError]->data4()))),
                                          STR(pKmdProcessOcaRecord->processName()));
            }
            else    // No process found
            {
                // Display this RC recovery error
                dPrintf("%s - %s\n", DML(rcLevel(static_cast<ULONG>(pRcErrorArray[ulRcError]->data3()))),
                                     DML(rcType(static_cast<ULONG>(pRcErrorArray[ulRcError]->data4()))));
            }




        }
    }

} // displayOcaRcErrors

} // oca namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
