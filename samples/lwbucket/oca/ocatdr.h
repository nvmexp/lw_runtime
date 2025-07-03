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
|*  Module: ocatdr.h                                                          *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OCATDR_H
#define _OCATDR_H

//******************************************************************************
//
//  oca namespace
//
//******************************************************************************
namespace oca
{

//******************************************************************************
//
//  Constants
//
//******************************************************************************
// Define the TDR bugcheck codes
#define VIDEO_TDR_ERROR                                 0x116   // TDR recovery failed
#define VIDEO_TDR_TIMEOUT_DETECTED                      0x117   // Driver failed to respond in a timely fashion
#define VIDEO_ENGINE_TIMEOUT_DETECTED                   0x141   // Video engine timeout
#define VIDEO_TDR_APPLICATION_BLOCKED                   0x142   // TDR application blocked

// Define the TDR reason codes
#define TDR_REASON_UNKNOWN                              0       // Unknown reason
#define TDR_REASON_RECOVERY_DISABLED                    1       // Recovery disabled in the registry
#define TDR_REASON_CONSELWTIVE_TIMEOUT                  2       // Conselwtive TDR without a successful buffer completion
#define TDR_REASON_DDI_RESET_FROM_TIMEOUT_FAILURE       3       // Failure code from ResetFromTimeout DDI
#define TDR_REASON_DDI_RESTART_FROM_TIMEOUT_FAILURE     4       // Failure code from RestartFromTimeout DDI
#define TDR_REASON_GDI_OFF_FAILURE                      5       // Internal error (OS bug or severe lack of resources)
#define TDR_REASON_GDI_ON_FAILURE                       6       // Internal error (OS bug or severe lack of resources)
#define TDR_REASON_GDI_RESET_THREAD_NEW_FAILURE         7       // Internal error (OS bug or severe lack of resources)
#define TDR_REASON_GDI_RESET_THREAD_START_FAILURE       8       // Internal error (OS bug or severe lack of resources)
#define TDR_REASON_VIDSCHI_PREPARE_FOR_RECOVERY_FAILURE 9       // Internal error (OS bug or severe lack of resources)
#define TDR_REASON_ADAPTER_PREPARE_TO_RESET_FAILURE     10      // Prepare for reset failure (Driver failed to exit all threads in time)
#define TDR_REASON_ADAPTER_RESET_FAILURE                11      // Internal error (OS bug or severe lack of resources)
#define TDR_REASON_APCS_ARE_DISABLED                    12      // Timeout called in an improper context (Kernel APC's disabled)
#define TDR_REASON_RECOVERY_LIMIT_EXHAUSTED             13      // Recovery limit exceeded (More than 5 TDR's in 60 seconds)

//******************************************************************************
//
//  Structures
//
//******************************************************************************
typedef struct _FORCE_DATA
{
    const char         *pAnnotationString;
    const char         *pForceString;

} FORCE_DATA, *PFORCE_DATA;

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  CString                     tdrName(ULONG bugcheckCode);
extern  CString                     tdrReason(ULONG64 reasonCode);
extern  CString                     tdrDescription(ULONG bugcheckCode, ULONG64 arg1, ULONG64 arg2, ULONG64 arg3, ULONG64 arg4);
extern  bool                        tdrForced(ULONG bugcheckCode, ULONG64 arg1, ULONG64 arg2, ULONG64 arg3, ULONG64 arg4);
extern  CString                     forcedString();

extern  const CVblankInfoOcaRecord* findTdrVsync();
extern  const CGpuWatchdogEvent*    findGpuWatchdogEvent();
extern  const CAdapterOcaRecord*    findTdrAdapter();
extern  const CContextOcaRecord*    findTdrContext(const CAdapterOcaRecord* pAdapterOcaRecord);
extern  const CBufferInfoRecord*    findTdrBuffer(const CAdapterOcaRecord* pAdapterOcaRecord, bool bIgnorePreemption);
extern  const CEngineIdOcaRecord*   findTdrEngine(const CAdapterOcaRecord* pAdapterOcaRecord, bool bIgnorePreemption);

extern  const CRmProtoBufRecord*    findRmProtoBuf();
extern  const void*                 rmProtoBufData();
extern  ULONG                       rmProtoBufSize();

extern  ULONG                       ocaAdapterCount();
extern  ULONG                       ocaEngineIdCount(ULONG64 ulAdapter);
extern  ULONG                       ocaResetEngineCount(ULONG64 ulAdapter);

extern  CKmdProcessOcaRecord**      findResetEngineProcesses(ULONG64 ulAdapter);
extern  void                        displayOcaResetEngineProcesses(ULONG ulResetEngineCount, CKmdProcessOcaRecord** pProcessArray);

} // oca namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OCATDR_H
