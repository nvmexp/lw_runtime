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
|*  Module: logging.h                                                         *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _LOGGING_H
#define _LOGGING_H

//******************************************************************************
//
// Constants
//
//******************************************************************************
// Known general log types (Event, Warning, and Error)
#define DM_INTERFACE                                "_DM_DDI"
#define LDDM_INTERFACE                              "_LDDM_DDI"
#define DL_INTERFACE                                "_DL"
#define CB_INTERFACE                                "_LDDM_CB"
#define IFACE_INTERFACE                             "_IFACE"
#define RM_INTERFACE                                "_RM_API"
#define AGP_INTERFACE                               "_AGP"
#define TIMED_OP_INTERFACE                          "_TIMED_OP"

#define CLIENT_ARB_TYPE                             "_CLIENTARB"
#define GDI_ACCEL_OP_TYPE                           "_GDI_HW_ACCEL"

// Known event log types
#define BUFFER_OPERATION                            "EVENT_BUFFER"
#define IDLE_OPERATION                              "EVENT_GENERAL_IDLE"
#define ISR_NOTIFY                                  "EVENT_ISR_NOTIFY"

// Known paging log types
#define BUILD_PAGING_BUFFER                         "_BUILD_PAGING_BUFFER"

// Invalid logging/paging/profile index
#define NO_EVENT_INDEX                              0xffffffff

// Invalid log index
#define ILWALID_LOG_INDEX                           0xffffffff

// Exit type event flag value (High bit)
#define EXIT_FLAG                                   0x80000000

// Maximum number of logs
#define MAX_LOGS                                    8

// Maximum number of log modules
#define MAX_LOG_MODULES                             32

// Invalid log values
#define ILWALID_LOG                                 MAX_LOGS
#define ILWALID_MODULE                              MAX_LOG_MODULES
#define ILWALID_TYPE                                0xffffffff
#define ILWALID_SUBTYPE                             0xffffffff

// Define the special log entry formatting characters
#define LOG_FORMAT_ENTRY                            '~'
#define LOG_FORMAT_OPEN                             '('
#define LOG_FORMAT_CLOSE                            ')'
#define LOG_FORMAT_SEPARATOR                        ':'
#define LOG_FORMAT_TYPE                             'T'
#define LOG_FORMAT_SUBTYPE                          't'
#define LOG_FORMAT_STATUS                           'S'
#define LOG_FORMAT_TIMESTAMP                        's'
#define LOG_FORMAT_ADDRESS                          'A'
#define LOG_FORMAT_ADAPTER                          'a'
#define LOG_FORMAT_DEVICE                           'd'
#define LOG_FORMAT_CONTEXT                          'c'
#define LOG_FORMAT_CHANNEL                          'C'
#define LOG_FORMAT_ALLOCATION                       'l'
#define LOG_FORMAT_PROCESS                          'p'
#define LOG_FORMAT_THREAD                           'h'
#define LOG_FORMAT_KMD_PROCESS                      'k'
#define LOG_FORMAT_DATA_0                           '0'
#define LOG_FORMAT_DATA_1                           '1'
#define LOG_FORMAT_DATA_2                           '2'
#define LOG_FORMAT_DATA_3                           '3'
#define LOG_FORMAT_DATA_4                           '4'
#define LOG_FORMAT_MODULE                           'm'
#define LOG_FORMAT_VALUE                            'v'
#define LOG_FORMAT_LOG                              'L'
#define LOG_FORMAT_EXIT                             'e'

//******************************************************************************
//
//  Macros
//
//******************************************************************************
#define ENTRY_EVENT(eventType)              (((eventType) & EXIT_FLAG) != EXIT_FLAG)
#define EXIT_EVENT(eventType)               (((eventType) & EXIT_FLAG) == EXIT_FLAG)
#define EVENT_TYPE(eventType)               ((eventType) & ~EXIT_FLAG)

//******************************************************************************
//
//  Class CLogging (Dummy class to hold log enumerations)
//
//******************************************************************************
class CLogging
{
// Logging Enum Helpers
ENUM(dmInterface)
ENUM(lddmInterface)
ENUM(dlLayerInterface)
ENUM(rmInterface)
ENUM(cbInterface)
ENUM(ifaceInterface)
ENUM(agpInterface)
ENUM(timedOpInterface)

ENUM(clientArbSubtype)
ENUM(gdiAccelOpType)

ENUM(bufferOperation)
ENUM(idleOperation)
ENUM(interrupt)

ENUM(buildPagingBuffer)

}; // class CLogging

//******************************************************************************
//
//  Logging Hook Class
//
//******************************************************************************
class CLoggingHook : public CHook
{
public:
                        CLoggingHook() : CHook(){};
virtual                ~CLoggingHook()          {};

        // Logging hook methods
virtual HRESULT         initialize(const PULONG pVersion, const PULONG pFlags);
virtual void            uninitialize(void);

}; // class CLoggingHook

//******************************************************************************
//
//  Functions
//
//******************************************************************************
// In logging.cpp
extern  bool                isDmInterfaceType(const char* pType);
extern  bool                isLddmInterfaceType(const char* pType);
extern  bool                isDlInterfaceType(const char* pType);
extern  bool                isCbInterfaceType(const char* pType);
extern  bool                isIfaceInterfaceType(const char* pType);
extern  bool                isRmInterfaceType(const char* pType);
extern  bool                isAgpInterfaceType(const char* pType);
extern  bool                isTimedOpInterfaceType(const char* pType);

extern  bool                isClientArbType(const char* pType);
extern  bool                isGdiAccelOpType(const char* pType);

extern  bool                isBufferOperationType(const char* pType);
extern  bool                isIdleOperationType(const char* pType);
extern  bool                isIsrNotifyType(const char* pType);

extern  bool                isBuildPagingBufferType(const char* pType);

extern  CString             dmInterfaceName(ULONG64 ulValue);
extern  CString             lddmInterfaceName(ULONG64 ulValue);
extern  CString             dlInterfaceName(ULONG64 ulValue);
extern  CString             cbInterfaceName(ULONG64 ulValue);
extern  CString             ifaceInterfaceName(ULONG64 ulValue);
extern  CString             rmInterfaceName(ULONG64 ulValue);
extern  CString             agpInterfaceName(ULONG64 ulValue);
extern  CString             timedOpInterfaceName(ULONG64 ulValue);

extern  CString             clientArbSubtypeName(ULONG64 ulValue);
extern  CString             gdiAccelOpTypeName(ULONG64 ulValue);

extern  CString             bufferOperationName(ULONG64 ulValue);
extern  CString             idleOperationName(ULONG64 ulValue);
extern  CString             isrNotifyName(ULONG64 ulValue);

extern  CString             buildPagingBufferName(ULONG64 ulValue);

extern  const CValue*       dmInterfaceValue(const char* pString);
extern  const CValue*       lddmInterfaceValue(const char* pString);
extern  const CValue*       dlInterfaceValue(const char* pString);
extern  const CValue*       cbInterfaceValue(const char* pString);
extern  const CValue*       ifaceInterfaceValue(const char* pString);
extern  const CValue*       rmInterfaceValue(const char* pString);
extern  const CValue*       agpInterfaceValue(const char* pString);
extern  const CValue*       timedOpInterfaceValue(const char* pString);

extern  const CValue*       clientArbSubtypeValue(const char* pString);
extern  const CValue*       gdiAccelOpTypeValue(const char* pString);

extern  const CValue*       bufferOperatiolwalue(const char* pString);
extern  const CValue*       idleOperatiolwalue(const char* pString);
extern  const CValue*       isrNotifyValue(const char* pString);

extern  const CValue*       buildPagingBufferValue(const char* pString);

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _LOGGING_H
