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
|*  Module: ostypes.h                                                         *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OSTYPES_H
#define _OSTYPES_H

//******************************************************************************
//
//  os namespace
//
//******************************************************************************
namespace os
{
//******************************************************************************
//
//  Fowards
//
//******************************************************************************
class CGuid;
class CMdl;
class CSingleListEntry;
class CListEntry;
class CDispatcherHeader;
class CKThread;
class CKThreads;
class CEThread;
class CEThreads;
class CKProcess;
class CKProcesses;
class CEProcess;
class CEProcesses;
class CSession;

//******************************************************************************
//
//  Template type definitions (Mostly smart pointer definitions)
//
//******************************************************************************
typedef TARGET_PTR<ThreadPointer>   THREAD;
typedef TARGET_PTR<ProcessPointer>  PROCESS;
typedef TARGET_PTR<SessionPointer>  SESSION;

typedef CRefPtr<CGuid>              CGuidPtr;
typedef CRefPtr<CMdl>               CMdlPtr;
typedef CRefPtr<CSingleListEntry>   CSingleListEntryPtr;
typedef CRefPtr<CListEntry>         CListEntryPtr;
typedef CRefPtr<CDispatcherHeader>  CDispatcherHeaderPtr;
typedef CRefPtr<CKThread>           CKThreadPtr;
typedef CRefPtr<CKThreads>          CKThreadsPtr;
typedef CRefPtr<CEThread>           CEThreadPtr;
typedef CRefPtr<CEThreads>          CEThreadsPtr;
typedef CRefPtr<CKProcess>          CKProcessPtr;
typedef CRefPtr<CKProcesses>        CKProcessesPtr;
typedef CRefPtr<CEProcess>          CEProcessPtr;
typedef CRefPtr<CEProcesses>        CEProcessesPtr;
typedef CRefPtr<CSession>           CSessionPtr;

typedef CArrayPtr<CKThreadPtr>      CKThreadArray;
typedef CArrayPtr<CEThreadPtr>      CEThreadArray;
typedef CArrayPtr<THREAD>           CThreadBase;
typedef CArrayPtr<CKProcessPtr>     CKProcessArray;
typedef CArrayPtr<CEProcessPtr>     CEProcessArray;
typedef CArrayPtr<PROCESS>          CProcessBase;

typedef LWnqObj<CDispatcherHeader>  CDispatcherHeaderObject;
typedef LWnqObj<CKThread>           CKThreadObject;
typedef LWnqObj<CEThread>           CEThreadObject;
typedef LWnqObj<CKProcess>          CKProcessObject;
typedef LWnqObj<CEProcess>          CEProcessObject;
typedef LWnqObj<CSession>           CSessionObject;

typedef CObjList<CDispatcherHeader> CDispatcherHeaderList;
typedef CObjList<CKThread>          CKThreadList;
typedef CObjList<CEThread>          CEThreadList;
typedef CObjList<CKProcess>         CKProcessList;
typedef CObjList<CEProcess>         CEProcessList;
typedef CObjList<CSession>          CSessionList;

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OSTYPES_H
