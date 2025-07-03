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
|*  Module: osname.h                                                          *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OSNAME_H
#define _OSNAME_H

//******************************************************************************
//
// os namespace entries
//
//******************************************************************************
// In ostypes.h
using os::THREAD;
using os::PROCESS;

using os::CDispatcherHeader;
using os::CKThread;
using os::CKThreads;
using os::CEThread;
using os::CEThreads;
using os::CKProcess;
using os::CKProcesses;
using os::CEProcess;
using os::CEProcesses;

using os::CDispatcherHeaderPtr;
using os::CKThreadPtr;
using os::CKThreadsPtr;
using os::CEThreadPtr;
using os::CEThreadsPtr;
using os::CKProcessPtr;
using os::CKProcessesPtr;
using os::CEProcessPtr;
using os::CEProcessesPtr;

using os::CKThreadArray;
using os::CEThreadArray;
using os::CThreadBase;
using os::CKProcessArray;
using os::CEProcessArray;
using os::CProcessBase;

using os::CDispatcherHeaderObject;
using os::CKThreadObject;
using os::CEThreadObject;
using os::CKProcessObject;
using os::CEProcessObject;

using os::CDispatcherHeaderList;
using os::CKThreadList;
using os::CEThreadList;
using os::CKProcessList;
using os::CEProcessList;

// In osdispatcher.h
using os::createDispatcherHeader;

// In osheader.h
using os::getImageDosHeader;
using os::getImageFileHeader;
using os::getImageExportDirectory;

// In osthread.h
using os::createKThread;
using os::createEThread;

using os::getKThreads;
using os::getKernelThreads;

using os::getEThreads;
using os::getExelwtiveThreads;

using os::findKThread;
using os::findEThread;

// In osprocess.h
using os::createKProcess;
using os::createEProcess;

using os::getKProcesses;
using os::getKernelProcesses;

using os::getEProcesses;
using os::getExelwtiveProcesses;

using os::findKProcess;
using os::findEProcess;

// In ossession.h
using os::createSession;

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OSNAME_H
