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
|*  Module: event.h                                                           *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _EVENT_H
#define _EVENT_H

//******************************************************************************
//
//  Constants
//
//******************************************************************************







//******************************************************************************
//
//  Type Definitions
//
//******************************************************************************






//******************************************************************************
//
//  Forwards
//
//******************************************************************************






//******************************************************************************
//
// Structures
//
//******************************************************************************







//******************************************************************************
//
// Macros
//
//******************************************************************************




//******************************************************************************
//
//  Extension Event Class
//
//******************************************************************************
class CExtensionEvent : public CEvent
{
public:
                        CExtensionEvent() : CEvent(){};
virtual                ~CExtensionEvent()           {};

        // Extension event methods
virtual HRESULT         breakpoint(PDEBUG_BREAKPOINT Bp) const;
virtual HRESULT         exception(PEXCEPTION_RECORD64 Exception, ULONG FirstChance) const;

virtual HRESULT         createThread(ULONG64 Handle, ULONG64 DataOffset, ULONG64 StartOffset) const;
virtual HRESULT         exitThread(ULONG ExitCode) const;

virtual HRESULT         createProcess(ULONG64 ImageFileHandle, ULONG64 Handle, ULONG64 BaseOffset, ULONG ModuleSize, PCSTR ModuleName, PCSTR ImageName, ULONG CheckSum, ULONG TimeDateStamp, ULONG64 InitialThreadHandle, ULONG64 ThreadDataOffset, ULONG64 StartOffset) const;
virtual HRESULT         exitProcess(ULONG ExitCode) const;

virtual HRESULT         loadModule(ULONG64 ImageFileHandle, ULONG64 BaseOffset, ULONG ModuleSize, PCSTR ModuleName, PCSTR ImageName, ULONG CheckSum, ULONG TimeDateStamp) const;
virtual HRESULT         unloadModule(PCSTR ImageBaseName, ULONG64 BaseOffset) const;

virtual HRESULT         systemError(ULONG Error, ULONG Level) const;

virtual HRESULT         sessionStatus(ULONG Status) const;

virtual HRESULT         changeDebuggeeState(ULONG Flags, ULONG64 Argument) const;
virtual HRESULT         changeEngineState(ULONG Flags, ULONG64 Argument) const;
virtual HRESULT         changeSymbolState(ULONG Flags, ULONG64 Argument) const;

}; // class CExtensionEvent

//******************************************************************************
//
//  Functions
//
//******************************************************************************









//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _EVENT_H
