//
//Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions
//are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
//FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
//COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
//BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//POSSIBILITY OF SUCH DAMAGE.
//

#ifndef __OSINCLUDE_H
#define __OSINCLUDE_H

//
// This file contains contains the window's specific datatypes and
// declares any windows specific functions.
//

#if !(defined(_WIN32) || defined(_WIN64))
#error Trying to include a windows specific file in a non windows build.
#endif

namespace glslang {

//
// Thread Local Storage Operations
//
typedef unsigned long OS_TLSIndex;
#define OS_ILWALID_TLS_INDEX ((unsigned long)0xFFFFFFFF)

OS_TLSIndex OS_AllocTLSIndex();
bool        OS_SetTLSValue(OS_TLSIndex nIndex, void *lpvValue);
bool        OS_FreeTLSIndex(OS_TLSIndex nIndex);

void* OS_GetTLSValue(OS_TLSIndex nIndex);

void InitGlobalLock();
void GetGlobalLock();
void ReleaseGlobalLock();

typedef unsigned int (__stdcall *TThreadEntrypoint)(void*);
void* OS_CreateThread(TThreadEntrypoint);
void OS_WaitForAllThreads(void* threads, int numThreads);

void OS_Sleep(int milliseconds);

void OS_DumpMemoryCounters();

} // end namespace glslang

#endif // __OSINCLUDE_H
