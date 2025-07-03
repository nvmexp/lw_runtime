/* Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _LWHOOK_H_
#define _LWHOOK_H_

typedef enum HookTypesEnum {
  PRE_CALL_HOOK,
  POST_CALL_HOOK,
  LW_HOOK_TYPES,
} HookTypes;

typedef enum HookSymbolsEnum {
  LW_HOOK_MEM_ALLOC,
  LW_HOOK_MEM_FREE,
  LW_HOOK_CTX_GET_LWRRENT,
  LW_HOOK_CTX_SET_LWRRENT,
  LW_HOOK_CTX_DESTROY,
  LW_HOOK_SYMBOLS,
} HookSymbols;

// One and only function to call to register a callback
// You need to dlsym this symbol in your application and call it to register
// callbacks
typedef void (*fnLwHookRegisterCallback)(HookSymbols symbol, HookTypes type,
                                         void* callback);
extern "C" {
void lwHookRegisterCallback(HookSymbols symbol, HookTypes type, void* callback);
}

// In case you want to intercept, the callbacks need the same type/parameters as
// the real functions
typedef LWresult LWDAAPI (*fnMemAlloc)(LWdeviceptr* dptr, size_t bytesize);
typedef LWresult LWDAAPI (*fnMemFree)(LWdeviceptr dptr);
typedef LWresult LWDAAPI (*fnCtxGetLwrrent)(LWcontext* pctx);
typedef LWresult LWDAAPI (*fnCtxSetLwrrent)(LWcontext ctx);
typedef LWresult LWDAAPI (*fnCtxDestroy)(LWcontext ctx);

#endif /* _LWHOOK_H_ */
