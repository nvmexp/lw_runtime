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

// This sample demonstrates a simple library to interpose LWCA symbols

#define __USE_GNU
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#include <lwca.h>
#include "liblwhook.h"

// For interposing dlsym(). See elf/dl-libc.c for the internal dlsym interface
// function For interposing dlopen(). Sell elf/dl-lib.c for the internal
// dlopen_mode interface function
extern "C" {
void* __libc_dlsym(void* map, const char* name);
}
extern "C" {
void* __libc_dlopen_mode(const char* name, int mode);
}

// We need to give the pre-processor a chance to replace a function, such as:
// lwMemAlloc => lwMemAlloc_v2
#define STRINGIFY(x) #x
#define LWDA_SYMBOL_STRING(x) STRINGIFY(x)

// We need to interpose dlsym since anyone using dlopen+dlsym to get the LWCA
// driver symbols will bypass the hooking mechanism (this includes the LWCA
// runtime). Its tricky though, since if we replace the real dlsym with ours, we
// can't dlsym() the real dlsym. To get around that, call the 'private' libc
// interface called __libc_dlsym to get the real dlsym.
typedef void* (*fnDlsym)(void*, const char*);

static void* real_dlsym(void* handle, const char* symbol) {
  static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(
      __libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
  return (*internal_dlsym)(handle, symbol);
}

// Main structure that gets initialized at library load time
// Choose a unique name, or it can clash with other preloaded libraries.
struct lwHookInfo {
  void* handle;
  void* preHooks[LW_HOOK_SYMBOLS];
  void* postHooks[LW_HOOK_SYMBOLS];

  // Debugging/Stats Info
  int bDebugEnabled;
  int hookedFunctionCalls[LW_HOOK_SYMBOLS];

  lwHookInfo() {
    const char* elwHookDebug;

    // Check environment for LW_HOOK_DEBUG to facilitate debugging
    elwHookDebug = getelw("LW_HOOK_DEBUG");
    if (elwHookDebug && elwHookDebug[0] == '1') {
      bDebugEnabled = 1;
      fprintf(stderr, "* %6d >> LWCA HOOK Library loaded.\n", getpid());
    }
  }

  ~lwHookInfo() {
    if (bDebugEnabled) {
      pid_t pid = getpid();
      // You can gather statistics, timings, etc.
      fprintf(stderr, "* %6d >> LWCA HOOK Library Unloaded - Statistics:\n",
              pid);
      fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
              LWDA_SYMBOL_STRING(lwMemAlloc),
              hookedFunctionCalls[LW_HOOK_MEM_ALLOC]);
      fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
              LWDA_SYMBOL_STRING(lwMemFree),
              hookedFunctionCalls[LW_HOOK_MEM_FREE]);
      fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
              LWDA_SYMBOL_STRING(lwCtxGetLwrrent),
              hookedFunctionCalls[LW_HOOK_CTX_GET_LWRRENT]);
      fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
              LWDA_SYMBOL_STRING(lwCtxSetLwrrent),
              hookedFunctionCalls[LW_HOOK_CTX_SET_LWRRENT]);
      fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
              LWDA_SYMBOL_STRING(lwCtxDestroy),
              hookedFunctionCalls[LW_HOOK_CTX_DESTROY]);
    }
    if (handle) {
      dlclose(handle);
    }
  }
};

static struct lwHookInfo lwhl;

// Exposed API
void lwHookRegisterCallback(HookSymbols symbol, HookTypes type,
                            void* callback) {
  if (type == PRE_CALL_HOOK) {
    lwhl.preHooks[symbol] = callback;
  } else if (type == POST_CALL_HOOK) {
    lwhl.postHooks[symbol] = callback;
  }
}

/*
 ** Interposed Functions
 */
void* dlsym(void* handle, const char* symbol) {
  // Early out if not a LWCA driver symbol
  if (strncmp(symbol, "lw", 2) != 0) {
    return (real_dlsym(handle, symbol));
  }

  if (strcmp(symbol, LWDA_SYMBOL_STRING(lwMemAlloc)) == 0) {
    return (void*)(&lwMemAlloc);
  } else if (strcmp(symbol, LWDA_SYMBOL_STRING(lwMemFree)) == 0) {
    return (void*)(&lwMemFree);
  } else if (strcmp(symbol, LWDA_SYMBOL_STRING(lwCtxGetLwrrent)) == 0) {
    return (void*)(&lwCtxGetLwrrent);
  } else if (strcmp(symbol, LWDA_SYMBOL_STRING(lwCtxSetLwrrent)) == 0) {
    return (void*)(&lwCtxSetLwrrent);
  } else if (strcmp(symbol, LWDA_SYMBOL_STRING(lwCtxDestroy)) == 0) {
    return (void*)(&lwCtxDestroy);
  }
  return (real_dlsym(handle, symbol));
}

/*
** If the user of this library does not wish to include LWCA specific
*code/headers in the code,
** then all the parameters can be changed and/or simply casted before calling
*the callback.
*/
#define LW_HOOK_GENERATE_INTERCEPT(hooksymbol, funcname, params, ...)        \
  LWresult LWDAAPI funcname params {                                         \
    static void* real_func =                                                 \
        (void*)real_dlsym(RTLD_NEXT, LWDA_SYMBOL_STRING(funcname));          \
    LWresult result = LWDA_SUCCESS;                                          \
                                                                             \
    if (lwhl.bDebugEnabled) {                                                \
      lwhl.hookedFunctionCalls[hooksymbol]++;                                \
    }                                                                        \
    if (lwhl.preHooks[hooksymbol]) {                                         \
      ((LWresult LWDAAPI(*) params)lwhl.preHooks[hooksymbol])(__VA_ARGS__);  \
    }                                                                        \
    result = ((LWresult LWDAAPI(*) params)real_func)(__VA_ARGS__);           \
    if (lwhl.postHooks[hooksymbol] && result == LWDA_SUCCESS) {              \
      ((LWresult LWDAAPI(*) params)lwhl.postHooks[hooksymbol])(__VA_ARGS__); \
    }                                                                        \
    return (result);                                                         \
  }

LW_HOOK_GENERATE_INTERCEPT(LW_HOOK_MEM_ALLOC, lwMemAlloc,
                           (LWdeviceptr * dptr, size_t bytesize), dptr,
                           bytesize)
LW_HOOK_GENERATE_INTERCEPT(LW_HOOK_MEM_FREE, lwMemFree, (LWdeviceptr dptr),
                           dptr)
LW_HOOK_GENERATE_INTERCEPT(LW_HOOK_CTX_GET_LWRRENT, lwCtxGetLwrrent,
                           (LWcontext * pctx), pctx)
LW_HOOK_GENERATE_INTERCEPT(LW_HOOK_CTX_SET_LWRRENT, lwCtxSetLwrrent,
                           (LWcontext ctx), ctx)
LW_HOOK_GENERATE_INTERCEPT(LW_HOOK_CTX_DESTROY, lwCtxDestroy, (LWcontext ctx),
                           ctx)
