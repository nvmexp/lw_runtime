/******************************************************************************
 * Copyright (c) 2016, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the LWPU CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(__LWDACC__) || defined(__LWCOMPILER_LWDA__)
#  if !defined(__LWDA_ARCH__) || (__LWDA_ARCH__>= 350 && defined(__LWDACC_RDC__))
#    define __THRUST_HAS_LWDART__ 1
#    define THRUST_RUNTIME_FUNCTION __host__ __device__ __forceinline__
#  else
#    define __THRUST_HAS_LWDART__ 0
#    define THRUST_RUNTIME_FUNCTION __host__ __forceinline__
#  endif
#else
#  define __THRUST_HAS_LWDART__ 0
#  define THRUST_RUNTIME_FUNCTION __host__ __forceinline__
#endif

#ifdef __LWDA_ARCH__
#define THRUST_DEVICE_CODE
#endif

#ifdef THRUST_AGENT_ENTRY_NOINLINE
#define THRUST_AGENT_ENTRY_INLINE_ATTR __noinline__
#else
#define THRUST_AGENT_ENTRY_INLINE_ATTR __forceinline__
#endif

#define THRUST_DEVICE_FUNCTION __device__ __forceinline__
#define THRUST_HOST_FUNCTION __host__     __forceinline__
#define THRUST_FUNCTION __host__ __device__ __forceinline__
#if 0
#define THRUST_ARGS(...) __VA_ARGS__
#define THRUST_STRIP_PARENS(X) X
#define THRUST_AGENT_ENTRY(ARGS) THRUST_FUNCTION static void entry(THRUST_STRIP_PARENS(THRUST_ARGS ARGS))
#else
#define THRUST_AGENT_ENTRY(...) THRUST_AGENT_ENTRY_INLINE_ATTR __device__ static void entry(__VA_ARGS__)
#endif

#ifdef THRUST_DEBUG_SYNC
#define THRUST_DEBUG_SYNC_FLAG true
#else
#define THRUST_DEBUG_SYNC_FLAG false
#endif

#define THRUST_LWB_NS_PREFIX namespace thrust {   namespace lwda_lwb {
#define THRUST_LWB_NS_POSTFIX }  }

#ifndef THRUST_IGNORE_LWB_VERSION_CHECK
#include <thrust/version.h>
#include <lwb/util_namespace.lwh> // This includes <lwb/version.lwh> in newer releases.
#if THRUST_VERSION != LWB_VERSION
#error The version of LWB in your include path is not compatible with this release of Thrust. LWB is now included in the LWCA Toolkit, so you no longer need to use your own checkout of LWB. Define THRUST_IGNORE_LWB_VERSION_CHECK to ignore this.
#endif
#endif
