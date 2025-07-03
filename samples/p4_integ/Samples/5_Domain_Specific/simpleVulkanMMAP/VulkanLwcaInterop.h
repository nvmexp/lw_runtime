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

#pragma once
#ifndef __VKLWDA_H__
#define __VKLWDA_H__

#include <lwda_runtime_api.h>
#include "lwca.h"
#define LWDA_DRIVER_API
#include <helper_lwda.h>

bool isDeviceCompatible(void *Uuid, size_t size) {
  int lwdaDevice = lwdaIlwalidDeviceId;
  int deviceCount;
  checkLwdaErrors(lwdaGetDeviceCount(&deviceCount));

  for (int i = 0; i < deviceCount; ++i) {
    lwdaDeviceProp devProp = {};
    checkLwdaErrors(lwdaGetDeviceProperties(&devProp, i));
    if (!memcmp(&devProp.uuid, Uuid, size)) {
      lwdaDevice = i;
      break;
    }
  }
  if (lwdaDevice == lwdaIlwalidDeviceId) {
    return false;
  }

  int deviceSupportsHandle = 0;
  int attributeVal = 0;
  int deviceComputeMode = 0;

  checkLwdaErrors(lwDeviceGetAttribute(
      &deviceComputeMode, LW_DEVICE_ATTRIBUTE_COMPUTE_MODE, lwdaDevice));
  checkLwdaErrors(lwDeviceGetAttribute(
      &attributeVal, LW_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
      lwdaDevice));

#if defined(__linux__)
  checkLwdaErrors(lwDeviceGetAttribute(
      &deviceSupportsHandle,
      LW_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
      lwdaDevice));
#else
  checkLwdaErrors(lwDeviceGetAttribute(
      &deviceSupportsHandle,
      LW_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, lwdaDevice));
#endif

  if ((deviceComputeMode != LW_COMPUTEMODE_DEFAULT) || !attributeVal ||
      !deviceSupportsHandle) {
    return false;
  }
  return true;
}

#endif  // __VKLWDA_H__
