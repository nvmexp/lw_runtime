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

// LWCA Runtime
#include <lwda_runtime.h>

// Helper functions
#include <helper_lwda.h>
#include <helper_math.h>
#include "volume.h"

void Volume_init(Volume *vol, lwdaExtent dataSize, void *h_data,
                 int allowStore) {
  // create 3D array
  vol->channelDesc = lwdaCreateChannelDesc<VolumeType>();
  checkLwdaErrors(
      lwdaMalloc3DArray(&vol->content, &vol->channelDesc, dataSize,
                        allowStore ? lwdaArraySurfaceLoadStore : 0));
  vol->size = dataSize;

  if (h_data) {
    // copy data to 3D array
    lwdaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_lwdaPitchedPtr(h_data, dataSize.width * sizeof(VolumeType),
                            dataSize.width, dataSize.height);
    copyParams.dstArray = vol->content;
    copyParams.extent = dataSize;
    copyParams.kind = lwdaMemcpyHostToDevice;
    checkLwdaErrors(lwdaMemcpy3D(&copyParams));
  }

  if (allowStore) {
    lwdaResourceDesc surfRes;
    memset(&surfRes, 0, sizeof(lwdaResourceDesc));
    surfRes.resType = lwdaResourceTypeArray;
    surfRes.res.array.array = vol->content;

    checkLwdaErrors(lwdaCreateSurfaceObject(&vol->volumeSurf, &surfRes));
  }

  lwdaResourceDesc texRes;
  memset(&texRes, 0, sizeof(lwdaResourceDesc));

  texRes.resType = lwdaResourceTypeArray;
  texRes.res.array.array = vol->content;

  lwdaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(lwdaTextureDesc));

  texDescr.normalizedCoords = true;
  texDescr.filterMode = lwdaFilterModeLinear;
  texDescr.addressMode[0] = lwdaAddressModeWrap;
  texDescr.addressMode[1] = lwdaAddressModeWrap;
  texDescr.addressMode[2] = lwdaAddressModeWrap;
  texDescr.readMode =
      lwdaReadModeNormalizedFloat;  // VolumeTypeInfo<VolumeType>::readMode;

  checkLwdaErrors(
      lwdaCreateTextureObject(&vol->volumeTex, &texRes, &texDescr, NULL));
}

void Volume_deinit(Volume *vol) {
  checkLwdaErrors(lwdaDestroyTextureObject(vol->volumeTex));
  checkLwdaErrors(lwdaDestroySurfaceObject(vol->volumeSurf));
  checkLwdaErrors(lwdaFreeArray(vol->content));
  vol->content = 0;
}
