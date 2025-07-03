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

#ifndef __LWDA_BUFIMPORT_KERNEL_H__
#define __LWDA_BUFIMPORT_KERNEL_H__

#include <lwda_runtime.h>
#include "helper_lwda.h"
#include "lwmedia_image_lwscibuf.h"
#include "lwscisync.h"
#include "lwmedia_utils/cmdline.h"

struct lwdaExternalResInterop {
  lwdaMipmappedArray_t *d_mipmapArray;
  lwdaArray_t *d_mipLevelArray;
  lwdaSurfaceObject_t *lwdaSurfaceLwmediaBuf;
  lwdaStream_t stream;
  lwdaExternalMemory_t extMemImageBuf;
  lwdaExternalSemaphore_t waitSem;
  lwdaExternalSemaphore_t signalSem;

  int32_t planeCount;
  uint64_t *planeOffset;
  int32_t *imageWidth;
  int32_t *imageHeight;
  unsigned int *d_outputImage;
};

struct lwdaResources {
  lwdaArray_t *d_yuvArray;
  lwdaStream_t stream;
  lwdaSurfaceObject_t *lwdaSurfaceLwmediaBuf;
  unsigned int *d_outputImage;
};

void runLwdaOperation(lwdaExternalResInterop &lwdaExtResObj,
                      LwSciSyncFence *fence, LwSciSyncFence *lwdaSignalfence,
                      int deviceId, int iterations);
void runLwdaOperation(Blit2DTest *ctx, lwdaResources &lwdaResObj, int deviceId);

void setupLwda(lwdaExternalResInterop &lwdaExtResObj, LwSciBufObj &inputBufObj,
               LwSciSyncObj &syncObj, LwSciSyncObj &lwdaSignalerSyncObj,
               int deviceId);
void setupLwda(Blit2DTest *ctx, lwdaResources &lwdaResObj, int deviceId);
void cleanupLwda(lwdaExternalResInterop &lwdaObjs);
void cleanupLwda(Blit2DTest *ctx, lwdaResources &lwdaResObj);

#endif
