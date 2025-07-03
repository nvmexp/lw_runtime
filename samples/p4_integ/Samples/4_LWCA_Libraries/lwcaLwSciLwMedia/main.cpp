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

/* standard headers */
#include <string.h>
#include <iostream>
#include <signal.h>
#include <thread>

/* Lwpu headers */
#include <lwscisync.h>
#include "lwmedia_utils/cmdline.h"
#include "lwmedia_image.h"
#include "lwmedia_2d.h"
#include "lwmedia_2d_lwscisync.h"
#include "lwmedia_surface.h"
#include "lwmedia_utils/image_utils.h"
#include "lwmedia_image_lwscibuf.h"
#include "lwda_consumer.h"
#include "lwmedia_producer.h"
#include "lwsci_setup.h"

#define checkLwSciErrors(call)                              \
  do {                                                      \
    LwSciError _status = call;                              \
    if (LwSciError_Success != _status) {                    \
      printf(                                               \
          "LWSCI call in file '%s' in line %i returned"     \
          " %d, expected %d\n",                             \
          __FILE__, __LINE__, _status, LwSciError_Success); \
      fflush(stdout);                                       \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  } while (0)

static void cleanup(Blit2DTest* ctx, LwMediaStatus status) {
  if (ctx->i2d != NULL) {
    LwMedia2DDestroy(ctx->i2d);
  }

  if (ctx->device != NULL) {
    LwMediaDeviceDestroy(ctx->device);
  }
  if (status != LWMEDIA_STATUS_OK) {
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[]) {
  TestArgs args;
  Blit2DTest ctx;
  LwMediaStatus status = LWMEDIA_STATUS_ERROR;
  LwSciSyncFence lwMediaSignalerFence = LwSciSyncFenceInitializer;
  LwSciSyncFence lwdaSignalerFence = LwSciSyncFenceInitializer;

  int lwdaDeviceId;
  uint64_t startTime, endTime;
  uint64_t operationStartTime, operationEndTime;
  double processingTime;

  /* Read configuration from command line and config file */
  memset(&args, 0, sizeof(TestArgs));
  memset(&ctx, 0, sizeof(Blit2DTest));

  /* ParseArgs parses the command line and the 2D configuration file and
   * populates all initParams and run time configuration in to appropriate
   * structures within args
   */
  if (ParseArgs(argc, argv, &args)) {
    PrintUsage();
    return -1;
  }
  /* Check version */
  LwMediaVersion version;
  status = LwMedia2DGetVersion(&version);
  if (status == LWMEDIA_STATUS_OK) {
    printf("Library version: %u.%u\n", version.major, version.minor);
    printf("Header version:  %u.%u\n", LWMEDIA_2D_VERSION_MAJOR,
           LWMEDIA_2D_VERSION_MINOR);
    if ((version.major != LWMEDIA_2D_VERSION_MAJOR) ||
        (version.minor != LWMEDIA_2D_VERSION_MINOR)) {
      printf("Library and Header mismatch!\n");
      cleanup(&ctx, status);
    }
  }

  // Create LwMedia device
  ctx.device = LwMediaDeviceCreate();
  if (!ctx.device) {
    printf("%s: Failed to create LwMedia device\n", __func__);
    cleanup(&ctx, status);
  }

  // Create 2D blitter
  ctx.i2d = LwMedia2DCreate(ctx.device);
  if (!ctx.i2d) {
    printf("%s: Failed to create LwMedia 2D i2d\n", __func__);
    cleanup(&ctx, status);
  }

  lwdaDeviceId = findLwdaDevice(argc, (const char**)argv);

  // LwMedia-LWCA operations without LwSCI APIs starts
  lwdaResources lwdaResObj;
  GetTimeMicroSec(&startTime);
  setupLwMedia(&args, &ctx);
  setupLwda(&ctx, lwdaResObj, lwdaDeviceId);

  GetTimeMicroSec(&operationStartTime);
  for (int i = 0; i < args.iterations; i++) {
    runLwMediaBlit2D(&args, &ctx);
    runLwdaOperation(&ctx, lwdaResObj, lwdaDeviceId);
  }
  GetTimeMicroSec(&operationEndTime);

  cleanupLwMedia(&ctx);
  cleanupLwda(&ctx, lwdaResObj);
  GetTimeMicroSec(&endTime);
  // LwMedia-LWCA operations without LwSCI APIs ends

  processingTime = (double)(operationEndTime - operationStartTime) / 1000.0;
  printf(
      "Overall Processing time of LwMedia-LWCA Operations without LwSCI APIs "
      "%.4f ms  with %zu iterations\n",
      processingTime, args.iterations);
  processingTime = (double)(endTime - startTime) / 1000.0;
  printf(
      "Overall Processing time of LwMedia-LWCA Operations + allocation/cleanup "
      "without LwSCI APIs %.4f ms  with %zu iterations\n",
      processingTime, args.iterations);

  LwSciBufObj dstLwSciBufobj, srcLwSciBufobj;
  LwSciSyncObj lwMediaSignalerSyncObj, lwdaSignalerSyncObj;
  lwdaExternalResInterop lwdaExtResObj;
  // LwMedia-LWCA operations via interop with LwSCI APIs starts
  GetTimeMicroSec(&startTime);
  setupLwMediaSignalerLwSciSync(&ctx, lwMediaSignalerSyncObj, lwdaDeviceId);
  setupLwdaSignalerLwSciSync(&ctx, lwdaSignalerSyncObj, lwdaDeviceId);
  setupLwMedia(&args, &ctx, srcLwSciBufobj, dstLwSciBufobj,
               lwMediaSignalerSyncObj, lwdaSignalerSyncObj, lwdaDeviceId);
  setupLwda(lwdaExtResObj, dstLwSciBufobj, lwMediaSignalerSyncObj,
            lwdaSignalerSyncObj, lwdaDeviceId);

  GetTimeMicroSec(&operationStartTime);
  for (int i = 0; i < args.iterations; i++) {
    runLwMediaBlit2D(&args, &ctx, lwMediaSignalerSyncObj, &lwdaSignalerFence,
                     &lwMediaSignalerFence);
    runLwdaOperation(lwdaExtResObj, &lwMediaSignalerFence, &lwdaSignalerFence,
                     lwdaDeviceId, args.iterations);
  }
  GetTimeMicroSec(&operationEndTime);

  cleanupLwMedia(&ctx, lwMediaSignalerSyncObj, lwdaSignalerSyncObj);
  cleanupLwda(lwdaExtResObj);
  cleanupLwSciSync(lwMediaSignalerSyncObj);
  cleanupLwSciSync(lwdaSignalerSyncObj);
  cleanupLwSciBuf(srcLwSciBufobj);
  cleanupLwSciBuf(dstLwSciBufobj);
  GetTimeMicroSec(&endTime);
  // LwMedia-LWCA operations via interop with LwSCI APIs ends

  processingTime = (double)(operationEndTime - operationStartTime) / 1000.0;
  printf(
      "Overall Processing time of LwMedia-LWCA Operations with LwSCI APIs %.4f "
      "ms  with %zu iterations\n",
      processingTime, args.iterations);
  processingTime = (double)(endTime - startTime) / 1000.0;
  printf(
      "Overall Processing time of LwMedia-LWCA Operations + allocation/cleanup "
      "with LwSCI APIs %.4f ms  with %zu iterations\n",
      processingTime, args.iterations);

  if (ctx.i2d != NULL) {
    LwMedia2DDestroy(ctx.i2d);
  }

  if (ctx.device != NULL) {
    LwMediaDeviceDestroy(ctx.device);
  }

  if (status == LWMEDIA_STATUS_OK) {
    return 0;
  } else {
    return 1;
  }
}
