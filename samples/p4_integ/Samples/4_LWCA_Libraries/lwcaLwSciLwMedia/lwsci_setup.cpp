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

#include "lwmedia_utils/cmdline.h"
#include <lwca.h>
#include <lwda_runtime.h>
#include "helper_lwda.h"
#include "lwsci_setup.h"
#include "lwmedia_2d_lwscisync.h"

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

void setupLwMediaSignalerLwSciSync(Blit2DTest *ctx, LwSciSyncObj &syncObj,
                                   int lwdaDeviceId) {
  LwSciSyncModule sciSyncModule;
  checkLwSciErrors(LwSciSyncModuleOpen(&sciSyncModule));
  LwSciSyncAttrList signalerAttrList, waiterAttrList;
  LwSciSyncAttrList synlwnreconciledList[2];
  LwSciSyncAttrList syncReconciledList, syncConflictList;

  checkLwSciErrors(LwSciSyncAttrListCreate(sciSyncModule, &signalerAttrList));
  checkLwSciErrors(LwSciSyncAttrListCreate(sciSyncModule, &waiterAttrList));

  LwMediaStatus status = LwMedia2DFillLwSciSyncAttrList(
      ctx->i2d, signalerAttrList, LWMEDIA_SIGNALER);
  if (status != LWMEDIA_STATUS_OK) {
    printf("%s: LwMedia2DFillLwSciSyncAttrList failed\n", __func__);
    exit(EXIT_FAILURE);
  }

  checkLwdaErrors(lwdaSetDevice(lwdaDeviceId));
  checkLwdaErrors(lwdaDeviceGetLwSciSyncAttributes(waiterAttrList, lwdaDeviceId,
                                                   lwdaLwSciSyncAttrWait));

  synlwnreconciledList[0] = signalerAttrList;
  synlwnreconciledList[1] = waiterAttrList;
  checkLwSciErrors(LwSciSyncAttrListReconcile(
      synlwnreconciledList, 2, &syncReconciledList, &syncConflictList));
  checkLwSciErrors(LwSciSyncObjAlloc(syncReconciledList, &syncObj));

  LwSciSyncAttrListFree(signalerAttrList);
  LwSciSyncAttrListFree(waiterAttrList);
  if (syncConflictList != nullptr) {
    LwSciSyncAttrListFree(syncConflictList);
  }
}

void setupLwdaSignalerLwSciSync(Blit2DTest *ctx, LwSciSyncObj &syncObj,
                                int lwdaDeviceId) {
  LwSciSyncModule sciSyncModule;
  checkLwSciErrors(LwSciSyncModuleOpen(&sciSyncModule));
  LwSciSyncAttrList signalerAttrList, waiterAttrList;
  LwSciSyncAttrList synlwnreconciledList[2];
  LwSciSyncAttrList syncReconciledList, syncConflictList;

  checkLwSciErrors(LwSciSyncAttrListCreate(sciSyncModule, &signalerAttrList));
  checkLwSciErrors(LwSciSyncAttrListCreate(sciSyncModule, &waiterAttrList));

  LwMediaStatus status =
      LwMedia2DFillLwSciSyncAttrList(ctx->i2d, waiterAttrList, LWMEDIA_WAITER);
  if (status != LWMEDIA_STATUS_OK) {
    printf("%s: LwMedia2DFillLwSciSyncAttrList failed\n", __func__);
    exit(EXIT_FAILURE);
  }

  checkLwdaErrors(lwdaSetDevice(lwdaDeviceId));
  checkLwdaErrors(lwdaDeviceGetLwSciSyncAttributes(
      signalerAttrList, lwdaDeviceId, lwdaLwSciSyncAttrSignal));

  synlwnreconciledList[0] = signalerAttrList;
  synlwnreconciledList[1] = waiterAttrList;
  checkLwSciErrors(LwSciSyncAttrListReconcile(
      synlwnreconciledList, 2, &syncReconciledList, &syncConflictList));
  checkLwSciErrors(LwSciSyncObjAlloc(syncReconciledList, &syncObj));

  LwSciSyncAttrListFree(signalerAttrList);
  LwSciSyncAttrListFree(waiterAttrList);
  if (syncConflictList != nullptr) {
    LwSciSyncAttrListFree(syncConflictList);
  }
}

void setupLwSciBuf(LwSciBufObj &bufobj, LwSciBufAttrList &lwmediaAttrlist,
                   int lwdaDeviceId) {
  LWuuid devUUID;
  LwSciBufAttrList conflictlist;
  LwSciBufAttrList bufUnreconciledAttrlist[1];

  LWresult res = lwDeviceGetUuid(&devUUID, lwdaDeviceId);
  if (res != LWDA_SUCCESS) {
    fprintf(stderr, "Driver API error = %04d \n", res);
    exit(EXIT_FAILURE);
  }

  LwSciBufAttrKeyValuePair attr_gpuid[] = {LwSciBufGeneralAttrKey_GpuId,
                                           &devUUID, sizeof(devUUID)};

  // set LWCA GPU ID to attribute list
  checkLwSciErrors(LwSciBufAttrListSetAttrs(
      lwmediaAttrlist, attr_gpuid,
      sizeof(attr_gpuid) / sizeof(LwSciBufAttrKeyValuePair)));

  bufUnreconciledAttrlist[0] = lwmediaAttrlist;

  checkLwSciErrors(LwSciBufAttrListReconcileAndObjAlloc(
      bufUnreconciledAttrlist, 1, &bufobj, &conflictlist));
  if (conflictlist != NULL) {
    LwSciBufAttrListFree(conflictlist);
  }
}

void cleanupLwSciBuf(LwSciBufObj &Bufobj) {
  if (Bufobj != NULL) {
    LwSciBufObjFree(Bufobj);
  }
}

void cleanupLwSciSync(LwSciSyncObj &syncObj) {
  if (LwSciSyncObjFree != NULL) {
    LwSciSyncObjFree(syncObj);
  }
}
