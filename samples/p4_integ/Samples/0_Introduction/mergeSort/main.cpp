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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <lwda_runtime.h>
#include <helper_functions.h>
#include <helper_lwda.h>
#include "mergeSort_common.h"

////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  uint *h_SrcKey, *h_SrcVal, *h_DstKey, *h_DstVal;
  uint *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;
  StopWatchInterface *hTimer = NULL;

  const uint N = 4 * 1048576;
  const uint DIR = 1;
  const uint numValues = 65536;

  printf("%s Starting...\n\n", argv[0]);

  int dev = findLwdaDevice(argc, (const char **)argv);

  if (dev == -1) {
    return EXIT_FAILURE;
  }

  printf("Allocating and initializing host arrays...\n\n");
  sdkCreateTimer(&hTimer);
  h_SrcKey = (uint *)malloc(N * sizeof(uint));
  h_SrcVal = (uint *)malloc(N * sizeof(uint));
  h_DstKey = (uint *)malloc(N * sizeof(uint));
  h_DstVal = (uint *)malloc(N * sizeof(uint));

  srand(2009);

  for (uint i = 0; i < N; i++) {
    h_SrcKey[i] = rand() % numValues;
  }

  fillValues(h_SrcVal, N);

  printf("Allocating and initializing LWCA arrays...\n\n");
  checkLwdaErrors(lwdaMalloc((void **)&d_DstKey, N * sizeof(uint)));
  checkLwdaErrors(lwdaMalloc((void **)&d_DstVal, N * sizeof(uint)));
  checkLwdaErrors(lwdaMalloc((void **)&d_BufKey, N * sizeof(uint)));
  checkLwdaErrors(lwdaMalloc((void **)&d_BufVal, N * sizeof(uint)));
  checkLwdaErrors(lwdaMalloc((void **)&d_SrcKey, N * sizeof(uint)));
  checkLwdaErrors(lwdaMalloc((void **)&d_SrcVal, N * sizeof(uint)));
  checkLwdaErrors(
      lwdaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint), lwdaMemcpyHostToDevice));
  checkLwdaErrors(
      lwdaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint), lwdaMemcpyHostToDevice));

  printf("Initializing GPU merge sort...\n");
  initMergeSort();

  printf("Running GPU merge sort...\n");
  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  mergeSort(d_DstKey, d_DstVal, d_BufKey, d_BufVal, d_SrcKey, d_SrcVal, N, DIR);
  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  printf("Time: %f ms\n", sdkGetTimerValue(&hTimer));

  printf("Reading back GPU merge sort results...\n");
  checkLwdaErrors(
      lwdaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint), lwdaMemcpyDeviceToHost));
  checkLwdaErrors(
      lwdaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint), lwdaMemcpyDeviceToHost));

  printf("Inspecting the results...\n");
  uint keysFlag = validateSortedKeys(h_DstKey, h_SrcKey, 1, N, numValues, DIR);

  uint valuesFlag = validateSortedValues(h_DstKey, h_DstVal, h_SrcKey, 1, N);

  printf("Shutting down...\n");
  closeMergeSort();
  sdkDeleteTimer(&hTimer);
  checkLwdaErrors(lwdaFree(d_SrcVal));
  checkLwdaErrors(lwdaFree(d_SrcKey));
  checkLwdaErrors(lwdaFree(d_BufVal));
  checkLwdaErrors(lwdaFree(d_BufKey));
  checkLwdaErrors(lwdaFree(d_DstVal));
  checkLwdaErrors(lwdaFree(d_DstKey));
  free(h_DstVal);
  free(h_DstKey);
  free(h_SrcVal);
  free(h_SrcKey);

  exit((keysFlag && valuesFlag) ? EXIT_SUCCESS : EXIT_FAILURE);
}
