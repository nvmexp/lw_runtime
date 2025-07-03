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

/*
 * This sample implements the same algorithm as the colwolutionSeparable
 * LWCA Sample, but without using the shared memory at all.
 * Instead, it uses textures in exactly the same way an OpenGL-based
 * implementation would do.
 * Refer to the "Performance" section of colwolutionSeparable whitepaper.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <lwda_runtime.h>

#include <helper_functions.h>
#include <helper_lwda.h>

#include "colwolutionTexture_common.h"

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  float *h_Kernel, *h_Input, *h_Buffer, *h_OutputCPU, *h_OutputGPU;

  lwdaArray *a_Src;
  lwdaTextureObject_t texSrc;
  lwdaChannelFormatDesc floatTex = lwdaCreateChannelDesc<float>();

  float *d_Output;

  float gpuTime;

  StopWatchInterface *hTimer = NULL;

  const int imageW = 3072;
  const int imageH = 3072 / 2;
  const unsigned int iterations = 10;

  printf("[%s] - Starting...\n", argv[0]);

  // use command-line specified LWCA device, otherwise use device with highest
  // Gflops/s
  findLwdaDevice(argc, (const char **)argv);

  sdkCreateTimer(&hTimer);

  printf("Initializing data...\n");
  h_Kernel = (float *)malloc(KERNEL_LENGTH * sizeof(float));
  h_Input = (float *)malloc(imageW * imageH * sizeof(float));
  h_Buffer = (float *)malloc(imageW * imageH * sizeof(float));
  h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
  h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
  checkLwdaErrors(lwdaMallocArray(&a_Src, &floatTex, imageW, imageH));
  checkLwdaErrors(
      lwdaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));

  lwdaResourceDesc texRes;
  memset(&texRes, 0, sizeof(lwdaResourceDesc));

  texRes.resType = lwdaResourceTypeArray;
  texRes.res.array.array = a_Src;

  lwdaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(lwdaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = lwdaFilterModeLinear;
  texDescr.addressMode[0] = lwdaAddressModeWrap;
  texDescr.addressMode[1] = lwdaAddressModeWrap;
  texDescr.readMode = lwdaReadModeElementType;

  checkLwdaErrors(lwdaCreateTextureObject(&texSrc, &texRes, &texDescr, NULL));

  srand(2009);

  for (unsigned int i = 0; i < KERNEL_LENGTH; i++) {
    h_Kernel[i] = (float)(rand() % 16);
  }

  for (unsigned int i = 0; i < imageW * imageH; i++) {
    h_Input[i] = (float)(rand() % 16);
  }

  setColwolutionKernel(h_Kernel);
  checkLwdaErrors(lwdaMemcpyToArray(a_Src, 0, 0, h_Input,
                                    imageW * imageH * sizeof(float),
                                    lwdaMemcpyHostToDevice));

  printf("Running GPU rows colwolution (%u identical iterations)...\n",
         iterations);
  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  for (unsigned int i = 0; i < iterations; i++) {
    colwolutionRowsGPU(d_Output, a_Src, imageW, imageH, texSrc);
  }

  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
  printf("Average colwolutionRowsGPU() time: %f msecs; //%f Mpix/s\n", gpuTime,
         imageW * imageH * 1e-6 / (0.001 * gpuTime));

  // While LWCA kernels can't write to textures directly, this copy is
  // inevitable
  printf("Copying colwolutionRowGPU() output back to the texture...\n");
  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  checkLwdaErrors(lwdaMemcpyToArray(a_Src, 0, 0, d_Output,
                                    imageW * imageH * sizeof(float),
                                    lwdaMemcpyDeviceToDevice));
  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  gpuTime = sdkGetTimerValue(&hTimer);
  printf("lwdaMemcpyToArray() time: %f msecs; //%f Mpix/s\n", gpuTime,
         imageW * imageH * 1e-6 / (0.001 * gpuTime));

  printf("Running GPU columns colwolution (%i iterations)\n", iterations);
  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  for (int i = 0; i < iterations; i++) {
    colwolutionColumnsGPU(d_Output, a_Src, imageW, imageH, texSrc);
  }

  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
  printf("Average colwolutionColumnsGPU() time: %f msecs; //%f Mpix/s\n",
         gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));

  printf("Reading back GPU results...\n");
  checkLwdaErrors(lwdaMemcpy(h_OutputGPU, d_Output,
                             imageW * imageH * sizeof(float),
                             lwdaMemcpyDeviceToHost));

  printf("Checking the results...\n");
  printf("...running colwolutionRowsCPU()\n");
  colwolutionRowsCPU(h_Buffer, h_Input, h_Kernel, imageW, imageH,
                     KERNEL_RADIUS);

  printf("...running colwolutionColumnsCPU()\n");
  colwolutionColumnsCPU(h_OutputCPU, h_Buffer, h_Kernel, imageW, imageH,
                        KERNEL_RADIUS);

  double delta = 0;
  double sum = 0;

  for (unsigned int i = 0; i < imageW * imageH; i++) {
    sum += h_OutputCPU[i] * h_OutputCPU[i];
    delta +=
        (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
  }

  double L2norm = sqrt(delta / sum);
  printf("Relative L2 norm: %E\n", L2norm);
  printf("Shutting down...\n");

  checkLwdaErrors(lwdaFree(d_Output));
  checkLwdaErrors(lwdaFreeArray(a_Src));
  free(h_OutputGPU);
  free(h_Buffer);
  free(h_Input);
  free(h_Kernel);

  sdkDeleteTimer(&hTimer);

  if (L2norm > 1e-6) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
