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
 * This sample demonstrates how 2D colwolutions
 * with very large kernel sizes
 * can be efficiently implemented
 * using FFT transformations.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include LWCA runtime and LWFFT
#include <lwda_runtime.h>
#include <lwfft.h>

// Helper functions for LWCA
#include <helper_functions.h>
#include <helper_lwda.h>

#include "colwolutionFFT2D_common.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
int snapTransformSize(int dataSize) {
  int hiBit;
  unsigned int lowPOT, hiPOT;

  dataSize = iAlignUp(dataSize, 16);

  for (hiBit = 31; hiBit >= 0; hiBit--)
    if (dataSize & (1U << hiBit)) {
      break;
    }

  lowPOT = 1U << hiBit;

  if (lowPOT == (unsigned int)dataSize) {
    return dataSize;
  }

  hiPOT = 1U << (hiBit + 1);

  if (hiPOT <= 1024) {
    return hiPOT;
  } else {
    return iAlignUp(dataSize, 512);
  }
}

float getRand(void) { return (float)(rand() % 16); }

bool test0(void) {
  float *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;

  float *d_Data, *d_PaddedData, *d_Kernel, *d_PaddedKernel;

  fComplex *d_DataSpectrum, *d_KernelSpectrum;

  lwfftHandle fftPlanFwd, fftPlanIlw;

  bool bRetVal;
  StopWatchInterface *hTimer = NULL;
  sdkCreateTimer(&hTimer);

  printf("Testing built-in R2C / C2R FFT-based colwolution\n");
  const int kernelH = 7;
  const int kernelW = 6;
  const int kernelY = 3;
  const int kernelX = 4;
  const int dataH = 2000;
  const int dataW = 2000;
  const int fftH = snapTransformSize(dataH + kernelH - 1);
  const int fftW = snapTransformSize(dataW + kernelW - 1);

  printf("...allocating memory\n");
  h_Data = (float *)malloc(dataH * dataW * sizeof(float));
  h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
  h_ResultCPU = (float *)malloc(dataH * dataW * sizeof(float));
  h_ResultGPU = (float *)malloc(fftH * fftW * sizeof(float));

  checkLwdaErrors(lwdaMalloc((void **)&d_Data, dataH * dataW * sizeof(float)));
  checkLwdaErrors(
      lwdaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

  checkLwdaErrors(
      lwdaMalloc((void **)&d_PaddedData, fftH * fftW * sizeof(float)));
  checkLwdaErrors(
      lwdaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

  checkLwdaErrors(lwdaMalloc((void **)&d_DataSpectrum,
                             fftH * (fftW / 2 + 1) * sizeof(fComplex)));
  checkLwdaErrors(lwdaMalloc((void **)&d_KernelSpectrum,
                             fftH * (fftW / 2 + 1) * sizeof(fComplex)));
  checkLwdaErrors(lwdaMemset(d_KernelSpectrum, 0,
                             fftH * (fftW / 2 + 1) * sizeof(fComplex)));

  printf("...generating random input data\n");
  srand(2010);

  for (int i = 0; i < dataH * dataW; i++) {
    h_Data[i] = getRand();
  }

  for (int i = 0; i < kernelH * kernelW; i++) {
    h_Kernel[i] = getRand();
  }

  printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
  checkLwdaErrors(lwfftPlan2d(&fftPlanFwd, fftH, fftW, LWFFT_R2C));
  checkLwdaErrors(lwfftPlan2d(&fftPlanIlw, fftH, fftW, LWFFT_C2R));

  printf("...uploading to GPU and padding colwolution kernel and input data\n");
  checkLwdaErrors(lwdaMemcpy(d_Kernel, h_Kernel,
                             kernelH * kernelW * sizeof(float),
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemcpy(d_Data, h_Data, dataH * dataW * sizeof(float),
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
  checkLwdaErrors(lwdaMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));

  padKernel(d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW, kernelY,
            kernelX);

  padDataClampToBorder(d_PaddedData, d_Data, fftH, fftW, dataH, dataW, kernelH,
                       kernelW, kernelY, kernelX);

  // Not including kernel transformation into time measurement,
  // since colwolution kernel is not changed very frequently
  printf("...transforming colwolution kernel\n");
  checkLwdaErrors(lwfftExecR2C(fftPlanFwd, (lwfftReal *)d_PaddedKernel,
                               (lwfftComplex *)d_KernelSpectrum));

  printf("...running GPU FFT colwolution: ");
  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  checkLwdaErrors(lwfftExecR2C(fftPlanFwd, (lwfftReal *)d_PaddedData,
                               (lwfftComplex *)d_DataSpectrum));
  modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
  checkLwdaErrors(lwfftExecC2R(fftPlanIlw, (lwfftComplex *)d_DataSpectrum,
                               (lwfftReal *)d_PaddedData));

  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  double gpuTime = sdkGetTimerValue(&hTimer);
  printf("%f MPix/s (%f ms)\n",
         (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

  printf("...reading back GPU colwolution results\n");
  checkLwdaErrors(lwdaMemcpy(h_ResultGPU, d_PaddedData,
                             fftH * fftW * sizeof(float),
                             lwdaMemcpyDeviceToHost));

  printf("...running reference CPU colwolution\n");
  colwolutionClampToBorderCPU(h_ResultCPU, h_Data, h_Kernel, dataH, dataW,
                              kernelH, kernelW, kernelY, kernelX);

  printf("...comparing the results: ");
  double sum_delta2 = 0;
  double sum_ref2 = 0;
  double max_delta_ref = 0;

  for (int y = 0; y < dataH; y++)
    for (int x = 0; x < dataW; x++) {
      double rCPU = (double)h_ResultCPU[y * dataW + x];
      double rGPU = (double)h_ResultGPU[y * fftW + x];
      double delta = (rCPU - rGPU) * (rCPU - rGPU);
      double ref = rCPU * rCPU + rCPU * rCPU;

      if ((delta / ref) > max_delta_ref) {
        max_delta_ref = delta / ref;
      }

      sum_delta2 += delta;
      sum_ref2 += ref;
    }

  double L2norm = sqrt(sum_delta2 / sum_ref2);
  printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
  bRetVal = (L2norm < 1e-6) ? true : false;
  printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");

  printf("...shutting down\n");
  sdkDeleteTimer(&hTimer);

  checkLwdaErrors(lwfftDestroy(fftPlanIlw));
  checkLwdaErrors(lwfftDestroy(fftPlanFwd));

  checkLwdaErrors(lwdaFree(d_DataSpectrum));
  checkLwdaErrors(lwdaFree(d_KernelSpectrum));
  checkLwdaErrors(lwdaFree(d_PaddedData));
  checkLwdaErrors(lwdaFree(d_PaddedKernel));
  checkLwdaErrors(lwdaFree(d_Data));
  checkLwdaErrors(lwdaFree(d_Kernel));

  free(h_ResultGPU);
  free(h_ResultCPU);
  free(h_Data);
  free(h_Kernel);

  return bRetVal;
}

bool test1(void) {
  float *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;

  float *d_Data, *d_Kernel, *d_PaddedData, *d_PaddedKernel;

  fComplex *d_DataSpectrum0, *d_KernelSpectrum0, *d_DataSpectrum,
      *d_KernelSpectrum;

  lwfftHandle fftPlan;

  bool bRetVal;
  StopWatchInterface *hTimer = NULL;
  sdkCreateTimer(&hTimer);

  printf("Testing custom R2C / C2R FFT-based colwolution\n");
  const uint fftPadding = 16;
  const int kernelH = 7;
  const int kernelW = 6;
  const int kernelY = 3;
  const int kernelX = 4;
  const int dataH = 2000;
  const int dataW = 2000;
  const int fftH = snapTransformSize(dataH + kernelH - 1);
  const int fftW = snapTransformSize(dataW + kernelW - 1);

  printf("...allocating memory\n");
  h_Data = (float *)malloc(dataH * dataW * sizeof(float));
  h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
  h_ResultCPU = (float *)malloc(dataH * dataW * sizeof(float));
  h_ResultGPU = (float *)malloc(fftH * fftW * sizeof(float));

  checkLwdaErrors(lwdaMalloc((void **)&d_Data, dataH * dataW * sizeof(float)));
  checkLwdaErrors(
      lwdaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

  checkLwdaErrors(
      lwdaMalloc((void **)&d_PaddedData, fftH * fftW * sizeof(float)));
  checkLwdaErrors(
      lwdaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

  checkLwdaErrors(lwdaMalloc((void **)&d_DataSpectrum0,
                             fftH * (fftW / 2) * sizeof(fComplex)));
  checkLwdaErrors(lwdaMalloc((void **)&d_KernelSpectrum0,
                             fftH * (fftW / 2) * sizeof(fComplex)));
  checkLwdaErrors(
      lwdaMalloc((void **)&d_DataSpectrum,
                 fftH * (fftW / 2 + fftPadding) * sizeof(fComplex)));
  checkLwdaErrors(
      lwdaMalloc((void **)&d_KernelSpectrum,
                 fftH * (fftW / 2 + fftPadding) * sizeof(fComplex)));

  printf("...generating random input data\n");
  srand(2010);

  for (int i = 0; i < dataH * dataW; i++) {
    h_Data[i] = getRand();
  }

  for (int i = 0; i < kernelH * kernelW; i++) {
    h_Kernel[i] = getRand();
  }

  printf("...creating C2C FFT plan for %i x %i\n", fftH, fftW / 2);
  checkLwdaErrors(lwfftPlan2d(&fftPlan, fftH, fftW / 2, LWFFT_C2C));

  printf("...uploading to GPU and padding colwolution kernel and input data\n");
  checkLwdaErrors(lwdaMemcpy(d_Data, h_Data, dataH * dataW * sizeof(float),
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemcpy(d_Kernel, h_Kernel,
                             kernelH * kernelW * sizeof(float),
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));
  checkLwdaErrors(lwdaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));

  padDataClampToBorder(d_PaddedData, d_Data, fftH, fftW, dataH, dataW, kernelH,
                       kernelW, kernelY, kernelX);

  padKernel(d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW, kernelY,
            kernelX);

  // LWFFT_ILWERSE works just as well...
  const int FFT_DIR = LWFFT_FORWARD;

  // Not including kernel transformation into time measurement,
  // since colwolution kernel is not changed very frequently
  printf("...transforming colwolution kernel\n");
  checkLwdaErrors(lwfftExecC2C(fftPlan, (lwfftComplex *)d_PaddedKernel,
                               (lwfftComplex *)d_KernelSpectrum0, FFT_DIR));
  spPostprocess2D(d_KernelSpectrum, d_KernelSpectrum0, fftH, fftW / 2,
                  fftPadding, FFT_DIR);

  printf("...running GPU FFT colwolution: ");
  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  checkLwdaErrors(lwfftExecC2C(fftPlan, (lwfftComplex *)d_PaddedData,
                               (lwfftComplex *)d_DataSpectrum0, FFT_DIR));

  spPostprocess2D(d_DataSpectrum, d_DataSpectrum0, fftH, fftW / 2, fftPadding,
                  FFT_DIR);
  modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW,
                       fftPadding);
  spPreprocess2D(d_DataSpectrum0, d_DataSpectrum, fftH, fftW / 2, fftPadding,
                 -FFT_DIR);

  checkLwdaErrors(lwfftExecC2C(fftPlan, (lwfftComplex *)d_DataSpectrum0,
                               (lwfftComplex *)d_PaddedData, -FFT_DIR));

  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  double gpuTime = sdkGetTimerValue(&hTimer);
  printf("%f MPix/s (%f ms)\n",
         (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

  printf("...reading back GPU FFT results\n");
  checkLwdaErrors(lwdaMemcpy(h_ResultGPU, d_PaddedData,
                             fftH * fftW * sizeof(float),
                             lwdaMemcpyDeviceToHost));

  printf("...running reference CPU colwolution\n");
  colwolutionClampToBorderCPU(h_ResultCPU, h_Data, h_Kernel, dataH, dataW,
                              kernelH, kernelW, kernelY, kernelX);

  printf("...comparing the results: ");
  double sum_delta2 = 0;
  double sum_ref2 = 0;
  double max_delta_ref = 0;

  for (int y = 0; y < dataH; y++)
    for (int x = 0; x < dataW; x++) {
      double rCPU = (double)h_ResultCPU[y * dataW + x];
      double rGPU = (double)h_ResultGPU[y * fftW + x];
      double delta = (rCPU - rGPU) * (rCPU - rGPU);
      double ref = rCPU * rCPU + rCPU * rCPU;

      if ((delta / ref) > max_delta_ref) {
        max_delta_ref = delta / ref;
      }

      sum_delta2 += delta;
      sum_ref2 += ref;
    }

  double L2norm = sqrt(sum_delta2 / sum_ref2);
  printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
  bRetVal = (L2norm < 1e-6) ? true : false;
  printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");

  printf("...shutting down\n");
  sdkDeleteTimer(&hTimer);
  checkLwdaErrors(lwfftDestroy(fftPlan));

  checkLwdaErrors(lwdaFree(d_KernelSpectrum));
  checkLwdaErrors(lwdaFree(d_DataSpectrum));
  checkLwdaErrors(lwdaFree(d_KernelSpectrum0));
  checkLwdaErrors(lwdaFree(d_DataSpectrum0));
  checkLwdaErrors(lwdaFree(d_PaddedKernel));
  checkLwdaErrors(lwdaFree(d_PaddedData));
  checkLwdaErrors(lwdaFree(d_Kernel));
  checkLwdaErrors(lwdaFree(d_Data));

  free(h_ResultGPU);
  free(h_ResultCPU);
  free(h_Kernel);
  free(h_Data);

  return bRetVal;
}

bool test2(void) {
  float *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;

  float *d_Data, *d_Kernel, *d_PaddedData, *d_PaddedKernel;

  fComplex *d_DataSpectrum0, *d_KernelSpectrum0;

  lwfftHandle fftPlan;

  bool bRetVal;
  StopWatchInterface *hTimer = NULL;
  sdkCreateTimer(&hTimer);

  printf("Testing updated custom R2C / C2R FFT-based colwolution\n");
  const int kernelH = 7;
  const int kernelW = 6;
  const int kernelY = 3;
  const int kernelX = 4;
  const int dataH = 2000;
  const int dataW = 2000;
  const int fftH = snapTransformSize(dataH + kernelH - 1);
  const int fftW = snapTransformSize(dataW + kernelW - 1);

  printf("...allocating memory\n");
  h_Data = (float *)malloc(dataH * dataW * sizeof(float));
  h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
  h_ResultCPU = (float *)malloc(dataH * dataW * sizeof(float));
  h_ResultGPU = (float *)malloc(fftH * fftW * sizeof(float));

  checkLwdaErrors(lwdaMalloc((void **)&d_Data, dataH * dataW * sizeof(float)));
  checkLwdaErrors(
      lwdaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

  checkLwdaErrors(
      lwdaMalloc((void **)&d_PaddedData, fftH * fftW * sizeof(float)));
  checkLwdaErrors(
      lwdaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

  checkLwdaErrors(lwdaMalloc((void **)&d_DataSpectrum0,
                             fftH * (fftW / 2) * sizeof(fComplex)));
  checkLwdaErrors(lwdaMalloc((void **)&d_KernelSpectrum0,
                             fftH * (fftW / 2) * sizeof(fComplex)));

  printf("...generating random input data\n");
  srand(2010);

  for (int i = 0; i < dataH * dataW; i++) {
    h_Data[i] = getRand();
  }

  for (int i = 0; i < kernelH * kernelW; i++) {
    h_Kernel[i] = getRand();
  }

  printf("...creating C2C FFT plan for %i x %i\n", fftH, fftW / 2);
  checkLwdaErrors(lwfftPlan2d(&fftPlan, fftH, fftW / 2, LWFFT_C2C));

  printf("...uploading to GPU and padding colwolution kernel and input data\n");
  checkLwdaErrors(lwdaMemcpy(d_Data, h_Data, dataH * dataW * sizeof(float),
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemcpy(d_Kernel, h_Kernel,
                             kernelH * kernelW * sizeof(float),
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));
  checkLwdaErrors(lwdaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));

  padDataClampToBorder(d_PaddedData, d_Data, fftH, fftW, dataH, dataW, kernelH,
                       kernelW, kernelY, kernelX);

  padKernel(d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW, kernelY,
            kernelX);

  // LWFFT_ILWERSE works just as well...
  const int FFT_DIR = LWFFT_FORWARD;

  // Not including kernel transformation into time measurement,
  // since colwolution kernel is not changed very frequently
  printf("...transforming colwolution kernel\n");
  checkLwdaErrors(lwfftExecC2C(fftPlan, (lwfftComplex *)d_PaddedKernel,
                               (lwfftComplex *)d_KernelSpectrum0, FFT_DIR));

  printf("...running GPU FFT colwolution: ");
  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  checkLwdaErrors(lwfftExecC2C(fftPlan, (lwfftComplex *)d_PaddedData,
                               (lwfftComplex *)d_DataSpectrum0, FFT_DIR));
  spProcess2D(d_DataSpectrum0, d_DataSpectrum0, d_KernelSpectrum0, fftH,
              fftW / 2, FFT_DIR);
  checkLwdaErrors(lwfftExecC2C(fftPlan, (lwfftComplex *)d_DataSpectrum0,
                               (lwfftComplex *)d_PaddedData, -FFT_DIR));

  checkLwdaErrors(lwdaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  double gpuTime = sdkGetTimerValue(&hTimer);
  printf("%f MPix/s (%f ms)\n",
         (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

  printf("...reading back GPU FFT results\n");
  checkLwdaErrors(lwdaMemcpy(h_ResultGPU, d_PaddedData,
                             fftH * fftW * sizeof(float),
                             lwdaMemcpyDeviceToHost));

  printf("...running reference CPU colwolution\n");
  colwolutionClampToBorderCPU(h_ResultCPU, h_Data, h_Kernel, dataH, dataW,
                              kernelH, kernelW, kernelY, kernelX);

  printf("...comparing the results: ");
  double sum_delta2 = 0;
  double sum_ref2 = 0;
  double max_delta_ref = 0;

  for (int y = 0; y < dataH; y++) {
    for (int x = 0; x < dataW; x++) {
      double rCPU = (double)h_ResultCPU[y * dataW + x];
      double rGPU = (double)h_ResultGPU[y * fftW + x];
      double delta = (rCPU - rGPU) * (rCPU - rGPU);
      double ref = rCPU * rCPU + rCPU * rCPU;

      if ((delta / ref) > max_delta_ref) {
        max_delta_ref = delta / ref;
      }

      sum_delta2 += delta;
      sum_ref2 += ref;
    }
  }

  double L2norm = sqrt(sum_delta2 / sum_ref2);
  printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
  bRetVal = (L2norm < 1e-6) ? true : false;
  printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");

  printf("...shutting down\n");
  sdkDeleteTimer(&hTimer);
  checkLwdaErrors(lwfftDestroy(fftPlan));

  checkLwdaErrors(lwdaFree(d_KernelSpectrum0));
  checkLwdaErrors(lwdaFree(d_DataSpectrum0));
  checkLwdaErrors(lwdaFree(d_PaddedKernel));
  checkLwdaErrors(lwdaFree(d_PaddedData));
  checkLwdaErrors(lwdaFree(d_Kernel));
  checkLwdaErrors(lwdaFree(d_Data));

  free(h_ResultGPU);
  free(h_ResultCPU);
  free(h_Kernel);
  free(h_Data);

  return bRetVal;
}

int main(int argc, char **argv) {
  printf("[%s] - Starting...\n", argv[0]);

  // Use command-line specified LWCA device, otherwise use device with highest
  // Gflops/s
  findLwdaDevice(argc, (const char **)argv);

  int nFailures = 0;

  if (!test0()) {
    nFailures++;
  }

  if (!test1()) {
    nFailures++;
  }

  if (!test2()) {
    nFailures++;
  }

  printf("Test Summary: %d errors\n", nFailures);

  if (nFailures > 0) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
