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

// This sample demonstrates Instantiated LWCA Graph Update
// with Jacobi Iterative Method in 3 different methods:
// 1 - JacobiMethodGpuLwdaGraphExecKernelSetParams() - LWCA Graph with
// lwdaGraphExecKernelNodeSetParams() 2 - JacobiMethodGpuLwdaGraphExelwpdate() -
// LWCA Graph with lwdaGraphExelwpdate() 3 - JacobiMethodGpu() - Non LWCA Graph
// method

// Jacobi method on a linear system A*x = b,
// where A is diagonally dominant and the exact solution consists
// of all ones.
// The dimension N_ROWS is included in jacobi.h

#include <lwda_runtime.h>
#include <helper_lwda.h>
#include <helper_timer.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "jacobi.h"

// Run the Jacobi method for A*x = b on GPU with LWCA Graph -
// lwdaGraphExecKernelNodeSetParams().
extern double JacobiMethodGpuLwdaGraphExecKernelSetParams(
    const float *A, const double *b, const float colw_threshold,
    const int max_iter, double *x, double *x_new, lwdaStream_t stream);

// Run the Jacobi method for A*x = b on GPU with Instantiated LWCA Graph Update
// API - lwdaGraphExelwpdate().
extern double JacobiMethodGpuLwdaGraphExelwpdate(
    const float *A, const double *b, const float colw_threshold,
    const int max_iter, double *x, double *x_new, lwdaStream_t stream);

// Run the Jacobi method for A*x = b on GPU without LWCA Graph.
extern double JacobiMethodGpu(const float *A, const double *b,
                              const float colw_threshold, const int max_iter,
                              double *x, double *x_new, lwdaStream_t stream);

// creates N_ROWS x N_ROWS matrix A with N_ROWS+1 on the diagonal and 1
// elsewhere. The elements of the right hand side b all equal 2*n, hence the
// exact solution x to A*x = b is a vector of ones.
void createLinearSystem(float *A, double *b);

// Run the Jacobi method for A*x = b on CPU.
void JacobiMethodCPU(float *A, double *b, float colw_threshold, int max_iter,
                     int *numit, double *x);

int main(int argc, char **argv) {
  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printf("Command line: jacobiLwdaGraphs [-option]\n");
    printf("Valid options:\n");
    printf(
        "-gpumethod=<0,1 or 2>  : 0 - [Default] "
        "JacobiMethodGpuLwdaGraphExecKernelSetParams\n");
    printf("                       : 1 - JacobiMethodGpuLwdaGraphExelwpdate\n");
    printf("                       : 2 - JacobiMethodGpu - Non LWCA Graph\n");
    printf("-device=device_num     : lwca device id");
    printf("-help         : Output a help message\n");
    exit(EXIT_SUCCESS);
  }

  int gpumethod = 0;
  if (checkCmdLineFlag(argc, (const char **)argv, "gpumethod")) {
    gpumethod = getCmdLineArgumentInt(argc, (const char **)argv, "gpumethod");

    if (gpumethod < 0 || gpumethod > 2) {
      printf("Error: gpumethod must be 0 or 1 or 2, gpumethod=%d is invalid\n",
             gpumethod);
      exit(EXIT_SUCCESS);
    }
  }

  int dev = findLwdaDevice(argc, (const char **)argv);

  double *b = NULL;
  float *A = NULL;
  checkLwdaErrors(lwdaMallocHost(&b, N_ROWS * sizeof(double)));
  memset(b, 0, N_ROWS * sizeof(double));
  checkLwdaErrors(lwdaMallocHost(&A, N_ROWS * N_ROWS * sizeof(float)));
  memset(A, 0, N_ROWS * N_ROWS * sizeof(float));

  createLinearSystem(A, b);
  double *x = NULL;
  // start with array of all zeroes
  x = (double *)calloc(N_ROWS, sizeof(double));

  float colw_threshold = 1.0e-2;
  int max_iter = 4 * N_ROWS * N_ROWS;
  int cnt = 0;

  // create timer
  StopWatchInterface *timerCPU = NULL, *timerGpu = NULL;
  sdkCreateTimer(&timerCPU);

  sdkStartTimer(&timerCPU);
  JacobiMethodCPU(A, b, colw_threshold, max_iter, &cnt, x);

  double sum = 0.0;
  // Compute error
  for (int i = 0; i < N_ROWS; i++) {
    double d = x[i] - 1.0;
    sum += fabs(d);
  }
  sdkStopTimer(&timerCPU);
  printf("CPU iterations : %d\n", cnt);
  printf("CPU error : %.3e\n", sum);
  printf("CPU Processing time: %f (ms)\n", sdkGetTimerValue(&timerCPU));

  float *d_A;
  double *d_b, *d_x, *d_x_new;
  lwdaStream_t stream1;
  checkLwdaErrors(lwdaStreamCreateWithFlags(&stream1, lwdaStreamNonBlocking));
  checkLwdaErrors(lwdaMalloc(&d_b, sizeof(double) * N_ROWS));
  checkLwdaErrors(lwdaMalloc(&d_A, sizeof(float) * N_ROWS * N_ROWS));
  checkLwdaErrors(lwdaMalloc(&d_x, sizeof(double) * N_ROWS));
  checkLwdaErrors(lwdaMalloc(&d_x_new, sizeof(double) * N_ROWS));

  checkLwdaErrors(lwdaMemsetAsync(d_x, 0, sizeof(double) * N_ROWS, stream1));
  checkLwdaErrors(
      lwdaMemsetAsync(d_x_new, 0, sizeof(double) * N_ROWS, stream1));
  checkLwdaErrors(lwdaMemcpyAsync(d_A, A, sizeof(float) * N_ROWS * N_ROWS,
                                  lwdaMemcpyHostToDevice, stream1));
  checkLwdaErrors(lwdaMemcpyAsync(d_b, b, sizeof(double) * N_ROWS,
                                  lwdaMemcpyHostToDevice, stream1));

  sdkCreateTimer(&timerGpu);
  sdkStartTimer(&timerGpu);

  double sumGPU = 0.0;
  if (gpumethod == 0) {
    sumGPU = JacobiMethodGpuLwdaGraphExecKernelSetParams(
        d_A, d_b, colw_threshold, max_iter, d_x, d_x_new, stream1);
  } else if (gpumethod == 1) {
    sumGPU = JacobiMethodGpuLwdaGraphExelwpdate(
        d_A, d_b, colw_threshold, max_iter, d_x, d_x_new, stream1);
  } else if (gpumethod == 2) {
    sumGPU = JacobiMethodGpu(d_A, d_b, colw_threshold, max_iter, d_x, d_x_new,
                             stream1);
  }

  sdkStopTimer(&timerGpu);
  printf("GPU Processing time: %f (ms)\n", sdkGetTimerValue(&timerGpu));

  checkLwdaErrors(lwdaFree(d_b));
  checkLwdaErrors(lwdaFree(d_A));
  checkLwdaErrors(lwdaFree(d_x));
  checkLwdaErrors(lwdaFree(d_x_new));

  checkLwdaErrors(lwdaFreeHost(A));
  checkLwdaErrors(lwdaFreeHost(b));

  printf("&&&& jacobiLwdaGraphs %s\n",
         (fabs(sum - sumGPU) < colw_threshold) ? "PASSED" : "FAILED");

  return (fabs(sum - sumGPU) < colw_threshold) ? EXIT_SUCCESS : EXIT_FAILURE;
}

void createLinearSystem(float *A, double *b) {
  int i, j;
  for (i = 0; i < N_ROWS; i++) {
    b[i] = 2.0 * N_ROWS;
    for (j = 0; j < N_ROWS; j++) A[i * N_ROWS + j] = 1.0;
    A[i * N_ROWS + i] = N_ROWS + 1.0;
  }
}

void JacobiMethodCPU(float *A, double *b, float colw_threshold, int max_iter,
                     int *num_iter, double *x) {
  double *x_new;
  x_new = (double *)calloc(N_ROWS, sizeof(double));
  int k;

  for (k = 0; k < max_iter; k++) {
    double sum = 0.0;
    for (int i = 0; i < N_ROWS; i++) {
      double temp_dx = b[i];
      for (int j = 0; j < N_ROWS; j++) temp_dx -= A[i * N_ROWS + j] * x[j];
      temp_dx /= A[i * N_ROWS + i];
      x_new[i] += temp_dx;
      sum += fabs(temp_dx);
    }

    for (int i = 0; i < N_ROWS; i++) x[i] = x_new[i];

    if (sum <= colw_threshold) break;
  }
  *num_iter = k + 1;
  free(x_new);
}
