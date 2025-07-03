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
 * This sample implements a conjugate gradient solver on GPU
 * using LWBLAS and LWSPARSE
 *
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Using updated (v2) interfaces to lwblas */
#include <lwblas_v2.h>
#include <lwda_runtime.h>
#include <lwsparse.h>

// Utilities and system includes
#include <helper_lwda.h>  // helper function LWCA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to LWCA Samples

const char *sSDKname = "conjugateGradient";

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz) {
  I[0] = 0, J[0] = 0, J[1] = 1;
  val[0] = (float)rand() / RAND_MAX + 10.0f;
  val[1] = (float)rand() / RAND_MAX;
  int start;

  for (int i = 1; i < N; i++) {
    if (i > 1) {
      I[i] = I[i - 1] + 3;
    } else {
      I[1] = 2;
    }

    start = (i - 1) * 3 + 2;
    J[start] = i - 1;
    J[start + 1] = i;

    if (i < N - 1) {
      J[start + 2] = i + 1;
    }

    val[start] = val[start - 1];
    val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

    if (i < N - 1) {
      val[start + 2] = (float)rand() / RAND_MAX;
    }
  }

  I[N] = nz;
}

int main(int argc, char **argv) {
  int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
  float *val = NULL;
  const float tol = 1e-5f;
  const int max_iter = 10000;
  float *x;
  float *rhs;
  float a, b, na, r0, r1;
  int *d_col, *d_row;
  float *d_val, *d_x, dot;
  float *d_r, *d_p, *d_Ax;
  int k;
  float alpha, beta, alpham1;

  // This will pick the best possible LWCA capable device
  lwdaDeviceProp deviceProp;
  int devID = findLwdaDevice(argc, (const char **)argv);

  if (devID < 0) {
    printf("exiting...\n");
    exit(EXIT_SUCCESS);
  }

  checkLwdaErrors(lwdaGetDeviceProperties(&deviceProp, devID));

  // Statistics about the GPU device
  printf(
      "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  /* Generate a random tridiagonal symmetric matrix in CSR format */
  M = N = 1048576;
  nz = (N - 2) * 3 + 4;
  I = (int *)malloc(sizeof(int) * (N + 1));
  J = (int *)malloc(sizeof(int) * nz);
  val = (float *)malloc(sizeof(float) * nz);
  genTridiag(I, J, val, N, nz);

  x = (float *)malloc(sizeof(float) * N);
  rhs = (float *)malloc(sizeof(float) * N);

  for (int i = 0; i < N; i++) {
    rhs[i] = 1.0;
    x[i] = 0.0;
  }

  /* Get handle to the LWBLAS context */
  lwblasHandle_t lwblasHandle = 0;
  lwblasStatus_t lwblasStatus;
  lwblasStatus = lwblasCreate(&lwblasHandle);

  checkLwdaErrors(lwblasStatus);

  /* Get handle to the LWSPARSE context */
  lwsparseHandle_t lwsparseHandle = 0;
  checkLwdaErrors(lwsparseCreate(&lwsparseHandle));

  checkLwdaErrors(lwdaMalloc((void **)&d_col, nz * sizeof(int)));
  checkLwdaErrors(lwdaMalloc((void **)&d_row, (N + 1) * sizeof(int)));
  checkLwdaErrors(lwdaMalloc((void **)&d_val, nz * sizeof(float)));
  checkLwdaErrors(lwdaMalloc((void **)&d_x, N * sizeof(float)));
  checkLwdaErrors(lwdaMalloc((void **)&d_r, N * sizeof(float)));
  checkLwdaErrors(lwdaMalloc((void **)&d_p, N * sizeof(float)));
  checkLwdaErrors(lwdaMalloc((void **)&d_Ax, N * sizeof(float)));

  /* Wrap raw data into lwSPARSE generic API objects */
  lwsparseSpMatDescr_t matA = NULL;
  checkLwdaErrors(lwsparseCreateCsr(&matA, N, N, nz, d_row, d_col, d_val,
                                    LWSPARSE_INDEX_32I, LWSPARSE_INDEX_32I,
                                    LWSPARSE_INDEX_BASE_ZERO, LWDA_R_32F));
  lwsparseDlwecDescr_t vecx = NULL;
  checkLwdaErrors(lwsparseCreateDlwec(&vecx, N, d_x, LWDA_R_32F));
  lwsparseDlwecDescr_t vecp = NULL;
  checkLwdaErrors(lwsparseCreateDlwec(&vecp, N, d_p, LWDA_R_32F));
  lwsparseDlwecDescr_t vecAx = NULL;
  checkLwdaErrors(lwsparseCreateDlwec(&vecAx, N, d_Ax, LWDA_R_32F));

  /* Initialize problem data */
  lwdaMemcpy(d_col, J, nz * sizeof(int), lwdaMemcpyHostToDevice);
  lwdaMemcpy(d_row, I, (N + 1) * sizeof(int), lwdaMemcpyHostToDevice);
  lwdaMemcpy(d_val, val, nz * sizeof(float), lwdaMemcpyHostToDevice);
  lwdaMemcpy(d_x, x, N * sizeof(float), lwdaMemcpyHostToDevice);
  lwdaMemcpy(d_r, rhs, N * sizeof(float), lwdaMemcpyHostToDevice);

  alpha = 1.0;
  alpham1 = -1.0;
  beta = 0.0;
  r0 = 0.;

  /* Allocate workspace for lwSPARSE */
  size_t bufferSize = 0;
  checkLwdaErrors(lwsparseSpMV_bufferSize(
      lwsparseHandle, LWSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
      &beta, vecAx, LWDA_R_32F, LWSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  void *buffer = NULL;
  checkLwdaErrors(lwdaMalloc(&buffer, bufferSize));

  /* Begin CG */
  checkLwdaErrors(lwsparseSpMV(lwsparseHandle, LWSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, LWDA_R_32F,
                               LWSPARSE_SPMV_ALG_DEFAULT, buffer));

  lwblasSaxpy(lwblasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
  lwblasStatus = lwblasSdot(lwblasHandle, N, d_r, 1, d_r, 1, &r1);

  k = 1;

  while (r1 > tol * tol && k <= max_iter) {
    if (k > 1) {
      b = r1 / r0;
      lwblasStatus = lwblasSscal(lwblasHandle, N, &b, d_p, 1);
      lwblasStatus = lwblasSaxpy(lwblasHandle, N, &alpha, d_r, 1, d_p, 1);
    } else {
      lwblasStatus = lwblasScopy(lwblasHandle, N, d_r, 1, d_p, 1);
    }

    checkLwdaErrors(lwsparseSpMV(
        lwsparseHandle, LWSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
        &beta, vecAx, LWDA_R_32F, LWSPARSE_SPMV_ALG_DEFAULT, buffer));
    lwblasStatus = lwblasSdot(lwblasHandle, N, d_p, 1, d_Ax, 1, &dot);
    a = r1 / dot;

    lwblasStatus = lwblasSaxpy(lwblasHandle, N, &a, d_p, 1, d_x, 1);
    na = -a;
    lwblasStatus = lwblasSaxpy(lwblasHandle, N, &na, d_Ax, 1, d_r, 1);

    r0 = r1;
    lwblasStatus = lwblasSdot(lwblasHandle, N, d_r, 1, d_r, 1, &r1);
    lwdaDeviceSynchronize();
    printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    k++;
  }

  lwdaMemcpy(x, d_x, N * sizeof(float), lwdaMemcpyDeviceToHost);

  float rsum, diff, err = 0.0;

  for (int i = 0; i < N; i++) {
    rsum = 0.0;

    for (int j = I[i]; j < I[i + 1]; j++) {
      rsum += val[j] * x[J[j]];
    }

    diff = fabs(rsum - rhs[i]);

    if (diff > err) {
      err = diff;
    }
  }

  lwsparseDestroy(lwsparseHandle);
  lwblasDestroy(lwblasHandle);
  if (matA) {
    checkLwdaErrors(lwsparseDestroySpMat(matA));
  }
  if (vecx) {
    checkLwdaErrors(lwsparseDestroyDlwec(vecx));
  }
  if (vecAx) {
    checkLwdaErrors(lwsparseDestroyDlwec(vecAx));
  }
  if (vecp) {
    checkLwdaErrors(lwsparseDestroyDlwec(vecp));
  }

  free(I);
  free(J);
  free(val);
  free(x);
  free(rhs);
  lwdaFree(d_col);
  lwdaFree(d_row);
  lwdaFree(d_val);
  lwdaFree(d_x);
  lwdaFree(d_r);
  lwdaFree(d_p);
  lwdaFree(d_Ax);

  printf("Test Summary:  Error amount = %f\n", err);
  exit((k <= max_iter) ? 0 : 1);
}
