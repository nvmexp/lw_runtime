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

/* Using updated (v2) interfaces to lwblas and lwsparse */
#include <lwblas_v2.h>
#include <lwda_runtime.h>
#include <lwsparse.h>

// Utilities and system includes
#include <helper_lwda.h>  // helper function LWCA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to LWCA Samples

const char *sSDKname = "conjugateGradientUM";

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
  int N = 0, nz = 0, *I = NULL, *J = NULL;
  float *val = NULL;
  const float tol = 1e-5f;
  const int max_iter = 10000;
  float *x;
  float *rhs;
  float a, b, na, r0, r1;
  float dot;
  float *r, *p, *Ax;
  int k;
  float alpha, beta, alpham1;

  printf("Starting [%s]...\n", sSDKname);

  // This will pick the best possible LWCA capable device
  lwdaDeviceProp deviceProp;
  int devID = findLwdaDevice(argc, (const char **)argv);
  checkLwdaErrors(lwdaGetDeviceProperties(&deviceProp, devID));

  if (!deviceProp.managedMemory) {
    // This samples requires being run on a device that supports Unified Memory
    fprintf(stderr, "Unified Memory not supported on this device\n");
    exit(EXIT_WAIVED);
  }

  // Statistics about the GPU device
  printf(
      "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  /* Generate a random tridiagonal symmetric matrix in CSR format */
  N = 1048576;
  nz = (N - 2) * 3 + 4;

  lwdaMallocManaged((void **)&I, sizeof(int) * (N + 1));
  lwdaMallocManaged((void **)&J, sizeof(int) * nz);
  lwdaMallocManaged((void **)&val, sizeof(float) * nz);

  genTridiag(I, J, val, N, nz);

  lwdaMallocManaged((void **)&x, sizeof(float) * N);
  lwdaMallocManaged((void **)&rhs, sizeof(float) * N);

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
  lwsparseStatus_t lwsparseStatus;
  lwsparseStatus = lwsparseCreate(&lwsparseHandle);

  checkLwdaErrors(lwsparseStatus);

  lwsparseMatDescr_t descr = 0;
  lwsparseStatus = lwsparseCreateMatDescr(&descr);

  checkLwdaErrors(lwsparseStatus);

  lwsparseSetMatType(descr, LWSPARSE_MATRIX_TYPE_GENERAL);
  lwsparseSetMatIndexBase(descr, LWSPARSE_INDEX_BASE_ZERO);

  // temp memory for CG
  checkLwdaErrors(lwdaMallocManaged((void **)&r, N * sizeof(float)));
  checkLwdaErrors(lwdaMallocManaged((void **)&p, N * sizeof(float)));
  checkLwdaErrors(lwdaMallocManaged((void **)&Ax, N * sizeof(float)));

  /* Wrap raw data into lwSPARSE generic API objects */
  lwsparseSpMatDescr_t matA = NULL;
  checkLwdaErrors(lwsparseCreateCsr(&matA, N, N, nz, I, J, val,
                                    LWSPARSE_INDEX_32I, LWSPARSE_INDEX_32I,
                                    LWSPARSE_INDEX_BASE_ZERO, LWDA_R_32F));
  lwsparseDlwecDescr_t vecx = NULL;
  checkLwdaErrors(lwsparseCreateDlwec(&vecx, N, x, LWDA_R_32F));
  lwsparseDlwecDescr_t vecp = NULL;
  checkLwdaErrors(lwsparseCreateDlwec(&vecp, N, p, LWDA_R_32F));
  lwsparseDlwecDescr_t vecAx = NULL;
  checkLwdaErrors(lwsparseCreateDlwec(&vecAx, N, Ax, LWDA_R_32F));

  lwdaDeviceSynchronize();

  for (int i = 0; i < N; i++) {
    r[i] = rhs[i];
  }

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

  checkLwdaErrors(lwsparseSpMV(lwsparseHandle, LWSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, LWDA_R_32F,
                               LWSPARSE_SPMV_ALG_DEFAULT, buffer));

  lwblasSaxpy(lwblasHandle, N, &alpham1, Ax, 1, r, 1);
  lwblasStatus = lwblasSdot(lwblasHandle, N, r, 1, r, 1, &r1);

  k = 1;

  while (r1 > tol * tol && k <= max_iter) {
    if (k > 1) {
      b = r1 / r0;
      lwblasStatus = lwblasSscal(lwblasHandle, N, &b, p, 1);
      lwblasStatus = lwblasSaxpy(lwblasHandle, N, &alpha, r, 1, p, 1);
    } else {
      lwblasStatus = lwblasScopy(lwblasHandle, N, r, 1, p, 1);
    }

    checkLwdaErrors(lwsparseSpMV(
        lwsparseHandle, LWSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
        &beta, vecAx, LWDA_R_32F, LWSPARSE_SPMV_ALG_DEFAULT, buffer));
    lwblasStatus = lwblasSdot(lwblasHandle, N, p, 1, Ax, 1, &dot);
    a = r1 / dot;

    lwblasStatus = lwblasSaxpy(lwblasHandle, N, &a, p, 1, x, 1);
    na = -a;
    lwblasStatus = lwblasSaxpy(lwblasHandle, N, &na, Ax, 1, r, 1);

    r0 = r1;
    lwblasStatus = lwblasSdot(lwblasHandle, N, r, 1, r, 1, &r1);
    lwdaDeviceSynchronize();
    printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    k++;
  }

  printf("Final residual: %e\n", sqrt(r1));

  fprintf(stdout, "&&&& conjugateGradientUM %s\n",
          (sqrt(r1) < tol) ? "PASSED" : "FAILED");

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

  lwdaFree(I);
  lwdaFree(J);
  lwdaFree(val);
  lwdaFree(x);
  lwdaFree(rhs);
  lwdaFree(r);
  lwdaFree(p);
  lwdaFree(Ax);

  printf("Test Summary:  Error amount = %f, result = %s\n", err,
         (k <= max_iter) ? "SUCCESS" : "FAILURE");
  exit((k <= max_iter) ? EXIT_SUCCESS : EXIT_FAILURE);
}
