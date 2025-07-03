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
 * This example demonstrates how to use the lwBLAS library API
 * for lower-upper (LU) decomposition of a matrix. LU decomposition
 * factors a matrix as the product of upper triangular matrix and
 * lower trianglular matrix.
 *
 * https://en.wikipedia.org/wiki/LU_decomposition
 *
 * This sample uses 10000 matrices of size 4x4 and performs
 * LU decomposition of them using batched decomposition API
 * of lwBLAS library. To test the correctness of upper and lower
 * matrices generated, they are multiplied and compared with the
 * original input matrix.
 *
 */

#include <stdio.h>
#include <stdlib.h>

// lwca libraries and helpers
#include <lwblas_v2.h>
#include <lwda_runtime.h>
#include <helper_lwda.h>

// configurable parameters
// dimension of matrix
#define N 4
#define BATCH_SIZE 10000

// use double precision data type
#define DOUBLE_PRECISION /* comment this to use single precision */
#ifdef DOUBLE_PRECISION
#define DATA_TYPE double
#define MAX_ERROR 1e-15
#else
#define DATA_TYPE float
#define MAX_ERROR 1e-6
#endif /* DOUBLE_PRCISION */

// use pivot vector while decomposing
#define PIVOT /* comment this to disable pivot use */

// helper functions

// wrapper around lwblas<t>getrfBatched()
lwblasStatus_t lwblasXgetrfBatched(lwblasHandle_t handle, int n,
                                   DATA_TYPE* const A[], int lda, int* P,
                                   int* info, int batchSize) {
#ifdef DOUBLE_PRECISION
  return lwblasDgetrfBatched(handle, n, A, lda, P, info, batchSize);
#else
  return lwblasSgetrfBatched(handle, n, A, lda, P, info, batchSize);
#endif
}

// wrapper around malloc
// clears the allocated memory to 0
// terminates the program if malloc fails
void* xmalloc(size_t size) {
  void* ptr = malloc(size);
  if (ptr == NULL) {
    printf("> ERROR: malloc for size %zu failed..\n", size);
    exit(EXIT_FAILURE);
  }
  memset(ptr, 0, size);
  return ptr;
}

// initalize identity matrix
void initIdentityMatrix(DATA_TYPE* mat) {
  // clear the matrix
  memset(mat, 0, N * N * sizeof(DATA_TYPE));

  // set all diagonals to 1
  for (int i = 0; i < N; i++) {
    mat[(i * N) + i] = 1.0;
  }
}

// initialize matrix with all elements as 0
void initZeroMatrix(DATA_TYPE* mat) {
  memset(mat, 0, N * N * sizeof(DATA_TYPE));
}

// fill random value in column-major matrix
void initRandomMatrix(DATA_TYPE* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mat[(j * N) + i] =
          (DATA_TYPE)1.0 + ((DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX);
    }
  }

  // diagonal dominant matrix to insure it is ilwertible matrix
  for (int i = 0; i < N; i++) {
    mat[(i * N) + i] += (DATA_TYPE)N;
  }
}

// print column-major matrix
void printMatrix(DATA_TYPE* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%20.16f ", mat[(j * N) + i]);
    }
    printf("\n");
  }
  printf("\n");
}

// matrix mulitplication
void matrixMultiply(DATA_TYPE* res, DATA_TYPE* mat1, DATA_TYPE* mat2) {
  initZeroMatrix(res);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        res[(j * N) + i] += mat1[(k * N) + i] * mat2[(j * N) + k];
      }
    }
  }
}

// check matrix equality
bool checkRelativeError(DATA_TYPE* mat1, DATA_TYPE* mat2, DATA_TYPE maxError) {
  DATA_TYPE err = (DATA_TYPE)0.0;
  DATA_TYPE refNorm = (DATA_TYPE)0.0;
  DATA_TYPE relError = (DATA_TYPE)0.0;
  DATA_TYPE relMaxError = (DATA_TYPE)0.0;

  for (int i = 0; i < N * N; i++) {
    refNorm = abs(mat1[i]);
    err = abs(mat1[i] - mat2[i]);

    if (refNorm != 0.0 && err > 0.0) {
      relError = err / refNorm;
      relMaxError = MAX(relMaxError, relError);
    }

    if (relMaxError > maxError) return false;
  }
  return true;
}

// decode lower and upper matrix from single matrix
// returned by getrfBatched()
void getLUdecoded(DATA_TYPE* mat, DATA_TYPE* L, DATA_TYPE* U) {
  // init L as identity matrix
  initIdentityMatrix(L);

  // copy lower triangular values from mat to L (skip diagonal)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < i; j++) {
      L[(j * N) + i] = mat[(j * N) + i];
    }
  }

  // init U as all zero
  initZeroMatrix(U);

  // copy upper triangular values from mat to U
  for (int i = 0; i < N; i++) {
    for (int j = i; j < N; j++) {
      U[(j * N) + i] = mat[(j * N) + i];
    }
  }
}

// generate permutation matrix from pivot vector
void getPmatFromPivot(DATA_TYPE* Pmat, int* P) {
  int pivot[N];

  // pivot vector in base-1
  // colwert it to base-0
  for (int i = 0; i < N; i++) {
    P[i]--;
  }

  // generate permutation vector from pivot
  // initialize pivot with identity sequence
  for (int k = 0; k < N; k++) {
    pivot[k] = k;
  }

  // swap the indices according to pivot vector
  for (int k = 0; k < N; k++) {
    int q = P[k];

    // swap pivot(k) and pivot(q)
    int s = pivot[k];
    int t = pivot[q];
    pivot[k] = t;
    pivot[q] = s;
  }

  // generate permutation matrix from pivot vector
  initZeroMatrix(Pmat);
  for (int i = 0; i < N; i++) {
    int j = pivot[i];
    Pmat[(j * N) + i] = (DATA_TYPE)1.0;
  }
}

int main(int argc, char** argv) {
  // lwBLAS variables
  lwblasStatus_t status;
  lwblasHandle_t handle;

  // host variables
  size_t matSize = N * N * sizeof(DATA_TYPE);

  DATA_TYPE* h_AarrayInput;
  DATA_TYPE* h_AarrayOutput;
  DATA_TYPE* h_ptr_array[BATCH_SIZE];

  int* h_pivotArray;
  int* h_infoArray;

  // device variables
  DATA_TYPE* d_Aarray;
  DATA_TYPE** d_ptr_array;

  int* d_pivotArray;
  int* d_infoArray;

  int err_count = 0;

  // seed the rand() function with time
  srand(12345);

  // find lwca device
  printf("> initializing..\n");
  int dev = findLwdaDevice(argc, (const char**)argv);
  if (dev == -1) {
    return (EXIT_FAILURE);
  }

  // initialize lwBLAS
  status = lwblasCreate(&handle);
  if (status != LWBLAS_STATUS_SUCCESS) {
    printf("> ERROR: lwBLAS initialization failed..\n");
    return (EXIT_FAILURE);
  }

#ifdef DOUBLE_PRECISION
  printf("> using DOUBLE precision..\n");
#else
  printf("> using SINGLE precision..\n");
#endif

#ifdef PIVOT
  printf("> pivot ENABLED..\n");
#else
  printf("> pivot DISABLED..\n");
#endif

  // allocate memory for host variables
  h_AarrayInput = (DATA_TYPE*)xmalloc(BATCH_SIZE * matSize);
  h_AarrayOutput = (DATA_TYPE*)xmalloc(BATCH_SIZE * matSize);

  h_pivotArray = (int*)xmalloc(N * BATCH_SIZE * sizeof(int));
  h_infoArray = (int*)xmalloc(BATCH_SIZE * sizeof(int));

  // allocate memory for device variables
  checkLwdaErrors(lwdaMalloc((void**)&d_Aarray, BATCH_SIZE * matSize));
  checkLwdaErrors(
      lwdaMalloc((void**)&d_pivotArray, N * BATCH_SIZE * sizeof(int)));
  checkLwdaErrors(lwdaMalloc((void**)&d_infoArray, BATCH_SIZE * sizeof(int)));
  checkLwdaErrors(
      lwdaMalloc((void**)&d_ptr_array, BATCH_SIZE * sizeof(DATA_TYPE*)));

  // fill matrix with random data
  printf("> generating random matrices..\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    initRandomMatrix(h_AarrayInput + (i * N * N));
  }

  // copy data to device from host
  printf("> copying data from host memory to GPU memory..\n");
  checkLwdaErrors(lwdaMemcpy(d_Aarray, h_AarrayInput, BATCH_SIZE * matSize,
                             lwdaMemcpyHostToDevice));

  // create pointer array for matrices
  for (int i = 0; i < BATCH_SIZE; i++) h_ptr_array[i] = d_Aarray + (i * N * N);

  // copy pointer array to device memory
  checkLwdaErrors(lwdaMemcpy(d_ptr_array, h_ptr_array,
                             BATCH_SIZE * sizeof(DATA_TYPE*),
                             lwdaMemcpyHostToDevice));

  // perform LU decomposition
  printf("> performing LU decomposition..\n");
#ifdef PIVOT
  status = lwblasXgetrfBatched(handle, N, d_ptr_array, N, d_pivotArray,
                               d_infoArray, BATCH_SIZE);
#else
  status = lwblasXgetrfBatched(handle, N, d_ptr_array, N, NULL, d_infoArray,
                               BATCH_SIZE);
#endif /* PIVOT */
  if (status != LWBLAS_STATUS_SUCCESS) {
    printf("> ERROR: lwblasDgetrfBatched() failed with error %s..\n",
           _lwdaGetErrorEnum(status));
    return (EXIT_FAILURE);
  }

  // copy data to host from device
  printf("> copying data from GPU memory to host memory..\n");
  checkLwdaErrors(lwdaMemcpy(h_AarrayOutput, d_Aarray, BATCH_SIZE * matSize,
                             lwdaMemcpyDeviceToHost));
  checkLwdaErrors(lwdaMemcpy(h_infoArray, d_infoArray, BATCH_SIZE * sizeof(int),
                             lwdaMemcpyDeviceToHost));
#ifdef PIVOT
  checkLwdaErrors(lwdaMemcpy(h_pivotArray, d_pivotArray,
                             N * BATCH_SIZE * sizeof(int),
                             lwdaMemcpyDeviceToHost));
#endif /* PIVOT */

  // verify the result
  printf("> verifying the result..\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    if (h_infoArray[i] == 0) {
      DATA_TYPE* A = h_AarrayInput + (i * N * N);
      DATA_TYPE* LU = h_AarrayOutput + (i * N * N);
      DATA_TYPE L[N * N];
      DATA_TYPE U[N * N];
      getLUdecoded(LU, L, U);

      // test P * A = L * U
      int* P = h_pivotArray + (i * N);
      DATA_TYPE Pmat[N * N];
#ifdef PIVOT
      getPmatFromPivot(Pmat, P);
#else
      initIdentityMatrix(Pmat);
#endif /* PIVOT */

      // perform matrix multiplication
      DATA_TYPE PxA[N * N];
      DATA_TYPE LxU[N * N];
      matrixMultiply(PxA, Pmat, A);
      matrixMultiply(LxU, L, U);

      // check for equality of matrices
      if (!checkRelativeError(PxA, LxU, (DATA_TYPE)MAX_ERROR)) {
        printf("> ERROR: accuracy check failed for matrix number %05d..\n",
               i + 1);
        err_count++;
      }

    } else if (h_infoArray[i] > 0) {
      printf(
          "> exelwtion for matrix %05d is successful, but U is singular and "
          "U(%d,%d) = 0..\n",
          i + 1, h_infoArray[i] - 1, h_infoArray[i] - 1);
    } else  // (h_infoArray[i] < 0)
    {
      printf("> ERROR: matrix %05d have an illegal value at index %d = %lf..\n",
             i + 1, -h_infoArray[i],
             *(h_AarrayInput + (i * N * N) + (-h_infoArray[i])));
    }
  }

  // free device variables
  checkLwdaErrors(lwdaFree(d_ptr_array));
  checkLwdaErrors(lwdaFree(d_infoArray));
  checkLwdaErrors(lwdaFree(d_pivotArray));
  checkLwdaErrors(lwdaFree(d_Aarray));

  // free host variables
  if (h_infoArray) free(h_infoArray);
  if (h_pivotArray) free(h_pivotArray);
  if (h_AarrayOutput) free(h_AarrayOutput);
  if (h_AarrayInput) free(h_AarrayInput);

  // destroy lwBLAS handle
  status = lwblasDestroy(handle);
  if (status != LWBLAS_STATUS_SUCCESS) {
    printf("> ERROR: lwBLAS uninitialization failed..\n");
    return (EXIT_FAILURE);
  }

  if (err_count > 0) {
    printf("> TEST FAILED for %d matrices, with precision: %g\n", err_count,
           MAX_ERROR);
    return (EXIT_FAILURE);
  }

  printf("> TEST SUCCESSFUL, with precision: %g\n", MAX_ERROR);
  return (EXIT_SUCCESS);
}
