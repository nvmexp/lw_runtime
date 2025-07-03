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
 * This example demonstrates how to get better performance by
 * batching LWBLAS calls with the use of using streams
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <float.h>
#endif

/* Using updated (v2) interfaces to lwblas and lwsparse */
#include <lwblas_v2.h>
#include <lwda_runtime.h>

// Utilities and system includes
#include <helper_lwda.h>

#include "batchLWBLAS.h"

const char *sSDKname = "batchLWBLAS";

//==============================================================================
// Device information utilities
//==============================================================================

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

int getDeviceVersion(void) {
  int device;
  struct lwdaDeviceProp properties;

  if (lwdaGetDevice(&device) != lwdaSuccess) {
    printf("failed to get device\n");
    return 0;
  }

  if (lwdaGetDeviceProperties(&properties, device) != lwdaSuccess) {
    printf("failed to get properties\n");
    return 0;
  }

  return properties.major * 100 + properties.minor * 10;
}

size_t getDeviceMemory(void) {
  struct lwdaDeviceProp properties;
  int device;

  if (lwdaGetDevice(&device) != lwdaSuccess) {
    return 0;
  }

  if (lwdaGetDeviceProperties(&properties, device) != lwdaSuccess) {
    return 0;
  }

  return properties.totalGlobalMem;
}
#if defined(__cplusplus)
}
#endif /* __cplusplus */

//==============================================================================
// random utilities
//==============================================================================

template <typename T_ELEM>
void fillupMatrix(T_ELEM *A, int lda, int rows, int cols, int seed = 0);

template <typename T_ELEM>
void fillupMatrix(T_ELEM *A, int lda, int rows, int cols, int seed) {
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      A[i + lda * j] = lwGet<T_ELEM>(
          ((double)(((lda * i + j + seed) % 253) + 1)) / 256.0,
          ((double)((((cols * i + j) + 123 + seed) % 253) + 1)) / 256.0);
    }
  }
}
/* Explicit instantiation */
template void fillupMatrix<float>(float *A, int lda, int rows, int cols,
                                  int seed);
template void fillupMatrix<double>(double *A, int lda, int rows, int cols,
                                   int seed);

/* For debugging */
void printLwType(const char *str, float A) {
  fprintf(stdout, "%s (0x%08x, %g)", str, floatAsUInt(A), A);
}

void printLwType(const char *str, double A) {
  fprintf(stdout, "%s (0x%016llx, %g)", str, doubleAsULL(A), A);
}

//==============================================================================
// defines and structures
//==============================================================================

#define LWBLAS_SGEMM_MAX_ULP_ERR (.3)
#define LWBLAS_DGEMM_MAX_ULP_ERR (1.e-3)
#define LWBLAS_SGEMM_MAX_RELATIVE_ERR (6.e-6)
#define LWBLAS_DGEMM_MAX_RELATIVE_ERR (0.0)
#define LWBLAS_GEMM_TEST_COUNT (30)
#define BENCH_MATRIX_M (128)
#define BENCH_MATRIX_K (128)
#define BENCH_MATRIX_N (128)

#define CLEANUP()                           \
  do {                                      \
    if (A) free(A);                         \
    if (B) free(B);                         \
    if (C) free(C);                         \
    for (int i = 0; i < opts.N; ++i) {      \
      if (devPtrA[i]) lwdaFree(devPtrA[i]); \
      if (devPtrB[i]) lwdaFree(devPtrB[i]); \
      if (devPtrC[i]) lwdaFree(devPtrC[i]); \
    }                                       \
    if (devPtrA) free(devPtrA);             \
    if (devPtrB) free(devPtrB);             \
    if (devPtrC) free(devPtrC);             \
    if (devPtrA_dev) lwdaFree(devPtrA_dev); \
    if (devPtrB_dev) lwdaFree(devPtrB_dev); \
    if (devPtrC_dev) lwdaFree(devPtrC_dev); \
    fflush(stdout);                         \
  } while (0)

enum testMethod { tmRegular, tmStream, tmBatched };

struct gemmOpts {
  int m;
  int n;
  int k;
  testMethod test_method;
  char *elem_type;
  int N;  // number of multiplications
};

template <typename T_ELEM>
struct gemmTestParams {
  lwblasOperation_t transa;
  lwblasOperation_t transb;
  int m;
  int n;
  int k;
  T_ELEM alpha;
  T_ELEM beta;
};

//==============================================================================
// template wrappers for lwca functions
//==============================================================================

static inline lwblasStatus_t lwblasXgemm(lwblasHandle_t handle,
                                         lwblasOperation_t transa,
                                         lwblasOperation_t transb, int m, int n,
                                         int k, float *alpha, const float *A,
                                         int lda, float *B, int ldb,
                                         float *beta, float *C, int ldc) {
  return lwblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

static inline lwblasStatus_t lwblasXgemm(lwblasHandle_t handle,
                                         lwblasOperation_t transa,
                                         lwblasOperation_t transb, int m, int n,
                                         int k, double *alpha, const double *A,
                                         int lda, double *B, int ldb,
                                         double *beta, double *C, int ldc) {
  return lwblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

static inline lwblasStatus_t lwblasXgemmBatched(
    lwblasHandle_t handle, lwblasOperation_t transa, lwblasOperation_t transb,
    int m, int n, int k, float *alpha, const float *Aarray[], int lda,
    const float *Barray[], int ldb, float *beta, float *Carray[], int ldc,
    int batchCount) {
#if LWDART_VERSION >= 4010
  return lwblasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
#else
  return LWBLAS_STATUS_SUCCESS;
#endif
}

static inline lwblasStatus_t lwblasXgemmBatched(
    lwblasHandle_t handle, lwblasOperation_t transa, lwblasOperation_t transb,
    int m, int n, int k, double *alpha, const double *Aarray[], int lda,
    const double *Barray[], int ldb, double *beta, double *Carray[], int ldc,
    int batchCount) {
#if LWDART_VERSION >= 4010
  return lwblasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
#else
  return LWBLAS_STATUS_SUCCESS;
#endif
}

//==============================================================================
// Primary Application code
//==============================================================================

static int processArgs(int argc, char *argv[], struct gemmOpts *opts) {
  int error = 0;
  int oldError;
  memset(opts, 0, sizeof(*opts));
  static char default_type[] = "d";  // default double
  opts->elem_type = default_type;
  opts->N = 10;

  while (argc) {
    oldError = error;

    if (*argv[0] == SWITCH_CHAR) {
      switch (*(argv[0] + 1)) {
        case 'm':
          opts->m = (int)atol(argv[0] + 2);
          break;

        case 'n':
          opts->n = (int)atol(argv[0] + 2);
          break;

        case 'k':
          opts->k = (int)atol(argv[0] + 2);
          break;

        case 'N':
          opts->N = (int)atol(argv[0] + 2);
          break;

        default:
          break;
      }
    }

    if (error > oldError) {
      fprintf(stderr, "Invalid switch '%c%s'\n", SWITCH_CHAR, argv[0] + 1);
    }

    argc -= 1;
    argv++;
  }

  return error;
}

template <typename T_ELEM>
static int TESTGEN(gemm)(const struct gemmOpts *opts, int matrixM, int matrixN,
                         int matrixK, int &numTests,
                         struct gemmTestParams<T_ELEM> *params) {
  static T_ELEM alpha[] = {lwGet<T_ELEM>(0, 0), lwGet<T_ELEM>(-1, -1),
                           lwGet<T_ELEM>(1, -2), lwGet<T_ELEM>(2, -1),
                           lwGet<T_ELEM>(0, -3)};
  static T_ELEM beta[] = {lwGet<T_ELEM>(0, 0), lwGet<T_ELEM>(-1, -1),
                          lwGet<T_ELEM>(1, -2), lwGet<T_ELEM>(2, -1),
                          lwGet<T_ELEM>(0, -3)};

#define NBR_ALPHAS (sizeof(alpha) / sizeof(alpha[0]))
#define NBR_BETAS (sizeof(beta) / sizeof(beta[0]))
  static T_ELEM theAlpha;
  static T_ELEM theBeta;
  static int state;
  static int m;
  static int n;
  static int k;

  if (numTests-- <= 0) {
    return -1;
  }

  theAlpha = alpha[lwRand() % NBR_ALPHAS];
  theBeta = beta[lwRand() % NBR_BETAS];
  params->transa = LWBLAS_OP_N;
  params->transb = LWBLAS_OP_N;
  m = matrixM;
  n = matrixN;
  k = matrixK;
  params->m = m;
  params->n = n;
  params->k = k;
  params->alpha = theAlpha;
  params->beta = theBeta;

  printf("#### args: ta=%d tb=%d m=%d n=%d k=%d ", (unsigned int)params->transa,
         (unsigned int)params->transb, params->m, params->n, params->k);
  printLwType(" alpha =", params->alpha);
  printLwType(" beta=", params->beta);
  printf("\n");

  m = lwRand() % matrixM;
  n = lwRand() % matrixN;
  k = lwRand() % matrixK;

  state = lwRand() % 9;
  return 0;
}

template <typename T_ELEM>
void fillupMatrixDebug(T_ELEM *A, int lda, int rows, int cols) {
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      A[i + lda * j] = lwGet<T_ELEM>(i + j);
    }
  }
}

template <typename T_ELEM>
int test_gemm_loop(struct gemmOpts &opts, float err, double max_relative_error,
                   lwblasHandle_t handle) {
  struct gemmTestParams<T_ELEM> params;
  lwdaStream_t *streamArray = 0;
  lwblasStatus_t status1, status2, status3;
  T_ELEM *A = NULL;
  T_ELEM *B = NULL;
  T_ELEM *C = NULL;
  T_ELEM **devPtrA = 0;
  T_ELEM **devPtrB = 0;
  T_ELEM **devPtrC = 0;
  T_ELEM **devPtrA_dev = NULL;
  T_ELEM **devPtrB_dev = NULL;
  T_ELEM **devPtrC_dev = NULL;
  int matrixM, matrixN, matrixK;
  int rowsA, rowsB, rowsC;
  int colsA, colsB, colsC;
  int matrixSizeA, matrixSizeB, matrixSizeC;
  int errors;
  double start, stop;

  printf("Testing %cgemm\n", *opts.elem_type);

  matrixM = (opts.m) ? opts.m : BENCH_MATRIX_M;
  matrixN = (opts.n) ? opts.n : BENCH_MATRIX_N;
  matrixK = (opts.k) ? opts.k : BENCH_MATRIX_K;

  rowsA = imax(1, matrixM);
  colsA = imax(1, matrixK);
  rowsB = imax(1, matrixK);
  colsB = imax(1, matrixN);
  rowsC = imax(1, matrixM);
  colsC = imax(1, matrixN);

  matrixSizeA = rowsA * colsA;
  matrixSizeB = rowsB * colsB;
  matrixSizeC = rowsC * colsC;

  devPtrA = (T_ELEM **)malloc(opts.N * sizeof(*devPtrA));
  devPtrB = (T_ELEM **)malloc(opts.N * sizeof(*devPtrB));
  devPtrC = (T_ELEM **)malloc(opts.N * sizeof(*devPtrC));

  for (int i = 0; i < opts.N; i++) {
    lwdaError_t err1 =
        lwdaMalloc((void **)&devPtrA[i], matrixSizeA * sizeof(devPtrA[0][0]));
    lwdaError_t err2 =
        lwdaMalloc((void **)&devPtrB[i], matrixSizeB * sizeof(devPtrB[0][0]));
    lwdaError_t err3 =
        lwdaMalloc((void **)&devPtrC[i], matrixSizeC * sizeof(devPtrC[0][0]));

    if ((err1 != lwdaSuccess) || (err2 != lwdaSuccess) ||
        (err3 != lwdaSuccess)) {
      CLEANUP();
      fprintf(stderr, "!!!! GPU memory allocation error\n");
      return LWBLASTEST_FAILED;
    }
  }

  // For batched processing we need those arrays on the device
  if (opts.test_method == tmBatched) {
    lwdaError_t err1 =
        lwdaMalloc((void **)&devPtrA_dev, opts.N * sizeof(*devPtrA));
    lwdaError_t err2 =
        lwdaMalloc((void **)&devPtrB_dev, opts.N * sizeof(*devPtrB));
    lwdaError_t err3 =
        lwdaMalloc((void **)&devPtrC_dev, opts.N * sizeof(*devPtrC));

    if ((err1 != lwdaSuccess) || (err2 != lwdaSuccess) ||
        (err3 != lwdaSuccess)) {
      CLEANUP();
      fprintf(stderr, "!!!! GPU memory allocation error\n");
      return LWBLASTEST_FAILED;
    }

    err1 = lwdaMemcpy(devPtrA_dev, devPtrA, opts.N * sizeof(*devPtrA),
                      lwdaMemcpyHostToDevice);
    err2 = lwdaMemcpy(devPtrB_dev, devPtrB, opts.N * sizeof(*devPtrB),
                      lwdaMemcpyHostToDevice);
    err3 = lwdaMemcpy(devPtrC_dev, devPtrC, opts.N * sizeof(*devPtrC),
                      lwdaMemcpyHostToDevice);

    if ((err1 != lwdaSuccess) || (err2 != lwdaSuccess) ||
        (err3 != lwdaSuccess)) {
      CLEANUP();
      fprintf(stderr, "!!!! cannot copy pointer array to device\n");
      return LWBLASTEST_FAILED;
    }
  }

  A = (T_ELEM *)malloc(matrixSizeA * sizeof(A[0]));
  B = (T_ELEM *)malloc(matrixSizeB * sizeof(B[0]));
  C = (T_ELEM *)malloc(matrixSizeC * sizeof(C[0]));

  if ((!A) || (!B) || (!C)) {
    CLEANUP();
    fprintf(stderr, "!!!! system memory allocation error\n");
    return LWBLASTEST_FAILED;
  }

  streamArray = (lwdaStream_t *)malloc(opts.N * sizeof(lwdaStream_t *));

  for (int i = 0; i < opts.N; i++) {
    if (opts.test_method == tmStream) {
      lwdaError_t lwdaErr = lwdaStreamCreate(&streamArray[i]);

      if (lwdaErr != lwdaSuccess) {
        CLEANUP();
        fprintf(stderr, "!!!! cannot create stream\n");
        return LWBLASTEST_FAILED;
      }
    } else {
      streamArray[i] = 0;
    }
  }

  errors = 0;
  int numTests = 1;

  while (TESTGEN(gemm)(&opts, matrixM, matrixN, matrixK, numTests, &params) ==
         0) {
    printf("#### args: lda=%d ldb=%d ldc=%d\n", rowsA, rowsB, rowsC);

    // fillup with Nan first (so lda padding is full on Nan)
    memset(A, 0xFF, matrixSizeA * sizeof(A[0]));
    fillupMatrixDebug(A, rowsA, params.m, params.k);
    memset(B, 0xFF, matrixSizeB * sizeof(B[0]));
    fillupMatrix(B, rowsB, params.k, params.n, 121);

    if (!lwEqual(params.beta, lwGet<T_ELEM>(0))) {
      fillupMatrix(C, rowsC, params.m, params.n);
    } else {
      /* fill with SNaNs to make sure ZGEMM doesn't access C */
      memset(C, 0xFF, matrixSizeC * sizeof(C[0]));
    }

    double flopsCoef = 2.0;

    for (int i = 0; i < opts.N; i++) {
      status1 = lwblasSetMatrix(rowsA, colsA, sizeof(A[0]), A, rowsA,
                                devPtrA[i], rowsA);
      status2 = lwblasSetMatrix(rowsB, colsB, sizeof(B[0]), B, rowsB,
                                devPtrB[i], rowsB);
      status3 = lwblasSetMatrix(rowsC, colsC, sizeof(C[0]), C, rowsC,
                                devPtrC[i], rowsC);

      if ((status1 != LWBLAS_STATUS_SUCCESS) || (status2 != status1) ||
          (status3 != status1)) {
        CLEANUP();
        fprintf(stderr, "!!!! GPU access error (write)\n");
        return LWBLASTEST_FAILED;
      }
    }

    start = second();

    if (opts.test_method == tmBatched) {
      lwblasSetStream(handle, streamArray[0]);
      status1 = lwblasXgemmBatched(handle, params.transa, params.transb,
                                   params.m, params.n, params.k, &params.alpha,
                                   (const T_ELEM **)devPtrA_dev, rowsA,
                                   (const T_ELEM **)devPtrB_dev, rowsB,
                                   &params.beta, devPtrC_dev, rowsC, opts.N);

      if (status1 != LWBLAS_STATUS_SUCCESS) {
        lwdaError_t lwdaStatus = lwdaGetLastError();
        CLEANUP();
        fprintf(stderr,
                "!!!! GPU program exelwtion error : lwblas Error=%d, lwca "
                "Error=%d,(%s)\n",
                status1, lwdaStatus, lwdaGetErrorString(lwdaStatus));
        return LWBLASTEST_FAILED;
      }
    } else {
      for (int i = 0; i < opts.N; i++) {
        lwblasSetStream(handle, streamArray[i]);
        status1 =
            lwblasXgemm(handle, params.transa, params.transb, params.m,
                        params.n, params.k, &params.alpha, devPtrA[i], rowsA,
                        devPtrB[i], rowsB, &params.beta, devPtrC[i], rowsC);

        if (status1 != LWBLAS_STATUS_SUCCESS) {
          lwdaError_t lwdaStatus = lwdaGetLastError();
          CLEANUP();
          fprintf(stderr,
                  "!!!! GPU program exelwtion error : lwblas Error=%d, lwca "
                  "Error=%d,(%s)\n",
                  status1, lwdaStatus, lwdaGetErrorString(lwdaStatus));
          return LWBLASTEST_FAILED;
        }
      }
    }

    lwdaError_t lwdaStatus = lwdaDeviceSynchronize();

    if (lwdaStatus != lwdaSuccess) {
      CLEANUP();
      fprintf(stderr,
              "!!!! GPU program exelwtion error on lwdaDeviceSynchronize : "
              "lwdaError=%d,(%s)\n",
              lwdaStatus, lwdaGetErrorString(lwdaStatus));
      return LWBLASTEST_FAILED;
    }

    stop = second();

    fprintf(stdout, "^^^^ elapsed = %10.8f sec  GFLOPS=%g\n", (stop - start),
            opts.N * (1e-9 * flopsCoef * params.m * params.n * params.k) /
                (stop - start));

  }  // end while (TESTGEN..

  CLEANUP();
  fprintf(stdout, "@@@@ %cgemm test %s\n", *opts.elem_type,
          errors ? "FAIL" : "OK");
  return LWBLASTEST_PASSED;
}

int main(int argc, char *argv[]) {
  struct gemmOpts opts;
  int errors, nTimes, nTotalErrors = 0;
  int status = LWBLASTEST_PASSED;

  printf("%s Starting...\n\n", sSDKname);

  int dev = findLwdaDevice(argc, (const char **)argv);

  if (dev == -1) {
    return LWBLASTEST_FAILED;
  }

  errors = processArgs(argc, argv, &opts);

  if (errors) {
    fprintf(stdout,
            "\n Usage: batchlwblas [-mSIZE_M] [-nSIZE_N] [-kSIZE_N] "
            "[-NSIZE_NUM_ITERATIONS] [-qatest] [-noprompt]\n");
    return LWBLASTEST_FAILED;
  }

  lwblasHandle_t handle;

  if (lwblasCreate(&handle) != LWBLAS_STATUS_SUCCESS) {
    fprintf(stdout, "LWBLAS initialization failed!\n");
    exit(EXIT_FAILURE);
  }

  // Run single kernels
  fprintf(stdout, "\n ==== Running single kernels ==== \n\n");
  nTimes = opts.N;
  opts.N = 1;
  *(opts.elem_type) = 's';
  status = test_gemm_loop<float>(opts, (float)LWBLAS_SGEMM_MAX_ULP_ERR,
                                 (double)LWBLAS_SGEMM_MAX_RELATIVE_ERR, handle);

  // Run Double version
  *(opts.elem_type) = 'd';

  if (getDeviceVersion() < DEV_VER_DBL_SUPPORT) {
    fprintf(stdout, "@@@@ dgemm test WAIVED due to lack of DP support\n");
    exit(EXIT_WAIVED);
  }

  status =
      test_gemm_loop<double>(opts, (float)LWBLAS_DGEMM_MAX_ULP_ERR,
                             (double)LWBLAS_DGEMM_MAX_RELATIVE_ERR, handle);
  nTotalErrors += (status == LWBLASTEST_PASSED ? 0 : 1);
  opts.N = nTimes;

  // Run with and without streams and then batched. The batched functions are a
  // feature new feature in 4.1
#if LWDART_VERSION >= 4010

  for (int ii = 0; ii < 3; ii++) {
#else

  for (int ii = 0; ii < 2; ii++) {
#endif

    switch (ii) {
      case 0:
        opts.test_method = tmRegular;
        fprintf(stdout, "\n ==== Running N=%d without streams ==== \n\n",
                opts.N);
        break;

      case 1:
        opts.test_method = tmStream;
        fprintf(stdout, "\n ==== Running N=%d with streams ==== \n\n", opts.N);
        break;

      case 2:
        opts.test_method = tmBatched;
        fprintf(stdout, "\n ==== Running N=%d batched ==== \n\n", opts.N);
        break;
    }

    // Run single version
    *(opts.elem_type) = 's';
    status =
        test_gemm_loop<float>(opts, (float)LWBLAS_SGEMM_MAX_ULP_ERR,
                              (double)LWBLAS_SGEMM_MAX_RELATIVE_ERR, handle);
    nTotalErrors += (status == LWBLASTEST_PASSED ? 0 : 1);

    // Run Double version
    *(opts.elem_type) = 'd';

    // Test doesn't meet minSpec, will will wave the DP test
    if (getDeviceVersion() < DEV_VER_DBL_SUPPORT) {
      fprintf(stdout, "@@@@ dgemm test WAIVED due to lack of DP support\n");
      exit(EXIT_WAIVED);
    } else {
      status =
          test_gemm_loop<double>(opts, (float)LWBLAS_DGEMM_MAX_ULP_ERR,
                                 (double)LWBLAS_DGEMM_MAX_RELATIVE_ERR, handle);
      nTotalErrors += (status == LWBLASTEST_PASSED ? 0 : 1);
    }
  }

  lwblasDestroy(handle);

  printf("\nTest Summary\n");
  printf("%d error(s)\n", nTotalErrors);
  exit(nTotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}
