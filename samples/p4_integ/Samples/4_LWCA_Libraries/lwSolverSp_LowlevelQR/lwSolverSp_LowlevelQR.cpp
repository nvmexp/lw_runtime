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
#include <ctype.h>
#include <lwda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lwsolverSp.h"
#include "lwsolverSp_LOWLEVEL_PREVIEW.h"
#include "helper_lwda.h"
#include "helper_lwsolver.h"

template <typename T_ELEM>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m,
                       int *n, int *nnz, T_ELEM **aVal, int **aRowInd,
                       int **aColInd, int extendSymMatrix);

void UsageSP(void) {
  printf("<options>\n");
  printf("-h          : display this help\n");
  printf("-file=<filename> : filename containing a matrix in MM format\n");
  printf("-device=<device_id> : <device_id> if want to run on specific GPU\n");

  exit(0);
}

void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts) {
  memset(&opts, 0, sizeof(opts));

  if (checkCmdLineFlag(argc, (const char **)argv, "-h")) {
    UsageSP();
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
    char *fileName = 0;
    getCmdLineArgumentString(argc, (const char **)argv, "file", &fileName);

    if (fileName) {
      opts.sparse_mat_filename = fileName;
    } else {
      printf("\nIncorrect filename passed to -file \n ");
      UsageSP();
    }
  }
}

int main(int argc, char *argv[]) {
  struct testOpts opts;
  lwsolverSpHandle_t lwsolverSpH =
      NULL;  // reordering, permutation and 1st LU factorization
  lwsparseHandle_t lwsparseH = NULL;  // residual evaluation
  lwdaStream_t stream = NULL;
  lwsparseMatDescr_t descrA = NULL;  // A is a base-0 general matrix

  csrqrInfoHost_t h_info =
      NULL;  // opaque info structure for LU with parital pivoting
  csrqrInfo_t d_info =
      NULL;  // opaque info structure for LU with parital pivoting

  int rowsA = 0;  // number of rows of A
  int colsA = 0;  // number of columns of A
  int nnzA = 0;   // number of nonzeros of A
  int baseA = 0;  // base index in CSR format

  // CSR(A) from I/O
  int *h_csrRowPtrA = NULL;  // <int> n+1
  int *h_csrColIndA = NULL;  // <int> nnzA
  double *h_csrValA = NULL;  // <double> nnzA

  double *h_x = NULL;      // <double> n,  x = A \ b
  double *h_b = NULL;      // <double> n, b = ones(m,1)
  double *h_bcopy = NULL;  // <double> n, b = ones(m,1)
  double *h_r = NULL;      // <double> n, r = b - A*x

  size_t size_internal = 0;
  size_t size_chol = 0;     // size of working space for csrlu
  void *buffer_cpu = NULL;  // working space for Cholesky
  void *buffer_gpu = NULL;  // working space for Cholesky

  int *d_csrRowPtrA = NULL;  // <int> n+1
  int *d_csrColIndA = NULL;  // <int> nnzA
  double *d_csrValA = NULL;  // <double> nnzA
  double *d_x = NULL;        // <double> n, x = A \ b
  double *d_b = NULL;        // <double> n, a copy of h_b
  double *d_r = NULL;        // <double> n, r = b - A*x

  // the constants used in residual evaluation, r = b - A*x
  const double minus_one = -1.0;
  const double one = 1.0;
  const double zero = 0.0;
  // the constant used in lwsolverSp
  // singularity is -1 if A is ilwertible under tol
  // tol determines the condition of singularity
  int singularity = 0;
  const double tol = 1.e-14;

  double x_inf = 0.0;  // |x|
  double r_inf = 0.0;  // |r|
  double A_inf = 0.0;  // |A|

  parseCommandLineArguments(argc, argv, opts);

  findLwdaDevice(argc, (const char **)argv);

  if (opts.sparse_mat_filename == NULL) {
    opts.sparse_mat_filename = sdkFindFilePath("lap2D_5pt_n32.mtx", argv[0]);
    if (opts.sparse_mat_filename != NULL)
      printf("Using default input file [%s]\n", opts.sparse_mat_filename);
    else
      printf("Could not find lap2D_5pt_n32.mtx\n");
  } else {
    printf("Using input file [%s]\n", opts.sparse_mat_filename);
  }

  printf("step 1: read matrix market format\n");

  if (opts.sparse_mat_filename) {
    if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true, &rowsA,
                                   &colsA, &nnzA, &h_csrValA, &h_csrRowPtrA,
                                   &h_csrColIndA, true)) {
      return 1;
    }
    baseA = h_csrRowPtrA[0];  // baseA = {0,1}
  } else {
    fprintf(stderr, "Error: input matrix is not provided\n");
    return 1;
  }

  if (rowsA != colsA) {
    fprintf(stderr, "Error: only support square matrix\n");
    return 1;
  }

  printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA,
         nnzA, baseA);

  checkLwdaErrors(lwsolverSpCreate(&lwsolverSpH));
  checkLwdaErrors(lwsparseCreate(&lwsparseH));
  checkLwdaErrors(lwdaStreamCreate(&stream));
  checkLwdaErrors(lwsolverSpSetStream(lwsolverSpH, stream));
  checkLwdaErrors(lwsparseSetStream(lwsparseH, stream));

  checkLwdaErrors(lwsparseCreateMatDescr(&descrA));

  checkLwdaErrors(lwsparseSetMatType(descrA, LWSPARSE_MATRIX_TYPE_GENERAL));

  if (baseA) {
    checkLwdaErrors(lwsparseSetMatIndexBase(descrA, LWSPARSE_INDEX_BASE_ONE));
  } else {
    checkLwdaErrors(lwsparseSetMatIndexBase(descrA, LWSPARSE_INDEX_BASE_ZERO));
  }

  h_x = (double *)malloc(sizeof(double) * colsA);
  h_b = (double *)malloc(sizeof(double) * rowsA);
  h_bcopy = (double *)malloc(sizeof(double) * rowsA);
  h_r = (double *)malloc(sizeof(double) * rowsA);

  assert(NULL != h_x);
  assert(NULL != h_b);
  assert(NULL != h_bcopy);
  assert(NULL != h_r);

  checkLwdaErrors(
      lwdaMalloc((void **)&d_csrRowPtrA, sizeof(int) * (rowsA + 1)));
  checkLwdaErrors(lwdaMalloc((void **)&d_csrColIndA, sizeof(int) * nnzA));
  checkLwdaErrors(lwdaMalloc((void **)&d_csrValA, sizeof(double) * nnzA));
  checkLwdaErrors(lwdaMalloc((void **)&d_x, sizeof(double) * colsA));
  checkLwdaErrors(lwdaMalloc((void **)&d_b, sizeof(double) * rowsA));
  checkLwdaErrors(lwdaMalloc((void **)&d_r, sizeof(double) * rowsA));

  for (int row = 0; row < rowsA; row++) {
    h_b[row] = 1.0;
  }

  memcpy(h_bcopy, h_b, sizeof(double) * rowsA);

  printf("step 2: create opaque info structure\n");
  checkLwdaErrors(lwsolverSpCreateCsrqrInfoHost(&h_info));

  printf("step 3: analyze qr(A) to know structure of L\n");
  checkLwdaErrors(lwsolverSpXcsrqrAnalysisHost(lwsolverSpH, rowsA, colsA, nnzA,
                                               descrA, h_csrRowPtrA,
                                               h_csrColIndA, h_info));

  printf("step 4: workspace for qr(A)\n");
  checkLwdaErrors(lwsolverSpDcsrqrBufferInfoHost(
      lwsolverSpH, rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA,
      h_csrColIndA, h_info, &size_internal, &size_chol));

  if (buffer_cpu) {
    free(buffer_cpu);
  }
  buffer_cpu = (void *)malloc(sizeof(char) * size_chol);
  assert(NULL != buffer_cpu);

  printf("step 5: compute A = L*L^T \n");
  checkLwdaErrors(lwsolverSpDcsrqrSetupHost(lwsolverSpH, rowsA, colsA, nnzA,
                                            descrA, h_csrValA, h_csrRowPtrA,
                                            h_csrColIndA, zero, h_info));

  checkLwdaErrors(lwsolverSpDcsrqrFactorHost(lwsolverSpH, rowsA, colsA, nnzA,
                                             NULL, NULL, h_info, buffer_cpu));

  printf("step 6: check if the matrix is singular \n");
  checkLwdaErrors(
      lwsolverSpDcsrqrZeroPivotHost(lwsolverSpH, h_info, tol, &singularity));

  if (0 <= singularity) {
    fprintf(stderr, "Error: A is not ilwertible, singularity=%d\n",
            singularity);
    return 1;
  }

  printf("step 7: solve A*x = b \n");
  checkLwdaErrors(lwsolverSpDcsrqrSolveHost(lwsolverSpH, rowsA, colsA, h_b, h_x,
                                            h_info, buffer_cpu));

  printf("step 8: evaluate residual r = b - A*x (result on CPU)\n");
  // use GPU gemv to compute r = b - A*x
  checkLwdaErrors(lwdaMemcpy(d_csrRowPtrA, h_csrRowPtrA,
                             sizeof(int) * (rowsA + 1),
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA,
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemcpy(d_csrValA, h_csrValA, sizeof(double) * nnzA,
                             lwdaMemcpyHostToDevice));

  checkLwdaErrors(
      lwdaMemcpy(d_r, h_bcopy, sizeof(double) * rowsA, lwdaMemcpyHostToDevice));
  checkLwdaErrors(
      lwdaMemcpy(d_x, h_x, sizeof(double) * colsA, lwdaMemcpyHostToDevice));

  /* Wrap raw data into lwSPARSE generic API objects */
  lwsparseSpMatDescr_t matA = NULL;
  if (baseA) {
    checkLwdaErrors(lwsparseCreateCsr(&matA, rowsA, colsA, nnzA, d_csrRowPtrA,
                                      d_csrColIndA, d_csrValA,
                                      LWSPARSE_INDEX_32I, LWSPARSE_INDEX_32I,
                                      LWSPARSE_INDEX_BASE_ONE, LWDA_R_64F));
  } else {
    checkLwdaErrors(lwsparseCreateCsr(&matA, rowsA, colsA, nnzA, d_csrRowPtrA,
                                      d_csrColIndA, d_csrValA,
                                      LWSPARSE_INDEX_32I, LWSPARSE_INDEX_32I,
                                      LWSPARSE_INDEX_BASE_ZERO, LWDA_R_64F));
  }

  lwsparseDlwecDescr_t vecx = NULL;
  checkLwdaErrors(lwsparseCreateDlwec(&vecx, colsA, d_x, LWDA_R_64F));
  lwsparseDlwecDescr_t vecAx = NULL;
  checkLwdaErrors(lwsparseCreateDlwec(&vecAx, rowsA, d_r, LWDA_R_64F));

  /* Allocate workspace for lwSPARSE */
  size_t bufferSize = 0;
  checkLwdaErrors(lwsparseSpMV_bufferSize(
      lwsparseH, LWSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx, &one,
      vecAx, LWDA_R_64F, LWSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  void *buffer = NULL;
  checkLwdaErrors(lwdaMalloc(&buffer, bufferSize));

  checkLwdaErrors(lwsparseSpMV(lwsparseH, LWSPARSE_OPERATION_NON_TRANSPOSE,
                               &minus_one, matA, vecx, &one, vecAx, LWDA_R_64F,
                               LWSPARSE_SPMV_ALG_DEFAULT, buffer));

  checkLwdaErrors(
      lwdaMemcpy(h_r, d_r, sizeof(double) * rowsA, lwdaMemcpyDeviceToHost));

  x_inf = vec_norminf(colsA, h_x);
  r_inf = vec_norminf(rowsA, h_r);
  A_inf = csr_mat_norminf(rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA,
                          h_csrColIndA);

  printf("(CPU) |b - A*x| = %E \n", r_inf);
  printf("(CPU) |A| = %E \n", A_inf);
  printf("(CPU) |x| = %E \n", x_inf);
  printf("(CPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf / (A_inf * x_inf));

  printf("step 9: create opaque info structure\n");
  checkLwdaErrors(lwsolverSpCreateCsrqrInfo(&d_info));

  checkLwdaErrors(lwdaMemcpy(d_csrRowPtrA, h_csrRowPtrA,
                             sizeof(int) * (rowsA + 1),
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA,
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemcpy(d_csrValA, h_csrValA, sizeof(double) * nnzA,
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(
      lwdaMemcpy(d_b, h_bcopy, sizeof(double) * rowsA, lwdaMemcpyHostToDevice));

  printf("step 10: analyze qr(A) to know structure of L\n");
  checkLwdaErrors(lwsolverSpXcsrqrAnalysis(lwsolverSpH, rowsA, colsA, nnzA,
                                           descrA, d_csrRowPtrA, d_csrColIndA,
                                           d_info));

  printf("step 11: workspace for qr(A)\n");
  checkLwdaErrors(lwsolverSpDcsrqrBufferInfo(
      lwsolverSpH, rowsA, colsA, nnzA, descrA, d_csrValA, d_csrRowPtrA,
      d_csrColIndA, d_info, &size_internal, &size_chol));

  printf("GPU buffer size = %lld bytes\n", (signed long long)size_chol);
  if (buffer_gpu) {
    checkLwdaErrors(lwdaFree(buffer_gpu));
  }
  checkLwdaErrors(lwdaMalloc(&buffer_gpu, sizeof(char) * size_chol));

  printf("step 12: compute A = L*L^T \n");
  checkLwdaErrors(lwsolverSpDcsrqrSetup(lwsolverSpH, rowsA, colsA, nnzA, descrA,
                                        d_csrValA, d_csrRowPtrA, d_csrColIndA,
                                        zero, d_info));

  checkLwdaErrors(lwsolverSpDcsrqrFactor(lwsolverSpH, rowsA, colsA, nnzA, NULL,
                                         NULL, d_info, buffer_gpu));

  printf("step 13: check if the matrix is singular \n");
  checkLwdaErrors(
      lwsolverSpDcsrqrZeroPivot(lwsolverSpH, d_info, tol, &singularity));

  if (0 <= singularity) {
    fprintf(stderr, "Error: A is not ilwertible, singularity=%d\n",
            singularity);
    return 1;
  }

  printf("step 14: solve A*x = b \n");
  checkLwdaErrors(lwsolverSpDcsrqrSolve(lwsolverSpH, rowsA, colsA, d_b, d_x,
                                        d_info, buffer_gpu));

  checkLwdaErrors(
      lwdaMemcpy(d_r, h_bcopy, sizeof(double) * rowsA, lwdaMemcpyHostToDevice));

  checkLwdaErrors(lwsparseSpMV(lwsparseH, LWSPARSE_OPERATION_NON_TRANSPOSE,
                               &minus_one, matA, vecx, &one, vecAx, LWDA_R_64F,
                               LWSPARSE_SPMV_ALG_DEFAULT, buffer));

  checkLwdaErrors(
      lwdaMemcpy(h_r, d_r, sizeof(double) * rowsA, lwdaMemcpyDeviceToHost));

  r_inf = vec_norminf(rowsA, h_r);

  printf("(GPU) |b - A*x| = %E \n", r_inf);
  printf("(GPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf / (A_inf * x_inf));

  if (lwsolverSpH) {
    checkLwdaErrors(lwsolverSpDestroy(lwsolverSpH));
  }
  if (lwsparseH) {
    checkLwdaErrors(lwsparseDestroy(lwsparseH));
  }
  if (stream) {
    checkLwdaErrors(lwdaStreamDestroy(stream));
  }
  if (descrA) {
    checkLwdaErrors(lwsparseDestroyMatDescr(descrA));
  }
  if (h_info) {
    checkLwdaErrors(lwsolverSpDestroyCsrqrInfoHost(h_info));
  }
  if (d_info) {
    checkLwdaErrors(lwsolverSpDestroyCsrqrInfo(d_info));
  }

  if (matA) {
    checkLwdaErrors(lwsparseDestroySpMat(matA));
  }
  if (vecx) {
    checkLwdaErrors(lwsparseDestroyDlwec(vecx));
  }
  if (vecAx) {
    checkLwdaErrors(lwsparseDestroyDlwec(vecAx));
  }

  if (h_csrValA) {
    free(h_csrValA);
  }
  if (h_csrRowPtrA) {
    free(h_csrRowPtrA);
  }
  if (h_csrColIndA) {
    free(h_csrColIndA);
  }

  if (h_x) {
    free(h_x);
  }
  if (h_b) {
    free(h_b);
  }
  if (h_bcopy) {
    free(h_bcopy);
  }
  if (h_r) {
    free(h_r);
  }

  if (buffer_cpu) {
    free(buffer_cpu);
  }
  if (buffer_gpu) {
    checkLwdaErrors(lwdaFree(buffer_gpu));
  }

  if (d_csrValA) {
    checkLwdaErrors(lwdaFree(d_csrValA));
  }
  if (d_csrRowPtrA) {
    checkLwdaErrors(lwdaFree(d_csrRowPtrA));
  }
  if (d_csrColIndA) {
    checkLwdaErrors(lwdaFree(d_csrColIndA));
  }
  if (d_x) {
    checkLwdaErrors(lwdaFree(d_x));
  }
  if (d_b) {
    checkLwdaErrors(lwdaFree(d_b));
  }
  if (d_r) {
    checkLwdaErrors(lwdaFree(d_r));
  }

  return 0;
}
