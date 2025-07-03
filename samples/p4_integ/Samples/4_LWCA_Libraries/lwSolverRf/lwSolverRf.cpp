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
 *  A framework of refactorization process.
 *
 *  step 1: compute P*A*Q = L*U by
 *    - reordering and
 *    - LU with partial pivoting in lwsolverSp
 *
 *  step 2: set up lwsolverRf by (P, Q, L, U)
 *
 *  step 3: analyze and refactor A
 *
 *  How to use
 *     ./lwSolverRf -P=symrcm -file <file>
 *     ./lwSolverRf -P=symamd -file <file>
 *
 */

#include "lwsolverRf.h"

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
#include "helper_string.h"

template <typename T_ELEM>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m,
                       int *n, int *nnz, T_ELEM **aVal, int **aRowInd,
                       int **aColInd, int extendSymMatrix);

void UsageRF(void) {
  printf("<options>\n");
  printf("-h          : display this help\n");
  printf("-P=<name>    : choose a reordering\n");
  printf("              symrcm (Reverse Lwthill-McKee)\n");
  printf("              symamd (Approximate Minimum Degree)\n");
  printf("-file=<filename> : filename containing a matrix in MM format\n");
  printf("-device=<device_id> : <device_id> if want to run on specific GPU\n");

  exit(0);
}

void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts) {
  memset(&opts, 0, sizeof(opts));

  if (checkCmdLineFlag(argc, (const char **)argv, "-h")) {
    UsageRF();
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "P")) {
    char *reorderType = NULL;
    getCmdLineArgumentString(argc, (const char **)argv, "P", &reorderType);

    if (reorderType) {
      if ((STRCASECMP(reorderType, "symrcm") != 0) &&
          (STRCASECMP(reorderType, "symamd") != 0)) {
        printf("\nIncorrect argument passed to -P option\n");
        UsageRF();
      } else {
        opts.reorder = reorderType;
      }
    }
  }

  if (!opts.reorder) {
    opts.reorder = "symrcm";  // Setting default reordering to be symrcm.
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
    char *fileName = 0;
    getCmdLineArgumentString(argc, (const char **)argv, "file", &fileName);

    if (fileName) {
      opts.sparse_mat_filename = fileName;
    } else {
      printf("\nIncorrect filename passed to -file \n ");
      UsageRF();
    }
  }
}

int main(int argc, char *argv[]) {
  struct testOpts opts;
  lwsolverRfHandle_t lwsolverRfH = NULL;  // refactorization
  lwsolverSpHandle_t lwsolverSpH =
      NULL;  // reordering, permutation and 1st LU factorization
  lwsparseHandle_t lwsparseH = NULL;  // residual evaluation
  lwdaStream_t stream = NULL;
  lwsparseMatDescr_t descrA = NULL;  // A is a base-0 general matrix

  csrluInfoHost_t info =
      NULL;  // opaque info structure for LU with parital pivoting

  int rowsA = 0;  // number of rows of A
  int colsA = 0;  // number of columns of A
  int nnzA = 0;   // number of nonzeros of A
  int baseA = 0;  // base index in CSR format
                  // lwsolverRf only works for base-0

  // lwsolverRf only works for square matrix,
  // assume n = rowsA = colsA

  // CSR(A) from I/O
  int *h_csrRowPtrA = NULL;  // <int> n+1
  int *h_csrColIndA = NULL;  // <int> nnzA
  double *h_csrValA = NULL;  // <double> nnzA

  int *h_Qreorder = NULL;  // <int> n
                           // reorder to reduce zero fill-in
                           // Qreorder = symrcm(A) or Qreroder = symamd(A)
  // B = Q*A*Q^T
  int *h_csrRowPtrB = NULL;  // <int> n+1
  int *h_csrColIndB = NULL;  // <int> nnzA
  double *h_csrValB = NULL;  // <double> nnzA
  int *h_mapBfromA = NULL;   // <int> nnzA

  double *h_x = NULL;  // <double> n,  x = A \ b
  double *h_b = NULL;  // <double> n, b = ones(m,1)
  double *h_r = NULL;  // <double> n, r = b - A*x

  // solve B*(Qx) = Q*b
  double *h_xhat = NULL;  // <double> n, Q*x_hat = x
  double *h_bhat = NULL;  // <double> n, b_hat = Q*b

  size_t size_perm = 0;
  size_t size_internal = 0;
  size_t size_lu = 0;       // size of working space for csrlu
  void *buffer_cpu = NULL;  // working space for
                            // - permutation: B = Q*A*Q^T
                            // - LU with partial pivoting in lwsolverSp

  // lwsolverSp computes LU with partial pivoting
  //     Plu*B*Qlu^T = L*U
  //   where B = Q*A*Q^T
  //
  // nnzL and nnzU are not known until factorization is done.
  // However upper bound of L+U is known after symbolic analysis of LU.
  int *h_Plu = NULL;  // <int> n
  int *h_Qlu = NULL;  // <int> n

  int nnzL = 0;
  int *h_csrRowPtrL = NULL;  // <int> n+1
  int *h_csrColIndL = NULL;  // <int> nnzL
  double *h_csrValL = NULL;  // <double> nnzL

  int nnzU = 0;
  int *h_csrRowPtrU = NULL;  // <int> n+1
  int *h_csrColIndU = NULL;  // <int> nnzU
  double *h_csrValU = NULL;  // <double> nnzU

  int *h_P = NULL;  // <int> n, P = Plu * Qreorder
  int *h_Q = NULL;  // <int> n, Q = Qlu * Qreorder

  int *d_csrRowPtrA = NULL;  // <int> n+1
  int *d_csrColIndA = NULL;  // <int> nnzA
  double *d_csrValA = NULL;  // <double> nnzA
  double *d_x = NULL;        // <double> n, x = A \ b
  double *d_b = NULL;        // <double> n, a copy of h_b
  double *d_r = NULL;        // <double> n, r = b - A*x

  int *d_P = NULL;  // <int> n, P*A*Q^T = L*U
  int *d_Q = NULL;  // <int> n

  double *d_T = NULL;  // working space in lwsolverRfSolve
                       // |d_T| = n * nrhs

  // the constants used in residual evaluation, r = b - A*x
  const double minus_one = -1.0;
  const double one = 1.0;
  // the constants used in lwsolverRf
  // nzero is the value below which zero pivot is flagged.
  // nboost is the value which is substitured for zero pivot.
  double nzero = 0.0;
  double nboost = 0.0;
  // the constant used in lwsolverSp
  // singularity is -1 if A is ilwertible under tol
  // tol determines the condition of singularity
  // pivot_threshold decides pivoting strategy
  int singularity = 0;
  const double tol = 1.e-14;
  const double pivot_threshold = 1.0;
  // the constants used in lwsolverRf
  const lwsolverRfFactorization_t fact_alg =
      LWSOLVERRF_FACTORIZATION_ALG0;  // default
  const lwsolverRfTriangularSolve_t solve_alg =
      LWSOLVERRF_TRIANGULAR_SOLVE_ALG1;  // default

  double x_inf = 0.0;  // |x|
  double r_inf = 0.0;  // |r|
  double A_inf = 0.0;  // |A|
  int errors = 0;

  double start, stop;
  double time_reorder;
  double time_perm;
  double time_sp_analysis;
  double time_sp_factor;
  double time_sp_solve;
  double time_sp_extract;
  double time_rf_assemble;
  double time_rf_reset;
  double time_rf_refactor;
  double time_rf_solve;

  parseCommandLineArguments(argc, argv, opts);

  printf("step 1.1: preparation\n");
  printf("step 1.1: read matrix market format\n");

  findLwdaDevice(argc, (const char **)argv);

  if (opts.sparse_mat_filename == NULL) {
    opts.sparse_mat_filename = sdkFindFilePath("lap2D_5pt_n100.mtx", argv[0]);
    printf("Using default input file [%s]\n", opts.sparse_mat_filename);
  } else {
    printf("Using input file [%s]\n", opts.sparse_mat_filename);
  }

  if (opts.sparse_mat_filename) {
    if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true, &rowsA,
                                   &colsA, &nnzA, &h_csrValA, &h_csrRowPtrA,
                                   &h_csrColIndA, true)) {
      return 1;
    }
    baseA = h_csrRowPtrA[0];  // baseA = {0,1}
  }

  if (rowsA != colsA) {
    fprintf(stderr, "Error: only support square matrix\n");
    return 1;
  }

  printf("WARNING: lwsolverRf only works for base-0 \n");
  if (baseA) {
    for (int i = 0; i <= rowsA; i++) {
      h_csrRowPtrA[i]--;
    }
    for (int i = 0; i < nnzA; i++) {
      h_csrColIndA[i]--;
    }
    baseA = 0;
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

  h_Qreorder = (int *)malloc(sizeof(int) * colsA);

  h_csrRowPtrB = (int *)malloc(sizeof(int) * (rowsA + 1));
  h_csrColIndB = (int *)malloc(sizeof(int) * nnzA);
  h_csrValB = (double *)malloc(sizeof(double) * nnzA);
  h_mapBfromA = (int *)malloc(sizeof(int) * nnzA);

  h_x = (double *)malloc(sizeof(double) * colsA);
  h_b = (double *)malloc(sizeof(double) * rowsA);
  h_r = (double *)malloc(sizeof(double) * rowsA);
  h_xhat = (double *)malloc(sizeof(double) * colsA);
  h_bhat = (double *)malloc(sizeof(double) * rowsA);

  assert(NULL != h_Qreorder);

  assert(NULL != h_csrRowPtrB);
  assert(NULL != h_csrColIndB);
  assert(NULL != h_csrValB);
  assert(NULL != h_mapBfromA);

  assert(NULL != h_x);
  assert(NULL != h_b);
  assert(NULL != h_r);
  assert(NULL != h_xhat);
  assert(NULL != h_bhat);

  checkLwdaErrors(
      lwdaMalloc((void **)&d_csrRowPtrA, sizeof(int) * (rowsA + 1)));
  checkLwdaErrors(lwdaMalloc((void **)&d_csrColIndA, sizeof(int) * nnzA));
  checkLwdaErrors(lwdaMalloc((void **)&d_csrValA, sizeof(double) * nnzA));
  checkLwdaErrors(lwdaMalloc((void **)&d_x, sizeof(double) * colsA));
  checkLwdaErrors(lwdaMalloc((void **)&d_b, sizeof(double) * rowsA));
  checkLwdaErrors(lwdaMalloc((void **)&d_r, sizeof(double) * rowsA));
  checkLwdaErrors(lwdaMalloc((void **)&d_P, sizeof(int) * rowsA));
  checkLwdaErrors(lwdaMalloc((void **)&d_Q, sizeof(int) * colsA));
  checkLwdaErrors(lwdaMalloc((void **)&d_T, sizeof(double) * rowsA * 1));

  printf("step 1.2: set right hand side vector (b) to 1\n");
  for (int row = 0; row < rowsA; row++) {
    h_b[row] = 1.0;
  }

  printf("step 2: reorder the matrix to reduce zero fill-in\n");
  printf("        Q = symrcm(A) or Q = symamd(A) \n");
  start = second();
  start = second();

  if (0 == strcmp(opts.reorder, "symrcm")) {
    checkLwdaErrors(lwsolverSpXcsrsymrcmHost(lwsolverSpH, rowsA, nnzA, descrA,
                                             h_csrRowPtrA, h_csrColIndA,
                                             h_Qreorder));
  } else if (0 == strcmp(opts.reorder, "symamd")) {
    checkLwdaErrors(lwsolverSpXcsrsymamdHost(lwsolverSpH, rowsA, nnzA, descrA,
                                             h_csrRowPtrA, h_csrColIndA,
                                             h_Qreorder));
  } else {
    fprintf(stderr, "Error: %s is unknow reordering\n", opts.reorder);
    return 1;
  }

  stop = second();
  time_reorder = stop - start;

  printf("step 3: B = Q*A*Q^T\n");
  memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int) * (rowsA + 1));
  memcpy(h_csrColIndB, h_csrColIndA, sizeof(int) * nnzA);

  start = second();
  start = second();

  checkLwdaErrors(lwsolverSpXcsrperm_bufferSizeHost(
      lwsolverSpH, rowsA, colsA, nnzA, descrA, h_csrRowPtrB, h_csrColIndB,
      h_Qreorder, h_Qreorder, &size_perm));

  if (buffer_cpu) {
    free(buffer_cpu);
  }
  buffer_cpu = (void *)malloc(sizeof(char) * size_perm);
  assert(NULL != buffer_cpu);

  // h_mapBfromA = Identity
  for (int j = 0; j < nnzA; j++) {
    h_mapBfromA[j] = j;
  }
  checkLwdaErrors(lwsolverSpXcsrpermHost(
      lwsolverSpH, rowsA, colsA, nnzA, descrA, h_csrRowPtrB, h_csrColIndB,
      h_Qreorder, h_Qreorder, h_mapBfromA, buffer_cpu));

  // B = A( mapBfromA )
  for (int j = 0; j < nnzA; j++) {
    h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
  }

  stop = second();
  time_perm = stop - start;

  printf("step 4: solve A*x = b by LU(B) in lwsolverSp\n");

  printf("step 4.1: create opaque info structure\n");
  checkLwdaErrors(lwsolverSpCreateCsrluInfoHost(&info));

  printf(
      "step 4.2: analyze LU(B) to know structure of Q and R, and upper bound "
      "for nnz(L+U)\n");
  start = second();
  start = second();

  checkLwdaErrors(lwsolverSpXcsrluAnalysisHost(
      lwsolverSpH, rowsA, nnzA, descrA, h_csrRowPtrB, h_csrColIndB, info));

  stop = second();
  time_sp_analysis = stop - start;

  printf("step 4.3: workspace for LU(B)\n");
  checkLwdaErrors(lwsolverSpDcsrluBufferInfoHost(
      lwsolverSpH, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
      info, &size_internal, &size_lu));

  if (buffer_cpu) {
    free(buffer_cpu);
  }
  buffer_cpu = (void *)malloc(sizeof(char) * size_lu);
  assert(NULL != buffer_cpu);

  printf("step 4.4: compute Ppivot*B = L*U \n");
  start = second();
  start = second();

  checkLwdaErrors(lwsolverSpDcsrluFactorHost(
      lwsolverSpH, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
      info, pivot_threshold, buffer_cpu));

  stop = second();
  time_sp_factor = stop - start;

  // TODO: check singularity by tol
  printf("step 4.5: check if the matrix is singular \n");
  checkLwdaErrors(
      lwsolverSpDcsrluZeroPivotHost(lwsolverSpH, info, tol, &singularity));

  if (0 <= singularity) {
    fprintf(stderr, "Error: A is not ilwertible, singularity=%d\n",
            singularity);
    return 1;
  }

  printf("step 4.6: solve A*x = b \n");
  printf("    i.e.  solve B*(Qx) = Q*b \n");
  start = second();
  start = second();

  // b_hat = Q*b
  for (int j = 0; j < rowsA; j++) {
    h_bhat[j] = h_b[h_Qreorder[j]];
  }
  // B*x_hat = b_hat
  checkLwdaErrors(lwsolverSpDcsrluSolveHost(lwsolverSpH, rowsA, h_bhat, h_xhat,
                                            info, buffer_cpu));

  // x = Q^T * x_hat
  for (int j = 0; j < rowsA; j++) {
    h_x[h_Qreorder[j]] = h_xhat[j];
  }

  stop = second();
  time_sp_solve = stop - start;

  printf("step 4.7: evaluate residual r = b - A*x (result on CPU)\n");
  // use GPU gemv to compute r = b - A*x
  checkLwdaErrors(lwdaMemcpy(d_csrRowPtrA, h_csrRowPtrA,
                             sizeof(int) * (rowsA + 1),
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA,
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemcpy(d_csrValA, h_csrValA, sizeof(double) * nnzA,
                             lwdaMemcpyHostToDevice));

  checkLwdaErrors(
      lwdaMemcpy(d_r, h_b, sizeof(double) * rowsA, lwdaMemcpyHostToDevice));
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

  printf("step 5: extract P, Q, L and U from P*B*Q^T = L*U \n");
  printf("        L has implicit unit diagonal\n");
  start = second();
  start = second();

  checkLwdaErrors(lwsolverSpXcsrluNnzHost(lwsolverSpH, &nnzL, &nnzU, info));

  h_Plu = (int *)malloc(sizeof(int) * rowsA);
  h_Qlu = (int *)malloc(sizeof(int) * colsA);

  h_csrValL = (double *)malloc(sizeof(double) * nnzL);
  h_csrRowPtrL = (int *)malloc(sizeof(int) * (rowsA + 1));
  h_csrColIndL = (int *)malloc(sizeof(int) * nnzL);

  h_csrValU = (double *)malloc(sizeof(double) * nnzU);
  h_csrRowPtrU = (int *)malloc(sizeof(int) * (rowsA + 1));
  h_csrColIndU = (int *)malloc(sizeof(int) * nnzU);

  assert(NULL != h_Plu);
  assert(NULL != h_Qlu);

  assert(NULL != h_csrValL);
  assert(NULL != h_csrRowPtrL);
  assert(NULL != h_csrColIndL);

  assert(NULL != h_csrValU);
  assert(NULL != h_csrRowPtrU);
  assert(NULL != h_csrColIndU);

  checkLwdaErrors(lwsolverSpDcsrluExtractHost(
      lwsolverSpH, h_Plu, h_Qlu, descrA, h_csrValL, h_csrRowPtrL, h_csrColIndL,
      descrA, h_csrValU, h_csrRowPtrU, h_csrColIndU, info, buffer_cpu));

  stop = second();
  time_sp_extract = stop - start;

  printf("nnzL = %d, nnzU = %d\n", nnzL, nnzU);

  /*  B = Qreorder*A*Qreorder^T
   *  Plu*B*Qlu^T = L*U
   *
   *  (Plu*Qreorder)*A*(Qlu*Qreorder)^T = L*U
   *
   *  Let P = Plu*Qreroder, Q = Qlu*Qreorder,
   *  then we have
   *      P*A*Q^T = L*U
   *  which is the fundamental relation in lwsolverRf.
   */
  printf("step 6: form P*A*Q^T = L*U\n");

  h_P = (int *)malloc(sizeof(int) * rowsA);
  h_Q = (int *)malloc(sizeof(int) * colsA);
  assert(NULL != h_P);
  assert(NULL != h_Q);

  printf("step 6.1: P = Plu*Qreroder\n");
  // gather operation, P = Qreorder(Plu)
  for (int j = 0; j < rowsA; j++) {
    h_P[j] = h_Qreorder[h_Plu[j]];
  }

  printf("step 6.2: Q = Qlu*Qreorder \n");
  // gather operation, Q = Qreorder(Qlu)
  for (int j = 0; j < colsA; j++) {
    h_Q[j] = h_Qreorder[h_Qlu[j]];
  }

  printf("step 7: create lwsolverRf handle\n");
  checkLwdaErrors(lwsolverRfCreate(&lwsolverRfH));

  printf("step 8: set parameters for lwsolverRf \n");
  // numerical values for checking "zeros" and for boosting.
  checkLwdaErrors(lwsolverRfSetNumericProperties(lwsolverRfH, nzero, nboost));

  // choose algorithm for refactorization and solve
  checkLwdaErrors(lwsolverRfSetAlgs(lwsolverRfH, fact_alg, solve_alg));

  // matrix mode: L and U are CSR format, and L has implicit unit diagonal
  checkLwdaErrors(
      lwsolverRfSetMatrixFormat(lwsolverRfH, LWSOLVERRF_MATRIX_FORMAT_CSR,
                                LWSOLVERRF_UNIT_DIAGONAL_ASSUMED_L));

  // fast mode for matrix assembling
  checkLwdaErrors(lwsolverRfSetResetValuesFastMode(
      lwsolverRfH, LWSOLVERRF_RESET_VALUES_FAST_MODE_ON));

  printf("step 9: assemble P*A*Q = L*U \n");
  start = second();
  start = second();

  checkLwdaErrors(lwsolverRfSetupHost(
      rowsA, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL, h_csrRowPtrL,
      h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P,
      h_Q, lwsolverRfH));

  checkLwdaErrors(lwdaDeviceSynchronize());
  stop = second();
  time_rf_assemble = stop - start;

  printf("step 10: analyze to extract parallelism \n");
  checkLwdaErrors(lwsolverRfAnalyze(lwsolverRfH));

  printf("step 11: import A to lwsolverRf \n");
  checkLwdaErrors(lwdaMemcpy(d_csrRowPtrA, h_csrRowPtrA,
                             sizeof(int) * (rowsA + 1),
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA,
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(lwdaMemcpy(d_csrValA, h_csrValA, sizeof(double) * nnzA,
                             lwdaMemcpyHostToDevice));
  checkLwdaErrors(
      lwdaMemcpy(d_P, h_P, sizeof(int) * rowsA, lwdaMemcpyHostToDevice));
  checkLwdaErrors(
      lwdaMemcpy(d_Q, h_Q, sizeof(int) * colsA, lwdaMemcpyHostToDevice));

  start = second();
  start = second();

  checkLwdaErrors(lwsolverRfResetValues(rowsA, nnzA, d_csrRowPtrA, d_csrColIndA,
                                        d_csrValA, d_P, d_Q, lwsolverRfH));

  checkLwdaErrors(lwdaDeviceSynchronize());
  stop = second();
  time_rf_reset = stop - start;

  printf("step 12: refactorization \n");
  start = second();
  start = second();

  checkLwdaErrors(lwsolverRfRefactor(lwsolverRfH));

  checkLwdaErrors(lwdaDeviceSynchronize());
  stop = second();
  time_rf_refactor = stop - start;

  printf("step 13: solve A*x = b \n");
  checkLwdaErrors(
      lwdaMemcpy(d_x, h_b, sizeof(double) * rowsA, lwdaMemcpyHostToDevice));

  start = second();
  start = second();

  checkLwdaErrors(
      lwsolverRfSolve(lwsolverRfH, d_P, d_Q, 1, d_T, rowsA, d_x, rowsA));

  checkLwdaErrors(lwdaDeviceSynchronize());
  stop = second();
  time_rf_solve = stop - start;

  printf("step 14: evaluate residual r = b - A*x (result on GPU)\n");
  checkLwdaErrors(
      lwdaMemcpy(d_r, h_b, sizeof(double) * rowsA, lwdaMemcpyHostToDevice));

  checkLwdaErrors(lwsparseSpMV(lwsparseH, LWSPARSE_OPERATION_NON_TRANSPOSE,
                               &minus_one, matA, vecx, &one, vecAx, LWDA_R_64F,
                               LWSPARSE_SPMV_ALG_DEFAULT, buffer));

  checkLwdaErrors(
      lwdaMemcpy(h_x, d_x, sizeof(double) * colsA, lwdaMemcpyDeviceToHost));
  checkLwdaErrors(
      lwdaMemcpy(h_r, d_r, sizeof(double) * rowsA, lwdaMemcpyDeviceToHost));

  x_inf = vec_norminf(colsA, h_x);
  r_inf = vec_norminf(rowsA, h_r);
  printf("(GPU) |b - A*x| = %E \n", r_inf);
  printf("(GPU) |A| = %E \n", A_inf);
  printf("(GPU) |x| = %E \n", x_inf);
  printf("(GPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf / (A_inf * x_inf));

  printf("===== statistics \n");
  printf(" nnz(A) = %d, nnz(L+U) = %d, zero fill-in ratio = %f\n", nnzA,
         nnzL + nnzU, ((double)(nnzL + nnzU)) / (double)nnzA);
  printf("\n");
  printf("===== timing profile \n");
  printf(" reorder A   : %f sec\n", time_reorder);
  printf(" B = Q*A*Q^T : %f sec\n", time_perm);
  printf("\n");
  printf(" lwsolverSp LU analysis: %f sec\n", time_sp_analysis);
  printf(" lwsolverSp LU factor  : %f sec\n", time_sp_factor);
  printf(" lwsolverSp LU solve   : %f sec\n", time_sp_solve);
  printf(" lwsolverSp LU extract : %f sec\n", time_sp_extract);
  printf("\n");
  printf(" lwsolverRf assemble : %f sec\n", time_rf_assemble);
  printf(" lwsolverRf reset    : %f sec\n", time_rf_reset);
  printf(" lwsolverRf refactor : %f sec\n", time_rf_refactor);
  printf(" lwsolverRf solve    : %f sec\n", time_rf_solve);

  if (lwsolverRfH) {
    checkLwdaErrors(lwsolverRfDestroy(lwsolverRfH));
  }
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
  if (info) {
    checkLwdaErrors(lwsolverSpDestroyCsrluInfoHost(info));
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

  if (h_Qreorder) {
    free(h_Qreorder);
  }

  if (h_csrRowPtrB) {
    free(h_csrRowPtrB);
  }
  if (h_csrColIndB) {
    free(h_csrColIndB);
  }
  if (h_csrValB) {
    free(h_csrValB);
  }
  if (h_mapBfromA) {
    free(h_mapBfromA);
  }

  if (h_x) {
    free(h_x);
  }
  if (h_b) {
    free(h_b);
  }
  if (h_r) {
    free(h_r);
  }
  if (h_xhat) {
    free(h_xhat);
  }
  if (h_bhat) {
    free(h_bhat);
  }

  if (buffer_cpu) {
    free(buffer_cpu);
  }

  if (h_Plu) {
    free(h_Plu);
  }
  if (h_Qlu) {
    free(h_Qlu);
  }
  if (h_csrRowPtrL) {
    free(h_csrRowPtrL);
  }
  if (h_csrColIndL) {
    free(h_csrColIndL);
  }
  if (h_csrValL) {
    free(h_csrValL);
  }
  if (h_csrRowPtrU) {
    free(h_csrRowPtrU);
  }
  if (h_csrColIndU) {
    free(h_csrColIndU);
  }
  if (h_csrValU) {
    free(h_csrValU);
  }

  if (h_P) {
    free(h_P);
  }
  if (h_Q) {
    free(h_Q);
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
  if (d_P) {
    checkLwdaErrors(lwdaFree(d_P));
  }
  if (d_Q) {
    checkLwdaErrors(lwdaFree(d_Q));
  }
  if (d_T) {
    checkLwdaErrors(lwdaFree(d_T));
  }

  return 0;
}
