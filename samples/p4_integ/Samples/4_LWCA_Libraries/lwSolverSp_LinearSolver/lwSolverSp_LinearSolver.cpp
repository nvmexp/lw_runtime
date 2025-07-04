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
 *  Test three linear solvers, including Cholesky, LU and QR.
 *  The user has to prepare a sparse matrix of "matrix market format" (with
 extension .mtx).
 *  For example, the user can download matrices in Florida Sparse Matrix
 Collection.
 *  (http://www.cise.ufl.edu/research/sparse/matrices/)
 *
 *  The user needs to choose a solver by the switch -R<solver> and
 *  to provide the path of the matrix by the switch -F<file>, then
 *  the program solves
 *          A*x = b
 *  and reports relative error
 *          |b-A*x|/(|A|*|x|+|b|)
 *
 *  How does it work?
 *     The example solves A*x = b by the following steps
 *  step 1: B = A(Q,Q)
 *     Q is the ordering to minimize zero fill-in.
 *     The user can choose symrcm or symamd.
 *  step 2: solve B*z = Q*b
 *  step 3: x = ilw(Q)*z
 *
 *  Above three steps can be combined by the formula
 *        (Q*A*Q')*(Q*x) = (Q*b)
 *
 *  The elapsed time is also reported so the user can compare efficiency of
 different solvers.
 *
 *  How to use
        /lwSolverSp_LinearSolver            // Default: Cholesky, symrcm &
 file=lap2D_5pt_n100.mtx
 *     ./lwSolverSp_LinearSolver -R=chol  -file=<file>   // cholesky
 factorization
 *     ./lwSolverSp_LinearSolver -R=lu -P=symrcm -file=<file>     // symrcm + LU
 with partial pivoting
 *     ./lwSolverSp_LinearSolver -R=qr -P=symamd -file=<file>     // symamd + QR
 factorization
 *
 *
 *  Remark: the absolute error on solution x is meaningless without knowing
 condition number of A.
 *     The relative error on residual should be close to machine zero,
 i.e. 1.e-15.
 */

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lwda_runtime.h>

#include "lwsolverSp.h"
#include "lwsparse.h"

#include "helper_lwda.h"
#include "helper_lwsolver.h"

template <typename T_ELEM>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m,
                       int *n, int *nnz, T_ELEM **aVal, int **aRowInd,
                       int **aColInd, int extendSymMatrix);

void UsageSP(void) {
  printf("<options>\n");
  printf("-h          : display this help\n");
  printf("-R=<name>   : choose a linear solver\n");
  printf("              chol (cholesky factorization), this is default\n");
  printf("              qr   (QR factorization)\n");
  printf("              lu   (LU factorization)\n");
  printf("-P=<name>    : choose a reordering\n");
  printf("              symrcm (Reverse Lwthill-McKee)\n");
  printf("              symamd (Approximate Minimum Degree)\n");
  printf("              metis  (nested dissection)\n");
  printf("-file=<filename> : filename containing a matrix in MM format\n");
  printf("-device=<device_id> : <device_id> if want to run on specific GPU\n");

  exit(0);
}

void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts) {
  memset(&opts, 0, sizeof(opts));

  if (checkCmdLineFlag(argc, (const char **)argv, "-h")) {
    UsageSP();
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "R")) {
    char *solverType = NULL;
    getCmdLineArgumentString(argc, (const char **)argv, "R", &solverType);

    if (solverType) {
      if ((STRCASECMP(solverType, "chol") != 0) &&
          (STRCASECMP(solverType, "lu") != 0) &&
          (STRCASECMP(solverType, "qr") != 0)) {
        printf("\nIncorrect argument passed to -R option\n");
        UsageSP();
      } else {
        opts.testFunc = solverType;
      }
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "P")) {
    char *reorderType = NULL;
    getCmdLineArgumentString(argc, (const char **)argv, "P", &reorderType);

    if (reorderType) {
      if ((STRCASECMP(reorderType, "symrcm") != 0) &&
          (STRCASECMP(reorderType, "symamd") != 0) &&
          (STRCASECMP(reorderType, "metis") != 0)) {
        printf("\nIncorrect argument passed to -P option\n");
        UsageSP();
      } else {
        opts.reorder = reorderType;
      }
    }
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
  lwsolverSpHandle_t handle = NULL;
  lwsparseHandle_t lwsparseHandle = NULL; /* used in residual evaluation */
  lwdaStream_t stream = NULL;
  lwsparseMatDescr_t descrA = NULL;

  int rowsA = 0; /* number of rows of A */
  int colsA = 0; /* number of columns of A */
  int nnzA = 0;  /* number of nonzeros of A */
  int baseA = 0; /* base index in CSR format */

  /* CSR(A) from I/O */
  int *h_csrRowPtrA = NULL;
  int *h_csrColIndA = NULL;
  double *h_csrValA = NULL;

  double *h_z = NULL;  /* z = B \ (Q*b) */
  double *h_x = NULL;  /* x = A \ b */
  double *h_b = NULL;  /* b = ones(n,1) */
  double *h_Qb = NULL; /* Q*b */
  double *h_r = NULL;  /* r = b - A*x */

  int *h_Q = NULL; /* <int> n */
                   /* reorder to reduce zero fill-in */
                   /* Q = symrcm(A) or Q = symamd(A) */
  /* B = Q*A*Q' or B = A(Q,Q) by MATLAB notation */
  int *h_csrRowPtrB = NULL; /* <int> n+1 */
  int *h_csrColIndB = NULL; /* <int> nnzA */
  double *h_csrValB = NULL; /* <double> nnzA */
  int *h_mapBfromA = NULL;  /* <int> nnzA */

  size_t size_perm = 0;
  void *buffer_cpu = NULL; /* working space for permutation: B = Q*A*Q^T */

  /* device copy of A: used in residual evaluation */
  int *d_csrRowPtrA = NULL;
  int *d_csrColIndA = NULL;
  double *d_csrValA = NULL;

  /* device copy of B: used in B*z = Q*b */
  int *d_csrRowPtrB = NULL;
  int *d_csrColIndB = NULL;
  double *d_csrValB = NULL;

  int *d_Q = NULL;     /* device copy of h_Q */
  double *d_z = NULL;  /* z = B \ Q*b */
  double *d_x = NULL;  /* x = A \ b */
  double *d_b = NULL;  /* a copy of h_b */
  double *d_Qb = NULL; /* a copy of h_Qb */
  double *d_r = NULL;  /* r = b - A*x */

  double tol = 1.e-12;
  const int reorder = 0; /* no reordering */
  int singularity = 0;   /* -1 if A is ilwertible under tol. */

  /* the constants are used in residual evaluation, r = b - A*x */
  const double minus_one = -1.0;
  const double one = 1.0;

  double b_inf = 0.0;
  double x_inf = 0.0;
  double r_inf = 0.0;
  double A_inf = 0.0;
  int errors = 0;
  int issym = 0;

  double start, stop;
  double time_solve_cpu;
  double time_solve_gpu;

  parseCommandLineArguments(argc, argv, opts);

  if (NULL == opts.testFunc) {
    opts.testFunc =
        "chol"; /* By default running Cholesky as NO solver selected with -R
                   option. */
  }

  findLwdaDevice(argc, (const char **)argv);

  if (opts.sparse_mat_filename == NULL) {
    opts.sparse_mat_filename = sdkFindFilePath("lap2D_5pt_n100.mtx", argv[0]);
    if (opts.sparse_mat_filename != NULL)
      printf("Using default input file [%s]\n", opts.sparse_mat_filename);
    else
      printf("Could not find lap2D_5pt_n100.mtx\n");
  } else {
    printf("Using input file [%s]\n", opts.sparse_mat_filename);
  }

  printf("step 1: read matrix market format\n");

  if (opts.sparse_mat_filename == NULL) {
    fprintf(stderr, "Error: input matrix is not provided\n");
    return EXIT_FAILURE;
  }

  if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true, &rowsA,
                                 &colsA, &nnzA, &h_csrValA, &h_csrRowPtrA,
                                 &h_csrColIndA, true)) {
    exit(EXIT_FAILURE);
  }
  baseA = h_csrRowPtrA[0];  // baseA = {0,1}
  printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA,
         nnzA, baseA);

  if (rowsA != colsA) {
    fprintf(stderr, "Error: only support square matrix\n");
    return 1;
  }

  checkLwdaErrors(lwsolverSpCreate(&handle));
  checkLwdaErrors(lwsparseCreate(&lwsparseHandle));

  checkLwdaErrors(lwdaStreamCreate(&stream));
  /* bind stream to lwsparse and lwsolver*/
  checkLwdaErrors(lwsolverSpSetStream(handle, stream));
  checkLwdaErrors(lwsparseSetStream(lwsparseHandle, stream));

  /* configure matrix descriptor*/
  checkLwdaErrors(lwsparseCreateMatDescr(&descrA));
  checkLwdaErrors(lwsparseSetMatType(descrA, LWSPARSE_MATRIX_TYPE_GENERAL));
  if (baseA) {
    checkLwdaErrors(lwsparseSetMatIndexBase(descrA, LWSPARSE_INDEX_BASE_ONE));
  } else {
    checkLwdaErrors(lwsparseSetMatIndexBase(descrA, LWSPARSE_INDEX_BASE_ZERO));
  }

  h_z = (double *)malloc(sizeof(double) * colsA);
  h_x = (double *)malloc(sizeof(double) * colsA);
  h_b = (double *)malloc(sizeof(double) * rowsA);
  h_Qb = (double *)malloc(sizeof(double) * rowsA);
  h_r = (double *)malloc(sizeof(double) * rowsA);

  h_Q = (int *)malloc(sizeof(int) * colsA);
  h_csrRowPtrB = (int *)malloc(sizeof(int) * (rowsA + 1));
  h_csrColIndB = (int *)malloc(sizeof(int) * nnzA);
  h_csrValB = (double *)malloc(sizeof(double) * nnzA);
  h_mapBfromA = (int *)malloc(sizeof(int) * nnzA);

  assert(NULL != h_z);
  assert(NULL != h_x);
  assert(NULL != h_b);
  assert(NULL != h_Qb);
  assert(NULL != h_r);
  assert(NULL != h_Q);
  assert(NULL != h_csrRowPtrB);
  assert(NULL != h_csrColIndB);
  assert(NULL != h_csrValB);
  assert(NULL != h_mapBfromA);

  checkLwdaErrors(
      lwdaMalloc((void **)&d_csrRowPtrA, sizeof(int) * (rowsA + 1)));
  checkLwdaErrors(lwdaMalloc((void **)&d_csrColIndA, sizeof(int) * nnzA));
  checkLwdaErrors(lwdaMalloc((void **)&d_csrValA, sizeof(double) * nnzA));
  checkLwdaErrors(
      lwdaMalloc((void **)&d_csrRowPtrB, sizeof(int) * (rowsA + 1)));
  checkLwdaErrors(lwdaMalloc((void **)&d_csrColIndB, sizeof(int) * nnzA));
  checkLwdaErrors(lwdaMalloc((void **)&d_csrValB, sizeof(double) * nnzA));
  checkLwdaErrors(lwdaMalloc((void **)&d_Q, sizeof(int) * colsA));
  checkLwdaErrors(lwdaMalloc((void **)&d_z, sizeof(double) * colsA));
  checkLwdaErrors(lwdaMalloc((void **)&d_x, sizeof(double) * colsA));
  checkLwdaErrors(lwdaMalloc((void **)&d_b, sizeof(double) * rowsA));
  checkLwdaErrors(lwdaMalloc((void **)&d_Qb, sizeof(double) * rowsA));
  checkLwdaErrors(lwdaMalloc((void **)&d_r, sizeof(double) * rowsA));

  /* verify if A has symmetric pattern or not */
  checkLwdaErrors(lwsolverSpXcsrissymHost(handle, rowsA, nnzA, descrA,
                                          h_csrRowPtrA, h_csrRowPtrA + 1,
                                          h_csrColIndA, &issym));

  if (0 == strcmp(opts.testFunc, "chol")) {
    if (!issym) {
      printf("Error: A has no symmetric pattern, please use LU or QR \n");
      exit(EXIT_FAILURE);
    }
  }

  printf("step 2: reorder the matrix A to minimize zero fill-in\n");
  printf(
      "        if the user choose a reordering by -P=symrcm, -P=symamd or "
      "-P=metis\n");

  if (NULL != opts.reorder) {
    if (0 == strcmp(opts.reorder, "symrcm")) {
      printf("step 2.1: Q = symrcm(A) \n");
      checkLwdaErrors(lwsolverSpXcsrsymrcmHost(
          handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrColIndA, h_Q));
    } else if (0 == strcmp(opts.reorder, "symamd")) {
      printf("step 2.1: Q = symamd(A) \n");
      checkLwdaErrors(lwsolverSpXcsrsymamdHost(
          handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrColIndA, h_Q));
    } else if (0 == strcmp(opts.reorder, "metis")) {
      printf("step 2.1: Q = metis(A) \n");
      checkLwdaErrors(lwsolverSpXcsrmetisndHost(handle, rowsA, nnzA, descrA,
                                                h_csrRowPtrA, h_csrColIndA,
                                                NULL, /* default setting. */
                                                h_Q));
    } else {
      fprintf(stderr, "Error: %s is unknown reordering\n", opts.reorder);
      return 1;
    }
  } else {
    printf("step 2.1: no reordering is chosen, Q = 0:n-1 \n");
    for (int j = 0; j < rowsA; j++) {
      h_Q[j] = j;
    }
  }

  printf("step 2.2: B = A(Q,Q) \n");

  memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int) * (rowsA + 1));
  memcpy(h_csrColIndB, h_csrColIndA, sizeof(int) * nnzA);

  checkLwdaErrors(lwsolverSpXcsrperm_bufferSizeHost(
      handle, rowsA, colsA, nnzA, descrA, h_csrRowPtrB, h_csrColIndB, h_Q, h_Q,
      &size_perm));

  if (buffer_cpu) {
    free(buffer_cpu);
  }
  buffer_cpu = (void *)malloc(sizeof(char) * size_perm);
  assert(NULL != buffer_cpu);

  /* h_mapBfromA = Identity */
  for (int j = 0; j < nnzA; j++) {
    h_mapBfromA[j] = j;
  }
  checkLwdaErrors(lwsolverSpXcsrpermHost(handle, rowsA, colsA, nnzA, descrA,
                                         h_csrRowPtrB, h_csrColIndB, h_Q, h_Q,
                                         h_mapBfromA, buffer_cpu));

  /* B = A( mapBfromA ) */
  for (int j = 0; j < nnzA; j++) {
    h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
  }

  printf("step 3: b(j) = 1 + j/n \n");
  for (int row = 0; row < rowsA; row++) {
    h_b[row] = 1.0 + ((double)row) / ((double)rowsA);
  }

  /* h_Qb = b(Q) */
  for (int row = 0; row < rowsA; row++) {
    h_Qb[row] = h_b[h_Q[row]];
  }

  printf("step 4: prepare data on device\n");
  checkLwdaErrors(lwdaMemcpyAsync(d_csrRowPtrA, h_csrRowPtrA,
                                  sizeof(int) * (rowsA + 1),
                                  lwdaMemcpyHostToDevice, stream));
  checkLwdaErrors(lwdaMemcpyAsync(d_csrColIndA, h_csrColIndA,
                                  sizeof(int) * nnzA, lwdaMemcpyHostToDevice,
                                  stream));
  checkLwdaErrors(lwdaMemcpyAsync(d_csrValA, h_csrValA, sizeof(double) * nnzA,
                                  lwdaMemcpyHostToDevice, stream));
  checkLwdaErrors(lwdaMemcpyAsync(d_csrRowPtrB, h_csrRowPtrB,
                                  sizeof(int) * (rowsA + 1),
                                  lwdaMemcpyHostToDevice, stream));
  checkLwdaErrors(lwdaMemcpyAsync(d_csrColIndB, h_csrColIndB,
                                  sizeof(int) * nnzA, lwdaMemcpyHostToDevice,
                                  stream));
  checkLwdaErrors(lwdaMemcpyAsync(d_csrValB, h_csrValB, sizeof(double) * nnzA,
                                  lwdaMemcpyHostToDevice, stream));
  checkLwdaErrors(lwdaMemcpyAsync(d_b, h_b, sizeof(double) * rowsA,
                                  lwdaMemcpyHostToDevice, stream));
  checkLwdaErrors(lwdaMemcpyAsync(d_Qb, h_Qb, sizeof(double) * rowsA,
                                  lwdaMemcpyHostToDevice, stream));
  checkLwdaErrors(lwdaMemcpyAsync(d_Q, h_Q, sizeof(int) * rowsA,
                                  lwdaMemcpyHostToDevice, stream));

  printf("step 5: solve A*x = b on CPU \n");
  start = second();

  /* solve B*z = Q*b */
  if (0 == strcmp(opts.testFunc, "chol")) {
    checkLwdaErrors(lwsolverSpDcsrlsvcholHost(
        handle, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
        h_Qb, tol, reorder, h_z, &singularity));
  } else if (0 == strcmp(opts.testFunc, "lu")) {
    checkLwdaErrors(lwsolverSpDcsrlsvluHost(
        handle, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
        h_Qb, tol, reorder, h_z, &singularity));

  } else if (0 == strcmp(opts.testFunc, "qr")) {
    checkLwdaErrors(lwsolverSpDcsrlsvqrHost(
        handle, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
        h_Qb, tol, reorder, h_z, &singularity));
  } else {
    fprintf(stderr, "Error: %s is unknown function\n", opts.testFunc);
    return 1;
  }

  /* Q*x = z */
  for (int row = 0; row < rowsA; row++) {
    h_x[h_Q[row]] = h_z[row];
  }

  if (0 <= singularity) {
    printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
           singularity, tol);
  }

  stop = second();
  time_solve_cpu = stop - start;

  printf("step 6: evaluate residual r = b - A*x (result on CPU)\n");
  checkLwdaErrors(lwdaMemcpyAsync(d_r, d_b, sizeof(double) * rowsA,
                                  lwdaMemcpyDeviceToDevice, stream));
  checkLwdaErrors(lwdaMemcpyAsync(d_x, h_x, sizeof(double) * colsA,
                                  lwdaMemcpyHostToDevice, stream));

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
      lwsparseHandle, LWSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
      &one, vecAx, LWDA_R_64F, LWSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  void *buffer = NULL;
  checkLwdaErrors(lwdaMalloc(&buffer, bufferSize));

  checkLwdaErrors(lwsparseSpMV(lwsparseHandle, LWSPARSE_OPERATION_NON_TRANSPOSE,
                               &minus_one, matA, vecx, &one, vecAx, LWDA_R_64F,
                               LWSPARSE_SPMV_ALG_DEFAULT, buffer));

  checkLwdaErrors(lwdaMemcpyAsync(h_r, d_r, sizeof(double) * rowsA,
                                  lwdaMemcpyDeviceToHost, stream));
  /* wait until h_r is ready */
  checkLwdaErrors(lwdaDeviceSynchronize());

  b_inf = vec_norminf(rowsA, h_b);
  x_inf = vec_norminf(colsA, h_x);
  r_inf = vec_norminf(rowsA, h_r);
  A_inf = csr_mat_norminf(rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA,
                          h_csrColIndA);

  printf("(CPU) |b - A*x| = %E \n", r_inf);
  printf("(CPU) |A| = %E \n", A_inf);
  printf("(CPU) |x| = %E \n", x_inf);
  printf("(CPU) |b| = %E \n", b_inf);
  printf("(CPU) |b - A*x|/(|A|*|x| + |b|) = %E \n",
         r_inf / (A_inf * x_inf + b_inf));

  printf("step 7: solve A*x = b on GPU\n");
  start = second();

  /* solve B*z = Q*b */
  if (0 == strcmp(opts.testFunc, "chol")) {
    checkLwdaErrors(lwsolverSpDcsrlsvchol(
        handle, rowsA, nnzA, descrA, d_csrValB, d_csrRowPtrB, d_csrColIndB,
        d_Qb, tol, reorder, d_z, &singularity));

  } else if (0 == strcmp(opts.testFunc, "lu")) {
    printf("WARNING: no LU available on GPU \n");
  } else if (0 == strcmp(opts.testFunc, "qr")) {
    checkLwdaErrors(lwsolverSpDcsrlsvqr(handle, rowsA, nnzA, descrA, d_csrValB,
                                        d_csrRowPtrB, d_csrColIndB, d_Qb, tol,
                                        reorder, d_z, &singularity));
  } else {
    fprintf(stderr, "Error: %s is unknow function\n", opts.testFunc);
    return 1;
  }
  checkLwdaErrors(lwdaDeviceSynchronize());
  if (0 <= singularity) {
    printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
           singularity, tol);
  }
  /* Q*x = z */
  checkLwdaErrors(lwsparseDsctr(lwsparseHandle, rowsA, d_z, d_Q, d_x,
                                LWSPARSE_INDEX_BASE_ZERO));
  checkLwdaErrors(lwdaDeviceSynchronize());

  stop = second();
  time_solve_gpu = stop - start;

  printf("step 8: evaluate residual r = b - A*x (result on GPU)\n");
  checkLwdaErrors(lwdaMemcpyAsync(d_r, d_b, sizeof(double) * rowsA,
                                  lwdaMemcpyDeviceToDevice, stream));

  checkLwdaErrors(lwsparseSpMV(lwsparseHandle, LWSPARSE_OPERATION_NON_TRANSPOSE,
                               &minus_one, matA, vecx, &one, vecAx, LWDA_R_64F,
                               LWSPARSE_SPMV_ALG_DEFAULT, buffer));

  checkLwdaErrors(lwdaMemcpyAsync(h_x, d_x, sizeof(double) * colsA,
                                  lwdaMemcpyDeviceToHost, stream));
  checkLwdaErrors(lwdaMemcpyAsync(h_r, d_r, sizeof(double) * rowsA,
                                  lwdaMemcpyDeviceToHost, stream));
  /* wait until h_x and h_r are ready */
  checkLwdaErrors(lwdaDeviceSynchronize());

  b_inf = vec_norminf(rowsA, h_b);
  x_inf = vec_norminf(colsA, h_x);
  r_inf = vec_norminf(rowsA, h_r);

  if (0 != strcmp(opts.testFunc, "lu")) {
    // only cholesky and qr have GPU version
    printf("(GPU) |b - A*x| = %E \n", r_inf);
    printf("(GPU) |A| = %E \n", A_inf);
    printf("(GPU) |x| = %E \n", x_inf);
    printf("(GPU) |b| = %E \n", b_inf);
    printf("(GPU) |b - A*x|/(|A|*|x| + |b|) = %E \n",
           r_inf / (A_inf * x_inf + b_inf));
  }

  fprintf(stdout, "timing %s: CPU = %10.6f sec , GPU = %10.6f sec\n",
          opts.testFunc, time_solve_cpu, time_solve_gpu);

  if (0 != strcmp(opts.testFunc, "lu")) {
    printf("show last 10 elements of solution vector (GPU) \n");
    printf("consistent result for different reordering and solver \n");
    for (int j = rowsA - 10; j < rowsA; j++) {
      printf("x[%d] = %E\n", j, h_x[j]);
    }
  }

  if (handle) {
    checkLwdaErrors(lwsolverSpDestroy(handle));
  }
  if (lwsparseHandle) {
    checkLwdaErrors(lwsparseDestroy(lwsparseHandle));
  }
  if (stream) {
    checkLwdaErrors(lwdaStreamDestroy(stream));
  }
  if (descrA) {
    checkLwdaErrors(lwsparseDestroyMatDescr(descrA));
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
  if (h_z) {
    free(h_z);
  }
  if (h_x) {
    free(h_x);
  }
  if (h_b) {
    free(h_b);
  }
  if (h_Qb) {
    free(h_Qb);
  }
  if (h_r) {
    free(h_r);
  }

  if (h_Q) {
    free(h_Q);
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

  if (buffer_cpu) {
    free(buffer_cpu);
  }

  if (buffer) {
    checkLwdaErrors(lwdaFree(buffer));
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
  if (d_csrValB) {
    checkLwdaErrors(lwdaFree(d_csrValB));
  }
  if (d_csrRowPtrB) {
    checkLwdaErrors(lwdaFree(d_csrRowPtrB));
  }
  if (d_csrColIndB) {
    checkLwdaErrors(lwdaFree(d_csrColIndB));
  }
  if (d_Q) {
    checkLwdaErrors(lwdaFree(d_Q));
  }
  if (d_z) {
    checkLwdaErrors(lwdaFree(d_z));
  }
  if (d_x) {
    checkLwdaErrors(lwdaFree(d_x));
  }
  if (d_b) {
    checkLwdaErrors(lwdaFree(d_b));
  }
  if (d_Qb) {
    checkLwdaErrors(lwdaFree(d_Qb));
  }
  if (d_r) {
    checkLwdaErrors(lwdaFree(d_r));
  }

  return 0;
}
