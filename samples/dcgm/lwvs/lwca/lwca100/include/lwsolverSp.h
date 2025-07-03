/*
 * Copyright 2014 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(LWSOLVERSP_H_)
#define LWSOLVERSP_H_

#include "lwsparse.h"
#include "lwblas_v2.h"
#include "lwsolver_common.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

struct lwsolverSpContext;
typedef struct lwsolverSpContext *lwsolverSpHandle_t;

struct csrqrInfo;
typedef struct csrqrInfo *csrqrInfo_t;

lwsolverStatus_t LWSOLVERAPI lwsolverSpCreate(lwsolverSpHandle_t *handle);
lwsolverStatus_t LWSOLVERAPI lwsolverSpDestroy(lwsolverSpHandle_t handle);
lwsolverStatus_t LWSOLVERAPI lwsolverSpSetStream (lwsolverSpHandle_t handle, lwdaStream_t streamId);
lwsolverStatus_t LWSOLVERAPI lwsolverSpGetStream(lwsolverSpHandle_t handle, lwdaStream_t *streamId);

lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrissymHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,
    int *issym);

/* -------- GPU linear solver by LU factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [lu] stands for LU factorization
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrlsvluHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,
    float tol, 
    int reorder,
    float *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrlsvluHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrlsvluHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const lwComplex *b,
    float tol,
    int reorder,
    lwComplex *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrlsvluHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const lwDoubleComplex *b,
    double tol,
    int reorder,
    lwDoubleComplex *x,
    int *singularity);


/* -------- GPU linear solver by QR factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [qr] stands for QR factorization
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrlsvqr(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *b,
    float tol,
    int reorder,
    float *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrlsvqr(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrlsvqr(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const lwComplex *b,
    float tol,
    int reorder,
    lwComplex *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrlsvqr(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const lwDoubleComplex *b,
    double tol,
    int reorder,
    lwDoubleComplex *x,
    int *singularity);



/* -------- CPU linear solver by QR factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [qr] stands for QR factorization
 */ 
lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrlsvqrHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,
    float tol,
    int reorder,
    float *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrlsvqrHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrlsvqrHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const lwComplex *b,
    float tol,
    int reorder,
    lwComplex *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrlsvqrHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const lwDoubleComplex *b,
    double tol,
    int reorder,
    lwDoubleComplex *x,
    int *singularity);


/* -------- CPU linear solver by Cholesky factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [chol] stands for Cholesky factorization
 *
 * Only works for symmetric positive definite matrix.
 * The upper part of A is ignored.
 */ 
lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrlsvcholHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *b,
    float tol,
    int reorder,
    float *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrlsvcholHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrlsvcholHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const lwComplex *b,
    float tol,
    int reorder,
    lwComplex *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrlsvcholHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const lwDoubleComplex *b,
    double tol,
    int reorder,
    lwDoubleComplex *x,
    int *singularity);

/* -------- GPU linear solver by Cholesky factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [chol] stands for Cholesky factorization
 *
 * Only works for symmetric positive definite matrix.
 * The upper part of A is ignored.
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrlsvchol(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *b,
    float tol,
    int reorder,
    // output
    float *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrlsvchol(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *b,
    double tol,
    int reorder,
    // output
    double *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrlsvchol(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const lwComplex *b,
    float tol,
    int reorder,
    // output
    lwComplex *x,
    int *singularity);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrlsvchol(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const lwDoubleComplex *b,
    double tol,
    int reorder,
    // output
    lwDoubleComplex *x,
    int *singularity);



/* ----------- CPU least square solver by QR factorization
 *       solve min|b - A*x| 
 * [lsq] stands for least square
 * [v] stands for vector
 * [qr] stands for QR factorization
 */ 
lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrlsqvqrHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,
    float tol,
    int *rankA,
    float *x,
    int *p,
    float *min_norm);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrlsqvqrHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,
    double tol,
    int *rankA,
    double *x,
    int *p,
    double *min_norm);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrlsqvqrHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const lwComplex *b,
    float tol,
    int *rankA,
    lwComplex *x,
    int *p,
    float *min_norm);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrlsqvqrHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const lwDoubleComplex *b,
    double tol,
    int *rankA,
    lwDoubleComplex *x,
    int *p,
    double *min_norm);

/* --------- CPU eigelwalue solver by shift ilwerse
 *      solve A*x = lambda * x 
 *   where lambda is the eigelwalue nearest mu0.
 * [eig] stands for eigelwalue solver
 * [si] stands for shift-ilwerse
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpScsreigvsiHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float mu0,
    const float *x0,
    int maxite,
    float tol,
    float *mu,
    float *x);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsreigvsiHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    double mu0,
    const double *x0,
    int maxite,
    double tol,
    double *mu,
    double *x);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsreigvsiHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    lwComplex mu0,
    const lwComplex *x0,
    int maxite,
    float tol,
    lwComplex *mu,
    lwComplex *x);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsreigvsiHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    lwDoubleComplex mu0,
    const lwDoubleComplex *x0,
    int maxite,
    double tol,
    lwDoubleComplex *mu,
    lwDoubleComplex *x);


/* --------- GPU eigelwalue solver by shift ilwerse
 *      solve A*x = lambda * x 
 *   where lambda is the eigelwalue nearest mu0.
 * [eig] stands for eigelwalue solver
 * [si] stands for shift-ilwerse
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpScsreigvsi(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float mu0,
    const float *x0,
    int maxite,
    float eps,
    float *mu,
    float *x);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsreigvsi(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    double mu0,
    const double *x0,
    int maxite,
    double eps,
    double *mu, 
    double *x);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsreigvsi(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    lwComplex mu0,
    const lwComplex *x0,
    int maxite,
    float eps,
    lwComplex *mu, 
    lwComplex *x);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsreigvsi(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    lwDoubleComplex mu0,
    const lwDoubleComplex *x0,
    int maxite,
    double eps,
    lwDoubleComplex *mu, 
    lwDoubleComplex *x);


// ----------- enclosed eigelwalues

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsreigsHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    lwComplex left_bottom_corner,
    lwComplex right_upper_corner,
    int *num_eigs);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsreigsHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    lwDoubleComplex left_bottom_corner,
    lwDoubleComplex right_upper_corner,
    int *num_eigs);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsreigsHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    lwComplex left_bottom_corner,
    lwComplex right_upper_corner,
    int *num_eigs);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsreigsHost(
    lwsolverSpHandle_t handle,
    int m,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    lwDoubleComplex left_bottom_corner,
    lwDoubleComplex right_upper_corner,
    int *num_eigs);



/* --------- CPU symrcm
 *   Symmetric reverse Lwthill McKee permutation         
 *
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrsymrcmHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *p);

/* --------- CPU symmdq 
 *   Symmetric minimum degree algorithm by quotient graph
 *
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrsymmdqHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *p);

/* --------- CPU symmdq 
 *   Symmetric Approximate minimum degree algorithm by quotient graph
 *
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrsymamdHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *p);

/* --------- CPU metis 
 *   symmetric reordering 
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrmetisndHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const int64_t *options,
    int *p);


/* --------- CPU zfd
 *  Zero free diagonal reordering
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrzfdHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *P,
    int *numnz);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrzfdHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *P,
    int *numnz);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrzfdHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *P,
    int *numnz);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrzfdHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *P,
    int *numnz);


/* --------- CPU permuation
 *   P*A*Q^T        
 *
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrperm_bufferSizeHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const int *p,
    const int *q,
    size_t *bufferSizeInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrpermHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    int *csrRowPtrA,
    int *csrColIndA,
    const int *p,
    const int *q,
    int *map,
    void *pBuffer);



/*
 *  Low-level API: Batched QR
 *
 */

lwsolverStatus_t LWSOLVERAPI lwsolverSpCreateCsrqrInfo(
    csrqrInfo_t *info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDestroyCsrqrInfo(
    csrqrInfo_t info);


lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrqrAnalysisBatched(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrqrBufferInfoBatched(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrqrBufferInfoBatched(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrqrBufferInfoBatched(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrqrBufferInfoBatched(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrqrsvBatched(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,   
    float *x,  
    int batchSize,
    csrqrInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrqrsvBatched(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,   
    double *x,  
    int batchSize,
    csrqrInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrqrsvBatched(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const lwComplex *b, 
    lwComplex *x,  
    int batchSize,
    csrqrInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrqrsvBatched(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const lwDoubleComplex *b,  
    lwDoubleComplex *x,  
    int batchSize,
    csrqrInfo_t info,
    void *pBuffer);




#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // define LWSOLVERSP_H_



