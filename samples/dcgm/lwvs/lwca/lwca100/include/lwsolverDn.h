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
 
 /*   lwSolverDN : Dense Linear Algebra Library

 */
 
#if !defined(LWSOLVERDN_H_)
#define LWSOLVERDN_H_

#include "driver_types.h"
#include "lwComplex.h"   /* import complex data type */
#include "lwblas_v2.h"
#include "lwsolver_common.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

struct lwsolverDnContext;
typedef struct lwsolverDnContext *lwsolverDnHandle_t;

struct syevjInfo;
typedef struct syevjInfo *syevjInfo_t;

struct gesvdjInfo;
typedef struct gesvdjInfo *gesvdjInfo_t;


lwsolverStatus_t LWSOLVERAPI lwsolverDnCreate(lwsolverDnHandle_t *handle);
lwsolverStatus_t LWSOLVERAPI lwsolverDnDestroy(lwsolverDnHandle_t handle);
lwsolverStatus_t LWSOLVERAPI lwsolverDnSetStream (lwsolverDnHandle_t handle, lwdaStream_t streamId);
lwsolverStatus_t LWSOLVERAPI lwsolverDnGetStream(lwsolverDnHandle_t handle, lwdaStream_t *streamId);


/* Cholesky factorization and its solver */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSpotrf_bufferSize( 
    lwsolverDnHandle_t handle, 
    lwblasFillMode_t uplo, 
    int n, 
    float *A, 
    int lda, 
    int *Lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDpotrf_bufferSize( 
    lwsolverDnHandle_t handle, 
    lwblasFillMode_t uplo, 
    int n, 
    double *A, 
    int lda, 
    int *Lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCpotrf_bufferSize( 
    lwsolverDnHandle_t handle, 
    lwblasFillMode_t uplo, 
    int n, 
    lwComplex *A, 
    int lda, 
    int *Lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZpotrf_bufferSize( 
    lwsolverDnHandle_t handle, 
    lwblasFillMode_t uplo, 
    int n, 
    lwDoubleComplex *A, 
    int lda, 
    int *Lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSpotrf( 
    lwsolverDnHandle_t handle, 
    lwblasFillMode_t uplo, 
    int n, 
    float *A, 
    int lda,  
    float *Workspace, 
    int Lwork, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDpotrf( 
    lwsolverDnHandle_t handle, 
    lwblasFillMode_t uplo, 
    int n, 
    double *A, 
    int lda, 
    double *Workspace, 
    int Lwork, 
    int *devInfo );



lwsolverStatus_t LWSOLVERAPI lwsolverDnCpotrf( 
    lwsolverDnHandle_t handle, 
    lwblasFillMode_t uplo, 
    int n, 
    lwComplex *A, 
    int lda, 
    lwComplex *Workspace, 
    int Lwork, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZpotrf( 
    lwsolverDnHandle_t handle, 
    lwblasFillMode_t uplo, 
    int n, 
    lwDoubleComplex *A, 
    int lda, 
    lwDoubleComplex *Workspace, 
    int Lwork, 
    int *devInfo );


lwsolverStatus_t LWSOLVERAPI lwsolverDnSpotrs(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    int nrhs,
    const float *A,
    int lda,
    float *B,
    int ldb,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDpotrs(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    int nrhs,
    const double *A,
    int lda,
    double *B,
    int ldb,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCpotrs(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    int nrhs,
    const lwComplex *A,
    int lda,
    lwComplex *B,
    int ldb,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZpotrs(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    int nrhs,
    const lwDoubleComplex *A,
    int lda,
    lwDoubleComplex *B,
    int ldb,
    int *devInfo);

/* batched Cholesky factorization and its solver */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSpotrfBatched(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    float *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDpotrfBatched(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    double *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCpotrfBatched(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwComplex *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZpotrfBatched(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwDoubleComplex *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSpotrsBatched(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    float *A[],
    int lda,
    float *B[],
    int ldb,
    int *d_info,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDpotrsBatched(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    double *A[],
    int lda,
    double *B[],
    int ldb,
    int *d_info,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCpotrsBatched(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    lwComplex *A[],
    int lda,
    lwComplex *B[],
    int ldb,
    int *d_info,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZpotrsBatched(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    lwDoubleComplex *A[],
    int lda,
    lwDoubleComplex *B[],
    int ldb,
    int *d_info,
    int batchSize);


/* LU Factorization */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSgetrf_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    int *Lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgetrf_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    int *Lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgetrf_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    lwComplex *A,
    int lda,
    int *Lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgetrf_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    lwDoubleComplex *A,
    int lda,
    int *Lwork );


lwsolverStatus_t LWSOLVERAPI lwsolverDnSgetrf( 
    lwsolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A, 
    int lda, 
    float *Workspace, 
    int *devIpiv, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgetrf( 
    lwsolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    double *Workspace, 
    int *devIpiv, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgetrf( 
    lwsolverDnHandle_t handle, 
    int m, 
    int n, 
    lwComplex *A, 
    int lda, 
    lwComplex *Workspace, 
    int *devIpiv, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgetrf( 
    lwsolverDnHandle_t handle, 
    int m, 
    int n, 
    lwDoubleComplex *A, 
    int lda, 
    lwDoubleComplex *Workspace, 
    int *devIpiv, 
    int *devInfo );

/* Row pivoting */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSlaswp( 
    lwsolverDnHandle_t handle, 
    int n, 
    float *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDlaswp( 
    lwsolverDnHandle_t handle, 
    int n, 
    double *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

lwsolverStatus_t LWSOLVERAPI lwsolverDnClaswp( 
    lwsolverDnHandle_t handle, 
    int n, 
    lwComplex *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZlaswp( 
    lwsolverDnHandle_t handle, 
    int n, 
    lwDoubleComplex *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

/* LU solve */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSgetrs( 
    lwsolverDnHandle_t handle, 
    lwblasOperation_t trans, 
    int n, 
    int nrhs, 
    const float *A, 
    int lda, 
    const int *devIpiv, 
    float *B, 
    int ldb, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgetrs( 
    lwsolverDnHandle_t handle, 
    lwblasOperation_t trans, 
    int n, 
    int nrhs, 
    const double *A, 
    int lda, 
    const int *devIpiv, 
    double *B, 
    int ldb, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgetrs( 
    lwsolverDnHandle_t handle, 
    lwblasOperation_t trans, 
    int n, 
    int nrhs, 
    const lwComplex *A, 
    int lda, 
    const int *devIpiv, 
    lwComplex *B, 
    int ldb, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgetrs( 
    lwsolverDnHandle_t handle, 
    lwblasOperation_t trans, 
    int n, 
    int nrhs, 
    const lwDoubleComplex *A, 
    int lda, 
    const int *devIpiv, 
    lwDoubleComplex *B, 
    int ldb, 
    int *devInfo );


/* QR factorization */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSgeqrf_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    int *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgeqrf_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    int *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgeqrf_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    lwComplex *A,
    int lda,
    int *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgeqrf_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    lwDoubleComplex *A,
    int lda,
    int *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnSgeqrf( 
    lwsolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A,  
    int lda, 
    float *TAU,  
    float *Workspace,  
    int Lwork, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgeqrf( 
    lwsolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    double *TAU, 
    double *Workspace, 
    int Lwork, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgeqrf( 
    lwsolverDnHandle_t handle, 
    int m, 
    int n, 
    lwComplex *A, 
    int lda, 
    lwComplex *TAU, 
    lwComplex *Workspace, 
    int Lwork, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgeqrf( 
    lwsolverDnHandle_t handle, 
    int m, 
    int n, 
    lwDoubleComplex *A, 
    int lda, 
    lwDoubleComplex *TAU, 
    lwDoubleComplex *Workspace, 
    int Lwork, 
    int *devInfo );


/* generate unitary matrix Q from QR factorization */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSorgqr_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDorgqr_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnLwngqr_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const lwComplex *A,
    int lda,
    const lwComplex *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZungqr_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const lwDoubleComplex *A,
    int lda,
    const lwDoubleComplex *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSorgqr(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int k,
    float *A,
    int lda,
    const float *tau,
    float *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDorgqr(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int k,
    double *A,
    int lda,
    const double *tau,
    double *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnLwngqr(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int k,
    lwComplex *A,
    int lda,
    const lwComplex *tau,
    lwComplex *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZungqr(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int k,
    lwDoubleComplex *A,
    int lda,
    const lwDoubleComplex *tau,
    lwDoubleComplex *work,
    int lwork,
    int *info);



/* compute Q**T*b in solve min||A*x = b|| */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSormqr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasOperation_t trans,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    const float *C,
    int ldc,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDormqr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasOperation_t trans,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    const double *C,
    int ldc,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnLwnmqr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasOperation_t trans,
    int m,
    int n,
    int k,
    const lwComplex *A,
    int lda,
    const lwComplex *tau,
    const lwComplex *C,
    int ldc,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZunmqr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasOperation_t trans,
    int m,
    int n,
    int k,
    const lwDoubleComplex *A,
    int lda,
    const lwDoubleComplex *tau,
    const lwDoubleComplex *C,
    int ldc,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSormqr(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasOperation_t trans,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    float *C,
    int ldc,
    float *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDormqr(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasOperation_t trans,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    double *C,
    int ldc,
    double *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnLwnmqr(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasOperation_t trans,
    int m,
    int n,
    int k,
    const lwComplex *A,
    int lda,
    const lwComplex *tau,
    lwComplex *C,
    int ldc,
    lwComplex *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZunmqr(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasOperation_t trans,
    int m,
    int n,
    int k,
    const lwDoubleComplex *A,
    int lda,
    const lwDoubleComplex *tau,
    lwDoubleComplex *C,
    int ldc,
    lwDoubleComplex *work,
    int lwork,
    int *devInfo);


/* L*D*L**T,U*D*U**T factorization */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSsytrf_bufferSize(
    lwsolverDnHandle_t handle,
    int n,
    float *A,
    int lda,
    int *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsytrf_bufferSize(
    lwsolverDnHandle_t handle,
    int n,
    double *A,
    int lda,
    int *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCsytrf_bufferSize(
    lwsolverDnHandle_t handle,
    int n,
    lwComplex *A,
    int lda,
    int *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZsytrf_bufferSize(
    lwsolverDnHandle_t handle,
    int n,
    lwDoubleComplex *A,
    int lda,
    int *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnSsytrf(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *ipiv,
    float *work,
    int lwork,
    int *info );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsytrf(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *ipiv,
    double *work,
    int lwork,
    int *info );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCsytrf(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwComplex *A,
    int lda,
    int *ipiv,
    lwComplex *work,
    int lwork,
    int *info );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZsytrf(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwDoubleComplex *A,
    int lda,
    int *ipiv,
    lwDoubleComplex *work,
    int lwork,
    int *info );


/* bidiagonal factorization */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSgebrd_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgebrd_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgebrd_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgebrd_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnSgebrd( 
    lwsolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A,  
    int lda,
    float *D, 
    float *E, 
    float *TAUQ,  
    float *TAUP, 
    float *Work,
    int Lwork, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgebrd( 
    lwsolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda,
    double *D, 
    double *E, 
    double *TAUQ, 
    double *TAUP, 
    double *Work,
    int Lwork, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgebrd( 
    lwsolverDnHandle_t handle, 
    int m, 
    int n, 
    lwComplex *A, 
    int lda, 
    float *D, 
    float *E, 
    lwComplex *TAUQ, 
    lwComplex *TAUP,
    lwComplex *Work, 
    int Lwork, 
    int *devInfo );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgebrd( 
    lwsolverDnHandle_t handle, 
    int m, 
    int n, 
    lwDoubleComplex *A,
    int lda, 
    double *D, 
    double *E, 
    lwDoubleComplex *TAUQ,
    lwDoubleComplex *TAUP, 
    lwDoubleComplex *Work, 
    int Lwork, 
    int *devInfo );

/* generates one of the unitary matrices Q or P**T determined by GEBRD*/
lwsolverStatus_t LWSOLVERAPI lwsolverDnSorgbr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side, 
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDorgbr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side, 
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnLwngbr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side, 
    int m,
    int n,
    int k,
    const lwComplex *A,
    int lda,
    const lwComplex *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZungbr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side, 
    int m,
    int n,
    int k,
    const lwDoubleComplex *A,
    int lda,
    const lwDoubleComplex *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSorgbr(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side, 
    int m,
    int n,
    int k,
    float *A,
    int lda,
    const float *tau,
    float *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDorgbr(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side, 
    int m,
    int n,
    int k,
    double *A,
    int lda,
    const double *tau,
    double *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnLwngbr(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side, 
    int m,
    int n,
    int k,
    lwComplex *A,
    int lda,
    const lwComplex *tau,
    lwComplex *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZungbr(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side, 
    int m,
    int n,
    int k,
    lwDoubleComplex *A,
    int lda,
    const lwDoubleComplex *tau,
    lwDoubleComplex *work,
    int lwork,
    int *info);


/* tridiagonal factorization */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSsytrd_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *d,
    const float *e,
    const float *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsytrd_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *d,
    const double *e,
    const double *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnChetrd_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    const lwComplex *A,
    int lda,
    const float *d,
    const float *e,
    const lwComplex *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZhetrd_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    const lwDoubleComplex *A,
    int lda,
    const double *d,
    const double *e,
    const lwDoubleComplex *tau,
    int *lwork);


lwsolverStatus_t LWSOLVERAPI lwsolverDnSsytrd(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *d,
    float *e,
    float *tau,
    float *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsytrd(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *d,
    double *e,
    double *tau,
    double *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnChetrd(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwComplex *A,
    int lda,
    float *d,
    float *e,
    lwComplex *tau,
    lwComplex *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZhetrd(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwDoubleComplex *A,
    int lda,
    double *d,
    double *e,
    lwDoubleComplex *tau,
    lwDoubleComplex *work,
    int lwork,
    int *info);



/* generate unitary Q comes from sytrd */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSorgtr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo, 
    int n,
    const float *A,
    int lda,
    const float *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDorgtr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnLwngtr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo, 
    int n,
    const lwComplex *A,
    int lda,
    const lwComplex *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZungtr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo, 
    int n,
    const lwDoubleComplex *A,
    int lda,
    const lwDoubleComplex *tau,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSorgtr(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo, 
    int n,
    float *A,
    int lda,
    const float *tau,
    float *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDorgtr(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo, 
    int n,
    double *A,
    int lda,
    const double *tau,
    double *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnLwngtr(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo, 
    int n,
    lwComplex *A,
    int lda,
    const lwComplex *tau,
    lwComplex *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZungtr(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo, 
    int n,
    lwDoubleComplex *A,
    int lda,
    const lwDoubleComplex *tau,
    lwDoubleComplex *work,
    int lwork,
    int *info);



/* compute op(Q)*C or C*op(Q) where Q comes from sytrd */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSormtr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasFillMode_t uplo,
    lwblasOperation_t trans,
    int m,
    int n,
    const float *A,
    int lda,
    const float *tau,
    const float *C,
    int ldc,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDormtr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasFillMode_t uplo,
    lwblasOperation_t trans,
    int m,
    int n,
    const double *A,
    int lda,
    const double *tau,
    const double *C,
    int ldc,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnLwnmtr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasFillMode_t uplo,
    lwblasOperation_t trans,
    int m,
    int n,
    const lwComplex *A,
    int lda,
    const lwComplex *tau,
    const lwComplex *C,
    int ldc,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZunmtr_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasFillMode_t uplo,
    lwblasOperation_t trans,
    int m,
    int n,
    const lwDoubleComplex *A,
    int lda,
    const lwDoubleComplex *tau,
    const lwDoubleComplex *C,
    int ldc,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSormtr(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasFillMode_t uplo,
    lwblasOperation_t trans,
    int m,
    int n,
    float *A,
    int lda,
    float *tau,
    float *C,
    int ldc,
    float *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDormtr(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasFillMode_t uplo,
    lwblasOperation_t trans,
    int m,
    int n,
    double *A,
    int lda,
    double *tau,
    double *C,
    int ldc,
    double *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnLwnmtr(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasFillMode_t uplo,
    lwblasOperation_t trans,
    int m,
    int n,
    lwComplex *A,
    int lda,
    lwComplex *tau,
    lwComplex *C,
    int ldc,
    lwComplex *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZunmtr(
    lwsolverDnHandle_t handle,
    lwblasSideMode_t side,
    lwblasFillMode_t uplo,
    lwblasOperation_t trans,
    int m,
    int n,
    lwDoubleComplex *A,
    int lda,
    lwDoubleComplex *tau,
    lwDoubleComplex *C,
    int ldc,
    lwDoubleComplex *work,
    int lwork,
    int *info);



/* singular value decomposition, A = U * Sigma * V^H */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSgesvd_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgesvd_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgesvd_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgesvd_bufferSize(
    lwsolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverDnSgesvd (
    lwsolverDnHandle_t handle, 
    signed char jobu, 
    signed char jobvt, 
    int m, 
    int n, 
    float *A, 
    int lda, 
    float *S, 
    float *U, 
    int ldu, 
    float *VT, 
    int ldvt, 
    float *work, 
    int lwork, 
    float *rwork, 
    int  *info );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgesvd (
    lwsolverDnHandle_t handle, 
    signed char jobu, 
    signed char jobvt, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    double *S, 
    double *U, 
    int ldu, 
    double *VT, 
    int ldvt, 
    double *work,
    int lwork, 
    double *rwork, 
    int *info );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgesvd (
    lwsolverDnHandle_t handle, 
    signed char jobu, 
    signed char jobvt, 
    int m, 
    int n, 
    lwComplex *A,
    int lda, 
    float *S, 
    lwComplex *U, 
    int ldu, 
    lwComplex *VT, 
    int ldvt,
    lwComplex *work, 
    int lwork, 
    float *rwork, 
    int *info );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgesvd (
    lwsolverDnHandle_t handle, 
    signed char jobu, 
    signed char jobvt, 
    int m, 
    int n, 
    lwDoubleComplex *A, 
    int lda, 
    double *S, 
    lwDoubleComplex *U, 
    int ldu, 
    lwDoubleComplex *VT, 
    int ldvt, 
    lwDoubleComplex *work, 
    int lwork, 
    double *rwork, 
    int *info );


/* standard symmetric eigelwalue solver, A*x = lambda*x, by divide-and-conquer  */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSsyevd_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsyevd_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    const double *A,
    int lda,
    const double *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCheevd_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    const lwComplex *A,
    int lda,
    const float *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZheevd_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    const lwDoubleComplex *A,
    int lda,
    const double *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSsyevd(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    float *A,
    int lda,
    float *W, 
    float *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsyevd(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    double *A,
    int lda,
    double *W, 
    double *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCheevd(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    lwComplex *A,
    int lda,
    float *W, 
    lwComplex *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZheevd(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    lwDoubleComplex *A,
    int lda,
    double *W, 
    lwDoubleComplex *work,
    int lwork,
    int *info);


/* generalized symmetric eigelwalue solver, A*x = lambda*B*x, by divide-and-conquer  */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSsygvd_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo, 
    int n,
    const float *A, 
    int lda,
    const float *B, 
    int ldb,
    const float *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsygvd_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype, 
    lwsolverEigMode_t jobz,  
    lwblasFillMode_t uplo,  
    int n,
    const double *A, 
    int lda,
    const double *B, 
    int ldb,
    const double *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnChegvd_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype, 
    lwsolverEigMode_t jobz,  
    lwblasFillMode_t uplo,  
    int n,
    const lwComplex *A, 
    int lda,
    const lwComplex *B, 
    int ldb,
    const float *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZhegvd_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,   
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo,  
    int n,
    const lwDoubleComplex *A,
    int lda,
    const lwDoubleComplex *B, 
    int ldb,
    const double *W,
    int *lwork);


lwsolverStatus_t LWSOLVERAPI lwsolverDnSsygvd(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,   
    lwsolverEigMode_t jobz,  
    lwblasFillMode_t uplo,  
    int n,
    float *A, 
    int lda,
    float *B, 
    int ldb,
    float *W, 
    float *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsygvd(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,  
    lwsolverEigMode_t jobz,  
    lwblasFillMode_t uplo,  
    int n,
    double *A, 
    int lda,
    double *B, 
    int ldb,
    double *W, 
    double *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnChegvd(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,   
    lwsolverEigMode_t jobz,  
    lwblasFillMode_t uplo,  
    int n,
    lwComplex *A,
    int lda,
    lwComplex *B, 
    int ldb,
    float *W, 
    lwComplex *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZhegvd(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,   
    lwsolverEigMode_t jobz,  
    lwblasFillMode_t uplo,  
    int n,
    lwDoubleComplex *A, 
    int lda,
    lwDoubleComplex *B, 
    int ldb,
    double *W, 
    lwDoubleComplex *work,
    int lwork,
    int *info);


lwsolverStatus_t LWSOLVERAPI lwsolverDnCreateSyevjInfo(
    syevjInfo_t *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDestroySyevjInfo(
    syevjInfo_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXsyevjSetTolerance(
    syevjInfo_t info,
    double tolerance);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXsyevjSetMaxSweeps(
    syevjInfo_t info,
    int max_sweeps);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXsyevjSetSortEig(
    syevjInfo_t info,
    int sort_eig);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXsyevjGetResidual(
    lwsolverDnHandle_t handle,
    syevjInfo_t info,
    double *residual);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXsyevjGetSweeps(
    lwsolverDnHandle_t handle,
    syevjInfo_t info,
    int *exelwted_sweeps);


lwsolverStatus_t LWSOLVERAPI lwsolverDnSsyevjBatched_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsyevjBatched_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    const double *A, 
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCheevjBatched_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    const lwComplex *A, 
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZheevjBatched_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    const lwDoubleComplex *A, 
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    );


lwsolverStatus_t LWSOLVERAPI lwsolverDnSsyevjBatched(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,
    int n,   
    float *A,
    int lda,
    float *W, 
    float *work,
    int lwork,
    int *info, 
    syevjInfo_t params,
    int batchSize
    );

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsyevjBatched(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo, 
    int n,
    double *A,
    int lda,
    double *W,
    double *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    );

lwsolverStatus_t LWSOLVERAPI lwsolverDnCheevjBatched(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo, 
    int n,
    lwComplex *A,
    int lda,
    float *W,
    lwComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZheevjBatched(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    lwDoubleComplex *A,
    int lda,
    double *W,
    lwDoubleComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    );


lwsolverStatus_t LWSOLVERAPI lwsolverDnSsyevj_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params);


lwsolverStatus_t LWSOLVERAPI lwsolverDnDsyevj_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params);


lwsolverStatus_t LWSOLVERAPI lwsolverDnCheevj_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,
    int n,
    const lwComplex *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params);


lwsolverStatus_t LWSOLVERAPI lwsolverDnZheevj_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo,
    int n,
    const lwDoubleComplex *A,
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params);


lwsolverStatus_t LWSOLVERAPI lwsolverDnSsyevj(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int n,
    float *A,
    int lda,
    float *W,
    float *work,
    int lwork,
    int *info,
    syevjInfo_t params);


lwsolverStatus_t LWSOLVERAPI lwsolverDnDsyevj(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *W, 
    double *work,
    int lwork,
    int *info,
    syevjInfo_t params);


lwsolverStatus_t LWSOLVERAPI lwsolverDnCheevj(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo,
    int n,
    lwComplex *A,
    int lda,
    float *W, 
    lwComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params);


lwsolverStatus_t LWSOLVERAPI lwsolverDnZheevj(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,
    int n,
    lwDoubleComplex *A,
    int lda,
    double *W, 
    lwDoubleComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params);


lwsolverStatus_t LWSOLVERAPI lwsolverDnSsygvj_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype, 
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo,
    int n,
    const float *A, 
    int lda,
    const float *B, 
    int ldb,
    const float *W,
    int *lwork,
    syevjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsygvj_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype, 
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo,
    int n,
    const double *A, 
    int lda,
    const double *B,
    int ldb,
    const double *W,
    int *lwork,
    syevjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnChegvj_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,
    int n,
    const lwComplex *A, 
    int lda,
    const lwComplex *B, 
    int ldb,
    const float *W,
    int *lwork,
    syevjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZhegvj_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,
    int n,
    const lwDoubleComplex *A, 
    int lda,
    const lwDoubleComplex *B, 
    int ldb,
    const double *W,
    int *lwork,
    syevjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSsygvj(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype, 
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo, 
    int n,
    float *A, 
    int lda,
    float *B, 
    int ldb,
    float *W,
    float *work,
    int lwork,
    int *info,
    syevjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsygvj(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype, 
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo, 
    int n,
    double *A, 
    int lda,
    double *B,
    int ldb,
    double *W, 
    double *work,
    int lwork,
    int *info,
    syevjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnChegvj(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype, 
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo,
    int n,
    lwComplex *A, 
    int lda,
    lwComplex *B, 
    int ldb,
    float *W,
    lwComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZhegvj(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype, 
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,  
    int n,
    lwDoubleComplex *A, 
    int lda,
    lwDoubleComplex *B, 
    int ldb,
    double *W, 
    lwDoubleComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params);


lwsolverStatus_t LWSOLVERAPI lwsolverDnCreateGesvdjInfo(
    gesvdjInfo_t *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDestroyGesvdjInfo(
    gesvdjInfo_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXgesvdjSetTolerance(
    gesvdjInfo_t info,
    double tolerance);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXgesvdjSetMaxSweeps(
    gesvdjInfo_t info,
    int max_sweeps);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXgesvdjSetSortEig(
    gesvdjInfo_t info,
    int sort_svd);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXgesvdjGetResidual(
    lwsolverDnHandle_t handle,
    gesvdjInfo_t info,
    double *residual);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXgesvdjGetSweeps(
    lwsolverDnHandle_t handle,
    gesvdjInfo_t info,
    int *exelwted_sweeps);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSgesvdjBatched_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int m,                
    int n,                
    const float *A,    
    int lda,           
    const float *S, 
    const float *U,   
    int ldu, 
    const float *V,
    int ldv,  
    int *lwork,
    gesvdjInfo_t params,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgesvdjBatched_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int m,
    int n,
    const double *A, 
    int lda,
    const double *S,
    const double *U,
    int ldu,
    const double *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgesvdjBatched_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int m,
    int n,
    const lwComplex *A,
    int lda,
    const float *S,
    const lwComplex *U,
    int ldu,
    const lwComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgesvdjBatched_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    int m, 
    int n, 
    const lwDoubleComplex *A,
    int lda,
    const double *S,
    const lwDoubleComplex *U,
    int ldu, 
    const lwDoubleComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSgesvdjBatched(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    int m, 
    int n, 
    float *A, 
    int lda, 
    float *S, 
    float *U,
    int ldu,
    float *V,
    int ldv, 
    float *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgesvdjBatched(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int m,
    int n,
    double *A,
    int lda,
    double *S,
    double *U,
    int ldu,
    double *V,
    int ldv, 
    double *work,
    int lwork,
    int *info, 
    gesvdjInfo_t params,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgesvdjBatched(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int m, 
    int n,
    lwComplex *A,
    int lda,
    float *S,
    lwComplex *U,
    int ldu,
    lwComplex *V,
    int ldv,
    lwComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgesvdjBatched(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int m,
    int n,
    lwDoubleComplex *A,
    int lda, 
    double *S, 
    lwDoubleComplex *U,
    int ldu,
    lwDoubleComplex *V,
    int ldv,
    lwDoubleComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSgesvdj_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    int econ,
    int m,
    int n, 
    const float *A,
    int lda,
    const float *S,
    const float *U,
    int ldu, 
    const float *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgesvdj_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    int econ,
    int m,
    int n,
    const double *A, 
    int lda,
    const double *S,
    const double *U,
    int ldu,
    const double *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgesvdj_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    int econ,
    int m,
    int n,
    const lwComplex *A,
    int lda,
    const float *S,
    const lwComplex *U,
    int ldu,
    const lwComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgesvdj_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    const lwDoubleComplex *A,
    int lda,
    const double *S,
    const lwDoubleComplex *U,
    int ldu,
    const lwDoubleComplex *V,
    int ldv, 
    int *lwork,
    gesvdjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSgesvdj(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    float *A, 
    int lda,
    float *S,
    float *U,
    int ldu,
    float *V,
    int ldv,
    float *work,
    int lwork,
    int *info,
    gesvdjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDgesvdj(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int econ, 
    int m, 
    int n, 
    double *A, 
    int lda,
    double *S,
    double *U,
    int ldu,
    double *V,
    int ldv,
    double *work,
    int lwork,
    int *info,
    gesvdjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCgesvdj(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    lwComplex *A,
    int lda,
    float *S,
    lwComplex *U,
    int ldu,
    lwComplex *V,
    int ldv,
    lwComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgesvdj(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    lwDoubleComplex *A,
    int lda,
    double *S,
    lwDoubleComplex *U, 
    int ldu, 
    lwDoubleComplex *V,
    int ldv,
    lwDoubleComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params);



#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* !defined(LWDENSE_H_) */
