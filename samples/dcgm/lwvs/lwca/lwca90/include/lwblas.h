/*
 * Copyright 1993-2014 LWPU Corporation.  All rights reserved.
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
 
/*
 * This is the public header file for the LWBLAS library, defining the API
 *
 * LWBLAS is an implementation of BLAS (Basic Linear Algebra Subroutines) 
 * on top of the LWCA runtime. 
 */

#if !defined(LWBLAS_H_)
#define LWBLAS_H_

#include <lwda_runtime.h>

#ifndef LWBLASWINAPI
#ifdef _WIN32
#define LWBLASWINAPI __stdcall
#else
#define LWBLASWINAPI 
#endif
#endif

#undef LWBLASAPI
#ifdef __LWDACC__
#define LWBLASAPI __host__
#else
#define LWBLASAPI
#endif

#include "lwblas_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* LWBLAS data types */
#define lwblasStatus lwblasStatus_t

lwblasStatus LWBLASWINAPI lwblasInit (void);
lwblasStatus LWBLASWINAPI lwblasShutdown (void);
lwblasStatus LWBLASWINAPI lwblasGetError (void);

lwblasStatus LWBLASWINAPI lwblasGetVersion(int *version);
lwblasStatus LWBLASWINAPI lwblasAlloc (int n, int elemSize, void **devicePtr);

lwblasStatus LWBLASWINAPI lwblasFree (void *devicePtr);


lwblasStatus LWBLASWINAPI lwblasSetKernelStream (lwdaStream_t stream);



/* ---------------- LWBLAS BLAS1 functions ---------------- */
/* NRM2 */
float LWBLASWINAPI lwblasSnrm2 (int n, const float *x, int incx);
double LWBLASWINAPI lwblasDnrm2 (int n, const double *x, int incx);
float LWBLASWINAPI lwblasScnrm2 (int n, const lwComplex *x, int incx);
double LWBLASWINAPI lwblasDznrm2 (int n, const lwDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* DOT */
float LWBLASWINAPI lwblasSdot (int n, const float *x, int incx, const float *y, 
                               int incy);
double LWBLASWINAPI lwblasDdot (int n, const double *x, int incx, const double *y, 
                               int incy);
lwComplex LWBLASWINAPI lwblasCdotu (int n, const lwComplex *x, int incx, const lwComplex *y, 
                               int incy);
lwComplex LWBLASWINAPI lwblasCdotc (int n, const lwComplex *x, int incx, const lwComplex *y, 
                               int incy);
lwDoubleComplex LWBLASWINAPI lwblasZdotu (int n, const lwDoubleComplex *x, int incx, const lwDoubleComplex *y, 
                               int incy);
lwDoubleComplex LWBLASWINAPI lwblasZdotc (int n, const lwDoubleComplex *x, int incx, const lwDoubleComplex *y, 
                               int incy);
/*------------------------------------------------------------------------*/
/* SCAL */
void LWBLASWINAPI lwblasSscal (int n, float alpha, float *x, int incx);
void LWBLASWINAPI lwblasDscal (int n, double alpha, double *x, int incx);
void LWBLASWINAPI lwblasCscal (int n, lwComplex alpha, lwComplex *x, int incx);
void LWBLASWINAPI lwblasZscal (int n, lwDoubleComplex alpha, lwDoubleComplex *x, int incx);

void LWBLASWINAPI lwblasCsscal (int n, float alpha, lwComplex *x, int incx);
void LWBLASWINAPI lwblasZdscal (int n, double alpha, lwDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* AXPY */
void LWBLASWINAPI lwblasSaxpy (int n, float alpha, const float *x, int incx, 
                               float *y, int incy);
void LWBLASWINAPI lwblasDaxpy (int n, double alpha, const double *x, 
                               int incx, double *y, int incy);
void LWBLASWINAPI lwblasCaxpy (int n, lwComplex alpha, const lwComplex *x, 
                               int incx, lwComplex *y, int incy);
void LWBLASWINAPI lwblasZaxpy (int n, lwDoubleComplex alpha, const lwDoubleComplex *x, 
                               int incx, lwDoubleComplex *y, int incy);
/*------------------------------------------------------------------------*/
/* COPY */
void LWBLASWINAPI lwblasScopy (int n, const float *x, int incx, float *y, 
                               int incy);
void LWBLASWINAPI lwblasDcopy (int n, const double *x, int incx, double *y, 
                               int incy);
void LWBLASWINAPI lwblasCcopy (int n, const lwComplex *x, int incx, lwComplex *y,
                               int incy);
void LWBLASWINAPI lwblasZcopy (int n, const lwDoubleComplex *x, int incx, lwDoubleComplex *y,
                               int incy);
/*------------------------------------------------------------------------*/
/* SWAP */
void LWBLASWINAPI lwblasSswap (int n, float *x, int incx, float *y, int incy);
void LWBLASWINAPI lwblasDswap (int n, double *x, int incx, double *y, int incy);
void LWBLASWINAPI lwblasCswap (int n, lwComplex *x, int incx, lwComplex *y, int incy);
void LWBLASWINAPI lwblasZswap (int n, lwDoubleComplex *x, int incx, lwDoubleComplex *y, int incy);           
/*------------------------------------------------------------------------*/
/* AMAX */
int LWBLASWINAPI lwblasIsamax (int n, const float *x, int incx);
int LWBLASWINAPI lwblasIdamax (int n, const double *x, int incx);
int LWBLASWINAPI lwblasIcamax (int n, const lwComplex *x, int incx);
int LWBLASWINAPI lwblasIzamax (int n, const lwDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* AMIN */
int LWBLASWINAPI lwblasIsamin (int n, const float *x, int incx);
int LWBLASWINAPI lwblasIdamin (int n, const double *x, int incx);

int LWBLASWINAPI lwblasIcamin (int n, const lwComplex *x, int incx);
int LWBLASWINAPI lwblasIzamin (int n, const lwDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* ASUM */
float LWBLASWINAPI lwblasSasum (int n, const float *x, int incx);
double LWBLASWINAPI lwblasDasum (int n, const double *x, int incx);
float LWBLASWINAPI lwblasScasum (int n, const lwComplex *x, int incx);
double LWBLASWINAPI lwblasDzasum (int n, const lwDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* ROT */
void LWBLASWINAPI lwblasSrot (int n, float *x, int incx, float *y, int incy, 
                              float sc, float ss);
void LWBLASWINAPI lwblasDrot (int n, double *x, int incx, double *y, int incy, 
                              double sc, double ss);
void LWBLASWINAPI lwblasCrot (int n, lwComplex *x, int incx, lwComplex *y, 
                              int incy, float c, lwComplex s);
void LWBLASWINAPI lwblasZrot (int n, lwDoubleComplex *x, int incx, 
                              lwDoubleComplex *y, int incy, double sc, 
                              lwDoubleComplex cs);
void LWBLASWINAPI lwblasCsrot (int n, lwComplex *x, int incx, lwComplex *y,
                               int incy, float c, float s);
void LWBLASWINAPI lwblasZdrot (int n, lwDoubleComplex *x, int incx, 
                               lwDoubleComplex *y, int incy, double c, double s);
/*------------------------------------------------------------------------*/
/* ROTG */
void LWBLASWINAPI lwblasSrotg (float *sa, float *sb, float *sc, float *ss);
void LWBLASWINAPI lwblasDrotg (double *sa, double *sb, double *sc, double *ss);
void LWBLASWINAPI lwblasCrotg (lwComplex *ca, lwComplex cb, float *sc,
                               lwComplex *cs);                                     
void LWBLASWINAPI lwblasZrotg (lwDoubleComplex *ca, lwDoubleComplex cb, double *sc,
                               lwDoubleComplex *cs);                                                               
/*------------------------------------------------------------------------*/
/* ROTM */
void LWBLASWINAPI lwblasSrotm(int n, float *x, int incx, float *y, int incy, 
                              const float* sparam);
void LWBLASWINAPI lwblasDrotm(int n, double *x, int incx, double *y, int incy, 
                              const double* sparam);
/*------------------------------------------------------------------------*/
/* ROTMG */
void LWBLASWINAPI lwblasSrotmg (float *sd1, float *sd2, float *sx1, 
                                const float *sy1, float* sparam);
void LWBLASWINAPI lwblasDrotmg (double *sd1, double *sd2, double *sx1, 
                                const double *sy1, double* sparam);
                           
/* --------------- LWBLAS BLAS2 functions  ---------------- */
/* GEMV */
void LWBLASWINAPI lwblasSgemv (char trans, int m, int n, float alpha,
                               const float *A, int lda, const float *x, int incx,
                               float beta, float *y, int incy);
void LWBLASWINAPI lwblasDgemv (char trans, int m, int n, double alpha,
                               const double *A, int lda, const double *x, int incx,
                               double beta, double *y, int incy);
void LWBLASWINAPI lwblasCgemv (char trans, int m, int n, lwComplex alpha,
                               const lwComplex *A, int lda, const lwComplex *x, int incx,
                               lwComplex beta, lwComplex *y, int incy);
void LWBLASWINAPI lwblasZgemv (char trans, int m, int n, lwDoubleComplex alpha,
                               const lwDoubleComplex *A, int lda, const lwDoubleComplex *x, int incx,
                               lwDoubleComplex beta, lwDoubleComplex *y, int incy);
/*------------------------------------------------------------------------*/
/* GBMV */
void LWBLASWINAPI lwblasSgbmv (char trans, int m, int n, int kl, int ku, 
                               float alpha, const float *A, int lda, 
                               const float *x, int incx, float beta, float *y, 
                               int incy);
void LWBLASWINAPI lwblasDgbmv (char trans, int m, int n, int kl, int ku, 
                               double alpha, const double *A, int lda, 
                               const double *x, int incx, double beta, double *y, 
                               int incy);
void LWBLASWINAPI lwblasCgbmv (char trans, int m, int n, int kl, int ku, 
                               lwComplex alpha, const lwComplex *A, int lda, 
                               const lwComplex *x, int incx, lwComplex beta, lwComplex *y, 
                               int incy);
void LWBLASWINAPI lwblasZgbmv (char trans, int m, int n, int kl, int ku, 
                               lwDoubleComplex alpha, const lwDoubleComplex *A, int lda, 
                               const lwDoubleComplex *x, int incx, lwDoubleComplex beta, lwDoubleComplex *y, 
                               int incy);                  
/*------------------------------------------------------------------------*/
/* TRMV */
void LWBLASWINAPI lwblasStrmv (char uplo, char trans, char diag, int n, 
                               const float *A, int lda, float *x, int incx);
void LWBLASWINAPI lwblasDtrmv (char uplo, char trans, char diag, int n, 
                               const double *A, int lda, double *x, int incx);
void LWBLASWINAPI lwblasCtrmv (char uplo, char trans, char diag, int n, 
                               const lwComplex *A, int lda, lwComplex *x, int incx);
void LWBLASWINAPI lwblasZtrmv (char uplo, char trans, char diag, int n, 
                               const lwDoubleComplex *A, int lda, lwDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* TBMV */
void LWBLASWINAPI lwblasStbmv (char uplo, char trans, char diag, int n, int k, 
                               const float *A, int lda, float *x, int incx);
void LWBLASWINAPI lwblasDtbmv (char uplo, char trans, char diag, int n, int k, 
                               const double *A, int lda, double *x, int incx);
void LWBLASWINAPI lwblasCtbmv (char uplo, char trans, char diag, int n, int k, 
                               const lwComplex *A, int lda, lwComplex *x, int incx);
void LWBLASWINAPI lwblasZtbmv (char uplo, char trans, char diag, int n, int k, 
                               const lwDoubleComplex *A, int lda, lwDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* TPMV */                                                    
void LWBLASWINAPI lwblasStpmv(char uplo, char trans, char diag, int n, const float *AP, float *x, int incx);

void LWBLASWINAPI lwblasDtpmv(char uplo, char trans, char diag, int n, const double *AP, double *x, int incx);

void LWBLASWINAPI lwblasCtpmv(char uplo, char trans, char diag, int n, const lwComplex *AP, lwComplex *x, int incx);
                                         
void LWBLASWINAPI lwblasZtpmv(char uplo, char trans, char diag, int n, const lwDoubleComplex *AP, lwDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/
/* TRSV */
void LWBLASWINAPI lwblasStrsv(char uplo, char trans, char diag, int n, const float *A, int lda, float *x, int incx);

void LWBLASWINAPI lwblasDtrsv(char uplo, char trans, char diag, int n, const double *A, int lda, double *x, int incx);

void LWBLASWINAPI lwblasCtrsv(char uplo, char trans, char diag, int n, const lwComplex *A, int lda, lwComplex *x, int incx);

void LWBLASWINAPI lwblasZtrsv(char uplo, char trans, char diag, int n, const lwDoubleComplex *A, int lda, 
                              lwDoubleComplex *x, int incx);       
/*------------------------------------------------------------------------*/
/* TPSV */
void LWBLASWINAPI lwblasStpsv(char uplo, char trans, char diag, int n, const float *AP, 
                              float *x, int incx);
                                                                                                            
void LWBLASWINAPI lwblasDtpsv(char uplo, char trans, char diag, int n, const double *AP, double *x, int incx);

void LWBLASWINAPI lwblasCtpsv(char uplo, char trans, char diag, int n, const lwComplex *AP, lwComplex *x, int incx);

void LWBLASWINAPI lwblasZtpsv(char uplo, char trans, char diag, int n, const lwDoubleComplex *AP, 
                              lwDoubleComplex *x, int incx);
/*------------------------------------------------------------------------*/                                         
/* TBSV */                                         
void LWBLASWINAPI lwblasStbsv(char uplo, char trans, 
                              char diag, int n, int k, const float *A, 
                              int lda, float *x, int incx);
    
void LWBLASWINAPI lwblasDtbsv(char uplo, char trans, 
                              char diag, int n, int k, const double *A, 
                              int lda, double *x, int incx);
void LWBLASWINAPI lwblasCtbsv(char uplo, char trans, 
                              char diag, int n, int k, const lwComplex *A, 
                              int lda, lwComplex *x, int incx);      
                                         
void LWBLASWINAPI lwblasZtbsv(char uplo, char trans, 
                              char diag, int n, int k, const lwDoubleComplex *A, 
                              int lda, lwDoubleComplex *x, int incx);  
/*------------------------------------------------------------------------*/                                         
/* SYMV/HEMV */
void LWBLASWINAPI lwblasSsymv (char uplo, int n, float alpha, const float *A,
                               int lda, const float *x, int incx, float beta, 
                               float *y, int incy);
void LWBLASWINAPI lwblasDsymv (char uplo, int n, double alpha, const double *A,
                               int lda, const double *x, int incx, double beta, 
                               double *y, int incy);
void LWBLASWINAPI lwblasChemv (char uplo, int n, lwComplex alpha, const lwComplex *A,
                               int lda, const lwComplex *x, int incx, lwComplex beta, 
                               lwComplex *y, int incy);
void LWBLASWINAPI lwblasZhemv (char uplo, int n, lwDoubleComplex alpha, const lwDoubleComplex *A,
                               int lda, const lwDoubleComplex *x, int incx, lwDoubleComplex beta, 
                               lwDoubleComplex *y, int incy);
/*------------------------------------------------------------------------*/       
/* SBMV/HBMV */
void LWBLASWINAPI lwblasSsbmv (char uplo, int n, int k, float alpha, 
                               const float *A, int lda, const float *x, int incx, 
                               float beta, float *y, int incy);
void LWBLASWINAPI lwblasDsbmv (char uplo, int n, int k, double alpha, 
                               const double *A, int lda, const double *x, int incx, 
                               double beta, double *y, int incy);
void LWBLASWINAPI lwblasChbmv (char uplo, int n, int k, lwComplex alpha, 
                               const lwComplex *A, int lda, const lwComplex *x, int incx, 
                               lwComplex beta, lwComplex *y, int incy);
void LWBLASWINAPI lwblasZhbmv (char uplo, int n, int k, lwDoubleComplex alpha, 
                               const lwDoubleComplex *A, int lda, const lwDoubleComplex *x, int incx, 
                               lwDoubleComplex beta, lwDoubleComplex *y, int incy);
/*------------------------------------------------------------------------*/       
/* SPMV/HPMV */
void LWBLASWINAPI lwblasSspmv(char uplo, int n, float alpha,
                              const float *AP, const float *x,
                              int incx, float beta, float *y, int incy);
void LWBLASWINAPI lwblasDspmv(char uplo, int n, double alpha,
                              const double *AP, const double *x,
                              int incx, double beta, double *y, int incy);
void LWBLASWINAPI lwblasChpmv(char uplo, int n, lwComplex alpha,
                              const lwComplex *AP, const lwComplex *x,
                              int incx, lwComplex beta, lwComplex *y, int incy);
void LWBLASWINAPI lwblasZhpmv(char uplo, int n, lwDoubleComplex alpha,
                              const lwDoubleComplex *AP, const lwDoubleComplex *x,
                              int incx, lwDoubleComplex beta, lwDoubleComplex *y, int incy);

/*------------------------------------------------------------------------*/       
/* GER */
void LWBLASWINAPI lwblasSger (int m, int n, float alpha, const float *x, int incx,
                              const float *y, int incy, float *A, int lda);
void LWBLASWINAPI lwblasDger (int m, int n, double alpha, const double *x, int incx,
                              const double *y, int incy, double *A, int lda);

void LWBLASWINAPI lwblasCgeru (int m, int n, lwComplex alpha, const lwComplex *x,
                               int incx, const lwComplex *y, int incy,
                               lwComplex *A, int lda);
void LWBLASWINAPI lwblasCgerc (int m, int n, lwComplex alpha, const lwComplex *x,
                               int incx, const lwComplex *y, int incy,
                               lwComplex *A, int lda);
void LWBLASWINAPI lwblasZgeru (int m, int n, lwDoubleComplex alpha, const lwDoubleComplex *x,
                               int incx, const lwDoubleComplex *y, int incy,
                               lwDoubleComplex *A, int lda);
void LWBLASWINAPI lwblasZgerc (int m, int n, lwDoubleComplex alpha, const lwDoubleComplex *x,
                               int incx, const lwDoubleComplex *y, int incy,
                               lwDoubleComplex *A, int lda);
/*------------------------------------------------------------------------*/       
/* SYR/HER */
void LWBLASWINAPI lwblasSsyr (char uplo, int n, float alpha, const float *x,
                              int incx, float *A, int lda);
void LWBLASWINAPI lwblasDsyr (char uplo, int n, double alpha, const double *x,
                              int incx, double *A, int lda);

void LWBLASWINAPI lwblasCher (char uplo, int n, float alpha, 
                              const lwComplex *x, int incx, lwComplex *A, int lda);
void LWBLASWINAPI lwblasZher (char uplo, int n, double alpha, 
                              const lwDoubleComplex *x, int incx, lwDoubleComplex *A, int lda);

/*------------------------------------------------------------------------*/       
/* SPR/HPR */
void LWBLASWINAPI lwblasSspr (char uplo, int n, float alpha, const float *x,
                              int incx, float *AP);
void LWBLASWINAPI lwblasDspr (char uplo, int n, double alpha, const double *x,
                              int incx, double *AP);
void LWBLASWINAPI lwblasChpr (char uplo, int n, float alpha, const lwComplex *x,
                              int incx, lwComplex *AP);
void LWBLASWINAPI lwblasZhpr (char uplo, int n, double alpha, const lwDoubleComplex *x,
                              int incx, lwDoubleComplex *AP);
/*------------------------------------------------------------------------*/       
/* SYR2/HER2 */
void LWBLASWINAPI lwblasSsyr2 (char uplo, int n, float alpha, const float *x, 
                               int incx, const float *y, int incy, float *A, 
                               int lda);
void LWBLASWINAPI lwblasDsyr2 (char uplo, int n, double alpha, const double *x, 
                               int incx, const double *y, int incy, double *A, 
                               int lda);
void LWBLASWINAPI lwblasCher2 (char uplo, int n, lwComplex alpha, const lwComplex *x, 
                               int incx, const lwComplex *y, int incy, lwComplex *A, 
                               int lda);
void LWBLASWINAPI lwblasZher2 (char uplo, int n, lwDoubleComplex alpha, const lwDoubleComplex *x, 
                               int incx, const lwDoubleComplex *y, int incy, lwDoubleComplex *A, 
                               int lda);

/*------------------------------------------------------------------------*/       
/* SPR2/HPR2 */
void LWBLASWINAPI lwblasSspr2 (char uplo, int n, float alpha, const float *x, 
                               int incx, const float *y, int incy, float *AP);
void LWBLASWINAPI lwblasDspr2 (char uplo, int n, double alpha,
                               const double *x, int incx, const double *y,
                               int incy, double *AP);
void LWBLASWINAPI lwblasChpr2 (char uplo, int n, lwComplex alpha,
                               const lwComplex *x, int incx, const lwComplex *y,
                               int incy, lwComplex *AP);
void LWBLASWINAPI lwblasZhpr2 (char uplo, int n, lwDoubleComplex alpha,
                               const lwDoubleComplex *x, int incx, const lwDoubleComplex *y,
                               int incy, lwDoubleComplex *AP);
/* ------------------------BLAS3 Functions ------------------------------- */
/* GEMM */
void LWBLASWINAPI lwblasSgemm (char transa, char transb, int m, int n, int k, 
                               float alpha, const float *A, int lda, 
                               const float *B, int ldb, float beta, float *C, 
                               int ldc);
void LWBLASWINAPI lwblasDgemm (char transa, char transb, int m, int n, int k,
                               double alpha, const double *A, int lda, 
                               const double *B, int ldb, double beta, double *C, 
                               int ldc);              
void LWBLASWINAPI lwblasCgemm (char transa, char transb, int m, int n, int k, 
                               lwComplex alpha, const lwComplex *A, int lda,
                               const lwComplex *B, int ldb, lwComplex beta,
                               lwComplex *C, int ldc);
void LWBLASWINAPI lwblasZgemm (char transa, char transb, int m, int n,
                               int k, lwDoubleComplex alpha,
                               const lwDoubleComplex *A, int lda,
                               const lwDoubleComplex *B, int ldb,
                               lwDoubleComplex beta, lwDoubleComplex *C,
                               int ldc);                   
/* -------------------------------------------------------*/
/* SYRK */
void LWBLASWINAPI lwblasSsyrk (char uplo, char trans, int n, int k, float alpha, 
                               const float *A, int lda, float beta, float *C, 
                               int ldc);
void LWBLASWINAPI lwblasDsyrk (char uplo, char trans, int n, int k,
                               double alpha, const double *A, int lda,
                               double beta, double *C, int ldc);

void LWBLASWINAPI lwblasCsyrk (char uplo, char trans, int n, int k,
                               lwComplex alpha, const lwComplex *A, int lda,
                               lwComplex beta, lwComplex *C, int ldc);
void LWBLASWINAPI lwblasZsyrk (char uplo, char trans, int n, int k,
                               lwDoubleComplex alpha,
                               const lwDoubleComplex *A, int lda,
                               lwDoubleComplex beta,
                               lwDoubleComplex *C, int ldc);
/* ------------------------------------------------------- */
/* HERK */
void LWBLASWINAPI lwblasCherk (char uplo, char trans, int n, int k,
                               float alpha, const lwComplex *A, int lda,
                               float beta, lwComplex *C, int ldc);
void LWBLASWINAPI lwblasZherk (char uplo, char trans, int n, int k,
                               double alpha,
                               const lwDoubleComplex *A, int lda,
                               double beta,
                               lwDoubleComplex *C, int ldc);
/* ------------------------------------------------------- */
/* SYR2K */
void LWBLASWINAPI lwblasSsyr2k (char uplo, char trans, int n, int k, float alpha, 
                                const float *A, int lda, const float *B, int ldb, 
                                float beta, float *C, int ldc);

void LWBLASWINAPI lwblasDsyr2k (char uplo, char trans, int n, int k,
                                double alpha, const double *A, int lda,
                                const double *B, int ldb, double beta,
                                double *C, int ldc);
void LWBLASWINAPI lwblasCsyr2k (char uplo, char trans, int n, int k,
                                lwComplex alpha, const lwComplex *A, int lda,
                                const lwComplex *B, int ldb, lwComplex beta,
                                lwComplex *C, int ldc);

void LWBLASWINAPI lwblasZsyr2k (char uplo, char trans, int n, int k,
                                lwDoubleComplex alpha, const lwDoubleComplex *A, int lda,
                                const lwDoubleComplex *B, int ldb, lwDoubleComplex beta,
                                lwDoubleComplex *C, int ldc);                             
/* ------------------------------------------------------- */
/* HER2K */
void LWBLASWINAPI lwblasCher2k (char uplo, char trans, int n, int k,
                                lwComplex alpha, const lwComplex *A, int lda,
                                const lwComplex *B, int ldb, float beta,
                                lwComplex *C, int ldc);

void LWBLASWINAPI lwblasZher2k (char uplo, char trans, int n, int k,
                                lwDoubleComplex alpha, const lwDoubleComplex *A, int lda,
                                const lwDoubleComplex *B, int ldb, double beta,
                                lwDoubleComplex *C, int ldc); 

/*------------------------------------------------------------------------*/       
/* SYMM*/
void LWBLASWINAPI lwblasSsymm (char side, char uplo, int m, int n, float alpha, 
                               const float *A, int lda, const float *B, int ldb,
                               float beta, float *C, int ldc);
void LWBLASWINAPI lwblasDsymm (char side, char uplo, int m, int n, double alpha, 
                               const double *A, int lda, const double *B, int ldb,
                               double beta, double *C, int ldc);
          
void LWBLASWINAPI lwblasCsymm (char side, char uplo, int m, int n, lwComplex alpha, 
                               const lwComplex *A, int lda, const lwComplex *B, int ldb,
                               lwComplex beta, lwComplex *C, int ldc);
          
void LWBLASWINAPI lwblasZsymm (char side, char uplo, int m, int n, lwDoubleComplex alpha, 
                               const lwDoubleComplex *A, int lda, const lwDoubleComplex *B, int ldb,
                               lwDoubleComplex beta, lwDoubleComplex *C, int ldc);
/*------------------------------------------------------------------------*/       
/* HEMM*/
void LWBLASWINAPI lwblasChemm (char side, char uplo, int m, int n,
                               lwComplex alpha, const lwComplex *A, int lda,
                               const lwComplex *B, int ldb, lwComplex beta,
                               lwComplex *C, int ldc);
void LWBLASWINAPI lwblasZhemm (char side, char uplo, int m, int n,
                               lwDoubleComplex alpha, const lwDoubleComplex *A, int lda,
                               const lwDoubleComplex *B, int ldb, lwDoubleComplex beta,
                               lwDoubleComplex *C, int ldc);  

/*------------------------------------------------------------------------*/       
/* TRSM*/
void LWBLASWINAPI lwblasStrsm (char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, const float *A, int lda,
                               float *B, int ldb);

void LWBLASWINAPI lwblasDtrsm (char side, char uplo, char transa,
                               char diag, int m, int n, double alpha,
                               const double *A, int lda, double *B,
                               int ldb);

void LWBLASWINAPI lwblasCtrsm (char side, char uplo, char transa, char diag,
                               int m, int n, lwComplex alpha, const lwComplex *A,
                               int lda, lwComplex *B, int ldb);

void LWBLASWINAPI lwblasZtrsm (char side, char uplo, char transa,
                               char diag, int m, int n, lwDoubleComplex alpha,
                               const lwDoubleComplex *A, int lda,
                               lwDoubleComplex *B, int ldb);                                                        
/*------------------------------------------------------------------------*/       
/* TRMM*/
void LWBLASWINAPI lwblasStrmm (char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, const float *A, int lda,
                               float *B, int ldb);
void LWBLASWINAPI lwblasDtrmm (char side, char uplo, char transa,
                               char diag, int m, int n, double alpha,
                               const double *A, int lda, double *B,
                               int ldb);
void LWBLASWINAPI lwblasCtrmm (char side, char uplo, char transa, char diag,
                               int m, int n, lwComplex alpha, const lwComplex *A,
                               int lda, lwComplex *B, int ldb);
void LWBLASWINAPI lwblasZtrmm (char side, char uplo, char transa,
                               char diag, int m, int n, lwDoubleComplex alpha,
                               const lwDoubleComplex *A, int lda, lwDoubleComplex *B,
                               int ldb);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* !defined(LWBLAS_H_) */
