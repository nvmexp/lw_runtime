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
 
 /*   lwblasXt : Host API, Out of Core and Multi-GPU BLAS Library

 */
 
#if !defined(LWBLAS_XT_H_)
#define LWBLAS_XT_H_

#include "driver_types.h"
#include "lwComplex.h"   /* import complex data type */

#include "lwblas_v2.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

struct lwblasXtContext;
typedef struct lwblasXtContext *lwblasXtHandle_t;

lwblasStatus_t LWBLASWINAPI lwblasXtCreate(lwblasXtHandle_t *handle);
lwblasStatus_t LWBLASWINAPI lwblasXtDestroy(lwblasXtHandle_t handle);
lwblasStatus_t LWBLASWINAPI lwblasXtGetNumBoards(int nbDevices, int deviceId[], int* nbBoards);
lwblasStatus_t LWBLASWINAPI lwblasXtMaxBoards( int *nbGpuBoards );
/* This routine selects the Gpus that the user want to use for LWBLAS-XT */
lwblasStatus_t LWBLASWINAPI lwblasXtDeviceSelect(lwblasXtHandle_t handle, int nbDevices, int deviceId[]);

/* This routine allows to change the dimension of the tiles ( blockDim x blockDim ) */
lwblasStatus_t LWBLASWINAPI lwblasXtSetBlockDim(lwblasXtHandle_t handle, int blockDim);
lwblasStatus_t LWBLASWINAPI lwblasXtGetBlockDim(lwblasXtHandle_t handle, int *blockDim);

typedef enum { 
    LWBLASXT_PINNING_DISABLED   = 0,  
    LWBLASXT_PINNING_ENABLED    = 1        
} lwblasXtPinnedMemMode_t;
/* This routine allows to LWBLAS-XT to pin the Host memory if it find out that some of the matrix passed
   are not pinned : Pinning/Unpinning the Host memory is still a costly operation
   It is better if the user controls the memory on its own (by pinning/unpinning oly when necessary)
*/
lwblasStatus_t LWBLASWINAPI lwblasXtGetPinningMemMode(lwblasXtHandle_t handle, lwblasXtPinnedMemMode_t *mode);
lwblasStatus_t LWBLASWINAPI lwblasXtSetPinningMemMode(lwblasXtHandle_t handle, lwblasXtPinnedMemMode_t mode);         

/* This routines is to provide a CPU Blas routines, used for too small sizes or hybrid computation */
typedef enum
{
    LWBLASXT_FLOAT=0,
    LWBLASXT_DOUBLE=1,  
    LWBLASXT_COMPLEX=2,
    LWBLASXT_DOUBLECOMPLEX=3,        
}lwblasXtOpType_t;

typedef enum
{
    LWBLASXT_GEMM=0,
    LWBLASXT_SYRK=1,  
    LWBLASXT_HERK=2,
    LWBLASXT_SYMM=3,
    LWBLASXT_HEMM=4,
    LWBLASXT_TRSM=5,
    LWBLASXT_SYR2K=6,  
    LWBLASXT_HER2K=7,        
        
    LWBLASXT_SPMM=8,
    LWBLASXT_SYRKX=9,
    LWBLASXT_HERKX=10,  
    LWBLASXT_TRMM=11,  
    LWBLASXT_ROUTINE_MAX=12,      
}lwblasXtBlasOp_t;


/* Lwrrently only 32-bit integer BLAS routines are supported */
lwblasStatus_t LWBLASWINAPI lwblasXtSetCpuRoutine(lwblasXtHandle_t handle, lwblasXtBlasOp_t blasOp, lwblasXtOpType_t type, void *blasFunctor );

/* Specified the percentage of work that should done by the CPU, default is 0 (no work) */
lwblasStatus_t LWBLASWINAPI lwblasXtSetCpuRatio(lwblasXtHandle_t handle, lwblasXtBlasOp_t blasOp, lwblasXtOpType_t type, float ratio );


/* GEMM */
lwblasStatus_t   LWBLASWINAPI lwblasXtSgemm(lwblasXtHandle_t  handle, 
                                            lwblasOperation_t transa,
                                            lwblasOperation_t transb, 
                                            size_t m,
                                            size_t n,
                                            size_t k,
                                            const  float *alpha,
                                            const  float *A, 
                                            size_t lda,
                                            const  float *B,
                                            size_t ldb, 
                                            const  float *beta,
                                            float *C,
                                            size_t ldc);
                                            
lwblasStatus_t   LWBLASWINAPI lwblasXtDgemm(lwblasXtHandle_t  handle, 
                                            lwblasOperation_t transa,
                                            lwblasOperation_t transb, 
                                            size_t m,
                                            size_t n,
                                            size_t k,
                                            const  double *alpha,
                                            const  double *A, 
                                            size_t lda,
                                            const  double *B,
                                            size_t ldb, 
                                            const  double *beta,
                                            double *C,
                                            size_t ldc);
                                            
lwblasStatus_t   LWBLASWINAPI lwblasXtCgemm(lwblasXtHandle_t  handle, 
                                            lwblasOperation_t transa,
                                            lwblasOperation_t transb, 
                                            size_t m,
                                            size_t n,
                                            size_t k,
                                            const  lwComplex *alpha,
                                            const  lwComplex *A, 
                                            size_t lda,
                                            const  lwComplex *B,
                                            size_t ldb, 
                                            const  lwComplex *beta,
                                            lwComplex *C,
                                            size_t ldc);
                                            
lwblasStatus_t   LWBLASWINAPI lwblasXtZgemm(lwblasXtHandle_t  handle, 
                                            lwblasOperation_t transa,
                                            lwblasOperation_t transb, 
                                            size_t m,
                                            size_t n,
                                            size_t k,
                                            const  lwDoubleComplex *alpha,
                                            const  lwDoubleComplex *A, 
                                            size_t lda,
                                            const  lwDoubleComplex *B,
                                            size_t ldb, 
                                            const  lwDoubleComplex *beta,
                                            lwDoubleComplex *C,
                                            size_t ldc);                                                                                             
/* ------------------------------------------------------- */                                 
/* SYRK */
lwblasStatus_t   LWBLASWINAPI lwblasXtSsyrk( lwblasXtHandle_t handle, 
                                             lwblasFillMode_t uplo, 
                                             lwblasOperation_t trans, 
                                             size_t n,
                                             size_t k,
                                             const float *alpha,
                                             const float *A,
                                             size_t lda,
                                             const float *beta,
                                             float *C,
                                             size_t ldc );
                                             
lwblasStatus_t   LWBLASWINAPI lwblasXtDsyrk( lwblasXtHandle_t handle, 
                                             lwblasFillMode_t uplo, 
                                             lwblasOperation_t trans, 
                                             size_t n,
                                             size_t k,
                                             const double *alpha,
                                             const double *A,
                                             size_t lda,
                                             const double *beta,
                                             double *C,
                                             size_t ldc );
                                             
lwblasStatus_t   LWBLASWINAPI lwblasXtCsyrk( lwblasXtHandle_t handle, 
                                             lwblasFillMode_t uplo, 
                                             lwblasOperation_t trans, 
                                             size_t n,
                                             size_t k,
                                             const lwComplex *alpha,
                                             const lwComplex *A,
                                             size_t lda,
                                             const lwComplex *beta,
                                             lwComplex *C,
                                             size_t ldc );
                                             
lwblasStatus_t   LWBLASWINAPI lwblasXtZsyrk( lwblasXtHandle_t handle, 
                                             lwblasFillMode_t uplo, 
                                             lwblasOperation_t trans, 
                                             size_t n,
                                             size_t k,
                                             const lwDoubleComplex *alpha,
                                             const lwDoubleComplex *A,
                                             size_t lda,
                                             const lwDoubleComplex *beta,
                                             lwDoubleComplex *C,
                                             size_t ldc );
/* -------------------------------------------------------------------- */                                  
/* HERK */                                
lwblasStatus_t   LWBLASWINAPI lwblasXtCherk( lwblasXtHandle_t handle, 
                                             lwblasFillMode_t uplo, 
                                             lwblasOperation_t trans, 
                                             size_t n,
                                             size_t k,
                                             const float *alpha,
                                             const lwComplex *A,
                                             size_t lda,
                                             const float *beta,
                                             lwComplex *C,
                                             size_t ldc );
                                             
lwblasStatus_t   LWBLASWINAPI lwblasXtZherk( lwblasXtHandle_t handle, 
                                             lwblasFillMode_t uplo, 
                                             lwblasOperation_t trans, 
                                             size_t n,
                                             size_t k,
                                             const double *alpha,
                                             const lwDoubleComplex *A,
                                             size_t lda,
                                             const double *beta,
                                             lwDoubleComplex *C,
                                             size_t ldc );                                                           
/* -------------------------------------------------------------------- */                                              
/* SYR2K */                                     
lwblasStatus_t   LWBLASWINAPI lwblasXtSsyr2k( lwblasXtHandle_t handle,
                                              lwblasFillMode_t uplo,
                                              lwblasOperation_t trans,
                                              size_t n,
                                              size_t k,
                                              const float *alpha,   
                                              const float *A,
                                              size_t lda,
                                              const float *B,
                                              size_t ldb,
                                              const float *beta,   
                                              float *C,
                                              size_t ldc);  
            
lwblasStatus_t   LWBLASWINAPI lwblasXtDsyr2k( lwblasXtHandle_t handle,
                                              lwblasFillMode_t uplo,
                                              lwblasOperation_t trans,
                                              size_t n,
                                              size_t k,
                                              const double *alpha,    
                                              const double *A,
                                              size_t lda,
                                              const double *B,
                                              size_t ldb,
                                              const double *beta,   
                                              double *C,
                                              size_t ldc);
            
lwblasStatus_t   LWBLASWINAPI lwblasXtCsyr2k( lwblasXtHandle_t handle,
                                              lwblasFillMode_t uplo,
                                              lwblasOperation_t trans,
                                              size_t n,
                                              size_t k,
                                              const lwComplex *alpha,   
                                              const lwComplex *A,
                                              size_t lda,
                                              const lwComplex *B,
                                              size_t ldb,
                                              const lwComplex *beta,   
                                              lwComplex *C,
                                              size_t ldc);
            
lwblasStatus_t   LWBLASWINAPI lwblasXtZsyr2k( lwblasXtHandle_t handle,
                                              lwblasFillMode_t uplo,
                                              lwblasOperation_t trans,
                                              size_t n,
                                              size_t k,
                                              const lwDoubleComplex *alpha,
                                              const lwDoubleComplex *A,
                                              size_t lda,
                                              const lwDoubleComplex *B,
                                              size_t ldb,
                                              const lwDoubleComplex *beta,   
                                              lwDoubleComplex *C,
                                              size_t ldc);  
/* -------------------------------------------------------------------- */                                                  
/* HERKX : variant extension of HERK */                                       
lwblasStatus_t   LWBLASWINAPI lwblasXtCherkx( lwblasXtHandle_t handle,
                                              lwblasFillMode_t uplo,
                                              lwblasOperation_t trans,
                                              size_t n,
                                              size_t k,
                                              const lwComplex *alpha,   
                                              const lwComplex *A,
                                              size_t lda,
                                              const lwComplex *B,
                                              size_t ldb,
                                              const float *beta,     
                                              lwComplex *C,
                                              size_t ldc);  
            
lwblasStatus_t   LWBLASWINAPI lwblasXtZherkx( lwblasXtHandle_t handle,
                                              lwblasFillMode_t uplo,
                                              lwblasOperation_t trans, 
                                              size_t n,
                                              size_t k,
                                              const lwDoubleComplex *alpha,  
                                              const lwDoubleComplex *A, 
                                              size_t lda,
                                              const lwDoubleComplex *B,
                                              size_t ldb,
                                              const double *beta,   
                                              lwDoubleComplex *C,
                                              size_t ldc);       
                         
/* -------------------------------------------------------------------- */                                
/* TRSM */                                                                         
lwblasStatus_t   LWBLASWINAPI lwblasXtStrsm( lwblasXtHandle_t handle, 
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             lwblasOperation_t trans,
                                             lwblasDiagType_t diag,
                                             size_t m,
                                             size_t n,
                                             const float *alpha,
                                             const float *A,
                                             size_t lda,
                                             float *B,
                                             size_t ldb);
    

lwblasStatus_t   LWBLASWINAPI lwblasXtDtrsm( lwblasXtHandle_t handle,
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             lwblasOperation_t trans,
                                             lwblasDiagType_t diag,
                                             size_t m,
                                             size_t n,
                                             const double *alpha, 
                                             const double *A, 
                                             size_t lda, 
                                             double *B,
                                             size_t ldb);
    
lwblasStatus_t   LWBLASWINAPI lwblasXtCtrsm( lwblasXtHandle_t handle,
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             lwblasOperation_t trans,
                                             lwblasDiagType_t diag,
                                             size_t m,
                                             size_t n,
                                             const lwComplex *alpha, 
                                             const lwComplex *A,
                                             size_t lda,
                                             lwComplex *B,
                                             size_t ldb);
                  
lwblasStatus_t   LWBLASWINAPI lwblasXtZtrsm( lwblasXtHandle_t handle, 
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             lwblasOperation_t trans,
                                             lwblasDiagType_t diag,
                                             size_t m,
                                             size_t n,
                                             const lwDoubleComplex *alpha, 
                                             const lwDoubleComplex *A,                                        
                                             size_t lda,
                                             lwDoubleComplex *B,
                                             size_t ldb);       
/* -------------------------------------------------------------------- */                                
/* SYMM : Symmetric Multiply Matrix*/                                                                         
lwblasStatus_t   LWBLASWINAPI lwblasXtSsymm( lwblasXtHandle_t handle, 
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             size_t m,
                                             size_t n,
                                             const float *alpha,
                                             const float *A,
                                             size_t lda,
                                             const float *B,
                                             size_t ldb,
                                             const float *beta,
                                             float *C,
                                             size_t ldc );    

lwblasStatus_t   LWBLASWINAPI lwblasXtDsymm( lwblasXtHandle_t handle,
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             size_t m,
                                             size_t n,
                                             const double *alpha, 
                                             const double *A, 
                                             size_t lda,
                                             const double *B,
                                             size_t ldb,
                                             const double *beta,
                                             double *C,
                                             size_t ldc );                                 
    
lwblasStatus_t   LWBLASWINAPI lwblasXtCsymm( lwblasXtHandle_t handle,
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             size_t m,
                                             size_t n,
                                             const lwComplex *alpha, 
                                             const lwComplex *A,
                                             size_t lda,
                                             const lwComplex *B,
                                             size_t ldb,
                                             const lwComplex *beta,
                                             lwComplex *C,
                                             size_t ldc );                                 
                  
lwblasStatus_t   LWBLASWINAPI lwblasXtZsymm( lwblasXtHandle_t handle, 
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             size_t m,
                                             size_t n,
                                             const lwDoubleComplex *alpha, 
                                             const lwDoubleComplex *A,  
                                             size_t lda,                                      
                                             const lwDoubleComplex *B,
                                             size_t ldb,
                                             const lwDoubleComplex *beta,
                                             lwDoubleComplex *C,
                                             size_t ldc );  
/* -------------------------------------------------------------------- */                                         
/* HEMM : Hermitian Matrix Multiply */                                       
 lwblasStatus_t  LWBLASWINAPI lwblasXtChemm( lwblasXtHandle_t handle,
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             size_t m,
                                             size_t n,
                                             const lwComplex *alpha, 
                                             const lwComplex *A,
                                             size_t lda,
                                             const lwComplex *B,
                                             size_t ldb,
                                             const lwComplex *beta,
                                             lwComplex *C,
                                             size_t ldc );                                 
                  
lwblasStatus_t   LWBLASWINAPI lwblasXtZhemm( lwblasXtHandle_t handle, 
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             size_t m,
                                             size_t n,
                                             const lwDoubleComplex *alpha, 
                                             const lwDoubleComplex *A,  
                                             size_t lda,                                      
                                             const lwDoubleComplex *B,
                                             size_t ldb,
                                             const lwDoubleComplex *beta,
                                             lwDoubleComplex *C,
                                             size_t ldc );  

/* -------------------------------------------------------------------- */ 
/* SYRKX : variant extension of SYRK  */                                     
lwblasStatus_t   LWBLASWINAPI lwblasXtSsyrkx( lwblasXtHandle_t handle,
                                              lwblasFillMode_t uplo,
                                              lwblasOperation_t trans,
                                              size_t n,
                                              size_t k,
                                              const float *alpha,   
                                              const float *A,
                                              size_t lda,
                                              const float *B,
                                              size_t ldb,
                                              const float *beta,   
                                              float *C,
                                              size_t ldc);  
            
lwblasStatus_t   LWBLASWINAPI lwblasXtDsyrkx( lwblasXtHandle_t handle,
                                              lwblasFillMode_t uplo,
                                              lwblasOperation_t trans,
                                              size_t n,
                                              size_t k,
                                              const double *alpha,    
                                              const double *A,
                                              size_t lda,
                                              const double *B,
                                              size_t ldb,
                                              const double *beta,   
                                              double *C,
                                              size_t ldc);
            
lwblasStatus_t   LWBLASWINAPI lwblasXtCsyrkx( lwblasXtHandle_t handle,
                                              lwblasFillMode_t uplo,
                                              lwblasOperation_t trans,
                                              size_t n,
                                              size_t k,
                                              const lwComplex *alpha,   
                                              const lwComplex *A,
                                              size_t lda,
                                              const lwComplex *B,
                                              size_t ldb,
                                              const lwComplex *beta,   
                                              lwComplex *C,
                                              size_t ldc);
            
lwblasStatus_t   LWBLASWINAPI lwblasXtZsyrkx( lwblasXtHandle_t handle,
                                              lwblasFillMode_t uplo,
                                              lwblasOperation_t trans,
                                              size_t n,
                                              size_t k,
                                              const lwDoubleComplex *alpha,
                                              const lwDoubleComplex *A,
                                              size_t lda,
                                              const lwDoubleComplex *B,
                                              size_t ldb,
                                              const lwDoubleComplex *beta,   
                                              lwDoubleComplex *C,
                                              size_t ldc);  
/* -------------------------------------------------------------------- */                                          
/* HER2K : variant extension of HERK  */                                    
lwblasStatus_t   LWBLASWINAPI lwblasXtCher2k( lwblasXtHandle_t handle,
                                              lwblasFillMode_t uplo,
                                              lwblasOperation_t trans,
                                              size_t n,
                                              size_t k,
                                              const lwComplex *alpha,   
                                              const lwComplex *A,
                                              size_t lda,
                                              const lwComplex *B,
                                              size_t ldb,
                                              const float *beta,     
                                              lwComplex *C,
                                              size_t ldc);  
            
lwblasStatus_t   LWBLASWINAPI lwblasXtZher2k( lwblasXtHandle_t handle,
                                              lwblasFillMode_t uplo,
                                              lwblasOperation_t trans, 
                                              size_t n,
                                              size_t k,
                                              const lwDoubleComplex *alpha,  
                                              const lwDoubleComplex *A, 
                                              size_t lda,
                                              const lwDoubleComplex *B,
                                              size_t ldb,
                                              const double *beta,   
                                              lwDoubleComplex *C,
                                              size_t ldc);       
                         
                                
/* -------------------------------------------------------------------- */                                              
/* SPMM : Symmetric Packed Multiply Matrix*/                                                                         
lwblasStatus_t   LWBLASWINAPI lwblasXtSspmm( lwblasXtHandle_t handle, 
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             size_t m,
                                             size_t n,
                                             const float *alpha,
                                             const float *AP,
                                             const float *B,
                                             size_t ldb,
                                             const float *beta,
                                             float *C,
                                             size_t ldc );    

lwblasStatus_t   LWBLASWINAPI lwblasXtDspmm( lwblasXtHandle_t handle,
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             size_t m,
                                             size_t n,
                                             const double *alpha, 
                                             const double *AP, 
                                             const double *B,
                                             size_t ldb,
                                             const double *beta,
                                             double *C,
                                             size_t ldc );                                 
    
lwblasStatus_t   LWBLASWINAPI lwblasXtCspmm( lwblasXtHandle_t handle,
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             size_t m,
                                             size_t n,
                                             const lwComplex *alpha, 
                                             const lwComplex *AP,
                                             const lwComplex *B,
                                             size_t ldb,
                                             const lwComplex *beta,
                                             lwComplex *C,
                                             size_t ldc );                                 
                  
lwblasStatus_t   LWBLASWINAPI lwblasXtZspmm( lwblasXtHandle_t handle, 
                                             lwblasSideMode_t side,
                                             lwblasFillMode_t uplo,
                                             size_t m,
                                             size_t n,
                                             const lwDoubleComplex *alpha, 
                                             const lwDoubleComplex *AP,                                        
                                             const lwDoubleComplex *B,
                                             size_t ldb,
                                             const lwDoubleComplex *beta,
                                             lwDoubleComplex *C,
                                             size_t ldc );                                                                                                       
                                             
/* -------------------------------------------------------------------- */   
/* TRMM */                                                                                    
lwblasStatus_t LWBLASWINAPI lwblasXtStrmm( lwblasXtHandle_t handle,
                                           lwblasSideMode_t side,
                                           lwblasFillMode_t uplo, 
                                           lwblasOperation_t trans,
                                           lwblasDiagType_t diag,                               
                                           size_t m,
                                           size_t n,
                                           const float *alpha, 
                                           const float *A,
                                           size_t lda,
                                           const float *B,
                                           size_t ldb,
                                           float *C,
                                           size_t ldc );

lwblasStatus_t LWBLASWINAPI lwblasXtDtrmm( lwblasXtHandle_t handle,
                                           lwblasSideMode_t side,
                                           lwblasFillMode_t uplo, 
                                           lwblasOperation_t trans,
                                           lwblasDiagType_t diag,                               
                                           size_t m,
                                           size_t n,
                                           const double *alpha, 
                                           const double *A,
                                           size_t lda,
                                           const double *B,
                                           size_t ldb,
                                           double *C,
                                           size_t ldc );

lwblasStatus_t LWBLASWINAPI lwblasXtCtrmm( lwblasXtHandle_t handle,
                                           lwblasSideMode_t side,
                                           lwblasFillMode_t uplo, 
                                           lwblasOperation_t trans,
                                           lwblasDiagType_t diag,                               
                                           size_t m,
                                           size_t n,
                                           const lwComplex *alpha, 
                                           const lwComplex *A,
                                           size_t lda,
                                           const lwComplex *B,
                                           size_t ldb,
                                           lwComplex *C,
                                           size_t ldc );

lwblasStatus_t LWBLASWINAPI lwblasXtZtrmm( lwblasXtHandle_t handle,
                                           lwblasSideMode_t side,
                                           lwblasFillMode_t uplo, 
                                           lwblasOperation_t trans,
                                           lwblasDiagType_t diag,                               
                                           size_t m,
                                           size_t n,
                                           const lwDoubleComplex *alpha, 
                                           const lwDoubleComplex *A,
                                           size_t lda,
                                           const lwDoubleComplex *B,
                                           size_t ldb,
                                           lwDoubleComplex *C,
                                           size_t ldc );
                                             
                                
#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* !defined(LWBLAS_XT_H_) */
