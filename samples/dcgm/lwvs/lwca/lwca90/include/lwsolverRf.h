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

#if !defined(LWSOLVERRF_H_)
#define LWSOLVERRF_H_

#include "driver_types.h"
#include "lwComplex.h"   
#include "lwsolver_common.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/* LWSOLVERRF mode */
typedef enum { 
    LWSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0, //default   
    LWSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1        
} lwsolverRfResetValuesFastMode_t;

/* LWSOLVERRF matrix format */
typedef enum { 
    LWSOLVERRF_MATRIX_FORMAT_CSR = 0, //default   
    LWSOLVERRF_MATRIX_FORMAT_CSC = 1        
} lwsolverRfMatrixFormat_t;

/* LWSOLVERRF unit diagonal */
typedef enum { 
    LWSOLVERRF_UNIT_DIAGONAL_STORED_L = 0, //default   
    LWSOLVERRF_UNIT_DIAGONAL_STORED_U = 1, 
    LWSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2,        
    LWSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3        
} lwsolverRfUnitDiagonal_t;

/* LWSOLVERRF factorization algorithm */
typedef enum {
    LWSOLVERRF_FACTORIZATION_ALG0 = 0, // default
    LWSOLVERRF_FACTORIZATION_ALG1 = 1,
    LWSOLVERRF_FACTORIZATION_ALG2 = 2,
} lwsolverRfFactorization_t;

/* LWSOLVERRF triangular solve algorithm */
typedef enum {
    LWSOLVERRF_TRIANGULAR_SOLVE_ALG0 = 0, 
    LWSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1, // default
    LWSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2,
    LWSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3
} lwsolverRfTriangularSolve_t;

/* LWSOLVERRF numeric boost report */
typedef enum {
    LWSOLVERRF_NUMERIC_BOOST_NOT_USED = 0, //default
    LWSOLVERRF_NUMERIC_BOOST_USED = 1
} lwsolverRfNumericBoostReport_t;

/* Opaque structure holding LWSOLVERRF library common */
struct lwsolverRfCommon;
typedef struct lwsolverRfCommon *lwsolverRfHandle_t;

/* LWSOLVERRF create (allocate memory) and destroy (free memory) in the handle */
lwsolverStatus_t LWSOLVERAPI lwsolverRfCreate(lwsolverRfHandle_t *handle);
lwsolverStatus_t LWSOLVERAPI lwsolverRfDestroy(lwsolverRfHandle_t handle);

/* LWSOLVERRF set and get input format */
lwsolverStatus_t LWSOLVERAPI lwsolverRfGetMatrixFormat(lwsolverRfHandle_t handle, 
                                                       lwsolverRfMatrixFormat_t *format, 
                                                       lwsolverRfUnitDiagonal_t *diag);

lwsolverStatus_t LWSOLVERAPI lwsolverRfSetMatrixFormat(lwsolverRfHandle_t handle, 
                                                       lwsolverRfMatrixFormat_t format, 
                                                       lwsolverRfUnitDiagonal_t diag);
    
/* LWSOLVERRF set and get numeric properties */
lwsolverStatus_t LWSOLVERAPI lwsolverRfSetNumericProperties(lwsolverRfHandle_t handle, 
                                                            double zero,
                                                            double boost);
											 
lwsolverStatus_t LWSOLVERAPI lwsolverRfGetNumericProperties(lwsolverRfHandle_t handle, 
                                                            double* zero,
                                                            double* boost);
											 
lwsolverStatus_t LWSOLVERAPI lwsolverRfGetNumericBoostReport(lwsolverRfHandle_t handle, 
                                                             lwsolverRfNumericBoostReport_t *report);

/* LWSOLVERRF choose the triangular solve algorithm */
lwsolverStatus_t LWSOLVERAPI lwsolverRfSetAlgs(lwsolverRfHandle_t handle,
                                               lwsolverRfFactorization_t factAlg,
                                               lwsolverRfTriangularSolve_t solveAlg);

lwsolverStatus_t LWSOLVERAPI lwsolverRfGetAlgs(lwsolverRfHandle_t handle, 
                                               lwsolverRfFactorization_t* factAlg,
                                               lwsolverRfTriangularSolve_t* solveAlg);

/* LWSOLVERRF set and get fast mode */
lwsolverStatus_t LWSOLVERAPI lwsolverRfGetResetValuesFastMode(lwsolverRfHandle_t handle, 
                                                              lwsolverRfResetValuesFastMode_t *fastMode);

lwsolverStatus_t LWSOLVERAPI lwsolverRfSetResetValuesFastMode(lwsolverRfHandle_t handle, 
                                                              lwsolverRfResetValuesFastMode_t fastMode);

/*** Non-Batched Routines ***/
/* LWSOLVERRF setup of internal structures from host or device memory */
lwsolverStatus_t LWSOLVERAPI lwsolverRfSetupHost(/* Input (in the host memory) */
                                                 int n,
                                                 int nnzA,
                                                 int* h_csrRowPtrA,
                                                 int* h_csrColIndA,
                                                 double* h_csrValA,
                                                 int nnzL,
                                                 int* h_csrRowPtrL,
                                                 int* h_csrColIndL,
                                                 double* h_csrValL,
                                                 int nnzU,
                                                 int* h_csrRowPtrU,
                                                 int* h_csrColIndU,
                                                 double* h_csrValU,
                                                 int* h_P,
                                                 int* h_Q,
                                                 /* Output */
                                                 lwsolverRfHandle_t handle);
    
lwsolverStatus_t LWSOLVERAPI lwsolverRfSetupDevice(/* Input (in the device memory) */
                                                   int n,
                                                   int nnzA,
                                                   int* csrRowPtrA,
                                                   int* csrColIndA,
                                                   double* csrValA,
                                                   int nnzL,
                                                   int* csrRowPtrL,
                                                   int* csrColIndL,
                                                   double* csrValL,
                                                   int nnzU,
                                                   int* csrRowPtrU,
                                                   int* csrColIndU,
                                                   double* csrValU,
                                                   int* P,
                                                   int* Q,
                                                   /* Output */
                                                   lwsolverRfHandle_t handle);

/* LWSOLVERRF update the matrix values (assuming the reordering, pivoting 
   and consequently the sparsity pattern of L and U did not change),
   and zero out the remaining values. */
lwsolverStatus_t LWSOLVERAPI lwsolverRfResetValues(/* Input (in the device memory) */
                                                   int n,
                                                   int nnzA,
                                                   int* csrRowPtrA, 
                                                   int* csrColIndA, 
                                                   double* csrValA,
                                                   int* P,
                                                   int* Q,
                                                   /* Output */
                                                   lwsolverRfHandle_t handle);

/* LWSOLVERRF analysis (for parallelism) */
lwsolverStatus_t LWSOLVERAPI lwsolverRfAnalyze(lwsolverRfHandle_t handle);

/* LWSOLVERRF re-factorization (for parallelism) */
lwsolverStatus_t LWSOLVERAPI lwsolverRfRefactor(lwsolverRfHandle_t handle);

/* LWSOLVERRF extraction: Get L & U packed into a single matrix M */
lwsolverStatus_t LWSOLVERAPI lwsolverRfAccessBundledFactorsDevice(/* Input */
                                                                  lwsolverRfHandle_t handle,
                                                                  /* Output (in the host memory) */
                                                                  int* nnzM, 
                                                                  /* Output (in the device memory) */
                                                                  int** Mp, 
                                                                  int** Mi, 
                                                                  double** Mx);

lwsolverStatus_t LWSOLVERAPI lwsolverRfExtractBundledFactorsHost(/* Input */
                                                                 lwsolverRfHandle_t handle, 
                                                                 /* Output (in the host memory) */
                                                                 int* h_nnzM,
                                                                 int** h_Mp, 
                                                                 int** h_Mi, 
                                                                 double** h_Mx);

/* LWSOLVERRF extraction: Get L & U individually */
lwsolverStatus_t LWSOLVERAPI lwsolverRfExtractSplitFactorsHost(/* Input */
                                                               lwsolverRfHandle_t handle, 
                                                               /* Output (in the host memory) */
                                                               int* h_nnzL, 
                                                               int** h_csrRowPtrL, 
                                                               int** h_csrColIndL, 
                                                               double** h_csrValL, 
                                                               int* h_nnzU, 
                                                               int** h_csrRowPtrU, 
                                                               int** h_csrColIndU, 
                                                               double** h_csrValU);

/* LWSOLVERRF (forward and backward triangular) solves */
lwsolverStatus_t LWSOLVERAPI lwsolverRfSolve(/* Input (in the device memory) */
                                             lwsolverRfHandle_t handle,
                                             int *P,
                                             int *Q,
                                             int nrhs,     //only nrhs=1 is supported
                                             double *Temp, //of size ldt*nrhs (ldt>=n)
                                             int ldt,      
                                             /* Input/Output (in the device memory) */
                                             double *XF,
                                             /* Input */
                                             int ldxf);

/*** Batched Routines ***/
/* LWSOLVERRF-batch setup of internal structures from host */
lwsolverStatus_t LWSOLVERAPI lwsolverRfBatchSetupHost(/* Input (in the host memory)*/
                                                      int batchSize,
                                                      int n,
                                                      int nnzA,
                                                      int* h_csrRowPtrA,
                                                      int* h_csrColIndA,
                                                      double* h_csrValA_array[],
                                                      int nnzL,
                                                      int* h_csrRowPtrL,
                                                      int* h_csrColIndL,
                                                      double *h_csrValL,
                                                      int nnzU,
                                                      int* h_csrRowPtrU,
                                                      int* h_csrColIndU,
                                                      double *h_csrValU,
                                                      int* h_P,
                                                      int* h_Q,
                                                      /* Output (in the device memory) */
                                                      lwsolverRfHandle_t handle);

/* LWSOLVERRF-batch update the matrix values (assuming the reordering, pivoting 
   and consequently the sparsity pattern of L and U did not change),
   and zero out the remaining values. */
lwsolverStatus_t LWSOLVERAPI lwsolverRfBatchResetValues(/* Input (in the device memory) */
                                                        int batchSize,
                                                        int n,
                                                        int nnzA,
                                                        int* csrRowPtrA,
                                                        int* csrColIndA,
                                                        double* csrValA_array[],
                                                        int* P,
                                                        int* Q,
                                                        /* Output */
                                                        lwsolverRfHandle_t handle);
 
/* LWSOLVERRF-batch analysis (for parallelism) */
lwsolverStatus_t LWSOLVERAPI lwsolverRfBatchAnalyze(lwsolverRfHandle_t handle);

/* LWSOLVERRF-batch re-factorization (for parallelism) */
lwsolverStatus_t LWSOLVERAPI lwsolverRfBatchRefactor(lwsolverRfHandle_t handle);

/* LWSOLVERRF-batch (forward and backward triangular) solves */
lwsolverStatus_t LWSOLVERAPI lwsolverRfBatchSolve(/* Input (in the device memory) */
                                                  lwsolverRfHandle_t handle,
                                                  int *P,
                                                  int *Q,
                                                  int nrhs,     //only nrhs=1 is supported
                                                  double *Temp, //of size 2*batchSize*(n*nrhs)
                                                  int ldt,      //only ldt=n is supported
                                                  /* Input/Output (in the device memory) */
                                                  double *XF_array[],
                                                  /* Input */
                                                  int ldxf);

/* LWSOLVERRF-batch obtain the position of zero pivot */    
lwsolverStatus_t LWSOLVERAPI lwsolverRfBatchZeroPivot(/* Input */
                                                      lwsolverRfHandle_t handle,
                                                      /* Output (in the host memory) */
                                                      int *position);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* LWSOLVERRF_H_ */
