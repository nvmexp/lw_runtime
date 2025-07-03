/*
 * Copyright 2015 LWPU Corporation.  All rights reserved.
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

#if !defined(LWSOLVERSP_LOWLEVEL_PREVIEW_H_)
#define LWSOLVERSP_LOWLEVEL_PREVIEW_H_

#include "lwsolverSp.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */


struct csrluInfoHost;
typedef struct csrluInfoHost *csrluInfoHost_t;


struct csrqrInfoHost;
typedef struct csrqrInfoHost *csrqrInfoHost_t;


struct csrcholInfoHost;
typedef struct csrcholInfoHost *csrcholInfoHost_t;


struct csrcholInfo;
typedef struct csrcholInfo *csrcholInfo_t;



/*
 * Low level API for CPU LU
 * 
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpCreateCsrluInfoHost(
    csrluInfoHost_t *info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDestroyCsrluInfoHost(
    csrluInfoHost_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrluAnalysisHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrluBufferInfoHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrluBufferInfoHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrluBufferInfoHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrluBufferInfoHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrluFactorHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    float pivot_threshold,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrluFactorHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    double pivot_threshold,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrluFactorHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    float pivot_threshold,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrluFactorHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    double pivot_threshold,
    void *pBuffer);


lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrluZeroPivotHost(
    lwsolverSpHandle_t handle,
    csrluInfoHost_t info,
    float tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrluZeroPivotHost(
    lwsolverSpHandle_t handle,
    csrluInfoHost_t info,
    double tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrluZeroPivotHost(
    lwsolverSpHandle_t handle,
    csrluInfoHost_t info,
    float tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrluZeroPivotHost(
    lwsolverSpHandle_t handle,
    csrluInfoHost_t info,
    double tol,
    int *position);


lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrluSolveHost(
    lwsolverSpHandle_t handle,
    int n,
    const float *b,
    float *x,
    csrluInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrluSolveHost(
    lwsolverSpHandle_t handle,
    int n,
    const double *b,
    double *x,
    csrluInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrluSolveHost(
    lwsolverSpHandle_t handle,
    int n,
    const lwComplex *b,
    lwComplex *x,
    csrluInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrluSolveHost(
    lwsolverSpHandle_t handle,
    int n,
    const lwDoubleComplex *b,
    lwDoubleComplex *x,
    csrluInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrluNnzHost(
    lwsolverSpHandle_t handle,
    int *nnzLRef,
    int *nnzURef,
    csrluInfoHost_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrluExtractHost(
    lwsolverSpHandle_t handle,
    int *P,
    int *Q,
    const lwsparseMatDescr_t descrL,
    float *csrValL,
    int *csrRowPtrL,
    int *csrColIndL,
    const lwsparseMatDescr_t descrU,
    float *csrValU,
    int *csrRowPtrU,
    int *csrColIndU,
    csrluInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrluExtractHost(
    lwsolverSpHandle_t handle,
    int *P,
    int *Q,
    const lwsparseMatDescr_t descrL,
    double *csrValL,
    int *csrRowPtrL,
    int *csrColIndL,
    const lwsparseMatDescr_t descrU,
    double *csrValU,
    int *csrRowPtrU,
    int *csrColIndU,
    csrluInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrluExtractHost(
    lwsolverSpHandle_t handle,
    int *P,
    int *Q,
    const lwsparseMatDescr_t descrL,
    lwComplex *csrValL,
    int *csrRowPtrL,
    int *csrColIndL,
    const lwsparseMatDescr_t descrU,
    lwComplex *csrValU,
    int *csrRowPtrU,
    int *csrColIndU,
    csrluInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrluExtractHost(
    lwsolverSpHandle_t handle,
    int *P,
    int *Q,
    const lwsparseMatDescr_t descrL,
    lwDoubleComplex *csrValL,
    int *csrRowPtrL,
    int *csrColIndL,
    const lwsparseMatDescr_t descrU,
    lwDoubleComplex *csrValU,
    int *csrRowPtrU,
    int *csrColIndU,
    csrluInfoHost_t info,
    void *pBuffer);


/*
 * Low level API for CPU QR
 *
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpCreateCsrqrInfoHost(
    csrqrInfoHost_t *info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDestroyCsrqrInfoHost(
    csrqrInfoHost_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrqrAnalysisHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrqrBufferInfoHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrqrBufferInfoHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrqrBufferInfoHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrqrBufferInfoHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrqrSetupHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float mu,
    csrqrInfoHost_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrqrSetupHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    double mu,
    csrqrInfoHost_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrqrSetupHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    lwComplex mu,
    csrqrInfoHost_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrqrSetupHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    lwDoubleComplex mu,
    csrqrInfoHost_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrqrFactorHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    float *b,
    float *x,
    csrqrInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrqrFactorHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    double *b,
    double *x,
    csrqrInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrqrFactorHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    lwComplex *b,
    lwComplex *x,
    csrqrInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrqrFactorHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    lwDoubleComplex *b,
    lwDoubleComplex *x,
    csrqrInfoHost_t info,
    void *pBuffer);


lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrqrZeroPivotHost(
    lwsolverSpHandle_t handle,
    csrqrInfoHost_t info,
    float tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrqrZeroPivotHost(
    lwsolverSpHandle_t handle,
    csrqrInfoHost_t info,
    double tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrqrZeroPivotHost(
    lwsolverSpHandle_t handle,
    csrqrInfoHost_t info,
    float tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrqrZeroPivotHost(
    lwsolverSpHandle_t handle,
    csrqrInfoHost_t info,
    double tol,
    int *position);


lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrqrSolveHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    float *b,
    float *x,
    csrqrInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrqrSolveHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    double *b,
    double *x,
    csrqrInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrqrSolveHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    lwComplex *b,
    lwComplex *x,
    csrqrInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrqrSolveHost(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    lwDoubleComplex *b,
    lwDoubleComplex *x,
    csrqrInfoHost_t info,
    void *pBuffer);


/*
 * Low level API for GPU QR
 *
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrqrAnalysis(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrqrBufferInfo(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrqrBufferInfo(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrqrBufferInfo(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrqrBufferInfo(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrqrSetup(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float mu,
    csrqrInfo_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrqrSetup(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    double mu,
    csrqrInfo_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrqrSetup(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    lwComplex mu,
    csrqrInfo_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrqrSetup(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    lwDoubleComplex mu,
    csrqrInfo_t info);


lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrqrFactor(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    float *b,
    float *x,
    csrqrInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrqrFactor(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    double *b,
    double *x,
    csrqrInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrqrFactor(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    lwComplex *b,
    lwComplex *x,
    csrqrInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrqrFactor(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    lwDoubleComplex *b,
    lwDoubleComplex *x,
    csrqrInfo_t info,
    void *pBuffer);


lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrqrZeroPivot(
    lwsolverSpHandle_t handle,
    csrqrInfo_t info,
    float tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrqrZeroPivot(
    lwsolverSpHandle_t handle,
    csrqrInfo_t info,
    double tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrqrZeroPivot(
    lwsolverSpHandle_t handle,
    csrqrInfo_t info,
    float tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrqrZeroPivot(
    lwsolverSpHandle_t handle,
    csrqrInfo_t info,
    double tol,
    int *position);


lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrqrSolve(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    float *b,
    float *x,
    csrqrInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrqrSolve(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    double *b,
    double *x,
    csrqrInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrqrSolve(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    lwComplex *b,
    lwComplex *x,
    csrqrInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrqrSolve(
    lwsolverSpHandle_t handle,
    int m,
    int n,
    lwDoubleComplex *b,
    lwDoubleComplex *x,
    csrqrInfo_t info,
    void *pBuffer);


/*
 * Low level API for CPU Cholesky
 * 
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpCreateCsrcholInfoHost(
    csrcholInfoHost_t *info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDestroyCsrcholInfoHost(
    csrcholInfoHost_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrcholAnalysisHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrcholBufferInfoHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrcholBufferInfoHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrcholBufferInfoHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrcholBufferInfoHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);


lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrcholFactorHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrcholFactorHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrcholFactorHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrcholFactorHost(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrcholZeroPivotHost(
    lwsolverSpHandle_t handle,
    csrcholInfoHost_t info,
    float tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrcholZeroPivotHost(
    lwsolverSpHandle_t handle,
    csrcholInfoHost_t info,
    double tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrcholZeroPivotHost(
    lwsolverSpHandle_t handle,
    csrcholInfoHost_t info,
    float tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrcholZeroPivotHost(
    lwsolverSpHandle_t handle,
    csrcholInfoHost_t info,
    double tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrcholSolveHost(
    lwsolverSpHandle_t handle,
    int n,
    const float *b,
    float *x,
    csrcholInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrcholSolveHost(
    lwsolverSpHandle_t handle,
    int n,
    const double *b,
    double *x,
    csrcholInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrcholSolveHost(
    lwsolverSpHandle_t handle,
    int n,
    const lwComplex *b,
    lwComplex *x,
    csrcholInfoHost_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrcholSolveHost(
    lwsolverSpHandle_t handle,
    int n,
    const lwDoubleComplex *b,
    lwDoubleComplex *x,
    csrcholInfoHost_t info,
    void *pBuffer);

/*
 * Low level API for GPU Cholesky
 * 
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpCreateCsrcholInfo(
    csrcholInfo_t *info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDestroyCsrcholInfo(
    csrcholInfo_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpXcsrcholAnalysis(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrcholBufferInfo(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrcholBufferInfo(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrcholBufferInfo(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrcholBufferInfo(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrcholFactor(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrcholFactor(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrcholFactor(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrcholFactor(
    lwsolverSpHandle_t handle,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrcholZeroPivot(
    lwsolverSpHandle_t handle,
    csrcholInfo_t info,
    float tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrcholZeroPivot(
    lwsolverSpHandle_t handle,
    csrcholInfo_t info,
    double tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrcholZeroPivot(
    lwsolverSpHandle_t handle,
    csrcholInfo_t info,
    float tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrcholZeroPivot(
    lwsolverSpHandle_t handle,
    csrcholInfo_t info,
    double tol,
    int *position);

lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrcholSolve(
    lwsolverSpHandle_t handle,
    int n,
    const float *b,
    float *x,
    csrcholInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrcholSolve(
    lwsolverSpHandle_t handle,
    int n,
    const double *b,
    double *x,
    csrcholInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrcholSolve(
    lwsolverSpHandle_t handle,
    int n,
    const lwComplex *b,
    lwComplex *x,
    csrcholInfo_t info,
    void *pBuffer);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrcholSolve(
    lwsolverSpHandle_t handle,
    int n,
    const lwDoubleComplex *b,
    lwDoubleComplex *x,
    csrcholInfo_t info,
    void *pBuffer);

/*
 * "diag" is a device array of size N.
 * lwsolverSp<t>csrcholDiag returns diag(L) to "diag" where A(P,P) = L*L**T
 * "diag" can estimate det(A) because det(A(P,P)) = det(A) = det(L)^2 if A = L*L**T.
 * 
 * lwsolverSp<t>csrcholDiag must be called after lwsolverSp<t>csrcholFactor.
 * otherwise "diag" is wrong.
 */
lwsolverStatus_t LWSOLVERAPI lwsolverSpScsrcholDiag(
    lwsolverSpHandle_t handle,
    csrcholInfo_t info,
    float *diag);

lwsolverStatus_t LWSOLVERAPI lwsolverSpDcsrcholDiag(
    lwsolverSpHandle_t handle,
    csrcholInfo_t info,
    double *diag);

lwsolverStatus_t LWSOLVERAPI lwsolverSpCcsrcholDiag(
    lwsolverSpHandle_t handle,
    csrcholInfo_t info,
    float *diag);

lwsolverStatus_t LWSOLVERAPI lwsolverSpZcsrcholDiag(
    lwsolverSpHandle_t handle,
    csrcholInfo_t info,
    double *diag);





#if defined(__cplusplus)
}
#endif /* __cplusplus */



#endif // LWSOLVERSP_LOWLEVEL_PREVIEW_H_


