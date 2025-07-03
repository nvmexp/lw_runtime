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

struct lwsolverDnContext;
typedef struct lwsolverDnContext *lwsolverDnHandle_t;

struct syevjInfo;
typedef struct syevjInfo *syevjInfo_t;

struct gesvdjInfo;
typedef struct gesvdjInfo *gesvdjInfo_t;


//------------------------------------------------------
// opaque lwsolverDnIRS structure for IRS solver
struct lwsolverDnIRSParams;
typedef struct lwsolverDnIRSParams* lwsolverDnIRSParams_t;

struct lwsolverDnIRSInfos;
typedef struct lwsolverDnIRSInfos* lwsolverDnIRSInfos_t;
//------------------------------------------------------

struct lwsolverDnParams;
typedef struct lwsolverDnParams *lwsolverDnParams_t;

typedef enum {
   LWSOLVERDN_GETRF = 0
} lwsolverDnFunction_t ;



#include "lwComplex.h"   /* import complex data type */
#include "lwblas_v2.h"
#include "lwsolver_common.h"



/*******************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif

lwsolverStatus_t LWSOLVERAPI lwsolverDnCreate(lwsolverDnHandle_t *handle);
lwsolverStatus_t LWSOLVERAPI lwsolverDnDestroy(lwsolverDnHandle_t handle);
lwsolverStatus_t LWSOLVERAPI lwsolverDnSetStream (lwsolverDnHandle_t handle, lwdaStream_t streamId);
lwsolverStatus_t LWSOLVERAPI lwsolverDnGetStream(lwsolverDnHandle_t handle, lwdaStream_t *streamId);

//============================================================
// IRS headers 
//============================================================

// =============================================================================
// IRS helper function API
// =============================================================================
lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSParamsCreate(
            lwsolverDnIRSParams_t* params_ptr );

lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSParamsDestroy(
            lwsolverDnIRSParams_t params );

lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSParamsSetRefinementSolver(
            lwsolverDnIRSParams_t params,
            lwsolverIRSRefinement_t refinement_solver );

lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSParamsSetSolverMainPrecision(
            lwsolverDnIRSParams_t params,
            lwsolverPrecType_t solver_main_precision ); 

lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSParamsSetSolverLowestPrecision(
            lwsolverDnIRSParams_t params,
            lwsolverPrecType_t solver_lowest_precision );

lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSParamsSetSolverPrecisions(
            lwsolverDnIRSParams_t params,
            lwsolverPrecType_t solver_main_precision,
            lwsolverPrecType_t solver_lowest_precision );

lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSParamsSetTol(
            lwsolverDnIRSParams_t params,
            double val );

lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSParamsSetTolInner(
            lwsolverDnIRSParams_t params,
            double val );

lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSParamsSetMaxIters(
            lwsolverDnIRSParams_t params,
            lwsolver_int_t maxiters );

lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSParamsSetMaxItersInner(
            lwsolverDnIRSParams_t params,
            lwsolver_int_t maxiters_inner );

lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSParamsGetMaxIters(
            lwsolverDnIRSParams_t params,
            lwsolver_int_t *maxiters );

lwsolverStatus_t LWSOLVERAPI
lwsolverDnIRSParamsEnableFallback(
    lwsolverDnIRSParams_t params );

lwsolverStatus_t LWSOLVERAPI
lwsolverDnIRSParamsDisableFallback(
    lwsolverDnIRSParams_t params );


// =============================================================================
// lwsolverDnIRSInfos prototypes
// =============================================================================
lwsolverStatus_t LWSOLVERAPI 
    lwsolverDnIRSInfosDestroy(
        lwsolverDnIRSInfos_t infos );

lwsolverStatus_t LWSOLVERAPI 
    lwsolverDnIRSInfosCreate(
        lwsolverDnIRSInfos_t* infos_ptr );

lwsolverStatus_t LWSOLVERAPI 
    lwsolverDnIRSInfosGetNiters(
            lwsolverDnIRSInfos_t infos,
            lwsolver_int_t *niters );

lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSInfosGetOuterNiters(
            lwsolverDnIRSInfos_t infos,
            lwsolver_int_t *outer_niters );

lwsolverStatus_t LWSOLVERAPI 
    lwsolverDnIRSInfosRequestResidual(
        lwsolverDnIRSInfos_t infos );

lwsolverStatus_t LWSOLVERAPI 
    lwsolverDnIRSInfosGetResidualHistory(
            lwsolverDnIRSInfos_t infos,
            void **residual_history );

lwsolverStatus_t LWSOLVERAPI
    lwsolverDnIRSInfosGetMaxIters(
            lwsolverDnIRSInfos_t infos,
            lwsolver_int_t *maxiters );

//============================================================
//  IRS functions API
//============================================================

/*******************************************************************************//*
 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gesv 
 * users API Prototypes */
/*******************************************************************************/
lwsolverStatus_t LWSOLVERAPI lwsolverDnZZgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZCgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZKgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZEgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZYgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCCgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCEgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCKgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCYgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDDgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDSgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDHgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDBgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDXgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSSgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSHgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSBgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSXgesv(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

/*******************************************************************************/


/*******************************************************************************//*
 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gesv_bufferSize 
 * users API Prototypes */
/*******************************************************************************/
lwsolverStatus_t LWSOLVERAPI lwsolverDnZZgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZCgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZKgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZEgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZYgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCCgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCKgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCEgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCYgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDDgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDSgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDHgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDBgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDXgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSSgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSHgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSBgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSXgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        lwsolver_int_t *dipiv,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);
/*******************************************************************************/


/*******************************************************************************//*
 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gels 
 * users API Prototypes */
/*******************************************************************************/
lwsolverStatus_t LWSOLVERAPI lwsolverDnZZgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZCgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZKgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZEgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZYgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCCgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCKgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCEgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCYgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDDgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDSgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDHgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDBgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDXgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSSgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSHgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSBgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSXgels(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *iter,
        lwsolver_int_t *d_info);
/*******************************************************************************/

/*******************************************************************************//*
 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gels_bufferSize 
 * API prototypes */
/*******************************************************************************/
lwsolverStatus_t LWSOLVERAPI lwsolverDnZZgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZCgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZKgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZEgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZYgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwDoubleComplex *dA, lwsolver_int_t ldda,
        lwDoubleComplex *dB, lwsolver_int_t lddb,
        lwDoubleComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCCgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCKgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCEgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCYgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        lwComplex *dA, lwsolver_int_t ldda,
        lwComplex *dB, lwsolver_int_t lddb,
        lwComplex *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDDgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDSgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDHgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDBgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDXgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        double *dA, lwsolver_int_t ldda,
        double *dB, lwsolver_int_t lddb,
        double *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSSgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSHgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSBgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSXgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        float *dA, lwsolver_int_t ldda,
        float *dB, lwsolver_int_t lddb,
        float *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);
/*******************************************************************************/



/*******************************************************************************//*
 * expert users API for IRS Prototypes
 * */
/*******************************************************************************/
lwsolverStatus_t LWSOLVERAPI lwsolverDnIRSXgesv(
        lwsolverDnHandle_t handle,
        lwsolverDnIRSParams_t gesv_irs_params,
        lwsolverDnIRSInfos_t  gesv_irs_infos,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        void *dA, lwsolver_int_t ldda,
        void *dB, lwsolver_int_t lddb,
        void *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *niters,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnIRSXgesv_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolverDnIRSParams_t params,
        lwsolver_int_t n, lwsolver_int_t nrhs,
        size_t *lwork_bytes);


lwsolverStatus_t LWSOLVERAPI lwsolverDnIRSXgels(
        lwsolverDnHandle_t handle,
        lwsolverDnIRSParams_t gels_irs_params,
        lwsolverDnIRSInfos_t  gels_irs_infos,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs,
        void *dA, lwsolver_int_t ldda,
        void *dB, lwsolver_int_t lddb,
        void *dX, lwsolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        lwsolver_int_t *niters,
        lwsolver_int_t *d_info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnIRSXgels_bufferSize(
        lwsolverDnHandle_t handle,
        lwsolverDnIRSParams_t params,
        lwsolver_int_t m, 
        lwsolver_int_t n, 
        lwsolver_int_t nrhs, 
        size_t *lwork_bytes);
/*******************************************************************************/


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

/* s.p.d. matrix ilwersion (POTRI) and auxiliary routines (TRTRI and LAUUM)  */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSpotri_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDpotri_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCpotri_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwComplex *A,
    int lda,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZpotri_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwDoubleComplex *A,
    int lda,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSpotri(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDpotri(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCpotri(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwComplex *A,
    int lda,
    lwComplex *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZpotri(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwDoubleComplex *A,
    int lda,
    lwDoubleComplex *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnStrtri_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    lwblasDiagType_t diag,
    int n,
    float *A,
    int lda,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDtrtri_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    lwblasDiagType_t diag,
    int n,
    double *A,
    int lda,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCtrtri_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    lwblasDiagType_t diag,
    int n,
    lwComplex *A,
    int lda,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZtrtri_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    lwblasDiagType_t diag,
    int n,
    lwDoubleComplex *A,
    int lda,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnStrtri(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    lwblasDiagType_t diag,
    int n,
    float *A,
    int lda,
    float *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDtrtri(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    lwblasDiagType_t diag,
    int n,
    double *A,
    int lda,
    double *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCtrtri(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    lwblasDiagType_t diag,
    int n,
    lwComplex *A,
    int lda,
    lwComplex *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZtrtri(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    lwblasDiagType_t diag,
    int n,
    lwDoubleComplex *A,
    int lda,
    lwDoubleComplex *work,
    int lwork,
    int *devInfo);

/* lauum, auxiliar routine for s.p.d matrix ilwersion */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSlauum_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDlauum_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnClauum_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwComplex *A,
    int lda,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZlauum_bufferSize(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwDoubleComplex *A,
    int lda,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSlauum(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDlauum(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnClauum(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwComplex *A,
    int lda,
    lwComplex *work,
    int lwork,
    int *devInfo);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZlauum(
    lwsolverDnHandle_t handle,
    lwblasFillMode_t uplo,
    int n,
    lwDoubleComplex *A,
    int lda,
    lwDoubleComplex *work,
    int lwork,
    int *devInfo);



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

/* Symmetric indefinite solve (SYTRS) */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSsytrs_bufferSize(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        int nrhs,
        const float *A,
        int lda,
        const int *ipiv,
        float *B,
        int ldb,
        int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsytrs_bufferSize(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        int nrhs,
        const double *A,
        int lda,
        const int *ipiv,
        double *B,
        int ldb,
        int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCsytrs_bufferSize(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        int nrhs,
        const lwComplex *A,
        int lda,
        const int *ipiv,
        lwComplex *B,
        int ldb,
        int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZsytrs_bufferSize(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        int nrhs,
        const lwDoubleComplex *A,
        int lda,
        const int *ipiv,
        lwDoubleComplex *B,
        int ldb,
        int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSsytrs(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        int nrhs,
        const float *A,
        int lda,
        const int *ipiv,
        float *B,
        int ldb,
        float *work,
        int lwork,
        int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsytrs(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        int nrhs,
        const double *A,
        int lda,
        const int *ipiv,
        double *B,
        int ldb,
        double *work,
        int lwork,
        int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCsytrs(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        int nrhs,
        const lwComplex *A,
        int lda,
        const int *ipiv,
        lwComplex *B,
        int ldb,
        lwComplex *work,
        int lwork,
        int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZsytrs(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        int nrhs,
        const lwDoubleComplex *A,
        int lda,
        const int *ipiv,
        lwDoubleComplex *B,
        int ldb,
        lwDoubleComplex *work,
        int lwork,
        int *info);

/* Symmetric indefinite ilwersion (sytri) */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSsytri_bufferSize(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        float *A,
        int lda,
        const int *ipiv,
        int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsytri_bufferSize(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        double *A,
        int lda,
        const int *ipiv,
        int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCsytri_bufferSize(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        lwComplex *A,
        int lda,
        const int *ipiv,
        int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZsytri_bufferSize(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        lwDoubleComplex *A,
        int lda,
        const int *ipiv,
        int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSsytri(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        float *A,
        int lda,
        const int *ipiv,
        float *work,
        int lwork,
        int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsytri(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        double *A,
        int lda,
        const int *ipiv,
        double *work,
        int lwork,
        int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCsytri(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        lwComplex *A,
        int lda,
        const int *ipiv,
        lwComplex *work,
        int lwork,
        int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZsytri(
        lwsolverDnHandle_t handle,
        lwblasFillMode_t uplo,
        int n,
        lwDoubleComplex *A,
        int lda,
        const int *ipiv,
        lwDoubleComplex *work,
        int lwork,
        int *info);


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

/* standard selective symmetric eigelwalue solver, A*x = lambda*x, by divide-and-conquer  */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSsyevdx_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo, 
    int n,
    const float *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsyevdx_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo, 
    int n,
    const double *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCheevdx_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo, 
    int n,
    const lwComplex *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZheevdx_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo, 
    int n,
    const lwDoubleComplex *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSsyevdx(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo, 
    int n,
    float *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W, 
    float *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsyevdx(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo, 
    int n,
    double *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W, 
    double *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCheevdx(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo, 
    int n,
    lwComplex *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W, 
    lwComplex *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZheevdx(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo, 
    int n,
    lwDoubleComplex *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W, 
    lwDoubleComplex *work,
    int lwork,
    int *info);

/* selective generalized symmetric eigelwalue solver, A*x = lambda*B*x, by divide-and-conquer  */
lwsolverStatus_t LWSOLVERAPI lwsolverDnSsygvdx_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,
    lwsolverEigMode_t jobz,
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo, 
    int n,
    const float *A, 
    int lda,
    const float *B, 
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsygvdx_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype, 
    lwsolverEigMode_t jobz,  
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo,  
    int n,
    const double *A, 
    int lda,
    const double *B, 
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnChegvdx_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype, 
    lwsolverEigMode_t jobz,  
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo,  
    int n,
    const lwComplex *A, 
    int lda,
    const lwComplex *B, 
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZhegvdx_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,   
    lwsolverEigMode_t jobz, 
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo,  
    int n,
    const lwDoubleComplex *A,
    int lda,
    const lwDoubleComplex *B, 
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork);


lwsolverStatus_t LWSOLVERAPI lwsolverDnSsygvdx(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,   
    lwsolverEigMode_t jobz,  
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo,  
    int n,
    float *A, 
    int lda,
    float *B, 
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W, 
    float *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDsygvdx(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,  
    lwsolverEigMode_t jobz,  
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo,  
    int n,
    double *A, 
    int lda,
    double *B, 
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W, 
    double *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnChegvdx(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,   
    lwsolverEigMode_t jobz,  
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo,  
    int n,
    lwComplex *A,
    int lda,
    lwComplex *B, 
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W, 
    lwComplex *work,
    int lwork,
    int *info);

lwsolverStatus_t LWSOLVERAPI lwsolverDnZhegvdx(
    lwsolverDnHandle_t handle,
    lwsolverEigType_t itype,   
    lwsolverEigMode_t jobz,  
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo,  
    int n,
    lwDoubleComplex *A, 
    int lda,
    lwDoubleComplex *B, 
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
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


/* batched approximate SVD */

lwsolverStatus_t LWSOLVERAPI lwsolverDnSgesvdaStridedBatched_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const float *d_A, 
    int lda,
    long long int strideA, 
    const float *d_S, 
    long long int strideS, 
    const float *d_U, 
    int ldu,
    long long int strideU, 
    const float *d_V, 
    int ldv,
    long long int strideV,
    int *lwork,
    int batchSize
    );


lwsolverStatus_t LWSOLVERAPI lwsolverDnDgesvdaStridedBatched_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const double *d_A, 
    int lda,
    long long int strideA, 
    const double *d_S,   
    long long int strideS, 
    const double *d_U,  
    int ldu,
    long long int strideU, 
    const double *d_V,
    int ldv,
    long long int strideV, 
    int *lwork,
    int batchSize
    );


lwsolverStatus_t LWSOLVERAPI lwsolverDnCgesvdaStridedBatched_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const lwComplex *d_A, 
    int lda,
    long long int strideA, 
    const float *d_S, 
    long long int strideS, 
    const lwComplex *d_U,
    int ldu,
    long long int strideU, 
    const lwComplex *d_V, 
    int ldv,
    long long int strideV, 
    int *lwork,
    int batchSize
    );

lwsolverStatus_t LWSOLVERAPI lwsolverDnZgesvdaStridedBatched_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const lwDoubleComplex *d_A,
    int lda,
    long long int strideA,
    const double *d_S, 
    long long int strideS, 
    const lwDoubleComplex *d_U, 
    int ldu,
    long long int strideU,
    const lwDoubleComplex *d_V,
    int ldv,
    long long int strideV, 
    int *lwork,
    int batchSize
    );


lwsolverStatus_t LWSOLVERAPI lwsolverDnSgesvdaStridedBatched(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    int rank, 
    int m,   
    int n,  
    const float *d_A, 
    int lda, 
    long long int strideA,
    float *d_S, 
    long long int strideS, 
    float *d_U, 
    int ldu, 
    long long int strideU,
    float *d_V, 
    int ldv,    
    long long int strideV, 
    float *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF,
    int batchSize);


lwsolverStatus_t LWSOLVERAPI lwsolverDnDgesvdaStridedBatched(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    int rank,
    int m, 
    int n, 
    const double *d_A,
    int lda,  
    long long int strideA, 
    double *d_S, 
    long long int strideS,
    double *d_U, 
    int ldu, 
    long long int strideU, 
    double *d_V, 
    int ldv, 
    long long int strideV,
    double *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF, 
    int batchSize);


lwsolverStatus_t LWSOLVERAPI lwsolverDnCgesvdaStridedBatched(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    int rank,  
    int m, 
    int n, 
    const lwComplex *d_A, 
    int lda,
    long long int strideA,
    float *d_S,
    long long int strideS,
    lwComplex *d_U, 
    int ldu,   
    long long int strideU,  
    lwComplex *d_V, 
    int ldv, 
    long long int strideV,
    lwComplex *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF, 
    int batchSize);


lwsolverStatus_t LWSOLVERAPI lwsolverDnZgesvdaStridedBatched(
    lwsolverDnHandle_t handle,
    lwsolverEigMode_t jobz, 
    int rank, 
    int m,   
    int n,  
    const lwDoubleComplex *d_A, 
    int lda,    
    long long int strideA,
    double *d_S,
    long long int strideS,
    lwDoubleComplex *d_U, 
    int ldu,   
    long long int strideU, 
    lwDoubleComplex *d_V,
    int ldv, 
    long long int strideV, 
    lwDoubleComplex *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF,
    int batchSize);

lwsolverStatus_t LWSOLVERAPI lwsolverDnCreateParams(
    lwsolverDnParams_t *params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnDestroyParams(
    lwsolverDnParams_t params);

lwsolverStatus_t LWSOLVERAPI lwsolverDnSetAdvOptions (
    lwsolverDnParams_t params,
    lwsolverDnFunction_t function,
    lwsolverAlgMode_t algo   );

/* 64-bit API for POTRF */
LWSOLVER_DEPRECATED(lwsolverDnXpotrf_bufferSize)
lwsolverStatus_t LWSOLVERAPI lwsolverDnPotrf_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwblasFillMode_t uplo,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType computeType,
    size_t *workspaceInBytes );

LWSOLVER_DEPRECATED(lwsolverDnXpotrf)
lwsolverStatus_t LWSOLVERAPI lwsolverDnPotrf(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwblasFillMode_t uplo,
    int64_t n,
    lwdaDataType dataTypeA,
    void *A,
    int64_t lda,
    lwdaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info );

/* 64-bit API for POTRS */
LWSOLVER_DEPRECATED(lwsolverDnXpotrs)
lwsolverStatus_t LWSOLVERAPI lwsolverDnPotrs(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwblasFillMode_t uplo,
    int64_t n,
    int64_t nrhs,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType dataTypeB,
    void *B,
    int64_t ldb,
    int *info);


/* 64-bit API for GEQRF */
LWSOLVER_DEPRECATED(lwsolverDnXgeqrf_bufferSize)
lwsolverStatus_t LWSOLVERAPI lwsolverDnGeqrf_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType dataTypeTau,
    const void *tau,
    lwdaDataType computeType,
    size_t *workspaceInBytes );

LWSOLVER_DEPRECATED(lwsolverDnXgeqrf)
lwsolverStatus_t LWSOLVERAPI lwsolverDnGeqrf(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    void *A,
    int64_t lda,
    lwdaDataType dataTypeTau,
    void *tau,
    lwdaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info );

/* 64-bit API for GETRF */
LWSOLVER_DEPRECATED(lwsolverDnXgetrf_bufferSize)
lwsolverStatus_t LWSOLVERAPI lwsolverDnGetrf_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType computeType,
    size_t *workspaceInBytes );

LWSOLVER_DEPRECATED(lwsolverDnXgetrf)
lwsolverStatus_t LWSOLVERAPI lwsolverDnGetrf(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    void *A,
    int64_t lda,
    int64_t *ipiv,
    lwdaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info );

/* 64-bit API for GETRS */
LWSOLVER_DEPRECATED(lwsolverDnXgetrs)
lwsolverStatus_t LWSOLVERAPI lwsolverDnGetrs(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwblasOperation_t trans,
    int64_t n,
    int64_t nrhs,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    const int64_t *ipiv,
    lwdaDataType dataTypeB,
    void *B,
    int64_t ldb,
    int *info );

/* 64-bit API for SYEVD */
LWSOLVER_DEPRECATED(lwsolverDnXsyevd_bufferSize)
lwsolverStatus_t LWSOLVERAPI lwsolverDnSyevd_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType dataTypeW,
    const void *W,
    lwdaDataType computeType,
    size_t *workspaceInBytes);

LWSOLVER_DEPRECATED(lwsolverDnXsyevd)
lwsolverStatus_t LWSOLVERAPI lwsolverDnSyevd(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,
    int64_t n,
    lwdaDataType dataTypeA,
    void *A,
    int64_t lda,
    lwdaDataType dataTypeW,
    void *W,
    lwdaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info);

/* 64-bit API for SYEVDX */
LWSOLVER_DEPRECATED(lwsolverDnXsyevdx_bufferSize)
lwsolverStatus_t LWSOLVERAPI lwsolverDnSyevdx_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwsolverEigMode_t jobz,
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    void *vl,
    void *vu,
    int64_t il,
    int64_t iu,
    int64_t *h_meig,
    lwdaDataType dataTypeW,
    const void *W,
    lwdaDataType computeType,
    size_t *workspaceInBytes);


LWSOLVER_DEPRECATED(lwsolverDnXsyevdx)
lwsolverStatus_t LWSOLVERAPI lwsolverDnSyevdx(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwsolverEigMode_t jobz,
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo,
    int64_t n,
    lwdaDataType dataTypeA,
    void *A,
    int64_t lda,
    void * vl,
    void * vu,
    int64_t il,
    int64_t iu,
    int64_t *meig64,
    lwdaDataType dataTypeW,
    void *W,
    lwdaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info);

/* 64-bit API for GESVD */
LWSOLVER_DEPRECATED(lwsolverDnXgesvd_bufferSize)
lwsolverStatus_t LWSOLVERAPI lwsolverDnGesvd_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    signed char jobu,
    signed char jobvt,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType dataTypeS,
    const void *S,
    lwdaDataType dataTypeU,
    const void *U,
    int64_t ldu,
    lwdaDataType dataTypeVT,
    const void *VT,
    int64_t ldvt,
    lwdaDataType computeType,
    size_t *workspaceInBytes);

LWSOLVER_DEPRECATED(lwsolverDnXgesvd)
lwsolverStatus_t LWSOLVERAPI lwsolverDnGesvd(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    signed char jobu,
    signed char jobvt,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    void *A,
    int64_t lda,
    lwdaDataType dataTypeS,
    void *S,
    lwdaDataType dataTypeU,
    void *U,
    int64_t ldu,
    lwdaDataType dataTypeVT,
    void *VT,
    int64_t ldvt,
    lwdaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info);

/*
 * new 64-bit API
 */
/* 64-bit API for POTRF */
lwsolverStatus_t LWSOLVERAPI lwsolverDnXpotrf_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwblasFillMode_t uplo,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXpotrf(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwblasFillMode_t uplo,
    int64_t n,
    lwdaDataType dataTypeA,
    void *A,
    int64_t lda,
    lwdaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info );

/* 64-bit API for POTRS */
lwsolverStatus_t LWSOLVERAPI lwsolverDnXpotrs(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwblasFillMode_t uplo,
    int64_t n,
    int64_t nrhs,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType dataTypeB,
    void *B,
    int64_t ldb,
    int *info);

/* 64-bit API for GEQRF */
lwsolverStatus_t LWSOLVERAPI lwsolverDnXgeqrf_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType dataTypeTau,
    const void *tau,
    lwdaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXgeqrf(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    void *A,
    int64_t lda,
    lwdaDataType dataTypeTau,
    void *tau,
    lwdaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info );

/* 64-bit API for GETRF */
lwsolverStatus_t LWSOLVERAPI lwsolverDnXgetrf_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXgetrf(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    void *A,
    int64_t lda,
    int64_t *ipiv,
    lwdaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info );

/* 64-bit API for GETRS */
lwsolverStatus_t LWSOLVERAPI lwsolverDnXgetrs(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwblasOperation_t trans,
    int64_t n,
    int64_t nrhs,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    const int64_t *ipiv,
    lwdaDataType dataTypeB,
    void *B,
    int64_t ldb,
    int *info );

/* 64-bit API for SYEVD */
lwsolverStatus_t LWSOLVERAPI lwsolverDnXsyevd_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType dataTypeW,
    const void *W,
    lwdaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXsyevd(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,
    int64_t n,
    lwdaDataType dataTypeA,
    void *A,
    int64_t lda,
    lwdaDataType dataTypeW,
    void *W,
    lwdaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info);

/* 64-bit API for SYEVDX */
lwsolverStatus_t LWSOLVERAPI lwsolverDnXsyevdx_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwsolverEigMode_t jobz,
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    void *vl,
    void *vu,
    int64_t il,
    int64_t iu,
    int64_t *h_meig,
    lwdaDataType dataTypeW,
    const void *W,
    lwdaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXsyevdx(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwsolverEigMode_t jobz,
    lwsolverEigRange_t range,
    lwblasFillMode_t uplo,
    int64_t n,
    lwdaDataType dataTypeA,
    void *A,
    int64_t lda,
    void * vl,
    void * vu,
    int64_t il,
    int64_t iu,
    int64_t *meig64,
    lwdaDataType dataTypeW,
    void *W,
    lwdaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info);

/* 64-bit API for GESVD */
lwsolverStatus_t LWSOLVERAPI lwsolverDnXgesvd_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    signed char jobu,
    signed char jobvt,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType dataTypeS,
    const void *S,
    lwdaDataType dataTypeU,
    const void *U,
    int64_t ldu,
    lwdaDataType dataTypeVT,
    const void *VT,
    int64_t ldvt,
    lwdaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXgesvd(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    signed char jobu,
    signed char jobvt,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    void *A,
    int64_t lda,
    lwdaDataType dataTypeS,
    void *S,
    lwdaDataType dataTypeU,
    void *U,
    int64_t ldu,
    lwdaDataType dataTypeVT,
    void *VT,
    int64_t ldvt,
    lwdaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info);

/* 64-bit API for GESVDP */
lwsolverStatus_t LWSOLVERAPI lwsolverDnXgesvdp_bufferSize(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwsolverEigMode_t jobz,
    int econ,
    int64_t m,
    int64_t n,
    lwdaDataType dataTypeA,
    const void *A,
    int64_t lda,
    lwdaDataType dataTypeS,
    const void *S,
    lwdaDataType dataTypeU,
    const void *U,
    int64_t ldu,
    lwdaDataType dataTypeV,
    const void *V,
    int64_t ldv,
    lwdaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXgesvdp(
    lwsolverDnHandle_t handle,
    lwsolverDnParams_t params,
    lwsolverEigMode_t jobz, 
    int econ,   
    int64_t m,   
    int64_t n,   
    lwdaDataType dataTypeA,
    void *A,            
    int64_t lda,     
    lwdaDataType dataTypeS,
    void *S,  
    lwdaDataType dataTypeU,
    void *U,    
    int64_t ldu,   
    lwdaDataType dataTypeV,
    void *V,  
    int64_t ldv, 
    lwdaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *d_info,
    double *h_err_sigma);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXgesvdr_bufferSize (
		lwsolverDnHandle_t handle,
		lwsolverDnParams_t params,
		signed char        jobu,
		signed char        jobv,
		int64_t            m,
		int64_t            n,
		int64_t            k,
		int64_t            p,
		int64_t            niters,
		lwdaDataType       dataTypeA,
		const void         *A,
		int64_t            lda,
		lwdaDataType       dataTypeSrand,
		const void         *Srand,
		lwdaDataType       dataTypeUrand,
		const void         *Urand,
		int64_t            ldUrand,
		lwdaDataType       dataTypeVrand,
		const void         *Vrand,
		int64_t            ldVrand,
		lwdaDataType       computeType,
		size_t             *workspaceInBytesOnDevice,
		size_t             *workspaceInBytesOnHost
		);

lwsolverStatus_t LWSOLVERAPI lwsolverDnXgesvdr(
		lwsolverDnHandle_t handle,
		lwsolverDnParams_t params,
		signed char        jobu,
		signed char        jobv,
		int64_t            m,
		int64_t            n,
		int64_t            k,
		int64_t            p,
		int64_t            niters,
		lwdaDataType       dataTypeA,
		void               *A,
		int64_t            lda,
		lwdaDataType       dataTypeSrand,
		void               *Srand,
		lwdaDataType       dataTypeUrand,
		void               *Urand,
		int64_t            ldUrand,
		lwdaDataType       dataTypeVrand,
		void               *Vrand,
		int64_t            ldVrand,
		lwdaDataType       computeType,
		void               *bufferOnDevice,
		size_t             workspaceInBytesOnDevice,
		void               *bufferOnHost,
		size_t             workspaceInBytesOnHost,
		int                *d_info
		);


#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* !defined(LWDENSE_H_) */
