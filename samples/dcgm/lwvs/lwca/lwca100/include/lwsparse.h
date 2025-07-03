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
 
#if !defined(LWSPARSE_H_)
#define LWSPARSE_H_


#ifndef LWSPARSEAPI
#ifdef _WIN32
#define LWSPARSEAPI __stdcall
#else
#define LWSPARSEAPI 
#endif
#endif

#include "driver_types.h"
#include "lwComplex.h"   /* import complex data type */

#include "lwda_fp16.h"

#include "library_types.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/* LWSPARSE status type returns */
typedef enum{
    LWSPARSE_STATUS_SUCCESS=0,
    LWSPARSE_STATUS_NOT_INITIALIZED=1,
    LWSPARSE_STATUS_ALLOC_FAILED=2,
    LWSPARSE_STATUS_ILWALID_VALUE=3,
    LWSPARSE_STATUS_ARCH_MISMATCH=4,
    LWSPARSE_STATUS_MAPPING_ERROR=5,
    LWSPARSE_STATUS_EXELWTION_FAILED=6,
    LWSPARSE_STATUS_INTERNAL_ERROR=7,
    LWSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8,
    LWSPARSE_STATUS_ZERO_PIVOT=9
} lwsparseStatus_t;

/* Opaque structure holding LWSPARSE library context */
struct lwsparseContext;
typedef struct lwsparseContext *lwsparseHandle_t;

/* Opaque structure holding the matrix descriptor */
struct lwsparseMatDescr;
typedef struct lwsparseMatDescr *lwsparseMatDescr_t;

/* Opaque structure holding the sparse triangular solve information */
struct lwsparseSolveAnalysisInfo;
typedef struct lwsparseSolveAnalysisInfo *lwsparseSolveAnalysisInfo_t;

/* Opaque structures holding the sparse triangular solve information */
struct csrsv2Info;
typedef struct csrsv2Info *csrsv2Info_t;

struct csrsm2Info;
typedef struct csrsm2Info *csrsm2Info_t;

struct bsrsv2Info;
typedef struct bsrsv2Info *bsrsv2Info_t;

struct bsrsm2Info;
typedef struct bsrsm2Info *bsrsm2Info_t;

/* Opaque structures holding incomplete Cholesky information */
struct csric02Info;
typedef struct csric02Info *csric02Info_t;

struct bsric02Info;
typedef struct bsric02Info *bsric02Info_t;

/* Opaque structures holding incomplete LU information */
struct csrilu02Info;
typedef struct csrilu02Info *csrilu02Info_t;

struct bsrilu02Info;
typedef struct bsrilu02Info *bsrilu02Info_t;

/* Opaque structures holding the hybrid (HYB) storage information */
struct lwsparseHybMat;
typedef struct lwsparseHybMat *lwsparseHybMat_t;

/* Opaque structures holding sparse gemm information */
struct csrgemm2Info;
typedef struct csrgemm2Info *csrgemm2Info_t;

/* Opaque structure holding the sorting information */
struct csru2csrInfo;
typedef struct csru2csrInfo *csru2csrInfo_t;

/* Opaque structure holding the coloring information */
struct lwsparseColorInfo;
typedef struct lwsparseColorInfo *lwsparseColorInfo_t;

/* Opaque structure holding the prune information */
struct pruneInfo;
typedef struct pruneInfo *pruneInfo_t;

/* Types definitions */
typedef enum { 
    LWSPARSE_POINTER_MODE_HOST = 0,  
    LWSPARSE_POINTER_MODE_DEVICE = 1        
} lwsparsePointerMode_t;

typedef enum { 
    LWSPARSE_ACTION_SYMBOLIC = 0,  
    LWSPARSE_ACTION_NUMERIC = 1        
} lwsparseAction_t;

typedef enum {
    LWSPARSE_MATRIX_TYPE_GENERAL = 0, 
    LWSPARSE_MATRIX_TYPE_SYMMETRIC = 1,     
    LWSPARSE_MATRIX_TYPE_HERMITIAN = 2, 
    LWSPARSE_MATRIX_TYPE_TRIANGULAR = 3 
} lwsparseMatrixType_t;

typedef enum {
    LWSPARSE_FILL_MODE_LOWER = 0, 
    LWSPARSE_FILL_MODE_UPPER = 1
} lwsparseFillMode_t;

typedef enum {
    LWSPARSE_DIAG_TYPE_NON_UNIT = 0, 
    LWSPARSE_DIAG_TYPE_UNIT = 1
} lwsparseDiagType_t; 

typedef enum {
    LWSPARSE_INDEX_BASE_ZERO = 0, 
    LWSPARSE_INDEX_BASE_ONE = 1
} lwsparseIndexBase_t;

typedef enum {
    LWSPARSE_OPERATION_NON_TRANSPOSE = 0,  
    LWSPARSE_OPERATION_TRANSPOSE = 1,  
    LWSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2  
} lwsparseOperation_t;

typedef enum {
    LWSPARSE_DIRECTION_ROW = 0,  
    LWSPARSE_DIRECTION_COLUMN = 1  
} lwsparseDirection_t;

typedef enum {
    LWSPARSE_HYB_PARTITION_AUTO = 0,  // automatically decide how to split the data into regular/irregular part
    LWSPARSE_HYB_PARTITION_USER = 1,  // store data into regular part up to a user specified treshhold
    LWSPARSE_HYB_PARTITION_MAX = 2    // store all data in the regular part
} lwsparseHybPartition_t;

// used in csrsv2, csric02, and csrilu02
typedef enum {
    LWSPARSE_SOLVE_POLICY_NO_LEVEL = 0, // no level information is generated, only reports structural zero.
    LWSPARSE_SOLVE_POLICY_USE_LEVEL = 1  
} lwsparseSolvePolicy_t;

typedef enum {
    LWSPARSE_SIDE_LEFT =0,
    LWSPARSE_SIDE_RIGHT=1
} lwsparseSideMode_t;

typedef enum {
    LWSPARSE_COLOR_ALG0 = 0, // default
    LWSPARSE_COLOR_ALG1 = 1
} lwsparseColorAlg_t;

typedef enum {
    LWSPARSE_ALG0 = 0, //default, naive
    LWSPARSE_ALG1 = 1, //merge path
    LWSPARSE_ALG_NAIVE = 0,
    LWSPARSE_ALG_MERGE_PATH = 1 //merge path alias
} lwsparseAlgMode_t;

/* LWSPARSE initialization and managment routines */
lwsparseStatus_t LWSPARSEAPI lwsparseCreate(lwsparseHandle_t *handle);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroy(lwsparseHandle_t handle);
lwsparseStatus_t LWSPARSEAPI lwsparseGetVersion(lwsparseHandle_t handle, int *version);
lwsparseStatus_t LWSPARSEAPI lwsparseGetProperty(libraryPropertyType type, int *value);
lwsparseStatus_t LWSPARSEAPI lwsparseSetStream(lwsparseHandle_t handle, lwdaStream_t streamId); 
lwsparseStatus_t LWSPARSEAPI lwsparseGetStream(lwsparseHandle_t handle, lwdaStream_t *streamId);


/* LWSPARSE type creation, destruction, set and get routines */
lwsparseStatus_t LWSPARSEAPI lwsparseGetPointerMode(lwsparseHandle_t handle, lwsparsePointerMode_t *mode);
lwsparseStatus_t LWSPARSEAPI lwsparseSetPointerMode(lwsparseHandle_t handle, lwsparsePointerMode_t mode);

/* sparse matrix descriptor */
/* When the matrix descriptor is created, its fields are initialized to: 
   LWSPARSE_MATRIX_TYPE_GENERAL
   LWSPARSE_INDEX_BASE_ZERO
   All other fields are uninitialized
*/                                   
lwsparseStatus_t LWSPARSEAPI lwsparseCreateMatDescr(lwsparseMatDescr_t *descrA);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroyMatDescr (lwsparseMatDescr_t descrA);

lwsparseStatus_t LWSPARSEAPI lwsparseCopyMatDescr(lwsparseMatDescr_t dest, const lwsparseMatDescr_t src);

lwsparseStatus_t LWSPARSEAPI lwsparseSetMatType(lwsparseMatDescr_t descrA, lwsparseMatrixType_t type);
lwsparseMatrixType_t LWSPARSEAPI lwsparseGetMatType(const lwsparseMatDescr_t descrA);

lwsparseStatus_t LWSPARSEAPI lwsparseSetMatFillMode(lwsparseMatDescr_t descrA, lwsparseFillMode_t fillMode);
lwsparseFillMode_t LWSPARSEAPI lwsparseGetMatFillMode(const lwsparseMatDescr_t descrA);
 
lwsparseStatus_t LWSPARSEAPI lwsparseSetMatDiagType(lwsparseMatDescr_t  descrA, lwsparseDiagType_t diagType);
lwsparseDiagType_t LWSPARSEAPI lwsparseGetMatDiagType(const lwsparseMatDescr_t descrA);

lwsparseStatus_t LWSPARSEAPI lwsparseSetMatIndexBase(lwsparseMatDescr_t descrA, lwsparseIndexBase_t base);
lwsparseIndexBase_t LWSPARSEAPI lwsparseGetMatIndexBase(const lwsparseMatDescr_t descrA);

/* sparse triangular solve and incomplete-LU and Cholesky (algorithm 1) */
lwsparseStatus_t LWSPARSEAPI lwsparseCreateSolveAnalysisInfo(lwsparseSolveAnalysisInfo_t *info);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroySolveAnalysisInfo(lwsparseSolveAnalysisInfo_t info);
lwsparseStatus_t LWSPARSEAPI lwsparseGetLevelInfo(lwsparseHandle_t handle, 
                                                  lwsparseSolveAnalysisInfo_t info, 
                                                  int *nlevels, 
                                                  int **levelPtr, 
                                                  int **levelInd);

/* sparse triangular solve (algorithm 2) */
lwsparseStatus_t LWSPARSEAPI lwsparseCreateCsrsv2Info(csrsv2Info_t *info);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroyCsrsv2Info(csrsv2Info_t info);

/* incomplete Cholesky (algorithm 2)*/
lwsparseStatus_t LWSPARSEAPI lwsparseCreateCsric02Info(csric02Info_t *info);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroyCsric02Info(csric02Info_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseCreateBsric02Info(bsric02Info_t *info);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroyBsric02Info(bsric02Info_t info);

/* incomplete LU (algorithm 2) */
lwsparseStatus_t LWSPARSEAPI lwsparseCreateCsrilu02Info(csrilu02Info_t *info);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroyCsrilu02Info(csrilu02Info_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseCreateBsrilu02Info(bsrilu02Info_t *info);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroyBsrilu02Info(bsrilu02Info_t info);

/* block-CSR triangular solve (algorithm 2) */
lwsparseStatus_t LWSPARSEAPI lwsparseCreateBsrsv2Info(bsrsv2Info_t *info);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroyBsrsv2Info(bsrsv2Info_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseCreateBsrsm2Info(bsrsm2Info_t *info);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroyBsrsm2Info(bsrsm2Info_t info);

/* hybrid (HYB) format */
lwsparseStatus_t LWSPARSEAPI lwsparseCreateHybMat(lwsparseHybMat_t *hybA);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroyHybMat(lwsparseHybMat_t hybA);

/* sorting information */
lwsparseStatus_t LWSPARSEAPI lwsparseCreateCsru2csrInfo(csru2csrInfo_t *info);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroyCsru2csrInfo(csru2csrInfo_t info);

/* coloring info */
lwsparseStatus_t LWSPARSEAPI lwsparseCreateColorInfo(lwsparseColorInfo_t *info);
lwsparseStatus_t LWSPARSEAPI lwsparseDestroyColorInfo(lwsparseColorInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseSetColorAlgs(lwsparseColorInfo_t info, lwsparseColorAlg_t alg);
lwsparseStatus_t LWSPARSEAPI lwsparseGetColorAlgs(lwsparseColorInfo_t info, lwsparseColorAlg_t *alg);

/* prune information */
lwsparseStatus_t LWSPARSEAPI lwsparseCreatePruneInfo(pruneInfo_t *info);

lwsparseStatus_t LWSPARSEAPI lwsparseDestroyPruneInfo(pruneInfo_t info);


/* --- Sparse Level 1 routines --- */

/* Description: Addition of a scalar multiple of a sparse vector x  
   and a dense vector y. */ 
lwsparseStatus_t LWSPARSEAPI lwsparseSaxpyi(lwsparseHandle_t handle, 
                                            int nnz, 
                                            const float *alpha, 
                                            const float *xVal, 
                                            const int *xInd, 
                                            float *y, 
                                            lwsparseIndexBase_t idxBase);
    
lwsparseStatus_t LWSPARSEAPI lwsparseDaxpyi(lwsparseHandle_t handle, 
                                            int nnz, 
                                            const double *alpha, 
                                            const double *xVal, 
                                            const int *xInd, 
                                            double *y, 
                                            lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseCaxpyi(lwsparseHandle_t handle, 
                                            int nnz, 
                                            const lwComplex *alpha, 
                                            const lwComplex *xVal, 
                                            const int *xInd, 
                                            lwComplex *y, 
                                            lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseZaxpyi(lwsparseHandle_t handle, 
                                            int nnz, 
                                            const lwDoubleComplex *alpha, 
                                            const lwDoubleComplex *xVal, 
                                            const int *xInd, 
                                            lwDoubleComplex *y, 
                                            lwsparseIndexBase_t idxBase);

/* Description: dot product of a sparse vector x and a dense vector y. */
lwsparseStatus_t LWSPARSEAPI lwsparseSdoti(lwsparseHandle_t handle,  
                                           int nnz, 
                                           const float *xVal, 
                                           const int *xInd, 
                                           const float *y,
                                           float *resultDevHostPtr,
                                           lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseDdoti(lwsparseHandle_t handle, 
                                           int nnz, 
                                           const double *xVal, 
                                           const int *xInd, 
                                           const double *y, 
                                           double *resultDevHostPtr,
                                           lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseCdoti(lwsparseHandle_t handle, 
                                           int nnz, 
                                           const lwComplex *xVal, 
                                           const int *xInd, 
                                           const lwComplex *y, 
                                           lwComplex *resultDevHostPtr,
                                           lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseZdoti(lwsparseHandle_t handle, 
                                           int nnz, 
                                           const lwDoubleComplex *xVal, 
                                           const int *xInd, 
                                           const lwDoubleComplex *y, 
                                           lwDoubleComplex *resultDevHostPtr,
                                           lwsparseIndexBase_t idxBase);

/* Description: dot product of complex conjugate of a sparse vector x
   and a dense vector y. */
lwsparseStatus_t LWSPARSEAPI lwsparseCdotci(lwsparseHandle_t handle, 
                                            int nnz, 
                                            const lwComplex *xVal, 
                                            const int *xInd, 
                                            const lwComplex *y, 
                                            lwComplex *resultDevHostPtr,
                                            lwsparseIndexBase_t idxBase);
    
lwsparseStatus_t LWSPARSEAPI lwsparseZdotci(lwsparseHandle_t handle, 
                                            int nnz, 
                                            const lwDoubleComplex *xVal, 
                                            const int *xInd, 
                                            const lwDoubleComplex *y, 
                                            lwDoubleComplex *resultDevHostPtr,
                                            lwsparseIndexBase_t idxBase);


/* Description: Gather of non-zero elements from dense vector y into 
   sparse vector x. */
lwsparseStatus_t LWSPARSEAPI lwsparseSgthr(lwsparseHandle_t handle, 
                                           int nnz, 
                                           const float *y, 
                                           float *xVal, 
                                           const int *xInd, 
                                           lwsparseIndexBase_t idxBase);
    
lwsparseStatus_t LWSPARSEAPI lwsparseDgthr(lwsparseHandle_t handle, 
                                           int nnz, 
                                           const double *y, 
                                           double *xVal, 
                                           const int *xInd, 
                                           lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseCgthr(lwsparseHandle_t handle, 
                                           int nnz, 
                                           const lwComplex *y, 
                                           lwComplex *xVal, 
                                           const int *xInd, 
                                           lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseZgthr(lwsparseHandle_t handle, 
                                           int nnz, 
                                           const lwDoubleComplex *y, 
                                           lwDoubleComplex *xVal, 
                                           const int *xInd, 
                                           lwsparseIndexBase_t idxBase);

/* Description: Gather of non-zero elements from desne vector y into 
   sparse vector x (also replacing these elements in y by zeros). */
lwsparseStatus_t LWSPARSEAPI lwsparseSgthrz(lwsparseHandle_t handle, 
                                            int nnz, 
                                            float *y, 
                                            float *xVal, 
                                            const int *xInd, 
                                            lwsparseIndexBase_t idxBase);
    
lwsparseStatus_t LWSPARSEAPI lwsparseDgthrz(lwsparseHandle_t handle, 
                                            int nnz, 
                                            double *y, 
                                            double *xVal, 
                                            const int *xInd, 
                                            lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseCgthrz(lwsparseHandle_t handle, 
                                            int nnz, 
                                            lwComplex *y, 
                                            lwComplex *xVal, 
                                            const int *xInd, 
                                            lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseZgthrz(lwsparseHandle_t handle, 
                                            int nnz, 
                                            lwDoubleComplex *y, 
                                            lwDoubleComplex *xVal, 
                                            const int *xInd, 
                                            lwsparseIndexBase_t idxBase);

/* Description: Scatter of elements of the sparse vector x into 
   dense vector y. */
lwsparseStatus_t LWSPARSEAPI lwsparseSsctr(lwsparseHandle_t handle, 
                                           int nnz, 
                                           const float *xVal, 
                                           const int *xInd, 
                                           float *y, 
                                           lwsparseIndexBase_t idxBase);
    
lwsparseStatus_t LWSPARSEAPI lwsparseDsctr(lwsparseHandle_t handle, 
                                           int nnz, 
                                           const double *xVal, 
                                           const int *xInd, 
                                           double *y, 
                                           lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseCsctr(lwsparseHandle_t handle, 
                                           int nnz, 
                                           const lwComplex *xVal, 
                                           const int *xInd, 
                                           lwComplex *y, 
                                           lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseZsctr(lwsparseHandle_t handle, 
                                           int nnz, 
                                           const lwDoubleComplex *xVal, 
                                           const int *xInd, 
                                           lwDoubleComplex *y, 
                                           lwsparseIndexBase_t idxBase);

/* Description: Givens rotation, where c and s are cosine and sine, 
   x and y are sparse and dense vectors, respectively. */
lwsparseStatus_t LWSPARSEAPI lwsparseSroti(lwsparseHandle_t handle, 
                                              int nnz, 
                                              float *xVal, 
                                              const int *xInd, 
                                              float *y, 
                                              const float *c, 
                                              const float *s, 
                                              lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseDroti(lwsparseHandle_t handle, 
                                              int nnz, 
                                              double *xVal, 
                                              const int *xInd, 
                                              double *y, 
                                              const double *c, 
                                              const double *s, 
                                              lwsparseIndexBase_t idxBase);


/* --- Sparse Level 2 routines --- */

lwsparseStatus_t  LWSPARSEAPI lwsparseSgemvi(lwsparseHandle_t handle,
                                    lwsparseOperation_t transA,
                                    int m,
                                    int n,
                                    const float *alpha, /* host or device pointer */
                                    const float *A,
                                    int lda,
                                    int nnz,
                                    const float *xVal,
                                    const int *xInd,
                                    const float *beta, /* host or device pointer */
                                    float *y,
                                    lwsparseIndexBase_t   idxBase,
                                    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseSgemvi_bufferSize( lwsparseHandle_t handle,
    lwsparseOperation_t transA,
    int m,
    int n,
    int nnz,
    int *pBufferSize);

lwsparseStatus_t  LWSPARSEAPI lwsparseDgemvi(lwsparseHandle_t handle,
                                    lwsparseOperation_t transA,
                                    int m,
                                    int n,
                                    const double *alpha, /* host or device pointer */
                                    const double *A,
                                    int lda,
                                    int nnz,
                                    const double *xVal,
                                    const int *xInd,
                                    const double *beta, /* host or device pointer */
                                    double *y,
                                    lwsparseIndexBase_t   idxBase,
                                    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDgemvi_bufferSize( lwsparseHandle_t handle,
    lwsparseOperation_t transA,
    int m,
    int n,
    int nnz,
    int *pBufferSize);

lwsparseStatus_t  LWSPARSEAPI lwsparseCgemvi(lwsparseHandle_t handle,
                                    lwsparseOperation_t transA,
                                    int m,
                                    int n,
                                    const lwComplex *alpha, /* host or device pointer */
                                    const lwComplex *A,
                                    int lda,
                                    int nnz,
                                    const lwComplex *xVal,
                                    const int *xInd,
                                    const lwComplex *beta, /* host or device pointer */
                                    lwComplex *y,
                                    lwsparseIndexBase_t   idxBase,
                                    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCgemvi_bufferSize( lwsparseHandle_t handle,
    lwsparseOperation_t transA,
    int m,
    int n,
    int nnz,
    int *pBufferSize);

lwsparseStatus_t  LWSPARSEAPI lwsparseZgemvi(lwsparseHandle_t handle,
                                    lwsparseOperation_t transA,
                                    int m,
                                    int n,
                                    const lwDoubleComplex *alpha, /* host or device pointer */
                                    const lwDoubleComplex *A,
                                    int lda,
                                    int nnz,
                                    const lwDoubleComplex *xVal,
                                    const int *xInd,
                                    const lwDoubleComplex *beta, /* host or device pointer */
                                    lwDoubleComplex *y,
                                    lwsparseIndexBase_t   idxBase,
                                    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZgemvi_bufferSize( lwsparseHandle_t handle,
    lwsparseOperation_t transA,
    int m,
    int n,
    int nnz,
    int *pBufferSize);


/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in CSR storage format, x and y are dense vectors. */ 
lwsparseStatus_t LWSPARSEAPI lwsparseScsrmv(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            const float *alpha,
                                            const lwsparseMatDescr_t descrA, 
                                            const float *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const float *x, 
                                            const float *beta, 
                                            float *y);
    
lwsparseStatus_t LWSPARSEAPI lwsparseDcsrmv(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            const double *alpha,
                                            const lwsparseMatDescr_t descrA, 
                                            const double *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const double *x, 
                                            const double *beta,  
                                            double *y);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrmv(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA, 
                                            int m, 
                                            int n,
                                            int nnz,
                                            const lwComplex *alpha,
                                            const lwsparseMatDescr_t descrA, 
                                            const lwComplex *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const lwComplex *x, 
                                            const lwComplex *beta, 
                                            lwComplex *y);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrmv(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            const lwDoubleComplex *alpha,
                                            const lwsparseMatDescr_t descrA, 
                                            const lwDoubleComplex *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const lwDoubleComplex *x, 
                                            const lwDoubleComplex *beta, 
                                            lwDoubleComplex *y);  

//Returns number of bytes
lwsparseStatus_t LWSPARSEAPI lwsparseCsrmvEx_bufferSize(lwsparseHandle_t handle, 
                                                        lwsparseAlgMode_t alg,
                                                        lwsparseOperation_t transA, 
                                                        int m, 
                                                        int n, 
                                                        int nnz,
                                                        const void *alpha,
                                                        lwdaDataType alphatype,
                                                        const lwsparseMatDescr_t descrA,
                                                        const void *csrValA,
                                                        lwdaDataType csrValAtype,
                                                        const int *csrRowPtrA,
                                                        const int *csrColIndA,
                                                        const void *x,
                                                        lwdaDataType xtype,
                                                        const void *beta,
                                                        lwdaDataType betatype,
                                                        void *y,
                                                        lwdaDataType ytype,
                                                        lwdaDataType exelwtiontype,
                                                        size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCsrmvEx(lwsparseHandle_t handle, 
                                             lwsparseAlgMode_t alg,
                                             lwsparseOperation_t transA, 
                                             int m, 
                                             int n, 
                                             int nnz,
                                             const void *alpha,
                                             lwdaDataType alphatype,
                                             const lwsparseMatDescr_t descrA,
                                             const void *csrValA,
                                             lwdaDataType csrValAtype,
                                             const int *csrRowPtrA,
                                             const int *csrColIndA,
                                             const void *x,
                                             lwdaDataType xtype,
                                             const void *beta,
                                             lwdaDataType betatype,
                                             void *y,
                                             lwdaDataType ytype,
                                             lwdaDataType exelwtiontype,
                                             void* buffer);

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in CSR storage format, x and y are dense vectors
   using a Merge Path load-balancing implementation. */ 
   lwsparseStatus_t LWSPARSEAPI lwsparseScsrmv_mp(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            const float *alpha,
                                            const lwsparseMatDescr_t descrA, 
                                            const float *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const float *x, 
                                            const float *beta, 
                                            float *y);
    
lwsparseStatus_t LWSPARSEAPI lwsparseDcsrmv_mp(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            const double *alpha,
                                            const lwsparseMatDescr_t descrA, 
                                            const double *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const double *x, 
                                            const double *beta,  
                                            double *y);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrmv_mp(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA, 
                                            int m, 
                                            int n,
                                            int nnz,
                                            const lwComplex *alpha,
                                            const lwsparseMatDescr_t descrA, 
                                            const lwComplex *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const lwComplex *x, 
                                            const lwComplex *beta, 
                                            lwComplex *y);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrmv_mp(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            const lwDoubleComplex *alpha,
                                            const lwsparseMatDescr_t descrA, 
                                            const lwDoubleComplex *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const lwDoubleComplex *x, 
                                            const lwDoubleComplex *beta, 
                                            lwDoubleComplex *y);  

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in HYB storage format, x and y are dense vectors. */    
lwsparseStatus_t LWSPARSEAPI lwsparseShybmv(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA,
                                            const float *alpha,
                                            const lwsparseMatDescr_t descrA,
                                            const lwsparseHybMat_t hybA,
                                            const float *x,
                                            const float *beta,
                                            float *y);

lwsparseStatus_t LWSPARSEAPI lwsparseDhybmv(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA,
                                            const double *alpha,
                                            const lwsparseMatDescr_t descrA,
                                            const lwsparseHybMat_t hybA,
                                            const double *x,
                                            const double *beta,
                                            double *y);

lwsparseStatus_t LWSPARSEAPI lwsparseChybmv(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA,
                                            const lwComplex *alpha,
                                            const lwsparseMatDescr_t descrA,
                                            const lwsparseHybMat_t hybA,
                                            const lwComplex *x,
                                            const lwComplex *beta,
                                            lwComplex *y);

lwsparseStatus_t LWSPARSEAPI lwsparseZhybmv(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA,
                                            const lwDoubleComplex *alpha,
                                            const lwsparseMatDescr_t descrA,
                                            const lwsparseHybMat_t hybA,
                                            const lwDoubleComplex *x,
                                            const lwDoubleComplex *beta,
                                            lwDoubleComplex *y);
    
/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in BSR storage format, x and y are dense vectors. */
lwsparseStatus_t LWSPARSEAPI lwsparseSbsrmv(lwsparseHandle_t handle,
                                            lwsparseDirection_t dirA,
                                            lwsparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const float *alpha,
                                            const lwsparseMatDescr_t descrA,
                                            const float *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            int  blockDim,
                                            const float *x,
                                            const float *beta,
                                            float *y);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrmv(lwsparseHandle_t handle,
                                            lwsparseDirection_t dirA,
                                            lwsparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const double *alpha,
                                            const lwsparseMatDescr_t descrA,
                                            const double *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            int  blockDim,
                                            const double *x,
                                            const double *beta,
                                            double *y);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrmv(lwsparseHandle_t handle,
                                            lwsparseDirection_t dirA,
                                            lwsparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const lwComplex *alpha,
                                            const lwsparseMatDescr_t descrA,
                                            const lwComplex *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            int  blockDim,
                                            const lwComplex *x,
                                            const lwComplex *beta,
                                            lwComplex *y);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrmv(lwsparseHandle_t handle,
                                            lwsparseDirection_t dirA,
                                            lwsparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const lwDoubleComplex *alpha,
                                            const lwsparseMatDescr_t descrA,
                                            const lwDoubleComplex *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            int  blockDim,
                                            const lwDoubleComplex *x,
                                            const lwDoubleComplex *beta,
                                            lwDoubleComplex *y);

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in extended BSR storage format, x and y are dense 
   vectors. */
lwsparseStatus_t LWSPARSEAPI lwsparseSbsrxmv(lwsparseHandle_t handle,
                                             lwsparseDirection_t dirA,
                                             lwsparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const float *alpha,
                                             const lwsparseMatDescr_t descrA,
                                             const float *bsrSortedValA,
                                             const int *bsrSortedMaskPtrA,
                                             const int *bsrSortedRowPtrA,
                                             const int *bsrSortedEndPtrA,
                                             const int *bsrSortedColIndA,
                                             int  blockDim,
                                             const float *x,
                                             const float *beta,
                                             float *y);


lwsparseStatus_t LWSPARSEAPI lwsparseDbsrxmv(lwsparseHandle_t handle,
                                             lwsparseDirection_t dirA,
                                             lwsparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const double *alpha,
                                             const lwsparseMatDescr_t descrA,
                                             const double *bsrSortedValA,
                                             const int *bsrSortedMaskPtrA,
                                             const int *bsrSortedRowPtrA,
                                             const int *bsrSortedEndPtrA,
                                             const int *bsrSortedColIndA,
                                             int  blockDim,
                                             const double *x,
                                             const double *beta,
                                             double *y);
    
lwsparseStatus_t LWSPARSEAPI lwsparseCbsrxmv(lwsparseHandle_t handle,
                                             lwsparseDirection_t dirA,
                                             lwsparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const lwComplex *alpha,
                                             const lwsparseMatDescr_t descrA,
                                             const lwComplex *bsrSortedValA,
                                             const int *bsrSortedMaskPtrA,
                                             const int *bsrSortedRowPtrA,
                                             const int *bsrSortedEndPtrA,
                                             const int *bsrSortedColIndA,
                                             int  blockDim,
                                             const lwComplex *x,
                                             const lwComplex *beta,
                                             lwComplex *y);


lwsparseStatus_t LWSPARSEAPI lwsparseZbsrxmv(lwsparseHandle_t handle,
                                             lwsparseDirection_t dirA,
                                             lwsparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const lwDoubleComplex *alpha,
                                             const lwsparseMatDescr_t descrA,
                                             const lwDoubleComplex *bsrSortedValA,
                                             const int *bsrSortedMaskPtrA,
                                             const int *bsrSortedRowPtrA,
                                             const int *bsrSortedEndPtrA,
                                             const int *bsrSortedColIndA,
                                             int  blockDim,
                                             const lwDoubleComplex *x,
                                             const lwDoubleComplex *beta,
                                             lwDoubleComplex *y);

/* Description: Solution of triangular linear system op(A) * x = alpha * f, 
   where A is a sparse matrix in CSR storage format, rhs f and solution x 
   are dense vectors. This routine implements algorithm 1 for the solve. */     
lwsparseStatus_t LWSPARSEAPI lwsparseCsrsv_analysisEx(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const lwsparseMatDescr_t descrA, 
                                                     const void *csrSortedValA,
                                                     lwdaDataType csrSortedValAtype,
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     lwsparseSolveAnalysisInfo_t info,
                                                     lwdaDataType exelwtiontype);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrsv_analysis(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const lwsparseMatDescr_t descrA, 
                                                     const float *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrsv_analysis(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const lwsparseMatDescr_t descrA, 
                                                     const double *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrsv_analysis(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const lwsparseMatDescr_t descrA, 
                                                     const lwComplex *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrsv_analysis(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const lwsparseMatDescr_t descrA, 
                                                     const lwDoubleComplex *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     lwsparseSolveAnalysisInfo_t info); 

lwsparseStatus_t LWSPARSEAPI lwsparseCsrsv_solveEx(lwsparseHandle_t handle, 
                                                   lwsparseOperation_t transA, 
                                                   int m,
                                                   const void *alpha, 
                                                   lwdaDataType alphatype,
                                                   const lwsparseMatDescr_t descrA, 
                                                   const void *csrSortedValA, 
                                                   lwdaDataType csrSortedValAtype,
                                                   const int *csrSortedRowPtrA, 
                                                   const int *csrSortedColIndA, 
                                                   lwsparseSolveAnalysisInfo_t info, 
                                                   const void *f, 
                                                   lwdaDataType ftype,
                                                   void *x,
                                                   lwdaDataType xtype,
                                                   lwdaDataType exelwtiontype);
  
lwsparseStatus_t LWSPARSEAPI lwsparseScsrsv_solve(lwsparseHandle_t handle, 
                                                  lwsparseOperation_t transA, 
                                                  int m,
                                                  const float *alpha, 
                                                  const lwsparseMatDescr_t descrA, 
                                                  const float *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  lwsparseSolveAnalysisInfo_t info, 
                                                  const float *f, 
                                                  float *x);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrsv_solve(lwsparseHandle_t handle, 
                                                  lwsparseOperation_t transA, 
                                                  int m, 
                                                  const double *alpha, 
                                                  const lwsparseMatDescr_t descrA, 
                                                  const double *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  lwsparseSolveAnalysisInfo_t info, 
                                                  const double *f, 
                                                  double *x);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrsv_solve(lwsparseHandle_t handle, 
                                                  lwsparseOperation_t transA, 
                                                  int m, 
                                                  const lwComplex *alpha, 
                                                  const lwsparseMatDescr_t descrA, 
                                                  const lwComplex *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  lwsparseSolveAnalysisInfo_t info, 
                                                  const lwComplex *f, 
                                                  lwComplex *x);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrsv_solve(lwsparseHandle_t handle, 
                                                  lwsparseOperation_t transA, 
                                                  int m, 
                                                  const lwDoubleComplex *alpha, 
                                                  const lwsparseMatDescr_t descrA, 
                                                  const lwDoubleComplex *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  lwsparseSolveAnalysisInfo_t info, 
                                                  const lwDoubleComplex *f, 
                                                  lwDoubleComplex *x);      

/* Description: Solution of triangular linear system op(A) * x = alpha * f, 
   where A is a sparse matrix in CSR storage format, rhs f and solution y 
   are dense vectors. This routine implements algorithm 1 for this problem. 
   Also, it provides a utility function to query size of buffer used. */
lwsparseStatus_t LWSPARSEAPI lwsparseXcsrsv2_zeroPivot(lwsparseHandle_t handle,
                                                       csrsv2Info_t info,
                                                       int *position);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrsv2_bufferSize(lwsparseHandle_t handle,
                                                        lwsparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const lwsparseMatDescr_t descrA,
                                                        float *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrsv2_bufferSize(lwsparseHandle_t handle,
                                                        lwsparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const lwsparseMatDescr_t descrA,
                                                        double *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrsv2_bufferSize(lwsparseHandle_t handle,
                                                        lwsparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const lwsparseMatDescr_t descrA,
                                                        lwComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrsv2_bufferSize(lwsparseHandle_t handle,
                                                        lwsparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const lwsparseMatDescr_t descrA,
                                                        lwDoubleComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrsv2_bufferSizeExt(lwsparseHandle_t handle,
                                                        lwsparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const lwsparseMatDescr_t descrA,
                                                        float *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrsv2_bufferSizeExt(lwsparseHandle_t handle,
                                                        lwsparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const lwsparseMatDescr_t descrA,
                                                        double *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrsv2_bufferSizeExt(lwsparseHandle_t handle,
                                                        lwsparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const lwsparseMatDescr_t descrA,
                                                        lwComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrsv2_bufferSizeExt(lwsparseHandle_t handle,
                                                        lwsparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const lwsparseMatDescr_t descrA,
                                                        lwDoubleComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrsv2Info_t info,
                                                        size_t *pBufferSize);


lwsparseStatus_t LWSPARSEAPI lwsparseScsrsv2_analysis(lwsparseHandle_t handle,
                                                      lwsparseOperation_t transA,
                                                      int m,
                                                      int nnz,
                                                      const lwsparseMatDescr_t descrA,
                                                      const float *csrSortedValA,
                                                      const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA,
                                                      csrsv2Info_t info,
                                                      lwsparseSolvePolicy_t policy,
                                                      void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrsv2_analysis(lwsparseHandle_t handle,
                                                      lwsparseOperation_t transA,
                                                      int m,
                                                      int nnz,
                                                      const lwsparseMatDescr_t descrA,
                                                      const double *csrSortedValA,
                                                      const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA,
                                                      csrsv2Info_t info,
                                                      lwsparseSolvePolicy_t policy,
                                                      void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrsv2_analysis(lwsparseHandle_t handle,
                                                      lwsparseOperation_t transA,
                                                      int m,
                                                      int nnz,
                                                      const lwsparseMatDescr_t descrA,
                                                      const lwComplex *csrSortedValA,
                                                      const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA,
                                                      csrsv2Info_t info,
                                                      lwsparseSolvePolicy_t policy,
                                                      void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrsv2_analysis(lwsparseHandle_t handle,
                                                      lwsparseOperation_t transA,
                                                      int m,
                                                      int nnz,
                                                      const lwsparseMatDescr_t descrA,
                                                      const lwDoubleComplex *csrSortedValA,
                                                      const int *csrSortedRowPtrA,
                                                      const int *csrSortedColIndA,
                                                      csrsv2Info_t info,
                                                      lwsparseSolvePolicy_t policy,
                                                      void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrsv2_solve(lwsparseHandle_t handle,
                                                   lwsparseOperation_t transA,
                                                   int m,
                                                   int nnz,
                                                   const float *alpha,
                                                   const lwsparseMatDescr_t descrA,
                                                   const float *csrSortedValA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   csrsv2Info_t info,
                                                   const float *f,
                                                   float *x,
                                                   lwsparseSolvePolicy_t policy,
                                                   void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrsv2_solve(lwsparseHandle_t handle,
                                                   lwsparseOperation_t transA,
                                                   int m,
                                                   int nnz,
                                                   const double *alpha,
                                                   const lwsparseMatDescr_t descrA,
                                                   const double *csrSortedValA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   csrsv2Info_t info,
                                                   const double *f,
                                                   double *x,
                                                   lwsparseSolvePolicy_t policy,
                                                   void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrsv2_solve(lwsparseHandle_t handle,
                                                   lwsparseOperation_t transA,
                                                   int m,
                                                   int nnz,
                                                   const lwComplex *alpha,
                                                   const lwsparseMatDescr_t descrA,
                                                   const lwComplex *csrSortedValA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   csrsv2Info_t info,
                                                   const lwComplex *f,
                                                   lwComplex *x,
                                                   lwsparseSolvePolicy_t policy,
                                                   void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrsv2_solve(lwsparseHandle_t handle,
                                                   lwsparseOperation_t transA,
                                                   int m,
                                                   int nnz,
                                                   const lwDoubleComplex *alpha,
                                                   const lwsparseMatDescr_t descrA,
                                                   const lwDoubleComplex *csrSortedValA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   csrsv2Info_t info,
                                                   const lwDoubleComplex *f,
                                                   lwDoubleComplex *x,
                                                   lwsparseSolvePolicy_t policy,
                                                   void *pBuffer);

/* Description: Solution of triangular linear system op(A) * x = alpha * f, 
   where A is a sparse matrix in block-CSR storage format, rhs f and solution y 
   are dense vectors. This routine implements algorithm 2 for this problem. 
   Also, it provides a utility function to query size of buffer used. */
lwsparseStatus_t LWSPARSEAPI lwsparseXbsrsv2_zeroPivot(lwsparseHandle_t handle,
                                                       bsrsv2Info_t info,
                                                       int *position);


lwsparseStatus_t LWSPARSEAPI lwsparseSbsrsv2_bufferSize(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        lwsparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        float *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockDim,
                                                        bsrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrsv2_bufferSize(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        lwsparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        double *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockDim,
                                                        bsrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrsv2_bufferSize(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        lwsparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        lwComplex *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockDim,
                                                        bsrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrsv2_bufferSize(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        lwsparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        lwDoubleComplex *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockDim,
                                                        bsrsv2Info_t info,
                                                        int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseSbsrsv2_bufferSizeExt(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        lwsparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        float *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockSize,
                                                        bsrsv2Info_t info,
                                                        size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrsv2_bufferSizeExt(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        lwsparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        double *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockSize,
                                                        bsrsv2Info_t info,
                                                        size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrsv2_bufferSizeExt(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        lwsparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        lwComplex *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockSize,
                                                        bsrsv2Info_t info,
                                                        size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrsv2_bufferSizeExt(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        lwsparseOperation_t transA,
                                                        int mb,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        lwDoubleComplex *bsrSortedValA,
                                                        const int *bsrSortedRowPtrA,
                                                        const int *bsrSortedColIndA,
                                                        int blockSize,
                                                        bsrsv2Info_t info,
                                                        size_t *pBufferSize);


lwsparseStatus_t LWSPARSEAPI lwsparseSbsrsv2_analysis(lwsparseHandle_t handle,
                                                      lwsparseDirection_t dirA,
                                                      lwsparseOperation_t transA,
                                                      int mb,
                                                      int nnzb,
                                                      const lwsparseMatDescr_t descrA,
                                                      const float *bsrSortedValA,
                                                      const int *bsrSortedRowPtrA,
                                                      const int *bsrSortedColIndA,
                                                      int blockDim,
                                                      bsrsv2Info_t info,
                                                      lwsparseSolvePolicy_t policy,
                                                      void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrsv2_analysis(lwsparseHandle_t handle,
                                                      lwsparseDirection_t dirA,
                                                      lwsparseOperation_t transA,
                                                      int mb,
                                                      int nnzb,
                                                      const lwsparseMatDescr_t descrA,
                                                      const double *bsrSortedValA,
                                                      const int *bsrSortedRowPtrA,
                                                      const int *bsrSortedColIndA,
                                                      int blockDim,
                                                      bsrsv2Info_t info,
                                                      lwsparseSolvePolicy_t policy,
                                                      void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrsv2_analysis(lwsparseHandle_t handle,
                                                      lwsparseDirection_t dirA,
                                                      lwsparseOperation_t transA,
                                                      int mb,
                                                      int nnzb,
                                                      const lwsparseMatDescr_t descrA,
                                                      const lwComplex *bsrSortedValA,
                                                      const int *bsrSortedRowPtrA,
                                                      const int *bsrSortedColIndA,
                                                      int blockDim,
                                                      bsrsv2Info_t info,
                                                      lwsparseSolvePolicy_t policy,
                                                      void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrsv2_analysis(lwsparseHandle_t handle,
                                                      lwsparseDirection_t dirA,
                                                      lwsparseOperation_t transA,
                                                      int mb,
                                                      int nnzb,
                                                      const lwsparseMatDescr_t descrA,
                                                      const lwDoubleComplex *bsrSortedValA,
                                                      const int *bsrSortedRowPtrA,
                                                      const int *bsrSortedColIndA,
                                                      int blockDim,
                                                      bsrsv2Info_t info,
                                                      lwsparseSolvePolicy_t policy,
                                                      void *pBuffer);


lwsparseStatus_t LWSPARSEAPI lwsparseSbsrsv2_solve(lwsparseHandle_t handle,
                                                   lwsparseDirection_t dirA,
                                                   lwsparseOperation_t transA,
                                                   int mb,
                                                   int nnzb,
                                                   const float *alpha,
                                                   const lwsparseMatDescr_t descrA,
                                                   const float *bsrSortedValA,
                                                   const int *bsrSortedRowPtrA,
                                                   const int *bsrSortedColIndA,
                                                   int blockDim,
                                                   bsrsv2Info_t info,
                                                   const float *f,
                                                   float *x,
                                                   lwsparseSolvePolicy_t policy,
                                                   void *pBuffer);


lwsparseStatus_t LWSPARSEAPI lwsparseDbsrsv2_solve(lwsparseHandle_t handle,
                                                   lwsparseDirection_t dirA,
                                                   lwsparseOperation_t transA,
                                                   int mb,
                                                   int nnzb,
                                                   const double *alpha,
                                                   const lwsparseMatDescr_t descrA,
                                                   const double *bsrSortedValA,
                                                   const int *bsrSortedRowPtrA,
                                                   const int *bsrSortedColIndA,
                                                   int blockDim,
                                                   bsrsv2Info_t info,
                                                   const double *f,
                                                   double *x, 
                                                   lwsparseSolvePolicy_t policy,
                                                   void *pBuffer);


lwsparseStatus_t LWSPARSEAPI lwsparseCbsrsv2_solve(lwsparseHandle_t handle,
                                                   lwsparseDirection_t dirA,
                                                   lwsparseOperation_t transA,
                                                   int mb,
                                                   int nnzb,
                                                   const lwComplex *alpha,
                                                   const lwsparseMatDescr_t descrA,
                                                   const lwComplex *bsrSortedValA,
                                                   const int *bsrSortedRowPtrA,
                                                   const int *bsrSortedColIndA,
                                                   int blockDim,
                                                   bsrsv2Info_t info,
                                                   const lwComplex *f,
                                                   lwComplex *x,
                                                   lwsparseSolvePolicy_t policy,
                                                   void *pBuffer);


lwsparseStatus_t LWSPARSEAPI lwsparseZbsrsv2_solve(lwsparseHandle_t handle,
                                                   lwsparseDirection_t dirA,
                                                   lwsparseOperation_t transA,
                                                   int mb,
                                                   int nnzb,
                                                   const lwDoubleComplex *alpha,
                                                   const lwsparseMatDescr_t descrA,
                                                   const lwDoubleComplex *bsrSortedValA,
                                                   const int *bsrSortedRowPtrA,
                                                   const int *bsrSortedColIndA,
                                                   int blockDim,
                                                   bsrsv2Info_t info,
                                                   const lwDoubleComplex *f,
                                                   lwDoubleComplex *x,
                                                   lwsparseSolvePolicy_t policy,
                                                   void *pBuffer);

/* Description: Solution of triangular linear system op(A) * x = alpha * f, 
   where A is a sparse matrix in HYB storage format, rhs f and solution x 
   are dense vectors. */
lwsparseStatus_t LWSPARSEAPI lwsparseShybsv_analysis(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     const lwsparseMatDescr_t descrA, 
                                                     lwsparseHybMat_t hybA,
                                                     lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseDhybsv_analysis(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     const lwsparseMatDescr_t descrA, 
                                                     lwsparseHybMat_t hybA,
                                                     lwsparseSolveAnalysisInfo_t info);
    
lwsparseStatus_t LWSPARSEAPI lwsparseChybsv_analysis(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     const lwsparseMatDescr_t descrA, 
                                                     lwsparseHybMat_t hybA,
                                                     lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseZhybsv_analysis(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     const lwsparseMatDescr_t descrA, 
                                                     lwsparseHybMat_t hybA,
                                                     lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseShybsv_solve(lwsparseHandle_t handle, 
                                                  lwsparseOperation_t trans, 
                                                  const float *alpha, 
                                                  const lwsparseMatDescr_t descrA,
                                                  const lwsparseHybMat_t hybA,
                                                  lwsparseSolveAnalysisInfo_t info,
                                                  const float *f,
                                                  float *x);

lwsparseStatus_t LWSPARSEAPI lwsparseChybsv_solve(lwsparseHandle_t handle, 
                                                  lwsparseOperation_t trans,
                                                  const lwComplex *alpha, 
                                                  const lwsparseMatDescr_t descrA,
                                                  const lwsparseHybMat_t hybA,
                                                  lwsparseSolveAnalysisInfo_t info,
                                                  const lwComplex *f,
                                                  lwComplex *x);

lwsparseStatus_t LWSPARSEAPI lwsparseDhybsv_solve(lwsparseHandle_t handle,
                                                  lwsparseOperation_t trans,
                                                  const double *alpha, 
                                                  const lwsparseMatDescr_t descrA,
                                                  const lwsparseHybMat_t hybA, 
                                                  lwsparseSolveAnalysisInfo_t info,
                                                  const double *f,
                                                  double *x);

lwsparseStatus_t LWSPARSEAPI lwsparseZhybsv_solve(lwsparseHandle_t handle, 
                                                  lwsparseOperation_t trans,
                                                  const lwDoubleComplex *alpha, 
                                                  const lwsparseMatDescr_t descrA,
                                                  const lwsparseHybMat_t hybA,
                                                  lwsparseSolveAnalysisInfo_t info,
                                                  const lwDoubleComplex *f,
                                                  lwDoubleComplex *x);


/* --- Sparse Level 3 routines --- */           
 
/* Description: sparse - dense matrix multiplication C = alpha * op(A) * B  + beta * C, 
   where A is a sparse matrix in CSR format, B and C are dense tall matrices.  */                 
lwsparseStatus_t LWSPARSEAPI lwsparseScsrmm(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int k,  
                                            int nnz,
                                            const float *alpha,
                                            const lwsparseMatDescr_t descrA, 
                                            const float  *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const float *B, 
                                            int ldb, 
                                            const float *beta, 
                                            float *C, 
                                            int ldc);
                     
lwsparseStatus_t LWSPARSEAPI lwsparseDcsrmm(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int k,  
                                            int nnz,
                                            const double *alpha,
                                            const lwsparseMatDescr_t descrA, 
                                            const double *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const double *B, 
                                            int ldb, 
                                            const double *beta, 
                                            double *C, 
                                            int ldc);
                     
lwsparseStatus_t LWSPARSEAPI lwsparseCcsrmm(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int k,  
                                            int nnz,
                                            const lwComplex *alpha,
                                            const lwsparseMatDescr_t descrA, 
                                            const lwComplex  *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const lwComplex *B, 
                                            int ldb, 
                                            const lwComplex *beta, 
                                            lwComplex *C, 
                                            int ldc);
                     
lwsparseStatus_t LWSPARSEAPI lwsparseZcsrmm(lwsparseHandle_t handle,
                                            lwsparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int k,  
                                            int nnz,
                                            const lwDoubleComplex *alpha,
                                            const lwsparseMatDescr_t descrA, 
                                            const lwDoubleComplex  *csrSortedValA, 
                                            const int *csrSortedRowPtrA, 
                                            const int *csrSortedColIndA, 
                                            const lwDoubleComplex *B, 
                                            int ldb, 
                                            const lwDoubleComplex *beta, 
                                            lwDoubleComplex *C, 
                                            int ldc);    

/* Description: sparse - dense matrix multiplication C = alpha * op(A) * B  + beta * C, 
   where A is a sparse matrix in CSR format, B and C are dense tall matrices.
   This routine allows transposition of matrix B, which may improve performance. */     
lwsparseStatus_t LWSPARSEAPI lwsparseScsrmm2(lwsparseHandle_t handle,
                                             lwsparseOperation_t transA,
                                             lwsparseOperation_t transB,
                                             int m,
                                             int n,
                                             int k,
                                             int nnz,
                                             const float *alpha,
                                             const lwsparseMatDescr_t descrA,
                                             const float *csrSortedValA,
                                             const int *csrSortedRowPtrA,
                                             const int *csrSortedColIndA,
                                             const float *B,
                                             int ldb,
                                             const float *beta,
                                             float *C,
                                             int ldc);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrmm2(lwsparseHandle_t handle,
                                             lwsparseOperation_t transA,
                                             lwsparseOperation_t transB,
                                             int m,
                                             int n,
                                             int k,
                                             int nnz,
                                             const double *alpha,
                                             const lwsparseMatDescr_t descrA,
                                             const double *csrSortedValA,
                                             const int *csrSortedRowPtrA,
                                             const int *csrSortedColIndA,
                                             const double *B,
                                             int ldb,
                                             const double *beta,
                                             double *C,
                                             int ldc);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrmm2(lwsparseHandle_t handle,
                                             lwsparseOperation_t transA,
                                             lwsparseOperation_t transB,
                                             int m,
                                             int n,
                                             int k,
                                             int nnz,
                                             const lwComplex *alpha,
                                             const lwsparseMatDescr_t descrA,
                                             const lwComplex *csrSortedValA,
                                             const int *csrSortedRowPtrA,
                                             const int *csrSortedColIndA,
                                             const lwComplex *B,
                                             int ldb,
                                             const lwComplex *beta,
                                             lwComplex *C,
                                             int ldc);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrmm2(lwsparseHandle_t handle,
                                             lwsparseOperation_t transA,
                                             lwsparseOperation_t transB,
                                             int m,
                                             int n,
                                             int k,
                                             int nnz,
                                             const lwDoubleComplex *alpha,
                                             const lwsparseMatDescr_t descrA,
                                             const lwDoubleComplex *csrSortedValA,
                                             const int *csrSortedRowPtrA,
                                             const int *csrSortedColIndA,
                                             const lwDoubleComplex *B,
                                             int ldb,
                                             const lwDoubleComplex *beta,
                                             lwDoubleComplex *C,
                                             int ldc);

/* Description: sparse - dense matrix multiplication C = alpha * op(A) * B  + beta * C, 
   where A is a sparse matrix in block-CSR format, B and C are dense tall matrices.
   This routine allows transposition of matrix B, which may improve performance. */ 
lwsparseStatus_t LWSPARSEAPI lwsparseSbsrmm(lwsparseHandle_t handle,
                                            lwsparseDirection_t dirA,
                                            lwsparseOperation_t transA,
                                            lwsparseOperation_t transB,
                                            int mb,
                                            int n,
                                            int kb,
                                            int nnzb,
                                            const float *alpha,
                                            const lwsparseMatDescr_t descrA,
                                            const float *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            const int  blockSize,
                                            const float *B,
                                            const int ldb,
                                            const float *beta,
                                            float *C,
                                            int ldc);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrmm(lwsparseHandle_t handle,
                                            lwsparseDirection_t dirA,
                                            lwsparseOperation_t transA,
                                            lwsparseOperation_t transB,
                                            int mb,
                                            int n, 
                                            int kb, 
                                            int nnzb,
                                            const double *alpha,
                                            const lwsparseMatDescr_t descrA,
                                            const double *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            const int  blockSize,
                                            const double *B,
                                            const int ldb,
                                            const double *beta,
                                            double *C,
                                            int ldc);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrmm(lwsparseHandle_t handle,
                                            lwsparseDirection_t dirA,
                                            lwsparseOperation_t transA,
                                            lwsparseOperation_t transB,
                                            int mb,
                                            int n, 
                                            int kb, 
                                            int nnzb,
                                            const lwComplex *alpha,
                                            const lwsparseMatDescr_t descrA,
                                            const lwComplex *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            const int  blockSize,
                                            const lwComplex *B,
                                            const int ldb,
                                            const lwComplex *beta,
                                            lwComplex *C,
                                            int ldc);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrmm(lwsparseHandle_t handle,
                                            lwsparseDirection_t dirA,
                                            lwsparseOperation_t transA,
                                            lwsparseOperation_t transB,
                                            int mb,
                                            int n, 
                                            int kb, 
                                            int nnzb,
                                            const lwDoubleComplex *alpha,
                                            const lwsparseMatDescr_t descrA,
                                            const lwDoubleComplex *bsrSortedValA,
                                            const int *bsrSortedRowPtrA,
                                            const int *bsrSortedColIndA,
                                            const int  blockSize,
                                            const lwDoubleComplex *B,
                                            const int ldb,
                                            const lwDoubleComplex *beta,
                                            lwDoubleComplex *C,
                                            int ldc);


/* Description: dense - sparse matrix multiplication C = alpha * A * B  + beta * C, 
   where A is column-major dense matrix, B is a sparse matrix in CSC format, 
   and C is column-major dense matrix. */
lwsparseStatus_t  LWSPARSEAPI lwsparseSgemmi(lwsparseHandle_t handle,
                                             int m,
                                             int n,
					     int k,
					     int nnz, 
                                             const float *alpha, /* host or device pointer */
                                             const float *A,
                                             int lda,
                                             const float *cscValB,
					     const int *cscColPtrB, 
					     const int *cscRowIndB, 
                                             const float *beta, /* host or device pointer */
                                             float *C,
                                             int ldc);

lwsparseStatus_t  LWSPARSEAPI lwsparseDgemmi(lwsparseHandle_t handle,
                                             int m,
                                             int n,
					     int k,
					     int nnz, 
                                             const double *alpha, /* host or device pointer */
                                             const double *A,
                                             int lda,
                                             const double *cscValB,
					     const int *cscColPtrB, 
					     const int *cscRowIndB, 
                                             const double *beta, /* host or device pointer */
                                             double *C,
                                             int ldc);

lwsparseStatus_t  LWSPARSEAPI lwsparseCgemmi(lwsparseHandle_t handle,
                                             int m,
                                             int n,
					     int k,
					     int nnz, 
                                             const lwComplex *alpha, /* host or device pointer */
                                             const lwComplex *A,
                                             int lda,
                                             const lwComplex *cscValB,
					     const int *cscColPtrB, 
					     const int *cscRowIndB, 
                                             const lwComplex *beta, /* host or device pointer */
                                             lwComplex *C,
                                             int ldc);

lwsparseStatus_t  LWSPARSEAPI lwsparseZgemmi(lwsparseHandle_t handle,
                                             int m,
                                             int n,
					     int k,
					     int nnz, 
                                             const lwDoubleComplex *alpha, /* host or device pointer */
                                             const lwDoubleComplex *A,
                                             int lda,
                                             const lwDoubleComplex *cscValB,
					     const int *cscColPtrB, 
					     const int *cscRowIndB, 
                                             const lwDoubleComplex *beta, /* host or device pointer */
                                             lwDoubleComplex *C,
                                             int ldc);


/* Description: Solution of triangular linear system op(A) * X = alpha * F, 
   with multiple right-hand-sides, where A is a sparse matrix in CSR storage 
   format, rhs F and solution X are dense tall matrices. 
   This routine implements algorithm 1 for this problem. */
lwsparseStatus_t LWSPARSEAPI lwsparseScsrsm_analysis(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const lwsparseMatDescr_t descrA, 
                                                     const float *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrsm_analysis(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const lwsparseMatDescr_t descrA, 
                                                     const double *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrsm_analysis(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const lwsparseMatDescr_t descrA, 
                                                     const lwComplex *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrsm_analysis(lwsparseHandle_t handle, 
                                                     lwsparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const lwsparseMatDescr_t descrA, 
                                                     const lwDoubleComplex *csrSortedValA, 
                                                     const int *csrSortedRowPtrA, 
                                                     const int *csrSortedColIndA, 
                                                     lwsparseSolveAnalysisInfo_t info); 


lwsparseStatus_t LWSPARSEAPI lwsparseScsrsm_solve(lwsparseHandle_t handle, 
                                                  lwsparseOperation_t transA, 
                                                  int m,
                                                  int n,
                                                  const float *alpha, 
                                                  const lwsparseMatDescr_t descrA, 
                                                  const float *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  lwsparseSolveAnalysisInfo_t info, 
                                                  const float *B, 
                                                  int ldb,
                                                  float *X,
                                                  int ldx);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrsm_solve(lwsparseHandle_t handle, 
                                                  lwsparseOperation_t transA, 
                                                  int m, 
                                                  int n,
                                                  const double *alpha, 
                                                  const lwsparseMatDescr_t descrA, 
                                                  const double *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  lwsparseSolveAnalysisInfo_t info, 
                                                  const double *B, 
                                                  int ldb,
                                                  double *X,
                                                  int ldx);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrsm_solve(lwsparseHandle_t handle, 
                                                  lwsparseOperation_t transA, 
                                                  int m, 
                                                  int n,
                                                  const lwComplex *alpha, 
                                                  const lwsparseMatDescr_t descrA, 
                                                  const lwComplex *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  lwsparseSolveAnalysisInfo_t info, 
                                                  const lwComplex *B,
                                                  int ldb,
                                                  lwComplex *X,
                                                  int ldx);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrsm_solve(lwsparseHandle_t handle, 
                                                  lwsparseOperation_t transA, 
                                                  int m, 
                                                  int n,
                                                  const lwDoubleComplex *alpha, 
                                                  const lwsparseMatDescr_t descrA, 
                                                  const lwDoubleComplex *csrSortedValA, 
                                                  const int *csrSortedRowPtrA, 
                                                  const int *csrSortedColIndA, 
                                                  lwsparseSolveAnalysisInfo_t info, 
                                                  const lwDoubleComplex *B,
                                                  int ldb,
                                                  lwDoubleComplex *X,
                                                  int ldx);                                                                 
                    
lwsparseStatus_t LWSPARSEAPI lwsparseCreateCsrsm2Info(
    csrsm2Info_t *info);

lwsparseStatus_t LWSPARSEAPI lwsparseDestroyCsrsm2Info(
    csrsm2Info_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseXcsrsm2_zeroPivot(
    lwsparseHandle_t handle,
    csrsm2Info_t info,
    int *position);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrsm2_bufferSizeExt(
    lwsparseHandle_t handle,
    int algo, /* algo = 0, 1 */
    lwsparseOperation_t transA,
    lwsparseOperation_t transB,
    int m,
    int nrhs,
    int nnz,
    const float *alpha,
    const lwsparseMatDescr_t descrA,
    const float *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const float *B,
    int ldb,
    csrsm2Info_t info,
    lwsparseSolvePolicy_t policy,
    size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrsm2_bufferSizeExt(
    lwsparseHandle_t handle,
    int algo, /* algo = 0, 1 */
    lwsparseOperation_t transA,
    lwsparseOperation_t transB,
    int m,
    int nrhs,
    int nnz,
    const double *alpha,
    const lwsparseMatDescr_t descrA,
    const double *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const double *B,
    int ldb,
    csrsm2Info_t info,
    lwsparseSolvePolicy_t policy,
    size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrsm2_bufferSizeExt(
    lwsparseHandle_t handle,
    int algo, /* algo = 0, 1 */
    lwsparseOperation_t transA,
    lwsparseOperation_t transB,
    int m,
    int nrhs,
    int nnz,
    const lwComplex *alpha,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const lwComplex *B,
    int ldb,
    csrsm2Info_t info,
    lwsparseSolvePolicy_t policy,
    size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrsm2_bufferSizeExt(
    lwsparseHandle_t handle,
    int algo, /* algo = 0, 1 */
    lwsparseOperation_t transA,
    lwsparseOperation_t transB,
    int m,
    int nrhs,
    int nnz,
    const lwDoubleComplex *alpha,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const lwDoubleComplex *B,
    int ldb,
    csrsm2Info_t info,
    lwsparseSolvePolicy_t policy,
    size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrsm2_analysis(
    lwsparseHandle_t handle,
    int algo, /* algo = 0, 1 */
    lwsparseOperation_t transA,
    lwsparseOperation_t transB,
    int m,
    int nrhs,
    int nnz,
    const float *alpha,
    const lwsparseMatDescr_t descrA,
    const float *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const float *B,
    int ldb,
    csrsm2Info_t info,
    lwsparseSolvePolicy_t policy,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrsm2_analysis(
    lwsparseHandle_t handle,
    int algo, /* algo = 0, 1 */
    lwsparseOperation_t transA,
    lwsparseOperation_t transB,
    int m,
    int nrhs,
    int nnz,
    const double *alpha,
    const lwsparseMatDescr_t descrA,
    const double *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const double *B,
    int ldb,
    csrsm2Info_t info,
    lwsparseSolvePolicy_t policy,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrsm2_analysis(
    lwsparseHandle_t handle,
    int algo, /* algo = 0, 1 */
    lwsparseOperation_t transA,
    lwsparseOperation_t transB,
    int m,
    int nrhs,
    int nnz,
    const lwComplex *alpha,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const lwComplex *B,
    int ldb,
    csrsm2Info_t info,
    lwsparseSolvePolicy_t policy,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrsm2_analysis(
    lwsparseHandle_t handle,
    int algo, /* algo = 0, 1 */
    lwsparseOperation_t transA,
    lwsparseOperation_t transB,
    int m,
    int nrhs,
    int nnz,
    const lwDoubleComplex *alpha,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const lwDoubleComplex *B,
    int ldb,
    csrsm2Info_t info,
    lwsparseSolvePolicy_t policy,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrsm2_solve(
    lwsparseHandle_t handle,
    int algo, /* algo = 0, 1 */
    lwsparseOperation_t transA,
    lwsparseOperation_t transB,
    int m,
    int nrhs,
    int nnz,
    const float *alpha,
    const lwsparseMatDescr_t descrA,
    const float *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float *B,
    int ldb,
    csrsm2Info_t info,
    lwsparseSolvePolicy_t policy,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrsm2_solve(
    lwsparseHandle_t handle,
    int algo, /* algo = 0, 1 */
    lwsparseOperation_t transA,
    lwsparseOperation_t transB,
    int m,
    int nrhs,
    int nnz,
    const double *alpha,
    const lwsparseMatDescr_t descrA,
    const double *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    double *B,
    int ldb,
    csrsm2Info_t info,
    lwsparseSolvePolicy_t policy,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrsm2_solve(
    lwsparseHandle_t handle,
    int algo, /* algo = 0, 1 */
    lwsparseOperation_t transA,
    lwsparseOperation_t transB,
    int m,
    int nrhs,
    int nnz,
    const lwComplex *alpha,
    const lwsparseMatDescr_t descrA,
    const lwComplex *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    lwComplex *B,
    int ldb,
    csrsm2Info_t info,
    lwsparseSolvePolicy_t policy,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrsm2_solve(
    lwsparseHandle_t handle,
    int algo, /* algo = 0, 1 */
    lwsparseOperation_t transA,
    lwsparseOperation_t transB,
    int m,
    int nrhs,
    int nnz,
    const lwDoubleComplex *alpha,
    const lwsparseMatDescr_t descrA,
    const lwDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    lwDoubleComplex *B,
    int ldb,
    csrsm2Info_t info,
    lwsparseSolvePolicy_t policy,
    void *pBuffer);


/* Description: Solution of triangular linear system op(A) * X = alpha * B, 
   with multiple right-hand-sides, where A is a sparse matrix in CSR storage 
   format, rhs B and solution X are dense tall matrices.
   This routine implements algorithm 2 for this problem. */
lwsparseStatus_t LWSPARSEAPI lwsparseXbsrsm2_zeroPivot(lwsparseHandle_t handle,
                                                       bsrsm2Info_t info,
                                                       int *position);

lwsparseStatus_t LWSPARSEAPI lwsparseSbsrsm2_bufferSize(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        lwsparseOperation_t transA,
                                                        lwsparseOperation_t transXY,
                                                        int mb,
                                                        int n,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        float *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockSize,
                                                        bsrsm2Info_t info,
                                                        int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrsm2_bufferSize(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        lwsparseOperation_t transA,
                                                        lwsparseOperation_t transXY,
                                                        int mb,
                                                        int n,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        double *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockSize,
                                                        bsrsm2Info_t info,
                                                        int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrsm2_bufferSize(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        lwsparseOperation_t transA,
                                                        lwsparseOperation_t transXY,
                                                        int mb,
                                                        int n,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        lwComplex *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockSize,
                                                        bsrsm2Info_t info,
                                                        int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrsm2_bufferSize(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        lwsparseOperation_t transA,
                                                        lwsparseOperation_t transXY,
                                                        int mb,
                                                        int n,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        lwDoubleComplex *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockSize,
                                                        bsrsm2Info_t info,
                                                        int *pBufferSizeInBytes);


lwsparseStatus_t LWSPARSEAPI lwsparseSbsrsm2_bufferSizeExt(lwsparseHandle_t handle,
                                                           lwsparseDirection_t dirA,
                                                           lwsparseOperation_t transA,
                                                           lwsparseOperation_t transB,
                                                           int mb,
                                                           int n,
                                                           int nnzb,
                                                           const lwsparseMatDescr_t descrA,
                                                           float *bsrSortedVal,
                                                           const int *bsrSortedRowPtr,
                                                           const int *bsrSortedColInd,
                                                           int blockSize,
                                                           bsrsm2Info_t info,
                                                           size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrsm2_bufferSizeExt(lwsparseHandle_t handle,
                                                           lwsparseDirection_t dirA,
                                                           lwsparseOperation_t transA,
                                                           lwsparseOperation_t transB,
                                                           int mb,
                                                           int n,
                                                           int nnzb,
                                                           const lwsparseMatDescr_t descrA,
                                                           double *bsrSortedVal,
                                                           const int *bsrSortedRowPtr,
                                                           const int *bsrSortedColInd,
                                                           int blockSize,
                                                           bsrsm2Info_t info,
                                                           size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrsm2_bufferSizeExt(lwsparseHandle_t handle,
                                                           lwsparseDirection_t dirA,
                                                           lwsparseOperation_t transA,
                                                           lwsparseOperation_t transB,
                                                           int mb,
                                                           int n,
                                                           int nnzb,
                                                           const lwsparseMatDescr_t descrA,
                                                           lwComplex *bsrSortedVal,
                                                           const int *bsrSortedRowPtr,
                                                           const int *bsrSortedColInd,
                                                           int blockSize,
                                                           bsrsm2Info_t info,
                                                           size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrsm2_bufferSizeExt(lwsparseHandle_t handle,
                                                           lwsparseDirection_t dirA,
                                                           lwsparseOperation_t transA,
                                                           lwsparseOperation_t transB,
                                                           int mb,
                                                           int n,
                                                           int nnzb,
                                                           const lwsparseMatDescr_t descrA,
                                                           lwDoubleComplex *bsrSortedVal,
                                                           const int *bsrSortedRowPtr,
                                                           const int *bsrSortedColInd,
                                                           int blockSize,
                                                           bsrsm2Info_t info,
                                                           size_t *pBufferSize);


lwsparseStatus_t LWSPARSEAPI lwsparseSbsrsm2_analysis(lwsparseHandle_t handle,
                                                      lwsparseDirection_t dirA,
                                                      lwsparseOperation_t transA,
                                                      lwsparseOperation_t transXY,
                                                      int mb,
                                                      int n,
                                                      int nnzb,
                                                      const lwsparseMatDescr_t descrA,
                                                      const float *bsrSortedVal,
                                                      const int *bsrSortedRowPtr,
                                                      const int *bsrSortedColInd,
                                                      int blockSize,
                                                      bsrsm2Info_t info,
                                                      lwsparseSolvePolicy_t policy,
                                                      void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrsm2_analysis(lwsparseHandle_t handle,
                                                      lwsparseDirection_t dirA,
                                                      lwsparseOperation_t transA,
                                                      lwsparseOperation_t transXY,
                                                      int mb,
                                                      int n,
                                                      int nnzb,
                                                      const lwsparseMatDescr_t descrA,
                                                      const double *bsrSortedVal,
                                                      const int *bsrSortedRowPtr,
                                                      const int *bsrSortedColInd,
                                                      int blockSize,
                                                      bsrsm2Info_t info,
                                                      lwsparseSolvePolicy_t policy,
                                                      void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrsm2_analysis(lwsparseHandle_t handle,
                                                      lwsparseDirection_t dirA,
                                                      lwsparseOperation_t transA,
                                                      lwsparseOperation_t transXY,
                                                      int mb,
                                                      int n,
                                                      int nnzb,
                                                      const lwsparseMatDescr_t descrA,
                                                      const lwComplex *bsrSortedVal,
                                                      const int *bsrSortedRowPtr,
                                                      const int *bsrSortedColInd,
                                                      int blockSize,
                                                      bsrsm2Info_t info,
                                                      lwsparseSolvePolicy_t policy,
                                                      void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrsm2_analysis(lwsparseHandle_t handle,
                                                      lwsparseDirection_t dirA,
                                                      lwsparseOperation_t transA,
                                                      lwsparseOperation_t transXY,
                                                      int mb,
                                                      int n,
                                                      int nnzb,
                                                      const lwsparseMatDescr_t descrA,
                                                      const lwDoubleComplex *bsrSortedVal,
                                                      const int *bsrSortedRowPtr,
                                                      const int *bsrSortedColInd,
                                                      int blockSize,
                                                      bsrsm2Info_t info,
                                                      lwsparseSolvePolicy_t policy,
                                                      void *pBuffer);


lwsparseStatus_t LWSPARSEAPI lwsparseSbsrsm2_solve(lwsparseHandle_t handle,
                                                   lwsparseDirection_t dirA,
                                                   lwsparseOperation_t transA,
                                                   lwsparseOperation_t transXY,
                                                   int mb,
                                                   int n,
                                                   int nnzb,
                                                   const float *alpha,
                                                   const lwsparseMatDescr_t descrA,
                                                   const float *bsrSortedVal,
                                                   const int *bsrSortedRowPtr,
                                                   const int *bsrSortedColInd,
                                                   int blockSize,
                                                   bsrsm2Info_t info,
                                                   const float *B,
                                                   int ldb,
                                                   float *X,
                                                   int ldx,
                                                   lwsparseSolvePolicy_t policy,
                                                   void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrsm2_solve(lwsparseHandle_t handle,
                                                   lwsparseDirection_t dirA,
                                                   lwsparseOperation_t transA,
                                                   lwsparseOperation_t transXY,
                                                   int mb,
                                                   int n,
                                                   int nnzb,
                                                   const double *alpha,
                                                   const lwsparseMatDescr_t descrA,
                                                   const double *bsrSortedVal,
                                                   const int *bsrSortedRowPtr,
                                                   const int *bsrSortedColInd,
                                                   int blockSize,
                                                   bsrsm2Info_t info,
                                                   const double *B,
                                                   int ldb,
                                                   double *X,
                                                   int ldx,
                                                   lwsparseSolvePolicy_t policy,
                                                   void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrsm2_solve(lwsparseHandle_t handle,
                                                   lwsparseDirection_t dirA,
                                                   lwsparseOperation_t transA,
                                                   lwsparseOperation_t transXY,
                                                   int mb,
                                                   int n,
                                                   int nnzb,
                                                   const lwComplex *alpha,
                                                   const lwsparseMatDescr_t descrA,
                                                   const lwComplex *bsrSortedVal,
                                                   const int *bsrSortedRowPtr,
                                                   const int *bsrSortedColInd,
                                                   int blockSize,
                                                   bsrsm2Info_t info,
                                                   const lwComplex *B,
                                                   int ldb,
                                                   lwComplex *X,
                                                   int ldx,
                                                   lwsparseSolvePolicy_t policy,
                                                   void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrsm2_solve(lwsparseHandle_t handle,
                                                   lwsparseDirection_t dirA,
                                                   lwsparseOperation_t transA,
                                                   lwsparseOperation_t transXY,
                                                   int mb,
                                                   int n,
                                                   int nnzb,
                                                   const lwDoubleComplex *alpha,
                                                   const lwsparseMatDescr_t descrA,
                                                   const lwDoubleComplex *bsrSortedVal,
                                                   const int *bsrSortedRowPtr,
                                                   const int *bsrSortedColInd,
                                                   int blockSize,
                                                   bsrsm2Info_t info,
                                                   const lwDoubleComplex *B,
                                                   int ldb,
                                                   lwDoubleComplex *X,
                                                   int ldx,
                                                   lwsparseSolvePolicy_t policy,
                                                   void *pBuffer);


/* --- Preconditioners --- */ 

/* Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
   of the matrix A stored in CSR format based on the information in the opaque 
   structure info that was obtained from the analysis phase (csrsv_analysis). 
   This routine implements algorithm 1 for this problem. */
lwsparseStatus_t LWSPARSEAPI lwsparseCsrilu0Ex(lwsparseHandle_t handle, 
                                              lwsparseOperation_t trans, 
                                              int m, 
                                              const lwsparseMatDescr_t descrA, 
                                              void *csrSortedValA_ValM, 
                                              lwdaDataType csrSortedValA_ValMtype,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              lwsparseSolveAnalysisInfo_t info,
                                              lwdaDataType exelwtiontype);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrilu0(lwsparseHandle_t handle, 
                                              lwsparseOperation_t trans, 
                                              int m, 
                                              const lwsparseMatDescr_t descrA, 
                                              float *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrilu0(lwsparseHandle_t handle, 
                                              lwsparseOperation_t trans, 
                                              int m, 
                                              const lwsparseMatDescr_t descrA, 
                                              double *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA, 
                                              lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrilu0(lwsparseHandle_t handle, 
                                              lwsparseOperation_t trans, 
                                              int m, 
                                              const lwsparseMatDescr_t descrA, 
                                              lwComplex *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA, 
                                              lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrilu0(lwsparseHandle_t handle, 
                                              lwsparseOperation_t trans, 
                                              int m, 
                                              const lwsparseMatDescr_t descrA, 
                                              lwDoubleComplex *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA, 
                                              lwsparseSolveAnalysisInfo_t info);

/* Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
   of the matrix A stored in CSR format based on the information in the opaque 
   structure info that was obtained from the analysis phase (csrsv2_analysis).
   This routine implements algorithm 2 for this problem. */
lwsparseStatus_t LWSPARSEAPI lwsparseScsrilu02_numericBoost(lwsparseHandle_t handle,
                                                            csrilu02Info_t info,
                                                            int enable_boost,    
                                                            double *tol,
                                                            float *boost_val);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrilu02_numericBoost(lwsparseHandle_t handle,
                                                            csrilu02Info_t info,
                                                            int enable_boost,    
                                                            double *tol,
                                                            double *boost_val);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrilu02_numericBoost(lwsparseHandle_t handle,
                                                            csrilu02Info_t info,
                                                            int enable_boost,    
                                                            double *tol,
                                                            lwComplex *boost_val);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrilu02_numericBoost(lwsparseHandle_t handle,
                                                            csrilu02Info_t info,
                                                            int enable_boost,    
                                                            double *tol,
                                                            lwDoubleComplex *boost_val);

lwsparseStatus_t LWSPARSEAPI lwsparseXcsrilu02_zeroPivot(lwsparseHandle_t handle,
                                                         csrilu02Info_t info,
                                                         int *position);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrilu02_bufferSize(lwsparseHandle_t handle,
                                                          int m,
                                                          int nnz,
                                                          const lwsparseMatDescr_t descrA,
                                                          float *csrSortedValA,
                                                          const int *csrSortedRowPtrA,
                                                          const int *csrSortedColIndA,
                                                          csrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrilu02_bufferSize(lwsparseHandle_t handle,
                                                          int m,
                                                          int nnz,
                                                          const lwsparseMatDescr_t descrA,
                                                          double *csrSortedValA,
                                                          const int *csrSortedRowPtrA,
                                                          const int *csrSortedColIndA,
                                                          csrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrilu02_bufferSize(lwsparseHandle_t handle,
                                                          int m,
                                                          int nnz,
                                                          const lwsparseMatDescr_t descrA,
                                                          lwComplex *csrSortedValA,
                                                          const int *csrSortedRowPtrA,
                                                          const int *csrSortedColIndA,
                                                          csrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrilu02_bufferSize(lwsparseHandle_t handle,
                                                          int m,
                                                          int nnz,
                                                          const lwsparseMatDescr_t descrA,
                                                          lwDoubleComplex *csrSortedValA,
                                                          const int *csrSortedRowPtrA,
                                                          const int *csrSortedColIndA,
                                                          csrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrilu02_bufferSizeExt(lwsparseHandle_t handle,
                                                             int m,
                                                             int nnz,
                                                             const lwsparseMatDescr_t descrA,
                                                             float *csrSortedVal,
                                                             const int *csrSortedRowPtr,
                                                             const int *csrSortedColInd,
                                                             csrilu02Info_t info,
                                                             size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrilu02_bufferSizeExt(lwsparseHandle_t handle,
                                                             int m,
                                                             int nnz,
                                                             const lwsparseMatDescr_t descrA,
                                                             double *csrSortedVal,
                                                             const int *csrSortedRowPtr,
                                                             const int *csrSortedColInd,
                                                             csrilu02Info_t info,
                                                             size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrilu02_bufferSizeExt(lwsparseHandle_t handle,
                                                             int m,
                                                             int nnz,
                                                             const lwsparseMatDescr_t descrA,
                                                             lwComplex *csrSortedVal,
                                                             const int *csrSortedRowPtr,
                                                             const int *csrSortedColInd,
                                                             csrilu02Info_t info,
                                                             size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrilu02_bufferSizeExt(lwsparseHandle_t handle,
                                                             int m,
                                                             int nnz,
                                                             const lwsparseMatDescr_t descrA,
                                                             lwDoubleComplex *csrSortedVal,
                                                             const int *csrSortedRowPtr,
                                                             const int *csrSortedColInd,
                                                             csrilu02Info_t info,
                                                             size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrilu02_analysis(lwsparseHandle_t handle,
                                                        int m,
                                                        int nnz,
                                                        const lwsparseMatDescr_t descrA,
                                                        const float *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrilu02Info_t info,
                                                        lwsparseSolvePolicy_t policy,
                                                        void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrilu02_analysis(lwsparseHandle_t handle,
                                                        int m,
                                                        int nnz,
                                                        const lwsparseMatDescr_t descrA,
                                                        const double *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrilu02Info_t info,
                                                        lwsparseSolvePolicy_t policy,
                                                        void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrilu02_analysis(lwsparseHandle_t handle,
                                                        int m,
                                                        int nnz,
                                                        const lwsparseMatDescr_t descrA,
                                                        const lwComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrilu02Info_t info,
                                                        lwsparseSolvePolicy_t policy,
                                                        void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrilu02_analysis(lwsparseHandle_t handle,
                                                        int m,
                                                        int nnz,
                                                        const lwsparseMatDescr_t descrA,
                                                        const lwDoubleComplex *csrSortedValA,
                                                        const int *csrSortedRowPtrA,
                                                        const int *csrSortedColIndA,
                                                        csrilu02Info_t info,
                                                        lwsparseSolvePolicy_t policy,
                                                        void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrilu02(lwsparseHandle_t handle,
                                               int m,
                                               int nnz,
                                               const lwsparseMatDescr_t descrA,
                                               float *csrSortedValA_valM,
                                               /* matrix A values are updated inplace 
                                                  to be the preconditioner M values */
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info,
                                               lwsparseSolvePolicy_t policy,
                                               void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrilu02(lwsparseHandle_t handle,
                                               int m,
                                               int nnz,
                                               const lwsparseMatDescr_t descrA,
                                               double *csrSortedValA_valM,
                                               /* matrix A values are updated inplace 
                                                  to be the preconditioner M values */
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info,
                                               lwsparseSolvePolicy_t policy,
                                               void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrilu02(lwsparseHandle_t handle,
                                               int m,
                                               int nnz,
                                               const lwsparseMatDescr_t descrA,
                                               lwComplex *csrSortedValA_valM,
                                               /* matrix A values are updated inplace 
                                                  to be the preconditioner M values */
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info,
                                               lwsparseSolvePolicy_t policy,
                                               void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrilu02(lwsparseHandle_t handle,
                                               int m,
                                               int nnz,
                                               const lwsparseMatDescr_t descrA,
                                               lwDoubleComplex *csrSortedValA_valM,
                                               /* matrix A values are updated inplace 
                                                  to be the preconditioner M values */
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info,
                                               lwsparseSolvePolicy_t policy,
                                               void *pBuffer);

/* Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
   of the matrix A stored in block-CSR format based on the information in the opaque 
   structure info that was obtained from the analysis phase (bsrsv2_analysis).
   This routine implements algorithm 2 for this problem. */
lwsparseStatus_t LWSPARSEAPI lwsparseSbsrilu02_numericBoost(lwsparseHandle_t handle,
                                                            bsrilu02Info_t info,
                                                            int enable_boost,
                                                            double *tol,
                                                            float *boost_val);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrilu02_numericBoost(lwsparseHandle_t handle,
                                                            bsrilu02Info_t info,
                                                            int enable_boost,
                                                            double *tol,
                                                            double *boost_val);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrilu02_numericBoost(lwsparseHandle_t handle,
                                                            bsrilu02Info_t info,
                                                            int enable_boost,
                                                            double *tol,
                                                            lwComplex *boost_val);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrilu02_numericBoost(lwsparseHandle_t handle,
                                                            bsrilu02Info_t info,
                                                            int enable_boost,
                                                            double *tol,
                                                            lwDoubleComplex *boost_val);

lwsparseStatus_t LWSPARSEAPI lwsparseXbsrilu02_zeroPivot(lwsparseHandle_t handle,
                                                         bsrilu02Info_t info,
                                                         int *position);

lwsparseStatus_t LWSPARSEAPI lwsparseSbsrilu02_bufferSize(lwsparseHandle_t handle,
                                                          lwsparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const lwsparseMatDescr_t descrA,
                                                          float *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockDim,
                                                          bsrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrilu02_bufferSize(lwsparseHandle_t handle,
                                                          lwsparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const lwsparseMatDescr_t descrA,
                                                          double *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockDim,
                                                          bsrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrilu02_bufferSize(lwsparseHandle_t handle,
                                                          lwsparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const lwsparseMatDescr_t descrA,
                                                          lwComplex *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockDim,
                                                          bsrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrilu02_bufferSize(lwsparseHandle_t handle,
                                                          lwsparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const lwsparseMatDescr_t descrA,
                                                          lwDoubleComplex *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockDim,
                                                          bsrilu02Info_t info,
                                                          int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseSbsrilu02_bufferSizeExt(lwsparseHandle_t handle,
                                                          lwsparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const lwsparseMatDescr_t descrA,
                                                          float *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockSize,
                                                          bsrilu02Info_t info,
                                                          size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrilu02_bufferSizeExt(lwsparseHandle_t handle,
                                                          lwsparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const lwsparseMatDescr_t descrA,
                                                          double *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockSize,
                                                          bsrilu02Info_t info,
                                                          size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrilu02_bufferSizeExt(lwsparseHandle_t handle,
                                                          lwsparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const lwsparseMatDescr_t descrA,
                                                          lwComplex *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockSize,
                                                          bsrilu02Info_t info,
                                                          size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrilu02_bufferSizeExt(lwsparseHandle_t handle,
                                                          lwsparseDirection_t dirA,
                                                          int mb,
                                                          int nnzb,
                                                          const lwsparseMatDescr_t descrA,
                                                          lwDoubleComplex *bsrSortedVal,
                                                          const int *bsrSortedRowPtr,
                                                          const int *bsrSortedColInd,
                                                          int blockSize,
                                                          bsrilu02Info_t info,
                                                          size_t *pBufferSize);


lwsparseStatus_t LWSPARSEAPI lwsparseSbsrilu02_analysis(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        int mb,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        float *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockDim,
                                                        bsrilu02Info_t info,
                                                        lwsparseSolvePolicy_t policy,
                                                        void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrilu02_analysis(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        int mb,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        double *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockDim,
                                                        bsrilu02Info_t info,
                                                        lwsparseSolvePolicy_t policy,
                                                        void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrilu02_analysis(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        int mb,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        lwComplex *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockDim,
                                                        bsrilu02Info_t info,
                                                        lwsparseSolvePolicy_t policy,
                                                        void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrilu02_analysis(lwsparseHandle_t handle,
                                                        lwsparseDirection_t dirA,
                                                        int mb,
                                                        int nnzb,
                                                        const lwsparseMatDescr_t descrA,
                                                        lwDoubleComplex *bsrSortedVal,
                                                        const int *bsrSortedRowPtr,
                                                        const int *bsrSortedColInd,
                                                        int blockDim,
                                                        bsrilu02Info_t info,
                                                        lwsparseSolvePolicy_t policy,
                                                        void *pBuffer);


lwsparseStatus_t LWSPARSEAPI lwsparseSbsrilu02(lwsparseHandle_t handle,
                                               lwsparseDirection_t dirA,
                                               int mb,
                                               int nnzb,
                                               const lwsparseMatDescr_t descrA,
                                               float *bsrSortedVal,
                                               const int *bsrSortedRowPtr,
                                               const int *bsrSortedColInd,
                                               int blockDim,
                                               bsrilu02Info_t info,
                                               lwsparseSolvePolicy_t policy,
                                               void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsrilu02(lwsparseHandle_t handle,
                                               lwsparseDirection_t dirA,
                                               int mb,
                                               int nnzb,
                                               const lwsparseMatDescr_t descrA,
                                               double *bsrSortedVal,
                                               const int *bsrSortedRowPtr,
                                               const int *bsrSortedColInd,
                                               int blockDim,
                                               bsrilu02Info_t info,
                                               lwsparseSolvePolicy_t policy,
                                               void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsrilu02(lwsparseHandle_t handle,
                                               lwsparseDirection_t dirA,
                                               int mb,
                                               int nnzb,
                                               const lwsparseMatDescr_t descrA,
                                               lwComplex *bsrSortedVal,
                                               const int *bsrSortedRowPtr,
                                               const int *bsrSortedColInd,
                                               int blockDim,
                                               bsrilu02Info_t info,
                                               lwsparseSolvePolicy_t policy,
                                               void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsrilu02(lwsparseHandle_t handle,
                                               lwsparseDirection_t dirA,
                                               int mb,
                                               int nnzb,
                                               const lwsparseMatDescr_t descrA,
                                               lwDoubleComplex *bsrSortedVal,
                                               const int *bsrSortedRowPtr,
                                               const int *bsrSortedColInd,
                                               int blockDim,
                                               bsrilu02Info_t info,
                                               lwsparseSolvePolicy_t policy,
                                               void *pBuffer);

/* Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
   of the matrix A stored in CSR format based on the information in the opaque 
   structure info that was obtained from the analysis phase (csrsv_analysis). 
   This routine implements algorithm 1 for this problem. */
lwsparseStatus_t LWSPARSEAPI lwsparseScsric0(lwsparseHandle_t handle, 
                                              lwsparseOperation_t trans, 
                                              int m, 
                                              const lwsparseMatDescr_t descrA,
                                              float *csrSortedValA_ValM,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */ 
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsric0(lwsparseHandle_t handle, 
                                              lwsparseOperation_t trans, 
                                              int m, 
                                              const lwsparseMatDescr_t descrA, 
                                              double *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA, 
                                              lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsric0(lwsparseHandle_t handle, 
                                              lwsparseOperation_t trans, 
                                              int m, 
                                              const lwsparseMatDescr_t descrA, 
                                              lwComplex *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA, 
                                              lwsparseSolveAnalysisInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsric0(lwsparseHandle_t handle, 
                                              lwsparseOperation_t trans, 
                                              int m, 
                                              const lwsparseMatDescr_t descrA, 
                                              lwDoubleComplex *csrSortedValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA, 
                                              lwsparseSolveAnalysisInfo_t info);

/* Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
   of the matrix A stored in CSR format based on the information in the opaque 
   structure info that was obtained from the analysis phase (csrsv2_analysis). 
   This routine implements algorithm 2 for this problem. */
lwsparseStatus_t LWSPARSEAPI lwsparseXcsric02_zeroPivot(lwsparseHandle_t handle,
                                                        csric02Info_t info,
                                                        int *position);

lwsparseStatus_t LWSPARSEAPI lwsparseScsric02_bufferSize(lwsparseHandle_t handle,
                                                         int m,
                                                         int nnz,
                                                         const lwsparseMatDescr_t descrA,
                                                         float *csrSortedValA,
                                                         const int *csrSortedRowPtrA,
                                                         const int *csrSortedColIndA,
                                                         csric02Info_t info,
                                                         int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsric02_bufferSize(lwsparseHandle_t handle,
                                                         int m,
                                                         int nnz,
                                                         const lwsparseMatDescr_t descrA,
                                                         double *csrSortedValA,
                                                         const int *csrSortedRowPtrA,
                                                         const int *csrSortedColIndA,
                                                         csric02Info_t info,
                                                         int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsric02_bufferSize(lwsparseHandle_t handle,
                                                         int m,
                                                         int nnz,
                                                         const lwsparseMatDescr_t descrA,
                                                         lwComplex *csrSortedValA,
                                                         const int *csrSortedRowPtrA,
                                                         const int *csrSortedColIndA,
                                                         csric02Info_t info,
                                                         int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsric02_bufferSize(lwsparseHandle_t handle,
                                                         int m,
                                                         int nnz,
                                                         const lwsparseMatDescr_t descrA,
                                                         lwDoubleComplex *csrSortedValA,
                                                         const int *csrSortedRowPtrA,
                                                         const int *csrSortedColIndA,
                                                         csric02Info_t info,
                                                         int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseScsric02_bufferSizeExt(lwsparseHandle_t handle,
                                                            int m,
                                                            int nnz,
                                                            const lwsparseMatDescr_t descrA,
                                                            float *csrSortedVal,
                                                            const int *csrSortedRowPtr,
                                                            const int *csrSortedColInd,
                                                            csric02Info_t info,
                                                            size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsric02_bufferSizeExt(lwsparseHandle_t handle,
                                                            int m,
                                                            int nnz,
                                                            const lwsparseMatDescr_t descrA,
                                                            double *csrSortedVal,
                                                            const int *csrSortedRowPtr,
                                                            const int *csrSortedColInd,
                                                            csric02Info_t info,
                                                            size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsric02_bufferSizeExt(lwsparseHandle_t handle,
                                                            int m,
                                                            int nnz,
                                                            const lwsparseMatDescr_t descrA,
                                                            lwComplex *csrSortedVal,
                                                            const int *csrSortedRowPtr,
                                                            const int *csrSortedColInd,
                                                            csric02Info_t info,
                                                            size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsric02_bufferSizeExt(lwsparseHandle_t handle,
                                                            int m,
                                                            int nnz,
                                                            const lwsparseMatDescr_t descrA,
                                                            lwDoubleComplex *csrSortedVal,
                                                            const int *csrSortedRowPtr,
                                                            const int *csrSortedColInd,
                                                            csric02Info_t info,
                                                            size_t *pBufferSize);


lwsparseStatus_t LWSPARSEAPI lwsparseScsric02_analysis(lwsparseHandle_t handle,
                                                       int m,
                                                       int nnz,
                                                       const lwsparseMatDescr_t descrA,
                                                       const float *csrSortedValA,
                                                       const int *csrSortedRowPtrA,
                                                       const int *csrSortedColIndA,
                                                       csric02Info_t info,
                                                       lwsparseSolvePolicy_t policy,
                                                       void *pBuffer);


lwsparseStatus_t LWSPARSEAPI lwsparseDcsric02_analysis(lwsparseHandle_t handle,
                                                       int m,
                                                       int nnz,
                                                       const lwsparseMatDescr_t descrA,
                                                       const double *csrSortedValA,
                                                       const int *csrSortedRowPtrA,
                                                       const int *csrSortedColIndA,
                                                       csric02Info_t info,
                                                       lwsparseSolvePolicy_t policy,
                                                       void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsric02_analysis(lwsparseHandle_t handle,
                                                       int m,
                                                       int nnz,
                                                       const lwsparseMatDescr_t descrA,
                                                       const lwComplex *csrSortedValA,
                                                       const int *csrSortedRowPtrA,
                                                       const int *csrSortedColIndA, 
                                                       csric02Info_t info,
                                                       lwsparseSolvePolicy_t policy,
                                                       void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsric02_analysis(lwsparseHandle_t handle,
                                                       int m,
                                                       int nnz,
                                                       const lwsparseMatDescr_t descrA,
                                                       const lwDoubleComplex *csrSortedValA,
                                                       const int *csrSortedRowPtrA,
                                                       const int *csrSortedColIndA,
                                                       csric02Info_t info,
                                                       lwsparseSolvePolicy_t policy,
                                                       void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseScsric02(lwsparseHandle_t handle,
                                              int m,
                                              int nnz,
                                              const lwsparseMatDescr_t descrA,
                                              float *csrSortedValA_valM,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              csric02Info_t info,
                                              lwsparseSolvePolicy_t policy,
                                              void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsric02(lwsparseHandle_t handle,
                                              int m,
                                              int nnz,
                                              const lwsparseMatDescr_t descrA,
                                              double *csrSortedValA_valM,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              csric02Info_t info,
                                              lwsparseSolvePolicy_t policy,
                                              void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsric02(lwsparseHandle_t handle,
                                              int m,
                                              int nnz,
                                              const lwsparseMatDescr_t descrA,
                                              lwComplex *csrSortedValA_valM,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              csric02Info_t info,
                                              lwsparseSolvePolicy_t policy,
                                              void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsric02(lwsparseHandle_t handle,
                                              int m,
                                              int nnz,
                                              const lwsparseMatDescr_t descrA,
                                              lwDoubleComplex *csrSortedValA_valM,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              csric02Info_t info,
                                              lwsparseSolvePolicy_t policy,
                                              void *pBuffer);

/* Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
   of the matrix A stored in block-CSR format based on the information in the opaque 
   structure info that was obtained from the analysis phase (bsrsv2_analysis). 
   This routine implements algorithm 1 for this problem. */
lwsparseStatus_t LWSPARSEAPI lwsparseXbsric02_zeroPivot(lwsparseHandle_t handle,
                                                        bsric02Info_t info,
                                                        int *position);

lwsparseStatus_t LWSPARSEAPI lwsparseSbsric02_bufferSize(lwsparseHandle_t handle,
                                                         lwsparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const lwsparseMatDescr_t descrA,
                                                         float *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockDim,
                                                         bsric02Info_t info,
                                                         int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsric02_bufferSize(lwsparseHandle_t handle,
                                                         lwsparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const lwsparseMatDescr_t descrA,
                                                         double *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockDim,
                                                         bsric02Info_t info,
                                                         int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsric02_bufferSize(lwsparseHandle_t handle,
                                                         lwsparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const lwsparseMatDescr_t descrA,
                                                         lwComplex *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockDim,
                                                         bsric02Info_t info,
                                                         int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsric02_bufferSize(lwsparseHandle_t handle,
                                                         lwsparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const lwsparseMatDescr_t descrA,
                                                         lwDoubleComplex *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockDim,
                                                         bsric02Info_t info,
                                                         int *pBufferSizeInBytes); 

lwsparseStatus_t LWSPARSEAPI lwsparseSbsric02_bufferSizeExt(lwsparseHandle_t handle,
                                                         lwsparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const lwsparseMatDescr_t descrA,
                                                         float *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockSize,
                                                         bsric02Info_t info,
                                                         size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsric02_bufferSizeExt(lwsparseHandle_t handle,
                                                         lwsparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const lwsparseMatDescr_t descrA,
                                                         double *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockSize,
                                                         bsric02Info_t info,
                                                         size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsric02_bufferSizeExt(lwsparseHandle_t handle,
                                                         lwsparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const lwsparseMatDescr_t descrA,
                                                         lwComplex *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockSize,
                                                         bsric02Info_t info,
                                                         size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsric02_bufferSizeExt(lwsparseHandle_t handle,
                                                         lwsparseDirection_t dirA,
                                                         int mb,
                                                         int nnzb,
                                                         const lwsparseMatDescr_t descrA,
                                                         lwDoubleComplex *bsrSortedVal,
                                                         const int *bsrSortedRowPtr,
                                                         const int *bsrSortedColInd,
                                                         int blockSize,
                                                         bsric02Info_t info,
                                                         size_t *pBufferSize);



lwsparseStatus_t LWSPARSEAPI lwsparseSbsric02_analysis(lwsparseHandle_t handle,
                                                       lwsparseDirection_t dirA,
                                                       int mb,
                                                       int nnzb,
                                                       const lwsparseMatDescr_t descrA,
                                                       const float *bsrSortedVal,
                                                       const int *bsrSortedRowPtr,
                                                       const int *bsrSortedColInd,
                                                       int blockDim,
                                                       bsric02Info_t info,
                                                       lwsparseSolvePolicy_t policy,
                                                       void *pInputBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsric02_analysis(lwsparseHandle_t handle,
                                                       lwsparseDirection_t dirA,
                                                       int mb,
                                                       int nnzb,
                                                       const lwsparseMatDescr_t descrA,
                                                       const double *bsrSortedVal,
                                                       const int *bsrSortedRowPtr,
                                                       const int *bsrSortedColInd,
                                                       int blockDim,
                                                       bsric02Info_t info,
                                                       lwsparseSolvePolicy_t policy,
                                                       void *pInputBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsric02_analysis(lwsparseHandle_t handle,
                                                       lwsparseDirection_t dirA,
                                                       int mb,
                                                       int nnzb,
                                                       const lwsparseMatDescr_t descrA,
                                                       const lwComplex *bsrSortedVal,
                                                       const int *bsrSortedRowPtr,
                                                       const int *bsrSortedColInd,
                                                       int blockDim,
                                                       bsric02Info_t info,
                                                       lwsparseSolvePolicy_t policy,
                                                       void *pInputBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsric02_analysis(lwsparseHandle_t handle,
                                                       lwsparseDirection_t dirA,
                                                       int mb,
                                                       int nnzb,
                                                       const lwsparseMatDescr_t descrA,
                                                       const lwDoubleComplex *bsrSortedVal,
                                                       const int *bsrSortedRowPtr,
                                                       const int *bsrSortedColInd,
                                                       int blockDim,
                                                       bsric02Info_t info,
                                                       lwsparseSolvePolicy_t policy,
                                                       void *pInputBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseSbsric02(lwsparseHandle_t handle,
                                              lwsparseDirection_t dirA,
                                              int mb,
                                              int nnzb,
                                              const lwsparseMatDescr_t descrA,
                                              float *bsrSortedVal,
                                              const int *bsrSortedRowPtr,
                                              const int *bsrSortedColInd,
                                              int blockDim,
                                              bsric02Info_t info,
                                              lwsparseSolvePolicy_t policy,
                                              void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsric02(lwsparseHandle_t handle,
                                              lwsparseDirection_t dirA,
                                              int mb,
                                              int nnzb,
                                              const lwsparseMatDescr_t descrA,
                                              double *bsrSortedVal,
                                              const int *bsrSortedRowPtr,
                                              const int *bsrSortedColInd,
                                              int blockDim,
                                              bsric02Info_t info,
                                              lwsparseSolvePolicy_t policy,
                                              void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsric02(lwsparseHandle_t handle,
                                              lwsparseDirection_t dirA,
                                              int mb,
                                              int nnzb,
                                              const lwsparseMatDescr_t descrA,
                                              lwComplex *bsrSortedVal,
                                              const int *bsrSortedRowPtr,
                                              const int *bsrSortedColInd,
                                              int blockDim,
                                              bsric02Info_t info,
                                              lwsparseSolvePolicy_t policy,
                                              void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsric02(lwsparseHandle_t handle,
                                              lwsparseDirection_t dirA,
                                              int mb,
                                              int nnzb,
                                              const lwsparseMatDescr_t descrA,
                                              lwDoubleComplex *bsrSortedVal,
                                              const int *bsrSortedRowPtr,
                                              const int *bsrSortedColInd,
                                              int blockDim,
                                              bsric02Info_t info,
                                              lwsparseSolvePolicy_t policy,
                                              void *pBuffer);


/* Description: Solution of tridiagonal linear system A * X = B, 
   with multiple right-hand-sides. The coefficient matrix A is 
   composed of lower (dl), main (d) and upper (du) diagonals, and 
   the right-hand-sides B are overwritten with the solution X. 
   These routine use pivoting. */
lwsparseStatus_t LWSPARSEAPI lwsparseSgtsv(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    float *B,
    int ldb);
 
lwsparseStatus_t LWSPARSEAPI lwsparseDgtsv(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    double *B,
    int ldb);

lwsparseStatus_t LWSPARSEAPI lwsparseCgtsv(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwComplex *dl,
    const lwComplex  *d,
    const lwComplex *du,
    lwComplex *B,
    int ldb);

lwsparseStatus_t LWSPARSEAPI lwsparseZgtsv(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwDoubleComplex *dl,
    const lwDoubleComplex  *d,
    const lwDoubleComplex *du,
    lwDoubleComplex *B,
    int ldb);


lwsparseStatus_t LWSPARSEAPI lwsparseSgtsv2_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    const float *B,
    int ldb,
    size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDgtsv2_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    const double *B,
    int ldb,
    size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCgtsv2_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwComplex *dl,
    const lwComplex  *d,
    const lwComplex *du,
    const lwComplex *B,
    int ldb,
    size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZgtsv2_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwDoubleComplex *dl,
    const lwDoubleComplex  *d,
    const lwDoubleComplex *du,
    const lwDoubleComplex *B,
    int ldb,
    size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseSgtsv2(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    float *B,
    int ldb,
    void* pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDgtsv2(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    double *B,
    int ldb,
    void* pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCgtsv2(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwComplex *dl,
    const lwComplex  *d,
    const lwComplex *du,
    lwComplex *B,
    int ldb,
    void* pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZgtsv2(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwDoubleComplex *dl,
    const lwDoubleComplex  *d,
    const lwDoubleComplex *du,
    lwDoubleComplex *B,
    int ldb,
    void* pBuffer);


/* Description: Solution of tridiagonal linear system A * X = B, 
   with multiple right-hand-sides. The coefficient matrix A is 
   composed of lower (dl), main (d) and upper (du) diagonals, and 
   the right-hand-sides B are overwritten with the solution X. 
   These routine does not use pivoting. */                               
lwsparseStatus_t LWSPARSEAPI lwsparseSgtsv_nopivot(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    float *B,
    int ldb);
                                 
lwsparseStatus_t LWSPARSEAPI lwsparseDgtsv_nopivot(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    double *B,
    int ldb);
                                                                 
lwsparseStatus_t LWSPARSEAPI lwsparseCgtsv_nopivot(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwComplex *dl,
    const lwComplex  *d,
    const lwComplex *du,
    lwComplex *B,
    int ldb);

lwsparseStatus_t LWSPARSEAPI lwsparseZgtsv_nopivot(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwDoubleComplex *dl,  
    const lwDoubleComplex  *d,  
    const lwDoubleComplex *du,
    lwDoubleComplex *B,     
    int ldb);                               
                                  

lwsparseStatus_t LWSPARSEAPI lwsparseSgtsv2_nopivot_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    const float *B,
    int ldb,
    size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDgtsv2_nopivot_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    const double *B,
    int ldb,
    size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCgtsv2_nopivot_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwComplex *dl,
    const lwComplex  *d,
    const lwComplex *du,
    const lwComplex *B,
    int ldb,
    size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZgtsv2_nopivot_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwDoubleComplex *dl,
    const lwDoubleComplex  *d,
    const lwDoubleComplex *du,
    const lwDoubleComplex *B,
    int ldb,
    size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseSgtsv2_nopivot(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    float *B,
    int ldb,
    void* pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDgtsv2_nopivot(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    double *B,
    int ldb,
    void* pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCgtsv2_nopivot(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwComplex *dl,
    const lwComplex  *d,
    const lwComplex *du,
    lwComplex *B,
    int ldb,
    void* pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZgtsv2_nopivot(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwDoubleComplex *dl,
    const lwDoubleComplex  *d,
    const lwDoubleComplex *du,
    lwDoubleComplex *B,
    int ldb,
    void* pBuffer);

/* Description: Solution of a set of tridiagonal linear systems 
   A_{i} * x_{i} = f_{i} for i=1,...,batchCount. The coefficient 
   matrices A_{i} are composed of lower (dl), main (d) and upper (du) 
   diagonals and stored separated by a batchStride. Also, the 
   right-hand-sides/solutions f_{i}/x_{i} are separated by a batchStride. */
lwsparseStatus_t LWSPARSEAPI lwsparseSgtsvStridedBatch(
    lwsparseHandle_t handle,
    int m, 
    const float *dl,
    const float  *d,
    const float *du,
    float *x,
    int batchCount,
    int batchStride);

lwsparseStatus_t LWSPARSEAPI lwsparseDgtsvStridedBatch(
    lwsparseHandle_t handle,
    int m, 
    const double *dl,
    const double  *d,
    const double *du,
    double *x,
    int batchCount,
    int batchStride);

lwsparseStatus_t LWSPARSEAPI lwsparseCgtsvStridedBatch(
    lwsparseHandle_t handle,
    int m, 
    const lwComplex *dl,
    const lwComplex  *d,
    const lwComplex *du,
    lwComplex *x,
    int batchCount,
    int batchStride);

lwsparseStatus_t LWSPARSEAPI lwsparseZgtsvStridedBatch(
    lwsparseHandle_t handle,
    int m, 
    const lwDoubleComplex *dl,
    const lwDoubleComplex  *d,
    const lwDoubleComplex *du,
    lwDoubleComplex *x,
    int batchCount,
    int batchStride);


lwsparseStatus_t LWSPARSEAPI lwsparseSgtsv2StridedBatch_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    const float *dl,
    const float  *d,
    const float *du,
    const float *x,
    int batchCount,
    int batchStride,
    size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDgtsv2StridedBatch_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    const double *dl,
    const double  *d,
    const double *du,
    const double *x,
    int batchCount,
    int batchStride,
    size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCgtsv2StridedBatch_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    const lwComplex *dl,
    const lwComplex  *d,
    const lwComplex *du,
    const lwComplex *x,
    int batchCount,
    int batchStride,
    size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZgtsv2StridedBatch_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    const lwDoubleComplex *dl,
    const lwDoubleComplex  *d,
    const lwDoubleComplex *du,
    const lwDoubleComplex *x,
    int batchCount,
    int batchStride,
    size_t *bufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseSgtsv2StridedBatch(
    lwsparseHandle_t handle,
    int m,
    const float *dl,
    const float  *d,
    const float *du,
    float *x,
    int batchCount,
    int batchStride,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDgtsv2StridedBatch(
    lwsparseHandle_t handle,
    int m,
    const double *dl,
    const double  *d,
    const double *du,
    double *x,
    int batchCount,
    int batchStride,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCgtsv2StridedBatch(
    lwsparseHandle_t handle,
    int m,
    const lwComplex *dl,
    const lwComplex  *d,
    const lwComplex *du,
    lwComplex *x,
    int batchCount,
    int batchStride,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZgtsv2StridedBatch(
    lwsparseHandle_t handle,
    int m,
    const lwDoubleComplex *dl,
    const lwDoubleComplex  *d,
    const lwDoubleComplex *du,
    lwDoubleComplex *x,
    int batchCount,
    int batchStride,
    void *pBuffer);


lwsparseStatus_t LWSPARSEAPI lwsparseSgtsvInterleavedBatch_bufferSizeExt(
    lwsparseHandle_t handle,
    int algo,
    int m,
    const float *dl,
    const float  *d,
    const float *du,
    const float *x,
    int batchCount,
    size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDgtsvInterleavedBatch_bufferSizeExt(
    lwsparseHandle_t handle,
    int algo,
    int m,
    const double *dl,
    const double  *d,
    const double *du,
    const double *x,
    int batchCount,
    size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCgtsvInterleavedBatch_bufferSizeExt(
    lwsparseHandle_t handle,
    int algo,
    int m,
    const lwComplex *dl,
    const lwComplex  *d,
    const lwComplex *du,
    const lwComplex *x,
    int batchCount,
    size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZgtsvInterleavedBatch_bufferSizeExt(
    lwsparseHandle_t handle,
    int algo,
    int m,
    const lwDoubleComplex *dl,
    const lwDoubleComplex  *d,
    const lwDoubleComplex *du,
    const lwDoubleComplex *x,
    int batchCount,
    size_t *pBufferSizeInBytes);


lwsparseStatus_t LWSPARSEAPI lwsparseSgtsvInterleavedBatch(
    lwsparseHandle_t handle,
    int algo,
    int m,
    float *dl,
    float  *d,
    float *du,
    float *x,
    int batchCount,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDgtsvInterleavedBatch(
    lwsparseHandle_t handle,
    int algo,
    int m,
    double *dl,
    double  *d,
    double *du,
    double *x,
    int batchCount,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCgtsvInterleavedBatch(
    lwsparseHandle_t handle,
    int algo,
    int m,
    lwComplex *dl,
    lwComplex  *d,
    lwComplex *du,
    lwComplex *x,
    int batchCount,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZgtsvInterleavedBatch(
    lwsparseHandle_t handle,
    int algo,
    int m,
    lwDoubleComplex *dl,
    lwDoubleComplex  *d,
    lwDoubleComplex *du,
    lwDoubleComplex *x,
    int batchCount,
    void *pBuffer);


/* Description: Solution of pentadiagonal linear system A * X = B, 
   with multiple right-hand-sides. The coefficient matrix A is 
   composed of lower (ds, dl), main (d) and upper (du, dw) diagonals, and 
   the right-hand-sides B are overwritten with the solution X. 
 */
lwsparseStatus_t  LWSPARSEAPI lwsparseSgpsvInterleavedBatch_bufferSizeExt(
    lwsparseHandle_t handle, 
    int algo,
    int m,
    const float *ds,
    const float *dl,
    const float  *d,
    const float *du,
    const float *dw,
    const float *x,
    int batchCount,
    size_t *pBufferSizeInBytes);

lwsparseStatus_t  LWSPARSEAPI lwsparseDgpsvInterleavedBatch_bufferSizeExt(
    lwsparseHandle_t handle,
    int algo,
    int m,
    const double *ds,
    const double *dl,
    const double  *d,
    const double *du,
    const double *dw,
    const double *x,
    int batchCount,
    size_t *pBufferSizeInBytes);

lwsparseStatus_t  LWSPARSEAPI lwsparseCgpsvInterleavedBatch_bufferSizeExt(
    lwsparseHandle_t handle, 
    int algo,
    int m,
    const lwComplex *ds,
    const lwComplex *dl,
    const lwComplex  *d,
    const lwComplex *du,
    const lwComplex *dw,
    const lwComplex *x,
    int batchCount,
    size_t *pBufferSizeInBytes);

lwsparseStatus_t  LWSPARSEAPI lwsparseZgpsvInterleavedBatch_bufferSizeExt(
    lwsparseHandle_t handle,
    int algo,
    int m,
    const lwDoubleComplex *ds,
    const lwDoubleComplex *dl,
    const lwDoubleComplex  *d,
    const lwDoubleComplex *du,
    const lwDoubleComplex *dw,
    const lwDoubleComplex *x,
    int batchCount,
    size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseSgpsvInterleavedBatch(
    lwsparseHandle_t handle,
    int algo,
    int m,
    float *ds,
    float *dl,
    float  *d,
    float *du,
    float *dw,
    float *x,
    int batchCount,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDgpsvInterleavedBatch(
    lwsparseHandle_t handle,
    int algo,
    int m,
    double *ds,
    double *dl,
    double  *d,
    double *du,
    double *dw,
    double *x,
    int batchCount,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCgpsvInterleavedBatch(
    lwsparseHandle_t handle,
    int algo,
    int m,
    lwComplex *ds,
    lwComplex *dl,
    lwComplex  *d,
    lwComplex *du,
    lwComplex *dw,
    lwComplex *x,
    int batchCount,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZgpsvInterleavedBatch(
    lwsparseHandle_t handle,
    int algo,
    int m,
    lwDoubleComplex *ds,
    lwDoubleComplex *dl,
    lwDoubleComplex  *d,
    lwDoubleComplex *du,
    lwDoubleComplex *dw,
    lwDoubleComplex *x,
    int batchCount,
    void *pBuffer);




/* --- Sparse Level 4 routines --- */

/* Description: Compute sparse - sparse matrix multiplication for matrices 
   stored in CSR format. */
lwsparseStatus_t LWSPARSEAPI lwsparseXcsrgemmNnz(lwsparseHandle_t handle,
                                                 lwsparseOperation_t transA, 
                                                 lwsparseOperation_t transB, 
                                                 int m, 
                                                 int n, 
                                                 int k, 
                                                 const lwsparseMatDescr_t descrA,
                                                 const int nnzA,
                                                 const int *csrSortedRowPtrA, 
                                                 const int *csrSortedColIndA,     
                                                 const lwsparseMatDescr_t descrB,
                                                 const int nnzB,
                                                 const int *csrSortedRowPtrB, 
                                                 const int *csrSortedColIndB,  
                                                 const lwsparseMatDescr_t descrC,                                                
                                                 int *csrSortedRowPtrC, 
                                                 int *nnzTotalDevHostPtr);                                              
                                              
lwsparseStatus_t LWSPARSEAPI lwsparseScsrgemm(lwsparseHandle_t handle,
                                              lwsparseOperation_t transA, 
                                              lwsparseOperation_t transB, 
                                              int m, 
                                              int n, 
                                              int k, 
                                              const lwsparseMatDescr_t descrA,
                                              const int nnzA,      
                                              const float *csrSortedValA, 
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              const lwsparseMatDescr_t descrB,
                                              const int nnzB,                                                    
                                              const float *csrSortedValB, 
                                              const int *csrSortedRowPtrB, 
                                              const int *csrSortedColIndB,
                                              const lwsparseMatDescr_t descrC, 
                                              float *csrSortedValC, 
                                              const int *csrSortedRowPtrC, 
                                              int *csrSortedColIndC);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrgemm(lwsparseHandle_t handle,
                                              lwsparseOperation_t transA, 
                                              lwsparseOperation_t transB, 
                                              int m, 
                                              int n, 
                                              int k, 
                                              const lwsparseMatDescr_t descrA,
                                              int nnzA,      
                                              const double *csrSortedValA, 
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              const lwsparseMatDescr_t descrB,
                                              int nnzB,                                                    
                                              const double *csrSortedValB, 
                                              const int *csrSortedRowPtrB, 
                                              const int *csrSortedColIndB,
                                              const lwsparseMatDescr_t descrC, 
                                              double *csrSortedValC, 
                                              const int *csrSortedRowPtrC, 
                                              int *csrSortedColIndC);
                                              
lwsparseStatus_t LWSPARSEAPI lwsparseCcsrgemm(lwsparseHandle_t handle,
                                              lwsparseOperation_t transA, 
                                              lwsparseOperation_t transB, 
                                              int m, 
                                              int n, 
                                              int k, 
                                              const lwsparseMatDescr_t descrA,
                                              int nnzA,      
                                              const lwComplex *csrSortedValA, 
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              const lwsparseMatDescr_t descrB,
                                              int nnzB,                                                    
                                              const lwComplex *csrSortedValB, 
                                              const int *csrSortedRowPtrB, 
                                              const int *csrSortedColIndB,
                                              const lwsparseMatDescr_t descrC, 
                                              lwComplex *csrSortedValC, 
                                              const int *csrSortedRowPtrC, 
                                              int *csrSortedColIndC); 
                                              
lwsparseStatus_t LWSPARSEAPI lwsparseZcsrgemm(lwsparseHandle_t handle,
                                              lwsparseOperation_t transA, 
                                              lwsparseOperation_t transB, 
                                              int m, 
                                              int n, 
                                              int k, 
                                              const lwsparseMatDescr_t descrA,
                                              int nnzA,      
                                              const lwDoubleComplex *csrSortedValA, 
                                              const int *csrSortedRowPtrA, 
                                              const int *csrSortedColIndA,
                                              const lwsparseMatDescr_t descrB,
                                              int nnzB,                                                    
                                              const lwDoubleComplex *csrSortedValB, 
                                              const int *csrSortedRowPtrB, 
                                              const int *csrSortedColIndB,
                                              const lwsparseMatDescr_t descrC, 
                                              lwDoubleComplex *csrSortedValC, 
                                              const int *csrSortedRowPtrC, 
                                              int *csrSortedColIndC);

/* Description: Compute sparse - sparse matrix multiplication for matrices 
   stored in CSR format. */

lwsparseStatus_t LWSPARSEAPI lwsparseCreateCsrgemm2Info(csrgemm2Info_t *info);

lwsparseStatus_t LWSPARSEAPI lwsparseDestroyCsrgemm2Info(csrgemm2Info_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrgemm2_bufferSizeExt(lwsparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int k,
                                                             const float *alpha,
                                                             const lwsparseMatDescr_t descrA,
                                                             int nnzA,
                                                             const int *csrSortedRowPtrA,
                                                             const int *csrSortedColIndA,
                                                             const lwsparseMatDescr_t descrB,
                                                             int nnzB,
                                                             const int *csrSortedRowPtrB,
                                                             const int *csrSortedColIndB,
                                                             const float *beta,
                                                             const lwsparseMatDescr_t descrD,
                                                             int nnzD,
                                                             const int *csrSortedRowPtrD,
                                                             const int *csrSortedColIndD,
                                                             csrgemm2Info_t info,
                                                             size_t *pBufferSizeInBytes );

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrgemm2_bufferSizeExt(lwsparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int k,
                                                             const double *alpha,
                                                             const lwsparseMatDescr_t descrA,
                                                             int nnzA,
                                                             const int *csrSortedRowPtrA,
                                                             const int *csrSortedColIndA,
                                                             const lwsparseMatDescr_t descrB,
                                                             int nnzB,
                                                             const int *csrSortedRowPtrB,
                                                             const int *csrSortedColIndB,
                                                             const double *beta,
                                                             const lwsparseMatDescr_t descrD,
                                                             int nnzD,
                                                             const int *csrSortedRowPtrD,
                                                             const int *csrSortedColIndD,
                                                             csrgemm2Info_t info,
                                                             size_t *pBufferSizeInBytes );

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrgemm2_bufferSizeExt(lwsparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int k,
                                                             const lwComplex *alpha,
                                                             const lwsparseMatDescr_t descrA,
                                                             int nnzA,
                                                             const int *csrSortedRowPtrA,
                                                             const int *csrSortedColIndA,
                                                             const lwsparseMatDescr_t descrB,
                                                             int nnzB,
                                                             const int *csrSortedRowPtrB,
                                                             const int *csrSortedColIndB,
                                                             const lwComplex *beta,
                                                             const lwsparseMatDescr_t descrD,
                                                             int nnzD,
                                                             const int *csrSortedRowPtrD,
                                                             const int *csrSortedColIndD,
                                                             csrgemm2Info_t info,
                                                             size_t *pBufferSizeInBytes );

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrgemm2_bufferSizeExt(lwsparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int k,
                                                             const lwDoubleComplex *alpha,
                                                             const lwsparseMatDescr_t descrA,
                                                             int nnzA,
                                                             const int *csrSortedRowPtrA,
                                                             const int *csrSortedColIndA,
                                                             const lwsparseMatDescr_t descrB,
                                                             int nnzB,
                                                             const int *csrSortedRowPtrB,
                                                             const int *csrSortedColIndB,
                                                             const lwDoubleComplex *beta,
                                                             const lwsparseMatDescr_t descrD,
                                                             int nnzD,
                                                             const int *csrSortedRowPtrD,
                                                             const int *csrSortedColIndD,
                                                             csrgemm2Info_t info,
                                                             size_t *pBufferSizeInBytes );


lwsparseStatus_t LWSPARSEAPI lwsparseXcsrgemm2Nnz(lwsparseHandle_t handle,
                                                  int m,
                                                  int n, 
                                                  int k,
                                                  const lwsparseMatDescr_t descrA,
                                                  int nnzA,
                                                  const int *csrSortedRowPtrA,
                                                  const int *csrSortedColIndA,
                                                  const lwsparseMatDescr_t descrB,
                                                  int nnzB,
                                                  const int *csrSortedRowPtrB,
                                                  const int *csrSortedColIndB,
                                                  const lwsparseMatDescr_t descrD,
                                                  int nnzD,
                                                  const int *csrSortedRowPtrD,
                                                  const int *csrSortedColIndD,
                                                  const lwsparseMatDescr_t descrC,
                                                  int *csrSortedRowPtrC,
                                                  int *nnzTotalDevHostPtr,
                                                  const csrgemm2Info_t info,
                                                  void *pBuffer );


lwsparseStatus_t LWSPARSEAPI lwsparseScsrgemm2(lwsparseHandle_t handle,
                                               int m,
                                               int n,
                                               int k,
                                               const float *alpha,
                                               const lwsparseMatDescr_t descrA,
                                               int nnzA,
                                               const float *csrSortedValA,
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               const lwsparseMatDescr_t descrB,
                                               int nnzB,
                                               const float *csrSortedValB,
                                               const int *csrSortedRowPtrB,
                                               const int *csrSortedColIndB,
                                               const float *beta,
                                               const lwsparseMatDescr_t descrD,
                                               int nnzD,
                                               const float *csrSortedValD,
                                               const int *csrSortedRowPtrD,
                                               const int *csrSortedColIndD,
                                               const lwsparseMatDescr_t descrC,
                                               float *csrSortedValC,
                                               const int *csrSortedRowPtrC,
                                               int *csrSortedColIndC,
                                               const csrgemm2Info_t info,
                                               void *pBuffer );

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrgemm2(lwsparseHandle_t handle,
                                               int m,
                                               int n,
                                               int k,
                                               const double *alpha,
                                               const lwsparseMatDescr_t descrA,
                                               int nnzA,
                                               const double *csrSortedValA,
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               const lwsparseMatDescr_t descrB,
                                               int nnzB,
                                               const double *csrSortedValB,
                                               const int *csrSortedRowPtrB,
                                               const int *csrSortedColIndB,
                                               const double *beta,
                                               const lwsparseMatDescr_t descrD,
                                               int nnzD,
                                               const double *csrSortedValD,
                                               const int *csrSortedRowPtrD,
                                               const int *csrSortedColIndD,
                                               const lwsparseMatDescr_t descrC,
                                               double *csrSortedValC,
                                               const int *csrSortedRowPtrC,
                                               int *csrSortedColIndC,
                                               const csrgemm2Info_t info,
                                               void *pBuffer );


lwsparseStatus_t LWSPARSEAPI lwsparseCcsrgemm2(lwsparseHandle_t handle,
                                               int m,
                                               int n,
                                               int k,
                                               const lwComplex *alpha,
                                               const lwsparseMatDescr_t descrA,
                                               int nnzA,
                                               const lwComplex *csrSortedValA,
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               const lwsparseMatDescr_t descrB,
                                               int nnzB,
                                               const lwComplex *csrSortedValB,
                                               const int *csrSortedRowPtrB,
                                               const int *csrSortedColIndB,
                                               const lwComplex *beta,
                                               const lwsparseMatDescr_t descrD,
                                               int nnzD,
                                               const lwComplex *csrSortedValD,
                                               const int *csrSortedRowPtrD,
                                               const int *csrSortedColIndD,
                                               const lwsparseMatDescr_t descrC,
                                               lwComplex *csrSortedValC,
                                               const int *csrSortedRowPtrC,
                                               int *csrSortedColIndC,
                                               const csrgemm2Info_t info,
                                               void *pBuffer );


lwsparseStatus_t LWSPARSEAPI lwsparseZcsrgemm2(lwsparseHandle_t handle,
                                               int m,
                                               int n,
                                               int k,
                                               const lwDoubleComplex *alpha,
                                               const lwsparseMatDescr_t descrA,
                                               int nnzA,
                                               const lwDoubleComplex *csrSortedValA,
                                               const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               const lwsparseMatDescr_t descrB,
                                               int nnzB,
                                               const lwDoubleComplex *csrSortedValB,
                                               const int *csrSortedRowPtrB,
                                               const int *csrSortedColIndB,
                                               const lwDoubleComplex *beta,
                                               const lwsparseMatDescr_t descrD,
                                               int nnzD,
                                               const lwDoubleComplex *csrSortedValD,
                                               const int *csrSortedRowPtrD,
                                               const int *csrSortedColIndD,
                                               const lwsparseMatDescr_t descrC,
                                               lwDoubleComplex *csrSortedValC,
                                               const int *csrSortedRowPtrC,
                                               int *csrSortedColIndC,
                                               const csrgemm2Info_t info,
                                               void *pBuffer );


/* Description: Compute sparse - sparse matrix addition of matrices 
   stored in CSR format */
lwsparseStatus_t LWSPARSEAPI lwsparseXcsrgeamNnz(lwsparseHandle_t handle,
                                                 int m,
                                                 int n,
                                                 const lwsparseMatDescr_t descrA,
                                                 int nnzA,
                                                 const int *csrSortedRowPtrA,
                                                 const int *csrSortedColIndA,
                                                 const lwsparseMatDescr_t descrB,
                                                 int nnzB,
                                                 const int *csrSortedRowPtrB,
                                                 const int *csrSortedColIndB,
                                                 const lwsparseMatDescr_t descrC,
                                                 int *csrSortedRowPtrC,
                                                 int *nnzTotalDevHostPtr);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrgeam(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              const float *alpha,
                                              const lwsparseMatDescr_t descrA,
                                              int nnzA,
                                              const float *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              const float *beta,
                                              const lwsparseMatDescr_t descrB,
                                              int nnzB,
                                              const float *csrSortedValB,
                                              const int *csrSortedRowPtrB,
                                              const int *csrSortedColIndB,
                                              const lwsparseMatDescr_t descrC,
                                              float *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrgeam(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              const double *alpha,
                                              const lwsparseMatDescr_t descrA,
                                              int nnzA,
                                              const double *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              const double *beta,
                                              const lwsparseMatDescr_t descrB,
                                              int nnzB,
                                              const double *csrSortedValB,
                                              const int *csrSortedRowPtrB,
                                              const int *csrSortedColIndB,
                                              const lwsparseMatDescr_t descrC,
                                              double *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);
    
lwsparseStatus_t LWSPARSEAPI lwsparseCcsrgeam(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              const lwComplex *alpha,
                                              const lwsparseMatDescr_t descrA,
                                              int nnzA,
                                              const lwComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              const lwComplex *beta,
                                              const lwsparseMatDescr_t descrB,
                                              int nnzB,
                                              const lwComplex *csrSortedValB,
                                              const int *csrSortedRowPtrB,
                                              const int *csrSortedColIndB,
                                              const lwsparseMatDescr_t descrC,
                                              lwComplex *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);
    
lwsparseStatus_t LWSPARSEAPI lwsparseZcsrgeam(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              const lwDoubleComplex *alpha,
                                              const lwsparseMatDescr_t descrA,
                                              int nnzA,
                                              const lwDoubleComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              const lwDoubleComplex *beta,
                                              const lwsparseMatDescr_t descrB,
                                              int nnzB,
                                              const lwDoubleComplex *csrSortedValB,
                                              const int *csrSortedRowPtrB,
                                              const int *csrSortedColIndB,
                                              const lwsparseMatDescr_t descrC,
                                              lwDoubleComplex *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);

lwsparseStatus_t LWSPARSEAPI lwsparseScsrgeam2_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *alpha,
    const lwsparseMatDescr_t descrA,
    int nnzA,
    const float *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const float *beta,
    const lwsparseMatDescr_t descrB,
    int nnzB,
    const float *csrSortedValB,
    const int *csrSortedRowPtrB,
    const int *csrSortedColIndB,
    const lwsparseMatDescr_t descrC,
    const float *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes );

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrgeam2_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *alpha,
    const lwsparseMatDescr_t descrA,
    int nnzA,
    const double *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const double *beta,
    const lwsparseMatDescr_t descrB,
    int nnzB,
    const double *csrSortedValB,
    const int *csrSortedRowPtrB,
    const int *csrSortedColIndB,
    const lwsparseMatDescr_t descrC,
    const double *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes );

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrgeam2_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwComplex *alpha,
    const lwsparseMatDescr_t descrA,
    int nnzA,
    const lwComplex *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const lwComplex *beta,
    const lwsparseMatDescr_t descrB,
    int nnzB,
    const lwComplex *csrSortedValB,
    const int *csrSortedRowPtrB,
    const int *csrSortedColIndB,
    const lwsparseMatDescr_t descrC,
    const lwComplex *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes );

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrgeam2_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwDoubleComplex *alpha,
    const lwsparseMatDescr_t descrA,
    int nnzA,
    const lwDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const lwDoubleComplex *beta,
    const lwsparseMatDescr_t descrB,
    int nnzB,
    const lwDoubleComplex *csrSortedValB,
    const int *csrSortedRowPtrB,
    const int *csrSortedColIndB,
    const lwsparseMatDescr_t descrC,
    const lwDoubleComplex *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes );

lwsparseStatus_t LWSPARSEAPI lwsparseXcsrgeam2Nnz(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwsparseMatDescr_t descrA,
    int nnzA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const lwsparseMatDescr_t descrB,
    int nnzB,
    const int *csrSortedRowPtrB,
    const int *csrSortedColIndB,
    const lwsparseMatDescr_t descrC,
    int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr,
    void *workspace );

lwsparseStatus_t LWSPARSEAPI lwsparseScsrgeam2(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *alpha,
    const lwsparseMatDescr_t descrA,
    int nnzA,
    const float *csrSortedValA,
    const int   *csrSortedRowPtrA,
    const int   *csrSortedColIndA,
    const float *beta,
    const lwsparseMatDescr_t descrB,
    int nnzB,
    const float *csrSortedValB,
    const int   *csrSortedRowPtrB,
    const int   *csrSortedColIndB,
    const lwsparseMatDescr_t descrC,
    float       *csrSortedValC,
    int         *csrSortedRowPtrC,
    int         *csrSortedColIndC,
    void *pBuffer );

lwsparseStatus_t  LWSPARSEAPI lwsparseDcsrgeam2(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *alpha,
    const lwsparseMatDescr_t descrA,
    int nnzA,
    const double *csrSortedValA,
    const int    *csrSortedRowPtrA,
    const int    *csrSortedColIndA,
    const double *beta,
    const lwsparseMatDescr_t descrB,
    int nnzB,
    const double *csrSortedValB,
    const int    *csrSortedRowPtrB,
    const int    *csrSortedColIndB,
    const lwsparseMatDescr_t descrC,
    double *csrSortedValC,
    int    *csrSortedRowPtrC,
    int    *csrSortedColIndC,
    void *pBuffer );

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrgeam2(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwComplex *alpha,
    const lwsparseMatDescr_t descrA,
    int nnzA,
    const lwComplex *csrSortedValA,
    const int       *csrSortedRowPtrA,
    const int       *csrSortedColIndA,
    const lwComplex *beta,
    const lwsparseMatDescr_t descrB,
    int nnzB,
    const lwComplex *csrSortedValB,
    const int       *csrSortedRowPtrB,
    const int       *csrSortedColIndB,
    const lwsparseMatDescr_t descrC,
    lwComplex *csrSortedValC,
    int       *csrSortedRowPtrC,
    int       *csrSortedColIndC,
    void *pBuffer );

lwsparseStatus_t  LWSPARSEAPI lwsparseZcsrgeam2(
    lwsparseHandle_t handle,
    int m,
    int n,
    const lwDoubleComplex *alpha,
    const lwsparseMatDescr_t descrA,
    int nnzA,
    const lwDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const lwDoubleComplex *beta,
    const lwsparseMatDescr_t descrB,
    int nnzB,
    const lwDoubleComplex *csrSortedValB,
    const int *csrSortedRowPtrB,
    const int *csrSortedColIndB,
    const lwsparseMatDescr_t descrC,
    lwDoubleComplex *csrSortedValC,
    int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    void *pBuffer );


/* --- Sparse Matrix Reorderings --- */

/* Description: Find an approximate coloring of a matrix stored in CSR format. */
lwsparseStatus_t LWSPARSEAPI lwsparseScsrcolor(lwsparseHandle_t handle,
                                               int m, 
                                               int nnz,
                                               const lwsparseMatDescr_t descrA, 
                                               const float *csrSortedValA, 
                                               const int *csrSortedRowPtrA, 
                                               const int *csrSortedColIndA,
                                               const float *fractionToColor,
                                               int *ncolors,
                                               int *coloring,
                                               int *reordering,
                                               const lwsparseColorInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsrcolor(lwsparseHandle_t handle,
                                               int m, 
                                               int nnz,
                                               const lwsparseMatDescr_t descrA, 
                                               const double *csrSortedValA, 
                                               const int *csrSortedRowPtrA, 
                                               const int *csrSortedColIndA,
                                               const double *fractionToColor,
                                               int *ncolors,
                                               int *coloring,
                                               int *reordering,
                                               const lwsparseColorInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsrcolor(lwsparseHandle_t handle,
                                               int m, 
                                               int nnz,
                                               const lwsparseMatDescr_t descrA, 
                                               const lwComplex *csrSortedValA, 
                                               const int *csrSortedRowPtrA, 
                                               const int *csrSortedColIndA,
                                               const float *fractionToColor,
                                               int *ncolors,
                                               int *coloring,
                                               int *reordering,
                                               const lwsparseColorInfo_t info);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsrcolor(lwsparseHandle_t handle,
                                               int m, 
                                               int nnz,
                                               const lwsparseMatDescr_t descrA, 
                                               const lwDoubleComplex *csrSortedValA, 
                                               const int *csrSortedRowPtrA, 
                                               const int *csrSortedColIndA,
                                               const double *fractionToColor,
                                               int *ncolors,
                                               int *coloring,
                                               int *reordering,
                                               const lwsparseColorInfo_t info);

/* --- Sparse Format Colwersion --- */

/* Description: This routine finds the total number of non-zero elements and 
   the number of non-zero elements per row or column in the dense matrix A. */
lwsparseStatus_t LWSPARSEAPI lwsparseSnnz(lwsparseHandle_t handle, 
                                          lwsparseDirection_t dirA, 
                                          int m, 
                                          int n, 
                                          const lwsparseMatDescr_t  descrA,
                                          const float *A, 
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalDevHostPtr);

lwsparseStatus_t LWSPARSEAPI lwsparseDnnz(lwsparseHandle_t handle, 
                                          lwsparseDirection_t dirA,  
                                          int m, 
                                          int n, 
                                          const lwsparseMatDescr_t  descrA,
                                          const double *A, 
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalDevHostPtr);

lwsparseStatus_t LWSPARSEAPI lwsparseCnnz(lwsparseHandle_t handle, 
                                          lwsparseDirection_t dirA,  
                                          int m, 
                                          int n, 
                                          const lwsparseMatDescr_t  descrA,
                                          const lwComplex *A,
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalDevHostPtr);

lwsparseStatus_t LWSPARSEAPI lwsparseZnnz(lwsparseHandle_t handle, 
                                          lwsparseDirection_t dirA,  
                                          int m, 
                                          int n, 
                                          const lwsparseMatDescr_t  descrA,
                                          const lwDoubleComplex *A,
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalDevHostPtr);

/* --- Sparse Format Colwersion --- */

/* Description: This routine finds the total number of non-zero elements and 
   the number of non-zero elements per row in a noncompressed csr matrix A. */
lwsparseStatus_t LWSPARSEAPI lwsparseSnnz_compress(lwsparseHandle_t handle, 
                                          int m, 
                                          const lwsparseMatDescr_t descr,
                                          const float *csrSortedValA, 
                                          const int *csrSortedRowPtrA, 
                                          int *nnzPerRow, 
                                          int *nnzC,
                                          float tol);

lwsparseStatus_t LWSPARSEAPI lwsparseDnnz_compress(lwsparseHandle_t handle, 
                                          int m, 
                                          const lwsparseMatDescr_t descr,
                                          const double *csrSortedValA, 
                                          const int *csrSortedRowPtrA, 
                                          int *nnzPerRow, 
                                          int *nnzC,
                                          double tol);

lwsparseStatus_t LWSPARSEAPI lwsparseCnnz_compress(lwsparseHandle_t handle, 
                                          int m, 
                                          const lwsparseMatDescr_t descr,
                                          const lwComplex *csrSortedValA, 
                                          const int *csrSortedRowPtrA, 
                                          int *nnzPerRow, 
                                          int *nnzC,
                                          lwComplex tol);

lwsparseStatus_t LWSPARSEAPI lwsparseZnnz_compress(lwsparseHandle_t handle, 
                                          int m, 
                                          const lwsparseMatDescr_t descr,
                                          const lwDoubleComplex *csrSortedValA, 
                                          const int *csrSortedRowPtrA, 
                                          int *nnzPerRow, 
                                          int *nnzC,
                                          lwDoubleComplex tol);
/* Description: This routine takes as input a csr form and compresses it to return a compressed csr form*/

lwsparseStatus_t LWSPARSEAPI lwsparseScsr2csr_compress(lwsparseHandle_t handle,
                                                      int m, 
                                                      int n,
                                                      const lwsparseMatDescr_t descrA,
                                                      const float *csrSortedValA,
                                                      const int *csrSortedColIndA,
                                                      const int *csrSortedRowPtrA, 
                                                      int nnzA,
                                                      const int *nnzPerRow, 
                                                      float *csrSortedValC,
                                                      int *csrSortedColIndC,
                                                      int *csrSortedRowPtrC,
                                                      float tol);        

lwsparseStatus_t LWSPARSEAPI lwsparseDcsr2csr_compress(lwsparseHandle_t handle,
                                                      int m, 
                                                      int n,
                                                      const lwsparseMatDescr_t descrA,
                                                      const double *csrSortedValA, 
                                                      const int *csrSortedColIndA,
                                                      const int * csrSortedRowPtrA, 
                                                      int  nnzA,
                                                      const int *nnzPerRow,  
                                                      double *csrSortedValC,
                                                      int *csrSortedColIndC,
                                                      int *csrSortedRowPtrC,
                                                      double tol);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsr2csr_compress(lwsparseHandle_t handle,
                                                        int m, 
                                                        int n,
                                                        const lwsparseMatDescr_t descrA,
                                                        const lwComplex *csrSortedValA, 
                                                        const int *csrSortedColIndA,
                                                        const int * csrSortedRowPtrA,
                                                        int nnzA,
                                                        const int *nnzPerRow,  
                                                        lwComplex *csrSortedValC,
                                                        int *csrSortedColIndC,
                                                        int *csrSortedRowPtrC,
                                                        lwComplex tol);                       

lwsparseStatus_t LWSPARSEAPI lwsparseZcsr2csr_compress(lwsparseHandle_t handle,
                                                      int m, 
                                                      int n,
                                                      const lwsparseMatDescr_t descrA,
                                                      const lwDoubleComplex *csrSortedValA,
                                                      const int *csrSortedColIndA,
                                                      const int * csrSortedRowPtrA,
                                                      int  nnzA,
                                                      const int *nnzPerRow, 
                                                      lwDoubleComplex *csrSortedValC,
                                                      int *csrSortedColIndC,
                                                      int *csrSortedRowPtrC,
                                                      lwDoubleComplex tol);                        
                                                                                                        
/* Description: This routine colwerts a dense matrix to a sparse matrix 
   in the CSR storage format, using the information computed by the 
   nnz routine. */
lwsparseStatus_t LWSPARSEAPI lwsparseSdense2csr(lwsparseHandle_t handle,
                                                int m, 
                                                int n,  
                                                const lwsparseMatDescr_t descrA,                            
                                                const float *A, 
                                                int lda,
                                                const int *nnzPerRow,                                                 
                                                float *csrSortedValA, 
                                                int *csrSortedRowPtrA, 
                                                int *csrSortedColIndA);
 
lwsparseStatus_t LWSPARSEAPI lwsparseDdense2csr(lwsparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const lwsparseMatDescr_t descrA,                                     
                                                const double *A, 
                                                int lda, 
                                                const int *nnzPerRow,                                                 
                                                double *csrSortedValA, 
                                                int *csrSortedRowPtrA, 
                                                int *csrSortedColIndA);
    
lwsparseStatus_t LWSPARSEAPI lwsparseCdense2csr(lwsparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const lwsparseMatDescr_t descrA,                                     
                                                const lwComplex *A, 
                                                int lda, 
                                                const int *nnzPerRow,                                                 
                                                lwComplex *csrSortedValA, 
                                                int *csrSortedRowPtrA, 
                                                int *csrSortedColIndA);
 
lwsparseStatus_t LWSPARSEAPI lwsparseZdense2csr(lwsparseHandle_t handle,
                                                int m, 
                                                int n,  
                                                const lwsparseMatDescr_t descrA,                                    
                                                const lwDoubleComplex *A, 
                                                int lda, 
                                                const int *nnzPerRow,                                                 
                                                lwDoubleComplex *csrSortedValA, 
                                                int *csrSortedRowPtrA, 
                                                int *csrSortedColIndA);

/* Description: This routine colwerts a sparse matrix in CSR storage format
   to a dense matrix. */
lwsparseStatus_t LWSPARSEAPI lwsparseScsr2dense(lwsparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const lwsparseMatDescr_t descrA,  
                                                const float *csrSortedValA, 
                                                const int *csrSortedRowPtrA, 
                                                const int *csrSortedColIndA,
                                                float *A, 
                                                int lda);
    
lwsparseStatus_t LWSPARSEAPI lwsparseDcsr2dense(lwsparseHandle_t handle, 
                                                int m, 
                                                int n, 
                                                const lwsparseMatDescr_t descrA, 
                                                const double *csrSortedValA, 
                                                const int *csrSortedRowPtrA, 
                                                const int *csrSortedColIndA,
                                                double *A, 
                                                int lda);
    
lwsparseStatus_t LWSPARSEAPI lwsparseCcsr2dense(lwsparseHandle_t handle, 
                                                int m, 
                                                int n, 
                                                const lwsparseMatDescr_t descrA, 
                                                const lwComplex *csrSortedValA, 
                                                const int *csrSortedRowPtrA, 
                                                const int *csrSortedColIndA,
                                                lwComplex *A, 
                                                int lda);
    
lwsparseStatus_t LWSPARSEAPI lwsparseZcsr2dense(lwsparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const lwsparseMatDescr_t descrA, 
                                                const lwDoubleComplex *csrSortedValA, 
                                                const int *csrSortedRowPtrA, 
                                                const int *csrSortedColIndA,
                                                lwDoubleComplex *A, 
                                                int lda); 
                                 
/* Description: This routine colwerts a dense matrix to a sparse matrix 
   in the CSC storage format, using the information computed by the 
   nnz routine. */
lwsparseStatus_t LWSPARSEAPI lwsparseSdense2csc(lwsparseHandle_t handle,
                                                int m, 
                                                int n,  
                                                const lwsparseMatDescr_t descrA,                            
                                                const float *A, 
                                                int lda,
                                                const int *nnzPerCol,                                                 
                                                float *cscSortedValA, 
                                                int *cscSortedRowIndA, 
                                                int *cscSortedColPtrA);
 
lwsparseStatus_t LWSPARSEAPI lwsparseDdense2csc(lwsparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const lwsparseMatDescr_t descrA,                                     
                                                const double *A, 
                                                int lda,
                                                const int *nnzPerCol,                                                
                                                double *cscSortedValA, 
                                                int *cscSortedRowIndA, 
                                                int *cscSortedColPtrA); 

lwsparseStatus_t LWSPARSEAPI lwsparseCdense2csc(lwsparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const lwsparseMatDescr_t descrA,                                     
                                                const lwComplex *A, 
                                                int lda, 
                                                const int *nnzPerCol,                                                 
                                                lwComplex *cscSortedValA, 
                                                int *cscSortedRowIndA, 
                                                int *cscSortedColPtrA);
    
lwsparseStatus_t LWSPARSEAPI lwsparseZdense2csc(lwsparseHandle_t handle,
                                                int m, 
                                                int n,  
                                                const lwsparseMatDescr_t descrA,                                    
                                                const lwDoubleComplex *A, 
                                                int lda, 
                                                const int *nnzPerCol,
                                                lwDoubleComplex *cscSortedValA, 
                                                int *cscSortedRowIndA, 
                                                int *cscSortedColPtrA);

/* Description: This routine colwerts a sparse matrix in CSC storage format
   to a dense matrix. */
lwsparseStatus_t LWSPARSEAPI lwsparseScsc2dense(lwsparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const lwsparseMatDescr_t descrA,  
                                                const float *cscSortedValA, 
                                                const int *cscSortedRowIndA, 
                                                const int *cscSortedColPtrA,
                                                float *A, 
                                                int lda);
    
lwsparseStatus_t LWSPARSEAPI lwsparseDcsc2dense(lwsparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const lwsparseMatDescr_t descrA, 
                                                const double *cscSortedValA, 
                                                const int *cscSortedRowIndA, 
                                                const int *cscSortedColPtrA,
                                                double *A, 
                                                int lda);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsc2dense(lwsparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const lwsparseMatDescr_t descrA, 
                                                const lwComplex *cscSortedValA, 
                                                const int *cscSortedRowIndA, 
                                                const int *cscSortedColPtrA,
                                                lwComplex *A, 
                                                int lda);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsc2dense(lwsparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const lwsparseMatDescr_t descrA, 
                                                const lwDoubleComplex *cscSortedValA, 
                                                const int *cscSortedRowIndA, 
                                                const int *cscSortedColPtrA,
                                                lwDoubleComplex *A, 
                                                int lda);
    
/* Description: This routine compresses the indecis of rows or columns.
   It can be interpreted as a colwersion from COO to CSR sparse storage
   format. */
lwsparseStatus_t LWSPARSEAPI lwsparseXcoo2csr(lwsparseHandle_t handle,
                                              const int *cooRowInd, 
                                              int nnz, 
                                              int m, 
                                              int *csrSortedRowPtr, 
                                              lwsparseIndexBase_t idxBase);
    
/* Description: This routine uncompresses the indecis of rows or columns.
   It can be interpreted as a colwersion from CSR to COO sparse storage
   format. */
lwsparseStatus_t LWSPARSEAPI lwsparseXcsr2coo(lwsparseHandle_t handle,
                                              const int *csrSortedRowPtr, 
                                              int nnz, 
                                              int m, 
                                              int *cooRowInd, 
                                              lwsparseIndexBase_t idxBase);     
    
/* Description: This routine colwerts a matrix from CSR to CSC sparse 
   storage format. The resulting matrix can be re-interpreted as a 
   transpose of the original matrix in CSR storage format. */
lwsparseStatus_t LWSPARSEAPI lwsparseCsr2cscEx(lwsparseHandle_t handle,
                                              int m, 
                                              int n, 
                                              int nnz,
                                              const void  *csrSortedVal, 
                                              lwdaDataType csrSortedValtype,
                                              const int *csrSortedRowPtr, 
                                              const int *csrSortedColInd, 
                                              void *cscSortedVal, 
                                              lwdaDataType cscSortedValtype,
                                              int *cscSortedRowInd, 
                                              int *cscSortedColPtr, 
                                              lwsparseAction_t copyValues, 
                                              lwsparseIndexBase_t idxBase,
                                              lwdaDataType exelwtiontype);
    
lwsparseStatus_t LWSPARSEAPI lwsparseScsr2csc(lwsparseHandle_t handle,
                                              int m, 
                                              int n, 
                                              int nnz,
                                              const float  *csrSortedVal, 
                                              const int *csrSortedRowPtr, 
                                              const int *csrSortedColInd, 
                                              float *cscSortedVal, 
                                              int *cscSortedRowInd, 
                                              int *cscSortedColPtr, 
                                              lwsparseAction_t copyValues, 
                                              lwsparseIndexBase_t idxBase);
    
lwsparseStatus_t LWSPARSEAPI lwsparseDcsr2csc(lwsparseHandle_t handle,
                                              int m, 
                                              int n,
                                              int nnz,
                                              const double  *csrSortedVal, 
                                              const int *csrSortedRowPtr, 
                                              const int *csrSortedColInd,
                                              double *cscSortedVal, 
                                              int *cscSortedRowInd, 
                                              int *cscSortedColPtr,
                                              lwsparseAction_t copyValues, 
                                              lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsr2csc(lwsparseHandle_t handle,
                                              int m, 
                                              int n,
                                              int nnz,
                                              const lwComplex  *csrSortedVal, 
                                              const int *csrSortedRowPtr, 
                                              const int *csrSortedColInd,
                                              lwComplex *cscSortedVal, 
                                              int *cscSortedRowInd, 
                                              int *cscSortedColPtr, 
                                              lwsparseAction_t copyValues, 
                                              lwsparseIndexBase_t idxBase);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsr2csc(lwsparseHandle_t handle,
                                              int m, 
                                              int n, 
                                              int nnz,
                                              const lwDoubleComplex *csrSortedVal, 
                                              const int *csrSortedRowPtr, 
                                              const int *csrSortedColInd, 
                                              lwDoubleComplex *cscSortedVal, 
                                              int *cscSortedRowInd, 
                                              int *cscSortedColPtr,
                                              lwsparseAction_t copyValues, 
                                              lwsparseIndexBase_t idxBase);

/* Description: This routine colwerts a dense matrix to a sparse matrix 
   in HYB storage format. */
lwsparseStatus_t LWSPARSEAPI lwsparseSdense2hyb(lwsparseHandle_t handle,
                                                int m,
                                                int n,
                                                const lwsparseMatDescr_t descrA,
                                                const float *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                lwsparseHybMat_t hybA,
                                                int userEllWidth,
                                                lwsparseHybPartition_t partitionType);

lwsparseStatus_t LWSPARSEAPI lwsparseDdense2hyb(lwsparseHandle_t handle,
                                                int m,
                                                int n,
                                                const lwsparseMatDescr_t descrA,
                                                const double *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                lwsparseHybMat_t hybA,
                                                int userEllWidth,
                                                lwsparseHybPartition_t partitionType);

lwsparseStatus_t LWSPARSEAPI lwsparseCdense2hyb(lwsparseHandle_t handle,
                                                int m,
                                                int n,
                                                const lwsparseMatDescr_t descrA,
                                                const lwComplex *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                lwsparseHybMat_t hybA,
                                                int userEllWidth,
                                                lwsparseHybPartition_t partitionType);

lwsparseStatus_t LWSPARSEAPI lwsparseZdense2hyb(lwsparseHandle_t handle,
                                                int m,
                                                int n,
                                                const lwsparseMatDescr_t descrA,
                                                const lwDoubleComplex *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                lwsparseHybMat_t hybA,
                                                int userEllWidth,
                                                lwsparseHybPartition_t partitionType);

/* Description: This routine colwerts a sparse matrix in HYB storage format
   to a dense matrix. */
lwsparseStatus_t LWSPARSEAPI lwsparseShyb2dense(lwsparseHandle_t handle,
                                                const lwsparseMatDescr_t descrA,
                                                const lwsparseHybMat_t hybA,
                                                float *A,
                                                int lda);

lwsparseStatus_t LWSPARSEAPI lwsparseDhyb2dense(lwsparseHandle_t handle,
                                                const lwsparseMatDescr_t descrA,
                                                const lwsparseHybMat_t hybA,
                                                double *A,
                                                int lda);
    
lwsparseStatus_t LWSPARSEAPI lwsparseChyb2dense(lwsparseHandle_t handle,
                                                const lwsparseMatDescr_t descrA,
                                                const lwsparseHybMat_t hybA,
                                                lwComplex *A,
                                                int lda);

lwsparseStatus_t LWSPARSEAPI lwsparseZhyb2dense(lwsparseHandle_t handle,
                                                const lwsparseMatDescr_t descrA,
                                                const lwsparseHybMat_t hybA,
                                                lwDoubleComplex *A,
                                                int lda);

/* Description: This routine colwerts a sparse matrix in CSR storage format
   to a sparse matrix in HYB storage format. */
lwsparseStatus_t LWSPARSEAPI lwsparseScsr2hyb(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              const lwsparseMatDescr_t descrA,
                                              const float *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              lwsparseHybMat_t hybA,
                                              int userEllWidth,
                                              lwsparseHybPartition_t partitionType);
    
lwsparseStatus_t LWSPARSEAPI lwsparseDcsr2hyb(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              const lwsparseMatDescr_t descrA,
                                              const double *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              lwsparseHybMat_t hybA,
                                              int userEllWidth,
                                              lwsparseHybPartition_t partitionType);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsr2hyb(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              const lwsparseMatDescr_t descrA,
                                              const lwComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              lwsparseHybMat_t hybA,
                                              int userEllWidth,
                                              lwsparseHybPartition_t partitionType);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsr2hyb(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              const lwsparseMatDescr_t descrA,
                                              const lwDoubleComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              lwsparseHybMat_t hybA,
                                              int userEllWidth,
                                              lwsparseHybPartition_t partitionType);

/* Description: This routine colwerts a sparse matrix in HYB storage format
   to a sparse matrix in CSR storage format. */
lwsparseStatus_t LWSPARSEAPI lwsparseShyb2csr(lwsparseHandle_t handle,
                                              const lwsparseMatDescr_t descrA,
                                              const lwsparseHybMat_t hybA,
                                              float *csrSortedValA,
                                              int *csrSortedRowPtrA,
                                              int *csrSortedColIndA);

lwsparseStatus_t LWSPARSEAPI lwsparseDhyb2csr(lwsparseHandle_t handle,
                                              const lwsparseMatDescr_t descrA,
                                              const lwsparseHybMat_t hybA,
                                              double *csrSortedValA,
                                              int *csrSortedRowPtrA,
                                              int *csrSortedColIndA);              

lwsparseStatus_t LWSPARSEAPI lwsparseChyb2csr(lwsparseHandle_t handle,
                                              const lwsparseMatDescr_t descrA,
                                              const lwsparseHybMat_t hybA,
                                              lwComplex *csrSortedValA,
                                              int *csrSortedRowPtrA,
                                              int *csrSortedColIndA);

lwsparseStatus_t LWSPARSEAPI lwsparseZhyb2csr(lwsparseHandle_t handle,
                                              const lwsparseMatDescr_t descrA,
                                              const lwsparseHybMat_t hybA,
                                              lwDoubleComplex *csrSortedValA,
                                              int *csrSortedRowPtrA,
                                              int *csrSortedColIndA);

/* Description: This routine colwerts a sparse matrix in CSC storage format
   to a sparse matrix in HYB storage format. */
lwsparseStatus_t LWSPARSEAPI lwsparseScsc2hyb(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              const lwsparseMatDescr_t descrA,
                                              const float *cscSortedValA,
                                              const int *cscSortedRowIndA,
                                              const int *cscSortedColPtrA,
                                              lwsparseHybMat_t hybA,
                                              int userEllWidth,
                                              lwsparseHybPartition_t partitionType);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsc2hyb(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              const lwsparseMatDescr_t descrA,
                                              const double *cscSortedValA,
                                              const int *cscSortedRowIndA,
                                              const int *cscSortedColPtrA,
                                              lwsparseHybMat_t hybA,
                                              int userEllWidth,
                                              lwsparseHybPartition_t partitionType);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsc2hyb(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              const lwsparseMatDescr_t descrA,
                                              const lwComplex *cscSortedValA,
                                              const int *cscSortedRowIndA,
                                              const int *cscSortedColPtrA,
                                              lwsparseHybMat_t hybA,
                                              int userEllWidth,
                                              lwsparseHybPartition_t partitionType);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsc2hyb(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              const lwsparseMatDescr_t descrA,
                                              const lwDoubleComplex *cscSortedValA,
                                              const int *cscSortedRowIndA,
                                              const int *cscSortedColPtrA,
                                              lwsparseHybMat_t hybA,
                                              int userEllWidth,
                                              lwsparseHybPartition_t partitionType);

/* Description: This routine colwerts a sparse matrix in HYB storage format
   to a sparse matrix in CSC storage format. */
lwsparseStatus_t LWSPARSEAPI lwsparseShyb2csc(lwsparseHandle_t handle,
                                              const lwsparseMatDescr_t descrA,
                                              const lwsparseHybMat_t hybA,
                                              float *cscSortedVal,
                                              int *cscSortedRowInd,
                                              int *cscSortedColPtr);

lwsparseStatus_t LWSPARSEAPI lwsparseDhyb2csc(lwsparseHandle_t handle,
                                              const lwsparseMatDescr_t descrA,
                                              const lwsparseHybMat_t hybA,
                                              double *cscSortedVal,
                                              int *cscSortedRowInd,
                                              int *cscSortedColPtr);

lwsparseStatus_t LWSPARSEAPI lwsparseChyb2csc(lwsparseHandle_t handle,
                                              const lwsparseMatDescr_t descrA,
                                              const lwsparseHybMat_t hybA,
                                              lwComplex *cscSortedVal,
                                              int *cscSortedRowInd,
                                              int *cscSortedColPtr);

lwsparseStatus_t LWSPARSEAPI lwsparseZhyb2csc(lwsparseHandle_t handle,
                                              const lwsparseMatDescr_t descrA,
                                              const lwsparseHybMat_t hybA,
                                              lwDoubleComplex *cscSortedVal,
                                              int *cscSortedRowInd,
                                              int *cscSortedColPtr);

/* Description: This routine colwerts a sparse matrix in CSR storage format
   to a sparse matrix in block-CSR storage format. */
lwsparseStatus_t LWSPARSEAPI lwsparseXcsr2bsrNnz(lwsparseHandle_t handle,
                                                 lwsparseDirection_t dirA,
                                                 int m,
                                                 int n,
                                                 const lwsparseMatDescr_t descrA,
                                                 const int *csrSortedRowPtrA,
                                                 const int *csrSortedColIndA,
                                                 int blockDim,
                                                 const lwsparseMatDescr_t descrC,
                                                 int *bsrSortedRowPtrC,
                                                 int *nnzTotalDevHostPtr);

lwsparseStatus_t LWSPARSEAPI lwsparseScsr2bsr(lwsparseHandle_t handle,
                                              lwsparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const lwsparseMatDescr_t descrA,
                                              const float *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              int blockDim,
                                              const lwsparseMatDescr_t descrC,
                                              float *bsrSortedValC,
                                              int *bsrSortedRowPtrC,
                                              int *bsrSortedColIndC);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsr2bsr(lwsparseHandle_t handle,
                                              lwsparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const lwsparseMatDescr_t descrA,
                                              const double *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              int blockDim,
                                              const lwsparseMatDescr_t descrC,
                                              double *bsrSortedValC,
                                              int *bsrSortedRowPtrC,
                                              int *bsrSortedColIndC);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsr2bsr(lwsparseHandle_t handle,
                                              lwsparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const lwsparseMatDescr_t descrA,
                                              const lwComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              int blockDim,
                                              const lwsparseMatDescr_t descrC,
                                              lwComplex *bsrSortedValC,
                                              int *bsrSortedRowPtrC,
                                              int *bsrSortedColIndC);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsr2bsr(lwsparseHandle_t handle,
                                              lwsparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const lwsparseMatDescr_t descrA,
                                              const lwDoubleComplex *csrSortedValA,
                                              const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              int blockDim,
                                              const lwsparseMatDescr_t descrC,
                                              lwDoubleComplex *bsrSortedValC,
                                              int *bsrSortedRowPtrC,
                                              int *bsrSortedColIndC);

/* Description: This routine colwerts a sparse matrix in block-CSR storage format
   to a sparse matrix in CSR storage format. */
lwsparseStatus_t LWSPARSEAPI lwsparseSbsr2csr(lwsparseHandle_t handle,
                                              lwsparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const lwsparseMatDescr_t descrA,
                                              const float *bsrSortedValA,
                                              const int *bsrSortedRowPtrA,
                                              const int *bsrSortedColIndA,
                                              int blockDim,
                                              const lwsparseMatDescr_t descrC,
                                              float *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);

lwsparseStatus_t LWSPARSEAPI lwsparseDbsr2csr(lwsparseHandle_t handle,
                                              lwsparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const lwsparseMatDescr_t descrA,
                                              const double *bsrSortedValA,
                                              const int *bsrSortedRowPtrA,
                                              const int *bsrSortedColIndA,
                                              int   blockDim,
                                              const lwsparseMatDescr_t descrC,
                                              double *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);

lwsparseStatus_t LWSPARSEAPI lwsparseCbsr2csr(lwsparseHandle_t handle,
                                              lwsparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const lwsparseMatDescr_t descrA,
                                              const lwComplex *bsrSortedValA,
                                              const int *bsrSortedRowPtrA,
                                              const int *bsrSortedColIndA,
                                              int blockDim,
                                              const lwsparseMatDescr_t descrC,
                                              lwComplex *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);

lwsparseStatus_t LWSPARSEAPI lwsparseZbsr2csr(lwsparseHandle_t handle,
                                              lwsparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const lwsparseMatDescr_t descrA,
                                              const lwDoubleComplex *bsrSortedValA,
                                              const int *bsrSortedRowPtrA,
                                              const int *bsrSortedColIndA,
                                              int blockDim,
                                              const lwsparseMatDescr_t descrC,
                                              lwDoubleComplex *csrSortedValC,
                                              int *csrSortedRowPtrC,
                                              int *csrSortedColIndC);

/* Description: This routine colwerts a sparse matrix in general block-CSR storage format
   to a sparse matrix in general block-CSC storage format. */
lwsparseStatus_t LWSPARSEAPI lwsparseSgebsr2gebsc_bufferSize(lwsparseHandle_t handle,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const float *bsrSortedVal,
                                                             const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd,
                                                             int rowBlockDim,
                                                             int colBlockDim,
                                                             int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDgebsr2gebsc_bufferSize(lwsparseHandle_t handle,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const double *bsrSortedVal,
                                                             const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd,
                                                             int rowBlockDim,
                                                             int colBlockDim,
                                                             int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCgebsr2gebsc_bufferSize(lwsparseHandle_t handle,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const lwComplex *bsrSortedVal,
                                                             const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd,
                                                             int rowBlockDim,
                                                             int colBlockDim,
                                                             int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZgebsr2gebsc_bufferSize(lwsparseHandle_t handle,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const lwDoubleComplex *bsrSortedVal,
                                                             const int *bsrSortedRowPtr,
                                                             const int *bsrSortedColInd,
                                                             int rowBlockDim,
                                                             int colBlockDim,
                                                             int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseSgebsr2gebsc_bufferSizeExt(lwsparseHandle_t handle,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const float *bsrSortedVal,
                                                                const int *bsrSortedRowPtr,
                                                                const int *bsrSortedColInd,
                                                                int rowBlockDim,
                                                                int colBlockDim,
                                                                size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseDgebsr2gebsc_bufferSizeExt(lwsparseHandle_t handle,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const double *bsrSortedVal,
                                                                const int *bsrSortedRowPtr,
                                                                const int *bsrSortedColInd,
                                                                int rowBlockDim,
                                                                int colBlockDim,
                                                                size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseCgebsr2gebsc_bufferSizeExt(lwsparseHandle_t handle,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const lwComplex *bsrSortedVal,
                                                                const int *bsrSortedRowPtr,
                                                                const int *bsrSortedColInd,
                                                                int rowBlockDim,
                                                                int colBlockDim,
                                                                size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseZgebsr2gebsc_bufferSizeExt(lwsparseHandle_t handle,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const lwDoubleComplex *bsrSortedVal,
                                                                const int *bsrSortedRowPtr,
                                                                const int *bsrSortedColInd,
                                                                int rowBlockDim,
                                                                int colBlockDim,
                                                                size_t *pBufferSize);


lwsparseStatus_t LWSPARSEAPI lwsparseSgebsr2gebsc(lwsparseHandle_t handle,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const float *bsrSortedVal,
                                                  const int *bsrSortedRowPtr,
                                                  const int *bsrSortedColInd,
                                                  int rowBlockDim,
                                                  int colBlockDim,
                                                  float *bscVal,
                                                  int *bscRowInd,
                                                  int *bscColPtr,
                                                  lwsparseAction_t copyValues,
                                                  lwsparseIndexBase_t baseIdx, 
                                                  void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDgebsr2gebsc(lwsparseHandle_t handle,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const double *bsrSortedVal,
                                                  const int *bsrSortedRowPtr,
                                                  const int *bsrSortedColInd,
                                                  int rowBlockDim,
                                                  int colBlockDim,
                                                  double *bscVal,
                                                  int *bscRowInd,
                                                  int *bscColPtr,
                                                  lwsparseAction_t copyValues,
                                                  lwsparseIndexBase_t baseIdx,
                                                  void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCgebsr2gebsc(lwsparseHandle_t handle,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const lwComplex *bsrSortedVal,
                                                  const int *bsrSortedRowPtr,
                                                  const int *bsrSortedColInd,
                                                  int rowBlockDim,
                                                  int colBlockDim,
                                                  lwComplex *bscVal,
                                                  int *bscRowInd,
                                                  int *bscColPtr,
                                                  lwsparseAction_t copyValues,
                                                  lwsparseIndexBase_t baseIdx,
                                                  void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZgebsr2gebsc(lwsparseHandle_t handle,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const lwDoubleComplex *bsrSortedVal,
                                                  const int *bsrSortedRowPtr,
                                                  const int *bsrSortedColInd,
                                                  int rowBlockDim,
                                                  int colBlockDim,
                                                  lwDoubleComplex *bscVal,
                                                  int *bscRowInd,
                                                  int *bscColPtr,
                                                  lwsparseAction_t copyValues,
                                                  lwsparseIndexBase_t baseIdx,
                                                  void *pBuffer);

/* Description: This routine colwerts a sparse matrix in general block-CSR storage format
   to a sparse matrix in CSR storage format. */
lwsparseStatus_t LWSPARSEAPI lwsparseXgebsr2csr(lwsparseHandle_t handle,
                                                lwsparseDirection_t dirA,
                                                int mb,
                                                int nb,
                                                const lwsparseMatDescr_t descrA,
                                                const int    *bsrSortedRowPtrA,
                                                const int    *bsrSortedColIndA,
                                                int   rowBlockDim,
                                                int   colBlockDim,
                                                const lwsparseMatDescr_t descrC,
                                                int    *csrSortedRowPtrC,
                                                int    *csrSortedColIndC );

lwsparseStatus_t LWSPARSEAPI lwsparseSgebsr2csr(lwsparseHandle_t handle,
                                                lwsparseDirection_t dirA,
                                                int mb,
                                                int nb,
                                                const lwsparseMatDescr_t descrA,
                                                const float *bsrSortedValA,
                                                const int    *bsrSortedRowPtrA,
                                                const int    *bsrSortedColIndA,
                                                int   rowBlockDim,
                                                int   colBlockDim,
                                                const lwsparseMatDescr_t descrC,
                                                float  *csrSortedValC,
                                                int    *csrSortedRowPtrC,
                                                int    *csrSortedColIndC );


lwsparseStatus_t LWSPARSEAPI lwsparseDgebsr2csr(lwsparseHandle_t handle,
                                                lwsparseDirection_t dirA,
                                                int mb,
                                                int nb,
                                                const lwsparseMatDescr_t descrA,
                                                const double *bsrSortedValA,
                                                const int    *bsrSortedRowPtrA,
                                                const int    *bsrSortedColIndA,
                                                int   rowBlockDim,
                                                int   colBlockDim,
                                                const lwsparseMatDescr_t descrC,
                                                double  *csrSortedValC,
                                                int    *csrSortedRowPtrC,
                                                int    *csrSortedColIndC );


lwsparseStatus_t LWSPARSEAPI lwsparseCgebsr2csr(lwsparseHandle_t handle,
                                                lwsparseDirection_t dirA,
                                                int mb,
                                                int nb,
                                                const lwsparseMatDescr_t descrA,
                                                const lwComplex *bsrSortedValA,
                                                const int    *bsrSortedRowPtrA,
                                                const int    *bsrSortedColIndA,
                                                int   rowBlockDim,
                                                int   colBlockDim,
                                                const lwsparseMatDescr_t descrC,
                                                lwComplex  *csrSortedValC,
                                                int    *csrSortedRowPtrC,
                                                int    *csrSortedColIndC );


lwsparseStatus_t LWSPARSEAPI lwsparseZgebsr2csr(lwsparseHandle_t handle,
                                                lwsparseDirection_t dirA,
                                                int mb,
                                                int nb,
                                                const lwsparseMatDescr_t descrA,
                                                const lwDoubleComplex *bsrSortedValA,
                                                const int    *bsrSortedRowPtrA,
                                                const int    *bsrSortedColIndA,
                                                int   rowBlockDim,
                                                int   colBlockDim,
                                                const lwsparseMatDescr_t descrC,
                                                lwDoubleComplex  *csrSortedValC,
                                                int    *csrSortedRowPtrC,
                                                int    *csrSortedColIndC );

/* Description: This routine colwerts a sparse matrix in CSR storage format
   to a sparse matrix in general block-CSR storage format. */
lwsparseStatus_t LWSPARSEAPI lwsparseScsr2gebsr_bufferSize(lwsparseHandle_t handle,
                                                           lwsparseDirection_t dirA,
                                                           int m,
                                                           int n,
                                                           const lwsparseMatDescr_t descrA,
                                                           const float *csrSortedValA,
                                                           const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA,
                                                           int rowBlockDim,
                                                           int colBlockDim,
                                                           int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsr2gebsr_bufferSize(lwsparseHandle_t handle,
                                                           lwsparseDirection_t dirA,
                                                           int m,
                                                           int n,
                                                           const lwsparseMatDescr_t descrA,
                                                           const double *csrSortedValA,
                                                           const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA,
                                                           int rowBlockDim,
                                                           int colBlockDim,
                                                           int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsr2gebsr_bufferSize(lwsparseHandle_t handle,
                                                           lwsparseDirection_t dirA,
                                                           int m,
                                                           int n,
                                                           const lwsparseMatDescr_t descrA,
                                                           const lwComplex *csrSortedValA,
                                                           const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA,
                                                           int rowBlockDim,
                                                           int colBlockDim,
                                                           int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsr2gebsr_bufferSize(lwsparseHandle_t handle,
                                                           lwsparseDirection_t dirA,
                                                           int m,
                                                           int n,
                                                           const lwsparseMatDescr_t descrA,
                                                           const lwDoubleComplex *csrSortedValA,
                                                           const int *csrSortedRowPtrA,
                                                           const int *csrSortedColIndA,
                                                           int rowBlockDim,
                                                           int colBlockDim,
                                                           int *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseScsr2gebsr_bufferSizeExt(lwsparseHandle_t handle,
                                                              lwsparseDirection_t dirA,
                                                              int m,
                                                              int n,
                                                              const lwsparseMatDescr_t descrA,
                                                              const float *csrSortedValA,
                                                              const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA,
                                                              int rowBlockDim,
                                                              int colBlockDim,
                                                              size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsr2gebsr_bufferSizeExt(lwsparseHandle_t handle,
                                                              lwsparseDirection_t dirA,
                                                              int m,
                                                              int n,
                                                              const lwsparseMatDescr_t descrA,
                                                              const double *csrSortedValA,
                                                              const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA,
                                                              int rowBlockDim,
                                                              int colBlockDim,
                                                              size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsr2gebsr_bufferSizeExt(lwsparseHandle_t handle,
                                                              lwsparseDirection_t dirA,
                                                              int m,
                                                              int n,
                                                              const lwsparseMatDescr_t descrA,
                                                              const lwComplex *csrSortedValA,
                                                              const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA,
                                                              int rowBlockDim,
                                                              int colBlockDim,
                                                              size_t *pBufferSize);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsr2gebsr_bufferSizeExt(lwsparseHandle_t handle,
                                                              lwsparseDirection_t dirA,
                                                              int m,
                                                              int n,
                                                              const lwsparseMatDescr_t descrA,
                                                              const lwDoubleComplex *csrSortedValA,
                                                              const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA,
                                                              int rowBlockDim,
                                                              int colBlockDim,
                                                              size_t *pBufferSize);



lwsparseStatus_t LWSPARSEAPI lwsparseXcsr2gebsrNnz(lwsparseHandle_t handle,
                                                   lwsparseDirection_t dirA,
                                                   int m,
                                                   int n,
                                                   const lwsparseMatDescr_t descrA,
                                                   const int *csrSortedRowPtrA,
                                                   const int *csrSortedColIndA,
                                                   const lwsparseMatDescr_t descrC,
                                                   int *bsrSortedRowPtrC,
                                                   int rowBlockDim,
                                                   int colBlockDim,
                                                   int *nnzTotalDevHostPtr,
                                                   void *pBuffer );

lwsparseStatus_t LWSPARSEAPI lwsparseScsr2gebsr(lwsparseHandle_t handle,
                                                lwsparseDirection_t dirA,
                                                int m,
                                                int n,
                                                const lwsparseMatDescr_t descrA,
                                                const float *csrSortedValA,
                                                const int *csrSortedRowPtrA,
                                                const int *csrSortedColIndA,
                                                const lwsparseMatDescr_t descrC,
                                                float *bsrSortedValC,
                                                int *bsrSortedRowPtrC,
                                                int *bsrSortedColIndC,
                                                int rowBlockDim,
                                                int colBlockDim,
                                                void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsr2gebsr(lwsparseHandle_t handle,
                                                lwsparseDirection_t dirA,
                                                int m,
                                                int n,
                                                const lwsparseMatDescr_t descrA,
                                                const double *csrSortedValA,
                                                const int *csrSortedRowPtrA,
                                                const int *csrSortedColIndA,
                                                const lwsparseMatDescr_t descrC,
                                                double *bsrSortedValC,
                                                int *bsrSortedRowPtrC,
                                                int *bsrSortedColIndC,
                                                int rowBlockDim,
                                                int colBlockDim,
                                                void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsr2gebsr(lwsparseHandle_t handle,
                                                lwsparseDirection_t dirA,
                                                int m,
                                                int n,
                                                const lwsparseMatDescr_t descrA,
                                                const lwComplex *csrSortedValA,
                                                const int *csrSortedRowPtrA,
                                                const int *csrSortedColIndA,
                                                const lwsparseMatDescr_t descrC,
                                                lwComplex *bsrSortedValC,
                                                int *bsrSortedRowPtrC,
                                                int *bsrSortedColIndC,
                                                int rowBlockDim,
                                                int colBlockDim,
                                                void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsr2gebsr(lwsparseHandle_t handle,
                                                lwsparseDirection_t dirA,
                                                int m,
                                                int n,
                                                const lwsparseMatDescr_t descrA,
                                                const lwDoubleComplex *csrSortedValA,
                                                const int *csrSortedRowPtrA,
                                                const int *csrSortedColIndA,
                                                const lwsparseMatDescr_t descrC,
                                                lwDoubleComplex *bsrSortedValC,
                                                int *bsrSortedRowPtrC,
                                                int *bsrSortedColIndC,
                                                int rowBlockDim,
                                                int colBlockDim,
                                                void *pBuffer);

/* Description: This routine colwerts a sparse matrix in general block-CSR storage format
   to a sparse matrix in general block-CSR storage format with different block size. */
lwsparseStatus_t LWSPARSEAPI lwsparseSgebsr2gebsr_bufferSize(lwsparseHandle_t handle,
                                                             lwsparseDirection_t dirA,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const lwsparseMatDescr_t descrA,
                                                             const float *bsrSortedValA,
                                                             const int *bsrSortedRowPtrA,
                                                             const int *bsrSortedColIndA,
                                                             int rowBlockDimA,
                                                             int colBlockDimA,
                                                             int rowBlockDimC,
                                                             int colBlockDimC,
                                                             int *pBufferSizeInBytes );

lwsparseStatus_t LWSPARSEAPI lwsparseDgebsr2gebsr_bufferSize(lwsparseHandle_t handle,
                                                             lwsparseDirection_t dirA,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const lwsparseMatDescr_t descrA,
                                                             const double *bsrSortedValA,
                                                             const int *bsrSortedRowPtrA,
                                                             const int *bsrSortedColIndA,
                                                             int rowBlockDimA,
                                                             int colBlockDimA,
                                                             int rowBlockDimC,
                                                             int colBlockDimC,
                                                             int *pBufferSizeInBytes );

lwsparseStatus_t LWSPARSEAPI lwsparseCgebsr2gebsr_bufferSize(lwsparseHandle_t handle,
                                                             lwsparseDirection_t dirA,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const lwsparseMatDescr_t descrA,
                                                             const lwComplex *bsrSortedValA,
                                                             const int *bsrSortedRowPtrA,
                                                             const int *bsrSortedColIndA,
                                                             int rowBlockDimA,
                                                             int colBlockDimA,
                                                             int rowBlockDimC,
                                                             int colBlockDimC,
                                                             int *pBufferSizeInBytes );

lwsparseStatus_t LWSPARSEAPI lwsparseZgebsr2gebsr_bufferSize(lwsparseHandle_t handle,
                                                             lwsparseDirection_t dirA,
                                                             int mb,
                                                             int nb,
                                                             int nnzb,
                                                             const lwsparseMatDescr_t descrA,
                                                             const lwDoubleComplex *bsrSortedValA,
                                                             const int *bsrSortedRowPtrA,
                                                             const int *bsrSortedColIndA,
                                                             int rowBlockDimA,
                                                             int colBlockDimA,
                                                             int rowBlockDimC,
                                                             int colBlockDimC,
                                                             int *pBufferSizeInBytes );


lwsparseStatus_t LWSPARSEAPI lwsparseSgebsr2gebsr_bufferSizeExt(lwsparseHandle_t handle,
                                                                lwsparseDirection_t dirA,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const lwsparseMatDescr_t descrA,
                                                                const float *bsrSortedValA,
                                                                const int    *bsrSortedRowPtrA,
                                                                const int    *bsrSortedColIndA,
                                                                int   rowBlockDimA,
                                                                int   colBlockDimA,
                                                                int   rowBlockDimC,
                                                                int   colBlockDimC,
                                                                size_t  *pBufferSize );

lwsparseStatus_t LWSPARSEAPI lwsparseDgebsr2gebsr_bufferSizeExt(lwsparseHandle_t handle,
                                                                lwsparseDirection_t dirA,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const lwsparseMatDescr_t descrA,
                                                                const double *bsrSortedValA,
                                                                const int    *bsrSortedRowPtrA,
                                                                const int    *bsrSortedColIndA,
                                                                int   rowBlockDimA,
                                                                int   colBlockDimA,
                                                                int   rowBlockDimC,
                                                                int   colBlockDimC,
                                                                size_t  *pBufferSize );

lwsparseStatus_t LWSPARSEAPI lwsparseCgebsr2gebsr_bufferSizeExt(lwsparseHandle_t handle,
                                                                lwsparseDirection_t dirA,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const lwsparseMatDescr_t descrA,
                                                                const lwComplex *bsrSortedValA,
                                                                const int    *bsrSortedRowPtrA,
                                                                const int    *bsrSortedColIndA,
                                                                int   rowBlockDimA,
                                                                int   colBlockDimA,
                                                                int   rowBlockDimC,
                                                                int   colBlockDimC,
                                                                size_t  *pBufferSize );

lwsparseStatus_t LWSPARSEAPI lwsparseZgebsr2gebsr_bufferSizeExt(lwsparseHandle_t handle,
                                                                lwsparseDirection_t dirA,
                                                                int mb,
                                                                int nb,
                                                                int nnzb,
                                                                const lwsparseMatDescr_t descrA,
                                                                const lwDoubleComplex *bsrSortedValA,
                                                                const int    *bsrSortedRowPtrA,
                                                                const int    *bsrSortedColIndA,
                                                                int   rowBlockDimA,
                                                                int   colBlockDimA,
                                                                int   rowBlockDimC,
                                                                int   colBlockDimC,
                                                                size_t  *pBufferSize );



lwsparseStatus_t LWSPARSEAPI lwsparseXgebsr2gebsrNnz(lwsparseHandle_t handle,
                                                     lwsparseDirection_t dirA,
                                                     int mb,
                                                     int nb,
                                                     int nnzb,
                                                     const lwsparseMatDescr_t descrA,
                                                     const int *bsrSortedRowPtrA,
                                                     const int *bsrSortedColIndA,
                                                     int rowBlockDimA,
                                                     int colBlockDimA,
                                                     const lwsparseMatDescr_t descrC,
                                                     int *bsrSortedRowPtrC,
                                                     int rowBlockDimC,
                                                     int colBlockDimC,
                                                     int *nnzTotalDevHostPtr,
                                                     void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseSgebsr2gebsr(lwsparseHandle_t handle,
                                                  lwsparseDirection_t dirA,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const lwsparseMatDescr_t descrA,
                                                  const float *bsrSortedValA,
                                                  const int *bsrSortedRowPtrA,
                                                  const int *bsrSortedColIndA,
                                                  int rowBlockDimA,
                                                  int colBlockDimA,
                                                  const lwsparseMatDescr_t descrC,
                                                  float *bsrSortedValC,
                                                  int *bsrSortedRowPtrC,
                                                  int *bsrSortedColIndC,
                                                  int rowBlockDimC,
                                                  int colBlockDimC,
                                                  void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDgebsr2gebsr(lwsparseHandle_t handle,
                                                  lwsparseDirection_t dirA,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const lwsparseMatDescr_t descrA,
                                                  const double *bsrSortedValA,
                                                  const int *bsrSortedRowPtrA,
                                                  const int *bsrSortedColIndA,
                                                  int rowBlockDimA,
                                                  int colBlockDimA,
                                                  const lwsparseMatDescr_t descrC,
                                                  double *bsrSortedValC,
                                                  int *bsrSortedRowPtrC,
                                                  int *bsrSortedColIndC,
                                                  int rowBlockDimC,
                                                  int colBlockDimC,
                                                  void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCgebsr2gebsr(lwsparseHandle_t handle,
                                                  lwsparseDirection_t dirA,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const lwsparseMatDescr_t descrA,
                                                  const lwComplex *bsrSortedValA,
                                                  const int *bsrSortedRowPtrA,
                                                  const int *bsrSortedColIndA,
                                                  int rowBlockDimA,
                                                  int colBlockDimA,
                                                  const lwsparseMatDescr_t descrC,
                                                  lwComplex *bsrSortedValC,
                                                  int *bsrSortedRowPtrC,
                                                  int *bsrSortedColIndC,
                                                  int rowBlockDimC,
                                                  int colBlockDimC,
                                                  void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZgebsr2gebsr(lwsparseHandle_t handle,
                                                  lwsparseDirection_t dirA,
                                                  int mb,
                                                  int nb,
                                                  int nnzb,
                                                  const lwsparseMatDescr_t descrA,
                                                  const lwDoubleComplex *bsrSortedValA,
                                                  const int *bsrSortedRowPtrA,
                                                  const int *bsrSortedColIndA,
                                                  int rowBlockDimA,
                                                  int colBlockDimA,
                                                  const lwsparseMatDescr_t descrC,
                                                  lwDoubleComplex *bsrSortedValC,
                                                  int *bsrSortedRowPtrC,
                                                  int *bsrSortedColIndC,
                                                  int rowBlockDimC,
                                                  int colBlockDimC,
                                                  void *pBuffer);

/* --- Sparse Matrix Sorting --- */

/* Description: Create a identity sequence p=[0,1,...,n-1]. */
lwsparseStatus_t LWSPARSEAPI lwsparseCreateIdentityPermutation(lwsparseHandle_t handle,
                                                               int n,
                                                               int *p);

/* Description: Sort sparse matrix stored in COO format */
lwsparseStatus_t LWSPARSEAPI lwsparseXcoosort_bufferSizeExt(lwsparseHandle_t handle,
                                                            int m,
                                                            int n,
                                                            int nnz,
                                                            const int *cooRowsA,
                                                            const int *cooColsA,
                                                            size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseXcoosortByRow(lwsparseHandle_t handle,
                                                   int m,
                                                   int n,
                                                   int nnz,
                                                   int *cooRowsA,
                                                   int *cooColsA,
                                                   int *P,
                                                   void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseXcoosortByColumn(lwsparseHandle_t handle,
                                                      int m,
                                                      int n, 
                                                      int nnz,
                                                      int *cooRowsA,
                                                      int *cooColsA,
                                                      int *P,
                                                      void *pBuffer);

/* Description: Sort sparse matrix stored in CSR format */
lwsparseStatus_t LWSPARSEAPI lwsparseXcsrsort_bufferSizeExt(lwsparseHandle_t handle,
                                                            int m,
                                                            int n,
                                                            int nnz,
                                                            const int *csrRowPtrA,
                                                            const int *csrColIndA,
                                                            size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseXcsrsort(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              int nnz,
                                              const lwsparseMatDescr_t descrA,
                                              const int *csrRowPtrA,
                                              int *csrColIndA,
                                              int *P,
                                              void *pBuffer);
    
/* Description: Sort sparse matrix stored in CSC format */
lwsparseStatus_t LWSPARSEAPI lwsparseXcscsort_bufferSizeExt(lwsparseHandle_t handle,
                                                            int m,
                                                            int n,
                                                            int nnz,
                                                            const int *cscColPtrA,
                                                            const int *cscRowIndA,
                                                            size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseXcscsort(lwsparseHandle_t handle,
                                              int m,
                                              int n,
                                              int nnz,
                                              const lwsparseMatDescr_t descrA,
                                              const int *cscColPtrA,
                                              int *cscRowIndA,
                                              int *P,
                                              void *pBuffer);

/* Description: Wrapper that sorts sparse matrix stored in CSR format 
   (without exposing the permutation). */
lwsparseStatus_t LWSPARSEAPI lwsparseScsru2csr_bufferSizeExt(lwsparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int nnz,
                                                             float *csrVal,
                                                             const int *csrRowPtr,
                                                             int *csrColInd,
                                                             csru2csrInfo_t  info,
                                                             size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsru2csr_bufferSizeExt(lwsparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int nnz,
                                                             double *csrVal,
                                                             const int *csrRowPtr,
                                                             int *csrColInd,
                                                             csru2csrInfo_t  info,
                                                             size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsru2csr_bufferSizeExt(lwsparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int nnz,
                                                             lwComplex *csrVal,
                                                             const int *csrRowPtr,
                                                             int *csrColInd,
                                                             csru2csrInfo_t  info,
                                                             size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsru2csr_bufferSizeExt(lwsparseHandle_t handle,
                                                             int m,
                                                             int n,
                                                             int nnz,
                                                             lwDoubleComplex *csrVal,
                                                             const int *csrRowPtr,
                                                             int *csrColInd,
                                                             csru2csrInfo_t  info,
                                                             size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseScsru2csr(lwsparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const lwsparseMatDescr_t descrA,
                                               float *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsru2csr(lwsparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const lwsparseMatDescr_t descrA,
                                               double *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsru2csr(lwsparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const lwsparseMatDescr_t descrA,
                                               lwComplex *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsru2csr(lwsparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const lwsparseMatDescr_t descrA,
                                               lwDoubleComplex *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

/* Description: Wrapper that un-sorts sparse matrix stored in CSR format 
   (without exposing the permutation). */
lwsparseStatus_t LWSPARSEAPI lwsparseScsr2csru(lwsparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const lwsparseMatDescr_t descrA,
                                               float *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDcsr2csru(lwsparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const lwsparseMatDescr_t descrA,
                                               double *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseCcsr2csru(lwsparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const lwsparseMatDescr_t descrA,
                                               lwComplex *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseZcsr2csru(lwsparseHandle_t handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               const lwsparseMatDescr_t descrA,
                                               lwDoubleComplex *csrVal,
                                               const int *csrRowPtr,
                                               int *csrColInd,
                                               csru2csrInfo_t  info,
                                               void *pBuffer);

/* Description: prune dense matrix to a sparse matrix with CSR format */
#if defined(__cplusplus)
lwsparseStatus_t LWSPARSEAPI lwsparseHpruneDense2csr_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    const __half *threshold,
    const lwsparseMatDescr_t descrC,
    const __half *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes);
#endif

lwsparseStatus_t LWSPARSEAPI lwsparseSpruneDense2csr_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    const float *threshold,
    const lwsparseMatDescr_t descrC,
    const float *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDpruneDense2csr_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    const double *threshold,
    const lwsparseMatDescr_t descrC,
    const double *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes);

#if defined(__cplusplus)
lwsparseStatus_t LWSPARSEAPI lwsparseHpruneDense2csrNnz(
    lwsparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    const __half *threshold,
    const lwsparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr,
    void *pBuffer);
#endif

lwsparseStatus_t LWSPARSEAPI lwsparseSpruneDense2csrNnz(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    const float *threshold,
    const lwsparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDpruneDense2csrNnz(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    const double *threshold,
    const lwsparseMatDescr_t descrC,
    int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr,
    void *pBuffer);

#if defined(__cplusplus)
lwsparseStatus_t LWSPARSEAPI lwsparseHpruneDense2csr(
    lwsparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    const __half *threshold,
    const lwsparseMatDescr_t descrC,
    __half *csrSortedValC,
    const int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    void *pBuffer);
#endif

lwsparseStatus_t LWSPARSEAPI lwsparseSpruneDense2csr(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    const float *threshold,
    const lwsparseMatDescr_t descrC,
    float *csrSortedValC,
    const int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDpruneDense2csr(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    const double *threshold,
    const lwsparseMatDescr_t descrC,
    double *csrSortedValC,
    const int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    void *pBuffer);

/* Description: prune sparse matrix with CSR format to another sparse matrix with CSR format */
#if defined(__cplusplus)
lwsparseStatus_t LWSPARSEAPI lwsparseHpruneCsr2csr_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const __half *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const __half *threshold,
    const lwsparseMatDescr_t descrC,
    const __half *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes);
#endif

lwsparseStatus_t LWSPARSEAPI lwsparseSpruneCsr2csr_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const float *threshold,
    const lwsparseMatDescr_t descrC,
    const float *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDpruneCsr2csr_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const double *threshold,
    const lwsparseMatDescr_t descrC,
    const double *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes);

#if defined(__cplusplus)
lwsparseStatus_t LWSPARSEAPI lwsparseHpruneCsr2csrNnz(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const __half *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const __half *threshold,
    const lwsparseMatDescr_t descrC,
    int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    void *pBuffer);
#endif

lwsparseStatus_t LWSPARSEAPI lwsparseSpruneCsr2csrNnz(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const float *threshold,
    const lwsparseMatDescr_t descrC,
    int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDpruneCsr2csrNnz(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const double *threshold,
    const lwsparseMatDescr_t descrC,
    int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    void *pBuffer);

#if defined(__cplusplus)
lwsparseStatus_t LWSPARSEAPI lwsparseHpruneCsr2csr(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const __half *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const __half *threshold,
    const lwsparseMatDescr_t descrC,
    __half *csrSortedValC,
    const int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    void *pBuffer);
#endif

lwsparseStatus_t LWSPARSEAPI lwsparseSpruneCsr2csr(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const float *threshold,
    const lwsparseMatDescr_t descrC,
    float *csrSortedValC,
    const int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDpruneCsr2csr(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    const double *threshold,
    const lwsparseMatDescr_t descrC,
    double *csrSortedValC,
    const int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    void *pBuffer);

/* Description: prune dense matrix to a sparse matrix with CSR format by percentage */
#if defined(__cplusplus)
lwsparseStatus_t LWSPARSEAPI lwsparseHpruneDense2csrByPercentage_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    const __half *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes);
#endif

lwsparseStatus_t LWSPARSEAPI lwsparseSpruneDense2csrByPercentage_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    const float *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDpruneDense2csrByPercentage_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    const double *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes);

#if defined(__cplusplus)
lwsparseStatus_t LWSPARSEAPI lwsparseHpruneDense2csrNnzByPercentage(
    lwsparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo_t info,
    void *pBuffer);
#endif

lwsparseStatus_t LWSPARSEAPI lwsparseSpruneDense2csrNnzByPercentage(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo_t info,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDpruneDense2csrNnzByPercentage(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo_t info,
    void *pBuffer);

#if defined(__cplusplus)
lwsparseStatus_t LWSPARSEAPI lwsparseHpruneDense2csrByPercentage(
    lwsparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    __half *csrSortedValC,
    const int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    pruneInfo_t info,
    void *pBuffer);
#endif

lwsparseStatus_t LWSPARSEAPI lwsparseSpruneDense2csrByPercentage(
    lwsparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    float *csrSortedValC,
    const int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    pruneInfo_t info,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDpruneDense2csrByPercentage(
    lwsparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    double *csrSortedValC,
    const int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    pruneInfo_t info,
    void *pBuffer);


/* Description: prune sparse matrix to a sparse matrix with CSR format by percentage*/
#if defined(__cplusplus)
lwsparseStatus_t LWSPARSEAPI lwsparseHpruneCsr2csrByPercentage_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const __half *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    const __half *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes);
#endif

lwsparseStatus_t LWSPARSEAPI lwsparseSpruneCsr2csrByPercentage_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    const float *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes);

lwsparseStatus_t LWSPARSEAPI lwsparseDpruneCsr2csrByPercentage_bufferSizeExt(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    const double *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes);

#if defined(__cplusplus)
lwsparseStatus_t LWSPARSEAPI lwsparseHpruneCsr2csrNnzByPercentage(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const __half *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo_t info,
    void *pBuffer);
#endif

lwsparseStatus_t LWSPARSEAPI lwsparseSpruneCsr2csrNnzByPercentage(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo_t info,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDpruneCsr2csrNnzByPercentage(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo_t info,
    void *pBuffer);

#if defined(__cplusplus)
lwsparseStatus_t LWSPARSEAPI lwsparseHpruneCsr2csrByPercentage(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const __half *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    __half *csrSortedValC,
    const int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    pruneInfo_t info,
    void *pBuffer);
#endif

lwsparseStatus_t LWSPARSEAPI lwsparseSpruneCsr2csrByPercentage(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const float *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    float *csrSortedValC,
    const int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    pruneInfo_t info,
    void *pBuffer);

lwsparseStatus_t LWSPARSEAPI lwsparseDpruneCsr2csrByPercentage(
    lwsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const lwsparseMatDescr_t descrA,
    const double *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage, /* between 0 to 100 */
    const lwsparseMatDescr_t descrC,
    double *csrSortedValC,
    const int *csrSortedRowPtrC,
    int *csrSortedColIndC,
    pruneInfo_t info,
    void *pBuffer);




#if defined(__cplusplus)
}
#endif /* __cplusplus */                         

#endif /* !defined(LWSPARSE_H_) */

