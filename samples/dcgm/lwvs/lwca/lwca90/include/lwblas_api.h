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

#if !defined(LWBLAS_API_H_)
#define LWBLAS_API_H_

#ifndef LWBLASWINAPI
#ifdef _WIN32
#define LWBLASWINAPI __stdcall
#else
#define LWBLASWINAPI 
#endif
#endif

#ifndef LWBLASAPI
#error "This file should not be included without defining LWBLASAPI"
#endif

#include "driver_types.h"
#include "lwComplex.h"   /* import complex data type */

#include <lwda_fp16.h>

#include "library_types.h"


#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/* LWBLAS status type returns */
typedef enum{
    LWBLAS_STATUS_SUCCESS         =0,
    LWBLAS_STATUS_NOT_INITIALIZED =1,
    LWBLAS_STATUS_ALLOC_FAILED    =3,
    LWBLAS_STATUS_ILWALID_VALUE   =7,
    LWBLAS_STATUS_ARCH_MISMATCH   =8,
    LWBLAS_STATUS_MAPPING_ERROR   =11,
    LWBLAS_STATUS_EXELWTION_FAILED=13,
    LWBLAS_STATUS_INTERNAL_ERROR  =14,
    LWBLAS_STATUS_NOT_SUPPORTED   =15,
    LWBLAS_STATUS_LICENSE_ERROR   =16
} lwblasStatus_t;


typedef enum {
    LWBLAS_FILL_MODE_LOWER=0, 
    LWBLAS_FILL_MODE_UPPER=1
} lwblasFillMode_t;

typedef enum {
    LWBLAS_DIAG_NON_UNIT=0, 
    LWBLAS_DIAG_UNIT=1
} lwblasDiagType_t; 

typedef enum {
    LWBLAS_SIDE_LEFT =0, 
    LWBLAS_SIDE_RIGHT=1
} lwblasSideMode_t; 


typedef enum {
    LWBLAS_OP_N=0,  
    LWBLAS_OP_T=1,  
    LWBLAS_OP_C=2  
} lwblasOperation_t;


typedef enum { 
    LWBLAS_POINTER_MODE_HOST   = 0,  
    LWBLAS_POINTER_MODE_DEVICE = 1        
} lwblasPointerMode_t;

typedef enum { 
    LWBLAS_ATOMICS_NOT_ALLOWED   = 0,  
    LWBLAS_ATOMICS_ALLOWED       = 1        
} lwblasAtomicsMode_t;

/*For different GEMM algorithm */
typedef enum {
    LWBLAS_GEMM_DFALT               = -1,
    LWBLAS_GEMM_DEFAULT             = -1,
    LWBLAS_GEMM_ALGO0               =  0,
    LWBLAS_GEMM_ALGO1               =  1,
    LWBLAS_GEMM_ALGO2               =  2,
    LWBLAS_GEMM_ALGO3               =  3,
    LWBLAS_GEMM_ALGO4               =  4,
    LWBLAS_GEMM_ALGO5               =  5,
    LWBLAS_GEMM_ALGO6               =  6,
    LWBLAS_GEMM_ALGO7               =  7,
    LWBLAS_GEMM_ALGO8               =  8,
    LWBLAS_GEMM_ALGO9               =  9,
    LWBLAS_GEMM_ALGO10              =  10,   
    LWBLAS_GEMM_ALGO11              =  11,
    LWBLAS_GEMM_ALGO12              =  12,        
    LWBLAS_GEMM_ALGO13              =  13,        
    LWBLAS_GEMM_ALGO14              =  14,        
    LWBLAS_GEMM_ALGO15              =  15,        
    LWBLAS_GEMM_ALGO16              =  16,        
    LWBLAS_GEMM_ALGO17              =  17,        
    LWBLAS_GEMM_DEFAULT_TENSOR_OP   =  99,        
    LWBLAS_GEMM_DFALT_TENSOR_OP     =  99,        
    LWBLAS_GEMM_ALGO0_TENSOR_OP     =  100,        
    LWBLAS_GEMM_ALGO1_TENSOR_OP     =  101,        
    LWBLAS_GEMM_ALGO2_TENSOR_OP     =  102,        
    LWBLAS_GEMM_ALGO3_TENSOR_OP     =  103,        
    LWBLAS_GEMM_ALGO4_TENSOR_OP     =  104        
} lwblasGemmAlgo_t;

/*Enum for default math mode/tensor operation*/
typedef enum {
    LWBLAS_DEFAULT_MATH = 0,
    LWBLAS_TENSOR_OP_MATH = 1
} lwblasMath_t;

/* For backward compatibility purposes */
typedef lwdaDataType lwblasDataType_t;

/* Opaque structure holding LWBLAS library context */
struct lwblasContext;
typedef struct lwblasContext *lwblasHandle_t;

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCreate_v2 (lwblasHandle_t *handle);
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDestroy_v2 (lwblasHandle_t handle);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasGetVersion_v2(lwblasHandle_t handle, int *version);
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasGetProperty(libraryPropertyType type, int *value);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSetStream_v2 (lwblasHandle_t handle, lwdaStream_t streamId); 
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasGetStream_v2 (lwblasHandle_t handle, lwdaStream_t *streamId); 

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasGetPointerMode_v2 (lwblasHandle_t handle, lwblasPointerMode_t *mode);
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSetPointerMode_v2 (lwblasHandle_t handle, lwblasPointerMode_t mode);         

LWBLASAPI lwblasStatus_t  LWBLASWINAPI lwblasGetAtomicsMode(lwblasHandle_t handle, lwblasAtomicsMode_t *mode);
LWBLASAPI lwblasStatus_t  LWBLASWINAPI lwblasSetAtomicsMode(lwblasHandle_t handle, lwblasAtomicsMode_t mode);         

LWBLASAPI lwblasStatus_t  LWBLASWINAPI lwblasGetMathMode(lwblasHandle_t handle, lwblasMath_t *mode);
LWBLASAPI lwblasStatus_t  LWBLASWINAPI lwblasSetMathMode(lwblasHandle_t handle, lwblasMath_t mode);         

/* 
 * lwblasStatus_t 
 * lwblasSetVector (int n, int elemSize, const void *x, int incx, 
 *                  void *y, int incy) 
 *
 * copies n elements from a vector x in CPU memory space to a vector y 
 * in GPU memory space. Elements in both vectors are assumed to have a 
 * size of elemSize bytes. Storage spacing between conselwtive elements
 * is incx for the source vector x and incy for the destination vector
 * y. In general, y points to an object, or part of an object, allocated
 * via lwblasAlloc(). Column major format for two-dimensional matrices
 * is assumed throughout LWBLAS. Therefore, if the increment for a vector 
 * is equal to 1, this access a column vector while using an increment 
 * equal to the leading dimension of the respective matrix accesses a 
 * row vector.
 *
 * Return Values
 * -------------
 * LWBLAS_STATUS_NOT_INITIALIZED  if LWBLAS library not been initialized
 * LWBLAS_STATUS_ILWALID_VALUE    if incx, incy, or elemSize <= 0
 * LWBLAS_STATUS_MAPPING_ERROR    if an error oclwrred accessing GPU memory   
 * LWBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
lwblasStatus_t LWBLASWINAPI lwblasSetVector (int n, int elemSize, const void *x, 
                                             int incx, void *devicePtr, int incy);

/* 
 * lwblasStatus_t 
 * lwblasGetVector (int n, int elemSize, const void *x, int incx, 
 *                  void *y, int incy)
 * 
 * copies n elements from a vector x in GPU memory space to a vector y 
 * in CPU memory space. Elements in both vectors are assumed to have a 
 * size of elemSize bytes. Storage spacing between conselwtive elements
 * is incx for the source vector x and incy for the destination vector
 * y. In general, x points to an object, or part of an object, allocated
 * via lwblasAlloc(). Column major format for two-dimensional matrices
 * is assumed throughout LWBLAS. Therefore, if the increment for a vector 
 * is equal to 1, this access a column vector while using an increment 
 * equal to the leading dimension of the respective matrix accesses a 
 * row vector.
 *
 * Return Values
 * -------------
 * LWBLAS_STATUS_NOT_INITIALIZED  if LWBLAS library not been initialized
 * LWBLAS_STATUS_ILWALID_VALUE    if incx, incy, or elemSize <= 0
 * LWBLAS_STATUS_MAPPING_ERROR    if an error oclwrred accessing GPU memory   
 * LWBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
lwblasStatus_t LWBLASWINAPI lwblasGetVector (int n, int elemSize, const void *x, 
                                             int incx, void *y, int incy);

/*
 * lwblasStatus_t 
 * lwblasSetMatrix (int rows, int cols, int elemSize, const void *A, 
 *                  int lda, void *B, int ldb)
 *
 * copies a tile of rows x cols elements from a matrix A in CPU memory
 * space to a matrix B in GPU memory space. Each element requires storage
 * of elemSize bytes. Both matrices are assumed to be stored in column 
 * major format, with the leading dimension (i.e. number of rows) of 
 * source matrix A provided in lda, and the leading dimension of matrix B
 * provided in ldb. In general, B points to an object, or part of an 
 * object, that was allocated via lwblasAlloc().
 *
 * Return Values 
 * -------------
 * LWBLAS_STATUS_NOT_INITIALIZED  if LWBLAS library has not been initialized
 * LWBLAS_STATUS_ILWALID_VALUE    if rows or cols < 0, or elemSize, lda, or 
 *                                ldb <= 0
 * LWBLAS_STATUS_MAPPING_ERROR    if error oclwrred accessing GPU memory
 * LWBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
lwblasStatus_t LWBLASWINAPI lwblasSetMatrix (int rows, int cols, int elemSize, 
                                             const void *A, int lda, void *B, 
                                             int ldb);

/*
 * lwblasStatus_t 
 * lwblasGetMatrix (int rows, int cols, int elemSize, const void *A, 
 *                  int lda, void *B, int ldb)
 *
 * copies a tile of rows x cols elements from a matrix A in GPU memory
 * space to a matrix B in CPU memory space. Each element requires storage
 * of elemSize bytes. Both matrices are assumed to be stored in column 
 * major format, with the leading dimension (i.e. number of rows) of 
 * source matrix A provided in lda, and the leading dimension of matrix B
 * provided in ldb. In general, A points to an object, or part of an 
 * object, that was allocated via lwblasAlloc().
 *
 * Return Values 
 * -------------
 * LWBLAS_STATUS_NOT_INITIALIZED  if LWBLAS library has not been initialized
 * LWBLAS_STATUS_ILWALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
 * LWBLAS_STATUS_MAPPING_ERROR    if error oclwrred accessing GPU memory
 * LWBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
lwblasStatus_t LWBLASWINAPI lwblasGetMatrix (int rows, int cols, int elemSize, 
                                             const void *A, int lda, void *B,
                                             int ldb);

/* 
 * lwblasStatus 
 * lwblasSetVectorAsync ( int n, int elemSize, const void *x, int incx, 
 *                       void *y, int incy, lwdaStream_t stream );
 *
 * lwblasSetVectorAsync has the same functionnality as lwblasSetVector
 * but the transfer is done asynchronously within the LWCA stream passed
 * in parameter.
 *
 * Return Values
 * -------------
 * LWBLAS_STATUS_NOT_INITIALIZED  if LWBLAS library not been initialized
 * LWBLAS_STATUS_ILWALID_VALUE    if incx, incy, or elemSize <= 0
 * LWBLAS_STATUS_MAPPING_ERROR    if an error oclwrred accessing GPU memory   
 * LWBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
lwblasStatus_t LWBLASWINAPI lwblasSetVectorAsync (int n, int elemSize, 
                                                  const void *hostPtr, int incx, 
                                                  void *devicePtr, int incy,
                                                  lwdaStream_t stream);
/* 
 * lwblasStatus 
 * lwblasGetVectorAsync( int n, int elemSize, const void *x, int incx, 
 *                       void *y, int incy, lwdaStream_t stream)
 * 
 * lwblasGetVectorAsync has the same functionnality as lwblasGetVector
 * but the transfer is done asynchronously within the LWCA stream passed
 * in parameter.
 *
 * Return Values
 * -------------
 * LWBLAS_STATUS_NOT_INITIALIZED  if LWBLAS library not been initialized
 * LWBLAS_STATUS_ILWALID_VALUE    if incx, incy, or elemSize <= 0
 * LWBLAS_STATUS_MAPPING_ERROR    if an error oclwrred accessing GPU memory   
 * LWBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
lwblasStatus_t LWBLASWINAPI lwblasGetVectorAsync (int n, int elemSize,
                                                  const void *devicePtr, int incx,
                                                  void *hostPtr, int incy,
                                                  lwdaStream_t stream);

/*
 * lwblasStatus_t 
 * lwblasSetMatrixAsync (int rows, int cols, int elemSize, const void *A, 
 *                       int lda, void *B, int ldb, lwdaStream_t stream)
 *
 * lwblasSetMatrixAsync has the same functionnality as lwblasSetMatrix
 * but the transfer is done asynchronously within the LWCA stream passed
 * in parameter.
 *
 * Return Values 
 * -------------
 * LWBLAS_STATUS_NOT_INITIALIZED  if LWBLAS library has not been initialized
 * LWBLAS_STATUS_ILWALID_VALUE    if rows or cols < 0, or elemSize, lda, or 
 *                                ldb <= 0
 * LWBLAS_STATUS_MAPPING_ERROR    if error oclwrred accessing GPU memory
 * LWBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
lwblasStatus_t LWBLASWINAPI lwblasSetMatrixAsync (int rows, int cols, int elemSize,
                                                  const void *A, int lda, void *B,
                                                  int ldb, lwdaStream_t stream);

/*
 * lwblasStatus_t 
 * lwblasGetMatrixAsync (int rows, int cols, int elemSize, const void *A, 
 *                       int lda, void *B, int ldb, lwdaStream_t stream)
 *
 * lwblasGetMatrixAsync has the same functionnality as lwblasGetMatrix
 * but the transfer is done asynchronously within the LWCA stream passed
 * in parameter.
 *
 * Return Values 
 * -------------
 * LWBLAS_STATUS_NOT_INITIALIZED  if LWBLAS library has not been initialized
 * LWBLAS_STATUS_ILWALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
 * LWBLAS_STATUS_MAPPING_ERROR    if error oclwrred accessing GPU memory
 * LWBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
lwblasStatus_t LWBLASWINAPI lwblasGetMatrixAsync (int rows, int cols, int elemSize,
                                                  const void *A, int lda, void *B,
                                                  int ldb, lwdaStream_t stream);


LWBLASAPI void LWBLASWINAPI lwblasXerbla (const char *srName, int info);
/* ---------------- LWBLAS BLAS1 functions ---------------- */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasNrm2Ex(lwblasHandle_t handle, 
                                                     int n, 
                                                     const void *x, 
                                                     lwdaDataType xType,
                                                     int incx, 
                                                     void *result,
                                                     lwdaDataType resultType,
                                                     lwdaDataType exelwtionType); /* host or device pointer */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSnrm2_v2(lwblasHandle_t handle, 
                                                     int n, 
                                                     const float *x, 
                                                     int incx, 
                                                     float *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDnrm2_v2(lwblasHandle_t handle, 
                                                     int n, 
                                                     const double *x, 
                                                     int incx, 
                                                     double *result);  /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasScnrm2_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const lwComplex *x, 
                                                      int incx, 
                                                      float *result);  /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDznrm2_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const lwDoubleComplex *x, 
                                                      int incx, 
                                                      double *result);  /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDotEx (lwblasHandle_t handle,
                                                     int n, 
                                                     const void *x,
                                                     lwdaDataType xType, 
                                                     int incx, 
                                                     const void *y, 
                                                     lwdaDataType yType,
                                                     int incy,
                                                     void *result,
                                                     lwdaDataType resultType,
                                                     lwdaDataType exelwtionType);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDotcEx (lwblasHandle_t handle,
                                                     int n, 
                                                     const void *x,
                                                     lwdaDataType xType, 
                                                     int incx, 
                                                     const void *y, 
                                                     lwdaDataType yType,
                                                     int incy,
                                                     void *result,
                                                     lwdaDataType resultType,
                                                     lwdaDataType exelwtionType);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSdot_v2 (lwblasHandle_t handle,
                                                     int n, 
                                                     const float *x, 
                                                     int incx, 
                                                     const float *y, 
                                                     int incy,
                                                     float *result);  /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDdot_v2 (lwblasHandle_t handle,
                                                     int n, 
                                                     const double *x, 
                                                     int incx, 
                                                     const double *y,
                                                     int incy,
                                                     double *result);  /* host or device pointer */
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCdotu_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      const lwComplex *x, 
                                                      int incx, 
                                                      const lwComplex *y, 
                                                      int incy,
                                                      lwComplex *result);  /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCdotc_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      const lwComplex *x, 
                                                      int incx, 
                                                      const lwComplex *y, 
                                                      int incy,
                                                      lwComplex *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZdotu_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      const lwDoubleComplex *x, 
                                                      int incx, 
                                                      const lwDoubleComplex *y, 
                                                      int incy,
                                                      lwDoubleComplex *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZdotc_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      const lwDoubleComplex *x, 
                                                      int incx,
                                                      const lwDoubleComplex *y, 
                                                      int incy,
                                                      lwDoubleComplex *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasScalEx(lwblasHandle_t handle, 
                                                     int n, 
                                                     const void *alpha,  /* host or device pointer */
                                                     lwdaDataType alphaType,
                                                     void *x, 
                                                     lwdaDataType xType,
                                                     int incx,
                                                     lwdaDataType exelwtionType);
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSscal_v2(lwblasHandle_t handle, 
                                                     int n, 
                                                     const float *alpha,  /* host or device pointer */
                                                     float *x, 
                                                     int incx);
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDscal_v2(lwblasHandle_t handle, 
                                                     int n, 
                                                     const double *alpha,  /* host or device pointer */
                                                     double *x, 
                                                     int incx);
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCscal_v2(lwblasHandle_t handle, 
                                                     int n, 
                                                     const lwComplex *alpha, /* host or device pointer */
                                                     lwComplex *x, 
                                                     int incx);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCsscal_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const float *alpha, /* host or device pointer */
                                                      lwComplex *x, 
                                                      int incx);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZscal_v2(lwblasHandle_t handle, 
                                                     int n, 
                                                     const lwDoubleComplex *alpha, /* host or device pointer */
                                                     lwDoubleComplex *x, 
                                                     int incx);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZdscal_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const double *alpha, /* host or device pointer */
                                                      lwDoubleComplex *x, 
                                                      int incx);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasAxpyEx (lwblasHandle_t handle,
                                                      int n,
                                                      const void *alpha, /* host or device pointer */
                                                      lwdaDataType alphaType,
                                                      const void *x,
                                                      lwdaDataType xType,
                                                      int incx,
                                                      void *y,
                                                      lwdaDataType yType,
                                                      int incy,
                                                      lwdaDataType exelwtiontype);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSaxpy_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      const float *alpha, /* host or device pointer */
                                                      const float *x, 
                                                      int incx, 
                                                      float *y, 
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDaxpy_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      const double *alpha, /* host or device pointer */
                                                      const double *x, 
                                                      int incx, 
                                                      double *y, 
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCaxpy_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      const lwComplex *alpha, /* host or device pointer */
                                                      const lwComplex *x, 
                                                      int incx, 
                                                      lwComplex *y, 
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZaxpy_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      const lwDoubleComplex *alpha, /* host or device pointer */
                                                      const lwDoubleComplex *x, 
                                                      int incx, 
                                                      lwDoubleComplex *y, 
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasScopy_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      const float *x, 
                                                      int incx, 
                                                      float *y, 
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDcopy_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      const double *x, 
                                                      int incx, 
                                                      double *y, 
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCcopy_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      const lwComplex *x, 
                                                      int incx, 
                                                      lwComplex *y,
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZcopy_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      const lwDoubleComplex *x, 
                                                      int incx, 
                                                      lwDoubleComplex *y,
                                                      int incy);
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSswap_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      float *x, 
                                                      int incx, 
                                                      float *y, 
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDswap_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      double *x, 
                                                      int incx, 
                                                      double *y, 
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCswap_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      lwComplex *x, 
                                                      int incx, 
                                                      lwComplex *y,
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZswap_v2 (lwblasHandle_t handle,
                                                      int n, 
                                                      lwDoubleComplex *x, 
                                                      int incx, 
                                                      lwDoubleComplex *y,
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasIsamax_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const float *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasIdamax_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const double *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasIcamax_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const lwComplex *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasIzamax_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const lwDoubleComplex *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasIsamin_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const float *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasIdamin_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const double *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasIcamin_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const lwComplex *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasIzamin_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const lwDoubleComplex *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */
 
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSasum_v2(lwblasHandle_t handle, 
                                                     int n, 
                                                     const float *x, 
                                                     int incx, 
                                                     float *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDasum_v2(lwblasHandle_t handle, 
                                                     int n, 
                                                     const double *x, 
                                                     int incx, 
                                                     double *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasScasum_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const lwComplex *x, 
                                                      int incx, 
                                                      float *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDzasum_v2(lwblasHandle_t handle, 
                                                      int n, 
                                                      const lwDoubleComplex *x, 
                                                      int incx, 
                                                      double *result); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSrot_v2 (lwblasHandle_t handle, 
                                                     int n, 
                                                     float *x, 
                                                     int incx, 
                                                     float *y, 
                                                     int incy, 
                                                     const float *c,  /* host or device pointer */
                                                     const float *s); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDrot_v2 (lwblasHandle_t handle, 
                                                     int n, 
                                                     double *x, 
                                                     int incx, 
                                                     double *y, 
                                                     int incy, 
                                                     const double *c,  /* host or device pointer */
                                                     const double *s); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCrot_v2 (lwblasHandle_t handle, 
                                                     int n, 
                                                     lwComplex *x, 
                                                     int incx, 
                                                     lwComplex *y, 
                                                     int incy, 
                                                     const float *c,      /* host or device pointer */
                                                     const lwComplex *s); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCsrot_v2(lwblasHandle_t handle, 
                                                     int n, 
                                                     lwComplex *x, 
                                                     int incx, 
                                                     lwComplex *y, 
                                                     int incy, 
                                                     const float *c,  /* host or device pointer */
                                                     const float *s); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZrot_v2 (lwblasHandle_t handle, 
                                                     int n, 
                                                     lwDoubleComplex *x, 
                                                     int incx, 
                                                     lwDoubleComplex *y, 
                                                     int incy, 
                                                     const double *c,            /* host or device pointer */
                                                     const lwDoubleComplex *s);  /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZdrot_v2(lwblasHandle_t handle, 
                                                     int n, 
                                                     lwDoubleComplex *x, 
                                                     int incx, 
                                                     lwDoubleComplex *y, 
                                                     int incy, 
                                                     const double *c,  /* host or device pointer */
                                                     const double *s); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSrotg_v2(lwblasHandle_t handle, 
                                                     float *a,   /* host or device pointer */
                                                     float *b,   /* host or device pointer */
                                                     float *c,   /* host or device pointer */
                                                     float *s);  /* host or device pointer */
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDrotg_v2(lwblasHandle_t handle, 
                                                     double *a,  /* host or device pointer */
                                                     double *b,  /* host or device pointer */
                                                     double *c,  /* host or device pointer */
                                                     double *s); /* host or device pointer */
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCrotg_v2(lwblasHandle_t handle, 
                                                     lwComplex *a,  /* host or device pointer */
                                                     lwComplex *b,  /* host or device pointer */
                                                     float *c,      /* host or device pointer */
                                                     lwComplex *s); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZrotg_v2(lwblasHandle_t handle, 
                                                     lwDoubleComplex *a,  /* host or device pointer */
                                                     lwDoubleComplex *b,  /* host or device pointer */
                                                     double *c,           /* host or device pointer */
                                                     lwDoubleComplex *s); /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSrotm_v2(lwblasHandle_t handle, 
                                                     int n, 
                                                     float *x, 
                                                     int incx, 
                                                     float *y, 
                                                     int incy, 
                                                     const float* param);  /* host or device pointer */

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDrotm_v2(lwblasHandle_t handle, 
                                                     int n, 
                                                     double *x, 
                                                     int incx, 
                                                     double *y, 
                                                     int incy, 
                                                     const double* param);  /* host or device pointer */
        
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSrotmg_v2(lwblasHandle_t handle, 
                                                      float *d1,        /* host or device pointer */
                                                      float *d2,        /* host or device pointer */
                                                      float *x1,        /* host or device pointer */
                                                      const float *y1,  /* host or device pointer */
                                                      float *param);    /* host or device pointer */
                                         
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDrotmg_v2(lwblasHandle_t handle, 
                                                      double *d1,        /* host or device pointer */  
                                                      double *d2,        /* host or device pointer */  
                                                      double *x1,        /* host or device pointer */  
                                                      const double *y1,  /* host or device pointer */  
                                                      double *param);    /* host or device pointer */  

/* --------------- LWBLAS BLAS2 functions  ---------------- */

/* GEMV */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSgemv_v2 (lwblasHandle_t handle, 
                                                      lwblasOperation_t trans, 
                                                      int m, 
                                                      int n, 
                                                      const float *alpha, /* host or device pointer */
                                                      const float *A, 
                                                      int lda, 
                                                      const float *x, 
                                                      int incx, 
                                                      const float *beta,  /* host or device pointer */
                                                      float *y, 
                                                      int incy);  
 
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDgemv_v2 (lwblasHandle_t handle, 
                                                      lwblasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */ 
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta, /* host or device pointer */
                                                      double *y, 
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgemv_v2 (lwblasHandle_t handle,
                                                      lwblasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      const lwComplex *alpha, /* host or device pointer */ 
                                                      const lwComplex *A,
                                                      int lda,
                                                      const lwComplex *x, 
                                                      int incx,
                                                      const lwComplex *beta, /* host or device pointer */ 
                                                      lwComplex *y,
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZgemv_v2 (lwblasHandle_t handle,
                                                      lwblasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      const lwDoubleComplex *alpha, /* host or device pointer */  
                                                      const lwDoubleComplex *A,
                                                      int lda, 
                                                      const lwDoubleComplex *x, 
                                                      int incx,
                                                      const lwDoubleComplex *beta, /* host or device pointer */  
                                                      lwDoubleComplex *y,
                                                      int incy);
/* GBMV */                                
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSgbmv_v2 (lwblasHandle_t handle, 
                                                      lwblasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku, 
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A, 
                                                      int lda, 
                                                      const float *x,
                                                      int incx,
                                                      const float *beta, /* host or device pointer */  
                                                      float *y,
                                                      int incy);                                
                                
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDgbmv_v2 (lwblasHandle_t handle,
                                                      lwblasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku, 
                                                      const double *alpha, /* host or device pointer */ 
                                                      const double *A,
                                                      int lda, 
                                                      const double *x,
                                                      int incx,
                                                      const double *beta, /* host or device pointer */ 
                                                      double *y,
                                                      int incy);
                                         
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgbmv_v2 (lwblasHandle_t handle,
                                                      lwblasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku, 
                                                      const lwComplex *alpha, /* host or device pointer */ 
                                                      const lwComplex *A,
                                                      int lda, 
                                                      const lwComplex *x,
                                                      int incx,
                                                      const lwComplex *beta, /* host or device pointer */ 
                                                      lwComplex *y,
                                                      int incy);                                             
                                         
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZgbmv_v2 (lwblasHandle_t handle,
                                                      lwblasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku, 
                                                      const lwDoubleComplex *alpha, /* host or device pointer */ 
                                                      const lwDoubleComplex *A,
                                                      int lda, 
                                                      const lwDoubleComplex *x,
                                                      int incx,
                                                      const lwDoubleComplex *beta, /* host or device pointer */ 
                                                      lwDoubleComplex *y,
                                                      int incy);   
                                         
/* TRMV */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasStrmv_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx);                                                 

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDtrmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCtrmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const lwComplex *A, 
                                                      int lda, 
                                                      lwComplex *x, 
                                                      int incx);
                                        
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZtrmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const lwDoubleComplex *A, 
                                                      int lda, 
                                                      lwDoubleComplex *x, 
                                                      int incx);
                                                                                                             
/* TBMV */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasStbmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx);                                                 

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDtbmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCtbmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const lwComplex *A, 
                                                      int lda, 
                                                      lwComplex *x, 
                                                      int incx);
                                               
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZtbmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const lwDoubleComplex *A, 
                                                      int lda, 
                                                      lwDoubleComplex *x, 
                                                      int incx);
                                         
/* TPMV */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasStpmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const float *AP, 
                                                      float *x, 
                                                      int incx);                                                 

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDtpmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const double *AP, 
                                                      double *x, 
                                                      int incx);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCtpmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const lwComplex *AP, 
                                                      lwComplex *x, 
                                                      int incx);
                                                
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZtpmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const lwDoubleComplex *AP, 
                                                      lwDoubleComplex *x, 
                                                      int incx);

/* TRSV */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasStrsv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx);                                                 

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDtrsv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCtrsv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const lwComplex *A, 
                                                      int lda, 
                                                      lwComplex *x, 
                                                      int incx);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZtrsv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const lwDoubleComplex *A, 
                                                      int lda, 
                                                      lwDoubleComplex *x, 
                                                      int incx);

/* TPSV */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasStpsv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const float *AP, 
                                                      float *x, 
                                                      int incx);  
                                                                                                            
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDtpsv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const double *AP, 
                                                      double *x, 
                                                      int incx);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCtpsv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const lwComplex *AP, 
                                                      lwComplex *x, 
                                                      int incx);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZtpsv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      const lwDoubleComplex *AP, 
                                                      lwDoubleComplex *x, 
                                                      int incx);
/* TBSV */                                         
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasStbsv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDtbsv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx);
                                         
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCtbsv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const lwComplex *A, 
                                                      int lda, 
                                                      lwComplex *x, 
                                                      int incx);
                                         
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZtbsv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      lwblasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const lwDoubleComplex *A, 
                                                      int lda, 
                                                      lwDoubleComplex *x, 
                                                      int incx);     
                                         
/* SYMV/HEMV */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSsymv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      int n,
                                                      const float *alpha, /* host or device pointer */ 
                                                      const float *A,
                                                      int lda,
                                                      const float *x,
                                                      int incx,
                                                      const float *beta, /* host or device pointer */ 
                                                      float *y,
                                                      int incy);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDsymv_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo, 
                                                      int n,
                                                      const double *alpha, /* host or device pointer */ 
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta, /* host or device pointer */ 
                                                      double *y,
                                                      int incy);
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCsymv_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo, 
                                                      int n,
                                                      const lwComplex *alpha, /* host or device pointer */ 
                                                      const lwComplex *A,
                                                      int lda,
                                                      const lwComplex *x,
                                                      int incx,
                                                      const lwComplex *beta, /* host or device pointer */ 
                                                      lwComplex *y,
                                                      int incy);                                     
                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZsymv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      int n,
                                                      const lwDoubleComplex *alpha,  /* host or device pointer */ 
                                                      const lwDoubleComplex *A,
                                                      int lda,
                                                      const lwDoubleComplex *x,
                                                      int incx,
                                                      const lwDoubleComplex *beta,   /* host or device pointer */ 
                                                      lwDoubleComplex *y,
                                                      int incy);                                            
                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasChemv_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo, 
                                                      int n,
                                                      const lwComplex *alpha, /* host or device pointer */ 
                                                      const lwComplex *A,
                                                      int lda,
                                                      const lwComplex *x,
                                                      int incx,
                                                      const lwComplex *beta, /* host or device pointer */ 
                                                      lwComplex *y,
                                                      int incy);                                     
                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZhemv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      int n,
                                                      const lwDoubleComplex *alpha,  /* host or device pointer */ 
                                                      const lwDoubleComplex *A,
                                                      int lda,
                                                      const lwDoubleComplex *x,
                                                      int incx,
                                                      const lwDoubleComplex *beta,   /* host or device pointer */ 
                                                      lwDoubleComplex *y,
                                                      int incy);   
                                     
/* SBMV/HBMV */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSsbmv_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo, 
                                                      int n,
                                                      int k,
                                                      const float *alpha,   /* host or device pointer */ 
                                                      const float *A,
                                                      int lda,
                                                      const float *x, 
                                                      int incx,
                                                      const float *beta,  /* host or device pointer */ 
                                                      float *y,
                                                      int incy);
                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDsbmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo, 
                                                      int n,
                                                      int k,
                                                      const double *alpha,   /* host or device pointer */ 
                                                      const double *A,
                                                      int lda,
                                                      const double *x, 
                                                      int incx,
                                                      const double *beta,   /* host or device pointer */ 
                                                      double *y,
                                                      int incy);
                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasChbmv_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo, 
                                                      int n,
                                                      int k,
                                                      const lwComplex *alpha, /* host or device pointer */ 
                                                      const lwComplex *A,
                                                      int lda,
                                                      const lwComplex *x, 
                                                      int incx,
                                                      const lwComplex *beta, /* host or device pointer */ 
                                                      lwComplex *y,
                                                      int incy);
                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZhbmv_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo, 
                                                      int n,
                                                      int k,
                                                      const lwDoubleComplex *alpha, /* host or device pointer */  
                                                      const lwDoubleComplex *A,
                                                      int lda,
                                                      const lwDoubleComplex *x, 
                                                      int incx,
                                                      const lwDoubleComplex *beta, /* host or device pointer */ 
                                                      lwDoubleComplex *y,
                                                      int incy);                                                                            
                                                                                                                                                   
/* SPMV/HPMV */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSspmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo,
                                                      int n, 
                                                      const float *alpha,  /* host or device pointer */                                           
                                                      const float *AP,
                                                      const float *x,
                                                      int incx,
                                                      const float *beta,   /* host or device pointer */  
                                                      float *y,
                                                      int incy);
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDspmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *AP,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta,  /* host or device pointer */  
                                                      double *y,
                                                      int incy);                                     
                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasChpmv_v2 (lwblasHandle_t handle, 
                                                      lwblasFillMode_t uplo,
                                                      int n,
                                                      const lwComplex *alpha, /* host or device pointer */  
                                                      const lwComplex *AP,
                                                      const lwComplex *x,
                                                      int incx,
                                                      const lwComplex *beta, /* host or device pointer */  
                                                      lwComplex *y,
                                                      int incy);
                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZhpmv_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      int n,
                                                      const lwDoubleComplex *alpha, /* host or device pointer */  
                                                      const lwDoubleComplex *AP,
                                                      const lwDoubleComplex *x,
                                                      int incx,
                                                      const lwDoubleComplex *beta, /* host or device pointer */  
                                                      lwDoubleComplex *y, 
                                                      int incy);

/* GER */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSger_v2 (lwblasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const float *x,
                                                     int incx,
                                                     const float *y,
                                                     int incy,
                                                     float *A,
                                                     int lda);
                                    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDger_v2 (lwblasHandle_t handle, 
                                                     int m,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */   
                                                     const double *x,
                                                     int incx,
                                                     const double *y,
                                                     int incy,
                                                     double *A,
                                                     int lda);
                                    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgeru_v2 (lwblasHandle_t handle, 
                                                      int m,
                                                      int n,
                                                      const lwComplex *alpha, /* host or device pointer */  
                                                      const lwComplex *x,
                                                      int incx,
                                                      const lwComplex *y,
                                                      int incy,
                                                      lwComplex *A,
                                                      int lda);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgerc_v2 (lwblasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const lwComplex *alpha, /* host or device pointer */  
                                                      const lwComplex *x,
                                                      int incx,
                                                      const lwComplex *y,
                                                      int incy,
                                                      lwComplex *A,
                                                      int lda);                                   

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZgeru_v2 (lwblasHandle_t handle, 
                                                      int m,
                                                      int n,
                                                      const lwDoubleComplex *alpha, /* host or device pointer */  
                                                      const lwDoubleComplex *x,
                                                      int incx,
                                                      const lwDoubleComplex *y,
                                                      int incy,
                                                      lwDoubleComplex *A,
                                                      int lda);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZgerc_v2 (lwblasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const lwDoubleComplex *alpha, /* host or device pointer */  
                                                      const lwDoubleComplex *x,
                                                      int incx,
                                                      const lwDoubleComplex *y,
                                                      int incy,
                                                      lwDoubleComplex *A,
                                                      int lda); 
                                    
/* SYR/HER */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSsyr_v2 (lwblasHandle_t handle,
                                                     lwblasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const float *x,
                                                     int incx,
                                                     float *A, 
                                                     int lda);
                                    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDsyr_v2 (lwblasHandle_t handle,
                                                     lwblasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */  
                                                     const double *x,
                                                     int incx,
                                                     double *A, 
                                                     int lda);  
                                        
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCsyr_v2 (lwblasHandle_t handle,
                                                     lwblasFillMode_t uplo,
                                                     int n,
                                                     const lwComplex *alpha, /* host or device pointer */  
                                                     const lwComplex *x,
                                                     int incx,
                                                     lwComplex *A, 
                                                     int lda);
                                    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZsyr_v2 (lwblasHandle_t handle,
                                                     lwblasFillMode_t uplo,
                                                     int n,
                                                     const lwDoubleComplex *alpha, /* host or device pointer */  
                                                     const lwDoubleComplex *x,
                                                     int incx,
                                                     lwDoubleComplex *A, 
                                                     int lda);                                          
                                                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCher_v2 (lwblasHandle_t handle,
                                                     lwblasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const lwComplex *x,
                                                     int incx,
                                                     lwComplex *A, 
                                                     int lda); 
                                    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZher_v2 (lwblasHandle_t handle,
                                                     lwblasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */  
                                                     const lwDoubleComplex *x,
                                                     int incx,
                                                     lwDoubleComplex *A, 
                                                     int lda); 

/* SPR/HPR */                                    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSspr_v2 (lwblasHandle_t handle,
                                                     lwblasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const float *x,
                                                     int incx,
                                                     float *AP);
                                    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDspr_v2 (lwblasHandle_t handle,
                                                     lwblasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */  
                                                     const double *x,
                                                     int incx,
                                                     double *AP);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasChpr_v2 (lwblasHandle_t handle,
                                                     lwblasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const lwComplex *x,
                                                     int incx,
                                                     lwComplex *AP);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZhpr_v2 (lwblasHandle_t handle,
                                                     lwblasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */  
                                                     const lwDoubleComplex *x,
                                                     int incx,
                                                     lwDoubleComplex *AP);                       
    
/* SYR2/HER2 */                                    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSsyr2_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      int n, 
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *x,
                                                      int incx,
                                                      const float *y,
                                                      int incy,
                                                      float *A,
                                                      int lda);
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDsyr2_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      int n, 
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *x,
                                                      int incx,
                                                      const double *y,
                                                      int incy,
                                                      double *A,
                                                      int lda);
                                         
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCsyr2_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo, int n, 
                                                      const lwComplex *alpha,  /* host or device pointer */  
                                                      const lwComplex *x,
                                                      int incx, 
                                                      const lwComplex *y,
                                                      int incy, 
                                                      lwComplex *A, 
                                                      int lda);   
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZsyr2_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      int n, 
                                                      const lwDoubleComplex *alpha,  /* host or device pointer */  
                                                      const lwDoubleComplex *x,
                                                      int incx,
                                                      const lwDoubleComplex *y,
                                                      int incy,
                                                      lwDoubleComplex *A,
                                                      int lda);                       
    

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCher2_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo, int n, 
                                                      const lwComplex *alpha,  /* host or device pointer */  
                                                      const lwComplex *x,
                                                      int incx, 
                                                      const lwComplex *y,
                                                      int incy, 
                                                      lwComplex *A, 
                                                      int lda);   

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZher2_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      int n, 
                                                      const lwDoubleComplex *alpha,  /* host or device pointer */  
                                                      const lwDoubleComplex *x,
                                                      int incx,
                                                      const lwDoubleComplex *y,
                                                      int incy,
                                                      lwDoubleComplex *A,
                                                      int lda);                       

/* SPR2/HPR2 */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSspr2_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      int n,
                                                      const float *alpha,  /* host or device pointer */  
                                                      const float *x,
                                                      int incx,
                                                      const float *y,
                                                      int incy,
                                                      float *AP);
                                                                          
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDspr2_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha,  /* host or device pointer */  
                                                      const double *x,
                                                      int incx, 
                                                      const double *y,
                                                      int incy,
                                                      double *AP);
    

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasChpr2_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      int n,
                                                      const lwComplex *alpha, /* host or device pointer */  
                                                      const lwComplex *x,
                                                      int incx,
                                                      const lwComplex *y,
                                                      int incy,
                                                      lwComplex *AP);
                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZhpr2_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      int n,
                                                      const lwDoubleComplex *alpha, /* host or device pointer */  
                                                      const lwDoubleComplex *x,
                                                      int incx,
                                                      const lwDoubleComplex *y,
                                                      int incy,
                                                      lwDoubleComplex *AP); 

/* ---------------- LWBLAS BLAS3 functions ---------------- */

/* GEMM */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSgemm_v2 (lwblasHandle_t handle, 
                                                      lwblasOperation_t transa,
                                                      lwblasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A, 
                                                      int lda,
                                                      const float *B,
                                                      int ldb, 
                                                      const float *beta, /* host or device pointer */  
                                                      float *C,
                                                      int ldc);
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDgemm_v2 (lwblasHandle_t handle, 
                                                      lwblasOperation_t transa,
                                                      lwblasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A, 
                                                      int lda,
                                                      const double *B,
                                                      int ldb, 
                                                      const double *beta, /* host or device pointer */  
                                                      double *C,
                                                      int ldc);
                                        
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgemm_v2 (lwblasHandle_t handle, 
                                                      lwblasOperation_t transa,
                                                      lwblasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const lwComplex *alpha, /* host or device pointer */  
                                                      const lwComplex *A, 
                                                      int lda,
                                                      const lwComplex *B,
                                                      int ldb, 
                                                      const lwComplex *beta, /* host or device pointer */  
                                                      lwComplex *C,
                                                      int ldc);
                                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgemm3m  (lwblasHandle_t handle, 
                                                      lwblasOperation_t transa,
                                                      lwblasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const lwComplex *alpha, /* host or device pointer */  
                                                      const lwComplex *A, 
                                                      int lda,
                                                      const lwComplex *B,
                                                      int ldb, 
                                                      const lwComplex *beta, /* host or device pointer */  
                                                      lwComplex *C,
                                                      int ldc);                                                      
 LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgemm3mEx (lwblasHandle_t handle, 
                                                     lwblasOperation_t transa, lwblasOperation_t transb,  
                                                     int m, int n, int k, 
                                                     const lwComplex *alpha, 
                                                     const void *A, 
                                                     lwdaDataType Atype, 
                                                     int lda, 
                                                     const void *B, 
                                                     lwdaDataType Btype, 
                                                     int ldb,
                                                     const lwComplex *beta, 
                                                     void *C, 
                                                     lwdaDataType Ctype, 
                                                     int ldc);
                                       

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZgemm_v2 (lwblasHandle_t handle, 
                                                      lwblasOperation_t transa,
                                                      lwblasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const lwDoubleComplex *alpha, /* host or device pointer */  
                                                      const lwDoubleComplex *A, 
                                                      int lda,
                                                      const lwDoubleComplex *B,
                                                      int ldb, 
                                                      const lwDoubleComplex *beta, /* host or device pointer */  
                                                      lwDoubleComplex *C,
                                                      int ldc);     
                                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZgemm3m  (lwblasHandle_t handle, 
                                                      lwblasOperation_t transa,
                                                      lwblasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const lwDoubleComplex *alpha, /* host or device pointer */  
                                                      const lwDoubleComplex *A, 
                                                      int lda,
                                                      const lwDoubleComplex *B,
                                                      int ldb, 
                                                      const lwDoubleComplex *beta, /* host or device pointer */  
                                                      lwDoubleComplex *C,
                                                      int ldc);                                                                   
                                                      
#if defined(__cplusplus)
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasHgemm    (lwblasHandle_t handle, 
                                                      lwblasOperation_t transa,
                                                      lwblasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const __half *alpha, /* host or device pointer */  
                                                      const __half *A, 
                                                      int lda,
                                                      const __half *B,
                                                      int ldb, 
                                                      const __half *beta, /* host or device pointer */  
                                                      __half *C,
                                                      int ldc);             
#endif
/* IO in FP16/FP32, computation in float */                                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSgemmEx  (lwblasHandle_t handle, 
                                                      lwblasOperation_t transa,
                                                      lwblasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const float *alpha, /* host or device pointer */  
                                                      const void *A, 
                                                      lwdaDataType Atype,
                                                      int lda,
                                                      const void *B,
                                                      lwdaDataType Btype,
                                                      int ldb, 
                                                      const float *beta, /* host or device pointer */  
                                                      void *C,
                                                      lwdaDataType Ctype,
                                                      int ldc); 
                                       
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasGemmEx  (lwblasHandle_t handle, 
                                                      lwblasOperation_t transa,
                                                      lwblasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const void *alpha, /* host or device pointer */  
                                                      const void *A, 
                                                      lwdaDataType Atype,
                                                      int lda,
                                                      const void *B,
                                                      lwdaDataType Btype,
                                                      int ldb, 
                                                      const void *beta, /* host or device pointer */  
                                                      void *C,
                                                      lwdaDataType Ctype,
                                                      int ldc,
                                                      lwdaDataType computeType,
                                                      lwblasGemmAlgo_t algo); 
 
/* IO in Int8 complex/lwComplex, computation in lwComplex */                                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgemmEx (lwblasHandle_t handle, 
                                                     lwblasOperation_t transa, lwblasOperation_t transb,  
                                                     int m, int n, int k, 
                                                     const lwComplex *alpha, 
                                                     const void *A, 
                                                     lwdaDataType Atype, 
                                                     int lda, 
                                                     const void *B, 
                                                     lwdaDataType Btype, 
                                                     int ldb,
                                                     const lwComplex *beta, 
                                                     void *C, 
                                                     lwdaDataType Ctype, 
                                                     int ldc);
                                                                                                                                                                                                                                                                                                   
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasUint8gemmBias (lwblasHandle_t handle, 
                                                           lwblasOperation_t transa, lwblasOperation_t transb, lwblasOperation_t transc,  
                                                           int m, int n, int k, 
                                                           const unsigned char *A, int A_bias, int lda, 
                                                           const unsigned char *B, int B_bias, int ldb,
                                                                 unsigned char *C, int C_bias, int ldc,
                                                           int C_mult, int C_shift);
                                                                                       
/* SYRK */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSsyrk_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      lwblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A,
                                                      int lda,
                                                      const float *beta, /* host or device pointer */  
                                                      float *C,
                                                      int ldc);
                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDsyrk_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      lwblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const double *alpha,  /* host or device pointer */  
                                                      const double *A,
                                                      int lda,
                                                      const double *beta,  /* host or device pointer */  
                                                      double *C,
                                                      int ldc);   
                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCsyrk_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      lwblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const lwComplex *alpha, /* host or device pointer */  
                                                      const lwComplex *A,
                                                      int lda,
                                                      const lwComplex *beta, /* host or device pointer */  
                                                      lwComplex *C,
                                                      int ldc);         
                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZsyrk_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      lwblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const lwDoubleComplex *alpha, /* host or device pointer */  
                                                      const lwDoubleComplex *A,
                                                      int lda,
                                                      const lwDoubleComplex *beta, /* host or device pointer */  
                                                      lwDoubleComplex *C, 
                                                      int ldc);
/* IO in Int8 complex/lwComplex, computation in lwComplex */  
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCsyrkEx ( lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      lwblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const lwComplex *alpha, /* host or device pointer */  
                                                      const void *A, 
                                                      lwdaDataType Atype, 
                                                      int lda,
                                                      const lwComplex *beta, /* host or device pointer */  
                                                      void *C, 
                                                      lwdaDataType Ctype, 
                                                      int ldc);  
                                                      
/* IO in Int8 complex/lwComplex, computation in lwComplex, Gaussian math */                                                          
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCsyrk3mEx(lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo, 
                                                      lwblasOperation_t trans, 
                                                      int n, 
                                                      int k,
                                                      const lwComplex *alpha, 
                                                      const void *A, 
                                                      lwdaDataType Atype, 
                                                      int lda,
                                                      const lwComplex *beta, 
                                                      void *C, 
                                                      lwdaDataType Ctype, 
                                                      int ldc);
                                                      
/* HERK */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCherk_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      lwblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha,  /* host or device pointer */  
                                                      const lwComplex *A,
                                                      int lda,
                                                      const float *beta,   /* host or device pointer */  
                                                      lwComplex *C,
                                                      int ldc);
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZherk_v2 (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      lwblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const double *alpha,  /* host or device pointer */  
                                                      const lwDoubleComplex *A,
                                                      int lda,
                                                      const double *beta,  /* host or device pointer */  
                                                      lwDoubleComplex *C,
                                                      int ldc);  
                                                        
/* IO in Int8 complex/lwComplex, computation in lwComplex */                                                       
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCherkEx  (lwblasHandle_t handle,
                                                      lwblasFillMode_t uplo,
                                                      lwblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha,  /* host or device pointer */  
                                                      const void *A, 
                                                      lwdaDataType Atype,
                                                      int lda,
                                                      const float *beta,   /* host or device pointer */  
                                                      void *C,
                                                      lwdaDataType Ctype,
                                                      int ldc);
                                                      
/* IO in Int8 complex/lwComplex, computation in lwComplex, Gaussian math */                                                          
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCherk3mEx (lwblasHandle_t handle,
                                                       lwblasFillMode_t uplo, 
                                                       lwblasOperation_t trans, 
                                                       int n, 
                                                       int k,
                                                       const float *alpha, 
                                                       const void *A, lwdaDataType Atype, 
                                                       int lda,
                                                       const float *beta, 
                                                       void *C, 
                                                       lwdaDataType Ctype, 
                                                       int ldc);
                                                       
                                                       
                                                                                                             
/* SYR2K */                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSsyr2k_v2 (lwblasHandle_t handle,
                                                       lwblasFillMode_t uplo,
                                                       lwblasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const float *alpha, /* host or device pointer */  
                                                       const float *A,
                                                       int lda,
                                                       const float *B,
                                                       int ldb,
                                                       const float *beta, /* host or device pointer */  
                                                       float *C,
                                                       int ldc);  
                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDsyr2k_v2 (lwblasHandle_t handle,
                                                       lwblasFillMode_t uplo,
                                                       lwblasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const double *alpha, /* host or device pointer */  
                                                       const double *A,
                                                       int lda,
                                                       const double *B,
                                                       int ldb,
                                                       const double *beta, /* host or device pointer */  
                                                       double *C,
                                                       int ldc);
                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCsyr2k_v2 (lwblasHandle_t handle,
                                                       lwblasFillMode_t uplo,
                                                       lwblasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const lwComplex *alpha, /* host or device pointer */  
                                                       const lwComplex *A,
                                                       int lda,
                                                       const lwComplex *B,
                                                       int ldb,
                                                       const lwComplex *beta, /* host or device pointer */  
                                                       lwComplex *C,
                                                       int ldc);
                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZsyr2k_v2 (lwblasHandle_t handle,
                                                       lwblasFillMode_t uplo,
                                                       lwblasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const lwDoubleComplex *alpha,  /* host or device pointer */  
                                                       const lwDoubleComplex *A,
                                                       int lda,
                                                       const lwDoubleComplex *B,
                                                       int ldb,
                                                       const lwDoubleComplex *beta,  /* host or device pointer */  
                                                       lwDoubleComplex *C,
                                                       int ldc);  
/* HER2K */                                       
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCher2k_v2 (lwblasHandle_t handle,
                                                       lwblasFillMode_t uplo,
                                                       lwblasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const lwComplex *alpha, /* host or device pointer */  
                                                       const lwComplex *A,
                                                       int lda,
                                                       const lwComplex *B,
                                                       int ldb,
                                                       const float *beta,   /* host or device pointer */  
                                                       lwComplex *C,
                                                       int ldc);  
                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZher2k_v2 (lwblasHandle_t handle,
                                                       lwblasFillMode_t uplo,
                                                       lwblasOperation_t trans, 
                                                       int n,
                                                       int k,
                                                       const lwDoubleComplex *alpha, /* host or device pointer */  
                                                       const lwDoubleComplex *A, 
                                                       int lda,
                                                       const lwDoubleComplex *B,
                                                       int ldb,
                                                       const double *beta, /* host or device pointer */  
                                                       lwDoubleComplex *C,
                                                       int ldc);     
/* SYRKX : eXtended SYRK*/
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSsyrkx (lwblasHandle_t handle,
                                                    lwblasFillMode_t uplo,
                                                    lwblasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const float *alpha, /* host or device pointer */ 
                                                    const float *A,
                                                    int lda,
                                                    const float *B,
                                                    int ldb,
                                                    const float *beta, /* host or device pointer */ 
                                                    float *C,
                                                    int ldc);
                                                   
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDsyrkx (lwblasHandle_t handle,
                                                    lwblasFillMode_t uplo,
                                                    lwblasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const double *alpha, /* host or device pointer */ 
                                                    const double *A,
                                                    int lda,
                                                    const double *B,
                                                    int ldb,
                                                    const double *beta, /* host or device pointer */ 
                                                    double *C,
                                                    int ldc);
                                                    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCsyrkx (lwblasHandle_t handle,
                                                    lwblasFillMode_t uplo,
                                                    lwblasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const lwComplex *alpha, /* host or device pointer */ 
                                                    const lwComplex *A,
                                                    int lda,
                                                    const lwComplex *B,
                                                    int ldb,
                                                    const lwComplex *beta, /* host or device pointer */ 
                                                    lwComplex *C, 
                                                    int ldc);
                                                    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZsyrkx (lwblasHandle_t handle,
                                                    lwblasFillMode_t uplo, 
                                                    lwblasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const lwDoubleComplex *alpha, /* host or device pointer */ 
                                                    const lwDoubleComplex *A,
                                                    int lda,
                                                    const lwDoubleComplex *B,
                                                    int ldb,
                                                    const lwDoubleComplex *beta, /* host or device pointer */ 
                                                    lwDoubleComplex *C, 
                                                    int ldc);
/* HERKX : eXtended HERK */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCherkx (lwblasHandle_t handle,
                                                    lwblasFillMode_t uplo,
                                                    lwblasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const lwComplex *alpha, /* host or device pointer */ 
                                                    const lwComplex *A,
                                                    int lda,
                                                    const lwComplex *B,
                                                    int ldb,
                                                    const float *beta, /* host or device pointer */ 
                                                    lwComplex *C,
                                                    int ldc);
                                                
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZherkx (lwblasHandle_t handle,
                                                    lwblasFillMode_t uplo,
                                                    lwblasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const lwDoubleComplex *alpha, /* host or device pointer */ 
                                                    const lwDoubleComplex *A,
                                                    int lda,
                                                    const lwDoubleComplex *B,
                                                    int ldb,
                                                    const double *beta, /* host or device pointer */ 
                                                    lwDoubleComplex *C,
                                                    int ldc);
/* SYMM */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSsymm_v2 (lwblasHandle_t handle,
                                                      lwblasSideMode_t side,
                                                      lwblasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A,
                                                      int lda,
                                                      const float *B,
                                                      int ldb,
                                                      const float *beta, /* host or device pointer */  
                                                      float *C,
                                                      int ldc);
                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDsymm_v2 (lwblasHandle_t handle,
                                                      lwblasSideMode_t side,
                                                      lwblasFillMode_t uplo,
                                                      int m, 
                                                      int n,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A,
                                                      int lda,
                                                      const double *B,
                                                      int ldb,
                                                      const double *beta, /* host or device pointer */  
                                                      double *C,
                                                      int ldc);                                     

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCsymm_v2 (lwblasHandle_t handle,
                                                      lwblasSideMode_t side,
                                                      lwblasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const lwComplex *alpha, /* host or device pointer */  
                                                      const lwComplex *A,
                                                      int lda,
                                                      const lwComplex *B,
                                                      int ldb,
                                                      const lwComplex *beta, /* host or device pointer */  
                                                      lwComplex *C,
                                                      int ldc);
                                                   
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZsymm_v2 (lwblasHandle_t handle,
                                                      lwblasSideMode_t side,
                                                      lwblasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const lwDoubleComplex *alpha, /* host or device pointer */  
                                                      const lwDoubleComplex *A,
                                                      int lda,
                                                      const lwDoubleComplex *B,
                                                      int ldb,
                                                      const lwDoubleComplex *beta, /* host or device pointer */  
                                                      lwDoubleComplex *C,
                                                      int ldc);   
                                     
/* HEMM */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasChemm_v2 (lwblasHandle_t handle,
                                                      lwblasSideMode_t side,
                                                      lwblasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const lwComplex *alpha, /* host or device pointer */  
                                                      const lwComplex *A,
                                                      int lda,
                                                      const lwComplex *B,
                                                      int ldb,
                                                      const lwComplex *beta, /* host or device pointer */  
                                                      lwComplex *C, 
                                                      int ldc); 

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZhemm_v2 (lwblasHandle_t handle,
                                                      lwblasSideMode_t side,
                                                      lwblasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const lwDoubleComplex *alpha, /* host or device pointer */  
                                                      const lwDoubleComplex *A,
                                                      int lda,
                                                      const lwDoubleComplex *B,
                                                      int ldb,
                                                      const lwDoubleComplex *beta, /* host or device pointer */  
                                                      lwDoubleComplex *C,
                                                      int ldc); 
    
/* TRSM */                                                                         
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasStrsm_v2 (lwblasHandle_t handle, 
                                                      lwblasSideMode_t side,
                                                      lwblasFillMode_t uplo,
                                                      lwblasOperation_t trans,
                                                      lwblasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A,
                                                      int lda,
                                                      float *B,
                                                      int ldb);
    

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDtrsm_v2 (lwblasHandle_t handle,
                                                      lwblasSideMode_t side,
                                                      lwblasFillMode_t uplo,
                                                      lwblasOperation_t trans,
                                                      lwblasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A, 
                                                      int lda, 
                                                      double *B,
                                                      int ldb);
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCtrsm_v2(lwblasHandle_t handle,
                                                     lwblasSideMode_t side,
                                                     lwblasFillMode_t uplo,
                                                     lwblasOperation_t trans,
                                                     lwblasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const lwComplex *alpha, /* host or device pointer */  
                                                     const lwComplex *A,
                                                     int lda,
                                                     lwComplex *B,
                                                     int ldb);
                  
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZtrsm_v2(lwblasHandle_t handle, 
                                                     lwblasSideMode_t side,
                                                     lwblasFillMode_t uplo,
                                                     lwblasOperation_t trans,
                                                     lwblasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const lwDoubleComplex *alpha, /* host or device pointer */  
                                                     const lwDoubleComplex *A,                                        
                                                     int lda,
                                                     lwDoubleComplex *B,
                                                     int ldb);              
                                                
 /* TRMM */  
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasStrmm_v2 (lwblasHandle_t handle,
                                                      lwblasSideMode_t side,
                                                      lwblasFillMode_t uplo,
                                                      lwblasOperation_t trans,
                                                      lwblasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A,
                                                      int lda, 
                                                      const float *B,
                                                      int ldb,
                                                      float *C,
                                                      int ldc);
                                               
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDtrmm_v2 (lwblasHandle_t handle,
                                                      lwblasSideMode_t side,
                                                      lwblasFillMode_t uplo,
                                                      lwblasOperation_t trans,
                                                      lwblasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A,
                                                      int lda,
                                                      const double *B,
                                                      int ldb,
                                                      double *C,
                                                      int ldc);
                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCtrmm_v2(lwblasHandle_t handle,
                                                     lwblasSideMode_t side,
                                                     lwblasFillMode_t uplo,
                                                     lwblasOperation_t trans,
                                                     lwblasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const lwComplex *alpha, /* host or device pointer */  
                                                     const lwComplex *A,
                                                     int lda,
                                                     const lwComplex *B,
                                                     int ldb,
                                                     lwComplex *C,
                                                     int ldc);
                  
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZtrmm_v2(lwblasHandle_t handle, lwblasSideMode_t side, 
                                                     lwblasFillMode_t uplo,
                                                     lwblasOperation_t trans,
                                                     lwblasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const lwDoubleComplex *alpha, /* host or device pointer */  
                                                     const lwDoubleComplex *A,
                                                     int lda,
                                                     const lwDoubleComplex *B,
                                                     int ldb,
                                                     lwDoubleComplex *C,
                                                     int ldc);
/* BATCH GEMM */
#if defined(__cplusplus)
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasHgemmBatched (lwblasHandle_t handle,
                                                          lwblasOperation_t transa,
                                                          lwblasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const __half *alpha,  /* host or device pointer */  
                                                          const __half *Aarray[], 
                                                          int lda,
                                                          const __half *Barray[],
                                                          int ldb, 
                                                          const __half *beta,   /* host or device pointer */  
                                                          __half *Carray[],
                                                          int ldc,
                                                          int batchCount);
#endif
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSgemmBatched (lwblasHandle_t handle,
                                                          lwblasOperation_t transa,
                                                          lwblasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const float *alpha,  /* host or device pointer */  
                                                          const float *Aarray[], 
                                                          int lda,
                                                          const float *Barray[],
                                                          int ldb, 
                                                          const float *beta,   /* host or device pointer */  
                                                          float *Carray[],
                                                          int ldc,
                                                          int batchCount);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDgemmBatched (lwblasHandle_t handle,
                                                          lwblasOperation_t transa,
                                                          lwblasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const double *alpha,  /* host or device pointer */ 
                                                          const double *Aarray[], 
                                                          int lda,
                                                          const double *Barray[],
                                                          int ldb, 
                                                          const double *beta,  /* host or device pointer */ 
                                                          double *Carray[],
                                                          int ldc,
                                                          int batchCount);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgemmBatched (lwblasHandle_t handle,
                                                          lwblasOperation_t transa,
                                                          lwblasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const lwComplex *alpha, /* host or device pointer */ 
                                                          const lwComplex *Aarray[], 
                                                          int lda,
                                                          const lwComplex *Barray[],
                                                          int ldb, 
                                                          const lwComplex *beta, /* host or device pointer */ 
                                                          lwComplex *Carray[],
                                                          int ldc,
                                                          int batchCount);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgemm3mBatched (lwblasHandle_t handle,
                                                          lwblasOperation_t transa,
                                                          lwblasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const lwComplex *alpha, /* host or device pointer */ 
                                                          const lwComplex *Aarray[], 
                                                          int lda,
                                                          const lwComplex *Barray[],
                                                          int ldb, 
                                                          const lwComplex *beta, /* host or device pointer */ 
                                                          lwComplex *Carray[],
                                                          int ldc,
                                                          int batchCount);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZgemmBatched (lwblasHandle_t handle,
                                                          lwblasOperation_t transa,
                                                          lwblasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const lwDoubleComplex *alpha, /* host or device pointer */ 
                                                          const lwDoubleComplex *Aarray[], 
                                                          int lda,
                                                          const lwDoubleComplex *Barray[],
                                                          int ldb, 
                                                          const lwDoubleComplex *beta, /* host or device pointer */ 
                                                          lwDoubleComplex *Carray[],
                                                          int ldc,
                                                          int batchCount); 

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSgemmStridedBatched (lwblasHandle_t handle,
                                                                 lwblasOperation_t transa,
                                                                 lwblasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const float *alpha,  /* host or device pointer */
                                                                 const float *A,
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const float *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const float *beta,   /* host or device pointer */
                                                                 float *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDgemmStridedBatched (lwblasHandle_t handle,
                                                                 lwblasOperation_t transa,
                                                                 lwblasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const double *alpha,  /* host or device pointer */
                                                                 const double *A, 
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const double *B,
                                                                 int ldb, 
                                                                 long long int strideB,
                                                                 const double *beta,   /* host or device pointer */
                                                                 double *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgemmStridedBatched (lwblasHandle_t handle,
                                                                 lwblasOperation_t transa,
                                                                 lwblasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const lwComplex *alpha,  /* host or device pointer */
                                                                 const lwComplex *A, 
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const lwComplex *B,
                                                                 int ldb, 
                                                                 long long int strideB,
                                                                 const lwComplex *beta,   /* host or device pointer */
                                                                 lwComplex *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgemm3mStridedBatched (lwblasHandle_t handle,
                                                                 lwblasOperation_t transa,
                                                                 lwblasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const lwComplex *alpha,  /* host or device pointer */
                                                                 const lwComplex *A, 
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const lwComplex *B,
                                                                 int ldb, 
                                                                 long long int strideB,
                                                                 const lwComplex *beta,   /* host or device pointer */
                                                                 lwComplex *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount);


LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZgemmStridedBatched (lwblasHandle_t handle,
                                                                 lwblasOperation_t transa,
                                                                 lwblasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const lwDoubleComplex *alpha,  /* host or device pointer */
                                                                 const lwDoubleComplex *A, 
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const lwDoubleComplex *B,
                                                                 int ldb, 
                                                                 long long int strideB,
                                                                 const lwDoubleComplex *beta,   /* host or device poi */
                                                                 lwDoubleComplex *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount);

#if defined(__cplusplus)
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasHgemmStridedBatched (lwblasHandle_t handle,
                                                                 lwblasOperation_t transa,
                                                                 lwblasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const __half *alpha,  /* host or device pointer */
                                                                 const __half *A, 
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const __half *B,
                                                                 int ldb, 
                                                                 long long int strideB,
                                                                 const __half *beta,   /* host or device pointer */
                                                                 __half *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount);
#endif
/* ---------------- LWBLAS BLAS-like extension ---------------- */
/* GEAM */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSgeam(lwblasHandle_t handle,
                                                  lwblasOperation_t transa, 
                                                  lwblasOperation_t transb,
                                                  int m, 
                                                  int n,
                                                  const float *alpha, /* host or device pointer */ 
                                                  const float *A, 
                                                  int lda,
                                                  const float *beta , /* host or device pointer */ 
                                                  const float *B, 
                                                  int ldb,
                                                  float *C, 
                                                  int ldc);
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDgeam(lwblasHandle_t handle,
                                                  lwblasOperation_t transa, 
                                                  lwblasOperation_t transb,
                                                  int m, 
                                                  int n,
                                                  const double *alpha, /* host or device pointer */ 
                                                  const double *A, 
                                                  int lda,
                                                  const double *beta, /* host or device pointer */ 
                                                  const double *B, 
                                                  int ldb,
                                                  double *C, 
                                                  int ldc);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgeam(lwblasHandle_t handle,
                                                  lwblasOperation_t transa, 
                                                  lwblasOperation_t transb,
                                                  int m, 
                                                  int n,
                                                  const lwComplex *alpha, /* host or device pointer */ 
                                                  const lwComplex *A, 
                                                  int lda,
                                                  const lwComplex *beta, /* host or device pointer */  
                                                  const lwComplex *B, 
                                                  int ldb,
                                                  lwComplex *C, 
                                                  int ldc);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZgeam(lwblasHandle_t handle,
                                                  lwblasOperation_t transa, 
                                                  lwblasOperation_t transb,
                                                  int m, 
                                                  int n,
                                                  const lwDoubleComplex *alpha, /* host or device pointer */ 
                                                  const lwDoubleComplex *A, 
                                                  int lda,
                                                  const lwDoubleComplex *beta, /* host or device pointer */  
                                                  const lwDoubleComplex *B, 
                                                  int ldb,
                                                  lwDoubleComplex *C, 
                                                  int ldc);
 
/* Batched LU - GETRF*/
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSgetrfBatched(lwblasHandle_t handle,
                                                  int n, 
                                                  float *A[],                      /*Device pointer*/
                                                  int lda, 
                                                  int *P,                          /*Device Pointer*/
                                                  int *info,                       /*Device Pointer*/
                                                  int batchSize);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDgetrfBatched(lwblasHandle_t handle,
                                                  int n, 
                                                  double *A[],                     /*Device pointer*/
                                                  int lda, 
                                                  int *P,                          /*Device Pointer*/
                                                  int *info,                       /*Device Pointer*/
                                                  int batchSize);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgetrfBatched(lwblasHandle_t handle,
                                                  int n, 
                                                  lwComplex *A[],                 /*Device pointer*/
                                                  int lda, 
                                                  int *P,                         /*Device Pointer*/
                                                  int *info,                      /*Device Pointer*/
                                                  int batchSize);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZgetrfBatched(lwblasHandle_t handle,
                                                  int n, 
                                                  lwDoubleComplex *A[],           /*Device pointer*/
                                                  int lda, 
                                                  int *P,                         /*Device Pointer*/
                                                  int *info,                      /*Device Pointer*/
                                                  int batchSize);

/* Batched ilwersion based on LU factorization from getrf */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSgetriBatched(lwblasHandle_t handle,
                                                  int n,
                                                  const float *A[],               /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  float *C[],                     /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDgetriBatched(lwblasHandle_t handle,
                                                  int n,
                                                  const double *A[],              /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  double *C[],                    /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCgetriBatched(lwblasHandle_t handle,
                                                  int n,
                                                  const lwComplex *A[],            /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  lwComplex *C[],                 /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZgetriBatched(lwblasHandle_t handle,
                                                  int n,
                                                  const lwDoubleComplex *A[],     /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  lwDoubleComplex *C[],           /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize);

/* Batched solver based on LU factorization from getrf */

LWBLASAPI lwblasStatus_t  LWBLASWINAPI lwblasSgetrsBatched( lwblasHandle_t handle, 
                                                            lwblasOperation_t trans, 
                                                            int n, 
                                                            int nrhs, 
                                                            const float *Aarray[], 
                                                            int lda, 
                                                            const int *devIpiv, 
                                                            float *Barray[], 
                                                            int ldb, 
                                                            int *info,
                                                            int batchSize);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDgetrsBatched( lwblasHandle_t handle, 
                                                           lwblasOperation_t trans, 
                                                           int n, 
                                                           int nrhs, 
                                                           const double *Aarray[], 
                                                           int lda, 
                                                           const int *devIpiv, 
                                                           double *Barray[], 
                                                           int ldb, 
                                                           int *info,
                                                           int batchSize);

LWBLASAPI lwblasStatus_t  LWBLASWINAPI lwblasCgetrsBatched( lwblasHandle_t handle, 
                                                            lwblasOperation_t trans, 
                                                            int n, 
                                                            int nrhs, 
                                                            const lwComplex *Aarray[], 
                                                            int lda, 
                                                            const int *devIpiv, 
                                                            lwComplex *Barray[], 
                                                            int ldb, 
                                                            int *info,
                                                            int batchSize);


LWBLASAPI lwblasStatus_t  LWBLASWINAPI lwblasZgetrsBatched( lwblasHandle_t handle, 
                                                            lwblasOperation_t trans, 
                                                            int n, 
                                                            int nrhs, 
                                                            const lwDoubleComplex *Aarray[], 
                                                            int lda, 
                                                            const int *devIpiv, 
                                                            lwDoubleComplex *Barray[], 
                                                            int ldb, 
                                                            int *info,
                                                            int batchSize);



/* TRSM - Batched Triangular Solver */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasStrsmBatched( lwblasHandle_t    handle, 
                                                          lwblasSideMode_t  side, 
                                                          lwblasFillMode_t  uplo,
                                                          lwblasOperation_t trans, 
                                                          lwblasDiagType_t  diag,
                                                          int m, 
                                                          int n, 
                                                          const float *alpha,           /*Host or Device Pointer*/
                                                          const float *A[], 
                                                          int lda,
                                                          float *B[], 
                                                          int ldb,
                                                          int batchCount);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDtrsmBatched( lwblasHandle_t    handle, 
                                                          lwblasSideMode_t  side, 
                                                          lwblasFillMode_t  uplo,
                                                          lwblasOperation_t trans, 
                                                          lwblasDiagType_t  diag,
                                                          int m, 
                                                          int n, 
                                                          const double *alpha,          /*Host or Device Pointer*/
                                                          const double *A[], 
                                                          int lda,
                                                          double *B[], 
                                                          int ldb,
                                                          int batchCount);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCtrsmBatched( lwblasHandle_t    handle, 
                                                          lwblasSideMode_t  side, 
                                                          lwblasFillMode_t  uplo,
                                                          lwblasOperation_t trans, 
                                                          lwblasDiagType_t  diag,
                                                          int m, 
                                                          int n, 
                                                          const lwComplex *alpha,       /*Host or Device Pointer*/
                                                          const lwComplex *A[], 
                                                          int lda,
                                                          lwComplex *B[], 
                                                          int ldb,
                                                          int batchCount);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZtrsmBatched( lwblasHandle_t    handle, 
                                                          lwblasSideMode_t  side, 
                                                          lwblasFillMode_t  uplo,
                                                          lwblasOperation_t trans, 
                                                          lwblasDiagType_t  diag,
                                                          int m, 
                                                          int n, 
                                                          const lwDoubleComplex *alpha, /*Host or Device Pointer*/
                                                          const lwDoubleComplex *A[], 
                                                          int lda,
                                                          lwDoubleComplex *B[], 
                                                          int ldb,
                                                          int batchCount);

/* Batched - MATILW*/
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSmatilwBatched(lwblasHandle_t handle,
                                                          int n, 
                                                          const float *A[],                  /*Device pointer*/
                                                          int lda, 
                                                          float *Ailw[],               /*Device pointer*/
                                                          int lda_ilw, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDmatilwBatched(lwblasHandle_t handle,
                                                          int n, 
                                                          const double *A[],                 /*Device pointer*/
                                                          int lda, 
                                                          double *Ailw[],              /*Device pointer*/
                                                          int lda_ilw, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCmatilwBatched(lwblasHandle_t handle,
                                                          int n, 
                                                          const lwComplex *A[],              /*Device pointer*/
                                                          int lda, 
                                                          lwComplex *Ailw[],           /*Device pointer*/
                                                          int lda_ilw, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZmatilwBatched(lwblasHandle_t handle,
                                                          int n, 
                                                          const lwDoubleComplex *A[],        /*Device pointer*/
                                                          int lda, 
                                                          lwDoubleComplex *Ailw[],     /*Device pointer*/
                                                          int lda_ilw, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize);

/* Batch QR Factorization */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSgeqrfBatched( lwblasHandle_t handle, 
                                                           int m, 
                                                           int n,
                                                           float *Aarray[],           /*Device pointer*/
                                                           int lda, 
                                                           float *TauArray[],        /* Device pointer*/                                                           
                                                           int *info,
                                                           int batchSize);

LWBLASAPI lwblasStatus_t LWBLASWINAPI  lwblasDgeqrfBatched( lwblasHandle_t handle, 
                                                            int m, 
                                                            int n,
                                                            double *Aarray[],           /*Device pointer*/
                                                            int lda, 
                                                            double *TauArray[],        /* Device pointer*/                                                            
                                                            int *info,
                                                            int batchSize);

LWBLASAPI lwblasStatus_t LWBLASWINAPI  lwblasCgeqrfBatched( lwblasHandle_t handle, 
                                                            int m, 
                                                            int n,
                                                            lwComplex *Aarray[],           /*Device pointer*/
                                                            int lda, 
                                                            lwComplex *TauArray[],        /* Device pointer*/                                                            
                                                            int *info,
                                                            int batchSize);
                                                            
LWBLASAPI lwblasStatus_t LWBLASWINAPI  lwblasZgeqrfBatched( lwblasHandle_t handle, 
                                                            int m, 
                                                            int n,
                                                            lwDoubleComplex *Aarray[],           /*Device pointer*/
                                                            int lda, 
                                                            lwDoubleComplex *TauArray[],        /* Device pointer*/                                                          
                                                            int *info,
                                                            int batchSize);
/* Least Square Min only m >= n and Non-transpose supported */
LWBLASAPI lwblasStatus_t LWBLASWINAPI  lwblasSgelsBatched( lwblasHandle_t handle, 
                                                           lwblasOperation_t trans, 
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           float *Aarray[], /*Device pointer*/
                                                           int lda, 
                                                           float *Carray[], /* Device pointer*/
                                                           int ldc,                                                                 
                                                           int *info, 
                                                           int *devInfoArray, /* Device pointer*/
                                                           int batchSize );
                                                                
LWBLASAPI lwblasStatus_t LWBLASWINAPI  lwblasDgelsBatched( lwblasHandle_t handle,
                                                           lwblasOperation_t trans,  
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           double *Aarray[], /*Device pointer*/
                                                           int lda, 
                                                           double *Carray[], /* Device pointer*/
                                                           int ldc,                                                                 
                                                           int *info, 
                                                           int *devInfoArray, /* Device pointer*/
                                                           int batchSize);
                                                                
LWBLASAPI lwblasStatus_t LWBLASWINAPI  lwblasCgelsBatched( lwblasHandle_t handle, 
                                                           lwblasOperation_t trans, 
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           lwComplex *Aarray[], /*Device pointer*/
                                                           int lda, 
                                                           lwComplex *Carray[], /* Device pointer*/
                                                           int ldc,                                                                 
                                                           int *info, 
                                                           int *devInfoArray,
                                                           int batchSize);
                                                                
LWBLASAPI lwblasStatus_t LWBLASWINAPI  lwblasZgelsBatched( lwblasHandle_t handle, 
                                                           lwblasOperation_t trans, 
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           lwDoubleComplex *Aarray[], /*Device pointer*/
                                                           int lda, 
                                                           lwDoubleComplex *Carray[], /* Device pointer*/
                                                           int ldc,                                                                 
                                                           int *info, 
                                                           int *devInfoArray,
                                                           int batchSize);                                                                                                                                                                                                
/* DGMM */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasSdgmm(lwblasHandle_t handle,
                                                  lwblasSideMode_t mode, 
                                                  int m, 
                                                  int n,
                                                  const float *A, 
                                                  int lda,
                                                  const float *x, 
                                                  int incx,
                                                  float *C, 
                                                  int ldc);
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDdgmm(lwblasHandle_t handle,
                                                  lwblasSideMode_t mode, 
                                                  int m, 
                                                  int n,
                                                  const double *A, 
                                                  int lda,
                                                  const double *x, 
                                                  int incx,
                                                  double *C, 
                                                  int ldc);

LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCdgmm(lwblasHandle_t handle,
                                                  lwblasSideMode_t mode, 
                                                  int m, 
                                                  int n,
                                                  const lwComplex *A, 
                                                  int lda,
                                                  const lwComplex *x, 
                                                  int incx,
                                                  lwComplex *C, 
                                                  int ldc);
    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZdgmm(lwblasHandle_t handle,
                                                  lwblasSideMode_t mode, 
                                                  int m, 
                                                  int n,
                                                  const lwDoubleComplex *A, 
                                                  int lda,
                                                  const lwDoubleComplex *x, 
                                                  int incx,
                                                  lwDoubleComplex *C, 
                                                  int ldc);

/* TPTTR : Triangular Pack format to Triangular format */
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasStpttr ( lwblasHandle_t handle, 
                                                     lwblasFillMode_t uplo, 
                                                     int n,                                     
                                                     const float *AP,
                                                     float *A,  
                                                     int lda );
                                       
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDtpttr ( lwblasHandle_t handle, 
                                                     lwblasFillMode_t uplo, 
                                                     int n,                                     
                                                     const double *AP,
                                                     double *A,  
                                                     int lda );
                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCtpttr ( lwblasHandle_t handle, 
                                                     lwblasFillMode_t uplo, 
                                                     int n,                                     
                                                     const lwComplex *AP,
                                                     lwComplex *A,  
                                                     int lda );
                                                    
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZtpttr ( lwblasHandle_t handle, 
                                                     lwblasFillMode_t uplo, 
                                                     int n,                                     
                                                     const lwDoubleComplex *AP,
                                                     lwDoubleComplex *A,  
                                                     int lda );
 /* TRTTP : Triangular format to Triangular Pack format */                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasStrttp ( lwblasHandle_t handle, 
                                                     lwblasFillMode_t uplo, 
                                                     int n,                                     
                                                     const float *A,
                                                     int lda,
                                                     float *AP );
                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasDtrttp ( lwblasHandle_t handle, 
                                                     lwblasFillMode_t uplo, 
                                                     int n,                                     
                                                     const double *A,
                                                     int lda,
                                                     double *AP );
                                      
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasCtrttp ( lwblasHandle_t handle, 
                                                     lwblasFillMode_t uplo, 
                                                     int n,                                     
                                                     const lwComplex *A,
                                                     int lda,
                                                     lwComplex *AP );
                                                     
LWBLASAPI lwblasStatus_t LWBLASWINAPI lwblasZtrttp ( lwblasHandle_t handle, 
                                                     lwblasFillMode_t uplo, 
                                                     int n,                                     
                                                     const lwDoubleComplex *A,
                                                     int lda,
                                                     lwDoubleComplex *AP );                                        
                                      
#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* !defined(LWBLAS_API_H_) */
