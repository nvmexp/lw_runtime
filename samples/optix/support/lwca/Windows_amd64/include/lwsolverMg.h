/*
 * Copyright 2019 LWPU Corporation.  All rights reserved.
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

#if !defined(LWSOLVERMG_H_)
#define LWSOLVERMG_H_

#include <stdint.h>
#include "lwsolverDn.h"


#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

struct lwsolverMgContext;
typedef struct lwsolverMgContext *lwsolverMgHandle_t;


/**
 * \beief This enum decides how 1D device Ids (or process ranks) get mapped to a 2D grid.
 */
typedef enum {

  LWDALIBMG_GRID_MAPPING_ROW_MAJOR = 1,
  LWDALIBMG_GRID_MAPPING_COL_MAJOR = 0

} lwsolverMgGridMapping_t;

/** \brief Opaque structure of the distributed grid */
typedef void * lwdaLibMgGrid_t;
/** \brief Opaque structure of the distributed matrix descriptor */
typedef void * lwdaLibMgMatrixDesc_t;


lwsolverStatus_t LWSOLVERAPI lwsolverMgCreate(
    lwsolverMgHandle_t *handle);

lwsolverStatus_t LWSOLVERAPI lwsolverMgDestroy(
    lwsolverMgHandle_t handle);

lwsolverStatus_t LWSOLVERAPI lwsolverMgDeviceSelect(
    lwsolverMgHandle_t handle,
    int nbDevices,
    int deviceId[]);


/**
 * \brief Allocates resources related to the shared memory device grid.
 * \param[out] grid the opaque data strlwture that holds the grid
 * \param[in] numRowDevices number of devices in the row
 * \param[in] numColDevices number of devices in the column
 * \param[in] deviceId This array of size height * width stores the
 *            device-ids of the 2D grid; each entry must correspond to a valid gpu or to -1 (denoting CPU).
 * \param[in] mapping whether the 2D grid is in row/column major
 * \returns the status code
 */
lwsolverStatus_t LWSOLVERAPI lwsolverMgCreateDeviceGrid(
    lwdaLibMgGrid_t* grid, 
    int32_t numRowDevices, 
    int32_t numColDevices,
    const int32_t deviceId[], 
    lwsolverMgGridMapping_t mapping);

/**
 * \brief Releases the allocated resources related to the distributed grid.
 * \param[in] grid the opaque data strlwture that holds the distributed grid
 * \returns the status code
 */
lwsolverStatus_t LWSOLVERAPI lwsolverMgDestroyGrid(
    lwdaLibMgGrid_t grid);

/**
 * \brief Allocates resources related to the distributed matrix descriptor.
 * \param[out] desc the opaque data strlwture that holds the descriptor
 * \param[in] numRows number of total rows
 * \param[in] numCols number of total columns
 * \param[in] rowBlockSize row block size
 * \param[in] colBlockSize column block size
 * \param[in] dataType the data type of each element in lwdaDataType
 * \param[in] grid the opaque data structure of the distributed grid
 * \returns the status code
 */
lwsolverStatus_t LWSOLVERAPI lwsolverMgCreateMatrixDesc(
    lwdaLibMgMatrixDesc_t * desc,
    int64_t numRows, 
    int64_t numCols, 
    int64_t rowBlockSize, 
    int64_t colBlockSize,
    lwdaDataType dataType, 
    const lwdaLibMgGrid_t grid);

/**
 * \brief Releases the allocated resources related to the distributed matrix descriptor.
 * \param[in] desc the opaque data strlwture that holds the descriptor
 * \returns the status code
 */
lwsolverStatus_t LWSOLVERAPI lwsolverMgDestroyMatrixDesc(
    lwdaLibMgMatrixDesc_t desc);



lwsolverStatus_t LWSOLVERAPI lwsolverMgSyevd_bufferSize(
    lwsolverMgHandle_t handle,
    lwsolverEigMode_t jobz, 
    lwblasFillMode_t uplo, 
    int N,
    void *array_d_A[], 
    int IA, 
    int JA, 
    lwdaLibMgMatrixDesc_t descrA,
    void *W,
    lwdaDataType dataTypeW,
    lwdaDataType computeType,
    int64_t *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverMgSyevd(
    lwsolverMgHandle_t handle,
    lwsolverEigMode_t jobz,
    lwblasFillMode_t uplo,
    int N,
    void *array_d_A[],
    int IA,
    int JA,
    lwdaLibMgMatrixDesc_t descrA,
    void *W,
    lwdaDataType dataTypeW,
    lwdaDataType computeType,
    void *array_d_work[],
    int64_t lwork,
    int *info );

lwsolverStatus_t LWSOLVERAPI lwsolverMgGetrf_bufferSize(
    lwsolverMgHandle_t handle,
    int M,
    int N,
    void *array_d_A[],
    int IA,
    int JA,
    lwdaLibMgMatrixDesc_t descrA,
    int *array_d_IPIV[],
    lwdaDataType computeType,
    int64_t *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverMgGetrf(
    lwsolverMgHandle_t handle,
    int M,
    int N,
    void *array_d_A[],
    int IA,
    int JA,
    lwdaLibMgMatrixDesc_t descrA,
    int *array_d_IPIV[],
    lwdaDataType computeType,
    void *array_d_work[],
    int64_t lwork,
    int *info );

lwsolverStatus_t LWSOLVERAPI lwsolverMgGetrs_bufferSize(
    lwsolverMgHandle_t handle,
    lwblasOperation_t TRANS,
    int N,
    int NRHS,
    void *array_d_A[],
    int IA, 
    int JA, 
    lwdaLibMgMatrixDesc_t descrA,
    int *array_d_IPIV[],  
    void *array_d_B[],
    int IB, 
    int JB, 
    lwdaLibMgMatrixDesc_t descrB,
    lwdaDataType computeType,
    int64_t *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverMgGetrs(
    lwsolverMgHandle_t handle,
    lwblasOperation_t TRANS,
    int N,
    int NRHS,
    void *array_d_A[],
    int IA, 
    int JA, 
    lwdaLibMgMatrixDesc_t descrA,
    int *array_d_IPIV[], 
    void *array_d_B[],
    int IB, 
    int JB, 
    lwdaLibMgMatrixDesc_t descrB,
    lwdaDataType computeType,
    void *array_d_work[],
    int64_t lwork,
    int *info );

lwsolverStatus_t LWSOLVERAPI lwsolverMgPotrf_bufferSize( 
    lwsolverMgHandle_t handle,
	lwblasFillMode_t uplo,
    int N, 
    void *array_d_A[],
    int IA,
    int JA, 
    lwdaLibMgMatrixDesc_t descrA,
    lwdaDataType computeType, 
	int64_t *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverMgPotrf( 
    lwsolverMgHandle_t handle,
	lwblasFillMode_t uplo,
    int N, 
    void *array_d_A[],
    int IA, 
    int JA, 
    lwdaLibMgMatrixDesc_t descrA,
    lwdaDataType computeType, 
    void *array_d_work[],
    int64_t lwork,
    int *h_info);

lwsolverStatus_t LWSOLVERAPI lwsolverMgPotrs_bufferSize( 
    lwsolverMgHandle_t handle,
	lwblasFillMode_t uplo,
    int n, 
	int nrhs,
    void *array_d_A[],
    int IA, 
    int JA, 
    lwdaLibMgMatrixDesc_t descrA,
    void *array_d_B[],
    int IB, 
    int JB, 
    lwdaLibMgMatrixDesc_t descrB,
    lwdaDataType computeType, 
	int64_t *lwork );

lwsolverStatus_t LWSOLVERAPI lwsolverMgPotrs( 
    lwsolverMgHandle_t handle,
	lwblasFillMode_t uplo,
    int n, 
	int nrhs,
    void *array_d_A[],
    int IA, 
    int JA, 
    lwdaLibMgMatrixDesc_t descrA,
    void *array_d_B[],
    int IB, 
    int JB, 
    lwdaLibMgMatrixDesc_t descrB,
    lwdaDataType computeType, 
    void *array_d_work[],
	int64_t lwork,
	int *h_info);

lwsolverStatus_t LWSOLVERAPI lwsolverMgPotri_bufferSize( 
    lwsolverMgHandle_t handle,
	lwblasFillMode_t uplo,
    int N, 
    void *array_d_A[],
    int IA, 
    int JA, 
    lwdaLibMgMatrixDesc_t descrA,
    lwdaDataType computeType, 
	int64_t *lwork);

lwsolverStatus_t LWSOLVERAPI lwsolverMgPotri( 
    lwsolverMgHandle_t handle,
	lwblasFillMode_t uplo,
    int N, 
    void *array_d_A[],
    int IA, 
    int JA, 
    lwdaLibMgMatrixDesc_t descrA,
    lwdaDataType computeType, 
    void *array_d_work[],
	int64_t lwork,
    int *h_info);



#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // LWSOLVERMG_H_
 

