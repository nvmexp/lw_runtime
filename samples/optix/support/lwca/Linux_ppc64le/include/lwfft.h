
 /* Copyright 2005-2014 LWPU Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to LWPU intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to LWPU and are being provided under the terms and
  * conditions of a form of LWPU software license agreement by and
  * between LWPU and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of LWPU is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
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
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
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

/*!
* \file lwfft.h
* \brief Public header file for the LWPU LWCA FFT library (LWFFT)
*/

#ifndef _LWFFT_H_
#define _LWFFT_H_


#include "lwComplex.h"
#include "driver_types.h"
#include "library_types.h"

#ifndef LWFFTAPI
#ifdef _WIN32
#define LWFFTAPI __stdcall
#elif __GNUC__ >= 4
#define LWFFTAPI __attribute__ ((visibility ("default")))
#else
#define LWFFTAPI
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LWFFT_VER_MAJOR 10
#define LWFFT_VER_MINOR 4
#define LWFFT_VER_PATCH 2
#define LWFFT_VER_BUILD 123

// lwFFT library version
//
// LWFFT_VERSION / 1000 - major version
// LWFFT_VERSION / 100 % 100 - minor version
// LWFFT_VERSION % 100 - patch level
#define LWFFT_VERSION 10402

// LWFFT API function return values
typedef enum lwfftResult_t {
  LWFFT_SUCCESS        = 0x0,
  LWFFT_ILWALID_PLAN   = 0x1,
  LWFFT_ALLOC_FAILED   = 0x2,
  LWFFT_ILWALID_TYPE   = 0x3,
  LWFFT_ILWALID_VALUE  = 0x4,
  LWFFT_INTERNAL_ERROR = 0x5,
  LWFFT_EXEC_FAILED    = 0x6,
  LWFFT_SETUP_FAILED   = 0x7,
  LWFFT_ILWALID_SIZE   = 0x8,
  LWFFT_UNALIGNED_DATA = 0x9,
  LWFFT_INCOMPLETE_PARAMETER_LIST = 0xA,
  LWFFT_ILWALID_DEVICE = 0xB,
  LWFFT_PARSE_ERROR = 0xC,
  LWFFT_NO_WORKSPACE = 0xD,
  LWFFT_NOT_IMPLEMENTED = 0xE,
  LWFFT_LICENSE_ERROR = 0x0F,
  LWFFT_NOT_SUPPORTED = 0x10

} lwfftResult;

#define MAX_LWFFT_ERROR 0x11


// LWFFT defines and supports the following data types


// lwfftReal is a single-precision, floating-point real data type.
// lwfftDoubleReal is a double-precision, real data type.
typedef float lwfftReal;
typedef double lwfftDoubleReal;

// lwfftComplex is a single-precision, floating-point complex data type that
// consists of interleaved real and imaginary components.
// lwfftDoubleComplex is the double-precision equivalent.
typedef lwComplex lwfftComplex;
typedef lwDoubleComplex lwfftDoubleComplex;

// LWFFT transform directions
#define LWFFT_FORWARD -1 // Forward FFT
#define LWFFT_ILWERSE  1 // Ilwerse FFT

// LWFFT supports the following transform types
typedef enum lwfftType_t {
  LWFFT_R2C = 0x2a,     // Real to Complex (interleaved)
  LWFFT_C2R = 0x2c,     // Complex (interleaved) to Real
  LWFFT_C2C = 0x29,     // Complex to Complex, interleaved
  LWFFT_D2Z = 0x6a,     // Double to Double-Complex
  LWFFT_Z2D = 0x6c,     // Double-Complex to Double
  LWFFT_Z2Z = 0x69      // Double-Complex to Double-Complex
} lwfftType;

// LWFFT supports the following data layouts
typedef enum lwfftCompatibility_t {
    LWFFT_COMPATIBILITY_FFTW_PADDING    = 0x01    // The default value
} lwfftCompatibility;

#define LWFFT_COMPATIBILITY_DEFAULT   LWFFT_COMPATIBILITY_FFTW_PADDING

//
// structure definition used by the shim between old and new APIs
//
#define MAX_SHIM_RANK 3

// lwfftHandle is a handle type used to store and access LWFFT plans.
typedef int lwfftHandle;


lwfftResult LWFFTAPI lwfftPlan1d(lwfftHandle *plan,
                                 int nx,
                                 lwfftType type,
                                 int batch);

lwfftResult LWFFTAPI lwfftPlan2d(lwfftHandle *plan,
                                 int nx, int ny,
                                 lwfftType type);

lwfftResult LWFFTAPI lwfftPlan3d(lwfftHandle *plan,
                                 int nx, int ny, int nz,
                                 lwfftType type);

lwfftResult LWFFTAPI lwfftPlanMany(lwfftHandle *plan,
                                   int rank,
                                   int *n,
                                   int *inembed, int istride, int idist,
                                   int *onembed, int ostride, int odist,
                                   lwfftType type,
                                   int batch);

lwfftResult LWFFTAPI lwfftMakePlan1d(lwfftHandle plan,
                                     int nx,
                                     lwfftType type,
                                     int batch,
                                     size_t *workSize);

lwfftResult LWFFTAPI lwfftMakePlan2d(lwfftHandle plan,
                                     int nx, int ny,
                                     lwfftType type,
                                     size_t *workSize);

lwfftResult LWFFTAPI lwfftMakePlan3d(lwfftHandle plan,
                                     int nx, int ny, int nz,
                                     lwfftType type,
                                     size_t *workSize);

lwfftResult LWFFTAPI lwfftMakePlanMany(lwfftHandle plan,
                                       int rank,
                                       int *n,
                                       int *inembed, int istride, int idist,
                                       int *onembed, int ostride, int odist,
                                       lwfftType type,
                                       int batch,
                                       size_t *workSize);

lwfftResult LWFFTAPI lwfftMakePlanMany64(lwfftHandle plan,
                                         int rank,
                                         long long int *n,
                                         long long int *inembed,
                                         long long int istride,
                                         long long int idist,
                                         long long int *onembed,
                                         long long int ostride, long long int odist,
                                         lwfftType type,
                                         long long int batch,
                                         size_t * workSize);

lwfftResult LWFFTAPI lwfftGetSizeMany64(lwfftHandle plan,
                                        int rank,
                                        long long int *n,
                                        long long int *inembed,
                                        long long int istride, long long int idist,
                                        long long int *onembed,
                                        long long int ostride, long long int odist,
                                        lwfftType type,
                                        long long int batch,
                                        size_t *workSize);




lwfftResult LWFFTAPI lwfftEstimate1d(int nx,
                                     lwfftType type,
                                     int batch,
                                     size_t *workSize);

lwfftResult LWFFTAPI lwfftEstimate2d(int nx, int ny,
                                     lwfftType type,
                                     size_t *workSize);

lwfftResult LWFFTAPI lwfftEstimate3d(int nx, int ny, int nz,
                                     lwfftType type,
                                     size_t *workSize);

lwfftResult LWFFTAPI lwfftEstimateMany(int rank,
                                       int *n,
                                       int *inembed, int istride, int idist,
                                       int *onembed, int ostride, int odist,
                                       lwfftType type,
                                       int batch,
                                       size_t *workSize);

lwfftResult LWFFTAPI lwfftCreate(lwfftHandle * handle);

lwfftResult LWFFTAPI lwfftGetSize1d(lwfftHandle handle,
                                    int nx,
                                    lwfftType type,
                                    int batch,
                                    size_t *workSize );

lwfftResult LWFFTAPI lwfftGetSize2d(lwfftHandle handle,
                                    int nx, int ny,
                                    lwfftType type,
                                    size_t *workSize);

lwfftResult LWFFTAPI lwfftGetSize3d(lwfftHandle handle,
                                    int nx, int ny, int nz,
                                    lwfftType type,
                                    size_t *workSize);

lwfftResult LWFFTAPI lwfftGetSizeMany(lwfftHandle handle,
                                      int rank, int *n,
                                      int *inembed, int istride, int idist,
                                      int *onembed, int ostride, int odist,
                                      lwfftType type, int batch, size_t *workArea);

lwfftResult LWFFTAPI lwfftGetSize(lwfftHandle handle, size_t *workSize);

lwfftResult LWFFTAPI lwfftSetWorkArea(lwfftHandle plan, void *workArea);

lwfftResult LWFFTAPI lwfftSetAutoAllocation(lwfftHandle plan, int autoAllocate);

lwfftResult LWFFTAPI lwfftExecC2C(lwfftHandle plan,
                                  lwfftComplex *idata,
                                  lwfftComplex *odata,
                                  int direction);

lwfftResult LWFFTAPI lwfftExecR2C(lwfftHandle plan,
                                  lwfftReal *idata,
                                  lwfftComplex *odata);

lwfftResult LWFFTAPI lwfftExecC2R(lwfftHandle plan,
                                  lwfftComplex *idata,
                                  lwfftReal *odata);

lwfftResult LWFFTAPI lwfftExecZ2Z(lwfftHandle plan,
                                  lwfftDoubleComplex *idata,
                                  lwfftDoubleComplex *odata,
                                  int direction);

lwfftResult LWFFTAPI lwfftExecD2Z(lwfftHandle plan,
                                  lwfftDoubleReal *idata,
                                  lwfftDoubleComplex *odata);

lwfftResult LWFFTAPI lwfftExecZ2D(lwfftHandle plan,
                                  lwfftDoubleComplex *idata,
                                  lwfftDoubleReal *odata);


// utility functions
lwfftResult LWFFTAPI lwfftSetStream(lwfftHandle plan,
                                    lwdaStream_t stream);

lwfftResult LWFFTAPI lwfftDestroy(lwfftHandle plan);

lwfftResult LWFFTAPI lwfftGetVersion(int *version);

lwfftResult LWFFTAPI lwfftGetProperty(libraryPropertyType type,
                                      int *value);

#ifdef __cplusplus
}
#endif

#endif /* _LWFFT_H_ */
