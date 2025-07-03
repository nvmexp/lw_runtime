/* Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

////////////////////////////////////////////////////////////////////////////////
// These are LWCA Helper functions for initialization and error checking

#ifndef COMMON_HELPER_LWDA_H_
#define COMMON_HELPER_LWDA_H_

#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <helper_string.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// Note, it is required that your SDK sample to include the proper header
// files, please refer the LWCA examples for examples of the needed LWCA
// headers, which may change depending on which LWCA functions are used.

// LWCA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char *_lwdaGetErrorEnum(lwdaError_t error) {
  return lwdaGetErrorName(error);
}
#endif

#ifdef LWDA_DRIVER_API
// LWCA Driver API errors
static const char *_lwdaGetErrorEnum(LWresult error) {
  static char unknown[] = "<unknown>";
  const char *ret = NULL;
  lwGetErrorName(error, &ret);
  return ret ? ret : unknown;
}
#endif

#ifdef LWBLAS_API_H_
// lwBLAS API errors
static const char *_lwdaGetErrorEnum(lwblasStatus_t error) {
  switch (error) {
    case LWBLAS_STATUS_SUCCESS:
      return "LWBLAS_STATUS_SUCCESS";

    case LWBLAS_STATUS_NOT_INITIALIZED:
      return "LWBLAS_STATUS_NOT_INITIALIZED";

    case LWBLAS_STATUS_ALLOC_FAILED:
      return "LWBLAS_STATUS_ALLOC_FAILED";

    case LWBLAS_STATUS_ILWALID_VALUE:
      return "LWBLAS_STATUS_ILWALID_VALUE";

    case LWBLAS_STATUS_ARCH_MISMATCH:
      return "LWBLAS_STATUS_ARCH_MISMATCH";

    case LWBLAS_STATUS_MAPPING_ERROR:
      return "LWBLAS_STATUS_MAPPING_ERROR";

    case LWBLAS_STATUS_EXELWTION_FAILED:
      return "LWBLAS_STATUS_EXELWTION_FAILED";

    case LWBLAS_STATUS_INTERNAL_ERROR:
      return "LWBLAS_STATUS_INTERNAL_ERROR";

    case LWBLAS_STATUS_NOT_SUPPORTED:
      return "LWBLAS_STATUS_NOT_SUPPORTED";

    case LWBLAS_STATUS_LICENSE_ERROR:
      return "LWBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef _LWFFT_H_
// lwFFT API errors
static const char *_lwdaGetErrorEnum(lwfftResult error) {
  switch (error) {
    case LWFFT_SUCCESS:
      return "LWFFT_SUCCESS";

    case LWFFT_ILWALID_PLAN:
      return "LWFFT_ILWALID_PLAN";

    case LWFFT_ALLOC_FAILED:
      return "LWFFT_ALLOC_FAILED";

    case LWFFT_ILWALID_TYPE:
      return "LWFFT_ILWALID_TYPE";

    case LWFFT_ILWALID_VALUE:
      return "LWFFT_ILWALID_VALUE";

    case LWFFT_INTERNAL_ERROR:
      return "LWFFT_INTERNAL_ERROR";

    case LWFFT_EXEC_FAILED:
      return "LWFFT_EXEC_FAILED";

    case LWFFT_SETUP_FAILED:
      return "LWFFT_SETUP_FAILED";

    case LWFFT_ILWALID_SIZE:
      return "LWFFT_ILWALID_SIZE";

    case LWFFT_UNALIGNED_DATA:
      return "LWFFT_UNALIGNED_DATA";

    case LWFFT_INCOMPLETE_PARAMETER_LIST:
      return "LWFFT_INCOMPLETE_PARAMETER_LIST";

    case LWFFT_ILWALID_DEVICE:
      return "LWFFT_ILWALID_DEVICE";

    case LWFFT_PARSE_ERROR:
      return "LWFFT_PARSE_ERROR";

    case LWFFT_NO_WORKSPACE:
      return "LWFFT_NO_WORKSPACE";

    case LWFFT_NOT_IMPLEMENTED:
      return "LWFFT_NOT_IMPLEMENTED";

    case LWFFT_LICENSE_ERROR:
      return "LWFFT_LICENSE_ERROR";

    case LWFFT_NOT_SUPPORTED:
      return "LWFFT_NOT_SUPPORTED";
  }

  return "<unknown>";
}
#endif

#ifdef LWSPARSEAPI
// lwSPARSE API errors
static const char *_lwdaGetErrorEnum(lwsparseStatus_t error) {
  switch (error) {
    case LWSPARSE_STATUS_SUCCESS:
      return "LWSPARSE_STATUS_SUCCESS";

    case LWSPARSE_STATUS_NOT_INITIALIZED:
      return "LWSPARSE_STATUS_NOT_INITIALIZED";

    case LWSPARSE_STATUS_ALLOC_FAILED:
      return "LWSPARSE_STATUS_ALLOC_FAILED";

    case LWSPARSE_STATUS_ILWALID_VALUE:
      return "LWSPARSE_STATUS_ILWALID_VALUE";

    case LWSPARSE_STATUS_ARCH_MISMATCH:
      return "LWSPARSE_STATUS_ARCH_MISMATCH";

    case LWSPARSE_STATUS_MAPPING_ERROR:
      return "LWSPARSE_STATUS_MAPPING_ERROR";

    case LWSPARSE_STATUS_EXELWTION_FAILED:
      return "LWSPARSE_STATUS_EXELWTION_FAILED";

    case LWSPARSE_STATUS_INTERNAL_ERROR:
      return "LWSPARSE_STATUS_INTERNAL_ERROR";

    case LWSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "LWSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  }

  return "<unknown>";
}
#endif

#ifdef LWSOLVER_COMMON_H_
// lwSOLVER API errors
static const char *_lwdaGetErrorEnum(lwsolverStatus_t error) {
  switch (error) {
    case LWSOLVER_STATUS_SUCCESS:
      return "LWSOLVER_STATUS_SUCCESS";
    case LWSOLVER_STATUS_NOT_INITIALIZED:
      return "LWSOLVER_STATUS_NOT_INITIALIZED";
    case LWSOLVER_STATUS_ALLOC_FAILED:
      return "LWSOLVER_STATUS_ALLOC_FAILED";
    case LWSOLVER_STATUS_ILWALID_VALUE:
      return "LWSOLVER_STATUS_ILWALID_VALUE";
    case LWSOLVER_STATUS_ARCH_MISMATCH:
      return "LWSOLVER_STATUS_ARCH_MISMATCH";
    case LWSOLVER_STATUS_MAPPING_ERROR:
      return "LWSOLVER_STATUS_MAPPING_ERROR";
    case LWSOLVER_STATUS_EXELWTION_FAILED:
      return "LWSOLVER_STATUS_EXELWTION_FAILED";
    case LWSOLVER_STATUS_INTERNAL_ERROR:
      return "LWSOLVER_STATUS_INTERNAL_ERROR";
    case LWSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "LWSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case LWSOLVER_STATUS_NOT_SUPPORTED:
      return "LWSOLVER_STATUS_NOT_SUPPORTED ";
    case LWSOLVER_STATUS_ZERO_PIVOT:
      return "LWSOLVER_STATUS_ZERO_PIVOT";
    case LWSOLVER_STATUS_ILWALID_LICENSE:
      return "LWSOLVER_STATUS_ILWALID_LICENSE";
  }

  return "<unknown>";
}
#endif

#ifdef LWRAND_H_
// lwRAND API errors
static const char *_lwdaGetErrorEnum(lwrandStatus_t error) {
  switch (error) {
    case LWRAND_STATUS_SUCCESS:
      return "LWRAND_STATUS_SUCCESS";

    case LWRAND_STATUS_VERSION_MISMATCH:
      return "LWRAND_STATUS_VERSION_MISMATCH";

    case LWRAND_STATUS_NOT_INITIALIZED:
      return "LWRAND_STATUS_NOT_INITIALIZED";

    case LWRAND_STATUS_ALLOCATION_FAILED:
      return "LWRAND_STATUS_ALLOCATION_FAILED";

    case LWRAND_STATUS_TYPE_ERROR:
      return "LWRAND_STATUS_TYPE_ERROR";

    case LWRAND_STATUS_OUT_OF_RANGE:
      return "LWRAND_STATUS_OUT_OF_RANGE";

    case LWRAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "LWRAND_STATUS_LENGTH_NOT_MULTIPLE";

    case LWRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "LWRAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case LWRAND_STATUS_LAUNCH_FAILURE:
      return "LWRAND_STATUS_LAUNCH_FAILURE";

    case LWRAND_STATUS_PREEXISTING_FAILURE:
      return "LWRAND_STATUS_PREEXISTING_FAILURE";

    case LWRAND_STATUS_INITIALIZATION_FAILED:
      return "LWRAND_STATUS_INITIALIZATION_FAILED";

    case LWRAND_STATUS_ARCH_MISMATCH:
      return "LWRAND_STATUS_ARCH_MISMATCH";

    case LWRAND_STATUS_INTERNAL_ERROR:
      return "LWRAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef LWJPEGAPI
// lwJPEG API errors
static const char *_lwdaGetErrorEnum(lwjpegStatus_t error) {
  switch (error) {
    case LWJPEG_STATUS_SUCCESS:
      return "LWJPEG_STATUS_SUCCESS";

    case LWJPEG_STATUS_NOT_INITIALIZED:
      return "LWJPEG_STATUS_NOT_INITIALIZED";

    case LWJPEG_STATUS_ILWALID_PARAMETER:
      return "LWJPEG_STATUS_ILWALID_PARAMETER";

    case LWJPEG_STATUS_BAD_JPEG:
      return "LWJPEG_STATUS_BAD_JPEG";

    case LWJPEG_STATUS_JPEG_NOT_SUPPORTED:
      return "LWJPEG_STATUS_JPEG_NOT_SUPPORTED";

    case LWJPEG_STATUS_ALLOCATOR_FAILURE:
      return "LWJPEG_STATUS_ALLOCATOR_FAILURE";

    case LWJPEG_STATUS_EXELWTION_FAILED:
      return "LWJPEG_STATUS_EXELWTION_FAILED";

    case LWJPEG_STATUS_ARCH_MISMATCH:
      return "LWJPEG_STATUS_ARCH_MISMATCH";

    case LWJPEG_STATUS_INTERNAL_ERROR:
      return "LWJPEG_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef LW_NPPIDEFS_H
// NPP API errors
static const char *_lwdaGetErrorEnum(NppStatus error) {
  switch (error) {
    case NPP_NOT_SUPPORTED_MODE_ERROR:
      return "NPP_NOT_SUPPORTED_MODE_ERROR";

    case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
      return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";

    case NPP_RESIZE_NO_OPERATION_ERROR:
      return "NPP_RESIZE_NO_OPERATION_ERROR";

    case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
      return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

    case NPP_BAD_ARG_ERROR:
      return "NPP_BAD_ARGUMENT_ERROR";

    case NPP_COEFF_ERROR:
      return "NPP_COEFFICIENT_ERROR";

    case NPP_RECT_ERROR:
      return "NPP_RECTANGLE_ERROR";

    case NPP_QUAD_ERROR:
      return "NPP_QUADRANGLE_ERROR";

    case NPP_MEM_ALLOC_ERR:
      return "NPP_MEMORY_ALLOCATION_ERROR";

    case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:
      return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

    case NPP_ILWALID_INPUT:
      return "NPP_ILWALID_INPUT";

    case NPP_POINTER_ERROR:
      return "NPP_POINTER_ERROR";

    case NPP_WARNING:
      return "NPP_WARNING";

    case NPP_ODD_ROI_WARNING:
      return "NPP_ODD_ROI_WARNING";
#else

    // These are for LWCA 5.5 or higher
    case NPP_BAD_ARGUMENT_ERROR:
      return "NPP_BAD_ARGUMENT_ERROR";

    case NPP_COEFFICIENT_ERROR:
      return "NPP_COEFFICIENT_ERROR";

    case NPP_RECTANGLE_ERROR:
      return "NPP_RECTANGLE_ERROR";

    case NPP_QUADRANGLE_ERROR:
      return "NPP_QUADRANGLE_ERROR";

    case NPP_MEMORY_ALLOCATION_ERR:
      return "NPP_MEMORY_ALLOCATION_ERROR";

    case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
      return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

    case NPP_ILWALID_HOST_POINTER_ERROR:
      return "NPP_ILWALID_HOST_POINTER_ERROR";

    case NPP_ILWALID_DEVICE_POINTER_ERROR:
      return "NPP_ILWALID_DEVICE_POINTER_ERROR";
#endif

    case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
      return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";

    case NPP_TEXTURE_BIND_ERROR:
      return "NPP_TEXTURE_BIND_ERROR";

    case NPP_WRONG_INTERSECTION_ROI_ERROR:
      return "NPP_WRONG_INTERSECTION_ROI_ERROR";

    case NPP_NOT_EVEN_STEP_ERROR:
      return "NPP_NOT_EVEN_STEP_ERROR";

    case NPP_INTERPOLATION_ERROR:
      return "NPP_INTERPOLATION_ERROR";

    case NPP_RESIZE_FACTOR_ERROR:
      return "NPP_RESIZE_FACTOR_ERROR";

    case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
      return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

    case NPP_MEMFREE_ERR:
      return "NPP_MEMFREE_ERR";

    case NPP_MEMSET_ERR:
      return "NPP_MEMSET_ERR";

    case NPP_MEMCPY_ERR:
      return "NPP_MEMCPY_ERROR";

    case NPP_MIRROR_FLIP_ERR:
      return "NPP_MIRROR_FLIP_ERR";
#else

    case NPP_MEMFREE_ERROR:
      return "NPP_MEMFREE_ERROR";

    case NPP_MEMSET_ERROR:
      return "NPP_MEMSET_ERROR";

    case NPP_MEMCPY_ERROR:
      return "NPP_MEMCPY_ERROR";

    case NPP_MIRROR_FLIP_ERROR:
      return "NPP_MIRROR_FLIP_ERROR";
#endif

    case NPP_ALIGNMENT_ERROR:
      return "NPP_ALIGNMENT_ERROR";

    case NPP_STEP_ERROR:
      return "NPP_STEP_ERROR";

    case NPP_SIZE_ERROR:
      return "NPP_SIZE_ERROR";

    case NPP_NULL_POINTER_ERROR:
      return "NPP_NULL_POINTER_ERROR";

    case NPP_LWDA_KERNEL_EXELWTION_ERROR:
      return "NPP_LWDA_KERNEL_EXELWTION_ERROR";

    case NPP_NOT_IMPLEMENTED_ERROR:
      return "NPP_NOT_IMPLEMENTED_ERROR";

    case NPP_ERROR:
      return "NPP_ERROR";

    case NPP_SUCCESS:
      return "NPP_SUCCESS";

    case NPP_WRONG_INTERSECTION_QUAD_WARNING:
      return "NPP_WRONG_INTERSECTION_QUAD_WARNING";

    case NPP_MISALIGNED_DST_ROI_WARNING:
      return "NPP_MISALIGNED_DST_ROI_WARNING";

    case NPP_AFFINE_QUAD_INCORRECT_WARNING:
      return "NPP_AFFINE_QUAD_INCORRECT_WARNING";

    case NPP_DOUBLE_SIZE_WARNING:
      return "NPP_DOUBLE_SIZE_WARNING";

    case NPP_WRONG_INTERSECTION_ROI_WARNING:
      return "NPP_WRONG_INTERSECTION_ROI_WARNING";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x6000
    /* These are 6.0 or higher */
    case NPP_LUT_PALETTE_BITSIZE_ERROR:
      return "NPP_LUT_PALETTE_BITSIZE_ERROR";

    case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
      return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";

    case NPP_QUALITY_INDEX_ERROR:
      return "NPP_QUALITY_INDEX_ERROR";

    case NPP_CHANNEL_ORDER_ERROR:
      return "NPP_CHANNEL_ORDER_ERROR";

    case NPP_ZERO_MASK_VALUE_ERROR:
      return "NPP_ZERO_MASK_VALUE_ERROR";

    case NPP_NUMBER_OF_CHANNELS_ERROR:
      return "NPP_NUMBER_OF_CHANNELS_ERROR";

    case NPP_COI_ERROR:
      return "NPP_COI_ERROR";

    case NPP_DIVISOR_ERROR:
      return "NPP_DIVISOR_ERROR";

    case NPP_CHANNEL_ERROR:
      return "NPP_CHANNEL_ERROR";

    case NPP_STRIDE_ERROR:
      return "NPP_STRIDE_ERROR";

    case NPP_ANCHOR_ERROR:
      return "NPP_ANCHOR_ERROR";

    case NPP_MASK_SIZE_ERROR:
      return "NPP_MASK_SIZE_ERROR";

    case NPP_MOMENT_00_ZERO_ERROR:
      return "NPP_MOMENT_00_ZERO_ERROR";

    case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
      return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";

    case NPP_THRESHOLD_ERROR:
      return "NPP_THRESHOLD_ERROR";

    case NPP_CONTEXT_MATCH_ERROR:
      return "NPP_CONTEXT_MATCH_ERROR";

    case NPP_FFT_FLAG_ERROR:
      return "NPP_FFT_FLAG_ERROR";

    case NPP_FFT_ORDER_ERROR:
      return "NPP_FFT_ORDER_ERROR";

    case NPP_SCALE_RANGE_ERROR:
      return "NPP_SCALE_RANGE_ERROR";

    case NPP_DATA_TYPE_ERROR:
      return "NPP_DATA_TYPE_ERROR";

    case NPP_OUT_OFF_RANGE_ERROR:
      return "NPP_OUT_OFF_RANGE_ERROR";

    case NPP_DIVIDE_BY_ZERO_ERROR:
      return "NPP_DIVIDE_BY_ZERO_ERROR";

    case NPP_RANGE_ERROR:
      return "NPP_RANGE_ERROR";

    case NPP_NO_MEMORY_ERROR:
      return "NPP_NO_MEMORY_ERROR";

    case NPP_ERROR_RESERVED:
      return "NPP_ERROR_RESERVED";

    case NPP_NO_OPERATION_WARNING:
      return "NPP_NO_OPERATION_WARNING";

    case NPP_DIVIDE_BY_ZERO_WARNING:
      return "NPP_DIVIDE_BY_ZERO_WARNING";
#endif

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x7000
    /* These are 7.0 or higher */
    case NPP_OVERFLOW_ERROR:
      return "NPP_OVERFLOW_ERROR";

    case NPP_CORRUPTED_DATA_ERROR:
      return "NPP_CORRUPTED_DATA_ERROR";
#endif
  }

  return "<unknown>";
}
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "LWCA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _lwdaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper LWCA error strings in the event
// that a LWCA host call returns an error
#define checkLwdaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling lwdaGetLastError
#define getLastLwdaError(msg) __getLastLwdaError(msg, __FILE__, __LINE__)

inline void __getLastLwdaError(const char *errorMessage, const char *file,
                               const int line) {
  lwdaError_t err = lwdaGetLastError();

  if (lwdaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastLwdaError() LWCA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            lwdaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// This will only print the proper error string when calling lwdaGetLastError
// but not exit program incase error detected.
#define printLastLwdaError(msg) __printLastLwdaError(msg, __FILE__, __LINE__)

inline void __printLastLwdaError(const char *errorMessage, const char *file,
                                 const int line) {
  lwdaError_t err = lwdaGetLastError();

  if (lwdaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastLwdaError() LWCA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            lwdaGetErrorString(err));
  }
}
#endif

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

// Float To Int colwersion
inline int ftoi(float value) {
  return (value >= 0 ? static_cast<int>(value + 0.5)
                     : static_cast<int>(value - 0.5));
}

// Beginning of GPU Architecture definitions
inline int _ColwertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline const char* _ColwertSMVer2ArchName(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the GPU Arch name)
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    const char* name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[] = {
      {0x30, "Kepler"},
      {0x32, "Kepler"},
      {0x35, "Kepler"},
      {0x37, "Kepler"},
      {0x50, "Maxwell"},
      {0x52, "Maxwell"},
      {0x53, "Maxwell"},
      {0x60, "Pascal"},
      {0x61, "Pascal"},
      {0x62, "Pascal"},
      {0x70, "Volta"},
      {0x72, "Xavier"},
      {0x75, "Turing"},
      {-1, "Graphics Device"}};

  int index = 0;

  while (nGpuArchNameSM[index].SM != -1) {
    if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchNameSM[index].name;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoArchName for SM %d.%d is undefined."
      "  Default to use %s\n",
      major, minor, nGpuArchNameSM[index - 1].name);
  return nGpuArchNameSM[index - 1].name;
}
  // end of GPU Architecture definitions

#ifdef __LWDA_RUNTIME_H__
// General GPU Device LWCA Initialization
inline int gpuDeviceInit(int devID) {
  int device_count;
  checkLwdaErrors(lwdaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuDeviceInit() LWCA error: "
            "no devices supporting LWCA.\n");
    exit(EXIT_FAILURE);
  }

  if (devID < 0) {
    devID = 0;
  }

  if (devID > device_count - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d LWCA capable GPU device(s) detected. <<\n",
            device_count);
    fprintf(stderr,
            ">> gpuDeviceInit (-device=%d) is not a valid"
            " GPU device. <<\n",
            devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  int computeMode = -1, major = 0, minor = 0;
  checkLwdaErrors(lwdaDeviceGetAttribute(&computeMode, lwdaDevAttrComputeMode, devID));
  checkLwdaErrors(lwdaDeviceGetAttribute(&major, lwdaDevAttrComputeCapabilityMajor, devID));
  checkLwdaErrors(lwdaDeviceGetAttribute(&minor, lwdaDevAttrComputeCapabilityMinor, devID));
  if (computeMode == lwdaComputeModeProhibited) {
    fprintf(stderr,
            "Error: device is running in <Compute Mode "
            "Prohibited>, no threads can use lwdaSetDevice().\n");
    return -1;
  }

  if (major < 1) {
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support LWCA.\n");
    exit(EXIT_FAILURE);
  }

  checkLwdaErrors(lwdaSetDevice(devID));
  printf("gpuDeviceInit() LWCA Device [%d]: \"%s\n", devID, _ColwertSMVer2ArchName(major, minor));

  return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId() {
  int lwrrent_device = 0, sm_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  uint64_t max_compute_perf = 0;
  checkLwdaErrors(lwdaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() LWCA error:"
            " no devices supporting LWCA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the best LWCA capable GPU device
  lwrrent_device = 0;

  while (lwrrent_device < device_count) {
    int computeMode = -1, major = 0, minor = 0;
    checkLwdaErrors(lwdaDeviceGetAttribute(&computeMode, lwdaDevAttrComputeMode, lwrrent_device));
    checkLwdaErrors(lwdaDeviceGetAttribute(&major, lwdaDevAttrComputeCapabilityMajor, lwrrent_device));
    checkLwdaErrors(lwdaDeviceGetAttribute(&minor, lwdaDevAttrComputeCapabilityMinor, lwrrent_device));

    // If this GPU is not running on Compute Mode prohibited,
    // then we can add it to the list
    if (computeMode != lwdaComputeModeProhibited) {
      if (major == 9999 && minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc =
            _ColwertSMVer2Cores(major,  minor);
      }
      int multiProcessorCount = 0, clockRate = 0;
      checkLwdaErrors(lwdaDeviceGetAttribute(&multiProcessorCount, lwdaDevAttrMultiProcessorCount, lwrrent_device));
      checkLwdaErrors(lwdaDeviceGetAttribute(&clockRate, lwdaDevAttrClockRate, lwrrent_device));
      uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

      if (compute_perf > max_compute_perf) {
        max_compute_perf = compute_perf;
        max_perf_device = lwrrent_device;
      }
    } else {
      devices_prohibited++;
    }

    ++lwrrent_device;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() LWCA error:"
            " all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}

// Initialization code to find the best LWCA Device
inline int findLwdaDevice(int argc, const char **argv) {
  int devID = 0;

  // If the command-line has a device number specified, use it
  if (checkCmdLineFlag(argc, argv, "device")) {
    devID = getCmdLineArgumentInt(argc, argv, "device=");

    if (devID < 0) {
      printf("Invalid command line parameter\n ");
      exit(EXIT_FAILURE);
    } else {
      devID = gpuDeviceInit(devID);

      if (devID < 0) {
        printf("exiting...\n");
        exit(EXIT_FAILURE);
      }
    }
  } else {
    // Otherwise pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    checkLwdaErrors(lwdaSetDevice(devID));
    int major = 0, minor = 0;
    checkLwdaErrors(lwdaDeviceGetAttribute(&major, lwdaDevAttrComputeCapabilityMajor, devID));
    checkLwdaErrors(lwdaDeviceGetAttribute(&minor, lwdaDevAttrComputeCapabilityMinor, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
           devID, _ColwertSMVer2ArchName(major, minor), major, minor);

  }

  return devID;
}

inline int findIntegratedGPU() {
  int lwrrent_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  checkLwdaErrors(lwdaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr, "LWCA error: no devices supporting LWCA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the integrated GPU which is compute capable
  while (lwrrent_device < device_count) {
    int computeMode = -1, integrated = -1;
    checkLwdaErrors(lwdaDeviceGetAttribute(&computeMode, lwdaDevAttrComputeMode, lwrrent_device));
    checkLwdaErrors(lwdaDeviceGetAttribute(&integrated, lwdaDevAttrIntegrated, lwrrent_device));
    // If GPU is integrated and is not running on Compute Mode prohibited,
    // then lwca can map to GLES resource
    if (integrated && (computeMode != lwdaComputeModeProhibited)) {
      checkLwdaErrors(lwdaSetDevice(lwrrent_device));

      int major = 0, minor = 0;
      checkLwdaErrors(lwdaDeviceGetAttribute(&major, lwdaDevAttrComputeCapabilityMajor, lwrrent_device));
      checkLwdaErrors(lwdaDeviceGetAttribute(&minor, lwdaDevAttrComputeCapabilityMinor, lwrrent_device));
      printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
             lwrrent_device, _ColwertSMVer2ArchName(major, minor), major, minor);

      return lwrrent_device;
    } else {
      devices_prohibited++;
    }

    lwrrent_device++;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "LWCA error:"
            " No GLES-LWCA Interop capable GPU found.\n");
    exit(EXIT_FAILURE);
  }

  return -1;
}

// General check for LWCA GPU SM Capabilities
inline bool checkLwdaCapabilities(int major_version, int minor_version) {
  int dev;
  int major = 0, minor = 0;

  checkLwdaErrors(lwdaGetDevice(&dev));
  checkLwdaErrors(lwdaDeviceGetAttribute(&major, lwdaDevAttrComputeCapabilityMajor, dev));
  checkLwdaErrors(lwdaDeviceGetAttribute(&minor, lwdaDevAttrComputeCapabilityMinor, dev));

  if ((major > major_version) ||
      (major == major_version &&
       minor >= minor_version)) {
    printf("  Device %d: <%16s >, Compute SM %d.%d detected\n", dev,
           _ColwertSMVer2ArchName(major, minor), major, minor);
    return true;
  } else {
    printf(
        "  No GPU device was found that can support "
        "LWCA compute capability %d.%d.\n",
        major_version, minor_version);
    return false;
  }
}
#endif

  // end of LWCA Helper Functions

#endif  // COMMON_HELPER_LWDA_H_
