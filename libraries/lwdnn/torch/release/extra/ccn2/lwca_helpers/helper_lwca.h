/**
 * Copyright 1993-2014 LWPU Corporation.  All rights reserved.
 *
 * Please refer to the LWPU end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// These are LWCA Helper functions for initialization and error checking

#ifndef HELPER_LWDA_H
#define HELPER_LWDA_H

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_string.h>

/*
inline void __ExitInTime(int seconds)
{
    fprintf(stdout, "> exiting in %d seconds: ", seconds);
    fflush(stdout);
    time_t t;
    int count;

    for (t=time(0)+seconds, count=seconds; time(0) < t; count--) {
        fprintf(stdout, "%d...", count);
#if defined(WIN32)
        Sleep(1000);
#else
        sleep(1);
#endif
    }

    fprintf(stdout,"done!\n\n");
    fflush(stdout);
}

#define EXIT_TIME_DELAY 2

inline void EXIT_DELAY(int return_code)
{
    __ExitInTime(EXIT_TIME_DELAY);
    exit(return_code);
}
*/

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// Note, it is required that your SDK sample to include the proper header files, please
// refer the LWCA examples for examples of the needed LWCA headers, which may change depending
// on which LWCA functions are used.

// LWCA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char *_lwdaGetErrorEnum(lwdaError_t error)
{
    switch (error)
    {
        case lwdaSuccess:
            return "lwdaSuccess";

        case lwdaErrorMissingConfiguration:
            return "lwdaErrorMissingConfiguration";

        case lwdaErrorMemoryAllocation:
            return "lwdaErrorMemoryAllocation";

        case lwdaErrorInitializationError:
            return "lwdaErrorInitializationError";

        case lwdaErrorLaunchFailure:
            return "lwdaErrorLaunchFailure";

        case lwdaErrorPriorLaunchFailure:
            return "lwdaErrorPriorLaunchFailure";

        case lwdaErrorLaunchTimeout:
            return "lwdaErrorLaunchTimeout";

        case lwdaErrorLaunchOutOfResources:
            return "lwdaErrorLaunchOutOfResources";

        case lwdaErrorIlwalidDeviceFunction:
            return "lwdaErrorIlwalidDeviceFunction";

        case lwdaErrorIlwalidConfiguration:
            return "lwdaErrorIlwalidConfiguration";

        case lwdaErrorIlwalidDevice:
            return "lwdaErrorIlwalidDevice";

        case lwdaErrorIlwalidValue:
            return "lwdaErrorIlwalidValue";

        case lwdaErrorIlwalidPitchValue:
            return "lwdaErrorIlwalidPitchValue";

        case lwdaErrorIlwalidSymbol:
            return "lwdaErrorIlwalidSymbol";

        case lwdaErrorMapBufferObjectFailed:
            return "lwdaErrorMapBufferObjectFailed";

        case lwdaErrorUnmapBufferObjectFailed:
            return "lwdaErrorUnmapBufferObjectFailed";

        case lwdaErrorIlwalidHostPointer:
            return "lwdaErrorIlwalidHostPointer";

        case lwdaErrorIlwalidDevicePointer:
            return "lwdaErrorIlwalidDevicePointer";

        case lwdaErrorIlwalidTexture:
            return "lwdaErrorIlwalidTexture";

        case lwdaErrorIlwalidTextureBinding:
            return "lwdaErrorIlwalidTextureBinding";

        case lwdaErrorIlwalidChannelDescriptor:
            return "lwdaErrorIlwalidChannelDescriptor";

        case lwdaErrorIlwalidMemcpyDirection:
            return "lwdaErrorIlwalidMemcpyDirection";

        case lwdaErrorAddressOfConstant:
            return "lwdaErrorAddressOfConstant";

        case lwdaErrorTextureFetchFailed:
            return "lwdaErrorTextureFetchFailed";

        case lwdaErrorTextureNotBound:
            return "lwdaErrorTextureNotBound";

        case lwdaErrorSynchronizationError:
            return "lwdaErrorSynchronizationError";

        case lwdaErrorIlwalidFilterSetting:
            return "lwdaErrorIlwalidFilterSetting";

        case lwdaErrorIlwalidNormSetting:
            return "lwdaErrorIlwalidNormSetting";

        case lwdaErrorMixedDeviceExelwtion:
            return "lwdaErrorMixedDeviceExelwtion";

        case lwdaErrorLwdartUnloading:
            return "lwdaErrorLwdartUnloading";

        case lwdaErrorUnknown:
            return "lwdaErrorUnknown";

        case lwdaErrorNotYetImplemented:
            return "lwdaErrorNotYetImplemented";

        case lwdaErrorMemoryValueTooLarge:
            return "lwdaErrorMemoryValueTooLarge";

        case lwdaErrorIlwalidResourceHandle:
            return "lwdaErrorIlwalidResourceHandle";

        case lwdaErrorNotReady:
            return "lwdaErrorNotReady";

        case lwdaErrorInsufficientDriver:
            return "lwdaErrorInsufficientDriver";

        case lwdaErrorSetOnActiveProcess:
            return "lwdaErrorSetOnActiveProcess";

        case lwdaErrorIlwalidSurface:
            return "lwdaErrorIlwalidSurface";

        case lwdaErrorNoDevice:
            return "lwdaErrorNoDevice";

        case lwdaErrorECLWncorrectable:
            return "lwdaErrorECLWncorrectable";

        case lwdaErrorSharedObjectSymbolNotFound:
            return "lwdaErrorSharedObjectSymbolNotFound";

        case lwdaErrorSharedObjectInitFailed:
            return "lwdaErrorSharedObjectInitFailed";

        case lwdaErrorUnsupportedLimit:
            return "lwdaErrorUnsupportedLimit";

        case lwdaErrorDuplicateVariableName:
            return "lwdaErrorDuplicateVariableName";

        case lwdaErrorDuplicateTextureName:
            return "lwdaErrorDuplicateTextureName";

        case lwdaErrorDuplicateSurfaceName:
            return "lwdaErrorDuplicateSurfaceName";

        case lwdaErrorDevicesUnavailable:
            return "lwdaErrorDevicesUnavailable";

        case lwdaErrorIlwalidKernelImage:
            return "lwdaErrorIlwalidKernelImage";

        case lwdaErrorNoKernelImageForDevice:
            return "lwdaErrorNoKernelImageForDevice";

        case lwdaErrorIncompatibleDriverContext:
            return "lwdaErrorIncompatibleDriverContext";

        case lwdaErrorPeerAccessAlreadyEnabled:
            return "lwdaErrorPeerAccessAlreadyEnabled";

        case lwdaErrorPeerAccessNotEnabled:
            return "lwdaErrorPeerAccessNotEnabled";

        case lwdaErrorDeviceAlreadyInUse:
            return "lwdaErrorDeviceAlreadyInUse";

        case lwdaErrorProfilerDisabled:
            return "lwdaErrorProfilerDisabled";

        case lwdaErrorProfilerNotInitialized:
            return "lwdaErrorProfilerNotInitialized";

        case lwdaErrorProfilerAlreadyStarted:
            return "lwdaErrorProfilerAlreadyStarted";

        case lwdaErrorProfilerAlreadyStopped:
            return "lwdaErrorProfilerAlreadyStopped";

#if __LWDA_API_VERSION >= 0x4000

        case lwdaErrorAssert:
            return "lwdaErrorAssert";

        case lwdaErrorTooManyPeers:
            return "lwdaErrorTooManyPeers";

        case lwdaErrorHostMemoryAlreadyRegistered:
            return "lwdaErrorHostMemoryAlreadyRegistered";

        case lwdaErrorHostMemoryNotRegistered:
            return "lwdaErrorHostMemoryNotRegistered";
#endif

        case lwdaErrorStartupFailure:
            return "lwdaErrorStartupFailure";

        case lwdaErrorApiFailureBase:
            return "lwdaErrorApiFailureBase";
    }

    return "<unknown>";
}
#endif

#ifdef __lwda_lwda_h__
// LWCA Driver API errors
static const char *_lwdaGetErrorEnum(LWresult error)
{
    switch (error)
    {
        case LWDA_SUCCESS:
            return "LWDA_SUCCESS";

        case LWDA_ERROR_ILWALID_VALUE:
            return "LWDA_ERROR_ILWALID_VALUE";

        case LWDA_ERROR_OUT_OF_MEMORY:
            return "LWDA_ERROR_OUT_OF_MEMORY";

        case LWDA_ERROR_NOT_INITIALIZED:
            return "LWDA_ERROR_NOT_INITIALIZED";

        case LWDA_ERROR_DEINITIALIZED:
            return "LWDA_ERROR_DEINITIALIZED";

        case LWDA_ERROR_PROFILER_DISABLED:
            return "LWDA_ERROR_PROFILER_DISABLED";

        case LWDA_ERROR_PROFILER_NOT_INITIALIZED:
            return "LWDA_ERROR_PROFILER_NOT_INITIALIZED";

        case LWDA_ERROR_PROFILER_ALREADY_STARTED:
            return "LWDA_ERROR_PROFILER_ALREADY_STARTED";

        case LWDA_ERROR_PROFILER_ALREADY_STOPPED:
            return "LWDA_ERROR_PROFILER_ALREADY_STOPPED";

        case LWDA_ERROR_NO_DEVICE:
            return "LWDA_ERROR_NO_DEVICE";

        case LWDA_ERROR_ILWALID_DEVICE:
            return "LWDA_ERROR_ILWALID_DEVICE";

        case LWDA_ERROR_ILWALID_IMAGE:
            return "LWDA_ERROR_ILWALID_IMAGE";

        case LWDA_ERROR_ILWALID_CONTEXT:
            return "LWDA_ERROR_ILWALID_CONTEXT";

        case LWDA_ERROR_CONTEXT_ALREADY_LWRRENT:
            return "LWDA_ERROR_CONTEXT_ALREADY_LWRRENT";

        case LWDA_ERROR_MAP_FAILED:
            return "LWDA_ERROR_MAP_FAILED";

        case LWDA_ERROR_UNMAP_FAILED:
            return "LWDA_ERROR_UNMAP_FAILED";

        case LWDA_ERROR_ARRAY_IS_MAPPED:
            return "LWDA_ERROR_ARRAY_IS_MAPPED";

        case LWDA_ERROR_ALREADY_MAPPED:
            return "LWDA_ERROR_ALREADY_MAPPED";

        case LWDA_ERROR_NO_BINARY_FOR_GPU:
            return "LWDA_ERROR_NO_BINARY_FOR_GPU";

        case LWDA_ERROR_ALREADY_ACQUIRED:
            return "LWDA_ERROR_ALREADY_ACQUIRED";

        case LWDA_ERROR_NOT_MAPPED:
            return "LWDA_ERROR_NOT_MAPPED";

        case LWDA_ERROR_NOT_MAPPED_AS_ARRAY:
            return "LWDA_ERROR_NOT_MAPPED_AS_ARRAY";

        case LWDA_ERROR_NOT_MAPPED_AS_POINTER:
            return "LWDA_ERROR_NOT_MAPPED_AS_POINTER";

        case LWDA_ERROR_ECC_UNCORRECTABLE:
            return "LWDA_ERROR_ECC_UNCORRECTABLE";

        case LWDA_ERROR_UNSUPPORTED_LIMIT:
            return "LWDA_ERROR_UNSUPPORTED_LIMIT";

        case LWDA_ERROR_CONTEXT_ALREADY_IN_USE:
            return "LWDA_ERROR_CONTEXT_ALREADY_IN_USE";

        case LWDA_ERROR_ILWALID_SOURCE:
            return "LWDA_ERROR_ILWALID_SOURCE";

        case LWDA_ERROR_FILE_NOT_FOUND:
            return "LWDA_ERROR_FILE_NOT_FOUND";

        case LWDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return "LWDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";

        case LWDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return "LWDA_ERROR_SHARED_OBJECT_INIT_FAILED";

        case LWDA_ERROR_OPERATING_SYSTEM:
            return "LWDA_ERROR_OPERATING_SYSTEM";

        case LWDA_ERROR_ILWALID_HANDLE:
            return "LWDA_ERROR_ILWALID_HANDLE";

        case LWDA_ERROR_NOT_FOUND:
            return "LWDA_ERROR_NOT_FOUND";

        case LWDA_ERROR_NOT_READY:
            return "LWDA_ERROR_NOT_READY";

        case LWDA_ERROR_LAUNCH_FAILED:
            return "LWDA_ERROR_LAUNCH_FAILED";

        case LWDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return "LWDA_ERROR_LAUNCH_OUT_OF_RESOURCES";

        case LWDA_ERROR_LAUNCH_TIMEOUT:
            return "LWDA_ERROR_LAUNCH_TIMEOUT";

        case LWDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
            return "LWDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";

        case LWDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            return "LWDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";

        case LWDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            return "LWDA_ERROR_PEER_ACCESS_NOT_ENABLED";

        case LWDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            return "LWDA_ERROR_PRIMARY_CONTEXT_ACTIVE";

        case LWDA_ERROR_CONTEXT_IS_DESTROYED:
            return "LWDA_ERROR_CONTEXT_IS_DESTROYED";

        case LWDA_ERROR_ASSERT:
            return "LWDA_ERROR_ASSERT";

        case LWDA_ERROR_TOO_MANY_PEERS:
            return "LWDA_ERROR_TOO_MANY_PEERS";

        case LWDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            return "LWDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";

        case LWDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            return "LWDA_ERROR_HOST_MEMORY_NOT_REGISTERED";

        case LWDA_ERROR_UNKNOWN:
            return "LWDA_ERROR_UNKNOWN";
    }

    return "<unknown>";
}
#endif

#ifdef LWBLAS_API_H_
// lwBLAS API errors
static const char *_lwdaGetErrorEnum(lwblasStatus_t error)
{
    switch (error)
    {
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
    }

    return "<unknown>";
}
#endif

#ifdef _LWFFT_H_
// lwFFT API errors
static const char *_lwdaGetErrorEnum(lwfftResult error)
{
    switch (error)
    {
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
    }

    return "<unknown>";
}
#endif


#ifdef LWSPARSEAPI
// lwSPARSE API errors
static const char *_lwdaGetErrorEnum(lwsparseStatus_t error)
{
    switch (error)
    {
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

#ifdef LWRAND_H_
// lwRAND API errors
static const char *_lwdaGetErrorEnum(lwrandStatus_t error)
{
    switch (error)
    {
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

#ifdef LW_NPPIDEFS_H
// NPP API errors
static const char *_lwdaGetErrorEnum(NppStatus error)
{
    switch (error)
    {
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
    }

    return "<unknown>";
}
#endif

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET lwdaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "LWCA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _lwdaGetErrorEnum(result), func);
        DEVICE_RESET
        // Make sure we call LWCA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper LWCA error strings in the event that a LWCA host call returns an error
#define checkLwdaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

// This will output the proper error string when calling lwdaGetLastError
#define getLastLwdaError(msg)      __getLastLwdaError (msg, __FILE__, __LINE__)

inline void __getLastLwdaError(const char *errorMessage, const char *file, const int line)
{
    lwdaError_t err = lwdaGetLastError();

    if (lwdaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastLwdaError() LWCA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, lwdaGetErrorString(err));
        DEVICE_RESET
        exit(EXIT_FAILURE);
    }
}
#endif

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// Beginning of GPU Architecture definitions
inline int _ColwertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
    return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions

#ifdef __LWDA_RUNTIME_H__
// General GPU Device LWCA Initialization
inline int gpuDeviceInit(int devID)
{
    int device_count;
    checkLwdaErrors(lwdaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuDeviceInit() LWCA error: no devices supporting LWCA.\n");
        exit(EXIT_FAILURE);
    }

    if (devID < 0)
    {
        devID = 0;
    }

    if (devID > device_count-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d LWCA capable GPU device(s) detected. <<\n", device_count);
        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
        fprintf(stderr, "\n");
        return -devID;
    }

    lwdaDeviceProp deviceProp;
    checkLwdaErrors(lwdaGetDeviceProperties(&deviceProp, devID));

    if (deviceProp.computeMode == lwdaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::lwdaSetDevice().\n");
        return -1;
    }

    if (deviceProp.major < 1)
    {
        fprintf(stderr, "gpuDeviceInit(): GPU device does not support LWCA.\n");
        exit(EXIT_FAILURE);
    }

    checkLwdaErrors(lwdaSetDevice(devID));
    printf("gpuDeviceInit() LWCA Device [%d]: \"%s\n", devID, deviceProp.name);

    return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId()
{
    int lwrrent_device     = 0, sm_per_multiproc  = 0;
    int max_perf_device    = 0;
    int device_count       = 0, best_SM_arch      = 0;
    
    unsigned long long max_compute_perf = 0;
    lwdaDeviceProp deviceProp;
    lwdaGetDeviceCount(&device_count);
    
    checkLwdaErrors(lwdaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuGetMaxGflopsDeviceId() LWCA error: no devices supporting LWCA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the best major SM Architecture GPU device
    while (lwrrent_device < device_count)
    {
        lwdaGetDeviceProperties(&deviceProp, lwrrent_device);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if (deviceProp.computeMode != lwdaComputeModeProhibited)
        {
            if (deviceProp.major > 0 && deviceProp.major < 9999)
            {
                best_SM_arch = MAX(best_SM_arch, deviceProp.major);
            }
        }

        lwrrent_device++;
    }

    // Find the best LWCA capable GPU device
    lwrrent_device = 0;

    while (lwrrent_device < device_count)
    {
        lwdaGetDeviceProperties(&deviceProp, lwrrent_device);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if (deviceProp.computeMode != lwdaComputeModeProhibited)
        {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc = _ColwertSMVer2Cores(deviceProp.major, deviceProp.minor);
            }

            unsigned long long compute_perf  = (unsigned long long) deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

            if (compute_perf  > max_compute_perf)
            {
                // If we find GPU with SM major > 2, search only these
                if (best_SM_arch > 2)
                {
                    // If our device==dest_SM_arch, choose this, or else pass
                    if (deviceProp.major == best_SM_arch)
                    {
                        max_compute_perf  = compute_perf;
                        max_perf_device   = lwrrent_device;
                    }
                }
                else
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = lwrrent_device;
                }
            }
        }

        ++lwrrent_device;
    }

    return max_perf_device;
}


// Initialization code to find the best LWCA Device
inline int findLwdaDevice(int argc, const char **argv)
{
    lwdaDeviceProp deviceProp;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, argv, "device=");

        if (devID < 0)
        {
            printf("Invalid command line parameter\n ");
            exit(EXIT_FAILURE);
        }
        else
        {
            devID = gpuDeviceInit(devID);

            if (devID < 0)
            {
                printf("exiting...\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        checkLwdaErrors(lwdaSetDevice(devID));
        checkLwdaErrors(lwdaGetDeviceProperties(&deviceProp, devID));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    return devID;
}

// General check for LWCA GPU SM Capabilities
inline bool checkLwdaCapabilities(int major_version, int minor_version)
{
    lwdaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

    checkLwdaErrors(lwdaGetDevice(&dev));
    checkLwdaErrors(lwdaGetDeviceProperties(&deviceProp, dev));

    if ((deviceProp.major > major_version) ||
        (deviceProp.major == major_version && deviceProp.minor >= minor_version))
    {
        printf("  GPU Device %d: <%16s >, Compute SM %d.%d detected\n", dev, deviceProp.name, deviceProp.major, deviceProp.minor);
        return true;
    }
    else
    {
        printf("  No GPU device was found that can support LWCA compute capability %d.%d.\n", major_version, minor_version);
        return false;
    }
}
#endif

// end of LWCA Helper Functions


#endif
