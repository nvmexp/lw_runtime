#pragma once

#include <assert.h>
#include <stdio.h>

#include <driver_types.h>

#include "complex.h"

#if 0
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
#endif

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

        /* Since LWCA 4.0*/
    case lwdaErrorAssert:
        return "lwdaErrorAssert";

    case lwdaErrorTooManyPeers:
        return "lwdaErrorTooManyPeers";

    case lwdaErrorHostMemoryAlreadyRegistered:
        return "lwdaErrorHostMemoryAlreadyRegistered";

    case lwdaErrorHostMemoryNotRegistered:
        return "lwdaErrorHostMemoryNotRegistered";

        /* Since LWCA 5.0 */
    case lwdaErrorOperatingSystem:
        return "lwdaErrorOperatingSystem";

    case lwdaErrorPeerAccessUnsupported:
        return "lwdaErrorPeerAccessUnsupported";

    case lwdaErrorLaunchMaxDepthExceeded:
        return "lwdaErrorLaunchMaxDepthExceeded";

    case lwdaErrorLaunchFileScopedTex:
        return "lwdaErrorLaunchFileScopedTex";

    case lwdaErrorLaunchFileScopedSurf:
        return "lwdaErrorLaunchFileScopedSurf";

    case lwdaErrorSyncDepthExceeded:
        return "lwdaErrorSyncDepthExceeded";

    case lwdaErrorLaunchPendingCountExceeded:
        return "lwdaErrorLaunchPendingCountExceeded";

    case lwdaErrorNotPermitted:
        return "lwdaErrorNotPermitted";

    case lwdaErrorNotSupported:
        return "lwdaErrorNotSupported";

        /* Since LWCA 6.0 */
    case lwdaErrorHardwareStackError:
        return "lwdaErrorHardwareStackError";

    case lwdaErrorIllegalInstruction:
        return "lwdaErrorIllegalInstruction";

    case lwdaErrorMisalignedAddress:
        return "lwdaErrorMisalignedAddress";

    case lwdaErrorIlwalidAddressSpace:
        return "lwdaErrorIlwalidAddressSpace";

    case lwdaErrorIlwalidPc:
        return "lwdaErrorIlwalidPc";

    case lwdaErrorIllegalAddress:
        return "lwdaErrorIllegalAddress";

        /* Since LWCA 6.5*/
    case lwdaErrorIlwalidPtx:
        return "lwdaErrorIlwalidPtx";

    case lwdaErrorIlwalidGraphicsContext:
        return "lwdaErrorIlwalidGraphicsContext";

    case lwdaErrorStartupFailure:
        return "lwdaErrorStartupFailure";

    case lwdaErrorApiFailureBase:
        return "lwdaErrorApiFailureBase";

        /* Since LWCA 8.0*/        
    case lwdaErrorLwlinkUncorrectable :   
        return "lwdaErrorLwlinkUncorrectable";
    }

    return "<unknown>";
}
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

#define DEVICE_RESET lwdaDeviceReset()

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "LWCA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(result), _lwdaGetErrorEnum(result), func);
        DEVICE_RESET;
        // Make sure we call LWCA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}
inline void __getLastLwdaError(const char *errorMessage, const char *file, const int line)
{
    lwdaError_t err = lwdaGetLastError();

    if (lwdaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastLwdaError() LWCA error : %s : (%d) %s.\n",
            file, line, errorMessage, (int)err, lwdaGetErrorString(err));
        DEVICE_RESET;
        exit(EXIT_FAILURE);
    }
}

// This will output the proper LWCA error strings in the event that a LWCA host call returns an error
#define checkLwdaErrors(val)		check ( (val), #val, __FILE__, __LINE__ )
// This will output the proper error string when calling lwdaGetLastError
#define getLastLwdaError(msg)		__getLastLwdaError (msg, __FILE__, __LINE__)
