/**
 * Copyright 1993-2013 LWPU Corporation.  All rights reserved.
 *
 * Please refer to the LWPU end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Helper functions for LWCA Driver API error handling (make sure that LWDA_H is included in your projects)
#ifndef HELPER_LWDA_DRVAPI_H
#define HELPER_LWDA_DRVAPI_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_string.h>
#include <drvapi_error_string.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

////////////////////////////////////////////////////////////////////////////////
// These are LWCA Helper functions

// add a level of protection to the LWCA SDK samples, let's force samples to explicitly include LWCA.H
#ifdef  __lwda_lwda_h__
// This will output the proper LWCA error strings in the event that a LWCA host call returns an error
#ifndef checkLwdaErrors
#define checkLwdaErrors(err)  __checkLwdaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkLwdaErrors(LWresult err, const char *file, const int line)
{
    if (LWDA_SUCCESS != err)
    {
        fprintf(stderr, "checkLwdaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getLwdaDrvErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#endif

#ifdef getLastLwdaDrvErrorMsg
#undef getLastLwdaDrvErrorMsg
#endif

#define getLastLwdaDrvErrorMsg(msg)           __getLastLwdaDrvErrorMsg  (msg, __FILE__, __LINE__)

inline void __getLastLwdaDrvErrorMsg(const char *msg, const char *file, const int line)
{
    LWresult err = lwCtxSynchronize();

    if (LWDA_SUCCESS != err)
    {
        fprintf(stderr, "getLastLwdaDrvErrorMsg -> %s", msg);
        fprintf(stderr, "getLastLwdaDrvErrorMsg -> lwCtxSynchronize API error = %04d \"%s\" in file <%s>, line %i.\n",
                err, getLwdaDrvErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// This function wraps the LWCA Driver API into a template function
template <class T>
inline void getLwdaAttribute(T *attribute, LWdevice_attribute device_attribute, int device)
{
    LWresult error_result = lwDeviceGetAttribute(attribute, device_attribute, device);

    if (error_result != LWDA_SUCCESS)
    {
        printf("lwDeviceGetAttribute returned %d\n-> %s\n", (int)error_result, getLwdaDrvErrorString(error_result));
        exit(EXIT_SUCCESS);
    }
}
#endif

// Beginning of GPU Architecture definitions
inline int _ColwertSMVer2CoresDRV(int major, int minor)
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

#ifdef __lwda_lwda_h__
// General GPU Device LWCA Initialization
inline int gpuDeviceInitDRV(int ARGC, const char **ARGV)
{
    int lwDevice = 0;
    int deviceCount = 0;
    LWresult err = lwInit(0);

    if (LWDA_SUCCESS == err)
    {
        checkLwdaErrors(lwDeviceGetCount(&deviceCount));
    }

    if (deviceCount == 0)
    {
        fprintf(stderr, "lwdaDeviceInit error: no devices supporting LWCA\n");
        exit(EXIT_FAILURE);
    }

    int dev = 0;
    dev = getCmdLineArgumentInt(ARGC, (const char **) ARGV, "device=");

    if (dev < 0)
    {
        dev = 0;
    }

    if (dev > deviceCount-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d LWCA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> lwdaDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
        fprintf(stderr, "\n");
        return -dev;
    }

    checkLwdaErrors(lwDeviceGet(&lwDevice, dev));
    char name[100];
    lwDeviceGetName(name, 100, lwDevice);

    int computeMode;
    getLwdaAttribute<int>(&computeMode, LW_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev);

    if (computeMode == LW_COMPUTEMODE_PROHIBITED)
    {
        fprintf(stderr, "Error: device is running in <LW_COMPUTEMODE_PROHIBITED>, no threads can use this LWCA Device.\n");
        return -1;
    }

    if (checkCmdLineFlag(ARGC, (const char **) ARGV, "quiet") == false)
    {
        printf("gpuDeviceInitDRV() Using LWCA Device [%d]: %s\n", dev, name);
    }

    return dev;
}

// This function returns the best GPU based on performance
inline int gpuGetMaxGflopsDeviceIdDRV()
{
    LWdevice lwrrent_device = 0, max_perf_device = 0;
    int device_count        = 0, sm_per_multiproc = 0;
    int max_compute_perf    = 0, best_SM_arch     = 0;
    int major = 0, minor = 0   , multiProcessorCount, clockRate;

    lwInit(0);
    checkLwdaErrors(lwDeviceGetCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuGetMaxGflopsDeviceIdDRV error: no devices supporting LWCA\n");
        exit(EXIT_FAILURE);
    }

    // Find the best major SM Architecture GPU device
    while (lwrrent_device < device_count)
    {
        checkLwdaErrors(lwDeviceComputeCapability(&major, &minor, lwrrent_device));

        if (major > 0 && major < 9999)
        {
            best_SM_arch = MAX(best_SM_arch, major);
        }

        lwrrent_device++;
    }

    // Find the best LWCA capable GPU device
    lwrrent_device = 0;

    while (lwrrent_device < device_count)
    {
        checkLwdaErrors(lwDeviceGetAttribute(&multiProcessorCount,
                                             LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                             lwrrent_device));
        checkLwdaErrors(lwDeviceGetAttribute(&clockRate,
                                             LW_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                             lwrrent_device));
        checkLwdaErrors(lwDeviceComputeCapability(&major, &minor, lwrrent_device));

        int computeMode;
        getLwdaAttribute<int>(&computeMode, LW_DEVICE_ATTRIBUTE_COMPUTE_MODE, lwrrent_device);

        if (computeMode != LW_COMPUTEMODE_PROHIBITED)
        {
            if (major == 9999 && minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc = _ColwertSMVer2CoresDRV(major, minor);
            }

            int compute_perf  = multiProcessorCount * sm_per_multiproc * clockRate;

            if (compute_perf  > max_compute_perf)
            {
                // If we find GPU with SM major > 2, search only these
                if (best_SM_arch > 2)
                {
                    // If our device==dest_SM_arch, choose this, or else pass
                    if (major == best_SM_arch)
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

// This function returns the best Graphics GPU based on performance
inline int gpuGetMaxGflopsGLDeviceIdDRV()
{
    LWdevice lwrrent_device = 0, max_perf_device = 0;
    int device_count     = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, best_SM_arch     = 0;
    int major = 0, minor = 0, multiProcessorCount, clockRate;
    int bTCC = 0;
    char deviceName[256];

    lwInit(0);
    checkLwdaErrors(lwDeviceGetCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuGetMaxGflopsGLDeviceIdDRV error: no devices supporting LWCA\n");
        exit(EXIT_FAILURE);
    }

    // Find the best major SM Architecture GPU device that are graphics devices
    while (lwrrent_device < device_count)
    {
        checkLwdaErrors(lwDeviceGetName(deviceName, 256, lwrrent_device));
        checkLwdaErrors(lwDeviceComputeCapability(&major, &minor, lwrrent_device));

#if LWDA_VERSION >= 3020
        checkLwdaErrors(lwDeviceGetAttribute(&bTCC,  LW_DEVICE_ATTRIBUTE_TCC_DRIVER, lwrrent_device));
#else

        // Assume a Tesla GPU is running in TCC if we are running LWCA 3.1
        if (deviceName[0] == 'T')
        {
            bTCC = 1;
        }

#endif

        int computeMode;
        getLwdaAttribute<int>(&computeMode, LW_DEVICE_ATTRIBUTE_COMPUTE_MODE, lwrrent_device);

        if (computeMode != LW_COMPUTEMODE_PROHIBITED)
        {
            if (!bTCC)
            {
                if (major > 0 && major < 9999)
                {
                    best_SM_arch = MAX(best_SM_arch, major);
                }
            }
        }

        lwrrent_device++;
    }

    // Find the best LWCA capable GPU device
    lwrrent_device = 0;

    while (lwrrent_device < device_count)
    {
        checkLwdaErrors(lwDeviceGetAttribute(&multiProcessorCount,
                                             LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                             lwrrent_device));
        checkLwdaErrors(lwDeviceGetAttribute(&clockRate,
                                             LW_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                             lwrrent_device));
        checkLwdaErrors(lwDeviceComputeCapability(&major, &minor, lwrrent_device));

#if LWDA_VERSION >= 3020
        checkLwdaErrors(lwDeviceGetAttribute(&bTCC,  LW_DEVICE_ATTRIBUTE_TCC_DRIVER, lwrrent_device));
#else

        // Assume a Tesla GPU is running in TCC if we are running LWCA 3.1
        if (deviceName[0] == 'T')
        {
            bTCC = 1;
        }

#endif

        int computeMode;
        getLwdaAttribute<int>(&computeMode, LW_DEVICE_ATTRIBUTE_COMPUTE_MODE, lwrrent_device);

        if (computeMode != LW_COMPUTEMODE_PROHIBITED)
        {
            if (major == 9999 && minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc = _ColwertSMVer2CoresDRV(major, minor);
            }

            // If this is a Tesla based GPU and SM 2.0, and TCC is disabled, this is a contendor
            if (!bTCC)   // Is this GPU running the TCC driver?  If so we pass on this
            {
                int compute_perf  = multiProcessorCount * sm_per_multiproc * clockRate;

                if (compute_perf  > max_compute_perf)
                {
                    // If we find GPU with SM major > 2, search only these
                    if (best_SM_arch > 2)
                    {
                        // If our device = dest_SM_arch, then we pick this one
                        if (major == best_SM_arch)
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
        }

        ++lwrrent_device;
    }

    return max_perf_device;
}

// General initialization call to pick the best LWCA Device
inline LWdevice findLwdaDeviceDRV(int argc, const char **argv)
{
    LWdevice lwDevice;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = gpuDeviceInitDRV(argc, argv);

        if (devID < 0)
        {
            printf("exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        char name[100];
        devID = gpuGetMaxGflopsDeviceIdDRV();
        checkLwdaErrors(lwDeviceGet(&lwDevice, devID));
        lwDeviceGetName(name, 100, lwDevice);
        printf("> Using LWCA Device [%d]: %s\n", devID, name);
    }

    lwDeviceGet(&lwDevice, devID);

    return lwDevice;
}

// This function will pick the best LWCA device available with OpenGL interop
inline LWdevice findLwdaGLDeviceDRV(int argc, const char **argv)
{
    LWdevice lwDevice;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = gpuDeviceInitDRV(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("no LWCA capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }
    else
    {
        char name[100];
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsGLDeviceIdDRV();
        checkLwdaErrors(lwDeviceGet(&lwDevice, devID));
        lwDeviceGetName(name, 100, lwDevice);
        printf("> Using LWCA/GL Device [%d]: %s\n", devID, name);
    }

    return devID;
}

// General check for LWCA GPU SM Capabilities
inline bool checkLwdaCapabilitiesDRV(int major_version, int minor_version, int devID)
{
    LWdevice lwDevice;
    char name[256];
    int major = 0, minor = 0;

    checkLwdaErrors(lwDeviceGet(&lwDevice, devID));
    checkLwdaErrors(lwDeviceGetName(name, 100, lwDevice));
    checkLwdaErrors(lwDeviceComputeCapability(&major, &minor, devID));

    if ((major > major_version) ||
        (major == major_version && minor >= minor_version))
    {
        printf("> Device %d: <%16s >, Compute SM %d.%d detected\n", devID, name, major, minor);
        return true;
    }
    else
    {
        printf("No GPU device was found that can support LWCA compute capability %d.%d.\n", major_version, minor_version);
        return false;
    }
}
#endif

// end of LWCA Helper Functions

#endif
