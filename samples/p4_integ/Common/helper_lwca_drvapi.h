/* Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
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

// Helper functions for LWCA Driver API error handling (make sure that LWDA_H is
// included in your projects)
#ifndef COMMON_HELPER_LWDA_DRVAPI_H_
#define COMMON_HELPER_LWDA_DRVAPI_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cstring>
#include <sstream>

#include <helper_string.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#ifndef COMMON_HELPER_LWDA_H_
inline int ftoi(float value) {
  return (value >= 0 ? static_cast<int>(value + 0.5)
                     : static_cast<int>(value - 0.5));
}
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

////////////////////////////////////////////////////////////////////////////////
// These are LWCA Helper functions

// add a level of protection to the LWCA SDK samples, let's force samples to
// explicitly include LWCA.H
#ifdef __lwda_lwda_h__
// This will output the proper LWCA error strings in the event that a LWCA host
// call returns an error
#ifndef checkLwdaErrors
#define checkLwdaErrors(err) __checkLwdaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkLwdaErrors(LWresult err, const char *file, const int line) {
  if (LWDA_SUCCESS != err) {
    const char *errorStr = NULL;
    lwGetErrorString(err, &errorStr);
    fprintf(stderr,
            "checkLwdaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}
#endif

// This function wraps the LWCA Driver API into a template function
template <class T>
inline void getLwdaAttribute(T *attribute, LWdevice_attribute device_attribute,
                             int device) {
  checkLwdaErrors(lwDeviceGetAttribute(attribute, device_attribute, device));
}
#endif

// Beginning of GPU Architecture definitions
inline int _ColwertSMVer2CoresDRV(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the #
  // of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM
             // minor version
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
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one to run
  // properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}
// end of GPU Architecture definitions

#ifdef __lwda_lwda_h__
// General GPU Device LWCA Initialization
inline int gpuDeviceInitDRV(int ARGC, const char **ARGV) {
  int lwDevice = 0;
  int deviceCount = 0;
  checkLwdaErrors(lwInit(0));

  checkLwdaErrors(lwDeviceGetCount(&deviceCount));

  if (deviceCount == 0) {
    fprintf(stderr, "lwdaDeviceInit error: no devices supporting LWCA\n");
    exit(EXIT_FAILURE);
  }

  int dev = 0;
  dev = getCmdLineArgumentInt(ARGC, (const char **)ARGV, "device=");

  if (dev < 0) {
    dev = 0;
  }

  if (dev > deviceCount - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d LWCA capable GPU device(s) detected. <<\n",
            deviceCount);
    fprintf(stderr,
            ">> lwdaDeviceInit (-device=%d) is not a valid GPU device. <<\n",
            dev);
    fprintf(stderr, "\n");
    return -dev;
  }

  checkLwdaErrors(lwDeviceGet(&lwDevice, dev));
  char name[100];
  checkLwdaErrors(lwDeviceGetName(name, 100, lwDevice));

  int computeMode;
  getLwdaAttribute<int>(&computeMode, LW_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev);

  if (computeMode == LW_COMPUTEMODE_PROHIBITED) {
    fprintf(stderr,
            "Error: device is running in <LW_COMPUTEMODE_PROHIBITED>, no "
            "threads can use this LWCA Device.\n");
    return -1;
  }

  if (checkCmdLineFlag(ARGC, (const char **)ARGV, "quiet") == false) {
    printf("gpuDeviceInitDRV() Using LWCA Device [%d]: %s\n", dev, name);
  }

  return dev;
}

// This function returns the best GPU based on performance
inline int gpuGetMaxGflopsDeviceIdDRV() {
  LWdevice lwrrent_device = 0;
  LWdevice max_perf_device = 0;
  int device_count = 0;
  int sm_per_multiproc = 0;
  unsigned long long max_compute_perf = 0;
  int major = 0;
  int minor = 0;
  int multiProcessorCount;
  int clockRate;
  int devices_prohibited = 0;

  lwInit(0);
  checkLwdaErrors(lwDeviceGetCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceIdDRV error: no devices supporting LWCA\n");
    exit(EXIT_FAILURE);
  }

  // Find the best LWCA capable GPU device
  lwrrent_device = 0;

  while (lwrrent_device < device_count) {
    checkLwdaErrors(lwDeviceGetAttribute(
        &multiProcessorCount, LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        lwrrent_device));
    checkLwdaErrors(lwDeviceGetAttribute(
        &clockRate, LW_DEVICE_ATTRIBUTE_CLOCK_RATE, lwrrent_device));
    checkLwdaErrors(lwDeviceGetAttribute(
        &major, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, lwrrent_device));
    checkLwdaErrors(lwDeviceGetAttribute(
        &minor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, lwrrent_device));

    int computeMode;
    getLwdaAttribute<int>(&computeMode, LW_DEVICE_ATTRIBUTE_COMPUTE_MODE,
                          lwrrent_device);

    if (computeMode != LW_COMPUTEMODE_PROHIBITED) {
      if (major == 9999 && minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc = _ColwertSMVer2CoresDRV(major, minor);
      }

      unsigned long long compute_perf =
          (unsigned long long)(multiProcessorCount * sm_per_multiproc *
                               clockRate);

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
            "gpuGetMaxGflopsDeviceIdDRV error: all devices have compute mode "
            "prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}

// General initialization call to pick the best LWCA Device
inline LWdevice findLwdaDeviceDRV(int argc, const char **argv) {
  LWdevice lwDevice;
  int devID = 0;

  // If the command-line has a device number specified, use it
  if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
    devID = gpuDeviceInitDRV(argc, argv);

    if (devID < 0) {
      printf("exiting...\n");
      exit(EXIT_SUCCESS);
    }
  } else {
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

inline LWdevice findIntegratedGPUDrv() {
  LWdevice lwrrent_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;
  int isIntegrated;

  lwInit(0);
  checkLwdaErrors(lwDeviceGetCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr, "LWCA error: no devices supporting LWCA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the integrated GPU which is compute capable
  while (lwrrent_device < device_count) {
    int computeMode = -1;
    checkLwdaErrors(lwDeviceGetAttribute(
        &isIntegrated, LW_DEVICE_ATTRIBUTE_INTEGRATED, lwrrent_device));
    checkLwdaErrors(lwDeviceGetAttribute(
        &computeMode, LW_DEVICE_ATTRIBUTE_COMPUTE_MODE, lwrrent_device));

    // If GPU is integrated and is not running on Compute Mode prohibited use
    // that
    if (isIntegrated && (computeMode != LW_COMPUTEMODE_PROHIBITED)) {
      int major = 0, minor = 0;
      char deviceName[256];
      checkLwdaErrors(lwDeviceGetAttribute(
          &major, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
          lwrrent_device));
      checkLwdaErrors(lwDeviceGetAttribute(
          &minor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
          lwrrent_device));
      checkLwdaErrors(lwDeviceGetName(deviceName, 256, lwrrent_device));
      printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
             lwrrent_device, deviceName, major, minor);

      return lwrrent_device;
    } else {
      devices_prohibited++;
    }

    lwrrent_device++;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr, "LWCA error: No Integrated LWCA capable GPU found.\n");
    exit(EXIT_FAILURE);
  }

  return -1;
}

// General check for LWCA GPU SM Capabilities
inline bool checkLwdaCapabilitiesDRV(int major_version, int minor_version,
                                     int devID) {
  LWdevice lwDevice;
  char name[256];
  int major = 0, minor = 0;

  checkLwdaErrors(lwDeviceGet(&lwDevice, devID));
  checkLwdaErrors(lwDeviceGetName(name, 100, lwDevice));
  checkLwdaErrors(lwDeviceGetAttribute(
      &major, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, lwDevice));
  checkLwdaErrors(lwDeviceGetAttribute(
      &minor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, lwDevice));

  if ((major > major_version) ||
      (major == major_version && minor >= minor_version)) {
    printf("> Device %d: <%16s >, Compute SM %d.%d detected\n", devID, name,
           major, minor);
    return true;
  } else {
    printf(
        "No GPU device was found that can support LWCA compute capability "
        "%d.%d.\n",
        major_version, minor_version);
    return false;
  }
}
#endif
bool inline findFatbinPath(const char *module_file, std::string &module_path, char **argv, std::ostringstream &ostrm)
{
    char *actual_path = sdkFindFilePath(module_file, argv[0]);

    if (actual_path)
    {
        module_path = actual_path;
    }
    else
    {
        printf("> findModulePath file not found: <%s> \n", module_file);
        return false;
    }

    if (module_path.empty())
    {
        printf("> findModulePath could not find file: <%s> \n", module_file);
        return false;
    }
    else
    {
        printf("> findModulePath found file at <%s>\n", module_path.c_str());
        if (module_path.rfind("fatbin") != std::string::npos)
        {
            std::ifstream fileIn(module_path.c_str(), std::ios::binary);
            ostrm << fileIn.rdbuf();
            fileIn.close();
        }
        return true;
    }
}

  // end of LWCA Helper Functions

#endif  // COMMON_HELPER_LWDA_DRVAPI_H_

