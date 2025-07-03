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

/* This sample queries the properties of the LWCA devices present
 * in the system.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <lwca.h>
#include <helper_lwda_drvapi.h>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  LWdevice dev;
  int major = 0, minor = 0;
  int deviceCount = 0;
  char deviceName[256];

  printf("%s Starting...\n\n", argv[0]);

  // note your project will need to link with lwca.lib files on windows
  printf("LWCA Device Query (Driver API) statically linked version \n");

  checkLwdaErrors(lwInit(0));

  checkLwdaErrors(lwDeviceGetCount(&deviceCount));

  // This function call returns 0 if there are no LWCA capable devices.
  if (deviceCount == 0) {
    printf("There are no available device(s) that support LWCA\n");
  } else {
    printf("Detected %d LWCA Capable device(s)\n", deviceCount);
  }

  for (dev = 0; dev < deviceCount; ++dev) {
    checkLwdaErrors(lwDeviceGetAttribute(
        &major, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    checkLwdaErrors(lwDeviceGetAttribute(
        &minor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));

    checkLwdaErrors(lwDeviceGetName(deviceName, 256, dev));

    printf("\nDevice %d: \"%s\"\n", dev, deviceName);

    int driverVersion = 0;
    checkLwdaErrors(lwDriverGetVersion(&driverVersion));
    printf("  LWCA Driver Version:                           %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  LWCA Capability Major/Minor version number:    %d.%d\n", major,
           minor);

    size_t totalGlobalMem;
    checkLwdaErrors(lwDeviceTotalMem(&totalGlobalMem, dev));

    char msg[256];
    SPRINTF(msg,
            "  Total amount of global memory:                 %.0f MBytes "
            "(%llu bytes)\n",
            (float)totalGlobalMem / 1048576.0f,
            (unsigned long long)totalGlobalMem);
    printf("%s", msg);

    int multiProcessorCount;
    getLwdaAttribute<int>(&multiProcessorCount,
                          LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);

    printf("  (%2d) Multiprocessors, (%3d) LWCA Cores/MP:     %d LWCA Cores\n",
           multiProcessorCount, _ColwertSMVer2CoresDRV(major, minor),
           _ColwertSMVer2CoresDRV(major, minor) * multiProcessorCount);

    int clockRate;
    getLwdaAttribute<int>(&clockRate, LW_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    printf(
        "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
        "GHz)\n",
        clockRate * 1e-3f, clockRate * 1e-6f);
    int memoryClock;
    getLwdaAttribute<int>(&memoryClock, LW_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                          dev);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           memoryClock * 1e-3f);
    int memBusWidth;
    getLwdaAttribute<int>(&memBusWidth,
                          LW_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
    printf("  Memory Bus Width:                              %d-bit\n",
           memBusWidth);
    int L2CacheSize;
    getLwdaAttribute<int>(&L2CacheSize, LW_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

    if (L2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             L2CacheSize);
    }

    int maxTex1D, maxTex2D[2], maxTex3D[3];
    getLwdaAttribute<int>(&maxTex1D,
                          LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, dev);
    getLwdaAttribute<int>(&maxTex2D[0],
                          LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, dev);
    getLwdaAttribute<int>(&maxTex2D[1],
                          LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, dev);
    getLwdaAttribute<int>(&maxTex3D[0],
                          LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, dev);
    getLwdaAttribute<int>(&maxTex3D[1],
                          LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, dev);
    getLwdaAttribute<int>(&maxTex3D[2],
                          LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, dev);
    printf(
        "  Max Texture Dimension Sizes                    1D=(%d) 2D=(%d, %d) "
        "3D=(%d, %d, %d)\n",
        maxTex1D, maxTex2D[0], maxTex2D[1], maxTex3D[0], maxTex3D[1],
        maxTex3D[2]);

    int maxTex1DLayered[2];
    getLwdaAttribute<int>(&maxTex1DLayered[0],
                          LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
                          dev);
    getLwdaAttribute<int>(&maxTex1DLayered[1],
                          LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
                          dev);
    printf(
        "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
        maxTex1DLayered[0], maxTex1DLayered[1]);

    int maxTex2DLayered[3];
    getLwdaAttribute<int>(&maxTex2DLayered[0],
                          LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
                          dev);
    getLwdaAttribute<int>(&maxTex2DLayered[1],
                          LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
                          dev);
    getLwdaAttribute<int>(&maxTex2DLayered[2],
                          LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
                          dev);
    printf(
        "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
        "layers\n",
        maxTex2DLayered[0], maxTex2DLayered[1], maxTex2DLayered[2]);

    int totalConstantMemory;
    getLwdaAttribute<int>(&totalConstantMemory,
                          LW_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, dev);
    printf("  Total amount of constant memory:               %u bytes\n",
           totalConstantMemory);
    int sharedMemPerBlock;
    getLwdaAttribute<int>(&sharedMemPerBlock,
                          LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev);
    printf("  Total amount of shared memory per block:       %u bytes\n",
           sharedMemPerBlock);
    int regsPerBlock;
    getLwdaAttribute<int>(&regsPerBlock,
                          LW_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
    printf("  Total number of registers available per block: %d\n",
           regsPerBlock);
    int warpSize;
    getLwdaAttribute<int>(&warpSize, LW_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
    printf("  Warp size:                                     %d\n", warpSize);
    int maxThreadsPerMultiProcessor;
    getLwdaAttribute<int>(&maxThreadsPerMultiProcessor,
                          LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                          dev);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           maxThreadsPerMultiProcessor);
    int maxThreadsPerBlock;
    getLwdaAttribute<int>(&maxThreadsPerBlock,
                          LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
    printf("  Maximum number of threads per block:           %d\n",
           maxThreadsPerBlock);

    int blockDim[3];
    getLwdaAttribute<int>(&blockDim[0], LW_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                          dev);
    getLwdaAttribute<int>(&blockDim[1], LW_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                          dev);
    getLwdaAttribute<int>(&blockDim[2], LW_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                          dev);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           blockDim[0], blockDim[1], blockDim[2]);
    int gridDim[3];
    getLwdaAttribute<int>(&gridDim[0], LW_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev);
    getLwdaAttribute<int>(&gridDim[1], LW_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev);
    getLwdaAttribute<int>(&gridDim[2], LW_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev);
    printf("  Max dimension size of a grid size (x,y,z):    (%d, %d, %d)\n",
           gridDim[0], gridDim[1], gridDim[2]);

    int textureAlign;
    getLwdaAttribute<int>(&textureAlign, LW_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
                          dev);
    printf("  Texture alignment:                             %u bytes\n",
           textureAlign);

    int memPitch;
    getLwdaAttribute<int>(&memPitch, LW_DEVICE_ATTRIBUTE_MAX_PITCH, dev);
    printf("  Maximum memory pitch:                          %u bytes\n",
           memPitch);

    int gpuOverlap;
    getLwdaAttribute<int>(&gpuOverlap, LW_DEVICE_ATTRIBUTE_GPU_OVERLAP, dev);

    int asyncEngineCount;
    getLwdaAttribute<int>(&asyncEngineCount,
                          LW_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev);
    printf(
        "  Conlwrrent copy and kernel exelwtion:          %s with %d copy "
        "engine(s)\n",
        (gpuOverlap ? "Yes" : "No"), asyncEngineCount);

    int kernelExecTimeoutEnabled;
    getLwdaAttribute<int>(&kernelExecTimeoutEnabled,
                          LW_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, dev);
    printf("  Run time limit on kernels:                     %s\n",
           kernelExecTimeoutEnabled ? "Yes" : "No");
    int integrated;
    getLwdaAttribute<int>(&integrated, LW_DEVICE_ATTRIBUTE_INTEGRATED, dev);
    printf("  Integrated GPU sharing Host Memory:            %s\n",
           integrated ? "Yes" : "No");
    int canMapHostMemory;
    getLwdaAttribute<int>(&canMapHostMemory,
                          LW_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev);
    printf("  Support host page-locked memory mapping:       %s\n",
           canMapHostMemory ? "Yes" : "No");

    int conlwrrentKernels;
    getLwdaAttribute<int>(&conlwrrentKernels,
                          LW_DEVICE_ATTRIBUTE_CONLWRRENT_KERNELS, dev);
    printf("  Conlwrrent kernel exelwtion:                   %s\n",
           conlwrrentKernels ? "Yes" : "No");

    int surfaceAlignment;
    getLwdaAttribute<int>(&surfaceAlignment,
                          LW_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, dev);
    printf("  Alignment requirement for Surfaces:            %s\n",
           surfaceAlignment ? "Yes" : "No");

    int eccEnabled;
    getLwdaAttribute<int>(&eccEnabled, LW_DEVICE_ATTRIBUTE_ECC_ENABLED, dev);
    printf("  Device has ECC support:                        %s\n",
           eccEnabled ? "Enabled" : "Disabled");

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    int tccDriver;
    getLwdaAttribute<int>(&tccDriver, LW_DEVICE_ATTRIBUTE_TCC_DRIVER, dev);
    printf("  LWCA Device Driver Mode (TCC or WDDM):         %s\n",
           tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                     : "WDDM (Windows Display Driver Model)");
#endif

    int unifiedAddressing;
    getLwdaAttribute<int>(&unifiedAddressing,
                          LW_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
    printf("  Device supports Unified Addressing (UVA):      %s\n",
           unifiedAddressing ? "Yes" : "No");

    int managedMemory;
    getLwdaAttribute<int>(&managedMemory, LW_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
                          dev);
    printf("  Device supports Managed Memory:                %s\n",
           managedMemory ? "Yes" : "No");

    int computePreemption;
    getLwdaAttribute<int>(&computePreemption,
                          LW_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED,
                          dev);
    printf("  Device supports Compute Preemption:            %s\n",
           computePreemption ? "Yes" : "No");

    int cooperativeLaunch;
    getLwdaAttribute<int>(&cooperativeLaunch,
                          LW_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, dev);
    printf("  Supports Cooperative Kernel Launch:            %s\n",
           cooperativeLaunch ? "Yes" : "No");

    int cooperativeMultiDevLaunch;
    getLwdaAttribute<int>(&cooperativeMultiDevLaunch,
                          LW_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH,
                          dev);
    printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           cooperativeMultiDevLaunch ? "Yes" : "No");

    int pciDomainID, pciBusID, pciDeviceID;
    getLwdaAttribute<int>(&pciDomainID, LW_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev);
    getLwdaAttribute<int>(&pciBusID, LW_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
    getLwdaAttribute<int>(&pciDeviceID, LW_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
    printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
           pciDomainID, pciBusID, pciDeviceID);

    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::lwdaSetDevice() with device "
        "simultaneously)",
        "Exclusive (only one host thread in one process is able to use "
        "::lwdaSetDevice() with this device)",
        "Prohibited (no host thread can use ::lwdaSetDevice() with this "
        "device)",
        "Exclusive Process (many threads in one process is able to use "
        "::lwdaSetDevice() with this device)",
        "Unknown", NULL};

    int computeMode;
    getLwdaAttribute<int>(&computeMode, LW_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev);
    printf("  Compute Mode:\n");
    printf("     < %s >\n", sComputeMode[computeMode]);
  }

  // If there are 2 or more GPUs, query to determine whether RDMA is supported
  if (deviceCount >= 2) {
    int gpuid[64];  // we want to find the first two GPUs that can support P2P
    int gpu_p2p_count = 0;
    int tccDriver = 0;

    for (int i = 0; i < deviceCount; i++) {
      checkLwdaErrors(lwDeviceGetAttribute(
          &major, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i));
      checkLwdaErrors(lwDeviceGetAttribute(
          &minor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i));
      getLwdaAttribute<int>(&tccDriver, LW_DEVICE_ATTRIBUTE_TCC_DRIVER, i);

      // Only boards based on Fermi or later can support P2P
      if ((major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
          // on Windows (64-bit), the Tesla Compute Cluster driver for windows
          // must be enabled to support this
          && tccDriver
#endif
          ) {
        // This is an array of P2P capable GPUs
        gpuid[gpu_p2p_count++] = i;
      }
    }

    // Show all the combinations of support P2P GPUs
    int can_access_peer;
    char deviceName0[256], deviceName1[256];

    if (gpu_p2p_count >= 2) {
      for (int i = 0; i < gpu_p2p_count; i++) {
        for (int j = 0; j < gpu_p2p_count; j++) {
          if (gpuid[i] == gpuid[j]) {
            continue;
          }
          checkLwdaErrors(
              lwDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
          checkLwdaErrors(lwDeviceGetName(deviceName0, 256, gpuid[i]));
          checkLwdaErrors(lwDeviceGetName(deviceName1, 256, gpuid[j]));
          printf(
              "> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : "
              "%s\n",
              deviceName0, gpuid[i], deviceName1, gpuid[j],
              can_access_peer ? "Yes" : "No");
        }
      }
    }
  }

  printf("Result = PASS\n");

  exit(EXIT_SUCCESS);
}
