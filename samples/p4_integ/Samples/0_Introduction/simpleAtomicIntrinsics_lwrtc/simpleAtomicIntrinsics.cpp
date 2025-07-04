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

/* A simple program demonstrating trivial use of global memory atomic
 * device functions (atomic*() functions).
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// Includes LWCA
#include <lwda_runtime.h>
#include <lwrtc_helper.h>

// Utilities and timing functions
#include <helper_functions.h>  // includes lwca.h and lwda_runtime_api.h

const char *sampleName = "simpleAtomicIntrinsics_lwrtc";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

extern "C" bool computeGold(int *gpuData, const int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  printf("%s starting...\n", sampleName);

  runTest(argc, argv);

  printf("%s completed, returned %s\n", sampleName,
         testResult ? "OK" : "ERROR!");

  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for LWCA
////////////////////////////////////////////////////////////////////////////////

void runTest(int argc, char **argv) {
  int dev = 0;

  char *lwbin, *kernel_file;
  size_t lwbinSize;

  kernel_file = sdkFindFilePath("simpleAtomicIntrinsics_kernel.lwh", argv[0]);
  compileFileToLWBIN(kernel_file, argc, argv, &lwbin, &lwbinSize, 0);

  LWmodule module = loadLWBIN(lwbin, argc, argv);
  LWfunction kernel_addr;

  checkLwdaErrors(lwModuleGetFunction(&kernel_addr, module, "testKernel"));

  StopWatchInterface *timer;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  unsigned int numData = 11;
  unsigned int memSize = sizeof(int) * numData;

  // allocate mem for the result on host side
  int *hOData = (int *)malloc(memSize);

  // initialize the memory
  for (unsigned int i = 0; i < numData; i++) hOData[i] = 0;

  // To make the AND and XOR tests generate something other than 0...
  hOData[8] = hOData[10] = 0xff;

  // allocate device memory for result
  LWdeviceptr dOData;
  checkLwdaErrors(lwMemAlloc(&dOData, memSize));
  checkLwdaErrors(lwMemcpyHtoD(dOData, hOData, memSize));

  // execute the kernel
  dim3 lwdaBlockSize(numThreads, 1, 1);
  dim3 lwdaGridSize(numBlocks, 1, 1);

  void *arr[] = {(void *)&dOData};
  checkLwdaErrors(lwLaunchKernel(kernel_addr, lwdaGridSize.x, lwdaGridSize.y,
                                 lwdaGridSize.z, /* grid dim */
                                 lwdaBlockSize.x, lwdaBlockSize.y,
                                 lwdaBlockSize.z, /* block dim */
                                 0, 0,            /* shared mem, stream */
                                 &arr[0],         /* arguments */
                                 0));

  checkLwdaErrors(lwCtxSynchronize());

  checkLwdaErrors(lwMemcpyDtoH(hOData, dOData, memSize));

  // Copy result from device to host
  sdkStopTimer(&timer);
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);

  // Compute reference solution
  testResult = computeGold(hOData, numThreads * numBlocks);

  // Cleanup memory
  free(hOData);
  checkLwdaErrors(lwMemFree(dOData));
}
