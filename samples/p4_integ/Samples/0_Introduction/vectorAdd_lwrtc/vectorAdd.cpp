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

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <cmath>

// For the LWCA runtime routines (prefixed with "lwda_")
#include <lwca.h>
#include <lwda_runtime.h>

// helper functions and utilities to work with LWCA
#include <helper_functions.h>

#include <lwrtc_helper.h>

/**
 * Host main routine
 */
int main(int argc, char **argv) {
  char *lwbin, *kernel_file;
  size_t lwbinSize;
  kernel_file = sdkFindFilePath("vectorAdd_kernel.lw", argv[0]);
  compileFileToLWBIN(kernel_file, argc, argv, &lwbin, &lwbinSize, 0);
  LWmodule module = loadLWBIN(lwbin, argc, argv);

  LWfunction kernel_addr;
  checkLwdaErrors(lwModuleGetFunction(&kernel_addr, module, "vectorAdd"));

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  float *h_A = reinterpret_cast<float *>(malloc(size));

  // Allocate the host input vector B
  float *h_B = reinterpret_cast<float *>(malloc(size));

  // Allocate the host output vector C
  float *h_C = reinterpret_cast<float *>(malloc(size));

  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / static_cast<float>(RAND_MAX);
    h_B[i] = rand() / static_cast<float>(RAND_MAX);
  }

  // Allocate the device input vector A
  LWdeviceptr d_A;
  checkLwdaErrors(lwMemAlloc(&d_A, size));

  // Allocate the device input vector B
  LWdeviceptr d_B;
  checkLwdaErrors(lwMemAlloc(&d_B, size));

  // Allocate the device output vector C
  LWdeviceptr d_C;
  checkLwdaErrors(lwMemAlloc(&d_C, size));

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in device memory
  printf("Copy input data from the host memory to the LWCA device\n");
  checkLwdaErrors(lwMemcpyHtoD(d_A, h_A, size));
  checkLwdaErrors(lwMemcpyHtoD(d_B, h_B, size));

  // Launch the Vector Add LWCA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("LWCA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  dim3 lwdaBlockSize(threadsPerBlock, 1, 1);
  dim3 lwdaGridSize(blocksPerGrid, 1, 1);

  void *arr[] = {reinterpret_cast<void *>(&d_A), reinterpret_cast<void *>(&d_B),
                 reinterpret_cast<void *>(&d_C),
                 reinterpret_cast<void *>(&numElements)};
  checkLwdaErrors(lwLaunchKernel(kernel_addr, lwdaGridSize.x, lwdaGridSize.y,
                                 lwdaGridSize.z, /* grid dim */
                                 lwdaBlockSize.x, lwdaBlockSize.y,
                                 lwdaBlockSize.z, /* block dim */
                                 0, 0,            /* shared mem, stream */
                                 &arr[0],         /* arguments */
                                 0));
  checkLwdaErrors(lwCtxSynchronize());

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the LWCA device to the host memory\n");
  checkLwdaErrors(lwMemcpyDtoH(h_C, d_C, size));

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  checkLwdaErrors(lwMemFree(d_A));
  checkLwdaErrors(lwMemFree(d_B));
  checkLwdaErrors(lwMemFree(d_C));

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");

  return 0;
}
