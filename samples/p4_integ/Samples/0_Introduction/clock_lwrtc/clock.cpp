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

/*
 * This example shows how to use the clock function to measure the performance
 * of block of threads of a kernel aclwrately. Blocks are exelwted in parallel
 * and out of order. Since there's no synchronization mechanism between blocks,
 * we measure the clock once for each block. The clock samples are written to
 * device memory.
 */

// System includes
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <lwda_runtime.h>
#include <lwrtc_helper.h>

// helper functions and utilities to work with LWCA
#include <helper_functions.h>

#define NUM_BLOCKS 64

#define NUM_THREADS 256

// It's interesting to change the number of blocks and the number of threads to
// understand how to keep the hardware busy.
//

// Here are some numbers I get on my G80:
//    blocks - clocks
//    1 - 3096
//    8 - 3232
//    16 - 3364
//    32 - 4615
//    64 - 9981

//
// With less than 16 blocks some of the multiprocessors of the device are idle.
// With
// more than 16 you are using all the multiprocessors, but there's only one
// block per
// multiprocessor and that doesn't allow you to hide the latency of the memory.
// With
// more than 32 the speed scales linearly.

// Start the main LWCA Sample here

int main(int argc, char **argv) {
  printf("LWCA Clock sample\n");

  typedef long clock_t;

  clock_t timer[NUM_BLOCKS * 2];

  float input[NUM_THREADS * 2];

  for (int i = 0; i < NUM_THREADS * 2; i++) {
    input[i] = (float)i;
  }

  char *lwbin, *kernel_file;
  size_t lwbinSize;

  kernel_file = sdkFindFilePath("clock_kernel.lw", argv[0]);
  compileFileToLWBIN(kernel_file, argc, argv, &lwbin, &lwbinSize, 0);

  LWmodule module = loadLWBIN(lwbin, argc, argv);
  LWfunction kernel_addr;

  checkLwdaErrors(lwModuleGetFunction(&kernel_addr, module, "timedReduction"));

  dim3 lwdaBlockSize(NUM_THREADS, 1, 1);
  dim3 lwdaGridSize(NUM_BLOCKS, 1, 1);

  LWdeviceptr dinput, doutput, dtimer;
  checkLwdaErrors(lwMemAlloc(&dinput, sizeof(float) * NUM_THREADS * 2));
  checkLwdaErrors(lwMemAlloc(&doutput, sizeof(float) * NUM_BLOCKS));
  checkLwdaErrors(lwMemAlloc(&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));
  checkLwdaErrors(lwMemcpyHtoD(dinput, input, sizeof(float) * NUM_THREADS * 2));

  void *arr[] = {(void *)&dinput, (void *)&doutput, (void *)&dtimer};

  checkLwdaErrors(lwLaunchKernel(
      kernel_addr, lwdaGridSize.x, lwdaGridSize.y,
      lwdaGridSize.z,                                    /* grid dim */
      lwdaBlockSize.x, lwdaBlockSize.y, lwdaBlockSize.z, /* block dim */
      sizeof(float) * 2 * NUM_THREADS, 0, /* shared mem, stream */
      &arr[0],                            /* arguments */
      0));

  checkLwdaErrors(lwCtxSynchronize());
  checkLwdaErrors(
      lwMemcpyDtoH(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));
  checkLwdaErrors(lwMemFree(dinput));
  checkLwdaErrors(lwMemFree(doutput));
  checkLwdaErrors(lwMemFree(dtimer));

  long double avgElapsedClocks = 0;

  for (int i = 0; i < NUM_BLOCKS; i++) {
    avgElapsedClocks += (long double)(timer[i + NUM_BLOCKS] - timer[i]);
  }

  avgElapsedClocks = avgElapsedClocks / NUM_BLOCKS;
  printf("Average clocks/block = %Lf\n", avgElapsedClocks);

  return EXIT_SUCCESS;
}
