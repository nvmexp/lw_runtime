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

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/utsname.h>
#endif

// Includes, system
#include <stdio.h>
#include <cassert>

// Includes LWCA
#include <lwda_runtime.h>
#include "lwrtc_helper.h"

// Utilities and timing functions
#include <helper_functions.h>  // includes lwca.h and lwda_runtime_api.h

const char *sampleName = "simpleAssert_lwrtc";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  printf("%s starting...\n", sampleName);

  runTest(argc, argv);

  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void runTest(int argc, char **argv) {
  int Nblocks = 2;
  int Nthreads = 32;

  // Kernel configuration, where a one-dimensional
  // grid and one-dimensional blocks are configured.

  dim3 dimGrid(Nblocks);
  dim3 dimBlock(Nthreads);

  printf("Launch kernel to generate assertion failures\n");
  char *lwbin, *kernel_file;
  size_t lwbinSize;

  kernel_file = sdkFindFilePath("simpleAssert_kernel.lw", argv[0]);
  compileFileToLWBIN(kernel_file, argc, argv, &lwbin, &lwbinSize, 0);

  LWmodule module = loadLWBIN(lwbin, argc, argv);
  LWfunction kernel_addr;

  checkLwdaErrors(lwModuleGetFunction(&kernel_addr, module, "testKernel"));

  int count = 60;
  void *args[] = {(void *)&count};

  checkLwdaErrors(lwLaunchKernel(
      kernel_addr, dimGrid.x, dimGrid.y, dimGrid.z, /* grid dim */
      dimBlock.x, dimBlock.y, dimBlock.z,           /* block dim */
      0, 0,                                         /* shared mem, stream */
      &args[0],                                     /* arguments */
      0));

  // Synchronize (flushes assert output).
  printf("\n-- Begin assert output\n\n");
  LWresult res = lwCtxSynchronize();

  printf("\n-- End assert output\n\n");

  // Check for errors and failed asserts in asynchronous kernel launch.
  if (res == LWDA_ERROR_ASSERT) {
    printf("Device assert failed as expected\n");
  }

  testResult = res == LWDA_ERROR_ASSERT;
}
