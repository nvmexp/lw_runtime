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

/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

// Includes
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>
#include <lwca.h>

// includes, project
#include <helper_lwda_drvapi.h>
#include <helper_functions.h>

// includes, LWCA
#include <builtin_types.h>

using namespace std;

// Variables
LWdevice lwDevice;
LWcontext lwContext;
LWmodule lwModule;
LWfunction vecAdd_kernel;
float *h_A;
float *h_B;
float *h_C;
LWdeviceptr d_A;
LWdeviceptr d_B;
LWdeviceptr d_C;

// Functions
int CleanupNoFailure();
void RandomInit(float *, int);
bool findModulePath(const char *, string &, char **, string &);

// define input fatbin file
#ifndef FATBIN_FILE
#define FATBIN_FILE "vectorAdd_kernel64.fatbin"
#endif

// Host code
int main(int argc, char **argv) {
  printf("Vector Addition (Driver API)\n");
  int N = 50000, devID = 0;
  size_t size = N * sizeof(float);

  // Initialize
  checkLwdaErrors(lwInit(0));

  lwDevice = findLwdaDeviceDRV(argc, (const char **)argv);
  // Create context
  checkLwdaErrors(lwCtxCreate(&lwContext, 0, lwDevice));

  // first search for the module path before we load the results
  string module_path;

  std::ostringstream fatbin;

  if (!findFatbinPath(FATBIN_FILE, module_path, argv, fatbin)) {
    exit(EXIT_FAILURE);
  } else {
    printf("> initLWDA loading module: <%s>\n", module_path.c_str());
  }

  if (!fatbin.str().size()) {
    printf("fatbin file empty. exiting..\n");
    exit(EXIT_FAILURE);
  }

  // Create module from binary file (FATBIN)
  checkLwdaErrors(lwModuleLoadData(&lwModule, fatbin.str().c_str()));

  // Get function handle from module
  checkLwdaErrors(
      lwModuleGetFunction(&vecAdd_kernel, lwModule, "VecAdd_kernel"));

  // Allocate input vectors h_A and h_B in host memory
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  // Initialize input vectors
  RandomInit(h_A, N);
  RandomInit(h_B, N);

  // Allocate vectors in device memory
  checkLwdaErrors(lwMemAlloc(&d_A, size));

  checkLwdaErrors(lwMemAlloc(&d_B, size));

  checkLwdaErrors(lwMemAlloc(&d_C, size));

  // Copy vectors from host memory to device memory
  checkLwdaErrors(lwMemcpyHtoD(d_A, h_A, size));

  checkLwdaErrors(lwMemcpyHtoD(d_B, h_B, size));

  if (1) {
    // This is the new LWCA 4.0 API for Kernel Parameter Passing and Kernel
    // Launch (simpler method)

    // Grid/Block configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    void *args[] = {&d_A, &d_B, &d_C, &N};

    // Launch the LWCA kernel
    checkLwdaErrors(lwLaunchKernel(vecAdd_kernel, blocksPerGrid, 1, 1,
                                   threadsPerBlock, 1, 1, 0, NULL, args, NULL));
  } else {
    // This is the new LWCA 4.0 API for Kernel Parameter Passing and Kernel
    // Launch (advanced method)
    int offset = 0;
    void *argBuffer[16];
    *((LWdeviceptr *)&argBuffer[offset]) = d_A;
    offset += sizeof(d_A);
    *((LWdeviceptr *)&argBuffer[offset]) = d_B;
    offset += sizeof(d_B);
    *((LWdeviceptr *)&argBuffer[offset]) = d_C;
    offset += sizeof(d_C);
    *((int *)&argBuffer[offset]) = N;
    offset += sizeof(N);

    // Grid/Block configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the LWCA kernel
    checkLwdaErrors(lwLaunchKernel(vecAdd_kernel, blocksPerGrid, 1, 1,
                                   threadsPerBlock, 1, 1, 0, NULL, NULL,
                                   argBuffer));
  }

#ifdef _DEBUG
  checkLwdaErrors(lwCtxSynchronize());
#endif

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  checkLwdaErrors(lwMemcpyDtoH(h_C, d_C, size));

  // Verify result
  int i;

  for (i = 0; i < N; ++i) {
    float sum = h_A[i] + h_B[i];

    if (fabs(h_C[i] - sum) > 1e-7f) {
      break;
    }
  }

  CleanupNoFailure();
  printf("%s\n", (i == N) ? "Result = PASS" : "Result = FAIL");

  exit((i == N) ? EXIT_SUCCESS : EXIT_FAILURE);
}

int CleanupNoFailure() {
  // Free device memory
  checkLwdaErrors(lwMemFree(d_A));
  checkLwdaErrors(lwMemFree(d_B));
  checkLwdaErrors(lwMemFree(d_C));

  // Free host memory
  if (h_A) {
    free(h_A);
  }

  if (h_B) {
    free(h_B);
  }

  if (h_C) {
    free(h_C);
  }

  checkLwdaErrors(lwCtxDestroy(lwContext));

  return EXIT_SUCCESS;
}
// Allocates an array with random float entries.
void RandomInit(float *data, int n) {
  for (int i = 0; i < n; ++i) {
    data[i] = rand() / (float)RAND_MAX;
  }
}
