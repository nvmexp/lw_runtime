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
 * vector addition. It loads a lwca fatbinary and runs vector addition kernel.
 * Uses both Driver and Runtime LWCA APIs for different purposes.
 */

// Includes
#include <lwca.h>
#include <lwda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <cstring>
#include <iostream>

// includes, project
#include <helper_lwda.h>
#include <helper_functions.h>

// includes, LWCA
#include <builtin_types.h>

using namespace std;

#ifndef FATBIN_FILE
#define FATBIN_FILE "vectorAdd_kernel64.fatbin"
#endif

// Variables
float *h_A;
float *h_B;
float *h_C;
float *d_A;
float *d_B;
float *d_C;

// Functions
int CleanupNoFailure(LWcontext &lwContext);
void RandomInit(float *, int);
bool findModulePath(const char *, string &, char **, ostringstream &);

static void check(LWresult result, char const *const func,
                  const char *const file, int const line) {
  if (result) {
    fprintf(stderr, "LWCA error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkLwdaDrvErrors(val) check((val), #val, __FILE__, __LINE__)

// Host code
int main(int argc, char **argv) {
  printf("simpleDrvRuntime..\n");
  int N = 50000, devID = 0;
  size_t size = N * sizeof(float);
  LWdevice lwDevice;
  LWfunction vecAdd_kernel;
  LWmodule lwModule = 0;
  LWcontext lwContext;

  // Initialize
  checkLwdaDrvErrors(lwInit(0));

  lwDevice = findLwdaDevice(argc, (const char **)argv);
  // Create context
  checkLwdaDrvErrors(lwCtxCreate(&lwContext, 0, lwDevice));

  // first search for the module path before we load the results
  string module_path;
  ostringstream fatbin;

  if (!findModulePath(FATBIN_FILE, module_path, argv, fatbin)) {
    exit(EXIT_FAILURE);
  } else {
    printf("> initLWDA loading module: <%s>\n", module_path.c_str());
  }

  if (!fatbin.str().size()) {
    printf("fatbin file empty. exiting..\n");
    exit(EXIT_FAILURE);
  }

  // Create module from binary file (FATBIN)
  checkLwdaDrvErrors(lwModuleLoadData(&lwModule, fatbin.str().c_str()));

  // Get function handle from module
  checkLwdaDrvErrors(
      lwModuleGetFunction(&vecAdd_kernel, lwModule, "VecAdd_kernel"));

  // Allocate input vectors h_A and h_B in host memory
  checkLwdaErrors(lwdaMallocHost(&h_A, size));
  checkLwdaErrors(lwdaMallocHost(&h_B, size));
  checkLwdaErrors(lwdaMallocHost(&h_C, size));

  // Initialize input vectors
  RandomInit(h_A, N);
  RandomInit(h_B, N);

  // Allocate vectors in device memory
  checkLwdaErrors(lwdaMalloc((void **)(&d_A), size));
  checkLwdaErrors(lwdaMalloc((void **)(&d_B), size));
  checkLwdaErrors(lwdaMalloc((void **)(&d_C), size));

  lwdaStream_t stream;
  checkLwdaErrors(lwdaStreamCreateWithFlags(&stream, lwdaStreamNonBlocking));
  // Copy vectors from host memory to device memory
  checkLwdaErrors(
      lwdaMemcpyAsync(d_A, h_A, size, lwdaMemcpyHostToDevice, stream));
  checkLwdaErrors(
      lwdaMemcpyAsync(d_B, h_B, size, lwdaMemcpyHostToDevice, stream));

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  void *args[] = {&d_A, &d_B, &d_C, &N};

  // Launch the LWCA kernel
  checkLwdaDrvErrors(lwLaunchKernel(vecAdd_kernel, blocksPerGrid, 1, 1,
                                    threadsPerBlock, 1, 1, 0, stream, args,
                                    NULL));

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  checkLwdaErrors(
      lwdaMemcpyAsync(h_C, d_C, size, lwdaMemcpyDeviceToHost, stream));
  checkLwdaErrors(lwdaStreamSynchronize(stream));
  // Verify result
  int i;

  for (i = 0; i < N; ++i) {
    float sum = h_A[i] + h_B[i];

    if (fabs(h_C[i] - sum) > 1e-7f) {
      break;
    }
  }

  checkLwdaDrvErrors(lwModuleUnload(lwModule));
  CleanupNoFailure(lwContext);
  printf("%s\n", (i == N) ? "Result = PASS" : "Result = FAIL");

  exit((i == N) ? EXIT_SUCCESS : EXIT_FAILURE);
}

int CleanupNoFailure(LWcontext &lwContext) {
  // Free device memory
  checkLwdaErrors(lwdaFree(d_A));
  checkLwdaErrors(lwdaFree(d_B));
  checkLwdaErrors(lwdaFree(d_C));

  // Free host memory
  if (h_A) {
    checkLwdaErrors(lwdaFreeHost(h_A));
  }

  if (h_B) {
    checkLwdaErrors(lwdaFreeHost(h_B));
  }

  if (h_C) {
    checkLwdaErrors(lwdaFreeHost(h_C));
  }

  checkLwdaDrvErrors(lwCtxDestroy(lwContext));

  return EXIT_SUCCESS;
}
// Allocates an array with random float entries.
void RandomInit(float *data, int n) {
  for (int i = 0; i < n; ++i) {
    data[i] = rand() / (float)RAND_MAX;
  }
}

bool inline findModulePath(const char *module_file, string &module_path,
                           char **argv, ostringstream &ostrm) {
  char *actual_path = sdkFindFilePath(module_file, argv[0]);

  if (actual_path) {
    module_path = actual_path;
  } else {
    printf("> findModulePath file not found: <%s> \n", module_file);
    return false;
  }

  if (module_path.empty()) {
    printf("> findModulePath could not find file: <%s> \n", module_file);
    return false;
  } else {
    printf("> findModulePath found file at <%s>\n", module_path.c_str());
    if (module_path.rfind("fatbin") != string::npos) {
      ifstream fileIn(module_path.c_str(), ios::binary);
      ostrm << fileIn.rdbuf();
    }
    return true;
  }
}
