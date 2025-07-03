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

/* This example demonstrates how to use the LWBLAS library
 * by scaling an array of floating-point values on the device
 * and comparing the result to the same operation performed
 * on the host.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, lwca */
#include <lwblasXt.h>
#include <lwda_runtime.h>
#include <helper_lwda.h>

/* Matrix size */
//#define N  (275)
#define N (1024)
// Restricting the max used GPUs as input matrix is not so large
#define MAX_NUM_OF_GPUS 2

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C) {
  int i;
  int j;
  int k;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float prod = 0;

      for (k = 0; k < n; ++k) {
        prod += A[k * n + i] * B[j * n + k];
      }

      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}

void findMultipleBestGPUs(int &num_of_devices, int *device_ids) {
  // Find the best LWCA capable GPU device
  int lwrrent_device = 0;

  int device_count;
  checkLwdaErrors(lwdaGetDeviceCount(&device_count));
  typedef struct gpu_perf_t {
    uint64_t compute_perf;
    int device_id;
  } gpu_perf;

  gpu_perf *gpu_stats = (gpu_perf *)malloc(sizeof(gpu_perf) * device_count);

  lwdaDeviceProp deviceProp;
  int devices_prohibited = 0;
  while (lwrrent_device < device_count) {
    lwdaGetDeviceProperties(&deviceProp, lwrrent_device);

    // If this GPU is not running on Compute Mode prohibited,
    // then we can add it to the list
    int sm_per_multiproc;
    if (deviceProp.computeMode != lwdaComputeModeProhibited) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc =
            _ColwertSMVer2Cores(deviceProp.major, deviceProp.minor);
      }

      gpu_stats[lwrrent_device].compute_perf =
          (uint64_t)deviceProp.multiProcessorCount * sm_per_multiproc *
          deviceProp.clockRate;
      gpu_stats[lwrrent_device].device_id = lwrrent_device;

    } else {
      devices_prohibited++;
    }

    ++lwrrent_device;
  }
  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() LWCA error:"
            " all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  } else {
    gpu_perf temp_elem;
    // Sort the GPUs by highest compute perf.
    for (int i = 0; i < lwrrent_device - 1; i++) {
      for (int j = 0; j < lwrrent_device - i - 1; j++) {
        if (gpu_stats[j].compute_perf < gpu_stats[j + 1].compute_perf) {
          temp_elem = gpu_stats[j];
          gpu_stats[j] = gpu_stats[j + 1];
          gpu_stats[j + 1] = temp_elem;
        }
      }
    }

    for (int i = 0; i < num_of_devices; i++) {
      device_ids[i] = gpu_stats[i].device_id;
    }
  }
  free(gpu_stats);
}

/* Main */
int main(int argc, char **argv) {
  lwblasStatus_t status;
  float *h_A;
  float *h_B;
  float *h_C;
  float *h_C_ref;
  float *d_A = 0;
  float *d_B = 0;
  float *d_C = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  int n2 = N * N;
  int i;
  float error_norm;
  float ref_norm;
  float diff;
  lwblasXtHandle_t handle;
  int *devices = NULL;

  int num_of_devices = 0;

  checkLwdaErrors(lwdaGetDeviceCount(&num_of_devices));

  if (num_of_devices > MAX_NUM_OF_GPUS) {
    num_of_devices = MAX_NUM_OF_GPUS;
  }
  devices = (int *)malloc(sizeof(int) * num_of_devices);

  findMultipleBestGPUs(num_of_devices, devices);
  lwdaDeviceProp deviceProp;
  printf("Using %d GPUs\n", num_of_devices);
  for (i = 0; i < num_of_devices; i++) {
    checkLwdaErrors(lwdaGetDeviceProperties(&deviceProp, devices[i]));
    printf("GPU ID = %d, Name = %s \n", devices[i], deviceProp.name);
  }

  /* Initialize LWBLAS */
  printf("simpleLWBLASXT test running..\n");

  status = lwblasXtCreate(&handle);

  if (status != LWBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! LWBLASXT initialization error\n");
    return EXIT_FAILURE;
  }

  /* Select devices for use in LWBLASXT math functions */
  status = lwblasXtDeviceSelect(handle, num_of_devices, devices);

  if (status != LWBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! LWBLASXT device selection error\n");
    return EXIT_FAILURE;
  }

  /* Optional: Set a block size for LWBLASXT math functions */
  status = lwblasXtSetBlockDim(handle, 64);

  if (status != LWBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! LWBLASXT set block dimension error\n");
    return EXIT_FAILURE;
  }

  /* Allocate host memory for the matrices */
  h_A = (float *)malloc(n2 * sizeof(h_A[0]));

  if (h_A == 0) {
    fprintf(stderr, "!!!! host memory allocation error (A)\n");
    return EXIT_FAILURE;
  }

  h_B = (float *)malloc(n2 * sizeof(h_B[0]));

  if (h_B == 0) {
    fprintf(stderr, "!!!! host memory allocation error (B)\n");
    return EXIT_FAILURE;
  }

  h_C_ref = (float *)malloc(n2 * sizeof(h_C[0]));

  if (h_C_ref == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C_ref)\n");
    return EXIT_FAILURE;
  }

  h_C = (float *)malloc(n2 * sizeof(h_C[0]));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  /* Fill the matrices with test data */
  for (i = 0; i < n2; i++) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
    h_C[i] = rand() / (float)RAND_MAX;
    h_C_ref[i] = h_C[i];
  }

  /* Performs operation using plain C code */
  simple_sgemm(N, alpha, h_A, h_B, beta, h_C_ref);

  /* Performs operation using lwblas */
  status = lwblasXtSgemm(handle, LWBLAS_OP_N, LWBLAS_OP_N, N, N, N, &alpha, h_A,
                         N, h_B, N, &beta, h_C, N);

  if (status != LWBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! kernel exelwtion error.\n");
    return EXIT_FAILURE;
  }

  /* Check result against reference */
  error_norm = 0;
  ref_norm = 0;

  for (i = 0; i < n2; ++i) {
    diff = h_C_ref[i] - h_C[i];
    error_norm += diff * diff;
    ref_norm += h_C_ref[i] * h_C_ref[i];
  }

  error_norm = (float)sqrt((double)error_norm);
  ref_norm = (float)sqrt((double)ref_norm);

  if (fabs(ref_norm) < 1e-7) {
    fprintf(stderr, "!!!! reference norm is 0\n");
    return EXIT_FAILURE;
  }

  /* Memory clean up */
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  if (lwdaFree(d_A) != lwdaSuccess) {
    fprintf(stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }

  if (lwdaFree(d_B) != lwdaSuccess) {
    fprintf(stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }

  if (lwdaFree(d_C) != lwdaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }

  /* Shutdown */
  status = lwblasXtDestroy(handle);

  if (status != LWBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }

  if (error_norm / ref_norm < 1e-6f) {
    printf("simpleLWBLASXT test passed.\n");
    exit(EXIT_SUCCESS);
  } else {
    printf("simpleLWBLASXT test failed.\n");
    exit(EXIT_FAILURE);
  }
}
