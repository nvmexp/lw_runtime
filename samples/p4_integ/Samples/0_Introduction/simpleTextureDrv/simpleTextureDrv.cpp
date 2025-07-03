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
* This sample demonstrates how use texture fetches in LWCA
*
* This sample takes an input PGM image (image_filename) and generates
* an output PGM image (image_filename_out).  This LWCA kernel performs
* a simple 2D transform (rotation) on the texture coordinates (u,v).
* The results between simpleTexture and simpleTextureDrv are identical.
* The main difference is the implementation.  simpleTextureDrv makes calls
* to the LWCA driver API and demonstrates how to use lwModuleLoad to load
* the LWCA ptx (*.ptx) kernel just prior to kernel launch.
*
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <cstring>

// includes, LWCA
#include <lwca.h>
#include <builtin_types.h>
// includes, project
#include <helper_lwda_drvapi.h>
#include <helper_functions.h>

using namespace std;

const char *image_filename = "teapot512.pgm";
const char *ref_filename = "ref_rotated.pgm";
float angle = 0.5f;  // angle to rotate image by (in radians)

#define MIN_EPSILON_ERROR 5e-3f

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C" void computeGold(float *reference, float *idata,
                            const unsigned int len);

static LWresult initLWDA(int argc, char **argv, LWfunction *);

const char *sSDKsample = "simpleTextureDrv (Driver API)";

// define input fatbin file
#ifndef FATBIN_FILE
#define FATBIN_FILE "simpleTexture_kernel64.fatbin"
#endif

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
LWdevice lwDevice;
LWcontext lwContext;
LWmodule lwModule;

void showHelp() {
  printf("\n> [%s] Command line options\n", sSDKsample);
  printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    showHelp();
    return 0;
  }

  runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for LWCA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  bool bTestResults = true;

  // initialize LWCA
  LWfunction transform = NULL;

  if (initLWDA(argc, argv, &transform) != LWDA_SUCCESS) {
    exit(EXIT_FAILURE);
  }

  // load image from disk
  float *h_data = NULL;
  unsigned int width, height;
  char *image_path = sdkFindFilePath(image_filename, argv[0]);

  if (image_path == NULL) {
    printf("Unable to find image file: '%s'\n", image_filename);
    exit(EXIT_FAILURE);
  }

  sdkLoadPGM(image_path, &h_data, &width, &height);

  size_t size = width * height * sizeof(float);
  printf("Loaded '%s', %d x %d pixels\n", image_filename, width, height);

  // load reference image from image (output)
  float *h_data_ref = (float *)malloc(size);
  char *ref_path = sdkFindFilePath(ref_filename, argv[0]);

  if (ref_path == NULL) {
    printf("Unable to find reference file %s\n", ref_filename);
    exit(EXIT_FAILURE);
  }

  sdkLoadPGM(ref_path, &h_data_ref, &width, &height);

  // allocate device memory for result
  LWdeviceptr d_data = (LWdeviceptr)NULL;
  checkLwdaErrors(lwMemAlloc(&d_data, size));

  // allocate array and copy image data
  LWarray lw_array;
  LWDA_ARRAY_DESCRIPTOR desc;
  desc.Format = LW_AD_FORMAT_FLOAT;
  desc.NumChannels = 1;
  desc.Width = width;
  desc.Height = height;
  checkLwdaErrors(lwArrayCreate(&lw_array, &desc));
  LWDA_MEMCPY2D copyParam;
  memset(&copyParam, 0, sizeof(copyParam));
  copyParam.dstMemoryType = LW_MEMORYTYPE_ARRAY;
  copyParam.dstArray = lw_array;
  copyParam.srcMemoryType = LW_MEMORYTYPE_HOST;
  copyParam.srcHost = h_data;
  copyParam.srcPitch = width * sizeof(float);
  copyParam.WidthInBytes = copyParam.srcPitch;
  copyParam.Height = height;
  checkLwdaErrors(lwMemcpy2D(&copyParam));

  // set texture parameters
  LWtexObject TexObject;
  LWDA_RESOURCE_DESC ResDesc;
  memset(&ResDesc, 0, sizeof(LWDA_RESOURCE_DESC));
  ResDesc.resType = LW_RESOURCE_TYPE_ARRAY;
  ResDesc.res.array.hArray = lw_array;

  LWDA_TEXTURE_DESC TexDesc;
  memset(&TexDesc, 0, sizeof(LWDA_TEXTURE_DESC));
  TexDesc.addressMode[0] = LW_TR_ADDRESS_MODE_WRAP;
  TexDesc.addressMode[1] = LW_TR_ADDRESS_MODE_WRAP;
  TexDesc.addressMode[2] = LW_TR_ADDRESS_MODE_WRAP;
  TexDesc.filterMode = LW_TR_FILTER_MODE_LINEAR;
  TexDesc.flags = LW_TRSF_NORMALIZED_COORDINATES;

  checkLwdaErrors(lwTexObjectCreate(&TexObject, &ResDesc, &TexDesc, NULL));

  // There are two ways to launch LWCA kernels via the Driver API.
  // In this LWCA Sample, we illustrate both ways to pass parameters
  // and specify parameters.  By default we use the simpler method.
  int block_size = 8;
  StopWatchInterface *timer = NULL;

  if (1) {
    // This is the new LWCA 4.0 API for Kernel Parameter passing and Kernel
    // Launching (simpler method)
    void *args[5] = {&d_data, &width, &height, &angle, &TexObject};

    checkLwdaErrors(lwLaunchKernel(transform, (width / block_size),
                                   (height / block_size), 1, block_size,
                                   block_size, 1, 0, NULL, args, NULL));
    checkLwdaErrors(lwCtxSynchronize());
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // launch kernel again for performance measurement
    checkLwdaErrors(lwLaunchKernel(transform, (width / block_size),
                                   (height / block_size), 1, block_size,
                                   block_size, 1, 0, NULL, args, NULL));
  } else {
    // This is the new LWCA 4.0 API for Kernel Parameter passing and Kernel
    // Launching (advanced method)
    int offset = 0;
    char argBuffer[256];

    // pass in launch parameters (not actually de-referencing LWdeviceptr).
    // LWdeviceptr is
    // storing the value of the parameters
    *((LWdeviceptr *)&argBuffer[offset]) = d_data;
    offset += sizeof(d_data);
    *((unsigned int *)&argBuffer[offset]) = width;
    offset += sizeof(width);
    *((unsigned int *)&argBuffer[offset]) = height;
    offset += sizeof(height);
    *((float *)&argBuffer[offset]) = angle;
    offset += sizeof(angle);
    *((LWtexObject *)&argBuffer[offset]) = TexObject;
    offset += sizeof(TexObject);

    void *kernel_launch_config[5] = {LW_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
                                     LW_LAUNCH_PARAM_BUFFER_SIZE, &offset,
                                     LW_LAUNCH_PARAM_END};

    // new LWCA 4.0 Driver API Kernel launch call (warmup)
    checkLwdaErrors(lwLaunchKernel(
        transform, (width / block_size), (height / block_size), 1, block_size,
        block_size, 1, 0, NULL, NULL, (void **)&kernel_launch_config));
    checkLwdaErrors(lwCtxSynchronize());
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // launch kernel again for performance measurement
    checkLwdaErrors(lwLaunchKernel(
        transform, (width / block_size), (height / block_size), 1, block_size,
        block_size, 1, 0, 0, NULL, (void **)&kernel_launch_config));
  }

  checkLwdaErrors(lwCtxSynchronize());
  sdkStopTimer(&timer);
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  printf("%.2f Mpixels/sec\n",
         (width * height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
  sdkDeleteTimer(&timer);

  // allocate mem for the result on host side
  float *h_odata = (float *)malloc(size);
  // copy result from device to host
  checkLwdaErrors(lwMemcpyDtoH(h_odata, d_data, size));

  // write result to file
  char output_filename[1024];
  strcpy(output_filename, image_path);
  strcpy(output_filename + strlen(image_path) - 4, "_out.pgm");
  sdkSavePGM(output_filename, h_odata, width, height);
  printf("Wrote '%s'\n", output_filename);

  // write regression file if necessary
  if (checkCmdLineFlag(argc, (const char **)argv, "regression")) {
    // write file for regression test
    sdkWriteFile<float>("./data/regression.dat", h_odata, width * height, 0.0f,
                        false);
  } else {
    // We need to reload the data from disk, because it is ilwerted upon output
    sdkLoadPGM(output_filename, &h_odata, &width, &height);

    printf("Comparing files\n");
    printf("\toutput:    <%s>\n", output_filename);
    printf("\treference: <%s>\n", ref_path);
    bTestResults = compareData(h_odata, h_data_ref, width * height,
                               MIN_EPSILON_ERROR, 0.15f);
  }

  // cleanup memory
  checkLwdaErrors(lwTexObjectDestroy(TexObject));
  checkLwdaErrors(lwMemFree(d_data));
  checkLwdaErrors(lwArrayDestroy(lw_array));

  free(image_path);
  free(ref_path);

  checkLwdaErrors(lwCtxDestroy(lwContext));

  exit(bTestResults ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! This initializes LWCA, and loads the *.ptx LWCA module containing the
//! kernel function.  After the module is loaded, lwModuleGetFunction
//! retrieves the LWCA function pointer "lwFunction"
////////////////////////////////////////////////////////////////////////////////
static LWresult initLWDA(int argc, char **argv, LWfunction *transform) {
  LWfunction lwFunction = 0;
  int major = 0, minor = 0, devID = 0;
  char deviceName[100];
  string module_path;

  lwDevice = findLwdaDeviceDRV(argc, (const char **)argv);

  // get compute capabilities and the devicename
  checkLwdaErrors(lwDeviceGetAttribute(
      &major, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, lwDevice));
  checkLwdaErrors(lwDeviceGetAttribute(
      &minor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, lwDevice));
  checkLwdaErrors(lwDeviceGetName(deviceName, sizeof(deviceName), lwDevice));
  printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

  checkLwdaErrors(lwCtxCreate(&lwContext, 0, lwDevice));

  // first search for the module_path before we try to load the results
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

  checkLwdaErrors(
      lwModuleGetFunction(&lwFunction, lwModule, "transformKernel"));

  *transform = lwFunction;

  return LWDA_SUCCESS;
}
