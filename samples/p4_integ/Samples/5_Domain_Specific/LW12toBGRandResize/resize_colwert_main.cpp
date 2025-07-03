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
LWPU HW Decoder, both dGPU and CheetAh, normally outputs LW12 pitch format
frames. For the inference using TensorRT, the input frame needs to be BGR planar
format with possibly different size. So, colwersion and resizing from LW12 to
BGR planar is usually required for the inference following decoding.
This LWCA code is to provide a reference implementation for colwersion and
resizing.

Limitaion
=========
    LW12resize needs the height to be a even value.

Note
====
    Resize function needs the pitch of image buffer to be 32 alignment.

Run
====
./LW12toBGRandResize
   OR
./LW12toBGRandResize -input=data/test1920x1080.lw12 -width=1920 -height=1080 \
-dst_width=640 -dst_height=480 -batch=40 -device=0

*/

#include <lwca.h>
#include <lwda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>

#include "resize_colwert.h"
#include "utils.h"

#define TEST_LOOP 20

typedef struct _lw12_to_bgr24_context_t {
  int width;
  int height;
  int pitch;

  int dst_width;
  int dst_height;
  int dst_pitch;

  int batch;
  int device;  // lwca device ID

  char *input_lw12_file;

  int ctx_pitch;    // the value will be suitable for Texture memroy.
  int ctx_heights;  // the value will be even.

} lw12_to_bgr24_context;

lw12_to_bgr24_context g_ctx;

static void printHelp(const char *app_name) {
  std::cout << "Usage:" << app_name << " [options]\n\n";
  std::cout << "OPTIONS:\n";
  std::cout << "\t-h,--help\n\n";
  std::cout << "\t-input=lw12file             lw12 input file\n";
  std::cout
      << "\t-width=width                input lw12 image width, <1 -- 4096>\n";
  std::cout
      << "\t-height=height              input lw12 image height, <1 -- 4096>\n";
  std::cout
      << "\t-pitch=pitch(optional)      input lw12 image pitch, <0 -- 4096>\n";
  std::cout
      << "\t-dst_width=width            output BGR image width, <1 -- 4096>\n";
  std::cout
      << "\t-dst_height=height          output BGR image height, <1 -- 4096>\n";
  std::cout
      << "\t-dst_pitch=pitch(optional)  output BGR image pitch, <0 -- 4096>\n";
  std::cout
      << "\t-batch=batch                process frames count, <1 -- 4096>\n\n";
  std::cout
      << "\t-device=device_num(optional)   lwca device number, <0 -- 4096>\n\n";

  return;
}

int parseCmdLine(int argc, char *argv[]) {
  char **argp = (char **)argv;
  char *arg = (char *)argv[0];

  memset(&g_ctx, 0, sizeof(g_ctx));

  if ((arg && (!strcmp(arg, "-h") || !strcmp(arg, "--help")))) {
    printHelp(argv[0]);
    return -1;
  }

  if (argc == 1) {
    // Run using default arguments

    g_ctx.input_lw12_file = sdkFindFilePath("test1920x1080.lw12", argv[0]);
    if (g_ctx.input_lw12_file == NULL) {
      printf("Cannot find input file test1920x1080.lw12\n Exiting\n");
      return EXIT_FAILURE;
    }
    g_ctx.width = 1920;
    g_ctx.height = 1080;
    g_ctx.dst_width = 640;
    g_ctx.dst_height = 480;
    g_ctx.batch = 24;
  } else if (argc > 1) {
    if (checkCmdLineFlag(argc, (const char **)argv, "width")) {
      g_ctx.width = getCmdLineArgumentInt(argc, (const char **)argv, "width");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "height")) {
      g_ctx.height = getCmdLineArgumentInt(argc, (const char **)argv, "height");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "pitch")) {
      g_ctx.pitch = getCmdLineArgumentInt(argc, (const char **)argv, "pitch");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
      getCmdLineArgumentString(argc, (const char **)argv, "input",
                               (char **)&g_ctx.input_lw12_file);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "dst_width")) {
      g_ctx.dst_width =
          getCmdLineArgumentInt(argc, (const char **)argv, "dst_width");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "dst_height")) {
      g_ctx.dst_height =
          getCmdLineArgumentInt(argc, (const char **)argv, "dst_height");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "dst_pitch")) {
      g_ctx.dst_pitch =
          getCmdLineArgumentInt(argc, (const char **)argv, "dst_pitch");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "batch")) {
      g_ctx.batch = getCmdLineArgumentInt(argc, (const char **)argv, "batch");
    }
  }

  g_ctx.device = findLwdaDevice(argc, (const char **)argv);

  if ((g_ctx.width == 0) || (g_ctx.height == 0) || (g_ctx.dst_width == 0) ||
      (g_ctx.dst_height == 0) || !g_ctx.input_lw12_file) {
    printHelp(argv[0]);
    return -1;
  }

  if (g_ctx.pitch == 0) g_ctx.pitch = g_ctx.width;
  if (g_ctx.dst_pitch == 0) g_ctx.dst_pitch = g_ctx.dst_width;

  return 0;
}

/*
  load lw12 yuvfile data into GPU device memory with batch of copy
 */
static int loadLW12Frame(unsigned char *d_inputLW12) {
  unsigned char *pLW12FrameData;
  unsigned char *d_lw12;
  int frameSize;
  std::ifstream lw12File(g_ctx.input_lw12_file, std::ifstream::in | std::ios::binary);

  if (!lw12File.is_open()) {
    std::cerr << "Can't open files\n";
    return -1;
  }

  frameSize = g_ctx.pitch * g_ctx.ctx_heights;

#if USE_UVM_MEM
  pLW12FrameData = d_inputLW12;
#else
  pLW12FrameData = (unsigned char *)malloc(frameSize);
  if (pLW12FrameData == NULL) {
    std::cerr << "Failed to malloc pLW12FrameData\n";
    return -1;
  }
#endif

  lw12File.read((char *)pLW12FrameData, frameSize);

  if (lw12File.gcount() < frameSize) {
    std::cerr << "can't get one frame!\n";
    return -1;
  }

#if USE_UVM_MEM
  // Prefetch to GPU for following GPU operation
  lwdaStreamAttachMemAsync(NULL, pLW12FrameData, 0, lwdaMemAttachGlobal);
#endif

  // expand one frame to multi frames for batch processing
  d_lw12 = d_inputLW12;
  for (int i = 0; i < g_ctx.batch; i++) {
    checkLwdaErrors(lwdaMemcpy2D((void *)d_lw12, g_ctx.ctx_pitch,
                                 pLW12FrameData, g_ctx.width, g_ctx.width,
                                 g_ctx.ctx_heights, lwdaMemcpyHostToDevice));

    d_lw12 += g_ctx.ctx_pitch * g_ctx.ctx_heights;
  }

#if (USE_UVM_MEM == 0)
  free(pLW12FrameData);
#endif
  lw12File.close();

  return 0;
}

/*
  1. resize interlace lw12 to target size
  2. colwert lw12 to bgr 3 progressive planars
 */
void lw12ResizeAndLW12ToBGR(unsigned char *d_inputLW12) {
  unsigned char *d_resizedLW12;
  float *d_outputBGR;
  int size;
  char filename[40];

  /* allocate device memory for resized lw12 output */
  size = g_ctx.dst_width * ceil(g_ctx.dst_height * 3.0f / 2.0f) * g_ctx.batch *
         sizeof(unsigned char);
  checkLwdaErrors(lwdaMalloc((void **)&d_resizedLW12, size));

  /* allocate device memory for bgr output */
  size = g_ctx.dst_pitch * g_ctx.dst_height * 3 * g_ctx.batch * sizeof(float);
  checkLwdaErrors(lwdaMalloc((void **)&d_outputBGR, size));

  lwdaStream_t stream;
  checkLwdaErrors(lwdaStreamCreate(&stream));
  /* create lwca event handles */
  lwdaEvent_t start, stop;
  checkLwdaErrors(lwdaEventCreate(&start));
  checkLwdaErrors(lwdaEventCreate(&stop));
  float elapsedTime = 0.0f;

  /* resize interlace lw12 */

  lwdaEventRecord(start, 0);
  for (int i = 0; i < TEST_LOOP; i++) {
    resizeLW12Batch(d_inputLW12, g_ctx.ctx_pitch, g_ctx.width, g_ctx.height,
                    d_resizedLW12, g_ctx.dst_width, g_ctx.dst_width,
                    g_ctx.dst_height, g_ctx.batch);
  }
  lwdaEventRecord(stop, 0);
  lwdaEventSynchronize(stop);

  lwdaEventElapsedTime(&elapsedTime, start, stop);
  printf(
      "  LWCA resize lw12(%dx%d --> %dx%d), batch: %d,"
      " average time: %.3f ms ==> %.3f ms/frame\n",
      g_ctx.width, g_ctx.height, g_ctx.dst_width, g_ctx.dst_height, g_ctx.batch,
      (elapsedTime / (TEST_LOOP * 1.0f)),
      (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch);

  sprintf(filename, "resized_lw12_%dx%d", g_ctx.dst_width, g_ctx.dst_height);

  /* colwert lw12 to bgr 3 progressive planars */
  lwdaEventRecord(start, 0);
  for (int i = 0; i < TEST_LOOP; i++) {
    lw12ToBGRplanarBatch(d_resizedLW12, g_ctx.dst_pitch,  // intput
                         d_outputBGR,
                         g_ctx.dst_pitch * sizeof(float),    // output
                         g_ctx.dst_width, g_ctx.dst_height,  // output
                         g_ctx.batch, 0);
  }
  lwdaEventRecord(stop, 0);
  lwdaEventSynchronize(stop);

  lwdaEventElapsedTime(&elapsedTime, start, stop);

  printf(
      "  LWCA colwert lw12(%dx%d) to bgr(%dx%d), batch: %d,"
      " average time: %.3f ms ==> %.3f ms/frame\n",
      g_ctx.dst_width, g_ctx.dst_height, g_ctx.dst_width, g_ctx.dst_height,
      g_ctx.batch, (elapsedTime / (TEST_LOOP * 1.0f)),
      (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch);

  sprintf(filename, "colwerted_bgr_%dx%d", g_ctx.dst_width, g_ctx.dst_height);
  dumpBGR(d_outputBGR, g_ctx.dst_pitch, g_ctx.dst_width, g_ctx.dst_height,
          g_ctx.batch, (char *)"t1", filename);

  /* release resources */
  checkLwdaErrors(lwdaEventDestroy(start));
  checkLwdaErrors(lwdaEventDestroy(stop));
  checkLwdaErrors(lwdaStreamDestroy(stream));
  checkLwdaErrors(lwdaFree(d_resizedLW12));
  checkLwdaErrors(lwdaFree(d_outputBGR));
}

/*
  1. colwert lw12 to bgr 3 progressive planars
  2. resize bgr 3 planars to target size
*/
void lw12ToBGRandBGRresize(unsigned char *d_inputLW12) {
  float *d_bgr;
  float *d_resizedBGR;
  int size;
  char filename[40];

  /* allocate device memory for bgr output */
  size = g_ctx.ctx_pitch * g_ctx.height * 3 * g_ctx.batch * sizeof(float);
  checkLwdaErrors(lwdaMalloc((void **)&d_bgr, size));

  /* allocate device memory for resized bgr output */
  size = g_ctx.dst_width * g_ctx.dst_height * 3 * g_ctx.batch * sizeof(float);
  checkLwdaErrors(lwdaMalloc((void **)&d_resizedBGR, size));

  lwdaStream_t stream;
  checkLwdaErrors(lwdaStreamCreate(&stream));
  /* create lwca event handles */
  lwdaEvent_t start, stop;
  checkLwdaErrors(lwdaEventCreate(&start));
  checkLwdaErrors(lwdaEventCreate(&stop));
  float elapsedTime = 0.0f;

  /* colwert interlace lw12 to bgr 3 progressive planars */
  lwdaEventRecord(start, 0);
  lwdaDeviceSynchronize();
  for (int i = 0; i < TEST_LOOP; i++) {
    lw12ToBGRplanarBatch(d_inputLW12, g_ctx.ctx_pitch, d_bgr,
                         g_ctx.ctx_pitch * sizeof(float), g_ctx.width,
                         g_ctx.height, g_ctx.batch, 0);
  }
  lwdaEventRecord(stop, 0);
  lwdaEventSynchronize(stop);

  lwdaEventElapsedTime(&elapsedTime, start, stop);
  printf(
      "  LWCA colwert lw12(%dx%d) to bgr(%dx%d), batch: %d,"
      " average time: %.3f ms ==> %.3f ms/frame\n",
      g_ctx.width, g_ctx.height, g_ctx.width, g_ctx.height, g_ctx.batch,
      (elapsedTime / (TEST_LOOP * 1.0f)),
      (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch);

  sprintf(filename, "colwerted_bgr_%dx%d", g_ctx.width, g_ctx.height);

  /* resize bgr 3 progressive planars */
  lwdaEventRecord(start, 0);
  for (int i = 0; i < TEST_LOOP; i++) {
    resizeBGRplanarBatch(d_bgr, g_ctx.ctx_pitch, g_ctx.width, g_ctx.height,
                         d_resizedBGR, g_ctx.dst_width, g_ctx.dst_width,
                         g_ctx.dst_height, g_ctx.batch);
  }
  lwdaEventRecord(stop, 0);
  lwdaEventSynchronize(stop);

  lwdaEventElapsedTime(&elapsedTime, start, stop);
  printf(
      "  LWCA resize bgr(%dx%d --> %dx%d), batch: %d,"
      " average time: %.3f ms ==> %.3f ms/frame\n",
      g_ctx.width, g_ctx.height, g_ctx.dst_width, g_ctx.dst_height, g_ctx.batch,
      (elapsedTime / (TEST_LOOP * 1.0f)),
      (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch);

  memset(filename, 0, sizeof(filename));
  sprintf(filename, "resized_bgr_%dx%d", g_ctx.dst_width, g_ctx.dst_height);
  dumpBGR(d_resizedBGR, g_ctx.dst_pitch, g_ctx.dst_width, g_ctx.dst_height,
          g_ctx.batch, (char *)"t2", filename);

  /* release resources */
  checkLwdaErrors(lwdaEventDestroy(start));
  checkLwdaErrors(lwdaEventDestroy(stop));
  checkLwdaErrors(lwdaStreamDestroy(stream));
  checkLwdaErrors(lwdaFree(d_bgr));
  checkLwdaErrors(lwdaFree(d_resizedBGR));
}

int main(int argc, char *argv[]) {
  unsigned char *d_inputLW12;

  if (parseCmdLine(argc, argv) < 0) return EXIT_FAILURE;

  g_ctx.ctx_pitch = g_ctx.width;
  int ctx_alignment = 32;
  g_ctx.ctx_pitch += (g_ctx.ctx_pitch % ctx_alignment != 0)
                         ? (ctx_alignment - g_ctx.ctx_pitch % ctx_alignment)
                         : 0;

  g_ctx.ctx_heights = ceil(g_ctx.height * 3.0f / 2.0f);

  /* load lw12 yuv data into d_inputLW12 with batch of copies */
#if USE_UVM_MEM
  checkLwdaErrors(lwdaMallocManaged(
      (void **)&d_inputLW12,
      (g_ctx.ctx_pitch * g_ctx.ctx_heights * g_ctx.batch), lwdaMemAttachHost));
  printf("\nUSE_UVM_MEM\n");
#else
  checkLwdaErrors(
      lwdaMalloc((void **)&d_inputLW12,
                 (g_ctx.ctx_pitch * g_ctx.ctx_heights * g_ctx.batch)));
#endif
  if (loadLW12Frame(d_inputLW12)) {
    std::cerr << "failed to load batch data!\n";
    return EXIT_FAILURE;
  }

  /* firstly resize lw12, then colwert lw12 to bgr */
  printf("\nTEST#1:\n");
  lw12ResizeAndLW12ToBGR(d_inputLW12);

  /* first colwert lw12 to bgr, then resize bgr */
  printf("\nTEST#2:\n");
  lw12ToBGRandBGRresize(d_inputLW12);

  checkLwdaErrors(lwdaFree(d_inputLW12));

  return EXIT_SUCCESS;
}
