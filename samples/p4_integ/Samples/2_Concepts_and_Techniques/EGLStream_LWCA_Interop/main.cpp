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
//
// DESCRIPTION:   Simple EGL stream sample app
//
//

//#define EGL_EGLEXT_PROTOTYPES

#include "lwdaEGL.h"
#include "lwda_consumer.h"
#include "lwda_producer.h"
#include "eglstrm_common.h"

/* ------  globals ---------*/

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

#define NUM_TRAILS 4

bool signal_stop = 0;

static void sig_handler(int sig) {
  signal_stop = 1;
  printf("Signal: %d\n", sig);
}

int main(int argc, char **argv) {
  TestArgs args;
  LWresult lwresult = LWDA_SUCCESS;
  unsigned int i, j;
  EGLint streamState = 0;

  test_lwda_consumer_s lwdaConsumer;
  test_lwda_producer_s lwdaProducer;

  memset(&lwdaProducer, 0, sizeof(test_lwda_producer_s));
  memset(&lwdaConsumer, 0, sizeof(test_lwda_consumer_s));

  // Hook up Ctrl-C handler
  signal(SIGINT, sig_handler);
  if (!eglSetupExtensions()) {
    printf("SetupExtentions failed \n");
    lwresult = LWDA_ERROR_UNKNOWN;
    goto done;
  }

  checkLwdaErrors(lwInit(0));

  int count;

  checkLwdaErrors(lwDeviceGetCount(&count));
  printf("Found %d lwca devices\n", count);

  LWdevice devId;

  if (!EGLStreamInit(&devId)) {
    printf("EGLStream Init failed.\n");
    lwresult = LWDA_ERROR_UNKNOWN;
    goto done;
  }
  lwresult = lwdaDeviceCreateProducer(&lwdaProducer, devId);
  if (lwresult != LWDA_SUCCESS) {
    goto done;
  }
  lwresult = lwdaDeviceCreateConsumer(&lwdaConsumer, devId);
  if (lwresult != LWDA_SUCCESS) {
    goto done;
  }
  checkLwdaErrors(lwCtxPushLwrrent(lwdaConsumer.context));
  if (LWDA_SUCCESS != (lwresult = lwEGLStreamConsumerConnect(
                           &(lwdaConsumer.lwdaConn), eglStream))) {
    printf("FAILED Connect LWCA consumer  with error %d\n", lwresult);
    goto done;
  } else {
    printf("Connected LWCA consumer, LwdaConsumer %p\n", lwdaConsumer.lwdaConn);
  }
  checkLwdaErrors(lwCtxPopLwrrent(&lwdaConsumer.context));

  checkLwdaErrors(lwCtxPushLwrrent(lwdaProducer.context));
  if (LWDA_SUCCESS ==
      (lwresult = lwEGLStreamProducerConnect(&(lwdaProducer.lwdaConn),
                                             eglStream, WIDTH, HEIGHT))) {
    printf("Connect LWCA producer Done, LwdaProducer %p\n",
           lwdaProducer.lwdaConn);
  } else {
    printf("Connect LWCA producer FAILED with error %d\n", lwresult);
    goto done;
  }
  checkLwdaErrors(lwCtxPopLwrrent(&lwdaProducer.context));

  // Initialize producer
  for (i = 0; i < NUM_TRAILS; i++) {
    if (streamState != EGL_STREAM_STATE_CONNECTING_KHR) {
      if (!eglQueryStreamKHR(g_display, eglStream, EGL_STREAM_STATE_KHR,
                             &streamState)) {
        printf("main: eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
        lwresult = LWDA_ERROR_UNKNOWN;
        goto done;
      }
    }
    args.inputWidth = WIDTH;
    args.inputHeight = HEIGHT;
    if (i % 2 != 0) {
      args.isARGB = 1;
      args.infile1 = sdkFindFilePath("lwda_f_1.yuv", argv[0]);
      args.infile2 = sdkFindFilePath("lwda_f_2.yuv", argv[0]);
    } else {
      args.isARGB = 0;
      args.infile1 = sdkFindFilePath("lwda_yuv_f_1.yuv", argv[0]);
      args.infile2 = sdkFindFilePath("lwda_yuv_f_2.yuv", argv[0]);
    }
    if ((i % 4) < 2) {
      args.pitchLinearOutput = 1;
    } else {
      args.pitchLinearOutput = 0;
    }

    checkLwdaErrors(lwCtxPushLwrrent(lwdaProducer.context));
    lwdaProducerInit(&lwdaProducer, g_display, eglStream, &args);
    checkLwdaErrors(lwCtxPopLwrrent(&lwdaProducer.context));

    checkLwdaErrors(lwCtxPushLwrrent(lwdaConsumer.context));
    lwda_consumer_init(&lwdaConsumer, &args);
    checkLwdaErrors(lwCtxPopLwrrent(&lwdaConsumer.context));

    printf("main - Lwca Producer and Consumer Initialized.\n");

    for (j = 0; j < 2; j++) {
      printf("Running for %s frame and %s input\n",
             args.isARGB ? "ARGB" : "YUV",
             args.pitchLinearOutput ? "Pitchlinear" : "BlockLinear");
      if (j == 0) {
        checkLwdaErrors(lwCtxPushLwrrent(lwdaProducer.context));
        lwresult = lwdaProducerTest(&lwdaProducer, lwdaProducer.fileName1);
        if (lwresult != LWDA_SUCCESS) {
          printf("Lwca Producer Test failed for frame = %d\n", j + 1);
          goto done;
        }
        checkLwdaErrors(lwCtxPopLwrrent(&lwdaProducer.context));
        checkLwdaErrors(lwCtxPushLwrrent(lwdaConsumer.context));
        lwresult = lwdaConsumerTest(&lwdaConsumer, lwdaConsumer.outFile1);
        if (lwresult != LWDA_SUCCESS) {
          printf("Lwca Consumer Test failed for frame = %d\n", j + 1);
          goto done;
        }
        checkLwdaErrors(lwCtxPopLwrrent(&lwdaConsumer.context));
      } else {
        checkLwdaErrors(lwCtxPushLwrrent(lwdaProducer.context));
        lwresult = lwdaProducerTest(&lwdaProducer, lwdaProducer.fileName2);
        if (lwresult != LWDA_SUCCESS) {
          printf("Lwca Producer Test failed for frame = %d\n", j + 1);
          goto done;
        }

        checkLwdaErrors(lwCtxPopLwrrent(&lwdaProducer.context));
        checkLwdaErrors(lwCtxPushLwrrent(lwdaConsumer.context));
        lwresult = lwdaConsumerTest(&lwdaConsumer, lwdaConsumer.outFile2);
        if (lwresult != LWDA_SUCCESS) {
          printf("Lwca Consumer Test failed for frame = %d\n", j + 1);
          goto done;
        }
        checkLwdaErrors(lwCtxPopLwrrent(&lwdaConsumer.context));
      }
    }
  }

  checkLwdaErrors(lwCtxPushLwrrent(lwdaProducer.context));
  if (LWDA_SUCCESS != (lwresult = lwdaProducerDeinit(&lwdaProducer))) {
    printf("Producer Disconnect FAILED. \n");
    goto done;
  }
  checkLwdaErrors(lwCtxPopLwrrent(&lwdaProducer.context));

  if (!eglQueryStreamKHR(g_display, eglStream, EGL_STREAM_STATE_KHR,
                         &streamState)) {
    printf("Lwca consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
    lwresult = LWDA_ERROR_UNKNOWN;
    goto done;
  }
  if (streamState != EGL_STREAM_STATE_DISCONNECTED_KHR) {
    if (LWDA_SUCCESS != (lwresult = lwda_consumer_deinit(&lwdaConsumer))) {
      printf("Consumer Disconnect FAILED.\n");
      goto done;
    }
  }
  printf("Producer and Consumer Disconnected \n");

done:
  if (!eglQueryStreamKHR(g_display, eglStream, EGL_STREAM_STATE_KHR,
                         &streamState)) {
    printf("Lwca consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
    lwresult = LWDA_ERROR_UNKNOWN;
  }
  if (streamState != EGL_STREAM_STATE_DISCONNECTED_KHR) {
    EGLStreamFini();
  }

  if (lwresult == LWDA_SUCCESS) {
    printf("&&&& EGLStream interop test PASSED\n");
  } else {
    printf("&&&& EGLStream interop test FAILED\n");
  }
  return 0;
}
