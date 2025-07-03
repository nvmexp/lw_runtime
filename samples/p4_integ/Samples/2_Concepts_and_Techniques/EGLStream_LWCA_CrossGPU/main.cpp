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

#include "lwdaEGL.h"
#include "lwda_consumer.h"
#include "lwda_producer.h"
#include "eglstrm_common.h"
#include "helper.h"
#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

bool signal_stop = 0;
extern bool verbose;

static void sig_handler(int sig) {
  signal_stop = 1;
  printf("Signal: %d\n", sig);
}

void DoneCons(int consumerStatus, int send_fd) {
  EGLStreamFini();
  // get the final status from producer, combine and print
  int producerStatus = -1;
  if (-1 == recv(send_fd, (void *)&producerStatus, sizeof(int), 0)) {
    printf("%s: Lwca Consumer could not receive status from producer.\n",
           __func__);
  }
  close(send_fd);

  if (producerStatus == 0 && consumerStatus == 0) {
    printf("&&&& EGLStream_LWDA_CrossGPU PASSED\n");
    exit(EXIT_SUCCESS);
  } else {
    printf("&&&& EGLStream_LWDA_CrossGPU FAILED\n");
    exit(EXIT_FAILURE);
  }
}

void DoneProd(int producerStatus, int connect_fd) {
  EGLStreamFini();
  if (-1 == send(connect_fd, (void *)&producerStatus, sizeof(int), 0)) {
    printf("%s: Lwca Producer could not send status to consumer.\n", __func__);
  }
  close(connect_fd);
  if (producerStatus == 0) {
    exit(EXIT_SUCCESS);
  } else {
    exit(EXIT_FAILURE);
  }
}

int WIDTH = 8192, HEIGHT = 8192;
int main(int argc, char **argv) {
  TestArgs args = {0, false};
  LWresult lwresult = LWDA_SUCCESS;
  unsigned int j = 0;
  lwdaError_t err = lwdaSuccess;
  EGLNativeFileDescriptorKHR fileDescriptor = EGL_NO_FILE_DESCRIPTOR_KHR;
  struct timespec start, end;
  LWeglFrame lwdaEgl1, lwdaEgl2;
  int consumerStatus = 0;
  int send_fd = -1;

  if (parseCmdLine(argc, argv, &args) < 0) {
    printUsage();
    lwresult = LWDA_ERROR_UNKNOWN;
    DoneCons(consumerStatus, send_fd);
  }

  printf("Width : %u, height: %u and iterations: %u\n", WIDTH, HEIGHT,
         NUMTRIALS);

  if (!args.isProducer)  // Consumer code
  {
    test_lwda_consumer_s lwdaConsumer;
    memset(&lwdaConsumer, 0, sizeof(test_lwda_consumer_s));
    lwdaConsumer.profileAPI = profileAPIs;

    // Hook up Ctrl-C handler
    signal(SIGINT, sig_handler);

    if (!EGLStreamInit(isCrossDevice, !args.isProducer,
                       EGL_NO_FILE_DESCRIPTOR_KHR)) {
      printf("EGLStream Init failed.\n");
      lwresult = LWDA_ERROR_UNKNOWN;
      consumerStatus = -1;
      DoneCons(consumerStatus, send_fd);
    }

    lwdaConsumer.lwdaDevId = lwdaDevIndexCons;
    lwresult = lwdaDeviceCreateConsumer(&lwdaConsumer);
    if (lwresult != LWDA_SUCCESS) {
      consumerStatus = -1;
      DoneCons(consumerStatus, send_fd);
    }

    lwCtxPushLwrrent(lwdaConsumer.context);

    launchProducer(&args);

    args.charCnt = WIDTH * HEIGHT * 4;

    lwresult = lwda_consumer_init(&lwdaConsumer, &args);
    if (lwresult != LWDA_SUCCESS) {
      printf("Lwca Consumer: Init failed, status: %d\n", lwresult);
      consumerStatus = -1;
      DoneCons(consumerStatus, send_fd);
    }

    lwCtxPopLwrrent(&lwdaConsumer.context);

    send_fd = UnixSocketConnect(SOCK_PATH);
    if (-1 == send_fd) {
      printf("%s: Lwca Consumer cannot create socket %s\n", __func__,
             SOCK_PATH);
      consumerStatus = -1;
      DoneCons(consumerStatus, send_fd);
    }

    lwCtxPushLwrrent(lwdaConsumer.context);
    lwdaConsumer.eglStream = g_consumerEglStream;
    lwdaConsumer.eglDisplay = g_consumerEglDisplay;

    // Send the EGL stream FD to producer
    fileDescriptor = eglGetStreamFileDescriptorKHR(lwdaConsumer.eglDisplay,
                                                   lwdaConsumer.eglStream);
    if (EGL_NO_FILE_DESCRIPTOR_KHR == fileDescriptor) {
      printf("%s: Lwca Consumer could not get EGL file descriptor.\n",
             __func__);
      eglDestroyStreamKHR(lwdaConsumer.eglDisplay, lwdaConsumer.eglStream);
      consumerStatus = -1;
      DoneCons(consumerStatus, send_fd);
    }

    if (verbose)
      printf("%s: Lwca Consumer EGL stream FD obtained : %d.\n", __func__,
             fileDescriptor);

    int res = -1;
    res = EGLStreamSendfd(send_fd, fileDescriptor);
    if (-1 == res) {
      printf("%s: Lwca Consumer could not send EGL file descriptor.\n",
             __func__);
      consumerStatus = -1;
      close(fileDescriptor);
    }

    if (LWDA_SUCCESS !=
        (lwresult = lwEGLStreamConsumerConnect(&(lwdaConsumer.lwdaConn),
                                               lwdaConsumer.eglStream))) {
      printf("FAILED Connect LWCA consumer with error %d\n", lwresult);
      consumerStatus = -1;
      DoneCons(consumerStatus, send_fd);
    }

    j = 0;
    for (j = 0; j < NUMTRIALS; j++) {
      lwresult = lwdaConsumerAcquireFrame(&lwdaConsumer, j);
      if (lwresult != LWDA_SUCCESS) {
        printf("Lwca Consumer Test failed for frame = %d\n", j + 1);
        consumerStatus = -1;
        DoneCons(consumerStatus, send_fd);
      }
      lwresult = lwdaConsumerReleaseFrame(&lwdaConsumer, j);
      if (lwresult != LWDA_SUCCESS) {
        printf("Lwca Consumer Test failed for frame = %d\n", j + 1);
        consumerStatus = -1;
        DoneCons(consumerStatus, send_fd);
      }

      lwresult = lwdaConsumerAcquireFrame(&lwdaConsumer, j);
      if (lwresult != LWDA_SUCCESS) {
        printf("Lwca Consumer Test failed for frame = %d\n", j + 1);
        consumerStatus = -1;
        DoneCons(consumerStatus, send_fd);
      }
      lwresult = lwdaConsumerReleaseFrame(&lwdaConsumer, j);
      if (lwresult != LWDA_SUCCESS) {
        printf("Lwca Consumer Test failed for frame = %d\n", j + 1);
        consumerStatus = -1;
        DoneCons(consumerStatus, send_fd);
      }
    }
    lwCtxSynchronize();
    close(fileDescriptor);
    err = lwdaGetValueMismatch();
    if (err != lwdaSuccess) {
      printf("Consumer: App failed with value mismatch\n");
      lwresult = LWDA_ERROR_UNKNOWN;
      consumerStatus = -1;
      DoneCons(consumerStatus, send_fd);
    }

    EGLint streamState = 0;
    if (!eglQueryStreamKHR(lwdaConsumer.eglDisplay, lwdaConsumer.eglStream,
                           EGL_STREAM_STATE_KHR, &streamState)) {
      printf("Main, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
      lwresult = LWDA_ERROR_UNKNOWN;
      consumerStatus = -1;
      DoneCons(consumerStatus, send_fd);
    }

    if (streamState != EGL_STREAM_STATE_DISCONNECTED_KHR) {
      if (LWDA_SUCCESS != (lwresult = lwda_consumer_Deinit(&lwdaConsumer))) {
        printf("Consumer Disconnect FAILED.\n");
        consumerStatus = -1;
        DoneCons(consumerStatus, send_fd);
      }
    }
  } else  // Producer
  {
    test_lwda_producer_s lwdaProducer;
    memset(&lwdaProducer, 0, sizeof(test_lwda_producer_s));
    lwdaProducer.profileAPI = profileAPIs;
    int producerStatus = 0;

    setelw("LWDA_EGL_PRODUCER_RETURN_WAIT_TIMEOUT", "1600", 0);

    int connect_fd = -1;
    // Hook up Ctrl-C handler
    signal(SIGINT, sig_handler);

    // Create connection to Consumer
    connect_fd = UnixSocketCreate(SOCK_PATH);
    if (-1 == connect_fd) {
      printf("%s: Lwca Producer could not create socket: %s.\n", __func__,
             SOCK_PATH);
      producerStatus = -1;
      DoneProd(producerStatus, connect_fd);
    }

    // Get the file descriptor of the stream from the consumer process
    // and re-create the EGL stream from it
    fileDescriptor = EGLStreamReceivefd(connect_fd);
    if (-1 == fileDescriptor) {
      printf("%s: Lwca Producer could not receive EGL file descriptor \n",
             __func__);
      producerStatus = -1;
      DoneProd(producerStatus, connect_fd);
    }

    if (!EGLStreamInit(isCrossDevice, 0, fileDescriptor)) {
      printf("EGLStream Init failed.\n");
      producerStatus = -1;
      lwresult = LWDA_ERROR_UNKNOWN;
      DoneProd(producerStatus, connect_fd);
    }

    lwdaProducer.eglDisplay = g_producerEglDisplay;
    lwdaProducer.eglStream = g_producerEglStream;
    lwdaProducer.lwdaDevId = lwdaDevIndexProd;

    lwresult = lwdaDeviceCreateProducer(&lwdaProducer);
    if (lwresult != LWDA_SUCCESS) {
      producerStatus = -1;
      DoneProd(producerStatus, connect_fd);
    }

    args.charCnt = WIDTH * HEIGHT * 4;
    lwCtxPushLwrrent(lwdaProducer.context);
    lwresult = lwdaProducerInit(&lwdaProducer, &args);
    if (lwresult != LWDA_SUCCESS) {
      printf("Lwca Producer: Init failed, status: %d\n", lwresult);
      producerStatus = -1;
      DoneProd(producerStatus, connect_fd);
    }

    // wait for consumer to connect first
    int err = 0;
    int wait_loop = 0;
    EGLint streamState = 0;
    do {
      err = eglQueryStreamKHR(lwdaProducer.eglDisplay, lwdaProducer.eglStream,
                              EGL_STREAM_STATE_KHR, &streamState);
      if ((0 != err) && (EGL_STREAM_STATE_CONNECTING_KHR != streamState)) {
        sleep(1);
        wait_loop++;
      }
    } while ((wait_loop < 10) && (0 != err) &&
             (streamState != EGL_STREAM_STATE_CONNECTING_KHR));

    if ((0 == err) || (wait_loop >= 10)) {
      printf(
          "%s: Lwca Producer eglQueryStreamKHR EGL_STREAM_STATE_KHR failed.\n",
          __func__);
      producerStatus = -1;
      DoneProd(producerStatus, connect_fd);
    }

    if (LWDA_SUCCESS != (lwresult = lwEGLStreamProducerConnect(
                             &(lwdaProducer.lwdaConn), lwdaProducer.eglStream,
                             WIDTH, HEIGHT))) {
      printf("Connect LWCA producer FAILED with error %d\n", lwresult);
      producerStatus = -1;
      DoneProd(producerStatus, connect_fd);
    }

    printf("main - Lwca Producer and Consumer Initialized.\n");

    lwdaProducerPrepareFrame(&lwdaEgl1, lwdaProducer.lwdaPtr, args.charCnt);
    lwdaProducerPrepareFrame(&lwdaEgl2, lwdaProducer.lwdaPtr1, args.charCnt);

    j = 0;
    for (j = 0; j < NUMTRIALS; j++) {
      lwresult = lwdaProducerPresentFrame(&lwdaProducer, lwdaEgl1, j);
      if (lwresult != LWDA_SUCCESS) {
        printf("Lwca Producer Test failed for frame = %d with lwca error:%d\n",
               j + 1, lwresult);
        producerStatus = -1;
        DoneProd(producerStatus, connect_fd);
      }

      lwresult = lwdaProducerPresentFrame(&lwdaProducer, lwdaEgl2, j);
      if (lwresult != LWDA_SUCCESS) {
        printf("Lwca Producer Test failed for frame = %d with lwca error:%d\n",
               j + 1, lwresult);
        producerStatus = -1;
        DoneProd(producerStatus, connect_fd);
      }

      lwresult = lwdaProducerReturnFrame(&lwdaProducer, lwdaEgl1, j);
      if (lwresult != LWDA_SUCCESS) {
        printf("Lwca Producer Test failed for frame = %d with lwca error:%d\n",
               j + 1, lwresult);
        producerStatus = -1;
        DoneProd(producerStatus, connect_fd);
      }

      lwresult = lwdaProducerReturnFrame(&lwdaProducer, lwdaEgl2, j);
      if (lwresult != LWDA_SUCCESS) {
        printf("Lwca Producer Test failed for frame = %d with lwca error:%d\n",
               j + 1, lwresult);
        producerStatus = -1;
        DoneProd(producerStatus, connect_fd);
      }
    }

    lwCtxSynchronize();
    err = lwdaGetValueMismatch();
    if (err != lwdaSuccess) {
      printf("Prod: App failed with value mismatch\n");
      lwresult = LWDA_ERROR_UNKNOWN;
      producerStatus = -1;
      DoneProd(producerStatus, connect_fd);
    }

    printf("Tear Down Start.....\n");
    if (!eglQueryStreamKHR(lwdaProducer.eglDisplay, lwdaProducer.eglStream,
                           EGL_STREAM_STATE_KHR, &streamState)) {
      printf("Main, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
      lwresult = LWDA_ERROR_UNKNOWN;
      producerStatus = -1;
      DoneProd(producerStatus, connect_fd);
    }

    if (streamState != EGL_STREAM_STATE_DISCONNECTED_KHR) {
      if (LWDA_SUCCESS != (lwresult = lwdaProducerDeinit(&lwdaProducer))) {
        printf("Producer Disconnect FAILED with %d\n", lwresult);
        producerStatus = -1;
        DoneProd(producerStatus, connect_fd);
      }
    }
    unsetelw("LWDA_EGL_PRODUCER_RETURN_WAIT_TIMEOUT");
  }

  return 0;
}
