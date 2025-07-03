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
// DESCRIPTION:   Simple LWCA consumer rendering sample app
//

#include <lwda_runtime.h>
#include "lwda_consumer.h"
#include "eglstrm_common.h"
#include <math.h>
#include <unistd.h>

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif
LWgraphicsResource lwdaResource;

static int count_acq = 0;
static double acquire_time[25000] = {0}, total_time_acq = 0;

static int count_rel = 0;
static double rel_time[25000] = {0}, total_time_rel = 0;

void acquireApiStat(void);
void acquireApiStat(void) {
  int i = 0;
  double min = 10000000, max = 0;
  double average_launch_time = 0, standard_deviation = 0;
  if (count_acq == 0) return;
  // lets compute the standard deviation
  min = max = acquire_time[1];
  average_launch_time = (total_time_acq - acquire_time[0]) / count_acq;
  for (i = 1; i < count_acq; i++) {
    standard_deviation += (acquire_time[i] - average_launch_time) *
                          (acquire_time[i] - average_launch_time);
    if (acquire_time[i] < min) min = acquire_time[i];
    if (acquire_time[i] > max) max = acquire_time[i];
  }
  standard_deviation = sqrt(standard_deviation / count_acq);
  printf("acquire Avg: %lf\n", average_launch_time);
  printf("acquire  SD: %lf\n", standard_deviation);
  printf("acquire min: %lf\n", min);
  printf("acquire max: %lf\n", max);

  min = max = rel_time[1];
  average_launch_time = (total_time_rel - rel_time[0]) / count_rel;
  for (i = 1; i < count_rel; i++) {
    standard_deviation += (rel_time[i] - average_launch_time) *
                          (rel_time[i] - average_launch_time);
    if (rel_time[i] < min) min = rel_time[i];
    if (rel_time[i] > max) max = rel_time[i];
  }
  standard_deviation = sqrt(standard_deviation / count_rel);
  printf("release Avg: %lf\n", average_launch_time);
  printf("release  SD: %lf\n", standard_deviation);
  printf("release min: %lf\n", min);
  printf("release max: %lf\n", max);
}
LWresult lwdaConsumerAcquireFrame(test_lwda_consumer_s *lwdaConsumer,
                                  int frameNumber) {
  LWresult lwStatus = LWDA_SUCCESS;
  LWeglFrame lwdaEgl;
  struct timespec start, end;
  EGLint streamState = 0;
  double lwrTime;

  if (!lwdaConsumer) {
    printf("%s: Bad parameter\n", __func__);
    goto done;
  }

  while (1) {
    if (!eglQueryStreamKHR(lwdaConsumer->eglDisplay, lwdaConsumer->eglStream,
                           EGL_STREAM_STATE_KHR, &streamState)) {
      printf("Lwca Consumer: eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
      lwStatus = LWDA_ERROR_UNKNOWN;
      goto done;
    }
    if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR) {
      printf("Lwca Consumer: EGL_STREAM_STATE_DISCONNECTED_KHR received\n");
      lwStatus = LWDA_ERROR_UNKNOWN;
      goto done;
    }

    if (streamState == EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR) {
      break;
    }
  }
  if (lwdaConsumer->profileAPI) {
    getTime(&start);
  }
  lwStatus =
      lwEGLStreamConsumerAcquireFrame(&(lwdaConsumer->lwdaConn), &lwdaResource,
                                      &lwdaConsumer->consLwdaStream, 16000);
  if (lwdaConsumer->profileAPI) {
    getTime(&end);
    lwrTime = TIME_DIFF(end, start);
    acquire_time[count_acq++] = lwrTime;
    if (count_acq == 25000) count_acq = 0;
    total_time_acq += lwrTime;
  }
  if (lwStatus == LWDA_SUCCESS) {
    LWdeviceptr pDevPtr = 0;
    lwdaError_t err;

    lwStatus =
        lwGraphicsResourceGetMappedEglFrame(&lwdaEgl, lwdaResource, 0, 0);
    if (lwStatus != LWDA_SUCCESS) {
      printf("Lwca get resource failed with %d\n", lwStatus);
      goto done;
    }
    pDevPtr = (LWdeviceptr)lwdaEgl.frame.pPitch[0];

    err = lwdaConsumer_filter(lwdaConsumer->consLwdaStream, (char *)pDevPtr,
                              WIDTH * 4, HEIGHT, PROD_DATA + frameNumber,
                              CONS_DATA + frameNumber, frameNumber);
    if (err != lwdaSuccess) {
      printf("Lwca Consumer: kernel failed with: %s\n",
             lwdaGetErrorString(err));
      goto done;
    }
  }

done:
  return lwStatus;
}

LWresult lwdaConsumerReleaseFrame(test_lwda_consumer_s *lwdaConsumer,
                                  int frameNumber) {
  LWresult lwStatus = LWDA_SUCCESS;
  struct timespec start, end;
  double lwrTime;

  if (!lwdaConsumer) {
    printf("%s: Bad parameter\n", __func__);
    goto done;
  }
  if (lwdaConsumer->profileAPI) {
    getTime(&start);
  }
  lwStatus = lwEGLStreamConsumerReleaseFrame(
      &lwdaConsumer->lwdaConn, lwdaResource, &lwdaConsumer->consLwdaStream);
  if (lwdaConsumer->profileAPI) {
    getTime(&end);
    lwrTime = TIME_DIFF(end, start);
    rel_time[count_rel++] = lwrTime;
    if (count_rel == 25000) count_rel = 0;
    total_time_rel += lwrTime;
  }
  if (lwStatus != LWDA_SUCCESS) {
    printf("lwEGLStreamConsumerReleaseFrame failed, status:%d\n", lwStatus);
    goto done;
  }

done:
  return lwStatus;
}

LWresult lwdaDeviceCreateConsumer(test_lwda_consumer_s *lwdaConsumer) {
  LWdevice device;
  LWresult status = LWDA_SUCCESS;

  if (LWDA_SUCCESS != (status = lwInit(0))) {
    printf("Failed to initialize LWCA\n");
    return status;
  }

  if (LWDA_SUCCESS !=
      (status = lwDeviceGet(&device, lwdaConsumer->lwdaDevId))) {
    printf("failed to get LWCA device\n");
    return status;
  }

  if (LWDA_SUCCESS !=
      (status = lwCtxCreate(&lwdaConsumer->context, 0, device))) {
    printf("failed to create LWCA context\n");
    return status;
  }

  int major = 0, minor = 0;
  char deviceName[256];
  lwDeviceGetAttribute(&major, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                       device);
  lwDeviceGetAttribute(&minor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                       device);
  lwDeviceGetName(deviceName, 256, device);
  printf(
      "LWCA Consumer on GPU Device %d: \"%s\" with compute capability "
      "%d.%d\n\n",
      device, deviceName, major, minor);

  lwCtxPopLwrrent(&lwdaConsumer->context);
  if (major < 6) {
    printf(
        "EGLStream_LWDA_CrossGPU requires SM 6.0 or higher arch GPU.  "
        "Exiting...\n");
    exit(2);  // EXIT_WAIVED
  }

  return status;
}

LWresult lwda_consumer_init(test_lwda_consumer_s *lwdaConsumer,
                            TestArgs *args) {
  LWresult status = LWDA_SUCCESS;
  int bufferSize;

  lwdaConsumer->charCnt = args->charCnt;
  bufferSize = args->charCnt;

  lwdaConsumer->pLwdaCopyMem = (unsigned char *)malloc(bufferSize);
  if (lwdaConsumer->pLwdaCopyMem == NULL) {
    printf("Lwca Consumer: malloc failed\n");
    goto done;
  }

  status = lwStreamCreate(&lwdaConsumer->consLwdaStream, 0);
  if (status != LWDA_SUCCESS) {
    printf("Lwca Consumer: lwStreamCreate failed, status:%d\n", status);
    goto done;
  }

  atexit(acquireApiStat);
done:
  return status;
}

LWresult lwda_consumer_Deinit(test_lwda_consumer_s *lwdaConsumer) {
  if (lwdaConsumer->pLwdaCopyMem) {
    free(lwdaConsumer->pLwdaCopyMem);
  }
  return lwEGLStreamConsumerDisconnect(&lwdaConsumer->lwdaConn);
}
