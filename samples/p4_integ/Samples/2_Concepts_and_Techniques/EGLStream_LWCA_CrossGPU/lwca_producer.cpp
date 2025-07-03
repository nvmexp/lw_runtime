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
// DESCRIPTION:   Simple lwca EGL stream producer app
//

#include "lwdaEGL.h"
#include "lwda_producer.h"
#include "eglstrm_common.h"
#include <lwda_runtime.h>
#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>
#include "lwda_runtime.h"
#include "math.h"

int lwdaPresentReturnData = INIT_DATA;
int fakePresent = 0;
LWeglFrame fakeFrame;
LWdeviceptr lwdaPtrFake;
extern bool isCrossDevice;

void lwdaProducerPrepareFrame(LWeglFrame *lwdaEgl, LWdeviceptr lwdaPtr,
                              int bufferSize) {
  lwdaEgl->frame.pPitch[0] = (void *)lwdaPtr;
  lwdaEgl->width = WIDTH;
  lwdaEgl->depth = 0;
  lwdaEgl->height = HEIGHT;
  lwdaEgl->pitch = WIDTH * 4;
  lwdaEgl->frameType = LW_EGL_FRAME_TYPE_PITCH;
  lwdaEgl->planeCount = 1;
  lwdaEgl->numChannels = 4;
  lwdaEgl->eglColorFormat = LW_EGL_COLOR_FORMAT_ARGB;
  lwdaEgl->lwFormat = LW_AD_FORMAT_UNSIGNED_INT8;
}

static int count_present = 0, count_return = 0;
static double present_time[25000] = {0}, total_time_present = 0;
static double return_time[25000] = {0}, total_time_return = 0;

void presentApiStat(void);
void presentApiStat(void) {
  int i = 0;
  double min = 10000000, max = 0;
  double average_launch_time = 0, standard_deviation = 0;
  if (count_present == 0) return;
  // lets compute the standard deviation
  min = max = present_time[1];
  average_launch_time = (total_time_present) / count_present;
  for (i = 1; i < count_present; i++) {
    standard_deviation += (present_time[i] - average_launch_time) *
                          (present_time[i] - average_launch_time);
    if (present_time[i] < min) min = present_time[i];
    if (present_time[i] > max) max = present_time[i];
  }
  standard_deviation = sqrt(standard_deviation / count_present);
  printf("present Avg: %lf\n", average_launch_time);
  printf("present  SD: %lf\n", standard_deviation);
  printf("present min: %lf\n", min);
  printf("present max: %lf\n", max);

  min = max = return_time[1];
  average_launch_time = (total_time_return - return_time[0]) / count_return;
  for (i = 1; i < count_return; i++) {
    standard_deviation += (return_time[i] - average_launch_time) *
                          (return_time[i] - average_launch_time);
    if (return_time[i] < min) min = return_time[i];
    if (return_time[i] > max) max = return_time[i];
  }
  standard_deviation = sqrt(standard_deviation / count_return);
  printf("return  Avg: %lf\n", average_launch_time);
  printf("return   SD: %lf\n", standard_deviation);
  printf("return  min: %lf\n", min);
  printf("return  max: %lf\n", max);
}
LWresult lwdaProducerPresentFrame(test_lwda_producer_s *lwdaProducer,
                                  LWeglFrame lwdaEgl, int t) {
  static int flag = 0;
  LWresult status = LWDA_SUCCESS;
  struct timespec start, end;
  double lwrTime;
  LWdeviceptr pDevPtr = (LWdeviceptr)lwdaEgl.frame.pPitch[0];
  lwdaProducer_filter(lwdaProducer->prodLwdaStream, (char *)pDevPtr, WIDTH * 4,
                      HEIGHT, lwdaPresentReturnData, PROD_DATA + t, t);
  if (lwdaProducer->profileAPI) {
    getTime(&start);
  }
  status = lwEGLStreamProducerPresentFrame(&lwdaProducer->lwdaConn, lwdaEgl,
                                           &lwdaProducer->prodLwdaStream);
  if (status != LWDA_SUCCESS) {
    printf("Lwca Producer: Present frame failed, status:%d\n", status);
    goto done;
  }
  flag++;
  if (lwdaProducer->profileAPI && flag > 10) {
    getTime(&end);
    lwrTime = TIME_DIFF(end, start);
    present_time[count_present++] = lwrTime;
    if (count_present == 25000) count_present = 0;
    total_time_present += lwrTime;
  }
done:
  return status;
}

int flag = 0;
LWresult lwdaProducerReturnFrame(test_lwda_producer_s *lwdaProducer,
                                 LWeglFrame lwdaEgl, int t) {
  LWresult status = LWDA_SUCCESS;
  struct timespec start, end;
  double lwrTime;
  LWdeviceptr pDevPtr = 0;

  pDevPtr = (LWdeviceptr)lwdaEgl.frame.pPitch[0];
  if (lwdaProducer->profileAPI) {
    getTime(&start);
  }

  while (1) {
    status = lwEGLStreamProducerReturnFrame(&lwdaProducer->lwdaConn, &lwdaEgl,
                                            &lwdaProducer->prodLwdaStream);
    if (status == LWDA_ERROR_LAUNCH_TIMEOUT) {
      continue;
    } else if (status != LWDA_SUCCESS) {
      printf("Lwca Producer: Return frame failed, status:%d\n", status);
      goto done;
    }
    break;
  }
  if (lwdaProducer->profileAPI) {
    getTime(&end);
    lwrTime = TIME_DIFF(end, start);
    return_time[count_return++] = lwrTime;
    if (count_return == 25000) count_return = 0;
    total_time_return += lwrTime;
  }
  if (flag % 2 == 0) {
    lwdaPresentReturnData++;
  }
  lwdaProducer_filter(lwdaProducer->prodLwdaStream, (char *)pDevPtr, WIDTH * 4,
                      HEIGHT, CONS_DATA + t, lwdaPresentReturnData, t);
  flag++;
done:
  return status;
}

LWresult lwdaDeviceCreateProducer(test_lwda_producer_s *lwdaProducer) {
  LWdevice device;
  LWresult status = LWDA_SUCCESS;

  if (LWDA_SUCCESS != (status = lwInit(0))) {
    printf("Failed to initialize LWCA\n");
    return status;
  }

  if (LWDA_SUCCESS !=
      (status = lwDeviceGet(&device, lwdaProducer->lwdaDevId))) {
    printf("failed to get LWCA device\n");
    return status;
  }

  if (LWDA_SUCCESS !=
      (status = lwCtxCreate(&lwdaProducer->context, 0, device))) {
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
      "LWCA Producer on GPU Device %d: \"%s\" with compute capability "
      "%d.%d\n\n",
      device, deviceName, major, minor);

  lwCtxPopLwrrent(&lwdaProducer->context);

  if (major < 6) {
    printf(
        "EGLStream_LWDA_CrossGPU requires SM 6.0 or higher arch GPU.  "
        "Exiting...\n");
    exit(2);  // EXIT_WAIVED
  }

  return status;
}

LWresult lwdaProducerInit(test_lwda_producer_s *lwdaProducer, TestArgs *args) {
  LWresult status = LWDA_SUCCESS;
  int bufferSize;

  lwdaProducer->charCnt = args->charCnt;
  bufferSize = lwdaProducer->charCnt;

  lwdaProducer->tempBuff = (char *)malloc(bufferSize);
  if (!lwdaProducer->tempBuff) {
    printf("Lwca Producer: Failed to allocate image buffer\n");
    status = LWDA_ERROR_UNKNOWN;
    goto done;
  }
  memset((void *)lwdaProducer->tempBuff, INIT_DATA, lwdaProducer->charCnt);

  // Fill this init data
  status = lwMemAlloc(&lwdaProducer->lwdaPtr, bufferSize);
  if (status != LWDA_SUCCESS) {
    printf("Lwca Producer: lwca Malloc failed, status:%d\n", status);
    goto done;
  }
  status = lwMemcpyHtoD(lwdaProducer->lwdaPtr, (void *)(lwdaProducer->tempBuff),
                        bufferSize);
  if (status != LWDA_SUCCESS) {
    printf("Lwca Producer: lwMemCpy failed, status:%d\n", status);
    goto done;
  }

  // Fill this init data
  status = lwMemAlloc(&lwdaProducer->lwdaPtr1, bufferSize);
  if (status != LWDA_SUCCESS) {
    printf("Lwca Producer: lwca Malloc failed, status:%d\n", status);
    goto done;
  }
  status = lwMemcpyHtoD(lwdaProducer->lwdaPtr1,
                        (void *)(lwdaProducer->tempBuff), bufferSize);
  if (status != LWDA_SUCCESS) {
    printf("Lwca Producer: lwMemCpy failed, status:%d\n", status);
    goto done;
  }

  status = lwStreamCreate(&lwdaProducer->prodLwdaStream, 0);
  if (status != LWDA_SUCCESS) {
    printf("Lwca Producer: lwStreamCreate failed, status:%d\n", status);
    goto done;
  }

  // Fill this init data
  status = lwMemAlloc(&lwdaPtrFake, 100);
  if (status != LWDA_SUCCESS) {
    printf("Lwca Producer: lwca Malloc failed, status:%d\n", status);
    goto done;
  }

  atexit(presentApiStat);
done:
  return status;
}

LWresult lwdaProducerDeinit(test_lwda_producer_s *lwdaProducer) {
  if (lwdaProducer->tempBuff) {
    free(lwdaProducer->tempBuff);
  }
  if (lwdaProducer->lwdaPtr) {
    lwMemFree(lwdaProducer->lwdaPtr);
  }
  return lwEGLStreamProducerDisconnect(&lwdaProducer->lwdaConn);
}
