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

#include "lwda_producer.h"
#include <helper_lwda_drvapi.h>
#include "lwdaEGL.h"
#include "eglstrm_common.h"

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

static LWresult lwdaProducerReadYUVFrame(FILE *file, unsigned int frameNum,
                                         unsigned int width,
                                         unsigned int height,
                                         unsigned char *pBuff) {
  int bOrderUV = 0;
  unsigned char *pYBuff, *pUBuff, *pVBuff, *pChroma;
  unsigned int frameSize = (width * height * 3) / 2;
  LWresult ret = LWDA_SUCCESS;
  unsigned int i;

  if (!pBuff || !file) return LWDA_ERROR_FILE_NOT_FOUND;

  pYBuff = pBuff;

  // YVU order in the buffer
  pVBuff = pYBuff + width * height;
  pUBuff = pVBuff + width * height / 4;

  if (fseek(file, frameNum * frameSize, SEEK_SET)) {
    printf("ReadYUVFrame: Error seeking file: %p\n", file);
    ret = LWDA_ERROR_NOT_PERMITTED;
    goto done;
  }
  // read Y U V separately
  for (i = 0; i < height; i++) {
    if (fread(pYBuff, width, 1, file) != 1) {
      printf("ReadYUVFrame: Error reading file: %p\n", file);
      ret = LWDA_ERROR_NOT_PERMITTED;
      goto done;
    }
    pYBuff += width;
  }

  pChroma = bOrderUV ? pUBuff : pVBuff;
  for (i = 0; i < height / 2; i++) {
    if (fread(pChroma, width / 2, 1, file) != 1) {
      printf("ReadYUVFrame: Error reading file: %p\n", file);
      ret = LWDA_ERROR_NOT_PERMITTED;
      goto done;
    }
    pChroma += width / 2;
  }

  pChroma = bOrderUV ? pVBuff : pUBuff;
  for (i = 0; i < height / 2; i++) {
    if (fread(pChroma, width / 2, 1, file) != 1) {
      printf("ReadYUVFrame: Error reading file: %p\n", file);
      ret = LWDA_ERROR_NOT_PERMITTED;
      goto done;
    }
    pChroma += width / 2;
  }
done:
  return ret;
}

static LWresult lwdaProducerReadARGBFrame(FILE *file, unsigned int frameNum,
                                          unsigned int width,
                                          unsigned int height,
                                          unsigned char *pBuff) {
  unsigned int frameSize = width * height * 4;
  LWresult ret = LWDA_SUCCESS;

  if (!pBuff || !file) return LWDA_ERROR_FILE_NOT_FOUND;

  if (fseek(file, frameNum * frameSize, SEEK_SET)) {
    printf("ReadYUVFrame: Error seeking file: %p\n", file);
    ret = LWDA_ERROR_NOT_PERMITTED;
    goto done;
  }

  // read ARGB data
  if (fread(pBuff, frameSize, 1, file) != 1) {
    if (feof(file))
      printf("ReadARGBFrame: file read to the end\n");
    else
      printf("ReadARGBFrame: Error reading file: %p\n", file);
    ret = LWDA_ERROR_NOT_PERMITTED;
    goto done;
  }
done:
  return ret;
}

LWresult lwdaProducerTest(test_lwda_producer_s *lwdaProducer, char *file) {
  int framenum = 0;
  LWarray lwdaArr[3] = {0};
  LWdeviceptr lwdaPtr[3] = {0, 0, 0};
  unsigned int bufferSize;
  LWresult lwStatus = LWDA_SUCCESS;
  unsigned int i, surfNum, uvOffset[3] = {0};
  unsigned int copyWidthInBytes[3] = {0, 0, 0}, copyHeight[3] = {0, 0, 0};
  LWeglColorFormat eglColorFormat;
  FILE *file_p;
  LWeglFrame lwdaEgl;
  LWcontext oldContext;

  file_p = fopen(file, "rb");
  if (!file_p) {
    printf("LwdaProducer: Error opening file: %s\n", file);
    goto done;
  }

  if (lwdaProducer->pitchLinearOutput) {
    if (lwdaProducer->isARGB) {
      lwdaPtr[0] = lwdaProducer->lwdaPtrARGB[0];
    } else {  // YUV case
      for (i = 0; i < 3; i++) {
        if (i == 0) {
          bufferSize = lwdaProducer->width * lwdaProducer->height;
        } else {
          bufferSize = lwdaProducer->width * lwdaProducer->height / 4;
        }

        lwdaPtr[i] = lwdaProducer->lwdaPtrYUV[i];
      }
    }
  } else {
    if (lwdaProducer->isARGB) {
      lwdaArr[0] = lwdaProducer->lwdaArrARGB[0];
    } else {
      for (i = 0; i < 3; i++) {
        lwdaArr[i] = lwdaProducer->lwdaArrYUV[i];
      }
    }
  }
  uvOffset[0] = 0;
  if (lwdaProducer->isARGB) {
    if (LWDA_SUCCESS !=
        lwdaProducerReadARGBFrame(file_p, framenum, lwdaProducer->width,
                                  lwdaProducer->height, lwdaProducer->pBuff)) {
      printf("lwca producer, read ARGB frame failed\n");
      goto done;
    }
    copyWidthInBytes[0] = lwdaProducer->width * 4;
    copyHeight[0] = lwdaProducer->height;
    surfNum = 1;
    eglColorFormat = LW_EGL_COLOR_FORMAT_ARGB;
  } else {
    if (LWDA_SUCCESS !=
        lwdaProducerReadYUVFrame(file_p, framenum, lwdaProducer->width,
                                 lwdaProducer->height, lwdaProducer->pBuff)) {
      printf("lwca producer, reading YUV frame failed\n");
      goto done;
    }
    surfNum = 3;
    eglColorFormat = LW_EGL_COLOR_FORMAT_YUV420_PLANAR;
    copyWidthInBytes[0] = lwdaProducer->width;
    copyHeight[0] = lwdaProducer->height;
    copyWidthInBytes[1] = lwdaProducer->width / 2;
    copyHeight[1] = lwdaProducer->height / 2;
    copyWidthInBytes[2] = lwdaProducer->width / 2;
    copyHeight[2] = lwdaProducer->height / 2;
    uvOffset[1] = lwdaProducer->width * lwdaProducer->height;
    uvOffset[2] =
        uvOffset[1] + lwdaProducer->width / 2 * lwdaProducer->height / 2;
  }
  if (lwdaProducer->pitchLinearOutput) {
    for (i = 0; i < surfNum; i++) {
      lwStatus =
          lwMemcpy(lwdaPtr[i], (LWdeviceptr)(lwdaProducer->pBuff + uvOffset[i]),
                   copyWidthInBytes[i] * copyHeight[i]);

      if (lwStatus != LWDA_SUCCESS) {
        printf("Lwca producer: lwMemCpy pitchlinear failed, lwStatus =%d\n",
               lwStatus);
        goto done;
      }
    }
  } else {
    // copy lwdaProducer->pBuff to lwdaArray
    LWDA_MEMCPY3D cpdesc;
    for (i = 0; i < surfNum; i++) {
      memset(&cpdesc, 0, sizeof(cpdesc));
      cpdesc.srcXInBytes = cpdesc.srcY = cpdesc.srcZ = cpdesc.srcLOD = 0;
      cpdesc.srcMemoryType = LW_MEMORYTYPE_HOST;
      cpdesc.srcHost = (void *)(lwdaProducer->pBuff + uvOffset[i]);
      cpdesc.dstXInBytes = cpdesc.dstY = cpdesc.dstZ = cpdesc.dstLOD = 0;
      cpdesc.dstMemoryType = LW_MEMORYTYPE_ARRAY;
      cpdesc.dstArray = lwdaArr[i];
      cpdesc.WidthInBytes = copyWidthInBytes[i];
      cpdesc.Height = copyHeight[i];
      cpdesc.Depth = 1;
      lwStatus = lwMemcpy3D(&cpdesc);
      if (lwStatus != LWDA_SUCCESS) {
        printf("Lwca producer: lwMemCpy failed, lwStatus =%d\n", lwStatus);
        goto done;
      }
    }
  }
  for (i = 0; i < surfNum; i++) {
    if (lwdaProducer->pitchLinearOutput)
      lwdaEgl.frame.pPitch[i] = (void *)lwdaPtr[i];
    else
      lwdaEgl.frame.pArray[i] = lwdaArr[i];
  }
  lwdaEgl.width = copyWidthInBytes[0];
  lwdaEgl.depth = 1;
  lwdaEgl.height = copyHeight[0];
  lwdaEgl.pitch = lwdaProducer->pitchLinearOutput ? lwdaEgl.width : 0;
  lwdaEgl.frameType = lwdaProducer->pitchLinearOutput ? LW_EGL_FRAME_TYPE_PITCH
                                                      : LW_EGL_FRAME_TYPE_ARRAY;
  lwdaEgl.planeCount = surfNum;
  lwdaEgl.numChannels = (eglColorFormat == LW_EGL_COLOR_FORMAT_ARGB) ? 4 : 1;
  lwdaEgl.eglColorFormat = eglColorFormat;
  lwdaEgl.lwFormat = LW_AD_FORMAT_UNSIGNED_INT8;

  static int numFramesPresented = 0;
  // If there is a frame presented before we check if consumer
  // is done with it using lwEGLStreamProducerReturnFrame.
  while (numFramesPresented) {
    LWeglFrame returnedLwdaEgl;
    lwStatus = lwEGLStreamProducerReturnFrame(&lwdaProducer->lwdaConn,
                                              &returnedLwdaEgl, NULL);
    if (lwStatus == LWDA_ERROR_LAUNCH_TIMEOUT) {
      continue;
    } else if (lwStatus != LWDA_SUCCESS) {
      printf("lwca Producer return frame FAILED with lwstatus= %d\n", lwStatus);
      return lwStatus;
    } else {
      numFramesPresented--;
    }
  }

  lwStatus =
      lwEGLStreamProducerPresentFrame(&lwdaProducer->lwdaConn, lwdaEgl, NULL);
  if (lwStatus != LWDA_SUCCESS) {
    printf("lwca Producer present frame FAILED with lwstatus= %d\n", lwStatus);
    goto done;
  }
  numFramesPresented++;

done:
  if (file_p) {
    fclose(file_p);
    file_p = NULL;
  }

  return lwStatus;
}

LWresult lwdaDeviceCreateProducer(test_lwda_producer_s *lwdaProducer,
                                  LWdevice device) {
  LWresult status = LWDA_SUCCESS;
  if (LWDA_SUCCESS != (status = lwInit(0))) {
    printf("Failed to initialize LWCA\n");
    return status;
  }

  int major = 0, minor = 0;
  char deviceName[256];
  checkLwdaErrors(lwDeviceGetAttribute(
      &major, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  checkLwdaErrors(lwDeviceGetAttribute(
      &minor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  checkLwdaErrors(lwDeviceGetName(deviceName, 256, device));
  printf(
      "LWCA Producer on GPU Device %d: \"%s\" with compute capability "
      "%d.%d\n\n",
      device, deviceName, major, minor);

  if (major < 6) {
    printf(
        "EGLStreams_LWDA_Interop requires SM 6.0 or higher arch GPU.  "
        "Exiting...\n");
    exit(2);  // EXIT_WAIVED
  }

  if (LWDA_SUCCESS !=
      (status = lwCtxCreate(&lwdaProducer->context, 0, device))) {
    printf("failed to create LWCA context\n");
    return status;
  }

  status = lwMemAlloc(&lwdaProducer->lwdaPtrARGB[0], (WIDTH * HEIGHT * 4));
  if (status != LWDA_SUCCESS) {
    printf("Create LWCA pointer failed, lwStatus=%d\n", status);
    return status;
  }

  status = lwMemAlloc(&lwdaProducer->lwdaPtrYUV[0], (WIDTH * HEIGHT));
  if (status != LWDA_SUCCESS) {
    printf("Create LWCA pointer failed, lwStatus=%d\n", status);
    return status;
  }
  status = lwMemAlloc(&lwdaProducer->lwdaPtrYUV[1], (WIDTH * HEIGHT) / 4);
  if (status != LWDA_SUCCESS) {
    printf("Create LWCA pointer failed, lwStatus=%d\n", status);
    return status;
  }
  status = lwMemAlloc(&lwdaProducer->lwdaPtrYUV[2], (WIDTH * HEIGHT) / 4);
  if (status != LWDA_SUCCESS) {
    printf("Create LWCA pointer failed, lwStatus=%d\n", status);
    return status;
  }

  LWDA_ARRAY3D_DESCRIPTOR desc = {0};

  desc.Format = LW_AD_FORMAT_UNSIGNED_INT8;
  desc.Depth = 1;
  desc.Flags = LWDA_ARRAY3D_SURFACE_LDST;
  desc.NumChannels = 4;
  desc.Width = WIDTH * 4;
  desc.Height = HEIGHT;
  status = lwArray3DCreate(&lwdaProducer->lwdaArrARGB[0], &desc);
  if (status != LWDA_SUCCESS) {
    printf("Create LWCA array failed, lwStatus=%d\n", status);
    return status;
  }

  for (int i = 0; i < 3; i++) {
    if (i == 0) {
      desc.NumChannels = 1;
      desc.Width = WIDTH;
      desc.Height = HEIGHT;
    } else {  // U/V surface as planar
      desc.NumChannels = 1;
      desc.Width = WIDTH / 2;
      desc.Height = HEIGHT / 2;
    }
    status = lwArray3DCreate(&lwdaProducer->lwdaArrYUV[i], &desc);
    if (status != LWDA_SUCCESS) {
      printf("Create LWCA array failed, lwStatus=%d\n", status);
      return status;
    }
  }

  lwdaProducer->pBuff = (unsigned char *)malloc((WIDTH * HEIGHT * 4));
  if (!lwdaProducer->pBuff) {
    printf("LwdaProducer: Failed to allocate image buffer\n");
  }

  checkLwdaErrors(lwCtxPopLwrrent(&lwdaProducer->context));
  return status;
}

void lwdaProducerInit(test_lwda_producer_s *lwdaProducer, EGLDisplay eglDisplay,
                      EGLStreamKHR eglStream, TestArgs *args) {
  lwdaProducer->fileName1 = args->infile1;
  lwdaProducer->fileName2 = args->infile2;

  lwdaProducer->frameCount = 2;
  lwdaProducer->width = args->inputWidth;
  lwdaProducer->height = args->inputHeight;
  lwdaProducer->isARGB = args->isARGB;
  lwdaProducer->pitchLinearOutput = args->pitchLinearOutput;

  // Set lwdaProducer default parameters
  lwdaProducer->eglDisplay = eglDisplay;
  lwdaProducer->eglStream = eglStream;
}

LWresult lwdaProducerDeinit(test_lwda_producer_s *lwdaProducer) {
  if (lwdaProducer->pBuff) free(lwdaProducer->pBuff);

  checkLwdaErrors(lwMemFree(lwdaProducer->lwdaPtrARGB[0]));
  checkLwdaErrors(lwMemFree(lwdaProducer->lwdaPtrYUV[0]));
  checkLwdaErrors(lwMemFree(lwdaProducer->lwdaPtrYUV[1]));
  checkLwdaErrors(lwMemFree(lwdaProducer->lwdaPtrYUV[2]));
  checkLwdaErrors(lwArrayDestroy(lwdaProducer->lwdaArrARGB[0]));
  checkLwdaErrors(lwArrayDestroy(lwdaProducer->lwdaArrYUV[0]));
  checkLwdaErrors(lwArrayDestroy(lwdaProducer->lwdaArrYUV[1]));
  checkLwdaErrors(lwArrayDestroy(lwdaProducer->lwdaArrYUV[2]));

  return lwEGLStreamProducerDisconnect(&lwdaProducer->lwdaConn);
}
