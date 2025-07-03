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

#include "lwda_consumer.h"
#include <helper_lwda_drvapi.h>
#include "eglstrm_common.h"

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

int checkbuf(FILE *fp1, FILE *fp2);

LWresult lwdaConsumerTest(test_lwda_consumer_s *data, char *fileName) {
  LWresult lwStatus = LWDA_SUCCESS;
  LWarray lwdaArr = NULL;
  LWeglFrame lwdaEgl;
  LWgraphicsResource lwdaResource;
  unsigned int i;
  int check_result;
  FILE *pInFile1 = NULL, *pInFile2 = NULL, *file_p = NULL;
  EGLint streamState = 0;

  if (!data) {
    printf("%s: Bad parameter\n", __func__);
    goto done;
  }

  if (!eglQueryStreamKHR(g_display, eglStream, EGL_STREAM_STATE_KHR,
                         &streamState)) {
    printf("Lwca consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
  }
  if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR) {
    printf("LWCA Consumer: - EGL_STREAM_STATE_DISCONNECTED_KHR received\n");
  }

  if (streamState == EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR) {
    lwStatus = lwEGLStreamConsumerAcquireFrame(&(data->lwdaConn), &lwdaResource,
                                               NULL, 16000);

    if (lwStatus == LWDA_SUCCESS) {
      LWdeviceptr pDevPtr = 0;
      int bufferSize;
      unsigned char *pLwdaCopyMem = NULL;
      unsigned int copyWidthInBytes = 0, copyHeight = 0;

      file_p = fopen(fileName, "wb+");
      if (!file_p) {
        printf("WriteFrame: file open failed %s\n", fileName);
        lwStatus = LWDA_ERROR_UNKNOWN;
        goto done;
      }
      lwStatus =
          lwGraphicsResourceGetMappedEglFrame(&lwdaEgl, lwdaResource, 0, 0);
      if (lwStatus != LWDA_SUCCESS) {
        printf("Lwca get resource failed with %d\n", lwStatus);
        goto done;
      }
      lwStatus = lwCtxSynchronize();
      if (lwStatus != LWDA_SUCCESS) {
        printf("lwCtxSynchronize failed \n");
        goto done;
      }
      if (!(lwdaEgl.planeCount >= 1 && lwdaEgl.planeCount <= 3)) {
        printf("Plane count is invalid\nExiting\n");
        goto done;
      }

      for (i = 0; i < lwdaEgl.planeCount; i++) {
        if (lwdaEgl.frameType == LW_EGL_FRAME_TYPE_PITCH) {
          pDevPtr = (LWdeviceptr)lwdaEgl.frame.pPitch[i];
          if (lwdaEgl.planeCount == 1) {
            bufferSize = lwdaEgl.pitch * lwdaEgl.height;
            copyWidthInBytes = lwdaEgl.pitch;
            copyHeight = data->height;
          } else if (i == 1 && lwdaEgl.planeCount == 2) {  // YUV 420
                                                           // semi-planar
            bufferSize = lwdaEgl.pitch * lwdaEgl.height / 2;
            copyWidthInBytes = lwdaEgl.pitch;
            copyHeight = data->height / 2;
          } else {
            bufferSize = data->width * data->height;
            copyWidthInBytes = data->width;
            copyHeight = data->height;
            if (i > 0) {
              bufferSize >>= 2;
              copyWidthInBytes >>= 1;
              copyHeight >>= 1;
            }
          }
        } else {
          lwdaArr = lwdaEgl.frame.pArray[i];
          if (lwdaEgl.planeCount == 1) {
            bufferSize = data->width * data->height * 4;
            copyWidthInBytes = data->width * 4;
            copyHeight = data->height;
          } else if (i == 1 && lwdaEgl.planeCount == 2) {  // YUV 420
                                                           // semi-planar
            bufferSize = data->width * data->height / 2;
            copyWidthInBytes = data->width;
            copyHeight = data->height / 2;
          } else {
            bufferSize = data->width * data->height;
            copyWidthInBytes = data->width;
            copyHeight = data->height;
            if (i > 0) {
              bufferSize >>= 2;
              copyWidthInBytes >>= 1;
              copyHeight >>= 1;
            }
          }
        }
        if (i == 0) {
          pLwdaCopyMem = (unsigned char *)malloc(bufferSize);
          if (pLwdaCopyMem == NULL) {
            printf("pLwdaCopyMem malloc failed\n");
            goto done;
          }
        }
        memset(pLwdaCopyMem, 0, bufferSize);
        if (data->pitchLinearOutput) {
          lwStatus = lwMemcpyDtoH(pLwdaCopyMem, pDevPtr, bufferSize);
          if (lwStatus != LWDA_SUCCESS) {
            printf(
                "lwda_consumer: pitch linear Memcpy failed, bufferSize =%d\n",
                bufferSize);
            goto done;
          }
          lwStatus = lwCtxSynchronize();
          if (lwStatus != LWDA_SUCCESS) {
            printf("lwda_consumer: lwCtxSynchronize failed after memcpy \n");
            goto done;
          }
        } else {
          LWDA_MEMCPY3D cpdesc;
          memset(&cpdesc, 0, sizeof(cpdesc));
          cpdesc.srcXInBytes = cpdesc.srcY = cpdesc.srcZ = cpdesc.srcLOD = 0;
          cpdesc.srcMemoryType = LW_MEMORYTYPE_ARRAY;
          cpdesc.srcArray = lwdaArr;
          cpdesc.dstXInBytes = cpdesc.dstY = cpdesc.dstZ = cpdesc.dstLOD = 0;
          cpdesc.dstMemoryType = LW_MEMORYTYPE_HOST;
          cpdesc.dstHost = (void *)pLwdaCopyMem;
          cpdesc.WidthInBytes = copyWidthInBytes;  // data->width * 4;
          cpdesc.Height = copyHeight;              // data->height;
          cpdesc.Depth = 1;

          lwStatus = lwMemcpy3D(&cpdesc);
          if (lwStatus != LWDA_SUCCESS) {
            printf(
                "Lwca consumer: lwMemCpy3D failed,  copyWidthInBytes=%d, "
                "copyHight=%d\n",
                copyWidthInBytes, copyHeight);
          }
          lwStatus = lwCtxSynchronize();
          if (lwStatus != LWDA_SUCCESS) {
            printf("lwCtxSynchronize failed after memcpy \n");
          }
        }
        if (lwStatus == LWDA_SUCCESS) {
          if (fwrite(pLwdaCopyMem, bufferSize, 1, file_p) != 1) {
            printf("Lwca consumer: output file write failed\n");
            lwStatus = LWDA_ERROR_UNKNOWN;
            goto done;
          }
        }
      }
      pInFile1 = fopen(data->fileName1, "rb");
      if (!pInFile1) {
        printf("Failed to open file :%s\n", data->fileName1);
        goto done;
      }
      pInFile2 = fopen(data->fileName2, "rb");
      if (!pInFile2) {
        printf("Failed to open file :%s\n", data->fileName2);
        goto done;
      }
      rewind(file_p);
      check_result = checkbuf(file_p, pInFile1);
      if (check_result == -1) {
        rewind(file_p);
        check_result = checkbuf(file_p, pInFile2);
        if (check_result == -1) {
          printf("Frame received does not match any valid image: FAILED\n");
        } else {
          printf("Frame check Passed\n");
        }
      } else {
        printf("Frame check Passed\n");
      }
      if (pLwdaCopyMem) {
        free(pLwdaCopyMem);
        pLwdaCopyMem = NULL;
      }
      lwStatus =
          lwEGLStreamConsumerReleaseFrame(&data->lwdaConn, lwdaResource, NULL);
      if (lwStatus != LWDA_SUCCESS) {
        printf("lwEGLStreamConsumerReleaseFrame failed with lwStatus = %d\n",
               lwStatus);
        goto done;
      }
    } else {
      printf("lwca AcquireFrame FAILED with  lwStatus=%d\n", lwStatus);
      goto done;
    }
  }

done:
  if (file_p) {
    fclose(file_p);
    file_p = NULL;
  }
  if (pInFile1) {
    fclose(pInFile1);
    pInFile1 = NULL;
  }
  if (pInFile1) {
    fclose(pInFile2);
    pInFile2 = NULL;
  }
  return lwStatus;
}

int checkbuf(FILE *fp1, FILE *fp2) {
  int match = 0;
  int ch1, ch2;
  if (fp1 == NULL) {
    printf("Invalid file pointer for first file\n");
    return -1;
  } else if (fp2 == NULL) {
    printf("Invalid file pointer for second file\n");
    return -1;
  } else {
    ch1 = getc(fp1);
    ch2 = getc(fp2);
    while ((ch1 != EOF) && (ch2 != EOF) && (ch1 == ch2)) {
      ch1 = getc(fp1);
      ch2 = getc(fp2);
    }
    if (ch1 == ch2) {
      match = 1;
    } else if (ch1 != ch2) {
      match = -1;
    }
  }
  return match;
}

LWresult lwdaDeviceCreateConsumer(test_lwda_consumer_s *lwdaConsumer,
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
      "LWCA Consumer on GPU Device %d: \"%s\" with compute capability "
      "%d.%d\n\n",
      device, deviceName, major, minor);

  if (LWDA_SUCCESS !=
      (status = lwCtxCreate(&lwdaConsumer->context, 0, device))) {
    printf("failed to create LWCA context\n");
    return status;
  }
  checkLwdaErrors(lwCtxPopLwrrent(&lwdaConsumer->context));
  return status;
}

void lwda_consumer_init(test_lwda_consumer_s *lwdaConsumer, TestArgs *args) {
  lwdaConsumer->pitchLinearOutput = args->pitchLinearOutput;
  lwdaConsumer->width = args->inputWidth;
  lwdaConsumer->height = args->inputHeight;
  lwdaConsumer->fileName1 = args->infile1;
  lwdaConsumer->fileName2 = args->infile2;

  lwdaConsumer->outFile1 = "lwda_out1.yuv";
  lwdaConsumer->outFile2 = "lwda_out2.yuv";
}

LWresult lwda_consumer_deinit(test_lwda_consumer_s *lwdaConsumer) {
  return lwEGLStreamConsumerDisconnect(&lwdaConsumer->lwdaConn);
}
