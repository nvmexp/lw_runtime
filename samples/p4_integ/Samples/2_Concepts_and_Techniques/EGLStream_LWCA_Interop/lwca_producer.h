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
// DESCRIPTION:   Simple lwca producer header file
//

#ifndef _LWDA_PRODUCER_H_
#define _LWDA_PRODUCER_H_
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "lwdaEGL.h"
#include "eglstrm_common.h"

extern EGLStreamKHR eglStream;
extern EGLDisplay g_display;

typedef struct _test_lwda_producer_s {
  //  Stream params
  char *fileName1;
  char *fileName2;
  unsigned char *pBuff;
  int frameCount;
  bool isARGB;
  bool pitchLinearOutput;
  unsigned int width;
  unsigned int height;
  LWcontext context;
  LWeglStreamConnection lwdaConn;
  LWdeviceptr lwdaPtrARGB[1];
  LWdeviceptr lwdaPtrYUV[3];
  LWarray lwdaArrARGB[1];
  LWarray lwdaArrYUV[3];
  EGLStreamKHR eglStream;
  EGLDisplay eglDisplay;
} test_lwda_producer_s;

void lwdaProducerInit(test_lwda_producer_s *lwdaProducer, EGLDisplay eglDisplay,
                      EGLStreamKHR eglStream, TestArgs *args);
LWresult lwdaProducerTest(test_lwda_producer_s *parserArg, char *file);
LWresult lwdaProducerDeinit(test_lwda_producer_s *lwdaProducer);
LWresult lwdaDeviceCreateProducer(test_lwda_producer_s *lwdaProducer,
                                  LWdevice device);
#endif
