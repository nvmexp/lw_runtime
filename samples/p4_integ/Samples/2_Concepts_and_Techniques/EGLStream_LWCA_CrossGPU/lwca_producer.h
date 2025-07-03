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
#include <lwda_runtime.h>
#include <lwca.h>

typedef struct _test_lwda_producer_s {
  //  Stream params
  LWcontext context;
  LWeglStreamConnection lwdaConn;
  int lwdaDevId;
  EGLStreamKHR eglStream;
  EGLDisplay eglDisplay;
  unsigned int charCnt;
  bool profileAPI;
  char *tempBuff;
  LWdeviceptr lwdaPtr;
  LWdeviceptr lwdaPtr1;
  LWstream prodLwdaStream;
} test_lwda_producer_s;

LWresult lwdaProducerInit(test_lwda_producer_s *lwdaProducer, TestArgs *args);
LWresult lwdaProducerPresentFrame(test_lwda_producer_s *parserArg,
                                  LWeglFrame lwdaEgl, int t);
LWresult lwdaProducerReturnFrame(test_lwda_producer_s *parserArg,
                                 LWeglFrame lwdaEgl, int t);
LWresult lwdaProducerDeinit(test_lwda_producer_s *lwdaProducer);
LWresult lwdaDeviceCreateProducer(test_lwda_producer_s *lwdaProducer);
lwdaError_t lwdaProducer_filter(LWstream cStream, char *pSrc, int width,
                                int height, char expectedVal, char newVal,
                                int frameNumber);
void lwdaProducerPrepareFrame(LWeglFrame *lwdaEgl, LWdeviceptr lwdaPtr,
                              int bufferSize);
#endif
