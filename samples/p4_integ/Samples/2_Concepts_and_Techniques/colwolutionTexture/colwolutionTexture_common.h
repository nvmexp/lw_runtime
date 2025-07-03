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

#ifndef COLWOLUTIONTEXTURE_COMMON_H
#define COLWOLUTIONTEXTURE_COMMON_H

#include <lwda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Colwolution kernel size (the only parameter inlined in the code)
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

////////////////////////////////////////////////////////////////////////////////
// Reference CPU colwolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void colwolutionRowsCPU(float *h_Dst, float *h_Src, float *h_Kernel,
                                   int imageW, int imageH, int kernelR);

extern "C" void colwolutionColumnsCPU(float *h_Dst, float *h_Src,
                                      float *h_Kernel, int imageW, int imageH,
                                      int kernelR);

////////////////////////////////////////////////////////////////////////////////
// GPU texture-based colwolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void setColwolutionKernel(float *h_Kernel);

extern "C" void colwolutionRowsGPU(float *d_Dst, lwdaArray *a_Src, int imageW,
                                   int imageH, lwdaTextureObject_t texSrc);

extern "C" void colwolutionColumnsGPU(float *d_Dst, lwdaArray *a_Src,
                                      int imageW, int imageH,
                                      lwdaTextureObject_t texSrc);

#endif
