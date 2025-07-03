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

#ifndef _BINDLESSTEXTURE_LW_
#define _BINDLESSTEXTURE_LW_

// includes, lwca
#include <vector_types.h>
#include <lwda_runtime.h>

// LWCA utilities and system includes
#include <helper_lwda.h>
#include <vector_types.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#pragma pack(push, 4)
struct Image {
  void *h_data;
  lwdaExtent size;
  lwdaResourceType type;
  lwdaArray_t dataArray;
  lwdaMipmappedArray_t mipmapArray;
  lwdaTextureObject_t textureObject;

  Image() { memset(this, 0, sizeof(Image)); }
};
#pragma pack(pop)

inline void _checkHost(bool test, const char *condition, const char *file,
                       int line, const char *func) {
  if (!test) {
    fprintf(stderr, "HOST error at %s:%d (%s) \"%s\" \n", file, line, condition,
            func);
    exit(EXIT_FAILURE);
  }
}

#define checkHost(condition) \
  _checkHost(condition, #condition, __FILE__, __LINE__, __FUNCTION__)

#endif
