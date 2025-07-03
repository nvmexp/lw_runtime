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

#ifndef LWDALWSCI_H
#define LWDALWSCI_H

#include <lwda_runtime.h>
#include <helper_lwda.h>
#include <lwscibuf.h>
#include <lwscisync.h>
#include <vector>

#define checkLwSciErrors(call)                              \
  do {                                                      \
    LwSciError _status = call;                              \
    if (LwSciError_Success != _status) {                    \
      printf(                                               \
          "LWSCI call in file '%s' in line %i returned"     \
          " %d, expected %d\n",                             \
          __FILE__, __LINE__, _status, LwSciError_Success); \
      fflush(stdout);                                       \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  } while (0)

extern void rotateKernel(lwdaTextureObject_t &texObj, const float angle,
                         unsigned int *d_outputData, const int imageWidth,
                         const int imageHeight, lwdaStream_t stream);
extern void launchGrayScaleKernel(unsigned int *d_rgbaImage,
                                  std::string image_filename, size_t imageWidth,
                                  size_t imageHeight, lwdaStream_t stream);

class lwdaLwSci {
 private:
  int m_isMultiGPU;
  int m_lwdaLwSciSignalDeviceId;
  int m_lwdaLwSciWaitDeviceId;
  unsigned char *image_data;
  size_t m_bufSize;
  size_t imageWidth;
  size_t imageHeight;

 public:
  LwSciSyncModule syncModule;
  LwSciBufModule buffModule;
  LwSciSyncAttrList synlwnreconciledList[2];
  LwSciSyncAttrList syncReconciledList;
  LwSciSyncAttrList syncConflictList;

  LwSciBufAttrList rawBufUnreconciledList[2];
  LwSciBufAttrList imageBufUnreconciledList[2];
  LwSciBufAttrList rawBufReconciledList;
  LwSciBufAttrList buffConflictList;
  LwSciBufAttrList imageBufReconciledList;
  LwSciBufAttrList imageBufConflictList;
  LwSciBufAttrList buffAttrListOut;

  LwSciSyncObj syncObj;
  LwSciBufObj rawBufObj;
  LwSciBufObj imageBufObj;
  LwSciSyncFence *fence;

  lwdaLwSci(int isMultiGPU, std::vector<int> &deviceIds,
            unsigned char *image_data, size_t imageWidth, size_t imageHeight);

  void initLwSci();

  void runLwdaLwSci(std::string &image_filename);

  void createLwSciRawBufObj();

  void createLwSciSyncObj();

  void createLwSciBufImageObj();
};

#endif  // LWDALWSCI_H
