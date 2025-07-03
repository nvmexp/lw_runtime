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



#ifndef __LWMEDIA_PRODUCER_H__

#define __LWMEDIA_PRODUCER_H__

#include "lwmedia_utils/cmdline.h"

#include "lwmedia_image.h"

#include "lwmedia_2d.h"

#include "lwmedia_surface.h"

#include "lwmedia_utils/image_utils.h"

#include "lwmedia_image_lwscibuf.h"

#include "lwscisync.h"



void runLwMediaBlit2D(TestArgs* args, Blit2DTest* ctx, LwSciSyncObj& syncObj,

                      LwSciSyncFence* preSyncFence, LwSciSyncFence* fence);

void runLwMediaBlit2D(TestArgs* args, Blit2DTest* ctx);

void setupLwMedia(TestArgs* args, Blit2DTest* ctx, LwSciBufObj& srcLwSciBufobj,

                  LwSciBufObj& dstLwSciBufobj, LwSciSyncObj& syncObj,

                  LwSciSyncObj& preSyncObj, int lwdaDeviceId);

void setupLwMedia(TestArgs* args, Blit2DTest* ctx);

void cleanupLwMedia(Blit2DTest* ctx, LwSciSyncObj& syncObj,

                    LwSciSyncObj& preSyncObj);

void cleanupLwMedia(Blit2DTest* ctx);

#endif

