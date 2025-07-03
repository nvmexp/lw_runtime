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



#include <string.h>

#include <iostream>

/* Lwpu headers */

#include "lwmedia_utils/cmdline.h"

#include "lwmedia_image.h"

#include "lwmedia_2d.h"

#include "lwmedia_surface.h"

#include "lwmedia_utils/image_utils.h"

#include "lwmedia_image_lwscibuf.h"

#include "lwmedia_producer.h"

#include "lwmedia_2d_lwscisync.h"

#include "lwsci_setup.h"



LwMediaImage *LwMediaImageCreateUsingLwScibuf(LwMediaDevice *device,

                                              LwMediaSurfaceType type,

                                              const LwMediaSurfAllocAttr *attrs,

                                              uint32_t numAttrs, uint32_t flags,

                                              LwSciBufObj &bufobj,

                                              int lwdaDeviceId) {

  LwSciBufModule module = NULL;

  LwSciError err = LwSciError_Success;

  LwMediaStatus status = LWMEDIA_STATUS_OK;

  LwSciBufAttrList attrlist = NULL;

  LwSciBufAttrList conflictlist = NULL;

  LwSciBufAttrValAccessPerm access_perm = LwSciBufAccessPerm_ReadWrite;

  LwSciBufAttrKeyValuePair attr_kvp = {LwSciBufGeneralAttrKey_RequiredPerm,

                                       &access_perm, sizeof(access_perm)};

  LwSciBufAttrKeyValuePair pairArrayOut[10];



  LwMediaImage *image = NULL;



  err = LwSciBufModuleOpen(&module);

  if (err != LwSciError_Success) {

    printf("%s: LwSciBuffModuleOpen failed. Error: %d \n", __func__, err);

    goto fail_cleanup;

  }



  err = LwSciBufAttrListCreate(module, &attrlist);

  if (err != LwSciError_Success) {

    printf("%s: SciBufAttrListCreate failed. Error: %d \n", __func__, err);

    goto fail_cleanup;

  }



  err = LwSciBufAttrListSetAttrs(attrlist, &attr_kvp, 1);

  if (err != LwSciError_Success) {

    printf("%s: AccessPermSetAttr failed. Error: %d \n", __func__, err);

    goto fail_cleanup;

  }



  status =

      LwMediaImageFillLwSciBufAttrs(device, type, attrs, numAttrs, 0, attrlist);



  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: ImageFillSciBufAttrs failed. Error: %d \n", __func__, err);

    goto fail_cleanup;

  }



  setupLwSciBuf(bufobj, attrlist, lwdaDeviceId);



  status = LwMediaImageCreateFromLwSciBuf(device, bufobj, &image);



  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: ImageCreatefromSciBuf failed. Error: %d \n", __func__, err);

    goto fail_cleanup;

  }



  LwSciBufAttrListFree(attrlist);



  if (module != NULL) {

    LwSciBufModuleClose(module);

  }



  return image;



fail_cleanup:

  if (attrlist != NULL) {

    LwSciBufAttrListFree(attrlist);

  }

  if (bufobj != NULL) {

    LwSciBufObjFree(bufobj);

    bufobj = NULL;

  }



  if (module != NULL) {

    LwSciBufModuleClose(module);

  }

  LwMediaImageDestroy(image);

  return NULL;

}



/* Create LwMediaImage surface based on the input attributes.

 * Returns LWMEDIA_STATUS_OK on success

 */

static LwMediaStatus createSurface(Blit2DTest *ctx,

                                   LwMediaSurfFormatAttr *surfFormatAttrs,

                                   LwMediaSurfAllocAttr *surfAllocAttrs,

                                   uint32_t numSurfAllocAttrs,

                                   LwMediaImage **image, LwSciBufObj &bufObj,

                                   int lwdaDeviceId) {

  LwMediaSurfaceType surfType;



  /* create source image */

  surfType =

      LwMediaSurfaceFormatGetType(surfFormatAttrs, LWM_SURF_FMT_ATTR_MAX);

  *image = LwMediaImageCreateUsingLwScibuf(ctx->device, /* device */

                                           surfType,    /* surface type */

                                           surfAllocAttrs, numSurfAllocAttrs, 0,

                                           bufObj, lwdaDeviceId);



  if (*image == NULL) {

    printf("Unable to create image\n");

    return LWMEDIA_STATUS_ERROR;

  }

  InitImage(*image, surfAllocAttrs[0].value, surfAllocAttrs[1].value);



  /*    printf("%s: LwMediaImageCreate:: Image size: %ux%u Image type: %d\n",

              __func__, surfAllocAttrs[0].value, surfAllocAttrs[1].value,

     surfType);*/



  return LWMEDIA_STATUS_OK;

}



/* Create LwMediaImage surface based on the input attributes.

 * Returns LWMEDIA_STATUS_OK on success

 */

static LwMediaStatus createSurfaceNonLwSCI(

    Blit2DTest *ctx, LwMediaSurfFormatAttr *surfFormatAttrs,

    LwMediaSurfAllocAttr *surfAllocAttrs, uint32_t numSurfAllocAttrs,

    LwMediaImage **image) {

  LwMediaSurfaceType surfType;



  /* create source image */

  surfType =

      LwMediaSurfaceFormatGetType(surfFormatAttrs, LWM_SURF_FMT_ATTR_MAX);



  *image = LwMediaImageCreateNew(ctx->device, surfType, surfAllocAttrs,

                                 numSurfAllocAttrs, 0);



  if (*image == NULL) {

    printf("Unable to create image\n");

    return LWMEDIA_STATUS_ERROR;

  }

  InitImage(*image, surfAllocAttrs[0].value, surfAllocAttrs[1].value);



  /*    printf("%s: LwMediaImageCreate:: Image size: %ux%u Image type: %d\n",

              __func__, surfAllocAttrs[0].value, surfAllocAttrs[1].value,

     surfType);*/



  return LWMEDIA_STATUS_OK;

}



static void destroySurface(LwMediaImage *image) { LwMediaImageDestroy(image); }



static LwMediaStatus blit2DImage(Blit2DTest *ctx, TestArgs *args,

                                 LwSciSyncObj &lwMediaSignalerSyncObj,

                                 LwSciSyncFence *preSyncFence,

                                 LwSciSyncFence *fence) {

  LwMediaStatus status;

  LwMediaImageSurfaceMap surfaceMap;



  status = ReadImage(args->inputFileName,              /* fileName */

                     0,                                /* frameNum */

                     args->srcSurfAllocAttrs[0].value, /* source image width */

                     args->srcSurfAllocAttrs[1].value, /* source image height */

                     ctx->srcImage,                    /* srcImage */

                     LWMEDIA_TRUE,                     /* uvOrderFlag */

                     1,                                /* bytesPerPixel */

                     MSB_ALIGNED);                     /* pixelAlignment */



  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: ReadImage failed for input buffer: %d\n", __func__, status);

    return status;

  }



  if ((args->srcRect.x1 <= args->srcRect.x0) ||

      (args->srcRect.y1 <= args->srcRect.y0)) {

    ctx->srcRect = NULL;

  } else {

    ctx->srcRect = &(args->srcRect);

  }



  if ((args->dstRect.x1 <= args->dstRect.x0) ||

      (args->dstRect.y1 <= args->dstRect.y0)) {

    ctx->dstRect = NULL;

  } else {

    ctx->dstRect = &(args->dstRect);

  }



  static int64_t launch = 0;

  // Start inserting pre-fence from second launch inorder to for LwMedia2Blit to

  // wait

  // for lwca signal on fence.

  if (launch) {

    status = LwMedia2DInsertPreLwSciSyncFence(ctx->i2d, preSyncFence);

    if (status != LWMEDIA_STATUS_OK) {

      printf("%s: LwMedia2DSetLwSciSyncObjforEOF   failed: %d\n", __func__,

             status);

      return status;

    }

    LwSciSyncFenceClear(preSyncFence);

  }

  launch++;



  status = LwMedia2DSetLwSciSyncObjforEOF(ctx->i2d, lwMediaSignalerSyncObj);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: LwMedia2DSetLwSciSyncObjforEOF   failed: %d\n", __func__,

           status);

    return status;

  }



  /* 2DBlit processing on input image */

  status = LwMedia2DBlitEx(ctx->i2d,          /* i2d */

                           ctx->dstImage,     /* dstSurface */

                           ctx->dstRect,      /* dstRect */

                           ctx->srcImage,     /* srcSurface */

                           ctx->srcRect,      /* srcRect */

                           &args->blitParams, /* params */

                           NULL);             /* paramsOut */



  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: LwMedia2DBlitEx failed: %d\n", __func__, status);

    return status;

  }



  status =

      LwMedia2DGetEOFLwSciSyncFence(ctx->i2d, lwMediaSignalerSyncObj, fence);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: LwMedia2DGetEOFLwSciSyncFence failed: %d\n", __func__, status);

    return status;

  }



  return LWMEDIA_STATUS_OK;

}



static LwMediaStatus blit2DImageNonLwSCI(Blit2DTest *ctx, TestArgs *args) {

  LwMediaStatus status;

  LwMediaImageSurfaceMap surfaceMap;



  status = ReadImage(args->inputFileName,              /* fileName */

                     0,                                /* frameNum */

                     args->srcSurfAllocAttrs[0].value, /* source image width */

                     args->srcSurfAllocAttrs[1].value, /* source image height */

                     ctx->srcImage,                    /* srcImage */

                     LWMEDIA_TRUE,                     /* uvOrderFlag */

                     1,                                /* bytesPerPixel */

                     MSB_ALIGNED);                     /* pixelAlignment */



  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: ReadImage failed for input buffer: %d\n", __func__, status);

    return status;

  }



  if ((args->srcRect.x1 <= args->srcRect.x0) ||

      (args->srcRect.y1 <= args->srcRect.y0)) {

    ctx->srcRect = NULL;

  } else {

    ctx->srcRect = &(args->srcRect);

  }



  if ((args->dstRect.x1 <= args->dstRect.x0) ||

      (args->dstRect.y1 <= args->dstRect.y0)) {

    ctx->dstRect = NULL;

  } else {

    ctx->dstRect = &(args->dstRect);

  }



  /* 2DBlit processing on input image */

  status = LwMedia2DBlitEx(ctx->i2d,          /* i2d */

                           ctx->dstImage,     /* dstSurface */

                           ctx->dstRect,      /* dstRect */

                           ctx->srcImage,     /* srcSurface */

                           ctx->srcRect,      /* srcRect */

                           &args->blitParams, /* params */

                           NULL);             /* paramsOut */

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: LwMedia2DBlitEx failed: %d\n", __func__, status);

    return status;

  }



  /* Write output image into buffer */

  ctx->bytesPerPixel = 1;

  WriteImageToAllocatedBuffer(ctx, ctx->dstImage, LWMEDIA_TRUE, LWMEDIA_FALSE,

                              ctx->bytesPerPixel);



  return LWMEDIA_STATUS_OK;

}



static void cleanup(Blit2DTest *ctx, LwMediaStatus status = LWMEDIA_STATUS_OK) {

  if (ctx->srcImage != NULL) {

    LwMedia2DImageUnRegister(ctx->i2d, ctx->srcImage);

    destroySurface(ctx->srcImage);

  }

  if (ctx->dstImage != NULL) {

    LwMedia2DImageUnRegister(ctx->i2d, ctx->dstImage);

    destroySurface(ctx->dstImage);

  }

  if (status != LWMEDIA_STATUS_OK) {

    exit(EXIT_FAILURE);

  }

}



void cleanupLwMedia(Blit2DTest *ctx, LwSciSyncObj &syncObj,

                    LwSciSyncObj &preSyncObj) {

  LwMediaStatus status;

  cleanup(ctx);

  status = LwMedia2DUnregisterLwSciSyncObj(ctx->i2d, syncObj);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: LwMediaImageSciBufInit failed\n", __func__);

    exit(EXIT_FAILURE);

  }

  status = LwMedia2DUnregisterLwSciSyncObj(ctx->i2d, preSyncObj);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: LwMediaImageSciBufInit failed\n", __func__);

    exit(EXIT_FAILURE);

  }

  LwMediaImageLwSciBufDeinit();

}



void cleanupLwMedia(Blit2DTest *ctx) {

  cleanup(ctx);

  free(ctx->dstBuffPitches);

  free(ctx->dstBuffer);

  free(ctx->dstBuff);

}



void setupLwMedia(TestArgs *args, Blit2DTest *ctx, LwSciBufObj &srcLwSciBufobj,

                  LwSciBufObj &dstLwSciBufobj, LwSciSyncObj &syncObj,

                  LwSciSyncObj &preSyncObj, int lwdaDeviceId) {

  LwMediaStatus status;

  status = LwMediaImageLwSciBufInit();

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: LwMediaImageSciBufInit failed\n", __func__);

    cleanup(ctx, status);

  }



  // Create source surface

  status = createSurface(ctx, args->srcSurfFormatAttrs, args->srcSurfAllocAttrs,

                         args->numSurfAllocAttrs, &ctx->srcImage,

                         srcLwSciBufobj, lwdaDeviceId);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: Unable to create buffer pools\n", __func__);

    cleanup(ctx, status);

  }



  // Create destination surface

  status = createSurface(ctx, args->dstSurfFormatAttrs, args->dstSurfAllocAttrs,

                         args->numSurfAllocAttrs, &ctx->dstImage,

                         dstLwSciBufobj, lwdaDeviceId);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: Unable to create buffer pools\n", __func__);

    cleanup(ctx, status);

  }



  // Register source  Surface

  status =

      LwMedia2DImageRegister(ctx->i2d, ctx->srcImage, LWMEDIA_ACCESS_MODE_READ);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: Unable to register source surface\n", __func__);

    cleanup(ctx, status);

  }

  // Register destination Surface

  status = LwMedia2DImageRegister(ctx->i2d, ctx->dstImage,

                                  LWMEDIA_ACCESS_MODE_READ_WRITE);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: Unable to register destination surface\n", __func__);

    cleanup(ctx, status);

  }



  status = LwMedia2DRegisterLwSciSyncObj(ctx->i2d, LWMEDIA_EOFSYNCOBJ, syncObj);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: Unable to LwMedia2DRegisterLwSciSyncObj\n", __func__);

  }



  status =

      LwMedia2DRegisterLwSciSyncObj(ctx->i2d, LWMEDIA_PRESYNCOBJ, preSyncObj);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: Unable to LwMedia2DRegisterLwSciSyncObj\n", __func__);

  }

}



// Create LwMedia src & dst image without LwSciBuf

void setupLwMedia(TestArgs *args, Blit2DTest *ctx) {

  LwMediaStatus status;



  // Create source surface

  status = createSurfaceNonLwSCI(ctx, args->srcSurfFormatAttrs,

                                 args->srcSurfAllocAttrs,

                                 args->numSurfAllocAttrs, &ctx->srcImage);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: Unable to create buffer pools\n", __func__);

    cleanup(ctx, status);

  }



  // Create destination surface

  status = createSurfaceNonLwSCI(ctx, args->dstSurfFormatAttrs,

                                 args->dstSurfAllocAttrs,

                                 args->numSurfAllocAttrs, &ctx->dstImage);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: Unable to create buffer pools\n", __func__);

    cleanup(ctx, status);

  }



  // Register source  Surface

  status =

      LwMedia2DImageRegister(ctx->i2d, ctx->srcImage, LWMEDIA_ACCESS_MODE_READ);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: Unable to register source surface\n", __func__);

    cleanup(ctx, status);

  }



  // Register destination Surface

  status = LwMedia2DImageRegister(ctx->i2d, ctx->dstImage,

                                  LWMEDIA_ACCESS_MODE_READ_WRITE);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: Unable to register destination surface\n", __func__);

    cleanup(ctx, status);

  }



  // Allocate buffer for writing image & set image parameters in Blit2DTest.

  ctx->bytesPerPixel = 1;

  AllocateBufferToWriteImage(ctx, ctx->dstImage, LWMEDIA_TRUE, /* uvOrderFlag */

                             LWMEDIA_FALSE);                   /* appendFlag */

}



void runLwMediaBlit2D(TestArgs *args, Blit2DTest *ctx) {

  // Blit2D function

  LwMediaStatus status = blit2DImageNonLwSCI(ctx, args);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: Blit2D failed\n", __func__);

    cleanup(ctx, status);

  }

}



void runLwMediaBlit2D(TestArgs *args, Blit2DTest *ctx,

                      LwSciSyncObj &lwMediaSignalerSyncObj,

                      LwSciSyncFence *preSyncFence, LwSciSyncFence *fence) {

  // Blit2D function

  LwMediaStatus status =

      blit2DImage(ctx, args, lwMediaSignalerSyncObj, preSyncFence, fence);

  if (status != LWMEDIA_STATUS_OK) {

    printf("%s: Blit2D failed\n", __func__);

    cleanup(ctx, status);

  }

}

