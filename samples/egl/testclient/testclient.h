/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef EGL_TESTCLIENT_H
#define EGL_TESTCLIENT_H

#if defined(LW_EGL_DESKTOP_COMPATIBLE_HEADERS)
// For mobile base types and lwos/lwutil functionality on desktop builds
#include "mobile_common.h"
#endif

#include "defs.h"
#include "eglapiinterface.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The callback function that EGL calls into when the test client API
 * is registered.
 */
LwError TestClientEglInit(LwEglApiClientFuncs *imports,
                          const LwEglApiUtilityFuncs *utilities);

/*
 * The test client library init function.
 */
LwError TestClientInit(void);

/* TODO: add test functions for other export interfaces.
#if LW_EGL_CFG_ANDROID_blob_cache
bool TestExportSetBlobFunc(LwEglDisplay* pDisplay,
                           const void *key,
                           LwU32 keySize,
                           const void *value,
                           LwU32 valueSize);

bool TestExpGetBlobFunc(LwEglDisplay* pDisplay,
                        const void *key,
                        LwU32 keySize,
                        void *value,
                        LwU32 valueSize);
#endif

bool TestExpLockEgl(void);

bool TestExpUnlockEgl(void);

bool TestExpGetFenceFromEglSyncObject(void *eglSyncKHR,
                                      LwRmSync **fenceOut);

bool TestExpGlsiImageRef(LwGlsiEglImageHandle image);

bool TestExpGlsiImageUnref(LwGlsiEglImageHandle image);

bool TestExpGlsiImageFromEglImage(LwGlsiEglImageHandle* image,
                                  LwEglImageHandle eglImage);

bool TestExpGlsiImageFromCommonImage(LwGlsiEglImageHandle* glsiImage,
                                     LwUPtr eglDpy,
                                     const LwCommonImage*  cmnImage);

bool TestExpGlsiImageToCommonImage(LwGlsiEglImageHandle glsiImage,
                                   const LwCommonClient* cmnClient,
                                   const LwCommonHandleMgr* cmnMgr,
                                   LwCommonImage* cmnImage);
*/
bool TestExpGlsiImageFromLwRmSurface(LwGlsiEglImageHandle *image,
                                     LwEglDisplayHandle display,
                                     const LwRmSurface *surf,
                                     LwU32 count);
/*
bool TestExpGlsiImageToLwRmSurface(LwGlsiEglImageHandle image,
                                   LwU32 maxCount,
                                   LwBool dupHMem,
                                   LwRmSurface* surf,
                                   LwU32* count);

bool TestExpDecompressEglImage(LwEglImageHandle image,
                               LwRmSync *fenceIn,
                               LwRmSync **fenceOut);

bool TestExpProcessToken(LwU32 tokenId, const void* readPtr, void * mt);

bool TestExpTokenServerThreadCleanup(void);

bool TestExpGetApiContextFromEGLContext(void *display,
                                        void *eglContext);

bool TestExpLeaveThreaded(void **, void *, void **);

bool TestExpResumeThreaded(void *, void *, void *);
*/

#ifdef __cplusplus
}
#endif

#endif
