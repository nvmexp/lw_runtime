/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef _EGLTEST_H
#define _EGLTEST_H

#include "testclient.h"
#include "testclientStream1.h"
#include "testclientStream2.h"
#include "socket.h"
#include <semaphore.h>

#define MAX_EGLSTREAM_TESTS 5

#define MAX_EGLSTREAM_FIFO_LEN 10

#define MAX_EGL_STREAM_ATTR 16

/*
 * Extension pointers
 */
#if defined(EGL_KHR_stream) && defined(EGL_KHR_stream_consumer_gltexture)
#define EXTENSION_LIST(T) \
    T( PFNEGLQUERYDEVICESEXTPROC,          eglQueryDevicesEXT,          EGL_EXT_device_base, 1 ) \
    T( PFNEGLQUERYDEVICESTRINGEXTPROC,     eglQueryDeviceStringEXT,     EGL_EXT_device_base, 1 ) \
    T( PFNEGLGETPLATFORMDISPLAYEXTPROC,    eglGetPlatformDisplayEXT,    EGL_EXT_platform_base, 1 ) \
    T( PFNEGLGETOUTPUTLAYERSEXTPROC,       eglGetOutputLayersEXT,       EGL_EXT_output_base, 0 ) \
    T( PFNEGLSTREAMCONSUMEROUTPUTEXTPROC,  eglStreamConsumerOutputEXT,  EGL_EXT_stream_consumer_egloutput, 0 ) \
    T( PFNEGLCREATESTREAMKHRPROC,          eglCreateStreamKHR,          EGL_KHR_stream, 1 ) \
    T( PFNEGLDESTROYSTREAMKHRPROC,         eglDestroyStreamKHR,         EGL_KHR_stream, 1 ) \
    T( PFNEGLQUERYSTREAMKHRPROC,           eglQueryStreamKHR,           EGL_KHR_stream, 1 ) \
    T( PFNEGLQUERYSTREAMU64KHRPROC,        eglQueryStreamu64KHR,        EGL_KHR_stream, 1 ) \
    T( PFNEGLQUERYSTREAMTIMEKHRPROC,       eglQueryStreamTimeKHR,       EGL_KHR_stream_fifo, 1 ) \
    T( PFNEGLSTREAMATTRIBKHRPROC,          eglStreamAttribKHR,          EGL_KHR_stream, 1 ) \
    T( PFNEGLSTREAMCONSUMERACQUIREKHRPROC, eglStreamConsumerAcquireKHR, EGL_KHR_stream_consumer_gltexture, 0 ) \
    T( PFNEGLSTREAMCONSUMERRELEASEKHRPROC, eglStreamConsumerReleaseKHR, EGL_KHR_stream_consumer_gltexture, 0 ) \
    T( PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC, eglStreamConsumerGLTextureExternalKHR, EGL_KHR_stream_consumer_gltexture, 0 ) \
    T( PFNEGLGETSTREAMFILEDESCRIPTORKHRPROC,         eglGetStreamFileDescriptorKHR,         EGL_KHR_stream_cross_process_fd, 1 ) \
    T( PFNEGLCREATESTREAMFROMFILEDESCRIPTORKHRPROC,  eglCreateStreamFromFileDescriptorKHR,  EGL_KHR_stream_cross_process_fd, 1 ) \
    T( PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC,     eglCreateStreamProducerSurfaceKHR,     EGL_KHR_stream_producer_eglsurface, 0 ) \
    T( PFNEGLSTREAMCONSUMERACQUIREATTRIBEXTPROC,     eglStreamConsumerAcquireAttribEXT,     EGL_KHR_stream_attrib, 0 ) \

#define EXTLST_DECL(tx, x, y, z)  tx x = NULL;
#define EXTLST_EXTERN(tx, x, y, z) extern tx x;
#define EXTLST_ENTRY(tx, x, y, z) { (extlst_fnptr_t *)&x, #x, #y, z },
#endif

typedef enum {
    EGLTEST_DATA_FORMAT_TYPE_YUV420 = 0,
    EGLTEST_DATA_FORMAT_TYPE_YUV422,
    EGLTEST_DATA_FORMAT_TYPE_YUV444,
    EGLTEST_DATA_FORMAT_TYPE_YUV422_10,
    EGLTEST_DATA_FORMAT_TYPE_RGB888,
    EGLTEST_DATA_FORMAT_TYPE_RAW8,
    EGLTEST_DATA_FORMAT_TYPE_RAW10,
    EGLTEST_DATA_FORMAT_TYPE_RAW12,
    EGLTEST_DATA_FORMAT_TYPE_RAW14,
    EGLTEST_DATA_FORMAT_TYPE_RAW16,
    EGLTEST_DATA_FORMAT_TYPE_NUM,
    EGLTEST_DATA_FORMAT_TYPE_Force32 = 0x7FFFFFFF
} EglTestDataFormatType;

typedef struct {
    struct LwRmSyncRec *sync;
    LwS8                eglIndex;
} EglTestStreamBuffer;

typedef struct {
    NativeDisplayType   nativeDisplay;
    NativeWindowType    nativeWindow;
    EGLDisplay          display;
    EGLSurface          surface;
    EGLConfig           config;
    EGLContext          context;
    EGLStreamKHR        eglStream;
    EGLint              eglStreamState;
    EGLint              width;
    EGLint              height;
    EGLint              latency;
    EGLint              acquireTimeout;
    EglTestStreamBuffer *streamBuffers;
    EGLint              streamBufferCount;
    EGLint              streamBufferLwrr;
    sem_t               streamBufferReady;
    EGLint              metadataCount;
    IServer            *iServer;
    IClient            *iClient;
} EglTestState;

extern EglTestState gEglState;

/*
 * Cross process modes
 */
typedef enum {
    SINGLE_PROCESS      = 0, // Single process.
    CROSS_PROCESS       = 1, // Cross-process.
    CROSS_PARTITION     = 2, // Cross-partition.
} EglTestProcessMode;

typedef enum {
    PRODUCER_CONSUMER   = 0,
    PRODUCER            = 1,
    CONSUMER            = 2,
} EglTestProcType;

typedef struct {
    char               *testName;
    LwU32               testNo;
    EglTestProcType     procType;
    LwU32               fifoLength;
    EglTestProcessMode  processMode;
    LwU32               maxFrames;
    LwU32               vmId;
    char                ipAddr[20];
} EglTestArgs;

extern EglTestArgs gTestArgs;

typedef struct {
    LwF64   data1;
    LwU64   data2;
} EglTestMetadata;

/*
 * MetaData handling methods
 */

/*
* Algorithm to generate metadata block data
* Producer uses this to set MetaData contents
* Consumer uses this to verify MetaData content
*/
LW_INLINE void getMetaDataFor(EglTestMetadata *metadatablock, LwU32 frameIdx, LwU32 valueOffset)
{
    assert(metadatablock);
    LwU32 temp = (frameIdx + 1) * 100 + (valueOffset + 1);
    metadatablock->data1 = 1.1 * static_cast<LwF64>(temp);
    metadatablock->data2 = static_cast<LwU64>(temp);
}

LW_INLINE bool isMetadataBlockEqual(EglTestMetadata *metadataBlock1, EglTestMetadata *metadataBlock2)
{
    return (metadataBlock1->data1 == metadataBlock2->data1 &&
            metadataBlock1->data2 == metadataBlock2->data2);
}

extern bool gSignalStop;

/*
 * This function is copied from khronos/egl/egl/egllocal.h
 */
static LW_INLINE EGLObjectKHR LwEglComputeHandle (void* obj)
{
    const uintptr_t EGL_HANDLE_MASK = 0xe31;
    assert(obj);
    /* We want to map the pointer and the type into some unique handle that
     * doesn't directly map to physical addresses. A simple XOR with a constant
     * and private value ensures that the handle retains the uniqueness of the
     * original pointer, but is not reversable by the client. */
    return (EGLObjectKHR) (((uintptr_t) obj) ^ EGL_HANDLE_MASK);
}

int  EGLSetupExtensions(EGLDisplay display);
GLboolean CheckExtension(const char *extString, const char *extName);
EGLBoolean LwEglTestInit();
EGLBoolean LwEglTestTerm();
EGLBoolean LwEglTestCreateContextSurface();
EGLBoolean LwEglTestClearContextSurface();

EGLBoolean LwEglTestExports();
EGLBoolean LwEglTestEglStream();

LwError LwEglTestProducerReturnFrame(LwEglApiStreamInfo         info,
                                     LwEglApiStreamFrame       *frame);
LwError LwEglTestConsumerUpdateFrameAttr(LwEglApiContext       *context,
                                         LwEglApiStreamInfo     info,
                                         LwEglApiStreamFrame   *acquireFrame,
                                         LwEglApiStreamFrame   *releaseFrame,
                                         const LwEglApiStreamUpdateAttrs *attrs);
LwError LwEglTestConsumerRegisterBuffer(LwEglApiContext        *context,
                                        LwEglApiStreamInfo      info,
                                        LwEglApiStreamFrame    *newBuffer);
LwError LwEglTestConsumerUnregisterBuffer(LwEglApiContext      *context,
                                          LwEglApiStreamInfo    info,
                                          LwEglApiStreamFrame  *oldBuffer);

#endif //_EGLTEST_H
