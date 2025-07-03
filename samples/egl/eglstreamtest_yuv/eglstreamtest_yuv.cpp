/*
 * Copyright (c) 2019, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include "lwwinsys.h"
#include "GLES3/gl31.h"
#include "GLES2/gl2ext.h"
#include "eglstreamtest_yuv.h"
#include "yuv_frame.h"
#include "stdlib.h"
static eglResources eglRes;
static bufferResources bufRes;
static glResources glRes;
static LwWinSysDesktopHandle desktop;
static LwWinSysWindowHandle  window;

static const char goldCrc[] = "6191D1B5C60D8B8B7397987591AD78FC";

static void initQuad(long inWidth, long inHeight, GLboolean yIlwert)
{
    GLfloat        v[16];
    GLuint        vertexID;

    // The vertex array is arranged like so:
    // x = Position x
    // y = Position y
    // z = Texture Coordinate s
    // w = Texture Coordinate t

    v[0] = (float)-inWidth/2.0f;
    v[1] = (float)-inHeight/2.0f;
    v[2] = 0.0f;
    v[3] = yIlwert ? 0.0f : 1.0f;
    v[4] = v[0] + (float)inWidth;
    v[5] = v[1] + 0.0f;
    v[6] = 1.0f;
    v[7] = yIlwert ? 0.0f : 1.0f;
    v[8] = v[0] + 0.0f;
    v[9] = v[1] + (float)inHeight;
    v[10] = 0.0f;
    v[11] = yIlwert ? 1.0f : 0.0f;
    v[12] = v[0] + (float)inWidth;
    v[13] = v[1] + (float)inHeight;
    v[14] = 1.0f;
    v[15] = yIlwert ? 1.0f : 0.0f;


    // Upload to VBO

    glGenBuffers(1, &vertexID);
    glBindBuffer(GL_ARRAY_BUFFER, vertexID);
    glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CHECK_GL_ERROR("Failed to upload Vtx data\n");

    glRes.quadVboId = vertexID;
}

static void matrixIdentity(float m[16])
{
    memset(m, 0, sizeof(float) * 16);
    m[4 * 0 + 0] = m[4 * 1 + 1] = m[4 * 2 + 2] = m[4 * 3 + 3] = 1.0;
}
static void matrixMultiply(float m0[16], float m1[16])
{
    int r, c, i;
    for (r = 0; r < 4; r++) {
        float m[4] = {0.0, 0.0, 0.0, 0.0};
        for (c = 0; c < 4; c++) {
            for (i = 0; i < 4; i++) {
                m[c] += m0[4 * i + r] * m1[4 * c + i];
            }
        }
        for (c = 0; c < 4; c++) {
            m0[4 * c + r] = m[c];
        }
    }
}
static void matrixOrtho(float m[16], float l, float r, float b,
                        float t, float n, float f)
{
    float m1[16];
    float rightMinusLeftIlw, topMinusBottomIlw, farMinusNearIlw;

    rightMinusLeftIlw = 1.0f / (r - l);
    topMinusBottomIlw = 1.0f / (t - b);
    farMinusNearIlw = 1.0f / (f - n);

    m1[ 0] = 2.0f * rightMinusLeftIlw;
    m1[ 1] = 0.0f;
    m1[ 2] = 0.0f;
    m1[ 3] = 0.0f;

    m1[ 4] = 0.0f;
    m1[ 5] = 2.0f * topMinusBottomIlw;
    m1[ 6] = 0.0f;
    m1[ 7] = 0.0f;

    m1[ 8] = 0.0f;
    m1[ 9] = 0.0f;
    m1[10] = -2.0f * farMinusNearIlw;
    m1[11] = 0.0f;

    m1[12] = -(r + l) * rightMinusLeftIlw;
    m1[13] = -(t + b) * topMinusBottomIlw;
    m1[14] = -(f + n) * farMinusNearIlw;
    m1[15] = 1.0f;

    matrixMultiply(m, m1);
}

static bool initializeLwWinSys()
{
    LwRect windowRect;
    LwRect *pwindowRect = NULL;

    if (LwWinSysInterfaceSelect(LwWinSysInterface_Default)) {
        fprintf(stderr, "Failed to select WinSys interface\n");
        return false;
    }

    if (LwWinSysDesktopOpen(NULL, &desktop)) {
        fprintf(stderr, "Failed to select Desktop\n");
        return false;
    }

    windowRect.left   = 0;
    windowRect.top    = 0;
    windowRect.right  = YUV_IMAGE_WIDTH;
    windowRect.bottom = YUV_IMAGE_HEIGHT;
    pwindowRect = &windowRect;

    const char* exts = NULL;
    exts = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    if (exts && strstr(exts, "EGL_EXT_platform_base")) {
        switch (LwWinSysDesktopGetType(desktop)) {
            case LwWinSysInterface_X11:
                eglRes.platform = EGL_PLATFORM_X11_EXT;
                break;
            case LwWinSysInterface_Wayland:
                eglRes.platform = EGL_PLATFORM_WAYLAND_EXT;
                break;
            case LwWinSysInterface_Device:
            case LwWinSysInterface_DRM:
            case LwWinSysInterface_WF:
                eglRes.platform = EGL_PLATFORM_DEVICE_EXT;
                break;
            default:
                eglRes.platform = 0;
                break;
        }
    } else {
        eglRes.platform = 0;
    }

    eglRes.nativeDisplay = LwWinSysDesktopGetNativeHandle(desktop);

    LwWinSysWindowAttr attr;
    attr.mask = 0;
    if (pwindowRect) {
        attr.offsetX = pwindowRect->left;
        attr.offsetY = pwindowRect->top;
        attr.sizeX   = pwindowRect->right  - pwindowRect->left;
        attr.sizeY   = pwindowRect->bottom - pwindowRect->top;
        attr.mask |= LwWinSysWindowAttr_Bounds;
    }
    attr.abits = 8;
    attr.rbits = 8;
    attr.gbits = 8;
    attr.bbits = 8;
    attr.mask |= LwWinSysWindowAttr_Colors;
    window = LwWinSysWindowCreateAttr(desktop, "eglstreamtest_yuv", &attr);
    eglRes.nativeWindow  = LwWinSysWindowGetNativeHandle(window);

    return true;
}

static bool initializeEgl()
{
    initializeEglExtFunctions();
    static EGLint configAttrs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_SURFACE_TYPE,  EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };

    if (eglRes.platform == EGL_PLATFORM_DEVICE_EXT) {
        configAttrs[3] = EGL_STREAM_BIT_KHR;
    }

    static EGLint contextAttrs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };
    EGLint numConfig;

    eglRes.display = eglGetPlatformDisplay(eglRes.platform,
                                           eglRes.nativeDisplay , NULL);
    if (!eglRes.display || !eglInitialize(eglRes.display, 0, 0)) {
        CHECK_EGL_ERROR("Failed to initialize EGL Display");
        return false;
    }

    if (!eglChooseConfig(eglRes.display, configAttrs, &eglRes.config, 1, &numConfig) || !numConfig){
        CHECK_EGL_ERROR("Failed to choose EGLConfig");
        return false;
    }

    if (!(eglRes.context = eglCreateContext(eglRes.display, eglRes.config, EGL_NO_CONTEXT, contextAttrs))) {
        CHECK_EGL_ERROR("Failed to create EGLContext\n");
        return false;
    }



    switch (eglRes.platform) {
        case 0:
            eglRes.surface = eglCreateWindowSurface(eglRes.display,
                                                    eglRes.config,
                                                    eglRes.nativeWindow, 0);
            break;
        case EGL_PLATFORM_DEVICE_EXT:
            eglRes.surface = (EGLSurface)eglRes.nativeWindow;
            break;
        case EGL_PLATFORM_X11_EXT:
            eglRes.surface = eglCreateWindowSurface(eglRes.display,
                                                    eglRes.config,
                                                    eglRes.nativeWindow, 0);
            break;
        default:
            eglRes.surface = eglCreatePlatformWindowSurface(eglRes.display,
                                                            eglRes.config,
                                                            eglRes.nativeWindow,
                                                            0);
            break;
    }

    if (!eglRes.surface) {
        goto fail1;
    }

    if (!eglMakeLwrrent(eglRes.display, eglRes.surface, eglRes.surface, eglRes.context)) {
        goto fail2;
    }

    return true;

fail2:
    if (!eglDestroySurface(eglRes.display, eglRes.surface)) {
        CHECK_EGL_ERROR("Failed to destroy surface\n");
    }

fail1:
    if (!eglDestroyContext(eglRes.display, eglRes.context)) {
        CHECK_EGL_ERROR("Failed to destroy context\n");
    }

    return false;
}



static bool connectConsumer()
{
    glGenTextures(1, &glRes.texId);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, glRes.texId);
    if (CHECK_GL_ERROR("Failed to Create Texture object\n")) {
        goto fail1;
    }

    if (!eglStreamConsumerGLTextureExternalKHR(eglRes.display, eglRes.stream)) {
        CHECK_EGL_ERROR("Failed to Connect consumer to Stream\n");
        goto fail1;
    }

    return true;
fail1:
    if (glRes.texId) {
        glDeleteTextures(1, &glRes.texId);
    }
    return false;
}

static bool connectProducer()
{
    if (CHECK_LW_ERROR(eglRes.apiProcs.stream2.producer.reserve(eglRes.display, eglRes.stream, &eglRes.pvtStream),
                   "Failed to reserve stream Producer\n")) {
        return false;
    }

    eglRes.producerCaps.image.useGlsi = LW_TRUE;
    eglRes.producerCaps.sync.useLwRm = LW_TRUE;
    eglRes.producerCaps.sync.types = LwCommonSyncTypeBit_Reg;
    eglRes.producerCaps.image.origin = LW_EGL_STREAM_Y_ORIGIN_TOP;

    if (CHECK_LW_ERROR(eglRes.apiProcs.stream2.producer.connect(eglRes.pvtStream, &eglRes.producerCaps),
                       "Failed to connect producer\n")) {
        return false;
    }

    return true;
}

static bool setupStream()
{
    eglRes.stream = eglCreateStreamKHR(eglRes.display, NULL);
    if (!eglRes.stream) {
        CHECK_EGL_ERROR("Failed to create EGLStream\n");
        return false;
    }

    LwEglApiGetAccess(&eglRes.apiProcs);

    if (!connectConsumer()) {
        return false;
    }

    if (!connectProducer()) {
        return false;
    }

    return true;
}

static bool allocateBuffers()
{
    LwColorFormat lwColorFormat[MAX_SURFACE_PLANES];
    LwU32 surfAttrs[] = {
        LwRmSurfaceAttribute_Layout, LwRmSurfaceLayout_Blocklinear,
        LwRmSurfaceAttribute_None
    };

    if (LwRmOpenNew(&bufRes.hRm)) {
        fprintf(stderr, "Failed to open RM Device\n");
        return false;
    }

    lwColorFormat[0] = LwColorFormat_Y8;
    lwColorFormat[1] = LwColorFormat_U8;
    lwColorFormat[2] = LwColorFormat_V8;

    LwRmMultiplanarSurfaceSetup(bufRes.yuvSurf, MAX_SURFACE_PLANES, YUV_IMAGE_WIDTH,
                                YUV_IMAGE_HEIGHT, LwYuvColorFormat_YUV420, lwColorFormat, surfAttrs);

    LwRmSurfaceSetup(&bufRes.testSurf, YUV_IMAGE_WIDTH, YUV_IMAGE_HEIGHT, LwColorFormat_A8R8G8B8, surfAttrs);

    LWRM_DEFINE_MEM_HANDLE_ATTR(memAttr);

    LWRM_MEM_HANDLE_SET_ATTR(memAttr, LwRmSurfaceComputeAlignment(bufRes.hRm , &bufRes.yuvSurf[0]),
                             LwOsMemAttribute_WriteCombined,  LwRmSurfaceComputeSize(&bufRes.yuvSurf[0]), LwRmMemTags_Tests);

    for (int i = 0; i < MAX_SURFACE_PLANES; i++) {
        if (LwRmMemHandleAllocAttr(bufRes.hRm, &memAttr, &bufRes.yuvSurf[i].hMem)) {
            fprintf(stderr,  "Failed to allocate backing from Surf Idx: %i\n", i);
            return false;
        }
    }

    LWRM_MEM_HANDLE_SET_ATTR(memAttr, LwRmSurfaceComputeAlignment(bufRes.hRm , &bufRes.testSurf),
                             LwOsMemAttribute_WriteCombined,  LwRmSurfaceComputeSize(&bufRes.testSurf), LwRmMemTags_Tests);
    if (LwRmMemHandleAllocAttr(bufRes.hRm, &memAttr, &bufRes.testSurf.hMem)) {
        fprintf(stderr, "Failed to allocate backing for test surface\n");
        for (int i = 0; i < MAX_SURFACE_PLANES; i++) {
            LwRmMemHandleFree(bufRes.yuvSurf[i].hMem);
        }
        return false;
    }

    return true;
}

static void fillBuffer()
{
    int offset1 = YUV_IMAGE_WIDTH * YUV_IMAGE_HEIGHT;
    int offset2 = offset1 + ((YUV_IMAGE_WIDTH / 2) * (YUV_IMAGE_HEIGHT / 2));

    LwRmSurfaceWrite(&bufRes.yuvSurf[0], 0, 0, YUV_IMAGE_WIDTH, YUV_IMAGE_HEIGHT, yuv_buffer);
    LwRmSurfaceWrite(&bufRes.yuvSurf[1], 0, 0, YUV_IMAGE_WIDTH / 2, YUV_IMAGE_HEIGHT / 2, &yuv_buffer[offset1]);
    LwRmSurfaceWrite(&bufRes.yuvSurf[2], 0, 0, YUV_IMAGE_WIDTH / 2, YUV_IMAGE_HEIGHT / 2, &yuv_buffer[offset2]);
}

static bool setupBuffers()
{

    if (!allocateBuffers()) {
        fprintf(stderr, "Failed to allocate YUV Buffer\n");
        return false;
    }

    fillBuffer();

    return true;
}

static bool streamBuffer()
{
    if (CHECK_LW_ERROR(eglRes.apiProcs.glsi.imageFromLwRmSurface(&eglRes.glsiImage,
                                                                 eglRes.display, bufRes.yuvSurf,
                                                                 MAX_SURFACE_PLANES),
                       "Could not create GLSI Image\n")) {
        return false;
    }

    bufRes.streamBuffer.sync = NULL;

    eglRes.streamFrame.buffer = (LwEglApiClientBuffer)&bufRes.streamBuffer;
    LW_EGL_API_FRAME_SYNC_INIT(eglRes.streamFrame.lwrmSync.sync);

    bufRes.streamBuffer.eglIndex = 0;
    eglRes.clientBuffer = (LwEglApiClientBuffer)&bufRes.streamBuffer;

    if (CHECK_LW_ERROR(eglRes.apiProcs.stream2.producer.bufferRegister(eglRes.pvtStream,
                                                                       eglRes.glsiImage,
                                                                       eglRes.clientBuffer),
                       "Failed to register Buffer\n")) {
        goto fail1;
    }

    if (CHECK_LW_ERROR(eglRes.apiProcs.stream2.producer.framePresent(eglRes.pvtStream,
                                                                     &eglRes.streamFrame),
                       "Failed to present frame\n")) {
        goto fail2;
    }
    return true;

fail2:
    CHECK_LW_ERROR(eglRes.apiProcs.stream2.producer.bufferUnregister(eglRes.pvtStream, eglRes.clientBuffer),
                   "Failed to unregister Client Buffer\n");
fail1:
    CHECK_LW_ERROR(eglRes.apiProcs.glsi.imageUnref(eglRes.glsiImage), "Failed to unref GLSI image\n");

    return false;
}

static bool initShaders()
{
    GLuint vtxShader;
    GLuint fragShader;
    const char *vtxSrc = vtxShaderSource;
    const char *fragSrc = fragShaderSource;

    int vtxShaderSourceSize = sizeof(vtxShaderSource);
    int fragShaderSourceSize = sizeof(fragShaderSource);

    glRes.shaderId = glCreateProgram();

    vtxShader = glCreateShader(GL_VERTEX_SHADER);
    fragShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vtxShader, 1, (const char **)&vtxSrc, &vtxShaderSourceSize);
    glCompileShader(vtxShader);
    glShaderSource(fragShader, 1, (const char **)&fragSrc, &fragShaderSourceSize);
    glCompileShader(fragShader);

    glAttachShader(glRes.shaderId, vtxShader);
    glAttachShader(glRes.shaderId, fragShader);
    glLinkProgram(glRes.shaderId);

    glDeleteShader(vtxShader);
    glDeleteShader(fragShader);

    if (CHECK_GL_ERROR("Failed to setup Shaders\n")) {
        return false;
    }

    return true;
}

static bool initGl()
{
    float px = YUV_IMAGE_WIDTH / 2.0f;
    float py = YUV_IMAGE_HEIGHT / 2.0f;
    float pz = 0.0f;
    float rot = 0.0f;
    GLuint vtxLoc, texLoc, transRotLoc, mvpLoc;
    if (!initShaders()) {
        return false;
    }

    float mvp[16];
    initQuad(YUV_IMAGE_WIDTH, YUV_IMAGE_HEIGHT, GL_TRUE);
    matrixIdentity(mvp);
    matrixOrtho(mvp, 0.0f, YUV_IMAGE_WIDTH, YUV_IMAGE_HEIGHT, 0.0f, 0.0f, 1.0f);


    vtxLoc = glGetAttribLocation(glRes.shaderId, "Vertex");
    texLoc = glGetUniformLocation(glRes.shaderId, "Texture0");
    transRotLoc = glGetUniformLocation(glRes.shaderId, "TransRot");
    mvpLoc = glGetUniformLocation(glRes.shaderId, "ModelViewProjectionMatrix");

    // To avoid dependencies on mipmaps
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glEnable(GL_BLEND);
    glUseProgram(glRes.shaderId);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, glRes.texId);
    glUniform1i(texLoc, 0);
    glBindBuffer(GL_ARRAY_BUFFER, glRes.quadVboId);
    glVertexAttribPointer(vtxLoc, 4, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glEnableVertexAttribArray(vtxLoc);
    glUniform4f(transRotLoc, (float)px, (float)py, (float)pz, (float)rot);
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, (GLfloat*)&(mvp));

    return true;
}

static bool consumerAcquire()
{
    if (!initGl()) {
        fprintf(stderr, "Failed to initalize GL\n");
        return false;
    }

    if (!eglStreamConsumerAcquireKHR(eglRes.display, eglRes.stream)) {
        CHECK_EGL_ERROR("eglStreamConsumerAcquireKHR failed.\n");
        return false;
    }

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    GLubyte *data = (GLubyte *)malloc(4 * YUV_IMAGE_WIDTH * YUV_IMAGE_HEIGHT);
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for buffer data\n");
        return false;
    }
    glReadPixels(0, 0, YUV_IMAGE_WIDTH, YUV_IMAGE_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, data);
    LwRmSurfaceWrite(&bufRes.testSurf, 0, 0, YUV_IMAGE_WIDTH, YUV_IMAGE_HEIGHT, data);

    if(!eglSwapBuffers(eglRes.display, eglRes.surface)) {
        CHECK_EGL_ERROR("eglSwapBuffers failed\n");
        free(data);
        return false;
    }

    LwRmSurfaceComputeMD5(&bufRes.testSurf, 1, bufRes.testSurfCrc, sizeof(bufRes.testSurfCrc));
    free(data);

    return true;
}

static void terminateLwWinSys() {
    LwWinSysWindowDestroy(window);
    LwWinSysDesktopClose(desktop);
}

static void terminateGlResources()
{
    if (glRes.texId) {
        glDeleteTextures(1, &glRes.texId);
        CHECK_GL_ERROR("Failed to delete texture object\n");
    }
    if (glRes.shaderId) {
        glDeleteProgram(glRes.shaderId);
        CHECK_GL_ERROR("Failed to delete Program object\n");
    }
}

static void terminateEglResources()
{
    if (eglRes.stream) {
        eglDestroyStreamKHR(eglRes.display, eglRes.stream);
        CHECK_EGL_ERROR("Failed to destroy stream\n");
    }
    if (eglRes.surface) {
        eglDestroySurface(eglRes.display, eglRes.surface);
        CHECK_EGL_ERROR("Failed to destroy surface\n");
    }
    if (eglRes.context) {
        eglMakeLwrrent(eglRes.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        CHECK_EGL_ERROR("Failed to release context\n");
        eglDestroyContext(eglRes.display, eglRes.context);
        CHECK_EGL_ERROR("Failed to destroy context\n");
    }
}

static void terminateBufferResources()
{
    for (int i = 0; i < MAX_SURFACE_PLANES; i++) {
        if (bufRes.yuvSurf[i].hMem) {
            LwRmMemHandleFree(bufRes.yuvSurf[i].hMem);
        }
    }
    if (bufRes.testSurf.hMem) {
        LwRmMemHandleFree(bufRes.testSurf.hMem);
    }
}

static void terminateStreamResources()
{
    CHECK_LW_ERROR(eglRes.apiProcs.glsi.imageUnref(eglRes.glsiImage), "Failed to unref glsi image\n");
    CHECK_LW_ERROR(eglRes.apiProcs.stream2.producer.bufferUnregister(eglRes.pvtStream, eglRes.clientBuffer),
                       "Failed to unregister Client Buffer\n");

    CHECK_LW_ERROR(eglRes.apiProcs.stream2.producer.disconnect(eglRes.pvtStream, (LwError) 0),
                       "Failed to disconnect Producer from Stream\n");
}

static void verifyCrc()
{
    if (!strcmp(bufRes.testSurfCrc, goldCrc)) {
        fprintf(stdout, "PASS.\n Test CRC: %s\n Gold CRC: %s\n", bufRes.testSurfCrc, goldCrc);
    } else {
        fprintf(stdout, "FAIL! \n Test CRC: %s\n Gold CRC: %s\n", bufRes.testSurfCrc, goldCrc);
    }
}

static void terminateAllResources()
{
    terminateGlResources();
    terminateStreamResources();
    terminateEglResources();
    terminateBufferResources();
    terminateLwWinSys();
}

int main()
{
    if (!initializeLwWinSys()) {
        fprintf(stderr, "Failed to initialize LwWinSys\n");
        return -1;
    }

    if (!initializeEgl()) {
        fprintf(stderr, "Failed to initialize EGL \n");
        goto fail1;
    }

    if (!setupStream()) {
        fprintf(stderr, "Failed to setup EGLStream\n");
        goto fail2;
    }

    if (!setupBuffers()) {
        fprintf(stderr, "Failed to setup Buffers\n");
        goto fail2;
    }

    if (!streamBuffer()) {
        fprintf(stderr, "Failed to insert Buffer into stream\n");
        goto fail3;
    }

    if (!consumerAcquire()) {
        fprintf(stderr, "Consumer failed to acquire frame\n");
        goto fail4;
    }
    verifyCrc();
    terminateAllResources();
    return 0;

fail4:
    terminateGlResources();
    terminateStreamResources();
fail3:
    terminateBufferResources();
fail2:
    terminateEglResources();
fail1:
    terminateLwWinSys();
    return -1;
}
