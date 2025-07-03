/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/

#include "lwntest_c.h"
#include "lwnTest/lwnTest_WindowFramebuffer.h"

#include "lwn_utils.h"
#include "lwnTest/lwnTest_Objects.h"

namespace lwnTest {

void WindowFramebuffer::destroy()
{
    if (m_width) {
        lwnWindowFree(m_window);
        lwnSyncFree(m_textureAvailableSync);
    }

    for (int i = 0; i < NUM_BUFFERS; i++) {
        if (m_colorTextures[i]) {
            m_allocator->freeTexture(m_colorTextures[i]);
            m_colorTextures[i] = NULL;
        }
    }
    if (zsTexture) {
        m_allocator->freeTexture(zsTexture);
        zsTexture = NULL;
    }
    m_width = 0;
    m_height = 0;
    delete m_allocator;
    m_allocator = NULL;
    m_acquiredTextureIndex = -1;
}

void WindowFramebuffer::setSize(int width, int height)
{
    if (width == m_width && height == m_height) {
        return;
    }

    // Disable tracking of created/deleted objects because we want allocations
    // to persist beyond a call cleaning up all API resources in exitGraphics().
    PushLWNObjectTracking();
    DisableLWNObjectTracking();

    // On a resize, destroy any existing framebuffer attachments.
    destroy();

    LWNtextureBuilder tb;
    lwnTextureBuilderSetDefaults(&tb);
    lwnTextureBuilderSetDevice(&tb, g_lwnDevice);
    lwnTextureBuilderSetFlags(&tb, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT | LWN_TEXTURE_FLAGS_DISPLAY_BIT);
    lwnTextureBuilderSetTarget(&tb, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetSize2D(&tb, width, height);
    lwnTextureBuilderSetFormat(&tb, LWN_FORMAT_RGBA8);
    LWNsizeiptr colorTextureSize = lwnTextureBuilderGetStorageSize(&tb);
    LWNuintptr colorTextureAlignment = lwnTextureBuilderGetStorageAlignment(&tb);

    LWNtextureBuilder tbz;
    lwnTextureBuilderSetDefaults(&tbz);
    lwnTextureBuilderSetDevice(&tbz, g_lwnDevice);
    lwnTextureBuilderSetFlags(&tbz, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT);
    lwnTextureBuilderSetTarget(&tbz, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetSize2D(&tbz, width, height);
    lwnTextureBuilderSetFormat(&tbz, LWN_FORMAT_DEPTH24_STENCIL8);
    LWNsizeiptr depthTextureSize = lwnTextureBuilderGetStorageSize(&tbz);
    LWNuintptr depthTextureAlignment = lwnTextureBuilderGetStorageAlignment(&tbz);

    size_t totalAllocationSize =
        (NUM_BUFFERS * (colorTextureSize + colorTextureAlignment) +
         depthTextureSize + depthTextureAlignment);

    m_allocator = new lwnUtil::MemoryPoolAllocator(g_lwnDevice, NULL, totalAllocationSize,
                                                   LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    for (int i = 0; i < NUM_BUFFERS; i++)
        m_colorTextures[i] = m_allocator->allocTexture(&tb);

    zsTexture = m_allocator->allocTexture(&tbz);

    LWNwindowBuilder wb;
    lwnWindowBuilderSetDefaults(&wb);
    lwnWindowBuilderSetDevice(&wb, g_lwnDevice);
    lwnWindowBuilderSetNativeWindow(&wb, m_nativeWindow);
    lwnWindowBuilderSetTextures(&wb, NUM_BUFFERS, m_colorTextures);
    m_window = lwnWindowBuilderCreateWindow(&wb);

    m_textureAvailableSync = lwnDeviceCreateSync(g_lwnDevice);

    m_width = width;
    m_height = height;

    PopLWNObjectTracking();
}

void WindowFramebuffer::setPresentInterval(int presentInterval)
{
    assert(m_window);
    lwnWindowSetPresentInterval(m_window, presentInterval);
}

int WindowFramebuffer::getPresentInterval()
{
    assert(m_window);
    return lwnWindowGetPresentInterval(m_window);
}

void WindowFramebuffer::bind()
{
    if (m_acquiredTextureIndex == -1) {
        lwnWindowAcquireTexture(m_window, m_textureAvailableSync, &m_acquiredTextureIndex);
        lwnQueueWaitSync(g_lwnQueue, m_textureAvailableSync);
        m_previouslyAcquiredTextureIndex = m_acquiredTextureIndex;
    }

    LWNcommandBuffer *cmdBuf = g_lwnQueueCB;
    LWNtexture *acquired = getAcquiredTexture();
    lwnCommandBufferSetRenderTargets(cmdBuf, 1, &acquired, NULL, zsTexture, NULL);
}

void WindowFramebuffer::present()
{
    assert(m_acquiredTextureIndex != -1);
    lwnQueuePresentTexture(g_lwnQueue, m_window, m_acquiredTextureIndex);
    m_acquiredTextureIndex = -1;
}

void WindowFramebuffer::present(LWNqueue *queue)
{
    assert(m_acquiredTextureIndex != -1);
    lwnQueuePresentTexture(queue, m_window, m_acquiredTextureIndex);
    m_acquiredTextureIndex = -1;
}

void WindowFramebuffer::setViewportScissor() const
{
    LWNcommandBuffer *cmdBuf = g_lwnQueueCB;
    lwnCommandBufferSetViewport(cmdBuf, 0, 0, m_width, m_height);
    lwnCommandBufferSetScissor(cmdBuf, 0, 0, m_width, m_height);
}

void WindowFramebuffer::readPixels(LWNbuffer *buffer) const
{
    LWNcommandBuffer *cmdBuf = g_lwnQueueCB;
    LWNcopyRegion region = { 0, 0, 0, m_width, m_height, 1 };
    assert(m_previouslyAcquiredTextureIndex != -1);
    LWNtexture *colorTex = m_colorTextures[m_previouslyAcquiredTextureIndex];

    ptrdiff_t rowStride = lwnCommandBufferGetCopyRowStride(cmdBuf);
    ptrdiff_t imgStride = lwnCommandBufferGetCopyImageStride(cmdBuf);
    lwnCommandBufferSetCopyRowStride(cmdBuf, 0);
    lwnCommandBufferSetCopyImageStride(cmdBuf, 0);
    lwnCommandBufferCopyTextureToBuffer(cmdBuf, colorTex, NULL, &region,
                                        lwnBufferGetAddress(buffer),
                                        LWN_COPY_FLAGS_NONE);
    lwnCommandBufferSetCopyRowStride(cmdBuf, rowStride);
    lwnCommandBufferSetCopyImageStride(cmdBuf, imgStride);
    g_lwnQueueCB->submit();
    lwnQueueFinish(g_lwnQueue);
}

void WindowFramebuffer::writePixels(LWNbuffer *buffer) const
{
    LWNcommandBuffer *cmdBuf = g_lwnQueueCB;
    LWNcopyRegion region = { 0, 0, 0, m_width, m_height, 1 };
    assert(m_acquiredTextureIndex != -1);
    LWNtexture *colorTex = m_colorTextures[m_previouslyAcquiredTextureIndex];
    lwnCommandBufferCopyBufferToTexture(cmdBuf, lwnBufferGetAddress(buffer),
                                        colorTex, NULL, &region,
                                        LWN_COPY_FLAGS_NONE);
    g_lwnQueueCB->submit();
    lwnQueueFinish(g_lwnQueue);
}

void WindowFramebuffer::writePixels(const void *data) const
{
    int dataSize = m_width * m_height * 4;

    LWNbufferBuilder bb;
    lwnBufferBuilderSetDefaults(&bb);
    lwnBufferBuilderSetDevice(&bb, g_lwnDevice);

    lwnUtil::MemoryPoolAllocator tempAllocator(g_lwnDevice, NULL, dataSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    LWNbuffer *tempBuffer = tempAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, dataSize);
    void *ptr = lwnBufferMap(tempBuffer);

    memcpy(ptr, data, dataSize);
    writePixels(tempBuffer);

    tempAllocator.freeBuffer(tempBuffer);
}

} // namespace lwnTest
