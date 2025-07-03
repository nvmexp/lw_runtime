/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnTest_WindowFramebuffer_h__
#define __lwnTest_WindowFramebuffer_h__

#include "lwnUtil/lwnUtil_Interface.h"

#include "lwn/lwn.h"
#include "lwnUtil/lwnUtil_PoolAllocator.h"

namespace lwnTest {

//
//                  LWN WINDOW FRAMEBUFFER CLASS
//
// Utility class provided to manage window-sized color and depth textures that
// will be set up as the default render targets for LWN tests in
// initGraphics() and presented on-screen in exitGraphics().
//
// Note:  We don't use the Framebuffer class here and explicitly use texture
// classes from "lwn::objects" instead of our wrapped objects in "lwn" because
// we don't want the window framebuffer's textures deleted at the end of every
// test.
//
class WindowFramebuffer
{
public:
    static const int NUM_BUFFERS = 2;

private:
    int                             m_width, m_height;
    lwnUtil::MemoryPoolAllocator    *m_allocator;
    LWNtexture                      *m_colorTextures[NUM_BUFFERS];
    LWNtexture                      *zsTexture;
    int                             m_acquiredTextureIndex;
    int                             m_previouslyAcquiredTextureIndex;
    LWNwindow                       *m_window;
    LWNsync                         *m_textureAvailableSync;
    LWNnativeWindow                 m_nativeWindow;

public:
    WindowFramebuffer() :
        m_width(0), m_height(0), m_allocator(NULL), zsTexture(NULL),
        m_acquiredTextureIndex(-1), m_previouslyAcquiredTextureIndex(-1),
        m_window(NULL), m_nativeWindow(NULL)
    {
        for (int i = 0; i < NUM_BUFFERS; i++)
            m_colorTextures[i] = NULL;
    }
    ~WindowFramebuffer()
    {
        assert(!m_allocator); // make sure destroy was called
    }

    // Delete the texture and view objects for the framebuffer.
    void destroy();

    // Set the size of the framebuffer to <width> x <height>, reallocating the
    // color and Z/stencil textures as necessary.
    void setSize(int width, int height);

    void setPresentInterval(int presentInterval);

    int getPresentInterval();

    int getNumBuffers() const { return NUM_BUFFERS; }

    LWNnativeWindow getNativeWindow() const { return m_nativeWindow; }

    LWNwindow *getWindow() const { return m_window; }

    // Bind the window framebuffer for rendering.
    void bind();

    // Present the window framebuffer on-screen.
    void present();

    // Present the window framebuffer on-screen, using the given queue.
    void present(LWNqueue *queue);

    // Query the acquired texture, returning a pointer in the native C
    // interface, or in the C+ interface (via reinterpret_cast).
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_C
    LWNtexture *getAcquiredTexture() const
    {
        assert(m_acquiredTextureIndex != -1);
        return m_colorTextures[m_acquiredTextureIndex];
    }
    LWNtexture *getDepthStencilTexture() const
    {
        return zsTexture;
    }
#else
    lwn::Texture *getAcquiredTexture() const
    {
        assert(m_acquiredTextureIndex != -1);
        LWNtexture *ctexture = m_colorTextures[m_acquiredTextureIndex];
        return reinterpret_cast<lwn::Texture *>(ctexture);
    }
    lwn::Texture *getDepthStencilTexture() const
    {
        return reinterpret_cast<lwn::Texture *>(zsTexture);
    }
#endif

    void setNativeWindow(LWNnativeWindow native) { m_nativeWindow = native; }

    // Set the viewport and scissor to the full framebuffer size.
    void setViewportScissor() const;

    // Read the contents of the screen to buffer.
    void readPixels(LWNbuffer *buffer) const;

    // Write the contents of a buffer to the screen.
    void writePixels(LWNbuffer *buffer) const;

    // Write the contents of a CPU pointer to the screen.
    void writePixels(const void *data) const;

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    //
    // Methods to provide a native C++ interface to the core WindowFramebuffer
    // class, using reinterpret_cast to colwert between C and C++ object
    // types.
    //
    void readPixels(lwn::Buffer *buffer) const
    {
        return readPixels(reinterpret_cast<LWNbuffer *>(buffer));
    }
    void writePixels(lwn::Buffer *buffer) const
    {
        return writePixels(reinterpret_cast<LWNbuffer *>(buffer));
    }
#endif

};

} // namespace lwnTest

#endif // #ifndef __lwnTest_WindowFramebuffer_h__
