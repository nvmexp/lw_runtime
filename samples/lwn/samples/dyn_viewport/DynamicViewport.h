/*
* Copyright (c) 2016 LWPU Corporation.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#ifndef _DYNAMIC_VIEWPORT_H
#define _DYNAMIC_VIEWPORT_H

#include <lwn/lwn_Cpp.h>
#include <lwn/lwn_CppMethods.h>

namespace lwnUtil {
    class QueueCommandBufferBase;
    class CompletionTracker;
    class GLSLCLibraryHelper;
    class GLSLCHelper;
    class MemoryPoolAllocator;
    class MemoryPoolAllocator;
}

class DynamicViewport
{
public:
    DynamicViewport(LWNnativeWindow *win, lwn::Device *dev);
    ~DynamicViewport();

    bool    init();

    // Sets the viewport and crop rectangle depending on adjustViewport and adjustCropRect.
    // If both are set the entire viewport is displayed.
    // If only adjustViewport is set the crop rectangle covers the entire window texture and
    // the entire window texture is displayed.
    // If only adjustCropRect is set. The viewport covers the entire window texture and
    // only a portion of this texture is displayed.
    void    resize(int x, int y, int w, int h, bool adjustViewport, bool adjustCropRect);

    // Returns the lwrrently used crop rectangle
    void    getCrop(lwn::Rectangle& crop);

    // Renders a quad into a m_windowtextureWidth x m_windowtextureHeight texture.
    void    display();

private:

    bool    initWindow();
    bool    initProgram();
    bool    initVertexData();

    static const int                    m_windowtextureWidth = 1920;
    static const int                    m_windowtextureHeight = 1080;
    static const int                    m_numBuffers = 3;

    LWNnativeWindow                     *m_nativeWindow;

    lwn::Device                         *m_device;
    lwn::Queue                          m_queue;
    lwn::Program                        m_program;
    lwn::Window                         m_window;
    lwn::Sync                           m_textureAvailableSync;
    lwn::Buffer                         *m_vertexBuffer;
    lwn::Texture                        *m_renderTarget[m_numBuffers];
    lwn::Texture                        *m_renderTargetDepth;

    lwn::Rectangle                      m_viewport;

    int                                 m_vertexBufferSize;
    lwn::BufferAddress                  m_vertexBufferAddress;

    lwn::VertexAttribState              m_vertexAttribState[2];
    lwn::VertexStreamState              m_vertexStreamState[2];

    lwnUtil::QueueCommandBufferBase     *m_queueCB;
    lwnUtil::CompletionTracker          *m_tracker;
    lwnUtil::GLSLCLibraryHelper         *m_glslcLibraryHelper;
    lwnUtil::GLSLCHelper                *m_glslcHelper;
    lwnUtil::MemoryPoolAllocator        *m_bufferPoolAllocator;
    lwnUtil::MemoryPoolAllocator        *m_texturePoolAllocator;
};
#endif
