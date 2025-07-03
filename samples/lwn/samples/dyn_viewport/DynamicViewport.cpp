/*
* Copyright (c) 2016 LWPU Corporation.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#include <stdlib.h>
#include <stdarg.h>

#include "lwn/lwn.h"
#include "lwn/lwn_FuncPtr.h"
#include "lwn/lwn_FuncPtrInline.h"

// Define some of the deprecated lwn types since lwnUtil_CommandMem requires them
typedef int LWNint;
typedef uintptr_t LWNuintptr;
typedef intptr_t LWNsizeiptr;

#define LWNUTIL_USE_CPP_INTERFACE 1

#include "lwnutil.h"
#include "lwnUtil/lwnUtil_AlignedStorageImpl.h"
#include "lwnUtil/lwnUtil_PoolAllocatorImpl.h"
#include "lwnUtil/lwnUtil_GlslcHelperImpl.h"
#include "lwnUtil/lwnUtil_QueueCmdBufImpl.h"
#include "lwnUtil/lwnUtil_CommandMemImpl.h"

#include "lwnTool/texpkg/lwnTool_DataTypes.h"

#include "DynamicViewport.h"

using namespace lwn;
using namespace lwnTool::texpkg;


DynamicViewport::DynamicViewport(LWNnativeWindow *win, Device *dev) :
    m_nativeWindow(win),
    m_device(dev),
    m_vertexBuffer(NULL),
    m_renderTargetDepth(NULL),
    m_vertexBufferSize(0),
    m_vertexBufferAddress(0),
    m_queueCB(NULL),
    m_tracker(NULL),
    m_glslcLibraryHelper(NULL),
    m_glslcHelper(NULL),
    m_bufferPoolAllocator(NULL),
    m_texturePoolAllocator(NULL)
{
    // Set initial viewport to full texture size
    m_viewport    = { 0, 0, m_windowtextureWidth, m_windowtextureHeight };
    memset(m_renderTarget, 0, m_numBuffers * sizeof(Texture*));
}

DynamicViewport::~DynamicViewport()
{
    if (m_queueCB) {
        m_queueCB->destroy();
    }

    m_queue.Finalize();
    m_program.Finalize();
    m_window.Finalize();
    m_textureAvailableSync.Finalize();

    delete m_texturePoolAllocator;
    delete m_bufferPoolAllocator;
    delete g_lwn.m_texIDPool;
    delete m_glslcHelper;
    delete m_glslcLibraryHelper;
    delete m_tracker;
    delete m_queueCB;
}

bool DynamicViewport::init()
{
    QueueBuilder qb;

    qb.SetDevice(m_device)
      .SetDefaults();

    m_queue.Initialize(&qb);

    m_tracker = new lwnUtil::CompletionTracker(reinterpret_cast<LWNdevice*>(m_device), 32);

    // Create command buffer
    m_queueCB = new lwnUtil::QueueCommandBufferBase;
    m_queueCB->init(reinterpret_cast<LWNdevice*>(m_device), reinterpret_cast<LWNqueue*>(&m_queue), m_tracker);

    // Create buffer pool allocator
    MemoryPoolFlags bufferPoolFlags = MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::SHADER_CODE | MemoryPoolFlags::COMPRESSIBLE;
    m_bufferPoolAllocator = new lwnUtil::MemoryPoolAllocator(m_device, NULL, 0, bufferPoolFlags);

    // Allocate descriptor pool
    g_lwn.m_texIDPool = new LWNsystemTexIDPool(reinterpret_cast<LWNdevice*>(m_device), dynamic_cast<LWNcommandBuffer*>(m_queueCB));

    // Create texture pool allocator
    const size_t texturePoolSize = 32 * 1024 * 1024;
    MemoryPoolFlags texturePoolFlags = MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::SHADER_CODE | MemoryPoolFlags::COMPRESSIBLE;
    m_texturePoolAllocator = new lwnUtil::MemoryPoolAllocator(m_device, NULL, texturePoolSize, texturePoolFlags);

    if (!initWindow()) {
        return false;
    }
    if (!initVertexData()) {
        return false;
    }
    if (!initProgram()) {
        return false;
    }

    m_queueCB->submit();
    m_queue.Finish();

    return true;
}

void DynamicViewport::resize(int x, int y, int w, int h, bool adjustViewport, bool adjustCropRect)
{
    assert((y + h) <= m_windowtextureHeight);
    assert((x + w) <= m_windowtextureWidth);

    if (adjustViewport) {
        m_viewport = { x, y, w, h };

        if (!adjustCropRect) {
            // Set crop rectangle to default in case it was changed previously
            m_window.SetCrop(0, 0, m_windowtextureWidth, m_windowtextureHeight);
        }
    }
    if (adjustCropRect) {
        m_window.SetCrop(x, y, w, h);

        if (!adjustViewport) {
            // Set the viewport to the default in case it was changed previously
            m_viewport = { 0, 0, m_windowtextureWidth, m_windowtextureHeight };
        }
    }
}

void DynamicViewport::getCrop(lwn::Rectangle& crop)
{
    m_window.GetCrop(&crop);
}

void DynamicViewport::display()
{
    lwnUtil::QueueCommandBuffer &cmdBuffer = *m_queueCB;

    int rtIdx = 0;

    m_window.AcquireTexture(&m_textureAvailableSync, &rtIdx);
    m_queue.WaitSync(&m_textureAvailableSync);

    cmdBuffer.SetRenderTargets(1, &m_renderTarget[rtIdx], NULL, m_renderTargetDepth, NULL);
    cmdBuffer.SetViewport(m_viewport.x, m_viewport.y, m_viewport.width, m_viewport.height);

    float clearColor[4] = { 0.2f, 0.2f, 0.2f, 1.0f };
    cmdBuffer.ClearColor(0, clearColor, ClearColorMask::RGBA);
    cmdBuffer.ClearDepthStencil(1.0f, LWN_TRUE, 0, 0);

    cmdBuffer.BindProgram(&m_program, ShaderStageBits::VERTEX | ShaderStageBits::FRAGMENT);
    cmdBuffer.BindVertexAttribState(2, m_vertexAttribState);
    cmdBuffer.BindVertexStreamState(2, m_vertexStreamState);
    cmdBuffer.BindVertexBuffer(0, m_vertexBufferAddress, m_vertexBufferSize);;

    cmdBuffer.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

    cmdBuffer.submit();

    m_queue.PresentTexture(&m_window, rtIdx);

    // Insert a fence every frame. If we run out of memory the m_tracker can wait
    // for previous fences to get signaled and the command and control memory that
    // was used so far can be freed and re-used.
    m_tracker->insertFence(&m_queue);
}

bool DynamicViewport::initWindow()
{
    TextureBuilder tb;

    tb.SetDefaults().SetDevice(m_device)
                    .SetTarget(TextureTarget::TARGET_2D)
                    .SetFormat(Format::RGBA8)
                    .SetFlags(TextureFlags::DISPLAY | TextureFlags::COMPRESSIBLE)
                    .SetSize2D(m_windowtextureWidth, m_windowtextureHeight);


    for (int i = 0; i < m_numBuffers; i++) {
        m_renderTarget[i] = reinterpret_cast<Texture*>(m_texturePoolAllocator->allocTexture(&tb));
    }

    tb.SetFormat(Format::DEPTH24_STENCIL8)
      .SetFlags(TextureFlags::COMPRESSIBLE);

    m_renderTargetDepth = reinterpret_cast<Texture*>(m_texturePoolAllocator->allocTexture(&tb));

    WindowBuilder wb;

    wb.SetDefaults().SetDevice(m_device)
      .SetNativeWindow(m_nativeWindow)
      .SetTextures(m_numBuffers, m_renderTarget);

    return (m_window.Initialize(&wb) == LWN_TRUE) && (m_textureAvailableSync.Initialize(m_device) == LWN_TRUE);
}

bool DynamicViewport::initProgram()
{
    m_glslcLibraryHelper = new lwnUtil::GLSLCLibraryHelper;
    m_glslcLibraryHelper->LoadDLL(NULL);
    m_glslcHelper = new lwnUtil::GLSLCHelper(reinterpret_cast<LWNdevice*>(m_device), 0x100000UL, m_glslcLibraryHelper);

    const char *vsstring =
        "#version 440 core\n"
        "layout(location = 0) in vec4 position;\n"
        "layout(location = 1) in vec4 color;\n"
        "out vec4 ocolor;\n"
        "void main() {\n"
        "  gl_Position = position;\n"
        "  ocolor = color;\n"
        "}\n";

    const char *fsstring =
        "#version 440 core\n"
        "in vec4 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = ocolor;\n"
        "}\n";

    if (!m_program.Initialize(m_device)) {
        return false;
    }

    ShaderStage stages[2] = { ShaderStage::VERTEX, ShaderStage::FRAGMENT };
    const char *shaderSrc[2] = { vsstring, fsstring };

    return (m_glslcHelper->CompileAndSetShaders(&m_program, stages, 2, shaderSrc) == LWN_TRUE);
}

bool DynamicViewport::initVertexData()
{
    struct Vertex {
        dt::vec3    position;
        dt::vec4    color;
    };

    const Vertex vertexData[] = {
        { dt::vec3(-0.75f, -0.75f, 0.0f), dt::vec4(1.0f, 0.0f, 0.0f, 1.0f) },
        { dt::vec3(+0.75f, -0.75f, 0.0f), dt::vec4(0.0f, 1.0f, 0.0f, 1.0f) },
        { dt::vec3(-0.75f, +0.75f, 0.0f), dt::vec4(0.0f, 0.0f, 1.0f, 1.0f) },
        { dt::vec3(+0.75f, +0.75f, 0.0f), dt::vec4(1.0f, 0.0f, 1.0f, 1.0f) },
    };

    BufferBuilder bb;

    bb.SetDevice(m_device).SetDefaults();
    m_vertexBuffer = m_bufferPoolAllocator->allocBuffer(&bb, lwnUtil::BufferAlignBits::BUFFER_ALIGN_VERTEX_BIT, sizeof(vertexData));

    void *p = m_vertexBuffer->Map();
    if (!p) {
        return false;
    }

    memcpy(p, vertexData, sizeof(vertexData));

    m_vertexBufferAddress = m_vertexBuffer->GetAddress();
    m_vertexBufferSize = sizeof(vertexData);

    m_vertexAttribState[0].SetDefaults()
                          .SetStreamIndex(0)
                          .SetFormat(Format::RGB32F, 0);

    m_vertexStreamState[0].SetDefaults()
                          .SetStride(sizeof(dt::vec3) + sizeof(dt::vec4));

    m_vertexAttribState[1].SetDefaults()
                          .SetStreamIndex(0)
                          .SetFormat(Format::RGBA32F, sizeof(dt::vec3));

    m_vertexStreamState[1].SetDefaults()
                          .SetStride(sizeof(dt::vec3) + sizeof(dt::vec4));

    return true;
}
