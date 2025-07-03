/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Rynda's std::function is not compatible with __attribute__((noescape)).
// Fortunately it links just fine without it.
#define LWN_NOESCAPE

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include <functional>

using namespace lwn;

class LWNCommandBufferCallCopyTest
{
    // We split the window horizontally into two 320x480 render targets, with
    // each render target made up of 40x40 cells.
    static const int renderTargetWidth = 320;
    static const int renderTargetHeight = 480;
    static const int cellSize = 40;
    static const int cellsPerRow = 8;
    static const int cellRows = 12;
    ct_assert(renderTargetWidth == cellsPerRow * cellSize);
    ct_assert(renderTargetHeight == cellRows * cellSize);

    // The vertex buffer we use holds a single quad for each cell on a row,
    // plus a quad for a draw call to draw the two render targets into the
    // window framebuffer.
    static const int totalVerts = cellsPerRow * 4 + 4;
    static const int finalDrawVertex = cellsPerRow * 4;

    // Some tests iterate over either or both of the type of command buffer
    // insertion (call or copy) or type of draw (direct in a command buffer or
    // indirect from some other memory).
    enum CallCopyType {
        CommandBufferCall,
        CommandBufferCopy,
    };
    enum DrawType {
        DrawArrays,
        DrawArraysIndirect,
    };

    bool m_isFastpath;
    std::function<void(CommandBuffer *, int, LWN_NOESCAPE const LWNcommandHandle *)> m_callCommands;

public:
    explicit LWNCommandBufferCallCopyTest(bool isFastpath);
    LWNTEST_CppMethods();
};

LWNCommandBufferCallCopyTest::LWNCommandBufferCallCopyTest(bool isFastpath) :
    m_isFastpath(isFastpath),
    m_callCommands(isFastpath ? &CommandBuffer::CallCommands_fastpath : &CommandBuffer::CallCommands)
{
}

lwString LWNCommandBufferCallCopyTest::getDescription() const
{
    lwString callCommands = m_isFastpath ? "CallCommands_fastpath" : "CallCommands";
    lwStringBuf sb;
    sb <<
        "Basic test for lwn::CommandBuffer::" << callCommands << " and lwn::CommandBuffer::CopyCommands.  "
        "This test renders sets of quads to two render targets, where the first half "
        "goes to the left side of the 'left' render target and the second half goes to "
        "the right side of the 'right' render target.  The two render targets are combined "
        "side-by-side on screen.  Each row tests a different configuration, where draws "
        "are generated inline, inserted via " << callCommands << ", or inserted via CopyCommands.  "
        "We also insert commands to change render targets inline or via " << callCommands << " or "
        "CopyCommands.  We test direct draws, where draw data are stored in the command "
        "memory and indirect draws, where they are fetched from separate memory.\n\n"
        "In a passing image, each row has R/G/B/Y quads on the far left and M/C/W/Gray "
        "quads on the far right.";
    return sb.str();
}

int LWNCommandBufferCallCopyTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(55, 9);
}

void LWNCommandBufferCallCopyTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // The main program simply passes through the primitive with the color
    // specified in the vertex buffer.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec2 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "  ocolor = color;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";
    Program *pgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    // The "display" program uses draws a "full-viewport" quad, constructing
    // texture coordinates from position and using a bound 2D texture to fetch
    // the source color.
    VertexShader displayVS(440);
    displayVS <<
        "layout(location=0) in vec2 position;\n"
        "out vec2 tc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "  tc = position * 0.5 + 0.5;\n"
        "}\n";
    FragmentShader displayFS(440);
    displayFS <<
        "layout(binding=0) uniform sampler2D smp;\n"
        "in vec2 tc;\n"
        "out vec4 color;\n"
        "void main() {\n"
        "  color = texture(smp, tc);\n"
        "}\n";
    Program *displayPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(displayPgm, displayVS, displayFS)) {
        LWNFailTest();
        return;
    }

    // Set up a basic default sampler.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    Sampler *smp = sb.CreateSampler();
    LWNuint smpID = smp->GetRegisteredID();

    // Create two textures to display left and right halves of the screen.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetFlags(TextureFlags::COMPRESSIBLE);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(lwrrentWindowWidth/2, lwrrentWindowHeight);
    tb.SetFormat(Format::RGBA8);
    tb.SetLevels(1);
    LWNsizeiptr texStorageSize = tb.GetPaddedStorageSize();
    MemoryPoolAllocator texAllocator(device, NULL, 2 * texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    Texture *textures[2];
    TextureHandle texHandles[2];
    for (int i = 0; i < 2; i++) {
        textures[i] = texAllocator.allocTexture(&tb);
        texHandles[i] = device->GetTextureHandle(textures[i]->GetRegisteredTextureID(), smpID);
    }

    // Set up the vertex format and vertex buffer.
    struct Vertex {
        dt::vec2 position;
        dt::vec3 color;
    };
    static const dt::vec3 vcolors[] =
    {
        dt::vec3(1.0, 0.0, 0.0),
        dt::vec3(0.0, 1.0, 0.0),
        dt::vec3(0.0, 0.0, 1.0),
        dt::vec3(1.0, 1.0, 0.0),
        dt::vec3(1.0, 0.0, 1.0),
        dt::vec3(0.0, 1.0, 1.0),
        dt::vec3(1.0, 1.0, 1.0),
        dt::vec3(0.5, 0.5, 0.5),
    };

    MemoryPoolAllocator allocator(device, NULL,
                                  totalVerts * sizeof(Vertex) + cellsPerRow * sizeof(DrawArraysIndirectData),
                                  LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, totalVerts, allocator, NULL);
    BufferAddress vboAddr = vbo->GetAddress();
    Vertex *vboMap = (Vertex *) vbo->Map();

    // Fill in the four quads, each covering ~1/8th of the width of the viewport.
    for (int i = 0; i < 8; i++) {
        float cx = (i - 4 + 0.5) / 4.0;
        float sx = 0.9 / 8;
        vboMap[4 * i + 0].position = dt::vec2(cx - sx, -0.9);
        vboMap[4 * i + 1].position = dt::vec2(cx - sx, +0.9);
        vboMap[4 * i + 2].position = dt::vec2(cx + sx, -0.9);
        vboMap[4 * i + 3].position = dt::vec2(cx + sx, +0.9);
        for (int j = 0; j < 4; j++) {
            vboMap[4 * i + j].color = vcolors[i];
        }
    }

    // Then fill in a final quad covering the entire viewport (for display).
    for (int i = 0; i < 4; i++) {
        vboMap[finalDrawVertex + i].position = dt::vec2((i&2) ? +1.0 : -1.0, (i&1) ? +1.0 : -1.0);
        vboMap[finalDrawVertex + i].color = dt::vec3(1.0, 0.0, 0.0);
    }

    // Set up an indirect draw buffer, with a separate draw structure for each
    // cell (in a row).
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *indirect = allocator.allocBuffer(&bb, BUFFER_ALIGN_INDIRECT_BIT, cellsPerRow * sizeof(LWNdrawArraysIndirectData));
    DrawArraysIndirectData *indirects = (DrawArraysIndirectData *) indirect->Map();
    for (int i = 0; i < cellsPerRow; i++) {
        indirects[i].count = 4;
        indirects[i].instanceCount = 1;
        indirects[i].first = 4 * i;
        indirects[i].baseInstance = 0;
    }
    BufferAddress indirectAddr = indirect->GetAddress();

    // Set up command buffer objects using either the coherent pool (for
    // direct submission and CallCommands) and the non-coherent pool (for
    // CopyCommands).
    CommandBuffer *commandBuffers[2];
    commandBuffers[CommandBufferCall] = device->CreateCommandBuffer();
    g_lwnCommandMem.populateCommandBuffer(commandBuffers[CommandBufferCall], CommandBufferMemoryManager::Coherent);
    commandBuffers[CommandBufferCopy] = device->CreateCommandBuffer();
    g_lwnCommandMem.populateCommandBuffer(commandBuffers[CommandBufferCopy], CommandBufferMemoryManager::NonCoherent);

    // Generate a separate set of command handles for each command buffer
    // (callable or copyable), draw type, and cell, plus a set of command
    // handles for each command buffer to bind each render target.
    CommandHandle drawHandles[2][2][cellsPerRow];
    CommandHandle setRtHandles[2][2];
    for (int callCopyType = 0; callCopyType < 2; callCopyType++) {
        CommandBuffer *cb = commandBuffers[callCopyType];
        for (int drawIndirect = 0; drawIndirect < 2; drawIndirect++) {
            for (int draw = 0; draw < cellsPerRow; draw++) {
                cb->BeginRecording();
                if (drawIndirect) {
                    cb->DrawArraysIndirect(DrawPrimitive::TRIANGLE_STRIP, indirectAddr + draw * sizeof(LWNdrawArraysIndirectData));
                } else {
                    cb->DrawArrays(DrawPrimitive::TRIANGLE_STRIP, draw * 4, 4);
                }
                drawHandles[callCopyType][drawIndirect][draw] = cb->EndRecording();
            }
        }
        for (int i = 0; i < 2; i++) {
            cb->BeginRecording();
            cb->SetRenderTargets(1, &textures[i], NULL, NULL, NULL);
            setRtHandles[callCopyType][i] = cb->EndRecording();
        }
    }

    // Clear both the left and right render targets to black.
    for (int i = 0; i < 2; i++) {
        queueCB.SetRenderTargets(1, &textures[i], NULL, NULL, NULL);
        queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    }

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, totalVerts * sizeof(Vertex));

    int row = 0;

    // Render a reference row with the first triangles sent to the left
    // texture and the last triangles sent to the right texture.  No command
    // buffer calls or copies are used.
    queueCB.SetViewportScissor(0, row * cellSize, renderTargetWidth, cellSize);
    queueCB.SetRenderTargets(1, &textures[0], NULL, NULL, NULL);
    for (int i = 0; i < 4; i++) {
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 4 * i, 4);
    }
    queueCB.SetRenderTargets(1, &textures[1], NULL, NULL, NULL);
    for (int i = 0; i < 4; i++) {
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 4 * i + 16, 4);
    }
    row++;

    // Send similar rows where the draw calls for both left and right are
    // injected by CallCommands and/or CopyCommands (cbType) and use either
    // direct or indirect draws (drawType).  We test APIs using either one or
    // both handles.
    for (int drawType = 0; drawType < 2; drawType++) {
        for (int cbType = 0; cbType < 2; cbType++) {
            CommandHandle *rowHandles = drawHandles[cbType][drawType];
            queueCB.SetViewportScissor(0, row * cellSize, renderTargetWidth, cellSize);
            queueCB.SetRenderTargets(1, &textures[0], NULL, NULL, NULL);
            if (cbType == CommandBufferCall) {
                m_callCommands(&queueCB, 4, &rowHandles[0]);
            } else {
                queueCB.CopyCommands(4, &rowHandles[0]);
            }
            queueCB.SetRenderTargets(1, &textures[1], NULL, NULL, NULL);
            if (cbType == CommandBufferCall) {
                m_callCommands(&queueCB, 1, &rowHandles[4]);
                m_callCommands(&queueCB, 1, &rowHandles[5]);
                m_callCommands(&queueCB, 1, &rowHandles[6]);
                m_callCommands(&queueCB, 1, &rowHandles[7]);
            } else {
                queueCB.CopyCommands(1, &rowHandles[4]);
                queueCB.CopyCommands(1, &rowHandles[5]);
                queueCB.CopyCommands(1, &rowHandles[6]);
                queueCB.CopyCommands(1, &rowHandles[7]);
            }
            row++;
        }
    }

    // Send some rows where we change render targets via CallCommands or
    // CopyCommands, and insert draws the same way or directly.
    for (int callCopyType = 0; callCopyType < 2; callCopyType++) {
        for (int callCopyDraw = 0; callCopyDraw < 2; callCopyDraw++) {
            CommandHandle *rowDrawHandles = drawHandles[callCopyType][0];
            CommandHandle *rowRTHandles = setRtHandles[callCopyType];
            queueCB.SetViewportScissor(0, row * cellSize, renderTargetWidth, cellSize);
            for (int rt = 0; rt < 2; rt++) {
                if (callCopyType == CommandBufferCall) {
                    m_callCommands(&queueCB, 1, &rowRTHandles[rt]);
                    if (callCopyDraw) {
                        m_callCommands(&queueCB, 4, &rowDrawHandles[rt * 4]);
                    }
                } else {
                    queueCB.CopyCommands(1, &rowRTHandles[rt]);
                    if (callCopyDraw) {
                        queueCB.CopyCommands(4, &rowDrawHandles[rt * 4]);
                    }
                }
                if (!callCopyDraw) {
                    for (int i = 0; i < 4; i++) {
                        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 16 * rt + 4 * i, 4);
                    }
                }
            }
            row++;
        }
    }

    // Generate a row where we use a mix of CallCommands, CopyCommands, and
    // direct draw insertion, with both direct and indirect draws.
    queueCB.SetViewportScissor(0, row * cellSize, renderTargetWidth, cellSize);
    queueCB.SetRenderTargets(1, &textures[0], NULL, NULL, NULL);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    m_callCommands(&queueCB, 1, &drawHandles[CommandBufferCall][1][1]);
    queueCB.CopyCommands(1, &drawHandles[CommandBufferCopy][0][2]);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 12, 4);
    queueCB.SetRenderTargets(1, &textures[1], NULL, NULL, NULL);
    m_callCommands(&queueCB, 1, &drawHandles[CommandBufferCall][0][4]);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 20, 4);
    queueCB.CopyCommands(1, &drawHandles[CommandBufferCopy][1][6]);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 28, 4);
    row++;

    // Test inserting commands into a simple command "user" command buffer
    // that has nothing but the tested commands in it.  <queueCB> will pile
    // up a bunch of commands over several sub-tests.
    for (int callCopyType = 0; callCopyType < 2; callCopyType++) {

        // Start by programming the viewport/scissor (outside the small
        // command buffer) and flushing the queue command buffer.
        queueCB.SetViewportScissor(0, row * cellSize, renderTargetWidth, cellSize);
        queueCB.submit();

        CommandHandle *rowHandles = drawHandles[callCopyType][DrawArrays];
        CommandBuffer *cb = commandBuffers[CommandBufferCall];
        cb->BeginRecording();
        cb->SetRenderTargets(1, &textures[0], NULL, NULL, NULL);
        if (callCopyType == CommandBufferCall) {
            m_callCommands(cb, 4, &rowHandles[0]);
        } else {
            cb->CopyCommands(4, &rowHandles[0]);
        }
        cb->SetRenderTargets(1, &textures[1], NULL, NULL, NULL);
        if (callCopyType == CommandBufferCall) {
            m_callCommands(cb, 4, &rowHandles[4]);
        } else {
            cb->CopyCommands(4, &rowHandles[4]);
        }
        CommandHandle recordedHandle = cb->EndRecording();
        queue->SubmitCommands(1, &recordedHandle);

        row++;
    }

    // Insert a barrier to make sure the prevous rendering is done before the
    // display pass.
    queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS | BarrierBits::ILWALIDATE_TEXTURE);

    // Display the two render targets in the left and right half of the window
    // framebuffer.
    g_lwnWindowFramebuffer.bind();
    queueCB.BindProgram(displayPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    for (int i = 0; i < 2; i++) {
        queueCB.SetViewportScissor(i * lwrrentWindowWidth / 2, 0, lwrrentWindowWidth / 2, lwrrentWindowHeight);
        queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandles[i]);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, finalDrawVertex, 4);
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNCommandBufferCallCopyTest, lwn_cmdbuf_callcopy, (false));
OGTEST_CppTest(LWNCommandBufferCallCopyTest, lwn_cmdbuf_callcopy_fastpath, (true));
