/*
 * Copyright (c) 2017 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <vector>

#include "lwntest_cpp.h"

#include "lwn_utils.h"

using namespace lwn;

class LWNDefaultStateTest
{
    static const int cellSize = 40;
    static const int cellMargin = 2;
    static const int cellsX = 640 / cellSize;
    static const int cellsY = 480 / cellSize;
    void displayResults(std::vector<bool> &results) const;

    // Utility class to manage reading back the contents of a framebuffer to a
    // wrapped system memory allocation, and to examine individual pixels.
    class ReadbackBuffer {
        Device *m_device;
        Queue *m_queue;
        QueueCommandBuffer &m_queueCB;
        Framebuffer &m_fbo;
        int m_width;
        int m_height;
        uint32_t *m_buffer;
    public:
        ReadbackBuffer(Device *device, Queue *queue, QueueCommandBuffer &queueCB,
                       Framebuffer &fbo, int w, int h) :
            m_device(device), m_queue(queue), m_queueCB(queueCB), m_fbo(fbo),
            m_width(w), m_height(h)
        {
            m_buffer = new uint32_t[w*h];
        }
        ~ReadbackBuffer()
        {
            delete m_buffer;
        }

        // read:  Read back the pixels for the full framebuffer.
        void read()
        {
            ReadTextureDataRGBA8(m_device, m_queue, m_queueCB, m_fbo.getColorTexture(0),
                                 m_width, m_height, m_buffer);
        }

        // pixel:  Return the packed 32-bit pixel at (x,y).
        uint32_t pixel(int x, int y)
        {
            assert(x < m_width && y < m_height);
            return m_buffer[y * m_width + x];
        }
    };

public:
    LWNTEST_CppMethods();
};

lwString LWNDefaultStateTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic test of API defaults for various rendering state.  This test "
        "runs with a dedicated context and tests various rendering "
        "operations that exercise default state values before they are "
        "explicitly set.  These tests are self-checking.  Passing sub-tests "
        "will be rendered as green squares; failing ones will be rendered "
        "in red.";
    return sb.str();
}

int LWNDefaultStateTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(5, 0);
}

// Display the contents of a boolean result vector on-screen as red/green
// cells.
void LWNDefaultStateTest::displayResults(std::vector<bool> &results) const
{
    CellIterator2D cell(cellsX, cellsY);
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    for (size_t i = 0; i < results.size(); i++) {
        queueCB.SetScissor(cell.x() * cellSize + cellMargin, cell.y() * cellSize + cellMargin,
                           cellSize - 2 * cellMargin, cellSize - 2 * cellMargin);
        queueCB.ClearColor(0, results[i] ? 0.0 : 1.0, results[i] ? 1.0 : 0.0, 0.0, 1.0);
        cell++;
    }
    queueCB.submit();
    queue->Finish();
}

void LWNDefaultStateTest::doGraphics() const
{
    std::vector<bool> results;

    // We use a temporary device for this test so we can test defaults before
    // the default state might be overwritten by another test.
    DisableLWNObjectTracking();
    DeviceState *testDevice = new DeviceState();
    if (!testDevice || !testDevice->isValid()) {
        delete testDevice;
        DeviceState::SetDefaultActive();
        LWNFailTest();
        return;
    }
    testDevice->SetActive();
    Device *device = testDevice->getDevice();
    QueueCommandBuffer &queueCB = testDevice->getQueueCB();
    Queue *queue = testDevice->getQueue();

    // Basic vertex/fragment shader combination that passes through position
    // and color data from vertex buffers.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = color;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";


    // Compile and call lwnProgramSetShaders.
    Program *pgm = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        // Simple triangle strip covering the center of the screen (-0.5 to
        // +0.5).
        { dt::vec3(-0.5, -0.5, 0.0), dt::vec3(1.0, 1.0, 1.0), },
        { dt::vec3(-0.5, +0.5, 0.0), dt::vec3(1.0, 1.0, 1.0), },
        { dt::vec3(+0.5, -0.5, 0.0), dt::vec3(1.0, 1.0, 1.0), },
        { dt::vec3(+0.5, +0.5, 0.0), dt::vec3(1.0, 1.0, 1.0), },

        // Points centered on the 4 center pixels on a 4x4 framebuffer.
        { dt::vec3(-0.25, -0.25, 0.0), dt::vec3(1.0, 1.0, 1.0), },
        { dt::vec3(-0.25, +0.25, 0.0), dt::vec3(1.0, 1.0, 1.0), },
        { dt::vec3(+0.25, -0.25, 0.0), dt::vec3(1.0, 1.0, 1.0), },
        { dt::vec3(+0.25, +0.25, 0.0), dt::vec3(1.0, 1.0, 1.0), },
    };

    MemoryPoolAllocator *vboAllocator =
        new MemoryPoolAllocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexData), *vboAllocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    // Set up a simple small framebuffer for off-screen rendering.
    const int fboWidth = 4;
    const int fboHeight = 4;
    Framebuffer fbo;
    ReadbackBuffer readback(device, queue, queueCB, fbo, fboWidth, fboHeight);
    fbo.setSize(fboWidth, fboHeight);
    fbo.setColorFormat(0, Format::RGBA8);
    fbo.setDepthStencilFormat(Format::DEPTH24_STENCIL8);
    fbo.alloc(device);
    fbo.bind(queueCB);
    queueCB.SetViewportScissor(0, 0, fboWidth, fboHeight);

    // Set up a stencil state object performing a simple EQUAL test that
    // increments only if the test passes.
    DepthStencilState dss;
    dss.SetDefaults();
    dss.SetStencilTestEnable(LWN_TRUE);
    dss.SetStencilFunc(Face::FRONT_AND_BACK, StencilFunc::EQUAL);
    dss.SetStencilOp(Face::FRONT_AND_BACK, StencilOp::KEEP, StencilOp::KEEP, StencilOp::INCR);
    queueCB.BindDepthStencilState(&dss);

    // Run several iterations of draws using default or explicitly-programmed
    // stencil test mask and reference values, with the stencil buffer cleared
    // to either zero or one.
    for (int loop = 0; loop < 5; loop++) {
        for (int clearval = 0; clearval < 2; clearval++) {

            // By default, the test should be "stencil == 0" and we draw only
            // if the stencil buffer is cleared to zero.
            bool draw = (clearval == 0);
            switch (loop) {
            case 0:
                // Use the default reference value (0) and value mask (0xFF).
                // Test will fail if the mask is programmed to zero (hardware
                // defaults) and the stencil buffer is cleared to one.
                break;
            case 1:
                // Now run with explicit value mask of 0x00; the EQUAL test
                // should pass regardless of the stencil buffer values.
                queueCB.SetStencilValueMask(Face::FRONT_AND_BACK, 0x00);
                draw = true;
                break;
            case 2:
                // Now run with explicit value mask of 0xFF, which should
                // behave like the default.
                queueCB.SetStencilValueMask(Face::FRONT_AND_BACK, 0xFF);
                break;
            case 3:
                // Now run with explicit reference value of 0x1 and explicit
                // value mask of 0xFF.  Should pass only when cleared to one.
                queueCB.SetStencilRef(Face::FRONT_AND_BACK, 0x01);
                draw = (clearval == 1);
                break;
            case 4:
                // Now run with explicit reference value of 0x0, which matches
                // the default behavior.
                queueCB.SetStencilRef(Face::FRONT_AND_BACK, 0x00);
                break;
            default:
                assert(0);
                break;
            }

            // Draw one quad with the stencil state above and then check if
            // (a) an uncovered pixel in the corner is drawn and (b) a covered
            // pixel near the center is drawn.
            queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
            queueCB.ClearDepthStencil(0, LWN_FALSE, clearval, ~0);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
            readback.read();
            results.push_back(readback.pixel(0, 0) == 0x00000000);
            results.push_back(readback.pixel(fboWidth / 2, fboHeight / 2) == (draw ? 0xFFFFFFFF : 0x00000000));
        }
    }

    // Run several iterations of draws using default or explicitly-programmed
    // stencil write mask values.  We clear the stencil buffer to zero, use an
    // "==0" test on the first draw and "==1" on the second.  We should
    // increment on both draws, and the second draw should happen only if we
    // incremented the stencil values on the first draw.
    queueCB.SetStencilValueMask(Face::FRONT_AND_BACK, 0xFF);
    for (int loop = 0; loop < 3; loop++) {

        // By default, we should draw on the second pass.
        bool draw = true;
        switch (loop) {
        case 0:
            // Use the default write mask of 0xFF, which should cause us to
            // update stencil on the first pass.
            break;
        case 1:
            // Use an explicit write mask of 0x00, which should cause us not
            // to update stencil on the first pass.
            queueCB.SetStencilMask(Face::FRONT_AND_BACK, 0x00);
            draw = false;
            break;
        case 2:
            // Use an explicit write mask of 0xFF, which should cause us to
            // update stencil on the first pass.
            queueCB.SetStencilMask(Face::FRONT_AND_BACK, 0xFF);
            break;
        default:
            assert(0);
            break;
        }

        // Draw the first quad, clear the color buffer, and draw the second
        // quad.
        queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
        queueCB.ClearDepthStencil(0, LWN_FALSE, 0, ~0);
        queueCB.SetStencilRef(Face::FRONT_AND_BACK, 0);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
        queueCB.SetStencilRef(Face::FRONT_AND_BACK, 1);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        readback.read();
        results.push_back(readback.pixel(0, 0) == 0x00000000);
        results.push_back(readback.pixel(fboWidth / 2, fboHeight / 2) == (draw ? 0xFFFFFFFF : 0x00000000));
    }

    // Reset depth/stencil state to defaults.
    dss.SetDefaults();
    queueCB.BindDepthStencilState(&dss);

    // Verify that we don't have a default point size of zero (bug 1899693).
    // Draw four points centered on the four pixels in the middle of the 4x4
    // viewport and verify that they show up in the framebuffer.  Also verify
    // that selected points outside this 2x2 range are not illuminated.
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.DrawArrays(DrawPrimitive::POINTS, 4, 4);
    readback.read();
    results.push_back(readback.pixel(1, 1) == 0xFFFFFFFF);
    results.push_back(readback.pixel(2, 1) == 0xFFFFFFFF);
    results.push_back(readback.pixel(1, 2) == 0xFFFFFFFF);
    results.push_back(readback.pixel(2, 2) == 0xFFFFFFFF);
    results.push_back(readback.pixel(1, 0) == 0x00000000);
    results.push_back(readback.pixel(0, 1) == 0x00000000);
    results.push_back(readback.pixel(3, 2) == 0x00000000);
    results.push_back(readback.pixel(2, 3) == 0x00000000);

    // Clean up resources explicitly, since we don't do automatic cleanup when
    // using non-default devices.
    pgm->Free();
    delete vboAllocator;
    fbo.destroy();

    // Clean up the temporary device and switch back to the main device to
    // display the results of the test.
    delete testDevice;
    DeviceState::SetDefaultActive();
    EnableLWNObjectTracking();
    displayResults(results);
}

OGTEST_CppTest(LWNDefaultStateTest, lwn_defaults, );
