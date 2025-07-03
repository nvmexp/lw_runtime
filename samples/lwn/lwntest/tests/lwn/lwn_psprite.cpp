/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;

class LWNPointSpriteTest
{
    static const int texSize = 32;
    void draw(Device *device, Queue *queue, QueueCommandBuffer &queueCB) const;
    WindowOriginMode m_origin;
public:
    LWNPointSpriteTest(WindowOriginMode origin) : m_origin(origin) {}
    LWNTEST_CppMethods();
};

lwString LWNPointSpriteTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic point sprite test for LWN.  This test draws a 3x2 grid of cells using "
        << (m_origin == WindowOriginMode::UPPER_LEFT ? "an upper left" : "a lower left") <<
        " origin.  Each cell has three point sprites arranged in a triangle with two "
        "sprites on the bottom.  Each point should display a texture map that is "
        "black with a yellow 'T' shape.  The joint of the 'T' should be at the top. "
        "The three columns display points of various sizes with larger points on the "
        << (m_origin == WindowOriginMode::UPPER_LEFT ? "right" : "left") << ".  The "
        << (m_origin == WindowOriginMode::UPPER_LEFT ? "top" : "bottom") <<
        " row draws point primitives, while the "
        << (m_origin == WindowOriginMode::UPPER_LEFT ? "bottom" : "top") <<
        " row draws a single triangle in point mode.";
    return sb.str();    
}

int LWNPointSpriteTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(38, 6);
}

void LWNPointSpriteTest::draw(Device *device, Queue *queue, QueueCommandBuffer &queueCB) const
{
    // Set up basic shaders that pass position through and render point sprites
    // using a texture lookup with gl_PointCoord as the coordinate.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "uniform sampler2D tex;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(tex, gl_PointCoord);\n"
        "}\n";

    Program *pgm = device->CreateProgram();

    // Compile and call lwnProgramSetShaders.
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-0.5, -0.5, 0.0) },
        { dt::vec3(+0.5, -0.5, 0.0) },
        { dt::vec3(+0.0, +0.5, 0.0) },
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    MemoryPoolAllocator bufAllocator(device, NULL, 64 * 1024, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, bufAllocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Allocate a texture for the point sprite.
    MemoryPoolAllocator texAllocator(device, NULL, 64 * 1024, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_2D).
        SetSize2D(texSize, texSize).SetFormat(Format::RGBA8).SetLevels(1);
    Texture *tex = texAllocator.allocTexture(&tb);

    // Allocate a sampler for rendering.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);
    Sampler *sampler = sb.CreateSampler();

    // Set up the texture handle.
    TextureHandle texHandle = device->GetTextureHandle(tex->GetRegisteredTextureID(), sampler->GetRegisteredID());

    // Fill the texture with a "T" shape.
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *texData = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, texSize * texSize * 4);
    BufferAddress texDataAddr = texData->GetAddress();
    LWNuint *texMem = (LWNuint *) texData->Map();
    for (int row = 0; row < texSize; row++) {
        for (int col = 0; col < texSize; col++) {
            if ((row == 2 || row == 3) && (col >= 2 && col < texSize - 2)) {
                *texMem++ = 0xFF00FFFF;
            } else if ((col == texSize / 2 - 1 || col == texSize / 2) && (row >= 2 && row < texSize - 2)) {
                *texMem++ = 0xFF00FFFF;
            } else {
                *texMem++ = 0x00000000;
            }
        }
    }

    CopyRegion copyRegion = { 0, 0, 0, texSize, texSize, 1 };
    queueCB.CopyBufferToTexture(texDataAddr, tex, NULL, &copyRegion, CopyFlags::NONE);

    // Set up to render polygons in point mode.
    PolygonState polygon;
    polygon.SetDefaults();
    polygon.SetPolygonMode(PolygonMode::POINT);
    queueCB.BindPolygonState(&polygon);


    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);

    for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 3; col++) {
            LWNfloat size;
            queueCB.SetViewportScissor(col * lwrrentWindowWidth / 3,
                                       row * lwrrentWindowHeight / 2,
                                       lwrrentWindowWidth / 3, lwrrentWindowHeight / 2);
            if (m_origin == WindowOriginMode::UPPER_LEFT) {
                size = LWNfloat(16 << col);  // 16, 32, 64 from left to right
            } else {
                size = LWNfloat(64 >> col);  // 64, 32, 16 from left to right
            }
            queueCB.SetPointSize(size);

            // Draw points on row 0 and (point-mode) triangles on row 1.
            queueCB.DrawArrays(row ? DrawPrimitive::TRIANGLES : DrawPrimitive::POINTS, 0, 3);
        }
    }

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queueCB.submit();
    queue->Finish();

    // Free temporary object allocations.
    pgm->Free();
    sampler->Free();
    bufAllocator.freeBuffer(vbo);
    bufAllocator.freeBuffer(texData);
    texAllocator.freeTexture(tex);
}

void LWNPointSpriteTest::doGraphics() const
{
    DisableLWNObjectTracking();

    // Set up and activate a temporary device and related state for this test,
    // which will use the window origin mode requested in the test.  Set the
    // appropriate window origin mode when creating.
    DeviceState *testDevice = new DeviceState(LWNdeviceFlagBits(0), m_origin);
    if (!testDevice || !testDevice->isValid()) {
        delete testDevice;
        DeviceState::SetDefaultActive();
        LWNFailTest();
        return;
    }

    Device *device = testDevice->getDevice();
    QueueCommandBuffer &queueCB = testDevice->getQueueCB();
    Queue *queue = testDevice->getQueue();
    testDevice->SetActive();

    Framebuffer fbo;
    fbo.setSize(lwrrentWindowWidth, lwrrentWindowHeight);
    fbo.setColorFormat(0, Format::RGBA8);
    fbo.alloc(device);
    fbo.bind(queueCB);

    queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);

    // Render the actual test.
    draw(device, queue, queueCB);

    // Read back the framebuffer to CPU memory.  Unfortunately, it's diffilwlt
    // to copy from our temporary-device FBO to the regular device window
    // framebuffer.
    LWNuint *fboData = new LWNuint[lwrrentWindowWidth * lwrrentWindowHeight];
    ReadTextureDataRGBA8(device, queue, queueCB, fbo.getColorTexture(0),
                         lwrrentWindowWidth, lwrrentWindowHeight, fboData);

    // Tear down the temporary device and related state.
    fbo.destroy();
    delete testDevice;
    DeviceState::SetDefaultActive();

    EnableLWNObjectTracking();

    // Our readback is top-to-bottom in UPPER_LEFT, but the writePixels()
    // operation below will use the regular device and will be bottom-to-top.
    // So ilwert the read-back data to compensate for that.
    if (m_origin == WindowOriginMode::UPPER_LEFT) {
        LWNuint *fboDataIlwerted = new LWNuint[lwrrentWindowWidth * lwrrentWindowHeight];
        for (int row = 0; row < lwrrentWindowHeight; row++) {
            memcpy(fboDataIlwerted + row * lwrrentWindowWidth,
                   fboData + (lwrrentWindowHeight - 1 - row) * lwrrentWindowWidth,
                   lwrrentWindowWidth * sizeof(LWNuint));
        }
        delete[] fboData;
        fboData = fboDataIlwerted;
    }

    // Now put the rendered image into the window framebuffer using the main
    // device.
    g_lwnWindowFramebuffer.writePixels(fboData);
    delete[] fboData;
}

OGTEST_CppTest(LWNPointSpriteTest, lwn_psprite_ll, (WindowOriginMode::LOWER_LEFT));
OGTEST_CppTest(LWNPointSpriteTest, lwn_psprite_ul, (WindowOriginMode::UPPER_LEFT));
