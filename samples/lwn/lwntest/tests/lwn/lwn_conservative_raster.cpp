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

struct Vertex {
    dt::vec4 position;
    dt::vec4 color;
};

class LWNConservativeRasterTest
{
public:
    LWNConservativeRasterTest(LWNuint numSamples,
                              const DrawPrimitive& primitive,
                              const char* primitiveName,
                              LWNboolean multisample,
                              LWNfloat dilate) :
        mNumSamples(numSamples),
        mPrimitive(primitive),
        mPrimitiveName(primitiveName),
        mMultisample(multisample),
        mDilate(dilate)
    {
    }

    LWNTEST_CppMethods();

private:
    void Draw(Device *device, QueueCommandBuffer& queueCB, MemoryPoolAllocator& allocator, int i, int j) const;

    LWNuint mNumSamples;
    DrawPrimitive mPrimitive;
    const char *mPrimitiveName;
    LWNboolean mMultisample;
    LWNfloat mDilate;
};

lwString LWNConservativeRasterTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Tests conservative rasterization of primitives drawn with pseudorandom vertices. Each "
          "primitive is drawn once in red without conservative rasterzation, and once in green "
          "with conservative rasterzation. Additive blending is used. In general, each primitive "
          "should appear yellow, with a fringe of green. The width of the fringe should be 0-2 "
          "pixels, depending on dilation. Non-multisample line primitives may have red pixels near "
          "edges, because non-multisample lines are rasterized with vertical or horizontal end "
          "caps."
          "\n"
          "Render target samples: " << mNumSamples << "\n"
          "Primitive type: " << mPrimitiveName << "\n"
          "Multisample enabled: " << (mMultisample ? "YES" : "NO") << "\n"
          "Dilation: " << mDilate;
    return sb.str();
}

int LWNConservativeRasterTest::isSupported() const
{
    return g_lwnDeviceCaps.supportsConservativeRaster && lwogCheckLWNAPIVersion(38, 5);
}


void LWNConservativeRasterTest::Draw(Device *device, QueueCommandBuffer& queueCB, MemoryPoolAllocator& allocator, int i, int j) const
{
    // To ensure that random number generator state is consistent regardless of cell selection,
    // query the RNG even if this cell is not selected.
    float x1,x2,x3,y1,y2,y3;
    x1 = lwFloatRand(-1,1);
    x2 = lwFloatRand(-1,1);
    x3 = lwFloatRand(-1,1);
    y1 = lwFloatRand(-1,1);
    y2 = lwFloatRand(-1,1);
    y3 = lwFloatRand(-1,1);

    if (!cellAllowed(i,j)) {
        return;
    }

    // Set up a VBO with random 2D positions and associated RGBA colors.
    Vertex vertexData[3] = {
        { dt::vec4(x1, y1, 0, 1), dt::vec4(0, 1, 0, 1) },
        { dt::vec4(x2, y2, 0, 1), dt::vec4(0, 1, 0, 1) },
        { dt::vec4(x3, y3, 0, 1), dt::vec4(0, 1, 0, 1) },
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    // Use additive blending to ensure that overlapping primitives are both visible.
    ColorState colorState;
    colorState.SetDefaults()
              .SetBlendEnable(0, LWN_TRUE);
    queueCB.BindColorState(&colorState);

    BlendState blendState;
    blendState.SetDefaults()
              .SetBlendFunc(BlendFunc::ONE, BlendFunc::ONE,
                            BlendFunc::ONE, BlendFunc::ONE);
    queueCB.BindBlendState(&blendState);

    // First, draw a primitive in green with conservative rasterization enabled...
    queueCB.SetConservativeRasterEnable(LWN_TRUE);
    if (mDilate != 0) {
        queueCB.SetConservativeRasterDilate(mDilate);
    }

    queueCB.DrawArrays(mPrimitive, 0, 3);

    // ...Then draw the same primitive in red with conservative rasterization disabled.
    for (int i = 0; i < 3; ++i) {
        vertexData[i].color = dt::vec4(1, 0, 0, 1);
    }
    vbo = stream.AllocateVertexBuffer(device, 3, allocator, vertexData);
    vboAddr = vbo->GetAddress();
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    queueCB.SetConservativeRasterEnable(LWN_FALSE);

    queueCB.DrawArrays(mPrimitive, 0, 3);

    // Finally, clean up for the next cell.
    blendState.SetDefaults();
    queueCB.BindBlendState(&blendState);
    colorState.SetDefaults();
    queueCB.BindColorState(&colorState);
}

void LWNConservativeRasterTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    Framebuffer fb(lwrrentWindowWidth, lwrrentWindowHeight);

    if (mNumSamples > 0) {
        fb.setColorFormat(Format::RGBA8);
        fb.setSamples(mNumSamples);
        fb.alloc(device);
        fb.bind(queueCB);
    }

    queueCB.SetLineWidth(10.0);
    queueCB.SetPointSize(10.0);

    MultisampleState multisampleState;
    multisampleState.SetDefaults()
                    .SetMultisampleEnable(mMultisample)
                    .SetSamples(mNumSamples);
    queueCB.BindMultisampleState(&multisampleState);

    VertexShader vs(430);
    vs <<
        "layout (location = 1) in vec4 incol;\n"
        "layout (location = 0) in vec4 pos;\n"
        "out vec4 color;\n"
        "void main() {\n"
        "    gl_Position = pos;\n"
        "    color = incol;\n"
        "}";
    FragmentShader fs(430);
    fs <<
        "in vec4 color;\n"
        "layout(location = 0) out vec4 outColor[1];\n"
        "void main() {\n"
        "    outColor[0] = color;\n"
        "}\n";

    Program *pgm = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    MemoryPoolAllocator allocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    queueCB.ClearColor(0, 0.0, 0.0, 0.25, 0.0);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            queueCB.SetViewportScissor(lwrrentWindowWidth*i/4, lwrrentWindowHeight*j/4,
                                       lwrrentWindowWidth/4, lwrrentWindowHeight/4);
            Draw(device, queueCB, allocator, i, j);
        }
    }

    if (mNumSamples > 0) {
        fb.downsample(queueCB);
        CopyRegion region = {0, 0, 0, lwrrentWindowWidth, lwrrentWindowHeight, 1 };
        queueCB.CopyTextureToTexture(fb.getColorTexture(0), NULL, &region,
                                     g_lwnWindowFramebuffer.getAcquiredTexture(), NULL, &region,
                                     CopyFlags::NONE);
        g_lwnWindowFramebuffer.bind();
    }

    queueCB.submit();
    queue->Finish();
    fb.destroy();
}

#define CONSRAST(N,MODENAME,MODE)    \
    OGTEST_CppTest(LWNConservativeRasterTest, lwn_conservative_raster_##MODENAME##_##N##_msdisable, (N, DrawPrimitive::MODE, #MODE, LWN_FALSE, 0)); \
    OGTEST_CppTest(LWNConservativeRasterTest, lwn_conservative_raster_##MODENAME##_##N##_msenable, (N, DrawPrimitive::MODE, #MODE, LWN_TRUE, 0)); \
    OGTEST_CppTest(LWNConservativeRasterTest, lwn_conservative_raster_##MODENAME##_##N##_dilate25, (N, DrawPrimitive::MODE, #MODE, LWN_TRUE, 0.25f)); \
    OGTEST_CppTest(LWNConservativeRasterTest, lwn_conservative_raster_##MODENAME##_##N##_dilate50, (N, DrawPrimitive::MODE, #MODE, LWN_TRUE, 0.5f)); \
    OGTEST_CppTest(LWNConservativeRasterTest, lwn_conservative_raster_##MODENAME##_##N##_dilate75, (N, DrawPrimitive::MODE, #MODE, LWN_TRUE, 0.75f)); \

#define CONSRAST2(N)    \
    CONSRAST(N,tri,TRIANGLES)    \
    CONSRAST(N,line,LINES)       \
    CONSRAST(N,point,POINTS)     \

CONSRAST2(0)
CONSRAST2(4)


class LWNSnapBiasTest {
public:
    explicit LWNSnapBiasTest(bool y) : testYdim(y)
    {
    }
    LWNTEST_CppMethods();

private:
    enum { NUM_CELLS = 8 };

    bool testYdim;
};

lwString LWNSnapBiasTest::getDescription() const
{
    return "Render very slivery triangles with conservative rasterization. If the triangle snaps to "
           "zero area, nothing is drawn, else a line is drawn. The x-axis decreases the separation "
           "between vertices. The y-axis increases the subpixel bias. There should be a diagonal pattern "
           " where lwlling follows the precision controlled by the subpixel bias.\n";
}

int LWNSnapBiasTest::isSupported() const
{
    return g_lwnDeviceCaps.supportsConservativeRaster && lwogCheckLWNAPIVersion(38, 5);
}

void LWNSnapBiasTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    queueCB.ClearColor(0, 0.25, 0.0, 0.0, 0.0);

    VertexShader vs(430);
    vs <<
        "layout (location = 0) in vec2 pos;\n"
        "void main() {\n"
        "    gl_Position = vec4(pos, 0.0, 1.0);\n"
        "}";

    FragmentShader fs(430);
    fs <<
        "layout(location = 0) out vec4 outColor;\n"
        "void main() {\n"
        "    outColor = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "}\n";

    Program *pgm = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    VertexStream stream(sizeof(dt::vec2));
    stream.addAttribute<dt::vec2>(0);
    VertexArrayState vertex = stream.CreateVertexArrayState();

    MemoryPoolAllocator allocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    queueCB.SetConservativeRasterEnable(LWN_TRUE);

    int viewportWidth = lwrrentWindowWidth / NUM_CELLS;
    int viewportHeight = lwrrentWindowHeight / NUM_CELLS;

    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, NULL);
    BufferAddress vboAddr = vbo->GetAddress();
    char* vboPointer = static_cast<char*>(vbo->Map());
    queueCB.BindVertexArrayState(vertex);
    int vboOffset = 0;

    for (int i = 0; i < NUM_CELLS; ++i) {
        for (int j = 0; j < NUM_CELLS; ++j) {
            if (cellAllowed(i, j)) {
                queueCB.SetViewportScissor(viewportWidth * i, viewportHeight * j,
                                           viewportWidth, viewportHeight);
                float delta = 1.0f / (128 * (1 << i));
                dt::vec2 vertexData[3];

                if (testYdim) {
                    queueCB.SetSubpixelPrecisionBias(0, j);
                    // have y-coord at a half-pixel
                    float y = 1.0f / viewportHeight;
                    float height = delta * 2.0f / viewportHeight;
                    vertexData[0] = dt::vec2(-0.5f, y);
                    vertexData[1] = dt::vec2(-0.5f, y + height);
                    vertexData[2] = dt::vec2(0.5f, y);
                } else {
                    queueCB.SetSubpixelPrecisionBias(j, 0);
                    // have x-coord at a half-pixel
                    float x = 1.0f / viewportWidth;
                    float width = delta * 2.0f / viewportWidth;
                    vertexData[0] = dt::vec2(x, -0.5f);
                    vertexData[1] = dt::vec2(x + width, -0.5f);
                    vertexData[2] = dt::vec2(x, 0.5f);
                }

                memcpy(vboPointer + vboOffset, vertexData, sizeof(vertexData));
                queueCB.BindVertexBuffer(0, vboAddr + vboOffset, sizeof(vertexData));
                queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
                vboOffset += sizeof(vertexData);
            }
        }
    }

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNSnapBiasTest, lwn_snapbiasx, (false));
OGTEST_CppTest(LWNSnapBiasTest, lwn_snapbiasy, (true));
