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

class LWNTriLinOpt
{
public:
    LWNTEST_CppMethods();
};

lwString LWNTriLinOpt::getDescription() const
{
    lwStringBuf sb;
    sb << "Test for trilinear optimization API - lwnSamplerBuilderSetLodSnap()."
          "Draw a quad in 32 rows with LOD values from 0 to number of levels."
          "For each row use a different snap value";
    return sb.str();    
}

int LWNTriLinOpt::isSupported() const
{
    return lwogCheckLWNAPIVersion(53, 14);
}

static void PrepareTextureRGBA8(GLuint* data, int w, int h, int checkersize, GLuint color)
{
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            data[x  + w * y] = color;
        }
    }
}

void LWNTriLinOpt::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec2 position;\n"
        "out vec2 texCoord;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "  texCoord = (vec2(position) + vec2(1.0, 1.0)) * vec2(0.5, 0.5);\n"
        "}\n";
    FragmentShader fs(440);
    const int levels = 6;
    fs << "const int levels = " << levels << ";\n"
          "uniform sampler2D sampler;\n"
          "in vec2 texCoord;\n"
          "out vec4 color;\n"
          "void main() {\n"
          "  float lod = texCoord.x * float(levels);\n"
          "  color = textureLod(sampler, texCoord, lod);\n"
          "}\n";

    Program *pgm = device->CreateProgram();

    // Compile and call lwnProgramSetShaders.
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_2D);
    const int texSize = 1 << levels;
    tb.SetSize2D(texSize, texSize);
    tb.SetFormat(Format::RGBA8);
    tb.SetLevels(levels); 
    LWNsizeiptr texStorageSize = tb.GetPaddedStorageSize();

    MemoryPoolAllocator texAllocator(device, NULL, texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    Texture *mipMapTex = texAllocator.allocTexture(&tb);

    // Set up a buffer with pitch data for the two textures, and copy into the
    // texture storage.
    MemoryPoolAllocator texSrcAllocator(device, NULL, texStorageSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *buffer = texSrcAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, texStorageSize);
    BufferAddress bufferAddr = buffer->GetAddress();
    GLuint* bufferMap = (GLuint*)buffer->Map();

    // ARGB
    static const GLuint rainbowColors[] = {
        0xFFFF0000,
        0xFF00FF00,
        0xFF0000FF,
        0xFFFFFFFF,
        0xFF00FFFF,
        0xFFFF00FF
    };

    for (int l = 0; l < tb.GetLevels(); l++) {
        int levelSize = texSize >> l;
        GLuint c = rainbowColors[l % __GL_ARRAYSIZE(rainbowColors)];
        PrepareTextureRGBA8(bufferMap, levelSize, levelSize, 1, c);

        TextureView view;
        view.SetDefaults().SetLevels(l, 1);

        CopyRegion copyRegion = { 0, 0, 0, levelSize, levelSize, 1 };
        queueCB.CopyBufferToTexture(bufferAddr, mipMapTex, &view, &copyRegion, CopyFlags::NONE);
        queueCB.submit();
        queue->Finish();
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec2 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec2(-1.0,  -1.0) },
        { dt::vec2( 1.0,  -1.0) },
        { dt::vec2(-1.0,   1.0) },
        { dt::vec2( 1.0,   1.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 4 * sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);

    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    const int numSnapValues = (1 << 5);

    // callwlate all the snap values
    // and feed them to the LOD snap API
    // values are in e2m3 format
    LWNfloat snaps[numSnapValues];
    for (int v = 0; v < numSnapValues; v++) {
        float m = (float)(v & 0x7); // 2:0
        int e = (v >> 3) & 0x3; // 4:3
        float slope = (1.0 + (m / 8.0)) * (1 << e);
        snaps[v] = 0.5 * (1.0 - (1.0 / slope));
    }

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    CellIterator2D cell(1, numSnapValues);
    int cw = lwrrentWindowWidth, ch = lwrrentWindowHeight / numSnapValues;
    for (int snap = 0; snap < numSnapValues; snap++, cell++) {
        queueCB.SetViewportScissor(cell.x() * cw + 1, cell.y() * ch + 1, cw - 2, ch - 2);

        sb.SetMinMagFilter(MinFilter::LINEAR_MIPMAP_LINEAR, MagFilter::NEAREST);
        sb.SetLodSnap(snaps[snap]);
        Sampler *smp = sb.CreateSampler();
        TextureHandle texHandle = device->GetTextureHandle(mipMapTex->GetRegisteredTextureID(), smp->GetRegisteredID());

        queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNTriLinOpt, lwn_trilinopt,);
