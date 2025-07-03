/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
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
using namespace lwn::dt;

// --------------------------------- LWNCopyTextureLimitsTest ----------------------------------------

class LWNCopyTextureLimitsTest {
    static const int cellSize = 60;
    static const int cellMargin = 2;
    static const int defaultHeight = 2 * cellSize - 2 * cellMargin;
public:
    LWNTEST_CppMethods();
};


lwString LWNCopyTextureLimitsTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Simple test for verifying that copy buffer to texture does not cause launch check errors due to size limits. "
        "Fill a buffer, copy it to the wide/tall texture, draw the texture to a cell in the default framebuffer. "
        "Cells should be checkered green. First half of cells shows wide(Wx2) textures, second half tall(2xH) textures. "
        "Two rows, first row copies to the entire texture, second row only copies to a subregion with offset.";
    return sb.str();
}

int LWNCopyTextureLimitsTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(51, 1);
}

static void FillTextureRGBA32F(float* data, int w, int h, int checkersize)
{
    // Fill with checkerboard pattern.
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            vec4 *v = (vec4 *)&data[ (x * 4) + (w * 4) * y];
            bool checker = !!( ((x  / checkersize) % 2) ^ ((y  / checkersize) % 2) );
            if (checker) {
                v->setX(0);
                v->setY(0.75 * ((float)y/h));
                v->setZ(0);
            } else {
                v->setX(0);
                v->setY(1 * ((float)x/w));
                v->setZ(0);
            }
            v->setW(1);
        }
    }
}

void LWNCopyTextureLimitsTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    static const int cellsX = lwrrentWindowWidth / cellSize;
    static const int cellsY = lwrrentWindowHeight / cellSize;
    cellTestInit(cellsX, cellsY);

    VertexShader vs(440);
    vs <<
        "layout(location = 0) in vec4 position;\n"
        "layout(location = 1) in vec2 tc;\n"
        "out vec2 texcoord;\n"
        "void main() {\n"
        "  gl_Position = position;\n"
        "  texcoord = vec2(0.5, 0.5) * (vec2(1.0, 1.0) + vec2(position));\n"
        "}\n";

    FragmentShader fs_texture(440);
    fs_texture << "precision highp float;\n"
        "layout(location = 0) out vec4 color;\n"
        "layout (binding=0) uniform sampler2D tex;\n"
        "in vec2 texcoord;\n"
        "void main() {\n"
        "  color = vec4(texture(tex, texcoord));\n"
        "}\n";

    Program* pgmTexture = device->CreateProgram();
    LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgmTexture, vs, fs_texture);
    if (!compiled) {
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec4 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec4(-1.0, -1.0, 0.0, 1.0) },
        { dt::vec4(-1.0, +1.0, 0.0, 1.0) },
        { dt::vec4(+1.0, -1.0, 0.0, 1.0) },
        { dt::vec4(+1.0, +1.0, 0.0, 1.0) },
    };

    MemoryPoolAllocator vertex_allocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexStreamSet streamSet(stream);
    VertexArrayState vertex = streamSet.CreateVertexArrayState();

    Buffer* vbo = stream.AllocateVertexBuffer(device, 4, vertex_allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    queueCB.BindProgram(pgmTexture, ShaderStageBits::ALL_GRAPHICS_BITS);

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.2, 0.2, 0.2, 1.0);

    // width of 4097 is where copy engine needs enable remap to work around width limits (block
    // linear surface width <= 64k bytes)
    const int dim[] =  { 2048, 4096, 4097 , 8192, 16384 };

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults().
       SetWrapMode(WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP_TO_BORDER).
       SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler *smp = sb.CreateSampler();
    LWNuint smpID = smp->GetRegisteredID();

    // just create a buffer big enough to fill the RGBA32F texture in any case
    // 16k is current texture size limit (Maxwell, Pascal makes this 32k but 
    // we don't support Pascal)
    const int dsize = 16384 * defaultHeight * 4 * sizeof(float);
    MemoryPool *bpool = device->CreateMemoryPool(NULL, dsize, MemoryPoolType::CPU_COHERENT);
    Buffer *buffer = bb.CreateBufferFromPool(bpool, 0, dsize);
    LWNfloat* data = (LWNfloat*)buffer->Map();
    LWNbufferAddress bufferAddress = buffer->GetAddress();

    for (unsigned int test_offsets = 0; test_offsets < 2; test_offsets++) {
        for (unsigned int i = 0; i < 2 * __GL_ARRAYSIZE(dim); i++) {
            unsigned int w = (i < __GL_ARRAYSIZE(dim)) ? dim[i % __GL_ARRAYSIZE(dim)] : defaultHeight;
            unsigned int h = (i < __GL_ARRAYSIZE(dim)) ? defaultHeight : dim[i % __GL_ARRAYSIZE(dim)];

            Framebuffer fb;
            fb.setSize(w, h);
            fb.setColorFormat(Format::RGBA32F);
            fb.alloc(device);
            
            fb.bind(queueCB);
            fb.setViewportScissor();
            
            queueCB.ClearColor(0, 0, 0.25, 0, 1);

            FillTextureRGBA32F(data, w, h, cellSize/2);

            unsigned int copyw = w;
            unsigned int copyh = h;
            unsigned int xoffs = 0;
            unsigned int yoffs = 0;

            if (test_offsets) {
                copyw = (i < __GL_ARRAYSIZE(dim)) ? w/2 : defaultHeight;
                xoffs = (i < __GL_ARRAYSIZE(dim)) ? w/2: 0;
                copyh = (i < __GL_ARRAYSIZE(dim)) ? defaultHeight : h/2;
                yoffs = (i < __GL_ARRAYSIZE(dim)) ? 0 : h/2;
            }

            CopyRegion copyRegion = { (int)xoffs, (int)yoffs, 0, (int)copyw, (int)copyh, 1 };
            queueCB.CopyBufferToTexture(bufferAddress, fb.getColorTexture(0), NULL, &copyRegion, 0);

            g_lwnWindowFramebuffer.bind();
            g_lwnWindowFramebuffer.setViewportScissor();

            int view_x, view_y, view_width, view_height;
            cellGetRectPadded(i % cellsX, test_offsets + (i / cellsX), 2, &view_x, &view_y, &view_width, &view_height);

            Texture *colorTex = fb.getColorTexture(0);
            LWNuint texID = colorTex->GetRegisteredTextureID();

            TextureHandle hTex = device->GetTextureHandle(texID, smpID);

            queueCB.SetViewportScissor(view_x, view_y, view_width, view_height);

            queueCB.BindTexture(ShaderStage::FRAGMENT, 0, hTex);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

            queueCB.submit();
            queue->Finish();

            fb.destroy();
        }
    }

    buffer->Free();
}

OGTEST_CppTest(LWNCopyTextureLimitsTest, lwn_copy_texture_limits, );
