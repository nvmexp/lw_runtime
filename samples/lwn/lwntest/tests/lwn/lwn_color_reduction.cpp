/*
 * Copyright (c) 2017 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include <time.h>

//////////////////////////////////////////////////////////////////////////

using namespace lwn;

class LWNcolorReductionTest
{
public:
    LWNcolorReductionTest(Format fmt, FormatClass fmtClass)
        : m_fmt(fmt),
          m_fmtClass(fmtClass),
          m_samples(4)
    {
        switch(m_fmt) {
            default:
                m_thresholdRange = 256;
                break;
            case Format::R11G11B10F:
                m_thresholdRange = 64;
                break;
        }
    }

    LWNTEST_CppMethods();
private:
    void setupTextureBuilder(Device *device, TextureBuilder& tb) const;

    Format      m_fmt;
    FormatClass m_fmtClass;
    int         m_samples;
    int         m_thresholdRange;

    const int   gridSize = 4; // 4x4 at 4x MSAA should cover a ROP tile for UNORM8
};

int LWNcolorReductionTest::isSupported() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();

    TextureBuilder tb;
    setupTextureBuilder(device, tb);

    // LWNstorageClass is pagekind ORed with a valid bit. Page
    // kind 0xFE is generic but we would expect something
    // else if compression can be used which will
    // allow color reduction.
    LWNstorageClass cls = tb.GetStorageClass();

    return (cls != 0x1FE) && lwogCheckLWNAPIVersion(53, 206);
}

lwString LWNcolorReductionTest::getDescription() const
{
    return "Test color reduction. Tests format class UNORM8, UNORM16,\n"
           "UNORM10, FP11, FP16, SRGB8. Draw to a texture of reducible\n"
           "format in 4x4 grids touching pixels as required to show\n"
           "reduction (once or multiple times)\n"
           "Four rows are drawn per format with falling threshold value:\n"
           "Row1: cover once, reduce conservative threshold, aggressive threshold is zero.\n"
           "Row2: cover once, reduce aggressive threshold, conservative threshold is zero.\n"
           "Row1: cover twice, reduce conservative threshold, aggressive threshold is zero.\n"
           "Row1: cover twice, reduce aggressive threshold, conservative threshold is zero.\n"
           "Draw results scaled from 4x4 and visualize areas that have been reduces by\n"
           "comparing color values in a shader, light green reduced, dark green not reduced.\n";
}

void LWNcolorReductionTest::setupTextureBuilder(Device *device, TextureBuilder& tb) const
{
    tb.SetDefaults();
    tb.SetDevice(device);
    tb.SetSize2D(lwrrentWindowWidth, lwrrentWindowHeight);
    tb.SetFlags(TextureFlags::COMPRESSIBLE);
    tb.SetFormat(m_fmt);
    tb.SetSamples(4);
    tb.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE);
}

void LWNcolorReductionTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    float c1 = 1.0;
    float delta = 0.0f;
    switch(m_fmt) {
        default:
            assert(0);
            break;
        case Format::RGBA8:
        case Format::BGRA8_SRGB:
            delta = (float)(1<<7)/(float)(1<<8);
            break;
        case Format::RGB10A2:
            delta = (float)(1<<7)/(float)(1<<10);
            break;
        case Format::RGBA16:
            delta = (float)(1<<7)/(float)(1<<16);
            break;
        case Format::RGBA16F:
            // randomly hand-picked so exponent is the same for both colors
            c1 = 22.0;      // binary (-1)0 * 1010011 - 1111 * 1.0110000000 = (-1)0 * 24 * 1.375
            delta = 2.0;    // binary (-1)0 * 1010011 - 1111 * 1.0100000000 = (-1)0 * 24 * 1.25
            break;
        case Format::R11G11B10F:
            // randomly hand-picked so exponent is the same for both colors
            c1 = 1536;      // binary (-1)0 * 1011001 - 1111 * 1.100000 = (-1)0 * 210 * 1.5
            delta = 512;    // binary (-1)0 * 1011001 - 1111 * 1.000000 = (-1)0 * 210 * 1.0
            break;
    }

    VertexShader vs(440);
    vs <<
        "layout(location = 0) in vec2 position;\n"
        "out flat vec4 c;\n"
        "out vec2 texCoord;\n"
        "void main() {\n"
        "  gl_Position = vec4(position.x + float(gl_InstanceID & 1), position.y, 0.0, 1.0);\n"
        "  texCoord = (vec2(position) + vec2(1.0, 1.0)) * vec2(0.5, 0.5);\n"
        "  if((gl_InstanceID & 1) == 0)\n"
        "     c = vec4(0.0, " << c1 << ", 0.0, 1.0);\n"
        "  else\n"
        "     c = vec4(0.0, " << (c1 - delta) << ", 0.0, 1.0);\n"
        "}\n";

    FragmentShader fs(440);
    fs << "precision highp float;\n"
        "layout(location = 0) out vec4 color;\n"
        "in flat vec4 c;\n"
        "in vec2 texCoord;\n"
        "void main() {\n"
        "    color = c;\n"
        "}\n";

    VertexShader vsReduce(440);
    vsReduce <<
        "layout(location = 0) in vec2 position;\n"
        "void main() {\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    FragmentShader fsReduce(440);
    fsReduce <<
        "uniform highp sampler2DMS msbuf;\n"
        "out vec4 color;\n"
        "void main() {\n"
        "   vec4 samples[4];\n"
        "   bool reduced = true;\n"
        "   ivec2 xy = ivec2(gl_FragCoord.xy/vec2(4.0,16.0));\n"
        "   for (int i = 0; i < 4; i++) {\n"
        "       samples[i] = texelFetch(msbuf, xy, i);\n"
        "   }\n"
        "   if (samples[0] == vec4(0,0,0,1) )\n"
        "       discard;\n"
        "   for (int i = 1; i < 4; i++) {\n"
        "       if (any(notEqual(samples[0], samples[i]))) {\n"
        "           reduced = false;\n"
        "           break;\n"
        "       }\n"
        "   }\n"
        "   color = vec4(0.0, (reduced == true) ? 1.0 : 0.3, 0.0, 1.0);\n"
        "}\n";

    // shader program
    Program *pgm = device->CreateProgram();
    LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
    if (!compiled) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        return;
    }

    Program *pgmReduce = device->CreateProgram();
    compiled = g_glslcHelper->CompileAndSetShaders(pgmReduce, vsReduce, fsReduce);
    if (!compiled) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        return;
    }

    CommandBuffer *cmdBuf = device->CreateCommandBuffer();

    const int command_size = 0x400000;
    const int control_size = 0x400000;

    // CPU_COHERENT means CPU_UNCACHED | GPU_CACHED | COMPRESSIBLE
    MemoryPool *mp = device->CreateMemoryPool(NULL, 2 * command_size, MemoryPoolType::CPU_COHERENT);

    int control_allocated = 2 * control_size;
    char *control_space = new char[control_allocated];
    memset(control_space, 0, control_allocated);

    // This is the 'classic' use: outside of a recording.
    cmdBuf->AddCommandMemory(mp, 0, command_size);
    cmdBuf->AddControlMemory(control_space , control_size);

    cmdBuf->BeginRecording();

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec2 position;
    };

    MemoryPoolAllocator vertex_allocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexStreamSet streamSet(stream);
    VertexArrayState quad = streamSet.CreateVertexArrayState();

    static const Vertex vertexData[] = {
        { dt::vec2(-1.0, -1.0) },
        { dt::vec2( 0.0, -1.0) },
        { dt::vec2(-1.0,  1.0) },
        { dt::vec2( 0.0,  1.0) },
        // used for reduce
        { dt::vec2(-1.0, -1.0) },
        { dt::vec2( 1.0, -1.0) },
        { dt::vec2(-1.0,  1.0) },
        { dt::vec2( 1.0,  1.0) },
    };

    Buffer *vboLines = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexData), vertex_allocator, vertexData);
    BufferAddress quadAddr = vboLines->GetAddress();

    cmdBuf->BindVertexBuffer(0, quadAddr, sizeof(vertexData));
    cmdBuf->BindVertexArrayState(quad);

    // MS state
    MultisampleState multisample;
    multisample.SetDefaults().SetSamples(m_samples);
    MultisampleState multisampleOff;
    multisampleOff.SetDefaults().SetMultisampleEnable(LWN_FALSE);

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults().SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    sb.SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);
    Sampler *sampler = sb.CreateSampler();

    LWNfloat fcolor_clear[] = { 0, 0, 0, 1 };
    cmdBuf->ClearColor(0, fcolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);

    TextureBuilder tb;
    setupTextureBuilder(device, tb);

    MemoryPool *pool = device->CreateMemoryPool(NULL, tb.GetStorageSize() + tb.GetStorageAlignment(), MemoryPoolType::GPU_ONLY);

    Texture* texMS = tb.CreateTextureFromPool(pool, 0);

    TextureHandle texHandle = device->GetTextureHandle(texMS->GetRegisteredTextureID(), sampler->GetRegisteredID());

    cmdBuf->SetRenderTargets(1, &texMS, NULL, NULL, NULL);
    cmdBuf->BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    cmdBuf->BindMultisampleState(&multisample);

    cmdBuf->ClearColor(0, fcolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);

    cmdBuf->SetColorReductionEnable(LWN_TRUE);

    CommandHandle handle = cmdBuf->EndRecording();
    // Doing first the queueCB.submit sets up the elw.
    queueCB.submit();

    queue->SubmitCommands(1, &handle);
    queue->Finish();

    CellIterator2D cell(lwrrentWindowWidth/gridSize, lwrrentWindowHeight/gridSize);
    const int inc = (m_thresholdRange / 32);
    for (int i = 0 ; i < 4; i++) {
        for (int t = 0; t < (m_thresholdRange - 1); t += inc) {
            int a = ((i & 1) == 0) ? ((m_thresholdRange - 1) - t) : 0;
            int b = ((i & 1) != 0) ? ((m_thresholdRange - 1)) - t : 0;

            cmdBuf->BeginRecording();
            cmdBuf->SetViewport(cell.x() * gridSize, cell.y() * gridSize, gridSize, gridSize);
            cmdBuf->SetScissor(cell.x() * gridSize, cell.y() * gridSize, gridSize, gridSize);
            cmdBuf->ClearColor(0, fcolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);

            cmdBuf->SetColorReductionThresholds(m_fmtClass, a, b);

            for (int y = 0; y < gridSize; y++) {
                for(int x = 0; x < gridSize; x++) {
                    cmdBuf->SetViewport(cell.x() * gridSize + x, cell.y() * gridSize + y, 1, 1);
                    cmdBuf->SetScissor(cell.x() * gridSize + x, cell.y() * gridSize + y, 1, 1);

                    cmdBuf->DrawArraysInstanced(DrawPrimitive::TRIANGLE_STRIP, 0, 4, 0, (i < 2 ) ? 2 : 3);
                }
            }
            CommandHandle handle = cmdBuf->EndRecording();

            queue->SubmitCommands(1, &handle);
            queue->Finish();

            cell++;
        }

        cell.nextRow();
    }

    queueCB.SetColorReductionEnable(LWN_FALSE);
    queueCB.SetColorReductionThresholds(m_fmtClass, 0, 0);
    queueCB.BindMultisampleState(&multisampleOff);

    queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_SHADER | BarrierBits::ILWALIDATE_TEXTURE);

    // rebind default framebuffer
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();

    queueCB.BindProgram(pgmReduce, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 4, 4);

    queueCB.submit();
    queue->Finish();

    cmdBuf->Free();
    delete [] control_space;
    texMS->Free();
    pool->Free();
    mp->Free();
}

#define COLOR_REDUCTION(cls, fmt, fmtClass) \
    OGTEST_CppTest(LWNcolorReductionTest, lwn_color_reduction_ ## cls,(fmt, fmtClass))

COLOR_REDUCTION(unorm8, Format::RGBA8, FormatClass::UNORM8);
COLOR_REDUCTION(unorm10, Format::RGB10A2, FormatClass::UNORM10);
COLOR_REDUCTION(unorm16, Format::RGBA16, FormatClass::UNORM16);
COLOR_REDUCTION(srgb8, Format::BGRA8_SRGB, FormatClass::SRGB8);
COLOR_REDUCTION(fp16, Format::RGBA16F, FormatClass::FP16);
COLOR_REDUCTION(fp11, Format::R11G11B10F, FormatClass::FP11);
