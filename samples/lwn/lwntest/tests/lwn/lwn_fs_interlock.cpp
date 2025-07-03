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

// Hack #define to run the same "depth test" fragment shader, but without
// interlocks.  That has significant atomicity issues and should be corrupted.
#define HACK_NO_INTERLOCK           0

class LWNFragmentShaderInterlockTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNFragmentShaderInterlockTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Simple fragment shader interlock test for LWN that uses fragment "
        "shader interlocks and image loads/stores to emulate depth testing.\n"
        "\n"
        "The test renders a set of red/green/blue/white rectangles with varying "
        "depth values rotated in a pinwheel pattern around the center of the "
        "viewport.  The test renders several copies of this pattern of various "
        "sizes.\n"
        "\n"
        "The left half of the image is rendered with colwentional depth testing "
        "and has the colors ilwerted (cyan/magenta/yellow/black).  The right "
        "half is rendered using fragment shader interlock.  Both image sets should "
        "be properly depth-tested without any visible corruption.";
    return sb.str();
}

int LWNFragmentShaderInterlockTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 13) && g_lwnDeviceCaps.supportsFragmentShaderInterlock;
}

void LWNFragmentShaderInterlockTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Boring pass-through vertex shader.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = color;\n"
        "}\n";

    // Boring pass-through fragment shader.
    FragmentShader fsBoring(440);
    fsBoring <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(1.0) - vec4(ocolor, 0.0);\n"
        "}\n";

    Program *colorPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(colorPgm, vs, fsBoring)) {
        LWNFailTest();
        return;
    }

    // The fragment shader does depth testing by reading from a R32F depth
    // image, comparing to interpolated depth, and storing depth/color only if
    // the depth test passes.
    FragmentShader fs(440);
    fs.addExtension(lwShaderExtension::LW_fragment_shader_interlock);
    fs <<
        "in vec3 ocolor;\n"
        "layout(rgba8, binding=0) coherent uniform image2D color;\n"
        "layout(r32f, binding=1) coherent uniform image2D depth;\n"
        "void main() {\n"
        "  ivec2 ixy = ivec2(gl_FragCoord.xy);\n"
#if !HACK_NO_INTERLOCK
        "  beginIlwocationInterlockLW();\n"
#endif
        "  float storedZ = imageLoad(depth, ixy).x;\n"
        "  if (gl_FragCoord.z < storedZ) {\n"
        "    imageStore(color, ixy, vec4(ocolor, 1.0));\n"
        "    imageStore(depth, ixy, vec4(gl_FragCoord.z, 0, 0, 0));\n"
        "  }\n"
#if !HACK_NO_INTERLOCK
        "  endIlwocationInterlockLW();\n"
#endif
        "}\n";

    Program *interlockPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(interlockPgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    // Interlock shaders need a lot of scratch memory (bug 1806472); make sure
    // we have enough so the debug layer doesn't bitch.
    const GLSLCoutput *glslcOutput = g_glslcHelper->GetGlslcOutput();
    size_t scratchSizeNeeded = g_glslcHelper->GetScratchMemoryRecommended(device, glslcOutput);
    if (scratchSizeNeeded > DEFAULT_SHADER_SCRATCH_MEMORY_SIZE) {
        MemoryPool *scratchPool = device->CreateMemoryPool(NULL, scratchSizeNeeded, MemoryPoolType::GPU_ONLY);
        queueCB.SetShaderScratchMemory(scratchPool, 0, scratchSizeNeeded);
        // scratchPool will be automatically cleaned up.
    }

    // Set up the vertex format and buffer to hold our procedurally-generated
    // quads.
    const int nQuads = 16;
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    MemoryPoolAllocator allocator(device, NULL, 4 * nQuads * sizeof(Vertex), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4 * nQuads, allocator, NULL);
    BufferAddress vboAddr = vbo->GetAddress();
    Vertex *vertices = (Vertex *) vbo->Map();

    // Coordinates of quad vertices prior to rotation.  The Z values varies
    // significantly from top to bottom, and slightly from left to right.
    dt::vec3 quad[4] = {
        dt::vec3(-0.25, -0.8, -0.2),
        dt::vec3(+0.25, -0.8, -0.0),
        dt::vec3(+0.25, +0.2, +0.8),
        dt::vec3(-0.25, +0.2, +0.6),
    };
    // Alternating colors for the quads.
    dt::vec3 colors[4] = {
        dt::vec3(1.0, 0.0, 0.0),
        dt::vec3(0.0, 1.0, 0.0),
        dt::vec3(0.0, 0.0, 1.0),
        dt::vec3(1.0, 1.0, 1.0),
    };
    Vertex *v = vertices;
    for (int i = 0; i < nQuads; i++) {
        double angle = (2 * 3.14159265358979323846 * i) / nQuads;
        for (int j = 0; j < 4; j++) {
            Vertex tv;
            tv.position[0] = cos(angle) * quad[j][0] - sin(angle) * quad[j][1];
            tv.position[1] = sin(angle) * quad[j][0] + cos(angle) * quad[j][1];
            tv.position[2] = quad[j][2];
            tv.color = colors[i % 4];
            *v++ = tv;
        }
    }

    // Set up an RGBA8 image for our color buffer and a R32F image for our
    // emulated depth buffer.
    MemoryPoolAllocator texAllocator(device, NULL, 32*1024*1024, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetSize2D(lwrrentWindowWidth, lwrrentWindowHeight);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetLevels(1);
    tb.SetFlags(TextureFlags::IMAGE);

    tb.SetFormat(Format::RGBA8);
    Texture *colorTex = texAllocator.allocTexture(&tb);
    ImageHandle colorImage = device->GetImageHandle(g_lwnTexIDPool->RegisterImage(colorTex));

    tb.SetFormat(Format::R32F);
    Texture *depthTex = texAllocator.allocTexture(&tb);
    ImageHandle depthImage = device->GetImageHandle(g_lwnTexIDPool->RegisterImage(depthTex));

    // Start by clearing our two textures via ROP, and throwing in a barrier
    // to ensure the clear finishes before we start rendering our quads.
    Texture *scratchTextures[2] = { colorTex, depthTex };
    queueCB.SetRenderTargets(2, scratchTextures, NULL, NULL, NULL);
    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);
    queueCB.ClearColor(1, 1.0, 0.0, 0.0, 0.0);      // clear the R32F "depth" to 1.0
    queueCB.SetRenderTargets(0, NULL, NULL, NULL, NULL);
    queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS | BarrierBits::ILWALIDATE_TEXTURE);

    // Now set up for rendering our quads.  We attach our color texture as a
    // render target also, so it can be used for the colwentional shader.  We
    // also use the window framebuffer's depth texture as the depth buffer.
    Texture *realDepthTexture = g_lwnWindowFramebuffer.getDepthStencilTexture();
    queueCB.SetRenderTargets(1, &colorTex, NULL, realDepthTexture, NULL);
    queueCB.ClearDepthStencil(1.0, LWN_TRUE, 0, 0);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, nQuads * 4 * sizeof(Vertex));
    queueCB.BindImage(ShaderStage::FRAGMENT, 0, colorImage);
    queueCB.BindImage(ShaderStage::FRAGMENT, 1, depthImage);

    // We render two passes, bouncing back and forth between interlock
    // disabled and enabled to make sure we switch properly.
    for (int pass = 0; pass < 2; pass++) {
        DepthStencilState dss;
        ChannelMaskState cms;
        dss.SetDefaults().SetDepthFunc(DepthFunc::LESS);
        cms.SetDefaults();
        for (int useInterlock = 0; useInterlock < 2; useInterlock++) {

            if (useInterlock) {
                // For interlock, use the interlock programs with color writes
                // and depth tests disabled.
                queueCB.BindProgram(interlockPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
                cms.SetChannelMask(0, LWN_FALSE, LWN_FALSE, LWN_FALSE, LWN_FALSE);
                dss.SetDepthTestEnable(LWN_FALSE);
            } else {
                // For color rendering, use the color program with color
                // writes and depth testing enabled.
                queueCB.BindProgram(colorPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
                cms.SetChannelMask(0, LWN_TRUE, LWN_TRUE, LWN_TRUE, LWN_TRUE);
                dss.SetDepthTestEnable(LWN_TRUE);
            }
            queueCB.BindDepthStencilState(&dss);
            queueCB.BindChannelMaskState(&cms);

            // Set the viewport on the left/right half taking up 2/3 of the
            // window height.
            int vpx = useInterlock * (lwrrentWindowWidth / 2);
            int vpy = 0;
            int vpw = lwrrentWindowWidth / 2;
            int vph = 2 * lwrrentWindowHeight / 3;
            int subpasses = 1;

            // On the second pass, use a smaller viewport above the first pass
            // viewport.  We will render primitives into several viewports of
            // decreasing size in "sub-passes" with no other state changes.
            if (pass) {
                vpy = vph;
                vpw /= 2;
                vph /= 2;
                subpasses = 4;
            }
            for (int subpass = 0; subpass < subpasses; subpass++) {
                queueCB.SetViewportScissor(vpx, vpy, vpw, vph);
                queueCB.DrawArrays(DrawPrimitive::QUADS, 0, 4 * nQuads);

                // Reduce the viewport size for the next subpass by half,
                // keeping it centered vertically.
                vpx += vpw;
                vpy += vph / 4;
                vpw /= 2;
                vph /= 2;
            }
        }
    }
    queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_TEXTURE);

    // Copy the contents of our color buffer texture into the window
    // framebuffer.
    Texture *acquiredTexture = g_lwnWindowFramebuffer.getAcquiredTexture();
    CopyRegion cr = { 0, 0, 0, lwrrentWindowWidth, lwrrentWindowHeight, 1 };
    queueCB.CopyTextureToTexture(colorTex, NULL, &cr, acquiredTexture, NULL, &cr, CopyFlags::NONE);

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNFragmentShaderInterlockTest, lwn_fs_interlock, );
