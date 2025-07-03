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

class LWNDownsampleTest
{
public:
    enum Variant {
        BasicTest,
        ManyDownsamples,
        ManyQueues
    };
private:
    enum MultisampleType {
        MultisampleType0,
        MultisampleType2,
        MultisampleType4,
        MultisampleType8,
        MultisampleTypeCount,
    };

    static int sampleCount(MultisampleType which)
    {
        switch (which) {
        default:                assert(0);
        case MultisampleType0:  return 0;
        case MultisampleType2:  return 2;
        case MultisampleType4:  return 4;
        case MultisampleType8:  return 8;
        }
    }

    Texture *generateTexture(MemoryPoolAllocator &allocator, Device *device, int w, int h, int samples,
                                    Format format, bool compressible = true) const;

    void basicTest(DeviceState *deviceState, MemoryPoolAllocator &texAllocator, Program *pgm, Program *displayPgm) const;
    void manyDownsamplesTest(DeviceState *deviceState, MemoryPoolAllocator &texAllocator) const;
    void manyQueuesTest(DeviceState *deviceState, MemoryPoolAllocator &texAllocator) const;

    Variant m_variant;
    bool    m_bTextureView;

    static const int maxLayers = 3;
    static const int srcLayer = 1, dstLayer = 2;

public:

    LWNDownsampleTest(Variant variant, bool bTextureView) : m_variant(variant), m_bTextureView(bTextureView) {}
    LWNTEST_CppMethods();
};

lwString LWNDownsampleTest::getDescription() const
{
    lwStringBuf sb;

    switch (m_variant) {
    default:
        assert(0);
    case BasicTest:
        sb <<
            "Basic texture downsample test.  We have four columns for 1x (non-MS), "
            "2x, 4x, and 8x multisampling.  Each cell draws a simple scene to the "
            "appropriate texture type, doing a downsample (if needed), and then "
            "magnifying the result on screen.  The bottom row uses the regular "
            "LWN (non-tiled downsample).  The middle row uses tiled downsamples "
            "with tiled caching disabled.  The top row uses tiled downsamples "
            "with tiled caching enabled.  The same triangle is drawn in all four "
            "columns, but the shade of red in the background varies by column.";
        break;
    case ManyDownsamples:
        sb <<
            "Texture downsample test exelwting a large number of back-to-back "
            "tiled downsamples.  We fill the screen with a bunch of 32x32 textures "
            "render to all of them, and then rip off a large number of conselwtive "
            "downsamples from multisample to single-sample textures.  We splat them "
            "all on screen.  The clear colors vary in a gradient pattern with red "
            "increasing from left to right, and green increasing from bottom to top.";
        break;
    case ManyQueues:
        sb <<
            "Texture downsample test exelwting a large number of back-to-back "
            "tiled downsamples on many different queues.  We fill the screen with a "
            "bunch of 32x32 textures render to all of them, and then rip off a "
            "large number of conselwtive downsamples from multisample to "
            "single-sample textures, alternating queues as we go.  We splat them "
            "all on screen.  The clear colors vary in a gradient pattern with red "
            "decreasing from left to right, and green decreasing from bottom to top.";
        break;
    }

    if (m_bTextureView) {
        sb << "This test uses texture views to downsample from "
              "one layer in a source array texture to another layer "
              "in a single sampled array texture and displays from "
              "this layer";
    }
    return sb.str();    
}

int LWNDownsampleTest::isSupported() const
{
    if (m_bTextureView) {
        return lwogCheckLWNAPIVersion(53, 101);
    }
    return lwogCheckLWNAPIVersion(52, 5);
}

// Allocate a texture with the specified properties.
lwn::Texture * LWNDownsampleTest::generateTexture(MemoryPoolAllocator &texAllocator, Device *device,
                                                  int w, int h, int samples,
                                                  Format format, bool compressible /* = true */) const
{
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetLevels(1);
    tb.SetFlags(compressible ? TextureFlags::COMPRESSIBLE : TextureFlags(0));
    tb.SetSamples(samples);
    tb.SetFormat(format);
    if (m_bTextureView) {
        tb.SetTarget(samples ? TextureTarget::TARGET_2D_MULTISAMPLE_ARRAY : TextureTarget::TARGET_2D_ARRAY);
        tb.SetSize3D(w, h, maxLayers);
    } else {
        tb.SetSize2D(w, h);
        tb.SetTarget(samples ? TextureTarget::TARGET_2D_MULTISAMPLE : TextureTarget::TARGET_2D);
    }
    return texAllocator.allocTexture(&tb);
}

void LWNDownsampleTest::basicTest(DeviceState *deviceState, MemoryPoolAllocator &texAllocator, Program *pgm, Program *displayPgm) const
{
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    static const int cellSize = 160;
    static const int cellMargin = 2;
    static const int cellTexSize = 16;      // render a small texture to show AA pattern

    // Program to draw an array texture mapped primitive with texture
    // coordinates generated from the screen-space position and
    // uniform to select the layer to draw.
    VertexShader displayVS(440);
    displayVS <<
        "layout(location=0) in vec3 position;\n"
        "out vec2 tc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  tc = position.xy * 0.5 + 0.5;\n"
        "}\n";

    FragmentShader displayFS(440);
    displayFS <<
        "layout(binding = 0) uniform sampler2DArray tex;\n"
        "layout(binding = 0) uniform Block {\n"
        "    int layer;\n"
        "};\n"
        "in vec2 tc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(tex, vec3(tc, layer));\n"
        "}\n";

    Program *displayLayerPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(displayLayerPgm, displayVS, displayFS)) {
        LWNFailTest();
        return;
    }

    // Set up a simple nearest-neighbor sampler for display.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler *pointSmp = sb.CreateSampler();

    // Set up basic textures for every multisample count (0, 2, 4, 8).
    Texture *tex[MultisampleTypeCount];
    TextureHandle texHandles[MultisampleTypeCount];
    for (int sidx = 0; sidx < MultisampleTypeCount; sidx++) {
        int samples = sampleCount(MultisampleType(sidx));
        tex[sidx] = generateTexture(texAllocator, device, cellTexSize, cellTexSize, samples, Format::RGBA8);
        texHandles[sidx] = device->GetTextureHandle(tex[sidx]->GetRegisteredTextureID(), pointSmp->GetRegisteredID());
    }

    // Set up multisample rendering state.
    MultisampleState mss;
    mss.SetDefaults().SetMultisampleEnable(LWN_TRUE);

    // Bind our 1x texture for display.
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandles[MultisampleType0]);

    MemoryPoolAllocator coherent_allocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    typedef struct {
        LWNint layer;
    } UniformBlock;

    UniformBlock uboData;
    uboData.layer = 0;
    Buffer *ubo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, &uboData, sizeof(uboData),
                                     BUFFER_ALIGN_UNIFORM_BIT, false);

    // Get a handle to be used for setting the buffer as a uniform buffer
    LWNbufferAddress uboAddr = ubo->GetAddress();
    UniformBlock* map = (UniformBlock*)ubo->Map();
    
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr, sizeof(uboData));

    TextureView srcView;
    TextureView dstView;
    if (m_bTextureView) {
        srcView.SetDefaults();
        srcView.SetTarget(TextureTarget::TARGET_RECTANGLE);
        srcView.SetLevels(0, 1);
        srcView.SetLayers(srcLayer, 1);

        dstView.SetDefaults();
        dstView.SetTarget(TextureTarget::TARGET_RECTANGLE);
        dstView.SetLevels(0, 1);
        dstView.SetLayers(dstLayer, 1);
    }

    for (int tiled = 0; tiled < 3; tiled++) {

        // tiled == 0:  Regular downsample
        // tiled == 1:  Tiled downsample, tiled caching disabled
        // tiled == 2:  Tiled downsample, tiled caching enabled
        if (tiled == 2) {
            queueCB.SetTiledCacheAction(TiledCacheAction::ENABLE);
        } else {
            queueCB.SetTiledCacheAction(TiledCacheAction::DISABLE);
        }

        // Loop over the multisample count columns.
        for (int sidx = 0; sidx < MultisampleTypeCount; sidx++) {

            // Render to the appropriate texture for the multisample type. We
            // pick a different clear color for each MS type.
            mss.SetSamples(sampleCount(MultisampleType(sidx)));
            queueCB.BindMultisampleState(&mss);
            queueCB.SetViewportScissor(0, 0, cellTexSize, cellTexSize);
            queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            if (m_bTextureView) {
                TextureView view;
                view.SetDefaults();
                view.SetTarget((sidx == MultisampleType0) ? TextureTarget::TARGET_2D: TextureTarget::TARGET_2D_MULTISAMPLE);
                view.SetLevels(0, 1);
                for (int l = 0; l < maxLayers; l++) {
                    view.SetLayers(l, 1);
                    TextureView* v = &view;
                    queueCB.SetRenderTargets(1, &tex[sidx], &v, NULL, NULL);
                    if (l != srcLayer) {
                        queueCB.ClearColor(0, 1.0, 20.f/255.f, 147.f/255.f, 0.0);
                    } else {
                        queueCB.ClearColor(0, 0.4 + 0.1 * sidx, 0.0, 0.0, 0.0);
                        queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
                    }
                }
            } else {
                queueCB.SetRenderTargets(1, &tex[sidx], NULL, NULL, NULL);
                queueCB.ClearColor(0, 0.4 + 0.1 * sidx, 0.0, 0.0, 0.0);
                queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
            }

            // For multisample columns, perform a downsample and discard of the
            // multisample texture.
            if (sidx != MultisampleType0) {
                map->layer = dstLayer;
                if (tiled) {
                    if(m_bTextureView) {
                        queueCB.TiledDownsampleTextureView(tex[sidx], &srcView, tex[MultisampleType0], &dstView);
                    } else {
                        queueCB.TiledDownsample(tex[sidx], tex[MultisampleType0]);
                    }
                } else {
                    if(m_bTextureView) {
                        queueCB.DownsampleTextureView(tex[sidx], &srcView, tex[MultisampleType0], &dstView);
                    } else {
                        queueCB.Downsample(tex[sidx], tex[MultisampleType0]);
                    }
                }
                queueCB.DiscardColor(0);
            } else {
                map->layer = srcLayer;
            }

            // Use the display program to render the 1x texture (either written
            // by the downsample) on-screen.
            g_lwnWindowFramebuffer.bind();
            mss.SetSamples(0);
            queueCB.BindMultisampleState(&mss);
            queueCB.BindProgram(m_bTextureView ? displayLayerPgm : displayPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            queueCB.SetViewportScissor(sidx * cellSize + cellMargin, tiled * cellSize + cellMargin,
                                       cellSize - 2 * cellMargin, cellSize - 2 * cellMargin);
            queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_TEXTURE);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 3, 3);
            if (m_bTextureView) {
                queueCB.submit();
                queue->Finish();
            }
        }
    }

    queueCB.SetTiledCacheAction(TiledCacheAction::DISABLE);
}

void LWNDownsampleTest::manyDownsamplesTest(DeviceState *deviceState, MemoryPoolAllocator &texAllocator) const
{
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

    // We fill the screen with a grid of unique 32x32 textures.
    const int texSize = 32;
    const int cellSize = 40;
    const int cellMargin = (cellSize - texSize) / 2;
    const int cellsX = 640 / cellSize;
    const int cellsY = 480 / cellSize;
    const int nTextures = cellsX * cellsY;

    MultisampleState mss;
    mss.SetDefaults().SetMultisampleEnable(LWN_TRUE);

    // Allocate both 1x and multisample textures for each grid cell, and render
    // to the multisample texture. We clear to a color that varies in a
    // gradient across the screen and draw a single triangle in the center of each.
    queueCB.SetViewportScissor(0, 0, texSize, texSize);
    Texture *texturesMS[nTextures];
    Texture *textures1X[nTextures];
    for (int i = 0; i < nTextures; i++) {
        int cy = i / cellsX;
        int samples = 2 << (cy % 3);  // 2, 4, or 8
        textures1X[i] = generateTexture(texAllocator, device, texSize, texSize, 0, Format::RGBA8);
        texturesMS[i] = generateTexture(texAllocator, device, texSize, texSize, samples, Format::RGBA8);

        float clearColor[4];
        clearColor[0] = float(i % cellsX) / (cellsX - 1);
        clearColor[1] = float(i / cellsX) / (cellsY - 1);
        clearColor[2] = 0.4;
        clearColor[3] = 1.0;

        mss.SetSamples(samples);
        queueCB.BindMultisampleState(&mss);
        if (m_bTextureView) {
                TextureView view;
                view.SetDefaults();
                view.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE);
                view.SetLevels(0, 1);
                for (int l = 0; l < maxLayers; l++) {
                    view.SetLayers(l, 1);
                    TextureView* v = &view;
                    queueCB.SetRenderTargets(1, &texturesMS[i], &v, NULL, NULL);
                    if (l != srcLayer) {
                        queueCB.ClearColor(0, 1.0, 20.f/255.f, 147.f/255.f, 0.0);
                    } else {
                        queueCB.ClearColor(0, clearColor, ClearColorMask::RGBA);
                    }
                    queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
                    queueCB.submit();
                } 
        } else {
            queueCB.SetRenderTargets(1, &texturesMS[i], NULL, NULL, NULL);
            queueCB.ClearColor(0, clearColor, ClearColorMask::RGBA);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
        }
    }

    TextureView srcView;
    TextureView dstView;
    if (m_bTextureView) {
        srcView.SetDefaults();
        srcView.SetTarget(TextureTarget::TARGET_RECTANGLE);
        srcView.SetLevels(0, 1);
        srcView.SetLayers(srcLayer, 1);

        dstView.SetDefaults();
        dstView.SetTarget(TextureTarget::TARGET_RECTANGLE);
        dstView.SetLevels(0, 1);
        dstView.SetLayers(dstLayer, 1);
    }

    // Now we rip through all the cells, downsampling each from the multisample
    // image to the equivalent 1x image. This tests our pacing to be sure we
    // can handle a large number of tiled downsample operations in a row.
    TextureView* pDstView = NULL;
    for (int i = 0; i < nTextures; i++) {
        if(m_bTextureView) {
            queueCB.TiledDownsampleTextureView(texturesMS[i], &srcView, textures1X[i], &dstView);
            pDstView = &dstView;
        } else {
            queueCB.TiledDownsample(texturesMS[i], textures1X[i]);
        }
    }

    // Finally, copy each of the 1x images on screen.
    Texture *windowTex = g_lwnWindowFramebuffer.getAcquiredTexture();
    for (int i = 0; i < nTextures; i++) {
        int cx = i % cellsX;
        int cy = i / cellsX;
        CopyRegion srcRegion = { 0, 0, 0, texSize, texSize, 1 };
        CopyRegion dstRegion = {
            cx * cellSize + cellMargin, cy * cellSize + cellMargin, 0,
            texSize, texSize, 1,
        };
        queueCB.CopyTextureToTexture(textures1X[i], pDstView, &srcRegion,
                                     windowTex, NULL, &dstRegion,
                                     CopyFlags::NONE);
    }
}

void LWNDownsampleTest::manyQueuesTest(DeviceState *deviceState, MemoryPoolAllocator &texAllocator) const
{
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // We fill the screen with a grid of unique 32x32 textures.
    const int texSize = 32;
    const int cellSize = 40;
    const int cellMargin = (cellSize - texSize) / 2;
    const int cellsX = 640 / cellSize;
    const int cellsY = 480 / cellSize;
    const int nTextures = cellsX * cellsY;

    // We create a relatively large number of queues for this test.
#if !defined(LW_HOS)
    const int nQueues = 32;
#else
    // !!! lwntest on HOS appears to run out of transfer memory quickly if you
    // create a lot of queues. 4 works; we should figure out how to increase.
    const int nQueues = 4;
#endif

    MultisampleState mss;
    mss.SetDefaults().SetMultisampleEnable(LWN_TRUE);

    // Allocate both 1x and multisample textures for each grid cell, and render
    // to the multisample texture. We clear to a color that varies in a
    // gradient across the screen and draw a single triangle in the center of each.
    queueCB.SetViewportScissor(0, 0, texSize, texSize);
    Texture *texturesMS[nTextures];
    Texture *textures1X[nTextures];
    for (int i = 0; i < nTextures; i++) {
        int cy = i / cellsX;
        int samples = 2 << (cy % 3);  // 2, 4, or 8
        textures1X[i] = generateTexture(texAllocator, device, texSize, texSize, 0, Format::RGBA8);
        texturesMS[i] = generateTexture(texAllocator, device, texSize, texSize, samples, Format::RGBA8);

        float clearColor[4];
        clearColor[0] = 1.0 - float(i % cellsX) / (cellsX - 1);
        clearColor[1] = 1.0 - float(i / cellsX) / (cellsY - 1);
        clearColor[2] = 0.4;
        clearColor[3] = 1.0;

        mss.SetSamples(samples);
        queueCB.BindMultisampleState(&mss);
        if (m_bTextureView) {
                TextureView view;
                view.SetDefaults();
                view.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE);
                view.SetLevels(0, 1);
                for (int l = 0; l < maxLayers; l++) {
                    view.SetLayers(l, 1);
                    TextureView* v = &view;
                    queueCB.SetRenderTargets(1, &texturesMS[i], &v, NULL, NULL);
                    if (l != srcLayer) {
                        queueCB.ClearColor(0, 1.0, 20.f/255.f, 147.f/255.f, 0.0);
                    } else {
                        queueCB.ClearColor(0, clearColor, ClearColorMask::RGBA);
                    }
                    queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
                    queueCB.submit();
                } 
        } else {
            queueCB.SetRenderTargets(1, &texturesMS[i], NULL, NULL, NULL);
            queueCB.ClearColor(0, clearColor, ClearColorMask::RGBA);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
        }
    }

#if 1
    // !!! For some reason, doing a Finish here makes this work. Need to work
    // out timing.
    queueCB.submit();
    queue->Finish();
#endif

    // Set up a command buffer object to collect commands, using our device
    // state's command memory manager.
    CommandBuffer cb;
    cb.Initialize(device);
    g_lwnCommandMem.populateCommandBuffer(&cb, CommandBufferMemoryManager::Coherent);

    // Set up a collection of queues and associated sync objects.
    Queue queues[nQueues];
    Sync syncs[nQueues];
    QueueBuilder qb;
    qb.SetDevice(device).SetDefaults();
    for (int i = 0; i < nQueues; i++) {
        queues[i].Initialize(&qb);
        syncs[i].Initialize(device);

        // Have each queue use its sync object to wait on the setup of the
        // multisample textures from the main queue.
        queueCB.FenceSync(&syncs[i], SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        queues[i].WaitSync(&syncs[i]);

        // Tiled downsamples require an initialized texture pool for any queue
        // doing a downsample. Bind the texture pool from the default device.
        TexIDPool *texIDPool = DeviceState::GetActive()->getTexIDPool();
        cb.BeginRecording();
        texIDPool->Bind(&cb);
        CommandHandle handle = cb.EndRecording();
        queues[i].SubmitCommands(1, &handle);
    }

    queueCB.submit();

    TextureView srcView;
    TextureView dstView;
    if (m_bTextureView) {
        srcView.SetDefaults();
        srcView.SetTarget(TextureTarget::TARGET_RECTANGLE);
        srcView.SetLevels(0, 1);
        srcView.SetLayers(srcLayer, 1);

        dstView.SetDefaults();
        dstView.SetTarget(TextureTarget::TARGET_RECTANGLE);
        dstView.SetLevels(0, 1);
        dstView.SetLayers(dstLayer, 1);
    }

    // Now we rip through all the cells, downsampling each from the multisample
    // image to the equivalent 1x image. We alternate through our array of
    // queues. This tests our pacing to be sure we can handle a large number of
    // tiled downsample operations in a row, and that we can use many different
    // queues to submit.
    TextureView* pDstView = NULL;
    for (int i = 0; i < nTextures; i++) {
        Queue *queue = &queues[i % nQueues];
        cb.BeginRecording();
        if (m_bTextureView) {
            cb.TiledDownsampleTextureView(texturesMS[i], &srcView, textures1X[i], &dstView);
            pDstView = &dstView;
        } else {
            cb.TiledDownsample(texturesMS[i], textures1X[i]);
        }
        CommandHandle handle = cb.EndRecording();
        queue->SubmitCommands(1, &handle);
    }

    // Prepare each of our temporary queues for teardown by signaling their
    // sync object and performing a finish. Have the main queue wait on all of these.
    for (int i = 0; i < nQueues; i++) {
        queues[i].FenceSync(&syncs[i], SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        queues[i].Finish();
    }

    // Finally, copy each of the 1x images on screen.
    Texture *windowTex = g_lwnWindowFramebuffer.getAcquiredTexture();
    for (int i = 0; i < nTextures; i++) {
        int cx = i % cellsX;
        int cy = i / cellsX;
        CopyRegion srcRegion = { 0, 0, 0, texSize, texSize, 1 };
        CopyRegion dstRegion = {
            cx * cellSize + cellMargin, cy * cellSize + cellMargin, 0,
            texSize, texSize, 1,
        };
        queueCB.CopyTextureToTexture(textures1X[i], pDstView, &srcRegion,
                                     windowTex, NULL, &dstRegion,
                                     CopyFlags::NONE);
    }

    // Tear down the command buffer object and all of our queues.
    cb.Finalize();
    for (int i = 0; i < nQueues; i++) {
        syncs[i].Finalize();
        queues[i].Finalize();
    }
    // Submit commands in the main queue's command buffer, which
    // include references to temporary sync objects.
    queueCB.submit();
}

void LWNDownsampleTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Basic program to draw a smooth-shaded primitive.
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

    Program *pgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    // Basic program to draw a texture mapped primitive with texture
    // coordinates generated from the screen-space position.
    VertexShader displayVS(440);
    displayVS <<
        "layout(location=0) in vec3 position;\n"
        "out vec2 tc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  tc = position.xy * 0.5 + 0.5;\n"
        "}\n";

    FragmentShader displayFS(440);
    displayFS <<
        "layout(binding = 0) uniform sampler2D tex;\n"
        "in vec2 tc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(tex, tc);\n"
        "}\n";

    Program *displayPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(displayPgm, displayVS, displayFS)) {
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {

        // Data for a simple RGB triangle.
        { dt::vec3(-0.375, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-0.375, +0.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+0.375, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },

        // Data for a "display" triangle that covers the whole viewport.
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec3(1.0, 1.0, 1.0) },
        { dt::vec3(-1.0, +3.0, 0.0), dt::vec3(1.0, 1.0, 1.0) },
        { dt::vec3(+3.0, -1.0, 0.0), dt::vec3(1.0, 1.0, 1.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator vboAllocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 6, vboAllocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Set up a memory pool allocator with a relatively large amount of memory
    // for texture allocations.
    MemoryPoolAllocator texAllocator(device, NULL, 128 * 1024 * 1024, (LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT |
                                                                       LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT |
                                                                       LWN_MEMORY_POOL_FLAGS_COMPRESSIBLE_BIT));

    // Set up some basics, clearing the screen and binding basic vertex and
    // program state.
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    // Perform the specified sub-test.
    switch (m_variant) {
    case BasicTest:
        basicTest(deviceState, texAllocator, pgm, displayPgm);
        break;
    case ManyDownsamples:
        manyDownsamplesTest(deviceState, texAllocator);
        break;
    case ManyQueues:
        manyQueuesTest(deviceState, texAllocator);
        break;
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

#define DOWNSAMPLE(__EXTENSION__, __SUBTEST__)                                              \
    OGTEST_CppTest(LWNDownsampleTest, lwn_downsample##__EXTENSION__, (__SUBTEST__, false)); \
    OGTEST_CppTest(LWNDownsampleTest, lwn_downsample##__EXTENSION__##_view, (__SUBTEST__, true));  

DOWNSAMPLE(, LWNDownsampleTest::BasicTest)
DOWNSAMPLE(_many, LWNDownsampleTest::ManyDownsamples)
DOWNSAMPLE(_queues, LWNDownsampleTest::ManyQueues)
