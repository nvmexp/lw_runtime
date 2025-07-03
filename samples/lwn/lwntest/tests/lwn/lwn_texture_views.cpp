/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// lwn_texture_views.cpp
//
// Tests covering texture views in LWN
// lwn_texture_views - Tests rendering *with* different combinations of layers and levels
// lwn_texture_view_render_targets - Tests rendering *to* different combinations of layers and levels
// lwn_texture_view_multiple_render_targets - Tests rendering and reading from different render targets (MRT)

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include <vector>

static const int BASE_TEX_SIZE = 16;
static const int NUM_LEVELS = 3;
static const int NUM_LAYERS = 3;

#define DEBUG_PRINT(x) do { \
    printf x; \
    fflush(stdout); \
} while (0)

using namespace lwn;
using namespace lwn::dt;

// Helper class for rendering all layers and levels of a texture to viewport extents
class TextureVisualizer {
public:
    ~TextureVisualizer();

    static TextureVisualizer* create(Device *device, bool multiLayer = true);
    void bind(QueueCommandBuffer &queueCB);
    void draw(Device *device, QueueCommandBuffer &queueCB, Texture *texture, TextureView *view);

private:
    TextureVisualizer() {};

    struct Vertex {
        dt::vec2 position;
    };

    VertexArrayState mVertexState;
    Buffer *mVbo;
    Program *mProgram;
    Sampler *mSampler;
    LWNuint mSamplerID;
    MemoryPoolAllocator* mAllocator;
};

TextureVisualizer* TextureVisualizer::create(Device *device, bool multiLayer /* =true */)
{
    TextureVisualizer* pTextureVisualizer = new TextureVisualizer();
    if (!pTextureVisualizer)
        return NULL;

    SamplerBuilder sb;
    VertexShader   vs(440);
    FragmentShader fs(440);

    VertexStream  vertexStream(sizeof(Vertex));

    static const Vertex vertexData[] = {
        { vec2(0.0, 0.0) },
        { vec2(1.0, 0.0) },
        { vec2(1.0, 1.0) },
        { vec2(0.0, 1.0) },
    };

    // 64k is more than for this test but anything below 64k always allocates one page in the sparse pool
    const LWNsizeiptr poolSize = 0x10000UL;

    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    pTextureVisualizer->mVertexState = vertexStream.CreateVertexArrayState();

    pTextureVisualizer->mAllocator = new MemoryPoolAllocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    if (!pTextureVisualizer->mAllocator)
        goto fail;

    pTextureVisualizer->mVbo = vertexStream.AllocateVertexBuffer(device, 4, *pTextureVisualizer->mAllocator, vertexData);
    if (!pTextureVisualizer->mVbo)
        goto fail;

    vs <<
        "layout(location=0) in vec2 position;\n"
        "out vec2 vPos;\n"

        "void main() {\n"

        "gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);\n"

        "vPos = position;\n"
        "}\n";

    if (multiLayer) {
        fs <<
            "in vec2 vPos;\n"
            "layout (binding=0) uniform sampler2DArray tex;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"

            "  float numLayers = float(textureSize(tex, 0).z);\n"
            "  float numLevels = float(textureQueryLevels(tex));\n"

            "  fcolor = textureLod(tex, vec3(fract(vPos * vec2(numLayers, numLevels)),\n"
            "                                floor(vPos.x * numLayers)),\n"
            "                      floor(vPos.y * numLevels));\n"
            "}\n";
    }
    else {
        fs <<
            "in vec2 vPos;\n"
            "layout (binding=0) uniform sampler2D tex;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  fcolor = texture(tex, vPos.xy);\n"
            "}\n";
    }

    pTextureVisualizer->mProgram = device->CreateProgram();
    if (!pTextureVisualizer->mProgram)
        goto fail;

    if (!g_glslcHelper->CompileAndSetShaders(pTextureVisualizer->mProgram, vs, fs)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        goto fail;
    }

    sb.SetDevice(device).SetDefaults()
      .SetMinMagFilter(MinFilter::NEAREST_MIPMAP_NEAREST, MagFilter::NEAREST);
    pTextureVisualizer->mSampler = sb.CreateSampler();
    if (!pTextureVisualizer->mSampler)
        goto fail;

    pTextureVisualizer->mSamplerID = pTextureVisualizer->mSampler->GetRegisteredID();

    return pTextureVisualizer;

fail:

    DEBUG_PRINT(("Error during TextureVisualizer::create - unable to create class.\n"));
    LWNFailTest();

    delete pTextureVisualizer;
    return NULL;
}

TextureVisualizer::~TextureVisualizer()
{
    delete mAllocator;
}

void TextureVisualizer::bind(QueueCommandBuffer &queueCB)
{
    queueCB.BindVertexArrayState(mVertexState);
    queueCB.BindVertexBuffer(0, mVbo->GetAddress(), 4 * sizeof(Vertex));
    queueCB.BindProgram(mProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
}

void TextureVisualizer::draw(Device *device, QueueCommandBuffer &queueCB, Texture *texture, TextureView *view)
{
    LWNuint texID = g_lwnTexIDPool->Register(texture, view);
    TextureHandle handle = device->GetTextureHandle(texID, mSamplerID);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, handle);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
}

// lwn_texture_views ==========================================================

class LWNTextureViewsTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNTextureViewsTest::getDescription() const
{
    return "Test for LWN layer and level selection. Create a 3-layer, 3-level 2D array texture "
           "inside a memory pool, and initialize all combinations of layers and levels with data:\n"
           "* Levels: 0) Vertical stripes, 1) Horizontal stripes, 2) Checkerboard\n"
           "* Layers: 0) Red/white, 1) Green/white, 2) Blue/white\n"
           "\n"
           "In the test output, display an array of texture view contents. Each column represents "
           "a pair of values passed to TextureView::SetLayers, and each row "
           "represents a pair of values passed to TextureView::SetLevels. Each "
           "cell displays all layers and levels represented within the corresponding view.\n"
           "\n"
           "The pairs of values used for both layers (left to right) and levels (bottom to top) are:\n"
           " (0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (2,0), (2,1)";
}

int LWNTextureViewsTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(28, 0);
}

static MemoryPool *CreateTexturePool(Device *device, Queue *queue, QueueCommandBuffer &queueCB)
{
    int imageSize = BASE_TEX_SIZE * BASE_TEX_SIZE * 4;
    MemoryPool *pool = device->CreateMemoryPool(NULL, imageSize, MemoryPoolType::CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    Buffer *textureBuffer = bb.CreateBufferFromPool(pool, 0, imageSize);
    BufferAddress textureBufferAddr = textureBuffer->GetAddress();
    u8lwec4* textureBufferPtr = static_cast<u8lwec4*>(textureBuffer->Map());
    if (!textureBufferPtr) {
        DEBUG_PRINT(("Could not map texture buffer\n"));
        LWNFailTest();
        return NULL;
    }

    TextureBuilder builder;
    builder.SetDevice(device).SetDefaults()
           .SetTarget(TextureTarget::TARGET_2D_ARRAY)
           .SetSize3D(BASE_TEX_SIZE, BASE_TEX_SIZE, NUM_LAYERS)
           .SetFormat(Format::RGBX8)
           .SetLevels(NUM_LEVELS);

    pool = device->CreateMemoryPool(NULL, builder.GetStorageSize(), MemoryPoolType::GPU_ONLY);
    if (!pool)
        return NULL;

    Texture *tex = builder.CreateTextureFromPool(pool, 0);
    if (!tex)
        return NULL;

    TextureView levelView;
    levelView.SetDefaults();
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        u8lwec4 color(0.0, 0.0, 0.0, 1.0);
        color[layer] = 1.0;
        for (int level = 0; level < NUM_LEVELS; ++level) {
            levelView.SetLevels(level, 1);
            int size = BASE_TEX_SIZE >> level;
            u8lwec4* ptr = textureBufferPtr;
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    int parity = 0;
                    switch (level) {
                    case 0:
                        parity = x & 0x1;
                        break;
                    case 1:
                        parity = y & 0x1;
                        break;
                    case 2:
                        parity = (x ^ y) & 0x1;
                        break;
                    default:
                        assert(0);
                    }
                    *ptr++ = parity ? color : u8lwec4(1.0, 1.0, 1.0, 1.0);
                }
            }
            CopyRegion copyRegion = { 0, 0, layer, size, size, 1 };
            queueCB.CopyBufferToTexture(textureBufferAddr, tex, &levelView, &copyRegion, CopyFlags::NONE);
            queueCB.submit();
            queue->Finish();
        }
    }

    return pool;
}

void LWNTextureViewsTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    MemoryPool *pool = CreateTexturePool(device, queue, queueCB);
    Texture *tex;

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);

    int numRows = NUM_LEVELS * (NUM_LEVELS + 3) / 2;
    int numCols = NUM_LAYERS * (NUM_LAYERS + 3) / 2;
    int row = 0;

    TextureVisualizer* pViewer = TextureVisualizer::create(device);
    if (!pViewer)
        goto fail;

    pViewer->bind(queueCB);

    TextureBuilder texBuilder;
    texBuilder.SetDevice(device)
              .SetDefaults()
              .SetTarget(TextureTarget::TARGET_2D_ARRAY)
              .SetSize3D(BASE_TEX_SIZE, BASE_TEX_SIZE, NUM_LAYERS)
              .SetFormat(Format::RGBX8)
              .SetLevels(NUM_LEVELS);

    tex = texBuilder.CreateTextureFromPool(pool, 0);
    if (!tex) {
        DEBUG_PRINT(("Error in CreateTextureFromPool from LWNTextureViewsTest::doGraphics()"));
        goto fail;
    }

    for (int baseLevel = 0; baseLevel < NUM_LEVELS; ++baseLevel) {
        for (int nLevels = 0; nLevels <= NUM_LEVELS - baseLevel; ++nLevels) {
            int col = 0;
            for (int baseLayer = 0; baseLayer < NUM_LAYERS; ++baseLayer) {
                for (int nLayers = 0; nLayers <= NUM_LAYERS - baseLayer; ++nLayers) {
                    int x0 = lwrrentWindowWidth * col / numCols + 1;
                    int x1 = lwrrentWindowWidth * (col + 1) / numCols - 1;
                    int y0 = lwrrentWindowHeight * row / numRows + 1;
                    int y1 = lwrrentWindowHeight * (row + 1) / numRows - 1;
                    queueCB.SetViewport(x0, y0, x1 - x0, y1 - y0);
                    TextureView texView;
                    texView.SetDefaults();
                    texView.SetLevels(baseLevel, nLevels);
                    texView.SetLayers(baseLayer, nLayers);
                    pViewer->draw(device, queueCB, tex, &texView);
                    col += 1;
                }
            }
            row += 1;
        }
    }

    queueCB.submit();
    queue->Finish();
    delete pViewer;
    return;

fail:
    LWNFailTest();
    queueCB.submit();
    queue->Finish();
    delete pViewer;
}

OGTEST_CppTest(LWNTextureViewsTest, lwn_texture_views, );

// lwn_texture_views_single_layer ==========================================================

class LWNTextureViewsSingleLayerTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNTextureViewsSingleLayerTest::getDescription() const
{
    return "Test for selecting a single layer and level of a texture 2D array.\n"
           "Create a 3 - layer, 3 - level 2D array texture and use texture views\n"
            "to specify a single layer and level of the array which is used as\n"
            "2D texture for rendering.\n"
            "Levels: 0) Vertical stripes, 1) Horizontal stripes, 2) Checkerboard\n"
            "Layers: 0) Red / white, 1) Green / white, 2) Blue / white\n"
            "\n"
            "The test output is a 3x3 image. Each row specifies a level and each\n"
            "column a layer of the texture array. The first row shows the 3 layers\n"
            "using the highest mip-map level.\n";
}

int LWNTextureViewsSingleLayerTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(28, 0);
}

void LWNTextureViewsSingleLayerTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    TextureBuilder texBuilder;
    texBuilder.SetDevice(device).SetDefaults();
    MemoryPool *pool = CreateTexturePool(device, queue, queueCB);
    Texture *tex;

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);

    int row = 0;

    TextureVisualizer* pViewer = TextureVisualizer::create(device, false);
    if (!pViewer)
        goto fail;

    pViewer->bind(queueCB);

    texBuilder.SetDefaults()
        .SetTarget(TextureTarget::TARGET_2D_ARRAY)
        .SetSize3D(BASE_TEX_SIZE, BASE_TEX_SIZE, NUM_LAYERS)
        .SetFormat(Format::RGBX8)
        .SetLevels(NUM_LEVELS);

    tex = texBuilder.CreateTextureFromPool(pool, 0);
    if (!tex) {
        DEBUG_PRINT(("Error in CreateTextureFromPool from LWNTextureViewsTest::doGraphics()"));
        goto fail;
    }

    for (int nLevels = 0; nLevels < NUM_LEVELS; ++nLevels) {
        int col = 0;
        for (int nLayers = 0; nLayers < NUM_LAYERS; ++nLayers) {
            int x0 = lwrrentWindowWidth * col / NUM_LAYERS + 1;
            int x1 = lwrrentWindowWidth * (col + 1) / NUM_LAYERS - 1;
            int y0 = lwrrentWindowHeight * row / NUM_LEVELS + 1;
            int y1 = lwrrentWindowHeight * (row + 1) / NUM_LEVELS - 1;
            queueCB.SetViewport(x0, y0, x1 - x0, y1 - y0);
            TextureView texView;
            texView.SetDefaults();
            texView.SetTarget(TextureTarget::TARGET_2D);
            texView.SetLevels(nLevels, 1);
            texView.SetLayers(nLayers, 1);
            pViewer->draw(device, queueCB, tex, &texView);
            col += 1;
        }
        row += 1;
    }

    queueCB.submit();
    queue->Finish();
    delete pViewer;
    return;

fail:
    LWNFailTest();
    queueCB.submit();
    queue->Finish();
    delete pViewer;
}

OGTEST_CppTest(LWNTextureViewsSingleLayerTest, lwn_texture_views_single_layer, );

// lwn_texture_view_render_targets =============================================

class LWNTextureViewRenderTargetsTest
{
public:
    LWNTEST_CppMethods();

private:
    static void drawImage(QueueCommandBuffer &queueCB, Texture *texture, int level, int layer);
};

lwString LWNTextureViewRenderTargetsTest::getDescription() const
{
    return "Test for rendering to particular layers and levels of a texture by using "
           "texture views. Create a 3-layer, 3-level 2D texture. For each layer/level, "
           "set the scissor larger than the image size and clear to a blue/green color. "
           "Then set the scissor one pixel smaller than the image on all sides and clear "
           "to white. Finally, blit all layers and levels to the framebuffer. There "
           "should be no sign of misalignment, overflow, or underflow. Repeat all this, "
           "but render to each target in reverse order to detect any problems that might "
           "have been masked by subsequent targets.";
}

int LWNTextureViewRenderTargetsTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(12, 0);
}

void LWNTextureViewRenderTargetsTest::drawImage(QueueCommandBuffer &queueCB, Texture *texture, int level, int layer)
{
    float green = (level + 1.0f) / NUM_LEVELS;
    float blue = (layer + 1.0f) / NUM_LAYERS;
    TextureView textureView;
    TextureView *pTextureView = &textureView;
    textureView.SetDefaults().SetLevels(level, 1).SetLayers(layer, 1);
    queueCB.SetRenderTargets(1, &texture, &pTextureView, NULL, NULL);
    int levelSize = BASE_TEX_SIZE >> level;

    queueCB.SetScissor(0, 0, levelSize, levelSize);
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);

    queueCB.SetScissor(-levelSize, -levelSize, 3 * levelSize, 3 * levelSize);
    queueCB.ClearColor(0, 0.0, green, blue, 1.0);
    queueCB.SetScissor(1, 1, levelSize - 2, levelSize - 2);
    queueCB.ClearColor(0, 1.0, 1.0, 1.0, 1.0);
}

void LWNTextureViewRenderTargetsTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    TextureBuilder builder;
    builder.SetDevice(device).SetDefaults()
           .SetFlags(TextureFlags::COMPRESSIBLE)
           .SetTarget(TextureTarget::TARGET_2D_ARRAY)
           .SetSize3D(BASE_TEX_SIZE, BASE_TEX_SIZE, NUM_LAYERS)
           .SetFormat(Format::RGBX8)
           .SetLevels(NUM_LEVELS);

    MemoryPoolAllocator allocator(device, NULL, builder.GetStorageSize(), LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    Texture *multiImage = allocator.allocTexture(&builder);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);
    queueCB.submit();

    builder.SetFlags(TextureFlags::COMPRESSIBLE);

    for (int level = 0; level < NUM_LEVELS; ++level) {
        for (int layer = 0; layer < NUM_LAYERS; ++layer) {
            drawImage(queueCB, multiImage, level, layer);
            queueCB.submit();
        }
    }
    queue->Finish();
    g_lwnWindowFramebuffer.bind();

    queueCB.SetViewportScissor(1, 1, lwrrentWindowWidth / 2 - 2, lwrrentWindowHeight - 2);

    TextureVisualizer* pViewer = TextureVisualizer::create(device);
    if (!pViewer)
        goto cleanup;

    pViewer->bind(queueCB);
    pViewer->draw(device, queueCB, multiImage, NULL);

    queueCB.submit();

    for (int level = NUM_LEVELS - 1; level >= 0; --level) {
        for (int layer = NUM_LAYERS - 1; layer >= 0; --layer) {
            drawImage(queueCB, multiImage, level, layer);
            queueCB.submit();
        }
    }
    queue->Finish();

    g_lwnWindowFramebuffer.bind();

    queueCB.SetViewportScissor(lwrrentWindowWidth / 2 + 1, 1, lwrrentWindowWidth / 2 - 2, lwrrentWindowHeight - 2);

    pViewer->bind(queueCB);
    pViewer->draw(device, queueCB, multiImage, NULL);

cleanup:
    queueCB.submit();
    queue->Finish();

    delete pViewer;
}

OGTEST_CppTest(LWNTextureViewRenderTargetsTest, lwn_texture_view_render_targets, );

// lwn_texture_view_multiple_render_targets =============================================

class MRTRenderer {
public:
    ~MRTRenderer();

    static MRTRenderer* create(Device *device, Queue *queue, QueueCommandBuffer &queueCB, bool isMRTDisabled);

    void bind(QueueCommandBuffer& queueCB);
    void draw(QueueCommandBuffer& queueCB);

    Texture *getColor(int index) const {
        assert(index < mNumColors);
        return mColors[index];
    }

    int getNumColors() const {
        return mNumColors;
    }

    struct UniformBlock {
        dt::vec4 colors[8];
        unsigned int mask;
    };

private:
    MRTRenderer() {};

    struct Vertex {
        dt::vec2 position;
    };

    VertexArrayState mVertexState;
    Buffer *mVbo;
    Buffer *mUbo;
    Program *mProgram;

    ChannelMaskState mChannelMaskState;
    ChannelMaskState mDefaultChannelMaskState;

    MemoryPoolAllocator* mAllocator;
    MemoryPoolAllocator* mRTAllocator;

    int mNumColors;
    Texture** mColors;
};

MRTRenderer::~MRTRenderer()
{
    for (int i = 0; i< mNumColors; i++) {
        mRTAllocator->freeTexture(mColors[i]);
    }

    delete [] mColors;
    delete mAllocator;
    delete mRTAllocator;
}

MRTRenderer* MRTRenderer::create(Device *device, Queue *queue, QueueCommandBuffer &queueCB, bool isMRTDisabled)
{
    MRTRenderer* pMRTRenderer = new MRTRenderer();
    if (!pMRTRenderer)
        return NULL;

    VertexShader vs(440);
    FragmentShader fs(440);

    TextureBuilder builder;
    builder.SetDevice(device)
        .SetDefaults()
        .SetFlags(TextureFlags::COMPRESSIBLE)
        .SetTarget(TextureTarget::TARGET_2D_ARRAY)
        .SetSize3D(BASE_TEX_SIZE, BASE_TEX_SIZE, 1)
        .SetFormat(Format::RGBX8)
        .SetLevels(1);

    UniformBlock uboData;

    static const Vertex vertexData[] = {
        { vec2(0.0, 0.0) },
        { vec2(0.0, 0.0) },
        { vec2(1.0, 1.0) },
        { vec2(0.0, 1.0) },
    };

    const LWNsizeiptr poolSize = 0x10000UL;

    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    pMRTRenderer->mVertexState = vertexStream.CreateVertexArrayState();

    pMRTRenderer->mAllocator = new MemoryPoolAllocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    if (!pMRTRenderer->mAllocator)
        goto fail;

    pMRTRenderer->mVbo = vertexStream.AllocateVertexBuffer(device, 4, *pMRTRenderer->mAllocator, vertexData);
    if (!pMRTRenderer->mVbo)
        goto fail;

    vs <<
        "void main() {\n"

        "  vec2 position; "
        "  if (gl_VertexID == 0) position = vec2(-1.0, -1.0);"
        "  if (gl_VertexID == 1) position = vec2(1.0, -1.0);"
        "  if (gl_VertexID == 2) position = vec2(1.0, 1.0);"
        "  if (gl_VertexID == 3) position = vec2(-1.0, 1.0);"

        "  gl_Position = vec4(position * 0.25, 0.0, 1.0);\n"
        "}\n";

    fs <<
        "layout(binding = 0) uniform Block {\n"
        "    vec4 colors[8];\n"
        "    uint mask;\n"
        "};\n"

        "layout(location = 0) out vec4 color0;\n"
        "layout(location = 1) out vec4 color1;\n"
        "layout(location = 2) out vec4 color2;\n"
        "layout(location = 3) out vec4 color3;\n"
        "layout(location = 4) out vec4 color4;\n"
        "layout(location = 5) out vec4 color5;\n"
        "layout(location = 6) out vec4 color6;\n"
        "layout(location = 7) out vec4 color7;\n";

    fs << "void main() {\n"
          "  if ((mask & 0x01) != 0) color0 = colors[0]; \n";

    if (!isMRTDisabled) {
        fs << "  if ((mask & 0x02) != 0) color1 = colors[1]; \n"
              "  if ((mask & 0x04) != 0) color2 = colors[2]; \n"
              "  if ((mask & 0x08) != 0) color3 = colors[3]; \n"
              "  if ((mask & 0x10) != 0) color4 = colors[4]; \n"
              "  if ((mask & 0x20) != 0) color5 = colors[5]; \n"
              "  if ((mask & 0x40) != 0) color6 = colors[6]; \n"
              "  if ((mask & 0x80) != 0) color7 = colors[7]; \n";
    }

    fs <<     "}\n";

    pMRTRenderer->mProgram = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pMRTRenderer->mProgram, vs, fs)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        LWNFailTest();
        goto fail;
    }

    device->GetInteger(DeviceInfo::COLOR_BUFFER_BINDINGS, &pMRTRenderer->mNumColors);

    pMRTRenderer->mRTAllocator = new MemoryPoolAllocator(device, NULL, (builder.GetStorageSize() + builder.GetStorageAlignment()) * pMRTRenderer->mNumColors, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    if (!pMRTRenderer->mRTAllocator)
        goto fail;

    pMRTRenderer->mColors = new Texture *[pMRTRenderer->mNumColors];
    if (!pMRTRenderer->mColors)
        goto fail;

    for (int i = 0; i < pMRTRenderer->mNumColors; i++) {
        pMRTRenderer->mColors[i] = pMRTRenderer->mRTAllocator->allocTexture(&builder);
        if (!pMRTRenderer->mColors[i])
            goto fail;
    }

    uboData.mask = 0x99;
    uboData.colors[0] = dt::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    uboData.colors[1] = dt::vec4(0.5f, 1.0f, 1.0f, 1.0f);
    uboData.colors[2] = dt::vec4(1.0f, 0.5f, 1.0f, 1.0f);
    uboData.colors[3] = dt::vec4(0.5f, 0.5f, 1.0f, 1.0f);
    uboData.colors[4] = dt::vec4(1.0f, 1.0f, 0.5f, 1.0f);
    uboData.colors[5] = dt::vec4(0.5f, 1.0f, 0.5f, 1.0f);
    uboData.colors[6] = dt::vec4(1.0f, 0.5f, 0.5f, 1.0f);
    uboData.colors[7] = dt::vec4(0.5f, 0.5f, 0.5f, 1.0f);

    pMRTRenderer->mUbo = AllocAndFillBuffer(device, queue, queueCB, *pMRTRenderer->mAllocator, &uboData, sizeof(UniformBlock),
                                            BUFFER_ALIGN_UNIFORM_BIT, false);
    if (!pMRTRenderer->mUbo)
        goto fail;

    pMRTRenderer->mChannelMaskState.SetDefaults();

    for (int i = 0; i < pMRTRenderer->mNumColors; i++)
        if ((uboData.mask & (1 << (unsigned int)i)) == 0)
            pMRTRenderer->mChannelMaskState.SetChannelMask(i, 0, 0, 0, 0);

    pMRTRenderer->mDefaultChannelMaskState.SetDefaults();

    return pMRTRenderer;

fail:
    LWNFailTest();
    delete pMRTRenderer;
    return NULL;
}

void MRTRenderer::bind(QueueCommandBuffer& queueCB)
{
    queueCB.SetRenderTargets(mNumColors, mColors, NULL, NULL, NULL);
    queueCB.BindChannelMaskState(&mDefaultChannelMaskState);
    queueCB.BindProgram(mProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, mUbo->GetAddress(), sizeof(UniformBlock));
}

void MRTRenderer::draw(QueueCommandBuffer& queueCB)
{
    // Clear each render target individually with one set of colors
    for (int i = 0; i < mNumColors; i++) {
        float r = ((i & 0x1) + 1.0f) / mNumColors;
        float g = ((i & 0x2) + 1.0f) / mNumColors;
        float b = ((i & 0x4) + 1.0f) / mNumColors;
        queueCB.ClearColor(i, r, g, b, 1.0);
    }

    queueCB.SetViewportScissor(0, 0, BASE_TEX_SIZE, BASE_TEX_SIZE);
    queueCB.BindChannelMaskState(&mChannelMaskState);

    // Render a smaller rectangle across all MRTs, with colors passed to shader via UBO.
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
    queueCB.BindChannelMaskState(&mDefaultChannelMaskState);

    // Insert a barrier to make sure the draw is finished done before reading from the texture.
    queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_SHADER |
                    BarrierBits::ILWALIDATE_TEXTURE);
}

class LWNTextureViewMultipleRenderTargetsTest
{
public:
    LWNTEST_CppMethods();

    LWNTextureViewMultipleRenderTargetsTest(bool isMRTDisabled) : mIsMRTDisabled(isMRTDisabled) {}

    const bool mIsMRTDisabled;
private:
    static void calcTileDimensions(int numColors, int* tileCountX, int* tileCountY);
};

lwString LWNTextureViewMultipleRenderTargetsTest::getDescription() const
{
    if (mIsMRTDisabled) {
        return "Test for checking that rendering to a single color output in the fragment\n"
               "shader doesn't output colors to other render targets that might be attached.\n"
               "Test should output 8 colored rectangles where only the bottom left rectangle\n"
               "has a different colored smaller rectangle embedded inside it.\n";
    } else {
        return "Test for rendering to and reading from multiple render targets:\n"
               "For each MRT, clear with a unique color.\n"
               "Render a small box to multiple MRTs.\n"
               "Upload UBO with mask and colors.\n"
               "Use ChannelMask so mask takes effect. (otherwise shader would output black)\n"
               "Channel mask is 0x99, which is 'zig - zag' pattern from bottom-left to top-right\n"
               "Texture Viewer will read each MRT in a seperate pass.\n";
    }
}

int LWNTextureViewMultipleRenderTargetsTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(12, 0);
}

// Divide the total number of render targets into x, y dimensions. This works as long as the number of render targets can be represented by a sum of pow2.
void LWNTextureViewMultipleRenderTargetsTest::calcTileDimensions(int numColors, int* tileCountX, int* tileCountY)
{
    int log2NumColors = 0;

    while (numColors >>= 1) {
        log2NumColors++;
    }

    int log2NumColorsY = (log2NumColors >> 1);
    int log2NumColorsX = (log2NumColors - log2NumColorsY);

    *tileCountX = (1 << log2NumColorsY);
    *tileCountY = (1 << log2NumColorsX);
}

void LWNTextureViewMultipleRenderTargetsTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    TextureVisualizer* pViewer = NULL;
    MRTRenderer* pRenderer = NULL;

    int x, y, tileCountX, tileCountY, viewportWidth, viewportHeight;

    // Clear the first render target to all black.
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);

    pRenderer = MRTRenderer::create(device, queue, queueCB, mIsMRTDisabled);
    if (!pRenderer)
        goto cleanup;

    // Bind all render targets. Clear them and render a centered rectangle in one pass.
    pRenderer->bind(queueCB);
    pRenderer->draw(queueCB);

    g_lwnWindowFramebuffer.bind();

    pViewer = TextureVisualizer::create(device);
    if (!pViewer)
        goto cleanup;

    pViewer->bind(queueCB);

    // Divide the framebuffer into different viewport tiles, one viewport per render target
    calcTileDimensions(pRenderer->getNumColors(), &tileCountX, &tileCountY);

    viewportWidth  = lwrrentWindowWidth / tileCountX;
    viewportHeight = lwrrentWindowHeight / tileCountY;

    // Bind each render target as a texture and render it to a viewport.
    for (y = 0; y < tileCountY; y++) {
        for (x = 0; x < tileCountX; x++) {
            queueCB.SetViewportScissor(x * viewportWidth, y * viewportHeight, viewportWidth, viewportHeight);
            pViewer->draw(device, queueCB, pRenderer->getColor(y * tileCountX + x), NULL);
        }
    }

cleanup:
    queueCB.submit();
    queue->Finish();   // Idle CPU until GPU operations are complete.

    delete pRenderer;
    delete pViewer;
}

OGTEST_CppTest(LWNTextureViewMultipleRenderTargetsTest, lwn_texture_view_multiple_render_targets, (false));
OGTEST_CppTest(LWNTextureViewMultipleRenderTargetsTest, lwn_texture_view_multiple_render_targets_disabled, (true));

class LWNTextureViewFormatSwizzleTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNTextureViewFormatSwizzleTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Test checks texture view's swizzle and format attributes.\n"
          "It fills random texture data and samples it with two samplers,\n"
          "one sampler has defined format and swizzle and second has a view.\n"
          "Sampled values are then compared.";
    return sb.str();
}

int LWNTextureViewFormatSwizzleTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 9);
}

struct FmtSwCombo
{
    Format fmt;
    TextureSwizzle r;
    TextureSwizzle g;
    TextureSwizzle b;
    TextureSwizzle a;
};

static void setNextCell(CommandBuffer* cmdPtr = NULL)
{
    static const int cellSize = 16;
    static const int cellMargin = 1;
    static const int cellPerRow = lwrrentWindowWidth / (cellSize + 2 * cellMargin);
    static int cellCounter = 0;
    static CommandBuffer* cmd = NULL;
    if (cmdPtr) {
        cmd = cmdPtr;
        cellCounter = 0;
    } else {
        int offset = cellSize + 2 * cellMargin;
        int x = cellCounter % cellPerRow * offset + cellMargin;
        int y = cellCounter / cellPerRow * offset + cellMargin;
        cmd->SetViewport(x, y, cellSize, cellSize);
        cellCounter++;
    }
}

static void compareDraw(MemoryPool* gpupool, FmtSwCombo& fsc, TextureBuilder& tb, TextureView& tv, LWNuint samplerID)
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    setNextCell();

    tb.SetFormat(fsc.fmt).SetSwizzle(fsc.r, fsc.g, fsc.b, fsc.a);
    Texture* tex = tb.CreateTextureFromPool(gpupool, 0);
    tv.SetFormat(fsc.fmt).SetSwizzle(fsc.r, fsc.g, fsc.b, fsc.a);

    LWNuint texA_ID, texB_ID;
    TextureHandle handleA, handleB;
    texA_ID = g_lwnTexIDPool->Register(tex, 0);
    texB_ID = g_lwnTexIDPool->Register(tex, &tv);
    handleA = device->GetTextureHandle(texA_ID, samplerID);
    handleB = device->GetTextureHandle(texB_ID, samplerID);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, handleA);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 1, handleB);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
    queueCB.submit();
    queue->Finish();

    tex->Free();
}

void LWNTextureViewFormatSwizzleTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0);
    queueCB.submit();

    // test texture dimension
    const int imgDim = 2;

    // combos (lwrrently w/o compressed format types)
    FmtSwCombo fFormats[] = {
        {Format::R8,    TextureSwizzle::R, TextureSwizzle::ONE, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::R8SN,  TextureSwizzle::R, TextureSwizzle::ONE, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::R16,   TextureSwizzle::R, TextureSwizzle::ONE, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::R16F,  TextureSwizzle::R, TextureSwizzle::ONE, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::R16SN, TextureSwizzle::R, TextureSwizzle::ONE, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::R32F,  TextureSwizzle::R, TextureSwizzle::ONE, TextureSwizzle::ONE, TextureSwizzle::ONE},

        {Format::RG8,   TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::RG8SN, TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::RG16,  TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::RG16F, TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::RG16SN,TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::RG32F, TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::ONE, TextureSwizzle::ONE},

        {Format::RGBA8,   TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},
        {Format::RGBA8SN, TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},
        {Format::RGBA16,  TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},
        {Format::RGBA16F, TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},
        {Format::RGBA16SN,TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},
        {Format::RGBA32F, TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},

        {Format::RGBX8,   TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::RGBX8SN, TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::RGBX16,  TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::RGBX16F, TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::RGBX16SN,TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::RGBX32F, TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},

        {Format::RGB9E5F,   TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::R11G11B10F,TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::RGBA4,     TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},
        {Format::RGB5,      TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::BGR5,      TextureSwizzle::B, TextureSwizzle::G, TextureSwizzle::R, TextureSwizzle::ONE},
        {Format::RGB5A1,    TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},

        {Format::A1BGR5,    TextureSwizzle::A, TextureSwizzle::B, TextureSwizzle::G, TextureSwizzle::R},
        {Format::BGR5A1,    TextureSwizzle::B, TextureSwizzle::G, TextureSwizzle::R, TextureSwizzle::A},
        {Format::RGB565,    TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::BGR565,    TextureSwizzle::B, TextureSwizzle::G, TextureSwizzle::R, TextureSwizzle::ONE},
        {Format::BGRX8,     TextureSwizzle::B, TextureSwizzle::G, TextureSwizzle::R, TextureSwizzle::ONE},
        {Format::BGRX8_SRGB,TextureSwizzle::B, TextureSwizzle::G, TextureSwizzle::R, TextureSwizzle::ONE},

        {Format::RGBX8_SRGB,TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::BGRA8,     TextureSwizzle::B, TextureSwizzle::G, TextureSwizzle::R, TextureSwizzle::A},
        {Format::BGRA8_SRGB,TextureSwizzle::B, TextureSwizzle::G, TextureSwizzle::R, TextureSwizzle::A},
        {Format::RGBA8_SRGB,TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},
    };
    FmtSwCombo uFormats[] = {
        {Format::R8UI,      TextureSwizzle::R, TextureSwizzle::ONE, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::R16UI,     TextureSwizzle::R, TextureSwizzle::ONE, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::R32UI,     TextureSwizzle::R, TextureSwizzle::ONE, TextureSwizzle::ONE, TextureSwizzle::ONE},

        {Format::RG8UI,     TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::RG16UI,    TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::RG32UI,    TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::ONE, TextureSwizzle::ONE},

        {Format::RGBA8UI,   TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},
        {Format::RGBA16UI,  TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},
        {Format::RGBA32UI,  TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},

        {Format::RGBX8UI,   TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::RGBX16UI,  TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::RGBX32UI,  TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
    };
    FmtSwCombo iFormats[] = {
        {Format::R8I,      TextureSwizzle::R, TextureSwizzle::ONE, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::R16I,     TextureSwizzle::R, TextureSwizzle::ONE, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::R32I,     TextureSwizzle::R, TextureSwizzle::ONE, TextureSwizzle::ONE, TextureSwizzle::ONE},

        {Format::RG8I,     TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::RG16I,    TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::ONE, TextureSwizzle::ONE},
        {Format::RG32I,    TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::ONE, TextureSwizzle::ONE},

        {Format::RGBA8I,   TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},
        {Format::RGBA16I,  TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},
        {Format::RGBA32I,  TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A},

        {Format::RGBX8I,   TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::RGBX16I,  TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
        {Format::RGBX32I,  TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::ONE},
    };

    Program *prog_f, *prog_u, *prog_i;
    VertexShader vs(440);
    FragmentShader fs_f(440), fs_u(440), fs_i(440);

    vs <<
          "out vec2 uv;\n"
          "void main() {\n"
          "  vec2 position; "
          "  if (gl_VertexID == 0) position = vec2(-1.0, -1.0);"
          "  if (gl_VertexID == 1) position = vec2(1.0, -1.0);"
          "  if (gl_VertexID == 2) position = vec2(1.0, 1.0);"
          "  if (gl_VertexID == 3) position = vec2(-1.0, 1.0);"
          "  gl_Position = vec4(position, 0.0, 1.0);\n"
          "  uv = position*0.5 + vec2(0.5,0.5);\n"
          "}\n";

    fs_f <<
        "in vec2 uv;\n"
        "layout (binding=0) uniform sampler2D texA;\n"
        "layout (binding=1) uniform sampler2D texB_tv;\n"
        "out vec4 fcolor;\n"
        "bool matchF(float f1, float f2) {\n"
        "  return f1 == f2 || (isnan(f1) && isnan(f2)) || (isinf(f1) && isinf(f2));\n"
        "}\n"
        "bool matchV(vec4 v1, vec4 v2) {\n"
        "  return matchF(v1.r, v2.r) && matchF(v1.g, v2.g) && matchF(v1.b, v2.b) && matchF(v1.a, v2.a);\n"
        "}\n"
        "void main() {\n"
        "  vec4 colA = texture(texA,uv);\n"
        "  fcolor = clamp(colA, 0.0, 1.0);\n"
        "  vec4 colB = texture(texB_tv,uv);\n"
        "  fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "  if(matchV(colA, colB))\n"
        "    fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "}\n";

    fs_u <<
        "in vec2 uv;\n"
        "layout (binding=0) uniform usampler2D texA;\n"
        "layout (binding=1) uniform usampler2D texB_tv;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  uvec4 colA = texture(texA,uv);\n"
        "  uvec4 colB = texture(texB_tv,uv);\n"
        "  fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "  if(colA == colB)\n"
        "    fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "}\n";

    fs_i <<
        "in vec2 uv;\n"
        "layout (binding=0) uniform isampler2D texA;\n"
        "layout (binding=1) uniform isampler2D texB_tv;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  ivec4 colA = texture(texA,uv);\n"
        "  ivec4 colB = texture(texB_tv,uv);\n"
        "  fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "  if(colA == colB)\n"
        "    fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "}\n";

    prog_f = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(prog_f, vs, fs_f);
    prog_u = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(prog_u, vs, fs_u);
    prog_i = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(prog_i, vs, fs_i);

    // texture builder
    TextureBuilder tb;
    tb.SetDevice(device);
    tb.SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(imgDim,imgDim);
    tb.SetFormat(Format::RGBA32F);

    // this is maximum memory we will need for any texture.
    // depth/stencil formats removed; Xi's suggestion
    // otherwise, DEPTH32F_STENCIL8 would report 65k size
    size_t maxTexMemoryReq = tb.GetStorageSize();

    // memory
    MemoryPool *cpuBufferMemPool = device->CreateMemoryPool(NULL, maxTexMemoryReq, MemoryPoolType::CPU_COHERENT);
    MemoryPool* gpuTexMemPool = device->CreateMemoryPool(NULL, maxTexMemoryReq, MemoryPoolType::GPU_ONLY);

    // buffer (fill with random data)
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *textureBuffer = bb.CreateBufferFromPool(cpuBufferMemPool, 0, maxTexMemoryReq);
    unsigned* textureBufferPtr = static_cast<unsigned*>(textureBuffer->Map());
    for (unsigned i=0; i<maxTexMemoryReq/sizeof(unsigned); i++) {
        textureBufferPtr[i] = lwBitRand(32);
    }

    // fill texture with data
    queueCB.CopyBufferToBuffer(textureBuffer->GetAddress(), gpuTexMemPool->GetBufferAddress(), maxTexMemoryReq, CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();

    // sampler
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults().SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler* smp = sb.CreateSampler();
    LWNuint samplerID = smp->GetRegisteredID();

    // texture view
    TextureView tv;
    tv.SetDefaults().SetTarget(TextureTarget::TARGET_2D);

    setNextCell(&queueCB);

    // 1. F-formats
    queueCB.BindProgram(prog_f, ShaderStageBits::ALL_GRAPHICS_BITS);
    for (unsigned i=0; i<__GL_ARRAYSIZE(fFormats); i++) {
        compareDraw(gpuTexMemPool, fFormats[i], tb, tv, samplerID);
    }

    // 2. UI-formats
    queueCB.BindProgram(prog_u, ShaderStageBits::ALL_GRAPHICS_BITS);
    for (unsigned i=0; i<__GL_ARRAYSIZE(uFormats); i++) {
        compareDraw(gpuTexMemPool, uFormats[i], tb, tv, samplerID);
    }

    // 3. I-formats
    queueCB.BindProgram(prog_i, ShaderStageBits::ALL_GRAPHICS_BITS);
    for (unsigned i=0; i<__GL_ARRAYSIZE(iFormats); i++) {
        compareDraw(gpuTexMemPool, iFormats[i], tb, tv, samplerID);
    }

    smp->Free();
    textureBuffer->Free();
    prog_f->Free();
    prog_i->Free();
    prog_u->Free();
    gpuTexMemPool->Free();
    cpuBufferMemPool->Free();
}

OGTEST_CppTest(LWNTextureViewFormatSwizzleTest, lwn_texture_view_format_swizzle, );
