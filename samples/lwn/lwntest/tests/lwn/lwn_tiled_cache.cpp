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

#include "lwnUtil/lwnUtil_TiledCacheState.h"
#include "lwnUtil/lwnUtil_TiledCacheStateImpl.h"

#include <fstream>
#include <string>
#include <sstream>

using namespace lwn;

using lwn::util::TiledCacheState;

class LWNTiledCacheTest
{
public:
    LWNTEST_CppMethods();

    enum TestOptions {
        TILED_CACHE_OFF,
        TILED_CACHE_ON,
        TILED_CACHE_STATE,
        TILED_CACHE_ENTRY_POINTS
    };

    LWNTiledCacheTest(TestOptions options) :
        mTestOptions(options)
    {}

    void testEntryPoints() const;
    void testState() const;
    void testTiledCache(bool tiledCacheEnabled) const;

private:
    TestOptions mTestOptions;
};

lwString LWNTiledCacheTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Tiled caching test.\n"
          "lwn_tiled_cache_entry_points: Runs tiled cache action routines to test entry points.\n"
          "lwn_tiled_cache_state: Tests the lwn-utils helper code that determines tile size/action, and draws a green square for each passing test.\n"
          "lwn_tiled_cache_off: Draws overlapping alpha-blended quads in MSAA buffer and downsamples to single sample AA. Tiled caching is off.\n "
          "lwn_tiled_cache_on: Draws overlapping alpha-blended quads in MSAA buffer, downsamples, and discards the source MSAA resource. Tiled caching is on.\n";

    return sb.str();
}

int LWNTiledCacheTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(38, 4);
}

void LWNTiledCacheTest::testEntryPoints() const
{
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    // Test some direct API calls
    queueCB.SetTiledCacheAction(TiledCacheAction::DISABLE_RENDER_TARGET_BINNING);
    queueCB.SetTiledCacheAction(TiledCacheAction::ENABLE_RENDER_TARGET_BINNING);
    queueCB.SetTiledCacheAction(TiledCacheAction::DISABLE);
    queueCB.SetTiledCacheAction(TiledCacheAction::FLUSH);
    queueCB.SetTiledCacheAction(TiledCacheAction::FLUSH_NO_TILING);
    queueCB.SetTiledCacheAction(TiledCacheAction::ENABLE);

    queueCB.SetTiledCacheTileSize(16, 16);
    queueCB.SetTiledCacheTileSize(600, 600);
    queueCB.SetTiledCacheTileSize(300, 300);
    queueCB.SetTiledCacheTileSize(2048, 2048);
    queueCB.SetTiledCacheTileSize(128, 128);

    // Clear color green
    queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);

    // Submit work
    queueCB.submit();
}

#define SETUP_TEST_RESULTS() \
    LWNuint testCount = 0; \
    float red[4] = { 1.0f, 0.0f, 0.0f, 1.0f }; \
    float green[4] = { 0.0f, 1.0f, 0.0f, 1.0f };

#define SHOW_TEST_RESULT(ok) \
    queueCB.SetViewportScissor(8 + 24 * (testCount++), lwrrentWindowHeight - 24, 16, 16); \
    queueCB.ClearColor(0, ok ? green : red);

// This routine tests tiled cache state routine in lwn_utils.
// Each test draws a green box if it passed and a black box if it did not
void LWNTiledCacheTest::testState() const
{
    const int w = 640;
    const int h = 480;

    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    TiledCacheState tiledCacheState(device);

    MemoryPoolAllocator allocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    Texture *colors[4];
    Format formats[4] = { Format::RGBA8, Format::RGBA8, Format::R32F, Format::R32F };

    for (int i = 0; i < 4; i++) {
        TextureBuilder textureBuilder;
        textureBuilder.SetDevice(device).SetDefaults()
            .SetTarget(TextureTarget::TARGET_2D)
            .SetFormat(formats[i])
            .SetSize2D(w, h);
        colors[i] = allocator.allocTexture(&textureBuilder);
    }

    TextureBuilder depthTB;
    depthTB.SetDevice(device).SetDefaults()
        .SetTarget(TextureTarget::TARGET_2D)
        .SetFormat(Format::DEPTH24_STENCIL8)
        .SetFlags(TextureFlags::COMPRESSIBLE)
        .SetSize2D(w, h);

    int tileSize4bpp[2];
    int tileSize16bpp[2];

    SETUP_TEST_RESULTS();

    // Validate tile size is larger as color footprint goes down
    tiledCacheState.UpdateTileState(queueCB, 1, 1, colors, NULL);

    // Verify that tiled caching is flushed after this call
    LWNtiledCacheAction lastAction = tiledCacheState.GetLastAction();
    SHOW_TEST_RESULT(lastAction == LWN_TILED_CACHE_ACTION_ENABLE || lastAction == LWN_TILED_CACHE_ACTION_FLUSH);

    tiledCacheState.GetTileSize(&tileSize4bpp[0], &tileSize4bpp[1]);
    tiledCacheState.UpdateTileState(queueCB, 4, 1, colors, NULL);
    tiledCacheState.GetTileSize(&tileSize16bpp[0], &tileSize16bpp[1]);

    // Test1: verify that 4x render target footprint = 1/4th tile area
    LWNuint tileSizeFactor = (tileSize4bpp[0] * tileSize4bpp[1]) / (tileSize16bpp[0] * tileSize16bpp[1]);
    SHOW_TEST_RESULT(tileSizeFactor == 4);

    // Test2: Even with small caches, a 4bpp tile size should always be greater than minimum tile size,
    //        or something probably went wrong in the callwlation somewhere or l2 size is not being returned.
    SHOW_TEST_RESULT(tileSize4bpp[0] > 16 && tileSize4bpp[1] > 16);

    // Test3: verify that second call flushes because of adding additional render targets
    SHOW_TEST_RESULT(tiledCacheState.GetLastAction() == LWN_TILED_CACHE_ACTION_FLUSH);

    // Test 4: verify that third call does not flush binner because binding an ordered subset of render targets
    tiledCacheState.UpdateTileState(queueCB, 3, 1, colors, NULL);
    SHOW_TEST_RESULT(tiledCacheState.GetLastAction() == LWN_TILED_CACHE_ACTION_ENABLE);

    // Test 5: verify that third call does not flush binner because binding an ordered subset of render targets
    Texture* colors2[4] = { NULL, colors[1], NULL, colors[2] };
    tiledCacheState.UpdateTileState(queueCB, 4, 1, colors2, NULL);
    SHOW_TEST_RESULT(tiledCacheState.GetLastAction() == LWN_TILED_CACHE_ACTION_ENABLE);

    // Test 6: verify that tiled caching is disabled if depth only (and we assume app is not rendering to stencil here)
    Texture * depthTexture = allocator.allocTexture(&depthTB);
    tiledCacheState.SetStrategy(TiledCacheState::SKIP_STENCIL_COMPONENT_BIT);
    tiledCacheState.UpdateTileState(queueCB, 0, 0, NULL, depthTexture);
    SHOW_TEST_RESULT(tiledCacheState.GetLastAction() == LWN_TILED_CACHE_ACTION_DISABLE);

    queueCB.submit();
}


void LWNTiledCacheTest::testTiledCache(bool tiledCacheEnabled) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(binding = 0) uniform Block {\n"
        "    vec4 color;\n"
        "    vec2 scale;\n"
        "    vec2 offset;\n"
        "};\n"
        "out vec4 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4((position.xy * scale + offset) * 2.0 - 1.0, position.z, 1.0);\n"
        "  ocolor = color;\n"
        "}\n";

    FragmentShader fs(440);
    fs <<
        "out vec4 fcolor;\n"
        "in vec4 ocolor;\n"
        "void main() {\n"
        "  fcolor = ocolor;\n"
        "}\n";

    Program *pgm = device->CreateProgram();

    // Compile and call lwnProgramSetShaders.
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        const char * infoLog = g_glslcHelper->GetInfoLog();
        printf("Shader compile error. infoLog =\n%s\n", infoLog);
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0) },
        { dt::vec3(-1.0, +1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, 0.0) },
        { dt::vec3(+1.0, +1.0, 0.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 0x10000UL, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    TextureBuilder textureBuilder;
    textureBuilder.SetDevice(device).SetDefaults().
        SetSize2D(lwrrentWindowWidth, lwrrentWindowHeight).
        SetFlags(TextureFlags::COMPRESSIBLE).
        SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE).
        SetFormat(Format::RGBA8).
        SetSamples(8);

    MemoryPoolAllocator textureAllocator(device, NULL, textureBuilder.GetPaddedStorageSize(), LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    Texture *msTexture = textureAllocator.allocTexture(&textureBuilder);

    queueCB.BindMultisampleState(&MultisampleState().SetDefaults().SetSamples(8));
    queueCB.SetRenderTargets(1, &msTexture, NULL, NULL, NULL);
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);

    struct UniformBlock
    {
        dt::vec4 color;
        dt::vec2 scale;
        dt::vec2 offset;
    };

    static const int NumQuadsX = 16;
    static const int NumQuadsY = 16;
    static const int NumQuads = NumQuadsX * NumQuadsY;

    BlendState blendState;
    blendState.SetDefaults().
        SetBlendFunc(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA, BlendFunc::ONE, BlendFunc::ONE).
        SetBlendEquation(BlendEquation::ADD, BlendEquation::ADD);

    ColorState colorState;
    colorState.SetDefaults().
        SetBlendEnable(0, true);

    queueCB.BindBlendState(&blendState);
    queueCB.BindColorState(&colorState);

    LWNint uboAlignment = 0;
    device->GetInteger(DeviceInfo::UNIFORM_BUFFER_ALIGNMENT, &uboAlignment);
    int uboSize = uboAlignment * ((sizeof(UniformBlock) + uboAlignment - 1) / uboAlignment);

    BufferBuilder uboBuilder;
    uboBuilder.SetDevice(device).SetDefaults();

    Buffer *ubo = allocator.allocBuffer(&uboBuilder, BUFFER_ALIGN_UNIFORM_BIT, uboSize * NumQuads);
    uint8_t *pUboStart = (uint8_t *)ubo->Map();

    for (int x = 0; x < NumQuadsX; x++) {
        for (int y = 0; y < NumQuadsY; y++) {

            UniformBlock* pUboData = (UniformBlock*)(pUboStart + ((y * NumQuadsX) + x) * uboSize);

            pUboData->scale = dt::vec2(0.15, 0.15);

            pUboData->color[0] = float(x) / (float(NumQuadsX) - 1);
            pUboData->color[1] = float(y) / (float(NumQuadsY) - 1);
            pUboData->color[2] = 0.0;
            pUboData->color[3] = 0.25;

            pUboData->offset[0] = float(x) / (float(NumQuadsX) - 1);
            pUboData->offset[1] = float(y) / (float(NumQuadsY) - 1);
        }
    }

    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    if (tiledCacheEnabled) {
        // This ia functional test, so we're not trying to optimize anything here
        queueCB.SetTiledCacheAction(TiledCacheAction::ENABLE);
        queueCB.SetTiledCacheTileSize(128, 128);
    }

    for (int i = 0; i < NumQuads; i++) {
        queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, ubo->GetAddress() + (i * uboSize), sizeof(UniformBlock));
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    }

    if (tiledCacheEnabled) {
        // Depth is not used in this test, but lets discard it here to test the entry point.
        queueCB.DiscardDepthStencil();
    }

    queueCB.Downsample(msTexture, g_lwnWindowFramebuffer.getAcquiredTexture());

    if (tiledCacheEnabled) {
        // Discard is orthogonal to tiled caching, but for purposes of
        // this test we are testing discard of source MSAA here.  LWN
        // lwrrently does not use the 3D engine for MSAA resolves, so
        // if we want resolves to be binned, we need a custom 3D
        // resolve path and use tiled barriers.
        queueCB.DiscardColor(0);
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();

    // Reset tiled cache state, multisample state, and lwrrently bound render target to
    // defaults.
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();

    queueCB.SetTiledCacheAction(TiledCacheAction::DISABLE);
    queueCB.BindMultisampleState(&MultisampleState().SetDefaults());
    queueCB.submit();
    queue->Finish();
}

void LWNTiledCacheTest::doGraphics() const
{
    switch (mTestOptions) {
    case TILED_CACHE_OFF:
        testTiledCache(false);
        break;

    case TILED_CACHE_ON:
        testTiledCache(true);
        break;

    case TILED_CACHE_STATE:
        testState();
        break;

    case TILED_CACHE_ENTRY_POINTS:
        testEntryPoints();
        break;

    default:
        assert(0);
        break;
    }
}

OGTEST_CppTest(LWNTiledCacheTest, lwn_tiled_cache_off, (LWNTiledCacheTest::TILED_CACHE_OFF));
OGTEST_CppTest(LWNTiledCacheTest, lwn_tiled_cache_on, (LWNTiledCacheTest::TILED_CACHE_ON));
OGTEST_CppTest(LWNTiledCacheTest, lwn_tiled_cache_state, (LWNTiledCacheTest::TILED_CACHE_STATE));
OGTEST_CppTest(LWNTiledCacheTest, lwn_tiled_cache_entry_points, (LWNTiledCacheTest::TILED_CACHE_ENTRY_POINTS));
