//
// lwn_shader_scratch.cpp
//
// Touch test for shader scratch memory, used with reg spilling shaders.
//

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "../../elw/cmdline.h"

#define DEBUG_MODE 0
#if DEBUG_MODE
    #define DEBUG_PRINT(x) do { \
        printf x; \
        fflush(stdout); \
    } while (0)
#else
    #define DEBUG_PRINT(x)
#endif

using namespace lwn;

// --------------------------------- LWNShaderScratchHarness ----------------------------------------

static bool s_gotDebugError = false;
static void LWNAPIENTRY DebugCallback(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
                                      DebugCallbackSeverity::Enum severity, LWNstring message, void *userParam)
{
    DEBUG_PRINT(("lwnDebug: %s\n", (const char*) message));
    s_gotDebugError = true;
}

enum Variant {
    // Run the shaders with sufficient scratch memory.
    SCRATCH_BASIC,

    // Run the shaders with throttle and insufficient scratch memory in debug layer enabled mode.
    SCRATCH_DEBUG,

    // Run the shsders with throttle scratch memory in debug layer disabled mode.
    SCRATCH_THROTTLE,

    TESTS_NUM
};

class LWNShaderScratchHarness {
    static const int maxScratchMemorySize          = 128 * 1024 * 1024;
    int minGfxScratchMemorySize;
    int minComputeScratchMemorySize;
    MemoryPoolAllocator *m_bufpool;
    MemoryPoolAllocator *m_texpool;
    MemoryPoolAllocator *m_ubopool;
    Program *m_program;
    Program *m_computeProgram;
    Program *m_computeDisplayProgram;
    VertexArrayState m_vertexState;
    Buffer *m_vbo;
    Buffer *m_ubo;
    Texture *m_computeTexture;
    Sampler *m_sampler;
    LWNuint m_vertexDataSize;
    MemoryPool *m_scratchpool;

    struct Vertex {
        dt::vec3 position;
        dt::vec3 uv;
    };

public:
    LWNShaderScratchHarness();
    ~LWNShaderScratchHarness();

    bool init(Device *device, QueueCommandBuffer& queueCB, lwnTest::GLSLCHelper *glslcHelper,
              Variant variant);
    void draw(Device *device, QueueCommandBuffer& queueCB);
    void drawCompute(Device *device, QueueCommandBuffer& queueCB);

    inline MemoryPool *getScratchMemoryPool(void) {
        return m_scratchpool;
    }
    inline int getMinGfxScratchMemorySize() { return this->minGfxScratchMemorySize; }
    inline int getMinComputeScratchMemorySize() { return this->minComputeScratchMemorySize; }
};

LWNShaderScratchHarness::LWNShaderScratchHarness()
        : m_bufpool(NULL), m_texpool(NULL), m_ubopool(NULL)
{}

LWNShaderScratchHarness::~LWNShaderScratchHarness()
{
    m_program->Free();
    m_computeProgram->Free();
    m_computeDisplayProgram->Free();
    m_sampler->Free();

    delete m_bufpool;
    delete m_texpool;
    delete m_ubopool;
    if (m_scratchpool) {
        m_scratchpool->Free();
    }
}

bool LWNShaderScratchHarness::init(Device *device, QueueCommandBuffer& queueCB, lwnTest::GLSLCHelper *glslcHelper,
                                   Variant variant)
{
    assert(glslcHelper);

    m_scratchpool = device->CreateMemoryPool(NULL, maxScratchMemorySize, MemoryPoolType::GPU_ONLY);
    // Set max SM for compilation.
    glslcHelper->SetShaderScratchMemory(m_scratchpool, 0, maxScratchMemorySize, queueCB);

    m_bufpool = new MemoryPoolAllocator(device, NULL, 32 * 0x10000, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    m_texpool = new MemoryPoolAllocator(device, NULL, 0x40000, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    // Build a vertex shader that simply passes through vertex position and a
    // texture coordinate.  Used for graphics shader testing and displaying
    // compute shader results.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 tc;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "out vec2 otc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  otc = tc.xy;\n"
        "}\n";

    // Build a fragment shader that simply loops through a scratch array to
    // compute a greenish constant color.
    FragmentShader fs(440);
    fs <<
        "out vec4 fcolor;\n"
        // Offset array index by a uniform in order to stop GLSLC from optimising the loop out.
        "layout(binding=0, std140) uniform Block {\n"
        "    int indexOffset;\n"
        "};\n"
        "void main() {\n"
        "    vec4 tmp[128];\n"
        "    for (int i = 0; i < 128; i++) {\n"
        "        tmp[indexOffset + i - 1337] = vec4(0.4, 0.85, 0.5, 0.8);\n"
        "    }\n"
        "    vec4 pop = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "    for (int i = 0; i < 128; i++) {\n"
        "        pop += tmp[indexOffset + i - 1337];\n"
        "    }\n"
        "    fcolor = pop / 128.0;\n"
        "}\n";
    m_program = device->CreateProgram();
    if (!glslcHelper->CompileAndSetShaders(m_program, vs, fs)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", glslcHelper->GetInfoLog()));
        return false;
    }

    minGfxScratchMemorySize = (int)glslcHelper->GetScratchMemoryMinimum(glslcHelper->GetCompiledOutput(0));

    // Build a compute shader that simply loops through a scratch array to
    // compute a greenish constant color and store it in a texture via image
    // stores.
    ComputeShader cs(440);
    cs <<
        "layout(binding=0,rgba32f) uniform image2D image;\n"
        // Offset array index by a uniform in order to stop GLSLC from optimising the loop out.
        "layout(binding=0, std140) uniform Block {\n"
        "    int indexOffset;\n"
        "};\n"
        "void main() {\n"
        "    vec4 fcolor;\n"
        "    vec4 tmp[128];\n"
        "    for (int i = 0; i < 128; i++) {\n"
        "        tmp[indexOffset + i - 1337] = vec4(0.4, 0.85, 0.5, 0.8);\n"
        "    }\n"
        "    vec4 pop = vec4(0.0, 1.0, 0.0, 0.0);\n"
        "    for (int i = 0; i < 128; i++) {\n"
        "        pop += tmp[indexOffset + i - 1337];\n"
        "    }\n"
        "    fcolor = pop / 128.0;\n"
        "    imageStore(image, ivec2(gl_GlobalIlwocationID.xy), fcolor);\n"
        "}\n";
    cs.setCSGroupSize(8,8);
    m_computeProgram = device->CreateProgram();
    if (!glslcHelper->CompileAndSetShaders(m_computeProgram, cs)) {
        DEBUG_PRINT(("Compute shader compile error. infoLog =\n%s\n", glslcHelper->GetInfoLog()));
        return false;
    }

    minComputeScratchMemorySize = (int)glslcHelper->GetScratchMemoryMinimum(glslcHelper->GetCompiledOutput(0));

    // Build a fragment shader to read in texture written by the compute
    // shader and display it on-screen.
    FragmentShader fstex(440);
    fstex <<
        "layout(binding=0) uniform sampler2D smp;\n"
        "out vec4 fcolor;\n"
        "in vec2 otc;\n"
        "void main() {\n"
        "    fcolor = texture(smp, otc);\n"
        "}\n";
    m_computeDisplayProgram = device->CreateProgram();
    if (!glslcHelper->CompileAndSetShaders(m_computeDisplayProgram, vs, fstex)) {
        DEBUG_PRINT(("Display shader compile error. infoLog =\n%s\n", glslcHelper->GetInfoLog()));
        return false;
    }

    // Set up a 128x128 texture to receive the results of image stores.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetFlags(TextureFlags::IMAGE);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(128, 128);
    tb.SetFormat(Format::RGBA32F);
    tb.SetLevels(1);
    m_computeTexture = m_texpool->allocTexture(&tb);

    // Set up a dummy sampler to be used to sample that texture.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    m_sampler = sb.CreateSampler();

    // Create vertex data.
    const int vertexCount = 4;
    static const Vertex vertexData[] = {
        { dt::vec3(-1, -1, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+1, -1, 0.0), dt::vec3(1.0, 1.0, 0.0) },
        { dt::vec3(+1, +1, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-1, +1, 0.0), dt::vec3(0.0, 0.0, 0.0) }
    };
    m_vertexDataSize = sizeof(vertexData);
    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, uv);
    m_vertexState = vertexStream.CreateVertexArrayState();
    m_vbo = vertexStream.AllocateVertexBuffer(device, vertexCount, *m_bufpool, vertexData);

    BufferBuilder uboBuilder;
    uboBuilder.SetDefaults().SetDevice(device);
    m_ubo = m_bufpool->allocBuffer(&uboBuilder, BUFFER_ALIGN_UNIFORM_BIT, sizeof(LWNuint));
    *((int*) m_ubo->Map()) = 1337;

    return true;
}

// Render a simple primitive using our scratch memory fragment shader.
void LWNShaderScratchHarness::draw(Device *device, QueueCommandBuffer& queueCB)
{
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, m_ubo->GetAddress(), sizeof(LWNuint));
    queueCB.BindProgram(m_program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(m_vertexState);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
}

// Render a simple primitive using our scratch memory compute shader shader.
void LWNShaderScratchHarness::drawCompute(Device *device, QueueCommandBuffer& queueCB)
{
    LWNuint imageID = g_lwnTexIDPool->RegisterImage(m_computeTexture);
    LWNtextureHandle th = device->GetTextureHandle(m_computeTexture->GetRegisteredTextureID(),
                                                    m_sampler->GetRegisteredID());
    LWNimageHandle ih = device->GetImageHandle(imageID);

    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, th);
    queueCB.BindImage(ShaderStage::COMPUTE, 0, ih);    

    // First, do a compute dispatch to get the scratch memory compute shader
    // to write into our texture (using image stores).
    queueCB.BindUniformBuffer(ShaderStage::COMPUTE, 0, m_ubo->GetAddress(), sizeof(LWNuint));
    queueCB.BindProgram(m_computeProgram, ShaderStageBits::COMPUTE);
    queueCB.DispatchCompute(16, 16, 1);

    // Use a pass barrier to wait on the compute stores.
    queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_TEXTURE);

    // Now render a primitive to display that texture.
    queueCB.BindProgram(m_computeDisplayProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(m_vertexState);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
}

// --------------------------------- LWNShaderScratchTest ----------------------------------------

class LWNShaderScratchTest {
    static const int cellSize = 64;
    static const int cellMargin = 1;
    Variant m_variant;
public:
    LWNTEST_CppMethods();
    LWNShaderScratchTest(Variant variant);

    void doShaderScratchTest() const;
    void doShaderScratchDebugLayerDrawTimeValidationTest() const;
};

LWNShaderScratchTest::LWNShaderScratchTest(Variant variant)
    : m_variant(variant)
{
}

lwString LWNShaderScratchTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Simple touch test of fragment and compute shaders requiring the use "
        "of shader scratch memory.  ";
    if (m_variant == SCRATCH_BASIC || m_variant == SCRATCH_THROTTLE) {
        sb << 
            "Renders two squares used in light green.  "
            "The first comes from a fragment shader that uses scratch memory to "
            "compute the green value.  The second comes from a compute shader that "
            "uses scratch memory to compute the green values and writes the result "
            "to memory using image stores.";
        if (m_variant == SCRATCH_THROTTLE)
            sb << "All the shaders are running on throttled scratch memory."
                  "This test is not supported if the debug layer is enabled.";
    } else {
        // m_variant == SCRATCH_DEBUG
        sb <<
            "Renders four squares in green, the first two squares test that draw time "
            "validations complain when graphics and compute shaders that need scratch "
            "memory to run but isn't given enough.  The last two squares test that draw "
            "time validations complain when graphics and compute shaders that runs on "
            "scratch memory throttle mode.";
    }
    return sb.str();
}

int LWNShaderScratchTest::isSupported() const
{
    if (m_variant == SCRATCH_DEBUG) {
        if (!g_lwnDeviceCaps.supportsDebugLayer) {
            return 0;
        }
        return lwogCheckLWNAPIVersion(40, 11);
    }
    // We do not support SCRATCH_THROTTLE test if the debug layer is enabled,
    // otherwise, it would throw a warning.
    if (m_variant == SCRATCH_THROTTLE && lwnDebugEnabled) return 0;

    return lwogCheckLWNAPIVersion(38, 0);
}

void LWNShaderScratchTest::doShaderScratchTest() const
{
    static const int cellsX = lwrrentWindowWidth / cellSize;
    static const int cellsY = lwrrentWindowHeight / cellSize;
    cellTestInit(cellsX, cellsY);

    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    queueCB.ClearColor(0, 0.15, 0.15, 0.15, 1.0);

    // Initialize test harness class.
    LWNShaderScratchHarness shaderScratchTest;
    if (!shaderScratchTest.init(device, queueCB, g_glslcHelper, m_variant)) {
        DEBUG_PRINT(("Loading of test assets failed\n"));
        LWNFailTest();
        return;
    }

    int k = 0;
    SetCellViewportScissorPadded(queueCB, k % cellsX, k / cellsX, cellMargin);
    if (m_variant == SCRATCH_THROTTLE) {
        queueCB.SetShaderScratchMemory(shaderScratchTest.getScratchMemoryPool(), 0,
                                       shaderScratchTest.getMinGfxScratchMemorySize());
    }
    shaderScratchTest.draw(device, queueCB);

    k++;
    SetCellViewportScissorPadded(queueCB, k % cellsX, k / cellsX, cellMargin);
    if (m_variant == SCRATCH_THROTTLE) {
        queueCB.SetShaderScratchMemory(shaderScratchTest.getScratchMemoryPool(), 0,
                                       shaderScratchTest.getMinComputeScratchMemorySize());
    }
    shaderScratchTest.drawCompute(device, queueCB);

    queueCB.submit();
    queue->Finish();
}

void LWNShaderScratchTest::doShaderScratchDebugLayerDrawTimeValidationTest() const
{
    DisableLWNObjectTracking();

    DeviceState *testDevice =
        new DeviceState(LWNdeviceFlagBits(LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_4_BIT |
                                                   LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT));
    if (!testDevice || !testDevice->isValid()) {
        delete testDevice;
        DeviceState::SetDefaultActive();
        LWNFailTest();
        return;
    }

    testDevice->SetActive();
    Device *device = testDevice->getDevice();
    Queue *queue = testDevice->getQueue();
    QueueCommandBuffer &queueCB = testDevice->getQueueCB();
    lwnTest::GLSLCHelper *glslcHelper = testDevice->getGLSLCHelper();

    device->InstallDebugCallback(DebugCallback, NULL, LWN_TRUE);

    enum DebugTestCases {
        DRAW_INSUFFICIENCY,
        COMPUTE_INSUFFICIENCY,
        DRAW_THROTTLE,
        COMPUTE_THROTTLE,
        CALLBACK_NUM
    };
    bool debugLayerTestGotCallback[CALLBACK_NUM] = { false };

    {
        // Initialize test harness class.
        LWNShaderScratchHarness shaderScratchTest;
        if (!shaderScratchTest.init(device, queueCB, glslcHelper, m_variant)) {
            DEBUG_PRINT(("Loading of test assets failed\n"));
            LWNFailTest();
            return;
        }

        // Triggering the scratch memory throttle mode generates debug warning message.
        queueCB.SetShaderScratchMemory(shaderScratchTest.getScratchMemoryPool(), 0,
                                       shaderScratchTest.getMinGfxScratchMemorySize());
        shaderScratchTest.draw(device, queueCB);
        s_gotDebugError = false;
        queueCB.submit(); // This should cause a debug layer DTV warning.
        debugLayerTestGotCallback[DRAW_THROTTLE] = s_gotDebugError;

        queueCB.SetShaderScratchMemory(shaderScratchTest.getScratchMemoryPool(), 0,
                                       shaderScratchTest.getMinComputeScratchMemorySize());
        shaderScratchTest.drawCompute(device, queueCB);
        s_gotDebugError = false;
        queueCB.submit(); // This should cause a debug layer DTV warning.
        debugLayerTestGotCallback[COMPUTE_THROTTLE] = s_gotDebugError;

        queue->Finish();

        // GLSLCHelper will fail shader compilations with SCRATCH_MEM_CHECK_INSUFFICIENT if we do this
        // when compiling shaders. Now that we've compiled the shaders, we change the scratch memory
        // size from underneath GLSLHelper to test debug layer functionality.

        // Any size smaller than the minimum SM size is insufficient, here we just use zero.
        queueCB.SetShaderScratchMemory(shaderScratchTest.getScratchMemoryPool(), 0, 0);

        shaderScratchTest.draw(device, queueCB);
        s_gotDebugError = false;
        queueCB.submit(); // This should cause a debug layer DTV error.
        debugLayerTestGotCallback[DRAW_INSUFFICIENCY] = s_gotDebugError;

        shaderScratchTest.drawCompute(device, queueCB);
        s_gotDebugError = false;
        queueCB.submit(); // This should cause a debug layer DTV error.
        debugLayerTestGotCallback[COMPUTE_INSUFFICIENCY] = s_gotDebugError;

        queue->Finish();

        // Bug 1841550, we could not have any draw cmd after drawing compute with insufficient memory,
        // otherwise, MMU fault.
    }

    delete testDevice;
    DeviceState::SetDefaultActive();

    // Draw debug layer check results as squares into the screen.

    QueueCommandBuffer &gqueueCB = *g_lwnQueueCB;
    Queue *gqueue = DeviceState::GetActive()->getQueue();
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    gqueueCB.ClearColor(0, 0.15, 0.15, 0.15, 1.0);
    static const int cellsX = lwrrentWindowWidth / cellSize;
    static const int cellsY = lwrrentWindowHeight / cellSize;
    cellTestInit(cellsX, cellsY);

    for (int k = 0; k < CALLBACK_NUM; k++) {
        SetCellViewportScissorPadded(gqueueCB, k % cellsX, k / cellsX, cellMargin);
        if (debugLayerTestGotCallback[k]) {
            gqueueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
        } else {
            gqueueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
        }
    }

    gqueueCB.submit();
    gqueue->Finish();
}

void LWNShaderScratchTest::doGraphics() const
{
    if (m_variant == SCRATCH_DEBUG) {
        doShaderScratchDebugLayerDrawTimeValidationTest();
    } else {
        doShaderScratchTest();
    }
}

OGTEST_CppTest(LWNShaderScratchTest, lwn_shader_scratch,          (SCRATCH_BASIC));
OGTEST_CppTest(LWNShaderScratchTest, lwn_shader_scratch_debug,    (SCRATCH_DEBUG));
OGTEST_CppTest(LWNShaderScratchTest, lwn_shader_scratch_throttle, (SCRATCH_THROTTLE));
