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

#define DEBUG_MODE 0
#if DEBUG_MODE
#define DEBUG_PRINT(x) printf x
#else
#define DEBUG_PRINT(x)
#endif

#define SHADOW_SIZE 8
#define FBO_SIZE (2 * SHADOW_SIZE)
#define ALLOCATOR_SIZE 0x100000 // An arbitrarily large-enough number. Increase if necessary.

using namespace lwn;
using namespace lwn::dt;

namespace {

class Harness {
public:
    Harness();
    ~Harness();
    bool run(Format format, MagFilter filter, CompareFunc compareFunc);

private:
    Texture *createShadowMap(Format format);
    void render(Format format, MagFilter filter, CompareFunc compareFunc);
    bool verify(Format format, MagFilter filter, CompareFunc compareFunc);

    Device *mDevice;
    Queue *mQueue;
    QueueCommandBuffer &mQueueCB;
    MemoryPoolAllocator mGpuAllocator;
    MemoryPoolAllocator mCoherentAllocator;
    Framebuffer mFbo;
    Program *mProgram;
};

Harness::Harness() :
    mDevice(DeviceState::GetActive()->getDevice()),
    mQueue(DeviceState::GetActive()->getQueue()),
    mQueueCB(DeviceState::GetActive()->getQueueCB()),
    mGpuAllocator(mDevice, NULL, ALLOCATOR_SIZE, LWN_MEMORY_POOL_TYPE_GPU_ONLY),
    mCoherentAllocator(mDevice, NULL, ALLOCATOR_SIZE, LWN_MEMORY_POOL_TYPE_CPU_COHERENT),
    mFbo()
{
    mFbo.setSize(FBO_SIZE, FBO_SIZE);
    mFbo.setColorFormat(0, Format::RGBA32F);
    mFbo.setFlags(TextureFlags::COMPRESSIBLE);
    mFbo.alloc(mDevice);

    // Vertices are trivial known constants, so generate them from a geometry shader, rather than
    // bother with vertex state and a VBO.
    VertexShader vs(440);
    vs << "void main() { }\n";
    GeometryShader gs(440);
    gs <<
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=4) out;\n"
        "out vec2 gST; // Texture coordinates\n"
        "void main() {\n"
        "  for (int t = 0; t <= 1; ++t) {\n"
        "    for (int s = 0; s <= 1; ++s) {\n"
        "      gST = vec2(s, t);\n"
        "      gl_Position = vec4(gST * 2.0 - vec2(1.0), 0.0, 1.0);\n"
        "      EmitVertex();\n"
        "    }\n"
        "  }\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec2 gST;\n"
        "out vec4 fColor;\n"
        "layout (binding=0) uniform sampler2DShadow shadowMap;\n"
        "void main() {\n"
        "  float z = floor(gST.y * " << FBO_SIZE << ".0) / " << (FBO_SIZE - 1) << ".0;\n"
        "  float shadow = texture(shadowMap, vec3(gST, z));\n"
        "  fColor = vec4(vec3(shadow), 1.0);\n"
        "}\n";
    mProgram = mDevice->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(mProgram, vs, gs, fs)) {
        DEBUG_PRINT(("Shader compile error.\n======\n%s\n\n", g_glslcHelper->GetInfoLog()));
        return;
    }
}

Harness::~Harness()
{
    mFbo.destroy();
}

// Yes, it's very silly to reuse this return value for every iteration of the loop. It saves a
// switch statement.
template <typename T>
static int setTexel(void* data, int index, T value)
{
    static_cast<T*>(data)[index] = value;
    return int(sizeof(T));
}

Texture *Harness::createShadowMap(Format format)
{
    char shadowMapData[SHADOW_SIZE * SHADOW_SIZE * 8];
    int stride = 1;
    for (int i = 0; i < SHADOW_SIZE * SHADOW_SIZE; ++i) {
        // Horizontal depth ramp from covering [0,1], inclusive.
        int x = i % SHADOW_SIZE;
        switch (format) {
        case Format::DEPTH16:
            stride = setTexel(shadowMapData, i, uint16_t(0xffffu * x / (SHADOW_SIZE - 1)));
            break;
        case Format::DEPTH24:
            stride = setTexel(shadowMapData, i, uint32_t(0xffffffu * x / (SHADOW_SIZE - 1)) << 8);
            break;
        case Format::DEPTH32F:
            stride = setTexel(shadowMapData, i, x / (SHADOW_SIZE - 1.0f));
            break;
        case Format::DEPTH24_STENCIL8:
            stride = setTexel(shadowMapData, i, uint32_t(0xffffffu * x / (SHADOW_SIZE - 1)) << 8);
            break;
        case Format::DEPTH32F_STENCIL8:
            {
                std::pair<float, float> data = { x / (SHADOW_SIZE - 1.0f), 0.0f };
                stride = setTexel(shadowMapData, i, data);
                break;
            }
        default:
            assert(!"Invalid depth format");
        }
    }
    return AllocAndFillTexture2D(mDevice, mQueue, mQueueCB, mGpuAllocator, mCoherentAllocator,
                                 shadowMapData, stride, SHADOW_SIZE, SHADOW_SIZE, format);
}

void Harness::render(Format format, MagFilter filter, CompareFunc compareFunc)
{
    Texture *shadowMap = createShadowMap(format);
    mFbo.bind(mQueueCB);
    mQueueCB.SetViewport(0, 0, FBO_SIZE, FBO_SIZE);
    mQueueCB.SetScissor(0, 0, FBO_SIZE, FBO_SIZE);
    SamplerBuilder samplerBuilder;
    samplerBuilder.SetDevice(mDevice).SetDefaults();
    samplerBuilder.SetMinMagFilter(MinFilter::NEAREST, filter)
                  .SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE,
                               WrapMode::CLAMP_TO_EDGE)
                  .SetCompare(CompareMode::COMPARE_R_TO_TEXTURE, compareFunc);
    Sampler *sampler = samplerBuilder.CreateSampler();
    TextureHandle texHandle = mDevice->GetTextureHandle(shadowMap->GetRegisteredTextureID(),
                                                        sampler->GetRegisteredID());
    mQueueCB.BindProgram(mProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
    mQueueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    mQueueCB.DrawArrays(DrawPrimitive::POINTS, 0, 1);
    mQueueCB.submit();
    mQueue->Finish();
    sampler->Free();
    mGpuAllocator.freeTexture(shadowMap);
}

// Pixel Z and shadow map Z each vary in only one dimension, so only one coordinate from each is
// needed.
static int sampleShadowMap(int shadowX, int fboY, CompareFunc compareFunc)
{
    // shadowX may be interpolated from out of bounds, so clamp it.
    if (shadowX < 0) {
        shadowX = 0;
    } else if (shadowX > SHADOW_SIZE - 1) {
        shadowX = SHADOW_SIZE - 1;
    }
    float referenceZ = fboY / (FBO_SIZE - 1.0f);
    float textureZ = shadowX / (SHADOW_SIZE - 1.0f);
    switch (compareFunc) {
    case CompareFunc::NEVER:
        return 0;
    case CompareFunc::LESS:
        return referenceZ < textureZ;
    case CompareFunc::LEQUAL:
        return referenceZ <= textureZ;
    case CompareFunc::EQUAL:
        return referenceZ == textureZ;
    case CompareFunc::GREATER:
        return referenceZ > textureZ;
    case CompareFunc::NOTEQUAL:
        return referenceZ != textureZ;
    case CompareFunc::GEQUAL:
        return referenceZ >= textureZ;
    case CompareFunc::ALWAYS:
        return 1;
    default:
        DEBUG_PRINT(("Unrecognized CompareFunc: %#x\n", LWNcompareFunc(compareFunc)));
        LWNFailTest();
    }
    return 0;
}

static float calcExpected(int x, int y, MagFilter filter, CompareFunc compareFunc)
{
    float s = (x + 0.5f) / FBO_SIZE;
    if (filter == MagFilter::NEAREST) {
        return sampleShadowMap(int(s * SHADOW_SIZE), y, compareFunc);
    }
    // The shadow map only varies with X, so we only need 1-dimensional interpolation.
    int shadowX0 = int(floor(s * SHADOW_SIZE - 0.5));
    int shadowX1 = shadowX0 + 1;
    // FBO_SIZE is always 2 * SHADOW_SIZE, so fractional parts are well-defined.
    float fracX = (x % 2) ? 0.25f : 0.75f;
    return (1.0f - fracX) * sampleShadowMap(shadowX0, y, compareFunc) +
           fracX * sampleShadowMap(shadowX1, y, compareFunc);
}

bool Harness::verify(Format format, MagFilter filter, CompareFunc compareFunc)
{
    LWNsizei bufferSize = FBO_SIZE * FBO_SIZE * sizeof(vec4);
    MemoryPoolAllocator allocator(mDevice, NULL, bufferSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(mDevice).SetDefaults();
    Buffer *pbo = allocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, bufferSize);
    CopyRegion copyRegion = { 0, 0, 0, FBO_SIZE, FBO_SIZE, 1 };
    mQueueCB.CopyTextureToBuffer(mFbo.getColorTexture(0), 0, &copyRegion, pbo->GetAddress(), CopyFlags::NONE);
    mQueueCB.submit();
    mQueue->Finish();
    vec4* pixel = static_cast<vec4*>(pbo->Map());
    for (int y = 0; y < FBO_SIZE; ++y) {
        for (int x = 0; x < FBO_SIZE; ++x) {
            float expected = calcExpected(x, y, filter, compareFunc);
            if (fabs(expected - pixel->x()) > 0.01) {
                DEBUG_PRINT(("Failure at (%d, %d): Expected %f, got %f\n", x, y, expected,
                             pixel->x()));
                DEBUG_PRINT(("Format: %#x, Filter: %#x, CompareFunc: %#x.\n", LWNformat(format),
                             LWNmagFilter(filter), LWNcompareFunc(compareFunc)));
                allocator.freeBuffer(pbo);
                return false;
            }
            ++pixel;
        }
    }
    allocator.freeBuffer(pbo);
    return true;
}

bool Harness::run(Format format, MagFilter filter, CompareFunc compareFunc)
{
    render(format, filter, compareFunc);
    return verify(format, filter, compareFunc);
}

} // namespace

class LWNShadowMapTest {
public:
    LWNTEST_CppMethods();
};

lwString LWNShadowMapTest::getDescription() const
{
    return "Test for shadow map sampling. Generate a depth texture and test it against a range of "
           "reference values. Ranges for both the texture and the reference values are in [0, 1]. "
           "The values 0 and 1 are included, because comparison functions are sensitive to "
           "equality. Results are compared to expected values computed on the CPU. Each cell in "
           "the test represents one combination of comparison function, depth format,  and "
           "magnification filter. Output is red/green.";
}

int LWNShadowMapTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(21, 6);
}

void LWNShadowMapTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    queueCB.ClearColor(0, 0.0, 0.0, 0.25, 1.0);
    CompareFunc compareFuncs[] = {
        CompareFunc::NEVER,
        CompareFunc::LESS,
        CompareFunc::EQUAL,
        CompareFunc::LEQUAL,
        CompareFunc::GREATER,
        CompareFunc::NOTEQUAL,
        CompareFunc::GEQUAL,
        CompareFunc::ALWAYS,
    };
    int numColumns = __GL_ARRAYSIZE(compareFuncs);
    MagFilter filters[] = {
        MagFilter::NEAREST,
        MagFilter::LINEAR,
    };
    int numFilters = __GL_ARRAYSIZE(filters);
    Format formats[] = {
        Format::DEPTH16,
        Format::DEPTH24,
        Format::DEPTH32F,
        Format::DEPTH24_STENCIL8,
        Format::DEPTH32F_STENCIL8,
    };
    int numFormats = __GL_ARRAYSIZE(formats);
    int numRows = numFilters * numFormats;
    cellTestInit(numColumns, numRows);
    Harness harness;
    for (int row = 0; row < numRows; ++row) {
        for (int column = 0; column < numColumns; ++column) {
            bool testResult = harness.run(formats[row % numFormats], filters[row / numFormats],
                                          compareFuncs[column]);
            g_lwnWindowFramebuffer.bind();
            SetCellViewportScissorPadded(queueCB, column, row, 1);
            if (testResult) {
                queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
            } else {
                queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
            }
        }
    }
    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNShadowMapTest, lwn_shadowmap_2d, );

