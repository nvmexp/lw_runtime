/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//
// lwn_texture_swizzle.cpp
//
// Simple test to verify that texture builder swizzling works.
//

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#define DEBUG_MODE 0
#if DEBUG_MODE
    #define DEBUG_PRINT(x) do { \
        printf x; \
        fflush(stdout); \
    } while (0)
#else
    #define DEBUG_PRINT(x)
#endif

#define TEXSWIZ_TEXSIZE 123

#ifndef ROUND_UP
    #define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))
#endif

using namespace lwn;
using namespace lwn::dt;

// Helper function to set a component vector
template <typename T>
static void SetVector(T* v, const float val[])
{
    v->setX(val[0]);
    v->setY(val[1]);
    v->setZ(val[2]);
    v->setW(val[3]);
}

static int GetFormatBPP(Format fmt)
{
    switch (fmt) {
        case Format::RGBA8: return 4;
        case Format::RGBX8: return 4;
        case Format::RGBA16: return 8;
        case Format::RGB10A2: return 4;
        case Format::RGBA32F: return 16;
        default:
            assert(!"Unsupported format.");
            return 0;
    }
}

#if DEBUG_MODE
static const char* SwizzleToStr(TextureSwizzle swiz)
{
    switch (swiz) {
        case TextureSwizzle::R: return "R";
        case TextureSwizzle::G: return "G";
        case TextureSwizzle::B: return "B";
        case TextureSwizzle::A: return "A";
        case TextureSwizzle::ONE: return "1";
        case TextureSwizzle::ZERO: return "0";
        default: return "?";
    }
}
#endif

const static uint32_t colors8[] = {
    0x800000FF, // R
    0x8000FF00, // G
    0x80FF0000, // B
    0xFF000000  // A
};
const static uint64_t colors16[] = {
    0x800000000000FFFFULL, // R
    0x80000000FFFF0000ULL, // G
    0x8000FFFF00000000ULL, // B
    0xFFFF000000000000ULL  // A
};
const static float colors32F[][4] = {
    { 1.0f, 0.0f, 0.0f, 0.5f },  // R
    { 0.0f, 1.0f, 0.0f, 0.5f },  // G
    { 0.0f, 0.0f, 1.0f, 0.5f },  // B
    { 0.0f, 0.0f, 0.0f, 1.0f }   // A
};

// ----------------------------------- LWNTextureSwizzle ------------------------------------------


class LWNTextureSwizzle {
    struct Vertex {
        vec3 position;
        vec3 uv;
    };

    Program *m_program;
    VertexArrayState m_vertexState;
    Buffer *m_vbo;
    MemoryPoolAllocator* m_pool;
    MemoryPoolAllocator* m_bufpool;
    Sampler *m_sampler;
    LWNuint m_vertexDataSize;
    uint32_t* m_screenBufMap;

    void fill(Buffer *buffer, Format fmt);
    vec4 getScreenBufPixel(int x, int y);

public:
    LWNTextureSwizzle();
    ~LWNTextureSwizzle();

    bool init(void);
    void draw(const Format fmt, const TextureSwizzle swizzle[]);
    bool verify(const Format fmt, const TextureSwizzle swizzle[], int rect[]);
};

LWNTextureSwizzle::LWNTextureSwizzle()
        : m_pool(NULL), m_bufpool(NULL)
{}

LWNTextureSwizzle::~LWNTextureSwizzle()
{
    delete m_pool;
    delete m_bufpool;
}

bool LWNTextureSwizzle::init(void)
{
    Device *device = DeviceState::GetActive()->getDevice();

    // Create mempool.
    const LWNsizeiptr poolSize = 16 * 1024 * 1024;
    m_pool = new MemoryPoolAllocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    m_bufpool = new MemoryPoolAllocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Compile shaders.
    // The frag shader shows alpha using a small checkerboard pattern.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 uv;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "out vec3 ouv;\n"
        "out vec3 opos;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ouv = uv;\n"
        "  opos = position;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec3 ouv;\n"
        "in vec3 opos;\n"
        "out vec4 fcolor;\n"
        "layout (binding=0) uniform sampler2D tex;"
        "void main() {\n"
        "  vec4 t = textureLod(tex, ouv.xy, 0.0);\n"
        "  float grid = float((int((opos.x + 1.0) * 10.0) % 2) ^ (int((opos.y + 1.0) * 10.0) % 2));\n"
        "  t.xyz = mix(vec3(grid), t.xyz, t.a);\n"
        "  t.a = 1.0;\n"
        "  fcolor = t;\n"
        "}\n";
    m_program = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_program, vs, fs)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return false;
    }

    // Create vertex data.
    const int vertexCount = 4;
    static const Vertex vertexData[] = {
        { vec3(-1, -1, 0.0), vec3(0.0, 1.0, 0.0) },
        { vec3(+1, -1, 0.0), vec3(1.0, 1.0, 0.0) },
        { vec3(+1, +1, 0.0), vec3(1.0, 0.0, 0.0) },
        { vec3(-1, +1, 0.0), vec3(0.0, 0.0, 0.0) }
    };
    m_vertexDataSize = sizeof(vertexData);
    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, uv);
    m_vertexState = vertexStream.CreateVertexArrayState();
    m_vbo = vertexStream.AllocateVertexBuffer(device, vertexCount, *m_bufpool, vertexData);

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults()
      .SetMinMagFilter(MinFilter::NEAREST_MIPMAP_NEAREST, MagFilter::NEAREST);
    m_sampler = sb.CreateSampler();
    m_screenBufMap = NULL;
    return true;
}

void LWNTextureSwizzle::fill(Buffer *buffer, Format fmt)
{
    // Blit some test data into the texture.
    // Creates a large 3x3 semi-transparent checkerboard pattern, with each tile rotating between
    // R, G, B, A. Should look something like:
    //
    //                                  R G B
    //                                  A R G
    //                                  B A R
    //
    // NOTE: The fragment shader alpha-blends the texture on top of a small checkerboard pattern.
    //

    union TexData {
        void* ptr;
        uint32_t* u32;
        uint64_t* u64;
        vec4_rgb10a2* rgb10a2;
        vec4* v4;
    };
    TexData data;
    data.ptr = (void*) buffer->Map();
    assert(data.ptr);

    for (int x = 0; x < TEXSWIZ_TEXSIZE; x++) {
        for (int y = 0; y < TEXSWIZ_TEXSIZE; y++) {
            int xgrid = (3 * x) / TEXSWIZ_TEXSIZE;
            int ygrid = (3 * y) / TEXSWIZ_TEXSIZE;
            int tile = (xgrid * 3 + ygrid) % 4;\

            switch (fmt) {
            case Format::RGBA8:
                // Fallthrough.
            case Format::RGBX8:
                data.u32[x * TEXSWIZ_TEXSIZE + y] = colors8[tile];
                break;
            case Format::RGBA16:
                data.u64[x * TEXSWIZ_TEXSIZE + y] = colors16[tile];
                break;
            case Format::RGB10A2:
                SetVector<vec4_rgb10a2>(&data.rgb10a2[x * TEXSWIZ_TEXSIZE + y], colors32F[tile]);
                break;
            case Format::RGBA32F:
                SetVector<vec4>(&data.v4[x * TEXSWIZ_TEXSIZE + y], colors32F[tile]);
                break;
            default:
                assert(!"Texture format not supported by this test.");
                break;
            }

        }
    }
}

void LWNTextureSwizzle::draw(const Format fmt, const TextureSwizzle swizzle[])
{
    Device *device = DeviceState::GetActive()->getDevice();
    QueueCommandBuffer &queueCB = DeviceState::GetActive()->getQueueCB();

    // Create texture.
    const int texsz = TEXSWIZ_TEXSIZE * TEXSWIZ_TEXSIZE * GetFormatBPP(fmt);
    TextureBuilder textureBuilder;
    textureBuilder.SetDevice(device)
                  .SetDefaults()
                  .SetTarget(TextureTarget::TARGET_2D)
                  .SetFormat(fmt)
                  .SetSwizzle(swizzle[0], swizzle[1], swizzle[2], swizzle[3])
                  .SetSize2D(TEXSWIZ_TEXSIZE, TEXSWIZ_TEXSIZE);
    Texture *texture = m_pool->allocTexture(&textureBuilder);
    TextureHandle texHandle = device->GetTextureHandle(texture->GetRegisteredTextureID(), m_sampler->GetRegisteredID());

    BufferBuilder bufferBuilder;
    bufferBuilder.SetDevice(device);
    bufferBuilder.SetDefaults();
    Buffer *texbuf = m_bufpool->allocBuffer(&bufferBuilder, BUFFER_ALIGN_COPY_READ_BIT, texsz);
    fill(texbuf, fmt);
    CopyRegion copyRegion = { 0, 0, 0, TEXSWIZ_TEXSIZE, TEXSWIZ_TEXSIZE, 1 };
    queueCB.CopyBufferToTexture(texbuf->GetAddress(), texture, NULL, &copyRegion, CopyFlags::NONE);

    queueCB.BindVertexArrayState(m_vertexState);
    queueCB.BindProgram(m_program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
}

vec4 LWNTextureSwizzle::getScreenBufPixel(int x, int y)
{
    assert(m_screenBufMap);
    uint32_t px = m_screenBufMap[y * lwrrentWindowWidth + x];
    return vec4(
        (float)(px       & 0xFF) / (float) 0xFF,
        (float)(px >>  8 & 0xFF) / (float) 0xFF,
        (float)(px >> 16 & 0xFF) / (float) 0xFF,
        (float)(px >> 24 & 0xFF) / (float) 0xFF
    );
}

bool LWNTextureSwizzle::verify(const Format fmt, const TextureSwizzle swizzle[], int rect[])
{
    Device *device = DeviceState::GetActive()->getDevice();

    if (!m_screenBufMap) {
        // Screen will be read on the first verify.
        const int screensz = lwrrentWindowWidth * lwrrentWindowHeight * 4 /* RGBA8 */;
        BufferBuilder bufferBuilder;
        bufferBuilder.SetDevice(device);
        bufferBuilder.SetDefaults();
        Buffer *screenbuf = m_bufpool->allocBuffer(&bufferBuilder, BUFFER_ALIGN_COPY_WRITE_BIT, screensz);
        g_lwnWindowFramebuffer.readPixels(screenbuf);
        m_screenBufMap = (uint32_t*) screenbuf->Map();
        assert(m_screenBufMap);
    }

    // Callwlate expected swizzled samples.
    const char textureData[] = "RGB"
                               "ARG"
                               "BAR";
    const int textureDataSz = 3 * 3;

    vec4 samples[textureDataSz];
    vec4 swizzledSamples[textureDataSz];
    for (int i = 0; i < textureDataSz; i++) {
        switch (textureData[i]) {
        case 'R':
            samples[i] = vec4(colors32F[0][0], colors32F[0][1], colors32F[0][2], colors32F[0][3]);
            break;
        case 'G':
            samples[i] = vec4(colors32F[1][0], colors32F[1][1], colors32F[1][2], colors32F[1][3]);
            break;
        case 'B':
            samples[i] = vec4(colors32F[2][0], colors32F[2][1], colors32F[2][2], colors32F[2][3]);
            break;
        case 'A':
            samples[i] = vec4(colors32F[3][0], colors32F[3][1], colors32F[3][2], colors32F[3][3]);
            break;
        default:
            assert(!"Unexpected texture data.");
            break;
        }
    }
    if (fmt == Format::RGBX8) {
        // This format is not capable of storing alpha.
        for (int i = 0; i < textureDataSz; i++) {
            samples[i][3] = 1.0f;
        }
    }
    if (fmt == Format::RGB10A2) {
        // This format is not capable of storing 0.5 alpha.
        // It stores it as 0.66666... instead.
        for (int i = 0; i < textureDataSz; i++) {
            if (fabs(samples[i][3] - 0.5f) < 0.01f) {
                samples[i][3] = 0.66666667f;
            }
        }
    }
    for (int i = 0; i < textureDataSz; i++) {
        for (int j = 0; j < 4; j++) {
            float value = 0.0f;
            switch (swizzle[j]) {
            case TextureSwizzle::R:
                value = samples[i][0];
                break;
            case TextureSwizzle::G:
                value = samples[i][1];
                break;
            case TextureSwizzle::B:
                value = samples[i][2];
                break;
            case TextureSwizzle::A:
                value = samples[i][3];
                break;
            case TextureSwizzle::ONE:
                value = 1.0f;
                break;
            case TextureSwizzle::ZERO:
                value = 0.0f;
                break;
            default:
                assert(!"Unexpected swizzle value.\n");
                break;
            }
            swizzledSamples[i][j] = value;
        }
    }

    // Take samples of each swizzle grid.
    // Read in a left-to-right, top-to-bottom fashion.
    float gridSizeX = (float) rect[2] / 3.0f, gridSizeY = (float) rect[3] / 3.0f;

    for (int sampleY = 0; sampleY <= 2; sampleY++) {
        for (int sampleX = 0; sampleX <= 2; sampleX++) {

            int pixelX = rect[0] + (int)(((float) sampleX + 0.5f) * gridSizeX);
            int pixelY = rect[1] + rect[3] - (int)(((float) sampleY + 0.5f) * gridSizeY);

            vec4 sample = getScreenBufPixel(pixelX, pixelY);
            DEBUG_PRINT(("    sample[%d][%d] = %.2f %.2f %.2f %.2f\n", sampleX, sampleY,
                    sample[0], sample[1], sample[2], sample[3]));

            int off = sampleY * 3 + sampleX;
            vec4 expected = swizzledSamples[off];
            vec4 expected2 = swizzledSamples[off];

            // Take into account that alpha is shown as blending with a grid.
            // Assuming alpha = 1.0 or alpha = 0.5.
            if (expected[3] < 1.0f) {

                // Match either the white blended grid or the black blended grid.

                expected[0] *= expected[3];
                expected[1] *= expected[3];
                expected[2] *= expected[3];

                expected2[0] *= expected[3]; expected2[0] += (1.0f - expected[3]);
                expected2[1] *= expected[3]; expected2[1] += (1.0f - expected[3]);
                expected2[2] *= expected[3]; expected2[2] += (1.0f - expected[3]);

                expected[3] = 1.0f;
                expected2[3] = 1.0f;
            }

            DEBUG_PRINT(("    expected[%d][%d] = %.2f %.2f %.2f %.2f\n", sampleX, sampleY,
                    expected[0], expected[1],
                    expected[2], expected[3]));
            DEBUG_PRINT(("    expected2[%d][%d] = %.2f %.2f %.2f %.2f\n", sampleX, sampleY,
                    expected2[0], expected2[1],
                    expected2[2], expected2[3]));

            float delta = fabs(sample[0] - expected[0]) +
                          fabs(sample[1] - expected[1]) +
                          fabs(sample[2] - expected[2]) +
                          fabs(sample[3] - expected[3]);
            float delta2 = fabs(sample[0] - expected2[0]) +
                           fabs(sample[1] - expected2[1]) +
                           fabs(sample[2] - expected2[2]) +
                           fabs(sample[3] - expected2[3]);
            if (delta > 0.02f && delta2 > 0.02f) {
                DEBUG_PRINT(("    MISMATCH DETECTED: delta %f delta2 %f.\n", delta, delta2));
                return false;
            }
        }
    }

    return true;
}

// --------------------------------- LWNTextureSwizzleTest ----------------------------------------

class LWNTextureSwizzleTest {
    static const int cellSize = 48;
    static const int cellMargin = 3;

public:
    LWNTEST_CppMethods();
};

lwString LWNTextureSwizzleTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Simple LWN texture builder swizzle test.\n"
        "Should see a 3x3 grid texture with the pattern:\n"
        "\n"
        "    R G B\n"
        "    A R G\n"
        "    B A R\n"
        "\n"
        "Swizzled with various swizzle parameters using different formats.\n"
        "Each format takes up 2 rows; the test self-checks the results against expected values.\n"
        "The last cell in each row is the self-checking result (green = good, red = bad).";
    return sb.str();
}

int LWNTextureSwizzleTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(20, 0);
}

void LWNTextureSwizzleTest::doGraphics() const
{
    static const int cellsX = lwrrentWindowWidth / cellSize;
    static const int cellsY = lwrrentWindowHeight / cellSize;
    CellIterator2D cell(cellsX, cellsY);

    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.25, 0.25, 0.25, 1.0);

    LWNTextureSwizzle swizTest;
    bool ret = swizTest.init();
    if (!ret) {
        LWNFailTest();
        return;
    }

#define MKSWZ(SR, SG, SB, SA) \
        { TextureSwizzle::SR, TextureSwizzle::SG, TextureSwizzle::SB, TextureSwizzle::SA },
    const static TextureSwizzle textureSwizzleCases[][4] = {

        MKSWZ(R, G, B, A)                   // Default swizzle.
        MKSWZ(R, G, B, ONE)                 // Default swizzle, no alpha.
        MKSWZ(ZERO, ZERO, ZERO, ZERO)       // All transparent (black).
        MKSWZ(ZERO, ZERO, ZERO, ONE)        // All black.
        MKSWZ(ONE, ONE, ONE, ONE)           // All white.
        MKSWZ(R, R, R, ONE)                 // All color channels set to red channel.
        MKSWZ(G, G, G, ONE)                 // All color channels set to green channel.
        MKSWZ(B, B, B, ONE)                 // All color channels set to blue channel.
        MKSWZ(A, A, A, ONE)                 // All color channels set to alpha channel.
        MKSWZ(ONE, ONE, ONE, R)             // Red as alpha, all white.
        MKSWZ(ONE, ONE, ONE, G)             // Green as alpha, all white.
        MKSWZ(ONE, ONE, ONE, B)             // Blue as alpha, all white.
        MKSWZ(ONE, ONE, ONE, A)             // Alpha as alpha, all white.
        MKSWZ(R, ZERO, ZERO, A)             // Only red.
        MKSWZ(ZERO, G, ZERO, A)             // Only green.
        MKSWZ(ZERO, ZERO, B, A)             // Only blue.
        MKSWZ(R, ZERO, ZERO, ONE)           // Only red (no alpha).
        MKSWZ(ZERO, G, ZERO, ONE)           // Only green (no alpha).
        MKSWZ(ZERO, ZERO, B, ONE)           // Only blue (no alpha).
        MKSWZ(B, G, R, A)                   // Reversed colors.
        MKSWZ(G, R, B, A)                   // Rotated colors.
        MKSWZ(B, R, G, A)                   // Rotated colors.
        MKSWZ(B, R, G, ONE)                 // Rotated colors (no alpha).
        MKSWZ(G, B, R, A)                   // Rotated colors.
        MKSWZ(G, B, R, ONE)                 // Rotated colors (no alpha).
    };
    int textureSwizzleNumCases = sizeof(textureSwizzleCases) / sizeof(textureSwizzleCases[0]);
#undef MKSWZ

#define GET_VIEWPORT(CELL) \
        int viewport[] = { \
            CELL.x() * cellSize + cellMargin, \
            CELL.y() * cellSize + cellMargin, \
            cellSize - 2 * cellMargin, \
            cellSize - 2 * cellMargin \
        }; \

    const static Format textureSwizzleFormats[] = {
        Format::RGBA8,
        Format::RGBX8,
        Format::RGBA16,
        Format::RGB10A2,
        Format::RGBA32F,
    };
    int textureSwizzleNumFormats = sizeof(textureSwizzleFormats) / sizeof(textureSwizzleFormats[0]);

    for (int fmt = 0; fmt < textureSwizzleNumFormats; fmt++) {
        for (int i = 0; i < textureSwizzleNumCases; i++) {

            DEBUG_PRINT(("--- rendering case: %s%s%s%s ---- \n",
                SwizzleToStr(textureSwizzleCases[i][0]),
                SwizzleToStr(textureSwizzleCases[i][1]),
                SwizzleToStr(textureSwizzleCases[i][2]),
                SwizzleToStr(textureSwizzleCases[i][3])
            ));

            assert(i < cellsX * cellsY);
            GET_VIEWPORT(cell);
            queueCB.SetViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
            queueCB.SetScissor(viewport[0], viewport[1], viewport[2], viewport[3]);
            swizTest.draw(textureSwizzleFormats[fmt], textureSwizzleCases[i]);
            cell++;

        }
        // Skip onto next row on format change.
        int nextRow = ROUND_UP(textureSwizzleNumCases, cellsX);
        for (int skip = textureSwizzleNumCases; skip < nextRow; skip++) {
            cell++;
        }
    }

    queueCB.submit();
    queue->Finish();

    // Verify that the tests were drawn properly.
    CellIterator2D vcell(cellsX, cellsY);
    for (int fmt = 0; fmt < textureSwizzleNumFormats; fmt++) {

        DEBUG_PRINT(("\n\n------------------ verifying format %d\n", fmt));
        bool failure = false;
        for (int i = 0; i < textureSwizzleNumCases; i++) {

            DEBUG_PRINT(("--- verifying case: %s%s%s%s ---- \n",
                SwizzleToStr(textureSwizzleCases[i][0]),
                SwizzleToStr(textureSwizzleCases[i][1]),
                SwizzleToStr(textureSwizzleCases[i][2]),
                SwizzleToStr(textureSwizzleCases[i][3])
            ));

            GET_VIEWPORT(vcell);
            bool result = swizTest.verify(textureSwizzleFormats[fmt], textureSwizzleCases[i], viewport);
            if (!result) {
                failure = true;
            }

            vcell++;
        }
        // Fill in the rest of the row with pass / fail.
        int nextRow = ROUND_UP(textureSwizzleNumCases, cellsX);
        for (int skip = textureSwizzleNumCases; skip < nextRow; skip++) {
            GET_VIEWPORT(vcell);
            queueCB.SetViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
            queueCB.SetScissor(viewport[0], viewport[1], viewport[2], viewport[3]);
            if (failure) {
                queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
            } else {
                queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
            }
            vcell++;
        }
    }

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNTextureSwizzleTest, lwn_texture_swizzle, );

