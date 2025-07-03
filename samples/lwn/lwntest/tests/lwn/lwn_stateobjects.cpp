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
#include "string.h"

/**********************************************************************/
// LWN State object tests.
// Tests LWN blend objects.
//
// Portions of the blend test have been liberally borrowed from
// ES31DrawBuffersIndexed.
//
// Areas for future enhancement:
// - Test an RGB format (no alpha channel)

#define DEBUG_MODE 0

#if DEBUG_MODE
#define DEBUG_PRINT(x) do { \
    printf x; \
    fflush(stdout); \
} while (0)
#else
#define DEBUG_PRINT(x)
#endif


/**********************************************************************/

using namespace lwn;


static uint8_t* getRandom(int len)
{
    uint8_t *mem = new uint8_t[len];
    for (int i=0; i < len; i++)
        mem[i] = lwIntRand(0, 255);
    return mem;
}


/**********************************************************************/
// Blending State Evaluator
/**********************************************************************/


static void clampColor(LWNfloat color[4])
{
    for (int c = 0; c < 4; c++) {
        color[c] = LW_MIN(LW_MAX(color[c], 0.0f), 1.0f);
    }
}

static void colwertColorUByteToFloat(const uint8_t in[4], LWNfloat out[4])
{
    for (int c = 0; c < 4; c++) {
        out[c] = ((float)in[c]) / 255.0f;
    }
}

static void colwertColorFloatToUByte(const LWNfloat in[4], uint8_t out[4])
{
    for (int c = 0; c < 4; c++) {
        out[c] = uint8_t((in[c] * 255.0f) + 0.5f);
    }
}

static void blendFactor(BlendFunc func, const LWNfloat bColor[4],
                        const LWNfloat sColor[4], const LWNfloat sColor1[4], const LWNfloat dColor[4],
                        LWNfloat out[4])
{
    for (int c = 0; c < 4; c++) {
        switch (func) {
        case BlendFunc::ZERO:
            out[c] = 0.0f;
            break;
        case BlendFunc::ONE:
            out[c] = 1.0f;
            break;
        case BlendFunc::SRC_COLOR:
            out[c] = sColor[c];
            break;
        case BlendFunc::ONE_MINUS_SRC_COLOR:
            out[c] = 1.0f - sColor[c];
            break;
        case BlendFunc::DST_COLOR:
            out[c] = dColor[c];
            break;
        case BlendFunc::ONE_MINUS_DST_COLOR:
            out[c] = 1.0f - dColor[c];
            break;
        case BlendFunc::SRC_ALPHA:
            out[c] = sColor[3];
            break;
        case BlendFunc::ONE_MINUS_SRC_ALPHA:
            out[c] = 1.0f - sColor[3];
            break;
        case BlendFunc::DST_ALPHA:
            out[c]= dColor[3];
            break;
        case BlendFunc::ONE_MINUS_DST_ALPHA:
            out[c]= 1.0f - dColor[3];
            break;
        case BlendFunc::SRC_ALPHA_SATURATE:
            {
                if (c == 3) {
                    out[3] = 1.0f;
                } else {
                    out[c] = LW_MIN(sColor[3], 1.0f - dColor[3]);
                }
            }
            break;
        case BlendFunc::CONSTANT_COLOR:
                out[c] = bColor[c];
            break;
        case BlendFunc::ONE_MINUS_CONSTANT_COLOR:
                out[c] = 1.0f - bColor[c];
            break;
        case BlendFunc::CONSTANT_ALPHA:
                out[c] = bColor[3];
            break;
        case BlendFunc::ONE_MINUS_CONSTANT_ALPHA:
                out[c] = 1.0f - bColor[3];
            break;
        case BlendFunc::SRC1_COLOR:
                out[c] = sColor1[c];
            break;
        case BlendFunc::ONE_MINUS_SRC1_COLOR:
                out[c] = 1.0f - sColor1[c];
            break;
        case BlendFunc::SRC1_ALPHA:
                out[c] = sColor1[3];
            break;
        case BlendFunc::ONE_MINUS_SRC1_ALPHA:
                out[c] = 1.0f - sColor1[3];
            break;
        default:
            assert(0);
            break;
        }
    }

    clampColor(out);
}

static void colwertRGBToSRGB(LWNfloat color[4])
{
    for (int i = 0; i < 3; i++)
    {
        if (color[i] <= 0.0031308)
        {
            color[i] *= 12.92;
        }
        else
        {
            color[i] = 1.055 * pow((double)color[i], 1.0/2.4) - 0.055;
        }
    }
}

static void colwertSRGBToRGB(LWNfloat color[4])
{
    for (int i = 0; i < 3; i++)
    {
        if (color[i] <= 0.04045)
        {
            color[i] = color[i] / 12.92;
        }
        else
        {
            color[i] = pow(((color[i] + 0.055)/1.055), 2.4);
        }
    }
}

static void blendColor(BlendEquation eq, BlendFunc sFunc, BlendFunc dFunc, bool sRGB, const LWNfloat bColor[4],
                       const uint8_t dColor[4], const uint8_t sColor[4], const uint8_t sColor1[4],
                       uint8_t out[4])
{
    LWNfloat sC[4], sC1[4], dC[4];
    LWNfloat sR[4], dR[4];
    LWNfloat o[4];

    colwertColorUByteToFloat(sColor, sC);
    colwertColorUByteToFloat(sColor1, sC1);
    colwertColorUByteToFloat(dColor, dC);

    if (sRGB) {
        colwertSRGBToRGB(sC);
        colwertSRGBToRGB(sC1);
        colwertSRGBToRGB(dC);
    }

    blendFactor(sFunc, bColor, sC, sC1, dC, sR);
    blendFactor(dFunc, bColor, sC, sC1, dC, dR);

    for (int c = 0; c < 4; c++) {
        switch (eq) {
        case BlendEquation::ADD:
            o[c] = (sC[c] * sR[c]) + (dC[c] * dR[c]);
            break;
        case BlendEquation::SUB:
            o[c] = (sC[c] * sR[c]) - (dC[c] * dR[c]);
            break;
        case BlendEquation::REVERSE_SUB:
            o[c] = (dC[c] * dR[c]) - (sC[c] * sR[c]);
            break;
        case BlendEquation::MIN:
            o[c] = LW_MIN(dC[c], sC[c]);
            break;
        case BlendEquation::MAX:
            o[c] = LW_MAX(dC[c], sC[c]);
            break;
        default:
            assert(0);
            break;
        }
    }

    clampColor(o);

    if (sRGB) {
        colwertRGBToSRGB(o);
    }

    colwertColorFloatToUByte(o, out);
}

static bool checkBlendResult(BlendEquation eq, BlendFunc src, BlendFunc dst, bool sRGB,
                             const LWNfloat blendColorConst[4],
                             const uint8_t *memDst, const uint8_t *memSrc0, const uint8_t *memSrc1,
                             const uint8_t *res)
{
    const int DELTA = 5;
    const int SRGB_DELTA = 16;
    bool valid = true;
    uint8_t expected[4];

    blendColor(eq, src, dst, sRGB, blendColorConst, memDst, memSrc0, memSrc1, expected);
    for (int c = 0; c < 4; c++) {
        int variance = abs((int)expected[c] - (int)res[c]);
        if (sRGB && (eq == BlendEquation::SUB || eq == BlendEquation::REVERSE_SUB)) {
            // Error can be more pronounced with the subtract blend equations
            valid = valid && (variance < SRGB_DELTA);
        } else {
            valid = valid && (variance < DELTA);
        }
    }
    return valid;
}


/**********************************************************************/
// Blending State Test
/**********************************************************************/

static const char *vsDrawSrc =
    "#version 440 core\n"
    "layout(location = 0) in vec4 position;\n"
    "layout(location = 1) in vec2 texcoord;\n"
    "layout(binding = 0) uniform Block {\n"
    "    vec2 scale;\n"
    "};\n"
    "out IO { vec2 tc; };\n"
    "void main() {\n"
    "  gl_Position = position;\n"
    "  tc = texcoord * scale;\n"
    "}\n";

// Needs to output color1 so we can test the SRC1 blend functions
static const char *fsDraw2Src =
    "#version 440 core\n"
    "layout(binding = 0) uniform sampler2D tex0;\n"
    "layout(binding = 1) uniform sampler2D tex1;\n"
    "layout(location = 0) out vec4 color0;\n"
    "layout(location = 1) out vec4 color1;\n"
    "in IO { vec2 tc; };\n"
    "void main() {\n"
    "  color0 = texture(tex0, tc);\n"
    "  color1 = texture(tex1, tc);\n"
    "}\n";


// Test data
static const BlendFunc blendFunctions[] = {
    BlendFunc::ZERO,
    BlendFunc::ONE,
    BlendFunc::SRC_COLOR,
    BlendFunc::ONE_MINUS_SRC_COLOR,
    BlendFunc::DST_COLOR,
    BlendFunc::ONE_MINUS_DST_COLOR,
    BlendFunc::SRC_ALPHA,
    BlendFunc::ONE_MINUS_SRC_ALPHA,
    BlendFunc::DST_ALPHA,
    BlendFunc::ONE_MINUS_DST_ALPHA,
    BlendFunc::SRC_ALPHA_SATURATE,
    BlendFunc::CONSTANT_COLOR,
    BlendFunc::ONE_MINUS_CONSTANT_COLOR,
    BlendFunc::CONSTANT_ALPHA,
    BlendFunc::ONE_MINUS_CONSTANT_ALPHA,
    BlendFunc::SRC1_COLOR,
    BlendFunc::ONE_MINUS_SRC1_COLOR,
    BlendFunc::SRC1_ALPHA,
    BlendFunc::ONE_MINUS_SRC1_ALPHA,
};
#define BLEND_FUNC_COUNT __GL_ARRAYSIZE(blendFunctions)

static const BlendEquation blendEquations[] = {
    BlendEquation::ADD,
    BlendEquation::MIN,
    BlendEquation::MAX,
    BlendEquation::SUB,
    BlendEquation::REVERSE_SUB,
};
#define BLEND_EQUATION_COUNT __GL_ARRAYSIZE(blendEquations)


// Implementation of LWN Blend tests
class LwnBlendTest {
protected:
    static const unsigned int testWidth = 8;
    static const unsigned int testHeight = 8;

    bool multisample, sRGB;

public:
    LWNTEST_CppMethods();

    LwnBlendTest(bool multisample, bool sRGB) : multisample(multisample), sRGB(sRGB) {}
};


int LwnBlendTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(32,1);
}

lwString LwnBlendTest::getDescription() const
{
    return
        "Test LWN blend state for a single rendertarget.\n"
        "We test an 8x8 set of texels using each combination, and display a result on the screen that\n"
        "gives an overall health for that set. Green=good, Red=at least one failure, Blue=unsupported.\n"
        "On the x axis is each possible source function\n"
        "On the y axis is each possible destination function * the number of possible blend equations\n"
        "We allow for a small margin of error between generated and expected results.\n";
}

void LwnBlendTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &cmd = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    cellTestInit(BLEND_FUNC_COUNT, BLEND_FUNC_COUNT * BLEND_EQUATION_COUNT);

    // We will test all permutations of blend state, each holding
    // a testWidth * testHeight set of pixels.
    int fbWidth  = testWidth * BLEND_FUNC_COUNT;
    int fbHeight = testHeight * BLEND_FUNC_COUNT * BLEND_EQUATION_COUNT;

    // allocator will create pool at first allocation
    // make coherent pool same size as texture pool (texture copies during texture allocation)
    const LWNsizeiptr coherent_poolsize = 0x100000UL;
    // safe guess, if the textures don't fit (in a future modification of this test) we'd notice allocation failures
    const LWNsizeiptr tex_poolsize = 0x100000UL;
    MemoryPoolAllocator allocator(device, NULL, coherent_poolsize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    MemoryPoolAllocator tex_allocator(device, NULL, tex_poolsize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    // init programs
    Program *pgm = device->CreateProgram();
    VertexShader vs(440);
    vs << vsDrawSrc;
    FragmentShader fs(440);
    fs << fsDraw2Src;


    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return;
    }
    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec2 texcoord;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec2(0.0, 0.0) },
        { dt::vec3( 1.0, -1.0, 0.0), dt::vec2(1.0, 0.0) },
        { dt::vec3(-1.0,  1.0, 0.0), dt::vec2(0.0, 1.0) },
        { dt::vec3( 1.0,  1.0, 0.0), dt::vec2(1.0, 1.0) },
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, texcoord);
    VertexArrayState vstate = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexData), allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Set up index buffer
    static const uint16_t indexData[] = {0, 1, 2, 1, 3, 2};
    Buffer *ibo = AllocAndFillBuffer(device, queue, cmd, allocator, indexData, sizeof(indexData),
                                               BUFFER_ALIGN_INDEX_BIT, false);
    BufferAddress iboAddr = ibo->GetAddress();

    // init textures
    Format testFormat = sRGB ? Format::RGBA8_SRGB : Format::RGBA8;
    uint8_t *memDst = getRandom(testWidth*testHeight*4);
    uint8_t *memSrc0 = getRandom(testWidth*testHeight*4);
    uint8_t *memSrc1 = getRandom(testWidth*testHeight*4);
    Texture *texDst = AllocAndFillTexture2D(device, queue, cmd, tex_allocator, allocator, memDst, 4, testWidth, testHeight, testFormat);
    Texture *texSrc0 = AllocAndFillTexture2D(device, queue, cmd, tex_allocator, allocator, memSrc0, 4, testWidth, testHeight, testFormat);
    Texture *texSrc1 = AllocAndFillTexture2D(device, queue, cmd ,tex_allocator, allocator, memSrc1, 4, testWidth, testHeight, testFormat);

    // init sampler
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetWrapMode(WrapMode::REPEAT, WrapMode::REPEAT, WrapMode::REPEAT)
      .SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler *sampler = sb.CreateSampler();
    LWNuint samplerID = sampler->GetRegisteredID();

    // combined texture/sampler handles
    TextureHandle texDstHandle = device->GetTextureHandle(texDst->GetRegisteredTextureID(), samplerID);
    TextureHandle texSrc0Handle = device->GetTextureHandle(texSrc0->GetRegisteredTextureID(), samplerID);
    TextureHandle texSrc1Handle = device->GetTextureHandle(texSrc1->GetRegisteredTextureID(), samplerID);

    // ubos
    LWNfloat uboScaleData[] = { BLEND_FUNC_COUNT, BLEND_FUNC_COUNT * BLEND_EQUATION_COUNT };
    Buffer *uboScale = AllocAndFillBuffer(device, queue, cmd, allocator, uboScaleData, sizeof(uboScaleData),
                                          BUFFER_ALIGN_UNIFORM_BIT, false);

    LWNfloat uboIdentityData[] = { 1.0, 1.0 };
    Buffer *uboIdentity = AllocAndFillBuffer(device, queue, cmd, allocator, uboIdentityData, sizeof(uboIdentityData),
                                             BUFFER_ALIGN_UNIFORM_BIT, false);

    // init and set rendertarget
    Framebuffer fbResult(fbWidth, fbHeight);
    fbResult.setFlags(TextureFlags::COMPRESSIBLE);
    fbResult.setColorFormat(0, testFormat);
    fbResult.setSamples(multisample ? 4 : 0);
    fbResult.alloc(device);

    fbResult.bind(cmd);
    cmd.SetViewportScissor(0, 0, fbWidth, fbHeight);

    // Fill the backbuffer with the contents of texDst, repeating the pattern
    // once for each cell. (We use the repeat wrap mode, and adjust the texture
    // coordinates with a scaling factor, which allows us to cover every cell
    // in a single draw call.)

    MultisampleState msState;
    msState.SetDefaults();
    msState.SetSamples(multisample ? 4 : 0);

    BlendState blend;
    ColorState color;
    blend.SetDefaults();
    color.SetDefaults();

    cmd.BindBlendState(&blend);
    cmd.BindColorState(&color);
    cmd.BindMultisampleState(&msState);
    cmd.BindVertexArrayState(vstate);
    cmd.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, uboScale->GetAddress(), 2*sizeof(float));
    cmd.BindTexture(ShaderStage::FRAGMENT, 0, texDstHandle);
    cmd.BindTexture(ShaderStage::FRAGMENT, 1, texDstHandle);
    cmd.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    cmd.DrawElements(DrawPrimitive::TRIANGLES, IndexType::UNSIGNED_SHORT, 6, iboAddr);

    // Set up state to test each cell with a separate draw call
    cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, uboIdentity->GetAddress(), 2*sizeof(float));
    cmd.BindTexture(ShaderStage::FRAGMENT, 0, texSrc0Handle);
    cmd.BindTexture(ShaderStage::FRAGMENT, 1, texSrc1Handle);

    // Set blend color constant
    const LWNfloat blendColorConst[] = {0.0, 0.5, 0.25, 0.75};
    cmd.SetBlendColor(blendColorConst);
    color.SetBlendEnable(0, true);
    cmd.BindColorState(&color);

    // Fill a testWidth x testHeight cell with the results of each combination
    // of blend state.
    unsigned int beq, src, dst;
    for (beq = 0; beq < BLEND_EQUATION_COUNT; beq++) {
        for (src = 0; src < BLEND_FUNC_COUNT; src++) {
            for (dst = 0; dst < BLEND_FUNC_COUNT; dst++) {
                if (cellAllowed(src, beq * BLEND_FUNC_COUNT + dst)) {
                    cmd.SetViewportScissor(testWidth * src, testHeight * (beq * BLEND_FUNC_COUNT + dst), testWidth, testHeight);
                    blend.SetBlendTarget(0)
                            .SetBlendFunc(blendFunctions[src], blendFunctions[dst], blendFunctions[src], blendFunctions[dst])
                            .SetBlendEquation(blendEquations[beq], blendEquations[beq]);
                    cmd.BindBlendState(&blend);
                    cmd.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
                    cmd.DrawElements(DrawPrimitive::TRIANGLES, IndexType::UNSIGNED_SHORT, 6, iboAddr);
                }
            }
        }
    }

    if (multisample) {
        fbResult.downsample(cmd);
    }

    // rebind the system framebuffer and clear to black
    g_lwnWindowFramebuffer.bind();
    cmd.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    cmd.ClearColor();
    cmd.submit();

    // Read back the results from the rendertarget
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *pbo = allocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, fbWidth * fbHeight * 4);
    Texture *texResult = fbResult.getColorTexture(0);
    CopyRegion copyRegion = { 0, 0, 0, fbWidth, fbHeight, 1 };
    cmd.CopyTextureToBuffer(texResult, NULL, &copyRegion, pbo->GetAddress(), CopyFlags::NONE);
    cmd.submit();
    queue->Finish(); // Ensure copy is finished before we check the buffer contents

    // Persistently map the result data
    uint8_t *resPtr = (uint8_t*) pbo->Map();

    // Compare result data to expected values
#if DEBUG_MODE
    unsigned int incorrectCount = 0;
#endif
    unsigned int x, y;
    for (beq = 0; beq < BLEND_EQUATION_COUNT; beq++) {
        for (src = 0; src < BLEND_FUNC_COUNT; src++) {
            for (dst = 0; dst < BLEND_FUNC_COUNT; dst++) {
                if (cellAllowed(src, beq * BLEND_FUNC_COUNT + dst)) {
                    bool valid = true;
                    int xCorner = testWidth * src;
                    int yCorner = testHeight * (beq * BLEND_FUNC_COUNT + dst);
                    // Test every pixel of result data in the cell
                    for (y = 0; y < testHeight; y++) {
                        for (x = 0; x < testWidth; x++) {
                            int smallOffset = (y*testWidth + x) * 4;
                            int bigOffset = ((yCorner+y)*fbWidth + (xCorner+x)) * 4;
                            bool correct = checkBlendResult(
                                    blendEquations[beq], blendFunctions[src], blendFunctions[dst], sRGB,
                                    blendColorConst,
                                    &memDst[smallOffset], &memSrc0[smallOffset], &memSrc1[smallOffset],
                                    &resPtr[bigOffset]);
#if DEBUG_MODE
                            // Output debug information when information is
                            // incorrect. We'll cap the number of errors we
                            // dump info about to 100 to avoid excess spew.
                            if (!correct && (incorrectCount < 1000)) {
                                uint8_t expected[4];
                                blendColor(blendEquations[beq], blendFunctions[src], blendFunctions[dst], sRGB,
                                           blendColorConst,
                                           &memDst[smallOffset], &memSrc0[smallOffset], &memSrc1[smallOffset],
                                           expected);
                                DEBUG_PRINT(("FAIL: eq:%2u sfunc:%2u dfunc:%2u x:%u y:%u  "
                                           "dst: %3u %3u %3u %3u  src0: %3u %3u %3u %3u  src1: %3u %3u %3u %3u  "
                                           "exp: %3u %3u %3u %3u  got: %3u %3u %3u %3u\n",
                                            (unsigned int)blendEquations[beq],
                                            (unsigned int)blendFunctions[src],
                                            (unsigned int)blendFunctions[dst],
                                            x, y,
                                            memDst[smallOffset+0], memDst[smallOffset+1],
                                            memDst[smallOffset+2], memDst[smallOffset+3],
                                            memSrc0[smallOffset+0], memSrc0[smallOffset+1],
                                            memSrc0[smallOffset+2], memSrc0[smallOffset+3],
                                            memSrc1[smallOffset+0], memSrc1[smallOffset+1],
                                            memSrc1[smallOffset+2], memSrc1[smallOffset+3],
                                            expected[0], expected[1], expected[2], expected[3],
                                            resPtr[bigOffset+0], resPtr[bigOffset+1],
                                            resPtr[bigOffset+2], resPtr[bigOffset+3]));
                                incorrectCount++;
                            }
#endif
                            valid = valid && correct;
                        }
                    }
                    // Output pass/fail
                    SetCellViewportScissorPadded(cmd, src, beq * BLEND_FUNC_COUNT + dst, 1);
                    if (valid) {
                        // Clear texture 0 to green
                        cmd.ClearColor(0, 0.0, 1.0, 0.0);
                    } else {
                        // Clear texture 0 to red
                        cmd.ClearColor(0, 1.0, 0.0, 0.0);
                    }
                }
            }
        }
    }

    // Reset state to ensure that subsequent LWN tests don't accidentally read
    // from freed memory.
    cmd.BindVertexBuffer(0, 0, 0);
    cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, 0, 0);
    cmd.BindTexture(ShaderStage::FRAGMENT, 0, 0);
    cmd.BindTexture(ShaderStage::FRAGMENT, 1, 0);
    cmd.submit();
    delete [] memDst;
    delete [] memSrc0;
    delete [] memSrc1;

    // Clean up the framebuffer.
    queue->Finish();
    fbResult.destroy();
}


OGTEST_CppTest(LwnBlendTest, lwn_blend,             (false, false) );
OGTEST_CppTest(LwnBlendTest, lwn_blend_srgb,        (false, true) );
OGTEST_CppTest(LwnBlendTest, lwn_blend_multisample, (true, false) );


/**********************************************************************/
// Logic Op Evaluator
/**********************************************************************/

static void logicColor(LogicOp op, uint8_t dColor[4], uint8_t sColor[4], uint8_t out[4])
{
    for (int c = 0; c < 4; c++) {
        switch (op) {
        case LogicOp::CLEAR:
            out[c] = 0;
            break;
        case LogicOp::AND:
            out[c] = sColor[c] & dColor[c];
            break;
        case LogicOp::AND_REVERSE:
            out[c] = sColor[c] & ~dColor[c];
            break;
        case LogicOp::COPY:
            out[c] = sColor[c];
            break;
        case LogicOp::AND_ILWERTED:
            out[c] = ~sColor[c] & dColor[c];
            break;
        case LogicOp::NOOP:
            out[c] = dColor[c];
            break;
        case LogicOp::XOR:
            out[c] = sColor[c] ^ dColor[c];
            break;
        case LogicOp::OR:
            out[c] = sColor[c] | dColor[c];
            break;
        case LogicOp::NOR:
            out[c] = ~(sColor[c] | dColor[c]);
            break;
        case LogicOp::EQUIV:
            out[c] = ~(sColor[c] ^ dColor[c]);
            break;
        case LogicOp::ILWERT:
            out[c] = ~dColor[c];
            break;
        case LogicOp::OR_REVERSE:
            out[c] = sColor[c] | ~dColor[c];
            break;
        case LogicOp::COPY_ILWERTED:
            out[c] = ~sColor[c];
            break;
        case LogicOp::OR_ILWERTED:
            out[c] = ~sColor[c] | dColor[c];
            break;
        case LogicOp::NAND:
            out[c] = ~(sColor[c] & dColor[c]);
            break;
        case LogicOp::SET:
            out[c] = ~0;
            break;
        default:
            assert(0);
            break;
        }
    }
}

static bool checkLogicResult(LogicOp op, uint8_t *memDst, uint8_t *memSrc, uint8_t *res)
{
    bool valid = true;
    uint8_t expected[4];

    logicColor(op, memDst, memSrc, expected);
    for (int c = 0; c < 4; c++) {
        valid = valid && (expected[c] == res[c]);
    }
    return valid;
}


/**********************************************************************/
// Logic Op Test
/**********************************************************************/

static const char *fsDraw1Src =
    "#version 440 core\n"
    "layout(binding = 0) uniform sampler2D tex;\n"
    "layout(location = 0) out vec4 color;\n"
    "in IO { vec2 tc; };\n"
    "void main() {\n"
    "  color = texture(tex, tc);\n"
    "}\n";

// draw permutations
static const LogicOp logicOps[] = {
    LogicOp::CLEAR,
    LogicOp::AND,
    LogicOp::AND_REVERSE,
    LogicOp::COPY,
    LogicOp::AND_ILWERTED,
    LogicOp::NOOP,
    LogicOp::XOR,
    LogicOp::OR,
    LogicOp::NOR,
    LogicOp::EQUIV,
    LogicOp::ILWERT,
    LogicOp::OR_REVERSE,
    LogicOp::COPY_ILWERTED,
    LogicOp::OR_ILWERTED,
    LogicOp::NAND,
    LogicOp::SET,
};
#define LOGIC_OP_COUNT __GL_ARRAYSIZE(logicOps)

// Logic op test class
class LwnLogicOpTest {
protected:
    static const unsigned int testWidth = 8;
    static const unsigned int testHeight = 8;
    static const unsigned int cellWrap = 4; // 16 logicops, wrap at 4 for a 4x4 grid.

    bool multisample;

public:
    LWNTEST_CppMethods();

    LwnLogicOpTest(bool multisample) : multisample(multisample) {}
};

int LwnLogicOpTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(8,0);
}

lwString LwnLogicOpTest::getDescription() const
{
    return
        "Test LWN logicop state for a single rendertarget.\n"
        "We test an 8x8 set of texels using each logicop, and display a result on the screen that\n"
        "gives an overall health indicator of results. Green=good, Red=at least one failure.\n";
}

void LwnLogicOpTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &cmd = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    assert((LOGIC_OP_COUNT % cellWrap) == 0);
    cellTestInit(cellWrap, (LOGIC_OP_COUNT / cellWrap));

    // We will test all permutations of blend state, each holding
    // a testWidth * testHeight set of pixels.
    int fbWidth  = testWidth * cellWrap;
    int fbHeight = testHeight * (LOGIC_OP_COUNT / cellWrap);

    // allocator will create pool at first allocation
    // 64k is more than for this test but anything below 64k always allocates one page in the sparse pool
    const LWNsizeiptr coherent_poolsize = 0x10000UL;
    // safe guess, if the textures don't fit (in a future modification of this test) we'd notice allocation failures
    const LWNsizeiptr tex_poolsize = 0x100000UL;

    MemoryPoolAllocator allocator(device, NULL, coherent_poolsize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    MemoryPoolAllocator tex_allocator(device, NULL, tex_poolsize, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);

    // init programs
    Program *pgm = device->CreateProgram();

    VertexShader vs(440);
    vs << vsDrawSrc;
    FragmentShader fs(440);
    fs << fsDraw1Src;


    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec2 texcoord;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec2(0.0, 0.0) },
        { dt::vec3( 1.0, -1.0, 0.0), dt::vec2(1.0, 0.0) },
        { dt::vec3(-1.0,  1.0, 0.0), dt::vec2(0.0, 1.0) },
        { dt::vec3( 1.0,  1.0, 0.0), dt::vec2(1.0, 1.0) },
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, texcoord);
    VertexArrayState vstate = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexData), allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Set up index buffer
    static const uint16_t indexData[] = {0, 1, 2, 1, 3, 2};
    Buffer *ibo = AllocAndFillBuffer(device, queue, cmd, allocator, indexData, sizeof(indexData),
                                     BUFFER_ALIGN_INDEX_BIT, false);
    BufferAddress iboAddr = ibo->GetAddress();

    // init textures
    uint8_t *memDst = getRandom(testWidth*testHeight*4);
    uint8_t *memSrc = getRandom(testWidth*testHeight*4);
    Texture *texDst = AllocAndFillTexture2D(device, queue, cmd, tex_allocator, allocator, memDst, 4, testWidth, testHeight, Format(Format::RGBA8));
    Texture *texSrc = AllocAndFillTexture2D(device, queue, cmd, tex_allocator, allocator, memSrc, 4, testWidth, testHeight, Format(Format::RGBA8));

    // init sampler
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetWrapMode(WrapMode::REPEAT, WrapMode::REPEAT, WrapMode::REPEAT)
      .SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler *sampler = sb.CreateSampler();
    LWNuint samplerID = sampler->GetRegisteredID();

    // combined handles
    TextureHandle texDstHandle = device->GetTextureHandle(texDst->GetRegisteredTextureID(), samplerID);
    TextureHandle texSrcHandle = device->GetTextureHandle(texSrc->GetRegisteredTextureID(), samplerID);

    // ubos
    LWNfloat uboScaleData[] = { cellWrap, LOGIC_OP_COUNT / cellWrap };
    Buffer *uboScale = AllocAndFillBuffer(device, queue, cmd, allocator, uboScaleData, sizeof(uboScaleData),
                                          BUFFER_ALIGN_UNIFORM_BIT, false);

    LWNfloat uboIdentityData[] = { 1.0, 1.0 };
    Buffer *uboIdentity = AllocAndFillBuffer(device, queue, cmd, allocator, uboIdentityData, sizeof(uboIdentityData),
                                             BUFFER_ALIGN_UNIFORM_BIT, false);

    // init and set rendertarget
    Framebuffer fbResult(fbWidth, fbHeight);
    fbResult.setFlags(TextureFlags::COMPRESSIBLE);
    fbResult.setColorFormat(0, Format::RGBA8);
    fbResult.alloc(device);

    fbResult.bind(cmd);
    cmd.SetViewportScissor(0, 0, fbWidth, fbHeight);

    // Fill the backbuffer with the contents of texDst, repeating the pattern
    // once for each cell. (We use the repeat wrap mode, and adjust the texture
    // coordinates with a scaling factor, which allows us to cover every cell
    // in a single draw call.)

    cmd.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    cmd.BindVertexArrayState(vstate);

    cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, uboScale->GetAddress(), 2*sizeof(float));
    cmd.BindTexture(ShaderStage::FRAGMENT, 0, texDstHandle);
    cmd.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    cmd.DrawElements(DrawPrimitive::TRIANGLES, IndexType::UNSIGNED_SHORT, 6, iboAddr);

    // Set up state to test each cell with a separate draw call
    cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, uboIdentity->GetAddress(), 2*sizeof(float));
    cmd.BindTexture(ShaderStage::FRAGMENT, 0, texSrcHandle);

    // Fill a testWidth x testHeight cell with the results of each logic op
    ColorState color;
    color.SetDefaults();
    unsigned int lop;
    for (lop = 0; lop < LOGIC_OP_COUNT; lop++) {
        if (cellAllowed(lop % cellWrap, lop / cellWrap)) {
            cmd.SetViewportScissor(testWidth * (lop % cellWrap), testHeight * (lop / cellWrap), testWidth, testHeight);
            color.SetLogicOp(logicOps[lop]);
            cmd.BindColorState(&color);
            cmd.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
            cmd.DrawElements(DrawPrimitive::TRIANGLES, IndexType::UNSIGNED_SHORT, 6, iboAddr);
        }
    }
    
    // reset logic op
    color.SetLogicOp(LogicOp::COPY);
    cmd.BindColorState(&color);
    if (multisample) {
        fbResult.downsample(cmd);
    }

    // rebind the system framebuffer and clear to black
    g_lwnWindowFramebuffer.bind();
    cmd.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    cmd.ClearColor();
    cmd.submit();

    // Read back the results from the rendertarget
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *pbo = allocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, fbWidth * fbHeight * 4);
    Texture *texResult = fbResult.getColorTexture(0);
    CopyRegion copyRegion = { 0, 0, 0, fbWidth, fbHeight, 1 };
    cmd.CopyTextureToBuffer(texResult, NULL, &copyRegion, pbo->GetAddress(), CopyFlags::NONE);
    cmd.submit();
    queue->Finish(); // Ensure copy is finished before we check the buffer contents

    // Persistently map the result data
    uint8_t *resPtr = (uint8_t*) pbo->Map();

    // Compare result data to expected values
#if DEBUG_MODE
    unsigned int incorrectCount = 0;
#endif
    unsigned int x, y;
    for (lop = 0; lop < LOGIC_OP_COUNT; lop++) {
        if (cellAllowed(lop % cellWrap, lop / cellWrap)) {
            bool valid = true;
            int xCorner = testWidth * (lop % cellWrap);
            int yCorner = testHeight * (lop / cellWrap);
            // Test every pixel of result data in the cell
            for (y = 0; y < testHeight; y++) {
                for (x = 0; x < testWidth; x++) {
                    int smallOffset = (y*testWidth + x) * 4;
                    int bigOffset = ((yCorner+y)*fbWidth + (xCorner+x)) * 4;
                    bool correct = checkLogicResult( logicOps[lop],
                                                     &memDst[smallOffset], &memSrc[smallOffset],
                                                     &resPtr[bigOffset]);
#if DEBUG_MODE
                    // Output debug information when information is
                    // incorrect. We'll cap the number of errors we
                    // dump info about to 100 to avoid excess spew.
                    if (!correct && (incorrectCount < 100)) {
                        uint8_t expected[4];

                        logicColor(logicOps[lop], &memDst[smallOffset], &memSrc[smallOffset], expected);
                        DEBUG_PRINT(("FAIL: op:%i  x:%u y:%u  "
                                "dst: %3u %3u %3u %3u  src: %3u %3u %3u %3u  "
                                "exp: %3u %3u %3u %3u  got: %3u %3u %3u %3u\n",
                                (unsigned int)logicOps[lop],
                                x, y,
                                memDst[smallOffset+0], memDst[smallOffset+1],
                                memDst[smallOffset+2], memDst[smallOffset+3],
                                memSrc[smallOffset+0], memSrc[smallOffset+1],
                                memSrc[smallOffset+2], memSrc[smallOffset+3],
                                expected[0], expected[1], expected[2], expected[3],
                                resPtr[bigOffset+0], resPtr[bigOffset+1],
                                resPtr[bigOffset+2], resPtr[bigOffset+3]));
                        incorrectCount++;
                    }
#endif
                    valid = valid && correct;
                }
            }
            // Output pass/fail
            SetCellViewportScissorPadded(cmd, lop % cellWrap, lop / cellWrap, 1);
            if (valid) {
                // Clear texture 0 to green
                cmd.ClearColor(0, 0.0, 1.0, 0.0);
            } else {
                // Clear texture 0 to red
                cmd.ClearColor(0, 1.0, 0.0, 0.0);
            }
        }
    }

    // Reset state to ensure that subsequent LWN tests don't accidentally read
    // from freed memory.
    cmd.BindVertexBuffer(0, 0, 0);
    cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, 0, 0);
    cmd.BindTexture(ShaderStage::FRAGMENT, 0, 0);
    cmd.submit();

    delete [] memDst;
    delete [] memSrc;

    queue->Finish();
    fbResult.destroy();
}


OGTEST_CppTest(LwnLogicOpTest, lwn_logicop,         (false) );

