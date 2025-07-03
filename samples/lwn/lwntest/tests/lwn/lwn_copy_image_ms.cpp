/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <vector>

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#if defined(LW_TEGRA)
#include "lwn_PrivateFormats.h"
#endif

using namespace lwn;


#define DEBUG_MODE 0

#if DEBUG_MODE
#define DEBUG_PRINT(x) do { \
    printf x; \
    fflush(stdout); \
} while (0)
#else
#define DEBUG_PRINT(x)
#endif




class LWNCopyImageMS
{
    static const int failureReportCount = 16;
    static const int texSize = 200;         // both images use this size texture for outputs
    static const int ctaSize = 8;           // run with 8x8 work groups
    int samples;

    bool createPrograms(Device *device, Program **initPgm, Program **checkPgm,
                        SamplerComponentType sctype, int samples) const;

public:
    LWNCopyImageMS(int samples) : samples(samples) {}
    LWNTEST_CppMethods();
};



lwString LWNCopyImageMS::getDescription() const
{
    lwStringBuf sb;
    sb << "Test multisample textures with the CopyTextureToTexture API in LWN. Tests a few color formats\n"
          "in the following way:\n"
          " * Creates a " << texSize << "x" << texSize << "x2 multisample surface array and fills it with\n"
          "   data containing the LSB of the texel's position and sample number.\n"
          " * Copies the entire contents of this into a second texture.\n"
          " * Copies a subsets of the contents into the second texture at an easily identified location.\n"
          " * Uses a compute shader to compare the contents of the second texture to what is expected.\n";
    return sb.str();
}


int LWNCopyImageMS::isSupported() const
{
    return lwogCheckLWNAPIVersion(53, 1);
}


bool LWNCopyImageMS::createPrograms(Device *device, Program **initPgm, Program **checkPgm, SamplerComponentType sctype, int samples) const
{
    const char *text_prefix = "";
    const char *text_round  = "";
    const char *text_writecast = "";
    const char *text_readcast  = "";
    switch (sctype) {
    case COMP_TYPE_FLOAT:
        text_prefix = "";
        text_round  = "round";
        text_writecast = "buf.factor * ";
        text_readcast = "buf.factor * ";
        break;
    case COMP_TYPE_INT:
        text_prefix = "i";
        text_round  = "";
        text_writecast = "int";
        text_readcast = "";
        break;
    case COMP_TYPE_UNSIGNED:
        text_prefix = "u";
        text_round  = "";
        text_writecast = "";
        text_readcast = "";
        break;
    default:
        assert(!"Invalid sampler component type");
        return false;
    }

    lwShader vs = VertexShader(430);
    vs <<
        "void main() {\n"
        "  vec2 position;\n"
        "  if (gl_VertexID == 0) position = vec2(-1.0, -1.0);\n"
        "  if (gl_VertexID == 1) position = vec2(1.0, -1.0);\n"
        "  if (gl_VertexID == 2) position = vec2(1.0, 1.0);\n"
        "  if (gl_VertexID == 3) position = vec2(-1.0, 1.0);\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    lwShader fs = FragmentShader(430);
    fs <<
        "out " << text_prefix << "vec4 fcolor;\n"
        "layout(std430, binding = 0) buffer SSBO {\n"
        "    float factor;\n"
        "    uint  mask;\n"
        "    int   z;\n"
        "} buf;\n"
        "void main() {\n"
        "  ivec3 loc = ivec3(ivec2(gl_FragCoord.xy), buf.z);\n"
        "  fcolor = " << text_prefix << "vec4(0, 0, 0, 1);\n"
        "  uint pattern = (((loc.x & 3) << 6) | ((loc.y & 3) << 4) | ((loc.z & 1) << 3) | gl_SampleID);\n"
        "  fcolor.x = " << text_writecast << "(pattern & buf.mask);\n"
        "}\n";
    *initPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(*initPgm, vs, fs)) {
        LWNFailTest();
        return false;
    }

    lwShader cs = ComputeShader(430);
    cs <<
        "#define MAX_FAILURES " << failureReportCount << "\n"
        "const int sampleCount = " << samples << ";\n"
        "layout(binding=0) uniform " << text_prefix << "sampler2DMSArray tex;\n"
        "layout(std430, binding = 0) buffer SSBO {\n"
        "  float factor;\n"
        "  uint  mask;\n"
        "  uint  checked;\n"
        "  uint  failures;\n"
        "  uvec4 failureLocation[MAX_FAILURES];\n"
        "  uvec4 failureValues[MAX_FAILURES];\n"
        "} buf;\n"
        "\n"
        "void main() {\n"
        "  uvec3 loc = gl_GlobalIlwocationID;\n"
        "  for (int s=0; s < sampleCount; s++) {\n"
        "    uvec3 exp = loc;\n"

        // There is a subsection that will be different due to the second
        // CopyTextureToTexture that we do. Adjust 'exp' for this region.
        "    ivec3 regionSrc = ivec3(7, 1, 0);\n"
        "    ivec3 regionDst = ivec3(13, 27, 1);\n"
        "    ivec3 regionDim = ivec3(15, 61, 1);\n"
        "    if (all(lessThanEqual(regionDst, loc)) && all(lessThan(loc, regionDst+regionDim))) {\n"
        "      exp += (regionSrc-regionDst);\n"
        "    }\n"

        "    uint expected = ((exp.x & 3) << 6) | ((exp.y & 3) << 4) | ((exp.z & 1) << 3) | uint(s);\n"
        "    expected = expected & buf.mask;\n"
        "    uint value = uint(" << text_round << "(" << text_readcast << "texelFetch(tex, ivec3(loc), s).x));\n"
        "    if (expected != value) {\n"
        "      uint index = atomicAdd(buf.failures, 1);\n"
        "      if (index < MAX_FAILURES) {\n"
        "        buf.failureLocation[index] = uvec4(loc, s);\n"
        "        buf.failureValues[index] = uvec4(expected, value, 0, 0);\n"
        "      }\n"
        "    }\n"
        "    atomicAdd(buf.checked, 1);\n"
        "  }\n"
        "}\n";
    cs.setCSGroupSize(ctaSize, ctaSize);
    *checkPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(*checkPgm, cs)) {
        LWNFailTest();
        return false;
    }

    return true;
}


void LWNCopyImageMS::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const struct {
        Format fmt;
        uint32_t maxval;
        uint32_t mask;
    } formats[] = {
        { Format::R8,       255,    0xff },
        { Format::R8SN,     127,    0x7f },
        { Format::R8UI,     0,      0xff },
        { Format::R8I,      0,      0x7f },
        { Format::R16F,     1,      0xff },
        { Format::R16,      65535,  0xff },
        { Format::R16SN,    32767,  0xff },
        { Format::R16UI,    0,      0xff },
        { Format::R16I,     0,      0xff },
        { Format::R32F,     1,      0xff },
        { Format::R32UI,    0,      0xff },
        { Format::R32I,     0,      0xff },
        { Format::RG8,      255,    0xff },
        { Format::RG8SN,    127,    0x7f },
        { Format::RG8UI,    0,      0xff },
        { Format::RG8I,     0,      0x7f },
        { Format::RG16F,    1,      0xff },
        { Format::RG16,     65535,  0xff },
        { Format::RG16SN,   32767,  0xff },
        { Format::RG16UI,   0,      0xff },
        { Format::RG16I,    0,      0xff },
        { Format::RG32F,    1,      0xff },
        { Format::RG32UI,   0,      0xff },
        { Format::RG32I,    0,      0xff },
        { Format::RGBA8,    255,    0xff },
        { Format::RGBA8SN,  127,    0x7f },
        { Format::RGBA8UI,  0,      0xff },
        { Format::RGBA8I,   0,      0x7f },
        { Format::RGBA16F,  1,      0xff },
        { Format::RGBA16,   65535,  0xff },
        { Format::RGBA16SN, 32767,  0xff },
        { Format::RGBA16UI, 0,      0xff },
        { Format::RGBA16I,  0,      0xff },
        { Format::RGBA32F,  1,      0xff },
        { Format::RGBA32UI, 0,      0xff },
        { Format::RGBA32I,  0,      0xff },
        { Format::RGB10A2,  1023,   0xff },
        { Format::RGB10A2UI, 0,     0xff },
        { Format::R11G11B10F, 1,    0x3f },
        { Format::RGBX8,    255,    0xff },
        { Format::RGBX8SN,  127,    0x7f },
        { Format::RGBX8UI,  0,      0xff },
        { Format::RGBX8I,   0,      0x7f },
        { Format::RGBX16F,  1,      0xff },
        { Format::RGBX16,   65535,  0xff },
        { Format::RGBX16SN, 32767,  0xff },
        { Format::RGBX16UI, 0,      0xff },
        { Format::RGBX16I,  0,      0xff },
        { Format::RGBX32F,  1,      0xff },
        { Format::RGBX32UI, 0,      0xff },
        { Format::RGBX32I,  0,      0xff },
        { Format::BGR5,     31,     0x1f },
        { Format::BGR5A1,   31,     0x1f },
        { Format::BGR565,   31,     0x1f },
        { Format::BGRA8,    255,    0xff },
        { Format::BGRX8,    255,    0xff },
    };
    const int NUM_FORMATS = int(__GL_ARRAYSIZE(formats));

    std::vector<bool> passed;
    passed.reserve(NUM_FORMATS);

    const int cellWidth = 6*3;
    const int cellHeight = (NUM_FORMATS+5)/6;
    cellTestInit(cellWidth, cellHeight);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.submit();


    struct ProgramSet {
        Program *initPgm;
        Program *checkPgm;
    } program[3];

    for (int i=0; i<COMP_TYPE_COUNT; i++) {
        if (!createPrograms(device, &program[i].initPgm, &program[i].checkPgm, SamplerComponentType(i), samples)) {
            return;
        }
    }

    // The following two structs need to match the ones in the compute
    // shaders generated by createPrograms
    struct InitSSBO {
        float       factor;
        uint32_t    mask;
        int32_t     z;
    };

    struct CheckSSBO {
        float       factor;
        uint32_t    mask;
        uint32_t    checked;
        uint32_t    failures;
        struct {
            uint32_t    x, y, z, w;
        } failureLocation[failureReportCount];
        struct {
            uint32_t    x, y, z, w;
        } failureValues[failureReportCount];
    };

    size_t ssboStorageSize = (sizeof(InitSSBO) * 2) + sizeof(CheckSSBO);
    MemoryPoolAllocator ssboAllocator(device, NULL, ssboStorageSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    Buffer *initSSBO[2];
    struct InitSSBO *initSSBOPtr[2];
    BufferAddress initSSBOGpuAddr[2];
    for (int z=0; z<2; z++) {
        initSSBO[z] = ssboAllocator.allocBuffer(&bb, BUFFER_ALIGN_SHADER_STORAGE_BIT, sizeof(InitSSBO));
        initSSBOPtr[z] = (struct InitSSBO *) initSSBO[z]->Map();
        initSSBOGpuAddr[z] = initSSBO[z]->GetAddress();
    }

    Buffer *checkSSBO = ssboAllocator.allocBuffer(&bb, BUFFER_ALIGN_SHADER_STORAGE_BIT, sizeof(CheckSSBO));
    struct CheckSSBO *checkSSBOPtr = (struct CheckSSBO *) checkSSBO->Map();
    BufferAddress checkSSBOGpuAddr = checkSSBO->GetAddress();

    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE_ARRAY);
    tb.SetSize3D(texSize, texSize, 2);
    tb.SetSamples(samples);

    size_t texStorageSize = 0;
    for (int fmtIndex = 0; fmtIndex < NUM_FORMATS; fmtIndex++) {
        const FormatDesc &desc = *FormatDesc::findByFormat(formats[fmtIndex].fmt);
        tb.SetFormat(desc.format);
        texStorageSize = LW_MAX(texStorageSize, tb.GetPaddedStorageSize());
    }

    MemoryPoolAllocator texAllocator(device, NULL, 2*texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    MultisampleState msState;
    msState.SetDefaults();
    msState.SetMultisampleEnable(LWN_TRUE);
    msState.SetSamples(samples);
    queueCB.BindMultisampleState(&msState);

    for (int fmtIndex = 0; fmtIndex < NUM_FORMATS; fmtIndex++)
    {
        const FormatDesc &desc = *FormatDesc::findByFormat(formats[fmtIndex].fmt);

        DEBUG_PRINT(("Testing %s\n", desc.formatName));
        const SamplerComponentType sctype = desc.samplerComponentType;
        const Format fmt = desc.format;
        const uint32_t mask = formats[fmtIndex].mask;
        const uint32_t maxval = formats[fmtIndex].maxval;
        tb.SetFormat(fmt);
        Texture *srcTex = texAllocator.allocTexture(&tb);
        Texture *dstTex = texAllocator.allocTexture(&tb);

        LWNuint dstTexID = dstTex->GetRegisteredTextureID();
        TextureHandle dstTexHandle = device->GetTexelFetchHandle(dstTexID);

        queueCB.BindProgram(program[sctype].initPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
        for (int z=0; z<2; z++) {
            initSSBOPtr[z]->factor = float(1.0 / (double)maxval);
            initSSBOPtr[z]->mask = mask;
            initSSBOPtr[z]->z = z;
            TextureView tv;
            TextureView *views = &tv;
            tv.SetDefaults().SetLayers(z, 1);
            queueCB.SetRenderTargets(1, &srcTex, &views, NULL, NULL);
            queueCB.BindStorageBuffer(ShaderStage::FRAGMENT, 0, initSSBOGpuAddr[z], sizeof(InitSSBO));
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
            queueCB.submit();
        }

        {
            CopyRegion region = { 0, 0, 0, texSize, texSize, 2 };
            queueCB.CopyTextureToTexture(srcTex, NULL, &region, dstTex, NULL, &region, CopyFlags::NONE);

            int regionW = 15;
            int regionH = 61;
            CopyRegion srcRegion = { 7, 1, 0, regionW, regionH, 1 };
            CopyRegion dstRegion = { 13, 27, 1, regionW, regionH, 1 };
            queueCB.CopyTextureToTexture(srcTex, NULL, &srcRegion, dstTex, NULL, &dstRegion, CopyFlags::NONE);
            queueCB.submit();
        }

        checkSSBOPtr->factor = maxval;
        checkSSBOPtr->mask = mask;
        checkSSBOPtr->checked = 0;
        checkSSBOPtr->failures = 0;
        queueCB.BindProgram(program[sctype].checkPgm, ShaderStageBits::COMPUTE);
        queueCB.BindStorageBuffer(ShaderStage::COMPUTE, 0, checkSSBOGpuAddr, sizeof(CheckSSBO));
        queueCB.BindTexture(ShaderStage::COMPUTE, 0, dstTexHandle);
        queueCB.DispatchCompute(texSize/ctaSize, texSize/ctaSize, 2);
        queueCB.submit();
        queue->Finish();
#if DEBUG_MODE
        DEBUG_PRINT(("checked: %u, failures: %u  (factor: %0.2f  mask: %x)\n",
                     checkSSBOPtr->checked,
                     checkSSBOPtr->failures,
                     checkSSBOPtr->factor,
                     checkSSBOPtr->mask));
        for (uint32_t i = 0; i < LW_MIN(checkSSBOPtr->failures, failureReportCount); i++) {
            DEBUG_PRINT(("failure %i at (%u,%u,%u,%u)  expected: 0x%08x   got: 0x%08x\n",
                         i,
                         checkSSBOPtr->failureLocation[i].x,
                         checkSSBOPtr->failureLocation[i].y,
                         checkSSBOPtr->failureLocation[i].z,
                         checkSSBOPtr->failureLocation[i].w,
                         checkSSBOPtr->failureValues[i].x,
                         checkSSBOPtr->failureValues[i].y));
        }
#endif
        passed.push_back((checkSSBOPtr->checked != texSize*texSize*2) && (checkSSBOPtr->failures == 0));

        texAllocator.freeTexture(srcTex);
        texAllocator.freeTexture(dstTex);
    }

    msState.SetDefaults();
    queueCB.BindMultisampleState(&msState);
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();

    for (int cellNum = 0; cellNum < NUM_FORMATS; cellNum++) {
        SetCellViewportScissorPadded(queueCB, cellNum % cellWidth, cellNum / cellWidth, 1);
        queueCB.ClearColor(0, passed[cellNum] ? 0.0 : 1.0, passed[cellNum] ? 1.0 : 0.0, 0.0);
        queueCB.submit();
    }
    queue->Finish();

    ssboAllocator.freeBuffer(initSSBO[0]);
    ssboAllocator.freeBuffer(initSSBO[1]);
    ssboAllocator.freeBuffer(checkSSBO);
}


OGTEST_CppTest(LWNCopyImageMS, lwn_copy_image_ms_2x, (2));
OGTEST_CppTest(LWNCopyImageMS, lwn_copy_image_ms_4x, (4));
OGTEST_CppTest(LWNCopyImageMS, lwn_copy_image_ms_8x, (8));



