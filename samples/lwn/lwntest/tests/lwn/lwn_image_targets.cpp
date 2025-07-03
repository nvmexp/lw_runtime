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

#if 0
// DEBUG HACK:  Set this #define to one of the TextureType enums below to run
// the tests with only a single texture type.
#define SINGLE_TEX_TYPE Texture2DMS
#endif

class LWNImageQueryTest
{
public:
    // Enum identifying the specific feature being exercised by each variant
    // of this test.
    enum TestVariant {
        TestVariantQuery,           // exercise image queries
        TestVariantLoad,            // exercise image loads
        TestVariantStore,           // exercise image stores
        TestVariantTexelFetch,      // exercise texel fetches (like image loads)
    };

private:
    // Render 32x32 quads in 40x40 cells.
    static const int cellSize = 32;
    static const int cellSpacing = 40;
    static const int cellPadding = (cellSpacing - cellSize) / 2;
    static const int cellsX = 640 / cellSpacing;
    static const int cellsY = 480 / cellSpacing;

    // Sizes of memory pools used for (theoretically) storing texture data.
    static const int texStorageSize = 32 * 1024 * 1024;

    // Number of image bindings used for testing.
    static const int imageBindingsUsed = 4;

    // The test iterates over texture targets in columns.
    enum TextureType {
        Texture1D,
        Texture1DArray,
        TextureRect,
        Texture2D,
        Texture2DArray,
        TextureLwbe,
        TextureLwbeArray,
        Texture3D,
        TextureBuffer,
        Texture2DMS,
        Texture2DMSArray,
        TextureTypeCount,
    };

    // We have several different types of methods for accessing images used by
    // the test.
    enum AccessVariant {
        AccessVariantImage0,
        AccessVariantImage3,
        AccessVariantImageIndexed,
        AccessVariantImageBindless,
        AccessVariantCount,
    };

    Program *mkprogram(Device *device, TextureType textype, AccessVariant accessVariant, TestVariant shaderVariant) const;
    Texture *mktexture(TextureType textype, TextureBuilder &tb, MemoryPoolAllocator &allocator, dt::ivec4 size,
                       QueueCommandBuffer &queueCB, BufferAddress texDataBufferAddr,
                       LWNuint **texDataPtr, LWNsizeiptr *texDataOffset,
                       LWNuint valueBias) const;
    TestVariant m_variant;
public:
    LWNImageQueryTest(TestVariant variant) : m_variant(variant) {}

    LWNTEST_CppMethods();
};

lwString LWNImageQueryTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Functional test for GLSL image uniforms of different targets.\n"
        "\n";
    switch (m_variant) {
    case TestVariantQuery:
        sb <<
            "This test renders primitives with textures of various targets and "
            "sizes and verifies that the values reported by the imageSize() and "
            "imageSamples() built-in functions are correct.  Cells are rendered "
            "in green (correct) or red (incorrect) depending on returned values.\n";
        break;
    case TestVariantLoad:
        sb <<
            "This test fills textures with a specific pattern of values and calls "
            "imageLoad() for various texels to determine that correct values were "
            "loaded.  Pixels are rendered in green (correct) or red (incorrect) "
            "depending on fetched values.\n";
        break;
    case TestVariantStore:
        sb <<
            "This test fills textures with a specific pattern of values using "
            "imageStore() in one pass, and then fetches them  using imageLoad() "
            "in a second pass to determine that correct values were stored and "
            "loaded.  Pixels are rendered in green (correct) or red (incorrect) "
            "depending on fetched values.\n";
        break;
    case TestVariantTexelFetch:
        sb <<
            "This test fills textures with a specific pattern of values and calls "
            "texelFetch() for various texels to determine that correct values were "
            "loaded.  Pixels are rendered in green (correct) or red (incorrect) "
            "depending on fetched values.\n";
        break;
    default:
        assert(0);
        break;
    }
    sb <<
        "\n"
        "Each column tests a different texture target type (1D, 1DA, RECT, 2D, "
        "2DA, LWBE, LWBEA, 3D, BUF, 2DMS, 2DMSA).  Rows test different texture "
        "sizes, as well as both bindless and bound textures."
        ;
    return sb.str();
}

int LWNImageQueryTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(38, 1);
}

Program *LWNImageQueryTest::mkprogram(Device *device, TextureType textype, AccessVariant accessVariant, TestVariant shaderVariant) const
{
    // Set up a pass-through vertex shader.
    VertexShader vs(450);
    vs <<
        "layout(location=0) in vec2 position;\n"
        "out vec2 tc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "  tc = 0.5 * position + 0.5;\n"
        "}\n";

    // Set up a fragment shader to perform the required functional checks.
    FragmentShader fs(450);
    fs.addExtension(lwShaderExtension::ARB_bindless_texture);

    // Variable used to access the image.
    const char *imageAccess = "image";

    // Table of basic information on image types for each texture target,
    // accessed using either texture or image load/store.
    const char *typePrefix = (shaderVariant == TestVariantTexelFetch) ? "usampler" : "uimage";
    static const struct ImageTypeInfo {
        const char *qsuffix;            // suffix on components returned by queries
        const char *asuffix;            // suffix on coordinates used for access
        const char *imageExtra;         // extra argument for imageLoad/imageStore calls
        const char *texelFetchExtra;    // extra argument(s) for texelFetch calls
        const char *typeSuffix;         // suffix used to set up image/sampler variable types
    } imageTypeInfo[] = {
        { ".x",   ".x",   "",      ", 0",   "1D" },
        { ".xy",  ".xy",  "",      ", 0",   "1DArray" },
        { ".xy",  ".xy",  "",      "",      "2DRect" },
        { ".xy",  ".xy",  "",      ", 0",   "2D" },
        { ".xyz", ".xyz", "",      ", 0",   "2DArray" },
        { ".xy",  ".xyz", "",      ", 0",   "Lwbe" },
        { ".xyz", ".xyz", "",      ", 0",   "LwbeArray" },
        { ".xyz", ".xyz", "",      ", 0",   "3D" },
        { ".x",   ".x",   "",      "",      "Buffer" },
        { ".xy",  ".xy",  ", smp", ", smp", "2DMS" },
        { ".xyz", ".xyz", ", smp", ", smp", "2DMSArray" },
    };
    const ImageTypeInfo *info = &imageTypeInfo[textype];
    ct_assert(__GL_ARRAYSIZE(imageTypeInfo) == TextureTypeCount);

    fs <<
        "layout(std140, binding = 0) uniform Block {\n"
        "  ivec4 size;\n"
        "  int which;\n"
        "  uint bias;\n"
        "  uvec2 imageHandle;\n"
        "};\n";
    if (accessVariant != AccessVariantImageBindless) {
        const char *r32ui = (shaderVariant == TestVariantTexelFetch) ? "" : ", r32ui";
        switch (accessVariant) {
        case AccessVariantImage0:
            fs << "layout(binding=0" << r32ui << ") uniform " << typePrefix << info->typeSuffix << " image;\n";
            break;
        case AccessVariantImage3:
            fs << "layout(binding=3" << r32ui << ") uniform " << typePrefix << info->typeSuffix << " image;\n";
            break;
        case AccessVariantImageIndexed:
            fs << "layout(binding=0" << r32ui << ") uniform " << typePrefix << info->typeSuffix << " images["<< imageBindingsUsed <<"];\n";
            imageAccess = "images[which]";
            break;
        default:
            assert(0);
        }
    }
    fs <<
        "in vec2 tc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  bool result = true;\n";
    if (accessVariant == AccessVariantImageBindless) {
        if (shaderVariant == TestVariantTexelFetch) {
            fs <<
                "  " << typePrefix << info->typeSuffix << " image = " <<
                typePrefix << info->typeSuffix << "(imageHandle);\n";
        } else {
            fs <<
                "  layout(r32ui) " << typePrefix << info->typeSuffix << " image = "
                "layout(r32ui) " << typePrefix << info->typeSuffix << "(imageHandle);\n";
        }
    }

    if (shaderVariant != TestVariantQuery) {
        fs <<
            "  ivec3 xyz = ivec3(tc * " << cellSize << ".0, 0);\n"
            "  xyz.z = xyz.x + xyz.y;\n"
            "  xyz = xyz % size.xyz;\n"
            "  uint expval = bias + xyz.x + (xyz.y + xyz.z * size.y) * size.x;\n";
    }

    switch (shaderVariant) {
    case TestVariantQuery:
        // Check imageSamples() for multisample targets.
        if (textype == Texture2DMS || textype == Texture2DMSArray) {
            fs <<
                "  if (imageSamples(" << imageAccess << ") != size.w) {\n"
                "    result = false;\n"
                "  }\n";
        }

        // Check imageSize() for all targets and color red/green based on results.
        fs <<
            "  if (imageSize(" << imageAccess << ") != size" << info->qsuffix << ") {\n"
            "    result = false;\n"
            "  }\n";
        break;

    case TestVariantLoad:
    case TestVariantTexelFetch:
        fs << "  uint texel = 0;\n";
        if (textype == Texture2DMS || textype == Texture2DMSArray) {
            fs <<
                "  uint baseexp = expval;\n"
                "  for (int smp = 0; smp < size.w; smp++) {\n";
        }
        if (shaderVariant == TestVariantLoad) {
            fs << "    texel = imageLoad(" << imageAccess << ", xyz" << info->asuffix << info->imageExtra << ").x;\n";
        } else {
            fs << "    texel = texelFetch(" << imageAccess << ", xyz" << info->asuffix << info->texelFetchExtra << ").x;\n";
        }
        if (textype == Texture2DMS || textype == Texture2DMSArray) {
            fs << "    expval = baseexp ^ smp;\n";
        }
        fs << "    if (texel != expval) result = false;\n";
        if (textype == Texture2DMS || textype == Texture2DMSArray) {
            fs << "  }\n";
        }
        break;

    case TestVariantStore:
        if (textype == Texture2DMS || textype == Texture2DMSArray) {
            fs <<
                "  uint baseexp = expval;\n"
                "  for (int smp = 0; smp < size.w; smp++) {\n"
                "    expval = baseexp ^ smp;\n";
        }
        fs <<
            "    imageStore(" << imageAccess << ", xyz" << info->asuffix << info->imageExtra << ", uvec4(expval, 0, 0, 0));\n"
            "    result = false;\n";
        if (textype == Texture2DMS || textype == Texture2DMSArray) {
            fs << " }\n";
        }
        break;

    default:
        assert(0);
        break;
    }

    // Color red/green based on test results.
    fs <<
        "  if (result) {\n"
        "    fcolor = vec4(0,1,0,1);\n"
        "  } else {\n"
        "    fcolor = vec4(1,0,0,1);\n"
        "  }\n";
#if 0
    // Debugging hack that displays the values fetched by the texture access
    // and expected by the test.
    fs <<
        "  fcolor = vec4(float(texel  & 0xFF) / 255.0, float(texel  >> 8) / 255.0,\n"
        "                float(expval & 0xFF) / 255.0, float(expval >> 8) / 255.0);\n";
#endif
    fs << "}\n";

    Program *program = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(program, vs, fs)) {
        program->Free();
        program = NULL;
    }
    return program;
}

Texture *LWNImageQueryTest::mktexture(TextureType textype, TextureBuilder &tb, 
                                      MemoryPoolAllocator &allocator, dt::ivec4 size,
                                      QueueCommandBuffer &queueCB, BufferAddress texDataBufferAddr,
                                      LWNuint **texDataPtr, LWNsizeiptr *texDataOffset,
                                      LWNuint valueBias) const
{
    // Set up information on the texture.
    int w = size[0];
    int h = size[1];
    int d = size[2];
    int samples = 0;
    TextureTarget target = TextureTarget::TARGET_2D;
    TextureFlags texFlags = TextureFlags::IMAGE;

    switch (textype) {
    case Texture1D:
        target = TextureTarget::TARGET_1D;
        h = 1;
        d = 1;
        break;
    case Texture1DArray:
        target = TextureTarget::TARGET_1D_ARRAY;
        d = 1;
        break;
    case Texture2D:
        target = TextureTarget::TARGET_2D;
        d = 1;
        break;
    case Texture2DArray:
        target = TextureTarget::TARGET_2D_ARRAY;
        break;
    case TextureRect:
        target = TextureTarget::TARGET_RECTANGLE;
        d = 1;
        break;
    case TextureLwbe:
        target = TextureTarget::TARGET_LWBEMAP;
        h = w;      // force square dimensions
        d = 6;
        break;
    case TextureLwbeArray:
        target = TextureTarget::TARGET_LWBEMAP_ARRAY;
        h = w;                              // force square dimensions
        d = 6 * ((d < 6) ? 1 : int(d / 6)); // force 6N faces
        break;
    case Texture3D:
        target = TextureTarget::TARGET_3D;
        break;
    case TextureBuffer:
        target = TextureTarget::TARGET_BUFFER;
        h = 1;
        d = 1;
        break;
    case Texture2DMS:
        target = TextureTarget::TARGET_2D_MULTISAMPLE;
        d = 1;
        samples = size[3];
        texFlags |= TextureFlags::COMPRESSIBLE;
        break;
    case Texture2DMSArray:
        target = TextureTarget::TARGET_2D_MULTISAMPLE_ARRAY;
        samples = size[3];
        texFlags |= TextureFlags::COMPRESSIBLE;
        break;
    default:
        assert(0);
        break;
    }

    tb.SetDefaults();
    tb.SetFlags(texFlags);
    tb.SetTarget(target);
    if (textype == TextureLwbe) {
        tb.SetSize3D(w, h, 6);
    } else {
        tb.SetSize3D(w, h, d);
    }
    tb.SetFormat(Format::R32UI);
    tb.SetLevels(1);
    tb.SetSamples(samples);
    Texture *tex = allocator.allocTexture(&tb);

    // For "load" tests, initialize the textures with a pattern used by the
    // tests.  We can't load texels directly for multisample textures.
    if (m_variant == TestVariantLoad || m_variant == TestVariantTexelFetch) {
        if (textype != Texture2DMS && textype != Texture2DMSArray) {
            LWNuint *texData = *texDataPtr;
            LWNuint value = valueBias;
            for (int texel = 0; texel < w * h * d; texel++) {
                *texData++ = value++;
            }
            CopyRegion copyRegion = { 0, 0, 0, w, h, d };
            queueCB.CopyBufferToTexture(texDataBufferAddr + *texDataOffset, tex, 0, &copyRegion, CopyFlags::NONE);
            *texDataOffset += (texData - *texDataPtr) * sizeof(LWNuint);
            *texDataPtr = texData;
        }
    }

    return tex;
}


void LWNImageQueryTest::doGraphics() const
{
    // Test a variety of width/height/depth/sample count values.
    dt::ivec4 textureSizes[] = {
        dt::ivec4(32, 17,  3, 2),
        dt::ivec4( 1,  8, 19, 4),
        dt::ivec4( 9, 26, 12, 8),
    };
    const unsigned int nTextureSizes = __GL_ARRAYSIZE(textureSizes);

    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Generate programs for each texture target and shader variant.
    Program *programs[TextureTypeCount][AccessVariantCount];
    Program *loadPrograms[TextureTypeCount][AccessVariantCount];
    for (int accessVariant = 0; accessVariant < AccessVariantCount; accessVariant++) {
        for (int textype = 0; textype < TextureTypeCount; textype++) {
            programs[textype][accessVariant] = NULL;
            loadPrograms[textype][accessVariant] = NULL;
#ifdef SINGLE_TEX_TYPE
            if (textype != SINGLE_TEX_TYPE) continue;
#endif
            // Don't compile shaders using texel fetches from lwbe maps; these targets aren't supported.
            if (m_variant == TestVariantTexelFetch && (textype == TextureLwbe || textype == TextureLwbeArray)) {
                continue;
            }
            programs[textype][accessVariant] = mkprogram(device, TextureType(textype), AccessVariant(accessVariant), m_variant);
            if (!programs[textype][accessVariant]) {
                LWNFailTest();
                return;
            }
            if (m_variant == TestVariantStore) {
                loadPrograms[textype][accessVariant] = mkprogram(device, TextureType(textype), AccessVariant(accessVariant), TestVariantLoad);
                if (!loadPrograms[textype][accessVariant]) {
                    LWNFailTest();
                    return;
                }
            }
        }
    }

    // For load tests, we need to initialize the multisample textures using a shader.
    Program *msInitProgram = NULL;
    if (m_variant == TestVariantLoad || m_variant == TestVariantTexelFetch) {

        VertexShader vs(450);
        vs <<
            "layout(location=0) in vec2 position;\n"
            "out v2g {\n"
            "  int layer;\n"
            "};\n"
            "void main() {\n"
            "  gl_Position = vec4(position, 0.0, 1.0);\n"
            "  layer = gl_InstanceID;\n"
            "}\n";

        GeometryShader gs(450);
        gs.setGSParameters(GL_TRIANGLES, GL_TRIANGLE_STRIP, 3, 1);
        gs <<
            "in v2g {\n"
            "  int layer;\n"
            "} iv[];\n"
            "void main() {\n"
            "  for (int i = 0; i < 3; i++) {\n"
            "     gl_Position = gl_in[i].gl_Position;\n"
            "     gl_Layer = iv[i].layer;\n"
            "     EmitVertex();\n"
            "  }\n"
            "}\n";

        FragmentShader fs(450);
        fs <<
            "layout(std140, binding = 0) uniform Block {\n"
            "  ivec4 size;\n"
            "  int which;\n"
            "  uint bias;\n"
            "  uvec2 imageHandle;\n"
            "};\n"
            "out uint ovalue;\n"
            "void main() {\n"
            "  int sid = gl_SampleID;\n"
            "  ivec3 xyz = ivec3(gl_FragCoord.xy, gl_Layer);\n"
            "  ovalue = bias + xyz.x + (xyz.y + xyz.z * size.y) * size.x;\n"
            "  ovalue ^= sid;  // or in the sample number \n"
            "}\n";

        msInitProgram = device->CreateProgram();
        if (!g_glslcHelper->CompileAndSetShaders(msInitProgram, vs, gs, fs)) {
            LWNFailTest();
            return;
        }
    }

    // Set up a pool/buffer for staging texel data.
    MemoryPoolAllocator texDataAllocator(device, NULL, texStorageSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *texDataBuffer = texDataAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, texStorageSize);
    BufferAddress texDataBufferAddr = texDataBuffer->GetAddress();
    LWNuint *texDataCPUAddress = (LWNuint *) texDataBuffer->Map();
    LWNsizeiptr texDataOffset = 0;

    // Set up a set of texture objects for each texture target and size.
    MemoryPoolAllocator texAllocator(device, NULL, texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    MemoryPoolAllocator texBOAllocator(device, NULL, texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    Texture *textures[TextureTypeCount][nTextureSizes];
    for (int i = 0; i < TextureTypeCount; i++) {
#ifdef SINGLE_TEX_TYPE
        if (i != SINGLE_TEX_TYPE) continue;
#endif
        MemoryPoolAllocator &allocator = (i == TextureBuffer) ? texBOAllocator : texAllocator;
        for (unsigned int j = 0; j < nTextureSizes; j++) {
            textures[i][j] = mktexture(TextureType(i), tb, allocator, textureSizes[j],
                                       queueCB, texDataBufferAddr, &texDataCPUAddress, &texDataOffset,
                                       j + i * nTextureSizes);
        }
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec2 position;
    };
    Vertex vertexData[] = {
        { dt::vec2(-1.0, -1.0) },
        { dt::vec2(-1.0, +1.0) },
        { dt::vec2(+1.0, -1.0) },
        { dt::vec2(+1.0, +1.0) },
    };
    MemoryPoolAllocator vboAllocator(device, NULL, 4 * sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, vboAllocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Set up a block of memory to be used as uniform buffers for the test.
    // Each test has its own block of UBO memory with expected results and
    // bindless handles.
    struct UBOData {
        dt::ivec4       size;
        int             which;
        unsigned int    bias;
        dt::uvec2       handle;
    };
    int uboVersions = (AccessVariantCount + 1) * nTextureSizes * TextureTypeCount;
    LWNint uboAlignment = 0;
    device->GetInteger(DeviceInfo::UNIFORM_BUFFER_ALIGNMENT, &uboAlignment);
    int uboSize = uboAlignment * ((sizeof(UBOData) + uboAlignment - 1) / uboAlignment);
    MemoryPoolAllocator uboAllocator(device, NULL, uboVersions * uboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    bb.SetDefaults();
    Buffer *ubo = uboAllocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, uboVersions * uboSize);
    BufferAddress uboAddr = ubo->GetAddress();
    char *uboMem = (char *) ubo->Map();
    int uboIndex = 0;

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    // Set up a dummy sampler that can be used to get a texture handle for
    // texel fetch tests.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    Sampler *smp = sb.CreateSampler();

    // Set up an array of image and texture handles for our images.
    LWNtextureHandle texHandles[TextureTypeCount][nTextureSizes];
    LWNimageHandle imageHandles[TextureTypeCount][nTextureSizes];
    for (int textype = 0; textype < TextureTypeCount; textype++) {
#ifdef SINGLE_TEX_TYPE
        if (textype != SINGLE_TEX_TYPE) continue;
#endif
        for (unsigned int texSizeIdx = 0; texSizeIdx < nTextureSizes; texSizeIdx++) {
            LWNuint id = g_lwnTexIDPool->RegisterImage(textures[textype][texSizeIdx]);
            imageHandles[textype][texSizeIdx] = device->GetImageHandle(id);
            texHandles[textype][texSizeIdx] = 
                device->GetTextureHandle(textures[textype][texSizeIdx]->GetRegisteredTextureID(),
                                        smp->GetRegisteredID());
        }
    }

    // For multisample load tests, initialize the textures with layered
    // rendering and a special shader that splats the expected values into
    // each sample.
    if (m_variant == TestVariantLoad || m_variant == TestVariantTexelFetch) {

        MultisampleState msState;
        msState.SetDefaults();
        queueCB.BindProgram(msInitProgram, ShaderStageBits::ALL_GRAPHICS_BITS);

        for (unsigned int texSizeIdx = 0; texSizeIdx < nTextureSizes; texSizeIdx++) {
            queueCB.SetViewportScissor(0, 0, textureSizes[texSizeIdx][0], textureSizes[texSizeIdx][1]);
            for (int array = 0; array < 2; array++) {
                TextureType textype = array ? Texture2DMSArray : Texture2DMS;
#ifdef SINGLE_TEX_TYPE
                if (textype != SINGLE_TEX_TYPE) continue;
#endif
                int layers = array ? textureSizes[texSizeIdx][2] : 1;
                dt::ivec4 size = textureSizes[texSizeIdx];
                size[2] = layers;
                UBOData *uboData = (UBOData *) (uboMem + uboIndex * uboSize);
                uboData->size = size;
                uboData->which = 0;
                uboData->bias = texSizeIdx + nTextureSizes * textype;
                uboData->handle = dt::uvec2(0);
                queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr + uboIndex * uboSize, uboSize);
                uboIndex++;
                assert(uboIndex <= uboVersions);

                msState.SetMultisampleEnable(LWN_TRUE);
                msState.SetSamples(size[3]);
                queueCB.BindMultisampleState(&msState);
                queueCB.SetRenderTargets(1, &textures[textype][texSizeIdx], NULL, NULL, NULL);
                queueCB.DrawArraysInstanced(DrawPrimitive::TRIANGLE_STRIP, 0, 4, 0, layers);
            }
        }
        msState.SetDefaults();
        queueCB.BindMultisampleState(&msState);
        g_lwnWindowFramebuffer.bind();
    }

    // Loop over texture targets, sizes, and other shader variants.
    for (int textype = 0; textype < TextureTypeCount; textype++) {
#ifdef SINGLE_TEX_TYPE
        if (textype != SINGLE_TEX_TYPE) continue;
#endif
        for (unsigned int texSizeIdx = 0; texSizeIdx < nTextureSizes; texSizeIdx++) {
            for (int accessVariant = 0; accessVariant < AccessVariantCount; accessVariant++) {

                bool skip = false;
                switch (m_variant) {
                case TestVariantTexelFetch:
                    // GLSL doesn't support texel fetches from lwbe maps and arrays.
                    if (textype == TextureLwbe || textype == TextureLwbeArray) skip = true;
                    break;
                case TestVariantLoad:
                case TestVariantStore:
                    // LWN's implementation of loads/stores to MS textures is broken.
                    if (textype == Texture2DMS || textype == Texture2DMSArray) skip = true;
                    break;
                default:
                    break;
                }

                if (skip) {
                    int row = accessVariant * nTextureSizes + texSizeIdx;
                    assert(row < cellsY);
                    queueCB.SetViewportScissor(textype * cellSpacing + cellPadding, row * cellSpacing + cellPadding, cellSize, cellSize);
                    queueCB.ClearColor(0, 0.0, 0.0, 1.0, 1.0);
                    continue;
                }

                // Set up the UBO for this test iteration with expected size
                // values, binding index to use, and the bindless handle (if
                // required).
                dt::ivec4 size = textureSizes[texSizeIdx];
                switch (textype) {
                case Texture1D:
                case TextureBuffer:
                    size[1] = 1;
                    size[2] = 1;
                    break;
                case Texture1DArray:
                case TextureRect:
                case Texture2D:
                case Texture2DMS:
                    size[2] = 1;
                    break;
                case TextureLwbe:
                    size[1] = size[0];
                    size[2] = 6;
                    break;
                case TextureLwbeArray:
                    size[1] = size[0];
                    size[2] = 6 * ((size[2] < 6) ? 1 : size[2] / 6);
                    break;
                case Texture2DArray:
                case Texture3D:
                case Texture2DMSArray:
                    break;
                default:
                    assert(0);
                    break;
                }

                // Select the binding to use based on the test variant.  For
                // "indexed", we pick one entry from an array of bindings
                // semi-randomly.
                int usedBinding = 0;
                switch (accessVariant) {
                case AccessVariantImage0:         usedBinding = 0; break;
                case AccessVariantImage3:         usedBinding = 3; break;
                case AccessVariantImageIndexed:   usedBinding = (textype + accessVariant) % imageBindingsUsed; break;
                case AccessVariantImageBindless:  usedBinding = 0; break;
                default:                          assert(0); break;
                }

                UBOData *uboData = (UBOData *) (uboMem + uboIndex * uboSize);
                uboData->size = size;
                uboData->which = usedBinding;
                uboData->bias = texSizeIdx + nTextureSizes * textype;
                if (accessVariant == AccessVariantImageBindless) {
                    if (m_variant == TestVariantTexelFetch) {
                        LWNtextureHandle texHandle = texHandles[textype][texSizeIdx];
                        uboData->handle = dt::uvec2(texHandle & 0xFFFFFFFF, texHandle >> 32);
                    } else {
                        LWNimageHandle imageHandle = imageHandles[textype][texSizeIdx];
                        uboData->handle = dt::uvec2(imageHandle & 0xFFFFFFFF, imageHandle >> 32);
                    }
                }
                queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr + uboIndex * uboSize, uboSize);
                uboIndex++;
                assert(uboIndex <= uboVersions);

                // For non-bindless tests, bind the wrong image handle to all
                // bindings except for <usedBinding>.
                if (accessVariant != AccessVariantImageBindless) {
                    for (int i = 0; i < imageBindingsUsed; i++) {
                        int idx = (i == usedBinding) ? texSizeIdx : ((texSizeIdx+1) % nTextureSizes);
                        if (m_variant == TestVariantTexelFetch) {
                            queueCB.BindTexture(ShaderStage::FRAGMENT, i, texHandles[textype][idx]);
                        } else {
                            queueCB.BindImage(ShaderStage::FRAGMENT, i, imageHandles[textype][idx]);
                        }
                    }
                }

                int row = accessVariant * nTextureSizes + texSizeIdx;
                assert(row < cellsY);
                queueCB.SetViewportScissor(textype * cellSpacing + cellPadding, row * cellSpacing + cellPadding, cellSize, cellSize);
                queueCB.BindProgram(programs[textype][accessVariant], ShaderStageBits::ALL_GRAPHICS_BITS);
                queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

                // For the store test, we check the results of the store using a load program.
                if (m_variant == TestVariantStore) {
                    queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS | BarrierBits::ILWALIDATE_TEXTURE);
                    queueCB.BindProgram(loadPrograms[textype][accessVariant], ShaderStageBits::ALL_GRAPHICS_BITS);
                    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
                }
            }
        }
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNImageQueryTest, lwn_image_targets_load,    (LWNImageQueryTest::TestVariantLoad));
OGTEST_CppTest(LWNImageQueryTest, lwn_image_targets_store,   (LWNImageQueryTest::TestVariantStore));
OGTEST_CppTest(LWNImageQueryTest, lwn_image_targets_query,   (LWNImageQueryTest::TestVariantQuery));
OGTEST_CppTest(LWNImageQueryTest, lwn_image_targets_texload, (LWNImageQueryTest::TestVariantTexelFetch));
