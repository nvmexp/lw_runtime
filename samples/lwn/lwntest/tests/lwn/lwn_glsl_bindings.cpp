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
#include "cmdline.h"

using namespace lwn;

class LWNGLSLBindingsTest
{
    static const int cellsX = 32;
    static const int cellsY = 24;

    // Bytes needed for each resource.
    static const int resBufferSize = 256;

    // Number of bindings to test for each resource type.
    static const int resNumTexture = 32;
    static const int resNumImage = 8;
    static const int resNumUBO = 14;
    static const int resNumSSBO = 16;

    // Maximum total bindings possible.
    static const int resTotalCount = 32;

    class VariableType {
    public:
        enum Enum {
            // Single uniform block, explicit binding
            VARIABLE_SINGLE_BOUND,
            // Array of uniform blocks, explicit binding
            VARIABLE_ARRAY_BOUND,
            // Single uniform block, not explicit binding
            VARIABLE_SINGLE_UNBOUND,
            // Array of uniform blocks, no explicit binding
            VARIABLE_ARRAY_UNBOUND,
            VARIABLE_TYPE_COUNT,
        };
        Enum m_type;
        VariableType(int e) : m_type(Enum(e)) {}

        // Are we testing arrays
        bool isArray() const
        {
            return m_type == VARIABLE_ARRAY_BOUND || m_type == VARIABLE_ARRAY_UNBOUND;
        }

        // Are we testing explicitly bound resources
        bool isBound() const
        {
            return m_type == VARIABLE_ARRAY_BOUND || m_type == VARIABLE_SINGLE_BOUND;
        }
    };

public:
    enum Usage {
        // Tests sampler resources 
        USAGE_TEXTURE,
        // Tests image resources
        USAGE_IMAGE,
        // Tests normal uniform resources
        USAGE_UBO,
        // Tests shader storage block resources
        USAGE_SSBO,
    };

    ShaderStage m_stage;
    Usage m_usage;
    LWNGLSLBindingsTest(ShaderStage stage, Usage usage) : m_stage(stage), m_usage(usage) {}

    LWNTEST_CppMethods();
};

lwString LWNGLSLBindingsTest::getDescription() const
{
    lwStringBuf sb;
    int numBindings = 0;
    sb << "GLSL bindings test for LWN.\nThis test is designed to test that ";
    switch (m_usage) {
    case USAGE_TEXTURE:
        sb << "sampler";
        numBindings = resNumTexture;
        break;
    case USAGE_IMAGE:
        sb << "image";
        numBindings = resNumImage;
        break;
    case USAGE_UBO:
        sb << "uniform block";
        numBindings = resNumUBO;
        break;
    case USAGE_SSBO:
        sb << "shader storage block";
        numBindings = resNumSSBO;
        break;
    default:
        assert(0);
        break;
    };
    sb << " bindings are assigned correctly by the GLSL compiler\n"
          "There are 4 groups of 4 rows, where each group corresponds to a single variation (ARRAY/SINGLE, BOUND/UNBOUND).\n"
          "\n"
          "From bottom to top:\n"
          "* Group 1: Single resource, binding explicitly set in the shader.\n"
          "* Group 2: Array of resources, binding for first entry explicitly specified in the shader.\n"
          "* Group 3: Single resource, binding not explicitly specified in shader.\n"
          "* Group 4: Array of resources, binding not explicitly specified in shader.\n"
          "\n"
          "Within each group there consists 4 rows.  Green squares indicate pass, while non-green indicates failure.\nEach square uses a shader where the binding for the resource\n"
          "is equal to the column number (starting with 0).\n"
          "\n"
          "Within each group, the rows from bottom to top:\n"
          "* Row 1: Compiles a shader for each possible binding, then checks the program interface query returns the correct binding information.\n"
          "* Row 2: Compiles a shader for each possible binding for the resource and sets the data in the LWN API based on that binding.\n"
          "* Row 3: Compiles and tests the program interface query returns the correct binding information in the presense of an alternate resource.\n"
          "* Row 4: Compiles and tests that alternate resources not explicitly bound get assigned to one of the known slots.\n"
          "\n"
          "There are a total of ";
    sb << numBindings;
    sb << " bindings tested (maximum number of columns for binding variants or 2X the number of columns for array variants);\n";

    return sb.str();
}

int LWNGLSLBindingsTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(24, 1);
}

void LWNGLSLBindingsTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    Sampler *smp = sb.CreateSampler();

    LWNuint resDataSize = resTotalCount * resBufferSize;

    struct Vertex {
        dt::vec2 position;
        dt::ivec1 selector;
        dt::vec1 expected;
        dt::vec1 altExpected;
    };

    // This test allocates one large buffer of "vboSize".  The buffer is filled in such that the data for each resource
    // corresponds to to the binding of that resource.  The buffer fills all binding slots with the values, so that if
    // any binding is made in the shader, the value of the variables in the block contain the binding of the block itself.
    // Textures and images are created from the same buffer so that queries will also pick the data corresponding to the
    // sampler/image binding location.
    LWNsizeiptr vboSize = 640*1024;

    MemoryPoolAllocator dataPoolAllocator(device, NULL, resDataSize + vboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    BufferAlignBits alignBits = BufferAlignBits(BUFFER_ALIGN_UNIFORM_BIT | BUFFER_ALIGN_IMAGE_BIT |
                                                BUFFER_ALIGN_COPY_READ_BIT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *resBuffer = dataPoolAllocator.allocBuffer(&bb, alignBits, resDataSize);
    BufferAddress resBufferAddr = resBuffer->GetAddress();
    LWNfloat *resBufferMap = (LWNfloat *) resBuffer->Map();
    for (int i = 0; i < resTotalCount; i++) {
        resBufferMap[i * (resBufferSize / sizeof(LWNfloat))] = LWNfloat(i + 1);
    }

    // Allocate textures from the same prefilled buffer that ubos/ssbos would be backed with.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetFlags(TextureFlags::IMAGE);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(1, 1);
    tb.SetFormat(Format::R32F);
    tb.SetLevels(1);
    LWNsizeiptr texSize = tb.GetPaddedStorageSize();

    LWNsizeiptr totalTexSize = resNumTexture * texSize;
    MemoryPoolAllocator texAllocator(device, NULL, totalTexSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    CopyRegion copyRegion = { 0, 0, 0, 1, 1, 1 };

    Texture *tex[resNumTexture];

    switch (m_usage) {
    case USAGE_TEXTURE:
        for (int i = 0; i < resNumTexture; i++) {
            tex[i] = texAllocator.allocTexture(&tb);
            TextureHandle texHandle = device->GetTextureHandle(tex[i]->GetRegisteredTextureID(), smp->GetRegisteredID());
            queueCB.CopyBufferToTexture(resBufferAddr + i * resBufferSize, tex[i], NULL, &copyRegion, CopyFlags::NONE);
            queueCB.BindTexture(ShaderStage::FRAGMENT, i, texHandle);
        }
        break;
    case USAGE_IMAGE:
        for (int i = 0; i < resNumImage; i++) {
            tex[i] = texAllocator.allocTexture(&tb);
            LWNuint id = g_lwnTexIDPool->RegisterImage(tex[i]);
            ImageHandle imageHandle = device->GetImageHandle(id);
            queueCB.CopyBufferToTexture(resBufferAddr + i * resBufferSize, tex[i], NULL, &copyRegion, CopyFlags::NONE);
            queueCB.BindImage(ShaderStage::FRAGMENT, i, imageHandle);
        }
        break;
    case USAGE_UBO:
        for (int i = 0; i < resNumUBO; i++) {
            queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, i, resBufferAddr + i * resBufferSize, sizeof(LWNfloat));
        }
        break;
    case USAGE_SSBO:
        for (int i = 0; i < resNumSSBO; i++) {
            queueCB.BindStorageBuffer(ShaderStage::FRAGMENT, i, resBufferAddr + i * resBufferSize, sizeof(LWNfloat));
        }
        break;
    }

    bb.SetDefaults();

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, selector);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, expected);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, altExpected);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = dataPoolAllocator.allocBuffer(&bb, BUFFER_ALIGN_VERTEX_BIT, vboSize);
    BufferAddress vboAddr = vbo->GetAddress();
    Vertex *vboMap = (Vertex *) vbo->Map();

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec2 position;\n"
        "layout(location=1) in int selector;\n"
        "layout(location=2) in float expected;\n"
        "layout(location=3) in float altExpected;\n"
        "out Block {\n"
        "  flat int selector;\n"
        "  float expected;\n"
        "  float altExpected;\n"
        "} v;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "  v.selector = selector;\n"
        "  v.expected = expected;\n"
        "  v.altExpected = altExpected;\n"
        "}\n";

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, vboSize);

    static const int resCount[] = { resNumTexture, resNumImage, resNumUBO, resNumSSBO };
    static const int vpWidth = lwrrentWindowWidth / cellsX;
    static const int vpHeight = lwrrentWindowHeight / cellsY;
    int vnum = 0;
    int row = 0;
    Program *pgm = NULL;
    LWNboolean loaded = LWN_FALSE;
    int bindingLocation = -1;
    int altBindingLocation = -1;

    // Loop through each variable type
    for (int vtindex = 0; vtindex < VariableType::VARIABLE_TYPE_COUNT; vtindex++) {
        // Alternate between testing the alternate non-bound blocks and the specified block.
        for (int altvar = 0; altvar < 2; altvar++) {
            VariableType vt = VariableType(vtindex);
            // Loop through each possible resource binding.
            for (int resource = 0; resource < resCount[m_usage]; resource++) {

                if (!vt.isBound()) {
                    // If we are using an array, only use half the possible number of bindings (the other
                    // half of the binding set will be filled in by the "alt" array if there is one).
                    if (resource >= (vt.isArray() ? resCount[m_usage] / 2 : 1)) {
                        continue;
                    }
                }

                int resindex = vt.isArray() ? resource % (resCount[m_usage] / 2) : 0;

                // Only recompile on the first entry of arrays, or on every entry if not arrays.
                if (resindex == 0) {
                    FragmentShader fs(440);
                    if (altvar) {
                        switch (m_usage) {
                        case USAGE_TEXTURE: fs << "uniform sampler2D texalt;\n"; break;
                        case USAGE_IMAGE:   fs << "layout(r32f) uniform image2D imagealt;\n"; break;
                        case USAGE_UBO:     fs << "uniform BlockAlt { float value; } blockalt;\n"; break;
                        case USAGE_SSBO:    fs << "buffer BlockAlt { float value; } blockalt;\n"; break;
                        default:            assert(0); break;
                        }
                    }
                    // Set the binding layout for this resource if we are testing the "bound" variant.
                    if (vt.isBound()) {
                        fs << "layout(binding = " << resource - resindex << ") ";
                    }
                    switch (m_usage) {
                    case USAGE_TEXTURE: fs << "uniform sampler2D tex"; break;
                    case USAGE_IMAGE:   fs << "layout(r32f) uniform image2D image"; break;
                    case USAGE_UBO:     fs << "uniform Block { float value; } block"; break;
                    case USAGE_SSBO:    fs << "buffer Block { float value; } block"; break;
                    default:            assert(0);
                    }
                    if (vt.isArray()) {
                        fs << "[" << resCount[m_usage] / 2 << "]";
                    }
                    fs << ";\n";
                    fs <<
                        "in Block {\n"
                        "  flat int selector;\n"
                        "  float expected;\n"
                        "  float altExpected;\n"
                        "} v;\n"
                        "out vec4 color;\n"
                        "void main() {\n"
                        "  float value;\n";
                    switch (m_usage) {
                    case USAGE_TEXTURE:
                        fs << "  value = texture(tex";
                        if (vt.isArray()) fs << "[v.selector]";
                        fs << ", vec2(0.0)).x;\n";
                        break;
                    case USAGE_IMAGE:
                        fs << "  value = imageLoad(image";
                        if (vt.isArray()) fs << "[v.selector]";
                        fs << ", ivec2(0)).x;\n";
                        break;
                    case USAGE_UBO:
                    case USAGE_SSBO:
                        fs << "  value = block";
                        if (vt.isArray()) fs << "[v.selector]";
                        fs << ".value;\n";
                        break;
                    default:
                        assert(0);
                    }
                    fs <<
                        "  if (value == v.expected) {\n"
                        "    color = vec4(0.0, 1.0, 0.0, 1.0);\n"
                        "  } else {\n"
                        "    color = vec4(1.0, 0.0, 0.0, 1.0);\n"
                        "  }\n";
                    if (altvar) {
                        // Additionally, test the alternate block/image/sampler binding is what is expected as obtained from the program interface query.
                        fs << "  if (v.altExpected != ";
                        switch (m_usage) {
                        case USAGE_TEXTURE: fs << "texture(texalt, vec2(0.0)).x"; break;
                        case USAGE_IMAGE:   fs << "imageLoad(imagealt, ivec2(0)).x"; break;
                        case USAGE_UBO:
                        case USAGE_SSBO:    fs << "blockalt.value"; break;
                        default:            assert(0);
                        }
                        fs <<
                            ") {\n"
                            "    color = vec4(0.9, 0.0, 0.0, 1.0);\n"
                            "  }\n";
                    }
                    fs << "}\n";
                    pgm = device->CreateProgram();
                    loaded = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
                    if (loaded) {
                        queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
                    } else {
                        printf("Log: %s\n", g_glslcHelper->GetInfoLog());
                    }
                }
                queueCB.SetViewportScissor(resource * vpWidth + 2, row * vpHeight + 2, vpWidth - 4, vpHeight - 4);
                if (loaded) {
                    LWNprogramResourceType resType = LWN_PROGRAM_RESOURCE_TYPE_SAMPLER;
                    const char *name = "invalid";
                    const char *altName = "altIlwalid";
                    LWNboolean usesSpirv = g_glslcHelper->WasLastCompileSpirv();

                    // Assign name based on usage
                    switch (m_usage) {
                    case USAGE_TEXTURE:
                        resType = LWN_PROGRAM_RESOURCE_TYPE_SAMPLER;
                        name = vt.isArray() ? "tex[0]" : "tex";
                        altName = "texalt";
                        break;
                    case USAGE_IMAGE:
                        resType = LWN_PROGRAM_RESOURCE_TYPE_IMAGE;
                        name = vt.isArray() ? "image[0]" : "image";
                        altName = "imagealt";
                        break;
                    case USAGE_UBO:
                        resType = LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK;
                        if (!usesSpirv) {
                            name = vt.isArray() ? "Block[0]" : "Block";
                            altName = "BlockAlt";
                        } else {
                            name = vt.isArray() ? "Block.block[0]" : "Block.block";
                            altName = "BlockAlt.blockalt";
                        }
                        break;
                    case USAGE_SSBO:
                        resType = LWN_PROGRAM_RESOURCE_TYPE_STORAGE_BLOCK;
                        if (!usesSpirv) {
                            name = vt.isArray() ? "Block[0]" : "Block";
                            altName = "BlockAlt";
                        } else {
                            name = vt.isArray() ? "Block.block[0]" : "Block.block";
                            altName = "BlockAlt.blockalt";
                        }
                        break;
                    default:
                        assert(0);
                    }
                    bindingLocation = g_glslcHelper->ProgramGetResourceLocation(pgm, ShaderStage::FRAGMENT, resType, name) + resindex;
                    altBindingLocation = g_glslcHelper->ProgramGetResourceLocation(pgm, ShaderStage::FRAGMENT, resType, altName);
                    bool passed = true;
                    if (vt.isBound()) {
                        passed = (bindingLocation == resource);
                    } else {
                        passed = (bindingLocation >= 0);
                    }
                    if (passed) {
                        queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
                    } else {
                        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
                    }
                } else {
                    queueCB.ClearColor(0, 0.5, 0.0, 0.0, 1.0);
                }

                queueCB.SetViewportScissor(resource * vpWidth + 2, (row + 1) * vpHeight + 2, vpWidth - 4, vpHeight - 4);
                if (loaded) {
                    for (int v = 0; v < 4; v++) {
                        vboMap[vnum + v].position[0] = (v & 2) ? +1.0 : -1.0;
                        vboMap[vnum + v].position[1] = (v & 1) ? +1.0 : -1.0;
                        vboMap[vnum + v].selector = resindex;
                        if (vt.isBound()) {
                            vboMap[vnum + v].expected = resource + 1.0;
                        } else {
                            vboMap[vnum + v].expected = bindingLocation + 1.0;
                        }
                        if (altvar) {
                            vboMap[vnum + v].altExpected = altBindingLocation + 1.0;
                        }
                    }
                    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, vnum, 4);
                    vnum += 4;
                } else {
                    queueCB.ClearColor(0, 0.5, 0.0, 0.0, 1.0);
                }
            }
            row += 2;
        }
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNGLSLBindingsTest, lwn_glsl_bindings_texture,
               (ShaderStage::FRAGMENT, LWNGLSLBindingsTest::USAGE_TEXTURE));
OGTEST_CppTest(LWNGLSLBindingsTest, lwn_glsl_bindings_image,
               (ShaderStage::FRAGMENT, LWNGLSLBindingsTest::USAGE_IMAGE));
OGTEST_CppTest(LWNGLSLBindingsTest, lwn_glsl_bindings_ubo,
               (ShaderStage::FRAGMENT, LWNGLSLBindingsTest::USAGE_UBO));
OGTEST_CppTest(LWNGLSLBindingsTest, lwn_glsl_bindings_ssbo,
               (ShaderStage::FRAGMENT, LWNGLSLBindingsTest::USAGE_SSBO));
