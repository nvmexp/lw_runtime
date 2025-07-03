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

class LWNTESTGLSLInitializedArrayTest
{
private:
    static const int cellsX = 32;
    static const int cellsY = 24;

    // ArrayType indicates whether to test an initialized or a constructed array.
    // Initialized arrays in GLSL look and are accessed like
    // int arry[3] = {1, 2, 3};
    // ...
    // int val = arry[1];
    //
    // Constructed arrays in GLSL look like
    // int val = int [3] (1, 2, 3)[1];
    enum ArrayType {
        TYPE_INITIALIZED = 0,
        TYPE_CONSTRUCTED = 1,
        TYPE_COUNT
    };

    // Indicates whether to use const qualifiers or not for the declared array.
    enum ArrayQualifier {
        QUALIFIER_CONST = 0,      // const qualified
        QUALIFIER_NONE = 1,       // no qualifier
        QUALIFIER_COUNT
    };

    // Enum indicating how the array is accessed, either with static indexed or dynamic indexing.
    enum ArrayAccessType {
        ACCESS_TYPE_DYNAMIC = 0,
        ACCESS_TYPE_STATIC = 1,
        ACCESS_TYPE_COUNT
    };

    enum ResourcesType {
        RESOURCES_NONE = 0,
        RESOURCES_SAMPLER = 1,
        RESOURCES_IMAGE = 2,
        RESOURCES_IMAGE_AND_SAMPLER = 3,
        RESOURCES_COUNT
    };

    // Test parameters used to construct the vertex shader string.
    struct TestParams {
        ArrayType m_arrayType;
        ArrayQualifier m_arrayQualifier;
        ArrayAccessType m_arrayAccessType;
        ResourcesType m_resourcesType;

        int m_arrayIndex; // Used to index into an initialized index array, which is then used to index the expected value array.

        TestParams(ArrayType arrayType, ArrayQualifier arrayQualifier,
                   ArrayAccessType arrayAccessType, int arrayIndex,
                   ResourcesType resourcesType) :
            m_arrayType(arrayType), m_arrayQualifier(arrayQualifier),
            m_arrayAccessType(arrayAccessType), m_resourcesType(resourcesType),
            m_arrayIndex(arrayIndex)
        {}
    };

public:
    // Create the program using shaders constructed from the parameters in <params>.  The variable <arrayString>
    // denotes the initializer string of the array to be tested.
    LWNboolean CreateProgram(Program *pgm, const TestParams & params, const char *arrayString) const;

    // Run the tests.
    void RunTests() const;

    LWNTEST_CppMethods();
};

lwString LWNTESTGLSLInitializedArrayTest::getDescription() const
{
    lwStringBuf sb;

    sb << "Tests initialized and constructed arrays in GLSL shaders.  Various initialized arrays are\n"
          "declared in the vertex and fragment shaders.  Each iteration of this test controls the properties\n"
          "of an initialized array in the vertex shaders.  Green indicates passing tests.\n"
          "The test cycles through multiple properties of the initializer array in this order:\n"
          "This test contains 4 rows of squares, each of length 32.  Each row corresponds to a set of tests\n"
          "which are run in the presence of image and/or sampler resources.  The first row from the bottom doesn't use any image/samplers,\n"
          "the 2nd row from the bottom uses a full sampler array (32 bindings), the 3rd row from the bottom uses a full image array (8) bindings,\n"
          "and the 4th row from the bottom uses both a full set of images and sampler resources.  Each resource is a 1x1 texture which contains\n"
          "all ones for each channel.  These are added to ensure the expected value is met.\n"
          "Within each row, there are groups of 8 squares, where each group selects a different expected vec4 from the\n"
          "initialized array by indexing into a second initialized index array in the shader.\n"
          "Each iteration always sends array index 2, but the order in which the entries in the expected\n"
          "value array are permuted each time.\n"
          "Within each group of 8, the first group of 4 has initialized arrays, and the second group of 4 has constructed arrays.\n"
          "Within each group of 4, the first group of 2 uses a const qualifier for the initialized array, the second group of 2 has no qualifier.\n"
          "Within each group of 2, the first square uses dynamic indexing from a UBO value, the second square uses static indexing.\n"
          "There are 128 squares in total.";

    return sb.str();
}

int LWNTESTGLSLInitializedArrayTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(38, 3);
}

LWNboolean LWNTESTGLSLInitializedArrayTest::CreateProgram(Program *pgm, const TestParams & params, const char *arrayString) const
{
    LWNboolean retVal = false;

    VertexShader vs(440);
    vs << "layout(location=0) in vec2 position;\n"
          "layout(location=1) in vec4 expectedVec;\n"
          "out vec3 ocolor;\n"
          "out int fragIndex;\n"
          "uniform Block {\n"
          "    int arrayIndex;\n"
          "};\n"
          "uniform sampler2D tex[32];\n"
          "layout(rgba8) uniform image2D img[8];\n";

    // Use const qualifiers or no qualifier on the array.
    if (params.m_arrayType == TYPE_INITIALIZED) {
        if (params.m_arrayQualifier == QUALIFIER_CONST) {
            vs << "const ";
        } else {
            assert(params.m_arrayQualifier == QUALIFIER_NONE);
        }

        vs << " vec4 expectedVecs [4] = {" << arrayString << "};";
    } else {
        assert(params.m_arrayType == TYPE_CONSTRUCTED);
        vs << " vec4 expectedVecs [4] = vec4[4] ( " << arrayString << " );\n";
    }

    vs << "void main() {\n"
          "  gl_Position = vec4(position, 0.0, 1.0);\n"
          "  const int expectedVecIndex[4] = {0, 1, 2, 3};\n"
          // Obtain the index for the expected value array by indexing with
          // the UBO variable.
          "  int index = expectedVecIndex[arrayIndex];\n";

    vs << "vec4 obtainedVec = expectedVecs";

    vs << "[";

    if (params.m_arrayAccessType == ACCESS_TYPE_DYNAMIC) {
        // Use the index obtained from accessing the expectedVecIndex array with
        // the UBO variable.
        vs << "index";
    } else if (params.m_arrayAccessType == ACCESS_TYPE_STATIC) {
        // Hardcode the index value.
        vs << params.m_arrayIndex;
    }

    vs << "];\n";

    bool outputSamplerShader =
        (params.m_resourcesType == RESOURCES_SAMPLER || params.m_resourcesType == RESOURCES_IMAGE_AND_SAMPLER);
    bool outputImageShader =
        (params.m_resourcesType == RESOURCES_IMAGE || params.m_resourcesType == RESOURCES_IMAGE_AND_SAMPLER);

    // For outputs which contain samplers/images, query the resource and add up each component.  Each component
    // is expected to be 1.
    if (outputSamplerShader) {
        vs << "  float texTotal = 0;\n"
              "  for (int i = 0; i < 32; ++i) {\n"
              "    vec4 texSample = texture(tex[i], vec2(0.0));\n"
              "    texTotal += texSample.x + texSample.y + texSample.z + texSample.a;\n"
              "  }\n";
    }

    if (outputImageShader) {
        vs << "  float imgTotal = 0;\n"
              "  for (int i = 0; i < 8; ++i) {\n"
              "    vec4 imgSample = imageLoad(img[i], ivec2(0));\n"
              "    imgTotal += imgSample.x + imgSample.y + imgSample.x + imgSample.a;\n"
              "  }\n";
    }

    vs << "  if ((obtainedVec == expectedVec)\n";

    // Check that all components of each queried resource were 1.
    if (outputSamplerShader) {
        vs << "      && (texTotal == 32*4)\n";
    }

    if (outputImageShader) {
        vs << "      && (imgTotal == 8*4)\n";
    }

    vs << "     ) {\n"
      // Write 0.5 to the green channel.  Another 0.5 gets added in the fragment shader.
      "      ocolor = vec3(0.0, 0.5, 0.0);\n"
      "  } else {\n"
      "    ocolor = vec3(1.0, 0.0, 0.0);\n"
      "  }\n"
      "  fragIndex = 1;\n"
      "}\n";

    FragmentShader fs(440);
    fs << "in vec3 ocolor;\n"
          "flat in int fragIndex;\n"
          "out vec4 fcolor;\n"
          // For added measure, ensure that initialized arrays using the input varying
          // can be used to dynamically index the array.  This value, when added to the
          // output from the vertex program, will give either green (passing) or a different color (fail).
          "const vec3 staticColors[3] = vec3[3] (vec3(1.0, 0.0, 0.0), vec3(0.0, 0.5, 0.0), vec3(0.0, 0.0, 1.0));\n"
          "void main() {\n"
          "  fcolor = vec4((ocolor.xyz + staticColors[fragIndex]), 1.0);\n"
          "}\n";


    retVal = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    return retVal;
}

void LWNTESTGLSLInitializedArrayTest::RunTests() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Vertex attributes
    struct Vertex {
        dt::vec2 position;
        dt::vec4 expectedVec;
    };

    // Uniform block
    struct UniformBlock {
        int arrayIndex;
    };


    // This test allocates one large pool of "dataPoolSize" used for vertex attributes and the uniform block.
    LWNsizeiptr dataPoolSize = 512 * 1024;

    MemoryPoolAllocator dataPoolAllocator(device, NULL, dataPoolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    // UBO
    BufferAlignBits uboAlignBits = BufferAlignBits(BUFFER_ALIGN_UNIFORM_BIT |
                                                   BUFFER_ALIGN_SHADER_STORAGE_BIT |
                                                   BUFFER_ALIGN_COPY_READ_BIT);
    Buffer *uboBuffer = dataPoolAllocator.allocBuffer(&bb, uboAlignBits, sizeof(UniformBlock));
    BufferAddress uboBufferAddr = uboBuffer->GetAddress();
    UniformBlock *uboBufferMap = (UniformBlock *) uboBuffer->Map();
    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboBufferAddr, sizeof(UniformBlock));

    // Create the 1x1 textures for tests with images and/or samplers.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    Sampler *smp = sb.CreateSampler();
    BufferAlignBits samplerBufferAlignBits = BufferAlignBits(BUFFER_ALIGN_UNIFORM_BIT |
                                                             BUFFER_ALIGN_IMAGE_BIT |
                                                             BUFFER_ALIGN_COPY_READ_BIT);
    Buffer *samplerBuffer = dataPoolAllocator.allocBuffer(&bb, samplerBufferAlignBits, dataPoolSize/4);
    BufferAddress samplerBufferAddr = samplerBuffer->GetAddress();
    unsigned char *samplerBufferMap = (unsigned char *)samplerBuffer->Map();

    // Set all entries to 1.  The shaders will add these up for each binding to ensure they are all saved
    // properly.
    memset(samplerBufferMap, 0xFFFFFFFF, dataPoolSize/4);

    // Allocate textures from the same prefilled buffer that ubos/ssbos would be backed with.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetFlags(TextureFlags::IMAGE);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(1, 1);
    tb.SetFormat(Format::RGBA8);
    tb.SetLevels(1);
    LWNsizeiptr texSize = tb.GetPaddedStorageSize();

    Texture *tex[32];
    Texture *img[8];

    // Allocate enough where we can use for images (8 bindings) and samplers (32 bindings), or both (40 bindings).
    MemoryPoolAllocator texAllocator(device, NULL, texSize*64, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    // Set all arrays of resource to all 1's.  These are set/bound regardless of the test, but they might not be accessed
    // in the shader on all variations.
    CopyRegion copyRegion = { 0, 0, 0, 1, 1, 1 };

    for (int i = 0; i < 32; ++i) {
        tex[i] = texAllocator.allocTexture(&tb);
        TextureHandle texHandle = device->GetTextureHandle(tex[i]->GetRegisteredTextureID(), smp->GetRegisteredID());
        queueCB.CopyBufferToTexture(samplerBufferAddr + i * sizeof(unsigned char) * 4, tex[i], NULL, &copyRegion, CopyFlags::NONE);
        queueCB.BindTexture(ShaderStage::VERTEX, i, texHandle);
    }

    // Create our images using the same data as the textures.
    for (int i = 32; i < 40; ++i) {
        LWNuint imageIndex = i - 32;
        img[imageIndex] = texAllocator.allocTexture(&tb);
        LWNuint id = g_lwnTexIDPool->RegisterImage(img[imageIndex]);
        ImageHandle imageHandle = device->GetImageHandle(id);
        queueCB.CopyBufferToTexture(samplerBufferAddr + imageIndex * sizeof(unsigned char) * 4, img[imageIndex], NULL, &copyRegion, CopyFlags::NONE);
        queueCB.BindImage(ShaderStage::VERTEX, imageIndex, imageHandle);
    }

    // Vertex attributes
    bb.SetDefaults();
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, expectedVec);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = dataPoolAllocator.allocBuffer(&bb, BUFFER_ALIGN_VERTEX_BIT, dataPoolSize/4);
    BufferAddress vboAddr = vbo->GetAddress();
    Vertex *vboMap = (Vertex *)vbo->Map();

    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, dataPoolSize/4);

    static const int vpWidth = lwrrentWindowWidth / cellsX;
    static const int vpHeight = lwrrentWindowHeight / cellsY;
    int vnum = 0;
    int lwrrRow = 0;
    int lwrrCol = 0;
    Program *pgm = NULL;

    LWNboolean loaded = LWN_FALSE;

    // Array of expected values.  Each iteration will be used as the first
    // dimensional index in the shader.
    float arrayVals[4][4] = {
        {0.0, 0.25, 0.5, 0.75},
        {0.25, 0.5, 0.75, 0.0},
        {0.5, 0.75, 0.0, 0.25},
        {0.75, 0.0, 0.25, 0.5},
    };

    // Iterate through the shader parameter combination.
    int totalCount = 0;
    for (int resourcesType = 0; resourcesType < RESOURCES_COUNT; ++resourcesType) {
        for (int iter = 0; iter < 4; ++iter) {
            // Select the third entry in the array.  All array values will be chosen since the array is permuted
            // each iteration.
            uboBufferMap->arrayIndex = 2;
            for (int arrayType = 0; arrayType < TYPE_COUNT; ++arrayType) {
                for (int arrayQualifier = 0; arrayQualifier < QUALIFIER_COUNT; ++arrayQualifier) {
                    for (int arrayAccessType = 0; arrayAccessType < ACCESS_TYPE_COUNT; ++arrayAccessType) {
                        // Create the array string to be used when initializing the array.
                        char arrayString[256];
                        // A permutation index to mix up the order in which the array rows are defined.
                        int pIndex[4] = {
                            iter,
                            (iter + 1) % 4,
                            (iter + 2) % 4,
                            (iter + 3) % 4,
                        };

                        lwog_snprintf(arrayString, 256, "vec4(%f, %f, %f, %f),\n"
                                "                        vec4(%f, %f, %f, %f),\n"
                                "                        vec4(%f, %f, %f, %f),\n"
                                "                        vec4(%f, %f, %f, %f)\n",
                            arrayVals[pIndex[0]][0], arrayVals[pIndex[0]][1], arrayVals[pIndex[0]][2], arrayVals[pIndex[0]][3],
                            arrayVals[pIndex[1]][0], arrayVals[pIndex[1]][1], arrayVals[pIndex[1]][2], arrayVals[pIndex[1]][3],
                            arrayVals[pIndex[2]][0], arrayVals[pIndex[2]][1], arrayVals[pIndex[2]][2], arrayVals[pIndex[2]][3],
                            arrayVals[pIndex[3]][0], arrayVals[pIndex[3]][1], arrayVals[pIndex[3]][2], arrayVals[pIndex[3]][3]);

                        TestParams params (ArrayType(arrayType), ArrayQualifier(arrayQualifier),
                                           ArrayAccessType(arrayAccessType), uboBufferMap->arrayIndex,
                                           ResourcesType(resourcesType));

                        lwrrCol = totalCount % cellsX;
                        lwrrRow = totalCount / cellsX;
                        totalCount++;
                        queueCB.SetViewportScissor(lwrrCol * vpWidth + 2, lwrrRow * vpHeight + 2, vpWidth - 4, vpHeight - 4);

                        pgm = device->CreateProgram();
                        loaded = CreateProgram(pgm, params, arrayString);

                        if (loaded) {
                            queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

                            for (int v = 0; v < 4; v++) {
                                vboMap[vnum + v].position[0] = (v & 2) ? +1.0 : -1.0;
                                vboMap[vnum + v].position[1] = (v & 1) ? +1.0 : -1.0;

                                for (int component = 0; component < 4; ++component) {
                                    // Use the 3rd entry in the array.
                                    vboMap[vnum + v].expectedVec[component] = arrayVals[pIndex[2]][component];
                                }
                            }
                            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, vnum, 4);
                            vnum += 4;
                        }
                        else {
                            queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
                        }
                        queueCB.submit();
                    }
                }
            }
            // Finish the queue within the outer loop since each iteration updates the
            // values in UBO memory.
            queue->Finish();
        }
    }
}

void LWNTESTGLSLInitializedArrayTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    RunTests();
}

OGTEST_CppTest(LWNTESTGLSLInitializedArrayTest, lwn_glsl_initialized_array, );
