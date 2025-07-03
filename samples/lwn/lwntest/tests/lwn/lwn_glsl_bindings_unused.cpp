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

// We test leaving the first entry of an image / sampler array unused (but declared)
// during the array tests.
// This is to exercise a test case for bug 1971498 where the reflection info for the array
// would not show up unless the first entry in the array was being used.
// This number must be less than the max number of arrays in all resource types,
// otherwise assertions will fire.
// This number must also be an array which will not be assigned an explicit binding,
// otherwise assertions will fire.
#define ARRY_BLOCK_WITH_NO_FIRST_ENTRY 2

class LWNTESTGLSLUnusedBindingsTest
{
public:
    // TEST_TYPE_SINGLE is used to test individual uniform blocks, while TEST_TYPE_ARRAY is for testing
    // arrays of uniform blocks.
    enum TestType {
        TEST_TYPE_SINGLE,
        TEST_TYPE_ARRAY
    };

    // Type of resource to test.
    enum Usage {
        USAGE_UBO = 0,
        USAGE_SSBO,
        USAGE_TEXTURE,
        USAGE_IMAGE,
        USAGE_TYPE_COUNT
    };

private:
    static const int cellsX = 32;
    static const int cellsY = 24;

    // Number of resources for each resource type.
    static const int resNumTexture = 32;
    static const int resNumImage = 8;
    static const int resNumUBO = 14;
    static const int resNumSSBO = 16;

    // Size of each array for the array test type.
    static const int arraySize = 2;

    // Bytes needed for each resource.
    static const int resBufferSize = 256;

    // Array size of each uniform declaration.  For TEST_TYPE_SINGLE, this is 1, otherwise m_arySize == arraySize.
    int m_arySize;

public:

    // Retrieves the first available slot in the used bit max.
    // validBitMask - Inidicates the valid slots which can be used (represented as a bit mask with LSB indicating binding slot "0").
    // maxBits - Total number of binding slots that can be used.
    // arySize - Size of each array.  Arrays must be able to fit into contiguous elements.
    // Returns the first free slot for the uniform block / array.
    static int GetFirstFreeSlot(int validBitMask, int maxBits, int arySize)
    {
        int freeSlot = -1;
        for (int i = 0; i < maxBits; i+=arySize) {
            bool isFree = true;

            // Check if all slots in this space are free.
            int queryMask = ((1 << arySize) - 1) << i;
            if ((validBitMask & queryMask) != queryMask) {
                isFree = false;
            }

            if (isFree) {
                freeSlot = i;
                break;
            }

            freeSlot = i;
        }

        // We found a free slot, report it.
        if ((freeSlot + arySize) <= maxBits) {
            return freeSlot;
        } else if (freeSlot != -1) {
            // This should not happen unless there is an error with setting up the test parameters.
            assert(!"Not enough free space to store block!\n");
            return -1;
        }

        return -1;
    }

    // Checks if the block (a uniform in SINGLE usage, an array of uniforms in array usage)
    // Since arrays of blocks are meant to be filled in, this function assumes if the first element is taken up, then
    // the rest of the slots for the array are also taken up.
    static bool IsBlockUsed(int blockNo, int usedMask, int arySize)
    {
        if (usedMask & (((1 << arySize) - 1) << blockNo * arySize)) {
            return true;
        }

        return false;
    }

    // Maps a block/array to a bit mask indicating resources that are used, with LSB indicating binding slot "0", and increasing
    // to GSB.
    // blockIdx - Indicates which block/resource (SINGLE case) or array (array case) to get the resource mask for.
    static int mapBlockToResourceMask(int blockIdx, int arySize)
    {
        int retMask = ((1 << arySize) - 1) << blockIdx * arySize;

        return retMask;
    }


    LWNTESTGLSLUnusedBindingsTest(Usage usage, TestType testType) :
        m_arySize((testType == TEST_TYPE_SINGLE? 1 : arraySize)) {}

    // Get the expected bindings, represented as a bit mask.
    // explicitMask - Inicates the slots which belong to uniforms containing a layout(binding =...) qualifer
    // usedMask - Indicates which slos are lwrrently oclwpied
    // resNum - Number of resources to use
    // numBlocks - the Number of uniform blocks / arrays being used.
    // usageType - the USAGE_* enumerator, used for leave-out-first entry testing.
    int GetExpectedMask(int explicitMask, int usedMask, int resNum, int numBlocks, Usage usageType) const;

    // Builds a vertex shader and a fragment shader which contains declared uniforms in the shaders.
    // The usedMask and explicitMask are used to drive the definition of these shaders.
    // The vertex shader is built with 4 uniforms declared and all are used.  The first
    // 2 uniforms declared share the uniform names with declared uniforms in the fragment
    // shader.  The other 2 uniforms declared in the vertex shader are not declared by the fragment
    // shader.
    // The fragment shader is built by declaring <n> uniforms, where <n> is determined by the number of arrays/blocks in the shader.
    // For non-array usage, <n> == <resNum>.  For array usage, <n> == <resNum>/<m_arySize>
    // The usedMask is used to actually use the uniforms in the fragment shader.  Since each
    // binding slot is backed by a bit mask corresponding to that slot, adding up the uniforms
    // as specified by the usedMask should produce the original expected value.
    // The explicitMask is used to assign layout specifiers to declared uniforms.
    //
    // The shaders also check to make sure the vertex bindings fall in the right place, and pass a magic
    // number (-666) to the fragment shader if these values are not expected.  This magic number is used to
    // indicate a failure from having unexpected bindings in the vertex shader.
    LWNboolean CreateProgram(Program *pgm, int usedMask, int explicitMask, int numBlocks, Usage usageType) const; 

    // Run a single test type.  Each test consists of 4 rows with a number of columns == <resNum>/<m_arySize>.
    void RunUsageTests(Usage type) const;

    LWNTEST_CppMethods();
};

int LWNTESTGLSLUnusedBindingsTest::GetExpectedMask(int explicitMask, int usedMask, int resNum, int numBlocks, Usage usageType) const
{
    assert(ARRY_BLOCK_WITH_NO_FIRST_ENTRY < numBlocks);

    // A mask indicating which bindings are used and also have explicit layouts
    int usedAndExplicitMask = (explicitMask & usedMask);

    // Given the current configuration, represents the expected used binding slots.  This is the function
    // return value.
    int expectedMask = usedAndExplicitMask;

    // Mask indicating which binding slots we can not occupy.  Initialize this to the bindings
    // which have explicit layout binding qualifiers.
    int offLimitsMask = explicitMask;

    // Mask indicating which binding slots are used and don't have an explicit layout binding.
    int usedAndNotExplicit = usedMask ^ usedAndExplicitMask;

    for (int j = 0; j < numBlocks; ++j) {
        offLimitsMask |= expectedMask;

        int arySizeMask = (1 << m_arySize) - 1;

        if ((usedAndNotExplicit & (arySizeMask << j*m_arySize)) == 0) {
            // This block is neither used nor explicitly bound, so ignore when
            // computing the expected bindings mask.
            continue;
        }

        // Find the first free slot of the next bit in the mask.
        int firstFreeSlot = GetFirstFreeSlot(~offLimitsMask, resNum, m_arySize);
        assert(firstFreeSlot > -1);

        for (int z = 0; z < m_arySize; ++z) {
            if (j == ARRY_BLOCK_WITH_NO_FIRST_ENTRY &&
                m_arySize > 1 &&
                z == 0 &&
                (usageType == USAGE_IMAGE || usageType == USAGE_TEXTURE)) {
                    // Leave out first entry of this array.
                    continue;
            }

            expectedMask |= (1 << (firstFreeSlot + z));
        }
    }

    return expectedMask;
}

lwString LWNTESTGLSLUnusedBindingsTest::getDescription() const
{
    const char *uniformStr = (m_arySize > 1 ? "uniform arrays" : "uniforms");
    const char *uniformStrSing = (m_arySize > 1 ? "uniform array" : "uniform");
    lwStringBuf sb;
    sb << "This test exercises binding shaders with multiple unused " << uniformStr << " and " << uniformStr << " shared between shader stages.\n"
          "To test unused " << uniformStr << ", the fragment shader will have a full set of " << uniformStr << " (corresponding to the resource\n"
          "being tested) declared, but not used.  As the columns of the test increase, the number of " << uniformStr << " that are used in the shaders\n"
          "are increased.  Each successive column will have 1 more " << uniformStrSing << " enabled from the end of the list in\n"
          "the order they are declared.  For example, in a test which tests 2 " << uniformStr << " for images, the images declared\n"
          "last and second to last in shader will be used by the shader.  The expectation is that those bindings will take\n"
          "up unused bindings slots.  The column number of each cell is how many " << uniformStr << " are used in the shader.\n"
          "There are 4 sets of 4 rows.  From bottom to top, each set of 4 rows corresponds to one of the 4 resource types:\n"
          "UBO uniforms, SSBO buffers, textures, and images.  Within each set, 4 different tests are run.  From bottom\n"
          "to top, these tests are:\n"
          "* Row 1: No layout qualifiers, test that successively enabling " << uniformStr << " will take up unused binding slots by querying\n"
          "         the program reflection information.\n"
          "* Row 2: No layout qualifiers, test that successively enabling " << uniformStr << " will take up unused binding slots by checking\n"
          "         the bit mask in the buffer backing the resources and render from it.\n"
          "* Row 3: Every other " << uniformStr << " declared will have an explicit layout associated with it, test that successively enabling\n"
          "         bindings fill up unused binding slots that do not have a uniform with a layout qualifier specifying that binding\n"
          "         slot by querying the program reflection information.\n"
          "* Row 4: Every other " << uniformStr << " declared will have an explicit layout associated with it, test that successively enabling\n"
          "         bindings fill up unused binding slots that do not have a " << uniformStr << " with a layout qualifier specifying that binding\n"
          "         slot by using the bit mask stored in the backing buffer to render.\n"
          "\n"
          "Additionally, the vertex shader is declared with 4 " << uniformStr << ".  2 of these " << uniformStr << " are shared with the fragment shader,\n"
          "and 2 of these " << uniformStr << " are unique to the vertex shader.  The purpose for the bindings in the vertex shader is to\n"
          "test that 1) " << uniformStr << " that are used in multiple stages can use different bindings, 2) " << uniformStr << " that are declared\n"
          "only in the vertex shader do not take up binding slots in the fragment shader where they are unused, and 3) some variety between blocks which could be\n"
          "used/unused differently in the vertex and fragment shaders.\n"
          "The first resource is unused in the vertex shader.  This will be used in the fragment shader on the last iteration, unused on other iterations.\n"
          "The second resource is used in the vertex shader.  This will be used in the fragment shader on the last 2 iterations.\n"
          "The third resource is used in the vertex shader.  This is unused in the fragment shader.\n"
          "The fourth resource is unused in the vertex shader and fragment shader.  The block contains two entries.\n"
          "The results of each test can be determined by the color of the cell:\n"
          "* Green      (0.0, 1.0, 0.0, 1.0) - Pass.  This is the only color to indicate a passing test.\n"
          "* Red        (1.0, 0.0, 0.0, 1.0) - Failed the bindings test (either expected bindings through reflection querying tests or from within the shader\n"
          "             through rendering tests\n"
          "* 50% Red    (0.5, 0.0, 0.0, 1.0) - Duplicate bindings were returned for a resource in a single stage.\n"
          "* 60% Red    (0.6, 0.0, 0.0, 1.0) - Error querying offsets for the uniforms in blocks.\n"
          "* 70% Red    (0.7, 0.0, 0.0, 1.0) - Error querying sizes for the uniforms in blocks.\n"
          "* 75% Red    (0.75, 0.0, 0.0, 1.0) - Unexpected bindings were reported by the vertex shader.\n";

    return sb.str();
}

int LWNTESTGLSLUnusedBindingsTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(38, 2);
}

LWNboolean LWNTESTGLSLUnusedBindingsTest::CreateProgram(Program *pgm, int usedMask, int explicitMask, int numBlocks, Usage usageType) const
{
    LWNboolean retVal = false;
    int vsShaderExpectedVal = 0;

    // Each vertex shader has 4 blocks or arrays declared inside of it.  We set up a mask indicating how many LSBs should be oclwpied
    // in the vertex shader and use that mask as the expected value to check for inside the vertex shader.
    // Only the second and third blocks are actually used in the vertex shader, and we are making the assumption that the vertex resource
    // bindings will be assigned to 0 and 1 and the unused resources will not take up binding slots.
    for (int i = 0; i < m_arySize; ++i) {
        vsShaderExpectedVal |= (0x3 << (2*i));
    }

    VertexShader vs(440);
    vs.addExtension(lwShaderExtension::LW_bindless_texture);
    vs << "layout(location=0) in vec2 position;\n"
          "layout(location=1) in int expected;\n";

    // Loop through all 4 uniforms/arrays inside the vertex shader to set up declarations.  The first 2 blocks/arrays are also
    // declared inside the fragment shader.  The second 2 blocks/arrays are not declared in the fragment shader.
    // For the last block (b == 3), declare the block with a "filler" entry.  This never gets used in the vertex shader.
    // A bindless sampler2D and image2D are added to ensure they don't interfere with non-UBO sampler/image bindings.
    for (int b = 0; b < 4; ++b) {
        int lwrrBlock = (b >= 2 ? ((b % 2) + numBlocks) : b);
        switch (usageType) {
        case USAGE_UBO:
            vs << "layout (std140) uniform Block" << lwrrBlock << " {int expected; vec4 arry[1]; int arry2[4]; "
                                                                  "  sampler2D fillerSamp; layout(r32f) image2D fillerImg;"
               << (b == 3 ? "int filler2;" : "") << " vec4 filler;} block" << lwrrBlock;
            break;
        case USAGE_SSBO:
            vs << "layout (std140) buffer Buffer" << lwrrBlock << " {int expected; vec4 arry[1]; int arry2[4];" << (b == 3 ? "int filler2;" : "") << " vec4 filler; } buffer" << lwrrBlock;
            break;
        case USAGE_TEXTURE:
            vs << "uniform sampler2D tex" << lwrrBlock;
            break;
        case USAGE_IMAGE:
            vs << "layout(r32f) uniform image2D img" << lwrrBlock;
            break;
        default:
            assert(!"Invalid usage type");
            break;
        };

        if (m_arySize > 1) {
            vs << "[" << m_arySize << "]";
        }

        vs << ";\n";
    }

    vs << "out interfaceBlock {\n"
        "  flat int vexpected;\n"
        "};\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0)";

    // Use the block1/buffer1 "arry" and "arry2" variable since block0/buffer0 are unused.  These variables should
    // be assigned the value of "0" and is only used in the vertex shader and not the fragment shader.
    if (usageType == USAGE_UBO) {
        vs << "+ block1" << (m_arySize > 1 ? "[0]" : "") << ".arry[0]";
        vs << "+ block1" << (m_arySize > 1 ? "[0]" : "") << ".arry2[2]";
    } else if (usageType == USAGE_SSBO) {
        vs << "+ buffer1" << (m_arySize > 1 ? "[0]" : "") << ".arry[0]";
        vs << "+ buffer1" << (m_arySize > 1 ? "[0]" : "") << ".arry2[2]";
    }

    vs << ";\n"
          "  if (0";

    // Check for the expected value in the vertex shader by taking a sum of the values in the buffers.  This should equal the bit mask that was assigned.
    // Resource 0 and resource 3 are unused in the vertex shader.
    for (int b = 1; b < 3; ++b) {
        int lwrrBlock = (b >= 2 ? (numBlocks + (b % 2)) : b);
        switch (usageType) {
        case USAGE_UBO:
            for (int zz = 0; zz < m_arySize; ++zz) {
                vs <<
                    " + block" << lwrrBlock;
                if (m_arySize > 1) {
                    vs << "[" << zz << "]";
                }
                vs << ".expected";
            }
            break;
        case USAGE_SSBO:
            for (int zz = 0; zz < m_arySize; ++zz) {
                vs <<
                    " + buffer" << lwrrBlock;
                if (m_arySize > 1) {
                    vs << "[" << zz << "]";
                }
                vs << ".expected";
            }
            break;
        case USAGE_TEXTURE:
            for (int z = 0; z < m_arySize; ++z) {
                vs << "+ ";
                vs << "ivec2(texture(tex" << lwrrBlock;
                if (m_arySize > 1) {
                    vs << "[" << z << "]";
                }
                vs << ", vec2(0.0)).x).x \n";
            }
            break;
        case USAGE_IMAGE:
            for (int z = 0; z < m_arySize; ++z) {
                vs << "+ ";
                vs << "ivec2(imageLoad(img" << lwrrBlock;
                if (m_arySize > 1) {
                    vs << "[" << z << "]";
                }
                vs << ", ivec2(0)).x).x \n";
            }
            break;
        default:
            assert(!"Invalid usage type");
            break;
        };
    }

    vs << "  == " << vsShaderExpectedVal << ") {\n"
        "    vexpected = expected;\n"
        "  } else {\n"
        "    vexpected = -666;\n"
        "  }\n"
    "}\n";

    FragmentShader fs(440);
    fs.addExtension(lwShaderExtension::LW_bindless_texture);
    fs << "in interfaceBlock {\n"
          "  flat int vexpected;\n"
          "};\n";

    for (int res = 0; res < numBlocks; ++res) {

        // Declare unused bindings to exercise the case where the total number of declared used and unused bindings
        // in a shader stage exceeds the maximum amount of bindings for the resource type.
        // Unused bindings effectively get ignored by the LWN/GLSLC compiler when bindings are assigned to resources
        // in each shader stage.  Only resources which are actually referenced in the shader or have a layout(binding =...)
        // qualifier will reserve actual binding slots and count against the maximum limit of bindings for that particular
        // resource.  There should be no error with declaring more than the maximum amount of a particular resource, as
        // long as less than the maximum limit are actually referenced and used in the shader.
        switch (usageType) {
        case USAGE_UBO:
            fs << "layout (std140) uniform UnusedBlock" << res << " { int unused; sampler2D unusedSamp; layout(r32f) image2D unusedImg;} unused" << res << ";\n";
            break;
        case USAGE_SSBO:
            fs << "layout (std140) buffer UnusedBuffer" << res << " { int unused; } unused" << res << ";\n";
            break;
        case USAGE_TEXTURE:
            fs << "uniform sampler2D unusedTex" << res << ";";
            break;
        case USAGE_IMAGE:
            fs << "layout(r32f) uniform image2D unusedImg" << res << ";";
            break;
        default:
            assert(!"Invalid usage type");
        }


        // Set explicit bindings as dictated by the input <explicitMask>.
        if (explicitMask & (1 << res*m_arySize)) {
            fs << "layout" << " (binding=" << res*m_arySize << ") ";
        }

        switch (usageType) {
        case USAGE_UBO:
            fs << "layout (std140) uniform Block" << res << " { int  expected; vec4 arry[1]; int arry2[4]; sampler2D fillerSamp; "
                                                            "   layout(r32f) image2D fillerImg; vec4 filler; } block" << res;
            break;
        case USAGE_SSBO:
            fs << "layout (std140) buffer Buffer" << res << " { int  expected; vec4 arry[1]; int arry2[4]; vec4 filler; } buffer" << res;
            break;
        case USAGE_TEXTURE:
            fs << "uniform sampler2D tex" << res;
            break;
        case USAGE_IMAGE:
            fs << "layout(r32f) uniform image2D img" << res;
            break;
        default:
            assert(!"Invalid usage type");
            break;
        };

        // Declare as arrays if we are performing an array test.
        if (m_arySize > 1) {
            fs << "[" << m_arySize << "]";
        }
        fs << ";\n";
    }

    fs <<  "out vec4 color;\n"
           "void main() {\n"
           "  int value = -1;\n"
           "  value = 0\n";

    // Sums up blocks as dictated by the <usedMask> using IsBlockUsed to query.  This ends up being a sum of the values
    // in the backed buffer which should (if the test passes) be equivalent to the expected bit mask sent in as a vertex
    // attribute.
    for (int res = 0, cntr = 0; res < numBlocks; res++, cntr++) {
        if (usageType == USAGE_UBO) {
            if (IsBlockUsed(res, usedMask, m_arySize)) {
                for (int z = 0; z < m_arySize; ++z) {
                    fs << " + block" << res;
                    if (m_arySize > 1) {
                        fs << "[" << z << "]";
                    }
                    fs << ".expected\n";
                }
            }
        } else if (usageType == USAGE_SSBO) {
            if (IsBlockUsed(res, usedMask, m_arySize)) {
                for (int z = 0; z < m_arySize; ++z) {
                    fs << " + buffer" << res;
                    if (m_arySize > 1) {
                        fs << "[" << z << "]";
                    }
                    fs << ".expected\n";
                }
            }
        } else if (usageType == USAGE_TEXTURE) {
            if (IsBlockUsed(res, usedMask, m_arySize)) {
                for (int z = 0; z < m_arySize; ++z) {
                    if (res == ARRY_BLOCK_WITH_NO_FIRST_ENTRY &&
                        m_arySize > 1 &&
                        z == 0) {
                            // Do not use first entry of this array.
                            continue;
                    }

                    fs << " + ivec2(texture(tex" << res;
                    if (m_arySize > 1){
                        fs << "[" << z << "]";
                    }
                    fs << ", vec2(0.0)).x).x";
                }
            }
        } else if (usageType == USAGE_IMAGE) {
            if (IsBlockUsed(res, usedMask, m_arySize)) {
                for (int z = 0; z < m_arySize; ++z) {
                    if (res == ARRY_BLOCK_WITH_NO_FIRST_ENTRY &&
                        m_arySize > 1 &&
                        z == 0) {
                            // Do not use first entry of this array.
                            continue;
                    }

                    fs << " + ivec2(imageLoad(img" << res;
                    if (m_arySize > 1){
                        fs << "[" << z << "]";
                    }
                    fs << ", ivec2(0)).x).x";
                }
            }
        }
    }

    // Draws different colors depending on failure.
    // cyan - Indicates failure from expected results in vertex shader.
    // red - Indicates failure from expected value.
    // green - Indicates test pass
    fs << ";\n"
          "  if (vexpected == -666) {\n"
          "    color = vec4(0.75, 0.0, 0.0, 1.0);\n"
          "  } else if (value == vexpected) {\n"
          "    color = vec4(0.0, 1.0, 0.0, 1.0);\n"
          "  } else {\n"
          "    color = vec4(0.0, 0.0, 0.0, 1.0);\n"
          "  }\n"
          "}\n";

    retVal = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    return retVal;
}

void LWNTESTGLSLUnusedBindingsTest::RunUsageTests(Usage usageType) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    Sampler *smp = sb.CreateSampler();

    struct Vertex {
        dt::vec2 position;
        dt::ivec1 expected;
    };

    // This test allocates one large pool of "dataPoolSize".  The pool is filled in such that the data for each resource
    // corresponds to the binding of that resource.  Each binding slot will correspond to a value
    // in the same location in the allocated backing buffer such that the value returned will
    // be equivalent to the bit mask representing the binding slot.
    // Textures and images are created from the same buffer so that queries will also pick the data corresponding to the
    // sampler/image binding location.  The VBO values are also stored in this pool.
    LWNsizeiptr dataPoolSize = 512 * 1024;
    LWNint resNum = 0;

    switch (usageType) {
    case (USAGE_IMAGE) :
        resNum = resNumImage;
        break;
    case (USAGE_TEXTURE) :
        resNum = resNumTexture;
        break;
    case (USAGE_SSBO) :
        resNum = resNumSSBO;
        break;
    case (USAGE_UBO) :
        resNum = resNumUBO;
        break;
    default:
        assert(false);
        printf("Error: Unknown usage!\n");
        break;
    };
    assert(resNum > 0);

    LWNint numBlocks = resNum / m_arySize;

    MemoryPoolAllocator dataPoolAllocator(device, NULL, dataPoolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    BufferAlignBits alignBits = BufferAlignBits(BUFFER_ALIGN_UNIFORM_BIT | BUFFER_ALIGN_IMAGE_BIT |
                                                BUFFER_ALIGN_COPY_READ_BIT);
    int sizeResBufferData = dataPoolSize / 4;
    Buffer *resBuffer = dataPoolAllocator.allocBuffer(&bb, alignBits, sizeResBufferData);
    BufferAddress resBufferAddr = resBuffer->GetAddress();
    int *resBufferMap = (int *) resBuffer->Map();
    memset(resBufferMap, 0, sizeResBufferData);
    Texture *tex[resNumTexture];

    // Set up a 1x1 texture for the texture queries.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetFlags(TextureFlags::IMAGE);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(1, 1);
    tb.SetFormat(Format::R32F);
    tb.SetLevels(1);
    LWNsizeiptr texSize = tb.GetPaddedStorageSize();

    LWNsizeiptr totalTexSize = resNum * texSize;
    MemoryPoolAllocator texAllocator(device, NULL, totalTexSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    for (int i = 0; i < resNum; i++) {
        // Fills up each entry with the bit map it corresponds to.
        if (usageType == USAGE_TEXTURE || usageType == USAGE_IMAGE) {
            ((float *)resBufferMap)[i * (resBufferSize / sizeof(float))] = float(1 << i);
        } else {
            resBufferMap[i * (resBufferSize / sizeof(LWNint))] = int(1 << i);
        }

        // Depending on resource type, bind the backing buffer for each to hold a bit mask indicating the slot in the corresponding
        // buffer.
        CopyRegion copyRegion = { 0, 0, 0, 1, 1, 1 };
        if (usageType == USAGE_UBO) {
            queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, i, resBufferAddr + i*resBufferSize, sizeof(LWNint));
            queueCB.BindUniformBuffer(ShaderStage::VERTEX, i, resBufferAddr + i*resBufferSize, sizeof(LWNint));
        } else if (usageType == USAGE_SSBO) {
            queueCB.BindStorageBuffer(ShaderStage::FRAGMENT, i, resBufferAddr + i*resBufferSize, sizeof(LWNint));
            queueCB.BindStorageBuffer(ShaderStage::VERTEX, i, resBufferAddr + i*resBufferSize, sizeof(LWNint));
        } else if (usageType == USAGE_TEXTURE) {
            tex[i] = texAllocator.allocTexture(&tb);
            TextureHandle texHandle = device->GetTextureHandle(tex[i]->GetRegisteredTextureID(), smp->GetRegisteredID());
            queueCB.CopyBufferToTexture(resBufferAddr + i * resBufferSize, tex[i], NULL, &copyRegion, CopyFlags::NONE);
            queueCB.BindTexture(ShaderStage::FRAGMENT, i, texHandle);
            queueCB.BindTexture(ShaderStage::VERTEX, i, texHandle);
        } else if (usageType == USAGE_IMAGE) {
            tex[i] = texAllocator.allocTexture(&tb);
            LWNuint id = g_lwnTexIDPool->RegisterImage(tex[i]);
            ImageHandle imageHandle = device->GetImageHandle(id);
            queueCB.CopyBufferToTexture(resBufferAddr + i * resBufferSize, tex[i], NULL, &copyRegion, CopyFlags::NONE);
            queueCB.BindImage(ShaderStage::FRAGMENT, i, imageHandle);
            queueCB.BindImage(ShaderStage::VERTEX, i, imageHandle);
        }
    }

    bb.SetDefaults();

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, expected);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = dataPoolAllocator.allocBuffer(&bb, BUFFER_ALIGN_VERTEX_BIT, dataPoolSize/4);
    BufferAddress vboAddr = vbo->GetAddress();
    Vertex *vboMap = (Vertex *)vbo->Map();


    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, dataPoolSize/4);

    static const int vpWidth = lwrrentWindowWidth / cellsX;
    static const int vpHeight = lwrrentWindowHeight / cellsY;
    int vnum = 0;
    // We draw 4 rows per test.
    int lwrrRow = (int)usageType * 4;
    Program *pgm = NULL;

    // Program resource type for each usage for reflection queries.
    static const LWNprogramResourceType resTypeArry[4] = {
        LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK,
        LWN_PROGRAM_RESOURCE_TYPE_STORAGE_BLOCK,
        LWN_PROGRAM_RESOURCE_TYPE_SAMPLER,
        LWN_PROGRAM_RESOURCE_TYPE_IMAGE
    };

    // First test type is for no layout specifiers.  Second test type is for interleaving layout binding specifiers on uniforms/arrays.
    for (int testType = 0; testType < 2; ++testType) {
        // usedMask indicates the blocks that are being bound. 
        int usedMask = 0;

        // expectedMask contains the expected results of bindings with the current iteration.
        int expectedMask = 0;

        // explicitMask indicates the binding slots that use the layout(binding) qualifier for unused and
        // used uniforms.
        // Every other 4 resources (4 uniforms or 2 arrays) are using explicit layout bindings.
        int explicitMask = testType == 1 ? 0x0F0F0F0F : 0;

        if (m_arySize > 1) {
            // The array that leaves the first entry out can't be any of the explicitly bound
            // arrays.
            assert(((((1 << m_arySize) - 1) << (ARRY_BLOCK_WITH_NO_FIRST_ENTRY * m_arySize)) & explicitMask) == 0);
        }

        // We iterate over each uniform or array of uniforms.
        for (int i = 0; i < numBlocks; ++i) {
            // Increase the number of used in the mask.
            // create the used mask in reverse order.
            usedMask |= mapBlockToResourceMask(numBlocks - i - 1, m_arySize);

            // Retrieve the expected results of binding this iteration's uniform/uniform array.
            expectedMask = GetExpectedMask(explicitMask, usedMask, resNum, numBlocks, usageType);

            LWNboolean loaded = LWN_FALSE;

            pgm = device->CreateProgram();
            loaded = CreateProgram(pgm, usedMask, explicitMask, numBlocks, usageType);


            if (loaded) {
                queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            }

            queueCB.SetViewportScissor(i * vpWidth + 2, lwrrRow * vpHeight + 2, vpWidth - 4, vpHeight - 4);
            if (loaded) {
                // Check the bindings match up.

                // We check all the bindings for all blocks, and create a bindings mask.  This mask should match
                // the expected mask exactly.  We perform program resource queries for this information for the first row,
                // then we perform actual rending to check within the shader for the next row.
                LWNprogramResourceType resType = resTypeArry[usageType];

                int obtainedBindings = 0;
                bool dupBinding = false;
                bool offsetError = false;
                bool sizeError = false;

                LWNboolean usesSpirv = g_glslcHelper->WasLastCompileSpirv();

                for (int b = 0; b < numBlocks; ++b) {
                    char nameBuf[256];
                    switch(usageType) {
                    case USAGE_UBO:
                        if (!usesSpirv) {
                            lwog_snprintf(nameBuf, 256, "Block%d%s", b, m_arySize > 1 ? "[0]" : "");
                        } else {
                            lwog_snprintf(nameBuf, 256, "Block%d.block%d%s", b, b, m_arySize > 1 ? "[0]" : "");
                        }
                        break;
                    case USAGE_SSBO:
                        if (!usesSpirv) {
                            lwog_snprintf(nameBuf, 256, "Buffer%d%s", b, m_arySize > 1 ? "[0]" : "");
                        } else {
                            lwog_snprintf(nameBuf, 256, "Buffer%d.buffer%d%s", b, b, m_arySize > 1 ? "[0]" : "");
                        }
                        break;
                    case USAGE_TEXTURE:
                        lwog_snprintf(nameBuf, 256, "tex%d%s", b, m_arySize > 1 ? "[0]" : "");
                        break;
                    case USAGE_IMAGE:
                        lwog_snprintf(nameBuf, 256, "img%d%s", b, m_arySize > 1 ? "[0]" : "");
                        break;
                    default:
                        assert(!"Invalid usage type");
                        break;
                    };

                    int loc = 0;
                    loc = g_glslcHelper->ProgramGetResourceLocation(pgm, ShaderStage::FRAGMENT, resType, nameBuf);

                    if (loc != -1) {
                        if (obtainedBindings & (1 << loc)) {
                            // Two buffers with the same binding, no good!
                            dupBinding = true;
                        }
                        for (int z = 0; z < m_arySize; ++z) {
                            if (m_arySize > 1 &&
                                b == ARRY_BLOCK_WITH_NO_FIRST_ENTRY &&
                                z == 0 &&
                                (usageType == USAGE_IMAGE || usageType == USAGE_TEXTURE)) {
                                // The unused array SHOULD still show up in reflection info as "<resource_type>[0]",
                                // so we exclude the first entry in our "obtainedBindings" mask.
                                continue;
                            }

                            // Since we only perform one query for arrays, set obtained bindings to
                            // reflect the whole array.
                            obtainedBindings |= (1 << (loc + z));
                        }
                    }

                    // Check to make sure the offsets and size for active UBOs/SSBOs are > -1 and >= 0 respectively.
                    // We only check SSBO offsets/size if we're using the GLSLC compiler, since the deprecated LWN online compiler
                    // doesn't support SSBO offset/size queries.  This check remains until the online compiler is removed.
                    if (loc != -1 &&
                        (usageType == USAGE_UBO || (usageType == USAGE_SSBO))) {
                        char expectedName [64] = { 0 };
                        char fillerName [64] = { 0 };
                        LWNboolean usesSpirv = g_glslcHelper->WasLastCompileSpirv();

                        if (usageType == USAGE_UBO) {
                            if (!usesSpirv) {
                                lwog_snprintf(expectedName, 64, "Block%d.expected", b);
                                lwog_snprintf(fillerName, 64, "Block%d.filler", b);
                            } else {
                                lwog_snprintf(expectedName, 64, "Block%d.block%d.expected", b, b);
                                lwog_snprintf(fillerName, 64, "Block%d.block%d.filler", b, b);
                            }
                        } else if (usageType == USAGE_SSBO) {
                            if (!usesSpirv) {
                                lwog_snprintf(expectedName, 64, "Buffer%d.expected", b);
                                lwog_snprintf(fillerName, 64, "Buffer%d.filler", b);
                            } else {
                                lwog_snprintf(expectedName, 64, "Buffer%d.buffer%d.expected", b, b);
                                lwog_snprintf(fillerName, 64, "Buffer%d.buffer%d.filler", b, b);
                            }
                        }

                        // Get the offset of the uniform within the block.
                        int expectedOff = g_glslcHelper->ProgramGetResourceLocation(pgm,
                                        ShaderStage::FRAGMENT,
                                        usageType == USAGE_UBO ? LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_OFFSET : LWN_PROGRAM_RESOURCE_TYPE_BUFFER_VARIABLE_OFFSET,
                                        expectedName);
                        int fillerOff = g_glslcHelper->ProgramGetResourceLocation(pgm,
                                        ShaderStage::FRAGMENT,
                                        usageType == USAGE_UBO ? LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_OFFSET : LWN_PROGRAM_RESOURCE_TYPE_BUFFER_VARIABLE_OFFSET,
                                        fillerName);

                        // Since this block is active, the offset should not be -1.
                        if (expectedOff < 0 || fillerOff <= 0) {
                            offsetError = true;
                        }

                        // Get the size of the block.
                        int bufferSize = g_glslcHelper->ProgramGetResourceLocation(pgm,
                                ShaderStage::FRAGMENT,
                                usageType == USAGE_UBO ? LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK_SIZE : LWN_PROGRAM_RESOURCE_TYPE_STORAGE_BLOCK_SIZE,
                                nameBuf);

                        // filler is a vec4, 16 bytes.
                        int sizeOfFiller = 16;

                        // Using std140, the size of the buffer is specified according to ARB_uniform_buffer_object
                        // by taking the offset of the last basic machine unit of the block, adding 1 (i.e. ((fillerOff + sizeOfFiller - 1) + 1)),
                        // and rounding to the next multiple of base alignment required for a
                        // vec4 (16 bytes).
                        int expectedBufferSize = (fillerOff + sizeOfFiller + 15) & ~0xF;

                        // Since this block is active, the size should not be <= 0.
                        if (bufferSize != expectedBufferSize) {
                            sizeError = true;
                        }
                    }
                }

                if (dupBinding) {
                    queueCB.ClearColor(0, 0.5, 0.0, 0.0, 1.0);
                } else if (offsetError) {
                    queueCB.ClearColor(0, 0.6, 0.0, 0.0, 1.0);
                } else if (sizeError) {
                    queueCB.ClearColor(0, 0.7, 0.0, 0.0, 1.0);
                } else if (obtainedBindings != expectedMask) {
                    queueCB.ClearColor(0, 0.8, 0.0, 0.0, 1.0);
                } else {
                    // Pass
                    queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
                }
            }
            else {
                queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);
            }

            // Next row, test results of test through rendering.
            queueCB.SetViewportScissor(i * vpWidth + 2, (lwrrRow + 1) * vpHeight + 2, vpWidth - 4, vpHeight - 4);
            if (loaded) {
                for (int v = 0; v < 4; v++) {
                    vboMap[vnum + v].position[0] = (v & 2) ? +1.0 : -1.0;
                    vboMap[vnum + v].position[1] = (v & 1) ? +1.0 : -1.0;
                    vboMap[vnum + v].expected = expectedMask;
                }
                queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, vnum, 4);
                vnum += 4;
            } else {
                queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);
            }
        }

        // We draw two rows per test type.
        lwrrRow += 2;
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

void LWNTESTGLSLUnusedBindingsTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    // Run each of the usage cases.
    for (int i = 0; i < LWNTESTGLSLUnusedBindingsTest::USAGE_TYPE_COUNT; ++i) {
        RunUsageTests((Usage)i);
    }
}

OGTEST_CppTest(LWNTESTGLSLUnusedBindingsTest, lwn_glsl_bindings_unused_single,
               (LWNTESTGLSLUnusedBindingsTest::USAGE_UBO, LWNTESTGLSLUnusedBindingsTest::TEST_TYPE_SINGLE));
OGTEST_CppTest(LWNTESTGLSLUnusedBindingsTest, lwn_glsl_bindings_unused_array,
               (LWNTESTGLSLUnusedBindingsTest::USAGE_UBO, LWNTESTGLSLUnusedBindingsTest::TEST_TYPE_ARRAY));
