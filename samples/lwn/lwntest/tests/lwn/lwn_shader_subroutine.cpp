/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
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

#define LWN_DEBUG_LOG     0
static void log_output(const char *fmt, ...)
{
#if LWN_DEBUG_LOG
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
#endif
}

// We use 4 subroutine uniforms in the fragment or compute shaders that we test.
// Each subroutine uniform is a different type.
#define NUM_ACTIVE_SUBROUTINE_UNIFORMS 4

// Array size to use for declaring subroutine uniform arrays in array tests.
#define arySize 5

// Array index to use for indexing into arrays when testing subroutine uniform arrays.
// This is kind of arbitrary, as long as it's less than arySize.
#define aryNdx 3

// Number of possible subroutine types
// Our types for our subroutines are:
// Type 0 -- "(transformColorSub0_T)"
// Type 1 -- "(transformColorSub1_T)"
// Type 2 -- "(transformColorSub1_T, transformColorSub0_T)"
// Type 3 -- "(transformColorSub2_T, transformColorSub3_t)"
#define NUM_TYPES 4

// Number of unique subroutine function bodies.  Each subroutine is duplicated 4 times, once for each
// possible subroutine type.
#define NUM_UNIQUE_SUBROUTINE_BODIES 4

// Enumerators for our subroutines in the test configurations for each cell.  The names of
// this enumeration have the following scheme: S<x><y>, where <x> denotes what the subroutine
// does (i.e. "R" means it sets the red component, "G" means it sets the green component, etc) and
// <y> indicates which type it is (i.e. y == "0" means the subroutine can be bound to uniforms
// of type transformColorSub0_T. y == "23" means the subroutine can be bound to uniforms of type
// transformColorSub2_T or type transformColorSub3_T (or both!).
enum {
    // Type 0 -- "(transformColorSub0_T)"
    SR0 = 0,
    SG0,
    SB0,
    SNO0,

    // Type 1 -- "(transformColorSub1_T)"
    SR1,
    SG1,
    SB1,
    SNO1,

    // Type 2 -- "(TransformColorSub1_T, transformColorSub0_T)"
    SR01,
    SG01,
    SB01,
    SNO01,

    // Type 3 -- "(transformColorSub2_T, transformColorSub3_t)"
    SR23,
    SG23,
    SB23,
    SNO23,

    // Total number of subroutines.
    NUM_SUBROUTINES
};

// Number of tests per row, which equates to the number of columns.
#define NUM_TESTS_PER_ROW 23

// A list of tests and which subroutines should be bound to which uniforms in that test.
// Each array in testconfigsNdxs corresponds to a single cell.
// Each entry in testConfigsNdxs[n] corresponds to a uniform, where the entry corresponds
// to transformColorUni<x>, where x is either 0, 1, 2, 3 and corresponds to the subroutine in
// testConfigsNdxs[n][x].
const static int testConfigNdxs[NUM_TESTS_PER_ROW][NUM_ACTIVE_SUBROUTINE_UNIFORMS] =
{
    // These are chosen semi-arbitrarily, but are meant to be enough combinations to
    // trigger range updates (when doing subroutine range testing with the SetProgramSubroutines
    // API), and also to mix up subroutines that can be bound to multiple types.

    // Column 0 can be bound to Sx0, or Sx01
    // Column 1 can be bound to Sx1, or Sx01
    // Column 2 can be bound to Sx23
    // Column 3 can be bound to Sx23
    { SR0,  SG1, SB23, SNO23 },

    { SNO0, SR1, SG23, SNO23 },
    { SNO0, SR1, SB23, SNO23 },
    { SNO0, SG1, SR23, SNO23 },
    { SNO0, SG1, SB23, SNO23 },
    { SNO0, SB1, SR23, SNO23 },
    { SNO0, SB1, SG23, SNO23 },

    { SR0, SNO1, SG23, SNO23 },
    { SR0, SNO1, SB23, SNO23 },
    { SG0, SNO1, SR23, SNO23 },
    { SG0, SNO1, SB23, SNO23 },
    { SB0, SNO1, SR23, SNO23 },
    { SB0, SNO1, SG23, SNO23 },

    { SR0, SG1, SNO23, SNO23 },
    { SR0, SB1, SNO23, SNO23 },
    { SG0, SR1, SNO23, SNO23 },
    { SG0, SB1, SNO23, SNO23 },
    { SB0, SR1, SNO23, SNO23 },
    { SB0, SG1, SNO23, SNO23 },

    { SR01,SG01,SNO23, SNO23 },
    { SR01,SB1, SNO23, SNO23 },
    { SG0, SR01,SNO23, SNO23 },
    { SG0, SB1, SNO23, SNO23 },
};

using namespace lwn;

class LWNsubroutineUniformTestBasic
{
    // Number of cells in the x & y direction.
    static const int cellsX = 32;
    static const int cellsY = 40;

    // Compute parameters
    static const int ctaWidth = 8;
    static const int ctaHeight = 8;

    // A structure indicating which parameters we are testing.  Each combination of parameters in the structure
    // equates to a separate row.
    struct TestConfig {
        // Wheter to run through a compute shader test, or run through non-compute graphics tests.
        bool m_doCompute;

        // Whether to test using explicit layout index qualifiers on subroutine definitions or not.  If using
        // explicit layout qualifiers, additional checks will be performed to ensure our reflection info
        // matches up to the expected subroutine index values..
        bool m_doExplicitLayout;

        // Whether to test arrays of subroutine uniforms (true) or scalar uniforms (false).
        bool m_doArray;

        // Whether to test dynamic indexing into the arrays (only valid if m_doArray == true);  Dynamic indexing
        // is performed by indexing a subroutine array with an index value that's obtained as an input varying.
        // If a test config is used where m_doArray == false and m_doDynamicIndex == true, the test will be skipped.
        bool m_doDynamicIndex;

        // Whether to use the "range" API or not.  The lwnCommandBufferSetSubroutines function allows partial updates
        // to select subroutine uniforms without having to set all subroutine uniforms at once (which OpenGL
        // requires).  When this is set to true, each test first batches up which subroutines have changed and
        // submits a sequence of lwnCommandBufferSetSubroutines calls for each batch.  If false, all subroutine uniform locations
        // are updated at once.
        bool m_doRange;
    };

    // GLSL program parameters containing compiled programs and common variables used to parse reflection info for
    // those programs.
    struct ProgramParams {
        // Graphics program to test subroutines.
        Program *m_graphicsPgm;
        // Compute program to test subroutines
        Program *m_computePgm;
        // Vs/Fs program to draw the results of the compute program.
        Program *m_computeDrawPgm;

        // Subroutines used in the vertex stage.  Vertex parameters stay the same with all tests.
        //

        // The name of the subroutine we do want to bind to.
        const char *vsGoodSubroutineName;

        // The name of the subroutine uniform in the vertex shader
        const char *vsSubroutineUniformName;

        //
        // Parameters for our main testing stages: fragment and compute

        // The base name of our subroutines.  Each subroutine is declared in 4 different types, using the same base name and the
        // same body.  A number indicating the subroutine type is appended to the end of the string to make the name
        // unique in the shader.
        const char *subroutineNamesBase[4];

        // Names of the uniforms.  The first 4 entries are the non-array versions, the last 4 entries are the
        // array versions.
        const char *uniformNames[8];

        // Location for our subroutine uniforms, used in explicit layout tests.
        int locationNdx[4];

        // Layout index for our subroutines, within a sepcific type.  The global index for the subroutine will depend
        // on which type it is, but we only use 4 unique subroutine bodies.  For instance, for "transformColorAddR2",
        // since layoutNdx[0] corresponds to our "red" subroutines, then the index assigned to "transformColorAddR2"
        // would be layoutNdx[0] + 4*2 (where 4 is the number of types, and 2 is the type index).
        int layoutNdx[4];

        // Information from our compiled subroutines from the GLSLC compile.
        const GLSLCsubroutineInfo * subroutines[NUM_TYPES * NUM_UNIQUE_SUBROUTINE_BODIES];

        // Information about our uniforms from GLSLC compile.
        const GLSLCsubroutineUniformInfo * activeSubroutineUniformList[NUM_ACTIVE_SUBROUTINE_UNIFORMS];

        // The max subroutine uniform location.  Even for non-arrays, subroutines might be spaced apart.  The spec states
        // that locations between subroutine uniforms are unused, but still must be set (unless updating a "range" of values),
        // so this variable is used to determine the lenght of the input array to SetProgramSubroutines.
        int maxUniformsLoc;

        // Subroutine linkage for each program stage.
        // Note: Right now we are only utilzing vertex, fragment, and compute stages.
        LWNsubroutineLinkageMapPtr subroutineLinkageMaps[GLSLC_NUM_SHADER_STAGES];

        // Compile shaders, retrieve subroutine info, etc.
        LWNboolean Init(TestConfig testConfig);

        ProgramParams() {
            memset(subroutineLinkageMaps, 0, sizeof(LWNsubroutineLinkageMapPtr) * GLSLC_NUM_SHADER_STAGES);

            // Vertex shader subroutines are used just to verify subroutines can be used in multiple
            // stages at the same time, but parameters for vertex subroutines don't change from test to test.
            vsGoodSubroutineName = "GetIndexGood";
            vsSubroutineUniformName = "getIndexUniform";
 
            // Fragment shader subroutine base names.  When writing to the GLSL shader string, a number indicating
            // which type it corresponds to will be appended to each instance.  Each subroutine base name corresponds
            // to a unique subroutine body.
            subroutineNamesBase[0] = "transformColorAddR";
            subroutineNamesBase[1] = "transformColorAddG";
            subroutineNamesBase[2] = "transformColorAddB";
            subroutineNamesBase[3] = "transformColorNoop";

            uniformNames[0] = "transformColorUni0";
            uniformNames[1] = "transformColorUni1";
            uniformNames[2] = "transformColorUni2";
            uniformNames[3] = "transformColorUni3";
            uniformNames[4] = "transformColorUni0[0]";
            uniformNames[5] = "transformColorUni1[0]";
            uniformNames[6] = "transformColorUni2[0]";
            uniformNames[7] = "transformColorUni3[0]";

            // We'll retrieve this once the program is compiled.
            maxUniformsLoc = -1;

            // Set the location index for our expected subroutines, where the base of each subroutine
            // array will be immediately after the last location of the previous subroutine array.
            for (int i = 0; i < 4; ++i){
                locationNdx[i] = i == 0 ? 0 : locationNdx[i-1] + arySize;
                // For now just assign each subroutine uniform sequential layouts, if using explicit layouts.
                layoutNdx[i] = i;
            }
        }

        // Cleanup.  Need to free the allocated linkage maps.
        ~ProgramParams() {
            for (int i = 0; i < GLSLC_NUM_SHADER_STAGES; ++i) {
                // Free our allocated linkage maps.
                __LWOG_FREE(subroutineLinkageMaps[i]);
                subroutineLinkageMaps[i] = NULL;
            }
        }
    };

public:

    LWNsubroutineUniformTestBasic()
    {
        // Some asserts on our global defines.  Certain defines have expected
        // values for now, maybe some extensions might remove this restriction
        // in the future.
        assert((aryNdx >= 0) && (aryNdx < arySize));
        assert(NUM_TYPES == 4);
        assert(NUM_UNIQUE_SUBROUTINE_BODIES == 4);
    }

    // Run a single test from the input test config, drawing results to row <lwrrRow>
    void RunTest(TestConfig config, const ProgramParams *programParams, int lwrrRow) const;

    // Helper function to draw using compute.  We dispatch the compute stage to write to an SSBO.
    // We then bind a vertex and fragment program to draw from the SSBO.
    void DrawCompute(QueueCommandBuffer &queueCB,
        const ProgramParams *programParams,
        BufferAddress ssboAddr, int ssboSize,
        BufferAddress uboAddr, int uboSize,
        int ssboWidth, int ssboHeight,
        int ctaWidth, int ctaHeight) const;

    // Helper function to determine if <location> is an expected active subroutine uniform location.  For non-array
    // tests, arrayNdx should be 0.
    bool IsActiveUniformLocation(int location, const ProgramParams *programParams,
                                 int numActiveSubroutines, int arrayNdx = 0) const;

    LWNTEST_CppMethods();
};

lwString LWNsubroutineUniformTestBasic::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic testing for shader subroutine support in LWN and GLSLC.\n"
        "This test exercises various features of shader subroutine support for graphics programs and compute programs,"
        "as well as tests the reflection query support from GLSLC.\n"
        "The test is designed to hit these major points:\n"
        " * Graphics and compute shaders with subroutines.\n"
        " * Shaders with explicit layouts, and testing reflection information matches.\n"
        " * Shaders which use arrays of subroutine uniforms, as well as scalars.\n"
        " * Shaders which dynamically index arrays of subroutine uniforms, as well as static indexing.\n"
        " * The \"range\" API of SetProgramSubroutines which doesn't require updating all subroutine uniforms\n"
        "   at the same time.\n\n"
        " * Shaders which use subroutines of multiple types, and testing they can be bound to subroutine uniforms of either type\n"
        " * Test the subroutine uniform reflection queries return the right index values for compatible subroutines for that uniform.\n"
        "For the shaders being tested, 4 subroutine uniforms of different types are declared and used.  Each subroutine sets a different "
        "component of the input color vector to a value (we use the value 1 exclusively) and returns.  There are 16 possible subroutines that the\n"
        "subroutine uniforms can be bound to, with 4 unique function bodies (4 for each type): 3 for setting each color component, and 1\n"
        "as a NOOP.  By binding subroutine uniforms to different subroutines, we can draw different colors.  These colors are used to "
        "verify the expected subroutines are being used when drawing.  Additionally, we configure 2 of the subroutines to be multi-type and "
        "bindable to uniforms of either type.\n\n"
        "Each row represents a different test configuration, and each column represents a different configuration of subroutine\n"
        "bindings.  Each even row is graphics shaders, each odd row is compute shaders.  Each odd group of 2 rows test explicit layouts,"
        "where explicit index/layout qualifiers are used in the shader, and reflection info is queried from GLSLC to make sure the "
        "returned values are what is expected.  Each odd group of 4 rows tests using subroutine uniform arrays, where each array is "
        "only indexed once.  Even groups of 4 don't use arrays for uniforms.\n"
        "Each odd group of 8 rows, if using arrays, tests dynamically indexing into the array instead of a static index.  The could result\n"
        "in more call sites generated by the compiler in the linkage map.\n"
        "The first 16 rows upload the entire ranges of subroutine uniforms in each call of SetProgramSubroutines, and the second 16 rows\n"
        "use a range API where the uniforms are sent in batches with multiple calls to SetProgramSubroutines\n"
        "The test is a failure if a red cell (1, 0, 0, 1) is drawn, or cells with borders (and thus not a single color) are drawn.\n"
        "Each test computes the expected color value given the test configuration, draws this as the border around the cell that's\n"
        "drawn by the shader.  If there is a mismatch between expected and what's drawn, this can be inspected by looking at the border color (expected)\n"
        "and comparing with the center of the cell (what's actually drawn).";

    return sb.str();
}

int LWNsubroutineUniformTestBasic::isSupported() const
{
    return lwogCheckLWNAPIVersion(41, 1);
}

bool LWNsubroutineUniformTestBasic::IsActiveUniformLocation(int location, const ProgramParams *programParams,
                                    int numActiveSubroutines, int arrayNdx /*= 0*/) const
{
    for (int i = 0; i < numActiveSubroutines; ++i) {
        if (programParams->activeSubroutineUniformList[i]->location + arrayNdx == location) {
            return true;
        }
    }

    return false;
}

// Perform a draw operation using a compute shader to generate the image to draw.
void LWNsubroutineUniformTestBasic::DrawCompute(QueueCommandBuffer &queueCB,
                                                const ProgramParams *programParams,
                                                BufferAddress ssboAddr, int ssboSize,
                                                BufferAddress uboAddr, int uboSize,
                                                int ssboWidth, int ssboHeight,
                                                int ctaWidth, int ctaHeight) const
{
    // Dispatch a compute shader to draw into individual slots of our UBO.
    queueCB.BindProgram(programParams->m_computePgm, ShaderStageBits::COMPUTE);
    queueCB.BindStorageBuffer(ShaderStage::COMPUTE, 0, ssboAddr, ssboSize);
    queueCB.BindUniformBuffer(ShaderStage::COMPUTE, 0, uboAddr + 0 * uboSize, uboSize);
    queueCB.DispatchCompute(ssboWidth / ctaWidth, ssboHeight / ctaHeight, 1);

    // Draw with our passthrough drawing.
    queueCB.BindProgram(programParams->m_computeDrawPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindStorageBuffer(ShaderStage::FRAGMENT, 0, ssboAddr, ssboSize);
    queueCB.BindStorageBuffer(ShaderStage::VERTEX, 0, ssboAddr, ssboSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
}

// Compiles either graphics or compute programs and stashes away the linkage maps and reflection info.
LWNboolean LWNsubroutineUniformTestBasic::ProgramParams::Init(TestConfig testConfig)
{
    Device *device = DeviceState::GetActive()->getDevice();

    // Whether we can set our shaders or not.
    LWNboolean loaded = LWN_TRUE;

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec2 position;\n"
        "layout(location=1) in vec4 icolor;\n"
        "layout(std140, binding = 0) uniform Block {\n"
        "  ivec4 viewport;\n"
        "  int testNdx;\n"
        "};\n"
        "out Block {\n"
        "  vec4 ocolor;\n"
        "  flat int testNdx;\n"
        "} v;\n"
        ""

        "subroutine int GetIndex(int);\n"
        "layout (index = 0) subroutine (GetIndex) int ILWALID_SUBROUTINE(int inNdx) { return inNdx * 9000; }\n"
        "layout (index = 1) subroutine (GetIndex) int " << vsGoodSubroutineName << "(int inNdx) { return inNdx * 1; }\n"
        "layout(location = 0) subroutine uniform GetIndex getIndexUniform;\n"
        ""
        "void main() {\n"
        "  gl_Position = vec4(position, 0.0, 1.0);\n"
        "  v.ocolor = icolor;\n"
        "  int selectorNdx = getIndexUniform(testNdx);\n"
        "  v.testNdx = selectorNdx;\n"
        "}\n";

    lwStringBuf subPrefix;

    // Used when specifying subroutine types.
    const char * subTypeStrArray[4] = {
        "transformColorSub0_T",
        "transformColorSub1_T",
        "transformColorSub2_T",
        "transformColorSub3_T",
    };

    // Used when specifying subroutines in the "subroutine( ... )" segment.
    const char * subImplTypeStrArray[4] = {
        "transformColorSub0_T",
        "transformColorSub1_T",
        "transformColorSub1_T, transformColorSub0_T",
        "transformColorSub2_T, transformColorSub3_T",
    };

    // Write our subroutine info to the shader string.
    for (int i = 0; i < 4; ++i) {
        subPrefix << " subroutine vec4 " << subTypeStrArray[i] << " (vec4 col, float amt);\n";
    }

    const char *subroutineBody[4] = {
        " (vec4 col, float amt) { col.x = amt;  return col; }\n",
        " (vec4 col, float amt) { col.y = amt;  return col; }\n",
        " (vec4 col, float amt) { col.z = amt;  return col; }\n",
        " (vec4 col, float amt) { return col; }\n",
    };

    // Subroutines
    for (int t = 0; t < 4; ++t) {
        for (int n = 0; n < 4; ++n) {
            if (testConfig.m_doExplicitLayout) {
                subPrefix << "layout(index = " << layoutNdx[n] + t * 4 << ") ";
            }

            subPrefix <<
                "subroutine (" << subImplTypeStrArray[t] << ")\n"
                "vec4 " << subroutineNamesBase[n] << t << subroutineBody[n];
        }
    }

    for (int t = 0; t < 4; ++t) {
        subPrefix << " layout(location = " << locationNdx[t] << ") subroutine uniform "
                  << subTypeStrArray[t] << " transformColorUni" << t;
        if (testConfig.m_doArray) {
            subPrefix << "[" << arySize << "]";
        }
        subPrefix << ";\n";
    }
    subPrefix << "\n";

    // Compile graphics program.
    Program *subroutineProgram;
    if (!testConfig.m_doCompute) {
        m_graphicsPgm = device->CreateProgram();

        FragmentShader fs(440);
        fs <<
            subPrefix.str().c_str() <<
            // Subroutine types
            "in Block {\n"
            "  vec4 ocolor;\n"
            "  flat int testNdx;\n"
            "} v;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "vec4 computedColor = vec4(0, 0, 0, 1);\n";


        for (int i = 0; i < 4; ++i) {
            fs << "computedColor = transformColorUni" << i;
            if (testConfig.m_doArray) {
                fs << "[";
                if (testConfig.m_doDynamicIndex) {
                    fs << "v.testNdx";
                } else {
                    fs << aryNdx;
                }
                fs << "]";
            }
            fs << "(computedColor, 1.0f); \n";
        }

        fs << "  fcolor = computedColor;\n";
        fs << "}\n";

        if (!g_glslcHelper->CompileAndSetShaders(m_graphicsPgm, vs, fs)) {
            log_output("Can't compile/set graphics program.\n");
            return false;
        }

        int fsLinkageMapSize = 0;
        int vsLinkageMapSize = 0;
        LWNsubroutineLinkageMapPtr fsLinkageMap = g_glslcHelper->GetSubroutineLinkageMap(LWN_SHADER_STAGE_FRAGMENT, 0, &fsLinkageMapSize);
        if (fsLinkageMapSize == 0) {
            log_output("size of linkage data == 0!");
            return false;
        }
        assert(fsLinkageMap);

        LWNsubroutineLinkageMapPtr vsLinkageMap = g_glslcHelper->GetSubroutineLinkageMap(LWN_SHADER_STAGE_VERTEX, 0, &vsLinkageMapSize);
        if ( vsLinkageMapSize == 0) {
            log_output("size of linkage data == 0!");
            return false;
        }
        assert(vsLinkageMap);

        subroutineLinkageMaps[LWN_SHADER_STAGE_VERTEX] = __LWOG_MALLOC(vsLinkageMapSize);
        memcpy(subroutineLinkageMaps[LWN_SHADER_STAGE_VERTEX], vsLinkageMap, vsLinkageMapSize);

        subroutineLinkageMaps[LWN_SHADER_STAGE_FRAGMENT] = __LWOG_MALLOC(fsLinkageMapSize);
        memcpy(subroutineLinkageMaps[LWN_SHADER_STAGE_FRAGMENT], fsLinkageMap, fsLinkageMapSize);

        subroutineProgram = m_graphicsPgm;
    }
    else {

        // Compile compute program
        ComputeShader cs(450);
        cs.setCSGroupSize(ctaWidth, ctaHeight);
        cs <<
            subPrefix.str().c_str() <<
            "layout(std430, binding = 0) buffer SSBO {\n"
            "  vec4 values[];\n"
            "};\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  int testNdx;\n"
            "};\n"
            "void main() {\n"
            "  ivec2 xy = ivec2(gl_GlobalIlwocationID.xy);\n"
            "  int index = xy.y * viewport.z /* width */ + xy.x;\n"
            "  vec4 computedColor = vec4(0, 0, 0, 1);\n";

        for (int i = 0; i < 4; ++i) {
            cs << "computedColor = transformColorUni" << i;
            if (testConfig.m_doArray) {
                cs << "[";
                if (testConfig.m_doDynamicIndex) {
                    cs << "testNdx";
                } else {
                    cs << aryNdx;
                }
                cs << "]";
            }
            cs << "(computedColor, 1.0f); \n";
        }

        cs << "  values[index] = computedColor;\n";
        cs << "}\n";

        // Common vertex shader for drawing a big pass-through quad.  We also
        // construct a set of texture coordinates (otc) using position.
        VertexShader vsCompute(450);
        vsCompute <<
            "layout(location=0) in vec2 position;\n"
            "layout(location=1) in vec4 color;\n"
            "out vec3 ocolor;\n"
            "out vec2 otc;\n"
            "void main() {\n"
            "  gl_Position = vec4(position, 0.0, 1.0);\n"
            "  ocolor = color.xyz;\n"
            "  otc = 0.5 * position.xy - 0.5;\n"
            " otc = otc + vec2(1,1);\n"
            "}\n";

        // Shader used to display a texture initialized from SSBO contents.
        FragmentShader fsCompute(450);
        fsCompute <<
            "layout(std430, binding = 0) buffer SSBO {\n"
            "  vec4 values[];\n"
            "};\n"
            "in vec2 otc;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  fcolor = values[(int(otc.y)* " << lwrrentWindowHeight << ")*" << lwrrentWindowWidth << " + int(otc.x)* " << lwrrentWindowWidth << "];\n"
            "}\n";

        m_computeDrawPgm = device->CreateProgram();
        if (!g_glslcHelper->CompileAndSetShaders(m_computeDrawPgm, vsCompute, fsCompute)) {
            log_output("Can't copmile compute program:\n%s\n", g_glslcHelper->GetInfoLog());
            loaded &= LWN_FALSE;
        }
        m_computePgm = device->CreateProgram();
        if (!g_glslcHelper->CompileAndSetShaders(m_computePgm, cs)) {
            log_output("Can't compile compute drawing programs:\n%s\n", g_glslcHelper->GetInfoLog());
            loaded &= LWN_FALSE;

        }
        int csLinkageMapSize = 0;


        LWNsubroutineLinkageMapPtr csLinkageMap = g_glslcHelper->GetSubroutineLinkageMap(LWN_SHADER_STAGE_COMPUTE, 0, &csLinkageMapSize);

        if (csLinkageMapSize == 0) {
            log_output("size of linkage data == 0!");
            return false;
        }
        assert(csLinkageMap);

        subroutineLinkageMaps[LWN_SHADER_STAGE_COMPUTE] = __LWOG_MALLOC(csLinkageMapSize);
        memcpy(subroutineLinkageMaps[LWN_SHADER_STAGE_COMPUTE], csLinkageMap, csLinkageMapSize);

        subroutineProgram = m_computePgm;
    }

    //
    // Set the subroutine linkage maps.
    LWNsubroutineLinkageMapPtr linkageMaps[GLSLC_NUM_SHADER_STAGES] = { NULL };

    int numMaps = 0;
    if (testConfig.m_doCompute) {
        linkageMaps[numMaps++] = subroutineLinkageMaps[LWN_SHADER_STAGE_COMPUTE];
    } else {
        linkageMaps[numMaps++] = subroutineLinkageMaps[LWN_SHADER_STAGE_VERTEX];
        linkageMaps[numMaps++] = subroutineLinkageMaps[LWN_SHADER_STAGE_FRAGMENT];
    }

    assert(subroutineProgram);
    subroutineProgram->SetSubroutineLinkage(numMaps, linkageMaps);

    // Get the reflection info
    //

    const LWNshaderStage stage = testConfig.m_doCompute ? LWN_SHADER_STAGE_COMPUTE : LWN_SHADER_STAGE_FRAGMENT;

    for (int i = 0; i < NUM_TYPES; ++i) {
        for (int j = 0; j < NUM_UNIQUE_SUBROUTINE_BODIES; ++j) {
            lwStringBuf subroutineName;
            subroutineName << subroutineNamesBase[j] << i;
            subroutines[i*NUM_UNIQUE_SUBROUTINE_BODIES + j] =
                g_glslcHelper->GetSubroutineInfo(stage, subroutineName.str().c_str());
            if (!subroutines[i*NUM_UNIQUE_SUBROUTINE_BODIES + j]) {
                log_output("Can not get location for subroutine %s.\n", subroutineName.str().c_str());
                return false;
            }
        }
    }

    for (int uniIndex = 0; uniIndex < 4; ++uniIndex) {
        activeSubroutineUniformList[uniIndex] =
            g_glslcHelper->GetSubroutineUniformInfo(stage, uniformNames[uniIndex + (testConfig.m_doArray ? 4 : 0)]);

        if (!activeSubroutineUniformList[uniIndex]){
            log_output("Can not get location for subroutine uniform %s.\n", uniformNames[uniIndex + (testConfig.m_doArray ? 4 : 0)]);
            return false;
        }

        int lastLocation = activeSubroutineUniformList[uniIndex]->location + arySize - 1;
        maxUniformsLoc = maxUniformsLoc > lastLocation ? maxUniformsLoc : lastLocation;
    }

    if (maxUniformsLoc < 0) {
        log_output("Could not get maximum uniform location from reflection info.  Location retrieved: %d\n", maxUniformsLoc);
        return false;
    }

    return true;
}

void LWNsubroutineUniformTestBasic::RunTest(TestConfig testConfig, const ProgramParams * programParams, int lwrrRow) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    Program *subroutineProgram = NULL;
    LWNsizeiptr vboSize = 640 * 1024;

    struct Vertex {
        dt::vec2 position;
        dt::vec4 color;
    };

    static const Vertex vertexData[] = {
        { dt::vec2(-1.0, -1.0), dt::vec4(0.0, 0.0, 0.0, 1.0) },
        { dt::vec2(-1.0, +1.0), dt::vec4(0.0, 0.0, 0.0, 1.0) },
        { dt::vec2(+1.0, -1.0), dt::vec4(0.0, 0.0, 0.0, 1.0) },
        { dt::vec2(+1.0, +1.0), dt::vec4(0.0, 0.0, 0.0, 1.0) },
    };

    struct UBOLayout {
        dt::ivec4       viewport;
        int             testNdx;
        int             filler[3];
    };

    // For graphics programs, we have an additional subroutine in our vertex shader.
    if (testConfig.m_doCompute == false) {

        subroutineProgram = programParams->m_graphicsPgm;

        const GLSLCsubroutineInfo * vsSubInfo =
            g_glslcHelper->GetSubroutineInfo(LWN_SHADER_STAGE_VERTEX, programParams->vsGoodSubroutineName);
        const GLSLCsubroutineUniformInfo * vsSubUniformInfo =
            g_glslcHelper->GetSubroutineUniformInfo(LWN_SHADER_STAGE_VERTEX, programParams->vsSubroutineUniformName);

        if (!vsSubInfo) {
            log_output("Can't get subroutine info for the vertex subroutine %s\n", programParams->vsGoodSubroutineName);
        }

        if (!vsSubUniformInfo) {
            log_output("Can't get subroutine uniform info for the vertex subroutine uniform named %s\n", programParams->vsSubroutineUniformName);
        }

        int vertexSubUniform = vsSubInfo->index;



        queueCB.SetProgramSubroutines(subroutineProgram, ShaderStage::VERTEX, 0, 1, &vertexSubUniform);
        queueCB.BindProgram(subroutineProgram, ShaderStageBits::ALL_GRAPHICS_BITS);

    } else {
        subroutineProgram = programParams->m_computePgm;
        // We'll bind this later when drawing.
    }

    // Set up VBO, UBO, and SSBO.
    // Boiler plate code, mostly ripped off from other tests.
    MemoryPoolAllocator dataPoolAllocator(device, NULL, vboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    MemoryPoolAllocator allocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    int ssboWidth = ctaWidth * (lwrrentWindowWidth / (ctaWidth));
    int ssboHeight = ctaHeight * (lwrrentWindowHeight / (ctaHeight));
    int ssboSize = (ssboWidth * ssboHeight * 2 + 1) * sizeof(dt::vec4);
    ssboSize = (ssboSize + 0x1F) & ~(0x1F);
    MemoryPoolAllocator ssboAllocator(device, NULL, 2 * ssboSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    Buffer *ssbo = ssboAllocator.allocBuffer(&bb, BUFFER_ALIGN_SHADER_STORAGE_BIT, ssboSize);
    BufferAddress ssboAddr = ssbo->GetAddress();

    LWNint uboAlignment;
    device->GetInteger(DeviceInfo::UNIFORM_BUFFER_ALIGNMENT, &uboAlignment);

    LWNsizeiptr uboSize = uboAlignment;
    if (uboSize < (LWNsizeiptr)sizeof(UBOLayout)) {
        uboSize = (LWNsizeiptr)sizeof(UBOLayout);
    }

    MemoryPoolAllocator uboAllocator(device, NULL, uboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo = uboAllocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, uboSize);
    BufferAddress uboAddr = ubo->GetAddress();
    char *uboMem = (char *)ubo->Map();
    UBOLayout *data = (UBOLayout *)(uboMem);
    data->viewport = dt::ivec4(0, 0, ssboWidth, ssboHeight);
    data->testNdx = aryNdx;

    queueCB.SetViewportScissor(0, 0, ssboWidth, ssboHeight);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(Vertex) * 4);

    static const int vpWidth = lwrrentWindowWidth / cellsX;
    static const int vpHeight = lwrrentWindowHeight / cellsY;

    bool gotExpectedPIQValues = true;
    int subroutineIndices[16] = { 0 };

    // Get the assigned subroutine index values for each subroutine.
    // If we are testing explicit layouts, then make sure the indexes are what we expect.
    for (int i = 0; i < NUM_TYPES; ++i) {
        for (int j = 0; j < NUM_UNIQUE_SUBROUTINE_BODIES; ++j) {
            int ndx = i*NUM_UNIQUE_SUBROUTINE_BODIES + j;

            subroutineIndices[ndx] = programParams->subroutines[ndx]->index;

            if (testConfig.m_doExplicitLayout) {
                int expectedNdx = (programParams->layoutNdx[j] + i * 4);
                if (subroutineIndices[ndx] != expectedNdx) {
                    // Get the name.
                    lwStringBuf subroutineName;
                    subroutineName << programParams->subroutineNamesBase[j] << i;
                    log_output("subroutine \"%s\" index mismatch, expected %d, got %d\n",
                        subroutineName.str().c_str(), expectedNdx, subroutineIndices[ndx]);
                    gotExpectedPIQValues = false;
                }
            }
        }
    }


    // Subroutine indexes into our GLSLCsubroutineInfo array for expected compatible subroutines for
    // each active uniform.
    // -1 is used to indicate there are no more expected subroutines to be compatible with that uniform.
    // Note: These are NOT the subroutine index values themselves, since those could be determined at compile time.
    // Instead this represents the subroutine based on the order of appearance in the shader, and can be used
    // to index into the programParams->subroutines GLSLCsubroutineInfo array.
    const int expectedCompatSubroutines[NUM_ACTIVE_SUBROUTINE_UNIFORMS][8] = {
        { 0, 1, 2, 3, 8, 9, 10, 11 },
        { 4, 5, 6, 7, 8, 9, 10, 11 },
        { 12, 13, 14, 15, -1, -1, -1, -1 },
        { 12, 13, 14, 15, -1, -1, -1, -1 }
    };

    int defaultSubroutineIndex[NUM_ACTIVE_SUBROUTINE_UNIFORMS] = { -1 };
    for (int i = 0; i < NUM_ACTIVE_SUBROUTINE_UNIFORMS; ++i) {
        defaultSubroutineIndex[i] = programParams->subroutines[expectedCompatSubroutines[i][0]]->index;
    }

    ShaderStage lwrrStage =
        testConfig.m_doCompute ? ShaderStage::COMPUTE : ShaderStage::FRAGMENT;

    // Test the PIQ information.
    for (int uniIndex = 0; uniIndex < 4; ++uniIndex) {
        const char * uniformName = programParams->uniformNames[uniIndex + (testConfig.m_doArray ? 4 : 0)];

        if (testConfig.m_doExplicitLayout) {
            if (programParams->activeSubroutineUniformList[uniIndex]->location != programParams->locationNdx[uniIndex]) {
                assert(false);
                log_output("Uniform %s expected to be at location %d, but is at location %d\n",
                    uniformName, programParams->locationNdx[uniIndex], programParams->activeSubroutineUniformList[uniIndex]->location);
                gotExpectedPIQValues = false;
            }
        }

        // Make sure we are compatible with the subroutines we are supposed to be compatible with.
        unsigned int numCompatibleSubroutines = 0;
        const unsigned int * reflectionSubroutineIndices = g_glslcHelper->GetCompatibleSubroutineIndices(lwrrStage, uniformName, &numCompatibleSubroutines);
        if (((uniIndex < 2) && (numCompatibleSubroutines != 8)) ||
            ((uniIndex >= 2) && (numCompatibleSubroutines != 4))) {
            log_output("Subroutine uniform %s should be compatible with %d subroutines, reflection info says only compatible with %d subroutines.",
                uniformName, (uniIndex < 2 ? 8 : 4), numCompatibleSubroutines);
            gotExpectedPIQValues = false;
        }

        for (int i = 0; i < 8; ++i) {
            if (expectedCompatSubroutines[uniIndex][i] == -1) {
                break;
            }
            assert(defaultSubroutineIndex[uniIndex] != -1);
            unsigned int subIndex = programParams->subroutines[expectedCompatSubroutines[uniIndex][i]]->index;

            bool foundCompatIndex = false;
            for (unsigned int j = 0; j < numCompatibleSubroutines; ++j) {
                if (subIndex == reflectionSubroutineIndices[j]) {
                    foundCompatIndex = true;
                    break;
                }
            }

            if (!foundCompatIndex) {
                gotExpectedPIQValues = false;
                log_output("Expected to find that subroutine uniform %s is compatible with with subroutine with index %d, but\n"
                           "GLSLC's compatible subroutine list doesn't contain that entry.\n", uniformName, subIndex);
            }
        }
    }

    // Sets up the expected color for each test.  If the value that's output is not the expected
    // color, we draw a border around the drawn square with the expected color.
    dt::vec4 expectedColors[NUM_TESTS_PER_ROW];
    for (int i = 0; i < NUM_TESTS_PER_ROW; ++i) {
        expectedColors[i] = dt::vec4(0.0, 0.0, 0.0, 0.0);

        for (int j = 0; j < 4; ++j) {
            switch (testConfigNdxs[i][j] % 4) {
                case 0: expectedColors[i].setX(1.0f); break; // red
                case 1: expectedColors[i].setY(1.0f); break; // green
                case 2: expectedColors[i].setZ(1.0f); break; // blue
                case 3: break;
                default: break;
            }
        }

        expectedColors[i].setW(1.0f);
    }

    // Even though we may have only a small number of active subroutine uniforms, our compiler
    // will assign locations all through the end, filling empty gaps between uniform locations with
    // empty uniform locations (which won't be used).  We find the maximum uniform location here, which
    // will determine the total number of uniform locations we may need to send to lwnCommandBufferSetProgramSubroutines.
    const int numSubroutineUniformLocations = programParams->maxUniformsLoc + 1;

    // Index vlaues to assign to each uniform.  This array is used directly as a parameter to
    // lwnCommandBufferSetProgramSubroutines.
    int * uniIndexes = (int *)calloc(numSubroutineUniformLocations, sizeof(int));

    // Array of bolleans indicating if the subroutine index assigned to the uniform location has changed from the previous
    // iteration.  This is used to "batch up" subroutines if exercising the range API.
    bool * uniChanged = (bool *)calloc(numSubroutineUniformLocations, sizeof(bool));

    // Set our defaults.  We just set to the first compatible subroutine for each type.
    for (int i = 0; i < numSubroutineUniformLocations; ++i) {
        // For both arrays and non-arrays, we explicitly set the layout locations of the uniforms to be <arySize>
        // apart.
        int subUniBase = i / arySize;

        uniIndexes[i] = defaultSubroutineIndex[subUniBase];
    }
    queueCB.SetProgramSubroutines(subroutineProgram, lwrrStage, 0, numSubroutineUniformLocations, uniIndexes);

    const bool doRange = testConfig.m_doRange;

    for (int col = 0; col < NUM_TESTS_PER_ROW; ++col) {
        if (gotExpectedPIQValues) {
            memset(uniChanged, 0, sizeof(bool) * numSubroutineUniformLocations);

            // Subroutine uniforms could be in discontinuous locations, where there may be inactive uniforms
            // between active uniforms (unused, and mainly due to explicit layout(location ...) qualifiers in the
            // shader.  Since we only allocate an array for the 4 subroutine uniforms, we keep the current subroutine location
            // for the active subroutines.
            int lwrrSubroutineUniformLocationNdx = 0;
            for (int i = 0; i < numSubroutineUniformLocations; ++i) {

                int passAryNdx = testConfig.m_doArray ? aryNdx : 0;
                if (!IsActiveUniformLocation(i, programParams, NUM_ACTIVE_SUBROUTINE_UNIFORMS, passAryNdx)) {
                    // Inactive uniforms should always be inactive.  Our logic for detecting which uniforms have changed
                    // to do range updates depends on inactive uniforms never changing their values.
                    // Our logic here is to ensure that array entries which we never touch will never accidently get
                    // assigned values by our logic (since an inactive uniform in a previous loop shouldn't all of a sudden
                    // be modified since we expect it to remain inactive).
                    assert(uniIndexes[i] == defaultSubroutineIndex[i/arySize]);
                    continue;
                }

                assert(testConfigNdxs[col][lwrrSubroutineUniformLocationNdx] < NUM_SUBROUTINES);
                assert(lwrrSubroutineUniformLocationNdx < NUM_ACTIVE_SUBROUTINE_UNIFORMS);

                int nextUniIndex = programParams->subroutines[testConfigNdxs[col][lwrrSubroutineUniformLocationNdx]]->index;
                if (doRange) {
                    int lastUniIndex = uniIndexes[i];
                    uniChanged[i] = (lastUniIndex != nextUniIndex);
                }
                uniIndexes[i] = nextUniIndex;
                lwrrSubroutineUniformLocationNdx++;
            }

            if (doRange) {
                // Determine changes between last subroutine bindings and this one.
                // We batch up modifications in the array and set in each batch with a separate
                // lwnCommandBufferSetProgramSubroutines call to exercise the "range" API of this call.
                for (int ii = 0; ii < numSubroutineUniformLocations; ++ii) {
                    int uniformCount = 0;
                    int startUniformIndex = ii;
                    if (uniChanged[ii]) {
                        for (int jj = ii; jj < numSubroutineUniformLocations; ++jj) {
                            if (uniChanged[jj] == true) {
                                uniformCount++;
                                // Make sure to skip over this one in the next batch.
                                ii = jj;
                            } else {
                                break;
                            }
                        }
                        // Send an update for the range of uniforms that we want to update.
                        queueCB.SetProgramSubroutines(
                            subroutineProgram, lwrrStage,
                            startUniformIndex, uniformCount, &uniIndexes[startUniformIndex]);
                    }
                }
            }
            else {
                // Update all the uniforms at once (GL-style) without using the range functionality.
                queueCB.SetProgramSubroutines(subroutineProgram, lwrrStage, 0, numSubroutineUniformLocations, uniIndexes);
            }
            queueCB.UpdateUniformBuffer(uboAddr, uboSize, 0, uboSize, data);

            queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr + 0 * uboSize, uboSize);

            // Set the scissor for this cell.  We first scissor a larger region for the cell and clear with the expected color.  Then
            // we reset the scissor to be a smaller region.  If the color drawn is not the expected color, then the diff can be seen
            // by looking at the cell's border (expected) and comparing with the center of the cell (drawn).
            queueCB.SetViewportScissor(col * vpWidth + 2, (lwrrRow + 1) * vpHeight + 2, vpWidth - 4, vpHeight - 4);
            queueCB.ClearColor(0, expectedColors[col].x(), expectedColors[col].y(), expectedColors[col].z(), expectedColors[col].w());
            queueCB.SetViewportScissor(col * vpWidth + 4, (lwrrRow + 1) * vpHeight + 4, vpWidth - 8, vpHeight - 8);

            if (testConfig.m_doCompute) {
                DrawCompute(queueCB, programParams, ssboAddr,
                    ssboSize, uboAddr, uboSize, ssboWidth,
                    ssboHeight, ctaWidth, ctaHeight);
            } else {
                queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
            }
        } else {
            // Womp womp, reflection information from GLSLC did not match up with what we explicitly programmed the shaders to.
            queueCB.SetViewportScissor(col * vpWidth + 2, (lwrrRow + 1) * vpHeight + 2, vpWidth - 4, vpHeight - 4);
            queueCB.ClearColor(0, 0.5, 0.0, 0.0, 1.0);
        }

        queueCB.submit();
    }

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queueCB.submit();
    queue->Finish();

    __LWOG_FREE(uniIndexes);
    __LWOG_FREE(uniChanged);
}

void LWNsubroutineUniformTestBasic::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

    queueCB.ClearColor(0, 0.0f, 0.0f, 0.0f, 1.0f);

    // We have 5 options we can set in the testConfig.  We test all combinations of the 5 options.
    int maxTestConfigs = 32;
    int lwrrTest = 0;
    for (int i = 0; i < maxTestConfigs; ++i) {
        TestConfig testConfig;
        testConfig.m_doCompute =        !!(i & 0x1);
        testConfig.m_doExplicitLayout = !!(i & 0x2);
        testConfig.m_doArray =          !!(i & 0x4);
        testConfig.m_doDynamicIndex =   !!(i & 0x8);
        testConfig.m_doRange =          !!(i & 0x10);

        // Skip over dynamic indexing if we aren't using arrays, since it doesn't make sense to
        // test dynamic indexing with scalars.
        if ((testConfig.m_doDynamicIndex == true) && (testConfig.m_doArray == false)) {
            continue;
        }

        ProgramParams programParams;
        if (!programParams.Init(testConfig)) {
            log_output("Problem compiling shaders or retrieving program parameters for test config %d.", i);
            return;
        }

        // Draw a row.
        RunTest(testConfig, &programParams, lwrrTest);
        lwrrTest++;
    }
}

OGTEST_CppTest(LWNsubroutineUniformTestBasic, lwn_shader_subroutine_basic, );

