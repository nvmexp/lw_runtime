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
#include "lwnTool/lwnTool_GlslcInterface.h"
#include "string.h"


using namespace lwn;

/**********************************************************************/
// LWN GLSLC Test
// Consistency tests on the GLSLC output structure.
//
// This test contains multiple subtests to check the GLSLC output
// structure for consistency and to ensure all the components are
// in the correct place.  Additionaly, there are compile only
// (no rendering) tests to exercise the GLSLC compilation path.
//
// Areas for future work:
// * More comprehensive compile tests.
// * PIQ
// * Different combinations of options.
// * Check assembly output for proper (and illegal values)
// * Negative tests.

// lwogtest doesn't like random spew to standard output.  We just eat any
// output unless LWN_DEBUG_LOG is set to 1.
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

// Enable the GLSLANGLW test.
#define TEST_GLSLANG 0

/**********************************************************************/
// GLSLC test shaders
/**********************************************************************/

// Most of these shaders come from other lwog tests.
static const char *vsstring = 
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(location = 0) in vec4 position;\n"
    "layout(location = 1) in vec4 tc;\n"
    "layout(binding = 0) uniform Block {\n"
    "    vec4 scale;\n"
    "    uint64_t bindlessTex;\n"
    "};\n"
    "out IO { vec4 ftc; };\n"
    "void main() {\n"
    "  gl_Position = position*scale;\n"
    // This line exists to trick the compiler into putting a value in the compiler
    // constant bank, so we can exercise binding that bank
    "  if (scale.z != 1.0 + 1.0/65536.0) {\n"
    "      gl_Position = vec4(0,0,0,0);\n"
    "  }\n"
    "  ftc = tc;\n"
    "}\n";

// Version of the fragment shader using bound textures.
static const char *fsstring = 
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(binding = 0) uniform sampler2D boundTex;\n"
    "layout(binding = 0) uniform Block {\n"
    "    vec4 scale;\n"
    "    uint64_t bindlessTex;\n"
    "};\n"
    "layout(location = 0) out vec4 color;\n"
    "in IO { vec4 ftc; };\n"
    "void main() {\n"
    "  color = texture(boundTex, ftc.xy);\n"
    "  if (scale.z != 1.0 + 1.0/65536.0) {\n"
    "      color = vec4(0,0,0,0);\n"
    "  }\n"
    "}\n";

static const char *gsstring =
    "#version 440 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices=3) out;\n"
    "in IO { vec4 ftc; } vi[];\n"
    "out IO { vec4 ftc; };\n"
    "void main() {\n"
    "  for (int i = 0; i < 3; i++) {\n"
    "    gl_Position = gl_in[i].gl_Position * vec4(1.0, -1.0, 1.0, 1.0);\n"
    "    ftc = vi[i].ftc * 2.0;\n"
    "    EmitVertex();\n"
    "  }\n"
    "}\n";

static const char *tcsstring =
    "#version 440 core\n"
    "#define iid gl_IlwocationID\n"
    "layout(vertices=3) out;\n"
    "in IO { vec4 ftc; } vi[];\n"
    "out IO { vec4 ftc; } vo[];\n"
    "void main() {\n"
    "  gl_out[iid].gl_Position = gl_in[iid].gl_Position.yxzw;\n"
    "  vo[iid].ftc = vi[iid].ftc * 2.0;\n"
    "  gl_TessLevelOuter[0] = 4.0;\n"
    "  gl_TessLevelOuter[1] = 4.0;\n"
    "  gl_TessLevelOuter[2] = 4.0;\n"
    "  gl_TessLevelInner[0] = 4.0;\n"
    "}\n";

static const char *tesstring =
    "#version 440 core\n"
    "layout(triangles) in;\n"
    "in IO { vec4 ftc; } vi[];\n"
    "out IO { vec4 ftc; };\n"
    "void main() {\n"
    "  gl_Position = (gl_in[0].gl_Position * gl_TessCoord.x + \n"
    "                 gl_in[1].gl_Position * gl_TessCoord.y + \n"
    "                 gl_in[2].gl_Position * gl_TessCoord.z);\n"
    "  gl_Position.xy *= 1.2;\n"
    "  ftc = 2.0 * (vi[0].ftc * gl_TessCoord.x +\n"
    "               vi[1].ftc * gl_TessCoord.y +\n"
    "               vi[2].ftc * gl_TessCoord.z);\n"
    "}\n";

// Separable shaders (from lwn_counters)
const char *vsstring_separable =
    "#version 440 core\n"
    "out gl_PerVertex {\n"
    "  vec4 gl_Position;\n"
    "};\n"
    "layout(location=0) in vec3 position;\n"
    "void main() {\n"
    "  gl_Position = vec4(position, 1.0);\n"
    "}\n";

const char *tcsstring_separable =
    "#version 440 core\n"
    "layout(vertices=4) out;\n"
    "in gl_PerVertex {\n"
    "  vec4 gl_Position;\n"
    "} gl_in[];\n"
    "out gl_PerVertex {\n"
    "  vec4 gl_Position;\n"
    "} gl_out[];\n"
    "void main() {\n"
    "  gl_out[gl_IlwocationID].gl_Position = gl_in[gl_IlwocationID].gl_Position;\n"
    "  barrier();\n"
    "  if (gl_IlwocationID == 0) {\n"
    "    for (int i = 0; i < 4; i++) { gl_TessLevelOuter[i] = 2.0; }\n"
    "    for (int i = 0; i < 2; i++) { gl_TessLevelInner[i] = 2.0; }\n"
    "  }\n"
    "}\n";

const char *tesstring_separable =
    "#version 440 core\n"
    "layout(quads) in;\n"
    "layout(equal_spacing) in;\n"
    "layout(ccw) in;\n"
    "in gl_PerVertex {\n"
    "  vec4 gl_Position;\n"
    "} gl_in[];\n"
    "out gl_PerVertex {\n"
    "  vec4 gl_Position;\n"
    "};\n"
    "void main() {\n"
    "  gl_Position = mix(mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x),\n"
    "                    mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x),\n"
    "                    gl_TessCoord.y);\n"
    "}\n";

const char *gsstring_separable =
    "#version 440 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip) out;\n"
    "layout(max_vertices=3) out;\n"
    "in gl_PerVertex {\n"
    "  vec4 gl_Position;\n"
    "} gl_in[];\n"
    "out gl_PerVertex {\n"
    "  vec4 gl_Position;\n"
    "};\n"
    "void main() {\n"
    "  for (int i = 0; i < 3; i++) {\n"
    "    gl_Position = gl_in[i].gl_Position;\n"
    "    EmitVertex();\n"
    "  }\n"
    "}\n";

enum TestResult {
    TEST_PASS = 0,
    TEST_FAIL,
    TEST_NOT_SUPPORT
};

// GLSLC test class.
class LwnGlslcTest
{
private:
    static const int cellSize = 64;
    static const int cellMargin = 2;
    static const int cellsX = 640 / (cellSize + cellMargin);
    static const int cellsY = 480 / (cellSize + cellMargin);

    void setCellRect(CommandBuffer *queueCB, int testId) const;

public:
    LWNTEST_CppMethods();
};

//
// Functions to check bits and pieces of the GLSLC output.
//

// Verifies the GPU code sections for a given compilation output.
static LWNboolean VerifyGpuCodeSections(const GLSLCoutput * output)
{
    if (!output) {
        return LWN_FALSE;
    }

    LWNshaderStage seenStages[6];

    int numGpuSections = 0;
    for (unsigned i = 0; i < output->numSections; ++i) {

        if (output->headers[i].genericHeader.common.type ==
            GLSLC_SECTION_TYPE_GPU_CODE) {
            GLSLCgpuCodeHeader gpuCodeHeader =
                output->headers[i].gpuCodeHeader;
            const char * dataSection = ((const char*)output + gpuCodeHeader.common.dataOffset);
            LWNshaderStage stage = gpuCodeHeader.stage;

            // Each GLSLCoutput contains the contents for 1 compilation,
            // so there should be no duplicate stages.
            for (int j = 0; j < numGpuSections; ++j) {
                if (stage == seenStages[j]) {
                    log_output("Duplicate stages in GPU GLSLC output.\n");
                    return LWN_FALSE;
                }
            }

            seenStages[numGpuSections] = stage;

            uint32_t dataSize = gpuCodeHeader.dataSize;
            uint32_t controlSize = gpuCodeHeader.controlSize;

            if (dataSize == 0 || controlSize == 0) {
                log_output("Data size or control size == 0\n");
                return LWN_FALSE;
            }

            // Data size should be at a granularity
            // of 256 bytes.
            if ( (dataSize & (255)) != 0 ) {
                log_output("dataSize not right granularity\n");
                return LWN_FALSE;
            }

            if ((int)(gpuCodeHeader.stage) > 5) {
                log_output("stage too large\n");
                return LWN_FALSE;
            }

            // Check the magic number of each section exists at the
            // proper offset.
            if ((((const unsigned int*)(dataSection + gpuCodeHeader.dataOffset))[0] !=
                    GLSLC_GFX_GPU_CODE_SECTION_DATA_MAGIC_NUMBER) ||
                ((const unsigned int*)(dataSection + gpuCodeHeader.controlOffset))[0] !=
                    GLSLC_GPU_CODE_SECTION_CONTROL_MAGIC_NUMBER) {
                log_output("Magic number error\n");
                return LWN_FALSE;
            }

            numGpuSections++;
        }
    }
    return LWN_TRUE;
}

// This function compares two GLSLC outputs byte-by-byte, including the size.  Will return false
// if the two outputs have byte or size differences, true otherwise.  If logging is enabled (via
// the define LWN_DEBUG_LOG), then the section of the GLSLCoutput (either header or data) is printed
// along with information about the offset of the section and which section type the byte difference
// belongs to.
static LWNboolean CompareBytes(const GLSLCoutput * glslcOutput1, const GLSLCoutput * glslcOutput2)
{
    LWNboolean match = LWN_TRUE;
    const unsigned char * bin1 = (const unsigned char*)glslcOutput1;
    const unsigned char * bin2 = (const unsigned char*)glslcOutput2;
    unsigned int size1 = glslcOutput1->size;
    unsigned int size2 = glslcOutput2->size;

    // Check the size.
    if (size1 != size2) {
        log_output("Size difference between GLSLCoutputs. output1 size: %d, output2 size: %d\n", size1, size2);
        return LWN_FALSE;
    }

    // Just do a simple memcmp.  If logging is enabled, we do more complex byte comparisons later in this function,
    // otherwise we are done.
    match = !(memcmp(glslcOutput1, glslcOutput2, size1));

    // Loop through each byte until we get to the first one that doesn't match.  Do some logging to figure out
    // which sections or section data the mismatching byte belongs to.  Print out info for each mismatching byte.
    // When logging is disabled, we just break on the first mismatched byte and return LWN_FALSE.
    if (!match && LWN_DEBUG_LOG) {
        int diffCounter = 0;
        for (unsigned int byteIndex = 0; byteIndex < size1; ++byteIndex) {
            if (bin1[byteIndex] != bin2[byteIndex]) {
                diffCounter++;
                log_output("Difference at byte %d = 0x%02x:0x%02x in ", byteIndex, bin1[byteIndex], bin2[byteIndex]);

                // Both outputs come from a compile of the same shaders, same options, and the output is the same size (checked
                // earlier in this function), so we arbitrarily use glslcOutput1 since we assume both structures are the same.
                const GLSLCoutput * output = glslcOutput1;

                if (byteIndex < sizeof(GLSLCoutput)) {
                    log_output("GLSLCoutput struct itself (top-level header)");
                } else {
                    // The byte difference is in one of the section headers, or a data section.
                    bool mismatchInHeader = (byteIndex < output->dataOffset);

                    // Find the actual section for which this byte mismatch is contained in (either header itself,
                    // or the data section of the header) and log the results.
                    for (unsigned int sectionIndex = 0; sectionIndex < output->numSections; ++sectionIndex) {
                        unsigned int headerOffset = (const char *)(&output->headers[sectionIndex].genericHeader) - (const char *)output;
                        unsigned int dataOffset = output->headers[sectionIndex].genericHeader.common.dataOffset;
                        unsigned int dataSize = output->headers[sectionIndex].genericHeader.common.size;

                        // If this is either the mismatching header or data section, print the name of the
                        // section type, and if the mismatch is in the data section, print the section
                        // offset.
                        // The first conditional checks if the mismatched byte is in the header,
                        // (we've already checked if byteIndex < sizeof(GLSLCoutput, so at this point we are NOT in the top-level
                        // GLSLCoutput struct).
                        // The second conditional checks if the mismatched byte is in the header's data section.
                        if ((byteIndex < (sizeof(GLSLCoutput) + sizeof(GLSLCsectionHeaderUnion)*(sectionIndex + 1))) ||
                            ((byteIndex > dataOffset) && byteIndex < (dataOffset + dataSize))) {

                            if (mismatchInHeader) {
                                log_output("header #%d of type ", sectionIndex);
                            } else {
                                log_output("data section for header #%d of type ", sectionIndex);
                            }

                            // Log which section type this header or data belongs to.
                            switch (output->headers[sectionIndex].genericHeader.common.type) {
                            case GLSLC_SECTION_TYPE_GPU_CODE:
                                log_output("GLSLC_SECTION_TYPE_GPU_CODE");
                                break;
                            case GLSLC_SECTION_TYPE_PERF_STATS:
                                log_output("GLSLC_SECTION_TYPE_PERF_STATS");
                                break;
                            case GLSLC_SECTION_TYPE_REFLECTION:
                                log_output("GLSLC_SECTION_TYPE_REFLECTION");
                                break;
                            case GLSLC_SECTION_TYPE_DEBUG_INFO:
                                log_output("GLSLC_SECTION_TYPE_DEBUG_INFO");
                                break;
                            default:
                                log_output("UNKNOWN SECTION!\n");
                                break;
                            };

                            // Print out which offset the header starts to help with debugging.
                            log_output(" which starts at offset %d\n", mismatchInHeader ? headerOffset : dataOffset);
                            break;
                        }
                    }
                }
                log_output("\n");
            }
        }

        log_output("Total bytes different: %d\n\n", diffCounter);
    }

    return match;
}

// This function tests that the control and data sections of two binaries match taking into account
// expected differences.  A customer request is that any entries in GLSLC's control and data sections
// which aren't indicative of functional differences in the generated code be tracked.
//
// This test is designed to fail if any entries are added to the control section which would cause otherwise
// identical shaders to produce byte-level differences in their control sections.  A simple example of these types of shaders
// are those shaders that only differ in GLSL comments in the sources.  While these shaders produce different
// debug data / hashes in the control section, the rest of the control section should be identical.
// If a disruptive entry is added to the GLSLC control section which would cause this test to fail, then the
// test must be modified (via the struct ExpectedControlDiffs below) to account for those diffs and the entry
// byte offsets in the control section must be reported to the customer so they can track.
//
// Note: There shouldn't be any expected diffs in the data section, only the control section.
static TestResult TestEntriesThatDontModifyBinaryOutput(void)
{
    VertexShader vs(450);
    FragmentShader fs_ver1(450);

    // The same as fs_ver1 except the shader source has comments.  This will force some diffs, such
    // as the debug hash, but should not affect the compiled code or the rest of the control section.
    FragmentShader fs_ver2(450);
    LWNboolean success = LWN_TRUE;

    vs <<
        "out gl_PerVertex {\n"
        "  vec4 gl_Position;\n"
        "};\n"
        "void main() {\n"
        "  gl_Position = vec4(1, 0, 1, 1);\n"
        "}\n";

    fs_ver1 <<
        "out vec4 ocol;\n"
        "void main() {\n"
        "  ocol = vec4(1, 0, 1, 1);\n"
        "}\n";

    fs_ver2 <<
        "// A comment should not affect the binary output.\n"
        "out vec4 ocol;\n"
        "void main() {\n"
        "// Another comment for kicks.\n"
        "  ocol = vec4(1, 0, 1, 1);\n"
        "}\n";

    // A list of expected diffs in the control section.  The diffs are represented by
    // offset/size pairs.  The offset is the offset from the beginning of the control
    // section.  The size is the size of the diff entry in bytes.
    struct ExpectedControlDiffs {
        // Offset from the beginning of the control section.
        int m_startControlOffset;

        // Size of the entry (in bytes).
        int m_size;
    } expectedControlDiffs[] = {
        { 1896, 16 }, // Debug build-id
        { 1912, 8 },  // Debug hash
        { 2000, 8 },  // Source hash
        { 2008, 8 },  // GLASM hash
        { 2016, 8 },  // Ucode hash
    };

    // Contains all of the expected control diffs.  We will use this to loop over the
    // control sections and zero-out these entries to make comparison simpler.
    struct DiffList {
        // An array of ExpectedControlDiffs structs
        ExpectedControlDiffs * m_expectedControlDiffs;

        // Number of entries in the array m_expectedControlDiffs
        int m_numControlDiffs;
    } diffList = {
        expectedControlDiffs, (sizeof(expectedControlDiffs) / sizeof(ExpectedControlDiffs))
    };

    // Set the debug level to G0.  This is sufficient to produce hash-related diffs in the control
    // section but not modify the generated code in the data section.
    g_glslcHelper->SetDebugLevel(GLSLC_DEBUG_LEVEL_G0);

    // Separable shouldn't cause compiled code differences in the shaders for this test,
    // but will force any hashes embedded in the GLSLC output sections that depend on GLSLCoptions to be
    // different.
    g_glslcHelper->SetSeparable(LWN_TRUE);

    // The outputs for our two programs.
    GLSLCoutput * outputs[2] = { NULL };

    // Compile version 1 and copy the binary.
    if (!g_glslcHelper->CompileShaders(vs, fs_ver1)) {
        return TEST_FAIL;
    }
    const GLSLCoutput * tmp = g_glslcHelper->GetCompiledOutput(0);
    outputs[0] = (GLSLCoutput *)malloc(tmp->size);
    memcpy(outputs[0], tmp, tmp->size);

    // Compile version 2 and copy the binary.
    if (!g_glslcHelper->CompileShaders(vs, fs_ver2)) {
        log_output(g_glslcHelper->GetInfoLog());
        free(outputs[0]);
        return TEST_FAIL;
    }
    tmp = g_glslcHelper->GetCompiledOutput(0);
    outputs[1] = (GLSLCoutput *)malloc(tmp->size);
    memcpy(outputs[1], tmp, tmp->size);

    if (outputs[0]->numSections != outputs[1]->numSections) {
        log_output("Outputs between the two programs contain a different number of sections!");
        success = LWN_FALSE;
    }

    // Go through each of the expected diffs and zero-out the entries in the control sections and then
    // compare the control and data sections.
    // This makes comparing the binaries afterwards much simpler.
    // We are only concerned about comparing the control and data sections here, not any other sections that
    // are part of the GLSLCoutput (which will probably contain mismatching sections, such as the
    // debug data section).
    for (unsigned int sectionIndex = 0;
        (sectionIndex < outputs[0]->numSections) && (success == LWN_TRUE);
        ++sectionIndex) {
        if (outputs[0]->headers[sectionIndex].genericHeader.common.type != GLSLC_SECTION_TYPE_GPU_CODE) {
            continue;
        }

        if (outputs[1]->headers[sectionIndex].genericHeader.common.type != GLSLC_SECTION_TYPE_GPU_CODE) {
            // Our binaries should have the same structure.
            success = LWN_FALSE;
            log_output("Version 1 and version 2 GLSLCoutputs do not have the same structure!");
            break;
        }

        GLSLCgpuCodeHeader * gpuCodeSectionHeaders[2];
        char * sectionData[2];
        char * dataSubsection[2];
        char * controlSubsection[2];

        // Fill out pointers to the proper sections.
        for (int j = 0; j < 2; ++j) {
            gpuCodeSectionHeaders[j] = &outputs[j]->headers[sectionIndex].gpuCodeHeader;
            sectionData[j] = gpuCodeSectionHeaders[j]->common.dataOffset + (char *)outputs[j];
            dataSubsection[j] = sectionData[j] + gpuCodeSectionHeaders[j]->dataOffset;
            controlSubsection[j] = sectionData[j] + gpuCodeSectionHeaders[j]->controlOffset;
        }

        if (gpuCodeSectionHeaders[0]->common.size != gpuCodeSectionHeaders[1]->common.size ||
            gpuCodeSectionHeaders[0]->controlSize != gpuCodeSectionHeaders[1]->controlSize ||
            gpuCodeSectionHeaders[0]->dataSize != gpuCodeSectionHeaders[1]->dataSize) {
            // Our gpu code sections and subsections should be the same size.
            log_output("Data section sizes or subsection sizes are not the same!");
            success = LWN_FALSE;
            break;
        }

        for (int diffIndex = 0; diffIndex < diffList.m_numControlDiffs; ++diffIndex) {
            if (
                (diffList.m_expectedControlDiffs[diffIndex].m_startControlOffset + diffList.m_expectedControlDiffs[diffIndex].m_size) >
                (int)gpuCodeSectionHeaders[0]->common.size) {
                log_output("Expected diff offset/size not in the binary's data range!  This is probably a test bug.");
                success = LWN_FALSE;
                break;
            }

            // Zero-out the expected diff sections.
            memset(controlSubsection[0] + diffList.m_expectedControlDiffs[diffIndex].m_startControlOffset,
                0,
                diffList.m_expectedControlDiffs[diffIndex].m_size);
            memset(controlSubsection[1] + diffList.m_expectedControlDiffs[diffIndex].m_startControlOffset,
                0,
                diffList.m_expectedControlDiffs[diffIndex].m_size);
        }

        if (memcmp(dataSubsection[0], dataSubsection[1], gpuCodeSectionHeaders[0]->dataSize)) {
            log_output("GPU data section differs!");
            success = LWN_FALSE;
        }

        if (memcmp(controlSubsection[0], controlSubsection[1], gpuCodeSectionHeaders[0]->controlSize)) {
            log_output("Control section differs!");
            success = LWN_FALSE;
        }
    }

    free(outputs[0]);
    free(outputs[1]);

    return success ? TEST_PASS : TEST_FAIL;
}

// A series of both negative and positive tests designed to check error handling for when there
// might not be enough binding slots in the shader to assign non-explicitly assigned resource bindings.
static TestResult TestOverloadedResourceBindings(void)
{
    int maxSamplers = 0;
    int maxImages = 0;
    int maxSsbos = 0;
    int maxUbos = 0;
    bool returnPass = true;

    // Get the maximum bindings per stage for each resource type.
    lwnDeviceGetInteger(g_lwnDevice, LWN_DEVICE_INFO_TEXTURE_BINDINGS_PER_STAGE, &maxSamplers);
    lwnDeviceGetInteger(g_lwnDevice, LWN_DEVICE_INFO_IMAGE_BINDINGS_PER_STAGE, &maxImages);
    lwnDeviceGetInteger(g_lwnDevice, LWN_DEVICE_INFO_SHADER_STORAGE_BUFFER_BINDINGS_PER_STAGE, &maxSsbos);
    lwnDeviceGetInteger(g_lwnDevice, LWN_DEVICE_INFO_UNIFORM_BUFFER_BINDINGS_PER_STAGE, &maxUbos);

    assert(maxSamplers > 0);
    assert(maxImages > 0);
    assert(maxSsbos > 0);
    assert(maxUbos > 0);

    enum ResourceTypeEnum {
        RESOURCE_TYPE_UBOS = 0,
        RESOURCE_TYPE_SSBOS,
        RESOURCE_TYPE_SAMPLERS,
        RESOURCE_TYPE_IMAGES,
        NUM_RESOURCE_TYPES
    };

    enum TestCase {
        // A default passing case.  Since we can't be 100% certain that the failures below are actually the failures
        // we expect (such as a syntax error, which is unexpected), this serves is a control to ensure that most
        // of common bits of the shader compiles fine.  The differences in the shaders with the fail cases should be
        // minor.  This will use the MAX amount of resources allowed in a shader.
        DEFAULT_PASS = 0,
        // Compilation will fail due to too many resources.
        // The shaders will have more active resources declared than can be allowed per one shader stage.  The shaders are
        // similar to DEFAULT_PASS, but include an extra resource past the maximum allowed.
        FAIL_TOO_MANY_RESOURCES,
        // Tests that an active array of size <n>, where <n> < MAX_BINDINGS-1, will not fit in a binding slot when another
        // binding is explicitly declared somewhere in the middle (i.e. the binding "splits" the array, and the array
        // must fit in continuous locations, so this should fail).
        FAIL_CANT_FIT_ARRAY,
        // Tests that arrays can be bound in a contiguous slot, where the first free slot does
        // not fall in a contiguous range and the array has to be assigned later in the bindings, but
        // should still fit in the remaining bindings.
        PASS_CAN_FIT_ARRAY,
        // Same as the FAIL_CANT_FIT_ARRAY case, with the exception that none of the array elements are referenced
        // in the shader, so the array becomes unused and doesn't take up binding slots (perfectly legal in LWN-land).
        PASS_UNUSED_ARRAY,
        NUM_TEST_CASES
    };

    // Expected compilation pass or fail for each of the cases in the TestCase enum.
    bool expectedPassResults[NUM_TEST_CASES] = {
        true,  // DEFAULT_PASS
        false, // FAIL_TOO_MANY_RESOURCES
        false, // FAIL_CANT_FIT_ARRAY
        true,  // PASS_CAN_FIT_ARRAY
        true   // PASS_UNUSED_ARRAY
    };

    // Loop over each resource type.
    for (int resType = RESOURCE_TYPE_UBOS; resType < NUM_RESOURCE_TYPES; ++resType) {
        for (int j = DEFAULT_PASS; j < NUM_TEST_CASES; ++j) {
            LWNboolean expectedPass = (expectedPassResults[j]);
            LWNboolean testingArray =
                (j == FAIL_CANT_FIT_ARRAY) ||
                (j == PASS_CAN_FIT_ARRAY) ||
                (j == PASS_UNUSED_ARRAY);
            int numResources = 0;
            int maxResources = 0;

            switch (resType) {
            case RESOURCE_TYPE_UBOS:
                maxResources = maxUbos;
                break;
            case RESOURCE_TYPE_SSBOS:
                maxResources = maxSsbos;
                break;
            case RESOURCE_TYPE_SAMPLERS:
                maxResources = maxSamplers;
                break;
            case RESOURCE_TYPE_IMAGES:
                maxResources = maxImages;
                break;
            default:
                assert(!"Unknown resource type");
                break;
                return TEST_FAIL;
            };

            // For non-arrays, use either max resources (expected to pass), or max resources + 1 (expected to
            // fail).
            // For arrays, we just use one array.
            if (!testingArray) {
                numResources = (expectedPass ? maxResources : maxResources + 1);
            } else {
                numResources = 1;
            }

            // Use the standard vertex shader.
            VertexShader vs(440);
            vs << vsstring;

            FragmentShader fsSamplerOverload(440);

            // String declaring the resource variables in the shader.
            lwStringBuf declarationString;
            // String assigning resource variables so they get marked as active in
            // tests that use them.
            lwStringBuf assignString;

            // Loop through each resource (except for arrays) and set up a declaration string and a
            // reassignment string which uses it.
            // "Wedge" is used in array cases to try and split the array, making the compiler unable
            // to find a continuous locations for the array.  When the array is not active (in cases
            // of PASS_UNUSED_ARRAY), wedge should _not_ cause the compiler to fail.
            for (int i = 0; i < numResources; ++i) {
                switch (resType) {
                case RESOURCE_TYPE_UBOS:
                    if (testingArray) {
                        declarationString << "layout (binding = 1) uniform Wedge { vec4 val; } wedge0;\n";
                    }
                    declarationString << "uniform Block" << i << " { vec4 val; } block";

                    if (j == PASS_UNUSED_ARRAY ||
                        j == PASS_CAN_FIT_ARRAY) {
                        assignString << " wedge0.val ";

                        if (j != PASS_UNUSED_ARRAY) {
                            assignString << " + ";
                        }
                    }

                    if ((j == FAIL_CANT_FIT_ARRAY) ||
                        (j == PASS_CAN_FIT_ARRAY)) {
                        assignString << "block0[0].val";
                    } else if (!testingArray) {
                        assignString << "block" << i << ".val\n";
                    }
                    break;
                case RESOURCE_TYPE_SSBOS:
                    if (testingArray) {
                        declarationString << "layout (binding = 1) buffer Wedge { vec4 val; } wedge0;\n";
                    }
                    declarationString << "buffer Buffer" << i << " { vec4 val; } buffer";

                    if (j == PASS_UNUSED_ARRAY ||
                        j == PASS_CAN_FIT_ARRAY) {
                        assignString << " wedge0.val ";

                        if (j != PASS_UNUSED_ARRAY) {
                            assignString << " + ";
                        }
                    }

                    if ((j == FAIL_CANT_FIT_ARRAY) ||
                        (j == PASS_CAN_FIT_ARRAY)) {
                        assignString << "buffer0[0].val";
                    } else if (!testingArray) {
                        assignString << "buffer" << i << ".val\n";
                    }

                    break;
                case RESOURCE_TYPE_SAMPLERS:

                    if (testingArray) {
                        declarationString << "layout (binding = 1) uniform sampler2D wedge0;\n";
                    }
                    declarationString << "uniform sampler2D tex";

                    if (j == PASS_UNUSED_ARRAY ||
                        j == PASS_CAN_FIT_ARRAY) {
                        assignString << " texture(wedge0, vec2(0.5, 0.5)) ";

                        if (j != PASS_UNUSED_ARRAY) {
                            assignString << " + ";
                        }
                    }

                    if ((j == FAIL_CANT_FIT_ARRAY) ||
                        (j == PASS_CAN_FIT_ARRAY)) {
                        assignString << "texture(tex0[0], vec2(0.5, 0.5)) ";
                    } else if (!testingArray) {
                        assignString << "texture(tex" << i << ", vec2(0.5, 0.5))";
                    }

                    break;
                case RESOURCE_TYPE_IMAGES:
                    if (testingArray) {
                        declarationString << "layout (binding = 1) layout(r32f) uniform image2D wedge0;\n";
                    }
                    declarationString << "layout(r32f) uniform image2D img";

                    if (j == PASS_UNUSED_ARRAY ||
                        j == PASS_CAN_FIT_ARRAY) {
                        assignString << " imageLoad(wedge0, ivec2(0)) ";

                        if (j != PASS_UNUSED_ARRAY) {
                            assignString << " + ";
                        }
                    }

                    if ((j == FAIL_CANT_FIT_ARRAY) ||
                        (j == PASS_CAN_FIT_ARRAY)) {
                        assignString << "imageLoad(img0[0], ivec2(0)) ";
                    } else if (!testingArray) {
                        assignString << "imageLoad(img" << i << ", ivec2(0))";
                    }

                    break;
                default:
                    assert(!"Unknown resource type");
                    return TEST_FAIL;
                };

                declarationString << i;

                // If this is an array, then we use an array which has a size to fill up the maximum amount
                // of bindings possible for this particular resource.
                if (testingArray) {
                    declarationString << "[";
                    if (j == FAIL_CANT_FIT_ARRAY || j == PASS_UNUSED_ARRAY) {
                        // Both of these test cases use an array that couldn't possibly fit with the wedge value
                        // also being used.
                        declarationString << maxResources;
                    } else {
                        // Assert the test is PASS_CAN_FIT_ARRAY since that's the only other possibility at
                        // this point.  If new cases are added, this will remind to not forget about this
                        // conditional.
                        assert(j == PASS_CAN_FIT_ARRAY);
                        // This test case tests with the "wedge" variable also used, but the array size _can_
                        // fit into contiguous binding slots (the wedge binding == 1), but not the first free
                        // slot.
                        declarationString << maxResources - 2;
                    }
                    declarationString << "]";
                }
                declarationString << ";\n";

                if (i == (numResources - 1)) {
                    assignString << ";\n";
                } else {
                    assignString << " +\n";
                }
            }

            fsSamplerOverload <<
                declarationString.str().c_str() <<
                "in IO { vec4 ftc; };\n"
                "out vec4 ocolor;\n"
                "void main() {\n"
                "  ocolor += \n" <<
                assignString.str().c_str();

            fsSamplerOverload << "}\n";

            LWNboolean compileSuccess = g_glslcHelper->CompileShaders(vs, fsSamplerOverload);

            // We either expect to pass or fail compilation.  If we don't get what's expected, we fail this test.
            if (expectedPass != compileSuccess) {
                if (compileSuccess) {
                    log_output("Expected to fail compilation, but succeeded!\n");
                } else {
                    log_output("Expected to succeed compilation, but failed!\n");
                }
                log_output("Resource type: %d\n", resType);
                log_output("Test test variant: %d\n", j);
                log_output("Fragment shader: %s\n", fsSamplerOverload.source().c_str());
                returnPass = LWN_FALSE;
            }
        }
    }

    return returnPass ? TEST_PASS : TEST_FAIL;
}

// Tests that GLSLCoutput binaries between compiles using the same shaders and same options are byte-for-byte
// identical.
static TestResult TestMatchingOutputs(void)
{
    LWNboolean success = LWN_TRUE;

    const char * shaders[] = {
        fsstring,
        vsstring,
        gsstring,
        tcsstring,
        tesstring,
    };

    LWNshaderStage stages[] =
    {
        LWN_SHADER_STAGE_FRAGMENT,
        LWN_SHADER_STAGE_VERTEX,
        LWN_SHADER_STAGE_GEOMETRY,
        LWN_SHADER_STAGE_TESS_CONTROL,
        LWN_SHADER_STAGE_TESS_EVALUATION
    };

    // Compile all the shaders multiple times, and compare the binary outputs of all.
    GLSLCoutput * lastOutput = NULL;

    // Pick some semi-large number of compiles since this test is non-deterministic by nature and we want to
    // maximize the chances of hitting a failure.
    int numLoops = 20;

    // Set the maximum debug level so that we get debug sections as output, which adds more sections to the GLSLCoutput
    // with potential for mismatching bytes.
    g_glslcHelper->SetDebugLevel(GLSLC_DEBUG_LEVEL_G2);
    for (int i = 0; i < numLoops; ++i) {
        if (!g_glslcHelper->CompileShaders(stages, __GL_ARRAYSIZE(stages), shaders)) {
            log_output("Error: %s\n", g_glslcHelper->GetInfoLog());
        }

        // Get the GLSLCoutput
        const GLSLCoutput * output = g_glslcHelper->GetGlslcOutput();
        int size = output->size;

        if (lastOutput != NULL) {
            assert(i > 0);

            // Check the byte difference.
            if (!CompareBytes(lastOutput, output)) {
                success = LWN_FALSE;
                break;
            }

            // So far, size and binary contents match.  If not, we should have aborted this loop sooner.
            assert(success == LWN_TRUE);
        }

        if (lastOutput) {
            __LWOG_FREE(lastOutput);
            lastOutput = NULL;
        }

        lastOutput = (GLSLCoutput *)__LWOG_MALLOC(size);
        memcpy(lastOutput, output, size);
    }

    if (lastOutput) {
        __LWOG_FREE(lastOutput);
        lastOutput = NULL;
    }
    g_glslcHelper->SetDebugLevel(GLSLC_DEBUG_LEVEL_NONE);

    return success ? TEST_PASS : TEST_FAIL;
}

static TestResult TestShaderStageDefines(void)
{
    // These shaders test that the per-shader-stage defines work appropriately.
    // This is a LwnGLSLC feature.

    const char *vsstring_define =
        "#version 440 core\n"
        "#extension GL_LW_gpu_shader5:require\n"

         // Test __VERTEX_SHADER define is the only thing enabled
        "#if !__VERTEX_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __GEOMETRY_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __TESSELLATION_CONTROL_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __TESSELLATION_EVALUATION_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __COMPUTE_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __FRAGMENT_SHADER\n"
        "compile_error\n"
        "#endif\n"

        "layout(location = 0) in vec4 position;\n"
        "layout(location = 1) in vec4 tc;\n"
        "out IO { vec4 ftc; };\n"
        "void main() {\n"
        "  gl_Position = position;\n"
        "  ftc = tc;\n"
        "}\n";

    static const char *fsstring_define =
        "#version 440 core\n"
        "#extension GL_LW_gpu_shader5:require\n"
        "layout(location = 0) out vec4 color;\n"

        // Test __FRAGMENT_SHADER is the only thing defined.
        "#if __VERTEX_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __GEOMETRY_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __TESSELLATION_CONTROL_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __TESSELLATION_EVALUATION_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __COMPUTE_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if !__FRAGMENT_SHADER\n"
        "compile_error\n"
        "#endif\n"

        "in IO { vec4 ftc; };\n"
        "void main() {\n"
        "  color = vec4(0.2f, 0.5f, 1.0f, 0.0f);\n"
        "}\n";

    static const char *gsstring_define =
        "#version 440 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "in IO { vec4 ftc; } vi[];\n"
        "out IO { vec4 ftc; };\n"

        // Test __GEOMETRY_SHADER is the only thing defined.
        "#if __VERTEX_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if !__GEOMETRY_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __TESSELLATION_CONTROL_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __TESSELLATION_EVALUATION_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __COMPUTE_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __FRAGMENT_SHADER\n"
        "compile_error\n"
        "#endif\n"

        "void main() {\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    gl_Position = gl_in[i].gl_Position * vec4(1.0, -1.0, 1.0, 1.0);\n"
        "    ftc = vi[i].ftc * 2.0;\n"
        "    EmitVertex();\n"
        "  }\n"
        "}\n";

    static const char *tcsstring_define =
        "#version 440 core\n"
        "#define iid gl_IlwocationID\n"
        "layout(vertices=3) out;\n"
        "in IO { vec4 ftc; } vi[];\n"
        "out IO { vec4 ftc; } vo[];\n"

        // Test __TESSELLATION_CONTROL_SHADER is the only thing defined
        "#if __VERTEX_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __GEOMETRY_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if !__TESSELLATION_CONTROL_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __TESSELLATION_EVALUATION_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __COMPUTE_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __FRAGMENT_SHADER\n"
        "compile_error\n"
        "#endif\n"

        "void main() {\n"
        "  gl_out[iid].gl_Position = gl_in[iid].gl_Position.yxzw;\n"
        "  vo[iid].ftc = vi[iid].ftc * 2.0;\n"
        "  gl_TessLevelOuter[0] = 4.0;\n"
        "  gl_TessLevelOuter[1] = 4.0;\n"
        "  gl_TessLevelOuter[2] = 4.0;\n"
        "  gl_TessLevelInner[0] = 4.0;\n"
        "}\n";

    static const char *tesstring_define =
        "#version 440 core\n"
        "layout(triangles) in;\n"
        "in IO { vec4 ftc; } vi[];\n"
        "out IO { vec4 ftc; };\n"

        // Test __TESSELLATION_EVALUATION_SHADER is the only thing defined.
        "#if __VERTEX_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __GEOMETRY_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __TESSELLATION_CONTROL_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if !__TESSELLATION_EVALUATION_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __COMPUTE_SHADER\n"
        "compile_error\n"
        "#endif\n"
        "#if __FRAGMENT_SHADER\n"
        "compile_error\n"
        "#endif\n"

        "void main() {\n"
        "  gl_Position = (gl_in[0].gl_Position * gl_TessCoord.x + \n"
        "                 gl_in[1].gl_Position * gl_TessCoord.y + \n"
        "                 gl_in[2].gl_Position * gl_TessCoord.z);\n"
        "  gl_Position.xy *= 1.2;\n"
        "  ftc = 2.0 * (vi[0].ftc * gl_TessCoord.x +\n"
        "               vi[1].ftc * gl_TessCoord.y +\n"
        "               vi[2].ftc * gl_TessCoord.z);\n"
        "}\n";

    const char * shaders[] = {
        fsstring_define,
        vsstring_define,
        gsstring_define,
        tcsstring_define,
        tesstring_define,
    };

    LWNshaderStage stages[] =
    {
        LWN_SHADER_STAGE_FRAGMENT,
        LWN_SHADER_STAGE_VERTEX,
        LWN_SHADER_STAGE_GEOMETRY,
        LWN_SHADER_STAGE_TESS_CONTROL,
        LWN_SHADER_STAGE_TESS_EVALUATION
    };

    if (!g_glslcHelper->CompileShaders(stages, __GL_ARRAYSIZE(stages), shaders)) {
        log_output(g_glslcHelper->GetInfoLog());
        return TEST_FAIL;
    }

    return TEST_PASS;
}

static TestResult TestPerfStats(void)
{
    LWNboolean success = LWN_TRUE;

    const char * shaders[] = {
        fsstring,
        vsstring,
        gsstring,
        tcsstring,
        tesstring,
    };

    LWNshaderStage stages[] =
    {
        LWN_SHADER_STAGE_FRAGMENT,
        LWN_SHADER_STAGE_VERTEX,
        LWN_SHADER_STAGE_GEOMETRY,
        LWN_SHADER_STAGE_TESS_CONTROL,
        LWN_SHADER_STAGE_TESS_EVALUATION
    };

    g_glslcHelper->EnablePerfStats(LWN_TRUE);
    if (!g_glslcHelper->CompileShaders(stages, __GL_ARRAYSIZE(stages), shaders)) {
        log_output("Error: %s\n", g_glslcHelper->GetInfoLog());
    }

    
    // Sanity check the perf statistics
    const GLSLCoutput * output =
        g_glslcHelper->GetGlslcOutput();

    if (!output) {
        log_output("Perf stats test: No output\n");
        return TEST_FAIL;
    }

    for (unsigned int i = 0; i < __GL_ARRAYSIZE(stages); ++i) {
        const GLSLCgpuCodeHeader * gpuCodeHeader =
            lwnTest::GLSLCHelper::ExtractGpuCodeSection(output, stages[i]);

        // No GPU code section for this stage!!
        if (!gpuCodeHeader) {
            log_output("Perf stats test: No gpu header for a stage.\n");
            return TEST_FAIL;
        }

        const GLSLCperfStatsHeader * perfStatsHeader =
            lwnTest::GLSLCHelper::ExtractPerfStatsSection(output, gpuCodeHeader);

        if (!perfStatsHeader) {
            log_output("Perf stats test: No perf stats section for a stage.\n");
            return TEST_FAIL;
        }

        GLSLCperfStatsData * psData =
            (GLSLCperfStatsData *)(((char*)output) + perfStatsHeader->common.dataOffset);

        // Double check the magic number.
        if (psData->magic != GLSLC_PERF_STATS_SECTION_MAGIC_NUMBER) {
            log_output("Perf stats test: Magic number is wrong.\n");
            return TEST_FAIL;
        }

        // Check that latency is > 0 and < 1000.  For the simple sahders
        // provided, anything out side of that would be an insane value.
        success = success &&
            ((psData->latency > 0) && (psData->latency < 1000));
        if (!success) {
            log_output("Perf stats test: Latency may be too large: %d\n", psData->latency);
            return TEST_FAIL;
        }

        // Oclwpancy should be between 0-1
        success = success &&
            ((psData->oclwpancy >= 0.0f) && (psData->oclwpancy <= 1.0f));
        if (!success) {
            log_output("Perf stats test: Oclwpancy is wrong: %f\n", psData->oclwpancy);
        }

        // Ensure the program size is <= the corresponding GPU section's data size.
        success = success &&
                  ((psData->programSize > 0) && (psData->programSize <= gpuCodeHeader->dataSize));
        if (!success) {
            log_output("Perf stats test: program size is wrong: %d\n", psData->programSize);
        }
    }

    return success ? TEST_PASS : TEST_FAIL;
}

static TestResult TestCompile(void)
{
    LWNboolean success = LWN_TRUE;

    const char * shaders[5] = {
        fsstring,
        vsstring,
        gsstring,
        tesstring,
        tcsstring
    };

    const char * shaders_separable[5] = {
        fsstring,
        vsstring_separable,
        gsstring_separable,
        tesstring_separable,
        tcsstring_separable
    };

    LWNshaderStage stages[5] =
    {
        LWN_SHADER_STAGE_FRAGMENT,
        LWN_SHADER_STAGE_VERTEX,
        LWN_SHADER_STAGE_GEOMETRY,
        LWN_SHADER_STAGE_TESS_EVALUATION,
        LWN_SHADER_STAGE_TESS_CONTROL,
    };

    // Test fs+vs, fs+vs+gs, fs+vs+gs+tes, fs+vs+gs+tes+tcs
    g_glslcHelper->SetSeparable(LWN_FALSE);
    for (int i = 0; i < 4; ++i) {
        success = success &&
            g_glslcHelper->CompileShaders(stages, i+2, shaders);

        if (!success) {
            log_output("Compile test: Compilation failure\n");
            log_output("Info log: \n%s\n", g_glslcHelper->GetInfoLog());
            return TEST_FAIL;
        }
        success = success &&
            VerifyGpuCodeSections(g_glslcHelper->GetGlslcOutput());

        if (!success) {
            log_output("Compile test: Verification failure\n");
            return TEST_PASS;
        }
    }


    // Test separable: vs, fs, gs, tcs, tes
    g_glslcHelper->SetSeparable(LWN_TRUE);
    for (unsigned int i = 0; i < __GL_ARRAYSIZE(shaders); ++i) {
        success = success &&
            g_glslcHelper->CompileShaders(&stages[i], 1, &shaders_separable[i]);

        if (!success) {
            log_output("Separable compile test: Compilation failure\n");
            log_output("Info log: \n%s\n", g_glslcHelper->GetInfoLog());
        }
        success = success &&
            VerifyGpuCodeSections(g_glslcHelper->GetGlslcOutput());
    }

    return success ? TEST_PASS : TEST_FAIL;
}

// Types of debug info sections.  Corresponds to the internal enums in the debug data.
enum DebugInfoSectionTypeEnum
{
    // GLASM with embedded debug information.
    DEBUG_INFO_SECTION_TYPE_ASM = 0,

    // ucode debug information in the format described in
    // //sw/devtools/Agora/Specs/Perfkit/DevtoolsAPIUCodeFormat.html.
    DEBUG_INFO_SECTION_TYPE_GPU_CODE,

    // Specialization information
    DEBUG_INFO_SECTION_TYPE_SPECIALIZATION,

    // Serialized XFB information
    DEBUG_INFO_SECTION_TYPE_XFB
};


static TestResult TestMagicNumbers(void)
{
    // Tests to ensure that the magic numbers in each section are correct.
    // We compile with all available sections and go through each one.
    LWNboolean success = LWN_TRUE;

    // Generate all available sections.
    g_glslcHelper->EnablePerfStats(LWN_TRUE);

    const char * shaders[2] = {
        vsstring,
        fsstring
    };

    LWNshaderStage stages[2] =
    {
        LWN_SHADER_STAGE_VERTEX,
        LWN_SHADER_STAGE_FRAGMENT
    };
    LWNboolean compSuccess = g_glslcHelper->CompileShaders(stages, 2, shaders);
    if (!compSuccess) {
        // xxx log?
        return TEST_FAIL;
    }

    const GLSLCoutput * output = g_glslcHelper->GetGlslcOutput();

    // This should actually indicate an error in the glslcHelper...
    success = success && (output != NULL);

    // Check output's magic number.
    int outputMagicNumber = output->magic;
    success = success && (outputMagicNumber == GLSLC_MAGIC_NUMBER);

    // Magic number must be the first 4 bytes of the output
    success = success && (((uint32_t *)output)[0] == GLSLC_MAGIC_NUMBER);

    // Loop through each section to check the magic numbers for that section.
    for (unsigned int i = 0; i < __GL_ARRAYSIZE(stages); ++i) {
        const char * sectionData = NULL;

        //
        // Test the gpu code section.
        //
        const GLSLCgpuCodeHeader * gpuCodeHeader =
            lwnTest::GLSLCHelper::ExtractGpuCodeSection(output, stages[i]);

        if (gpuCodeHeader == NULL) {
            return TEST_FAIL;
        }

        sectionData = ((const char*)output) + gpuCodeHeader->common.dataOffset;

        success = success && (gpuCodeHeader != NULL);

        const char * ucodeData = NULL;
        const char * controlData = NULL;

        // Check ucode data and control sections.
        ucodeData = sectionData + gpuCodeHeader->dataOffset;
        controlData = sectionData + gpuCodeHeader->controlOffset;

        success = success &&
            (((const uint32_t *)ucodeData)[0] ==
             GLSLC_GFX_GPU_CODE_SECTION_DATA_MAGIC_NUMBER);

        success = success &&
            (((const uint32_t *)controlData)[0] ==
             GLSLC_GPU_CODE_SECTION_CONTROL_MAGIC_NUMBER);


        //
        // Check the perf stats section.
        //
        const GLSLCperfStatsHeader * perfStatsHeader =
            lwnTest::GLSLCHelper::ExtractPerfStatsSection(output, gpuCodeHeader);

        if (perfStatsHeader == NULL) {
            return TEST_FAIL;
        }

        sectionData = ((const char*)output) + perfStatsHeader->common.dataOffset;

        // Perf stats section has a single magic number at the beginning
        // of the data section.
        success = success &&
            (((const uint32_t *)sectionData)[0] == GLSLC_PERF_STATS_SECTION_MAGIC_NUMBER);

        // Test the "magic" entry in the public struct corresponds to
        // the first 4 bytes.
        success = success &&
            (((const GLSLCperfStatsData *)sectionData)->magic ==
            ((const uint32_t *)sectionData)[0]);

    }

    return success ? TEST_PASS : TEST_FAIL;
}

// Colwenience defines for the byte offsets to various data within the application-opaque debug info section
// and debug info headers.
// Defines for DebugInfo section
#define DEBUG_INFO_OFFSET_NUM_SECTIONS 40
#define DEBUG_INFO_OFFSET_HEADERS_START 48
#define DEBUG_INFO_SIZE_HEADER 36

// Defines for DebugInfo subsection headers.
#define DEBUG_INFO_HEADER_OFFSET_DATA_OFFSET 4
#define DEBUG_INFO_HEADER_OFFSET_TYPE 8
#define DEBUG_INFO_HEADER_OFFSET_STAGE 16

// Magic number for the SASS debug sections.
#define SASS_DEBUG_SECTION_MAGIC_NUMBER 0xDEB700150000C0DEULL

// Colwenience define to extract a value of type <dataType> offset from <byteOffset> bytes
// in the block of memory pointed to be <bytePtr>
#define GET_VALUE_FROM_OFFSET(bytePtr, dataType, byteOffset) *((const dataType *)(&((reinterpret_cast<const uint8_t *>(bytePtr))[byteOffset])))

// Helper function to return the number of sections from an opaque DebugInfo data structure.
static int32_t GetDebugInfoNumSections(const uint8_t * debugInfoData) {
    return GET_VALUE_FROM_OFFSET(debugInfoData, int32_t, DEBUG_INFO_OFFSET_NUM_SECTIONS);
}

// Extracts a debug subsection of type <debugInfoSectionType> for stage <stage> from the debug info data structures.
// <stage> is only relevant when searching for subsections like GLASM / SASS info which have separate entries
// per stage.
// Returns NULL if we can't find any matches...
static const uint8_t * ExtractDebugSubSection(const uint8_t * debugInfo, DebugInfoSectionTypeEnum debugInfoSectionType, LWNshaderStage stage)
{
    const uint8_t * sectionHeader = NULL;
    int32_t numSections = GetDebugInfoNumSections(debugInfo);
    const uint8_t *headerStart = &debugInfo[DEBUG_INFO_OFFSET_HEADERS_START];

    assert(debugInfo);

    for (int32_t i = 0; i < numSections; ++i) {
        sectionHeader = &headerStart[DEBUG_INFO_SIZE_HEADER * i];
        int32_t headerType = GET_VALUE_FROM_OFFSET(sectionHeader, int32_t, DEBUG_INFO_HEADER_OFFSET_TYPE);

        // Check the stage-specific types for GLASM and SASS sections.  If the stage doesn't match the stage we are looking for,
        // just keep going.
        if (headerType == debugInfoSectionType) {
            if (debugInfoSectionType == DEBUG_INFO_SECTION_TYPE_ASM &&
                        GET_VALUE_FROM_OFFSET(sectionHeader, int32_t, DEBUG_INFO_HEADER_OFFSET_STAGE) != stage) {
                sectionHeader = NULL;
            } else if (debugInfoSectionType == DEBUG_INFO_SECTION_TYPE_GPU_CODE &&
                        GET_VALUE_FROM_OFFSET(sectionHeader, int32_t, DEBUG_INFO_HEADER_OFFSET_STAGE)) {
                sectionHeader = NULL;
            }

            if (sectionHeader != NULL) {
                // We found what we are looking for.
                break;
            }
        }
    }

    return sectionHeader;
}

// Get the debug section from the GLSLCoutput.  There should be only one.
static const GLSLCdebugInfoHeader * ExtractDebugSection(const GLSLCoutput * glslcOutput)
{
    const GLSLCdebugInfoHeader * outHeader = NULL;

    for (uint32_t i = 0; i < glslcOutput->numSections; ++i) {
        GLSLCsectionTypeEnum type = glslcOutput->headers[i].genericHeader.common.type;
        if (type == GLSLC_SECTION_TYPE_DEBUG_INFO) {
            outHeader = &glslcOutput->headers[i].debugInfoHeader;
            break;
        }
    }

    return outHeader;
}

// Check the data consistency of the GLASM debug section.
// We just check some common smoke checks of the data to make sure it exists and is not very corrupt.  This
// will not find subtle corruptions, but will help to find large-scale corruptions of the data.
static bool CheckAsmDebugSection(const uint8_t * debugInfoData, LWNshaderStage stage) {
    const uint8_t * debugInfoSectionHeader = ExtractDebugSubSection(debugInfoData, DEBUG_INFO_SECTION_TYPE_ASM, stage);

    if (!debugInfoSectionHeader) {
        log_output("No corresponding GLASM debug section for stage %d\n", stage);
        return false;
    }

    const char * glasmDebugInfo = ((const char *)debugInfoData) + GET_VALUE_FROM_OFFSET(debugInfoSectionHeader, uint32_t, DEBUG_INFO_HEADER_OFFSET_DATA_OFFSET);

    if (strlen(glasmDebugInfo) == 0) {
        log_output("GLASM debug section has zero length for stage %d\n", stage);
        return false;
    }

    if (strncmp("!!LW", glasmDebugInfo, 4)) {
        log_output("GLASM string does not begin with \"!!LW\" for stage %d\n", stage);
        return false;
    }

    // Search the GLASM debug info string for any oclwrrences of MSDB entries
    if (!strstr(glasmDebugInfo, "#MSDB")) {
        log_output("GLASM string does not have #MSDB for stage %d\n", stage);
        return false;
    }

    // Passed all checks
    return true;
}

// Perform simple smoke check of the SASS debug section.
static bool CheckSassDebugSection(const uint8_t * debugInfoData, LWNshaderStage stage) {
    const uint8_t * debugInfoSectionHeader = ExtractDebugSubSection(debugInfoData, DEBUG_INFO_SECTION_TYPE_GPU_CODE, stage);

    if (!debugInfoSectionHeader) {
        log_output("No corresponding GPU code debug section for stage %d\n", stage);
        return false;
    }

    const uint8_t * sassDebugInfo = (const uint8_t *)debugInfoData + GET_VALUE_FROM_OFFSET(debugInfoSectionHeader, uint32_t, DEBUG_INFO_HEADER_OFFSET_DATA_OFFSET);

    // Validate magic number (first element of the sass debug section).
    if (GET_VALUE_FROM_OFFSET(sassDebugInfo, uint64_t, 0) != SASS_DEBUG_SECTION_MAGIC_NUMBER) {
        log_output("Unexpected magic number as first entry in the SASS debug section for stage %d\n", stage);
        return false;
    }

    return true;
}

// A function to spot-check SASS dumping functionality.
static TestResult TestDebugSections(void)
{
    const char * shaders[2] = {
        vsstring,
        fsstring
    };

    LWNshaderStage stages[2] =
    {
        LWN_SHADER_STAGE_VERTEX,
        LWN_SHADER_STAGE_FRAGMENT
    };

    g_glslcHelper->SetDebugLevel(GLSLC_DEBUG_LEVEL_G2);

    if (!g_glslcHelper->CompileShaders(stages, 2, shaders)) {
        log_output("Failed to compile.\n");
        return TEST_FAIL;
    }

    const GLSLCoutput * output = g_glslcHelper->GetGlslcOutput();

    for (unsigned int i = 0; i < __GL_ARRAYSIZE(stages); ++i) {

        const GLSLCdebugInfoHeader * debugInfoHeader = ExtractDebugSection(output);

        if (debugInfoHeader == NULL) {
            log_output("debugInfoHeader is NULL.\n");
            return TEST_FAIL;
        }

        const uint8_t * debugInfoData = ((uint8_t *)output) + debugInfoHeader->common.dataOffset;

        if (GET_VALUE_FROM_OFFSET(debugInfoData, int32_t, 0) != GLSLC_DEBUG_SECTION_MAGIC_NUMBER) {
            log_output("debug info data does not begin with the debug info magic number.\n");
            return TEST_FAIL;
        }

        if (!CheckAsmDebugSection(debugInfoData, stages[i])) {
            log_output("GLASM debug section check failed.\n");
            return TEST_FAIL;
        }

        if (!CheckSassDebugSection(debugInfoData, stages[i])) {
            log_output("SASS debug section check failed.\n");
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

// A function to spot-check SASS dumping functionality.
static TestResult TestSassDump(void)
{
#if !defined(_MSC_VER)
    // SASS dumping is only enabled on Windows versions of GLSLC.
    return TEST_NOT_SUPPORT;
#endif

    const char * shaders[2] = {
        vsstring,
        fsstring
    };

    LWNshaderStage stages[2] =
    {
        LWN_SHADER_STAGE_VERTEX,
        LWN_SHADER_STAGE_FRAGMENT
    };

    g_glslcHelper->EnableSassDump(LWN_TRUE);

    // Enable CBF to test that the resulting SASS strings contain two programs.
    g_glslcHelper->EnableCBF(LWN_TRUE);

    if (!g_glslcHelper->CompileShaders(stages, 2, shaders)) {
        log_output("Failed to compile.\n");
        return TEST_FAIL;
    }

    const GLSLCoutput * output = g_glslcHelper->GetGlslcOutput();

    for (unsigned int i = 0; i < __GL_ARRAYSIZE(stages); ++i) {
        const GLSLCgpuCodeHeader * gpuCodeHeader =
            lwnTest::GLSLCHelper::ExtractGpuCodeSection(output, stages[i]);

        if (!gpuCodeHeader) {
            log_output("No corresponding GPU code section for stage %d\n", stages[i]);
            return TEST_FAIL;
        }

        // Get the Sass dump index.
        uint32_t sassDumpHeaderIndex = gpuCodeHeader->asmDumpSectionIdx;

        if (sassDumpHeaderIndex > output->numSections) {
            log_output("asmDumpSectionIdx for stage %d is greater than the total "
                       "number of output sections!\n", stages[i]);
        }

        const GLSLCasmDumpHeader * asmHeader =
            &output->headers[sassDumpHeaderIndex].asmDumpHeader;

        // Make sure the corresponding assembly dump section actual is an assembly dump.
        if (asmHeader->common.type != GLSLC_SECTION_TYPE_ASM_DUMP) {
            log_output("asmDumpSectionIdx(%d) for gpu code section stage %d "
                       "is not an assembly dump section!\n",
                sassDumpHeaderIndex, stages[i]);
            return TEST_FAIL;
        }

        // Make sure the stages correspond.
        if (asmHeader->stage != gpuCodeHeader->stage) {
            log_output("Assembly dump section (index %d) shader stage (%d) does not "
                       "match corresponding GPU code section's stage (%d)\n",
                sassDumpHeaderIndex, asmHeader->stage, gpuCodeHeader->stage);
            return TEST_FAIL;
        }

        // Get the actual SASS disassembly.
        const char *sassString = (const char *)output + asmHeader->common.dataOffset;

        // Make sure the string ends in the terminating NULL character.  Failure of this
        // check indicates an error in GLSLC's computation of the data size or possible
        // string corruption.
        if (sassString[asmHeader->common.size - 1] != '\0') {
            log_output("Assembly dump section (index %d) does not end in a terminating NULL character!\n",
                sassDumpHeaderIndex);
            return TEST_FAIL;
        }

        // Search for some known string in the SASS output, in this case the !!SPA5.3 string.
        // This is just meant as a spot-check as a proxy to determining if the SASS string is valid.
        // If the format of internal SASS disassembly ever change (should be ultra-rare), then this test
        // will need to be modified to account for that.

        // First make sure that !!SPA5.3 plus terminating NULL character can actually fit in the returned
        // sass string.
        unsigned int spaStringLength = 8;
        if (asmHeader->common.size < spaStringLength) {
            log_output("Assembly dump section (index %d) data size isn't large enough to even hold \"!!SPA5.3\"\n",
                sassDumpHeaderIndex);
            return TEST_FAIL;
        }

        const char * firstSPA = strstr(sassString, "!!SPA5.3");
        if (!firstSPA) {
            log_output("Assembly dump section (index %d) does not contain \"!!SPA5.3\" in the string!\n",
                sassDumpHeaderIndex);
            return TEST_FAIL;
        }

        // Check for vtxA/B pair for CBF SASS dumps.
        if (asmHeader->stage == LWN_SHADER_STAGE_VERTEX) {
            // Start from the end of the first "!!SPA5.3" and look for the second one.
            const char * secondSPA = strstr((firstSPA + 8), "!!SPA5.3");
            if (!secondSPA) {
                log_output("Assembly dump section (index %d) does not contain both vtxA and vtxB for CBF! "
                           "(looking for two \"!!SPA5.3\" in the string)\n", sassDumpHeaderIndex);
                return TEST_FAIL;
            }
        }

        // Check for illegal symbol(s) in the generated SASS string
        if (strstr(sassString, "?")) {
            log_output("Assembly dump section (index %d) contains illegal symbol(s) in the string!\n",
                sassDumpHeaderIndex);
            return TEST_FAIL;
        }
        // Check for illegal information in the generated SASS string
        if (strstr(sassString, "&")) {
            log_output("Assembly dump section (index %d) contains illegal information in the string!\n",
                sassDumpHeaderIndex);
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

// A function which tests these APIs:
//   uint8_t glslcCompareControlSections(const void* pControlSectionA, const void* pControlSectionB);
//   uint8_t glslcGetDebugDataHash(const void* pControlSection, GLSLCdebugDataHash* pDebugDataHash);
//   uint8_t glslcSetDebugDataHash(void* pControlSection, const GLSLCdebugDataHash* pDebugDataHash);
static TestResult TestDebugHashInsertion(void)
{
    const char * shaders[] = {
        vsstring,
        fsstring
    };

    LWNshaderStage stages[] =
    {
        LWN_SHADER_STAGE_VERTEX,
        LWN_SHADER_STAGE_FRAGMENT
    };

    // Set the debug level so our control binaries have a non-zero hash embedded.
    g_glslcHelper->SetDebugLevel(GLSLC_DEBUG_LEVEL_G2);
    LWNboolean compileSuccess = g_glslcHelper->CompileShaders(stages, 2, shaders);
    g_glslcHelper->SetDebugLevel(GLSLC_DEBUG_LEVEL_NONE);

    if (!compileSuccess) {
        log_output("Failed to compile.\n");
        return TEST_FAIL;
    }

    // Get the GLSLCoutput
    const GLSLCoutput * output = g_glslcHelper->GetGlslcOutput();

    const GLSLCgpuCodeHeader * gpuCodeHeader =
        lwnTest::GLSLCHelper::ExtractGpuCodeSection(output, stages[0]);

    uint32_t controlSize = gpuCodeHeader->controlSize;
    char * sectionData = ((char*)output + gpuCodeHeader->common.dataOffset);
    char * controlSection = sectionData + gpuCodeHeader->controlOffset;

    // Test that there is a valid debug hash that is non-zero.
    GLSLCdebugDataHash hashSection;
    memset(&hashSection, 0, sizeof(GLSLCdebugDataHash));
    if (!g_glslcLibraryHelper->glslcGetDebugDataHash(controlSection, &hashSection)) {
        log_output("glslcGetDebugDataHash returned false.");
        return TEST_FAIL;
    }

    if ((((uint64_t)hashSection.debugHashHi32 << 32) | ((uint64_t)hashSection.debugHashLo32)) == 0) {
        log_output("TestDebugHashInsertion failed: Debug hash recieved through glslcGetDebugDataHash was 0");
        return TEST_FAIL;
    }

    // Test that we can set a known hash value in the binary, then test that we get back that known
    // hash value when using the GLSLC API.
    uint64_t knownHashLo32 = 0x12344321;
    uint64_t knownHashHi32 = 0x98766789;

    // Set the hash to a known value.
    memset(&hashSection, 0, sizeof(GLSLCdebugDataHash));
    hashSection.debugHashLo32 = knownHashLo32;
    hashSection.debugHashHi32 = knownHashHi32;

    if (!g_glslcLibraryHelper->glslcSetDebugDataHash(controlSection, &hashSection)) {
        log_output("glslcSetDebugDataHash returned false.");
        return TEST_FAIL;
    }

    memset(&hashSection, 0, sizeof(GLSLCdebugDataHash));
    if (!g_glslcLibraryHelper->glslcGetDebugDataHash(controlSection, &hashSection)) {
        log_output("glslcGetDebugDataHash returned false.");
        return TEST_FAIL;
    }

    if ((hashSection.debugHashHi32 != knownHashHi32) ||
        (hashSection.debugHashLo32 != knownHashLo32)) {
        log_output("TestDebugHashInsertion failed: Was not able to set the hash in the control section.");
        return TEST_FAIL;
    }

    // Compare control sections using the GLSLC API.  We test a straight binary
    // copy (compare returns true), a binary copy with debug hash modified
    // (compare returns true since glslcCompareControlSections returns
    // functional equivalence), and then test that modifying a function byte
    // causes CompareControlSections to return false.

    char * controlSectionCopy = (char *)malloc(controlSize);

    if (!controlSectionCopy) {
        log_output("TestDebugHashInsertion failed: Could not allocate controlSectionCopy.");
        return TEST_FAIL;
    }

    memcpy(controlSectionCopy, controlSection, controlSize);

    uint8_t sectionsEqual = g_glslcLibraryHelper->glslcCompareControlSections(controlSection, controlSectionCopy);

    if (!sectionsEqual) {
        log_output("TestDebugHashInsertion failed: Identical control sections"
                    "don't return equal from glslcCompareControlSections.");
        free(controlSectionCopy);
        return TEST_FAIL;
    }

    // Artificially change the copy to have a different debug hash and retry the comparison - should return true again.
    memset(&hashSection, 0, sizeof(GLSLCdebugDataHash));
    hashSection.debugHashLo32 = knownHashLo32;
    hashSection.debugHashHi32 = knownHashHi32;
     g_glslcLibraryHelper->glslcSetDebugDataHash(controlSectionCopy, &hashSection);

    sectionsEqual = g_glslcLibraryHelper->glslcCompareControlSections(controlSection, controlSectionCopy);

    if (!sectionsEqual) {
        log_output("TestDebugHashInsertion failed: Control sections with different debug hash don't return equal from glslcCompareControlSections.");
        free(controlSectionCopy);
        return TEST_FAIL;
    }

    // Now modify some random byte and make sure the function returns false on the comparison.
    ((uint32_t *)controlSectionCopy)[10] = 0x12345678;

    sectionsEqual = g_glslcLibraryHelper->glslcCompareControlSections(controlSection, controlSectionCopy);

    if (sectionsEqual) {
        log_output("TestDebugHashInsertion failed: Control sections with "
                   "different byte return equal (should not) from "
                   "glslcCompareControlSections.");
        free(controlSectionCopy);
        return TEST_FAIL;
    }

    free(controlSectionCopy);
    return TEST_PASS;
}

extern int glslang;
extern int glslangFallbackOnError;
extern int glslangFallbackOnAbsolute;

#if TEST_GLSLANG
static TestResult TestBuiltinGlslang(void)
{
#if !defined(_WIN32)
    // This test is only supported on Windows versions of GLSLC.
    return TEST_NOT_SUPPORT;
#endif

    LWNboolean success = LWN_TRUE;

    LWNshaderStage stages[] =
    {
        LWN_SHADER_STAGE_FRAGMENT,
        LWN_SHADER_STAGE_VERTEX,
    };

    static const char *vsstring2 =
        "#version 440 core\n"
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = color;\n"
        "}\n";
    static const char *fsstring2 =
        "#version 440 core\n"
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";

    // Fail when compiling with glslang, but pass when falling back.
    const char * failingShaders[] = {
        fsstring,
        vsstring,
    };

    const char * passingShaders[] = {
        fsstring2,
        vsstring2,
    };

    struct TestCases {
        const char ** shaderSources;

        uint8_t doGlslangShim;
        uint8_t glslangFallbackOnError;
        uint8_t glslangFallbackOnAbsolute;

        LWNboolean expectedFinalSuccess;
        const char * expectedGlslangLog;
    } testCases[] = {
        { passingShaders, 1, 0, 0, LWN_TRUE,  "GLSLANG: Compile success.  Using resulting spirv to continue compilation."},
        // Pass if not using the built-in glslang.
        { failingShaders, 0, 0, 0, LWN_TRUE,  "No infolog available."},
        // Fail with glslang.
        { failingShaders, 1, 0, 0, LWN_FALSE, "GLSLANG: Failed to compile one or more shaders, halting compilation."},
        // Pass after falling back on error.
        { failingShaders, 1, 1, 0, LWN_TRUE,  "GLSLANG: Failed to compile one or more shaders.  Falling back to original GLSL input"},
        // Pass after falling back on absolute.
        { failingShaders, 1, 0, 1, LWN_TRUE,  "GLSLANG: Failed to compile one or more shaders.  Falling back to original GLSL input"},
    };

    int testNum = sizeof(testCases) / sizeof(testCases[0]);
    for (int i = 0; i < testNum; ++i) {
        log_output("Test id = %d\n", i);

        g_glslcHelper->OverrideGlslangOptions(testCases[i].doGlslangShim,
            testCases[i].glslangFallbackOnError, testCases[i].glslangFallbackOnAbsolute);
        success = success && g_glslcHelper->CompileShaders(stages, 2, testCases[i].shaderSources);
        // Restore to the default.
        g_glslcHelper->OverrideGlslangOptions(glslang, glslangFallbackOnError, glslangFallbackOnAbsolute);

        if (success != testCases[i].expectedFinalSuccess) {
            log_output("Test failed, expected result %d, but we got %d.\n", testCases[i].expectedFinalSuccess, success);
            log_output("Info log: \n%s\n", g_glslcHelper->GetInfoLog());
            return TEST_FAIL;
        }

        if (strstr(g_glslcHelper->GetInfoLog(), testCases[i].expectedGlslangLog) == NULL) {
            log_output("Test failed, didn't find the expected log: \n%s\n.\n", testCases[i].expectedGlslangLog);
            log_output("Info log: \n%s\n", g_glslcHelper->GetInfoLog());
            return TEST_FAIL;
        }

        // Skip the rest if it is a expected failure.
        if (!testCases[i].expectedFinalSuccess) {
            success = LWN_TRUE;
            continue;
        }

        success = success && VerifyGpuCodeSections(g_glslcHelper->GetGlslcOutput());
        if (!success) {
            log_output("Compile test: Verification failure\n");
            return TEST_FAIL;
        }
    }

    return success ? TEST_PASS : TEST_FAIL;
}
#endif //#if test_glslang

void LwnGlslcTest::setCellRect(CommandBuffer *queueCB, int testId) const
{
    int cx = testId % cellsX;
    int cy = testId / cellsX;

    queueCB->SetViewport(cx * (cellSize + cellMargin), cy * (cellSize + cellMargin),
        cellSize, cellSize);
    queueCB->SetScissor(cx * (cellSize + cellMargin), cy * (cellSize + cellMargin),
        cellSize, cellSize);
}

int LwnGlslcTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(5,0);
}

lwString LwnGlslcTest::getDescription() const
{
    return
        "Tests the GLSLC output interface.\n"
        "This test performs no rendering (that is done by other\n"
        "LWN tests), but instead tests compilation functionality\n"
        "and the consistency of the GLSLC output structures\n";
}

typedef TestResult (*TEST_FUNCTION_TYPE)(void);

// Add tests here!!
// This array contains function pointers for each test.  As functions are
// added, the output fail/pass squares will automatically adjust.
// To add a test, create the function with the interface TEST_FUNCTION_TYPE,
// and add the entry to the blow array.
static TEST_FUNCTION_TYPE testFunctions [] =
{
    TestCompile,
    TestMagicNumbers,
    TestPerfStats,
    TestMatchingOutputs,
    TestOverloadedResourceBindings,
    TestEntriesThatDontModifyBinaryOutput,
    TestSassDump,
    TestDebugHashInsertion,
    TestDebugSections,
    TestShaderStageDefines
#if TEST_GLSLANG
    , TestBuiltinGlslang
#endif
};

void LwnGlslcTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

    unsigned int numTests = __GL_ARRAYSIZE(testFunctions);
    assert(numTests <= cellsX * cellsY);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);

    for (unsigned int i = 0; i < numTests; i++) {
        // Call the test function.
        TestResult res = testFunctions[i]();

        setCellRect(queueCB, i);

        // Clears the cell based on pass/fail from the test function.
        if (res == TEST_PASS) {
            queueCB.ClearColor(0, 0.0, 1.0, 0.0);
        } else if (res == TEST_FAIL) {
            queueCB.ClearColor(0, 1.0, 0.0, 0.0);
        } else {
            // TEST_NOT_SUPPORT
            queueCB.ClearColor(0, 0.0, 0.0, 1.0);
        }

        // Reset for the next test.
        g_glslcHelper->Reset();
    }

    queueCB.submit();
}


OGTEST_CppTest(LwnGlslcTest, lwn_glslc, );

