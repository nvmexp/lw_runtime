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

// Macros used to verify that data in SSBO written to by the application can also be seen in the shader.
#define CHECK_INITIALIZED_INDEX 12
#define CHECK_INITIALIZED_X 2
#define CHECK_INITIALIZED_Y 3
#define CHECK_INITIALIZED_Z 4
#define CHECK_INITIALIZED_W 5
class LWNSSBOTest
{
public:
    enum StorageMode {
        SSBOStorage,
        GlobalStorage,
    };

private:
    static const int ctaWidth = 8;
    static const int ctaHeight = 8;
    ShaderStage m_stage;
    StorageMode m_storage;
    bool isCompute() const  { return m_stage == ShaderStage::COMPUTE; }
    bool isBound() const    { return m_storage == SSBOStorage; }
    void showBuffer(QueueCommandBuffer &queueCB, Program *program, Buffer *buffer, Texture *texture,
                    int ssboWidth, int ssboHeight) const;

    // Function to check to add code to the shader (pointed to by <assignToString>) to check the results
    // of a .length() GLSL function call on one or more SSBO variables.
    void CheckExpectedLength(bool isBound, int numSSBOs, const char * ssboVarNames[], const int * expectedValueLengths,
                             int expectedValuesFailBorder, const char * assignToString, lwShader * shader) const;
public:
    LWNSSBOTest(ShaderStage stage, StorageMode storage) : m_stage(stage), m_storage(storage) {}
    LWNTEST_CppMethods();
};

lwString LWNSSBOTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic test of SSBO (shader storage buffer object) functionality in LWN, using " <<
        (isBound() ? "bound" : "bindless") << " accesses in " << (isCompute() ? "compute" : "graphics") <<
        "shaders.  This test renders a Gouraud shaded rectangle in four quadrants, each of "
        "which has a white inner border and a black outer border.  The quadrants "
        "are produced by the following process:\n\n"
        "* LOWER_LEFT:  Render a quad, storing per-fragment colors in an SSBO using (x,y)\n"
        "* LOWER_RIGHT:  Render another quad, loading per-fragment colors from "
        "the previous SSBO using (x,y)\n"
        "* UPPER_LEFT:  Use the SSBO as the source of a copy to texture, then render "
        "a texture mapped quad\n"
        "* UPPER_RIGHT:  Render a quad, where each (x,y) uses an atomic to get a unique "
        "offset in the SSBO to store its (x,y) and color.  Then render a cloud of points "
        "each using vertex ID (and no attributes) to fetch (x,y) and color.\n\n"
        "At the end of the test, we check the atomic value against an expected value and "
        "draw a red overlay on the UPPER_RIGHT cell if it doesn't match.\n";
    if (isBound()) {
        sb << "For uniform SSBOs (non-bindless), .length() is tested on the unsized arrays "
              "in the SSBOs.  A failure for .length() to produce the expected result will draw "
              "a red border around the the test.\n";
    }
    return sb.str();
}

int LWNSSBOTest::isSupported() const
{
    if (isCompute()) {
        return lwogCheckLWNAPIVersion(21, 4);
    } else {
        return lwogCheckLWNAPIVersion(21, 3);
    }
}

void LWNSSBOTest::showBuffer(QueueCommandBuffer &queueCB, Program *program, Buffer *buffer, Texture *texture,
                             int ssboWidth, int ssboHeight) const
{
    CopyRegion copyRegion = { 0, 0, 0, ssboWidth, ssboHeight, 1 };
    queueCB.CopyBufferToTexture(buffer->GetAddress(), texture, NULL, &copyRegion, CopyFlags::NONE);
    queueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
}

void LWNSSBOTest::CheckExpectedLength(bool isBound, int numSSBOs, const char * ssboVarNames[], const int * expectedValueLengths,
                                int expectedValuesFailBorder, const char * assignToString, lwShader * shader) const
{
    // Only perform this check for non-bindless SSBOs.
    if (!isBound) {
        return;
    }

    *shader << "  if (";

    for (int i = 0; i < numSSBOs; ++i) {
        *shader << "(" << (ssboVarNames[i]) << ".length() != " << expectedValueLengths[i] << ") &&";
    }

    *shader << " (edgedist < " << expectedValuesFailBorder << ")) \n"
        << " { " << assignToString << " }\n";
}

void LWNSSBOTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // We program our SSBO to hold 1/4 of the full window size rounded down to
    // multiples of compute CTA size units.  In the point cloud case, we'll
    // store an (x,y) and (R,G,B,A) per vertex -- padded out to two vec4's per
    // vertex plus a single counter value at the beginning also padded to a
    // vec4.
    int ssboWidth = ctaWidth * (lwrrentWindowWidth / (2 * ctaWidth));
    int ssboHeight = ctaHeight * (lwrrentWindowHeight / (2 * ctaHeight));
    int ssboSize = (ssboWidth * ssboHeight * 2 + 1) * sizeof(dt::vec4);
    ssboSize = (ssboSize + 0x1F) & ~(0x1F);

    // Names of the common SSBO variables used when testing .length().  One or
    // both of these might be used.
    const char *expectedLengthVarNames[2] = {
        "values",
        "values2"
    };

    // Expected values of the .length() command on the 2 types of SSBO variables we use.
    int expectedLengthValues[2] = {
        // For the first type of SSBO, the only field in the ssbo is "values" which is a vec4, so the expected
        // length will be ssboSize / sizeof(vec4).  Using std430, sizeof(vec4) is expected to be 16.
        (ssboSize / 16),

        // For the second type of SSBO, the ssbo contains a counter (4 bytes) and an array "values" of a Point struct type.
        // The struct Point's size is 32-bytes according to the std430 rules.
        ((ssboSize - 4) / 32)
    };

    // The border width to draw red in the case of ".length()" failures.
    int expectedValuesFailBorder = 16;

    // Code fragment to compute a base color from the compute shader (x,y); we
    // don't have access to interpolation.
    static const char *csComputeBaseColorCode =
        "  color.x = (float(xy.x) + 0.5) / float(viewport.z);\n"
        "  color.y = (float(xy.y) + 0.5) / float(viewport.w);\n"
        "  color.z = (float(xy.x + xy.y) + 1.0) / float(viewport.z + viewport.w);\n"
        "  color.w = 1.0;\n";

    // Code fragment to compute the maximum distance from an (x,y) coordinate
    // to the edge.
    static const char *edgeComputeDistCode =
        "  ivec2 edgedist2 = min(xy, (viewport.zw - 1) - xy);\n"
        "  int edgedist = min(edgedist2.x, edgedist2.y);\n";

    // Code fragment to discard helper pixels and (x,y) coordinates outside
    // the viewport.
    static const char *edgeCheckCode =
        "  if (gl_HelperIlwocation || edgedist < 0) {\n"
        "    discard;\n"
        "  }\n";

    // Code fragment to set a black or white color on fragments near the edge
    // of the viewport, using the distance computed in fsEdgeCheck.
    static const char *edgeAdjustColorCode =
        "  if (edgedist < 8) {\n"
        "    color = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "    if (edgedist < 4) {\n"
        "      color = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "    }\n"
        "  }\n";



    // Common vertex shader for drawing a big pass-through quad.  We also
    // construct a set of texture coordinates (otc) using position.
    VertexShader vs(450);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "out vec2 otc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = color;\n"
        "  otc = 0.5 * position.xy - 0.5;\n"
        "}\n";

    // Shader used to store the fragment color to an SSBO using an index
    // computed from a viewport-relative (x,y).
    FragmentShader ssboStoreFS(450);
    ssboStoreFS <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n";
    if (isBound()) {
        ssboStoreFS << 
            "layout(std430, binding = 0) buffer SSBO {\n"
            "  vec4 values[];\n"
            "};\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        ssboStoreFS.addExtension(lwShaderExtension::LW_gpu_shader5);
        ssboStoreFS <<

            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  vec4 *values;\n"
            "};\n";
    }

    ssboStoreFS <<
        "void main() {\n"
        "  vec4 color = vec4(ocolor, 1.0);\n"
        "  ivec2 xy = ivec2(gl_FragCoord.xy) - viewport.xy;\n" <<
        edgeComputeDistCode <<
        edgeCheckCode <<
        edgeAdjustColorCode <<
        "  int index = xy.y * viewport.z /* width */ + xy.x;\n"
        "  values[index] = color / 2.0;\n";
        CheckExpectedLength(isBound(), 1, expectedLengthVarNames, expectedLengthValues,
                        expectedValuesFailBorder,
                        "values[index] = vec4(1.0, 0.0, 0.0, 1.0); \n"
                        "color = vec4(1.0, 0.0, 0.0, 1.0); \n",
                        &ssboStoreFS);
        ssboStoreFS << "  fcolor = color;\n"
        "}\n";

    ComputeShader ssboStoreCS(450);
    ssboStoreCS.setCSGroupSize(ctaWidth, ctaHeight);
    if (isBound()) {
        ssboStoreCS <<
            "layout(std430, binding = 0) buffer SSBO {\n"
            "  vec4 values[];\n"
            "};\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        ssboStoreCS.addExtension(lwShaderExtension::LW_gpu_shader5);
        ssboStoreCS <<

            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  vec4 *values;\n"
            "};\n";
    }

    ssboStoreCS <<
        "void main() {\n"
        "  vec4 color;\n"
        "  ivec2 xy = ivec2(gl_GlobalIlwocationID.xy);\n" <<
        csComputeBaseColorCode <<
        edgeComputeDistCode <<
        edgeAdjustColorCode <<
        "  int index = xy.y * viewport.z /* width */ + xy.x;\n"
        // Do a verification check that a SSBO dummy value written by the application can be read by the shader.  This is a
        // regression check for bug 200255431.  This value will get overwritten immediately by the store results.
        "  if ((index == " << CHECK_INITIALIZED_INDEX << " &&\n"
        "    values[" << CHECK_INITIALIZED_INDEX << "].x == " << CHECK_INITIALIZED_X << " && \n"
        "    values[" << CHECK_INITIALIZED_INDEX << "].y == " << CHECK_INITIALIZED_Y << " && \n "
        "    values[" << CHECK_INITIALIZED_INDEX << "].z == " << CHECK_INITIALIZED_Z << ")\n"
        "|| (index != " << CHECK_INITIALIZED_INDEX << ")) {\n "
        "  values[index] = color / 2.0;\n"
        "}\n";
        CheckExpectedLength(isBound(), 1, expectedLengthVarNames, expectedLengthValues,
                        expectedValuesFailBorder,
                        "values[index] = vec4(1.0, 0.0, 0.0, 1.0);",
                        &ssboStoreCS);
        ssboStoreCS << "}\n";

    // Shader used to load a final fragment color from a SSBO using an index
    // computed from a viewport-relative (x,y).
    FragmentShader ssboLoadFS(450);
    ssboLoadFS <<
        "out vec4 fcolor;\n";
    if (isBound()) {
        ssboLoadFS <<
            "layout(std430, binding = 0) buffer SSBO {\n"
            "  vec4 values[];\n"
            "};\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        ssboLoadFS.addExtension(lwShaderExtension::LW_gpu_shader5);
        ssboLoadFS <<

            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  vec4 *values;\n"
            "};\n";
    }
    ssboLoadFS <<
        "void main() {\n"
        "  ivec2 xy = ivec2(gl_FragCoord.xy) - viewport.xy;\n" <<
        edgeComputeDistCode <<
        edgeCheckCode <<
        "  int index = xy.y * viewport.z /* width */ + xy.x;\n" <<
        "  fcolor = values[index] * 2.0;\n";
        CheckExpectedLength(isBound(), 1, expectedLengthVarNames, expectedLengthValues,
                        expectedValuesFailBorder,
                        "fcolor = vec4(1.0, 0.0, 0.0, 1.0);",
                        &ssboLoadFS);
        ssboLoadFS << "}\n";

    ComputeShader ssboLoadCS(450);
    ssboLoadCS.setCSGroupSize(ctaWidth, ctaHeight);
    if (isBound()) {
        ssboLoadCS <<
            "layout(std430, binding = 0) buffer SSBO {\n"
            "  vec4 values[];\n"
            "};\n"
            "layout(std430, binding = 1) buffer SSBO2 {\n"
            "  vec4 values2[];\n"
            "};\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        ssboLoadCS.addExtension(lwShaderExtension::LW_gpu_shader5);
        ssboLoadCS <<

            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  vec4 *values, *values2;\n"
            "};\n";
    }
    ssboLoadCS <<
        "void main() {\n"
        "  ivec2 xy = ivec2(gl_GlobalIlwocationID.xy);\n" <<
        edgeComputeDistCode <<
        "  int index = xy.y * viewport.z /* width */ + xy.x;\n"
        "  if (index < viewport.z * viewport.w) {\n"
        "    " << (isBound() ? "values2" : "values2") << "[index] = values[index] * 2.0 / 3.0;\n";

    // This shader uses the same length for both values and values2.
    int ssboLoadCSExpectedValueLengths[2] = {
        expectedLengthValues[0],
        expectedLengthValues[0]
    };

    CheckExpectedLength(isBound(), 2, expectedLengthVarNames, ssboLoadCSExpectedValueLengths,
                    expectedValuesFailBorder,
                    "values2[index] = vec4(1.0, 0.0, 0.0, 1.0);",
                    &ssboLoadCS);

    ssboLoadCS << "  }\n"
        "}\n";

    // Shader used to display a texture initialized from SSBO contents.
    FragmentShader texDisplay2FS(450);
    texDisplay2FS <<
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "layout(binding=0) uniform sampler2D tex;\n"
        "void main() {\n"
        "  fcolor = texture(tex, otc) * 2.0;\n"
        "}\n";

    FragmentShader texDisplay3FS(450);
    texDisplay3FS <<
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "layout(binding=0) uniform sampler2D tex;\n"
        "void main() {\n"
        "  fcolor = texture(tex, otc) * 3.0;\n"
        "}\n";

    FragmentShader texDisplay4FS(450);
    texDisplay4FS <<
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "layout(binding=0) uniform sampler2D tex;\n"
        "void main() {\n"
        "  fcolor = texture(tex, otc) * 4.0;\n"
        "}\n";

    // Shader used to store a fragment's (x,y) and color to an SSBO using a
    // unique index derived from an atomic.
    FragmentShader ssboCloudStoreFS(450);
    ssboCloudStoreFS <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "struct Point {\n"
        "  ivec2 fragcoord;\n"
        "  vec4 color;\n"
        "};\n";
    if (isBound()) {
        ssboCloudStoreFS <<
            "layout(std430, binding = 0) buffer SSBO {\n"
            "  uint  counter;\n"
            "  Point values[];\n"
            "};\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        ssboCloudStoreFS.addExtension(lwShaderExtension::LW_gpu_shader5);
        ssboCloudStoreFS <<

            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  uint *ssbo;\n"
            "};\n";
    }
    ssboCloudStoreFS <<
        "void main() {\n" <<
        (isBound() ? "" : "  Point *values = (Point *)(ssbo + 4); \n") <<
        "  vec4 color = vec4(ocolor, 1.0);\n"
        "  ivec2 xy = ivec2(gl_FragCoord.xy) - viewport.xy;\n" <<
        edgeComputeDistCode <<
        edgeCheckCode <<
        edgeAdjustColorCode;
    if (isBound()) {
        ssboCloudStoreFS << "  uint index = atomicAdd(counter, 1);\n";
    } else {
        ssboCloudStoreFS << "  uint index = atomicAdd(ssbo, 1);\n";
    }
    ssboCloudStoreFS <<
        "  if (index < viewport.z * viewport.w) {\n"
        "    values[index].fragcoord = xy;\n"
        "    values[index].color = 4.0 * color;\n"
        "  }\n"
        "  fcolor = vec4(1.0, 0.0, 0.0, 1.0);  // draw red (should be overwritten)\n";
    CheckExpectedLength(isBound(), 1, expectedLengthVarNames, &expectedLengthValues[1],
                    expectedValuesFailBorder,
                    "values[index].color = vec4(1.0, 0.0, 0.0, 1.0);",
                    &ssboCloudStoreFS);
    ssboCloudStoreFS << "}\n";

    ComputeShader ssboCloudStoreCS(450);
    ssboCloudStoreCS.setCSGroupSize(ctaWidth, ctaHeight);
    ssboCloudStoreCS <<
        "struct Point {\n"
        "  ivec2 fragcoord;\n"
        "  vec4 color;\n"
        "};\n";
    if (isBound()) {
        ssboCloudStoreCS <<
            "layout(std430, binding = 0) buffer SSBO {\n"
            "  uint  counter;\n"
            "  Point values[];\n"
            "};\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        ssboCloudStoreCS.addExtension(lwShaderExtension::LW_gpu_shader5);
        ssboCloudStoreCS <<

            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  uint *ssbo;\n"
            "};\n";
    }
    ssboCloudStoreCS <<
        "void main() {\n" <<
        (isBound() ? "" : "  Point *values = (Point *)(ssbo + 4); \n") <<
        "  vec4 color;\n"
        "  ivec2 xy = ivec2(gl_GlobalIlwocationID.xy);\n" <<
        csComputeBaseColorCode <<
        edgeComputeDistCode <<
        edgeAdjustColorCode;
    if (isBound()) {
        ssboCloudStoreCS << "  uint index = atomicAdd(counter, 1);\n";
    } else {
        ssboCloudStoreCS << "  uint index = atomicAdd(ssbo, 1);\n";
    }
    ssboCloudStoreCS <<
        "  if (index < viewport.z * viewport.w) {\n"
        "    values[index].fragcoord = xy;\n"
        "    values[index].color = 4.0 * color;\n"
        "  }\n";
    CheckExpectedLength(isBound(), 1, expectedLengthVarNames, &expectedLengthValues[1],
                    expectedValuesFailBorder,
                    "values[index].color = vec4(1.0, 0.0, 0.0, 1.0);",
                    &ssboCloudStoreCS);
    ssboCloudStoreCS << "}\n";

    // Shader used to load a fragment's (x,y) and color from an SSBO using
    // gl_VertexID.
    VertexShader ssboCloudLoadVS(450);
    ssboCloudLoadVS <<
        "struct Point {\n"
        "  ivec2 fragcoord;\n"
        "  vec4 color;\n"
        "};\n";
    if (isBound()) {
        ssboCloudLoadVS <<
            "layout(std430, binding = 0) buffer SSBO {\n"
            "  uint counter;\n"
            "  Point values[];\n"
            "};\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        ssboCloudLoadVS.addExtension(lwShaderExtension::LW_gpu_shader5);
        ssboCloudLoadVS <<

            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  uint *ssbo;\n"
            "};\n";
    }
    ssboCloudLoadVS <<
        "out vec4 ocolor;\n"
        "void main() {\n" << 
        (isBound() ? "" : "  Point *values = (Point *)(ssbo + 4); \n") <<
        "  vec2 fragcoord = vec2(values[gl_VertexID].fragcoord);\n"
        "  vec2 viewsize = vec2(viewport.zw);\n"
        "  gl_Position.xy = 2.0 * ((fragcoord + vec2(0.5)) / viewsize) - 1.0;\n"
        "  gl_Position.zw = vec2(0.0, 1.0);\n"
        "  ocolor = values[gl_VertexID].color / 4.0;\n";
    ssboCloudLoadVS << "  ivec2 xy = ivec2(fragcoord.xy);\n" << edgeComputeDistCode;
    CheckExpectedLength(isBound(), 1, expectedLengthVarNames, &expectedLengthValues[1],
                    expectedValuesFailBorder,
                    "ocolor = vec4(1.0, 0.0, 0.0, 1.0);",
                    &ssboCloudLoadVS);

    ssboCloudLoadVS << "}\n";

    // Displays the color loaded from the vertex shader.
    FragmentShader ssboCloudLoadFS(450);
    ssboCloudLoadFS <<
        "in vec4 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = ocolor;\n"
        "}\n";

    ComputeShader ssboCloudLoadCS(450);
    ssboCloudLoadCS.setCSGroupSize(ctaWidth, ctaHeight);
    ssboCloudLoadCS <<
        "struct Point {\n"
        "  ivec2 fragcoord;\n"
        "  vec4 color;\n"
        "};\n";
    if (isBound()) {
        ssboCloudLoadCS <<
            "layout(std430, binding = 0) buffer SSBO {\n"
            "  uint counter;\n"
            "  Point values[];\n"
            "};\n"
            "layout(std430, binding = 1) buffer SSBO2 {\n"
            "  vec4 colors[];\n"
            "};\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        ssboCloudLoadCS.addExtension(lwShaderExtension::LW_gpu_shader5);
        ssboCloudLoadCS <<

            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  uint *ssbo;\n"
            "  vec4 *colors;\n"
            "};\n";
    }
    ssboCloudLoadCS <<
        "void main() {\n" << 
        (isBound() ? "" : "  Point *values = (Point *)(ssbo + 4); \n") <<
        "  uvec2 iid = gl_GlobalIlwocationID.xy;\n"
        "  uint indexIn = iid.y * viewport.z /* width */ + iid.x;\n"
        "  ivec2 xy = values[indexIn].fragcoord;\n"
        "  vec4 color = values[indexIn].color;\n"
        "  int indexOut = xy.y * viewport.z /* width */ + xy.x;\n"
        "  if (indexOut < viewport.z * viewport.w) {\n"
        "    colors[indexOut] = color / 16.0;\n"
        "  }\n";
        ssboCloudLoadCS << edgeComputeDistCode;

        const char *ssboCloudLoadCSVarNames[2] = {
            expectedLengthVarNames[0],
            "colors"
        };

        int ssboCloudLoadCSExpectedValues[2] = {
            expectedLengthValues[1],
            expectedLengthValues[0]
        };

        CheckExpectedLength(isBound(), 2, ssboCloudLoadCSVarNames, ssboCloudLoadCSExpectedValues,
                        expectedValuesFailBorder,
                        "colors[indexOut] = vec4(1.0, 0.0, 0.0, 1.0);",
                        &ssboCloudLoadCS);

        ssboCloudLoadCS << "}\n";

    // Load up all the programs.
    enum TestProgramType {
        SSBOStore,
        SSBOLoad,
        SSBOCloudStore,
        SSBOCloudLoad,
        DisplayTex2,
        DisplayTex3,
        DisplayTex4,
        TestProgramCount
    };
    struct TestProgramConfig {
        VertexShader *vs;
        FragmentShader *fs;
        ComputeShader *cs;
    };
    TestProgramConfig programConfigs[TestProgramCount] = {
        { &vs, &ssboStoreFS, &ssboStoreCS },
        { &vs, &ssboLoadFS, &ssboLoadCS },
        { &vs, &ssboCloudStoreFS, &ssboCloudStoreCS },
        { &ssboCloudLoadVS, &ssboCloudLoadFS, &ssboCloudLoadCS },
        { &vs, &texDisplay2FS, NULL },
        { &vs, &texDisplay3FS, NULL },
        { &vs, &texDisplay4FS, NULL },
    };
    Program *programs[TestProgramCount];
    for (int i = 0; i < TestProgramCount; i++) {
        programs[i] = device->CreateProgram();
        ComputeShader *cs = programConfigs[i].cs;
        if (isCompute() && cs) {
            if (!g_glslcHelper->CompileAndSetShaders(programs[i], *cs)) {
                LWNFailTest();
                return;
            }
        } else {
            VertexShader *vs = programConfigs[i].vs;
            FragmentShader *fs = programConfigs[i].fs;
            if (!g_glslcHelper->CompileAndSetShaders(programs[i], *vs, *fs)) {
                LWNFailTest();
                return;
            }
        }
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-1.0, +1.0, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(+1.0, +1.0, 0.0), dt::vec3(1.0, 1.0, 1.0) },
    };
    MemoryPoolAllocator allocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device,  4, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    MemoryPoolAllocator ssboAllocator(device, NULL, 2 * ssboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ssbo = ssboAllocator.allocBuffer(&bb, BUFFER_ALIGN_SHADER_STORAGE_BIT, ssboSize);
    BufferAddress ssboAddr = ssbo->GetAddress();
    // We initialize an entry of the SSBO with a known value so that we can verify the shader can read from this SSBO.  This is part of regression
    // checking for bug 200255431.
    dt::vec4 *ssboMem = (dt::vec4 *) ssbo->Map();
    ssboMem[CHECK_INITIALIZED_INDEX] = dt::vec4(CHECK_INITIALIZED_X, CHECK_INITIALIZED_Y, CHECK_INITIALIZED_Z, CHECK_INITIALIZED_W);
    Buffer *ssbo2 = ssboAllocator.allocBuffer(&bb, BUFFER_ALIGN_SHADER_STORAGE_BIT, ssboSize);
    BufferAddress ssbo2Addr = ssbo2->GetAddress();

    // We program four separate sysmem UBO copies, one for each quadrant, and
    // fill each with a single ivec4 holding (x,y,w,h).
    int uboVersions = 4;
    LWNint uboAlignment;
    device->GetInteger(DeviceInfo::UNIFORM_BUFFER_ALIGNMENT, &uboAlignment);
    int uboSize = uboAlignment;
    if (uboSize < LWNint(sizeof(dt::ivec4))) {
        uboSize = sizeof(dt::ivec4);
    }
    MemoryPoolAllocator uboAllocator(device, NULL, uboVersions * uboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo = uboAllocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, uboVersions * uboSize);
    BufferAddress uboAddr = ubo->GetAddress();
    char *uboMem = (char *) ubo->Map();
    struct UBOLayout {
        dt::ivec4       viewport;
        BufferAddress   ssbo, ssbo2;
    };
    for (int i = 0; i < 4; i++) {
        UBOLayout *data = (UBOLayout *)(uboMem + i * uboSize);
        data->viewport = dt::ivec4((i & 1) ? ssboWidth : 0, (i & 2) ? ssboHeight : 0,
                                   ssboWidth, ssboHeight);
        data->ssbo = ssboAddr;
        data->ssbo2 = ssbo2Addr;
    }

    // Set up a texture and sampler for rendering using a copy of the SSBO
    // contents.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_2D).
        SetSize2D(ssboWidth, ssboHeight).SetFormat(Format::RGBA32F).SetLevels(1);
    LWNsizeiptr texSize = tb.GetStorageSize();
    MemoryPoolAllocator texAllocator(device, NULL, texSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    Texture *tex = texAllocator.allocTexture(&tb);

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);
    Sampler *sampler = sb.CreateSampler();

    TextureHandle texHandle = device->GetTextureHandle(tex->GetRegisteredTextureID(), sampler->GetRegisteredID());

    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    if (isBound()) {
	    queueCB.BindStorageBuffer(ShaderStage::FRAGMENT, 0, ssboAddr, ssboSize);
	    queueCB.BindStorageBuffer(ShaderStage::VERTEX, 0, ssboAddr, ssboSize);
    }
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    if (isCompute()) {
        if (isBound()) {
	        queueCB.BindStorageBuffer(ShaderStage::COMPUTE, 0, ssboAddr, ssboSize);
	        queueCB.BindStorageBuffer(ShaderStage::COMPUTE, 1, ssbo2Addr, ssboSize);
        }
        queueCB.BindTexture(ShaderStage::COMPUTE, 0, texHandle);
    }

    ShaderStageBits programBindMask = ShaderStageBits::ALL_GRAPHICS_BITS;
    if (isCompute()) {
        programBindMask = ShaderStageBits::COMPUTE;
    }

    // Render the store-to-SSBO pass in the lower left quadrant.
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr + 0 * uboSize, uboSize);
    queueCB.SetViewportScissor(0, 0, ssboWidth, ssboHeight);
    queueCB.BindProgram(programs[SSBOStore], programBindMask);
    if (isCompute()) {
        queueCB.BindUniformBuffer(ShaderStage::COMPUTE, 0, uboAddr + 0 * uboSize, uboSize);
        queueCB.DispatchCompute(ssboWidth / ctaWidth, ssboHeight / ctaHeight, 1);
        showBuffer(queueCB, programs[DisplayTex2], ssbo, tex, ssboWidth, ssboHeight);
    } else {
        queueCB.BindProgram(programs[SSBOStore], ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    }

    // Insert a barrier to make sure the store is fully done before the
    // subsequent load pass.
    queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_SHADER | BarrierBits::ILWALIDATE_TEXTURE);

    // Render the load-from-SSBO pass in the lower right quadrant.
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr + 1 * uboSize, uboSize);
    queueCB.SetViewportScissor(ssboWidth, 0, ssboWidth, ssboHeight);
    queueCB.BindProgram(programs[SSBOLoad], programBindMask);
    if (isCompute()) {
        queueCB.BindUniformBuffer(ShaderStage::COMPUTE, 0, uboAddr + 1 * uboSize, uboSize);
        queueCB.DispatchCompute(ssboWidth / ctaWidth, ssboHeight / ctaHeight, 1);
        showBuffer(queueCB, programs[DisplayTex3], ssbo2, tex, ssboWidth, ssboHeight);
    } else {
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    }

    // Copy from the SSBO to a texture and render with the texture in the
    // upper left.
    queueCB.SetViewportScissor(0, ssboHeight, ssboWidth, ssboHeight);
    showBuffer(queueCB, programs[DisplayTex2], ssbo, tex, ssboWidth, ssboHeight);

    // Now generate data for the the point cloud pass.  CopyBuffer here
    // initializes the atomic counter to zero (by copying from the zero at the
    // beginning of our first UBO).
    queueCB.CopyBufferToBuffer(uboAddr, ssboAddr, sizeof(LWNuint), CopyFlags::NONE);
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr + 3 * uboSize, uboSize);
    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr + 3 * uboSize, uboSize);
    queueCB.SetViewportScissor(ssboWidth, ssboHeight, ssboWidth, ssboHeight);
    queueCB.BindProgram(programs[SSBOCloudStore], programBindMask);
    if (isCompute()) {
        queueCB.BindUniformBuffer(ShaderStage::COMPUTE, 0, uboAddr + 3 * uboSize, uboSize);
        queueCB.DispatchCompute(ssboWidth / ctaWidth, ssboHeight / ctaHeight, 1);
    } else {
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    }

    // Wait for the point cloud data to be ready.
    queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_SHADER | BarrierBits::ILWALIDATE_TEXTURE);

    // Render the point cloud.
    queueCB.BindProgram(programs[SSBOCloudLoad], programBindMask);
    if (isCompute()) {
        queueCB.DispatchCompute(ssboWidth / ctaWidth, ssboHeight / ctaHeight, 1);
        showBuffer(queueCB, programs[DisplayTex4], ssbo2, tex, ssboWidth, ssboHeight);
    } else {
        // Set up "null" vertex state for the point cloud pass, which uses no
        // attributes other than gl_VertexID.
        queueCB.BindVertexAttribState(0, NULL);
        queueCB.BindVertexStreamState(0, NULL);
        queueCB.DrawArrays(DrawPrimitive::POINTS, 0, ssboWidth * ssboHeight);
    }

    // When finished, copy the first word of the SSBO (the counter) back to
    // our UBO memory and check with the CPU.  If it doesn't match, clear a
    // red rectangle on top of our upper right quadrant.
    queueCB.CopyBufferToBuffer(ssboAddr, uboAddr, sizeof(LWNuint), CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();
    if (*((LWNuint *) uboMem) != LWNuint(ssboWidth * ssboHeight)) {
#if 0
        printf("Count is %d, expected %d\n", *((LWNuint *) uboMem), ssboWidth * ssboHeight);
#endif
        queueCB.SetViewportScissor(3 * ssboWidth / 2 - 16, 3 * ssboHeight / 2 - 16, 32, 32);
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
        queueCB.submit();
    }

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNSSBOTest, lwn_ssbo_basic, (ShaderStage::FRAGMENT, LWNSSBOTest::SSBOStorage));
OGTEST_CppTest(LWNSSBOTest, lwn_ssbo_compute, (ShaderStage::COMPUTE, LWNSSBOTest::SSBOStorage));

OGTEST_CppTest(LWNSSBOTest, lwn_ssbo_global_basic, (ShaderStage::FRAGMENT, LWNSSBOTest::GlobalStorage));
OGTEST_CppTest(LWNSSBOTest, lwn_ssbo_global_compute, (ShaderStage::COMPUTE, LWNSSBOTest::GlobalStorage));
