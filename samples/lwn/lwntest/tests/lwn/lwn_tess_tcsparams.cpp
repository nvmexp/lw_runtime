/*
 * Copyright (c) 2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include "resources/lwn_tess_tcsparams/tes_allstate.spv.hex"
#include "resources/lwn_tess_tcsparams/tcs_allstate.spv.hex"
#include "resources/lwn_tess_tcsparams/tes_partialstate.spv.hex"
#include "resources/lwn_tess_tcsparams/tcs_partialstate.spv.hex"
#include "resources/lwn_tess_tcsparams/tes_nostate.spv.hex"
#include "resources/lwn_tess_tcsparams/tcs_nostate.spv.hex"
#include "resources/lwn_tess_tcsparams/vert.spv.hex"
#include "resources/lwn_tess_tcsparams/frag.spv.hex"

using namespace lwn;

class LWNTessellationTCSParamsTest
{
private:
    bool m_isSeparable;
public:
    LWNTessellationTCSParamsTest(bool isSeparable) : m_isSeparable(isSeparable) {}
    LWNTEST_CppMethods();
};

lwString LWNTessellationTCSParamsTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Test out that tessellation parameters can be specified in the tessellation control shader. "
        "The LWN API, as of GLSLC version 1.16, supports specifying tessellation state not only inside "
        "the tess eval shader but also, by using SPIR-V shaders, in the control shader. "
        "This test tests different combinations of tessellation control/eval shaders to make sure that "
        "the correct tessellation hardware state is set in various situations.\n\n"
        "this test uses 3 TCS shaders and 3 TES shaders and combines them to 8 different "
        "combinations of state (1 combination that doesn't have any state so it is excluded from this test since it is considered invalid combination). "
        "The three variants of the tess control shader are (1) specifying full tess state of quads, fractional odd spacing, and "
        "point mode; (2) specifying partial state of quads, point mode, and (3) specifying no state at all. "
        "The three variants of the tess evaluation shader are (1) specifying full state of quads, fractional odd spacing, point mode, and point size of 6; "
        "(2) specifying partial state of quads, fractional odd spacing, and point size of 4, and (3) only specifying point size of 2. "
        "In other words, in TCS (2) point mode is declared but not in TES (2), and in TES (2) fractional odd spacing is declared but not in TCS (2). "
        "The shaders are combined in the following way:\n"
        "The left column uses TCS (1), middle column uses TCS (2), and right column uses TCS (3)\n"
        "The bottom row uses TES (1), middle row uses TES (2), and top row uses TES (3)\n\n";

        if (!m_isSeparable) {
        sb << "Both the TCS and TES are compiled into the same program so the "
              "parameters from each will be merged together by LWN to set the tessellator "
              "state.\n"
              "The bottom row should all be tessellated quads using point mode and fractional odd spacing."
              "In the middle row, the left two should appear similar to bottom row, and right one should be tessellated quad with fractional spacing without point mode."
              "The top left is same as bottom row.  The top middle is tessellated quad as points, but using default spacing; "
              "and top right is excluded because TCS(3) and TES(3) don't have state and no state sent to hardware and would also trigger debug "
              "warning since no state is set in either TES or TCS for the program.";
        } else {
            sb << "The TCS and TES are compiled as separable programs and combined together at runtime by binding the different stages separately before the draw call. "
                  "The parameters are NOT merged together and since the TES is bound lastly, if it contains a primitive declaration (which it does in TES (1) and (2)), then its state will override the state set by the TCS."
                  "The bottom row is tessellated quads drawn as points with fractional odd spacing (because all three bottom quads have full state specified in TES (1)). "
                  "The middle row is tessellated quads drawn as triangles with fractional odd spacing (because TES (2) specifies only fractional odd spacing and quad primitive mode). "
                  "The top left is tessellated quad in point mode with fractional odd spacing (because TES (3) specifies no state so TCS (1) full state is used). "
                  "The top middle is tessellated quad in point mode with normal spcaing (because TES (3) specifies no state so TCS(2) partial state is used). "
                  "The top right is excluded because TCS (3) and TES (3) have no state and no commands would be sent which is considered a draw-time error. ";
        }
    return sb.str();
}

int LWNTessellationTCSParamsTest::isSupported() const
{
    return lwogCheckLWNGLSLCPackageVersion(73) &&
           lwogCheckLWNAPIVersion(53,312);
}

void LWNTessellationTCSParamsTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    const int32_t tessellationPrimitiveModeMissingWarningID = 1303;  // LWN_DEBUG_MESSAGE_TESSELLATION_PRIMITIVE_MODE_MISSING

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 0x100UL, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Clear the framebuffer.
    float clearColor[] = { 0, 0, 0, 1 };
    queueCB.ClearColor(0, clearColor, ClearColorMask::RGBA);

    // A set of combined programs and a set of separable programs for each stage.
    Program *combinedPrograms[3][3]; // holds merged programs , or for separable holds TCS programs
    Program *sepProgramsTCS[3]; // for separable, holds the TES programs.
    Program *sepProgramsTES[3]; // for separable, holds the TES programs.
    Program *sepVsFsProgram = NULL;

    unsigned char * spvShaderSources[2][3] =
    {
        {
            tcs_allstate_spv,
            tcs_partialstate_spv,
            tcs_nostate_spv
        },
        {
            tes_allstate_spv,
            tes_partialstate_spv,
            tes_nostate_spv
        }
    };

    unsigned int spvShaderSourcesLens[2][3] =
    {
        {
            tcs_allstate_spv_len,
            tcs_partialstate_spv_len,
            tcs_nostate_spv_len
        },
        {
            tes_allstate_spv_len,
            tes_partialstate_spv_len,
            tes_nostate_spv_len
        }
    };

    LWNshaderStage spvStages[4];
    spvStages[0] = LWN_SHADER_STAGE_VERTEX;
    spvStages[1] = LWN_SHADER_STAGE_FRAGMENT;
    spvStages[2] = LWN_SHADER_STAGE_TESS_CONTROL;
    spvStages[3] = LWN_SHADER_STAGE_TESS_EVALUATION;

    g_glslcHelper->SetSeparable(m_isSeparable);

    LWNboolean compiled = LWN_TRUE;

    // Compile each set of TCS+TES as either the merged (non-separable) or separable cases.
    for (int tcsIdx = 0; tcsIdx < 3; ++tcsIdx) {
        for (int tesIdx = 0; tesIdx < 3; ++tesIdx) {
            const unsigned char * spvShaders[4] = { vert_spv,
                                                    frag_spv,
                                                    spvShaderSources[0][tcsIdx],
                                                    spvShaderSources[1][tesIdx]};

            // tcs_nostate_spv and tes_nostate_spv would produce an invalid tessellation state.
            // Please see the description about the tests above.
            // Suppress the warnings for these messages not to fail while doing QA testing.
            if ((tcsIdx == 2) && (tesIdx == 2)) {
            	DebugWarningIgnore(tessellationPrimitiveModeMissingWarningID);
            } else {
                DebugWarningAllow(tessellationPrimitiveModeMissingWarningID);
            }

            if (!m_isSeparable) {

                SpirvParams spvParams;

                spvParams.sizes[0] = vert_spv_len;
                spvParams.sizes[1] = frag_spv_len;
                spvParams.sizes[2] = spvShaderSourcesLens[0][tcsIdx];
                spvParams.sizes[3] = spvShaderSourcesLens[1][tesIdx];

                combinedPrograms[tcsIdx][tesIdx] = device->CreateProgram();
                compiled = compiled &&
                           g_glslcHelper->lwnUtil::GLSLCHelper::CompileAndSetShaders(
                                   reinterpret_cast<LWNprogram*> (combinedPrograms[tcsIdx][tesIdx]),
                                   spvStages, 4, reinterpret_cast<const char **>(&spvShaders[0]), &spvParams);
            } else {
                if (tesIdx == 0) {
                    // Copile the TCS shaders when they first appear in the loops.
                    SpirvParams spvParams;

                    spvParams.sizes[0] = spvShaderSourcesLens[0][tcsIdx];

                    sepProgramsTCS[tcsIdx] = device->CreateProgram();
                    compiled = compiled &&
                               g_glslcHelper->lwnUtil::GLSLCHelper::CompileAndSetShaders(
                                       reinterpret_cast<LWNprogram *> (sepProgramsTCS[tcsIdx]),
                                       &spvStages[2], 1, reinterpret_cast<const char **> (&spvShaders[2]),
                                       &spvParams);
                }
                if (tcsIdx == 0) {
                    // Compile the TES shaders when they first appear in the loops.
                    SpirvParams spvParams;

                    spvParams.sizes[0] = spvShaderSourcesLens[1][tesIdx];

                    sepProgramsTES[tesIdx] = device->CreateProgram();
                    compiled = compiled &&
                               g_glslcHelper->lwnUtil::GLSLCHelper::CompileAndSetShaders(
                                       reinterpret_cast<LWNprogram *> (sepProgramsTES[tesIdx]),
                                       &spvStages[3], 1, reinterpret_cast<const  char **> (&spvShaders[3]), &spvParams);
                }
            }
        }
    }
    // Unmask the tessellationPrimitiveModeMissingWarningID type of warnings.
    DebugWarningAllow(tessellationPrimitiveModeMissingWarningID);

    // Compile the vertex and fragment separable programs.
    if (m_isSeparable) {
        const unsigned char * spvShaders[2] = {vert_spv,  frag_spv};

        SpirvParams spvParams;
        spvParams.sizes[0] = vert_spv_len;
        spvParams.sizes[1] = frag_spv_len;

        sepVsFsProgram = device->CreateProgram();
        compiled = compiled &&
                   g_glslcHelper->lwnUtil::GLSLCHelper::CompileAndSetShaders(
                           reinterpret_cast<LWNprogram*> (sepVsFsProgram),
                           spvStages, 2, reinterpret_cast<const char **> (&spvShaders[0]), &spvParams);
    }

    g_glslcHelper->SetSeparable(LWN_FALSE);

    // If any compilation fails, clear the screen to red and bail.
    if (!compiled) {
        float clearColor[] = { 1, 0, 0, 1 };
        queueCB.ClearColor(0, clearColor, ClearColorMask::RGBA);
        return;
    }

    // Set up the vertex state and buffer objects for the test, with one single-quad patch.
    struct Vertex {
        dt::vec3 position;
    };
    static const Vertex vertexData[] = {
        // Single quad patch.
        { dt::vec3(-0.9f, -0.9f, 0.5f) },
        { dt::vec3(+0.9f, -0.9f, 0.5f) },
        { dt::vec3(-0.9f, +0.9f, 0.5f) },
        { dt::vec3(+0.9f, +0.9f, 0.5f) },
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexData), allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    PolygonState polygon;
    polygon.SetDefaults();
    polygon.SetPolygonMode(PolygonMode::LINE);

    // Set up pipeline objects for each configuration.
    queueCB.BindPolygonState(&polygon);
    queueCB.BindVertexArrayState(vertex);

    LWNfloat levels[] = { 12, 12 ,12 ,12 };
    queueCB.SetOuterTessellationLevels(levels);
    queueCB.SetInnerTessellationLevels(levels);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {

            if (i == 2 && j == 2) {
                // The top right square would be programs with no state which
                // means no state would be sent to hardware and is considered
                // an error.
                continue;
            }

            int patchSize = 4;
            queueCB.SetViewport(i * (lwrrentWindowWidth / 3), j * lwrrentWindowHeight / 3,
                                lwrrentWindowWidth / 3, lwrrentWindowHeight / 3);
            queueCB.SetScissor(i * (lwrrentWindowWidth / 3), j * lwrrentWindowHeight / 3,
                               lwrrentWindowWidth / 3, lwrrentWindowHeight / 3);
            queueCB.SetPatchSize(patchSize);
            if (m_isSeparable) {
                queueCB.BindProgram(sepVsFsProgram, ShaderStageBits::VERTEX | ShaderStageBits::FRAGMENT);
                queueCB.BindProgram(sepProgramsTCS[i], ShaderStageBits::TESS_CONTROL);
                queueCB.BindProgram(sepProgramsTES[j], ShaderStageBits::TESS_EVALUATION);
            } else {
                queueCB.BindProgram(combinedPrograms[i][j], ShaderStageBits::ALL_GRAPHICS_BITS);
            }
            queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
            queueCB.DrawArrays(DrawPrimitive::PATCHES, 0, patchSize);
        }
    }

    queueCB.submit();

    queue->Finish();
}

OGTEST_CppTest(LWNTessellationTCSParamsTest, lwn_tess_tcsparams, (false));
OGTEST_CppTest(LWNTessellationTCSParamsTest, lwn_tess_tcsparams_sep, (true));
