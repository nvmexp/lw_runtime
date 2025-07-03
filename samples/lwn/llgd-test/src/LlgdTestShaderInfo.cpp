/*
 * Copyright (c) 2017-2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <LlgdTest.h>
#include <LlgdTestUtil.h>
#include <LlgdTestUtilLWN.h>

#include <lwndevtools_bootstrap.h>

#include <vector>

// Inclue compiled shader data
#include "compiledshader/simpleTexturedModel_vertexMainData.h"
#include "compiledshader/simpleTexturedModel_vertexControlData.h"
#include "compiledshader/simpleTexturedModel_fragmentMainData.h"
#include "compiledshader/simpleTexturedModel_fragmentControlData.h"
#include "compiledshader/cmaacProcessZShapeC_computeMainData.h"
#include "compiledshader/cmaacProcessZShapeC_computeControlData.h"

class ShaderInfoValidator {
public:
    void Initialize();
    bool Test();

private:

    void InitializeBuffers();
    void InitializePrograms();

    // Programs to verify
    llgd_lwn::ProgramHolder phNotInitialized;
    llgd_lwn::ProgramHolder phJustInitialized;
    llgd_lwn::ProgramHolder phGraphicsShaderBinded;
    llgd_lwn::ProgramHolder phComputeShaderBinded;

    // Shader buffers
    llgd_lwn::ShaderBufferHolder sbhVertexShader;
    llgd_lwn::ShaderBufferHolder sbhFragmentShader;
    llgd_lwn::ShaderBufferHolder sbhComputeShader;

    // We need to have one live queue for some of the Getter things in
    // devtools bootstrap land.
    llgd_lwn::QueueHolder qh;

    // Bootstrap layer interface
    const LWNdevtoolsBootstrapFunctions* devtools;

    // Constant
    static const std::vector<LWNshaderStage> s_allShaderStages;

};

// static member declaration
const std::vector<LWNshaderStage> ShaderInfoValidator::s_allShaderStages{
    LWN_SHADER_STAGE_VERTEX,
    LWN_SHADER_STAGE_FRAGMENT,
    LWN_SHADER_STAGE_GEOMETRY,
    LWN_SHADER_STAGE_TESS_CONTROL,
    LWN_SHADER_STAGE_TESS_EVALUATION,
    LWN_SHADER_STAGE_COMPUTE
};

//------------------------------------------------
// Initialize
//------------------------------------------------
void ShaderInfoValidator::Initialize()
{
    qh.Initialize(g_device);
    devtools = lwnDevtoolsBootstrap();

    InitializeBuffers();
    InitializePrograms();
}

//------------------------------------------------
// InitializeBuffers
//------------------------------------------------
void ShaderInfoValidator::InitializeBuffers()
{
    sbhVertexShader.Initialize(g_device, simpleTexturedModel_vertexMainData, simpleTexturedModel_vertexMainDataSize);
    sbhFragmentShader.Initialize(g_device, simpleTexturedModel_fragmentMainData, simpleTexturedModel_fragmentMainDataSize);
    sbhComputeShader.Initialize(g_device, cmaacProcessZShapeC_computeMainData, cmaacProcessZShapeC_computeControlDataSize);
}

//------------------------------------------------
// InitializePrograms
//------------------------------------------------
void ShaderInfoValidator::InitializePrograms()
{
    //
    // Initialize programs not binded with shaders.
    //

    // A program only initialized.
    phJustInitialized.Initialize(static_cast<Device*>(g_device));

    //
    // Initialize programs with shaders
    //

    // Vertex shader and fragment shader
    {
        std::vector<ShaderData> lwnShaderData(2);
        lwnShaderData[0].data = sbhVertexShader->GetAddress();
        lwnShaderData[0].control = simpleTexturedModel_vertexControlData;
        lwnShaderData[1].data = sbhFragmentShader->GetAddress();
        lwnShaderData[1].control = simpleTexturedModel_fragmentControlData;

        // Initialize the program and provide it with the compiled shader
        phGraphicsShaderBinded.Initialize(static_cast<Device*>(g_device));
        phGraphicsShaderBinded->SetShaders(lwnShaderData.size(), &lwnShaderData[0]);
    }

    // Compute shader
    {
        std::vector<ShaderData> lwnShaderData(1);
        lwnShaderData[0].data = sbhComputeShader->GetAddress();
        lwnShaderData[0].control = cmaacProcessZShapeC_computeControlData;

        // Initialize the program and provide it with the compiled shader
        phComputeShaderBinded.Initialize(static_cast<Device*>(g_device));
        phComputeShaderBinded->SetShaders(lwnShaderData.size(), &lwnShaderData[0]);
    }
}

//------------------------------------------------
// Test
//------------------------------------------------
bool ShaderInfoValidator::Test()
{
    // Case1: pass a nullptr program
    for (auto shaderStage : s_allShaderStages)
    {
        auto infoFlags = devtools->GetShaderInfoFlags(nullptr, shaderStage);
        TEST(infoFlags.defaultValues);
    }

    // Case2: pass an only initialized program
    // Note: GetShaderInfoFlags requires lwnProgramSetShader called except for compute stage.
    {
        auto infoFlags = devtools->GetShaderInfoFlags(phJustInitialized, LWN_SHADER_STAGE_COMPUTE);
        TEST(infoFlags.defaultValues);
    }

    // Case3: pass an initialized program by graphics shader data
    for (auto shaderStage : s_allShaderStages)
    {
        // All stages, including compute, should have InfoFlags even if compiledState
        // is not allocated by ProgramPrepareCompiledState. Because compiledState will
        // be initialized in lwnProgramSetShader function if valid shaders are set.
        auto infoFlags = devtools->GetShaderInfoFlags(phGraphicsShaderBinded, shaderStage);
        TEST(!infoFlags.defaultValues);

        // Vertex shader and fragment shader are binded.
        // They ahave determined shaderInfoFlags
        switch (shaderStage)
        {
        case LWN_SHADER_STAGE_VERTEX:
            TEST(!infoFlags.hasBindlessBufferWrites);
            TEST(!infoFlags.hasImageWrites);
            TEST(!infoFlags.hasBindlessTextureExtension);
            break;
        case LWN_SHADER_STAGE_FRAGMENT:
            TEST(!infoFlags.hasBindlessBufferWrites);
            TEST(infoFlags.hasImageWrites); // The fs shader calls storeImage
            TEST(!infoFlags.hasBindlessTextureExtension);
            break;
        default:
            break;
        }
    }

    // Case4: pass an initialized program only with compute shader data.
    // Graphics shaders' infoFlags are not allocated. Thus we can only
    // retrieve shaderInfoFlags with compute shader stage. If we call
    // like GetShaderInfoFlags(phComputeShaderBinded, STAGE_VERTEX),
    // memory access violation error will happen.
    {
        auto infoFlags = devtools->GetShaderInfoFlags(phComputeShaderBinded, LWN_SHADER_STAGE_COMPUTE);
        TEST(!infoFlags.defaultValues);
        TEST(!infoFlags.hasBindlessBufferWrites);
        TEST(infoFlags.hasImageWrites); // The shader calls storeImage
        TEST(!infoFlags.hasBindlessTextureExtension);
    }

    return true;
}

LLGD_DEFINE_TEST(ShaderInfo, UNIT,
LwError Execute()
{
    ShaderInfoValidator v;
    v.Initialize();

    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
