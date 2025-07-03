/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
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

#include <liblwn-llgd.h>

#include <class/cl9097.h>

static uint32_t LwnGraphicsShaderStageToHwStage(LWNshaderStage shaderStage)
{
    CHECK(shaderStage < LWN_SHADER_STAGE_COMPUTE);
    static const uint32_t lwnGraphicsShaderStageToHwStage[]{
        1 /*__LWN_HW_SHADER_STAGE_VERTEX_B*/,
        5 /*__LWN_HW_SHADER_STAGE_PIXEL*/,
        4 /*__LWN_HW_SHADER_STAGE_GEOMETRY*/,
        2 /*__LWN_HW_SHADER_STAGE_TESS_CONTROL*/,
        3 /*__LWN_HW_SHADER_STAGE_TESS_EVAL*/,
    };
    return lwnGraphicsShaderStageToHwStage[shaderStage];
}

class DecodeGraphicsShaderValidator {
public:
    bool Test();

private:
    bool TestGetGraphicsShaderValuesFromMME();
    bool TestGetGraphicsShaderLMemSize();
    bool TestGetActiveGraphicsShaderStages();
};

bool DecodeGraphicsShaderValidator::Test()
{
    if (!TestGetGraphicsShaderValuesFromMME()) { return false; }
    if (!TestGetGraphicsShaderLMemSize()) { return false; }
    if (!TestGetActiveGraphicsShaderStages()) { return false; }
    return true;
}

// Test llgdLwnGetGraphicsShaderProgramId and llgdLwnGetGraphicsShaderGpuOffset
bool DecodeGraphicsShaderValidator::TestGetGraphicsShaderValuesFromMME()
{
    MmeShadowRegisters mme;
    uint32_t programId = 12;
    uint32_t offset = 0x4000;

    // Set the program ID scratch register
    auto SetProgramId = [&mme = mme](uint32_t hwStage, uint32_t programId) {
        uint32_t mme_scratch_program_id = 0x14 + hwStage + 8 /*RM_MME_FIRST_USABLE_SHADOW_SCRATCH*/;
        mme.Set(LW9097_SET_MME_SHADOW_SCRATCH(mme_scratch_program_id), programId, 0, 0);
    };

    // Set the program offset scratch register
    auto SetProgramOffset = [&mme = mme](uint32_t hwStage, uint32_t offset) {
        uint32_t mme_scratch_program_offset = 0x1A + hwStage + 8 /*RM_MME_FIRST_USABLE_SHADOW_SCRATCH*/;
        mme.Set(LW9097_SET_MME_SHADOW_SCRATCH(mme_scratch_program_offset), offset, 0, 0);
    };

    for (int ishaderStage = 0; ishaderStage < static_cast<int>(LWN_SHADER_STAGE_COMPUTE); ++ishaderStage)
    {
        LWNshaderStage shaderStage = static_cast<LWNshaderStage>(ishaderStage);
        const uint32_t hwStage = LwnGraphicsShaderStageToHwStage(shaderStage);

        SetProgramId(hwStage, programId);
        SetProgramOffset(hwStage, offset);

        TEST_EQ_FMT(llgdLwnGetGraphicsShaderProgramId(shaderStage, mme), programId, "shaderStage = %d", shaderStage);
        TEST_EQ_FMT(llgdLwnGetGraphicsShaderGpuOffset(shaderStage, mme), offset, "shaderStage = %d", shaderStage);
    }
    return true;
}

// Test llgdLwnGetGraphicsShaderLMemSize
bool DecodeGraphicsShaderValidator::TestGetGraphicsShaderLMemSize()
{
    uint32_t sph[3];
    sph[0] = 0x00000000; // doesn't matter for this test
    sph[1] = 0x00000000; // LW9097_SPHV3T1_SHADER_LOCAL_MEMORY_LOW_SIZE  (55:32)
    sph[2] = 0x00000000; // LW9097_SPHV3T1_SHADER_LOCAL_MEMORY_HIGH_SIZE (87:64)

    TEST_EQ(llgdLwnGetGraphicsShaderLMemSize(&sph), 0x00000000);

    sph[0] = 0x00000000;
    sph[1] = 0xFFFFFFFF;
    sph[2] = 0xFFFFFFFF;

    TEST_EQ(llgdLwnGetGraphicsShaderLMemSize(&sph), 0x02000000);

    sph[0] = 0x00000000;
    sph[1] = 0xAA111100;
    sph[2] = 0xBB222200;

    TEST_EQ(llgdLwnGetGraphicsShaderLMemSize(&sph), 0x00333300);

    sph[0] = 0xA34EC18F;
    sph[1] = 0xCD1111AA;
    sph[2] = 0x1E2222BB;

    TEST_EQ(llgdLwnGetGraphicsShaderLMemSize(&sph), 0x001111B0 + 0x002222C0);

    sph[0] = 0xA34EC18F;
    sph[1] = 0xCDE348AA;
    sph[2] = 0x1EF6CBFB;

    TEST_EQ(llgdLwnGetGraphicsShaderLMemSize(&sph), 0x00E348B0 + 0x00F6CC00);

    return true;
}

// Test llgdLwnGetActiveGraphicsShaderStages
bool DecodeGraphicsShaderValidator::TestGetActiveGraphicsShaderStages()
{
    MmeShadowRegisters mme;

    static const int NUM_SHADER_STAGES = 6;
    bool expectedActiveStages[NUM_SHADER_STAGES];
    bool actualActiveStages[NUM_SHADER_STAGES];

#define SET_ACTIVE_GRAPHICS_STAGE(name, id, val) \
    mme.Set(SetPipeline_Shader_##id##_0, val, 0, 0); \
    expectedActiveStages[LWN_SHADER_STAGE_##name] = val ? true : false;

    auto SetActiveGraphicsStages = [&](uint8_t vs, uint8_t fs, uint8_t gs, uint8_t tcs, uint8_t tes) {
        SET_ACTIVE_GRAPHICS_STAGE(VERTEX, 1, vs);
        SET_ACTIVE_GRAPHICS_STAGE(FRAGMENT, 5, fs);
        SET_ACTIVE_GRAPHICS_STAGE(GEOMETRY, 4, gs);
        SET_ACTIVE_GRAPHICS_STAGE(TESS_CONTROL, 2, tcs);
        SET_ACTIVE_GRAPHICS_STAGE(TESS_EVALUATION, 3, tes);
    };

    auto ValidateActiveGraphicsStages = [&]() {
        for (int ishaderStage = 0; ishaderStage < static_cast<int>(LWN_SHADER_STAGE_COMPUTE); ++ishaderStage)
        {
            TEST_EQ_FMT(expectedActiveStages[ishaderStage], actualActiveStages[ishaderStage], "ShaderStage = %d", ishaderStage);
        }
        return true;
    };

    SetActiveGraphicsStages(0, 0, 0, 0, 0);
    llgdLwnGetActiveGraphicsShaderStages(actualActiveStages, mme);
    ValidateActiveGraphicsStages();

    SetActiveGraphicsStages(0, 1, 0, 1, 0);
    llgdLwnGetActiveGraphicsShaderStages(actualActiveStages, mme);
    ValidateActiveGraphicsStages();

    SetActiveGraphicsStages(1, 0, 0, 1, 1);
    llgdLwnGetActiveGraphicsShaderStages(actualActiveStages, mme);
    ValidateActiveGraphicsStages();

    SetActiveGraphicsStages(0, 1, 0, 0, 0);
    llgdLwnGetActiveGraphicsShaderStages(actualActiveStages, mme);
    ValidateActiveGraphicsStages();

    SetActiveGraphicsStages(1, 1, 1, 1, 1);
    llgdLwnGetActiveGraphicsShaderStages(actualActiveStages, mme);
    ValidateActiveGraphicsStages();

    return true;
}

LLGD_DEFINE_TEST(DecodeGraphicsShader, UNIT,
LwError Execute()
{
    DecodeGraphicsShaderValidator v;
    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
