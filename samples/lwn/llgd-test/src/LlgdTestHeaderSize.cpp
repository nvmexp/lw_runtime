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

class HeaderSizeValidator {
public:
    bool Test();
};

bool HeaderSizeValidator::Test()
{
    int tex_head_size;
    int smp_head_size;

    g_device->GetInteger(DeviceInfo::TEXTURE_DESCRIPTOR_SIZE, &tex_head_size);
    g_device->GetInteger(DeviceInfo::SAMPLER_DESCRIPTOR_SIZE, &smp_head_size);

    TEST_EQ(llgdLwnGetTextureHeaderSize(), static_cast<uint32_t>(tex_head_size));
    TEST_EQ(llgdLwnGetSamplerHeaderSize(), static_cast<uint32_t>(smp_head_size));

    return true;
}

LLGD_DEFINE_TEST(HeaderSize, UNIT,
LwError Execute()
{
    HeaderSizeValidator v;
    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
