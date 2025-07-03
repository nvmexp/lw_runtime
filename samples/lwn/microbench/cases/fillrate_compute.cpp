/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Compute fillrate test
//
// Measures how long a compute shader takes to fill RGBA8 textures
// of varying sizes with varying kernel sizes.

// TODO: Test formats other than RGBA8?

#include "fillrate_compute.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <sstream>

typedef BenchmarkFillrateComputeLWN::TestDescr TestDescr;

/* For reference:
   LWN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_X:        1536;
   LWN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_Y:        1024;
   LWN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_Z:          64;
   LWN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_THREADS:  1536;
   LWN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_X:  65535;
   LWN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_Y:  65535;
   LWN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_Z:  65535;
*/

const TestDescr BenchmarkFillrateComputeLWN::subtests[] = {
    {  512,  512,  4,  8 },
    {  512,  512, 32, 32 },
    { 1024, 1024,  4,  8 },
    { 1024, 1024, 32, 32 },
};

struct GpuCounters
{
    LWNcounterData gpuTime0;
    LWNcounterData gpuTime1;
};

BenchmarkFillrateComputeLWN::BenchmarkFillrateComputeLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkFillrateComputeLWN::getNumSubTests()
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

int BenchmarkFillrateComputeLWN::numSubtests() const
{
    return BenchmarkFillrateComputeLWN::getNumSubTests();
}

BenchmarkCase::Description BenchmarkFillrateComputeLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];

    sprintf(testName, "fillrate_compute.texWidth=%d.texHeight=%d.csLocalSizeX=%d.csLocalSizeY=%d", t.texWidth, t.texHeight, t.csLocalSizeX, t.csLocalSizeY);

    Description d;
    d.name  = testName;
    d.units = "pix/s";
    return d;
}

void BenchmarkFillrateComputeLWN::setupTexture(LWNcommandBuffer* cmd, int texWidth, int texHeight)
{
    LWNtextureBuilder textureBuilder;
    lwnTextureBuilderSetDevice(&textureBuilder, device());
    lwnTextureBuilderSetDefaults(&textureBuilder);
    lwnTextureBuilderSetTarget(&textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(&textureBuilder, LWN_FORMAT_RGBA8);
    lwnTextureBuilderSetSize2D(&textureBuilder, texWidth, texHeight);
    lwnTextureBuilderSetFlags(&textureBuilder, LWN_TEXTURE_FLAGS_IMAGE_BIT);

    uintptr_t texSize = lwnTextureBuilderGetStorageSize(&textureBuilder);
    uintptr_t texAlign = lwnTextureBuilderGetStorageAlignment(&textureBuilder);
    uintptr_t poolOffset = gpuPool()->alloc(texSize, texAlign);

    lwnTextureBuilderSetStorage(&textureBuilder, gpuPool()->pool(), poolOffset);

    lwnTextureInitialize(&m_image, &textureBuilder);
    uint32_t textureID = descriptorPool()->allocTextureID();
    descriptorPool()->registerImage(textureID, &m_image);
    m_imageHandle = lwnDeviceGetImageHandle(device(), textureID);
}

void BenchmarkFillrateComputeLWN::init(int subtest)
{
    const TestDescr& testDescr = subtests[subtest];
    m_testDescr = &testDescr;

    m_numDispatches = 0;

    // create buffer for counters
    const int counterBufSize = sizeof(GpuCounters);
    m_counters = new LwnUtil::Buffer(device(), coherentPool(), nullptr, counterBufSize,
        BUFFER_ALIGN_COUNTER_BIT);

    assert(m_counters);

    // Create program from the device, provide them shader code and compile/link them
    // This program writes to the first half of the texture
    m_pgm = new LWNprogram;
    lwnProgramInitialize(m_pgm, device());

    // setup the compute shader based on test csLocalSizeX and csLocalSizeY
    // Since we don't have a function similar to glDispatchComputeGroupSize,
    // we have to hardcode the local sizes.
    std::stringstream ss;
    ss << "#version 440 core\n" <<
        "#extension GL_LW_gpu_shader5:require\n" <<
        "layout(binding=0, rgba8) uniform writeonly image2D image;\n" <<
        "layout(local_size_x=" << m_testDescr->csLocalSizeX << ", local_size_y=" << m_testDescr->csLocalSizeY << ", local_size_z=1) in;\n" <<
        // BlockLinear colwersion for better texture coalescing
        "ivec2 getCoord(ivec2 c) {" <<
        "ivec2 coord;" <<
        "coord.x = (c.x & 0xFFFFFFE1) | ((c.x & 0x4) >> 1) | ((c.x & 0x10) >> 2) | ((c.y & 3) << 3);" <<
        "coord.y = ((c.y & 0xFFFFFFFC) | ((c.x & 2) >> 1) | ((c.x & 8) >> 2));" <<
        "return coord; }" <<
        "void main() {\n" <<
          "    vec4 color = vec4(1.0, 1.0, 0.0, 1.0);\n" <<
          "    imageStore(image, getCoord(ivec2(gl_GlobalIlwocationID.xy)),color);" <<
          "}\n";

    std::string str = ss.str();
    LWNshaderStage stages[1] = { LWN_SHADER_STAGE_COMPUTE };
    const char *sources[1] = { (const char*)str.c_str() };
    int32_t nSources = 1;

    if (!LwnUtil::compileAndSetShaders(m_pgm, stages, nSources, sources))
    {
        PRINTF("\nERROR in shader compile\n");
        assert(0);
    }

    // command buffer init
    m_cmdBuf = new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 8, 64 * 1024, 64 * 1024);
    m_cmd = m_cmdBuf->cmd();
    lwnCommandBufferBeginRecording(m_cmd);


    float clearColor[] = { 0, 0, 0, 1 };
    lwnCommandBufferClearColor(m_cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(m_cmd, 1.0, LWN_TRUE, 0, 0);

    setupTexture(m_cmd, m_testDescr->texWidth, m_testDescr->texHeight);

    // bind image
    lwnCommandBufferBindImage(m_cmd, LWN_SHADER_STAGE_COMPUTE, 0, m_imageHandle);
    lwnCommandBufferBindProgram(m_cmd, m_pgm, LWN_SHADER_STAGE_COMPUTE_BIT);

    // initial timestamp
    lwnCommandBufferReportCounter(m_cmd, LWN_COUNTER_TYPE_TIMESTAMP, m_counters->address() + offsetof(GpuCounters, gpuTime0));
}

void BenchmarkFillrateComputeLWN::draw(const DrawParams* drawParams)
{
    if (!lwnCommandBufferIsRecording(m_cmd))
    {
        lwnCommandBufferBeginRecording(m_cmd);
    }

    m_numDispatches++;

    lwnCommandBufferDispatchCompute(m_cmd, m_testDescr->texWidth / m_testDescr->csLocalSizeX, m_testDescr->texHeight / m_testDescr->csLocalSizeY, 1);
    lwnCommandBufferReportCounter(m_cmd, LWN_COUNTER_TYPE_TIMESTAMP, m_counters->address() + offsetof(GpuCounters, gpuTime1));

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(m_cmd);
    m_cmdBuf->submit(1, &cmdHandle);
}



double BenchmarkFillrateComputeLWN::measuredValue(int subtest, double elapsedTime)
{
    // get elapsed time (in seconds)
    const GpuCounters *counterVA = (const GpuCounters *)m_counters->ptr();
    uint64_t gpuTimeNs0 = lwnDeviceGetTimestampInNanoseconds(device(), &counterVA->gpuTime0);
    uint64_t gpuTimeNs1 = lwnDeviceGetTimestampInNanoseconds(device(), &counterVA->gpuTime1);
    uint64_t gpuTimeNs = gpuTimeNs1 - gpuTimeNs0;
    m_gpuTime = (double)gpuTimeNs / 1000000000.0f;

    uint64_t numPixels = m_numDispatches * (m_testDescr->texWidth * m_testDescr->texHeight);

    return (double)numPixels / m_gpuTime;
}

void BenchmarkFillrateComputeLWN::deinit(int subtest)
{
    lwnProgramFinalize(m_pgm);
    delete m_pgm;
    delete m_cmdBuf;
    delete m_counters;

    lwnTextureFinalize(&m_image);
}

BenchmarkFillrateComputeLWN::~BenchmarkFillrateComputeLWN()
{
}
