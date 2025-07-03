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

#define DEBUG_MODE 0
#if DEBUG_MODE
#define DEBUG_PRINT(x) printf x
#else
#define DEBUG_PRINT(x)
#endif

using namespace lwn;

class LWNInternalShaderPoolTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNInternalShaderPoolTest::getDescription() const
{
    return "Testing internal shader pool on Windows.\n"
           "This is a stress test.\n"
           "The size of one exelwtable program is around 1 MB,\n"
           "looping 1024 times to test filling up the internal\n"
           "shader pool with 1GB worth of programs.\n"
           "Displays green if it passed, red otherwise.\n";
}

int LWNInternalShaderPoolTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 9);
}

void LWNInternalShaderPoolTest::doGraphics() const
{
    LWNdeviceFlagBits deviceFlags = LWNdeviceFlagBits(LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_4_BIT |
                                                      LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT);
    DeviceState *testDevice = new DeviceState(deviceFlags);

    if (!testDevice || !testDevice->isValid()) {
        delete testDevice;
        DeviceState::SetDefaultActive();
        LWNFailTest();
        return;
    }
    Device *device = testDevice->getDevice();
    testDevice->SetActive();

    bool failure = false;

    // This pool won't actually be used.
    int poolSize = 0x200000;
    lwnTest::GLSLCHelper glslcHelper(device, poolSize, g_glslcLibraryHelper, NULL);

    VertexShader vs(440);
    vs <<
        "layout(binding = 0) uniform Block {\n"
        "    vec4 inputColor;\n"
        "};\n"
        "void main() {\n"
        "  gl_Position = inputColor;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "#pragma optlevel none\n"
        "#extension GL_LW_gpu_shader5:require\n"
        "layout(binding = 0) uniform Block {\n"
        "    vec4 inputColor;\n"
        "};\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "  color = inputColor;\n";
    // To make the FS ucode size around 1 MB.
    // The value (12000) is fine-tuned from experiment.
    for (int i = 0; i < 12000; ++i) {
        fs << "color *= color;\n";
    }
    fs << "}\n";

    if (!glslcHelper.CompileShaders(vs, fs)) {
        DEBUG_PRINT(("We need a valid shader here!\n"));
        failure = true;
    }
    const GLSLCoutput * glslcOutput = glslcHelper.GetCompiledOutput(0);
    ShaderData shaderData[2];
    memset(&shaderData[0], 0, sizeof(ShaderData) * 2);

    LWNmemoryPoolFlags poolFlags = LWNmemoryPoolFlags(LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT |
                                                      LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT |
                                                      LWN_MEMORY_POOL_FLAGS_SHADER_CODE_BIT);
    LWNmemoryPool *pool = lwnDeviceCreateMemoryPool(reinterpret_cast<LWNdevice *>(device), NULL, poolSize, poolFlags);
    LWNbufferAddress poolBase = lwnMemoryPoolGetBufferAddress(pool);

    int stageCount = 0;
    for (unsigned int i = 0; i < glslcOutput->numSections; ++i) {
        if (glslcOutput->headers[i].genericHeader.common.type ==
            GLSLC_SECTION_TYPE_GPU_CODE) {
            const char * control = NULL;
            GLSLCgpuCodeHeader gpuCodeHeader =
                (GLSLCgpuCodeHeader)(glslcOutput->headers[i].gpuCodeHeader);
            const char * data = (char *)glslcOutput + gpuCodeHeader.common.dataOffset;
            control = data + gpuCodeHeader.controlOffset;

            shaderData[stageCount].data = poolBase;
            shaderData[stageCount].control = control;
            ++stageCount;
            assert(stageCount<=2);
        }
    }

    SYSTEMTIME start, end;
    GetLocalTime(&start);
    DEBUG_PRINT(("starts at: (h,m,s) = (%d,%d,%d).\n", start.wHour, start.wMinute, start.wSecond));

    // Testing 1GB capacity instead of 2GB capacity as the SetShader will run into "malloc failure"
    // before hitting the 2GB capacity.
    // Bug 1811877, We run out of memory in glCompileASMProgram before actually hitting the 2GB shader heap limit here.
    static const int maxProgramCount = 1024;
    Program *program[maxProgramCount];
    for (int i = 0; i < maxProgramCount; ++i) {
        program[i] = device->CreateProgram();
        if (!program[i] || !program[i]->SetShaders(stageCount, &(shaderData[0]))) {
            failure = true;
            DEBUG_PRINT(("Failure at loop ID = %d\n", i));
        }
    }

    GetLocalTime(&end);
    DEBUG_PRINT(("ends at: (h,m,s) = (%d,%d,%d).\n", end.wHour, end.wMinute, end.wSecond));

    for (int i = 0; i < maxProgramCount; ++i) {
        program[i]->Free();
    }
    lwnMemoryPoolFree(pool);

    // Manually clean up API resources that we created.
    delete testDevice;
    DeviceState::SetDefaultActive();

    // Render the results to screen.
    QueueCommandBuffer &gqueueCB = *g_lwnQueueCB;
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();

    // Renders all green if everything passed, red if failed.
    LWNfloat color[] = { 0.0, 0.0, 0.0, 1.0 };
    if (failure) {
        color[0] = 1.0;
        DEBUG_PRINT(("failed.\n"));
    } else {
        color[1] = 1.0;
        DEBUG_PRINT(("passed.\n"));
    }
    gqueueCB.ClearColor(0, color, ClearColorMask::RGBA);

    Queue *gqueue = DeviceState::GetActive()->getQueue();
    gqueueCB.submit();
    gqueue->Finish();
}

OGTEST_CppTest(LWNInternalShaderPoolTest, lwn_internal_shader_pool, );
