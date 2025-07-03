/*
* LWIDIA_COPYRIGHT_BEGIN
*
* Copyright 2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* LWIDIA_COPYRIGHT_END
*/

#include "tests/vkstress.h"
#include "vktest_platform.h"

#include <boost/program_options.hpp>
#include <iostream>
namespace po = boost::program_options;

namespace
{
RC RunTest(GpuTest *test)
{
    StickyRC rc;

    CHECK_RC_MSG(test->Setup(), "Setup failed");
    // Even if Run() fails we still need to call Cleanup()
    rc = test->Run();

    rc = test->Cleanup();

    return rc;
}
}

struct ErrorMessage
{
    int code;
    const char *message;
};

#undef DEFINE_RC
#define DEFINE_RC(errno, code, message) { RC::code, message },

static const ErrorMessage ExposedMessages[] =
{
DEFINE_RC(2, SOFTWARE_ERROR, "software error ")
DEFINE_RC(3, UNSUPPORTED_FUNCTION, "function is not supported")
DEFINE_RC(5, BAD_COMMAND_LINE_ARGUMENT, "bad command line argument")
DEFINE_RC(8, BAD_PARAMETER, "bad parameter passed to function")
DEFINE_RC(9, CANNOT_ALLOCATE_MEMORY, "cannot allocate memory")
DEFINE_RC(10, CANNOT_OPEN_FILE, "cannot open file")
DEFINE_RC(11, FILE_DOES_NOT_EXIST, "file does not exist")
DEFINE_RC(12, FILE_READ_ERROR, "failed while reading a file")
DEFINE_RC(83, GOLDEN_VALUE_MISCOMPARE, "CRC/Checksum miscompare")
DEFINE_RC(128, ILWALID_INPUT, "Invalid input")
DEFINE_RC(150, NO_TESTS_RUN, "No tests were run.")
DEFINE_RC(154, GPU_COMPUTE_MISCOMPARE, "compute test failed")
DEFINE_RC(244, UNSUPPORTED_HARDWARE_FEATURE, "feature is not supported in the hardware")
DEFINE_RC(282, UNEXPECTED_TEST_PERFORMANCE, "Performance varies from expected value")
DEFINE_RC(309, UNEXPECTED_TEST_COVERAGE, "Coverage value varies from expected value")
DEFINE_RC(332, CANNOT_MEET_WAVEFORM, "Cannot pulse fast enough to aclwrately make a waveform")
DEFINE_RC(375, MODS_VK_NOT_READY,                   "Vulkan not ready")
DEFINE_RC(376, MODS_VK_TIMEOUT,                     "Vulkan time out")
DEFINE_RC(377, MODS_VK_EVENT_SET,                   "Vulkan event set")
DEFINE_RC(378, MODS_VK_EVENT_RESET,                 "Vulkan event reset")
DEFINE_RC(379, MODS_VK_INCOMPLETE,                  "Vulkan Incomplete")
DEFINE_RC(380, MODS_VK_ERROR_OUT_OF_HOST_MEMORY,    "Vulkan out of host memory")
DEFINE_RC(381, MODS_VK_ERROR_OUT_OF_DEVICE_MEMORY,  "Vulkan out of device memory")
DEFINE_RC(382, MODS_VK_ERROR_INITIALIZATION_FAILED, "Vulkan init failed")
DEFINE_RC(383, MODS_VK_ERROR_DEVICE_LOST,           "Vulkan device lost")
DEFINE_RC(384, MODS_VK_ERROR_MEMORY_MAP_FAILED,     "Vulkan memory map failed")
DEFINE_RC(385, MODS_VK_ERROR_LAYER_NOT_PRESENT,     "Vulkan layer not present")
DEFINE_RC(386, MODS_VK_ERROR_EXTENSION_NOT_PRESENT, "Vulkan extension not present")
DEFINE_RC(387, MODS_VK_ERROR_FEATURE_NOT_PRESENT,   "Vulkan feature not present")
DEFINE_RC(388, MODS_VK_ERROR_INCOMPATIBLE_DRIVER,   "Vulkan Incompatible driver")
DEFINE_RC(389, MODS_VK_ERROR_TOO_MANY_OBJECTS,      "Vulkan Too many objects")
DEFINE_RC(390, MODS_VK_ERROR_FORMAT_NOT_SUPPORTED,  "Vulkan Format not supported")
DEFINE_RC(391, MODS_VK_ERROR_FRAGMENTED_POOL,       "Vulkan fragmented pool")
DEFINE_RC(392, MODS_VK_ERROR_SURFACE_LOST_KHR,      "Vulkan surface lost")
DEFINE_RC(393, MODS_VK_ERROR_NATIVE_WINDOW_IN_USE_KHR,  "Vulkan native window in use KHR")
DEFINE_RC(394, MODS_VK_SUBOPTIMAL_KHR,                  "Vulkan suboptimal KHR")
DEFINE_RC(395, MODS_VK_ERROR_OUT_OF_DATE_KHR,           "Vulkan out of date")
DEFINE_RC(396, MODS_VK_ERROR_INCOMPATIBLE_DISPLAY_KHR,  "Vulkan incompatible display")
DEFINE_RC(397, MODS_VK_ERROR_VALIDATION_FAILED_EXT,     "Vulkan validation failed")
DEFINE_RC(398, MODS_VK_ERROR_ILWALID_SHADER_LW,         "Vulkan invalid shader")
DEFINE_RC(399, MODS_VK_ERROR_OUT_OF_POOL_MEMORY_KHR,    "Vulkan out of pool memory")
DEFINE_RC(400, MODS_VK_ERROR_ILWALID_EXTERNAL_HANDLE_KHR, "Vulkan invalid external handle KHR")
DEFINE_RC(401, MODS_VK_GENERIC_ERROR,                     "Vulkan Generic Error")
DEFINE_RC(582, GPU_STRESS_TEST_FAILED, "gpu stress test found pixel miscompares")
DEFINE_RC(627, MODE_NOT_SUPPORTED, "Supplied mode not supported by the display.")
DEFINE_RC(776, ILWALID_ARGUMENT, "Invalid argument.")
};


int main(int argc, char const*argv[])
{
    PlatformOnEntry();

    po::options_description optionsDescription("Arguments");
    po::variables_map variablesMap;

    unsigned int runtimeMs = 10*60*1000;
    unsigned int bufferCheckMs = 5000;
    unsigned int gpuIndex = 0;

    optionsDescription.add_options()
        ("buffercheck_ms,b", po::value<unsigned int>(&bufferCheckMs), "Time between checks for errors in ms")
        ("gpu_index,g", po::value<unsigned int>(&gpuIndex), "GPU index")
        ("help,h", "Print help")
        ("runtime_ms,r", po::value<unsigned int>(&runtimeMs), "Runtime in ms")
        ("small_allocations,s", "Use small memory allocations")
        ;

    try
    {
        po::store(po::command_line_parser(argc, argv).
                  options(optionsDescription).run(), variablesMap);
        po::notify(variablesMap);
    }
    catch (po::error &x)
    {
        fprintf(stderr, "Command line error: %s\n", x.what());
        cout << optionsDescription;
        return 1;
    }

    if (variablesMap.count("help"))
    {
        cout << optionsDescription;
        return 1;
    }

    VkStress graphicsTest;

#ifdef NDEBUG
    graphicsTest.SetDumpPngOnError(false);
#endif

    graphicsTest.GetTestConfiguration()->SetDisplayWidth(512);
    graphicsTest.GetTestConfiguration()->SetDisplayHeight(400);
    graphicsTest.GetTestConfiguration()->SetGpuIndex(gpuIndex);

    if (variablesMap.count("small_allocations"))
    {
        // Match the SetupVkHeatStress:
        graphicsTest.SetStencil(false);
        graphicsTest.SetZtest(false);
        graphicsTest.SetTexReadsPerDraw(0);
        graphicsTest.SetNumLights(8);
        graphicsTest.SetPpV(50);

        graphicsTest.SetMaxHeat(true);
    }
    else
    {
        // Match the SetupVkPowerStress:
        graphicsTest.SetUseRandomTextureData(true);
        graphicsTest.SetMaxTextureSize(512);
        graphicsTest.SetUseHTex(false);

        graphicsTest.SetMaxPower(true);
    }

    graphicsTest.SetPulseMode(VkStress::RANDOM);

    if (bufferCheckMs == 0)
    {
        graphicsTest.SetBufferCheckMs(graphicsTest.BUFFERCHECK_MS_DISABLED);
    }
    else
    {
        graphicsTest.SetBufferCheckMs(bufferCheckMs);
    }

    graphicsTest.SetRuntimeMs(runtimeMs);

    printf("Running PowerPulse v1.0 on GPU index = %d ...\n", gpuIndex);

    const int rc_VkStress = RunTest(&graphicsTest).Get();
    if (rc_VkStress != RC::OK)
    {
        const char *errorMessage = "unknown";
        for (const ErrorMessage &em :  ExposedMessages)
        {
            if (em.code == rc_VkStress)
            {
                errorMessage = em.message;
                break;
            }
        }
        printf("Exit error code %d - %s\n", rc_VkStress, errorMessage);
        return 1;
    }

    return 0;
}

extern const char* const s_PowerPulseConstantShader = HS_(R"glsl(
#version 450  // GL version 4.5 assumed
#extension GL_ARB_compute_shader: enable
#extension GL_LW_shader_sm_builtins : require  // Needed for gl_SMCountLW
#extension GL_ARB_gpu_shader_int64 : enable    //uint64_t datatype
#extension GL_ARB_shader_clock : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable //float16_t
#pragma optionLW (unroll all)
const double TwoExp32         = 4294967296.0;  // 0x100000000 (won't fit in UINT32)
const double TwoExp32Less1    = 4294967295.0;  //  0xffffffff
#define TwoExp32Ilw           (1.0/TwoExp32)
#define TwoExp32Less1Ilw      (1.0/TwoExp32Less1)
#define SZ_ARRAY    16
#define MAX_SMS     144
// Lwca to GLSL Colwersions:
// gl_NumWorkGroups.xyz == gridDim.xyz
// gl_WorkGroupID.xyz == blockIdx.xyz
// gl_WorkGroupSize.xyz == blockDim.xyz
// gl_LocalIlwocationID.xyz == threadIdx.xyz
#define gX gl_GlobalIlwocationID.x
#define gY gl_GlobalIlwocationID.y
#define lX gl_LocalIlwocationID.x
#define lY gl_LocalIlwocationID.y

layout (local_size_x = 32, local_size_y = 1) in;

layout (constant_id = 0) const int ASYNC_COMPUTE = 0;

layout(std140, binding=0) uniform parameters
{
    int simWidth;
    int simHeight;
    int outerLoops;
    int innerLoops;
    uint64_t runtimeClks; //maximum number of clks to run the shader
    uint64_t pGpuSem; // graphics GPU semaphore to signal once all the SMs are loaded
};

layout(std430, binding=1) volatile coherent buffer stats
{
    uint     smLoaded[MAX_SMS];
    uint     smKeepRunning[MAX_SMS];     //final keepRunning value of thread 0 of each warp.
    uint64_t smStartClock[MAX_SMS];      //clock value when the kernel started
    uint64_t smEndClock[MAX_SMS];        //clock value when the kernel stopped
    uint     smFP16Miscompares[MAX_SMS]; //miscompares using float16_t ops
    uint     smFP32Miscompares[MAX_SMS]; //miscompares using float ops
    uint     smFP64Miscompares[MAX_SMS]; //miscompares using double ops
    uint     smIterations[MAX_SMS];      //exelwtion counter for performance measurements
};
layout(std430, binding=2) buffer Cells { uint64_t cells[]; };
layout(std430, binding=3) volatile coherent buffer KeepRunning { uint keepRunning; };

// Shared memory writes increase stress
shared float   s_Values[gl_WorkGroupSize.x][gl_WorkGroupSize.y];
shared uint64_t s_StartClk[gl_WorkGroupSize.x][gl_WorkGroupSize.y];
shared uint64_t s_EndClk[gl_WorkGroupSize.x][gl_WorkGroupSize.y];
uint GetRandom(inout uint pState)
{
    uint64_t temp = pState;
    pState = uint((1664525UL * temp + 1013904223UL) & 0xffffffff);
    return pState;
}

void InitRandomState(in uint seed, inout uint pState)
{
    uint i;
    pState = seed;
    GetRandom(pState);
    // randomize a random number of times between 0 - 31
    for (i = 0; i< (pState & 0x1F); i++)
        GetRandom(pState);
}

float GetRandomFloat(inout uint pState, in double min, in double max)
{
    double binSize = (max - min) * TwoExp32Less1Ilw;
    float temp = float(min + binSize * GetRandom(pState));
    return temp;
}

double GetRandomDouble(inout uint  pState, in double min, in double max)
{
    double binSize = (max - min) * TwoExp32Ilw;
    double coarse = binSize * GetRandom(pState);
    return min + coarse + (GetRandom(pState) * binSize * TwoExp32Less1Ilw);
}

float GetRandomFloatMeanSD(inout uint  pState, float mean, float stdDeviation)
{
    float rand1, rand2, s, z;
    //s must be 1 < s < 1
    do {
        rand1 = GetRandomFloat(pState, -1.0, 1.0);
        rand2 = GetRandomFloat(pState, -1.0, 1.0);
        s = rand1 * rand1 + rand2 * rand2;
    } while (s < 1E-2 || s >= 1.0);

    s = sqrt((-2.0 * log(s)) / s);
    z = rand1 * s;

    return(mean + z * stdDeviation);
}

double GetRandomDoubleMeanSD(inout uint  pState,
                             in double mean,
                             in double stdDeviation)
{
    double rand1, rand2, s, z;

    //s must be 0 < s < 1
    do {
        rand1 = GetRandomDouble(pState, -1.0, 1.0);
        rand2 = GetRandomDouble(pState, -1.0, 1.0);
        s = rand1 * rand1 + rand2 * rand2;
    } while (s == 0 || s >= 1.0);

    s = sqrt((-2.0 * log(float(s))) / s);
    z = rand1 * s;

    return(mean + z * stdDeviation);
}

float GenRandomFloat(in uint seed)
{
    uint randomState = 0;

    InitRandomState(seed, randomState);
    return GetRandomFloatMeanSD(randomState, 0.0F, 5.0F);
}

double GenRandomDouble(in uint seed)
{
    uint randomState = 0;

    InitRandomState(seed, randomState);
    return GetRandomDoubleMeanSD(randomState, 0.0LF, 5.0LF);
}

void Callwlate(inout float16_t halves[SZ_ARRAY],
               inout float floats[SZ_ARRAY],
               inout float16_t accHalf,
               inout float accFloat)
{
    // Run multiple loops of the aclwmulator to improve stressfulness
    for (int inner = 0; inner < innerLoops; inner++)
    {
        #pragma unroll
        for (int i = 0; i < SZ_ARRAY / 2; i++)
        {
            accHalf = fma(halves[i], halves[i + SZ_ARRAY / 2], accHalf);
            accFloat = fma(floats[i], floats[i + SZ_ARRAY / 2], accFloat);
            s_Values[lX][lY] = floats[i];
        }
    }
}

void main()
{
    if ((gX >= simWidth) || (gY >= simHeight))
    {
        return;
    }
    if ((lX == 0) && (lY == 0))
    {
        atomicExchange(smLoaded[gl_SMIDLW], 1);
    }
    s_StartClk[lX][lY] = clockARB();
    s_EndClk[lX][lY] = s_StartClk[lX][lY] + runtimeClks;
    uint iterations = 0;
    float16_t halves[SZ_ARRAY];
    float floats[SZ_ARRAY];

    float16_t initHalf = float16_t(0.0);
    float initFloat = 0.0;

    // Initialize vectors with random data
    for (int i = 0; i < SZ_ARRAY; i++)
    {
        halves[i]  = float16_t(GenRandomFloat(gl_LocalIlwocationIndex+i));
        floats[i] = GenRandomFloat(gl_LocalIlwocationIndex+i);
    }
    // now wait for all the SMs to load up their compute threads.
    if ((lX + lY) == 0)
    {
        uint allSMsLoaded = 0;
        int i = 0;
        //wait for all of the SMs to load some compute work
        while((ASYNC_COMPUTE == 0) && (allSMsLoaded < gl_SMCountLW) && (clockARB() < s_EndClk[lX][lY]))
        {
            allSMsLoaded = 0;
            for (i = 0; i < gl_SMCountLW; i++)
            {
                allSMsLoaded += (smLoaded[i] & 0x1);
            }
        }
    }
    barrier();
    //wait for the START cmd
    while((ASYNC_COMPUTE == 0) && ((keepRunning & 0x3) != 2) && (clockARB() < s_EndClk[lX][lY]));

    // Initialize aclwmulation result with vector dot-product FP16/FP64 operations
    Callwlate(halves, floats, initHalf, initFloat);
    iterations++;
    float16_t resHalf = initHalf;
    float  resFloat  = initFloat;
    cells[gY * simWidth + gX] = 0;

    //run until the STOP cmd
    while( ( ((keepRunning & 0x3) != 3) || (ASYNC_COMPUTE != 0) )
             && (clockARB() < s_EndClk[lX][lY]) )
    {
        // This is the heart of the power-stress kernel. The general principle is to activate
        // multiple arithmetic engines at a time. For instance, on SM75 f32 and f16 exercise
        // different engines, so we can issue instructions to both to increase power draw.
        //
        // The instructions used here aren't as optimal as hand-tuned SASS code,
        // but they seem to work well enough for the purposes of drawing power.
        //
        // We set our result to half the previous result and half the new
        // aclwmulated value. This allows us to easily check if there was an
        // error in the computation by comparing against the original value.
        for (int outer = 0; outer < outerLoops; outer++)
        {
            float16_t accHalf = float16_t(0.0);
            float accFloat  = 0.0;

            Callwlate(halves, floats, accHalf, accFloat);
            iterations++;

            resHalf = (float16_t(0.5) * resHalf) + (float16_t(0.5) * accHalf);
            resFloat = (0.5 * resFloat) + (0.5 * accFloat);
        }
        if (resHalf != initHalf)
        {
            atomicAdd(smFP16Miscompares[gl_SMIDLW], 1);
        }
        if (resFloat != initFloat)
        {
            atomicAdd(smFP32Miscompares[gl_SMIDLW], 1);
        }
    }
    barrier();
    if ((lX + lY) == 0)
    {
        smKeepRunning[gl_SMIDLW] =  keepRunning;
        smStartClock[gl_SMIDLW] = s_StartClk[lX][lY];
        smEndClock[gl_SMIDLW] = clockARB();
    }
    if (lX == 0)
    {
        atomicAdd(smIterations[gl_SMIDLW], iterations);
    }
    barrier();
}
)glsl");
