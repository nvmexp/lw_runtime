/*
* LWIDIA_COPYRIGHT_BEGIN
*
* Copyright 2020 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* LWIDIA_COPYRIGHT_END
*/

#include "tests/vkstress.h"
#include "vktest_platform.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#define MOON_RC_0    ( QUAL_OK,             0, "PASSING RESULT")
#define MOON_RC_1    ( QUAL_EARLY_EXIT,     1, "Exited early due to user intervention")
#define MOON_RC_2    ( QUAL_RUNTIME_ERROR,  2, "Failed due to SW error.  Stacktrace in result.xml")
#define MOON_RC_3    ( QUAL_TDR_ERROR,      3, "Failed due to TDR error")
#define MOON_RC_4    ( QUAL_DBE_ERROR,      4, "Failed due to DBE ECC error")
#define MOON_RC_5    ( QUAL_RC_ERROR,       5, "Failed due to RC error")
#define MOON_RC_6    ( QUAL_DATA_ERROR,     6, "Failed due to GPU data mismatch")
#define MOON_RC_7    ( QUAL_PEX_FA_ERROR,   7, "Failed due to FA PEX error")
#define MOON_RC_8    ( QUAL_PEX_UN_ERROR,   8, "Failed due to UN PEX error")
#define MOON_RC_9    ( QUAL_PEX_NF_ERROR,   9, "Failed due to NF PEX error")
#define MOON_RC_10   ( QUAL_PEX_CE_ERROR,  10, "Failed due to CE PEX error")
#define MOON_RC_11   ( QUAL_SBE_ERROR,     11, "Failed due to SBE ECC error")
#define MOON_RC_CNT 12

#define ENUM_DEF0(TUPLE) BOOST_PP_TUPLE_ELEM(3,0,TUPLE) =  BOOST_PP_TUPLE_ELEM(3,1,TUPLE),
#define ENUM_DEF(CNT,IDX,PREFIX) ENUM_DEF0( BOOST_PP_CAT(PREFIX,IDX) )
enum moonshine_rc_e{
    BOOST_PP_REPEAT(MOON_RC_CNT, ENUM_DEF, MOON_RC_)
};

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

int main(int argc, char const*argv[])
{
    PlatformOnEntry();

    po::options_description optionsDescription;
    po::variables_map variablesMap;

    unsigned int runtimeMs = 4000;
    unsigned int bufferCheckMs = 1000;
    unsigned int workloadPct = 100;
    unsigned int width = 512;
    unsigned int height = 400;
    unsigned int gpuIndex = 0;

    optionsDescription.add_options()
        ("r,r", po::value<unsigned int>(&runtimeMs), "")
        ("b,b", po::value<unsigned int>(&bufferCheckMs), "")
        ("l,l", po::value<unsigned int>(&workloadPct), "")
        ("w,w", po::value<unsigned int>(&width), "")
        ("h,h", po::value<unsigned int>(&height), "")
        ("g,g", po::value<unsigned int>(&gpuIndex), "")
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
        return QUAL_EARLY_EXIT;
    }

    VkStress graphicsTest;

#ifdef NDEBUG
    graphicsTest.SetDumpPngOnError(false);
#endif

    graphicsTest.GetTestConfiguration()->SetDisplayWidth(width);
    graphicsTest.GetTestConfiguration()->SetDisplayHeight(height);
    graphicsTest.GetTestConfiguration()->SetGpuIndex(gpuIndex);

    // Match the SetupVkPowerStress:
    graphicsTest.SetUseRandomTextureData(true);
    graphicsTest.SetMaxTextureSize(512);
    graphicsTest.SetUseHTex(false);

    graphicsTest.SetMaxPower(true);

    if (workloadPct < 100)
    {
        graphicsTest.SetPulseMode(graphicsTest.SWEEP);
        graphicsTest.SetLowHz(250);
        graphicsTest.SetHighHz(250);
        if (workloadPct == 0)
        {
            graphicsTest.SetDutyPct(0.001);
        }
        else
        {
            graphicsTest.SetDutyPct(workloadPct);
        }
    }

    if (bufferCheckMs == 0)
    {
        graphicsTest.SetBufferCheckMs(graphicsTest.BUFFERCHECK_MS_DISABLED);
    }
    else
    {
        graphicsTest.SetBufferCheckMs(bufferCheckMs);
    }

    graphicsTest.SetRuntimeMs(runtimeMs);

    const RC rc_VkStress = RunTest(&graphicsTest);

    switch (rc_VkStress)
    {
        case RC::OK:
            return QUAL_OK;
        case RC::GPU_STRESS_TEST_FAILED:
            return QUAL_DATA_ERROR;
        default:
            return QUAL_RUNTIME_ERROR;
    }
}
