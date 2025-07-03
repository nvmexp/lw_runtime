/*
* LWIDIA_COPYRIGHT_BEGIN
*
* Copyright 2017-2022 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* LWIDIA_COPYRIGHT_END
*/

#include "tests/vkfusion.h"
#include "tests/vktriangle.h"
#include "tests/vkstress.h"
#include "util_init.hpp"
#include "vktest_platform.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <string>
#include <sstream>

RC RunTest(const char *name, GpuTest *test)
{
    StickyRC rc;
    Printf(Tee::PriNormal, "%s Start.\n", name);

    sample_info info = {};
    init_window_size(info,
        test->GetTestConfiguration()->DisplayWidth(),
        test->GetTestConfiguration()->DisplayHeight());
    info.name = name;
    init_window(info);

    test->SetWindowParams(info.connection, info.window);

    CHECK_RC_MSG(test->Setup(), "Setup failed");
    // Even if Run() fails we still need to call Cleanup()
    rc = test->Run();

    rc = test->Cleanup();

    destroy_window(info);

    Printf(Tee::PriNormal, "%s %s.\n", name, (rc != OK) ? "FAILED:" : "SUCCESS");

    return rc;
}

int main(int argc, char const*argv[])
{
    PlatformOnEntry();

    po::options_description optionsDescription("VkTest usage");
    po::variables_map variablesMap;

    unsigned int bufferCheckMs = VkStress::BUFFERCHECK_MS_DISABLED;
    unsigned int framesPerSubmit = 1;
    unsigned int height = 0;
    unsigned int presentMs = 0;
    unsigned int runtimeMs = 0;
    unsigned int test = 0;
    unsigned int width = 0;

    optionsDescription.add_options()
        ("b,b", po::value<unsigned int>(&bufferCheckMs), "BufferCheckMs")
        ("d,d", "Enable debug markers")
        ("f,f", po::value<unsigned int>(&framesPerSubmit), "Frames per submit")
        ("h,h", po::value<unsigned int>(&height), "Render target height")
        ("p,p", po::value<unsigned int>(&presentMs), "Present time (milliseconds)")
        ("r,r", po::value<unsigned int>(&runtimeMs), "Runtime (milliseconds)")
        ("t,t", po::value<unsigned int>(&test), "Select a test number")
        ("w,w", po::value<unsigned int>(&width), "Render target width")
        ("help", "Show this help message")
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
        return 1;
    }

    if (variablesMap.count("help"))
    {
        std::ostringstream oss;
        oss << optionsDescription;
        std::string str = oss.str();
        Printf(Tee::PriNormal, "%s", str.c_str());
        return 1;
    }
    VulkanDevice::EnableExt("VK_KHR_swapchain");
    if ((test == 0) || (test == 267))
    {
        VkStress stressTest;
        stressTest.SetEnableValidation(true);
        stressTest.SetBufferCheckMs(bufferCheckMs);
        stressTest.SetFramesPerSubmit(framesPerSubmit);
        if (height)
            stressTest.GetTestConfiguration()->SetDisplayHeight(height);
        if (presentMs)
            stressTest.SetDetachedPresentMs(presentMs);
        if (runtimeMs)
            stressTest.SetRuntimeMs(runtimeMs);
        if (width)
            stressTest.GetTestConfiguration()->SetDisplayWidth(width);
        if (variablesMap.count("d"))
            stressTest.SetEnableDebugMarkers(true);

        RC rc_VkStress = RunTest("VkStress", &stressTest);
        if (rc_VkStress != OK)
        {
            Printf(Tee::PriNormal, "VkStress reported error:%d\n", rc_VkStress.Get());
            return 1;
        }
    }
    if (test == 21)
    {
        VkFusion::Test toysTest;
        toysTest.SetEnableValidation(true);
        toysTest.GetTestConfiguration()->SetDisplayWidth(1024);
        toysTest.GetTestConfiguration()->SetDisplayHeight(1024);
        toysTest.m_Raytracing.SetRunMask(1);
        toysTest.m_Raytracing.SetSurfaceWidth(width ? width : 4096);
        toysTest.m_Raytracing.SetSurfaceHeight(height ? height : 4096);
        toysTest.m_Mats.SetRunMask(2);
        toysTest.m_Mats.SetMaxFbMb(64);
        toysTest.m_Graphics.SetRunMask(4);
        toysTest.m_Graphics.SetSurfaceWidth(width ? width : 1024);
        toysTest.m_Graphics.SetSurfaceHeight(height ? height : 1024);
        toysTest.SetRuntimeMs(runtimeMs ? runtimeMs : 5000);
        toysTest.SetDetachedPresentMs(presentMs);
        RC rc_VkFusion = RunTest("VkFusion", &toysTest);
        if (rc_VkFusion != OK)
        {
            Printf(Tee::PriNormal, "VkFusion reported error:%d\n", rc_VkFusion.Get());
            return 1;
        }
    }
    if ((test == 0) || (test == 266))
    {
        VkTriangle triangleTest;
        if (height)
            triangleTest.GetTestConfiguration()->SetDisplayHeight(height);
        if (width)
            triangleTest.GetTestConfiguration()->SetDisplayWidth(width);
        RC rc_VkTriangle = RunTest("VkTriangle", &triangleTest);
        if (rc_VkTriangle != OK)
        {
            Printf(Tee::PriNormal, "VkTriangle reported error:%d\n", rc_VkTriangle.Get());
            return 1;
        }
    }

    INT32 errorCount = ErrorLogger::GetErrorCount();
    if (errorCount > 0)
    {
        Printf(Tee::PriNormal, "VkTest ErrorLogger logged %d errors.\n", errorCount);
        return 1;
    }

    return 0;
}
