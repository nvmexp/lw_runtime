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
namespace po = boost::program_options;

namespace lib_vkstress
{
    int vkstress_wrapper(int argc, char const* argv[], uint32_t& early_exit)
    {
        static bool platformInitialized = false;
        if (!platformInitialized)
        {
            PlatformOnEntry();
            platformInitialized = true;
        }
        SetEarlyExit(&early_exit);

        po::options_description optionsDescription("Arguments");
        po::variables_map variablesMap;

        unsigned int runtimeSec = 10;
        unsigned int gpuIndex = 0;

        optionsDescription.add_options()
            ("adapter", po::value<unsigned int>(&gpuIndex), "GPU index")
            ("time_to_run", po::value<unsigned int>(&runtimeSec), "Runtime in seconds")
            ;

        try
        {
            po::store(po::command_line_parser(argc, argv).
                      options(optionsDescription).run(), variablesMap);
            po::notify(variablesMap);
        }
        catch (const po::error &x)
        {
            fprintf(stderr, "Command line error: %s\n", x.what());
            return 1;
        }

        VkStress test;

        test.GetTestConfiguration()->SetDisplayWidth(512);
        test.GetTestConfiguration()->SetDisplayHeight(400);
        test.GetTestConfiguration()->SetGpuIndex(gpuIndex);
        test.SetRuntimeMs(1000*runtimeSec);

        test.SetUseRandomTextureData(true);
        test.SetMaxTextureSize(512);
        test.SetUseHTex(false);
        test.SetMaxPower(true);
        test.SetPulseMode(VkStress::RANDOM);

        StickyRC rc = test.Setup();
        if (rc == RC::OK)
        {
            rc = test.Run();
        }
        rc = test.Cleanup();
        return (rc == RC::OK) ? 0 : 1;
    }
}
