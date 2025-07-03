#include "BusGrind.h"
#include "lwml.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

#include <cstdlib>
#include <cstring>
#include "BusGrindMain.h"

/*****************************************************************************/
BusGrind::BusGrind()
{
    m_infoStruct.name = BG_PLUGIN_NAME;
    m_infoStruct.shortDescription = "This plugin will exercise the PCIe bus for a given list of GPUs.";
    m_infoStruct.testGroups = "Perf";
    m_infoStruct.selfParallel = true;
    m_infoStruct.logFileTag = BG_PLUGIN_LF_NAME;

    TestParameters *tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "True");
    tp->AddString(BG_STR_TEST_PINNED, "True");
    tp->AddString(BG_STR_TEST_UNPINNED, "True");
    tp->AddString(BG_STR_TEST_P2P_ON, "True");
    tp->AddString(BG_STR_TEST_P2P_OFF, "True");

    tp->AddString(BG_STR_IS_ALLOWED, "False");

    tp->AddDouble(BG_STR_MAX_PCIE_REPLAYS, 80.0, 1.0, 1000000.0);

    tp->AddDouble(BG_STR_MAX_MEMORY_CLOCK, 0.0, 0.0, 100000.0);
    tp->AddDouble(BG_STR_MAX_GRAPHICS_CLOCK, 0.0, 0.0, 100000.0);
    // CRC_ERROR_THRESHOLD is the number of CRC errors per second, per RM recommendation
    tp->AddDouble(BG_STR_CRC_ERROR_THRESHOLD, 100.0, 0.0, 1000000.0);
    tp->AddString(BG_STR_LWSWITCH_NON_FATAL_CHECK, "False");

    /* Bandwidth parameters' allowed range */
    double minBwIterations = 1.0;
    double maxBwIterations = 1000000000000.0;
    double minBwIntsPerCopy = 1.0;
    double maxBwIntsPerCopy = 1000000000.0;

    /* latency parameters' allowed range */
    double minLatIterations = 1.0;
    double maxLatIterations = 1000000000000.0;

    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_INTS_PER_COPY,
                         10000000.0, minBwIntsPerCopy, maxBwIntsPerCopy);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_ITERATIONS,
                         50.0, minBwIterations, maxBwIterations);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_BANDWIDTH,
                         0.0, 0.0, 1000.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_GEN,
                         1.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_PINNED, BG_STR_MIN_PCI_WIDTH,
                         1.0, 1.0, 16.0);

    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_INTS_PER_COPY,
                         10000000.0, minBwIntsPerCopy, maxBwIntsPerCopy);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_ITERATIONS,
                         50.0, minBwIterations, maxBwIterations);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_BANDWIDTH,
                         0.0, 0.0, 1000.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_GEN,
                         1.0, 0.0, 3.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED, BG_STR_MIN_PCI_WIDTH,
                         1.0, 1.0, 16.0);

    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_CONLWRRENT_PINNED, BG_STR_INTS_PER_COPY,
                         10000000.0, minBwIntsPerCopy, maxBwIntsPerCopy);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_CONLWRRENT_PINNED, BG_STR_ITERATIONS,
                         50.0, minBwIterations, maxBwIterations);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_CONLWRRENT_PINNED, BG_STR_MIN_BANDWIDTH,
                         0.0, 0.0, 1000.0);

    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_CONLWRRENT_UNPINNED, BG_STR_INTS_PER_COPY,
                         10000000.0, minBwIntsPerCopy, maxBwIntsPerCopy);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_CONLWRRENT_UNPINNED, BG_STR_ITERATIONS,
                         50.0, minBwIterations, maxBwIterations);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_CONLWRRENT_UNPINNED, BG_STR_MIN_BANDWIDTH,
                         0.0, 0.0, 1000.0);

    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_LATENCY_PINNED, BG_STR_ITERATIONS,
                         5000.0, minLatIterations, maxLatIterations);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_LATENCY_PINNED, BG_STR_MAX_LATENCY,
                         100000.0, 0.0, 1000000.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_LATENCY_PINNED, BG_STR_MIN_BANDWIDTH,
                         0.0, 0.0, 1000.0);

    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_LATENCY_UNPINNED, BG_STR_ITERATIONS,
                         5000.0, minLatIterations, maxLatIterations);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_LATENCY_UNPINNED, BG_STR_MAX_LATENCY,
                         100000.0, 0.0, 1000000.0);
    tp->AddSubTestDouble(BG_SUBTEST_H2D_D2H_LATENCY_UNPINNED, BG_STR_MIN_BANDWIDTH,
                         0.0, 0.0, 1000.0);

    tp->AddSubTestDouble(BG_SUBTEST_P2P_BW_P2P_ENABLED, BG_STR_INTS_PER_COPY,
                         10000000.0, minBwIntsPerCopy, maxBwIntsPerCopy);
    tp->AddSubTestDouble(BG_SUBTEST_P2P_BW_P2P_ENABLED, BG_STR_ITERATIONS,
                         50.0, minBwIterations, maxBwIterations);

    tp->AddSubTestDouble(BG_SUBTEST_P2P_BW_P2P_DISABLED, BG_STR_INTS_PER_COPY,
                         10000000.0, minBwIntsPerCopy, maxBwIntsPerCopy);
    tp->AddSubTestDouble(BG_SUBTEST_P2P_BW_P2P_DISABLED, BG_STR_ITERATIONS,
                         50.0, minBwIterations, maxBwIterations);

    tp->AddSubTestDouble(BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_ENABLED, BG_STR_INTS_PER_COPY,
                         10000000.0, minBwIntsPerCopy, maxBwIntsPerCopy);
    tp->AddSubTestDouble(BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_ENABLED, BG_STR_ITERATIONS,
                         50.0, minBwIterations, maxBwIterations);

    tp->AddSubTestDouble(BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_DISABLED, BG_STR_INTS_PER_COPY,
                         10000000.0, minBwIntsPerCopy, maxBwIntsPerCopy);
    tp->AddSubTestDouble(BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_DISABLED, BG_STR_ITERATIONS,
                         50.0, minBwIterations, maxBwIterations);


    tp->AddSubTestDouble(BG_SUBTEST_1D_EXCH_BW_P2P_ENABLED, BG_STR_INTS_PER_COPY,
                         10000000.0, minBwIntsPerCopy, maxBwIntsPerCopy);
    tp->AddSubTestDouble(BG_SUBTEST_1D_EXCH_BW_P2P_ENABLED, BG_STR_ITERATIONS,
                         50.0, minBwIterations, maxBwIterations);

    tp->AddSubTestDouble(BG_SUBTEST_1D_EXCH_BW_P2P_DISABLED, BG_STR_INTS_PER_COPY,
                         10000000.0, minBwIntsPerCopy, maxBwIntsPerCopy);
    tp->AddSubTestDouble(BG_SUBTEST_1D_EXCH_BW_P2P_DISABLED, BG_STR_ITERATIONS,
                         50.0, minBwIterations, maxBwIterations);

    tp->AddSubTestDouble(BG_SUBTEST_P2P_LATENCY_P2P_ENABLED, BG_STR_ITERATIONS,
                         5000.0, minLatIterations, maxLatIterations);

    tp->AddSubTestDouble(BG_SUBTEST_P2P_LATENCY_P2P_DISABLED, BG_STR_ITERATIONS,
                         5000.0, minLatIterations, maxLatIterations);


    m_infoStruct.defaultTestParameters = tp;
}

/*****************************************************************************/
void BusGrind::Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList)
{
    InitializeForGpuList(gpuList);

    if (!testParameters->GetBoolFromString(BG_STR_IS_ALLOWED))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, BG_PLUGIN_NAME);
        AddError(d);
        SetResult(LWVS_RESULT_SKIP);
        return;
    }

    int st = main_entry(gpuList, this, testParameters);
    if (main_should_stop)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
        AddError(d);
        SetResult(LWVS_RESULT_SKIP);
    }
    else if (st)
    {
        // Fatal error in plugin or test could not be initialized
        SetResult(LWVS_RESULT_FAIL);
    }
}

/*****************************************************************************/
extern "C" {
    Plugin *maker() {
        return new BusGrind;
    }
    class proxy {
    public:
        proxy()
        {
            factory["BusGrind"] = maker;
        }
    };    
    proxy p;
}                                            

