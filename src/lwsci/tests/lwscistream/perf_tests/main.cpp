//! \file
//! \brief LwSciStream perf test main.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include <limits>
#include <algorithm>
#include <cmath>
#include <unistd.h>
#include <string>
#include "util.h"
#include "perftest.h"

TestArg             testArg{};

std::vector<double> presentIntervals;
std::vector<double> presentIntervals2;
std::vector<double> releaseIntervals;

static void runTest(std::vector<std::string>& ipcEpNames);
static void help(void);
static void perfAnalyze(std::vector<double>& ret);

int main(int argc, char **argv)
{
    int32_t opt;
    std::vector<std::string> ipcEpNames(MAX_CONS);

    while ((opt = getopt(argc, argv, "pP:c:C:n:f:b:s:r:a:m:lvh")) != -1) {
        switch (opt) {
        case 'p':
            testArg.testType = CrossProcProd;
            break;
        case 'P':
            testArg.testType = CrossProcProd;
            testArg.isC2c = true;
            assert(atoi(optarg) < MAX_CONS);
            ipcEpNames[atoi(optarg)] = argv[optind++];
            break;
        case 'c':
            testArg.testType = CrossProcCons;
            testArg.consIndex = atoi(optarg);
            break;
        case 'C':
            testArg.testType = CrossProcCons;
            testArg.consIndex = atoi(optarg);
            testArg.isC2c = true;
            assert(atoi(optarg) < MAX_CONS);
            ipcEpNames[testArg.consIndex] = argv[optind++];
            break;
        case 'n':
        {
            testArg.numConsumers = atoi(optarg);
            if ((testArg.numConsumers == 0U) ||
                (testArg.numConsumers > MAX_CONS)) {
                printf("The number of consumers is not supported. " \
                       "It should be in [1, %d].\n", MAX_CONS);
                help();
                return 0;
            }
            break;
        }
        case 'f':
            testArg.numFrames = atoi(optarg);
            break;
        case 'b':
            testArg.bufSize = atof(optarg) * 1024U * 1024U;
            break;
        case 's':
            testArg.numSyncs = atoi(optarg);
            if (testArg.numSyncs > NUM_ELEMENTS) {
                testArg.numSyncs = NUM_ELEMENTS;
            }
            break;
        case 'r':
            testArg.sleepUs = 1000000U / atoi(optarg);
            break;
        case 'l':
            testArg.latency = true;
            break;
        case 'v':
            testArg.verbose = true;
            break;
        case 'a':
            testArg.avgTarget = atof(optarg);
            break;
        case 'm':
            testArg.maxTarget = atof(optarg);
            break;
        default:
            help();
            return 0;
        }
    }

    runTest(ipcEpNames);

    return 0;
}

static void runTest(std::vector<std::string>& ipcEpNames)
{
    // Open sync/buf modules
    LwSciBufModule bufModule{ nullptr };
    CHECK_LWSCIERR(LwSciBufModuleOpen(&bufModule));

    LwSciSyncModule syncModule{ nullptr };
    CHECK_LWSCIERR(LwSciSyncModuleOpen(&syncModule));

    switch (testArg.testType) {
    case IntraProcess:
    {
        PerfTest perfTest(bufModule, syncModule);
        perfTest.run();

        break;
    }
    case CrossProcProd:
    {
        // Init ipc channel
        CHECK_LWSCIERR(LwSciIpcInit());

        std::vector<LwSciIpcEndpoint> ipcEndpoint(testArg.numConsumers, 0U);
        for (uint32_t i = 0U; i < testArg.numConsumers; i++) {
            if (testArg.isC2c) {
                CHECK_LWSCIERR(
                    LwSciIpcOpenEndpoint(ipcEpNames[i].c_str(),
                                         &ipcEndpoint[i]));
            } else {
                char ipcEpName[32];
                sprintf(ipcEpName, "%s%d", "lwscistream_", 2 * i);
                CHECK_LWSCIERR(
                    LwSciIpcOpenEndpoint(ipcEpName, &ipcEndpoint[i]));
            }
            LwSciIpcResetEndpoint(ipcEndpoint[i]);
        }

        PerfTestProd* prodTest =
            new PerfTestProd(ipcEndpoint, bufModule, syncModule);

        prodTest->run();

        delete prodTest;

        // Deinit ipc
        LwSciIpcDeinit();

        break;
    }
    case CrossProcCons:
    {
        // Init ipc channel
        CHECK_LWSCIERR(LwSciIpcInit());

        LwSciIpcEndpoint ipcEndpoint{ 0U };
        if (testArg.isC2c) {
            CHECK_LWSCIERR(
                LwSciIpcOpenEndpoint(ipcEpNames[testArg.consIndex].c_str(),
                                     &ipcEndpoint));
        } else {
            char ipcEpName[32];
            sprintf(ipcEpName, "%s%d",
                    "lwscistream_", 2 * testArg.consIndex + 1);
            CHECK_LWSCIERR(LwSciIpcOpenEndpoint(ipcEpName, &ipcEndpoint));
        }
        LwSciIpcResetEndpoint(ipcEndpoint);

        PerfTestCons* consTest =
            new PerfTestCons(ipcEndpoint, bufModule, syncModule);

        consTest->run();

        delete consTest;

        // Deinit ipc
        LwSciIpcDeinit();

        break;
    }
    default:
        help();
        break;
    }

    // Close buf/sync module
    if (bufModule != nullptr) {
        LwSciBufModuleClose(bufModule);
        bufModule = nullptr;
    }
    if (syncModule != nullptr) {
        LwSciSyncModuleClose(syncModule);
        syncModule = nullptr;
    }

    // Performance analysis
    if (testArg.latency) {
        if (presentIntervals.size() > 0) {
            printf("\nProducer present:\n");
            perfAnalyze(presentIntervals);
        }
        if (testArg.isC2c && presentIntervals2.size() > 0) {
            printf("\nProducer present + C2C copy done:\n");
            perfAnalyze(presentIntervals2);
        }
        if (!testArg.isC2c && releaseIntervals.size() > 0) {
            printf("\nConsumer release:\n");
            perfAnalyze(releaseIntervals);
        }
    }
}

static void help(void)
{
    printf("\n==============================================================="\
           "\n LwSciStream Perf Test App: Intra-process by default."          \
           "\n [-h]                Print this usage.\n"                       \
           "\n [-p]                Inter-process producer."                   \
           "\n [-P] <N> <s>        Inter-SoC producer, LwSciIpc endpoint name"\
           " connected to indexed consumer."                                  \
           "\n [-c] <N>            Inter-process consumer, consumer index."   \
           "\n [-C] <N> <s>        Inter-SoC consumer, consumer index and."   \
           " LwSciIpc endpoint name for this consumer."                       \
           "\n [-n] <N>            Multicast stream with N Consumer(s)."      \
           "\n                     Default: 1."                               \
           "\n                     (Set by producer.)"                        \
           "\n [-f] <N>            Number of frames. Default: 100."           \
           "\n                     (Set by producer.)"                        \
           "\n [-b] <N>            Buffer size: N MB. Default: 1MB."          \
           "\n                     (Set by producer.)"                        \
           "\n [-s] <N>            Number of sync objects per client."        \
           "\n                     Default: 1."                               \
           "\n [-r] <N>            Producer present rate at N Hz."            \
           "\n [-l]                Measure latency. Default: false."          \
           "\n                     (Set by both producer and consumer.)"      \
           "\n Following options will be ignored if not measuring latency:"   \
           "\n [-v]                Verbose. Print latency for each packet and"\
           "\n                     each consumer. Default: false."            \
           "\n                     (Set by consumer.)"                        \
           "\n [-a] <num>          Average KPI target (us)."                  \
           "\n                     (Set by consumer.)"                        \
           "\n [-m] <num>          99.99 percentile KPI target (us)."         \
           "\n                     (Set by consumer.)"                        \
           );
}

static void perfAnalyze(std::vector<double>& ret)
{
    // Present time
    double total{ 0.0f };
    double best{ std::numeric_limits<double>::max() };
    double worst{ std::numeric_limits<double>::min() };
    uint32_t numMax{ 0U };

    for (uint32_t i{ 0U }; i < ret.size(); i++) {
        total += ret[i];
        best = std::min(ret[i], best);
        worst = std::max(ret[i], worst);
        // 5% tolerance range.
        if (testArg.maxTarget > 0.0f &&
            ret[i] > testArg.maxTarget * 1.05) {
            numMax++;
        }
        if (testArg.verbose) {
            printf("%8.5f\n", ret[i]);
        }
    }
    double avg{ total / (double)ret.size() };

    printf("* Best case:  %8.5f\n", best);
    printf("* Worst case: %8.5f\n", worst);
    printf("* Average:    %8.5f\n", avg);
    if (testArg.avgTarget > 0.0f) {
        printf("----------------------------------------\n");
        printf("**Target Average (us): %8.5f. \n", testArg.avgTarget);
        // 5% tolerance range.
        printf("  [%s]\n", avg <= testArg.avgTarget * 1.05 ?
               "Passed" : "Failed");
    }
    if (testArg.maxTarget > 0.0f) {
        printf("----------------------------------------\n");
        printf("**Target 99.99%% (us):  %8.5f. \n", testArg.maxTarget);
        printf("  Number of frames doesn't meet the 99.99%% Target: %d\n",
                numMax);
        // 99.99% of the test should meet the 99.99% target.
        printf("  [%s]\n", numMax <= std::ceil(ret.size() * 0.0001) ?
               "Passed" : "Failed");
    }
}
