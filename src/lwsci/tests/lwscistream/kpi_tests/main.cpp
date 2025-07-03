//! \file
//! \brief LwSciStream kpi perf test.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include <string>
#include <unordered_map>

#include "test_stream_setup.h"
#include "test_sync_setup.h"
#include "test_buf_setup.h"
#include "test_streaming.h"

template<typename T> StreamTest * createTest() { return new T; }

typedef std::unordered_map<std::string, StreamTest*(*)()> TestMap;

TestMap testMap{
    { "ProducerCreate", &createTest<ProducerCreate>},
    { "ConsumerCreate", &createTest<ConsumerCreate> },
    { "PoolCreate", &createTest<PoolCreate> },
    { "FifoQueueCreate", &createTest<FifoQueueCreate> }
};

TestMap prodTestMap{
    { "IpcSrcCreate", &createTest<IpcSrcCreateProd> },
    { "IpcDstCreate", &createTest<IpcDstCreateProd> },
    { "ConnectProd2IpcSrc", &createTest<ConnectProd2IpcSrcProd> },
    { "ConnectIpcDst2Cons", &createTest<ConnectIpcDst2ConsProd> },
    { "SyncAttr", &createTest<SyncAttrProd> },
    { "SyncObj", &createTest<SyncObjProd> },
    { "ConsumerElements", &createTest<ElementsProd> },
    { "ReconciledElements", &createTest<ReconciledElementsProd> },
    { "PacketCreate", &createTest<PacketCreateProd> },
    { "PacketStatus", &createTest<PacketStatusProd> },
    { "ProducerGet", &createTest<ProducerGetProd> },
    { "ConsumerAcquire", &createTest<ConsumerAcquireProd> },
    { "ProducerPresent", &createTest<ProducerPresentProd> },
    { "ConsumerRelease", &createTest<ConsumerReleaseProd> }
};

TestMap consTestMap{
    { "IpcSrcCreate", &createTest<IpcSrcCreateCons> },
    { "IpcDstCreate", &createTest<IpcDstCreateCons> },
    { "ConnectProd2IpcSrc", &createTest<ConnectProd2IpcSrcCons> },
    { "ConnectIpcDst2Cons", &createTest<ConnectIpcDst2ConsCons> },
    { "SyncAttr", &createTest<SyncAttrCons> },
    { "SyncObj", &createTest<SyncObjCons> },
    { "ConsumerElements", &createTest<ElementsCons> },
    { "ReconciledElements", &createTest<ReconciledElementsCons> },
    { "PacketCreate", &createTest<PacketCreateCons> },
    { "PacketStatus", &createTest<PacketStatusCons> },
    { "ProducerGet", &createTest<ProducerGetCons> },
    { "ConsumerAcquire", &createTest<ConsumerAcquireCons> },
    { "ProducerPresent", &createTest<ProducerPresentCons> },
    { "ConsumerRelease", &createTest<ConsumerReleaseCons> }
};

static void help(void)
{
    printf("\n============================================"\
        "\n LwSciStream KPI:"                              \
        "\n -h: Print this message."                       \
        "\n Process Mode: default is single process"       \
        "\n -p: Producer process."                         \
        "\n -c: Consumer process."                         \
        "\n Specify test name:"                            \
        "\n -t <test_name>: Supported tests:\n");
    for (TestMap::iterator itr = testMap.begin();
         itr != testMap.end();
         itr++) {
        printf("      %s\n", itr->first.c_str());
    }
    for (TestMap::iterator itr = prodTestMap.begin();
         itr != prodTestMap.end();
         itr++) {
        printf("      %s\n", itr->first.c_str());
    }
}

enum TestProcMode {
    SingleProc,
    ProdProc,
    ConsProc
};

int main(int argc, char **argv)
{
    TestProcMode procMode = SingleProc;
    std::string testName = "";
    StreamTest *streamTest = nullptr;

    if (argc < 2) {
        printf("Too few arguments.\n");
        goto fail;
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-t")) {
            // Run a specific test
            if (++i == argc) {
                printf("Incorrect arguments.\n");
                goto fail;
            }
            testName = argv[i];
        } else if (!strcmp(argv[i], "-p")) {
            procMode = ProdProc;
        } else if (!strcmp(argv[i], "-c")) {
            procMode = ConsProc;
        } else if (!strcmp(argv[i], "-h")) {
            help();
        } else {
            printf("Incorrect arguments.\n");
            goto fail;
        }
    }

    switch (procMode)
    {
    case SingleProc:
        if (testMap.find(testName) != testMap.end()) {
            streamTest = testMap[testName]();
        }
        break;
    case ProdProc:
        if (prodTestMap.find(testName) != prodTestMap.end()) {
            streamTest = prodTestMap[testName]();
        }
        break;
    case ConsProc:
        if (consTestMap.find(testName) != consTestMap.end()) {
            streamTest = consTestMap[testName]();
        }
        break;
    default:
        break;
    }

    if (streamTest == nullptr) {
        printf("Unsupported test case.\n");
        goto fail;
    }

    streamTest->run();
    delete streamTest;

    return 0;

fail:
    help();
    return -1;
}
