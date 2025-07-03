//! \file
//! \brief LwSciStream perf test definition.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include <pthread.h>
#include "perftest.h"

extern TestArg testArg;

//==================== PerfTest ====================//

static void* runConsumer(void *arg)
{
    ConsArg *consArg = static_cast<ConsArg *>(arg);
    assert(consArg != nullptr);

    PerfConsumer cons(consArg->bufModule, consArg->syncModule);
    cons.createStream(0U, consArg->upstreamBlock);
    cons.run();

    return nullptr;
}

PerfTest::PerfTest(LwSciBufModule buf,
                   LwSciSyncModule sync):
    bufModule(buf),
    syncModule(sync)
{
}

void PerfTest::run(void)
{
    // Create producer
    PerfProducer* prod = new PerfProducer(bufModule, syncModule);
    LwSciStreamBlock upstreamBlock = prod->createStream();

    // Create consumer threads
    std::vector<pthread_t> consTid(testArg.numConsumers);
    std::vector<ConsArg *> consArg(testArg.numConsumers, nullptr);
    for (uint32_t i{ 0U }; i < testArg.numConsumers; i++) {
        consArg[i] = new ConsArg(bufModule, syncModule, upstreamBlock);
        pthread_create(&consTid[i], nullptr, runConsumer, consArg[i]);
    }

    // Run producer
    prod->run();
    delete prod;

    // Consumer thread join
    for (uint32_t i{ 0U }; i < consTid.size(); i++) {
        pthread_join(consTid[i], nullptr);
        delete consArg[i];
    }
}

//==================== PerfTestProd ====================//

PerfTestProd::PerfTestProd(std::vector<LwSciIpcEndpoint>& ipc,
                           LwSciBufModule buf,
                           LwSciSyncModule sync) :
    bufModule(buf),
    syncModule(sync),
    ipcEndpoint(ipc)
{
}

void PerfTestProd::run(void)
{
    PerfProducer prod(bufModule, syncModule);
    LwSciStreamBlock upstreamBlock = prod.createStream(&ipcEndpoint);
    prod.run();
}

//==================== PerfTestCons ====================//

PerfTestCons::PerfTestCons(LwSciIpcEndpoint ipc,
                           LwSciBufModule buf,
                           LwSciSyncModule sync) :
    bufModule(buf),
    syncModule(sync),
    ipcEndpoint(ipc)
{
}

void PerfTestCons::run(void)
{
    PerfConsumer cons(bufModule, syncModule);
    cons.createStream(ipcEndpoint, 0U);
    cons.run();
}