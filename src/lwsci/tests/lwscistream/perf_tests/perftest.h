//! \file
//! \brief LwSciStream perf test declaration.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef PERFPTEST_H
#define PERFPTEST_H

#include <vector>
#include "perfproducer.h"
#include "perfconsumer.h"

struct ConsArg {
    ConsArg(LwSciBufModule buf,
            LwSciSyncModule sync,
            LwSciStreamBlock upBlk) :
        bufModule(buf),
        syncModule(sync),
        upstreamBlock(upBlk){ };

    LwSciBufModule      bufModule{ nullptr };
    LwSciSyncModule     syncModule{ nullptr };
    LwSciStreamBlock    upstreamBlock{ 0U };
};

class PerfTest
{
public:
    PerfTest(LwSciBufModule buf,
             LwSciSyncModule sync);
    virtual ~PerfTest(void) = default;
    void run(void);

private:
    LwSciBufModule                  bufModule{ nullptr };
    LwSciSyncModule                 syncModule{ nullptr };
};


class PerfTestProd
{
public:
    PerfTestProd(std::vector<LwSciIpcEndpoint>& ipc,
                 LwSciBufModule buf,
                 LwSciSyncModule sync);
    virtual ~PerfTestProd(void) = default;
    void run(void);

private:
    LwSciBufModule                  bufModule;
    LwSciSyncModule                 syncModule;

    std::vector<LwSciIpcEndpoint>   ipcEndpoint;
};


class PerfTestCons
{
public:
    PerfTestCons(LwSciIpcEndpoint ipc,
                 LwSciBufModule buf,
                 LwSciSyncModule sync);
    virtual ~PerfTestCons(void) = default;
    void run(void);

private:
    void createStream(void);

    LwSciBufModule     bufModule;
    LwSciSyncModule    syncModule;

    LwSciIpcEndpoint   ipcEndpoint;
};

#endif // PERFPTEST_H
