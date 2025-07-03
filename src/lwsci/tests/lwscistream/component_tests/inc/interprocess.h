//! \file
//! \brief LwSciStream API testing.
//!
//! \copyright
//! Copyright (c) 2020-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef INTERPROCESS_H
#define INTERPROCESS_H

#include "gtest/gtest.h"
#include "util.h"

constexpr uint32_t MAX_CHANNELS = 4U;

class InterProcess
{
protected:
    LwSciSyncModule syncModule = nullptr;
    LwSciBufModule bufModule = nullptr;
    Endpoint ipc[MAX_CHANNELS];
    uint32_t channelCount = 1U;

public:
    void SetUp(void);
    void TearDown(void);
    InterProcess(void);
    ~InterProcess(void);

private:
    void initIpcChannel(void);
    void deinitIpcChannel(void);
};


InterProcess::InterProcess(void)
{
    memset(ipc, 0, sizeof(ipc));
}

InterProcess::~InterProcess(void)
{
    if (bufModule != nullptr) {
        LwSciBufModuleClose(bufModule);
        bufModule = nullptr;
    }

    if (syncModule != nullptr) {
        LwSciSyncModuleClose(syncModule);
        syncModule = nullptr;
    }

    deinitIpcChannel();
}

void InterProcess::initIpcChannel(void)
{
    ASSERT_EQ(LwSciError_Success, LwSciIpcInit()) << "LwSciIpcInit() Failed";

    for (uint32_t i = 0U; i < channelCount; ++i) {
        ASSERT_EQ(LwSciError_Success, LwSciIpcOpenEndpoint(ipc[i].chname,
                                                           &(ipc[i].endpoint)));
        LwSciIpcResetEndpoint(ipc[i].endpoint);
    }
}

void InterProcess::deinitIpcChannel(void)
{
    LwSciIpcDeinit();
}

// Open Buf and Sync module and init Ipc module
void InterProcess::SetUp(void)
{
    ASSERT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
    ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

    initIpcChannel();
}

// Teardown Buf and Sync modules
void InterProcess::TearDown()
{
    if (bufModule != nullptr) {
        LwSciBufModuleClose(bufModule);
        bufModule = nullptr;
    }

    if (syncModule != nullptr) {
        LwSciSyncModuleClose(syncModule);
        syncModule = nullptr;
    }
}

static bool isTrue(bool param) {return param;}

static void validateNumChannels(uint32_t numChannels)
{
    ASSERT_LE(numChannels, MAX_CHANNELS)
        << "Number of Ipc channels is more than the limit";
}

static void validateIpcChannel(const char *chname,
                               const char **validChannelsList)
{
    for (uint32_t i = 0U; i < MAX_CHANNELS; ++i) {
        if(!strcmp(chname, validChannelsList[i]))
            return;
    }

    FAIL() << "Invalid Ipc channel name";
}

#endif // !INTERPROCESS_H
