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

#include <sys/wait.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>

#include "interprocessproducer.h"
#include "interprocessconsumer.h"

// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0

#include "gtest/gtest.h"

#define PRINTF(...)                                                            \
    do {                                                                       \
        printf("[          ] ");                                               \
        printf(__VA_ARGS__);                                                   \
    } while (0)

#define PID_MAX 32768

// C++ stream interface
class TestCout : public std::stringstream
{
public:
    ~TestCout()
    {
        PRINTF("%s\n", str().c_str());
    }
};

#define TEST_COUT TestCout()

//==========================================================================
// Define IpcStreaming test suite.
//==========================================================================
class IpcStreaming :
    public ::testing::Test
{
public:
    pid_t pid {PID_MAX}, pid1 {PID_MAX}, pid2 {PID_MAX};
    int wait_for_child_fork(pid_t pid);

protected:
    void TearDown()
    {
        if (pid == 0) {
            // Cleanup already done do nothing
            // WAR: exit here so we do not have duplicate output
            // https://github.com/google/googletest/issues/1153
            exit(testing::Test::HasFailure());
        } else if (pid1 == 0) {
            // Cleanup already done do nothing
            // WAR: exit here so we do not have duplicate output
            // https://github.com/google/googletest/issues/1153
            exit(testing::Test::HasFailure());
        } else if (pid2 == 0) {
            // Cleanup already done do nothing
            // WAR: exit here so we do not have duplicate output
            // https://github.com/google/googletest/issues/1153
            exit(testing::Test::HasFailure());
        } else {
            // Cleanup already done do nothing
        }
    }
};

int IpcStreaming::wait_for_child_fork(pid_t pid)
{
    int status;

    // Wait till child exits and get the exit status
    if (0 > waitpid(pid, &status, 0)) {
        TEST_COUT << "waitpid() error";
        return -1;
    }

    // Analyze the exit status
    if (WIFEXITED(status)) {
        const int exitStatus = WEXITSTATUS(status);
        if (exitStatus != 0) {
            TEST_COUT << "Non-zero exit status " << exitStatus
                      << " from child";
        }
        return exitStatus;
    } else {
        TEST_COUT << "Abnormal exit from child";
        return -2;
    }
}


TEST_F(IpcStreaming, MailboxStreaming)
{
    pid = fork();
    ASSERT_GE(pid, 0) << "Fork failed";

    if (pid == 0) {
        // Forked Child process

        // To prevent having 2 copies of cons object created due to fork,
        // declare it here. This will also ensure proper cleanup of cons
        // resources by destructor before child ends.
        InterProcessConsumer cons("lwscistream_1");

        cons.SetUp();
        cons.createBlocks(QueueType::Mailbox);
        cons.createStream();
        cons.disconnectStream();
        cons.TearDown();
    } else {
        // Parent process
        const char *prodChannels[] = {"lwscistream_0"};
        InterProcessProducer prod(prodChannels);

        prod.SetUp();
        prod.createBlocks();
        prod.createStream();
        prod.disconnectStream();
        prod.TearDown();

        // Wait for child to exit
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

TEST_F(IpcStreaming, MailboxMulticastStreaming)
{
    pid = fork();
    ASSERT_GE(pid, 0) << "Fork failed";

    if (pid == 0) {
        // Forked Child process

        // To prevent having 2 copies of cons object created due to fork,
        // declare it here. This will also ensure proper cleanup of cons
        // resources by destructor before child ends.
        InterProcessConsumer cons("lwscistream_1");

        cons.SetUp();
        cons.createBlocks(QueueType::Mailbox, 2U);
        cons.createStream();
        cons.disconnectStream();
        cons.TearDown();
    } else {
        // Parent process
        const char *prodChannels[] = {"lwscistream_0"};
        InterProcessProducer prod(prodChannels);

        prod.SetUp();
        prod.createBlocks();
        prod.createStream();
        prod.disconnectStream();
        prod.TearDown();

        // Wait for child to exit
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

TEST_F(IpcStreaming, MulticastStreaming)
{
    (pid1 = fork()) && (pid2 = fork()); // Creates 2 child processes

    if (pid1 == 0) {
        // Forked Child process 1

        // To prevent having 2 copies of cons1 object created due to fork,
        // declare it here. This will also ensure proper cleanup of cons1
        // resources by destructor before child ends.
        InterProcessConsumer cons1("lwscistream_1");

        cons1.SetUp();
        cons1.createBlocks(QueueType::Mailbox, 2U);
        cons1.createStream();
        cons1.disconnectStream();
        cons1.TearDown();
    } else if(pid2 == 0) {
        // Forked Child process 2

        // To prevent having 2 copies of cons2 object created due to fork,
        // declare it here. This will also ensure proper cleanup of cons2
        // resources by destructor before child ends.
        InterProcessConsumer cons2("lwscistream_3");

        cons2.SetUp();
        cons2.createBlocks(QueueType::Fifo, 2U);
        cons2.createStream();
        cons2.disconnectStream();
        cons2.TearDown();
    } else {
        // Parent process
        ASSERT_GE(pid1, 0) << "Fork failed";
        ASSERT_GE(pid2, 0) << "Fork failed";

        const char *prodChannels[] = {"lwscistream_0", "lwscistream_2"};
        InterProcessProducer prod(prodChannels, 2U);

        prod.SetUp();
        prod.createBlocks();
        prod.createStream();
        prod.disconnectStream();
        prod.TearDown();

        // Wait for childs to exit
        ASSERT_EQ(0, wait_for_child_fork(pid1));
        ASSERT_EQ(0, wait_for_child_fork(pid2));
    }
}
