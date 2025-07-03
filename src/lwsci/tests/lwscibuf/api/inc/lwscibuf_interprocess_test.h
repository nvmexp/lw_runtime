/*
 * lwscibuf_interprocess_test.h
 *
 * Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#ifndef INCLUDED_LWSCIBUF_INTERPROCESS_TEST_H
#define INCLUDED_LWSCIBUF_INTERPROCESS_TEST_H

#include "lwscibuf_test_integration.h"
#include "lwscibuf_ipc_peer.h"

#include <memory>
#include <vector>

class LwSciBufInterProcessTest : public ::testing::Test
{
protected:

    void TearDown() override
    {
        for (const auto& peer : peers) {
            peer->TearDown();
        }

        if (pid != 0) {
            // WAR: exit here so we do not have duplicate output
            // https://github.com/google/googletest/issues/1153
            exit(testing::Test::HasFailure());
        }
    };

    static int wait_for_child_fork(int pid)
    {
        int status;
        if (0 > waitpid(pid, &status, 0)) {
            TEST_COUT << " Waitpid error!";
            return (-1);
        }
        if (WIFEXITED(status)) {
            const int exit_status = WEXITSTATUS(status);
            if (exit_status != 0) {
                TEST_COUT << "Non-zero exit status " << exit_status
                          << " from test!";
            }
            return exit_status;
        } else {
            TEST_COUT << " Non-normal exit from child!";
            return (-2);
        }
    };

    static void
    testAccessPermissions(std::shared_ptr<LwSciBufIpcPeer> upstreamPeer,
                          std::shared_ptr<LwSciBufIpcPeer> downstreamPeer,
                          std::shared_ptr<LwSciBufObjRefRec> bufObj,
                          LwSciBufAttrValAccessPerm accessTest,
                          const uint8_t* readData, size_t rdSize,
                          const uint8_t* writeData, size_t wrSize)
    {
        const uint8_t* rdPtr;
        uint8_t* wrPtr;
        /* get mem handle */
        LwSciBufRmHandle rmHandle;
        uint64_t offset, len, size;
        ASSERT_EQ(
            LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset, &len),
            LwSciError_Success);

        ASSERT_TRUE(CheckBufferAccessFlags(bufObj.get(), rmHandle))
            << "Access Permissions mismatch between Handle and Object.";

        if (accessTest == LwSciBufAccessPerm_Readonly) {
            NEGATIVE_TEST();
            ASSERT_EQ(LwSciBufObjGetCpuPtr(bufObj.get(), (void**)&wrPtr),
                      LwSciError_BadParameter);
        } else {
            ASSERT_EQ(LwSciBufObjGetCpuPtr(bufObj.get(), (void**)&wrPtr),
                      LwSciError_Success);
        }

        if (accessTest == LwSciBufAccessPerm_Readonly) {
            ASSERT_EQ(
                LwSciBufObjGetConstCpuPtr(bufObj.get(), (const void**)&rdPtr),
                LwSciError_Success);
        } else {
            rdPtr = wrPtr;
        }

        if (downstreamPeer) {
            ASSERT_EQ(downstreamPeer->waitComplete(), LwSciError_Success);
        }

        if (readData) {
            ASSERT_EQ(memcmp(readData, rdPtr, rdSize), 0)
                << " Test Failed: Buffer content doesn't match with expected "
                   "Data.";
        }

        if (writeData) {
            memcpy(wrPtr, writeData, wrSize);
            ASSERT_EQ(memcmp(writeData, wrPtr, wrSize), 0)
                << "Test Failed: Buffer content doesn't match with written "
                   "Data.";
        }

        // TODO: Handle multiple upstream/downstream peers when a test requires
        // this
        if (upstreamPeer) {
            ASSERT_EQ(upstreamPeer->signalComplete(), LwSciError_Success);
            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);
        }
        if (downstreamPeer) {
            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);
        }
    }

    std::vector<std::shared_ptr<LwSciBufIpcPeer>> peers;
    pid_t pid = 0;
};

#endif // INCLUDED_LWSCIBUF_INTERPROCESS_TEST_H
