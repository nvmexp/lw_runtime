/*
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#ifndef INCLUDED_LWSCISYNC_INTERPROCESS_TEST_H
#define INCLUDED_LWSCISYNC_INTERPROCESS_TEST_H

#include "lwscisync_ipc_peer.h"

#include <memory>
#include <vector>

class LwSciSyncInterProcessTest : public ::testing::Test
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

    std::vector<std::shared_ptr<LwSciSyncIpcPeer>> peers;
    bool ipcinit = false;
    pid_t pid = 0;
};

#endif // INCLUDED_LWSCISYNC_INTERPROCESS_TEST_H
