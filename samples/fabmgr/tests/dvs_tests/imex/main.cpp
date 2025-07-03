/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include <string>
#include "gtest/gtest.h"
#include "Elw.h"
#include "Logger.h"

extern "C" {
#include "commandline/commandline.h"
}

enum
{
    ARGNAME_HELP,
    ARGNAME_VERBOSE,
    ARGNAME__COUNT
};

struct all_args allArgs[] =
{
    {
        ARGNAME_HELP,
        "-h",
        "--help",
        "Display the help messages",
        "\n\t"
        "Display a detailed usage description and exit.",
        CMDLINE_OPTION_NO_VALUE_ALLOWED,
    },
    {
        ARGNAME_VERBOSE,
        "-v",
        "--verbose",
        "Display verbose messages about progress and errors",
        "\n\t"
        "Display verbose messages about progress and errors.",
        CMDLINE_OPTION_NO_VALUE_ALLOWED,
    },
};


using namespace std;
using namespace Log;
using namespace boost;

ILoggedElwironment* g_pElw = NULL;

TEST(imex, memoryImport)
{
}

int main(int argc, char **argv)
{
    StdoutLogger stdoutLog;
    NullLogger   nullLog;
    void        *pCmdLine = NULL;
    int          status   = 0;

    testing::InitGoogleTest(&argc, argv);

    status = cmdline_init(argc, argv, allArgs, ARGNAME__COUNT, &pCmdLine);
    if (status != 0)
    {
        cout << "Command line init failed 0x" << hex << status;
        cmdline_printOptionsSummary(pCmdLine, 1);
    }
    // Help
    if (cmdline_exists(pCmdLine, ARGNAME_HELP))
    {
        cmdline_printOptionsSummary(pCmdLine, 1);
        status = 0;
        goto done;
    }

    // Verbose mode
    if (cmdline_exists(pCmdLine, ARGNAME_VERBOSE))
    {
        g_pElw = new NonPrivilegedElwironment(stdoutLog, stdoutLog);
    }
    else
    {
        g_pElw = new NonPrivilegedElwironment(nullLog, nullLog);
    }
    status = RUN_ALL_TESTS();
done:
    delete g_pElw;
    cmdline_destroy(pCmdLine);
    return status;
}

