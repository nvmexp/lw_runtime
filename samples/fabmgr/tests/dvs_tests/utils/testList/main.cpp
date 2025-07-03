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

#include <iostream>
#include <sstream>

#include "supportContext.hpp"

#include "Elw.h"
#include "Logger.h"
#include "UtilOS.h"

#include "g_testlist.h"


// A simple helper binary that will output some useful platform-specific information
// for the test runner scripts to access and execute at run-time.
//
// --gpuid - This option will output either "igpu" or "dgpu" for each GPU
// installed in the system.
//
// --binsuffix - This prints out the binary suffix (ex. ".exe" on Windows,
// "_srt" on Android, "" on Linux, etc)
//
// --testlist - Prints out the names of each FM test binary on a separate line.

using namespace Log;

// Environment set for verbose rmapi reporting (for device queries)
ILoggedElwironment *g_pElw = NULL;

static NullLogger nullLog;
static StderrLogger stderrLog;
static StdoutLogger stdoutLog;


static void print_usage(const char *binname)
{
    std::stringstream ss;
    ss << "Usage:\n  " << binname << " [--gpuid | --binsuffix | --testlist]\n";
    stdoutLog.write(ss.str().c_str());
    stdoutLog.write("  --gpuid: Prints out the GPU type (\"igpu\" or \"dgpu\") for each GPU as a separate line\n");
    stdoutLog.write("  --binsuffix: Prints out the binary suffix (included binary file extension) for the test binaries in the FM test suite.\n");
    stdoutLog.write("  --testlist: Prints out a list of the FM test suite names (without the binary suffix) as separate lines.\n");
}

void print_gpu_ids()
{
    SupportContext ctx;
    std::vector<PlatformClassEnum> platClass = ctx.PlatformClass();

    for (size_t i = 0; i < platClass.size(); ++i) {
        std::stringstream ss;
        ss << "GPU " << i << ": ";

        switch(platClass[i]) {
            case PLATFORM_CLASS_IGPU:
                ss << "igpu\n";
                break;
            case PLATFORM_CLASS_DGPU:
                ss << "dgpu\n";
                break;
            default:
                ss << "Invalid GPU type\n";
                break;
        }

        stdoutLog.write(ss.str().c_str());
    }
}

void print_bin_suffix()
{
    std::stringstream ss;
    ss << g_bin_suffix << "\n";
    stdoutLog.write(ss.str());
}

void print_test_list()
{
    for (int i = 0; i < sizeof(g_testlist_str)/sizeof(const char *); ++i){
        std::stringstream ss;
        ss << g_testlist_str[i] << '\n';
        stdoutLog.write(ss.str().c_str());
    }
}

int main(int argc, const char **argv) {

    g_pElw = new NonPrivilegedElwironment(nullLog, nullLog);


    if (argc != 2) {
        stderrLog.write("Invalid command.  Usage is:\n");
        print_usage(argv[0]);
        exit(0);
    }
    if (strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "-h") == 0)
    {
        print_usage(argv[0]);
        exit(0);
    }

    if (strcmp(argv[1], "--gpuid") == 0) {
        print_gpu_ids();
        exit(0);
    }

    if (strcmp(argv[1], "--binsuffix") == 0) {
        print_bin_suffix();
        exit(0);
    }

    if (strcmp(argv[1], "--testlist") == 0) {
        print_test_list();
        exit(0);
    }

    // Anything else is unhandled.
    stderrLog.write("Invalid command!");
    print_usage(argv[0]);

    return 1;
}
