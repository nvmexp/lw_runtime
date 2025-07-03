#include "args.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

TestArgs g_args;

static bool ParseBoolean(char *argv[], int index, const char *option, bool *pvar)
{
    if (strcmp(argv[index], option) == 0) {
        *pvar = true;
        return true;
    }
    return false;
}

#ifdef WAYLAND_LWN_PRESENT
static bool ParseInt(char *argv[], int &index, const char *option, int *pvar)
{
    if (strcmp(argv[index], option) == 0) {
        index++;

        if (argv[index]) {
            char* endptr;
            *pvar = (int)strtoul(argv[index], &endptr, 0);

            if (*endptr == '\0') {
                return true;
            }
        }
        printf("Must specify valid argument for %s\n", argv[index - 1]);
        exit(1);
    }
    return false;
}
#endif

static void PrintUsage(char *progName)
{
    printf("Usage: %s <options>\n"
           "    -debug          : Enable debug layer/debug info\n"
           "    -fullScreen     : Create fullScreen window\n"
           "    -tripleBuffer   : Create swap chain with 3 buffers\n"
           "    -nocompress     : Uncompressed display buffer\n"
           "    -frameRate      : Compute framerate statistics\n"
#ifdef WAYLAND_LWN_PRESENT
           "    -cpuLoad <load> : CPU frame load in milliseconds\n"
           "    -gpuLoad <load> : GPU frame load in milliseconds\n"
#endif
            , progName);
}

void TestArgs::ParseArgs(int argc, char *argv[])
{

    for (int i = 1; i < argc; i++) {
        if (ParseBoolean(argv, i, "-debug", &m_debug)) {
            continue;
        }
        if (ParseBoolean(argv, i, "-fullScreen", &m_fullScreen)) {
            continue;
        }
        if (ParseBoolean(argv, i, "-tripleBuffer", &m_tripleBuffer)) {
            continue;
        }
        bool nocompress;
        if (ParseBoolean(argv, i, "-nocompress", &nocompress)) {
            m_compress = !nocompress;
            continue;
        }
        if (ParseBoolean(argv, i, "-frameRate", &m_measureFrameRate)) {
            continue;
        }
#ifdef WAYLAND_LWN_PRESENT
        if (ParseInt(argv, i, "-cpuLoad", &m_cpuLoad)) {
            continue;
        }
        if (ParseInt(argv, i, "-gpuLoad", &m_gpuLoad)) {
            continue;
        }
#endif
        // Unhandled argument: print usage
        PrintUsage(argv[0]);

        bool help;
        if (ParseBoolean(argv, i, "-h", &help)) {
            exit(0);
        } else {
            printf("\n**** Unknown argument: %s\n", argv[i]);
            exit(1);
        }
    }
}
