/*
 * Copyright (c) 2006 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "ogtest.h"
#include "cmdline.h"
#include "ctype.h"
#include "str_util.h"

#ifdef LW_HOS
const char *defaultResultsFilename = "host:/results.txt";
#else
const char *defaultResultsFilename = "results.txt";
#endif

static char charToLowerCase(char c)
{
    return (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c;
}

int strincmp(const char *s1, const char *s2, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) {
        if (charToLowerCase(s1[i]) == charToLowerCase(s2[i]) &&
            s1[i] != '\0') {
            continue;
        }
        return charToLowerCase(s1[i]) - charToLowerCase(s2[i]);
    }
    return 0;
}

#ifndef _WIN32
int stricmp(const char *s1, const char *s2) {
    return strincmp(s1, s2, 256);
}
#endif



static void printUsage(void) {
    printf("\nlwntest options:\n");
    printf("  -width <n>                width of window for run  (640 default)\n");
    printf("  -height <n>               height of window for run (480 default)\n");
    printf("  -results <file>           changes results output file to <file>\n"
           "                            if not used, results are output to %s\n", defaultResultsFilename);
    printf("  -describe <test>          print the matching test description and exit immediately\n");
    printf("  -makedocs                 generate documentation source files\n");
    printf("  -debugtests               include running tests marked DEBUG\n");
    printf("  -unskip                   include running tests marked SKIP\n");
    printf("  -t <test>                 run test by name\n");
    printf("  -loop <n>                 loop over all specified tests <n> times\n");
    printf("  -repeat <n>               repeat each test <n> times\n");
    printf("  -repeatsUseDiffSeeds      enables use of different random seeds for each repeat\n");
    printf("  -printcrcs                print crc (and md5, if enabled) to stdout, after usual \"done\" message.\n");
    printf("  -genlist <file>           write a list of tests supported on the device to <file>.\n");
    printf("  +tests <file>             exclude any test not specified in <file>.\n");
    printf("  -tests <file>             exclude any test specified in <file>.\n");
    printf("  -lwndebug <n>             For LWN tests, use a debug context and set the debug level to <n>. <n> must be in the range [0, 4]\n");
    printf("  -v                        Enable verbose output from failing tests\n");
    printf("  -compileAsFp16 <mask>     Prepends \"#extension GL_LW_desktop_lowp_mediump : enable\" to the\n"
           "                            shaders represented by <shader mask>. Defaults to fragment stage (0x2) when used\n"
           "                            without the <shader mask> argument. <shader mask> should be of type LWNshaderStageBits.\n"
           "                            (0x1 == vertex, 0x2 == fragment, 0x4 == geometry, 0x8 == tess control, 0x10 == tess eval, 0x20 == compute).\n");
    printf("  -lwnGlslcDebugLevel <n>   Set a global GLSLC debug level by setting compile option GLSLCdebugInfoLevel to\n"
           "                            GLSLC_DEBUG_LEVEL_G<n>.  <n> must be an integer in the range [0, 2].\n");
    printf("  -lwnGlslcOptLevel <n>     Set a global GLSLC optimization level by setting compile option GLSLCoptLevel to\n"
           "                            <n>, where <n> can be 0 (default optimizations) or 1 (no optimizations).\n");
    printf("  -lwnGlslcOutputFile       Output the concatenated binaries from running with the GLSLC DLL to a file.\n");
    printf("  -lwnGlslcInputFile        Input the concatenated binaries from a previous run using the lwnGlslcOutputFile flag.\n");
    printf("  -noglslc                  Allow compiling without GLSLC (i.e. the hidden online compiler path).  This is a temporary flag.\n");
#if defined(_WIN32)
    printf("  -lwnGlslcDLL              Run LWN tests with GLSLC enabled, loading the GLSLC offline shader compiler DLL on demand from disk.\n");
    printf("  -lwnGlasmCache <entries>  Run LWN tests using a GLASM cach (lwnDeviceSetIntermediateShaderCache) with <entries> number of potential cache entries.\n");
#if defined(SPIRV_ENABLED)
    printf("  -lwnGlslcSpirv            Run tests through the SPIR-V path.  Tests using GLSL which can't\n"
           "                            be colwerted to SPIR-V by glslang will fall back to GLSL compilation.\n");
    printf("  -lwnGlslcSpirvLogErrors   Dump glslang GLSL-to-SPIR-V compilation errors to the console.\n");
#endif
    printf("  -glslang                  Shim input GLSL to SPIR-V inside GLSLC with GLSLC built-in glslang, this options will be ignored if the SPIR-V is passed to GLSLC. This is only for internal testing and does not work with production LwnGlslc builds.\n");
    printf("  -glslangFallbackOnError   Fallback to original GLSL in case of glslang error.\n");
    printf("  -glslangFallbackOnAbsolute Fallback to original GLSL even if glslang passes (supercedes -glslangFallbackOnError).\n");
    printf("  -noUnloadICD              Disable manual unloading of display driver for debug purposes.\n");
    printf("  -memDebug                 Use the Windows debug C runtime to report memory leaks/corruption.\n");
    printf("  -memBreak <n>             Use the Windows debug C runtime to break on allocation <N>.\n");
    printf("  -i                        Interactive mode; wait for input before proceeding to a new test.\n");
#endif
    printf("  -thread                   Run tests in individual synchronized worker threads.\n");
    printf("  -ftime                    Report time for individual test exelwtion.\n");
    printf("  -gl                       Bootstrap LWN with a GL context. Needed for some tests, but not the orthodox way to initialize LWN.\n");
#if defined(_WIN32)
    printf("  -glPresent                Use OpenGL donor context to swap buffers.\n");
#endif
    printf("  -tiledCache               Turn on tiled caching (TC is off by default).\n");
    printf("  -nolwda                   Turn off LWCA interop coverage.\n");
    printf("\nImage comparison options:\n");
    printf("  -o <dir>                  image output directory\n");
    printf("  -g <dir>                  gold reference directory\n");
    printf("  -m <dir>                  write if missing gold\n");
    printf("\nOther options:\n");
    printf("  -newline                  use a newline to delimit summary\n");
    printf("\nMemory heap options:\n");
    printf("  -mallocHeapMB             specify malloc heap size (in MB)\n");
    printf("  -graphicsHeapMB           specify graphics heap size (in MB)\n");
    printf("  -devtoolsHeapMB           specify devtools heap size (in MB)\n");
    printf("  -compilerHeapMB           specify compiler heap size (in MB)\n");
    printf("  -firmwareMemMB            specify firmware memory size (in MB)\n");
    printf("\nQueue options:\n");
    printf("  -queueCommandMemKB        specify queue command memory size (in KB)\n");
    printf("  -queueComputeMemKB        specify queue compute memory size (in KB)\n");
    printf("  -queueFlushThresholdKB    specify queue flush threshold size (in KB)\n");
    printf("  -noZlwll                  run tests with Zlwll disabled\n");
    printf("\n");
}

// argumentPresent:
// Takes command line and the index requested - if the argument needed isn't there
// or of the wrong type, it outputs an error message and terminates
// argType 0 - string, 1 - int, 2 - float
static int argumentPresent(int argc, char** argv, int i, int argType, const char *error)
{
    if(i >= argc || 
        ((argType == 0) ? argv[i][0] == '-' :
         (argType == 1) ? !isdigit(argv[i][0]) :
                          (!isdigit(argv[i][0]) && argv[i][0] != '.') )) {
        printf("%s\n", error);
        Terminate(EXIT_STATUS_USER_ERROR);
        return 0;
    }
    return 1;
}

// ParseCmdLine:
//
//  Parse the cmdline parameters to lwntest and sets globals to reflect the
//  meaning of those parameters.
void ParseCmdLine(int argc, char** argv)
{
    int i;

    resultsFilename = defaultResultsFilename;

    // Parse command line
    for (i = 1; i < argc; i++) {
        if (!stricmp(argv[i], "-width")) {
            if (argumentPresent(argc, argv, (i + 1), 1, "lwntest: -width option must be followed by a number")) {
                lwrrentWindowWidth = atoi(argv[++i]);
            }
            continue;
        }
        if (!stricmp(argv[i], "-height")) {
            if (argumentPresent(argc, argv, (i + 1), 1, "lwntest: -height option must be followed by a number\n")) {
                lwrrentWindowHeight = atoi(argv[++i]);
            }
            continue;
        }
        if (!stricmp(argv[i], "-results")) {
            if (argumentPresent(argc, argv, (i + 1), 0, "lwntest: -results must be followed by a filename")) {
                resultsFilename = argv[++i];
            }
            continue;
        }
        if (!stricmp(argv[i], "-describe")) {
            describe = 1;
            if (argumentPresent(argc, argv, (i + 1), 0, "lwntest: -describe must be followed by a test name")) {
                nameSingleTestRun = argv[++i];
            }
            continue;
        }
        if (!stricmp(argv[i], "-makedocs")) {
            makedocs = 1;
            continue;
        }
        if (!stricmp(argv[i], "-test") || !strcmp(argv[i], "-t")) {
            if (argumentPresent(argc, argv, (i + 1), 0, "lwntest: -t must be followed by a test name")) {
                nameSingleTestRun = argv[++i];
            }
            continue;
        }
        if (!stricmp(argv[i], "-debugtests")) {
            useDebugTests = 1;
            continue;
        }
        if (!stricmp(argv[i], "-unskip")) {
            useSkippedTests = 1;
            continue;
        }
        if (!stricmp(argv[i], "-loop")) {
            if ((i + 1 < argc) && isdigit(argv[i + 1][0])) {
                loopCount = atoi(argv[++i]);
                if (loopCount < 1) {
                    printf("lwntest: -loop option must be a positive integer, got: '%s'\n", argv[i]);
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            } else {
                printf("lwntest: -loop option must be followed by a positive integer\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            continue;
        }
        if (!stricmp(argv[i], "-repeat") || !stricmp(argv[i], "-repeatcount")) {
            if ((i + 1 < argc) && isdigit(argv[i + 1][0])) {
                repeatCount = atoi(argv[++i]);
                if (repeatCount < 1) {
                    printf("lwntest: -repeat option must be a positive integer, got: '%s'\n", argv[i]);
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            } else {
                printf("lwntest: -repeat option must be followed by a positive integer\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            continue;
        }
        if (!stricmp(argv[i], "-repeatsUseDiffSeeds")) {
            repeatsUseDifferentSeeds = 1;
            continue;
        }
        if (!stricmp(argv[i], "-crconly")) {
            // We allow this option for compatibility, but it doesn't do
            // anything because crconly is the only approach we implement.
            continue;
        }
        if (!stricmp(argv[i], "-md5")) {
            // We allow this option for compatibility, but it doesn't do
            // anything because we always enable MD5 logic.
            continue;
        }
        if (!stricmp(argv[i], "-printcrcs")) {
            crcPrint = 1;
            continue;
        }

        if (!stricmp(argv[i], "-genlist")) {
            if (argumentPresent(argc, argv, (i + 1), 0, "lwntest: -genlist must be followed by a filename")) {
                genlistFilename = argv[++i];
            }
            continue;
        }

        if (!stricmp(argv[i], "+tests")) {
            if (argumentPresent(argc, argv, (i + 1), 0, "lwntest: +tests must be followed by a filename")) {
                includeFilename = argv[++i];
            }
            continue;
        }

        if (!stricmp(argv[i], "-tests")) {
            if (argumentPresent(argc, argv, (i + 1), 0, "lwntest: -tests must be followed by a filename")) {
                excludeFilename = argv[++i];
            }
            continue;
        }
        if (!stricmp(argv[i], "-newline")) {
            newlineDelim = 1;
            continue;
        }
        if (!stricmp(argv[i], "-noZlwll")) {
            noZlwll = 1;
            continue;
        }

        if (!stricmp(argv[i], "-seed")) {
            if (argumentPresent(argc, argv, (i + 1), 1, "lwntest: -seed option must be followed by a number")) {
                test_seed = atoi(argv[++i]);
            }
            continue;
        }

        if (!stricmp(argv[i], "-lwndebug")) {
            lwnDebugEnabled = 1;
            if (i + 1 < argc && argv[i + 1] != NULL && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '4') {
                lwnDebugLevel = (argv[++i][0] - '0');
            }
            continue;
        }

        if (!stricmp(argv[i], "-v")) {
            lwnTestVerbose = 1;
            continue;
        }

        if (!stricmp(argv[i], "-compileAsFp16")) {
            // Can be optionally followed by an integer bit mask. If no mask is specified,
            // default to fragment stage only.
            lwnCompileAsFp16Mask = 0x2;

            if (argv[i + 1] != NULL) {
                char* endPtr = NULL;
                unsigned int cmdLineValue = (unsigned int)(strtol(argv[i + 1], &endPtr, 0));

                if (endPtr && (*endPtr == '\0')) {
                    if (0xFFFFFFFC0u & cmdLineValue) {
                        // If the stage mask is out of range (highest bit that can be set is 0x20), output an error.
                        printf("lwntest: -compileAsFp16 argument must correspond to a valid stage in LWNshaderStageBits LWN data type.\n");
                        Terminate(EXIT_STATUS_USER_ERROR);
                    }

                    lwnCompileAsFp16Mask = cmdLineValue;
                    ++i;
                }
            }
            continue;
        }

        if (!stricmp(argv[i], "-thread")) {
            useWorkerThreads = 1;
            continue;
        }

        if (!stricmp(argv[i], "-ftime")) {
            reportTestTime = 1;
            continue;
        }

        if (!stricmp(argv[i], "-gl")) {
            useGL = 1;
            continue;
        }
#if defined(_WIN32)
        if (!stricmp(argv[i], "-glPresent")) {
            useGLPresent = 1;
            continue;
        }
#endif
        if (!stricmp(argv[i], "-tiledCache")) {
            enableTiledCache = 1;
            continue;
        }
        if (!stricmp(argv[i], "-nolwda")) {
            lwdaEnabled = 0;
            continue;
        }

#if defined(_WIN32)
        if (!stricmp(argv[i], "-lwnGlslcDLL")) {
            if (argumentPresent(argc, argv, (i + 1), 0, "lwntest: -lwnGlslcDLL must be followed by the DLL file location.")) {
                lwnGlslcDLL = argv[++i];
            }
            continue;
        }

        if (!stricmp(argv[i], "-lwnGlasmCache")) {
            if (argumentPresent(argc, argv, (i + 1), 1, "lwntest: -lwnGlasmCache option must be followed by a number")) {
                lwnGlasmCacheNumEntries = atoi(argv[++i]);
            }
            continue;
        }
#endif 
        if (!stricmp(argv[i], "-lwnGlslcOutputFile")) {
            if (argumentPresent(argc, argv, (i + 1), 0, "lwntest: -lwnGlslcOutputFile must be followed by the output file location.")) {
                lwnGlslcOutputFile = argv[++i];

                if (lwnGlslcOutputFile && lwnGlslcInputFile) {
                    printf("lwntest: -lwnGlslcInputFile and -lwnGlslcOutputFile options can not both be set.");
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            }
            continue;
        }

        if (!stricmp(argv[i], "-lwnGlslcInputFile")) {
            if (argumentPresent(argc, argv, (i + 1), 0, "lwntest: -lwnGlslcInputFile must be followed by the output file location.")) {
                lwnGlslcInputFile = argv[++i];

                if (lwnGlslcOutputFile && lwnGlslcInputFile) {
                    printf("lwntest: -lwnGlslcInputFile and -lwnGlslcOutputFile options can not both be set.");
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            }
            continue;
        }

        // Set the global GLSLCdebugLevelInfo for all shader compiles to either G0, G1, or G2.
        if (!stricmp(argv[i], "-lwnGlslcDebugLevel")) {
            if ((i + 1 < argc) && isdigit(argv[i + 1][0])) {
                lwnGlslcDebugLevel = atoi(argv[++i]);
                if ((lwnGlslcDebugLevel < 0) || (lwnGlslcDebugLevel > 2)) {
                    printf("lwntest: -lwnGlslcDebugLevel option must be a positive integer in the range of [0, 2], got: '%s'\n", argv[i]);
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            } else {
                printf("lwntest: -lwnGlslcDebugLevel option must be a positive integer in the range of [0, 2].\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            continue;
        }

        // Set the global GLSLCoptLevel for all shader compiles to either 0 (default) or 1 (no optimizations).
        if (!stricmp(argv[i], "-lwnGlslcOptLevel")) {
            int valid = (i + 1 < argc) && isdigit(argv[i + 1][0]) && (argv[i + 1][1] == '\0');
            if (valid) {
                lwnGlslcOptLevel = atoi(argv[i+1]);
            }

            ++i;

            if (!valid || lwnGlslcOptLevel < 0 || lwnGlslcOptLevel > 1) {
                printf("lwntest: -lwnGlslcOptLevel option must be a positive integer in the range of [0, 1]\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }

            continue;
        }

        if (!stricmp(argv[i], "-noglslc")) {
            // Do nothing here.  -noglslc is left in for legacy purposes for tests which may not have been updated
            // to not use this flag.  Once all tests ilwoking lwntest have been colwerted to not use this flag, it
            // will be removed.
            continue;
        }

#if defined(SPIRV_ENABLED)
        if (!stricmp(argv[i], "-lwnGlslcSpirv")) {
            useSpirv = 1;
            continue;
        }

        if (!stricmp(argv[i], "-lwnGlslcSpirvLogErrors")) {
            logSpirvErrors = 1;
            continue;
        }
#endif

#if defined(_WIN32)
        if (!stricmp(argv[i], "-glslang")) {
            glslang = 1;
            continue;
        }

        if (!stricmp(argv[i], "-glslangFallbackOnError")) {
            if (!glslang) {
                printf("lwntest: -glslangFallbackOnError can only be used if -glslang is set\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            glslangFallbackOnError = 1;
            continue;
        }

        if (!stricmp(argv[i], "-glslangFallbackOnAbsolute")) {
            if (!glslang) {
                printf("lwntest: -glslangFallbackOnAbsolute can only be used if -glslang is set\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            glslangFallbackOnAbsolute = 1;
            continue;
        }

        if (!stricmp(argv[i], "-noUnloadICD")) {
            noUnloadICD = 1;
            continue;
        }

        if (!stricmp(argv[i], "-memDebug")) {
            memDebug = 1;
            continue;
        }
        if (!stricmp(argv[i], "-memBreak")) {
            if (argumentPresent(argc, argv, (i + 1), 1, "lwntest: -memBreak option must be followed by a number")) {
                memDebugBreakAllocNum = atoi(argv[++i]);
            }
            continue;
        }
        if (!stricmp(argv[i], "-i")) {
            interactiveMode = 1;
            continue;
        }
#endif

        if (!stricmp(argv[i], "-golddir") || !strcmp(argv[i], "-g") || !strcmp(argv[i], "-l")) {
            if (argumentPresent(argc, argv, (i + 1), 0, "lwntest: -goldDir must be followed by a directory")) {
                goldDir = argv[++i];
            }
            continue;
        }

        if (!stricmp(argv[i], "-outputdir") || !strcmp(argv[i], "-o")) {
            if (argumentPresent(argc, argv, (i + 1), 0, "lwntest: -outputdir must be followed by a directory")) {
                outputDir = argv[++i];
            }
            continue;
        }

        if (!stricmp(argv[i], "-missdir") || !strcmp(argv[i], "-m")) {
            if (argumentPresent(argc, argv, (i + 1), 0, "lwntest: -missdir must be followed by a directory")) {
                missDir = argv[++i];
            }
            continue;
        }

#if defined(LW_HOS)
        if (!stricmp(argv[i], "-mallocHeapMB")) {
            if ((i + 1 < argc) && isdigit(argv[i + 1][0])) {
                hosMallocHeapMB = atoi(argv[++i]);
                if (hosMallocHeapMB < 1) {
                    printf("lwntest: -mallocHeapMB option must be a positive integer, got: '%s'\n", argv[i]);
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            } else {
                printf("lwntest: -mallocHeapMB option must be followed by a positive integer\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            continue;
        }
        if (!stricmp(argv[i], "-graphicsHeapMB")) {
            if ((i + 1 < argc) && isdigit(argv[i + 1][0])) {
                hosGraphicsHeapMB = atoi(argv[++i]);
                if (hosGraphicsHeapMB < 1) {
                    printf("lwntest: -graphicsHeapMB option must be a positive integer, got: '%s'\n", argv[i]);
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            } else {
                printf("lwntest: -graphicsHeapMB option must be followed by a positive integer\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            continue;
        }
        if (!stricmp(argv[i], "-devtoolsHeapMB")) {
            if ((i + 1 < argc) && isdigit(argv[i + 1][0])) {
                hosDevtoolsHeapMB = atoi(argv[++i]);
                if (hosDevtoolsHeapMB < 1) {
                    printf("lwntest: -devtoolsHeapMB option must be a positive integer, got: '%s'\n", argv[i]);
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            } else {
                printf("lwntest: -devtoolsHeapMB option must be followed by a positive integer\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            continue;
        }
        if (!stricmp(argv[i], "-compilerHeapMB")) {
            if ((i + 1 < argc) && isdigit(argv[i + 1][0])) {
                hosCompilerHeapMB = atoi(argv[++i]);
                if (hosCompilerHeapMB < 1) {
                    printf("lwntest: -compilerHeapMB option must be a positive integer, got: '%s'\n", argv[i]);
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            } else {
                printf("lwntest: -compilerHeapMB option must be followed by a positive integer\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            continue;
        }
        if (!stricmp(argv[i], "-firmwareMemMB")) {
            if ((i + 1 < argc) && isdigit(argv[i + 1][0])) {
                hosFirmwareMemMB = atoi(argv[++i]);
                if (hosFirmwareMemMB < 1) {
                    printf("lwntest: -firmwareMemMB option must be a positive integer, got: '%s'\n", argv[i]);
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            } else {
                printf("lwntest: -firmwareMemMB option must be followed by a positive integer\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            continue;
        }
#endif
        if (!stricmp(argv[i], "-queueCommandMemKB")) {
            if ((i + 1 < argc) && isdigit(argv[i + 1][0])) {
                queueCommandMemKB = atoi(argv[++i]);
                if (queueCommandMemKB < 1) {
                    printf("lwntest: -queueCommandMemKB option must be a positive integer, got: '%s'\n", argv[i]);
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            } else {
                printf("lwntest: -queueCommandMemKB option must be followed by a positive integer\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            continue;
        }
        if (!stricmp(argv[i], "-queueComputeMemKB")) {
            if ((i + 1 < argc) && isdigit(argv[i + 1][0])) {
                queueComputeMemKB = atoi(argv[++i]);
                if (queueComputeMemKB < 1) {
                    printf("lwntest: -queueComputeMemKB option must be a positive integer, got: '%s'\n", argv[i]);
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            } else {
                printf("lwntest: -queueComputeMemKB option must be followed by a positive integer\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            continue;
        }
        if (!stricmp(argv[i], "-queueFlushThresholdKB")) {
            if ((i + 1 < argc) && isdigit(argv[i + 1][0])) {
                queueFlushThresholdKB = atoi(argv[++i]);
                if (queueFlushThresholdKB < 1) {
                    printf("lwntest: -queueFlushThresholdKB option must be a positive integer, got: '%s'\n", argv[i]);
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
            } else {
                printf("lwntest: -queueFlushThresholdKB option must be followed by a positive integer\n");
                Terminate(EXIT_STATUS_USER_ERROR);
            }
            continue;
        }

        // The -winsys parameter is ignored. It's only here for compatibility with the lwogtest harness plugin.
        if (!stricmp(argv[i], "-winsys")) {
            ++i;
            continue;
        }

        if (stricmp(argv[i], "-help") && stricmp(argv[i], "--help") &&
            stricmp(argv[i], "-h")) {
            printf("\n");
            printf("\n");
            printf("Unknown argument: %s\nUse the -h argument for help, e.g.:\n   %s -h\n",
                   argv[i], argv[0]);
            Terminate(EXIT_STATUS_USER_ERROR);
        } else {
            printUsage();
            // After printing the usage lwntest should terminate.
            Terminate(EXIT_STATUS_NORMAL_NO_RESULTS);
        }
        Terminate(EXIT_STATUS_NORMAL);
    }
}
