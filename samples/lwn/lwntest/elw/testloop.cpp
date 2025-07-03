/*
 * Copyright (c) 1999 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "ogtest.h"
#include "lwogtypes.h"
#include "contexts.h"
#include "elw.h"
#include "cmdline.h"
#include "md5.h"
#include "sums.h"
#include "testloop.h"
#include "readpixels.h"
#include "tga.h"
#include <lwnTest/lwnTest_Mislwtils.h>
#include <lwnTest/lwnTest_GlslcHelper.h>

#define STATUS_PASSED             0
#define STATUS_FAILED             1
#define STATUS_NOGOLD             2
#define STATUS_BADCRC             3


#define TEST_FLAG_FAILED              0
#define TEST_FLAG_NOGOLD              1
#define TEST_FLAG_ERRORS              2
#define TEST_FLAG_WARNINGS            3

// If using -lwnGlslcInputFile, indicates a test had shaders not found in the input binary.
#define TEST_FLAG_NOT_IN_GLSLC_BINARY 3

#if defined(SPIRV_ENABLED)
// If using -lwnGlslcSpirv, indicates if a test used shaders that glslang could not compile
// a SPIR-V binary module for.
#define TEST_FLAG_GLSLC_SPIRV_ERROR 4
#endif

#ifdef LW_HOS
const char *testlistFilename = "host:/testlist.md";
#else
const char *testlistFilename = "testlist.md";
#endif

int testIndex = 0; // current test index
int loopIndex = 0;
int repeatIndex = 0;
unsigned char *framebuffer = NULL;
const char **testsFailed = NULL;
const char **testsWithoutGold = NULL;
const char **testsWithErrors = NULL;
const char **testsWithWarnings = NULL;
const char **testsWithNoShadersInGlslcBinary = NULL;
const char **testsNotRequested = NULL;
const char **testsNotSupported = NULL;
#if defined(SPIRV_ENABLED)
const char **testsWithSpirvCompileErrors = NULL;
#endif
int *testsFlags = NULL;
int nPassed = 0;
int nFailed = 0;
int nNoGold = 0;
int nErrors = 0;
int nWarnings = 0;
int nTestsWithErrors = 0;
int nTestsWithWarnings = 0;
int nTestsWithNoShadersInGlslcBinary = 0;
int nTestsNotRequested = 0;
int nTestsNotSupported = 0;
#if defined(SPIRV_ENABLED)
int nTestsWithSpirvCompileErrors = 0;
#endif
int nBadCRC = 0;
int nNewgolds = 0;
int nSkipped = 0;
int nFailedListed = 0;
int nNoGoldListed = 0;

static uint64_t testTimerStart;
static uint64_t testTimerEnd;
static uint64_t testTimerFrequency;

static int statusFailed(int status)
{
    return (status != STATUS_PASSED);
}

static void updateStatus(const char *name, int status, int path)
{
    // Tally for the last loop
    switch (status) {
    case STATUS_PASSED:
        nPassed++;
        break;
    case STATUS_FAILED:
        nFailed++;
        if ((testsFlags[testIndex] & (1 << TEST_FLAG_FAILED)) == 0) {
            testsFailed[nFailedListed++] = name;
            testsFlags[testIndex] |= 1 << TEST_FLAG_FAILED;
        }
        break;
    case STATUS_NOGOLD:
        nNoGold++;
        if ((testsFlags[testIndex] & (1 << TEST_FLAG_NOGOLD)) == 0) {
            testsWithoutGold[nNoGoldListed++] = name;
            testsFlags[testIndex] |= 1 << TEST_FLAG_NOGOLD;
        }
        break;
    case STATUS_BADCRC:
        nBadCRC++;
        break;
    default:
        assert(0);
        break;
    }

    if (lwnDebugErrorMessageCount) {
        nErrors++;
        if ((testsFlags[testIndex] & (1 << TEST_FLAG_ERRORS)) == 0) {
            testsWithErrors[nTestsWithErrors++] = name;
            testsFlags[testIndex] |= 1 << TEST_FLAG_ERRORS;
        }
    }

    if (lwnDebugWarningMessageCount) {
        nWarnings++;
        if ((testsFlags[testIndex] & (1 << TEST_FLAG_WARNINGS)) == 0) {
            testsWithWarnings[nTestsWithWarnings++] = name;
            testsFlags[testIndex] |= 1 << TEST_FLAG_WARNINGS;
        }
    }

    if (lwnGlslcBinaryMissCount) {
        if ((testsFlags[testIndex] & (1 << TEST_FLAG_NOT_IN_GLSLC_BINARY)) == 0) {
            testsWithNoShadersInGlslcBinary[nTestsWithNoShadersInGlslcBinary++] = name;
            testsFlags[testIndex] |= 1 << TEST_FLAG_NOT_IN_GLSLC_BINARY;
        }
        lwnGlslcBinaryMissCount = 0;
    }

#if defined(SPIRV_ENABLED)
    if (lwnGlslcSpirvErrorCount) {
        if ((testsFlags[testIndex] & (1 << TEST_FLAG_GLSLC_SPIRV_ERROR)) == 0) {
            testsWithSpirvCompileErrors[nTestsWithSpirvCompileErrors++] = name;
            testsFlags[testIndex] |= 1 << TEST_FLAG_GLSLC_SPIRV_ERROR;
        }
        lwnGlslcSpirvErrorCount = 0;
    }
#endif
}

// Handy Helper function for QuickResultsSummary
static void PrintSummaryLine(FILE *f, const char *desc, int num, int total)
{
    fprintf(f, "  %4d/%04d (%5.1f%%) tests %s\n", num, total, 100.0*num/total, desc);
}

static void QuickResultsSummary(FILE *f)
{
    char delim = ' ';
    int i,t,nTestsTotal;
    int aborted;

    t = nPassed + nFailed + nNoGold + nBadCRC;
    nTestsTotal = TestCount - nSkipped;
    aborted = t && nTestsTotal && t < nTestsTotal && (!interactiveMode);

    if (lwnCompileAsFp16Mask) {
        printf("*** All GLSL shaders compiled with default mediump precision in stages represented by stage mask 0x%02x ***\n",
               lwnCompileAsFp16Mask);
    }
    fprintf(f, "\nQuick%sResults Summary:\n", aborted ? " **aborted** " : " ");
    fprintf(f, "%4d Tests Run\n", nTestsTotal);
    if (t > 0) {
        PrintSummaryLine(f, "passed",  nPassed, t);
        if (nBadCRC) {
            PrintSummaryLine(f, "passed, but with bad CRCs", nBadCRC, t);
        }
        PrintSummaryLine(f, "failed",  nFailed, t);
        PrintSummaryLine(f, "missing golds",  nNoGold, t);
        if (nTestsWithNoShadersInGlslcBinary) {
            PrintSummaryLine(f, "missing entries in input GLSLC binary cache", nTestsWithNoShadersInGlslcBinary, t);
        }
        if (aborted) {
            PrintSummaryLine(f, "completed      **** run aborted ****",  t, nTestsTotal);
        }
    }

    if (nSkipped) {
        fprintf(f, "\n%4d Tests Skipped\n", nSkipped);

        if (nTestsNotRequested) {
            fprintf(f, "  %4d tests not requested\n", nTestsNotRequested);
        }
        if (nTestsNotSupported) {
            fprintf(f, "  %4d tests not supported\n", nTestsNotSupported);
        }
    }
    if (nErrors) {
        PrintSummaryLine(f, "with reported errors", nErrors, t);
    }
    if (nWarnings) {
        PrintSummaryLine(f, "with reported warnings", nWarnings, t);
    }

#if defined(SPIRV_ENABLED)
    if (nTestsWithSpirvCompileErrors) {
        PrintSummaryLine(f, "with glslang SPIR-V compilation errors", nTestsWithSpirvCompileErrors, t);
    }
#endif

    if (newlineDelim) {
        delim = '\n';
    }
    if (nNoGold) {
        fprintf(f, "\n  Tests missing golds:");
        for (i = 0; i < nNoGoldListed; i++) {
            fprintf(f, "%c%s", delim, testsWithoutGold[i]);
        }
        fprintf(f, "\n");
    }
    if (nFailed) {
        int len = 0;
        fprintf(f, "\n  Tests which failed:");
        for (i = 0; i < nFailedListed; i++) {
            if (delim != '\n') {
                len += 1+(int)strlen(testsFailed[i]);
            }
            fprintf(f, "%c%s", delim, testsFailed[i]);
            if (len > 2000) {
                // line will get truncated in file
                fprintf(f, "\n  More tests which failed:");
                len = 0;
            }
        }
        fprintf(f, "\n");
    }
    if (nTestsWithErrors) {
        int len = 0;
        fprintf(f, "\n  Tests with reported errors:");
        for (i = 0; i < nTestsWithErrors; i++) {
            if (delim != '\n') {
                len += 1 + (int) strlen(testsWithErrors[i]);
            }
            fprintf(f, "%c%s", delim, testsWithErrors[i]);
            if (len > 2000) {
                // line will get truncated in file
                fprintf(f, "\n  More tests with reported errors:");
                len = 0;
            }
        }
        fprintf(f, "\n");
    }
    if (nTestsWithWarnings) {
        int len = 0;
        fprintf(f, "\n  Tests with reported warnings:");
        for (i = 0; i < nTestsWithWarnings; i++) {
            if (delim != '\n') {
                len += 1 + (int) strlen(testsWithWarnings[i]);
            }
            fprintf(f, "%c%s", delim, testsWithWarnings[i]);
            if (len > 2000) {
                // line will get truncated in file
                fprintf(f, "\n  More tests with reported warnings:");
                len = 0;
            }
        }
        fprintf(f, "\n");
    }
    if (nTestsWithNoShadersInGlslcBinary) {
        int len =0;
        fprintf(f, "\n  Tests with missing binaries in input GLSLC binary cache:");
        for (i = 0; i < nTestsWithNoShadersInGlslcBinary; i++) {
            if (delim != '\n') {
                len += 1 + (int) strlen(testsWithNoShadersInGlslcBinary[i]);
            }
            fprintf(f, "%c%s", delim, testsWithNoShadersInGlslcBinary[i]);
            if (len > 2000) {
                // line will get truncated in file
                fprintf(f, "\n  More tests with missing binaries in input GLSLC binary cache:");
                len = 0;
            }
        }
        fprintf(f, "\n");
    }

#if defined(SPIRV_ENABLED)
    if (nTestsWithSpirvCompileErrors) {
        int len =0;
        fprintf(f, "\n  Tests with glslang SPIR-V compilation errors:");
        for (i = 0; i < nTestsWithSpirvCompileErrors; i++) {
            if (delim != '\n') {
                len += 1 + (int) strlen(testsWithSpirvCompileErrors[i]);
            }
            fprintf(f, "%c%s", delim, testsWithSpirvCompileErrors[i]);
            if (len > 2000) {
                // line will get truncated in file
                fprintf(f, "\n  More tests with glslang SPIR-V compilation issues:");
                len = 0;
            }
        }
        fprintf(f, "\n");
    }
#endif

    if (lwnGlslcBinaryCacheApiMismatch) {
        fprintf(f, "\n  All tests required recompile for GLSLC API components.\n");
    }
    if (lwnGlslcBinaryCacheGpuMismatch) {
        fprintf(f, "\n  All tests required recompile for GLSLC GPU code components.\n");
    }

    if (nNewgolds > 0) {
        fprintf(f, "  %3d/%d tests created\n",  nNewgolds, nNewgolds);
    }
}


static int TestDisabled(int mask)
{
    return ((!useDebugTests) && (mask & LWOG_TEST_PATH_MASK_DEBUG_TEST)) ||
           ((!useSkippedTests) && (mask & LWOG_TEST_PATH_MASK_SKIP));
}

TestSupport GetTestSupport(OGTEST *lwrTest)
{
    char singlerun[512];
    int isWildcard = 0;

    if (nameSingleTestRun) {
        lwog_snprintf(singlerun, 512, "%s", nameSingleTestRun);
        if (nameSingleTestRun[strlen(nameSingleTestRun)-1] == '*') {
            singlerun[strrchr(singlerun, '*')-singlerun] = '\0';
            isWildcard = 2;
        }
        if (nameSingleTestRun[0] == '*') {
            lwog_snprintf(singlerun, 512, "%s", singlerun+1);
            isWildcard += 1;
        }
    } else {
        isWildcard = 0;
    }

    if (nameSingleTestRun && !isWildcard && strcmp(singlerun, TEST_BASE_NAME(lwrTest))) {
        return NotRequested;
    }

    // handle wildcards in -t
    if (nameSingleTestRun && isWildcard && strcmp(singlerun, TEST_BASE_NAME(lwrTest))) {
        if (isWildcard) {
            int len, found = 0;

            len = (int)strlen(singlerun);

            switch (isWildcard) {
                case 1: // *foo
                    if (!strncmp(singlerun, TEST_FULL_NAME(lwrTest, 0) + strlen(TEST_FULL_NAME(lwrTest, 0)) - len, len)) {
                        found = 1;
                        break;
                    }
                    break;
                case 2: // foo*
                    if (!strncmp(TEST_FULL_NAME(lwrTest, 0), singlerun, len)) {
                        found = 1;
                        break;
                    }
                    break;
                case 3: // *foo*
                    if (strstr(TEST_FULL_NAME(lwrTest, 0), singlerun)) {
                        found = 1;
                        break;
                    }
                    break;
            }
            if (!found) {
                return NotRequested;
            }
        } else {
            return NotRequested;
        }
    }

    if (includeFilename && !(lwrTest->pathMask & LWOG_TEST_PATH_MASK_INCLUDE)) {
        return NotRequested;
    }

    if (excludeFilename && (lwrTest->pathMask & LWOG_TEST_PATH_MASK_EXCLUDE)) {
        return NotRequested;
    }

    if (!lwrTest->isSupportedFunc(TEST_BASE_NAME(lwrTest))) {
        return NotSupported;
    }

    if (!(lwrTest->pathMask & LWOG_TEST_PATH_MASK_RUNTIMESUPP)) {
        return NotRequested;
    }

    return Supported;
}


static void writeImage(const char *pathname, const char *testname) {
    char fname[512];
    if (pathname != NULL) {
        int result;
        if (loopCount > 1) {
            lwog_snprintf(fname, sizeof(fname), "%s/%s_loop_%d_%d.tga", pathname, testname, loopIndex+1, loopCount);
        } else {
            lwog_snprintf(fname, sizeof(fname), "%s/%s.tga", pathname, testname);
        }
        result = TgaRGBA32Save(fname, lwrrentWindowWidth, lwrrentWindowHeight, framebuffer);
        (void)result;
        assert(result == 0);
    }
}

static void flush_results(void)
{
    fflush(results);
}

static void RunSingleTestDiff(int index, int path)
{
    unsigned char md5[16] = {0};
#define TEMP_BUF_LEN 512
    char buf[TEMP_BUF_LEN];
    int status = STATUS_PASSED;
    OGTEST *lwrTest;

    lwrTest = GET_TEST(index);

    buf[0] = '\0';

    if (crcPrint || goldDir || outputDir) {
        md5Generate(TEST_FULL_NAME(lwrTest, path), framebuffer,
                lwrrentWindowWidth, lwrrentWindowHeight, 32, md5);
    }

    if (goldDir) {
        int md5Match = 0;

        if (lwrTest->md5ValidMask) {
            unsigned char *goldmd5 = NULL;
            int md5GoldIdx;
            for (md5GoldIdx = 0; md5GoldIdx < MAX_MD5_GOLDS; ++md5GoldIdx) {
                goldmd5 = lwrTest->md5[md5GoldIdx];
                md5Match = !memcmp(md5, goldmd5, 16);
                if (md5Match) {
                    break;
                }
            }
        }

        if (!md5Match) {
            // Failed the crc or no crc
            // All checks treated as crconly.
            if (lwrTest->md5ValidMask & 1) {
                status = STATUS_FAILED;
                strncatf(buf, TEMP_BUF_LEN, "FAIL! crconly mode: skipping gold image compare");
                // Note: gold image compare is not implemented.
            } else {
                status = STATUS_NOGOLD;
                strncatf(buf, TEMP_BUF_LEN, "FAIL! crconly mode: could not find gold CRC value");
            }
            updateStatus(TEST_FULL_NAME(lwrTest, path), status, path);

            if (outputDir && statusFailed(status)) {
                // write crc, md5 to output
                writeImage(outputDir, TEST_FULL_NAME(lwrTest, path));
            }

            /* If the gold image is missing and there is a "missing directory"
                specified, write the TGA file into the missing directory. */
            if (STATUS_NOGOLD == status && missDir) {
                // write image
                writeImage(missDir, TEST_FULL_NAME(lwrTest, path));
                if (!outputDir) {
                    md5Write(TEST_FULL_NAME(lwrTest, path), md5);
                }
            }

        } else {
            // Passed the crc test
            strncatf(buf, TEMP_BUF_LEN, "pass!");
            status = STATUS_PASSED;
            updateStatus(TEST_FULL_NAME(lwrTest, path), status, path);
        }
    } else {
        if (outputDir) {
            writeImage(outputDir, TEST_FULL_NAME(lwrTest, path));
            nNewgolds++;
        }
        strncatf(buf, TEMP_BUF_LEN, "done");
    }

    if (lwnDebugEnabled && lwnDebugErrorMessageCount) {
        strncatf(buf, TEMP_BUF_LEN, " (errors)");
    }

    if (lwnDebugEnabled && lwnDebugWarningMessageCount) {
        strncatf(buf, TEMP_BUF_LEN, " (warnings)");
    }

    if (reportTestTime) {
        strncatf(buf, TEMP_BUF_LEN, " %10.3f msec",
                 1000.0 * (double) (testTimerEnd - testTimerStart) / (double) testTimerFrequency);
    }

    if (repeatCount > 1) {
        strncatf(buf, TEMP_BUF_LEN, " [rep %d/%d]", repeatIndex + 1, repeatCount);
    }

    if (loopIndex > 0 || loopCount > 1) {
        strncatf(buf, TEMP_BUF_LEN, " [loop %d/%d]", loopIndex + 1, loopCount);
    }

    if (outputDir) {
        md5Write(TEST_FULL_NAME(lwrTest, path), md5);
    }

    if (crcPrint) {
        strncatf(buf, TEMP_BUF_LEN, " (md5: %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x)",
                    (unsigned int)md5[0], (unsigned int)md5[1], (unsigned int)md5[2], (unsigned int)md5[3],
                    (unsigned int)md5[4], (unsigned int)md5[5], (unsigned int)md5[6], (unsigned int)md5[7],
                    (unsigned int)md5[8], (unsigned int)md5[9], (unsigned int)md5[10], (unsigned int)md5[11],
                    (unsigned int)md5[12], (unsigned int)md5[13], (unsigned int)md5[14], (unsigned int)md5[15]);
    }

    fprintf(stdout, "%s\n", buf);
    fflush(stdout);

    if (results) {
        fprintf(results, "%s\n", buf);
        flush_results();
    }

    if (lwnTest::failed() && lwnTestVerbose) {
        // Print any test error prints to stdout and result file.
        lwnTest::printFailures(stdout);
        if (results) {
            lwnTest::printFailures(results);
        }
    }
    lwnTest::resetFailedStatus();
}

typedef struct {
    OGTEST *lwrTest;
    int path;
} TestArgs;

static void DrawTestSetupAndRun(void *args)
{
    OGTEST *lwrTest = ((TestArgs*)args)->lwrTest;
    int path = ((TestArgs*)args)->path;

    if (!repeatsUseDifferentSeeds || repeatIndex == 0) {
        lwSRand(test_seed);
    }

    lwrTest->initFunc(1.0, TEST_BASE_NAME(lwrTest), path);
    assert(lwrTest->doFunc);
    lwrTest->doFunc(1.0, TEST_BASE_NAME(lwrTest), path);
}

static void DrawTestTeardown(void *args)
{
    OGTEST *lwrTest = ((TestArgs*) args)->lwrTest;
    int path = ((TestArgs*) args)->path;

    lwrTest->exitFunc(TEST_BASE_NAME(lwrTest), path);

    // read back texture data
    GetFramebufferData(framebuffer);
}

static void DrawTest(void *args)
{
    DrawTestSetupAndRun(args);
    DrawTestTeardown(args);
}

static void RunSingleTest(int index)
{
    int path = 0;
    OGTEST *lwrTest = GET_TEST(index);
    TestArgs args = { lwrTest, path };

    TestSupport support = GetTestSupport(lwrTest);

    if (Supported != support) {
        nSkipped++;
        switch (support) {
        case NotRequested:
            if (loopIndex == 0 && repeatIndex == 0) {
                testsNotRequested[nTestsNotRequested++] = TEST_FULL_NAME(lwrTest, 0);
            }
            break;
        case NotSupported:
            if (loopIndex == 0 && repeatIndex == 0) {
                testsNotSupported[nTestsNotSupported++] = TEST_FULL_NAME(lwrTest, 0);
            }
            break;
        default:
            assert(0);
        }
        // TODO zapisi u listu
        return;
    }

    lwogSetWindowTitle(lwrTest->name);

    {
#define TMPBUF_LEN 256
        char tmpbuf[TMPBUF_LEN] = "";

        if (lwog_snprintf(tmpbuf, TMPBUF_LEN, "Do %-40s: ", TEST_FULL_NAME(lwrTest, path)) == -1) {
            assert(!"Testname doesn't fit in tmpbuf.  Output will be truncated.");
        }
        fprintf(stdout, "%s", tmpbuf);
        fflush(stdout);

        if (results) {
            if (lwog_snprintf(tmpbuf, TMPBUF_LEN, "Running test '%-40s' ", TEST_FULL_NAME(lwrTest, path)) == -1) {
                assert(!"Testname doesn't fit in tmpbuf.  results file output will be truncated.");
            }
            fprintf(results, "%s", tmpbuf);
            flush_results();
        }
    }

    testTimerStart = lwogGetTimerValue();

    // Reset the debug message counts.
    lwnDebugErrorMessageCount = 0;
    lwnDebugWarningMessageCount = 0;

    // Run the test
    if (useWorkerThreads) {
#if defined(LW_WINDOWS)
        // If a GL donor context is used to present on Windows, the present is done in
        // ExitGraphics and requires a current context (which doesn't exist on the worker
        // thread). So we have the worker thread do the initGraphics/doGraphics and let
        // the main thread take care of exitGraphics (including the present).
        if (useGLPresent) {
            lwogRunOnWorkerThread(DrawTestSetupAndRun, &args, 0);
            DrawTestTeardown(&args);
        } else {
            lwogRunOnWorkerThread(DrawTest, &args, 0);
        }
#else
        lwogRunOnWorkerThread(DrawTest, &args, 0);
#endif
    } else {
        DrawTest(&args);
    }

    testTimerEnd = lwogGetTimerValue();

    RunSingleTestDiff(index, path);

    lwogSwapBuffers();
}


static void RunInteractiveMode()
{
    int index;

    // Start by processing any initial events.
    lwogHandleWindowEvents();

    // When starting in interactive mode, pretend that our current test
    // was "-1" (before the first test) and that we pressed the down
    // arrow.
    interactiveTestData.inputPending = 0;
    interactiveTestData.testIndex = -1;
    interactiveTestData.testDirection = +1;
    interactiveTestData.nameLen = 0;
    interactiveTestData.name[0] = 0;

    // If "-t" is specified, act as though that name were typed as input.
    // Clamp the length to our buffer size, and strip off any "*" suffix.
    if (nameSingleTestRun) {
        size_t nameLen = strlen(nameSingleTestRun);
        if (nameLen > __GL_ARRAYSIZE(interactiveTestData.name) - 2) {
            nameLen = __GL_ARRAYSIZE(interactiveTestData.name) - 2;
        }
        memcpy(interactiveTestData.name, nameSingleTestRun, nameLen);
        if (nameLen && interactiveTestData.name[nameLen - 1] == '*') {
            nameLen--;
        }
        interactiveTestData.name[nameLen] = 0;
        if (nameLen) {
            interactiveTestData.nameLen = nameLen;
            interactiveTestData.testDirection = 0;
        } else {
            printf("Ignoring empty '-t' string.\n");
        }
        nameSingleTestRun = NULL;
    }

    for (;;) {
        OGTEST *lwrTest;
        if (interactiveTestData.testDirection == -1) {

            // If we processed an up arrow event, loop backwards in the
            // test list to find a supported test.
            for (index = interactiveTestData.testIndex - 1; index >= 0; index--) {
                lwrTest = GET_TEST(index);
                if (Supported == GetTestSupport(lwrTest)) {
                    interactiveTestData.testIndex = index;
                    break;
                }
            }
            if (index < 0) {
                printf("At the beginning of the test list; re-running first test.\n");
            }

        } else if (interactiveTestData.testDirection == +1) {

            // If we processed a down arrow event, loop forwards in the
            // test list to find a supported test.  If we don't get a hit,
            // re-run the current test (the last supported one).
            for (index = interactiveTestData.testIndex + 1; index < TestCount; index++) {
                lwrTest = GET_TEST(index);
                if (Supported == GetTestSupport(lwrTest)) {
                    interactiveTestData.testIndex = index;
                    break;
                }
            }

            // If our current index is less than zero, we scanned the entire
            // test list and found nothing supported.  Report that as an
            // error.
            if (interactiveTestData.testIndex < 0) {
                printf("Failed to find any valid test in interactive mode.\n");
                Terminate(EXIT_STATUS_NOT_NORMAL);
            }

            if (index >= TestCount) {
                printf("At the end of the test list; re-running last test.\n");
            }

        } else if (interactiveTestData.nameLen) {

            // If we got text input, use the input string (followed by '*') as
            // a search string, as though it were specified with the "-t"
            // option.  Search for the first supported test, and use that.
            assert(interactiveTestData.nameLen < __GL_ARRAYSIZE(interactiveTestData.name) - 1);
            interactiveTestData.name[interactiveTestData.nameLen] = '*';
            interactiveTestData.name[interactiveTestData.nameLen + 1] = 0;

            nameSingleTestRun = interactiveTestData.name;
            for (index = 0; index < TestCount; index++) {
                lwrTest = GET_TEST(index);
                if (Supported == GetTestSupport(lwrTest)) {
                    interactiveTestData.testIndex = index;
                    break;
                }
            }
            nameSingleTestRun = NULL;

            // If we didn't find a matching test, report an error, stripping
            // off the '*' suffix.  We will re-run the existing test.
            if (index >= TestCount) {
                interactiveTestData.name[interactiveTestData.nameLen]  = 0;
                printf("Failed to locate a supported test matching '%s'.\n", interactiveTestData.name);
            }

            // If we didn't find a match and we don't have any existing test
            // (i.e., application passed "-t xxx" for some invalid "xxx"),
            // start over acting as though no name were specified.
            if (interactiveTestData.testIndex < 0) {
                interactiveTestData.testDirection = +1;
                interactiveTestData.nameLen = 0;
                interactiveTestData.name[0] = 0;
                continue;
            }
        }

        // Run the lwrrently selected test, and then handle additional input.
        lwrTest = GET_TEST(interactiveTestData.testIndex);
        RunSingleTest(interactiveTestData.testIndex);
        lwogHandleWindowEvents();
    }
}


static int CompareName(const void* a, const void* b)
{
    // We are sorting two arrays of "OGTEST *", so our inputs are pointers to
    // pointers to OGTEST.
    const OGTEST * const *pta = (const OGTEST * const *) a;
    const OGTEST * const *ptb = (const OGTEST * const *) b;
    const OGTEST *ta = *pta;
    const OGTEST *tb = *ptb;
    const char* na = TEST_PROF_NAME(ta);
    const char* nb = TEST_PROF_NAME(tb);

    // works like strcmp upto 120 characters
    // except that '_' comes before anything else except '\0'
    return stricmp(na, nb);
}

static int CompareNameSearch(const void* a, const void* b)
{
    // We are comparing the string <a> against an "OGTEST *" pointed to by <b>.
    const OGTEST * const *ptb = (const OGTEST * const *) b;
    const OGTEST *tb = *ptb;
    const char *na = (const char *) a;
    const char* nb = TEST_PROF_NAME(tb);

    // works like strcmp upto 120 characters
    // except that '_' comes before anything else except '\0'
    return stricmp(na, nb);
}

//
// Our full set of test descriptors is split into several groups, and the
// logic in PrepareTestList() is used to build a single sorted list of
// pointers to test descriptors.  Each group has its own TestGroup structure,
// and we build an array of these structures by including tests.h in
// "enumerate groups" mode, where the BEGIN_TEST_GROUP macro is ilwoked once
// for each group.
//
int TestCount;
OGTEST **lwog_FullTestList = NULL;

#define TEST_GROUP_ENUMERATE_GROUPS     1

#define BEGIN_TEST_GROUP(x)     extern TestGroup lwog_ ## x ## _TestGroup;
#define END_TEST_GROUP(x)
#include "tests.h"

#undef BEGIN_TEST_GROUP
#define BEGIN_TEST_GROUP(x)     &lwog_ ## x ## _TestGroup,

TestGroup *allTestGroups[] = {
#include "tests.h"
};

#undef TEST_GROUP_ENUMERATE_GROUPS
#undef BEGIN_TEST_GROUP
#undef END_TEST_GROUP

static void PrepareTestList(void)
{
    int i, j;

    // Count the number of tests so we can build a full test list.
    TestCount = 0;
    for (i = 0; i < (int)__GL_ARRAYSIZE(allTestGroups); i++) {
        TestCount += allTestGroups[i]->nTests;
    }

    lwog_FullTestList = (OGTEST **)__LWOG_MALLOC(TestCount * sizeof(OGTEST *));

    // Fill the full test list with pointers into the groups' descriptor
    // arrays.
    TestCount = 0;
    for (i = 0; i < (int)__GL_ARRAYSIZE(allTestGroups); i++) {
        for (j = 0; j < (int) allTestGroups[i]->nTests; j++) {
            lwog_FullTestList[TestCount++] = &allTestGroups[i]->tests[j];
        }
    }

    qsort(lwog_FullTestList, TestCount, sizeof(OGTEST *), CompareName);
}

static void ProcessTestListItem(const char *item, unsigned int maskbit)
{
    OGTEST **test = (OGTEST **)bsearch(item, lwog_FullTestList, TestCount, sizeof(OGTEST *),
                                       CompareNameSearch);
    if (!test) {
        fprintf(stdout, "Warning:  Can't find test named %s from test list file.\n", item);
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }
    (*test)->pathMask |= maskbit;
}

static void ProcessTestList(const char *filename, unsigned int maskbit)
{
    char name[256];
    int c;
    int namelen = 0;
    int incomment = 0;

    FILE *includes = fopen(filename, "rt");
    if (!includes) {
        fprintf(stdout, "Error: Failed to open test list file:  %s\n", filename);
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }

    do {
        c = fgetc(includes);
        switch (c) {
        case ' ':
        case '\t':
        case '\r':
        case '\n':
        case EOF:
        case '#':
            // If we're lwrrently working on a test name and find a special
            // character (whitespace or comment), treat that as the end of the
            // test and process it.
            if (namelen) {
                name[namelen] = 0;
                ProcessTestListItem(name, maskbit);
                namelen = 0;
            }

            // Newlines end a comment line, '#' starts one.
            if (c == '\r' || c == '\n' || c == EOF) {
                incomment = 0;
            } else if (c == '#') {
                incomment = 1;
            }
            break;

        default:
            // Other characters append to the test name if not in a comment.
            if (!incomment) {
                if (namelen < 255) {
                    name[namelen++] = (char) c;
                } else {
                    fprintf(stdout, "Error: Test list file item too long.");
                    Terminate(EXIT_STATUS_NOT_NORMAL);
                }
            }
            break;
        }
    } while (c != EOF);
    fclose(includes);
}

static void getTestDescription(char *str, size_t n, OGTEST *lwrtest)
{
    // Assume the full description doesn't fit
    GLboolean fit = GL_TRUE;
    int result = 0;

    if (makedocs) {
        result = lwog_snprintf(str, n, "# %s {#%s}\n", TEST_BASE_NAME(lwrtest), TEST_BASE_NAME(lwrtest));
    } else {
        result = lwog_snprintf(str, n, "Test description: %s\n", TEST_BASE_NAME(lwrtest));
    }

    if (result != -1) {

        // mark the end of the buffer for overflow detection
        str[n - 1] = '\0';
        lwrtest->getDescription(str + strlen(str), TEST_BASE_NAME(lwrtest));
        if (str[n - 1] != '\0') {
            assert(!"Test description buffer overflow!");
            // the strcpy below will null terminate the buffer within its limits
            fit = GL_FALSE;
        }

    }

    if (!fit) {
        // Safely append "...<snip>" to the str buffer so that the truncation
        // is obvious
        char snipstr[] = "\n...<snip>";
        size_t sniplen = strlen(snipstr);
        assert(sniplen < n);
        lwog_snprintf(str + n - sniplen - 1, sniplen + 1, "%s", snipstr);
    }
}




void InitNonGraphics()
{
    int i, j;

    // Initialize all cells to be enabled
    for (i = 0; i < MAX_CELL_ROW_COL; i++) {
        for (j = 0; j < MAX_CELL_ROW_COL; j++) {
            defaultCellGrid[i][j] = 1;
        }
    }

    lwSetRandFunc(lwRand);

    printf("Initializing...\n");

    PrepareTestList();

    if (includeFilename) {
        ProcessTestList(includeFilename, LWOG_TEST_PATH_MASK_INCLUDE);
    }
    if (excludeFilename) {
        ProcessTestList(excludeFilename, LWOG_TEST_PATH_MASK_EXCLUDE);
    }

    for (i = 0; i < TestCount; i++) {
        OGTEST *lwrTest = GET_TEST(i);
        if (TestDisabled(lwrTest->pathMask)) {
            lwrTest->pathMask &= ~LWOG_TEST_PATH_MASK_RUNTIMESUPP;
        } else {
            lwrTest->pathMask |= LWOG_TEST_PATH_MASK_RUNTIMESUPP;
        }
    }

    if ((testsFailed = (const char **) __LWOG_MALLOC(TestCount * sizeof(char *))) == NULL) {
        fprintf(stdout, "Error: Failed to malloc for list of tests that failed\n");
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }
    if ((testsWithoutGold = (const char **) __LWOG_MALLOC(TestCount * sizeof(char *))) == NULL) {
        fprintf(stdout, "Error: Failed to malloc for list of tests without golds\n");
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }
    if ((testsWithErrors = (const char **) __LWOG_MALLOC(TestCount * sizeof(char *))) == NULL) {
        fprintf(stdout, "Error: Failed to malloc for list of tests with errors\n");
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }
    if ((testsWithWarnings = (const char **) __LWOG_MALLOC(TestCount * sizeof(char *))) == NULL) {
        fprintf(stdout, "Error: Failed to malloc for list of tests with warnings\n");
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }
    if ((testsWithNoShadersInGlslcBinary = (const char **) __LWOG_MALLOC(TestCount * sizeof(char *))) == NULL) {
        fprintf(stdout, "Error: Failed to malloc for list of tests missing shaders in input GLSLC binary\n");
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }

    if ((testsNotRequested = (const char **) __LWOG_MALLOC(TestCount * sizeof(char *))) == NULL) {
        fprintf(stdout, "Error: Failed to malloc for list of tests that weren't included\n");
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }

    if ((testsNotSupported = (const char **) __LWOG_MALLOC(TestCount * sizeof(char *))) == NULL) {
        fprintf(stdout, "Error: Failed to malloc for list of tests that aren't supported\n");
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }

#if defined(SPIRV_ENABLED)
    if ((testsWithSpirvCompileErrors = (const char **) __LWOG_MALLOC(TestCount * sizeof(char *))) == NULL) {
        fprintf(stdout, "Error: Failed to malloc for list of tests with SPIR-V compile errors.\n");
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }
#endif

    if ((testsFlags = (int *) __LWOG_MALLOC(TestCount * sizeof(int))) == NULL) {
        fprintf(stdout, "Error: Failed to malloc failed array\n");
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }
    for (i = 0; i < TestCount; i++) {
        testsFlags[i] = 0;
    }

    framebuffer = (unsigned char *)__LWOG_MALLOC(lwrrentWindowWidth*lwrrentWindowHeight*4);

    // Open results file
    results = fopen(resultsFilename, "wt");

    if (results == NULL) {
        fprintf(stdout, "WARNING: Could not open file %s for writing\n", resultsFilename);
    }

    testTimerFrequency = lwogGetTimerFrequency();
}

// InitDeferredNonGraphics - Initializes non-graphics items that should be
// initialized after the LWN device has been initialized.
static void InitDeferredNonGraphics()
{
    // Read gold checksum files into TestStruct
    checksumGoldFileRead(goldDir, "lwnmd5.txt", 0);

    // Open checksum files for write
    if (outputDir) {
        checksumFileOpen(&md5File, outputDir, "lwnmd5.txt");
    } else {
        checksumFileOpen(&md5File, missDir, "lwnmd5.txt");
    }
}

void MainLoop()
{
    int i;
    int writeDescriptions = makedocs || (describe && nameSingleTestRun);

#if defined(LWNTEST_LWDA_ENABLED)
    if (lwdaEnabled) {
        if (!InitializeLWDA()) {
            printf("LWCA initialization failed.  Skipping LWCA tests.\n");
            lwdaEnabled = 0;
        }
    }
#else
    lwdaEnabled = 0;
#endif

    if (!InitializeLWN()) {
        printf("lwntest initialization failed.  Exiting.\n");
        return;
    }
    CreateFramebufferBuffer();

#if defined(LW_WINDOWS)
    // Starting with version 53.6 we do no longer need an OpenGL donor context
    // to swap buffers. If the donor context is not required for interop or to
    // test the "old" GL present path we can delete it.
    if (lwogCheckLWNAPIVersion(53, 6)) {
        if (!useGL && !useGLPresent) {
            lwogDeleteDonorContext();
        }
        if (!useGLPresent) {
            lwogSetNativeWindow();
        }
    }
#endif

    if (writeDescriptions || genlistFilename)
    {
        FILE *testlist = NULL;
        FILE *genlist = NULL;
        int found = 0;
        int all = makedocs || (nameSingleTestRun && !strcmp(nameSingleTestRun, "ALL"));

        if (makedocs) {
            testlist = fopen(testlistFilename, "wt");
            if (!testlist) {
                printf("Failed to open %s for writing\n", testlistFilename);
                Terminate(EXIT_STATUS_NOT_NORMAL);
            }
            fprintf(testlist, "Test List {#lwntest_test_list}\n");
            fprintf(testlist, "=========\n");
        }
        if (genlistFilename) {
            genlist = fopen(genlistFilename, "wt");
            if (!genlist) {
                printf("Failed to open %s for writing\n", genlistFilename);
                Terminate(EXIT_STATUS_NOT_NORMAL);
            }
        }

        for (i = 0; i < TestCount; i++) {
            char buf[RETURN_TEST_DESCRIPTION_BUFFER_LENGTH];
            size_t len;

            if (!all &&  Supported != GetTestSupport(GET_TEST(i))) {
                continue;
            } else {
                found = 1;
            }

            if (genlistFilename) {
                fprintf(genlist, "%s\n", TEST_BASE_NAME(GET_TEST(i)));
            }

            if (writeDescriptions) {
                getTestDescription(buf, RETURN_TEST_DESCRIPTION_BUFFER_LENGTH, GET_TEST(i));
                if (makedocs) {
                    fprintf(testlist, "%s\n\n", buf);
                } else {
                    printf("%s\n", buf);
                }
                len = strlen(buf);
                if (len > RETURN_TEST_DESCRIPTION_BUFFER_LENGTH-2) {
                    printf("length of description for test %s is close to the limit and might have overflowed, len=%d\n",
                        TEST_BASE_NAME(GET_TEST(i)), (int)len);
                    Terminate(EXIT_STATUS_USER_ERROR);
                }
                fflush(stdout);
            }
        }

        if (makedocs) {
            fclose(testlist);
        }
        if (genlistFilename) {
            fclose(genlist);
        }
        if (!found && nameSingleTestRun) {
            printf("No such test found: %s\n", nameSingleTestRun);
            Terminate(EXIT_STATUS_USER_ERROR);
        }
        Terminate(EXIT_STATUS_NORMAL_NO_RESULTS);
    }

    InitDeferredNonGraphics();

    // Clear the window to start out in case the first test takes a long time.
    lwogClearWindow();

    if (interactiveMode) {
        RunInteractiveMode();
    } else {
        for (loopIndex = 0; loopIndex < loopCount; loopIndex++) {
            nTestsNotRequested = 0;
            nTestsNotSupported = 0;
            nSkipped = 0;
            for (testIndex = 0; testIndex < TestCount; testIndex++) {
                for (repeatIndex = 0; repeatIndex < repeatCount; repeatIndex++) {
                    RunSingleTest(testIndex);
                    lwogHandleWindowEvents();
                }
            }
        }
    }

    Terminate(EXIT_STATUS_NORMAL);
}


void Terminate(lwogExitStatus exitStatus)
{
    if (exitStatus == EXIT_STATUS_NORMAL    ||
        exitStatus == EXIT_STATUS_USER_QUIT ||
        exitStatus == EXIT_STATUS_TRAPPED_EXIT) {
        // Only print the summary for "normal" or "good" exits.
        // Needed to help keep our result parsers sane.
        if (results) QuickResultsSummary(results);
        QuickResultsSummary(stdout);

        // Output the GLSLC binary if requested.
        if (lwnGlslcOutputFile) {
            WriteGlslcOutput();
        }
    }
    if (results) fclose(results);

    if (md5File) {
        checksumFileClose(&md5File);
    }
    DestroyFramebufferBuffer();
    FinalizeLWN();
    if (testsFailed) {
        __LWOG_FREE((void *)testsFailed);
    }
    if (testsWithoutGold) {
        __LWOG_FREE((void *)testsWithoutGold);
    }
    if (testsWithErrors) {
        __LWOG_FREE((void *) testsWithErrors);
    }
    if (testsWithWarnings) {
        __LWOG_FREE((void *) testsWithWarnings);
    }
    if (testsWithNoShadersInGlslcBinary) {
        __LWOG_FREE((void *) testsWithNoShadersInGlslcBinary);
    }
    if (testsNotRequested) {
        __LWOG_FREE((void *) testsNotRequested);
    }
    if (testsNotSupported) {
        __LWOG_FREE((void *) testsNotSupported);
    }
#if defined(SPIRV_ENABLED)
    if (testsWithSpirvCompileErrors) {
        __LWOG_FREE((void *) testsWithSpirvCompileErrors);
    }
#endif
    if (testsFlags) {
        __LWOG_FREE(testsFlags);
    }
    __LWOG_FREE(lwog_FullTestList);
    if (cmdlineFileBuffer) {
        __LWOG_FREE(cmdlineFileBuffer);
    }
    if (cmdlineFileBufferPtrs) {
        __LWOG_FREE(cmdlineFileBufferPtrs);
    }
    if (framebuffer) {
        __LWOG_FREE(framebuffer);
    }
    lwogTerminate(exitStatus);
}
