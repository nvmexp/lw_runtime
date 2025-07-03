/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <AftermathTest.h>
#include <AftermathTestLogging.h>
#include <AftermathTestUtilsLWN.h>

#include <lwerror.h>
#include <lwassert.h>

#include <nn/fs.h>
#include <nn/os/os_MemoryHeapApi.h>
#include <nn/init/init_Malloc.h>
#include <lw/lw_ServiceName.h>

#include <algorithm>
#include <iterator>

using namespace AftermathTest;

const TestMask DEFAULT_TEST_MASK = {
    Test::Type::TYPE_UNIT,
    Test::Type::TYPE_INTEGRATION
};

const TestMask MANUAL_TEST_MASK = {
    Test::Type::TYPE_MANUAL
};

static void intersect(TestMask& m1, const TestMask& m2)
{
    TestMask tmpMask;

    std::set_intersection(m1.begin(), m1.end(), m2.begin(), m2.end(),
        std::inserter(tmpMask, tmpMask.end()));

    m1 = tmpMask;
}

static bool contains(const TestMask& m, Test::Type type)
{
    return m.find(type) != m.end();
}

static LwError ParseArgs(Options & output)
{
    LwError e = LwSuccess;
    int argc = nn::os::GetHostArgc();
    char** argv = nn::os::GetHostArgv();

    output.testMask = DEFAULT_TEST_MASK;
    output.testNames.clear();
    output.testNamesBeginWith.clear();
    output.tempFileDir = nullptr;
    output.keepTempFiles = false;
    output.disableDebugLayer = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--unit") == 0 ||
            strcmp(argv[i], "-u") == 0) {
            intersect(output.testMask, MANUAL_TEST_MASK);
            output.testMask.insert(Test::Type::TYPE_UNIT);
        } else if (strcmp(argv[i], "--full") == 0 ||
                   strcmp(argv[i], "-f") == 0) {
            intersect(output.testMask, MANUAL_TEST_MASK);
            output.testMask.insert(Test::Type::TYPE_UNIT);
            output.testMask.insert(Test::Type::TYPE_INTEGRATION);
        } else if (strcmp(argv[i], "--manual") == 0 ||
                   strcmp(argv[i], "-m") == 0) {
            output.testMask.insert(Test::Type::TYPE_MANUAL);
        } else if (strcmp(argv[i], "--test") == 0 ||
                   strcmp(argv[i], "-t") == 0) {
            if (i + 1 >= argc) {
                AftermathErr("Missing parameter for option '%s'", argv[i]);
                ORIGINATE_ERROR(LwError_BadParameter);
            }

            output.testNames.push_back(argv[i + 1]);
            output.testMask.insert(Test::Type::TYPE_ALL);
            ++i;
        } else if (strcmp(argv[i], "--testBeginsWith") == 0 ||
                   strcmp(argv[i], "-tb") == 0) {
            if (i + 1 >= argc) {
                AftermathErr("Missing parameter for option '%s'", argv[i]);
                ORIGINATE_ERROR(LwError_BadParameter);
            }

            output.testNamesBeginWith.push_back(argv[i + 1]);
            output.testMask.insert(Test::Type::TYPE_ALL);
            ++i;
        } else if (strcmp(argv[i], "--tempFileDir") == 0 ||
            strcmp(argv[i], "-td") == 0) {
            if (i + 1 >= argc) {
                AftermathErr("Missing parameter for option '%s'", argv[i]);
                ORIGINATE_ERROR(LwError_BadParameter);
            }

            output.tempFileDir = argv[i + 1];
            ++i;
        } else if (strcmp(argv[i], "--keepTempFiles") == 0 ||
            strcmp(argv[i], "-kf") == 0) {
            output.keepTempFiles = true;
        }
        else if (strcmp(argv[i], "--disableDebugLayer") == 0 ||
            strcmp(argv[i], "-dd") == 0) {
            output.disableDebugLayer = true;
        } else {
            AftermathInfo("Usage: %s [ --unit | --full ] [ --manual ] [ --test <name> ] [ --testBeginsWith <begin of name> ] [--tempFileDir <dir>] [--keepTempFiles]",
                argv[0]);
            AftermathInfo("  --unit: execute unit tests only");
            AftermathInfo("  --full: execute unit and integration tests");
            AftermathInfo("  --manual: include tests that require manual setup");
            AftermathInfo("  --test <name>: only execute test <name>");
            AftermathInfo("  --testBeginsWith <name>: execute tests whose name begins with <name>");
            AftermathInfo("  --tempFileDir <dir>: HOST directory where the test can write temporary files");
            AftermathInfo("  --keepTempFiles: Do not delete temporary files when the test is done");
            AftermathInfo("  --disableDebugLayer: Do not enable the LWN debug layer");

            output.testMask.clear();

            break;
        }
    }

    return e;
}

static void* FsAllocate(size_t size)
{
    void* p = malloc(size);
    return p;
}

static void FsDeallocate(void* p, size_t size)
{
    (void)size;
    free(p);
}

extern "C" void nninitStartup()
{
    const size_t heapSize = 128 << 20;
    const size_t mallocSize = 64 << 20;
    uintptr_t address;
    nn::Result result;

    result = nn::os::SetMemoryHeapSize(heapSize);
    LW_ASSERT(result.IsSuccess());

    result = nn::os::AllocateMemoryBlock(&address, mallocSize);
    LW_ASSERT(result.IsSuccess());

    nn::init::InitializeAllocator(reinterpret_cast<void *>(address), mallocSize);

    lw::SetGraphicsServiceName("lwdrv:t");
}

static bool IsTestEnabled(Test* test, Options& options)
{
    if (! contains(options.testMask, Test::Type::TYPE_ALL)) {
        if (! contains(options.testMask, test->GetType())) {
            return false;
        }
    }

    const bool nameOptionsExist = !options.testNames.empty() || !options.testNamesBeginWith.empty();

    if (!options.testNames.empty()) {
        for (const auto testName : options.testNames) {
            if (strcmp(test->GetName(), testName) == 0) {
                return true;
            }
        }
    }

    if (!options.testNamesBeginWith.empty()) {
        for (const auto nameBegin : options.testNamesBeginWith) {
            if (strstr(test->GetName(), nameBegin) == test->GetName()) {
                return true;
            }
        }
    }

    if (nameOptionsExist) {
        // Name options exist but didn't hit any conditions
        return false;
    }

    return true;
}

LwError RunAll()
{
    LwError e = LwSuccess;
    Options options;

    PROPAGATE_ERROR(ParseArgs(options));

    // Set up the single device shared by all (sub)tests,
    // unless a test has explicitly opted-out and has its
    // own device setup.
#ifndef DEFER_INIT_GRAPHICS
    LWN::SetupLWNGraphics();
    const DeviceFlagBits flags = options.disableDebugLayer ? 0 : DeviceFlagBits::DEBUG_ENABLE_LEVEL_4;
    LWN::SetupLWNDevice(flags);
#endif

    const Test::TestList& tests = Test::GetTests();

    for (Test* lwrrentTest : tests) {
        Logging::Msg(Logging::LogLevel::LEVEL_INFO, "Test %-50s: ",
            lwrrentTest->GetName());

        if (IsTestEnabled(lwrrentTest, options)) {
            PROPAGATE_ERROR(lwrrentTest->Init());
            PROPAGATE_ERROR(lwrrentTest->Execute(options));
            PROPAGATE_ERROR(lwrrentTest->TearDown());

            Logging::Msg(Logging::LogLevel::LEVEL_INFO, "done.\n");
        } else {
            Logging::Msg(Logging::LogLevel::LEVEL_INFO, "skip!\n");
        }
    }

    return e;
}

void RunAftermathTests()
{
    AftermathInfo("Starting...");

    LwError result = RunAll();

    if (result != LwSuccess) {
        AftermathErr("Failed!");
    } else {
        AftermathInfo("Success!");
    }
}

extern "C" void nnMain()
{
    // Initializes the file system allocator
    nn::fs::SetAllocator(FsAllocate, FsDeallocate);

    RunAftermathTests();
}
