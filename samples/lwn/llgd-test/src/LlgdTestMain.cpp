/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <LlgdTest.h>
#include <LlgdTestUtil.h>
#include <LlgdTestUtilLWN.h>

#include <lwerror.h>

#if defined(LW_HOS)
#include <nn/os/os_MemoryHeapApi.h>
#include <nn/init/init_Malloc.h>
#include <lw/lw_ServiceName.h>
#endif

#include <list>
#include <set>
#include <algorithm>
#include <iterator>
#include <vector>

typedef std::set<LlgdTest::Type> TestMask;

const TestMask DEFAULT_TEST_MASK = {
    LlgdTest::Type::TYPE_UNIT,
    LlgdTest::Type::TYPE_INTEGRATION
};

const TestMask MANUAL_TEST_MASK = {
    LlgdTest::Type::TYPE_MANUAL
};

static void intersect(TestMask & m1, const TestMask & m2)
{
    TestMask tmpMask;

    std::set_intersection(m1.begin(), m1.end(), m2.begin(), m2.end(),
        std::inserter(tmpMask, tmpMask.end()));

    m1 = tmpMask;
}

static bool contains(const TestMask & m, LlgdTest::Type type)
{
    return m.find(type) != m.end();
}

struct Options
{
    TestMask testMask;
    std::vector<char *> testNames;
    std::vector<char *> testNamesBeginWith;
    uint32_t logLevel;
};

static LwError ParseArgs(int argc, char** argv, Options & output)
{
    LwError e = LwSuccess;

    output.testMask = DEFAULT_TEST_MASK;
    output.testNames.clear();
    output.testNamesBeginWith.clear();

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--unit") == 0 ||
            strcmp(argv[i], "-u") == 0)
        {
            intersect(output.testMask, MANUAL_TEST_MASK);
            output.testMask.insert(LlgdTest::Type::TYPE_UNIT);
        } else if (strcmp(argv[i], "--full") == 0 ||
                   strcmp(argv[i], "-f") == 0)
        {
            intersect(output.testMask, MANUAL_TEST_MASK);
            output.testMask.insert(LlgdTest::Type::TYPE_UNIT);
            output.testMask.insert(LlgdTest::Type::TYPE_INTEGRATION);
        } else if (strcmp(argv[i], "--manual") == 0 ||
                   strcmp(argv[i], "-m") == 0)
        {
            output.testMask.insert(LlgdTest::Type::TYPE_MANUAL);
        } else if (strcmp(argv[i], "--test") == 0 ||
                   strcmp(argv[i], "-t") == 0)
        {
            if (i + 1 >= argc)
            {
                LlgdErr("Missing parameter for option '%s'", argv[i]);
                ORIGINATE_ERROR(LwError_BadParameter);
            }

            output.testNames.push_back(argv[i + 1]);
            output.testMask.insert(LlgdTest::Type::TYPE_ALL);
            ++i;
        } else if (strcmp(argv[i], "--testBeginsWith") == 0 ||
                   strcmp(argv[i], "-tb") == 0)
        {
            if (i + 1 >= argc)
            {
                LlgdErr("Missing parameter for option '%s'", argv[i]);
                ORIGINATE_ERROR(LwError_BadParameter);
            }

            output.testNamesBeginWith.push_back(argv[i + 1]);
            output.testMask.insert(LlgdTest::Type::TYPE_ALL);
            ++i;
        } else
        {
            LlgdInfo("Usage: %s [ --unit | --full ] [ --manual ] [ --test <name> ] [ --testBeginsWith <begin of name> ]",
                argv[0]);
            LlgdInfo("                  --unit: execute unit tests only");
            LlgdInfo("                  --full: execute unit and integration tests");
            LlgdInfo("                --manual: include tests that require manual setup");
            LlgdInfo("           --test <name>: only execute test <name>");
            LlgdInfo(" --testBeginsWith <name>: execute tests whose name begins with <name>");

            output.testMask.clear();

            break;
        }
    }

    return e;
}

#if defined(LW_HOS)
extern "C" void nninitStartup()
{
    const size_t heapSize = 128 << 20;
    const size_t mallocSize = 64 << 20;
    uintptr_t address;
    nn::Result result;

    result = nn::os::SetMemoryHeapSize(heapSize);
    CHECK(result.IsSuccess());

    result = nn::os::AllocateMemoryBlock(&address, mallocSize);
    CHECK(result.IsSuccess());

    nn::init::InitializeAllocator(reinterpret_cast<void *>(address), mallocSize);

    lw::SetGraphicsServiceName("lwdrv:t");
}
#endif

static bool IsTestEnabled(LlgdTest * test, Options & options)
{
    if (! contains(options.testMask, LlgdTest::Type::TYPE_ALL))
        if (! contains(options.testMask, test->GetType()))
            return false;

    const bool nameOptionsExist = !options.testNames.empty() || !options.testNamesBeginWith.empty();

    if (!options.testNames.empty())
    {
        for (const auto testName : options.testNames)
        {
            if (strcmp(test->GetName(), testName) == 0)
            {
                return true;
            }
        }
    }

    if (!options.testNamesBeginWith.empty())
    {
        for (const auto nameBegin : options.testNamesBeginWith)
        {
            if (strstr(test->GetName(), nameBegin) == test->GetName())
            {
                return true;
            }
        }
    }

    if (nameOptionsExist)
    {
        // Name options exist but didn't hit any conditions
        return false;
    }

    return true;
}

static LwError RunAll(int argc, char** argv)
{
    LwError e = LwSuccess;
    Options options;

    PROPAGATE_ERROR(ParseArgs(argc, argv, options));

    const LlgdTest::TestList & tests = LlgdTest::GetTests();

    for (LlgdTest * lwrrentTest : tests)
    {
        LlgdMsg(LlgdLogLevel::LEVEL_INFO, "Test %-50s: ",
            lwrrentTest->GetName());

        if (IsTestEnabled(lwrrentTest, options))
        {
            PROPAGATE_ERROR(lwrrentTest->Init());
            PROPAGATE_ERROR(lwrrentTest->Execute());
            PROPAGATE_ERROR(lwrrentTest->TearDown());

            LlgdMsg(LlgdLogLevel::LEVEL_INFO, "done.\n");
        } else
        {
            LlgdMsg(LlgdLogLevel::LEVEL_INFO, "skip!\n");
        }
    }

    return e;
}

static void RunLlgdTests(int argc, char** argv)
{
    LlgdInfo("Starting...");

    LwError result = RunAll(argc, argv);

    if (result != LwSuccess)
        LlgdErr("Failed!");
    else
        LlgdInfo("Success!");
}

#if defined(LW_HOS)
extern "C" void nnMain()
{
    llgd_lwn::SetupLWNMemory();
    llgd_lwn::SetupLWNMappings();
    g_device.Initialize();

    int argc = nn::os::GetHostArgc();
    char **argv = nn::os::GetHostArgv();
    RunLlgdTests(argc, argv);
}
#else
int main(int argc, char** argv)
{
    llgd_lwn::SetupLWNMappings();
    g_device.Initialize();

    RunLlgdTests(argc, argv);
}
#endif