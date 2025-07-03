/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#pragma once

#include <lwerror.h>

#include <list>
#include <set>
#include <vector>

#include <AftermathApi.h>

 // Aftermath test functionality not exposed in API header!
struct AftermathTestTrackedResourcesCounts
{
    size_t numTextures;
    size_t numBuffers;
    size_t numSamplerPools;
    size_t numTexturePools;
    size_t numFinalizedTextures;
    size_t numFinalizedBuffers;
    size_t numFinalizedSamplerPools;
    size_t numFinalizedTexturePools;
};
AFTERMATHAPI AftermathApiError aftermathTestGetNumTrackedResources(AftermathTestTrackedResourcesCounts* counts);

namespace AftermathTest {

struct Options;

class Test
{
public:
    typedef std::list<Test*> TestList;

    enum class Type
    {
        // unit tests that fully execute within the test process
        TYPE_UNIT,

        // integration tests that have external dependencies
        // those will be exelwted in sanity
        TYPE_INTEGRATION,

        // tests that need manual intervention, like backdooring into system
        // services to work
        TYPE_MANUAL,

        // placeholder to select all tags
        TYPE_ALL
    };

    // auto-register the test instance
    Test() { s_tests.push_back(this); }

    // required methods
    virtual Type GetType() const = 0;

    virtual const char* GetName() const = 0;

    virtual LwError Execute(const Options& options) = 0;

    // optional helper methods
    virtual LwError Init() { return LwSuccess; }

    virtual LwError TearDown() { return LwSuccess; }

    virtual ~Test() { }

    // access to auto-registered instances
    static const TestList & GetTests() { return s_tests; }

private:
    static TestList s_tests;
};

typedef std::set<Test::Type> TestMask;

struct Options
{
    TestMask testMask;
    std::vector<char*> testNames;
    std::vector<char*> testNamesBeginWith;
    uint32_t logLevel;
    char* tempFileDir;
    bool keepTempFiles;
    bool disableDebugLayer;
};

#define AFTERMATH_DEFINE_TEST(TEST_NAME, TEST_TYPE, BODY)           \
    class AftermathTest ## TEST_NAME : public AftermathTest::Test { \
    public:                                                         \
        Type GetType() const { return Type::TYPE_ ## TEST_TYPE; }   \
        const char * GetName() const { return #TEST_NAME; }         \
                                                                    \
        BODY                                                        \
    };                                                              \
                                                                    \
    static AftermathTest ## TEST_NAME s_instance;

} // namespace AftermathTest
