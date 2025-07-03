/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
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

class LlgdTest
{
public:
    typedef std::list<LlgdTest *> TestList;
    
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
    LlgdTest() { s_tests.push_back(this); }

    // required methods
    virtual Type GetType() const = 0;

    virtual const char * GetName() const = 0;

    virtual LwError Execute() = 0;

    // optional helper methods
    virtual LwError Init() { return LwSuccess; }

    virtual LwError TearDown() { return LwSuccess; }

    virtual ~LlgdTest() { }

    // access to auto-registered instances
    static const TestList & GetTests() { return s_tests; }

private:
    static TestList s_tests;
};

#define LLGD_DEFINE_TEST(TEST_NAME, TEST_TYPE, BODY) \
    class LlgdTest ## TEST_NAME : public LlgdTest { \
    public: \
        Type GetType() const { return Type::TYPE_ ## TEST_TYPE; }  \
        const char * GetName() const { return #TEST_NAME; } \
        \
        BODY \
    }; \
    \
    static LlgdTest ## TEST_NAME s_instance ## TEST_NAME;
