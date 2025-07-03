//! \file
//! \brief LwSciSync kpi perf test.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include <string>
#include <unordered_map>

#include "test.h"

template<typename T> SyncTest * createTest() { return new T; }

typedef std::unordered_map<std::string, SyncTest*(*)()> TestMap;

TestMap testMap{
    { "ModuleOpen", &createTest<ModuleOpen> },
    { "AttrListCreate", &createTest<AttrListCreate> },
    { "AttrListSetAttrs_Signaler", &createTest<AttrListSetAttrs_Signaler> },
    { "AttrListSetAttrs_Waiter", &createTest<AttrListSetAttrs_Waiter> },
    { "AttrListSetInternalAttrs_Signaler", &createTest<AttrListSetInternalAttrs_Signaler> },
    { "AttrListSetInternalAttrs_Waiter", &createTest<AttrListSetInternalAttrs_Waiter> },
    { "AttrListReconcile", &createTest<AttrListReconcile> },
    { "ObjAlloc", &createTest<ObjAlloc> },
    { "ObjDup", &createTest<ObjDup> },
    { "ObjGetPrimitiveType", &createTest<ObjGetPrimitiveType> },
    { "ObjGetNumPrimitives", &createTest<ObjGetNumPrimitives> },
    { "ObjRef", &createTest<ObjRef> },
    { "FenceExtract", &createTest<FenceExtract> },
    { "FenceUpdate", &createTest<FenceUpdate> },
    { "FenceDup", &createTest<FenceDup> },
};

static void help(void)
{
    printf("\n============================================"\
        "\n LwSciSync KPI:"                                \
        "\n -h: Print this message."                       \
        "\n -a: Run all tests."                            \
        "\n -t <test_name>: Supported tests:\n");
    for (TestMap::iterator itr = testMap.begin();
         itr != testMap.end();
         itr++) {
        printf("      %s\n", itr->first.c_str());
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Too few arguments.\n");
        help();
        return -1;
    }

    if (!strcmp(argv[1], "-t")) {
        // Run a specific test
        if (argc < 3) {
            printf("Incorrect arguments.\n");
            help();
            return -1;
        }
        if (testMap.find(argv[2]) == testMap.end()) {
            printf("Unsupported test case.\n");
            help();
            return -1;
        }

        SyncTest *syncTest = testMap[argv[2]]();
        syncTest->run();
        delete syncTest;

    } else if (!strcmp(argv[1], "-a")) {
        // Run all tests
        for (TestMap::iterator itr = testMap.begin();
            itr != testMap.end();
            itr++) {

            printf("%s (us): ", itr->first.c_str());
            SyncTest *syncTest = itr->second();
            syncTest->run();
            delete syncTest;
        }
    } else if (!strcmp(argv[1], "-h")) {
        help();
    } else {
        printf("Incorrect arguments.\n");
        help();
        return -1;
    }

    return 0;
}