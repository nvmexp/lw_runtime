//! \file
//! \brief LwSciBuf kpi perf test.
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

template<typename T> BufTest * createTest() { return new T; }

typedef std::unordered_map<std::string, BufTest*(*)()> TestMap;

TestMap testMap{
    { "ModuleOpen", &createTest<ModuleOpen> },
    { "AttrListCreate", &createTest<AttrListCreate> },
    { "AttrListSetAttrs_Camera", &createTest<AttrListSetAttrs_Camera> },
    { "AttrListSetAttrs_ISP", &createTest<AttrListSetAttrs_ISP> },
    { "AttrListSetAttrs_Display", &createTest<AttrListSetAttrs_Display> },
    { "AttrListSetInternalAttrs_Camera", &createTest<AttrListSetInternalAttrs_Camera> },
    { "AttrListSetInternalAttrs_ISP", &createTest<AttrListSetInternalAttrs_ISP> },
    { "AttrListSetInternalAttrs_Display", &createTest<AttrListSetInternalAttrs_Display> },
    { "AttrListReconcile_Camera", &createTest<AttrListReconcile_Camera> },
    { "AttrListReconcile_Isp_Display", &createTest<AttrListReconcile_Isp_Display> },
    { "ObjAlloc", &createTest<ObjAlloc> },
    { "ObjRef", &createTest<ObjRef> },
};

static void help(void)
{
    printf("\n============================================"\
        "\n LwSciBuf KPI:"                                \
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

        BufTest *bufTest = testMap[argv[2]]();
        bufTest->run();
        delete bufTest;

    } else if (!strcmp(argv[1], "-a")) {
        // Run all tests
        for (TestMap::iterator itr = testMap.begin();
            itr != testMap.end();
            itr++) {

            printf("%s (us): ", itr->first.c_str());
            BufTest *bufTest = itr->second();
            bufTest->run();
            delete bufTest;
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
