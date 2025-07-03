
/* Copyright by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* LWIDIA_COPYRIGHT_END
*/

//! \file unizen.i
//! \brief: This is the definition file used to generate extensions.c via swig

%module unizen

%{
#include "lwtest.h"
#include "alltests.h"
%}

LwSuite* LwSuiteNew(SuiteSetup Setup, SuiteTeardown Teardown, const char *suiteName);
void LwSuiteAddSuite(LwSuite* parentSuite, LwSuite* childSuite);
void LwSuiteRun(LwSuite* testSuite);
LwTest *GetTestCase(LwSuite* testSuite, const char *tcName);
LwString* LwStringNew(void);
void LwSuiteDetails(LwSuite* testSuite, LwString* details);
void SkipAllTests(LwSuite* testSuite);


typedef void (*TestFunction)(LwTest *);
typedef void (*SuiteSetup)   (void);
typedef void (*SuiteTeardown)(void);

typedef struct
{
    int length;
    int size;
    char* buffer;
} LwString;

struct LwTest
{
    const char* name;
    TestFunction setup;
    TestFunction function;
    TestFunction teardown;
    int failed;
    int ran;
    int bSkip;
    const char* message;
    const void* params;
};
struct LwSuite
{
    char *name;
    int numTCs;
    int numChildren;
    LwSuite* parent;
    LwSuite* children[MAX_SUB_SUITES];
    LwTest*  list[MAX_TEST_CASES];
    int failCount;
    SuiteSetup setup;
    SuiteTeardown teardown;
};

// all suite declarations go here

LwSuite* suite_fifoServiceTop_GF100();
