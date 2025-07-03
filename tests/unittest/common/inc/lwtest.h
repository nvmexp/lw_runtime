 /* Copyright 2010-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

//! \file lwtest.h
//! \brief: This file contain all the function and data structure declration
//!         required for core unittest infra.

#ifndef _LWTEST_H_
#define _LWTEST_H_

#include <stdio.h>
#include "lwtypes.h"
#include "rmassert.h"
#include <setjmp.h>

typedef LwSPtr MOCK_RETURN_TYPE;
typedef LwUPtr MOCK_PTR_TYPE;

/* LwString */
#define LW_ALLOC(TYPE)  ((TYPE*) malloc(sizeof(TYPE)))

#define HUGE_STRING_LEN 8192
#define STRING_MAX      512
#define STRING_INC      512

typedef struct
{
    int length;
    int size;
    char* buffer;
} LwString;

char* LwStrDupe(const char* old);
void LwStringInit(LwString* str);
LwString* LwStringNew(void);
void LwStringAppend(LwString* str, const char* text);
void LwStringAppendChar(LwString* str, char ch);
void LwStringAppendFormat(LwString* str, const char* format, ...);
void LwStringInsert(LwString* str, const char* text, int pos);
void LwStringResize(LwString* str, int newSize);
void LwStringFree(LwString* str);

/* LwTest */
typedef struct LwTest LwTest;
typedef void (*TestFunction)(LwTest *);
typedef int (*LwstomAssertFunc) (void *);

// TO DO: Provide an API to add a test that include specifying Verif section
struct LwTest
{
    const char* name;
    TestFunction setup;
    TestFunction function;
    TestFunction verify;
    TestFunction teardown;
    int failed;
    int ran;
    int bSkip;
    const char* message;
    const void* params;
};

void LwTestInit(LwTest* tc, const char* name, TestFunction setup,
                TestFunction function, TestFunction teardown);
LwTest* LwTestNew(const char* name, TestFunction setup, TestFunction function,
                  TestFunction teardown);
void LwTestRun(LwTest* tc);
void LwFail(LwTest* tc, const char* file, int line, const char* message2,
                 const char* message, int exitOnError);
void LwAssert(LwTest* tc, const char* file, int line, const char* message,
                   int condition, int exitOnError);
void LwAssertStrEquals(LwTest* tc, const char* file, int line,
                               const char* message, const char* expected,
                               const char* actual, int exitOnError);
void LwAssertIntEquals(LwTest* tc, const char* file, int line,
                               const char* message, MOCK_RETURN_TYPE  expected, MOCK_RETURN_TYPE  actual, int exitOnError);
void LwAssertDblEquals(LwTest* tc, const char* file, int line,
                               const char* message, double expected,
                               double actual, double delta, int exitOnError);
void LwAssertPtrEquals(LwTest* tc, const char* file, int line,
                               const char* message, void* expected,
                               void* actual, int exitOnError);
void LwAssertGreater(LwTest* tc, const char* file, int line,
                             const char* message, MOCK_RETURN_TYPE v1, MOCK_RETURN_TYPE v2, int exitOnError);
void LwAssertGreaterOrEquals(LwTest* tc, const char* file, int line,
                                     const char* message, MOCK_RETURN_TYPE v1, MOCK_RETURN_TYPE v2, int exitOnError);

void LwAssertLess(LwTest* tc, const char* file, int line,
                          const char* message, MOCK_RETURN_TYPE v1, MOCK_RETURN_TYPE v2, int exitOnError);
void LwAssertLessOrEquals(LwTest* tc, const char* file, int line,
                                  const char* message, MOCK_RETURN_TYPE v1, MOCK_RETURN_TYPE v2, int exitOnError);
void LwAssertArrayEquals(LwTest* tc, const char* file, int line,
                                 const char* message, void* expected,
                                 void* actual, int count, int exitOnError);
void LwLwstomAssert(LwTest* tc, const char* file, int line,
                    LwstomAssertFunc func, void *arg, const char* message, int exitOnError);
void LwVerifRmAssert(LwTest* tc, const char* file, int line,
                     const char* message, int status, int count, int exitOnError);

/* LwSuite */

#define MAX_TEST_CASES  1024
#define MAX_SUB_SUITES  16

typedef void (*SuiteSetup)   (void);
typedef void (*SuiteTeardown)(void);

typedef struct LwSuite LwSuite;
struct LwSuite
{
    char *name;
    int numTCs;
    int numChildren;
    LwSuite* parent;
    LwSuite* children[MAX_SUB_SUITES];
    LwTest*  list[MAX_TEST_CASES];
    int failCount;
    int numTCExelwted;
    SuiteSetup setup;
    SuiteTeardown teardown;
};

typedef struct LwTrace LwTrace;
struct LwTrace
{
    LwSuite* suite;
    LwTrace* next;
    LwTrace* prev;
};

//
//enums to distinguish between various points
// from where the control is returned via longjump
//
typedef enum RETURN_FROM_FUNCTION {
    RETURN_FROM_SETJMP,
    RETURN_FROM_LW_FAIL,
    RETURN_FROM_RM_ASSERT,
    RETURN_FROM_EXPECTED_RM_ASSERT,
    RETURN_FROM_SIGSEGV,
    RETURN_FROM_SIGBUS,
    RETURN_FROM_ACCESS_VIOLATION_EXCEPTION,
} RETURN_FROM_FUNCTION;

void LwSuiteInit(LwSuite* testSuite);
LwSuite* LwSuiteNew(SuiteSetup Setup, SuiteTeardown Teardown, const char *suiteName);
void LwSuiteAddTest(LwSuite* testSuite, LwTest *testCase);
void LwSuiteAddSuite(LwSuite* parentSuite, LwSuite* childSuite);
int LwSuiteTCCount(LwSuite* suite);
int LwSuiteTCExelwtedCount(LwSuite* suite);
void LwSuiteRun(LwSuite* testSuite, unsigned int);
void LwSuiteSetup(LwSuite* suite);
void LwSuiteTeardown(LwSuite* suite);
void LwSuiteDetails(LwSuite* testSuite, LwString* details);
void LwTracePop(void);
void LwTracePush(LwSuite* testSuite);
void LwTraceSetup(void);
void LwTraceTeardown(void);
void DestroySuite(LwSuite* testSuite);
void DestroyTestCase(LwTest *tc);
LwTest *GetTestCase(LwSuite* testSuite, const char *tcName);
void SkipAllTests(LwSuite* testSuite);
void failDueToRmAssert(char *file, unsigned int line);
void returnDueToExpectedRmAssert();

#endif /* _LWTEST_H_ */
