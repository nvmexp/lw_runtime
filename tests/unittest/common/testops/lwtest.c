 /* Copyright 2010-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

//! \file lwtest.c
//! \brief: This file contains Test Case/Suite manipulation Logic

#include <assert.h>
#include <setjmp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include "lwmock.h"
#include "lwtest.h"
#include "utility.h"

void destroyAllList();
void destroyRegopLists();

#ifdef UNIT_LINUX
#include <unistd.h>
#include <signal.h>
#endif

void colorMeRed(const char* str);

static LwTrace* traceStart = NULL;
static LwTrace* traceEnd   = NULL;

#ifdef UNIT_LINUX
sigjmp_buf jmpbuf;
#else
jmp_buf jmpbuf;
#endif

static void runWithExceptionHandled(LwTest* tc);
void runWithExceptionHandledWindows(LwTest* tc);

//
// file level globals Specifying the file and line
// where RM_ASSERT was hit
//
static char* assertInFile = NULL;
static unsigned int assertOnLine = 0;

#define UNKNOWN_LINE_NUMBER 0

static void
LwFailInternal
(
    LwTest* tc,
    const char* file,
    int line,
    LwString* string
);

/*-------------------------------------------------------------------------*
 * LwStr
 *-------------------------------------------------------------------------*/

/*
 * @brief Duplicate data of a string
 *
 * @param src : source string
 *
 * @return destination str pointer
 */
char*
LwStrDupe
(
    const char* src
)
{
    int len = strlen(src);
    char* newStr = (char *)malloc((sizeof(char)) * (len+1));
    strcpy(newStr, src);
    return newStr;
}

/*-------------------------------------------------------------------------*
 * LwString
 *-------------------------------------------------------------------------*/

/*
 * @brief Initialize Lwstring structure
 *
 * Primarly use when dealing with Lwstring Variable
 *
 * @param[in] str  Lwstring which need initialization
 */
void
LwStringInit
(
    LwString* str
)
{
    str->length = 0;
    str->size = STRING_MAX;
    str->buffer = (char*) malloc(sizeof(char) * str->size);
    str->buffer[0] = '\0';
}

/*
 * @brief Alloc and initialize Lwstring Struct
 *
 * Primarly use when dealing with Lwsring pointer
 *
 * @return pointer to Lwstring
 */
LwString*
LwStringNew
(
    void
)
{
    LwString* str = LW_ALLOC(LwString);
    str->length = 0;
    str->size = STRING_MAX;
    str->buffer = (char*) malloc(sizeof(char) * str->size);
    str->buffer[0] = '\0';
    return str;
}

/*
 * @brief Adjust string size
 *
 * @param[in] str      Lwstring whose buffer need to adjust
 * @param[in] newSize : Size with which string need to adjust
 */
void
LwStringResize
(
    LwString* str,
    int newSize
)
{
    str->buffer = (char*) realloc(str->buffer, sizeof(char) * newSize);
    str->size = newSize;
}

/*
 * @brief Append specified string to existing string
 *
 * @param[in] str   String structure pointer
 * @param[in] text  Text to be added
 */
void
LwStringAppend
(
    LwString* str,
    const char* text
)
{
    int length;

    if (text == NULL)
    {
        text = "NULL";
    }

    length = strlen(text);
    if (str->length + length + 1 >= str->size)
        LwStringResize(str, str->length + length + 1 + STRING_INC);
    str->length += length;
    strcat(str->buffer, text);
}

/*
 * @brief Append a char to existing String
 *
 * @param[in] str  String structure pointer,
 * @param[in] ch   Character to be added
 */
void
LwStringAppendChar
(
    LwString* str,
    char ch
)
{
    char text[2];
    text[0] = ch;
    text[1] = '\0';
    LwStringAppend(str, text);
}

/*
 * @brief Append variable length format to String
 *
 * @param[in] str     String structure pointer, format
 * @param[in] format  format to append
 */
void
LwStringAppendFormat
(
    LwString* str,
    const char* format,
    ...
)
{
    va_list argp;
    char buf[HUGE_STRING_LEN];
    va_start(argp, format);
    vsprintf(buf, format, argp);
    va_end(argp);
    LwStringAppend(str, buf);
}

/*
 * @brief Insert text at a given position in the string
 *
 * @param[in] str   string structure pointer
 * @param[in] text  String which need to insert
 * @param[in] pos   Position where to insert "text" in "str"
 */
void
LwStringInsert
(
    LwString* str,
    const char* text,
    int pos
)
{
    int length = strlen(text);
    if (pos > str->length)
        pos = str->length;
    if (str->length + length + 1 >= str->size)
        LwStringResize(str, str->length + length + 1 + STRING_INC);
    memmove(str->buffer + pos + length, str->buffer + pos, (str->length - pos) + 1);
    str->length += length;
    memcpy(str->buffer + pos, text, length);
}

/*
 * @brief Free allocated LwString object
 *
 * @param[in] str   string structure pointer
 */
void LwStringFree(LwString* str)
{
    free(str->buffer);
    free(str);
}

/*-------------------------------------------------------------------------*
 * LwTest
 *-------------------------------------------------------------------------*/

/*
 * @brief Initialize LwTest Struct
 *
 * @param[in]    t      Pointer of LwTest which need to initialize
 * @param[in] name      Name of the test case
 * @param[in] setup     Pointer to Setup funtion for TestCase
 * @param[in] function  Pointer to Actual test logic function
 * @param[in] teardown  Pointer to Teardown function of test case.
 */
void
LwTestInit
(
    LwTest* t,
    const char* name,
    TestFunction setup,
    TestFunction function,
    TestFunction teardown
)
{
    t->name = LwStrDupe(name);
    t->failed = 0;
    t->ran = 0;
    t->bSkip = 0;
    t->message = NULL;
    t->setup = setup;
    t->function = function;
    t->verify = NULL;
    t->teardown = teardown;
}

/*
 * @brief Allocate and initialize LwTest
 *
 * @param[in] name      Name of the test case
 * @param[in] setup     Pointer to setup function for Test case
 * @param[in] function  Pointer to actual test logic function
 * @param[in] teardown  pointer to teardown function of test case
 *
 * @return Pointer of LwTest type
 */
LwTest*
LwTestNew
(
    const char* name,
    TestFunction setup,
    TestFunction function,
    TestFunction teardown
)
{
    LwTest* tc = LW_ALLOC(LwTest);
    LwTestInit(tc, name, setup, function, teardown);
    return tc;
}

/*
 * @brief Fail due to Unexpected Failures(Exceptions/Signals/RM_ASSERT)
 *        Oclwring in the SUT
 *
 * @param[in] tc        Pointer to the the test case
 * @param[in] msg       Message to be printed with the Failure
 * @param[in] file      File where failure oclwred
 * @param[in] line      Line where failure oclwred
 *
 */
void LwFailDueToUnexpectedFailuresInSut
(
    LwTest* tc,
    const char* msg,
    const char* file,
    const int line
)
{
    LwString string;
    LwStringInit(&string);
    LwStringAppend(&string, msg);
    LwFailInternal(tc, file, line, &string);
}

/*
 * @brief Run test case
 *
 * @param[in] tc : Pointer of Test case whose logic needs to execute
 */
void
LwTestRun
(
    LwTest* tc
)
{
    RETURN_FROM_FUNCTION returnFrom;
    tc->ran = 1;

#ifdef UNIT_LINUX
    returnFrom = sigsetjmp(jmpbuf, 1);
#else
    returnFrom = setjmp(jmpbuf);
#endif

    switch (returnFrom)
    {
    case RETURN_FROM_SETJMP:

        if (tc->setup != NULL)
        {
            (tc->setup)(tc);
        }
        if (!tc->failed && (tc->function != NULL))
        {
            runWithExceptionHandled(tc);
        }
        break;

    case RETURN_FROM_LW_FAIL:
        break;
    case RETURN_FROM_EXPECTED_RM_ASSERT:
        if (tc->verify)
            (tc->verify)(tc);

        break;

    case RETURN_FROM_RM_ASSERT:
        {
            const char* msg = "Test Case Failed due to RM_ASSERT failure in SUT";
            LwFailDueToUnexpectedFailuresInSut(tc, msg, assertInFile, assertOnLine);
        }
        break;

    case RETURN_FROM_SIGSEGV:
        {
            const char* msg = "Test Case Failed due to Memory Violation Error/Segmentaion Fault";
            LwFailDueToUnexpectedFailuresInSut(tc, msg, "Unknown File", UNKNOWN_LINE_NUMBER);
        }
        break;

    case RETURN_FROM_SIGBUS:
        {
            const char* msg = "Test Case Failed due to Bus Error/Segmentaion Fault";
            LwFailDueToUnexpectedFailuresInSut(tc, msg, "Unknown File", UNKNOWN_LINE_NUMBER);
        }
        break;

    case RETURN_FROM_ACCESS_VIOLATION_EXCEPTION:
        {
            const char* msg = "Test Case Failed due to Access Violation Exception";
            LwFailDueToUnexpectedFailuresInSut(tc, msg, "Unknown File", UNKNOWN_LINE_NUMBER);
        }
        break;

    }
    if (tc->teardown != NULL)
    {
        (tc->teardown)(tc);
    }
}

/*
 * @brief Update Test status as fail and return
 *
 * Update test case as fail and append file and line number in failing message
 *
 * @param[in] tc      Pointer to testCase for which to update failing log
 * @param[in] file    Str containing file name to appened in failing log
 * @param[in] line    Line number of the failing point
 * @param[in] string  String conatining other error message which need to display
 */
static void
LwFailInternal
(
    LwTest* tc,
    const char* file,
    int line,
    LwString* string
)
{
    char buf[HUGE_STRING_LEN];

    sprintf(buf, "%s:%d: ", file, line);
    LwStringInsert(string, buf, 0);
    LwStringAppend(string, "\n ");

    tc->failed = 1;

    if (tc->message)
    {
        LwStringInsert(string, tc->message, 0);
    }
    tc->message = string->buffer;
}

/*
 * @brief Insert failing message
 *
 * @param[in] tc          Pointer to testCase for which to update failing log
 * @param[in] file        Str containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] message2    Extra message to add
 * @param[in] message     Required message
 * @param[in] exitOnError If error, jump out from the test.
 */
void
LwFail
(
    LwTest* tc,
    const char* file,
    int line,
    const char* message2,
    const char* message,
    int exitOnError
)
{
    LwString string;

    LwStringInit(&string);
    if (message2 != NULL)
    {
        LwStringAppend(&string, message2);
        LwStringAppend(&string, ": ");
    }
    LwStringAppend(&string, message);
    LwFailInternal(tc, file, line, &string);
    if (exitOnError)
    {
        longjmp(jmpbuf, RETURN_FROM_LW_FAIL);
    }
}

/*
 * @brief Insert assert msg
 *
 * @param[in] tc          Pointer to Testcase
 * @param[in] file        String containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] message     Required message if assert fail
 * @param[in] condition   Condition on which the assert is
 * @param[in] exitOnError If error, jump out from the test.
 */
void
LwAssert
(
    LwTest* tc,
    const char* file,
    int line,
    const char* message,
    int condition,
    int exitOnError
)
{
    if (condition) return;
    LwFail(tc, file, line, NULL, message, exitOnError);
}

/*
 * @brief Assert on Strings equals
 *
 * @param[in] tc          Pointer to testCase
 * @param[in] file        String containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] message     Required message if assert fail
 * @param[in] expected    Expected string
 * @param[in] actual      Actual string
 * @param[in] exitOnError If error, jump out from the test.
 */
void
LwAssertStrEquals
(
    LwTest* tc,
    const char* file,
    int line,
    const char* message,
    const char* expected,
    const char* actual,
    int exitOnError
)
{
    char buf[STRING_MAX] = {0};

    if ((expected == NULL && actual == NULL) ||
        (expected != NULL && actual != NULL &&
         strcmp(expected, actual) == 0))
    {
        return;
    }
    sprintf(buf, "expected <%s> but was <%s>", expected, actual);
    LwFail(tc, file, line, message, buf, exitOnError);
}

/*
 * @brief  Assert on integer equals
 *
 * @param[in] tc          Pointer to testCase
 * @param[in] file        String containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] message     Required message if assert fail
 * @param[in] expected    Expected value
 * @param[in] actual      Actual value
 * @param[in] exitOnError If error, jump out from the test.
 */
void
LwAssertIntEquals
(
    LwTest* tc,
    const char* file,
    int line,
    const char* message,
    MOCK_RETURN_TYPE expected,
    MOCK_RETURN_TYPE actual,
    int exitOnError
)
{
    char buf[STRING_MAX] = {0};
    if (expected == actual) return;
    sprintf(buf, "expected <0x%08x> but was <0x%08x>", expected, actual);
    LwFail(tc, file, line, message, buf, exitOnError);
}

/*
 * @brief Assert on integer equals
 *
 * @param[in] tc          Pointer to testCase
 * @param[in] file        String containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] message     Required message if assert fail
 * @param[in] expected    Expected value
 * @param[in] actual      Actual value
 * @param[in] delt        Tolerence part
 * @param[in] exitOnError If error, jump out from the test.
 */
void
LwAssertDblEquals
(
    LwTest* tc,
    const char* file,
    int line,
    const char* message,
    double expected,
    double actual,
    double delta,
    int exitOnError
)
{
    char buf[STRING_MAX] = {0};
    if (fabs(expected - actual) <= delta) return;
    sprintf(buf, "expected <%lf> but was <%lf>", expected, actual);
    LwFail(tc, file, line, message, buf, exitOnError);
}

/*
 * @brief Assert on Pointer equals
 *
 * @param[in] tc          Pointer to testCase
 * @param[in] file        String containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] message     Required message if assert fail
 * @param[in] expected    Expected value
 * @param[in] actual      Actual value
 * @param[in] exitOnError If error, jump out from the test.
 */
void
LwAssertPtrEquals
(
    LwTest* tc,
    const char* file,
    int line,
    const char* message,
    void* expected,
    void* actual,
    int exitOnError
)
{
    char buf[STRING_MAX] = {0};
    if (expected == actual) return;
    sprintf(buf, "expected pointer <0x%p> but was <0x%p>", expected, actual);
    LwFail(tc, file, line, message, buf, exitOnError);
}

/*
 * @brief Assert on "v1 > v2"
 *
 * @param[in] tc          Pointer to testCase
 * @param[in] file        String containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] message     Required message if assert fail
 * @param[in] v1          value 1
 * @param[in] v2          value 2
 * @param[in] exitOnError If error, jump out from the test.
 */
void
LwAssertGreater
(
    LwTest* tc,
    const char* file,
    int line,
    const char* message,
    MOCK_RETURN_TYPE v1,
    MOCK_RETURN_TYPE v2,
    int exitOnError
)
{
    char buf[STRING_MAX] = {0};
    if (v1 > v2) return;
    sprintf(buf, "expected <%d> greater than <%d>", v1, v2);
    LwFail(tc, file, line, message, buf, exitOnError);
}

/*
 * @brief Assert on "v1 >= v2"
 *
 * @param[in] tc          Pointer to testCase
 * @param[in] file        String containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] message     Required message if assert fail
 * @param[in] v1          value 1
 * @param[in] v2          value 2
 * @param[in] exitOnError If error, jump out from the test.
 */
void
LwAssertGreaterOrEquals
(
    LwTest* tc,
    const char* file,
    int line,
    const char* message,
    MOCK_RETURN_TYPE v1,
    MOCK_RETURN_TYPE v2,
    int exitOnError
)
{
    char buf[STRING_MAX] = {0};
    if (v1 >= v2) return;
    sprintf(buf, "expected <%d> greater than or equal to <%d>", v1, v2);
    LwFail(tc, file, line, message, buf, exitOnError);
}

/*
 * @brief Assert on "v1 < v2"
 *
 * @param[in] tc          Pointer to testCase
 * @param[in] file        String containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] message     Required message if assert fail
 * @param[in] v1          value 1
 * @param[in] v2          value 2
 * @param[in] exitOnError If error, jump out from the test.
 */
void
LwAssertLess
(
    LwTest* tc,
    const char* file,
    int line,
    const char* message,
    MOCK_RETURN_TYPE v1,
    MOCK_RETURN_TYPE v2,
    int exitOnError
)
{
    char buf[STRING_MAX] = {0};
    if (v1 < v2) return;
    sprintf(buf, "expected <%d> less than <%d>", v1, v2);
    LwFail(tc, file, line, message, buf, exitOnError);
}

/*
 * @brief Assert on "v1 <= v2"
 *
 * @param[in] tc          Pointer to testCase
 * @param[in] file        String containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] message     Required message if assert fail
 * @param[in] v1          value 1
 * @param[in] v2          value 2
 * @param[in] exitOnError If error, jump out from the test.
 */
void
LwAssertLessOrEquals
(
    LwTest* tc,
    const char* file,
    int line,
    const char* message,
    MOCK_RETURN_TYPE v1,
    MOCK_RETURN_TYPE v2,
    int exitOnError
)
{
    char buf[STRING_MAX] = {0};
    if (v1 <= v2) return;
    sprintf(buf, "expected <%d> less than or equal to <%d>", v1, v2);
    LwFail(tc, file, line, message, buf, exitOnError);
}

/*
 * @brief Assert on array values
 *
 * @param[in] tc          Pointer to testCase
 * @param[in] file        String containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] message     Required message if assert fail
 * @param[in] expected    Expected array
 * @param[in] actual      Actual array
 * @param[in] count       Number of array elements to compare
 * @param[in] exitOnError If error, jump out from the test.
 */
void
LwAssertArrayEquals
(
    LwTest* tc,
    const char* file,
    int line,
    const char* message,
    void* expected,
    void* actual,
    int count,
    int exitOnError
)
{
    unsigned char *ex8 = (unsigned char*)expected;
    unsigned char *ac8 = (unsigned char*)actual;
    char buf[STRING_MAX] = {0};
    int  x;

    for (x = 0; x < count; x++)
    {
        if (ex8[x] != ac8[x])
            break;
    }
    if (x == count) return;

    sprintf(buf, "expected <0x%02x> but was <0x%02x> at byte offset <%d>", ex8[x], ac8[x], x);
    LwFail(tc, file, line, message, buf, exitOnError);
}

/*
 * @brief call User defined Assert function
 *
 * @param[in] tc          Pointer to testCase
 * @param[in] file        String containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] func        function pointer to Assert function
 * @param[in] arg         Arguments used by user defined assert function
 * @param[in] message     Required message if assert fail
 * @param[in] exitOnError If error, jump out from the test.
 */
void LwLwstomAssert
(
    LwTest* tc,
    const char* file,
    int line,
    LwstomAssertFunc func,
    void *arg,
    const char* message,
    int exitOnError
)
{
    int status;
    char buf[STRING_MAX] = {0};

    status = func(arg);
    if (!status)
    {
        LwFail(tc, file, line, message, buf, exitOnError);
    }
}

// only for RmUnit
#ifdef RM_UNITTEST
/*
 * @brief Verify RM_ASSERT status
 *
 * @param[in] tc          Pointer to testCase
 * @param[in] file        String containing file name to appened in failing log
 * @param[in] line        Line number of the failing point
 * @param[in] message     Required message if assert fail
 * @param[in] status      Expected RM_ASSERT value
 * @param[in] count       index of RM_ASSERT in SUT which need to verify
 * @param[in] exitOnError If error, jump out from the test.
 */
void
LwVerifRmAssert
(
    LwTest* tc,
    const char* file,
    int line,
    const char* message,
    int status,
    int count,
    int exitOnError
)
{
    char buf[STRING_MAX] = {0};

    if ( verifyRmAssertHit(status, count) ) return;

    if (status)
    {
        sprintf(buf, "expected <%d> RM_ASSERT to be TRUE, got FALSE", count);
    }
    else
    {
        sprintf(buf, "expected <%d> RM_ASSERT to be FALSE, got TRUE", count);
    }

    LwFail(tc, file, line, message, buf, exitOnError);
}

#endif // RM_UNITTEST
/*-------------------------------------------------------------------------*
 * LwSuite
 *-------------------------------------------------------------------------*/

//!
//! \brief LwSuiteInit : Initialize Test Suite
//!
//! \param testSuite : Specify pointer to test Suite
//!
void
LwSuiteInit
(
    LwSuite* testSuite
)
{
    testSuite->numTCs      = 0;
    testSuite->numChildren = 0;
    testSuite->failCount   = 0;
    testSuite->numTCExelwted = 0;
}

/*
 * @brief Allocate and initialize new suite
 *
 * @return suite pointer
 */
LwSuite*
LwSuiteNew
(
    SuiteSetup setupFn,
    SuiteTeardown teardownFn,
    const char *suiteName
)
{
    int i;
    LwSuite* testSuite = LW_ALLOC(LwSuite);
    LwSuiteInit(testSuite);

    testSuite->name     = LwStrDupe(suiteName);
    testSuite->setup    = setupFn;
    testSuite->teardown = teardownFn;
    testSuite->parent   = NULL;
    for (i = 0; i < MAX_SUB_SUITES; i++)
    {
        testSuite->children[i] = NULL;
    }

    return testSuite;
}

/*
 * @brief Add testcase to testsuite
 *
 * @param[in] testSuite  Pointer to Suite in which to add test case.
 * @param[in] testCase   Pointer to test case which need to add
 */
void
LwSuiteAddTest
(
    LwSuite* testSuite,
    LwTest *testCase
)
{
    assert(testSuite->numTCs < MAX_TEST_CASES);
    testSuite->list[testSuite->numTCs] = testCase;
    testSuite->numTCs++;
}

/*
 * @brief Add Suite within suite
 *
 * @param[in] parentSuite  Pointer to suite under which we need to add a suite
 * @param[in] childSuite   Pointer to suite which need to add under parent suite
 */
void
LwSuiteAddSuite
(
    LwSuite* parentSuite,
    LwSuite* childSuite
)
{
    assert((parentSuite->numChildren + 1) < MAX_SUB_SUITES);
    parentSuite->children[parentSuite->numChildren] = childSuite;
    parentSuite->numChildren++;
    childSuite->parent = parentSuite;
}

/*
 * @brief Execute testcase under specified Suite in DFS way
 *
 * @params[in] testSuite  Suite pointer whose child needs exelwtion
 */
void
LwSuiteRun
(
    LwSuite* testSuite,
    unsigned int logVerbose
)
{
    int i;
    int bAnyTestEnabled;

    LwTracePush(testSuite);
    for (i = 0; i < testSuite->numChildren; i++)
    {
        LwSuiteRun(testSuite->children[i], logVerbose);
    }

    bAnyTestEnabled = 0;
    for (i = 0; i < testSuite->numTCs; ++i)
    {
        LwTest *testCase;
        testCase = testSuite->list[i];

        if (i==0 && logVerbose)
            printf("SuiteName : %s\n", testSuite->name);

        if(bAnyTestEnabled == 0 && (!testCase->bSkip))
        {
            LwTraceSetup();
            bAnyTestEnabled = 1;
        }

        if (!testCase->bSkip)
        {
            LwTestRun(testCase);

            _LwMockTeardown(testCase);
            destroyRegopLists();
// only for RmUnit
#ifdef RM_UNITTEST
            destroyAllList();
#endif // RM_UNITTEST
            utApiEnableDbgPrintf(LW_FALSE);

            if (testCase->failed)
            {
                testSuite->failCount += 1;
                colorMeRed("F");
            }
            else
            {
                printf(".");
            }
            testSuite->numTCExelwted += 1;
        }
        else
        {
            printf("S");
        }
    }

    if(bAnyTestEnabled)
    {
        LwTraceTeardown();
        LwTracePop();
    } // if(bAnyTestEnabled)

    if (logVerbose && i)
        printf("\n");
}

/*
 * @brief Lunch Suite setup
 *
 * @param[in] suite  Suite pointer
 */
void
LwSuiteSetup
(
    LwSuite* suite
)
{
    if (suite->setup)
    {
        suite->setup();
    }
}

/*
 * @brief Returns number of testcases in a suite
 *
 * @param suite  Suite pointer
 */
int
LwSuiteTCCount
(
    LwSuite* suite
)
{
    int i;
    int numTCs;

    if (suite == NULL)
        return 0;

    numTCs = suite->numTCs;
    for (i = 0; i < suite->numChildren; i++)
        numTCs += LwSuiteTCCount(suite->children[i]);
    return numTCs;
}

/*
 * @brief Returns number of testcases exelwted for specified Suite
 *
 * @param[in] suite  Suite pointer
 */
int
LwSuiteTCExelwtedCount
(
    LwSuite* suite
)
{
    int i;
    int numTCExelwted;

    if (suite == NULL)
        return 0;

    numTCExelwted = suite->numTCExelwted;
    for (i = 0; i < suite->numChildren; i++)
        numTCExelwted += LwSuiteTCExelwtedCount(suite->children[i]);
    return numTCExelwted;
}

/*
 * @brief Free all the allocations made by suite
 *
 * @param[in] suite   Suite pointer
 */
void
LwSuiteTeardown
(
    LwSuite* suite
)
{
    if (suite->teardown)
    {
        suite->teardown();
    }
}

/*
 * @brief Return the number of testcases that fail
 *
 * @param[in] suite  Suite pointer
 */
int
LwSuiteFailCount
(
    LwSuite* suite
)
{
    int i;
    int failCount;

    if (suite == NULL)
        return 0;

    failCount = suite->failCount;
    for (i = 0; i < suite->numChildren; i++)
        failCount += LwSuiteFailCount(suite->children[i]);
    return failCount;
}

/*
 * @brief Gives details of failure of the suite
 *
 * @param[in] testSuite  Suite about which to get the details
 * @param[in] details    String where to store the detail
 * @param[out] failCount Failure test count
 */
void
LwSuiteFailures
(
    LwSuite* testSuite,
    LwString* details,
    int *failCount
)
{
    int i;
    for (i = 0; i < testSuite->numChildren; i++)
    {
        LwSuiteFailures(testSuite->children[i], details, failCount);
    }
    for (i = 0 ; i < testSuite->numTCs ; ++i)
    {
        LwTest* testCase = testSuite->list[i];
        if (testCase->failed)
        {
            (*failCount)++;
            LwStringAppendFormat(details, "%d) For %s:\n %s\n",
                *failCount, testCase->name, testCase->message);
        }
    }
}

/*
 * @brief  Detail about the suite
 *
 * @param[in] testSuite  Suite pointer about which to get the detail
 * @param[in] details    String where to store the detail
 */
void
LwSuiteDetails
(
    LwSuite* testSuite,
    LwString* details
)
{
    int failCount = LwSuiteFailCount(testSuite);
    int numTCs    = LwSuiteTCCount(testSuite);
    int numTCExelwted = LwSuiteTCExelwtedCount(testSuite);
    int skippedTest = numTCs - numTCExelwted;
    int passCount = numTCExelwted - failCount;

    if (failCount == 0)
    {
        FILE *pDummySuccessFile;
        char *fileName = "Success.txt";
        // check if UNIT_ASSERT was hit
        if (checkIfAssertLogFile())
            printf("\nUNIT_ASSERT was hit, check unitTestAssert.log(in present directory) for more details \n");
        else
        {
            // empty success file is added if everyting works fine
            // to check passing criteria on dvs

            if((pDummySuccessFile = fopen(fileName, "w")) == NULL)
            {
                printf("\nall tests ran successfully, but problem in creating success.txt \n");
                return;
            }
            fclose(pDummySuccessFile);
        }
        LwStringAppendFormat(details, "\n\nTotal TCs   : %d\n", numTCs);
    }
    else
    {
        if (failCount == 1)
            LwStringAppend(details, "\n\nThere was 1 failure:\n");
        else
            LwStringAppendFormat(details, "\nThere were %d failures:\n", failCount);

        if (checkIfAssertLogFile())
            printf("\nUNIT_ASSERT was hit, check unitTestAssert.log(in present directory) for more details \n");

        failCount = 0;
        LwSuiteFailures(testSuite, details, &failCount);
        colorMeRed("\n\n!!!FAILURES!!!");
        LwStringAppendFormat(details, "\nTotal TCs   : %d\n", numTCs);
    }

    LwStringAppendFormat(details, "TCs Passed  : %d\n", passCount);
    LwStringAppendFormat(details, "TCs Failed  : %d\n",  failCount);
    LwStringAppendFormat(details, "TCs Skipped : %d\n", skippedTest);
}

/*
 * @brief  Free suite from stack as its all childs got exelwted
 */
void
LwTracePop
(
    void
)
{
    LwTrace* detached;
    if (traceEnd == NULL)
        return;
    detached = traceEnd;
    if (traceEnd->prev != NULL)
    {
        traceEnd->prev->next = NULL;
        traceEnd = traceEnd->prev;
    }
    free(detached);
}

/*
 * @brief Push suite in stack in order to schedule its children
 *        for exelwtion
 *
 * @param[in] testSuite  Suite which need to push
 */
void
LwTracePush
(
    LwSuite* testSuite
)
{
    LwTrace* node = LW_ALLOC(LwTrace);
    assert(node != NULL);
    node->next  = NULL;
    node->suite = testSuite;

    if (traceStart != NULL)
    {
        node->prev = traceEnd;
        traceEnd->next = node;
        traceEnd       = node;
    }
    else
    {
        node->prev = NULL;
        traceStart = node;
        traceEnd   = node;
    }
}

/*
 * @brief Execute Test Suite Setup
 */
void
LwTraceSetup
(
    void
)
{
    LwTrace *node = traceStart;
    while (node != NULL)
    {
        LwSuiteSetup(node->suite);
        node = node->next;
    }
}

/*
 * @brief Free all allocations made for Test Suite
 */
void
LwTraceTeardown
(
    void
)
{
    LwTrace *node = traceEnd;
    while (node != NULL)
    {
        LwSuiteTeardown(node->suite);
        node = node->prev;
    }
}

/*
 * @brief Destroy TestCase structure while cleanup
 *
 * @param[in] tc  testcase which need to destroy
 */
void
DestroyTestCase
(
    LwTest *tc
)
{
    // free the memory used for assertInFile if still allocated
    if (assertInFile)
    {
        free(assertInFile);
        assertInFile = NULL;
    }

    free((char *)tc->name);
    tc->setup = NULL;
    tc->function = NULL;
    tc->teardown = NULL;
    free((char *)tc->message);
    free(tc);
}

/*
 * @brief Destroy Suite structure while cleanup
 *
 * @param[in] testSuite  Suite which need to destroy
 */
void
DestroySuite
(
    LwSuite* testSuite
)
{
    int i;
    for (i = 0; i < testSuite->numChildren; i++)
    {
        DestroySuite(testSuite->children[i]);
    }

    for (i = 0; i < testSuite->numTCs; ++i)
    {
        DestroyTestCase(testSuite->list[i]);
        testSuite->list[i] = NULL;
    }
    free(testSuite->name);
    testSuite->setup = NULL;
    testSuite->teardown = NULL;
    free(testSuite);
}

/*
 * @brief Get Pointer to Test case of specified test
 *        under specified testSuite
 *
 * @param[in] testSuite  Suite, which needs to search.
 * @param[in] tcName     Name of test case
 */
LwTest *
GetTestCase
(
    LwSuite* testSuite,
    const char *tcName
)
{
    int i;
    LwTest *testCase = NULL;
    LwString *str = LwStringNew();
    LwStringAppend(str, "(TestFunction)");
    LwStringAppend(str, tcName);

    for (i = 0; i < testSuite->numTCs; i++)
    {
        testCase = testSuite->list[i];
        if (!strcmp(testCase->name, str->buffer))
            break;
        testCase = NULL;
    }

    free(str->buffer);
    free(str);
    return testCase;
}

/*
 * @brief Disable all test cases under specified Suite
 *
 * @param[in] testSuite  Suite, whoes tests need to disable
 */
void
SkipAllTests
(
    LwSuite* testSuite
)
{
    int i;

    for (i = 0; i < testSuite->numChildren; i++)
    {
        SkipAllTests(testSuite->children[i]);
    }

    for (i = 0; i < testSuite->numTCs; ++i)
    {
        testSuite->list[i]->bSkip = 1;
    }
}

/*
 * @brief Fail the test since, RM_ASSERT has failed in the SUT
 *
 * @param[in] file  file name where assert was hit
 *
 * @param[in] line  Line on which Assert was hit
 */
void
failDueToRmAssert(char *file, unsigned int line)
{
    if (assertInFile)
    {
        free(assertInFile);
        assertInFile = NULL;
    }

    assertInFile = (char *)malloc(sizeof(char)*( strlen(file)+1 ));
    strcpy(assertInFile, file);
    assertOnLine = line;
    longjmp(jmpbuf, RETURN_FROM_RM_ASSERT);
}

/*
 * @brief Return from the SUT since RM_ASSERT has oclwred
 *        though, this one is expected We can not continue
 *        as it might result in seg fault or application crash
 */
void
returnDueToExpectedRmAssert()
{
    longjmp(jmpbuf, RETURN_FROM_EXPECTED_RM_ASSERT);
}

#ifdef UNIT_LINUX
// following code Valid only for Linux environment

// structs to hold signal information
static struct sigaction actNewSegv;
static struct sigaction actOldSegv;
static struct sigaction actNewBus;
static struct sigaction actOldBus;

/*
 * @brief handle SIGSEGV signal
 *
 * @param[in] signum  Signal being handled
 *
 */
static void handleExceptionSegv(int signum)
{
    siglongjmp(jmpbuf, RETURN_FROM_SIGSEGV);
}

/*
 * @brief handle SIGBUS signal
 *
 * @param[in] signum  Signal being handled
 *
 */
static void handleExceptionBus(int signum)
{
    siglongjmp(jmpbuf, RETURN_FROM_SIGBUS);
}

/*
 * @brief install exception handler
 *
 */
static void installExceptionHandler()
{
    memset (&actNewSegv, 0, sizeof(actNewSegv));
    memset (&actOldSegv, 0, sizeof(actOldSegv));
    memset (&actNewBus, 0, sizeof(actNewBus));
    memset (&actOldBus, 0, sizeof(actOldBus));

    actNewSegv.sa_handler = &handleExceptionSegv;
    actNewSegv.sa_flags = SA_RESETHAND;

    if (sigaction(SIGSEGV, &actNewSegv, &actOldSegv) != 0)
    {
        printf ("%s:%d:Unable to install Signal Handler, Exiting ...", __FUNCTION__, __LINE__);
        kill(getpid(), SIGKILL);
    }

    actNewBus.sa_handler = &handleExceptionBus;
    actNewBus.sa_flags = SA_RESETHAND;

    if (sigaction(SIGBUS, &actNewBus, &actOldBus) != 0)
    {
        printf ("%s:%d:Unable to install Signal Handler, Exiting ...", __FUNCTION__, __LINE__);
        kill(getpid(), SIGKILL);
    }

}

#endif // UNIT_LINUX

/*
 * @brief run the test case with exceptions handled
 *
 * @param[in] tc  Pointer to test case to be run
 *
 */
static void runWithExceptionHandled(LwTest* tc)
{
#ifdef UNIT_LINUX

    installExceptionHandler();

    // execute the test function
    (tc->function)(tc);

    // if there is a verify function, then execute it
    if (tc->verify != NULL)
    {
        (tc->verify)(tc);
    }

    // restore signal data structure
    if (sigaction(SIGSEGV, &actOldSegv, NULL) != 0)
    {
        printf ("%s:%d:Unable to Restore Signal Data Structure, Exiting ...", __FUNCTION__, __LINE__);
        kill(getpid(), SIGKILL);
    }

    if (sigaction(SIGBUS, &actOldBus, NULL) != 0)
    {
        printf ("%s:%d:Unable to Restore Signal Data Structure, Exiting ...", __FUNCTION__, __LINE__);
        kill(getpid(), SIGKILL);
    }

#else

#ifdef UNIT_WINDOWS

    runWithExceptionHandledWindows(tc);

#else

    // execute the test function
    (tc->function)(tc);

    // if there is a verify function, then execute it
    if (tc->verify != NULL)
    {
        (tc->verify)(tc);
    }

#endif // UNIT_WINDOWS

#endif // UNIT_LINUX
}
