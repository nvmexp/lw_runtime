/*
 * Copyright (c) 2007-2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//###########################################################################
//############################### INCLUDES ##################################
//###########################################################################

#include "lwassert.h"
#include "lwutil.h"
#include "lwtest.h"
#include "string.h"

#if LWOS_IS_WINDOWS || LWOS_IS_LINUX || LWOS_IS_QNX
#include <stdio.h>
#endif

#if LWOS_IS_LINUX
#include <fcntl.h>
#include <unistd.h>
#endif

//###########################################################################
//############################### DEFINES ###################################
//###########################################################################

#define DBERR(err) (err)
#define MAX_SUBTEST_NAME_LENGTH 80

//###########################################################################
//############################### TYPEDEFS ##################################
//###########################################################################

//===========================================================================
// LwTestAppState - states that an application can be in
//===========================================================================
typedef enum LwTestAppStateEnum {
    LwTestAppState_Uninitialized,
    LwTestAppState_Initialized,
    LwTestAppState_RanTests,
    LwTestAppState_Terminated,

    LwTestAppState_Count,
    LwTestAppState_Force32          =   0x7fffffff
} LwTestAppState;

//===========================================================================
// LwTestSubtest - info about 1 test
//===========================================================================
typedef struct LwTestSubtestRec {
    struct LwTestAppRec *app;
    char                *name;      // name of test
    LwError              result;
    LwU32                index;     // test index
    LwU32                startTime;
    LwBool               skip;
} LwTestSubtest;

//===========================================================================
// LwTestApp - Global information about an application
//===========================================================================
typedef struct LwTestAppRec {
    LwU32           useCount;      // # of times LwTestInitialize() called
    LwTestAppState  state;         // state of the app

    LwU32           testCnt;       // number of tests run so far
    LwU32           failureCnt;    // number of tests that failed so far
    LwU32           filteredCnt;   // Number of tests filtered out
    LwError         err;           // return value of program

    LwU32           tlsIndex;      // per-thread data index (current subtest)
    LwOsMutexHandle mutex;         // mutex for thread accesses to LwTestApp

    LwTestSubtest  *failHead;      // head of failure list
    LwTestSubtest **failTail;      // tail of failure list

    void          (*lwRun)(LwU32); // Points to LwRun() function

    char *subtestFilter;            // Filter for subtests
    char *subtestList;              // subtest list from sanity harness

    LwBool rest;                    // Whether to print results in REST format
    LwBool listAll;                 // Whether to list tests
} LwTestApp;

//###########################################################################
//############################### GLOBALS ###################################
//###########################################################################

static LwTestApp *gs_app=0;

#if LWOS_IS_LINUX
static int marker_fd = -1;
#endif

//###########################################################################
//############################### CODE ######################################
//###########################################################################

//===========================================================================
// LwTestGetLwrSubtest() - return the current subtest for this thread
//===========================================================================
static LwTestSubtest* LwTestGetLwrSubtest(void)
{
    LW_ASSERT(gs_app && gs_app->tlsIndex != LWOS_ILWALID_TLS_INDEX);

    return LwOsTlsGet(gs_app->tlsIndex);
}

//===========================================================================
// LwTestSetLwrSubtest() - set the current subtest for this thread
//===========================================================================
static void LwTestSetLwrSubtest(LwTestSubtest* sub)
{
    LW_ASSERT(gs_app && gs_app->tlsIndex != LWOS_ILWALID_TLS_INDEX);

    LwOsTlsSet(gs_app->tlsIndex, sub);
}

#if LWOS_IS_LINUX

//===========================================================================
// LwTestOpenFTrace() - Open ftrace for writing.
//===========================================================================
static void LwTestOpenFTrace(void)
{
    if(marker_fd >= 0)
        return;

    marker_fd = open("/sys/kernel/debug/tracing/trace_marker", O_WRONLY);
}

//===========================================================================
// LwTestCloseFTrace() - Close ftrace
//===========================================================================
static void LwTestCloseFTrace(void)
{
    if(marker_fd < 0)
        return;

    close(marker_fd);
    marker_fd = -1;
}

//===========================================================================
// LwTestWriteFTrace() - Write a string into ftrace
//===========================================================================
void LwTestWriteFTrace(const char *str)
{
    ssize_t written;

    if(marker_fd < 0)
        return;

    written = write(marker_fd, str, LwOsStrlen(str));
    (void)written; // to keep compiler happy
}

#else /* LWOS_IS_LINUX */

static void LwTestOpenFTrace(void) { }
static void LwTestCloseFTrace(void) { }
void LwTestWriteFTrace(const char *str) { }

#endif /* LWOS_IS_LINUX */

//===========================================================================
// LwTestInitialize() - must called before any other LwTest* functions
//===========================================================================
void LwTestInitialize(
            int                          *argc,
            char                        *argv[],
            LwTestApplication           *pApp)
{
    static LwTestApp app[1] = {{0}};
    LwU32  err;

    (void)err; // prevent warnings
    LW_ASSERT(!gs_app || gs_app == app);

    if (!gs_app) {
        LW_ASSERT(app->state == LwTestAppState_Uninitialized);
        LW_ASSERT(!app->useCount);
        app->state = LwTestAppState_Initialized;
        app->useCount = 1;
        app->failTail = &app->failHead;
        app->testCnt    = 0;
        app->failureCnt = 0;
        app->subtestFilter = NULL;
        app->subtestList = NULL;
        app->rest = LW_FALSE;
        app->listAll = LW_FALSE;
        app->tlsIndex = LwOsTlsAlloc();
        err = LwOsMutexCreate(&app->mutex);
        LW_ASSERT(!err);
        gs_app = app;

        LwTestOpenFTrace();
    } else {
        LW_ASSERT(app->state == LwTestAppState_Initialized);

        LwOsMutexLock(app->mutex);
        LW_ASSERT(app->useCount);
        app->useCount++;
        LwOsMutexUnlock(app->mutex);
    }

    if (pApp)
        *pApp= app;
}

//===========================================================================
// LwTestSetRunFunction() - set the LwRun() function.
//===========================================================================
void LwTestSetRunFunction(LwTestApplication app, void (*func)(LwU32))
{
    app = app?app:gs_app;
    LW_ASSERT(gs_app == app);
    LwOsMutexLock(app->mutex);
    // if called more than once, use the function from the first call.
    app->lwRun = app->lwRun ? app->lwRun : func;
    LwOsMutexUnlock(app->mutex);
}

//===========================================================================
// LwTestSubtestEnd() - called after each test
//===========================================================================
static void LwTestSubtestEnd(LwTestSubtestHandle sub)
{
    char* result = "pass";

    LW_ASSERT(sub->app == gs_app);
    // We could lift the restriction of one active subtest per thread, but that
    // wouldn't work with helper macros in lwtest.h.
    LW_ASSERT(LwTestGetLwrSubtest() == sub);

    if (sub->result) {
        LwOsMutexLock(sub->app->mutex);
        sub->app->failureCnt++;
        LwOsMutexUnlock(sub->app->mutex);
        result = "fail";
    }

    if (sub->skip) {
        LwOsMutexLock(sub->app->mutex);
        sub->app->testCnt--;
        sub->app->filteredCnt++;
        LwOsMutexUnlock(sub->app->mutex);
    } else {
        if (sub->app->rest)
            LwTestPrintf("[REST:, test_case=%s, disposition=%s, duration=%d]\n", sub->name, result, LwOsGetTimeMS() - sub->startTime);
        else if (sub->app->listAll)
            LwTestPrintf("%s\n", sub->name);
        else
            LwTestPrintf("[%s: %s]\n", result, sub->name);
    }

    LwOsFree(sub->name);
    sub->name = NULL;

    LwTestSetLwrSubtest(NULL);
    LwOsFree(sub);
}

//===========================================================================
// LwTestTerminate() - called at end of all tests to report results
//===========================================================================
LwError LwTestTerminate(LwTestApplication app)
{
    LwError err;

    app = app?app:gs_app;
    LW_ASSERT(gs_app == app);

    LwOsMutexLock(app->mutex);

    LW_ASSERT(app->state == LwTestAppState_Initialized ||
                 app->state == LwTestAppState_RanTests);
    LW_ASSERT(app->useCount);

    if (LwTestGetLwrSubtest()) {
        LwTestSubtestEnd(LwTestGetLwrSubtest());
    }

    if (app->failureCnt && !app->err)
        app->err = LwError_TestApplicationFailed;

    err = app->err;
    if (--app->useCount == 0) {

        LwTestCloseFTrace();

        LwOsMutexUnlock(app->mutex);
        LwOsMutexDestroy(app->mutex);

        app->state = LwTestAppState_Terminated;

        if (app->listAll == LW_FALSE) {
            LwTestPrintf( "---------------\n" );
            LwTestPrintf("total subtests: %d\n", app->testCnt);
            LwTestPrintf("total failures: %d\n", app->failureCnt);
            if (app->filteredCnt > 0)
                LwTestPrintf("total skipped:  %d\n", app->filteredCnt);

            LwTestPrintf("\n\n");
        }

        if (app->err &&
            (app->err != LwError_TestApplicationFailed ||
             !app->failureCnt)) {
            LwTestPrintf("LwTestMain() returned error.\n\n");
        }

        if (app->subtestFilter)
            LwOsFree(app->subtestFilter);

        if (app->subtestList)
            LwOsFree(app->subtestList);

        LwOsTlsFree(app->tlsIndex);

        LwOsMemset(app, 0, sizeof(*app));
        gs_app = 0;
    } else {
        LwOsMutexUnlock(app->mutex);
    }
    return err;
}

//===========================================================================
// LwTestError() - indicate that the test should return an error
//===========================================================================
void LwTestError(LwTestApplication app, LwError err)
{
    app = app?app:gs_app;
    LW_ASSERT(gs_app == app);

    LwOsMutexLock(app->mutex);

    LW_ASSERT(app->state == LwTestAppState_Initialized ||
                 app->state == LwTestAppState_RanTests);
    LW_ASSERT(app->useCount);

    if (err) {
        app->err = err;
    } else if (!app->err) {
        app->err = LwError_TestApplicationFailed;
    }

    LwOsMutexUnlock(app->mutex);
}

//===========================================================================
//
//===========================================================================
static void ReadInputFile(LwTestApplication app, char* filename)
{
    LwOsFileHandle file = NULL;
    LwOsStatType stat;
    char* data = NULL;
    char* tmp;
    size_t read;
    LwBool inName = LW_TRUE;
    unsigned int i;

    LwError err = LwOsFopen(filename, LWOS_OPEN_READ, &file);
    if (err)
        goto fail;

    err = LwOsFstat(file, &stat);
    if (err)
        goto fail;

    data = (char*)LwOsAlloc((size_t)stat.size);
    if (!data)
        goto fail;

    err = LwOsFread(file, data, (size_t)stat.size, &read);
    if (err)
        goto fail;

    app->subtestList = (char*)LwOsAlloc((size_t)stat.size);
    if (!app->subtestList)
        goto fail;

    tmp = app->subtestList;
    for (i = 0; i < read; i++)
    {
        if (inName)
            *(tmp++) = data[i];


        if (data[i] == ',' || data[i] == '\n')
            inName = !inName;
    }
    *tmp = '\0';

    LwTestPrintf("Subtest list set to: %s\n", app->subtestList);

fail:
    if (data)
        LwOsFree(data);

    if (file)
        LwOsFclose(file);
}

//===========================================================================
//
//===========================================================================
static LwBool ShouldRunSubtest(LwTestSubtest* sub)
{
    if (sub->app->subtestFilter)
    {
        LwU32 filterLen = LwOsStrlen(sub->app->subtestFilter);
        const char *filter = sub->app->subtestFilter;

        if (!((strncmp(sub->name, filter, filterLen) == 0 &&
              filterLen == LwOsStrlen(sub->name)) ||
              (filter[filterLen-1] == '*' &&
               strncmp(sub->name, filter, filterLen-1) == 0)))
            return LW_FALSE;
    }

    if (sub->app->subtestList)
    {
        char tmpName[MAX_SUBTEST_NAME_LENGTH];

        LwOsSnprintf(tmpName, MAX_SUBTEST_NAME_LENGTH, "%s,", sub->name);

        if (!strstr(sub->app->subtestList, tmpName))
            return LW_FALSE;
    }

    return LW_TRUE;
}

//===========================================================================
// LwTestSubtestBegin() - call to determine whether a subtest should be run
//===========================================================================
LwBool LwTestSubtestBegin(
            LwTestApplication app,
            LwTestSubtestHandle *pSubHandle,
            const char *testNameFormat, ...)
{
    char tmpName[MAX_SUBTEST_NAME_LENGTH];
    va_list valist;
    LwTestSubtest* sub;

    app = app?app:gs_app;
    LW_ASSERT(gs_app == app);

    LwOsMutexLock(app->mutex);

    LW_ASSERT(app->state == LwTestAppState_Initialized ||
                 app->state == LwTestAppState_RanTests);

    sub = (LwTestSubtest*)LwOsAlloc(sizeof(LwTestSubtest));
    if (!sub)
        goto fail;

    LwOsMemset(sub, 0, sizeof(LwTestSubtest));

    if (LwTestGetLwrSubtest()) {
        LwTestSubtestEnd(LwTestGetLwrSubtest());
    }

    // Keep our own copy of subtest name, cause we have no guarantees about what
    // the app does with the string between calling this and LwTestSubtestEnd
    if (testNameFormat)
    {
        LwU32 len;
        LwS32 success;
        va_start(valist, testNameFormat);
        success = LwOsVsnprintf(
            tmpName,
            MAX_SUBTEST_NAME_LENGTH,
            testNameFormat,
            valist);
        va_end(valist);
        if (success > 0)
        {
            len = LwOsStrlen(tmpName);
            sub->name = LwOsAlloc(len + 1);
            if (!sub->name)
                goto fail;
            LwOsStrncpy(sub->name, tmpName, len);
        }
        else
        {
            // LwOsVsnprintf failed
            len = LwOsStrlen(testNameFormat);
            sub->name = LwOsAlloc(len + 1);
            if (!sub->name)
                goto fail;
            LwOsStrncpy(sub->name, testNameFormat, len);
        }
        sub->name[len] = '\0';
    }
    else
    {
        sub->name = LwOsAlloc(1);
        sub->name[0] = '\0';
    }

    sub->result  = LwSuccess;
    sub->skip = LW_FALSE;
    sub->app = app;

    if (!ShouldRunSubtest(sub))
    {
        // The filter does not match, skip test.
        app->filteredCnt++;
        // Free name if filtered out, because SubtestEnd is *not*
        // called for filtered-out tests.
        if (sub->name)
            LwOsFree(sub->name);
        sub->name = NULL;
        LwOsFree(sub);
        LwOsMutexUnlock(app->mutex);
        return LW_FALSE;
    }

    sub->startTime = LwOsGetTimeMS();
    sub->index = app->testCnt++;
    LwTestSetLwrSubtest(sub);

    if (app->listAll)
        return LW_FALSE;  // Just print the test. Don't run it.

    if (pSubHandle)
        *pSubHandle = sub;

    // LwRun() function is for setting a breakpoint before the subtest begins.
    if (app->lwRun)
        app->lwRun(app->testCnt - 1);   // This calls LwRun()

    LwOsMutexUnlock(app->mutex);
    return LW_TRUE;   // yes, do run the test

fail:
    LwTestPrintf("Failed to allocate memory for subtest\n");
    if (sub)
        LwOsFree(sub);
    LwOsMutexUnlock(app->mutex);
    return LW_FALSE;
}

//===========================================================================
// LwTestSubtestSkip() - Call if a subtest should be skipped after starting
//===========================================================================
void LwTestSubtestSkip(
            LwTestApplication app,
            LwTestSubtestHandle sub,
            const char *reason)
{
    app = app ? app : gs_app;
    if (!sub && !LwTestGetLwrSubtest())
        LwTestSubtestBegin(app, 0, "unknown");

    sub = sub ? sub : LwTestGetLwrSubtest();
    LW_ASSERT(sub->app == app);
    LW_ASSERT(LwTestGetLwrSubtest() == sub);

    sub->skip = LW_TRUE;
    LwTestPrintf( "[skip: %s] %s\n",
            sub->name,
            reason?reason:" " );
}

//===========================================================================
// LwTestSubtestFail() - Call if a subtest fails
//===========================================================================
void LwTestSubtestFail(
            LwTestApplication app,
            LwTestSubtestHandle sub,
            const char *reason,
            const char *file,
            int line)
{
    app = app ? app : gs_app;
    if (!sub && !LwTestGetLwrSubtest())
        LwTestSubtestBegin(app, 0, "unknown");

    sub = sub ? sub : LwTestGetLwrSubtest();
    LW_ASSERT(sub->app == app);
    LW_ASSERT(LwTestGetLwrSubtest() == sub);

    sub->result = LwError_TestApplicationFailed;
    LwTestPrintf( "[fail: %s  at %s:%d] %s\n",
            sub->name,
            file,
            line,
            reason?reason:" " );
}

//===========================================================================
// LwTestReadline() - printf to host
//===========================================================================
LwError LwTestReadline(
                const char *prompt,
                char *buffer,
                size_t bufferLength,
                size_t *count)
{
    return LwError_NotSupported;
}

//===========================================================================
// LwTestPrintf() - printf to host
//===========================================================================
void LwTestPrintf(const char *format, ...)
{
    va_list ap;
    va_start( ap, format );
    LwTestVprintf(format, ap);
    va_end( ap );
}

//===========================================================================
// LwTestVprintf() - vprintf to host
//===========================================================================
void LwTestVprintf(const char *format, va_list ap)
{
    LW_ASSERT(gs_app);
#if LWOS_IS_QNX
    //This is required because QNX LwOsDebugVprintf
    //always and only prints to system log.
    vprintf(format, ap);
#else
    LwOsDebugVprintf(format, ap);
#endif
}

//===========================================================================
// LwTestSetSubtestFilter()
//===========================================================================
void LwTestSetSubtestFilter(LwTestApplication app, const char* Filter)
{
    int len = LwOsStrlen(Filter);
    char* copy = LwOsAlloc(len + 1);
    if (!copy)
    {
        LwTestPrintf("Failed to set subtest filter.\n");
        return;
    }
    LwOsStrncpy(copy, Filter, len + 1);

    LwOsMutexLock(app->mutex);

    if (app->subtestFilter)
        LwOsFree(app->subtestFilter);
    app->subtestFilter = copy;

    LwTestPrintf("Test filter set to: '%s'\n", app->subtestFilter);

    LwOsMutexUnlock(app->mutex);
}

//===========================================================================
// LwTestResultsForREST()
//===========================================================================
void LwTestResultsForREST(LwTestApplication app, char* filename)
{
    LW_ASSERT(filename);

    LwOsMutexLock(app->mutex);
    app->rest = LW_TRUE;
    ReadInputFile(app, filename);
    LwOsMutexUnlock(app->mutex);
}


//===========================================================================
// LwTestSetListAll()
//===========================================================================
void LwTestSetListAll(LwTestApplication app, LwBool listAll)
{
    LwOsMutexLock(app->mutex);
    app->listAll = listAll;
    LwOsMutexUnlock(app->mutex);
}
