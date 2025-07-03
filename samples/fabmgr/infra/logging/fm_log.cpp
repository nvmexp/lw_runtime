/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include "fm_log.h"

#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
#include "modsdrv.h"
#endif

using namespace std;

// these variables are used in header file using extern
FMLogLevel_t fmLogLevel = FM_LOG_LEVEL_DISABLED;

// local global variables
static FILE* fmLogFd = NULL;
static FMLogMod_t fmLogMode = FM_LOG_MODE_FILE;
static volatile unsigned int initLock = 0;
static bool fmLogInitialized = false;
static bool maxLogFileSizeReached = false;
static unsigned int maxLogFileSize = 1024; // in MB
static const int BYTES_IN_MB = 1024 * 1024;

#if defined(_WINDOWS)
HANDLE eventHandle;
static const int MAX_SYS_LOG_BUF_SIZE = 2048;
#endif

static void
fmLogPrintHeader(void)
{
    // Log date and time
    lwosLocalTime localTime = {};
    lwosGetLocalTime(&localTime);
    fmLogPrintf("Fabric Manager Log initializing at: %d/%d/%d %02d:%02d:%02d.%03d\n",
                localTime.month, localTime.dayOfMonth, localTime.year,
                localTime.hour, localTime.min, localTime.sec, localTime.msec);
}

static void
fmLogOpenLogFile(char* logFileName, bool appendToLog)
{
    fmLogFd = fopen(logFileName, appendToLog ? "a" : "w");
    if (NULL == fmLogFd) {
        fprintf(stderr, "WARNING: failed to open fabric manager log file %s errno = %s\n",
                logFileName, strerror(errno));
        fprintf(stderr, "INFO: using stderr for fabric manager logging\n");
        fmLogFd = stderr;
    }

    if (fmLogFd && stderr != fmLogFd) {
        // timeout in ms
        lwosLockFile(fmLogFd, 10);
    }
}

static int
getLogPriority()
{
    int logPriority = 0;
#ifdef __linux__
    // colwert our log level to syslog level
    switch (fmLogLevel) {
        case FM_LOG_LEVEL_CRITICAL:
            logPriority = LOG_CRIT;
            break;
        case FM_LOG_LEVEL_ERROR:
            logPriority = LOG_ERR;
            break;
        case FM_LOG_LEVEL_WARNING:
            logPriority = LOG_WARNING;
            break;
        case FM_LOG_LEVEL_INFO:
            logPriority = LOG_INFO;
            break;
        case FM_LOG_LEVEL_DEBUG:
            logPriority = LOG_DEBUG;
            break;
        default:
            logPriority = LOG_DEBUG;
            break;
    }

#else
    // colwert our log level to syslog level
    switch (fmLogLevel) {
        case FM_LOG_LEVEL_CRITICAL:
            logPriority = EVENTLOG_WARNING_TYPE;
            break;
        case FM_LOG_LEVEL_ERROR:
            logPriority = EVENTLOG_ERROR_TYPE;
            break;
        case FM_LOG_LEVEL_WARNING:
            logPriority = EVENTLOG_WARNING_TYPE;
            break;
        case FM_LOG_LEVEL_INFO:
            logPriority = EVENTLOG_INFORMATION_TYPE;
            break;
        default: {
            logPriority = EVENTLOG_INFORMATION_TYPE;
        }
    }
#endif

    return logPriority;
}

static int
fmLogDoSysLog(const char *fmt, va_list args)
{
    int logPriority = getLogPriority();
#ifdef __linux__
    vsyslog(logPriority, fmt, args);
#else
    LPCSTR  lpszStrings;
    char sysLogMsgBuf[MAX_SYS_LOG_BUF_SIZE] = {0};
    vsnprintf(sysLogMsgBuf, MAX_SYS_LOG_BUF_SIZE, fmt, args);
    lpszStrings = sysLogMsgBuf;
    int ret = ReportEvent(eventHandle, logPriority, 0, 0, NULL, 1, 0, &lpszStrings, NULL);
#endif
    return 0;
}

static void
fmLogCheckLogFileSizeLimit(void)
{
    if (fmLogFd && stderr != fmLogFd) {
        long logFileLen = ftell(fmLogFd);
        // colwert bytes to MB
        unsigned int fileSize = logFileLen/BYTES_IN_MB;
        if (fileSize >= maxLogFileSize) {
            // use direct write instead of fmLogPrintf() as it will cause relwrsion.
            fprintf(fmLogFd, "fabric manager log file reached configured size limit, skipping further logs\n");
            fflush(fmLogFd);
            maxLogFileSizeReached = true;
        }
    }
}

static int
fmLogDoFileLog(const char *fmt, va_list args)
{
    if (NULL == fmLogFd) {
        return 1;
    }

    // skip logging if we reached the limits
    if (true == maxLogFileSizeReached) {
        return 0;
    }

    vfprintf(fmLogFd, fmt, args);
    fflush(fmLogFd);

    // check whether the log file reached its size limit and stop logging
    fmLogCheckLogFileSizeLimit();

    return 0;
}

std::string
fmLogGetDateTimeStamp(void)
{
    time_t rawTime;
    struct tm *pTimeInfo;
    const unsigned int dateTimeLength = 60;
    char dateTimeString[dateTimeLength] = { 0 };

    // get current calendar date and time and then colwert it to tm format
    time(&rawTime);
    pTimeInfo = localtime(&rawTime);

    // format it to Month Day Year Hour Minute Second format
    strftime(dateTimeString, dateTimeLength, "%b %d %Y %H:%M:%S", pTimeInfo);

    // colwert to std string for return instead of heap allocation or function arguments
    std::string tempStr(dateTimeString);
    return tempStr;
}

int
fmLogPrintf(const char *fmt, ...)
{
    va_list args;
    int retVal;

    // skip logging if not initialized or the level is disabled.
    if ((false == fmLogInitialized) || (FM_LOG_LEVEL_DISABLED == fmLogLevel)) {
        return 0;
    }

    va_start(args, fmt);

    if (FM_LOG_MODE_SYSLOG == fmLogMode) {
        retVal = fmLogDoSysLog(fmt, args);
    } else {
        retVal = fmLogDoFileLog(fmt, args);
    }

    va_end(args);

    return retVal;
}

#if defined(_WINDOWS)
void logWindowsEvent(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    fmLogDoSysLog(fmt, args);
    va_end(args);
}
#endif

void
fabricManagerInitLog(unsigned int logLevel, char* logFileName,
                     bool appendToLog, unsigned int logFileSize, bool useSysLog)
{
    // logfile name should be valid if syslog option is not used.
    if ((false == useSysLog) && (NULL == logFileName)) {
        fprintf(stderr, "fabric manager log initialization requested with null parameters\n");
    }
#if defined(_WINDOWS)
    eventHandle = RegisterEventSource(NULL, "Fabric Manager");
#endif

    lwmlSpinLock(&initLock);

    if (true == fmLogInitialized) {
        lwmlUnlock(&initLock);
        return; // initialize only once
    }

    fmLogLevel = (FMLogLevel_t)logLevel;
    maxLogFileSize = logFileSize;
    if (true == useSysLog) {
        fmLogMode = FM_LOG_MODE_SYSLOG;
    }

    // nothing specific to setup for syslog
    // for file Log, we need to do some setup like, open the file, lock it etc
    if (logFileName != NULL && FM_LOG_MODE_FILE == fmLogMode) {
        fmLogOpenLogFile(logFileName, appendToLog);
    }

    // set/initialize state across both mods.
    fmLogInitialized = true;
    fmLogPrintHeader();

    lwmlUnlock(&initLock);
}

void
fabricManagerShutdownLog(void)
{
    lwmlSpinLock(&initLock);

    if (false == fmLogInitialized) {
        // logging is not initialized, simply return
        lwmlUnlock(&initLock);
        return;
    }

    if (fmLogFd && stderr != fmLogFd) {
        lwosUnlockFile(fmLogFd);
        fclose(fmLogFd);
    }

#if defined(_WINDOWS)
    bool ret = DeregisterEventSource(eventHandle);
#endif
    fmLogInitialized = false;
    fmLogFd = NULL;
    fmLogLevel = FM_LOG_LEVEL_DISABLED;
    lwmlUnlock(&initLock);
}

#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
int
fmLogModsPrintf(int lvl, const char *fmt, ...)
{
    va_list args;
    int sysLogPriority;

    // colwert our log level to syslog level
    switch (lvl) {
        case FM_LOG_LEVEL_CRITICAL:
        case FM_LOG_LEVEL_ERROR:
            sysLogPriority = PRI_HIGH;
            break;
        case FM_LOG_LEVEL_WARNING:
        case FM_LOG_LEVEL_INFO:
        case FM_LOG_LEVEL_DEBUG:
            sysLogPriority = PRI_DEBUG;
            break;
        default:
            sysLogPriority = PRI_DEBUG;
            break;
    }

    va_start(args, fmt);

    ModsDrvVPrintf(sysLogPriority, fmt, args);

    va_end(args);

    return 0;
}
#endif
