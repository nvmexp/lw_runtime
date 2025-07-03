#pragma once

#include <syslog.h>
#include "logging.h"

#define SYSLOG_CRITICAL(fmt, ...)                            \
    PRINT_CRITICAL(fmt, fmt, ##__VA_ARGS__);                 \
    syslog(LOG_CRIT, fmt, ##__VA_ARGS__);

#define SYSLOG_ERROR(fmt, ...)                               \
    PRINT_ERROR(fmt, fmt,  ##__VA_ARGS__);                   \
    syslog(LOG_ERR, fmt, ##__VA_ARGS__);

#define SYSLOG_WARNING(fmt, ...)                             \
    PRINT_WARNING(fmt, fmt, ##__VA_ARGS__);                  \
    syslog(LOG_WARNING, fmt, ##__VA_ARGS__);

#define SYSLOG_NOTICE(fmt, ...)                              \
    PRINT_INFO(fmt, fmt, ##__VA_ARGS__);                     \
    syslog(LOG_NOTICE, fmt, ##__VA_ARGS__);

#define SYSLOG_INFO(fmt, ...)                                \
    PRINT_INFO(fmt, fmt, ##__VA_ARGS__);                     \
    syslog(LOG_INFO, fmt, ##__VA_ARGS__);

#define DCGM_MAX_LOG_ROTATE  5

void dcgmLoggingInit(char *elwDebugLevel, char *elwDebugAppend,
                     char *elwDebugFile, char *elwDebugFileRotate);
