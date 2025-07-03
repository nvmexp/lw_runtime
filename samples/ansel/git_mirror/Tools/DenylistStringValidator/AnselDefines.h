#pragma once

#include <iostream>

typedef unsigned int UINT;

//#define SUPPRESS_LOGS

#ifdef SUPPRESS_LOGS
#define LOGV(prefix, ...)
#else
#define LOGV(prefix, ...) \
    std::cout << prefix; printf(__VA_ARGS__); std::cout << '\n';
#endif

#define LOG_DEBUG(...)		LOGV("Debug: ", __VA_ARGS__)
#define LOG_VERBOSE(...)	LOGV("Verbose: ", __VA_ARGS__)
#define LOG_INFO(...)		LOGV("Info: ", __VA_ARGS__)
#define LOG_WARN(...)		LOGV("Warn: ", __VA_ARGS__)
#define LOG_ERROR(...)		LOGV("Error: ", __VA_ARGS__)
#define LOG_FATAL(...)		LOGV("Fatal: ", __VA_ARGS__)