#pragma once

#include <ctime>
#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>
#include <cstdarg>
#include <assert.h>
#include <windows.h>

enum class LogSeverity
{
    kDebug = -2,                //Messages you only need if you hunt down bugs - very detailed
    kVerbose = -1,              //Messages with extra details which may give clues if things go funny 
    kInfo = 0,                  //Basic information we would like to collect normally for basic understanding of the state
    kWarning = 1,               //Bad things that happen in code, but we think we can handle them
    kError = 2,                 //Things go very wrong, Ansel doesn't function as intended
    kFatal = 3,                 //Things that force us to shut down
    kDisableLogging = 4,        
    
    kIlwalid,
    kFirst = kDebug,
    kLast = kIlwalid - 1
};

//Add your channels here
//Children should go immediately AFTER parents
enum class LogChannel
{
    kInput = 0,
    kInput_RawInput,
    kInput_RawInput_Mouse,
    kInput_RawInput_Kbd,
    kInput_RawInput_Gamepad,
    kInput_Hooks,
    kYamlParser,
    kAnselSdk,
    kRenderBuffer,
    kRenderBuffer_Colwerter,
    kDebugProfiling,

    kNumElems,
    kFirst = kInput,
    kLast = kNumElems - 1
};

struct LogChannelDesc
{
    LogChannel first;       //first subchannel, including the channel itself
    LogChannel last;        //last subchannel
    const wchar_t* name;    //channels' name 
};

//One entry per LogChannel
const LogChannelDesc gLogChannelsDeclarartion[] = 
{
    { LogChannel::kInput, LogChannel::kInput_Hooks, L"input" },
    { LogChannel::kInput_RawInput, LogChannel::kInput_RawInput_Gamepad, L"input.rawinput" },
    { LogChannel::kInput_RawInput_Mouse, LogChannel::kInput_RawInput_Mouse, L"input.rawinput.mouse" },
    { LogChannel::kInput_RawInput_Kbd, LogChannel::kInput_RawInput_Kbd, L"input.rawinput.kbd" },
    { LogChannel::kInput_RawInput_Gamepad, LogChannel::kInput_RawInput_Gamepad, L"input.rawinput.gamepad" },
    { LogChannel::kInput_Hooks, LogChannel::kInput_Hooks, L"input.hooks" },
    { LogChannel::kYamlParser, LogChannel::kYamlParser, L"yamlparser" },
    { LogChannel::kAnselSdk, LogChannel::kAnselSdk, L"anselsdk"},
    { LogChannel::kRenderBuffer, LogChannel::kRenderBuffer_Colwerter, L"renderbuffer"},
    { LogChannel::kRenderBuffer_Colwerter, LogChannel::kRenderBuffer_Colwerter, L"renderbuffer.colwerter"},
    { LogChannel::kDebugProfiling, LogChannel::kDebugProfiling, L"debugprofiling"}
};

static_assert((sizeof(gLogChannelsDeclarartion) / sizeof(LogChannelDesc)) == (int)LogChannel::kNumElems, "gLogChannelsDeclarartion isn't full!");

LogSeverity getLogSeverity(); //for logging without channels
LogSeverity getLogSeverity(LogChannel chan);

const std::wstring & getLogFilename();
void setLogFilename(const wchar_t * logFilename);

void initLog();
void writeLog(const char* text, LogSeverity severity, const char* filename, int linenumber);
void deinitLog();
bool isLogInitialized();

extern LogSeverity gLogSeverityLevel;

class LogSeverityPerChannel
{
public:
    LogSeverityPerChannel()
    {
        initToGlobal();
    }

    LogSeverity getLogSeverity(LogChannel chan) const
    {
        assert(chan < LogChannel::kNumElems);
        return m_logSeverityLevelPerChannel[(int)chan];
    }
        
    void init(LogSeverity forNoChannel, LogSeverity forTopChannels, LogSeverity setExplcitly[(int)LogChannel::kNumElems],
        LogSeverity forChildren[(int)LogChannel::kNumElems]);

    bool init(const wchar_t* configString);

    void initToGlobal()
    {
        for (int i = 0; i < (int)LogChannel::kNumElems; ++i)
        {
            m_logSeverityLevelPerChannel[i] = ::getLogSeverity();
        }
    }
    
private:
    
    void setLogSeverity(LogChannel chan, LogSeverity logSeverityLevel)
    {
        assert(chan < LogChannel::kNumElems);
        m_logSeverityLevelPerChannel[(int)chan] = logSeverityLevel;
    }

    LogSeverity m_logSeverityLevelPerChannel[(int)LogChannel::kNumElems];
};

extern LogSeverityPerChannel  gLogSeverityLevelPerChannel;

bool setLogSeverity(LogSeverity logSeverityLevel, const wchar_t* str = L"");

extern CRITICAL_SECTION gLogCriticalSection;

inline const char* severityToString(LogSeverity severity)
{
    if (LogSeverity::kVerbose == severity)
        return " [V]";
    else if (LogSeverity::kDebug == severity)
        return " [DBG]";
    else if (LogSeverity::kWarning == severity)
        return " [WARN]";
    else if (LogSeverity::kError == severity)
        return " [ERROR]";
    else if (LogSeverity::kFatal == severity)
        return " [FATAL]";
    return "";
}

inline const char* getLogSeverityName(LogSeverity ls)
{
    switch (ls)
    {
        case LogSeverity::kDebug: return "Debug";
        case LogSeverity::kVerbose: return "Verbose";
        case LogSeverity::kInfo: return "Info";
        case LogSeverity::kWarning: return "Warning";
        case LogSeverity::kError: return "Error";
        case LogSeverity::kFatal: return "Fatal";
        case LogSeverity::kDisableLogging: return "Disable Logging";
        default: return "Invalid Log Severity";
    }
}

void logHelperCommon(LogSeverity severity, const char* filename, int linenumber, const char* format, va_list args);
void logHelper(LogSeverity severity, const char* filename, int linenumber, const char* format, ...);
void logHelper(LogSeverity severity, const char* filename, int linenumber, LogChannel chan, const char* format, ...);

//those helpers allow us to choose the right severity checking tsrategy based on whether the  channel was supplied into the macro
//we want the check outside of the macro so that we could not evaluate log() function parameters if we don't need logging

template<typename T>
static inline bool checkSeverityHelper(LogSeverity s, T)
{
    return gLogSeverityLevel <= s;
}

template<>
static inline bool checkSeverityHelper<LogChannel>(LogSeverity s, LogChannel chan)
{
    return gLogSeverityLevelPerChannel.getLogSeverity(chan) <= s;
}

#ifndef LOG_INFO

//this fun is required to WAR a bug in MSVC that treats VA_ARGS as a single parameter
//the whole thing in general is required to plug in the first argument compile time, and avoid evaluating the rest of them
#define GET_FIRST_PARAM(param, ...) param 
#define GET_FIRST_PARAM_HELPER(tuple) GET_FIRST_PARAM tuple 

#define LOG_DEBUG(...)      do { if (checkSeverityHelper(LogSeverity::kDebug, GET_FIRST_PARAM_HELPER((__VA_ARGS__))))       logHelper(LogSeverity::kDebug, __FILE__, __LINE__, __VA_ARGS__); } while (0)
#define LOG_VERBOSE(...)    do { if (checkSeverityHelper(LogSeverity::kVerbose, GET_FIRST_PARAM_HELPER((__VA_ARGS__))))     logHelper(LogSeverity::kVerbose, __FILE__, __LINE__, __VA_ARGS__); } while (0)
#define LOG_INFO(...)       do { if (checkSeverityHelper(LogSeverity::kInfo, GET_FIRST_PARAM_HELPER((__VA_ARGS__))))        logHelper(LogSeverity::kInfo, __FILE__, __LINE__, __VA_ARGS__); } while (0)
#define LOG_WARN(...)       do { if (checkSeverityHelper(LogSeverity::kWarning, GET_FIRST_PARAM_HELPER((__VA_ARGS__)))) logHelper(LogSeverity::kWarning, __FILE__, __LINE__, __VA_ARGS__); } while (0)
#define LOG_ERROR(...)  do { if (checkSeverityHelper(LogSeverity::kError, GET_FIRST_PARAM_HELPER((__VA_ARGS__))))   logHelper(LogSeverity::kError, __FILE__, __LINE__, __VA_ARGS__); } while (0)
#define LOG_FATAL(...)  do { if (checkSeverityHelper(LogSeverity::kFatal, GET_FIRST_PARAM_HELPER((__VA_ARGS__))))   logHelper(LogSeverity::kFatal, __FILE__, __LINE__, __VA_ARGS__); } while (0)


#endif // LOG_INFO
