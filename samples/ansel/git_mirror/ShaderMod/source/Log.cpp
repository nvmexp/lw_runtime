#include <string>
#include <iomanip>
#include <thread>
#include <assert.h>

#include "darkroom/StringColwersion.h"

#include "Log.h"


#define WIDECHAR_LOG    0

FILE * gLogFileBuf = nullptr;
CRITICAL_SECTION gLogCriticalSection;
int gLogCriticalSectionRefCount = 0;

LogSeverity gLogSeverityLevel = LogSeverity::kDisableLogging;
LogSeverityPerChannel  gLogSeverityLevelPerChannel;

std::wstring gLogFilename;

LogSeverity getLogSeverity() { return gLogSeverityLevel; }

bool setLogSeverity(LogSeverity logSeverityLevel, const wchar_t* str)
{
    gLogSeverityLevel = logSeverityLevel;
    return gLogSeverityLevelPerChannel.init(str);
}

LogSeverity getLogSeverity(LogChannel chan)
{
    return gLogSeverityLevelPerChannel.getLogSeverity(chan);
}

bool isLogInitialized() { return gLogCriticalSectionRefCount > 0; }

const std::wstring & getLogFilename() { return gLogFilename; }
void setLogFilename(const wchar_t * logFilename)
{
    // Ignore multiple calls with same log file name
    if (gLogFilename == logFilename)
        return;

    if (gLogFileBuf)
    {
        fclose(gLogFileBuf);
        gLogFileBuf = nullptr;
    }

    gLogFilename = logFilename;
}

void writeLog(const char* text, LogSeverity severity, const char* filename, int linenumber)
{
    if (gLogFilename.size() == 0)
        return;

    if (!gLogFileBuf)
    {
        gLogFileBuf = _wfsopen(gLogFilename.c_str(), L"wt", _SH_DENYWR);
    }

    if (gLogFileBuf)
    {
#if (WIDECHAR_LOG == 1)
        std::wstring utf16 = darkroom::getWstrFromUtf8(text);
        fwprintf(gLogFileBuf, L"%s", utf16.c_str());
#else
        fprintf(gLogFileBuf, "%s", text);
#endif
        fflush(gLogFileBuf);
    }
}

void initLog()
{
    if (gLogCriticalSectionRefCount == 0)
        InitializeCriticalSection(&gLogCriticalSection);
    ++gLogCriticalSectionRefCount;
}
void deinitLog()
{
    --gLogCriticalSectionRefCount;
    if (gLogCriticalSectionRefCount == 0)
    {
        DeleteCriticalSection(&gLogCriticalSection);
    }

    if (gLogFileBuf)
        fclose(gLogFileBuf);
    
    gLogFileBuf = nullptr;
}

void LogSeverityPerChannel::init(LogSeverity forNoChannel, LogSeverity forTopChannels, LogSeverity setExplcitly[(int)LogChannel::kNumElems],
    LogSeverity forChildren[(int)LogChannel::kNumElems])
{
    if (forTopChannels == LogSeverity::kIlwalid)
        forTopChannels = forNoChannel;

    for (int i = 0; i < (int)LogChannel::kNumElems; ++i)
    {
        m_logSeverityLevelPerChannel[i] = forTopChannels;
    }

    for (int i = 0; i < (int)LogChannel::kNumElems; ++i)
    {
        if (setExplcitly[i] != LogSeverity::kIlwalid)
        {
            m_logSeverityLevelPerChannel[i] = setExplcitly[i];
        }

        LOG_VERBOSE("Setting log severity for channel %s to %d", darkroom::getUtf8FromWstr(gLogChannelsDeclarartion[i].name).c_str(), m_logSeverityLevelPerChannel[i]);
        
        LogSeverity severityOfChildren = m_logSeverityLevelPerChannel[i]; //inherited or set explicitly

        if (forChildren[i] != LogSeverity::kIlwalid)
        {
            severityOfChildren = forChildren[i];
        }

        //inherit severity
        for (int from = (int)gLogChannelsDeclarartion[i].first + 1, to = (int)gLogChannelsDeclarartion[i].last; from <= to; ++from)
        {
            m_logSeverityLevelPerChannel[from] = severityOfChildren;
        }
    }
}

bool LogSeverityPerChannel::init(const wchar_t* cfg)
{
    const wchar_t* configString = cfg;

    LogSeverity forTopChannels = LogSeverity::kIlwalid;

    LogSeverity logSeverityExplicit[(int)LogChannel::kNumElems];
    LogSeverity logSeverityForChildren[(int)LogChannel::kNumElems];

    for (int i = 0; i < (int)LogChannel::kNumElems; ++i)
    {
        logSeverityExplicit[i] = LogSeverity::kIlwalid;
        logSeverityForChildren[i] = LogSeverity::kIlwalid;
    }

    const wchar_t* lhsBegin = configString, *lhsEnd = nullptr, *rhsBegin = nullptr, *rhsEnd = nullptr;

    bool parsingSucceeded = true;

    while (*configString != L'\0')
    {
        while (*configString == L' ') //ignore whitespaces
            ++configString;

        if (*configString == L'\0') //empty
            break;

        const wchar_t* lhsBegin = configString;

        while (*configString != L' ' && *configString != L'=' && *configString != L'\0') //look for the rhs
            ++configString;

        if (*configString == L'\0') // no rhs
        {
            LOG_ERROR("Parsing LogChannelsFiltering key failed: can't find the end of the left-hand side at %d", lhsBegin - cfg);

            parsingSucceeded = false;
            break;
        }

        const wchar_t* lhsEnd = configString;

        while (*configString == L' ') //ignore whitespaces
            ++configString;

        if (*configString != L'=') //no rhs
        {
            LOG_ERROR("Parsing LogChannelsFiltering key failed: can't find \'=\' at %d", lhsEnd - cfg);

            parsingSucceeded = false;
            break;
        }

        ++configString;

        while (*configString == L' ') //ignore whitespaces
            ++configString;

        if (*configString == L'\0') //empty rhs
        {
            LOG_ERROR("Parsing LogChannelsFiltering key failed: nothing to parse after \'=\' at %d", lhsEnd - cfg);

            parsingSucceeded = false;
            break;
        }

        const wchar_t* rhsBegin = configString;

        while (*configString != L' ' && *configString != L';' && *configString != L'\0') //look for the end of rhs
            ++configString;

        const wchar_t* rhsEnd = configString;

        while (*configString == L' ') //ignore whitespaces
            ++configString;

        if (*configString != L';' && *configString != L'\0') //rhs ends badly
        {
            LOG_ERROR("Parsing LogChannelsFiltering key failed: illegal characters (semicolon missing?) after the right-hand side at %d", rhsEnd - cfg);

            parsingSucceeded = false;
            break;
        }

        if (*configString == L';')
            ++configString;

        if (lhsEnd == lhsBegin) //empty lhs
        {
            LOG_ERROR("Parsing LogChannelsFiltering key failed: empty left-hand side at %d", lhsBegin - cfg);

            parsingSucceeded = false;
            break;
        }

        bool isAllChannelsMarker = false;
        bool isAllTopChannelsMarker = false;

        if (*(lhsEnd - 1) == L'*')
        {
            isAllChannelsMarker = true;
            --lhsEnd;

            if (lhsEnd == lhsBegin)
            {
                isAllTopChannelsMarker = true;
            }
            else
            {
                if (*(lhsEnd - 1) != L'.' || lhsEnd - lhsBegin < 2) //can't be true
                {
                    LOG_ERROR("Parsing LogChannelsFiltering key failed: illegal usage of \".*\" at %d (use * for all top channels or .* for all children of a channel)", lhsBegin - cfg);

                    parsingSucceeded = false;
                    break;
                }

                --lhsEnd;
            }
        }

        LogChannel chan = LogChannel::kNumElems;

        if (!isAllTopChannelsMarker)
        {
            for (int chName = 0; chName < (int)LogChannel::kNumElems; ++chName)
            {
                if (wcsncmp(gLogChannelsDeclarartion[chName].name, lhsBegin, lhsEnd - lhsBegin) == 0)
                {
                    chan = (LogChannel)chName;
                    break;
                }
            }

            if (chan == LogChannel::kNumElems)
            {
                LOG_ERROR("Parsing LogChannelsFiltering key failed: channel name not recognized at %d", lhsBegin - cfg);

                parsingSucceeded = false;
                break;
            }
        }

        wchar_t* end = nullptr;
        long temp = wcstol(rhsBegin, &end, 10);

        if (end == rhsBegin || end == nullptr)
        {
            LOG_ERROR("Parsing LogChannelsFiltering key failed: severity level parsing failed at %d (not a valid integer)", rhsBegin - cfg);

            parsingSucceeded = false;
            break;
        }

        if (end != rhsEnd)
        {
            LOG_ERROR("Parsing LogChannelsFiltering key failed: severity level followed by unrecognized characters at at %d", end - cfg);

            parsingSucceeded = false;
            break;
        }

        if (temp < (int)LogSeverity::kFirst || temp >(int)LogSeverity::kLast)
        {
            LOG_ERROR("Parsing LogChannelsFiltering key failed: severity level out of bounds at %d", rhsBegin - cfg);

            parsingSucceeded = false;
            break;
        }


        if (isAllTopChannelsMarker)
        {
            if (forTopChannels != LogSeverity::kIlwalid) //duplicate
            {
                LOG_ERROR("Parsing LogChannelsFiltering key failed: duplicate setting for top channels found at %d", lhsBegin - cfg);

                parsingSucceeded = false;
                break;
            }

            forTopChannels = (LogSeverity)temp;
        }
        else
        {
            if (isAllChannelsMarker)
            {
                if (logSeverityForChildren[(int)chan] != LogSeverity::kIlwalid) //duplicate
                {
                    LOG_ERROR("Parsing LogChannelsFiltering key failed: duplicate setting for a channel's children found at %d", lhsBegin - cfg);

                    parsingSucceeded = false;
                    break;
                }

                logSeverityForChildren[(int)chan] = (LogSeverity)temp;
            }
            else
            {
                if (logSeverityExplicit[(int)chan] != LogSeverity::kIlwalid) //duplicate
                {
                    LOG_ERROR("Parsing LogChannelsFiltering key failed: duplicate setting for a channel found at %d", lhsBegin - cfg);

                    parsingSucceeded = false;
                    break;
                }

                logSeverityExplicit[(int)chan] = (LogSeverity)temp;
            }
        }
    }

    if (!parsingSucceeded)
    {
        initToGlobal();

        return false;
    }

    init(::getLogSeverity(), forTopChannels, logSeverityExplicit, logSeverityForChildren);

    return true;
}

void logHelperCommon(LogSeverity severity, const char* filename, int linenumber, const char* format, va_list args)
{
    using namespace std::chrono;

    EnterCriticalSection(&gLogCriticalSection);

    char timeBuf[64];

    struct tm timeInfo;
    const std::time_t result = std::time(nullptr);
    localtime_s(&timeInfo, &result);
    const size_t timeBufSize = strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", &timeInfo);

    const std::chrono::milliseconds ms = duration_cast<std::chrono::milliseconds>(system_clock::now().time_since_epoch());

    std::string buffer;

    {
        std::stringstream ss;
        ss << timeBuf << "." << std::setfill('0') << std::setw(3) << ms.count() % 1000 << std::setfill(' ') << ": [" << std::this_thread::get_id() << "] " << severityToString(severity) << " ";
        buffer = ss.str();
    }

    int numChars = _vscprintf(format, args);

    if (numChars >= 0) //-1 is returnd if parsing failed
    {
        std::vector<char> formattedText; //we use a vector because c functions add zero character at the end of the string
        formattedText.resize(numChars + 1);
        vsnprintf_s(formattedText.data(), formattedText.size(), _TRUNCATE, format, args);

        buffer += formattedText.data();
    }
    else
    {
        buffer += "<FORMATTING LOG MESSAGE FAILED>";
    }

    buffer += "\n";

    writeLog(buffer.c_str(), severity, filename, linenumber);

    LeaveCriticalSection(&gLogCriticalSection);
}

void logHelper(LogSeverity severity, const char* filename, int linenumber, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    logHelperCommon(severity, filename, linenumber, format, args);
    va_end(args);
}

void logHelper(LogSeverity severity, const char* filename, int linenumber, LogChannel chan, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    logHelperCommon(severity, filename, linenumber, format, args);
    va_end(args);
}
