#include <algorithm>
#include <memory>
#include <varargs.h>
#include "ErrorReporting.h"

void ErrorManager::init(int maxNumEntries)
{
    m_errorEntryIdx = 0;
    m_errorEntries = decltype(m_errorEntries)(maxNumEntries, { 0.0f, -1.0f, L"" });
}

void ErrorManager::diminishLifeTime(double dtSeconds)
{
    for (size_t errCnt = 0, errCntEnd = m_errorEntries.size(); errCnt < errCntEnd; ++errCnt)
    {
        if (m_errorEntries[errCnt].lifeTime > 0.0f)
        {
            m_errorEntries[errCnt].lifeTime -= float(dtSeconds);
            m_errorEntries[errCnt].elapsedTime += float(dtSeconds);
        }
    }
}

float ErrorManager::getErrorLifetime(size_t entryIndex) const
{
    return m_errorEntries[entryIndex].lifeTime;
}

float ErrorManager::getErrorElapsedTime(size_t entryIndex) const
{
    return m_errorEntries[entryIndex].elapsedTime;
}

const std::wstring& ErrorManager::getErrorString(size_t entryIndex) const
{
    return m_errorEntries[entryIndex].message;
}

size_t ErrorManager::getErrorCount() const
{
    return m_errorEntries.size();
}

size_t ErrorManager::getFirstErrorIndex() const
{
    return m_errorEntryIdx;
}

bool ErrorManager::isEmpty() const
{
    for (size_t errCnt = 0, errCntEnd = m_errorEntries.size(); errCnt < errCntEnd; ++errCnt)
    {
        if (m_errorEntries[errCnt].lifeTime > 0.0f)
            return false;
    }
    return true;
}
