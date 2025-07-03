#pragma once

#include <vector>
#include <string>


#pragma warning( push )  
#pragma warning(disable : 4996) // _snwprintf is unsafe
/*
    Implements a simple ring buffer of error messages
*/
class ErrorManager
{
public:
    struct ErrorEntry
    {
        float elapsedTime;
        float lifeTime;
        std::wstring message;
    };

    void init(int maxNumEntries);
    template<typename... Args>
    void addError(float lifeTime, const std::wstring& format, Args... args)
    {
        size_t size = _snwprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
        std::unique_ptr<wchar_t[]> buf(new wchar_t[size]);
        _snwprintf(buf.get(), size, format.c_str(), args...);
        const auto errorMsg = std::wstring(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside

        int newErrorEntryIdx = m_errorEntryIdx - 1;
        if (newErrorEntryIdx < 0)
            newErrorEntryIdx = (int)m_errorEntries.size() - 1;

        m_errorEntries[newErrorEntryIdx] = { 0.0f, lifeTime, errorMsg };
        m_errorEntryIdx = newErrorEntryIdx;
    }
    void diminishLifeTime(double dtSeconds);

    // Get ring buffer size
    size_t getErrorCount() const;
    // Get current position in the ring buffer
    size_t getFirstErrorIndex() const;

    // If there is any error that is active
    bool ErrorManager::isEmpty() const;

    const std::wstring& getErrorString(size_t entryIndex) const;
    float getErrorLifetime(size_t entryIndex) const;
    float getErrorElapsedTime(size_t entryIndex) const;
private:
    std::vector<ErrorEntry> m_errorEntries;
    int m_errorEntryIdx;
};

#pragma warning( pop )  
