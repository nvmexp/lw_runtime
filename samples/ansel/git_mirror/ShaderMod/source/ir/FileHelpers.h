#pragma once

#include <assert.h>

typedef void* HANDLE;

namespace shadermod
{
namespace ir
{
namespace filehelpers
{
    void getLwrrentPath(wchar_t * path, size_t maxSize);

    bool getFileTime(const wchar_t* filepath, uint64_t& dateTime);
    bool directoryExists(const wchar_t *path);
    bool validateFileExists(const std::wstring& absolutePath);
    bool createDirectoryRelwrsively(const wchar_t *path);
    std::wstring GetFileName(const std::wstring& fullFilePath);
    bool SplitPathIntoDirectoryAndFileName(const std::wstring& fullFilePath, std::wstring& directory_out, std::wstring& fileName_out);

    class MappedFile
    {
    public:

        const unsigned char* begin() const
        {
            return m_mapAddr;
        }

        const unsigned char* end() const
        {
            return m_endMapaddr;
        }

        bool isValid() const
        {
            assert(m_endMapaddr >= m_mapAddr);

            return m_mapAddr != nullptr;
        }

        bool open(const wchar_t* filename);
        void close();
        
        MappedFile(); 
        MappedFile(const wchar_t* filename);
        ~MappedFile();

    private:
        HANDLE m_fileHandle;
        HANDLE m_mappingHandle;
        unsigned char* m_mapAddr;
        unsigned char* m_endMapaddr;
    };

}
}
}
