#include <windows.h>

#include <Shlwapi.h>
#include "Log.h"
#include "ir/FileHelpers.h"

namespace shadermod
{
namespace ir
{
namespace filehelpers
{
    bool getFileTime(const wchar_t* filepath, uint64_t& dateTime)
    {
        HANDLE hFile = CreateFile(
            filepath,
            GENERIC_READ,
            0,
            NULL,
            OPEN_EXISTING,
            0,
            NULL
        );

        if (hFile != ILWALID_HANDLE_VALUE)
        {
            FILETIME lastwritetime;
            bool result = (GetFileTime(hFile, 0, 0, &lastwritetime) != 0);
            CloseHandle(hFile);
            dateTime = static_cast<uint64_t>(lastwritetime.dwLowDateTime) | (static_cast<uint64_t>(lastwritetime.dwHighDateTime) << 32);

            return result;
        }
        return false;
    }

    bool directoryExists(const wchar_t *path)
    {
        if (PathIsRoot(path))
            return true;

        if (PathIsDirectory(path))
            return true;

        if (PathFileExists(path))
            return true;

        return false;
    }

    bool validateFileExists(const std::wstring& absolutePath)
    {
        //Seems like all existence functions fail for files surrounded by double quotes
        return (_waccess(absolutePath.c_str(), 0) == 0);
    }

    bool createDirectoryRelwrsively(const wchar_t *path)
    {
        if (directoryExists(path))
            return true;

        wchar_t folder[MAX_PATH];
        ZeroMemory(folder, MAX_PATH * sizeof(wchar_t));

        const wchar_t* endBackslash = wcschr(path, L'\\'), *endFrontslash = wcschr(path, L'/');
        const wchar_t* endFirst = endBackslash < endFrontslash ? endBackslash : endFrontslash;
        const wchar_t* endLast = endBackslash >= endFrontslash ? endBackslash : endFrontslash;
        const wchar_t* end = endFirst ? endFirst : endLast; //if the pointer is zero, try the other one

        bool success = true;
        while (end != NULL)
        {
            wcsncpy_s(folder, path, end - path + 1);

            if (!directoryExists(folder))
            {
                if (!CreateDirectory(folder, NULL))
                {
                    success = false;
                    break;
                }
            }

            end += 1;
            endBackslash = wcschr(end, L'\\'); 
            endFrontslash = wcschr(end, L'/');
            endFirst = endBackslash < endFrontslash ? endBackslash : endFrontslash;
            endLast = endBackslash >= endFrontslash ? endBackslash : endFrontslash;
            end = endFirst ? endFirst : endLast; //if the pointer is zero, try the other one
        }
        return success;
    }

    std::wstring GetFileName(const std::wstring& fullFilePath)
    {
        std::wstring directory;
        std::wstring fileName;
        SplitPathIntoDirectoryAndFileName(fullFilePath, directory, fileName);
        return fileName;
    }

    bool SplitPathIntoDirectoryAndFileName(const std::wstring& fullFilePath, std::wstring& directory_out, std::wstring& fileName_out)
    {
        directory_out = L"";
        size_t posOfLastSlash = fullFilePath.find_last_of('\\');
        if (posOfLastSlash == std::string::npos)
        {
            fileName_out = fullFilePath;
            return false;
        }
        directory_out = fullFilePath.substr(0, posOfLastSlash + 1);
        fileName_out = fullFilePath.substr(posOfLastSlash + 1);
        return true;
    }

    void getLwrrentPath(wchar_t * path, size_t maxSize)
    {
        GetModuleFileName(NULL, path, (DWORD)maxSize);
        // TODO[error]: add check if funciton failed
        for (ptrdiff_t i = wcslen(path) - 1; i >= 0; --i)
        {
            if (path[i] == L'\\' || path[i] == L'/')
            {
                path[i] = L'\0';
                break;
            }
        }
    }

    bool MappedFile::open(const wchar_t* filename)
    {
        if (m_mapAddr)
            return false; //already open

        m_fileHandle = CreateFile(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

        if (m_fileHandle == ILWALID_HANDLE_VALUE)
        {
            return false;
        }

        unsigned long fileSize = GetFileSize(m_fileHandle, NULL);

        if (!fileSize)
        {
            CloseHandle(m_fileHandle);
            m_fileHandle = ILWALID_HANDLE_VALUE;

            return false;
        }

        m_mappingHandle = CreateFileMapping(m_fileHandle, NULL, PAGE_READONLY, 0, 0, NULL);

        if (m_mappingHandle == NULL)
        {
            CloseHandle(m_fileHandle);
            m_fileHandle = ILWALID_HANDLE_VALUE;

            return false;
        }

        // Map the view and test the results.

        m_mapAddr = (unsigned char*)MapViewOfFile(m_mappingHandle, FILE_MAP_READ, 0, 0, 0);

        if (m_mapAddr == NULL)
        {
            CloseHandle(m_mappingHandle);
            m_mappingHandle = NULL;
            CloseHandle(m_fileHandle);
            m_fileHandle = ILWALID_HANDLE_VALUE;

            return false;
        }

        m_endMapaddr = m_mapAddr + fileSize;

        return true;
    }

    void MappedFile::close()
    {
        if (m_mapAddr)
            UnmapViewOfFile(m_mapAddr);

        m_mapAddr = m_endMapaddr = nullptr;

        if (m_mappingHandle != NULL)
            CloseHandle(m_mappingHandle); 

        m_mappingHandle = NULL;

        if (m_fileHandle != ILWALID_HANDLE_VALUE)
            CloseHandle(m_fileHandle);

        m_fileHandle = ILWALID_HANDLE_VALUE;
    }
            
    MappedFile::MappedFile() :
        m_fileHandle(ILWALID_HANDLE_VALUE),
        m_mappingHandle(NULL),
        m_mapAddr(nullptr),
        m_endMapaddr(nullptr)
    {}

    MappedFile::MappedFile(const wchar_t* filename):
        m_fileHandle(ILWALID_HANDLE_VALUE),
        m_mappingHandle(NULL),
        m_mapAddr(nullptr),
        m_endMapaddr(nullptr)
    {
        open(filename);
    }

    MappedFile::~MappedFile()
    {
        close();
    }
}
}
}
