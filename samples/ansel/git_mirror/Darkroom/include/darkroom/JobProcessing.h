#pragma once
#include <string>
#include <chrono>
#pragma warning(push)
#pragma warning(disable:4668)
#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>
#pragma warning(pop)

#include "Errors.h"

namespace darkroom
{
    enum class JobType
    {
        UNKNOWN,
        REGULAR,
        REGULAR_STEREO,
        REGULAR_STEREO_EXR,
        SPHERICAL_MONO,
        SPHERICAL_MONO_EXR,
        SPHERICAL_STEREO,
        SPHERICAL_STEREO_EXR,
        HIGHRES,
        HIGHRES_EXR
    };

    int initializeJobProcessing(const std::wstring& highresBlenderAbsolutePath,
        const std::wstring& sphericalEquirectAbsolutePath,
        const std::wstring& colwertAbsolutePath,
        const std::wstring& thumbnailToolAbsolutePath,
        const std::wstring& tempPath);


    std::string safeQuote(const std::string& input);
    std::wstring safeQuote(const std::wstring& input);

    std::chrono::time_point<std::chrono::system_clock> generateTimestamp();

    std::wstring generateFileName(JobType type, const std::chrono::time_point<std::chrono::system_clock>& timepoint, const std::wstring& titleName, const std::wstring & ext = L".jpg", const std::wstring& suffix = L"");
    
    // auto-detect the type of job to be performed, perform it and optionally remove the path (with all the files inside)
    // use WaitForSingleObject(process, INFINITE) to determine if the process has finished (so you could remove the working directory)
    // TODO: colwert this mess into a structure
    Error processJob(const std::wstring& path, 
        HANDLE& process,
        const std::wstring& appTitleName,
        const std::wstring& tagDescription,
        const std::wstring& appCMSID,
        const std::wstring& appShortName,
        const std::wstring& outputPath,
        std::wstring& outputFilename,
        const std::wstring& tagModel,
        const std::wstring& tagSoftware,
        const std::wstring& tagDrsName,
        const std::wstring& tagDrsProfileName,
        const std::wstring& tagActiveFilters,
        bool isRawHDR = false,
        bool keepIntermediateShots = false,
        bool forceLosslessHighRes = false,
        bool forceLossless360 = false,
        float enhancedHighresCoeff = 0.5f,
        bool isFrequencyTransferEnabled = false,
        bool generateThumbnail = false
        );

    // deletes directory relwrsively
    bool deleteDirectory(const std::wstring& path);

    // exelwtes a process and gives a handle to monitor it
    bool exelwteProcess(const std::wstring& exelwtable, const std::vector<std::wstring>& args, const std::wstring& workDir, HANDLE& process);
}
