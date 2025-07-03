#include "darkroom/Errors.h"
#include "darkroom/Director.h"
#include "darkroom/JobProcessing.h"
#include "darkroom/Blend.h"
#include "darkroom/StringColwersion.h"
#include <fstream>
#include <functional> 
#include <cctype>
#include <locale>
#include <codecvt>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <array>

#include <tchar.h>
#include <shellapi.h>
#include <Shlwapi.h>

#pragma comment(lib, "Shlwapi.lib")
#pragma warning(disable:4127) // condition is always constant
#pragma warning(disable:4571) // Informational: catch(...) semantics changed since Visual C++ 7.1; structured exceptions (SEH) are no longer caught
#pragma warning(disable:4265) // class has virtual functions, but destructor is not virtual - lambdas cause that

#include <thread>

namespace darkroom
{
    namespace
    {
        const unsigned int gUseConservativeModeForMultipliersHigherThan = 4u;
        const bool gThrottleTools = true;
        const std::wstring gTagMake = L"LWPU";

        static std::wstring s_highresBlenderAbsolutePath;
        static std::wstring s_sphericalEquirectAbsolutePath;
        static std::wstring s_colwertAbsolutePath;
        static std::wstring s_thumbnailToolAbsolutePath;
        static std::wstring s_tempPath;
        static std::wstring s_wSphericalTileName;
        static std::wstring s_wHighresTileName;
        static std::wstring s_wRegularName;
        static std::wstring s_wThumbnailName;
        static std::wstring s_wRegularStereoTileName;
        static std::wstring s_throttleArgument;

        std::wstring generateCmdLine(const std::wstring& exelwtable, const std::vector<std::wstring>& args)
        {
            std::wostringstream argsJoined;
            std::copy(args.begin(), args.end(), std::ostream_iterator<std::wstring, wchar_t>(argsJoined, L" "));
            return exelwtable + L" " + argsJoined.str();
        }

        namespace windows
        {
            std::vector<std::wstring> getFileList(const std::wstring& path)
            {
                HANDLE hFind;
                WIN32_FIND_DATA data;

                std::vector<std::wstring> list;

                hFind = FindFirstFile(path.c_str(), &data);
                if (hFind != ILWALID_HANDLE_VALUE)
                {
                    do {
                        std::wstring fileName(data.cFileName);
                        if (fileName.compare(L".") != 0 && fileName.compare(L"..") != 0)
                            list.push_back(fileName);
                    } while (FindNextFile(hFind, &data));
                    FindClose(hFind);
                }
                return list;
            }

            bool validateFileExists(const std::wstring& absolutePath)
            {
                //Seems like all existence functions fail for files surrounded by double quotes
                return (_waccess(absolutePath.c_str(), 0) == 0);  
            }

            std::wstring findExtension(const std::wstring& filename)
            {
                const wchar_t* extPtr = PathFindExtension(filename.c_str());
                if (*extPtr != '\0')
                    return std::wstring(extPtr);
                else
                    return std::wstring();
            }
        }
        
        JobType determineJobType(const std::wstring& path, size_t& shotCount, std::wstring& regularScreenshotPath, std::wstring& thumbnailPath)
        {
            const auto fileList = windows::getFileList(path + L"\\*.*");

            for (auto& filename : fileList)
            {
                if (filename == L"captures.txt")
                {
                    shotCount = fileList.size() - 1;
                    for (auto& filename : fileList)
                    {
                        const auto ext = windows::findExtension(filename);
                        if (ext == L".exr")
                            return JobType::SPHERICAL_MONO_EXR;
                        else if (ext == L".bmp")
                            return JobType::SPHERICAL_MONO;
                    }
                }
                else if (filename == L"capturesL.txt" || filename == L"capturesR.txt")
                {
                    shotCount = fileList.size() - 2;
                    for (auto& filename : fileList)
                    {
                        const auto ext = windows::findExtension(filename);
                        if (ext == L".exr")
                            return JobType::SPHERICAL_STEREO_EXR;
                        else if (ext == L".bmp")
                            return JobType::SPHERICAL_STEREO;
                    }
                }
                else if (filename.find(s_wHighresTileName.c_str(), 0) != std::string::npos)
                {
                    shotCount = 0;
                    const auto ext = windows::findExtension(filename);
                    if (ext == L".exr")
                    {
                        shotCount += fileList.size();
                        // In case there is 'regular.bmp' shot - this means we want to enhance super-resolution shot.
                        // In this case we decrement shotCount for the following function to work properly - shotCount
                        // should reflect the shot count for stitching/blending process and this only includes super-resolution
                        // tiles (otherwise the shot count wont be accepted as valid for the given multiplier).
                        size_t shotsFiltered = 0u;
                        for (auto& filename : fileList)
                        {
                            const bool isRegular = filename.find(s_wRegularName.c_str(), 0) != std::string::npos;
                            const bool isThumbnail = filename.find(s_wThumbnailName.c_str(), 0) != std::string::npos;
                            
                            if (isThumbnail || isRegular)
                            {
                                shotCount -= 1;
                                shotsFiltered += 1u;
                                if (isRegular)
                                    regularScreenshotPath = filename;
                                else
                                    thumbnailPath = filename;
                                if (shotsFiltered == 2)
                                    break;
                            }
                        }
                        return JobType::HIGHRES_EXR;
                    }
                    else if (ext == L".bmp")
                    {
                        shotCount += fileList.size();
                        // In case there is 'regular.bmp' shot - this means we want to enhance super-resolution shot.
                        // In this case we decrement shotCount for the following function to work properly - shotCount
                        // should reflect the shot count for stitching/blending process and this only includes super-resolution
                        // tiles (otherwise the shot count wont be accepted as valid for the given multiplier).
                        size_t shotsFiltered = 0u;
                        for (auto& filename : fileList)
                        {
                            const bool isRegular = filename.find(s_wRegularName.c_str(), 0) != std::string::npos;
                            const bool isThumbnail = filename.find(s_wThumbnailName.c_str(), 0) != std::string::npos;

                            if (isThumbnail || isRegular)
                            {
                                shotCount -= 1;
                                shotsFiltered += 1u;
                                if (isRegular)
                                    regularScreenshotPath = filename;
                                else
                                    thumbnailPath = filename;
                                if (shotsFiltered == 2)
                                    break;
                            }
                        }
                        return JobType::HIGHRES;
                    }
                }
                else if (filename.find(s_wRegularStereoTileName.c_str(), 0) != std::string::npos)
                {
                    shotCount = 2;
                    const auto ext = windows::findExtension(filename);
                    for (auto& filename : fileList)
                    {
                        if (filename.find(s_wThumbnailName.c_str(), 0) != std::string::npos)
                        {
                            thumbnailPath = filename;
                            break;
                        }
                    }

                    if (ext == L".exr")
                        return JobType::REGULAR_STEREO_EXR;
                    else if (ext == L".bmp")
                        return JobType::REGULAR_STEREO;
                }
            }

            return JobType::UNKNOWN;
        }

        void replaceStringInPlace(std::string& str, const std::string& search, const std::string& replace) 
        {
            size_t pos = 0;
            while ((pos = str.find(search, pos)) != std::string::npos)
            {
                str.replace(pos, search.length(), replace);
                pos += replace.length();
            }
        }
    }
    
    bool exelwteProcess(const std::wstring& exelwtable, const std::vector<std::wstring>& args, const std::wstring& workDir, HANDLE& process)
    {
        STARTUPINFO si;
        PROCESS_INFORMATION pi;

        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        ZeroMemory(&pi, sizeof(pi));

        const auto cmdLineString = generateCmdLine(exelwtable, args);
        wchar_t* cmdLine = const_cast<wchar_t*>(cmdLineString.c_str()); // ouch

        if (CreateProcess(NULL, cmdLine, NULL, NULL, FALSE, CREATE_UNICODE_ELWIRONMENT | NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW, NULL, workDir.c_str(), &si, &pi) == FALSE)
        {
            const DWORD error = GetLastError();
            return false;
        }

        process = pi.hProcess;

        CloseHandle(pi.hThread);

        return true;
    }

    std::chrono::time_point<std::chrono::system_clock> generateTimestamp()
    {
        using namespace std::chrono;
        return system_clock::now();
    }

    std::wstring generateFileName(JobType type, const std::chrono::time_point<std::chrono::system_clock>& timepoint, const std::wstring& titleName, const std::wstring & ext, const std::wstring& suffix)
    {
        using namespace std::chrono;

        const auto millisecondsSinceEpoch = duration_cast<milliseconds>(timepoint.time_since_epoch()).count();
        const auto centiseconds = ((millisecondsSinceEpoch / 10) % 100);
        const auto timetnow = system_clock::to_time_t(timepoint);
        tm nowLocal;
        localtime_s(&nowLocal, &timetnow);
        std::wstring shottype;

        if (type == JobType::HIGHRES || type == JobType::HIGHRES_EXR)
            shottype = L"Super-Resolution";
        else if (type == JobType::REGULAR)
            shottype = L"Screenshot";
        else if (type == JobType::SPHERICAL_MONO || type == JobType::SPHERICAL_MONO_EXR)
            shottype = L"360";
        else if (type == JobType::SPHERICAL_STEREO || type == JobType::SPHERICAL_STEREO_EXR)
            shottype = L"360-Stereo";
        else if (type == JobType::REGULAR_STEREO || type == JobType::REGULAR_STEREO_EXR)
            shottype = L"Screenshot-Stereo";

        std::wstringstream in;

        in << titleName << L" ";

        in << shottype << L" " << nowLocal.tm_year + 1900 << L"."
            << std::setfill(L'0') << std::setw(2) << nowLocal.tm_mon + 1 << L"."
            << std::setfill(L'0') << std::setw(2) << nowLocal.tm_mday << L" - "
            << std::setfill(L'0') << std::setw(2) << nowLocal.tm_hour << L"."
            << std::setfill(L'0') << std::setw(2) << nowLocal.tm_min << L"."
            << std::setfill(L'0') << std::setw(2) << nowLocal.tm_sec << L"."
            << std::setfill(L'0') << std::setw(2) << centiseconds << suffix << ext;

        return in.str();
    }

    bool deleteDirectory(const std::wstring& path)
    {
        size_t len = _tcslen(path.c_str());
        TCHAR *pszFrom = new TCHAR[len + 2];
        std::copy(path.cbegin(), path.cend(), pszFrom);
        pszFrom[len] = 0;
        pszFrom[len + 1] = 0;

        SHFILEOPSTRUCT fileop;
        fileop.hwnd = NULL;    // no status display
        fileop.wFunc = FO_DELETE;  // delete operation
        fileop.pFrom = pszFrom;  // source file name as double null terminated string
        fileop.pTo = NULL;    // no destination needed
        fileop.fFlags = FOF_NOCONFIRMATION | FOF_SILENT;  // do not prompt the user
        fileop.fAnyOperationsAborted = FALSE;
        fileop.lpszProgressTitle = NULL;
        fileop.hNameMappings = NULL;

        int ret = SHFileOperation(&fileop);
        delete[] pszFrom;
        return ret == 0;
    }

    void generateRemovePath(std::wofstream& out, const std::wstring& path)
    {
        out << L"if %errorlevel% neq 0 exit /b %errorlevel%" << std::endl;
        out << L"cd .." << std::endl;
        out << L"RMDIR /S /Q " << path << std::endl;
    }

    int initializeJobProcessing(const std::wstring& highresBlenderAbsolutePath, 
                                const std::wstring& sphericalEquirectAbsolutePath,
                                const std::wstring& colwertAbsolutePath,
                                const std::wstring& thumbnailToolAbsolutePath,
                                const std::wstring& tempPath)
    {
        using windows::validateFileExists;

        // TODO: add thumbnailTool path here as soon as we'll finalize the tool presence
        const std::vector<std::wstring> paths = { highresBlenderAbsolutePath, sphericalEquirectAbsolutePath, colwertAbsolutePath };

        for (const auto& path : paths)
            if (!validateFileExists(path))
                return -1;

        //Add the double quotes here to:
        //1. fix being able to run programs with spaces in their paths
        //2. fix for file checks failing if paths are double quoted.
        s_highresBlenderAbsolutePath = L'"' + highresBlenderAbsolutePath + L'"';
        s_sphericalEquirectAbsolutePath = L'"' + sphericalEquirectAbsolutePath + L'"';
        s_colwertAbsolutePath = L'"' + colwertAbsolutePath + L'"';
        s_thumbnailToolAbsolutePath = L'"' + thumbnailToolAbsolutePath + L'"';
        s_tempPath = tempPath;
            
        s_wSphericalTileName = darkroom::getWstrFromUtf8(darkroom::SphericalTileName);
        s_wHighresTileName = darkroom::getWstrFromUtf8(darkroom::HighresTileName);
        s_wRegularStereoTileName = darkroom::getWstrFromUtf8(darkroom::RegularStereoTileName);
        s_wRegularName = darkroom::getWstrFromUtf8(darkroom::RegularName);
        s_wThumbnailName = darkroom::getWstrFromUtf8(darkroom::ThumbnailName);

        if (gThrottleTools)
        {
            std::wstringstream ss;
            ss << L"--threads " << std::thread::hardware_conlwrrency() / 2;
            s_throttleArgument = ss.str();
        }

        return 0;
    }

    std::wstring safeQuote(const std::wstring& input)
    {
        const std::wstring quote(L"\"");
        auto inputCopy = input;
        // trim from start
        inputCopy.erase(inputCopy.begin(), std::find_if(inputCopy.begin(), inputCopy.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        // trim from end
        inputCopy.erase(std::find_if(inputCopy.rbegin(), inputCopy.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), inputCopy.end());

        if (inputCopy.empty())
            return std::wstring(L"\"None\"");
        return quote + inputCopy + quote;
    };

    std::string safeQuote(const std::string& input)
    {
        const std::string quote("\"");
        auto inputCopy = input;
        // trim from start
        inputCopy.erase(inputCopy.begin(), std::find_if(inputCopy.begin(), inputCopy.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        // trim from end
        inputCopy.erase(std::find_if(inputCopy.rbegin(), inputCopy.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), inputCopy.end());

        if (inputCopy.empty())
            return std::string("\"None\"");
        return quote + inputCopy + quote;
    };

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
        bool isRawHDR,
        bool keepIntermediateShots, 
        bool forceLosslessHighRes, 
        bool forceLossless360, 
        float enhancedHighresCoeff, 
        bool isFrequencyTransferEnabled,
        bool generateThumbnail)
    {
        using windows::findExtension;
        using windows::validateFileExists;

        size_t shotCount = 0;
        std::wstring regularScreenshotPath, thumbnailScreenshotPath;
        const auto shotType = determineJobType(path, shotCount, regularScreenshotPath, thumbnailScreenshotPath);
        
        if (shotType == JobType::UNKNOWN) 
            return Error::kUnknownJobType;
        
        // TODO avoroshilov: obsolete? tools now decide extension on their own
        std::wstring ext = findExtension(outputFilename);

        const bool isJobTypeEXR =
            (shotType == JobType::HIGHRES_EXR) ||
            (shotType == JobType::REGULAR_STEREO_EXR) ||
            (shotType == JobType::SPHERICAL_MONO_EXR) ||
            (shotType == JobType::SPHERICAL_STEREO_EXR);
        if (isJobTypeEXR)
            ext = L".exr";

        else if (shotType == JobType::SPHERICAL_MONO ||
            shotType == JobType::SPHERICAL_STEREO)
        {
            ext = forceLossless360 ? L".png" : L".jpg";
        }
        else if (shotType == JobType::HIGHRES)
        {
            ext = forceLosslessHighRes ? L".png" : L".jpg";
        }
        else if (shotType == JobType::REGULAR_STEREO)
        {
            ext = L".jpg";
        }

        const auto timepoint = generateTimestamp();
        const auto tempFileName = generateFileName(shotType, timepoint, appTitleName, ext, isRawHDR ? L" Raw" : L"");
        outputFilename = outputPath + tempFileName;

        const std::wstring quote(L"\"");
        const auto quotedPath = quote + path + quote;
        const auto filename = quote + outputFilename + quote;
        const auto fullBatchFileName = path + L"\\run.bat";

        const std::wstring escapedAppTitleName = safeQuote(appTitleName);
        const std::wstring escapedTagDescription = safeQuote(tagDescription);
        const std::wstring escapedAppCMSID = safeQuote(appCMSID);
        const std::wstring escapedAppShortName = safeQuote(appShortName);
        const std::wstring escapedTagModel = safeQuote(tagModel);
        const std::wstring escapedTagSoftware = safeQuote(tagSoftware);
        const std::wstring escapedTagDrsName = safeQuote(tagDrsName);
        const std::wstring escapedTagDrsProfileName = safeQuote(tagDrsProfileName);
        const std::wstring escapedTagActiveFilters = safeQuote(tagActiveFilters);
        bool batchReady = false;

        {
            std::wofstream out(fullBatchFileName);
            
            // Output stream must be in UTF-8 to support variety of characters
            const std::locale locale_utf8 = std::locale(std::locale(), new std::codecvt_utf8<wchar_t>());
            out.imbue(locale_utf8);
            // Setting codepage to UTF-8 since we're saving run.bat in UTF-8
            out << L"chcp 65001" << std::endl;

            if (shotType == JobType::HIGHRES || shotType == JobType::HIGHRES_EXR)
            {
                if (isHighresAllowed(static_cast<unsigned int>(shotCount)) != Error::kSuccess)
                    return Error::kIlwalidArgumentCount;

                // execute HighresBlender --threads ... [-conserve] --make ... --model ... --type ... --software ... --output ...
                std::vector<std::wstring> highresBlenderArgs;
                const auto highresMultiplier = (static_cast<unsigned int>(sqrt(static_cast<float>(shotCount))) + 1) / 2;

                highresBlenderArgs.push_back(s_throttleArgument);

                if (highresMultiplier >= gUseConservativeModeForMultipliersHigherThan)
                    highresBlenderArgs.push_back(L"--conserve");

                const auto guardedPath = quote + outputPath + tempFileName.substr(0, tempFileName.find_last_of(L".")) + quote;
                highresBlenderArgs.insert(highresBlenderArgs.end(), 
                {
                    L"--make", gTagMake,
                    L"--model", escapedTagModel,
                    L"--type", L"SuperResolution",
                    L"--description", escapedTagDescription,
                    L"--software", escapedTagSoftware,
                    L"--drsname", escapedTagDrsName,
                    L"--drsprofilename", escapedTagDrsProfileName,
                    L"--apptitlename", escapedAppTitleName,
                    L"--appcmsid", escapedAppCMSID,
                    L"--appshortname", escapedAppShortName,
                    L"--activefilters", escapedTagActiveFilters,
                    L"--output", guardedPath
                });

                if (forceLosslessHighRes)
                    highresBlenderArgs.push_back(L"--lossless");

                if (!regularScreenshotPath.empty())
                {
                    // setup freq transfer parameters
                    if (isFrequencyTransferEnabled)
                    {
                        std::wstringstream ss;
                        ss << L"--freq-transfer-alpha " << enhancedHighresCoeff << " --freq-transfer-input " << quote << regularScreenshotPath << quote;
                        highresBlenderArgs.push_back(ss.str());
                    }
                }

                if (generateThumbnail)
                {
                    const auto thumbnailFileName = generateFileName(shotType, timepoint, appTitleName, L".png", gThumbnailSuffix);
                    const auto thumbnailPath = quote + outputPath + thumbnailFileName + quote;
                    out << generateCmdLine(s_colwertAbsolutePath, { 
                        L"--make", gTagMake,
                        L"--model", escapedTagModel,
                        L"--type", L"SuperResolution",
                        L"--description", escapedTagDescription,
                        L"--software", escapedTagSoftware,
                        L"--drsname", escapedTagDrsName,
                        L"--drsprofilename", escapedTagDrsProfileName,
                        L"--apptitlename", escapedAppTitleName,
                        L"--appcmsid", escapedAppCMSID,
                        L"--appshortname", escapedAppShortName,
                        L"--activefilters", escapedTagActiveFilters,
                        L"--colwert", thumbnailScreenshotPath, thumbnailPath }) << std::endl;
                }

                out << generateCmdLine(s_highresBlenderAbsolutePath, highresBlenderArgs) << std::endl;

                if (!keepIntermediateShots)
                    generateRemovePath(out, quotedPath);

                batchReady = true;
            }
            else if (shotType == JobType::SPHERICAL_MONO || shotType == JobType::SPHERICAL_MONO_EXR)
            {
                std::vector<std::wstring> sphericalEquirectArgs;

                // SphericalEquirect captures.exe result.jpg
                out << generateCmdLine(s_sphericalEquirectAbsolutePath, 
                {
                    s_throttleArgument, L"captures.txt", filename, 
                    L"--360",
                    L"--make", gTagMake,
                    L"--model", escapedTagModel,
                    L"--type", L"360Mono",
                    L"--description", escapedTagDescription,
                    L"--software", escapedTagSoftware,
                    L"--drsname", escapedTagDrsName,
                    L"--drsprofilename", escapedTagDrsProfileName,
                    L"--apptitlename", escapedAppTitleName,
                    L"--appcmsid", escapedAppCMSID,
                    L"--appshortname", escapedAppShortName,
                    L"--activefilters", escapedTagActiveFilters,
                }) << std::endl;

                if (generateThumbnail)
                {
                    const auto thumbnailFileName = generateFileName(shotType, timepoint, appTitleName, L".jpg", gThumbnailSuffix);
                    const auto thumbnailPath = quote + outputPath + thumbnailFileName + quote;
                    out << generateCmdLine(s_sphericalEquirectAbsolutePath,
                    {
                        s_throttleArgument, L"capturesThumbnail.txt",
                        thumbnailPath, L"--360",
                        L"--make", gTagMake,
                        L"--model", escapedTagModel,
                        L"--type", L"360Mono",
                        L"--description", escapedTagDescription,
                        L"--software", escapedTagSoftware,
                        L"--drsname", escapedTagDrsName,
                        L"--drsprofilename", escapedTagDrsProfileName,
                        L"--apptitlename", escapedAppTitleName,
                        L"--appcmsid", escapedAppCMSID,
                        L"--appshortname", escapedAppShortName,
                        L"--activefilters", escapedTagActiveFilters,
                    }) << std::endl;
                }

                if (!keepIntermediateShots)
                    generateRemovePath(out, quotedPath);

                batchReady = true;
            }
            else if (shotType == JobType::SPHERICAL_STEREO || shotType == JobType::SPHERICAL_STEREO_EXR)
            {
                // Using 'outputFilename' since it is not guarded with quotes
                std::wstring ext = findExtension(outputFilename);
                bool isHDR = false;
                if (ext == L".exr")
                {
                    isHDR = true;
                }

                wchar_t * resultFilenameL = isHDR ? L"resultL.exr" : L"resultL.jpg";
                wchar_t * resultFilenameR = isHDR ? L"resultR.exr" : L"resultR.jpg";

                // SphericalEquirect capturesL.exe resultL.jpg
                out << generateCmdLine(s_sphericalEquirectAbsolutePath, { s_throttleArgument, L"capturesL.txt", resultFilenameL }) << std::endl;
                // SphericalEquirect capturesR.exe resultR.jpg
                out << generateCmdLine(s_sphericalEquirectAbsolutePath, { s_throttleArgument, L"capturesR.txt", resultFilenameR }) << std::endl;
                // append two images + add tags
                out << generateCmdLine(s_colwertAbsolutePath, 
                {
                    L"--append-vertically", resultFilenameL, resultFilenameR,
                    filename,
                    L"--make", gTagMake,
                    L"--model", escapedTagModel,
                    L"--type", L"360Stereo",
                    L"--description", escapedTagDescription,
                    L"--software", escapedTagSoftware,
                    L"--drsname", escapedTagDrsName,
                    L"--drsprofilename", escapedTagDrsProfileName,
                    L"--apptitlename", escapedAppTitleName,
                    L"--appcmsid", escapedAppCMSID,
                    L"--appshortname", escapedAppShortName,
                    L"--activefilters", escapedTagActiveFilters,
                }) << std::endl;

                if (generateThumbnail)
                {
                    const auto thumbnailFileName = generateFileName(shotType, timepoint, appTitleName, L".jpg", gThumbnailSuffix);
                    const auto thumbnailPath = quote + outputPath + thumbnailFileName + quote;
                    out << generateCmdLine(s_sphericalEquirectAbsolutePath,
                    {
                        s_throttleArgument, L"capturesThumbnail.txt",
                        thumbnailPath, L"--360",
                        L"--make", gTagMake,
                        L"--model", escapedTagModel,
                        L"--type", L"360Stereo",
                        L"--description", escapedTagDescription,
                        L"--software", escapedTagSoftware,
                        L"--drsname", escapedTagDrsName,
                        L"--drsprofilename", escapedTagDrsProfileName,
                        L"--apptitlename", escapedAppTitleName,
                        L"--appcmsid", escapedAppCMSID,
                        L"--appshortname", escapedAppShortName,
                        L"--activefilters", escapedTagActiveFilters,
                    }) << std::endl;
                }

                if (!keepIntermediateShots)
                    generateRemovePath(out, quotedPath);

                batchReady = true;
            }
            else if (shotType == JobType::REGULAR_STEREO || shotType == JobType::REGULAR_STEREO_EXR)
            {
                // append two images + add tags
                out << generateCmdLine(s_colwertAbsolutePath, 
                { 
                    L"--append-horizontally",
                    s_wRegularStereoTileName + (shotType == JobType::REGULAR_STEREO ? L"L.bmp" : L"L.exr"),
                    s_wRegularStereoTileName + (shotType == JobType::REGULAR_STEREO ? L"R.bmp" : L"R.exr"),
                    filename,
                    L"--make", gTagMake,
                    L"--model", escapedTagModel,
                    L"--type", L"Stereo",
                    L"--description", escapedTagDescription,
                    L"--software", escapedTagSoftware,
                    L"--drsname", escapedTagDrsName,
                    L"--drsprofilename", escapedTagDrsProfileName,
                    L"--apptitlename", escapedAppTitleName,
                    L"--appcmsid", escapedAppCMSID,
                    L"--appshortname", escapedAppShortName,
                    L"--activefilters", escapedTagActiveFilters,
                }) << std::endl;

                if (generateThumbnail)
                {
                    const auto thumbnailFileName = generateFileName(shotType, timepoint, appTitleName, L".png", gThumbnailSuffix);
                    const auto thumbnailPath = quote + outputPath + thumbnailFileName + quote;
                    out << generateCmdLine(s_colwertAbsolutePath, { 
                        L"--make", gTagMake,
                        L"--model", escapedTagModel,
                        L"--type", L"Stereo",
                        L"--description", escapedTagDescription,
                        L"--software", escapedTagSoftware,
                        L"--drsname", escapedTagDrsName,
                        L"--drsprofilename", escapedTagDrsProfileName,
                        L"--apptitlename", escapedAppTitleName,
                        L"--appcmsid", escapedAppCMSID,
                        L"--appshortname", escapedAppShortName,
                        L"--activefilters", escapedTagActiveFilters,
                        L"--colwert", thumbnailScreenshotPath, thumbnailPath }) << std::endl;
                }

                if (!keepIntermediateShots)
                    generateRemovePath(out, quotedPath);

                batchReady = true;
            }
            else
                return Error::kOperationFailed; // shouldn't be here ever
        }

        if (batchReady)
        {
            if (!exelwteProcess(L"cmd.exe", { L"/C run.bat" }, path, process))
                return Error::kCouldntStartupTheProcess;
        }

        return Error::kSuccess;
    }
}
