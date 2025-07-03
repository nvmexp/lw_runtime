// HighresBlender.cpp : Defines the entry point for the console application.
//

#include <fstream>
#include <iterator>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <thread>
#include <array>
#include <unordered_map>
#include <Windows.h>

#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")
#include "darkroom/Bmp.h"
#include "darkroom/Exr.h"
#include "darkroom/Jpeg.h"
#include "darkroom/Png.h"
#include "darkroom/InternalLimits.h"
#include "darkroom/Errors.h"
#include "darkroom/Blend.h"
#include "darkroom/Director.h"
#include "darkroom/PixelFormat.h"
#include "darkroom/Versioning.hpp"
#include "darkroom/FrequencyTransfer.h"
#include "darkroom/CmdLineParser.hpp"
#include "darkroom/StringColwersion.h"
#include "darkroom/Jpeg.h"

using std::ifstream;
using std::ofstream;
using std::cout;
using std::endl;

namespace
{
    HANDLE gFileMapping = nullptr, gFileMappingFreqTransfer = nullptr;
    uint64_t gAllocationGranularity = 0;

    std::vector<std::wstring> tileNames;
    std::vector<std::vector<unsigned char>> tilesBmp;
    std::vector<unsigned char*> tileBmpPtrs;
    std::vector<std::vector<float>> tilesExr;
    std::vector<float*> tileExrPtrs;

    namespace frequency_transfer
    {
        std::vector<darkroom::FFTPersistentData> persistentData;
        std::vector<unsigned char> regular8bit;
        std::vector<float> regular32bit;
        unsigned int regularWidth = 0, regularHeight = 0;
        float alpha = .5f;
    }

    unsigned int tilesInRow = 0;
    unsigned int multiplier = 0;
    unsigned int tileWidth = 0, tileHeight = 0;

    unsigned int workerCount = 0;

    enum class OutputFiletype
    {
        JPEG,
        PNG,
        EXR
    };

    struct FrequencyTransferTileParameters
    {
        uint32_t paddingOffsetW;
        uint32_t paddingOffsetH;
        uint32_t paddingSizeW;
        uint32_t paddingSizeH;
    };

    namespace windows
    {
        std::vector<std::wstring> discoverTiles(const std::wstring& path)
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

        std::wstring findExtension(const std::wstring& filename)
        {
            const wchar_t* extPtr = PathFindExtension(filename.c_str());
            if (*extPtr != '\0')
                return std::wstring(extPtr);
            else
                return std::wstring();
        }

        std::string findExtension(const std::string& filename)
        {
            const auto* extPtr = PathFindExtensionA(filename.c_str());
            if (*extPtr != '\0')
                return std::string(extPtr);
            else
                return std::string();
        }
    }

    namespace memorymapping
    {
        void* lastMappedAddress = nullptr;
        void* lastMappedAddressFreqTransfer = nullptr;
        uint64_t lwrrentRow = 0, rawDataSize = 0, imageHeight = 0, imagePitch = 0, rowsPerWindow = 0;
    }

    void signalFatal(bool cond, const char* message)
    {
        if (cond)
        {
            std::cerr << message << std::endl;
            exit(-1);
        }
    }

    std::unordered_map<uint16_t, std::string> generateExifTable(darkroom::CmdLineParser& cmdParser)
    {
        // These ID#s are defined in the file Jpeg.h
        std::unordered_map<uint16_t, std::string> result;
        if (cmdParser.hasOption("make"))
            result[darkroom::gJPEG_TAG_SOURCE] = cmdParser.getOptionAs<std::string>("make");
        if (cmdParser.hasOption("model"))
        {
            const auto model = cmdParser.getOptionAs<std::string>("model");
            result[darkroom::gJPEG_TAG_MODEL_1] = model;
            result[darkroom::gJPEG_TAG_MODEL_2] = model;
        }
        if (cmdParser.hasOption("software"))
            result[darkroom::gJPEG_TAG_SOFTWARE] = cmdParser.getOptionAs<std::string>("software");
        if (cmdParser.hasOption("description"))
            result[darkroom::gJPEG_TAG_DESCRIPTION] = cmdParser.getOptionAs<std::string>("description");
        if (cmdParser.hasOption("type"))
            result[darkroom::gJPEG_TAG_TYPE] = cmdParser.getOptionAs<std::string>("type");
        // Exif.Image.ProcessingSoftware
        if (cmdParser.hasOption("drsname"))
            result[darkroom::gJPEG_TAG_DRSNAME] = cmdParser.getOptionAs<std::string>("drsname");
        // Exif.Image.AsShotProfileName
        if (cmdParser.hasOption("drsprofilename"))
            result[darkroom::gJPEG_TAG_DRSPROFILENAME] = cmdParser.getOptionAs<std::string>("drsprofilename");
        if (cmdParser.hasOption("apptitlename"))
            result[darkroom::gJPEG_TAG_APPTITLENAME] = cmdParser.getOptionAs<std::string>("apptitlename");
        if (cmdParser.hasOption("appcmsid"))
            result[darkroom::gJPEG_TAG_APPCMSID] = cmdParser.getOptionAs<std::string>("appcmsid");
        if (cmdParser.hasOption("appshortname"))
            result[darkroom::gJPEG_TAG_APPSHORTNAME] = cmdParser.getOptionAs<std::string>("appshortname");
        if (cmdParser.hasOption("activefilters"))
            result[darkroom::gJPEG_TAG_ACTIVEFILTERS] = cmdParser.getOptionAs<std::string>("activefilters");
        return result;
    }

    std::unordered_map<std::string, std::string> generatePngTagTable(darkroom::CmdLineParser& cmdParser)
    {
        std::unordered_map<std::string, std::string> result;
        if (cmdParser.hasOption("make"))
            result["Source"] = cmdParser.getOptionAs<std::string>("make");
        if (cmdParser.hasOption("model"))
            result["Model"] = cmdParser.getOptionAs<std::string>("model");
        if (cmdParser.hasOption("software"))
            result["Software"] = cmdParser.getOptionAs<std::string>("software");
        if (cmdParser.hasOption("description"))
            result["Description"] = cmdParser.getOptionAs<std::string>("description");
        if (cmdParser.hasOption("drsname"))
            result["DRSName"] = cmdParser.getOptionAs<std::string>("drsname");
        if (cmdParser.hasOption("drsprofilename"))
            result["DRSProfileName"] = cmdParser.getOptionAs<std::string>("drsprofilename");
        if (cmdParser.hasOption("apptitlename"))
            result["AppTitleName"] = cmdParser.getOptionAs<std::string>("apptitlename");
        if (cmdParser.hasOption("appcmsid"))
            result["AppCMSID"] = cmdParser.getOptionAs<std::string>("appcmsid");
        if (cmdParser.hasOption("appshortname"))
            result["AppShortName"] = cmdParser.getOptionAs<std::string>("appshortname");
        if (cmdParser.hasOption("activefilters"))
            result["ActiveFilters"] = cmdParser.getOptionAs<std::string>("activefilters");
        if (cmdParser.hasOption("type"))
        {
            const auto type = cmdParser.getOptionAs<std::string>("type");
            result["Comment"] = type;
            result["MakerNote"] = type;
        }
        return result;
    }

    std::vector<FrequencyTransferTileParameters> callwlateTileParameters(uint32_t regularSizeW, uint32_t regularSizeH,
        uint32_t highresSizeW, uint32_t highresSizeH, 
        uint32_t highresTileOffsetW, uint32_t highresTileOffsetH,
        uint32_t highresTileSizeW, uint32_t highresTileSizeH,
        uint32_t& maxPaddingW, uint32_t& maxPaddingH,
        uint32_t& maxOffsetW, uint32_t& maxOffsetH)
    {
        std::vector<FrequencyTransferTileParameters> result;
        uint32_t highresTilePaddingOffsetW = 0u, highresTilePaddingOffsetH = 0u,
                highresTilePaddingSizeW = 0u, highresTilePaddingSizeH = 0u;

        if (highresTileSizeH > highresTileSizeW)
        {
            const auto highresSubtileCount = uint32_t(std::ceil(float(highresTileSizeH) / float(highresTileSizeW)));
            const auto highresSubtileHeight = highresTileSizeH / highresSubtileCount;
            for (auto i = 0u; i < highresSubtileCount; ++i)
            {
                const auto highresTileOffset = highresTileOffsetH + i * highresSubtileHeight;
                // last subtile can be slightly higher than others, because of the integer divide rounding in the previous line
                const auto highresTileHeight = i == (highresSubtileCount - 1u) ? (highresTileSizeH - i * highresSubtileHeight) : highresSubtileHeight;
                result.push_back({ highresTileOffsetW, highresTileOffset, highresTileSizeW, highresTileHeight });

                uint32_t highresTilePaddingOffsetW = 0u, highresTilePaddingOffsetH = 0u,
                    highresTilePaddingSizeW = 0u, highresTilePaddingSizeH = 0u;

                darkroom::getFrequencyTransferTilePaddings(
                    regularSizeW, regularSizeH,
                    highresSizeW, highresSizeH,
                    highresTileOffsetW, highresTileOffset,
                    highresTileSizeW, highresTileHeight,
                    &highresTilePaddingOffsetW,
                    &highresTilePaddingOffsetH,
                    &highresTilePaddingSizeW,
                    &highresTilePaddingSizeH);

                maxPaddingW = max(maxPaddingW, highresTilePaddingSizeW);
                maxPaddingH = max(maxPaddingH, highresTilePaddingSizeH);
                maxOffsetW = max(maxOffsetW, highresTilePaddingOffsetW);
                maxOffsetH = max(maxOffsetH, highresTilePaddingOffsetH);
            }
        }
        // horizontal strip
        else
        {
            const auto highresSubtileCount = uint32_t(std::ceil(float(highresTileSizeW) / float(highresTileSizeH)));
            const auto highresSubtileWidth = highresTileSizeW / highresSubtileCount;
            for (auto i = 0u; i < highresSubtileCount; ++i)
            {
                const auto highresTileOffset = highresTileOffsetW + i * highresSubtileWidth;
                // last subtile can be slightly wider than others, because of the integer divide rounding in the previous line
                const auto highresTileWidth = i == (highresSubtileCount - 1u) ? (highresTileSizeW - i * highresSubtileWidth) : highresSubtileWidth;
                result.push_back({ highresTileOffset, highresTileOffsetH, highresTileWidth, highresTileSizeH });
                
                uint32_t highresTilePaddingOffsetW = 0u, highresTilePaddingOffsetH = 0u,
                    highresTilePaddingSizeW = 0u, highresTilePaddingSizeH = 0u;

                darkroom::getFrequencyTransferTilePaddings(
                    regularSizeW, regularSizeH,
                    highresSizeW, highresSizeH,
                    highresTileOffset, highresTileOffsetH,
                    highresTileWidth, highresTileSizeH,
                    &highresTilePaddingOffsetW,
                    &highresTilePaddingOffsetH,
                    &highresTilePaddingSizeW,
                    &highresTilePaddingSizeH);

                maxPaddingW = max(maxPaddingW, highresTilePaddingSizeW);
                maxPaddingH = max(maxPaddingH, highresTilePaddingSizeH);
                maxOffsetW = max(maxOffsetW, highresTilePaddingOffsetW);
                maxOffsetH = max(maxOffsetH, highresTilePaddingOffsetH);
            }
        }
        return result;
    }

    template<typename T> 
    void processFrequencyTransferForOblongStrip(const std::vector<FrequencyTransferTileParameters>& tileParameters,
        darkroom::FFTPersistentData * pPersistentData, const T * regular, uint32_t regularSizeW, uint32_t regularSizeH,
        T * highres, uint32_t highresSizeW, uint32_t highresSizeH, T * highresTileFixed, float alpha)
    {
        for (const auto& tileParams : tileParameters)
        {
            darkroom::processFrequencyTransfer(pPersistentData, regular, regularSizeW, regularSizeH, highres, highresSizeW, highresSizeH,
                tileParams.paddingOffsetW, tileParams.paddingOffsetH, tileParams.paddingSizeW, tileParams.paddingSizeH,
                highresTileFixed, alpha);
        }
    }
}

int main(int argc, char* argv[])
{
    darkroom::Error retcode = darkroom::Error::kSuccess;
    darkroom::CmdLineParser cmdParser;
    cmdParser.addSingularOption("conserve", "Run in memory conservative mode, which might slow the process down, but requires less memory");
    cmdParser.addSingularOption("progress", "Print 100 dots ('.' + \\n), each representing 1% of work done");
    cmdParser.addSingularOption("version", "Print tool version");
    cmdParser.addSingularOption("lossless", "Force lossless image output");
    cmdParser.addRegularOption("freq-transfer-input", "Regular-sized image to transfer frequency spectrum from");
    cmdParser.addRegularOption("freq-transfer-alpha", "Frequency transfer interpolation coefficient\n\t\t\t\t0.0 - most aggressive\n\t\t\t\t1.0 - least aggressive");
    cmdParser.addRegularOption("output", "Target file path without extension (the tool will add jpg or png automatically)");
    cmdParser.addRegularOption("threads", "How many threads the tool is allowed to use");
    cmdParser.addRegularOption("make", "EXIF tag 'Make'");
    cmdParser.addRegularOption("model", "EXIF tag 'Model' and 'UniqueModel'");
    cmdParser.addRegularOption("drsname", "EXIF tag for DRSName");
    cmdParser.addRegularOption("drsprofilename", "EXIF tag for DRSProfileName");
    cmdParser.addRegularOption("apptitlename", "EXIF tag for AppTitleName");
    cmdParser.addRegularOption("appcmsid", "EXIF tag for AppCMSID");
    cmdParser.addRegularOption("appshortname", "EXIF tag for AppShortName");
    cmdParser.addRegularOption("activefilters", "EXIF tag for ActiveFilters");
    cmdParser.addRegularOption("software", "EXIF tag 'Software'");
    cmdParser.addRegularOption("description", "EXIF tag 'ImageDescription'");
    cmdParser.addRegularOption("type", "EXIF tag 'MakerNote'");
    const bool optionsParsed = cmdParser.parse(argc, argv);

    const bool showProgress = cmdParser.hasOption("progress");

    if (cmdParser.hasOption("version"))
    {
        std::cout << "HighresBlender version: " << getVersion() << std::endl;
        return 0;
    }

    unsigned int threadCount = 0;
    if (cmdParser.hasOption("threads"))
        threadCount = cmdParser.getOptionAs<unsigned int>("threads");
    
    workerCount = threadCount == 0 ? std::thread::hardware_conlwrrency() : threadCount;

    std::string outputFilename = cmdParser.getOptionAs<std::string>("output");
    if (outputFilename.empty())
        outputFilename = "result";

    const bool conserveMemory = cmdParser.hasOption("conserve");
    tileNames = windows::discoverTiles(L"highres-*.*");
    if (tileNames.empty())
    {
        cout << "Usage:" << endl;
        cmdParser.printUsage();
        return 0;
    }

    std::wstring regularImage;

    if (cmdParser.hasOption("freq-transfer-input"))
    {
        // regular sized image filename to transfer the frequency spectrum from
        regularImage = cmdParser.getOptionAs<std::wstring>("freq-transfer-input");
    }

    const bool inputIsExr = windows::findExtension(tileNames[0]) == L".exr";
    const unsigned int tileCount = static_cast<unsigned int>(tileNames.size());

    if (optionsParsed && darkroom::isHighresAllowed(tileCount) == darkroom::Error::kSuccess)
    {
        multiplier = static_cast<unsigned int>(sqrtf(float(tileCount)) / 2.0f + 1.0f);
        // the tool will try blend at least 9 images (3x3 images, 50% overlap, 
        // which cover the same area as 2x2 images with no overlap, which corresponds to a minimal highresMultiplier setting of 2)
        if (inputIsExr)
        {
            tilesExr.resize(tileCount);
            tileExrPtrs.resize(tileCount, nullptr);
            tilesInRow = static_cast<unsigned int>(sqrtf(float(tileExrPtrs.size())));
        }
        else
        {
            tilesBmp.resize(tileCount);
            tileBmpPtrs.resize(tileCount, nullptr);
            tilesInRow = static_cast<unsigned int>(sqrtf(float(tileBmpPtrs.size())));
        }

        const std::wstring highresTileName = darkroom::getWstrFromUtf8(darkroom::HighresTileName);
        for (auto i = 0u; i < tilesInRow; ++i)
        {
            for (auto j = 0u; j < tilesInRow; ++j)
            {
                std::wstringstream in;
                in << highresTileName << i << L"-" << j << (inputIsExr ? L".exr" : L".bmp");
                tileNames[i * tilesInRow + j] = in.str();
            }
        }
        const unsigned int tilesPreload = conserveMemory ? 0 : tileCount;

        // load first two rows
        if (inputIsExr)
        {
            std::unordered_map<std::string, std::string> tags;
            tilesExr[0] = darkroom::loadExr(darkroom::getUtf8FromWstr(tileNames[0]), tileWidth, tileHeight, tags);
            tileExrPtrs[0] = &tilesExr[0][0];
        }
        else
        {
            tilesBmp[0] = darkroom::loadBmp(darkroom::getUtf8FromWstr(tileNames[0]), tileWidth, tileHeight);
            tileBmpPtrs[0] = &tilesBmp[0][0];
        }
        tileHeight &= ~1u; // a bit of cheating for odd tile height (for now)

        if (tilesPreload > 0)
        {
            for (auto i = 1u; i < tilesPreload; ++i)
            {
                unsigned int w, h;
                if (inputIsExr)
                {
                    std::unordered_map<std::string, std::string> tags;
                    tilesExr[i] = darkroom::loadExr(darkroom::getUtf8FromWstr(tileNames[i]), w, h, tags);
                    h &= ~1u; // a bit of cheating for odd tile height (for now)
                    signalFatal(tileWidth != w || tileHeight != h, "Tile size mismatch");
                    tileExrPtrs[i] = &tilesExr[i][0];
                }
                else
                {
                    tilesBmp[i] = darkroom::loadBmp(darkroom::getUtf8FromWstr(tileNames[i]), w, h);
                    h &= ~1u; // a bit of cheating for odd tile height (for now)
                    signalFatal(tileWidth != w || tileHeight != h, "Tile size mismatch");
                    tileBmpPtrs[i] = &tilesBmp[i][0];
                }
            }
        }

        const unsigned int wr = static_cast<unsigned int>(tileWidth * multiplier);
        const unsigned int hr = static_cast<unsigned int>(tileHeight * multiplier);

        // 65500 is libjpeg-turbo limit for a maximum amount of pixel along any direction
        // determine the output filetype here
        OutputFiletype outputType;
        if (!inputIsExr)
        {
            if (wr < 65500 && !cmdParser.hasOption("lossless"))
            {
                outputType = OutputFiletype::JPEG;
                outputFilename += ".jpg";
            }
            else if ((wr < 65500 && cmdParser.hasOption("lossless")) || (wr >= 65500))
            {
                outputType = OutputFiletype::PNG;
                outputFilename += ".png";
            }
        }
        else
        {
            outputType = OutputFiletype::EXR;
            outputFilename += ".exr";
        }

        auto progressCallbackImpl = [](float p) 
        {
            static float progressReportedSoFar = 0.0f;
            const float percentReadyFromLastTime = p - progressReportedSoFar;
            if (percentReadyFromLastTime > 0.01f)
            {
                const uint32_t dotsToPrint = uint32_t(percentReadyFromLastTime / 0.01f);
                progressReportedSoFar += dotsToPrint * 0.01f;
                for (auto x = 0u; x < dotsToPrint; ++x)
                    std::cout << "." << std::endl;
            }			 
        };

        const darkroom::ProgressCallback progressCallback = showProgress ? darkroom::ProgressCallback(progressCallbackImpl) : nullptr;

        if (conserveMemory)
        {
            // determine system allocation granularity
            SYSTEM_INFO sysInfo;
            GetSystemInfo(&sysInfo);
            gAllocationGranularity = sysInfo.dwPageSize * 16;

            // map bmp file and generate/copy bmp header
            size_t elementSizeInBytes = sizeof(unsigned char);
            if (inputIsExr)
                elementSizeInBytes = sizeof(float);
            uint32_t bytesPerRow = static_cast<uint32_t>(3 * wr * elementSizeInBytes);
            const uint64_t totalFileSize = uint64_t(uint64_t(bytesPerRow) * hr);
            DWORD high = static_cast<DWORD>((totalFileSize >> 32) & 0xFFFFFFFFul);
            DWORD low = static_cast<DWORD>(totalFileSize & 0xFFFFFFFFul);
            HANDLE hf = CreateFile(L"backing.raw", GENERIC_READ | GENERIC_WRITE, NULL, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
            gFileMapping = CreateFileMapping(hf, NULL, PAGE_READWRITE, high, low, NULL);
            HANDLE hfFreqTransfer = nullptr;
            // allocate additional file mapping in case we're going to transfer frequencies
            if (!regularImage.empty())
            {
                hfFreqTransfer = CreateFile(L"backing_ft.raw", GENERIC_READ | GENERIC_WRITE, NULL, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
                gFileMappingFreqTransfer = CreateFileMapping(hfFreqTransfer, NULL, PAGE_READWRITE, high, low, NULL);
            }

            auto loadTilesBmp = [](const std::vector<size_t>& tileNumbers)
            {
                for (auto i : tileNumbers)
                {
                    // first tile is always loaded
                    if (i == 0)
                        continue;
                    unsigned int w = 0, h = 0;
                    tilesBmp[i] = darkroom::loadBmp(darkroom::getUtf8FromWstr(tileNames[i]), w, h);
                    h &= ~1u; // a bit of cheating for odd tile height (for now)
                    signalFatal(tileWidth != w || tileHeight != h, "Tile size mismatch");
                    tileBmpPtrs[i] = &tilesBmp[i][0];
                }
            };

            auto loadTilesExr = [](const std::vector<size_t>& tileNumbers)
            {
                for (auto i : tileNumbers)
                {
                    // first tile is always loaded
                    if (i == 0)
                        continue;
                    unsigned int w = 0, h = 0;
                    std::unordered_map<std::string, std::string> tags;
                    tilesExr[i] = darkroom::loadExr(darkroom::getUtf8FromWstr(tileNames[i]), w, h, tags);
                    h &= ~1u; // a bit of cheating for odd tile height (for now)
                    signalFatal(tileWidth != w || tileHeight != h, "Tile size mismatch");
                    tileExrPtrs[i] = &tilesExr[i][0];
                }
            };

            auto unloadTilesBmp = [](const std::vector<size_t>& tileNumbers)
            {
                for (auto i : tileNumbers)
                {
                    tilesBmp[i] = std::vector<unsigned char>();
                    tileBmpPtrs[i] = nullptr;
                }
            };

            auto unloadTilesExr = [](const std::vector<size_t>& tileNumbers)
            {
                for (auto i : tileNumbers)
                {
                    tilesExr[i] = std::vector<float>();
                    tileExrPtrs[i] = nullptr;
                }
            };

            auto remapCallbackBmp = [](uint64_t start, uint64_t end, void* lwrrentMapping) -> unsigned char*
            {
                if (lwrrentMapping != nullptr)
                {
                    const auto roundedStart = reinterpret_cast<void*>(reinterpret_cast<uint64_t>(lwrrentMapping) & ~(gAllocationGranularity - 1ull));
                    UnmapViewOfFile(roundedStart);
                }

                const uint64_t roundedStart = start & ~(gAllocationGranularity - 1ull);
                const DWORD high = DWORD((roundedStart >> 32) & 0xFFFFFFFFul);
                const DWORD low = DWORD(roundedStart & 0xFFFFFFFFul);

                unsigned char* fileMappingPtr = static_cast<unsigned char*>(MapViewOfFile(gFileMapping, FILE_MAP_ALL_ACCESS, high, low, SIZE_T(end - roundedStart)));
                return fileMappingPtr + (start - roundedStart);
            };

            auto remapCallbackExr = [](uint64_t start, uint64_t end, void* lwrrentMapping) -> float*
            {
                if (lwrrentMapping != nullptr)
                {
                    const auto roundedStart = reinterpret_cast<void*>(reinterpret_cast<uint64_t>(lwrrentMapping) & ~(gAllocationGranularity - 1ull));
                    UnmapViewOfFile(roundedStart);
                }

                const uint64_t roundedStart = start & ~(gAllocationGranularity - 1ull);
                const DWORD high = DWORD((roundedStart >> 32) & 0xFFFFFFFFul);
                const DWORD low = DWORD(roundedStart & 0xFFFFFFFFul);

                unsigned char* fileMappingPtr = static_cast<unsigned char*>(MapViewOfFile(gFileMapping, FILE_MAP_ALL_ACCESS, high, low, SIZE_T(end - roundedStart)));
                return reinterpret_cast<float*>(fileMappingPtr + (start - roundedStart));
            };

            {
                if (inputIsExr)
                {
                    retcode = darkroom::blendHighres<float>(&tileExrPtrs[0],
                        static_cast<unsigned int>(tileExrPtrs.size()),
                        tileWidth, tileHeight, nullptr, bytesPerRow,
                        loadTilesExr, unloadTilesExr, remapCallbackExr, progressCallback, true,
                        darkroom::BufferFormat::RGB32, threadCount);
                }
                else
                {
                    retcode = darkroom::blendHighres<unsigned char>(&tileBmpPtrs[0],
                        static_cast<unsigned int>(tileBmpPtrs.size()),
                        tileWidth, tileHeight, nullptr, bytesPerRow,
                        loadTilesBmp, unloadTilesBmp, remapCallbackBmp, progressCallback, true,
                        darkroom::BufferFormat::BGR8, threadCount);
                }
            }

            // unload all tiles
            tilesBmp.clear();

            {
                // for PNG we can encode in portions, so use 0.25Gb chunks
                const uint64_t windowSize = 1024ull * 1024ull * 256ull; // 256Mb
                if (inputIsExr)
                    memorymapping::imagePitch = wr * 3 * sizeof(float);
                else
                    memorymapping::imagePitch = wr * 3;
                memorymapping::rawDataSize = memorymapping::imagePitch * hr;
                memorymapping::rowsPerWindow = windowSize / memorymapping::imagePitch;
                memorymapping::imageHeight = hr;

                if (!regularImage.empty())
                {
                    // load regular sized image to transfer the frequency spectrum from
                    const auto ext = windows::findExtension(regularImage);
                    if (ext == L".exr")
                        frequency_transfer::regular32bit = darkroom::loadExr(regularImage, frequency_transfer::regularWidth, frequency_transfer::regularHeight, std::unordered_map<std::string, std::string>());
                    else if (ext == L".bmp")
                        frequency_transfer::regular8bit = darkroom::loadBmp(regularImage, frequency_transfer::regularWidth, frequency_transfer::regularHeight, darkroom::BufferFormat::RGB8);
                    else if (ext == L".png")
                        frequency_transfer::regular8bit = darkroom::loadPng(regularImage, frequency_transfer::regularWidth, frequency_transfer::regularHeight, darkroom::BufferFormat::RGB8, std::unordered_map<std::string, std::string>());
                    else if (ext == L".jpg")
                        frequency_transfer::regular8bit = darkroom::loadJpeg(regularImage, frequency_transfer::regularWidth, frequency_transfer::regularHeight,
                            darkroom::BufferFormat::RGB8, std::unordered_map<uint16_t, std::string>(), std::vector<char>());
                
                    // only initialize frequency transfer utilities in case we have regular image loaded
                    if (!frequency_transfer::regular8bit.empty() || !frequency_transfer::regular32bit.empty())
                    {
                        // initialize frequency transfer for each thread
                        frequency_transfer::persistentData.resize(workerCount);
                        for (auto& data : frequency_transfer::persistentData)
                            darkroom::initFrequencyTransferProcessing(&data);

                        if (cmdParser.hasOption("freq-transfer-alpha"))
                            frequency_transfer::alpha = cmdParser.getOptionAs<float>("freq-transfer-alpha");
                    }
                }
                    
                auto remapCallback = [](unsigned int& rowsMapped) -> const void*
                {
                    if (memorymapping::lastMappedAddress)
                    {
                        UnmapViewOfFile(memorymapping::lastMappedAddress);
                        memorymapping::lastMappedAddress = nullptr;
                    }

                    if (memorymapping::lastMappedAddressFreqTransfer)
                    {
                        UnmapViewOfFile(memorymapping::lastMappedAddressFreqTransfer);
                        memorymapping::lastMappedAddressFreqTransfer = nullptr;
                    }

                    if (memorymapping::lwrrentRow >= memorymapping::imageHeight)
                        return nullptr;

                    uint32_t highresWidth = tileWidth * multiplier,
                        highresHeight = tileHeight * multiplier,
                        highresTileWidth = highresWidth / workerCount;

                    std::vector<FrequencyTransferTileParameters> tileParameters;

                    uint32_t paddingW = 0u, paddingH = 0u, offsetW = 0u, offsetH = 0u;

                    if (!frequency_transfer::persistentData.empty())
                    {
                        for (uint32_t t = 0; t < workerCount; ++t)
                        {
                            const auto highresTileOffset = t * highresTileWidth;
                            // last tile can be wider (integer divide can round width down)
                            if (t == workerCount - 1)
                                highresTileWidth = max(highresWidth - highresTileOffset, highresTileWidth);

                            auto rows = uint32_t(memorymapping::rowsPerWindow);
                            if (uint32_t(memorymapping::lwrrentRow) + rows > memorymapping::imageHeight)
                                rows = uint32_t(memorymapping::imageHeight) - uint32_t(memorymapping::lwrrentRow);

                            uint32_t tilePaddingW = 0u, tilePaddingH = 0u, tileOffsetW = 0, tileOffsetH = 0;

                            const auto params = callwlateTileParameters(frequency_transfer::regularWidth, frequency_transfer::regularHeight,
                                highresWidth, highresHeight, highresTileOffset, uint32_t(memorymapping::lwrrentRow), highresTileWidth, rows,
                                tilePaddingW, tilePaddingH, tileOffsetW, tileOffsetH);

                            paddingW = max(paddingW, tilePaddingW);
                            paddingH = max(paddingH, tilePaddingH);
                            offsetW = max(offsetW, tileOffsetW);
                            offsetH = max(offsetH, tileOffsetH);

                            tileParameters.insert(tileParameters.end(), params.cbegin(), params.cend());
                        }
                    }

                    const auto paddingAbove = offsetH > memorymapping::lwrrentRow ? memorymapping::lwrrentRow : offsetH;

                    const auto oldLwrrentRow = memorymapping::lwrrentRow;
                    const uint64_t start = (memorymapping::lwrrentRow - paddingAbove) * memorymapping::imagePitch;
                    const uint64_t endRow = min(memorymapping::lwrrentRow + memorymapping::rowsPerWindow, memorymapping::imageHeight);
                    rowsMapped = static_cast<unsigned int>(endRow - memorymapping::lwrrentRow);

                    if (rowsMapped == 0u)
                        return nullptr;

                    uint64_t roundedStart = start & ~(gAllocationGranularity - 1ull);
                    uint64_t end = endRow * memorymapping::imagePitch;
                    if (endRow != memorymapping::imageHeight)
                        end = min(end + paddingH * memorymapping::imagePitch, memorymapping::imageHeight * memorymapping::imagePitch);
                    const auto startRow = uint32_t(memorymapping::lwrrentRow);
                    memorymapping::lwrrentRow += rowsMapped;
                    const DWORD high = DWORD((roundedStart >> 32) & 0xFFFFFFFFul);
                    const DWORD low = DWORD(roundedStart & 0xFFFFFFFFul);

                    memorymapping::lastMappedAddress = MapViewOfFile(gFileMapping, FILE_MAP_READ, high, low, SIZE_T(end - roundedStart));
                    auto* data = static_cast<unsigned char*>(memorymapping::lastMappedAddress) + start - roundedStart + paddingAbove * memorymapping::imagePitch;
                    auto* highresStart = data - memorymapping::imagePitch * startRow;
                    unsigned char* dataFreqTransfer = nullptr;

                    // before unmapping the block, first perform frequency transfer if needed
                    // divide rectangular area in equal pieces (one per thread) and start up worker threads
                    if (!frequency_transfer::persistentData.empty())
                    {
                        memorymapping::lastMappedAddressFreqTransfer = MapViewOfFile(gFileMappingFreqTransfer, FILE_MAP_ALL_ACCESS, high, low, SIZE_T(end - roundedStart));
                        dataFreqTransfer = static_cast<unsigned char*>(memorymapping::lastMappedAddressFreqTransfer) + start - roundedStart + paddingAbove * memorymapping::imagePitch;
                        auto* highresOutput = dataFreqTransfer - memorymapping::imagePitch * startRow;

                        std::vector<std::thread> workers;
                        // 8 bit per channel case
                        if (!frequency_transfer::regular8bit.empty())
                        {
                            for (uint32_t t = 0; t < workerCount; ++t)
                            {
                                const auto highresTileOffset = t * highresTileWidth;
                                // last tile can be wider (integer divide can round width down)
                                if (t == workerCount - 1)
                                    highresTileWidth = highresWidth - highresTileOffset;

                                auto rows = uint32_t(memorymapping::rowsPerWindow);
                                if (uint32_t(oldLwrrentRow) + rows > memorymapping::imageHeight)
                                    rows = uint32_t(memorymapping::imageHeight) - uint32_t(oldLwrrentRow);
                                tileParameters = callwlateTileParameters(frequency_transfer::regularWidth, frequency_transfer::regularHeight,
                                    highresWidth, highresHeight, highresTileOffset, uint32_t(oldLwrrentRow), highresTileWidth, rows,
                                    paddingW, paddingH, offsetW, offsetH);

                                workers.push_back(std::thread(
                                    processFrequencyTransferForOblongStrip<unsigned char>,
                                    tileParameters,
                                    &frequency_transfer::persistentData[t],
                                    const_cast<const unsigned char*>(frequency_transfer::regular8bit.data()),
                                    frequency_transfer::regularWidth, frequency_transfer::regularHeight,
                                    highresStart, highresWidth, highresHeight,
                                    highresOutput, frequency_transfer::alpha));
                            }
                        }
                        // 32 bit per channel case (EXR)
                        else
                        {
                            for (uint32_t t = 0; t < workerCount; ++t)
                            {
                                const auto highresTileOffset = t * highresTileWidth;
                                // last tile can be wider (integer divide can round width down)
                                if (t == workerCount - 1)
                                    highresTileWidth = highresWidth - highresTileOffset;
                                
                                auto rows = uint32_t(memorymapping::rowsPerWindow);
                                if (uint32_t(oldLwrrentRow) + rows > memorymapping::imageHeight)
                                    rows = uint32_t(memorymapping::imageHeight) - uint32_t(oldLwrrentRow);
                                tileParameters = callwlateTileParameters(frequency_transfer::regularWidth, frequency_transfer::regularHeight,
                                    highresWidth, highresHeight, highresTileOffset, uint32_t(oldLwrrentRow), highresTileWidth, rows,
                                    paddingW, paddingH, offsetW, offsetH);

                                workers.push_back(std::thread(
                                    processFrequencyTransferForOblongStrip<float>,
                                    tileParameters,
                                    &frequency_transfer::persistentData[t],
                                    const_cast<const float*>(frequency_transfer::regular32bit.data()),
                                    frequency_transfer::regularWidth, frequency_transfer::regularHeight,
                                    reinterpret_cast<float*>(highresStart), highresWidth, highresHeight,
                                    reinterpret_cast<float*>(highresOutput), frequency_transfer::alpha));
                            }
                        }

                        for (auto& thread : workers)
                            thread.join();
                    }

                    return dataFreqTransfer ? dataFreqTransfer : data;
                };

                if (outputType == OutputFiletype::JPEG)
                    darkroom::saveJpeg(nullptr, remapCallback, outputFilename, wr, hr, darkroom::BufferFormat::RGB8, generateExifTable(cmdParser));
                else if (outputType == OutputFiletype::PNG)
                    darkroom::savePng(nullptr, remapCallback, outputFilename, wr, hr, darkroom::BufferFormat::RGB8, generatePngTagTable(cmdParser), 2, threadCount);
                else if (outputType == OutputFiletype::EXR)
                    darkroom::saveExr(nullptr, remapCallback, outputFilename, wr, hr, darkroom::BufferFormat::RGB32, generatePngTagTable(cmdParser));

                if (memorymapping::lastMappedAddress)
                    UnmapViewOfFile(memorymapping::lastMappedAddress);
                if (memorymapping::lastMappedAddressFreqTransfer)
                    UnmapViewOfFile(memorymapping::lastMappedAddressFreqTransfer);

                // cleanup memory allocated by frequency transfer functions
                for (auto& data : frequency_transfer::persistentData)
                    darkroom::deinitFrequencyTransferProcessing(&data);
            }

            CloseHandle(gFileMapping);
            CloseHandle(hf);

            if (gFileMappingFreqTransfer)
            {
                CloseHandle(gFileMappingFreqTransfer);
                CloseHandle(hfFreqTransfer);
            }

            if (retcode != darkroom::Error::kSuccess)
                std::cout << "HighresBlender failed with (" << darkroom::errToString(retcode) << ")" << std::endl;
        }
        else
        {
            std::vector<unsigned char> resultUchar;
            std::vector<float> resultFloat;
            if (inputIsExr)
            {
                const uint32_t bytesPerRow = static_cast<size_t>(3 * wr * sizeof(float));
                resultFloat.resize(size_t(wr) * size_t(hr) * 3);
                retcode = darkroom::blendHighres<float>(&tileExrPtrs[0],
                    static_cast<unsigned int>(tileExrPtrs.size()),
                    tileWidth, tileHeight, &resultFloat[0], bytesPerRow,
                    nullptr, nullptr, nullptr, progressCallback, false,
                    darkroom::BufferFormat::RGB32, threadCount);
            }
            else
            {
                const uint32_t bytesPerRow = static_cast<size_t>(3 * wr);
                resultUchar.resize(size_t(wr) * size_t(hr) * 3);
                retcode = darkroom::blendHighres<unsigned char>(&tileBmpPtrs[0],
                    static_cast<unsigned int>(tileBmpPtrs.size()),
                    tileWidth, tileHeight, &resultUchar[0], bytesPerRow,
                    nullptr, nullptr, nullptr, progressCallback, false,
                    darkroom::BufferFormat::BGR8, threadCount);
            }
            if (retcode == darkroom::Error::kSuccess)
            {
                if (!regularImage.empty())
                {
                    // load regular sized image to transfer the frequency spectrum from
                    const auto ext = windows::findExtension(regularImage);
                    if (ext == L".exr")
                        frequency_transfer::regular32bit = darkroom::loadExr(regularImage, frequency_transfer::regularWidth, frequency_transfer::regularHeight, std::unordered_map<std::string, std::string>());
                    else if (ext == L".bmp")
                        frequency_transfer::regular8bit = darkroom::loadBmp(regularImage, frequency_transfer::regularWidth, frequency_transfer::regularHeight, darkroom::BufferFormat::RGB8);
                    else if (ext == L".png")
                        frequency_transfer::regular8bit = darkroom::loadPng(regularImage, frequency_transfer::regularWidth, frequency_transfer::regularHeight, darkroom::BufferFormat::RGB8, std::unordered_map<std::string, std::string>());
                    else if (ext == L".jpg")
                        frequency_transfer::regular8bit = darkroom::loadJpeg(regularImage, frequency_transfer::regularWidth, frequency_transfer::regularHeight,
                            darkroom::BufferFormat::RGB8, std::unordered_map<uint16_t, std::string>(), std::vector<char>());

                    workerCount = threadCount == 0 ? std::thread::hardware_conlwrrency() : threadCount;

                    // only initialize frequency transfer utilities in case we have regular image loaded
                    if (!frequency_transfer::regular8bit.empty() || !frequency_transfer::regular32bit.empty())
                    {
                        // initialize frequency transfer for each thread
                        frequency_transfer::persistentData.resize(workerCount);
                        for (auto& data : frequency_transfer::persistentData)
                            darkroom::initFrequencyTransferProcessing(&data);

                        if (cmdParser.hasOption("freq-transfer-alpha"))
                            frequency_transfer::alpha = cmdParser.getOptionAs<float>("freq-transfer-alpha");
                    }
                }

                // before unmapping the block, first perform frequency transfer if needed
                // divide rectangular area in equal pieces (one per thread) and start up worker threads
                if (!frequency_transfer::persistentData.empty())
                {
                    std::vector<unsigned char> resultUcharCopy = resultUchar;
                    std::vector<float> resultFloatCopy = resultFloat;
                    const auto highresWidth = tileWidth * multiplier;
                    const auto highresHeight = tileHeight * multiplier;
                    auto highresTileWidth = highresWidth / workerCount;

                    std::vector<std::thread> workers;
                    // 8 bit per channel case
                    if (!frequency_transfer::regular8bit.empty())
                    {
                        for (uint32_t t = 0; t < workerCount; ++t)
                        {
                            const auto highresTileOffset = t * highresTileWidth;
                            // last tile can be wider (integer divide can round width down)
                            if (t == workerCount - 1)
                                highresTileWidth = highresWidth - highresTileOffset;

                            uint32_t paddingW = 0u, paddingH = 0u, offsetW = 0u, offsetH = 0u;
                            const auto tileParameters = callwlateTileParameters(frequency_transfer::regularWidth, frequency_transfer::regularHeight,
                                highresWidth, highresHeight, highresTileOffset, 0u, highresTileWidth, hr, paddingW, paddingH, offsetW, offsetH);

                            workers.push_back(std::thread(
                                processFrequencyTransferForOblongStrip<unsigned char>,
                                tileParameters,
                                &frequency_transfer::persistentData[t],
                                const_cast<const unsigned char*>(frequency_transfer::regular8bit.data()),
                                frequency_transfer::regularWidth, frequency_transfer::regularHeight,
                                resultUcharCopy.data(), highresWidth, highresHeight,
                                resultUchar.data(), frequency_transfer::alpha));
                        }
                    }
                    // 32 bit per channel case (EXR)
                    else
                    {
                        for (uint32_t t = 0; t < workerCount; ++t)
                        {
                            const auto highresTileOffset = t * highresTileWidth;
                            // last tile can be wider (integer divide can round width down)
                            if (t == workerCount - 1)
                                highresTileWidth = highresWidth - highresTileOffset;

                            uint32_t paddingW = 0u, paddingH = 0u, offsetW = 0u, offsetH = 0u;
                            const auto tileParameters = callwlateTileParameters(frequency_transfer::regularWidth, frequency_transfer::regularHeight,
                                highresWidth, highresHeight, highresTileOffset, 0u, highresTileWidth, hr, paddingW, paddingH, offsetW, offsetH);

                            workers.push_back(std::thread(
                                processFrequencyTransferForOblongStrip<float>,
                                tileParameters,
                                &frequency_transfer::persistentData[t],
                                const_cast<const float*>(frequency_transfer::regular32bit.data()),
                                frequency_transfer::regularWidth, frequency_transfer::regularHeight,
                                resultFloatCopy.data(), tileWidth * multiplier, tileHeight * multiplier,
                                resultFloat.data(), frequency_transfer::alpha));
                        }
                    }

                    for (auto& thread : workers)
                        thread.join();
                    // cleanup memory allocated by frequency transfer functions
                    for (auto& data : frequency_transfer::persistentData)
                        darkroom::deinitFrequencyTransferProcessing(&data);
                }


                if (outputType == OutputFiletype::JPEG)
                    darkroom::saveJpeg(resultUchar.data(), nullptr, outputFilename, wr, hr, darkroom::BufferFormat::RGB8, generateExifTable(cmdParser));
                else if (outputType == OutputFiletype::PNG)
                    darkroom::savePng(resultUchar.data(), nullptr, outputFilename, wr, hr, darkroom::BufferFormat::RGB8, generatePngTagTable(cmdParser), 2, threadCount);
                else if (outputType == OutputFiletype::EXR)
                    darkroom::saveExr(resultFloat.data(), nullptr, outputFilename, wr, hr, darkroom::BufferFormat::RGB32, generatePngTagTable(cmdParser));
            }
            else
                std::cout << "HighresBlender failed with (" << darkroom::errToString(retcode) << ")" << std::endl;
        }
    }
    else
    {
        cout << "Usage:" << endl;
        cmdParser.printUsage();
    }

    return static_cast<int>(retcode);
}

