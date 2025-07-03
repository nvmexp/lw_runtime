#define NOMINMAX
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <stdint.h>

#include "darkroom/Versioning.hpp"
#include "darkroom/ImageLoader.h"
#include "darkroom/Exr.h"
#include "darkroom/Png.h"
#include "darkroom/Bmp.h"
#include "darkroom/Jpeg.h"
#include "darkroom/Errors.h"
#include "darkroom/ImageOperations.h"
#include "darkroom/CmdLineParser.hpp"

#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using darkroom::Error;

namespace
{
    const auto THUMBNAILTOOL_VERSION_MAJOR = 1;
    const auto THUMBNAILTOOL_VERSION_MINOR = 0;

    const auto DEFAULT_MEMORY_LIMIT = 512 * 1024 * 1024; // 512Mb is the default memory limit

    std::vector<float> gOutputAclwm, gOutputSum;
    uint32_t gDesiredWidth = 0u, gDesiredHeight = 0u, gInputWidth = 0u, gInputHeight = 0u, gChannels = 3u, gLwrrentRow = 0u;

    enum class ThumbnailToolError
    {
        kOk,
        kIlwalidArguments,
        kIlwalidInput,
        kInternalError,
        kNotEnoughMemory,
        kInputFileNotSpecified,
        kOutputFileNotSpecified,
        kThumbnailDimensionsUnspecified
    };

    std::wstring findExtension(const std::wstring& filename)
    {
        const wchar_t* extPtr = PathFindExtensionW(filename.c_str());
        if (*extPtr != '\0')
            return std::wstring(extPtr);
        else
            return std::wstring();
    }
    std::string findExtension(const std::string& filename)
    {
        const char* extPtr = PathFindExtensionA(filename.c_str());
        if (*extPtr != '\0')
            return std::string(extPtr);
        else
            return std::string();
    }

    ThumbnailToolError checkArguments(const darkroom::CmdLineParser& cmdParser)
    {		
        if (!cmdParser.hasOption("input"))
        {
            cout << "Input file is not specified" << endl;
            return ThumbnailToolError::kInputFileNotSpecified;
        }
        else
        {
            const auto inputFilename = cmdParser.getOptionAs<std::string>("input");
            auto ext = findExtension(inputFilename);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".jp2" && ext != ".png" && ext != ".bmp" && ext != ".exr")
            {
                cout << "Unsupported input file extension " << ext << "!" << endl;
                return ThumbnailToolError::kIlwalidArguments;
            }
        }
        if (!cmdParser.hasOption("output"))
        {
            cout << "Output file is not specified" << endl;
            return ThumbnailToolError::kOutputFileNotSpecified;
        }
        else
        {
            const auto outputFilename = cmdParser.getOptionAs<std::string>("output");
            auto ext = findExtension(outputFilename);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png")
            {
                cout << "Unsupported output file extension" << endl;
                return ThumbnailToolError::kIlwalidArguments;
            }
        }
        if (cmdParser.hasOption("tonemap-operator"))
        {
            const auto op = cmdParser.getOptionAs<std::string>("tonemap-operator");
            if (op != "reinhard" && op != "clamp" && op != "filmic" && op != "filmic-raw")
            {
                cout << "Unknown tonemap operator ('" << op << "')" << endl;
                return ThumbnailToolError::kIlwalidArguments;
            }
        }
        return ThumbnailToolError::kOk;
    }

    darkroom::ImageMetadata generateImageMetadata(const darkroom::CmdLineParser& cmdParser)
    {
        darkroom::ImageMetadata metadata;
        if (cmdParser.hasOption("make"))
            metadata.tagMake = cmdParser.getOptionAs<std::string>("make");
        if (cmdParser.hasOption("model"))
            metadata.tagModel = cmdParser.getOptionAs<std::string>("model");
        if (cmdParser.hasOption("software"))
            metadata.tagSoftware = cmdParser.getOptionAs<std::string>("software");
        if (cmdParser.hasOption("description"))
            metadata.tagDescription = cmdParser.getOptionAs<std::string>("description");
        if (cmdParser.hasOption("type"))
            metadata.tagType = cmdParser.getOptionAs<std::string>("type");
        if (cmdParser.hasOption("drsname"))
            metadata.tagDrsName = cmdParser.getOptionAs<std::string>("drsname");
        if (cmdParser.hasOption("drsprofilename"))
            metadata.tagDrsProfileName = cmdParser.getOptionAs<std::string>("drsprofilename");
        if (cmdParser.hasOption("apptitlename"))
            metadata.tagAppTitleName = cmdParser.getOptionAs<std::string>("apptitlename");
        if (cmdParser.hasOption("appcmsid"))
            metadata.tagAppCMSID = cmdParser.getOptionAs<std::string>("appcmsid");
        if (cmdParser.hasOption("appshortname"))
            metadata.tagAppShortName = cmdParser.getOptionAs<std::string>("appshortname");
        if (cmdParser.hasOption("activefilters"))
            metadata.tagActiveFilters = cmdParser.getOptionAs<std::string>("activefilters");
        return metadata;
    }
}


int main(int argc, char *argv[])
{
    try
    {
        darkroom::CmdLineParser cmdParser;
        cmdParser.addSingularOption("help", "Print help");
        cmdParser.addSingularOption("version", "Print tool version");
        cmdParser.addRegularOption("max-memory", "Maximum amount of memory allowed to be used during operation");
        cmdParser.addRegularOption("tonemap-operator", "Tonemap operator to use [reinhard, clamp, filmic, filmic-raw] (reinhard by default)");
        cmdParser.addRegularOption("input", "Input filename");
        cmdParser.addRegularOption("output", "Output filename");
        cmdParser.addRegularOption("width", "Thumbnail width");
        cmdParser.addRegularOption("height", "Thumbnail height");
        cmdParser.addRegularOption("make", "EXIF tag 'Make'");
        cmdParser.addRegularOption("model", "EXIF tag 'Model' and 'UniqueModel'");
        cmdParser.addRegularOption("software", "EXIF tag 'Software'");
        cmdParser.addRegularOption("description", "EXIF tag 'ImageDescription'");
        cmdParser.addRegularOption("drsname", "EXIF tag for DRSName");
        cmdParser.addRegularOption("drsprofilename", "EXIF tag for DRSProfileName");
        cmdParser.addRegularOption("apptitlename", "EXIF tag for AppTitleName");
        cmdParser.addRegularOption("appcmsid", "EXIF tag for AppCMSID");
        cmdParser.addRegularOption("appshortname", "EXIF tag for AppShortName");
        cmdParser.addRegularOption("activefilters", "EXIF tag for ActiveFilters");
        cmdParser.addRegularOption("type", "EXIF tag 'MakerNote'");
        const bool optionsParsed = cmdParser.parse(argc, argv);
        uint32_t threadCount = 0u;

        if (!optionsParsed)
        {
            cout << "Invalid arguments" << endl;
            return int(ThumbnailToolError::kIlwalidArguments);
        }

        if (cmdParser.hasOption("help"))
        {
            cmdParser.printUsage();
            return int(ThumbnailToolError::kOk);
        }

        if (cmdParser.hasOption("version"))
        {
            cout << "ThumbnailTool version: " << THUMBNAILTOOL_VERSION_MAJOR << "." << THUMBNAILTOOL_VERSION_MINOR << endl;
            return int(ThumbnailToolError::kOk);
        }

        const auto argumentsCorrectnessStatus = checkArguments(cmdParser);

        if (argumentsCorrectnessStatus != ThumbnailToolError::kOk)
            return int(argumentsCorrectnessStatus);

        if (cmdParser.hasOption("threads"))
            threadCount = cmdParser.getOptionAs<uint32_t>("threads");

        const auto inputFilename = cmdParser.getOptionAs<std::string>("input");
        const auto outputFilename = cmdParser.getOptionAs<std::string>("output");
        auto inputExt = findExtension(inputFilename);
        std::transform(inputExt.begin(), inputExt.end(), inputExt.begin(), ::tolower);
        auto outputExt = findExtension(outputFilename);
        std::transform(outputExt.begin(), outputExt.end(), outputExt.begin(), ::tolower);
        const auto isInputExr = inputExt == ".exr";
        const auto needsTonemapping = isInputExr;
        size_t maxBytes = DEFAULT_MEMORY_LIMIT;
        if (cmdParser.hasOption("max-memory"))
            maxBytes = cmdParser.getOptionAs<size_t>("max-memory");
        darkroom::TonemapOperator tonemapOp = darkroom::TonemapOperator::kReinhardSimple;
        if (cmdParser.hasOption("tonemap-operator"))
        {
            const auto op = cmdParser.getOptionAs<std::string>("tonemap-operator");
            if (op == "reinhard")
                tonemapOp = darkroom::TonemapOperator::kReinhardSimple;
            else if (op == "clamp")
                tonemapOp = darkroom::TonemapOperator::kClamp;
            else if (op == "filmic")
                tonemapOp = darkroom::TonemapOperator::kFilmic;
            else if (op == "filmic-raw")
                tonemapOp = darkroom::TonemapOperator::kFilmicLinear;
        }
        // compute thumbnail width and height
        // it's possible to only specify one of the two and the other will be
        // computed such that the thumbnail is of the same aspect ratio as the input
        // image. It's also possible to set both width and height
        const auto widthSpecified = cmdParser.hasOption("width");
        const auto heightSpecified = cmdParser.hasOption("height");

        if (isInputExr)
        {
            if (darkroom::getExrImageSize(inputFilename, gInputWidth, gInputHeight) != darkroom::Error::kSuccess)
                return int(ThumbnailToolError::kIlwalidInput);
        }
        else
        {
            if (darkroom::getImageSize(inputFilename, gInputWidth, gInputHeight) != darkroom::Error::kSuccess)
                return int(ThumbnailToolError::kIlwalidInput);
        }

        // callwlate thumbnail size
        if (widthSpecified && heightSpecified)
        {
            gDesiredWidth = cmdParser.getOptionAs<uint32_t>("width");
            gDesiredHeight = cmdParser.getOptionAs<uint32_t>("height");
        }
        else if (widthSpecified)
        {
            gDesiredWidth = cmdParser.getOptionAs<uint32_t>("width");
            gDesiredHeight = gDesiredWidth * gInputHeight / gInputWidth;
        }
        else if (heightSpecified)
        {
            gDesiredHeight = cmdParser.getOptionAs<uint32_t>("height");
            gDesiredWidth = gDesiredHeight * gInputWidth / gInputHeight;
        }
        else
        {
            gDesiredWidth = gInputWidth;
            gDesiredHeight = gInputHeight;
        }

        // we can only make the size lower, otherwise the algorithm doesn't make much sense
        if (gDesiredHeight > gInputHeight || gDesiredWidth > gInputWidth)
        {
            cout << "Thumbnail size is larger than input image size" << endl;
            return int(ThumbnailToolError::kIlwalidArguments);
        }

        gOutputAclwm.resize(gDesiredWidth * gDesiredHeight * 3, 0);
        gOutputSum.resize(gDesiredWidth * gDesiredHeight, 0);

        darkroom::Error error = darkroom::Error::kSuccess;
        std::vector<unsigned char> input;
        // define process callbacks (for 8 bit and >8 bit per channel images), which call aclwmulation function for N rows read 
        // by one of the streaming load functions
        const auto processCallback = [](const unsigned char* rows, uint32_t rowsRead)
        {
            const auto error = darkroom::downscaleAclwmulate(rows, gOutputAclwm.data(), gOutputSum.data(),
                gChannels, gInputWidth, gInputHeight, gChannels, gDesiredWidth, gDesiredHeight,
                0u, gLwrrentRow, gInputWidth, rowsRead);
            if (error == darkroom::Error::kSuccess)
                gLwrrentRow += rowsRead;
            return error;
        };

        const auto processCallbackExr = [](const float* rows, uint32_t rowsRead)
        {
            const auto error = darkroom::downscaleAclwmulate(rows, gOutputAclwm.data(), gOutputSum.data(),
                gChannels, gInputWidth, gInputHeight, gChannels, gDesiredWidth, gDesiredHeight,
                0u, gLwrrentRow, gInputWidth, rowsRead);
            if (error == darkroom::Error::kSuccess)
                gLwrrentRow += rowsRead;
            return error;
        };

        // callwlate amount of the memory allowance by the amount allocated here for the output image
        const size_t memoryRequiredHere = gDesiredWidth * gDesiredHeight * gChannels + 
        // and aclwmulator image gOutputAclwm
        gDesiredWidth * gDesiredHeight * gChannels * sizeof(float) +
        // and gOutputSum vector (counts amount of pixels aclwmulated in gOutputAclwm)
        gDesiredWidth * gDesiredHeight * sizeof(float);

        if (memoryRequiredHere > maxBytes)
        {
            cout << "Maximum memory limit reached before processing started. Not enough memory to continue." << endl;
            return int(ThumbnailToolError::kNotEnoughMemory);
        }
        
        // Reduce amount of memory allowance by the memory allocated here (in this file)
        maxBytes -= memoryRequiredHere;

        // start loading input image and aclwmulating it
        if (inputExt == ".jpg" || inputExt == ".jpeg" || inputExt == ".jp2")
        {
            // To speedup JPEG decompression we callwlate scaling factor here.
            // libjpeg allows faster loading if decoded image/height are a fraction of the original dimensions.
            // The fraction can be 1/N, where N is 1..8
            darkroom::JpegStreamingScale scale = darkroom::JpegStreamingScale::k1_1;
            const float ratio = std::min(float(gInputWidth) / float(gDesiredWidth), float(gInputHeight) / float(gDesiredHeight));
            if (ratio > 8.0f)
                scale = darkroom::JpegStreamingScale::k1_8;
            else if (ratio > 7.0f)
                scale = darkroom::JpegStreamingScale::k1_7;
            else if (ratio > 6.0f)
                scale = darkroom::JpegStreamingScale::k1_6;
            else if (ratio > 5.0f)
                scale = darkroom::JpegStreamingScale::k1_5;
            else if (ratio > 4.0f)
                scale = darkroom::JpegStreamingScale::k1_4;
            else if (ratio > 3.0f)
                scale = darkroom::JpegStreamingScale::k1_3;
            else if (ratio > 2.0f)
                scale = darkroom::JpegStreamingScale::k1_2;

            error = darkroom::loadJpegStreaming(inputFilename, processCallback, maxBytes, gInputWidth, gInputHeight,
                darkroom::BufferFormat::RGB8, std::unordered_map<uint16_t, std::string>(), std::vector<char>(),
                darkroom::JpegStreamingQuality::kLow, scale);
        }
        else if (inputExt == ".png")
            error = darkroom::loadPngStreaming(inputFilename, processCallback, maxBytes, gInputWidth, gInputHeight, darkroom::BufferFormat::RGB8);
        else if (inputExt == ".exr")
            error = darkroom::loadExrStreaming(inputFilename, processCallbackExr, maxBytes,
                gInputWidth, gInputHeight, std::unordered_map<std::string, std::string>(), darkroom::BufferFormat::RGB32);

        if (error != darkroom::Error::kSuccess)
        {
            std::cout << "Internal error oclwred or corrupted input image" << std::endl;
            return int(ThumbnailToolError::kInternalError);
        }

        // allocate memory for the output image
        std::vector<unsigned char> output(gDesiredWidth * gDesiredHeight * gChannels);

        // average aclwmulator image
        // optionally tonemap HDR -> SDR
        if (needsTonemapping)
        {
            error = darkroom::downscaleAverage(gOutputAclwm.data(), gOutputAclwm.data(), gOutputSum.data(), gChannels, gDesiredWidth, gDesiredHeight);

            if (error != darkroom::Error::kSuccess)
                return int(ThumbnailToolError::kInternalError);

            if (tonemapOp == darkroom::TonemapOperator::kReinhardSimple)
                darkroom::tonemap<darkroom::TonemapOperator::kReinhardSimple>(gOutputAclwm.data(), output.data(), gDesiredWidth, gDesiredHeight, gChannels);
            else if (tonemapOp == darkroom::TonemapOperator::kClamp)
                darkroom::tonemap<darkroom::TonemapOperator::kClamp>(gOutputAclwm.data(), output.data(), gDesiredWidth, gDesiredHeight, gChannels);
            else if (tonemapOp == darkroom::TonemapOperator::kFilmic)
                darkroom::tonemap<darkroom::TonemapOperator::kFilmic>(gOutputAclwm.data(), output.data(), gDesiredWidth, gDesiredHeight, gChannels);
            else if (tonemapOp == darkroom::TonemapOperator::kFilmicLinear)
                darkroom::tonemap<darkroom::TonemapOperator::kFilmicLinear>(gOutputAclwm.data(), output.data(), gDesiredWidth, gDesiredHeight, gChannels);
            else
                return int(ThumbnailToolError::kInternalError);
        }
        else
        {
            error = darkroom::downscaleAverage(output.data(), gOutputAclwm.data(), gOutputSum.data(), gChannels, gDesiredWidth, gDesiredHeight);

            if (error != darkroom::Error::kSuccess)
                return int(ThumbnailToolError::kInternalError);
        }
        // finally, save the result
        if (outputExt == ".jpg" || outputExt == ".jpeg")
        {
            std::unordered_map<uint16_t, std::string> tags;
            std::vector<char> xmp;
            darkroom::ImageMetadata metadata = generateImageMetadata(cmdParser);
            darkroom::colwertFromImageMetadata(metadata, tags, xmp);
            darkroom::saveJpeg(output.data(), nullptr, outputFilename, gDesiredWidth, gDesiredHeight, darkroom::BufferFormat::RGB8, tags, xmp);
        }
        else if (outputExt == ".png")
        {
            std::unordered_map<std::string, std::string> tags;
            darkroom::ImageMetadata metadata = generateImageMetadata(cmdParser);
            darkroom::colwertFromImageMetadata(metadata, tags);
            darkroom::savePng(output.data(), nullptr, outputFilename, gDesiredWidth, gDesiredHeight, darkroom::BufferFormat::RGB8, tags);
        }
    }
    catch (...)
    {
        return int(ThumbnailToolError::kInternalError);
    }
    return 0;
}

