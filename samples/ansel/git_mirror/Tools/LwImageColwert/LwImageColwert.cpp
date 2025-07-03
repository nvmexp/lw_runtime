#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdint.h>

#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")

#include "darkroom/Versioning.hpp"
#include "darkroom/ImageLoader.h"
#include "darkroom/Exr.h"
#include "darkroom/Png.h"
#include "darkroom/Bmp.h"
#include "darkroom/Jpeg.h"
#include "darkroom/Errors.h"
#include "darkroom/ImageOperations.h"
#include "darkroom/CmdLineParser.hpp"
#include "turbojpeg.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using darkroom::Error;

const unsigned int gJpegQuality = 99u;

namespace
{
    std::string findExtension(const std::string& filename)
    {
        const char* extPtr = PathFindExtension(filename.c_str());
        if (*extPtr != '\0')
            return std::string(extPtr);
        else
            return std::string();
    }

    void printUsage()
    {
        cout << "LwImageColwert.exe [--append-vertically|--append-horizontally] image1[.exr|.bmp|.png|.jpg] image2[.exr|.bmp|.png|.jpg] output[.exr|.bmp|.png|.jpg|.jps]" << endl;
        cout << "LwImageColwert.exe --colwert image1[.bmp|.png|.jpg] output[.bmp|.png|.jpg]" << endl;
        cout << "This utility appends two images horizontally or vertically. It can also colwert jpg, png and bmp images." << endl;
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
    darkroom::CmdLineParser cmdParser;
    cmdParser.addSingularOption("version", "Print tool version");
    cmdParser.addSingularOption("colwert", "Colwert between image formats");
    cmdParser.addSingularOption("append-horizontally", "Append images left-right");
    cmdParser.addSingularOption("append-vertically", "Append images top-bottom");
    cmdParser.addRegularOption("threads", "How many threads the tool is allowed to use");
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
    unsigned int threadCount = 0u;

    if (cmdParser.hasOption("threads"))
        threadCount = cmdParser.getOptionAs<uint32_t>("threads");

    if (!optionsParsed)
    {
        printUsage();
        cmdParser.printUsage();
        return 0;
    }
    
    if (cmdParser.hasOption("version"))
        cout << "LwImageColwert version: " << getVersion() << endl;

    // append mode
    if (cmdParser.hasOption("append-horizontally") || cmdParser.hasOption("append-vertically"))
    {
        const std::string inputFile1 = cmdParser.getFreeOptionAs<std::string>(0);
        const std::string inputFile2 = cmdParser.getFreeOptionAs<std::string>(1);
        const std::string outputFile = cmdParser.getFreeOptionAs<std::string>(2);
        const std::string in1(inputFile1), in2(inputFile2), out(outputFile);
        unsigned int w1 = 0, h1 = 0, w2 = 0, h2 = 0;
        std::vector<unsigned char> in1bufUChar, in2bufUChar;
        std::vector<float> in1bufFloat, in2bufFloat;

        if (findExtension(in1) == ".exr" && findExtension(in2) == ".exr")
        {
            std::unordered_map<std::string, std::string> tags;
            in1bufFloat = darkroom::loadExr(in1, w1, h1, tags);
            in2bufFloat = darkroom::loadExr(in2, w2, h2, tags);
        }
        else
        {
            in1bufUChar = darkroom::loadImage(in1, w1, h1, darkroom::BufferFormat::RGB8);
            in2bufUChar = darkroom::loadImage(in2, w2, h2, darkroom::BufferFormat::RGB8);
        }

        if ((!in1bufUChar.empty() && !in2bufUChar.empty()) || 
            (!in1bufFloat.empty() && !in2bufFloat.empty()) && 
            w1 == w2 && h1 == h2)
        {
            const bool isExr = in1bufUChar.empty();
            std::vector<unsigned char> outbufUchar;
            std::vector<float> outbufFloat;
            unsigned int outputWidth = w1, outputHeight = h1;
            if (cmdParser.hasOption("append-vertically"))
            {
                if (isExr)
                    darkroom::appendVertically(outbufFloat, in1bufFloat, in2bufFloat);
                else
                    darkroom::appendVertically(outbufUchar, in1bufUChar, in2bufUChar);
                outputHeight *= 2;
            }
            else if (cmdParser.hasOption("append-horizontally"))
            {
                if (isExr)
                    darkroom::appendHorizontally(outbufFloat, in1bufFloat, in2bufFloat, w1, h1);
                else
                    darkroom::appendHorizontally(outbufUchar, in1bufUChar, in2bufUChar, w1, h1);
                outputWidth *= 2;
            }

            const std::string outext = findExtension(out);
            if (!isExr && outext == ".bmp")
                darkroom::saveBmp(outbufUchar.data(), out, outputWidth, outputHeight, darkroom::BufferFormat::RGB8);
            else if (!isExr && (outext == ".jpg" || outext == ".jpeg" || outext == ".jps"))
            {
                std::unordered_map<uint16_t, std::string> tags;
                std::vector<char> xmp;
                darkroom::ImageMetadata metadata = generateImageMetadata(cmdParser);
                darkroom::colwertFromImageMetadata(metadata, tags, xmp);
                darkroom::saveJpeg(outbufUchar.data(), nullptr, out, outputWidth, outputHeight, darkroom::BufferFormat::RGB8, tags);
            }
            else if (!isExr && outext == ".png")
            {
                std::unordered_map<std::string, std::string> tags;
                darkroom::ImageMetadata metadata = generateImageMetadata(cmdParser);
                darkroom::colwertFromImageMetadata(metadata, tags);
                darkroom::savePng(outbufUchar.data(), nullptr, out, outputWidth, outputHeight, darkroom::BufferFormat::RGB8, tags, 2, threadCount);
            }
            else if (isExr && outext == ".exr")
            {
                std::unordered_map<std::string, std::string> tags;
                darkroom::ImageMetadata metadata = generateImageMetadata(cmdParser);
                darkroom::colwertFromImageMetadata(metadata, tags);
                darkroom::saveExr(outbufFloat.data(), nullptr, out, outputWidth, outputHeight, darkroom::BufferFormat::RGB32, tags);
            }
            else
                cout << "Couldn't save output image (input/output format inconsistency)" << endl;
        }
        else
            cout << "Couldn't load one of the input files or images are not of equal dimensions" << endl;
    }
    // colwert mode
    else if (cmdParser.hasOption("colwert"))
    {
        const std::string inputFile = cmdParser.getFreeOptionAs<std::string>(0);
        const std::string outputFile = cmdParser.getFreeOptionAs<std::string>(1);
        if (!inputFile.empty() && !outputFile.empty())
        {
            const std::string in(inputFile);
            unsigned int w = 0, h = 0;
            const auto inbuf = darkroom::loadImage(in, w, h, darkroom::BufferFormat::RGB8);
            if (!inbuf.empty())
            {
                const std::string outext = findExtension(outputFile);
                if (outext == ".bmp")
                    darkroom::saveBmp(inbuf.data(), outputFile, w, h);
                else if (outext == ".jpg" || outext == ".jpeg" || outext == ".jps")
                {
                    std::unordered_map<uint16_t, std::string> tags;
                    std::vector<char> xmp;
                    darkroom::ImageMetadata metadata = generateImageMetadata(cmdParser);
                    darkroom::colwertFromImageMetadata(metadata, tags, xmp);
                    darkroom::saveJpeg(inbuf.data(), nullptr, outputFile, w, h, darkroom::BufferFormat::RGB8, tags);
                }
                else if (outext == ".png")
                {
                    std::unordered_map<std::string, std::string> tags;
                    darkroom::ImageMetadata metadata = generateImageMetadata(cmdParser);
                    darkroom::colwertFromImageMetadata(metadata, tags);
                    darkroom::savePng(inbuf.data(), nullptr, outputFile, w, h, darkroom::BufferFormat::RGB8, tags, 2, threadCount);
                }
            }
            else
                cout << "Couldn't load one of the input files or images are not of equal dimensions" << endl;
        }
    }
    else
    {
        printUsage();
        cmdParser.printUsage();
    }

    return 0;
}

