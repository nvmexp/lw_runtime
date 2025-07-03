// SphericalEquirect.cpp : Defines the entry point for the console application.
//
#include <fstream>
#include <sstream>
#include <iterator>
#include <iostream>
#include <chrono>
#include <vector>
#include <array>

#include "darkroom/Versioning.hpp"
#include "darkroom/Bmp.h"
#include "darkroom/Exr.h"
#include "darkroom/Png.h"
#include "darkroom/Jpeg.h"
#include "darkroom/Errors.h"
#include "darkroom/Equirect.h"
#include "darkroom/PixelFormat.h"
#include "darkroom/JobProcessing.h"
#include "darkroom/CmdLineParser.hpp"

#include "turbojpeg.h"

#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")

namespace
{
    bool isExr = false;
    std::vector<std::string> tileNames;
    std::vector<std::vector<float>> tilesExr;
    std::vector<std::vector<unsigned char>> tilesBmp;
    std::vector<const float*> tileExrPtrs;
    std::vector<const unsigned char*> tileBmpPtrs;

    auto cPi = 3.14159265358979323846264338327950288f;
    auto cPi2 = cPi / 2.0f;
    const float toRad = cPi / 180.0f;
    const unsigned int gJpegQuality = 99u;

    const std::string gpanoPacket =
        "<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>"
            "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\" x:xmptk=\"Ansel\">"
                "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
                    "<rdf:Description rdf:about=\"\" xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\" xmlns:GPano=\"http://ns.google.com/photos/1.0/panorama/\">"
                        "<GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>"
                        "<GPano:ProjectionType>equirectangular</GPano:ProjectionType>"
                        "<GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels>"
                        "<GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>"
                        "<GPano:CroppedAreaImageWidthPixels>!width</GPano:CroppedAreaImageWidthPixels>"
                        "<GPano:CroppedAreaImageHeightPixels>!height</GPano:CroppedAreaImageHeightPixels>"
                        "<GPano:FullPanoWidthPixels>!width</GPano:FullPanoWidthPixels>"
                        "<GPano:FullPanoHeightPixels>!height</GPano:FullPanoHeightPixels>"
                    "</rdf:Description>"
                "</rdf:RDF>"
            "</x:xmpmeta>"
        "<?xpacket end=\"w\"?>";

    std::string replaceAll(std::string str, const std::string& from, const std::string& to) 
    {
        size_t start_pos = 0;
        while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
        }
        return str;
    }

    void signalFatal(bool cond, const char* message)
    {
        if (cond)
        {
            std::cerr << message << std::endl;
            exit(-1);
        }
    }

    float toFloat(const std::wstring& in)
    {
        float result;
        std::wstringstream ss(in);
        ss >> result;
        return result;
    }

    float toFloat(const std::string& in)
    {
        float result;
        std::stringstream ss(in);
        ss >> result;
        return result;
    }

    std::vector<darkroom::TileInfo> processPTMenderScript(const std::string& fname,
        std::vector<std::string>& tileNames,
        unsigned int& panoWidth)
    {
        std::ifstream in(fname);
        std::vector<darkroom::TileInfo> tileInfo;
        if (in)
        {
            while (in)
            {
                std::string o, type, width, height, name, roll, pitch, yaw, hfov, blendFactor;
                in >> o;
                if (o == "p")
                {
                    in >> type >> width >> height >> hfov >> name;
                    panoWidth = static_cast<unsigned int>(toFloat(width.substr(1, width.size() - 1)));
                }
                else if (o == "o")
                {
                    in >> type >> width >> height >> name >> roll >> pitch >> yaw >> hfov;
                    // read blendFactor optional parameter
                    in >> blendFactor;
                    darkroom::TileInfo info;
                    if (blendFactor[0] != 'b')
                    {
                        info.blendFactor = 1.0f;
                        for (auto it = blendFactor.rbegin(); it != blendFactor.rend(); ++it)
                            in.putback(*it);
                    }
                    else
                        info.blendFactor = toFloat(blendFactor.substr(1, blendFactor.size() - 1));
                    info.yaw = toFloat(yaw.substr(1, yaw.size() - 1)) * toRad;
                    info.pitch = -toFloat(pitch.substr(1, pitch.size() - 1)) * toRad;
                    info.horizontalFov = toFloat(hfov.substr(1, hfov.size() - 1)) * toRad;
                    tileInfo.push_back(info);
                    tileNames.push_back(name.substr(2, name.size() - 3));
                }
            }
        }
        else
        {
            const std::string& message = std::string("Couldn't open file '") + fname + "'";
            signalFatal(true, message.c_str());
        }
        return tileInfo;
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

    template<typename T>
    std::vector<const T*> getPointers(const std::vector<std::vector<T>>& tiles)
    {
        std::vector<const T*> tilePtrs;
        for (auto& tile : tiles)
            if (!tile.empty())
                tilePtrs.push_back(&tile[0]);
        return tilePtrs;
    }
}

int main(int argc, char* argv[])
{
    darkroom::Error retcode = darkroom::Error::kSuccess;
    const auto start = std::chrono::system_clock::now();
    darkroom::CmdLineParser cmdParser;
    cmdParser.addSingularOption("version", "Print tool version");
    cmdParser.addSingularOption("360", "Add XMP/GPano tags");
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

    if (cmdParser.hasOption("version"))
    {
        std::cout << "SphericalEquirect version: " << getVersion() << std::endl;
        return 0;
    }

    // PTMender mode: 
    // SphericalEquirect captures.txt result.jpg
    // or SphericalEquirect --threads 4 captures.txt result.jpg
    const auto ptMenderScript = cmdParser.getFreeOptionAs<std::string>(0);
    const auto resultName = cmdParser.getFreeOptionAs<std::string>(1);
    if (!ptMenderScript.empty() && !resultName.empty())
    {
        unsigned int threadCount = 0, descriptionIndex = 1;;
        // -threads should only be the first argument (or not present)
        if (cmdParser.hasOption("threads"))
            threadCount = cmdParser.getOptionAs<unsigned int>("threads");
        unsigned int preferredWidth = 0u;
        std::vector<darkroom::TileInfo> tileInfo = processPTMenderScript(ptMenderScript, tileNames, preferredWidth);

        unsigned int w = 0, h = 0;
        const bool isLossless = cmdParser.hasOption("lossless");

        if (tileNames.size() > 0)
        {
            std::ifstream tileFile(tileNames[0] + ".exr", std::ios::in);
            if (tileFile.good())
            {
                isExr = true;
            }
        }
        else
            return static_cast<int>(darkroom::Error::kOperationFailed);

        std::vector<unsigned char> result;
        std::vector<float> resultExr;
        const unsigned int hr = preferredWidth / 2;

        if (isExr)
        {
            std::unordered_map<std::string, std::string> tags;
            darkroom::loadExr(tileNames[0u] + ".exr", w, h, tags);
            tilesExr.resize(tileNames.size());
            tileExrPtrs.resize(tileNames.size());
        }
        else
        {
            darkroom::loadBmp(tileNames[0u] + ".bmp", w, h);
            tilesBmp.resize(tileNames.size());
            tileBmpPtrs.resize(tileNames.size());
        }

        const auto loadCallback = [](const std::vector<size_t>& loadList)
        {
            if (isExr)
            {
                for (auto index : loadList)
                {
                    uint32_t w, h;
                    std::unordered_map<std::string, std::string> tags;
                    tilesExr[index] = darkroom::loadExr(tileNames[index] + ".exr", w, h, tags);
                    tileExrPtrs[index] = &tilesExr[index][0];
                }
            }
            else
            {
                for (auto index : loadList)
                {
                    uint32_t w, h;
                    tilesBmp[index] = darkroom::loadBmp(tileNames[index] + ".bmp", w, h);
                    tileBmpPtrs[index] = &tilesBmp[index][0];
                }
            }
        };

        const auto unloadCallback = [](const std::vector<size_t>& unloadList)
        {
            if (isExr)
            {
                for (auto index : unloadList)
                {
                    tilesExr[index] = std::vector<float>();
                    tileExrPtrs[index] = nullptr;
                }
            }
            else
            {
                for (auto index : unloadList)
                {
                    tilesBmp[index] = std::vector<unsigned char>();
                    tileBmpPtrs[index] = nullptr;
                }
            }
        };

        if (isExr)
        {
            resultExr.resize(preferredWidth * hr * 3);
            retcode = darkroom::sphericalEquirect(&tileExrPtrs[0], 
                loadCallback, unloadCallback,
                &tileInfo[0], 
                static_cast<unsigned int>(tileExrPtrs.size()), 
                w, h, &resultExr[0], preferredWidth, hr, threadCount);
        }
        else
        {
            result.resize(preferredWidth * hr * 3);
            retcode = darkroom::sphericalEquirect(&tileBmpPtrs[0], 
                loadCallback, unloadCallback,
                &tileInfo[0], 
                static_cast<unsigned int>(tileBmpPtrs.size()), 
                w, h, &result[0], preferredWidth, hr, threadCount);
        }

        if (retcode == darkroom::Error::kSuccess)
        {
            std::vector<char> xmpPacket;
            if (cmdParser.hasOption("360") && !isExr && !isLossless)
            {
                const std::string identifier = "http://ns.adobe.com/xap/1.0/";
                xmpPacket.insert(xmpPacket.end(), identifier.cbegin(), identifier.cend());
                xmpPacket.push_back('\0');
                std::string gpano = gpanoPacket;
                {
                    std::stringstream ss;
                    ss << preferredWidth;
                    gpano = replaceAll(gpano, "!width", ss.str());
                }
                {
                    std::stringstream ss;
                    ss << hr;
                    gpano = replaceAll(gpano, "!height", ss.str());
                }
                xmpPacket.insert(xmpPacket.end(), gpano.cbegin(), gpano.cend());
            }

            const auto ext = findExtension(resultName);

            if (isExr)
            {
                std::unordered_map<std::string, std::string> tags;
                darkroom::ImageMetadata metadata = generateImageMetadata(cmdParser);
                darkroom::colwertFromImageMetadata(metadata, tags);
                retcode = darkroom::saveExr(resultExr.data(), nullptr, resultName, preferredWidth, hr, darkroom::BufferFormat::RGB32, tags);
            }
            else if (ext == ".jpg" || ext == ".jpeg")
            {
                std::unordered_map<uint16_t, std::string> tags;
                std::vector<char> xmp;
                darkroom::ImageMetadata metadata = generateImageMetadata(cmdParser);
                darkroom::colwertFromImageMetadata(metadata, tags, xmp);
                retcode = darkroom::saveJpeg(result.data(), nullptr, resultName, preferredWidth, hr, darkroom::BufferFormat::BGR8, tags, xmpPacket);
            }
            else if (ext == ".png")
            {
                std::unordered_map<std::string, std::string> tags;
                darkroom::ImageMetadata metadata = generateImageMetadata(cmdParser);
                darkroom::colwertFromImageMetadata(metadata, tags);
                retcode = darkroom::savePng(result.data(), nullptr, resultName, preferredWidth, hr, darkroom::BufferFormat::BGR8, tags);
            }

            if (retcode != darkroom::Error::kSuccess)
                std::cout << "Saving result has failed with (" << darkroom::errToString(retcode) << ")" << std::endl;
        }
        else
            std::cout << "sphericalEquirect failed with (" << darkroom::errToString(retcode) << ")" << std::endl;
    }
    else
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "SphericalEquirect [options] captures.txt result[.jpg|.exr]" << std::endl;
        std::cout << "\tConsumes PTMender-compatible script 'captures.txt'" << std::endl;
        std::cout << "SphericalEquirect produces equirectangular image out of tiles" << std::endl;
        cmdParser.printUsage();
    }

    const auto end = std::chrono::system_clock::now();
    const auto duration = end - start;

    std::cout << "Total time = " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " ms" << std::endl;

    return static_cast<int>(retcode);
}

