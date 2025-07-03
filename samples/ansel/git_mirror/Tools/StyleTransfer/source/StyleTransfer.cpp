#ifdef _M_AMD64

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <stdint.h>
#include <thread>

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

#include "style_transfer.h"

#pragma comment(lib, "ole32.lib")

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using darkroom::Error;

const unsigned int gJpegQuality = 99u;

namespace
{
}

std::vector<std::wstring> getAllFileNamesWithinFolder(const std::wstring& folder, const std::vector<std::wstring> & exts)
{
    std::vector<std::wstring> names;
    std::wstring searchPath;

    searchPath = folder + L"*.*";

    WIN32_FIND_DATA fd;
    HANDLE hFind = FindFirstFile(searchPath.c_str(), &fd);

    if (hFind != ILWALID_HANDLE_VALUE)
    {
        do
        {
            // read all (real) files in current folder
            // , delete '!' read other 2 default folder . and ..
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            {
                std::wstring filename(fd.cFileName);
                const size_t filename_length = filename.length();

                // If the exts list is empty, return the full file list
                bool validExtension = (exts.size() == 0);
                for (size_t extIdx = 0, extIdxEnd = exts.size(); extIdx < extIdxEnd; ++extIdx)
                {
                    const size_t lastof = filename.rfind(exts[extIdx]);
                    const size_t exts_length = exts[extIdx].length();
                    if (lastof == filename_length - exts_length)
                    {
                        validExtension = true;
                        break;
                    }
                }

                if (validExtension)
                    names.push_back(fd.cFileName);
            }
        } while (::FindNextFile(hFind, &fd));

        FindClose(hFind);
    }

    return names;
}

std::wstring findExtension(const std::wstring& filename)
{
    const wchar_t * extPtr = PathFindExtensionW(filename.c_str());
    if (*extPtr != L'\0')
        return std::wstring(extPtr);
    else
        return std::wstring();
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

namespace imageproc
{
    void rgb2ycbcrPixel(
            float r, float g, float b,
            float & y, float & cb, float & cr
            )
    {
        y = 0.0f + ( .299f*r + .587f*g + .114f*b);
        cb = .5f + (-.169f*r +-.331f*g + .500f*b);
        cr = .5f + ( .500f*r +-.419f*g +-.081f*b);
    }

    void ycbcr2rgbPixel(
            float y, float cb, float cr,
            float & r, float & g, float & b
            )
    {
        float cb_ = cb - .5f;
        float cr_ = cr - .5f;

        r = y + 0.000f*cb_ + 1.400f*cr_;
        g = y +-0.343f*cb_ +-0.711f*cr_;
        b = y + 1.765f*cb_ + 0.000f*cr_;
    }

    void rgb2hsvPixel(
            float r, float g, float b,
            float & h, float & s, float & v
            )
    {
        float K = 0.f;

        if (g < b)
        {
            std::swap(g, b);
            K = -1.f;
        }

        if (r < g)
        {
            std::swap(r, g);
            K = -2.f / 6.f - K;
        }

        float chroma = r - (g < b ? g : b);
        h = fabs(K + (g - b) / (6.f * chroma + 1e-20f));
        s = chroma / (r + 1e-20f);
        v = r;
    }

    void hsv2rgbPixel(
        float h, float s, float v,
        float & r, float & g, float & b
        )
    {
        float hSect = 6.f * h;
        int   hi = ((int)hSect) % 6;
        float f  = hSect - (int)hSect;
        float p  = v * (1.0f - s);
        float q  = v * (1.0f - s * f);
        float t  = v * (1.0f - s * (1.0f - f));

        switch (hi)
        {
            case 0: r = v, g = t, b = p; break;
            case 1: r = q, g = v, b = p; break;
            case 2: r = p, g = v, b = t; break;
            case 3: r = p, g = q, b = v; break;
            case 4: r = t, g = p, b = v; break;
            case 5: r = v, g = p, b = q; break;
        }
    }

    enum class ColorSpaces
    {
        kRGB,
        kHSV,
        kYCbCr,

        kNUM_ENTRIES
    };

    std::vector<float> colwertFromRGB(const std::vector<unsigned char> & image_rgb, unsigned int w, unsigned int h, unsigned int numChannels, ColorSpaces targetColorSpace)
    {
        std::vector<float> result_colw;
        result_colw.resize(w*h*numChannels);

        for (uint32_t ih = 0; ih < h; ++ih)
        {
            size_t ih_w = ih*w;
            for (uint32_t iw = 0; iw < w; ++iw)
            {
                size_t pixelOffset = (iw+ih_w)*numChannels;

                float r = image_rgb[pixelOffset+0] / 255.f;
                float g = image_rgb[pixelOffset+1] / 255.f;
                float b = image_rgb[pixelOffset+2] / 255.f;

                if (targetColorSpace == ColorSpaces::kHSV)
                {
                    float h, s, v;
                    rgb2hsvPixel(r, g, b, h, s, v);

                    result_colw[pixelOffset+0] = h;
                    result_colw[pixelOffset+1] = s;
                    result_colw[pixelOffset+2] = v;
                }
                else if (targetColorSpace == ColorSpaces::kYCbCr)
                {
                    float y, cb, cr;
                    rgb2ycbcrPixel(r, g, b, y, cb, cr);

                    result_colw[pixelOffset+0] = y;
                    result_colw[pixelOffset+1] = cb;
                    result_colw[pixelOffset+2] = cr;
                }
            }
        }

        return result_colw;
    }

    std::vector<unsigned char> colwertToRGB(const std::vector<float> & image_colw, unsigned int w, unsigned int h, unsigned int numChannels, ColorSpaces sourceColorSpace)
    {
        std::vector<unsigned char> result_rgb;
        result_rgb.resize(w*h*numChannels);

        for (uint32_t ih = 0; ih < h; ++ih)
        {
            size_t ih_w = ih*w;
            for (uint32_t iw = 0; iw < w; ++iw)
            {
                size_t pixelOffset = (iw+ih_w)*numChannels;

                float r, g, b;

                if (sourceColorSpace == ColorSpaces::kHSV)
                {
                    float h = image_colw[pixelOffset+0];
                    float s = image_colw[pixelOffset+1];
                    float v = image_colw[pixelOffset+2];

                    hsv2rgbPixel(h, s, v, r, g, b);
                }
                else if (sourceColorSpace == ColorSpaces::kYCbCr)
                {
                    float y  = image_colw[pixelOffset+0];
                    float cb = image_colw[pixelOffset+1];
                    float cr = image_colw[pixelOffset+2];

                    ycbcr2rgbPixel(y, cb, cr, r, g, b);
                }

                auto clampColor = [](float val)
                {
                    unsigned char color;

                    if (val > 255.f)
                        color = 255;
                    else if (val < 0.f)
                        color = 0;
                    else
                        color = (unsigned char)val;

                    return color;
                };

                result_rgb[pixelOffset+0] = clampColor(r * 255.f);
                result_rgb[pixelOffset+1] = clampColor(g * 255.f);
                result_rgb[pixelOffset+2] = clampColor(b * 255.f);
            }
        }

        return result_rgb;
    }

    struct ColorStats
    {
        double mean;
        double stddev;
    };

    void gatherColorStats(const std::vector<float> & image, unsigned int w, unsigned int h, unsigned int numChannels, ColorStats * outStats)
    {
        const uint32_t totalPixelsNum = w*h;
        const double totalPixelsNumD = (double)totalPixelsNum;

        auto sqr = [](double val)
        {
            return val*val;
        };
        for (int ich = 0; ich < 3; ++ich)
        {
            outStats[ich] = { 0.0f, 0.0f };

            // Callwlate mean
            for (uint32_t iwh = 0; iwh < totalPixelsNum; ++iwh)
            {
                outStats[ich].mean += image[iwh*numChannels+ich];
            }
            outStats[ich].mean /= totalPixelsNumD;

            // Callwlate standard deviation
            for (uint32_t iwh = 0; iwh < totalPixelsNum; ++iwh)
            {
                outStats[ich].stddev += sqr(image[iwh*numChannels+ich] - outStats[ich].mean);
            }
            outStats[ich].stddev /= totalPixelsNumD;
            outStats[ich].stddev = sqrt(outStats[ich].stddev);
        }
    }

    void matchColorStatsInImage(std::vector<float> & image, unsigned int w, unsigned int h, unsigned int numChannels, ColorStats * imageStats, ColorStats * targetStats)
    {
        for (int ich = 0; ich < 3; ++ich)
        {
            for (uint32_t iwh = 0; iwh < w*h; ++iwh)
            {
                float pixelVal = (float)( (targetStats[ich].stddev / imageStats[ich].stddev) * (image[iwh*numChannels+ich] - imageStats[ich].mean) + targetStats[ich].mean );

                if (pixelVal < 0.0f)
                    pixelVal = 0.0f;
                else if (pixelVal > 1.0f)
                    pixelVal = 1.0f;

                image[iwh*numChannels+ich] = pixelVal;
            }
        }
    }

    void recombineImages(std::vector<float> & imageDst, const std::vector<float> & imageSrc, unsigned int w, unsigned int h, unsigned int numChannels, ColorSpaces colorSpace)
    {
        for (uint32_t ih = 0; ih < h; ++ih)
        {
            size_t ih_w = ih*w;
            for (uint32_t iw = 0; iw < w; ++iw)
            {
                size_t pixelOffset = (iw+ih_w)*numChannels;

                if (colorSpace == imageproc::ColorSpaces::kHSV)
                {
                    // Hue (replace)
                    imageDst[pixelOffset+0] = imageSrc[pixelOffset+0];

                    // Saturation (min of the two)
                    float satStylized = imageDst[pixelOffset+1];
                    float satOriginal = imageSrc[pixelOffset+1];
                    imageDst[pixelOffset+1] = (satStylized < satOriginal ? satStylized : satOriginal);

                    // Value stays the same
                }
                else if (colorSpace == imageproc::ColorSpaces::kYCbCr)
                {
                    // Luminance (Y) stays the same
                    
                    // Replace CbCr channels
                    imageDst[pixelOffset+1] = imageSrc[pixelOffset+1];
                    imageDst[pixelOffset+2] = imageSrc[pixelOffset+2];
                }
            }
        }
    }

    template<typename T>
    T clampPixelValue(float val)
    {
        static_assert(false, "Unsupported type");
    }
    template<>
    unsigned char clampPixelValue<unsigned char>(float val)
    {
        return val > 255.0f ? 255 : (val < 0.0f ? 0 : (unsigned char)val);
    }
    template<>
    float clampPixelValue<float>(float val)
    {
        return val;
    }

    template<typename T>
    std::vector<T> resampleImage(const std::vector<T> & image, unsigned int w, unsigned int h, unsigned int numChannels, unsigned int nw, unsigned int nh)
    {
        std::vector<T> result_rs;
        result_rs.resize(nw*nh*numChannels);

        float sw = w / (float)nw, sh = h / (float)nh;
        for (uint32_t inw = 0; inw < nw; ++inw)
        {
            for (uint32_t inh = 0; inh < nh; ++inh)
            {
                float owf = inw * sw, owf1 = (inw + 1) * sw;
                float ohf = inh * sh, ohf1 = (inh + 1) * sh;

                uint32_t pixelOffsetRS = (inw+inh*nw)*numChannels;

                float accPixelsColor[4] = { 0.f, 0.f, 0.f, 0.f };
                int accPixelsCnt = 0;

                int samplesW = (int)sw;
                if (samplesW == 0) samplesW = 1;
                int samplesH = (int)sh;
                if (samplesH == 0) samplesH = 1;

                for (int woffset = 0; woffset < samplesW; ++woffset)
                {
                    float iwaf = owf + woffset;
                    if (iwaf >= (float)w)
                        continue;

                    for (int hoffset = 0; hoffset < samplesH; ++hoffset)
                    {
                        float ihaf = ohf + hoffset;
                        if (ihaf >= (float)h)
                            continue;

                        unsigned int ow = (int)iwaf, ow1 = ow+1;
                        unsigned int oh = (int)ihaf, oh1 = oh+1;

                        if (ow1 >= w) ow1 = w-1;
                        if (oh1 >= h) oh1 = h-1;

                        unsigned int pixelOffsetUS_TL = (ow  + oh *w)*numChannels;
                        unsigned int pixelOffsetUS_TR = (ow1 + oh *w)*numChannels;
                        unsigned int pixelOffsetUS_BL = (ow  + oh1*w)*numChannels;
                        unsigned int pixelOffsetUS_BR = (ow1 + oh1*w)*numChannels;

                        float weight_w0 = iwaf - ow;
                        float weight_h0 = ihaf - oh;

                        for (uint32_t ich = 0; ich < numChannels; ++ich)
                            accPixelsColor[ich] += 
                                image[pixelOffsetUS_TL+ich]*(1.f-weight_w0)*(1.f-weight_h0) +
                                image[pixelOffsetUS_TR+ich]*(weight_w0)*(1.f-weight_h0) +
                                image[pixelOffsetUS_BL+ich]*(1.f-weight_w0)*(weight_h0) +
                                image[pixelOffsetUS_BR+ich]*(weight_w0)*(weight_h0);

                        ++accPixelsCnt;
                    }
                }

                for (uint32_t ich = 0; ich < numChannels; ++ich)
                {
                    result_rs[pixelOffsetRS + ich] = clampPixelValue<T>(accPixelsColor[ich] / accPixelsCnt);
                }
            }
        }

        return result_rs;
    }
}

void cancelStyle()
{

}

// example arguments:
// --content data/gta5_1280x720.jpg --style data/style.jpg --output data/result.jpg --vgg model/model_vgg.t7 --decoder model/decoder.t7

int main(int argc, char *argv[])
{
    darkroom::CmdLineParser cmdParser;
    cmdParser.addSingularOption("version", "Print tool version");
    cmdParser.addRegularOption("content", "Content image");
    cmdParser.addRegularOption("style", "Style image");
    cmdParser.addRegularOption("output", "Result image");
    cmdParser.addRegularOption("folder-content", "Folder with content images");
    cmdParser.addRegularOption("folder-style", "Folder with style images");
    cmdParser.addRegularOption("folder-output", "Target folder for batch processing");
    cmdParser.addRegularOption("vgg", "Feature extractor network (.t7)");
    cmdParser.addRegularOption("decoder", "Decoder network (.t7)");
    cmdParser.addRegularOption("threads", "How many threads the tool is allowed to use");
    cmdParser.addRegularOption("benchmark", "How many times to perform forward pass (also report average run time)");
    cmdParser.addRegularOption("make", "EXIF tag 'Make'");
    cmdParser.addRegularOption("model", "EXIF tag 'Model' and 'UniqueModel'");
    cmdParser.addRegularOption("software", "EXIF tag 'Software'");
    cmdParser.addRegularOption("type", "EXIF tag 'MakerNote'");
    cmdParser.addSingularOption("skip-existing", "If file with the output name exist, skip processing");
    cmdParser.addSingularOption("two-step", "Improved quality style transfer");
    cmdParser.addSingularOption("three-step", "Improved quality style transfer");
    cmdParser.addSingularOption("preserve-colors", "Preserve content colors on the resulting style images");
    cmdParser.addRegularOption("preserve-colors-mode", "Preserve content colors on the resulting style images, ex. post-ycbcr-pixel or pre-hsv-distr");
    const bool optionsParsed = cmdParser.parse(argc, argv);
    unsigned int threadCount = 0u;
    unsigned int benchmark = 0u;

    if (cmdParser.hasOption("threads"))
        threadCount = cmdParser.getOptionAs<uint32_t>("threads");

    if (cmdParser.hasOption("benchmark"))
        benchmark = cmdParser.getOptionAs<uint32_t>("benchmark");

    if (!optionsParsed || 
        !(cmdParser.hasOption("content") || cmdParser.hasOption("folder-content")) || 
        !(cmdParser.hasOption("style") || cmdParser.hasOption("folder-style")) || 
        !(cmdParser.hasOption("output") || cmdParser.hasOption("folder-output")) ||
        !cmdParser.hasOption("vgg") || 
        !cmdParser.hasOption("decoder"))
    {
        cmdParser.printUsage();
        return 0;
    }
    
    decltype(restyleUpdateStyle)* restyleUpdateStyleFunc = nullptr;
    decltype(restyleIsInitialized)* restyleIsInitializedFunc = nullptr;
    decltype(restyleEstimateVRamUsage)* restyleEstimateVRamUsageFunc = nullptr;
    decltype(restyleInitializeWithStyle)* restyleInitializeWithStyleFunc = nullptr;
    decltype(restyleInitializeWithStyleStatistics)* restyleInitializeWithStyleStatisticsFunc = nullptr;
    decltype(restyleCalcAdainMoments)* restyleCalcAdainMomentsFunc = nullptr;
    decltype(restyleForward)* restyleForwardFunc = nullptr;
    decltype(restyleForwardHDR)* restyleForwardHDRFunc = nullptr;
    decltype(restyleDeinitialize)* restyleDeinitializeFunc = nullptr;
    decltype(restyleResizeUsed)* restyleResizeUsedFunc = nullptr;
    decltype(restyleGetVersion)* restyleGetVersionFunc = nullptr;

    #define STRINGIZE_WITH_PREFIX_(pre,x) pre#x
    #define STRINGIZE_L(x) STRINGIZE_WITH_PREFIX_(L,x)

#ifdef _DEBUG
    const std::wstring librestyleName = L"librestyle64." STRINGIZE_L(RESTYLE_LIB_VERSION_FROM_PROPS) L"d.dll";
#else
    const std::wstring librestyleName = L"librestyle64." STRINGIZE_L(RESTYLE_LIB_VERSION_FROM_PROPS) L".dll";
#endif

    const auto handle = LoadLibrary(librestyleName.c_str());

    wprintf_s(L"Loading librestyle: %s\n", librestyleName.c_str());
    if (!handle)
        printf("Failed to load librestyle*.dll\n");

#define CHECK_FUNC(func_handle)\
    {\
        if (!func_handle)\
            printf("Failed to get address of " #func_handle "\n");\
    }

    restyleUpdateStyleFunc = reinterpret_cast<decltype(restyleUpdateStyleFunc)>(GetProcAddress(handle, "restyleUpdateStyle"));
    CHECK_FUNC(restyleUpdateStyleFunc);
    restyleIsInitializedFunc = reinterpret_cast<decltype(restyleIsInitializedFunc)>(GetProcAddress(handle, "restyleIsInitialized"));
    CHECK_FUNC(restyleIsInitializedFunc);
    restyleEstimateVRamUsageFunc = reinterpret_cast<decltype(restyleEstimateVRamUsageFunc)>(GetProcAddress(handle, "restyleEstimateVRamUsage"));
    CHECK_FUNC(restyleEstimateVRamUsageFunc);
    restyleInitializeWithStyleFunc = reinterpret_cast<decltype(restyleInitializeWithStyleFunc)>(GetProcAddress(handle, "restyleInitializeWithStyle"));
    CHECK_FUNC(restyleInitializeWithStyleFunc);
    restyleInitializeWithStyleStatisticsFunc = reinterpret_cast<decltype(restyleInitializeWithStyleStatisticsFunc)>(GetProcAddress(handle, "restyleInitializeWithStyleStatistics"));
    CHECK_FUNC(restyleInitializeWithStyleStatisticsFunc);
    restyleCalcAdainMomentsFunc = reinterpret_cast<decltype(restyleCalcAdainMomentsFunc)>(GetProcAddress(handle, "restyleCalcAdainMoments"));
    CHECK_FUNC(restyleCalcAdainMomentsFunc);
    restyleForwardFunc = reinterpret_cast<decltype(restyleForwardFunc)>(GetProcAddress(handle, "restyleForward"));
    CHECK_FUNC(restyleForwardFunc);
    restyleForwardHDRFunc = reinterpret_cast<decltype(restyleForwardHDRFunc)>(GetProcAddress(handle, "restyleForwardHDR"));
    CHECK_FUNC(restyleForwardHDRFunc);
    restyleDeinitializeFunc = reinterpret_cast<decltype(restyleDeinitializeFunc)>(GetProcAddress(handle, "restyleDeinitialize"));
    CHECK_FUNC(restyleDeinitializeFunc);
    restyleResizeUsedFunc = reinterpret_cast<decltype(restyleResizeUsedFunc)>(GetProcAddress(handle, "restyleResizeUsed"));
    CHECK_FUNC(restyleResizeUsedFunc);
    restyleGetVersionFunc = reinterpret_cast<decltype(restyleGetVersionFunc)>(GetProcAddress(handle, "restyleGetVersion"));
    CHECK_FUNC(restyleGetVersionFunc);

    static bool loadedLibraries = false;

    if (restyleUpdateStyleFunc != nullptr &&
        restyleIsInitializedFunc != nullptr &&
        restyleEstimateVRamUsageFunc != nullptr &&
        restyleInitializeWithStyleFunc != nullptr &&
        restyleInitializeWithStyleStatisticsFunc != nullptr &&
        restyleCalcAdainMomentsFunc != nullptr &&
        restyleForwardFunc != nullptr &&
        restyleDeinitializeFunc != nullptr &&
        restyleResizeUsedFunc != nullptr &&
        restyleGetVersionFunc != nullptr)
    {
        loadedLibraries = true;
    }
    else
    {
        std::cerr << "Failed to resolve functions" << endl;
        return -1;
    }
    const auto vggPath = cmdParser.getOptionAs<std::string>("vgg");
    const auto decoderPath = cmdParser.getOptionAs<std::string>("decoder");

    bool batchMode = false;
    if (cmdParser.hasOption("folder-content") && cmdParser.hasOption("folder-style"))
    {
        batchMode = true;
    }

    auto restyleWrapper = [&restyleForwardHDRFunc, &restyleForwardFunc](void * image, size_t pitch, int numRestyleCycles, bool isHDR)
    {
        if (isHDR)
        {
            for (int styleIter = 0; styleIter < numRestyleCycles; ++styleIter)
            {
                restyleForwardHDRFunc(reinterpret_cast<float *>(image), (uint32_t)pitch);
            }
        }
        else
        {
            for (int styleIter = 0; styleIter < numRestyleCycles; ++styleIter)
            {
                restyleForwardFunc(reinterpret_cast<unsigned char *>(image), (uint32_t)pitch);
            }
        }
    };

    if (batchMode)
    {
        bool preserveColorsPost = false;
        bool preserveColorsPre = false;
        bool preserveColorsDistribution = false;
        imageproc::ColorSpaces preserveColorSpace = imageproc::ColorSpaces::kYCbCr;

        if (cmdParser.hasOption("preserve-colors-mode"))
        {
            std::string mode = cmdParser.getOptionAs<std::string>("preserve-colors-mode");
            darkroom::tolowerInplace(mode);
            std::istringstream issMode(mode);
            std::vector<std::string> settings; 
            for (std::string token; std::getline(issMode, token, '-');)
            {
                settings.push_back(token);
            }
            bool isModeValid = true;
            if (settings.size() != 3)
            {
                isModeValid = false;
            }
            else
            {
                if (settings[0] == "pre")
                {
                    preserveColorsPost = false;
                    preserveColorsPre = true;
                }
                else
                {
                    preserveColorsPost = true;
                    preserveColorsPre = false;
                }

                if (settings[1] == "ycbcr")
                {
                    preserveColorSpace = imageproc::ColorSpaces::kYCbCr;
                }
                else
                {
                    preserveColorSpace = imageproc::ColorSpaces::kHSV;
                }

                if (settings[2] == "distr")
                {
                    preserveColorsDistribution = true;
                }
                else
                {
                    preserveColorsDistribution = false;
                }

                if (preserveColorsPre && preserveColorsDistribution)
                {
                    // This requires rescaling content to match style, but doesn't give any benefit
                    isModeValid = false;
                }
            }

            if (!isModeValid)
            {
                std::cout << "Invalid preserve color mode selected!" << std::endl;
            }
        }
        else if (cmdParser.hasOption("preserve-colors"))
        {
            preserveColorsPost = true;
            preserveColorsPre = false;
            preserveColorsDistribution = false;
            preserveColorSpace = imageproc::ColorSpaces::kHSV;
        }

        const bool preserveColors = preserveColorsPost || preserveColorsPre;

        const bool skipExisting = cmdParser.hasOption("skip-existing");
        const bool threeStep = cmdParser.hasOption("three-step");
        const bool twoStep = cmdParser.hasOption("two-step") || threeStep;

        const auto contentFolder = cmdParser.getOptionAs<std::wstring>("folder-content") + L"\\";
        const auto styleFolder = cmdParser.getOptionAs<std::wstring>("folder-style") + L"\\";
        const auto outputFolder = cmdParser.getOptionAs<std::wstring>("folder-output") + L"\\";

        const bool cycleLastOnly = false;
        const bool cycleFirstOnlyEnabled = false;
        const bool cycleFirstOnly = cycleLastOnly ? false : cycleFirstOnlyEnabled;
        const int cycleNum = 1;

        std::vector<std::wstring> supportedExtensions;
        supportedExtensions.push_back(L"bmp");
        supportedExtensions.push_back(L"png");
        supportedExtensions.push_back(L"jpg");
        supportedExtensions.push_back(L"jpeg");
        supportedExtensions.push_back(L"exr");

        std::vector<std::wstring> contentFilenames = getAllFileNamesWithinFolder(contentFolder, supportedExtensions);
        std::vector<std::wstring> styleFilenames = getAllFileNamesWithinFolder(styleFolder, supportedExtensions);

        const size_t totalWorkItems = contentFilenames.size() * styleFilenames.size();
        size_t processedWorkItems = 0;

        std::map<std::wstring, std::pair<std::vector<float>, std::vector<float>>> styleStatsMap;

        bool isHDR = false;

        for (size_t cidx = 0, cidx_end = contentFilenames.size(); cidx < cidx_end; ++cidx)
        {
            uint32_t wc = 0u, hc = 0u;
            
            const size_t numChannelsContent = 4;
            const size_t numChannelsContentHDR = 4;
            std::wstring contentExt = findExtension(contentFilenames[cidx]);
            
            std::vector<unsigned char> content;
            std::vector<float> contentHDR;
            std::unordered_map<std::string, std::string> tagsHDR;

            if (contentExt.compare(L".exr") == 0)
            {				
                contentHDR = darkroom::loadExr(contentFolder + contentFilenames[cidx], wc, hc, tagsHDR, darkroom::BufferFormat::RGBA32);
                isHDR = true;
            }				
            else
            {
                content = darkroom::loadImage(contentFolder + contentFilenames[cidx], wc, hc, darkroom::BufferFormat::RGBA8);
                
                // WIC PNG loader sometimes produces black RGB line (alpha=255) at the bottom of the image
                bool allBlackRow = true;
                if (contentExt.compare(L".png") == 0)
                {
                    for (uint32_t cnt = 0; (cnt < wc) && allBlackRow; ++cnt)
                    {
                        uint32_t numChannelsToCheck = numChannelsContent;
                        if (numChannelsToCheck > 3)
                            numChannelsToCheck = 3;
                        for (uint32_t ich = 0; ich < numChannelsToCheck; ++ich)
                        {
                            if (content[(cnt+(hc-1)*wc)*numChannelsContent+ich] != 0)
                            {
                                allBlackRow = false;
                                break;
                            }
                        }
                    }

                    // If WIC produced all-balck row of pixels to the bottom, copy previous row
                    if (allBlackRow && (hc > 1))
                    {
                        for (uint32_t cnt = 0; cnt < wc; ++cnt)
                        {
                            for (uint32_t ich = 0; ich < numChannelsContent; ++ich)
                            {
                                content[(cnt+(hc-1)*wc)*numChannelsContent+ich] = content[(cnt+(hc-2)*wc)*numChannelsContent+ich];
                            }
                        }
                    }
                }
            }
            

            std::vector<float> content_colw;
            if (preserveColors)
                content_colw = imageproc::colwertFromRGB(content, wc, hc, numChannelsContent, preserveColorSpace);

            imageproc::ColorStats colorStatsContent[3];
            if (preserveColorsDistribution)
                imageproc::gatherColorStats(content_colw, wc, hc, numChannelsContent, colorStatsContent);

            uint32_t wc2, wc3;
            uint32_t hc2, hc3;
            std::vector<unsigned char> content_ds2, content_ds3;
            std::vector<float> contentHDR_ds2, contentHDR_ds3;
            if (twoStep)
            {
                wc2 = (wc >> 1);
                hc2 = (hc >> 1);

                if (isHDR)
                    contentHDR_ds2 = imageproc::resampleImage(contentHDR, wc, hc, 4, wc2, hc2);
                else
                    content_ds2 = imageproc::resampleImage(content, wc, hc, 4, wc2, hc2);
            }
            if (threeStep)
            {
                wc3 = (wc >> 2);
                hc3 = (hc >> 2);

                if (isHDR)
                {
                    contentHDR_ds3 = imageproc::resampleImage(contentHDR_ds2, wc2, hc2, 4, wc3, hc3);
                    // We don't need /2 content anymore
                    contentHDR_ds2.resize(0);
                }
                else
                {
                    content_ds3 = imageproc::resampleImage(content_ds2, wc2, hc2, 4, wc3, hc3);
                    // We don't need /2 content anymore
                    content_ds2.resize(0);
                }
            }

            for (size_t sidx = 0, sidx_end = styleFilenames.size(); sidx < sidx_end; ++sidx)
            {
                ++processedWorkItems;
                std::cout << "\n\nProcessing " << std::setfill (' ') << std::setw (3) << processedWorkItems << "/" << totalWorkItems << std::endl;

                bool firstCycled = false;

                std::wstring suffix;
                
                if (cycleNum > 1)
                {
                    std::wostringstream ss;
                    ss << cycleNum;
                    suffix += L"_" + ss.str() + L"c";

                    if (cycleFirstOnly)
                        suffix += L"f";
                    if (cycleLastOnly)
                        suffix += L"l";
                }

                if (threeStep)
                    suffix += L"_3s";
                else if (twoStep)
                    suffix += L"_2s";

                if (preserveColors)
                {
                    if (preserveColorsPost)
                        suffix += L"_pc";
                    if (preserveColorsPre)
                        suffix += L"_pcp";
                    if (preserveColorsDistribution)
                        suffix += L"Distr";
                    if (preserveColorSpace == imageproc::ColorSpaces::kHSV)
                        suffix += L"H";
                    if (preserveColorSpace == imageproc::ColorSpaces::kYCbCr)
                        suffix += L"Y";
                }

                std::wstring outFilename = L"o_" + contentFilenames[cidx] + L"_" + styleFilenames[sidx] + suffix + contentExt;
                std::wcout << L"Working on " << outFilename << L"\n" << std::endl;

                if (PathFileExists((outputFolder + outFilename).c_str()))
                {
                    if (skipExisting)
                    {
                        std::cout << "File already present, skipping.. " << std::endl;
                        continue;
                    }
                    else
                    {
                        std::cout << "File already present, overwriting.. " << std::endl;
                    }
                }

                std::pair<std::vector<float>, std::vector<float>> stats;
                std::pair<std::vector<float>, std::vector<float>> stats_ds2;
                std::pair<std::vector<float>, std::vector<float>> stats_ds3;

                auto styleInfoIt = styleStatsMap.find(styleFilenames[sidx]);
                if (styleInfoIt != styleStatsMap.end())
                {
                    stats = styleInfoIt->second;

                    if (twoStep)
                    {
                        auto styleDSInfoIt = styleStatsMap.find(styleFilenames[sidx] + L"_ds");
                        if (styleDSInfoIt != styleStatsMap.end())
                        {
                            stats_ds2 = styleDSInfoIt->second;
                        }
                        else
                        {
                            std::cout << "Troubles loading cache for DS style" << std::endl;
                            continue;
                        }
                    }
                    if (threeStep)
                    {
                        auto styleDS3InfoIt = styleStatsMap.find(styleFilenames[sidx] + L"_ds3");
                        if (styleDS3InfoIt != styleStatsMap.end())
                        {
                            stats_ds3 = styleDS3InfoIt->second;
                        }
                        else
                        {
                            std::cout << "Troubles loading cache for DS3 style" << std::endl;
                            continue;
                        }
                    }
                }
                else
                {
                    uint32_t ws = 0u, hs = 0u;
                    const size_t numChannelsStyle = 3;
                    std::vector<unsigned char> style = darkroom::loadImage(styleFolder + styleFilenames[sidx], ws, hs, darkroom::BufferFormat::RGB8);
                    
                    uint32_t ws2, hs2;
                    uint32_t ws3, hs3;
                    std::vector<unsigned char> style_ds2, style_ds3;

                    if (preserveColorsPre)
                    {
                        imageproc::ColorStats colorStatsStyle[3];

                        std::vector<float> style_colw;
                        style_colw = imageproc::colwertFromRGB(style, ws, hs, numChannelsStyle, preserveColorSpace);

                        if (preserveColorsDistribution)
                        {
                            imageproc::gatherColorStats(style_colw, ws, hs, numChannelsStyle, colorStatsStyle);
                            imageproc::matchColorStatsInImage(style_colw, ws, hs, numChannelsStyle, colorStatsStyle, colorStatsContent);
                        }
                        else
                        {

                        }

                        style = imageproc::colwertToRGB(style_colw, ws, hs, numChannelsStyle, preserveColorSpace);
                    }

                    if (!style.empty())
                    {
                        std::vector<float> mean(1024), var(1024);
                        size_t statisticsSize = 0u;
                        const auto status = restyleCalcAdainMomentsFunc(vggPath.c_str(), style.data(), style.size(), hs, ws, &mean[0], &var[0], &statisticsSize);
                        mean.resize(statisticsSize);
                        var.resize(statisticsSize);
                        stats.first = var;
                        stats.second = mean;

                        //const auto status = restyleCalcAdainMoments(vggPath, style, hs, ws, stats);

                        if (status == Status::kFailedNotEnoughMemory)
                        {
                            std::cout << "Not enough GPU VRAM to callwlate style statistics" << std::endl;
                            continue;
                        }
                        else if (status == Status::kFailed)
                        {
                            std::cout << "Failed to callwlate style statistics" << std::endl;
                            continue;
                        }

                        if (!preserveColorsPre)
                            styleStatsMap.insert(std::make_pair(styleFilenames[sidx], stats));
                    }

                    if (twoStep)
                    {
                        ws2 = (ws >> 1);
                        hs2 = (hs >> 1);
                        style_ds2 = imageproc::resampleImage(style, ws, hs, 3, ws2, hs2);

                        if (!style_ds2.empty())
                        {
                            std::vector<float> mean(1024), var(1024);
                            size_t statisticsSize = 0u;
                            const auto status = restyleCalcAdainMomentsFunc(vggPath.c_str(), style_ds2.data(), style_ds2.size(), hs / 2, ws / 2, &mean[0], &var[0], &statisticsSize);
                            mean.resize(statisticsSize);
                            var.resize(statisticsSize);
                            stats_ds2.first = var;
                            stats_ds2.second = mean;

                            if (status == Status::kFailedNotEnoughMemory)
                            {
                                std::cout << "Not enough GPU VRAM to callwlate style statistics" << std::endl;
                                continue;
                            }
                            else if (status == Status::kFailed)
                            {
                                std::cout << "Failed to callwlate style statistics" << std::endl;
                                continue;
                            }

                            if (!preserveColorsPre)
                                styleStatsMap.insert(std::make_pair(styleFilenames[sidx]+L"_ds", stats_ds2));
                        }
                    }

                    if (threeStep)
                    {
                        ws3 = (ws >> 2);
                        hs3 = (hs >> 2);
                        style_ds3 = imageproc::resampleImage(style_ds2, ws2, hs2, 3, ws3, hs3);

                        if (!style_ds3.empty())
                        {
                            std::vector<float> mean(1024), var(1024);
                            size_t statisticsSize = 0u;
                            const auto status = restyleCalcAdainMomentsFunc(vggPath.c_str(), style_ds3.data(), style_ds3.size(), hs / 4, ws / 4, &mean[0], &var[0], &statisticsSize);
                            mean.resize(statisticsSize);
                            var.resize(statisticsSize);
                            stats_ds3.first = var;
                            stats_ds3.second = mean;

                            if (status == Status::kFailedNotEnoughMemory)
                            {
                                std::cout << "Not enough GPU VRAM to callwlate style statistics" << std::endl;
                                continue;
                            }
                            else if (status == Status::kFailed)
                            {
                                std::cout << "Failed to callwlate style statistics" << std::endl;
                                continue;
                            }

                            if (!preserveColorsPre)
                                styleStatsMap.insert(std::make_pair(styleFilenames[sidx] + L"_ds3", stats_ds3));
                        }
                    }
                }

                if (stats.first.empty() || stats.second.empty())
                {
                    std::cout << "Internal error while callwlating statistics" << std::endl;
                    continue;
                }

                std::vector<unsigned char> contentStylized;
                std::vector<float> contentHDRStylized;

                if (twoStep)
                {
                    std::vector<unsigned char> contentStylizedDS2;
                    std::vector<float> contentHDRStylizedDS2;
                    if (threeStep)
                    {
                        std::vector<unsigned char> contentStylizedDS3 = content_ds3;
                        std::vector<float> contentHDRStylizedDS3 = contentHDR_ds3;

                        if (restyleIsInitializedFunc() == Status::kOk)
                        {
                            restyleDeinitializeFunc();
                        }

                        const auto status = restyleInitializeWithStyleStatisticsFunc(vggPath.c_str(), decoderPath.c_str(), stats.first.data(), stats.second.data(), stats.first.size(), hc/4, wc/4, isHDR, false);

                        void * restyleBuffer = isHDR ? (void *)contentHDRStylizedDS3.data() : (void *)contentStylizedDS3.data();
                        size_t restyleBufferPitch = isHDR ? (wc/4) * numChannelsContentHDR * sizeof(float) : (wc/4) * numChannelsContent * sizeof(unsigned char);
                        restyleWrapper(restyleBuffer, restyleBufferPitch, ((firstCycled && cycleFirstOnly) || cycleLastOnly) ? 1 : cycleNum, isHDR);

                        restyleDeinitializeFunc();
                        firstCycled = true;

                        if (isHDR)
                            contentHDRStylizedDS2 = imageproc::resampleImage(contentHDRStylizedDS3, wc3, hc3, numChannelsContentHDR, wc2, hc2);
                        else
                            contentStylizedDS2 = imageproc::resampleImage(contentStylizedDS3, wc3, hc3, numChannelsContent, wc2, hc2);
                    }
                    else
                    {
                        if (isHDR)
                            contentHDRStylizedDS2 = contentHDR_ds2;
                        else
                            contentStylizedDS2 = content_ds2;
                    }

                    if (restyleIsInitializedFunc() == Status::kOk)
                    {
                        restyleDeinitializeFunc();
                    }

                    const auto status = restyleInitializeWithStyleStatisticsFunc(vggPath.c_str(), decoderPath.c_str(), stats.first.data(), stats.second.data(), stats.first.size(), hc/2, wc/2, isHDR, false);

                    void * restyleBuffer = isHDR ? (void *)contentHDRStylizedDS2.data() : (void *)contentStylizedDS2.data();
                    size_t restyleBufferPitch = isHDR ? (wc/2) * numChannelsContentHDR * sizeof(float) : (wc/2) * numChannelsContent * sizeof(unsigned char);
                    restyleWrapper(restyleBuffer, restyleBufferPitch, ((firstCycled && cycleFirstOnly) || cycleLastOnly) ? 1 : cycleNum, isHDR);

                    restyleDeinitializeFunc();
                    firstCycled = true;

                    if (isHDR)
                        contentHDRStylized = imageproc::resampleImage(contentHDRStylizedDS2, wc2, hc2, numChannelsContentHDR, wc, hc);
                    else
                        contentStylized = imageproc::resampleImage(contentStylizedDS2, wc2, hc2, numChannelsContent, wc, hc);
                }
                else
                {
                    contentHDRStylized = contentHDR;
                    contentStylized = content;
                }

                if (restyleIsInitializedFunc() != Status::kOk)
                {
                    const auto status = restyleInitializeWithStyleStatisticsFunc(vggPath.c_str(), decoderPath.c_str(), stats.first.data(), stats.second.data(), stats.first.size(), hc, wc, isHDR, false);
                }
                else
                {
                    restyleUpdateStyleFunc(stats.first.data(), stats.second.data(), stats.first.size());
                }

                void * restyleBuffer = isHDR ? (void *)contentHDRStylized.data() : (void *)contentStylized.data();
                size_t restyleBufferPitch = isHDR ? wc * numChannelsContentHDR * sizeof(float) : wc * numChannelsContent * sizeof(unsigned char);
                restyleWrapper(restyleBuffer, restyleBufferPitch, (firstCycled && cycleFirstOnly) ? 1 : cycleNum, isHDR);

                if (preserveColorsPost)
                {
                    std::vector<float> contentStylized_colw;
                    contentStylized_colw = imageproc::colwertFromRGB(contentStylized, wc, hc, numChannelsContent, preserveColorSpace);

                    if (preserveColorsDistribution)
                    {
                        imageproc::ColorStats colorStatsStylized[3];
                        imageproc::ColorStats colorStatsOriginal[3];

                        imageproc::gatherColorStats(contentStylized_colw, wc, hc, numChannelsContent, colorStatsStylized);
                        imageproc::gatherColorStats(content_colw, wc, hc, numChannelsContent, colorStatsOriginal);
                    
                        imageproc::matchColorStatsInImage(contentStylized_colw, wc, hc, numChannelsContent, colorStatsStylized, colorStatsOriginal);
                    }
                    else
                    {
                        imageproc::recombineImages(contentStylized_colw, content_colw, wc, hc, numChannelsContent, preserveColorSpace);
                    }

                    contentStylized = imageproc::colwertToRGB(contentStylized_colw, wc, hc, numChannelsContent, preserveColorSpace);
                }

                createDirectoryRelwrsively(outputFolder.c_str());
                if (contentExt == L".bmp")
                {
                    darkroom::saveBmp(contentStylized.data(), outputFolder + outFilename, wc, hc, darkroom::BufferFormat::RGBA8);
                }
                else if (contentExt == L".png")
                {
                    darkroom::savePng(contentStylized.data(), nullptr, outputFolder + outFilename, wc, hc, darkroom::BufferFormat::RGBA8);
                }
                else if (contentExt == L".jpg" || contentExt == L".jpeg")
                {
                    darkroom::saveJpeg(contentStylized.data(), nullptr, outputFolder + outFilename, wc, hc, darkroom::BufferFormat::RGBA8);
                }
                else if (contentExt == L".exr")
                {
                    darkroom::saveExr(contentHDRStylized.data(), nullptr, outputFolder + outFilename, wc, hc, darkroom::BufferFormat::RGBA32, tagsHDR);
                }
            }

            restyleDeinitializeFunc();
        }
    }
    else
    {
        const auto contentPath = cmdParser.getOptionAs<std::wstring>("content");
        const auto stylePath = cmdParser.getOptionAs<std::wstring>("style");
        const auto outputPath = cmdParser.getOptionAs<std::wstring>("output");

        uint32_t wc = 0u, hc = 0u, ws = 0u, hs = 0u;
        auto content = darkroom::loadImage(contentPath, wc, hc, darkroom::BufferFormat::RGBA8);
        const auto style = darkroom::loadImage(stylePath, ws, hs, darkroom::BufferFormat::RGB8);

        if (content.empty() || style.empty())
            return -1;

        size_t model_vram = 0;
        size_t forward_vram = 0;

        if (restyleIsInitializedFunc() == Status::kOk)
        {
            cout << "restyleIsInitialized" << endl;
        }

        restyleEstimateVRamUsageFunc(vggPath.c_str(), decoderPath.c_str(), hs, ws, hc, wc, model_vram, forward_vram, false);
        std::cout << "Estimation: " << (model_vram / (1024 * 1024)) << " Mb for model " << (forward_vram / (1024 * 1024)) << " Mb for forward pass\n";
        std::cout << "Estimation: " << model_vram << " b for model " << forward_vram << " b for forward pass\n";

        const Status status = restyleInitializeWithStyleFunc(vggPath.c_str(), decoderPath.c_str(), style.data(), style.size(), hs, ws, hc, wc, false, false);

        if (status == Status::kFailed)
        {
            std::cout << "Failed to initialize style transfer network" << std::endl;
            return -1;
        }
        if (status == Status::kFailedNotEnoughMemory)
        {
            std::cout << "Not enough memory to initialize style transfer network" << std::endl;
            return -1;
        }

        if (benchmark == 0u)
        {
            restyleForwardFunc(&content[0], wc * 4);

            if (content.empty())
                return -1;
            // styleTransfer function also swizzles R and B channels, so this is why BGR8 is used here
            darkroom::saveJpeg(content.data(), nullptr, outputPath, wc, hc, darkroom::BufferFormat::RGBA8);
        }
        else
        {
            const auto start = std::chrono::system_clock::now();

            for (auto i = 0u; i < benchmark; ++i)
                restyleForwardFunc(&content[0], wc * 4);

            const auto end = std::chrono::system_clock::now();
            const auto duration = end - start;

            const auto total = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
            std::cout << "Total time is " << total << " ms" << std::endl;
            std::cout << "Average inference time is  " << float(total) / benchmark << " ms" << std::endl;
        }

        restyleDeinitializeFunc();
    }

    return 0;
}
#endif
