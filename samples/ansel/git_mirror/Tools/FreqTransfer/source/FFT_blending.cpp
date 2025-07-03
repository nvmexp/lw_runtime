#include <stdio.h>
#include <stdlib.h>
#include <locale>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#include <thread>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

#define DBG_ALLOW_FFTW			0
#define DBG_ALLOW_LWFFT			0

#define DBG_VERBOSE				0


#if (DBG_ALLOW_FFTW == 1)
#	pragma comment (lib, "libfftw3f-3.lib")
#endif
#if (DBG_ALLOW_LWFFT == 1)
#	pragma comment (lib, "lwdart_static.lib")
#	pragma comment (lib, "lwfft.lib")
#endif

// Darkroom
#include "darkroom/Bmp.h"
#include "darkroom/Exr.h"
#include "darkroom/Png.h"
#include "darkroom/Jpeg.h"
#include "darkroom/Metadata.h"
#include "darkroom/StringColwersion.h"

#define USE_PADDING_FFT			0

#define DBG_OUT_PADDING			0
#define DBG_OUTPUT_SPECTRUM		0
#define DBG_OUTPUT_SPECTRUM_LOG	1

#include "timer.h"
#include "helpers_common.h"
#include "fftchannel.h"
#include "fftchannel_kissfft.h"

#if (DBG_ALLOW_FFTW == 1)
#include "fftchannel_fftw.h"
#endif
#if (DBG_ALLOW_LWFFT == 1)
#include "fftchannel_lwfft.h"
#endif

#define FFT_LIB_KISSFFT			1
#define FFT_LIB_FFTW			2
#define FFT_LIB_LWFFT			3
#define FFT_LIB					FFT_LIB_KISSFFT

#define SAFE_DELETE(x)	if (x) { delete x; (x) = nullptr; }

namespace windows
{
    std::wstring findExtension(const std::wstring& filename)
    {
        size_t filenameLen = filename.size();

        size_t extPos = filename.rfind(L'.');
        if (extPos == std::wstring::npos)
            return std::wstring();

        extPos += 1;

        const size_t maxExtensionLen = 4;
        if (filenameLen - extPos > maxExtensionLen)
        {
            // Some fancy extension, dunno how to load that
            return std::wstring();
        }
        std::wstring extLowerCase = filename.substr(extPos);
        darkroom::tolowerInplace(extLowerCase);

        return extLowerCase;
    }

    class MTSection
    {
    public:

        CRITICAL_SECTION m_critSection;
        void begin()
        {
            EnterCriticalSection(&m_critSection);
        }
        void end()
        {
            LeaveCriticalSection(&m_critSection);
        }

        void init()
        {
            InitializeCriticalSection(&m_critSection);
        }
        void deinit()
        {
            DeleteCriticalSection(&m_critSection);
        }
    };

    class MTSectionAuto
    {
    public:

        MTSection * m_section = nullptr;
        MTSectionAuto(MTSection * sect):
            m_section(sect)
        {
            if (m_section)
                m_section->begin();
        }

        ~MTSectionAuto()
        {
            if (m_section)
                m_section->end();
        }
    };
}

windows::MTSection mtProgress;

void initMultithreading()
{
    mtProgress.init();
}
void deinitMultithreading()
{
    mtProgress.deinit();
}

int g_totalTileCount = 0;
float g_totalProgress = 0.0f;
int reportProgress(float progressFraction)
{
    windows::MTSectionAuto mtProgressAuto(&mtProgress);

    ++g_totalTileCount;
    g_totalProgress += progressFraction;

    printf("%4.1f%%\n", g_totalProgress * 100);

    return g_totalTileCount;
}

unsigned char saturate_uc(float val)
{
    if (val < 0)
        return 0;
    if (val > 255)
        return 255;

    return (unsigned char)val;
}

template <typename T>
T mymin(T a, T b) { return (a < b) ? a : b; }

#if (DBG_OUTPUT_SPECTRUM_LOG == 1)
unsigned char getOutputSpectrumVal(float val)
{
    const float c = 255.0f / logf(1 + 255.0f);
    return saturate_uc(c*logf(1 + fabsf(val)));
}
#else
unsigned char getOutputSpectrumVal(float val)
{
    return (unsigned char)(val);
}
#endif

template <typename T>
inline T sqr(T val) { return val*val; }

struct imageDesc
{
    std::vector<float> dataInFloat;
    std::vector<ubyte> dataInUByte;
    uint dataInW;
    uint dataInH;
    ubyte dataInC;
};

std::wstring buildFilename(const std::wstring & inFilename, const std::wstring & suffix)
{
    std::wstring outFilename;
    size_t dotPos = inFilename.find_last_of(L'.');

    // Try to infer output filename
    std::wstring filename, ext;
    filename = inFilename.substr(0, dotPos);
    ext = inFilename.substr(dotPos, std::string::npos);

    outFilename = filename + suffix + ext;

    return outFilename;
}

class ChannelPair
{
public:
    FFTChannel * regular = nullptr;
    FFTChannel * highres = nullptr;

    ChannelPair(FFTChannel * reg, FFTChannel * hr):
        regular(reg),
        highres(hr)
    {
    }
};

// Use these two functions ONLY when you use processData directly
//	fior general case, use initFFTTiledProcessing/deinitFFTTiledProcessing
void initFFTProcessing(FFTChannelPersistentData ** ppPersistentDataRegular, FFTChannelPersistentData ** ppPersistentDataHighres)
{
#if (FFT_LIB == FFT_LIB_FFTW)
    *ppPersistentDataRegular = FFTChannelFFTW::allocatePersistentData();
    *ppPersistentDataHighres = FFTChannelFFTW::allocatePersistentData();
#elif (FFT_LIB == FFT_LIB_LWFFT)
    *ppPersistentDataRegular = FFTChannelLwFFT::allocatePersistentData();
    *ppPersistentDataHighres = FFTChannelLwFFT::allocatePersistentData();
#else
    *ppPersistentDataRegular = FFTChannelKissFFT::allocatePersistentData();
    *ppPersistentDataHighres = FFTChannelKissFFT::allocatePersistentData();
#endif
}

void deinitFFTProcessing(FFTChannelPersistentData * pPersistentDataRegular, FFTChannelPersistentData * pPersistentDataHighres)
{
#if (FFT_LIB == FFT_LIB_FFTW)
    FFTChannelFFTW::freePersistentData(pPersistentDataRegular);
    FFTChannelFFTW::freePersistentData(pPersistentDataHighres);
#elif (FFT_LIB == FFT_LIB_LWFFT)
    FFTChannelLwFFT::freePersistentData(pPersistentDataRegular);
    FFTChannelLwFFT::freePersistentData(pPersistentDataHighres);
#else
    FFTChannelKissFFT::freePersistentData(pPersistentDataRegular);
    FFTChannelKissFFT::freePersistentData(pPersistentDataHighres);
#endif
}

class FFTPersistentData
{
public:

    FFTChannelPersistentData * pPersistentDataRegular = nullptr;
    FFTChannelPersistentData * pPersistentDataHighres = nullptr;

    std::vector<unsigned char> regularPart;
    std::vector<unsigned char> highresPart;
    std::vector<unsigned char> outputPart;
};

void initFFTTiledProcessing(FFTPersistentData * pPersistentData)
{
    initFFTProcessing(&pPersistentData->pPersistentDataRegular, &pPersistentData->pPersistentDataHighres);
}

void deinitFFTTiledProcessing(FFTPersistentData * pPersistentData)
{
    deinitFFTProcessing(pPersistentData->pPersistentDataRegular, pPersistentData->pPersistentDataHighres);
}

#include "processing_fast.h"

/*
    This function performs FFT processing of a single tile. The function performs necessary padding and data copying,
    the actual FFT is happening in the processData* function call.

    This function doesn't upscale regular (downscaled) image, and as a consequence, cannot perform efficiency padding,
    since highres/lowres ratio should be integer, and offsets/sizes should map directly for correct FFT results.
    Thus the function is faster on factor-independent FFTs (like lwFFT for example), but might experience significant
    slowdowns on factor-dependent FFTs (like KissFFT), although best case is still quite fast.

    Memory consumption is less than for version with effeectiveness padding, and also it is faster in factor-independent
    FFT implementations.

    alpha is the interpolation/blending parameter (the higher it is - the less affect transfer will have)
    maxFreq is the frequency transfer distance, default is 0.5 (Nyquist freq)
    interpShapeType is the area shape of transferred spectrum, rectangle - biggest area for certain
        maxFreq, circle - safest transfer, although requires nearly 1:1 aspect of the tiles
*/
template <typename T>
void processTiling(
        FFTPersistentData * pPersistentData,
        T * regular, int regularSizeW, int regularSizeH,
        T * highres, int highresSizeW, int highresSizeH,
        int highresTileOffsetW, int highresTileOffsetH, int highresTileSizeW, int highresTileSizeH,
        T * highresTileFixed, float alpha, float maxFreq,
        FreqTransferInterpShape interpShapeType = FreqTransferInterpShape::kCIRCLE
        )
{
    const uint numChannels = 3;

    // Tiling
    // Lwrrently regular has symmetrical padding
    //	and highres could have asymmetrical padding due to potential sub-regular-pixel offsets
    const uint tilePaddingInRegular = 8;

    // Callwlate tile size
    const uint tileOutW = highresTileSizeW;
    const uint tileOutH = highresTileSizeH;

    uint highresMultW = (uint)ceil(highresSizeW / (float)regularSizeW);
    uint highresMultH = (uint)ceil(highresSizeH / (float)regularSizeH);


    // Regular sizes/offsets callwlation
    ///////////////////////////////////////////////////////////////////////////////
    // Callwlating regular tile box
    //	the box should be callwlated taking into account that highres tile could have a box
    //	that doesn't directly maps onto regular pixel grid, thus we need to be conservative
    const uint regularTileOutOffsetW = (uint)floor(highresTileOffsetW / (float)highresMultW);
    const uint regularTileOutOffsetH = (uint)floor(highresTileOffsetH / (float)highresMultH);
    const uint regularTileOutSizeW = (uint)ceil((highresTileSizeW + highresTileOffsetW) / (float)highresMultW) - regularTileOutOffsetW;
    const uint regularTileOutSizeH = (uint)ceil((highresTileSizeH + highresTileOffsetH) / (float)highresMultH) - regularTileOutOffsetH;

    // Paddings for the tile in regular space
    uint regularTilePaddingOffsetW = tilePaddingInRegular;
    uint regularTilePaddingOffsetH = tilePaddingInRegular;
    uint regularTilePaddingSizeW = tilePaddingInRegular;
    uint regularTilePaddingSizeH = tilePaddingInRegular;

    // We want to make our regular padded tile offset&size even
    //	this will guarantee that highres tile will also be even
    //	this subsequently will guarantee that none of the FFT libs will fail
    if ((regularTileOutOffsetW - regularTilePaddingOffsetW)&1)
    {
        ++regularTilePaddingOffsetW;
    }
    if ((regularTileOutOffsetH - regularTilePaddingOffsetH)&1)
    {
        ++regularTilePaddingOffsetH;
    }
    if ((regularTileOutSizeW + regularTilePaddingOffsetW + regularTilePaddingSizeW)&1)
    {
        ++regularTilePaddingSizeW;
    }
    if ((regularTileOutSizeH + regularTilePaddingOffsetH + regularTilePaddingSizeH)&1)
    {
        ++regularTilePaddingSizeH;
    }

    const uint regularTileOutWPad = regularTileOutSizeW + regularTilePaddingOffsetW + regularTilePaddingSizeW;
    const uint regularTileOutHPad = regularTileOutSizeH + regularTilePaddingOffsetH + regularTilePaddingSizeH;

    size_t regularTileOutChannelSize = regularTileOutWPad*regularTileOutHPad;

    pPersistentData->regularPart.resize(regularTileOutChannelSize*numChannels*sizeof(float));
    float * regularPart = reinterpret_cast<float *>(&pPersistentData->regularPart[0]);
    float * regularPartR = regularPart;
    float * regularPartG = regularPartR+regularTileOutChannelSize;
    float * regularPartB = regularPartG+regularTileOutChannelSize;


    // Highres sizes/offsets callwlation
    ///////////////////////////////////////////////////////////////////////////////
    // In case highres tile doesn't fit onto pixel borders f regular, we need additional padding
    //	this padding could be different on each edge of the highres tile
    const int highresAdditionalPaddingOffsetW = highresTileOffsetW - regularTileOutOffsetW * highresMultW;
    const int highresAdditionalPaddingOffsetH = highresTileOffsetH - regularTileOutOffsetH * highresMultH;
    const int highresAdditionalPaddingSizeW = (regularTileOutSizeW+regularTileOutOffsetW) * highresMultW - (highresTileSizeW+highresTileOffsetW);
    const int highresAdditionalPaddingSizeH = (regularTileOutSizeH+regularTileOutOffsetH) * highresMultH - (highresTileSizeH+highresTileOffsetH);

    assert(highresAdditionalPaddingOffsetW >= 0);
    assert(highresAdditionalPaddingOffsetH >= 0);
    assert(highresAdditionalPaddingSizeW >= 0);
    assert(highresAdditionalPaddingSizeH >= 0);

    const uint highresTilePaddingOffsetW = regularTilePaddingOffsetW*highresMultW + highresAdditionalPaddingOffsetW;
    const uint highresTilePaddingOffsetH = regularTilePaddingOffsetH*highresMultH + highresAdditionalPaddingOffsetH;
    const uint highresTilePaddingSizeW = regularTilePaddingSizeW*highresMultW + highresAdditionalPaddingSizeW;
    const uint highresTilePaddingSizeH = regularTilePaddingSizeH*highresMultH + highresAdditionalPaddingSizeH;

    const uint tileOutWPad = tileOutW + highresTilePaddingOffsetW + highresTilePaddingSizeW;
    const uint tileOutHPad = tileOutH + highresTilePaddingOffsetH + highresTilePaddingSizeH;

    size_t tileOutChannelSize = tileOutWPad*tileOutHPad;

    pPersistentData->highresPart.resize(tileOutChannelSize*numChannels*sizeof(float));
    float * highresPart = reinterpret_cast<float *>(&pPersistentData->highresPart[0]);
    float * highresPartR = highresPart;
    float * highresPartG = highresPartR+tileOutChannelSize;
    float * highresPartB = highresPartG+tileOutChannelSize;

    // Output 
    pPersistentData->outputPart.resize(tileOutChannelSize*numChannels*sizeof(float));
    float * tileOut = reinterpret_cast<float *>(&pPersistentData->outputPart[0]);
    memset(tileOut, 0, tileOutChannelSize*numChannels*sizeof(float));


    // Copying data and introduce padding
    ///////////////////////////////////////////////////////////////////////////////

#if (DBG_VERBOSE == 1)
    printf("Copying the input data to channels\n");
#endif
    // Copy highres part
    for (int x = -(int)highresTilePaddingOffsetW; x < (int)(tileOutW + highresTilePaddingSizeW); ++x)
    {
        for (int y = -(int)highresTilePaddingOffsetH; y < (int)(tileOutH + highresTilePaddingSizeH); ++y)
        {
            int padX = x + highresTilePaddingOffsetW;
            int padY = y + highresTilePaddingOffsetH;

            int imageX = x + highresTileOffsetW;
            int imageY = y + highresTileOffsetH;

            if (imageX < 0) imageX = 0;
            if (imageX >= highresSizeW) imageX = highresSizeW - 1;

            if (imageY < 0) imageY = 0;
            if (imageY >= highresSizeH) imageY = highresSizeH - 1;

            highresPartB[padX+padY*tileOutWPad] = highres[(imageX + imageY*highresSizeW)*numChannels+2];
            highresPartG[padX+padY*tileOutWPad] = highres[(imageX + imageY*highresSizeW)*numChannels+1];
            highresPartR[padX+padY*tileOutWPad] = highres[(imageX + imageY*highresSizeW)*numChannels  ];
        }
    }

    // Copy regular part
    for (int x = -(int)regularTilePaddingOffsetW; x < (int)(regularTileOutSizeW + regularTilePaddingSizeW); ++x)
    {
        for (int y = -(int)regularTilePaddingOffsetH; y < (int)(regularTileOutSizeH + regularTilePaddingSizeH); ++y)
        {
            int padX = x + regularTilePaddingOffsetW;
            int padY = y + regularTilePaddingOffsetH;

            int imageX = x + regularTileOutOffsetW;
            int imageY = y + regularTileOutOffsetH;

            if (imageX < 0) imageX = 0;
            if (imageX >= regularSizeW) imageX = regularSizeW - 1;

            if (imageY < 0) imageY = 0;
            if (imageY >= regularSizeH) imageY = regularSizeH - 1;

            int regularIdx = (imageX + imageY*regularSizeW);

            regularPartB[padX+padY*regularTileOutWPad] = regular[regularIdx*numChannels+2];
            regularPartG[padX+padY*regularTileOutWPad] = regular[regularIdx*numChannels+1];
            regularPartR[padX+padY*regularTileOutWPad] = regular[regularIdx*numChannels  ];
        }
    }

#if (DBG_VERBOSE == 1)
    printf("done.\n");
#endif


    // Fix up the highres tile
    ///////////////////////////////////////////////////////////////////////////////

    processDataFast(
        pPersistentData->pPersistentDataRegular, regularPart, (int)regularTileOutWPad, (int)regularTileOutHPad, 1.0,
        pPersistentData->pPersistentDataHighres, highresPart, (int)tileOutWPad, (int)tileOutHPad,
        tileOut, alpha, maxFreq, interpShapeType
        );

    // Copy processed part
    ///////////////////////////////////////////////////////////////////////////////

    copyFixedData(
        tileOut, tileOutWPad, tileOutHPad, highresTilePaddingOffsetW, highresTilePaddingOffsetH, highresTileSizeW, highresTileSizeH,
        highresTileFixed, highresSizeW, highresSizeH, highresTileOffsetW, highresTileOffsetH
        );
}

std::vector<ubyte> loadImage8bpc(const std::wstring & filename, unsigned int & w, unsigned int & h, darkroom::ImageMetadata & metadata)
{
    metadata = darkroom::ImageMetadata();

    std::wstring ext = windows::findExtension(filename);
    if (ext == L"bmp")
    {
        std::vector<unsigned char> data = darkroom::loadBmp(filename, w, h);

        // colwert BGR8 -> RGB8
        for (size_t idx = 0, idxEnd = (size_t)w*h; idx < idxEnd; ++idx)
        {
            unsigned char tmp = data[idx*3];
            data[idx*3  ] = data[idx*3+2];
            data[idx*3+2] = tmp;
        }

        return data;
    }
    else if (ext == L"jpg" || ext == L"jpeg")
    {
        std::unordered_map<uint16_t, std::string> tagsJPEG;
        std::vector<char> xmpPacket;
        
        std::vector<unsigned char> data = darkroom::loadJpeg(filename, w, h, darkroom::BufferFormat::RGB8, tagsJPEG, xmpPacket);
        darkroom::colwertToImageMetadata(tagsJPEG, xmpPacket, metadata);

        return data;
    }
    else if (ext == L"png")
    {
        std::unordered_map<std::string, std::string> tagsPNG;
        std::vector<char> xmpPacket;

        std::vector<unsigned char> data = darkroom::loadPng(filename, w, h, darkroom::BufferFormat::RGB8, tagsPNG);
        darkroom::colwertToImageMetadata(tagsPNG, metadata);

        return data;
    }
    else
    {
        printf("unsupported regular capture format");
    }

    return std::vector<ubyte>();
}

struct CmdLineOptions
{
    std::wstring inRegularFilename;
    std::wstring inHighresFilename;
    std::wstring outFilename;
    std::wstring outRegularSpectrumFilename;
    std::wstring outHighresSpectrumFilename;
    double alpha = 1.0;
    float maxFreq = 0.5;

    bool tilesEnable = true;
    int tileWidth = 2048;
    int tileHeight = 2048;

    uint32_t nthreads = 8;

    FreqTransferInterpShape interpShapeType = FreqTransferInterpShape::kCIRCLE;
} g_cmdLineOptions;

void processCmdLineArgs(int argc, char * argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::string lwrArg = argv[i];
        darkroom::tolowerInplace(lwrArg);

        char * nextArg = (i < argc-1) ? argv[i+1] : nullptr;

        if (lwrArg == "-in0" || lwrArg == "-small")
        {
            if (nextArg)
            {
                g_cmdLineOptions.inRegularFilename = darkroom::getWstrFromUtf8(nextArg);
                printf("Input smaller file: %s\n", nextArg);
            }
            else
            {
                printf("Error in cmd line argument \"-in0\" - no file specified\n");
            }
        }
        else if (lwrArg == "-in1" || lwrArg == "-big")
        {
            if (nextArg)
            {
                g_cmdLineOptions.inHighresFilename = darkroom::getWstrFromUtf8(nextArg);
                printf("Input bigger file: %s\n", nextArg);
            }
            else
            {
                printf("Error in cmd line argument \"-in1\" - no file specified\n");
            }
        }
        else if (lwrArg == "-out")
        {
            if (nextArg)
            {
                g_cmdLineOptions.outFilename = darkroom::getWstrFromUtf8(nextArg);
                printf("Output file: %s\n", nextArg);
            }
            else
            {
                printf("Error in cmd line argument \"-out\" - no file specified\n");
            }
        }
        else if (lwrArg == "-a" || lwrArg == "-alpha")
        {
            if (nextArg)
            {
                g_cmdLineOptions.alpha = atof(nextArg);
                printf("Alpha is set to %f\n", g_cmdLineOptions.alpha);
            }
            else
            {
                printf("Error in cmd line argument \"-alpha\" - no coefficient specified\n");
            }
        }
        else if (lwrArg == "-maxfreq")
        {
            if (nextArg)
            {
                g_cmdLineOptions.maxFreq = (float)atof(nextArg);
                printf("Max. freq is set to %f\n", g_cmdLineOptions.maxFreq);
            }
            else
            {
                printf("Error in cmd line argument \"-maxfreq\" - no coefficient specified\n");
            }
        }
        else if (lwrArg == "-full")
        {
            g_cmdLineOptions.tilesEnable = false;
            printf("Using full-frame processing. Faster but requires significantly more memory.\n");
        }
        else if (lwrArg == "-shape")
        {
            if (nextArg)
            {
                printf("Shape type specified: %s\n", nextArg);
                std::string strNextArg = nextArg;
                if (strNextArg == "circle")
                {
                    g_cmdLineOptions.interpShapeType = FreqTransferInterpShape::kCIRCLE;
                }
                else if (strNextArg == "rect" || strNextArg == "rectangle")
                {
                    g_cmdLineOptions.interpShapeType = FreqTransferInterpShape::kRECTANGLE;
                }
                else if (strNextArg == "ellipse")
                {
                    g_cmdLineOptions.interpShapeType = FreqTransferInterpShape::kELLIPSE;
                }
                else
                {
                    printf("Unkown shape type!\n");
                }
            }
            else
            {
                printf("Error in cmd line argument \"-shape\" - no shape specified\n");
            }
        }
        else if (lwrArg == "-nthreads")
        {
            if (nextArg)
            {
                g_cmdLineOptions.nthreads = atoi(nextArg);
                printf("Max # threads is set to %d\n", g_cmdLineOptions.nthreads);
            }
            else
            {
                printf("Error in cmd line argument \"-nthreads\" - no number specified\n");
            }
        }
        else if (lwrArg == "-tilew")
        {
            if (nextArg)
            {
                g_cmdLineOptions.tileWidth = atoi(nextArg);
                printf("Tile width = %d\n", g_cmdLineOptions.tileWidth);
            }
            else
            {
                printf("Error in cmd line argument \"-tilew\" - no number specified\n");
            }
        }
        else if (lwrArg == "-tileh")
        {
            if (nextArg)
            {
                g_cmdLineOptions.tileHeight = atoi(nextArg);
                printf("Tile height = %d\n", g_cmdLineOptions.tileHeight);
            }
            else
            {
                printf("Error in cmd line argument \"-tileh\" - no number specified\n");
            }
        }
        else if (lwrArg == "-?" || lwrArg == "-h" || lwrArg == "-help")
        {
            printf("Command line arguments:\n");
            printf("-alpha <fp number>         - set blending coefficient\n");
            printf("-in0 <filename>            - input filename, smaller picture\n");
            printf("-small <filename>          - same as \"-in0\"\n");
            printf("-in1 <filename>            - input filename, larger picture\n");
            printf("-big <filename>            - same as \"-in1\"\n");
            printf("-out <filename>            - output filename\n");
            printf("-full                      - full-picture processing (requires more memory)\n");
            printf("-shape                     - blending shape, circle/rect/ellipse\n");
            printf("-nthreads                  - max. amount of threads\n");
            printf("-tilew, -tileh             - target tile size if not in full-picture mode\n");
        }
    }
}

int main(int argc, char ** argv)
{
    // Setting up cmd line parameters
    if (argc == 1)
    {
        // TODO: show list of options here
        printf("Use -? command line argument to get a list of available options\n");
        return 0;
    }
    processCmdLineArgs(argc, argv);

    if (g_cmdLineOptions.inRegularFilename.size() == 0)
    {
        printf("ERROR: filename of a smaller file is not set, use the \"-in0\" option (or see the help with \"-?\" command line option)\n");
        return 0;
    }
    if (g_cmdLineOptions.inHighresFilename.size() == 0)
    {
        printf("ERROR: filename of a bigger file is not set, use the \"-in1\" option (or see the help with \"-?\" command line option)\n");
        return 0;
    }

    if (g_cmdLineOptions.outFilename.size() == 0)
    {
        g_cmdLineOptions.outFilename = buildFilename(g_cmdLineOptions.inHighresFilename, L"_out");
        printf("Output filename inferred: %ls\n", g_cmdLineOptions.outFilename.c_str());
    }
#if (DBG_OUTPUT_SPECTRUM == 1)
    // Inferring spectrum filenames, if needed
    {
        g_cmdLineOptions.outRegularSpectrumFilename = buildFilename(g_cmdLineOptions.inRegularFilename, L"_spec");
        printf("Output smaller spectrum inferred: %ls\n", g_cmdLineOptions.outRegularSpectrumFilename.c_str());
        g_cmdLineOptions.outHighresSpectrumFilename = buildFilename(g_cmdLineOptions.inHighresFilename, L"_spec");
        printf("Output bigger spectrum inferred: %ls\n", g_cmdLineOptions.inHighresFilename.c_str());
    }
#endif

    std::wstring regularExt = windows::findExtension(g_cmdLineOptions.inRegularFilename);

    imageDesc regular;
    imageDesc highres;

    enum class ProcessingType
    {
        kUBYTE = 0,
        kFLOAT,
        KNUM_ENTRIES
    };
    ProcessingType processingType;

    darkroom::ImageMetadata imageMetadata;

    uint dataOutW;
    uint dataOutH;
    float * dataOut = nullptr;
    unsigned char * dataOut8bit = nullptr;
    if (regularExt == L"exr")
    {
        processingType = ProcessingType::kFLOAT;

        regular.dataInFloat = darkroom::loadExr(g_cmdLineOptions.inRegularFilename, regular.dataInW, regular.dataInH, std::unordered_map<std::string, std::string>());
        if (regular.dataInFloat.size() == 0)
        {
            printf("failed to read the file \"%ls\"\n", g_cmdLineOptions.inRegularFilename.c_str());
            exit(EXIT_FAILURE);
        }
        regular.dataInC = (unsigned char)(regular.dataInFloat.size() / (regular.dataInW*regular.dataInH));

        highres.dataInFloat = darkroom::loadExr(g_cmdLineOptions.inHighresFilename, highres.dataInW, highres.dataInH, std::unordered_map<std::string, std::string>());
        if (highres.dataInFloat.size() == 0)
        {
            printf("failed to read the file \"%ls\"\n", g_cmdLineOptions.inHighresFilename.c_str());
            exit(EXIT_FAILURE);
        }
        highres.dataInC = (unsigned char)(highres.dataInFloat.size() / (highres.dataInW*highres.dataInH));

        dataOutW = highres.dataInW;
        dataOutH = highres.dataInH;
        size_t dataOutChannelSize = dataOutW*dataOutH;
        dataOut = (float *)malloc(dataOutChannelSize*highres.dataInC*sizeof(float));
        memset(dataOut, 0, dataOutChannelSize*highres.dataInC*sizeof(float));
    }
    else
    {
        processingType = ProcessingType::kUBYTE;

        regular.dataInUByte = loadImage8bpc(g_cmdLineOptions.inRegularFilename, regular.dataInW, regular.dataInH, imageMetadata);
        if (regular.dataInUByte.size() == 0)
        {
            printf("failed to read the file \"%ls\"\n", g_cmdLineOptions.inRegularFilename.c_str());
            exit(EXIT_FAILURE);
        }
        regular.dataInC = (unsigned char)(regular.dataInUByte.size() / (regular.dataInW*regular.dataInH));

        highres.dataInUByte = loadImage8bpc(g_cmdLineOptions.inHighresFilename, highres.dataInW, highres.dataInH, imageMetadata);
        if (highres.dataInUByte.size() == 0)
        {
            printf("failed to read the file \"%ls\"\n", g_cmdLineOptions.inHighresFilename.c_str());
            exit(EXIT_FAILURE);
        }
        highres.dataInC = (unsigned char)(highres.dataInUByte.size() / (highres.dataInW*highres.dataInH));

        dataOutW = highres.dataInW;
        dataOutH = highres.dataInH;
        size_t dataOut8bitChannelSize = dataOutW*dataOutH;
        dataOut8bit = (unsigned char *)malloc(dataOut8bitChannelSize*highres.dataInC*sizeof(unsigned char));
        memset(dataOut8bit, 0, dataOut8bitChannelSize*highres.dataInC*sizeof(unsigned char));
    }

    if (g_cmdLineOptions.tilesEnable && (g_cmdLineOptions.tileWidth == -1 || g_cmdLineOptions.tileHeight == -1))
    {
        g_cmdLineOptions.tileWidth = regular.dataInW;
        g_cmdLineOptions.tileHeight = regular.dataInH;
    }

    int numChannels = highres.dataInC;

    Timer perfTimer;
    perfTimer.start();

    const bool useTiling = g_cmdLineOptions.tilesEnable;
    if (useTiling)
    {
        const bool forceSingleThreaded = (g_cmdLineOptions.nthreads == 0);

        if (forceSingleThreaded)
        {
            uint windowSizeW = g_cmdLineOptions.tileWidth;
            uint windowSizeH = g_cmdLineOptions.tileHeight;

            uint multW = (uint)ceil(highres.dataInW / (float)windowSizeW);
            uint multH = (uint)ceil(highres.dataInH / (float)windowSizeH);

            FFTPersistentData persistentData;
            initFFTTiledProcessing(&persistentData);

            int tileCounter = 0;
            for (uint cntw = 0; cntw < multW; ++cntw)
            {
                uint offsetW = cntw * windowSizeW;

                if (offsetW >= highres.dataInW)
                    continue;

                uint tileSizeW = mymin(windowSizeW, highres.dataInW - offsetW);

                for (uint cnth = 0; cnth < multH; ++cnth)
                {
                    tileCounter++;

                    uint offsetH = cnth * windowSizeH;

                    if (offsetH >= highres.dataInH)
                        continue;

                    uint tileSizeH = mymin(windowSizeH, highres.dataInH - offsetH);

                    printf("Tile %d / %d : (%d, %d)+(%d, %d)\n", tileCounter, multW*multH, offsetW, offsetH, tileSizeW, tileSizeH);

                    if (processingType == ProcessingType::kFLOAT)
                    {
                        assert(dataOut != nullptr);
                        processTilingFast(
                            &persistentData,
                            &regular.dataInFloat[0], regular.dataInW, regular.dataInH,
                            &highres.dataInFloat[0], highres.dataInW, highres.dataInH,
                            offsetW, offsetH, tileSizeW, tileSizeH,
                            dataOut, (float)g_cmdLineOptions.alpha, g_cmdLineOptions.maxFreq, g_cmdLineOptions.interpShapeType
                            );
                    }
                    else if (processingType == ProcessingType::kUBYTE)
                    {
                        assert(dataOut8bit != nullptr);
                        processTilingFast(
                            &persistentData,
                            &regular.dataInUByte[0], regular.dataInW, regular.dataInH,
                            &highres.dataInUByte[0], highres.dataInW, highres.dataInH,
                            offsetW, offsetH, tileSizeW, tileSizeH,
                            dataOut8bit, (float)g_cmdLineOptions.alpha, g_cmdLineOptions.maxFreq, g_cmdLineOptions.interpShapeType
                            );
                    }
                    else
                    {
                        printf("Unknown processing type!");
                    }
                }
            }

            deinitFFTTiledProcessing(&persistentData);
        }
        else
        {
            auto threadFunc = [&](uint32_t threadOffsetW, uint32_t threadOffsetH, uint32_t threadWindowSizeW, uint32_t threadWindowSizeH, float fractionOfWork)
            {
                FFTPersistentData threadPersistentData;
                initFFTTiledProcessing(&threadPersistentData);

                uint32_t maxTileSizeW = g_cmdLineOptions.tileWidth;
                uint32_t maxTileSizeH = g_cmdLineOptions.tileHeight;

                // Ensure tiles are close to the given aspect ratio
                if (maxTileSizeW > threadWindowSizeW)
                {
                    float aspectH = maxTileSizeH / (float)maxTileSizeW;
                    // +1 to get mul = 1 later
                    maxTileSizeW = threadWindowSizeW + 1;
                    maxTileSizeH = int(threadWindowSizeW * aspectH) + 1;
                }
                if (maxTileSizeH > threadWindowSizeH)
                {
                    float aspectW = maxTileSizeW / (float)maxTileSizeH;
                    maxTileSizeW = int(threadWindowSizeH * aspectW) + 1;
                    // +1 to get mul = 1 later
                    maxTileSizeH = threadWindowSizeH + 1;
                }

                const uint32_t mulW = threadWindowSizeW / maxTileSizeW + 1;
                const uint32_t mulH = threadWindowSizeH / maxTileSizeH + 1;

                uint32_t windowSizeW = threadWindowSizeW / mulW + 1;
                uint32_t windowSizeH = threadWindowSizeH / mulH + 1;

                for (uint cntw = 0; cntw < mulW; ++cntw)
                {
                    uint localOffsetW = cntw * windowSizeW;
                    uint offsetW = localOffsetW + threadOffsetW;
                    uint tileSizeW = mymin(windowSizeW, threadWindowSizeW - localOffsetW);
                    for (uint cnth = 0; cnth < mulH; ++cnth)
                    {
                        uint localOffsetH = cnth * windowSizeH;
                        uint offsetH = localOffsetH + threadOffsetH;
                        uint tileSizeH = mymin(windowSizeH, threadWindowSizeH - localOffsetH);

                        if (processingType == ProcessingType::kFLOAT)
                        {
                            assert(dataOut != nullptr);
                            processTilingFast(
                                &threadPersistentData,
                                &regular.dataInFloat[0], regular.dataInW, regular.dataInH,
                                &highres.dataInFloat[0], highres.dataInW, highres.dataInH,
                                offsetW, offsetH, tileSizeW, tileSizeH,
                                dataOut, (float)g_cmdLineOptions.alpha, g_cmdLineOptions.maxFreq, g_cmdLineOptions.interpShapeType
                            );
                        }
                        else if (processingType == ProcessingType::kUBYTE)
                        {
                            assert(dataOut8bit != nullptr);
                            processTilingFast(
                                &threadPersistentData,
                                &regular.dataInUByte[0], regular.dataInW, regular.dataInH,
                                &highres.dataInUByte[0], highres.dataInW, highres.dataInH,
                                offsetW, offsetH, tileSizeW, tileSizeH,
                                dataOut8bit, (float)g_cmdLineOptions.alpha, g_cmdLineOptions.maxFreq, g_cmdLineOptions.interpShapeType
                            );
                        }

                        reportProgress(fractionOfWork / (mulW * mulH));
                    }
                }

                deinitFFTTiledProcessing(&threadPersistentData);
            };

            uint32_t maxWindowSize = (g_cmdLineOptions.tileWidth > g_cmdLineOptions.tileHeight) ? g_cmdLineOptions.tileWidth : g_cmdLineOptions.tileHeight;

            // Handle portrait case
            int splitAxis = 0;
            if (highres.dataInW < highres.dataInH)
                splitAxis = 1;

            uint32_t numThreads = 1;
        
            if (splitAxis == 0)
                numThreads = highres.dataInW / maxWindowSize + 1;
            else
                numThreads = highres.dataInH / maxWindowSize + 1;

            if (numThreads > g_cmdLineOptions.nthreads)
                numThreads = g_cmdLineOptions.nthreads;

            uint32_t threadWindowSizeW, threadWindowSizeH;

            if (splitAxis == 0)
            {
                threadWindowSizeW = highres.dataInW / numThreads + 1;
                threadWindowSizeH = highres.dataInH;
            }
            else
            {
                threadWindowSizeW = highres.dataInW;
                threadWindowSizeH = highres.dataInH / numThreads + 1;
            }

            initMultithreading();

            std::vector<std::thread> threads;
            for (uint32_t t = 0; t < numThreads; ++t)
            {
                uint32_t lwrThreadWindowSizeW = threadWindowSizeW;
                uint32_t lwrThreadWindowSizeH = threadWindowSizeH;

                uint32_t threadOffsetW = 0;
                uint32_t threadOffsetH = 0;

                if (splitAxis == 0)
                {
                    threadOffsetW = t * threadWindowSizeW;
                }
                else
                {
                    threadOffsetH = t * threadWindowSizeH;
                }

                lwrThreadWindowSizeW = mymin(lwrThreadWindowSizeW, highres.dataInW - threadOffsetW);
                lwrThreadWindowSizeH = mymin(lwrThreadWindowSizeH, highres.dataInH - threadOffsetH);

                threads.push_back(std::thread(
                        threadFunc,
                            threadOffsetW,
                            threadOffsetH,
                            lwrThreadWindowSizeW,
                            lwrThreadWindowSizeH,
                            lwrThreadWindowSizeW * lwrThreadWindowSizeH / float(highres.dataInW * highres.dataInH)
                        ));
            }

            for (uint32_t t = 0; t < numThreads; ++t)
            {
                threads[t].join();
            }

            deinitMultithreading();
        }
    }
    else
    {
        uint windowSizeW = highres.dataInW;
        uint windowSizeH = highres.dataInH;

        FFTPersistentData persistentData;
        initFFTTiledProcessing(&persistentData);

        uint offsetW = 0;
        uint tileSizeW = highres.dataInW;

        uint offsetH = 0;
        uint tileSizeH = highres.dataInH;

        if (processingType == ProcessingType::kFLOAT)
        {
            assert(dataOut != nullptr);
            processTiling(
                &persistentData,
                &regular.dataInFloat[0], regular.dataInW, regular.dataInH,
                &highres.dataInFloat[0], highres.dataInW, highres.dataInH,
                offsetW, offsetH, tileSizeW, tileSizeH,
                dataOut, (float)g_cmdLineOptions.alpha, g_cmdLineOptions.maxFreq, g_cmdLineOptions.interpShapeType
                );
        }
        else if (processingType == ProcessingType::kUBYTE)
        {
            assert(dataOut8bit != nullptr);
            processTiling(
                &persistentData,
                &regular.dataInUByte[0], regular.dataInW, regular.dataInH,
                &highres.dataInUByte[0], highres.dataInW, highres.dataInH,
                offsetW, offsetH, tileSizeW, tileSizeH,
                dataOut8bit, (float)g_cmdLineOptions.alpha, g_cmdLineOptions.maxFreq, g_cmdLineOptions.interpShapeType
                );
        }
        else
        {
            printf("Unknown processing type!");
        }

        deinitFFTTiledProcessing(&persistentData);
    }

    double elapsedTime = perfTimer.time();
    printf("total processing time = %.2fms\n", elapsedTime);

    printf("Saving output image(s):\n");
    if (processingType == ProcessingType::kFLOAT)
    {
        std::unordered_map<std::string, std::string> tags;
        darkroom::colwertFromImageMetadata(imageMetadata, tags);
        darkroom::saveExr(dataOut, nullptr, g_cmdLineOptions.outFilename, dataOutW, dataOutH, darkroom::BufferFormat::RGB32, tags);
    }
    else if (processingType == ProcessingType::kUBYTE)
    {
        std::wstring outExt = windows::findExtension(g_cmdLineOptions.outFilename);

        if (outExt == L"bmp")
        {
            darkroom::saveBmp(dataOut8bit, g_cmdLineOptions.outFilename, dataOutW, dataOutH, darkroom::BufferFormat::RGB8);
        }
        else if (outExt == L"jpg" || outExt == L"jpeg")
        {
            std::unordered_map<uint16_t, std::string> tags;
            std::vector<char> xmpPacket;
            darkroom::colwertFromImageMetadata(imageMetadata, tags, xmpPacket);
            darkroom::saveJpeg(dataOut8bit, nullptr, g_cmdLineOptions.outFilename, dataOutW, dataOutH, darkroom::BufferFormat::RGB8, tags, xmpPacket);
        }
        else if (outExt == L"png")
        {
            std::unordered_map<std::string, std::string> tags;
            darkroom::colwertFromImageMetadata(imageMetadata, tags);
            darkroom::savePng(dataOut8bit, nullptr, g_cmdLineOptions.outFilename, dataOutW, dataOutH, darkroom::BufferFormat::RGB8, tags);
        }
        else
        {
            printf("unknown output file format");
        }
    }
    else
    {
        printf("Unknown result type!");
    }

    SAFE_FREE(dataOut);
    SAFE_FREE(dataOut8bit);

    printf("\nFINISHED!\n");
}

