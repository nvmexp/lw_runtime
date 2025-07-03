#include <vector>
#include <string>

const UINT bpp = sizeof(DWORD);
struct Bitmap
{
    UINT width;
    UINT height;
    std::vector<DWORD> pixels;
};


const char * RESULT_IMG_SUBDIR = "ResultImages";
// make TDK compliant outputdir for Shark. 
void MakeOutputdir(const char * outputDir)
{
    if (!outputDir) return;
    std::string outDir(outputDir);
    // images go into a sub-directory for TDK compliance.
    // https://wiki.lwpu.com/engwiki/index.php/TestDrive
    outDir += "\\";
    outDir += RESULT_IMG_SUBDIR;
    std::string::size_type pos = 0;
    // skip first dir, if absolute path
    pos = outDir.find_first_of("/\\", 0);
    if (pos != std::string::npos && outDir[1] == ':') pos = outDir.find_first_of("/\\", pos + 1);
    while (pos != std::string::npos)
    {
        std::string lwrDir = outDir.substr(0, pos);
        _mkdir(lwrDir.c_str());
        pos = outDir.find_first_of("/\\", pos + 1);

    }
    //make the output dir - ignore failure.
    _mkdir(outDir.c_str());
}

void SaveBitmap(Bitmap *pBitmap, const char *outputDir, const char * outfileName)
{
    BITMAPFILEHEADER bmFileHeader;
    BITMAPINFOHEADER bmInfoHeader;
    DWORD dwBmpPitch = pBitmap->width * bpp;

    bmFileHeader.bfType = 'MB';
    bmFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    bmFileHeader.bfSize = bmFileHeader.bfOffBits + dwBmpPitch * pBitmap->height;
    bmFileHeader.bfReserved1 = bmFileHeader.bfReserved2 = 0;

    memset(&bmInfoHeader, 0, sizeof(BITMAPINFOHEADER));
    bmInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmInfoHeader.biWidth = pBitmap->width;
    bmInfoHeader.biHeight = pBitmap->height;
    bmInfoHeader.biPlanes = 1;
    bmInfoHeader.biBitCount = bpp << 3;
    bmInfoHeader.biCompression = BI_RGB;

    FILE* pOutFile;
    if (outputDir != NULL)
    {
        MakeOutputdir(outputDir);
        _chdir(outputDir);
        _chdir(RESULT_IMG_SUBDIR);
    }
    fopen_s(&pOutFile, outfileName, "wb");

    fwrite(&bmFileHeader, sizeof(BITMAPFILEHEADER), 1, pOutFile);
    fwrite(&bmInfoHeader, sizeof(BITMAPINFOHEADER), 1, pOutFile);

    for (int i = pBitmap->height - 1; i >= 0; --i)
    {
        fwrite(&pBitmap->pixels[i*pBitmap->width], bpp, pBitmap->width, pOutFile);
    }

    fclose(pOutFile);
}