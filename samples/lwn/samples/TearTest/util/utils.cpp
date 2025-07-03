/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "stdafx.h"
#include "utils.h"

static void CreateTextImage(const char* path, uint32_t width, uint32_t height, void *pdata)
{
    static const int BytesPerLine = 20;

    char *dataName;
    char *png;
    const char *baseName;

    baseName = strrchr(path, '\\');
    if (!baseName) baseName = strrchr(path, '/');

    baseName = baseName ? (baseName + 1) : path;

    dataName = strdup(baseName);
    png = strstr(dataName, ".png");
    if (!png) png = strstr(dataName, ".PNG");

    if (png == NULL) {
        printf("Did not find .png suffix\n");
        return;
    }
    strncpy(png+1, "txt", 3);

    FILE *fp = fopen(dataName, "w");

    if (fp == NULL) {
        printf("Could not open file (%s) for writing\n", dataName);
        return;
    }

    *png = '\0';

    fprintf(fp, "#pragma once\n\n");

    fprintf(fp,
       "#ifndef IMAGE_STRUCT\n"
       "#define IMAGE_STRUCT\n\n"
       "struct Image {\n"
       "    uint32_t width;\n"
       "    uint32_t height;\n"
       "    const uint8_t *data;\n"
       "};\n"
       "\n"
       "#endif // IMAGE_STRUCT\n"
       "\n");

    fprintf(fp, "extern const uint8_t %sData[];\n\n", dataName);

    fprintf(fp, "Image %sImage = {%u, %u, %sData};\n\n", dataName, width, height, dataName);

    fprintf(fp, "const uint8_t %sData[] = {\n", dataName);

    uint8_t *bytes = (uint8_t *) pdata;
    int numBytes = width * height * 4; /* RGBA */

    for (int i = 0; i < numBytes; )
    {
        for (int j = 0; j < BytesPerLine && i < numBytes; j++, i++, bytes++)
        {
            fprintf(fp, "0x%02x,", *bytes);
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "};\n");

    fclose(fp);
}

#ifdef IMAGE_SUPPORT

static bool ilInitialized = false;

uint32_t CreateImage(const char* path, uint32_t *pWidth, uint32_t *pHeight, void **ppdata)
{
    ILboolean success;
    uint32_t ilTex;

    if (!ilInitialized)
    {
        ilInitialized = true;
        ilInit();
    }

    ilGenImages(1, &ilTex);
    ilBindImage(ilTex);
    ilEnable(IL_ORIGIN_SET);
    ilOriginFunc(IL_ORIGIN_UPPER_LEFT);

    success = ilLoadImage(path);

    if (!success)
    {
        ilDeleteImage(ilTex);
        printf("Error while loading %s.\n", path);
        return 0;
    }

    ilColwertImage(IL_RGBA, IL_UNSIGNED_BYTE);

    *ppdata  = ilGetData();
    *pWidth  = ilGetInteger(IL_IMAGE_WIDTH);
    *pHeight = ilGetInteger(IL_IMAGE_HEIGHT);

#if 0   // If we want to save image into text format
    CreateTextImage(path, *pWidth, *pHeight, *ppdata);
#endif

    return ilTex;
}

void DestroyImage(uint32_t ilTex)
{
    ilDeleteImage(ilTex);
}

void SaveImage(const char* path, unsigned int width, unsigned int height, void *data, bool hasAlpha)
{
    ILuint image;

    if (!ilInitialized)
    {
        ilInitialized = true;
        ilInit();
    }

    ilGenImages(1, &image);
    ilBindImage(image);

    ilTexImage(width, height, 1, hasAlpha ? 4 : 3, hasAlpha ? IL_RGBA : IL_RGB, IL_BYTE, data);
    ilEnable(IL_FILE_OVERWRITE);

    if (ilSaveImage(path))
    {
        printf("Saved image to %s.\n", path);
    }
    else
    {
        printf("Error while saving %s.\n", path);
    }

    ilDeleteImage(image);
}

#else

uint32_t CreateImage(const char* path, uint32_t *pWidth, uint32_t *pHeight, void **ppdata)
{
    return 0;
}

void DestroyImage(uint32_t ilTex)
{}

void SaveImage(const char* path, unsigned int width, unsigned int height, void *data, bool hasAlpha)
{
    printf("Image saving not supported\n");
}

#endif

void CopyToClipboard(const string& str)
{
    HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, str.size() + 1);
    memcpy(GlobalLock(hMem), str.c_str(), str.size() + 1);
    GlobalUnlock(hMem);
    OpenClipboard(0);
    EmptyClipboard();
    SetClipboardData(CF_TEXT, hMem);
    CloseClipboard();
}

void PasteFromClipboard(string& str)
{
    if (!OpenClipboard(0))
    {
        return;
    }

    HANDLE hData = GetClipboardData(CF_TEXT);
    if (hData)
    {
        char *pszText = (char*)GlobalLock(hData);
        if (pszText)
        {
            str = pszText;
        }
        GlobalUnlock(hData);

    }
    CloseClipboard();
}
