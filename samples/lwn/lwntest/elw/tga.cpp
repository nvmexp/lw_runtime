/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdlib.h>
#include <stdio.h>
#include "tga.h"
#include "ogtest.h"


static void fwritec(FILE *file, unsigned char c)
{
    fwrite(&c, 1, 1, file);
}

static void fwrites(FILE *file, unsigned short s)
{
    fwrite(&s, 2, 1, file);
}

static bool freadc(FILE *file, unsigned char test_c)
{
    unsigned char c;
    fread(&c, 1, 1, file);
    return (c == test_c);
}

static bool freads(FILE *file, unsigned short test_s)
{
    unsigned short s;
    fread(&s, 2, 1, file);
    return (s == test_s);
}


int TgaRGBA32Load(char *filename, int width, int height, unsigned char *data)
{
    // Verifies that the width and height match, before allocating <data> and
    // unpacking the contents of <filename> into it.
    FILE *tga = fopen(filename, "rb");

    // We only accept the header we write.
#define CHECK(x) if (!x) { fclose(tga); return -1; }
    CHECK(freadc(tga, 0))
    CHECK(freadc(tga, 0))   // color map type
    CHECK(freadc(tga, 2))   // data type (uncompressed RGBA)
    CHECK(freads(tga, 0))   // color map origin
    CHECK(freads(tga, 0))   // color map length
    CHECK(freadc(tga, 0))   // color map depth
    CHECK(freads(tga, 0))   // origin x
    CHECK(freads(tga, 0))   // origin y
    CHECK(freads(tga, width)) // width
    CHECK(freads(tga, height)) // height
    CHECK(freadc(tga, 32))  // bits per pixel
    CHECK(freadc(tga, 0))   // image descriptor
#undef CHECK

    unsigned char *out_ptr = data;
    unsigned char *rgba = (unsigned char *) malloc(4*width*height);
    fread(rgba, 4*width*height, 1, tga);
    unsigned char *in_ptr = rgba;
    for (int y=0; y<height; y++)
    {
        for (int x=0; x<width; x++)
        {
            out_ptr[2] = *in_ptr++;
            out_ptr[1] = *in_ptr++;
            out_ptr[0] = *in_ptr++;
            out_ptr[3] = *in_ptr++;
            out_ptr += 4;
        }
    }
    free(rgba);

    fclose(tga);
    return 0;
}


int TgaRGBA32Save(char *filename, int width, int height, unsigned char *data)
{
    // Saves <data> to <filename>.
    FILE *tga = fopen(filename, "wb");
    if (!tga) {
        return -1;
    }

    fwritec(tga, 0);    // id
    fwritec(tga, 0);    // color map type
    fwritec(tga, 2);    // data type (uncompressed RGBA)
    fwrites(tga, 0);    // color map origin
    fwrites(tga, 0);    // color map length
    fwritec(tga, 0);    // color map depth
    fwrites(tga, 0);    // origin x
    fwrites(tga, 0);    // origin y
    fwrites(tga, width); // width
    fwrites(tga, height); // height
    fwritec(tga, 32);   // bits per pixel
    fwritec(tga, 0);    // image descriptor

    unsigned char *in_ptr = data;
    unsigned char *bgra = (unsigned char *) malloc(4*width*height);
    unsigned char *out_ptr = bgra;
    for (int y=0; y<height; y++)
    {
        for (int x=0; x<width; x++)
        {
            out_ptr[2] = *in_ptr++;
            out_ptr[1] = *in_ptr++;
            out_ptr[0] = *in_ptr++;
            out_ptr[3] = *in_ptr++;
            out_ptr += 4;
        }
    }
    fwrite(bgra, 4*width*height, 1, tga);
    free(bgra);

    fclose(tga);

    return 0;
}



