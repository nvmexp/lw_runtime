/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#define NOMINMAX

#include <cmath>
#include <algorithm>
#include <cstdio>
#include "../varray/vertex.h"
#include "glutfont.h"

extern const BitmapFontRec glutBitmap8By13;
extern const BitmapFontRec glutBitmap9By15;

BitmapText::BitmapText(FontName fontName)
{
    m_viewportWidth = 0;
    m_viewportHeight = 0;

    switch(fontName)
    {
    case BITMAP_8X13:
        m_font = &glutBitmap8By13;
        m_charMaxWidth = 8;
        m_charMaxHeight = 13;
        break;

    case BITMAP_9X15:
        m_font = &glutBitmap9By15;
        m_charMaxWidth = 9;
        m_charMaxHeight = 15;
        break;

    default:
        assert(0);
        return;
    }

    int totalChars = m_font->num_chars;

    m_numCharsX = sqrt(totalChars);
    m_numCharsY = totalChars / m_numCharsX;

    assert(m_numCharsX * m_numCharsY == totalChars);

    // While bitmaps have to be padded to byte boundaries in
    // x-direction, no need for padding when stored as texels.

    m_texWidth  = m_numCharsX * m_charMaxWidth;
    m_texHeight = m_numCharsY * m_charMaxHeight;

    m_texScaleX = 1.0 / m_texWidth;
    m_texScaleY = 1.0 / m_texHeight;
}

void BitmapText::GetBitmapTexture(uint8_t *mem)
{
    const uint8_t decodeBit = 0x80;

    for (int j = 0, row = 0; j < m_font->num_chars; row += m_charMaxHeight)
    {
        for (int i = 0, col = 0; i < m_numCharsX; i++, j++, col += m_charMaxWidth)
        {
            const BitmapCharRec *pbitmap = m_font->ch[j];

            if (pbitmap == NULL)
            {
                continue;
            }

            const uint8_t *psrc = pbitmap->bitmap;
            uint8_t *pdst = &mem[row * m_texWidth + col];

            for (int h = 0; h < pbitmap->height; h++, pdst += m_texWidth)
            {
                int width = pbitmap->width;
                uint8_t *prow = pdst;

                do
                {
                    // Can decode at most 8 bits per ubyte
                    uint8_t value = *psrc;
                    int numBits = std::min(8, width);   // Number of bits to decode
                    width -= numBits;

                    while (numBits-- > 0)
                    {
                        *prow++ = (value & decodeBit) ? 255 : 0;
                        value <<= 1;
                    }

                    psrc++;
                } while (width > 0);
            }
        }
    }
}

void BitmapText::FillVertices(float x, float y, float *color, const char *s, void *vertices)
{
    // Incoming screen (x,y) location is mapped into (-1, 1)
    float xc = -1.0f + x * m_viewportScaleX;
    float yc = +1.0f - y * m_viewportScaleY;        // To account for ilwerted ortho

    Color4 color4 = *(Color4 *) color;
    VertexVCT *vct = (VertexVCT *) vertices;

    for (   ; *s != 0; s++, vct += 6)
    {
        int letter = *s;
        const BitmapCharRec *pbitmap = m_font->ch[letter];

        if (pbitmap == NULL)
        {
            // Generate degenerate triangles - not common case - don't optimize
            static BitmapCharRec nullBitmap = {0, 0, 0.0, 0.0, 0.0};
            pbitmap = &nullBitmap;
        }

        float x1 = xc - pbitmap->xorig * m_viewportScaleX;
        float y1 = yc - pbitmap->yorig * m_viewportScaleY;

        float x2 = x1 + pbitmap->width  * m_viewportScaleX;
        float y2 = y1 + pbitmap->height * m_viewportScaleY;

        int row = letter / m_numCharsX;                   // In font char units
        int col = letter - row * m_numCharsX;

        col *= m_charMaxWidth;
        row *= m_charMaxHeight;

        float u1 = col * m_texScaleX;
        float v1 = row * m_texScaleY;
        float u2 = u1 + pbitmap->width  * m_texScaleX;
        float v2 = v1 + pbitmap->height * m_texScaleY;

        // Vertices are defined as follows.
        //
        //    0---3
        //    | \ |
        //    |  \|
        //    1---2
        //
        // With triangles {0, 1, 2}, {3, 0, 2}
        //                           {3, 4, 5}

        vct[0] = {{x1, y2}, {color4}, {u1, v2}};
        vct[1] = {{x1, y1}, {color4}, {u1, v1}};
        vct[2] = {{x2, y1}, {color4}, {u2, v1}};
        vct[3] = {{x2, y2}, {color4}, {u2, v2}};

        vct[4] = vct[0];
        vct[5] = vct[2];

        xc += pbitmap->advance * m_viewportScaleX;
    }
}
