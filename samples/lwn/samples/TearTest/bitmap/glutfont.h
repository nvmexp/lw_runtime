#pragma once

/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <cassert>
#include <cstdint>
#include <cstring>
#include "glutbitmap.h"

// This code is designed to be usable in other projects in future and
// potentially with different API's. As a result, this code does not
// try to load texture images or make specific API calls for rendering
// text - instead that responsibility is left to the end-user of this
// code. Once the class has been constructed, the user code is:
//
// 1. Expected to create a texture map
//    - Call into this class to obtain widht/height of UINT8 texture
//    - Allocate memory for the texture
//    - Call into this code for the memory to be filled
//    - Create texture using their favorite API.
//
// 2. For rendering text,
//    - Ensure that this code has the correct viewport size
//    - Call into this code with to inquire number of vertices that
//      are required for a string.
//    - Each vertex is hardcoded to be {Vertex3f, Color4f, Tex2f}
//    - Call into this code with screen position (x, y) that is
//      colwerted to {-1, 1} using viewort. The color field of the
//      vertex is the desired color. The Texcoord's look up in
//      texture for the font that was created in step 1.
//    - End user code is responsible for setting up the texture
//      state, shader state etc. and then draw the polygons.

class BitmapText
{
public:
    enum FontName {BITMAP_8X13, BITMAP_9X15};

    BitmapText(FontName fontName);
    ~BitmapText(){}

    void GetBitmapTextureSize(int *ptexWidth, int *ptexHeight)
    {
        *ptexWidth  = m_texWidth;
        *ptexHeight = m_texHeight;
    }

    void GetBitmapTexture(uint8_t *mem);

    void Viewport(int width, int height)
    {
        m_viewportWidth = width;
        m_viewportHeight = height;

        m_viewportScaleX = 2.0 / width;
        m_viewportScaleY = 2.0 / height;
    }

    int GetVertexCount(const char *s)
    {
        assert(m_viewportWidth != 0);
        return 6 * strlen(s);
    }

    void FillVertices(float x, float y, float *color, const char *string, void *vertices);

private:
    const BitmapFontRec *m_font;
    int m_charMaxWidth;
    int m_charMaxHeight;

    int m_numCharsX;        // Number of characters in X-direction
    int m_numCharsY;        // Number of characters in Y-direction

    int m_texWidth;
    int m_texHeight;

    float m_texScaleX;
    float m_texScaleY;

    int m_viewportWidth;
    int m_viewportHeight;

    float m_viewportScaleX;
    float m_viewportScaleY;
};
