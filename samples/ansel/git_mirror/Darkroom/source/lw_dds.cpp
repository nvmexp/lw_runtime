/*
 * Copyright 2008-2012 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to LWPU ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */
// PHYSX-CHANGES:
//   - Removed dependency on OpenGL. -jdolan
//   - MACOS does not always equate to BIG_ENDIAN... fixed it for all platforms. -jdolan

///////////////////////////////////////////////////////////////////////////////
//
// Description:
// 
// Loads DDS images (DXTC1, DXTC3, DXTC5, RGB (888, 888X), and RGBA (8888) are
// supported) for use in OpenGL. Image is flipped when its loaded as DX images
// are stored with different coordinate system. If file has mipmaps and/or 
// lwbemaps then these are loaded as well. Volume textures can be loaded as 
// well but they must be uncompressed.
//
// When multiple textures are loaded (i.e a volume or lwbemap texture), 
// additional faces can be accessed using the array operator. 
//
// The mipmaps for each face are also stored in a list and can be accessed like 
// so: image.get_mipmap() (which accesses the first mipmap of the first 
// image). To get the number of mipmaps call the get_num_mipmaps function for
// a given texture.
//
// Call the is_volume() or is_lwbemap() function to check that a loaded image
// is a volume or lwbemap texture respectively. If a volume texture is loaded
// then the get_depth() function should return a number greater than 1. 
// Mipmapped volume textures and DXTC compressed volume textures are supported.
//
///////////////////////////////////////////////////////////////////////////////
//
// Update: 9/15/2003
//
// Added functions to create new image from a buffer of pixels. Added function
// to save current image to disk.
//
// Update: 6/11/2002
//
// Added some colwenience functions to handle uploading textures to OpenGL. The
// following functions have been added:
//
//     bool upload_texture1D();
//     bool upload_texture2D(unsigned int imageIndex = 0, GLenum target = GL_TEXTURE_2D);
//     bool upload_textureRectangle();
//     bool upload_texture3D();
//     bool upload_textureLwbemap();
//
// See function implementation below for instructions/comments on using each
// function.
//
// The open function has also been updated to take an optional second parameter
// specifying whether the image should be flipped on load. This defaults to 
// true.
//
///////////////////////////////////////////////////////////////////////////////
// Sample usage
///////////////////////////////////////////////////////////////////////////////
//
// Loading a compressed texture:
//
// CDDSImage image;
// GLuint texobj;
//
// image.load("compressed.dds");
// 
// glGenTextures(1, &texobj);
// glEnable(GL_TEXTURE_2D);
// glBindTexture(GL_TEXTURE_2D, texobj);
//
// glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, image.get_format(), 
//     image.get_width(), image.get_height(), 0, image.get_size(), 
//     image);
//
// for (int i = 0; i < image.get_num_mipmaps(); i++)
// {
//     CSurface mipmap = image.get_mipmap(i);
//
//     glCompressedTexImage2DARB(GL_TEXTURE_2D, i+1, image.get_format(), 
//         mipmap.get_width(), mipmap.get_height(), 0, mipmap.get_size(), 
//         mipmap);
// } 
///////////////////////////////////////////////////////////////////////////////
// 
// Loading an uncompressed texture:
//
// CDDSImage image;
// GLuint texobj;
//
// image.load("uncompressed.dds");
//
// glGenTextures(1, &texobj);
// glEnable(GL_TEXTURE_2D);
// glBindTexture(GL_TEXTURE_2D, texobj);
//
// glTexImage2D(GL_TEXTURE_2D, 0, image.get_components(), image.get_width(), 
//     image.get_height(), 0, image.get_format(), GL_UNSIGNED_BYTE, image);
//
// for (int i = 0; i < image.get_num_mipmaps(); i++)
// {
//     glTexImage2D(GL_TEXTURE_2D, i+1, image.get_components(), 
//         image.get_mipmap(i).get_width(), image.get_mipmap(i).get_height(), 
//         0, image.get_format(), GL_UNSIGNED_BYTE, image.get_mipmap(i));
// }
//
///////////////////////////////////////////////////////////////////////////////
// 
// Loading an uncompressed lwbemap texture:
//
// CDDSImage image;
// GLuint texobj;
// GLenum target;
// 
// image.load("lwbemap.dds");
// 
// glGenTextures(1, &texobj);
// glEnable(GL_TEXTURE_LWBE_MAP_ARB);
// glBindTexture(GL_TEXTURE_LWBE_MAP_ARB, texobj);
// 
// for (int n = 0; n < 6; n++)
// {
//     target = GL_TEXTURE_LWBE_MAP_POSITIVE_X_ARB+n;
// 
//     glTexImage2D(target, 0, image.get_components(), image[n].get_width(), 
//         image[n].get_height(), 0, image.get_format(), GL_UNSIGNED_BYTE, 
//         image[n]);
// 
//     for (int i = 0; i < image[n].get_num_mipmaps(); i++)
//     {
//         glTexImage2D(target, i+1, image.get_components(), 
//             image[n].get_mipmap(i).get_width(), 
//             image[n].get_mipmap(i).get_height(), 0,
//             image.get_format(), GL_UNSIGNED_BYTE, image[n].get_mipmap(i));
//     }
// }
//
///////////////////////////////////////////////////////////////////////////////
// 
// Loading a volume texture:
//
// CDDSImage image;
// GLuint texobj;
// 
// image.load("volume.dds");
// 
// glGenTextures(1, &texobj);
// glEnable(GL_TEXTURE_3D);
// glBindTexture(GL_TEXTURE_3D, texobj);
// 
// PFNGLTEXIMAGE3DPROC glTexImage3D;
// glTexImage3D(GL_TEXTURE_3D, 0, image.get_components(), image.get_width(), 
//     image.get_height(), image.get_depth(), 0, image.get_format(), 
//     GL_UNSIGNED_BYTE, image);
// 
// for (int i = 0; i < image.get_num_mipmaps(); i++)
// {
//     glTexImage3D(GL_TEXTURE_3D, i+1, image.get_components(), 
//         image[0].get_mipmap(i).get_width(), 
//         image[0].get_mipmap(i).get_height(), 
//         image[0].get_mipmap(i).get_depth(), 0, image.get_format(), 
//         GL_UNSIGNED_BYTE, image[0].get_mipmap(i));
// }

#include <stdio.h>
#include <assert.h>
//#include "PsFile.h"
#include "lw_dds.h"

using namespace std;
using namespace lw_dds;

///////////////////////////////////////////////////////////////////////////////
// CDDSImage public functions

///////////////////////////////////////////////////////////////////////////////
// default constructor
CDDSImage::CDDSImage()
  : m_format(TextureUnknown),
    m_components(0),
    m_type(TextureNone),
    m_valid(false)
{
}

CDDSImage::~CDDSImage()
{
}

void CDDSImage::create_textureFlat(TextureFormat format, unsigned int components, const CTexture &baseImage)
{
    assert(format != 0);
    assert(components != 0);
    assert(baseImage.get_depth() == 1);

    // remove any existing images
    clear();
    
    m_format = format;
    m_components = components;
    m_type = TextureFlat;

    m_images.push_back(baseImage);

    m_valid = true;
}

void CDDSImage::create_texture3D(TextureFormat format, unsigned int components, const CTexture &baseImage)
{
    assert(format != 0);
    assert(components != 0);
    assert(baseImage.get_depth() > 1);

    // remove any existing images
    clear();

    m_format = format;
    m_components = components;
    m_type = Texture3D;

    m_images.push_back(baseImage);

    m_valid = true;
}

inline bool same_size(const CTexture &a, const CTexture &b)
{
    if (a.get_width() != b.get_width())
        return false;
    if (a.get_height() != b.get_height())
        return false;
    if (a.get_depth() != b.get_depth())
        return false;

    return true;
}

void CDDSImage::create_textureLwbemap(TextureFormat format, unsigned int components,
                                      const CTexture &positiveX, const CTexture &negativeX, 
                                      const CTexture &positiveY, const CTexture &negativeY, 
                                      const CTexture &positiveZ, const CTexture &negativeZ)
{
    assert(format != 0);
    assert(components != 0);
    assert(positiveX.get_depth() == 1);

    // verify that all dimensions are the same 
    assert(same_size(positiveX, negativeX));
    assert(same_size(positiveX, positiveY));
    assert(same_size(positiveX, negativeY));
    assert(same_size(positiveX, positiveZ));
    assert(same_size(positiveX, negativeZ));

    // remove any existing images
    clear();

    m_format = format;
    m_components = components;
    m_type = TextureLwbemap;

    m_images.push_back(positiveX);
    m_images.push_back(negativeX);
    m_images.push_back(positiveY);
    m_images.push_back(negativeY);
    m_images.push_back(positiveZ);
    m_images.push_back(negativeZ);

    m_valid = true;
}

///////////////////////////////////////////////////////////////////////////////
// loads DDS image
//
// filename - fully qualified name of DDS image
// flipImage - specifies whether image is flipped on load, default is true
bool CDDSImage::load(const wchar_t * filename, bool flipImage)
{
    // clear any previously loaded images
    clear();
    
    // open file
    FILE *fp = 0;
    _wfopen_s(&fp, filename, L"rb");
    if (fp == NULL)
        return false;

    bool success = load(fp, flipImage);

    fclose(fp);

    return success;
}

///////////////////////////////////////////////////////////////////////////////
// loads DDS image from FILE stream
//
// filename - fully qualified name of DDS image
// flipImage - specifies whether image is flipped on load, default is true
bool CDDSImage::load(FILE* fp, bool flipImage)
{
    assert(fp != 0);
    
    // clear any previously loaded images
    clear();
    
    // read in file marker, make sure its a DDS file
    char filecode[4];
    fread(filecode, 1, 4, fp);
    if (strncmp(filecode, "DDS ", 4) != 0)
    {
        return false;
    }

    // read in DDS header
    DDS_HEADER ddsh;
    fread(&ddsh, sizeof(DDS_HEADER), 1, fp);

    swap_endian(&ddsh.dwSize);
    swap_endian(&ddsh.dwFlags);
    swap_endian(&ddsh.dwHeight);
    swap_endian(&ddsh.dwWidth);
    swap_endian(&ddsh.dwPitchOrLinearSize);
    swap_endian(&ddsh.dwMipMapCount);
    swap_endian(&ddsh.ddspf.dwSize);
    swap_endian(&ddsh.ddspf.dwFlags);
    swap_endian(&ddsh.ddspf.dwFourCC);
    swap_endian(&ddsh.ddspf.dwRGBBitCount);
    swap_endian(&ddsh.dwCaps1);
    swap_endian(&ddsh.dwCaps2);

    // default to flat texture type (1D, 2D, or rectangle)
    m_type = TextureFlat;

    // check if image is a lwbemap
    if (ddsh.dwCaps2 & DDSF_LWBEMAP)
        m_type = TextureLwbemap;

    // check if image is a volume texture
    if ((ddsh.dwCaps2 & DDSF_VOLUME) && (ddsh.dwDepth > 0))
        m_type = Texture3D;

    // figure out what the image format is
    if (ddsh.ddspf.dwFlags & DDSF_FOURCC) 
    {
        switch(ddsh.ddspf.dwFourCC)
        {
            case FOURCC_DXT1:
                m_format = TextureDXT1;
                m_components = 3;
                break;
            case FOURCC_DXT3:
                m_format = TextureDXT3;
                m_components = 4;
                break;
            case FOURCC_DXT5:
                m_format = TextureDXT5;
                m_components = 4;
                break;
            default:
                return false;
        }
    }
    else if (ddsh.ddspf.dwFlags == DDSF_RGBA && ddsh.ddspf.dwRGBBitCount == 32)
    {
        m_format = TextureBGRA; 
        m_components = 4;
    }
    else if (ddsh.ddspf.dwFlags == DDSF_RGB  && ddsh.ddspf.dwRGBBitCount == 32)
    {
        m_format = TextureBGRA; 
        m_components = 4;
    }
    else if (ddsh.ddspf.dwFlags == DDSF_RGB  && ddsh.ddspf.dwRGBBitCount == 24)
    {
        m_format = TextureBGR; 
        m_components = 3;
    }
    else if (ddsh.ddspf.dwRGBBitCount == 8)
    {
        m_format = TextureLuminance;
        m_components = 1;
    }
    else 
    {
        return false;
    }
    
    // store primary surface width/height/depth
    unsigned int width, height, depth;
    width = ddsh.dwWidth;
    height = ddsh.dwHeight;
    depth = clamp_size(ddsh.dwDepth);   // set to 1 if 0
    
    // use correct size callwlation function depending on whether image is 
    // compressed
    unsigned int (CDDSImage::*sizefunc)(unsigned int, unsigned int);
    sizefunc = (is_compressed() ? &CDDSImage::size_dxtc : &CDDSImage::size_rgb);

    // load all surfaces for the image (6 surfaces for lwbemaps)
    for (unsigned int n = 0; n < (unsigned int)(m_type == TextureLwbemap ? 6 : 1); n++)
    {
        // add empty texture object
        m_images.push_back(CTexture());

        // get reference to newly added texture object
        CTexture &img = m_images[n];
        
        // callwlate surface size
        unsigned int size = (this->*sizefunc)(width, height)*depth;

        // load surface
        unsigned char *pixels = new unsigned char[size];
        fread(pixels, 1, size, fp);

        img.create(width, height, depth, size, pixels);
        
        delete [] pixels;

        if (flipImage) flip(img);
        
        unsigned int w = clamp_size(width >> 1);
        unsigned int h = clamp_size(height >> 1);
        unsigned int d = clamp_size(depth >> 1); 

        // store number of mipmaps
        unsigned int numMipmaps = ddsh.dwMipMapCount;

        // number of mipmaps in file includes main surface so decrease count 
        // by one
        if (numMipmaps != 0)
            numMipmaps--;

        // load all mipmaps for current surface
        for (unsigned int i = 0; i < numMipmaps && (w || h); i++)
        {
            // add empty surface
            img.add_mipmap(CSurface());

            // get reference to newly added mipmap
            CSurface &mipmap = img.get_mipmap(i);

            // callwlate mipmap size
            size = (this->*sizefunc)(w, h)*d;

            unsigned char *pixelsMipMap = new unsigned char[size];
            fread(pixelsMipMap, 1, size, fp);

            mipmap.create(w, h, d, size, pixelsMipMap);
            
            delete [] pixelsMipMap;

            if (flipImage) flip(mipmap);

            // shrink to next power of 2
            w = clamp_size(w >> 1);
            h = clamp_size(h >> 1);
            d = clamp_size(d >> 1); 
        }
    }

    // swap lwbemaps on y axis (since image is flipped in OGL)
    if (m_type == TextureLwbemap && flipImage)
    {
        CTexture tmp;
        tmp = m_images[3];
        m_images[3] = m_images[2];
        m_images[2] = tmp;
    }
    
    m_valid = true;

    return true;
}

void CDDSImage::write_texture(const CTexture &texture, FILE *fp)
{
    assert(get_num_mipmaps() == texture.get_num_mipmaps());
    
    fwrite(texture, 1, texture.get_size(), fp);
    
    for (unsigned int i = 0; i < texture.get_num_mipmaps(); i++)
    {
        const CSurface &mipmap = texture.get_mipmap(i);
        fwrite(mipmap, 1, mipmap.get_size(), fp);
    }
}

bool CDDSImage::save(const wchar_t * filename, bool flipImage)
{
    assert(m_valid);
    assert(m_type != TextureNone);

    // open file
    FILE *fp = 0;
    _wfopen_s(&fp, filename, L"wb");
    if (fp == NULL)
        return false;

    bool result = save(fp, flipImage);

    fclose(fp);

    return result;
}

bool CDDSImage::save(FILE* fp, bool flipImage)
{
    assert(m_valid);
    assert(m_type != TextureNone);
    assert(fp != 0);

    DDS_HEADER ddsh;
    unsigned int headerSize = sizeof(DDS_HEADER);
    memset(&ddsh, 0, headerSize);
    ddsh.dwSize = headerSize;
    ddsh.dwFlags = DDSF_CAPS | DDSF_WIDTH | DDSF_HEIGHT | DDSF_PIXELFORMAT;
    ddsh.dwHeight = get_height();
    ddsh.dwWidth = get_width();

    if (is_compressed())
    {
        ddsh.dwFlags |= DDSF_LINEARSIZE;
        ddsh.dwPitchOrLinearSize = get_size();
    }
    else
    {
        ddsh.dwFlags |= DDSF_PITCH;
        ddsh.dwPitchOrLinearSize = get_dword_aligned_linesize(get_width(), m_components * 8);
    }
    
    if (m_type == Texture3D)
    {
        ddsh.dwFlags |= DDSF_DEPTH;
        ddsh.dwDepth = get_depth();
    }

    if (get_num_mipmaps() > 0)
    {
        ddsh.dwFlags |= DDSF_MIPMAPCOUNT;
        ddsh.dwMipMapCount = get_num_mipmaps() + 1;
    }

    ddsh.ddspf.dwSize = sizeof(DDS_PIXELFORMAT);

    if (is_compressed())
    {
        ddsh.ddspf.dwFlags = DDSF_FOURCC;
        
        if (m_format == TextureDXT1)
            ddsh.ddspf.dwFourCC = FOURCC_DXT1;
        if (m_format == TextureDXT3)
            ddsh.ddspf.dwFourCC = FOURCC_DXT3;
        if (m_format == TextureDXT5)
            ddsh.ddspf.dwFourCC = FOURCC_DXT5;
    }
    else
    {
        ddsh.ddspf.dwFlags = (m_components == 4) ? DDSF_RGBA : DDSF_RGB;
        ddsh.ddspf.dwRGBBitCount = m_components * 8;
        ddsh.ddspf.dwRBitMask = 0x00ff0000;
        ddsh.ddspf.dwGBitMask = 0x0000ff00;
        ddsh.ddspf.dwBBitMask = 0x000000ff;
 
        if (m_components == 4)
        {
            ddsh.ddspf.dwFlags |= DDSF_ALPHAPIXELS;
            ddsh.ddspf.dwABitMask = 0xff000000;
        }
    }
    
    ddsh.dwCaps1 = DDSF_TEXTURE;
    
    if (m_type == TextureLwbemap)
    {
        ddsh.dwCaps1 |= DDSF_COMPLEX;
        ddsh.dwCaps2 = DDSF_LWBEMAP | DDSF_LWBEMAP_ALL_FACES;
    }

    if (m_type == Texture3D)
    {
        ddsh.dwCaps1 |= DDSF_COMPLEX;
        ddsh.dwCaps2 = DDSF_VOLUME;
    }

    if (get_num_mipmaps() > 0)
        ddsh.dwCaps1 |= DDSF_COMPLEX | DDSF_MIPMAP;

    // write file header
    fwrite("DDS ", 1, 4, fp);
    
    // write dds header
    fwrite(&ddsh, 1, sizeof(DDS_HEADER), fp);

    if (m_type != TextureLwbemap)
    {
        CTexture tex = m_images[0];
        if (flipImage) flip_texture(tex);
        write_texture(tex, fp);
    }
    else
    {
        assert(m_images.size() == 6);

        for (unsigned int i = 0; i < m_images.size(); i++)
        {
            CTexture lwbeFace;

            if (i == 2) 
                lwbeFace = m_images[3];
            else if (i == 3) 
                lwbeFace = m_images[2];
            else 
                lwbeFace = m_images[i];

            if (flipImage) flip_texture(lwbeFace);
            write_texture(lwbeFace, fp);
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// free image memory
void CDDSImage::clear()
{
    m_components = 0;
    m_format = TextureUnknown;
    m_type = TextureNone;
    m_valid = false;

    m_images.clear();
}

///////////////////////////////////////////////////////////////////////////////
// clamps input size to [1-size]
inline unsigned int CDDSImage::clamp_size(unsigned int size)
{
    if (size <= 0)
        size = 1;

    return size;
}

///////////////////////////////////////////////////////////////////////////////
// CDDSImage private functions
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// callwlates size of DXTC texture in bytes
inline unsigned int CDDSImage::size_dxtc(unsigned int width, unsigned int height)
{
    return ((width+3)/4)*((height+3)/4)*
        (m_format == TextureDXT1 ? 8 : 16);   
}

///////////////////////////////////////////////////////////////////////////////
// callwlates size of uncompressed RGB texture in bytes
inline unsigned int CDDSImage::size_rgb(unsigned int width, unsigned int height)
{
    return width*height*m_components;
}

///////////////////////////////////////////////////////////////////////////////
// Swap the bytes in a 32 bit value
inline void CDDSImage::swap_endian(void *val)
{
#if defined(LW_DDS_BIG_ENDIAN)
    unsigned int *ival = (unsigned int *)val;

    *ival = ((*ival >> 24) & 0x000000ff) |
            ((*ival >>  8) & 0x0000ff00) |
            ((*ival <<  8) & 0x00ff0000) |
            ((*ival << 24) & 0xff000000);
#else
    (void)val;
#endif
}

///////////////////////////////////////////////////////////////////////////////
// flip image around X axis
void CDDSImage::flip(CSurface &surface)
{
    unsigned int linesize;
    unsigned int offset;

    if (!is_compressed())
    {
        assert(surface.get_depth() > 0);

        unsigned int imagesize = surface.get_size()/surface.get_depth();
        linesize = imagesize / surface.get_height();

        for (unsigned int n = 0; n < surface.get_depth(); n++)
        {
            offset = imagesize*n;
            unsigned char *top = (unsigned char*)surface + offset;
            unsigned char *bottom = top + (imagesize-linesize);
    
            for (unsigned int i = 0; i < (surface.get_height() >> 1); i++)
            {
                swap(bottom, top, linesize);

                top += linesize;
                bottom -= linesize;
            }
        }
    }
    else
    {
        void (CDDSImage::*flipblocks)(DXTColBlock*, unsigned int);
        unsigned int xblocks = surface.get_width() / 4;
        unsigned int yblocks = surface.get_height() / 4;
        unsigned int blocksize;

        switch (m_format)
        {
            case TextureDXT1: 
                blocksize = 8;
                flipblocks = &CDDSImage::flip_blocks_dxtc1; 
                break;
            case TextureDXT3: 
                blocksize = 16;
                flipblocks = &CDDSImage::flip_blocks_dxtc3; 
                break;
            case TextureDXT5: 
                blocksize = 16;
                flipblocks = &CDDSImage::flip_blocks_dxtc5; 
                break;
            case TextureUnknown:
            case TextureBGRA:
            case TextureBGR:
            case TextureLuminance:
            default:
                return;
        }

        linesize = xblocks * blocksize;

        DXTColBlock *top;
        DXTColBlock *bottom;
    
        for (unsigned int j = 0; j < (yblocks >> 1); j++)
        {
            top = (DXTColBlock*)((unsigned char*)surface+ j * linesize);
            bottom = (DXTColBlock*)((unsigned char*)surface + (((yblocks-j)-1) * linesize));

            (this->*flipblocks)(top, xblocks);
            (this->*flipblocks)(bottom, xblocks);

            swap(bottom, top, linesize);
        }
    }
}    

void CDDSImage::flip_texture(CTexture &texture)
{
    flip(texture);
    
    for (unsigned int i = 0; i < texture.get_num_mipmaps(); i++)
    {
        flip(texture.get_mipmap(i));
    }
}

///////////////////////////////////////////////////////////////////////////////
// swap to sections of memory
void CDDSImage::swap(void *byte1, void *byte2, unsigned int size)
{
    unsigned char *tmp = new unsigned char[size];

    memcpy(tmp, byte1, size);
    memcpy(byte1, byte2, size);
    memcpy(byte2, tmp, size);

    delete [] tmp;
}

///////////////////////////////////////////////////////////////////////////////
// flip a DXT1 color block
void CDDSImage::flip_blocks_dxtc1(DXTColBlock *line, unsigned int numBlocks)
{
    DXTColBlock *lwrblock = line;

    for (unsigned int i = 0; i < numBlocks; i++)
    {
        swap(&lwrblock->row[0], &lwrblock->row[3], sizeof(unsigned char));
        swap(&lwrblock->row[1], &lwrblock->row[2], sizeof(unsigned char));

        lwrblock++;
    }
}

///////////////////////////////////////////////////////////////////////////////
// flip a DXT3 color block
void CDDSImage::flip_blocks_dxtc3(DXTColBlock *line, unsigned int numBlocks)
{
    DXTColBlock *lwrblock = line;
    DXT3AlphaBlock *alphablock;

    for (unsigned int i = 0; i < numBlocks; i++)
    {
        alphablock = (DXT3AlphaBlock*)lwrblock;

        swap(&alphablock->row[0], &alphablock->row[3], sizeof(unsigned short));
        swap(&alphablock->row[1], &alphablock->row[2], sizeof(unsigned short));

        lwrblock++;

        swap(&lwrblock->row[0], &lwrblock->row[3], sizeof(unsigned char));
        swap(&lwrblock->row[1], &lwrblock->row[2], sizeof(unsigned char));

        lwrblock++;
    }
}

///////////////////////////////////////////////////////////////////////////////
// flip a DXT5 alpha block
void CDDSImage::flip_dxt5_alpha(DXT5AlphaBlock *block)
{
    unsigned char gBits[4][4];
    
    const unsigned int mask = 0x00000007;          // bits = 00 00 01 11
    unsigned int bits = 0;
    memcpy(&bits, &block->row[0], sizeof(unsigned char) * 3);

    gBits[0][0] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[0][1] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[0][2] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[0][3] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[1][0] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[1][1] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[1][2] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[1][3] = (unsigned char)(bits & mask);

    bits = 0;
    memcpy(&bits, &block->row[3], sizeof(unsigned char) * 3);

    gBits[2][0] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[2][1] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[2][2] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[2][3] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[3][0] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[3][1] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[3][2] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[3][3] = (unsigned char)(bits & mask);

    unsigned int *pBits = ((unsigned int*) &(block->row[0]));

    *pBits = *pBits | (gBits[3][0] << 0);
    *pBits = *pBits | (gBits[3][1] << 3);
    *pBits = *pBits | (gBits[3][2] << 6);
    *pBits = *pBits | (gBits[3][3] << 9);

    *pBits = *pBits | (gBits[2][0] << 12);
    *pBits = *pBits | (gBits[2][1] << 15);
    *pBits = *pBits | (gBits[2][2] << 18);
    *pBits = *pBits | (gBits[2][3] << 21);

    pBits = ((unsigned int*) &(block->row[3]));

#if defined(LW_DDS_BIG_ENDIAN)
    *pBits &= 0x000000ff;
#else
    *pBits &= 0xff000000;
#endif

    *pBits = *pBits | (gBits[1][0] << 0);
    *pBits = *pBits | (gBits[1][1] << 3);
    *pBits = *pBits | (gBits[1][2] << 6);
    *pBits = *pBits | (gBits[1][3] << 9);

    *pBits = *pBits | (gBits[0][0] << 12);
    *pBits = *pBits | (gBits[0][1] << 15);
    *pBits = *pBits | (gBits[0][2] << 18);
    *pBits = *pBits | (gBits[0][3] << 21);
}

///////////////////////////////////////////////////////////////////////////////
// flip a DXT5 color block
void CDDSImage::flip_blocks_dxtc5(DXTColBlock *line, unsigned int numBlocks)
{
    DXTColBlock *lwrblock = line;
    DXT5AlphaBlock *alphablock;
    
    for (unsigned int i = 0; i < numBlocks; i++)
    {
        alphablock = (DXT5AlphaBlock*)lwrblock;
        
        flip_dxt5_alpha(alphablock);

        lwrblock++;

        swap(&lwrblock->row[0], &lwrblock->row[3], sizeof(unsigned char));
        swap(&lwrblock->row[1], &lwrblock->row[2], sizeof(unsigned char));

        lwrblock++;
    }
}

///////////////////////////////////////////////////////////////////////////////
// CTexture implementation
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// default constructor
CTexture::CTexture()
  : CSurface()  // initialize base class part
{
}

///////////////////////////////////////////////////////////////////////////////
// creates an empty texture
CTexture::CTexture(unsigned int w, unsigned int h, unsigned int d, unsigned int imgsize, const unsigned char *pixels)
  : CSurface(w, h, d, imgsize, pixels)  // initialize base class part
{
}

CTexture::~CTexture()
{
}

///////////////////////////////////////////////////////////////////////////////
// copy constructor
CTexture::CTexture(const CTexture &copy)
  : CSurface(copy)
{
    for (unsigned int i = 0; i < copy.get_num_mipmaps(); i++)
        m_mipmaps.push_back(copy.get_mipmap(i));
}

///////////////////////////////////////////////////////////////////////////////
// assignment operator
CTexture &CTexture::operator= (const CTexture &rhs)
{
    if (this != &rhs)
    {
        CSurface::operator = (rhs);

        m_mipmaps.clear();
        for (unsigned int i = 0; i < rhs.get_num_mipmaps(); i++)
            m_mipmaps.push_back(rhs.get_mipmap(i));
    }

    return *this;
}

void CTexture::create(unsigned int w, unsigned int h, unsigned int d, unsigned int imgsize, const unsigned char *pixels)
{
    CSurface::create(w, h, d, imgsize, pixels);

    m_mipmaps.clear();
}

void CTexture::clear()
{
    CSurface::clear();

    m_mipmaps.clear();
}

///////////////////////////////////////////////////////////////////////////////
// CSurface implementation
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// default constructor
CSurface::CSurface()
  : m_width(0),
    m_height(0),
    m_depth(0),
    m_size(0),
    m_pixels(NULL)
{
}

///////////////////////////////////////////////////////////////////////////////
// creates an empty image
CSurface::CSurface(unsigned int w, unsigned int h, unsigned int d, unsigned int imgsize, const unsigned char *pixels)
  : m_width(0),
    m_height(0),
    m_depth(0),
    m_size(0),
    m_pixels(NULL)
{
    create(w, h, d, imgsize, pixels);
}

///////////////////////////////////////////////////////////////////////////////
// copy constructor
CSurface::CSurface(const CSurface &copy)
  : m_width(0),
    m_height(0),
    m_depth(0),
    m_size(0),
    m_pixels(NULL)
{
    if (copy.get_size() != 0)
    {
        m_size = copy.get_size();
        m_width = copy.get_width();
        m_height = copy.get_height();
        m_depth = copy.get_depth();

        m_pixels = new unsigned char[m_size];
        memcpy(m_pixels, copy, m_size);
    }
}

///////////////////////////////////////////////////////////////////////////////
// assignment operator
CSurface &CSurface::operator= (const CSurface &rhs)
{
    if (this != &rhs)
    {
        clear();

        if (rhs.get_size())
        {
            m_size = rhs.get_size();
            m_width = rhs.get_width();
            m_height = rhs.get_height();
            m_depth = rhs.get_depth();

            m_pixels = new unsigned char[m_size];
            memcpy(m_pixels, rhs, m_size);
        }
    }

    return *this;
}

///////////////////////////////////////////////////////////////////////////////
// clean up image memory
CSurface::~CSurface()
{
    clear();
}

///////////////////////////////////////////////////////////////////////////////
// returns a pointer to image
CSurface::operator unsigned char*() const
{ 
    return m_pixels; 
}

///////////////////////////////////////////////////////////////////////////////
// creates an empty image
void CSurface::create(unsigned int w, unsigned int h, unsigned int d, unsigned int imgsize, const unsigned char *pixels)
{
    assert(w != 0);
    assert(h != 0);
    assert(d != 0);
    assert(imgsize != 0);
    assert(pixels);

    clear();

    m_width = w;
    m_height = h;
    m_depth = d;
    m_size = imgsize;
    m_pixels = new unsigned char[imgsize];
    memcpy(m_pixels, pixels, imgsize);
}

///////////////////////////////////////////////////////////////////////////////
// free surface memory
void CSurface::clear()
{
    if (m_pixels != NULL)
    {
        delete [] m_pixels;
        m_pixels = NULL;
    }
}
