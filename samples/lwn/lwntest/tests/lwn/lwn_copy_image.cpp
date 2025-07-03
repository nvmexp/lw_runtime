/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#if defined(LW_TEGRA)
#include "lwn_PrivateFormats.h"
#endif

using namespace lwn;


#define DEBUG_MODE 0

#if DEBUG_MODE
#define DEBUG_PRINT(x) do { \
    printf x; \
    fflush(stdout); \
} while (0)
#else
#define DEBUG_PRINT(x)
#endif


const int texSize = 200; // NPOT

enum testModes {TEX1D, TEX2D, RECTANGLE};

// Divide by block size, while rounding up
#define BLOCK_DIV(N, S) (((N) + (S) - 1) / (S))
#define ROUND_UP(N, S) (BLOCK_DIV(N, S) * (S))
#define ROUND_DN(N, S) ((N) - ((N) % (S)))

// All formats exercised by tests in this file
static const Format formats[] = {
    Format::R8,
    Format::R8SN,
    Format::R8UI,
    Format::R8I,
    Format::R16F,
    Format::R16,
    Format::R16SN,
    Format::R16UI,
    Format::R16I,
    Format::R32F,
    Format::R32UI,
    Format::R32I,
    Format::RG8,
    Format::RG8SN,
    Format::RG8UI,
    Format::RG8I,
    Format::RG16F,
    Format::RG16,
    Format::RG16SN,
    Format::RG16UI,
    Format::RG16I,
    Format::RG32F,
    Format::RG32UI,
    Format::RG32I,
    Format::RGB8,
    Format::RGB8SN,
    Format::RGB8UI,
    Format::RGB8I,
    Format::RGB16F,
    Format::RGB16,
    Format::RGB16SN,
    Format::RGB16UI,
    Format::RGB16I,
    Format::RGB32F,
    Format::RGB32UI,
    Format::RGB32I,
    Format::RGBA8,
    Format::RGBA8SN,
    Format::RGBA8UI,
    Format::RGBA8I,
    Format::RGBA16F,
    Format::RGBA16,
    Format::RGBA16SN,
    Format::RGBA16UI,
    Format::RGBA16I,
    Format::RGBA32F,
    Format::RGBA32UI,
    Format::RGBA32I,
    // Depth, stencil, and depth+stencil all swizzle differently. It's not yet
    // known what the final output will look like.
    Format::STENCIL8,
    Format::DEPTH16,
    Format::DEPTH24,
    Format::DEPTH32F,
    Format::DEPTH24_STENCIL8,
    Format::DEPTH32F_STENCIL8,
    Format::RGBX8_SRGB,
    Format::RGBA8_SRGB,
    Format::RGBA4,
    Format::RGB5,
    Format::RGB5A1,
    Format::RGB565,
    Format::RGB10A2,
    Format::RGB10A2UI,
    Format::R11G11B10F,
    Format::RGB9E5F,
    Format::RGB_DXT1,
    Format::RGBA_DXT1,
    Format::RGBA_DXT3,
    Format::RGBA_DXT5,
    Format::RGB_DXT1_SRGB,
    Format::RGBA_DXT1_SRGB,
    Format::RGBA_DXT3_SRGB,
    Format::RGBA_DXT5_SRGB,
    Format::RGTC1_UNORM,
    Format::RGTC1_SNORM,
    Format::RGTC2_UNORM,
    Format::RGTC2_SNORM,
    Format::BPTC_UNORM,
    Format::BPTC_UNORM_SRGB,
    Format::BPTC_SFLOAT,
    Format::BPTC_UFLOAT,
    Format::R8_UI2F,
    Format::R8_I2F,
    Format::R16_UI2F,
    Format::R16_I2F,
    Format::R32_UI2F,
    Format::R32_I2F,
    Format::RG8_UI2F,
    Format::RG8_I2F,
    Format::RG16_UI2F,
    Format::RG16_I2F,
    Format::RG32_UI2F,
    Format::RG32_I2F,
    Format::RGB8_UI2F,
    Format::RGB8_I2F,
    Format::RGB16_UI2F,
    Format::RGB16_I2F,
    Format::RGB32_UI2F,
    Format::RGB32_I2F,
    Format::RGBA8_UI2F,
    Format::RGBA8_I2F,
    Format::RGBA16_UI2F,
    Format::RGBA16_I2F,
    Format::RGBA32_UI2F,
    Format::RGBA32_I2F,
    Format::RGB10A2SN,
    Format::RGB10A2I,
    Format::RGB10A2_UI2F,
    Format::RGB10A2_I2F,
    Format::RGBX8,
    Format::RGBX8SN,
    Format::RGBX8UI,
    Format::RGBX8I,
    Format::RGBX16F,
    Format::RGBX16,
    Format::RGBX16SN,
    Format::RGBX16UI,
    Format::RGBX16I,
    Format::RGBX32F,
    Format::RGBX32UI,
    Format::RGBX32I,
    Format::RGBA_ASTC_4x4,
    Format::RGBA_ASTC_5x4,
    Format::RGBA_ASTC_5x5,
    Format::RGBA_ASTC_6x5,
    Format::RGBA_ASTC_6x6,
    Format::RGBA_ASTC_8x5,
    Format::RGBA_ASTC_8x6,
    Format::RGBA_ASTC_8x8,
    Format::RGBA_ASTC_10x5,
    Format::RGBA_ASTC_10x6,
    Format::RGBA_ASTC_10x8,
    Format::RGBA_ASTC_10x10,
    Format::RGBA_ASTC_12x10,
    Format::RGBA_ASTC_12x12,
    Format::RGBA_ASTC_4x4_SRGB,
    Format::RGBA_ASTC_5x4_SRGB,
    Format::RGBA_ASTC_5x5_SRGB,
    Format::RGBA_ASTC_6x5_SRGB,
    Format::RGBA_ASTC_6x6_SRGB,
    Format::RGBA_ASTC_8x5_SRGB,
    Format::RGBA_ASTC_8x6_SRGB,
    Format::RGBA_ASTC_8x8_SRGB,
    Format::RGBA_ASTC_10x5_SRGB,
    Format::RGBA_ASTC_10x6_SRGB,
    Format::RGBA_ASTC_10x8_SRGB,
    Format::RGBA_ASTC_10x10_SRGB,
    Format::RGBA_ASTC_12x10_SRGB,
    Format::RGBA_ASTC_12x12_SRGB,
    Format::BGR5,
    Format::BGR5A1,
    Format::BGR565,
    Format::A1BGR5,
    Format::BGRA8,
    Format::BGRX8,
    Format::BGRA8_SRGB,
    Format::BGRX8_SRGB,
#if defined(LW_TEGRA)
    Format::Enum(LWN_FORMAT_PRIVATE_RGB_ETC1),
    Format::Enum(LWN_FORMAT_PRIVATE_RGBA_ETC1),
    Format::Enum(LWN_FORMAT_PRIVATE_RGB_ETC1_SRGB),
    Format::Enum(LWN_FORMAT_PRIVATE_RGBA_ETC1_SRGB),
#endif
};

static const int NUM_FORMATS = int(__GL_ARRAYSIZE(formats));

struct FormatBlockInfo {
    Format format;
    LWNuint blockWidth, blockHeight;
    LWNuint blockSize;
};

static const FormatBlockInfo* GetFormatBlockInfo(const FormatDesc& fmtDesc)
{
    static const FormatBlockInfo blocks[] = {
#define FORMATBLOCK(fmt, bw, bh, bs) { Format::fmt, bw, bh, bs }
#define PRIVATEBLOCK(fmt, bw, bh, bs) { Format::Enum(LWN_FORMAT_PRIVATE_ ## fmt), bw, bh, bs }
        FORMATBLOCK(RGB_DXT1,               4,  4,  8),
        FORMATBLOCK(RGBA_DXT1,              4,  4,  8),
        FORMATBLOCK(RGBA_DXT3,              4,  4,  16),
        FORMATBLOCK(RGBA_DXT5,              4,  4,  16),
        FORMATBLOCK(RGB_DXT1_SRGB,          4,  4,  8),
        FORMATBLOCK(RGBA_DXT1_SRGB,         4,  4,  8),
        FORMATBLOCK(RGBA_DXT3_SRGB,         4,  4,  16),
        FORMATBLOCK(RGBA_DXT5_SRGB,         4,  4,  16),
        FORMATBLOCK(RGTC1_UNORM,            4,  4,  8),
        FORMATBLOCK(RGTC1_SNORM,            4,  4,  8),
        FORMATBLOCK(RGTC2_UNORM,            4,  4,  16),
        FORMATBLOCK(RGTC2_SNORM,            4,  4,  16),
        FORMATBLOCK(BPTC_UNORM,             4,  4,  16),
        FORMATBLOCK(BPTC_UNORM_SRGB,        4,  4,  16),
        FORMATBLOCK(BPTC_SFLOAT,            4,  4,  16),
        FORMATBLOCK(BPTC_UFLOAT,            4,  4,  16),
        FORMATBLOCK(RGBA_ASTC_4x4,          4,  4,  16),
        FORMATBLOCK(RGBA_ASTC_5x4,          5,  4,  16),
        FORMATBLOCK(RGBA_ASTC_5x5,          5,  5,  16),
        FORMATBLOCK(RGBA_ASTC_6x5,          6,  5,  16),
        FORMATBLOCK(RGBA_ASTC_6x6,          6,  6,  16),
        FORMATBLOCK(RGBA_ASTC_8x5,          8,  5,  16),
        FORMATBLOCK(RGBA_ASTC_8x6,          8,  6,  16),
        FORMATBLOCK(RGBA_ASTC_8x8,          8,  8,  16),
        FORMATBLOCK(RGBA_ASTC_10x5,         10, 5,  16),
        FORMATBLOCK(RGBA_ASTC_10x6,         10, 6,  16),
        FORMATBLOCK(RGBA_ASTC_10x8,         10, 8,  16),
        FORMATBLOCK(RGBA_ASTC_10x10,        10, 10, 16),
        FORMATBLOCK(RGBA_ASTC_12x10,        12, 10, 16),
        FORMATBLOCK(RGBA_ASTC_12x12,        12, 12, 16),
        FORMATBLOCK(RGBA_ASTC_4x4_SRGB,     4,  4,  16),
        FORMATBLOCK(RGBA_ASTC_5x4_SRGB,     5,  4,  16),
        FORMATBLOCK(RGBA_ASTC_5x5_SRGB,     5,  5,  16),
        FORMATBLOCK(RGBA_ASTC_6x5_SRGB,     6,  5,  16),
        FORMATBLOCK(RGBA_ASTC_6x6_SRGB,     6,  6,  16),
        FORMATBLOCK(RGBA_ASTC_8x5_SRGB,     8,  5,  16),
        FORMATBLOCK(RGBA_ASTC_8x6_SRGB,     8,  6,  16),
        FORMATBLOCK(RGBA_ASTC_8x8_SRGB,     8,  8,  16),
        FORMATBLOCK(RGBA_ASTC_10x5_SRGB,    10, 5,  16),
        FORMATBLOCK(RGBA_ASTC_10x6_SRGB,    10, 6,  16),
        FORMATBLOCK(RGBA_ASTC_10x8_SRGB,    10, 8,  16),
        FORMATBLOCK(RGBA_ASTC_10x10_SRGB,   10, 10, 16),
        FORMATBLOCK(RGBA_ASTC_12x10_SRGB,   12, 10, 16),
        FORMATBLOCK(RGBA_ASTC_12x12_SRGB,   12, 12, 16),
#if defined(LW_TEGRA)
        PRIVATEBLOCK(RGB_ETC1,               4,  4,  8),
        PRIVATEBLOCK(RGBA_ETC1,              4,  4, 16),
        PRIVATEBLOCK(RGB_ETC1_SRGB,          4,  4,  8),
        PRIVATEBLOCK(RGBA_ETC1_SRGB,         4,  4, 16),
#endif

#undef FORMATBLOCK
    };
    for (LWNuint i = 0; i < __GL_ARRAYSIZE(blocks); i++) {
        if (fmtDesc.format == blocks[i].format) {
            return &blocks[i];
        }
    }
    return NULL;
}



class LWNCopyImage
{
public:
    LWNTEST_CppMethods();

    int m_texDim;
    LWNCopyImage(int texDim) {
        m_texDim = texDim;
    }
};

lwString LWNCopyImage::getDescription() const
{
    lwStringBuf sb;
    sb << "Test the CopyImage API in LWN. Tests each color format (excluding compressed, depth and\n"
          "stencil formats) in the following way:\n"
          " * Creates a " << texSize << "x" << texSize << " surface and fills it with sequential data.\n"
          " * Copies the entire contents of this into a second texture.\n"
          " * Copies two subsets of the contents into the second texture at easily identified locations.\n"
          "We repeat this test 3 times, with a TEXTURE_2D_ARRAY target as destination (" << texSize << "x" << texSize << "x2):\n"
          " * The first time the destination is miplevel 0 of layer 0.\n"
          " * The second time to miplevel 0 of layer 1.\n"
          " * The third time to miplevel 1 of layer 0.\n";
    return sb.str();
}

int LWNCopyImage::isSupported() const
{
    return lwogCheckLWNAPIVersion(31, 1);
}

static void memCopyImage(uint8_t *srcMem, LWNuint srcX, LWNuint srcY,
                         uint8_t *dstMem, LWNuint dstX, LWNuint dstY, LWNuint dstWidth,
                         LWNuint width, LWNuint height, int bpp, bool isSigned,
                         const FormatBlockInfo *fbi)
{
    LWNuint srcWidth = texSize;
    LWNuint y;
    assert(!(fbi && isSigned));

    if (fbi) {
        // Compressed format, so colwert all parameters into blocks.
        bpp = fbi->blockSize;
        srcWidth = BLOCK_DIV(srcWidth, fbi->blockWidth);
        dstWidth = BLOCK_DIV(dstWidth, fbi->blockWidth);

        // So if the source or destination is not aligned properly, then the
        // width and height of the covered region might differ between the
        // source and destination. So we'll mandate that source and destination
        // must always be block aligned.
        assert(srcX % fbi->blockWidth == 0);
        assert(srcY % fbi->blockHeight == 0);
        assert(dstX % fbi->blockWidth == 0);
        assert(dstY % fbi->blockHeight == 0);

        srcX = BLOCK_DIV(srcX, fbi->blockWidth);
        srcY = BLOCK_DIV(srcY, fbi->blockHeight);
        dstX = BLOCK_DIV(dstX, fbi->blockWidth);
        dstY = BLOCK_DIV(dstY, fbi->blockHeight);
        width = BLOCK_DIV(width, fbi->blockWidth);
        height = BLOCK_DIV(height, fbi->blockHeight);
    }

    if (isSigned) {
        for (y = 0; y < height; y++)
        {
            uint8_t *src = srcMem + (((srcY + y) * srcWidth + srcX) * bpp);
            uint8_t *dst = dstMem + (((dstY + y) * dstWidth + dstX) * bpp);
            for (LWNuint c = 0; c < width * bpp; c++) {
                // Force the canonical representation of -1 for signed 8-bit formats
                *dst = (*src == 0x80) ? 0x81 : *src;
                src++;
                dst++;
            }
        }
    } else {
        for (y = 0; y < height; y++)
        {
            uint8_t *src = srcMem + (((srcY + y) * srcWidth + srcX) * bpp);
            uint8_t *dst = dstMem + (((dstY + y) * dstWidth + dstX) * bpp);
            memcpy(dst, src, width * bpp);
        }
    }
}

static void bothCopyImage(
        QueueCommandBuffer &queueCB, Texture *srcTex, Texture *dstTex,
        uint8_t *srcMem, uint8_t *dstMem, LWNuint bpp, bool isSigned, const FormatBlockInfo *fbi,
        int dstLevel, int dstLayer, int dstWidth,
        int srcX, int srcY, int dstX, int dstY,
        int width, int height,
        int texDim)
{
    memCopyImage(srcMem, srcX, srcY,
                 dstMem, dstX, dstY, dstWidth,
                 width, height, bpp, isSigned, fbi);

    CopyRegion src;
    CopyRegion dst;

    if (texDim == TEX1D) {
        // 1D array textures are treated by hardware as "w x 1 x layers", but
        // the API treats them as "w x layers x 1".

        src.xoffset = srcX;
        src.yoffset = 0;
        src.zoffset = 0;
        src.width = width;
        src.height = 1;
        src.depth = 1;

        dst.xoffset = dstX;
        dst.yoffset = dstLayer;
        dst.zoffset = 0;
        dst.width = width;
        dst.height = 1;
        dst.depth = 1;

    } else {

        // down align to block-width if we copy from miplevel 0 to a higher miplevel.
        // Src region could be non-block-aligned. Both regions need to
        // the same. Down-align because up-align could bring us out of 
        // bounds.
        if (fbi && dstLevel) {
            width = fbi->blockWidth * (width / fbi->blockWidth);
            height = fbi->blockHeight * (height / fbi->blockHeight);
        }

        src.xoffset = srcX;
        src.yoffset = srcY;
        src.zoffset = 0;
        src.width = width;
        src.height = height;
        src.depth = 1;

        dst.xoffset = dstX;
        dst.yoffset = dstY;
        dst.zoffset = dstLayer;
        dst.width = width;
        dst.height = height;
        dst.depth = 1;
    }

    TextureView levelView;
    levelView.SetDefaults().SetLevels(dstLevel, 1);
    queueCB.CopyTextureToTexture(srcTex, NULL, &src, dstTex, &levelView, &dst, CopyFlags::NONE);
}


void LWNCopyImage::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const int cellWidth = 6*3;
    const int cellHeight = (NUM_FORMATS+5)/6;
    int cellNum = 0;
    cellTestInit(cellWidth, cellHeight);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.submit();

    // memory pools: allocate enough space for the largest formats
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();

    assert(m_texDim == TEX2D || m_texDim == TEX1D);

    if (m_texDim == TEX2D) {
        tb.SetTarget(TextureTarget::TARGET_2D_ARRAY);
        tb.SetSize3D(texSize, texSize, 2);
    } else {
        tb.SetTarget(TextureTarget::TARGET_1D_ARRAY);
        tb.SetSize3D(texSize, 2, 1);
    }

    tb.SetLevels(2);

    size_t texStorageSize = 0;
    for (int fmtIndex = 0; fmtIndex < NUM_FORMATS; fmtIndex++) {
        const FormatDesc &desc = *FormatDesc::findByFormat(formats[fmtIndex]);
        if (!(desc.flags & FLAG_COPYIMAGE)) {
            continue;
        }
        if (desc.format == Format::STENCIL8 && !g_lwnDeviceCaps.supportsStencil8) {
            continue;
        }
        if ((desc.flags & FLAG_COMPRESSED) && (m_texDim == TEX1D)) {
            continue;
        }

        tb.SetFormat(desc.format);
        texStorageSize = LW_MAX(texStorageSize, tb.GetPaddedStorageSize());
    }
    MemoryPoolAllocator texAllocator(device, NULL, 2 * texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    // allocate enough memory for our largest formats; 32-bit 4-component textures
    size_t maxMemSize = texStorageSize;
    MemoryPoolAllocator bufAllocator(device, NULL, 2 * maxMemSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *srcBuf = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, maxMemSize);
    BufferAddress srcBufAddr = srcBuf->GetAddress();
    Buffer *dstBuf = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, maxMemSize);
    BufferAddress dstBufAddr = dstBuf->GetAddress();
    uint8_t * srcMem = (uint8_t *) srcBuf->Map();
    uint8_t * dstMem = (uint8_t *) dstBuf->Map();
    uint8_t * simMem = new uint8_t[maxMemSize];

    // fill with a repeating pattern
    for (LWNuint i = 0; i < maxMemSize; i++) {
        srcMem[i] = (i & 0xFF);
    }

    for (int fmtIndex = 0; fmtIndex < NUM_FORMATS; fmtIndex++)
    {
        const FormatDesc &desc = *FormatDesc::findByFormat(formats[fmtIndex]);

        bool skip = false;
        // only check texture formats supported by CopyImage
        if (!(desc.flags & FLAG_COPYIMAGE)) {
            skip = true;
        }
        if (desc.format == Format::STENCIL8 && !g_lwnDeviceCaps.supportsStencil8) {
            skip = true;
        }

        if ((desc.flags & FLAG_COMPRESSED) && (m_texDim == TEX1D)) {
            skip = true;
        }

        if (skip) {
            for (int i = 0; i < 3; i++) {
                SetCellViewportScissorPadded(queueCB, cellNum % cellWidth, cellNum / cellWidth, 1);
                queueCB.ClearColor(0, 0.0, 0.0, 1.0);
                cellNum++;
            }
            continue;
        }

        DEBUG_PRINT(("Testing %s\n", desc.formatName));
        const Format fmt = desc.format;
        const LWNuint bpp = desc.stride;
        const bool compressed = (0 != (desc.flags & FLAG_COMPRESSED));
        const FormatBlockInfo *fbi = (compressed) ? GetFormatBlockInfo(desc) : NULL;

        for (LWNuint i = 0; i < maxMemSize; i++) {
            simMem[i] = (i & 0xFF);
        }

        tb.SetFormat(desc.format);
        if (FormatIsDepthStencil((LWNformat)(lwn::Format::Enum) desc.format)) {
            tb.SetFlags(tb.GetFlags() | TextureFlags::COMPRESSIBLE);
        }

        if (m_texDim == TEX2D) {
            tb.SetTarget(TextureTarget::TARGET_2D);
            tb.SetSize2D(texSize, texSize);
        } else {
            tb.SetTarget(TextureTarget::TARGET_1D);
            tb.SetSize1D(texSize);
        }

        Texture *srcTex = texAllocator.allocTexture(&tb);

        if (m_texDim == TEX2D) {
            tb.SetSize3D(texSize, texSize, 2);
            tb.SetTarget(TextureTarget::TARGET_2D_ARRAY);
        } else {
            tb.SetSize3D(texSize, 2, 1);
            tb.SetTarget(TextureTarget::TARGET_1D_ARRAY);
        }

        Texture *dstTex = texAllocator.allocTexture(&tb);

        for (LWNuint subtest = 0; subtest < 3; subtest++)
        {
            int layer = (subtest == 1) ? 1 : 0;
            int level = (subtest == 2) ? 1 : 0;
            int levelSize  = texSize >> level;
            LWNuint levelMemSize = 0;

            if (m_texDim == TEX2D) {
                levelMemSize = bpp * levelSize * levelSize;
            } else {
                levelMemSize = bpp * levelSize;
            }

            DEBUG_PRINT(("Subtest %u\n", subtest));

            CopyRegion b2tRegion;
            // Initialize the source texture
            if (m_texDim == TEX2D) {
                b2tRegion = { 0, 0, 0, texSize, texSize, 1 };
            } else {
                b2tRegion = { 0, 0, 0, texSize, 1, 1 };
            }

            queueCB.CopyBufferToTexture(srcBufAddr, srcTex, NULL, &b2tRegion, CopyFlags::NONE);

            // Determine whether it's a signed normalized format
            bool isSigned = false;
            switch (fmt) {
                case Format::R8SN:
                case Format::RG8SN:
                case Format::RGBA8SN:
                case Format::RGBX8SN:
                    isSigned = true;
                    break;
                default:
                    // pass-through
                    break;
            }

            if (m_texDim == TEX2D) {
                // Copy 1 - Complete texture copy
                bothCopyImage(
                        queueCB, srcTex, dstTex,
                        srcMem, simMem, bpp, isSigned, fbi,
                        level, layer, levelSize,
                        0, 0,  0, 0,  levelSize, levelSize,
                        m_texDim);

                // Copy 2 - Regional copy near bottom of image
                if (!compressed && (level != 0)) {
                    bothCopyImage(
                            queueCB, srcTex, dstTex,
                            srcMem, simMem, bpp, isSigned, fbi,
                            level, layer, levelSize,
                            23, 18,  2, 72,  13, 26,
                            m_texDim);
                }

                // Copy 3 - Regional copy near top of image (for easier debugging)
                LWNuint bW = fbi ? fbi->blockWidth : 1;
                LWNuint bH = fbi ? fbi->blockHeight : 1;
                bothCopyImage(
                        queueCB, srcTex, dstTex,
                        srcMem, simMem, bpp, isSigned, fbi,
                        level, layer, levelSize,
                        1*bW, 1*bH,  2*bW, 2*bH,  1*bW, 1*bH,
                        m_texDim);
            } else {
                // Copy 1 - Complete texture copy
                bothCopyImage(
                        queueCB, srcTex, dstTex,
                        srcMem, simMem, bpp, isSigned, fbi,
                        level, layer, levelSize,
                        0, 0,  0, 0,  levelSize, 1,
                        m_texDim);

                // Copy 2 - Regional copy near bottom of image
                if (!compressed && (level != 0)) {
                    bothCopyImage(
                            queueCB, srcTex, dstTex,
                            srcMem, simMem, bpp, isSigned, fbi,
                            level, layer, levelSize,
                            23, 0,  2, 0,  13, 1,
                            m_texDim);
                }

                // Copy 3 - Regional copy near top of image (for easier debugging)
                LWNuint bW = fbi ? fbi->blockWidth : 1;
                bothCopyImage(
                        queueCB, srcTex, dstTex,
                        srcMem, simMem, bpp, isSigned, fbi,
                        level, layer, levelSize,
                        1*bW, 0,  2*bW, 0,  1*bW, 1,
                        m_texDim);

            }

            // TODO
            // - Add a mirror case

            // Readback and compare
            CopyRegion t2bRegion;
            TextureView t2bView;
            if (m_texDim == TEX2D) {
                t2bRegion = { 0, 0, layer, levelSize, levelSize, 1 };
            } else {
                t2bRegion = { 0, layer, 0, levelSize, 1, 1 };
            }

            t2bView.SetDefaults().SetLevels(level, 1);
            queueCB.CopyTextureToBuffer(dstTex, &t2bView, &t2bRegion, dstBufAddr, CopyFlags::NONE);

            queueCB.submit();
            queue->Finish();

            // Specialty comparison function
            bool failed = false;
            const int failureThreshold = 25;
            int failures = 0;
            LWNuint numCheckedBytesPerPixel = bpp;
            switch (fmt) {
                case Format::RGBX8_SRGB:
                case Format::RGBX8:
                case Format::BGRX8_SRGB:
                case Format::BGRX8:
                    numCheckedBytesPerPixel = 3;
                    break;
                case Format::RGBX16:
                case Format::RGBX16F:
                    numCheckedBytesPerPixel = 6;
                    break;
                case Format::RGBX32F:
                    numCheckedBytesPerPixel = 12;
                    break;
                case Format::DEPTH32F_STENCIL8:
                    numCheckedBytesPerPixel = 5;
                    break;
                default:
                    break;
            }
            for (LWNuint off = 0; off < levelMemSize; off += bpp) {
                for (LWNuint b = 0; b < numCheckedBytesPerPixel; b++) {
                    if (simMem[off + b] != dstMem[off + b]) {

                        // For RGB5 and BGR5 formats, there is an extra
                        // "alpha" bit in the stored memory, which won't be
                        // preserved by the copy operation.  If the 7 LSBs of
                        // an odd byte match, consider the results to match.
                        if ((b & 1) && (fmt == Format::RGB5 || fmt == Format::BGR5)) {
                            if ((simMem[off + b] & 0x7F) == (dstMem[off + b] & 0x7F)) {
                                continue;
                            }
                        }
#if DEBUG_MODE
                        LWNuint x, y;

                        if (fbi) {
                            LWNuint block = off / fbi->blockSize;
                            LWNuint texWidthInBlocks = BLOCK_DIV(levelSize, fbi->blockWidth);
                            x = (block % texWidthInBlocks) * fbi->blockWidth;
                            if (m_texDim == TEX2D) {
                                y = (block / texWidthInBlocks) * fbi->blockHeight;
                            } else {
                                y = 1;
                            }
                        } else {
                            LWNuint pixel = off / bpp;
                            x = pixel % (levelSize);
                            if (m_texDim == TEX2D) {
                                LWNuint pixel = off / bpp;
                                y = pixel / (levelSize);
                            } else {
                                y = 1;
                            }
                        }

                        DEBUG_PRINT(("mismatch @ level %u, off %u: (%u,%u,%u)  sim: %u  gpu: %u\n",
                                    level, off + b, x, y, layer, simMem[off + b], dstMem[off + b]));
                        failures++;
                        if (failures > failureThreshold) break;
#endif
                        failed = true;
                    }
                }
                if (failures > failureThreshold) break;
            }
            SetCellViewportScissorPadded(queueCB, cellNum % cellWidth, cellNum / cellWidth, 1);
            queueCB.ClearColor(0, failed ? 1.0 : 0.0, failed ? 0.0 : 1.0, 0.0);
            cellNum++;
        }

        texAllocator.freeTexture(srcTex);
        texAllocator.freeTexture(dstTex);
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();

    bufAllocator.freeBuffer(srcBuf);
    bufAllocator.freeBuffer(dstBuf);
    delete [] simMem;
}

class LWNCopyImageLinear
{
public:
    LWNTEST_CppMethods();

    int m_texTarget;
    LWNCopyImageLinear(int texTarget) {
        m_texTarget = texTarget;
    }
};

lwString LWNCopyImageLinear::getDescription() const
{
    lwStringBuf sb;
    sb << "Test the CopyImage API in LWN. Tests each color format (excluding compressed, depth and\n"
          "stencil formats) in the following way:\n"
          " * Creates a " << texSize << "x" << texSize << " surface and fills it with sequential data.\n"
          " * Copies the entire contents of this into a second texture.\n"
          " * Copies two subsets of the contents into the second texture at easily identified locations.\n"
          "We repeat this test with a TARGET_2D target and a TARGET_RECTANGLE as destinations (" << texSize << "x" << texSize << "x2):\n";

    return sb.str();
}

int LWNCopyImageLinear::isSupported() const
{
    return lwogCheckLWNAPIVersion(31, 1);
}

static void bothCopyImageLinear(
        Queue *queue, QueueCommandBuffer &queueCB, Texture *srcTex, Texture *dstTex,
        uint8_t *srcMem, uint8_t *dstMem, LWNuint bpp, bool isSigned, const FormatBlockInfo *fbi,
        int dstWidth,
        int srcX, int srcY, int dstX, int dstY,
        int width, int height)
{
    memCopyImage(srcMem, srcX, srcY,
                 dstMem, dstX, dstY, dstWidth,
                 width, height, bpp, isSigned, fbi);

    CopyRegion src = { srcX, srcY, 0, width, height, 1 };
    CopyRegion dst = { dstX, dstY, 0, width, height, 1 };

    queueCB.CopyTextureToTexture(srcTex, NULL, &src, dstTex, NULL, &dst, CopyFlags::NONE);
}


void LWNCopyImageLinear::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const int cellWidth = 6*3;
    const int cellHeight = (NUM_FORMATS+5)/6;
    int cellNum = 0;
    cellTestInit(cellWidth, cellHeight);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    // memory pools: allocate enough space for the largest formats
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetFlags(TextureFlags::LINEAR);

    assert(m_texTarget == TEX2D || m_texTarget == RECTANGLE);

    if (m_texTarget == TEX2D) {
        tb.SetTarget(TextureTarget::TARGET_2D);
    } else {
        tb.SetTarget(TextureTarget::TARGET_RECTANGLE);
    }

    tb.SetSize2D(texSize, texSize);

    size_t texStorageSize = 0;
    for (int fmtIndex = 0; fmtIndex < NUM_FORMATS; fmtIndex++) {
        const FormatDesc &desc = *FormatDesc::findByFormat(formats[fmtIndex]);

        if (!(desc.flags & FLAG_COPYIMAGE)) {
            continue;
        }
        if (desc.format == Format::STENCIL8 && !g_lwnDeviceCaps.supportsStencil8) {
            continue;
        }
        if (desc.stride == 0) {
            continue;
        }

        tb.SetFormat(desc.format);

        LWNint linearTexAlignment = -1;
        device->GetInteger(DeviceInfo::LINEAR_RENDER_TARGET_STRIDE_ALIGNMENT, &linearTexAlignment);
        if (linearTexAlignment & (linearTexAlignment - 1)) {
            LWNFailTest();
            return;
        }

        LWNint stride = ROUND_UP(desc.stride * texSize, linearTexAlignment);

        tb.SetStride(stride);

        texStorageSize = LW_MAX(texStorageSize, tb.GetPaddedStorageSize());
    }
    MemoryPoolAllocator texAllocator(device, NULL, 2 * texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    // allocate enough memory for our largest formats; 32-bit 4-component textures
    size_t maxMemSize = texStorageSize;
    MemoryPoolAllocator bufAllocator(device, NULL, 2 * maxMemSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *srcBuf = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, maxMemSize);
    BufferAddress srcBufAddr = srcBuf->GetAddress();
    Buffer *dstBuf = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, maxMemSize);
    BufferAddress dstBufAddr = dstBuf->GetAddress();
    uint8_t * srcMem = (uint8_t *) srcBuf->Map();
    uint8_t * dstMem = (uint8_t *) dstBuf->Map();
    uint8_t * simMem = new uint8_t[maxMemSize];

    // fill with a repeating pattern
    for (LWNuint i = 0; i < maxMemSize; i++) {
        srcMem[i] = (i & 0xFF);
    }

    for (int fmtIndex = 0; fmtIndex < NUM_FORMATS; fmtIndex++)
    {
        const FormatDesc &desc = *FormatDesc::findByFormat(formats[fmtIndex]);

        bool skip = false;
        // only check texture formats supported by CopyImage
        if (!(desc.flags & FLAG_COPYIMAGE)) {
            skip = true;
        }
        if (desc.format == Format::STENCIL8 && !g_lwnDeviceCaps.supportsStencil8) {
            skip = true;
        }
        if (desc.stride == 0) {
            skip = true;
        }

        if (skip) {
            SetCellViewportScissorPadded(queueCB, cellNum % cellWidth, cellNum / cellWidth, 1);
            queueCB.ClearColor(0, 0.0, 0.0, 1.0);
            cellNum++;
            continue;
        }

        DEBUG_PRINT(("Testing %s\n", desc.formatName));
        const Format fmt = desc.format;
        const LWNuint bpp = desc.stride;
        const bool compressed = (0 != (desc.flags & FLAG_COMPRESSED));
        const FormatBlockInfo *fbi = (compressed) ? GetFormatBlockInfo(desc) : NULL;

        for (LWNuint i = 0; i < maxMemSize; i++) {
            simMem[i] = (i & 0xFF);
        }

        tb.SetFormat(desc.format);
        if (FormatIsDepthStencil((LWNformat)(lwn::Format::Enum) desc.format)) {
            tb.SetFlags(tb.GetFlags() | TextureFlags::COMPRESSIBLE);
        }

        if (m_texTarget == TEX2D) {
            tb.SetTarget(TextureTarget::TARGET_2D);
        } else {
            tb.SetTarget(TextureTarget::TARGET_RECTANGLE);
        }

        tb.SetSize2D(texSize, texSize);

        LWNint linearTexAlignment = -1;
        device->GetInteger(DeviceInfo::LINEAR_RENDER_TARGET_STRIDE_ALIGNMENT, &linearTexAlignment);
        if (linearTexAlignment & (linearTexAlignment - 1)) {
            LWNFailTest();
            return;
        }

        LWNint stride = ROUND_UP(desc.stride * texSize, linearTexAlignment);

        tb.SetStride(stride);

        Texture *srcTex = texAllocator.allocTexture(&tb);
        Texture *dstTex = texAllocator.allocTexture(&tb);

        LWNuint texMemSize = bpp * texSize * texSize;

        // Initialize the source texture

        CopyRegion b2tRegion = { 0, 0, 0, texSize, texSize, 1 };
        queueCB.CopyBufferToTexture(srcBufAddr, srcTex, NULL, &b2tRegion, CopyFlags::NONE);

        // Determine whether it's a signed normalized format
        bool isSigned = false;
        switch (fmt) {
            case Format::R8SN:
            case Format::RG8SN:
            case Format::RGBA8SN:
            case Format::RGBX8SN:
                isSigned = true;
                break;
            default:
                // pass-through
                break;
        }

        // Complete texture copy
        bothCopyImageLinear(
                queue, queueCB, srcTex, dstTex,
                srcMem, simMem, bpp, isSigned, fbi,
                texSize,
                0, 0,  0, 0,  texSize, texSize);

        // TODO
        // - Add a mirror case

        // Readback and compare
        TextureView t2bView;

        CopyRegion t2bRegion = { 0, 0, 0, texSize, texSize, 1 };
        t2bView.SetDefaults();
        queueCB.CopyTextureToBuffer(dstTex, &t2bView, &t2bRegion, dstBufAddr, CopyFlags::NONE);

        queueCB.submit();
        queue->Finish();

        // Specialty comparison function
        bool failed = false;
        const int failureThreshold = 25;
        int failures = 0;
        LWNuint numCheckedBytesPerPixel = bpp;
        switch (fmt) {
            case Format::RGBX8_SRGB:
            case Format::RGBX8:
            case Format::BGRX8_SRGB:
            case Format::BGRX8:
                numCheckedBytesPerPixel = 3;
                break;
            case Format::RGBX16:
            case Format::RGBX16F:
                numCheckedBytesPerPixel = 6;
                break;
            case Format::RGBX32F:
                numCheckedBytesPerPixel = 12;
                break;
            default:
                break;
        }
        for (LWNuint off = 0; off < texMemSize; off += bpp) {
            for (LWNuint b = 0; b < numCheckedBytesPerPixel; b++) {
                if (simMem[off + b] != dstMem[off + b]) {

                    // For RGB5 and BGR5 formats, there is an extra
                    // "alpha" bit in the stored memory, which won't be
                    // preserved by the copy operation.  If the 7 LSBs of
                    // an odd byte match, consider the results to match.
                    if ((b & 1) && (fmt == Format::RGB5 || fmt == Format::BGR5)) {
                        if ((simMem[off + b] & 0x7F) == (dstMem[off + b] & 0x7F)) {
                            continue;
                        }
                    }
#if DEBUG_MODE
                    LWNuint x, y;
                    if (fbi) {
                        LWNuint block = off / fbi->blockSize;
                        LWNuint texWidthInBlocks = BLOCK_DIV(texSize, fbi->blockWidth);
                        y = (block / texWidthInBlocks) * fbi->blockHeight;
                        x = (block % texWidthInBlocks) * fbi->blockWidth;
                    } else {
                        LWNuint pixel = off / bpp;
                        y = pixel / (texSize);
                        x = pixel % (texSize);
                    }
                    DEBUG_PRINT(("mismatch @ level %u, off %u: (%u,%u,%u)  sim: %u  gpu: %u\n",
                                0, off + b, x, y, 0, simMem[off + b], dstMem[off + b]));
                    failures++;
                    if (failures > failureThreshold) break;
#endif
                    failed = true;
                }
            }
            if (failures > failureThreshold) break;
        }
        SetCellViewportScissorPadded(queueCB, cellNum % cellWidth, cellNum / cellWidth, 1);
        queueCB.ClearColor(0, failed ? 1.0 : 0.0, failed ? 0.0 : 1.0, 0.0);

        cellNum++;
        texAllocator.freeTexture(srcTex);
        texAllocator.freeTexture(dstTex);
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();

    bufAllocator.freeBuffer(srcBuf);
    bufAllocator.freeBuffer(dstBuf);
    delete [] simMem;
}

OGTEST_CppTest(LWNCopyImageLinear, lwn_copy_image_linear_2d, ((TEX2D)));
OGTEST_CppTest(LWNCopyImageLinear, lwn_copy_image_linear_rectangle, ((RECTANGLE)));
OGTEST_CppTest(LWNCopyImage, lwn_copy_image, ((TEX2D)));
OGTEST_CppTest(LWNCopyImage, lwn_copy_image_1d, ((TEX1D)));
