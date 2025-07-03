/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;

#define DEBUG_VIZ 0     // display dst texture contents

const int texWidth  = 134; // don't use same w&h and use npot
const int texHeight = 144;

// Value of 1.0 in various different number formats
enum OneBits
{
    ONE_BITS_INT8   = 0x7F,
    ONE_BITS_UINT8  = 0xFF,
    ONE_BITS_INT16  = 0x7FFF,
    ONE_BITS_UINT16 = 0xFFFF,
    ONE_BITS_INT32  = 0x7FFFFFFF,
    ONE_BITS_UINT32 = 0xFFFFFFFF,
    ONE_BITS_FP16   = 0x3C00,
    ONE_BITS_FP32   = 0x3F800000
};

struct Desc
{
    Format   format;
    uint32_t oneBits;
};

// All formats exercised by tests in this file
static const Desc ALL_FORMATS[] = {
#define DESC(fmt, ob) { Format::fmt, ONE_BITS_ ## ob }
    DESC(R8,         UINT8),
    DESC(R8SN,       INT8),
    DESC(R8UI,       UINT8),
    DESC(R8I,        INT8),
    DESC(R16F,       FP16),
    DESC(R16,        UINT16),
    DESC(R16SN,      INT16),
    DESC(R16UI,      UINT16),
    DESC(R16I,       INT16),
    DESC(R32F,       FP32),
    DESC(R32UI,      UINT32),
    DESC(R32I,       INT32),
    DESC(RG8,        UINT8),
    DESC(RG8SN,      INT8),
    DESC(RG8UI,      UINT8),
    DESC(RG8I,       INT8),
    DESC(RG16F,      FP16),
    DESC(RG16,       UINT16),
    DESC(RG16SN,     INT16),
    DESC(RG16UI,     UINT16),
    DESC(RG16I,      INT16),
    DESC(RG32F,      FP32),
    DESC(RG32UI,     UINT32),
    DESC(RG32I,      INT32),
    DESC(RGB8,       UINT8),
    DESC(RGB8SN,     INT8),
    DESC(RGB8UI,     UINT8),
    DESC(RGB8I,      INT8),
    DESC(RGB16F,     FP16),
    DESC(RGB16,      UINT16),
    DESC(RGB16SN,    INT16),
    DESC(RGB16UI,    UINT16),
    DESC(RGB16I,     INT16),
    DESC(RGB32F,     FP32),
    DESC(RGB32UI,    UINT32),
    DESC(RGB32I,     INT32),
    DESC(RGBA8,      UINT8),
    DESC(RGBA32F,    FP32),
    DESC(RGBA8SN,    INT8),
    DESC(RGBA8UI,    UINT8),
    DESC(RGBA8I,     INT8),
    DESC(RGBA16F,    FP16),
    DESC(RGBA16,     UINT16),
    DESC(RGBA16SN,   INT16),
    DESC(RGBA16UI,   UINT16),
    DESC(RGBA16I,    INT16),
    DESC(RGBA32F,    FP32),
    DESC(RGBA32UI,   UINT32),
    DESC(RGBA32I,    INT32),
#if 0
    // Depth, stencil, and depth+stencil all swizzle differently. It's not yet
    // known what the final output will look like.
    FORMATDESC(STENCIL8,            0 + 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_STENCIL),
    FORMATDESC(DEPTH16,             2 + 0, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH),
    FORMATDESC(DEPTH24,             4 + 0, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH),
    FORMATDESC(DEPTH32F,            4 + 0, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH),
    FORMATDESC(DEPTH24_STENCIL8,    3 + 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH | FLAG_STENCIL),
    FORMATDESC(DEPTH32F_STENCIL8,   4 + 4, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH | FLAG_STENCIL),
    FORMATDESC(RGBX8_SRGB,          4 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBA8_SRGB,          4 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBA4,               2,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(RGB5,                2,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE),
#endif
#if 0
    // These formats are not supported by CopyTextureToTexture for colwersion
    // and I didn't want to handcode format colwersion for these
    FORMATDESC(RGB5A1,              2,     FLOAT,       5,5,5,1,     FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(RGB565,              2,     FLOAT,       5,6,5,0,     FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(RGB10A2,             4,     FLOAT,       10,10,10,2,  FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGB10A2UI,           4,     UNSIGNED,                 FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R11G11B10F,          4,     FLOAT,       11,11,10,0,  FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
#endif
    DESC(RGBX8,      UINT8),
    DESC(RGBX8SN,    INT8),
    DESC(RGBX8UI,    UINT8),
    DESC(RGBX8I,     INT8),
    DESC(RGBX16F,    FP16),
    DESC(RGBX16,     UINT16),
    DESC(RGBX16SN,   INT16),
    DESC(RGBX16UI,   UINT16),
    DESC(RGBX16I,    INT16),
    DESC(RGBX32F,    FP32),
    DESC(RGBX32UI,   UINT32),
    DESC(RGBX32I,    INT32),
    DESC(BGRA8,      UINT8),
    DESC(BGRX8,      UINT8),
    DESC(BGRA8_SRGB, UINT8),
    DESC(BGRX8_SRGB, UINT8)
#undef FORMATDESC
};

static const int NUM_FORMATS = int(__GL_ARRAYSIZE(ALL_FORMATS));

namespace
{
    class TestClearPattern
    {
    public:
        TestClearPattern(
            const Desc *desc,
            lwn::Buffer *readBackBuf,
            lwn::Texture *msaaResolveTex,
            const lwn::Texture *dstTex, lwn::TextureView *dstView, const lwn::CopyRegion dstReg) :
            m_testDesc(desc),
            m_formatDesc(FormatDesc::findByFormat(desc->format)),
            m_readBackBuf(readBackBuf),
            m_msaaResolveTex(msaaResolveTex),
            m_dstTex(dstTex),
            m_dstView(dstView),
            m_dstReg(dstReg),
            m_dstFormat(dstTex->GetFormat())
        {
        }

        template <int numComp, typename src_t> void copyComponents(uint32_t *dst, const void* src)
        {
            for (int i = 0; i < numComp; i++) {
                dst[i] = ((const src_t *)src)[i];
            }
        }

        // Decode raw component data into a uint32_t array
        void decodeToDwords(uint32_t *dst, const void *pixel)
        {
            switch (m_formatDesc->format)
            {
            case lwn::Format::R8:
            case lwn::Format::R8SN:
            case lwn::Format::R8UI:
            case lwn::Format::R8I:
                copyComponents<1, uint8_t>(dst, pixel);
                break;
            case lwn::Format::R16:
            case lwn::Format::R16SN:
            case lwn::Format::R16F:
            case lwn::Format::R16UI:
            case lwn::Format::R16I:
                copyComponents<1, uint16_t>(dst, pixel);
                break;
            case lwn::Format::R32F:
            case lwn::Format::R32UI:
            case lwn::Format::R32I:
                copyComponents<1, uint32_t>(dst, pixel);
                break;
            case lwn::Format::RG8:
            case lwn::Format::RG8SN:
            case lwn::Format::RG8UI:
            case lwn::Format::RG8I:
                copyComponents<2, uint8_t>(dst, pixel);
                break;
            case lwn::Format::RG16:
            case lwn::Format::RG16F:
            case lwn::Format::RG16SN:
            case lwn::Format::RG16UI:
            case lwn::Format::RG16I:
                copyComponents<2, uint16_t>(dst, pixel);
                break;
            case lwn::Format::RG32F:
            case lwn::Format::RG32UI:
            case lwn::Format::RG32I:
                copyComponents<2, uint32_t>(dst, pixel);
                break;
            case lwn::Format::RGB8UI:
            case lwn::Format::RGB8I:
            case lwn::Format::RGBX8UI:
            case lwn::Format::RGBX8I:
                copyComponents<3, uint8_t>(dst, pixel);
                break;
            case lwn::Format::RGB16UI:
            case lwn::Format::RGB16I:
            case lwn::Format::RGBX16UI:
            case lwn::Format::RGBX16I:
                copyComponents<3, uint16_t>(dst, pixel);
                break;
            case lwn::Format::RGB32UI:
            case lwn::Format::RGB32I:
            case lwn::Format::RGBX32UI:
            case lwn::Format::RGBX32I:
                copyComponents<3, uint32_t>(dst, pixel);
                break;
            case lwn::Format::RGBA8:
            case lwn::Format::RGBA8SN:
            case lwn::Format::RGBA8UI:
            case lwn::Format::RGBA8I:
            case lwn::Format::RGBX8:
            case lwn::Format::RGBX8SN:
            case lwn::Format::BGRA8:
            case lwn::Format::BGRA8_SRGB:
            case lwn::Format::BGRX8:
            case lwn::Format::BGRX8_SRGB:
                copyComponents<4, uint8_t>(dst, pixel);
                break;
            case lwn::Format::RGBA16:
            case lwn::Format::RGBA16F:
            case lwn::Format::RGBA16SN:
            case lwn::Format::RGBA16UI:
            case lwn::Format::RGBA16I:
            case lwn::Format::RGBX16:
            case lwn::Format::RGBX16SN:
            case lwn::Format::RGBX16F:
                copyComponents<4, uint16_t>(dst, pixel);
                break;
            case lwn::Format::RGBA32F:
            case lwn::Format::RGBA32UI:
            case lwn::Format::RGBA32I:
            case lwn::Format::RGBX32F:
                copyComponents<4, uint32_t>(dst, pixel);
                break;
            default:
                assert(false);
            }
        }

        int componentIndex(int idx) const
        {
            switch (m_formatDesc->format) {
            case lwn::Format::BGRX8:
            case lwn::Format::BGRA8:
            case lwn::Format::BGRA8_SRGB:
            case lwn::Format::BGRX8_SRGB:
                // Swap RGB -> BGR
                if (idx == 3)
                    return 3;
                return 2 - idx;
            default:
                break;
            }
            return idx;
        }

        bool expectPixel(const uint8_t *pixels, int x, int y, const uint32_t *expected)
        {
            int bytesPerPixel = m_formatDesc->stride;

            const uint8_t *src = pixels + m_dstReg.width*bytesPerPixel*y + x*bytesPerPixel;

            uint32_t rgba[4] = { 0, 0, 0, 0 };
            decodeToDwords(rgba, src);

            for (int i = 0; i < 4; i++) {
                int numBits = m_formatDesc->numBitsPerComponent[i];
                uint32_t v = rgba[componentIndex(i)];
                if (numBits != 0 && v != expected[i])
                    return false;
            }
            return true;
        }

        lwn::CopyRegion computeSquareRegion(lwn::CopyRegion d, int idx)
        {
            int width = m_dstReg.width;
            int height = m_dstReg.height;

            int subx = (idx & 1);
            int suby = (idx >> 1) & 1;

            int subw = width / 3;
            int subh = height / 3;

            d.xoffset = width / 2 * subx + subw / 3;
            d.yoffset = height / 2 * suby + subh / 3;
            d.width = subw - subw/3;
            d.height = subh - subh/3;
            return d;
        }

        void clearTexture(QueueCommandBuffer &queueCB, const lwn::CopyRegion *dstReg, const float *colors, const uint32_t *intColors)
        {
            switch (m_formatDesc->samplerComponentType) {
            case COMP_TYPE_INT:
                queueCB.ClearTexturei(m_dstTex, m_dstView, dstReg, (const int32_t *)intColors, LWN_CLEAR_COLOR_MASK_RGBA);
                break;
            case COMP_TYPE_UNSIGNED:
                queueCB.ClearTextureui(m_dstTex, m_dstView, dstReg, intColors, LWN_CLEAR_COLOR_MASK_RGBA);
                break;
            case COMP_TYPE_FLOAT:
                queueCB.ClearTexture(m_dstTex, m_dstView, dstReg, colors, LWN_CLEAR_COLOR_MASK_RGBA);
                break;
            default:
                assert(false);
            }
        }

        bool test(lwn::Queue* queue, QueueCommandBuffer &queueCB)
        {
            const float fullColor[4] = { 1.f, 1.f, 1.f, 1.f };
            uint32_t fullColorInt[4];
            const float colors[4][4] = {
                { 1.f,  0.f,  0.f,  1.f },
                { 0.f,  1.f,  0.f,  1.f },
                { 0.f,  0.f,  1.f,  1.f },
                { 0.f,  0.f,  0.f,  0.f }
            };
            uint32_t intColors[4][4];

            for (int i = 0; i < 4; i++) {
                fullColorInt[i] = fullColor[i] == 1.f ? m_testDesc->oneBits : 0;
                for (int j = 0; j < 4; j++) {
                    intColors[i][j] = colors[i][j] == 1.f ? m_testDesc->oneBits : 0;
                }
            }

            int width = m_dstReg.width;
            int height = m_dstReg.height;
            // Full level/layer clear with white.
            clearTexture(queueCB, &m_dstReg, fullColor, fullColorInt);

            // Four smaller clears with different colors.
            for (int i = 0; i < 4; i++) {
                lwn::CopyRegion dstReg = computeSquareRegion(m_dstReg, i);
                clearTexture(queueCB, &dstReg, colors[i], intColors[i]);
            }

            // Fill readBackBuf with a known pattern.
            uint8_t * dstMem = (uint8_t *)m_readBackBuf->Map();

            // Copy the raw bits and let getPixel try to deal with it
            if (m_msaaResolveTex) {
                assert(m_dstReg.xoffset == 0 && m_dstReg.yoffset == 0 && m_dstReg.zoffset == 0);
                assert(!m_dstView);
                queueCB.Downsample(m_dstTex, m_msaaResolveTex);
                queueCB.CopyTextureToBuffer(m_msaaResolveTex, nullptr, &m_dstReg, m_readBackBuf->GetAddress(), CopyFlags::NONE);
            }
            else {
                queueCB.CopyTextureToBuffer(m_dstTex, m_dstView, &m_dstReg, m_readBackBuf->GetAddress(), CopyFlags::NONE);
            }

            queueCB.submit();
            queue->Finish();

            // First test that the corners of the screen are white from the
            // first full level clear.
            if (!expectPixel(dstMem, 0, 0, fullColorInt))
                return false;
            if (!expectPixel(dstMem, width - 1, height - 1, fullColorInt))
                return false;

            // Check that all the cleared subrectangles have the expected color
            for (int i = 0; i < 4; i++) {
                const uint32_t *expectedColor = intColors[i];
                lwn::CopyRegion reg = computeSquareRegion(m_dstReg, i);

                // Check that the middle of the cleared rectangle is cleared
                int midx = reg.xoffset + reg.width / 2;
                int midy = reg.yoffset + reg.height / 2;
                if (!expectPixel(dstMem, midx, midy, expectedColor))
                    return false;

                // Check that pixels match the expected clear color at the
                // clear region border
                for (int y = reg.yoffset; y < reg.height; y++) {
                    int left = reg.xoffset;
                    int right = reg.xoffset + reg.width - 1;

                    if (!expectPixel(dstMem, left, y, expectedColor))
                        return false;
                    if (!expectPixel(dstMem, right, y, expectedColor))
                        return false;

                    // Check that the clear didn't bleed outside the region
                    if (!expectPixel(dstMem, left - 1, y, fullColorInt))
                        return false;
                    if (!expectPixel(dstMem, right + 1, y, fullColorInt))
                        return false;
                }

                for (int x = reg.xoffset; x < reg.width; x++) {
                    int top = reg.yoffset;
                    int bottom = reg.yoffset + reg.height - 1;

                    if (!expectPixel(dstMem, x, top, expectedColor))
                        return false;
                    if (!expectPixel(dstMem, x, bottom, expectedColor))
                        return false;

                    // Check that the clear didn't bleed outside the region
                    if (!expectPixel(dstMem, x, top - 1, fullColorInt))
                        return false;
                    if (!expectPixel(dstMem, x, bottom + 1, fullColorInt))
                        return false;
                }

            }
            return true;
        }

    private:
        const Desc             *m_testDesc;
        const FormatDesc       *m_formatDesc;
        const lwn::Buffer      *m_readBackBuf;
        const lwn::Texture     *m_msaaResolveTex;
        const lwn::Texture     *m_dstTex;
        const lwn::TextureView *m_dstView;
        const lwn::CopyRegion   m_dstReg;
        lwn::Format             m_dstFormat;
    };
};

class LWNClearTexture
{
private:
    lwn::Format m_formatOverride;
    int         m_numSamples;

public:
    LWNTEST_CppMethods();

    LWNClearTexture(lwn::Format formatOverride, int numSamples) :
        m_formatOverride(formatOverride),
        m_numSamples(numSamples)
    {
        // Format override means instead of looping over the formats in
        // ALL_FORMATS, we run this test only with the given format.
        if (m_formatOverride == lwn::Format::NONE) {
            assert(m_numSamples == 0);
        } else {
            assert(m_numSamples >= 2);
        }
    }
};

lwString LWNClearTexture::getDescription() const
{
    lwStringBuf sb;
    sb << "Test the ClearTexture API in LWN. Tests each color format (excluding compressed, depth and\n"
        "stencil formats) in the following way:\n"
        " * Creates a " << texWidth << "x" << texHeight << " surface and first clears the whole surface.\n"
        " * Additionally clear a few easily identifiable regions of the target texture, with identifiable clear colors.\n"
        " * Copy the resulting image into a buffer for inspection with the CPU\n\n"
        "The test checks that the texture was cleared using the expected values and outputs\n"
        "a series of green, blue or red squares.  Green means pass, blue means unsupported format (also treated pass)\n"
        "or red (means test failure).\n";
    return sb.str();
}

int LWNClearTexture::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 20);
}

static void setupTextureParams(Device *dev, TextureBuilder &normalTex, TextureBuilder &layerTex, const FormatDesc *fmt, int numSamples)
{
    normalTex.SetDevice(dev).SetDefaults();
    layerTex.SetDevice(dev).SetDefaults();

    normalTex.SetFormat(fmt->format);

    normalTex.SetTarget(TextureTarget::TARGET_2D);
    normalTex.SetLevels(2);
    normalTex.SetSize2D(texWidth, texHeight);

    if (numSamples >= 2) {
        normalTex.SetSamples(numSamples);
        normalTex.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE);
        normalTex.SetLevels(1);
    }

    layerTex.SetFormat(fmt->format);
    layerTex.SetTarget(TextureTarget::TARGET_2D_ARRAY);
    layerTex.SetSize3D(texWidth, texHeight, 2);
    layerTex.SetLevels(1);
}

// The current implementation of ClearTexture supports textures that can be
// bound as a color render target. It may be extended later to support more
// formats.
static bool isSupportedFormat(const FormatDesc &fmt)
{
    // skip depth and stencil formats
    if (fmt.flags & (FLAG_DEPTH | FLAG_STENCIL)) {
        return false;
    }
    // skip all formats that are not bindable as render targets
    return (fmt.flags & FLAG_RENDER) != 0;
}

void LWNClearTexture::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const int cellWidth = 6*3;
    int cellHeight = (NUM_FORMATS+5)/6;
    if (cellHeight < 10)
        cellHeight = 10;
    int cellNum = 0;
    cellTestInit(cellWidth, cellHeight);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.submit();

    TextureBuilder msaaResolveTB;
    msaaResolveTB.SetDevice(device).SetDefaults();
    msaaResolveTB.SetFormat(m_formatOverride);
    msaaResolveTB.SetTarget(TextureTarget::TARGET_2D);
    msaaResolveTB.SetSize2D(texWidth, texHeight);

    // Memory pools: allocate enough space for the largest formats
    size_t texStorageSize = 0;
    size_t bufStorageSize = 0;
    for (int fmtIndex = 0; fmtIndex < NUM_FORMATS; fmtIndex++) {
        const Desc &fmt = ALL_FORMATS[fmtIndex];
        const FormatDesc *formatDesc = FormatDesc::findByFormat(fmt.format);
        if (!isSupportedFormat(*formatDesc))
            continue;

        TextureBuilder texNormal;
        TextureBuilder texLayers;
        setupTextureParams(device, texNormal, texLayers, formatDesc, m_numSamples);

        size_t tsize = texNormal.GetPaddedStorageSize();
        bufStorageSize = LW_MAX(bufStorageSize, tsize);

        size_t tsize2 = texLayers.GetPaddedStorageSize();
        tsize += tsize2;

        if (m_numSamples >= 2) {
            // Allocate a temporary texture for MSAA resolve
            size_t tempStorageSize= msaaResolveTB.GetPaddedStorageSize();
            tsize += tempStorageSize;
        }

        bufStorageSize = LW_MAX(bufStorageSize, tsize2);
        texStorageSize = LW_MAX(texStorageSize, tsize);
    }

    MemoryPoolAllocator texAllocator(device, NULL, texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    Texture *msaaResolveTex = nullptr;
    if (m_numSamples) {
        assert(m_numSamples >= 2);
        // Used as the destination of a downsample, must be a normal, non-multisample
        // texture
        msaaResolveTex = texAllocator.allocTexture(&msaaResolveTB);

        MultisampleState msaa;
        msaa.SetDefaults();
        msaa.SetSamples(m_numSamples);
        msaa.SetMultisampleEnable(LWN_TRUE);
        queueCB.BindMultisampleState(&msaa);
    }

    // allocate enough memory for our largest formats; 32-bit 4-component textures
    MemoryPoolAllocator bufAllocator(device, NULL, bufStorageSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *dstBuf = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, bufStorageSize);
    assert(dstBuf);

    const int numFormats = m_formatOverride == lwn::Format::NONE ? NUM_FORMATS : 1;
    for (int fmtIndex = 0; fmtIndex < numFormats; fmtIndex++)
    {
        const Desc *desc = nullptr;
        if (m_formatOverride == lwn::Format::NONE) {
            desc = &ALL_FORMATS[fmtIndex];
        }
        else {
            for (int i = 0; i < NUM_FORMATS; i++) {
                if (ALL_FORMATS[i].format == m_formatOverride) {
                    desc = &ALL_FORMATS[i];
                    break;
                }
            }
            assert(desc);
        }
        const FormatDesc *formatDesc = FormatDesc::findByFormat(desc->format);

        if (!isSupportedFormat(*formatDesc)) {
            for (int i = 0; i < 3; i++) {
                SetCellViewportScissorPadded(queueCB, cellNum % cellWidth, cellNum / cellWidth, 1);
                queueCB.ClearColor(0, 0.0, 0.0, 1.0);
                cellNum++;
            }
            continue;
        }

        TextureBuilder tbNormal;
        TextureBuilder tbLayers;
        setupTextureParams(device, tbNormal, tbLayers, formatDesc, m_numSamples);

        Texture *dstTexNormal = texAllocator.allocTexture(&tbNormal);
        Texture *dstTexLayers = texAllocator.allocTexture(&tbLayers);

        // The test code for multisample doesn't support texture views or multiple layers,
        // so run only the first subtest in that case.
        const int numSubTests = m_numSamples >= 2 ? 1 : 3;
        for (int subtest = 0; subtest < numSubTests; subtest++)
        {
            int layer = (subtest == 1) ? 1 : 0;
            int level = (subtest == 2) ? 1 : 0;
            int levelWidth = texWidth >> level;
            int levelHeight = texHeight >> level;

            lwn::CopyRegion clearDstReg;
            clearDstReg.xoffset = 0;
            clearDstReg.yoffset = 0;
            clearDstReg.zoffset = layer;
            clearDstReg.width = levelWidth;
            clearDstReg.height = levelHeight;
            clearDstReg.depth = 1;

            TextureView levelView;
            levelView.SetDefaults();
            TextureView *viewPtr = nullptr;
            if (level != 0) {
                viewPtr = &levelView;
                viewPtr->SetLevels(level, 1);
            }

            lwn::Texture *dstTex = layer != 0 ? dstTexLayers : dstTexNormal;

            TestClearPattern clearTest(desc, dstBuf, msaaResolveTex, dstTex, viewPtr, clearDstReg);
            bool failed = !clearTest.test(queue, queueCB);

            SetCellViewportScissorPadded(queueCB, cellNum % cellWidth, cellNum / cellWidth, 1);
            queueCB.ClearColor(0, failed ? 1.0 : 0.0, failed ? 0.0 : 1.0, 0.0);

#if DEBUG_VIZ
            // display dst texture contents
            CopyRegion t2bRegion = { (fmtIndex&3)*texWidth, (fmtIndex>>2)*(texHeight+2)+50, 0, levelWidth, levelHeight, 1 };
            queueCB.CopyTextureToTexture(dstTex, viewPtr, &clearDstReg, g_lwnWindowFramebuffer.getAcquiredTexture(), nullptr, &t2bRegion, CopyFlags::NONE);
            queueCB.submit();
            queue->Finish();
#endif
            cellNum++;
        }

        texAllocator.freeTexture(dstTexNormal);
        texAllocator.freeTexture(dstTexLayers);
    }

    if (msaaResolveTex) {
        texAllocator.freeTexture(msaaResolveTex);
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();

    bufAllocator.freeBuffer(dstBuf);
}

OGTEST_CppTest(LWNClearTexture, lwn_clear_texture, (lwn::Format::NONE, 0));
OGTEST_CppTest(LWNClearTexture, lwn_clear_texture_rgba8_msaa2, (lwn::Format::RGBA8, 2));
OGTEST_CppTest(LWNClearTexture, lwn_clear_texture_rgba8_msaa4, (lwn::Format::RGBA8, 4));
OGTEST_CppTest(LWNClearTexture, lwn_clear_texture_rgba8_msaa8, (lwn::Format::RGBA8, 8));
