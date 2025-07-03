#ifndef __lwnUtil_TiledCacheStateImpl_h__
#define __lwnUtil_TiledCacheStateImpl_h__

/*
** Copyright 2016, LWPU Corporation.
** All Rights Reserved.
**
** THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
** LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
** IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*/

namespace lwn {
namespace util {

template<typename T>
static T imin(T a, T b)
{
    return a < b ? a : b;
}

template<typename T>
static T imax(T a, T b)
{
    return a > b ? a : b;
}

void TiledCacheState::init(LWNdevice *device)
{
    mDevice = device;
    mL2UsageRatio = 0.3f;

    int l2Size = 0;
    lwnDeviceGetInteger(mDevice, LWN_DEVICE_INFO_L2_SIZE, &l2Size);
    mL2Size = l2Size;

    memset(mColorTextures, 0, sizeof(mColorTextures));

    mDepthStencil = NULL;
    mColorCount = 0;
    mLastAction = LWN_TILED_CACHE_ACTION_DISABLE;
    mStrategyBits = TiledCacheStrategyBits(0);
}

void TiledCacheState::UpdateTileState(
    LWNcommandBuffer* cmdBuf,
    int numColors,
    int sampleCount,
    const LWNtexture* const* colors,
    const LWNtexture* depthStencil)
{
    // Limit the number of MRTs tested.
    numColors = imin(numColors, COLOR_COUNT);
    const LWNtexture* colorTextures[COLOR_COUNT];

    // Flatten color array (ignore all NULL color targets)
    bool needBinnerFlush = false;
    int j = 0;
    for (int i = 0; i < numColors; i++) {
        if (colors[i]) {
            colorTextures[j++] = colors[i];
        }
    }

    // We only care if any non-NULL color targets change.
    numColors = j;

    bool depthOnly = false;
    if (depthStencil) {
        LWNformat depthStencilFormat = lwnTextureGetFormat(depthStencil);

        if (depthStencil != mDepthStencil) {
            // If the depth/stencil target changed, go ahead and flush the binner.
            needBinnerFlush = true;
            mDepthStencil = depthStencil;
        }

        bool formatHasStencil =
            (depthStencilFormat == LWN_FORMAT_DEPTH24_STENCIL8) ||
            (depthStencilFormat == LWN_FORMAT_DEPTH32F_STENCIL8) ||
            (depthStencilFormat == LWN_FORMAT_STENCIL8);

        bool mayUseStencil = (mStrategyBits & SKIP_STENCIL_COMPONENT_BIT) == 0 && formatHasStencil;

        depthOnly = (numColors == 0) && !mayUseStencil;

        if (depthOnly) {
            // For depth only rendering, disable tiled caching for better performance.
            lwnCommandBufferSetTiledCacheAction(cmdBuf, mLastAction = LWN_TILED_CACHE_ACTION_DISABLE);
        }
    }

    if (!depthOnly) {
        // Enable tiled caching. Note this is call is ignored if tiled caching is already enabled.
        lwnCommandBufferSetTiledCacheAction(cmdBuf, mLastAction = LWN_TILED_CACHE_ACTION_ENABLE);

        if (!needBinnerFlush) {
            // Assume color targets have been added since last dirty state.
            needBinnerFlush = true;

            // No: check if ordered subset of color targets are bound from previous set.
            // A "substring in string" algorithm is faster than a "subset in set" algorithm, if
            // we assume that an app typically does not reorder render targets.

            // The goal of this algorithm is to identify situations where the application binds a
            // subset of render targets. For example if you bind render target A, B, C in one pass
            // and A, B in another pass, then this render target change will be filtered and the
            // binner will not be flushed. Since A, B is still fresh in L2, it is optimal to continue
            // to bin the subsequent rendering.

            // Note that if we skip flushing the binner, we also skip callwlating the tile size for the new
            // reduced set since changing the tile size also causes a binner flush.
            if (numColors <= mColorCount) {
                for (int i = 0; i <= (mColorCount - numColors); ++i) {
                    for (j = 0; j < numColors; ++j) {
                        if (mColorTextures[i + j] != colorTextures[j]) {
                            break;
                        }
                    }
                    if (j == numColors) {
                        needBinnerFlush = false;
                        break;
                    }
                }
            }
        }

        if (needBinnerFlush) {
            // Cache the current set of color targets.
            memcpy(mColorTextures, colorTextures, sizeof(LWNtexture*) * numColors);

            // Zero out unused entries. Not strictly needed.
            if (mColorCount > numColors) {
                int unusedNumColorOffsets = (mColorCount - numColors);
                memset(&mColorTextures[numColors], 0, sizeof(LWNtexture*) * unusedNumColorOffsets);
            }

            mColorCount = numColors;

            // Set optimal tile size.
            UpdateTileSize(numColors, sampleCount, colors, depthStencil);

            lwnCommandBufferSetTiledCacheTileSize(cmdBuf, mTileWidth, mTileHeight);
            lwnCommandBufferSetTiledCacheAction(cmdBuf, mLastAction = LWN_TILED_CACHE_ACTION_FLUSH);
        }
    }

}

void TiledCacheState::UpdateTileSize(
    int numColors,
    int sampleCount,
    const LWNtexture* const* colors,
    const LWNtexture* depthStencil)
{
    int bytesPerPixel = 0;

    if (depthStencil) {
        int depthBytes;
        int stencilBytes;
        LWNformat depthStencilFormat = lwnTextureGetFormat(depthStencil);

        GetBytesPerPixelDepthStencil(depthStencilFormat, &depthBytes, &stencilBytes);

        if ((mStrategyBits & SKIP_DEPTH_COMPONENT_BIT) == 0) {
            bytesPerPixel += depthBytes;
        }

        if ((mStrategyBits & SKIP_STENCIL_COMPONENT_BIT) == 0) {
            bytesPerPixel += stencilBytes;
        }
    }

    if (colors)  {
        for (int i = 0; i < numColors; i++) {
            bytesPerPixel += GetBytesPerPixelColor(lwnTextureGetFormat(colors[i]));
        }
    }

    // Scale non-linearly by AA size (1x, 2x, 3x, etc)
    bytesPerPixel *= ((sampleCount + 2) >> 1);

    if (bytesPerPixel) {
        size_t bytesPerTile = size_t(mL2Size * mL2UsageRatio);
        size_t pixelsPerTile = bytesPerTile / bytesPerPixel;
        int log2TileSize = 0;

        while (pixelsPerTile >>= 1) {
            log2TileSize++;
        }

        int log2TileWidth  = imin(14, imax(4, (log2TileSize >> 1)));
        int log2TileHeight = imin(14, imax(4, log2TileSize - log2TileWidth));

        mTileWidth = 1 << log2TileWidth;
        mTileHeight = 1 << log2TileHeight;
    }
}

int TiledCacheState::GetBytesPerPixelColor(LWNformat format)
{
    switch (format) {
    case LWN_FORMAT_R8:
    case LWN_FORMAT_R8UI:
    case LWN_FORMAT_R8SN:
    case LWN_FORMAT_R8I:
        return 1;

    case LWN_FORMAT_R16F:
    case LWN_FORMAT_R16:
    case LWN_FORMAT_R16SN:
    case LWN_FORMAT_R16UI:
    case LWN_FORMAT_R16I:
    case LWN_FORMAT_RG8SN:
    case LWN_FORMAT_RG8UI:
    case LWN_FORMAT_RG8I:
    case LWN_FORMAT_DEPTH16:
        return 2;

    case LWN_FORMAT_R32F:
    case LWN_FORMAT_R32UI:
    case LWN_FORMAT_R32I:
    case LWN_FORMAT_RG16F:
    case LWN_FORMAT_RG16:
    case LWN_FORMAT_RG16SN:
    case LWN_FORMAT_RG16UI:
    case LWN_FORMAT_RG16I:
    case LWN_FORMAT_RGBA8:
    case LWN_FORMAT_RGBA8SN:
    case LWN_FORMAT_RGBA8UI:
    case LWN_FORMAT_RGBA8I:
    case LWN_FORMAT_RGBX8:
    case LWN_FORMAT_RGBX8SN:
    case LWN_FORMAT_RGBX8UI:
    case LWN_FORMAT_RGBX8I:
    case LWN_FORMAT_RGB10A2:
    case LWN_FORMAT_RGB10A2UI:
    case LWN_FORMAT_R11G11B10F:
    case LWN_FORMAT_RGBX8_SRGB:
    case LWN_FORMAT_RGBA8_SRGB:
    case LWN_FORMAT_RGB5A1:
    case LWN_FORMAT_RGB565:
        return 4;

    case LWN_FORMAT_RG32F:
    case LWN_FORMAT_RG32UI:
    case LWN_FORMAT_RG32I:
    case LWN_FORMAT_RGBA16F:
    case LWN_FORMAT_RGBA16:
    case LWN_FORMAT_RGBA16SN:
    case LWN_FORMAT_RGBA16UI:
    case LWN_FORMAT_RGBA16I:
    case LWN_FORMAT_RGBX16F:
    case LWN_FORMAT_RGBX16:
    case LWN_FORMAT_RGBX16SN:
    case LWN_FORMAT_RGBX16UI:
    case LWN_FORMAT_RGBX16I:
        return 8;

    case LWN_FORMAT_RGBA32F:
    case LWN_FORMAT_RGBA32UI:
    case LWN_FORMAT_RGBA32I:
    case LWN_FORMAT_RGBX32F:
    case LWN_FORMAT_RGBX32UI:
    case LWN_FORMAT_RGBX32I:
        return 16;

    default:
        // unsupported color target format
        assert(0);
        return 0;
    }
}

void TiledCacheState::GetBytesPerPixelDepthStencil(LWNformat format, int *depthBytes, int *stencilBytes)
{
    // depth and stencil are partitioned in hardware, so track them seperately
    // for purposes of tile size
    switch (format) {

    case LWN_FORMAT_STENCIL8:
        *depthBytes = 0;
        *stencilBytes = 1;
        break;

    case LWN_FORMAT_DEPTH24:
        *depthBytes = 3;
        *stencilBytes = 0;
        break;

    case LWN_FORMAT_DEPTH32F:
        *depthBytes = 4;
        *stencilBytes = 0;
        break;

    case LWN_FORMAT_DEPTH24_STENCIL8:
        *depthBytes = 3;
        *stencilBytes = 1;
        break;

    case LWN_FORMAT_DEPTH32F_STENCIL8:
        *depthBytes = 4;
        *stencilBytes = 1;
        break;

    default:
        // unsupported depth target format
        assert(0);
        *depthBytes = 0;
        *stencilBytes = 0;
        break;
    }
}

} // namespace util
} // namespace lwn

#endif
