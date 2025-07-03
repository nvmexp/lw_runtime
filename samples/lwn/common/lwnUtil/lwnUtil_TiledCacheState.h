#ifndef __lwnUtil_TiledCacheState_h__
#define __lwnUtil_TiledCacheState_h__

/*
** Copyright 2016, LWPU Corporation.
** All Rights Reserved.
**
** THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
** LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
** IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*/

#include "lwnUtil_Interface.h"

namespace lwn {
namespace util {

// The TiledCacheState class helps optimize state changes for deciding
// when to flush tiled cache binner or set tile state.
class TiledCacheState {
    void init(LWNdevice* device);
public:
    enum TiledCacheStrategyBits
    {
        SKIP_STENCIL_COMPONENT_BIT  = 1,
        SKIP_DEPTH_COMPONENT_BIT    = 2,
        INCLUDE_MULTIPLE_LAYERS_BIT = 4
    };

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_C
    explicit TiledCacheState(LWNdevice* device) { init(device); }
#else
    explicit TiledCacheState(lwn::Device* device)
    {
        init(reinterpret_cast<LWNdevice *>(device));
    }
#endif

    inline void SetStrategy(TiledCacheStrategyBits strategyBits) {
        mStrategyBits = strategyBits;
    }

    inline void SetL2UsageRatio(float l2UsageRatio) {
        mL2UsageRatio = l2UsageRatio;
    }

    void UpdateTileState(
        LWNcommandBuffer *cmdBuf,
        int numColors,
        int sampleCount,
        const LWNtexture* const* colors,
        const LWNtexture* depthStencil);

    inline size_t GetL2CacheSize() {
        return mL2Size;
    }

    inline void GetTileSize(int* tileWidth, int* tileHeight) {
        *tileWidth = mTileWidth;
        *tileHeight = mTileHeight;
    }

    inline LWNtiledCacheAction GetLastAction() {
        return mLastAction;
    }

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    //
    // Methods to provide a native C++ interface to the core TiledCacheState
    // class, using reinterpret_cast to colwert between C and C++ object
    // types.
    //
    void UpdateTileState(
        lwn::CommandBuffer *cmdBuf,
        int numColors,
        int sampleCount,
        const lwn::Texture * const* colors,
        const lwn::Texture * depthStencil)
    {
        LWNcommandBuffer *ccb = reinterpret_cast<LWNcommandBuffer *>(cmdBuf);
        const LWNtexture * const *ccolors = reinterpret_cast<const LWNtexture * const *>(colors);
        const LWNtexture *cds = reinterpret_cast<const LWNtexture *>(depthStencil);
        UpdateTileState(ccb, numColors, sampleCount, ccolors, cds);
    }
#endif

private:

    void UpdateTileSize(int numColors, int sampleCount,
                        const LWNtexture* const* colors, const LWNtexture* depthStencil);

    int GetBytesPerPixelColor(LWNformat format);
    void GetBytesPerPixelDepthStencil(LWNformat format, int *depthBytes, int *stencilBytes);

    // Number of render targets cached
    static const int COLOR_COUNT = 4;

    LWNdevice*             mDevice;
    TiledCacheStrategyBits mStrategyBits;

    const LWNtexture*      mColorTextures[COLOR_COUNT];
    int                    mColorCount;
    const LWNtexture*      mDepthStencil;

    size_t                 mL2Size;
    float                  mL2UsageRatio;

    int                    mTileWidth;
    int                    mTileHeight;
    LWNtiledCacheAction    mLastAction;
};

inline TiledCacheState::TiledCacheStrategyBits operator|(TiledCacheState::TiledCacheStrategyBits a, TiledCacheState::TiledCacheStrategyBits b) {
    return TiledCacheState::TiledCacheStrategyBits(int(a) | int(b));
}

} // namespace util
} // namespace lwn

#endif
