#pragma once

#include <wincodec.h>

#include "darkroom/Errors.h"

#include <string>
#include <vector>

namespace darkroom
{
    Error saveJxr(const std::vector<BYTE>& hdrData, const unsigned int width, const unsigned int height,
        const WICPixelFormatGUID srcFormatGUID, const std::wstring& shotName, bool forceFloatDst, bool doLossless);
}
