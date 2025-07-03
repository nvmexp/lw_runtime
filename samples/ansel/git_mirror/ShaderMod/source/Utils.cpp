#pragma once
#include "Utils.h"
#include <iomanip>
#include <sstream>
#include <chrono>
#include <ctime>
#include <Shlwapi.h>
#include "anselutils/Utils.h"
#include "ir/FileHelpers.h"
#include "darkroom/StringColwersion.h"
#include "Log.h"


namespace lwanselutils
{
    DXGI_FORMAT dxgiTypelessFormats[] =
    {
        DXGI_FORMAT_UNKNOWN,                     // = 0,
        DXGI_FORMAT_R32G32B32A32_TYPELESS,       // = 1,
        DXGI_FORMAT_R32G32B32A32_TYPELESS,       // = 2,
        DXGI_FORMAT_R32G32B32A32_TYPELESS,       // = 3,
        DXGI_FORMAT_R32G32B32A32_TYPELESS,       // = 4,
        DXGI_FORMAT_R32G32B32_TYPELESS,          // = 5,
        DXGI_FORMAT_R32G32B32_TYPELESS,          // = 6,
        DXGI_FORMAT_R32G32B32_TYPELESS,          // = 7,
        DXGI_FORMAT_R32G32B32_TYPELESS,          // = 8,
        DXGI_FORMAT_R16G16B16A16_TYPELESS,       // = 9,
        DXGI_FORMAT_R16G16B16A16_TYPELESS,       // = 10,
        DXGI_FORMAT_R16G16B16A16_TYPELESS,       // = 11,
        DXGI_FORMAT_R16G16B16A16_TYPELESS,       // = 12,
        DXGI_FORMAT_R16G16B16A16_TYPELESS,       // = 13,
        DXGI_FORMAT_R16G16B16A16_TYPELESS,       // = 14,
        DXGI_FORMAT_R32G32_TYPELESS,             // = 15,
        DXGI_FORMAT_R32G32_TYPELESS,             // = 16,
        DXGI_FORMAT_R32G32_TYPELESS,             // = 17,
        DXGI_FORMAT_R32G32_TYPELESS,             // = 18,
        DXGI_FORMAT_R32G8X24_TYPELESS,           // = 19,
        DXGI_FORMAT_R32G8X24_TYPELESS,           // = 20,
        DXGI_FORMAT_R32G8X24_TYPELESS,           // = 21,
        DXGI_FORMAT_R32G8X24_TYPELESS,           // = 22,
        DXGI_FORMAT_R10G10B10A2_TYPELESS,        // = 23,
        DXGI_FORMAT_R10G10B10A2_TYPELESS,        // = 24,
        DXGI_FORMAT_R10G10B10A2_TYPELESS,        // = 25,
        DXGI_FORMAT_R11G11B10_FLOAT,             // = 26,
        DXGI_FORMAT_R8G8B8A8_TYPELESS,           // = 27,
        DXGI_FORMAT_R8G8B8A8_TYPELESS,           // = 28,
        DXGI_FORMAT_R8G8B8A8_TYPELESS,           // = 29,
        DXGI_FORMAT_R8G8B8A8_TYPELESS,           // = 30,
        DXGI_FORMAT_R8G8B8A8_TYPELESS,           // = 31,
        DXGI_FORMAT_R8G8B8A8_TYPELESS,           // = 32,
        DXGI_FORMAT_R16G16_TYPELESS,             // = 33,
        DXGI_FORMAT_R16G16_TYPELESS,             // = 34,
        DXGI_FORMAT_R16G16_TYPELESS,             // = 35,
        DXGI_FORMAT_R16G16_TYPELESS,             // = 36,
        DXGI_FORMAT_R16G16_TYPELESS,             // = 37,
        DXGI_FORMAT_R16G16_TYPELESS,             // = 38,
        DXGI_FORMAT_R32_TYPELESS,                // = 39,
        DXGI_FORMAT_R32_TYPELESS,                // = 40,
        DXGI_FORMAT_R32_TYPELESS,                // = 41,
        DXGI_FORMAT_R32_TYPELESS,                // = 42,
        DXGI_FORMAT_R32_TYPELESS,                // = 43,
        DXGI_FORMAT_R24G8_TYPELESS,              // = 44,
        DXGI_FORMAT_R24G8_TYPELESS,              // = 45,
        DXGI_FORMAT_R24G8_TYPELESS,              // = 46,
        DXGI_FORMAT_R24G8_TYPELESS,              // = 47,
        DXGI_FORMAT_R8G8_TYPELESS,               // = 48,
        DXGI_FORMAT_R8G8_TYPELESS,               // = 49,
        DXGI_FORMAT_R8G8_TYPELESS,               // = 50,
        DXGI_FORMAT_R8G8_TYPELESS,               // = 51,
        DXGI_FORMAT_R8G8_TYPELESS,               // = 52,
        DXGI_FORMAT_R16_TYPELESS,                // = 53,
        DXGI_FORMAT_R16_TYPELESS,                // = 54,
        DXGI_FORMAT_R16_TYPELESS,                // = 55,
        DXGI_FORMAT_R16_TYPELESS,                // = 56,
        DXGI_FORMAT_R16_TYPELESS,                // = 57,
        DXGI_FORMAT_R16_TYPELESS,                // = 58,
        DXGI_FORMAT_R16_TYPELESS,                // = 59,
        DXGI_FORMAT_R8_TYPELESS,                 // = 60,
        DXGI_FORMAT_R8_TYPELESS,                 // = 61,
        DXGI_FORMAT_R8_TYPELESS,                 // = 62,
        DXGI_FORMAT_R8_TYPELESS,                 // = 63,
        DXGI_FORMAT_R8_TYPELESS,                 // = 64,
        DXGI_FORMAT_R8_TYPELESS,                 // = 65,
        DXGI_FORMAT_R1_UNORM,                    // = 66,
        DXGI_FORMAT_R9G9B9E5_SHAREDEXP,          // = 67,
        DXGI_FORMAT_R8G8_B8G8_UNORM,             // = 68,
        DXGI_FORMAT_G8R8_G8B8_UNORM,             // = 69,
        DXGI_FORMAT_BC1_TYPELESS,                // = 70,
        DXGI_FORMAT_BC1_TYPELESS,                // = 71,
        DXGI_FORMAT_BC1_TYPELESS,                // = 72,
        DXGI_FORMAT_BC2_TYPELESS,                // = 73,
        DXGI_FORMAT_BC2_TYPELESS,                // = 74,
        DXGI_FORMAT_BC2_TYPELESS,                // = 75,
        DXGI_FORMAT_BC3_TYPELESS,                // = 76,
        DXGI_FORMAT_BC3_TYPELESS,                // = 77,
        DXGI_FORMAT_BC3_TYPELESS,                // = 78,
        DXGI_FORMAT_BC4_TYPELESS,                // = 79,
        DXGI_FORMAT_BC4_TYPELESS,                // = 80,
        DXGI_FORMAT_BC4_TYPELESS,                // = 81,
        DXGI_FORMAT_BC5_TYPELESS,                // = 82,
        DXGI_FORMAT_BC5_TYPELESS,                // = 83,
        DXGI_FORMAT_BC5_TYPELESS,                // = 84,
        DXGI_FORMAT_B5G6R5_UNORM,                // = 85,
        DXGI_FORMAT_B5G5R5A1_UNORM,              // = 86,
        DXGI_FORMAT_B8G8R8A8_TYPELESS,           // = 87,
        DXGI_FORMAT_B8G8R8X8_TYPELESS,           // = 88,
        DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM,  // = 89,
        DXGI_FORMAT_B8G8R8A8_TYPELESS,           // = 90,
        DXGI_FORMAT_B8G8R8A8_TYPELESS,           // = 91,
        DXGI_FORMAT_B8G8R8A8_TYPELESS,           // = 92,
        DXGI_FORMAT_B8G8R8A8_TYPELESS,           // = 93,
        DXGI_FORMAT_BC6H_TYPELESS,               // = 94,
        DXGI_FORMAT_BC6H_TYPELESS,               // = 95,
        DXGI_FORMAT_BC6H_TYPELESS,               // = 96,
        DXGI_FORMAT_BC7_TYPELESS,                // = 97,
        DXGI_FORMAT_BC7_TYPELESS,                // = 98,
        DXGI_FORMAT_BC7_TYPELESS,                // = 99,
        DXGI_FORMAT_AYUV,                        // = 100,
        DXGI_FORMAT_Y410,                        // = 101,
        DXGI_FORMAT_Y416,                        // = 102,
        DXGI_FORMAT_LW12,                        // = 103,
        DXGI_FORMAT_P010,                        // = 104,
        DXGI_FORMAT_P016,                        // = 105,
        DXGI_FORMAT_420_OPAQUE,                  // = 106,
        DXGI_FORMAT_YUY2,                        // = 107,
        DXGI_FORMAT_Y210,                        // = 108,
        DXGI_FORMAT_Y216,                        // = 109,
        DXGI_FORMAT_LW11,                        // = 110,
        DXGI_FORMAT_AI44,                        // = 111,
        DXGI_FORMAT_IA44,                        // = 112,
        DXGI_FORMAT_P8,                          // = 113,
        DXGI_FORMAT_A8P8,                        // = 114,
        DXGI_FORMAT_B4G4R4A4_UNORM               // = 115,
        // DXGI_FORMAT_P208,                     // = 130,
        // DXGI_FORMAT_V208,                     // = 131,
        // DXGI_FORMAT_V408                      // = 132,
    };

#define StringizeCase(v) case v: return #v;
    std::string GetDxgiFormatName(UINT format)
    {
        switch (format)
        {
            StringizeCase(DXGI_FORMAT_UNKNOWN);
            StringizeCase(DXGI_FORMAT_R32G32B32A32_TYPELESS);
            StringizeCase(DXGI_FORMAT_R32G32B32A32_FLOAT);
            StringizeCase(DXGI_FORMAT_R32G32B32A32_UINT);
            StringizeCase(DXGI_FORMAT_R32G32B32A32_SINT);
            StringizeCase(DXGI_FORMAT_R32G32B32_TYPELESS);
            StringizeCase(DXGI_FORMAT_R32G32B32_FLOAT);
            StringizeCase(DXGI_FORMAT_R32G32B32_UINT);
            StringizeCase(DXGI_FORMAT_R32G32B32_SINT);
            StringizeCase(DXGI_FORMAT_R16G16B16A16_TYPELESS);
            StringizeCase(DXGI_FORMAT_R16G16B16A16_FLOAT);
            StringizeCase(DXGI_FORMAT_R16G16B16A16_UNORM);
            StringizeCase(DXGI_FORMAT_R16G16B16A16_UINT);
            StringizeCase(DXGI_FORMAT_R16G16B16A16_SNORM);
            StringizeCase(DXGI_FORMAT_R16G16B16A16_SINT);
            StringizeCase(DXGI_FORMAT_R32G32_TYPELESS);
            StringizeCase(DXGI_FORMAT_R32G32_FLOAT);
            StringizeCase(DXGI_FORMAT_R32G32_UINT);
            StringizeCase(DXGI_FORMAT_R32G32_SINT);
            StringizeCase(DXGI_FORMAT_R32G8X24_TYPELESS);
            StringizeCase(DXGI_FORMAT_D32_FLOAT_S8X24_UINT);
            StringizeCase(DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS);
            StringizeCase(DXGI_FORMAT_X32_TYPELESS_G8X24_UINT);
            StringizeCase(DXGI_FORMAT_R10G10B10A2_TYPELESS);
            StringizeCase(DXGI_FORMAT_R10G10B10A2_UNORM);
            StringizeCase(DXGI_FORMAT_R10G10B10A2_UINT);
            StringizeCase(DXGI_FORMAT_R11G11B10_FLOAT);
            StringizeCase(DXGI_FORMAT_R8G8B8A8_TYPELESS);
            StringizeCase(DXGI_FORMAT_R8G8B8A8_UNORM);
            StringizeCase(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
            StringizeCase(DXGI_FORMAT_R8G8B8A8_UINT);
            StringizeCase(DXGI_FORMAT_R8G8B8A8_SNORM);
            StringizeCase(DXGI_FORMAT_R8G8B8A8_SINT);
            StringizeCase(DXGI_FORMAT_R16G16_TYPELESS);
            StringizeCase(DXGI_FORMAT_R16G16_FLOAT);
            StringizeCase(DXGI_FORMAT_R16G16_UNORM);
            StringizeCase(DXGI_FORMAT_R16G16_UINT);
            StringizeCase(DXGI_FORMAT_R16G16_SNORM);
            StringizeCase(DXGI_FORMAT_R16G16_SINT);
            StringizeCase(DXGI_FORMAT_R32_TYPELESS);
            StringizeCase(DXGI_FORMAT_D32_FLOAT);
            StringizeCase(DXGI_FORMAT_R32_FLOAT);
            StringizeCase(DXGI_FORMAT_R32_UINT);
            StringizeCase(DXGI_FORMAT_R32_SINT);
            StringizeCase(DXGI_FORMAT_R24G8_TYPELESS);
            StringizeCase(DXGI_FORMAT_D24_UNORM_S8_UINT);
            StringizeCase(DXGI_FORMAT_R24_UNORM_X8_TYPELESS);
            StringizeCase(DXGI_FORMAT_X24_TYPELESS_G8_UINT);
            StringizeCase(DXGI_FORMAT_R8G8_TYPELESS);
            StringizeCase(DXGI_FORMAT_R8G8_UNORM);
            StringizeCase(DXGI_FORMAT_R8G8_UINT);
            StringizeCase(DXGI_FORMAT_R8G8_SNORM);
            StringizeCase(DXGI_FORMAT_R8G8_SINT);
            StringizeCase(DXGI_FORMAT_R16_TYPELESS);
            StringizeCase(DXGI_FORMAT_R16_FLOAT);
            StringizeCase(DXGI_FORMAT_D16_UNORM);
            StringizeCase(DXGI_FORMAT_R16_UNORM);
            StringizeCase(DXGI_FORMAT_R16_UINT);
            StringizeCase(DXGI_FORMAT_R16_SNORM);
            StringizeCase(DXGI_FORMAT_R16_SINT);
            StringizeCase(DXGI_FORMAT_R8_TYPELESS);
            StringizeCase(DXGI_FORMAT_R8_UNORM);
            StringizeCase(DXGI_FORMAT_R8_UINT);
            StringizeCase(DXGI_FORMAT_R8_SNORM);
            StringizeCase(DXGI_FORMAT_R8_SINT);
            StringizeCase(DXGI_FORMAT_A8_UNORM);
            StringizeCase(DXGI_FORMAT_R1_UNORM);
            StringizeCase(DXGI_FORMAT_R9G9B9E5_SHAREDEXP);
            StringizeCase(DXGI_FORMAT_R8G8_B8G8_UNORM);
            StringizeCase(DXGI_FORMAT_G8R8_G8B8_UNORM);
            StringizeCase(DXGI_FORMAT_BC1_TYPELESS);
            StringizeCase(DXGI_FORMAT_BC1_UNORM);
            StringizeCase(DXGI_FORMAT_BC1_UNORM_SRGB);
            StringizeCase(DXGI_FORMAT_BC2_TYPELESS);
            StringizeCase(DXGI_FORMAT_BC2_UNORM);
            StringizeCase(DXGI_FORMAT_BC2_UNORM_SRGB);
            StringizeCase(DXGI_FORMAT_BC3_TYPELESS);
            StringizeCase(DXGI_FORMAT_BC3_UNORM);
            StringizeCase(DXGI_FORMAT_BC3_UNORM_SRGB);
            StringizeCase(DXGI_FORMAT_BC4_TYPELESS);
            StringizeCase(DXGI_FORMAT_BC4_UNORM);
            StringizeCase(DXGI_FORMAT_BC4_SNORM);
            StringizeCase(DXGI_FORMAT_BC5_TYPELESS);
            StringizeCase(DXGI_FORMAT_BC5_UNORM);
            StringizeCase(DXGI_FORMAT_BC5_SNORM);
            StringizeCase(DXGI_FORMAT_B5G6R5_UNORM);
            StringizeCase(DXGI_FORMAT_B5G5R5A1_UNORM);
            StringizeCase(DXGI_FORMAT_B8G8R8A8_UNORM);
            StringizeCase(DXGI_FORMAT_B8G8R8X8_UNORM);
            StringizeCase(DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM);
            StringizeCase(DXGI_FORMAT_B8G8R8A8_TYPELESS);
            StringizeCase(DXGI_FORMAT_B8G8R8A8_UNORM_SRGB);
            StringizeCase(DXGI_FORMAT_B8G8R8X8_TYPELESS);
            StringizeCase(DXGI_FORMAT_B8G8R8X8_UNORM_SRGB);
            StringizeCase(DXGI_FORMAT_BC6H_TYPELESS);
            StringizeCase(DXGI_FORMAT_BC6H_UF16);
            StringizeCase(DXGI_FORMAT_BC6H_SF16);
            StringizeCase(DXGI_FORMAT_BC7_TYPELESS);
            StringizeCase(DXGI_FORMAT_BC7_UNORM);
            StringizeCase(DXGI_FORMAT_BC7_UNORM_SRGB);
            StringizeCase(DXGI_FORMAT_AYUV);
            StringizeCase(DXGI_FORMAT_Y410);
            StringizeCase(DXGI_FORMAT_Y416);
            StringizeCase(DXGI_FORMAT_LW12);
            StringizeCase(DXGI_FORMAT_P010);
            StringizeCase(DXGI_FORMAT_P016);
            StringizeCase(DXGI_FORMAT_420_OPAQUE);
            StringizeCase(DXGI_FORMAT_YUY2);
            StringizeCase(DXGI_FORMAT_Y210);
            StringizeCase(DXGI_FORMAT_Y216);
            StringizeCase(DXGI_FORMAT_LW11);
            StringizeCase(DXGI_FORMAT_AI44);
            StringizeCase(DXGI_FORMAT_IA44);
            StringizeCase(DXGI_FORMAT_P8);
            StringizeCase(DXGI_FORMAT_A8P8);
            StringizeCase(DXGI_FORMAT_B4G4R4A4_UNORM);
            StringizeCase(DXGI_FORMAT_FORCE_UINT);
        default: return "Invalid DXGI_FORMAT";
        }
    }

    DXGI_FORMAT colwertFromTypelessIfNeeded(DXGI_FORMAT inFormat, bool checkColw)
    {
        DXGI_FORMAT colwTypedFmt = inFormat;

        switch (inFormat)
        {
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:     colwTypedFmt = DXGI_FORMAT_R32G32B32A32_FLOAT; break;
        case DXGI_FORMAT_R32G32B32_TYPELESS:        colwTypedFmt = DXGI_FORMAT_R32G32B32_FLOAT; break;;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:     colwTypedFmt = DXGI_FORMAT_R16G16B16A16_FLOAT; break;
        case DXGI_FORMAT_R32G32_TYPELESS:           colwTypedFmt = DXGI_FORMAT_R32G32_FLOAT; break;
        case DXGI_FORMAT_R32G8X24_TYPELESS:         colwTypedFmt = DXGI_FORMAT_D32_FLOAT_S8X24_UINT; break;
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:  colwTypedFmt = DXGI_FORMAT_D32_FLOAT_S8X24_UINT; break;
        case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:   colwTypedFmt = DXGI_FORMAT_D32_FLOAT_S8X24_UINT; break;
        case DXGI_FORMAT_R24G8_TYPELESS:            colwTypedFmt = DXGI_FORMAT_D24_UNORM_S8_UINT; break;
        case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:     colwTypedFmt = DXGI_FORMAT_D24_UNORM_S8_UINT; break;
        case DXGI_FORMAT_X24_TYPELESS_G8_UINT:      colwTypedFmt = DXGI_FORMAT_D24_UNORM_S8_UINT; break;
        case DXGI_FORMAT_R16G16_TYPELESS:           colwTypedFmt = DXGI_FORMAT_R16G16_FLOAT; break;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:      colwTypedFmt = DXGI_FORMAT_R10G10B10A2_UNORM; break;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:         colwTypedFmt = DXGI_FORMAT_R8G8B8A8_UNORM; break;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:         colwTypedFmt = DXGI_FORMAT_B8G8R8A8_UNORM; break;
        case DXGI_FORMAT_B8G8R8X8_TYPELESS:         colwTypedFmt = DXGI_FORMAT_B8G8R8A8_UNORM; break;
        case DXGI_FORMAT_R32_TYPELESS:              colwTypedFmt = DXGI_FORMAT_R32_FLOAT; break;
        case DXGI_FORMAT_R16_TYPELESS:              colwTypedFmt = DXGI_FORMAT_R16_FLOAT; break;
        case DXGI_FORMAT_R8G8_TYPELESS:             colwTypedFmt = DXGI_FORMAT_R8G8_UNORM; break;
        case DXGI_FORMAT_R8_TYPELESS:               colwTypedFmt = DXGI_FORMAT_R8_UNORM; break;
        case DXGI_FORMAT_BC1_TYPELESS:              colwTypedFmt = DXGI_FORMAT_BC1_UNORM; break;
        case DXGI_FORMAT_BC2_TYPELESS:              colwTypedFmt = DXGI_FORMAT_BC2_UNORM; break;
        case DXGI_FORMAT_BC3_TYPELESS:              colwTypedFmt = DXGI_FORMAT_BC3_UNORM; break;
        case DXGI_FORMAT_BC4_TYPELESS:              colwTypedFmt = DXGI_FORMAT_BC4_UNORM; break;
        case DXGI_FORMAT_BC5_TYPELESS:              colwTypedFmt = DXGI_FORMAT_BC5_UNORM; break;
        case DXGI_FORMAT_BC7_TYPELESS:              colwTypedFmt = DXGI_FORMAT_BC7_UNORM; break;
        case DXGI_FORMAT_UNKNOWN:                   // = 0
        case DXGI_FORMAT_BC6H_TYPELESS:             // = 94
            LOG_WARN("No available format for typeless-to-typed format colwersion of %s!", DxgiFormat_cstr(inFormat));
            colwTypedFmt = DXGI_FORMAT_UNKNOWN;
            break;
        default:
            break;
        };

        if (checkColw && colwTypedFmt != inFormat)
        {
            LOG_VERBOSE("%s: %s colwerted to %s!", __func__, DxgiFormat_cstr(inFormat), DxgiFormat_cstr(colwTypedFmt));
        }

        return colwTypedFmt;
    }

    DXGI_FORMAT colwertToTypeless(DXGI_FORMAT inFormat, bool checkColw)
    {
        if (checkColw && dxgiTypelessFormats[inFormat] != inFormat)
        {
            LOG_VERBOSE("%s: %s colwerted to %s!", __func__, DxgiFormat_cstr(inFormat), DxgiFormat_cstr(dxgiTypelessFormats[inFormat]));

            DXGI_FORMAT revColwFmt = colwertFromTypelessIfNeeded(dxgiTypelessFormats[inFormat], false);
            if (revColwFmt != inFormat)
            {
                LOG_WARN("Reverse colwersion won't provide the same original typed format - %s: %s colwerts back to %s instead!", DxgiFormat_cstr(inFormat), DxgiFormat_cstr(dxgiTypelessFormats[inFormat]), DxgiFormat_cstr(revColwFmt));
            }
        }
        return dxgiTypelessFormats[inFormat];
    }

    DXGI_FORMAT getSRVFormatDepth(DXGI_FORMAT inFormat, bool checkColw)
    {
        DXGI_FORMAT SRVDepthFmt = inFormat;

        switch (inFormat)
        {
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:  SRVDepthFmt = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS; break;
        case DXGI_FORMAT_D32_FLOAT:             SRVDepthFmt = DXGI_FORMAT_R32_FLOAT; break;
        case DXGI_FORMAT_D24_UNORM_S8_UINT:     SRVDepthFmt = DXGI_FORMAT_R24_UNORM_X8_TYPELESS; break;
        case DXGI_FORMAT_D16_UNORM:             SRVDepthFmt = DXGI_FORMAT_R16_UNORM; break;
        }

        if (checkColw && SRVDepthFmt != inFormat)
        {
            LOG_VERBOSE("%s: %s colwerted to %s!", __func__, DxgiFormat_cstr(inFormat), DxgiFormat_cstr(SRVDepthFmt));
        }

        return SRVDepthFmt;
    }

    DXGI_FORMAT getSRVFormatStencil(DXGI_FORMAT inFormat, bool checkColw)
    {
        DXGI_FORMAT SRVStencilFmt = inFormat;

        switch (inFormat)
        {
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:  SRVStencilFmt = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT; break;
        case DXGI_FORMAT_D24_UNORM_S8_UINT:     SRVStencilFmt = DXGI_FORMAT_X24_TYPELESS_G8_UINT; break;
        }

        if (checkColw && SRVStencilFmt != inFormat)
        {
            LOG_VERBOSE("%s: %s colwerted to %s!", __func__, DxgiFormat_cstr(inFormat), DxgiFormat_cstr(SRVStencilFmt));
        }

        return SRVStencilFmt;
    }

    void buildSplitStringFromNumber(uint64_t number, wchar_t * buf, size_t bufSize)
    {
#define SEPARATION_SIGN L" "
        int numberMillions = (int)number / 1000000;
        int numberThousands = ((int)number / 1000) % 1000;
        int numberUnits = (int)number % 1000;

        if (numberMillions != 0)
            swprintf_s(buf, bufSize, L"%d" SEPARATION_SIGN L"%03d" SEPARATION_SIGN L"%03d", numberMillions, numberThousands, numberUnits);
        else if (numberThousands != 0)
            swprintf_s(buf, bufSize, L"%d" SEPARATION_SIGN L"%03d", numberThousands, numberUnits);
        else
            swprintf_s(buf, bufSize, L"%d", numberUnits);
#undef SEPARATION_SIGN
    }

    float colwertToHorizontalFov(const ansel::Camera& cam, const ansel::Configuration& cfg, uint32_t viewportWidth, uint32_t viewportHeight)
    {
        if (cfg.fovType == ansel::kHorizontalFov)
            return cam.fov;

        return static_cast<float>(anselutils::colwertVerticalToHorizontalFov(cam.fov, viewportWidth, viewportHeight));
    }

    std::string appendTimeA(const char * inString_pre, const char * inString_post)
    {
        std::chrono::time_point<std::chrono::system_clock> lwrrentTime;
        lwrrentTime = std::chrono::system_clock::now();

        long long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(lwrrentTime.time_since_epoch()).count();

        std::time_t tt = std::chrono::system_clock::to_time_t(lwrrentTime);

        char time[32];
        tm buf;
        // TODO: VS version of localtime_s seems to have weird argument sequence
        localtime_s(&buf, &tt);
        std::strftime(time, 32, "%Y%m%d_%H%M%S_", &buf);

        int ms_cnt = milliseconds % 1000;

        std::stringstream in;
        in << inString_pre << time << std::setw(3) << std::setfill('0') << ms_cnt;

        if (inString_post)
            in << inString_post;
        
        return in.str();
    }

    std::wstring appendTimeW(const wchar_t * inString_pre, const wchar_t * inString_post)
    {
        std::chrono::time_point<std::chrono::system_clock> lwrrentTime;
        lwrrentTime = std::chrono::system_clock::now();

        long long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(lwrrentTime.time_since_epoch()).count();

        std::time_t tt = std::chrono::system_clock::to_time_t(lwrrentTime);

        char time[32];
        tm buf;
        // TODO: VS version of localtime_s seems to have weird argument sequence
        localtime_s(&buf, &tt);
        std::strftime(time, 32, "%Y%m%d_%H%M%S_", &buf);

        int ms_cnt = milliseconds % 1000;

        std::wstringstream in;

        in << inString_pre << darkroom::getWstrFromUtf8(time) << std::setw(3) << std::setfill(L'0') << ms_cnt;

        if (inString_post)
            in << inString_post;

        return in.str();
    }
        

    bool CreateDirectoryRelwrsively(const wchar_t *path)
    {
        return ::shadermod::ir::filehelpers::createDirectoryRelwrsively(path);
    }

    std::wstring getAppNameFromProcess()
    {
        wchar_t appPath[MAX_PATH];
        GetModuleFileName(NULL, appPath, MAX_PATH);
        const wchar_t* fileNameInPath = PathFindFileNameW(appPath);
        const wchar_t* fileNameExtension = PathFindExtension(appPath);
        return std::wstring(fileNameInPath, fileNameExtension);
    }
}
