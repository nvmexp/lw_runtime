#pragma once

#include "ir/TypeEnums.h"
#include "D3D11CommandProcessor.h"

#include <dxgiformat.h>

namespace shadermod
{
namespace ir
{
namespace ircolwert
{

    static FragmentFormat DXGIformatToFormat(DXGI_FORMAT inFFormat, bool skipAssert = false)
    {
        switch (inFFormat)
        {
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
            return FragmentFormat::kBGRA8_uint;
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
            return FragmentFormat::kRGBA8_uint;
        case DXGI_FORMAT_R16G16B16A16_UNORM:
            return FragmentFormat::kRGBA16_uint;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
            return FragmentFormat::kRGBA16_fp;
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
            return FragmentFormat::kRGBA32_fp;

        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return FragmentFormat::kSRGBA8_uint;
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return FragmentFormat::kSBGRA8_uint;

        case DXGI_FORMAT_R10G10B10A2_UNORM:
            return FragmentFormat::kR10G10B10A2_uint;

        case DXGI_FORMAT_R11G11B10_FLOAT:
            return FragmentFormat::kR11G11B10_float;

        case DXGI_FORMAT_R8G8_UNORM:
        case DXGI_FORMAT_R8G8_TYPELESS:
            return FragmentFormat::kRG8_uint;
        case DXGI_FORMAT_R16G16_FLOAT:
            return FragmentFormat::kRG16_fp;
        case DXGI_FORMAT_R16G16_UNORM:
            return FragmentFormat::kRG16_uint;
        case DXGI_FORMAT_R32G32_FLOAT:
            return FragmentFormat::kRG32_fp;
        case DXGI_FORMAT_R32G32_UINT:
            return FragmentFormat::kRG32_uint;

        case DXGI_FORMAT_R8_UNORM:
        case DXGI_FORMAT_R8_TYPELESS:
            return FragmentFormat::kR8_uint;
        case DXGI_FORMAT_R16_FLOAT:
            return FragmentFormat::kR16_fp;
        case DXGI_FORMAT_R16_UNORM:
        case DXGI_FORMAT_D16_UNORM:
            return FragmentFormat::kR16_uint;
        case DXGI_FORMAT_R32_FLOAT:
            return FragmentFormat::kR32_fp;
        case DXGI_FORMAT_R32_UINT:
            return FragmentFormat::kR32_uint;

        case DXGI_FORMAT_R24G8_TYPELESS:
        case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
        case DXGI_FORMAT_D24_UNORM_S8_UINT:
            return FragmentFormat::kD24S8;

        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
            return FragmentFormat::kD32_fp_S8X24_uint;

        case DXGI_FORMAT_D32_FLOAT:
            return FragmentFormat::kD32_fp;

        case DXGI_FORMAT_UNKNOWN:
            return FragmentFormat::kNUM_ENTRIES;

        default:
            if (skipAssert)
            {
                return FragmentFormat::kUnknown;
            }

            assert(!"unsupported format!");
            return FragmentFormat::kRGBA8_uint;
        }
    }

    static DXGI_FORMAT formatToDXGIFormat(FragmentFormat inFFormat)
    {
        switch (inFFormat)
        {
        case FragmentFormat::kBGRA8_uint:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        case FragmentFormat::kRGBA8_uint:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case FragmentFormat::kRGBA16_uint:
            return DXGI_FORMAT_R16G16B16A16_UNORM;
        case FragmentFormat::kRGBA16_fp:
            return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case FragmentFormat::kRGBA32_fp:
            return DXGI_FORMAT_R32G32B32A32_FLOAT;

        case FragmentFormat::kSRGBA8_uint:
            return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        case FragmentFormat::kSBGRA8_uint:
            return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;

        case FragmentFormat::kR10G10B10A2_uint:
            return DXGI_FORMAT_R10G10B10A2_UNORM;

        case FragmentFormat::kR11G11B10_float:
            return DXGI_FORMAT_R11G11B10_FLOAT;

        case FragmentFormat::kRG8_uint:
            return DXGI_FORMAT_R8G8_UNORM;
        case FragmentFormat::kRG16_fp:
            return DXGI_FORMAT_R16G16_FLOAT;
        case FragmentFormat::kRG16_uint:
            return DXGI_FORMAT_R16G16_UNORM;
        case FragmentFormat::kRG32_fp:
            return DXGI_FORMAT_R32G32_FLOAT;
        case FragmentFormat::kRG32_uint:
            return DXGI_FORMAT_R32G32_UINT;

        case FragmentFormat::kR8_uint:
            return DXGI_FORMAT_R8_UNORM;
        case FragmentFormat::kR16_fp:
            return DXGI_FORMAT_R16_FLOAT;
        case FragmentFormat::kR16_uint:
            return DXGI_FORMAT_R16_UNORM;
        case FragmentFormat::kR32_fp:
            return DXGI_FORMAT_R32_FLOAT;
        case FragmentFormat::kR32_uint:
            return DXGI_FORMAT_R32_UINT;

        case FragmentFormat::kD32_fp_S8X24_uint:
            return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        case FragmentFormat::kD32_fp:
            return DXGI_FORMAT_R32_FLOAT;

        case FragmentFormat::kD24S8:
            return DXGI_FORMAT_R24G8_TYPELESS;

        default:
            {
                // TODO[error]: throw an error here
                return DXGI_FORMAT_R8G8B8A8_UNORM;
            }
        }
    }

    static DXGI_FORMAT formatToDXGIFormat_DepthSRV(FragmentFormat inFFormat)
    {
        switch (inFFormat)
        {
        case FragmentFormat::kD24S8:
            {
                return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
            }
        case FragmentFormat::kD32_fp_S8X24_uint:
            {
                return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
            }
        case FragmentFormat::kD32_fp:
            {
                return DXGI_FORMAT_R32_FLOAT;
            }

        // These funny formats needed in case depth needs to be resolved (it will be written into color format)
        case FragmentFormat::kR32_fp:
            {
                return DXGI_FORMAT_R32_FLOAT;
            }
        case FragmentFormat::kR16_uint:
            {
                return DXGI_FORMAT_R16_UNORM;
            }
        case FragmentFormat::kRGBA8_uint:
            {
                return DXGI_FORMAT_R8G8B8A8_UNORM;
            }
        default:
            {
                // TODO[error]: throw an error here
                return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
            }
        }
    }

    static size_t formatBitsPerPixel(DXGI_FORMAT format)
    {
        switch (format)
        {
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
        case DXGI_FORMAT_R32G32B32A32_UINT:
        case DXGI_FORMAT_R32G32B32A32_SINT:
            return 128;

        case DXGI_FORMAT_R32G32B32_TYPELESS:
        case DXGI_FORMAT_R32G32B32_FLOAT:
        case DXGI_FORMAT_R32G32B32_UINT:
        case DXGI_FORMAT_R32G32B32_SINT:
            return 96;

        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_UNORM:
        case DXGI_FORMAT_R16G16B16A16_UINT:
        case DXGI_FORMAT_R16G16B16A16_SNORM:
        case DXGI_FORMAT_R16G16B16A16_SINT:
        case DXGI_FORMAT_R32G32_TYPELESS:
        case DXGI_FORMAT_R32G32_FLOAT:
        case DXGI_FORMAT_R32G32_UINT:
        case DXGI_FORMAT_R32G32_SINT:
        case DXGI_FORMAT_R32G8X24_TYPELESS:
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
        case DXGI_FORMAT_Y416:
        case DXGI_FORMAT_Y210:
        case DXGI_FORMAT_Y216:
            return 64;

        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R10G10B10A2_UINT:
        case DXGI_FORMAT_R11G11B10_FLOAT:
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_R8G8B8A8_UINT:
        case DXGI_FORMAT_R8G8B8A8_SNORM:
        case DXGI_FORMAT_R8G8B8A8_SINT:
        case DXGI_FORMAT_R16G16_TYPELESS:
        case DXGI_FORMAT_R16G16_FLOAT:
        case DXGI_FORMAT_R16G16_UNORM:
        case DXGI_FORMAT_R16G16_UINT:
        case DXGI_FORMAT_R16G16_SNORM:
        case DXGI_FORMAT_R16G16_SINT:
        case DXGI_FORMAT_R32_TYPELESS:
        case DXGI_FORMAT_D32_FLOAT:
        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R32_UINT:
        case DXGI_FORMAT_R32_SINT:
        case DXGI_FORMAT_R24G8_TYPELESS:
        case DXGI_FORMAT_D24_UNORM_S8_UINT:
        case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
        case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
        case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
        case DXGI_FORMAT_R8G8_B8G8_UNORM:
        case DXGI_FORMAT_G8R8_G8B8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8X8_UNORM:
        case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8X8_TYPELESS:
        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        case DXGI_FORMAT_AYUV:
        case DXGI_FORMAT_Y410:
        case DXGI_FORMAT_YUY2:
            return 32;

        case DXGI_FORMAT_P010:
        case DXGI_FORMAT_P016:
            return 24;

        case DXGI_FORMAT_R8G8_TYPELESS:
        case DXGI_FORMAT_R8G8_UNORM:
        case DXGI_FORMAT_R8G8_UINT:
        case DXGI_FORMAT_R8G8_SNORM:
        case DXGI_FORMAT_R8G8_SINT:
        case DXGI_FORMAT_R16_TYPELESS:
        case DXGI_FORMAT_R16_FLOAT:
        case DXGI_FORMAT_D16_UNORM:
        case DXGI_FORMAT_R16_UNORM:
        case DXGI_FORMAT_R16_UINT:
        case DXGI_FORMAT_R16_SNORM:
        case DXGI_FORMAT_R16_SINT:
        case DXGI_FORMAT_B5G6R5_UNORM:
        case DXGI_FORMAT_B5G5R5A1_UNORM:
        case DXGI_FORMAT_A8P8:
        case DXGI_FORMAT_B4G4R4A4_UNORM:
            return 16;

        case DXGI_FORMAT_LW12:
        case DXGI_FORMAT_420_OPAQUE:
        case DXGI_FORMAT_LW11:
            return 12;

        case DXGI_FORMAT_R8_TYPELESS:
        case DXGI_FORMAT_R8_UNORM:
        case DXGI_FORMAT_R8_UINT:
        case DXGI_FORMAT_R8_SNORM:
        case DXGI_FORMAT_R8_SINT:
        case DXGI_FORMAT_A8_UNORM:
        case DXGI_FORMAT_AI44:
        case DXGI_FORMAT_IA44:
        case DXGI_FORMAT_P8:
            return 8;

        case DXGI_FORMAT_R1_UNORM:
            return 1;

        case DXGI_FORMAT_BC1_TYPELESS:
        case DXGI_FORMAT_BC1_UNORM:
        case DXGI_FORMAT_BC1_UNORM_SRGB:
        case DXGI_FORMAT_BC4_TYPELESS:
        case DXGI_FORMAT_BC4_UNORM:
        case DXGI_FORMAT_BC4_SNORM:
            return 4;

        case DXGI_FORMAT_BC2_TYPELESS:
        case DXGI_FORMAT_BC2_UNORM:
        case DXGI_FORMAT_BC2_UNORM_SRGB:
        case DXGI_FORMAT_BC3_TYPELESS:
        case DXGI_FORMAT_BC3_UNORM:
        case DXGI_FORMAT_BC3_UNORM_SRGB:
        case DXGI_FORMAT_BC5_TYPELESS:
        case DXGI_FORMAT_BC5_UNORM:
        case DXGI_FORMAT_BC5_SNORM:
        case DXGI_FORMAT_BC6H_TYPELESS:
        case DXGI_FORMAT_BC6H_UF16:
        case DXGI_FORMAT_BC6H_SF16:
        case DXGI_FORMAT_BC7_TYPELESS:
        case DXGI_FORMAT_BC7_UNORM:
        case DXGI_FORMAT_BC7_UNORM_SRGB:
            return 8;

        default:
            return 0;
        }
    }

    static D3D11_FILTER filterToDXGIFilter(FilterType min, FilterType mag, FilterType mip)
    {
        // omg
        // there is bit-pattern, however D3D doesn't guarantee that it won't change, so no risk

        if (min == FilterType::kLinear && mag == FilterType::kLinear && mip == FilterType::kLinear)
            return D3D11_FILTER_MIN_MAG_MIP_LINEAR;

        if (min == FilterType::kLinear && mag == FilterType::kLinear && mip == FilterType::kPoint)
            return D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;

        if (min == FilterType::kPoint && mag == FilterType::kLinear && mip == FilterType::kLinear)
            return D3D11_FILTER_MIN_POINT_MAG_MIP_LINEAR;

        if (min == FilterType::kPoint && mag == FilterType::kLinear && mip == FilterType::kPoint)
            return D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;

        if (min == FilterType::kPoint && mag == FilterType::kPoint && mip == FilterType::kLinear)
            return D3D11_FILTER_MIN_MAG_POINT_MIP_LINEAR;

        if (min == FilterType::kPoint && mag == FilterType::kPoint && mip == FilterType::kPoint)
            return D3D11_FILTER_MIN_MAG_MIP_POINT;

        if (min == FilterType::kLinear && mag == FilterType::kPoint && mip == FilterType::kLinear)
            return D3D11_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR;

        if (min == FilterType::kLinear && mag == FilterType::kPoint && mip == FilterType::kPoint)
            return D3D11_FILTER_MIN_LINEAR_MAG_MIP_POINT;

        // TODO[error]: throw an error here
        return D3D11_FILTER_MIN_MAG_MIP_POINT;
    }

    static D3D11_TEXTURE_ADDRESS_MODE addressToDXGIAddress(AddressType inAddress)
    {
        switch (inAddress)
        {
        case AddressType::kWrap:
            {
                return D3D11_TEXTURE_ADDRESS_WRAP;
            }
        case AddressType::kClamp:
            {
                return D3D11_TEXTURE_ADDRESS_CLAMP;
            }
        case AddressType::kMirror:
            {
                return D3D11_TEXTURE_ADDRESS_MIRROR;
            }
        case AddressType::kBorder:
            {
                return D3D11_TEXTURE_ADDRESS_BORDER;
            }
        default:
            {
                // TODO[error]: throw an error here
                return D3D11_TEXTURE_ADDRESS_CLAMP;
            }
        }
    }

    static CmdProcConstDataType userConstTypeToCmdProcConstElementDataType(UserConstDataType type)
    {
        switch (type)
        {
        case UserConstDataType::kBool:
            return CmdProcConstDataType::kBool;
        case UserConstDataType::kInt:
            return CmdProcConstDataType::kInt;
        case UserConstDataType::kUInt:
            return CmdProcConstDataType::kUInt;
        case UserConstDataType::kFloat:
            return CmdProcConstDataType::kFloat;
        default:
            return CmdProcConstDataType::kNUM_ENTRIES;
        }
    }

    static CmdProcSystemConst constTypeToCmdProcSystemConst(ConstType type)
    {
        switch (type)
        {
        case ConstType::kDT:
            return CmdProcSystemConst::kDT;
        case ConstType::kElapsedTime:
            return CmdProcSystemConst::kElapsedTime;
        case ConstType::kFrame:
            return CmdProcSystemConst::kFrame;
        case ConstType::kScreenSize:
            return CmdProcSystemConst::kScreenSize;
        case ConstType::kCaptureState:
            return CmdProcSystemConst::kCaptureState;
        case ConstType::kTileUV:
            return CmdProcSystemConst::kTileUV;
        case ConstType::kDepthAvailable:
            return CmdProcSystemConst::kDepthAvailable;
        case ConstType::kHDRAvailable:
            return CmdProcSystemConst::kHDRAvailable;
        case ConstType::kHUDlessAvailable:
            return CmdProcSystemConst::kHUDlessAvailable;
        default:
            return CmdProcSystemConst::kNUM_ENTRIES;
        }
    }

}
}
}
