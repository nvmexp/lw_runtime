#pragma once
namespace shadermod
{
namespace ir
{

    enum class FragmentFormat
    {
        kRGBA8_uint,
        kR32_uint,
        kR16_uint,
        kR8_uint,
        kR32_fp,
        kR16_fp,
        kRG32_uint,
        kRG16_uint,
        kRG8_uint,
        kRG32_fp,
        kRG16_fp,
        kRGBA32_fp,
        kRGBA16_fp,
        kRGBA16_uint,
        kBGRA8_uint,
        kR10G10B10A2_uint,

        kR11G11B10_float,

        kSRGBA8_uint,
        kSBGRA8_uint,

        // Depth formats
        kD24S8,
        kD32_fp_S8X24_uint,
        kD32_fp,

        kNUM_ENTRIES,
        kUnknown
    };

    enum class AddressType
    {
        kWrap,
        kClamp,
        kMirror,
        kBorder,

        kNUM_ENTRIES
    };

    enum class FilterType
    {
        kPoint,
        kLinear,

        kNUM_ENTRIES
    };

    enum class ConstType
    {
        kDT,        // float
        kElapsedTime,    // float
        kFrame,        // int
        kScreenSize,    // float2
        kCaptureState,    // int
        kTileUV,      // float4
        kDepthAvailable,  // int
        kHDRAvailable,    // int
        kHUDlessAvailable,  // int

        kUserConstantBase,

        kNUM_ENTRIES = kUserConstantBase
    };

    enum struct UserConstDataType
    {
        kBool = 0,
        kInt,
        kUInt,
        kFloat,
        NUM_ENTRIES
    };

    enum struct UiControlType
    {
        kSlider = 0,
        kCheckbox,
        kFlyout,
        kEditbox,
        kColorPicker,
        kRadioButton,

        NUM_ENTRIES
    };

    namespace userConstTypes
    {
        enum struct Bool: unsigned int
        {
            kFalse = 0,
            kTrue = 1
        };

        typedef int Int;
        typedef unsigned int UInt;
        typedef float Float;
    }
}
}
