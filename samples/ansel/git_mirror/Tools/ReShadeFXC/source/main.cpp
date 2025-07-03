#include "effect_parser.hpp"
#include "effect_codegen.hpp"
#include "effect_preprocessor.hpp"
#include <acef.h>
#include <Windows.h> // MAKELANGID
#include <memory>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>

namespace
{
    const std::pair<std::string, int> localization_names[] = {
        { "de",     MAKELANGID(LANG_GERMAN, SUBLANG_GERMAN) },
        { "es",     MAKELANGID(LANG_SPANISH, SUBLANG_SPANISH) },
        { "es_ES",  MAKELANGID(LANG_SPANISH, SUBLANG_SPANISH_MODERN) },
        { "es_MX",  MAKELANGID(LANG_SPANISH, SUBLANG_SPANISH_MEXICAN) },
        { "fr",     MAKELANGID(LANG_FRENCH, SUBLANG_FRENCH) },
        { "it",     MAKELANGID(LANG_ITALIAN, SUBLANG_ITALIAN) },
        { "ru",     MAKELANGID(LANG_RUSSIAN, SUBLANG_RUSSIAN_RUSSIA) },
        { "zh",     MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_TRADITIONAL) },
        { "zh_CHT", MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_TRADITIONAL) },
        { "zh_CHS", MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_SIMPLIFIED) },
        { "ja",     MAKELANGID(LANG_JAPANESE, SUBLANG_JAPANESE_JAPAN) },
        { "cs",     MAKELANGID(LANG_CZECH, SUBLANG_CZECH_CZECH_REPUBLIC) },
        { "da",     MAKELANGID(LANG_DANISH, SUBLANG_DANISH_DENMARK) },
        { "el",     MAKELANGID(LANG_GREEK, SUBLANG_GREEK_GREECE) },
        { "en",     MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US) },
        { "en_US",  MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US) },
        { "en_UK",  MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_UK) },
        { "fi",     MAKELANGID(LANG_FINNISH, SUBLANG_FINNISH_FINLAND) },
        { "hu",     MAKELANGID(LANG_HUNGARIAN, SUBLANG_HUNGARIAN_HUNGARY) },
        { "ko",     MAKELANGID(LANG_KOREAN, SUBLANG_KOREAN) },
        { "nl",     MAKELANGID(LANG_DUTCH, SUBLANG_DUTCH) },
        { "nb",     MAKELANGID(LANG_NORWEGIAN, SUBLANG_NORWEGIAN_BOKMAL) },
        { "pl",	    MAKELANGID(LANG_POLISH, SUBLANG_POLISH_POLAND) },
        { "pt",     MAKELANGID(LANG_PORTUGUESE, SUBLANG_PORTUGUESE) },
        { "pt_PT",  MAKELANGID(LANG_PORTUGUESE, SUBLANG_PORTUGUESE) },
        { "pt_BR",  MAKELANGID(LANG_PORTUGUESE, SUBLANG_PORTUGUESE_BRAZILIAN) },
        { "sl",     MAKELANGID(LANG_SLOVENIAN, SUBLANG_SLOVENIAN_SLOVENIA) },
        { "sk",     MAKELANGID(LANG_SLOVAK, SUBLANG_SLOVAK_SLOVAKIA) },
        { "sv",     MAKELANGID(LANG_SWEDISH, SUBLANG_SWEDISH) },
        { "th",     MAKELANGID(LANG_THAI, SUBLANG_THAI_THAILAND) },
        { "tr",     MAKELANGID(LANG_TURKISH, SUBLANG_TURKISH_TURKEY) },
    };

    reshadefx::module module;
    acef::EffectStorage acef_module;
    std::filesystem::path hlsl_module_path;
    std::filesystem::path input_path, output_path, errors_path;
    std::unordered_map<std::string, size_t> read_texture_lookup;
    std::unordered_map<std::string, size_t> write_texture_lookup;

    template <typename T>
    inline auto alloc_mem(size_t count)
    {
        return static_cast<T *>(acef_module.allocateMem(count * sizeof(T)));
    }
    inline auto alloc_and_copy_string(const std::string &input)
    {
        size_t len = input.size() + 1; // string + null-terminator
        char *const output = alloc_mem<char>(len);
        memcpy(output, input.c_str(), len);
        return output;
    }

    auto colwert_format(reshadefx::texture_format value)
    {
        switch (value)
        {
        case reshadefx::texture_format::r8:
            return acef::FragmentFormat::kR8_uint;
        case reshadefx::texture_format::r16f:
            return acef::FragmentFormat::kR16_fp;
        case reshadefx::texture_format::r32f:
            return acef::FragmentFormat::kR32_fp;
        case reshadefx::texture_format::rg8:
            return acef::FragmentFormat::kRG8_uint;
        case reshadefx::texture_format::rg16:
            return acef::FragmentFormat::kRG16_uint;
        case reshadefx::texture_format::rg16f:
            return acef::FragmentFormat::kRG16_fp;
        case reshadefx::texture_format::rg32f:
            return acef::FragmentFormat::kRG32_fp;
        default:
        case reshadefx::texture_format::rgba8:
            return acef::FragmentFormat::kRGBA8_uint;
        case reshadefx::texture_format::rgba16:
            return acef::FragmentFormat::kRGBA16_uint;
        case reshadefx::texture_format::rgba16f:
            return acef::FragmentFormat::kRGBA16_fp;
        case reshadefx::texture_format::rgba32f:
            return acef::FragmentFormat::kRGBA32_fp;
        case reshadefx::texture_format::rgb10a2:
            return acef::FragmentFormat::kR10G10B10A2_uint;
        }
    }
    auto colwert_address_mode(reshadefx::texture_address_mode value)
    {
        switch (value)
        {
        case reshadefx::texture_address_mode::wrap:
            return acef::ResourceSamplerAddressType::kWrap;
        case reshadefx::texture_address_mode::mirror:
            return acef::ResourceSamplerAddressType::kMirror;
        case reshadefx::texture_address_mode::border:
            return acef::ResourceSamplerAddressType::kBorder;
        default:
        case reshadefx::texture_address_mode::clamp:
            return acef::ResourceSamplerAddressType::kClamp;
        }
    }
    auto colwert_pass_blend_op(reshadefx::pass_blend_op value)
    {
        switch (value)
        {
        default:
        case reshadefx::pass_blend_op::add:
            return acef::BlendOp::kAdd;
        case reshadefx::pass_blend_op::subtract:
            return acef::BlendOp::kSub;
        case reshadefx::pass_blend_op::rev_subtract:
            return acef::BlendOp::kRevSub;
        case reshadefx::pass_blend_op::min:
            return acef::BlendOp::kMin;
        case reshadefx::pass_blend_op::max:
            return acef::BlendOp::kMax;
        }
    }
    auto colwert_pass_blend_func(reshadefx::pass_blend_func value)
    {
        switch (value)
        {
        default:
        case reshadefx::pass_blend_func::zero:
            return acef::BlendCoef::kZero;
        case reshadefx::pass_blend_func::one:
            return acef::BlendCoef::kOne;
        case reshadefx::pass_blend_func::src_color:
            return acef::BlendCoef::kSrcColor;
        case reshadefx::pass_blend_func::src_alpha:
            return acef::BlendCoef::kSrcAlpha;
        case reshadefx::pass_blend_func::ilw_src_color:
            return acef::BlendCoef::kIlwSrcColor;
        case reshadefx::pass_blend_func::ilw_src_alpha:
            return acef::BlendCoef::kIlwSrcAlpha;
        case reshadefx::pass_blend_func::dst_color:
            return acef::BlendCoef::kDstColor;
        case reshadefx::pass_blend_func::dst_alpha:
            return acef::BlendCoef::kDstAlpha;
        case reshadefx::pass_blend_func::ilw_dst_color:
            return acef::BlendCoef::kIlwDstColor;
        case reshadefx::pass_blend_func::ilw_dst_alpha:
            return acef::BlendCoef::kIlwDstAlpha;
        }
    }
    auto colwert_pass_stencil_op(reshadefx::pass_stencil_op value)
    {
        switch (value)
        {
        default:
        case reshadefx::pass_stencil_op::zero:
            return acef::StencilOp::kZero;
        case reshadefx::pass_stencil_op::keep:
            return acef::StencilOp::kKeep;
        case reshadefx::pass_stencil_op::ilwert:
            return acef::StencilOp::kIlwert;
        case reshadefx::pass_stencil_op::replace:
            return acef::StencilOp::kReplace;
        case reshadefx::pass_stencil_op::incr:
            return acef::StencilOp::kIncr;
        case reshadefx::pass_stencil_op::incr_sat:
            return acef::StencilOp::kIncrSat;
        case reshadefx::pass_stencil_op::decr:
            return acef::StencilOp::kDecr;
        case reshadefx::pass_stencil_op::decr_sat:
            return acef::StencilOp::kDecrSat;
        }
    }
    auto colwert_pass_stencil_func(reshadefx::pass_stencil_func value)
    {
        switch (value)
        {
        case reshadefx::pass_stencil_func::never:
            return acef::ComparisonFunc::kNever;
        case reshadefx::pass_stencil_func::equal:
            return acef::ComparisonFunc::kEqual;
        case reshadefx::pass_stencil_func::not_equal:
            return acef::ComparisonFunc::kNotEqual;
        case reshadefx::pass_stencil_func::less:
            return acef::ComparisonFunc::kLess;
        case reshadefx::pass_stencil_func::less_equal:
            return acef::ComparisonFunc::kLessEqual;
        case reshadefx::pass_stencil_func::greater:
            return acef::ComparisonFunc::kGreater;
        case reshadefx::pass_stencil_func::greater_equal:
            return acef::ComparisonFunc::kGreaterEqual;
        default:
        case reshadefx::pass_stencil_func::always:
            return acef::ComparisonFunc::kAlways;
        }
    }

    void colwert_samplers()
    {
        std::unordered_set<uint32_t> used_bindings;
        for (const reshadefx::sampler_info &info : module.samplers)
        {
            // Avoid adding the same sampler decription multiple times
            if (used_bindings.find(info.binding) != used_bindings.end())
                continue;
            used_bindings.insert(info.binding);

            acef_module.resourceHeader.binStorage.samplersNum++;
            acef::ResourceSampler &out = acef_module.samplers.emplace_back();

            switch (info.filter)
            {
            case reshadefx::texture_filter::min_mag_mip_point:
                out.binStorage.filterMin = acef::ResourceSamplerFilterType::kPoint;
                out.binStorage.filterMag = acef::ResourceSamplerFilterType::kPoint;
                out.binStorage.filterMip = acef::ResourceSamplerFilterType::kPoint;
                break;
            case reshadefx::texture_filter::min_mag_point_mip_linear:
                out.binStorage.filterMin = acef::ResourceSamplerFilterType::kPoint;
                out.binStorage.filterMag = acef::ResourceSamplerFilterType::kPoint;
                out.binStorage.filterMip = acef::ResourceSamplerFilterType::kLinear;
                break;
            case reshadefx::texture_filter::min_point_mag_linear_mip_point:
                out.binStorage.filterMin = acef::ResourceSamplerFilterType::kPoint;
                out.binStorage.filterMag = acef::ResourceSamplerFilterType::kLinear;
                out.binStorage.filterMip = acef::ResourceSamplerFilterType::kPoint;
                break;
            case reshadefx::texture_filter::min_point_mag_mip_linear:
                out.binStorage.filterMin = acef::ResourceSamplerFilterType::kPoint;
                out.binStorage.filterMag = acef::ResourceSamplerFilterType::kLinear;
                out.binStorage.filterMip = acef::ResourceSamplerFilterType::kLinear;
                break;
            case reshadefx::texture_filter::min_linear_mag_mip_point:
                out.binStorage.filterMin = acef::ResourceSamplerFilterType::kLinear;
                out.binStorage.filterMag = acef::ResourceSamplerFilterType::kPoint;
                out.binStorage.filterMip = acef::ResourceSamplerFilterType::kPoint;
                break;
            case reshadefx::texture_filter::min_linear_mag_point_mip_linear:
                out.binStorage.filterMin = acef::ResourceSamplerFilterType::kLinear;
                out.binStorage.filterMag = acef::ResourceSamplerFilterType::kPoint;
                out.binStorage.filterMip = acef::ResourceSamplerFilterType::kLinear;
                break;
            case reshadefx::texture_filter::min_mag_linear_mip_point:
                out.binStorage.filterMin = acef::ResourceSamplerFilterType::kLinear;
                out.binStorage.filterMag = acef::ResourceSamplerFilterType::kLinear;
                out.binStorage.filterMip = acef::ResourceSamplerFilterType::kPoint;
                break;
            default:
            case reshadefx::texture_filter::min_mag_mip_linear:
                out.binStorage.filterMin = acef::ResourceSamplerFilterType::kLinear;
                out.binStorage.filterMag = acef::ResourceSamplerFilterType::kLinear;
                out.binStorage.filterMip = acef::ResourceSamplerFilterType::kLinear;
                break;
            }

            out.binStorage.addrU = colwert_address_mode(info.address_u);
            out.binStorage.addrV = colwert_address_mode(info.address_v);
            out.binStorage.addrW = colwert_address_mode(info.address_w);

            if (info.min_lod > 0 || info.max_lod < FLT_MAX)
            {
                std::cout << "sampler " << info.unique_name << " uses unsupported \"MinLOD\" or \"MaxLOD\" value" << std::endl;
                if (!errors_path.empty()) std::ofstream(errors_path, std::ios::app) << "sampler " << info.unique_name << " uses unsupported \"MinLOD\" or \"MaxLOD\" value" << std::endl;
            }
            if (info.lod_bias != 0)
            {
                std::cout << "sampler " << info.unique_name << " uses unsupported \"MipLODBias\" value" << std::endl;
                if (!errors_path.empty()) std::ofstream(errors_path, std::ios::app) << "sampler " << info.unique_name << " uses unsupported \"MipLODBias\" value" << std::endl;
            }
            if (info.srgb)
            {
                std::cout << "sampler " << info.unique_name << " uses unsupported \"SRGBTexture\" value" << std::endl;
                if (!errors_path.empty()) std::ofstream(errors_path, std::ios::app) << "sampler " << info.unique_name << " uses unsupported \"SRGBTexture\" value" << std::endl;
            }
        }
    }
    void colwert_textures()
    {
        // First need to callwlate texture counts so to be able to callwlate handles below
        for (const reshadefx::texture_info &info : module.textures)
        {
            if (info.semantic.empty()) // Only normal textures count, system textures are just special handles
            {
                if (std::find_if(info.annotations.begin(), info.annotations.end(),
                    [](const auto &a) { return a.name == "source"; }) != info.annotations.end())
                    acef_module.resourceHeader.binStorage.texturesFromFileNum++;
                else
                    acef_module.resourceHeader.binStorage.texturesIntermediateNum++;
            }
        }

        // Alocate sufficient enough memory for buffer handles
        acef_module.resourceHeader.readBufferTextureHandles = alloc_mem<uint32_t>(module.textures.size() + 2);
        acef_module.resourceHeader.writeBufferTextureHandles = alloc_mem<uint32_t>(module.textures.size() + 2);

        // Define default back buffer resources
        acef::ResourceTextureIntermediate default_write_tex;
        default_write_tex.binStorage.width.binStorage.mul = 1;
        default_write_tex.binStorage.width.binStorage.texSizeBase = acef::TextureSizeBase::kColorBufferWidth;
        default_write_tex.binStorage.height.binStorage.mul = 1;
        default_write_tex.binStorage.height.binStorage.texSizeBase = acef::TextureSizeBase::kColorBufferHeight;
        default_write_tex.binStorage.format = acef::FragmentFormat::kRGBA8_uint;
        default_write_tex.binStorage.levels = 1;

        acef_module.texturesIntermediate.push_back(default_write_tex); // Target for passes where index % 2 == 0
        acef_module.texturesIntermediate.push_back(default_write_tex); // Target for passes where index % 2 == 1
        acef_module.resourceHeader.binStorage.texturesIntermediateNum += 2;
        acef_module.resourceHeader.readBufferTextureHandles[0] = 0;
        acef_module.resourceHeader.readBufferTextureHandles[1] = 1;
        acef_module.resourceHeader.writeBufferTextureHandles[0] = 0;
        acef_module.resourceHeader.writeBufferTextureHandles[1] = 1;

        uint32_t read_texture_index = 2, write_texture_index = 2;
        // First two intermediate textures are default back buffer resource from above
        uint32_t intermediate_texture_index = 2, file_texture_index = 0;

        for (const reshadefx::texture_info &info : module.textures)
        {
            if (!info.semantic.empty())
            {
                uint32_t handle = 0xFFFFFFFF;
                if (info.semantic == "COLOR" || info.semantic == "SV_TARGET")
                    handle = uint32_t(acef::SystemTexture::kInputColor);
                else if (info.semantic == "DEPTH" || info.semantic == "SV_DEPTH")
                    handle = uint32_t(acef::SystemTexture::kInputDepth);
                else if (info.semantic == "HDR")
                    handle = uint32_t(acef::SystemTexture::kInputHDR);
                else if (info.semantic == "HUDLESS")
                    handle = uint32_t(acef::SystemTexture::kInputHUDless);
                else if (info.semantic == "COLOR_BASE")
                    handle = uint32_t(acef::SystemTexture::kInputColorBase);

                read_texture_lookup[info.unique_name] = read_texture_index;

                acef_module.resourceHeader.readBufferTextureHandles[read_texture_index++] = handle;
                continue;
            }

            // Try to figure out relative dimensions
            acef::TextureSizeStorage width;
            if (info.width & 0x80000000) // Check for sign bit
            {
                width.binStorage.mul = static_cast<float>(-static_cast<int32_t>(info.width)) / 10000.0f;
                width.binStorage.texSizeBase = acef::TextureSizeBase::kColorBufferWidth;
            }
            else
            {
                width.binStorage.mul = static_cast<float>(info.width);
                width.binStorage.texSizeBase = acef::TextureSizeBase::kOne;
            }
            acef::TextureSizeStorage height;
            if (info.width & 0x80000000)
            {
                height.binStorage.mul = static_cast<float>(-static_cast<int32_t>(info.height)) / 10000.0f;
                height.binStorage.texSizeBase = acef::TextureSizeBase::kColorBufferHeight;
            }
            else
            {
                height.binStorage.mul = static_cast<float>(info.height);
                height.binStorage.texSizeBase = acef::TextureSizeBase::kOne;
            }

            // If texture has a source annotation then it loads an image from a file
            if (const auto source_it = std::find_if(info.annotations.begin(), info.annotations.end(),
                [](const auto &a) { return a.name == "source"; });
                source_it != info.annotations.end())
            {
                assert(info.levels == 1); // File textures should not have mipmaps

                acef::ResourceTextureFromFile &out_texture = acef_module.texturesFromFile.emplace_back();
                out_texture.binStorage.width = width;
                out_texture.binStorage.height = height;
                out_texture.binStorage.format = colwert_format(info.format);

                const std::string &source = source_it->value.string_data;
                out_texture.pathUtf8 = alloc_and_copy_string(source);
                out_texture.binStorage.pathLen = static_cast<uint32_t>(source.size());

                // Texture handle is overall index in all three texture types, in succession:
                //   texturesParametrizedNum + texturesIntermediateNum + texturesFromFileNum
                uint32_t handle = file_texture_index++;
                handle += acef_module.resourceHeader.binStorage.texturesParametrizedNum; // Total count was callwlated above
                handle += acef_module.resourceHeader.binStorage.texturesIntermediateNum;

                read_texture_lookup[info.unique_name] = read_texture_index; // Remember handle index for pass creation

                acef_module.resourceHeader.readBufferTextureHandles[read_texture_index++] = handle;
            }
            else
            {
                acef::ResourceTextureIntermediate &out_texture = acef_module.texturesIntermediate.emplace_back();
                out_texture.binStorage.width = width;
                out_texture.binStorage.height = height;
                out_texture.binStorage.format = colwert_format(info.format);
                out_texture.binStorage.levels = info.levels;

                // Texture handle is overall index in all three texture types, see above for details
                uint32_t handle = intermediate_texture_index++;
                handle += acef_module.resourceHeader.binStorage.texturesParametrizedNum;

                read_texture_lookup[info.unique_name] = read_texture_index;
                write_texture_lookup[info.unique_name] = write_texture_index;

                acef_module.resourceHeader.readBufferTextureHandles[read_texture_index++] = handle;
                acef_module.resourceHeader.writeBufferTextureHandles[write_texture_index++] = handle;
            }
        }

        acef_module.resourceHeader.binStorage.readBuffersNum = read_texture_index;
        acef_module.resourceHeader.binStorage.writeBuffersNum = write_texture_index;
    }
    void colwert_uniforms()
    {
        // ReShade only defines one single global constant buffer ("_Global")
        acef_module.resourceHeader.binStorage.constantBuffersNum = 1;

        // Define said constant buffer
        acef::ResourceConstantBuffer &cbuffer = acef_module.constantBuffers.emplace_back();

        cbuffer.binStorage.constantsNum = static_cast<uint32_t>(module.uniforms.size());
        cbuffer.constantHandle = alloc_mem<uint32_t>(module.uniforms.size());
        cbuffer.constantOffsetInComponents = alloc_mem<uint32_t>(module.uniforms.size());
        cbuffer.constantNameLens = alloc_mem<uint16_t>(module.uniforms.size());
        cbuffer.constantNameOffsets = alloc_mem<uint32_t>(module.uniforms.size());

        uint32_t constant_index = 0;

        for (const reshadefx::uniform_info &info : module.uniforms)
        {
            uint32_t constant_handle = 0xFFFFFFFF; // Undefined constant

            // If uniform has a source annotation then it is a special system constant
            if (const auto source_it = std::find_if(info.annotations.begin(), info.annotations.end(),
                [](const auto &a) { return a.name == "source"; });
                source_it != info.annotations.end())
            {
                const std::string &source = source_it->value.string_data;
                if (source == "frametime")
                    constant_handle = uint32_t(acef::SystemConstant::kDT);
                else if (source == "framecount")
                    constant_handle = uint32_t(acef::SystemConstant::kFrame);
                else if (source == "timer")
                    constant_handle = uint32_t(acef::SystemConstant::kElapsedTime);
                else if (source == "screensize")
                    constant_handle = uint32_t(acef::SystemConstant::kScreenSize);
                else if (source == "tile_uv")
                    constant_handle = uint32_t(acef::SystemConstant::kTileUV);
                else if (source == "capture_state")
                    constant_handle = uint32_t(acef::SystemConstant::kCaptureState);
                else if (source == "bufready_depth")
                    constant_handle = uint32_t(acef::SystemConstant::kDepthAvailable);
                else if (source == "bufready_hdr")
                    constant_handle = uint32_t(acef::SystemConstant::kHDRAvailable);
                else if (source == "bufready_hudless")
                    constant_handle = uint32_t(acef::SystemConstant::kHUDlessAvailable);
            }
            else
            {
                constant_handle = static_cast<uint32_t>(acef_module.userConstants.size());

                acef::UserConstant &constant = acef_module.userConstants.emplace_back();
                constant.controlNameAscii = alloc_and_copy_string(info.name);
                constant.binStorage.controlNameLen = static_cast<uint32_t>(info.name.size());
                constant.binStorage.dataDimensionality = static_cast<uint8_t>(info.type.rows); // Note: Matrices are not supported

                switch (info.type.base)
                {
                case reshadefx::type::t_bool:
                    constant.binStorage.dataType = acef::UserConstDataType::kBool;
                    break;
                case reshadefx::type::t_int:
                    constant.binStorage.dataType = acef::UserConstDataType::kInt;
                    break;
                case reshadefx::type::t_uint:
                    constant.binStorage.dataType = acef::UserConstDataType::kUInt;
                    break;
                case reshadefx::type::t_float:
                    constant.binStorage.dataType = acef::UserConstDataType::kFloat;
                    break;
                }

                // These defaults may be altered depending on the UI Type
                float default_ui_min = 0.0f;
                float default_ui_max = 1.0f;
                float default_ui_value_step = 0.01f;

                std::string ui_type;
                std::vector<std::string> variable_names;
                if (const auto it = std::find_if(info.annotations.begin(), info.annotations.end(),
                    [](const auto &a) { return a.name == "ui_type"; });
                    it != info.annotations.end())
                {
                    ui_type = it->value.string_data;
                }

                if (info.type.is_boolean())
                {
                    constant.binStorage.uiControlType = acef::UIControlType::kCheckbox;
                }
                else if (ui_type == "drag" || ui_type == "slider")
                {
                    constant.binStorage.uiControlType = acef::UIControlType::kSlider;
                }
                else if (ui_type == "color")
                {
                    constant.binStorage.uiControlType = acef::UIControlType::kColorPicker;
                    variable_names.push_back("Red");
                    variable_names.push_back("Green");
                    variable_names.push_back("Blue");
                    variable_names.push_back("Alpha");
                }
                else
                if (const auto it = std::find_if(info.annotations.begin(), info.annotations.end(),
                    [](const auto &a) { return a.name == "ui_items"; });
                    it != info.annotations.end())
                {
                    if (ui_type == "radio")
                    {
                        constant.binStorage.uiControlType = acef::UIControlType::kRadioButton;
                    }
                    else
                    {
                        constant.binStorage.uiControlType = acef::UIControlType::kFlyout;
                    }

                    std::string ui_items;
                    if (it != info.annotations.end())
                        ui_items = it->value.string_data;
                    // Make sure last item in the list is null-terminated like all others
                    if (ui_items.size() && ui_items[ui_items.size() - 1] != '\0')
                        ui_items.push_back('\0');

                    // Number of options is equal to the number of nulls in the list
                    constant.binStorage.optionsNum = static_cast<uint16_t>(std::count(ui_items.begin(), ui_items.end(), '\0'));

                    constant.optionNames = alloc_mem<acef::UILocalizedStringStorage>(constant.binStorage.optionsNum);
                    constant.optionNameByteOffsets = alloc_mem<uint64_t>(constant.binStorage.optionsNum);
                    constant.optiolwalues = alloc_mem<acef::TypelessVariableStorage>(constant.binStorage.optionsNum);
                    constant.optionNamesBuffers = alloc_mem<acef::UILocalizedStringBuffers>(constant.binStorage.optionsNum);

                    for (size_t offset = 0, next, k = 0; (next = ui_items.find_first_of('\0', offset)) != std::string::npos; offset = next + 1, ++k)
                    {
                        *reinterpret_cast<uint32_t *>(constant.optiolwalues[k].binStorage.data) = static_cast<uint32_t>(k);

                        constant.optionNames[k].binStorage.localizationsNum = 0;
                        constant.optionNameByteOffsets[k] = 0;

                        const std::string item_name = ui_items.c_str() + offset;
                        constant.optionNamesBuffers[k].defaultStringAscii = alloc_and_copy_string(item_name);
                        constant.optionNames[k].binStorage.defaultStringLen = static_cast<uint16_t>(next - offset);
                    }

                    default_ui_min = 0.0f;
                    default_ui_max = static_cast<float>(constant.binStorage.optionsNum - 1);
                    default_ui_value_step = 1.0f;
                }
                else // Default to input box if control type is unknown
                {
                    default_ui_min = -1000000.0f;
                    default_ui_max = +1000000.0f;
                    constant.binStorage.uiControlType = acef::UIControlType::kEditbox;
                }

                // Look for UI boundary annotations and also colwert them to the same type as the constant
                for (uint32_t row = 0, offset = 0; row < info.type.rows; row++, offset += 4)
                {
                    const auto colwert_annotation = [&info, row](const char *name, float default, void *data) {
                        if (const auto it = std::find_if(info.annotations.begin(), info.annotations.end(),
                            [name](const auto &a) { return a.name == name; });
                            it != info.annotations.end())
                        {
                            // Propagate last row of annotation to all remaining rows of constant
                            const uint32_t i = (row < it->type.rows) ? row : it->type.rows - 1;
                            // Colwert annotation value to floating-point
                            default = it->type.is_floating_point() ?
                                it->value.as_float[i] :
                                static_cast<float>(it->value.as_int[i]);
                        }
                        // Colwert floating-point value to constant data type
                        *static_cast<uint32_t *>(data) = info.type.is_floating_point() ?
                            *reinterpret_cast<uint32_t *>(&default) : static_cast<uint32_t>(default);
                    };

                    colwert_annotation("ui_min", default_ui_min, constant.binStorage.uiMinimumValue.binStorage.data + offset);
                    colwert_annotation("ui_max", default_ui_max, constant.binStorage.uiMaximumValue.binStorage.data + offset);
                    colwert_annotation("ui_step", default_ui_value_step, constant.binStorage.uiValueStep.binStorage.data + offset);
                }

                constant.binStorage.minimumValue = constant.binStorage.uiMinimumValue;
                constant.binStorage.maximumValue = constant.binStorage.uiMaximumValue;

                // Set Variable Names
                const size_t num_variable_names = constant.binStorage.dataDimensionality;
                while (variable_names.size() < num_variable_names) variable_names.emplace_back(""); // Make sure we fill the whole array.
                constant.variableNames = alloc_mem<acef::UILocalizedStringStorage>(num_variable_names);
                constant.variableNameByteOffsets = alloc_mem<uint64_t>(num_variable_names);
                constant.variableNamesBuffers = alloc_mem<acef::UILocalizedStringBuffers>(num_variable_names);
                for (size_t k = 0; k < num_variable_names; k++)
                {
                    constant.variableNames[k].binStorage.localizationsNum = 0;
                    constant.variableNameByteOffsets[k] = 0;

                    std::string item_name = variable_names[k];

                    constant.variableNamesBuffers[k].defaultStringAscii = alloc_and_copy_string(item_name);
                    constant.variableNames[k].binStorage.defaultStringLen = static_cast<uint16_t>(item_name.size());
                }

                // Get default label name
                std::string ui_label = info.name; // Default to variable name if no explicit label is set
                if (const auto it = std::find_if(info.annotations.begin(), info.annotations.end(),
                    [](const auto &a) { return a.name == "ui_label"; });
                    it != info.annotations.end())
                {
                    ui_label = it->value.string_data;
                }
                constant.labelBuffers.defaultStringAscii = alloc_and_copy_string(ui_label);
                constant.binStorage.label.binStorage.defaultStringLen = static_cast<uint16_t>(ui_label.size());

                // Get localized label names
                uint16_t &num_label_localizations = constant.binStorage.label.binStorage.localizationsNum = 0;
                constant.labelBuffers.localizedStrings = alloc_mem<acef::UILocalizedStringBuffers::LocalizedString>(ARRAYSIZE(localization_names));
                for (size_t i = 0; i < ARRAYSIZE(localization_names); ++i)
                {
                    if (const auto it = std::find_if(info.annotations.begin(), info.annotations.end(),
                        [i](const auto &a) { return a.name == "ui_label_" + localization_names[i].first; });
                        it != info.annotations.end())
                    {
                        ui_label = it->value.string_data;

                        auto &localized_string = constant.labelBuffers.localizedStrings[num_label_localizations++];
                        localized_string.stringUtf8 = alloc_and_copy_string(ui_label);
                        localized_string.binStorage.langid = static_cast<uint16_t>(localization_names[i].second);
                        localized_string.binStorage.strLen = static_cast<uint16_t>(ui_label.size());
                    }
                }

                // Get default tooltip string
                std::string ui_tooltip;
                if (const auto it = std::find_if(info.annotations.begin(), info.annotations.end(),
                    [](const auto &a) { return a.name == "ui_tooltip"; });
                    it != info.annotations.end())
                {
                    ui_tooltip = it->value.string_data;
                }
                constant.hintBuffers.defaultStringAscii = alloc_and_copy_string(ui_tooltip);
                constant.binStorage.hint.binStorage.defaultStringLen = static_cast<uint16_t>(ui_tooltip.size());

                // Get localized tooltip strings
                uint16_t &num_tooltip_localizations = constant.binStorage.hint.binStorage.localizationsNum = 0;
                constant.hintBuffers.localizedStrings = alloc_mem<acef::UILocalizedStringBuffers::LocalizedString>(ARRAYSIZE(localization_names));
                for (size_t i = 0; i < ARRAYSIZE(localization_names); ++i)
                {
                    if (const auto it = std::find_if(info.annotations.begin(), info.annotations.end(),
                        [i](const auto &a) { return a.name == "ui_tooltip_" + localization_names[i].first; });
                        it != info.annotations.end())
                    {
                        ui_tooltip = it->value.string_data;

                        auto &localized_string = constant.hintBuffers.localizedStrings[num_tooltip_localizations++];
                        localized_string.stringUtf8 = alloc_and_copy_string(ui_tooltip);
                        localized_string.binStorage.langid = static_cast<uint16_t>(localization_names[i].second);
                        localized_string.binStorage.strLen = static_cast<uint16_t>(ui_tooltip.size());
                    }
                }

                // Copy initial value to default value storage
                if (info.has_initializer_value)
                {
                    for (uint32_t row = 0; row < 4; ++row)
                    {
                        *reinterpret_cast<uint32_t *>(&constant.binStorage.defaultValue.binStorage.data[row * 4]) =
                            info.initializer_value.as_uint[row];
                    }
                }
            }

            cbuffer.constantHandle[constant_index] = constant_handle;
            cbuffer.constantOffsetInComponents[constant_index] = info.offset / 4; // Offset is in bytes, so divide by 4 to get offset in components
            cbuffer.constantNameLens[constant_index] = 0; // Binding via offset, so do not need name
            cbuffer.constantNameOffsets[constant_index] = 0;

            ++constant_index;
        }

        cbuffer.constantNames = "";

        acef_module.uiControlsHeader.binStorage.userConstantsNum = static_cast<uint32_t>(acef_module.userConstants.size());
    }

    void colwert_technique(const reshadefx::technique_info &technique)
    {
        if (technique.passes.empty())
            return; // Empty techniques need not be colwerted

        acef_module.passes.reserve(acef_module.passesHeader.binStorage.passesNum =
            static_cast<uint32_t>(technique.passes.size()));

        uint32_t pass_index = 0;
        uint32_t last_back_buffer_pass = std::numeric_limits<uint32_t>::max();

        for (const reshadefx::pass_info &pass_in : technique.passes)
        {
            acef::Pass &pass_out = acef_module.passes.emplace_back();

            { // Fill out rasterizer state description
                auto &rs_state = pass_out.binStorage.rasterizerState.binStorage;
                rs_state.fillMode = acef::RasterizerFillMode::kSolid;
                rs_state.lwllMode = acef::RasterizerLwllMode::kNone;
                rs_state.depthBias = 0;
                rs_state.depthBiasClamp = 0.0f;
                rs_state.slopeScaledDepthBias = 0.0f;
                rs_state.frontCounterClockwise = false;
                rs_state.depthClipEnable = true;
                rs_state.scissorEnable = false;
                rs_state.multisampleEnable = false;
                rs_state.antialiasedLineEnable = false;
            }

            { // Fill out depth-stencil state description
                auto &ds_state = pass_out.binStorage.depthStencilState.binStorage;
                ds_state.backFace.binStorage.failOp = colwert_pass_stencil_op(pass_in.stencil_op_fail);
                ds_state.backFace.binStorage.depthFailOp = colwert_pass_stencil_op(pass_in.stencil_op_depth_fail);
                ds_state.backFace.binStorage.passOp = colwert_pass_stencil_op(pass_in.stencil_op_pass);
                ds_state.backFace.binStorage.func = colwert_pass_stencil_func(pass_in.stencil_comparison_func);
                ds_state.frontFace.binStorage.failOp = ds_state.backFace.binStorage.failOp;
                ds_state.frontFace.binStorage.depthFailOp = ds_state.backFace.binStorage.depthFailOp;
                ds_state.frontFace.binStorage.passOp = ds_state.backFace.binStorage.passOp;
                ds_state.frontFace.binStorage.func = ds_state.backFace.binStorage.func;
                ds_state.depthWriteMask = acef::DepthWriteMask::kZero;
                ds_state.depthFunc = acef::ComparisonFunc::kAlways;
                ds_state.stencilReadMask = pass_in.stencil_read_mask;
                ds_state.stencilWriteMask = pass_in.stencil_write_mask;
                ds_state.isDepthEnabled = false;
                ds_state.isStencilEnabled = pass_in.stencil_enable;
            }

            { // Fill out alpha-blending state description
                auto &bl_state = pass_out.binStorage.alphaBlendState.binStorage;
                bl_state.renderTargetState[0].binStorage.src = colwert_pass_blend_func(pass_in.src_blend);
                bl_state.renderTargetState[0].binStorage.dst = colwert_pass_blend_func(pass_in.dest_blend);
                bl_state.renderTargetState[0].binStorage.op = colwert_pass_blend_op(pass_in.blend_op);
                bl_state.renderTargetState[0].binStorage.srcAlpha = colwert_pass_blend_func(pass_in.src_blend_alpha);
                bl_state.renderTargetState[0].binStorage.dstAlpha = colwert_pass_blend_func(pass_in.dest_blend_alpha);
                bl_state.renderTargetState[0].binStorage.opAlpha = colwert_pass_blend_op(pass_in.blend_op_alpha);
                bl_state.renderTargetState[0].binStorage.renderTargetWriteMask = static_cast<acef::ColorWriteEnableBits>(pass_in.color_write_mask);
                bl_state.renderTargetState[0].binStorage.isEnabled = pass_in.blend_enable;
                bl_state.alphaToCoverageEnable = false;
                bl_state.independentBlendEnable = false;

                // Fill in alpha-blending states for unused buffers:
                for (uint8_t i = 1; i < acef::AlphaBlendStateStorage::renderTargetsNum; i++)
                {
                    bl_state.renderTargetState[i].binStorage.src = acef::BlendCoef::kOne;
                    bl_state.renderTargetState[i].binStorage.dst = acef::BlendCoef::kOne;
                    bl_state.renderTargetState[i].binStorage.op = acef::BlendOp::kAdd;
                    bl_state.renderTargetState[i].binStorage.srcAlpha = acef::BlendCoef::kOne;
                    bl_state.renderTargetState[i].binStorage.dstAlpha = acef::BlendCoef::kOne;
                    bl_state.renderTargetState[i].binStorage.opAlpha = acef::BlendOp::kAdd;
                    bl_state.renderTargetState[i].binStorage.renderTargetWriteMask = acef::ColorWriteEnableBits::kAll;
                    bl_state.renderTargetState[i].binStorage.isEnabled = false;
                }
            }

            { // Attach global constant buffer
                pass_out.binStorage.constantBuffersPSNum = 1;
                pass_out.constantBuffersPSSlots = alloc_mem<uint32_t>(1);
                pass_out.constantBuffersPSIndices = alloc_mem<uint32_t>(1);
                pass_out.constantBuffersPSNameLens = alloc_mem<uint16_t>(1);
                pass_out.constantBuffersPSNameOffsets = alloc_mem<uint32_t>(1);

                pass_out.constantBuffersPSSlots[0] = 0; // b0
                pass_out.constantBuffersPSIndices[0] = 0; // See 'colwert_uniforms'
                pass_out.constantBuffersPSNameLens[0] = 0; // Binding via slot, so do not need name
                pass_out.constantBuffersPSNameOffsets[0] = 0;
                pass_out.constantBuffersPSNames = "";

                // Assign same constant buffers to vertex shaders
                pass_out.binStorage.constantBuffersVSNum = pass_out.binStorage.constantBuffersPSNum;
                pass_out.constantBuffersVSSlots = pass_out.constantBuffersPSSlots;
                pass_out.constantBuffersVSIndices = pass_out.constantBuffersPSIndices;
                pass_out.constantBuffersVSNameLens = pass_out.constantBuffersPSNameLens;
                pass_out.constantBuffersVSNameOffsets = pass_out.constantBuffersPSNameOffsets;
                pass_out.constantBuffersVSNames = pass_out.constantBuffersPSNames;
            }

            { // Attach global samplers
                pass_out.samplersSlots = alloc_mem<uint32_t>(module.num_sampler_bindings);
                pass_out.samplersIndices = alloc_mem<uint32_t>(module.num_sampler_bindings);
                pass_out.samplersNameLens = alloc_mem<uint16_t>(module.num_sampler_bindings);
                pass_out.samplersNameOffsets = alloc_mem<uint32_t>(module.num_sampler_bindings);
                pass_out.samplersNames = "";

                std::unordered_set<uint32_t> used_bindings;
                for (const reshadefx::sampler_info &info : module.samplers)
                {
                    if (used_bindings.find(info.binding) != used_bindings.end())
                        continue;
                    used_bindings.insert(info.binding);

                    const uint32_t i = pass_out.binStorage.samplersNum++;
                    pass_out.samplersSlots[i] = info.binding;
                    pass_out.samplersIndices[i] = i; // See 'colwert_samplers'
                    pass_out.samplersNameLens[i] = 0; // Binding via slot, so do not need name
                    pass_out.samplersNameOffsets[i] = 0;
                }
            }

            { // Attach shader resources
                pass_out.readBuffersSlots = alloc_mem<uint32_t>(module.num_texture_bindings);
                pass_out.readBuffersIndices = alloc_mem<uint32_t>(module.num_texture_bindings);
                pass_out.readBuffersNameLens = alloc_mem<uint16_t>(module.num_texture_bindings);
                pass_out.readBuffersNameOffsets = alloc_mem<uint32_t>(module.num_texture_bindings);
                pass_out.readBuffersNames = "";

                for (const reshadefx::sampler_info &info : module.samplers)
                {
                    const uint32_t i = pass_out.binStorage.readBuffersNum++;
                    pass_out.readBuffersSlots[i] = info.texture_binding;
                    pass_out.readBuffersNameLens[i] = 0; // Binding via slot, so do not need name
                    pass_out.readBuffersNameOffsets[i] = 0;

                    const auto existing_texture = std::find_if(module.textures.begin(), module.textures.end(),
                        [&tex_name = info.texture_name](const auto &item) { return item.unique_name == tex_name; });

                    if (last_back_buffer_pass < pass_index && (existing_texture->semantic == "COLOR" || existing_texture->semantic == "SV_TARGET"))
                    {
                        // Replace back buffer input with default intermediate texture
                        // Default targets are stored at index 0 and 1
                        pass_out.readBuffersIndices[i] = (pass_index + 1) % 2;
                    }
                    else
                    {
                        // Index into 'readBufferTextureHandles', see 'colwert_textures'
                        pass_out.readBuffersIndices[i] = read_texture_lookup.at(info.texture_name);
                    }
                }
            }

            { // Attach render targets
                uint32_t num_targets = 1;
                for (uint32_t i = 1; i < 8 && !pass_in.render_target_names[i].empty(); ++i)
                    num_targets = i + 1;

                pass_out.binStorage.writeBuffersNum = num_targets;
                pass_out.writeBuffersSlots = alloc_mem<uint32_t>(num_targets);
                pass_out.writeBuffersIndices = alloc_mem<uint32_t>(num_targets);
                pass_out.writeBuffersNameLens = alloc_mem<uint16_t>(num_targets);
                pass_out.writeBuffersNameOffsets = alloc_mem<uint32_t>(num_targets);
                pass_out.writeBuffersNames = "";

                // Default viewport to back buffer dimensions, unless a texture is used
                pass_out.binStorage.width.binStorage.mul = 1;
                pass_out.binStorage.width.binStorage.texSizeBase = acef::TextureSizeBase::kColorBufferWidth;
                pass_out.binStorage.height.binStorage.mul = 1;
                pass_out.binStorage.height.binStorage.texSizeBase = acef::TextureSizeBase::kColorBufferHeight;

                bool is_back_buffer_pass = true;

                for (uint32_t i = 0; i < num_targets; ++i)
                {
                    if (pass_in.render_target_names[i].empty())
                    {
                        // Default targets are stored at index 0 and 1
                        pass_out.writeBuffersIndices[i] = pass_index % 2;
                    }
                    else
                    {
                        is_back_buffer_pass = false;

                        // Index into 'writeBufferTextureHandles', see 'colwert_textures'
                        pass_out.writeBuffersIndices[i] = write_texture_lookup.at(pass_in.render_target_names[i]);

                        const auto existing_texture = std::find_if(module.textures.begin(), module.textures.end(),
                            [&tex_name = pass_in.render_target_names[i]](const auto &tex) { return tex.unique_name == tex_name; });

                        if (existing_texture->semantic.empty())
                        {
                            // Find texture dimensions of the render target and assign them as viewport
                            uint32_t handle = acef_module.resourceHeader.writeBufferTextureHandles[pass_out.writeBuffersIndices[i]];
                            handle -= acef_module.resourceHeader.binStorage.texturesParametrizedNum;

                            pass_out.binStorage.width = acef_module.texturesIntermediate[handle].binStorage.width;
                            pass_out.binStorage.height = acef_module.texturesIntermediate[handle].binStorage.height;
                        }

                    }

                    pass_out.writeBuffersSlots[i] = i;
                    pass_out.writeBuffersNameLens[i] = 0; // Binding via slot, so do not need name
                    pass_out.writeBuffersNameOffsets[i] = 0;
                }

                // Keep track of the last pass writing to the back buffer
                if (is_back_buffer_pass)
                    last_back_buffer_pass = pass_index;
            }

            { // Attach vertex and pixel shader
                const std::string shader_path = "\\/temp\\/\\" + hlsl_module_path.filename().string();

                pass_out.binStorage.pixelShaderIndex = acef_module.resourceHeader.binStorage.pixelShadersNum++;
                acef::ResourcePixelShader &ps = acef_module.pixelShaders.emplace_back();
                ps.filePathUtf8 = alloc_and_copy_string(shader_path);
                ps.binStorage.filePathLen = static_cast<uint32_t>(shader_path.size());
                ps.entryFunctionAscii = alloc_and_copy_string(pass_in.ps_entry_point);
                ps.binStorage.entryFunctionLen = static_cast<uint32_t>(pass_in.ps_entry_point.size());

                pass_out.binStorage.vertexShaderIndex = acef_module.resourceHeader.binStorage.vertexShadersNum++;
                acef::ResourceVertexShader &vs = acef_module.vertexShaders.emplace_back();
                vs.filePathUtf8 = alloc_and_copy_string(shader_path);
                vs.binStorage.filePathLen = static_cast<uint32_t>(shader_path.size());
                vs.entryFunctionAscii = alloc_and_copy_string(pass_in.vs_entry_point);
                vs.binStorage.entryFunctionLen = static_cast<uint32_t>(pass_in.vs_entry_point.size());
            }

            if (pass_in.srgb_write_enable)
            {
                std::cout << "technique " << technique.name << " pass " << pass_index << " uses unsupported \"SRGBWriteEnable\" value" << std::endl;
                if (!errors_path.empty()) std::ofstream(errors_path, std::ios::app) << "technique " << technique.name << " pass " << pass_index << " uses unsupported \"SRGBWriteEnable\" value" << std::endl;
            }
            if (pass_in.clear_render_targets)
            {
                std::cout << "technique " << technique.name << " pass " << pass_index << " uses unsupported \"ClearRenderTargets\" value" << std::endl;
                if (!errors_path.empty()) std::ofstream(errors_path, std::ios::app) << "technique " << technique.name << " pass " << pass_index << " uses unsupported \"ClearRenderTargets\" value" << std::endl;
            }
            if (pass_in.stencil_enable && pass_in.stencil_reference_value != 0xFFFFFFFF)
            {
                std::cout << "technique " << technique.name << " pass " << pass_index << " uses unsupported \"StencilRef\" value" << std::endl;
                if (!errors_path.empty()) std::ofstream(errors_path, std::ios::app) << "technique " << technique.name << " pass " << pass_index << " uses unsupported \"StencilRef\" value" << std::endl;
            }
            if (pass_in.num_vertices != 3 || pass_in.topology != reshadefx::primitive_topology::triangle_list)
            {
                std::cout << "technique " << technique.name << " pass " << pass_index << " uses unsupported \"PrimitiveTopology\" or \"VertexCount\" value" << std::endl;
                if (!errors_path.empty()) std::ofstream(errors_path, std::ios::app) << "technique " << technique.name << " pass " << pass_index << " uses unsupported \"PrimitiveTopology\" or \"VertexCount\" value" << std::endl;
            }

            ++pass_index;
        }

        if (last_back_buffer_pass < acef_module.passes.size())
            // Last pass writing to the backbuffer is marked by having no write buffers attached
            // TODO: This messes up effects which depend on the output in subsequent passes (e.g. MotionBlur)
            acef_module.passes[last_back_buffer_pass].binStorage.writeBuffersNum = 0;
    }

    inline void replace_oclwrances(std::string &input, size_t offset, size_t end, const std::string &value, const std::string &replacement)
    {
        while ((offset = input.find(value, offset)) != std::string::npos && offset < end)
        {
            input.replace(offset, value.size(), replacement);
            offset += replacement.size();
        }
    }

    inline uint64_t file_timestamp(const std::filesystem::path &path)
    {
        return std::filesystem::last_write_time(path).time_since_epoch().count();
    }
}

int wmain(int argc, wchar_t *argv[])
{
    // Process command line arguments
    // The loop upper bound is "argc - 1" to ensure each command argument is followed by a value argument
    for (int i = 1; i < argc - 1; i++)
    {
        const std::wstring arg = argv[i];
        const std::wstring value = argv[++i];

        if (arg == L"--input-path")
            input_path = std::move(value);
        else if (arg == L"--error-file")
            errors_path = std::move(value);
        else if (arg == L"--output-path")
            output_path = std::move(value);
    }

    if (argc <= 1 || input_path.empty() || output_path.empty() || !std::filesystem::exists(input_path))
    {
        std::cout << "usage: ";
        std::wcout << argv[0];
        std::cout << " <arguments>" << std::endl;
        std::cout << "  --input-path <file>      Full path to the ReShade FX file to compile" << std::endl;
        std::cout << "  --output-path <file>     Full path to the ACEF output file that the compiler should create" << std::endl;
        std::cout << std::endl;
        std::cout << "  --error-file <file>      Full path to the error log file" << std::endl;
        return 1;
    }

    output_path.replace_extension(".acef");

    reshadefx::preprocessor pp;
    if (input_path.has_parent_path())
        pp.add_include_path(input_path.parent_path());
    pp.add_macro_definition("__RESHADE__", "40600");
    pp.add_macro_definition("__RESHADE_FXC__", "1");

    // Add special uniform value which holds the screen dimensions
    pp.append_string("uniform float2 __screensize < source = \"screensize\"; >;\n");
    pp.add_macro_definition("BUFFER_WIDTH", "int(__screensize.x)"); // Cast these to integers to allow bitwise operations
    pp.add_macro_definition("BUFFER_HEIGHT", "int(__screensize.y)");
    pp.add_macro_definition("BUFFER_RCP_WIDTH", "(1.0 / __screensize.x)");
    pp.add_macro_definition("BUFFER_RCP_HEIGHT", "(1.0 / __screensize.y)");
    pp.add_macro_definition("BUFFER_COLOR_DEPTH", "8");

    if (!pp.append_file(input_path))
    {
        const std::string errors = pp.errors();
        std::cout << "failed to preprocess " << input_path << ':' << std::endl << errors;
        if (!errors_path.empty()) std::ofstream(errors_path) << errors;
        return 2;
    }

    std::string &source = pp.output();

    // ReShade expects for the screen dimensions to be known at compile-time to callwlate texture dimensions
    // To get around that, go through the code and find all texture size assignments and negate the ones that are based on the screen dimensions
    // This way this can later be detected when colwerting to ACEF
    for (size_t tex_offset = 0; (tex_offset = source.find("texture", tex_offset)) != std::string::npos; ++tex_offset)
    {
        // Skip oclwrances that are not the keyword
        if (tex_offset + 7 >= source.size() || (source[tex_offset + 7] != ' ' && source[tex_offset + 7] != '\t' && source[tex_offset + 7] != '2' /*texture2D*/))
            continue;

        // Get boundaries of texture definition code block
        const size_t tex_definition_beg = source.find('{', tex_offset);
        const size_t tex_definition_end = source.find("};", tex_offset);
        if (tex_definition_beg == std::string::npos || tex_definition_end == std::string::npos ||
            (std::string_view(source.c_str() + tex_offset, tex_definition_beg - tex_offset).find(';') != std::string::npos && // There can be no semicolons between keyword and block unless there are annotations
                std::string_view(source.c_str() + tex_offset, tex_definition_beg - tex_offset).find('<') == std::string::npos))
            continue;

        // Replace with high negative number, because width/height are integers, so cannot use fractionals and the sign is used to distinguish from normal dimensions
        replace_oclwrances(source, tex_definition_beg, tex_definition_end, "int(__screensize.x)" /*BUFFER_WIDTH*/, "-10000");
        replace_oclwrances(source, tex_definition_beg, tex_definition_end, "int(__screensize.y)" /*BUFFER_HEIGHT*/, "-10000");
    }

    // Parse and compile source code to HLSL SM 4
    reshadefx::parser parser;
    const std::unique_ptr<reshadefx::codegen> codegen(
        reshadefx::create_codegen_hlsl(40, true, false));

    if (!parser.parse(std::move(source), codegen.get()))
    {
        const std::string errors = pp.errors() + parser.errors();
        std::cout << "failed to compile " << input_path << ':' << std::endl << errors;
        if (!errors_path.empty()) std::ofstream(errors_path) << errors;
        return 3;
    }

    codegen->write_result(module);

    if (module.techniques.empty())
    {
        const std::string errors = "effect file does not contain any techniques";
        std::cout << "failed to colwert " << input_path << ':' << std::endl << errors;
        if (!errors_path.empty()) std::ofstream(errors_path) << errors;
        return 4;
    }

    std::cout << "success";
    if (const std::string errors = pp.errors() + parser.errors(); !errors.empty())
    {
        std::cout << " with warnings:" << std::endl << errors;
        if (!errors_path.empty()) std::ofstream(errors_path) << errors;
    }
    else
    {
        std::cout << std::endl;
        if (!errors_path.empty()) std::ofstream(errors_path) << std::string(); // Clear error file contents
    }

    // Write ACEF header information
    auto &file_header = acef_module.header.binStorage;
    file_header.magicWord = compilerMagicWordAndVersion;
    strcpy_s(file_header.hash, "4.6.0");
    strcpy_s(file_header.compiler, "ReShade FX");
    file_header.timestamp = file_timestamp(input_path);

    // Write dependencies (included files)
    const auto included_files = pp.included_files();
    file_header.dependenciesNum = static_cast<uint32_t>(included_files.size());
    acef_module.header.fileTimestamps = alloc_mem<uint64_t>(included_files.size());
    acef_module.header.filePathLens = alloc_mem<uint16_t>(included_files.size());
    acef_module.header.filePathOffsets = alloc_mem<uint32_t>(included_files.size());

    std::string path_data;
    for (size_t i = 0; i < included_files.size(); ++i)
    {
        const std::string file_path = included_files[i].string();
        acef_module.header.fileTimestamps[i] = file_timestamp(included_files[i]);
        acef_module.header.filePathLens[i] = static_cast<uint16_t>(file_path.size());
        acef_module.header.filePathOffsets[i] = static_cast<uint32_t>(path_data.size());

        path_data += file_path;
    }

    acef_module.header.filePathsUtf8 = alloc_and_copy_string(path_data);

    // Dump HLSL module to output file
    hlsl_module_path = input_path.filename();
    hlsl_module_path.replace_extension("hlsl");
    hlsl_module_path = output_path.parent_path() / hlsl_module_path;

    if (std::ofstream file(hlsl_module_path); file.is_open())
        file << module.hlsl;

    // Colwert ReShade FX module to ACEF format
    colwert_textures();
    colwert_samplers();
    colwert_uniforms();

    // Just use the first technique in the effect file for now
    // TODO: Might want an option to choose a specific technique!
    colwert_technique(module.techniques[0]);

    // Write out ACEF to the target file
    acef::calcByteOffsets(&acef_module);
    acef::save(output_path.c_str(), acef_module);

    // Free up all allocated memory
    acef_module.freeAllocatedMem();

    return 0;
}
