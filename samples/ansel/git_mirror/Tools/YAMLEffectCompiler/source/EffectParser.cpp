#include <assert.h>
#include <string>
#include <iostream>
#include <deque>
#include <fstream>
#include <locale>

#include <windows.h> //for LANGID

#include "darkroom/StringColwersion.h"

#include "yaml-cpp/yaml.h"
#include "EffectParser.h"

// DBG DBG TODO: fix logging

//#include "Log.h"

#define LOG_DEBUG(...)
#define LOG_INFO(...)
#define LOG_WARN(...)
#define LOG_ERROR(...)


#define INFO_NODE_NAME "__info"
#define UNBOUNDED_NUMBER_KEYWORD "unbounded"

namespace shadermod
{
    namespace effectParser
    {
        template<typename T>
        struct DecoratedScalar
        {
            DecoratedScalar() : m_value(), m_line(-1), m_col(-1), m_pos(-1)
            {}

            DecoratedScalar(const T& ref) : m_value(ref), m_line(-1), m_col(-1), m_pos(-1)
            {}

            DecoratedScalar(const T& ref, const YAML::Mark& mark) :
                m_value(ref), m_pos(mark.pos), m_line(mark.line), m_col(mark.column)
            {
            }

            YAML::Mark getMark() const
            {
                YAML::Mark ret;
                ret.pos = m_pos, ret.column = m_col, ret.line = m_line;

                return ret;
            }

            std::string encode() const
            {
                YAML::Node n(m_value);
                assert(n.IsScalar());

                std::stringbuf buf;
                std::ostream strstream(&buf);
                strstream << "P" << m_pos << "L" << m_line << "C" << m_col << "N__" << n.Scalar();
                return buf.str();
            }

            bool decode(const std::string& nodeAsStr, T* fallback = nullptr)
            {
                int n = sscanf_s(nodeAsStr.c_str(), "P%dL%dC%d", &m_pos, &m_line, &m_col);

                if (n != 3)
                    return false;

                size_t pos = nodeAsStr.find("N__");

                if (pos == std::string::npos || pos + 3 > nodeAsStr.size())
                    return false;

                YAML::Node nd(nodeAsStr.substr(pos + 3));

                if (fallback)
                {
                    m_value = nd.as<T>(*fallback);

                    return true;
                }
                else
                {
                    try
                    {
                        m_value = nd.as<T>();
                    }
                    catch (...)
                    {
                        return false;
                    }

                    return true;
                }
            }

            T m_value;
            int m_line, m_col, m_pos;
        };
    
        typedef DecoratedScalar<std::string> ExtendedName;
    }
}

namespace YAML
{
    template<typename T>
    struct colwert<shadermod::effectParser::DecoratedScalar<T>>
    {
        static Node encode(const shadermod::effectParser::DecoratedScalar<T>& rhs)
        {
            Node node(rhs.encode());

            return node;
        }

        static bool decode(const Node& node, shadermod::effectParser::DecoratedScalar<T>& rhs)
        {
            if (!node.IsScalar())
            {
                return false;
            }

            std::string nodeAsStr = node.as<std::string>();

            return rhs.decode(nodeAsStr);
        }
    };

    template<typename T>
    YAML::Emitter& operator << (YAML::Emitter& out, const shadermod::effectParser::DecoratedScalar<T>& rhs)
    {
        out << rhs.encode();
        return out;
    }
}

namespace shadermod
{
    namespace effectParser
    {
        ir::FragmentFormat colwertSurfaceTypeToIrFragmentFormat(const SurfaceType& ref, ir::FragmentFormat colorInputChanFmt, ir::FragmentFormat depthInputChanFmt)
        {
            if (colorInputChanFmt == ir::FragmentFormat::kNUM_ENTRIES)
                colorInputChanFmt = ir::FragmentFormat::kBGRA8_uint; //if there are no color inputs to the effect

            if (depthInputChanFmt == ir::FragmentFormat::kNUM_ENTRIES)
                depthInputChanFmt = ir::FragmentFormat::kD24S8; //if there are no depth inputs to the effect

            switch (ref)
            {
            case SurfaceType::kRGBA8_uint:
                return ir::FragmentFormat::kRGBA8_uint;
            case SurfaceType::kBGRA8_uint:
                return ir::FragmentFormat::kBGRA8_uint;
            case SurfaceType::kR10G10B10A2_uint:
                return ir::FragmentFormat::kR10G10B10A2_uint;
            case SurfaceType::kR11G11B10_float:
                return ir::FragmentFormat::kR11G11B10_float;
            case SurfaceType::kRGBA32_fp:
                return ir::FragmentFormat::kRGBA32_fp;
            case SurfaceType::kSRGBA8_uint:
                return ir::FragmentFormat::kSRGBA8_uint;
            case SurfaceType::kSBGRA8_uint:
                return ir::FragmentFormat::kSBGRA8_uint;
            case SurfaceType::kD24S8:
                return ir::FragmentFormat::kD24S8;
            case SurfaceType::kMatchInputColorChannel:
                return  colorInputChanFmt;
            case SurfaceType::kMatchInputDepthChannel:
                return  depthInputChanFmt;
            default:
                return ir::FragmentFormat::kNUM_ENTRIES;
            }
        }

        const char*  SurfaceTypeStr[] = { "Undefined", "RGBA8_uint", "RGBA32_fp", "BGRA8_uint", "R10G10B10A2_uint", "R11G11B10_float", "SRGBA8_uint", "SBGRA8_uint", "D24S8", "match-color-input", "match-depth-input"};
        DECLARE_STRING_TO_ENUM_MAP(SurfaceType, SurfaceTypeStr, SurfaceTypeMap);

        const char*  SystemChannelTypeStr[] = { "PIPE_INPUTS_COLOR", "PIPE_INPUTS_DEPTH", "PIPE_INPUTS_HUDLESS", "PIPE_INPUTS_HDR", "PIPE_INPUTS_COLOR_BASE" };
        DECLARE_STRING_TO_ENUM_MAP(SystemChannelType, SystemChannelTypeStr, SystemChannelTypeMap);
        
        const char*  ShaderPassChannelTypeStr[] =
        {
            "TARGET0_COLOR",
            "TARGET1_COLOR",
            "TARGET2_COLOR",
            "TARGET3_COLOR",
            "TARGET4_COLOR",
            "TARGET5_COLOR",
            "TARGET6_COLOR",
            "TARGET7_COLOR",
        };
        DECLARE_STRING_TO_ENUM_MAP(ShaderPassChannelType, ShaderPassChannelTypeStr, ShaderPassChannelTypeMap);
                
        void colwertSamplerFilterTypeToIrFilterType(const SamplerFilterType& ref,
            ir::FilterType& flt_min, ir::FilterType& flt_mag, ir::FilterType& flt_mip)
        {
            unsigned int refint = (unsigned int)ref;

            bool mipbit = (refint & 0x1) != 0;
            bool magbit = (refint & 0x2) != 0;
            bool minbit = (refint & 0x4) != 0;

            flt_min = minbit ? ir::FilterType::kLinear : ir::FilterType::kPoint;
            flt_mag = magbit ? ir::FilterType::kLinear : ir::FilterType::kPoint;
            flt_mip = mipbit ? ir::FilterType::kLinear : ir::FilterType::kPoint;
        }

        const char* SamplerFilterTypeStr[] = 
        { 
            "MIN_MAG_MIP_POINT", 
            "MIN_MAG_POINT_MIP_LINEAR", 
            "MIN_POINT_MAG_LINEAR_MIP_POINT",
            "MIN_POINT_MAG_MIP_LINEAR",
            "MIN_LINEAR_MAG_MIP_POINT",
            "MIN_LINEAR_MAG_POINT_MIP_LINEAR",
            "MIN_MAG_LINEAR_MIP_POINT",
            "MIN_MAG_MIP_LINEAR"
        };
        DECLARE_STRING_TO_ENUM_MAP(SamplerFilterType, SamplerFilterTypeStr, SamplerFilterTypeMap);
        
        SamplerFilterType combineSubFilterTypes(const SamplerSubFilterType& flt_min,
            const SamplerSubFilterType& flt_mag, const SamplerSubFilterType& flt_mip)
        {
            if (flt_min == SamplerSubFilterType::NUM_ENTRIES || flt_mag == SamplerSubFilterType::NUM_ENTRIES ||
                flt_mip == SamplerSubFilterType::NUM_ENTRIES)
            {
                return SamplerFilterType::NUM_ENTRIES;
            }

            //feodorb: I was tempted to reduce this to 1 line by using math, but that would hinge on the numeric values of SamplerFilterType
            //which would more bug prone. better maintainability vs larger codebase - I chose maintainability. Storing the bits and scrolling through them
            //is far cheaper than debugging things.
                    
            if (flt_min == SamplerSubFilterType::kLINEAR)
            {
                if (flt_mag == SamplerSubFilterType::kLINEAR)
                {
                    if (flt_mip == SamplerSubFilterType::kLINEAR)
                    {
                        return SamplerFilterType::kMinLinMagLinMipLin;
                    }
                    else if (flt_mip == SamplerSubFilterType::kPOINT)
                    {
                        return SamplerFilterType::kMinLinMagLinMipPt;
                    }
                }
                else if (flt_mag == SamplerSubFilterType::kPOINT)
                {
                    if (flt_mip == SamplerSubFilterType::kLINEAR)
                    {
                        return SamplerFilterType::kMinLinMagPtMipLin;
                    }
                    else if (flt_mip == SamplerSubFilterType::kPOINT)
                    {
                        return SamplerFilterType::kMinLinMagPtMipPt;
                    }
                }
            }
            else if (flt_min == SamplerSubFilterType::kPOINT)
            {
                if (flt_mag == SamplerSubFilterType::kLINEAR)
                {
                    if (flt_mip == SamplerSubFilterType::kLINEAR)
                    {
                        return SamplerFilterType::kMinPtMagLinMipLin;
                    }
                    else if (flt_mip == SamplerSubFilterType::kPOINT)
                    {
                        return SamplerFilterType::kMinPtMagLinMipPt;
                    }
                }
                else if (flt_mag == SamplerSubFilterType::kPOINT)
                {
                    if (flt_mip == SamplerSubFilterType::kLINEAR)
                    {
                        return SamplerFilterType::kMinPtMagPtMipLin;
                    }
                    else if (flt_mip == SamplerSubFilterType::kPOINT)
                    {
                        return SamplerFilterType::kMinPtMagPtMipPt;
                    }
                }
            }

            assert(false && "Invalid sampler sub filter type combination!");

            return SamplerFilterType::NUM_ENTRIES;
        }


        const char* SamplerSubFilterTypeStr[] = 
        { 
            "POINT", 
            "LINEAR" 
        };
        DECLARE_STRING_TO_ENUM_MAP(SamplerSubFilterType, SamplerSubFilterTypeStr, SamplerSubFilterTypeMap);
    
        ir::AddressType colwertSamplerAddressTypeToIrSamplerAddressType(const SamplerAddressType& ref)
        {
            switch (ref)
            {
            case SamplerAddressType::kClamp:
                return ir::AddressType::kClamp;
            case SamplerAddressType::kMirror:
                return ir::AddressType::kMirror;
            case SamplerAddressType::kWrap:
                return ir::AddressType::kWrap;
            default:
                assert(false && "Invalid address type!");
                return ir::AddressType::kWrap;
            }
        }

        const char* SamplerAddressTypeStr[] =
        {
            "WRAP",
            "CLAMP",
            "MIRROR"
        };
        DECLARE_STRING_TO_ENUM_MAP(SamplerAddressType, SamplerAddressTypeStr, SamplerAddressTypeMap);
        
        
        ir::ConstType colwertSystemConstantsTypeToIrConstType(const SystemConstantsType& ref)
        {
            switch (ref)
            {
            case SystemConstantsType::kDT:
                return ir::ConstType::kDT;
            case SystemConstantsType::kElapsedTime:
                return ir::ConstType::kElapsedTime;
            case SystemConstantsType::kFrame:
                return ir::ConstType::kFrame;
            case SystemConstantsType::kScreenSize:
                return ir::ConstType::kScreenSize;
            case SystemConstantsType::kCaptureState:
                return ir::ConstType::kCaptureState;
            case SystemConstantsType::kTileUV:
                return ir::ConstType::kTileUV;
            case SystemConstantsType::kDepthAvailable:
                return ir::ConstType::kDepthAvailable;
            case SystemConstantsType::kHDRAvailable:
                return ir::ConstType::kHDRAvailable;
            case SystemConstantsType::kHUDlessAvailable:
                return ir::ConstType::kHUDlessAvailable;
            default:
                assert(false && "Invalid system constant!");
                return ir::ConstType::kNUM_ENTRIES;
            }
        }

        const char* SystemConstantsTypeStr[] =
        {
            "DT",
            "ELAPSED_TIME",
            "FRAME",
            "SCREEN_SIZE",
            "CAPTURE_STATE",
            "TILE_UV_RANGE",
            "BUFREADY_DEPTH",
            "BUFREADY_HDR",
            "BUFREADY_HUDLESS"
        };
        DECLARE_STRING_TO_ENUM_MAP(SystemConstantsType, SystemConstantsTypeStr, SystemConstantsTypeMap);

    
        const char* UserConstOptionalLanguageTypeStr[] =
        {
            "de-DE",
            "es-ES",
            "es-MX",
            "fr-FR",
            "it-IT",
            "ru-RU",
            "zh-CHS",
            "zh-CHT",
            "ja-JP",
            "cs-CZ",
            "da-DK",
            "el-GR",
            "en-UK",
            "fi-FI",
            "hu",
            "ko-KR",
            "nl-NL",
            "nb-NO",
            "pl",
            "pt-PT",
            "pt-BR",
            "sl-SI",
            "sk-SK",
            "sv-SE",
            "th-TH",
            "tr-TR"
        };
        DECLARE_STRING_TO_ENUM_MAP(UserConstOptionalLanguageType, UserConstOptionalLanguageTypeStr, UserConstOptionalLanguageTypeMap);
            
        unsigned short colwertUserConstOptionalLanguageTypeToLANGID(const UserConstOptionalLanguageType& ref)
        {
            switch (ref)
            {
            case UserConstOptionalLanguageType::k_de_DE:
                return MAKELANGID(LANG_GERMAN, SUBLANG_GERMAN);
            case UserConstOptionalLanguageType::k_es_ES:
                return MAKELANGID(LANG_SPANISH, SUBLANG_SPANISH_MODERN);
            case UserConstOptionalLanguageType::k_es_MX:
                return MAKELANGID(LANG_SPANISH, SUBLANG_SPANISH_MEXICAN);
            case UserConstOptionalLanguageType::k_fr_FR:
                return MAKELANGID(LANG_FRENCH, SUBLANG_FRENCH);
            case UserConstOptionalLanguageType::k_it_IT:
                return MAKELANGID(LANG_ITALIAN, SUBLANG_ITALIAN);
            case UserConstOptionalLanguageType::k_ru_RU:
                return MAKELANGID(LANG_RUSSIAN, SUBLANG_RUSSIAN_RUSSIA);
            case UserConstOptionalLanguageType::k_zh_CHS:
                return MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_SIMPLIFIED);
            case UserConstOptionalLanguageType::k_zh_CHT:
                return MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_TRADITIONAL);
            case UserConstOptionalLanguageType::k_ja_JP:
                return MAKELANGID(LANG_JAPANESE, SUBLANG_JAPANESE_JAPAN);
            case UserConstOptionalLanguageType::k_cs_CZ:
                return MAKELANGID(LANG_CZECH, SUBLANG_CZECH_CZECH_REPUBLIC);
            case UserConstOptionalLanguageType::k_da_DK:
                return MAKELANGID(LANG_DANISH, SUBLANG_DANISH_DENMARK);
            case UserConstOptionalLanguageType::k_el_GR:
                return MAKELANGID(LANG_GREEK, SUBLANG_GREEK_GREECE);
            case UserConstOptionalLanguageType::k_en_UK:
                return MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_UK);
            case UserConstOptionalLanguageType::k_fi_FI:
                return MAKELANGID(LANG_FINNISH, SUBLANG_FINNISH_FINLAND);
            case UserConstOptionalLanguageType::k_hu:
                return MAKELANGID(LANG_HUNGARIAN, SUBLANG_HUNGARIAN_HUNGARY);
            case UserConstOptionalLanguageType::k_ko_KR:
                return MAKELANGID(LANG_KOREAN, SUBLANG_KOREAN);
            case UserConstOptionalLanguageType::k_nl_NL:
                return MAKELANGID(LANG_DUTCH, SUBLANG_DUTCH);
            case UserConstOptionalLanguageType::k_nb_NO:
                return MAKELANGID(LANG_NORWEGIAN, SUBLANG_NORWEGIAN_BOKMAL);
            case UserConstOptionalLanguageType::k_pl:
                return MAKELANGID(LANG_POLISH, SUBLANG_POLISH_POLAND);
            case UserConstOptionalLanguageType::k_pt_PT:
                return MAKELANGID(LANG_PORTUGUESE, SUBLANG_PORTUGUESE);
            case UserConstOptionalLanguageType::k_pt_BR:
                return MAKELANGID(LANG_PORTUGUESE, SUBLANG_PORTUGUESE_BRAZILIAN);
            case UserConstOptionalLanguageType::k_sl_SI:
                return MAKELANGID(LANG_SLOVENIAN, SUBLANG_SLOVENIAN_SLOVENIA);
            case UserConstOptionalLanguageType::k_sk_SK:
                return MAKELANGID(LANG_SLOVAK, SUBLANG_SLOVAK_SLOVAKIA);
            case UserConstOptionalLanguageType::k_sv_SE:
                return MAKELANGID(LANG_SWEDISH, SUBLANG_SWEDISH);
            case UserConstOptionalLanguageType::k_th_TH:
                return MAKELANGID(LANG_THAI, SUBLANG_THAI_THAILAND);
            case UserConstOptionalLanguageType::k_tr_TR:
                return MAKELANGID(LANG_TURKISH, SUBLANG_TURKISH_TURKEY);
            default:
                return (unsigned short) -1;
            }
        }

        bool SamplerDescWrapperForSet::operator<(const SamplerDescWrapperForSet& ref) const
        {
            assert(&m_primaryStorage == &(ref.m_primaryStorage));
            const SamplerDesc& left = m_primaryStorage[m_index];
            const SamplerDesc& right = m_primaryStorage[ref.m_index];
            
            if (left.m_adressU < right.m_adressU)
                return true;
            else if (right.m_adressU < left.m_adressU)
                return false;

            if (left.m_adressV < right.m_adressV)
                return true;
            else if (right.m_adressV < left.m_adressV)
                return false;

            if (left.m_filter < right.m_filter)
                return true;
            else if (right.m_filter < left.m_filter)
                return false;

            return false;
        }

        bool CBufferConstantBindingDesc::BindingKey::operator<(const CBufferConstantBindingDesc::BindingKey& ref) const
        {
            if (m_subcomponentIdx < ref.m_subcomponentIdx)
                return true;
            else if (ref.m_subcomponentIdx < m_subcomponentIdx)
                return false;

            if (m_componentIdx < ref.m_componentIdx)
                return true;
            else if (ref.m_componentIdx < m_componentIdx)
                return false;

            if (m_name < ref.m_name)
                return true;
            else if (ref.m_name < m_name)
                return false;

            return false;
        }

        bool CBufferConstantBindingDesc::operator<(const CBufferConstantBindingDesc& ref) const
        {
            if (m_variableIdx < ref.m_variableIdx)
                return true;
            else if (ref.m_variableIdx < m_variableIdx)
                return false;

            if (m_userConstName < ref.m_userConstName)
                return true;
            else if (ref.m_userConstName < m_userConstName)
                return false;
            
            return false;
        }

        bool CBufferDescWrapperForSet::operator<(const CBufferDescWrapperForSet& ref) const
        {
            assert(&m_primaryStorage == &(ref.m_primaryStorage));

            const CBufferDesc& left = m_primaryStorage[m_index];
            const CBufferDesc& right = m_primaryStorage[ref.m_index];

            return left.m_variableBindings < right.m_variableBindings;
        }
        
        bool InputTextureDatasourceWrapperForSet::operator<(const InputTextureDatasourceWrapperForSet& ref) const
        {
            assert(&m_primaryStorage == &(ref.m_primaryStorage));
            const InputTextureDatasource& left = m_primaryStorage[m_index];
            const InputTextureDatasource& right = m_primaryStorage[ref.m_index];


            if (left.m_filename < right.m_filename)
                return true;
            else if (right.m_filename < left.m_filename)
                return false;

            if (left.m_type < right.m_type)
                return true;
            else if (right.m_type < left.m_type)
                return false;

            if (left.m_width < right.m_width)
                return true;
            else if (right.m_width < left.m_width)
                return false;

            if (left.m_height < right.m_height)
                return true;
            else if (right.m_height < left.m_height)
                return false;

            return false;
        }

        const char*  ProceduralTextureTypeStr[] =
        {
            "NOISE"
        };
        DECLARE_STRING_TO_ENUM_MAP(ProceduralTextureType, ProceduralTextureTypeStr, ProceduralTextureTypeMap);

        bool ProceduralTextureDatasourceWrapperForSet::operator<(const ProceduralTextureDatasourceWrapperForSet& ref) const
        {
            assert(&m_primaryStorage == &(ref.m_primaryStorage));
            const ProceduralTextureDatasource& left = m_primaryStorage[m_index];
            const ProceduralTextureDatasource& right = m_primaryStorage[ref.m_index];


            if (left.m_procedureType < right.m_procedureType)
                return true;
            else if (right.m_procedureType < left.m_procedureType)
                return false;

            if (left.m_surfaceType < right.m_surfaceType)
                return true;
            else if (right.m_surfaceType < left.m_surfaceType)
                return false;

            if (left.m_width < right.m_width)
                return true;
            else if (right.m_width < left.m_width)
                return false;

            if (left.m_height < right.m_height)
                return true;
            else if (right.m_height < left.m_height)
                return false;

            return false;
        }

                        
        void SystemTextureDatasource::referenceChannel(const SystemChannelType& channel)
        {
            static_assert(sizeof(size_t) * 8 >= (int)SystemChannelType::NUM_ENTRIES, "size_t is not large enough to accomodate a bitmap for every possible SystemChannelType");

            m_channelsReferenced |= (1ull << (size_t)channel);
        }

        bool SystemTextureDatasource::isChannelReferenced(const SystemChannelType& channel) const
        {
            return 	(m_channelsReferenced & (1ull << (size_t)channel)) != 0;
        }

        bool ShaderPassDatasource::BindingKey::operator<(const ShaderPassDatasource::BindingKey& ref) const
        {
            if (m_slotIdx < ref.m_slotIdx)
                return true;
            else if (ref.m_slotIdx < m_slotIdx)
                return false;

            if (m_name < ref.m_name)
                return true;
            else if (ref.m_name < m_name)
                return false;

            return false;
        }
                
        void ShaderPassDatasource::referenceChannel(const ShaderPassChannelType& channel, const SurfaceType& surfType)
        {
            static_assert(sizeof(unsigned int) * 8 >= (int) ShaderPassChannelType::NUM_ENTRIES, "dword is not large enough to accomodate a bitmap for every possible ShaderPassChannelType");
        
            m_channelsReferenced |= (1ull << (unsigned int)channel);

            if (m_typesReferenced.size() <= (unsigned int)channel)
            {
                m_typesReferenced.resize((unsigned int)channel + 1, SurfaceType::kUndefined);
            }

            m_typesReferenced[(unsigned int)channel] = surfType;
        }
    
        ShaderPassDatasource::TextureInput::TextureInput() : m_type(eUndefined) {}
        ShaderPassDatasource::TextureInput::TextureInput(const SystemInput& ref) : m_systemInput(ref), m_type(eSystemInput) {}
        ShaderPassDatasource::TextureInput::TextureInput(const ProceduralTextureInput& ref) : m_proceduralTextureInput(ref), m_type(eProceduralTextureInput) {}
        ShaderPassDatasource::TextureInput::TextureInput(const UserTextureInput& ref) : m_textureInput(ref), m_type(eUserTextureInput) {}
        ShaderPassDatasource::TextureInput::TextureInput(const ShaderPassInput& ref) : m_shaderPassInput(ref), m_type(eShaderPassInput) {}

        const ShaderPassDatasource::SystemInput* ShaderPassDatasource::TextureInput::getSystemInput() const
        {
            if (m_type == eSystemInput)
                return &m_systemInput;
            else
                return nullptr;
        }

        const ShaderPassDatasource::UserTextureInput* ShaderPassDatasource::TextureInput::getUserTextureInput() const
        {
            if (m_type == eUserTextureInput)
                return &m_textureInput;
            else
                return nullptr;
        }

        const ShaderPassDatasource::ProceduralTextureInput* ShaderPassDatasource::TextureInput::getProceduralTextureInput() const
        {
            if (m_type == eProceduralTextureInput)
                return &m_proceduralTextureInput;
            else
                return nullptr;
        }

        const ShaderPassDatasource::ShaderPassInput* ShaderPassDatasource::TextureInput::getShaderPassInput() const
        {
            if (m_type == eShaderPassInput)
                return &m_shaderPassInput;
            else
                return nullptr;
        }
    

        bool  ShaderPassDatasource::TextureInput::operator<(const ShaderPassDatasource::TextureInput& b) const
        {
            const ShaderPassDatasource::TextureInput& a = *this;

            if (a.m_type < b.m_type)
                return true;
            else if (b.m_type < a.m_type)
                return false;

            switch (a.m_type)
            {
            case ShaderPassDatasource::TextureInput::eSystemInput:
                return compareSystemInput(*(a.getSystemInput()), *(b.getSystemInput()));
            case ShaderPassDatasource::TextureInput::eProceduralTextureInput:
                return compareProceduralTextureInput(*(a.getProceduralTextureInput()), *(b.getProceduralTextureInput()));
            case ShaderPassDatasource::TextureInput::eUserTextureInput:
                return compareUserTextureInput(*(a.getUserTextureInput()), *(b.getUserTextureInput()));
            case ShaderPassDatasource::TextureInput::eShaderPassInput:
                return compareShaderPassInput(*(a.getShaderPassInput()), *(b.getShaderPassInput()));
            default:
                return false;
            }

            return false;
        }

        bool ShaderPassDatasource::TextureInput::compareSystemInput(const ShaderPassDatasource::SystemInput& a, const ShaderPassDatasource::SystemInput& b) const
        {
            if (a.m_channel < b.m_channel)
                return true;
            else if (b.m_channel < a.m_channel)
                return false;

            return false;
        }

        bool ShaderPassDatasource::TextureInput::compareProceduralTextureInput(const ShaderPassDatasource::ProceduralTextureInput& a, const ShaderPassDatasource::ProceduralTextureInput& b) const
        {
            if (a.m_procTexDatasource < b.m_procTexDatasource)
                return true;
            else if (b.m_procTexDatasource < a.m_procTexDatasource)
                return false;

            return false;
        }

        bool ShaderPassDatasource::TextureInput::compareUserTextureInput(const ShaderPassDatasource::UserTextureInput& a, const ShaderPassDatasource::UserTextureInput& b) const
        {
            if (a.m_textureDatasource < b.m_textureDatasource)
                return true;
            else if (b.m_textureDatasource < a.m_textureDatasource)
                return false;

            return false;
        }

        bool ShaderPassDatasource::TextureInput::compareShaderPassInput(const ShaderPassDatasource::ShaderPassInput& a, const ShaderPassDatasource::ShaderPassInput& b) const
        {
            if (a.m_shaderPassDatasource < b.m_shaderPassDatasource)
                return true;
            else if (b.m_shaderPassDatasource < a.m_shaderPassDatasource)
                return false;

            if (a.m_channel < b.m_channel)
                return true;
            else if (b.m_channel < a.m_channel)
                return false;

            return false;
        }
        
        bool ShaderPassDatasourceWrapperForSet::operator<(const ShaderPassDatasourceWrapperForSet& ref) const
        {
            assert(&m_primaryStorage == &(ref.m_primaryStorage));
            const ShaderPassDatasource& left = m_primaryStorage[m_index];
            const ShaderPassDatasource& right = m_primaryStorage[ref.m_index];

            if (left.m_filename < right.m_filename)
                return true;
            else if (right.m_filename < left.m_filename)
                return false;

            if (left.m_shadername < right.m_shadername)
                return true;
            else if (right.m_shadername < left.m_shadername)
                return false;
                    
            if (left.m_width < right.m_width)
                return true;
            else if (right.m_width < left.m_width)
                return false;

            if (left.m_height < right.m_height)
                return true;
            else if (right.m_height < left.m_height)
                return false;

            if (left.m_scaleWidth < right.m_scaleWidth)
                return true;
            else if (right.m_scaleWidth < left.m_scaleWidth)
                return false;

            if (left.m_scaleHeight < right.m_scaleHeight)
                return true;
            else if (right.m_scaleHeight < left.m_scaleHeight)
                return false;

            if (left.m_samplers < right.m_samplers)
                return true;
            else if (right.m_samplers < left.m_samplers)
                return false;

            if (left.m_cbuffers < right.m_cbuffers)
                return true;
            else if (right.m_cbuffers < left.m_cbuffers)
                return false;

            //only datasources which have differnet surface types on same channels are different
            unsigned int crossChannelsReferenced = left.m_channelsReferenced & right.m_channelsReferenced;

            for (unsigned int v = crossChannelsReferenced; v; v &= v - 1)
            {
                unsigned long bitidx;
                int res = _BitScanForward(&bitidx, v);
                assert(res);

                SurfaceType leftType = left.m_typesReferenced[bitidx];
                SurfaceType rightType = right.m_typesReferenced[bitidx];

                if (leftType < rightType)
                    return true;
                else if (rightType < leftType)
                    return false;
            }
            
            return left.m_textures < right.m_textures;
        }
            
        template<typename DatasourceT>
        MultipassConfigParserError EffectParser::readWidthHeight(const YAML::Node& datasourceNode, DatasourceT& datasrc)
        {
            YAML::Node width = datasourceNode["width"];

            datasrc.m_width = 0;

            if (width)
            {
                if (!width.IsScalar())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(width));

                std::string str = width.as<ExtendedName>().m_value;
                darkroom::tolowerInplace(str);

                if (str == "default")
                {
                    datasrc.m_width = 0;
                }
                else
                {
                    int w = width.as<DecoratedScalar<int> >(0).m_value;

                    if (w <= 0)
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidPositiveInt, getMarkFromNode(width));

                    datasrc.m_width = w;
                }
            }

            YAML::Node height = datasourceNode["height"];

            datasrc.m_height = 0;

            if (height)
            {
                if (!height.IsScalar())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(height));

                std::string str = height.as<ExtendedName>().m_value;
                darkroom::tolowerInplace(str);

                if (str == "default")
                {
                    datasrc.m_height = 0;
                }
                else
                {
                    int h = height.as<DecoratedScalar<int> >(0).m_value;

                    if (h <= 0)
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidPositiveInt, getMarkFromNode(height));

                    datasrc.m_height = h;
                }
            }

            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
        }

        MultipassConfigParserError EffectParser::processSamplerState(const YAML::Node& samplerNode, size_t& out)
        {
            if (!samplerNode.IsMap())
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, getMarkFromNode(samplerNode));
            }

            SamplerDesc desc;

            YAML::Node addressU = samplerNode["addressU"];

            desc.m_adressU = SamplerAddressType::NUM_ENTRIES;

            if (addressU)
            {
                if (!addressU.IsScalar())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(addressU));

                desc.m_adressU = SamplerAddressTypeMap::colwert(addressU.as<ExtendedName>().m_value, SamplerAddressType::NUM_ENTRIES);

                if (desc.m_adressU == SamplerAddressType::NUM_ENTRIES)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(addressU));
            }
            else
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eRequiredPropertyMissing, "addressU", getMarkFromNode(samplerNode));
            }

            YAML::Node addressV = samplerNode["addressV"];

            desc.m_adressV = SamplerAddressType::NUM_ENTRIES;

            if (addressV)
            {
                if (!addressV.IsScalar())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(addressV));

                desc.m_adressV = SamplerAddressTypeMap::colwert(addressV.as<ExtendedName>().m_value, SamplerAddressType::NUM_ENTRIES);

                if (desc.m_adressV == SamplerAddressType::NUM_ENTRIES)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(addressV));
            }
            else
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eRequiredPropertyMissing, "addressV", getMarkFromNode(samplerNode));
            }

            YAML::Node filter = samplerNode["filter"];
            YAML::Node filterMin = samplerNode["filterMin"];
            YAML::Node filterMag = samplerNode["filterMag"];
            YAML::Node filterMip = samplerNode["filterMip"];
            
            desc.m_filter = SamplerFilterType::NUM_ENTRIES;

            if (filter)
            {
                //feodorb: we don't allow using both ways of setting a filter in one place
                if (filterMin)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, getMarkFromNode(filterMin));

                if (filterMag)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, getMarkFromNode(filterMag));

                if (filterMip)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, getMarkFromNode(filterMip));
                
                if (!filter.IsScalar())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(filter));

                desc.m_filter = SamplerFilterTypeMap::colwert(filter.as<ExtendedName>().m_value, SamplerFilterType::NUM_ENTRIES);

                if (desc.m_filter == SamplerFilterType::NUM_ENTRIES)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(filter));
            }
            else if (filterMin || filterMag || filterMip)
            {
                //feodorb: thois is for good error reporting: is one is set, we help the user to set the split filter settings propertly
                //if none are set, we suggest using the combined filter field
                if (!filterMin)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eRequiredPropertyMissing, "filterMin", getMarkFromNode(samplerNode));

                if (!filterMag)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eRequiredPropertyMissing, "filterMag", getMarkFromNode(samplerNode));

                if (!filterMip)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eRequiredPropertyMissing, "filterMip", getMarkFromNode(samplerNode));

                if (!filterMin.IsScalar())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(filterMin));
            
                if (!filterMag.IsScalar())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(filterMag));

                if (!filterMip.IsScalar())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(filterMip));
                
                SamplerSubFilterType fltMin = SamplerSubFilterTypeMap::colwert(filterMin.as<ExtendedName>().m_value, SamplerSubFilterType::NUM_ENTRIES);
                
                if (fltMin == SamplerSubFilterType::NUM_ENTRIES)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(filterMin));

                SamplerSubFilterType fltMag = SamplerSubFilterTypeMap::colwert(filterMag.as<ExtendedName>().m_value, SamplerSubFilterType::NUM_ENTRIES);

                if (fltMag == SamplerSubFilterType::NUM_ENTRIES)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(filterMag));

                SamplerSubFilterType fltMip = SamplerSubFilterTypeMap::colwert(filterMip.as<ExtendedName>().m_value, SamplerSubFilterType::NUM_ENTRIES);

                if (fltMip == SamplerSubFilterType::NUM_ENTRIES)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(filterMip));

                desc.m_filter = combineSubFilterTypes(fltMin, fltMag, fltMip);

                if (desc.m_filter == SamplerFilterType::NUM_ENTRIES)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eInternalError, "Filter type colwersion failed!", getMarkFromNode(samplerNode));
            }
            else
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eRequiredPropertyMissing, "filter", getMarkFromNode(samplerNode));
            }

            auto iter = insertSamplerDesc(desc);

            out = iter.first;

            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
        }

        MultipassConfigParserError EffectParser::processCBuffer(const YAML::Node& cbufferNode, size_t& out)
        {
            if (!cbufferNode.IsMap())
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, getMarkFromNode(cbufferNode));
            }

            CBufferDesc bufdesc;

            for (YAML::const_iterator it = cbufferNode.begin(); it != cbufferNode.end(); ++it)
            {
                if (!it->first.IsScalar())
                {
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidProperty, getMarkFromNode(it->first));
                }

                if (it->first.Scalar() == INFO_NODE_NAME)
                    continue;

                CBufferConstantBindingDesc desc;

                if (!it->second.IsScalar())
                {
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidProperty, getMarkFromNode(it->second));
                }

                std::string constsname = it->second.as<ExtendedName>().m_value;

                SystemConstantsType sysconst = SystemConstantsTypeMap::colwert(constsname, SystemConstantsType::NUM_ENTRIES);

                desc.m_variableIdx = sysconst;

                if (sysconst == SystemConstantsType::NUM_ENTRIES)
                {
                    desc.m_userConstName = constsname;
                }
                
                std::string constslot = it->first.as<std::string>();

                if (constslot.length() < 1)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, constslot);
                            
                if (constslot[0] == '$')
                {
                    if (constslot.length() < 3 || constslot[1] != 'c')
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, constslot);

                    size_t dotpos = constslot.find('.');

                    CBufferPackOffsetSubcomponentType subcomponentIdx = CBufferPackOffsetSubcomponentType::eX;

                    if (dotpos != std::string::npos)
                    {
                        if (constslot.length() <= (dotpos + 1) || dotpos < 3)
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, constslot);

                        switch (constslot[dotpos + 1])
                        {
                        case  'x':
                        case  'X':
                            subcomponentIdx = CBufferPackOffsetSubcomponentType::eX;
                            break;

                        case  'y':
                        case  'Y':
                            subcomponentIdx = CBufferPackOffsetSubcomponentType::eY;
                            break;

                        case  'z':
                        case  'Z':
                            subcomponentIdx = CBufferPackOffsetSubcomponentType::eZ;
                            break;

                        case  'w':
                        case  'W':
                            subcomponentIdx = CBufferPackOffsetSubcomponentType::eW;
                            break;

                        default:
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, constslot);
                        }

                        constslot = constslot.substr(0, dotpos);
                    }

                    int componentIndex;

                    try
                    {
                        componentIndex = std::stoi(constslot.substr(2));
                    }
                    catch (const std::ilwalid_argument& ref)
                    {
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, constslot + " " + ref.what());
                    }

                    if (componentIndex < 0 || componentIndex >= MAX_CONSTS_IN_BUFFER)
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIndexOutOfRange, constslot);

                    auto r = bufdesc.m_variableBindings.insert(
                            std::make_pair(CBufferConstantBindingDesc::BindingKey(componentIndex, subcomponentIdx), desc));
                    
                    if (!r.second)
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eDuplicateBindingSlot, constslot);
                }
                else
                {
                    auto r = bufdesc.m_variableBindings.insert(std::make_pair(CBufferConstantBindingDesc::BindingKey(constslot), desc));

                    if (!r.second)
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eDuplicateBindingSlot, constslot);
                }
            }
        
            auto cbit = insertCBufferDesc(bufdesc);

            out = cbit.first;

            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
        }
            
        YAML::Mark EffectParser::getMarkFromNode(const YAML::Node& node)
        {
            ExtendedName exname;

            switch (node.Type())
            {
                case YAML::NodeType::Map:
                {
                        YAML::Node sysinfo = node[INFO_NODE_NAME];
                        assert(sysinfo && sysinfo.IsScalar());
                        exname.decode(sysinfo.Scalar());
                        break;
                }

                case YAML::NodeType::Sequence:
                {
                         YAML::Node sysinfo = node[0];
                         assert(sysinfo && sysinfo.IsScalar());
                         exname.decode(sysinfo.Scalar());
                         break;
                }
                case YAML::NodeType::Scalar:
                    exname.decode(node.Scalar());
                    break;

                }

            return exname.getMark();
        }

        MultipassConfigParserError EffectParser::processDatasource(const YAML::Node& mainDatasourceNode, ShaderPassDatasource::TextureInput& mainOut)
        {
            enum EDatasourceType { eShader, eFilename, eProcedural, eSystem };

            struct DatasourceProcessorInOut
            {
                YAML::Node								datasourceNode;//in
                EDatasourceType							type;
                ShaderPassDatasource::TextureInput		output;//out
                int										parentIndex;
                int										firstChildIndex;
                int										nextSiblingIndex;
            };

            std::vector<DatasourceProcessorInOut> localStack;
            int lwrrNode = 0;

            //depth-first construction

            DatasourceProcessorInOut newEl;
            newEl.datasourceNode = mainDatasourceNode;
            newEl.parentIndex = -1;
            newEl.nextSiblingIndex = -1;
            newEl.firstChildIndex = -1;
            localStack.push_back(newEl);

            while (lwrrNode < (int) localStack.size())
            {
                DatasourceProcessorInOut& lwrrElem = localStack[lwrrNode];

                if (!lwrrElem.datasourceNode.IsMap())
                {
                    if (newEl.parentIndex == -1)
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap,"main");
                    else
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, getMarkFromNode(lwrrElem.datasourceNode));
                }

                lwrrElem.type = eSystem;

                if (lwrrElem.datasourceNode["shader"])
                    lwrrElem.type = eShader;
                else if (lwrrElem.datasourceNode["filename"])
                    lwrrElem.type = eFilename;
                else if (lwrrElem.datasourceNode["procedure"])
                    lwrrElem.type = eProcedural;

                if (lwrrElem.type == eShader)
                {
                    if (lwrrElem.datasourceNode["textures"])
                    {
                        YAML::Node textures = lwrrElem.datasourceNode["textures"];

                        if (!textures.IsNull())
                        {
                            if (!textures.IsMap())
                            {
                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, getMarkFromNode(textures));
                            }

                            YAML::const_iterator it2 = textures.begin(), end2 = textures.end();

                            
                            size_t sizeBefore = localStack[lwrrNode].firstChildIndex = int(localStack.size());
                            
                            bool added = false;

                            for (; it2 != end2; ++it2)
                            {
                                if (it2->first.IsScalar() && it2->first.Scalar() == std::string(INFO_NODE_NAME))
                                    continue;

                                if (it2->second.IsNull())
                                    continue;

                                DatasourceProcessorInOut newEl;
                                newEl.datasourceNode = it2->second;
                                newEl.parentIndex = lwrrNode;
                                newEl.nextSiblingIndex = int(localStack.size() + 1);
                                newEl.firstChildIndex = -1;
                                localStack.push_back(newEl);
                                
                                added = true;
                            }

                            if (added)
                            {
                                localStack[lwrrNode].firstChildIndex = int(sizeBefore);
                                localStack.back().nextSiblingIndex = -1;
                            }
                        }
                    }
                }

                ++lwrrNode;
            }

            lwrrNode = 0;

            while (localStack[lwrrNode].firstChildIndex != -1)
                lwrrNode = localStack[lwrrNode].firstChildIndex;

            while (lwrrNode >= 0)
            {
                DatasourceProcessorInOut& lwrrElem = localStack[lwrrNode];

                if (lwrrElem.type == eShader)
                {
                    ShaderPassDatasource datasrc;

                    if (lwrrElem.datasourceNode["textures"])
                    {
                        YAML::Node textures = lwrrElem.datasourceNode["textures"];

                        if (!textures.IsNull())
                        {
                            if (!textures.IsMap())
                            {
                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, getMarkFromNode(textures));
                            }

                            int chldIdx = localStack[lwrrNode].firstChildIndex;
                            assert(chldIdx != -1);

                            for (YAML::const_iterator it2 = textures.begin(); it2 != textures.end(); ++it2)
                            {
                                if (it2->first.Scalar() == INFO_NODE_NAME)
                                    continue;

                                assert(chldIdx != -1);

                                ShaderPassDatasource::TextureInput& input = localStack[chldIdx].output;

                                if (!it2->first.IsScalar())
                                {
                                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidProperty, getMarkFromNode(it2->first));
                                }
                                
                                std::string texslot = it2->first.as<std::string>();

                                if (texslot.length() < 1)
                                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, texslot);

                                if (texslot[0] == '$')
                                {
                                    if (texslot.length() < 3 || texslot[1] != 't')
                                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, texslot);

                                    int idx;

                                    try
                                    {
                                        idx = std::stoi(texslot.substr(2));
                                    }
                                    catch (const std::ilwalid_argument& ref)
                                    {
                                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, texslot + " " + ref.what());
                                    }

                                    if (idx < 0 || idx >= MAX_TEXTURES)
                                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIndexOutOfRange, texslot);

                                    auto r = datasrc.m_textures.insert(std::make_pair(ShaderPassDatasource::BindingKey(idx), input));

                                    if (!r.second)
                                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eDuplicateBindingSlot, texslot);
                                }
                                else
                                {
                                    auto r = datasrc.m_textures.insert(std::make_pair(ShaderPassDatasource::BindingKey(texslot), input));

                                    if (!r.second)
                                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eDuplicateBindingSlot, texslot);
                                }
                            
                                chldIdx = localStack[chldIdx].nextSiblingIndex;
                            }
                        }
                    }

                    YAML::Node shaderNode = lwrrElem.datasourceNode["shader"];

                    datasrc.m_shadername.clear();
                    datasrc.m_filename.clear();

                    if (!shaderNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(shaderNode));

                    std::string decoratedName = shaderNode.as<ExtendedName>().m_value;
                    auto pos = decoratedName.find('@');

                    if (pos == std::string::npos)
                    {
                        YAML::Node filenameNode = lwrrElem.datasourceNode["filename"];

                        if (!filenameNode)
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eRequiredPropertyMissing, "filename", getMarkFromNode(lwrrElem.datasourceNode));

                        if (!filenameNode.IsScalar())
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(filenameNode));

                        datasrc.m_shadername = decoratedName;
                        datasrc.m_filename = filenameNode.as<ExtendedName>().m_value;
                    }
                    else
                    {
                        datasrc.m_shadername = decoratedName.substr(0, pos);
                        datasrc.m_filename = decoratedName.substr(pos + 1, std::string::npos);
                    }

                    if (lwrrElem.datasourceNode["constant-buffers"])
                    {
                        YAML::Node cbuffers = lwrrElem.datasourceNode["constant-buffers"];

                        if (!cbuffers.IsNull())
                        {
                            if (!cbuffers.IsMap())
                            {
                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, getMarkFromNode(cbuffers));
                            }

                            for (YAML::const_iterator it2 = cbuffers.begin(); it2 != cbuffers.end(); ++it2)
                            {
                                if (!it2->first.IsScalar())
                                {
                                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidProperty, getMarkFromNode(it2->first));
                                }

                                if (it2->first.Scalar() == INFO_NODE_NAME)
                                    continue;
                                
                                if (it2->second.IsNull())
                                    continue;

                                size_t cbuf;
                                MultipassConfigParserError err = processCBuffer(it2->second, cbuf);

                                if (err)
                                    return err;
                                
                                std::string bufslot = it2->first.as<std::string>();

                                if (bufslot.length() < 1)
                                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, bufslot);

                                if (bufslot[0] == '$')
                                {
                                    if (bufslot.length() < 3 || bufslot[1] != 'b')
                                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, bufslot);

                                    int idx;

                                    try
                                    {
                                        idx = std::stoi(bufslot.substr(2));
                                    }
                                    catch (const std::ilwalid_argument& ref)
                                    {
                                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, bufslot + " " + ref.what());
                                    }

                                    if (idx < 0 || idx >= MAX_CBUFFERS)
                                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIndexOutOfRange, bufslot);

                                    auto r = datasrc.m_cbuffers.insert(std::make_pair(ShaderPassDatasource::BindingKey(idx), cbuf));

                                    if (!r.second)
                                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eDuplicateBindingSlot, bufslot);
                                }
                                else
                                {
                                    auto r = datasrc.m_cbuffers.insert(std::make_pair(ShaderPassDatasource::BindingKey(bufslot), cbuf));

                                    if (!r.second)
                                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eDuplicateBindingSlot, bufslot);
                                }
                            }
                        }
                    }

                    if (lwrrElem.datasourceNode["samplerstates"])
                    {
                        YAML::Node samplerstates = lwrrElem.datasourceNode["samplerstates"];

                        if (!samplerstates.IsMap())
                        {
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, getMarkFromNode(samplerstates));
                        }

                        for (YAML::const_iterator it2 = samplerstates.begin(); it2 != samplerstates.end(); ++it2)
                        {
                            if (it2->second.IsNull())
                                continue;

                            if (!it2->first.IsScalar())
                            {
                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidProperty, getMarkFromNode(it2->first));
                            }

                            if (it2->first.Scalar() == INFO_NODE_NAME)
                                continue;

                            size_t samplerIdx;
                            MultipassConfigParserError err = processSamplerState(it2->second, samplerIdx);

                            if (err)
                                return err;

                            std::string bufslot = it2->first.as<std::string>();


                            if (bufslot.length() < 1)
                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, bufslot);

                            if (bufslot[0] == '$')
                            {
                                if (bufslot.length() < 3 || bufslot[1] != 's')
                                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, bufslot);

                                int idx;

                                try
                                {
                                    idx = std::stoi(bufslot.substr(2));
                                }
                                catch (const std::ilwalid_argument& ref)
                                {
                                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, bufslot + " " + ref.what());
                                }

                                if (idx < 0 || idx >= MAX_SAMPLERS)
                                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIndexOutOfRange, bufslot);

                                auto r = datasrc.m_samplers.insert(std::make_pair(ShaderPassDatasource::BindingKey(idx), samplerIdx));
                    
                                if (!r.second)
                                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eDuplicateBindingSlot, bufslot);
    
                            }
                            else
                            {
                                auto r = datasrc.m_samplers.insert(std::make_pair(ShaderPassDatasource::BindingKey(bufslot), samplerIdx));

                                if (!r.second)
                                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eDuplicateBindingSlot, bufslot);
                            }
                        }
                    }

                    MultipassConfigParserError err = readWidthHeight(lwrrElem.datasourceNode, datasrc);

                    if (err)
                        return err;

                    YAML::Node scaleWidth = lwrrElem.datasourceNode["scale-width"];

                    datasrc.m_scaleWidth = 1.0f;

                    if (scaleWidth)
                    {
                        if (!scaleWidth.IsScalar())
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(scaleWidth));

                        float sw = scaleWidth.as<DecoratedScalar<float> >(-1.0f).m_value;

                        if (sw <= 0 || sw > 1.0f)
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eValueOutOfRange, 
                                scaleWidth.as<ExtendedName>().m_value + std::string(" should be in range (0, 1] "), getMarkFromNode(scaleWidth));

                        datasrc.m_scaleWidth = sw;
                    }

                    YAML::Node scaleHeight = lwrrElem.datasourceNode["scale-height"];

                    datasrc.m_scaleHeight = 1.0f;

                    if (scaleHeight)
                    {
                        if (!scaleHeight.IsScalar())
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(scaleHeight));

                        float sh = scaleHeight.as<DecoratedScalar<float> >(-1.0f).m_value;

                        if (sh <= 0 || sh > 1.0f)
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eValueOutOfRange,
                                scaleHeight.as<ExtendedName>().m_value + std::string(" should be in range (0, 1] "), getMarkFromNode(scaleHeight));

                        datasrc.m_scaleHeight = sh;
                    }

                    YAML::Node surfaceType = lwrrElem.datasourceNode["type"];

                    SurfaceType stype = SurfaceType::kMatchInputColorChannel;

                    if (surfaceType)
                    {
                        if (!surfaceType.IsScalar())
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(surfaceType));
                                            
                        stype = SurfaceTypeMap::colwert(surfaceType.as<ExtendedName>().m_value, SurfaceType::NUM_ENTRIES);

                        if (stype == SurfaceType::NUM_ENTRIES)
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(surfaceType));
                    }

                    YAML::Node channel = lwrrElem.datasourceNode["channel"];

                    ShaderPassDatasource::ShaderPassInput shaderPassInput;

                    shaderPassInput.m_channel = ShaderPassChannelType::kTARGET0_COLOR;

                    if (channel)
                    {
                        if (!channel.IsScalar())
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(channel));

                        shaderPassInput.m_channel = ShaderPassChannelTypeMap::colwert(channel.as<ExtendedName>().m_value, ShaderPassChannelType::NUM_ENTRIES);

                        if (shaderPassInput.m_channel == ShaderPassChannelType::NUM_ENTRIES)
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(channel));
                    }

                    datasrc.referenceChannel(shaderPassInput.m_channel, stype);

                    auto passIterator = insertShaderPassDatasource(datasrc);

                    shaderPassInput.m_shaderPassDatasource = passIterator.first;

                    m_shaderPassDatasources[passIterator.first].referenceChannel(shaderPassInput.m_channel, stype);

                    lwrrElem.output = ShaderPassDatasource::TextureInput(shaderPassInput);
                }
                else if (lwrrElem.type == eFilename)
                {
                    InputTextureDatasource datasrc;

                    YAML::Node filenameNode = lwrrElem.datasourceNode["filename"];

                    if (!filenameNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(filenameNode));

                    datasrc.m_filename = filenameNode.as<ExtendedName>().m_value;

                    // jukim - adding option to exclude certain files from hash
                    YAML::Node hashExcludeNode = lwrrElem.datasourceNode["excludeHash"];
                    if (hashExcludeNode)
                    {
                        if (!hashExcludeNode.IsScalar())
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(hashExcludeNode));
                        try
                        {
                            datasrc.m_excludeHash = hashExcludeNode.as<DecoratedScalar<bool>>().m_value;
                        }
                        catch (const YAML::TypedBadColwersion<DecoratedScalar<ir::userConstTypes::UInt> >&)
                        {
                            ExtendedName exname = hashExcludeNode.as<ExtendedName>();

                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidBool, exname.m_value, exname.getMark());
                        }
                    }
                    else
                    {
                        datasrc.m_excludeHash = false;
                    }

                    MultipassConfigParserError err = readWidthHeight(lwrrElem.datasourceNode, datasrc);

                    if (err)
                        return err;

                    YAML::Node surfaceType = lwrrElem.datasourceNode["type"];

                    datasrc.m_type = SurfaceType::kUndefined; //it is Undefined here because for textures we'd like to match the file format by default

                    if (surfaceType)
                    {
                        if (!surfaceType.IsScalar())
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(surfaceType));

                        datasrc.m_type = SurfaceTypeMap::colwert(surfaceType.as<ExtendedName>().m_value, SurfaceType::NUM_ENTRIES);

                        if (datasrc.m_type == SurfaceType::NUM_ENTRIES)
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(surfaceType));
                    }

                    auto passIterator = insertInputTextureDatasource(datasrc);

                    ShaderPassDatasource::UserTextureInput userTexInput;
                    userTexInput.m_textureDatasource = passIterator.first;

                    lwrrElem.output = ShaderPassDatasource::TextureInput(userTexInput);
                }
                else if (lwrrElem.type == eProcedural)
                {
                    ProceduralTextureDatasource datasrc;

                    YAML::Node procedureNode = lwrrElem.datasourceNode["procedure"];

                    if (!procedureNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(procedureNode));

                    datasrc.m_procedureType = ProceduralTextureTypeMap::colwert(procedureNode.as<ExtendedName>().m_value, ProceduralTextureType::NUM_ENTRIES);

                    if (datasrc.m_procedureType == ProceduralTextureType::NUM_ENTRIES)
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(procedureNode));

                    MultipassConfigParserError err = readWidthHeight(lwrrElem.datasourceNode, datasrc);

                    if (err)
                        return err;

                    YAML::Node surfaceType = lwrrElem.datasourceNode["type"];

                    datasrc.m_surfaceType = SurfaceType::kUndefined; //stick to the default format, as we populate that on cpu in Ansel

                    if (surfaceType)
                    {
                        if (!surfaceType.IsScalar())
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(surfaceType));

                        datasrc.m_surfaceType = SurfaceTypeMap::colwert(surfaceType.as<ExtendedName>().m_value, SurfaceType::NUM_ENTRIES);

                        if (datasrc.m_surfaceType == SurfaceType::NUM_ENTRIES)
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(surfaceType));
                    }

                    auto passIterator = insertProceduralTextureDatasource(datasrc);

                    ShaderPassDatasource::ProceduralTextureInput procTexInput;
                    procTexInput.m_procTexDatasource = passIterator.first;

                    lwrrElem.output = ShaderPassDatasource::ProceduralTextureInput(procTexInput);
                }
                else //if(type == eSystem)
                {
                    ShaderPassDatasource::SystemInput sysInput;
                    YAML::Node channel = lwrrElem.datasourceNode["channel"];

                    sysInput.m_channel = SystemChannelType::kPIPE_INPUTS_COLOR;

                    if (channel)
                    {
                        if (!channel.IsScalar())
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(channel));

                        sysInput.m_channel = SystemChannelTypeMap::colwert(channel.as<ExtendedName>().m_value, SystemChannelType::NUM_ENTRIES);

                        if (sysInput.m_channel == SystemChannelType::NUM_ENTRIES)
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(channel));
                    }

                    m_systemDatasource.referenceChannel(sysInput.m_channel);

                    lwrrElem.output = ShaderPassDatasource::TextureInput(sysInput);
                }

                if (localStack[lwrrNode].nextSiblingIndex == -1)
                {
                    lwrrNode = localStack[lwrrNode].parentIndex;
                }
                else
                {
                    lwrrNode = localStack[lwrrNode].nextSiblingIndex;

                    while (localStack[lwrrNode].firstChildIndex != -1)
                        lwrrNode = localStack[lwrrNode].firstChildIndex;
                }
            }

            mainOut = localStack[0].output;

            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
        }

        const char*  userConstParamsStr[] = { 
            "name",
            "type",
            "ui-control",
            "default-value",
            "minimum-value",
            "maximum-value",
            "ui-value-step",
            "ui-sticky-value",
            "ui-sticky-region",
            "options",
            "default-option",
            "ui-label",
            "ui-hint", 
            "ui-label-localized",
            "ui-hint-localized",
            "ui-value-unit",
            "ui-value-unit-localized",
            "ui-value-min",
            "ui-value-max",
        };
        DECLARE_STRING_TO_ENUM_MAP(UserConstDesc::EParams, userConstParamsStr, UserConstParamsMap);
            
        const char*  userConstDataTypeStr[] = {
            "bool",
            "int",
            "uint",
            "float"
        };
        DECLARE_STRING_TO_ENUM_MAP(UserConstDataType, userConstDataTypeStr, UserConstDataTypeMap);

        const char*  userConstUiControlTypeStr[] = {
            "slider",
            "checkbox",
            "flyout",
            "editbox"
        };
        DECLARE_STRING_TO_ENUM_MAP(UserConstUiControlType, userConstUiControlTypeStr, UserConstUiControlTypeMap);

        ir::UserConstDataType colwertUserConstDataTypeToIrUserConstType(const UserConstDataType& ref)
        {
            switch (ref)
            {
            case UserConstDataType::kBool:
                return ir::UserConstDataType::kBool;
            case UserConstDataType::kFloat:
                return ir::UserConstDataType::kFloat;
            case UserConstDataType::kInt:
                return ir::UserConstDataType::kInt;
            case UserConstDataType::kUInt:
                return ir::UserConstDataType::kUInt;
            default:
                assert(false && "Invalid user constant data type!");
                return ir::UserConstDataType::NUM_ENTRIES;
            }
        }
                
        MultipassConfigParserError EffectParser::processUserConstantsNode(const YAML::Node& userConstantsNode)
        {
            YAML::const_iterator it = userConstantsNode.begin();
            ++it; // skip the info node

            std::set<std::string> nameUniquenessCheck;

            for (; it != userConstantsNode.end(); ++it)
            {
                if (!it->IsMap())
                {
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, getMarkFromNode(*it));
                }

                m_userConstDescs.push_back(UserConstDesc());
                UserConstDesc& newConst = m_userConstDescs.back();
                
                YAML::Node nameNode = (*it)["name"];

                if (!nameNode)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eRequiredPropertyMissing, "name", getMarkFromNode(*it));

                if (!nameNode.IsScalar())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(nameNode));

                newConst.name = nameNode.as<ExtendedName>().m_value;

                auto nameIt = nameUniquenessCheck.insert(newConst.name);

                if (!nameIt.second)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eDuplicateNameNotAllowed, 
                        std::string("user constant ") + newConst.name, getMarkFromNode(nameNode));

                YAML::Node typeNode = (*it)["type"];

                if (!typeNode)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eRequiredPropertyMissing, "type", getMarkFromNode(*it));

                if (!typeNode.IsScalar())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(typeNode));

                UserConstDataType type = UserConstDataTypeMap::colwert(typeNode.as<ExtendedName>().m_value, UserConstDataType::NUM_ENTRIES);

                if (type == UserConstDataType::NUM_ENTRIES)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(typeNode));

                newConst.type = type;
            
                YAML::Node minNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kMinimumValue]];

                bool isMinUnBounded = false;

                if (minNode && !minNode.IsNull())
                {
                    MultipassConfigParserError err = readMinimumValueNodeHelper(minNode, type, newConst.minimumValue, isMinUnBounded);

                    if (err)
                        return err;
                }
                else
                {
                    switch (type)
                    {
                    case UserConstDataType::kBool:
                        newConst.minimumValue.boolValue = ir::userConstTypes::Bool::kFalse;
                        break;
                    case UserConstDataType::kUInt:
                        newConst.minimumValue.uintValue = 0;
                        break;
                    case UserConstDataType::kInt:
                        newConst.minimumValue.intValue = 0;
                        break;
                    case UserConstDataType::kFloat:
                        newConst.minimumValue.floatValue = 0.0f;
                        break;
                    default:
                        assert(false && "unsupported type!");
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eInternalError, "unsupported type!");

                    }
                }
                            
                YAML::Node maxNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kMaximumValue]];

                bool isMaxUnBounded = false;

                if (maxNode && !maxNode.IsNull())
                {
                    MultipassConfigParserError err = readMaximumValueNodeHelper(maxNode, type, newConst.maximumValue, isMaxUnBounded);

                    if (err)
                        return err;
                }
                else
                {
                    switch (type)
                    {
                    case UserConstDataType::kBool:
                        newConst.maximumValue.boolValue = ir::userConstTypes::Bool::kTrue;
                        break;
                    case UserConstDataType::kUInt:
                        newConst.maximumValue.uintValue = 1;
                        break;
                    case UserConstDataType::kInt:
                        newConst.maximumValue.intValue = 1;
                        break;
                    case UserConstDataType::kFloat:
                        newConst.maximumValue.floatValue = 1.0f;
                        break;
                    default:
                        assert(false && "unsupported type!");
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eInternalError, "unsupported type!");

                    }
                }
                            
                if (!compareValueLess(newConst.minimumValue, newConst.maximumValue, newConst.type))
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eValueOutOfRange,
                        "minimum value should be less than maximum value!", getMarkFromNode(*it));

                YAML::Node optionsNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kOptions]];
                bool areOptionsProvided = optionsNode && !optionsNode.IsNull();
                YAML::Node uiControlNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kUiControl]];

                if (!uiControlNode || uiControlNode.IsNull())
                {
                    if (areOptionsProvided)
                    {
                        newConst.uiControlType = UserConstUiControlType::kFlyout;
                    }
                    else if (type == UserConstDataType::kBool)
                    {
                        newConst.uiControlType = UserConstUiControlType::kCheckbox;
                    }
                    else
                    {
                        bool bounded = !isMaxUnBounded && !isMinUnBounded;

                        if (bounded)
                        {
                            newConst.uiControlType = UserConstUiControlType::kSlider;
                        }
                        else
                        {
                            newConst.uiControlType = UserConstUiControlType::kEditbox;
                        }
                    }
                }
                else
                {
                    if (!uiControlNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(uiControlNode));

                    UserConstUiControlType uiControl = UserConstUiControlTypeMap::colwert(uiControlNode.as<ExtendedName>().m_value, UserConstUiControlType::NUM_ENTRIES);

                    if (uiControl == UserConstUiControlType::NUM_ENTRIES)
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidEnumValue, getMarkFromNode(uiControlNode));

                    if (uiControl == UserConstUiControlType::kFlyout && !areOptionsProvided)
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eRequiredPropertyMissing,
                            userConstParamsStr[(int)UserConstDesc::EParams::kOptions],
                            getMarkFromNode(uiControlNode));

                    if (uiControl != UserConstUiControlType::kFlyout && areOptionsProvided)
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty,
                            userConstParamsStr[(int)UserConstDesc::EParams::kOptions],
                            getMarkFromNode(optionsNode));

                    if (uiControl == UserConstUiControlType::kSlider && type == UserConstDataType::kFloat && (isMaxUnBounded || isMinUnBounded))
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidValue,
                            "minimum and maximum can't be unbounded for a floating point slider", getMarkFromNode(uiControlNode));

                    newConst.uiControlType = uiControl;
                }

                if (newConst.uiControlType == UserConstUiControlType::kFlyout && minNode && !minNode.IsNull())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty,
                        "minimum can't be expicitly set for a flyout", getMarkFromNode(minNode));

                if (newConst.uiControlType == UserConstUiControlType::kFlyout && maxNode && !maxNode.IsNull())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty,
                        "maximum can't be expicitly set for a flyout", getMarkFromNode(maxNode));
                            
                if (areOptionsProvided)
                {
                    MultipassConfigParserError err = readOptionsNodeHelper(optionsNode, newConst);

                    if (err)
                        return err;
                }
                else
                {
                    newConst.options.resize(0);
                }

                YAML::Node defaultOptionNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kDefaultOption]];

                if (defaultOptionNode && !defaultOptionNode.IsNull())
                {
                    if (!areOptionsProvided)
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, userConstParamsStr[(int)UserConstDesc::EParams::kDefaultOption],
                            getMarkFromNode(defaultOptionNode));

                    if (!defaultOptionNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(defaultOptionNode));

                    try
                    {
                        newConst.defaultOption = defaultOptionNode.as<DecoratedScalar<ir::userConstTypes::UInt> >().m_value;
                    }
                    catch (const YAML::TypedBadColwersion<DecoratedScalar<ir::userConstTypes::UInt> >&)
                    {
                        ExtendedName exname = defaultOptionNode.as<ExtendedName>();

                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidUInt, exname.m_value, exname.getMark());
                    }

                    if (newConst.defaultOption >= (int) newConst.options.size())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eValueOutOfRange,
                            "default option larger or equal to the number of options!", getMarkFromNode(defaultOptionNode));

                }
                else if (areOptionsProvided)
                {
                    if (!newConst.options.size())
                    {
                        newConst.defaultOption = -1;
                    }
                    else
                    {
                        newConst.defaultOption = 0;
                    }
                }
                else
                {
                    newConst.defaultOption = -1;
                }


                YAML::Node defaultValueNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kDefaultValue]];

                if (defaultValueNode && !defaultValueNode.IsNull())
                {
                    if (defaultOptionNode && !defaultOptionNode.IsNull())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty,
                            "default-value and default-option are mutually exclusive", getMarkFromNode(defaultValueNode));

                    if (!defaultValueNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(defaultValueNode));

                    MultipassConfigParserError err = readValueFromNode(defaultValueNode, newConst.defaultValue, newConst.type);

                    if (err)
                        return err;

                    if (compareValueLess(newConst.defaultValue, newConst.minimumValue, newConst.type) ||
                        compareValueMore(newConst.defaultValue, newConst.maximumValue, newConst.type))
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eValueOutOfRange,
                            "default value out of the bounds of [minimum-value maximum-value]", getMarkFromNode(defaultValueNode));

                    if (areOptionsProvided)
                    {
                        unsigned int matchCounter = 0;

                        for (auto opt = newConst.options.cbegin(), end = newConst.options.cend(); opt != end; ++opt)
                        {
                            if (compareValueEqual(opt->value, newConst.defaultValue, newConst.type))
                                newConst.defaultOption = (int) (opt - newConst.options.cbegin()), matchCounter += 1;
                        }

                        if (matchCounter == 0)
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidValue,
                                "default value doesn't match any of the options", getMarkFromNode(defaultValueNode));

                        if (matchCounter > 1)
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidValue,
                                "default value matches more than one option", getMarkFromNode(defaultValueNode));
                    }
                }
                else
                {
                    if (areOptionsProvided && newConst.defaultOption >= 0)
                    {
                        newConst.defaultValue = newConst.options[newConst.defaultOption].value;
                    }
                    else
                    {
                        newConst.defaultValue = newConst.minimumValue;
                    }
                }

                //ui-value-min
                YAML::Node uiValueMinNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kUiValueMin]];

                if (uiValueMinNode && !uiValueMinNode.IsNull())
                {
                    if (newConst.uiControlType != UserConstUiControlType::kSlider || newConst.type == UserConstDataType::kBool)
                    {
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, "ui-value-min can be only used with non-boolean sliders!", getMarkFromNode(uiValueMinNode));
                    }

                    if (!uiValueMinNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(uiValueMinNode));

                    MultipassConfigParserError err = readValueFromNode(uiValueMinNode, newConst.uiValueMin, newConst.type);

                    if (err)
                        return err;

                }
                else
                {
                    newConst.uiValueMin = newConst.minimumValue;
                }

                //ui-value-max
                YAML::Node uiValueMaxNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kUiValueMax]];

                if (uiValueMaxNode && !uiValueMaxNode.IsNull())
                {
                    if (newConst.uiControlType != UserConstUiControlType::kSlider || newConst.type == UserConstDataType::kBool)
                    {
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, "ui-value-max can be only used with non-boolean sliders!", getMarkFromNode(uiValueMaxNode));
                    }

                    if (!uiValueMaxNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(uiValueMaxNode));

                    MultipassConfigParserError err = readValueFromNode(uiValueMaxNode, newConst.uiValueMax, newConst.type);

                    if (err)
                        return err;

                }
                else
                {
                    newConst.uiValueMax = newConst.maximumValue;
                }

                if (compareValueEqual(newConst.uiValueMin, newConst.uiValueMax, newConst.type))
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eValueOutOfRange,
                        "ui-value-min can't be equal to ui-value-max!", getMarkFromNode(*it));

                //ui-value-step
                YAML::Node uiValueStepNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kUiValueStep]];

                if (uiValueStepNode && !uiValueStepNode.IsNull())
                {
                    if ((newConst.uiControlType != UserConstUiControlType::kSlider && newConst.uiControlType != UserConstUiControlType::kEditbox)
                        || newConst.type == UserConstDataType::kBool)
                    {
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty,
                            "ui-value-step can be only used with non-boolean sliders or editboxes!", getMarkFromNode(uiValueStepNode));
                    }

                    if (!uiValueStepNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(uiValueStepNode));

                    MultipassConfigParserError err = readValueFromNode(uiValueStepNode, newConst.uiValueStep, newConst.type);

                    if (err)
                        return err;

                    if (compareValueLess(newConst.uiValueStep, getZeroValue(newConst.type), newConst.type))
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eValueOutOfRange,
                            "ui-value-step should be non-negative!", getMarkFromNode(uiValueStepNode));
                }
                else
                {
                    newConst.uiValueStep = getZeroValue(newConst.type);
                }

                //ui-sticky-value

                YAML::Node uiStickyValueNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kStickyValue]];

                if (uiStickyValueNode && !uiStickyValueNode.IsNull())
                {
                    if (newConst.type != UserConstDataType::kFloat || newConst.uiControlType != UserConstUiControlType::kSlider)
                    {
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, "sticky value can be only used with float sliders!", getMarkFromNode(uiStickyValueNode));
                    }

                    if (!uiStickyValueNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(uiStickyValueNode));

                    try
                    {
                        newConst.stickyValue = uiStickyValueNode.as<DecoratedScalar<ir::userConstTypes::Float> >().m_value;
                    }
                    catch (const YAML::TypedBadColwersion<DecoratedScalar<ir::userConstTypes::Float> >&)
                    {
                        ExtendedName exname = uiStickyValueNode.as<ExtendedName>();

                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidFloat, exname.m_value, exname.getMark());
                    }
                }
                else if (newConst.type == UserConstDataType::kFloat)
                {
                    newConst.stickyValue = newConst.defaultValue.floatValue;
                }
                else
                {
                    newConst.stickyValue = 0.0f;
                }

                //ui-sticky-region

                YAML::Node uiStickyRegionNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kStickyRegion]];

                if (uiStickyRegionNode && !uiStickyRegionNode.IsNull())
                {
                    if (newConst.type != UserConstDataType::kFloat || newConst.uiControlType != UserConstUiControlType::kSlider)
                    {
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, "sticky region can be only used with float sliders!", getMarkFromNode(uiStickyRegionNode));
                    }

                    if (!uiStickyRegionNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(uiStickyRegionNode));

                    try
                    {
                        newConst.stickyRegion = uiStickyRegionNode.as<DecoratedScalar<ir::userConstTypes::Float> >().m_value;
                    }
                    catch (const YAML::TypedBadColwersion<DecoratedScalar<ir::userConstTypes::Float> >&)
                    {
                        ExtendedName exname = uiStickyRegionNode.as<ExtendedName>();

                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidFloat, exname.m_value, exname.getMark());
                    }

                    if (newConst.stickyRegion < 0.0f || newConst.stickyRegion > 1.0f)
                    {
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eValueOutOfRange,
                            uiStickyRegionNode.as<ExtendedName>().m_value + std::string(" should be in range [0, 1] "),
                            getMarkFromNode(uiStickyRegionNode));
                    }
                }
                else
                {
                    newConst.stickyRegion = 0.0f;
                }

                // ui-label
                YAML::Node uiLabelNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kUiLabel]];

                if (uiLabelNode && !uiLabelNode.IsNull())
                {
                    if (!uiLabelNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(uiLabelNode));

                    ExtendedName exname = uiLabelNode.as<ExtendedName>();

                    newConst.uiLabel = exname.m_value;
                }
                else
                {
                    newConst.uiLabel = newConst.name;
                }

                // ui-label-localized
                YAML::Node uiLabelLocalizedNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kUiLabelLocalized]];

                if (uiLabelLocalizedNode && !uiLabelLocalizedNode.IsNull())
                {
                    MultipassConfigParserError err = readLocalizedStringFromNode(uiLabelLocalizedNode, newConst.uiLabelLocalized);

                    if (err)
                        return err;
                }
                else
                {
                    newConst.uiLabelLocalized.clear();
                }

                // ui-hint
                YAML::Node uiHintNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kUiHint]];

                if (uiHintNode && !uiHintNode.IsNull())
                {
                    if (!uiHintNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(uiHintNode));

                    ExtendedName exname = uiHintNode.as<ExtendedName>();

                    newConst.uiHint = exname.m_value;
                }
                else
                {
                    newConst.uiHint.resize(0);
                }

                // ui-hint-localized
                YAML::Node uiHintLocalizedNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kUiHintLocalized]];

                if (uiHintLocalizedNode && !uiHintLocalizedNode.IsNull())
                {
                    MultipassConfigParserError err = readLocalizedStringFromNode(uiHintLocalizedNode, newConst.uiHintLocalized);

                    if (err)
                        return err;
                }
                else
                {
                    newConst.uiHintLocalized.clear();
                }

                // ui-value-unit
                YAML::Node uiValueUnitNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kUiValueUnit]];

                if (uiValueUnitNode && !uiValueUnitNode.IsNull())
                {
                    if (!uiValueUnitNode.IsScalar())
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(uiValueUnitNode));

                    ExtendedName exname = uiValueUnitNode.as<ExtendedName>();

                    newConst.uiValueUnit = exname.m_value;
                }
                else
                {
                    newConst.uiValueUnit.resize(0);
                }

                // ui-value-unit-localized
                YAML::Node uiValueUnitLocalizedNode = (*it)[userConstParamsStr[(int)UserConstDesc::EParams::kUiValueUnitLocalized]];

                if (uiValueUnitLocalizedNode && !uiValueUnitLocalizedNode.IsNull())
                {
                    MultipassConfigParserError err = readLocalizedStringFromNode(uiValueUnitLocalizedNode, newConst.uiValueUnitLocalized);

                    if (err)
                        return err;
                }
                else
                {
                    newConst.uiValueUnitLocalized.clear();
                }
            }
                        
            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
        }
                        
        MultipassConfigParserError EffectParser::readValueFromNode(const YAML::Node& node, FlexibleDataType& val, UserConstDataType type)
        {
            assert(node.IsDefined() && node.IsScalar());

            if (type == UserConstDataType::kBool)
            {
                ExtendedName exname = node.as<ExtendedName>();
                std::string str = exname.m_value;
                darkroom::tolowerInplace(str);

                if (str == "0" || str == "n" || str == "false" || str == "no" || str == "disabled" || str == "off")
                {
                    val.boolValue = ir::userConstTypes::Bool::kFalse;
                }
                else if (str == "1" || str == "y" || str == "true" || str == "yes" || str == "enabled" || str == "on")
                {
                    val.boolValue = ir::userConstTypes::Bool::kTrue;
                }
                else
                {
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidBool, exname.m_value, exname.getMark());
                }
            }
            else if (type == UserConstDataType::kFloat)
            {
                try
                {
                    val.floatValue = node.as<DecoratedScalar<ir::userConstTypes::Float> >().m_value;
                }
                catch (const YAML::TypedBadColwersion<DecoratedScalar<ir::userConstTypes::Float> >&)
                {
                    ExtendedName exname = node.as<ExtendedName>();

                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidFloat, exname.m_value, exname.getMark());
                }
            }
            else if (type == UserConstDataType::kInt)
            {
                try
                {
                    val.intValue = node.as<DecoratedScalar<ir::userConstTypes::Int> >().m_value;
                }
                catch (const YAML::TypedBadColwersion<DecoratedScalar<ir::userConstTypes::Int> >&)
                {
                    ExtendedName exname = node.as<ExtendedName>();

                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidInt, exname.m_value, exname.getMark());
                }
            }
            else if (type == UserConstDataType::kUInt)
            {
                try
                {
                    val.uintValue = node.as<DecoratedScalar<ir::userConstTypes::UInt> >().m_value;
                }
                catch (const YAML::TypedBadColwersion<DecoratedScalar<ir::userConstTypes::UInt> >&)
                {
                    ExtendedName exname = node.as<ExtendedName>();

                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidUInt, exname.m_value, exname.getMark());
                }
            }
            else
            {
                assert(false && "Unsupported UserConstDataType!");
            }

            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
        }
        
        MultipassConfigParserError EffectParser::readLocalizedStringFromNode(const YAML::Node& node, UserConstLocalizedString& val)
        {
            if (!node.IsMap())
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, getMarkFromNode(node));
            }

            for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
            {
                if (it->second.IsNull())
                    continue;

                if (!it->first.IsScalar())
                {
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidProperty, getMarkFromNode(it->first));
                }

                if (it->first.Scalar() == INFO_NODE_NAME)
                    continue;

                std::string localeNameS = it->first.as<std::string>();

                UserConstOptionalLanguageType localeName = UserConstOptionalLanguageTypeMap::colwert(localeNameS, UserConstOptionalLanguageType::NUM_ENTRIES);

                if (localeName == UserConstOptionalLanguageType::NUM_ENTRIES)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eUnexpectedProperty, localeNameS);

                if (!it->second.IsScalar())
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(it->second));

                ExtendedName exname = it->second.as<ExtendedName>();

                val.str[(int)localeName] = exname.m_value;
            }
            
            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
        }
        
        MultipassConfigParserError EffectParser::readMinimumValueNodeHelper(const YAML::Node& node, const UserConstDataType& type, FlexibleDataType& value, bool& isUnbounded)
        {
            isUnbounded = false;

            if (!node.IsScalar())
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(node));

            ExtendedName exname = node.as<ExtendedName>();

            if (exname.m_value == UNBOUNDED_NUMBER_KEYWORD)
            {
                if (type == UserConstDataType::kBool)
                {
                    value.boolValue = ir::userConstTypes::Bool::kFalse;
                }
                else if (type == UserConstDataType::kFloat)
                {
                    value.floatValue = -FLT_MAX;
                }
                else if (type == UserConstDataType::kInt)
                {
                    value.intValue = INT_MIN;
                }
                else if (type == UserConstDataType::kUInt)
                {
                    value.uintValue = 0;
                }
                else
                {
                    assert(false && "user const type unsupported!");
                }

                isUnbounded = true;

                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
            }
            else
            {
                MultipassConfigParserError err = readValueFromNode(node, value, type);

                return err;
            }
        }

        MultipassConfigParserError EffectParser::readMaximumValueNodeHelper(const YAML::Node& node, const UserConstDataType& type, FlexibleDataType& value, bool& isUnbounded)
        {
            isUnbounded = false;

            if (!node.IsScalar())
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(node));

            ExtendedName exname = node.as<ExtendedName>();

            if (exname.m_value == UNBOUNDED_NUMBER_KEYWORD)
            {
                if (type == UserConstDataType::kBool)
                {
                    value.boolValue = ir::userConstTypes::Bool::kTrue;
                }
                else if (type == UserConstDataType::kFloat)
                {
                    value.floatValue = FLT_MAX;
                }
                else if (type == UserConstDataType::kInt)
                {
                    value.intValue = INT_MAX;
                }
                else if (type == UserConstDataType::kUInt)
                {
                    value.uintValue = UINT_MAX;
                }
                else
                {
                    assert(false && "user const type unsupported!");
                }

                isUnbounded = true;

                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
            }
            else
            {
                MultipassConfigParserError err = readValueFromNode(node, value, type);

                return err;
            }
        }

        std::string EffectParser::stringifyValue(const FlexibleDataType& val, UserConstDataType type)
        {
            std::stringbuf buf;
            std::ostream strstream(&buf);

            if (type == UserConstDataType::kBool)
                strstream << (val.boolValue == ir::userConstTypes::Bool::kTrue ? L"True" : L"False");
            else if (type == UserConstDataType::kInt)
                strstream << val.intValue;
            else if (type == UserConstDataType::kUInt)
                strstream << val.uintValue;
            else if (type == UserConstDataType::kFloat)
                strstream << val.floatValue;
            else
                assert(false && "Unsupported type!");

            return buf.str();
        }
        

        bool EffectParser::compareValueLess(const FlexibleDataType& left, const FlexibleDataType& right, UserConstDataType type)
        {
            bool less = false;
        
            if (type == UserConstDataType::kBool)
                less = left.boolValue < right.boolValue;
            else if (type == UserConstDataType::kInt)
                less = left.intValue < right.intValue;
            else if (type == UserConstDataType::kUInt)
                less = left.uintValue < right.uintValue;
            else if (type == UserConstDataType::kFloat)
                less = left.floatValue < right.floatValue;
            else
                assert(false && "Unsupported type!");

            return less;
        }

        bool EffectParser::compareValueMore(const FlexibleDataType& left, const FlexibleDataType& right, UserConstDataType type)
        {
            bool more = false;

            if (type == UserConstDataType::kBool)
                more = left.boolValue > right.boolValue;
            else if (type == UserConstDataType::kInt)
                more = left.intValue > right.intValue;
            else if (type == UserConstDataType::kUInt)
                more = left.uintValue > right.uintValue;
            else if (type == UserConstDataType::kFloat)
                more = left.floatValue > right.floatValue;
            else
                assert(false && "Unsupported type!");

            return more;
        }

        bool EffectParser::compareValueEqual(const FlexibleDataType& left, const FlexibleDataType& right, UserConstDataType type)
        {
            bool equal = false;

            if (type == UserConstDataType::kBool)
                equal = left.boolValue == right.boolValue;
            else if (type == UserConstDataType::kInt)
                equal = left.intValue == right.intValue;
            else if (type == UserConstDataType::kUInt)
                equal = left.uintValue == right.uintValue;
            else if (type == UserConstDataType::kFloat)
                equal = left.floatValue == right.floatValue;
            else
                assert(false && "Unsupported type!");

            return equal;
        }

        FlexibleDataType EffectParser::getZeroValue(UserConstDataType type)
        {
            FlexibleDataType ret;

            if (type == UserConstDataType::kBool)
                ret.boolValue = ir::userConstTypes::Bool::kFalse;
            else if (type == UserConstDataType::kInt)
                ret.intValue = 0;
            else if (type == UserConstDataType::kUInt)
                ret.uintValue = 0;
            else if (type == UserConstDataType::kFloat)
                ret.floatValue = 0.0f;
            else
                assert(false && "Unsupported type!");

            return ret;
        }


        MultipassConfigParserError EffectParser::readOptionsNodeHelper(const YAML::Node& node, UserConstDesc& desc)
        {
            desc.options.resize(0);

            if (!node.IsSequence())
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotSequence, getMarkFromNode(node));

            YAML::const_iterator it = node.begin();
            ++it; //skip the info node

            for (; it != node.end(); ++it)
            {
                UserConstDesc::ListOption listOption;

                if (it->IsScalar())
                {
                    MultipassConfigParserError err = readValueFromNode(*it, listOption.value, desc.type);

                    if (err)
                        return err;

                    listOption.name = stringifyValue(listOption.value, desc.type);
                }
                else if (it->IsMap())
                {
                    if (!it->first.IsScalar())
                    {
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidProperty, getMarkFromNode(it->first));
                    }
                    
                    MultipassConfigParserError err = readValueFromNode(it->first, listOption.value, desc.type);

                    if (err)
                        return err;

                    if (!it->second.IsMap())
                    {
                        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidProperty, getMarkFromNode(it->second));
                    }

                    YAML::Node nameNode = it->second["name"];

                    if (!nameNode)
                    {
                        listOption.name = stringifyValue(listOption.value, desc.type);
                    }
                    else
                    {
                        if (nameNode.IsNull())
                        {
                            listOption.name = "";
                        }
                        else if (!nameNode.IsScalar())
                        {
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotScalar, getMarkFromNode(nameNode));
                        }
                        else
                        {
                            listOption.name = node.as<ExtendedName>().m_value;
                        }
                    }

                    YAML::Node nameLocalizedNode = it->second["name-localized"];

                    if (nameLocalizedNode)
                    {
                        MultipassConfigParserError err =  readLocalizedStringFromNode(nameLocalizedNode, listOption.nameLocalized);

                        if (err)
                            return err;
                    }
                }
                else
                {
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, getMarkFromNode(*it));
                }

                desc.options.push_back(listOption);
            }
        
            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
        }
                
        std::pair<size_t, bool> EffectParser::insertShaderPassDatasource(const ShaderPassDatasource& ref)
        {
            size_t lastIdx = m_shaderPassDatasources.size();
            m_shaderPassDatasources.push_back(ref);
            auto iter = m_shaderPassDatasourcesSet.insert(ShaderPassDatasourceWrapperForSet(m_shaderPassDatasources, lastIdx));

            if (!iter.second)
            {
                m_shaderPassDatasources.pop_back();
            }

            return std::pair<size_t, bool>(iter.first->m_index, iter.second);
        }

        std::pair<size_t, bool> EffectParser::insertInputTextureDatasource(const InputTextureDatasource& ref)
        {
            size_t lastIdx = m_inputTextureDatasources.size();
            m_inputTextureDatasources.push_back(ref);
            auto iter = m_inputTextureDatasourcesSet.insert(InputTextureDatasourceWrapperForSet(m_inputTextureDatasources, lastIdx));

            if (!iter.second)
            {
                m_inputTextureDatasources.pop_back();
            }

            return std::pair<size_t, bool>(iter.first->m_index, iter.second);
        }

        std::pair<size_t, bool> EffectParser::insertProceduralTextureDatasource(const ProceduralTextureDatasource& ref)
        {
            size_t lastIdx = m_proceduralTextureDatasources.size();
            m_proceduralTextureDatasources.push_back(ref);
            auto iter = m_proceduralTextureDatasourcesSet.insert(ProceduralTextureDatasourceWrapperForSet(m_proceduralTextureDatasources, lastIdx));

            if (!iter.second)
            {
                m_proceduralTextureDatasources.pop_back();
            }

            return std::pair<size_t, bool>(iter.first->m_index, iter.second);
        }

        std::pair<size_t, bool> EffectParser::insertSamplerDesc(const SamplerDesc& ref)
        {
            size_t lastIdx = m_samplers.size();
            m_samplers.push_back(ref);
            auto iter = m_samplersSet.insert(SamplerDescWrapperForSet(m_samplers, lastIdx));

            if (!iter.second)
            {
                m_samplers.pop_back();
            }

            return std::pair<size_t, bool>(iter.first->m_index, iter.second);
        }

        std::pair<size_t, bool> EffectParser::insertCBufferDesc(const CBufferDesc& ref)
        {
            size_t lastIdx = m_cbuffers.size();
            m_cbuffers.push_back(ref);
            auto iter = m_cbuffersSet.insert(CBufferDescWrapperForSet(m_cbuffers, lastIdx));

            if (!iter.second)
            {
                m_cbuffers.pop_back();
            }

            return std::pair<size_t, bool>(iter.first->m_index, iter.second);
        }
    
        MultipassConfigParserError EffectParser::processImportPropertiesNoRelwrsion(YAML::Node& output, const YAML::Node& root, const YAML::Node& main)
        {
            output.reset();
                                
            struct StackEntry
            {
                YAML::Node					parentNodeIn;
                YAML::const_iterator		iteratorIntoChildrenIn;

                int							parentImportNode;
                int							prevSiblingImportNode;
            };

            YAML::Emitter emitOut;
            
            YAML::Node lwrrNode = main;
            YAML::NodeType::value	lwrrNodeType = lwrrNode.Type();
                                
            std::deque<StackEntry> treeStack;

            if (lwrrNodeType  == YAML::NodeType::Sequence)
            {
                emitOut << YAML::BeginSeq << ExtendedName("", lwrrNode.Mark());
            }
            else if (lwrrNodeType == YAML::NodeType::Map)
            {
                emitOut << YAML::BeginMap << YAML::Key << INFO_NODE_NAME << YAML::Value <<
                    ExtendedName("", lwrrNode.Mark());
            }
            else
            {
                assert(false && "Unexpected node type!");
            }
                                
            treeStack.push_back({ lwrrNode, YAML::const_iterator(), -1, -1 });
            treeStack.back().iteratorIntoChildrenIn = lwrrNode.begin();
            unsigned int lwrrElemIdx = 0;

            while (treeStack.size())
            {
                StackEntry& lwrrEntry = treeStack[lwrrElemIdx];

                YAML::NodeType::value	lwrrNodeType = lwrrEntry.parentNodeIn.Type();

                if (lwrrNodeType == YAML::NodeType::Map)
                {
                    bool added = false;

                    for (YAML::const_iterator end = lwrrEntry.parentNodeIn.end(); lwrrEntry.iteratorIntoChildrenIn != end; ++lwrrEntry.iteratorIntoChildrenIn)
                    {
                        if (!lwrrEntry.iteratorIntoChildrenIn->first.IsScalar())
                        {
                            YAML::Emitter out;
                            out << lwrrEntry.iteratorIntoChildrenIn->first;

                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIlwalidProperty, std::string(out.c_str()));
                        }

                        std::string propName = lwrrEntry.iteratorIntoChildrenIn->first.as<std::string>();

                        if (propName == "import")
                            continue;

                        bool propertyOverridden = false;
                        int siblingImportNode = lwrrEntry.prevSiblingImportNode;
                        int importerNode = -1;

                        while (siblingImportNode != -1)
                        {
                            if (treeStack[siblingImportNode].parentNodeIn[propName])
                            {
                                propertyOverridden = true;
                                break;
                            }

                            importerNode = siblingImportNode;
                            siblingImportNode = treeStack[siblingImportNode].prevSiblingImportNode;
                        }

                        if (!propertyOverridden && importerNode != -1)
                        {
                            if (treeStack[importerNode].parentNodeIn[propName])
                            {
                                propertyOverridden = true;
                            }
                        }

                        if (propertyOverridden)
                            continue;

                        emitOut << YAML::Key << propName << YAML::Value;

                        if (lwrrEntry.iteratorIntoChildrenIn->second.IsScalar())
                        {
                            emitOut << ExtendedName(lwrrEntry.iteratorIntoChildrenIn->second.Scalar(), lwrrEntry.iteratorIntoChildrenIn->second.Mark());
                        }
                        else if (lwrrEntry.iteratorIntoChildrenIn->second.IsSequence())
                        {
                            emitOut << YAML::BeginSeq << ExtendedName("", lwrrEntry.iteratorIntoChildrenIn->second.Mark());

                            lwrrElemIdx = int(treeStack.size());
                            treeStack.push_back({ lwrrEntry.iteratorIntoChildrenIn->second, YAML::const_iterator(), lwrrEntry.parentImportNode, -1 });
                            treeStack.back().iteratorIntoChildrenIn = lwrrEntry.iteratorIntoChildrenIn->second.begin();
                            ++lwrrEntry.iteratorIntoChildrenIn;
                        
                            added = true;
                            break;
                        }
                        else if (lwrrEntry.iteratorIntoChildrenIn->second.IsMap())
                        {
                            emitOut << YAML::BeginMap << YAML::Key << INFO_NODE_NAME << YAML::Value <<
                                            ExtendedName("", lwrrEntry.iteratorIntoChildrenIn->second.Mark());

                            lwrrElemIdx = int(treeStack.size());
                            treeStack.push_back({ lwrrEntry.iteratorIntoChildrenIn->second, YAML::const_iterator(), lwrrEntry.parentImportNode, -1 });
                            treeStack.back().iteratorIntoChildrenIn = lwrrEntry.iteratorIntoChildrenIn->second.begin();
                            ++lwrrEntry.iteratorIntoChildrenIn;

                            added = true;
                            break;
                        }
                        else
                        {
                            YAML::Emitter out;
                            out << lwrrEntry.parentNodeIn;

                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eEmptyMapNotAllowed, std::string(out.c_str()));
                        }
                    }

                    if (added)
                        continue;

                    unsigned int importNodeIdx = int(treeStack.size()) - lwrrElemIdx - 1;

                    YAML::Node import = lwrrEntry.parentNodeIn["import"];

                    bool needToGoUpStack = true;

                    if (import)
                    {
                        if (import.Type() != YAML::NodeType::Sequence)
                        {
                            YAML::Emitter out;
                            out << import;

                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eBadImportDirective, std::string(out.c_str()));
                        }

                        int importDirectiveIdx = int(import.size()) - importNodeIdx - 1;

                        if (importDirectiveIdx >= 0)
                        {
                            YAML::Node importEntry = import[importDirectiveIdx];

                            if (importEntry.Type() != YAML::NodeType::Scalar)
                            {
                                YAML::Emitter out;
                                out << import;

                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eBadImportDirective, std::string(out.c_str()));
                            }

                            std::string str = importEntry.as<std::string>();

                            if (str == "main" || str == "user-constants")
                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eIllegalParent, str);

                            YAML::Node parent_raw = root[str];

                            if (!parent_raw)
                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eParentNotFound, str);

                            if (!parent_raw.IsMap())
                            {
                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, parent_raw.Mark());
                            }

                            for (int prevImportParent = lwrrEntry.parentImportNode; prevImportParent != -1; prevImportParent = treeStack[prevImportParent].parentImportNode)
                            {
                                if (treeStack[prevImportParent].parentNodeIn == lwrrEntry.parentNodeIn)
                                {
                                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eRelwrsiveImport, str);
                                }
                            }

                            treeStack.push_back({ parent_raw, YAML::const_iterator(), static_cast<int>(lwrrElemIdx), static_cast<int>(treeStack.size()) - 1 });
                            treeStack.back().iteratorIntoChildrenIn = parent_raw.begin();

                            lwrrElemIdx = int(treeStack.size()) - 1;

                            needToGoUpStack = false;
                        }
                        else
                        {
                            assert(importDirectiveIdx == -1);
                        }
                    }

                    if (needToGoUpStack)
                    {
                        //erase import nodes
                        if (importNodeIdx)
                        {
                            treeStack.resize(lwrrElemIdx + 1);
                        }

                        if (lwrrEntry.prevSiblingImportNode != -1)
                        {
                            lwrrElemIdx = lwrrEntry.parentImportNode;
                        }
                        else
                        {
                            emitOut << YAML::EndMap;

                            assert(lwrrElemIdx == treeStack.size() - 1);
                            --lwrrElemIdx;
                            treeStack.pop_back();
                        }
                    }
                }
                else if (lwrrNodeType == YAML::NodeType::Sequence)
                {
                    bool added = false;

                    for (YAML::const_iterator end = lwrrEntry.parentNodeIn.end(); lwrrEntry.iteratorIntoChildrenIn != end; ++lwrrEntry.iteratorIntoChildrenIn)
                    {
                        if (lwrrEntry.iteratorIntoChildrenIn->IsScalar())
                        {
                            emitOut << ExtendedName(lwrrEntry.iteratorIntoChildrenIn->Scalar(), lwrrEntry.iteratorIntoChildrenIn->Mark());
                        }
                        else if (lwrrEntry.iteratorIntoChildrenIn->IsSequence())
                        {
                            emitOut << YAML::BeginSeq << ExtendedName("", lwrrEntry.iteratorIntoChildrenIn->Mark());

                            lwrrElemIdx = int(treeStack.size());
                            treeStack.push_back({ *(lwrrEntry.iteratorIntoChildrenIn), YAML::const_iterator(), lwrrEntry.parentImportNode, -1 });
                            treeStack.back().iteratorIntoChildrenIn = lwrrEntry.iteratorIntoChildrenIn->begin();
                            ++lwrrEntry.iteratorIntoChildrenIn;

                            added = true;
                            break;
                        }
                        else if (lwrrEntry.iteratorIntoChildrenIn->IsMap())
                        {
                            emitOut << YAML::BeginMap << YAML::Key << INFO_NODE_NAME << YAML::Value 
                                            << ExtendedName("", lwrrEntry.iteratorIntoChildrenIn->Mark());

                            lwrrElemIdx = int(treeStack.size());
                            treeStack.push_back({ *(lwrrEntry.iteratorIntoChildrenIn), YAML::const_iterator(), lwrrEntry.parentImportNode, -1 });
                            treeStack.back().iteratorIntoChildrenIn = lwrrEntry.iteratorIntoChildrenIn->begin();
                            ++lwrrEntry.iteratorIntoChildrenIn;

                            added = true;
                            break;
                        }
                        else
                        {
                            YAML::Emitter out;
                            out << lwrrEntry.parentNodeIn;

                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eEmptyMapNotAllowed, std::string(out.c_str()));
                        }
                    }

                    if (added)
                        continue;

                    emitOut << YAML::EndSeq;

                    assert(lwrrElemIdx == treeStack.size() - 1);
                    --lwrrElemIdx;
                    treeStack.pop_back();
                }
                else
                {
                    assert(false && "unexpected node type!");
                }
            }

            try
            {
                output = YAML::Load(emitOut.c_str());
            }
            catch (const YAML::Exception& e)
            {
                (void)e;
                assert(e.msg.c_str() && false);
            }

            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
        }

        MultipassConfigParserError EffectParser::validateFileForTabs(const wchar_t* filename)
        {
            using namespace std;

            ifstream file;
            file.open(filename, ios::in);

            bool tabfound = false;
            bool incomment = false;

            int line = 0, pos = 0;

            if (file.good())
            {
                file.seekg(0, ios::beg);

                char c = file.get();

                while (!file.eof())
                {
                    if (c == '#' && !incomment)
                        incomment = true;

                    if (c == '\n' && incomment)
                        incomment = false;

                    if (c == '\t' && !incomment)
                    {
                        tabfound = true;
                        break;
                    }
            
                    if (c == '\n')
                        pos = 0, ++line;
                    else
                        ++pos;

                    c = file.get();
                };
            
                if (tabfound)
                {
                    std::wstringbuf buf;
                    std::wostream strstream(&buf);
                    strstream << "line " << line << " position " << pos << " in file " << filename;
                
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eTabInYamlDetected, darkroom::getUtf8FromWstr(buf.str()));
                }

                file.close();
            }
            else
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eFileNotfound, darkroom::getUtf8FromWstr(filename));
            }

            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
        }


        MultipassConfigParserError EffectParser::parse(const wchar_t* filename)
        {
            MultipassConfigParserError errTabs = validateFileForTabs(filename);

            if (errTabs)
                return errTabs;
                
            YAML::Node config;

            try
            {
                std::ifstream fin(filename);
                
                if (!fin)
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eFileNotfound, darkroom::getUtf8FromWstr(filename));

                config = YAML::Load(fin);
            }
            catch (const YAML::Exception& e)
            {
                std::stringbuf buf;
                std::ostream strstream(&buf);
                strstream << "Yaml error " << e.msg.c_str() << " at position " << e.mark.pos << " line " << e.mark.line << " column " << e.mark.column;

                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eYAMLError, buf.str());
            }

            if (!config.IsMap())
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, config.Mark());
            }

            YAML::Node mainNode = config["main"];

            if (!mainNode)
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eMainNotFound);
            }

            if (!mainNode.IsMap())
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotMap, mainNode.Mark());
            }

            YAML::Node afterInheritance;
            MultipassConfigParserError inheritError = processImportPropertiesNoRelwrsion(afterInheritance, config, mainNode);

            if (inheritError)
                return inheritError;
            
#define DEBUG_OUTPUT_PROCESSED_YAML 1

#if DEBUG_OUTPUT_PROCESSED_YAML
            LOG_DEBUG(LogChannel::kYamlParser, "%s", (std::string("\n\nHere's the output YAML:\n\n") + (YAML::Emitter() << afterInheritance).c_str()).c_str());
#endif
            MultipassConfigParserError processErr = processDatasource(afterInheritance, m_mainOutputs);
        
            if (processErr)
                return processErr;

            YAML::Node userConstantsNode = config["user-constants"];

            if (!userConstantsNode || userConstantsNode.IsNull())
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
            }

            if (!userConstantsNode.IsSequence())
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eNodeNotSequence, userConstantsNode.Mark());
            }

            YAML::Node userConstsAfterInheritance;
            MultipassConfigParserError inheritError2 = processImportPropertiesNoRelwrsion(userConstsAfterInheritance, config, userConstantsNode);

            if (inheritError2)
                return inheritError2;

#define DEBUG_OUTPUT_PROCESSED_YAML 1

#if DEBUG_OUTPUT_PROCESSED_YAML
            LOG_DEBUG(LogChannel::kYamlParser, "%s", (std::string("\n\nHere's the output YAML:\n\n") + (YAML::Emitter() << userConstsAfterInheritance).c_str()).c_str());
#endif
            MultipassConfigParserError processErr2 = processUserConstantsNode(userConstsAfterInheritance);

            if (processErr2)
                return processErr2;
            
            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
        }
    }


}