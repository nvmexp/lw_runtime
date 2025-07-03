#pragma once

#include <string>
#include <iostream>
#include <set>
#include <map>
#include <vector>

#include "MultipassConfigParserError.h"
#include "ir/TypeEnums.h"


namespace YAML
{
    class Node;
}

namespace shadermod
{
    namespace effectParser
    {
        const unsigned int MAX_TEXTURES = 32;
        const unsigned int MAX_CBUFFERS = 32;
        const unsigned int MAX_CONSTS_IN_BUFFER = 128;
        const unsigned int MAX_SAMPLERS = 32;

        enum struct SurfaceType
        {
            kUndefined,
            kRGBA8_uint,
            kRGBA32_fp,
            kBGRA8_uint,
            kR10G10B10A2_uint,
            kR11G11B10_float,
            kSRGBA8_uint,
            kSBGRA8_uint,
            // Depth formats
            kD24S8,
            // Special values:
            kMatchInputColorChannel,
            kMatchInputDepthChannel,
            NUM_ENTRIES
        };

        ir::FragmentFormat colwertSurfaceTypeToIrFragmentFormat(const SurfaceType& ref, ir::FragmentFormat colorInputChanFmt, ir::FragmentFormat depthInputChanFmt);

        enum struct SystemChannelType
        {
            //TODO: fill up
            kPIPE_INPUTS_COLOR,
            kPIPE_INPUTS_DEPTH,
            kPIPE_INPUTS_HUDLESS,
            kPIPE_INPUTS_HDR,
            kPIPE_INPUTS_COLOR_BASE,
            NUM_ENTRIES
        };

        enum struct ShaderPassChannelType
        {
            kTARGET0_COLOR,
            kTARGET1_COLOR,
            kTARGET2_COLOR,
            kTARGET3_COLOR,
            kTARGET4_COLOR,
            kTARGET5_COLOR,
            kTARGET6_COLOR,
            kTARGET7_COLOR,
            NUM_ENTRIES
        };

        enum struct SamplerFilterType
        {
            kMinPtMagPtMipPt,
            kMinPtMagPtMipLin,
            kMinPtMagLinMipPt,
            kMinPtMagLinMipLin,
            kMinLinMagPtMipPt,
            kMinLinMagPtMipLin,
            kMinLinMagLinMipPt,
            kMinLinMagLinMipLin,
            NUM_ENTRIES
        };

        enum struct SamplerSubFilterType
        {
            kPOINT,
            kLINEAR,
            NUM_ENTRIES
        };

        void colwertSamplerFilterTypeToIrFilterType(const SamplerFilterType& ref, 
                    ir::FilterType& flt_min, ir::FilterType& flt_mag, ir::FilterType& flt_mip);

        SamplerFilterType combineSubFilterTypes(const SamplerSubFilterType& flt_min, 
                    const SamplerSubFilterType& flt_mag, const SamplerSubFilterType& flt_mip);
            
        enum struct SamplerAddressType
        {
            kWrap,
            kClamp,
            kMirror,
            NUM_ENTRIES
        };

        ir::AddressType colwertSamplerAddressTypeToIrSamplerAddressType(const SamplerAddressType& ref);

        enum struct SystemConstantsType
        {
            //TODO: fill up
            kDT,				// float
            kElapsedTime,		// float
            kFrame,				// int
            kScreenSize,		// float2
            kCaptureState,		// int
            kTileUV,			// float4
            kDepthAvailable,	// int
            kHDRAvailable,		// int
            kHUDlessAvailable,	// int
            NUM_ENTRIES
        };

        ir::ConstType colwertSystemConstantsTypeToIrConstType(const SystemConstantsType& ref);
        
        struct SamplerDesc
        {
            SamplerAddressType		m_adressU;
            SamplerAddressType		m_adressV;
            SamplerFilterType		m_filter;
        };

        template<typename EnumType, const char* strings[]>
        class StringToEnumMap
        {
        public:
            StringToEnumMap(const StringToEnumMap&) = delete;
            StringToEnumMap& operator=(const StringToEnumMap&) = delete;

            static const std::map<std::string, EnumType>& get()
            {
                static StringToEnumMap map;

                return map.m_map;
            }

            static EnumType colwert(const std::string& query, const EnumType& fallbackValue)
            {
                auto it = get().find(query);

                if (it == get().cend())
                    return fallbackValue;
                else
                    return it->second;
            }

        private:
            StringToEnumMap()
            {
                for (int i = 0; i < (int)EnumType::NUM_ENTRIES; ++i)
                {
                    m_map.insert(std::make_pair(std::string(strings[i]), (EnumType)i));
                }
            }

            std::map<std::string, EnumType> m_map;
        };

        //we need this to do the assertion before we lose the array size information
#define DECLARE_STRING_TO_ENUM_MAP(EnumType, strings, name) typedef  StringToEnumMap<EnumType, strings> name; \
    static_assert((sizeof(strings) / sizeof(char *)) == (int)EnumType::NUM_ENTRIES, "Parser map size mismatch");

        const char* UserConstOptionalLanguageTypeStr[];
        
        struct SamplerDescWrapperForSet
        {
            SamplerDescWrapperForSet(const std::vector<SamplerDesc>& primaryStrorage, size_t idx) :
            m_primaryStorage(primaryStrorage),
            m_index(idx)
            {
            }

            bool operator<(const SamplerDescWrapperForSet& ref) const;
        
            const std::vector<SamplerDesc>&					m_primaryStorage;
            const size_t									m_index;
        };
            
        enum struct CBufferPackOffsetSubcomponentType
        {
            eX,
            eY,
            eZ,
            eW,
            eNoPackOffset,
            NUM_ENTRIES
        };

        struct CBufferConstantBindingDesc
        {
            struct BindingKey
            {
                BindingKey(const std::string& name) : m_name(name), m_componentIdx(0),
                    m_subcomponentIdx(CBufferPackOffsetSubcomponentType::eNoPackOffset)
                {
                }

                BindingKey(unsigned int componentIdx, CBufferPackOffsetSubcomponentType subcomponentIdx): m_name(),
                    m_componentIdx(componentIdx), m_subcomponentIdx(subcomponentIdx)
                {
                }

                bool operator<(const BindingKey& ref) const;

                bool isBoundByName() const
                {
                    return m_subcomponentIdx == CBufferPackOffsetSubcomponentType::eNoPackOffset;
                }

                CBufferPackOffsetSubcomponentType		m_subcomponentIdx;
                unsigned int							m_componentIdx;
                std::string								m_name;
            };
            
    
            bool operator<(const CBufferConstantBindingDesc& ref) const;
            
            SystemConstantsType						m_variableIdx;
            std::string								m_userConstName;
        };

        struct CBufferDesc
        {
            std::map<CBufferConstantBindingDesc::BindingKey, CBufferConstantBindingDesc>	m_variableBindings;
        };

        struct CBufferDescWrapperForSet
        {
            CBufferDescWrapperForSet(const std::vector<CBufferDesc>& primaryStrorage, size_t idx) :
            m_primaryStorage(primaryStrorage),
            m_index(idx)
            {
            }

            bool operator<(const CBufferDescWrapperForSet& ref) const;
        
            const std::vector<CBufferDesc>&		m_primaryStorage;
            const size_t						m_index;
        };

        struct InputTextureDatasource
        {
            std::string			m_filename;
            bool				m_excludeHash;
            SurfaceType			m_type;
            unsigned int		m_width;
            unsigned int		m_height;
        };

        struct InputTextureDatasourceWrapperForSet
        {
            InputTextureDatasourceWrapperForSet(const std::vector<InputTextureDatasource>& primaryStrorage, size_t idx) :
            m_primaryStorage(primaryStrorage),
            m_index(idx)
            {

            }

            bool operator<(const InputTextureDatasourceWrapperForSet& ref) const;

            const std::vector<InputTextureDatasource>&		m_primaryStorage;
            const size_t									m_index;
        };

        enum struct ProceduralTextureType
        {
            eNoise,
            NUM_ENTRIES
        };

        struct ProceduralTextureDatasource
        {
            ProceduralTextureType		m_procedureType;
            SurfaceType					m_surfaceType;
            unsigned int				m_width;
            unsigned int				m_height;
        };

        struct ProceduralTextureDatasourceWrapperForSet
        {
            ProceduralTextureDatasourceWrapperForSet(const std::vector<ProceduralTextureDatasource>& primaryStrorage, size_t idx) :
            m_primaryStorage(primaryStrorage),
            m_index(idx)
            {

            }

            bool operator<(const ProceduralTextureDatasourceWrapperForSet& ref) const;

            const std::vector<ProceduralTextureDatasource>&		m_primaryStorage;
            const size_t										m_index;
        };


        struct SystemTextureDatasource
        {
            SystemTextureDatasource() : m_channelsReferenced(0)
            {

            }
            void referenceChannel(const SystemChannelType& channel);
            bool isChannelReferenced(const SystemChannelType& channel) const;

            size_t								m_channelsReferenced;
        };

        struct ShaderPassDatasource
        {
            ShaderPassDatasource() : m_channelsReferenced(0), m_height(0), m_width(0), m_scaleHeight(1.0f), m_scaleWidth(1.0f) {}
            void referenceChannel(const ShaderPassChannelType& channel, const SurfaceType& surfType);

            struct SystemInput
            {
                SystemChannelType				m_channel;
            };

            struct UserTextureInput
            {
                size_t							m_textureDatasource;
            };

            struct ProceduralTextureInput
            {
                size_t							m_procTexDatasource;
            };
            
            struct ShaderPassInput
            {
                size_t							m_shaderPassDatasource;
                ShaderPassChannelType			m_channel;
            };

            class TextureInput
            {
            public:
                enum TextureInputType { eUndefined, eSystemInput, eProceduralTextureInput, eUserTextureInput, eShaderPassInput };

                TextureInput();
                TextureInput(const SystemInput& ref);
                TextureInput(const ProceduralTextureInput& ref);
                TextureInput(const UserTextureInput& ref);
                TextureInput(const ShaderPassInput& ref);
                

                const SystemInput* getSystemInput() const;
                const ProceduralTextureInput* getProceduralTextureInput() const;
                const UserTextureInput* getUserTextureInput() const;
                const ShaderPassInput* getShaderPassInput() const;

                TextureInputType getType() const
                {
                    return m_type;
                }

                bool operator<(const ShaderPassDatasource::TextureInput& b) const;

            protected:
            
                bool compareSystemInput(const ShaderPassDatasource::SystemInput& a, const ShaderPassDatasource::SystemInput& b) const;
                bool compareProceduralTextureInput(const ShaderPassDatasource::ProceduralTextureInput& a, const ShaderPassDatasource::ProceduralTextureInput& b) const;
                bool compareUserTextureInput(const ShaderPassDatasource::UserTextureInput& a, const ShaderPassDatasource::UserTextureInput& b) const;
                bool compareShaderPassInput(const ShaderPassDatasource::ShaderPassInput& a, const ShaderPassDatasource::ShaderPassInput& b) const;
        
                union
                {
                    SystemInput			m_systemInput;
                    ProceduralTextureInput	m_proceduralTextureInput;
                    UserTextureInput	m_textureInput;
                    ShaderPassInput		m_shaderPassInput;
                };

                TextureInputType m_type;
            };

            struct BindingKey
            {
                static const unsigned int boundByName = 0xFFffFFff;
                
                unsigned int		m_slotIdx;
                std::string			m_name;

                bool isBoundByName() const
                {
                    return m_slotIdx == boundByName;
                }
            
                BindingKey(const std::string& name) : m_name(name), m_slotIdx(boundByName)
                {}

                BindingKey(unsigned int idx) : m_name(), m_slotIdx(idx)
                {}

                bool operator<(const BindingKey& ref) const;
            };
                        
            std::string										m_filename;
            std::string										m_shadername;
            unsigned int									m_width;
            unsigned int									m_height;
            float											m_scaleWidth;
            float											m_scaleHeight;
            
            std::map<BindingKey, size_t>					m_samplers;
            std::map<BindingKey, size_t>					m_cbuffers;
            std::map<BindingKey, TextureInput>				m_textures;

            unsigned int									m_channelsReferenced;
            std::vector<SurfaceType>						m_typesReferenced;
        };

        struct ShaderPassDatasourceWrapperForSet
        {
            ShaderPassDatasourceWrapperForSet(const std::vector<ShaderPassDatasource>& primaryStrorage, size_t idx) :
            m_primaryStorage(primaryStrorage),
            m_index(idx)
            {
            }

            bool operator<(const ShaderPassDatasourceWrapperForSet& ref) const;


            const std::vector<ShaderPassDatasource>&		m_primaryStorage;
            const size_t									m_index;
        };

        enum class UserConstDataType
        {
            kBool = 0,
            kInt,
            kUInt,
            kFloat,
            NUM_ENTRIES
        };

        enum class UserConstUiControlType
        {
            kSlider = 0,
            kCheckbox,
            kFlyout,
            kEditbox,
            NUM_ENTRIES
        };

        ir::UserConstDataType colwertUserConstDataTypeToIrUserConstType(const UserConstDataType& arg);
        
        union FlexibleDataType
        {
            ir::userConstTypes::Bool boolValue;
            ir::userConstTypes::Int intValue;
            ir::userConstTypes::UInt uintValue;
            ir::userConstTypes::Float floatValue;
        };
        
        enum struct UserConstOptionalLanguageType
        {
            k_de_DE = 0,
            k_es_ES,
            k_es_MX,
            k_fr_FR,
            k_it_IT,
            k_ru_RU,
            k_zh_CHS,
            k_zh_CHT,
            k_ja_JP,
            k_cs_CZ,
            k_da_DK,
            k_el_GR,
            k_en_UK,
            k_fi_FI,
            k_hu,
            k_ko_KR,
            k_nl_NL,
            k_nb_NO,
            k_pl,
            k_pt_PT,
            k_pt_BR,
            k_sl_SI,
            k_sk_SK,
            k_sv_SE,
            k_th_TH,
            k_tr_TR,
            NUM_ENTRIES
        };

        unsigned short colwertUserConstOptionalLanguageTypeToLANGID(const UserConstOptionalLanguageType& ref);

        struct UserConstLocalizedString
        {
            std::string str[(int) UserConstOptionalLanguageType::NUM_ENTRIES];

            void clear()
            {
                for (std::string* ps = str, * ends = str + (int)UserConstOptionalLanguageType::NUM_ENTRIES; ps < ends; ++ps)
                    ps->resize(0);
            }
        };

        struct UserConstDesc
        {
            enum class EParams: unsigned int
            {
                kName =0, //+
                kType, //+
                kUiControl, // +
                kDefaultValue, // +
                kMinimumValue, //+
                kMaximumValue, // +
                kUiValueStep, //+
                kStickyValue, //+
                kStickyRegion, //+
                kOptions, // +
                kDefaultOption, //+
                kUiLabel, //+
                kUiHint, //+
                kUiLabelLocalized, //+
                kUiHintLocalized, //+
                kUiValueUnit, //+
                kUiValueUnitLocalized, //+
                kUiValueMin, //+
                kUiValueMax, //+
                NUM_ENTRIES
            };
            
            struct ListOption
            {
                FlexibleDataType			value;
                std::string					name;
                UserConstLocalizedString	nameLocalized;
            };

            std::string						name;
            UserConstDataType				type;
            UserConstUiControlType			uiControlType;
            FlexibleDataType				defaultValue;
            FlexibleDataType				minimumValue;
            FlexibleDataType				maximumValue;
            FlexibleDataType				uiValueStep;
            float							stickyValue;
            float							stickyRegion;
            std::vector<ListOption>			options;
            int								defaultOption; //-1 stands for "invalid" when there are no options at all
            std::string						uiLabel;
            std::string						uiHint;
            UserConstLocalizedString		uiLabelLocalized;
            UserConstLocalizedString		uiHintLocalized;
            std::string						uiValueUnit;
            UserConstLocalizedString		uiValueUnitLocalized;
            FlexibleDataType				uiValueMin;
            FlexibleDataType				uiValueMax;
        };

        class EffectParser
        {
        public:
            
            EffectParser()
            {
            }

            static MultipassConfigParserError validateFileForTabs(const wchar_t* filename);
        
            MultipassConfigParserError parse(const wchar_t* filename);

            const std::vector<ShaderPassDatasource>& getShaderPassDatasources() const
            {
                return m_shaderPassDatasources;
            }
                
            const std::vector<InputTextureDatasource>& getInputTextureDatasources() const
            {
                return m_inputTextureDatasources;
            }

            const std::vector<ProceduralTextureDatasource>& getProceduralTextureDatasource() const
            {
                return m_proceduralTextureDatasources;
            }

            const SystemTextureDatasource& getSystemDatasource() const
            {
                return m_systemDatasource;
            }

            const std::vector<SamplerDesc>& getSamplers() const
            {
                return m_samplers;
            }
            
            const std::vector<CBufferDesc>& getCBuffers() const
            {
                return m_cbuffers;
            }

            const std::vector<UserConstDesc>& getUserConstants() const
            {
                return m_userConstDescs;
            }

            const 	ShaderPassDatasource::TextureInput& getMainOututs() const
            {
                return m_mainOutputs;
            }
            
        protected:
            static MultipassConfigParserError readValueFromNode(const YAML::Node& node, FlexibleDataType& val, UserConstDataType type);
            static std::string stringifyValue(const FlexibleDataType& value, UserConstDataType type);
            static bool compareValueLess(const FlexibleDataType& left, const FlexibleDataType& right, UserConstDataType type);
            static bool compareValueMore(const FlexibleDataType& left, const FlexibleDataType& right, UserConstDataType type);
            static bool compareValueEqual(const FlexibleDataType& left, const FlexibleDataType& right, UserConstDataType type);
            static FlexibleDataType getZeroValue(UserConstDataType type);

            static MultipassConfigParserError readLocalizedStringFromNode(const YAML::Node& node, UserConstLocalizedString& val);
            static MultipassConfigParserError readMinimumValueNodeHelper(const YAML::Node& node, const UserConstDataType& type, FlexibleDataType& value, bool& isUnbounded);
            static MultipassConfigParserError readMaximumValueNodeHelper(const YAML::Node& node, const UserConstDataType& type, FlexibleDataType& value, bool& isUnbounded);
            static MultipassConfigParserError readOptionsNodeHelper(const YAML::Node& node, UserConstDesc& desc);

            static YAML::Mark	getMarkFromNode(const YAML::Node& node);

            MultipassConfigParserError processSamplerState(const YAML::Node& samplerNode, size_t& out);
            MultipassConfigParserError processCBuffer(const YAML::Node& cbufferNode, size_t& out);
            MultipassConfigParserError processDatasource(const YAML::Node& datasourceNode, ShaderPassDatasource::TextureInput& out);
                                    
            MultipassConfigParserError processUserConstantsNode(const YAML::Node& userConstantsNode);
            
            template<typename DatasourceT> MultipassConfigParserError readWidthHeight(const YAML::Node& datasourceNode, DatasourceT& datasrc);
            std::pair<size_t, bool> insertShaderPassDatasource(const ShaderPassDatasource& ref);
            std::pair<size_t, bool> insertInputTextureDatasource(const InputTextureDatasource& ref);
            std::pair<size_t, bool> insertProceduralTextureDatasource(const ProceduralTextureDatasource& ref);

            std::pair<size_t, bool> insertSamplerDesc(const SamplerDesc& ref);
            std::pair<size_t, bool> insertCBufferDesc(const CBufferDesc& ref);
            
#define DEBUG_IMPORT_PROPERTY 0 
        
            static MultipassConfigParserError processImportPropertiesNoRelwrsion(YAML::Node& output, const YAML::Node& root, const YAML::Node& main);


            std::vector<ShaderPassDatasource>				m_shaderPassDatasources;
            std::set<ShaderPassDatasourceWrapperForSet>		m_shaderPassDatasourcesSet;

            std::vector<InputTextureDatasource>				m_inputTextureDatasources;
            std::set<InputTextureDatasourceWrapperForSet>	m_inputTextureDatasourcesSet;

            std::vector<ProceduralTextureDatasource>		m_proceduralTextureDatasources;
            std::set<ProceduralTextureDatasourceWrapperForSet>	m_proceduralTextureDatasourcesSet;

            SystemTextureDatasource							m_systemDatasource;

            std::vector<SamplerDesc>						m_samplers;
            std::set<SamplerDescWrapperForSet>				m_samplersSet;

            std::vector<CBufferDesc>						m_cbuffers;
            std::set<CBufferDescWrapperForSet>				m_cbuffersSet;

            ShaderPassDatasource::TextureInput				m_mainOutputs;

            std::vector<UserConstDesc>						m_userConstDescs;
        };
    }
}