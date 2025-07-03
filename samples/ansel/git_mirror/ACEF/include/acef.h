#pragma once

#include <vector>
#include <unordered_map>
#include <assert.h>

const uint64_t compilerMagicWordAndVersion = 0xFEBECAF000000005ull;

namespace acef
{

#pragma pack(push, 1)
#pragma warning(disable : 4505)

/*
IMPORTANT!! If ACEF file structure is modified, do not forget to update `eraseTimestamps` function

Most structs contain two parts: binary serializeable data and metadata.
Binary serializeable data contains only simple structs (or complex structs with binary/metadata splitting) whose values will be written to file by value
Metadata contains pointers and other data that doesn't make sense to store

Structures that have 'Storage' in name must not contain any metadata, since they could be included in serialization hierarchy
*/

enum class CaptureKinds : uint32_t
{
    kREGULAR = 0,
    kHIGHRES = 1,
    k360 = 2,
    kCYLINDRIC = 3,

    // Modifiers
    kSTEREO = 4,
    kRAWHDR = 5,

    kNUM_ENTRIES = 6
};

// Levels of support for certain capture type (?) for an effect
enum class CompatibilityLevel : uint32_t
{
    kFULL = 0,
    kPARTIAL = 1,
    kNONE = 2,

    kNUM_ENTRIES = 3
};

struct Header
{
    static const uint8_t compilerStringLen = 16;
    static const uint8_t hashStringLen = 16;
    static const uint8_t reservedBytes = 128;
    static const uint32_t totalCaptureTypes = (uint32_t)CaptureKinds::kNUM_ENTRIES;

    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint64_t) +
                sizeof(uint64_t) +
                compilerStringLen * sizeof(char) +
                hashStringLen * sizeof(char) +
                reservedBytes * sizeof(uint8_t) +
                totalCaptureTypes * sizeof(uint32_t) +
                sizeof(uint64_t) +

                sizeof(uint32_t) +

                sizeof(uint64_t) +
                sizeof(uint64_t) +
                sizeof(uint64_t)
                ;

            assert(manualSize == sizeof(Header::BinaryStorage));

            return manualSize;
        }

        uint64_t magicWord = 0;
        uint64_t version = 0;
        char compiler[compilerStringLen] = {0};
        char hash[hashStringLen] = { 0 };
        uint8_t reserved[reservedBytes] = { 0 };
        uint32_t captureCompatibility[totalCaptureTypes] = { 0 };
        uint64_t timestamp = 0;

        uint32_t dependenciesNum = 0;

        // Byte offsets from the beginning of the file
        uint64_t resourcesChunkByteOffset = 0;
        uint64_t uiControlsChunkByteOffset = 0;
        uint64_t passesChunkByteOffset = 0;
    } binStorage;

    uint64_t * fileTimestamps = nullptr;
    uint16_t * filePathLens = nullptr;
    uint32_t * filePathOffsets = nullptr;
    char * filePathsUtf8 = nullptr;     // relative to the binary root, or special macros
};

enum class TextureSizeBase : uint32_t
{
    kOne                        = 0,    // 1, constant to set custom dimensions
    kColorBufferWidth           = 1,
    kColorBufferHeight          = 2,
    kDepthBufferWidth           = 3,
    kDepthBufferHeight          = 4,
    kTextureWidth               = 5,    // Applicable only to textureFromFile
    kTextureHeight              = 6     // Applicable only to textureFromFile
};

// This struct will allow to set relative texture sizes
// e.g. if we want a texture to be 25% of the framebuffer size
struct TextureSizeStorage
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(float) +
                sizeof(uint32_t);

            assert(manualSize == sizeof(TextureSizeStorage::BinaryStorage));

            return manualSize;
        }

        float mul;
        TextureSizeBase texSizeBase;
    } binStorage;
};


// ACEF Resource Chunk
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

enum class SystemTexture : uint32_t
{
    // All textures with handle < kSystemTextureBase are treated as indices into all three texture types, in succession:
    //    - 0 .. (texturesParametrizedNum-1) -- textureParametrized
    //    - texturesParametrizedNum .. (texturesParametrizedNum+texturesIntermediateNum-1) -- textureIntermediate
    //    - texturesParametrizedNum+texturesIntermediateNum .. (texturesParametrizedNum+texturesIntermediateNum+texturesFromFileNum-1) -- textureFromFile

    kSystemTextureBase = 0x8000000,
    kInputColor         = kSystemTextureBase,
    kInputDepth         = kSystemTextureBase + 1,
    kInputHUDless       = kSystemTextureBase + 2,
    kInputHDR           = kSystemTextureBase + 3,
    kInputColorBase     = kSystemTextureBase + 4,
};

// This block should be followed up by the blob of handles, and byteOffsets
class ResourceHeader
{
public:

    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint32_t) +
                sizeof(uint32_t) +
                sizeof(uint32_t) +
                sizeof(uint32_t) +
                sizeof(uint32_t) +
                sizeof(uint32_t) +

                sizeof(uint32_t) +
                sizeof(uint32_t) +
                sizeof(uint32_t);

            assert(manualSize == sizeof(ResourceHeader::BinaryStorage));

            return manualSize;
        }

        uint32_t readBuffersNum;        // SRVs
        uint32_t writeBuffersNum;        // RTVs
        uint32_t pixelShadersNum;
        uint32_t vertexShadersNum;
        uint32_t samplersNum;
        uint32_t constantBuffersNum;

        uint32_t texturesParametrizedNum;
        uint32_t texturesIntermediateNum;
        uint32_t texturesFromFileNum;
    } binStorage;

    // This is a special case of integrated resource description
    //    we consider read/write buffers inherit their fmt from the underlying texture resources
    //    hence all is needed is a pointer (index) to the texture resource

    // Handle is because below system texture base they are treated as indices for the user textures
    //    and when above system texture base, they are treated as unique Ansel-provided system textures

    // Texture index is overall index in all three texture types, in succession: texturesParametrizedNum+texturesIntermediateNum+texturesFromFileNum
    uint32_t * writeBufferTextureHandles;
    uint32_t * readBufferTextureHandles;

    uint64_t * pixelShaderByteOffsets;
    uint64_t * vertexShaderByteOffsets;

    uint64_t * samplerByteOffsets;
    uint64_t * constantBufferByteOffsets;

    uint64_t * textureParametrizedByteOffsets;
    uint64_t * textureIntermediateByteOffsets;
    uint64_t * textureFromFileByteOffsets;
};

// This block should be followed up by blob of filePath and entryFunction
class ResourcePixelShader
{
public:

    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint32_t) +
                sizeof(uint32_t);

            assert(manualSize == sizeof(ResourcePixelShader::BinaryStorage));

            return manualSize;
        }

        uint32_t filePathLen;
        uint32_t entryFunctionLen;
    } binStorage;

    char * filePathUtf8;        // relative to the binary root
    char * entryFunctionAscii;
};

// This block should be followed up by blob of filePath and entryFunction
class ResourceVertexShader
{
public:

    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint32_t) +
                sizeof(uint32_t);

            assert(manualSize == sizeof(ResourceVertexShader::BinaryStorage));

            return manualSize;
        }

        uint32_t filePathLen;
        uint32_t entryFunctionLen;
    } binStorage;

    char * filePathUtf8;        // relative to the binary root
    char * entryFunctionAscii;
};

// Texture resource
/////////////////////////////////////////////////////////////////////////////////
enum class FragmentFormat : uint32_t
{
    kRGBA8_uint = 0,
    kBGRA8_uint,
    kRGBA16_uint,
    kRGBA16_fp,
    kRGBA32_fp,

    kSRGBA8_uint,
    kSBGRA8_uint,

    kR10G10B10A2_uint,

    kR11G11B10_float,

    kRG8_uint,
    kRG16_uint,
    kRG16_fp,
    kRG32_uint,
    kRG32_fp,

    kR8_uint,
    kR16_uint,
    kR16_fp,
    kR32_uint,
    kR32_fp,

    // Depth formats
    kD24S8,
    kD32_fp_S8X24_uint,
    kD32_fp,

    kNUM_ENTRIES
};

enum class ResourceTextureParametrizedType : uint32_t
{
    kNOISE = 0
};

const uint32_t textureParametrizedNumParameters = 4;
struct ResourceTextureParametrized
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                TextureSizeStorage::BinaryStorage::storageByteSize() +
                TextureSizeStorage::BinaryStorage::storageByteSize() +
                sizeof(uint32_t) +
                sizeof(uint32_t) +
                4*sizeof(float);

            assert(manualSize == sizeof(ResourceTextureParametrized::BinaryStorage));

            return manualSize;
        }

        TextureSizeStorage width;
        TextureSizeStorage height;
        FragmentFormat format;
        ResourceTextureParametrizedType type;
        float parameters[textureParametrizedNumParameters];
    } binStorage;
};

struct ResourceTextureIntermediate
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                TextureSizeStorage::BinaryStorage::storageByteSize() + // width
                TextureSizeStorage::BinaryStorage::storageByteSize() + // height
                sizeof(uint32_t) + // format
                sizeof(uint32_t); // levels

            assert(manualSize == sizeof(ResourceTextureIntermediate::BinaryStorage));

            return manualSize;
        }

        TextureSizeStorage width;
        TextureSizeStorage height;
        FragmentFormat format;
        uint32_t levels;
    } binStorage;
};

// This block should be followed up by blob of path
struct ResourceTextureFromFile
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                TextureSizeStorage::BinaryStorage::storageByteSize() +
                TextureSizeStorage::BinaryStorage::storageByteSize() +
                sizeof(uint32_t) +
                sizeof(uint32_t) +
                sizeof(bool);

            assert(manualSize == sizeof(ResourceTextureFromFile::BinaryStorage));

            return manualSize;
        }

        TextureSizeStorage width;
        TextureSizeStorage height;
        FragmentFormat format;

        uint32_t pathLen;
        bool excludeHash;
    } binStorage;

    char * pathUtf8;        // relative to the binary root
};
/////////////////////////////////////////////////////////////////////////////////


enum class ResourceSamplerAddressType : uint32_t
{
    kWrap = 0,
    kClamp = 1,
    kMirror = 2,
    kBorder = 3
};

enum class ResourceSamplerFilterType : uint32_t
{
    kPoint = 0,
    kLinear = 1
};

struct ResourceSampler
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint32_t) +
                sizeof(uint32_t) +
                sizeof(uint32_t) +

                sizeof(uint32_t) +
                sizeof(uint32_t) +
                sizeof(uint32_t);

            assert(manualSize == sizeof(ResourceSampler::BinaryStorage));

            return manualSize;
        }

        ResourceSamplerAddressType addrU;
        ResourceSamplerAddressType addrV;
        ResourceSamplerAddressType addrW;

        ResourceSamplerFilterType filterMin;
        ResourceSamplerFilterType filterMag;
        ResourceSamplerFilterType filterMip;
    } binStorage;
};


enum class SystemConstant : uint32_t
{
    // All constants with handle < kSystemConstantBase are treated as indices into UserConstants

    kSystemConstantBase = 0x8000000,
    kDT = kSystemConstantBase,                      // float
    kElapsedTime = kSystemConstantBase + 1,         // float
    kFrame = kSystemConstantBase + 2,               // int
    kScreenSize = kSystemConstantBase + 3,          // float2
    kCaptureState = kSystemConstantBase + 4,        // int
    kTileUV = kSystemConstantBase + 5,              // float4
    kDepthAvailable = kSystemConstantBase + 6,      // int
    kHDRAvailable = kSystemConstantBase + 7,        // int
    kHUDlessAvailable = kSystemConstantBase + 8,    // int
};

inline
bool isConstantSystem(uint32_t constHandle)
{
    return (constHandle >= (uint32_t)SystemConstant::kSystemConstantBase);
}

// This block should be followed up by the constant offsets and handles
struct ResourceConstantBuffer
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint32_t);

            assert(manualSize == sizeof(ResourceConstantBuffer::BinaryStorage));

            return manualSize;
        }

        uint32_t constantsNum;
    } binStorage;

    uint32_t *    constantOffsetInComponents;
    uint16_t *    constantNameLens;
    uint32_t *    constantNameOffsets;
    char  *       constantNames;

    uint32_t * constantHandle;
};

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


// ACEF UI Controls Chunk
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

// This block should be followed up by byteOffsetUserConstants
struct UIControlsHeader
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint32_t);

            assert(manualSize == sizeof(UIControlsHeader::BinaryStorage));

            return manualSize;
        }

        uint32_t userConstantsNum;
    } binStorage;

    uint64_t * userConstantByteOffsets;
};

// This block should be followed up by localizedStrings and then by all utf8String buffers
struct UILocalizedStringStorage
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint16_t) +
                sizeof(uint16_t);

            assert(manualSize == sizeof(UILocalizedStringStorage::BinaryStorage));

            return manualSize;
        }

        uint16_t localizationsNum;
        uint16_t defaultStringLen;
    } binStorage;
};

struct UILocalizedStringBuffers
{
    struct LocalizedString
    {
        struct BinaryStorage
        {
            static size_t storageByteSize()
            {
                const size_t manualSize =
                    sizeof(uint16_t) +
                    sizeof(uint16_t);

                assert(manualSize == sizeof(LocalizedString::BinaryStorage));

                return manualSize;
            }

            uint16_t langid;
            uint16_t strLen;
        } binStorage;

        char * stringUtf8;
    };

    char * defaultStringAscii;
    LocalizedString * localizedStrings;
};

enum class UserConstDataType : uint32_t
{
    kBool = 0,
    kInt = 1,
    kUInt = 2,
    kFloat = 3
};

enum class UIControlType : uint32_t
{
    kSlider = 0,
    kCheckbox = 1,
    kFlyout = 2,
    kEditbox = 3,
    kColorPicker = 4,
    kRadioButton = 5
};

// TODO avoroshilov ACEF: add dimension? e.g. (type==float, dim=4) == float4

struct TypelessVariableStorage
{
    static const uint32_t numBytes = 4 * 4;
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                numBytes*sizeof(uint8_t);

            assert(manualSize == sizeof(TypelessVariableStorage::BinaryStorage));

            return manualSize;
        }

        // Lwrrently all the data fits into 4*4 bytes
        // This data should be reinterpretable starting as float/bool/int/uint up to float4/...
        uint8_t data[numBytes] = { 0 };
    } binStorage;
};

// This block should be followed up by variableNames, optionNames, optiolwalues, label/hint/uiValueUnit localized string blocks and controlNameAscii buffer
struct UserConstant
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint32_t) +

                UILocalizedStringStorage::BinaryStorage::storageByteSize() +
                UILocalizedStringStorage::BinaryStorage::storageByteSize() +
                UILocalizedStringStorage::BinaryStorage::storageByteSize() +

                TypelessVariableStorage::BinaryStorage::storageByteSize() + 
                TypelessVariableStorage::BinaryStorage::storageByteSize() + 
                TypelessVariableStorage::BinaryStorage::storageByteSize() + 

                TypelessVariableStorage::BinaryStorage::storageByteSize() + 
                TypelessVariableStorage::BinaryStorage::storageByteSize() + 
                TypelessVariableStorage::BinaryStorage::storageByteSize() + 

                sizeof(float) +
                sizeof(float) +

                sizeof(uint16_t) +
                sizeof(uint16_t) +

                sizeof(uint32_t) +
                sizeof(uint32_t) +

                sizeof(uint8_t);

            assert(manualSize == sizeof(UserConstant::BinaryStorage));

            return manualSize;
        }

        uint32_t controlNameLen;

        UILocalizedStringStorage label;
        UILocalizedStringStorage hint;
        UILocalizedStringStorage uiValueUnit;

        TypelessVariableStorage defaultValue;
        TypelessVariableStorage minimumValue;
        TypelessVariableStorage maximumValue;

        TypelessVariableStorage uiMinimumValue;
        TypelessVariableStorage uiMaximumValue;
        TypelessVariableStorage uiValueStep;        // zero means maximum sensitivity (infinitesimal step size)

        float stickyValue;
        float stickyRegion;

        // For e.g. flyouts or other switchers
        uint16_t optionsNum;
        uint16_t optionDefault;

        UIControlType uiControlType;
        UserConstDataType dataType;
        uint8_t dataDimensionality;
    } binStorage = { 0 };

    char * controlNameAscii;

    UILocalizedStringBuffers labelBuffers;
    UILocalizedStringBuffers hintBuffers;
    UILocalizedStringBuffers uiValueUnitBuffers;

    TypelessVariableStorage * optiolwalues;

    uint64_t * optionNameByteOffsets;

    UILocalizedStringStorage * optionNames;
    UILocalizedStringBuffers * optionNamesBuffers;

    uint64_t * variableNameByteOffsets;

    UILocalizedStringStorage * variableNames;
    UILocalizedStringBuffers * variableNamesBuffers;
};

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


// ACEF Passes Chunk
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

// This block should be followed up by byteOffsetPasses
struct PassesHeader
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint32_t);

            assert(manualSize == sizeof(PassesHeader::BinaryStorage));

            return manualSize;
        }

        uint32_t passesNum;
    } binStorage;

    uint64_t * passByteOffsets;
};

enum class RasterizerFillMode : uint32_t
{
    kWireframe    = 2,
    kSolid    = 3
};

enum class RasterizerLwllMode : uint32_t
{
    kNone = 1,
    kFront = 2,
    kBack = 3
};

struct RasterizerStateStorage
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint32_t) +
                sizeof(uint32_t) + 

                sizeof(int32_t) + 
                sizeof(float) + 
                sizeof(float) +
                
                sizeof(uint8_t) +
                sizeof(uint8_t) +
                sizeof(uint8_t) +
                sizeof(uint8_t) +
                sizeof(uint8_t);

            assert(manualSize == sizeof(RasterizerStateStorage::BinaryStorage));

            return manualSize;
        }


        //D3D11_RASTERIZER_DESC

        RasterizerFillMode fillMode;
        RasterizerLwllMode lwllMode;

        int32_t depthBias;
        float depthBiasClamp;
        float slopeScaledDepthBias;

        uint8_t frontCounterClockwise;
        uint8_t depthClipEnable;
        uint8_t scissorEnable;
        uint8_t multisampleEnable;
        uint8_t antialiasedLineEnable;
    } binStorage;
};

enum class DepthWriteMask : uint32_t
{
    kZero = 0,
    kAll = 1
};

enum class ComparisonFunc : uint32_t
{
    kNever = 1,
    kLess = 2,
    kEqual = 3,
    kLessEqual = 4,
    kGreater = 5,
    kNotEqual = 6,
    kGreaterEqual = 7,
    kAlways = 8
};

enum class StencilOp : uint32_t
{
    kKeep = 1,
    kZero = 2,
    kReplace = 3,
    kIncrSat = 4,
    kDecrSat = 5,
    kIlwert = 6,
    kIncr = 7,
    kDecr = 8
};

struct DepthStencilOpStorage
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint32_t) +
                sizeof(uint32_t) + 
                sizeof(uint32_t) + 
                sizeof(uint32_t);

            assert(manualSize == sizeof(DepthStencilOpStorage::BinaryStorage));

            return manualSize;
        }

        StencilOp failOp;
        StencilOp depthFailOp;
        StencilOp passOp;
        ComparisonFunc func;
    } binStorage;
};

struct DepthStencilStateStorage
{
    static const uint8_t defaultStencilReadMask = 0xFF;        //D3D11_DEFAULT_STENCIL_READ_MASK
    static const uint8_t defaultStencilWriteMask = 0xFF;    //D3D11_DEFAULT_STENCIL_WRITE_MASK

    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                DepthStencilOpStorage::BinaryStorage::storageByteSize() +
                DepthStencilOpStorage::BinaryStorage::storageByteSize() +

                sizeof(uint32_t) + 
                sizeof(uint32_t) + 

                sizeof(uint8_t) +
                sizeof(uint8_t) + 
                sizeof(uint8_t) + 
                sizeof(uint8_t);

            assert(manualSize == sizeof(DepthStencilStateStorage::BinaryStorage));

            return manualSize;
        }
        //D3D11_DEPTH_STENCIL_DESC

        DepthStencilOpStorage frontFace;
        DepthStencilOpStorage backFace;

        DepthWriteMask depthWriteMask;
        ComparisonFunc depthFunc;

        uint8_t stencilReadMask;
        uint8_t stencilWriteMask;
        uint8_t isDepthEnabled;
        uint8_t isStencilEnabled;
    } binStorage;
};

enum class BlendCoef : uint32_t
{
    kZero = 1,
    kOne = 2,
    kSrcColor = 3,
    kIlwSrcColor = 4,
    kSrcAlpha = 5,
    kIlwSrcAlpha = 6,
    kDstAlpha = 7,
    kIlwDstAlpha = 8,
    kDstColor = 9,
    kIlwDstColor = 10,
    kSrcAlphaSat = 11,
    kBlendFactor = 14,
    kIlwBlendFactor = 15,
    kSrc1Color = 16,
    kIlwSrc1Color = 17,
    kSrc1Alpha = 18,
    kIlwSrc1Alpha = 19
};

enum class BlendOp : uint32_t
{
    kAdd = 1,
    kSub = 2,
    kRevSub = 3,
    kMin = 4,
    kMax = 5
};

enum class ColorWriteEnableBits : uint32_t
{
    kRed = 1,
    kGreen = 2,
    kBlue = 4,
    kAlpha = 8,
    kAll = ( ((kRed|kGreen)  | kBlue)  | kAlpha )
};

struct AlphaBlendRenderTargetStateStorage
{
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                sizeof(uint32_t) + 
                sizeof(uint32_t) + 
                sizeof(uint32_t) +

                sizeof(uint32_t) + 
                sizeof(uint32_t) + 
                sizeof(uint32_t) +

                sizeof(uint32_t) + 
                sizeof(uint8_t);

            assert(manualSize == sizeof(AlphaBlendRenderTargetStateStorage::BinaryStorage));

            return manualSize;
        }

        BlendCoef src;
        BlendCoef dst;
        BlendOp op;

        BlendCoef srcAlpha;
        BlendCoef dstAlpha;
        BlendOp opAlpha;

        ColorWriteEnableBits renderTargetWriteMask;
        uint8_t isEnabled;
    } binStorage;
};

struct AlphaBlendStateStorage
{
    static const uint8_t renderTargetsNum = 8;
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                AlphaBlendStateStorage::renderTargetsNum*AlphaBlendRenderTargetStateStorage::BinaryStorage::storageByteSize() + 
                sizeof(uint8_t) + 
                sizeof(uint8_t);

            assert(manualSize == sizeof(AlphaBlendStateStorage::BinaryStorage));

            return manualSize;
        }

        //D3D11_BLEND_DESC

        //D3D11_RENDER_TARGET_BLEND_DESC RenderTarget[ 8 ];
        AlphaBlendRenderTargetStateStorage renderTargetState[ renderTargetsNum ];

        uint8_t alphaToCoverageEnable;
        uint8_t independentBlendEnable;
    } binStorage;
};


// This block should be followed by all the arrays (slots, indices for each resource type)
struct Pass
{
    static const uint32_t resourceBindByName = (uint32_t)-1;
    static const uint32_t vertexShaderDefault = (uint32_t)-1;
    struct BinaryStorage
    {
        static size_t storageByteSize()
        {
            const size_t manualSize =
                RasterizerStateStorage::BinaryStorage::storageByteSize() + 
                AlphaBlendStateStorage::BinaryStorage::storageByteSize() + 
                DepthStencilStateStorage::BinaryStorage::storageByteSize() + 

                TextureSizeStorage::BinaryStorage::storageByteSize() +
                TextureSizeStorage::BinaryStorage::storageByteSize() +

                sizeof(uint32_t) + 
                sizeof(uint32_t) + 

                sizeof(uint32_t) + 
                sizeof(uint32_t) + 
                sizeof(uint32_t) + 
                sizeof(uint32_t) + 
                sizeof(uint32_t);

            assert(manualSize == sizeof(Pass::BinaryStorage));

            return manualSize;
        }

        RasterizerStateStorage rasterizerState;
        AlphaBlendStateStorage alphaBlendState;
        DepthStencilStateStorage depthStencilState;

        // If we have output resource attached, these parameters should be <= output resource's to avoid clipping
        TextureSizeStorage width;
        TextureSizeStorage height;

        uint32_t pixelShaderIndex;
        uint32_t vertexShaderIndex;

        uint32_t readBuffersNum;        // "SRVs"
        uint32_t writeBuffersNum;        // MRT
        uint32_t constantBuffersVSNum;
        uint32_t constantBuffersPSNum;
        uint32_t samplersNum;
    } binStorage;

    uint32_t *  readBuffersSlots;
    uint16_t *  readBuffersNameLens;
    uint32_t *  readBuffersNameOffsets;
    char *    readBuffersNames;
    uint32_t *  readBuffersIndices;
    uint32_t *  writeBuffersSlots;
    uint16_t *  writeBuffersNameLens;
    uint32_t *  writeBuffersNameOffsets;
    char *    writeBuffersNames;
    uint32_t *  writeBuffersIndices;
    uint32_t *  constantBuffersVSSlots;
    uint16_t *  constantBuffersVSNameLens;
    uint32_t *  constantBuffersVSNameOffsets;
    char *    constantBuffersVSNames;
    uint32_t *  constantBuffersVSIndices;
    uint32_t *  constantBuffersPSSlots;
    uint16_t *  constantBuffersPSNameLens;
    uint32_t *  constantBuffersPSNameOffsets;
    char *    constantBuffersPSNames;
    uint32_t *  constantBuffersPSIndices;
    uint32_t *  samplersSlots;
    uint16_t *  samplersNameLens;
    uint32_t *  samplersNameOffsets;
    char *    samplersNames;
    uint32_t *  samplersIndices;
};

#pragma pack(pop)


class EffectStorage
{
public:
    Header header;
    ResourceHeader resourceHeader;
    UIControlsHeader uiControlsHeader;
    PassesHeader passesHeader;

    uint64_t totalByteSize;

    std::vector<void *> allocationsArray;
    void * allocateMem(size_t byteSize)
    {
        void * rawMem = malloc(byteSize);
        memset(rawMem, 0, byteSize); // Make sure all memory allocated for the file is zeroed out in order to prevent undefined bits that ilwalidate saved hashes.
        allocationsArray.push_back(rawMem);
        return rawMem;
    }
    void freeAllocatedMem()
    {
        for (size_t allocIdx = 0, allocEnd = allocationsArray.size(); allocIdx < allocEnd; ++allocIdx)
        {
            free(allocationsArray[allocIdx]);
        }
        allocationsArray.resize(0);
    }

    std::vector<ResourcePixelShader> pixelShaders;
    std::vector<ResourceVertexShader> vertexShaders;
    std::vector<ResourceSampler> samplers;
    std::vector<ResourceConstantBuffer> constantBuffers;

    std::vector<ResourceTextureParametrized> texturesParametrized;
    std::vector<ResourceTextureIntermediate> texturesIntermediate;
    std::vector<ResourceTextureFromFile> texturesFromFile;

    std::vector<UserConstant> userConstants;

    std::vector<Pass> passes;
};

static
void initRasterizerState(RasterizerStateStorage * rasterizerState)
{
    if (rasterizerState == nullptr)
        return;

    RasterizerStateStorage::BinaryStorage & rs = rasterizerState->binStorage;

    // Defaults from D3D11_RASTERIZER_DESC
    rs.fillMode = RasterizerFillMode::kSolid;
    rs.lwllMode = RasterizerLwllMode::kBack;
    rs.frontCounterClockwise = 0;
    rs.depthBias = 0;
    rs.slopeScaledDepthBias = 0.0f;
    rs.depthBiasClamp = 0.0f;
    rs.depthClipEnable = 1;
    rs.scissorEnable = 0;
    rs.multisampleEnable = 0;
    rs.antialiasedLineEnable = 0;
}

static
void initDepthStencilState(DepthStencilStateStorage * depthStencilState)
{
    if (depthStencilState == nullptr)
        return;

    DepthStencilStateStorage::BinaryStorage & ds = depthStencilState->binStorage;

    // Defaults from D3D11_DEPTH_STENCIL_DESC
    ds.isDepthEnabled = 1;
    ds.depthWriteMask = DepthWriteMask::kAll;
    ds.depthFunc = ComparisonFunc::kLess;
    ds.isStencilEnabled = 0;
    ds.stencilReadMask = DepthStencilStateStorage::defaultStencilReadMask;
    ds.stencilWriteMask = DepthStencilStateStorage::defaultStencilWriteMask;
    ds.frontFace.binStorage.func = ComparisonFunc::kAlways;
    ds.backFace.binStorage.func = ComparisonFunc::kAlways;
    ds.frontFace.binStorage.depthFailOp = StencilOp::kKeep;
    ds.backFace.binStorage.depthFailOp = StencilOp::kKeep;
    ds.frontFace.binStorage.passOp = StencilOp::kKeep;
    ds.backFace.binStorage.passOp = StencilOp::kKeep;
    ds.frontFace.binStorage.failOp = StencilOp::kKeep;
    ds.backFace.binStorage.failOp = StencilOp::kKeep;
}

static
void initAlphaBlendState(AlphaBlendStateStorage * alphaBlendState)
{
    if (alphaBlendState == nullptr)
        return;

    AlphaBlendStateStorage::BinaryStorage & as = alphaBlendState->binStorage;

    // Defaults from D3D11_BLEND_DESC
    as.alphaToCoverageEnable = 0;
    as.independentBlendEnable = 0;

    for (uint8_t idx = 0; idx < AlphaBlendStateStorage::renderTargetsNum; ++idx)
    {
        AlphaBlendRenderTargetStateStorage::BinaryStorage & asrts = as.renderTargetState[idx].binStorage;
        asrts.isEnabled = 0;

        asrts.src = BlendCoef::kOne;
        asrts.dst = BlendCoef::kZero;
        asrts.op = BlendOp::kAdd;

        asrts.srcAlpha = BlendCoef::kOne;
        asrts.dstAlpha = BlendCoef::kZero;
        asrts.opAlpha = BlendOp::kAdd;

        asrts.renderTargetWriteMask = ColorWriteEnableBits::kAll;
    }
}

static
void save(const wchar_t * filename, const acef::EffectStorage & acefEffectStorage)
{
    FILE * fp = nullptr;
    _wfopen_s(&fp, filename, L"wb");

#define DBG_CHECK_FILE_OFFSETS      0

    auto serializeData = [](FILE *fp, const void * dataRaw, size_t elementByteSize, size_t numElements)
    {
        fwrite(dataRaw, elementByteSize, numElements, fp);
    };

    const acef::Header & header = acefEffectStorage.header;
    serializeData(fp, &header.binStorage, sizeof(acef::Header::BinaryStorage), 1);
    
    // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
    auto calcTotalChars = [](uint16_t * bufNameLens, const uint32_t bufNum)
    {
        size_t totalCharacters = 0;
        for (uint32_t bufIdx = 0; bufIdx < bufNum; ++bufIdx)
        {
            totalCharacters += bufNameLens[bufIdx];
        }
        return totalCharacters;
    };

    serializeData(fp, header.fileTimestamps, sizeof(uint64_t), header.binStorage.dependenciesNum);
    serializeData(fp, header.filePathLens, sizeof(uint16_t), header.binStorage.dependenciesNum);
    serializeData(fp, header.filePathOffsets, sizeof(uint32_t), header.binStorage.dependenciesNum);
    serializeData(fp, header.filePathsUtf8, sizeof(char), calcTotalChars(header.filePathLens, header.binStorage.dependenciesNum));

    // Resources Chunk
    /////////////////////////////////////////////////////////////////////////////////
    {
#if (DBG_CHECK_FILE_OFFSETS == 1)
        {
            const long int fileOffset = ftell(fp);
            assert(fileOffset == acefEffectStorage.header.binStorage.resourcesChunkByteOffset);
        }
#endif

        const acef::ResourceHeader & resourceHeader = acefEffectStorage.resourceHeader;
        serializeData(fp, &resourceHeader.binStorage, sizeof(acef::ResourceHeader::BinaryStorage), 1);

        serializeData(fp, resourceHeader.readBufferTextureHandles, sizeof(uint32_t), resourceHeader.binStorage.readBuffersNum);
        serializeData(fp, resourceHeader.writeBufferTextureHandles, sizeof(uint32_t), resourceHeader.binStorage.writeBuffersNum);

        serializeData(fp, resourceHeader.pixelShaderByteOffsets, sizeof(uint64_t), resourceHeader.binStorage.pixelShadersNum);
        serializeData(fp, resourceHeader.vertexShaderByteOffsets, sizeof(uint64_t), resourceHeader.binStorage.vertexShadersNum);

        serializeData(fp, resourceHeader.samplerByteOffsets, sizeof(uint64_t), resourceHeader.binStorage.samplersNum);
        serializeData(fp, resourceHeader.constantBufferByteOffsets, sizeof(uint64_t), resourceHeader.binStorage.constantBuffersNum);

        serializeData(fp, resourceHeader.textureParametrizedByteOffsets, sizeof(uint64_t), resourceHeader.binStorage.texturesParametrizedNum);
        serializeData(fp, resourceHeader.textureIntermediateByteOffsets, sizeof(uint64_t), resourceHeader.binStorage.texturesIntermediateNum);
        serializeData(fp, resourceHeader.textureFromFileByteOffsets, sizeof(uint64_t), resourceHeader.binStorage.texturesFromFileNum);

#if (DBG_CHECK_FILE_OFFSETS == 1)
        {
            const long int fileOffset = ftell(fp);
            if (acefEffectStorage.resourceHeader.binStorage.pixelShadersNum > 0)
            {
                assert(fileOffset == acefEffectStorage.header.binStorage.resourcesChunkByteOffset + acefEffectStorage.resourceHeader.pixelShaderByteOffsets[0]);
            }
        }
#endif

        // ResourcePixelShader
        assert(resourceHeader.binStorage.pixelShadersNum == acefEffectStorage.pixelShaders.size());
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.pixelShadersNum; ++idx)
        {
            const acef::ResourcePixelShader & pixelShader = acefEffectStorage.pixelShaders[idx];
            serializeData(fp, &pixelShader.binStorage, sizeof(acef::ResourcePixelShader::BinaryStorage), 1);
            serializeData(fp, pixelShader.filePathUtf8, sizeof(char), pixelShader.binStorage.filePathLen);
            serializeData(fp, pixelShader.entryFunctionAscii, sizeof(char), pixelShader.binStorage.entryFunctionLen);
        }

#if (DBG_CHECK_FILE_OFFSETS == 1)
        {
            const long int fileOffset = ftell(fp);
            if (acefEffectStorage.resourceHeader.binStorage.vertexShadersNum > 0)
            {
                assert(fileOffset == acefEffectStorage.header.binStorage.resourcesChunkByteOffset + acefEffectStorage.resourceHeader.vertexShaderByteOffsets[0]);
            }
        }
#endif

        // ResourceVertexShader
        assert(resourceHeader.binStorage.vertexShadersNum == acefEffectStorage.vertexShaders.size());
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.vertexShadersNum; ++idx)
        {
            const acef::ResourceVertexShader & vertexShader = acefEffectStorage.vertexShaders[idx];
            serializeData(fp, &vertexShader.binStorage, sizeof(acef::ResourceVertexShader::BinaryStorage), 1);
            serializeData(fp, vertexShader.filePathUtf8, sizeof(char), vertexShader.binStorage.filePathLen);
            serializeData(fp, vertexShader.entryFunctionAscii, sizeof(char), vertexShader.binStorage.entryFunctionLen);
        }

#if (DBG_CHECK_FILE_OFFSETS == 1)
        {
            const long int fileOffset = ftell(fp);
            if (acefEffectStorage.resourceHeader.binStorage.texturesParametrizedNum > 0)
            {
                assert(fileOffset == acefEffectStorage.header.binStorage.resourcesChunkByteOffset + acefEffectStorage.resourceHeader.textureParametrizedByteOffsets[0]);
            }
        }
#endif

        // ResourceTextureParametrized
        assert(resourceHeader.binStorage.texturesParametrizedNum == acefEffectStorage.texturesParametrized.size());
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.texturesParametrizedNum; ++idx)
        {
            const acef::ResourceTextureParametrized & texParametrized = acefEffectStorage.texturesParametrized[idx];
            serializeData(fp, &texParametrized.binStorage, sizeof(acef::ResourceTextureParametrized::BinaryStorage), 1);
        }

#if (DBG_CHECK_FILE_OFFSETS == 1)
        {
            const long int fileOffset = ftell(fp);
            if (acefEffectStorage.resourceHeader.binStorage.texturesIntermediateNum > 0)
            {
                assert(fileOffset == acefEffectStorage.header.binStorage.resourcesChunkByteOffset + acefEffectStorage.resourceHeader.textureIntermediateByteOffsets[0]);
            }
        }
#endif

        // ResourceTextureIntermediate
        assert(resourceHeader.binStorage.texturesIntermediateNum == acefEffectStorage.texturesIntermediate.size());
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.texturesIntermediateNum; ++idx)
        {
            const acef::ResourceTextureIntermediate & texIntermediate = acefEffectStorage.texturesIntermediate[idx];
            serializeData(fp, &texIntermediate.binStorage, sizeof(acef::ResourceTextureIntermediate::BinaryStorage), 1);
        }

#if (DBG_CHECK_FILE_OFFSETS == 1)
        {
            const long int fileOffset = ftell(fp);
            if (acefEffectStorage.resourceHeader.binStorage.texturesFromFileNum > 0)
            {
                assert(fileOffset == acefEffectStorage.header.binStorage.resourcesChunkByteOffset + acefEffectStorage.resourceHeader.textureFromFileByteOffsets[0]);
            }
        }
#endif

        // ResourceTextureFromFile
        assert(resourceHeader.binStorage.texturesFromFileNum == acefEffectStorage.texturesFromFile.size());
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.texturesFromFileNum; ++idx)
        {
            const acef::ResourceTextureFromFile & texFromFile = acefEffectStorage.texturesFromFile[idx];
            serializeData(fp, &texFromFile.binStorage, sizeof(acef::ResourceTextureFromFile::BinaryStorage), 1);
            serializeData(fp, texFromFile.pathUtf8, sizeof(char), texFromFile.binStorage.pathLen);
        }

#if (DBG_CHECK_FILE_OFFSETS == 1)
        {
            const long int fileOffset = ftell(fp);
            if (acefEffectStorage.resourceHeader.binStorage.samplersNum > 0)
            {
                assert(fileOffset == acefEffectStorage.header.binStorage.resourcesChunkByteOffset + acefEffectStorage.resourceHeader.samplerByteOffsets[0]);
            }
        }
#endif

        // ResourceSampler
        assert(resourceHeader.binStorage.samplersNum == acefEffectStorage.samplers.size());
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.samplersNum; ++idx)
        {
            const acef::ResourceSampler & sampler = acefEffectStorage.samplers[idx];
            serializeData(fp, &sampler.binStorage, sizeof(acef::ResourceSampler::BinaryStorage), 1);
        }

#if (DBG_CHECK_FILE_OFFSETS == 1)
        {
            const long int fileOffset = ftell(fp);
            if (acefEffectStorage.resourceHeader.binStorage.constantBuffersNum > 0)
            {
                assert(fileOffset == acefEffectStorage.header.binStorage.resourcesChunkByteOffset + acefEffectStorage.resourceHeader.constantBufferByteOffsets[0]);
            }
        }
#endif

        // ResourceConstantBuffer
        assert(resourceHeader.binStorage.constantBuffersNum == acefEffectStorage.constantBuffers.size());
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.constantBuffersNum; ++idx)
        {
/*
            // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
            auto calcTotalChars = [](uint16_t * bufNameLens, const uint32_t bufNum)
            {
                size_t totalCharacters = 0;
                for (uint32_t bufIdx = 0; bufIdx < bufNum; ++bufIdx)
                {
                    totalCharacters += bufNameLens[bufIdx];
                }
                return totalCharacters;
            };
*/
            const acef::ResourceConstantBuffer & constBuf = acefEffectStorage.constantBuffers[idx];
            serializeData(fp, &constBuf.binStorage, sizeof(acef::ResourceConstantBuffer::BinaryStorage), 1);
            serializeData(fp, constBuf.constantOffsetInComponents, sizeof(uint32_t), constBuf.binStorage.constantsNum);
            serializeData(fp, constBuf.constantNameLens, sizeof(uint16_t), constBuf.binStorage.constantsNum);
            serializeData(fp, constBuf.constantNameOffsets, sizeof(uint32_t), constBuf.binStorage.constantsNum);
            serializeData(fp, constBuf.constantNames, sizeof(char), calcTotalChars(constBuf.constantNameLens, constBuf.binStorage.constantsNum));
            serializeData(fp, constBuf.constantHandle, sizeof(uint32_t), constBuf.binStorage.constantsNum);
        }
    }

    // UI Controls Chunk
    /////////////////////////////////////////////////////////////////////////////////
    {
        auto serializeLocalizedStringBuffers = [&serializeData](FILE * fp, const acef::UILocalizedStringBuffers & locStringBuffers, const acef::UILocalizedStringStorage & locStringStorage)
        {
            serializeData(fp, locStringBuffers.defaultStringAscii, sizeof(char), locStringStorage.binStorage.defaultStringLen);

            for (uint32_t locIdx = 0; locIdx < locStringStorage.binStorage.localizationsNum; ++locIdx)
            {
                const acef::UILocalizedStringBuffers::LocalizedString & locStr = locStringBuffers.localizedStrings[locIdx];
                serializeData(fp, &locStr.binStorage, sizeof(acef::UILocalizedStringBuffers::LocalizedString::BinaryStorage), 1);
                serializeData(fp, locStr.stringUtf8, sizeof(char), locStr.binStorage.strLen);
            }
        };

#if (DBG_CHECK_FILE_OFFSETS == 1)
        {
            const long int fileOffset = ftell(fp);
            assert(fileOffset == acefEffectStorage.header.binStorage.uiControlsChunkByteOffset);
        }
#endif
        const acef::UIControlsHeader & uiControlsHeader = acefEffectStorage.uiControlsHeader;
        serializeData(fp, &uiControlsHeader.binStorage, sizeof(acef::UIControlsHeader::BinaryStorage), 1);

        serializeData(fp, uiControlsHeader.userConstantByteOffsets, sizeof(uint64_t), uiControlsHeader.binStorage.userConstantsNum);

        assert(uiControlsHeader.binStorage.userConstantsNum == acefEffectStorage.userConstants.size());
        for (uint32_t idx = 0; idx < uiControlsHeader.binStorage.userConstantsNum; ++idx)
        {
#if (DBG_CHECK_FILE_OFFSETS == 1)
            {
                const long int fileOffset = ftell(fp);
                assert(fileOffset == acefEffectStorage.header.binStorage.uiControlsChunkByteOffset + acefEffectStorage.uiControlsHeader.userConstantByteOffsets[idx]);
            }
#endif

            const acef::UserConstant & userConstant = acefEffectStorage.userConstants[idx];
            serializeData(fp, &userConstant.binStorage, sizeof(acef::UserConstant::BinaryStorage), 1);

            serializeData(fp, userConstant.controlNameAscii, sizeof(char), userConstant.binStorage.controlNameLen);

            serializeLocalizedStringBuffers(fp, userConstant.labelBuffers, userConstant.binStorage.label);
            serializeLocalizedStringBuffers(fp, userConstant.hintBuffers, userConstant.binStorage.hint);
            serializeLocalizedStringBuffers(fp, userConstant.uiValueUnitBuffers, userConstant.binStorage.uiValueUnit);

            serializeData(fp, userConstant.optiolwalues, sizeof(acef::TypelessVariableStorage::BinaryStorage), userConstant.binStorage.optionsNum);

            serializeData(fp, userConstant.optionNameByteOffsets, sizeof(uint64_t), userConstant.binStorage.optionsNum);

            for (uint32_t optIdx = 0; optIdx < userConstant.binStorage.optionsNum; ++optIdx)
            {
#if (DBG_CHECK_FILE_OFFSETS == 1)
                {
                    const long int fileOffset = ftell(fp);
                    assert(fileOffset == acefEffectStorage.header.binStorage.uiControlsChunkByteOffset + acefEffectStorage.uiControlsHeader.userConstantByteOffsets[idx] + userConstant.optionNameByteOffsets[optIdx]);
                }
#endif
                const acef::UILocalizedStringStorage & optionNameStorage = userConstant.optionNames[optIdx];
                const acef::UILocalizedStringBuffers & optionNameBuffers = userConstant.optionNamesBuffers[optIdx];

                serializeData(fp, &optionNameStorage.binStorage, sizeof(acef::UILocalizedStringStorage::BinaryStorage), 1);
                serializeLocalizedStringBuffers(fp, optionNameBuffers, optionNameStorage);
            }

            // Labels for vector elements in a grouped variable
            serializeData(fp, userConstant.variableNameByteOffsets, sizeof(uint64_t), userConstant.binStorage.dataDimensionality);

            for (uint32_t varIdx = 0; varIdx < userConstant.binStorage.dataDimensionality; ++varIdx)
            {
#if (DBG_CHECK_FILE_OFFSETS == 1)
                {
                    const long int fileOffset = ftell(fp);
                    assert(fileOffset == acefEffectStorage.header.binStorage.uiControlsChunkByteOffset + acefEffectStorage.uiControlsHeader.userConstantByteOffsets[idx] + userConstant.variableNameByteOffsets[varIdx]);
                }
#endif
                const acef::UILocalizedStringStorage & variableNameStorage = userConstant.variableNames[varIdx];
                const acef::UILocalizedStringBuffers & variableNameBuffers = userConstant.variableNamesBuffers[varIdx];

                serializeData(fp, &variableNameStorage.binStorage, sizeof(acef::UILocalizedStringStorage::BinaryStorage), 1);
                serializeLocalizedStringBuffers(fp, variableNameBuffers, variableNameStorage);
            }
        }
    }

    // Passes Chunk
    /////////////////////////////////////////////////////////////////////////////////
    {
#if (DBG_CHECK_FILE_OFFSETS == 1)
        {
            const long int fileOffset = ftell(fp);
            assert(fileOffset == acefEffectStorage.header.binStorage.passesChunkByteOffset);
        }
#endif

        const acef::PassesHeader & passesHeader = acefEffectStorage.passesHeader;
        serializeData(fp, &passesHeader.binStorage, sizeof(acef::PassesHeader::BinaryStorage), 1);

        serializeData(fp, passesHeader.passByteOffsets, sizeof(uint64_t), passesHeader.binStorage.passesNum);

        assert(passesHeader.binStorage.passesNum == acefEffectStorage.passes.size());
        for (uint32_t idx = 0; idx < passesHeader.binStorage.passesNum; ++idx)
        {
            const acef::Pass & pass = acefEffectStorage.passes[idx];
            serializeData(fp, &pass.binStorage, sizeof(acef::Pass::BinaryStorage), 1);

            // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
            auto serializeBufferNames = [&serializeData](FILE * fp, char * bufferNames, uint16_t * bufferNameLens, const uint32_t buffersNum)
            {
                size_t totalCharacters = 0;
                for (uint32_t bufIdx = 0; bufIdx < buffersNum; ++bufIdx)
                {
                    totalCharacters += bufferNameLens[bufIdx];
                }
                serializeData(fp, bufferNames, sizeof(char), totalCharacters);
            };

            // Read Buffers
            serializeData(fp, pass.readBuffersSlots, sizeof(uint32_t), pass.binStorage.readBuffersNum);
            serializeData(fp, pass.readBuffersNameLens, sizeof(uint16_t), pass.binStorage.readBuffersNum);
            serializeData(fp, pass.readBuffersNameOffsets, sizeof(uint32_t), pass.binStorage.readBuffersNum);
            serializeBufferNames(fp, pass.readBuffersNames, pass.readBuffersNameLens, pass.binStorage.readBuffersNum);
            serializeData(fp, pass.readBuffersIndices, sizeof(uint32_t), pass.binStorage.readBuffersNum);

            // Write Buffers
            serializeData(fp, pass.writeBuffersSlots, sizeof(uint32_t), pass.binStorage.writeBuffersNum);
            serializeData(fp, pass.writeBuffersNameLens, sizeof(uint16_t), pass.binStorage.writeBuffersNum);
            serializeData(fp, pass.writeBuffersNameOffsets, sizeof(uint32_t), pass.binStorage.writeBuffersNum);
            serializeBufferNames(fp, pass.writeBuffersNames, pass.writeBuffersNameLens, pass.binStorage.writeBuffersNum);
            serializeData(fp, pass.writeBuffersIndices, sizeof(uint32_t), pass.binStorage.writeBuffersNum);

            // Constant Buffers
            //VS
            serializeData(fp, pass.constantBuffersVSSlots, sizeof(uint32_t), pass.binStorage.constantBuffersVSNum);
            serializeData(fp, pass.constantBuffersVSNameLens, sizeof(uint16_t), pass.binStorage.constantBuffersVSNum);
            serializeData(fp, pass.constantBuffersVSNameOffsets, sizeof(uint32_t), pass.binStorage.constantBuffersVSNum);
            serializeBufferNames(fp, pass.constantBuffersVSNames, pass.constantBuffersVSNameLens, pass.binStorage.constantBuffersVSNum);
            serializeData(fp, pass.constantBuffersVSIndices, sizeof(uint32_t), pass.binStorage.constantBuffersVSNum);

            //PS
            serializeData(fp, pass.constantBuffersPSSlots, sizeof(uint32_t), pass.binStorage.constantBuffersPSNum);
            serializeData(fp, pass.constantBuffersPSNameLens, sizeof(uint16_t), pass.binStorage.constantBuffersPSNum);
            serializeData(fp, pass.constantBuffersPSNameOffsets, sizeof(uint32_t), pass.binStorage.constantBuffersPSNum);
            serializeBufferNames(fp, pass.constantBuffersPSNames, pass.constantBuffersPSNameLens, pass.binStorage.constantBuffersPSNum);
            serializeData(fp, pass.constantBuffersPSIndices, sizeof(uint32_t), pass.binStorage.constantBuffersPSNum);

            // Samplers
            serializeData(fp, pass.samplersSlots, sizeof(uint32_t), pass.binStorage.samplersNum);
            serializeData(fp, pass.samplersNameLens, sizeof(uint16_t), pass.binStorage.samplersNum);
            serializeData(fp, pass.samplersNameOffsets, sizeof(uint32_t), pass.binStorage.samplersNum);
            serializeBufferNames(fp, pass.samplersNames, pass.samplersNameLens, pass.binStorage.samplersNum);
            serializeData(fp, pass.samplersIndices, sizeof(uint32_t), pass.binStorage.samplersNum);
        }
    }

#if (DBG_CHECK_FILE_OFFSETS == 1)
    {
        const long int fileOffset = ftell(fp);
        assert(fileOffset == acefEffectStorage.totalByteSize);
    }
#endif

    fclose(fp);
}

enum class FileReadingStatus
{
    kOK = 0,
    kILWALID_ARGS,
    kDOESNT_EXIST
};

// WARNING! Allocates memory, so needs to free memory if full effect will be loaded later
static
FileReadingStatus loadHeaderData(const wchar_t * filename, acef::EffectStorage * acefEffectStorage)
{
    if (acefEffectStorage == nullptr || filename == nullptr)
        return FileReadingStatus::kILWALID_ARGS;

    auto deserializeData = [](FILE *fp, void * dataRaw, size_t elementByteSize, size_t numElements)
    {
        fread(dataRaw, elementByteSize, numElements, fp);
    };
    auto allocateAndDeserializeData = [&acefEffectStorage](FILE *fp, size_t elementByteSize, size_t numElements) -> void *
    {
        void * dataRaw = acefEffectStorage->allocateMem(elementByteSize * numElements);
        fread(dataRaw, elementByteSize, numElements, fp);
        return dataRaw;
    };

    FILE * fp = nullptr;
    _wfopen_s(&fp, filename, L"rb");
    if (!fp)
    {
        return FileReadingStatus::kDOESNT_EXIST;
    }

    acef::Header & header = acefEffectStorage->header;
    deserializeData(fp, &header.binStorage, sizeof(acef::Header::BinaryStorage), 1);

    // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
    auto calcTotalChars = [](uint16_t * bufNameLens, const uint32_t bufNum)
    {
        size_t totalCharacters = 0;
        for (uint32_t bufIdx = 0; bufIdx < bufNum; ++bufIdx)
        {
            totalCharacters += bufNameLens[bufIdx];
        }
        return totalCharacters;
    };

    header.fileTimestamps = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), header.binStorage.dependenciesNum);
    header.filePathLens = (uint16_t *)allocateAndDeserializeData(fp, sizeof(uint16_t), header.binStorage.dependenciesNum);
    header.filePathOffsets = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), header.binStorage.dependenciesNum);
    header.filePathsUtf8 = (char *)allocateAndDeserializeData(fp, sizeof(char), calcTotalChars(header.filePathLens, header.binStorage.dependenciesNum));

    fclose(fp);

    return FileReadingStatus::kOK;
}

static void eraseTimestamps(unsigned char * contents)
{
    if (contents == nullptr)
        return;

    /*
        Original ACEF writing routine:
            const acef::Header & header = acefEffectStorage.header;
            serializeData(fp, &header.binStorage, sizeof(acef::Header::BinaryStorage), 1);

            serializeData(fp, header.fileTimestamps, sizeof(uint64_t), header.binStorage.dependenciesNum);
    */

    // First - read the header to determine number of dependencies
    acef::Header header;
    memcpy(&header.binStorage, contents, sizeof(acef::Header::BinaryStorage));
    const size_t headerBinStorageOffset = offsetof(acef::Header, binStorage);
    const size_t binStorageTimestampOffset = offsetof(acef::Header::BinaryStorage, timestamp);
    memset(contents + headerBinStorageOffset + binStorageTimestampOffset, 0, sizeof(uint64_t));

    acef::Header DBGheader;
    memcpy(&DBGheader.binStorage, contents, sizeof(acef::Header::BinaryStorage));

    // Then - erase the timestamp data
    unsigned char * timestampsContent = contents + sizeof(acef::Header::BinaryStorage);
    memset(timestampsContent, 0, header.binStorage.dependenciesNum * sizeof(uint64_t));
}

static
void load(const wchar_t * filename, acef::EffectStorage * acefEffectStorage)
{
    if (acefEffectStorage == nullptr || filename == nullptr)
        return;

    auto deserializeData = [](FILE *fp, void * dataRaw, size_t elementByteSize, size_t numElements)
    {
        fread(dataRaw, elementByteSize, numElements, fp);
    };
    auto allocateAndDeserializeData = [&acefEffectStorage](FILE *fp, size_t elementByteSize, size_t numElements) -> void *
    {
        void * dataRaw = acefEffectStorage->allocateMem(elementByteSize * numElements);
        fread(dataRaw, elementByteSize, numElements, fp);
        return dataRaw;
    };

    auto gotoByteOffset = [](FILE *fp, uint64_t byteOffset)
    {
        fseek(fp, (long)byteOffset, SEEK_SET);
    };

    FILE * fp = nullptr;
    _wfopen_s(&fp, filename, L"rb");

    acef::Header & header = acefEffectStorage->header;
    deserializeData(fp, &header.binStorage, sizeof(acef::Header::BinaryStorage), 1);

    // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
    auto calcTotalChars = [](uint16_t * bufNameLens, const uint32_t bufNum)
    {
        size_t totalCharacters = 0;
        for (uint32_t bufIdx = 0; bufIdx < bufNum; ++bufIdx)
        {
            totalCharacters += bufNameLens[bufIdx];
        }
        return totalCharacters;
    };

    header.fileTimestamps = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), header.binStorage.dependenciesNum);
    header.filePathLens = (uint16_t *)allocateAndDeserializeData(fp, sizeof(uint16_t), header.binStorage.dependenciesNum);
    header.filePathOffsets = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), header.binStorage.dependenciesNum);
    header.filePathsUtf8 = (char *)allocateAndDeserializeData(fp, sizeof(char), calcTotalChars(header.filePathLens, header.binStorage.dependenciesNum));

    // Resources Chunk
    /////////////////////////////////////////////////////////////////////////////////
    {
        gotoByteOffset(fp, header.binStorage.resourcesChunkByteOffset);

        acef::ResourceHeader & resourceHeader = acefEffectStorage->resourceHeader;
        deserializeData(fp, &resourceHeader.binStorage, sizeof(acef::ResourceHeader::BinaryStorage), 1);

        resourceHeader.readBufferTextureHandles = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), resourceHeader.binStorage.readBuffersNum);
        resourceHeader.writeBufferTextureHandles = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), resourceHeader.binStorage.writeBuffersNum);

        resourceHeader.pixelShaderByteOffsets = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), resourceHeader.binStorage.pixelShadersNum);
        resourceHeader.vertexShaderByteOffsets = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), resourceHeader.binStorage.vertexShadersNum);

        resourceHeader.samplerByteOffsets = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), resourceHeader.binStorage.samplersNum);
        resourceHeader.constantBufferByteOffsets = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), resourceHeader.binStorage.constantBuffersNum);

        resourceHeader.textureParametrizedByteOffsets = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), resourceHeader.binStorage.texturesParametrizedNum);
        resourceHeader.textureIntermediateByteOffsets = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), resourceHeader.binStorage.texturesIntermediateNum);
        resourceHeader.textureFromFileByteOffsets = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), resourceHeader.binStorage.texturesFromFileNum);

        // ResourcePixelShader
        acefEffectStorage->pixelShaders.resize(resourceHeader.binStorage.pixelShadersNum);
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.pixelShadersNum; ++idx)
        {
            gotoByteOffset(fp, header.binStorage.resourcesChunkByteOffset + resourceHeader.pixelShaderByteOffsets[idx]);

            acef::ResourcePixelShader & pixelShader = acefEffectStorage->pixelShaders[idx];
            deserializeData(fp, &pixelShader.binStorage, sizeof(acef::ResourcePixelShader::BinaryStorage), (size_t)1);
            pixelShader.filePathUtf8 = (char *)allocateAndDeserializeData(fp, sizeof(char), pixelShader.binStorage.filePathLen);
            pixelShader.entryFunctionAscii = (char *)allocateAndDeserializeData(fp, sizeof(char), pixelShader.binStorage.entryFunctionLen);
        }

        // ResourceVertexShader
        acefEffectStorage->vertexShaders.resize(resourceHeader.binStorage.vertexShadersNum);
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.vertexShadersNum; ++idx)
        {
            gotoByteOffset(fp, header.binStorage.resourcesChunkByteOffset + resourceHeader.vertexShaderByteOffsets[idx]);

            acef::ResourceVertexShader & vertexShader = acefEffectStorage->vertexShaders[idx];
            deserializeData(fp, &vertexShader.binStorage, sizeof(acef::ResourceVertexShader::BinaryStorage), 1);
            vertexShader.filePathUtf8 = (char *)allocateAndDeserializeData(fp, sizeof(char), vertexShader.binStorage.filePathLen);
            vertexShader.entryFunctionAscii = (char *)allocateAndDeserializeData(fp, sizeof(char), vertexShader.binStorage.entryFunctionLen);
        }

        // ResourceTextureParametrized
        acefEffectStorage->texturesParametrized.resize(resourceHeader.binStorage.texturesParametrizedNum);
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.texturesParametrizedNum; ++idx)
        {
            gotoByteOffset(fp, header.binStorage.resourcesChunkByteOffset + resourceHeader.textureParametrizedByteOffsets[idx]);

            acef::ResourceTextureParametrized & texParametrized = acefEffectStorage->texturesParametrized[idx];
            deserializeData(fp, &texParametrized.binStorage, sizeof(acef::ResourceTextureParametrized::BinaryStorage), 1);
        }

        // ResourceTextureIntermediate
        acefEffectStorage->texturesIntermediate.resize(resourceHeader.binStorage.texturesIntermediateNum);
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.texturesIntermediateNum; ++idx)
        {
            gotoByteOffset(fp, header.binStorage.resourcesChunkByteOffset + resourceHeader.textureIntermediateByteOffsets[idx]);

            acef::ResourceTextureIntermediate & texIntermediate = acefEffectStorage->texturesIntermediate[idx];
            deserializeData(fp, &texIntermediate.binStorage, sizeof(acef::ResourceTextureIntermediate::BinaryStorage), 1);
        }

        // ResourceTextureFromFile
        acefEffectStorage->texturesFromFile.resize(resourceHeader.binStorage.texturesFromFileNum);
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.texturesFromFileNum; ++idx)
        {
            gotoByteOffset(fp, header.binStorage.resourcesChunkByteOffset + resourceHeader.textureFromFileByteOffsets[idx]);

            acef::ResourceTextureFromFile & texFromFile = acefEffectStorage->texturesFromFile[idx];
            deserializeData(fp, &texFromFile.binStorage, sizeof(acef::ResourceTextureFromFile::BinaryStorage), 1);
            texFromFile.pathUtf8 = (char *)allocateAndDeserializeData(fp, sizeof(char), texFromFile.binStorage.pathLen);
        }

        // ResourceSampler
        acefEffectStorage->samplers.resize(resourceHeader.binStorage.samplersNum);
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.samplersNum; ++idx)
        {
            gotoByteOffset(fp, header.binStorage.resourcesChunkByteOffset + resourceHeader.samplerByteOffsets[idx]);

            acef::ResourceSampler & sampler = acefEffectStorage->samplers[idx];
            deserializeData(fp, &sampler.binStorage, sizeof(acef::ResourceSampler::BinaryStorage), 1);
        }

        // ResourceConstantBuffer
        acefEffectStorage->constantBuffers.resize(resourceHeader.binStorage.constantBuffersNum);
        for (uint32_t idx = 0; idx < resourceHeader.binStorage.constantBuffersNum; ++idx)
        {
            gotoByteOffset(fp, header.binStorage.resourcesChunkByteOffset + resourceHeader.constantBufferByteOffsets[idx]);

/*
            // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
            auto calcTotalChars = [](uint16_t * bufNameLens, const uint32_t bufNum)
            {
                size_t totalCharacters = 0;
                for (uint32_t bufIdx = 0; bufIdx < bufNum; ++bufIdx)
                {
                    totalCharacters += bufNameLens[bufIdx];
                }
                return totalCharacters;
            };
*/

            acef::ResourceConstantBuffer & constBuf = acefEffectStorage->constantBuffers[idx];
            deserializeData(fp, &constBuf.binStorage, sizeof(acef::ResourceConstantBuffer::BinaryStorage), 1);
            constBuf.constantOffsetInComponents = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), constBuf.binStorage.constantsNum);
            constBuf.constantNameLens = (uint16_t *)allocateAndDeserializeData(fp, sizeof(uint16_t), constBuf.binStorage.constantsNum);
            constBuf.constantNameOffsets = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), constBuf.binStorage.constantsNum);
            constBuf.constantNames = (char *)allocateAndDeserializeData(fp, sizeof(char), calcTotalChars(constBuf.constantNameLens, constBuf.binStorage.constantsNum));
            constBuf.constantHandle = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), constBuf.binStorage.constantsNum);
        }
    }

    // UI Controls Chunk
    /////////////////////////////////////////////////////////////////////////////////
    {
        gotoByteOffset(fp, header.binStorage.uiControlsChunkByteOffset);

        auto deserializeLocalizedStringBuffers = [&deserializeData, &allocateAndDeserializeData, &acefEffectStorage](FILE * fp, acef::UILocalizedStringBuffers & locStringBuffers, acef::UILocalizedStringStorage & locStringStorage)
        {
            locStringBuffers.defaultStringAscii = (char *)allocateAndDeserializeData(fp, sizeof(char), locStringStorage.binStorage.defaultStringLen);

            locStringBuffers.localizedStrings = (acef::UILocalizedStringBuffers::LocalizedString *)acefEffectStorage->allocateMem(locStringStorage.binStorage.localizationsNum * sizeof(acef::UILocalizedStringBuffers::LocalizedString));
            for (uint32_t locIdx = 0; locIdx < locStringStorage.binStorage.localizationsNum; ++locIdx)
            {
                // TODO avoroshilov ACEF: add byteOffsets for LocalizedString

                acef::UILocalizedStringBuffers::LocalizedString & locStr = locStringBuffers.localizedStrings[locIdx];
                deserializeData(fp, &locStr.binStorage, sizeof(acef::UILocalizedStringBuffers::LocalizedString::BinaryStorage), 1);
                locStr.stringUtf8 = (char *)allocateAndDeserializeData(fp, sizeof(char), locStr.binStorage.strLen);
            }
        };

        acef::UIControlsHeader & uiControlsHeader = acefEffectStorage->uiControlsHeader;
        deserializeData(fp, &uiControlsHeader.binStorage, sizeof(acef::UIControlsHeader::BinaryStorage), 1);

        uiControlsHeader.userConstantByteOffsets = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), uiControlsHeader.binStorage.userConstantsNum);

        acefEffectStorage->userConstants.resize(uiControlsHeader.binStorage.userConstantsNum);
        for (uint32_t idx = 0; idx < uiControlsHeader.binStorage.userConstantsNum; ++idx)
        {
            gotoByteOffset(fp, header.binStorage.uiControlsChunkByteOffset + uiControlsHeader.userConstantByteOffsets[idx]);

            acef::UserConstant & userConstant = acefEffectStorage->userConstants[idx];
            deserializeData(fp, &userConstant.binStorage, sizeof(acef::UserConstant::BinaryStorage), 1);

            userConstant.controlNameAscii = (char *)allocateAndDeserializeData(fp, sizeof(char), userConstant.binStorage.controlNameLen);

            deserializeLocalizedStringBuffers(fp, userConstant.labelBuffers, userConstant.binStorage.label);
            deserializeLocalizedStringBuffers(fp, userConstant.hintBuffers, userConstant.binStorage.hint);
            deserializeLocalizedStringBuffers(fp, userConstant.uiValueUnitBuffers, userConstant.binStorage.uiValueUnit);

            userConstant.optiolwalues = (acef::TypelessVariableStorage *)allocateAndDeserializeData(fp, sizeof(acef::TypelessVariableStorage), userConstant.binStorage.optionsNum);

            userConstant.optionNameByteOffsets = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), userConstant.binStorage.optionsNum);

            userConstant.optionNames = (acef::UILocalizedStringStorage *)acefEffectStorage->allocateMem(userConstant.binStorage.optionsNum * sizeof(acef::UILocalizedStringStorage));
            userConstant.optionNamesBuffers = (acef::UILocalizedStringBuffers *)acefEffectStorage->allocateMem(userConstant.binStorage.optionsNum * sizeof(acef::UILocalizedStringBuffers));
            for (uint32_t optIdx = 0; optIdx < userConstant.binStorage.optionsNum; ++optIdx)
            {
                // TODO avoroshilov ACEF: add byteOffsets for UIControlOptions
                gotoByteOffset(fp, header.binStorage.uiControlsChunkByteOffset + uiControlsHeader.userConstantByteOffsets[idx] + userConstant.optionNameByteOffsets[optIdx]);

                acef::UILocalizedStringStorage & optionNameStorage = userConstant.optionNames[optIdx];
                acef::UILocalizedStringBuffers & optionNameBuffers = userConstant.optionNamesBuffers[optIdx];

                deserializeData(fp, &optionNameStorage.binStorage, sizeof(acef::UILocalizedStringStorage::BinaryStorage), 1);
                deserializeLocalizedStringBuffers(fp, optionNameBuffers, optionNameStorage);
            }

            // Labels for vector elements in a grouped variable
            userConstant.variableNameByteOffsets = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), userConstant.binStorage.dataDimensionality);
            userConstant.variableNames = (acef::UILocalizedStringStorage *)acefEffectStorage->allocateMem(userConstant.binStorage.dataDimensionality * sizeof(acef::UILocalizedStringStorage));
            userConstant.variableNamesBuffers = (acef::UILocalizedStringBuffers *)acefEffectStorage->allocateMem(userConstant.binStorage.dataDimensionality * sizeof(acef::UILocalizedStringBuffers));
            for (uint32_t varIdx = 0; varIdx < userConstant.binStorage.dataDimensionality; ++varIdx)
            {
                gotoByteOffset(fp, header.binStorage.uiControlsChunkByteOffset + uiControlsHeader.userConstantByteOffsets[idx] + userConstant.variableNameByteOffsets[varIdx]);

                acef::UILocalizedStringStorage & variableNameStorage = userConstant.variableNames[varIdx];
                acef::UILocalizedStringBuffers & variableNameBuffers = userConstant.variableNamesBuffers[varIdx];

                deserializeData(fp, &variableNameStorage.binStorage, sizeof(acef::UILocalizedStringStorage::BinaryStorage), 1);
                deserializeLocalizedStringBuffers(fp, variableNameBuffers, variableNameStorage);
            }
        }
    }

    // Passes Chunk
    /////////////////////////////////////////////////////////////////////////////////
    {
        gotoByteOffset(fp, header.binStorage.passesChunkByteOffset);

        acef::PassesHeader & passesHeader = acefEffectStorage->passesHeader;
        deserializeData(fp, &passesHeader.binStorage, sizeof(acef::PassesHeader::BinaryStorage), 1);

        passesHeader.passByteOffsets = (uint64_t *)allocateAndDeserializeData(fp, sizeof(uint64_t), passesHeader.binStorage.passesNum);

        acefEffectStorage->passes.resize(passesHeader.binStorage.passesNum);
        for (uint32_t idx = 0; idx < passesHeader.binStorage.passesNum; ++idx)
        {
            gotoByteOffset(fp, header.binStorage.passesChunkByteOffset + passesHeader.passByteOffsets[idx]);

            acef::Pass & pass = acefEffectStorage->passes[idx];
            deserializeData(fp, &pass.binStorage, sizeof(acef::Pass::BinaryStorage), 1);

            // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
            auto allocateAndDeserializeBufferNames = [&allocateAndDeserializeData](FILE * fp, uint16_t * bufferNameLens, const uint32_t buffersNum) -> char *
            {
                size_t totalCharacters = 0;
                for (uint32_t bufIdx = 0; bufIdx < buffersNum; ++bufIdx)
                {
                    totalCharacters += bufferNameLens[bufIdx];
                }
                return (char *)allocateAndDeserializeData(fp, sizeof(char), totalCharacters);
            };

            // Read Buffers
            pass.readBuffersSlots = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.readBuffersNum);
            pass.readBuffersNameLens = (uint16_t *)allocateAndDeserializeData(fp, sizeof(uint16_t), pass.binStorage.readBuffersNum);
            pass.readBuffersNameOffsets = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.readBuffersNum);
            pass.readBuffersNames = allocateAndDeserializeBufferNames(fp, pass.readBuffersNameLens, pass.binStorage.readBuffersNum);
            pass.readBuffersIndices = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.readBuffersNum);

            // Write Buffers
            pass.writeBuffersSlots = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.writeBuffersNum);
            pass.writeBuffersNameLens = (uint16_t *)allocateAndDeserializeData(fp, sizeof(uint16_t), pass.binStorage.writeBuffersNum);
            pass.writeBuffersNameOffsets = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.writeBuffersNum);
            pass.writeBuffersNames = allocateAndDeserializeBufferNames(fp, pass.writeBuffersNameLens, pass.binStorage.writeBuffersNum);
            pass.writeBuffersIndices = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.writeBuffersNum);

            // Constant Buffers
            //VS
            pass.constantBuffersVSSlots = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.constantBuffersVSNum);
            pass.constantBuffersVSNameLens = (uint16_t *)allocateAndDeserializeData(fp, sizeof(uint16_t), pass.binStorage.constantBuffersVSNum);
            pass.constantBuffersVSNameOffsets = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.constantBuffersVSNum);
            pass.constantBuffersVSNames = allocateAndDeserializeBufferNames(fp, pass.constantBuffersVSNameLens, pass.binStorage.constantBuffersVSNum);
            pass.constantBuffersVSIndices = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.constantBuffersVSNum);

            //PS
            pass.constantBuffersPSSlots = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.constantBuffersPSNum);
            pass.constantBuffersPSNameLens = (uint16_t *)allocateAndDeserializeData(fp, sizeof(uint16_t), pass.binStorage.constantBuffersPSNum);
            pass.constantBuffersPSNameOffsets = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.constantBuffersPSNum);
            pass.constantBuffersPSNames = allocateAndDeserializeBufferNames(fp, pass.constantBuffersPSNameLens, pass.binStorage.constantBuffersPSNum);
            pass.constantBuffersPSIndices = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.constantBuffersPSNum);

            // Samplers
            pass.samplersSlots = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.samplersNum);
            pass.samplersNameLens = (uint16_t *)allocateAndDeserializeData(fp, sizeof(uint16_t), pass.binStorage.samplersNum);
            pass.samplersNameOffsets = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.samplersNum);
            pass.samplersNames = allocateAndDeserializeBufferNames(fp, pass.samplersNameLens, pass.binStorage.samplersNum);
            pass.samplersIndices = (uint32_t *)allocateAndDeserializeData(fp, sizeof(uint32_t), pass.binStorage.samplersNum);
        }
    }

    fclose(fp);
}

static
bool compare(const acef::EffectStorage & effectStorage1, const acef::EffectStorage & effectStorage2)
{
    auto compareData = [](const void * dataRaw1, const void * dataRaw2, size_t elementByteSize, size_t numElements) -> bool
    {
        for (size_t idx = 0, idxEnd = elementByteSize * numElements; idx < idxEnd; ++idx)
        {
            if (reinterpret_cast<const uint8_t *>(dataRaw1)[idx] != reinterpret_cast<const uint8_t *>(dataRaw2)[idx])
            {
                assert(false && "comparison failure");
                return false;
            }
        }
        return true;
    };

    bool result = true;
    result = result && compareData(&effectStorage1.header.binStorage, &effectStorage2.header.binStorage, sizeof(acef::Header::BinaryStorage), 1);

    // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
    auto calcTotalChars = [](uint16_t * bufNameLens, const uint32_t bufNum)
    {
        size_t totalCharacters = 0;
        for (uint32_t bufIdx = 0; bufIdx < bufNum; ++bufIdx)
        {
            totalCharacters += bufNameLens[bufIdx];
        }
        return totalCharacters;
    };

    result = result && compareData(effectStorage1.header.fileTimestamps, effectStorage2.header.fileTimestamps, sizeof(uint64_t), effectStorage2.header.binStorage.dependenciesNum);
    result = result && compareData(effectStorage1.header.filePathLens, effectStorage2.header.filePathLens, sizeof(uint16_t), effectStorage2.header.binStorage.dependenciesNum);
    result = result && compareData(effectStorage1.header.filePathOffsets, effectStorage2.header.filePathOffsets, sizeof(uint32_t), effectStorage2.header.binStorage.dependenciesNum);
    result = result && compareData(effectStorage1.header.filePathsUtf8, effectStorage2.header.filePathsUtf8, sizeof(char), calcTotalChars(effectStorage2.header.filePathLens, effectStorage2.header.binStorage.dependenciesNum));

    // Resources Chunk
    /////////////////////////////////////////////////////////////////////////////////
    {
        const acef::ResourceHeader & resourceHeader1 = effectStorage1.resourceHeader;
        const acef::ResourceHeader & resourceHeader2 = effectStorage2.resourceHeader;
        result = result && compareData(&resourceHeader1.binStorage, &resourceHeader2.binStorage, sizeof(acef::ResourceHeader::BinaryStorage), 1);

        result = result && compareData(resourceHeader1.readBufferTextureHandles, resourceHeader2.readBufferTextureHandles, sizeof(uint32_t), resourceHeader2.binStorage.readBuffersNum);
        result = result && compareData(resourceHeader1.writeBufferTextureHandles, resourceHeader2.writeBufferTextureHandles, sizeof(uint32_t), resourceHeader2.binStorage.writeBuffersNum);

        result = result && compareData(resourceHeader1.pixelShaderByteOffsets, resourceHeader2.pixelShaderByteOffsets, sizeof(uint64_t), resourceHeader2.binStorage.pixelShadersNum);
        result = result && compareData(resourceHeader1.vertexShaderByteOffsets, resourceHeader2.vertexShaderByteOffsets, sizeof(uint64_t), resourceHeader2.binStorage.vertexShadersNum);

        result = result && compareData(resourceHeader1.samplerByteOffsets, resourceHeader2.samplerByteOffsets, sizeof(uint64_t), resourceHeader2.binStorage.samplersNum);
        result = result && compareData(resourceHeader1.constantBufferByteOffsets, resourceHeader2.constantBufferByteOffsets, sizeof(uint64_t), resourceHeader2.binStorage.constantBuffersNum);

        result = result && compareData(resourceHeader1.textureParametrizedByteOffsets, resourceHeader2.textureParametrizedByteOffsets, sizeof(uint64_t), resourceHeader2.binStorage.texturesParametrizedNum);
        result = result && compareData(resourceHeader1.textureIntermediateByteOffsets, resourceHeader2.textureIntermediateByteOffsets, sizeof(uint64_t), resourceHeader2.binStorage.texturesIntermediateNum);
        result = result && compareData(resourceHeader1.textureFromFileByteOffsets, resourceHeader2.textureFromFileByteOffsets, sizeof(uint64_t), resourceHeader2.binStorage.texturesFromFileNum);

        // ResourcePixelShader
        assert(resourceHeader1.binStorage.pixelShadersNum == resourceHeader2.binStorage.pixelShadersNum);
        for (uint32_t idx = 0; idx < resourceHeader2.binStorage.pixelShadersNum; ++idx)
        {
            const acef::ResourcePixelShader & pixelShader1 = effectStorage1.pixelShaders[idx];
            const acef::ResourcePixelShader & pixelShader2 = effectStorage2.pixelShaders[idx];
            result = result && compareData(&pixelShader1.binStorage, &pixelShader2.binStorage, sizeof(acef::ResourcePixelShader::BinaryStorage), 1);

            assert(pixelShader1.binStorage.filePathLen == pixelShader2.binStorage.filePathLen);
            result = result && compareData(pixelShader1.filePathUtf8, pixelShader2.filePathUtf8, sizeof(char), pixelShader2.binStorage.filePathLen);
            assert(pixelShader1.binStorage.entryFunctionLen == pixelShader2.binStorage.entryFunctionLen);
            result = result && compareData(pixelShader1.entryFunctionAscii, pixelShader2.entryFunctionAscii, sizeof(char), pixelShader2.binStorage.entryFunctionLen);
        }

        // ResourceVertexShader
        assert(resourceHeader1.binStorage.vertexShadersNum == resourceHeader2.binStorage.vertexShadersNum);
        for (uint32_t idx = 0; idx < resourceHeader2.binStorage.vertexShadersNum; ++idx)
        {
            const acef::ResourceVertexShader & vertexShader1 = effectStorage1.vertexShaders[idx];
            const acef::ResourceVertexShader & vertexShader2 = effectStorage2.vertexShaders[idx];
            result = result && compareData(&vertexShader1.binStorage, &vertexShader2.binStorage, sizeof(acef::ResourceVertexShader::BinaryStorage), 1);

            assert(vertexShader1.binStorage.filePathLen == vertexShader2.binStorage.filePathLen);
            result = result && compareData(vertexShader1.filePathUtf8, vertexShader2.filePathUtf8, sizeof(char), vertexShader2.binStorage.filePathLen);
            assert(vertexShader1.binStorage.entryFunctionLen == vertexShader2.binStorage.entryFunctionLen);
            result = result && compareData(vertexShader1.entryFunctionAscii, vertexShader2.entryFunctionAscii, sizeof(char), vertexShader2.binStorage.entryFunctionLen);
        }

        // ResourceTextureParametrized
        assert(resourceHeader1.binStorage.texturesParametrizedNum == resourceHeader2.binStorage.texturesParametrizedNum);
        for (uint32_t idx = 0; idx < resourceHeader2.binStorage.texturesParametrizedNum; ++idx)
        {
            const acef::ResourceTextureParametrized & texParametrized1 = effectStorage1.texturesParametrized[idx];
            const acef::ResourceTextureParametrized & texParametrized2 = effectStorage2.texturesParametrized[idx];
            result = result && compareData(&texParametrized1.binStorage, &texParametrized2.binStorage, sizeof(acef::ResourceTextureParametrized::BinaryStorage), 1);
        }

        // ResourceTextureIntermediate
        assert(resourceHeader1.binStorage.texturesIntermediateNum == resourceHeader2.binStorage.texturesIntermediateNum);
        for (uint32_t idx = 0; idx < resourceHeader2.binStorage.texturesIntermediateNum; ++idx)
        {
            const acef::ResourceTextureIntermediate & texIntermediate1 = effectStorage1.texturesIntermediate[idx];
            const acef::ResourceTextureIntermediate & texIntermediate2 = effectStorage2.texturesIntermediate[idx];
            result = result && compareData(&texIntermediate1.binStorage, &texIntermediate2.binStorage, sizeof(acef::ResourceTextureIntermediate::BinaryStorage), 1);
        }

        // ResourceTextureFromFile
        assert(resourceHeader1.binStorage.texturesFromFileNum == resourceHeader2.binStorage.texturesFromFileNum);
        for (uint32_t idx = 0; idx < resourceHeader2.binStorage.texturesFromFileNum; ++idx)
        {
            const acef::ResourceTextureFromFile & texFromFile1 = effectStorage1.texturesFromFile[idx];
            const acef::ResourceTextureFromFile & texFromFile2 = effectStorage2.texturesFromFile[idx];
            result = result && compareData(&texFromFile1.binStorage, &texFromFile2.binStorage, sizeof(acef::ResourceTextureFromFile::BinaryStorage), 1);

            assert(texFromFile1.binStorage.pathLen == texFromFile2.binStorage.pathLen);
            result = result && compareData(texFromFile1.pathUtf8, texFromFile2.pathUtf8, sizeof(char), texFromFile2.binStorage.pathLen);
        }

        // ResourceSampler
        assert(resourceHeader1.binStorage.samplersNum == resourceHeader2.binStorage.samplersNum);
        for (uint32_t idx = 0; idx < resourceHeader2.binStorage.samplersNum; ++idx)
        {
            const acef::ResourceSampler & sampler1 = effectStorage1.samplers[idx];
            const acef::ResourceSampler & sampler2 = effectStorage2.samplers[idx];
            result = result && compareData(&sampler1.binStorage, &sampler2.binStorage, sizeof(acef::ResourceSampler::BinaryStorage), 1);
        }

        // ResourceConstantBuffer
        assert(resourceHeader1.binStorage.constantBuffersNum == resourceHeader2.binStorage.constantBuffersNum);
        for (uint32_t idx = 0; idx < resourceHeader2.binStorage.constantBuffersNum; ++idx)
        {
            const acef::ResourceConstantBuffer & constBuf1 = effectStorage1.constantBuffers[idx];
            const acef::ResourceConstantBuffer & constBuf2 = effectStorage2.constantBuffers[idx];
            result = result && compareData(&constBuf1.binStorage, &constBuf2.binStorage, sizeof(acef::ResourceConstantBuffer::BinaryStorage), 1);

/*
            // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
            auto calcTotalChars = [](uint16_t * bufNameLens, const uint32_t bufNum)
            {
                size_t totalCharacters = 0;
                for (uint32_t bufIdx = 0; bufIdx < bufNum; ++bufIdx)
                {
                    totalCharacters += bufNameLens[bufIdx];
                }
                return totalCharacters;
            };
*/

            assert(constBuf1.binStorage.constantsNum == constBuf2.binStorage.constantsNum);
            result = result && compareData(constBuf1.constantOffsetInComponents, constBuf2.constantOffsetInComponents, sizeof(uint32_t), constBuf2.binStorage.constantsNum);
            result = result && compareData(constBuf1.constantNameLens, constBuf2.constantNameLens, sizeof(uint16_t), constBuf2.binStorage.constantsNum);
            result = result && compareData(constBuf1.constantNameOffsets, constBuf2.constantNameOffsets, sizeof(uint32_t), constBuf2.binStorage.constantsNum);
            result = result && compareData(constBuf1.constantNames, constBuf2.constantNames, sizeof(char), calcTotalChars(constBuf2.constantNameLens, constBuf2.binStorage.constantsNum));
            result = result && compareData(constBuf1.constantHandle, constBuf2.constantHandle, sizeof(uint32_t), constBuf2.binStorage.constantsNum);
        }
    }

    // UI Controls Chunk
    /////////////////////////////////////////////////////////////////////////////////
    {
        auto compareLocalizedStringBuffers = [&compareData](
                const acef::UILocalizedStringBuffers & locStringBuffers1, const acef::UILocalizedStringStorage & locStringStorage1,
                const acef::UILocalizedStringBuffers & locStringBuffers2, const acef::UILocalizedStringStorage & locStringStorage2) -> bool
        {
            bool result = true;
            (void)locStringStorage1;
            assert(locStringStorage1.binStorage.defaultStringLen == locStringStorage2.binStorage.defaultStringLen);
            result = result && compareData(locStringBuffers1.defaultStringAscii, locStringBuffers2.defaultStringAscii, sizeof(char), locStringStorage2.binStorage.defaultStringLen);

            for (uint32_t locIdx = 0; locIdx < locStringStorage2.binStorage.localizationsNum; ++locIdx)
            {
                const acef::UILocalizedStringBuffers::LocalizedString & locStr1 = locStringBuffers1.localizedStrings[locIdx];
                const acef::UILocalizedStringBuffers::LocalizedString & locStr2 = locStringBuffers2.localizedStrings[locIdx];
                result = result && compareData(&locStr1.binStorage, &locStr2.binStorage, sizeof(acef::UILocalizedStringBuffers::LocalizedString::BinaryStorage), 1);

                assert(locStr1.binStorage.strLen == locStr2.binStorage.strLen);
                result = result && compareData(locStr1.stringUtf8, locStr2.stringUtf8, sizeof(char), locStr2.binStorage.strLen);
            }

            return result;
        };

        const acef::UIControlsHeader & uiControlsHeader1 = effectStorage1.uiControlsHeader;
        const acef::UIControlsHeader & uiControlsHeader2 = effectStorage2.uiControlsHeader;
        result = result && compareData(&uiControlsHeader1.binStorage, &uiControlsHeader2.binStorage, sizeof(acef::UIControlsHeader::BinaryStorage), 1);

        result = result && compareData(uiControlsHeader1.userConstantByteOffsets, uiControlsHeader2.userConstantByteOffsets, sizeof(uint64_t), uiControlsHeader2.binStorage.userConstantsNum);

        assert(uiControlsHeader1.binStorage.userConstantsNum == uiControlsHeader2.binStorage.userConstantsNum);
        for (uint32_t idx = 0; idx < uiControlsHeader2.binStorage.userConstantsNum; ++idx)
        {
            const acef::UserConstant & userConstant1 = effectStorage1.userConstants[idx];
            const acef::UserConstant & userConstant2 = effectStorage2.userConstants[idx];
            result = result && compareData(&userConstant1.binStorage, &userConstant2.binStorage, sizeof(acef::UserConstant::BinaryStorage), 1);

            assert(userConstant1.binStorage.controlNameLen == userConstant2.binStorage.controlNameLen);
            result = result && compareData(userConstant1.controlNameAscii, userConstant2.controlNameAscii, sizeof(char), userConstant2.binStorage.controlNameLen);

            result = result && compareLocalizedStringBuffers(userConstant1.labelBuffers, userConstant1.binStorage.label, userConstant2.labelBuffers, userConstant2.binStorage.label);
            result = result && compareLocalizedStringBuffers(userConstant1.hintBuffers, userConstant1.binStorage.hint, userConstant2.hintBuffers, userConstant2.binStorage.hint);
            result = result && compareLocalizedStringBuffers(userConstant1.uiValueUnitBuffers, userConstant1.binStorage.uiValueUnit, userConstant2.uiValueUnitBuffers, userConstant2.binStorage.uiValueUnit);

            assert(userConstant1.binStorage.optionsNum == userConstant2.binStorage.optionsNum);
            result = result && compareData(userConstant1.optiolwalues, userConstant2.optiolwalues, sizeof(acef::TypelessVariableStorage::BinaryStorage), userConstant2.binStorage.optionsNum);
            result = result && compareData(userConstant1.optionNameByteOffsets, userConstant2.optionNameByteOffsets, sizeof(uint64_t), userConstant2.binStorage.optionsNum);
            for (uint32_t optIdx = 0; optIdx < userConstant2.binStorage.optionsNum; ++optIdx)
            {
                const acef::UILocalizedStringStorage & optionNameStorage1 = userConstant1.optionNames[optIdx];
                const acef::UILocalizedStringBuffers & optionNameBuffers1 = userConstant1.optionNamesBuffers[optIdx];
                const acef::UILocalizedStringStorage & optionNameStorage2 = userConstant2.optionNames[optIdx];
                const acef::UILocalizedStringBuffers & optionNameBuffers2 = userConstant2.optionNamesBuffers[optIdx];

                result = result && compareData(&optionNameStorage1.binStorage, &optionNameStorage2.binStorage, sizeof(acef::UILocalizedStringStorage::BinaryStorage), 1);
                result = result && compareLocalizedStringBuffers(optionNameBuffers1, optionNameStorage1, optionNameBuffers2, optionNameStorage2);
            }

            // Labels for vector elements in a grouped variable
            assert(userConstant1.binStorage.dataDimensionality == userConstant2.binStorage.dataDimensionality);
            result = result && compareData(userConstant1.variableNameByteOffsets, userConstant2.variableNameByteOffsets, sizeof(uint64_t), userConstant2.binStorage.dataDimensionality);
            for (uint32_t varIdx = 0; varIdx < userConstant2.binStorage.dataDimensionality; ++varIdx)
            {
                const acef::UILocalizedStringStorage & variableNameStorage1 = userConstant1.variableNames[varIdx];
                const acef::UILocalizedStringBuffers & variableNameBuffers1 = userConstant1.variableNamesBuffers[varIdx];
                const acef::UILocalizedStringStorage & variableNameStorage2 = userConstant2.variableNames[varIdx];
                const acef::UILocalizedStringBuffers & variableNameBuffers2 = userConstant2.variableNamesBuffers[varIdx];

                result = result && compareData(&variableNameStorage1.binStorage, &variableNameStorage2.binStorage, sizeof(acef::UILocalizedStringStorage::BinaryStorage), 1);
                result = result && compareLocalizedStringBuffers(variableNameBuffers1, variableNameStorage1, variableNameBuffers2, variableNameStorage2);
            }
        }
    }

    // Passes Chunk
    /////////////////////////////////////////////////////////////////////////////////
    {
        const acef::PassesHeader & passesHeader1 = effectStorage1.passesHeader;
        const acef::PassesHeader & passesHeader2 = effectStorage2.passesHeader;
        result = result && compareData(&passesHeader1.binStorage, &passesHeader2.binStorage, sizeof(acef::PassesHeader::BinaryStorage), 1);

        assert(passesHeader1.binStorage.passesNum == passesHeader2.binStorage.passesNum);
        result = result && compareData(passesHeader1.passByteOffsets, passesHeader2.passByteOffsets, sizeof(uint64_t), passesHeader2.binStorage.passesNum);

        assert(passesHeader1.binStorage.passesNum == passesHeader2.binStorage.passesNum);
        for (uint32_t idx = 0; idx < passesHeader2.binStorage.passesNum; ++idx)
        {
            const acef::Pass & pass1 = effectStorage1.passes[idx];
            const acef::Pass & pass2 = effectStorage2.passes[idx];
            result = result && compareData(&pass1.binStorage, &pass2.binStorage, sizeof(acef::Pass::BinaryStorage), 1);

/*
            // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
            auto calcTotalChars = [](uint16_t * bufNameLens, const uint32_t bufNum)
            {
                size_t totalCharacters = 0;
                for (uint32_t bufIdx = 0; bufIdx < bufNum; ++bufIdx)
                {
                    totalCharacters += bufNameLens[bufIdx];
                }
                return totalCharacters;
            };
*/

            // Read Buffers
            assert(pass1.binStorage.readBuffersNum == pass2.binStorage.readBuffersNum);
            result = result && compareData(pass1.readBuffersSlots, pass2.readBuffersSlots, sizeof(uint32_t), pass2.binStorage.readBuffersNum);
            result = result && compareData(pass1.readBuffersNameLens, pass2.readBuffersNameLens, sizeof(uint16_t), pass2.binStorage.readBuffersNum);
            result = result && compareData(pass1.readBuffersNameOffsets, pass2.readBuffersNameOffsets, sizeof(uint32_t), pass2.binStorage.readBuffersNum);
            result = result && compareData(pass1.readBuffersNames, pass2.readBuffersNames, sizeof(char), calcTotalChars(pass2.readBuffersNameLens, pass2.binStorage.readBuffersNum));
            result = result && compareData(pass1.readBuffersIndices, pass2.readBuffersIndices, sizeof(uint32_t), pass2.binStorage.readBuffersNum);

            // Write Buffers
            assert(pass1.binStorage.writeBuffersNum == pass2.binStorage.writeBuffersNum);
            result = result && compareData(pass1.writeBuffersSlots, pass2.writeBuffersSlots, sizeof(uint32_t), pass2.binStorage.writeBuffersNum);
            result = result && compareData(pass1.writeBuffersNameLens, pass2.writeBuffersNameLens, sizeof(uint16_t), pass2.binStorage.writeBuffersNum);
            result = result && compareData(pass1.writeBuffersNameOffsets, pass2.writeBuffersNameOffsets, sizeof(uint32_t), pass2.binStorage.writeBuffersNum);
            result = result && compareData(pass1.writeBuffersNames, pass2.writeBuffersNames, sizeof(char), calcTotalChars(pass2.writeBuffersNameLens, pass2.binStorage.writeBuffersNum));
            result = result && compareData(pass1.writeBuffersIndices, pass2.writeBuffersIndices, sizeof(uint32_t), pass2.binStorage.writeBuffersNum);

            // Constant Buffers
            //VS
            assert(pass1.binStorage.constantBuffersVSNum == pass2.binStorage.constantBuffersVSNum);
            result = result && compareData(pass1.constantBuffersVSSlots, pass2.constantBuffersVSSlots, sizeof(uint32_t), pass2.binStorage.constantBuffersVSNum);
            result = result && compareData(pass1.constantBuffersVSNameLens, pass2.constantBuffersVSNameLens, sizeof(uint16_t), pass2.binStorage.constantBuffersVSNum);
            result = result && compareData(pass1.constantBuffersVSNameOffsets, pass2.constantBuffersVSNameOffsets, sizeof(uint32_t), pass2.binStorage.constantBuffersVSNum);
            result = result && compareData(pass1.constantBuffersVSNames, pass2.constantBuffersVSNames, sizeof(char), calcTotalChars(pass2.constantBuffersVSNameLens, pass2.binStorage.constantBuffersVSNum));
            result = result && compareData(pass1.constantBuffersVSIndices, pass2.constantBuffersVSIndices, sizeof(uint32_t), pass2.binStorage.constantBuffersVSNum);
            //PS
            assert(pass1.binStorage.constantBuffersPSNum == pass2.binStorage.constantBuffersPSNum);
            result = result && compareData(pass1.constantBuffersPSSlots, pass2.constantBuffersPSSlots, sizeof(uint32_t), pass2.binStorage.constantBuffersPSNum);
            result = result && compareData(pass1.constantBuffersPSNameLens, pass2.constantBuffersPSNameLens, sizeof(uint16_t), pass2.binStorage.constantBuffersPSNum);
            result = result && compareData(pass1.constantBuffersPSNameOffsets, pass2.constantBuffersPSNameOffsets, sizeof(uint32_t), pass2.binStorage.constantBuffersPSNum);
            result = result && compareData(pass1.constantBuffersPSNames, pass2.constantBuffersPSNames, sizeof(char), calcTotalChars(pass2.constantBuffersPSNameLens, pass2.binStorage.constantBuffersPSNum));
            result = result && compareData(pass1.constantBuffersPSIndices, pass2.constantBuffersPSIndices, sizeof(uint32_t), pass2.binStorage.constantBuffersPSNum);

            // Samplers
            assert(pass1.binStorage.samplersNum == pass2.binStorage.samplersNum);
            result = result && compareData(pass1.samplersSlots, pass2.samplersSlots, sizeof(uint32_t), pass2.binStorage.samplersNum);
            result = result && compareData(pass1.samplersNameLens, pass2.samplersNameLens, sizeof(uint16_t), pass2.binStorage.samplersNum);
            result = result && compareData(pass1.samplersNameOffsets, pass2.samplersNameOffsets, sizeof(uint32_t), pass2.binStorage.samplersNum);
            result = result && compareData(pass1.samplersNames, pass2.samplersNames, sizeof(char), calcTotalChars(pass2.samplersNameLens, pass2.binStorage.samplersNum));
            result = result && compareData(pass1.samplersIndices, pass2.samplersIndices, sizeof(uint32_t), pass2.binStorage.samplersNum);
        }
    }

    return result;
}

static
void calcByteOffsets(acef::EffectStorage * acefEffectStorage)
{
    auto colwertIntVectorToUint64Raw = [&acefEffectStorage](const std::vector<int> & vec) -> uint64_t *
    {
        if (vec.size() == 0)
            return nullptr;

        uint64_t * rawMem = (uint64_t *)acefEffectStorage->allocateMem(vec.size() * sizeof(uint64_t));

        for (int i = 0, iend = (int)vec.size(); i < iend; ++i)
        {
            *(rawMem+i) = vec[i];
        }
        return rawMem;
    };

    int totalOffset = 0;

    // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
    auto calcTotalChars = [](uint16_t * bufNameLens, const uint32_t bufNum)
    {
        size_t totalCharacters = 0;
        for (uint32_t bufIdx = 0; bufIdx < bufNum; ++bufIdx)
        {
            totalCharacters += bufNameLens[bufIdx];
        }
        return totalCharacters;
    };

    size_t headerBuffersByteSize =
        acefEffectStorage->header.binStorage.dependenciesNum * sizeof(uint64_t) +  // fileTimestamps
        acefEffectStorage->header.binStorage.dependenciesNum * sizeof(uint16_t) +  // filePathLens
        acefEffectStorage->header.binStorage.dependenciesNum * sizeof(uint32_t) +  // filePathOffsets
        calcTotalChars(acefEffectStorage->header.filePathLens, acefEffectStorage->header.binStorage.dependenciesNum) * sizeof(char);  // filePaths
    size_t totalHeaderByteSize = sizeof(acefEffectStorage->header.binStorage) + headerBuffersByteSize;

    totalOffset += (int)totalHeaderByteSize;

#define DBG_INTRACHUNK_OFFSET_NOHEADER  0

    // Resource Chunk
    /////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////

    acefEffectStorage->header.binStorage.resourcesChunkByteOffset = totalOffset;

#if (DBG_INTRACHUNK_OFFSET_NOHEADER == 1)
    // Offset within the resource chunk is relative to the resource chunk header end
    totalOffset += (int)acef::ResourceHeader::BinaryStorage::storageByteSize();
    int totalOffsetInResourceChunk = 0;
#else
    // Offset within the resource chunk is relative to the start ot resource chunk and not to the resource chunk header end
    int totalOffsetInResourceChunk = (int)acef::ResourceHeader::BinaryStorage::storageByteSize();
#endif

    // Metadata in Resources Chunk Header
    totalOffsetInResourceChunk += (int)(acefEffectStorage->resourceHeader.binStorage.readBuffersNum * sizeof(uint32_t));
    totalOffsetInResourceChunk += (int)(acefEffectStorage->resourceHeader.binStorage.writeBuffersNum * sizeof(uint32_t));

    totalOffsetInResourceChunk += (int)(acefEffectStorage->resourceHeader.binStorage.pixelShadersNum * sizeof(uint64_t));
    totalOffsetInResourceChunk += (int)(acefEffectStorage->resourceHeader.binStorage.vertexShadersNum * sizeof(uint64_t));

    totalOffsetInResourceChunk += (int)(acefEffectStorage->resourceHeader.binStorage.samplersNum * sizeof(uint64_t));
    totalOffsetInResourceChunk += (int)(acefEffectStorage->resourceHeader.binStorage.constantBuffersNum * sizeof(uint64_t));

    totalOffsetInResourceChunk += (int)(acefEffectStorage->resourceHeader.binStorage.texturesParametrizedNum * sizeof(uint64_t));
    totalOffsetInResourceChunk += (int)(acefEffectStorage->resourceHeader.binStorage.texturesIntermediateNum * sizeof(uint64_t));
    totalOffsetInResourceChunk += (int)(acefEffectStorage->resourceHeader.binStorage.texturesFromFileNum * sizeof(uint64_t));

    // Pixel shaders
    //////////////////////////////////////////////////////////////
    {
        std::vector<int> resourceChunkPSOffsets;

        // Go through all resources, callwlate offsets and sizes
        for (int i = 0, iend = (int)acefEffectStorage->resourceHeader.binStorage.pixelShadersNum; i < iend; ++i)
        {
            const acef::ResourcePixelShader & psResource = acefEffectStorage->pixelShaders[i];
            resourceChunkPSOffsets.push_back(totalOffsetInResourceChunk);

            // Size/offset
            int filenameLen = (int)psResource.binStorage.filePathLen;
            int entryFunctionLen = (int)psResource.binStorage.entryFunctionLen;

            int lwrPSBuffersByteSize = filenameLen * sizeof(char) + entryFunctionLen * sizeof(char);
            int totalPSByteSize = (int)acef::ResourcePixelShader::BinaryStorage::storageByteSize() + lwrPSBuffersByteSize;

            totalOffsetInResourceChunk += totalPSByteSize;
        }

        acefEffectStorage->resourceHeader.pixelShaderByteOffsets = colwertIntVectorToUint64Raw(resourceChunkPSOffsets);
    }


    // Vertex shaders
    //////////////////////////////////////////////////////////////
    {
        std::vector<int> resourceChunkVSOffsets;

        // Go through all resources, callwlate offsets and sizes
        for (int i = 0, iend = (int)acefEffectStorage->resourceHeader.binStorage.vertexShadersNum; i < iend; ++i)
        {
            const acef::ResourceVertexShader & vsResource = acefEffectStorage->vertexShaders[i];
            resourceChunkVSOffsets.push_back(totalOffsetInResourceChunk);

            // Size/offset
            int filenameLen = (int)vsResource.binStorage.filePathLen;
            int entryFunctionLen = (int)vsResource.binStorage.entryFunctionLen;

            int lwrBuffersByteSize = filenameLen * sizeof(char) + entryFunctionLen * sizeof(char);
            int totalByteSize = (int)acef::ResourcePixelShader::BinaryStorage::storageByteSize() + lwrBuffersByteSize;

            totalOffsetInResourceChunk += totalByteSize;
        }

        acefEffectStorage->resourceHeader.vertexShaderByteOffsets = colwertIntVectorToUint64Raw(resourceChunkVSOffsets);
    }


    // Textures
    //////////////////////////////////////////////////////////////

    // Parametrized
    {
        std::vector<int> resourceChunkTexParametrizedOffsets;

        for (int i = 0, iend = (int)acefEffectStorage->resourceHeader.binStorage.texturesParametrizedNum; i < iend; ++i)
        {
            resourceChunkTexParametrizedOffsets.push_back(totalOffsetInResourceChunk);

            // Size/offset
            size_t totalByteSize = acef::ResourceTextureParametrized::BinaryStorage::storageByteSize();
            totalOffsetInResourceChunk += (int)totalByteSize;
        }

        acefEffectStorage->resourceHeader.textureParametrizedByteOffsets = colwertIntVectorToUint64Raw(resourceChunkTexParametrizedOffsets);
    }

    // Intermediate
    {
        std::vector<int> resourceChunkTexIntermediateOffsets;

        for (int i = 0, iend = (int)acefEffectStorage->resourceHeader.binStorage.texturesIntermediateNum; i < iend; ++i)
        {
            resourceChunkTexIntermediateOffsets.push_back(totalOffsetInResourceChunk);

            // Size/offset
            size_t totalByteSize = acef::ResourceTextureIntermediate::BinaryStorage::storageByteSize();
            totalOffsetInResourceChunk += (int)totalByteSize;
        }

        acefEffectStorage->resourceHeader.textureIntermediateByteOffsets = colwertIntVectorToUint64Raw(resourceChunkTexIntermediateOffsets);
    }

    // FromFile
    {
        std::vector<int> resourceChunkTexFromFileOffsets;

        for (int i = 0, iend = (int)acefEffectStorage->resourceHeader.binStorage.texturesFromFileNum; i < iend; ++i)
        {
            const acef::ResourceTextureFromFile & texResource = acefEffectStorage->texturesFromFile[i];

            resourceChunkTexFromFileOffsets.push_back(totalOffsetInResourceChunk);

            // Size/offset
            uint32_t pathLen = (int)texResource.binStorage.pathLen;
            size_t buffersByteSize = pathLen * sizeof(char);
            size_t totalByteSize = acef::ResourceTextureFromFile::BinaryStorage::storageByteSize() + buffersByteSize;

            totalOffsetInResourceChunk += (int)totalByteSize;
        }

        acefEffectStorage->resourceHeader.textureFromFileByteOffsets = colwertIntVectorToUint64Raw(resourceChunkTexFromFileOffsets);
    }

    // Samplers
    //////////////////////////////////////////////////////////////
    {
        std::vector<int> resourceChunkSamplerOffsets;

        for (int i = 0, iend = (int)acefEffectStorage->resourceHeader.binStorage.samplersNum; i < iend; ++i)
        {
            resourceChunkSamplerOffsets.push_back(totalOffsetInResourceChunk);

            // Size/offset
            size_t totalByteSize = acef::ResourceSampler::BinaryStorage::storageByteSize();

            totalOffsetInResourceChunk += (int)totalByteSize;
        }

        acefEffectStorage->resourceHeader.samplerByteOffsets = colwertIntVectorToUint64Raw(resourceChunkSamplerOffsets);
    }

    // Constant buffers
    //////////////////////////////////////////////////////////////
    {
        std::vector<int> resourceChunkConstBufOffsets;

        for (int i = 0, iend = (int)acefEffectStorage->resourceHeader.binStorage.constantBuffersNum; i < iend; ++i)
        {
            const acef::ResourceConstantBuffer & constBufResource = acefEffectStorage->constantBuffers[i];

            resourceChunkConstBufOffsets.push_back(totalOffsetInResourceChunk);

/*
            // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
            auto calcTotalChars = [](uint16_t * bufNameLens, const uint32_t bufNum)
            {
                size_t totalCharacters = 0;
                for (uint32_t bufIdx = 0; bufIdx < bufNum; ++bufIdx)
                {
                    totalCharacters += bufNameLens[bufIdx];
                }
                return totalCharacters;
            };
*/

            // Size/offset
            size_t buffersByteSize =
                    constBufResource.binStorage.constantsNum * sizeof(uint32_t) +  // OffsetInComponents
                    constBufResource.binStorage.constantsNum * sizeof(uint16_t) +  // NameLens
                    constBufResource.binStorage.constantsNum * sizeof(uint32_t) +  // NameOffsets
                    calcTotalChars(constBufResource.constantNameLens, constBufResource.binStorage.constantsNum) * sizeof(char) +  // Names
                    constBufResource.binStorage.constantsNum * sizeof(uint32_t);  // Handle
            size_t totalByteSize = acef::ResourceConstantBuffer::BinaryStorage::storageByteSize() + buffersByteSize;

            totalOffsetInResourceChunk += (int)totalByteSize;
        }

        acefEffectStorage->resourceHeader.constantBufferByteOffsets = colwertIntVectorToUint64Raw(resourceChunkConstBufOffsets);

    }

    totalOffset += totalOffsetInResourceChunk;

    // UI Controls Chunk
    /////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////

    acefEffectStorage->header.binStorage.uiControlsChunkByteOffset = totalOffset;

#if (DBG_INTRACHUNK_OFFSET_NOHEADER == 1)
    totalOffset += (int)acef::UIControlsHeader::BinaryStorage::storageByteSize();
    int totalOffsetInUIControlsChunk = 0;
#else
    int totalOffsetInUIControlsChunk = (int)acef::UIControlsHeader::BinaryStorage::storageByteSize();
#endif

    // Metadata in UI Controls Chunk Header
    totalOffsetInUIControlsChunk += (int)(acefEffectStorage->uiControlsHeader.binStorage.userConstantsNum * sizeof(uint64_t));

    {
        std::vector<int> uiControlsChunkUserConstantOffsets;

        std::vector<int> uiControlsChunkUserConstantStringOffsets;

        for (int i = 0, iend = (int)acefEffectStorage->uiControlsHeader.binStorage.userConstantsNum; i < iend; ++i)
        {
            const acef::UserConstant & userConst = acefEffectStorage->userConstants[i];

            uiControlsChunkUserConstantOffsets.push_back(totalOffsetInUIControlsChunk);

            // Size/offset
            size_t buffersByteSize = acef::UserConstant::BinaryStorage::storageByteSize();

            // controlNameAscii
            buffersByteSize += userConst.binStorage.controlNameLen * sizeof(char);

            auto byteOffsetStringWithLocalization = [](const acef::UILocalizedStringStorage & locStringStorage, const acef::UILocalizedStringBuffers & locStringBuffers)
            {
                size_t buffersByteSize = 0;

                buffersByteSize += locStringStorage.binStorage.defaultStringLen * sizeof(char);  // default string buf

                for (int locIdx = 0, locNum = locStringStorage.binStorage.localizationsNum; locIdx < locNum; ++locIdx)
                {
                    buffersByteSize += acef::UILocalizedStringBuffers::LocalizedString::BinaryStorage::storageByteSize();
                    buffersByteSize += locStringBuffers.localizedStrings[locIdx].binStorage.strLen * sizeof(char);
                }

                return buffersByteSize;
            };

            // label
            buffersByteSize += byteOffsetStringWithLocalization(userConst.binStorage.label, userConst.labelBuffers);

            // hint
            buffersByteSize += byteOffsetStringWithLocalization(userConst.binStorage.hint, userConst.hintBuffers);

            // uiValueUnit
            buffersByteSize += byteOffsetStringWithLocalization(userConst.binStorage.uiValueUnit, userConst.uiValueUnitBuffers);

            int numOptions = (int)userConst.binStorage.optionsNum;
            buffersByteSize += numOptions * acef::TypelessVariableStorage::BinaryStorage::storageByteSize();

            // optionNameByteOffsets
            buffersByteSize += numOptions * sizeof(uint64_t);

            uiControlsChunkUserConstantStringOffsets.resize(0);
            for (int optIdx = 0, optIdxEnd = numOptions; optIdx < optIdxEnd; ++optIdx)
            {
                uiControlsChunkUserConstantStringOffsets.push_back((int)(buffersByteSize));
                buffersByteSize += acef::UILocalizedStringStorage::BinaryStorage::storageByteSize();
                buffersByteSize += byteOffsetStringWithLocalization(userConst.optionNames[optIdx], userConst.optionNamesBuffers[optIdx]);
            }
            acefEffectStorage->userConstants[i].optionNameByteOffsets = colwertIntVectorToUint64Raw(uiControlsChunkUserConstantStringOffsets);

            // variableNameByteOffsets
            int dataDimensionality = (int)userConst.binStorage.dataDimensionality;
            buffersByteSize += dataDimensionality * sizeof(uint64_t);

            uiControlsChunkUserConstantStringOffsets.resize(0);
            for (int dimIdx = 0; dimIdx < dataDimensionality; ++dimIdx)
            {
                uiControlsChunkUserConstantStringOffsets.push_back((int)(buffersByteSize));
                buffersByteSize += acef::UILocalizedStringStorage::BinaryStorage::storageByteSize();
                buffersByteSize += byteOffsetStringWithLocalization(userConst.variableNames[dimIdx], userConst.variableNamesBuffers[dimIdx]);
            }
            acefEffectStorage->userConstants[i].variableNameByteOffsets = colwertIntVectorToUint64Raw(uiControlsChunkUserConstantStringOffsets);

            totalOffsetInUIControlsChunk += (int)buffersByteSize;
        }

        acefEffectStorage->uiControlsHeader.userConstantByteOffsets = colwertIntVectorToUint64Raw(uiControlsChunkUserConstantOffsets);
    }

    totalOffset += totalOffsetInUIControlsChunk;

    // Passes Chunk
    /////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////

    acefEffectStorage->header.binStorage.passesChunkByteOffset = totalOffset;

#if (DBG_INTRACHUNK_OFFSET_NOHEADER == 1)
    totalOffset += (int)acef::PassesHeader::BinaryStorage::storageByteSize();
    int totalOffsetInPassesChunk = 0;
#else
    int totalOffsetInPassesChunk = (int)acef::PassesHeader::BinaryStorage::storageByteSize();
#endif

    // Metadata in Passes Chunk Header
    totalOffsetInPassesChunk += (int)(acefEffectStorage->passesHeader.binStorage.passesNum * sizeof(uint64_t));

    {
        std::vector<int> passesChunkPassByteOffsets;

        for (int i = 0, iend = (int)acefEffectStorage->passesHeader.binStorage.passesNum; i < iend; ++i)
        {
            const acef::Pass & pass = acefEffectStorage->passes[i];

            passesChunkPassByteOffsets.push_back(totalOffsetInPassesChunk);

            // Size/offset
            uint32_t readBuffersNum = pass.binStorage.readBuffersNum;
            uint32_t writeBuffersNum = pass.binStorage.writeBuffersNum;
            uint32_t constantBuffersVSNum = pass.binStorage.constantBuffersVSNum;
            uint32_t constantBuffersPSNum = pass.binStorage.constantBuffersPSNum;
            uint32_t samplersNum = pass.binStorage.samplersNum;

            size_t buffersByteSize = 0;

/*
            // TODO avoroshilov ACEF: calcTotalChars is deprecated, could actually be replaced with the last buf nameOffset + nameLen
            auto calcTotalChars = [](uint16_t * bufNameLens, const uint32_t bufNum)
            {
                size_t totalCharacters = 0;
                for (uint32_t bufIdx = 0; bufIdx < bufNum; ++bufIdx)
                {
                    totalCharacters += bufNameLens[bufIdx];
                }
                return totalCharacters;
            };
*/
            // Read buffers slots, nameLens, names, indices
            buffersByteSize +=
                    readBuffersNum * sizeof(uint32_t) +  // slots
                    readBuffersNum * sizeof(uint16_t) +  // nameLens
                    readBuffersNum * sizeof(uint32_t) +  // nameOffsets
                    calcTotalChars(pass.readBuffersNameLens, readBuffersNum) * sizeof(char) +  // names
                    readBuffersNum * sizeof(uint32_t);  // indices
            // Write buffers slots, nameLens, names, indices
            buffersByteSize +=
                    writeBuffersNum * sizeof(uint32_t) +  // slots
                    writeBuffersNum * sizeof(uint16_t) +  // nameLens
                    writeBuffersNum * sizeof(uint32_t) +  // nameOffsets
                    calcTotalChars(pass.writeBuffersNameLens, writeBuffersNum) * sizeof(char) +  // names
                    writeBuffersNum * sizeof(uint32_t);  // indices
            // Constant buffers slots, nameLens, names, indices
            //VS
            buffersByteSize +=
                    constantBuffersVSNum * sizeof(uint32_t) +  // slots
                    constantBuffersVSNum * sizeof(uint16_t) +  // nameLens
                    constantBuffersVSNum * sizeof(uint32_t) +  // nameOffsets
                    calcTotalChars(pass.constantBuffersVSNameLens, constantBuffersVSNum) * sizeof(char) +  // names
                    constantBuffersVSNum * sizeof(uint32_t);  // indices
            //PS
            buffersByteSize +=
                    constantBuffersPSNum * sizeof(uint32_t) +  // slots
                    constantBuffersPSNum * sizeof(uint16_t) +  // nameLens
                    constantBuffersPSNum * sizeof(uint32_t) +  // nameOffsets
                    calcTotalChars(pass.constantBuffersPSNameLens, constantBuffersPSNum) * sizeof(char) +  // names
                    constantBuffersPSNum * sizeof(uint32_t);  // indices
            // Samplers slots, nameLens, names, indices
            buffersByteSize +=
                    samplersNum * sizeof(uint32_t) +  // slots
                    samplersNum * sizeof(uint16_t) +  // nameLens
                    samplersNum * sizeof(uint32_t) +  // nameOffsets
                    calcTotalChars(pass.samplersNameLens, samplersNum) * sizeof(char) +  // names
                    samplersNum * sizeof(uint32_t);  // indices

            size_t totalByteSize = acef::Pass::BinaryStorage::storageByteSize() + buffersByteSize;

            totalOffsetInPassesChunk += (int)totalByteSize;
        }

        acefEffectStorage->passesHeader.passByteOffsets = colwertIntVectorToUint64Raw(passesChunkPassByteOffsets);
    }

    totalOffset += totalOffsetInPassesChunk;

    acefEffectStorage->totalByteSize = totalOffset;
}

}
